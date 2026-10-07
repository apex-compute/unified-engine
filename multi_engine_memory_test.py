"""Compare engine DRAM reads/writes in private windows and a split region.

Run directly to avoid the full model suite, for example:
    python multi_engine_memory_test.py --dev xdma1 --json kintex7_memory_results.json
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import socket
import statistics
import time

import torch

import multi_engine_shard as shard
import user_dma_core as core
from andromeda_hw_info import decode_hardware_info, format_hardware_info


def benchmark_windows(num_engines):
    """Use every physical engine, including the older U55C's shared MC ports."""
    if (core.ANDROMEDA_CORE_COUNT == shard.ALVEO_U55C_BOARD_CORES
            and core.AVAILABLE_DRAM_SIZE_GB == 8
            and num_engines > shard.ALVEO_U55C_MC_COUNT):
        return [shard.EngineWindow(i, ((base, shard.ALVEO_U55C_SEGMENT_STRIDE),))
                for i, base in enumerate(shard.alveo_u55c_core_bases(num_engines, dram_size_gb=8))]
    return shard.board_private_windows(num_engines)


@dataclass(frozen=True)
class MemoryRegion:
    engine: int
    source: int
    destination: int


def memory_regions(windows, engine_ids, transfer_bytes, layout, private_spacing_bytes=None):
    """Keep private ISA placement fixed; vary only nonoverlapping data ranges."""
    if layout not in ("private", "split"):
        raise ValueError(f"unknown layout: {layout}")
    if not engine_ids or len(set(engine_ids)) != len(engine_ids):
        raise ValueError("engine_ids must be nonempty and unique")
    if any(i < 0 or i >= len(windows) for i in engine_ids):
        raise ValueError("engine index outside board windows")
    if transfer_bytes <= 0 or transfer_bytes % (core.UE_VECTOR_SIZE * 2):
        raise ValueError("transfer size must be a positive multiple of 128 bytes")
    if transfer_bytes > core.URAM_FULL_ELEMENTS * 2:
        raise ValueError("transfer size exceeds one engine's URAM A")
    if private_spacing_bytes is not None:
        if (private_spacing_bytes < 2 * transfer_bytes
                or private_spacing_bytes % (core.UE_VECTOR_SIZE * 2)):
            raise ValueError("private spacing must be 128-byte aligned and fit source plus destination")
        if core.AVAILABLE_DRAM_SIZE_GB is None:
            raise ValueError("custom private spacing requires hardware DRAM capacity")
    count = len(engine_ids)
    regions = []
    for slot, engine in enumerate(engine_ids):
        if layout == "private" and private_spacing_bytes is not None:
            source = windows[0].base + shard.ENGINE_TENSOR_OFFSET + engine * private_spacing_bytes
            destination = source + transfer_bytes
            end = destination + transfer_bytes
            dram_start = core.DRAM_START_ADDR if core.AVAILABLE_DRAM_SIZE_GB < 4 else 0
            dram_end = dram_start + core.AVAILABLE_DRAM_SIZE_GB * (1 << 30)
            if source < dram_start or end > dram_end:
                raise ValueError("custom private region exceeds hardware DRAM capacity")
            for window in windows:
                program_start = window.base + shard.ENGINE_PROGRAM_OFFSET
                program_end = window.base + shard.ENGINE_FOOTPRINT_BYTES
                if source < program_end and program_start < end:
                    raise ValueError("custom private region overlaps ISA storage")
            regions.append(MemoryRegion(engine, source, destination))
            continue
        owner = windows[engine] if layout == "private" else windows[engine_ids[0]]
        base = owner.base + shard.ENGINE_TENSOR_OFFSET
        source = base if layout == "private" else base + slot * transfer_bytes
        destination = (base + transfer_bytes if layout == "private"
                       else base + (count + slot) * transfer_bytes)
        limit = owner.base + min(owner.primary_bytes, shard.ENGINE_PROGRAM_OFFSET)
        if destination + transfer_bytes > limit:
            raise ValueError("data region would overlap ISA storage or leave its window")
        regions.append(MemoryRegion(engine, source, destination))
    return regions


def _upload_program(ue, emit):
    ue.clear_capture_buffer()
    ue.start_capture()
    emit(ue)
    ue.generate_instruction_halt()
    ue.stop_capture()
    address = ue.get_program_dram_addr()
    written = ue.write_captured_instructions_to_dram(address)
    expected = ue.get_capture_instruction_size_bytes()
    if written != expected:
        raise IOError(f"program upload at 0x{address:x}: {written}/{expected} bytes")
    ue.allocate_program_dram(expected)
    return address


def _start(ue, address):
    # A fresh HALT is required: an idle queue alone could mean a missed launch.
    ue.write_reg32(core.UE_INT_REG, 0)
    ue.start_execute_from_dram(address)


def _wait_for_halt(ue, timeout=10.0):
    deadline = time.monotonic() + timeout
    while True:
        if (ue.read_reg32(core.UE_INT_REG) & 3) == core.INT_CAUSE_HALT and not ue.is_queue_busy():
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"engine at 0x{ue._base_addr:x} did not halt")
        time.sleep(0.001)


def _clear_flags(ues, clear_programs):
    # Untimed, host-separated clear works on images without CHECK_CLEAR too.
    # Wait for EVERY queue before reusing any flag from the previous phase.
    for ue, address in zip(ues, clear_programs):
        _start(ue, address)
    for ue in ues:
        _wait_for_halt(ue)


def _transfer_program(ue, region, engine_ids, direction, transfer_bytes, iterations):
    master = region.engine == engine_ids[0]
    if len(engine_ids) > 1:
        if master:
            ue.generate_instruction_flag_set()
        else:
            ue.generate_instruction_flag_check_set(engine_ids[0])
    ue.loop_start(iterations)
    if direction == "read":
        ue.accelerator_memory_to_sram(region.source, 0, transfer_bytes // 2)
    else:
        ue.sram_to_accelerator_memory(0, region.destination, transfer_bytes // 2)
    ue.loop_end()
    if len(engine_ids) > 1:
        if master:
            for engine in engine_ids[1:]:
                ue.generate_instruction_flag_check_set(engine)
            ue.generate_instruction_flag_clear()
        else:
            ue.generate_instruction_flag_set()


def _measure(ues, programs, clear_programs):
    _clear_flags(ues, clear_programs)
    for index in list(range(1, len(ues))) + [0]:
        _start(ues[index], programs[index])
    for ue in ues:
        _wait_for_halt(ue)
    # Workers include host launch skew while waiting for the master. Only the
    # master's timer spans release -> transfers -> all workers completed.
    latency_us = ues[0].report_latency_in_us()
    if not math.isfinite(latency_us) or latency_us <= 0:
        raise RuntimeError(f"invalid hardware latency: {latency_us}")
    return latency_us


def _write_data(ue, address, data):
    expected = data.numel() * data.element_size()
    written = ue.dma_write(ue.h2c_device, address, data, expected)
    if written != expected:
        raise IOError(f"data upload at 0x{address:x}: {written}/{expected} bytes")


def _run_case(windows, engine_ids, layout, size_kib, iterations, samples,
              private_spacing_bytes=None):
    transfer_bytes = size_kib * 1024
    regions = memory_regions(windows, engine_ids, transfer_bytes, layout, private_spacing_bytes)
    ues = [core.UnifiedEngine(
        BASE_ADDR=core.UE_0_BASE_ADDR + region.engine * 0x10000,
        params_dram_base=windows[region.engine].base,
        tensor_dram_base=windows[region.engine].base + shard.ENGINE_TENSOR_OFFSET,
        program_dram_base=windows[region.engine].base + shard.ENGINE_PROGRAM_OFFSET,
    ) for region in regions]
    for ue in ues:
        if ue.is_queue_busy():
            raise RuntimeError(f"engine at 0x{ue._base_addr:x} is already busy")
    clear_programs = [_upload_program(ue, lambda u: u.generate_instruction_flag_clear())
                      for ue in ues]
    programs = {}
    for direction in ("read", "write"):
        programs[direction] = [
            _upload_program(ue, lambda u, r=region, d=direction: _transfer_program(
                u, r, engine_ids, d, transfer_bytes, iterations))
            for ue, region in zip(ues, regions)
        ]
    timings = {"read": [], "write": []}
    generator = torch.Generator().manual_seed(0)
    poison = torch.zeros(transfer_bytes // 2, dtype=torch.bfloat16)
    for sample in range(samples + 1):  # First complete, verified round is warmup.
        patterns = []
        for ue, region in zip(ues, regions):
            # Distinct finite, nonzero BF16 bit patterns, changed every sample.
            data = torch.randint(1, 0x7F80, (transfer_bytes // 2,),
                                 generator=generator, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
            patterns.append(data)
            _write_data(ue, region.source, data)
            _write_data(ue, region.destination, poison)
        for direction in ("read", "write"):
            latency = _measure(ues, programs[direction], clear_programs)
            if sample:
                timings[direction].append(latency)
        for ue, region, expected in zip(ues, regions, patterns):
            actual = torch.empty_like(expected)
            received = ue.dma_read(ue.c2h_device, region.destination, actual, transfer_bytes)
            if received != transfer_bytes:
                raise IOError(f"readback at 0x{region.destination:x}: {received}/{transfer_bytes} bytes")
            if not torch.equal(expected.view(torch.int16), actual.view(torch.int16)):
                mismatches = torch.count_nonzero(expected.view(torch.int16) != actual.view(torch.int16)).item()
                raise AssertionError(f"{layout} engine {region.engine}, sample {sample}: "
                                     f"{mismatches} mismatched words")
    _clear_flags(ues, clear_programs)
    rows = []
    for direction in ("read", "write"):
        latency = statistics.median(timings[direction])
        total_bytes = len(ues) * transfer_bytes * iterations
        rows.append(dict(layout=layout, engines=list(engine_ids), direction=direction,
                         private_spacing_bytes=private_spacing_bytes if layout == "private" else None,
                         source_spacing_bytes=regions[1].source - regions[0].source if len(regions) > 1 else None,
                         size_kib=size_kib, iterations=iterations, samples=samples,
                         regions=[asdict(r) for r in regions], latency_us=latency,
                         latency_samples_us=timings[direction], total_bytes=total_bytes,
                         mb_per_s=total_bytes / latency, verified=True))
    return rows


def run_memory_comparison(num_engines=None, sizes_kib=(64, 256, 512), iterations=32, samples=5,
                          private_spacing_bytes=None):
    """Single-engine baselines followed by equal-work private/split comparisons."""
    if core.HW_INFO_RAW is None:
        core.configure_clock_from_hardware()
    if not core.QUEUE_MODE_ENABLED:
        raise RuntimeError("memory comparison requires the queued instruction engine")
    count = core.ANDROMEDA_CORE_COUNT if num_engines is None else num_engines
    # Keep this a bounded microbenchmark, well below the HALT deadline and
    # 32-bit hardware timer rollover on the supported boards.
    if not 1 <= iterations <= 1024 or samples < 1 or not sizes_kib:
        raise ValueError("iterations must be 1..1024; samples and sizes must be nonempty/positive")
    windows = benchmark_windows(count)
    # Validate every size before the first hardware write.
    for size in sizes_kib:
        for layout in ("private", "split"):
            memory_regions(windows, list(range(count)), size * 1024, layout, private_spacing_bytes)
    rows = []
    # Engine construction consumes the global RNG for legacy compatibility.
    # This optional benchmark must not change inputs in the surrounding suite.
    with torch.random.fork_rng(devices=[]):
        for size in sizes_kib:
            cases = [("private", [engine], private_spacing_bytes) for engine in range(count)]
            if count > 1:
                cases += [(layout, list(range(count)), private_spacing_bytes) for layout in ("private", "split")]
            if (private_spacing_bytes is None and count == 2
                    and core.ANDROMEDA_CORE_COUNT == 2 and core.AVAILABLE_DRAM_SIZE_GB == 4):
                # Keep the former single-controller layout as an explicit
                # comparison alongside the corrected dual-controller default.
                cases.append(("private", list(range(count)), 512 << 20))
            for layout, engine_ids, spacing in cases:
                print(f"Memory {layout}: engines={engine_ids}, {size} KiB/engine, "
                      f"{iterations} transfers, {samples} samples, "
                      f"spacing={'board' if spacing is None else str(spacing >> 20) + ' MiB'}", flush=True)
                result = _run_case(windows, engine_ids, layout, size, iterations, samples,
                                   spacing)
                rows.extend(result)
                for row in result:
                    print(f"  {row['direction']:5s}: {row['mb_per_s']:.2f} MB/s aggregate, "
                          f"{row['latency_us']:.3f} us, exact data PASS", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", default="xdma0")
    parser.add_argument("--engines", type=int, default=None,
                        help="Use the first N hardware engines (default: all)")
    parser.add_argument("--sizes-kib", type=int, nargs="+", default=[64, 256, 512])
    parser.add_argument("--iterations", type=int, default=32,
                        help="Transfers per timing sample (1..1024; default: 32)")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--private-spacing-mib", type=int,
                        help="Space private data buffers by this many MiB; keep board ISA placement")
    parser.add_argument("--json", type=Path, help="Save hardware metadata and every timing sample")
    args = parser.parse_args()
    core.set_dma_device(args.dev)
    core.configure_clock_from_hardware()
    info = decode_hardware_info(core.HW_INFO_RAW)
    probe = core.UnifiedEngine()
    version = probe.read_reg32(core.UE_FPGA_VERSION_ADDR)
    print(f"{socket.gethostname()} / {args.dev}, image=0x{version:08x}")
    print(format_hardware_info(info))
    spacing = None if args.private_spacing_mib is None else args.private_spacing_mib * (1 << 20)
    if spacing is not None:
        print(f"Private data spacing: {args.private_spacing_mib} MiB (0x{spacing:x} bytes)")
    rows = run_memory_comparison(args.engines, args.sizes_kib, args.iterations, args.samples, spacing)
    print("\n| KiB/engine | Engines | Layout | Read MB/s | Write MB/s | Exact data |")
    print("|---:|:---|:---|---:|---:|:---|")
    for read, write in zip(rows[::2], rows[1::2]):
        layout = read['layout']
        if layout == "private" and read['source_spacing_bytes'] is not None:
            layout += f" ({read['source_spacing_bytes'] / (1 << 20):g} MiB apart)"
        print(f"| {read['size_kib']} | {','.join(map(str, read['engines']))} | "
              f"{layout} | {read['mb_per_s']:.2f} | {write['mb_per_s']:.2f} | PASS |")
    if args.json:
        args.json.write_text(json.dumps(dict(
            timestamp_utc=datetime.now(timezone.utc).isoformat(), hostname=socket.gethostname(),
            device=args.dev, image=f"0x{version:08x}", hardware_info=asdict(info),
            method="Median hardware latency; one warmup; repeated same buffer; "
                   "aggregate decimal MB/s; excludes host DMA; bit-exact roundtrip each sample",
            results=rows), indent=2) + "\n", encoding="utf-8")
        print(f"Results: {args.json.resolve()}")


if __name__ == "__main__":
    main()
