#!/usr/bin/env python3
"""FPGA consistency check for Qwen2.5-Omni layer-0 MLP down projection.

This is deliberately a hardware-vs-hardware test.  It compares the production
runtime-M streaming path at M=40 with forty independently emitted M=1 streaming
calls.  Both sides read the same deterministic BF16 input rows and the exact
layer-0 IF4 down-projection bytes from the current params artifact.  The host
only uploads bytes and compares the two BF16 result buffers; it never evaluates
the learned projection.

The test neither loads nor programs an FPGA image.  It uses software queue reset
and a bare HALT probe before/after the check, and leaves engines 0..7 idle.

Run from the repository root after activating the documented PyTorch env::

    CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' \
      python models/qwen2.5_omni_7b/qwen2.5_omni_7b_down_m40_hw_test.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

import user_dma_core
from user_dma_core import TYPE, UnifiedEngine


M = 40
K = 18_944
N = 3_584
ENGINES = 8
ENGINE_STRIDE = 0x0001_0000
EXPECTED_BUILD = 0x0305_D87D

# Scratch is confined to Omni's transient tensor region.  A subsequent normal
# model run recreates these activations.  No model/private-weight windows are
# touched, and none of these addresses contains the FPGA image.
SCRATCH_BASE = 0x1_E800_0000
SCRATCH_LIMIT = 0x1_F000_0000
RESET_HALT_ADDR = 0x1_FA00_0000
DYNAMIC_PROGRAM_ADDR = 0x1_FA10_0000
REFERENCE_PROGRAM_ADDR = 0x1_FA40_0000
MASTER_ISA_LIMIT = 0x1_FC80_0000

QUEUE_RESET_COMMAND = 0x8000_8000


def _align(value: int, alignment: int = 64) -> int:
    return (value + alignment - 1) // alignment * alignment


def _wait_idle(engine: UnifiedEngine, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while engine.is_queue_busy() and time.monotonic() < deadline:
        time.sleep(0.001)
    if engine.is_queue_busy():
        raise TimeoutError(f"{what}: queue remained busy after {timeout_s:.1f}s")


def _reset_queues(engines: list[UnifiedEngine], timeout_s: float = 3.0) -> None:
    for engine in engines:
        engine.write_reg32(user_dma_core.UE_QUEUE_CTRL_ADDR, QUEUE_RESET_COMMAND)
    for index, engine in enumerate(engines):
        _wait_idle(engine, timeout_s, f"engine {index} software reset")


def _halt_probe(engines: list[UnifiedEngine]) -> None:
    """Execute one shared immutable HALT on each selected queue."""
    compiler = engines[0]
    compiler.clear_capture_buffer()
    compiler.clear_inst_id()
    compiler.reset_isa_reg_counter()
    compiler.reset_inst_ptr_counter()
    compiler.start_capture()
    compiler.generate_instruction_halt()
    compiler.stop_capture()
    expected = compiler.get_capture_instruction_size_bytes()
    written = compiler.write_captured_instructions_to_dram(RESET_HALT_ADDR)
    if written != expected:
        raise IOError(f"HALT DMA wrote {written} of {expected} bytes")
    compiler.clear_capture_buffer()
    for index, engine in enumerate(engines):
        engine.start_execute_from_dram(RESET_HALT_ADDR)
        _wait_idle(engine, 3.0, f"engine {index} HALT probe")


def _read_layer0_down() -> tuple[bytes, bytes, dict, dict]:
    params_path = SCRIPT_DIR / "qwen2.5_omni_7b_bin" / "params.bin"
    manifest_path = params_path.with_suffix(".json")
    with manifest_path.open() as file_obj:
        top = json.load(file_obj)
    lm = top["regions"]["lm"]
    key = "language_model.layers.0.mlp.down_proj.weight.if4"
    section = lm["manifest"][key]
    if list(section["shape"]) != [N, K]:
        raise ValueError(f"{key} shape={section['shape']}, expected {[N, K]}")

    blocks = N * K // 64
    scale_bytes = blocks * 2
    data_bytes = N * K // 2
    if int(section["size"]) != scale_bytes + data_bytes:
        raise ValueError(
            f"{key} size={section['size']}, expected {scale_bytes + data_bytes}"
        )
    absolute = int(lm["offset"]) + int(section["offset"])
    with params_path.open("rb") as file_obj:
        file_obj.seek(absolute)
        scales = file_obj.read(scale_bytes)
        packed = file_obj.read(data_bytes)
    if len(scales) != scale_bytes or len(packed) != data_bytes:
        raise EOFError(f"short params.bin read for {key}")
    return scales, packed, top, section


def _dma_write_exact(engine: UnifiedEngine, address: int, payload, size: int,
                     label: str) -> None:
    written = engine.dma_write(engine.h2c_device, address, payload, size)
    if written != size:
        raise IOError(f"{label}: DMA wrote {written} of {size} bytes")


def _dma_read_bf16(engine: UnifiedEngine, address: int, elements: int,
                   label: str) -> torch.Tensor:
    size = elements * 2
    payload = bytearray(size)
    read = engine.dma_read(engine.c2h_device, address, payload, size)
    if read != size:
        raise IOError(f"{label}: DMA read {read} of {size} bytes")
    return torch.frombuffer(payload, dtype=torch.bfloat16).clone()


def _capture_program(
    engine: UnifiedEngine,
    program_addr: int,
    a_addr: int,
    scale_addr: int,
    data_addr: int,
    out_addr: int,
    rows_per_call: int,
    call_count: int,
) -> tuple[int, int]:
    """Capture one M=40 call or forty row-addressed M=1 calls."""
    if rows_per_call * call_count != M:
        raise ValueError("captured calls must cover exactly M=40 input rows")
    engine.clear_capture_buffer()
    engine.clear_inst_id()
    engine.reset_isa_reg_counter()
    engine.reset_inst_ptr_counter()
    engine.start_capture()
    m_reg = engine.alloc_isa_reg()
    k_reg = engine.alloc_isa_reg()
    n_reg = engine.alloc_isa_reg()
    engine.generate_instruction_add_set(m_reg, rows_per_call)
    engine.generate_instruction_add_set(k_reg, K)
    engine.generate_instruction_add_set(n_reg, N)
    for call in range(call_count):
        row = call * rows_per_call
        engine.quantized_matmat_core(
            M=rows_per_call,
            K=K,
            N=N,
            A_DRAM_ADDR=a_addr + row * K * 2,
            B_DRAM_ADDR=data_addr,
            OUTPUT_DRAM_ADDR=out_addr + row * N * 2,
            SCALE_DRAM_ADDR=scale_addr,
            data_type=TYPE.IF4,
            gpr_M_reg=m_reg,
            gpr_K_reg=k_reg,
            gpr_N_reg=n_reg,
        )
    engine.stop_capture()
    engine.generate_instruction_halt()
    program_bytes = engine.get_capture_instruction_size_bytes()
    if program_addr + program_bytes > MASTER_ISA_LIMIT:
        raise MemoryError(
            f"program 0x{program_addr:x}+{program_bytes} exceeds master ISA"
        )
    written = engine.write_captured_instructions_to_dram(program_addr)
    if written != program_bytes:
        raise IOError(
            f"program at 0x{program_addr:x}: DMA wrote {written} of {program_bytes}"
        )
    count = engine.get_capture_count()
    engine.clear_capture_buffer()
    return count, program_bytes


def _execute(engine: UnifiedEngine, address: int, timeout_s: float,
             label: str) -> tuple[int, float]:
    started = time.monotonic()
    engine.start_execute_from_dram(address)
    _wait_idle(engine, timeout_s, label)
    elapsed = time.monotonic() - started
    cycles = int(engine.read_reg32(user_dma_core.UE_LATENCY_COUNT_ADDR)) * int(
        user_dma_core.UE_PIPELINE_COUNTER_CLK_DIV
    )
    return cycles, elapsed


def _make_input() -> torch.Tensor:
    # Host data generation only.  The learned IF4 projection is never evaluated
    # by PyTorch.  A bounded distribution avoids infinities while exercising
    # all positive/negative input lanes.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0x25_0A_40)
    return (torch.randn((M, K), generator=generator) * 0.25).to(torch.bfloat16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", default="xdma0")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    user_dma_core.set_dma_device(args.dev)
    user_dma_core.configure_clock_from_hardware()
    info = user_dma_core.configured_hardware_info()
    if info.core_count < ENGINES or info.dram_size_gb != 8:
        raise RuntimeError(
            f"need U55C >=8 engines and 8 GiB; HW_INFO says "
            f"cores={info.core_count}, DRAM={info.dram_size_gb} GiB"
        )
    if torch.cuda.is_available():
        raise RuntimeError("a CUDA/ROCm accelerator is visible; hide it before this test")

    engines = [
        UnifiedEngine(
            BASE_ADDR=user_dma_core.UE_0_BASE_ADDR + index * ENGINE_STRIDE,
            program_dram_base=(
                DYNAMIC_PROGRAM_ADDR if index == 0 else RESET_HALT_ADDR
            ),
            tensor_dram_base=SCRATCH_BASE,
            params_dram_base=SCRATCH_BASE,
            init_unified_engine=False,
        )
        for index in range(ENGINES)
    ]
    builds = [
        int(engine.user_read_reg32(user_dma_core.UE_FPGA_VERSION_ADDR)) & 0xFFFF_FFFF
        for engine in engines
    ]
    if builds != [EXPECTED_BUILD] * ENGINES:
        raise RuntimeError(
            "installed build mismatch: "
            + ", ".join(f"e{i}=0x{value:08x}" for i, value in enumerate(builds))
        )

    print(f"HW_INFO: {user_dma_core.hardware_info_summary()}")
    print("FPGA builds: " + ", ".join(f"e{i}=0x{x:08x}" for i, x in enumerate(builds)))
    print("FPGA image programming: NOT PERFORMED")
    print(
        "Host role: deterministic BF16 input generation, DMA, and result comparison only; "
        "no CPU matmul/model inference"
    )
    print(
        "GPU visibility: disabled "
        f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}, "
        f"torch.cuda.is_available()={torch.cuda.is_available()})"
    )

    test_error: BaseException | None = None
    try:
        _reset_queues(engines)
        _halt_probe(engines)

        scales, packed, manifest, section = _read_layer0_down()
        cursor = SCRATCH_BASE
        scale_addr = cursor
        cursor = _align(cursor + len(scales))
        data_addr = cursor
        cursor = _align(cursor + len(packed))
        a_addr = cursor
        cursor = _align(cursor + M * K * 2)
        dynamic_out_addr = cursor
        cursor = _align(cursor + M * N * 2)
        reference_out_addr = cursor
        cursor = _align(cursor + M * N * 2)
        if cursor > SCRATCH_LIMIT:
            raise MemoryError(
                f"scratch ends at 0x{cursor:x}, limit is 0x{SCRATCH_LIMIT:x}"
            )

        input_bf16 = _make_input().contiguous()
        sentinel = torch.full((M * N,), -123.0, dtype=torch.bfloat16)
        _dma_write_exact(engines[0], scale_addr, scales, len(scales), "IF4 scales")
        _dma_write_exact(engines[0], data_addr, packed, len(packed), "IF4 packed data")
        _dma_write_exact(engines[0], a_addr, input_bf16, input_bf16.numel() * 2,
                         "BF16 A")
        _dma_write_exact(engines[0], dynamic_out_addr, sentinel,
                         sentinel.numel() * 2, "dynamic output sentinel")
        _dma_write_exact(engines[0], reference_out_addr, sentinel,
                         sentinel.numel() * 2, "reference output sentinel")

        dynamic_compiler = UnifiedEngine(
            BASE_ADDR=user_dma_core.UE_0_BASE_ADDR,
            program_dram_base=DYNAMIC_PROGRAM_ADDR,
            tensor_dram_base=SCRATCH_BASE,
            params_dram_base=SCRATCH_BASE,
            init_unified_engine=False,
        )
        reference_compiler = UnifiedEngine(
            BASE_ADDR=user_dma_core.UE_0_BASE_ADDR,
            program_dram_base=REFERENCE_PROGRAM_ADDR,
            tensor_dram_base=SCRATCH_BASE,
            params_dram_base=SCRATCH_BASE,
            init_unified_engine=False,
        )
        dyn_inst, dyn_bytes = _capture_program(
            dynamic_compiler, DYNAMIC_PROGRAM_ADDR, a_addr, scale_addr,
            data_addr, dynamic_out_addr, rows_per_call=M, call_count=1
        )
        ref_inst, ref_bytes = _capture_program(
            reference_compiler, REFERENCE_PROGRAM_ADDR, a_addr, scale_addr,
            data_addr, reference_out_addr, rows_per_call=1, call_count=M
        )

        dyn_cycles, dyn_wall = _execute(
            engines[0], DYNAMIC_PROGRAM_ADDR, args.timeout, "runtime-M=40 program"
        )
        ref_cycles, ref_wall = _execute(
            engines[0], REFERENCE_PROGRAM_ADDR, args.timeout, "forty M=1 program"
        )
        dynamic = _dma_read_bf16(
            engines[0], dynamic_out_addr, M * N, "runtime-M output"
        ).reshape(M, N)
        reference = _dma_read_bf16(
            engines[0], reference_out_addr, M * N, "M=1 reference output"
        ).reshape(M, N)

        dyn_bits = dynamic.reshape(-1).view(torch.uint16)
        ref_bits = reference.reshape(-1).view(torch.uint16)
        mismatch_mask = dyn_bits != ref_bits
        mismatch_count = int(mismatch_mask.sum())
        delta = (dynamic.float() - reference.float()).abs()
        max_abs = float(delta.nan_to_num(nan=float("inf")).max())
        row_mismatches = mismatch_mask.reshape(M, N).sum(dim=1)
        bad_rows = [
            (row, int(count))
            for row, count in enumerate(row_mismatches.tolist())
            if count
        ]
        output_elements = M * N
        dyn_sha = hashlib.sha256(dyn_bits.numpy().tobytes()).hexdigest()
        ref_sha = hashlib.sha256(ref_bits.numpy().tobytes()).hexdigest()
        weight_sha = hashlib.sha256(scales + packed).hexdigest()
        unchanged_dynamic = int((dynamic == sentinel.reshape(M, N)).sum())
        unchanged_reference = int((reference == sentinel.reshape(M, N)).sum())

        print(
            f"Artifact: schema={manifest.get('schema_version')} generation="
            f"{manifest.get('generation_id')} params_size={manifest.get('params_size')}"
        )
        print(
            f"Weight: layer0 down IF4 shape={section['shape']} bytes={section['size']} "
            f"sha256={weight_sha}"
        )
        print(
            f"Scratch: scales=0x{scale_addr:x}, data=0x{data_addr:x}, A=0x{a_addr:x}, "
            f"dynamic=0x{dynamic_out_addr:x}, reference=0x{reference_out_addr:x}, "
            f"end=0x{cursor:x}"
        )
        print(
            f"Programs: runtime-M=40 {dyn_inst} instructions/{dyn_bytes} bytes; "
            f"40xM=1 {ref_inst} instructions/{ref_bytes} bytes"
        )
        print(
            f"Timing: runtime-M=40 {dyn_cycles} cycles ({dyn_wall:.6f}s host wall); "
            f"40xM=1 {ref_cycles} cycles ({ref_wall:.6f}s host wall)"
        )
        print(
            f"RESULT mismatches={mismatch_count}/{output_elements} max_abs_diff={max_abs:g} "
            f"bad_rows={bad_rows}"
        )
        print(
            f"RESULT sha256 runtime-M={dyn_sha} M1-reference={ref_sha}; "
            f"sentinel_remaining runtime-M={unchanged_dynamic} M1-reference={unchanged_reference}"
        )
        if mismatch_count:
            first = int(torch.nonzero(mismatch_mask, as_tuple=False)[0])
            row, col = divmod(first, N)
            raise AssertionError(
                f"first mismatch row={row} col={col}: "
                f"runtime-M={float(dynamic[row, col])}, M1={float(reference[row, col])}"
            )
        if unchanged_dynamic or unchanged_reference:
            raise AssertionError("one or more output elements retained the sentinel")
        print("PASS: runtime M=40 is BF16 bit-identical to forty FPGA M=1 calls")
        return 0
    except BaseException as exc:
        test_error = exc
        raise
    finally:
        try:
            _reset_queues(engines)
            _halt_probe(engines)
            busy = [index for index, engine in enumerate(engines) if engine.is_queue_busy()]
            if busy:
                raise RuntimeError(f"cleanup left busy queues: {busy}")
            print("Cleanup: engines 0..7 software-reset, HALT-probed, and idle")
        except BaseException as cleanup_error:
            if test_error is None:
                raise
            print(f"CLEANUP ERROR after test failure: {cleanup_error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
