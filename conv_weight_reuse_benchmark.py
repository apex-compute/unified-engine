#!/usr/bin/env python3
"""Compare convolution implementations with identical stored tensors.

Run this script from the candidate checkout, selecting the driver checkout with
--repo. For IF4/IF8, compare --path conv on main and candidate. Main has no native
FLOAT16 CONV, so compare --path matmul on main with --path conv on candidate for
BF16/FP16. The matmul baseline includes device-side im2col, like the existing
DPDFNet native convolution emitter; weights remain unquantized.

--library selects a matching opaque SDK shared library. Its cycle counts are
RTL simulation results with a deterministic AXI bus, not physical DDR timings.
--device selects a programmed FPGA instead. Both time the entire resident
program (staging, arithmetic, writeback), excluding host packing and transfers.
Every timed output is checked against PyTorch using the stored input precision.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes as ct
import io
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys

import torch
import torch.nn.functional as F


# (input channels, output channels, input height/width, kernel)
CASES = {"stem3x3": (3, 64, 10, 3), "dense3x3": (64, 64, 10, 3),
         "wide3x3": (256, 64, 10, 3), "pointwise": (64, 64, 8, 1)}


def simulation_engine(udc, library):
    """Use the production emitter with the SDK's register and memory boundary."""
    lib = ct.CDLL(str(Path(library).resolve()))
    lib.ue_sim_spec_json.restype = ct.c_char_p
    udc.UE_AXI_DATA_WIDTH_BITS = json.loads(lib.ue_sim_spec_json())["profile"]["axi_data_width"]
    lib.ue_sim_spec_sha256.restype = ct.c_char_p
    lib.ue_sim_create.argtypes = [ct.c_char_p, ct.c_char_p]
    lib.ue_sim_create.restype = ct.c_void_p
    lib.ue_sim_destroy.argtypes = [ct.c_void_p]
    lib.ue_sim_reset.argtypes = [ct.c_void_p]
    lib.ue_sim_error.argtypes = [ct.c_void_p]
    lib.ue_sim_error.restype = ct.c_char_p
    lib.ue_sim_load.argtypes = [ct.c_void_p, ct.c_uint64, ct.c_void_p, ct.c_size_t]
    lib.ue_sim_read_memory.argtypes = lib.ue_sim_load.argtypes
    lib.ue_sim_write32.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_uint8]
    lib.ue_sim_read32.argtypes = [ct.c_void_p, ct.c_uint32, ct.POINTER(ct.c_uint32)]
    lib.ue_sim_wait_halt.argtypes = [ct.c_void_p, ct.c_uint64]

    class Engine(udc.UnifiedEngine):
        def __init__(self):
            super().__init__(params_dram_base=0x1000000, tensor_dram_base=0x30000000,
                             program_dram_base=0x50000000, clock_period_ns=3.0,
                             conv_geometry_mode=udc.CONV_GEOMETRY_QUEUE_CONFIG)
            self.sim = lib.ue_sim_create(lib.ue_sim_spec_sha256(), None)
            if not self.sim:
                raise RuntimeError("Could not create simulator")
            self.check(lib.ue_sim_reset(self.sim))

        def check(self, result):
            if result:
                raise RuntimeError(lib.ue_sim_error(self.sim).decode())

        def close(self):
            lib.ue_sim_destroy(self.sim)

        def dma_write(self, device, address, buffer, size):
            payload = ct.create_string_buffer(bytes(udc.dma_payload(buffer, size)))
            self.check(lib.ue_sim_load(self.sim, address, payload, size))
            return size

        def dma_read(self, device, address, buffer, size):
            self.check(lib.ue_sim_read_memory(self.sim, address, buffer.data_ptr(), size))
            return size

        def write_reg32(self, address, value):
            self.check(lib.ue_sim_write32(self.sim, address, value, 15))

        def read_reg32(self, address):
            result = ct.c_uint32()
            self.check(lib.ue_sim_read32(self.sim, address, ct.byref(result)))
            return result.value

        def wait_queue(self, *args, **kwargs):
            self.check(lib.ue_sim_wait_halt(self.sim, 5_000_000))

        def read_latency_cycles(self):
            # Simulated execution is independent of host wall time and cannot
            # wrap within the bounded wait above.
            return self.read_latency_counter_ticks() * udc.UE_PIPELINE_COUNTER_CLK_DIV

    return Engine()


def native_matmul(udc, engine, x, weights, bias, formats):
    """HWC input -> device im2col -> production native matmul -> HWC output."""
    channels, height, width = x.shape
    oc, _, kh, kw = weights.shape
    cp = (channels + 63) // 64 * 64
    oh, ow = height - kh + 1, width - kw + 1
    m, k = oh * ow, kh * kw * cp
    packed_x = torch.zeros(height, width, cp, dtype=formats.a.dtype)
    packed_x[:, :, :channels] = x.permute(1, 2, 0)
    packed_w = torch.zeros(oc, kh, kw, cp, dtype=formats.b.dtype)
    packed_w[:, :, :, :channels] = weights.permute(0, 2, 3, 1)
    def upload(value):
        addr = engine.allocate_params_dram(value.numel() * value.element_size())
        engine.dma_to_accelerator_memory(addr, value)
        return addr
    input_addr, weight_addr, bias_addr = map(upload, (packed_x, packed_w, bias))
    patch_addr = input_addr if kh == kw == 1 else engine.allocate_tensor_dram(m * k * 2)
    output_addr = engine.allocate_tensor_dram(m * oc * 2)
    engine.start_capture()
    if kh != 1 or kw != 1:
        for oy in range(oh):
            for ky in range(kh):
                for kx in range(kw):
                    source = input_addr + ((oy + ky) * width + kx) * cp * 2
                    target = patch_addr + (oy * ow * k + (ky * kw + kx) * cp) * 2
                    engine.accelerator_memory_to_sram(source, 0, 0,
                        memcpy_length_bytes=ow * cp * 2)
                    engine.sram_to_accelerator_memory(0, target, 0,
                        memcpy_length_bytes=ow * cp * 2,
                        stride_bytes_per_chunk=cp * 2, stride_jump_bytes=k * 2)
    engine.matmat_mul_core(M=m, K=k, N=oc, A_DRAM_ADDR=patch_addr,
        B_DRAM_ADDR=weight_addr, C_DRAM_ADDR=bias_addr, OUTPUT_DRAM_ADDR=output_addr,
        is_B_quantized=False, bias_mode="broadcast_N", formats=formats)
    engine.generate_instruction_halt()
    engine.stop_capture()
    address = engine.get_program_dram_addr()
    engine.write_captured_instructions_to_dram(address)
    engine.start_execute_from_dram(address)
    engine.wait_queue(30)
    cycles = engine.read_latency_cycles()
    result = engine.dma_from_accelerator_memory(output_addr, (m, oc), dtype=formats.output.dtype)
    return result.reshape(oh, ow, oc).permute(2, 0, 1), cycles, m * k * 2 if patch_addr != input_addr else 0


def run_case(udc, engine, name, precision, path, *, formats=None):
    channels, oc, hw, kernel = CASES[name]
    fmt = udc.StorageFormat.FP16 if precision == "fp16" else udc.StorageFormat.BF16
    formats = formats or udc.TensorFormats.uniform(fmt)
    generator = torch.Generator().manual_seed(126)
    x = (torch.randint(-8, 9, (channels, hw, hw), generator=generator).float() / 8).to(formats.a.dtype)
    codes = torch.randint(-3, 4, (oc, channels, kernel, kernel), generator=generator)
    native = precision in ("bf16", "fp16")
    weights = (codes.float() / 16 + (1 / 1024 if native else 0)).to(formats.b.dtype)
    bias = (torch.arange(oc).float() % 5 / 16).to(formats.bias.dtype)
    reference = F.conv2d(x.float()[None], weights.float(), bias.float())[0].to(formats.output.dtype)
    if path == "matmul":
        if not native:
            raise ValueError("The matmul baseline is for native BF16/FP16 weights")
        result, cycles, scratch = native_matmul(udc, engine, x, weights, bias, formats)
    else:
        kind = udc.TYPE.FLOAT16 if native else udc.TYPE[precision.upper()]
        result = engine.run_conv2d_layer(x, weights if native else codes,
            stride_s=1, pad=0, bias=bias, data_type=kind,
            scale_mag=1.0 if native else 1/16,
            **({"formats": formats} if native else {}))
        cycles, scratch = engine.last_conv_cycles, 0
    error = (reference.float() - result.float()).square().mean().item()
    signal = reference.float().square().mean().item()
    snr = 10 * math.log10(signal / max(error, 1e-30))
    if not torch.isfinite(result).all() or snr < 40:
        raise AssertionError(f"{name}/{precision}/{path}: SNR={snr:.2f} dB")
    return {"case": name, "precision": precision, "path": path,
            "cycles": cycles, "snr_db": snr, "im2col_bytes": scratch,
            "shape": [channels, oc, hw, kernel],
            "clock_period_ns": engine._clock_period_ns}


def run_hardware_suite(record):
    """Numerical and performance coverage appended after the legacy HW suite."""
    import user_dma_core as udc
    from itertools import product
    python_state, torch_state = random.getstate(), torch.get_rng_state()
    try:
        for precision in ("bf16", "fp16"):
            for name in CASES:
                for path in ("matmul", "conv"):
                    samples = []
                    for _ in range(3):
                        engine = udc.UnifiedEngine(conv_geometry_mode=udc.CONV_GEOMETRY_QUEUE_CONFIG)
                        samples.append(run_case(udc, engine, name, precision, path))
                    cycles = statistics.median(s["cycles"] for s in samples)
                    score = min(s["snr_db"] for s in samples)
                    label = f"conv_reuse_{precision}_{name}_{path}"
                    record(label, str(CASES[name]), cycles=cycles, snr_db=score)
                    print(f"{label}: PASS cycles={cycles} SNR={score:.2f} dB", flush=True)
        # The storage selectors are independent, including the output format.
        for a, b, output in product(udc.StorageFormat, repeat=3):
            formats = udc.TensorFormats(a=a, b=b, output=output, bias=output)
            for name in ("stem3x3", "dense3x3"):
                engine = udc.UnifiedEngine(conv_geometry_mode=udc.CONV_GEOMETRY_QUEUE_CONFIG)
                row = run_case(udc, engine, name, "bf16", "conv", formats=formats)
                label = f"conv_reuse_mixed_{name}_fmt{formats.bits:02x}"
                record(label, str(CASES[name]), cycles=row["cycles"], snr_db=row["snr_db"])
                print(f"{label}: PASS SNR={row['snr_db']:.2f} dB", flush=True)
    finally:
        random.setstate(python_state)
        torch.set_rng_state(torch_state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument("--library", type=Path)
    backend.add_argument("--device")
    parser.add_argument("--path", choices=("conv", "matmul"), required=True)
    parser.add_argument("--formats", default="if4,if8,bf16,fp16")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    sys.path.insert(0, str(args.repo.resolve() / "pcie_utils"))
    import user_dma_core as udc
    torch.set_num_threads(4)
    rows = []
    for name in args.cases.split(","):
        for precision in args.formats.split(","):
            samples = []
            for _ in range(args.repeats):
                with contextlib.redirect_stdout(io.StringIO()):
                    engine = (simulation_engine(udc, args.library) if args.library else
                              udc.UnifiedEngine(device_name=args.device,
                                  conv_geometry_mode=udc.CONV_GEOMETRY_QUEUE_CONFIG))
                    try:
                        samples.append(run_case(udc, engine, name, precision, args.path))
                    finally:
                        if args.library:
                            engine.close()
            row = samples[0] | {"cycles": statistics.median(s["cycles"] for s in samples),
                                "cycle_samples": [s["cycles"] for s in samples],
                                "snr_db": min(s["snr_db"] for s in samples)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    revision = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    report = {"driver_revision": revision, "backend": "rtl" if args.library else "fpga",
              "driver_dirty": subprocess.run(
                  ["git", "-C", str(args.repo), "diff", "--quiet", "HEAD"]).returncode != 0,
              "library": str(args.library) if args.library else None,
              "rows": rows}
    if args.library:
        lib = ct.CDLL(str(args.library.resolve()))
        lib.ue_sim_spec_sha256.restype = ct.c_char_p
        report["hardware_spec_sha256"] = lib.ue_sim_spec_sha256().decode()
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
