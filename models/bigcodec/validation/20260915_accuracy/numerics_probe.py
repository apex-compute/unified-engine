#!/usr/bin/env python3
"""Capture numerical proofs offline; --execute runs them on locked Italy hardware.

No checkpoint, audio, or ignored diagnostic files are required. Run with the
BigCodec Python environment. Default output is numerics_probe_offline.log;
--execute writes numerics_probe_execute.json. Each native case uploads its
program, constants and inputs, executes one START/HALT, then reads DRAM.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "models/bigcodec"))
from bigcodec_device import shared, udc
from bigcodec_lstm import _two_product_sram, _two_sum_sram, _compensated_cell_sram, _tanh_sram
from bigcodec_tanh import compensated_tanh_sram
from test_bigcodec_lstm import NativeArithmeticMemoryEngine
from test_bigcodec_tanh import TanhEngine

INPUT_BASE, MODEL_BASE, TENSOR_BASE = 0x80000000, 0x90000000, 0xB0000000
GUARD, SENTINEL = 64, -3.25


def digest(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def metrics(actual, expected):
    difference = actual.double() - expected.double()
    denominator = float(expected.double().norm())
    return dict(relative_l2=float(difference.norm()) / denominator if denominator else None,
                max_abs=float(difference.abs().max()),
                exact_mismatches=int((actual != expected).sum()),
                bf16_bit_mismatches=int((actual.bfloat16().view(torch.int16)
                                         != expected.bfloat16().view(torch.int16)).sum()))


def source_hashes():
    paths = {Path(__file__).resolve(), ROOT / "models/dpdfnet8khz/dpdfnet8khz_engine.py"}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(ROOT) and path.suffix == ".py" and path.is_file():
                paths.add(path)
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def allocate(name, source, output_elements):
    source = source.flatten().bfloat16().contiguous()
    if source.numel() % 64 or output_elements % 64:
        raise ValueError("Buffers must contain pad64 elements")
    inputs = torch.full((source.numel() + 2 * GUARD,), SENTINEL, dtype=torch.bfloat16)
    inputs[GUARD:-GUARD] = source
    outputs = torch.full((output_elements + 2 * GUARD,), SENTINEL, dtype=torch.bfloat16)
    outputs[GUARD:-GUARD] = float("nan")
    return dict(name=name, source=source, inputs=inputs, outputs=outputs, constants={},
                input_address=INPUT_BASE + GUARD * 2, output_address=TENSOR_BASE + GUARD * 2)


def buffers_intact(case, inputs, outputs):
    return dict(input_unchanged=torch.equal(inputs.view(torch.int16), case["inputs"].view(torch.int16)),
                guards_intact=bool((outputs[:GUARD] == SENTINEL).all()
                                   and (outputs[-GUARD:] == SENTINEL).all()))


def capture(case, image, emit, model_engine):
    # The committed interpreter checks SRAM geometry, operand banks and DMA
    # bounds. Independent FP32 contracts below also check the emitted math.
    model_engine.regions = {INPUT_BASE: case["inputs"].clone(),
                            TENSOR_BASE: case["outputs"].clone(), **case["constants"]}
    emit(model_engine)
    modeled_buffers = model_engine.regions[TENSOR_BASE]
    case["modeled"] = modeled_buffers[GUARD:-GUARD].float().clone()
    oracle_guards = buffers_intact(case, model_engine.regions[INPUT_BASE], modeled_buffers)
    program_address = image.align(128)
    engine = shared._WholeGraphEngine(program_address)
    with contextlib.redirect_stdout(io.StringIO()):
        engine.start_capture()
        emit(engine)
        halt_index = engine.capture_count
        engine.generate_instruction_halt()
        engine.stop_capture()
    issues = udc.check_isa_jumps(engine.capture_buffer, program_address, name=case["name"])
    kinds = [(instruction.words[0] >> 8) & 15 for instruction in engine.capture_buffer]
    if (issues or kinds.count(udc.INSTRUCTION_HALT) != 1 or udc.INSTRUCTION_SWI in kinds
            or any(kind != udc.INSTRUCTION_NOP for kind in kinds[halt_index + 1:])):
        raise AssertionError(f"Invalid single-HALT program: {issues}")
    program = b"".join(instruction.get_bytes() for instruction in engine.capture_buffer)
    if len(program) % 64:
        raise AssertionError("Program fetch alignment")
    image.write(program_address, program)
    case.update(raw=torch.frombuffer(bytearray(image.data), dtype=torch.uint8),
                program_address=program_address)
    case["record"] = dict(name=case["name"], instructions=engine.capture_count,
        program_bytes=len(program), program_sha256=hashlib.sha256(program).hexdigest(),
        resident_bytes=case["raw"].numel(), resident_sha256=digest(case["raw"]),
        input_sha256=digest(case["source"]), modeled_sha256=digest(case["modeled"]),
        program_address=program_address, model_base=MODEL_BASE,
        input_address=case["input_address"], output_address=case["output_address"],
        halt_index=halt_index, hardware_executed=False, offline_guards=oracle_guards)
    check_ranges(case)
    case["record"]["offline_contract"] = validate(case, case["modeled"])
    if not all(oracle_guards.values()) or not case["record"]["offline_contract"]["passed"]:
        raise AssertionError(case["record"])
    return case


def arithmetic_case():
    generator = torch.Generator().manual_seed(15092026)
    random_count = 100352
    a = (torch.randn(random_count, generator=generator) * 20).bfloat16().float()
    b = (torch.rand(random_count, generator=generator) * 2 - 1).bfloat16().float()
    mantissas = torch.arange(128, 256).float() / 128
    grid_a = mantissas[:, None].expand(128, 128).flatten()
    grid_b = (mantissas[None, :] / 2).expand(128, 128).flatten()
    a, b = torch.cat((a, grid_a, -grid_a)), torch.cat((b, grid_b, grid_b))
    # Cancellation, small cell increments, zero, and a non-exact TwoSum case.
    a[:9] = torch.tensor([16., 1., 1.0078125, .00390625, -16., -1., 0., 20.625, 1.0078125])
    b[:9] = torch.tensor([.00390625, -1., -1., 16., -.00390625, 1., 0., .9921875, -.0034332275390625])
    count = a.numel()
    case = allocate("two_product_and_approximate_sum", torch.cat((a, b)), 4 * count)
    case.update(a=a, b=b, count=count)

    def emit(engine):
        for offset in range(0, count, 4096):
            take = min(4096, count - offset)
            engine.accelerator_memory_to_sram(case["input_address"] + offset * 2, 0, take)
            engine.accelerator_memory_to_sram(case["input_address"] + (count + offset) * 2, 0x2000, take)
            for section, helper in ((0, _two_product_sram), (2, _two_sum_sram)):
                helper(engine, 0, 0x2000, 0x4000, 0x6000, take)
                for part, address in enumerate((0x4000, 0x6000)):
                    engine.sram_to_accelerator_memory(address,
                        case["output_address"] + ((section + part) * count + offset) * 2, take)

    capture(case, shared._ImageBuilder(MODEL_BASE, TENSOR_BASE), emit, NativeArithmeticMemoryEngine())
    case["record"].update(random_pairs=random_count, mantissa_grid_pairs=32768, total_pairs=count,
        contract="Native double-RNE numerical components must match; high+low products must equal FP32 products. "
                 "Recovered sums are approximate: relative L2 <= 1e-6, max absolute error <= 0.001. "
                 "Signed-zero bit differences are reported separately.")
    return case


def small_update_case():
    count = 64
    case = allocate("cell_small_updates", torch.cat([torch.full((count,), value)
                    for value in (1., 1., 1. / 256, 16., 0.)]), count * 2)

    def emit(engine):
        for index in range(5):
            engine.accelerator_memory_to_sram(case["input_address"] + index * count * 2,
                                               index * 0x2000, count)
        for _ in range(64):
            _compensated_cell_sram(engine, 0, 0x2000, 0x4000, 0x6000, 0x8000, count)
        engine.sram_to_accelerator_memory(0x6000, case["output_address"], count)
        engine.sram_to_accelerator_memory(0x8000, case["output_address"] + count * 2, count)

    capture(case, shared._ImageBuilder(MODEL_BASE, TENSOR_BASE), emit, NativeArithmeticMemoryEngine())
    case["record"].update(steps=64, initial_cell=16., increment=1. / 256,
        contract="64 high+low cell updates must recover exactly 16.25; compare native components with double-RNE model.")
    return case


def tanh_case():
    # Every positive normal BF16 code through 32, both signs, signed zeros and
    # pad64. Quiet values cover 2^-126..2^-10. Subnormal input flush is outside
    # this contract and is not silently treated as an arithmetic regression.
    positive = torch.arange(0x0080, 0x4201, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
    value = torch.cat((positive, -positive, torch.tensor([0., -0.], dtype=torch.bfloat16)))
    value = torch.cat((value, torch.zeros((-value.numel()) % 64, dtype=torch.bfloat16)))
    count = value.numel()
    case = allocate("compensated_tanh_grid", value, count * 2)
    case.update(count=count, positive_count=positive.numel())
    image = shared._ImageBuilder(MODEL_BASE, TENSOR_BASE)
    identity_value = torch.eye(64, dtype=torch.bfloat16).flatten()
    identity = image.allocate(identity_value, alignment=128)
    case["constants"][identity] = identity_value

    def emit(engine):
        engine.accelerator_memory_to_sram(identity, 0x80000, 4096)
        for offset in range(0, count, 4096):
            take = min(4096, count - offset)
            engine.accelerator_memory_to_sram(case["input_address"] + offset * 2, 0, take)
            for section, helper in enumerate((_tanh_sram, compensated_tanh_sram)):
                helper(engine, 0, 0x4000, take)
                engine.sram_to_accelerator_memory(0x4000,
                    case["output_address"] + (section * count + offset) * 2, take)

    capture(case, image, emit, TanhEngine())
    case["record"].update(count=count, positive_normal_bf16_codes=["0x0080", "0x4200"],
        contract="Independent Torch FP32 tanh reference; compensated error <= 1.2 times ideal BF16 rounding floor, "
                 "max absolute error < 0.0023, <= 35% of baseline error, exact quiet normal values <= 2^-10, "
                 "odd symmetry and output in [-1,1]. No bit-exact reciprocal/SFU assertion; subnormals excluded.")
    return case


def validate(case, actual):
    result = dict(finite=bool(torch.isfinite(actual).all()))
    if not result["finite"]:
        return {**result, "passed": False}
    if case["name"] == "two_product_and_approximate_sum":
        p, pe, s, se = actual.reshape(4, case["count"])
        result["components_vs_native_double_rne_model"] = metrics(actual, case["modeled"])
        result["product_recovered_vs_exact_fp32"] = metrics(p + pe, case["a"] * case["b"])
        result["sum_recovered_vs_exact_fp32"] = metrics(s + se, case["a"] + case["b"])
        sums = result["sum_recovered_vs_exact_fp32"]
        passed = (result["components_vs_native_double_rne_model"]["exact_mismatches"] == 0
                  and result["product_recovered_vs_exact_fp32"]["exact_mismatches"] == 0
                  and sums["relative_l2"] <= 1e-6 and sums["max_abs"] <= .001)
    elif case["name"] == "cell_small_updates":
        high, low = actual.reshape(2, 64)
        result["components_vs_native_double_rne_model"] = metrics(actual, case["modeled"])
        result["recovered_vs_exact_fp32"] = metrics(high + low, torch.full((64,), 16.25))
        passed = (result["components_vs_native_double_rne_model"]["exact_mismatches"] == 0
                  and result["recovered_vs_exact_fp32"]["exact_mismatches"] == 0)
    else:
        baseline, compensated = actual.reshape(2, case["count"])
        source = case["source"].float()
        reference = source.tanh()
        ideal = reference.bfloat16().float()
        result.update(baseline_vs_fp32=metrics(baseline, reference),
                      compensated_vs_fp32=metrics(compensated, reference),
                      ideal_bf16_vs_fp32=metrics(ideal, reference))
        quiet = source.abs() <= 2. ** -10
        positive_count = case["positive_count"]
        result["quiet_exact_mismatches"] = int((compensated[quiet] != source[quiet]).sum())
        result["odd_symmetry"] = torch.equal(compensated[:positive_count],
                                              -compensated[positive_count:2 * positive_count])
        new, old, floor = (result[key]["relative_l2"] for key in
                          ("compensated_vs_fp32", "baseline_vs_fp32", "ideal_bf16_vs_fp32"))
        passed = (new <= 1.2 * floor and new <= .35 * old
                  and result["compensated_vs_fp32"]["max_abs"] < .0023
                  and result["quiet_exact_mismatches"] == 0 and result["odd_symmetry"]
                  and bool((compensated.abs() <= 1).all()))
    result["passed"] = bool(passed)
    return result


def check_ranges(case, dram_size_gb=None):
    spans = [(INPUT_BASE, INPUT_BASE + case["inputs"].numel() * 2, MODEL_BASE),
             (MODEL_BASE, MODEL_BASE + case["raw"].numel(), TENSOR_BASE),
             (TENSOR_BASE, TENSOR_BASE + case["outputs"].numel() * 2, 0xD0000000)]
    if any(start % 128 or end <= start or end > limit for start, end, limit in spans):
        raise ValueError("Probe buffers exceed their disjoint DRAM arenas")
    required_end = max(end for _, end, _ in spans)
    if dram_size_gb is not None:
        if isinstance(dram_size_gb, bool) or not isinstance(dram_size_gb, int) or dram_size_gb <= 0:
            raise ValueError("Invalid reported DRAM capacity")
        visible_end = INPUT_BASE + min(dram_size_gb * 2**30, 2**31)
        if required_end > visible_end:
            raise ValueError("Probe exceeds detected addressable DRAM")
    return required_end


def execute_case(engine, clock_ns, case, args):
    version = engine.user_read_reg32(udc.UE_FPGA_VERSION_ADDR) & 0xFFFFFFFF
    if version != args.expected_version or engine.is_queue_busy():
        raise RuntimeError("Wrong FPGA version or queue busy; no reset/upload performed")
    engine.software_reset(run_dram_self_test=False)
    for address, value in ((MODEL_BASE, case["raw"]), (INPUT_BASE, case["inputs"]),
                           (TENSOR_BASE, case["outputs"])):
        size = value.numel() * value.element_size()
        if engine.dma_write(engine.h2c_device, address, value, size) != size:
            raise RuntimeError("Short DMA upload")
    record = case["record"]
    engine.write_reg32(udc.UE_INT_REG, 1)
    started = time.perf_counter()
    engine.start_execute_from_dram(case["program_address"])
    record.update(hardware_executed=True, starts=1, hardware_version=f"0x{version:08x}")
    while engine.read_reg32(udc.UE_INT_REG) & 3 != udc.INT_CAUSE_HALT or engine.is_queue_busy():
        if time.perf_counter() - started > args.timeout:
            raise TimeoutError(f"{case['name']}: HALT timeout")
        time.sleep(.0001)
    record.update(halts=1, host_s=time.perf_counter() - started,
                  fpga_cycles=engine.read_latency_cycles())
    record["fpga_s"] = record["fpga_cycles"] * clock_ns / 1e9
    outputs, inputs = torch.empty_like(case["outputs"]), torch.empty_like(case["inputs"])
    for address, value in ((TENSOR_BASE, outputs), (INPUT_BASE, inputs)):
        if engine.dma_read(engine.c2h_device, address, value, value.numel() * 2) != value.numel() * 2:
            raise RuntimeError("Short DMA readback")
    actual = outputs[GUARD:-GUARD].float()
    record.update(buffers_intact(case, inputs, outputs))
    record.update(native_contract=validate(case, actual), output_sha256=digest(actual))
    record["passed"] = bool(record["input_unchanged"] and record["guards_intact"]
                             and record["native_contract"]["passed"])
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Explicitly run on Italy; otherwise no device access")
    parser.add_argument("--suite", choices=("all", "arithmetic", "tanh"), default="all")
    parser.add_argument("--expected-version", type=lambda value: int(value, 0), default=0x40519E0A)
    parser.add_argument("--cpu-core", type=int, default=9)
    parser.add_argument("--timeout", type=float, default=60.)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.cpu_core not in os.sched_getaffinity(0) or not 0 <= args.expected_version < 2**32
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("Invalid CPU core, version or timeout")
    output = args.output or HERE / ("numerics_probe_execute.json" if args.execute else "numerics_probe_offline.log")
    if output.suffix not in (".json", ".log"):
        parser.error("Report output must use .json or .log")
    os.sched_setaffinity(0, {args.cpu_core})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    udc.UE_AXI_DATA_WIDTH_BITS = 256
    cases = []
    if args.suite in ("all", "arithmetic"):
        cases.extend((arithmetic_case(), small_update_case()))
    if args.suite in ("all", "tanh"):
        cases.append(tanh_case())
    report = dict(mode="native" if args.execute else "offline_capture", hostname=socket.gethostname(),
        torch_version=str(torch.__version__), cpu_core=args.cpu_core, source_sha256=source_hashes(),
        native_arithmetic_model="RNE to 10 fraction bits, then RNE to BF16; finite normal arithmetic. "
                                "This is not a complete matmul/SFU/underflow emulator.",
        cases=[case["record"] for case in cases], passed=not args.execute)

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        if args.execute:
            if socket.gethostname().split(".")[0].lower() != "italy":
                raise RuntimeError("Native execution is restricted to Italy")
            from bigcodec_precompiled import StreamingEngine
            from yolov5_common import configure_hardware_runtime
            # Existing cross-task lock is opened read-only; never create or chmod it.
            with open("/tmp/pcie_ci_hw_italy.lock", "r") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                clock_ns, info, _ = configure_hardware_runtime(device="rk", dev="xdma0", cycle_override_ns=None)
                if info.axi_data_width_bits != 256:
                    raise RuntimeError("AXI256 hardware required")
                required_end = max(check_ranges(case, info.dram_size_gb) for case in cases)
                report.update(source_sha256=source_hashes(), clock_period_ns=clock_ns,
                              detected_dram_gib=info.dram_size_gb, required_dram_end=required_end)
                with StreamingEngine(clock_period_ns=clock_ns,
                        conv_geometry_mode=udc.CONV_GEOMETRY_QUEUE_CONFIG) as engine:
                    for case in cases:
                        result = execute_case(engine, clock_ns, case, args)
                        save()
                        print(json.dumps(result, allow_nan=False), flush=True)
                        if not result["passed"]:
                            raise AssertionError(f"{case['name']}: numerical contract failed")
                report["passed"] = True
    except BaseException as error:
        report.update(passed=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
    print(json.dumps(dict(mode=report["mode"], passed=report["passed"], report=str(output),
                         cases=len(cases)), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
