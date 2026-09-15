#!/usr/bin/env python3
"""Reproduce the isolated decoder LSTM comparison on frozen, committed tokens.

Default: regenerate official references and capture two native programs without
device access. --execute runs both programs on locked Italy hardware. Optional
--review-native BASELINE IMPROVED evaluates supplied saved native tensors on CPU.
Only a JSON report is written; no checkpoint, program image or tensor is copied.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import io
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE.parent))
import numerics_probe as proof
from bigcodec_common import CHECKPOINT_SHA256, DEFAULT_CHECKPOINT, load_models, load_tokens, sha256_file
from bigcodec_lstm import prepare_lstm, emit_lstm, scratch_bytes

TOKENS = ROOT / "models/bigcodec/validation/20260914_optimized_italy/p232_007_optimized_bf16_fpga.tokens.npz"
TOKEN_SHA256 = "34da5227f77c3f643159facff138f27827176c9555631c81629032ecf9d07044"
INPUT_SHA256 = "66242b3fbc6279796be42664c97e58c00a740d69e6a7773aade20ed42dbd4cd9"
REFERENCE_SHA256 = "01f5ae1c5305e95b236164acb3e9d0359a5a1d26590244c0b7108ba2f7406f4b"
SHAPE, NATIVE_SAMPLES = (317, 1536), 63295


def references(checkpoint):
    if sha256_file(TOKENS) != TOKEN_SHA256:
        raise ValueError("Frozen token file hash mismatch")
    token_ids, metadata = load_tokens(TOKENS)
    encoder, decoder = load_models(checkpoint, remove_weight_norm=False)
    del encoder
    with torch.inference_mode():
        embedding = decoder.vq2emb(torch.from_numpy(token_ids).reshape(1, -1, 1)).transpose(1, 2)
        source = decoder.model[0](embedding)
        reference = decoder.model[1](source)[0].T.contiguous()
        packed = source[0].T.contiguous().bfloat16()
        # Separate the initial BF16 input conversion from recurrent arithmetic.
        input_rounding_control = decoder.model[1](packed.float().T.unsqueeze(0))[0].T.contiguous()
    if (tuple(packed.shape) != SHAPE or proof.digest(packed) != INPUT_SHA256
            or proof.digest(reference) != REFERENCE_SHA256):
        raise ValueError("Regenerated input/reference differs from the historical fixed control")
    decoder.remove_weight_norm()
    tail = decoder.model[2:]
    with torch.inference_mode():
        reference_wave = tail(reference.T.unsqueeze(0)).flatten()[:NATIVE_SAMPLES]
        input_rounding_wave = tail(input_rounding_control.T.unsqueeze(0)).flatten()[:NATIVE_SAMPLES]
    provenance = dict(checkpoint_sha256=CHECKPOINT_SHA256, tokens=str(TOKENS.relative_to(ROOT)),
        tokens_sha256=TOKEN_SHA256, token_count=317, source_audio=metadata,
        input_bf16_sha256=proof.digest(packed), reference_fp32_sha256=proof.digest(reference),
        reference_wave_fp32_sha256=proof.digest(reference_wave), native_shape=list(SHAPE),
        native_samples=NATIVE_SAMPLES, sample_rate=16000,
        input_rounding_only=dict(lstm=proof.metrics(input_rounding_control, reference),
                                 fp32_tail_waveform=proof.metrics(input_rounding_wave, reference_wave)))
    return decoder, packed, reference, reference_wave, provenance


def capture(decoder, source, improved):
    name = "fused_compensated_cell_tanh" if improved else "baseline"
    elements = source.numel()
    workspace = scratch_bytes(SHAPE, compensated_cell=improved, preserve_cell_residual=True)
    case = proof.allocate(name, source, elements)
    # Output and workspace have separate guards; match the original probe ABI.
    output_stop = proof.GUARD + elements
    scratch_start = output_stop + proof.GUARD
    outputs = torch.full((elements + workspace // 2 + 3 * proof.GUARD,),
                         proof.SENTINEL, dtype=torch.bfloat16)
    outputs[proof.GUARD:output_stop] = float("nan")
    outputs[scratch_start:-proof.GUARD] = float("nan")
    guards = torch.ones(outputs.numel(), dtype=torch.bool)
    guards[proof.GUARD:output_stop] = False
    guards[scratch_start:-proof.GUARD] = False
    case.update(outputs=outputs, output_stop=output_stop, guard_mask=guards)
    image = proof.shared._ImageBuilder(proof.MODEL_BASE, proof.TENSOR_BASE)
    zero = image.allocate(torch.zeros(64, dtype=torch.bfloat16), alignment=128)
    identity = image.allocate(torch.eye(64, dtype=torch.bfloat16), alignment=128)
    plan = prepare_lstm(decoder.model[1].lstm, image, input_shape=SHAPE,
        input_address=case["input_address"], output_address=case["output_address"],
        scratch_address=proof.TENSOR_BASE + scratch_start * 2,
        identity_address=identity, zero_address=zero, skip=decoder.model[1].skip,
        recurrent_precision="bf16", compensated_cell=improved,
        preserve_cell_residual=True, compensated_tanh=improved, fused_projection=improved)
    program_address = image.align(128)
    engine = proof.shared._WholeGraphEngine(program_address)
    with contextlib.redirect_stdout(io.StringIO()):
        engine.start_capture()
        emit_lstm(engine, plan)
        halt = engine.capture_count
        engine.generate_instruction_halt()
        engine.stop_capture()
    kinds = [(instruction.words[0] >> 8) & 15 for instruction in engine.capture_buffer]
    if (proof.udc.check_isa_jumps(engine.capture_buffer, program_address, name=name)
            or kinds.count(proof.udc.INSTRUCTION_HALT) != 1 or proof.udc.INSTRUCTION_SWI in kinds
            or any(kind != proof.udc.INSTRUCTION_NOP for kind in kinds[halt + 1:])):
        raise ValueError("Invalid single-HALT LSTM program")
    program = b"".join(instruction.get_bytes() for instruction in engine.capture_buffer)
    if len(program) % 64:
        raise ValueError("Invalid program fetch alignment")
    image.write(program_address, program)
    case.update(raw=torch.frombuffer(bytearray(image.data), dtype=torch.uint8),
                program_address=program_address)
    record = dict(variant=name, input_sha256=proof.digest(source),
        program_bytes=len(program), program_sha256=proof.digest(torch.frombuffer(bytearray(program), dtype=torch.uint8)),
        resident_bytes=case["raw"].numel(), resident_sha256=proof.digest(case["raw"]),
        scratch_bytes=workspace, instructions=engine.capture_count, halt_index=halt,
        compensated_cell=improved, compensated_tanh=improved, fused_projection=improved,
        recurrent_precision="bf16", hardware_executed=False)
    case["record"] = record
    proof.check_ranges(case)
    return case


def execute(engine, clock_ns, case, timeout, expected_version):
    version = engine.user_read_reg32(proof.udc.UE_FPGA_VERSION_ADDR) & 0xFFFFFFFF
    if version != expected_version or engine.is_queue_busy():
        raise RuntimeError("Wrong FPGA version or busy queue; no reset/upload performed")
    engine.software_reset(run_dram_self_test=False)
    for address, value in ((proof.MODEL_BASE, case["raw"]), (proof.INPUT_BASE, case["inputs"]),
                           (proof.TENSOR_BASE, case["outputs"])):
        size = value.numel() * value.element_size()
        if engine.dma_write(engine.h2c_device, address, value, size) != size:
            raise RuntimeError("Short DMA upload")
    engine.write_reg32(proof.udc.UE_INT_REG, 1)
    started = time.perf_counter()
    engine.start_execute_from_dram(case["program_address"])
    record = case["record"]
    record.update(hardware_executed=True, starts=1, hardware_version=f"0x{version:08x}")
    while (engine.read_reg32(proof.udc.UE_INT_REG) & 3 != proof.udc.INT_CAUSE_HALT
           or engine.is_queue_busy()):
        if time.perf_counter() - started > timeout:
            raise TimeoutError(f"{case['name']}: HALT timeout")
        time.sleep(.0001)
    record.update(halts=1, host_s=time.perf_counter() - started, fpga_cycles=engine.read_latency_cycles())
    record["fpga_s"] = record["fpga_cycles"] * clock_ns / 1e9
    outputs, inputs = torch.empty_like(case["outputs"]), torch.empty_like(case["inputs"])
    for address, value in ((proof.TENSOR_BASE, outputs), (proof.INPUT_BASE, inputs)):
        if engine.dma_read(engine.c2h_device, address, value, value.numel() * 2) != value.numel() * 2:
            raise RuntimeError("Short DMA readback")
    actual = outputs[proof.GUARD:case["output_stop"]].float().reshape(SHAPE)
    record.update(guards_intact=bool((outputs[case["guard_mask"]] == proof.SENTINEL).all()),
                  input_unchanged=torch.equal(inputs.view(torch.int16), case["inputs"].view(torch.int16)),
                  finite=bool(torch.isfinite(actual).all()), output_sha256=proof.digest(actual))
    if not all(record[key] for key in ("guards_intact", "input_unchanged", "finite")):
        raise AssertionError("Native input, guard or finite-output check failed")
    return actual


def comparison(actual, reference, reference_wave, decoder):
    if tuple(actual.shape) != SHAPE or not bool(torch.isfinite(actual).all()):
        raise ValueError("Invalid native LSTM output")
    with torch.inference_mode():
        wave = decoder.model[2:](actual.T.unsqueeze(0)).flatten()[:NATIVE_SAMPLES]
    return dict(lstm_vs_official_fp32=proof.metrics(actual, reference),
                fp32_tail_waveform_vs_official=proof.metrics(wave, reference_wave),
                native_output_fp32_sha256=proof.digest(actual), tail_wave_fp32_sha256=proof.digest(wave))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--review-native", type=Path, nargs=2, metavar=("BASELINE", "IMPROVED"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cpu-core", type=int, default=11)
    parser.add_argument("--expected-version", type=lambda value: int(value, 0), default=0x40519E0A)
    parser.add_argument("--timeout", type=float, default=60.)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.cpu_core not in os.sched_getaffinity(0) or not 0 <= args.expected_version < 2**32
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("Invalid CPU core, version or timeout")
    mode = "native" if args.execute else "review_saved_native" if args.review_native else "offline_capture"
    output = args.output or HERE / (mode + ".json")
    if output.suffix != ".json":
        parser.error("Report output must use .json")
    protected = [TOKENS, args.checkpoint, *(args.review_native or [])]
    if any(output.resolve() == path.resolve() or output.exists() and path.exists()
           and output.samefile(path) for path in protected):
        parser.error("Report output must differ from all input files")
    os.sched_setaffinity(0, {args.cpu_core})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    proof.udc.UE_AXI_DATA_WIDTH_BITS = 256
    decoder, source, reference, reference_wave, provenance = references(args.checkpoint)
    report = dict(mode=mode, hostname=socket.gethostname(), cpu_core=args.cpu_core,
        torch_version=str(torch.__version__), source_sha256=proof.source_hashes(), provenance=provenance,
        scope="Two-layer decoder ResLSTM including skip, zero initial hidden/cell state. Frozen FPGA-selected "
              "token IDs feed official vq2emb and first convolution. Native kernels receive the same BF16 input; "
              "reference ResLSTM receives the original FP32 input. Both native outputs feed the unchanged "
              "official FP32 decoder tail. No encoder or token selection rerun.",
        metric="Relative L2 = ||actual-reference||2 / ||reference||2. Waveform uses the first 63,295 native "
               "16 kHz samples, without resampling, alignment, gain or polarity fitting. Isolated LSTM "
               "timings exclude all other operators and host transfers; they are not whole-model RTF.",
        variants=[], passed=False)

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        if args.review_native:
            for name, path in zip(("baseline", "fused_compensated_cell_tanh"), args.review_native):
                saved = torch.load(path, map_location="cpu", weights_only=True)
                actual, record = saved["actual"].float(), saved["record"]
                if (record["input_sha256"] != INPUT_SHA256 or record["starts"] != 1 or record["halts"] != 1
                        or not all(record[key] for key in ("hardware_executed", "finite", "guards_intact", "input_unchanged"))
                        or record["output_sha256"] != proof.digest(actual)):
                    raise ValueError("Saved native provenance or output hash mismatch")
                report["variants"].append(dict(variant=name, saved_tensor=str(path),
                    saved_tensor_sha256=sha256_file(path), native_record=record,
                    comparison=comparison(actual, reference, reference_wave, decoder)))
        else:
            cases = [capture(decoder, source, improved) for improved in (False, True)]
            report["variants"] = [case["record"] for case in cases]
            save()
            if args.execute:
                if socket.gethostname().split(".")[0].lower() != "italy":
                    raise RuntimeError("Native execution is restricted to Italy")
                from bigcodec_precompiled import StreamingEngine
                from yolov5_common import configure_hardware_runtime
                native_outputs = []
                with open("/tmp/pcie_ci_hw_italy.lock", "r") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    clock_ns, info, _ = configure_hardware_runtime(device="rk", dev="xdma0", cycle_override_ns=None)
                    if info.axi_data_width_bits != 256:
                        raise RuntimeError("AXI256 hardware required")
                    required_end = max(proof.check_ranges(case, info.dram_size_gb) for case in cases)
                    report.update(clock_period_ns=clock_ns, detected_dram_gib=info.dram_size_gb,
                                  required_dram_end=required_end, source_sha256=proof.source_hashes())
                    with StreamingEngine(clock_period_ns=clock_ns,
                            conv_geometry_mode=proof.udc.CONV_GEOMETRY_QUEUE_CONFIG) as engine:
                        for case in cases:
                            native_outputs.append(execute(engine, clock_ns, case, args.timeout, args.expected_version))
                            save()
                # CPU tail comparisons do not hold the hardware lock.
                for case, actual in zip(cases, native_outputs):
                    case["record"]["comparison"] = comparison(actual, reference, reference_wave, decoder)
                    save()
        if mode != "offline_capture":
            baseline, improved = (row["comparison"] for row in report["variants"])
            report["error_reduction_fraction"] = {
                key: 1 - improved[key]["relative_l2"] / baseline[key]["relative_l2"]
                for key in ("lstm_vs_official_fp32", "fp32_tail_waveform_vs_official")}
            if any(value <= 0 for value in report["error_reduction_fraction"].values()):
                raise AssertionError("Improved kernel must reduce both isolated comparison errors")
        report["passed"] = True
    except BaseException as error:
        report.update(passed=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
    print(json.dumps(dict(mode=mode, passed=report["passed"], report=str(output),
                         error_reduction_fraction=report.get("error_reduction_fraction"))), flush=True)


if __name__ == "__main__":
    main()
