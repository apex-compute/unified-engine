#!/usr/bin/env python3
"""Audit actual DMA and START/HALT calls for both resident DPDFNet models.

This diagnostic resets the selected FPGA and runs two frames per model.
Acquire the normal hardware lock before invoking it. It is not a benchmark.
Audio analysis/synthesis are host operations; the audited boundary is the
complete neural graph, whose input and output are BF16 spectra in DRAM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NATIVE8 = ROOT / "models" / "dpdfnet8khz"
for path in (ROOT, HERE, NATIVE8, ROOT / "models" / "yolov5s"):
    sys.path.insert(0, str(path))

import user_dma_core as udc
import dpdfnet_precompiled as model16
import dpdfnet8khz_precompiled as model8
from dpdfnet_audio import read_audio as audio16
from dpdfnet8khz_audio import read_audio as audio8
from dpdfnet8khz_common import sha256
from dpdfnet8khz_engine import StreamingEngine
from yolov5_common import configure_hardware_runtime


class ObservedEngine(StreamingEngine):
    def __init__(self, **kwargs):
        self.events = []
        self.phase = "setup"
        super().__init__(**kwargs)

    def dma_write(self, device, address, buffer, size):
        actual = super().dma_write(device, address, buffer, size)
        event = {"phase": self.phase, "kind": "dram_write",
                 "address": hex(address), "bytes": size, "transferred": actual}
        if self.phase == "model_upload":
            event["sha256"] = hashlib.sha256(self._write_bytes(buffer, size)).hexdigest()
        self.events.append(event)
        return actual

    def dma_read(self, device, address, buffer, size):
        actual = super().dma_read(device, address, buffer, size)
        self.events.append({"phase": self.phase, "kind": "dram_read",
                            "address": hex(address), "bytes": size, "transferred": actual})
        return actual

    def start_execute_from_dram(self, address):
        super().start_execute_from_dram(address)
        self.events.append({"phase": self.phase, "kind": "start",
                            "program_address": hex(address)})


def audit(module, read_audio, bin_path, input_path, clock, info):
    bin_digest = sha256(bin_path)
    payload = module.load_artifact(bin_path)
    hardware = payload["hardware"]
    if hardware.get("axi_data_width_bits", 256) != info.axi_data_width_bits:
        raise RuntimeError("artifact transport does not match the live FPGA")
    frames = read_audio(input_path).frames[:2]
    input_layout = module.TensorLayout.from_manifest(hardware["tensors"]["spec"])
    output_layout = module.TensorLayout.from_manifest(hardware["tensors"]["spec_e"])

    class ObservedBackend(module.WholeGraphBackend):
        def _wait(self):
            super()._wait()
            cause = self.ue.read_reg32(udc.UE_INT_REG) & 3
            busy = self.ue.is_queue_busy()
            if cause != udc.INT_CAUSE_HALT or busy:
                raise RuntimeError("output readback was reached without HALT and queue idle")
            self.ue.events.append({"phase": self.ue.phase, "kind": "halt_observed",
                                   "interrupt_cause": cause, "queue_busy": busy})

    with ObservedEngine(clock_period_ns=clock,
                        conv_geometry_mode=udc.CONV_GEOMETRY_QUEUE_CONFIG) as engine:
        engine.software_reset(run_dram_self_test=False)
        engine.phase = "model_upload"
        backend = ObservedBackend(engine, payload, axi_data_width_bits=256, timeout_s=10)
        with torch.inference_mode():
            for index, frame in enumerate(frames):
                engine.phase = f"frame_{index}"
                if not torch.isfinite(backend.execute(frame)).all():
                    raise RuntimeError("nonfinite diagnostic output")
        version = f"0x{engine.get_hardware_version():08x}"
        events = engine.events
    uploads = [event for event in events if event["phase"] == "model_upload"]
    expected_upload = {"phase": "model_upload", "kind": "dram_write",
                       "address": hex(hardware["model_base"]),
                       "bytes": hardware["model_image"].numel(),
                       "transferred": hardware["model_image"].numel(),
                       "sha256": hardware["model_sha256"]}
    if uploads != [expected_upload] or any(event["phase"] == "setup" for event in events):
        raise RuntimeError("expected exactly one complete resident image upload after reset")
    for index in range(len(frames)):
        group = [event for event in events if event["phase"] == f"frame_{index}"]
        if [event["kind"] for event in group] != [
                "dram_write", "start", "halt_observed", "dram_read"]:
            raise RuntimeError("unexpected per-frame transfer/execution sequence")
        for event, layout in ((group[0], input_layout), (group[-1], output_layout)):
            if (event["address"] != hex(layout.address)
                    or event["bytes"] != layout.size_bytes
                    or event["transferred"] != layout.size_bytes):
                raise RuntimeError("frame transfer address or size mismatch")
        if group[1]["program_address"] != hex(hardware["program_address"]):
            raise RuntimeError("frame did not start the complete resident program")
    if sha256(bin_path) != bin_digest:
        raise RuntimeError("bin changed during the audit")
    rate = 8000 if payload["model"] == "dpdfnet2_8khz" else 16000
    return {"model": payload["model"], "bin": str(bin_path.resolve()),
            "bin_sha256": bin_digest, "hardware_version": version,
            "sample_rate": rate, "new_samples_per_hop": rate // 100,
            "hop_ms": 10, "input_boundary": "host STFT spectrum, padded BF16",
            "output_boundary": "enhanced spectrum, host iSTFT follows",
            "graph_nodes": hardware["graph_operations"],
            "validated_single_terminal_halt_no_swi": True,
            "initial_model_uploads": 1, "audited_frames": len(frames),
            "program_and_parameter_uploads_per_frame": 0,
            "intermediate_host_dram_transfers": 0,
            "resident_image_bytes": hardware["model_image"].numel(),
            "program_bytes": hardware["program_size"],
            "parameter_data_bytes": hardware["program_offset"],
            "state_resident": True, "passed": True, "events": events}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin8", type=Path,
                        default=NATIVE8 / "dpdfnet8khz_bin/dpdfnet2_8khz-andromeda.bin")
    parser.add_argument("--bin16", type=Path,
                        default=HERE / "dpdfnet_bin/dpdfnet2-andromeda.bin")
    args = parser.parse_args()
    if args.output.resolve() in {p.resolve() for p in (args.input, args.bin8, args.bin16)}:
        parser.error("output must differ from input and deployment bins")
    clock, info, _ = configure_hardware_runtime(device="rk", dev="xdma0", cycle_override_ns=None)
    if info.axi_data_width_bits != 256:
        parser.error("both selected deployments require RK AXI-256")
    source_digest = sha256(args.input)
    report = {"benchmark": False, "method": "observed driver API DMA/start calls and HALT/queue registers",
              "input": str(args.input.resolve()), "input_sha256": source_digest,
              "axi_data_width_bits": info.axi_data_width_bits,
              "detected_clock_ns": clock,
              "models": [audit(module, frontend, path, args.input, clock, info)
                         for module, frontend, path in (
                             (model8, audio8, args.bin8), (model16, audio16, args.bin16))]}
    if sha256(args.input) != source_digest:
        raise RuntimeError("audio input changed during the audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Both resident-bin frame contracts passed: " + str(args.output))


if __name__ == "__main__":
    main()
