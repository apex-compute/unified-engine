"""16 kHz runner reporting and device-boundary contracts without FPGA access."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dpdfnet_precompiled as dp
import dpdfnet_run_from_bin as runner
from test_dpdfnet_runtime import FakeEngine


class RunnerContractTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.bin = self.folder / "model.bin"
        self.bin.write_bytes(b"validated model fixture")
        self.input = self.folder / "frames.npy"
        self.output = self.folder / "enhanced.npy"
        self.report = self.folder / "enhanced.metrics.json"
        self.frames = np.zeros((3, 1, 1, 161, 2), dtype=np.float32)
        np.save(self.input, self.frames)
        spec = dp.make_layout("spec", (1, 1, 161, 2), dp.INPUT_BASE)
        output = dp.make_layout("spec_e", spec.shape, dp.TENSOR_BASE)
        self.expected = torch.linspace(-2, 2, 322).reshape(spec.shape).bfloat16()
        packed = dp.pack_tensor(self.expected, output)
        program = ((int(dp.udc.UE_MODE.BF16_DOT_PRODUCT) << 172).to_bytes(32, "little")
                   + (dp.udc.INSTRUCTION_HALT << 8).to_bytes(32, "little"))
        self.payload = {
            "onnx_sha256": "a" * 64,
            "hardware": {
                "model_base": dp.MODEL_BASE,
                "model_image": torch.frombuffer(bytearray(program), dtype=torch.uint8).clone(),
                "program_offset": 0,
                "program_size": len(program),
                "program_address": dp.MODEL_BASE + 128,
                "tensors": {"spec": spec.manifest(), "spec_e": output.manifest()},
            },
        }
        self.events = []
        self.now = 0.0
        events, owner = self.events, self

        class Engine(FakeEngine):
            def __init__(self, **kwargs):
                events.append("engine-created")
                super().__init__([packed] * len(owner.frames))
                self.frame = -1
                self.closed = False

            def software_reset(self, **kwargs):
                events.append("reset")

            def dma_write(self, device, address, value, size):
                events.append("model-upload" if address == dp.MODEL_BASE else "input-write")
                return super().dma_write(device, address, value, size)

            def start_execute_from_dram(self, address):
                events.append("START")
                self.frame += 1
                owner.now += [0.004, 0.007, 0.014][self.frame]
                return super().start_execute_from_dram(address)

            def read_reg32(self, address):
                events.append("HALT")
                return super().read_reg32(address)

            def read_latency_cycles(self):
                return [1_000_000, 2_000_000, 4_000_000][self.frame]

            def dma_read(self, *args):
                events.append("output-read")
                return super().dma_read(*args)

            def get_hardware_version(self):
                return 0xDF0749DE

            def close(self):
                self.closed = True
                events.append("close")

        self.Engine = Engine

    def invoke(self, *, live_width=256, extra=(), configure_error=None,
               engine_type=None, input_path=None, output_path=None):
        args = ["runner", "--bin", str(self.bin),
                "--input", str(input_path or self.input),
                "--output", str(output_path or self.output), *extra]

        def load(path):
            self.events.append("artifact-validated")
            return self.payload

        def configure(**kwargs):
            self.events.append("hardware-validated")
            if configure_error:
                raise configure_error
            return 3.0, SimpleNamespace(axi_data_width_bits=live_width, raw=0x80214D40), 3.0

        validate_target = runner.validate_runtime_hardware

        def target(payload, width):
            result = validate_target(payload, width)
            self.events.append("target-validated")
            return result

        stdout = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(sys, "argv", args))
            stack.enter_context(mock.patch.object(runner, "load_artifact", side_effect=load))
            stack.enter_context(mock.patch.object(runner, "configure_hardware_runtime", side_effect=configure))
            stack.enter_context(mock.patch.object(runner, "validate_runtime_hardware", side_effect=target))
            stack.enter_context(mock.patch.object(runner, "StreamingEngine", engine_type or self.Engine))
            stack.enter_context(mock.patch.object(dp, "validate_hardware",
                                                  side_effect=lambda payload: self.events.append("backend-validated")))
            stack.enter_context(mock.patch.object(runner.time, "perf_counter", side_effect=lambda: self.now))
            stack.enter_context(mock.patch.object(runner.os, "sched_getaffinity", return_value={6}))
            stack.enter_context(contextlib.redirect_stdout(stdout))
            runner.main()
        return json.loads(next(line.split(":", 1)[1] for line in stdout.getvalue().splitlines()
                               if line.startswith("TEST_RESULT:")))

    def test_real_backend_preserves_single_bin_start_halt_contract_and_report_hashes(self):
        result = self.invoke()
        self.assertEqual(self.events, [
            "artifact-validated", "hardware-validated", "target-validated",
            "engine-created", "reset", "backend-validated", "model-upload",
            *[event for _ in self.frames for event in ("input-write", "START", "HALT", "output-read")],
            "close"])
        np.testing.assert_array_equal(np.load(self.output),
                                      np.stack([self.expected.float().numpy()] * 3))
        metrics = json.loads(self.report.read_text())
        np.testing.assert_allclose(metrics["host_frame_ms"], [4, 7, 14])
        np.testing.assert_allclose(metrics["fpga_frame_ms"], [3, 6, 12])
        self.assertEqual(result["host_frame_deadline_misses"], 1)
        self.assertEqual(result["fpga_frame_deadline_misses"], 1)
        self.assertAlmostEqual(result["host_frame_ms_p95"], 13.3)
        self.assertAlmostEqual(result["fpga_frame_ms_p99"], 11.88)
        self.assertAlmostEqual(result["host_neural_rtf"], 0.025 / 0.030)
        for key, path in (("bin_sha256", self.bin), ("input_sha256", self.input),
                          ("output_sha256", self.output), ("metrics_sha256", self.report)):
            self.assertEqual(result[key], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(result["onnx_sha256"], "a" * 64)
        self.assertEqual(result["hardware_version"], "0xdf0749de")
        self.assertEqual(result["compiled_target_source"], "legacy-v2-compiler-contract")
        self.assertEqual(result["compiled_axi_data_width_bits"], 256)
        self.assertEqual(result["host_cpu_affinity"], [6])
        self.assertEqual(result["frame_hop_ms"], 10)
        self.assertEqual(result["model_upload_writes"], 1)
        self.assertEqual(result["dense_convolution_precision"], "BF16")
        self.assertEqual(result["precision_source"], "program-instructions")
        for key in ("input_upload_writes", "program_kicks", "halts", "output_reads"):
            self.assertEqual(result[key], 3)

    def test_incompatible_live_or_explicit_target_fails_before_reset_and_upload(self):
        for live, declared in ((512, None), (256, 512), (512, 512)):
            with self.subTest(live=live, declared=declared):
                self.events.clear()
                self.payload["hardware"].pop("axi_data_width_bits", None)
                if declared is not None:
                    self.payload["hardware"]["axi_data_width_bits"] = declared
                with self.assertRaisesRegex(RuntimeError, "AXI-256"):
                    self.invoke(live_width=live)
                self.assertEqual(self.events, ["artifact-validated", "hardware-validated"])
        self.payload["hardware"]["axi_data_width_bits"] = 256
        self.assertEqual(runner.validate_runtime_hardware(self.payload, 256),
                         (256, "artifact-metadata"))

    def test_device_is_closed_after_execution_failure_without_writing_results(self):
        class FailedEngine(self.Engine):
            def dma_read(self, *args):
                raise OSError("read failed")

        with self.assertRaisesRegex(OSError, "read failed"):
            self.invoke(engine_type=FailedEngine)
        self.assertEqual(self.events[-1], "close")
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())

    def test_cpu_affinity_is_applied_only_to_an_available_core(self):
        with mock.patch.object(runner.os, "sched_setaffinity") as pin:
            self.invoke(extra=["--cpu-core", "6"])
        pin.assert_called_once_with(0, {6})
        self.events.clear()
        with mock.patch.object(runner.os, "sched_setaffinity") as pin, \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.invoke(extra=["--cpu-core", "7"])
        pin.assert_not_called()
        self.assertEqual(self.events, [])

    def test_report_and_output_cannot_overwrite_each_other_or_input_bin(self):
        before = self.bin.read_bytes(), self.input.read_bytes()
        cases = [(["--report", str(path)], None) for path in (self.bin, self.input, self.output)]
        cases.append(([], self.bin))
        for extra, output in cases:
            with self.subTest(extra=extra, output=output), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.invoke(extra=extra, output_path=output)
                self.assertEqual(self.events, [])
        self.assertEqual(before, (self.bin.read_bytes(), self.input.read_bytes()))

    def test_input_or_bin_mutation_invalidates_run(self):
        owner = self

        class ChangedEngine(self.Engine):
            def close(self):
                super().close()
                owner.bin.write_bytes(b"changed artifact")

        with self.assertRaisesRegex(RuntimeError, "changed during execution"):
            self.invoke(engine_type=ChangedEngine)
        self.assertFalse(self.report.exists())

    def test_audio_timing_and_float_wav_preserve_sample_rate_and_sample_count(self):
        import dpdfnet_audio
        import soundfile as sf

        input_path = self.folder / "noisy.wav"
        output_path = self.folder / "enhanced.wav"
        sf.write(input_path, np.zeros(480), 48000, subtype="FLOAT")
        audio = SimpleNamespace(frames=self.frames, source_samples=480,
                                source_sample_rate=48000)
        waveform = np.linspace(-0.25, 0.25, 480, dtype=np.float32)

        def read(path):
            self.now += 0.001
            return audio

        def synthesize(output, original):
            self.assertIs(original, audio)
            self.assertEqual(output.shape, self.frames.shape)
            self.now += 0.002
            return waveform

        with mock.patch.object(dpdfnet_audio, "read_audio", side_effect=read), \
                mock.patch.object(dpdfnet_audio, "synthesize_audio", side_effect=synthesize):
            result = self.invoke(input_path=input_path, output_path=output_path)
        actual, sample_rate = sf.read(output_path, dtype="float32")
        np.testing.assert_array_equal(actual, waveform)
        self.assertEqual(sample_rate, 48000)
        self.assertEqual(result["model_sample_rate"], 16000)
        self.assertEqual(result["output_samples"], 480)
        self.assertAlmostEqual(result["steady_audio_processing_s"], 0.028)
        self.assertAlmostEqual(result["steady_audio_rtf"], 2.8)

    def test_spectrum_loader_rejects_complex_and_float32_overflow(self):
        for value, error in ((np.ones((1, 1, 161, 2), dtype=np.complex64), TypeError),
                             (np.full((1, 1, 161, 2), 1e100), ValueError)):
            np.save(self.input, value)
            with self.assertRaises(error):
                runner._load_frames(self.input)


if __name__ == "__main__":
    unittest.main()
