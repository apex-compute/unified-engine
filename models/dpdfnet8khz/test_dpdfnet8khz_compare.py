"""Regression tests for accuracy-gated real-time smoke reports."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
from dpdfnet8khz_compare import compare, timing_statistics
from dpdfnet8khz_common import load_config


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.manifest_path = self.directory / "reference.json"
        self.manifest = {
            "model": "dpdfnet2_8khz", "model_sample_rate": 8000,
            "onnx_sha256": load_config()["onnx_sha256"], "cases": []}

    def add_case(self, case_id, *, frames=3, cpu_value=0.1, fpga_value=0.1,
                 host=None, **overrides):
        shape = (frames, 1, 1, 81, 2)
        input_path = self.directory / f"{case_id}_input.npy"
        cpu_path = self.directory / f"{case_id}_cpu.npy"
        fpga_path = self.directory / f"{case_id}_fpga.npy"
        for path, value in ((input_path, 0.3), (cpu_path, cpu_value), (fpga_path, fpga_value)):
            np.save(path, np.full(shape, value, dtype=np.float32))
        self.manifest["cases"].append({
            "id": case_id, "frames": frames, "input": input_path.name, "cpu": cpu_path.name,
            "input_sha256": digest(input_path), "cpu_sha256": digest(cpu_path),
            "finite_spec_and_state": True, "state_reset": True,
            "neural_inference_s": frames * 0.0006})
        self.manifest_path.write_text(json.dumps(self.manifest))
        metrics = {
            "model": "dpdfnet2_8khz", "frames": frames,
            "onnx_sha256": self.manifest["onnx_sha256"],
            "input_sha256": digest(input_path), "output_sha256": digest(fpga_path),
            "bin_sha256": "f" * 64, "axi_data_width_bits": 256,
            "hardware_version": "0xb97c477a", "cycle_override_ns": None,
            "detected_clock_ns": 3.0, "effective_clock_ns": 3.0,
            "trace_enabled": False, "backend": "hardware", "full_graph": True,
            "state_resident": True, "cpu_neural_ops": 0,
            "program_kicks": frames, "halts": frames, "input_upload_writes": frames,
            "output_reads": frames, "intermediate_upload_writes": 0,
            "intermediate_output_reads": 0, "host_frame_ms": host or [9.0] * frames,
            "fpga_frame_ms": [8.0] * frames,
        }
        metrics.update(overrides)
        self.write_metrics(case_id, metrics)
        return metrics

    def write_metrics(self, case_id, metrics):
        path = self.directory / f"{case_id}_fpga.metrics.json"
        path.write_text(json.dumps(metrics))
        summary = {key: value for key, value in metrics.items()
                   if key not in ("host_frame_ms", "fpga_frame_ms")}
        summary.update(metrics_file=str(path), metrics_sha256=digest(path))
        (self.directory / f"{case_id}_fpga.log").write_text(
            "initialization\nTEST_RESULT:" + json.dumps(summary) + "\n")

    def report(self):
        return compare(self.manifest_path, self.directory)

    def test_good_corpus_recomputes_weighted_tails_and_includes_silence(self):
        self.add_case("speech", host=[8.0, 9.0, 10.0])
        self.add_case("silence", frames=1, cpu_value=0.0, fpga_value=0.0, host=[7.0])
        result = self.report()
        self.assertTrue(result["realtime_smoke_pass"])
        self.assertFalse(result["speech_quality_qualified"])
        self.assertEqual(result["host_timing"]["frames"], 4)
        self.assertEqual(result["host_timing"]["mean_ms"], 8.5)
        self.assertAlmostEqual(result["host_timing"]["p95_ms"], 9.85)
        self.assertAlmostEqual(result["host_timing"]["p99_ms"], 9.97)
        self.assertEqual(result["host_timing"]["deadline_misses"], 0)
        self.assertIsNone(result["cases"][1]["accuracy"]["relative_l2"])

    def test_fast_but_grossly_wrong_output_cannot_pass(self):
        self.add_case("wrong", fpga_value=1e38, host=[1.0, 1.0, 1.0])
        result = self.report()
        self.assertFalse(result["realtime_smoke_pass"])
        self.assertTrue(result["cases"][0]["timing"]["observed_deadline_pass"])
        self.assertFalse(result["cases"][0]["accuracy"]["passed"])
        self.assertGreater(result["cases"][0]["accuracy"]["relative_l2"], 1e38)
        json.dumps(result, allow_nan=False)

    def test_nonfinite_and_zero_reference_noise_fail_accuracy(self):
        self.add_case("nan", fpga_value=float("nan"))
        self.add_case("silence", cpu_value=0.0, fpga_value=0.01)
        result = self.report()
        self.assertFalse(result["realtime_smoke_pass"])
        self.assertTrue(all(not case["accuracy"]["passed"] for case in result["cases"]))
        json.dumps(result, allow_nan=False)

    def test_one_deadline_miss_fails_even_with_fast_mean(self):
        self.add_case("jitter", host=[1.0, 1.0, 10.01])
        result = self.report()
        self.assertFalse(result["realtime_smoke_pass"])
        self.assertLess(result["host_timing"]["mean_ms"], 10.0)
        self.assertEqual(result["host_timing"]["deadline_misses"], 1)

    def test_hash_tampering_and_missing_cases_fail_closed(self):
        self.add_case("changed")
        self.add_case("missing")
        (self.directory / "changed_cpu.npy").write_bytes(b"tampered")
        (self.directory / "missing_fpga.npy").unlink()
        result = self.report()
        self.assertFalse(result["realtime_smoke_pass"])
        self.assertEqual(result["cases_total"], 2)
        self.assertTrue(all(case["validation_errors"] for case in result["cases"]))

    def test_metrics_tampering_or_missing_raw_frames_rejected(self):
        self.add_case("changed")
        self.add_case("missing_raw", host_frame_ms=[1.0])
        (self.directory / "changed_fpga.metrics.json").write_text("{}")
        result = self.report()
        self.assertFalse(result["realtime_smoke_pass"])
        self.assertIn("SHA256 mismatch", result["cases"][0]["validation_errors"][0])
        self.assertIn("every compared frame", result["cases"][1]["validation_errors"][0])

    def test_wrong_target_trace_clock_and_host_fallback_are_excluded(self):
        for case_id, changes in (("axi", {"axi_data_width_bits": 512}),
                                 ("trace", {"trace_enabled": True}),
                                 ("clock", {"cycle_override_ns": 3.0}),
                                 ("effective_clock", {"effective_clock_ns": 2.0}),
                                 ("host_ops", {"cpu_neural_ops": 1})):
            self.add_case(case_id, **changes)
        result = self.report()
        self.assertFalse(result["realtime_smoke_pass"])
        self.assertIsNone(result["host_timing"])
        self.assertTrue(all(case["timing"]["qualification_exclusions"] for case in result["cases"]))

    def test_mixed_bin_revisions_do_not_qualify_one_deployment(self):
        self.add_case("old")
        self.add_case("new", bin_sha256="e" * 64)
        result = self.report()
        self.assertEqual(result["cases_smoke_pass"], 2)
        self.assertFalse(result["same_deployment"])
        self.assertFalse(result["realtime_smoke_pass"])

    def test_invalid_timings_and_relaxed_deadline_rejected(self):
        for values in ([], [0.0], [-1.0], [float("inf")], [[1.0]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                timing_statistics(values, 10.0)
        self.add_case("case")
        with self.assertRaisesRegex(ValueError, "10 ms hop"):
            compare(self.manifest_path, self.directory, deadline_ms=20.0)


if __name__ == "__main__":
    unittest.main()
