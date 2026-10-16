"""Paired CPU comparison checks using independently constructed WAV fixtures."""

import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf


sys.path.insert(0, str(Path(__file__).resolve().parent))
from dpdfnet_compare_cpu import compare_results, write_comparison


class DPDFNetCompareCPUTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.report = {"sample_rate": 16000, "complete": True, "cases": []}

    def case(self, identifier, cpu, fpga, *, scores=((10, 0.8, 2), (8, 0.85, 2.2)),
             frames=10, cpu_seconds=0.1, fpga_seconds=1, host_seconds=1.1, control=False):
        case = {"id": identifier, "noise": "bus", "speaker": "speaker1", "snr_db": 2.5,
                "samples": len(cpu), "outputs": {}}
        if control:
            case["group"] = "clean_control"
        folder = self.directory / identifier
        folder.mkdir()
        for index, (backend, values) in enumerate((("cpu", cpu), ("fpga", fpga))):
            path = folder / (backend + ".wav")
            sf.write(path, np.asarray(values, dtype=np.float32), 16000, subtype="FLOAT")
            runtime = {"frames": frames}
            if backend == "cpu":
                runtime["neural_inference_s"] = cpu_seconds
            else:
                runtime.update(fpga_execution_s_sum=fpga_seconds, execution_elapsed_s=host_seconds)
            case["outputs"][backend] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "runtime": runtime,
                "metrics": dict(zip(("si_sdr_db", "stoi", "pesq_wb"), scores[index])),
            }
        self.report["cases"].append(case)
        return case

    def run_comparison(self):
        self.report["expected_cases"] = len(self.report["cases"])
        (self.directory / "results.json").write_text(json.dumps(self.report))
        return compare_results(self.directory)

    def test_paired_quality_pooled_waveform_and_weighted_timing_exclude_controls(self):
        self.case("first", [1, 0], [0.5, 0.5])
        self.case("second", [2, 0, 0, 0], [1, 0, 0, 0],
                  scores=((20, 0.7, 3), (21, 0.69, 2.5)), frames=30,
                  cpu_seconds=0.6, fpga_seconds=6, host_seconds=6.6)
        self.case("control", [1, 1], [100, 100], control=True,
                  scores=((1000, 1000, 1000), (2000, 2000, 2000)), cpu_seconds=500)
        result = self.run_comparison()
        self.assertEqual((result["cases_total"], result["noisy_cases"], result["clean_controls_excluded"]),
                         (3, 2, 1))
        metric = result["summary"]["quality"]["si_sdr_db"]
        self.assertEqual((metric["paired_cases"], metric["cpu_mean"], metric["fpga_mean"]), (2, 15, 14.5))
        self.assertEqual((metric["mean_delta"], metric["median_delta"]), (-0.5, -0.5))
        self.assertEqual((metric["fpga_worse_cases"], metric["fpga_better_cases"]), (1, 1))
        waveform = result["summary"]["waveform"]
        self.assertEqual((waveform["samples"], waveform["sum_squared_error"], waveform["cpu_energy"]),
                         (6, 1.5, 5))
        self.assertAlmostEqual(waveform["relative_l2"], math.sqrt(1.5 / 5))
        self.assertEqual(waveform["rmse"], 0.5)
        self.assertEqual(waveform["max_abs_error"], 1)
        timing = result["summary"]["timing"]
        self.assertEqual((timing["paired_cases"], timing["frames"]), (2, 40))
        self.assertAlmostEqual(timing["cpu_neural_ms_per_frame"], 17.5)
        self.assertAlmostEqual(timing["fpga_execution_ms_per_frame"], 175)
        self.assertAlmostEqual(timing["fpga_host_ms_per_frame"], 192.5)
        self.assertAlmostEqual(timing["fpga_execution_over_cpu"], 10)
        self.assertEqual(result["cases"][2]["waveform"]["max_abs_error"], 99)

    def test_missing_backend_and_metric_report_paired_exclusions(self):
        missing = self.case("missing", [1, 0], [1, 0])
        del missing["outputs"]["fpga"]
        partial = self.case("partial", [1, 0], [0.5, 0.5])
        partial["outputs"]["cpu"]["metrics"]["stoi"] = None
        partial["outputs"]["cpu"]["metrics"]["reasons"] = {"stoi": "speech too short"}
        partial["outputs"]["fpga"]["runtime"]["frames"] = 11
        result = self.run_comparison()
        quality = result["summary"]["quality"]
        self.assertEqual(quality["si_sdr_db"]["paired_cases"], 1)
        self.assertEqual(quality["stoi"]["paired_cases"], 0)
        self.assertEqual(quality["stoi"]["excluded_cases"], 2)
        self.assertIsNone(quality["stoi"]["mean_delta"])
        self.assertIn("speech too short", quality["stoi"]["exclusions"][1]["reason"])
        self.assertIn("fpga output missing", result["summary"]["waveform"]["exclusions"][0]["reason"])
        self.assertEqual(result["summary"]["timing"]["paired_cases"], 0)
        self.assertIsNone(result["summary"]["timing"]["cpu_neural_ms_per_frame"])
        self.assertEqual(result["summary"]["waveform"]["paired_cases"], 1)

    def test_zero_cpu_energy_has_no_invented_relative_error(self):
        self.case("silent_cpu", [0, 0], [1, -1])
        result = self.run_comparison()
        for waveform in (result["cases"][0]["waveform"], result["summary"]["waveform"]):
            self.assertIsNone(waveform["relative_l2"])
            self.assertTrue(waveform["relative_l2_reason"])
            self.assertEqual(waveform["rmse"], 1)

    def test_tampered_wav_is_rejected_before_comparison(self):
        self.case("tampered", [1, 0], [0.5, 0.5])
        path = self.directory / "tampered/fpga.wav"
        sf.write(path, np.array([0.25, 0.25], dtype=np.float32), 16000, subtype="FLOAT")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.run_comparison()

    def test_verified_wav_still_requires_matching_rate_shape_and_finite_values(self):
        for index, (values, rate, message) in enumerate((
                ([1, 0], 8000, "sample rate"), ([1], 16000, "sample count"),
                ([[1, 0], [0, 1]], 16000, "mono"), ([1, np.nan], 16000, "finite"))):
            with self.subTest(message=message):
                self.report["cases"] = []
                case = self.case(f"invalid{index}", [1, 0], [1, 0])
                path = self.directory / case["id"] / "fpga.wav"
                sf.write(path, np.asarray(values, dtype=np.float32), rate, subtype="FLOAT")
                case["outputs"]["fpga"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaisesRegex(ValueError, message):
                    self.run_comparison()

    def test_missing_sha_excludes_unverified_scores_and_csv_retains_reasons(self):
        case = self.case("unverified", [1, 0], [1, 0])
        del case["outputs"]["cpu"]["sha256"]
        result = self.run_comparison()
        self.assertEqual(result["summary"]["quality"]["pesq_wb"]["paired_cases"], 0)
        json_path, csv_path = write_comparison(result, self.directory / "cpu_comparison")
        self.assertEqual(json.loads(json_path.read_text())["cases_total"], 1)
        with csv_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertIn("SHA256 missing", rows[0]["reasons"])
        original = (self.directory / "results.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            write_comparison(result, self.directory / "results")
        self.assertEqual((self.directory / "results.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
