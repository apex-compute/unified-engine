"""Fair-subset aggregation and reporting regressions for noisy speech evaluation."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


sys.path.insert(0, str(Path(__file__).resolve().parent))
from dpdfnet_evaluate_noisy import aggregate, read_mono, write_results


def scored(si_sdr, stoi, pesq, reasons=None):
    return {"metrics": {"si_sdr_db": si_sdr, "stoi": stoi, "pesq_wb": pesq,
                        "reasons": {} if reasons is None else reasons}}


class DPDFNetNoisyEvalTest(unittest.TestCase):
    def cases(self):
        return [
            {"id": "a", "noise": "traffic", "snr_db": 0, "speaker": "one",
             "duration_s": 1, "outputs": {
                 "noisy": scored(10, 0.5, 2), "cpu": scored(15, 0.8, 3),
                 "fpga": scored(15, None, 3.5, {"stoi": "insufficient active speech"})}},
            {"id": "b", "noise": "traffic", "snr_db": 5, "speaker": "two",
             "duration_s": 1000, "outputs": {
                 "noisy": scored(20, 0.6, 3), "cpu": {"error": "runner failed"},
                 "fpga": scored(None, 0.5, None)}},
            {"id": "c", "noise": "babble", "snr_db": 10, "speaker": "one",
             "duration_s": 4, "outputs": {
                 "noisy": scored(None, None, 1), "cpu": scored(20, 0.7, None),
                 "fpga": scored(30, 0.4, 1.2)}},
        ]

    def test_each_metric_uses_its_own_valid_values_and_paired_deltas(self):
        result = aggregate(self.cases(), "fpga")
        self.assertEqual(result["si_sdr_db"]["valid_cases"], 2)
        self.assertEqual(result["si_sdr_db"]["paired_cases"], 1)
        self.assertEqual(result["si_sdr_db"]["mean"], 22.5)
        self.assertEqual(result["si_sdr_db"]["mean_improvement"], 5)
        self.assertAlmostEqual(result["stoi"]["mean"], 0.45)
        self.assertEqual(result["stoi"]["paired_cases"], 1)
        self.assertAlmostEqual(result["stoi"]["mean_improvement"], -0.1)
        self.assertEqual(result["stoi"]["improved"], 0)
        self.assertEqual(result["stoi"]["regressed"], 1)
        self.assertAlmostEqual(result["pesq_wb"]["mean_improvement"], 0.85)
        self.assertEqual(result["pesq_wb"]["improved"], 2)

    def test_failed_cases_do_not_become_zero_scores_or_zero_improvements(self):
        result = aggregate(self.cases(), "cpu")
        self.assertEqual(result["si_sdr_db"]["valid_cases"], 2)
        self.assertEqual(result["si_sdr_db"]["mean"], 17.5)
        self.assertEqual(result["si_sdr_db"]["paired_cases"], 1)
        self.assertEqual(result["si_sdr_db"]["mean_improvement"], 5)
        self.assertEqual(result["pesq_wb"]["valid_cases"], 1)
        self.assertEqual(result["pesq_wb"]["mean"], 3)
        empty = aggregate(self.cases(), "unavailable")
        for metric in ("si_sdr_db", "stoi", "pesq_wb"):
            self.assertIsNone(empty[metric]["mean"])
            self.assertIsNone(empty[metric]["mean_improvement"])
            self.assertEqual(empty[metric]["valid_cases"], 0)
            self.assertEqual(empty[metric]["paired_cases"], 0)

    def test_clean_controls_are_kept_for_listening_but_excluded_from_means(self):
        cases = self.cases()
        cases.append({"id": "control", "group": "clean_control", "noise": "clean_control",
                      "speaker": "one", "snr_db": None, "outputs": {
                          "noisy": scored(None, 1, 4.64),
                          "cpu": scored(1000, 1, 4.64), "fpga": scored(1000, 1, 4.64)}})
        report = {"cases": cases, "backends": ["cpu", "fpga"]}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_results(directory, report)
            saved = json.loads((directory / "results.json").read_text())
            self.assertEqual(saved["summary"]["fpga"]["si_sdr_db"]["mean"], 22.5)
            self.assertNotIn("clean_control", saved["by_noise"])
            self.assertEqual(len(saved["cases"]), 4)
            self.assertEqual(saved["cases"][0]["outputs"]["fpga"]["metrics"]["reasons"],
                             {"stoi": "insufficient active speech"})
            self.assertEqual(saved["cases"][1]["outputs"]["cpu"]["error"], "runner failed")
            self.assertIn("control/clean.wav", (directory / "report.md").read_text())
            with (directory / "results.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(sum(row["id"] == "control" for row in rows), 3)
            failed = next(row for row in rows if row["id"] == "b" and row["backend"] == "cpu")
            self.assertEqual(failed["error"], "runner failed")
            self.assertEqual(failed["si_sdr_db"], "")

    def test_mono_preparation_preserves_amplitude_and_sample_positions(self):
        waveform = np.array([[2.0, 0.0], [-1.0, 0.5], [0.0, -2.0], [0.0, 0.0]],
                            dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.wav"
            sf.write(path, waveform, 16000, subtype="FLOAT")
            mono = read_mono(path)
        np.testing.assert_array_equal(mono, np.array([1.0, -0.25, -1.0, 0.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
