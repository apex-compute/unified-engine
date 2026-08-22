#!/usr/bin/env python3
"""Host regressions for the dedicated YOLOv5n entrypoints."""

import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parent / "yolov5"
REPO_ROOT = SCRIPT_DIR.parents[1]
for path in (SCRIPT_DIR, SHARED_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import yolov5n_compile
import yolov5n_run_from_bin
import yolov5n_test
import model_auto_test
from yolov5_common import load_yolov5_config


class NanoDirectoryTests(unittest.TestCase):
    def test_config_and_assets_are_owned_by_nano_directory(self):
        profile, config, resource_dir = load_yolov5_config("n")
        self.assertEqual(profile.key, "n")
        self.assertEqual(resource_dir, SCRIPT_DIR)
        self.assertEqual(config["paths"]["bin_dir"], "yolov5n_bin")
        self.assertEqual(
            resource_dir / config["paths"]["weights"],
            SCRIPT_DIR / "yolov5n_bin/yolov5n-v7.0.pt")
        self.assertEqual(
            resource_dir / config["paths"]["artifact"],
            SCRIPT_DIR / "yolov5n_bin/yolov5n-andromeda.bin")

    def test_wrappers_pin_nano_and_forward_argv(self):
        cases = (
            (yolov5n_compile, ["--force"]),
            (yolov5n_test, ["--backend", "cpu"]),
            (yolov5n_run_from_bin, ["--cpu"]),
        )
        for module, argv in cases:
            with self.subTest(module=module.__name__):
                with mock.patch.object(module, "_shared_main") as shared:
                    module.main(argv)
                shared.assert_called_once_with(
                    argv, pinned_variant="n", config_path=module.CONFIG_PATH)

    def test_harness_forwards_hardware_profile(self):
        for module in (yolov5n_test, yolov5n_run_from_bin):
            with self.subTest(module=module.__name__):
                self.assertTrue(model_auto_test._script_supports_flag(
                    str(Path(module.__file__)), "--dev"))
                self.assertTrue(model_auto_test._script_supports_flag(
                    str(Path(module.__file__)), "--device"))

    def test_dedicated_clis_cannot_switch_to_small(self):
        for module in (
                yolov5n_compile, yolov5n_test, yolov5n_run_from_bin):
            with self.subTest(module=module.__name__):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        module.main(["--variant", "s"])


if __name__ == "__main__":
    unittest.main()
