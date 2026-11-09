"""Rebuild both artifact strategies from cached official checkpoints, without an FPGA."""
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parents[1]):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from yolov5_artifact import (load_single_bin, SUPPORTED_INPUT_RESOLUTIONS,
                             select_single_bin_profile, validate_single_bin)
from yolov5s_compile import build_artifact


@unittest.skipUnless(os.environ.get('ANDROMEDA_YOLO_CHECKPOINT_DIR'),
                     'set ANDROMEDA_YOLO_CHECKPOINT_DIR to cached official checkpoints')
class CanonicalCompileTests(unittest.TestCase):
    def check_variant(self, variant):
        root = Path(os.environ['ANDROMEDA_YOLO_CHECKPOINT_DIR'])
        checkpoint = root / f'yolov5{variant}-v7.0.pt'
        self.assertTrue(checkpoint.is_file(), f'Missing cached checkpoint: {checkpoint}')
        with tempfile.TemporaryDirectory() as temporary:
            for winograd in (False, True):
                with self.subTest(variant=variant, winograd=winograd):
                    output = Path(temporary) / f'{winograd}.bin'
                    with contextlib.redirect_stdout(io.StringIO()):
                        build_artifact(variant=variant, checkpoint=checkpoint,
                                       output=output, winograd=winograd)
                    payload = load_single_bin(output)
                    self.assertEqual(payload['artifact_version'], 8 if winograd else 7)
                    for resolution in SUPPORTED_INPUT_RESOLUTIONS:
                        _, selected = select_single_bin_profile(payload, resolution)
                        self.assertEqual('winograd_layers' in selected['hardware'], winograd)
                    if winograd:
                        with contextlib.redirect_stdout(io.StringIO()):
                            self.assertEqual(build_artifact(variant=variant, output=output), output)
                        with self.assertRaises(RuntimeError):
                            validate_single_bin(dict(payload, artifact_version=7))
                        with contextlib.redirect_stdout(io.StringIO()):
                            with self.assertRaisesRegex(RuntimeError, 'different convolution strategy'):
                                build_artifact(variant=variant, output=output, winograd=False)

    def test_yolov5s_both_strategies(self):
        self.check_variant('s')

    def test_yolov5n_both_strategies(self):
        self.check_variant('n')


if __name__ == '__main__':
    unittest.main()
