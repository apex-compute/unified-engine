"""Exercise CI bin preparation with cold, native-only and optimized caches."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import user_dma_core
import user_hw_test as hardware_tests


class ConvModelBinTests(unittest.TestCase):
    def check_cache(self, cache):
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            os.chdir(directory)
            try:
                self.run_model_suite(cache)
            finally:
                os.chdir(previous)

    def run_model_suite(self, cache):
        native_bins, optimized_bins = {}, {}
        for name in ('yolov5s', 'yolov5n'):
            bins = Path('models') / name / f'{name}_bin'
            bins.mkdir(parents=True)
            native_bins[name] = bins / f'{name}-andromeda.bin'
            optimized_bins[name] = bins / f'{name}-andromeda_winograd.bin'
            if cache in ('native', 'optimized'):
                native_bins[name].write_bytes(b'native cache')
            if cache == 'optimized':
                optimized_bins[name].write_bytes(b'optimized cache')
        unet = Path('models/unet/unet_bin/unet-andromeda.bin')
        unet.parent.mkdir(parents=True)
        unet.write_bytes(b'unet cache')
        compiled, executed = [], []

        def run(args, label):
            script = Path(args[0])
            name = script.parent.name
            if name == 'unet':
                return dict(mask_agreement=1.0, logit_rmse=0.0)
            self.assertIn(name, optimized_bins)
            if script.name.endswith('_compile.py'):
                self.assertIn('--optimize', args)
                destination = Path(args[args.index('--output') + 1])
                self.assertEqual(destination, optimized_bins[name])
                destination.write_bytes(b'compiled optimized bin')
                compiled.append(name)
                return {}
            self.assertIn('--bin', args)
            artifact = Path(args[args.index('--bin') + 1])
            self.assertEqual(artifact, optimized_bins[name])
            self.assertTrue(artifact.is_file(), f'{label} ran before its bin was compiled')
            executed.append((name, '--cpu' in args))
            return dict(detections=[dict(class_id=0, label='person', confidence=0.9,
                                         box_xyxy=[0, 0, 10, 10])])

        with mock.patch.object(hardware_tests, '_conv_model_run', side_effect=run), \
                mock.patch.object(hardware_tests, 'record_test'), \
                mock.patch.object(user_dma_core, '_current_device_name', return_value='xdma0'), \
                mock.patch.object(user_dma_core, 'configured_hardware_info',
                                  return_value=SimpleNamespace(axi_data_width_bits=512)), \
                contextlib.redirect_stdout(io.StringIO()):
            hardware_tests.conv_model_tests()
        self.assertEqual(compiled, [] if cache == 'optimized' else ['yolov5s', 'yolov5n'])
        self.assertEqual(executed, [('yolov5s', False), ('yolov5s', True),
                                    ('yolov5n', False), ('yolov5n', True)])
        if cache in ('native', 'optimized'):
            for path in native_bins.values():
                self.assertEqual(path.read_bytes(), b'native cache')

    def test_cold_cache_builds_before_execution(self):
        self.check_cache('cold')

    def test_native_cache_does_not_skip_optimized_compilation(self):
        self.check_cache('native')

    def test_optimized_cache_is_reused(self):
        self.check_cache('optimized')

    def test_compiler_must_produce_requested_artifact(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(hardware_tests, '_conv_model_run', return_value={}):
            artifact = str(Path(directory) / 'missing.bin')
            with self.assertRaisesRegex(FileNotFoundError, 'compiler did not produce'):
                hardware_tests._conv_model_bin(artifact, [['compile.py']], 'model')


if __name__ == '__main__':
    unittest.main()
