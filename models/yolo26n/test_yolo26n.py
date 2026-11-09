"""Host tests; cached-checkpoint and official-source integration checks are opt-in."""
import contextlib
import copy
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from yolo26_common import (CONFIG, FloatBackend, TensorSpec, decode_heads, dense_conv,
                           execute_model, load_checkpoint, select, select_conv)
from yolo26_artifact import (CompileBackend, bundle_digest, compile_artifact,
                             execute_cpu, load_artifact, select_profile, validate_artifact)
from yolov5_common import letterbox_image
from yolo26n_compile import build_artifact
from yolov5_precompiled import _ImageBuilder, _PackedMap, MODEL_BASE, MODEL_LIMIT
import yolo_attention


class PrimitiveTests(unittest.TestCase):
    def test_grouped_convolution_matches_pytorch(self):
        torch.manual_seed(21)
        for groups, outputs in [(2, 12), (8, 8), (8, 16)]:
            with self.subTest(groups=groups, outputs=outputs):
                conv = torch.nn.Conv2d(8, outputs, 3, padding=1, stride=2, groups=groups)
                x = torch.randn(1, 8, 11, 13)
                torch.testing.assert_close(dense_conv(conv)(x), conv(x))

    def test_compiled_selection_preserves_channels_and_zero_padding(self):
        x = torch.randn(128, 3, 5).to(torch.bfloat16)
        for optimize in (False, True):
            backend = CompileBackend(optimize=optimize)
            output = select(backend, 'select', TensorSpec('input', tuple(x.shape)), 96, 32, 64)
            selected = dict(model=dict(input_shape=list(x.shape)), operations=backend.operations,
                            weights=backend.weights, head_outputs=[output.key])
            actual = execute_cpu(selected, x)[0]
            expected = torch.cat((x[96:], torch.zeros(32, 3, 5, dtype=torch.bfloat16)))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_invalid_channel_selection_rejected(self):
        for start, count, out in [(-1, 4, 4), (60, 8, 8), (0, 32, 16)]:
            with self.assertRaises(ValueError):
                select_conv(64, start, count, out)

    def test_attention_scale_and_qk_padding(self):
        torch.manual_seed(9)
        q, k, v = torch.randn(32, 3, 5), torch.randn(32, 3, 5), torch.randn(64, 3, 5)
        expected = F.scaled_dot_product_attention(q.flatten(1).T, k.flatten(1).T, v.flatten(1).T)
        padded = [torch.cat((t, torch.zeros_like(t))) for t in (q, k)]
        actual = FloatBackend().attention('attn', *padded, v, 32 ** -.5)
        torch.testing.assert_close(actual.flatten(1).T, expected)

    def test_attention_masks_padded_spatial_positions(self):
        source = _PackedMap((64, 13, 13), 64, tuple(range(64)))
        image = _ImageBuilder(MODEL_BASE, MODEL_LIMIT)
        plan = yolo_attention.prepare({'scale': 32 ** -.5}, [source] * 3, source, image)
        self.assertEqual((plan['count'], plan['aligned']), (169, 192))
        # Reconstruct the emitted mask from the model image, independently of
        # the runtime attention implementation.
        raw = torch.frombuffer(image.data, dtype=torch.uint8)
        start = plan['mask'] - MODEL_BASE
        mask = raw[start:start + 169 * 192 * 2].view(torch.bfloat16).view(169, 192)
        self.assertTrue(torch.all(mask[:, :169] == 0))
        self.assertTrue(torch.all(mask[:, 169:] < -1000))
        self.assertGreaterEqual(yolo_attention.scratch_bytes(source), 2 * (3 * 192 * 64 + 169 * 192 + 169 * 64))

    def test_checkpoint_digest_checked_before_unpickling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.pt'
            path.write_bytes(b'not a checkpoint')
            with self.assertRaisesRegex(RuntimeError, 'SHA-256'):
                load_checkpoint(path)


class DecodeTests(unittest.TestCase):
    def heads(self):
        heads = [torch.full((84, 1, 1), -100.) for _ in range(3)]
        for h in heads:
            h[:4] = torch.tensor([1., 2., 3., 4.]).view(4, 1, 1)
        return heads

    def test_anchor_distances_strides_and_score_order(self):
        heads = self.heads()
        heads[0][6] = 2
        heads[2][5] = 3
        results = decode_heads(heads, [str(i) for i in range(80)])
        self.assertEqual([d.class_id for d in results], [1, 2])
        self.assertEqual(results[0].box, (-16., -48., 112., 144.))
        self.assertEqual(results[1].box, (-4., -12., 28., 36.))
        self.assertAlmostEqual(results[0].score, torch.sigmoid(torch.tensor(3.)).item())

    def test_multiple_classes_at_same_anchor_are_not_suppressed(self):
        heads = self.heads()
        heads[0][4], heads[0][5] = 4, 3
        results = decode_heads(heads, [str(i) for i in range(80)])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].box, results[1].box)
        self.assertEqual(len(decode_heads(heads, [str(i) for i in range(80)], max_det=1)), 1)

    def test_invalid_or_nonfinite_inputs_rejected(self):
        for options in [dict(confidence=-.1), dict(confidence=float('nan')), dict(max_det=301)]:
            with self.assertRaises(ValueError):
                decode_heads(self.heads(), list(range(80)), **options)
        heads = self.heads()
        heads[0][0] = float('nan')
        with self.assertRaises(ValueError):
            decode_heads(heads, list(range(80)))


@unittest.skipUnless(os.environ.get('ANDROMEDA_YOLO26_CHECKPOINT'),
                     'set ANDROMEDA_YOLO26_CHECKPOINT to the cached official yolo26n.pt')
class ArtifactTests(unittest.TestCase):
    def test_rebuild_both_canonical_bundles_and_reject_tampering(self):
        self.check_variant('n', os.environ['ANDROMEDA_YOLO26_CHECKPOINT'])

    @unittest.skipUnless(os.environ.get('ANDROMEDA_YOLO26S_CHECKPOINT'), 'set ANDROMEDA_YOLO26S_CHECKPOINT')
    def test_small_variant_all_strategies(self):
        self.check_variant('s', os.environ['ANDROMEDA_YOLO26S_CHECKPOINT'])

    def check_variant(self, variant, checkpoint):
        model = load_checkpoint(checkpoint, variant=variant)
        reference = None
        _, image, _ = letterbox_image(HERE.parent / 'test_samples/people.jpg', (256, 256))
        with tempfile.TemporaryDirectory() as directory:
            for winograd, optimize in ((False, False), (True, False), (True, True)):
                with self.subTest(variant=variant, winograd=winograd, optimize=optimize):
                    path = Path(directory) / 'model.bin'
                    with contextlib.redirect_stdout(io.StringIO()):
                        compile_artifact(model, path, winograd=winograd, optimize=optimize)
                    payload = load_artifact(path)
                    self.assertEqual(payload['winograd'], winograd)
                    with contextlib.redirect_stdout(io.StringIO()):
                        if optimize:
                            self.assertEqual(build_artifact(output=path, variant=variant), path)
                        else:
                            with self.assertRaisesRegex(RuntimeError, 'another convolution strategy'):
                                build_artifact(output=path, variant=variant)
                    for resolution in CONFIG['resolutions']:
                        selected = select_profile(payload, resolution)
                        self.assertEqual(sum(op['op'] == 'attention' for op in selected['operations']), 4 if variant == 'n' else 8)
                        if optimize:
                            self.assertTrue(any(op['op'] == 'select' for op in selected['operations']))
                            self.assertTrue(any(op['op'] == 'depthwise' for op in selected['operations']))
                    with torch.inference_mode():
                        heads = execute_cpu(select_profile(payload, '256x256'), image)
                    if reference is None:
                        reference = heads
                    else:
                        for actual, expected in zip(heads, reference):
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    corrupted = copy.deepcopy(payload)
                    corrupted['names'][0] = 'wrong class'
                    with self.assertRaisesRegex(RuntimeError, 'digest'):
                        validate_artifact(corrupted)
                    corrupted['bundle_sha256'] = bundle_digest(corrupted)
                    with self.assertRaisesRegex(RuntimeError, 'canonical'):
                        validate_artifact(corrupted)


@unittest.skipUnless(os.environ.get('ANDROMEDA_YOLO26_CHECKPOINT') and os.environ.get('ANDROMEDA_ULTRALYTICS_SRC'),
                     'set ANDROMEDA_YOLO26_CHECKPOINT and ANDROMEDA_ULTRALYTICS_SRC for official parity checks')
class UpstreamTests(unittest.TestCase):
    def test_raw_heads_and_nms_free_decode_match_official_model(self):
        self.check_variant('n', os.environ['ANDROMEDA_YOLO26_CHECKPOINT'])

    @unittest.skipUnless(os.environ.get('ANDROMEDA_YOLO26S_CHECKPOINT'), 'set ANDROMEDA_YOLO26S_CHECKPOINT')
    def test_small_variant_raw_heads_and_decode(self):
        self.check_variant('s', os.environ['ANDROMEDA_YOLO26S_CHECKPOINT'])

    def check_variant(self, variant, checkpoint):
        sys.path.insert(0, os.environ['ANDROMEDA_ULTRALYTICS_SRC'])
        from ultralytics.nn.tasks import DetectionModel  # noqa: F401
        model = load_checkpoint(checkpoint, variant=variant)  # Verify SHA before the upstream pickle loader.
        reference = torch.load(checkpoint, map_location='cpu', weights_only=False)['model'].float().eval()
        reference.model[-1].end2end = True
        with torch.inference_mode():
            for resolution in CONFIG['resolutions']:
                width, height = map(int, resolution.split('x'))
                for image in ('people.jpg', 'vette.jpg'):
                    with self.subTest(variant=variant, resolution=resolution, image=image):
                        _, x, _ = letterbox_image(HERE.parent / 'test_samples' / image, (height, width))
                        official, raw = reference(x.unsqueeze(0))
                        pred = raw['one2one']
                        golden = torch.cat((pred['boxes'][0], pred['scores'][0]))
                        heads = execute_model(model, x, FloatBackend(quantized=False))
                        actual = torch.cat([h.flatten(1) for h in heads], dim=1)
                        torch.testing.assert_close(actual, golden, atol=.002, rtol=1e-4)
                        detections = decode_heads(heads, model.names)
                        actual = torch.tensor([(*d.box, d.score, d.class_id) for d in detections])
                        golden = official[0][official[0, :, 4] >= .25]
                        torch.testing.assert_close(actual, golden, atol=.02, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
