"""Host-only regression tests for the YOLOv5 integration."""

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import user_dma_core
import model_auto_test
import yolov5s_run_from_bin
import yolov5_precompiled
from yolov5_artifact import (
    _pack_codes,
    _unpack_codes,
    SUPPORTED_INPUT_RESOLUTIONS,
    available_input_resolutions,
    artifact_variant,
    get_canonical_artifact,
    parse_input_resolution,
    select_single_bin_profile,
)
from yolov5_precompiled import (
    _scan_queue_configs,
    compile_precompiled_hardware,
    PRECOMPILED_AXI_DATA_WIDTH_BITS,
    validate_precompiled_hardware,
)
from yolov5_common import (
    box_iou_one_to_many,
    configure_hardware_runtime,
    decode_yolov5,
    fold_conv_bn,
    gather_if8_is_profitable,
    get_yolov5_variant,
    letterbox_image,
    non_max_suppression,
    quantize_conv_for_andromeda,
    quantize_conv_if4,
)


# Resolve the public whole-graph backend without coupling import-time helper
# tests to a direct class import.
WholeGraphAndromedaBackend = getattr(
    yolov5_precompiled, "WholeGraphAndromedaBackend", None)


def _pack_tiled_result_reference(
        canonical: torch.Tensor, tiles, oc_chunk: int) -> torch.Tensor:
    """Serialize exact tile results in compiler/runtime DRAM order."""
    oc_count, _out_h, _out_w = canonical.shape
    assert oc_count % oc_chunk == 0
    chunks = []
    for oc0 in range(0, oc_count, oc_chunk):
        slots = []
        for oy0, ox0, th, tw, *_rest in tiles:
            values = canonical[
                oc0:oc0 + oc_chunk, oy0:oy0 + th, ox0:ox0 + tw] \
                .permute(1, 2, 0).reshape(-1)
            padded = torch.zeros(
                ((values.numel() + user_dma_core.UE_VECTOR_SIZE - 1)
                 // user_dma_core.UE_VECTOR_SIZE)
                * user_dma_core.UE_VECTOR_SIZE,
                dtype=canonical.dtype)
            padded[:values.numel()] = values
            slots.append(padded)
        chunks.append(torch.cat(slots))
    return torch.cat(chunks)


class VariantTests(unittest.TestCase):
    def test_nano_profile_and_artifact_are_pinned(self):
        nano = get_yolov5_variant("yolov5n")
        self.assertEqual(nano.key, "n")
        self.assertEqual(nano.model_name, "YOLOv5n")
        self.assertEqual(nano.parameter_count, 1_872_157)
        self.assertEqual(nano.detect_input_channels, (64, 128, 256))
        self.assertEqual(
            nano.checkpoint_sha256,
            "4f180cf23ba0717ada0badd6c685026d73d48f184d00fc159c2641284b2ac0a3")

        artifact = get_canonical_artifact("n")
        self.assertEqual(artifact.format, "andromeda.yolov5n.single-bin")
        self.assertEqual(
            artifact.graph_sha256,
            "69dbe201b089756886973e98b9f24e8b58c27baf195f7b31c97f7d9c9c2e4272")
        self.assertEqual(
            artifact.weights_sha256,
            "b8ca688e49a44c87b657de6b5c4c9d7b0193718c40ccc8f860733cd89865f666")
        self.assertEqual(artifact.model_bytes, 19_718_784)
        self.assertEqual(artifact.program_bytes, 290_560)
        self.assertEqual(
            artifact.model_sha256,
            "fe1674e69a2645d2b62937184f5cffdef89f663437615d318a6e58269fbc7cce")
        self.assertEqual(
            artifact.program_sha256,
            "e1a8cf92d99aaa1516c9c39122a660c65f04f8ab2baaf8af1662ce52419aa924")
        self.assertEqual(artifact_variant({"format": artifact.format}), "n")

    def test_variant_configs_match_pinned_profiles(self):
        paths = {
            "s": SCRIPT_DIR / "yolov5s_config.json",
            "n": REPO_ROOT / "models/yolov5n/yolov5n_config.json",
        }
        for key, path in paths.items():
            with self.subTest(variant=key):
                config = json.loads(path.read_text(encoding="utf-8"))
                profile = get_yolov5_variant(key)
                self.assertEqual(config["model"]["name"], profile.model_name)
                self.assertEqual(
                    config["source"]["weights_url"], profile.checkpoint_url)
                self.assertEqual(
                    config["source"]["weights_sha256"],
                    profile.checkpoint_sha256)
                self.assertEqual(config["model"]["input_size"], 256)
                self.assertEqual(
                    tuple(config["model"]["input_resolutions"]),
                    SUPPORTED_INPUT_RESOLUTIONS)
                self.assertEqual(
                    config["model"]["precision"],
                    "channel IF4 + gather IF8")

    def test_unknown_variant_and_artifact_format_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "unsupported YOLOv5 variant"):
            get_yolov5_variant("x")
        with self.assertRaisesRegex(RuntimeError, "unsupported YOLO artifact"):
            artifact_variant({"format": "andromeda.yolov5x.single-bin"})

    def test_model_harness_checks_canonical_nano_direct_bin_fixture(self):
        reported_version = 0x1234ABCD
        result = {
            "model": "yolov5n",
            "decoded_text": "person, person",
            "n_detections": 2,
            "backend": "hardware",
            "geometry_abi": "conv-config-inst-v1",
            "hardware_version": f"0x{reported_version:08x}",
            "precompiled": True,
            "artifact_version": 6,
            "full_graph": True,
            "model_upload_writes": 1,
            "input_upload_writes": 1,
            "program_kicks": 1,
            "output_reads": 1,
            "intermediate_upload_writes": 0,
            "intermediate_output_reads": 0,
            "input_resolution": "256x256",
            "input_shape": [3, 256, 256],
            "artifact": "/tmp/yolov5n-andromeda.bin",
        }
        text = "TEST_RESULT:" + json.dumps(result, separators=(",", ":"))
        self.assertTrue(model_auto_test._check_yolov5n_single_bin(text)[0])
        self.assertFalse(model_auto_test._check_yolov5_single_bin(text)[0])

    def test_model_harness_registers_only_canonical_direct_bin_entries(self):
        entries = {
            test["name"]: test for test in model_auto_test.TESTS
            if test["name"].startswith("yolov5")
        }
        self.assertEqual(set(entries), {"yolov5s", "yolov5n"})
        expected = {
            "yolov5s": (
                "models/yolov5s/yolov5s_run_from_bin.py",
                "models/yolov5s/yolov5s_compile.py",
                model_auto_test._check_yolov5_single_bin,
            ),
            "yolov5n": (
                "models/yolov5n/yolov5n_run_from_bin.py",
                "models/yolov5n/yolov5n_compile.py",
                model_auto_test._check_yolov5n_single_bin,
            ),
        }
        for name, (script, compile_script, checker) in expected.items():
            with self.subTest(name=name):
                entry = entries[name]
                self.assertEqual(entry["script"], script)
                self.assertEqual(entry["compile_script"], compile_script)
                self.assertIs(entry["pass_check"], checker)


class DirectBinHarnessTests(unittest.TestCase):
    @staticmethod
    def _fixture():
        return {
            "name": "yolov5s-direct-bin-fixture",
            "script": "models/yolov5s/yolov5s_run_from_bin.py",
            "compile_script": "models/yolov5s/yolov5s_compile.py",
            "extra_args": ["--cpu"],
            "pass_check": lambda text: (text == "runtime-output", text),
        }

    def test_compile_script_runs_first_with_selected_interpreter(self):
        test = self._fixture()
        selected_python = "/opt/yolo/bin/python"
        compile_script = str(
            Path(model_auto_test.SCRIPT_DIR) / test["compile_script"])
        run_script = str(Path(model_auto_test.SCRIPT_DIR) / test["script"])
        with mock.patch.object(
                model_auto_test, "_python_executable_for_test",
                return_value=selected_python), mock.patch.object(
                    model_auto_test, "_run_captured_subprocess",
                    side_effect=(("compiler-output", 0, 0.25),
                                 ("runtime-output", 0, 0.5))) as run, \
                mock.patch.object(model_auto_test, "RANDOMIZE_DRAM", False), \
                contextlib.redirect_stdout(io.StringIO()):
            result = model_auto_test.run_test(test)

        self.assertTrue(result["passed"])
        self.assertEqual(result["stdout"], "runtime-output")
        self.assertEqual(run.call_args_list, [
            mock.call(
                [selected_python, "-u", compile_script], verbose=False,
                activity="compiling yolov5s-direct-bin-fixture"),
            mock.call(
                [selected_python, "-u", run_script, "--cpu"], verbose=False,
                activity="running yolov5s-direct-bin-fixture"),
        ])

    def test_compile_failure_stops_before_poison_and_runtime(self):
        test = self._fixture()
        selected_python = "/opt/yolo/bin/python"
        compile_script = str(
            Path(model_auto_test.SCRIPT_DIR) / test["compile_script"])
        with mock.patch.object(
                model_auto_test, "_python_executable_for_test",
                return_value=selected_python), mock.patch.object(
                    model_auto_test, "_run_captured_subprocess",
                    return_value=("compiler failed", 7, 0.25)) as run, \
                mock.patch.object(model_auto_test, "RANDOMIZE_DRAM", True), \
                mock.patch.object(model_auto_test, "randomize_dram") as poison, \
                contextlib.redirect_stdout(io.StringIO()):
            result = model_auto_test.run_test(test)

        self.assertFalse(result["passed"])
        self.assertEqual(result["stdout"], "compiler failed")
        self.assertEqual(result["pass_reason"], "compile step exited with code 7")
        run.assert_called_once_with(
            [selected_python, "-u", compile_script], verbose=False,
            activity="compiling yolov5s-direct-bin-fixture")
        poison.assert_not_called()

    def test_hardware_runner_calls_whole_graph_execute_once(self):
        image = torch.zeros(3, 8, 8, dtype=torch.bfloat16)
        heads = [
            torch.zeros(255, 1, 1, dtype=torch.bfloat16),
            torch.zeros(255, 1, 1, dtype=torch.bfloat16),
            torch.zeros(255, 1, 1, dtype=torch.bfloat16),
        ]
        payload = {
            "artifact_version": 6,
            "runtime": {
                "postprocessing": {
                    "confidence_threshold": 0.25,
                    "iou_threshold": 0.45,
                    "max_detections": 300,
                },
                "geometry_abi": "conv-config-inst-v1",
                "output_suffix": "_detections_hw.jpg",
            },
        }
        selected = {
            "model": {"input_shape": [3, 8, 8]},
            "operations": ["must-not-be-host-dispatched"],
            "head_outputs": ["head0", "head1", "head2"],
            "hardware": {"abi": "andromeda-yolov5-whole-graph-v1"},
        }
        profile = types.SimpleNamespace(key="s", model_name="YOLOv5s")
        config = {"paths": {
            "artifact": "yolov5s_bin/yolov5s-andromeda.bin",
            "default_image": "bus.jpg",
            "bin_dir": "yolov5s_bin",
        }}

        class FakeEngine:
            def __init__(self):
                self._clock_period_ns = 10.0
                self.software_reset = mock.Mock()

        engine = FakeEngine()

        class FakeWholeGraphBackend:
            instance = None

            def __init__(self, ue, selected_payload, **_kwargs):
                self.__class__.instance = self
                self.ue = ue
                self.payload = selected_payload
                self.images = []
                self.hw_version = 0x1234ABCD
                self.cycles = {"whole_graph": 0}
                self.static_dram_load_seconds = 0.0
                self.static_dram_load_bytes = 64
                self.static_dram_load_writes = 1

            def execute(self, image_chw):
                self.images.append(image_chw)
                return heads

        per_op = mock.Mock(side_effect=AssertionError(
            "hardware runner called the host per-operation executor"))
        hw_info = types.SimpleNamespace(axi_data_width_bits=256)
        with tempfile.TemporaryDirectory() as directory, \
                contextlib.ExitStack() as stack, \
                contextlib.redirect_stdout(io.StringIO()):
            resource_dir = Path(directory)
            for name in ("PrecompiledAndromedaBackend",
                         "WholeGraphAndromedaBackend"):
                if hasattr(yolov5s_run_from_bin, name):
                    stack.enter_context(mock.patch.object(
                        yolov5s_run_from_bin, name, FakeWholeGraphBackend))
            stack.enter_context(mock.patch.object(
                yolov5s_run_from_bin, "_cap_hardware_host_threads"))
            stack.enter_context(mock.patch.object(
                yolov5s_run_from_bin, "load_yolov5_config",
                return_value=(profile, config, resource_dir)))
            stack.enter_context(mock.patch.object(
                yolov5s_run_from_bin, "load_single_bin", return_value=payload))
            stack.enter_context(mock.patch.object(
                yolov5s_run_from_bin, "artifact_variant", return_value="s"))
            stack.enter_context(mock.patch.object(
                yolov5s_run_from_bin, "available_input_resolutions",
                return_value=("8x8",)))
            stack.enter_context(mock.patch.object(
                yolov5s_run_from_bin, "select_single_bin_profile",
                return_value=("8x8", selected)))
            stack.enter_context(mock.patch.object(
                yolov5s_run_from_bin, "artifact_model_view",
                return_value=object()))
            stack.enter_context(mock.patch.object(
                yolov5s_run_from_bin, "letterbox_image",
                return_value=(object(), image, object())))
            stack.enter_context(mock.patch.object(
                yolov5s_run_from_bin, "configure_hardware_runtime",
                return_value=(10.0, hw_info, 10.0)))
            stack.enter_context(mock.patch.object(
                yolov5s_run_from_bin.user_dma_core, "UnifiedEngine",
                return_value=engine))
            stack.enter_context(mock.patch.object(
                yolov5s_run_from_bin, "execute_single_bin", per_op))
            stack.enter_context(mock.patch.object(
                yolov5s_run_from_bin, "decode_yolov5", return_value=object()))
            stack.enter_context(mock.patch.object(
                yolov5s_run_from_bin, "non_max_suppression", return_value=[]))
            stack.enter_context(mock.patch.object(
                yolov5s_run_from_bin, "restore_boxes", return_value=[]))
            stack.enter_context(mock.patch.object(
                yolov5s_run_from_bin, "draw_detections"))
            stack.enter_context(mock.patch.object(
                yolov5s_run_from_bin, "sha256_file", return_value="0" * 64))
            yolov5s_run_from_bin.main([
                "--backend", "hardware",
                "--bin", str(resource_dir / "model.bin"),
                "--image", str(resource_dir / "image.jpg"),
                "--output", str(resource_dir / "output.jpg"),
            ])

        per_op.assert_not_called()
        self.assertIsNotNone(FakeWholeGraphBackend.instance)
        self.assertEqual(len(FakeWholeGraphBackend.instance.images), 1)
        self.assertIs(FakeWholeGraphBackend.instance.images[0], image)
        engine.software_reset.assert_called_once()


class MultiResolutionTests(unittest.TestCase):
    def test_resolution_parser_and_profile_selection_fail_closed(self):
        self.assertEqual(parse_input_resolution("640x480"), (480, 640))
        self.assertEqual(parse_input_resolution(256), (256, 256))
        for invalid in ("640", "abcx480", "640x481", (640,), True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    parse_input_resolution(invalid)

        payload = {
            "model": {"input_shape": [3, 256, 256]},
            "operations": ["default-operations"],
            "head_outputs": ["default-heads"],
            "hardware": {"profile": "default"},
            "graph_sha256": "default-graph",
            "profiles": {
                "640x480": {
                    "input_shape": [3, 480, 640],
                    "operations": ["rectangular-operations"],
                    "head_outputs": ["rectangular-heads"],
                    "hardware": {"profile": "rectangular"},
                    "graph_sha256": "rectangular-graph",
                },
            },
            "bundle_sha256": "bundle",
        }
        self.assertEqual(
            available_input_resolutions(payload), ("256x256", "640x480"))
        key, selected = select_single_bin_profile(payload, "640x480")
        self.assertEqual(key, "640x480")
        self.assertEqual(selected["model"]["input_shape"], [3, 480, 640])
        self.assertEqual(selected["hardware"]["profile"], "rectangular")
        self.assertNotIn("profiles", selected)
        with self.assertRaisesRegex(ValueError, "not embedded.*available"):
            select_single_bin_profile(payload, "320x320")

    def test_rectangular_letterbox_uses_width_by_height_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "wide.png"
            Image.new("RGB", (800, 400), (255, 0, 0)).save(source)
            original, tensor, info = letterbox_image(source, (480, 640))
        self.assertEqual(original.size, (800, 400))
        self.assertEqual(tuple(tensor.shape), (3, 480, 640))
        self.assertAlmostEqual(info.ratio, 0.8)
        self.assertEqual((info.pad_left, info.pad_top), (0, 80))
        self.assertAlmostEqual(float(tensor[0, 240, 320]), 1.0)
        self.assertAlmostEqual(float(tensor[0, 0, 0]), 114 / 255.0, places=6)


class QuantizationTests(unittest.TestCase):
    def test_if4_odd_code_roundtrip_rejects_wrong_payload_size(self):
        codes = torch.tensor([[[[1]], [[2]], [[15]]]], dtype=torch.uint8)
        packed, shape = _pack_codes(codes, "if4")
        self.assertEqual(packed.tolist(), [0x21, 0x0F])
        self.assertTrue(torch.equal(_unpack_codes(packed, shape, "if4"), codes))
        with self.assertRaisesRegex(RuntimeError, "IF4 code payload size"):
            _unpack_codes(torch.cat((packed, packed[:1])), shape, "if4")

    def test_gather_if8_quantization_and_artifact_code_roundtrip(self):
        torch.manual_seed(5)
        conv = torch.nn.Conv2d(3, 8, 6, stride=2, padding=2, bias=False)
        bn = torch.nn.BatchNorm2d(8).eval()
        prepared = quantize_conv_for_andromeda(
            conv, bn, include_dequant=True)
        self.assertTrue(prepared.gather)
        self.assertEqual(prepared.data_type, user_dma_core.TYPE.IF8)
        self.assertEqual(tuple(prepared.codes.shape), (8, 3, 6, 6))
        self.assertEqual(tuple(prepared.block_scales.shape), (8, 2))
        self.assertGreaterEqual(int(prepared.codes.min()), 0)
        self.assertLessEqual(int(prepared.codes.max()), 255)
        packed, shape = _pack_codes(prepared.codes, "if8")
        self.assertEqual(packed.numel(), prepared.codes.numel())
        self.assertTrue(torch.equal(
            _unpack_codes(packed, shape, "if8"), prepared.codes))

    def test_vectorized_tile_pack_matches_per_tile_reference(self):
        torch.manual_seed(11)
        x = torch.randn(65, 7, 9, dtype=torch.bfloat16)
        _, _, _, tiles = user_dma_core.plan_conv2d_layer_tiles(
            c_in=65, oc_count=64, in_h=7, in_w=9,
            kernel_h=3, kernel_w=3, stride_s=2, pad=1)
        x_padded = torch.nn.functional.pad(x, (1, 1, 1, 1))
        expected = []
        for tile in tiles:
            y0, x0 = tile[4:6]
            win_h, win_w = tile[6:8]
            window = x_padded[:, y0:y0 + win_h, x0:x0 + win_w]
            expected.append(
                user_dma_core.conv2d_pack_activation_map(window, 0))
        expected = torch.cat(expected)
        actual = user_dma_core.conv2d_pack_activation_tiles(
            x, tiles, pad=1)
        self.assertTrue(torch.equal(actual, expected))

    def test_vectorized_tile_unpack_matches_exact_reference(self):
        out_h, out_w, oc_chunk, tiles = \
            user_dma_core.plan_conv2d_layer_tiles(
                c_in=64, oc_count=64, in_h=7, in_w=9,
                kernel_h=3, kernel_w=3, stride_s=1, pad=1)
        canonical = torch.arange(
            64 * out_h * out_w, dtype=torch.int64).remainder(1024) \
            .to(torch.bfloat16).view(64, out_h, out_w)
        big = _pack_tiled_result_reference(canonical, tiles, oc_chunk)
        actual = user_dma_core.conv2d_unpack_tiled_result(
            big, tiles, out_h, out_w, 64, oc_chunk)
        self.assertTrue(torch.equal(actual, canonical))

    def test_vectorized_tile_unpack_preserves_multiple_oc_chunks(self):
        out_h, out_w, oc_chunk, tiles = \
            user_dma_core.plan_conv2d_layer_tiles(
                c_in=3, oc_count=255, in_h=7, in_w=6,
                kernel_h=1, kernel_w=1, stride_s=1, pad=0)
        self.assertEqual(oc_chunk, 255)
        self.assertEqual(
            [(group[2], group[3]) for group in
             user_dma_core.conv2d_tile_geometry_groups(tiles)],
            [(6, 5), (6, 1), (1, 5), (1, 1)])
        canonical = torch.arange(
            510 * out_h * out_w, dtype=torch.int64).remainder(2048) \
            .to(torch.bfloat16).view(510, out_h, out_w)
        big = _pack_tiled_result_reference(canonical, tiles, oc_chunk)
        actual = user_dma_core.conv2d_unpack_tiled_result(
            big, tiles, out_h, out_w, 510, oc_chunk)
        self.assertTrue(torch.equal(actual, canonical))

    def test_vectorized_tile_unpack_handles_padded_detect_tiles(self):
        out_h, out_w, oc_chunk, tiles = \
            user_dma_core.plan_conv2d_layer_tiles(
                c_in=64, oc_count=255, in_h=10, in_w=10,
                kernel_h=1, kernel_w=1, stride_s=1, pad=0)
        self.assertEqual(oc_chunk, 255)
        self.assertGreater(len(tiles), 1)
        self.assertTrue(any(
            tile[2] * tile[3] * oc_chunk % 64 for tile in tiles))
        canonical = torch.arange(
            255 * out_h * out_w, dtype=torch.int64).remainder(1024) \
            .to(torch.bfloat16).view(255, out_h, out_w)
        big = _pack_tiled_result_reference(canonical, tiles, oc_chunk)
        actual = user_dma_core.conv2d_unpack_tiled_result(
            big, tiles, out_h, out_w, 255, oc_chunk)
        self.assertTrue(torch.equal(actual, canonical))

    def test_conv_if4_uses_native_block_layout(self):
        torch.manual_seed(7)
        conv = torch.nn.Conv2d(65, 3, 3, bias=False)
        bn = torch.nn.BatchNorm2d(3).eval()
        with torch.no_grad():
            bn.weight.copy_(torch.tensor((0.75, 1.25, -0.5)))
            bn.bias.copy_(torch.tensor((0.1, -0.2, 0.3)))
            bn.running_mean.copy_(torch.tensor((0.2, -0.1, 0.4)))
            bn.running_var.copy_(torch.tensor((0.8, 1.1, 0.6)))

        prepared = quantize_conv_if4(conv, bn, include_dequant=True)
        self.assertEqual(tuple(prepared.codes.shape), (3, 65, 3, 3))
        self.assertEqual(tuple(prepared.block_scales.shape), (3, 18))
        self.assertEqual(tuple(prepared.bias.shape), (3,))
        self.assertTrue(torch.isfinite(prepared.block_scales.float()).all())
        self.assertTrue((prepared.block_scales != 0).all())
        self.assertGreaterEqual(int(prepared.codes.min()), 0)
        self.assertLessEqual(int(prepared.codes.max()), 15)

        folded, _ = fold_conv_bn(conv, bn)
        error = (folded - prepared.dequant_weight.float()).square().mean()
        signal = folded.square().mean()
        self.assertLess(float(error / signal), 0.04)

    def test_scale_packer_preserves_if4_dispatch_sign(self):
        signed = torch.tensor((
            (-0.5, 0.25),
            (0.75, -1.0),
            (-1.5, 2.0),
        ), dtype=torch.bfloat16)
        packed = user_dma_core.conv2d_pack_scale_stream(
            signed, oc_count=3, taps=2, out_h=2, out_w=1)
        self.assertTrue(torch.equal(packed[:6], signed.flatten()))
        self.assertTrue(torch.equal(packed[6:], signed.flatten()))

    def test_layer_scale_chunks_preserve_all_oc_rows(self):
        _, _, chunk, _ = user_dma_core.plan_conv2d_layer_tiles(
            c_in=512, oc_count=512, in_h=20, in_w=20,
            kernel_h=3, kernel_w=3, stride_s=1, pad=1)
        scales = torch.arange(512, dtype=torch.float32).to(torch.bfloat16)
        scales = scales[:, None].expand(512, 72).contiguous()
        selected = [
            user_dma_core._conv2d_chunk_scale(
                scales, 1.0, oc0=oc0, n_oc=chunk)
            for oc0 in range(0, 512, chunk)
        ]
        self.assertEqual(chunk, 64)
        self.assertTrue(torch.equal(torch.cat(selected), scales))
        self.assertEqual(
            user_dma_core._conv2d_chunk_scale(
                None, 0.75, oc0=64, n_oc=64),
            -0.75)


class PlannerTests(unittest.TestCase):
    def test_hardware_runtime_always_initializes_from_hw_info(self):
        detected_clock = 1000.0 / 333.25
        info = types.SimpleNamespace(axi_data_width_bits=512)
        cases = (
            ("rk", "xdma7", None, "xdma7", detected_clock),
            ("rk", "xdma3", 2.75, "xdma3", 2.75),
            ("efinix", "ignored", None, "efinix", detected_clock),
        )
        for device, dev, override, expected_dev, expected_clock in cases:
            with self.subTest(device=device, override=override), \
                 mock.patch.object(user_dma_core, "set_dma_device") as set_dev, \
                 mock.patch.object(
                     user_dma_core, "configure_clock_from_hardware",
                     return_value=detected_clock) as configure, \
                 mock.patch.object(
                     user_dma_core, "configured_hardware_info",
                     return_value=info) as configured_info:
                effective, actual_info, actual_detected = \
                    configure_hardware_runtime(
                        device=device, dev=dev,
                        cycle_override_ns=override)
            set_dev.assert_called_once_with(expected_dev)
            configure.assert_called_once_with()
            configured_info.assert_called_once_with()
            self.assertEqual(effective, expected_clock)
            self.assertIs(actual_info, info)
            self.assertEqual(actual_detected, detected_clock)

    def test_hardware_runner_caps_host_threads_without_raising_small_values(self):
        with mock.patch.object(
                yolov5s_run_from_bin.torch, "get_num_threads", return_value=24), \
             mock.patch.object(
                 yolov5s_run_from_bin.torch, "set_num_threads") as set_threads:
            yolov5s_run_from_bin._cap_hardware_host_threads()
            set_threads.assert_called_once_with(4)

        with mock.patch.object(
                yolov5s_run_from_bin.torch, "get_num_threads", return_value=2), \
             mock.patch.object(
                 yolov5s_run_from_bin.torch, "set_num_threads") as set_threads:
            yolov5s_run_from_bin._cap_hardware_host_threads()
            set_threads.assert_not_called()

    def test_dma_tensor_buffer_path_is_byte_exact(self):
        engine = object.__new__(user_dma_core.UnifiedEngine)
        raw = torch.arange(32, dtype=torch.uint8)
        for dtype in (torch.uint8, torch.uint16, torch.int32,
                      torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                source = raw.view(dtype)
                with tempfile.NamedTemporaryFile() as device:
                    size = source.numel() * source.element_size()
                    self.assertEqual(
                        engine.dma_write(device.name, 0, source, size), size)
                    result = torch.empty_like(source)
                    self.assertEqual(
                        engine.dma_read(device.name, 0, result, size), size)
                self.assertTrue(torch.equal(
                    result.view(torch.uint8), source.view(torch.uint8)))

    def test_dma_read_updates_noncontiguous_tensor_and_preserves_short_tail(self):
        engine = object.__new__(user_dma_core.UnifiedEngine)
        source = torch.tensor(
            [0x3F80, 0xBF00, 0x0001, 0x7F7F], dtype=torch.uint16) \
            .view(torch.bfloat16).reshape(2, 2)
        storage = torch.zeros(2, 2, dtype=torch.bfloat16)
        result = storage.t()
        self.assertFalse(result.is_contiguous())
        with tempfile.NamedTemporaryFile() as device:
            size = source.numel() * source.element_size()
            self.assertEqual(
                engine.dma_write(device.name, 0, source, size), size)
            self.assertEqual(
                engine.dma_read(device.name, 0, result, size), size)
        self.assertTrue(torch.equal(
            result.view(torch.uint16), source.view(torch.uint16)))

        prefix = torch.tensor([0x1111, 0x2222], dtype=torch.uint16)
        result = torch.tensor(
            [0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD], dtype=torch.uint16)
        with tempfile.NamedTemporaryFile() as device:
            self.assertEqual(
                engine.dma_write(device.name, 0, prefix, 4), 4)
            self.assertEqual(
                engine.dma_read(device.name, 0, result, 8), 4)
        self.assertEqual(
            result.tolist(), [0x1111, 0x2222, 0xCCCC, 0xDDDD])

    def test_latency_counter_units_are_hidden_by_engine_wrapper(self):
        engine = object.__new__(user_dma_core.UnifiedEngine)
        engine.read_reg32 = mock.Mock(return_value=7)
        engine._clock_period_ns = 3.0

        self.assertEqual(
            engine.read_latency_cycles(),
            7 * user_dma_core.UE_PIPELINE_COUNTER_CLK_DIV)
        self.assertAlmostEqual(engine.report_latency_in_us(), 0.336)
        self.assertEqual(
            engine.read_reg32.call_args_list,
            [mock.call(user_dma_core.UE_LATENCY_COUNT_ADDR)] * 2)

    def test_oc_chunks_have_one_geometry(self):
        _, _, chunk, _ = user_dma_core.plan_conv2d_layer_tiles(
            c_in=512, oc_count=512, in_h=20, in_w=20,
            kernel_h=3, kernel_w=3, stride_s=1, pad=1)
        self.assertEqual(chunk, 64)
        self.assertEqual(512 % chunk, 0)

        _, _, chunk, _ = user_dma_core.plan_conv2d_layer_tiles(
            c_in=256, oc_count=512, in_h=40, in_w=40,
            kernel_h=3, kernel_w=3, stride_s=1, pad=1)
        self.assertEqual(chunk, 128)
        self.assertEqual(512 % chunk, 0)

    def test_exact_tiles_form_ordered_four_group_partition(self):
        out_h, out_w, oc_chunk, tiles = \
            user_dma_core.plan_conv2d_layer_tiles(
                c_in=65, oc_count=64, in_h=7, in_w=9,
                kernel_h=3, kernel_w=3, stride_s=2, pad=1)
        self.assertEqual((out_h, out_w, oc_chunk), (4, 5, 64))
        groups = user_dma_core.conv2d_tile_geometry_groups(tiles)
        self.assertEqual(groups, [
            (0, 2, 3, 2, 7, 5),
            (2, 3, 3, 1, 7, 3),
            (3, 5, 1, 2, 3, 5),
            (5, 6, 1, 1, 3, 3),
        ])
        occupancy = torch.zeros(out_h, out_w, dtype=torch.int32)
        for oy0, ox0, th, tw, y0, x0, win_h, win_w in tiles:
            occupancy[oy0:oy0 + th, ox0:ox0 + tw] += 1
            self.assertLessEqual(y0 + win_h, 7 + 2)
            self.assertLessEqual(x0 + win_w, 9 + 2)
        self.assertTrue(torch.equal(occupancy, torch.ones_like(occupancy)))
        self.assertEqual(
            sum(tile[2] * tile[3] for tile in tiles), out_h * out_w)

        layout, activation_bytes, chunk_output_bytes = \
            user_dma_core.conv2d_tiled_dram_layout(
                tiles, ct=2, oc_chunk=oc_chunk)
        self.assertEqual(activation_bytes, 33_280)
        self.assertEqual(chunk_output_bytes, 2_560)
        self.assertEqual(
            [(item[6], item[8]) for item in layout],
            [(0, 0), (17_920, 1_536),
             (23_296, 1_920), (30_976, 2_432)])
        self.assertTrue(all(
            value % 128 == 0
            for item in layout for value in (item[6], item[7], item[8], item[9])))

    def test_repeated_noncontiguous_geometry_is_rejected(self):
        tiles = [
            (0, 0, 2, 2, 0, 0, 4, 4),
            (0, 2, 2, 1, 0, 2, 4, 3),
            (2, 0, 2, 2, 2, 0, 4, 4),
        ]
        with self.assertRaisesRegex(AssertionError, "not one contiguous group"):
            user_dma_core.conv2d_tile_geometry_groups(tiles)

    def test_detect_head_keeps_255_channels_together(self):
        for channels in (64, 128):
            with self.subTest(channels=channels):
                _, _, chunk, _ = user_dma_core.plan_conv2d_layer_tiles(
                    c_in=channels, oc_count=255, in_h=32, in_w=32,
                    kernel_h=1, kernel_w=1, stride_s=1, pad=0)
                self.assertEqual(chunk, 255)

    def test_gather_planner_uses_rewound_scale_bram(self):
        channel_plan = user_dma_core.plan_conv2d_layer_tiles(
            c_in=3, oc_count=32, in_h=256, in_w=256,
            kernel_h=6, kernel_w=6, stride_s=2, pad=2,
            gather=False)
        gather_plan = user_dma_core.plan_conv2d_layer_tiles(
            c_in=3, oc_count=32, in_h=256, in_w=256,
            kernel_h=6, kernel_w=6, stride_s=2, pad=2,
            gather=True)
        self.assertEqual(gather_plan[2], 32)
        self.assertLess(len(gather_plan[3]), len(channel_plan[3]))
        self.assertGreater(gather_plan[3][0][2] * gather_plan[3][0][3], 6)

    def test_gather_planner_respects_bias_bram_capacity(self):
        _, _, oc_chunk, tiles = user_dma_core.plan_conv2d_layer_tiles(
            c_in=3, oc_count=32, in_h=256, in_w=256,
            kernel_h=6, kernel_w=6, stride_s=2, pad=2,
            gather=True, bias_enabled=True)
        tile_h, tile_w = tiles[0][2:4]
        self.assertLessEqual(
            tile_h * tile_w * oc_chunk,
            user_dma_core.BIAS_BRAM_ELEMENTS)

    def test_hardware_version_is_masked_and_cached(self):
        engine = user_dma_core.UnifiedEngine.__new__(user_dma_core.UnifiedEngine)
        engine.hw_version = 0x1_1234ABCD
        self.assertEqual(engine.get_hardware_version(), 0x1234ABCD)
        self.assertEqual(engine.hw_version, 0x1234ABCD)

        # Main no longer performs the destructive full initialization for
        # every engine. Version reporting fetches only the version register on
        # first use and caches it for subsequent reads.
        engine.hw_version = None
        engine.user_read_reg32 = mock.Mock(return_value=0x89ABCDEF)
        self.assertEqual(engine.get_hardware_version(), 0x89ABCDEF)
        self.assertEqual(engine.get_hardware_version(), 0x89ABCDEF)
        engine.user_read_reg32.assert_called_once_with(
            user_dma_core.UE_FPGA_VERSION_ADDR)

    def test_queued_geometry_is_ordered(self):
        self.assertEqual(user_dma_core.UE_HW_INFO_ADDR, 0x000000A0)
        self.assertEqual(
            user_dma_core.UE_LAST_REG_ADDR,
            user_dma_core.UE_HW_INFO_ADDR,
        )
        self.assertEqual(
            (
                user_dma_core.UE_CONV_GEOM_ADDR,
                user_dma_core.UE_CONV_CTRL_ADDR,
                user_dma_core.UE_CONV_STRIDE_ADDR,
                user_dma_core.UE_CONV_PIXSTEP_ADDR,
            ),
            (0x0000006C, 0x000000A4, 0x000000A8, 0x000000AC),
        )
        engine = user_dma_core.UnifiedEngine.__new__(user_dma_core.UnifiedEngine)
        engine.conv_geometry_mode = user_dma_core.CONV_GEOMETRY_QUEUE_CONFIG
        engine.is_capture_on = True
        engine.capture_buffer = []
        engine.capture_count = 0
        engine._inst_id = 0
        engine._capture_loop_stack = []
        engine._capture_conv_geometry = None
        writes = []
        engine.write_reg32 = lambda address, value: writes.append((address, value))
        geometry = dict(
            out_w=4, out_h=4, ct=1, kernel_w=1, kernel_h=1,
            oc_count=64, row_stride=4, col_stride=1,
            pix_col_step=1, pix_row_step=4)
        engine.write_conv2d_geometry_registers(**geometry)
        engine.write_conv2d_geometry_registers(**geometry)
        self.assertEqual(writes, [])
        self.assertEqual(engine.capture_count, 2)
        self.assertEqual(engine.capture_buffer[0].words[0], 0x00000C00)
        self.assertEqual(engine.capture_buffer[1].words[0], 0x00000C01)
        expected = user_dma_core.pack_conv2d_geometry_words(**geometry)
        for instruction in engine.capture_buffer:
            self.assertEqual(tuple(instruction.words[1:5]), expected)
            self.assertEqual(instruction.words[5:], [0, 0, 0])
        engine.clear_capture_buffer()
        self.assertIsNone(engine._capture_conv_geometry)

    def test_gather_if8_policy_requires_schedule_and_byte_savings(self):
        self.assertTrue(gather_if8_is_profitable(
            channels=3, out_channels=32, kernel_h=6, kernel_w=6))
        self.assertTrue(gather_if8_is_profitable(
            channels=16, out_channels=32, kernel_h=3, kernel_w=3))
        for out_channels in (32, 64):
            with self.subTest(out_channels=out_channels):
                self.assertFalse(gather_if8_is_profitable(
                    channels=32, out_channels=out_channels,
                    kernel_h=3, kernel_w=3))
        self.assertFalse(gather_if8_is_profitable(
            channels=50, out_channels=64, kernel_h=3, kernel_w=3))

    def test_gather_geometry_rejects_inconsistent_or_oversized_chunks(self):
        engine = user_dma_core.UnifiedEngine.__new__(user_dma_core.UnifiedEngine)
        engine.is_capture_on = False
        engine.write_reg32 = lambda _address, _value: None
        common = dict(
            out_w=1, out_h=1, ct=1, kernel_w=3, kernel_h=3,
            oc_count=2, row_stride=1, col_stride=1,
            pix_col_step=1, pix_row_step=1, gather=True)
        engine.write_conv2d_geometry_registers(
            **common, c_in=8, blocks_per_pixel=4, chunks=2)
        with self.assertRaisesRegex(ValueError, "four RTL"):
            engine.write_conv2d_geometry_registers(
                **common, c_in=33, blocks_per_pixel=10, chunks=5)
        with self.assertRaisesRegex(ValueError, "!= oc_count"):
            engine.write_conv2d_geometry_registers(
                **common, c_in=8, blocks_per_pixel=3, chunks=2)


class PrecompiledBinTests(unittest.TestCase):
    @staticmethod
    def _one_conv_payload():
        # A deliberately small graph exercises the same fixed-address compiler
        # as the 60-convolution artifact without needing a checkpoint.
        codes_shape = [8, 3, 6, 6]
        code_count = 8 * 3 * 6 * 6
        return {
            "model": {"input_shape": [3, 32, 40]},
            "operations": [{
                "op": "conv", "name": "conv", "inputs": ["input"],
                "output": "conv", "output_shape": [8, 16, 20],
                "stride": 2, "pad": 2, "dilation": 1,
                "activate": True,
            }],
            "weights": {
                "conv": {
                    "precision": "if8",
                    "layout": "gather",
                    "codes_packed": torch.zeros(code_count, dtype=torch.uint8),
                    "codes_shape": codes_shape,
                    "block_scales": torch.ones(
                        8, 2, dtype=torch.bfloat16),
                    "bias": torch.zeros(8, dtype=torch.bfloat16),
                },
            },
            "head_outputs": ["conv"],
        }

    @staticmethod
    def _multi_op_payload():
        """Small graph containing every non-host YOLO primitive."""
        return {
            "model": {"input_shape": [3, 8, 8]},
            "operations": [
                {
                    "op": "conv", "name": "stem", "inputs": ["input"],
                    "output": "stem", "output_shape": [8, 8, 8],
                    "stride": 1, "pad": 0, "dilation": 1,
                    "activate": True,
                },
                {
                    "op": "maxpool", "name": "pool", "inputs": ["stem"],
                    "output": "pool", "output_shape": [8, 8, 8],
                    "kernel": 3, "stride": 1, "pad": 1,
                },
                {
                    "op": "add", "name": "residual",
                    "inputs": ["stem", "pool"], "output": "residual",
                    "output_shape": [8, 8, 8],
                },
                {
                    "op": "upsample2x", "name": "up_stem",
                    "inputs": ["stem"], "output": "up_stem",
                    "output_shape": [8, 16, 16],
                },
                {
                    "op": "upsample2x", "name": "up_residual",
                    "inputs": ["residual"], "output": "up_residual",
                    "output_shape": [8, 16, 16],
                },
                {
                    "op": "concat", "name": "concat",
                    "inputs": ["up_stem", "up_residual"],
                    "output": "concat", "output_shape": [16, 16, 16],
                },
            ],
            "head_outputs": ["pool", "residual", "concat"],
            "weights": {
                "stem": {
                    "precision": "if8",
                    "layout": "gather",
                    "codes_packed": torch.zeros(8 * 3, dtype=torch.uint8),
                    "codes_shape": [8, 3, 1, 1],
                    "block_scales": torch.ones(8, 1, dtype=torch.bfloat16),
                    "bias": torch.zeros(8, dtype=torch.bfloat16),
                },
            },
        }

    @staticmethod
    def _program_instruction_types(hardware):
        offset = hardware["program_offset"]
        size = hardware["program_size"]
        image = hardware["model_image"]
        if offset < 0 or size <= 0 or size % 32 or offset + size > image.numel():
            raise AssertionError("invalid whole-graph program range")
        return [
            int(image[position + 1]) & 0xF
            for position in range(offset, offset + size, 32)
        ]

    class FakeRuntimeEngine:
        def __init__(self, *, write_shortfall=0, read_shortfall=0,
                     never_halt=False):
            self.h2c_device = "fake-h2c"
            self.c2h_device = "fake-c2h"
            self.hw_version = 0x1234ABCD
            self.conv_geometry_mode = user_dma_core.CONV_GEOMETRY_QUEUE_CONFIG
            self.write_shortfall = int(write_shortfall)
            self.read_shortfall = int(read_shortfall)
            self.never_halt = bool(never_halt)
            self.writes = []
            self.reads = []
            self.starts = []
            self.register_writes = []
            self.register_reads = []
            self.waits = 0
            self._interrupt_status = []

        def dma_write(self, device, address, _value, size):
            self.writes.append((device, address, size))
            return max(0, size - self.write_shortfall)

        def dma_read(self, device, address, value, size):
            self.reads.append((device, address, size))
            if hasattr(value, "zero_"):
                value.zero_()
            return max(0, size - self.read_shortfall)

        def start_execute_from_dram(self, address):
            self.starts.append(address)
            self._interrupt_status = ([user_dma_core.INT_CAUSE_NONE]
                                      if self.never_halt else [
                                          user_dma_core.INT_CAUSE_NONE,
                                          user_dma_core.INT_CAUSE_HALT,
                                      ])

        def write_reg32(self, address, value):
            self.register_writes.append((address, value))

        def read_reg32(self, address):
            self.register_reads.append(address)
            if address == user_dma_core.UE_INT_REG:
                if self._interrupt_status:
                    return self._interrupt_status.pop(0)
                return (user_dma_core.INT_CAUSE_NONE if self.never_halt
                        else user_dma_core.INT_CAUSE_HALT)
            if address == user_dma_core.UE_LATENCY_COUNT_ADDR:
                return 7
            return 0

        def read_latency_cycles(self):
            return 7 * user_dma_core.UE_PIPELINE_COUNTER_CLK_DIV

        def is_queue_busy(self):
            return False

        def wait_queue(self, *_args, **_kwargs):
            self.waits += 1

        def start_capture(self):  # pragma: no cover - explicit tripwire.
            raise AssertionError("whole-graph runtime must not capture")

        def run_conv2d_layer(self, *_args, **_kwargs):  # pragma: no cover
            raise AssertionError("whole-graph runtime must not dispatch a convolution")

        def run_maxpool2d_layer(self, *_args, **_kwargs):  # pragma: no cover
            raise AssertionError("whole-graph runtime must not dispatch a maxpool")

        def run_nn_upsample_2x(self, *_args, **_kwargs):  # pragma: no cover
            raise AssertionError("whole-graph runtime must not dispatch an upsample")

        def run_eltwise_add_layer(self, *_args, **_kwargs):  # pragma: no cover
            raise AssertionError("whole-graph runtime must not dispatch an add")

    def test_strided_write_unrolls_chunks_beyond_axi_burst_counter(self):
        class RecordingEngine:
            def __init__(self):
                self.calls = []

            def sram_to_accelerator_memory(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        limit = yolov5_precompiled._MAX_STRIDED_WRITE_CHUNK_BYTES
        oversized = limit + PRECOMPILED_AXI_DATA_WIDTH_BITS // 8
        engine = RecordingEngine()
        yolov5_precompiled._copy_contiguous_or_strided_write(
            engine, sram=0x1000, destination=0x2000,
            total=oversized * 2, chunk=oversized, jump=oversized * 2)
        self.assertEqual(engine.calls, [
            ((0x1000, 0x2000, 0), {"memcpy_length_bytes": oversized}),
            ((0x1000 + oversized, 0x2000 + oversized * 2, 0),
             {"memcpy_length_bytes": oversized}),
        ])

        engine = RecordingEngine()
        yolov5_precompiled._copy_contiguous_or_strided_write(
            engine, sram=0x1000, destination=0x2000,
            total=limit * 2, chunk=limit, jump=limit * 2)
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(engine.calls[0][1], {
            "memcpy_length_bytes": limit * 2,
            "stride_bytes_per_chunk": limit,
            "stride_jump_bytes": limit * 2,
        })

    def test_offline_compiler_emits_one_whole_graph_program(self):
        payload = self._multi_op_payload()
        payload["hardware"] = compile_precompiled_hardware(payload)
        validate_precompiled_hardware(payload)
        hardware = payload["hardware"]
        required = {
            "abi", "geometry_abi", "input_base", "input_bytes",
            "model_base", "model_limit", "tensor_base", "tensor_limit",
            "program_address", "program_offset", "program_size",
            "model_image", "model_sha256", "program_sha256", "tensors",
            "heads", "head_bundle_address", "head_bundle_bytes",
            "operations", "scratch_address", "scratch_bytes",
        }
        self.assertFalse(required - set(hardware), sorted(required - set(hardware)))
        self.assertFalse({
            "entries", "params_image", "program_image", "dynamic_params_base",
        }.intersection(hardware))
        self.assertGreater(hardware["model_image"].numel(), 0)
        self.assertEqual(hardware["geometry_abi"], "conv-config-inst-v1")
        self.assertEqual(
            hardware["abi"], "andromeda-yolov5-whole-graph-v1")
        self.assertEqual(
            hardware["program_address"],
            hardware["model_base"] + hardware["program_offset"])
        self.assertEqual(hardware["program_size"] % 64, 0)
        model_bytes = hardware["model_image"].numpy().tobytes()
        self.assertEqual(
            hardware["model_sha256"], hashlib.sha256(model_bytes).hexdigest())
        program = model_bytes[
            hardware["program_offset"]:
            hardware["program_offset"] + hardware["program_size"]]
        self.assertEqual(
            hardware["program_sha256"], hashlib.sha256(program).hexdigest())

        instruction_types = self._program_instruction_types(hardware)
        self.assertEqual(instruction_types.count(user_dma_core.INSTRUCTION_HALT), 1)
        self.assertNotIn(user_dma_core.INSTRUCTION_SWI, instruction_types)
        self.assertEqual(
            [kind for kind in instruction_types
             if kind != user_dma_core.INSTRUCTION_NOP][-1],
            user_dma_core.INSTRUCTION_HALT)
        configs = _scan_queue_configs(
            hardware["model_image"], hardware["program_offset"],
            hardware["program_size"])
        self.assertGreaterEqual(len(configs), 2)  # stem convolution and max-pool
        self.assertEqual(
            [(entry["graph_index"], entry["name"], entry["op"])
             for entry in hardware["operations"]],
            [(index, operation["name"], operation["op"])
             for index, operation in enumerate(payload["operations"])])
        self.assertEqual(
            set(hardware["tensors"]),
            {"input", *(operation["output"] for operation in payload["operations"])})
        self.assertEqual(len(hardware["heads"]), 3)
        self.assertEqual(
            [head["name"] for head in hardware["heads"]],
            payload["head_outputs"])
        next_offset = 0
        for head in hardware["heads"]:
            self.assertEqual(head["offset"], next_offset)
            self.assertEqual(
                head["address"], hardware["head_bundle_address"] + next_offset)
            next_offset += head["size_bytes"]
        self.assertEqual(next_offset, hardware["head_bundle_bytes"])

        # Debug ranges describe every operation but are not dispatch entries:
        # none may contain a host-visible checkpoint, and a single terminal
        # HALT follows all of them.
        next_offset = 0
        for entry in hardware["operations"]:
            self.assertEqual(entry["program_offset"], next_offset)
            start = hardware["program_offset"] + entry["program_offset"]
            stop = start + entry["program_size"]
            entry_types = [
                int(hardware["model_image"][position + 1]) & 0xF
                for position in range(start, stop, 32)
            ]
            self.assertNotIn(user_dma_core.INSTRUCTION_HALT, entry_types)
            self.assertNotIn(user_dma_core.INSTRUCTION_SWI, entry_types)
            next_offset += entry["program_size"]
        self.assertLess(next_offset, hardware["program_size"])

        rebuilt = compile_precompiled_hardware(self._multi_op_payload())
        self.assertEqual(rebuilt["model_sha256"], hardware["model_sha256"])
        self.assertEqual(rebuilt["program_sha256"], hardware["program_sha256"])
        self.assertEqual(rebuilt["program_offset"], hardware["program_offset"])
        self.assertEqual(rebuilt["program_size"], hardware["program_size"])

    def test_runtime_has_one_model_upload_and_one_io_transaction_per_image(self):
        self.assertIsNotNone(
            WholeGraphAndromedaBackend,
            "yolov5_precompiled.WholeGraphAndromedaBackend is unavailable")
        self.assertIs(
            yolov5_precompiled.PrecompiledAndromedaBackend,
            WholeGraphAndromedaBackend)
        payload = self._multi_op_payload()
        payload["hardware"] = compile_precompiled_hardware(payload)
        hardware = payload["hardware"]
        engine = self.FakeRuntimeEngine()
        backend = WholeGraphAndromedaBackend(
            engine, payload,
            axi_data_width_bits=PRECOMPILED_AXI_DATA_WIDTH_BITS)
        self.assertEqual(engine.writes, [(
            engine.h2c_device, hardware["model_base"],
            hardware["model_image"].numel())])
        self.assertEqual(backend.model_upload_writes, 1)
        self.assertEqual(backend.input_upload_writes, 0)
        self.assertEqual(backend.program_kicks, 0)
        self.assertEqual(backend.output_reads, 0)

        image = torch.zeros(3, 8, 8, dtype=torch.bfloat16)
        forbidden = AssertionError("runtime attempted per-operation host work")
        with mock.patch.object(
                user_dma_core, "conv2d_pack_activation_tiles",
                side_effect=forbidden), mock.patch.object(
                user_dma_core, "conv2d_unpack_tiled_result",
                side_effect=forbidden), mock.patch.object(
                user_dma_core.UnifiedEngine, "run_conv2d_layer",
                side_effect=forbidden), mock.patch.object(
                user_dma_core.UnifiedEngine, "run_maxpool2d_layer",
                side_effect=forbidden), mock.patch.object(
                user_dma_core.UnifiedEngine, "run_nn_upsample_2x",
                side_effect=forbidden), mock.patch.object(
                user_dma_core.UnifiedEngine, "run_eltwise_add_layer",
                side_effect=forbidden):
            first = backend.execute(image)
            second = backend.execute(image)

        expected_shapes = ((8, 8, 8), (8, 8, 8), (16, 16, 16))
        self.assertEqual(tuple(tuple(value.shape) for value in first), expected_shapes)
        self.assertEqual(tuple(tuple(value.shape) for value in second), expected_shapes)
        # One immutable deployment upload total, then one packed input upload per
        # image. There are no activation or per-node parameter transfers.
        self.assertEqual(len(engine.writes), 3)
        self.assertEqual(
            engine.writes[1:], [
                (engine.h2c_device, hardware["input_base"], hardware["input_bytes"]),
                (engine.h2c_device, hardware["input_base"], hardware["input_bytes"]),
            ])
        self.assertEqual(engine.starts, [hardware["program_address"]] * 2)
        self.assertEqual(len(engine.reads), 2)
        self.assertEqual(engine.reads[0], engine.reads[1])
        self.assertEqual(engine.reads[0], (
            engine.c2h_device, hardware["head_bundle_address"],
            hardware["head_bundle_bytes"]))
        self.assertEqual(backend.model_upload_writes, 1)
        self.assertEqual(backend.input_upload_writes, 2)
        self.assertEqual(backend.program_kicks, 2)
        self.assertEqual(backend.output_reads, 2)
        self.assertEqual(backend.intermediate_upload_writes, 0)
        self.assertEqual(backend.intermediate_output_reads, 0)
        self.assertFalse(any(address in (
            user_dma_core.UE_CONV_GEOM_ADDR, user_dma_core.UE_CONV_CTRL_ADDR,
            user_dma_core.UE_CONV_STRIDE_ADDR, user_dma_core.UE_CONV_PIXSTEP_ADDR)
            for address, _value in engine.register_writes))

        before = (len(engine.writes), len(engine.starts), len(engine.reads))
        with self.assertRaisesRegex(ValueError, "expects CHW input"):
            backend.execute(torch.zeros(3, 7, 8, dtype=torch.bfloat16))
        self.assertEqual(
            (len(engine.writes), len(engine.starts), len(engine.reads)), before)

    def test_whole_graph_backend_fails_closed_before_or_at_final_io(self):
        self.assertIsNotNone(
            WholeGraphAndromedaBackend,
            "yolov5_precompiled.WholeGraphAndromedaBackend is unavailable")
        payload = self._multi_op_payload()
        payload["hardware"] = compile_precompiled_hardware(payload)

        incompatible = copy.deepcopy(payload)
        incompatible["hardware"]["abi"] = "andromeda-yolov5-precompiled-v4"
        stale_engine = self.FakeRuntimeEngine()
        with self.assertRaisesRegex(RuntimeError, "incompatible fixed-address ABI"):
            WholeGraphAndromedaBackend(
                stale_engine, incompatible,
                axi_data_width_bits=PRECOMPILED_AXI_DATA_WIDTH_BITS)
        self.assertEqual(stale_engine.writes, [])

        wide_engine = self.FakeRuntimeEngine()
        with self.assertRaisesRegex(RuntimeError, "AXI-256"):
            WholeGraphAndromedaBackend(
                wide_engine, payload, axi_data_width_bits=512)
        self.assertEqual(wide_engine.writes, [])

        short_write = self.FakeRuntimeEngine(write_shortfall=1)
        with self.assertRaisesRegex(RuntimeError, "wrote .* of"):
            WholeGraphAndromedaBackend(
                short_write, payload,
                axi_data_width_bits=PRECOMPILED_AXI_DATA_WIDTH_BITS)

        timeout_engine = self.FakeRuntimeEngine(never_halt=True)
        backend = WholeGraphAndromedaBackend(
            timeout_engine, payload,
            axi_data_width_bits=PRECOMPILED_AXI_DATA_WIDTH_BITS,
            timeout_s=0.001)
        with self.assertRaisesRegex(TimeoutError, "never reported HALT"):
            backend.execute(torch.zeros(3, 8, 8, dtype=torch.bfloat16))
        self.assertEqual(len(timeout_engine.starts), 1)
        self.assertEqual(timeout_engine.reads, [])

        short_read_engine = self.FakeRuntimeEngine(read_shortfall=2)
        backend = WholeGraphAndromedaBackend(
            short_read_engine, payload,
            axi_data_width_bits=PRECOMPILED_AXI_DATA_WIDTH_BITS)
        with self.assertRaisesRegex(RuntimeError, "read .* of"):
            backend.execute(torch.zeros(3, 8, 8, dtype=torch.bfloat16))
        self.assertEqual(len(short_read_engine.starts), 1)
        self.assertEqual(len(short_read_engine.reads), 1)

        broken = copy.deepcopy(payload)
        hardware = broken["hardware"]
        image = hardware["model_image"].clone()
        first_type_byte = hardware["program_offset"] + 1
        image[first_type_byte] = (
            (int(image[first_type_byte]) & 0xF0) | user_dma_core.INSTRUCTION_HALT)
        hardware["model_image"] = image
        raw = image.numpy().tobytes()
        hardware["model_sha256"] = hashlib.sha256(raw).hexdigest()
        program = raw[
            hardware["program_offset"]:
            hardware["program_offset"] + hardware["program_size"]]
        hardware["program_sha256"] = hashlib.sha256(program).hexdigest()
        with self.assertRaisesRegex(RuntimeError, "one.*HALT|HALT.*one"):
            validate_precompiled_hardware(broken)

    def test_offline_upsample_uses_canonical_width_and_restores_hw_info(self):
        payload = self._one_conv_payload()
        payload["operations"].append({
            "op": "upsample2x", "name": "upsample",
            "inputs": ["conv"], "output": "upsample",
            "output_shape": [8, 32, 40],
        })
        payload["head_outputs"] = ["upsample"]
        original_width = user_dma_core.UE_AXI_DATA_WIDTH_BITS
        images = []
        try:
            for configured_width in (None, 512):
                with self.subTest(configured_width=configured_width):
                    user_dma_core.UE_AXI_DATA_WIDTH_BITS = configured_width
                    hardware = compile_precompiled_hardware(payload)
                    self.assertEqual(
                        user_dma_core.UE_AXI_DATA_WIDTH_BITS, configured_width)
                    images.append((
                        hardware["model_sha256"],
                        hardware["program_sha256"],
                        hardware["program_offset"],
                        hardware["program_size"],
                    ))
        finally:
            user_dma_core.UE_AXI_DATA_WIDTH_BITS = original_width
        self.assertEqual(images[0], images[1])

    def test_channel_if4_program_embeds_the_planned_geometry(self):
        codes_shape = [8, 64, 1, 1]
        payload = {
            "model": {"input_shape": [64, 4, 4]},
            "operations": [{
                "op": "conv", "name": "conv", "inputs": ["input"],
                "output": "conv", "output_shape": [8, 4, 4],
                "stride": 1, "pad": 0, "dilation": 1,
                "activate": True,
            }],
            "weights": {"conv": {
                "precision": "if4", "layout": "channels",
                "codes_packed": torch.zeros(256, dtype=torch.uint8),
                "codes_shape": codes_shape,
                "block_scales": torch.ones(8, 1, dtype=torch.bfloat16),
                "bias": torch.zeros(8, dtype=torch.bfloat16),
            }},
            "head_outputs": ["conv"],
        }
        hardware = compile_precompiled_hardware(payload)
        configs = _scan_queue_configs(
            hardware["model_image"], hardware["program_offset"],
            hardware["program_size"])
        self.assertEqual(len(configs), 1)
        expected = user_dma_core.pack_conv2d_geometry_words(
            out_w=4, out_h=4, ct=1, kernel_w=1, kernel_h=1,
            # Persistent maps are physically padded to one 64-lane vector.
            oc_count=64, row_stride=4, col_stride=1,
            pix_col_step=1, pix_row_step=4)
        self.assertEqual(configs[0], expected)


class PostprocessTests(unittest.TestCase):
    @staticmethod
    def _dummy_model():
        detect = types.SimpleNamespace(anchors=torch.tensor((
            ((10 / 8, 13 / 8), (16 / 8, 30 / 8), (33 / 8, 23 / 8)),
            ((30 / 16, 61 / 16), (62 / 16, 45 / 16), (59 / 16, 119 / 16)),
            ((116 / 32, 90 / 32), (156 / 32, 198 / 32), (373 / 32, 326 / 32)),
        ), dtype=torch.float32))
        return types.SimpleNamespace(
            names=[f"class_{i}" for i in range(80)],
            stride=torch.tensor((8.0, 16.0, 32.0)),
            model=[None] * 24 + [detect])

    def test_decode_shapes(self):
        model = self._dummy_model()
        raw = [
            torch.zeros(255, 32, 32),
            torch.zeros(255, 16, 16),
            torch.zeros(255, 8, 8),
        ]
        decoded = decode_yolov5(raw, model)
        self.assertEqual(tuple(decoded.shape), (4032, 85))
        self.assertTrue(torch.isfinite(decoded).all())

    def test_decode_rectangular_640x480_heads(self):
        model = self._dummy_model()
        raw = [
            torch.zeros(255, 60, 80),
            torch.zeros(255, 30, 40),
            torch.zeros(255, 15, 20),
        ]
        decoded = decode_yolov5(raw, model)
        self.assertEqual(tuple(decoded.shape), (18_900, 85))
        self.assertTrue(torch.isfinite(decoded).all())

    def test_nms_is_class_aware(self):
        model = self._dummy_model()
        model.names[2] = "car"
        # xywh + objectness + 80 class probabilities.
        rows = torch.zeros(3, 85)
        rows[:, :4] = torch.tensor((100.0, 100.0, 80.0, 80.0))
        rows[:, 4] = 0.9
        rows[0, 5 + 2] = 0.9
        rows[1, 5 + 2] = 0.8  # same class/box: suppressed
        rows[2, 5 + 5] = 0.85  # different class: retained
        detections = non_max_suppression(
            rows, model, conf_threshold=0.25, iou_threshold=0.45)
        self.assertEqual(len(detections), 2)
        self.assertEqual({d.class_id for d in detections}, {2, 5})

    def test_nms_rejects_nonpositive_limits(self):
        model = self._dummy_model()
        rows = torch.zeros(1, 85)
        with self.assertRaisesRegex(ValueError, "max_det"):
            non_max_suppression(rows, model, max_det=0)
        with self.assertRaisesRegex(ValueError, "max_candidates"):
            non_max_suppression(rows, model, max_candidates=-1)

    def test_iou_identity(self):
        box = torch.tensor((1.0, 2.0, 5.0, 8.0))
        iou = box_iou_one_to_many(box, box.unsqueeze(0))
        self.assertAlmostEqual(float(iou[0]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
