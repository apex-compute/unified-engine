"""Host-only regression tests for the YOLOv5 integration."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import user_dma_core
import model_auto_test
import yolov5_run_from_bin
import yolov5_test
from yolov5_artifact import (
    _pack_codes,
    _unpack_codes,
    artifact_variant,
    get_canonical_artifact,
)
from yolov5_precompiled import (
    _scan_queue_configs,
    compile_precompiled_hardware,
    PrecompiledAndromedaBackend,
    validate_precompiled_hardware,
)
from yolov5_common import (
    box_iou_one_to_many,
    decode_yolov5,
    fold_conv_bn,
    gather_if8_is_profitable,
    get_yolov5_variant,
    non_max_suppression,
    quantize_conv_for_andromeda,
    quantize_conv_if4,
)


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
        self.assertEqual(artifact.params_bytes, 18_219_952)
        self.assertEqual(artifact.program_bytes, 93_504)
        self.assertEqual(artifact_variant({"format": artifact.format}), "n")

    def test_variant_configs_match_pinned_profiles(self):
        paths = {
            "s": SCRIPT_DIR / "yolov5_config.json",
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
                    config["model"]["precision"],
                    "channel IF4 + gather IF8")
                self.assertEqual(
                    {int(value, 16) for value in
                     config["hardware"]["gather_if8_fpga_hashes"]},
                    set(user_dma_core.UE_GATHER_IF8_HW_VERSIONS))

    def test_unknown_variant_and_artifact_format_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "unsupported YOLOv5 variant"):
            get_yolov5_variant("x")
        with self.assertRaisesRegex(RuntimeError, "unsupported YOLO artifact"):
            artifact_variant({"format": "andromeda.yolov5x.single-bin"})

    def test_model_harness_checks_nano_identity_fixture_and_artifact(self):
        result = {
            "model": "yolov5n",
            "decoded_text": "person, person",
            "n_detections": 2,
            "backend": "hardware",
            "geometry_abi": "conv-config-inst-v1",
            "hardware_version": "0x77e8adf3",
            "precompiled": True,
            "artifact_version": 4,
            "artifact": "/tmp/yolov5n-andromeda.bin",
        }
        text = "TEST_RESULT:" + json.dumps(result, separators=(",", ":"))
        self.assertTrue(model_auto_test._check_yolov5n(text)[0])
        self.assertTrue(model_auto_test._check_yolov5n_single_bin(text)[0])
        self.assertFalse(model_auto_test._check_yolov5(text)[0])


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
    def test_hardware_runner_caps_host_threads_without_raising_small_values(self):
        with mock.patch.object(
                yolov5_run_from_bin.torch, "get_num_threads", return_value=24), \
             mock.patch.object(
                 yolov5_run_from_bin.torch, "set_num_threads") as set_threads:
            yolov5_run_from_bin._cap_hardware_host_threads()
            set_threads.assert_called_once_with(4)

        with mock.patch.object(
                yolov5_run_from_bin.torch, "get_num_threads", return_value=2), \
             mock.patch.object(
                 yolov5_run_from_bin.torch, "set_num_threads") as set_threads:
            yolov5_run_from_bin._cap_hardware_host_threads()
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

    def test_checkpoint_runner_selects_queue_config_geometry(self):
        fake_engine = mock.Mock()
        fake_backend = object()
        known_versions = {0xD93EEA82}
        with mock.patch.object(
                yolov5_test.user_dma_core, "UnifiedEngine",
                return_value=fake_engine) as engine_constructor, \
                mock.patch.object(
                    yolov5_test, "AndromedaBackend",
                    return_value=fake_backend) as backend_constructor:
            backend = yolov5_test._create_hardware_backend(
                clock=3.0,
                known_hw_versions=known_versions,
                allow_unknown_hardware=False,
                timeout_s=30.0)

        self.assertIs(backend, fake_backend)
        engine_constructor.assert_called_once_with(
            clock_period_ns=3.0,
            allow_unknown_conv_hardware=False,
            conv_geometry_mode=user_dma_core.CONV_GEOMETRY_QUEUE_CONFIG,
            allow_unknown_queue_config_hardware=False,
            allow_unknown_gather_if8_hardware=False)
        fake_engine.software_reset.assert_called_once_with()
        backend_constructor.assert_called_once_with(
            fake_engine,
            known_hw_versions=known_versions,
            allow_unknown_hardware=False,
            timeout_s=30.0)

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

    def test_native_conv_capability_gate(self):
        engine = user_dma_core.UnifiedEngine.__new__(user_dma_core.UnifiedEngine)
        engine.hw_version = 0xCF133B89
        engine._allow_unknown_conv_hardware = False
        with self.assertRaisesRegex(RuntimeError, "predates"):
            engine.require_native_conv_hardware()
        engine._allow_unknown_conv_hardware = True
        engine.require_native_conv_hardware()
        engine._allow_unknown_conv_hardware = False
        engine.hw_version = next(iter(user_dma_core.UE_NATIVE_CONV_HW_VERSIONS))
        engine.require_native_conv_hardware()

    def test_queued_geometry_is_ordered_and_hash_gated(self):
        engine = user_dma_core.UnifiedEngine.__new__(user_dma_core.UnifiedEngine)
        engine.hw_version = 0xDCACD7AA
        engine._allow_unknown_conv_hardware = False
        engine._allow_unknown_queue_config_hardware = False
        engine.conv_geometry_mode = user_dma_core.CONV_GEOMETRY_QUEUE_CONFIG
        with self.assertRaisesRegex(RuntimeError, "live-csr-v1"):
            engine.require_conv_geometry_hardware()

        engine.hw_version = 0xD93EEA82
        engine.require_conv_geometry_hardware()
        engine.hw_version = 0x9EF15FC1
        engine.require_conv_geometry_hardware()
        engine.hw_version = 0x663DE8D5
        engine.require_conv_geometry_hardware()
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

    def test_gather_if8_has_a_separate_fail_closed_capability_gate(self):
        engine = user_dma_core.UnifiedEngine.__new__(user_dma_core.UnifiedEngine)
        engine.hw_version = 0x9EF15FC1
        engine._allow_unknown_gather_if8_hardware = False
        with self.assertRaisesRegex(RuntimeError, "scale address too late"):
            engine.require_gather_if8_hardware()
        engine.hw_version = 0x77E8ADF3
        engine.require_gather_if8_hardware()
        engine.hw_version = 0x9EF15FC1
        engine._allow_unknown_gather_if8_hardware = True
        engine.require_gather_if8_hardware()

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
        # and direct backend as the 60-convolution artifact without a checkpoint.
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
        }

    def test_offline_image_and_direct_backend_do_not_capture_at_runtime(self):
        payload = self._one_conv_payload()
        payload["hardware"] = compile_precompiled_hardware(payload)
        validate_precompiled_hardware(payload)
        hardware = payload["hardware"]
        self.assertEqual(len(hardware["entries"]), 1)
        self.assertGreater(hardware["params_image"].numel(), 0)
        self.assertGreater(hardware["program_image"].numel(), 0)
        self.assertEqual(hardware["geometry_abi"], "conv-config-inst-v1")
        self.assertEqual(
            hardware["abi"], "andromeda-yolov5-precompiled-v4")
        self.assertEqual(hardware["entries"][0]["queue_config_count"], 4)
        entry = hardware["entries"][0]
        configs = _scan_queue_configs(
            hardware["program_image"], entry["program_offset"],
            entry["program_size"])
        expected_configs = []
        _out_h, _out_w, _oc_chunk, tiles = \
            user_dma_core.plan_conv2d_layer_tiles(
                c_in=3, oc_count=8, in_h=32, in_w=40,
                kernel_h=6, kernel_w=6, stride_s=2, pad=2,
                gather=True, bias_enabled=True)
        for _start, _stop, th, tw, _win_h, win_w in \
                user_dma_core.conv2d_tile_geometry_groups(tiles):
            expected_configs.append(user_dma_core.pack_conv2d_geometry_words(
                out_w=tw, out_h=th, ct=1,
                kernel_w=6, kernel_h=6, oc_count=8,
                row_stride=win_w, col_stride=1,
                pix_col_step=2, pix_row_step=2 * win_w,
                gather=True, c_in=3, blocks_per_pixel=16, chunks=2))
        self.assertEqual(configs, expected_configs)

        rebuilt = compile_precompiled_hardware(self._one_conv_payload())
        self.assertEqual(rebuilt["params_sha256"], hardware["params_sha256"])
        self.assertEqual(rebuilt["program_sha256"], hardware["program_sha256"])
        self.assertEqual(rebuilt["entries"], hardware["entries"])

        class FakeRuntimeEngine:
            def __init__(self):
                self.h2c_device = "fake-h2c"
                self.c2h_device = "fake-c2h"
                self.hw_version = 0x77E8ADF3
                self.conv_geometry_mode = user_dma_core.CONV_GEOMETRY_QUEUE_CONFIG
                self._allow_unknown_conv_hardware = False
                self._allow_unknown_queue_config_hardware = False
                self._allow_unknown_gather_if8_hardware = False
                self.writes = []
                self.starts = []
                self.registers = []
                self.busy_status = []
                self.interrupt_status = []

            def require_native_conv_hardware(self):
                return user_dma_core.UnifiedEngine.require_native_conv_hardware(self)

            def require_queue_conv_config_hardware(self):
                return user_dma_core.UnifiedEngine.require_queue_conv_config_hardware(self)

            def require_gather_if8_hardware(self):
                return user_dma_core.UnifiedEngine.require_gather_if8_hardware(self)

            def dma_write(self, _device, address, value, size):
                self.writes.append((address, size))
                return size

            def is_queue_busy(self):
                return self.busy_status.pop(0) if self.busy_status else False

            def write_reg32(self, address, value):
                self.registers.append((address, value))

            def start_execute_from_dram(self, address):
                self.starts.append(address)
                # Exercise delayed completion and a fast program whose busy
                # pulse is not necessarily visible through PCIe.
                self.interrupt_status = [
                    user_dma_core.INT_CAUSE_NONE,
                    user_dma_core.INT_CAUSE_HALT,
                ]

            def read_reg32(self, address):
                if (address == user_dma_core.UE_INT_REG
                        and self.interrupt_status):
                    return self.interrupt_status.pop(0)
                if address == user_dma_core.UE_LATENCY_COUNT_ADDR:
                    raise AssertionError(
                        "model backend bypassed the latency wrapper")
                return 0

            def read_latency_cycles(self):
                self.latency_wrapper_calls = getattr(
                    self, "latency_wrapper_calls", 0) + 1
                return 7 * user_dma_core.UE_PIPELINE_COUNTER_CLK_DIV

            def dma_read(self, _device, _address, value, size):
                value.zero_()
                return size

            def start_capture(self):  # pragma: no cover - explicit tripwire.
                raise AssertionError("direct-bin runtime must not capture")

        engine = FakeRuntimeEngine()
        backend = PrecompiledAndromedaBackend(engine, payload)
        forbidden = AssertionError(
            "direct-bin runtime must not repack static data or recapture")
        with mock.patch.object(
                user_dma_core, "conv2d_pack_weight_stream",
                side_effect=forbidden), mock.patch.object(
                user_dma_core, "conv2d_pack_weight_stream_gather",
                side_effect=forbidden), mock.patch.object(
                user_dma_core, "conv2d_pack_scale_stream",
                side_effect=forbidden), mock.patch.object(
                user_dma_core, "conv2d_pack_scale_stream_gather",
                side_effect=forbidden), mock.patch.object(
                user_dma_core, "conv2d_pack_bias_stream",
                side_effect=forbidden), mock.patch.object(
                user_dma_core.UnifiedEngine, "run_conv2d_layer",
                side_effect=forbidden):
            result = backend.conv_prepared(
                "conv", torch.zeros(3, 32, 40, dtype=torch.bfloat16), None,
                stride=2, pad=2, dilation=1, activate=True)
        self.assertEqual(tuple(result.shape), (8, 16, 20))
        # Initial immutable params/program uploads plus one dynamic activation.
        self.assertEqual(len(engine.writes), 3)
        self.assertEqual(len(engine.starts), 1)
        self.assertEqual(backend.static_dram_load_writes, 2)
        self.assertEqual(
            backend.static_dram_load_bytes,
            hardware["params_image"].numel()
            + hardware["program_image"].numel())
        self.assertEqual(
            backend.cycles["conv"],
            7 * user_dma_core.UE_PIPELINE_COUNTER_CLK_DIV)
        self.assertEqual(engine.latency_wrapper_calls, 1)
        self.assertEqual(engine.registers, [(user_dma_core.UE_INT_REG, 1)])
        self.assertFalse(any(address in (
            user_dma_core.UE_CONV_GEOM_ADDR, user_dma_core.UE_CONV_CTRL_ADDR,
            user_dma_core.UE_CONV_STRIDE_ADDR, user_dma_core.UE_CONV_PIXSTEP_ADDR)
            for address, _value in engine.registers))

        incompatible = copy.deepcopy(payload)
        incompatible["hardware"]["abi"] = "andromeda-yolov5-precompiled-v3"
        stale_engine = FakeRuntimeEngine()
        with self.assertRaisesRegex(RuntimeError, "incompatible fixed-address ABI"):
            PrecompiledAndromedaBackend(stale_engine, incompatible)
        self.assertEqual(stale_engine.writes, [])

        old_engine = FakeRuntimeEngine()
        old_engine.hw_version = 0xDCACD7AA
        with self.assertRaisesRegex(RuntimeError, "live-csr-v1"):
            PrecompiledAndromedaBackend(old_engine, payload)
        self.assertEqual(old_engine.writes, [])

        stale_gather_engine = FakeRuntimeEngine()
        stale_gather_engine.hw_version = 0x9EF15FC1
        with self.assertRaisesRegex(RuntimeError, "scale address too late"):
            PrecompiledAndromedaBackend(stale_gather_engine, payload)
        self.assertEqual(stale_gather_engine.writes, [])

        backend.timeout_s = 0.001
        with self.assertRaisesRegex(TimeoutError, "never reported HALT"):
            backend._wait_strict()
        engine.dma_read = lambda _device, _address, _value, size: size - 2
        with self.assertRaisesRegex(RuntimeError, "read .* of"):
            backend._read_bf16(0xB0000000, 64)

        broken = copy.deepcopy(payload)
        program = broken["hardware"]["program_image"].clone()
        config_offsets = [
            offset for offset in range(0, program.numel(), 32)
            if (int(program[offset + 1]) & 0xF) == user_dma_core.INSTRUCTION_CONFIG
        ]
        self.assertTrue(config_offsets)
        program[config_offsets[0] + 20] = 1  # reserved word w5
        broken["hardware"]["program_image"] = program
        broken["hardware"]["program_sha256"] = hashlib.sha256(
            program.numpy().tobytes()).hexdigest()
        with self.assertRaisesRegex(RuntimeError, "reserved"):
            validate_precompiled_hardware(broken)

        broken = copy.deepcopy(payload)
        broken["hardware"]["entries"][0]["queue_config_count"] += 1
        with self.assertRaisesRegex(RuntimeError, "count"):
            validate_precompiled_hardware(broken)

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
        }
        hardware = compile_precompiled_hardware(payload)
        entry = hardware["entries"][0]
        configs = _scan_queue_configs(
            hardware["program_image"], entry["program_offset"],
            entry["program_size"])
        self.assertEqual(len(configs), 1)
        expected = user_dma_core.pack_conv2d_geometry_words(
            out_w=4, out_h=4, ct=1, kernel_w=1, kernel_h=1,
            oc_count=8, row_stride=4, col_stride=1,
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
