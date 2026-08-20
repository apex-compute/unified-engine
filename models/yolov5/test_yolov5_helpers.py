"""Host-only regression tests for the YOLOv5 integration."""

import copy
import hashlib
import json
from pathlib import Path
import sys
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
import yolov5_test
from yolov5_artifact import (
    artifact_variant,
    get_canonical_artifact,
)
from yolov5_precompiled import (
    compile_precompiled_hardware,
    PrecompiledAndromedaBackend,
    validate_precompiled_hardware,
)
from yolov5_common import (
    box_iou_one_to_many,
    decode_yolov5,
    fold_conv_bn,
    get_yolov5_variant,
    non_max_suppression,
    quantize_conv_if4,
)


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
        self.assertEqual(artifact.params_bytes, 16_082_992)
        self.assertEqual(artifact.program_bytes, 43_136)
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
            "hardware_version": "0x9ef15fc1",
            "precompiled": True,
            "artifact_version": 3,
            "artifact": "/tmp/yolov5n-andromeda.bin",
        }
        text = "TEST_RESULT:" + json.dumps(result, separators=(",", ":"))
        self.assertTrue(model_auto_test._check_yolov5n(text)[0])
        self.assertTrue(model_auto_test._check_yolov5n_single_bin(text)[0])
        self.assertFalse(model_auto_test._check_yolov5(text)[0])


class QuantizationTests(unittest.TestCase):
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
            allow_unknown_queue_config_hardware=False)
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

    def test_detect_head_keeps_255_channels_together(self):
        for channels in (64, 128):
            with self.subTest(channels=channels):
                _, _, chunk, _ = user_dma_core.plan_conv2d_layer_tiles(
                    c_in=channels, oc_count=255, in_h=80, in_w=80,
                    kernel_h=1, kernel_w=1, stride_s=1, pad=0)
                self.assertEqual(chunk, 255)

    def test_gather_planner_uses_rewound_scale_bram(self):
        channel_plan = user_dma_core.plan_conv2d_layer_tiles(
            c_in=3, oc_count=32, in_h=640, in_w=640,
            kernel_h=6, kernel_w=6, stride_s=2, pad=2,
            gather=False)
        gather_plan = user_dma_core.plan_conv2d_layer_tiles(
            c_in=3, oc_count=32, in_h=640, in_w=640,
            kernel_h=6, kernel_w=6, stride_s=2, pad=2,
            gather=True)
        self.assertEqual(gather_plan[2], 32)
        self.assertLess(len(gather_plan[3]), len(channel_plan[3]))
        self.assertGreater(gather_plan[3][0][2] * gather_plan[3][0][3], 6)

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

    def test_gather_geometry_rejects_inconsistent_or_oversized_chunks(self):
        engine = user_dma_core.UnifiedEngine.__new__(user_dma_core.UnifiedEngine)
        engine.is_capture_on = False
        engine.write_reg32 = lambda _address, _value: None
        base = dict(
            out_w=1, out_h=1, ct=1, kernel_w=1, kernel_h=1,
            oc_count=2, row_stride=1, col_stride=1,
            pix_col_step=1, pix_row_step=1, gather=True, c_in=3)
        with self.assertRaisesRegex(ValueError, "four RTL"):
            engine.write_conv2d_geometry_registers(
                **base, blocks_per_pixel=10, chunks=5)
        with self.assertRaisesRegex(ValueError, "!= oc_count"):
            engine.write_conv2d_geometry_registers(
                **base, blocks_per_pixel=3, chunks=2)


class PrecompiledBinTests(unittest.TestCase):
    @staticmethod
    def _one_conv_payload():
        # A deliberately small graph exercises the same fixed-address compiler
        # and direct backend as the 60-convolution artifact without a checkpoint.
        codes_shape = [64, 64, 1, 1]
        code_count = 64 * 64
        return {
            "model": {"input_shape": [64, 4, 4]},
            "operations": [{
                "op": "conv", "name": "conv", "inputs": ["input"],
                "output": "conv", "output_shape": [64, 4, 4],
                "stride": 1, "pad": 0, "dilation": 1,
                "activate": True,
            }],
            "weights": {
                "conv": {
                    "codes_packed": torch.zeros(
                        (code_count + 1) // 2, dtype=torch.uint8),
                    "codes_shape": codes_shape,
                    "block_scales": torch.ones(
                        64, 1, dtype=torch.bfloat16),
                    "bias": torch.zeros(64, dtype=torch.bfloat16),
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
        self.assertGreater(hardware["entries"][0]["queue_config_count"], 0)

        class FakeRuntimeEngine:
            def __init__(self):
                self.h2c_device = "fake-h2c"
                self.c2h_device = "fake-c2h"
                self.hw_version = 0xD93EEA82
                self.conv_geometry_mode = user_dma_core.CONV_GEOMETRY_QUEUE_CONFIG
                self._allow_unknown_conv_hardware = False
                self._allow_unknown_queue_config_hardware = False
                self.writes = []
                self.starts = []
                self.registers = []
                self.busy_status = []
                self.interrupt_status = []

            def require_native_conv_hardware(self):
                return user_dma_core.UnifiedEngine.require_native_conv_hardware(self)

            def require_queue_conv_config_hardware(self):
                return user_dma_core.UnifiedEngine.require_queue_conv_config_hardware(self)

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
                    return 7
                return 0

            def dma_read(self, _device, _address, value, size):
                value.zero_()
                return size

            def start_capture(self):  # pragma: no cover - explicit tripwire.
                raise AssertionError("direct-bin runtime must not capture")

        engine = FakeRuntimeEngine()
        backend = PrecompiledAndromedaBackend(engine, payload)
        result = backend.conv_prepared(
            "conv", torch.zeros(64, 4, 4, dtype=torch.bfloat16), None,
            stride=1, pad=0, dilation=1, activate=True)
        self.assertEqual(tuple(result.shape), (64, 4, 4))
        # Initial immutable params/program uploads plus one dynamic activation.
        self.assertEqual(len(engine.writes), 3)
        self.assertEqual(len(engine.starts), 1)
        self.assertEqual(backend.static_dram_load_writes, 2)
        self.assertEqual(
            backend.static_dram_load_bytes,
            hardware["params_image"].numel()
            + hardware["program_image"].numel())
        self.assertEqual(backend.latency_counter_ticks["conv"], 7)
        self.assertEqual(
            backend.cycles["conv"],
            7 * user_dma_core.UE_PIPELINE_COUNTER_CLK_DIV)
        self.assertEqual(engine.registers, [(user_dma_core.UE_INT_REG, 1)])
        self.assertFalse(any(address in (
            user_dma_core.UE_CONV_GEOM_ADDR, user_dma_core.UE_CONV_CTRL_ADDR,
            user_dma_core.UE_CONV_STRIDE_ADDR, user_dma_core.UE_CONV_PIXSTEP_ADDR)
            for address, _value in engine.registers))

        old_engine = FakeRuntimeEngine()
        old_engine.hw_version = 0xDCACD7AA
        with self.assertRaisesRegex(RuntimeError, "live-csr-v1"):
            PrecompiledAndromedaBackend(old_engine, payload)
        self.assertEqual(old_engine.writes, [])

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
            torch.zeros(255, 80, 80),
            torch.zeros(255, 40, 40),
            torch.zeros(255, 20, 20),
        ]
        decoded = decode_yolov5(raw, model)
        self.assertEqual(tuple(decoded.shape), (25200, 85))
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
