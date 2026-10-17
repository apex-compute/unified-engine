"""Independent BF16 memory execution and offline ISA checks for hard VQ."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parents[1] / "models" / "yolov5s"):
    sys.path.insert(0, str(path))
from bigcodec_quantizer import (CODEBOOK_SIZE, FEATURE_DIM, WIDTH,
                               bit_reversed_indices, decode_split_tokens,
                               emit_positive_mask, emit_quantizer,
                               prepare_quantizer, quantizer_scratch_bytes)
import user_dma_core as udc
from yolov5_precompiled import _WholeGraphEngine, _instruction_types


def bf16(value):
    return value.to(torch.bfloat16).float()


class MemoryEngine:
    """Device-memory interpreter, rounding every externally stored BF16 result."""
    def __init__(self):
        self.regions = {}
        self.cursor = 0x90000000
        self.sram = torch.zeros(0x100000 // 2, dtype=torch.bfloat16)
        self.matmul_shapes = []

    def allocate(self, value, *, alignment=128):
        self.cursor = (self.cursor + alignment - 1) // alignment * alignment
        address = self.cursor
        self.regions[address] = value.flatten().to(torch.bfloat16).clone()
        self.cursor += value.numel() * 2
        return address

    def view(self, address, count):
        for base, value in self.regions.items():
            if base <= address and address + count * 2 <= base + value.numel() * 2:
                assert address % 2 == 0
                return value[(address - base) // 2:][:count]
        raise AssertionError(f"Unallocated device range {address:#x}, {count} values")

    def matmat_mul_core(self, *, M, K, N, A_DRAM_ADDR, B_DRAM_ADDR,
                        OUTPUT_DRAM_ADDR, C_DRAM_ADDR=None, clamp_enable=False,
                        clamp_min=0.0, clamp_max=float("inf")):
        assert all(a % 128 == 0 for a in (A_DRAM_ADDR, B_DRAM_ADDR, OUTPUT_DRAM_ADDR))
        self.matmul_shapes.append((M, K, N))
        a = self.view(A_DRAM_ADDR, M * K).float().reshape(M, K)
        b = self.view(B_DRAM_ADDR, N * K).float().reshape(N, K)
        result = a @ b.t()
        if C_DRAM_ADDR is not None:
            result += self.view(C_DRAM_ADDR, N).float()
        if clamp_enable:
            result = result.clamp(clamp_min, clamp_max)
        self.view(OUTPUT_DRAM_ADDR, M * N).copy_(result.flatten())

    def activation_core(self, *, M, N, A_DRAM_ADDR, OUTPUT_DRAM_ADDR,
                        IDENTITY_DRAM_ADDR, activation, clamp_min=0.0,
                        clamp_max=float("inf")):
        assert activation == "clamp"
        identity = self.view(IDENTITY_DRAM_ADDR, N * N).float().reshape(N, N)
        torch.testing.assert_close(identity, torch.eye(N), rtol=0, atol=0)
        value = self.view(A_DRAM_ADDR, M * N).float().clamp(clamp_min, clamp_max)
        self.view(OUTPUT_DRAM_ADDR, M * N).copy_(value)

    def eltwise_core_dram(self, *, M, N, dram_a, dram_b, dram_out, mode, scalar=None):
        assert N % WIDTH == 0
        a = self.view(dram_a, M * N).float()
        if mode == udc.UE_MODE.MUL_BROADCAST:
            assert dram_b is None
            result = a * scalar
        else:
            b = self.view(dram_b, M * N).float()
            fn = {udc.UE_MODE.ELTWISE_MUL: torch.mul,
                  udc.UE_MODE.ELTWISE_SUB: torch.sub,
                  udc.UE_MODE.ELTWISE_ADD: torch.add}[mode]
            result = fn(a, b)
        self.view(dram_out, M * N).copy_(result)

    def accelerator_memcpy(self, source, destination, size):
        assert source % 128 == destination % 128 == size % 128 == 0
        self.view(destination, size // 2).copy_(self.view(source, size // 2).clone())

    def accelerator_memory_to_sram(self, address, sram, elements):
        assert address % 128 == sram % 128 == 0
        self.sram[sram // 2:][:elements].copy_(self.view(address, elements))

    def sram_to_accelerator_memory(self, sram, address, elements):
        assert address % 128 == sram % 128 == 0
        self.view(address, elements).copy_(self.sram[sram // 2:][:elements])

    def sram_view(self, address, count):
        assert address >= 0 and address % 128 == 0 and count % 64 == 0
        assert address // 0x80000 == (address + count * 2 - 1) // 0x80000
        assert address + count * 2 <= 0x100000
        return self.sram[address // 2:][:count]

    def broadcast_mul(self, scalar, sram_start_addr, sram_wb_addr, element_size):
        assert sram_start_addr < 0x80000
        self.sram_view(sram_wb_addr, element_size).copy_(
            self.sram_view(sram_start_addr, element_size).float() * scalar)

    def broadcast_add(self, scalar, sram_start_addr, sram_wb_addr, element_size):
        assert sram_start_addr < 0x80000
        self.sram_view(sram_wb_addr, element_size).copy_(
            self.sram_view(sram_start_addr, element_size).float() + scalar)

    def eltwise_add_core(self, a, b, output, count):
        assert a // 0x80000 != b // 0x80000
        self.sram_view(output, count).copy_(
            self.sram_view(a, count).float() + self.sram_view(b, count).float())

    def eltwise_mul_core(self, a, b, output, count):
        assert a // 0x80000 != b // 0x80000
        self.sram_view(output, count).copy_(
            self.sram_view(a, count).float() * self.sram_view(b, count).float())

    def eltwise_sub_core(self, a, b, output, count):
        assert a < 0x80000 <= b
        self.sram_view(output, count).copy_(
            self.sram_view(a, count).float() - self.sram_view(b, count).float())

    def start_queue_for_maxpool2d_operation(self, *, act_sram_start_addr,
            output_sram_wb_addr, kernel_w, kernel_h, out_w, out_h, w_pad, stride_s):
        assert act_sram_start_addr < 0x80000 and kernel_w > 0 and kernel_h > 0
        windows = []
        for y in range(out_h):
            for x in range(out_w):
                samples = [self.sram_view(act_sram_start_addr +
                    ((y * stride_s + ky) * w_pad + x * stride_s + kx) * 128, 64).float()
                    for ky in range(kernel_h) for kx in range(kernel_w)]
                windows.append(torch.stack(samples).amax(0))
        self.sram_view(output_sram_wb_addr, out_h * out_w * 64).copy_(torch.cat(windows))

    float_to_bf19 = staticmethod(udc.UnifiedEngine.float_to_bf19)

    def start_queue_for_bf16_matvec_operation(self, *, max_clear_en, fmax_context_addr,
            vector_sram_start_addr, matrix_sram_start_addr, output_sram_wb_addr,
            K, N, lalu_mode, lalu_scalar):
        assert lalu_mode == udc.LALU_MODE.MODE_RSQRT
        assert lalu_scalar == self.float_to_bf19(1.0)
        a = self.sram[vector_sram_start_addr // 2:][:K].float()
        b = self.sram[matrix_sram_start_addr // 2:][:N * K].float().reshape(N, K)
        self.sram[output_sram_wb_addr // 2:][:N].copy_(torch.rsqrt(b @ a))

    def bf16_transpose_core(self, *, M, N, INPUT_DRAM_ADDR, OUTPUT_DRAM_ADDR,
                            IDENTITY_DRAM_ADDR):
        value = self.view(INPUT_DRAM_ADDR, M * N).reshape(M, N).t().contiguous()
        self.view(OUTPUT_DRAM_ADDR, M * N).copy_(value.flatten())


def fixture(codebook=None, frames=3, *, center_scores=False, compensated_codebook=False):
    generator = torch.Generator().manual_seed(417)
    if codebook is None:
        codebook = torch.randn(CODEBOOK_SIZE, 8, generator=generator)
    in_weight = torch.zeros(8, FEATURE_DIM)
    in_weight[:, :8] = torch.eye(8)
    out_weight = torch.randn(FEATURE_DIM, 8, generator=generator) * 0.05
    module = SimpleNamespace(
        codebook=SimpleNamespace(weight=codebook),
        in_proj=SimpleNamespace(weight=in_weight, bias=torch.zeros(8)),
        out_proj=SimpleNamespace(weight=out_weight, bias=torch.zeros(FEATURE_DIM)))
    engine = MemoryEngine()
    plan = prepare_quantizer(module, engine, source_address=0xA0000000,
                             destination_address=0xA0100000, token_address=0xA0200000,
                             scratch_address=0xB0000000, frames=frames,
                             center_scores=center_scores,
                             compensated_codebook=compensated_codebook)
    for address, count in ((plan.source_address, frames * FEATURE_DIM),
                           (plan.destination_address, frames * FEATURE_DIM),
                           (plan.token_address, frames * WIDTH),
                           (plan.scratch_address, plan.scratch_bytes // 2)):
        # Poison scratch to expose missing padding/initialization.
        engine.regions[address] = torch.full((count,), float("nan"), dtype=torch.bfloat16)
    return engine, plan, module


class QuantizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_split_ids_roundtrip(self):
        ids = torch.arange(CODEBOOK_SIZE)
        values = torch.zeros(CODEBOOK_SIZE, WIDTH, dtype=torch.bfloat16)
        values[:, 0] = (ids & 255).to(torch.bfloat16)
        values[:, 1] = (ids >> 8).to(torch.bfloat16)
        torch.testing.assert_close(decode_split_tokens(values), ids)
        for lane, bad in ((0, 256), (1, 32), (0, 0.5), (2, 1), (0, float("nan"))):
            invalid = values[:1].clone()
            invalid[0, lane] = bad
            with self.assertRaises(ValueError):
                decode_split_tokens(invalid)

    def test_ieee_positive_mask_covers_every_finite_bf16_value(self):
        # Ideal BF16 arithmetic, including subnormals; device flushing is below.
        values = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
        values[~torch.isfinite(values)] = 0
        engine = MemoryEngine()
        source = engine.allocate(values)
        output = engine.allocate(torch.zeros_like(values))
        identity = engine.allocate(torch.eye(WIDTH))
        emit_positive_mask(engine, source=source, destination=output,
                           elements=values.numel(), identity_address=identity)
        torch.testing.assert_close(engine.view(output, values.numel()).float(),
                                   (values.float() > 0).float(), rtol=0, atol=0)

    def test_rk_mask_flushes_only_positive_subnormal_gaps(self):
        class FlushMaskEngine(MemoryEngine):
            """Model the measured mask contract, not cycle-accurate hardware."""
            def broadcast_mul(self, scalar, sram_start_addr, sram_wb_addr, element_size):
                value = self.sram_view(sram_start_addr, element_size).float()
                value = value.masked_fill(value.abs() < torch.finfo(torch.bfloat16).tiny, 0)
                self.sram_view(sram_wb_addr, element_size).copy_(value * scalar)

        # RK 0xdf0749de measured 127 mismatches over all 65536 patterns, all
        # positive subnormals. Those score gaps therefore retain the left entry.
        values = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
        values[~torch.isfinite(values)] = 0
        engine = FlushMaskEngine()
        source = engine.allocate(values)
        output = engine.allocate(torch.zeros_like(values))
        identity = engine.allocate(torch.eye(WIDTH))
        emit_positive_mask(engine, source=source, destination=output,
                           elements=values.numel(), identity_address=identity)
        actual = engine.view(output, values.numel()).float()
        expected = (values.float() >= torch.finfo(torch.bfloat16).tiny).float()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        differs = actual != (values.float() > 0).float()
        self.assertEqual(int(differs.sum()), 127)
        self.assertTrue((values[differs].float() > 0).all())
        self.assertTrue((values[differs].float() < torch.finfo(torch.bfloat16).tiny).all())
        self.assertTrue(torch.isfinite(actual).all())

    def test_full_8192_search_matches_independent_bf16_argmax(self):
        engine, plan, module = fixture(frames=3)
        z = torch.zeros(3, FEATURE_DIM)
        z[:, :8] = torch.randn(3, 8, generator=torch.Generator().manual_seed(911))
        engine.view(plan.source_address, z.numel()).copy_(z.flatten())
        emit_quantizer(engine, plan)
        # Independent full score matrix/argmax; no tournament operations here.
        projected = bf16(z[:, :8])
        inverse = bf16(torch.rsqrt(bf16(bf16(projected.square()).sum(1, keepdim=True))))
        normalized = bf16(projected * inverse)
        codebook = bf16(F.normalize(module.codebook.weight, dim=1))
        scores = bf16(normalized @ codebook.t())
        expected_ids = scores.argmax(1)
        actual_ids = decode_split_tokens(engine.view(plan.token_address, 3 * WIDTH).reshape(3, WIDTH))
        torch.testing.assert_close(actual_ids, expected_ids)
        selected = bf16(module.codebook.weight)[expected_ids]
        expected = bf16(selected @ bf16(module.out_proj.weight).t())
        actual = engine.view(plan.destination_address, expected.numel()).float().reshape_as(expected)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertIn((1, WIDTH, CODEBOOK_SIZE), engine.matmul_shapes)

    def test_normalization_global_ties_and_zero_input(self):
        codebook = torch.zeros(CODEBOOK_SIZE, 8)
        codebook[:, 0] = -1
        # Halving an unreordered codebook would choose4096 over2048 on a tie.
        # Different magnitudes also catch an accidental unnormalized lookup.
        codebook[2048, 0] = 1.25
        codebook[4096, 0] = 2.5
        engine, plan, module = fixture(codebook, frames=2)
        source = torch.zeros(2, FEATURE_DIM)
        source[0, 0] = 3
        engine.view(plan.source_address, source.numel()).copy_(source.flatten())
        emit_quantizer(engine, plan)
        actual = decode_split_tokens(engine.view(plan.token_address, 2 * WIDTH).reshape(2, WIDTH))
        torch.testing.assert_close(actual, torch.tensor([2048, 0]))
        expected = bf16(bf16(codebook)[actual] @ bf16(module.out_proj.weight).t())
        torch.testing.assert_close(engine.view(plan.destination_address, expected.numel()).float().reshape_as(expected),
                                   expected, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(expected).all())

    def test_centered_score_epilogue_resolves_rounding_tie_and_keeps_true_ties(self):
        codebook = torch.zeros(CODEBOOK_SIZE, 8)
        codebook[:, 0] = -1
        angles = torch.tensor([0.14371448755264282, 0.14611472189426422])
        codebook[5:7, 0] = angles.cos()
        codebook[5:7, 1] = angles.sin()
        codebook[7] = codebook[6]  # A genuine tie still selects the lower ID.
        normalized_query = bf16(torch.tensor([1., 1.]) / 2 ** .5)
        scores = normalized_query @ bf16(F.normalize(codebook[5:7], dim=1))[:, :2].T
        self.assertGreater(float(scores[1]), float(scores[0]))
        self.assertEqual(float(bf16(scores)[0]), float(bf16(scores)[1]))
        for centered, expected in ((False, [5, 0]), (True, [6, 0])):
            with self.subTest(centered=centered):
                engine, plan, _ = fixture(codebook, frames=2, center_scores=centered)
                source = torch.zeros(2, FEATURE_DIM)
                source[0, :2] = 1
                engine.view(plan.source_address, source.numel()).copy_(source.flatten())
                emit_quantizer(engine, plan)
                ids = decode_split_tokens(engine.view(plan.token_address, 2 * WIDTH).reshape(2, WIDTH))
                torch.testing.assert_close(ids, torch.tensor(expected))
                self.assertEqual("score_bias" in plan.constants, centered)
                if centered:
                    self.assertTrue((engine.view(plan.constants["score_bias"], CODEBOOK_SIZE) == -1).all())

    def test_centering_default_preserves_legacy_image_and_validates_flag(self):
        old, old_plan, _ = fixture(frames=1)
        explicit, explicit_plan, module = fixture(frames=1, center_scores=False)
        self.assertEqual(old.cursor, explicit.cursor)
        self.assertEqual(old_plan.constants, explicit_plan.constants)
        self.assertEqual(old.regions.keys(), explicit.regions.keys())
        for address in old.regions:
            torch.testing.assert_close(old.regions[address], explicit.regions[address],
                                       rtol=0, atol=0, equal_nan=True)
        before = explicit.cursor
        with self.assertRaisesRegex(ValueError, "center_scores must be a bool"):
            prepare_quantizer(module, explicit, source_address=old_plan.source_address,
                              destination_address=old_plan.destination_address,
                              token_address=old_plan.token_address,
                              scratch_address=old_plan.scratch_address, frames=1,
                              center_scores=1)
        self.assertEqual(explicit.cursor, before)

    def test_compensated_codebook_packing_and_normalization(self):
        old, old_plan, _ = fixture(frames=3, center_scores=True)
        engine, plan, module = fixture(frames=3, center_scores=True, compensated_codebook=True)
        self.assertEqual(old.cursor, engine.cursor)
        self.assertEqual(old_plan.constants, plan.constants)
        weights = engine.view(plan.constants["in_weight"], WIDTH * FEATURE_DIM).reshape(WIDTH, FEATURE_DIM)
        biases = engine.view(plan.constants["in_bias"], WIDTH)
        torch.testing.assert_close(weights[:8], weights[8:16], rtol=0, atol=0)
        torch.testing.assert_close(biases[:8], biases[8:16], rtol=0, atol=0)
        self.assertTrue((weights[16:] == 0).all())
        sum_weight = engine.view(plan.constants["sum_weight"], WIDTH * WIDTH).reshape(WIDTH, WIDTH)
        self.assertTrue((sum_weight[:, :8] == 1).all())
        self.assertTrue((sum_weight[:, 8:] == 0).all())
        score_weight = engine.view(plan.constants["score_weight"], CODEBOOK_SIZE * WIDTH).float().reshape(CODEBOOK_SIZE, WIDTH)
        original = F.normalize(module.codebook.weight, dim=1)[bit_reversed_indices()]
        original_error = (score_weight[:, :8] - original).double().norm()
        residual_error = (score_weight[:, :8] + score_weight[:, 8:16] - original).double().norm()
        self.assertLess(float(residual_error), float(original_error) / 100)
        self.assertTrue((score_weight[:, 16:] == 0).all())

        source = torch.zeros(3, FEATURE_DIM)
        source[1:, :8] = torch.randn(2, 8, generator=torch.Generator().manual_seed(913))
        engine.view(plan.source_address, source.numel()).copy_(source.flatten())
        emit_quantizer(engine, plan)
        normalized = engine.view(plan.scratch["normalized"], 3 * WIDTH).float().reshape(3, WIDTH)
        torch.testing.assert_close(normalized[:, :8], normalized[:, 8:16], rtol=0, atol=0)
        # A duplicated query must retain the norm of eight coordinates.
        projected = bf16(source[:, :8])
        inverse = bf16(torch.rsqrt(bf16(bf16(projected.square()).sum(1, keepdim=True)).clamp_min(1e-24)))
        expected_query = bf16(projected * inverse)
        torch.testing.assert_close(normalized[:, :8], expected_query, rtol=0, atol=0)
        scores = bf16(normalized @ score_weight.T - 1)
        ordered_ids = bit_reversed_indices()
        # Undo storage order before argmax so equal scores choose the lowest ID.
        expected_ids = scores[:, ordered_ids].argmax(1)
        actual_ids = decode_split_tokens(engine.view(plan.token_address, 3 * WIDTH).reshape(3, WIDTH))
        torch.testing.assert_close(actual_ids, expected_ids)
        self.assertEqual(int(actual_ids[0]), 0)

    def test_compensated_codebook_changes_no_instructions_and_rejects_invalid_flag(self):
        programs = []
        for compensated in (False, True):
            image, plan, module = fixture(frames=1, center_scores=True,
                                          compensated_codebook=compensated)
            engine = _WholeGraphEngine(0x98000000)
            with patch.object(udc, "UE_AXI_DATA_WIDTH_BITS", 256), contextlib.redirect_stdout(io.StringIO()):
                engine.start_capture()
                emit_quantizer(engine, plan)
                engine.generate_instruction_halt()
                engine.stop_capture()
            programs.append(b"".join(instruction.get_bytes() for instruction in engine.capture_buffer))
        self.assertEqual(programs[0], programs[1])
        before = image.cursor
        with self.assertRaisesRegex(ValueError, "compensated_codebook must be a bool"):
            prepare_quantizer(module, image, source_address=plan.source_address,
                              destination_address=plan.destination_address,
                              token_address=plan.token_address,
                              scratch_address=plan.scratch_address, frames=1,
                              compensated_codebook=1)
        self.assertEqual(image.cursor, before)

    def test_order_scratch_bounds_and_reject_overlap(self):
        order = bit_reversed_indices()
        torch.testing.assert_close(order[order], torch.arange(CODEBOOK_SIZE))
        self.assertEqual(quantizer_scratch_bytes(2) - quantizer_scratch_bytes(1), 384)
        engine, plan, module = fixture(frames=1)
        with self.assertRaises(ValueError):
            prepare_quantizer(module, engine, source_address=plan.source_address,
                              destination_address=plan.source_address,
                              token_address=plan.token_address,
                              scratch_address=plan.scratch_address, frames=1)

    def test_reject_stale_weight_norm_parameters(self):
        engine, plan, module = fixture(frames=1)
        module.in_proj.weight_g = torch.ones(8, 1)
        with self.assertRaisesRegex(ValueError, "remove_weight_norm=True"):
            prepare_quantizer(module, engine, source_address=plan.source_address,
                              destination_address=plan.destination_address,
                              token_address=plan.token_address,
                              scratch_address=plan.scratch_address, frames=1)

    def test_captured_isa_has_one_halt_no_cpu_or_swi(self):
        _, plan, _ = fixture(frames=1)
        engine = _WholeGraphEngine(0x98000000)
        with patch.object(udc, "UE_AXI_DATA_WIDTH_BITS", 256), contextlib.redirect_stdout(io.StringIO()):
            engine.start_capture()
            emit_quantizer(engine, plan)
            engine.generate_instruction_halt()
            engine.stop_capture()
        issues = udc.check_isa_jumps(engine.capture_buffer, 0x98000000, name="BigCodec VQ")
        self.assertEqual(issues, [])
        raw = b"".join(instruction.get_bytes() for instruction in engine.capture_buffer)
        types = _instruction_types(raw)
        self.assertEqual(types.count(udc.INSTRUCTION_HALT), 1)
        self.assertNotIn(udc.INSTRUCTION_SWI, types)
        self.assertEqual(len(raw) % 64, 0)
        self.assertGreater(len(types), 100)


if __name__ == "__main__":
    unittest.main()
