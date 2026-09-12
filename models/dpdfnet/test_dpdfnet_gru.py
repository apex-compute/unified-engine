"""Exercise production GRU lowering with independent BF16 memory semantics."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import onnx
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))
from dpdfnet_compile import GraphCompiler
from dpdfnet_precompiled import make_layout
import user_dma_core as udc


def bf16(value):
    return value.to(torch.bfloat16).float()


class MemoryEngine:
    def __init__(self):
        self.regions = {}
        self.sram = torch.zeros(0x100000 // 2, dtype=torch.bfloat16)
        self.bias = None
        self.matmul_calls = []
        self.sram_read_calls = []

    def view(self, address, count):
        for base, data in self.regions.items():
            offset = (address - base) // 2
            if address >= base and offset + count <= data.numel():
                return data[offset:offset + count]
        raise AssertionError(f"out-of-bounds memory: {address:#x}, {count}")

    def matmat_mul_core(self, *, M, K, N, A_DRAM_ADDR, B_DRAM_ADDR,
                        OUTPUT_DRAM_ADDR, C_DRAM_ADDR=None,
                        bias_mode="broadcast_N", sigmoid_enable=False):
        self.matmul_calls.append((M, B_DRAM_ADDR, OUTPUT_DRAM_ADDR))
        matrix = self.view(B_DRAM_ADDR, N * K).float().reshape(N, K)
        bias = (self.view(C_DRAM_ADDR, N).float()
                if C_DRAM_ADDR is not None else 0)
        inputs = self.view(A_DRAM_ADDR, M * K).float().reshape(M, K)
        # Execute each hardware matvec separately, including its BF16 boundary.
        result = torch.stack([matrix @ row + bias for row in inputs])
        if sigmoid_enable:
            result = torch.sigmoid(result)
        self.view(OUTPUT_DRAM_ADDR, M * N).copy_(result.flatten())

    def activation_core(self, *, M, N, A_DRAM_ADDR, OUTPUT_DRAM_ADDR,
                        IDENTITY_DRAM_ADDR, activation):
        assert activation == "sigmoid"
        source = self.view(A_DRAM_ADDR, M * N).float()
        self.view(OUTPUT_DRAM_ADDR, M * N).copy_(torch.sigmoid(source))

    def eltwise_core_dram(self, *, M, N, dram_a, dram_b, dram_out, mode):
        left = self.view(dram_a, M * N).float()
        right = self.view(dram_b, M * N).float()
        operation = {
            udc.UE_MODE.ELTWISE_ADD: torch.add,
            udc.UE_MODE.ELTWISE_SUB: torch.sub,
            udc.UE_MODE.ELTWISE_MUL: torch.mul,
        }[mode]
        self.view(dram_out, M * N).copy_(operation(left, right))

    def accelerator_memory_to_sram(self, address, sram_address, elements):
        self.sram_read_calls.append((address, sram_address, elements))
        self.sram_view(sram_address, elements).copy_(self.view(address, elements))

    def sram_to_accelerator_memory(self, sram_address, address, elements):
        self.view(address, elements).copy_(self.sram_view(sram_address, elements))

    def sram_view(self, address, elements):
        assert address % 128 == 0
        assert 0 <= address < address + elements * 2 <= 0x100000
        assert address // 0x80000 == (address + elements * 2 - 1) // 0x80000
        return self.sram[address // 2:address // 2 + elements]

    def accelerator_memory_to_bias_sram(self, address, elements):
        self.bias = self.view(address, elements).float().clone()

    def start_queue_for_bf16_matvec_operation(
            self, *, max_clear_en, fmax_context_addr,
            vector_sram_start_addr, matrix_sram_start_addr, output_sram_wb_addr,
            K, N, bias_enable=False, lalu_mode=udc.LALU_MODE.BYPASS,
            lalu_a=0, lalu_b=0):
        assert max_clear_en == 1 and fmax_context_addr == 0
        assert vector_sram_start_addr < 0x80000 <= matrix_sram_start_addr
        vector = self.sram_view(vector_sram_start_addr, K).float()
        matrix = self.sram_view(matrix_sram_start_addr, N * K).float().reshape(N, K)
        result = matrix @ vector
        if bias_enable:
            result += self.bias[:N]
        if lalu_mode == udc.LALU_MODE.ACT_NO_X:
            assert (lalu_a, lalu_b) == (udc.LALU_ACT_SIGMOID_A,
                                       udc.LALU_ACT_SIGMOID_B)
            result = torch.sigmoid(result)
        else:
            assert lalu_mode == udc.LALU_MODE.BYPASS
        self.sram_view(output_sram_wb_addr, N).copy_(result)

    def sram_eltwise(self, left, right, output, count, operation):
        assert left < 0x80000 <= right
        value = operation(self.sram_view(left, count).float(),
                          self.sram_view(right, count).float())
        self.sram_view(output, count).copy_(value)

    def eltwise_add_core(self, left, right, output, count):
        self.sram_eltwise(left, right, output, count, torch.add)

    def eltwise_sub_core(self, left, right, output, count):
        self.sram_eltwise(left, right, output, count, torch.sub)

    def eltwise_mul_core(self, left, right, output, count):
        self.sram_eltwise(left, right, output, count, torch.mul)

    def broadcast_mul(self, scalar, sram_start_addr, sram_wb_addr, element_size):
        assert sram_start_addr < 0x80000
        value = self.sram_view(sram_start_addr, element_size).float() * scalar
        self.sram_view(sram_wb_addr, element_size).copy_(value)

    def broadcast_add(self, scalar, sram_start_addr, sram_wb_addr, element_size):
        assert sram_start_addr < 0x80000
        value = self.sram_view(sram_start_addr, element_size).float() + scalar
        self.sram_view(sram_wb_addr, element_size).copy_(value)


class DPDFNetGRUTest(unittest.TestCase):
    def test_batched_input_projection_preserves_both_recurrences(self):
        # Include a short sequence, an incomplete tile, and the production size.
        for sequence in (8, 17, 48):
            with self.subTest(sequence=sequence):
                torch.manual_seed(sequence)
                engine = MemoryEngine()
                compiler = GraphCompiler.__new__(GraphCompiler)
                compiler.onnx = onnx
                compiler.layouts = {}
                compiler.emitter = SimpleNamespace(engine=engine)
                cursor = 0x100000

                def allocate(name, shape, value=None):
                    nonlocal cursor
                    layout = make_layout(name, shape, cursor)
                    cursor += layout.size_bytes + 128
                    compiler.layouts[name] = layout
                    if value is None:
                        value = torch.zeros(shape)
                    engine.regions[layout.address] = value.to(
                        torch.bfloat16).flatten().clone()
                    return layout

                x = bf16(torch.randn(sequence, 1, 64) * 0.2)
                w = bf16(torch.randn(2, 192, 64) * 0.04)
                r = bf16(torch.randn(2, 192, 64) * 0.04)
                b = bf16(torch.randn(2, 384) * 0.1)
                h0 = bf16(torch.randn(2, 1, 64) * 0.2)
                for name, value in (("X", x), ("W", w), ("R", r),
                                    ("B", b), ("H", h0)):
                    allocate(name, tuple(value.shape), value)
                output = allocate("Y", (sequence, 2, 1, 64))
                final = allocate("Y_h", (2, 1, 64))
                compiler.identity_address = allocate(
                    "identity", (64, 64), torch.eye(64)).address
                compiler.gru_aux = {0: {
                    name: allocate(f"scratch/{name}", shape)
                    for name, shape in (
                        ("xg", (sequence, 192)), ("hg", (1, 192)),
                        *((name, (1, 64)) for name in
                          ("z", "r", "candidate", "tmp", "h")),
                    )
                }}
                node = onnx.helper.make_node(
                    "GRU", ["X", "W", "R", "B", "", "H"], ["Y", "Y_h"],
                    direction="bidirectional", hidden_size=64,
                    linear_before_reset=1)
                compiler.emit_gru(0, node)

                expected = torch.empty(sequence, 2, 1, 64)
                expected_h = torch.empty(2, 1, 64)
                for direction in range(2):
                    h = h0[direction, 0]
                    order = (range(sequence) if direction == 0
                             else reversed(range(sequence)))
                    for timestep in order:
                        xg = bf16(w[direction] @ x[timestep, 0]
                                  + b[direction, :192]).reshape(3, 64)
                        hg = bf16(r[direction] @ h
                                  + b[direction, 192:]).reshape(3, 64)
                        gates = bf16(torch.sigmoid(bf16(xg[:2] + hg[:2])))
                        candidate = bf16(xg[2] + bf16(gates[1] * hg[2]))
                        # The production tanh helper retains these BF16 stages.
                        candidate = bf16(torch.sigmoid(bf16(2 * candidate)))
                        candidate = bf16(bf16(2 * candidate) - 1)
                        h = bf16(bf16(bf16(h - candidate) * gates[0]) + candidate)
                        expected[timestep, direction, 0] = h
                    expected_h[direction, 0] = h
                torch.testing.assert_close(
                    engine.regions[output.address].float().reshape(expected.shape),
                    expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    engine.regions[final.address].float().reshape(expected_h.shape),
                    expected_h, rtol=0, atol=0)
                input_weights = {compiler.layout("W").address,
                                 compiler.layout("W").address + 192 * 64 * 2}
                projection_rows = [rows for rows, weight, _ in engine.matmul_calls
                                   if weight in input_weights]
                self.assertEqual(sum(projection_rows), 2 * sequence)
                self.assertEqual(len(projection_rows), 2 * ((sequence + 15) // 16))
                self.assertTrue(all(1 <= rows <= 16 for rows in projection_rows))
                recurrent_weights = {compiler.layout("R").address,
                                     compiler.layout("R").address + 192 * 64 * 2}
                recurrent_loads = [call for call in engine.sram_read_calls
                                   if call[0] in recurrent_weights]
                self.assertEqual(len(recurrent_loads), 2)


if __name__ == "__main__":
    unittest.main()
