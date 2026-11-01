"""FP16 operator suite using explicit per-instruction tensor formats.

This is not the BF16 model suite. References start from the exact FP16 payloads
and evaluate in FP64; SNR never squares FP16 data (which can overflow).
"""

import math

import torch
import torch.nn.functional as F

import user_dma_core as udc
from attention_reference import snr_db, storage_attention_reference_fp64


class FP16Suite:
    def __init__(self, record, *, formats=None, prefix="fp16"):
        self.formats = formats or udc.TensorFormats.uniform(udc.StorageFormat.FP16)
        self.prefix = prefix
        self.record = record
        self.generator = torch.Generator().manual_seed(20261021)

    def random(self, *shape, scale=1.0):
        return (torch.randn(shape, generator=self.generator) * scale).half()

    @staticmethod
    def stage(engine, tensor):
        tensor = tensor.contiguous()
        address = engine.allocate_tensor_dram(tensor.numel() * tensor.element_size())
        engine.dma_write(engine.h2c_device, address, tensor,
                         tensor.numel() * tensor.element_size())
        return address

    def execute(self, engine, emit, shape):
        output = engine.allocate_tensor_dram(math.prod(shape) * 2)
        engine.start_capture()
        emit(output)
        engine.stop_capture()
        engine.generate_instruction_halt()
        program = engine.get_program_dram_addr()
        engine.write_captured_instructions_to_dram(program)
        engine.start_execute_from_dram(program)
        engine.wait_queue(60.0)
        result = udc.DeviceTensor(shape, engine, dram_addr=output, dtype=self.formats.output.dtype).to_cpu()
        assert result.dtype == self.formats.output.dtype
        return result

    def check(self, name, reference, output, engine, *, exact=False, floor=40.0):
        score = snr_db(reference, output)
        if exact:
            assert torch.equal(reference, output), f"{name}: expected exact storage values"
        else:
            assert score >= floor, f"{name}: {score:.6f} dB below {floor:g} dB"
        self.record(self.prefix + "_" + name, str(tuple(output.shape)), snr_db=score,
                    cycles=engine.read_latency_cycles(),
                    inst_bytes=engine.get_capture_instruction_size_bytes())
        print(f"{self.prefix.upper()} {name}: {score:.6f} dB; reference=FP64(exact stored operands)")

    def conversion(self):
        # All finite binary16 encodings, including every subnormal and both
        # signed zeros, traverse storage -> BF19 -> MUL -> storage.
        words = torch.arange(65536, dtype=torch.int32)
        words = words[(words & 0x7C00) != 0x7C00].to(torch.uint16)
        x = words.view(torch.float16).reshape(1, -1)
        engine = udc.UnifiedEngine()
        source = self.stage(engine, x)
        output = self.execute(engine, lambda dst: engine.eltwise_core_dram(
            M=1, N=x.numel(), dram_a=source, dram_b=None, dram_out=dst,
            mode=udc.UE_MODE.MUL_BROADCAST, scalar=1.0, formats=self.formats), x.shape)
        assert torch.equal(words, output.flatten().view(torch.uint16)), "FP16 identity changed an encoding"
        self.check("conversion_all_finite", x, output, engine, exact=True)

    def elementwise(self):
        for shape in ((1, 64), (3, 192), (2, 4160)):
            a, b = self.random(*shape), self.random(*shape)
            for mode, scalar in ((udc.UE_MODE.ELTWISE_ADD, None),
                                 (udc.UE_MODE.ELTWISE_MUL, None),
                                 (udc.UE_MODE.ADD_BROADCAST, -0.3125),
                                 (udc.UE_MODE.MUL_BROADCAST, 0.03125)):
                engine = udc.UnifiedEngine()
                aa, bb = self.stage(engine, a), self.stage(engine, b)
                rhs = b.double() if scalar is None else float(torch.tensor(scalar).half())
                ref = a.double() + rhs if mode in (udc.UE_MODE.ELTWISE_ADD, udc.UE_MODE.ADD_BROADCAST) else a.double() * rhs
                output = self.execute(engine, lambda dst: engine.eltwise_core_dram(
                    M=shape[0], N=shape[1], dram_a=aa,
                    dram_b=bb if scalar is None else None, dram_out=dst,
                    mode=mode, scalar=scalar, formats=self.formats), shape)
                self.check(mode.name.lower(), ref, output, engine)

    def matmul(self, m, k, n, *, dynamic=False, activation=None, bias=False):
        engine = udc.UnifiedEngine()
        a, b = self.random(m, k, scale=1 / math.sqrt(k)), self.random(n, k)
        c = self.random(m, n, scale=0.125) if bias else None
        aa, bb = self.stage(engine, a), self.stage(engine, b)
        ca = self.stage(engine, c) if c is not None else None
        ref = a.double() @ b.double().T
        if c is not None:
            ref += c.double()
        if activation == "softmax":
            ref = ref.half().double().softmax(-1)
        elif activation == "gelu":
            ref = ref * torch.sigmoid(float(torch.tensor(1.702).half()) * ref)
        elif activation == "silu":
            ref = ref * torch.sigmoid(ref)
        elif activation == "sigmoid":
            ref = torch.sigmoid(ref)
        elif activation == "clamp":
            ref = ref.clamp(-0.5, 0.75)
        elif activation == "log":
            ref = ref.clamp(min=float(torch.tensor(0.001).half())).log()

        def emit(dst):
            kwargs = {}
            if dynamic:
                for name, value in (("M", m), ("K", k), ("N", n)):
                    reg = engine.alloc_isa_reg()
                    engine.generate_instruction_add_set(reg, value)
                    kwargs[f"gpr_{name}_reg"] = reg
            if activation:
                kwargs[activation + "_enable"] = True
            engine.matmat_mul_core(m, k, n, aa, bb, dst, C_DRAM_ADDR=ca,
                                   bias_mode="full_matrix", clamp_min=-0.5,
                                   clamp_max=0.75, formats=self.formats, **kwargs)

        output = self.execute(engine, emit, (m, n))
        label = f"matmul_{'dynamic' if dynamic else 'legacy'}_K{k}_{activation or 'dot'}_bias{int(bias)}"
        self.check(label, ref, output, engine, floor=38.0 if activation == "softmax" else 40.0)

    def attention(self, batch, seq, dim, dynamic):
        engine = udc.UnifiedEngine()
        q, k, v = self.random(batch, dim), self.random(seq, dim), self.random(seq, dim)
        bias = self.random(batch, seq, scale=0.125)
        qa, ka, va, ba = (self.stage(engine, t) for t in (q, k, v, bias))
        identity = self.stage(engine, torch.eye(64, dtype=torch.float16))
        scratch = engine.allocate_tensor_dram((dim * seq + seq * seq + batch * dim) * 2)
        scale_bits = udc.StorageFormat.FP16.encode_scalar(1 / math.sqrt(dim))
        reference = storage_attention_reference_fp64(
            q, k, v, bias, scale_bits=scale_bits,
            scale_format=udc.StorageFormat.FP16,
            scratch_format=udc.StorageFormat.FP16, query_chunk_size=batch)

        def emit(dst):
            kwargs = {}
            if dynamic:
                for name, value in (("batch", batch), ("aligned_seq_len", seq)):
                    reg = engine.alloc_isa_reg()
                    engine.generate_instruction_add_set(reg, value)
                    kwargs[f"gpr_{name}_reg"] = reg
            engine.unified_attention_core(batch, seq, dim, qa, ka, va, ba, dst,
                                          scratch, identity, formats=self.formats,
                                          v_format=udc.StorageFormat.FP16,
                                          scratch_format=udc.StorageFormat.FP16,
                                          identity_format=udc.StorageFormat.FP16, **kwargs)

        output = self.execute(engine, emit, (batch, dim))
        self.check(f"attention_{'dynamic' if dynamic else 'legacy'}_S{seq}_D{dim}",
                   reference, output, engine)

    def maxpool(self):
        for channels in (3, 64, 65):
            engine = udc.UnifiedEngine()
            # Negative-only values expose a zero halo accidentally winning max.
            x = -(self.random(channels, 7, 9).abs() + 0.125).half()
            output = engine.run_maxpool2d_layer(x, kernel=3, stride_s=2, pad=1, formats=self.formats)
            ref = F.max_pool2d(x.double(), 3, 2, 1).half()
            self.check("maxpool_negative_halo", ref, output, engine, exact=True)

    def padding(self):
        for selection, word in ((udc.WB_PADDING_ZERO, 0), (udc.WB_PADDING_NEG_INF, 0xFC00)):
            engine = udc.UnifiedEngine()
            a = torch.zeros(64, dtype=torch.float16)
            a[0] = 1
            b = torch.arange(64).half().reshape(64, 1).expand(64, 64).contiguous()
            aa, bb = self.stage(engine, a), self.stage(engine, b)
            def emit(dst):
                engine.accelerator_memory_to_sram(aa, 0, 64)
                engine.accelerator_memory_to_sram(bb, 0x80000, 4096)
                engine.ue_op_descriptor(
                    mode=udc.UE_MODE.BF16_DOT_PRODUCT, max_clear_en=1,
                    uram_a_start_addr=0, uram_b_start_addr=0,
                    uram_wb_addr=128, uram_length=1, stride_z=1,
                    output_size=33, dma_length=33 * 64, wb_padding_select=selection,
                    formats=self.formats)
                engine.sram_to_accelerator_memory(0x4000, dst, 64)
            output = self.execute(engine, emit, (64,))
            assert torch.equal(output[:33], torch.arange(33).half())
            assert torch.equal(output[33:].view(torch.uint16), torch.full((31,), word, dtype=torch.uint16))
            self.record(f"fp16_writeback_padding_{selection}", "33 valid + 31 padded lanes")

    def convolution(self):
        # Small exact integer/fraction case: 3 input channels, non-multiple
        # output channels, FP16 bias/scale, zero halo and fused ReLU.
        for relu in (False, True):
            engine = udc.UnifiedEngine(conv_geometry_mode=udc.CONV_GEOMETRY_QUEUE_CONFIG)
            channels, oc, h, w = 3, 5, 4, 5
            x = torch.arange(channels * h * w).remainder(7).sub(3).half().reshape(channels, h, w)
            codes = torch.arange(oc * channels * 9).remainder(5).sub(2).short().reshape(oc, channels, 3, 3)
            bias = torch.arange(oc).sub(2).mul(0.125).half()
            acts = udc.conv2d_pack_activation_map(x, 1)
            weights = udc.conv2d_pack_weight_stream(codes, h, w, udc.TYPE.IF4)
            scales = udc.conv2d_pack_scale_stream(-0.5, oc, 9, h, w, dtype=torch.float16)
            biases = udc.conv2d_pack_bias_stream(bias, h, w, dtype=torch.float16)
            aa, wa, sa, ba = (self.stage(engine, t) for t in (acts, weights, scales, biases))
            output = self.execute(engine, lambda dst: engine.conv2d_core(
                aa, wa, sa, dst, c_in=channels, in_h=h, in_w=w,
                kernel_h=3, kernel_w=3, stride_s=1, pad=1, oc_count=oc,
                data_type=udc.TYPE.IF4, BIAS_DRAM_ADDR=ba, relu_enable=relu,
                formats=self.formats),
                (((oc * h * w + 63) // 64) * 64,))
            output = udc.conv2d_unpack_result(output, h, w, oc)
            ref = F.conv2d(x.double().unsqueeze(0), codes.double() * 0.5,
                           bias.double(), padding=1)[0]
            self.check(f"conv_padding_bias_relu{int(relu)}", ref.relu() if relu else ref,
                       output, engine, exact=True)

    def quantized(self, kind, integer=False, k=128):
        engine = udc.UnifiedEngine()
        n, blocks = 64, k // 64
        count = n * k
        if kind == udc.TYPE.IF8:
            indices = torch.arange(count) % (256 if integer else 254)
            codes = (indices if integer else indices + (indices >= 127)).to(torch.uint8)
            if integer:
                values = codes.view(torch.int8).double()
            else:
                sign = torch.where(codes & 128 != 0, -1.0, 1.0).double()
                exponent, fraction = ((codes >> 3) & 15).long(), (codes & 7).double()
                values = sign * torch.where(exponent == 0, fraction * 2**-9,
                                            (1 + fraction / 8) * torch.pow(2.0, exponent.double() - 7))
            packed = codes
        else:
            codes = (torch.arange(count) % 16).to(torch.uint8)
            if kind == udc.TYPE.TQ4:
                table = torch.tensor([-1.9375, -1.5, -1, -0.75, -0.5, -0.25, -0.125, 0,
                                      0.0625, 0.1875, 0.375, 0.625, 0.875, 1.25, 1.625, 1.9375]).to(self.formats.codebook.dtype)
                codebook = engine.prepare_tq4_codebook_dram(table, formats=self.formats)
                values = table[codes.long()].double()
            elif integer:
                values = torch.where(codes.long() >= 8, codes.long() - 16, codes.long()).double()
            else:
                table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
                values = table[codes.long()].double()
            packed = codes[::2] | (codes[1::2] << 4)
        scales = (torch.arange(n * blocks).remainder(3).float() + 1).mul(0.25).to(self.formats.scale.dtype)
        values = values.reshape(-1, 64) * scales.double().reshape(-1, 1)
        stored_scales = -scales if integer else scales
        packed_addr, scale_addr = self.stage(engine, packed), self.stage(engine, stored_scales)

        def dequant(dst):
            if kind == udc.TYPE.TQ4:
                engine.load_tq4_codebook(codebook)
            engine.start_queue_for_bf16_dequantize_operation(
                packed_addr, scale_addr, kind, 0x80100, count, formats=self.formats)
            engine.sram_to_accelerator_memory(0x80100, dst, count)

        output = self.execute(engine, dequant, (n, k))
        label = kind.name.lower() + ("_int" if integer else "_fp")
        # Compare exact storage values: IF8 integers times 0.75 can land on
        # BF16 midpoints even though the product is exact in BF19/FP16.
        self.check(label + "_dequant", values.reshape(n, k).to(self.formats.output.dtype),
                   output, engine, exact=True)
        # One-hot probes across the 64-lane boundary exercise the actual
        # streaming quantized DOT path without hiding a scale/codebook error.
        for pos in (0, 63, k - 1):
            a = torch.zeros(k, dtype=self.formats.a.dtype)
            a[pos] = 1
            aa = self.stage(engine, a)

            def dot(dst):
                if kind == udc.TYPE.TQ4:
                    engine.load_tq4_codebook(codebook)
                engine.accelerator_memory_to_sram(aa, 0, k)
                engine.accelerator_memory_to_scale_sram(scale_addr, n * blocks)
                engine.start_queue_for_dot_product_operation(
                    1, 0, 0, 0x80100, k, n, packed_addr, data_type=kind, formats=self.formats)
                engine.sram_to_accelerator_memory(0x80100, dst, n)

            engine.clear_capture_buffer()
            output = self.execute(engine, dot, (n,))
            self.check(f"{label}_dot_onehot_{pos}",
                       values.reshape(n, k)[:, pos].to(self.formats.output.dtype),
                       output, engine, exact=True)

    def quantize_fp4(self, blocks=16, *, page_cross=False):
        """Quantize complete 64-lane blocks, up to the 1,024-scale QBRAM limit.

        No input or packed-output padding is needed for these complete blocks.
        The page-crossing case also checks a following URAM copy so a premature
        scale-transfer completion cannot hide behind the final HALT.
        """
        assert 0 < blocks <= udc.QBRAM_ELEMENTS
        name = "fp4_quantize" if blocks == 16 and not page_cross else f"fp4_quantize_{blocks}blocks"
        if page_cross:
            name += "_page_cross"
        engine = udc.UnifiedEngine()
        table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
        codes = torch.arange(blocks * 64).remainder(16).to(torch.uint8)
        expected_scales = torch.tensor([0.25, 0.5, 1.0, 2.0]).to(self.formats.output.dtype)[torch.arange(blocks) % 4]
        x = (table[codes.long()].reshape(-1, 64) * expected_scales[:, None]).flatten().to(self.formats.a.dtype)
        source = self.stage(engine, x)
        output_addr = engine.allocate_tensor_dram(x.numel() // 2)
        # QBRAM writes whole AXI beats; reserve any tail but validate only the
        # scales produced by this quantization operation.
        scale_read_bytes = udc.ue_round_up_to_axi_beat_bytes(expected_scales.numel() * 2)
        scales_addr = engine.allocate_tensor_dram(scale_read_bytes + (4096 if page_cross else 0))
        if page_cross:
            beat_bytes = udc.ue_axi_beat_bytes()
            scales_addr = ((scales_addr + beat_bytes + 4095) // 4096) * 4096 - beat_bytes
        sentinel = torch.full((scale_read_bytes,), 0xA5, dtype=torch.uint8)
        engine.dma_write(engine.h2c_device, scales_addr, sentinel, scale_read_bytes)
        trailing_addr = None
        if page_cross:
            trailing_addr = self.stage(
                engine, torch.full((x.numel() // 2,), 0xA5, dtype=torch.uint8))
        engine.start_capture()
        engine.accelerator_memory_to_sram(source, 0, x.numel())
        engine.start_queue_for_quantize_operation(0, 0x80000, udc.TYPE.IF4, x.numel(),
                                                  formats=self.formats)
        engine.sram_to_accelerator_memory(0x80000, output_addr, x.numel() // 4)
        engine.ue_memcpy_to_dram(udc.MEMCPY_TYPE.SCALE_BRAM, 0, 0, scales_addr,
                                 scale_read_bytes)
        if page_cross:
            engine.sram_to_accelerator_memory(0x80000, trailing_addr, x.numel() // 4)
        engine.stop_capture()
        engine.generate_instruction_halt()
        program = engine.get_program_dram_addr()
        engine.write_captured_instructions_to_dram(program)
        engine.start_execute_from_dram(program)
        engine.wait_queue(10)
        packed = torch.zeros(x.numel() // 2, dtype=torch.uint8)
        engine.dma_read(engine.c2h_device, output_addr, packed, packed.numel())
        if page_cross:
            trailing = torch.zeros_like(packed)
            engine.dma_read(engine.c2h_device, trailing_addr, trailing, trailing.numel())
            assert torch.equal(packed, trailing), f"{name}: following URAM copy changed packed FP4 output"
        actual_codes = torch.stack((packed & 15, packed >> 4), -1).flatten()
        actual_scales = engine.dma_from_accelerator_memory(scales_addr, expected_scales.shape,
                                                             dtype=self.formats.output.dtype)
        restored = (table[actual_codes.long()].reshape(-1, 64) * actual_scales[:, None]).flatten().to(self.formats.a.dtype)
        if not torch.equal(expected_scales, actual_scales) or not torch.equal(x, restored):
            # Keep both exact checks below. Capture the payload before the first
            # assertion so a scale failure cannot hide the packed-code evidence.
            print(f"FP16 {name} diagnostics: input=0x{source:x}, packed=0x{output_addr:x}, "
                  f"scales=0x{scales_addr:x}, scale_read_bytes={scale_read_bytes}")
            for label, scales in (("expected", expected_scales), ("actual", actual_scales)):
                words = scales.contiguous().view(torch.uint16).flatten().tolist()
                print(f"  {label} scale values: {scales.flatten().tolist()}")
                print(f"  {label} scale words: " + " ".join(f"{word:04x}" for word in words))
            print("  packed FP4 first 64 bytes: " + " ".join(f"{byte:02x}" for byte in packed[:64].tolist()))
            # Compare decoded levels so the two zero encodings are equivalent.
            level_mismatch = table[actual_codes.long()] != table[codes.long()]
            print(f"  decoded-level mismatches per 64-lane block: "
                  f"{level_mismatch.reshape(-1, 64).sum(dim=1).tolist()}")
            mismatches = (x != restored).nonzero().flatten()
            print(f"  reconstructed mismatches: {mismatches.numel()}/{x.numel()}")
            expected_words, actual_words = x.view(torch.uint16), restored.view(torch.uint16)
            for position in mismatches[:16].tolist():
                print(f"  lane {position} (block {position // 64}): "
                      f"packed=0x{int(packed[position // 2]):02x}, "
                      f"code expected={int(codes[position]):x} actual={int(actual_codes[position]):x}, "
                      f"value expected={float(x[position]):g}/0x{int(expected_words[position]):04x} "
                      f"actual={float(restored[position]):g}/0x{int(actual_words[position]):04x}")
            for index, instruction in enumerate(engine.capture_buffer):
                print(f"  instruction {index}: " + " ".join(f"{word:08x}" for word in instruction.words))
        if not torch.equal(expected_scales, actual_scales):
            original_words = actual_scales.contiguous().view(torch.uint16).flatten().tolist()
            print(f"  initial scale sentinel unchanged: {all(word == 0xA5A5 for word in original_words)}")
        self.check(name + "_scales", expected_scales, actual_scales, engine, exact=True)
        self.check(name + "_exact_levels", x, restored, engine, exact=True)


def run_fp16_suite(*, extended, record):
    print("FP16 STORAGE OPERATOR SUITE: BF19 arithmetic; independent FP64 references.")
    print("BF16 model caches, full model inference and BF16 reference sweeps are NOT exercised.")
    suite = FP16Suite(record)
    reset_engine = udc.UnifiedEngine()
    reset_engine.software_reset(run_dram_self_test=False)
    suite.conversion()
    suite.elementwise()
    for dynamic in (False, True):
        for k in ((64, 128, 1024, 8192) if extended else (64, 128)):
            for bias in (False, True):
                suite.matmul(3, k, 64, dynamic=dynamic, bias=bias)
            suite.matmul(3, k, 64, dynamic=dynamic, bias=True, activation="softmax")
        for activation in ("gelu", "silu", "sigmoid", "clamp", "log"):
            suite.matmul(3, 128, 64, dynamic=dynamic, bias=True, activation=activation)
        for seq, dim in (((64, 64), (128, 128), (64, 1024)) if extended else ((64, 64),)):
            suite.attention(3, seq, dim, dynamic)
    suite.maxpool()
    suite.padding()
    suite.convolution()
    for kind in (udc.TYPE.IF4, udc.TYPE.IF8):
        for integer in (False, True):
            suite.quantized(kind, integer)
    suite.quantized(udc.TYPE.TQ4)
    suite.quantize_fp4()
    if extended:
        suite.quantize_fp4(32)
        suite.quantize_fp4(1024, page_cross=True)
    print("FP16 STORAGE OPERATOR SUITE PASSED")
