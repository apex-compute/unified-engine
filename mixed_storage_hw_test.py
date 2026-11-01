"""Runtime tensor-format coverage on one BF16-default FPGA image.

References start with the actual stored operands. Intermediate casts model
explicit tensor writeback boundaries; SNR arithmetic is always FP64.
"""

import math
import random

import torch
import torch.nn.functional as F

import user_dma_core as udc
from attention_reference import storage_attention_reference_fp64
from fp16_hw_test import FP16Suite, run_fp16_suite


BF16, FP16 = udc.StorageFormat.BF16, udc.StorageFormat.FP16


def stored_scalar(value, storage):
    return torch.tensor([storage.encode_scalar(value)], dtype=torch.uint16).view(storage.dtype).item()


def storage_boundary(value, storage):
    """FP64 values rounded at an explicit tensor/scalar storage boundary."""
    return value.to(storage.dtype).double()


def rms_reference(x, gamma, formats):
    # This is an ideal-arithmetic reference with actual storage boundaries,
    # not a model of the engine's BF19 adder/multiplier approximation.
    x64 = x.double()
    inverse = storage_boundary(x.shape[-1] ** 0.5 / x64.square().sum(-1, keepdim=True).sqrt(),
                               formats.output)
    normalized = storage_boundary(x64 * inverse, formats.output)
    return storage_boundary(normalized * gamma.double(), formats.output)


def layer_norm_reference(x, gamma, beta, formats, *, inv_n=None):
    x64 = x.double()
    if inv_n is None:
        mean = x64.mean(-1, keepdim=True)
    else:
        # Dynamic LayerNorm stores x*inv_n before its ADD_REDUCE.
        mean = storage_boundary(x64 * inv_n.double(), formats.output).sum(-1, keepdim=True)
    mean = storage_boundary(mean, formats.output)
    centered = storage_boundary(x64 - mean, formats.output)
    normalized = rms_reference(centered, gamma, formats)
    return storage_boundary(normalized + beta.double(), formats.output)


def rope_reference(x, cos, signed_sin, formats):
    x64 = x.double()
    half = x.shape[-1] // 2
    swapped = torch.cat((x64[:, half:], x64[:, :half]), dim=-1)
    product_cos = storage_boundary(x64 * cos.double(), formats.output)
    product_sin = storage_boundary(swapped * signed_sin.double(), formats.output)
    return storage_boundary(product_cos + product_sin, formats.output)


class MixedSuite(FP16Suite):
    def __init__(self, record):
        super().__init__(record, prefix="mixed", formats=udc.TensorFormats())
        self.generator = torch.Generator().manual_seed(20261022)

    def sample(self, shape, storage, scale=1.0):
        return (torch.randn(shape, generator=self.generator) * scale).to(storage.dtype)

    def bf16_identity(self):
        """All finite BF16 words, including the existing subnormal policy."""
        self.formats = udc.TensorFormats()
        words = torch.arange(65536, dtype=torch.int32)
        words = words[(words & 0x7F80) != 0x7F80]
        # BF19 cannot represent the lower half of BF16's subnormal range.
        # Preserve the established signed-zero flush; other finite words
        # must survive a BF16 -> BF19 -> multiply-by-one -> BF16 round trip.
        magnitude = words & 0x7FFF
        expected = torch.where(magnitude < 0x40, words & 0x8000, words).to(torch.uint16)
        source = words.to(torch.uint16).view(torch.bfloat16).reshape(1, -1)
        engine = udc.UnifiedEngine()
        address = self.stage(engine, source)
        output = self.execute(engine, lambda dst: engine.eltwise_core_dram(
            M=1, N=source.numel(), dram_a=address, dram_b=None, dram_out=dst,
            mode=udc.UE_MODE.MUL_BROADCAST, scalar=1.0, formats=self.formats), source.shape)
        assert torch.equal(expected, output.flatten().view(torch.uint16)), "BF16 identity changed an encoding"
        self.check("conversion_all_finite", expected.view(torch.bfloat16).reshape(source.shape),
                   output, engine, exact=True)

    def vectors(self, formats):
        self.formats = formats
        shape = (3, 128)
        a, b = self.sample(shape, formats.a), self.sample(shape, formats.b)
        for mode in (udc.UE_MODE.ELTWISE_ADD, udc.UE_MODE.ELTWISE_MUL,
                     udc.UE_MODE.MUL_BROADCAST):
            engine = udc.UnifiedEngine()
            aa, bb = self.stage(engine, a), self.stage(engine, b)
            scalar = stored_scalar(0.1001, formats.scalar)
            rhs = scalar if mode == udc.UE_MODE.MUL_BROADCAST else b.double()
            reference = a.double() + rhs if mode == udc.UE_MODE.ELTWISE_ADD else a.double() * rhs
            output = self.execute(engine, lambda dst: engine.eltwise_core_dram(
                M=shape[0], N=shape[1], dram_a=aa,
                dram_b=None if mode == udc.UE_MODE.MUL_BROADCAST else bb,
                dram_out=dst, mode=mode,
                scalar=0.1001 if mode == udc.UE_MODE.MUL_BROADCAST else None,
                formats=formats), shape)
            self.check(f"{mode.name.lower()}_fmt{formats.bits:02x}", reference, output,
                       engine, floor=40.0)

    def mixed_matmul(self, formats, k, dynamic, activation=None):
        self.formats = formats
        engine = udc.UnifiedEngine()
        m, n = 3, 64
        a = self.sample((m, k), formats.a, 1 / math.sqrt(k))
        b = self.sample((n, k), formats.b)
        c = self.sample((m, n), formats.bias, 0.125)
        aa, bb, ca = (self.stage(engine, t) for t in (a, b, c))
        reference = a.double() @ b.double().T + c.double()
        if activation == "softmax":
            reference = reference.to(formats.output.dtype).double().softmax(-1)
        elif activation == "gelu":
            coefficient = -stored_scalar(-1.702, formats.params)
            reference = reference * torch.sigmoid(coefficient * reference)

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
                                   bias_mode="full_matrix", formats=formats, **kwargs)

        output = self.execute(engine, emit, (m, n))
        self.check(f"matmul_{'dynamic' if dynamic else 'legacy'}_K{k}_"
                   f"{activation or 'dot'}_fmt{formats.bits:02x}",
                   reference, output, engine, floor=38.0 if activation == "softmax" else 40.0)

    def mixed_attention(self, formats, v_format, scratch_format, dynamic, dim=128,
                        identity_format=None):
        self.formats = formats
        engine = udc.UnifiedEngine()
        batch, seq = 3, 64
        q = self.sample((batch, dim), formats.a)
        k = self.sample((seq, dim), formats.b)
        v = self.sample((seq, dim), v_format)
        bias = self.sample((batch, seq), formats.bias, 0.125)
        qa, ka, va, ba = (self.stage(engine, t) for t in (q, k, v, bias))
        if identity_format is None:
            identity_format = FP16 if v_format == BF16 else BF16
        identity = self.stage(engine, torch.eye(64, dtype=identity_format.dtype))
        scratch = engine.allocate_tensor_dram((dim * seq + seq * seq + batch * dim) * 2)
        scale_bits = formats.scalar.encode_scalar(1 / math.sqrt(dim))
        reference = storage_attention_reference_fp64(
            q, k, v, bias, scale_bits=scale_bits, scale_format=formats.scalar,
            scratch_format=scratch_format, query_chunk_size=batch)

        def emit(dst):
            kwargs = {}
            if dynamic:
                for name, value in (("batch", batch), ("aligned_seq_len", seq)):
                    reg = engine.alloc_isa_reg()
                    engine.generate_instruction_add_set(reg, value)
                    kwargs[f"gpr_{name}_reg"] = reg
            engine.unified_attention_core(
                batch, seq, dim, qa, ka, va, ba, dst, scratch, identity,
                formats=formats, v_format=v_format, scratch_format=scratch_format,
                identity_format=identity_format, **kwargs)

        output = self.execute(engine, emit, (batch, dim))
        self.check(f"attention_{'dynamic' if dynamic else 'legacy'}_D{dim}_"
                   f"fmt{formats.bits:02x}_v{int(v_format)}_scratch{int(scratch_format)}",
                   reference, output, engine, floor=38.0)

    def mixed_pool(self, input_format, output_format):
        self.formats = udc.TensorFormats(a=input_format, output=output_format)
        engine = udc.UnifiedEngine()
        x = -(self.sample((65, 7, 9), input_format).abs() + 0.125).to(input_format.dtype)
        x[0, 1:4, 1:4] = torch.tensor(2**-24, dtype=input_format.dtype)
        output = engine.run_maxpool2d_layer(x, kernel=3, stride_s=2, pad=1,
                                           formats=self.formats)
        reference = F.max_pool2d(x.double(), 3, 2, 1).to(output_format.dtype)
        self.check(f"maxpool_a{int(input_format)}_out{int(output_format)}",
                   reference, output, engine, exact=True)

    def normalization(self, formats, dynamic, *, layer_norm=False):
        self.formats = formats
        engine = udc.UnifiedEngine()
        m, n = 3, 128
        x = self.sample((m, n), formats.a, 0.5)
        gamma = (self.sample((n,), formats.b, 0.125) + 1.0).to(formats.b.dtype)
        aa, ga = self.stage(engine, x), self.stage(engine, gamma)
        if layer_norm:
            beta = self.sample((n,), formats.b, 0.0625)
            inv_n = torch.full((n,), 1.0 / n, dtype=formats.b.dtype)
            ba = self.stage(engine, beta)
            za = self.stage(engine, torch.zeros(n, dtype=formats.b.dtype))
            ia = self.stage(engine, inv_n) if dynamic else None
            reference = layer_norm_reference(x, gamma, beta, formats,
                                              inv_n=inv_n if dynamic else None)
        else:
            reference = rms_reference(x, gamma, formats)

        def emit(dst):
            kwargs = {}
            if dynamic:
                reg = engine.alloc_isa_reg()
                engine.generate_instruction_add_set(reg, m)
                kwargs['gpr_M_reg'] = reg
            if layer_norm:
                engine.layer_norm_core_dram(
                    m, n, aa, dst, GAMMA_DRAM_ADDR=ga, BETA_DRAM_ADDR=ba,
                    ZEROS_DRAM_ADDR=za, INV_N_DRAM_ADDR=ia,
                    formats=formats, **kwargs)
            else:
                engine.rms_norm_core_dram(m, n, aa, dst, ga, formats=formats, **kwargs)

        output = self.execute(engine, emit, (m, n))
        name = 'layer_norm' if layer_norm else 'rms_norm'
        self.check(f"{name}_{'dynamic' if dynamic else 'legacy'}_fmt{formats.bits:02x}",
                   reference, output, engine, floor=40.0)

    def mixed_rope(self, formats, dynamic):
        self.formats = formats
        engine = udc.UnifiedEngine()
        m, n = 3, 128
        x = self.sample((m, n), formats.a)
        angle = self.sample((m, n // 2), BF16).double()
        angle_cos, angle_sin = angle.cos(), angle.sin()
        cos = torch.cat((angle_cos, angle_cos), dim=-1).to(formats.b.dtype)
        signed_sin = torch.cat((-angle_sin, angle_sin), dim=-1).to(formats.b.dtype)
        # Each row is [cos[N], signed_sin[N]], matching both compiler layouts.
        coefficients = torch.cat((cos, signed_sin), dim=-1)
        aa, ca = self.stage(engine, x), self.stage(engine, coefficients)
        reference = rope_reference(x, cos, signed_sin, formats)

        def emit(dst):
            kwargs = {}
            if dynamic:
                reg = engine.alloc_isa_reg()
                engine.generate_instruction_add_set(reg, m)
                kwargs['gpr_M_reg'] = reg
            engine.rope_hf_core_dram(m, n, aa, dst, ca, ca + n * 2,
                                     formats=formats, **kwargs)

        output = self.execute(engine, emit, (m, n))
        self.check(f"rope_{'dynamic' if dynamic else 'legacy'}_fmt{formats.bits:02x}",
                   reference, output, engine, floor=40.0)

    def alternating_queue(self):
        """Switch formats, then restore implicit BF16 after all eight flags were set."""
        engine = udc.UnifiedEngine()
        configurations = [udc.TensorFormats(a=bits & 1, b=(bits >> 1) & 1,
                                            output=(bits >> 2) & 1)
                          for bits in (0, 7, 1, 6, 2, 5, 3, 4, 0)]
        configurations += [udc.TensorFormats.uniform(FP16), None]
        cases = []
        for requested in configurations:
            formats = requested if requested is not None else udc.TensorFormats()
            # Exactly representable sums make stale format bits unambiguous.
            a = torch.arange(64).remainder(13).mul(0.125).to(formats.a.dtype)
            b = torch.arange(64).remainder(7).mul(-0.25).to(formats.b.dtype)
            cases.append((formats, self.stage(engine, a), self.stage(engine, b),
                          (a.double() + b.double()).to(formats.output.dtype)))
        output_address = engine.allocate_tensor_dram(len(cases) * 128)
        engine.start_capture()
        for index, (formats, aa, bb, _) in enumerate(cases):
            engine.eltwise_core_dram(M=1, N=64, dram_a=aa, dram_b=bb,
                                     dram_out=output_address + index * 128,
                                     mode=udc.UE_MODE.ELTWISE_ADD, formats=configurations[index])
        engine.stop_capture()
        engine.generate_instruction_halt()
        program = engine.get_program_dram_addr()
        engine.write_captured_instructions_to_dram(program)
        engine.start_execute_from_dram(program)
        engine.wait_queue(60.0)
        words = torch.empty((len(cases), 64), dtype=torch.uint16)
        engine.dma_read(engine.c2h_device, output_address, words, words.numel() * 2)
        for index, (formats, _, _, reference) in enumerate(cases):
            actual = words[index].view(formats.output.dtype)
            assert torch.equal(reference.view(torch.uint16), words[index]), "Format changed in flight"
            self.check(f"alternating_queue_{index}_fmt{formats.bits:02x}",
                       reference, actual, engine, exact=True)


def run_bf16_boundary_suite(*, extended, record):
    """Explicit all-BF16 descriptors through the same runtime software paths."""
    print("BF16 EXPLICIT FORMAT SUITE: runtime image, all tensor-format bits zero.")
    suite = MixedSuite(record)
    suite.prefix = "bf16_explicit"
    formats = udc.TensorFormats()
    suite.bf16_identity()
    suite.vectors(formats)
    for dynamic in (False, True):
        suite.mixed_matmul(formats, 64, dynamic)
        suite.mixed_matmul(formats, 128, dynamic, "softmax")
        suite.normalization(formats, dynamic)
        suite.normalization(formats, dynamic, layer_norm=True)
        suite.mixed_rope(formats, dynamic)
        if extended:
            suite.mixed_matmul(formats, 8192, dynamic, "softmax")
            suite.mixed_attention(formats, BF16, BF16, dynamic, dim=1024,
                                  identity_format=BF16)
    suite.formats = formats
    suite.quantized(udc.TYPE.IF8, integer=True)
    suite.quantized(udc.TYPE.TQ4)
    print("BF16 EXPLICIT FORMAT SUITE PASSED")


def run_mixed_suite(*, extended, record):
    print("MIXED TENSOR FORMAT SUITE: explicit BF16/FP16 boundaries on one image.")
    python_state, torch_state = random.getstate(), torch.get_rng_state()
    try:
        # Preserve the FP16 reference suite's original starting state, then restore
        # the BF16 suite's end fingerprint. No legacy inputs or floors change.
        random.seed(0)
        torch.manual_seed(0)
        torch.randn(4, dtype=torch.bfloat16)
        run_fp16_suite(extended=extended, record=record)
        suite = MixedSuite(record)
        for bits in range(8):
            formats = udc.TensorFormats(
                a=bits & 1, b=(bits >> 1) & 1, output=(bits >> 2) & 1,
                bias=1 - ((bits >> 2) & 1), scalar=1 - (bits & 1),
                params=1 - ((bits >> 1) & 1))
            suite.vectors(formats)
            for dynamic in (False, True):
                suite.normalization(formats, dynamic)
                suite.normalization(formats, dynamic, layer_norm=True)
                suite.mixed_rope(formats, dynamic)
                for k, activation in ((64, None), (128, None), (128, "softmax"), (128, "gelu")):
                    suite.mixed_matmul(formats, k, dynamic, activation)
                if extended:
                    suite.mixed_attention(formats, FP16 if bits & 1 else BF16,
                                          BF16 if formats.output == FP16 else FP16, dynamic)
        for source in (BF16, FP16):
            for output in (BF16, FP16):
                suite.mixed_pool(source, output)
                quant = FP16Suite(record, formats=udc.TensorFormats(a=source, output=output),
                                  prefix=f"mixed_quant_a{int(source)}_out{int(output)}")
                quant.quantize_fp4(32)
        for bits in range(8):
            formats = udc.TensorFormats(a=bits & 1, output=1 - (bits & 1),
                                        scale=(bits >> 1) & 1, codebook=(bits >> 2) & 1)
            quant = FP16Suite(record, formats=formats, prefix=f"mixed_aux_fmt{formats.bits:02x}")
            quant.quantized(udc.TYPE.TQ4)
            if extended:
                quant.quantized(udc.TYPE.IF4, integer=bool(bits & 4))
                quant.quantized(udc.TYPE.IF8, integer=bool(bits & 4))
        suite.alternating_queue()
        if extended:
            formats = udc.TensorFormats(a=FP16, b=BF16, output=FP16, bias=BF16, scalar=FP16)
            for dynamic in (False, True):
                suite.mixed_matmul(formats, 8192, dynamic, "softmax")
                suite.mixed_attention(formats, BF16, FP16, dynamic, dim=1024)
        run_bf16_boundary_suite(extended=extended, record=record)
        print("MIXED TENSOR FORMAT SUITE PASSED")
    finally:
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
