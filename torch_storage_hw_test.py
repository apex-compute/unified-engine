"""Append-only PyTorch coverage for independent BF16/FP16 instruction fields.

Homogeneous FP16 operations use native torch.float16. Mixed operands/output
promote losslessly to FP32, since torch matmul requires one common dtype.
Quantized weights are decoded in FP32 without an artificial 16-bit writeback.
Only the SNR calculation uses FP64. Existing reference suites are untouched.
"""

from itertools import product
import math
import random

import torch

import user_dma_core as udc
from attention_reference import snr_db
from fp16_hw_test import FP16Suite


def format_combinations(*fields):
    for values in product(udc.StorageFormat, repeat=len(fields)):
        yield udc.TensorFormats(**dict(zip(fields, values)))


def stored_scalar(value, storage):
    return torch.tensor(storage.encode_scalar(value), dtype=torch.uint16).view(storage.dtype)


def reference_dtype(output_dtype, *operands):
    # Include the destination: BF16 inputs -> FP16 output must not round the
    # dot to BF16 before converting it to FP16.
    dtype = output_dtype
    for operand in operands:
        if operand is not None:
            dtype = torch.promote_types(dtype, operand.dtype)
    return dtype


@torch.autocast(device_type="cpu", enabled=False)
def torch_dot_reference(a, b, bias, output_dtype, *, bounds=None):
    dtype = reference_dtype(output_dtype, a, b, bias)
    a, b = a.to(dtype), b.to(dtype)
    result = a @ b.T if bias is None else torch.addmm(bias.to(dtype), a, b.T)
    if bounds is not None:
        result = result.clamp(*bounds)
    return result.to(output_dtype)


@torch.autocast(device_type="cpu", enabled=False)
def torch_vector_reference(a, b, mode, output_dtype):
    dtype = reference_dtype(output_dtype, a, b)
    a, b = a.to(dtype), b.to(dtype)
    if mode in (udc.UE_MODE.ELTWISE_ADD, udc.UE_MODE.ADD_BROADCAST):
        result = a + b
    elif mode in (udc.UE_MODE.ELTWISE_MUL, udc.UE_MODE.MUL_BROADCAST):
        result = a * b
    elif mode == udc.UE_MODE.ELTWISE_SUB:
        result = a - b
    else:
        raise ValueError(f"Unsupported vector mode: {mode}")
    return result.to(output_dtype)


@torch.autocast(device_type="cpu", enabled=False)
def torch_fp16_attention_reference(q, k, v, bias, *, scale_bits):
    assert all(t.dtype == torch.float16 for t in (q, k, v, bias))
    scale = torch.tensor(scale_bits, dtype=torch.uint16).view(torch.float16)
    return torch.addmm(bias, q * scale, k.T).softmax(-1) @ v


class TorchStorageSuite(FP16Suite):
    def __init__(self, record):
        super().__init__(record, prefix="torch_formats")
        self.generator = torch.Generator().manual_seed(20261102)

    def sample(self, shape, storage, scale=1.0):
        return (torch.randn(shape, generator=self.generator) * scale).to(storage.dtype)

    @staticmethod
    def dimensions(engine, dynamic, **sizes):
        if not dynamic:
            return {}
        kwargs = {}
        for name, value in sizes.items():
            reg = engine.alloc_isa_reg()
            engine.generate_instruction_add_set(reg, value)
            kwargs[f"gpr_{name}_reg"] = reg
        return kwargs

    def score(self, name, reference, output, engine):
        assert reference.dtype == output.dtype == self.formats.output.dtype
        assert reference.shape == output.shape
        score = snr_db(reference, output)
        assert score >= 40.0, f"{name}: {score:.6f} dB below 40 dB"
        self.record(f"{self.prefix}_{name}", str(tuple(output.shape)), snr_db=score,
                    cycles=engine.read_latency_cycles(),
                    inst_bytes=engine.get_capture_instruction_size_bytes())
        print(f"TORCH FORMATS {name}: {score:.6f} dB")

    def vector(self, formats, mode, dynamic, shape=(3, 128)):
        self.formats = formats
        engine = udc.UnifiedEngine()
        a = self.sample(shape, formats.a)
        broadcast = mode in (udc.UE_MODE.ADD_BROADCAST, udc.UE_MODE.MUL_BROADCAST)
        scalar = -0.3127 if mode == udc.UE_MODE.ADD_BROADCAST else 0.1001
        b = stored_scalar(scalar, formats.scalar) if broadcast else self.sample(shape, formats.b)
        aa = self.stage(engine, a)
        ba = None if broadcast else self.stage(engine, b)

        def emit(dst):
            engine.eltwise_core_dram(
                M=shape[0], N=shape[1], dram_a=aa, dram_b=ba, dram_out=dst,
                mode=mode, scalar=scalar if broadcast else None, formats=formats,
                **self.dimensions(engine, dynamic, M=shape[0], N=shape[1]))

        output = self.execute(engine, emit, shape)
        self.score(f"{mode.name.lower()}_{'dynamic' if dynamic else 'legacy'}_fmt{formats.bits:02x}",
                   torch_vector_reference(a, b, mode, formats.output.dtype), output, engine)

    def dot(self, formats, dynamic, k, *, bias_mode="full_matrix", clamp=False):
        self.formats = formats
        engine = udc.UnifiedEngine()
        m, n = 3, 64
        a = self.sample((m, k), formats.a, 1 / math.sqrt(k))
        b = self.sample((n, k), formats.b)
        bias = None if bias_mode is None else self.sample(
            (n,) if bias_mode == "broadcast_N" else (m, n), formats.bias, 0.125)
        aa, ba = self.stage(engine, a), self.stage(engine, b)
        ca = None if bias is None else self.stage(engine, bias)

        def emit(dst):
            engine.matmat_mul_core(
                m, k, n, aa, ba, dst, C_DRAM_ADDR=ca,
                bias_mode=bias_mode or "full_matrix", formats=formats,
                clamp_enable=clamp, clamp_min=-0.3333, clamp_max=0.1001,
                **self.dimensions(engine, dynamic, M=m, K=k, N=n))

        output = self.execute(engine, emit, (m, n))
        bounds = tuple(stored_scalar(x, formats.params).item() for x in (-0.3333, 0.1001)) if clamp else None
        self.score(f"dot_{'dynamic' if dynamic else 'legacy'}_K{k}_{bias_mode or 'no_bias'}"
                   f"{'_clamp' if clamp else ''}_fmt{formats.bits:02x}",
                   torch_dot_reference(a, b, bias, formats.output.dtype, bounds=bounds), output, engine)

    def streamed_dot(self, formats, dynamic, kind, *, integer=False, bias_mode="full_matrix"):
        self.formats = formats
        engine = udc.UnifiedEngine()
        # TQ4 reloads its codebook from URAM-B[0] at each DOT. A single row
        # keeps the high-level core's output from replacing a later codebook.
        m, k, n = (1 if kind == udc.TYPE.TQ4 else 3), 128, 64
        a = self.sample((m, k), formats.a, 1 / math.sqrt(k))
        bias = self.sample((n,) if bias_mode == "broadcast_N" else (m, n), formats.bias, 0.125)
        codebook_addr = None
        if kind == udc.TYPE.IF8:
            index = torch.arange(n * k) % (256 if integer else 254)
            codes = (index if integer else index + (index >= 127)).to(torch.uint8)
            if integer:
                values = codes.view(torch.int8).float()
            else:
                exponent, fraction = ((codes >> 3) & 15).int(), (codes & 7).float()
                values = torch.where(exponent == 0, fraction * 2**-9,
                                     (1 + fraction / 8) * 2.0**(exponent - 7))
                values *= torch.where(codes & 128 != 0, -1.0, 1.0)
            packed = codes
        else:
            codes = torch.arange(n * k).remainder(16).to(torch.uint8)
            if kind == udc.TYPE.TQ4:
                table = torch.tensor([-1.937, -1.503, -1.001, -.751, -.501, -.253, -.127, 0,
                                      .063, .189, .377, .627, .877, 1.253, 1.627, 1.939],
                                     dtype=formats.codebook.dtype)
                codebook_addr = engine.prepare_tq4_codebook_dram(table, formats=formats)
            elif integer:
                table = torch.tensor([i if i < 8 else i - 16 for i in range(16)], dtype=torch.float32)
            else:
                table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
            values = table[codes.long()].float()
            packed = codes[::2] | (codes[1::2] << 4)
        scales = torch.tensor([0.101, 0.313, 0.761], dtype=formats.scale.dtype)[torch.arange(n * k // 64) % 3]
        weights = (values.reshape(-1, 64) * scales.float()[:, None]).reshape(n, k)
        aa, ba, sa, ca = (self.stage(engine, tensor) for tensor in
                          (a, packed, -scales if integer else scales, bias))

        def emit(dst):
            if codebook_addr is not None:
                engine.load_tq4_codebook(codebook_addr)
            engine.quantized_matmat_core(
                m, k, n, aa, ba, dst, sa, C_DRAM_ADDR=ca, bias_mode=bias_mode,
                data_type=kind, formats=formats,
                **self.dimensions(engine, dynamic, M=m, K=k, N=n))

        output = self.execute(engine, emit, (m, n))
        self.score(f"stream_{kind.name.lower()}_{'int' if integer else 'fp'}_"
                   f"{'dynamic' if dynamic else 'legacy'}_{bias_mode}_fmt{formats.bits:02x}",
                   torch_dot_reference(a, weights, bias, formats.output.dtype), output, engine)

    def attention(self, batch, seq, dim, dynamic):
        self.formats = udc.TensorFormats.uniform(udc.StorageFormat.FP16)
        engine = udc.UnifiedEngine()
        q, k, v = self.random(batch, dim), self.random(seq, dim), self.random(seq, dim)
        bias = self.random(batch, seq, scale=0.125)
        qa, ka, va, ba = (self.stage(engine, t) for t in (q, k, v, bias))
        identity = self.stage(engine, torch.eye(64, dtype=torch.float16))
        scratch = engine.allocate_tensor_dram((dim * seq + seq * seq + batch * dim) * 2)
        scale_bits = udc.StorageFormat.FP16.encode_scalar(1 / math.sqrt(dim))

        def emit(dst):
            engine.unified_attention_core(
                batch, seq, dim, qa, ka, va, ba, dst, scratch, identity,
                formats=self.formats, v_format=udc.StorageFormat.FP16,
                scratch_format=udc.StorageFormat.FP16, identity_format=udc.StorageFormat.FP16,
                **self.dimensions(engine, dynamic, batch=batch, aligned_seq_len=seq))

        output = self.execute(engine, emit, (batch, dim))
        self.score(f"attention_fp16_{'dynamic' if dynamic else 'legacy'}_S{seq}_D{dim}",
                   torch_fp16_attention_reference(q, k, v, bias, scale_bits=scale_bits), output, engine)


def run_torch_storage_suite(*, extended, record):
    """Run only after the legacy suites; preserve their end RNG fingerprint."""
    python_state, torch_state = random.getstate(), torch.get_rng_state()
    try:
        print("PYTORCH TENSOR FORMAT SUITE: independent BF16/FP16 selections.")
        print("References: native homogeneous 16-bit ops; mixed/quantized math in FP32; SNR in FP64.")
        suite = TorchStorageSuite(record)
        for dynamic in (False, True):
            for shape in (((3, 128), (2, 4160)) if extended else ((3, 128),)):
                for formats in format_combinations("a", "b", "output"):
                    for mode in (udc.UE_MODE.ELTWISE_ADD, udc.UE_MODE.ELTWISE_MUL, udc.UE_MODE.ELTWISE_SUB):
                        suite.vector(formats, mode, dynamic, shape)
                for formats in format_combinations("a", "scalar", "output"):
                    for mode in (udc.UE_MODE.ADD_BROADCAST, udc.UE_MODE.MUL_BROADCAST):
                        suite.vector(formats, mode, dynamic, shape)
            for k in ((64, 128, 8192) if extended else (64, 128)):
                for formats in format_combinations("a", "b", "output"):
                    suite.dot(formats, dynamic, k, bias_mode=None)
                for formats in format_combinations("a", "b", "bias", "output"):
                    suite.dot(formats, dynamic, k)
                for formats in format_combinations("a", "b", "bias", "output", "params"):
                    suite.dot(formats, dynamic, k, clamp=True)
            if extended:
                for formats in format_combinations("a", "b", "bias", "output"):
                    suite.dot(formats, dynamic, 128, bias_mode="broadcast_N")
            for bias_mode in (("full_matrix", "broadcast_N") if extended else ("full_matrix",)):
                for formats in format_combinations("a", "scale", "bias", "output"):
                    for kind in (udc.TYPE.IF4, udc.TYPE.IF8):
                        for integer in (False, True):
                            suite.streamed_dot(formats, dynamic, kind, integer=integer, bias_mode=bias_mode)
                for formats in format_combinations("a", "scale", "bias", "output", "codebook"):
                    suite.streamed_dot(formats, dynamic, udc.TYPE.TQ4, bias_mode=bias_mode)
            for seq, dim in (((64, 64), (128, 128), (64, 1024)) if extended else ((64, 64),)):
                suite.attention(3, seq, dim, dynamic)
        print("PYTORCH TENSOR FORMAT SUITE PASSED")
    finally:
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
