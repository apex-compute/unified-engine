"""Static BF16 BigCodec vector quantizer for the Andromeda instruction queue.

The runtime path consists entirely of DRAM/SRAM copies, matrix products and
pointwise arithmetic. A 13-level hard tournament replaces dynamic argmax and
embedding gather. Tokens use two BF16 integers (low 8 bits, high 5 bits), since a
single BF16 value cannot represent all 8192 integer token IDs exactly.

The original identity-dot comparison on RK 256-bit build 0xdf0749de flushed
positive subnormal score differences (less than 2**-126) to zero. Exhaustive
finite-pattern testing of the SRAM mask on build 0x40519e0a instead found 63
positive gaps below 2**-127 flushed to zero; larger gaps were classified
correctly. Flushed gaps retain the left candidate as a tie. Zero outputs may
have a negative sign bit, so this is a numeric contract, not IEEE bit equality.
Software tests also cover ideal BF16 and input-flushing arithmetic models.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import user_dma_core as udc

WIDTH = 64
CODEBOOK_SIZE = 8192
CODE_DIM = 8
FEATURE_DIM = 1024
PAYLOAD_ROWS = CODE_DIM + 3  # embedding, token low/high, score
SCORE_ROW = PAYLOAD_ROWS - 1


def _align(value: int) -> int:
    return (int(value) + 127) // 128 * 128


def _scratch_layout(frames: int) -> tuple[dict[str, int], int]:
    if not isinstance(frames, int) or frames < 1:
        raise ValueError("frames must be a positive integer")
    sizes = {
        "projected": frames * WIDTH * 2,
        "norm_work": frames * WIDTH * 2,
        "normalized": frames * WIDTH * 2,
        "ping": PAYLOAD_ROWS * CODEBOOK_SIZE * 2,
        "pong": PAYLOAD_ROWS * CODEBOOK_SIZE * 2,
        "mask": CODEBOOK_SIZE * 2,
        "inverse_mask": CODEBOOK_SIZE * 2,
        "work_left": CODEBOOK_SIZE * 2,
        "work_right": CODEBOOK_SIZE * 2,
        "gather_left": PAYLOAD_ROWS * WIDTH * 2,
        "gather_right": PAYLOAD_ROWS * WIDTH * 2,
        "transpose_in": WIDTH * WIDTH * 2,
        "transpose_out": WIDTH * WIDTH * 2,
    }
    offsets, cursor = {}, 0
    for name, size in sizes.items():
        offsets[name] = cursor
        cursor += _align(size)
    return offsets, cursor


def quantizer_scratch_bytes(frames: int) -> int:
    return _scratch_layout(frames)[1]


def bit_reversed_indices(count: int = CODEBOOK_SIZE) -> torch.Tensor:
    if count < 2 or count & (count - 1):
        raise ValueError("codebook size must be a power of two")
    values = torch.arange(count, dtype=torch.int64)
    result = torch.zeros_like(values)
    for bit in range(count.bit_length() - 1):
        result = (result << 1) | ((values >> bit) & 1)
    return result


@dataclass(frozen=True)
class QuantizerPlan:
    frames: int
    source_address: int
    destination_address: int
    token_address: int
    scratch_address: int
    scratch_bytes: int
    scratch: dict[str, int]
    constants: dict[str, int]
    small_gathers: dict[int, tuple[int, int]]
    use_sram_tournament: bool = True


def prepare_quantizer(quantizer, image, *, source_address: int,
                      destination_address: int, scratch_address: int,
                      frames: int, token_address: int) -> QuantizerPlan:
    """Pack constants; reserve ``quantizer_scratch_bytes(frames)`` externally.

    ``quantizer`` is the single FactorizedVectorQuantize module, or its one-layer
    ResidualVQ wrapper, loaded with ``remove_weight_norm=True``.
    ``image.allocate(tensor, alignment=128)`` must store BF16
    tensors. Source/output are [frames, 1024] BF16; tokens are [frames, 64] with
    low/high bytes in lanes 0/1 and zero padding. All four buffers must be disjoint.
    """
    if hasattr(quantizer, "layers"):
        if len(quantizer.layers) != 1:
            raise ValueError("Only BigCodec's single codebook is supported")
        quantizer = quantizer.layers[0]
    if any(hasattr(layer, "weight_g") or hasattr(layer, "weight_v")
           for layer in (quantizer.in_proj, quantizer.out_proj)):
        raise ValueError("Load BigCodec with remove_weight_norm=True before packing weights")
    offsets, scratch_bytes = _scratch_layout(frames)
    addresses = (source_address, destination_address, scratch_address, token_address)
    if any(not isinstance(a, int) or a < 0 or a % 128 for a in addresses):
        raise ValueError("Quantizer buffer addresses must be nonnegative and 128-byte aligned")
    regions = [(source_address, frames * FEATURE_DIM * 2),
               (destination_address, frames * FEATURE_DIM * 2),
               (token_address, frames * WIDTH * 2), (scratch_address, scratch_bytes)]
    for i, (start, size) in enumerate(regions):
        for other, other_size in regions[i + 1:]:
            if start < other + other_size and other < start + size:
                raise ValueError("Quantizer buffers overlap")
    codebook = quantizer.codebook.weight.detach().cpu().float()
    w_in = quantizer.in_proj.weight.detach().cpu().float()
    b_in = quantizer.in_proj.bias.detach().cpu().float()
    w_out = quantizer.out_proj.weight.detach().cpu().float()
    b_out = quantizer.out_proj.bias.detach().cpu().float()
    if (codebook.shape != (CODEBOOK_SIZE, CODE_DIM)
            or w_in.shape != (CODE_DIM, FEATURE_DIM)
            or b_in.shape != (CODE_DIM,)
            or w_out.shape != (FEATURE_DIM, CODE_DIM)
            or b_out.shape != (FEATURE_DIM,)):
        raise ValueError("Expected the official 1024→8→8192 BigCodec quantizer")
    if any(not torch.isfinite(t).all() for t in (codebook, w_in, b_in, w_out, b_out)):
        raise ValueError("Quantizer parameters must be finite")
    if (torch.linalg.vector_norm(codebook, dim=1) < 1e-12).any():
        raise ValueError("Zero codebook vectors do not have the required unit-norm score equivalence")
    order = bit_reversed_indices()
    constants = {}

    def allocate(name, value):
        constants[name] = int(image.allocate(value.to(torch.bfloat16).contiguous(), alignment=128))

    weight_in = torch.zeros(WIDTH, FEATURE_DIM)
    weight_in[:CODE_DIM] = w_in
    bias_in = torch.zeros(WIDTH)
    bias_in[:CODE_DIM] = b_in
    weight_out = torch.zeros(FEATURE_DIM, WIDTH)
    weight_out[:, :CODE_DIM] = w_out
    score_weights = torch.zeros(CODEBOOK_SIZE, WIDTH)
    score_weights[:, :CODE_DIM] = F.normalize(codebook, dim=1)[order]
    payload = torch.zeros(PAYLOAD_ROWS - 1, CODEBOOK_SIZE)
    payload[:CODE_DIM] = codebook[order].t()
    payload[CODE_DIM] = order & 255
    payload[CODE_DIM + 1] = order >> 8
    tokens = torch.zeros(WIDTH, WIDTH)
    tokens[0, CODE_DIM] = 1
    tokens[1, CODE_DIM + 1] = 1
    for name, value in (
            ("in_weight", weight_in), ("in_bias", bias_in),
            ("out_weight", weight_out), ("out_bias", b_out),
            ("score_weight", score_weights), ("payload", payload),
            ("identity", torch.eye(WIDTH)), ("sum_weight", torch.ones(WIDTH, WIDTH)),
            ("ones", torch.ones(CODEBOOK_SIZE)),
            ("zeros", torch.zeros(WIDTH, WIDTH)), ("token_weight", tokens)):
        allocate(name, value)
    gathers = {}
    for count in (64, 32, 16, 8, 4, 2):
        pair = []
        for start in (0, count // 2):
            select = torch.zeros(WIDTH, WIDTH)
            rows = torch.arange(count // 2)
            select[rows, rows + start] = 1
            pair.append(int(image.allocate(select.to(torch.bfloat16), alignment=128)))
        gathers[count] = tuple(pair)
    return QuantizerPlan(frames, source_address, destination_address, token_address,
                         scratch_address, scratch_bytes,
                         {k: scratch_address + v for k, v in offsets.items()},
                         constants, gathers)


def _sram_positive_mask(engine, source, elements, *, scratch_address):
    from bigcodec_device import sram_clamp, sram_scale
    for scalar in (2.0 ** 126, 128.0, None):
        sram_clamp(engine, source, source, elements,
                   scratch_address=scratch_address, lo=0.0, hi=1.0)
        if scalar is not None:
            sram_scale(engine, source, source, elements, scalar)


def emit_positive_mask(engine, *, source: int, destination: int, elements: int,
                       identity_address: int) -> None:
    """Emit a finite 0/1 comparison mask, subject to device subnormal flushing.

    Clamping before each scale keeps every intermediate finite. 2**126 followed
    by 2**7 also covers subnormals when the arithmetic preserves them. Exhaustive
    finite-pattern testing on RK build 0x40519e0a found 63 value mismatches
    against x > 0: positive BF16 encodings 0x0001..0x003f, below 2**-127, become
    zero. All larger positive gaps, negative inputs and zeros were classified
    correctly. Some zero results have a negative sign bit. A flushed positive
    gap therefore keeps the left candidate as a tie. The older identity-dot
    mask on 0xdf0749de flushed all 127 positive BF16 subnormals below 2**-126.
    """
    if elements < WIDTH or elements % WIDTH:
        raise ValueError("Mask element count must be a positive multiple of 64")
    # Keep every rounding boundary, but load/store each tile only once. The
    # native wide MAXPOOL comparator replaces the three identity-matrix dots.
    # identity_address remains part of the public interface for existing plans.
    for offset in range(0, elements, 32768):
        take = min(32768, elements - offset)
        engine.accelerator_memory_to_sram(source + offset * 2, 0, take)
        _sram_positive_mask(engine, 0, take, scratch_address=0x20000)
        engine.sram_to_accelerator_memory(0, destination + offset * 2, take)


def _emit_sram_tournament(engine, plan):
    """Reduce 8192 candidates to 64 entirely in SRAM, preserving BF16 choices.

    The last six rounds retain the existing gather-matrix implementation for
    sub-vector candidate counts. A/B banks and all temporary ranges are fixed
    and disjoint; each multiply and addition still writes BF16 independently.
    """
    from bigcodec_device import sram_copy, sram_scale, sram_shift
    current, following = 0, 0x30000
    mask, clamp_work, inverse, work = 0x60000, 0x62000, 0x66000, 0x68000
    mask_b, inverse_b, product_b = 0x80000, 0x82000, 0x84000
    engine.accelerator_memory_to_sram(plan.constants['payload'], current,
                                      (PAYLOAD_ROWS - 1) * CODEBOOK_SIZE)
    engine.accelerator_memory_to_sram(
        plan.scratch['ping'] + SCORE_ROW * CODEBOOK_SIZE * 2,
        current + SCORE_ROW * CODEBOOK_SIZE * 2, CODEBOOK_SIZE)
    count = CODEBOOK_SIZE
    while count > WIDTH:
        half = count // 2
        left_score = current + SCORE_ROW * count * 2
        right_score = left_score + half * 2
        sram_copy(engine, left_score, mask_b, half)
        engine.eltwise_sub_core(right_score, mask_b, mask, half)
        _sram_positive_mask(engine, mask, half, scratch_address=clamp_work)
        sram_copy(engine, mask, mask_b, half)
        # The mask is exactly 0 or 1, so this matches rounded(1-mask) exactly.
        sram_scale(engine, mask, inverse, half, -1)
        sram_shift(engine, inverse, inverse, half, 1)
        sram_copy(engine, inverse, inverse_b, half)
        for row in range(PAYLOAD_ROWS):
            left = current + row * count * 2
            right = left + half * 2
            engine.eltwise_mul_core(left, inverse_b, product_b, half)
            engine.eltwise_mul_core(right, mask_b, work, half)
            engine.eltwise_add_core(work, product_b, following + row * half * 2, half)
        current, following = following, current
        count = half
    engine.sram_to_accelerator_memory(current, plan.scratch['ping'], PAYLOAD_ROWS * WIDTH)


def _emit_rsqrt(engine, source: int, destination: int, rows: int, identity: int):
    engine.accelerator_memory_to_sram(identity, 0x80000, WIDTH * WIDTH)
    for row in range(rows):
        engine.accelerator_memory_to_sram(source + row * WIDTH * 2, 0, WIDTH)
        engine.start_queue_for_bf16_matvec_operation(
            max_clear_en=0, fmax_context_addr=0, vector_sram_start_addr=0,
            matrix_sram_start_addr=0x80000, output_sram_wb_addr=0,
            K=WIDTH, N=WIDTH, lalu_mode=udc.LALU_MODE.MODE_RSQRT,
            lalu_scalar=engine.float_to_bf19(1.0))
        engine.sram_to_accelerator_memory(0, destination + row * WIDTH * 2, WIDTH)


def emit_quantizer(engine, plan: QuantizerPlan) -> None:
    """Append the complete quantizer; caller owns START/HALT and host transfers."""
    c, s, frames = plan.constants, plan.scratch, plan.frames
    engine.matmat_mul_core(
        M=frames, K=FEATURE_DIM, N=WIDTH, A_DRAM_ADDR=plan.source_address,
        B_DRAM_ADDR=c["in_weight"], C_DRAM_ADDR=c["in_bias"],
        OUTPUT_DRAM_ADDR=s["projected"])
    engine.eltwise_core_dram(
        M=frames, N=WIDTH, dram_a=s["projected"], dram_b=s["projected"],
        dram_out=s["norm_work"], mode=udc.UE_MODE.ELTWISE_MUL)
    # The padded lanes are zero. Repeating the sum in every lane permits a
    # pointwise rsqrt and multiply, including the upstream zero-input epsilon.
    engine.matmat_mul_core(
        M=frames, K=WIDTH, N=WIDTH, A_DRAM_ADDR=s["norm_work"],
        B_DRAM_ADDR=c["sum_weight"], OUTPUT_DRAM_ADDR=s["norm_work"],
        clamp_enable=True, clamp_min=1e-24)
    _emit_rsqrt(engine, s["norm_work"], s["norm_work"], frames, c["identity"])
    engine.eltwise_core_dram(
        M=frames, N=WIDTH, dram_a=s["projected"], dram_b=s["norm_work"],
        dram_out=s["normalized"], mode=udc.UE_MODE.ELTWISE_MUL)

    def binary(left, right, out, count, mode):
        engine.eltwise_core_dram(M=1, N=count, dram_a=left, dram_b=right,
                                dram_out=out, mode=mode)

    for frame in range(frames):
        current, following = s["ping"], s["pong"]
        if not plan.use_sram_tournament:
            engine.accelerator_memcpy(c["payload"], current,
                                      (PAYLOAD_ROWS - 1) * CODEBOOK_SIZE * 2)
        engine.matmat_mul_core(
            M=1, K=WIDTH, N=CODEBOOK_SIZE,
            A_DRAM_ADDR=s["normalized"] + frame * WIDTH * 2,
            B_DRAM_ADDR=c["score_weight"],
            OUTPUT_DRAM_ADDR=current + SCORE_ROW * CODEBOOK_SIZE * 2)
        count = CODEBOOK_SIZE
        if plan.use_sram_tournament:
            _emit_sram_tournament(engine, plan)
            count = WIDTH
        while count > 1:
            half = count // 2
            next_stride = max(WIDTH, half)
            if count <= WIDTH:
                left_weight, right_weight = plan.small_gathers[count]
                for weight, output in ((left_weight, s["gather_left"]),
                                       (right_weight, s["gather_right"])):
                    engine.matmat_mul_core(
                        M=PAYLOAD_ROWS, K=WIDTH, N=WIDTH,
                        A_DRAM_ADDR=current, B_DRAM_ADDR=weight,
                        OUTPUT_DRAM_ADDR=output)
                left, right, stride = s["gather_left"], s["gather_right"], WIDTH
            else:
                left, right, stride = current, current + half * 2, count
            binary(right + SCORE_ROW * stride * 2,
                   left + SCORE_ROW * stride * 2, s["mask"], next_stride,
                   udc.UE_MODE.ELTWISE_SUB)
            emit_positive_mask(engine, source=s["mask"], destination=s["mask"],
                               elements=next_stride, identity_address=c["identity"])
            binary(c["ones"], s["mask"], s["inverse_mask"], next_stride,
                   udc.UE_MODE.ELTWISE_SUB)
            for row in range(PAYLOAD_ROWS):
                binary(left + row * stride * 2, s["inverse_mask"], s["work_left"],
                       next_stride, udc.UE_MODE.ELTWISE_MUL)
                binary(right + row * stride * 2, s["mask"], s["work_right"],
                       next_stride, udc.UE_MODE.ELTWISE_MUL)
                binary(s["work_left"], s["work_right"],
                       following + row * next_stride * 2, next_stride,
                       udc.UE_MODE.ELTWISE_ADD)
            current, following = following, current
            count = half
        # Transpose the 11 winner scalars into a single 64-lane feature row.
        engine.accelerator_memcpy(c["zeros"], s["transpose_in"], WIDTH * WIDTH * 2)
        engine.accelerator_memcpy(current, s["transpose_in"], PAYLOAD_ROWS * WIDTH * 2)
        engine.bf16_transpose_core(
            M=WIDTH, N=WIDTH, INPUT_DRAM_ADDR=s["transpose_in"],
            OUTPUT_DRAM_ADDR=s["transpose_out"], IDENTITY_DRAM_ADDR=c["identity"])
        engine.matmat_mul_core(
            M=1, K=WIDTH, N=FEATURE_DIM, A_DRAM_ADDR=s["transpose_out"],
            B_DRAM_ADDR=c["out_weight"], C_DRAM_ADDR=c["out_bias"],
            OUTPUT_DRAM_ADDR=plan.destination_address + frame * FEATURE_DIM * 2)
        engine.matmat_mul_core(
            M=1, K=WIDTH, N=WIDTH, A_DRAM_ADDR=s["transpose_out"],
            B_DRAM_ADDR=c["token_weight"],
            OUTPUT_DRAM_ADDR=plan.token_address + frame * WIDTH * 2)


def decode_split_tokens(values: torch.Tensor) -> torch.Tensor:
    """Validate and decode downloaded [T,64] BF16 low/high token rows."""
    values = torch.as_tensor(values).float()
    if values.ndim != 2 or values.shape[1] != WIDTH or values.shape[0] == 0:
        raise ValueError("Expected nonempty [frames,64] token rows")
    if (not torch.isfinite(values).all() or (values[:, 2:] != 0).any()
            or (values[:, :2] != values[:, :2].round()).any()
            or (values[:, :2] < 0).any() or (values[:, 0] > 255).any()
            or (values[:, 1] > 31).any()):
        raise ValueError("Invalid split-byte BigCodec tokens")
    return values[:, 0].long() | (values[:, 1].long() << 8)
