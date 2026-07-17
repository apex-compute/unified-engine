#!/usr/bin/env python3
"""
MobileNetV2 (1.0, 224) inference on accelerator.

  - Config from mobilenetv2_config.json; weights from a single bin (see below).
  - Forward pass: stem 3x3 conv -> 17 inverted residual blocks -> 1x1 head ->
                  global avg pool -> classifier (1001-class w/ background).
  - Inverted residual: expand 1x1 -> depthwise 3x3 -> project 1x1 (linear),
    plus residual when stride==1 and in_ch==out_ch.

Weights:
  - HF model (google/mobilenet_v2_1.0_224) downloaded into mobilenetv2_bin/hf_model
    and remapped + packed into params DRAM at weight_init() time. BN is folded
    into adjacent conv weights (no runtime BN).

Usage:
  python mobilenetv2_test.py
  python mobilenetv2_test.py --image path/to/image.jpg
  python mobilenetv2_test.py --dev xdma0 [--device kintex7] [--cycle 5.15]

Fixed layout: mobilenetv2_test.py, mobilenetv2_config.json, mobilenetv2_bin/
live in the same folder. user_dma_core.py is two folders up (repo root);
that directory is added to sys.path.
"""

import builtins
import json
import math
import os
import sys
import warnings
warnings.filterwarnings("ignore", message=".*torchao.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageClassification
from huggingface_hub import snapshot_download
import time

_original_print = builtins.print
_SILENT_MODE = False

def quiet_print(*args, **kwargs):
    if _SILENT_MODE:
        return
    _original_print(*args, **kwargs)

builtins.print = quiet_print

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, DMA_DEVICE_C2H, DRAM_INSTRUCTION_ADDR, TYPE,
    UE_VECTOR_SIZE, URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS,
    configure_device, UnifiedEngine,
    UE_MODE, URAM_SECTION, URAM_WRITE_SRC, BROADCAST_MODE, LALU_MODE,
)

_BPE = 2  # bf16 bytes per element

# MobileNetV2 uses ReLU6 (clamp 0..6). The UE's LALU_MODE.CLAMP performs
# max(a, min(x, b)) where a, b are bf16 fields baked into each instruction.
# matmat_mul_core's `clamp_enable=True` selects LALU CLAMP; the bounds come from
# the `clamp_min` (default 0.0) and `clamp_max` (default +inf) kwargs. We pass
# `clamp_max=6.0` at every ReLU6 matmul call so the clamp is ReLU6 rather than
# plain ReLU. (Older library versions read these bounds from module constants
# `LALU_CLAMP_RELU_A`/`LALU_CLAMP_RELU_B`; the synced core takes them as kwargs.)

# ---------------------------------------------------------------------------
# Helper ops (built on top of UnifiedEngine primitives)
# ---------------------------------------------------------------------------

def dram_zero_fill(ue: UnifiedEngine, DRAM_ADDR: int, num_elements: int) -> None:
    """Fill a DRAM region with zeros using SRAM as staging buffer."""
    from user_dma_core import URAM_SECTION, URAM_START_ADDR
    chunk_elems = min(URAM_NEAR_FULL_ELEMENTS, num_elements)
    zeros = torch.zeros(chunk_elems, dtype=torch.bfloat16)
    zeros_dram = ue.get_params_dram_addr()
    ue.allocate_params_dram(chunk_elems * _BPE)
    ue.dma_write(DMA_DEVICE_H2C, zeros_dram, zeros, chunk_elems * _BPE)
    ue.accelerator_memory_to_sram(
        accelerator_dram_address=zeros_dram,
        sram_address=0x00000,
        element_size=chunk_elems,
    )
    offset = 0
    while offset < num_elements:
        take = min(chunk_elems, num_elements - offset)
        ue.sram_to_accelerator_memory(
            sram_address=0x00000,
            accelerator_dram_address=DRAM_ADDR + offset * _BPE,
            element_size=take,
        )
        offset += take


def conv2d_1x1_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                    WEIGHT_DRAM_ADDR: int, BIAS_DRAM_ADDR: int,
                    H: int, W: int, C_in: int, C_out: int,
                    relu6_enable: bool = False) -> None:
    """Conv2d(kernel=1) = matmul. HWC layout. Weight (C_out, C_in).
    relu6_enable -> clamp_enable on the matmul, with clamp_max=6.0 passed so
    the clamp is ReLU6 (upper bound 6.0) rather than plain ReLU (+inf).

    PBI: if ``ue._pbi_M_reg`` is set (a GPR index 1..15), the matmul's M-tile
    loop trip count is taken from that register at runtime instead of being
    unrolled at compile time. We prime it with this layer's M=H*W via ADD_SET
    just before the call, collapsing thousands of static M-tile instructions
    (M can reach H*W ~= 25k on early SSD levels) into one hardware loop body.
    Absent the attribute (e.g. the classifier test), the legacy path is used.
    """
    gpr_M_reg = getattr(ue, "_pbi_M_reg", None)
    if gpr_M_reg is not None:
        # matmat_mul_core_pbi grabs 4 isa-regs + 13 inst-ptrs monotonically and
        # never frees them; across the ~35 pointwise convs that would exhaust the
        # 15-entry register file. Restart the transient allocators just above the
        # held M-reg before every call so that scratch gets reused, while the
        # M-reg itself (allocated first, index gpr_M_reg) is preserved.
        ue._isa_reg_counter = gpr_M_reg + 1
        ue.reset_inst_ptr_counter()
        ue.generate_instruction_add_set(dst_reg_idx=gpr_M_reg, immediate_value=H * W)
    ue.matmat_mul_core(
        M=H * W, K=C_in, N=C_out,
        A_DRAM_ADDR=INPUT_DRAM_ADDR, B_DRAM_ADDR=WEIGHT_DRAM_ADDR,
        OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
        C_DRAM_ADDR=BIAS_DRAM_ADDR, bias_mode="broadcast_N",
        clamp_enable=relu6_enable, clamp_max=6.0,
        gpr_M_reg=gpr_M_reg,
    )


def conv2d_3x3_stride2_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                             IM2COL_DRAM_ADDR: int, WEIGHT_DRAM_ADDR: int, BIAS_DRAM_ADDR: int,
                             ZERO_PAD_DRAM_ADDR: int,
                             H_in: int, W_in: int, C_in: int, C_out: int,
                             relu6_enable: bool = False) -> None:
    """Conv2d(kernel=3, stride=2, pad=1) via row-by-row im2col + matmul. HWC layout.
    Weight pre-reshaped (C_out, 9*C_in). C_in and C_out multiples of 64.
    IM2COL_DRAM needs W_out*9*C_in elements (reused per row).
    """
    K = 9 * C_in
    H_out = (H_in - 1) // 2 + 1
    W_out = (W_in - 1) // 2 + 1
    W_CHUNK = max(1, URAM_NEAR_FULL_ELEMENTS // K)
    for h_out in range(H_out):
        h_in = h_out * 2
        for w_start in range(0, W_out, W_CHUNK):
            w_end = min(w_start + W_CHUNK, W_out)
            w_count = w_end - w_start
            sram_off = 0
            for w_out in range(w_start, w_end):
                w_in = w_out * 2
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        nh, nw = h_in + dy, w_in + dx
                        patch_sram = sram_off + ((dy + 1) * 3 + (dx + 1)) * C_in * _BPE
                        if 0 <= nh < H_in and 0 <= nw < W_in:
                            ue.accelerator_memory_to_sram(
                                INPUT_DRAM_ADDR + (nh * W_in + nw) * C_in * _BPE,
                                patch_sram, C_in)
                        else:
                            ue.accelerator_memory_to_sram(ZERO_PAD_DRAM_ADDR, patch_sram, C_in)
                sram_off += K * _BPE
            ue.sram_to_accelerator_memory(0x00000, IM2COL_DRAM_ADDR, w_count * K)
            ue.matmat_mul_core(
                M=w_count, K=K, N=C_out,
                A_DRAM_ADDR=IM2COL_DRAM_ADDR,
                B_DRAM_ADDR=WEIGHT_DRAM_ADDR,
                OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR + (h_out * W_out + w_start) * C_out * _BPE,
                C_DRAM_ADDR=BIAS_DRAM_ADDR, bias_mode="broadcast_N",
                clamp_enable=relu6_enable, clamp_max=6.0)


def conv2d_3x3_dw_tapwise_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                                ACC_DRAM_ADDR: int, IDENTITY_DRAM_ADDR: int, ZERO_PAD_DRAM_ADDR: int,
                                block_params: list, H_in: int, W_in: int, C: int,
                                stride: int = 1, W_chunk_max: int = 32,
                                relu6_enable: bool = True) -> None:
    """Depthwise Conv2d(kernel=3, pad=1, groups=C) via tap-wise eltwise mul+add.

    Replaces the block-diagonal matmul (which multiplied through 63/64 zero
    columns per row, wasting 64x the arithmetic) with the exact-compute form:

        out[h, w, c] = bias[c] + sum_{(dy,dx) in 3x3} in[h+dy, w+dx, c] * w[c,dy,dx]

    For each 64-channel block we hold the accumulator (W_chunk x 64) in URAM_B,
    gather the shifted input slab (W_chunk x 64) into URAM_A per tap, multiply
    by a pre-tiled per-channel weight tile (URAM_B), and add into the
    accumulator. After all 9 taps + bias, an identity-matrix matmul with
    clamp_enable=True does the ReLU6 (the only validated CLAMP path on this HW
    is via matmat_mul_core). HWC layout. C must be a multiple of 64. stride
    must be 1 or 2.

    block_params: list of (C//64) dicts, each {"taps": [9 DRAM addrs],
    "bias": DRAM addr}. Each tile is (W_chunk_max, 64), every row a copy of
    the per-channel kernel column for that tap (or bias column).

    ACC_DRAM_ADDR: scratch buffer sized (max_H_out * max_W_out * 64), reused.
    IDENTITY_DRAM_ADDR: a (64, 64) bf16 identity matrix for the clamp pass.
    """
    BLK = 64
    n_blocks = C // BLK
    assert n_blocks == len(block_params)
    if stride == 1:
        H_out, W_out = H_in, W_in
    elif stride == 2:
        H_out = (H_in - 1) // 2 + 1
        W_out = (W_in - 1) // 2 + 1
    else:
        raise ValueError(f"unsupported stride={stride}")

    # SRAM layout (128-byte aligned, banks A/B kept disjoint where eltwise
    # ops require it: eltwise_mul/eltwise_add both assert A and B sit in
    # different URAMs).
    A_SRAM    = 0x10000  # URAM_A — gathered shifted-input slab
    B_SRAM    = 0x80000  # URAM_B — weight tile / bias tile
    TEMP_SRAM = 0x40000  # URAM_A — eltwise_mul output (different bank from ACC)
    ACC_SRAM  = 0xC0000  # URAM_B — running accumulator
    SCAT_SRAM = 0x00000  # URAM_A — scratch for the final strided scatter

    W_chunk = min(W_chunk_max, W_out)

    for blk in range(n_blocks):
        c_off = blk * BLK
        bp = block_params[blk]
        tap_addrs = bp["taps"]
        bias_addr = bp["bias"]

        # (1) Tap-wise accumulate every output pixel of this 64-block into
        #     ACC_DRAM (contiguous (H_out*W_out, 64) layout — single block).
        for h_out in range(H_out):
            h_in = h_out * stride
            for w_start in range(0, W_out, W_chunk):
                w_end = min(w_start + W_chunk, W_out)
                w_count = w_end - w_start

                # ACC <- bias_tile. The bias_tile is (W_chunk_max, 64) where
                # every row equals the (64,) bias column for this block, so
                # loading w_count*64 elements broadcasts bias across pixels.
                ue.accelerator_memory_to_sram(bias_addr, ACC_SRAM, w_count * BLK)

                for tap_idx in range(9):
                    dy = (tap_idx // 3) - 1
                    dx = (tap_idx % 3) - 1

                    # Strided gather: each output pixel pulls 64 channels from
                    # input at (h_in+dy, w_in+dx). The 64-channel slabs for
                    # consecutive output pixels sit a constant ``stride*C`` bytes
                    # apart in DRAM, so the whole in-bounds run of the row is ONE
                    # strided DMA (chunk=64ch=128B, jump=stride*C*_BPE) instead of
                    # w_count per-pixel DMAs. Out-of-bounds edge pixels (at most
                    # the first/last column, or the whole row when nh is OOB) are
                    # zero-filled per-pixel from ZERO_PAD — those are rare.
                    nh = h_in + dy
                    if not (0 <= nh < H_in):
                        # Entire row out of bounds (top/bottom halo): zero all.
                        for w_offset in range(w_count):
                            ue.accelerator_memory_to_sram(
                                ZERO_PAD_DRAM_ADDR, A_SRAM + w_offset * BLK * _BPE, BLK)
                    else:
                        # Valid output-pixel window [p_first, p_last] (global w),
                        # i.e. those with 0 <= p*stride+dx < W_in.
                        p_first = 1 if dx < 0 else 0            # ceil(-dx/stride) for dx in {-1,0,1}
                        p_last = (W_in - 1 - dx) // stride      # floor((W_in-1-dx)/stride)
                        lo = max(w_start, p_first) - w_start
                        hi = min(w_end - 1, p_last) - w_start + 1
                        # OOB columns before the valid window -> zero per-pixel.
                        for w_offset in range(0, lo):
                            ue.accelerator_memory_to_sram(
                                ZERO_PAD_DRAM_ADDR, A_SRAM + w_offset * BLK * _BPE, BLK)
                        # In-bounds run -> single strided DMA.
                        if hi > lo:
                            nw0 = (w_start + lo) * stride + dx
                            ue.accelerator_memory_to_sram(
                                INPUT_DRAM_ADDR + (nh * W_in + nw0) * C * _BPE + c_off * _BPE,
                                A_SRAM + lo * BLK * _BPE, BLK,
                                memcpy_length_bytes=(hi - lo) * BLK * _BPE,
                                stride_bytes_per_chunk=BLK * _BPE,
                                stride_jump_bytes=stride * C * _BPE,
                            )
                        # OOB columns after the valid window -> zero per-pixel.
                        for w_offset in range(hi, w_count):
                            ue.accelerator_memory_to_sram(
                                ZERO_PAD_DRAM_ADDR, A_SRAM + w_offset * BLK * _BPE, BLK)

                    # Weight tile for this tap broadcasts the same (64,)
                    # kernel column across all w_count rows.
                    ue.accelerator_memory_to_sram(tap_addrs[tap_idx], B_SRAM, w_count * BLK)

                    ue.eltwise_mul_core(A_SRAM, B_SRAM, TEMP_SRAM, w_count * BLK)
                    ue.eltwise_add_core(TEMP_SRAM, ACC_SRAM, ACC_SRAM, w_count * BLK)

                # Write completed ACC slab to scratch DRAM (H_out*W_out, 64) at
                # the (h_out, w_start, :) location.
                ue.sram_to_accelerator_memory(
                    ACC_SRAM,
                    ACC_DRAM_ADDR + (h_out * W_out + w_start) * BLK * _BPE,
                    w_count * BLK)

        # (2) ReLU6: in-place clamp matmul (M=H_out*W_out, K=64, N=64). A @ I
        #     yields A bit-exactly (0.0 and 1.0 are both exact bf16), so the
        #     only real effect is the LALU CLAMP on the output, applying
        #     ReLU6 with bounds (0, 6.0) via clamp_max=6.0.
        if relu6_enable:
            ue.matmat_mul_core(
                M=H_out * W_out, K=BLK, N=BLK,
                A_DRAM_ADDR=ACC_DRAM_ADDR,
                B_DRAM_ADDR=IDENTITY_DRAM_ADDR,
                OUTPUT_DRAM_ADDR=ACC_DRAM_ADDR,
                C_DRAM_ADDR=None, bias_mode="broadcast_N",
                clamp_enable=True, clamp_max=6.0,
            )

        # (3) Strided scatter from contiguous scratch to OUTPUT_DRAM at the
        #     channel offset c_off (HWC interleaved layout, stride C bytes).
        for h_out in range(H_out):
            for w_start in range(0, W_out, W_chunk):
                w_end = min(w_start + W_chunk, W_out)
                w_count = w_end - w_start
                ue.accelerator_memory_to_sram(
                    ACC_DRAM_ADDR + (h_out * W_out + w_start) * BLK * _BPE,
                    SCAT_SRAM, w_count * BLK)
                ue.sram_to_accelerator_memory(
                    SCAT_SRAM,
                    OUTPUT_DRAM_ADDR + (h_out * W_out + w_start) * C * _BPE + c_off * _BPE,
                    w_count * BLK,
                    stride_bytes_per_chunk=BLK * _BPE,
                    stride_jump_bytes=C * _BPE)


def eltwise_add_dram(ue: UnifiedEngine, A_ADDR: int, B_ADDR: int,
                     OUT_ADDR: int, num_elements: int) -> None:
    """Element-wise add of two DRAM buffers, written back to OUT_ADDR.
    Used for the residual connection in inverted residual blocks.
    """
    chunk = min(URAM_NEAR_FULL_ELEMENTS, num_elements)
    offset = 0
    while offset < num_elements:
        take = min(chunk, num_elements - offset)
        ue.accelerator_memory_to_sram(A_ADDR + offset * _BPE, 0x00000, take)
        ue.accelerator_memory_to_sram(B_ADDR + offset * _BPE, 0x80000, take)
        ue.eltwise_add_core(
            vector_A_sram_start_addr=0x00000,
            vector_B_sram_start_addr=0x80000,
            vector_C_sram_wb_addr=0x00000,
            element_size=take,
        )
        ue.sram_to_accelerator_memory(0x00000, OUT_ADDR + offset * _BPE, take)
        offset += take


def global_avgpool_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                        H: int, W: int, C: int) -> None:
    """Spatial avg-pool over an HWC buffer: sum all H*W rows then scale by 1/(H*W).
    Mirrors swin's post-encoder pool: row-wise eltwise accumulate then broadcast_mul.
    """
    dram_zero_fill(ue, OUTPUT_DRAM_ADDR, C)
    for row in range(H * W):
        ue.accelerator_memory_to_sram(INPUT_DRAM_ADDR + row * C * _BPE, 0x00000, C)
        ue.accelerator_memory_to_sram(OUTPUT_DRAM_ADDR, 0x80000, C)
        ue.eltwise_add_core(
            vector_A_sram_start_addr=0x00000,
            vector_B_sram_start_addr=0x80000,
            vector_C_sram_wb_addr=0x80000,
            element_size=C,
        )
        ue.sram_to_accelerator_memory(0x80000, OUTPUT_DRAM_ADDR, C)
    ue.accelerator_memory_to_sram(OUTPUT_DRAM_ADDR, 0x00000, C)
    ue.broadcast_mul(
        scalar=1.0 / (H * W),
        sram_start_addr=0x00000,
        sram_wb_addr=0x00000,
        element_size=C,
    )
    ue.sram_to_accelerator_memory(0x00000, OUTPUT_DRAM_ADDR, C)


# ---------------------------------------------------------------------------
# Weight prep helpers
# ---------------------------------------------------------------------------

def _bn_fold(w_conv, bn_w, bn_b, bn_mean, bn_var, eps=1e-3):
    """Fold BatchNorm into conv weight and bias at load time.

    Returns (w_fused, b_fused) where:
      w_fused[c] = w_conv[c] * (bn_w[c] / sqrt(bn_var[c] + eps))
      b_fused[c] = bn_b[c]   - bn_mean[c] * bn_w[c] / sqrt(bn_var[c] + eps)

    HF MobileNetV2 uses BN eps=1e-3 (TF default), NOT torch's 1e-5.
    """
    scale = bn_w / (bn_var + eps).sqrt()
    w_fused = w_conv * scale.view(-1, *([1] * (w_conv.dim() - 1)))
    b_fused = bn_b - bn_mean * scale
    return w_fused.to(torch.bfloat16), b_fused.to(torch.bfloat16)


def _pad_channels_in_weight(w: torch.Tensor, C_in_padded: int, C_out_padded: int) -> torch.Tensor:
    """Pad a Conv weight (C_out, C_in, kH, kW) to (C_out_padded, C_in_padded, kH, kW)
    with zeros. Padded channels contribute zero to the output and zero from the input.
    """
    C_out, C_in = w.shape[0], w.shape[1]
    out = torch.zeros(C_out_padded, C_in_padded, *w.shape[2:], dtype=w.dtype)
    out[:C_out, :C_in] = w
    return out


def _pack_dw_tapwise(w_dw_padded: torch.Tensor, bias_padded: torch.Tensor,
                     W_chunk_max: int) -> list:
    """Tile a BN-folded depthwise weight + bias for the tap-wise eltwise scheme.

    Per 64-channel block, returns:
      - 9 weight tiles (one per (dy, dx) tap), each (W_chunk_max, 64), every
        row a copy of the per-channel kernel column w[c0:c0+64, 0, dy, dx].
        eltwise_mul against the gathered input slab then applies the per-
        channel kernel value uniformly across all pixels in the chunk —
        exactly what depthwise needs.
      - 1 bias tile (W_chunk_max, 64), same broadcasting trick.

    Returns: list of (C//64) dicts, each {"taps": [9 tiles], "bias": tile}.
    """
    BLK = 64
    assert w_dw_padded.shape[1] == 1 and w_dw_padded.shape[2] == 3 and w_dw_padded.shape[3] == 3
    C = w_dw_padded.shape[0]
    assert C % BLK == 0
    n_blocks = C // BLK
    blocks = []
    for b in range(n_blocks):
        c0 = b * BLK
        taps = []
        for dy in range(3):
            for dx in range(3):
                col = w_dw_padded[c0:c0 + BLK, 0, dy, dx]  # (64,)
                tile = col.unsqueeze(0).expand(W_chunk_max, BLK).contiguous()
                taps.append(tile)
        bias_col = bias_padded[c0:c0 + BLK]  # (64,)
        bias_tile = bias_col.unsqueeze(0).expand(W_chunk_max, BLK).contiguous()
        blocks.append({"taps": taps, "bias": bias_tile})
    return blocks


# ---------------------------------------------------------------------------
# Config / HF model
# ---------------------------------------------------------------------------

def _ensure_hf_model(script_dir: str, cfg: dict):
    """Download HF model if not present. Returns (model, model_dir)."""
    paths = cfg["paths"]
    model_dir = os.path.join(script_dir, paths["hf_model_dir"])
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        snapshot_download(repo_id=paths["hf_model_repo"], local_dir=model_dir,
                          ignore_patterns=["*.h5", "*.ot", "*.msgpack"])
    model = AutoModelForImageClassification.from_pretrained(model_dir, dtype=torch.bfloat16, device_map=None)
    model.eval()
    return model, model_dir


# ---------------------------------------------------------------------------
# MobileNetV2 Unified Engine
# ---------------------------------------------------------------------------

# Default DRAM partition over a 4 GB address space.
MBV2_PARAMS_BASE  = 0x00000000
MBV2_TENSOR_BASE  = 0x40000000
MBV2_PROGRAM_BASE = 0x80000000


def _set_dram_layout_for_device(device: str) -> None:
    """Select MobileNetV2 arenas before constructing the engine."""
    global MBV2_PARAMS_BASE, MBV2_TENSOR_BASE, MBV2_PROGRAM_BASE
    if device == "efinix":
        # Fits the 2 GB Efinix DMA aperture and matches existing efinix cache metadata.
        MBV2_PARAMS_BASE = 0x01000000
        MBV2_TENSOR_BASE = 0x10000000
        MBV2_PROGRAM_BASE = 0x20000000
    else:
        MBV2_PARAMS_BASE = 0x00000000
        MBV2_TENSOR_BASE = 0x40000000
        MBV2_PROGRAM_BASE = 0x80000000


def _cache_layout_meta(size: int, extra: dict | None = None) -> dict:
    meta = {
        "size": size,
        "device": user_dma_core.CURRENT_DEVICE,
        "params_dram_base": MBV2_PARAMS_BASE,
        "tensor_dram_base": MBV2_TENSOR_BASE,
        "program_dram_base": MBV2_PROGRAM_BASE,
        "dram_end_addr": user_dma_core.DRAM_END_ADDR,
        "graph_version": 1,
    }
    if extra:
        meta.update(extra)
    return meta


def _cache_layout_matches(meta: dict) -> bool:
    if not meta:
        return False
    expected = {
        "device": user_dma_core.CURRENT_DEVICE,
        "params_dram_base": MBV2_PARAMS_BASE,
        "tensor_dram_base": MBV2_TENSOR_BASE,
        "program_dram_base": MBV2_PROGRAM_BASE,
        "graph_version": 1,
    }
    return all(meta.get(k) == v for k, v in expected.items())


class MobileNetV2_UnifiedEngine(UnifiedEngine):
    """MobileNetV2-1.0 (224x224, 1001-class) on Unified Engine FPGA accelerator."""

    # --- Architecture constants ---
    IMAGE_SIZE      = 224
    NUM_CHANNELS    = 3
    STEM_OUT        = 32
    HEAD_OUT        = 1280
    # (expand_ratio, out_channels, repeats, first_stride) — 17 IR blocks total
    IR_SETTING = [
        (1,  16, 1, 1),
        (6,  24, 2, 2),
        (6,  32, 3, 2),
        (6,  64, 4, 2),
        (6,  96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]
    NUM_LABELS_DEFAULT = 1001  # background + 1000

    ALIGN = 64  # UE_VECTOR_SIZE — channel and patch dims pad to this
    DMA_CHUNK_BYTES = 1 * 1024 * 1024

    # W_chunk used inside conv2d_3x3_dw_tapwise_dram. Also the row count
    # baked into the pre-tiled depthwise weight / bias tiles in DRAM. Smaller
    # = less params DRAM; larger = fewer eltwise_mul/add instructions per
    # output row (gathers stay 1-per-pixel regardless). 32 hits the sweet
    # spot: full rows of small late-stage layers (W_out=7..14) fit in one
    # chunk, big early layers (W_out=112) get 4 chunks per row.
    DW_W_CHUNK_MAX = 32

    USE_BIN_CACHE = True

    def __init__(self, script_dir: str = SCRIPT_DIR, weights_bin: str | None = None):
        super().__init__(params_dram_base=MBV2_PARAMS_BASE,
                         tensor_dram_base=MBV2_TENSOR_BASE,
                         program_dram_base=MBV2_PROGRAM_BASE)

        cfg = self.load_config(script_dir=script_dir)
        self.cfg = cfg
        self.script_dir = script_dir
        self.weights_bin = weights_bin or os.path.join(script_dir, cfg["paths"]["weights_bin"])

        # Expand the IR setting into a flat list of per-block configs with
        # padded channel counts so weight_init / tensor_init / compile_full_fused
        # can index uniformly.
        self.blocks = self._expand_blocks()

        if self.USE_BIN_CACHE:
            self.params_from_bin = self.load_params()
        else:
            self.params_from_bin = False
        if not self.params_from_bin:
            self.weight_init()
        self.tensor_init()

    @staticmethod
    def load_config(config_path: str | None = None, script_dir: str | None = None) -> dict:
        sd = script_dir or SCRIPT_DIR
        cp = config_path or os.path.join(sd, "mobilenetv2_config.json")
        with open(cp) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def pad_dim(x: int) -> int:
        return ((x + 63) // 64) * 64

    def _alloc_param(self, tensor: torch.Tensor) -> int:
        """Allocate params DRAM, write bf16 tensor. Returns DRAM address."""
        t = tensor.to(torch.bfloat16).contiguous()
        addr = self.get_params_dram_addr()
        nbytes = t.numel() * _BPE
        self.allocate_params_dram(nbytes)
        self.dma_to_accelerator_memory(addr, t)
        return addr

    def _alloc_tensor(self, num_elements: int) -> int:
        return self.allocate_tensor_dram(num_elements * _BPE)

    def write_captured_instructions_to_dram(self, start_addr: int = DRAM_INSTRUCTION_ADDR) -> int:
        """Chunked override to avoid segfault on large instruction DMA writes."""
        if not self.capture_buffer or self.capture_count == 0:
            return 0
        total_bytes = self.capture_count * 32
        all_bytes = bytearray()
        for inst in self.capture_buffer:
            all_bytes.extend(inst.get_bytes())
        data = bytes(all_bytes)
        if not _SILENT_MODE:
            _original_print(f"Writing {self.capture_count:,} instructions ({total_bytes / 1024**2:.1f} MB) to DRAM at 0x{start_addr:x}...")
        offset = 0
        while offset < total_bytes:
            chunk = min(self.DMA_CHUNK_BYTES, total_bytes - offset)
            self.dma_write(DMA_DEVICE_H2C, start_addr + offset, data[offset:offset + chunk], chunk)
            offset += chunk
        if not _SILENT_MODE:
            _original_print(f"Successfully wrote {total_bytes / 1024**2:.1f} MB ({self.capture_count:,} instructions) to DRAM")
        return total_bytes

    def _expand_blocks(self) -> list[dict]:
        """Flatten IR_SETTING into 17 per-block dicts.

        Each dict has:
          in_ch, out_ch       — true MBV2 channel counts
          in_ch_p, out_ch_p   — padded to multiples of 64
          mid_ch, mid_ch_p    — expand intermediate channels (= in_ch * expand_ratio)
          stride              — 1 or 2
          expand_ratio        — 1 or 6 (when 1, the expand 1x1 is skipped)
          has_residual        — stride==1 and in_ch==out_ch
          H_in, W_in          — input spatial size to this block
          H_out, W_out        — output spatial size
        """
        blocks = []
        in_ch = self.STEM_OUT
        H = W = self.IMAGE_SIZE // 2  # after stem stride-2
        for expand_ratio, out_ch, repeats, first_stride in self.IR_SETTING:
            for r in range(repeats):
                stride = first_stride if r == 0 else 1
                mid_ch = in_ch * expand_ratio
                H_out = (H - 1) // stride + 1 if stride > 1 else H
                W_out = (W - 1) // stride + 1 if stride > 1 else W
                blocks.append({
                    "in_ch": in_ch, "out_ch": out_ch,
                    "in_ch_p": self.pad_dim(in_ch), "out_ch_p": self.pad_dim(out_ch),
                    "mid_ch": mid_ch, "mid_ch_p": self.pad_dim(mid_ch),
                    "stride": stride, "expand_ratio": expand_ratio,
                    "has_residual": stride == 1 and in_ch == out_ch,
                    "H_in": H, "W_in": W, "H_out": H_out, "W_out": W_out,
                })
                in_ch = out_ch
                H, W = H_out, W_out
        return blocks

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def weight_init(self) -> None:
        """Load HF model weights to DRAM. BN folded into conv weights.

        HF MBV2 state_dict layout:
          mobilenet_v2.conv_stem.first_conv.{convolution,normalization}.*
          mobilenet_v2.conv_stem.conv_3x3.{convolution,normalization}.*       (block 0 depthwise)
          mobilenet_v2.conv_stem.reduce_1x1.{convolution,normalization}.*     (block 0 project)
          mobilenet_v2.layer.{0..15}.{expand_1x1,conv_3x3,reduce_1x1}.{convolution,normalization}.*
          mobilenet_v2.conv_1x1.{convolution,normalization}.*                 (head)
          classifier.{weight,bias}
        """
        model, _ = _ensure_hf_model(self.script_dir, self.cfg)
        sd = model.state_dict()
        pad = self.pad_dim

        # Shared zero-pad row for im2col boundary fill — largest C we'll ever
        # need is the maximum in_ch_p across blocks plus the head input (320p).
        max_c = max(self.pad_dim(self.STEM_OUT),
                    max(b["in_ch_p"] for b in self.blocks),
                    max(b["mid_ch_p"] for b in self.blocks))
        self.ZERO_PAD = self._alloc_param(torch.zeros(max_c, dtype=torch.bfloat16))

        # Identity (64, 64) bf16 — exact since 0.0 and 1.0 are both exact in
        # bf16. Used as the right operand for each depthwise's post-pass
        # clamp matmul (A @ I = A, with clamp_enable=True applying ReLU6).
        self.IDENTITY_64_DRAM = self._alloc_param(torch.eye(64, dtype=torch.bfloat16))

        # ---- Stem: 3x3 stride-2 conv, 3->32 (padded 64->64) ----
        # Image is padded from 3 to 64 input channels (extra channels zero).
        stem_w = sd["mobilenet_v2.conv_stem.first_conv.convolution.weight"]  # (32, 3, 3, 3)
        stem_w, stem_b = _bn_fold(
            stem_w,
            sd["mobilenet_v2.conv_stem.first_conv.normalization.weight"],
            sd["mobilenet_v2.conv_stem.first_conv.normalization.bias"],
            sd["mobilenet_v2.conv_stem.first_conv.normalization.running_mean"],
            sd["mobilenet_v2.conv_stem.first_conv.normalization.running_var"],
        )
        stem_w_p = _pad_channels_in_weight(stem_w, C_in_padded=64, C_out_padded=64)
        stem_b_p = torch.zeros(64, dtype=torch.bfloat16); stem_b_p[:self.STEM_OUT] = stem_b
        # Permute (C_out, C_in, 3, 3) -> (C_out, 3, 3, C_in) before flattening so
        # the K axis is ordered (spatial * C_in + c_in), matching what im2col
        # writes per pixel: 9 patches of C_in channels each (channel innermost).
        stem_w_im2col = stem_w_p.permute(0, 2, 3, 1).contiguous().reshape(64, 9 * 64)
        self.STEM_W = self._alloc_param(stem_w_im2col)
        self.STEM_B = self._alloc_param(stem_b_p)

        # ---- Per-block weights ----
        self.block_weights: list[dict] = []
        for i, b in enumerate(self.blocks):
            prefix = ("mobilenet_v2.conv_stem"
                      if i == 0
                      else f"mobilenet_v2.layer.{i - 1}")
            bw = {}

            # Expand 1x1 (skipped for first block, expand_ratio=1)
            if b["expand_ratio"] != 1:
                w = sd[f"{prefix}.expand_1x1.convolution.weight"].squeeze(-1).squeeze(-1)  # (mid, in, 1, 1)->(mid, in)
                w_full = sd[f"{prefix}.expand_1x1.convolution.weight"]
                w_fused, b_fused = _bn_fold(
                    w_full,
                    sd[f"{prefix}.expand_1x1.normalization.weight"],
                    sd[f"{prefix}.expand_1x1.normalization.bias"],
                    sd[f"{prefix}.expand_1x1.normalization.running_mean"],
                    sd[f"{prefix}.expand_1x1.normalization.running_var"],
                )
                w_fused = w_fused.squeeze(-1).squeeze(-1)  # (mid, in)
                w_p = torch.zeros(b["mid_ch_p"], b["in_ch_p"], dtype=torch.bfloat16)
                w_p[:b["mid_ch"], :b["in_ch"]] = w_fused
                b_p = torch.zeros(b["mid_ch_p"], dtype=torch.bfloat16); b_p[:b["mid_ch"]] = b_fused
                bw["expand_w"] = self._alloc_param(w_p)
                bw["expand_b"] = self._alloc_param(b_p)

            # Depthwise 3x3 (block-diagonal pack)
            w_full = sd[f"{prefix}.conv_3x3.convolution.weight"]  # (mid, 1, 3, 3)
            w_fused, b_fused = _bn_fold(
                w_full,
                sd[f"{prefix}.conv_3x3.normalization.weight"],
                sd[f"{prefix}.conv_3x3.normalization.bias"],
                sd[f"{prefix}.conv_3x3.normalization.running_mean"],
                sd[f"{prefix}.conv_3x3.normalization.running_var"],
            )
            # Pad channels to mid_ch_p with zero kernels, then tile for the
            # tap-wise depthwise scheme (9 weight tiles + 1 bias tile per
            # 64-channel block).
            w_p = torch.zeros(b["mid_ch_p"], 1, 3, 3, dtype=torch.bfloat16)
            w_p[:b["mid_ch"]] = w_fused
            b_p = torch.zeros(b["mid_ch_p"], dtype=torch.bfloat16); b_p[:b["mid_ch"]] = b_fused
            tiles = _pack_dw_tapwise(w_p, b_p, self.DW_W_CHUNK_MAX)
            bw["dw_blocks"] = [
                {
                    "taps": [self._alloc_param(t) for t in blk["taps"]],
                    "bias": self._alloc_param(blk["bias"]),
                }
                for blk in tiles
            ]

            # Project (reduce) 1x1 — LINEAR (no ReLU6), still BN-folded
            w_full = sd[f"{prefix}.reduce_1x1.convolution.weight"]  # (out, mid, 1, 1)
            w_fused, b_fused = _bn_fold(
                w_full,
                sd[f"{prefix}.reduce_1x1.normalization.weight"],
                sd[f"{prefix}.reduce_1x1.normalization.bias"],
                sd[f"{prefix}.reduce_1x1.normalization.running_mean"],
                sd[f"{prefix}.reduce_1x1.normalization.running_var"],
            )
            w_fused = w_fused.squeeze(-1).squeeze(-1)  # (out, mid)
            w_p = torch.zeros(b["out_ch_p"], b["mid_ch_p"], dtype=torch.bfloat16)
            w_p[:b["out_ch"], :b["mid_ch"]] = w_fused
            b_p = torch.zeros(b["out_ch_p"], dtype=torch.bfloat16); b_p[:b["out_ch"]] = b_fused
            bw["project_w"] = self._alloc_param(w_p)
            bw["project_b"] = self._alloc_param(b_p)

            self.block_weights.append(bw)

        # ---- Head: 1x1 conv 320 -> 1280 ----
        w_full = sd["mobilenet_v2.conv_1x1.convolution.weight"]
        w_fused, b_fused = _bn_fold(
            w_full,
            sd["mobilenet_v2.conv_1x1.normalization.weight"],
            sd["mobilenet_v2.conv_1x1.normalization.bias"],
            sd["mobilenet_v2.conv_1x1.normalization.running_mean"],
            sd["mobilenet_v2.conv_1x1.normalization.running_var"],
        )
        w_fused = w_fused.squeeze(-1).squeeze(-1)  # (1280, 320)
        head_in_p = self.pad_dim(320)   # 320 already aligned
        head_out_p = self.pad_dim(1280) # 1280 already aligned
        w_p = torch.zeros(head_out_p, head_in_p, dtype=torch.bfloat16)
        w_p[:1280, :320] = w_fused
        b_p = torch.zeros(head_out_p, dtype=torch.bfloat16); b_p[:1280] = b_fused
        self.HEAD_W = self._alloc_param(w_p)
        self.HEAD_B = self._alloc_param(b_p)

        # ---- Classifier: (num_labels, 1280) ----
        cls_w = model.classifier.weight.data.to(torch.bfloat16)  # (1001, 1280)
        cls_b = model.classifier.bias.data.to(torch.bfloat16)    # (1001,)
        self.NUM_LABELS = cls_w.shape[0]
        self.NUM_LABELS_PAD = self.pad_dim(self.NUM_LABELS)
        cls_w_p = torch.zeros(self.NUM_LABELS_PAD, head_out_p, dtype=torch.bfloat16)
        cls_w_p[:self.NUM_LABELS, :1280] = cls_w
        cls_b_p = torch.zeros(self.NUM_LABELS_PAD, dtype=torch.bfloat16)
        cls_b_p[:self.NUM_LABELS] = cls_b
        self.CLASSIFIER_W = self._alloc_param(cls_w_p)
        self.CLASSIFIER_B = self._alloc_param(cls_b_p)

        del model

    def tensor_init(self) -> None:
        """Allocate intermediate DRAM buffers for forward pass."""
        # Input image in HWC layout, padded to 64 input channels.
        H = self.IMAGE_SIZE
        self.IMAGE_DRAM = self._alloc_tensor(H * H * 64)

        # Per-block scratch:
        #   block_in (HWC, in_ch_p) — pre-block activation (also the residual source)
        #   expanded (HWC, mid_ch_p) — after expand 1x1 + ReLU6
        #   dw_out   (HWC_out, mid_ch_p) — after depthwise 3x3 + ReLU6
        #   block_out (HWC_out, out_ch_p) — after project 1x1 (linear)
        #
        # We chain block_out -> next block_in by aliasing addresses; allocate
        # separate per-block buffers for clarity. A future pass can A/B ping-pong.
        self.block_tensors: list[dict] = []
        # IM2COL scratch sized to the worst case across stem/depthwise.
        max_im2col_dw = max(
            max(1, URAM_NEAR_FULL_ELEMENTS // (9 * 64 + 64)) * 9 * 64 + max(1, URAM_NEAR_FULL_ELEMENTS // (9 * 64 + 64)) * 64
            for _ in [0]  # constant across blocks (depthwise uses 64-channel slabs)
        )
        max_im2col_stem = max(1, URAM_NEAR_FULL_ELEMENTS // (9 * 64)) * 9 * 64
        self.IM2COL_DRAM = self._alloc_tensor(max(max_im2col_dw, max_im2col_stem))

        # Stem output: 32-channel-padded-to-64, spatial IMAGE_SIZE/2
        stem_spatial = self.IMAGE_SIZE // 2
        self.STEM_OUT_DRAM = self._alloc_tensor(stem_spatial * stem_spatial * 64)

        # Per-block depthwise accumulator scratch — sized to the largest
        # (H_out * W_out * 64) across all dw layers. Used contiguously per
        # 64-channel block during the tap-wise accumulate pass, then again
        # in-place as the source/dest of the clamp matmul.
        max_dw_acc_elems = max(b["H_out"] * b["W_out"] * 64 for b in self.blocks)
        self.DW_ACC_DRAM = self._alloc_tensor(max_dw_acc_elems)

        for b in self.blocks:
            t = {}
            t["expanded"] = self._alloc_tensor(b["H_in"] * b["W_in"] * b["mid_ch_p"])
            t["dw_out"]   = self._alloc_tensor(b["H_out"] * b["W_out"] * b["mid_ch_p"])
            t["block_out"] = self._alloc_tensor(b["H_out"] * b["W_out"] * b["out_ch_p"])
            self.block_tensors.append(t)

        # Head output: (7*7, 1280)
        final_spatial = self.blocks[-1]["H_out"]
        self.HEAD_OUT_DRAM = self._alloc_tensor(final_spatial * final_spatial * self.pad_dim(self.HEAD_OUT))
        self.POOLED_DRAM   = self._alloc_tensor(self.pad_dim(self.HEAD_OUT))
        self.CLASSIFIER_OUT_DRAM = self._alloc_tensor(self.pad_dim(self.NUM_LABELS))

    # ------------------------------------------------------------------
    # Bin dump / load (params + programs)
    # ------------------------------------------------------------------

    BIN_SUBDIR = "mobilenetv2_bin"

    def dump_params(self):
        bin_dir = os.path.join(self.script_dir, self.BIN_SUBDIR)
        os.makedirs(bin_dir, exist_ok=True)
        bin_path = os.path.join(bin_dir, "params.bin")
        meta_path = os.path.join(bin_dir, "params.json")
        total = self.get_params_dram_usage()
        CHUNK = 1 * 1024 * 1024
        with open(bin_path, "wb") as f:
            offset = 0
            while offset < total:
                sz = min(CHUNK, total - offset)
                buf = bytearray(sz)
                self.dma_read(DMA_DEVICE_C2H, self._params_dram_base + offset, buf, sz)
                f.write(buf)
                offset += sz
        with open(meta_path, "w") as f:
            json.dump(_cache_layout_meta(total, {"num_labels": self.NUM_LABELS}), f)
        _original_print(f"  Params dumped: {total / 1024**2:.1f} MB → {bin_path}")
        model, _ = _ensure_hf_model(self.script_dir, self.cfg)
        labels_path = os.path.join(bin_dir, "labels.json")
        with open(labels_path, "w") as f:
            json.dump({str(k): v for k, v in model.config.id2label.items()}, f)
        del model
        _original_print(f"  Labels saved: {labels_path}")

    def load_params(self) -> bool:
        bin_dir = os.path.join(self.script_dir, self.BIN_SUBDIR)
        bin_path = os.path.join(bin_dir, "params.bin")
        meta_path = os.path.join(bin_dir, "params.json")
        if not os.path.exists(bin_path) or not os.path.exists(meta_path):
            return False
        with open(meta_path) as f:
            meta = json.load(f)
        if "device" in meta and not _cache_layout_matches(meta):
            _original_print("  Params cache layout mismatch; regenerating params")
            return False
        total = meta["size"]
        self.NUM_LABELS = meta["num_labels"]
        self.NUM_LABELS_PAD = self.pad_dim(self.NUM_LABELS)
        CHUNK = 1 * 1024 * 1024
        with open(bin_path, "rb") as f:
            offset = 0
            while offset < total:
                data = f.read(min(CHUNK, total - offset))
                self.dma_write(DMA_DEVICE_H2C, self._params_dram_base + offset, data, len(data))
                offset += len(data)
        self.allocate_params_dram(total)
        _original_print(f"  Params loaded: {total / 1024**2:.1f} MB from bin")
        return True

    def dump_programs(self, program_addr: int):
        bin_dir = os.path.join(self.script_dir, self.BIN_SUBDIR)
        os.makedirs(bin_dir, exist_ok=True)
        bin_path = os.path.join(bin_dir, "programs.bin")
        meta_path = os.path.join(bin_dir, "programs.json")
        size = self.get_program_dram_usage()
        buf = bytearray(size)
        self.dma_read(DMA_DEVICE_C2H, program_addr, buf, size)
        with open(bin_path, "wb") as f:
            f.write(buf)
        with open(meta_path, "w") as f:
            json.dump(_cache_layout_meta(size), f)
        _original_print(f"  Program dumped: {size / 1024**2:.1f} MB → {bin_path}")

    def load_programs(self) -> int | None:
        bin_dir = os.path.join(self.script_dir, self.BIN_SUBDIR)
        bin_path = os.path.join(bin_dir, "programs.bin")
        meta_path = os.path.join(bin_dir, "programs.json")
        if not os.path.exists(bin_path) or not os.path.exists(meta_path):
            return None
        with open(meta_path) as f:
            meta = json.load(f)
        if "device" in meta and not _cache_layout_matches(meta):
            _original_print("  Program cache layout mismatch; recompiling")
            return None
        with open(bin_path, "rb") as f:
            data = f.read()
        addr = self.get_program_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, addr, data, len(data))
        self.allocate_program_dram(len(data))
        _original_print(f"  Program loaded: {len(data) / 1024**2:.1f} MB from bin")
        return addr

    # ------------------------------------------------------------------
    # Fully-fused: stem + 17 IR blocks + head + pool + classifier
    # ------------------------------------------------------------------

    def compile_full_fused(self) -> int:
        """Compile the entire forward pass as a single instruction stream."""
        self.start_capture()

        # ================================================================
        # STEM: 3x3 stride-2 conv, in=3 (padded to 64) -> out=32 (padded to 64), ReLU6
        # ================================================================
        conv2d_3x3_stride2_dram(
            self,
            INPUT_DRAM_ADDR=self.IMAGE_DRAM,
            OUTPUT_DRAM_ADDR=self.STEM_OUT_DRAM,
            IM2COL_DRAM_ADDR=self.IM2COL_DRAM,
            WEIGHT_DRAM_ADDR=self.STEM_W,
            BIAS_DRAM_ADDR=self.STEM_B,
            ZERO_PAD_DRAM_ADDR=self.ZERO_PAD,
            H_in=self.IMAGE_SIZE, W_in=self.IMAGE_SIZE,
            C_in=64, C_out=64,
            relu6_enable=True,
        )

        # ================================================================
        # 17 INVERTED RESIDUAL BLOCKS
        # ================================================================
        prev_out = self.STEM_OUT_DRAM
        for i, b in enumerate(self.blocks):
            bw = self.block_weights[i]
            bt = self.block_tensors[i]

            # -- (1) Expand 1x1 + ReLU6 (skipped when expand_ratio == 1) --
            if b["expand_ratio"] != 1:
                conv2d_1x1_dram(
                    self,
                    INPUT_DRAM_ADDR=prev_out, OUTPUT_DRAM_ADDR=bt["expanded"],
                    WEIGHT_DRAM_ADDR=bw["expand_w"], BIAS_DRAM_ADDR=bw["expand_b"],
                    H=b["H_in"], W=b["W_in"],
                    C_in=b["in_ch_p"], C_out=b["mid_ch_p"],
                    relu6_enable=True,
                )
                dw_input = bt["expanded"]
            else:
                dw_input = prev_out  # mid_ch == in_ch, no expand needed

            # -- (2) Depthwise 3x3 + ReLU6 (tap-wise eltwise; clamp via
            #         identity matmul because LALU CLAMP only fires on the
            #         matmul path on this HW). --
            conv2d_3x3_dw_tapwise_dram(
                self,
                INPUT_DRAM_ADDR=dw_input, OUTPUT_DRAM_ADDR=bt["dw_out"],
                ACC_DRAM_ADDR=self.DW_ACC_DRAM,
                IDENTITY_DRAM_ADDR=self.IDENTITY_64_DRAM,
                ZERO_PAD_DRAM_ADDR=self.ZERO_PAD,
                block_params=bw["dw_blocks"],
                H_in=b["H_in"], W_in=b["W_in"], C=b["mid_ch_p"],
                stride=b["stride"], W_chunk_max=self.DW_W_CHUNK_MAX,
                relu6_enable=True,
            )

            # -- (3) Project 1x1 (LINEAR — no activation) --
            conv2d_1x1_dram(
                self,
                INPUT_DRAM_ADDR=bt["dw_out"], OUTPUT_DRAM_ADDR=bt["block_out"],
                WEIGHT_DRAM_ADDR=bw["project_w"], BIAS_DRAM_ADDR=bw["project_b"],
                H=b["H_out"], W=b["W_out"],
                C_in=b["mid_ch_p"], C_out=b["out_ch_p"],
                relu6_enable=False,
            )

            # -- (4) Residual add (stride==1 and in_ch==out_ch) --
            if b["has_residual"]:
                eltwise_add_dram(
                    self,
                    A_ADDR=prev_out, B_ADDR=bt["block_out"],
                    OUT_ADDR=bt["block_out"],
                    num_elements=b["H_out"] * b["W_out"] * b["out_ch_p"],
                )

            prev_out = bt["block_out"]

        # ================================================================
        # HEAD: 1x1 conv 320 -> 1280 + ReLU6
        # ================================================================
        last_spatial = self.blocks[-1]["H_out"]
        conv2d_1x1_dram(
            self,
            INPUT_DRAM_ADDR=prev_out, OUTPUT_DRAM_ADDR=self.HEAD_OUT_DRAM,
            WEIGHT_DRAM_ADDR=self.HEAD_W, BIAS_DRAM_ADDR=self.HEAD_B,
            H=last_spatial, W=last_spatial,
            C_in=self.pad_dim(320), C_out=self.pad_dim(self.HEAD_OUT),
            relu6_enable=True,
        )

        # ================================================================
        # POOL: global avg over 7x7
        # ================================================================
        global_avgpool_dram(
            self,
            INPUT_DRAM_ADDR=self.HEAD_OUT_DRAM, OUTPUT_DRAM_ADDR=self.POOLED_DRAM,
            H=last_spatial, W=last_spatial, C=self.pad_dim(self.HEAD_OUT),
        )

        # ================================================================
        # CLASSIFIER: (1, 1280) @ (num_labels_pad, 1280)^T + bias
        # HW argmax fires automatically for M=1 matmul.
        # ================================================================
        self.matmat_mul_core(
            M=1, K=self.pad_dim(self.HEAD_OUT), N=self.NUM_LABELS_PAD,
            A_DRAM_ADDR=self.POOLED_DRAM,
            B_DRAM_ADDR=self.CLASSIFIER_W,
            OUTPUT_DRAM_ADDR=self.CLASSIFIER_OUT_DRAM,
            C_DRAM_ADDR=self.CLASSIFIER_B,
            bias_mode="broadcast_N",
        )

        self.stop_capture()
        self.generate_instruction_halt()

        prog_addr = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(prog_addr)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        self.clear_capture_buffer()
        return prog_addr

    def run_full_fused(self, pixel_values: torch.Tensor, program_addr: int,
                       timeout: float = 120.0) -> int:
        """Run the fully-fused forward pass. Returns predicted class index (HW argmax).

        pixel_values: (1, 3, H, W) bf16 — converted to HWC and padded to 64 channels.
        Pure HW inference latency (start_execute -> wait_queue) is stashed on
        self.last_inference_seconds for the caller to print.
        """
        import time as _t
        image_chw = pixel_values.squeeze(0).to(torch.bfloat16)              # (3, H, W)
        image_hwc = image_chw.permute(1, 2, 0).contiguous()                  # (H, W, 3)
        H = image_hwc.shape[0]
        image_hwc_p = torch.zeros(H, H, 64, dtype=torch.bfloat16)
        image_hwc_p[:, :, :3] = image_hwc
        self.dma_to_accelerator_memory(self.IMAGE_DRAM, image_hwc_p.contiguous().flatten())
        t_inf_start = _t.perf_counter()
        self.start_execute_from_dram(program_addr)
        self.wait_queue(timeout)
        self.last_inference_seconds = _t.perf_counter() - t_inf_start
        return self.get_arg_max_index()


# ---------------------------------------------------------------------------
# CPU vs HW per-stage diff harness
# ---------------------------------------------------------------------------

def _diff_against_cpu(ue: "MobileNetV2_UnifiedEngine",
                     pixel_values: torch.Tensor,
                     script_dir: str = SCRIPT_DIR) -> None:
    """Compare HW intermediate buffers to a CPU reference, one stage at a time.

    Runs the pythonic CPU MBV2 (mobilenetv2_cpu_test.py) with forward hooks on
    stem / each IR block / head, then reads the matching HW DRAM buffers via
    dma_from_accelerator_memory. For each stage prints mean|diff| and max|diff|
    so the first row of significant divergence localizes the bug.

    Call AFTER run_full_fused so the HW buffers (STEM_OUT_DRAM, block_out per
    block, HEAD_OUT_DRAM, POOLED_DRAM) still hold the latest activations.
    """
    import importlib.util
    cpu_path = os.path.join(script_dir, "mobilenetv2_cpu_test.py")
    spec = importlib.util.spec_from_file_location("mobilenetv2_cpu_test", cpu_path)
    cpu_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpu_mod)

    cfg = cpu_mod._load_config(script_dir)
    sd, _ = cpu_mod._ensure_hf_model(script_dir, cfg)
    model = cpu_mod.MobileNetV2Backbone(cfg).eval()
    cpu_mod._load_hf_weights(sd, model)

    captured: dict[str, torch.Tensor] = {}

    def make_hook(key: str):
        def h(_m, _inp, out):
            captured[key] = out.detach().clone()
        return h

    model.stem.register_forward_hook(make_hook("stem"))
    for i, blk in enumerate(model.blocks):
        blk.register_forward_hook(make_hook(f"block_{i}"))
    model.head.register_forward_hook(make_hook("head"))

    with torch.no_grad():
        feat = model(pixel_values)
    # Pool stage = spatial mean over (H,W); matches global_avgpool_dram.
    captured["pool"] = feat.mean(dim=[2, 3])  # (1, 1280)

    # --- Read HW intermediates back ---
    H_stem = ue.IMAGE_SIZE // 2
    hw_stem = ue.dma_from_accelerator_memory(ue.STEM_OUT_DRAM, (H_stem, H_stem, 64))
    hw_blocks = []
    for i, b in enumerate(ue.blocks):
        bt = ue.block_tensors[i]
        hw_blocks.append(ue.dma_from_accelerator_memory(
            bt["block_out"], (b["H_out"], b["W_out"], b["out_ch_p"])))
    last_s = ue.blocks[-1]["H_out"]
    hw_head = ue.dma_from_accelerator_memory(
        ue.HEAD_OUT_DRAM, (last_s, last_s, ue.pad_dim(ue.HEAD_OUT)))
    hw_pool = ue.dma_from_accelerator_memory(
        ue.POOLED_DRAM, (ue.pad_dim(ue.HEAD_OUT),))

    def _report(name: str, cpu_t: torch.Tensor, hw_t: torch.Tensor, c_unpad: int):
        if cpu_t.dim() == 4:
            cpu_hwc = cpu_t.squeeze(0).permute(1, 2, 0).contiguous().float()  # (H, W, C)
            hw_unpad = hw_t[..., :c_unpad].float()
        else:  # pool: (1, C) and (C,)
            cpu_hwc = cpu_t.squeeze(0).float()
            hw_unpad = hw_t[:c_unpad].float()
        diff = (cpu_hwc - hw_unpad).abs()
        rel = diff.mean().item() / (cpu_hwc.abs().mean().item() + 1e-9)
        _original_print(
            f"  {name:10s} {str(tuple(cpu_hwc.shape)):20s} "
            f"mean|d|={diff.mean().item():.3e}  max|d|={diff.max().item():.3e}  "
            f"rel={rel:.2%}"
        )

    _original_print("\n=== CPU vs HW per-stage diff ===")
    _report("stem", captured["stem"], hw_stem, ue.STEM_OUT)
    for i, b in enumerate(ue.blocks):
        _report(f"block_{i}", captured[f"block_{i}"], hw_blocks[i], b["out_ch"])
    _report("head", captured["head"], hw_head, ue.HEAD_OUT)
    _report("pool", captured["pool"], hw_pool, ue.HEAD_OUT)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(image_path: str, size: int = 224) -> torch.Tensor:
    """Load and preprocess image for MobileNetV2 (TF/HF preprocessing):
       resize to size x size, scale to [-1, 1]. Returns (1, 3, size, size)."""
    image = Image.open(image_path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    img_t = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
    return (img_t - 0.5) * 2.0  # [0,1] -> [-1,1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _clock_ns_default_for_device(device: str) -> float:
    """Return default clock period (ns) for FPGA type — mirrors user_hw_test.py."""
    if device == "efinix":                         return 4.0
    if device == "kintex7":                       return 5.1594
    if device in ("rk", "puzhi"):                 return 3.0
    if device in ("bittware", "bittware_256"):     return 3.3333
    if device == "alveo":                          return 4.0
    return 10.0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MobileNetV2-1.0 accelerator inference.")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--dev", type=str, default=None, help="DMA device override (default: board profile)")
    parser.add_argument("--cycle", type=float, default=None, help='Clock cycle time in ns. Overrides --device default.')
    parser.add_argument("--device", type=str, default="kintex7", help='FPGA board profile (kintex7, rk, puzhi, bittware, bittware_256, alveo, efinix).')
    parser.add_argument("--cleanup", action="store_true",
                        help="Delete compile artifacts from mobilenetv2_bin/ before running. "
                             "Cached HF weights (hf_model/) are preserved.")
    parser.add_argument("--diff", action="store_true",
                        help="After HW run, read intermediate buffers and compare to CPU reference.")
    args = parser.parse_args()

    if args.cleanup:
        bin_dir = os.path.join(SCRIPT_DIR, "mobilenetv2_bin")
        artifacts = ["params.bin", "params.json", "programs.bin", "programs.json", "labels.json"]
        removed = 0
        for name in artifacts:
            p = os.path.join(bin_dir, name)
            if os.path.exists(p):
                os.remove(p); removed += 1
        print(f"[cleanup] Removed {removed} compile artifact(s) from {bin_dir}")

    global _SILENT_MODE
    _SILENT_MODE = True

    profile = configure_device(args.device, dma_device=args.dev)
    _set_dram_layout_for_device(args.device)
    global DMA_DEVICE_H2C, DMA_DEVICE_C2H
    DMA_DEVICE_H2C = user_dma_core.DMA_DEVICE_H2C
    DMA_DEVICE_C2H = user_dma_core.DMA_DEVICE_C2H
    axi_width_bits = profile.get("axi_data_width_bits") or (512 if args.device in ("bittware", "rk") else 256)
    os.environ["UE_AXI_DATA_WIDTH_BITS"] = str(axi_width_bits)
    user_dma_core.UE_AXI_DATA_WIDTH_BITS = axi_width_bits
    clock = args.cycle if args.cycle is not None else _clock_ns_default_for_device(args.device)
    user_dma_core.CLOCK_CYCLE_TIME_NS = clock
    user_dma_core.UE_PEAK_GFLOPS = 0.128 / clock
    _original_print(f"FPGA profile: device={args.device}, clock={clock:.4f} ns, UE_AXI_DATA_WIDTH_BITS={axi_width_bits}")
    _original_print(f"Using DMA: H2C={DMA_DEVICE_H2C}, C2H={DMA_DEVICE_C2H}, USER={user_dma_core.DMA_DEVICE_USER}")
    _original_print(f"DRAM layout: params=0x{MBV2_PARAMS_BASE:08X}, tensor=0x{MBV2_TENSOR_BASE:08X}, "
                    f"program=0x{MBV2_PROGRAM_BASE:08X}, end=0x{user_dma_core.DRAM_END_ADDR:08X}")

    image_path = args.image or os.path.join(SCRIPT_DIR, "../../test_samples/vette.jpg")
    pixel_values = preprocess_image(image_path, size=MobileNetV2_UnifiedEngine.IMAGE_SIZE)

    _original_print(f"MobileNetV2-1.0-{MobileNetV2_UnifiedEngine.IMAGE_SIZE} on {user_dma_core.DMA_DEVICE_H2C}")

    import threading
    import time as _time

    def _progress_timer(label, start_time, stop_event):
        while not stop_event.wait(1.0):
            elapsed = _time.perf_counter() - start_time
            _original_print(f"\r  {label} ({elapsed:.0f}s)", end="", flush=True)

    t0 = _time.perf_counter()
    ue = MobileNetV2_UnifiedEngine(script_dir=SCRIPT_DIR)
    ue.software_reset()
    t_weights = _time.perf_counter()
    _original_print(f"  Weights: {t_weights - t0:.3f}s")

    prog_addr = ue.load_programs() if ue.USE_BIN_CACHE else None
    if prog_addr is None:
        if ue.params_from_bin:
            raise RuntimeError("Cannot compile without weight addresses — delete mobilenetv2_bin/ and re-run")
        stop_c = threading.Event()
        timer_c = threading.Thread(target=_progress_timer,
            args=("Compiling", t_weights, stop_c), daemon=True)
        timer_c.start()
        prog_addr = ue.compile_full_fused()
        stop_c.set(); timer_c.join()
        if ue.USE_BIN_CACHE:
            ue.dump_params()
            ue.dump_programs(prog_addr)
    t_compile = _time.perf_counter()
    _original_print(f"\r  Compile: {t_compile - t_weights:.3f}s")

    t_exec_start = _time.perf_counter()
    stop = threading.Event()
    timer = threading.Thread(target=_progress_timer, args=("Executing", t_exec_start, stop), daemon=True)
    timer.start()
    predicted_idx = ue.run_full_fused(pixel_values, prog_addr)
    stop.set(); timer.join()
    t_exec = _time.perf_counter()
    _original_print(f"\r  Executing: {t_exec - t_exec_start:.3f}s")

    labels_path = os.path.join(SCRIPT_DIR, "mobilenetv2_bin", "labels.json")
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            id2label = json.load(f)
    else:
        # No cached labels.json (bin caching disabled) — pull straight from HF.
        hf_model, _ = _ensure_hf_model(SCRIPT_DIR, ue.cfg)
        id2label = {str(k): v for k, v in hf_model.config.id2label.items()}
        del hf_model
    label = id2label.get(str(predicted_idx), str(predicted_idx))
    _original_print(f"\n  Image: {image_path}")
    _original_print(f"  Prediction: {label!r} (class {predicted_idx})")
    _original_print(f"  Inference (pure HW): {ue.last_inference_seconds * 1000:.3f} ms")

    import json as _json
    _original_print("TEST_RESULT:" + _json.dumps({
        "decoded_text": label,
    }))

    if args.diff:
        # pixel_values is (1, 3, H, W) float; CPU module expects (1, 3, H, W) too.
        if pixel_values.dim() == 3:
            cpu_input = pixel_values.unsqueeze(0).float()
        else:
            cpu_input = pixel_values.float()
        _diff_against_cpu(ue, cpu_input, script_dir=SCRIPT_DIR)


if __name__ == "__main__":
    main()
