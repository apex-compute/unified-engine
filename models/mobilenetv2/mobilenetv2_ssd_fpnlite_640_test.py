#!/usr/bin/env python3
"""
MobileNetV2-SSD-FPNLite (640x640) — accelerator inference.

  Backbone : MBV2-1.0, same as mobilenetv2_test.py but at 640 input.
             Taps C3 (stride 8, 32ch -> pad 64), C4 (stride 16, 96 -> pad 128),
             C5 (stride 32, 1280ch — head 1x1 output).
  Neck     : FPN-Lite, 5 levels P3..P7, 128 ch. Laterals = 1x1 + bias (no BN/act).
             P5 = lat_c5; P4 = SepConv(lat_c4 + nearest_2x(P5)); P3 = SepConv(lat_c3 + nearest_2x(P4));
             P6 = SepConv(P5, stride=2); P7 = SepConv(P6, stride=2).
  Head     : Shared 4-DSConv tower (per-level BN), final cls (A*K = 546 -> pad 576)
             and box (A*4 = 24 -> pad 64) DSConvs with bias.
  Postproc : anchors + decode + sigmoid + per-class NMS on CPU after DMA-back.

The only new HW primitive vs mobilenetv2_test.py is nearest_upsample_2x_dram
(validated in user_hw_test.py); everything else is built from the existing
conv2d_1x1_dram / conv2d_3x3_stride2_dram / conv2d_3x3_dw_tapwise_dram /
eltwise_add_dram helpers, with a zero-bias tile to use the depthwise primitive
as a bare 3x3 depthwise (no BN, no clamp).

Bin caching is OFF — weights and program are rebuilt every run while we iterate.

Usage:
    python mobilenetv2_ssd_fpnlite_640_hw_test.py
    python mobilenetv2_ssd_fpnlite_640_hw_test.py --image path/to/image.jpg
    python mobilenetv2_ssd_fpnlite_640_hw_test.py --dev xdma0 [--cycle 5.16]
"""

import builtins
import json
import math
import os
import struct
import sys
import tarfile
import time
import urllib.request
import warnings
warnings.filterwarnings("ignore", message=".*torchao.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

_original_print = builtins.print
_SILENT_MODE = False

def quiet_print(*args, **kwargs):
    if _SILENT_MODE:
        return
    _original_print(*args, **kwargs)

builtins.print = quiet_print

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, DMA_DEVICE_C2H, DRAM_INSTRUCTION_ADDR,
    URAM_NEAR_FULL_ELEMENTS,
    configure_device, UnifiedEngine,
)

# ---------------------------------------------------------------------------
# Inlined: _load_config, ConvBNReLU6, InvertedResidual
# ---------------------------------------------------------------------------
def _load_config(script_dir: str = SCRIPT_DIR) -> dict:
    with open(os.path.join(script_dir, "mobilenetv2_ssd_fpnlite_640_config.json")) as f:
        return json.load(f)


class ConvBNReLU6(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU6. The MBV2 building block.

    HF MobileNetV2 uses TF-style SAME padding: for stride-2 kernel-k convs,
    pad is (k-1) on the bottom/right side only. Mirrored here so weights from
    HF parity-check cleanly.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, groups: int = 1, tf_padding: bool = True,
                 bn_eps: float = 1e-3):
        super().__init__()
        self._k = kernel_size
        self._use_tf_pad = tf_padding and stride > 1 and kernel_size > 1
        sym_pad = 0 if self._use_tf_pad else (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, sym_pad,
                              groups=groups, bias=False)
        # TF/HF MobileNetV2 use BN eps=1e-3 (PyTorch default is 1e-5).
        self.bn = nn.BatchNorm2d(out_ch, eps=bn_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_tf_pad:
            x = F.pad(x, (0, self._k - 1, 0, self._k - 1))
        return F.relu6(self.bn(self.conv(x)))


class InvertedResidual(nn.Module):
    """Expand 1x1 -> depthwise 3x3 -> project 1x1 (linear). Residual when stride==1 and in==out."""

    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int):
        super().__init__()
        assert stride in (1, 2)
        hidden_ch = in_ch * expand_ratio
        self.use_residual = (stride == 1 and in_ch == out_ch)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU6(in_ch, hidden_ch, kernel_size=1))
        layers += [
            ConvBNReLU6(hidden_ch, hidden_ch, kernel_size=3, stride=stride, groups=hidden_ch),
            nn.Conv2d(hidden_ch, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch, eps=1e-3),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        return x + y if self.use_residual else y

# ---------------------------------------------------------------------------
# Inlined: ops, weight helpers
# ---------------------------------------------------------------------------
# ReLU6 is applied by passing clamp_max=6.0 to matmat_mul_core(clamp_enable=True);
# the synced library takes the clamp bounds as kwargs (clamp_min=0.0/clamp_max).

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
# Inlined: SSD model, TF loader, anchors, NMS, COCO_LABELS
# ---------------------------------------------------------------------------
_ssd_cfg = _load_config(SCRIPT_DIR)
TF_CKPT_NAME = _ssd_cfg["paths"]["tf_ckpt_name"]
TF_CKPT_URL  = _ssd_cfg["paths"]["tf_ckpt_url"]
BIN_DIR      = os.path.join(SCRIPT_DIR, _ssd_cfg["paths"]["bin_dir"])


def _ensure_tf_checkpoint() -> str:
    """Download + extract TF SSD-MBV2-FPNLite-640 checkpoint if not present.

    Returns the prefix path passed to tf.train.load_checkpoint (without
    .index / .data-... suffix).
    """
    ckpt_dir = os.path.join(BIN_DIR, TF_CKPT_NAME, "checkpoint")
    ckpt_prefix = os.path.join(ckpt_dir, "ckpt-0")
    if os.path.exists(ckpt_prefix + ".index"):
        return ckpt_prefix

    os.makedirs(BIN_DIR, exist_ok=True)
    tar_path = os.path.join(BIN_DIR, TF_CKPT_NAME + ".tar.gz")
    if not os.path.exists(tar_path):
        print(f"Downloading {TF_CKPT_URL} -> {tar_path}")
        urllib.request.urlretrieve(TF_CKPT_URL, tar_path)
        print("Download complete.")

    print(f"Extracting {tar_path} ...")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(BIN_DIR)
    assert os.path.exists(ckpt_prefix + ".index"), f"missing {ckpt_prefix}.index after extract"
    return ckpt_prefix


# ---------------------------------------------------------------------------
# Multi-scale MobileNetV2 backbone (returns C3, C4, C5)
# ---------------------------------------------------------------------------

class MobileNetV2BackboneSSD(nn.Module):
    """MBV2-1.0 with head conv. Returns C3 (32ch s8), C4 (96ch s16), C5 (1280ch s32).

    TF SSD-FPNLite uses the post-Conv_1 (1280ch) tensor as C5, not the raw block-16
    (320ch) output. We include the head conv and tap its output for C5.

    Block index -> post-block stride / channels:
        blocks[ 0]  16ch  stride 2
        blocks[1-2] 24ch  stride 4
        blocks[3-5] 32ch  stride 8     <- C3 (after block 5)
        blocks[6-9] 64ch  stride 16
        blocks[10-12] 96ch stride 16   <- C4 (after block 12)
        blocks[13-15] 160ch stride 32
        blocks[16]  320ch stride 32
        head:       1280ch stride 32   <- C5
    """

    C3_BLOCK_IDX = 5
    C4_BLOCK_IDX = 12
    C3_CH, C4_CH, C5_CH = 32, 96, 1280

    def __init__(self, cfg: dict):
        super().__init__()
        b = cfg["backbone"]
        stem_ch = b["stem_out_channels"]
        head_ch = b["head_out_channels"]
        irs = b["inverted_residual_setting"]

        self.stem = ConvBNReLU6(3, stem_ch, kernel_size=3, stride=2)

        blocks = []
        in_ch = stem_ch
        for t, c, n, s in irs:
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(InvertedResidual(in_ch, c, stride, t))
                in_ch = c
        self.blocks = nn.ModuleList(blocks)
        self.head = ConvBNReLU6(in_ch, head_ch, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        c3 = c4 = None
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == self.C3_BLOCK_IDX:
                c3 = x
            elif i == self.C4_BLOCK_IDX:
                c4 = x
        c5 = self.head(x)
        return c3, c4, c5


# ---------------------------------------------------------------------------
# TF-style separable conv (matches tf.keras.layers.SeparableConv2D)
# ---------------------------------------------------------------------------

class SeparableConvBNReLU6(nn.Module):
    """TF SeparableConv2D + BN + ReLU6.

    Layout: depthwise 3x3 -> pointwise 1x1 -> BN -> ReLU6. Single BN at the end,
    no BN/activation between dw and pw (this matches TF Keras SeparableConv2D,
    NOT the MBV2 inverted-residual depthwise-separable pattern).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self._k = kernel_size
        self._use_tf_pad = stride > 1 and kernel_size > 1
        sym = 0 if self._use_tf_pad else (kernel_size - 1) // 2
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, stride, sym, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_tf_pad:
            x = F.pad(x, (0, self._k - 1, 0, self._k - 1))
        return F.relu6(self.bn(self.pw(self.dw(x))))


# ---------------------------------------------------------------------------
# FPN-Lite neck
# ---------------------------------------------------------------------------

class FPNLiteNeck(nn.Module):
    """TF SSD-FPNLite top-down neck.

    Architecture (matches checkpoint):
      lateral_c5 = Conv1x1(C5,  out_ch) + bias                     # no BN/act
      lateral_c4 = Conv1x1(C4,  out_ch) + bias
      lateral_c3 = Conv1x1(C3,  out_ch) + bias
      P5 = lateral_c5
      P4 = SepConv3x3+BN+ReLU6 ( lateral_c4 + nearest_upsample(P5) )
      P3 = SepConv3x3+BN+ReLU6 ( lateral_c3 + nearest_upsample(P4) )
      P6 = SepConv3x3+BN+ReLU6 ( P5, stride=2 )
      P7 = SepConv3x3+BN+ReLU6 ( P6, stride=2 )

    Note: P5 has no smoothing conv in TF FPNLite.
    """

    def __init__(self, c3_ch: int, c4_ch: int, c5_ch: int, out_ch: int = 128):
        super().__init__()
        self.out_ch = out_ch
        # Laterals: plain 1x1 Conv2D with bias, no BN, no activation.
        self.lat_c5 = nn.Conv2d(c5_ch, out_ch, 1, 1, 0, bias=True)
        self.lat_c4 = nn.Conv2d(c4_ch, out_ch, 1, 1, 0, bias=True)
        self.lat_c3 = nn.Conv2d(c3_ch, out_ch, 1, 1, 0, bias=True)
        # Smoothing (only P4 and P3; not P5).
        self.smooth_p4 = SeparableConvBNReLU6(out_ch, out_ch, kernel_size=3, stride=1)
        self.smooth_p3 = SeparableConvBNReLU6(out_ch, out_ch, kernel_size=3, stride=1)
        # Coarse extra levels.
        self.coarse_p6 = SeparableConvBNReLU6(out_ch, out_ch, kernel_size=3, stride=2)
        self.coarse_p7 = SeparableConvBNReLU6(out_ch, out_ch, kernel_size=3, stride=2)

    def forward(self, c3, c4, c5):
        p5 = self.lat_c5(c5)
        p4 = self.smooth_p4(self.lat_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest"))
        p3 = self.smooth_p3(self.lat_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest"))
        p6 = self.coarse_p6(p5)
        p7 = self.coarse_p7(p6)
        return [p3, p4, p5, p6, p7]


# ---------------------------------------------------------------------------
# SSDLite WeightSharedConvolutionalBoxPredictor head
# ---------------------------------------------------------------------------

class SSDLiteHead(nn.Module):
    """WeightSharedConvolutionalBoxPredictor (use_depthwise=True, share_prediction_tower=True).

    Single shared 4-DSConv tower across all FPN levels. Tower convs (depthwise+pointwise)
    are shared; the BN statistics inside the tower are PER LEVEL. After the tower,
    a shared SeparableConv produces cls logits (A*num_classes) and another produces
    box deltas (A*4); both have bias on the pointwise (final) conv and no BN/activation.
    """

    def __init__(self, in_ch: int = 128, num_anchors: int = 6, num_classes: int = 91,
                 n_levels: int = 5, n_tower: int = 4):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.n_levels = n_levels
        self.n_tower = n_tower

        # Shared tower DSConv kernels (one set, applied at every level).
        self.tower_dw = nn.ModuleList(
            [nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False) for _ in range(n_tower)]
        )
        self.tower_pw = nn.ModuleList(
            [nn.Conv2d(in_ch, in_ch, 1, 1, 0, bias=False) for _ in range(n_tower)]
        )
        # Per-level BN: tower_bn[level][stage]
        self.tower_bn = nn.ModuleList([
            nn.ModuleList([nn.BatchNorm2d(in_ch, eps=1e-3) for _ in range(n_tower)])
            for _ in range(n_levels)
        ])

        # Shared final predictors (DSConv with bias on pw, no BN, no activation).
        self.cls_dw = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False)
        self.cls_pw = nn.Conv2d(in_ch, num_anchors * num_classes, 1, 1, 0, bias=True)
        self.box_dw = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False)
        self.box_pw = nn.Conv2d(in_ch, num_anchors * 4, 1, 1, 0, bias=True)

    def _tower(self, x: torch.Tensor, level: int) -> torch.Tensor:
        for s in range(self.n_tower):
            x = self.tower_pw[s](self.tower_dw[s](x))
            x = self.tower_bn[level][s](x)
            x = F.relu6(x)
        return x

    def forward(self, feats):
        cls_outs, box_outs = [], []
        for level, f in enumerate(feats):
            t = self._tower(f, level)
            cls = self.cls_pw(self.cls_dw(t))   # (B, A*K, H, W)
            box = self.box_pw(self.box_dw(t))   # (B, A*4, H, W)
            B, _, H, W = cls.shape
            cls = cls.permute(0, 2, 3, 1).reshape(B, H * W * self.num_anchors, self.num_classes)
            box = box.permute(0, 2, 3, 1).reshape(B, H * W * self.num_anchors, 4)
            cls_outs.append(cls)
            box_outs.append(box)
        return torch.cat(cls_outs, dim=1), torch.cat(box_outs, dim=1)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MobileNetV2_SSD_FPNLite(nn.Module):
    def __init__(self, cfg: dict, fpn_ch: int = 128, num_anchors: int = 6,
                 num_classes: int = 91):
        super().__init__()
        self.backbone = MobileNetV2BackboneSSD(cfg)
        self.neck = FPNLiteNeck(
            c3_ch=self.backbone.C3_CH,
            c4_ch=self.backbone.C4_CH,
            c5_ch=self.backbone.C5_CH,
            out_ch=fpn_ch,
        )
        self.head = SSDLiteHead(in_ch=fpn_ch, num_anchors=num_anchors, num_classes=num_classes)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        feats = self.neck(c3, c4, c5)
        cls, box = self.head(feats)
        return cls, box, feats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pure-Python TF v2 checkpoint reader (no tensorflow dependency).
#
# The TF checkpoint is two files:
#   ckpt-0.index               - LevelDB-style SSTable (uncompressed in TF):
#                                maps variable-name string -> BundleEntryProto bytes.
#   ckpt-0.data-00000-of-00001 - concatenated raw tensor bytes.
#
# We parse the SSTable (footer -> index block -> data blocks -> prefix-compressed
# entries), parse the BundleEntryProto (only the dtype/shape/offset/size fields),
# and slice each tensor out of the data file via numpy.frombuffer.
# ---------------------------------------------------------------------------

_TF_MAGIC = 0xdb4775248b80fb57   # LevelDB SSTable kTableMagicNumber
_TF_DTYPE = {
    1:  np.dtype("<f4"),   # DT_FLOAT
    2:  np.dtype("<f8"),   # DT_DOUBLE
    3:  np.dtype("<i4"),   # DT_INT32
    4:  np.dtype("u1"),    # DT_UINT8
    5:  np.dtype("<i2"),   # DT_INT16
    6:  np.dtype("i1"),    # DT_INT8
    9:  np.dtype("<i8"),   # DT_INT64
    19: np.dtype("<f2"),   # DT_HALF
}


def _read_varint(buf, pos):
    shift = 0; result = 0
    while True:
        b = buf[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
        if shift > 70:
            raise ValueError("varint too long")


def _read_block_handle(buf, pos):
    off, pos = _read_varint(buf, pos)
    sz, pos = _read_varint(buf, pos)
    return off, sz, pos


def _snappy_decompress(data: bytes) -> bytes:
    """Pure-Python Snappy frame-less decoder. See:
       https://github.com/google/snappy/blob/main/format_description.txt
    """
    out_len, pos = _read_varint(data, 0)
    out = bytearray()
    n = len(data)
    while pos < n:
        tag = data[pos]; pos += 1
        kind = tag & 0x03
        if kind == 0:                           # literal
            ll = tag >> 2
            if ll < 60:
                length = ll + 1
            else:
                nbytes = ll - 59
                length = int.from_bytes(data[pos:pos + nbytes], "little") + 1
                pos += nbytes
            out.extend(data[pos:pos + length]); pos += length
        else:                                   # copy
            if kind == 1:                       # 1-byte offset
                length = ((tag >> 2) & 0x07) + 4
                offset = ((tag >> 5) & 0x07) << 8 | data[pos]; pos += 1
            elif kind == 2:                     # 2-byte offset
                length = (tag >> 2) + 1
                offset = int.from_bytes(data[pos:pos + 2], "little"); pos += 2
            else:                               # kind == 3, 4-byte offset
                length = (tag >> 2) + 1
                offset = int.from_bytes(data[pos:pos + 4], "little"); pos += 4
            start = len(out) - offset
            # Spec requires byte-at-a-time copy (overlap is legal & common).
            for i in range(length):
                out.append(out[start + i])
    if len(out) != out_len:
        raise ValueError(f"snappy length mismatch: got {len(out)}, expected {out_len}")
    return bytes(out)


def _read_block(file_bytes: bytes, off: int, sz: int) -> bytes:
    """Read a LevelDB SSTable block, transparently snappy-decompressing if needed.

    The 5-byte trailer immediately follows the block contents: 1 byte compression
    type (0 = none, 1 = snappy) + 4 byte CRC32C (skipped).
    """
    raw = file_bytes[off:off + sz]
    comp = file_bytes[off + sz]
    if comp == 0:
        return raw
    if comp == 1:
        return _snappy_decompress(raw)
    raise ValueError(f"unsupported block compression type {comp}")


def _iter_block_entries(block):
    """Yield (key, value) from a LevelDB SSTable block (prefix-compressed)."""
    n = len(block)
    restart_count = struct.unpack_from("<I", block, n - 4)[0]
    entries_end = n - 4 - 4 * restart_count
    pos = 0
    prev_key = b""
    while pos < entries_end:
        shared, pos = _read_varint(block, pos)
        unshared, pos = _read_varint(block, pos)
        vlen, pos = _read_varint(block, pos)
        key = prev_key[:shared] + block[pos:pos + unshared]
        pos += unshared
        val = block[pos:pos + vlen]
        pos += vlen
        yield key, val
        prev_key = key


def _parse_shape(buf):
    """Parse TensorShapeProto -> tuple of dim sizes."""
    dims = []
    pos = 0; n = len(buf)
    while pos < n:
        tag, pos = _read_varint(buf, pos)
        field = tag >> 3; wt = tag & 7
        if wt == 2:
            ln, pos = _read_varint(buf, pos)
            if field == 2:  # repeated TensorShapeProto.Dim
                end = pos + ln
                size = 0
                while pos < end:
                    t2, pos = _read_varint(buf, pos)
                    f2 = t2 >> 3; w2 = t2 & 7
                    if w2 == 0:
                        v, pos = _read_varint(buf, pos)
                        if f2 == 1:
                            size = v
                    elif w2 == 2:
                        ln2, pos = _read_varint(buf, pos)
                        pos += ln2  # skip Dim.name
                    else:
                        raise ValueError(f"shape dim wire type {w2}")
                dims.append(size)
            else:
                pos += ln
        elif wt == 0:
            _, pos = _read_varint(buf, pos)
        elif wt == 1:
            pos += 8
        elif wt == 5:
            pos += 4
        else:
            raise ValueError(f"shape wire type {wt}")
    return tuple(dims)


def _parse_bundle_entry(buf):
    """Parse BundleEntryProto -> (dtype, shape, offset, size). Other fields ignored."""
    dtype = 0; offset = 0; size = 0; shape = ()
    pos = 0; n = len(buf)
    while pos < n:
        tag, pos = _read_varint(buf, pos)
        field = tag >> 3; wt = tag & 7
        if wt == 0:
            val, pos = _read_varint(buf, pos)
            if field == 1: dtype = val
            elif field == 4: offset = val
            elif field == 5: size = val
        elif wt == 2:
            ln, pos = _read_varint(buf, pos)
            if field == 2:
                shape = _parse_shape(bytes(buf[pos:pos + ln]))
            pos += ln
        elif wt == 1:
            pos += 8
        elif wt == 5:
            pos += 4
        else:
            raise ValueError(f"bundle entry wire type {wt}")
    return dtype, shape, offset, size


def _read_tf_checkpoint(prefix: str):
    """Pure-Python TF v2 checkpoint reader.

    Returns {variable_name: numpy_array_with_TF_shape}. No tensorflow import.
    """
    with open(prefix + ".index", "rb") as f:
        idx = f.read()
    with open(prefix + ".data-00000-of-00001", "rb") as f:
        data = f.read()

    if len(idx) < 48:
        raise ValueError("index file too small to contain SSTable footer")
    magic = struct.unpack_from("<Q", idx, len(idx) - 8)[0]
    if magic != _TF_MAGIC:
        raise ValueError(f"bad SSTable magic 0x{magic:x}")
    footer = idx[len(idx) - 48:]
    _meta_off, _meta_sz, p = _read_block_handle(footer, 0)
    idx_off, idx_sz, _ = _read_block_handle(footer, p)

    # The index block lists (last_key_in_block -> BlockHandle) for each data block.
    index_block = _read_block(idx, idx_off, idx_sz)
    data_block_handles = []
    for _key, handle_bytes in _iter_block_entries(index_block):
        off, sz, _ = _read_block_handle(handle_bytes, 0)
        data_block_handles.append((off, sz))

    raw_entries = {}
    for off, sz in data_block_handles:
        block = _read_block(idx, off, sz)
        for key, val in _iter_block_entries(block):
            raw_entries[bytes(key).decode("utf-8")] = bytes(val)

    tensors = {}
    for name, entry in raw_entries.items():
        if not entry:
            continue
        dtype, shape, offset, size = _parse_bundle_entry(entry)
        if dtype not in _TF_DTYPE:
            continue  # skip non-numeric (e.g. DT_STRING object-graph)
        np_dtype = _TF_DTYPE[dtype]
        count = size // np_dtype.itemsize
        arr = np.frombuffer(data, dtype=np_dtype, count=count, offset=offset).copy()
        if shape:
            arr = arr.reshape(shape)
        tensors[name] = arr
    return tensors


# ---------------------------------------------------------------------------
# TF checkpoint -> our PyTorch modules
# ---------------------------------------------------------------------------

def _tf_kernel_to_torch(t):
    """TF Conv2D kernel (H,W,Cin,Cout) -> PyTorch (Cout,Cin,H,W)."""
    return torch.from_numpy(t).permute(3, 2, 0, 1).contiguous()


def _tf_dwkernel_to_torch(t):
    """TF DepthwiseConv2D / SeparableConv2D depthwise kernel
       (H,W,Cin,multiplier=1) -> PyTorch (Cin,1,H,W)."""
    return torch.from_numpy(t).permute(2, 3, 0, 1).contiguous()


def _load_tf_weights(model: "MobileNetV2_SSD_FPNLite", ckpt_prefix: str, verbose: bool = False):
    """Copy TF SSD-MBV2-FPNLite-640 weights into our PyTorch model in-place.

    Uses the pure-Python checkpoint reader above; no tensorflow import required.
    """
    ckpt = _read_tf_checkpoint(ckpt_prefix)
    def G(name):
        return ckpt[name + "/.ATTRIBUTES/VARIABLE_VALUE"]

    # ---------- Backbone ----------
    # MBV2 1.0 layer_with_weights numbering (each conv = 1 idx, each BN = 1 idx):
    #   0: stem conv          1: stem BN
    #   2: blk0.dw_conv       3: blk0.dw_BN
    #   4: blk0.pw_conv       5: blk0.pw_BN
    #   For i in 1..16, base = 6 + (i-1)*6:
    #     base:    blk[i] expand_pw_conv      base+1: BN
    #     base+2:  blk[i] depthwise_conv      base+3: BN
    #     base+4:  blk[i] project_pw_conv     base+5: BN
    #   102: head conv        103: head BN
    fe = "model/_feature_extractor/classification_backbone/layer_with_weights-"

    def cp_conv2d(idx: int, m: nn.Conv2d):
        m.weight.data.copy_(_tf_kernel_to_torch(G(f"{fe}{idx}/kernel")))

    def cp_dwconv2d(idx: int, m: nn.Conv2d):
        m.weight.data.copy_(_tf_dwkernel_to_torch(G(f"{fe}{idx}/depthwise_kernel")))

    def cp_bn(idx: int, m: nn.BatchNorm2d):
        m.weight.data.copy_(torch.from_numpy(G(f"{fe}{idx}/gamma")))
        m.bias.data.copy_(torch.from_numpy(G(f"{fe}{idx}/beta")))
        m.running_mean.data.copy_(torch.from_numpy(G(f"{fe}{idx}/moving_mean")))
        m.running_var.data.copy_(torch.from_numpy(G(f"{fe}{idx}/moving_variance")))

    bb = model.backbone
    cp_conv2d(0, bb.stem.conv); cp_bn(1, bb.stem.bn)

    # blk0: [dw_ConvBNReLU6, project_Conv2d, project_BN]
    blk0 = bb.blocks[0].block
    cp_dwconv2d(2, blk0[0].conv); cp_bn(3, blk0[0].bn)
    cp_conv2d(4, blk0[1]);        cp_bn(5, blk0[2])

    for i in range(1, 17):
        base = 6 + (i - 1) * 6
        blk = bb.blocks[i].block  # [expand_ConvBNReLU6, dw_ConvBNReLU6, project_Conv2d, project_BN]
        cp_conv2d(base + 0, blk[0].conv); cp_bn(base + 1, blk[0].bn)
        cp_dwconv2d(base + 2, blk[1].conv); cp_bn(base + 3, blk[1].bn)
        cp_conv2d(base + 4, blk[2]);     cp_bn(base + 5, blk[3])

    cp_conv2d(102, bb.head.conv); cp_bn(103, bb.head.bn)

    # ---------- FPN ----------
    # Laterals: 1x1 with bias.
    n = model.neck
    fpn = "model/_feature_extractor/_fpn_features_generator"
    # top_layers/0 = lateral for C5 (1280->128).
    n.lat_c5.weight.data.copy_(_tf_kernel_to_torch(G(f"{fpn}/top_layers/0/kernel")))
    n.lat_c5.bias.data.copy_(torch.from_numpy(G(f"{fpn}/top_layers/0/bias")))
    # residual_blocks/0 = lateral for C4 (96->128); /1 = lateral for C3 (32->128).
    n.lat_c4.weight.data.copy_(_tf_kernel_to_torch(G(f"{fpn}/residual_blocks/0/0/kernel")))
    n.lat_c4.bias.data.copy_(torch.from_numpy(G(f"{fpn}/residual_blocks/0/0/bias")))
    n.lat_c3.weight.data.copy_(_tf_kernel_to_torch(G(f"{fpn}/residual_blocks/1/0/kernel")))
    n.lat_c3.bias.data.copy_(torch.from_numpy(G(f"{fpn}/residual_blocks/1/0/bias")))

    # conv_layers/0 = smooth P4; conv_layers/1 = smooth P3.
    def cp_sepconv(prefix: str, m: SeparableConvBNReLU6):
        m.dw.weight.data.copy_(_tf_dwkernel_to_torch(G(f"{prefix}/0/depthwise_kernel")))
        m.pw.weight.data.copy_(_tf_kernel_to_torch(G(f"{prefix}/0/pointwise_kernel")))
        m.bn.weight.data.copy_(torch.from_numpy(G(f"{prefix}/1/gamma")))
        m.bn.bias.data.copy_(torch.from_numpy(G(f"{prefix}/1/beta")))
        m.bn.running_mean.data.copy_(torch.from_numpy(G(f"{prefix}/1/moving_mean")))
        m.bn.running_var.data.copy_(torch.from_numpy(G(f"{prefix}/1/moving_variance")))

    cp_sepconv(f"{fpn}/conv_layers/0", n.smooth_p4)
    cp_sepconv(f"{fpn}/conv_layers/1", n.smooth_p3)
    # Coarse: _coarse_feature_layers/0 -> P6, /1 -> P7.
    coarse = "model/_feature_extractor/_coarse_feature_layers"
    cp_sepconv(f"{coarse}/0", n.coarse_p6)
    cp_sepconv(f"{coarse}/1", n.coarse_p7)

    # ---------- Head (tower + final cls/box) ----------
    bp = "model/_box_predictor"
    h = model.head
    # Shared tower DSConv kernels (depthwise + pointwise) per stage 0..3.
    for s in range(h.n_tower):
        h.tower_dw[s].weight.data.copy_(
            _tf_dwkernel_to_torch(G(f"{bp}/_head_scope_conv_layers/PredictionTower/{s}/depthwise_kernel"))
        )
        h.tower_pw[s].weight.data.copy_(
            _tf_kernel_to_torch(G(f"{bp}/_head_scope_conv_layers/PredictionTower/{s}/pointwise_kernel"))
        )
    # Per-level BN: levels 0..4 (P3..P7), BN slot indices 1, 4, 7, 10 (one per tower stage).
    bn_slots = [1, 4, 7, 10]
    for lvl in range(h.n_levels):
        for s, slot in enumerate(bn_slots):
            bn = h.tower_bn[lvl][s]
            base = f"{bp}/_base_tower_layers_for_heads/box_encodings/{lvl}/{slot}"
            bn.weight.data.copy_(torch.from_numpy(G(f"{base}/gamma")))
            bn.bias.data.copy_(torch.from_numpy(G(f"{base}/beta")))
            bn.running_mean.data.copy_(torch.from_numpy(G(f"{base}/moving_mean")))
            bn.running_var.data.copy_(torch.from_numpy(G(f"{base}/moving_variance")))

    # Final cls predictor: dw + pw + bias.
    cls_p = f"{bp}/_prediction_heads/class_predictions_with_background/_class_predictor_layers/0"
    h.cls_dw.weight.data.copy_(_tf_dwkernel_to_torch(G(f"{cls_p}/depthwise_kernel")))
    h.cls_pw.weight.data.copy_(_tf_kernel_to_torch(G(f"{cls_p}/pointwise_kernel")))
    h.cls_pw.bias.data.copy_(torch.from_numpy(G(f"{cls_p}/bias")))
    # Final box predictor.
    box_p = f"{bp}/_box_prediction_head/_box_encoder_layers/0"
    h.box_dw.weight.data.copy_(_tf_dwkernel_to_torch(G(f"{box_p}/depthwise_kernel")))
    h.box_pw.weight.data.copy_(_tf_kernel_to_torch(G(f"{box_p}/pointwise_kernel")))
    h.box_pw.bias.data.copy_(torch.from_numpy(G(f"{box_p}/bias")))

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Loaded TF weights into model: {n_params/1e6:.2f}M params")


# ---------------------------------------------------------------------------
# SSD postprocess: anchors, box decode, NMS
# ---------------------------------------------------------------------------

# Matches `multiscale_anchor_generator` in pipeline.config.
_ANCHOR_LEVELS = (3, 4, 5, 6, 7)
_ASPECT_RATIOS = (1.0, 2.0, 0.5)
_SCALES_PER_OCTAVE = 2
_ANCHOR_SCALE = 4.0
# Faster R-CNN box coder scale factors (y, x, h, w).
_BOX_SCALES = (10.0, 10.0, 5.0, 5.0)


def generate_anchors(image_size: int) -> torch.Tensor:
    """Generate multiscale SSD anchors in input-pixel coords.

    Returns: (N, 4) tensor of (cy, cx, h, w). Anchor order matches the head's
    flattened output: levels P3..P7, then row-major (i, j), then anchor index.
    """
    octave_scales = [2 ** (k / _SCALES_PER_OCTAVE) for k in range(_SCALES_PER_OCTAVE)]
    anchors = []
    for level in _ANCHOR_LEVELS:
        stride = 1 << level                   # 8, 16, 32, 64, 128
        base = _ANCHOR_SCALE * stride         # 32, 64, 128, 256, 512
        feat_size = image_size // stride
        # Anchor center grid (cy, cx) in input-pixel coords.
        yy = (torch.arange(feat_size, dtype=torch.float32) + 0.5) * stride
        xx = (torch.arange(feat_size, dtype=torch.float32) + 0.5) * stride
        cy, cx = torch.meshgrid(yy, xx, indexing="ij")
        cy = cy.reshape(-1, 1)                # (HW, 1)
        cx = cx.reshape(-1, 1)
        # Each anchor: (scale * sqrt(ratio), scale / sqrt(ratio)) for (w, h).
        hs, ws = [], []
        for octave in octave_scales:
            for ratio in _ASPECT_RATIOS:
                hs.append(base * octave / (ratio ** 0.5))
                ws.append(base * octave * (ratio ** 0.5))
        # Order TF uses: outer loop aspect ratio? Actually multiscale_anchor_generator
        # iterates `for scale in scales: for aspect in aspect_ratios:` which matches above.
        h = torch.tensor(hs).reshape(1, -1)   # (1, A)
        w = torch.tensor(ws).reshape(1, -1)
        # Broadcast to (HW, A) then flatten.
        cy = cy.expand(-1, h.shape[1])
        cx = cx.expand(-1, w.shape[1])
        h = h.expand(cy.shape[0], -1)
        w = w.expand(cx.shape[0], -1)
        anchors.append(torch.stack([cy, cx, h, w], dim=-1).reshape(-1, 4))
    return torch.cat(anchors, dim=0)


def decode_boxes(box_deltas: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Decode (ty, tx, th, tw) deltas against (cy, cx, h, w) anchors.

    Returns (N, 4) boxes as (ymin, xmin, ymax, xmax) in input-pixel coords.
    """
    ty = box_deltas[..., 0] / _BOX_SCALES[0]
    tx = box_deltas[..., 1] / _BOX_SCALES[1]
    th = box_deltas[..., 2] / _BOX_SCALES[2]
    tw = box_deltas[..., 3] / _BOX_SCALES[3]
    cy_a, cx_a, h_a, w_a = anchors.unbind(-1)
    cy = ty * h_a + cy_a
    cx = tx * w_a + cx_a
    h = torch.exp(th) * h_a
    w = torch.exp(tw) * w_a
    ymin = cy - 0.5 * h
    xmin = cx - 0.5 * w
    ymax = cy + 0.5 * h
    xmax = cx + 0.5 * w
    return torch.stack([ymin, xmin, ymax, xmax], dim=-1)


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    """Greedy NMS, pure torch (torchvision-free). boxes: (N,4) xyxy, scores: (N,).
    Returns kept indices ordered by descending score, matching torchvision.ops.nms."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(int(i))
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest]); yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest]); yy2 = torch.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_thresh]
    return torch.tensor(keep, dtype=torch.int64)


def nms_detections(boxes: torch.Tensor, scores: torch.Tensor,
                   score_thresh: float = 0.3, iou_thresh: float = 0.6,
                   max_total: int = 100):
    """Per-class NMS. scores: (N, K) for K foreground classes. Returns lists."""
    nms = _nms
    keep_boxes, keep_scores, keep_labels = [], [], []
    K = scores.shape[1]
    for c in range(K):
        s = scores[:, c]
        m = s > score_thresh
        if not m.any():
            continue
        b, s = boxes[m], s[m]
        keep = nms(b, s, iou_thresh)
        keep_boxes.append(b[keep])
        keep_scores.append(s[keep])
        keep_labels.append(torch.full((keep.numel(),), c + 1, dtype=torch.int64))  # +1: skip background
    if not keep_boxes:
        return torch.empty(0, 4), torch.empty(0), torch.empty(0, dtype=torch.int64)
    b = torch.cat(keep_boxes); s = torch.cat(keep_scores); l = torch.cat(keep_labels)
    if s.numel() > max_total:
        top = torch.topk(s, max_total).indices
        b, s, l = b[top], s[top], l[top]
    return b, s, l


# COCO 1..90 class names (TF mscoco_label_map.pbtxt; channel 0 = background).
COCO_LABELS = [
    "background",
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A",
    "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "N/A", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]




_BPE = 2  # bf16 bytes per element


# ---------------------------------------------------------------------------
# New helper (validated in user_hw_test.nearest_upsample_2x_test):
# 2x nearest-neighbor upsample on an HWC bf16 tensor.
# ---------------------------------------------------------------------------

def nearest_upsample_2x_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int,
                              OUTPUT_DRAM_ADDR: int, H_in: int, W_in: int, C: int,
                              sram_scratch: int = 0x00000) -> None:
    """For each input pixel, read C channels into SRAM once then scatter to the
    2x2 output block. Output layout (2*H_in, 2*W_in, C), HWC.
    Same pattern proven in user_hw_test.nearest_upsample_2x_test (SNR=inf dB).
    """
    pixel_bytes = C * _BPE
    in_row_bytes  = W_in * pixel_bytes
    out_row_bytes = (2 * W_in) * pixel_bytes
    for i in range(H_in):
        for j in range(W_in):
            in_off = i * in_row_bytes + j * pixel_bytes
            ue.accelerator_memory_to_sram(
                accelerator_dram_address=INPUT_DRAM_ADDR + in_off,
                sram_address=sram_scratch,
                element_size=C,
            )
            base_top = (2 * i) * out_row_bytes + (2 * j) * pixel_bytes
            base_bot = base_top + out_row_bytes
            for out_off in (base_top, base_top + pixel_bytes,
                            base_bot, base_bot + pixel_bytes):
                ue.sram_to_accelerator_memory(
                    sram_address=sram_scratch,
                    accelerator_dram_address=OUTPUT_DRAM_ADDR + out_off,
                    element_size=C,
                )


def _pack_dw_tapwise_bias_addr(w_dw_padded: torch.Tensor, W_chunk_max: int,
                                alloc_param, bias_addr: int) -> list:
    """Tile a depthwise kernel as 9 tap tiles per 64-channel block; bias points
    to a caller-supplied DRAM address (zero-bias tile, or BN-folded bias tile).
    Returns: list of (C//64) dicts {"taps": [9 DRAM addrs], "bias": bias_addr}.
    """
    BLK = 64
    assert w_dw_padded.shape[1] == 1 and w_dw_padded.shape[2] == 3 and w_dw_padded.shape[3] == 3
    C = w_dw_padded.shape[0]
    assert C % BLK == 0
    n_blocks = C // BLK
    blocks = []
    for b in range(n_blocks):
        c0 = b * BLK
        tap_addrs = []
        for dy in range(3):
            for dx in range(3):
                col = w_dw_padded[c0:c0 + BLK, 0, dy, dx]
                tile = col.unsqueeze(0).expand(W_chunk_max, BLK).contiguous()
                tap_addrs.append(alloc_param(tile))
        blocks.append({"taps": tap_addrs, "bias": bias_addr})
    return blocks


# ---------------------------------------------------------------------------
# SSD-FPNLite Unified Engine
# ---------------------------------------------------------------------------

SSD_PARAMS_BASE  = 0x00000000
SSD_TENSOR_BASE  = 0x40000000
SSD_PROGRAM_BASE = 0x80000000


class SSDFPNLite_UnifiedEngine(UnifiedEngine):
    """MobileNetV2-SSD-FPNLite (640x640) on the Unified Engine accelerator."""

    ALIGN = 64
    DMA_CHUNK_BYTES = 1 * 1024 * 1024
    DW_W_CHUNK_MAX  = 32

    def __init__(self, script_dir: str = SCRIPT_DIR):
        super().__init__(params_dram_base=SSD_PARAMS_BASE,
                         tensor_dram_base=SSD_TENSOR_BASE,
                         program_dram_base=SSD_PROGRAM_BASE)

        self.script_dir = script_dir
        self.cfg = _load_config(script_dir=script_dir)

        m = self.cfg["model"]
        b = self.cfg["backbone"]
        self.IMAGE_SIZE   = m["image_size"]
        self.NUM_CHANNELS = m["num_channels"]
        self.STEM_OUT     = b["stem_out_channels"]
        self.HEAD_OUT     = b["head_out_channels"]
        self.IR_SETTING   = [tuple(x) for x in b["inverted_residual_setting"]]
        self.C3_BLOCK_IDX = b["c3_block_idx"]
        self.C4_BLOCK_IDX = b["c4_block_idx"]
        self.FPN_CH       = self.cfg["fpn"]["channels"]
        self.NUM_ANCHORS  = m["num_anchors"]
        self.NUM_CLASSES  = m["num_classes"]
        self.N_LEVELS     = m["n_levels"]
        self.N_TOWER      = m["n_tower"]
        self.CLS_OUT_CH   = self.NUM_ANCHORS * self.NUM_CLASSES
        self.BOX_OUT_CH   = self.NUM_ANCHORS * 4
        # Padded output channels are deterministic from config and are needed by
        # tensor_init() even on the cached-params path (where weight_init(), which
        # also sets these, is skipped). Compute them here so both paths have them.
        self.CLS_OUT_CH_PAD = self.pad_dim(self.CLS_OUT_CH)
        self.BOX_OUT_CH_PAD = self.pad_dim(self.BOX_OUT_CH)

        # Flatten IR_SETTING into 17 per-block dicts with TRUE & padded channels.
        self.blocks = self._expand_blocks()

        self.params_from_bin = self.load_params()
        if not self.params_from_bin:
            # Build the CPU module ONCE, load TF weights into it, then mine its
            # state_dict for every fold/copy below. This avoids re-implementing TF
            # checkpoint name plumbing and ensures we use exactly the same tensors
            # the CPU reference does.
            self._cpu_model = MobileNetV2_SSD_FPNLite(
                self.cfg,
                fpn_ch=self.FPN_CH,
                num_anchors=self.NUM_ANCHORS,
                num_classes=self.NUM_CLASSES,
            ).eval()
            ckpt = _ensure_tf_checkpoint()
            _load_tf_weights(self._cpu_model, ckpt, verbose=False)
            self.weight_init()
            self.dump_params()

        self.tensor_init()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def pad_dim(x: int) -> int:
        return ((x + 63) // 64) * 64

    def _alloc_param(self, tensor: torch.Tensor) -> int:
        t = tensor.to(torch.bfloat16).contiguous()
        addr = self.get_params_dram_addr()
        nbytes = t.numel() * _BPE
        self.allocate_params_dram(nbytes)
        self.dma_to_accelerator_memory(addr, t)
        return addr

    def _alloc_tensor(self, num_elements: int) -> int:
        return self.allocate_tensor_dram(num_elements * _BPE)

    def write_captured_instructions_to_dram(self, start_addr: int = DRAM_INSTRUCTION_ADDR) -> int:
        """Chunked override — see mobilenetv2_test.py for rationale."""
        if not self.capture_buffer or self.capture_count == 0:
            return 0
        total_bytes = self.capture_count * 32
        all_bytes = bytearray()
        for inst in self.capture_buffer:
            all_bytes.extend(inst.get_bytes())
        data = bytes(all_bytes)
        if not _SILENT_MODE:
            _original_print(f"Writing {self.capture_count:,} instructions "
                            f"({total_bytes / 1024**2:.1f} MB) to DRAM at 0x{start_addr:x}...")
        offset = 0
        while offset < total_bytes:
            chunk = min(self.DMA_CHUNK_BYTES, total_bytes - offset)
            self.dma_write(DMA_DEVICE_H2C, start_addr + offset,
                           data[offset:offset + chunk], chunk)
            offset += chunk
        if not _SILENT_MODE:
            _original_print(f"Successfully wrote {total_bytes / 1024**2:.1f} MB "
                            f"({self.capture_count:,} instructions) to DRAM")
        return total_bytes

    def _expand_blocks(self) -> list[dict]:
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
    # Weight init
    # ------------------------------------------------------------------

    def _fold_pw_bn(self, pw_conv, bn) -> tuple[torch.Tensor, torch.Tensor]:
        """Fold BN(eps=1e-3) into a 1x1 conv (Cout, Cin, 1, 1). Returns (w_fused 2D, b_fused 1D)."""
        w_fused, b_fused = _bn_fold(
            pw_conv.weight.data,
            bn.weight.data, bn.bias.data,
            bn.running_mean.data, bn.running_var.data,
            eps=bn.eps,
        )
        return w_fused.squeeze(-1).squeeze(-1), b_fused

    def _pack_dw_no_bn(self, dw_conv, out_ch_padded: int) -> list:
        """Tile a depthwise kernel (no BN to fold) with zero bias for the tap-wise primitive."""
        w_raw = dw_conv.weight.data.to(torch.bfloat16)  # (C, 1, 3, 3)
        w_p = torch.zeros(out_ch_padded, 1, 3, 3, dtype=torch.bfloat16)
        w_p[:w_raw.shape[0]] = w_raw
        return _pack_dw_tapwise_bias_addr(
            w_p, self.DW_W_CHUNK_MAX, self._alloc_param, self.ZERO_BIAS_TILE
        )

    def weight_init(self) -> None:
        """Lay out every tensor the program will reference into params DRAM."""
        m = self._cpu_model
        pad = self.pad_dim

        # ---- Shared scratch tiles ----
        # Zero pad row: a 64-element zero row for stem im2col boundary fill, etc.
        max_c = max(pad(self.STEM_OUT),
                    max(b["in_ch_p"] for b in self.blocks),
                    max(b["mid_ch_p"] for b in self.blocks),
                    self.FPN_CH, pad(self.HEAD_OUT))
        self.ZERO_PAD = self._alloc_param(torch.zeros(max_c, dtype=torch.bfloat16))
        # Identity (64, 64) for the depthwise post-pass clamp matmul (when relu6_enable=True).
        self.IDENTITY_64 = self._alloc_param(torch.eye(64, dtype=torch.bfloat16))
        # Single zero bias tile (W_chunk_max, 64) shared across every "no-BN" depthwise.
        self.ZERO_BIAS_TILE = self._alloc_param(
            torch.zeros(self.DW_W_CHUNK_MAX, 64, dtype=torch.bfloat16)
        )

        # ============================================================
        # Backbone: stem + 17 IR blocks + head 1x1.  Mirrors mobilenetv2_test.weight_init().
        # ============================================================
        bb = m.backbone
        # Stem (3x3 stride-2, BN-folded, im2col-shaped, 3->32 padded 64->64).
        stem_w, stem_b = _bn_fold(
            bb.stem.conv.weight.data,
            bb.stem.bn.weight.data, bb.stem.bn.bias.data,
            bb.stem.bn.running_mean.data, bb.stem.bn.running_var.data,
            eps=bb.stem.bn.eps,
        )
        stem_w_p = _pad_channels_in_weight(stem_w, C_in_padded=64, C_out_padded=64)
        stem_b_p = torch.zeros(64, dtype=torch.bfloat16); stem_b_p[:self.STEM_OUT] = stem_b
        stem_w_im2col = stem_w_p.permute(0, 2, 3, 1).contiguous().reshape(64, 9 * 64)
        self.STEM_W = self._alloc_param(stem_w_im2col)
        self.STEM_B = self._alloc_param(stem_b_p)

        # IR blocks
        self.block_weights: list[dict] = []
        for i, b in enumerate(self.blocks):
            seq = bb.blocks[i].block  # nn.Sequential
            bw = {}

            if b["expand_ratio"] == 1:
                # blocks[0]: [dw_ConvBNReLU6, project_Conv2d, project_BN]
                dw_cb, proj_conv, proj_bn = seq[0], seq[1], seq[2]
            else:
                # blocks[1..16]: [expand_ConvBNReLU6, dw_ConvBNReLU6, project_Conv2d, project_BN]
                exp_cb, dw_cb, proj_conv, proj_bn = seq[0], seq[1], seq[2], seq[3]
                exp_w, exp_b = _bn_fold(
                    exp_cb.conv.weight.data,
                    exp_cb.bn.weight.data, exp_cb.bn.bias.data,
                    exp_cb.bn.running_mean.data, exp_cb.bn.running_var.data,
                    eps=exp_cb.bn.eps,
                )
                exp_w_2d = exp_w.squeeze(-1).squeeze(-1)
                w_p = torch.zeros(b["mid_ch_p"], b["in_ch_p"], dtype=torch.bfloat16)
                w_p[:b["mid_ch"], :b["in_ch"]] = exp_w_2d
                b_p = torch.zeros(b["mid_ch_p"], dtype=torch.bfloat16); b_p[:b["mid_ch"]] = exp_b
                bw["expand_w"] = self._alloc_param(w_p)
                bw["expand_b"] = self._alloc_param(b_p)

            # Depthwise (BN-folded into the dw weight & bias, tap-wise tiled).
            dw_w, dw_b = _bn_fold(
                dw_cb.conv.weight.data,
                dw_cb.bn.weight.data, dw_cb.bn.bias.data,
                dw_cb.bn.running_mean.data, dw_cb.bn.running_var.data,
                eps=dw_cb.bn.eps,
            )
            w_p = torch.zeros(b["mid_ch_p"], 1, 3, 3, dtype=torch.bfloat16)
            w_p[:b["mid_ch"]] = dw_w
            b_p = torch.zeros(b["mid_ch_p"], dtype=torch.bfloat16); b_p[:b["mid_ch"]] = dw_b
            tiles = _pack_dw_tapwise(w_p, b_p, self.DW_W_CHUNK_MAX)
            bw["dw_blocks"] = [
                {"taps": [self._alloc_param(t) for t in blk["taps"]],
                 "bias": self._alloc_param(blk["bias"])}
                for blk in tiles
            ]

            # Project 1x1 (LINEAR — no ReLU6, BN-folded).
            proj_w, proj_b = _bn_fold(
                proj_conv.weight.data,
                proj_bn.weight.data, proj_bn.bias.data,
                proj_bn.running_mean.data, proj_bn.running_var.data,
                eps=proj_bn.eps,
            )
            proj_w_2d = proj_w.squeeze(-1).squeeze(-1)
            w_p = torch.zeros(b["out_ch_p"], b["mid_ch_p"], dtype=torch.bfloat16)
            w_p[:b["out_ch"], :b["mid_ch"]] = proj_w_2d
            b_p = torch.zeros(b["out_ch_p"], dtype=torch.bfloat16); b_p[:b["out_ch"]] = proj_b
            bw["project_w"] = self._alloc_param(w_p)
            bw["project_b"] = self._alloc_param(b_p)

            self.block_weights.append(bw)

        # Head 1x1 320 -> 1280 + ReLU6 (BN-folded). Provides C5 (already aligned).
        head_w, head_b = _bn_fold(
            bb.head.conv.weight.data,
            bb.head.bn.weight.data, bb.head.bn.bias.data,
            bb.head.bn.running_mean.data, bb.head.bn.running_var.data,
            eps=bb.head.bn.eps,
        )
        head_w_2d = head_w.squeeze(-1).squeeze(-1)
        head_in_p  = pad(320)   # 320
        head_out_p = pad(1280)  # 1280
        w_p = torch.zeros(head_out_p, head_in_p, dtype=torch.bfloat16)
        w_p[:1280, :320] = head_w_2d
        b_p = torch.zeros(head_out_p, dtype=torch.bfloat16); b_p[:1280] = head_b
        self.HEAD_W = self._alloc_param(w_p)
        self.HEAD_B = self._alloc_param(b_p)

        # ============================================================
        # FPN-Lite neck.
        # Laterals are plain 1x1 + bias (no BN). C5 stays as-is; C4 96 -> pad 128;
        # C3 32 -> pad 64. All produce 128ch outputs.
        # ============================================================
        n = m.neck

        def alloc_lateral(lat_conv, in_ch_padded: int) -> tuple[int, int]:
            w = lat_conv.weight.data.squeeze(-1).squeeze(-1).to(torch.bfloat16)  # (128, Cin_true)
            b = lat_conv.bias.data.to(torch.bfloat16)                            # (128,)
            w_p = torch.zeros(self.FPN_CH, in_ch_padded, dtype=torch.bfloat16)
            w_p[:, :w.shape[1]] = w
            return self._alloc_param(w_p), self._alloc_param(b)

        self.LAT_C5_W, self.LAT_C5_B = alloc_lateral(n.lat_c5, pad(1280))
        self.LAT_C4_W, self.LAT_C4_B = alloc_lateral(n.lat_c4, pad(96))
        self.LAT_C3_W, self.LAT_C3_B = alloc_lateral(n.lat_c3, pad(32))

        # SeparableConv = dw (no BN) + pw + BN + ReLU6. dw goes via tap-wise with
        # zero bias; pw absorbs BN.
        def alloc_sepconv(sep):
            dw_blocks = self._pack_dw_no_bn(sep.dw, self.FPN_CH)
            pw_w_2d, pw_b = self._fold_pw_bn(sep.pw, sep.bn)  # (128, 128), (128,)
            return {
                "dw_blocks": dw_blocks,
                "pw_w": self._alloc_param(pw_w_2d),
                "pw_b": self._alloc_param(pw_b),
            }

        self.FPN_SMOOTH_P4 = alloc_sepconv(n.smooth_p4)
        self.FPN_SMOOTH_P3 = alloc_sepconv(n.smooth_p3)
        self.FPN_COARSE_P6 = alloc_sepconv(n.coarse_p6)
        self.FPN_COARSE_P7 = alloc_sepconv(n.coarse_p7)

        # ============================================================
        # SSD head: shared 4-DSConv tower with per-level BN, then final cls/box.
        # dw kernels are truly shared across levels. pw kernels are shared across
        # levels too, but we precompute 5 BN-folded copies of (pw_w, pw_b) per
        # tower stage — one per level — so each level's tower call points at its
        # own folded weights via conv2d_1x1_dram with clamp.
        # ============================================================
        h = m.head

        # Per-stage shared dw (no BN). Same pack as FPN sepconv.
        self.TOWER_DW: list[list[dict]] = []
        for s in range(self.N_TOWER):
            self.TOWER_DW.append(self._pack_dw_no_bn(h.tower_dw[s], self.FPN_CH))

        # Per (level, stage) folded pw weights + bias.
        self.TOWER_PW_W: list[list[int]] = [[None] * self.N_TOWER for _ in range(self.N_LEVELS)]
        self.TOWER_PW_B: list[list[int]] = [[None] * self.N_TOWER for _ in range(self.N_LEVELS)]
        for level in range(self.N_LEVELS):
            for s in range(self.N_TOWER):
                pw_w_2d, pw_b = self._fold_pw_bn(h.tower_pw[s], h.tower_bn[level][s])
                self.TOWER_PW_W[level][s] = self._alloc_param(pw_w_2d)
                self.TOWER_PW_B[level][s] = self._alloc_param(pw_b)

        # Final cls/box predictors: dw (no BN) + pw with bias (TF bias direct), no act.
        self.CLS_DW = self._pack_dw_no_bn(h.cls_dw, self.FPN_CH)
        cls_pw_w_pad = pad(self.CLS_OUT_CH)  # 546 -> 576
        cls_w_2d = h.cls_pw.weight.data.squeeze(-1).squeeze(-1).to(torch.bfloat16)  # (546, 128)
        cls_b    = h.cls_pw.bias.data.to(torch.bfloat16)                            # (546,)
        w_p = torch.zeros(cls_pw_w_pad, self.FPN_CH, dtype=torch.bfloat16)
        w_p[:self.CLS_OUT_CH] = cls_w_2d
        b_p = torch.zeros(cls_pw_w_pad, dtype=torch.bfloat16); b_p[:self.CLS_OUT_CH] = cls_b
        self.CLS_PW_W = self._alloc_param(w_p)
        self.CLS_PW_B = self._alloc_param(b_p)
        self.CLS_OUT_CH_PAD = cls_pw_w_pad

        self.BOX_DW = self._pack_dw_no_bn(h.box_dw, self.FPN_CH)
        box_pw_w_pad = pad(self.BOX_OUT_CH)  # 24 -> 64
        box_w_2d = h.box_pw.weight.data.squeeze(-1).squeeze(-1).to(torch.bfloat16)  # (24, 128)
        box_b    = h.box_pw.bias.data.to(torch.bfloat16)                            # (24,)
        w_p = torch.zeros(box_pw_w_pad, self.FPN_CH, dtype=torch.bfloat16)
        w_p[:self.BOX_OUT_CH] = box_w_2d
        b_p = torch.zeros(box_pw_w_pad, dtype=torch.bfloat16); b_p[:self.BOX_OUT_CH] = box_b
        self.BOX_PW_W = self._alloc_param(w_p)
        self.BOX_PW_B = self._alloc_param(b_p)
        self.BOX_OUT_CH_PAD = box_pw_w_pad

    # ------------------------------------------------------------------
    # Tensor (activation) DRAM allocation
    # ------------------------------------------------------------------

    # Spatial sizes per FPN level, in (P3, P4, P5, P6, P7) order. Anchor levels 3..7.
    LEVEL_HW = ((80, 80), (40, 40), (20, 20), (10, 10), (5, 5))

    def tensor_init(self) -> None:
        # ---- Backbone scratch ----
        H = self.IMAGE_SIZE
        self.IMAGE_DRAM = self._alloc_tensor(H * H * 64)  # 640x640x64 image (3 padded -> 64)

        # IM2COL scratch (used only by the stem); same sizing as 224 model.
        max_im2col_stem = max(1, URAM_NEAR_FULL_ELEMENTS // (9 * 64)) * 9 * 64
        self.IM2COL_DRAM = self._alloc_tensor(max_im2col_stem)

        # Stem output: 320x320x64.
        stem_spatial = self.IMAGE_SIZE // 2
        self.STEM_OUT_DRAM = self._alloc_tensor(stem_spatial * stem_spatial * 64)

        # Depthwise accumulator scratch (one 64-channel block at a time).
        # Worst case across backbone IR blocks AND FPN/head sepconvs is block 0's
        # dw at 320x320 (stride 1) = 320*320*64 elements = 13 MB. FPN/head all
        # have smaller spatial (≤80x80), so the backbone bound covers them too.
        max_dw_acc_elems = max(
            max(b["H_out"] * b["W_out"] * 64 for b in self.blocks),
            80 * 80 * 64,  # tower stage on P3 (and FPN smooth_p3)
        )
        self.DW_ACC_DRAM = self._alloc_tensor(max_dw_acc_elems)

        # Per-block scratch (expanded, dw_out, block_out).
        self.block_tensors: list[dict] = []
        for b in self.blocks:
            t = {}
            if b["expand_ratio"] != 1:
                t["expanded"] = self._alloc_tensor(b["H_in"] * b["W_in"] * b["mid_ch_p"])
            t["dw_out"]    = self._alloc_tensor(b["H_out"] * b["W_out"] * b["mid_ch_p"])
            t["block_out"] = self._alloc_tensor(b["H_out"] * b["W_out"] * b["out_ch_p"])
            self.block_tensors.append(t)

        # Head 1x1 output = C5 (20x20x1280).
        last_spatial = self.blocks[-1]["H_out"]
        self.HEAD_OUT_DRAM = self._alloc_tensor(last_spatial * last_spatial * self.pad_dim(self.HEAD_OUT))

        # ---- FPN buffers ----
        FCH = self.FPN_CH
        (H3, W3), (H4, W4), (H5, W5), (H6, W6), (H7, W7) = self.LEVEL_HW

        # P5
        self.P5_DRAM    = self._alloc_tensor(H5 * W5 * FCH)
        self.P5_UP_DRAM = self._alloc_tensor((2 * H5) * (2 * W5) * FCH)  # 40x40x128

        # P4
        self.P4_LAT_DRAM = self._alloc_tensor(H4 * W4 * FCH)
        self.P4_ADD_DRAM = self._alloc_tensor(H4 * W4 * FCH)
        self.P4_DW_DRAM  = self._alloc_tensor(H4 * W4 * FCH)
        self.P4_DRAM     = self._alloc_tensor(H4 * W4 * FCH)
        self.P4_UP_DRAM  = self._alloc_tensor((2 * H4) * (2 * W4) * FCH)  # 80x80x128

        # P3
        self.P3_LAT_DRAM = self._alloc_tensor(H3 * W3 * FCH)
        self.P3_ADD_DRAM = self._alloc_tensor(H3 * W3 * FCH)
        self.P3_DW_DRAM  = self._alloc_tensor(H3 * W3 * FCH)
        self.P3_DRAM     = self._alloc_tensor(H3 * W3 * FCH)

        # P6, P7 (coarse, each stride-2 sepconv from the prior P-level)
        self.P6_DW_DRAM = self._alloc_tensor(H6 * W6 * FCH)
        self.P6_DRAM    = self._alloc_tensor(H6 * W6 * FCH)
        self.P7_DW_DRAM = self._alloc_tensor(H7 * W7 * FCH)
        self.P7_DRAM    = self._alloc_tensor(H7 * W7 * FCH)

        # Convenience: anchor-order list of (addr, H, W) for the head loop.
        self.feat_addrs = [self.P3_DRAM, self.P4_DRAM, self.P5_DRAM, self.P6_DRAM, self.P7_DRAM]

        # ---- Head buffers ----
        # Two ping-pong scratch buffers sized to the largest level (P3 80x80x128).
        # We process one level at a time so the same two buffers serve all 5 levels
        # across the 4-stage tower plus the cls/box dw step.
        max_tower_elems = max(h * w * FCH for (h, w) in self.LEVEL_HW)
        self.TOWER_A = self._alloc_tensor(max_tower_elems)
        self.TOWER_B = self._alloc_tensor(max_tower_elems)

        # Per-level final cls/box outputs (padded channel count).
        self.CLS_OUT_DRAM: list[int] = []
        self.BOX_OUT_DRAM: list[int] = []
        for (h, w) in self.LEVEL_HW:
            self.CLS_OUT_DRAM.append(self._alloc_tensor(h * w * self.CLS_OUT_CH_PAD))
            self.BOX_OUT_DRAM.append(self._alloc_tensor(h * w * self.BOX_OUT_CH_PAD))

    # ------------------------------------------------------------------
    # Bin cache: params + programs
    # ------------------------------------------------------------------

    def _bin_dir(self) -> str:
        return os.path.join(self.script_dir, "mobilenetv2_ssd_fpnlite_bin")

    def dump_params(self) -> None:
        bin_dir = self._bin_dir()
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
            json.dump({"size": total}, f)
        _original_print(f"  Params dumped: {total / 1024**2:.1f} MB → {bin_path}")

    def load_params(self) -> bool:
        bin_dir = self._bin_dir()
        bin_path = os.path.join(bin_dir, "params.bin")
        meta_path = os.path.join(bin_dir, "params.json")
        if not os.path.exists(bin_path) or not os.path.exists(meta_path):
            return False
        with open(meta_path) as f:
            meta = json.load(f)
        total = meta["size"]
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

    def dump_programs(self, program_addr: int) -> None:
        bin_dir = self._bin_dir()
        os.makedirs(bin_dir, exist_ok=True)
        bin_path = os.path.join(bin_dir, "programs.bin")
        meta_path = os.path.join(bin_dir, "programs.json")
        size = self.get_program_dram_usage()
        buf = bytearray(size)
        self.dma_read(DMA_DEVICE_C2H, program_addr, buf, size)
        with open(bin_path, "wb") as f:
            f.write(buf)
        with open(meta_path, "w") as f:
            json.dump({"size": size}, f)
        _original_print(f"  Program dumped: {size / 1024**2:.1f} MB → {bin_path}")

    def load_programs(self) -> int | None:
        bin_dir = self._bin_dir()
        bin_path = os.path.join(bin_dir, "programs.bin")
        meta_path = os.path.join(bin_dir, "programs.json")
        if not os.path.exists(bin_path) or not os.path.exists(meta_path):
            return None
        with open(bin_path, "rb") as f:
            data = f.read()
        addr = self.get_program_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, addr, data, len(data))
        self.allocate_program_dram(len(data))
        _original_print(f"  Program loaded: {len(data) / 1024**2:.1f} MB from bin")
        return addr

    # ------------------------------------------------------------------
    # Compile: stem -> backbone -> head -> FPN -> SSD head (single program)
    # ------------------------------------------------------------------

    def _sepconv(self, sep_w: dict, src: int, dw_out: int, dst: int,
                 H_in: int, W_in: int, stride: int) -> None:
        """SeparableConv2D = dw(no BN, no clamp) -> pw + BN-folded bias + ReLU6."""
        H_out = (H_in - 1) // stride + 1 if stride > 1 else H_in
        W_out = (W_in - 1) // stride + 1 if stride > 1 else W_in
        conv2d_3x3_dw_tapwise_dram(
            self,
            INPUT_DRAM_ADDR=src, OUTPUT_DRAM_ADDR=dw_out,
            ACC_DRAM_ADDR=self.DW_ACC_DRAM,
            IDENTITY_DRAM_ADDR=self.IDENTITY_64,
            ZERO_PAD_DRAM_ADDR=self.ZERO_PAD,
            block_params=sep_w["dw_blocks"],
            H_in=H_in, W_in=W_in, C=self.FPN_CH,
            stride=stride, W_chunk_max=self.DW_W_CHUNK_MAX,
            relu6_enable=False,
        )
        conv2d_1x1_dram(
            self,
            INPUT_DRAM_ADDR=dw_out, OUTPUT_DRAM_ADDR=dst,
            WEIGHT_DRAM_ADDR=sep_w["pw_w"], BIAS_DRAM_ADDR=sep_w["pw_b"],
            H=H_out, W=W_out, C_in=self.FPN_CH, C_out=self.FPN_CH,
            relu6_enable=True,
        )

    def _tower_level(self, level: int, src: int, H: int, W: int) -> None:
        """Run 4-stage shared tower + cls + box for one FPN level.
        dw kernels are shared (TOWER_DW[s]); pw weights/bias are level-specific
        (TOWER_PW_W/B[level][s]) since BN is per-level.
        After the tower, cls and box use shared dw + level-shared final pw with bias.
        """
        FCH = self.FPN_CH
        cur = src
        for s in range(self.N_TOWER):
            # dw (no BN, no clamp): cur -> TOWER_A
            conv2d_3x3_dw_tapwise_dram(
                self,
                INPUT_DRAM_ADDR=cur, OUTPUT_DRAM_ADDR=self.TOWER_A,
                ACC_DRAM_ADDR=self.DW_ACC_DRAM,
                IDENTITY_DRAM_ADDR=self.IDENTITY_64,
                ZERO_PAD_DRAM_ADDR=self.ZERO_PAD,
                block_params=self.TOWER_DW[s],
                H_in=H, W_in=W, C=FCH,
                stride=1, W_chunk_max=self.DW_W_CHUNK_MAX,
                relu6_enable=False,
            )
            # pw + BN-folded bias + ReLU6: TOWER_A -> TOWER_B
            conv2d_1x1_dram(
                self,
                INPUT_DRAM_ADDR=self.TOWER_A, OUTPUT_DRAM_ADDR=self.TOWER_B,
                WEIGHT_DRAM_ADDR=self.TOWER_PW_W[level][s],
                BIAS_DRAM_ADDR=self.TOWER_PW_B[level][s],
                H=H, W=W, C_in=FCH, C_out=FCH,
                relu6_enable=True,
            )
            cur = self.TOWER_B

        # cls: dw (shared, no BN, no clamp) -> TOWER_A ; pw + TF bias (no act) -> CLS_OUT
        conv2d_3x3_dw_tapwise_dram(
            self,
            INPUT_DRAM_ADDR=cur, OUTPUT_DRAM_ADDR=self.TOWER_A,
            ACC_DRAM_ADDR=self.DW_ACC_DRAM,
            IDENTITY_DRAM_ADDR=self.IDENTITY_64,
            ZERO_PAD_DRAM_ADDR=self.ZERO_PAD,
            block_params=self.CLS_DW,
            H_in=H, W_in=W, C=FCH,
            stride=1, W_chunk_max=self.DW_W_CHUNK_MAX,
            relu6_enable=False,
        )
        conv2d_1x1_dram(
            self,
            INPUT_DRAM_ADDR=self.TOWER_A, OUTPUT_DRAM_ADDR=self.CLS_OUT_DRAM[level],
            WEIGHT_DRAM_ADDR=self.CLS_PW_W, BIAS_DRAM_ADDR=self.CLS_PW_B,
            H=H, W=W, C_in=FCH, C_out=self.CLS_OUT_CH_PAD,
            relu6_enable=False,
        )

        # box (same shape; reuses TOWER_A scratch after cls is done writing to CLS_OUT).
        conv2d_3x3_dw_tapwise_dram(
            self,
            INPUT_DRAM_ADDR=cur, OUTPUT_DRAM_ADDR=self.TOWER_A,
            ACC_DRAM_ADDR=self.DW_ACC_DRAM,
            IDENTITY_DRAM_ADDR=self.IDENTITY_64,
            ZERO_PAD_DRAM_ADDR=self.ZERO_PAD,
            block_params=self.BOX_DW,
            H_in=H, W_in=W, C=FCH,
            stride=1, W_chunk_max=self.DW_W_CHUNK_MAX,
            relu6_enable=False,
        )
        conv2d_1x1_dram(
            self,
            INPUT_DRAM_ADDR=self.TOWER_A, OUTPUT_DRAM_ADDR=self.BOX_OUT_DRAM[level],
            WEIGHT_DRAM_ADDR=self.BOX_PW_W, BIAS_DRAM_ADDR=self.BOX_PW_B,
            H=H, W=W, C_in=FCH, C_out=self.BOX_OUT_CH_PAD,
            relu6_enable=False,
        )

    def compile_full_fused(self) -> int:
        """Emit one instruction stream: stem -> 17 IR -> head -> FPN -> SSD-head."""
        self.start_capture()

        # PBI: one GPR holds the per-layer M=H*W for every 1x1 conv. conv2d_1x1_dram
        # primes it via ADD_SET and passes gpr_M_reg, so each pointwise GEMM's M-tile
        # loop runs as a runtime hardware loop instead of a compile-time unroll. Held
        # for the whole program (never freed) so matmat_mul_core_pbi's internal tile
        # registers get distinct indices from alloc_isa_reg.
        self._pbi_M_reg = self.alloc_isa_reg()

        # =================== STEM ===================
        conv2d_3x3_stride2_dram(
            self,
            INPUT_DRAM_ADDR=self.IMAGE_DRAM,
            OUTPUT_DRAM_ADDR=self.STEM_OUT_DRAM,
            IM2COL_DRAM_ADDR=self.IM2COL_DRAM,
            WEIGHT_DRAM_ADDR=self.STEM_W, BIAS_DRAM_ADDR=self.STEM_B,
            ZERO_PAD_DRAM_ADDR=self.ZERO_PAD,
            H_in=self.IMAGE_SIZE, W_in=self.IMAGE_SIZE,
            C_in=64, C_out=64,
            relu6_enable=True,
        )

        # =================== 17 IR BLOCKS ===================
        prev_out = self.STEM_OUT_DRAM
        for i, b in enumerate(self.blocks):
            bw = self.block_weights[i]
            bt = self.block_tensors[i]

            # Expand 1x1 + ReLU6 (skipped for block 0).
            if b["expand_ratio"] != 1:
                conv2d_1x1_dram(
                    self,
                    INPUT_DRAM_ADDR=prev_out, OUTPUT_DRAM_ADDR=bt["expanded"],
                    WEIGHT_DRAM_ADDR=bw["expand_w"], BIAS_DRAM_ADDR=bw["expand_b"],
                    H=b["H_in"], W=b["W_in"],
                    C_in=b["in_ch_p"], C_out=b["mid_ch_p"],
                    relu6_enable=True,
                )
                dw_in = bt["expanded"]
            else:
                dw_in = prev_out

            # Depthwise 3x3 (BN-folded) + ReLU6.
            conv2d_3x3_dw_tapwise_dram(
                self,
                INPUT_DRAM_ADDR=dw_in, OUTPUT_DRAM_ADDR=bt["dw_out"],
                ACC_DRAM_ADDR=self.DW_ACC_DRAM,
                IDENTITY_DRAM_ADDR=self.IDENTITY_64,
                ZERO_PAD_DRAM_ADDR=self.ZERO_PAD,
                block_params=bw["dw_blocks"],
                H_in=b["H_in"], W_in=b["W_in"], C=b["mid_ch_p"],
                stride=b["stride"], W_chunk_max=self.DW_W_CHUNK_MAX,
                relu6_enable=True,
            )

            # Project 1x1 (LINEAR, BN-folded — no activation).
            conv2d_1x1_dram(
                self,
                INPUT_DRAM_ADDR=bt["dw_out"], OUTPUT_DRAM_ADDR=bt["block_out"],
                WEIGHT_DRAM_ADDR=bw["project_w"], BIAS_DRAM_ADDR=bw["project_b"],
                H=b["H_out"], W=b["W_out"],
                C_in=b["mid_ch_p"], C_out=b["out_ch_p"],
                relu6_enable=False,
            )

            # Residual add when stride==1 and in==out.
            if b["has_residual"]:
                eltwise_add_dram(
                    self,
                    A_ADDR=prev_out, B_ADDR=bt["block_out"],
                    OUT_ADDR=bt["block_out"],
                    num_elements=b["H_out"] * b["W_out"] * b["out_ch_p"],
                )

            prev_out = bt["block_out"]

        # Remember C3 / C4 / C5 addresses for the FPN.
        C3_ADDR = self.block_tensors[self.C3_BLOCK_IDX]["block_out"]   # 80x80x64 (32 padded)
        C4_ADDR = self.block_tensors[self.C4_BLOCK_IDX]["block_out"]   # 40x40x128 (96 padded)

        # =================== HEAD 1x1 (-> C5) ===================
        last_spatial = self.blocks[-1]["H_out"]
        conv2d_1x1_dram(
            self,
            INPUT_DRAM_ADDR=prev_out, OUTPUT_DRAM_ADDR=self.HEAD_OUT_DRAM,
            WEIGHT_DRAM_ADDR=self.HEAD_W, BIAS_DRAM_ADDR=self.HEAD_B,
            H=last_spatial, W=last_spatial,
            C_in=self.pad_dim(320), C_out=self.pad_dim(self.HEAD_OUT),
            relu6_enable=True,
        )
        C5_ADDR = self.HEAD_OUT_DRAM   # 20x20x1280

        # =================== FPN-LITE ===================
        # P5 = lat_c5(C5)   [1x1 + bias, no activation]
        conv2d_1x1_dram(
            self,
            INPUT_DRAM_ADDR=C5_ADDR, OUTPUT_DRAM_ADDR=self.P5_DRAM,
            WEIGHT_DRAM_ADDR=self.LAT_C5_W, BIAS_DRAM_ADDR=self.LAT_C5_B,
            H=20, W=20, C_in=self.pad_dim(1280), C_out=self.FPN_CH,
            relu6_enable=False,
        )
        # P5 upsampled to 40x40.
        nearest_upsample_2x_dram(
            self, INPUT_DRAM_ADDR=self.P5_DRAM, OUTPUT_DRAM_ADDR=self.P5_UP_DRAM,
            H_in=20, W_in=20, C=self.FPN_CH,
        )
        # P4_lat = lat_c4(C4)   [C4 is 40x40x128, last 32 channels padded to 0]
        conv2d_1x1_dram(
            self,
            INPUT_DRAM_ADDR=C4_ADDR, OUTPUT_DRAM_ADDR=self.P4_LAT_DRAM,
            WEIGHT_DRAM_ADDR=self.LAT_C4_W, BIAS_DRAM_ADDR=self.LAT_C4_B,
            H=40, W=40, C_in=self.pad_dim(96), C_out=self.FPN_CH,
            relu6_enable=False,
        )
        eltwise_add_dram(
            self, A_ADDR=self.P4_LAT_DRAM, B_ADDR=self.P5_UP_DRAM,
            OUT_ADDR=self.P4_ADD_DRAM, num_elements=40 * 40 * self.FPN_CH,
        )
        # smooth_p4 SepConv -> P4
        self._sepconv(self.FPN_SMOOTH_P4, self.P4_ADD_DRAM, self.P4_DW_DRAM, self.P4_DRAM,
                      H_in=40, W_in=40, stride=1)
        # P4 upsampled to 80x80.
        nearest_upsample_2x_dram(
            self, INPUT_DRAM_ADDR=self.P4_DRAM, OUTPUT_DRAM_ADDR=self.P4_UP_DRAM,
            H_in=40, W_in=40, C=self.FPN_CH,
        )
        # P3_lat = lat_c3(C3)   [C3 is 80x80x64, last 32 channels padded to 0]
        conv2d_1x1_dram(
            self,
            INPUT_DRAM_ADDR=C3_ADDR, OUTPUT_DRAM_ADDR=self.P3_LAT_DRAM,
            WEIGHT_DRAM_ADDR=self.LAT_C3_W, BIAS_DRAM_ADDR=self.LAT_C3_B,
            H=80, W=80, C_in=self.pad_dim(32), C_out=self.FPN_CH,
            relu6_enable=False,
        )
        eltwise_add_dram(
            self, A_ADDR=self.P3_LAT_DRAM, B_ADDR=self.P4_UP_DRAM,
            OUT_ADDR=self.P3_ADD_DRAM, num_elements=80 * 80 * self.FPN_CH,
        )
        # smooth_p3 SepConv -> P3
        self._sepconv(self.FPN_SMOOTH_P3, self.P3_ADD_DRAM, self.P3_DW_DRAM, self.P3_DRAM,
                      H_in=80, W_in=80, stride=1)
        # coarse_p6 SepConv(P5, stride=2) -> P6
        self._sepconv(self.FPN_COARSE_P6, self.P5_DRAM, self.P6_DW_DRAM, self.P6_DRAM,
                      H_in=20, W_in=20, stride=2)
        # coarse_p7 SepConv(P6, stride=2) -> P7
        self._sepconv(self.FPN_COARSE_P7, self.P6_DRAM, self.P7_DW_DRAM, self.P7_DRAM,
                      H_in=10, W_in=10, stride=2)

        # =================== SSD HEAD (tower + cls + box, per level) ===================
        for level in range(self.N_LEVELS):
            H, W = self.LEVEL_HW[level]
            self._tower_level(level, src=self.feat_addrs[level], H=H, W=W)

        self.stop_capture()
        self.generate_instruction_halt()

        prog_addr = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(prog_addr)
        inst_count = self.capture_count
        self.allocate_program_dram(inst_count * 32)
        self.clear_capture_buffer()
        return prog_addr, inst_count

    # ------------------------------------------------------------------
    # Run + DMA-back -> per-level cls/box tensors -> CPU postprocess
    # ------------------------------------------------------------------

    def run_full_fused(self, pixel_values: torch.Tensor, program_addr: int,
                       timeout: float = 600.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute the program; return concatenated (cls_logits, box_deltas) in TF anchor order.

        pixel_values: (1, 3, H, W) float. Padded to (H, W, 64) HWC bf16 and shipped
        to IMAGE_DRAM. After wait_queue, reads back all 5 levels' cls/box tensors,
        slices the padded channels, reshapes to (H*W*A, K) / (H*W*A, 4) and concats.
        """
        image_chw = pixel_values.squeeze(0).to(torch.bfloat16)               # (3, H, W)
        image_hwc = image_chw.permute(1, 2, 0).contiguous()                  # (H, W, 3)
        H = image_hwc.shape[0]
        image_hwc_p = torch.zeros(H, H, 64, dtype=torch.bfloat16)
        image_hwc_p[:, :, :3] = image_hwc
        self.dma_to_accelerator_memory(self.IMAGE_DRAM, image_hwc_p.contiguous().flatten())

        t0 = time.perf_counter()
        self.start_execute_from_dram(program_addr)
        self.wait_queue(timeout)
        self.last_inference_seconds = time.perf_counter() - t0

        # Read each level's cls/box, drop the channel padding, flatten to (H*W*A, K).
        A, K = self.NUM_ANCHORS, self.NUM_CLASSES
        cls_pieces, box_pieces = [], []
        for level, (h, w) in enumerate(self.LEVEL_HW):
            cls_p = self.dma_from_accelerator_memory(
                self.CLS_OUT_DRAM[level], (h, w, self.CLS_OUT_CH_PAD)
            )[..., :self.CLS_OUT_CH]                            # (h, w, 546)
            box_p = self.dma_from_accelerator_memory(
                self.BOX_OUT_DRAM[level], (h, w, self.BOX_OUT_CH_PAD)
            )[..., :self.BOX_OUT_CH]                            # (h, w, 24)
            cls_pieces.append(cls_p.reshape(h * w * A, K).float())
            box_pieces.append(box_p.reshape(h * w * A, 4).float())
        cls = torch.cat(cls_pieces, dim=0)                      # (51150, 91)
        box = torch.cat(box_pieces, dim=0)                      # (51150, 4)
        return cls, box


# ---------------------------------------------------------------------------
# Image preprocessing (TF/HF MBV2: [-1, 1] after resize)
# ---------------------------------------------------------------------------

def preprocess_image(image_path: str, size: int = 640) -> tuple[torch.Tensor, "Image.Image", int, int]:
    img = Image.open(image_path).convert("RGB")
    W0, H0 = img.size
    img_r = img.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(img_r, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return x, img, W0, H0


def draw_and_save(img: "Image.Image", W0: int, H0: int, size: int,
                  keep_boxes, keep_scores, keep_labels, order, out_path: str) -> None:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                  max(12, int(min(W0, H0) * 0.02)))
    except OSError:
        font = ImageFont.load_default()
    import colorsys
    def _color(c):
        r, g, b = colorsys.hsv_to_rgb((c * 0.137) % 1.0, 0.85, 1.0)
        return int(r * 255), int(g * 255), int(b * 255)
    sx = W0 / size; sy = H0 / size
    line_w = max(2, int(min(W0, H0) * 0.004))
    for idx in order:
        ymin, xmin, ymax, xmax = keep_boxes[idx].tolist()
        x0, y0 = xmin * sx, ymin * sy
        x1, y1 = xmax * sx, ymax * sy
        cls_id = int(keep_labels[idx])
        color = _color(cls_id)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=line_w)
        text = f"{COCO_LABELS[cls_id]} {keep_scores[idx].item()*100:.1f}%"
        tb = draw.textbbox((x0, y0), text, font=font)
        pad = 2
        draw.rectangle([tb[0] - pad, tb[1] - pad, tb[2] + pad, tb[3] + pad], fill=color)
        draw.text((x0, y0), text, fill=(0, 0, 0), font=font)
    out.save(out_path, quality=92)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _clock_ns_default_for_device(device: str) -> float:
    """Return default clock period (ns) for FPGA type — mirrors user_hw_test.py."""
    if device == "kintex7":                       return 5.1594
    if device in ("rk", "puzhi"):                 return 3.0
    if device in ("bittware", "bittware_256"):     return 3.3333
    if device == "alveo":                          return 4.0
    return 10.0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MBV2-SSD-FPNLite (640) on accelerator.")
    parser.add_argument("--image", type=str, default=None,
                        help="Input image (default: ../../test_samples/vette.jpg)")
    parser.add_argument("--dev", type=str, default=None, help="DMA device override (default: board profile)")
    parser.add_argument("--cycle", type=float, default=None, help='Clock cycle time in ns. Overrides --device default.')
    parser.add_argument("--device", type=str, default="kintex7", help='FPGA board profile (kintex7, rk, puzhi, bittware, bittware_256, alveo).')
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--iou-thresh", type=float, default=0.6)
    args = parser.parse_args()

    global _SILENT_MODE
    _SILENT_MODE = True

    profile = configure_device(args.device, dma_device=args.dev)
    global DMA_DEVICE_H2C, DMA_DEVICE_C2H
    DMA_DEVICE_H2C = user_dma_core.DMA_DEVICE_H2C
    DMA_DEVICE_C2H = user_dma_core.DMA_DEVICE_C2H
    axi_width_bits = int(profile.get("axi_data_width_bits") or (512 if args.device in ("bittware", "rk") else 256))
    os.environ["UE_AXI_DATA_WIDTH_BITS"] = str(axi_width_bits)
    user_dma_core.UE_AXI_DATA_WIDTH_BITS = axi_width_bits
    clock = args.cycle if args.cycle is not None else (profile.get("clock_period_ns") or _clock_ns_default_for_device(args.device))
    user_dma_core.CLOCK_CYCLE_TIME_NS = clock
    user_dma_core.UE_PEAK_GFLOPS = 0.128 / clock
    _original_print(f"FPGA profile: device={profile['device']}, clock={clock:.4f} ns, UE_AXI_DATA_WIDTH_BITS={axi_width_bits}")
    _original_print(f"Using DMA: H2C={DMA_DEVICE_H2C}, C2H={DMA_DEVICE_C2H}, USER={user_dma_core.DMA_DEVICE_USER}")

    image_path = args.image or os.path.join(SCRIPT_DIR, "../../test_samples/vette.jpg")
    _image_size = _load_config(SCRIPT_DIR)["model"]["image_size"]
    pixel_values, img, W0, H0 = preprocess_image(image_path, size=_image_size)

    _original_print(f"MBV2-SSD-FPNLite-640 on {DMA_DEVICE_H2C}")

    import threading
    def _progress(label, t_start, stop_event):
        while not stop_event.wait(1.0):
            elapsed = time.perf_counter() - t_start
            _original_print(f"\r  {label} ({elapsed:.0f}s)", end="", flush=True)

    t0 = time.perf_counter()
    ue = SSDFPNLite_UnifiedEngine(script_dir=SCRIPT_DIR)
    ue.software_reset()
    t_w = time.perf_counter()
    _original_print(f"  Weights: {t_w - t0:.1f}s")

    prog_addr = ue.load_programs()
    if prog_addr is None:
        if ue.params_from_bin:
            raise RuntimeError("Cannot compile without weight addresses — delete mobilenetv2_ssd_fpnlite_bin/ and re-run")
        stop_c = threading.Event()
        timer_c = threading.Thread(target=_progress, args=("Compiling", t_w, stop_c), daemon=True)
        timer_c.start()
        prog_addr, inst_count = ue.compile_full_fused()
        stop_c.set(); timer_c.join()
        t_c = time.perf_counter()
        ue.dump_programs(prog_addr)
        _original_print(f"\r  Compile: {t_c - t_w:.1f}s (instructions={inst_count:,})")
    else:
        t_c = time.perf_counter()
        _original_print(f"\r  Compile: {t_c - t_w:.1f}s (from bin)")

    t_e = time.perf_counter()
    stop = threading.Event()
    timer = threading.Thread(target=_progress, args=("Executing", t_e, stop), daemon=True)
    timer.start()
    cls_logits, box_deltas = ue.run_full_fused(pixel_values, prog_addr)
    stop.set(); timer.join()
    _original_print(f"\r  Executing: {time.perf_counter() - t_e:.1f}s "
                    f"(pure HW: {ue.last_inference_seconds * 1000:.1f} ms)")

    # ---------- CPU postprocess ----------
    size = ue.IMAGE_SIZE
    anchors = generate_anchors(size)
    assert anchors.shape[0] == cls_logits.shape[0], (anchors.shape, cls_logits.shape)
    boxes = decode_boxes(box_deltas, anchors)
    scores = torch.sigmoid(cls_logits[:, 1:])  # drop background
    keep_boxes, keep_scores, keep_labels = nms_detections(
        boxes, scores, score_thresh=args.score_thresh, iou_thresh=args.iou_thresh,
    )

    sx = W0 / size; sy = H0 / size
    _original_print(f"\n  Image: {image_path}  (orig {W0}x{H0})")
    _original_print(f"  Detections (score >= {args.score_thresh}):")
    if keep_scores.numel() == 0:
        _original_print("    (none)")
        order = torch.empty(0, dtype=torch.long)
    else:
        order = torch.argsort(keep_scores, descending=True)
        for idx in order:
            ymin, xmin, ymax, xmax = keep_boxes[idx].tolist()
            label = COCO_LABELS[int(keep_labels[idx])]
            _original_print(f"    {keep_scores[idx].item()*100:5.1f}%  {label:18s}  "
                            f"[{xmin*sx:7.1f}, {ymin*sy:7.1f}, {xmax*sx:7.1f}, {ymax*sy:7.1f}]")

    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(SCRIPT_DIR, f"{stem}_detections_hw.jpg")
    draw_and_save(img, W0, H0, size, keep_boxes, keep_scores, keep_labels, order, out_path)
    _original_print(f"  Saved: {out_path}")

    import json as _json
    det_labels = [COCO_LABELS[int(keep_labels[i])] for i in order] if keep_scores.numel() > 0 else []
    _original_print("TEST_RESULT:" + _json.dumps({
        "decoded_text": ", ".join(det_labels) if det_labels else "(no detections)",
        "n_detections": len(det_labels),
    }))


if __name__ == "__main__":
    main()
