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
import sys
import time
import warnings
warnings.filterwarnings("ignore", message=".*torchao.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import torch
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
    set_dma_device, UnifiedEngine,
)

# Reuse all the building blocks from the 224 MBV2 hardware test.
from mobilenetv2_test import (
    conv2d_1x1_dram,
    conv2d_3x3_stride2_dram,
    conv2d_3x3_dw_tapwise_dram,
    eltwise_add_dram,
    dram_zero_fill,
    _bn_fold,
    _pad_channels_in_weight,
    _pack_dw_tapwise,
    LALU_BF16_SIX,
)

# CPU reference (also used for TF weight loading and postprocess).
from mobilenetv2_ssd_fpnlite_640_cpu_test import (
    MobileNetV2_SSD_FPNLite,
    _ensure_tf_checkpoint,
    _load_tf_weights,
    generate_anchors,
    decode_boxes,
    nms_detections,
    COCO_LABELS,
)
from mobilenetv2_cpu_test import _load_config

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

    # --- Backbone (same as 224 MBV2) ---
    IMAGE_SIZE   = 640
    NUM_CHANNELS = 3
    STEM_OUT     = 32
    HEAD_OUT     = 1280
    IR_SETTING = [
        (1,  16, 1, 1),
        (6,  24, 2, 2),
        (6,  32, 3, 2),
        (6,  64, 4, 2),
        (6,  96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]
    # C3 = block 5 (32ch, stride 8, 80x80). C4 = block 12 (96ch, stride 16, 40x40).
    # C5 = head 1x1 output (1280ch, stride 32, 20x20).
    C3_BLOCK_IDX = 5
    C4_BLOCK_IDX = 12

    # --- FPN / head ---
    FPN_CH      = 128
    NUM_ANCHORS = 6
    NUM_CLASSES = 91
    N_LEVELS    = 5     # P3..P7
    N_TOWER     = 4
    CLS_OUT_CH  = NUM_ANCHORS * NUM_CLASSES   # 546 -> pad 576
    BOX_OUT_CH  = NUM_ANCHORS * 4             # 24  -> pad 64

    ALIGN = 64
    DMA_CHUNK_BYTES = 1 * 1024 * 1024
    DW_W_CHUNK_MAX  = 32

    USE_BIN_CACHE = False  # keep off while iterating

    def __init__(self, script_dir: str = SCRIPT_DIR):
        super().__init__(params_dram_base=SSD_PARAMS_BASE,
                         tensor_dram_base=SSD_TENSOR_BASE,
                         program_dram_base=SSD_PROGRAM_BASE)

        # Make matmat_mul_core's clamp_enable be ReLU6 instead of plain ReLU
        # (matches what mobilenetv2_test.py does — same monkey-patch).
        user_dma_core.LALU_CLAMP_RELU_B = LALU_BF16_SIX

        self.script_dir = script_dir
        self.cfg = _load_config(script_dir=script_dir)

        # Flatten IR_SETTING into 17 per-block dicts with TRUE & padded channels.
        self.blocks = self._expand_blocks()

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
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        self.clear_capture_buffer()
        return prog_addr

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
    parser.add_argument("--dev", type=str, default="xdma0")
    parser.add_argument("--cycle", type=float, default=None, help='Clock cycle time in ns. Overrides --device default.')
    parser.add_argument("--device", type=str, default="kintex7", help='FPGA board profile (kintex7, rk, puzhi, bittware, bittware_256, alveo).')
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--iou-thresh", type=float, default=0.6)
    args = parser.parse_args()

    global _SILENT_MODE
    _SILENT_MODE = True

    set_dma_device(args.dev)
    axi_width_bits = 512 if args.device in ("bittware", "rk") else 256
    os.environ["UE_AXI_DATA_WIDTH_BITS"] = str(axi_width_bits)
    user_dma_core.UE_AXI_DATA_WIDTH_BITS = axi_width_bits
    clock = args.cycle if args.cycle is not None else _clock_ns_default_for_device(args.device)
    user_dma_core.CLOCK_CYCLE_TIME_NS = clock
    user_dma_core.UE_PEAK_GFLOPS = 0.128 / clock
    _original_print(f"FPGA profile: device={args.device}, clock={clock:.4f} ns, UE_AXI_DATA_WIDTH_BITS={axi_width_bits}")

    image_path = args.image or os.path.join(SCRIPT_DIR, "../../test_samples/vette.jpg")
    pixel_values, img, W0, H0 = preprocess_image(image_path,
                                                  size=SSDFPNLite_UnifiedEngine.IMAGE_SIZE)

    _original_print(f"MBV2-SSD-FPNLite-640 on {args.dev}")

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

    stop_c = threading.Event()
    timer_c = threading.Thread(target=_progress, args=("Compiling", t_w, stop_c), daemon=True)
    timer_c.start()
    prog_addr = ue.compile_full_fused()
    stop_c.set(); timer_c.join()
    t_c = time.perf_counter()
    _original_print(f"\r  Compile: {t_c - t_w:.1f}s "
                    f"(instructions={ue.get_capture_instruction_size_bytes() // 32:,})")

    t_e = time.perf_counter()
    stop = threading.Event()
    timer = threading.Thread(target=_progress, args=("Executing", t_e, stop), daemon=True)
    timer.start()
    cls_logits, box_deltas = ue.run_full_fused(pixel_values, prog_addr)
    stop.set(); timer.join()
    _original_print(f"\r  Executing: {time.perf_counter() - t_e:.1f}s "
                    f"(pure HW: {ue.last_inference_seconds * 1000:.1f} ms)")

    # ---------- CPU postprocess ----------
    size = SSDFPNLite_UnifiedEngine.IMAGE_SIZE
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


if __name__ == "__main__":
    main()
