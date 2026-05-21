#!/usr/bin/env python3
"""
MobileNetV2-SSD-FPNLite (640x640) — CPU pythonic implementation.

Architecture follows the TensorFlow Object Detection API config
`ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8`:

  Backbone: MobileNetV2-1.0 (input-size-agnostic)
            taps C3 (stride 8, 32ch), C4 (stride 16, 96ch), C5 (stride 32, 320ch)
  Neck:     FPN-Lite, 128 channels, depthwise-separable 3x3 smoothing,
            top-down add + nearest upsample, two extra levels P6/P7 from P5.
  Head:     SSDLite shared cls+box, depthwise-separable, 6 anchors/loc,
            91 classes (90 COCO + background).

This file only builds the network and runs a forward pass to validate
output shapes. Weight conversion from the TF checkpoint, anchor generation,
box decode, and NMS are next steps.

Backbone primitives are reused from mobilenetv2_cpu_test.py (same dir).

Usage:
    python mobilenetv2_ssd_fpnlite_640_cpu_test.py
    python mobilenetv2_ssd_fpnlite_640_cpu_test.py --size 1056
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from mobilenetv2_cpu_test import ConvBNReLU6, InvertedResidual, _load_config


# ---------------------------------------------------------------------------
# Multi-scale MobileNetV2 backbone (returns C3, C4, C5)
# ---------------------------------------------------------------------------

class MobileNetV2BackboneMultiScale(nn.Module):
    """MBV2-1.0 backbone exposing 3 taps used by SSD-FPNLite.

    For MBV2-1.0 with IR_SETTING:
        (1, 16,1,1), (6, 24,2,2), (6, 32,3,2), (6, 64,4,2),
        (6, 96,3,1), (6,160,3,2), (6,320,1,1)
    block indices and their post-block strides:
        blocks[ 0]: 16ch stride 2
        blocks[1-2]: 24ch stride 4
        blocks[3-5]: 32ch stride 8     <- C3 (after block 5)
        blocks[6-9]: 64ch stride 16
        blocks[10-12]: 96ch stride 16  <- C4 (after block 12)
        blocks[13-15]:160ch stride 32
        blocks[16]: 320ch stride 32    <- C5 (after block 16)
    """

    C3_BLOCK_IDX = 5    # stride 8,  32ch
    C4_BLOCK_IDX = 12   # stride 16, 96ch
    C5_BLOCK_IDX = 16   # stride 32, 320ch
    C3_CH, C4_CH, C5_CH = 32, 96, 320

    def __init__(self, cfg: dict):
        super().__init__()
        b = cfg["backbone"]
        stem_ch = b["stem_out_channels"]
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
        # No head conv: FPN taps from block outputs directly.

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        c3 = c4 = c5 = None
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == self.C3_BLOCK_IDX:
                c3 = x
            elif i == self.C4_BLOCK_IDX:
                c4 = x
            elif i == self.C5_BLOCK_IDX:
                c5 = x
        return c3, c4, c5


# ---------------------------------------------------------------------------
# FPN-Lite neck
# ---------------------------------------------------------------------------

class DSConvBNReLU6(nn.Module):
    """Depthwise-separable 3x3: DW Conv -> BN -> ReLU6 -> PW Conv -> BN -> ReLU6.

    Used for FPNLite smoothing convs and SSDLite head convs.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = ConvBNReLU6(in_ch, in_ch, kernel_size=3, stride=stride, groups=in_ch)
        self.pw = ConvBNReLU6(in_ch, out_ch, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class FPNLiteNeck(nn.Module):
    """Top-down FPN-Lite: lateral 1x1, nearest upsample + add, DSConv smoothing.

    Produces P3, P4, P5 from C3, C4, C5; then P6 = DSConv stride-2 on P5,
    P7 = DSConv stride-2 on P6. All FPN feature maps have `out_ch` channels.
    """

    def __init__(self, c3_ch: int, c4_ch: int, c5_ch: int, out_ch: int = 128):
        super().__init__()
        self.out_ch = out_ch
        # Lateral 1x1 convs (no activation in classic FPN; TF FPNLite uses
        # plain 1x1 conv + BN; ReLU6 is OK as a forward-only stand-in here).
        self.lat3 = ConvBNReLU6(c3_ch, out_ch, kernel_size=1)
        self.lat4 = ConvBNReLU6(c4_ch, out_ch, kernel_size=1)
        self.lat5 = ConvBNReLU6(c5_ch, out_ch, kernel_size=1)
        # Smoothing convs (DSConv 3x3)
        self.smooth3 = DSConvBNReLU6(out_ch, out_ch)
        self.smooth4 = DSConvBNReLU6(out_ch, out_ch)
        self.smooth5 = DSConvBNReLU6(out_ch, out_ch)
        # Extra levels
        self.p6 = DSConvBNReLU6(out_ch, out_ch, stride=2)
        self.p7 = DSConvBNReLU6(out_ch, out_ch, stride=2)

    def forward(self, c3, c4, c5):
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)
        p5 = self.smooth5(p5)
        p6 = self.p6(p5)
        p7 = self.p7(p6)
        return [p3, p4, p5, p6, p7]


# ---------------------------------------------------------------------------
# SSDLite shared head
# ---------------------------------------------------------------------------

class SSDLiteHead(nn.Module):
    """Shared SSDLite cls + box heads across FPN levels.

    Per level: DSConv 3x3 (smooth) -> 1x1 conv to A*K (cls) or A*4 (box).
    Weights are shared across the 5 levels (single module, applied each).
    """

    def __init__(self, in_ch: int = 128, num_anchors: int = 6, num_classes: int = 91):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        # Tower: 4 DSConvs (TF default) then a 1x1 prediction conv.
        # FPNLite uses 4 shared depthwise-separable convs as the tower.
        self.cls_tower = nn.Sequential(
            DSConvBNReLU6(in_ch, in_ch),
            DSConvBNReLU6(in_ch, in_ch),
            DSConvBNReLU6(in_ch, in_ch),
            DSConvBNReLU6(in_ch, in_ch),
        )
        self.box_tower = nn.Sequential(
            DSConvBNReLU6(in_ch, in_ch),
            DSConvBNReLU6(in_ch, in_ch),
            DSConvBNReLU6(in_ch, in_ch),
            DSConvBNReLU6(in_ch, in_ch),
        )
        self.cls_pred = nn.Conv2d(in_ch, num_anchors * num_classes, kernel_size=1)
        self.box_pred = nn.Conv2d(in_ch, num_anchors * 4, kernel_size=1)

    def forward(self, feats):
        cls_outs, box_outs = [], []
        for f in feats:
            cls = self.cls_pred(self.cls_tower(f))   # (B, A*K, H, W)
            box = self.box_pred(self.box_tower(f))   # (B, A*4, H, W)
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
        self.backbone = MobileNetV2BackboneMultiScale(cfg)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=640,
                        help="Input image size (square). Supported: 640, 1056.")
    parser.add_argument("--num-classes", type=int, default=91)
    parser.add_argument("--num-anchors", type=int, default=6)
    parser.add_argument("--fpn-ch", type=int, default=128)
    args = parser.parse_args()

    cfg = _load_config()
    model = MobileNetV2_SSD_FPNLite(
        cfg,
        fpn_ch=args.fpn_ch,
        num_anchors=args.num_anchors,
        num_classes=args.num_classes,
    ).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"MobileNetV2-SSD-FPNLite ({args.size}x{args.size}): {n_params/1e6:.2f}M params")

    torch.manual_seed(0)
    x = torch.randn(1, 3, args.size, args.size)
    with torch.no_grad():
        _ = model(x)  # warmup
        t0 = time.perf_counter()
        cls, box, feats = model(x)
        t_inf = time.perf_counter() - t0

    print(f"Input  shape: {tuple(x.shape)}")
    for i, f in enumerate(feats, start=3):
        print(f"  P{i} shape: {tuple(f.shape)}")
    print(f"cls out: {tuple(cls.shape)}  (B, total_anchors, num_classes)")
    print(f"box out: {tuple(box.shape)}  (B, total_anchors, 4)")

    # Sanity: anchor count must equal sum H*W*A across levels.
    expected = sum(f.shape[-1] * f.shape[-2] for f in feats) * args.num_anchors
    assert cls.shape[1] == expected, (cls.shape, expected)
    print(f"Total anchors: {expected}")
    print(f"Inference (pure CPU): {t_inf * 1000:.3f} ms")


if __name__ == "__main__":
    main()
