#!/usr/bin/env python3
"""
MobileNetV2 (1.0) — CPU pythonic implementation.

Mirrors the project's _test.py download pattern (HF snapshot_download via
mobilenetv2_config.json), but runs entirely on CPU with plain torch ops.
No user_dma_core / UnifiedEngine yet — this is the reference forward pass.

Forward: stem conv -> 7 stages of inverted residual blocks -> head conv.
Default input 640x640 (target for MBV2-SSD 640). The MBV2 backbone is
resolution-independent — only spatial dims change.

Usage:
    python mobilenetv2_test.py
    python mobilenetv2_test.py --image path/to/image.jpg
    python mobilenetv2_test.py --size 224
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import snapshot_download

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Config / HF download
# ---------------------------------------------------------------------------

def _load_config(script_dir: str = SCRIPT_DIR) -> dict:
    with open(os.path.join(script_dir, "mobilenetv2_config.json")) as f:
        return json.load(f)


def _ensure_hf_model(script_dir: str, cfg: dict):
    """Ensure HF MobileNetV2 weights are present. Returns (state_dict, model_dir).

    Same pattern used across the repo's other _test.py files: snapshot_download
    on first run, then load the state_dict. Source choice doesn't matter for
    our forward pass; we just need raw tensors to remap later.
    """
    paths = cfg["paths"]
    model_dir = os.path.join(script_dir, paths["hf_model_dir"])
    hf_repo = paths["hf_model_repo"]

    if not os.path.exists(os.path.join(model_dir, "config.json")):
        print(f"Downloading {hf_repo} -> {os.path.abspath(model_dir)} ...")
        snapshot_download(
            repo_id=hf_repo,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin"],
        )
        print("Download complete.")

    # Prefer safetensors directly to avoid pulling in transformers' MBV2 class.
    st_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(st_path):
        from safetensors.torch import load_file
        return load_file(st_path), model_dir

    pt_path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(pt_path):
        return torch.load(pt_path, map_location="cpu"), model_dir

    # Last resort: let transformers materialize the state dict.
    from transformers import AutoModelForImageClassification
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    return model.state_dict(), model_dir


# ---------------------------------------------------------------------------
# MobileNetV2 (pythonic CPU)
# ---------------------------------------------------------------------------

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


class MobileNetV2Backbone(nn.Module):
    """MobileNetV2 1.0 backbone. Returns the final 1280-channel feature map."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.head(x)
        return x


# ---------------------------------------------------------------------------
# HF -> ours weight remap
# ---------------------------------------------------------------------------

def _load_hf_weights(hf_sd: dict, model: MobileNetV2Backbone) -> None:
    """Copy HF MobileNetV2 weights into our pythonic module in-place.

    HF (transformers MobileNetV2) layout:
      mobilenet_v2.conv_stem.first_conv.*        -> stem (3x3 s2, 3->32)
      mobilenet_v2.conv_stem.conv_3x3.*          -> blocks[0] depthwise (first IR, no expand)
      mobilenet_v2.conv_stem.reduce_1x1.*        -> blocks[0] project (32->16)
      mobilenet_v2.layer.{i}.expand_1x1.*        -> blocks[i+1] expand   (i=0..15)
      mobilenet_v2.layer.{i}.conv_3x3.*          -> blocks[i+1] depthwise
      mobilenet_v2.layer.{i}.reduce_1x1.*        -> blocks[i+1] project
      mobilenet_v2.conv_1x1.*                    -> head (1x1, 320->1280)
    """

    def cp_conv(prefix: str, conv: nn.Conv2d):
        conv.weight.data.copy_(hf_sd[f"{prefix}.convolution.weight"])

    def cp_bn(prefix: str, bn: nn.BatchNorm2d):
        bn.weight.data.copy_(hf_sd[f"{prefix}.normalization.weight"])
        bn.bias.data.copy_(hf_sd[f"{prefix}.normalization.bias"])
        bn.running_mean.data.copy_(hf_sd[f"{prefix}.normalization.running_mean"])
        bn.running_var.data.copy_(hf_sd[f"{prefix}.normalization.running_var"])

    def cp_convbn(prefix: str, cb: ConvBNReLU6):
        cp_conv(prefix, cb.conv)
        cp_bn(prefix, cb.bn)

    cp_convbn("mobilenet_v2.conv_stem.first_conv", model.stem)

    # blocks[0]: first IR, no expand -> [depthwise, project_conv, project_bn]
    blk0 = model.blocks[0].block
    cp_convbn("mobilenet_v2.conv_stem.conv_3x3", blk0[0])
    cp_conv("mobilenet_v2.conv_stem.reduce_1x1", blk0[1])
    cp_bn("mobilenet_v2.conv_stem.reduce_1x1", blk0[2])

    # blocks[1..16]: with expand -> [expand, depthwise, project_conv, project_bn]
    for i in range(16):
        blk = model.blocks[i + 1].block
        p = f"mobilenet_v2.layer.{i}"
        cp_convbn(f"{p}.expand_1x1", blk[0])
        cp_convbn(f"{p}.conv_3x3", blk[1])
        cp_conv(f"{p}.reduce_1x1", blk[2])
        cp_bn(f"{p}.reduce_1x1", blk[3])

    cp_convbn("mobilenet_v2.conv_1x1", model.head)


def _parity_check(model: MobileNetV2Backbone, model_dir: str, size: int = 224):
    """Compare our backbone output to HF's MobileNetV2Model on the same input."""
    from transformers import MobileNetV2Model
    hf = MobileNetV2Model.from_pretrained(model_dir).eval()
    torch.manual_seed(0)
    x = torch.randn(1, 3, size, size)
    with torch.no_grad():
        ours = model(x)
        theirs = hf(pixel_values=x).last_hidden_state
    diff = (ours - theirs).abs()
    print(f"Parity vs HF MobileNetV2Model @ {size}x{size}:")
    print(f"  ours   shape={tuple(ours.shape)}   mean={ours.mean().item():+.6f}")
    print(f"  theirs shape={tuple(theirs.shape)} mean={theirs.mean().item():+.6f}")
    print(f"  mean|diff|={diff.mean().item():.2e}  max|diff|={diff.max().item():.2e}")


def _classify(feat: torch.Tensor, hf_sd: dict, model_dir: str, k: int = 5):
    """Use HF's pooled classifier head to turn our features into ImageNet top-k."""
    pooled = feat.mean(dim=[2, 3])
    w = hf_sd["classifier.weight"]
    b = hf_sd["classifier.bias"]
    logits = pooled @ w.t() + b
    probs = F.softmax(logits, dim=-1)
    top = torch.topk(probs[0], k)
    with open(os.path.join(model_dir, "config.json")) as f:
        id2label = json.load(f)["id2label"]
    return [(id2label[str(i.item())], p.item()) for i, p in zip(top.indices, top.values)]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",
                        default=os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)),
                                             "test_samples", "vette.jpg"),
                        help="Input image path (default: test_samples/vette.jpg)")
    parser.add_argument("--size", type=int, default=None,
                        help="Override input size (default from config)")
    args = parser.parse_args()

    cfg = _load_config()
    input_size = args.size or cfg["model"]["input_size"]

    # Step 1: confirm download path works (mirrors other models' _test.py).
    sd, model_dir = _ensure_hf_model(SCRIPT_DIR, cfg)
    print(f"HF state_dict loaded: {len(sd)} tensors from {model_dir}")
    sample_keys = list(sd.keys())[:5]
    print("First 5 keys:")
    for k in sample_keys:
        print(f"  {k}  {tuple(sd[k].shape)}")

    # Step 2: build pythonic CPU MBV2 and load HF pretrained weights into it.
    model = MobileNetV2Backbone(cfg).eval()
    _load_hf_weights(sd, model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MobileNetV2 backbone (HF pretrained): {n_params/1e6:.2f}M params")

    # Step 3: parity check vs HF's own MBV2 at 224 (its trained size).
    _parity_check(model, model_dir, size=224)

    # Step 4: forward at the requested input size.
    if args.image:
        img = Image.open(args.image).convert("RGB").resize((input_size, input_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5  # MBV2 normalize: [-1, 1]
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    else:
        torch.manual_seed(0)
        x = torch.randn(1, 3, input_size, input_size)

    import time as _time
    with torch.no_grad():
        # Warm up once so the timed run isn't paying first-call overhead
        # (kernel select, allocator warmup, etc.).
        _ = model(x)
        t0 = _time.perf_counter()
        y = model(x)
        t_inf = _time.perf_counter() - t0
    exp_spatial = input_size // 32
    print(f"Input  shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}  (expected: (1, 1280, {exp_spatial}, {exp_spatial}))")
    print(f"Inference (pure CPU): {t_inf * 1000:.3f} ms")

    if args.image:
        for label, prob in _classify(y, sd, model_dir):
            print(f"  {prob*100:5.2f}%  {label}")


if __name__ == "__main__":
    main()
