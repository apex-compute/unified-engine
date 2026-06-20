#!/usr/bin/env python
"""CPU golden oracle for the MobileSAM vision encoder.

Runs the OFFICIAL mobile_sam (vit_t) TinyViT image encoder on the HW's EXACT
input (test_samples/vette.jpg, BILINEAR resize to 1024, /255, NO pixel mean/std
normalization -- matching mobilesam_test.py's input pipeline) and dumps the final
encoder output (the 64x64x256 image embedding) plus every block boundary so the HW
can be compared stage-by-stage.

Output: models/mobilesam/cpu_golden.npz
  key "neck"   -> (64, 64, 256)  HWC, the image embedding  (matches HW NECK_OUT)
  key "S0B0" .. "S3B1", "PM01"/"PM12"/"PM23", "patch_embed" -> (M, C) HWC

Run with the apex-compute conda python:
  /home/rohit/miniconda3/envs/apex-compute/bin/python models/mobilesam/cpu_reference.py
"""
import os
import numpy as np
import torch
from PIL import Image
from mobile_sam import sam_model_registry

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(SCRIPT_DIR, "mobilesam_bin", "mobile_sam.pt")
IMG_PATH = os.path.join(SCRIPT_DIR, "..", "..", "test_samples", "vette.jpg")
OUT_NPZ = os.path.join(SCRIPT_DIR, "cpu_golden.npz")
ENC_IN = 1024


def build_input():
    img = Image.open(IMG_PATH).convert("RGB").resize((ENC_IN, ENC_IN), Image.BILINEAR)
    arr = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    # NO pixel mean/std normalization -- HW feeds /255 directly.
    return arr.unsqueeze(0).float()  # (1,3,1024,1024) fp32


class SigmoidGELU(torch.nn.Module):
    """The HW's gelu activation: x * sigmoid(1.702 x) (sigmoid approximation), NOT
    the exact erf gelu of nn.GELU(). The oracle MUST match what the HW computes, or
    every gelu stage shows a false ~28 dB 'divergence'."""
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


def _swap_gelu(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.GELU):
            setattr(module, name, SigmoidGELU())
        else:
            _swap_gelu(child)


def main():
    sam = sam_model_registry["vit_t"](checkpoint=WEIGHTS)
    sam.eval()
    enc = sam.image_encoder  # TinyViT
    _swap_gelu(enc)          # match the HW's sigmoid-approx gelu
    x = build_input()

    captured = {}

    def hwc(t):
        """Normalize a captured tensor to (M, C) HWC, like HW DRAM."""
        if t.ndim == 4:           # (B,C,H,W)
            t = t[0].permute(1, 2, 0).reshape(-1, t.shape[1])
        elif t.ndim == 3:         # (B, L, C)
            t = t[0]
        return t.detach().float().numpy()

    hooks = []
    # patch_embed
    hooks.append(enc.patch_embed.register_forward_hook(
        lambda m, i, o: captured.__setitem__("patch_embed", hwc(o))))
    # layer 0 = ConvLayer (MBConv blocks), layers 1..3 = BasicLayer
    for li, layer in enumerate(enc.layers):
        for bi, blk in enumerate(layer.blocks):
            name = f"S{li}B{bi}"
            hooks.append(blk.register_forward_hook(
                lambda m, i, o, n=name: captured.__setitem__(n, hwc(o))))
        if getattr(layer, "downsample", None) is not None:
            name = f"PM{li}{li+1}"
            hooks.append(layer.downsample.register_forward_hook(
                lambda m, i, o, n=name: captured.__setitem__(n, hwc(o))))

    # S0B1 MBConv sub-op bisect (each block loses ~10 dB): conv1+gelu (act1),
    # conv2+gelu (act2), conv3 pre-residual. All BCHW -> HxW; c1/c2 are hidden=256ch.
    _mb = enc.layers[0].blocks[1]
    hooks.append(_mb.act1.register_forward_hook(
        lambda m, i, o: captured.__setitem__("S0B1_c1", hwc(o))))
    hooks.append(_mb.act2.register_forward_hook(
        lambda m, i, o: captured.__setitem__("S0B1_c2", hwc(o))))
    hooks.append(_mb.conv3.register_forward_hook(
        lambda m, i, o: captured.__setitem__("S0B1_c3", hwc(o))))

    # S2B0 sub-op bisect: local_conv input = post-attention-residual (A),
    # local_conv output = post-local-conv (B). Both BCHW -> HxW order, directly
    # comparable to the HW S2_DBG_A / S2_REV_DRAM buffers.
    _s2b0_lc = enc.layers[2].blocks[0].local_conv
    hooks.append(_s2b0_lc.register_forward_pre_hook(
        lambda m, i: captured.__setitem__("S2B0_attn", hwc(i[0]))))
    hooks.append(_s2b0_lc.register_forward_hook(
        lambda m, i, o: captured.__setitem__("S2B0_lc", hwc(o))))

    # Finer attention-path split: proj input = merged multi-head output (pre-proj),
    # proj output = post-proj. Both are (nwin, 196, 160) in WINDOW order, matching the
    # HW S2_MERGED_DRAM / S2_PROJ_DRAM buffers (which are (nwin, 256, 192) padded).
    _s2b0_proj = enc.layers[2].blocks[0].attn.proj
    hooks.append(_s2b0_proj.register_forward_pre_hook(
        lambda m, i: captured.__setitem__("S2B0_merged", i[0].detach().float().numpy())))
    hooks.append(_s2b0_proj.register_forward_hook(
        lambda m, i, o: captured.__setitem__("S2B0_proj", o.detach().float().numpy())))

    # qkv split (window order, (nwin,196,160)): per-head layout col = head*32 + d,
    # matching the model's W3.reshape(heads,3,hd,C). Splits qkv-matmul vs multihead_reshape.
    _s2b0_qkv = enc.layers[2].blocks[0].attn.qkv
    def _cap_qkv(m, i, o):   # o: (nwin, 196, 480)
        Bn, N, _ = o.shape
        qkv = o.detach().float().reshape(Bn, N, 5, 96)
        captured["S2B0_q"] = qkv[..., :32].reshape(Bn, N, 160).numpy()
        captured["S2B0_k"] = qkv[..., 32:64].reshape(Bn, N, 160).numpy()
        captured["S2B0_v"] = qkv[..., 64:96].reshape(Bn, N, 160).numpy()
    hooks.append(_s2b0_qkv.register_forward_hook(_cap_qkv))

    # attn.norm output (window order, (nwin,196,160)) — the EXACT LayerNorm the model
    # approximates with padded channels. vs HW S2_WIN: splits LN/partition vs qkv-matmul.
    _s2b0_norm = enc.layers[2].blocks[0].attn.norm
    hooks.append(_s2b0_norm.register_forward_hook(
        lambda m, i, o: captured.__setitem__("S2B0_norm", o.detach().float().numpy())))

    with torch.no_grad():
        out = enc(x)  # (1,256,64,64)
    for h in hooks:
        h.remove()

    captured["neck"] = out[0].permute(1, 2, 0).reshape(-1, out.shape[1]).numpy()  # (4096,256)

    np.savez(OUT_NPZ, **captured)
    print(f"saved {OUT_NPZ}")
    for k in sorted(captured):
        v = captured[k]
        print(f"  {k:14s} shape={v.shape}  max_abs={np.abs(v).max():.4f}")


if __name__ == "__main__":
    main()
