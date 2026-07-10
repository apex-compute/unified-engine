"""Isolate pure quantization loss: run the pi0.5 torch reference twice with
IDENTICAL inputs+noise, once in bf16 and once in IF4, and report the SNR
between the two action outputs (and per-layer prefix K/V). This is the
'quantization floor' -- the best the FPGA (which is IF4) could ever match a
bf16/full-precision model. Runs entirely on GPU, no FPGA. Fast (weights cached).

    python models/pi05_libero/compare_bf16_if4.py [--device cuda] [--sample 0]
"""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
import pi05_torch_ref as _ref


def build_inputs(device, compute_dtype, sample):
    sample_dir = Path.home() / "apex-compute-ML" / "simple-llm" / "src" / "models" / "pi0_5" / "sample_data"
    img = np.load(sample_dir / f"sample_{sample}_image.npy").astype(np.float32) / 127.5 - 1.0
    wrist = np.load(sample_dir / f"sample_{sample}_wrist_image.npy").astype(np.float32) / 127.5 - 1.0
    img_t = torch.from_numpy(img).to(device, compute_dtype)
    wrist_t = torch.from_numpy(wrist).to(device, compute_dtype)
    pad_t = torch.zeros_like(img_t)
    images = torch.stack([img_t, wrist_t, pad_t], dim=0).permute(0, 3, 1, 2).float()
    images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
    images = images.permute(0, 2, 3, 1).to(compute_dtype)
    meta = json.loads((sample_dir / "meta.json").read_text())
    state = torch.tensor([meta["state_example"]], device=device, dtype=torch.float32)
    # SAME prompt tokens for both runs (fixed seed) so the only variable is quant.
    g = torch.Generator(device=device).manual_seed(1234)
    prompt_tokens = torch.randint(0, 257152, (1, 16), generator=g, device=device)
    noise = torch.zeros(1, 10, 32, device=device, dtype=torch.float32)
    noise[..., :7] = torch.from_numpy(
        np.random.RandomState(0).randn(10, 7).astype(np.float32)).to(device)
    return images, prompt_tokens, state, noise


def snr(a, b):
    return 20 * torch.log10(b.norm() / (a - b).norm().clamp_min(1e-12)).item()


def run(device, quant, sample):
    compute_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "if4": torch.bfloat16}[quant]
    W = _ref.Weights(device=device, dtype=compute_dtype, quant=quant)
    images, prompt, state, noise = build_inputs(device, compute_dtype, sample)
    with torch.no_grad():
        actions, kv = _ref.run_pi05(images, prompt, state, W, noise=noise, return_kv=True)
    return actions[..., :7].squeeze(0).float().cpu(), kv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    print(f"[compare] device={args.device} sample={args.sample}")
    print("[compare] running bf16 reference...")
    a_bf16, kv_bf16 = run(args.device, "bf16", args.sample)
    print("[compare] running if4 reference...")
    a_if4, kv_if4 = run(args.device, "if4", args.sample)

    torch.set_printoptions(precision=4, sci_mode=False)
    print("\n=== bf16 vs IF4 action SNR (pure quantization loss, no FPGA) ===")
    print(f"  overall: SNR={snr(a_if4, a_bf16):.2f}dB  MSE={((a_if4-a_bf16)**2).mean().item():.6f}")
    for d in range(7):
        print(f"  dim{d}: SNR={snr(a_if4[:, d], a_bf16[:, d]):6.2f}dB")

    print("\n=== bf16 vs IF4 prefix K/V SNR (per layer) ===")
    k_bf, v_bf = kv_bf16
    k_if, v_if = kv_if4
    for L in range(len(k_bf)):
        rk = k_bf[L].squeeze(0).squeeze(1).float().cpu()
        fk = k_if[L].squeeze(0).squeeze(1).float().cpu()
        rv = v_bf[L].squeeze(0).squeeze(1).float().cpu()
        fv = v_if[L].squeeze(0).squeeze(1).float().cpu()
        print(f"  layer{L:2d}: K_SNR={snr(fk, rk):6.2f}dB  V_SNR={snr(fv, rv):6.2f}dB")

    print("\nbf16 actions (10,7):")
    print(a_bf16)
    print("if4 actions (10,7):")
    print(a_if4)


if __name__ == "__main__":
    main()
