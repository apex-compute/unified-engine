#!/usr/bin/env python3
"""End-to-end FPGA Token2Wav (DiT + BigVGAN) against the CPU reference waveform.

    python qwen2.5_omni_7b_token2wav_fpga_e2e_test.py --dev xdma0 --fixture t2w_fixtures/t2w_ref_40.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

import torch

import user_dma_core


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ref_mod = _load("qwen2_5_omni_7b_token2wav_ref", "qwen2.5_omni_7b_token2wav_ref.py")
fpga = _load("qwen2_5_omni_7b_token2wav_fpga", "qwen2.5_omni_7b_token2wav_fpga.py")
cores_mod = _load("qwen2_5_omni_7b_t2w_cores", "qwen2.5_omni_7b_t2w_cores.py")


def snr_db(ref, got):
    ref = ref.float().reshape(-1)
    got = got.float().reshape(-1)
    err = (ref - got).pow(2).sum().item()
    sig = ref.pow(2).sum().item()
    return float("inf") if err == 0 else 10.0 * math.log10(sig / max(err, 1e-30))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="xdma0")
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--fixture", default="t2w_fixtures/t2w_ref_40.pt")
    ap.add_argument("--out", default="t2w_fixtures/t2w_fpga_e2e.wav")
    args = ap.parse_args()

    t2w, cond, ref_mel = ref_mod.load_token2wav()
    fx = torch.load(args.fixture)
    codes = fx["codes"]

    user_dma_core.set_dma_device(args.dev)
    user_dma_core.configure_clock_from_hardware()
    cores = cores_mod.Cores(args.cores)
    print(f"  FPGA build 0x{cores.engines[0].user_read_reg32(user_dma_core.UE_FPGA_VERSION_ADDR):08x}"
          f", {args.cores} engine(s)")

    setup0 = time.perf_counter()
    pipe = fpga.Token2WavFpga(cores, t2w, codes=codes.shape[1], verbose=True)
    print(f"  setup {time.perf_counter() - setup0:.1f}s")
    t0 = time.perf_counter()
    wav, info = pipe.synthesize(codes, cond, ref_mel, noise=None if fx is None else fx["noise"][0])
    total = time.perf_counter() - t0
    print(f"  FPGA Token2Wav: DiT {info['dit_s']:.1f}s + BigVGAN {info['bigvgan_s']:.1f}s "
          f"= {total:.1f}s for {wav.numel() / 24000:.2f}s of audio")
    import soundfile as sf
    sf.write(args.out, wav.detach().numpy(), 24000)
    if fx is not None:
        print(f"  mel vs CPU reference: SNR {snr_db(fx['mel'][0], info['mel']):.1f} dB")
        wav_ref = fx["wav"]
        print(f"  waveform vs CPU reference: SNR {snr_db(wav_ref, wav):.1f} dB, corr "
              f"{torch.corrcoef(torch.stack([wav_ref.flatten(), wav.flatten()]))[0, 1]:.4f}")
        sf.write(args.out.replace(".wav", "_cpu.wav"), wav_ref.detach().numpy(), 24000)
    print("  wrote", args.out)


if __name__ == "__main__":
    main()
