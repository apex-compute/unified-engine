#!/usr/bin/env python3
"""Hardware check of the FPGA BigVGAN against the FP32 CPU model, stage by stage.

    python qwen2.5_omni_7b_token2wav_bigvgan_test.py --dev xdma0 --fixture t2w_fixtures/t2w_ref_40.pt --upto 1
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


@torch.no_grad()
def cpu_stages(big, mel):
    """conv_pre output and every upsample stage's output, [rows, C]."""
    h = big.conv_pre(big.process_mel_spectrogram(mel.unsqueeze(0)))
    outs = {"pre": h[0].t()}
    n = len(big.ups)
    for i in range(n):
        h = big.ups[i][0](h)
        r = sum(big.resblocks[i * big.num_residual_blocks + j](h)
                for j in range(big.num_residual_blocks)) / big.num_residual_blocks
        h = r
        outs[i] = h[0].t()
    return outs, h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="xdma0")
    ap.add_argument("--fixture", default="t2w_fixtures/t2w_ref_40.pt")
    ap.add_argument("--upto", type=int, default=None,
                    help="stop after this many stages (0 = conv_pre only)")
    ap.add_argument("--cores", type=int, default=8)
    args = ap.parse_args()

    t2w, _cond, _rm = ref_mod.load_token2wav()
    big = t2w.code2wav_bigvgan_model
    fx = torch.load(args.fixture)
    mel = fx["mel"][0]
    T = mel.shape[1]
    cpu, _ = cpu_stages(big, mel)

    user_dma_core.set_dma_device(args.dev)
    user_dma_core.configure_clock_from_hardware()
    cores = cores_mod.Cores(args.cores)
    print(f"  FPGA build 0x{cores.engines[0].user_read_reg32(user_dma_core.UE_FPGA_VERSION_ADDR):08x}"
          f", {args.cores} engine(s)")

    run = fpga.BigVGANFpga(cores, big, frames=T)
    t0 = time.perf_counter()
    run.compile()
    print(f"  compiled in {time.perf_counter() - t0:.1f}s; weights "
          f"{cores.weights.get_params_dram_usage() / 2**20:.0f} MiB, pool "
          f"{cores.pool_used() / 2**20:.0f} MiB")

    run.load_mel(mel)
    keys = ["pre"] + list(range(len(run.strides)))
    start = 0
    total = time.perf_counter()
    for n, key in enumerate(keys):
        end, addr, ci = run.marks[key]
        t0 = time.perf_counter()
        cores.run(run.sequence[start:end])
        dt = time.perf_counter() - t0
        start = end
        rows = run.lens[0] if key == "pre" else run.lens[key + 1]
        C = cpu[key].shape[1]
        Cp = fpga._cpad(C)
        got = cores.read(addr, rows * Cp).float().reshape(rows, Cp)[:, :C]
        print(f"  {'conv_pre' if key == 'pre' else f'stage{key}'}: {dt:.2f}s  "
              f"SNR vs CPU {snr_db(cpu[key], got):.1f} dB")
        if args.upto is not None and n == args.upto:
            return
    t0 = time.perf_counter()
    cores.run(run.sequence[start:])
    print(f"  post: {time.perf_counter() - t0:.2f}s   (BigVGAN total {time.perf_counter() - total:.1f}s)")
    out = cores.read(run.out_addr, run.lens[-1] * 64).float().reshape(-1, 64)[:, 0].clamp(-1, 1)
    wav_ref = fx["wav"]
    print(f"  waveform: SNR {snr_db(wav_ref, out):.1f} dB, "
          f"corr {torch.corrcoef(torch.stack([wav_ref.flatten(), out.flatten()]))[0, 1]:.4f}")


if __name__ == "__main__":
    main()
