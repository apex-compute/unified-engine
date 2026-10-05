#!/usr/bin/env python3
"""Hardware check of one FPGA DiT evaluation against the FP32 CPU reference.

    python qwen2.5_omni_7b_token2wav_fpga_test.py --dev xdma0 --layers 1 --codes 40
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
def ref_hidden(dit, noise, t, cond, ref_mel, codes, nlayers):
    T = noise.shape[1]
    ts = torch.tensor([t], dtype=torch.float32)
    temb = dit.time_embed(ts)
    ce = dit.text_embed(codes, drop_code=False)
    ceu = dit.text_embed(codes, drop_code=True)
    spk = cond.unsqueeze(1).repeat(1, T, 1)
    x = dit.input_embed(noise, spk, ref_mel, ce, drop_audio_cond=False,
                        code_embed_uncond=ceu, apply_cfg=True)
    pos = torch.arange(T)[None, :]
    pe = dit.rotary_embed(x, pos)
    bd = dit._create_block_diff(x)
    for blk in dit.transformer_blocks[:nlayers]:
        x = blk(x, temb, position_embeddings=pe, block_diff=bd)
    return x


@torch.no_grad()
def debug_layer0(run, dit, noise, t, cond, ref_mel, codes, T):
    Tp = run.Tp
    ts = torch.tensor([t], dtype=torch.float32)
    temb = dit.time_embed(ts)
    ce = dit.text_embed(codes, drop_code=False)
    ceu = dit.text_embed(codes, drop_code=True)
    spk = cond.unsqueeze(1).repeat(1, T, 1)
    x = dit.input_embed(noise, spk, ref_mel, ce, drop_audio_cond=False,
                        code_embed_uncond=ceu, apply_cfg=True)
    blk = dit.transformer_blocks[0]
    norm, g_msa, sh_m, sc_m, g_m = blk.attn_norm(x, emb=temb)
    def hw(addr, width, rows=None):
        r = run._read(addr, run.rows * width).float().reshape(2, Tp, width)[:, :T]
        return r
    q = blk.attn.to_q(norm); k = blk.attn.to_k(norm); v = blk.attn.to_v(norm)
    print(f"  [dbg] Q  {snr_db(q, hw(run.Q, 1024)):.1f} dB  K {snr_db(k, hw(run.K, 1024)):.1f} dB  V {snr_db(v, hw(run.V, 1024)):.1f} dB")
    pe = dit.rotary_embed(x, torch.arange(T)[None, :])
    bd = dit._create_block_diff(x)
    mask = (bd >= -float(blk.look_backward_block)) & (bd <= float(blk.look_ahead_block))
    attn_out = blk.attn(norm, position_embeddings=pe, attention_mask=mask)
    # pre-to_out attention (merge heads) via the module pieces
    B = 2
    qh = q.view(B, T, 16, 64).transpose(1, 2); kh = k.view(B, T, 16, 64).transpose(1, 2)
    vh = v.view(B, T, 16, 64).transpose(1, 2)
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import apply_rotary_pos_emb
    cos, sin = pe
    qh = qh.clone(); kh = kh.clone()
    qh[:, :1], kh[:, :1] = apply_rotary_pos_emb(qh[:, :1], kh[:, :1], cos, sin)
    # head-major rope'd Q/K from hw: plane [h][b][Tp][64]
    qhw = run._read(run.QH, run.rows * 1024).float().reshape(16, 2, Tp, 64)[:, :, :T].permute(1, 0, 2, 3)
    khw = run._read(run.KH, run.rows * 1024).float().reshape(16, 2, Tp, 64)[:, :, :T].permute(1, 0, 2, 3)
    print(f"  [dbg] Qrope head0 {snr_db(qh[:, :1], qhw[:, :1]):.1f} dB  Krope head0 {snr_db(kh[:, :1], khw[:, :1]):.1f} dB  Q other {snr_db(qh[:, 1:], qhw[:, 1:]):.1f} dB")
    att = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
    att_tm = att.transpose(1, 2).reshape(B, T, 1024)
    print(f"  [dbg] attention (merged) {snr_db(att_tm, hw(run.ATT, 1024)):.1f} dB")
    print(f"  [dbg] attn out proj (Y before ff overwrote: skip)")
    x1 = x + g_msa.unsqueeze(1) * attn_out
    n2 = blk.ff_norm(x1) * (1 + sc_m[:, None]) + sh_m[:, None]
    print(f"  [dbg] LN2 out (XN) {snr_db(n2, hw(run.XN, 1024)):.1f} dB")
    h1 = torch.nn.functional.gelu(blk.ff.ff[0](n2), approximate="tanh")
    print(f"  [dbg] gelu hidden (H1) {snr_db(h1, hw(run.H1, 2048)):.1f} dB")
    y = blk.ff.ff[3](h1)
    print(f"  [dbg] ff out (Y) {snr_db(y, hw(run.Y, 1024)):.1f} dB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="xdma0")
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--codes", type=int, default=40)
    ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--sample", default=None,
                    help="fixture .pt: run the full 36-evaluation sampler and compare the mel")
    args = ap.parse_args()

    t2w, cond, ref_mel = ref_mod.load_token2wav()
    dit = t2w.code2wav_dit_model
    codes = ref_mod.fixed_codes(args.codes)
    fixture = torch.load(args.sample) if args.sample else None
    if fixture is not None:
        codes = fixture["codes"]
        args.codes = codes.shape[1]
    T = args.codes * dit.repeats
    torch.manual_seed(5)
    noise = torch.randn(1, T, 80)

    print("--- reference ---")
    hid = ref_hidden(dit, noise, args.t, cond, ref_mel, codes, args.layers)
    full = dit(hidden_states=noise, condition_vector=ref_mel, speaker_embedding=
               cond.unsqueeze(1).repeat(1, T, 1), quantized_code=codes,
               time_step=torch.tensor(args.t), apply_cfg=True)
    print(f"  ref hidden {tuple(hid.shape)}  ref out {tuple(full.shape)}")

    print(f"--- hardware: {args.dev} ---")
    user_dma_core.set_dma_device(args.dev)
    user_dma_core.configure_clock_from_hardware()
    cores = cores_mod.Cores(args.cores)
    print(f"  FPGA build 0x{cores.engines[0].user_read_reg32(user_dma_core.UE_FPGA_VERSION_ADDR):08x}"
          f", {args.cores} engine(s)")

    run = fpga.DiTFpga(cores, dit, frames=T, layers=args.layers)
    run.set_static(cond, ref_mel, codes)
    run.compile()
    run.run(noise[0], args.t)                      # warm-up, also loads caches
    t0 = time.perf_counter()
    pred = run.run(noise[0], args.t)
    dt = time.perf_counter() - t0
    x_hw = run._read(run.X, run.rows * fpga.HID).float().reshape(2, run.Tp, fpga.HID)[:, :T]
    print(f"  one evaluation: {dt:.2f}s")
    print(f"  hidden after {args.layers} layer(s): SNR {snr_db(hid, x_hw):.1f} dB "
          f"(guided {snr_db(hid[0], x_hw[0]):.1f}, null {snr_db(hid[1], x_hw[1]):.1f})")
    if args.layers >= 1:
        debug_layer0(run, dit, noise, args.t, cond, ref_mel, codes, T)
    if fixture is not None:
        sample_check(run, t2w, fixture, T)
    if args.layers == fpga.LAYERS:
        print(f"  final mel prediction: SNR {snr_db(full, pred):.1f} dB "
              f"(guided {snr_db(full[0], pred[0]):.1f}, null {snr_db(full[1], pred[1]):.1f})")


def sample_check(run, t2w, fixture, T):
    print("--- full sampler (36 FPGA DiT evaluations) ---")
    t0 = time.perf_counter()
    mel = run.sample(fixture["noise"][0], progress=False)
    dt = time.perf_counter() - t0
    ref_mel_out = fixture["mel"][0]
    print(f"  sampler wall {dt:.1f}s ({dt / 36:.2f}s/eval)")
    print(f"  mel vs CPU reference: SNR {snr_db(ref_mel_out, mel):.1f} dB")
    wav_hw = t2w.code2wav_bigvgan_model(mel.unsqueeze(0))
    wav_ref = fixture["wav"]
    print(f"  waveform (FPGA mel -> CPU BigVGAN) vs reference: SNR {snr_db(wav_ref, wav_hw):.1f} dB, "
          f"corr {torch.corrcoef(torch.stack([wav_ref.flatten(), wav_hw.flatten()]))[0, 1]:.4f}")
    return mel, wav_hw


if __name__ == "__main__":
    main()
