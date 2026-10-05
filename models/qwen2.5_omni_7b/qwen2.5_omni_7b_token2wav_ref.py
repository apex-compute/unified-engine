#!/usr/bin/env python3
"""CPU reference + fixed fixtures for the FPGA Token2Wav port (no hardware).

Loads only the Token2Wav checkpoint tensors, runs the FP32 reference with a
FIXED noise draw and FIXED codec IDs, and saves everything the FPGA stages are
compared against: the DiT input noise, the reference mel and the reference
waveform. The sampled FPGA Talker output is not a correctness oracle, so tests
use these fixed codes.

    python qwen2.5_omni_7b_token2wav_ref.py --codes 40 --out t2w_ref_40.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

import torch


def _load_sibling(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def model_dir() -> str:
    cfg = json.load(open(os.path.join(SCRIPT_DIR, "qwen2.5_omni_7b_config.json")))
    return os.path.join(SCRIPT_DIR, cfg["paths"]["hf_model_dir"])


def load_token2wav(speaker: str = "Chelsie"):
    """FP32 Token2Wav model plus the speaker's conditioning tensors."""
    from transformers import Qwen2_5OmniConfig, Qwen2_5OmniToken2WavModel
    talker_mod = _load_sibling("qwen2_5_omni_7b_talker", "qwen2.5_omni_7b_talker.py")
    md = model_dir()
    cfg = Qwen2_5OmniConfig.from_pretrained(md)
    t2w = Qwen2_5OmniToken2WavModel(cfg.token2wav_config)
    t2w.load_state_dict(talker_mod._load_submodule_state(md, "token2wav"), strict=True)
    t2w.to(dtype=torch.float32).eval()
    spk = torch.load(os.path.join(md, "spk_dict.pt"), map_location="cpu",
                     weights_only=False)[speaker]
    return t2w, spk["cond"].float(), spk["ref_mel"].float()


def fixed_codes(n: int, seed: int = 1234) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 8192, (1, n), generator=g, dtype=torch.long)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", type=int, default=40)
    ap.add_argument("--out", default=None)
    ap.add_argument("--speaker", default="Chelsie")
    args = ap.parse_args()

    t2w, cond, ref_mel = load_token2wav(args.speaker)
    dit = t2w.code2wav_dit_model
    print("dit config: block_size", dit.block_size, "heads", dit.num_attention_heads,
          "repeats", dit.repeats, "layers", dit.layers)
    print("look_ahead", [i for i, b in enumerate(dit.transformer_blocks) if b.look_ahead_block],
          "look_backward", [i for i, b in enumerate(dit.transformer_blocks) if b.look_backward_block])
    print("cond", tuple(cond.shape), "ref_mel", tuple(ref_mel.shape))
    print("rope", dit.config.rope_parameters)

    codes = fixed_codes(args.codes)
    frames = args.codes * dit.repeats
    torch.manual_seed(0)
    noise = torch.randn([1, frames, dit.mel_dim])

    state = torch.get_rng_state()
    torch.manual_seed(0)
    t0 = time.perf_counter()
    mel = dit.sample(cond, ref_mel, codes)
    dit_s = time.perf_counter() - t0
    # sample() draws its own noise from the global RNG right after the seed.
    torch.manual_seed(0)
    noise_used = torch.randn([1, frames, dit.mel_dim])
    t0 = time.perf_counter()
    wav = t2w.code2wav_bigvgan_model(mel)
    big_s = time.perf_counter() - t0
    torch.set_rng_state(state)
    print(f"reference: DiT {dit_s:.2f}s BigVGAN {big_s:.2f}s mel {tuple(mel.shape)} "
          f"wav {tuple(wav.shape)}")
    if args.out:
        torch.save({"codes": codes, "noise": noise_used, "mel": mel, "wav": wav,
                    "cond": cond, "ref_mel": ref_mel}, args.out)
        print("saved", args.out)


if __name__ == "__main__":
    main()
