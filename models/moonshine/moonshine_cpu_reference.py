#!/usr/bin/env python3
"""
Moonshine-base CPU/CUDA reference (moonshine-ai/moonshine-base). NO HARDWARE.

A self-contained port of HF `MoonshineForConditionalGeneration` written as the
flat matmul / conv / eltwise op sequence the accelerator path (moonshine_test.py)
mirrors, so each stage has a CPU oracle. Loads the raw HF safetensors state dict
directly; `transformers` is only needed for `--validate`.

Also owns the host-side plumbing shared with moonshine_test.py: HF weight
download into moonshine_bin/, tokenizer, audio load/resample, golden dump.

Usage (repo root):
  python models/moonshine/moonshine_cpu_reference.py                  # apex.wav, cuda if available
  python models/moonshine/moonshine_cpu_reference.py --device cpu --audio x.wav
  python models/moonshine/moonshine_cpu_reference.py --validate       # exit 0 iff tokens == HF
  python models/moonshine/moonshine_cpu_reference.py --dump-golden    # moonshine_bin/cpu_golden.npz

Architecture (HF shapes, unpadded):
  conv stem   : PCM[N] -> conv1(k127,s64,no bias) -> tanh -> GroupNorm(1) ->
                conv2(k7,s3) -> gelu -> conv3(k3,s2) -> gelu -> [T, 416]
  encoder x8  : pre-LN (no bias) -> MHA (8 heads, head_dim 52, partial RoPE 32,
                interleaved) -> residual -> pre-LN -> fc1 gelu fc2 -> residual
  final LN
  decoder x8  : pre-LN -> causal self-attn (+RoPE) -> residual
                -> pre-LN -> cross-attn over encoder -> residual
                -> pre-LN -> gated MLP (x, gate = chunk(fc1); silu(gate)*x) -> residual
  final LN -> proj_out (tied to embed_tokens) -> logits[32768]

Accelerator obligations (see moonshine_config.json hw_notes): hidden 416->448,
head_dim 52->64, fixed encoder frame count and fixed 194-step decode.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


with open(os.path.join(SCRIPT_DIR, "moonshine_config.json")) as _f:
    CFG = json.load(_f)


def _cfg():
    return CFG


# ---------------------------------------------------------------------------
# primitive ops (each maps to one accelerator core)
# ---------------------------------------------------------------------------

def layer_norm(x, w, eps):
    # bias-free LayerNorm (HF nn.LayerNorm(bias=False))
    return F.layer_norm(x, (x.shape[-1],), weight=w, bias=None, eps=eps)


def linear(x, w, b=None):
    return F.linear(x, w, b)


def rope_tables(T, rotary_dim, theta, device, dtype):
    """cos/sin for interleaved partial RoPE. Returns [T, rotary_dim] with each
    frequency repeated twice (pairs (2i, 2i+1) share a frequency)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim))
    pos = torch.arange(T, device=device).float()
    freqs = torch.outer(pos, inv_freq)                      # [T, rotary_dim/2]
    cos = freqs.cos().repeat_interleave(2, dim=-1)          # [T, rotary_dim]
    sin = freqs.sin().repeat_interleave(2, dim=-1)
    return cos.to(dtype), sin.to(dtype)


def _rotate_half_interleaved(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x, cos, sin):
    """x: [B, H, T, D]; cos/sin: [T, rotary_dim]. Rotates first rotary_dim lanes."""
    rd = cos.shape[-1]
    x_rot, x_pass = x[..., :rd], x[..., rd:]
    cos = cos[None, None]
    sin = sin[None, None]
    x_rot = x_rot * cos + _rotate_half_interleaved(x_rot) * sin
    return torch.cat([x_rot, x_pass], dim=-1)


def attention(q, k, v, scaling, causal=False, mask=None):
    """q: [B,H,Tq,D], k/v: [B,H,Tk,D]. Eager softmax attention (HF eager path)."""
    scores = torch.matmul(q, k.transpose(-1, -2)) * scaling
    if causal:
        Tq, Tk = q.shape[-2], k.shape[-2]
        cm = torch.ones(Tq, Tk, dtype=torch.bool, device=q.device).tril(diagonal=Tk - Tq)
        scores = scores.masked_fill(~cm, float("-inf"))
    if mask is not None:
        scores = scores + mask
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(probs, v)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class MoonshineReference:
    def __init__(self, state_dict, cfg=None, device="cpu", dtype=torch.float32):
        self.cfg = cfg or _cfg()
        enc, dec = self.cfg["encoder"], self.cfg["decoder"]
        self.device = torch.device(device)
        self.dtype = dtype
        self.H = enc["num_heads"]
        self.D = enc["head_dim"]
        self.scaling = self.D ** -0.5
        self.rotary_dim = enc["rotary_dim"]
        self.theta = enc["rope_theta"]
        self.eps = enc["norm_eps"]
        self.n_enc = enc["num_layers"]
        self.n_dec = dec["num_layers"]
        self.bos = dec["decoder_start_token_id"]
        self.eos = dec["eos_token_id"]
        self.w = {k: v.to(self.device, dtype) for k, v in state_dict.items()}
        if "proj_out.weight" not in self.w:  # tied
            self.w["proj_out.weight"] = self.w["model.decoder.embed_tokens.weight"]

    # -- helpers -----------------------------------------------------------
    def p(self, name):
        return self.w[name]

    def _split_heads(self, x):  # [B,T,H*D] -> [B,H,T,D]
        B, T, _ = x.shape
        return x.view(B, T, self.H, self.D).transpose(1, 2)

    def _merge_heads(self, x):  # [B,H,T,D] -> [B,T,H*D]
        B, H, T, D = x.shape
        return x.transpose(1, 2).reshape(B, T, H * D)

    # -- conv stem ---------------------------------------------------------
    def conv_stem(self, pcm):
        """pcm: [B, N] float32 16kHz. Returns [B, T, 416]."""
        p = "model.encoder."
        x = pcm.to(self.device, self.dtype).unsqueeze(1)                        # [B,1,N]
        x = torch.tanh(F.conv1d(x, self.p(p + "conv1.weight"), None, stride=64))
        x = F.group_norm(x, 1, self.p(p + "groupnorm.weight"), self.p(p + "groupnorm.bias"), eps=1e-5)
        x = F.gelu(F.conv1d(x, self.p(p + "conv2.weight"), self.p(p + "conv2.bias"), stride=3))
        x = F.gelu(F.conv1d(x, self.p(p + "conv3.weight"), self.p(p + "conv3.bias"), stride=2))
        return x.permute(0, 2, 1).contiguous()                                  # [B,T,C]

    # -- attention block (shared by enc self / dec self / dec cross) --------
    def attn(self, prefix, x, kv_src=None, cos=None, sin=None, causal=False, cache=None):
        """prefix: '...self_attn.' or '...encoder_attn.'
        kv_src: encoder states for cross-attn (K/V from kv_src, no RoPE).
        cache: dict with 'k','v' [B,H,Tpast,D] for incremental decode (self-attn),
               or precomputed cross K/V (cross-attn)."""
        q = self._split_heads(linear(x, self.p(prefix + "q_proj.weight")))
        if kv_src is not None:                      # cross-attention
            if cache is not None and "k" in cache:
                k, v = cache["k"], cache["v"]
            else:
                k = self._split_heads(linear(kv_src, self.p(prefix + "k_proj.weight")))
                v = self._split_heads(linear(kv_src, self.p(prefix + "v_proj.weight")))
                if cache is not None:
                    cache["k"], cache["v"] = k, v
        else:                                       # self-attention
            k = self._split_heads(linear(x, self.p(prefix + "k_proj.weight")))
            v = self._split_heads(linear(x, self.p(prefix + "v_proj.weight")))
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
            if cache is not None:
                if "k" in cache:
                    k = torch.cat([cache["k"], k], dim=2)
                    v = torch.cat([cache["v"], v], dim=2)
                cache["k"], cache["v"] = k, v
        o = attention(q, k, v, self.scaling, causal=causal)
        return linear(self._merge_heads(o), self.p(prefix + "o_proj.weight"))

    # -- encoder -----------------------------------------------------------
    def encoder_layer(self, i, x, cos, sin):
        p = f"model.encoder.layers.{i}."
        h = layer_norm(x, self.p(p + "input_layernorm.weight"), self.eps)
        x = x + self.attn(p + "self_attn.", h, cos=cos, sin=sin, causal=False)
        h = layer_norm(x, self.p(p + "post_attention_layernorm.weight"), self.eps)
        h = linear(F.gelu(linear(h, self.p(p + "mlp.fc1.weight"), self.p(p + "mlp.fc1.bias"))),
                   self.p(p + "mlp.fc2.weight"), self.p(p + "mlp.fc2.bias"))
        return x + h

    @torch.no_grad()
    def encode(self, pcm, dump=None):
        x = self.conv_stem(pcm)
        if dump is not None:
            dump["conv_stem"] = x
        T = x.shape[1]
        cos, sin = rope_tables(T, self.rotary_dim, self.theta, self.device, self.dtype)
        for i in range(self.n_enc):
            x = self.encoder_layer(i, x, cos, sin)
            if dump is not None:
                dump[f"enc_layer_{i}"] = x
        x = layer_norm(x, self.p("model.encoder.layer_norm.weight"), self.eps)
        if dump is not None:
            dump["encoder_out"] = x
        return x

    # -- decoder -----------------------------------------------------------
    def decoder_layer(self, i, x, enc, cos, sin, self_cache, cross_cache):
        p = f"model.decoder.layers.{i}."
        h = layer_norm(x, self.p(p + "input_layernorm.weight"), self.eps)
        x = x + self.attn(p + "self_attn.", h, cos=cos, sin=sin, causal=True, cache=self_cache)
        h = layer_norm(x, self.p(p + "post_attention_layernorm.weight"), self.eps)
        x = x + self.attn(p + "encoder_attn.", h, kv_src=enc, cache=cross_cache)
        h = layer_norm(x, self.p(p + "final_layernorm.weight"), self.eps)
        h = linear(h, self.p(p + "mlp.fc1.weight"), self.p(p + "mlp.fc1.bias"))
        xx, gate = h.chunk(2, dim=-1)              # NOTE: gate is the SECOND chunk
        h = linear(F.silu(gate) * xx, self.p(p + "mlp.fc2.weight"), self.p(p + "mlp.fc2.bias"))
        return x + h

    @torch.no_grad()
    def decode_step(self, tokens, enc, pos0, self_caches, cross_caches):
        """tokens: [B, t] new token ids at positions pos0..pos0+t-1. Returns logits [B, t, V]."""
        x = F.embedding(tokens.to(self.device), self.p("model.decoder.embed_tokens.weight"))
        T = tokens.shape[1]
        cos, sin = rope_tables(pos0 + T, self.rotary_dim, self.theta, self.device, self.dtype)
        cos, sin = cos[pos0:], sin[pos0:]
        for i in range(self.n_dec):
            x = self.decoder_layer(i, x, enc, cos, sin, self_caches[i], cross_caches[i])
        x = layer_norm(x, self.p("model.decoder.norm.weight"), self.eps)
        return linear(x, self.p("proj_out.weight"))

    @torch.no_grad()
    def generate(self, pcm, max_new_tokens=None, dump=None):
        """Greedy decode. pcm: [B=1, N]. Returns list of token ids (without BOS/EOS)."""
        max_new_tokens = max_new_tokens or self.cfg["defaults"]["max_new_tokens"]
        enc = self.encode(pcm, dump=dump)
        # HF caps generation at ~6.5 tokens/sec of audio when max_length is default.
        self_caches = [dict() for _ in range(self.n_dec)]
        cross_caches = [dict() for _ in range(self.n_dec)]
        tok = torch.tensor([[self.bos]], dtype=torch.long)
        out = []
        for step in range(max_new_tokens):
            logits = self.decode_step(tok, enc, step, self_caches, cross_caches)
            nxt = int(logits[0, -1].argmax())
            if dump is not None and step == 0:
                dump["logits_step0"] = logits[0, -1]
            if nxt == self.eos:
                break
            out.append(nxt)
            tok = torch.tensor([[nxt]], dtype=torch.long)
        return out


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------

def load_state_dict(path):
    from safetensors.torch import load_file
    return load_file(path)


HF_REPO = CFG["paths"]["hf_model_repo"]
BIN_DIR = os.path.join(SCRIPT_DIR, CFG["paths"]["weights_dir"])
WEIGHT_FILES = ["model.safetensors", "config.json", "tokenizer.json", "generation_config.json"]
SAMPLE_RATE = CFG["model"]["sample_rate"]
DEFAULT_AUDIO = os.path.normpath(os.path.join(SCRIPT_DIR, CFG["defaults"]["default_audio"]))


# ---------------------------------------------------------------------------
# weights / tokenizer / audio
# ---------------------------------------------------------------------------

def ensure_weights():
    os.makedirs(BIN_DIR, exist_ok=True)
    missing = [f for f in WEIGHT_FILES if not os.path.exists(os.path.join(BIN_DIR, f))]
    if missing:
        print(f"Downloading {missing} from {HF_REPO} to {BIN_DIR} ...")
        from huggingface_hub import hf_hub_download
        for f in missing:
            hf_hub_download(repo_id=HF_REPO, filename=f, local_dir=BIN_DIR)
    else:
        print(f"Using cached weights in {BIN_DIR}")
    return {f: os.path.join(BIN_DIR, f) for f in WEIGHT_FILES}


def load_tokenizer(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def load_audio(path, max_seconds):
    """Returns float32 mono 16 kHz numpy [N]."""
    import soundfile as sf
    pcm, sr = sf.read(path, dtype="float32", always_2d=True)
    pcm = pcm.mean(axis=1)
    if sr != SAMPLE_RATE:
        try:
            import torchaudio.functional as taf
            pcm = taf.resample(torch.from_numpy(pcm), sr, SAMPLE_RATE).numpy()
        except ImportError:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr, SAMPLE_RATE)
            pcm = resample_poly(pcm, SAMPLE_RATE // g, sr // g).astype(np.float32)
    n_max = int(max_seconds * SAMPLE_RATE)
    if len(pcm) > n_max:
        print(f"[warn] audio {len(pcm)/SAMPLE_RATE:.1f}s truncated to {max_seconds}s")
        pcm = pcm[:n_max]
    return pcm.astype(np.float32)


# ---------------------------------------------------------------------------
# validation against HF
# ---------------------------------------------------------------------------

def snr_db(ref, got):
    ref = ref.float().flatten()
    got = got.float().flatten()
    err = (ref - got).norm()
    return float("inf") if err == 0 else 20 * torch.log10(ref.norm() / err).item()


def validate_vs_hf(ref_model, pcm, device):
    from transformers import MoonshineForConditionalGeneration
    hf = MoonshineForConditionalGeneration.from_pretrained(BIN_DIR, dtype=torch.float32).to(device).eval()
    hf.config._attn_implementation = "eager"
    x = torch.from_numpy(pcm)[None].to(device)
    with torch.no_grad():
        hf_enc = hf.model.encoder(x).last_hidden_state
        hf_ids = hf.generate(x, max_new_tokens=CFG["defaults"]["max_new_tokens"])[0].tolist()
    dump = {}
    our_enc = ref_model.encode(x, dump=dump)
    print(f"[validate] encoder_out SNR vs HF: {snr_db(hf_enc, our_enc):.1f} dB "
          f"(cos {torch.nn.functional.cosine_similarity(hf_enc.flatten(), our_enc.flatten(), dim=0):.6f})")
    our_ids = ref_model.generate(x)
    hf_ids = [t for t in hf_ids if t not in (ref_model.bos, ref_model.eos)]
    print(f"[validate] token match: {our_ids == hf_ids}  (ours {len(our_ids)} / hf {len(hf_ids)})")
    if our_ids != hf_ids:
        print(f"  ours: {our_ids}\n  hf  : {hf_ids}")
    return our_ids == hf_ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Moonshine-base ASR -- CPU/CUDA reference (no hardware)")
    ap.add_argument("--audio", default=DEFAULT_AUDIO)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-new-tokens", type=int, default=CFG["defaults"]["max_new_tokens"])
    ap.add_argument("--max-seconds", type=float, default=CFG["defaults"]["max_audio_seconds"])
    ap.add_argument("--validate", action="store_true", help="compare against HF transformers")
    ap.add_argument("--dump-golden", action="store_true", help="write moonshine_bin/cpu_golden.npz")
    args = ap.parse_args()

    device = args.device

    files = ensure_weights()
    tok = load_tokenizer(files["tokenizer.json"])
    pcm = load_audio(args.audio, args.max_seconds)
    print(f"audio: {args.audio}  {len(pcm)/SAMPLE_RATE:.2f}s @ {SAMPLE_RATE} Hz")

    t0 = time.time()
    model = MoonshineReference(load_state_dict(files["model.safetensors"]), CFG, device=device)
    print(f"weights loaded ({len(model.w)} tensors) in {time.time()-t0:.1f}s on {device}")

    dump = {} if args.dump_golden else None
    t0 = time.time()
    ids = model.generate(torch.from_numpy(pcm)[None], max_new_tokens=args.max_new_tokens, dump=dump)
    dt = time.time() - t0
    text = tok.decode(ids, skip_special_tokens=True)
    print(f"\n>>> {text}\n({len(ids)} tokens, {dt:.2f}s, RTF {dt/(len(pcm)/SAMPLE_RATE):.3f})")

    if args.dump_golden:
        out = os.path.join(BIN_DIR, "cpu_golden.npz")
        np.savez(out, **{k: v.float().cpu().numpy() for k, v in dump.items()},
                 token_ids=np.array(ids, dtype=np.int64), pcm=pcm)
        print(f"golden written: {out} ({list(dump.keys())})")

    if args.validate:
        ok = validate_vs_hf(model, pcm, device)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
