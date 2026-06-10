"""gemma3_real.py — REAL gemma3-1b on hardware: prefill + DECODE (generation) vs HF.

Loads google/gemma-3-1b-it (IF4 projections), runs LAYERS decoder layers, and:
  - PREFILL: processes the prompt, populating per-layer KV caches (growing buffers).
  - DECODE : generates GEN tokens one at a time, each appending K/V to the cache via
             register-addressed DMA and attending the cache (decode_attention).
Everything is preconfigured core funcs / generic IRHarness helpers — nothing gemma
specific in the lib. Compared against HuggingFace greedy generation.

Run:
    UE_REPO=/home/rohit/unified-engine python examples/gemma3_real.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("UE_REPO", "/home/rohit/unified-engine")

import math
import torch

from ir_harness import IRHarness
from ir_harness import Tensor, pad64

REPO = "google/gemma-3-1b-it"
SEQ = 16          # prompt length in tokens (set from the actual prompt at runtime)
LAYERS = 26
GEN = 4           # tokens to generate
MAXPOS = 512      # KV-cache / rope-table capacity = max context (prompt+gen). Attention
                  # is bucketized in 64-wide chunks; num_buckets is derived automatically
                  # (prefill ceil(SEQ*group/64), decode ceil(MAXPOS/64)) so any SEQ<=MAXPOS works.


class Gemma3Real(IRHarness):
    HD = 256; GRP = 4; NQ = 4; H = 1152; I = 6912; EPS = 1e-6
    SW_PATTERN = 6
    @property
    def QS(self): return SEQ * self.GRP

    def prep(self):
        sd = self.hf_sd
        def C(t, n): return self.const(t.float(), n)
        proj = {}
        for li in range(LAYERS):
            p = f"model.layers.{li}"
            for s, f in [("q","self_attn.q_proj"),("k","self_attn.k_proj"),("v","self_attn.v_proj"),
                         ("o","self_attn.o_proj"),("gate","mlp.gate_proj"),("up","mlp.up_proj"),("down","mlp.down_proj")]:
                proj[f"l{li}.{s}"] = sd[f"{p}.{f}.weight"].float()
        proj["lm_head"] = sd["model.embed_tokens.weight"].float()   # tied embedding = lm_head [vocab, H]
        self.scan_weights(proj)
        for name in proj: self.lower_precision(name, "if4")

        self.load_weights()
        
        self.lm_head = self.W["lm_head"]
        self.N = {}
        for li in range(LAYERS):
            p = f"model.layers.{li}"
            for s, f in [("input","input_layernorm"),("q_norm","self_attn.q_norm"),("k_norm","self_attn.k_norm"),
                         ("post_attn","post_attention_layernorm"),("pre_ffn","pre_feedforward_layernorm"),("post_ffn","post_feedforward_layernorm")]:
                self.N[f"l{li}.{s}"] = C(sd[f"{p}.{f}.weight"] + 1.0, f"l{li}.{s}")
        self.final_norm = C(sd["model.norm.weight"] + 1.0, "final_norm")

        # rope tables over MAXPOS positions (serve prefill slice + decode by position)
        self.rope, self.rope_base = {}, {}
        pos = torch.arange(MAXPOS).float()[:, None]; d = self.HD // 2
        for tag, theta in (("global", 1000000.0), ("local", 10000.0)):
            inv = 1.0 / (theta ** (torch.arange(d).float() / d)); ang = pos * inv[None, :]
            packed = torch.cat([ang.cos(), ang.cos(), -ang.sin(), ang.sin()], -1)
            tbl = self.const(packed, f"rope_{tag}")
            self.rope_base[tag] = tbl.addr
            self.rope[tag] = (Tensor(addr=tbl.addr, shape=(SEQ, self.HD), name=f"cos_{tag}"),
                              Tensor(addr=tbl.addr + self.HD * 2, shape=(SEQ, self.HD), name=f"sin_{tag}"))

        # per-layer KV caches (growing buffers in tensor DRAM)
        self.kc = [self.growing_buffer(f"kc{li}", MAXPOS, self.HD) for li in range(LAYERS)]
        self.vc = [self.growing_buffer(f"vc{li}", MAXPOS, self.HD) for li in range(LAYERS)]

        # prefill attention scaffolding. The flash bucket is 64-wide, so the bias MUST be
        # built at the aligned size (ceil(QS/64)*64), causal, with columns past the real
        # q_seq masked to -inf. Otherwise the padded rows are all -inf -> softmax NaN, and
        # padded cols read uninitialized DRAM. (Golden SEQ=16 hid this since QS==64.)
        A = pad64(self.QS)
        pbias = torch.full((A, A), float("-inf"))
        pbias.masked_fill_(torch.tril(torch.ones(A, A, dtype=torch.bool)), 0.0)
        pbias[:, self.QS:] = float("-inf")
        self.mask = self.const(pbias, "mask")
        self.identity = self.const(torch.eye(64), "identity")
        self.scratch = self.const(torch.zeros(16 * A * self.HD), "scratch")
        self.attn_p = self.const(torch.zeros(A * A), "attn_p")
        # decode scaffolding. Bias spans the full context (MAXPOS) so multi-bucket decode
        # (context > 64) is supported: each step re-DMAs (1, MAXPOS) with [0:pos+1]=0, rest
        # -inf, and the bucket body reads the first bucket_reg*64 entries.
        self.dmask = self.const(torch.zeros(1, MAXPOS), "dmask")       # gemma decode bias: (1, aligned_q)
        self.dx = self.const(torch.zeros(1, self.H), "dx")            # decode input token
        self.dscratch = self.const(torch.zeros(16 * 64 * self.HD), "dscratch")
        self.dattn_p = self.const(torch.zeros(64 * 64), "dattn_p")

        self.x = self.const(self.x_input, "x")                        # HF embedded prompt
        self.seq = self.gpr("seq_len"); self.qseq = self.gpr("q_seq_len"); self.bucket = self.gpr("bucket_idx")

    def _rope_tag(self, li): return "global" if (li + 1) % self.SW_PATTERN == 0 else "local"

    def build(self):
        # ---- PREFILL program (populates KV caches) ----
        with self.program("prefill"):
            self.loop_reg(self.seq)
            x = self.x
            for li in range(LAYERS):
                W = lambda p: self.W[f"l{li}.{p}"]; N = lambda n: self.N[f"l{li}.{n}"]
                rope = self.rope[self._rope_tag(li)]
                r = x
                h = self.chain(("rms_norm", x, N("input"), f"h{li}"))
                q = self.chain(("mm", h, W("q"), f"q{li}")); k = self.chain(("mm", h, W("k"), f"k{li}")); v = self.chain(("mm", h, W("v"), f"v{li}"))
                a = self.attention(q, k, v, q_norm=N("q_norm"), k_norm=N("k_norm"), cos=rope[0], sin=rope[1],
                                   group=self.GRP, head_dim=self.HD, scratch=self.scratch, identity=self.identity,
                                   mask=self.mask, attn_p=self.attn_p, bucket_reg=self.bucket, qseq_reg=self.qseq,
                                   k_cache=self.kc[li], v_cache=self.vc[li])
                a = self.chain(("mm", a, W("o"), f"ao{li}")); a = self.chain(("rms_norm", a, N("post_attn"), f"an{li}"))
                x = self.chain(("add", r, a, f"xa{li}"))
                r = x; h = self.chain(("rms_norm", x, N("pre_ffn"), f"hf{li}"))
                gg = self.chain(("mm", h, W("gate"), f"g{li}", {"act":"gelu"})); uu = self.chain(("mm", h, W("up"), f"u{li}"))
                h = self.chain(("mul", gg, uu, f"m{li}")); h = self.chain(("mm", h, W("down"), f"d{li}"))
                h = self.chain(("rms_norm", h, N("post_ffn"), f"dn{li}")); x = self.chain(("add", r, h, f"x{li}"))
            self.out = self.chain(("rms_norm", x, self.final_norm, "fn"))

        # ---- DECODE program (single token, attends KV cache) ----
        with self.program("decode"):
            self.loop_reg(None)                          # decode norms are M=1, legacy
            x = self.dx
            for li in range(LAYERS):
                W = lambda p: self.W[f"l{li}.{p}"]; N = lambda n: self.N[f"l{li}.{n}"]
                rb = self.rope_base[self._rope_tag(li)]
                r = x
                h = self.chain(("rms_norm", x, N("input"), f"Dh{li}", {"M_reg": None}))
                # decode projections use quantized_matmat_core (qmatmul) — the M=1 decode kernel
                # the reference uses; faster than the general matmat_mul_core legacy path.
                q = self.chain(("qmatmul", h, W("q"), f"Dq{li}")); k = self.chain(("qmatmul", h, W("k"), f"Dk{li}")); v = self.chain(("qmatmul", h, W("v"), f"Dv{li}"))
                a = self.decode_attention(q, k, v, q_norm=N("q_norm"), k_norm=N("k_norm"), rope_base=rb,
                                          pos_reg=self.seq, group=self.GRP, head_dim=self.HD,
                                          k_cache=self.kc[li], v_cache=self.vc[li], scratch=self.dscratch,
                                          identity=self.identity, mask=self.dmask, bucket_reg=self.bucket)
                a = self.chain(("qmatmul", a, W("o"), f"Dao{li}")); a = self.chain(("rms_norm", a, N("post_attn"), f"Dan{li}", {"M_reg": None}))
                x = self.chain(("add", r, a, f"Dxa{li}"))
                r = x; h = self.chain(("rms_norm", x, N("pre_ffn"), f"Dhf{li}", {"M_reg": None}))
                gg = self.chain(("qmatmul", h, W("gate"), f"Dg{li}", {"act":"gelu"})); uu = self.chain(("qmatmul", h, W("up"), f"Du{li}"))
                h = self.chain(("mul", gg, uu, f"Dm{li}")); h = self.chain(("qmatmul", h, W("down"), f"Dd{li}"))
                h = self.chain(("rms_norm", h, N("post_ffn"), f"Ddn{li}", {"M_reg": None})); x = self.chain(("add", r, h, f"Dx{li}"))
            self.dout = self.chain(("rms_norm", x, self.final_norm, "Dfn", {"M_reg": None}))
            # ON-DEVICE lm_head: quantized_matmat_core (qmatmul) — the SAME kernel the reference
            # uses. It updates the hardware argmax register (read via get_arg_max_index), which
            # matmat_mul_core+write_back_disable does NOT. write_back_disable skips the 262k-logit
            # DMA, so token pick needs no host matmul.
            self.chain(("qmatmul", self.dout, self.lm_head, "Dlogits", {"write_back_disable": True}))
            # ON-DEVICE position advance: seq self-increments so re-running decode walks
            # the sequence with no host re-prime (mirrors gemma3_test's trailing add_inc).
            self.advance(self.seq)


DEFAULT_PROMPT = "x+3=5, what is x?"        # same default prompt as gemma3_test.py


def main():
    import argparse
    global SEQ, GEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="prompt text (any length up to MAXPOS-gen tokens)")
    ap.add_argument("--gen", type=int, default=0,
                    help="max tokens to generate (0 = until EOS, capped at MAXPOS-prompt)")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO)
    model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32).eval()
    prompt = args.prompt
    # gemma-3-1b is instruction-tuned: feed the chat template so it answers (and emits
    # <end_of_turn>) instead of free-continuing and looping on a raw prompt.
    chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, tokenize=False)
    ids = tok(chat, return_tensors="pt", add_special_tokens=False).input_ids
    SEQ = ids.shape[1]                                   # SEQ = the real prompt length
    GEN = (MAXPOS - SEQ) if args.gen <= 0 else min(args.gen, MAXPOS - SEQ)   # default: fill context
    eos_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")} - {None}
    print(f"\nPROMPT: {prompt!r}\n  ({SEQ} tokens, generating up to {GEN} until EOS)")
    norm = model.config.text_config.hidden_size ** 0.5 if hasattr(model.config, "text_config") else model.config.hidden_size ** 0.5
    embed = model.model.embed_tokens.weight.float()
    with torch.no_grad():
        Gemma3Real.x_input = model.model.embed_tokens(ids)[0].float()              # HF embeds (prompt)

    Gemma3Real.hf_sd = model.state_dict()
    m = Gemma3Real()

    def emit(logits):
        l = logits.softmax(-1); t = l.argmax().item()
        return t, l[t].item()

    print("\n--- generating ---\n")
    # Hovering throughput line: pin the bottom terminal row via an ANSI scroll region; tokens
    # stream above it, the counter refreshes in place. Only on a real TTY (skip if piped).
    import shutil
    import time
    timer = time.perf_counter()
    dec_t = 0.0; n_dec = 0
    _tty = sys.stdout.isatty()
    def _status(setup=False, teardown=False):
        if not _tty:
            return
        rows = shutil.get_terminal_size().lines
        if setup:
            sys.stdout.write(f"\033[1;{rows-1}r\033[{rows-1};1H"); sys.stdout.flush(); return
        if teardown:
            sys.stdout.write(f"\033[r\033[{rows};1H\033[2K"); sys.stdout.flush(); return
        el = time.perf_counter() - timer
        rate = n_dec / el if el > 0 else 0.0
        sys.stdout.write("\0337" + f"\033[{rows};1H\033[2K")      # save cursor, go to bottom row
        sys.stdout.write(f" decoding… {n_dec} tokens  (pos {SEQ+n_dec}/{MAXPOS})  {el:.1f}s  {rate:.1f} tok/s")
        sys.stdout.write("\0338"); sys.stdout.flush()             # restore cursor
    # Set up the scroll region BEFORE streaming any token, so the prefill's first token
    # streams inside the region too (else the status line clears the row it landed on).
    _status(setup=True)

    # PREFILL: process the prompt, populate KV caches, predict the first token.
    m.execute("prefill", bind={m.seq: SEQ, m.qseq: m.QS, m.bucket: max(1, (m.QS + 63) // 64)})
    last = m.ue.dma_from_accelerator_memory(m.out.addr, (SEQ, m.H)).float()[-1]
    t, _ = emit(last @ embed.T); our = [t]
    if t not in eos_ids:
        print(tok.decode([t]), end="", flush=True)

    # DECODE loop — one token at a time. seq is device-resident: prefill left it at SEQ and
    # the decode program self-advances it (add_inc), so we never re-prime it here — only the
    # per-token bucket count + the new embedding/bias are pushed (the gemma3_test dispatch stub).
    for step in range(GEN - 1):
        if our[-1] in eos_ids:                                # stop on EOS
            break
        pos = SEQ + step
        m.ue.dma_to_accelerator_memory(m.dx.addr, (embed[our[-1]] * norm).view(1, m.H).to(torch.bfloat16))
        dmask = torch.full((1, MAXPOS), -1e36); dmask[0, :pos + 1] = 0.0   # gemma decode bias (full ctx)
        m.ue.dma_to_accelerator_memory(m.dmask.addr, dmask.to(torch.bfloat16))
        _t = time.perf_counter()
        m.execute("decode", bind={m.bucket: max(1, (pos + 1 + 63) // 64)}, banner=False)
        t = m.ue.get_arg_max_index()                             # device argmax (no host matmul)
        dec_t += time.perf_counter() - _t; n_dec += 1
        our.append(t)
        if t in eos_ids:
            break
        print(tok.decode([t]), end="", flush=True)               # stream as a paragraph
        _status()                                                # refresh the hovering counter
    _status(teardown=True)
    print()

    n_dec = max(1, n_dec)
    print(f"\n\ndecode speed: {n_dec/dec_t:.2f} tok/s  ({dec_t/n_dec*1e3:.1f} ms/tok, {n_dec} tokens)")


if __name__ == "__main__":
    main()
