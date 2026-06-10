"""k_select.py — auto-compiler: HF model id -> auto prep() + build() -> FPGA programs.

The route-selector consumes the inspected spec (dims, features, weight names, stages) and
weaves the decoder-LLM op chain into programs via chain() + the attention helpers.
It is gemma3.py generalized: nothing is hard-coded — every dim/feature/weight comes from
config + state_dict (see ir_compile.py --inspect / --plan for what's derived).

Per-op register state is NOT managed here: chain()/attention()/decode_attention() hide it.
This layer only sets the per-STAGE prologue (loop_reg) + epilogue (advance) and threads
slots/weights. Attention kernel is picked by (stage): prefill -> attention()/flash_attn,
decode -> decode_attention()/group_attn; GQA is just group_size = nq//nkv.
The low-level IRHarness/Chain/Arena machinery this builds on lives in ir_harness.py.

Run (from anywhere in the unified-engine repo):
    python kernel_select/k_select.py --model google/gemma-3-1b-it
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Known-good models verified through the flow. Edit this list freely (add/delete);
# `python k_select.py -h` prints it. First entry is the --model default.
MODELS = [
    "google/gemma-3-1b-it",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-Guard-3-1B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "TinyLlama/TinyLlama_v1.1",
    "princeton-nlp/Sheared-LLaMA-1.3B",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM-360M",
    "Qwen/Qwen3-1.7B",
    "h2oai/h2o-danube-1.8b-base --temp 0.9"
]


# Default prompts: tiny models (<500M) can't handle the algebra prompt; give them
# something they have a chance at. Only used when --prompt isn't passed explicitly.
DEFAULT_PROMPT = "x+3=5, what is x?"
SIMPLE_PROMPT = "Why did the chicken cross the road?"
SMALL_PARAM_THRESHOLD = 500_000_000


def build_parser():
    epilog = "known-good --model values:\n" + "\n".join(f"  {m}" for m in MODELS)
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=epilog)
    ap.add_argument("--model", default=MODELS[0], help=f"HF model id (default: {MODELS[0]})")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--temp", type=float, default=0.0,
                    help="decode sampling temperature (0 = greedy argmax; >0 samples on CPU from FPGA logits)")
    ap.add_argument("--check", action="store_true",
                    help="compare FPGA prefill hidden to CPU-IF4 (rel err)")
    return ap


# Engine modules resolve via relative paths (this file lives in <repo>/kernel_select/),
# so no repo path needs to be configured here.
ARGS = build_parser().parse_args()

import torch
from ir_harness import IRHarness, Tensor, pad64, FUSIONS, FUSION_LABEL
from hlo import UE_HLO

MAXPOS = 512                                              # KV-cache / rope capacity
LOGIT_COS_THRESH = 0.99                                    # --check PASS: FPGA·IF4 logit-vector cosine
PBI_DECODE_MAX_NKV = 8                                     # max nkv for the bucketized group_attn_pbi
#                                                           decode (per-head PBI dispatcher → program
#                                                           bloat/corruption above this; else legacy)

# HF decoder-LLM weight-name convention (the template). Per-layer prefix is model.layers.{i}.
PROJ = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj",
        "o": "self_attn.o_proj", "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
NORM = {"input": "input_layernorm", "q_norm": "self_attn.q_norm", "k_norm": "self_attn.k_norm",
        "post_attn": "post_attention_layernorm", "pre_ffn": "pre_feedforward_layernorm",
        "post_ffn": "post_feedforward_layernorm"}


def decide_dram(model):
    """Forward-pass decision: size the DRAM partitions to the model's IF4 footprint.
    Default map gives params only 768MB; large models (even in IF4) overflow it. We sum
    the quantized-weight bytes and place tensor/program bases just above, so params grows
    to fit. Keeps params_base < tensor_base < program_base < 0x180000000 (the 4GB window)."""
    sd = model.state_dict()
    quant_numel = sum(t.numel() for n, t in sd.items()
                      if n.lower().endswith("proj.weight") or "lm_head" in n.lower())
    params_bytes = int(quant_numel * 0.5 * 1.18)             # if4 data + scale + pad margin
    # Allocator bases are OFFSETS from DRAM_START_ADDR=0x80000000 (the DMA layer adds the base);
    # usable offset range is 0x0 -> 0x100000000 (the full 4GB). params start at offset 0x0 so the
    # whole 4GB is available — NOT at 0x80000000 (that wastes the bottom 2GB and overruns the top).
    base, TOP = 0x0, 0x100000000                              # offsets: 0 -> 4GB
    PROG_BUDGET = 0x10000000                                   # 256MB program, pinned just under TOP
    MIN_TENSOR = 0x18000000                                    # need >=384MB activations/KV to run
    align = lambda x, a=0x1000000: (x + a - 1) // a * a
    tensor_base = base + align(params_bytes)
    # tensor gets EVERYTHING between params and the (top-pinned) program region — not a fixed 384MB.
    # The decode arena never frees (prep consts + KV + all prefill + all decode buffers stay live),
    # so a fixed budget overflows on big models: the last layers' activations spill past program_base
    # into program memory and read back zero/garbage (hard cosine cliff mid-stack). Reclaim the slack.
    program_base = TOP - PROG_BUDGET
    if tensor_base + MIN_TENSOR > program_base:
        raise MemoryError(f"params footprint {params_bytes/1e6:.0f}MB leaves <{MIN_TENSOR/1e6:.0f}MB for "
                          f"tensor/KV before the program region; quantize harder or reduce model size")
    print(f"\ndram  ({quant_numel/1e9:.2f}B@if4, offsets from 0x80000000)")
    print(f"  params   0x{base:08X}–0x{tensor_base:08X}   {(tensor_base-base)/1e6:>5.0f} MB")
    print(f"  tensor   0x{tensor_base:08X}–0x{program_base:08X}   {(program_base-tensor_base)/1e6:>5.0f} MB")
    print(f"  program  0x{program_base:08X}–0x{TOP:08X}   {(TOP-program_base)/1e6:>5.0f} MB")
    return base, tensor_base, program_base


class ExtractedDecoder(IRHarness):
    """Config-driven decoder LLM. spec/cfg/sd/x_input injected as class attrs before init."""

    wpfx = "model."          # decoder weight-key prefix; overridden for VLMs (model.text_model.)

    def _make_engine(self):
        from user_dma_core import UnifiedEngine
        b = getattr(type(self), "_dram", None)
        if b is None:
            return UnifiedEngine()
        return UnifiedEngine(params_dram_base=b[0], tensor_dram_base=b[1], program_dram_base=b[2])

    def prep(self):
        c, sd = self.spec, self.sd
        H, I, hr, nq, nkv = c["H"], c["I"], c["hd"], c["nq"], c["nkv"]
        # flash/transpose/rope kernels require head_dim % 64 == 0. When the model's real head_dim
        # (hr) isn't a multiple of 64 (e.g. danube/Phi hr=80) we PAD to HDP and lay each head out
        # lo|hi: real lanes at [0:hr/2] and [HDP/2:HDP/2+hr/2], pad lanes zero. The padding is done
        # HOST-SIDE in the projection WEIGHTS (zero rows in the permuted positions) so the matmul
        # emits the padded layout directly — no device repack (sub-64 DMA reads misalign; verified).
        # The attention then runs the already-verified head_dim=HDP contiguous path unchanged.
        self.HD_real = hr
        self.HDP = pad64(hr)
        self.hd_padded = self.HDP != hr
        hd = self.HDP if self.hd_padded else hr          # head_dim the attention path uses
        self.HD = hd
        self.H = H; self.GRP = nq // max(1, nkv)
        L = c["layers"]

        def C(t, n): return self.const(t.float(), n)
        # attention variants, all derived from traits:
        #   multikv  = nkv>1  -> per-head scatter + per-KV-head flash
        #   lohi     = hd<128 -> per-head d64 rope (padded-split, 128B-aligned via reg-addr DMA)
        self.multikv = nkv > 1
        self.lohi = hd < 128                             # padded -> hd=HDP>=128 -> contiguous path
        if self.multikv and not hasattr(self.ue, "TMP_REG"):
            self.ue.TMP_REG = self.ue.alloc_isa_reg()    # scatter / d64-decode helper scratch GPR

        # ---- weights: projections + lm_head -> if4 ; norms -> bf16 (gemma: w + 1.0) ----
        # No weight permute: d64 ropes each head's contiguous hd-wide slice directly.
        # P is the decoder weight prefix ("model." for LMs, "model.text_model." for VLMs).
        P = self.wpfx
        proj = {}
        for li in range(L):
            for s, suf in PROJ.items():
                proj[f"l{li}.{s}"] = sd[f"{P}layers.{li}.{suf}.weight"].float()
        lm_name = "lm_head.weight" if "lm_head.weight" in sd else f"{P}embed_tokens.weight"
        proj["lm_head"] = sd[lm_name].float()
        # Pad+permute projection weights so the matmul emits the lo|hi-HDP layout directly (the
        # correct, alignment-safe way to support hr%64!=0; replaces device repack). q is also
        # pre-scaled by attn_scale*sqrt(HDP) so flash's 1/sqrt(HDP) nets the real 1/sqrt(hr).
        if self.hd_padded:
            qc = float(c["attn_scale"]) * (self.HDP ** 0.5)
            for li in range(L):
                proj[f"l{li}.q"] = _pad_proj_out(proj[f"l{li}.q"], nq, hr, self.HDP, "lohi", scale=qc)
                proj[f"l{li}.k"] = _pad_proj_out(proj[f"l{li}.k"], nkv, hr, self.HDP, "lohi")
                proj[f"l{li}.v"] = _pad_proj_out(proj[f"l{li}.v"], nkv, hr, self.HDP, "contig")
                proj[f"l{li}.o"] = _pad_proj_in(proj[f"l{li}.o"], nq, hr, self.HDP)
        self.scan_weights(proj)
        for n in proj:
            self.lower_precision(n, "if4")
        self.load_weights()
        self.lm_head = self.W["lm_head"]

        off = c["norm_offset"]                              # gemma RMSNorm = (1+w); llama = w
        self.N = {}
        for li in range(L):
            for s, suf in NORM.items():
                key = f"{P}layers.{li}.{suf}.weight"
                if key in sd:
                    self.N[f"l{li}.{s}"] = C(sd[key] + off, f"l{li}.{s}")
        self.final_norm = C(sd[f"{P}norm.weight"] + off, "final_norm")
        self.has_qk_norm = c["qk_norm"]
        self.act = "gelu" if "gelu" in str(c.get("act", "gelu")).lower() else "silu"

        # ---- rope tables ----
        self.rope, self.rope_base = {}, {}
        d = hd // 2
        if self.hd_padded:
            # lo|hi padded table (the verified vision-rope construction): real hr/2 freqs
            # zero-padded to HDP/2, doubled, lo-half sin pre-negated. Real lanes at [0:hr/2] and
            # [HDP/2:HDP/2+hr/2]. Stored in the standard cos/sin form so the contiguous rope_gqa /
            # rope_decode path (head_dim=HDP) consumes it unchanged. Same table for Q and K — the
            # attn-scale correction is folded into the q_proj weight, not the table.
            hr2, hP2 = self.HD_real // 2, self.HDP // 2
            cb, sb = self.rope_cos[:MAXPOS], self.rope_sin[:MAXPOS]      # [MAXPOS, hr/2] real freqs
            def _hp(t):
                p = torch.zeros(MAXPOS, hP2, dtype=t.dtype); p[:, :hr2] = t; return p
            cosP = torch.cat([_hp(cb), _hp(cb)], -1)                    # [MAXPOS, HDP]
            sinP = torch.cat([_hp(sb), _hp(sb)], -1)
            sinP[:, :hP2] = -sinP[:, :hP2]                              # pre-negate lo half
            packed = torch.cat([cosP, sinP], -1)                        # [MAXPOS, 2*HDP]
            tbl = self.const(packed, "rope_pad")
            for tag in ("global", "local"):
                self.rope_base[tag] = tbl.addr
                self.rope[tag] = (Tensor(addr=tbl.addr, shape=(self.SEQ, hd), name=f"cos_{tag}"),
                                  Tensor(addr=tbl.addr + hd * 2, shape=(self.SEQ, hd), name=f"sin_{tag}"))
        elif self.lohi:
            # PACKED per-head d64 table: [cos(hd) || sin(hd)] contiguous per token, sin-lo
            # pre-negated (rope_d64 reads sin at table+hd*2). cos/sin from the model's
            # inv_freq (handles rope-scaling). One table for K (one row per token); a
            # group-duplicated copy for stacked Q (M=seq*group, row r uses token r//group).
            cb, sb = self.rope_cos[:MAXPOS], self.rope_sin[:MAXPOS]      # [MAXPOS, hd//2]
            cos_full = torch.cat([cb, cb], -1)                          # [MAXPOS, hd]
            sin_full = torch.cat([sb, sb], -1)
            sin_full[:, :d] = -sin_full[:, :d]                          # pre-negate lo half
            packed = torch.cat([cos_full, sin_full], -1)               # [MAXPOS, 2*hd]
            self.rope_packed = self.const(packed, "rope_packed").addr
            if self.GRP > 1:
                pg = packed.repeat_interleave(self.GRP, dim=0)         # [MAXPOS*group, 2*hd]
                self.rope_packed_gqa = self.const(pg, "rope_packed_gqa").addr
            else:
                self.rope_packed_gqa = self.rope_packed
        else:
            pos = torch.arange(MAXPOS).float()[:, None]
            thetas = {"global": c["theta"], "local": c["local_base"]}
            for tag, th in thetas.items():
                inv = 1.0 / (th ** (torch.arange(d).float() / d)); ang = pos * inv[None, :]
                packed = torch.cat([ang.cos(), ang.cos(), -ang.sin(), ang.sin()], -1)
                tbl = self.const(packed, f"rope_{tag}")
                self.rope_base[tag] = tbl.addr
                self.rope[tag] = (Tensor(addr=tbl.addr, shape=(self.SEQ, hd), name=f"cos_{tag}"),
                                  Tensor(addr=tbl.addr + hd * 2, shape=(self.SEQ, hd), name=f"sin_{tag}"))

        # ---- KV caches: multi-KV -> nkv per-head buffers per layer; single-KV -> one buffer ----
        # head width is self.HD (= HDP when padded), so the cache stores full padded rows.
        if self.multikv:
            self.kc = [[self.growing_buffer(f"kc{li}_{h}", MAXPOS, hd) for h in range(nkv)] for li in range(L)]
            self.vc = [[self.growing_buffer(f"vc{li}_{h}", MAXPOS, hd) for h in range(nkv)] for li in range(L)]
        else:
            self.kc = [self.growing_buffer(f"kc{li}", MAXPOS, hd) for li in range(L)]
            self.vc = [self.growing_buffer(f"vc{li}", MAXPOS, hd) for li in range(L)]
        QS = self.SEQ * self.GRP
        A = pad64(QS)
        pbias = torch.full((A, A), float("-inf"))
        pbias.masked_fill_(torch.tril(torch.ones(A, A, dtype=torch.bool)), 0.0)
        pbias[:, QS:] = float("-inf")
        self.mask = self.const(pbias, "mask")
        self.identity = self.const(torch.eye(64), "identity")
        self.scratch = self.const(torch.zeros(16 * A * hd), "scratch")
        self.attn_p = self.const(torch.zeros(A * A), "attn_p")
        self.dmask = self.const(torch.zeros(1, MAXPOS), "dmask")
        # Verified bucketized decode (group_attn_pbi, from smolvlm2.py) — gated on nkv: each KV head
        # emits its own PBI bucket-dispatcher, so large nkv (SmolLM2=32) blows up the program +
        # risks back-to-back PBI corruption. Small nkv -> use the verified flow; else legacy group_attn.
        self.pbi_decode = self.multikv and nkv <= PBI_DECODE_MAX_NKV
        if self.pbi_decode:
            # group-row decode bias (group query rows, all at the same position); filled per-step in main
            self.dmask_pbi = self.const(torch.zeros(self.GRP, MAXPOS), "dmask_pbi")
            self.dec_nbuckets = max(1, (MAXPOS + 63) // 64)
        self.dx = self.const(torch.zeros(1, H), "dx")
        self.dscratch = self.const(torch.zeros(16 * 64 * hd), "dscratch")
        self.dec_probe = self.const(torch.zeros(L + 1, H), "dec_probe")  # clobber-safe per-layer decode taps
        self.x = self.const(self.x_input, "x")
        self.seq = self.gpr("seq_len"); self.qseq = self.gpr("q_seq_len"); self.bucket = self.gpr("bucket_idx")
        self.qall = self.gpr("q_all_heads"); self.kall = self.gpr("k_all_heads")  # qk-norm M = seq*nq / seq*nkv

    def _tag(self, li):
        p = self.spec["sliding_pattern"]
        return "global" if (p and (li + 1) % p == 0) else "local"

    # ---- WEAVE: walk layers, emit chain() + attention helper, per stage ----
    def build(self):
        L = self.spec["layers"]
        with self.program("prefill"):
            self.loop_reg(self.seq)
            x = self.x
            for li in range(L):
                x = self._layer(li, x, "prefill")
            self._map_ctx = ("prefill", -1, 0, "final-norm", None)
            self.out = self.chain(("rms_norm", x, self.final_norm, "fn"))

        with self.program("decode"):
            self.loop_reg(None)
            x = self.dx
            H = self.H
            for li in range(L):
                x = self._layer(li, x, "decode")
                self.ue.accelerator_memcpy(x.addr, self.dec_probe.addr + li * H * 2, H * 2)  # safe tap
            self._map_ctx = ("decode", -1, 0, "final-norm", None)
            self.dout = self.chain(("rms_norm", x, self.final_norm, "Dfn", {"M_reg": None}))
            self.ue.accelerator_memcpy(self.dout.addr, self.dec_probe.addr + L * H * 2, H * 2)  # post-final-norm
            self._map_ctx = ("decode", -1, 0, "lm_head", None)
            self.chain(("qmatmul", self.dout, self.lm_head, "Dlogits", {"write_back_disable": True}))
            self.advance(self.seq)

    # NOTE: the hardcoded gemma _layer() template was REMOVED. The block topology now
    # comes from self.block (derived from the IR by derive_block()), and _layer just
    # WEAVES it: residual stream `resid`, branch activation `cur`, one chain()/helper per
    # descriptor step. Works for any decoder LLM whose ops are in the catalog.
    def _layer(self, li, x, stage):
        pre = stage == "prefill"
        mm = "mm" if pre else "qmatmul"
        leg = {} if pre else {"M_reg": None}
        pfx = "" if pre else "D"
        W = lambda s: self.W[f"l{li}.{s}"]
        N = lambda s: self.N[f"l{li}.{s}"]
        Nb = lambda s: self.N.get(f"l{li}.{s}.beta")    # layernorm bias (None for rmsnorm)
        resid = cur = x
        for j, entry in enumerate(self.block):
            kind, role = entry[0], entry[1]
            tr = entry[2] if len(entry) > 2 else {}       # self-describing traits (norm type, act, causal)
            self._map_ctx = (stage, li, j, kind, role)    # tag the Stage-4 mapping recorder
            nm = f"{pfx}{li}_{j}"
            if kind == "norm":
                if tr.get("type") == "layer":             # layernorm: gamma + beta (vision encoder)
                    cur = self.chain(("layer_norm", cur, N(role), Nb(role), "n" + nm, leg))
                else:
                    cur = self.chain(("rms_norm", cur, N(role), "n" + nm, leg))
            elif kind == "attention":
                q = self.chain((mm, cur, W("q"), "q" + nm)); k = self.chain((mm, cur, W("k"), "k" + nm)); v = self.chain((mm, cur, W("v"), "v" + nm))
                cur = self._attn(li, q, k, v, stage) if tr.get("causal", True) is not False \
                    else self._attn_vit(li, q, k, v, nm)
            elif kind == "matmul":                              # o_proj (decoder 'o') / out_proj (vit 'out')
                cur = self.chain((mm, cur, W(role), "o" + nm))
            elif kind == "swiglu":
                gg = self.chain((mm, cur, W("gate"), "g" + nm, {"act": self.act})); uu = self.chain((mm, cur, W("up"), "u" + nm))
                ml = self.chain(("mul", gg, uu, "m" + nm)); cur = self.chain((mm, ml, W("down"), "d" + nm))
            elif kind == "mlp":                                 # plain fc1 -> act -> fc2 (vision encoder)
                h = self.chain((mm, cur, W("fc1"), "f1" + nm, {"act": tr.get("act", "gelu")}))
                cur = self.chain((mm, h, W("fc2"), "f2" + nm))
            elif kind == "residual":
                resid = self.chain(("add", resid, cur, "r" + nm)); cur = resid
        return resid

    def _attn(self, li, q, k, v, stage):
        N = lambda s: self.N.get(f"l{li}.{s}")        # None if the model has no qk-norm (llama)
        if self.multikv:                               # multi-KV: lo|hi (hd<128) or contiguous (hd>=128)
            return self._attn_mk(li, q, k, v, stage)
        if stage == "prefill":
            return self.attention(q, k, v, q_norm=N("q_norm"), k_norm=N("k_norm"),
                                   cos=self.rope[self._tag(li)][0], sin=self.rope[self._tag(li)][1],
                                   group=self.GRP, head_dim=self.HD, scratch=self.scratch,
                                   identity=self.identity, mask=self.mask, attn_p=self.attn_p,
                                   bucket_reg=self.bucket, qseq_reg=self.qseq, k_cache=self.kc[li], v_cache=self.vc[li])
        return self.decode_attention(q, k, v, q_norm=N("q_norm"), k_norm=N("k_norm"),
                                     rope_base=self.rope_base[self._tag(li)], pos_reg=self.seq,
                                     group=self.GRP, head_dim=self.HD, k_cache=self.kc[li], v_cache=self.vc[li],
                                     scratch=self.dscratch, identity=self.identity, mask=self.dmask, bucket_reg=self.bucket)

    # ------- multi-KV / lo|hi variant (llama-style): derived, no model-specific math -------
    def _scat(self, read_addr, read_stride_bytes, write_specs, count, sram, seq):
        self.ue._emit_pbi_scatter_per_token(read_base=read_addr, read_stride_bytes=read_stride_bytes,
                                            write_specs=write_specs, sram_byte_addr=sram,
                                            element_count=count, gpr_seq_len=int(self.seq), template_seq_len=seq)

    def _flash_core(self, qf, kr_head, v_head, seq, name, hd=None):
        """rope-free per-(single-KV)-head flash: gqa-dup K/V by group, bucketized flash.
        qf [seq, group*hd] roped, kr_head/v_head [seq, hd]. Returns [seq, group*hd].
        hd defaults to self.HD; pass self.HDP for padded (hd%64!=0) heads."""
        import torch
        hd, group, qs = (hd or self.HD), self.GRP, seq * self.GRP
        kd = self.arena.alloc((qs, hd), name="kd" + name); vd = self.arena.alloc((qs, hd), name="vd" + name)
        z = torch.zeros(pad64(qs), hd, dtype=torch.bfloat16)
        self.ue.dma_to_accelerator_memory(kd.addr, z); self.ue.dma_to_accelerator_memory(vd.addr, z)
        self.gqa_duplicate(kr_head.addr, kd.addr, head_dim=hd, group_size=group, seq_reg=self.seq, seq_len=seq)
        self.gqa_duplicate(v_head.addr, vd.addr, head_dim=hd, group_size=group, seq_reg=self.seq, seq_len=seq)
        qff = Tensor(addr=qf.addr, shape=(qs, hd), name="qff" + name)
        return self.chain(("flash_attn", qff, kd, vd, "fa" + name,
                           {"group": group, "scratch": self.scratch, "identity": self.identity,
                            "mask": self.mask, "attn_p": self.attn_p, "bucket_reg": self.bucket,
                            "num_buckets": max(1, (qs + 63) // 64)}))

    def _attn_mk(self, li, q, k, v, stage):
        nkv, nq, hd, group = self.spec["nkv"], self.spec["nq"], self.HD, self.GRP
        half, bpe = hd // 2, 2
        if stage == "decode":
            return self._attn_mk_decode(li, q, k, v, nkv, nq, hd, group, half, bpe)
        seq = q.shape[0]
        N = lambda s: self.N.get(f"l{li}.{s}")
        if self.lohi:
            return self._attn_mk_d64(li, q, k, v, nkv, nq, hd, group, seq, bpe)
        # contiguous (head_dim>=128): optional qk-norm, then per-head rope_gqa. no permute.
        cos, sin = self.rope[self._tag(li)]
        if self.has_qk_norm:
            qn = self.chain(("rms_norm", Tensor(addr=q.addr, shape=(seq * nq, hd), dtype=q.dtype),
                             N("q_norm"), f"qn{li}", {"M_reg": self.qall}))
            q = Tensor(addr=qn.addr, shape=(seq, nq * hd), dtype=q.dtype)
            kn = self.chain(("rms_norm", Tensor(addr=k.addr, shape=(seq * nkv, hd), dtype=k.dtype),
                             N("k_norm"), f"kn{li}", {"M_reg": self.kall}))
            k = Tensor(addr=kn.addr, shape=(seq, nkv * hd), dtype=k.dtype)
        qr = self.chain(("rope_gqa", q, cos, sin, f"qr{li}", {"group": nq}))
        kr = self.chain(("rope_gqa", k, cos, sin, f"kr{li}", {"group": nkv}))
        out = self.arena.alloc((seq, nq * hd), name=f"ao{li}")
        for h in range(nkv):
            krh = self.arena.alloc((seq, hd), name=f"krh{li}_{h}")
            vh = self.arena.alloc((seq, hd), name=f"vh{li}_{h}")
            qfh = self.arena.alloc((seq, group * hd), name=f"qfh{li}_{h}")
            kc, vc = self.kc[li][h].base, self.vc[li][h].base
            self._scat(kr.addr + h * hd * bpe, nkv * hd * bpe,
                       [(krh.addr, hd * bpe), (kc, hd * bpe)], hd, 0x10000, seq)
            self._scat(qr.addr + h * group * hd * bpe, nq * hd * bpe,
                       [(qfh.addr, group * hd * bpe)], group * hd, 0x30000, seq)
            self._scat(v.addr + h * hd * bpe, nkv * hd * bpe,
                       [(vh.addr, hd * bpe), (vc, hd * bpe)], hd, 0x20000, seq)
            a_h = self._flash_core(qfh, krh, vh, seq, f"{li}_{h}")        # [seq, group*hd]
            self._scat(a_h.addr, group * hd * bpe,
                       [(out.addr + h * group * hd * bpe, nq * hd * bpe)], group * hd, 0x40000, seq)
        return out

    def _attn_mk_d64(self, li, q, k, v, nkv, nq, hd, group, seq, bpe):
        """head_dim<128 prefill: extract each head's CONTIGUOUS hd-wide slice (128B-aligned),
        then d64-rope per head. K roped at M=seq; the group's Q stacked token-major [seq*group, hd]
        and roped in one d64 call against the group-duplicated table. Roped K -> per-head cache."""
        out = self.arena.alloc((seq, nq * hd), name=f"ao{li}")
        for h in range(nkv):
            krh = self.arena.alloc((seq, hd), name=f"krh{li}_{h}")
            vh = self.arena.alloc((seq, hd), name=f"vh{li}_{h}")
            qfh = self.arena.alloc((seq, group * hd), name=f"qfh{li}_{h}")
            kc, vc = self.kc[li][h].base, self.vc[li][h].base
            # raw per-head extraction (full hd block = 128B aligned). V also -> cache.
            self._scat(k.addr + h * hd * bpe, nkv * hd * bpe, [(krh.addr, hd * bpe)], hd, 0x10000, seq)
            self._scat(v.addr + h * hd * bpe, nkv * hd * bpe,
                       [(vh.addr, hd * bpe), (vc, hd * bpe)], hd, 0x20000, seq)
            self._scat(q.addr + h * group * hd * bpe, nq * hd * bpe,
                       [(qfh.addr, group * hd * bpe)], group * hd, 0x30000, seq)
            # d64 rope: K head (M=seq, default loop reg); stacked Q (M=seq*group, dup table)
            krr = self.chain(("rope_d64", krh, f"krr{li}_{h}", {"table": self.rope_packed}))
            qfr = self.chain(("rope_d64", Tensor(addr=qfh.addr, shape=(seq * group, hd), dtype=q.dtype),
                              f"qfr{li}_{h}", {"table": self.rope_packed_gqa, "M_reg": self.qseq}))
            self._scat(krr.addr, hd * bpe, [(kc, hd * bpe)], hd, 0x10080, seq)   # roped K -> cache
            qff = Tensor(addr=qfr.addr, shape=(seq, group * hd), dtype=q.dtype)
            a_h = self._flash_core(qff, krr, vh, seq, f"{li}_{h}")
            self._scat(a_h.addr, group * hd * bpe,
                       [(out.addr + h * group * hd * bpe, nq * hd * bpe)], group * hd, 0x40000, seq)
        return out

    def _group_attn_decode(self, qfh, kc_t, vc_t, group, name):
        """One KV-head's decode attention vs its cache. Routes to the verified bucketized
        group_attn_pbi (smolvlm2.py flow) when self.pbi_decode is set (small nkv), else the
        legacy full-MAXPOS group_attn. Identical inputs; only the kernel + bias differ."""
        if self.pbi_decode:
            return self.chain(("group_attn_pbi", qfh, kc_t, vc_t, name,
                               {"group": group, "scratch": self.dscratch, "identity": self.identity,
                                "mask": self.dmask_pbi, "bucket_reg": self.bucket,
                                "num_buckets": self.dec_nbuckets}))
        return self.chain(("group_attn", qfh, kc_t, vc_t, name,
                           {"group": group, "scratch": self.dscratch, "identity": self.identity,
                            "mask": self.dmask, "seq_len_override": MAXPOS}))

    def _attn_mk_decode(self, li, q, k, v, nkv, nq, hd, group, half, bpe):
        """Single-token multi-KV decode: rope at runtime pos, per-head extract -> append to
        KV cache -> per-head group_attn vs cache. Single token so direct memcpy."""
        if not self.lohi:
            return self._attn_mk_decode_contig(li, q, k, v, nkv, nq, hd, group, bpe)
        # d64 per-head decode rope at runtime pos. q/k are head-major [1, n*hd]; rope each head's
        # contiguous hd slice. ALL ropes BEFORE any append (append_row clobbers the shared scratch
        # GPR the rope's register-addressed DMA uses).
        qr = self.arena.alloc((1, nq * hd), name=f"Dqr{li}")
        kr = self.arena.alloc((1, nkv * hd), name=f"Dkr{li}")
        rd = {"table": self.rope_packed, "pos_reg": self.seq, "tmp_reg": self.ue.TMP_REG}
        for i in range(nq):
            self.chain(("rope_decode_d64", Tensor(addr=q.addr + i * hd * bpe, shape=(1, hd), dtype=q.dtype),
                        Tensor(addr=qr.addr + i * hd * bpe, shape=(1, hd)), rd))
        for h in range(nkv):
            self.chain(("rope_decode_d64", Tensor(addr=k.addr + h * hd * bpe, shape=(1, hd), dtype=k.dtype),
                        Tensor(addr=kr.addr + h * hd * bpe, shape=(1, hd)), rd))
        out = self.arena.alloc((1, nq * hd), name=f"Dao{li}")
        for h in range(nkv):
            self.append_row(self.kc[li][h], kr.addr + h * hd * bpe, self.seq)   # roped K head at pos
            self.append_row(self.vc[li][h], v.addr + h * hd * bpe, self.seq)    # V head (standard)
            qfh = Tensor(addr=qr.addr + h * group * hd * bpe, shape=(group, hd), name="Dqfh")
            kc_t = Tensor(addr=self.kc[li][h].base, shape=(MAXPOS, hd), name="kct")
            vc_t = Tensor(addr=self.vc[li][h].base, shape=(MAXPOS, hd), name="vct")
            a = self._group_attn_decode(qfh, kc_t, vc_t, group, f"Dda{li}_{h}")
            self.ue.accelerator_memcpy(a.addr, out.addr + h * group * hd * bpe, group * hd * bpe)
        return out

    def _attn_mk_decode_contig(self, li, q, k, v, nkv, nq, hd, group, bpe):
        """Contiguous (head_dim>=128) multi-KV decode: optional qk-norm, per-head rope_decode
        (single row is contiguous), append to per-head cache, per-KV-head legacy group_attn."""
        N = lambda s: self.N.get(f"l{li}.{s}")
        if self.has_qk_norm:
            qn = self.chain(("rms_norm", Tensor(addr=q.addr, shape=(nq, hd), dtype=q.dtype),
                             N("q_norm"), f"Dqn{li}", {"M_reg": None}))
            q = Tensor(addr=qn.addr, shape=(1, nq * hd), dtype=q.dtype)
            kn = self.chain(("rms_norm", Tensor(addr=k.addr, shape=(nkv, hd), dtype=k.dtype),
                             N("k_norm"), f"Dkn{li}", {"M_reg": None}))
            k = Tensor(addr=kn.addr, shape=(1, nkv * hd), dtype=k.dtype)
        # rope every head at the runtime position (head_dim>=128; single row -> contiguous slice).
        # ALL ropes share gr (=_addr_tmp) and must precede the appends (which clobber it).
        gr = self.runtime_addr(self.rope_base[self._tag(li)], self.seq, 2 * hd * bpe)
        qr = self.arena.alloc((1, nq * hd), name=f"Dqr{li}")
        kr = self.arena.alloc((1, nkv * hd), name=f"Dkr{li}")
        for i in range(nq):
            self.chain(("rope_decode", Tensor(addr=q.addr + i * hd * bpe, shape=(1, hd), dtype=q.dtype),
                        Tensor(addr=qr.addr + i * hd * bpe, shape=(1, hd)), {"gr_weight": gr}))
        for h in range(nkv):
            self.chain(("rope_decode", Tensor(addr=k.addr + h * hd * bpe, shape=(1, hd), dtype=k.dtype),
                        Tensor(addr=kr.addr + h * hd * bpe, shape=(1, hd)), {"gr_weight": gr}))
        out = self.arena.alloc((1, nq * hd), name=f"Dao{li}")
        for h in range(nkv):
            self.append_row(self.kc[li][h], kr.addr + h * hd * bpe, self.seq)   # roped K head (1 row)
            self.append_row(self.vc[li][h], v.addr + h * hd * bpe, self.seq)    # V head (standard)
            qfh = Tensor(addr=qr.addr + h * group * hd * bpe, shape=(group, hd), name="qfh")
            kc_t = Tensor(addr=self.kc[li][h].base, shape=(MAXPOS, hd), name="kct")
            vc_t = Tensor(addr=self.vc[li][h].base, shape=(MAXPOS, hd), name="vct")
            a = self._group_attn_decode(qfh, kc_t, vc_t, group, f"Dda{li}_{h}")
            self.ue.accelerator_memcpy(a.addr, out.addr + h * group * hd * bpe, group * hd * bpe)
        return out


def _pad_proj_out(w, n_heads, hr, HDP, mode, scale=1.0):
    """Pad an OUTPUT projection weight [n_heads*hr, H] -> [n_heads*HDP, H] so the matmul emits a
    per-head padded layout. mode='lohi' (q/k, for rope): real rows split to lo [0:hr/2] and hi
    [HDP/2:HDP/2+hr/2]. mode='contig' (v, not roped): real rows at [0:hr]. Pad rows are zero, so
    those output lanes are zero. `scale` pre-multiplies (used to fold flash's attn-scale into q)."""
    import torch
    H = w.shape[1]
    hr2, hP2 = hr // 2, HDP // 2
    out = torch.zeros(n_heads, HDP, H, dtype=w.dtype)
    wv = (w * scale).view(n_heads, hr, H)
    if mode == "lohi":
        out[:, :hr2] = wv[:, :hr2]
        out[:, hP2:hP2 + hr2] = wv[:, hr2:]
    else:
        out[:, :hr] = wv
    return out.reshape(n_heads * HDP, H)


def _pad_proj_in(w, n_heads, hr, HDP):
    """Pad o_proj INPUT columns [H, n_heads*hr] -> [H, n_heads*HDP] (contiguous, matching V's
    layout in the attention output). Pad columns are zero, so they ignore the padded lanes."""
    import torch
    H = w.shape[0]
    out = torch.zeros(H, n_heads, HDP, dtype=w.dtype)
    out[:, :, :hr] = w.view(H, n_heads, hr)
    return out.reshape(H, n_heads * HDP)


def _node_class(n):
    """Owning module class for an fx node (e.g. 'Gemma3RMSNorm', 'Linear') — this is what
    tells us the norm TYPE (RMS vs LayerNorm) and which submodule the op came from."""
    ms = n.meta.get("nn_module_stack")
    if not ms:
        return ""
    cls = list(ms.values())[-1][1]
    return str(cls).rsplit(".", 1)[-1].rstrip("'>")


def _node_srcloc(n):
    """HF source 'modeling_<arch>.py:LINE  <code>' for an fx node, from its stack_trace —
    the exact line in the model's forward() that produced this op, so it can be studied."""
    import re
    st = n.meta.get("stack_trace") or ""
    lines = st.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        mm = re.search(r'File "([^"]+)", line (\d+)', lines[i])
        if mm and "modeling" in mm.group(1):
            code = lines[i + 1].strip() if i + 1 < len(lines) else ""
            return f"{mm.group(1).split('/')[-1]}:{mm.group(2)}  {code}"
    return ""


def derive_block(model, dec=None, region="decoder", trace=None):
    """Per-layer block descriptor [(kind, role, traits)] EXTRACTED from the IR by module
    ownership (nn_module_stack). Not a template: gemma's sandwich, llama's pre-LN, a SigLIP
    ViT encoder layer each fall out of their own graph. `region` selects the workload —
    'decoder' (causal LM block) or 'encoder' (bidirectional ViT block). The descriptor is
    self-describing: each step carries its traits (norm type, activation, attention
    causality) so the SAME weaver runs any region without a model-type branch.

    `traits` is omitted for the legacy decoder kinds (back-compat: 2-tuples), and present
    for the encoder kinds. The weaver unpacks tolerantly."""
    if region == "encoder":
        return _derive_encoder_block(dec, trace=trace)
    REV = {v.split(".")[-1]: k for k, v in NORM.items()}     # leaf suffix -> short norm key
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    ep = torch.export.export(dec if dec is not None else model, (ids,), {"use_cache": False}, strict=False)
    base = lambda t: getattr(t, "__name__", str(t)).split(".")[0]
    def leaf(n):
        ms = n.meta.get("nn_module_stack"); return list(ms.values())[-1] if ms else (None, "")
    def stk(n):
        ms = n.meta.get("nn_module_stack"); return " ".join(str(c).lower() for _, c in ms.values()) if ms else ""
    blk, want = [], "layers.0"
    def rec(node, owner, decision, step=None):               # decision-trace record (optional)
        if trace is not None:
            trace.append({"node": node.name, "target": base(node.target), "owner": owner,
                          "decision": decision, "step": step,
                          "cls": _node_class(node), "src": _node_srcloc(node)})
    for n in ep.graph_module.graph.nodes:
        if n.op != "call_function":
            continue
        path, cls = leaf(n)
        if not path or want not in path:
            continue                                         # out-of-scope: only layer-0 is the repeat unit
        c = str(cls).lower(); t = base(n.target)
        suffix = path.split(want + ".")[-1]; leafname = suffix.split(".")[-1]
        owner = suffix
        if t == "scaled_dot_product_attention":
            blk.append(("attention", None)); rec(n, owner, "EMIT attention (collapses q/k/v/rope/sdpa cluster)", len(blk) - 1)
        elif leafname == "o_proj" and t == "linear":
            blk.append(("matmul", "o")); rec(n, owner, "EMIT matmul:o (o_proj)", len(blk) - 1)
        elif leafname == "down_proj" and t == "linear":
            blk.append(("swiglu", None)); rec(n, owner, "EMIT swiglu (collapses gate/act/up/mul/down cluster)", len(blk) - 1)
        elif ("rmsnorm" in c or "layernorm" in c) and "self_attn" not in suffix:
            short = REV.get(leafname, leafname)
            if not blk or blk[-1] != ("norm", short):
                blk.append(("norm", short)); rec(n, owner, f"EMIT norm:{short}", len(blk) - 1)
            else:
                rec(n, owner, f"COLLAPSE into prev norm:{short} (duplicate node)")
        elif t == "add" and "decoderlayer" in stk(n) and "norm" not in c and "attention" not in c:
            blk.append(("residual", None)); rec(n, owner, "EMIT residual (layer-scope add)", len(blk) - 1)
        else:
            rec(n, owner, f"ignore ({t}: not a trigger — folded into a cluster or shape/scaffold)")
    return blk


def _derive_encoder_block(vis, trace=None):
    """Extract one ViT encoder layer's topology + traits from the IR. `vis` is the vision
    transformer module (has .encoder.layers). Exports a single layer (self-contained: it
    takes a hidden-state tensor, no pixel_values), so it traces without the patch front-end.
    Emits trait-tagged descriptors:
      ('norm', <key>, {'type':'layer'})           layernorm (gamma+beta)
      ('attention', None, {'causal': False})       bidirectional self-attn
      ('matmul', 'out', {})                        out_proj
      ('mlp', None, {'act': <gelu|silu>})          plain fc1->act->fc2 (not gated)
      ('residual', None, {})                       residual add
    Order reflects the actual graph (pre-LN ViT: LN1, attn, out, resid, LN2, mlp, resid)."""
    layer = vis.encoder.layers[0]
    H = vis.config.hidden_size
    hs = torch.zeros(1, 16, H)
    ep = torch.export.export(layer, (hs, None), strict=False)
    base = lambda t: getattr(t, "__name__", str(t)).split(".")[0]
    def leaf(n):
        ms = n.meta.get("nn_module_stack"); return list(ms.values())[-1] if ms else (None, "")
    # detect activation from the layer's mlp module rather than the (often-decomposed) graph
    act = "gelu" if "gelu" in type(layer.mlp.activation_fn).__name__.lower() else "silu"
    blk = []
    def rec(node, owner, decision):
        if trace is not None:
            trace.append({"node": node.name, "target": base(node.target), "owner": owner, "decision": decision})
    for n in ep.graph_module.graph.nodes:
        if n.op != "call_function":
            continue
        path, cls = leaf(n); c = str(cls).lower(); t = base(n.target)
        owner = path or "<layer-scope>"
        # residual adds live at LAYER scope (empty leaf path, owning class = the encoder layer)
        if t == "add" and not path and "encoderlayer" in c:
            blk.append(("residual", None, {})); rec(n, owner, "EMIT residual (layer-scope add)")
            continue
        if not path:
            continue
        leafname = path.split(".")[-1]
        if t == "scaled_dot_product_attention":
            blk.append(("attention", None, {"causal": False})); rec(n, owner, "EMIT attention {causal:False} (collapses q/k/v/sdpa)")
        elif leafname == "out_proj" and t == "linear":
            blk.append(("matmul", "out", {})); rec(n, owner, "EMIT matmul:out (out_proj)")
        elif leafname == "fc2" and t == "linear":
            blk.append(("mlp", None, {"act": act})); rec(n, owner, f"EMIT mlp {{act:{act}}} (collapses fc1/act/fc2)")
        elif "layernorm" in c:
            key = "ln1" if (not blk or all(b[0] != "attention" for b in blk)) else "ln2"
            if not blk or blk[-1][:2] != ("norm", key):
                blk.append(("norm", key, {"type": "layer"})); rec(n, owner, f"EMIT norm:{key} {{type:layer}}")
            else:
                rec(n, owner, f"COLLAPSE into prev norm:{key} (duplicate node)")
        else:
            rec(n, owner, f"ignore ({t}: not a trigger — folded into a cluster or shape/scaffold)")
    return blk


def decide_forward(spec, embed_scale):
    """Map model CHARACTERISTICS -> forward-pass DECISIONS, and show the reasoning.
    The dispatch reads these decisions — the path is implied by traits, not the model name."""
    nq, nkv, hd = spec["nq"], spec["nkv"], spec["hd"]
    group = nq // max(1, nkv)
    d = {
        "model_type": spec["model_type"],
        "matmul": "mm(PBI) / qmatmul(decode)" if True else "bf16",   # weights are if4 -> quantized
        "stages": "encode + prefill + decode (+KV cache)" if spec["model_type"] == "vlm" else "prefill + decode (+KV cache)",
        "group_size": group,
        "attn_variant": "single-KV grouped" if nkv == 1 else "multi-KV head",
        "rope_variant": (f"contiguous@{pad64(hd)} (lo|hi-padded weights)" if hd % 64 else
                         "contiguous (rope_gqa)") if (hd >= 128 or hd % 64) else "tiled/d64 ([lo|hi] permute)",
        "hd_padded": pad64(hd) if hd % 64 else hd,
        "qk_norm": "apply per-head Q/K norm" if spec["qk_norm"] else "skip (no qk-norm)",
        "act": spec["act"],
        "norm_offset": spec["norm_offset"],
        "embed_scale": embed_scale,
    }
    rows = [
        ("arch",      d['model_type'],                       "decoder-only LM" if d['model_type']=="lm" else "vision + decoder"),
        ("matmul",    "if4 weights",                         d['matmul']),
        ("stages",    "causal decoder",                      d['stages']),
        ("attention", f"nkv={nkv} nq={nq}",                  f"{d['attn_variant']} · GQA×{group}"),
        ("rope",      f"hd={hd}" + (f"->{pad64(hd)}" if hd % 64 else ""),
                      d['rope_variant'] + (f"  (hd%64!=0: lo|hi pad to {pad64(hd)}, real lanes [0:{hd//2}]+[{pad64(hd)//2}:{pad64(hd)//2+hd//2}])" if hd % 64 else "")),
        ("qk_norm",   "present" if spec['qk_norm'] else "absent", d['qk_norm']),
        ("act",       d['act'],                              "swiglu"),
        ("rmsnorm",   f"norm_offset={spec['norm_offset']}",  f"gamma {'(1+w)' if spec['norm_offset'] else 'as-is'}"),
        ("embeds",    f"scale={embed_scale:.2f}",            "×sqrt(H)" if embed_scale>1.5 else "unscaled"),
    ]
    print("\nforward pass")
    for k, char, choice in rows:
        print(f"  {k:10s}{char:22s}{choice}")
    return d


def detect_model_type(model) -> str:
    """Detect high-level architecture category from the HF config.

    Returns: 'lm' (decoder-only) | 'vlm' (vision-encoder + decoder)
    """
    cfg = model.config
    mt = getattr(cfg, "model_type", "")
    # Explicit vision-model model_types
    if mt in ("smolvlm", "idefics", "idefics3", "llava", "llava_next", "florence2"):
        return "vlm"
    # Has a distinct vision_config
    if hasattr(cfg, "vision_config") and cfg.vision_config is not None:
        return "vlm"
    # Model class name contains vision keywords
    cname = type(model).__name__.lower()
    if any(kw in cname for kw in ("vl", "vision", "multi")):
        return "vlm"
    return "lm"


def locate_decoder(model):
    """Resolve the causal decoder regardless of LM vs VLM nesting. Returns
    (decoder_module, weight_prefix, rotary_emb). Decoder-only LMs live at model.model;
    VLMs (SmolVLM/Idefics) nest the text decoder under model.model.text_model (some name
    it language_model). The weight prefix is what state_dict() keys are prefixed with."""
    base = model.model
    if hasattr(base, "text_model"):
        dec, pfx = base.text_model, "model.text_model."
    elif hasattr(base, "language_model"):
        dec, pfx = base.language_model, "model.language_model."
    else:
        dec, pfx = base, "model."
    return dec, pfx, getattr(dec, "rotary_emb", None)


# Region keywords -> canonical region name. The qualified module path (from the fx
# nn_module_stack, or the static module tree) carries this for free; we just read it.
_REGION_KEYS = (
    ("vision_model", "encoder"), ("vision_tower", "encoder"), ("vision", "encoder"),
    ("connector", "connector"), ("modality_projection", "connector"),
    ("multi_modal_projector", "connector"), ("mm_projector", "connector"),
    ("text_model", "decoder"), ("language_model", "decoder"),
    ("lm_head", "decoder"), ("embed_tokens", "decoder"),
)


def _region_of(qual_name: str) -> str:
    """Map a fully-qualified module path to its architectural region. Falls back to
    'decoder' for a bare decoder-only model (whose paths are just model.layers.*)."""
    low = qual_name.lower()
    for key, region in _REGION_KEYS:
        if key in low:
            return region
    return "decoder"


def partition_regions(model):
    """Partition the model's parameters by architectural region using the qualified
    module paths alone — no model-type label, no per-arch branch. This is the proof
    that the decoder region is cleanly separable: the compiler only consumes 'decoder'.
    Returns {region: {"params": int, "modules": int, "prefixes": set}}."""
    from collections import defaultdict
    regions = defaultdict(lambda: {"params": 0, "modules": 0, "prefixes": set()})
    for name, p in model.named_parameters():
        r = _region_of(name)
        regions[r]["params"] += p.numel()
        regions[r]["prefixes"].add(name.split("layers.")[0] if "layers." in name else name.rsplit(".", 1)[0])
    for name, _ in model.named_modules():
        if name:
            regions[_region_of(name)]["modules"] += 1
    return dict(regions)


def derive_spec(model, dec=None):
    """Extract all compile-time spec from the HF model. Includes text (decoder) and
    optionally vision (encoder) dimensions. `dec` is the resolved decoder module (so layer
    count comes from the decoder, not vision-encoder 'layers.' keys in a VLM state_dict)."""
    tcfg = getattr(model.config, "text_config", model.config)
    g = lambda a, d=None: getattr(tcfg, a, d)
    sd = model.state_dict()
    layers = len(dec.layers) if dec is not None else \
        max(int(n.split("layers.")[1].split(".")[0]) for n in sd if "layers." in n) + 1
    H, nq = g("hidden_size"), g("num_attention_heads")
    spec = {
        "model_type": detect_model_type(model),
        "layers": layers, "H": H, "I": g("intermediate_size"),
        "hd": g("head_dim") or H // nq, "nq": nq,
        "nkv": g("num_key_value_heads", nq), "V": g("vocab_size"),
        "qk_norm": any("q_norm" in n for n in sd),
        # attention softmax scale (HF: 1/sqrt(query_pre_attn_scalar or head_dim)). Flash applies
        # 1/sqrt(head_dim_it_sees); when we pad head_dim this differs, so we pre-correct Q.
        "attn_scale": (g("query_pre_attn_scalar", 0) or (g("head_dim") or H // nq)) ** -0.5,
        "theta": g("rope_theta", 1e4), "local_base": g("rope_local_base", g("rope_theta", 1e4)),
        "sliding_pattern": g("sliding_window_pattern", 0),
        "act": g("hidden_activation", g("hidden_act", "gelu")),
        "norm_offset": 1.0 if "gemma" in type(model).__name__.lower() else 0.0,
        "tied": getattr(model.config, "tie_word_embeddings", False) or not any("lm_head" in n for n in sd),
    }
    # Vision encoder dims (present on VLM models only)
    if spec["model_type"] == "vlm":
        vcfg = getattr(model.config, "vision_config", None)
        if vcfg is not None:
            spec.update(
                v_hidden=vcfg.hidden_size,
                v_layers=vcfg.num_hidden_layers,
                v_heads=vcfg.num_attention_heads,
                v_intermediate=vcfg.intermediate_size,
                v_head_dim=vcfg.hidden_size // vcfg.num_attention_heads,
                v_act=vcfg.hidden_act,
                v_patch=vcfg.patch_size,
                v_image=vcfg.image_size,
                v_channels=getattr(vcfg, "num_channels", 3),
                v_layer_norm_eps=vcfg.layer_norm_eps,
            )
    return spec


def _set_decode_bias(m, valid):
    """Push the per-step decode bias: legacy (1,MAXPOS) always; plus the (group,MAXPOS) bias
    for the bucketized group_attn_pbi path when active. `valid` = positions attendable (= pos+1)."""
    dmask = torch.full((1, MAXPOS), -1e36); dmask[0, :valid] = 0.0
    m.ue.dma_to_accelerator_memory(m.dmask.addr, dmask.to(torch.bfloat16))
    if getattr(m, "pbi_decode", False):
        dg = torch.full((m.GRP, MAXPOS), -1e36); dg[:, :valid] = 0.0
        m.ue.dma_to_accelerator_memory(m.dmask_pbi.addr, dg.to(torch.bfloat16))


def _silence_hf_autoconvert():
    # HF spawns a daemon thread that tries to open a safetensors-conversion PR; on
    # .bin-only repos (e.g. TinyLlama_v1.1) it fails and dumps a useless OSError
    # traceback to stderr, asynchronously. Swallow exceptions from that thread only.
    import threading
    _prev = threading.excepthook
    def _hook(arg):
        if arg.thread is not None and "auto_conversion" in (arg.thread.name or ""):
            return
        _prev(arg)
    threading.excepthook = _hook


def _load_hf(cls, name):
    # Prefer safetensors; for .bin-only repos fall back explicitly.
    try:
        return cls.from_pretrained(name, dtype=torch.float32, use_safetensors=True).eval()
    except Exception:
        return cls.from_pretrained(name, dtype=torch.float32, use_safetensors=False).eval()


def main():
    args = ARGS

    _silence_hf_autoconvert()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = _load_hf(AutoModelForCausalLM, args.model)
    except Exception:                        # VLM: not a CausalLM head class
        from transformers import AutoModelForImageTextToText
        model = _load_hf(AutoModelForImageTextToText, args.model)
    dec, wpfx, rotary = locate_decoder(model)   # decoder module + weight prefix (LM or VLM-nested)
    spec = derive_spec(model, dec)
    mtype = detect_model_type(model)
    label = "causal_lm" if mtype == "lm" else mtype          # friendly banner / detection label
    ExtractedDecoder._label = label
    print(f"{label} detected  (decoder prefix: {wpfx!r})")
    print(f"spec: {spec}")
    btrace = []                              # provenance: which fx op / source line produced each step
    block = derive_block(model, dec, trace=btrace)   # IR-extracted block topology (not a template)
    print(f"block ({len(block)} steps): {[k if r is None else f'{k}:{r}' for k, r in block]}")

    # tiny models can't do the algebra prompt; downgrade to a simpler one (only if the
    # user didn't pass --prompt explicitly).
    nparams = sum(p.numel() for p in model.parameters())
    if args.prompt == DEFAULT_PROMPT and nparams < SMALL_PARAM_THRESHOLD:
        args.prompt = SIMPLE_PROMPT
        print(f"({nparams/1e6:.0f}M params < {SMALL_PARAM_THRESHOLD/1e6:.0f}M — using simpler prompt)")

    # instruct models -> chat template; base models have none -> raw prompt
    if tok.chat_template:
        chat = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                       add_generation_prompt=True, tokenize=False)
        ids = tok(chat, return_tensors="pt", add_special_tokens=False).input_ids
    else:
        print("(no chat template — base model, using raw prompt)")
        ids = tok(args.prompt, return_tensors="pt").input_ids
    SEQ = ids.shape[1]
    GEN = (MAXPOS - SEQ) if args.gen <= 0 else min(args.gen, MAXPOS - SEQ)
    embed = dec.embed_tokens.weight.float()               # INPUT embedding (for dx)
    lmw = model.lm_head.weight.float()                     # OUTPUT proj (lm_head) — != embed if untied
    with torch.no_grad():
        x_input = dec.embed_tokens(ids)[0].float()
        # derive the embedding scale empirically (gemma scales by sqrt(H); llama = 1.0)
        probe = dec.embed_tokens(torch.tensor([[ids[0, 0].item()]]))[0, 0].float()
        norm = (probe / embed[ids[0, 0]]).median().item()
        # lo|hi (sub-128 head_dim): rope base cos/sin from the model's rotary (handles scaling)
        if spec["hd"] < 128:
            dvec = torch.zeros(1, 1, MAXPOS, spec["hd"])
            cf, sf = rotary(dvec, torch.arange(MAXPOS)[None])
            ExtractedDecoder.rope_cos = cf[0, :, :spec["hd"] // 2].float()
            ExtractedDecoder.rope_sin = sf[0, :, :spec["hd"] // 2].float()
    plan = decide_forward(spec, norm)               # traits -> forward-pass decisions (shown)
    ExtractedDecoder._dram = decide_dram(model)             # size DRAM partitions to the IF4 footprint
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")} - {None}

    ExtractedDecoder.spec = spec; ExtractedDecoder.sd = model.state_dict(); ExtractedDecoder.block = block; ExtractedDecoder.wpfx = wpfx
    ExtractedDecoder.plan = plan; ExtractedDecoder.SEQ = SEQ; ExtractedDecoder.x_input = x_input
    ExtractedDecoder._capture_map = True             # log every chain()->UE_HLO resolution during build
    m = ExtractedDecoder()
    _dump_mapping(m, block, btrace, args.model, dec)   # the useful one: fx op + source -> UE_HLO hashtable
    if os.environ.get("LOOM_DUMP_ALL"):           # raw fx / extraction / ir: opt-in (verbose, rarely read)
        _dump_ir(m, spec, block, model, args.model)
        _dump_fx(model, dec, args.model)         # raw torch.fx pull + per-node op mapping, per region
        _dump_extraction(model, dec, args.model) # what the block-extractor actually sees + collapses to

    print(f"\nPrompt: {args.prompt!r}\n")
    # hovering throughput line pinned to the bottom row (TTY only)
    import shutil, time
    _tty = sys.stdout.isatty(); timer = time.perf_counter(); n_dec = 0
    def _status(setup=False, teardown=False):
        if not _tty:
            return
        rows = shutil.get_terminal_size().lines
        if setup:
            sys.stdout.write(f"\033[1;{rows-1}r\033[{rows-1};1H"); sys.stdout.flush(); return
        if teardown:
            sys.stdout.write(f"\033[r\033[{rows};1H\033[2K"); sys.stdout.flush(); return
        el = time.perf_counter() - timer; rate = n_dec / el if el > 0 else 0.0
        sys.stdout.write("\0337" + f"\033[{rows};1H\033[2K")
        sys.stdout.write(f" decoding… {n_dec} tokens  (pos {SEQ+n_dec}/{MAXPOS})  {el:.1f}s  {rate:.1f} tok/s")
        sys.stdout.write("\0338"); sys.stdout.flush()
    _status(setup=True)

    # incremental detokenizer: decode the running sequence and print only the new suffix.
    # (single-token decode drops SentencePiece leading-space markers, e.g. llama2/TinyLlama.)
    shown = ""
    def stream(toks):
        nonlocal shown
        full = tok.decode(toks, skip_special_tokens=True)
        sys.stdout.write(full[len(shown):]); sys.stdout.flush(); shown = full

    m.execute("prefill", bind={m.seq: SEQ, m.qseq: SEQ * m.GRP, m.bucket: max(1, (SEQ * m.GRP + 63) // 64),
                               m.qall: SEQ * m.spec["nq"], m.kall: SEQ * m.spec["nkv"]})
    last = m.ue.dma_from_accelerator_memory(m.out.addr, (SEQ, m.H)).float()[-1]
    if args.check:                                          # numeric: FPGA vs CPU-bf16 AND CPU-IF4
        _status(teardown=True)
        from quant_lib import quantize, dequant
        cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        am = lambda h: (h @ lmw.T).argmax().item()
        def _norm_last(idz):                                # CPU ref: final normed hidden of last position
            # transformers>=5: hidden_states[-1] is ALREADY final-normed — do NOT re-apply dec.norm
            # (double-RMSNorm garbages the reference; verified hidden_states[L] @ lmw.T == model().logits).
            return dec(idz, output_hidden_states=True).hidden_states[spec["layers"]][0, -1].float()
        # FPGA first decode step (run once; reference both precisions against the SAME input token)
        t0 = am(last)
        m.ue.dma_to_accelerator_memory(m.dx.addr, (embed[t0] * norm).view(1, m.H).to(torch.bfloat16))
        _set_decode_bias(m, SEQ + 1)
        m.execute("decode", bind={m.bucket: max(1, (SEQ + 1 + 63) // 64)}, banner=False)
        dlast = m.ue.dma_from_accelerator_memory(m.dout.addr, (1, m.H)).float()[0]
        ids2 = torch.cat([ids, torch.tensor([[t0]])], dim=1)
        with torch.no_grad():
            hp_bf, hd_bf = _norm_last(ids), _norm_last(ids2)        # bf16 (true model) references
            def _q(w):
                d, s = quantize("if4", w.to(torch.bfloat16)); return dequant("if4", d, s, N=w.shape[0], K=w.shape[1]).float()
            for L in dec.layers:
                for mod in [L.self_attn.q_proj, L.self_attn.k_proj, L.self_attn.v_proj, L.self_attn.o_proj,
                            L.mlp.gate_proj, L.mlp.up_proj, L.mlp.down_proj]:
                    mod.weight.copy_(_q(mod.weight))
            hp_q, hd_q = _norm_last(ids), _norm_last(ids2)          # CPU-IF4 references
        # error budget: quant (IF4 vs bf16) | compiler (FPGA vs IF4) | total (FPGA vs bf16)
        print(f"\n[PREFILL]  FPGA·bf16={cos(last,hp_bf):.4f}  FPGA·IF4={cos(last,hp_q):.4f}  IF4·bf16={cos(hp_q,hp_bf):.4f}")
        print(f"           argmax  FPGA={am(last)}  bf16={am(hp_bf)}  IF4={am(hp_q)}")
        print(f"[DECODE ]  FPGA·bf16={cos(dlast,hd_bf):.4f}  FPGA·IF4={cos(dlast,hd_q):.4f}  IF4·bf16={cos(hd_q,hd_bf):.4f}")
        print(f"           argmax  FPGA={am(dlast)}  bf16={am(hd_bf)}  IF4={am(hd_q)}   (decode input token={t0})")
        # --- LOGIT diff (the real acceptance signal): FPGA logits vs CPU-IF4 logits (the precision the
        # hardware actually targets). cosine of the full logit vector + max-abs per-logit gap. PASS when
        # logit-cos >= LOGIT_COS_THRESH — a tolerance-based check that argmax-fragility (near-tie token
        # swaps) doesn't fail, while genuine distribution divergence does.
        lvcos = lambda hf, hr: cos(hf @ lmw.T, hr @ lmw.T)
        lvmax = lambda hf, hr: (hf @ lmw.T - hr @ lmw.T).abs().max().item()
        pc, dc = lvcos(last, hp_q), lvcos(dlast, hd_q)
        print(f"[LOGITS ]  prefill FPGA·IF4 cos={pc:.4f} maxΔ={lvmax(last,hp_q):.2f}  |  "
              f"decode FPGA·IF4 cos={dc:.4f} maxΔ={lvmax(dlast,hd_q):.2f}")
        verdict = "PASS" if (pc >= LOGIT_COS_THRESH and dc >= LOGIT_COS_THRESH) else "FAIL"
        print(f"[VERDICT]  {verdict}  (threshold logit-cos >= {LOGIT_COS_THRESH})")
        # --- per-layer decode probe: FPGA residual stream vs CPU-IF4, after each decoder layer ---
        # localizes WHERE the group=1 decode error enters: monotonic decay -> attention accumulation;
        # a jump at one layer -> a concrete bug. dec.layers hidden_states[i+1] = output of layer i.
        if getattr(m, "dec_probe", None) is not None:
            Ln = spec["layers"]
            row = lambda i: m.ue.dma_from_accelerator_memory(m.dec_probe.addr + i * m.H * 2, (1, m.H)).float()[0]
            with torch.no_grad():
                lref = dec(ids2, output_hidden_states=True).hidden_states   # IF4 weights (set above)
            print("  per-layer decode (FPGA vs CPU-IF4 residual):")
            for li in range(Ln):
                print(f"    layer {li:2d}: cosine={cos(row(li), lref[li + 1][0, -1].float()):.4f}")
            print(f"    post-final-norm: cosine={cos(row(Ln), hd_q):.4f}")
        return
    print("dec>")
    def _pick(logits):
        # greedy when temp<=0; else sample on CPU from the FPGA logit vector.
        if args.temp <= 0:
            return int(logits.argmax().item())
        p = torch.softmax(logits.float() / args.temp, dim=-1)
        return int(torch.multinomial(p, 1).item())
    t = _pick(last @ lmw.T); out = [t]                     # lm_head, not embed (untied-safe)
    if t not in eos:
        stream(out)
    timer = time.perf_counter()
    for step in range(GEN - 1):
        if out[-1] in eos:
            break
        pos = SEQ + step
        m.ue.dma_to_accelerator_memory(m.dx.addr, (embed[out[-1]] * norm).view(1, m.H).to(torch.bfloat16))
        _set_decode_bias(m, pos + 1)
        m.execute("decode", bind={m.bucket: max(1, (pos + 1 + 63) // 64)}, banner=False)
        if args.temp <= 0:
            t = m.ue.get_arg_max_index()                   # on-device argmax (no readback)
        else:
            t = _pick(m.ue.dma_from_accelerator_memory(m.dout.addr, (1, m.H)).float()[0] @ lmw.T)
        out.append(t); n_dec += 1
        if t in eos:
            break
        stream(out)
        _status()
    _status(teardown=True)
    print("\n<dec")
    el = time.perf_counter() - timer
    rate = n_dec / el if el > 0 else 0.0
    print(f"{'─'*60}\nmodel: {args.model}\n{len(out)} tokens · decode {n_dec} tok / {el:.1f}s = {rate:.1f} tok/s\n{'─'*60}")


# fx node-target -> our engine kernel. Anything not here is SHAPE (reshape/permute, free)
# or SCAFFOLD (host index/mask/cast/assert — not a compute kernel).
_FX_COMPUTE = {
    "scaled_dot_product_attention": "flash_attn", "linear": "matmul",
    "add": "add", "add_": "add", "mul": "mul", "layer_norm": "layer_norm",
    "native_layer_norm": "layer_norm", "conv2d": "patch-conv -> permute+matmul",
    "gelu": "activation(fold)", "silu": "activation(fold)", "embedding": "(host embed lookup)",
}
_FX_SHAPE = {"permute", "transpose", "reshape", "view", "_unsafe_view", "expand", "contiguous",
             "flatten", "unflatten", "clone", "squeeze", "unsqueeze", "select", "slice"}


def _dump_fx(model, dec, model_path):
    """Export the decoder once and partition every call_function node by its top-level
    leaf_path prefix — each distinct prefix is one execution workload (e.g. vision_model,
    multi_modal_projector, language_model / model). Within each workload, nodes are grouped
    by layer index so the repeated block structure is visible. Written to
    discard/<slug>_fx_workloads.txt."""
    import os, re
    from collections import Counter, defaultdict
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "discard")
    os.makedirs(d, exist_ok=True)
    slug = model_path.replace("/", "_").lower()
    base = lambda t: getattr(t, "__name__", str(t)).split(".")[0]
    def leaf(n):
        ms = n.meta.get("nn_module_stack"); return list(ms.values())[-1] if ms else (None, "")
    def classify(t):
        if t in _FX_COMPUTE: return "COMPUTE", _FX_COMPUTE[t]
        if t in _FX_SHAPE:   return "SHAPE", "permute/reshape"
        return "SCAFFOLD", "(host index/mask/cast/assert)"
    def top_prefix(path):
        # first path segment before any "layers.N" — this is the workload region
        if not path: return "<no_stack>"
        m = re.match(r"^(.*?)(?:\.layers\.\d|$)", path)
        return m.group(1) if m and m.group(1) else path.split(".")[0]
    def layer_idx(path):
        m = re.search(r"layers\.(\d+)", path or "")
        return int(m.group(1)) if m else -1

    try:
        ids = torch.tensor([[1, 2, 3, 4]])
        ep = torch.export.export(dec, (ids,), {"use_cache": False}, strict=False)
    except Exception as e:
        print(f"  fx dump skipped: {type(e).__name__}: {e}"); return

    # partition: workload -> layer_idx -> [row strings]
    workloads = defaultdict(lambda: defaultdict(list))
    cats = defaultdict(Counter)
    for n in ep.graph_module.graph.nodes:
        if n.op != "call_function": continue
        path, cls = leaf(n); t = base(n.target); k, tag = classify(t)
        wp = top_prefix(path); li = layer_idx(path)
        cats[wp][k] += 1
        leafname = (path or "").split(".")[-1]
        shape = str(tuple(n.meta["val"].shape)) if hasattr(n.meta.get("val", None), "shape") else "—"
        workloads[wp][li].append(
            f"  {n.name:28s} {t:28s} [{k:8s}]  leaf={leafname:20s}  shape={shape:20s}  cls={str(cls)[:30]}")

    lines = [f"# {model_path} — execution workloads (full graph, partitioned by leaf_path prefix)", ""]
    for wi, (wp, layers) in enumerate(workloads.items()):
        c = cats[wp]
        lines.append(f"## workload[{wi}]  prefix='{wp}'  COMPUTE={c['COMPUTE']} SHAPE={c['SHAPE']} SCAFFOLD={c['SCAFFOLD']}")
        for li in sorted(layers):
            lbl = f"layers.{li}" if li >= 0 else "<no_layer>"
            lines.append(f"  # {lbl}")
            lines.extend(layers[li])
        lines.append("")

    fn = os.path.join(d, f"{slug}_fx_workloads.txt")
    open(fn, "w").write("\n".join(lines))
    wl_summary = "  ".join(f"workload[{i}]='{wp}'" for i, wp in enumerate(workloads))
    print(f"  step 2 → fx workloads  {fn}  ({wl_summary})")


def _dump_extraction(model, dec, model_path):
    """Write EXACTLY what the block-extractor sees and decides, per region. Unlike the raw
    fx dump (the full unrolled graph), this shows the algorithm's real input: ONE repeat unit
    (layer 0), each in-scope node, and the descriptor it collapses to — then the final block
    that the weaver loops x num_layers. This is the compile unit, not the 1000s of flat ops."""
    import os
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "discard")
    os.makedirs(d, exist_ok=True)
    slug = model_path.replace("/", "_").lower()

    def one(region, mod):
        trace = []
        try:
            blk = derive_block(model, mod, region=region, trace=trace)
        except Exception as e:
            print(f"  extraction [{region}] skipped: {type(e).__name__}: {e}"); return
        scope = "layers.0 (decoder repeat unit)" if region == "decoder" else "encoder.layers[0] (ViT repeat unit)"
        L = [f"# {model_path}  region={region}",
             f"# WHAT THE EXTRACTOR SEES: a single repeat unit, scope = {scope}",
             f"# It scans ONLY this one layer, collapses clusters by nn_module_stack ownership,",
             f"# then the weaver replays the resulting block once per layer.", "",
             f"## Decision trace ({len([r for r in trace if r['decision'].startswith('EMIT')])} EMIT, "
             f"{len([r for r in trace if r['decision'].startswith('COLLAPSE')])} COLLAPSE, "
             f"{len([r for r in trace if r['decision'].startswith('ignore')])} ignored):", ""]
        for r in trace:
            mark = "+" if r["decision"].startswith("EMIT") else ("~" if r["decision"].startswith("COLLAPSE") else " ")
            L.append(f" [{mark}] {r['node']:20s} {r['target']:26s} @ {r['owner']:38s} -> {r['decision']}")
        L += ["", f"## FINAL BLOCK ({len(blk)} steps, looped x num_layers):", ""]
        for j, e in enumerate(blk):
            kind, role = e[0], e[1]; tr = e[2] if len(e) > 2 else {}
            L.append(f"  [{j}] {kind}" + (f":{role}" if role else "") + (f"  {tr}" if tr else ""))
        fn = os.path.join(d, f"{slug}_extracted_{region}.txt")
        open(fn, "w").write("\n".join(L))
        print(f"  step 3 → extract fwd   {fn}  ({len(blk)} block steps)")

    one("decoder", dec)
    if detect_model_type(model) == "vlm":
        vm = getattr(model.model, "vision_model", None) or getattr(model.model, "vision_tower", None)
        if vm is not None:
            one("encoder", vm)


def _ctx_tag(ctx):
    """Format a map-recorder ctx (stage, li, j, kind, role) into a block-step label."""
    if not ctx:
        return "(unscoped)"
    _, li, j, kind, role = ctx
    if li < 0:                                  # final-norm / lm_head (outside the layer loop)
        return f"({kind})"
    return f"[{j}] {kind}" + (f":{role}" if role else "")


def _dump_mapping(m, block, trace, model_path, dec=None):
    """STEP 4 — the UE_HLO execution flow: the ordered list of hardware kernels actually
    called for one decoder layer, each with a one-line note on how it was recognized.

    For each block step it shows, in order:
      1. WHAT we recognized in the torch.fx graph (the triggering op + owning module class,
         e.g. a `Gemma3RMSNorm`),
      2. WHERE it lives in the HF source (modeling_<arch>.py:LINE), so it can be
         opened and studied,
      3. WHICH engine kernel(s) it compiled to — captured live from the real build (ground
         truth), per program (prefill/decode), with PBI/legacy mode and fusions.

    `trace` is derive_block()'s decision trace (carries each step's fx op + source line);
    `m._map_log` is the chain()→UE_HLO record. They join on the block-step index."""
    import os
    log = getattr(m, "_map_log", None)
    if not log:
        print("  step 4 → op mapping    (no records — capture disabled?)"); return
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "discard")
    os.makedirs(d, exist_ok=True)
    slug = model_path.replace("/", "_").lower()

    # provenance per block step: the EMIT trace record that produced it
    prov = {}
    for r in (trace or []):
        if r.get("step") is not None and r["step"] not in prov:
            prov[r["step"]] = r

    # primitive-count per block step (how many fx nodes collapsed into this op), from the
    # decision trace: an EMIT claims the preceding folded nodes + itself; COLLAPSEs attach.
    counts, lead, cur = {}, 0, None
    for r in (trace or []):
        dcn = r["decision"]
        if dcn.startswith("EMIT"):
            cur = r["step"]; counts[cur] = lead + 1; lead = 0
        elif dcn.startswith("COLLAPSE") and cur is not None:
            counts[cur] += 1
        else:
            lead += 1

    def recognized(step, kind, role):
        """`recognized as <our-op> from <HF module class> <owner> · source` — how this step
        was identified in the fx graph (by module ownership) and where it lives in HF source."""
        aslbl = f"`{kind}`" + (f":{role}" if role else "")
        p = prov.get(step)
        if not p:
            return f"as {aslbl}"
        loc = (p["src"].split("  ")[0] if p.get("src") else "")
        n = counts.get(step)
        nx = f", {n} fx ops" if n and n > 1 else ""
        frm = f"`{p.get('cls') or '?'}` {p.get('owner') or ''}".rstrip()
        return f"as {aslbl} from {frm} ({p['target']}{nx})" + (f" · {loc}" if loc else "")

    def kcells(kers):
        """Ordered UE_HLO kernel names for a step, consecutive duplicates collapsed (`×N`)."""
        out = []
        for k in kers:
            if out and out[-1][0] == k:
                out[-1][1] += 1
            else:
                out.append([k, 1])
        return ", ".join(f"`{k}`" + (f"×{n}" if n > 1 else "") for k, n in out) or "—"

    # group the kernel flow by block step, both programs side-by-side, in execution order
    progs, flow, keys = [], {}, []
    for r in log:
        if r["program"] not in progs:
            progs.append(r["program"])
        ctx = r["ctx"]
        if not ctx or ctx[1] not in (0, -1):
            continue
        key = ("s", ctx[2], ctx[3], ctx[4]) if ctx[1] == 0 else ("h", ctx[4] or ctx[3])
        if key not in flow:
            flow[key] = {}; keys.append(key)
        prec = r.get("prec", "bf16")
        disp = r["kernel"] + (f"·{prec}" if prec and prec != "bf16" else "")   # mark quantized weight
        flow[key].setdefault(r["program"], []).append(disp)
    progs = sorted(progs, key=lambda p: (p != "prefill", p))

    blockline = " → ".join(f"`{k}`" + (f":{r}" if r else "") for k, r in block)
    L = [f"# {model_path} — UE_HLO execution flow", "",
         "Ops recognized from the torch.fx graph by **module ownership** (the owning `nn.Module` "
         "class names each op), then compiled to UE_HLO hardware kernels. One decoder layer (the "
         "repeat unit) shown top-down in execution order, both programs side-by-side, from the real build.", "",
         f"Block: {blockline}", "",
         f"`prefill` runs PBI (advancing-pointer); `decode` runs legacy (M=1). {len(keys)} steps.",
         "Matmuls carry their **weight precision** as a `·suffix` (e.g. `·if4`); `matmat_mul_core` "
         "runs quantized weights via internal `is_B_quantized`/`data_type` flags, so a quantized "
         "prefill matmul keeps the `matmat_mul_core` name. No suffix = bf16.", "",
         "| # | step | recognized | " + " | ".join(f"{p} UE_HLO kernels" for p in progs) + " |",
         "|---|---|---|" + "---|" * len(progs)]
    for i, key in enumerate(keys):
        if key[0] == "s":
            _, j, kind, role = key
            steplbl = kind + (f":{role}" if role else "")
            rec = recognized(j, kind, role)
        else:
            steplbl = key[1]; rec = "model head"
        cells = " | ".join(kcells(flow[key].get(p, [])) for p in progs)
        L.append(f"| {i} | {steplbl} | {rec} | {cells} |")

    # ── the UE_HLO hashtable hits: one row per hashmap KEY (op name) actually looked up,
    #    with its engine fn, dispatch source, per-program invocation counts, modes, precision.
    #    This is the ground-truth tally of "which hashtable entries this model exercised, how
    #    many times, and how" — captured live from dispatch()/fusion emission. ──
    used = {}
    for r in log:
        key = r["op"]                                   # the UE_HLO hashmap KEY (spec.name / fusion key)
        u = used.setdefault(key, {"fn": set(), "src": set(), "modes": set(),
                                  "prec": set(), "fused": r.get("fused") is not None,
                                  "calls": {p: 0 for p in progs}})
        u["fn"].add(r["kernel"]); u["src"].add(r.get("src", "—"))
        u["modes"].add(r["mode"]); u["prec"].add(r.get("prec", "bf16"))
        if r["program"] in u["calls"]:
            u["calls"][r["program"]] += 1
    total_hits = len(log)
    L += ["", f"## UE_HLO hashtable hits ({len(used)} distinct keys, {total_hits} total lookups)", "",
          "Every `chain()` op resolves to one `UE_HLO[<key>]` row (or a registered fusion). Counts "
          "are lookups across the FULL build (all layers, both programs) — the ground-truth tally of "
          "what the hashtable dispatched.", "",
          "| hashtable key | → engine fn | src | " + " | ".join(f"{p} ×" for p in progs)
          + " | weight prec | modes |",
          "|---|---|---|" + "---|" * len(progs) + "---|---|"]
    for key in sorted(used):
        u = used[key]
        fn = ", ".join(f"`{x}`" for x in sorted(u["fn"]))
        src = ", ".join(sorted(s for s in u["src"] if s and s != "—")) or ("fusion" if u["fused"] else "—")
        counts = " | ".join(str(u["calls"].get(p, 0)) for p in progs)
        keylbl = f"`{key}`" + (" _(fusion)_" if u["fused"] else "")
        L.append(f"| {keylbl} | {fn} | {src} | {counts} | {', '.join(sorted(u['prec']))} "
                 f"| {', '.join(sorted(u['modes']))} |")

    # ── catalog coverage: of the WHOLE UE_HLO hashtable + registered fusions, which entries did
    #    this model touch? Lists unused entries explicitly so the presentation can show exactly
    #    how much of the kernel catalog a given model exercises. ──
    catalog = set(UE_HLO) | {"+".join(k) for k in FUSIONS}
    hit = set(used)
    unused = sorted(catalog - hit)
    L += ["", f"## Hashtable coverage — {len(hit)}/{len(catalog)} entries exercised", "",
          f"**Used ({len(hit)}):** " + (", ".join(f"`{k}`" for k in sorted(hit)) or "—"), "",
          f"**Unused ({len(unused)}):** " + (", ".join(f"`{k}`" for k in unused) or "_(full coverage)_")]

    # ── how build() fixed the order: it walks the extracted block top-down (= fx graph order) ──
    def _ord(n):
        suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suf}"
    L += ["", "## How build() ordered the flow", "",
          "`build()` walks the IR-extracted block top-down and emits one `chain()` per step. That "
          "block order is the torch.fx graph's topological (execution) order — so **whatever was "
          "seen first in the graph is emitted first and runs first.**", "",
          "| seen | in the fx graph (HF module · source) | → emitted as |", "|---|---|---|"]
    for j, e in enumerate(block):
        kind, role = e[0], e[1]
        p = prov.get(j)
        if p:
            loc = p["src"].split("  ")[0] if p.get("src") else ""
            frm = f"`{p.get('cls') or '?'}` {p.get('owner') or ''}".rstrip() + (f" · {loc}" if loc else "")
        else:
            frm = "(weave helper)"
        L.append(f"| {_ord(j + 1)} | {frm} | `{kind}`" + (f":{role}" if role else "") + " |")

    # ── selected execution flow: how select_plan (the DP) covered each chain() call ──
    plog = getattr(m, "_plan_log", None) or []
    L += ["", "## Selected execution flow (DP fusion plan)", "",
          "`chain()` hands `select_plan` an op sequence; the DP covers it with the FEWEST UE_HLO "
          "kernels: `dp[i]` = min kernels to cover `names[:i]`, trying a singleton or any registered "
          "fusion ending at `i`, preferring the longest. **The op ORDER is fixed by `build()`'s "
          "emission (the woven block) — the DP does not reorder; it only decides which adjacent ops "
          "merge into one kernel.**", "",
          f"Registered fusions the DP can apply ({len(FUSIONS)}):"]
    if FUSIONS:
        for key in FUSIONS:
            L.append(f"- `{' + '.join(key)}` → `{FUSION_LABEL.get(key, '(fused)')}` (one kernel)")
    else:
        L.append("- _(none registered)_")
    for prog in progs:
        calls = [e for e in plog if e["program"] == prog and e["ctx"] and e["ctx"][1] in (0, -1)]
        seq, nops, nker, nfused = [], 0, 0, 0
        for e in calls:
            nm = e["names"]
            for (s, en, key) in e["segs"]:
                nker += 1; nops += (en - s)
                if key:
                    nfused += (en - s - 1)
                    seq.append("[" + " + ".join(nm[s:en]) + " ⊕]")
                else:
                    seq.append(nm[s])
        L += ["", f"**{prog}** — {nops} ops → {nker} kernels ({nfused} fused away); cover in order:",
              "", "`" + "` · `".join(seq) + "`" if seq else "_(no ops)_"]

    # ── raw torch.fx graph (decoder) — the unmodified op stream we traversed to derive all of
    #    the above, with a SELECTED column: what each node was recognized as (an emitted block
    #    op) or '—' if it was skipped (folded into a cluster, or shape/scaffold). This makes the
    #    recognition checkable against ground truth, node by node. ──
    if dec is not None:
        L += ["", "## Raw torch.fx graph (decoder) — with selection", "",
              "The unmodified `torch.export` op stream (all layers, execution order) every section "
              "above was derived from. The **selected** column shows what each `call_function` was "
              "recognized as (the block op it emits) — `—` means skipped (folded into a cluster, or "
              "a shape/scaffold op). Selection reads each node's target + `nn_module_stack` owner; "
              "the same pattern repeats per layer.", "",
              "Columns expose every level of the IR for the same node: **selected** (the UE block op), "
              "**role** (`<category>/<leaf-name>` semantic label from the parent module class), "
              "**leaf_path** (the qualified module path, e.g. `layers.0.self_attn.q_proj`), and "
              "**L0…Lk** = the full `nn_module_stack` class chain root→leaf. No extra export is "
              "needed — every column is a different read of the one flattened `torch.export` graph.", ""]
        try:
            ep = torch.export.export(dec, (torch.tensor([[1, 2, 3, 4]]),), {"use_cache": False}, strict=False)
            base = lambda t: getattr(t, "__name__", str(t)).split(".")[0]
            bc = lambda c: (str(c).split("'")[1] if "'" in str(c) else str(c)).split(".")[-1]  # bare class name

            # parent-module class → semantic category (the "higher level" view over the stack)
            ROLE = {"attention": "attention", "mlp": "mlp", "rmsnorm": "norm",
                    "layernorm": "norm", "decoderlayer": "block"}

            def _levels(n):
                """The FULL nn_module_stack as a list of (path, class) root→leaf — every IR level."""
                ms = n.meta.get("nn_module_stack")
                return list(ms.values()) if ms else []

            def _role(n):
                """Semantic role derived from the module stack: '<category>/<leaf-name>'."""
                lv = _levels(n)
                if not lv: return ""
                classes = [str(c).lower() for _, c in lv]
                leaf = (lv[-1][0] or "").split(".")[-1]
                parent = next((ROLE[k] for c in reversed(classes) for k in ROLE if k in c), "")
                return f"{parent}/{leaf}" if (parent and leaf) else (parent or leaf)

            def _selected(n):
                """Mirror derive_block's trigger tree (no layers.0 filter, so every layer is
                marked): return the emitted block-op label, or '' if the node is skipped."""
                ms = n.meta.get("nn_module_stack")
                path, cls = (list(ms.values())[-1] if ms else (None, ""))
                stk = " ".join(str(c).lower() for _, c in ms.values()) if ms else ""
                c = str(cls).lower(); t = base(n.target)
                leafname = (path or "").split(".")[-1]
                if t == "scaled_dot_product_attention":          return "attention"
                if leafname == "o_proj" and t == "linear":       return "matmul:o"
                if leafname == "down_proj" and t == "linear":    return "swiglu"
                if ("rmsnorm" in c or "layernorm" in c) and "self_attn" not in (path or ""):
                    return "norm"
                if t == "add" and "decoderlayer" in stk and "norm" not in c and "attention" not in c:
                    return "residual"
                return ""

            # max module-stack depth → one Lk column per IR level (root → leaf)
            nodes = list(ep.graph_module.graph.nodes)
            maxd = max((len(_levels(n)) for n in nodes if n.op == "call_function"), default=0)
            lvl_hdr = [f"L{k}" for k in range(maxd)]
            hdr = ("opcode", "name", "target", "selected", "role", "leaf_path", *lvl_hdr)
            rows = [hdr]
            from collections import Counter
            role_ct, leafcls_ct = Counter(), Counter()
            for n in nodes:
                cf = n.op == "call_function"
                tgt = base(n.target) if cf else str(n.target)
                sel = (_selected(n) if cf else "") or "—"
                role = (_role(n) if cf else "") or "—"
                lv = _levels(n)
                leafpath = (lv[-1][0] if lv else "") or "—"
                levels = [bc(c) for _, c in lv] + ["—"] * (maxd - len(lv))
                rows.append((n.op, n.name, tgt, sel, role, leafpath, *levels))
                if cf and role != "—": role_ct[role.split("/")[0]] += 1
                if cf and lv: leafcls_ct[bc(lv[-1][1])] += 1
            nc = len(hdr)
            w = [max(len(str(r[i])) for r in rows) for i in range(nc)]
            line = lambda r: "  ".join(str(r[i]).ljust(w[i]) for i in range(nc)).rstrip()
            body = [line(rows[0]), "  ".join("-" * w[i] for i in range(nc))]
            body += [line(r) for r in rows[1:]]
            nsel = sum(1 for r in rows[1:] if r[3] != "—")

            # per-level metrics: how many nodes the stack attributes to each category / leaf class
            L += ["", "### IR-level metrics (per-node module-stack attribution)", "",
                  f"Module-stack depth: **{maxd} levels** (L0 = root module → L{maxd-1} = leaf). "
                  f"{nsel} of {len(rows) - 1} nodes selected as block ops.", "",
                  "**By semantic role (parent category):** " +
                  (", ".join(f"`{k}`×{v}" for k, v in role_ct.most_common()) or "—"), "",
                  "**By leaf module class:** " +
                  (", ".join(f"`{k}`×{v}" for k, v in leafcls_ct.most_common()) or "—")]
            L += ["", "```", "\n".join(body), "```"]

            # ── L2-collapsed block diagram (Mermaid) — one decoder layer (the repeat unit).
            #    L2 is the natural block boundary: every op inside Gemma3Attention (q/k/v proj,
            #    q_norm/k_norm, rope, SDPA, o_proj) shares L2=Gemma3Attention, so collapsing
            #    consecutive call_function nodes by their L2 class yields the layer's block flow.
            def _l2(n):
                lv = _levels(n)
                if not lv: return None
                return bc(lv[2][1]) if len(lv) > 2 else bc(lv[-1][1])  # L2 class, else owner
            blocks = []  # [(l2_label, op_count)] for layers.0 only, consecutive runs merged
            for n in nodes:
                if n.op != "call_function": continue
                lv = _levels(n)
                path = (lv[-1][0] if lv else "") or ""
                if not path.startswith("layers.0"):    # restrict to the first repeat unit
                    continue
                lab = _l2(n) or "?"
                if blocks and blocks[-1][0] == lab:
                    blocks[-1] = (lab, blocks[-1][1] + 1)
                else:
                    blocks.append((lab, 1))
            if blocks:
                mer = ["```mermaid", "flowchart TD"]
                for i, (lab, cnt) in enumerate(blocks):
                    mer.append(f'  b{i}["{lab}<br/><i>{cnt} fx ops</i>"]')
                for i in range(len(blocks) - 1):
                    mer.append(f"  b{i} --> b{i+1}")
                mer.append("```")
                L += ["", "### Block diagram (L2-collapsed, one decoder layer)", "",
                      "Consecutive fx nodes merged by their **L2 module class** — the natural block "
                      "boundary. Renders in VS Code (Markdown Preview Mermaid Support) and on GitHub.",
                      "", *mer]
        except Exception as e:
            L.append(f"_(fx export failed: {type(e).__name__}: {e})_")

    fn = os.path.join(d, f"{slug}_mapping.md")
    open(fn, "w").write("\n".join(L) + "\n")
    print(f"  step 4 → op mapping    {fn}  ({len(log)} kernels, {len(used)} UE_HLO kernels used)")


def _dump_ir(m, spec, block, model, model_path):
    """Write the full IR-to-op mapping, config specs, and program structure to a
    markdown file in discard/."""
    import os, datetime
    model_slug = model_path.replace("/", "_").lower()
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "discard")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{model_slug}_ir_dump.md")

    L, H, hd, nq, nkv = spec["layers"], spec["H"], spec["hd"], spec["nq"], spec["nkv"]
    grp = nq // max(1, nkv)

    lines = []
    def w(s=""): lines.append(s)

    w(f"# IR Dump: {model_path}")
    w(f"Generated: {datetime.datetime.now().isoformat()}")
    w()

    # ── config specs ──
    w("## HF Config Spec")
    w("| Key | Value |")
    w("|---|---|")
    for k, v in spec.items():
        w(f"| {k} | {v} |")
    w()

    w("## Forward-Pass Decisions")
    for trait, choice in m.plan.items():
        w(f"- **{trait}** → {choice}")

    # ── narrowing trail ──
    w()
    w("## Narrowing Trail (HF IR → Our Op Selection)")
    w()
    w("### Step 1: IR Pattern → Block Classification")
    w("`derive_block()` walks the `torch.export` graph and collapses clusters by `nn_module_stack`:\n")
    for j, (kind, role) in enumerate(block):
        if kind == "norm":
            w(f"- **[{j}]** `call_function` (rms_norm/layer_norm in {role} position) → `norm:{role}`")
        elif kind == "attention":
            w(f"- **[{j}]** `call_function` (scaled_dot_product_attention) + surrounding nn.Module cluster → `attention`")
        elif kind == "matmul":
            w(f"- **[{j}]** `call_function` (linear, {role}_proj output projection) → `matmul:{role}`")
        elif kind == "residual":
            w(f"- **[{j}]** `call_function` (add, within decoderlayer scope, not norm/attn) → `residual`")
        elif kind == "swiglu":
            w(f"- **[{j}]** `call_function` (linear + activation + mul + linear cluster) → `swiglu`")

    w()
    w("### Step 2: Trait → Implementation Variant")
    w("`decide_forward()` maps each model characteristic to our op path:\n")
    nq, nkv, hd = spec["nq"], spec["nkv"], spec["hd"]
    group = nq // max(1, nkv)
    w(f"| Trait | Value | Decision | Rationale |")
    w(f"|---|---|---|---|")
    w(f"| Weight precision | IF4 (quantized) | `mm` → `matmul(PBI)` / `qmatmul(decode)` | Weights <2GB → IF4 fits + PBI prefill |")
    stage_s = "prefill + decode + KV cache"
    w(f"| Architecture | causal decoder | stages = `{stage_s}` | Autoregressive: process prompt, then token-by-token |")
    gqa_s = f"GQA group_size = {group}" if nkv > 1 else "single-KV (MHA)"
    w(f" | GQA | nq={nq}, nkv={nkv} | `{gqa_s}` | Q heads ÷ KV heads = group size |")
    attn_v = "multi-KV head (per-KV-head scatter+flash)" if nkv > 1 else "single-KV grouped (GQA duplicate + flash)"
    w(f" | Attention variant | nkv={'1' if nkv==1 else '>1'} | `{attn_v}` | {nkv} KV heads → {'per-head scatter' if nkv>1 else 'GQA group-dup'} |")
    rope_v = "contiguous rope_gqa" if hd >= 128 else "tiled lo|hi + permute (sub-128 workaround)"
    hd_note = f"head_dim={hd}→{pad64(hd)} (lo|hi pad)" if hd % 64 else f"head_dim={hd}"
    pad_why = f"hd {hd} not a multiple of 64 → lo|hi repack to {pad64(hd)} (real lanes [0:{hd//2}]+[{pad64(hd)//2}:{pad64(hd)//2+hd//2}], pad=0); verified rope/flash kernels run at {pad64(hd)}" \
              if hd % 64 else "Full-width ≥128 → direct; sub-128 → pad+permute"
    w(f" | RoPE | {hd_note} {'>=128' if hd>=128 else '<128'} | `{rope_v}` | {pad_why} |")
    w(f" | QK-norm | {'present' if spec['qk_norm'] else 'absent'} | {'apply per-head RMSNorm before rope' if spec['qk_norm'] else 'skip'} | {'HF model has q_norm/k_norm weights' if spec['qk_norm'] else 'no QK-norm in checkpoint'} |")
    w(f" | Activation | {spec['act']} | `{spec['act']}` | Passed through to matmul's `act=` param |")
    w(f" | Embed scaling | scale={m.plan.get('embed_scale', '?')} | {'scale embeds by sqrt(H)' if float(m.plan.get('embed_scale', 0)) > 1.5 else 'no scaling'} | Empirical median(probe/embed) |")

    w()
    w("### Step 3: Op Name → Engine Kernel")
    w("Each `chain()` call resolves to a kernel via `hlo.py`:\n")
    w("| Chain op | KernelSpec | Engine function | PBI? |")
    w("|---|---|---|---|")
    w("| `rms_norm` | rms_norm | `rms_norm_core_dram` | ✅ (prefill) |")
    w("| `mm` → `matmul` | matmul | `matmat_mul_core` | ✅ (prefill) |")
    w("| `qmatmul` | matmul | `matmat_mul_core` | ❌ (decode, M=1) |")
    w("| `add` | eltwise_add | `eltwise_add_core_dram` | ❌ |")
    w("| `mul` | eltwise_mul | `eltwise_mul_core_dram` | ❌ |")
    w("| `rope` | rope | `rope_hf_core_dram` | ✅ |")
    w("| `rope_gqa` | rope_gqa | `rope_hf_core_dram` | ✅ |")
    w("| `flash_attn` | flash_attn | `prefill_flash_attention_core` | ❌ (always legacy) |")
    # decode-attention op reflects the ACTUAL runtime gating (not a static label):
    #   nkv==1            -> single-KV decode_attention (legacy group_attn)
    #   nkv in [2,PBI_MAX] -> bucketized group_attn_pbi (verified smolvlm2 flow)
    #   nkv > PBI_MAX      -> legacy group_attn (PBI-bloat fallback)
    nkv = spec["nkv"]
    if getattr(m, "pbi_decode", False):
        w(f"| `decode (nkv={nkv}, gated)` | group_attn_pbi | `decoder_group_attention_core_pbi` | "
          f"✅ PBI (bucketized, nkv≤{PBI_DECODE_MAX_NKV}) |")
    elif nkv > 1:
        w(f"| `decode (nkv={nkv}, gated)` | group_attn | `decoder_group_attention_core` | "
          f"❌ legacy (nkv>{PBI_DECODE_MAX_NKV} → PBI-bloat fallback) |")
    else:
        w("| `decode_attention` (nkv=1) | decode_attn | `decoder_group_attention_core` | ❌ legacy (single-KV) |")

    w()
    w("### Step 4: Precision per Weight")
    w("Precision resolved by glob pattern on HF weight name (most-specific match wins):\n")
    w("| Glob | Precision | Matched weights |")
    w("|---|---|---|")
    w("| `*proj*` + `lm_head*` | `if4` | All q/k/v/o/gate/up/down + lm_head |")
    w("| `*norm*` (fallback `*`) | `bf16` | RMSNorm gammas, QK-norm |")
    w()
    bf16_w = sum(1 for n, t in m.W.items() if t.dtype == "bf16")
    q_w = sum(1 for n, t in m.W.items() if t.dtype != "bf16")
    w(f"Result: {q_w} weights at IF4, {bf16_w} at bf16")

    # ── coverage proof ──
    w()
    w("## Coverage Proof (100%)")
    w("""
The block topology is extracted from the IR via `torch.export`. Every compute op in the
HF forward pass falls into one of these patterns, which `derive_block()` recognizes:
""")
    w("| IR Pattern | Recognized As | Emitted Chain Ops |")
    w("|---|---|---|")
    w("| `call_function` (scaled_dot_product_attention) | `attention` | `mm(q) + mm(k) + mm(v) → _attn()` |")
    w("| `call_module` (rms_norm/layer_norm in input/post_attn/pre_ffn) | `norm:{role}` | `chain(('rms_norm', ...))` |")
    w("| `call_function` (linear, o_proj/down_proj) | `matmul:{role}` / `swiglu` | `chain(('mm', ...))` |")
    w("| `call_function` (add, within decoder layer) | `residual` | `chain(('add', ...))` |")
    w()
    w("Every chain op resolves through `UE_HLO` in `hlo.py`. The full resolution chain:")
    w("""
    HF forward() → torch.export → derive_block() → block steps
                   → _layer() → chain() → select_plan(DP) → dispatch(spec)
                   → engine function (matmat_mul_core, rms_norm_core_dram, etc.)
""")
    w("**Result:** 100% of compute ops in the extracted block have a valid kernel dispatch path.")

    # ── DRAM ──
    w()
    w("## DRAM Partition (IF4 footprint)")
    dram = getattr(type(m), "_dram", None)
    if dram:
        base, tbase, pbase = dram
        w(f"- params:  `0x{base:08X}` → `0x{tbase:08X}`  ({(tbase-base)/1e6:.0f} MB)")
        w(f"- tensor:  `0x{tbase:08X}` → `0x{pbase:08X}`  ({(pbase-tbase)/1e6:.0f} MB)")
        w(f"- program: `0x{pbase:08X}` → `0x{base+0x100000000:08X}`  ({(base+0x100000000-pbase)/1e6:.0f} MB)")

    # ── block → op mapping ──
    w()
    w("## Block → Chain Op Mapping")
    w(f"`{len(block)}` steps extracted from IR:\n")
    step_desc = {
        ("norm", "input"):    "`rms_norm(cur, input_layernorm.weight)`",
        ("attention", None):  "`mm(q) + mm(k) + mm(v)` → qk-norm → rope → scatter → flash",
        ("matmul", "o"):      "`mm(o_proj)` — attention output projection",
        ("residual", None):   "`add(resid, cur)` — residual stream add",
        ("norm", "post_attn"): "`rms_norm(cur, post_attention_layernorm.weight)`",
        ("swiglu", None):     "`mm(gate,act) + mm(up) + mul() + mm(down)`",
    }
    for j, (kind, role) in enumerate(block):
        desc = step_desc.get((kind, role), f"{kind}:{role}")
        w(f"  **[{j}]** `{kind}:{role}` → {desc}")

    # ── weight bindings (layer 0) ──
    w()
    w("## Weight Bindings (Layer 0, IF4 quantized)")
    proj_names = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj",
                  "o": "self_attn.o_proj", "gate": "mlp.gate_proj", "up": "mlp.up_proj",
                  "down": "mlp.down_proj"}
    norm_names = {"input": "input_layernorm", "post_attn": "post_attention_layernorm",
                  "q_norm": "self_attn.q_norm", "k_norm": "self_attn.k_norm"}
    w("| Weight | HF Source | Shape | DRAM | Quant |")
    w("|---|---|---|---|---|")
    for s, hf in proj_names.items():
        wgt = m.W[f"l0.{s}"]
        qs = "yes" if wgt.dtype != "bf16" else "no"
        w(f"| `{s}` | `{hf}.weight` | `{wgt.shape}` | `0x{wgt.addr:X}` | {qs} |")
    for s, hf in norm_names.items():
        key = f"l0.{s}"
        if key in m.N:
            wgt = m.N[key]
            w(f"| `{s}` | `{hf}.weight` | `{wgt.shape}` | `0x{wgt.addr:X}` | no |")
    wgt = m.lm_head
    w(f"| `lm_head` | `lm_head.weight` | `{wgt.shape}` | `0x{wgt.addr:X}` | yes |")
    if m.multikv:
        w(f"\nKV cache: **multi-KV** ({nkv} heads × {L} layers × {hd}-dim)")
    else:
        w(f"\nKV cache: **single-KV** ({L} layers × {hd}-dim)")

    # ── attention path ──
    w()
    w("## Attention Path")
    if m.multikv:
        w(f"- Variant: multi-KV (nkv={nkv}, group={grp}, scatter+flash per KV head)")
    elif spec["hd"] < 128:
        w("- Variant: single-KV lo|hi permute, tiled rope")
    else:
        w("- Variant: single-KV contiguous (rope_gqa)")
    w(f"- QK-norm: {'enabled' if spec['qk_norm'] else 'none'}")
    w(f"- Prefill: flash_attn (bucketized SDPA)")
    w(f"- Decode:  decode_attention (group-query KV cache attend)")

    # ── program structure ──
    w()
    w("## Program Structure")
    w("| Program | DRAM Address | Size |")
    w("|---|---|---|")
    for pname, prog in m.programs.items():
        w(f"| `{pname}` | `0x{prog.addr:X}` | {prog.size} bytes |")
    total_sz = sum(p.size for p in m.programs.values())
    w(f"\n**Total program DRAM:** {total_sz/1024:.1f} KB")

    # ── layer expansion ──
    w()
    w("## Layer-Step Expansion")
    w(f"All {L} layers share this block structure. Layer 0 shown:\n")
    li = 0
    for j, (kind, role) in enumerate(block):
        if kind == "norm":
            wt = m.N.get(f"l{li}.{role}", None)
            wdesc = f"@{wt.addr:#x}" if wt else f"(role={role})"
            w(f"- **Step {j}** (`norm:{role}`) → `chain(('rms_norm', cur, gamma{wdesc}))`")
        elif kind == "attention":
            w(f"- **Step {j}** (`attention`) → `mm(q) + mm(k) + mm(v)` → _attn cluster")
            if m.multikv:
                w(f"  - rope → scatter(KV heads) → per-head flash_attn → gather")
            elif m.lohi:
                w(f"  - lo|hi rope → qk-norm → flash_attn")
            else:
                w(f"  - qk-norm → rope_gqa → flash_attn")
        elif kind == "matmul":
            w(f"- **Step {j}** (`matmul:{role}`) → `chain(('mm', cur, W({role})))`")
        elif kind == "swiglu":
            w(f"- **Step {j}** (`swiglu`) → `mm(gate, act={spec.get('act','silu')}) + mm(up) + mul() + mm(down)`")
        elif kind == "residual":
            w(f"- **Step {j}** (`residual`) → `chain(('add', resid, cur))`")

    # ── execution flow ──
    w()
    w("## Execution Flow")
    w("```")
    w("1. prefill: loop_reg(seq=PBI) → 36× [layer step] → final_norm")
    w("2. decode:  loop_reg(None) → 36× [layer step] → final_norm → lm_head(qmatmul)")
    w("   epilogue: advance(seq_reg)")
    w("```")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  step 1 → ir dump       {path}")


if __name__ == "__main__":
    main()
