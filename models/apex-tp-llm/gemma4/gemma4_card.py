"""Gemma 4 26B-A4B on the Apex cards, tensor-parallel: one engine = one shard of every layer.

All model compute runs on the cards. The host only moves bytes, issues commands, picks the 8 expert
indices from the router logits the cards report, and picks the next token.

Per layer four segments, each ending in a halt (D needs no expert pick: the host picks while it runs):
  A  attention : [finish the previous layer's FFN: sum the dense and the MoE shares separately,
                 post_ff_norm_1 / _2, add, post_ff_norm, residual add, * layer_scalar]
                 rmsnorm, q/k(/v), q_norm/k_norm (v_norm), RoPE, KV write, softmax(q.K^T + mask) for
                 both heads in one matmul, P.V in one matmul, o projection          -> share (H)
                 (shards without attention heads only finish the FFN; their share stays 0)
  R  router    : [sum the attention shares, post_attention_norm, residual add]
                 rmsnorm (no weight), router logits (128, every shard)              -> logits
  D  dense FFN : this shard's dense blocks                                       -> SHARE_E[0:H]
  E  experts   : softmax over the 8 logits the host copied back;
                 for each of the 8 picked experts a call into its precompiled subroutine
                                                                 -> share (2H: dense | MoE)
then the LM head (this shard's vocabulary rows). 4 * 30 + 1 = 121 segments per token.

Global layers (head_dim 512, K = V, partial RoPE over pairs (i, i + 256), i < 64) are handled by
permuting each head's dimensions so the 128 rotated ones come first ([0:64] + [256:320], then the
rest); q, k, q_norm, k_norm and the o-projection columns get the same permutation, which leaves
q.k and the attention output unchanged.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

BF16 = 2
VEC = 64
NEG = -1.0e30                    # bf16 -inf stand-in for masks (exp underflows to 0)
PV_CH = 512                      # T_ext is rounded up to a multiple of this
TOPK = 8


def bf16_bits(a):
    """float32 values -> bf16 bit patterns (int16), round to nearest even, like torch's cast."""
    u = np.asarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16).view(np.int16)


NEG_BF = bf16_bits([NEG])[0]
_PACK8H = __import__("struct").Struct("<8h").pack

# ISA registers (the vendor allocator hands out 1..~30 as scratch; regs 1-3 are the port's)
R_ROPE, R_POS_K1, R_POS_K2 = 40, 41, 48          # byte offsets: RoPE row, K/V row (head_dim 256 / 512)
R_REBUILD = 43                                   # host: 1 when V^T must be rebuilt (R_LEN just grew)
R_MASK_NEW, R_MASK_OLD = 44, 45                  # byte offsets of the new / expired mask block in a mask row
R_DATA = 46                                      # never executed: data slots
R_BLK_V1, R_BLK_V2 = 61, 47                      # byte offset of this position's 64-row V block (256 / 512)
R_VT_STRIDE, R_VT_IN, R_VT_OUT, R_LEN, R_BLK_VT = 57, 58, 59, 60, 62
# sliding layers keep K / V / V^T in a ring of RING rows (slot = pos % RING): their own length, V^T block,
# rebuild flag and new-mask block registers (R_POS_K1, R_BLK_V1, R_MASK_NEW, R_MASK_OLD are slot-based)
R_LEN_S, R_BLK_VT_S, R_MASK_NEW_G, R_REBUILD_S = 42, 39, 38, 37
R_RET = 50
R_EXP = (51, 52, 53, 54, 55, 56, 63, 49)
R_W = 36                                         # address of the one-hot that picks the current expert weight
R_XT = 35                                        # block program: byte offset of the current token's row in XS
REG_FIXED_MIN = 35                               # every register >= this is ours
# fused expert path: the weight scales the 64-128 hidden values inside the expert and the down projection
# accumulates through its bias input straight into the MoE share (no per-expert H-sized copies / adds);
# the FFN finish sums the dense and MoE shares in one 2H pass
FUSE = os.environ.get("GEMMA4_FUSE", "1") == "1"
RING_EXTRA = 64                                  # ring = sliding window + one block

# SRAM scratch for the RoPE (128-byte aligned slots)
SA_X_LO, SA_X_HI, SA_OUT_LO, SA_OUT_HI, SA_TMP, SA_PASS = 0x40000, 0x40200, 0x40400, 0x40600, 0x40800, 0x40A00
SB_COS, SB_NSIN, SB_SIN, SB_TMP = 0x80000, 0x80200, 0x80400, 0x80600


def load_base_port(name, work=None):
    """The Qwen3.5 port module: its engine class (eps-padded RMSNorm, identity matrix, fast IF4 GEMV
    wrapper, bounded allocators) and helpers are model-independent enough to build on."""
    import importlib.util
    work = os.path.expanduser(work or os.environ.get("APEX_TP_WORK", "~/apex_tp"))
    path = os.path.join(work, "models", "q35_9b", "qwen3.5_2b_test.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod, os.path.join(work, "models", "q35_9b", "qwen3.5_2b_config.json")


# ---------------------------------------------------------------- FP4 pair helpers (host-side layout only)
def pair_rows(pair, N, K, idx):
    """Rows idx of an IF4 [N, K] weight (scales [N*K/64] bf16, data [N*K/2] uint8)."""
    s, d = pair
    idx = torch.as_tensor(idx)
    return (s.view(N, K // 64)[idx].reshape(-1).contiguous(), d.view(N, K // 2)[idx].reshape(-1).contiguous())


def pair_col_blocks(pair, N, K, bperm):
    """Columns of an IF4 [N, K] weight reordered by whole 64-blocks along K."""
    s, d = pair
    bperm = torch.as_tensor(bperm)
    return (s.view(N, K // 64)[:, bperm].reshape(-1).contiguous(),
            d.view(N, K // 64, 32)[:, bperm, :].reshape(-1).contiguous())


def pair_cat_rows(pairs):
    return (torch.cat([p[0] for p in pairs]), torch.cat([p[1] for p in pairs]))


def pair_cat_k(pairs, N, Ks):
    """[N, K1] ++ [N, K2] along K."""
    s = torch.cat([p[0].view(N, k // 64) for p, k in zip(pairs, Ks)], 1).reshape(-1)
    d = torch.cat([p[1].view(N, k // 2) for p, k in zip(pairs, Ks)], 1).reshape(-1)
    return s.contiguous(), d.contiguous()


def global_perm(hd, rot_pairs):
    """Head-dimension order for the global layers: the rotated dims first, as pairs (i, i + rot_pairs)."""
    half = hd // 2
    return torch.cat([torch.arange(0, rot_pairs), torch.arange(half, half + rot_pairs),
                      torch.arange(rot_pairs, half), torch.arange(half + rot_pairs, hd)])


def rope_tables(cfg, max_pos):
    """cos / -sin / sin tables, [max_pos, 128] each, for the sliding layers (128 frequency pairs over
    head_dim 256, theta 1e4) and the global layers (64 pairs over head_dim 512, theta 1e6, padded to
    128), exactly as transformers' default and proportional RoPE compute them (host-side preparation)."""
    pos = torch.arange(max_pos, dtype=torch.float32)
    out = {}
    for kind, hd, rp, key in (("s", cfg["head_dim"], None, "sliding_attention"),
                              ("g", cfg["global_head_dim"], None, "full_attention")):
        rope = cfg["rope_parameters"][key]
        theta = rope["rope_theta"]
        prop = rope.get("partial_rotary_factor", 1.0)
        n = int(prop * hd // 2)
        inv = 1.0 / (theta ** (torch.arange(0, 2 * n, 2, dtype=torch.float32) / hd))
        fr = torch.outer(pos, inv)
        tab = {}
        for name, v in (("cos", fr.cos()), ("nsin", -fr.sin()), ("sin", fr.sin())):
            t = torch.zeros(max_pos, 128)
            t[:, :n] = v
            tab[name] = t.to(torch.bfloat16)
        out[kind] = (n, tab)
    return out


class Gemma4Shard:
    """One engine holding one shard. Build every engine of a card before calling prepare() on any
    (constructing an engine self-tests the first 16 KB of DRAM)."""

    def __init__(self, T, qwen_cfg, dev, engine, cdir, wdir, plan, max_ctx, window=None, layers=None,
                 router_part=(0, 1), max_block=0):
        """router_part = (i, n): this engine computes router rows [i*128/n, (i+1)*128/n) (the engines of a
        card split the router; the other engine's logits land in engine 0's LOGITS, see compile()).
        max_block: the longest image block (bidirectional in the sliding layers, see compile_block()); the
        sliding rings grow by it so a block's first token still finds its whole window."""
        self.router_part = router_part
        self.T, self.U = T, T.user_dma_core
        self.fast_h2c = None
        self.prof_w = [0, 0]
        T.set_dma_device(dev)
        T.DMA_DEVICE_H2C = self.U.DMA_DEVICE_H2C
        T.DMA_DEVICE_C2H = self.U.DMA_DEVICE_C2H
        self.U.configure_clock_from_hardware()
        self.ue = T.Qwen3_5_2b_UnifiedEngine(device="cpu", config_path=qwen_cfg,
                                             BASE_ADDR=self.U.UE_0_BASE_ADDR + engine * 0x10000, **(window or {}))
        alloc = self.ue.alloc_isa_reg

        def guarded_alloc():
            r = alloc()
            assert r < REG_FIXED_MIN, "vendor scratch register %d reaches the fixed registers (>= %d)" % (
                r, REG_FIXED_MIN)
            return r
        self.ue.alloc_isa_reg = guarded_alloc
        self.cdir, self.wdir, self.plan = cdir, wdir, plan
        cfg = plan["config"]
        self.cfg = cfg
        self.H = cfg["hidden_size"]
        self.layer_ids = list(layers) if layers is not None else list(range(cfg["num_hidden_layers"]))
        self.L = len(self.layer_ids)
        self.NE = cfg["num_experts"]
        self.NR = self.NE // router_part[1]              # router rows of this engine
        self.eps = cfg["rms_norm_eps"]
        self.window_len = cfg["sliding_window"]
        self.T_ext = (max_ctx + PV_CH - 1) // PV_CH * PV_CH
        self.max_block = (max_block + VEC - 1) // VEC * VEC
        # rows of a sliding layer's K / V / V^T: the window, one block, and room for the longest image block
        self.RING = min(self.window_len + RING_EXTRA + self.max_block, self.T_ext)
        self.max_ctx = max_ctx

    # ---------------------------------------------------------------- uploads
    def _up(self, t):
        T = self.T
        t = t.to(torch.bfloat16).contiguous()
        nb = t.numel() * BF16
        a = self.ue.allocate_params_dram(nb)
        self.ue.dma_write(T.DMA_DEVICE_H2C, a, T._bf(t), nb)
        return a

    def _t(self, numel, init=None):
        a = self.ue.allocate_tensor_dram(numel * BF16)
        if init is not None:
            t = torch.full((numel,), float(init), dtype=torch.bfloat16)
            self.ue.dma_write(self.T.DMA_DEVICE_H2C, a, t, numel * BF16)
        return a

    def prepare(self):
        T, ue, H = self.T, self.ue, self.H
        ue.software_reset()
        ue._preallocate_identity_matrix()
        up4 = lambda p: T._upload_fp4_pair(ue, p)       # noqa: E731
        self.layers = []
        self.rope = rope_tables(self.cfg, self.T_ext)
        for li in self.layer_ids:
            w = torch.load(os.path.join(self.cdir, "layer_%02d.pt" % li), weights_only=False, map_location="cpu")
            hd, sliding = w["head_dim"], w["sliding"]
            nq = len(w["q_heads"])
            d = {"li": li, "nq": nq, "hd": hd, "sliding": sliding, "scalar": w["layer_scalar"],
                 "ln_in": self._up(w["ln_in"]), "ln_post_attn": self._up(w["ln_post_attn"]),
                 "ln_pre_ff": self._up(w["ln_pre_ff"]), "ln_pre_ff2": self._up(w["ln_pre_ff2"]),
                 "ln_post_ff1": self._up(w["ln_post_ff1"]), "ln_post_ff2": self._up(w["ln_post_ff2"]),
                 "ln_post_ff": self._up(w["ln_post_ff"]),
                 "router_w": self._up(w["router_w"][self.router_part[0] * self.NR:(self.router_part[0] + 1) * self.NR])}
            if nq:
                q, k, o = w["q"], w["k"], w["o"]
                qn, kn = w["q_norm"].float(), w["k_norm"].float()
                if not sliding:                         # rotated dims first (see the module docstring)
                    perm = global_perm(hd, self.rope["g"][0])
                    q = pair_rows(q, nq * hd, H, torch.cat([h * hd + perm for h in range(nq)]))
                    k = pair_rows(k, hd, H, perm)
                    bp = perm.view(-1, 64)[:, 0] // 64
                    o = pair_col_blocks(o, H, nq * hd, torch.cat([h * (hd // 64) + bp for h in range(nq)]))
                    qn, kn = qn[perm], kn[perm]
                d.update({"q": up4(q), "k": up4(k), "v": up4(w["v"]) if w["v"] is not None else None,
                          "o": up4(o), "q_norm": self._up(qn), "k_norm": self._up(kn)})
            dn = w["dense"]["n"]
            d["dense"] = {"n": dn, "gu": up4(w["dense"]["gu"]), "down": up4(w["dense"]["down"])}
            ex = []
            for e in range(self.NE):
                m = w["experts"][e]
                if e in w["extra"]:                     # this shard also holds block 10: one 128-neuron expert
                    x = w["extra"][e]
                    gu = pair_cat_rows([pair_rows(m["gu"], 128, H, range(0, 64)), pair_rows(x["gu"], 128, H, range(0, 64)),
                                        pair_rows(m["gu"], 128, H, range(64, 128)), pair_rows(x["gu"], 128, H, range(64, 128))])
                    down = pair_cat_k([m["down"], x["down"]], H, [64, 64])
                    n = 128
                else:
                    gu, down, n = m["gu"], m["down"], 64
                ex.append({"n": n, "gu": up4(gu), "down": up4(down)})
            d["experts"] = ex
            self.layers.append(d)
        self.lm_head = up4(torch.load(os.path.join(self.cdir, "lm_head.pt"), weights_only=False))
        # this shard's rows of the (scaled, bf16) embedding table: the host copies a token's row from here
        ep = os.path.join(self.cdir, "embed.pt")
        self.EMB = self._up(torch.load(ep, weights_only=False)) if os.path.exists(ep) else None
        self.final_norm = self._up(torch.load(os.path.join(self.wdir, "final_norm.pt"), weights_only=False))
        # constants
        self.onesH = self._up(torch.ones(H))
        self.ones512 = self._up(torch.ones(512))
        self.onehot = [self._up(torch.nn.functional.one_hot(torch.tensor(i), VEC).float()) for i in range(TOPK)]
        self.tabs = {}
        for kind in ("s", "g"):
            n, tab = self.rope[kind]
            self.tabs[kind] = {name: self._up(t) for name, t in tab.items()}
        # ---- persistent tensors
        hd_max = 512
        self.X = self._t(H)                       # replaced by the program header (the host writes it)
        self.XN, self.XR, self.XD, self.XE = self._t(H), self._t(H), self._t(H), self._t(H)
        self.Q, self.QN, self.QR = self._t(2 * hd_max), self._t(2 * hd_max), self._t(2 * hd_max)
        self.Kt, self.KN, self.KR, self.VN = self._t(hd_max), self._t(hd_max), self._t(hd_max), self._t(hd_max)
        self.ATT = self._t(2 * hd_max)
        self.SHARE_A = self._t(H, 0.0)            # stays 0 on shards without attention heads
        self.SHARE_E = self._t(2 * H)             # [dense | MoE]
        self.SUM_A, self.TMPH = self._t(H), self._t(H)
        self.SUM_DM = self._t(2 * H)              # [sum dense | sum MoE]
        self.SUM_D, self.SUM_M = self.SUM_DM, self.SUM_DM + H * BF16
        self.ZERO_H = self._t(H, 0.0)
        self.LOGITS = self._t(self.NE)
        self.W8 = self._t(VEC)
        self.Y, self.ACC = self._t(H), self._t(H)
        n_dense = max(d["dense"]["n"] for d in self.layers)
        self.GU_D, self.HID_D = self._t(2 * n_dense), self._t(n_dense)
        self.GU, self.HID = self._t(256), self._t(128)
        self.ROPE_ROW = {kind: {name: self._t(128) for name in ("cos", "nsin", "sin")} for kind in ("s", "g")}
        for d in self.layers:
            if d["nq"]:
                hd = d["hd"]
                d["T"] = self.RING if d["sliding"] else self.T_ext
                n = d["T"] * hd
                d["K"], d["V"], d["VT"] = self._t(n, 0.0), self._t(n, 0.0), self._t(n, 0.0)
                d["SC"] = self._t(d["nq"] * d["T"], 0.0)
                d["MASK"] = self._t(d["T"])
        self.reset_state()

    def reset_state(self):
        """Fresh conversation: every mask row back to 'nothing visible'."""
        T, ue = self.T, self.ue
        for d in self.layers:
            if d["nq"]:
                m = torch.full((d["T"],), NEG).to(torch.bfloat16)
                ue.dma_write(T.DMA_DEVICE_H2C, d["MASK"], m, d["T"] * BF16)

    # ---------------------------------------------------------------- emission helpers
    def _gemv(self, A, pair, N, K, OUT, gelu=False, bias=None):
        """y = x @ W^T (+ bias) with W in IF4 (the vendor's fast M=1 path)."""
        self.ue.quantized_matmat_core(M=1, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=pair[1], OUTPUT_DRAM_ADDR=OUT,
                                      SCALE_DRAM_ADDR=pair[0], C_DRAM_ADDR=bias, bias_mode="broadcast_N",
                                      data_type=self.U.TYPE.IF4, gelu_enable=gelu)

    def _rms(self, X, gamma, OUT, M=1, N=None):
        self.ue.rms_norm_core_dram(M=M, N=N or self.H, A_DRAM_ADDR=X, OUTPUT_DRAM_ADDR=OUT, GAMMA_DRAM_ADDR=gamma,
                                   eps=self.eps)

    def _to_a(self, addr, n):
        U = self.U
        self.ue.ue_memcpy_from_dram(addr, n * BF16, U.MEMCPY_TYPE.URAM.value, U.URAM_START_ADDR,
                                    U.URAM_SECTION.URAM_A.value)

    def _to_b(self, addr, n):
        U = self.U
        self.ue.ue_memcpy_from_dram(addr, n * BF16, U.MEMCPY_TYPE.URAM.value, U.URAM_START_ADDR,
                                    U.URAM_SECTION.URAM_B.value)

    def _a_to(self, addr, n):
        U = self.U
        self.ue.ue_memcpy_to_dram(U.MEMCPY_TYPE.URAM.value, U.URAM_SECTION.URAM_A.value, U.URAM_START_ADDR, addr,
                                  n * BF16)

    def _sum_parts(self, base, stride_b, n_parts, OUT, n=None):
        """OUT = sum of n_parts n-vectors (default H) at base + j * stride_b."""
        n = n or self.H
        self._to_a(base, n)
        for j in range(1, n_parts):
            self._to_b(base + j * stride_b, n)
            self.ue.eltwise_add_core(0x00000, 0x80000, 0x00000, n)
        self._a_to(OUT, n)

    def _add_into_x(self, V, scalar=None):
        """X = (X + V) [* scalar]."""
        U, H = self.U, self.H
        self._to_a(V, H)
        self._to_b(self.X, H)
        self.ue.eltwise_add_core(0x00000, 0x80000, 0x00000, H)
        if scalar is not None and scalar != 1.0:
            self.ue.start_queue_broadcast(U.UE_MODE.MUL_BROADCAST, U.BROADCAST_MODE.SCALAR_IN_REG, 0x00000, 0x00000,
                                          H, scalar=float(scalar))
        self._a_to(self.X, H)

    def _finish_attn(self, d):
        """X += post_attention_norm(sum of the attention shares)."""
        self._sum_parts(self.PARTS, self.H * BF16, self.parts_n, self.SUM_A)
        self._rms(self.SUM_A, d["ln_post_attn"], self.TMPH)
        self._add_into_x(self.TMPH)

    def _finish_ffn(self, d):
        """X = (X + post_ff_norm(post_ff_norm_1(sum dense) + post_ff_norm_2(sum MoE))) * layer_scalar."""
        H, nb = self.H, self.H * BF16
        if FUSE:                                       # [dense | MoE] of every card in one 2H pass
            self._sum_parts(self.PARTS, 2 * nb, self.parts_n, self.SUM_DM, 2 * H)
        else:
            self._sum_parts(self.PARTS, 2 * nb, self.parts_n, self.SUM_D)
            self._sum_parts(self.PARTS + nb, 2 * nb, self.parts_n, self.SUM_M)
        self._rms(self.SUM_D, d["ln_post_ff1"], self.SUM_D)
        self._rms(self.SUM_M, d["ln_post_ff2"], self.SUM_M)
        self._to_a(self.SUM_D, H)
        self._to_b(self.SUM_M, H)
        self.ue.eltwise_add_core(0x00000, 0x80000, 0x00000, H)
        self._a_to(self.TMPH, H)
        self._rms(self.TMPH, d["ln_post_ff"], self.TMPH)
        self._add_into_x(self.TMPH, d["scalar"])

    def _scalar_to_lalu(self, vec_dram, onehot_dram, onehot_reg=None):
        """LALU register := vec[i] for the one-hot e_i (dot product, ReLU-clamped: the weights are > 0).
        onehot_reg: the one-hot's address comes from that register at run time."""
        ue, U = self.ue, self.U
        SA, SB = 0x60000, 0xFE000
        ue.accelerator_memory_to_sram(vec_dram, SA, VEC)
        ue.accelerator_memory_to_sram(onehot_dram, SB, VEC, general_reg_src=onehot_reg)
        _, va = ue.sram_address_to_uram_address(SA)
        _, ma = ue.sram_address_to_uram_address(SB)
        ue.ue_arithmetic_op(0, 0, 1, U.LALU_CLAMP_RELU_A, U.LALU_CLAMP_RELU_B, U.LALU_MODE.CLAMP.value, 0,
                            U.URAM_SECTION.URAM_A.value, 0, 0, U.URAM_WRITE_SRC.URAM_WB_DISABLE.value,
                            U.UE_MODE.BF16_DOT_PRODUCT, 0, va, ma, 1, 0, VEC, 1)

    def _ident(self, n):
        if n not in self._idents:
            self._idents[n] = self._up(torch.eye(n))
        return self._idents[n]

    def _rope_load(self, kind):
        ue, rr = self.ue, self.ROPE_ROW[kind]
        ue.accelerator_memory_to_sram(rr["cos"], SB_COS, 128)
        ue.accelerator_memory_to_sram(rr["nsin"], SB_NSIN, 128)
        ue.accelerator_memory_to_sram(rr["sin"], SB_SIN, 128)

    def _rope(self, src, dst, hd, half):
        """HF RoPE over the first 2*half dims as pairs (i, i + half); the rest passes through.
        cos / -sin / sin of this position must be in SB_* (see _rope_load)."""
        ue = self.ue
        ue.accelerator_memory_to_sram(src, SA_X_LO, half)
        ue.accelerator_memory_to_sram(src + half * BF16, SA_X_HI, half)
        ue.eltwise_mul_core(SA_X_LO, SB_COS, SB_TMP, half)
        ue.eltwise_mul_core(SA_X_HI, SB_NSIN, SA_TMP, half)
        ue.eltwise_add_core(SA_TMP, SB_TMP, SA_OUT_LO, half)
        ue.eltwise_mul_core(SA_X_HI, SB_COS, SB_TMP, half)
        ue.eltwise_mul_core(SA_X_LO, SB_SIN, SA_TMP, half)
        ue.eltwise_add_core(SA_TMP, SB_TMP, SA_OUT_HI, half)
        ue.sram_to_accelerator_memory(SA_OUT_LO, dst, half)
        ue.sram_to_accelerator_memory(SA_OUT_HI, dst + half * BF16, half)
        rest = hd - 2 * half
        if rest:
            ue.accelerator_memory_to_sram(src + 2 * half * BF16, SA_PASS, rest)
            ue.sram_to_accelerator_memory(SA_PASS, dst + 2 * half * BF16, rest)

    # ---------------------------------------------------------------- per-token setup (start of A0)
    def _emit_rope_rows(self):
        """This position's RoPE rows (R_ROPE) into fixed slots."""
        ue, sh = self.ue, self.T.ue_35bit_addr_shifter
        for kind in ("s", "g"):
            for name in ("cos", "nsin", "sin"):
                ue.generate_instruction_add_imm(R_ROPE, sh(self.tabs[kind][name]), ue._ISA_TMP_REG)
                ue.accelerator_memory_to_sram(self.tabs[kind][name], 0x40000, 128, general_reg_src=ue._ISA_TMP_REG)
                ue.sram_to_accelerator_memory(0x40000, self.ROPE_ROW[kind][name], 128)

    def _emit_token_setup(self, tpl_new, tpl_old):
        ue, sh = self.ue, self.T.ue_35bit_addr_shifter
        self._emit_rope_rows()
        # mask rows: the block holding this position, and (sliding layers) the block leaving the window
        SA_NEW, SA_OLD = 0x40400, 0x40480
        ue.accelerator_memory_to_sram(tpl_new, SA_NEW, VEC)
        ue.accelerator_memory_to_sram(tpl_old, SA_OLD, VEC)
        for d in self.layers:
            if not d["nq"]:
                continue
            row = d["MASK"]
            ue.generate_instruction_add_imm(R_MASK_NEW if d["sliding"] else R_MASK_NEW_G, sh(row), ue._ISA_TMP_REG)
            ue.sram_to_accelerator_memory(SA_NEW, row, VEC, general_reg_src=ue._ISA_TMP_REG)
            if d["sliding"]:
                ue.generate_instruction_add_imm(R_MASK_OLD, sh(row), ue._ISA_TMP_REG)
                ue.sram_to_accelerator_memory(SA_OLD, row, VEC, general_reg_src=ue._ISA_TMP_REG)

    # ---------------------------------------------------------------- segments
    def _emit_attention(self, d, part="all", mask=None):
        """part "all": the token's attention (the normal program). The block program splits it: "kv" writes
        this token's K / V (and V^T) only, "q" computes q and attends with the mask row `mask`."""
        T, ue, H = self.T, self.ue, self.H
        nq, hd, sh = d["nq"], d["hd"], T.ue_35bit_addr_shifter
        do_q, do_kv = part in ("all", "q"), part in ("all", "kv")
        mask = d["MASK"] if mask is None else mask
        kind = "s" if d["sliding"] else "g"
        half = self.rope[kind][0]
        # sliding layers: slot-based registers into the ring; global layers: position-based
        if d["sliding"]:
            r_pos_k, r_blk_v, r_blk_vt, r_len, r_rebuild = R_POS_K1, R_BLK_V1, R_BLK_VT_S, R_LEN_S, R_REBUILD_S
        else:
            r_pos_k, r_blk_v, r_blk_vt, r_len, r_rebuild = R_POS_K2, R_BLK_V2, R_BLK_VT, R_LEN, R_REBUILD
        TL = d["T"]
        self._rms(self.X, d["ln_in"], self.XN)
        if do_q:
            self._gemv(self.XN, d["q"], nq * hd, H, self.Q)
        if do_kv:
            self._gemv(self.XN, d["k"], hd, H, self.Kt)
            if d["v"] is not None:
                self._gemv(self.XN, d["v"], hd, H, self.Kt + hd * BF16)       # raw V right after raw K
                self._rms(self.Kt + hd * BF16, self.ones512, self.VN, N=hd)
            else:
                self._rms(self.Kt, self.ones512, self.VN, N=hd)               # K = V: v_norm(raw k), no RoPE
        if do_q:
            self._rms(self.Q, d["q_norm"], self.QN, M=nq, N=hd)
        if do_kv:
            self._rms(self.Kt, d["k_norm"], self.KN, N=hd)
        self._rope_load(kind)
        if do_q:
            for h in range(nq):
                self._rope(self.QN + h * hd * BF16, self.QR + h * hd * BF16, hd, half)
        if do_kv:
            self._rope(self.KN, self.KR, hd, half)
            # KV write at row pos
            for src, cache in ((self.KR, d["K"]), (self.VN, d["V"])):
                ue.accelerator_memory_to_sram(src, 0x40000, hd)
                ue.generate_instruction_add_imm(r_pos_k, sh(cache), ue._ISA_TMP_REG)
                ue.sram_to_accelerator_memory(0x40000, cache, hd, general_reg_src=ue._ISA_TMP_REG)
        rc = ue._isa_reg_counter
        ue._isa_reg_counter = max(rc, 6)
        if do_kv:
            # V^T packed [hd, R_LEN]: transpose this position's 64-row block into it (row stride R_LEN*2) ...
            ue.generate_instruction_shl(R_VT_STRIDE, r_len, 1)
            ue.generate_instruction_add_imm(r_blk_v, sh(d["V"]), R_VT_IN)
            ue.generate_instruction_add_imm(r_blk_vt, sh(d["VT"]), R_VT_OUT)
            ue.bf16_transpose_core(M=VEC, N=hd, INPUT_DRAM_ADDR=d["V"], OUTPUT_DRAM_ADDR=d["VT"],
                                   IDENTITY_DRAM_ADDR=ue._identity_dram_addr, gpr_input_addr=R_VT_IN,
                                   gpr_out_addr=R_VT_OUT, gpr_out_row_stride_reg=R_VT_STRIDE)
            # ... and when R_LEN has just grown, rebuild all of it with the new row stride
            patch = ue._emit_forward_skip_jz(r_rebuild, self._cap_base)
            ue.bf16_transpose_core(M=TL, N=hd, INPUT_DRAM_ADDR=d["V"], OUTPUT_DRAM_ADDR=d["VT"],
                                   IDENTITY_DRAM_ADDR=ue._identity_dram_addr, gpr_M_reg=r_len)
            patch()
        if do_q:
            # both heads in ONE matmul (scale 1.0: the norms carry it), shared mask row, then ONE P.V matmul
            ue.matmat_mul_core(M=nq, K=hd, N=TL, A_DRAM_ADDR=self.QR, B_DRAM_ADDR=d["K"],
                               OUTPUT_DRAM_ADDR=d["SC"], softmax_enable=True, C_DRAM_ADDR=mask,
                               bias_mode="broadcast_N", gpr_N_reg=r_len)
            ue.matmat_mul_core(M=nq, K=TL, N=hd, A_DRAM_ADDR=d["SC"], B_DRAM_ADDR=d["VT"],
                               OUTPUT_DRAM_ADDR=self.ATT, gpr_K_reg=r_len)
        ue._isa_reg_counter = rc
        if do_q:
            self._gemv(self.ATT, d["o"], H, nq * hd, self.SHARE_A)

    def _emit_router(self, d):
        self._finish_attn(d)
        self._rms(self.X, self.onesH, self.XR)
        self.ue.matmat_mul_core(M=1, K=self.H, N=self.NR, A_DRAM_ADDR=self.XR, B_DRAM_ADDR=d["router_w"],
                                OUTPUT_DRAM_ADDR=self.LOGITS_OUT)

    def _emit_dense(self, d):
        """D: the dense FFN (this shard's neurons) -> dense half of SHARE_E. It needs no expert pick, so it
        runs while the host picks the experts."""
        ue, H = self.ue, self.H
        self._rms(self.X, d["ln_pre_ff"], self.XD)
        self._rms(self.X, d["ln_pre_ff2"], self.XE)
        n = d["dense"]["n"]
        self._gemv(self.XD, d["dense"]["gu"], 2 * n, H, self.GU_D)
        ue.matmat_mul_core(M=n // VEC, K=VEC, N=VEC, A_DRAM_ADDR=self.GU_D, B_DRAM_ADDR=self._ident(VEC),
                           OUTPUT_DRAM_ADDR=self.HID_D, gelu_enable=True)
        self._to_a(self.HID_D, n)
        self._to_b(self.GU_D + n * BF16, n)
        ue.eltwise_mul_core(0x00000, 0x80000, 0x00000, n)
        self._a_to(self.HID_D, n)
        self._gemv(self.HID_D, d["dense"]["down"], H, n, self.SHARE_E)

    def _emit_experts(self, d, prog_base, lg8_addr):
        """E: the 8 picked experts -> MoE half of SHARE_E. Starts right after the data slots of lg8_addr: the
        host's ONE DMA writes [8 logits | pad | the 8 register slots below]."""
        ue, U, H, sh = self.ue, self.U, self.H, self.T.ue_35bit_addr_shifter
        nb = H * BF16
        for i in range(TOPK):                         # slots the host overwrites (one DMA with the logits)
            ue.generate_instruction_add_set(R_EXP[i], 0)
        ue.matmat_mul_core(M=1, K=VEC, N=VEC, A_DRAM_ADDR=lg8_addr, B_DRAM_ADDR=ue._identity_dram_addr,
                           OUTPUT_DRAM_ADDR=self.W8, softmax_enable=True)
        if FUSE:
            self._to_a(self.ZERO_H, H)                 # the subroutines accumulate into the MoE share
            self._a_to(self.SHARE_E + nb, H)
            for i in range(TOPK):
                ue.generate_instruction_add_set(R_W, sh(self.onehot[i]))
                ue.pad_capture_to_64b_boundary()
                ue.generate_instruction_add_set(R_RET, sh(prog_base + (ue.capture_count + 2) * self.INST))
                ue.generate_instruction_jump_reg_abs(R_EXP[i])
            return
        for i in range(TOPK):
            ue.pad_capture_to_64b_boundary()
            ue.generate_instruction_add_set(R_RET, sh(prog_base + (ue.capture_count + 2) * self.INST))
            ue.generate_instruction_jump_reg_abs(R_EXP[i])           # the expert subroutine writes Y
            self._scalar_to_lalu(self.W8, self.onehot[i])
            self._to_a(self.Y, H)
            ue.start_queue_broadcast(U.UE_MODE.MUL_BROADCAST, U.BROADCAST_MODE.LALU_RESULT, 0x00000, 0x00000, H)
            if i:
                self._to_b(self.ACC, H)
                ue.eltwise_add_core(0x00000, 0x80000, 0x00000, H)
            self._a_to(self.SHARE_E + nb if i == TOPK - 1 else self.ACC, H)

    def _emit_expert_sub(self, e):
        """One expert, this shard's neurons (64, or 128 with block 10): Y = down(gelu(gate) * up)."""
        ue, H, n = self.ue, self.H, e["n"]
        self._gemv(self.XE, e["gu"], 2 * n, H, self.GU)
        ue.matmat_mul_core(M=n // VEC, K=VEC, N=VEC, A_DRAM_ADDR=self.GU, B_DRAM_ADDR=self._ident(VEC),
                           OUTPUT_DRAM_ADDR=self.HID, gelu_enable=True)
        self._to_a(self.HID, n)
        self._to_b(self.GU + n * BF16, n)
        ue.eltwise_mul_core(0x00000, 0x80000, 0x00000, n)
        if FUSE:
            # weight of this pick (one-hot address in R_W) on the n hidden values, then
            # MoE share += down(hidden) through the GEMV's bias input
            U = self.U
            self._scalar_to_lalu(self.W8, self.onehot[0], onehot_reg=R_W)
            ue.start_queue_broadcast(U.UE_MODE.MUL_BROADCAST, U.BROADCAST_MODE.LALU_RESULT, 0x00000, 0x00000, n)
            self._a_to(self.HID, n)
            acc = self.SHARE_E + H * BF16
            self._gemv(self.HID, e["down"], H, n, acc, bias=acc)
        else:
            self._a_to(self.HID, n)
            self._gemv(self.HID, e["down"], H, n, self.Y)
        ue.generate_instruction_jump_reg_abs(R_RET)

    def _emit_head(self, d_last, vocab_n):
        self._finish_ffn(d_last)
        self._rms(self.X, self.final_norm, self.XN)
        self.ue.quantized_matmat_core(M=1, K=self.H, N=vocab_n, A_DRAM_ADDR=self.XN, B_DRAM_ADDR=self.lm_head[1],
                                      OUTPUT_DRAM_ADDR=self.HEAD_OUT, SCALE_DRAM_ADDR=self.lm_head[0],
                                      data_type=self.U.TYPE.IF4)

    # ---------------------------------------------------------------- compile
    def _place(self, emit):
        ue, T = self.ue, self.T
        ue.clear_capture_buffer()
        T.UnifiedEngine.start_capture(ue)
        ue._compile_mode = True
        base = (ue.get_program_dram_addr() + 63) & ~63
        self._cap_base = base
        emit(base)
        ue._compile_mode = False
        T.UnifiedEngine.stop_capture(ue)
        ue.write_captured_instructions_to_dram(base)
        ue.allocate_program_dram(base - ue.get_program_dram_addr() + ue.get_capture_instruction_size_bytes() + 64)
        ue.clear_capture_buffer()
        return base

    def compile(self, parts_n, vocab_n, parts_addr=None, logits_addr=None):
        """Build the per-token program (3 halted segments per layer + the LM head) and one subroutine per
        (layer, expert). parts_addr: share another engine's buffer of copied-in shares (one copy per card).
        The parts buffer holds parts_n slots of 2H (attention shares use the first H of each slot's
        stride-H layout, FFN shares the full 2H)."""
        ue = self.ue
        self.INST = self.U.INSTRUCTION_SIZE_BYTES
        self._idents = {}
        self.parts_n = parts_n
        self.PARTS = parts_addr if parts_addr is not None else self._t(parts_n * 2 * self.H)
        self.LOGITS_OUT = logits_addr if logits_addr is not None else self.LOGITS + self.router_part[0] * self.NR * BF16
        self.HEAD_OUT = self._t(vocab_n)
        self._ident(VEC)
        self.sub_addr = [[self._place(lambda base, e=e: self._emit_expert_sub(e)) for e in d["experts"]]
                         for d in self.layers]
        segs = []
        self.lg8_addr = []
        INST, H = self.INST, self.H
        x_slots, tpl_slots = H * BF16 // INST, VEC * BF16 // INST
        self.pos_regs = [R_POS_K1, R_POS_K2, R_ROPE, R_LEN, R_BLK_V1, R_BLK_V2, R_BLK_VT, R_MASK_NEW, R_MASK_OLD,
                         R_REBUILD, R_LEN_S, R_BLK_VT_S, R_MASK_NEW_G, R_REBUILD_S]

        def data_slots(n):
            for _ in range(n):
                ue.generate_instruction_add_set(R_DATA, 0)

        def body(base):
            # header, written by the host with ONE DMA per token: X | TPL_NEW | TPL_OLD | register slots
            data_slots(x_slots + 2 * tpl_slots)
            self.X = base
            tpl_new, tpl_old = base + x_slots * INST, base + (x_slots + tpl_slots) * INST
            segs.append(("A", ue.capture_count))
            for r in self.pos_regs:
                ue.generate_instruction_add_set(r, 0)
            for i, d in enumerate(self.layers):
                if i == 0:
                    self._emit_token_setup(tpl_new, tpl_old)
                else:
                    self._finish_ffn(self.layers[i - 1])
                if d["nq"]:
                    self._emit_attention(d)
                ue.generate_instruction_halt()
                segs.append(("R", ue.capture_count))
                self._emit_router(d)
                ue.generate_instruction_halt()
                segs.append(("D", ue.capture_count))
                self._emit_dense(d)
                ue.generate_instruction_halt()
                ue.pad_capture_to_64b_boundary()
                lg8 = base + ue.capture_count * INST      # the host writes [8 logits | pad | 8 register slots]
                data_slots(VEC * BF16 // INST)
                self.lg8_addr.append(lg8)
                segs.append(("E", ue.capture_count))
                self._emit_experts(d, base, lg8)
                ue.generate_instruction_halt()
                segs.append(("A" if i + 1 < self.L else "H", ue.capture_count))
            self._emit_head(self.layers[-1], vocab_n)
            ue.generate_instruction_halt()

        self.X = None
        base = self._place(body)
        self.prog_base = base
        self.seg_kind = [k for k, _ in segs]
        self.seg_start = [base + idx * INST for _, idx in segs]
        self._addset_encoder()
        self.sel_bytes = [[[self._enc(R_EXP[i], self.T.ue_35bit_addr_shifter(a)) for i in range(TOPK)]
                           for a in row] for row in self.sub_addr]
        return len(self.seg_start)

    # ---------------------------------------------------------------- image blocks
    def _emit_x_load(self):
        """XB := XS[token] (R_XT)."""
        ue, sh = self.ue, self.T.ue_35bit_addr_shifter
        ue.generate_instruction_add_imm(R_XT, sh(self.XS), ue._ISA_TMP_REG)
        ue.accelerator_memory_to_sram(self.XS, 0x00000, self.H, general_reg_src=ue._ISA_TMP_REG)
        self._a_to(self.XB, self.H)

    def _emit_x_store(self):
        """XS[token] (R_XT) := XB."""
        ue, sh = self.ue, self.T.ue_35bit_addr_shifter
        self._to_a(self.XB, self.H)
        ue.generate_instruction_add_imm(R_XT, sh(self.XS), ue._ISA_TMP_REG)
        ue.sram_to_accelerator_memory(0x00000, self.XS, self.H, general_reg_src=ue._ISA_TMP_REG)

    def compile_block(self):
        """The block program: the prompt's image tokens, layer by layer, so that in the sliding layers every
        token of the block also attends to the block's later tokens (transformers: AND(window, OR(causal,
        same block))); the global layers stay causal. Each token's residual lives in XS (on the card).
            K        : per token, layer 0's K / V
            per layer: B  token entry (registers, RoPE rows, X from XS), q + attention (host-written mask row)
                       R, D, E as in the normal program
                       F  finish the FFN, the NEXT layer's K / V, X back to XS
        so a layer's K / V of the whole block exist before any of its tokens attends. Call after compile()."""
        ue, H, INST = self.ue, self.H, self.INST
        assert self.max_block, "the shard was built without max_block"
        self.XS = self._t(self.max_block * H)
        self.XB = self._t(H)
        self.MS, self.MG = self._t(self.RING), self._t(self.T_ext)
        self.blk_regs = self.pos_regs + [R_XT]
        saved_x = self.X
        self.X = self.XB
        segs, self.blk_lg8 = [], []

        def data_slots(n):
            for _ in range(n):
                ue.generate_instruction_add_set(R_DATA, 0)

        def entry():                                  # the host overwrites these slots for each token
            for r in self.blk_regs:
                ue.generate_instruction_add_set(r, 0)
            self._emit_rope_rows()
            self._emit_x_load()

        def body(base):
            segs.append(("K", -1, ue.capture_count))
            entry()
            if self.layers[0]["nq"]:
                self._emit_attention(self.layers[0], "kv")
            ue.generate_instruction_halt()
            for i, d in enumerate(self.layers):
                segs.append(("B", i, ue.capture_count))
                entry()
                if d["nq"]:
                    self._emit_attention(d, "q", mask=self.MS if d["sliding"] else self.MG)
                ue.generate_instruction_halt()
                segs.append(("R", i, ue.capture_count))
                self._emit_router(d)
                ue.generate_instruction_halt()
                segs.append(("D", i, ue.capture_count))
                self._emit_dense(d)
                ue.generate_instruction_halt()
                ue.pad_capture_to_64b_boundary()
                lg8 = base + ue.capture_count * INST
                data_slots(VEC * BF16 // INST)
                self.blk_lg8.append(lg8)
                segs.append(("E", i, ue.capture_count))
                self._emit_experts(d, base, lg8)
                ue.generate_instruction_halt()
                segs.append(("F", i, ue.capture_count))
                self._finish_ffn(d)
                if i + 1 < self.L and self.layers[i + 1]["nq"]:
                    self._emit_attention(self.layers[i + 1], "kv")
                self._emit_x_store()
                ue.generate_instruction_halt()

        base = self._place(body)
        self.X = saved_x
        self.bseg_kind = [k for k, _, _ in segs]
        self.bseg_layer = [i for _, i, _ in segs]
        self.bseg_start = [base + idx * INST for _, _, idx in segs]
        self._mask_cache = {}
        return len(self.bseg_start)

    def block_begin(self, p0, n, xs_bytes):
        """A block of n image tokens at positions p0..p0+n-1, their embeddings (n rows, bf16 bits) into XS."""
        assert 0 < n <= self.max_block and p0 + n <= self.max_ctx
        self.blk = (p0, n)
        self._mask_cache = {}
        self._write(self.XS, xs_bytes)

    def block_slots(self, seg, t):
        """Token t's registers into the entry slots of block segment seg (K or B), ONE DMA: the same values as
        set_position, except that the lengths already cover the whole block (all of its K / V exist before
        anyone attends) and token 0 rebuilds V^T with that length."""
        sh, R = self.T.ue_35bit_addr_shifter, self.RING
        p0, n = self.blk
        pos = p0 + t
        Lb = (p0 + n - 1 + VEC) // VEC * VEC
        blk = pos // VEC * VEC
        slot = pos % R
        blk_s = slot // VEC * VEC
        rb = 1 if t == 0 else 0
        vals = [sh(slot * 256 * BF16), sh(pos * 512 * BF16), sh(pos * 128 * BF16), min(Lb, self.T_ext),
                sh(blk_s * 256 * BF16), sh(blk * 512 * BF16), sh(blk * BF16), sh(blk_s * BF16), sh(blk_s * BF16),
                rb, min(Lb, R), sh(blk_s * BF16), sh(blk * BF16), rb, sh(t * self.H * BF16)]
        self._write(self.bseg_start[seg], b"".join(self._enc(r, v) for r, v in zip(self.blk_regs, vals)))

    def _held(self, last):
        """Position each ring slot holds once positions 0..last are written (-1: none)."""
        s = np.arange(self.RING)
        return last - ((last - s) % self.RING)

    def _mask_row(self, sliding, pos, last, block):
        """bf16 bits of a mask row: what query `pos` may see once positions 0..last are written. block =
        (first, last) of the image block pos belongs to, or None."""
        if sliding:
            kv = self._held(last)
            vis = (kv >= 0) & (kv > pos - self.window_len) & (kv <= pos)
            if block is not None:
                vis |= (kv >= block[0]) & (kv <= block[1]) & (kv > pos - self.window_len)
            n = self.RING
        else:
            n = (last + VEC) // VEC * VEC
            kv = np.arange(n)
            vis = kv <= pos
        row = np.full(n, NEG_BF, dtype=np.int16)
        row[vis] = 0
        return row

    def block_mask(self, seg, t):
        """Token t's mask row for block segment seg (B) of a layer with heads, ONE DMA."""
        d = self.layers[self.bseg_layer[seg]]
        if not d["nq"]:
            return
        p0, n = self.blk
        key = (d["sliding"], t)
        row = self._mask_cache.get(key)
        if row is None:
            row = self._mask_cache[key] = self._mask_row(d["sliding"], p0 + t, p0 + n - 1,
                                                         (p0, p0 + n - 1)).tobytes()
        self._write(self.MS if d["sliding"] else self.MG, row)

    def block_end(self):
        """Back to the normal program: every layer's mask row as the next text token expects it (what the
        block's last token sees causally; the block is ordinary history from now on)."""
        p0, n = self.blk
        last = p0 + n - 1
        rows = {s: self._mask_row(s, last, last, None) for s in (True, False)}
        for d in self.layers:
            if d["nq"]:
                r = rows[d["sliding"]]
                full = np.full(d["T"], NEG_BF, dtype=np.int16)
                full[:min(len(r), d["T"])] = r[:d["T"]]
                self._write(d["MASK"], full.tobytes())
        self.blk = None

    def _addset_encoder(self):
        """Learn where ADD_SET keeps its 32-bit immediate, so the host can produce instruction bytes directly."""
        ue, T = self.ue, self.T
        import random
        import struct

        def capture(r, v):
            ue.clear_capture_buffer()
            T.UnifiedEngine.start_capture(ue)
            ue.generate_instruction_add_set(r, v)
            T.UnifiedEngine.stop_capture(ue)
            w = list(ue.capture_buffer[0].words)
            ue.clear_capture_buffer()
            return w

        self._addset = {}
        for r in set(self.pos_regs) | set(R_EXP) | {R_XT}:
            base = capture(r, 0)
            bitmap = []
            for k in range(32):
                diff = [(i, a ^ b) for i, (a, b) in enumerate(zip(capture(r, 1 << k), base)) if a != b]
                assert len(diff) == 1 and bin(diff[0][1]).count("1") == 1, "ADD_SET bit %d is not a single bit" % k
                bitmap.append((diff[0][0], diff[0][1].bit_length() - 1))
            self._addset[r] = (base, bitmap)
            for v in [random.getrandbits(32) for _ in range(4)] + [0xFFFFFFFF]:
                assert self._enc(r, v) == b"".join(struct.pack("<I", x & 0xFFFFFFFF) for x in capture(r, v))

    def _enc(self, reg, value):
        """ADD_SET reg, value as instruction bytes."""
        import struct
        base, bitmap = self._addset[reg]
        w = list(base)
        value = int(value) & 0xFFFFFFFF
        for k, (i, b) in enumerate(bitmap):
            if value >> k & 1:
                w[i] |= 1 << b
        return b"".join(struct.pack("<I", v & 0xFFFFFFFF) for v in w)

    # ---------------------------------------------------------------- host commands
    def set_position(self, pos, x_i16):
        """Token start, ONE DMA: the embedding row, both mask templates and the position registers."""
        sh, R = self.T.ue_35bit_addr_shifter, self.RING
        # the ring is a whole number of 64-blocks, so a slot's place in its block is the position's: the same
        # two mask templates serve the global rows (by position) and the sliding rows (by slot)
        blk = pos // VEC * VEC
        slot = pos % R
        blk_s = slot // VEC * VEC
        t_new = np.full(VEC, NEG_BF, dtype=np.int16)
        t_new[:pos - blk + 1] = 0
        if pos >= self.window_len:
            c = (pos - self.window_len) % R           # this slot leaves the sliding window now
            blk_old = c // VEC * VEC
            t_old = np.zeros(VEC, dtype=np.int16)
            t_old[:c - blk_old + 1] = NEG_BF
        else:
            blk_old, t_old = blk_s, t_new
        L = (pos + VEC) // VEC * VEC
        vals = [sh(slot * 256 * BF16), sh(pos * 512 * BF16), sh(pos * 128 * BF16), min(L, self.T_ext),
                sh(blk_s * 256 * BF16), sh(blk * 512 * BF16), sh(blk * BF16), sh(blk_s * BF16), sh(blk_old * BF16),
                1 if pos % VEC == 0 else 0,
                min(L, R), sh(blk_s * BF16), sh(blk * BF16), 1 if pos % VEC == 0 and pos < R else 0]
        buf = (bytes(x_i16) + t_new.tobytes() + t_old.tobytes()
               + b"".join(self._enc(r, v) for r, v in zip(self.pos_regs, vals)))
        self._write(self.X, buf)

    def _write(self, addr, buf):
        if self.fast_h2c is not None:
            if os.pwrite(self.fast_h2c, buf, addr) != len(buf):
                raise OSError("short h2c write at 0x%x" % addr)
        else:
            self.ue.dma_write(self.T.DMA_DEVICE_H2C, addr, torch.frombuffer(bytearray(buf), dtype=torch.uint8), len(buf))

    def set_experts_bf(self, li_idx, idx, lg_bf, block=False):
        """The host's pick for layer slot li_idx, ONE DMA: the 8 logits (bf16 bit patterns, as read from the
        card) padded to 64 with -inf, and the 8 register slots pointing at the experts' subroutines.
        block: into the block program's E segment."""
        pad = self.__dict__.get("_lg_pad")
        if pad is None:
            pad = self._lg_pad = np.full(VEC - TOPK, NEG_BF, dtype=np.int16).tobytes()
        sb = self.sel_bytes[li_idx]
        self._write((self.blk_lg8 if block else self.lg8_addr)[li_idx],
                    _PACK8H(*lg_bf) + pad + b"".join(sb[e][i] for i, e in enumerate(idx)))

    def run(self, addr, limit_s=30.0):
        ue = self.ue
        ue.start_execute_from_dram(addr)
        t = time.time()
        while ue.is_queue_busy():
            if time.time() - t > limit_s:
                raise RuntimeError("program at 0x%x did not finish" % addr)
        return ue.report_latency_in_us()
