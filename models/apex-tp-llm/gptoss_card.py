"""GPT-OSS-20B on the Apex cards, tensor-parallel: one engine = one shard of every layer.

All model compute runs on the cards. The host only moves bytes, issues commands, picks the 4 expert
indices from the router logits the cards report, and picks the next token.

Per layer three segments, each ending in a halt:
  A  attention : [add the shards' shares into the residual] rmsnorm, q/k/v (+bias), YaRN RoPE, KV write,
                 softmax(q.K^T*scale + bias) per head with the sink as an extra score column, P.V,
                 o projection of this shard's heads (+o bias on shard 0)           -> share
  R  router    : [add shares] rmsnorm, router logits (all 32, every shard)          -> logits
  E  experts   : softmax over the 4 logits the host copied back, then for each of the 4 picked
                 experts a call into its precompiled subroutine (this shard's neurons), weighted sum
                                                                                     -> share
then the LM head (this shard's vocabulary rows). 73 segments per token.

Score row layout per (layer, head): columns 0..63 are the sink block (column 0 holds the sink logit,
1..63 are -inf, the matching K/V rows are zero), position p lives at column 64 + p. The masks for
the full and the sliding (128) window are kept in the bias rows and updated each token by a
compiled, register-addressed copy of two 64-element templates the host writes.
"""
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
import torch

BF16 = 2
VEC = 64
SINK_COLS = 64
NEG = -1.0e30                    # bf16 -inf stand-in for masks (exp underflows to 0)
PV_CH = 512                      # T_ext is rounded up to a multiple of this
KQ = 128                         # width of Q' / K' rows: [q or k (64) | one-hot head / sinks (64)]


def bf16_bits(a):
    """float32 values -> bf16 bit patterns (int16), round to nearest even, like torch's cast."""
    u = np.asarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16).view(np.int16)


NEG_BF = bf16_bits([NEG])[0]
_PACK4H = __import__("struct").Struct("<4h").pack

# ISA registers (the vendor allocator hands out 1..~30 as scratch; regs 1-3 are the port's)
R_LEN, R_VT_STRIDE, R_VT_IN, R_VT_OUT, R_BLK_V, R_BLK_VT = 60, 57, 58, 59, 61, 62
R_MASK_NEW, R_MASK_OLD = 44, 45   # byte offsets of the new / expired mask block in a bias row
R_RET, R_EXP = 50, (51, 52, 53, 54)
R_POS_KQ, R_REBUILD = 42, 43       # host: K' row byte offset of this position; 1 when V^T must be rebuilt
REG_FIXED_MIN = 40                 # every register >= this is ours: the vendor's scratch allocator must stay below


def load_base_port(name, work=None):
    """The Qwen3.5 port module: its engine class (eps-padded RMSNorm, identity matrix, fast IF4 GEMV
    wrapper, bounded allocators) and helpers are model-independent enough to build on."""
    work = os.path.expanduser(work or os.environ.get("APEX_TP_WORK", "~/apex_tp"))
    path = os.path.join(work, "models", "q35_9b", "qwen3.5_2b_test.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod, os.path.join(work, "models", "q35_9b", "qwen3.5_2b_config.json")


def yarn_tables(cfg, max_pos):
    """cos/sin tables in the layout of the port's 64-wide RoPE helper (first half filled), scaled by
    YaRN's attention factor, exactly as transformers computes them (host-side preparation)."""
    from transformers import AutoConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    hf = AutoConfig.for_model(**{k: v for k, v in cfg.items() if k not in ("architectures", "transformers_version")})
    inv, att = ROPE_INIT_FUNCTIONS["yarn"](hf, "cpu")
    pos = torch.arange(max_pos, dtype=torch.float32)
    fr = torch.outer(pos, inv.float())                        # [T, 32]
    half = inv.numel()
    cos = torch.zeros(max_pos, 2 * half, dtype=torch.bfloat16)
    nsin = torch.zeros_like(cos)
    sin = torch.zeros_like(cos)
    cos[:, :half] = (fr.cos() * att).to(torch.bfloat16)
    nsin[:, :half] = (-fr.sin() * att).to(torch.bfloat16)
    sin[:, :half] = (fr.sin() * att).to(torch.bfloat16)
    return cos, nsin, sin


class GptOssShard:
    """One engine holding one shard. Build every engine of a card before calling prepare() on any
    (constructing an engine self-tests the first 16 KB of DRAM)."""

    def __init__(self, T, qwen_cfg, dev, engine, cdir, wdir, plan, max_ctx, window=None, n_layers=None):
        self.T, self.U = T, T.user_dma_core
        self.fast_h2c = None             # optional persistent fd of the card's h2c channel (see dma_fd)
        self.prof_w, self.prof_b = [0, 0], [0, 0]   # ns/count: small pwrites, set_experts buffer building
        T.set_dma_device(dev)
        T.DMA_DEVICE_H2C = self.U.DMA_DEVICE_H2C
        T.DMA_DEVICE_C2H = self.U.DMA_DEVICE_C2H
        self.U.configure_clock_from_hardware()
        self.ue = T.Qwen3_5_2b_UnifiedEngine(device="cpu", config_path=qwen_cfg,
                                             BASE_ADDR=self.U.UE_0_BASE_ADDR + engine * 0x10000, **(window or {}))
        alloc = self.ue.alloc_isa_reg

        def guarded_alloc():                          # the vendor allocator does not know our fixed registers
            r = alloc()
            assert r < REG_FIXED_MIN, "vendor scratch register %d reaches the fixed registers (>= %d)" % (
                r, REG_FIXED_MIN)
            return r
        self.ue.alloc_isa_reg = guarded_alloc
        self.cdir, self.wdir, self.plan = cdir, wdir, plan
        cfg = plan["config"]
        self.cfg = cfg
        self.H, self.L, self.HD = cfg["hidden_size"], n_layers or cfg["num_hidden_layers"], cfg["head_dim"]
        self.NE, self.TOPK = cfg["num_local_experts"], cfg["num_experts_per_tok"]
        self.eps = cfg["rms_norm_eps"]
        self.limit = cfg["swiglu_limit"]
        self.window_len = cfg["sliding_window"]
        # score/V^T columns: the sink block + the positions, rounded up to whole P.V chunks of PV_CH
        self.CH = PV_CH
        self.T_ext = (SINK_COLS + max_ctx + PV_CH - 1) // PV_CH * PV_CH
        self.T_al = self.T_ext - SINK_COLS
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
        T, ue, H, HD = self.T, self.ue, self.H, self.HD
        ue.software_reset()
        ue._preallocate_identity_matrix()
        self.layers = []
        for li in range(self.L):
            w = torch.load(os.path.join(self.cdir, "layer_%02d.pt" % li), weights_only=False, map_location="cpu")
            nq = len(w["q_heads"])
            # the 1/sqrt(64) score scale rides in W_k / b_k (a power of 2: exact on the bf16 scales; RoPE is
            # linear, so rope(k/8) = rope(k)/8)
            k_pair = (w["k"][0].float().mul(0.125).to(w["k"][0].dtype), w["k"][1])
            d = {"nq": nq, "sliding": w["sliding"],
                 "ln_in": self._up(w["ln_in"]), "ln_post": self._up(w["ln_post"]),
                 "q": T._upload_fp4_pair(ue, w["q"]), "q_b": self._up(w["q_b"]),
                 "k": T._upload_fp4_pair(ue, k_pair), "k_b": self._up(w["k_b"].float() * 0.125),
                 "v": T._upload_fp4_pair(ue, w["v"]), "v_b": self._up(w["v_b"]),
                 "o": T._upload_fp4_pair(ue, w["o"]),
                 "o_b": self._up(w["o_b"]) if w["o_b"] is not None else None,
                 "sinks": w["sinks"].float()}
            rw = torch.zeros(VEC, H, dtype=torch.bfloat16)          # router padded to 64 outputs
            rw[:self.NE] = w["router_w"]
            rb = torch.full((VEC,), -1.0e4, dtype=torch.bfloat16)    # padding logits never win
            rb[:self.NE] = w["router_b"]
            d["router_w"], d["router_b"] = self._up(rw), self._up(rb)
            ex = []
            for e in w["experts"]:
                n = e["gu_b"].numel() // 2
                ex.append({"n": n, "gu": T._upload_fp4_pair(ue, e["gu"]),
                           "g_b": self._up(e["gu_b"][:n]), "u_b": self._up(e["gu_b"][n:]),
                           "down": T._upload_fp4_pair(ue, e["down"]),
                           "down_b": self._up(e["down_b"]) if e["down_b"] is not None else None})
            d["experts"] = ex
            self.layers.append(d)
        self.lm_head = T._upload_fp4_pair(ue, torch.load(os.path.join(self.cdir, "lm_head.pt"), weights_only=False))
        self.vocab = self.plan["cards"][0]["vocab"]                   # replaced by the caller per shard
        self.final_norm = self._up(torch.load(os.path.join(self.wdir, "final_norm.pt"), weights_only=False))
        # constants
        self.ones64 = self._up(torch.ones(VEC))
        self.onehot = [self._up(torch.nn.functional.one_hot(torch.tensor(i), VEC).float()) for i in range(VEC)]
        cos, nsin, sin = yarn_tables(self.cfg, self.T_al)
        ue._fa_cos_pad_dram, ue._fa_neg_sin_pad_dram, ue._fa_sin_hi_pad_dram = self._up(cos), self._up(nsin), self._up(sin)
        # ---- persistent tensors
        nq_max = max(d["nq"] for d in self.layers)
        self.X = self._t(H)                       # residual stream (the host writes the embedding at token start)
        self.parts_n = None                       # set by the runner: how many shares the host copies in
        self.XN, self.XN2 = self._t(H), self._t(H)
        self.Q = self._t(nq_max * HD)
        # Q' rows [rope(q_h) | onehot(h)] (KQ wide): the one-hot half picks head h's sink from K' row 0
        qp = torch.zeros(nq_max, KQ)
        qp[torch.arange(nq_max), HD + torch.arange(nq_max)] = 1.0
        self.QP = self._up(qp)
        self.Kt, self.Vt, self.KR = self._t(HD), self._t(HD), self._t(HD)
        self.ATT = self._t(nq_max * HD)
        self.SHARE_A, self.SHARE_E = self._t(H), self._t(H)
        self.LOGITS = self._t(VEC)
        self.LG4, self.W4 = self._t(VEC, NEG), self._t(VEC)
        self.Y, self.ACC = self._t(H), self._t(H)
        n_max = max(e["n"] for d in self.layers for e in d["experts"])
        self.G, self.Uu, self.HID = self._t(n_max), self._t(n_max), self._t(n_max)
        self.TPL_NEW, self.TPL_OLD = self._t(VEC, 0.0), self._t(VEC, 0.0)
        kv = self.T_ext * HD
        for d in self.layers:
            # K' [T_ext, KQ]: row 0 = [0 | sinks], rows 64.. = [k_t / 8 | 0]; V [T_ext, 64];
            # V^T packed [64, R_LEN] (row stride = the current visible length, rebuilt when it grows)
            d["K"], d["V"], d["VT"] = self._t(self.T_ext * KQ, 0.0), self._t(kv, 0.0), self._t(kv, 0.0)
            d["SC"] = self._t(d["nq"] * self.T_ext, 0.0)   # softmax rows, packed [nq, R_LEN]
            d["MASK"] = self._t(self.T_ext)                # one mask row shared by every head
        self.reset_state()

    def reset_state(self):
        """Fresh conversation: the mask back to 'only the sink column visible', K' row 0 = the sinks."""
        T, ue = self.T, self.ue
        for d in self.layers:
            m = torch.full((self.T_ext,), NEG)
            m[0] = 0.0
            ue.dma_write(T.DMA_DEVICE_H2C, d["MASK"], m.to(torch.bfloat16), self.T_ext * BF16)
            k0 = torch.zeros(KQ)
            k0[self.HD:self.HD + d["nq"]] = d["sinks"]
            ue.dma_write(T.DMA_DEVICE_H2C, d["K"], k0.to(torch.bfloat16), KQ * BF16)

    # ---------------------------------------------------------------- emission helpers
    def _gemv(self, A, pair, N, K, OUT, bias=None, clamp=None, gelu=False):
        """y = act(x @ W^T + b) with W in IF4 (the vendor's fast M=1 path)."""
        kw = {}
        if clamp is not None:
            kw.update(clamp_enable=True, clamp_min=clamp[0], clamp_max=clamp[1])
        self.ue.quantized_matmat_core(M=1, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=pair[1], OUTPUT_DRAM_ADDR=OUT,
                                      SCALE_DRAM_ADDR=pair[0], C_DRAM_ADDR=bias, bias_mode="broadcast_N",
                                      data_type=self.U.TYPE.IF4, gelu_enable=gelu, **kw)

    def _sub_pair(self, pair, row0, K):
        """Rows row0.. of an IF4 [N, K] weight: data is K/2 bytes per row, scales K/64 bf16 per row."""
        return (pair[0] + row0 * (K // 64) * BF16, pair[1] + row0 * (K // 2))

    def _reduce(self):
        """X += shares[0] + ... + shares[n-1] (the shards' partial outputs, copied in by the host)."""
        ue, U, H = self.ue, self.U, self.H
        nb = H * BF16
        ue.ue_memcpy_from_dram(self.PARTS, nb, U.MEMCPY_TYPE.URAM.value, U.URAM_START_ADDR, U.URAM_SECTION.URAM_A.value)
        for j in range(1, self.parts_n):
            ue.ue_memcpy_from_dram(self.PARTS + j * nb, nb, U.MEMCPY_TYPE.URAM.value, U.URAM_START_ADDR,
                                   U.URAM_SECTION.URAM_B.value)
            ue.eltwise_add_core(0x00000, 0x80000, 0x00000, H)
        ue.ue_memcpy_from_dram(self.X, nb, U.MEMCPY_TYPE.URAM.value, U.URAM_START_ADDR, U.URAM_SECTION.URAM_B.value)
        ue.eltwise_add_core(0x00000, 0x80000, 0x00000, H)
        ue.ue_memcpy_to_dram(U.MEMCPY_TYPE.URAM.value, U.URAM_SECTION.URAM_A.value, U.URAM_START_ADDR, self.X, nb)

    def _rms(self, X, gamma, OUT):
        self.ue.rms_norm_core_dram(M=1, N=self.H, A_DRAM_ADDR=X, OUTPUT_DRAM_ADDR=OUT, GAMMA_DRAM_ADDR=gamma, eps=self.eps)

    def _scalar_to_lalu(self, vec_dram, onehot_dram):
        """LALU register := vec[i] for the one-hot e_i (dot product, ReLU-clamped: the weights are > 0)."""
        ue, U = self.ue, self.U
        SA, SB = 0x60000, 0xFE000
        ue.accelerator_memory_to_sram(vec_dram, SA, VEC)
        ue.accelerator_memory_to_sram(onehot_dram, SB, VEC)
        _, va = ue.sram_address_to_uram_address(SA)
        _, ma = ue.sram_address_to_uram_address(SB)
        ue.ue_arithmetic_op(0, 0, 1, U.LALU_CLAMP_RELU_A, U.LALU_CLAMP_RELU_B, U.LALU_MODE.CLAMP.value, 0,
                            U.URAM_SECTION.URAM_A.value, 0, 0, U.URAM_WRITE_SRC.URAM_WB_DISABLE.value,
                            U.UE_MODE.BF16_DOT_PRODUCT, 0, va, ma, 1, 0, VEC, 1)

    def _ident(self, n):
        if n not in self._idents:
            self._idents[n] = self._up(torch.eye(n))
        return self._idents[n]

    # ---------------------------------------------------------------- per-token mask update (start of A0)
    def _emit_mask_update(self, tpl_new, tpl_old):
        ue, sh = self.ue, self.T.ue_35bit_addr_shifter
        SA_NEW, SA_OLD = 0x40000, 0x40080
        ue.accelerator_memory_to_sram(tpl_new, SA_NEW, VEC)
        ue.accelerator_memory_to_sram(tpl_old, SA_OLD, VEC)
        for d in self.layers:                          # one shared mask row per layer
            row = d["MASK"]
            ue.generate_instruction_add_imm(R_MASK_NEW, sh(row), ue._ISA_TMP_REG)
            ue.sram_to_accelerator_memory(SA_NEW, row, VEC, general_reg_src=ue._ISA_TMP_REG)
            if d["sliding"]:
                ue.generate_instruction_add_imm(R_MASK_OLD, sh(row), ue._ISA_TMP_REG)
                ue.sram_to_accelerator_memory(SA_OLD, row, VEC, general_reg_src=ue._ISA_TMP_REG)

    # ---------------------------------------------------------------- segments
    def _emit_attention(self, d):
        T, ue, H, HD, U = self.T, self.ue, self.H, self.HD, self.U
        nq, sh = d["nq"], T.ue_35bit_addr_shifter
        self._rms(self.X, d["ln_in"], self.XN)
        self._gemv(self.XN, d["q"], nq * HD, H, self.Q, bias=d["q_b"])
        self._gemv(self.XN, d["k"], HD, H, self.Kt, bias=d["k_b"])
        self._gemv(self.XN, d["v"], HD, H, self.Vt, bias=d["v_b"])
        for h in range(nq):                            # rope(q_h) into the first half of Q' row h
            T._emit_rope_split_64(ue, self.Q + h * HD * BF16, self.QP + h * KQ * BF16, ue._fa_cos_pad_dram,
                                  ue._fa_neg_sin_pad_dram, ue._fa_sin_hi_pad_dram, HD, 0, load_cos_sin=(h == 0))
        T._emit_rope_split_64(ue, self.Kt, self.KR, ue._fa_cos_pad_dram, ue._fa_neg_sin_pad_dram,
                              ue._fa_sin_hi_pad_dram, HD, 0, load_cos_sin=False)
        # KV write at row SINK_COLS + pos (K' rows are KQ wide: their byte offset is in R_POS_KQ)
        for src, cache, reg in ((self.KR, d["K"], R_POS_KQ), (self.Vt, d["V"], ue._ISA_POS_K_REG)):
            ue.accelerator_memory_to_sram(src, 0x40000, HD)
            ue.generate_instruction_add_imm(reg, sh(cache), ue._ISA_TMP_REG)
            ue.sram_to_accelerator_memory(0x40000, cache, HD, general_reg_src=ue._ISA_TMP_REG)
        rc = ue._isa_reg_counter
        ue._isa_reg_counter = max(rc, 6)               # keep the cores' scratch registers off regs 1-5
        # V^T is kept packed, [64, R_LEN]: transpose the 64-row block that holds this position into it
        # (row stride R_LEN*2 bytes) ...
        ue.generate_instruction_shl(R_VT_STRIDE, R_LEN, 1)
        ue.generate_instruction_add_imm(R_BLK_V, sh(d["V"]), R_VT_IN)
        ue.generate_instruction_add_imm(R_BLK_VT, sh(d["VT"]), R_VT_OUT)
        ue.bf16_transpose_core(M=VEC, N=HD, INPUT_DRAM_ADDR=d["V"], OUTPUT_DRAM_ADDR=d["VT"],
                               IDENTITY_DRAM_ADDR=ue._identity_dram_addr, gpr_input_addr=R_VT_IN,
                               gpr_out_addr=R_VT_OUT, gpr_out_row_stride_reg=R_VT_STRIDE)
        # ... and when R_LEN has just grown (the host sets R_REBUILD at pos % 64 == 0) rebuild all of it with
        # the new row stride
        patch = ue._emit_forward_skip_jz(R_REBUILD, self._cap_base)
        ue.bf16_transpose_core(M=self.T_ext, N=HD, INPUT_DRAM_ADDR=d["V"], OUTPUT_DRAM_ADDR=d["VT"],
                               IDENTITY_DRAM_ADDR=ue._identity_dram_addr, gpr_M_reg=R_LEN)
        patch()
        # every head in ONE matmul: Q' [nq, KQ] . K'^T, softmax over [sink, visible keys], shared mask row,
        # rows packed [nq, R_LEN]; then P.V as ONE runtime-K matmul on the packed rows and the packed V^T
        ue.matmat_mul_core(M=nq, K=KQ, N=self.T_ext, A_DRAM_ADDR=self.QP, B_DRAM_ADDR=d["K"],
                           OUTPUT_DRAM_ADDR=d["SC"], softmax_enable=True, C_DRAM_ADDR=d["MASK"],
                           bias_mode="broadcast_N", gpr_N_reg=R_LEN)
        ue.matmat_mul_core(M=nq, K=self.T_ext, N=HD, A_DRAM_ADDR=d["SC"], B_DRAM_ADDR=d["VT"],
                           OUTPUT_DRAM_ADDR=self.ATT, gpr_K_reg=R_LEN)
        ue._isa_reg_counter = rc
        self._gemv(self.ATT, d["o"], H, nq * HD, self.SHARE_A, bias=d["o_b"])

    def _emit_router(self, d):
        self._reduce()
        self._rms(self.X, d["ln_post"], self.XN2)
        self.ue.matmat_mul_core(M=1, K=self.H, N=VEC, A_DRAM_ADDR=self.XN2, B_DRAM_ADDR=d["router_w"],
                                OUTPUT_DRAM_ADDR=self.LOGITS, C_DRAM_ADDR=d["router_b"], bias_mode="broadcast_N")

    def _emit_experts(self, d, prog_base, lg4_addr):
        ue, U, H, sh = self.ue, self.U, self.H, self.T.ue_35bit_addr_shifter
        nb = H * BF16
        # 4 slots the host overwrites with "R_EXP_i := subroutine of the picked expert" (one DMA with the logits)
        for i in range(self.TOPK):
            ue.generate_instruction_add_set(R_EXP[i], 0)
        # weights of the 4 picked experts: softmax over the 4 logits the host copied back (pads are -inf)
        ue.matmat_mul_core(M=1, K=VEC, N=VEC, A_DRAM_ADDR=lg4_addr, B_DRAM_ADDR=ue._identity_dram_addr,
                           OUTPUT_DRAM_ADDR=self.W4, softmax_enable=True)
        for i in range(self.TOPK):
            ue.pad_capture_to_64b_boundary()
            ue.generate_instruction_add_set(R_RET, sh(prog_base + (ue.capture_count + 2) * self.INST))
            ue.generate_instruction_jump_reg_abs(R_EXP[i])           # the expert subroutine writes Y
            self._scalar_to_lalu(self.W4, self.onehot[i])
            ue.ue_memcpy_from_dram(self.Y, nb, U.MEMCPY_TYPE.URAM.value, U.URAM_START_ADDR, U.URAM_SECTION.URAM_A.value)
            ue.start_queue_broadcast(U.UE_MODE.MUL_BROADCAST, U.BROADCAST_MODE.LALU_RESULT, 0x00000, 0x00000, H)
            if i:
                ue.ue_memcpy_from_dram(self.ACC, nb, U.MEMCPY_TYPE.URAM.value, U.URAM_START_ADDR,
                                       U.URAM_SECTION.URAM_B.value)
                ue.eltwise_add_core(0x00000, 0x80000, 0x00000, H)
            ue.ue_memcpy_to_dram(U.MEMCPY_TYPE.URAM.value, U.URAM_SECTION.URAM_A.value, U.URAM_START_ADDR,
                                 self.SHARE_E if i == self.TOPK - 1 else self.ACC, nb)

    def _emit_expert_sub(self, e):
        """One expert, this shard's neurons: Y = down(gelu(clamp(gate)) * clamp(up + 1, -6, 8)) (+ bias on shard 0).
        clamp(up, -7, 7) + 1 == clamp(up + 1, -6, 8): the +1 rides in the up bias."""
        ue, U, H, n = self.ue, self.U, self.H, e["n"]
        gate = self._sub_pair(e["gu"], 0, H)
        up = self._sub_pair(e["gu"], n, H)
        # the clamp rides in the gate GEMV (it must come before GELU); GELU (x * sigmoid(1.702 x)) is an
        # identity matmul with the LALU on its output, done on [n/64, 64] blocks with a 64x64 identity:
        # same values, 8.6 us instead of 45 us for the n x n identity. (The LALU is NOT applied on
        # broadcast/eltwise ops, so it cannot be done in URAM.)
        assert n % VEC == 0
        self._gemv(self.XN2, gate, n, H, self.G, bias=e["g_b"], clamp=(float("-inf"), self.limit))
        ue.matmat_mul_core(M=n // VEC, K=VEC, N=VEC, A_DRAM_ADDR=self.G, B_DRAM_ADDR=self._ident(VEC),
                           OUTPUT_DRAM_ADDR=self.HID, gelu_enable=True)
        self._gemv(self.XN2, up, n, H, self.Uu, bias=e["u_b1"], clamp=(-self.limit + 1.0, self.limit + 1.0))
        ue.ue_memcpy_from_dram(self.HID, n * BF16, U.MEMCPY_TYPE.URAM.value, U.URAM_START_ADDR, U.URAM_SECTION.URAM_A.value)
        ue.ue_memcpy_from_dram(self.Uu, n * BF16, U.MEMCPY_TYPE.URAM.value, U.URAM_START_ADDR, U.URAM_SECTION.URAM_B.value)
        ue.eltwise_mul_core(0x00000, 0x80000, 0x00000, n)
        ue.ue_memcpy_to_dram(U.MEMCPY_TYPE.URAM.value, U.URAM_SECTION.URAM_A.value, U.URAM_START_ADDR, self.HID, n * BF16)
        self._gemv(self.HID, e["down"], H, n, self.Y, bias=e["down_b"])
        ue.generate_instruction_jump_reg_abs(R_RET)

    def _emit_head(self, vocab_n):
        self._reduce()
        self._rms(self.X, self.final_norm, self.XN)
        self.ue.quantized_matmat_core(M=1, K=self.H, N=vocab_n, A_DRAM_ADDR=self.XN, B_DRAM_ADDR=self.lm_head[1],
                                      OUTPUT_DRAM_ADDR=self.HEAD_OUT, SCALE_DRAM_ADDR=self.lm_head[0],
                                      data_type=self.U.TYPE.IF4)

    # ---------------------------------------------------------------- compile
    def _place(self, emit):
        """Capture `emit(base)` as its own 64-byte-aligned program at `base`; returns `base`."""
        ue, T = self.ue, self.T
        ue.clear_capture_buffer()
        T.UnifiedEngine.start_capture(ue)
        ue._compile_mode = True
        base = (ue.get_program_dram_addr() + 63) & ~63
        self._cap_base = base                         # forward jumps inside are patched against it
        emit(base)
        ue._compile_mode = False
        T.UnifiedEngine.stop_capture(ue)
        ue.write_captured_instructions_to_dram(base)
        ue.allocate_program_dram(base - ue.get_program_dram_addr() + ue.get_capture_instruction_size_bytes() + 64)
        ue.clear_capture_buffer()
        return base

    def compile(self, parts_n, vocab_n, parts_addr=None):
        """Build the per-token program (73 halted segments) and one subroutine per (layer, expert).
        parts_addr: share another engine's buffer of copied-in shares (one copy per card)."""
        ue = self.ue
        self.INST = self.U.INSTRUCTION_SIZE_BYTES
        self._idents = {}
        self.parts_n = parts_n
        self.PARTS = parts_addr if parts_addr is not None else self._t(parts_n * self.H)
        self.HEAD_OUT = self._t(vocab_n)
        for li, d in enumerate(self.layers):         # (up bias + 1) for the shifted clamp
            w = torch.load(os.path.join(self.cdir, "layer_%02d.pt" % li), weights_only=False, map_location="cpu")
            for e, we in zip(d["experts"], w["experts"]):
                e["u_b1"] = self._up(we["gu_b"][e["n"]:].float() + 1.0)
        self._ident(VEC)                             # the block GELU of the expert subroutines
        self.sub_addr = [[self._place(lambda base, e=e: self._emit_expert_sub(e)) for e in d["experts"]]
                         for d in self.layers]
        segs = []                                    # (kind of the NEXT segment, capture index where it starts)
        self.lg4_addr = []
        INST, H = self.INST, self.H
        x_slots, tpl_slots = H * BF16 // INST, VEC * BF16 // INST
        self.pos_regs = [ue._ISA_POS_K_REG, ue._ISA_POS_ROPE_REG, R_LEN, R_BLK_V, R_BLK_VT, R_MASK_NEW, R_MASK_OLD,
                         R_POS_KQ, R_REBUILD]

        def data_slots(n):
            for _ in range(n):                         # never executed: the host writes data over them
                ue.generate_instruction_add_set(46, 0)

        def body(base):
            # header, written by the host with ONE DMA per token: X | TPL_NEW | TPL_OLD | 9 register slots
            data_slots(x_slots + 2 * tpl_slots)
            self.X = base
            tpl_new, tpl_old = base + x_slots * INST, base + (x_slots + tpl_slots) * INST
            segs.append(("A", ue.capture_count))
            for r in self.pos_regs:
                ue.generate_instruction_add_set(r, 0)
            for li, d in enumerate(self.layers):
                if li == 0:
                    self._emit_mask_update(tpl_new, tpl_old)
                else:
                    self._reduce()
                self._emit_attention(d)
                ue.generate_instruction_halt()
                segs.append(("R", ue.capture_count))
                self._emit_router(d)
                ue.generate_instruction_halt()
                ue.pad_capture_to_64b_boundary()
                lg4 = base + ue.capture_count * INST   # the host writes [4 logits | 4 register slots] with ONE DMA
                data_slots(VEC * BF16 // INST)
                self.lg4_addr.append(lg4)
                segs.append(("E", ue.capture_count))
                self._emit_experts(d, base, lg4)
                ue.generate_instruction_halt()
                segs.append(("A" if li + 1 < self.L else "H", ue.capture_count))
            self._emit_head(vocab_n)
            ue.generate_instruction_halt()

        self.X = None
        base = self._place(body)
        self.prog_base = base
        self.seg_kind = [k for k, _ in segs]
        self.seg_start = [base + idx * INST for _, idx in segs]
        self._addset_encoder()
        self.sel_bytes = [[[self._enc(R_EXP[i], self.T.ue_35bit_addr_shifter(a)) for i in range(self.TOPK)]
                           for a in row] for row in self.sub_addr]
        return len(self.seg_start)

    def _addset_encoder(self):
        """Learn where ADD_SET keeps its 32-bit immediate, so the host can produce the instruction bytes
        directly (no Python capture per token)."""
        ue, T = self.ue, self.T
        import random

        def capture(r, v):
            ue.clear_capture_buffer()
            T.UnifiedEngine.start_capture(ue)
            ue.generate_instruction_add_set(r, v)
            T.UnifiedEngine.stop_capture(ue)
            w = list(ue.capture_buffer[0].words)
            ue.clear_capture_buffer()
            return w

        self._addset = {}
        for r in set(self.pos_regs) | set(R_EXP):
            base = capture(r, 0)
            bitmap = []                                # immediate bit k -> (word, bit) in the descriptor
            for k in range(32):
                diff = [(i, a ^ b) for i, (a, b) in enumerate(zip(capture(r, 1 << k), base)) if a != b]
                assert len(diff) == 1 and bin(diff[0][1]).count("1") == 1, "ADD_SET bit %d is not a single bit" % k
                bitmap.append((diff[0][0], diff[0][1].bit_length() - 1))
            self._addset[r] = (base, bitmap)
            for v in [random.getrandbits(32) for _ in range(4)] + [0xFFFFFFFF]:
                assert self._enc(r, v) == b"".join(__import__("struct").pack("<I", x & 0xFFFFFFFF)
                                                   for x in capture(r, v)), "ADD_SET encoder mismatch"

    def _enc_slow(self, reg, value):
        import struct
        base, bitmap = self._addset[reg]
        w = list(base)
        value = int(value) & 0xFFFFFFFF
        for k, (i, b) in enumerate(bitmap):
            if value >> k & 1:
                w[i] |= 1 << b
        return b"".join(struct.pack("<I", v & 0xFFFFFFFF) for v in w)

    def _enc(self, reg, value):
        """ADD_SET reg, value as instruction bytes. Table form of _enc_slow (the instruction as one int,
        OR of 4 per-byte tables); built on first use per register and checked against _enc_slow."""
        tabs = self.__dict__.setdefault("_enc_tabs", {})
        if reg not in tabs:
            import random
            base, bitmap = self._addset[reg]
            nw = len(base)
            base_int = sum((v & 0xFFFFFFFF) << (32 * i) for i, v in enumerate(base))
            bit_int = [1 << (32 * i + b) for i, b in bitmap]
            byte_tabs = []
            for j in range(4):
                t = []
                for v in range(256):
                    x = 0
                    for t_bit in range(8):
                        if v >> t_bit & 1 and 8 * j + t_bit < len(bit_int):
                            x |= bit_int[8 * j + t_bit]
                    t.append(x)
                byte_tabs.append(t)
            tabs[reg] = (base_int, byte_tabs, 4 * nw)
            for v in [0, 0xFFFFFFFF] + [random.getrandbits(32) for _ in range(64)]:
                assert self._enc(reg, v) == self._enc_slow(reg, v), "table ADD_SET encoder mismatch"
        base_int, (t0, t1, t2, t3), nb = tabs[reg]
        value = int(value) & 0xFFFFFFFF
        return (base_int | t0[value & 255] | t1[value >> 8 & 255] | t2[value >> 16 & 255]
                | t3[value >> 24]).to_bytes(nb, "little")

    # ---------------------------------------------------------------- host commands (per token / per layer)
    def _tiny(self, addr, regs):
        """Write and run a small program of ADD_SETs at `addr` (registers persist across programs)."""
        ue, T = self.ue, self.T
        ue.clear_capture_buffer()
        T.UnifiedEngine.start_capture(ue)
        for r, v in regs:
            ue.generate_instruction_add_set(r, v)
        ue.generate_instruction_halt()
        T.UnifiedEngine.stop_capture(ue)
        ue.write_captured_instructions_to_dram(addr)
        ue.clear_capture_buffer()
        ue.start_execute_from_dram(addr)
        while ue.is_queue_busy():
            pass

    def set_position(self, pos, x_i16):
        """Token start, ONE DMA: the embedding row, both mask templates and the 9 position registers."""
        sh, HD = self.T.ue_35bit_addr_shifter, self.HD
        col = SINK_COLS + pos
        blk = col // VEC * VEC
        t_new = np.full(VEC, NEG_BF, dtype=np.int16)
        t_new[:col - blk + 1] = 0
        if pos >= self.window_len:
            c = col - self.window_len                 # this column leaves the sliding window now
            blk_old = c // VEC * VEC
            t_old = np.zeros(VEC, dtype=np.int16)
            t_old[:c - blk_old + 1] = NEG_BF
        else:
            blk_old, t_old = blk, t_new
        L = SINK_COLS + (pos + VEC) // VEC * VEC
        vals = [sh(col * HD * BF16), sh(pos * HD * BF16), min(L, self.T_ext), sh(blk * HD * BF16),
                sh(blk * BF16), sh(blk * BF16), sh(blk_old * BF16),
                sh(col * KQ * BF16), 1 if pos % VEC == 0 else 0]     # R_POS_KQ, R_REBUILD (R_LEN just grew)
        buf = (bytes(x_i16) + t_new.tobytes() + t_old.tobytes()
               + b"".join(self._enc(r, v) for r, v in zip(self.pos_regs, vals)))
        self._write(self.X, buf)

    def _write(self, addr, buf):
        """Small host-to-card write: one pwrite on the persistent fd when the runner gave one."""
        if self.fast_h2c is not None:
            t = time.perf_counter_ns()
            if os.pwrite(self.fast_h2c, buf, addr) != len(buf):
                raise OSError("short h2c write at 0x%x" % addr)
            self.prof_w[0] += time.perf_counter_ns() - t
            self.prof_w[1] += 1
        else:
            self.ue.dma_write(self.T.DMA_DEVICE_H2C, addr, torch.frombuffer(bytearray(buf), dtype=torch.uint8), len(buf))

    def set_experts(self, li, idx, logits):
        """The host's pick for layer li, ONE DMA: the 4 logits (copied back as data) and the 4 register
        slots that point the E segment at the picked experts' subroutines."""
        t = time.perf_counter_ns()
        if not hasattr(self, "_lg_pad"):
            self._lg_pad = np.full(VEC - self.TOPK, NEG_BF, dtype=np.int16).tobytes()
        buf = bf16_bits(logits).tobytes() + self._lg_pad + b"".join(
            self.sel_bytes[li][e][i] for i, e in enumerate(idx))
        self.prof_b[0] += time.perf_counter_ns() - t
        self.prof_b[1] += 1
        self._write(self.lg4_addr[li], buf)

    def set_experts_bf(self, li, idx, lg_bf):
        """set_experts with the 4 logits given as bf16 bit patterns, exactly as read from the card
        (no float round trip). Same bytes as set_experts; built with one struct pack."""
        pad = self.__dict__.get("_lg_pad")
        if pad is None:
            pad = self._lg_pad = np.full(VEC - self.TOPK, NEG_BF, dtype=np.int16).tobytes()
        sb = self.sel_bytes[li]
        self._write(self.lg4_addr[li], _PACK4H(*lg_bf) + pad + sb[idx[0]][0] + sb[idx[1]][1] + sb[idx[2]][2]
                    + sb[idx[3]][3])

    def run(self, addr, limit_s=30.0):
        import time
        ue = self.ue
        ue.start_execute_from_dram(addr)
        t = time.time()
        while ue.is_queue_busy():
            if time.time() - t > limit_s:
                raise RuntimeError("program at 0x%x did not finish" % addr)
        return ue.report_latency_in_us()
