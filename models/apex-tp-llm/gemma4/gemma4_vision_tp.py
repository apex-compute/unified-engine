"""Gemma 4 image encoder, tensor-parallel over the language model's shards (see gemma4_vision.py for the math).

Shard k of P holds the whole residual X and some heads + some MLP neurons. Per layer two halted segments, each
ending with this shard's partial output (SHARE, [NP, H]); the card's two engines are summed on the card, the host
carries every card's share into every card's PARTS buffer, and the next segment starts by summing PARTS itself:
    A_l : [l > 0: finish MLP l-1, X += rms(sum PARTS)]  rms, q/k/v (own heads), norms, RoPE, attention, o -> SHARE
          (l = 0 first: patch + position embedding, the heads' RoPE tables)
    M_l : [finish attention l, X += rms(sum PARTS)]  rms, gelu(gate) * up (own neurons), down          -> SHARE
    P   : finish MLP 26; shard 0 also pools, standardizes and projects into the LM          -> OUT [S, 2816]
One program per detail level S (70 / 140 / 280 soft tokens) with a FIXED patch layout (level_inputs), so any
image shape runs without recompiling. The host only prepares the inputs (patches, the one-hot position
selector, the padding mask) and carries bytes.
"""
import torch

from gemma4_vision import BF16, NEG, plan_patches

LEVELS = (70, 140, 280)


def level_np(S):
    return (9 * S + 63) // 64 * 64


def level_inputs(pixel_values, position_ids, S, POS, pool=3):
    """Inputs of level S in its fixed layout: row j * S + m = patch j of soft token m (m < n_soft); every other
    row is a padding copy of patch 0 that nobody attends to. Returns bf16 pixels [NP, 768], one-hot position
    selector [NP, 2*POS], mask [NP, NP], and n_soft."""
    px, pp, n_soft, _ = plan_patches(pixel_values, position_ids, pool)
    assert n_soft <= S and int(pp.max()) < POS
    NP = level_np(S)
    i = torch.arange(px.shape[0])
    rows = (i // n_soft) * S + i % n_soft
    pix = px[0].expand(NP, -1).clone()
    pix[rows] = px
    pos = pp[0].expand(NP, -1).clone()
    pos[rows] = pp
    oh = torch.zeros(NP, 2 * POS)
    oh[torch.arange(NP), pos[:, 0]] = 1
    oh[torch.arange(NP), POS + pos[:, 1]] = 1
    real = torch.zeros(NP, dtype=torch.bool)
    real[rows] = True
    bias = torch.zeros(NP, NP)
    bias[:, ~real] = NEG
    bf = lambda t: t.to(torch.bfloat16).contiguous()  # noqa: E731
    return bf(pix), bf(oh), bf(bias), n_soft


def split_plan(P, NH, NB, ratio=3.4):
    """Heads (contiguous, as even as possible) and 64-neuron MLP blocks per shard; the blocks balance the load,
    one head counting as `ratio` blocks of work."""
    heads = [NH // P + (k < NH % P) for k in range(P)]
    load = [h * ratio for h in heads]
    blocks = [0] * P
    for _ in range(NB):
        k = min(range(P), key=lambda j: load[j])
        blocks[k] += 1
        load[k] += 1
    h0 = [sum(heads[:k]) for k in range(P)]
    b0 = [sum(blocks[:k]) for k in range(P)]
    return [(list(range(h0[k], h0[k] + heads[k])), list(range(b0[k], b0[k] + blocks[k]))) for k in range(P)]


def pair_rows(pair, N, K, idx):
    s, d = pair
    idx = torch.as_tensor(idx)
    return s.view(N, K // 64)[idx].reshape(-1).contiguous(), d.view(N, K)[idx].reshape(-1).contiguous()


def pair_cols(pair, N, K, blocks):
    s, d = pair
    b = torch.as_tensor(blocks)
    return (s.view(N, K // 64)[:, b].reshape(-1).contiguous(),
            d.view(N, K // 64, 64)[:, b, :].reshape(-1).contiguous())


class VisionShard:
    def __init__(self, T, ue, W, k, P):
        self.T, self.ue, self.U = T, ue, T.user_dma_core
        self.k, self.P = k, P
        self.d = W["dims"]
        self.heads, self.blocks = split_plan(P, self.d["NH"], self.d["IP"] // 64)[k]
        self.W = W
        self.progs = {}

    def _up(self, t):
        t = t.to(torch.bfloat16).contiguous()
        a = self.ue.allocate_params_dram(t.numel() * BF16)
        self.ue.dma_write(self.T.DMA_DEVICE_H2C, a, t, t.numel() * BF16)
        return a

    def _up_pair(self, p):
        s, dd = p
        sa = self.ue.allocate_params_dram(s.numel() * BF16)
        self.ue.dma_write(self.T.DMA_DEVICE_H2C, sa, s.contiguous(), s.numel() * BF16)
        da = self.ue.allocate_params_dram(dd.numel())
        self.ue.dma_write(self.T.DMA_DEVICE_H2C, da, dd.contiguous(), dd.numel())
        return sa, da

    def load(self):
        W, d = self.W, self.d
        HID, HDP, IP, QD = d["HID"], d["HDP"], d["IP"], d["NH"] * d["HDP"]
        rows_h = torch.cat([torch.arange(h * HDP, (h + 1) * HDP) for h in self.heads])
        rows_n = torch.cat([torch.arange(b * 64, (b + 1) * 64) for b in self.blocks])
        hblk = [h * HDP // 64 + i for h in self.heads for i in range(HDP // 64)]
        pw = W["patch_w"].float()
        A = {"patch_w2": self._up(2 * pw), "patch_b": self._up(-pw.sum(1)), "pos_t": self._up(W["pos_t"]),
             "rope_t": self._up(W["rope_t"]), "rope_bias": self._up(W["rope_bias"]), "v_norm": self._up(W["v_norm"]),
             "layers": []}
        if self.k == 0:
            A.update({"std_diag": self._up(W["std_diag"]), "std_c": self._up(W["std_c"]),
                      "embed_w": self._up(W["embed_w"]), "ones": self._up(torch.ones(HID))})
        for L in W["layers"]:
            a = {n: self._up(L[n]) for n in ("q_norm", "k_norm", "ln_in", "ln_post_attn", "ln_pre_ff", "ln_post_ff")}
            q, k, v = (pair_rows(L[n], QD, HID, rows_h) for n in ("q", "k", "v"))
            a["qkv"] = self._up_pair((torch.cat([q[0], k[0], v[0]]), torch.cat([q[1], k[1], v[1]])))
            a["o"] = self._up_pair(pair_cols(L["o"], HID, QD, hblk))
            a["gate"] = self._up_pair(pair_rows(L["gate"], IP, HID, rows_n))
            a["up"] = self._up_pair(pair_rows(L["up"], IP, HID, rows_n))
            a["down"] = self._up_pair(pair_cols(L["down"], HID, IP, self.blocks))
            A["layers"].append(a)
        self.A = A
        self.W = None

    def buffers(self, n_cards, inputs=None, parts=None, np_max=None):
        """Card memory for the largest level. inputs / parts: engine 0's, shared by the card's engines; they go
        to the params region (the LM leaves ~200 MB free there, but only ~100 MB of tensor region on some engines)."""
        d, al, alp = self.d, self.ue.allocate_tensor_dram, self.ue.allocate_params_dram
        n = np_max or level_np(max(LEVELS))
        HID, HDP, POS = d["HID"], d["HDP"], d["POS"]
        hl, nl = len(self.heads), 64 * len(self.blocks)
        self.inputs = inputs or {"PIXEL": alp(n * 768 * BF16), "ONEHOT": alp(n * 2 * POS * BF16),
                                 "BIAS": alp(n * n * BF16)}
        self.PARTS = parts or alp(n_cards * n * HID * BF16)
        self.n_cards = n_cards
        self.B = {"X": al(n * HID * BF16), "H": al(n * HID * BF16), "T": al(n * HID * BF16),
                  "SHARE": al(n * HID * BF16), "QKV": al(n * 3 * hl * HDP * BF16), "Q": al(hl * n * HDP * BF16),
                  "K": al(hl * n * HDP * BF16), "V": al(hl * n * HDP * BF16), "ATTN": al(hl * n * HDP * BF16),
                  "CONCAT": al(n * hl * HDP * BF16), "G": al(n * nl * BF16), "UP": al(n * nl * BF16),
                  "ROPE": al(hl * n * 2 * HDP * BF16), "SCRATCH": al(((HDP + n) * n + n * HDP) * BF16),
                  "EYE": al(64 * 64 * BF16)}
        if self.k == 0:
            m = max(LEVELS)
            self.B.update({"POOL": al(m * HID * BF16), "Y": al(m * HID * BF16), "YN": al(m * HID * BF16),
                           "OUT": al(m * d["OUT"] * BF16)})
        self.ue.dma_write(self.T.DMA_DEVICE_H2C, self.B["EYE"], torch.eye(64, dtype=torch.bfloat16).contiguous(),
                          64 * 64 * BF16)

    def program(self, S, depth=None):
        """Level S: segments [("A", l), ("M", l)] * depth + [("P", -1)] and their start addresses. depth < DEPTH
        (unit tests) stops after that many layers; X is read back then."""
        key = (S, depth)
        if key in self.progs:
            return self.progs[key]
        T, ue, U, d, A, B, I = self.T, self.ue, self.U, self.d, self.A, self.B, self.inputs
        depth = d["DEPTH"] if depth is None else depth
        HID, HDP, POS = d["HID"], d["HDP"], d["POS"]
        hl, nl = len(self.heads), 64 * len(self.blocks)
        QD, QKV = hl * HDP, 3 * hl * HDP
        NP = level_np(S)
        hs = NP * HDP * BF16
        MT, US, U0 = U.MEMCPY_TYPE.URAM.value, U.URAM_SECTION.URAM_A.value, U.URAM_START_ADDR
        rows_max = 65536 // HDP
        IF8 = U.TYPE.IF8
        INST = U.INSTRUCTION_SIZE_BYTES

        def eltwise(a, b, out, total, mul=False):
            done = 0
            while done < total:
                c = min(65536, total - done)
                off = done * BF16
                ue.accelerator_memory_to_sram(a + off, 0x00000, c)
                ue.accelerator_memory_to_sram(b + off, 0x80000, c)
                (ue.eltwise_mul_core if mul else ue.eltwise_add_core)(0x00000, 0x80000, 0x00000, c)
                ue.sram_to_accelerator_memory(0x00000, out + off, c)
                done += c

        def sum_parts(out):
            total, stride, done = NP * HID, NP * HID * BF16, 0
            while done < total:
                c = min(65536, total - done)
                off = done * BF16
                ue.accelerator_memory_to_sram(self.PARTS + off, 0x00000, c)
                for j in range(1, self.n_cards):
                    ue.accelerator_memory_to_sram(self.PARTS + j * stride + off, 0x80000, c)
                    ue.eltwise_add_core(0x00000, 0x80000, 0x00000, c)
                ue.sram_to_accelerator_memory(0x00000, out + off, c)
                done += c

        def mm(M, K, N, a, w, out, reg, bias=None, gelu=False):
            if isinstance(w, tuple):
                ue.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=a, B_DRAM_ADDR=w[1], OUTPUT_DRAM_ADDR=out,
                                   C_DRAM_ADDR=bias, bias_mode="broadcast_N", is_B_quantized=True, data_type=IF8,
                                   SCALE_DRAM_ADDR=w[0], gelu_enable=gelu, gpr_M_reg=reg)
            else:
                ue.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=a, B_DRAM_ADDR=w, OUTPUT_DRAM_ADDR=out,
                                   C_DRAM_ADDR=bias, bias_mode="broadcast_N", gelu_enable=gelu, gpr_M_reg=reg)

        def rms(M, N, a, out, gamma, reg):
            T.UnifiedEngine.rms_norm_core_dram(ue, M=M, N=N, A_DRAM_ADDR=a, OUTPUT_DRAM_ADDR=out,
                                               GAMMA_DRAM_ADDR=gamma, gpr_M_reg=reg)

        segs = []
        saved_regs = ue._isa_reg_counter
        ue._isa_reg_counter = 8                        # clear of the registers the LM programs use
        rM = ue.alloc_isa_reg()

        def finish(gamma):                             # X += rms(sum of every card's share)
            sum_parts(B["T"])
            rms(NP, HID, B["T"], B["T"], gamma, rM)
            eltwise(B["X"], B["T"], B["X"], NP * HID)

        ue.clear_capture_buffer()
        T.UnifiedEngine.start_capture(ue)
        base = (ue.get_program_dram_addr() + 63) & ~63
        for li in range(depth):
            L = A["layers"][li]
            segs.append(("A", li, ue.capture_count))
            ue.generate_instruction_add_set(rM, NP)
            if li == 0:
                mm(NP, 768, HID, I["PIXEL"], A["patch_w2"], B["X"], rM, bias=A["patch_b"])
                mm(NP, 2 * POS, HID, I["ONEHOT"], A["pos_t"], B["T"], rM)
                eltwise(B["X"], B["T"], B["X"], NP * HID)
                for h in range(hl):
                    mm(NP, 2 * POS, 2 * HDP, I["ONEHOT"], A["rope_t"], B["ROPE"] + h * NP * 2 * HDP * BF16, rM,
                       bias=A["rope_bias"])
            else:
                finish(A["layers"][li - 1]["ln_post_ff"])
            rms(NP, HID, B["X"], B["H"], L["ln_in"], rM)
            mm(NP, HID, QKV, B["H"], L["qkv"], B["QKV"], rM)
            for h in range(hl):                        # token-major -> head-major [hl, NP, HDP]
                for part, dst in ((0, B["Q"]), (QD, B["K"]), (2 * QD, B["V"])):
                    r0 = 0
                    while r0 < NP:
                        r = min(rows_max, NP - r0)
                        ue.ue_memcpy_from_dram(B["QKV"] + (r0 * QKV + part + h * HDP) * BF16, r * HDP * BF16, MT, U0,
                                               US, stride_bytes_per_chunk=HDP * BF16, stride_jump_bytes=QKV * BF16)
                        ue.ue_memcpy_to_dram(MT, US, U0, dst + h * hs + r0 * HDP * BF16, r * HDP * BF16)
                        r0 += r
            ue.generate_instruction_add_set(rM, hl * NP)
            rms(hl * NP, HDP, B["Q"], B["Q"], L["q_norm"], rM)
            rms(hl * NP, HDP, B["K"], B["K"], L["k_norm"], rM)
            rms(hl * NP, HDP, B["V"], B["V"], A["v_norm"], rM)
            for buf in (B["Q"], B["K"]):
                ue.rope_hf_core_dram(M=hl * NP, N=HDP, input_dram_addr=buf, output_dram_addr=buf,
                                     cos_dram_addr=B["ROPE"], sin_dram_addr=B["ROPE"] + HDP * BF16, gpr_M_reg=rM)
            ue.generate_instruction_add_set(rM, NP)
            for h in range(hl):
                ue.unified_attention_core(batch=NP, aligned_seq_len=NP, head_dim=HDP,
                                          Q_DRAM_ADDR=B["Q"] + h * hs, K_DRAM_ADDR=B["K"] + h * hs,
                                          V_DRAM_ADDR=B["V"] + h * hs, BIAS_DRAM_ADDR=I["BIAS"],
                                          OUTPUT_DRAM_ADDR=B["ATTN"] + h * hs, SCRATCH_DRAM_ADDR=B["SCRATCH"],
                                          IDENTITY_DRAM_ADDR=B["EYE"], q_scale=1.0)
            for h in range(hl):
                r0 = 0
                while r0 < NP:
                    r = min(rows_max, NP - r0)
                    ue.ue_memcpy_from_dram(B["ATTN"] + h * hs + r0 * HDP * BF16, r * HDP * BF16, MT, U0, US)
                    ue.ue_memcpy_to_dram(MT, US, U0, B["CONCAT"] + (r0 * QD + h * HDP) * BF16, r * HDP * BF16,
                                         stride_bytes_per_chunk=HDP * BF16, stride_jump_bytes=QD * BF16)
                    r0 += r
            mm(NP, QD, HID, B["CONCAT"], L["o"], B["SHARE"], rM)
            ue.generate_instruction_halt()
            segs.append(("M", li, ue.capture_count))
            ue.generate_instruction_add_set(rM, NP)
            finish(L["ln_post_attn"])
            rms(NP, HID, B["X"], B["H"], L["ln_pre_ff"], rM)
            mm(NP, HID, nl, B["H"], L["gate"], B["G"], rM, gelu=True)
            mm(NP, HID, nl, B["H"], L["up"], B["UP"], rM)
            eltwise(B["G"], B["UP"], B["G"], NP * nl, mul=True)
            mm(NP, nl, HID, B["G"], L["down"], B["SHARE"], rM)
            ue.generate_instruction_halt()
        segs.append(("P", -1, ue.capture_count))
        ue.generate_instruction_add_set(rM, NP)
        finish(A["layers"][depth - 1]["ln_post_ff"])
        if self.k == 0 and depth == d["DEPTH"]:
            blk = S * HID
            eltwise(B["X"], B["X"] + blk * BF16, B["POOL"], blk)
            for j in range(2, 9):
                eltwise(B["POOL"], B["X"] + j * blk * BF16, B["POOL"], blk)
            ue.generate_instruction_add_set(rM, S)
            mm(S, HID, HID, B["POOL"], A["std_diag"], B["Y"], rM, bias=A["std_c"])
            rms(S, HID, B["Y"], B["YN"], A["ones"], rM)
            mm(S, HID, d["OUT"], B["YN"], A["embed_w"], B["OUT"], rM)
        ue.generate_instruction_halt()
        T.UnifiedEngine.stop_capture(ue)
        n_inst, size = ue.capture_count, ue.get_capture_instruction_size_bytes()
        ue.write_captured_instructions_to_dram(base)
        ue.allocate_program_dram(base - ue.get_program_dram_addr() + size + 64)
        ue.clear_capture_buffer()
        ue._isa_reg_counter = saved_regs
        p = {"kinds": [(kk, l) for kk, l, _ in segs], "start": [base + i * INST for _, _, i in segs],
             "instructions": n_inst, "mb": size / 1e6, "NP": NP}
        self.progs[key] = p
        return p

    def add_program(self, S, other_share):
        """Engine 0 only: SHARE += the card's other engine's SHARE (level S)."""
        key = ("add", S)
        if key in self.progs:
            return self.progs[key]
        T, ue = self.T, self.ue
        total, done = level_np(S) * self.d["HID"], 0
        ue.clear_capture_buffer()
        T.UnifiedEngine.start_capture(ue)
        while done < total:
            c = min(65536, total - done)
            off = done * BF16
            ue.accelerator_memory_to_sram(self.B["SHARE"] + off, 0x00000, c)
            ue.accelerator_memory_to_sram(other_share + off, 0x80000, c)
            ue.eltwise_add_core(0x00000, 0x80000, 0x00000, c)
            ue.sram_to_accelerator_memory(0x00000, self.B["SHARE"] + off, c)
            done += c
        ue.generate_instruction_halt()
        T.UnifiedEngine.stop_capture(ue)
        addr = (ue.get_program_dram_addr() + 63) & ~63
        ue.write_captured_instructions_to_dram(addr)
        ue.allocate_program_dram(addr - ue.get_program_dram_addr() + ue.get_capture_instruction_size_bytes() + 64)
        ue.clear_capture_buffer()
        self.progs[key] = addr
        return addr
