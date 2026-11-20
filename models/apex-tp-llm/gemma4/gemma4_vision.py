"""Gemma 4 vision: the image encoder (27 layers + pooling + projection into the language model), run on one card.

Host side only prepares inputs: the image processor resizes the image and cuts it into 16x16 patches; the
host orders the patches, writes the one-hot position selector (which table column each patch reads) and the
attention mask for padding rows. Everything computed from the model's weights runs on the card:

    patch embedding   X = 2*W_p @ pixels - W_p @ 1  +  onehot @ pos_t^T       (position tables on the card)
    rotary tables     ROPE = onehot @ rope_t^T + rope_bias                      (once per image, per head)
    27 x  [ rms -> q/k/v (IF8) -> head-wise rms -> 2D RoPE -> attention -> o (IF8) -> rms -> + ,
            rms -> gelu(gate) * up (IF8) -> down (IF8) -> rms -> + ]
    3x3 pooling       sum of 9 row blocks (patches are ordered offset-in-block major)
    standardize       P @ diag(std_scale * sqrt(H) / 9) + (-std_bias * std_scale)
    projection        rms(no weight) -> embed_vision (bf16)  ->  soft tokens [n, 2816] for the LM

Weights come from extract_gemma4_vision.py (padded / permuted head layout, see there).
"""
import time

import numpy as np
import torch

BF16 = 2
NEG = -1.0e30
NP_MAX = 2560                     # padded patches of the largest image (280 soft tokens = 2520 patches)


def plan_patches(pixel_values, position_ids, pool=3):
    """HF processor output -> (pixels [NR, 768] f32, positions [NR, 2], order) with the real patches ordered
    offset-in-pooling-block major: row j * n_soft + m is patch j (= x%3 + 3*(y%3)) of soft token m."""
    pid = position_ids[0]
    real = ~(pid == -1).all(-1)
    px, pp = pixel_values[0][real].float(), pid[real].long()
    gx, gy = int(pp[:, 0].max()) + 1, int(pp[:, 1].max()) + 1
    assert gx % pool == 0 and gy % pool == 0, (gx, gy)
    m = (pp[:, 0] // pool) + (gx // pool) * (pp[:, 1] // pool)
    j = (pp[:, 0] % pool) + pool * (pp[:, 1] % pool)
    n_soft = (gx // pool) * (gy // pool)
    order = torch.empty(px.shape[0], dtype=torch.long)
    order[j * n_soft + m] = torch.arange(px.shape[0])
    return px[order], pp[order], n_soft, (gx, gy)


class VisionCard:
    """Card-side encoder on engine `ue` (port module `T`). alloc_w / alloc_t default to the engine's params /
    tensor allocators (the OllApeX integration passes allocators for memory the LM does not use)."""

    def __init__(self, T, ue, weights_path, alloc_w=None, alloc_t=None):
        self.T, self.ue = T, ue
        self.U = T.user_dma_core
        self.weights_path = weights_path
        self.alloc_w = alloc_w or ue.allocate_params_dram
        self.alloc_t = alloc_t or ue.allocate_tensor_dram
        self.A = None
        self.B = None
        self.progs = {}

    # ---------------------------------------------------------------- weights
    def _up(self, t):
        t = t.to(torch.bfloat16).contiguous()
        a = self.alloc_w(t.numel() * BF16)
        self.ue.dma_write(self.T.DMA_DEVICE_H2C, a, t, t.numel() * BF16)
        return a

    def _up_pair(self, pairs):
        s = torch.cat([p[0] for p in pairs]).contiguous()
        d = torch.cat([p[1] for p in pairs]).contiguous()
        sa = self.alloc_w(s.numel() * BF16)
        self.ue.dma_write(self.T.DMA_DEVICE_H2C, sa, s, s.numel() * BF16)
        da = self.alloc_w(d.numel())
        self.ue.dma_write(self.T.DMA_DEVICE_H2C, da, d, d.numel())
        return sa, da

    def load(self):
        if self.A is not None:
            return 0.0
        t0 = time.time()
        W = torch.load(self.weights_path, weights_only=False, map_location="cpu")
        self.d = d = W["dims"]
        pw = W["patch_w"].float()
        A = {"patch_w2": self._up(2 * pw), "patch_b": self._up(-pw.sum(1)),       # W_p @ (2p - 1)
             "pos_t": self._up(W["pos_t"]), "rope_t": self._up(W["rope_t"]), "rope_bias": self._up(W["rope_bias"]),
             "v_norm": self._up(W["v_norm"]), "std_diag": self._up(W["std_diag"]), "std_c": self._up(W["std_c"]),
             "embed_w": self._up(W["embed_w"]), "ones": self._up(torch.ones(d["HID"])), "layers": []}
        for L in W["layers"]:
            a = {k: self._up(L[k]) for k in ("q_norm", "k_norm", "ln_in", "ln_post_attn", "ln_pre_ff", "ln_post_ff")}
            a["qkv"] = self._up_pair([L["q"], L["k"], L["v"]])
            for k in ("o", "gate", "up", "down"):
                a[k] = self._up_pair([L[k]])
            A["layers"].append(a)
        self.A = A
        return time.time() - t0

    def _buffers(self):
        if self.B is not None:
            return self.B
        d, n, al = self.d, NP_MAX, self.alloc_t
        HID, NH, HDP, IP, POS = d["HID"], d["NH"], d["HDP"], d["IP"], d["POS"]
        QD, nm = NH * HDP, NP_MAX // 9 + 1
        self.B = B = {"PIXEL": al(n * 768 * BF16), "ONEHOT": al(n * 2 * POS * BF16), "X": al(n * HID * BF16),
                      "H": al(n * HID * BF16), "T": al(n * HID * BF16), "QKV": al(n * 3 * QD * BF16),
                      "Q": al(n * QD * BF16), "K": al(n * QD * BF16), "V": al(n * QD * BF16),
                      "ATTN": al(n * QD * BF16), "CONCAT": al(n * QD * BF16), "G": al(n * IP * BF16),
                      "UP": al(n * IP * BF16), "ROPE": al(NH * n * 2 * HDP * BF16), "BIAS": al(n * n * BF16),
                      "SCRATCH": al(((HDP + n) * n + n * HDP) * BF16), "EYE": al(64 * 64 * BF16),
                      "POOL": al(nm * HID * BF16), "Y": al(nm * HID * BF16), "YN": al(nm * HID * BF16),
                      "OUT": al(nm * d["OUT"] * BF16)}
        self.ue.dma_write(self.T.DMA_DEVICE_H2C, B["EYE"], torch.eye(64, dtype=torch.bfloat16).contiguous(),
                          64 * 64 * BF16)
        return B

    # ---------------------------------------------------------------- program
    def program(self, NP, n_soft, depth=None, start=0):
        """Card program for NP padded patches and n_soft soft tokens (cached per size). Unit tests only: depth < DEPTH
        stops after that many layers (X is read back then), start > 0 begins at that layer with X written by the host."""
        depth = self.d["DEPTH"] if depth is None else depth
        key = (NP, n_soft, depth, start)
        if key in self.progs:
            return self.progs[key]
        T, ue, U, d, A = self.T, self.ue, self.U, self.d, self.A
        B = self._buffers()
        HID, NH, HDP, IP, POS = d["HID"], d["NH"], d["HDP"], d["IP"], d["POS"]
        QD, QKV = NH * HDP, 3 * NH * HDP
        assert NP % 64 == 0 and NP <= NP_MAX and 9 * n_soft <= NP
        hs = NP * HDP * BF16                                    # one head in the head-major buffers
        MT, US, U0 = U.MEMCPY_TYPE.URAM.value, U.URAM_SECTION.URAM_A.value, U.URAM_START_ADDR
        rows_max = 65536 // HDP
        IF8 = U.TYPE.IF8

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

        saved_regs = ue._isa_reg_counter
        ue._isa_reg_counter = 8                                  # clear of the registers the LM programs use
        rM = ue.alloc_isa_reg()
        ue.clear_capture_buffer()
        T.UnifiedEngine.start_capture(ue)
        ue.generate_instruction_add_set(rM, NP)
        if start == 0:                                           # patch embedding + position embedding
            mm(NP, 768, HID, B["PIXEL"], A["patch_w2"], B["X"], rM, bias=A["patch_b"])
            mm(NP, 2 * POS, HID, B["ONEHOT"], A["pos_t"], B["T"], rM)
            eltwise(B["X"], B["T"], B["X"], NP * HID)
        # rotary table of every head row (head-major, like Q / K)
        for h in range(NH):
            mm(NP, 2 * POS, 2 * HDP, B["ONEHOT"], A["rope_t"], B["ROPE"] + h * NP * 2 * HDP * BF16, rM,
               bias=A["rope_bias"])
        for li in range(start, depth):
            L = A["layers"][li]
            rms(NP, HID, B["X"], B["H"], L["ln_in"], rM)
            mm(NP, HID, QKV, B["H"], L["qkv"], B["QKV"], rM)
            for h in range(NH):                                  # token-major -> head-major [NH, NP, HDP]
                for part, dst in ((0, B["Q"]), (QD, B["K"]), (2 * QD, B["V"])):
                    r0 = 0
                    while r0 < NP:
                        r = min(rows_max, NP - r0)
                        ue.ue_memcpy_from_dram(B["QKV"] + (r0 * QKV + part + h * HDP) * BF16, r * HDP * BF16, MT, U0,
                                               US, stride_bytes_per_chunk=HDP * BF16, stride_jump_bytes=QKV * BF16)
                        ue.ue_memcpy_to_dram(MT, US, U0, dst + h * hs + r0 * HDP * BF16, r * HDP * BF16)
                        r0 += r
            ue.generate_instruction_add_set(rM, NH * NP)
            rms(NH * NP, HDP, B["Q"], B["Q"], L["q_norm"], rM)
            rms(NH * NP, HDP, B["K"], B["K"], L["k_norm"], rM)
            rms(NH * NP, HDP, B["V"], B["V"], A["v_norm"], rM)
            for buf in (B["Q"], B["K"]):
                ue.rope_hf_core_dram(M=NH * NP, N=HDP, input_dram_addr=buf, output_dram_addr=buf,
                                     cos_dram_addr=B["ROPE"], sin_dram_addr=B["ROPE"] + HDP * BF16, gpr_M_reg=rM)
            ue.generate_instruction_add_set(rM, NP)
            for h in range(NH):
                ue.unified_attention_core(batch=NP, aligned_seq_len=NP, head_dim=HDP,
                                          Q_DRAM_ADDR=B["Q"] + h * hs, K_DRAM_ADDR=B["K"] + h * hs,
                                          V_DRAM_ADDR=B["V"] + h * hs, BIAS_DRAM_ADDR=B["BIAS"],
                                          OUTPUT_DRAM_ADDR=B["ATTN"] + h * hs, SCRATCH_DRAM_ADDR=B["SCRATCH"],
                                          IDENTITY_DRAM_ADDR=B["EYE"], q_scale=1.0)      # Gemma: no 1/sqrt(d)
            for h in range(NH):
                r0 = 0
                while r0 < NP:
                    r = min(rows_max, NP - r0)
                    ue.ue_memcpy_from_dram(B["ATTN"] + h * hs + r0 * HDP * BF16, r * HDP * BF16, MT, U0, US)
                    ue.ue_memcpy_to_dram(MT, US, U0, B["CONCAT"] + (r0 * QD + h * HDP) * BF16, r * HDP * BF16,
                                         stride_bytes_per_chunk=HDP * BF16, stride_jump_bytes=QD * BF16)
                    r0 += r
            mm(NP, QD, HID, B["CONCAT"], L["o"], B["T"], rM)
            rms(NP, HID, B["T"], B["T"], L["ln_post_attn"], rM)
            eltwise(B["X"], B["T"], B["X"], NP * HID)
            rms(NP, HID, B["X"], B["H"], L["ln_pre_ff"], rM)
            mm(NP, HID, IP, B["H"], L["gate"], B["G"], rM, gelu=True)
            mm(NP, HID, IP, B["H"], L["up"], B["UP"], rM)
            eltwise(B["G"], B["UP"], B["G"], NP * IP, mul=True)
            mm(NP, IP, HID, B["G"], L["down"], B["T"], rM)
            rms(NP, HID, B["T"], B["T"], L["ln_post_ff"], rM)
            eltwise(B["X"], B["T"], B["X"], NP * HID)
        if depth == d["DEPTH"]:
            blk = n_soft * HID
            eltwise(B["X"], B["X"] + blk * BF16, B["POOL"], blk)
            for j in range(2, 9):
                eltwise(B["POOL"], B["X"] + j * blk * BF16, B["POOL"], blk)
            ue.generate_instruction_add_set(rM, n_soft)
            mm(n_soft, HID, HID, B["POOL"], A["std_diag"], B["Y"], rM, bias=A["std_c"])
            rms(n_soft, HID, B["Y"], B["YN"], A["ones"], rM)
            mm(n_soft, HID, d["OUT"], B["YN"], A["embed_w"], B["OUT"], rM)
        ue.generate_instruction_halt()
        T.UnifiedEngine.stop_capture(ue)
        n_inst = ue.capture_count
        size = ue.get_capture_instruction_size_bytes()
        addr = ue.get_program_dram_addr()
        ue.write_captured_instructions_to_dram(addr)
        ue.allocate_program_dram(size)
        ue.clear_capture_buffer()
        ue._isa_reg_counter = saved_regs
        self.progs[key] = {"addr": addr, "instructions": n_inst, "bytes": size}
        return self.progs[key]

    # ---------------------------------------------------------------- run
    def encode(self, pixel_values, position_ids, depth=None, start=0, x_in=None):
        """HF processor output -> (soft tokens bf16 [n_soft, OUT] or X [NR, HID] if depth < DEPTH, info)."""
        T, ue, H2C = self.T, self.ue, self.T.DMA_DEVICE_H2C
        info = {"weights_upload_s": self.load()}
        d = self.d
        t0 = time.time()
        px, pp, n_soft, grid = plan_patches(pixel_values, position_ids, d["POOL"])
        NR = px.shape[0]
        NP = (NR + 63) // 64 * 64
        assert int(pp.max()) < d["POS"], "image grid larger than the position tables"
        pix = torch.zeros(NP, 768)
        pix[:NR] = px
        pix[NR:] = px[0]                                         # padding rows: copies of a real patch (finite)
        oh = torch.zeros(NP, 2 * d["POS"])
        oh[torch.arange(NR), pp[:, 0]] = 1
        oh[torch.arange(NR), d["POS"] + pp[:, 1]] = 1
        oh[NR:, pp[0, 0]] = 1
        oh[NR:, d["POS"] + pp[0, 1]] = 1
        bias = torch.zeros(NP, NP)
        bias[:, NR:] = NEG                                       # padding rows are never attended to
        info["host_prep_s"] = time.time() - t0
        t0 = time.time()
        new = (NP, n_soft, depth or d["DEPTH"], start) not in self.progs
        p = self.program(NP, n_soft, depth, start)
        info["compile_s"] = (time.time() - t0) if new else 0.0
        B = self.B
        for name, t in (("PIXEL", pix), ("ONEHOT", oh), ("BIAS", bias)):
            t = t.to(torch.bfloat16).contiguous()
            ue.dma_write(H2C, B[name], t, t.numel() * BF16)
        if start:                                                # unit tests: X of layer `start` from the host
            x = torch.zeros(NP, d["HID"])
            x[:NR] = x_in
            x[NR:] = x_in[0]
            x = x.to(torch.bfloat16).contiguous()
            ue.dma_write(H2C, B["X"], x, x.numel() * BF16)
        t0 = time.time()
        ue.start_execute_from_dram(p["addr"])
        while ue.is_queue_busy():
            time.sleep(0.005)
            if time.time() - t0 > 600:
                raise RuntimeError("the image encoder did not finish within 600 s")
        info["card_s"] = ue.report_latency_in_us() / 1e6
        info["wall_s"] = time.time() - t0
        if depth is not None and depth < d["DEPTH"]:
            out = torch.zeros(NR * d["HID"], dtype=torch.bfloat16)
            ue.dma_read(T.DMA_DEVICE_C2H, B["X"], out, out.numel() * BF16)
            out = out.view(NR, d["HID"])
        else:
            out = torch.zeros(n_soft * d["OUT"], dtype=torch.bfloat16)
            ue.dma_read(T.DMA_DEVICE_C2H, B["OUT"], out, out.numel() * BF16)
            out = out.view(n_soft, d["OUT"])
        info.update({"patches": NR, "padded": NP, "tokens": n_soft, "grid": grid, "instructions": p["instructions"],
                     "program_mb": p["bytes"] / 1e6, "finite": bool(torch.isfinite(out.float()).all())})
        return out, info
