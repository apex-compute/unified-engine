#!/usr/bin/env python3
"""Qwen2.5-VL-3B language-model method group (36-layer GQA decoder).

``Qwen25VLLMMixin`` carries the LM methods and is mixed into
``Qwen25VL_UnifiedEngine`` in qwen2.5_vl_3b_test_new.py; it is never
instantiated on its own. Like the vision mixin it imports nothing from the
test module, so the split stays cycle-free.

PHASE B OF THE SHARED PARAMS WINDOW. Vision weights and LM weights occupy the
SAME addresses from 0x8000_0000 -- they never run together. ``lm_weight_init``
rewinds the params cursor and loads over whatever the encoder left there, so
anything the encoder produced must already be on the host by then (the 144
image embeddings are, via ``run_vision_encoder``).

ATTENTION SHAPE. Q/K/V are laid out head-major and attention runs ONE
unified_attention_core call per Q head, reading its group's K/V head straight
out of the cache. The previous build instead packed Q as [seq*group_size,
head_dim] and duplicated K/V per group, which made the causal bias
(seq*8)^2 -- 64x larger, 2 GiB at a 4096 context. Per-head keeps it seq^2.
"""
import json
import math
import os
import sys
import time

_SD = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SD)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SD)))

import torch

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, TYPE, UE_MODE, UE_VECTOR_SIZE, ue_35bit_addr_shifter)

LM_QUANT_PRECISION = "if4"


class Qwen25VLLMMixin:
    """LM methods for Qwen25VL_UnifiedEngine (see module docstring)."""

    # ---- weights -----------------------------------------------------------

    def _lm_dims(self) -> dict:
        fi = self._cfg["file_info"]
        qh = fi["num_kv_heads"] * fi["group_size"]
        return dict(H=fi["hidden_size"], AHD=fi["actual_head_dim"],
                    KVH=fi["num_kv_heads"], G=fi["group_size"], QH=qh,
                    MLP=fi["mlp_elements"], NL=fi["num_layers"],
                    VOCAB=fi["embedding_vocab"])

    def _read_lm_region(self) -> dict:
        bin_path = os.path.join(self.script_dir, self._cfg["paths"]["params"])
        json_path = bin_path.rsplit(".", 1)[0] + ".json"
        with open(json_path) as f:
            manifest = json.load(f)
        r = (manifest.get("regions") or {}).get("lm")
        if r is None:
            raise KeyError(f"no 'lm' region in {json_path}")
        return dict(bin_path=bin_path, base_offset=int(r["offset"]),
                    size=int(r["size"]), sections=r["manifest"])

    def lm_weight_init(self) -> None:
        """Load LM weights over the params window. Idempotent.

        Q/K/gate/up/down and the LM head are IF4; V and O stay BF16 (the
        previous build kept them full-precision for attention accuracy, and the
        bin is written that way). The embedding is NOT uploaded -- lookup is a
        host-side gather and only the selected rows are ever DMA'd.
        """
        if getattr(self, "_lm_weight_init_done", False):
            return
        d = self._lm_dims()
        region = self._read_lm_region()
        sec, sfx = region["sections"], LM_QUANT_PRECISION

        if getattr(self, "_vision_weight_init_done", False):
            self._loud("  [LM] reclaiming the params window from vision weights")
        self.reset_params_dram_addr()
        start = self.get_params_dram_addr()
        self._loud(f"  [LM] loading {d['NL']} layers ({sfx.upper()} Q/K/MLP, "
                   f"BF16 V/O) at 0x{start:X} ...")

        def need(k):
            if k not in sec:
                raise KeyError(f"LM weight {k!r} missing from params.bin")
            return sec[k]

        with open(region["bin_path"], "rb") as f:
            base = region["base_offset"]
            self.lm_layer_addrs = []
            for i in range(d["NL"]):
                pre = f"language_model.layers.{i}"
                la = {}
                for tag, key in (("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"),
                                 ("gate", "mlp.gate_proj"), ("up", "mlp.up_proj"),
                                 ("down", "mlp.down_proj")):
                    la[f"{tag}_scale"], la[f"{tag}_data"] = self._dma_if4(
                        f, need(f"{pre}.{key}.weight.{sfx}"), base, f"{pre}.{key}")
                for tag, key in (("v", "self_attn.v_proj"), ("o", "self_attn.o_proj")):
                    la[f"{tag}_weight"] = self._dma_bf16(
                        f, need(f"{pre}.{key}.weight"), base, f"{pre}.{key}")
                for tag, key in (("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"),
                                 ("v", "self_attn.v_proj")):
                    la[f"{tag}_bias"] = self._dma_bf16(
                        f, need(f"{pre}.{key}.bias"), base, f"{pre}.{key}.bias")
                la["ln1"] = self._dma_bf16(
                    f, need(f"{pre}.input_layernorm.weight"), base, f"{pre}.ln1")
                la["ln2"] = self._dma_bf16(
                    f, need(f"{pre}.post_attention_layernorm.weight"), base, f"{pre}.ln2")
                self.lm_layer_addrs.append(la)
                if (i + 1) % 8 == 0 or i == d["NL"] - 1:
                    self._loud(f"    layer {i + 1}/{d['NL']} loaded")

            self.final_norm_addr = self._dma_bf16(
                f, need("language_model.norm.weight"), base, "final_norm")
            self.lm_head_scale, self.lm_head_data = self._dma_if4(
                f, need(f"lm_head.weight.{sfx}"), base, "lm_head")

            # Embedding: host-side gather. Read once into host RAM; the device
            # never sees it, which is what keeps the LM inside 1808 MiB.
            s = need("language_model.embed_tokens.weight")
            f.seek(base + s["offset"])
            raw = f.read(s["size"])
        self.embedding_weight = torch.frombuffer(
            bytearray(raw), dtype=torch.bfloat16).reshape(d["VOCAB"], d["H"])

        self._lm_weight_end = self.get_params_dram_addr()
        used = self._lm_weight_end - start
        if self._lm_weight_end > self.PARAMS_LIMIT:
            raise MemoryError(
                f"LM weights overflow the params window: end "
                f"0x{self._lm_weight_end:X} > 0x{self.PARAMS_LIMIT:X}")
        self._lm_weight_init_done = True
        self._vision_weight_init_done = False   # vision weights are gone now
        self._loud(f"  [LM] weights loaded: {used / 2**20:.1f} MiB "
                   f"(embedding {len(raw) / 2**20:.1f} MiB kept on host)")

        self._ensure_tokenizer()

    def _ensure_tokenizer(self):
        """Load the tokenizer on demand.

        The VLM path needs it BEFORE lm_weight_init -- it builds the prompt (and
        from it the mRoPE positions) while the vision weights are still resident
        in the params window.
        """
        if not hasattr(self, "tokenizer"):
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(self.script_dir, self._cfg["paths"]["hf_model_dir"]),
                trust_remote_code=True)
        return self.tokenizer

    def get_embedding_for_tokens(self, token_ids) -> torch.Tensor:
        return self.embedding_weight[torch.as_tensor(list(token_ids),
                                                     dtype=torch.long)].contiguous()

    # ---- RoPE --------------------------------------------------------------

    def _mrope_dim_source(self) -> torch.Tensor:
        """Which of (t, h, w) each of the AHD/2 half-dims takes its position from.

        mrope_section [16, 24, 24] splits the 64 half-dims: the first 16 rotate
        by the temporal index, the next 24 by height, the last 24 by width. Text
        tokens set t = h = w, so the same table construction covers both cases.
        """
        sec = self._cfg["special"]["rope"]["mrope_section"]
        return torch.cat([torch.full((n,), i, dtype=torch.long)
                          for i, n in enumerate(sec)])

    def _build_rope_table(self, positions: torch.Tensor) -> torch.Tensor:
        """[n, 4*AHD/2] rows of [cos | cos | -sin | sin] for rope_hf_core.

        ``positions`` is either [n] (1-D, text-only) or [n, 3] carrying the
        mRoPE (t, h, w) triple per token -- image tokens occupy a 2-D grid, so
        their height and width indices differ from their sequence order.

        Half-width AHD/2 duplicated because rotate-half at width AHD pairs lane
        i with i+AHD/2; sin's lower half is pre-negated so the core's two
        elementwise passes are add-only.
        """
        d = self._lm_dims()
        half = d["AHD"] // 2
        theta = self._cfg["special"]["rope"]["theta"]
        inv = 1.0 / (theta ** (torch.arange(half, dtype=torch.float32) / half))
        if positions.dim() == 1:
            f = torch.outer(positions.float(), inv)
        else:
            # Per half-dim, take the position component that dim rotates by.
            f = positions.float()[:, self._mrope_dim_source()] * inv
        c, s = f.cos().to(torch.bfloat16), f.sin().to(torch.bfloat16)
        return torch.cat([c, c, -s, s], dim=1).contiguous()

    def load_rope_for_positions(self, positions: torch.Tensor, *,
                                decode: bool = False) -> None:
        """Upload the RoPE table: one row per position, NO per-head tiling.

        Every head's rope call reads rows 0..M-1, because the heads are rotated
        one at a time (their planes are head_rows apart and so not contiguous).
        """
        self.dma_to_accelerator_memory(
            self.LM_ROPE_DEC if decode else self.LM_ROPE_PRE,
            self._build_rope_table(positions).flatten())

    # ---- tensors -----------------------------------------------------------

    def lm_tensor_init(self) -> None:
        """Allocate LM activations and the KV cache.

        Vision tensors are NOT reused: vision runs to completion first and its
        output is on the host, so the whole tensor region is rewound and
        re-carved for the LM.
        """
        d = self._lm_dims()
        H, AHD, KVH, QH = d["H"], d["AHD"], d["KVH"], d["QH"]
        MLP, NL = d["MLP"], d["NL"]
        C, P = self.MAX_CONTEXT_SIZE, self.PREFILL_MAX_SEQ_LEN
        bpe = self.bytes_per_element
        self._tensor_dram_addr = self._tensor_dram_base

        def alloc(n, what):
            return self.allocate_tensor_dram(n * bpe, label=what)

        # Double-buffered layer I/O: layer li reads A and writes B when li is
        # even, and the reverse when odd, so no inter-layer copy is emitted.
        self.LM_IO_A = alloc(P * H, "lm.io_a")
        self.LM_IO_B = alloc(P * H, "lm.io_b")
        self.LM_PRE_NORM = alloc(P * H, "lm.pre_norm")
        self.LM_Q = alloc(P * QH * AHD, "lm.q")
        self.LM_K = alloc(P * KVH * AHD, "lm.k")
        self.LM_V = alloc(P * KVH * AHD, "lm.v")
        # Sized for the largest program (prefill at PREFILL_MAX_SEQ_LEN); each
        # program strides these planes by its own M.
        head_rows = P
        self.LM_HEAD_ROWS = head_rows
        self.LM_Q_HM = alloc(QH * head_rows * AHD, "lm.q_hm")
        self.LM_ATTN_HM = alloc(QH * head_rows * AHD, "lm.attn_hm")
        self.LM_ATTN_RESULT = alloc(P * QH * AHD, "lm.attn_result")
        self.LM_ATTN_PROJ = alloc(P * H, "lm.attn_proj")
        self.LM_RESIDUAL = alloc(P * H, "lm.residual")
        self.LM_MLP_NORM = alloc(P * H, "lm.mlp_norm")
        self.LM_MLP_GATE = alloc(P * MLP, "lm.mlp_gate")
        self.LM_MLP_UP = alloc(P * MLP, "lm.mlp_up")
        self.LM_MLP_MULT = alloc(P * MLP, "lm.mlp_mult")
        self.LM_MLP_DOWN = alloc(P * H, "lm.mlp_down")
        self.LM_OUT_NORM = alloc(H, "lm.out_norm")
        self.LOGITS = alloc(d["VOCAB"], "lm.logits")
        # Repetition-penalty bias: the LM-head matmul's C term, so the HW argmax
        # of (logits + bias) is the penalized token and no logits come back.
        self.PENALTY_BIAS = alloc(d["VOCAB"], "lm.penalty_bias")

        aligned_P = ((P + 63) // 64) * 64
        aligned_C = ((C + 63) // 64) * 64
        # SIZES ARE COMPUTED ONCE AND REUSED. lm_reset_attention_state has to
        # zero exactly these element counts; when it recomputed them from its
        # own expressions the two drifted, its scratch fill overran by 260 K
        # elements and wiped LM_IDENTITY (allocated right after) -- which made
        # the attention V-transpose produce zeros and every token garbage.
        n_bias = max(aligned_P * aligned_P, QH * aligned_C)
        # score/P inside the core is [aligned_seq, aligned_seq], so decode at a
        # long context -- not prefill -- sets the scratch size.
        # V.T [AHD, A] + score [A, A] + scaled_q [BATCH, AHD], where A is the
        # largest aligned_seq_len any call uses and BATCH the largest 64-aligned
        # batch. The scaled_q term is batch*AHD -- sizing it QH*AHD left the
        # buffer 6144 elements short at decode's A=2048, so the FIRST attention
        # call of each layer overran into LM_IDENTITY and every LATER call read
        # a corrupted identity: head 0 correct, heads 1..15 garbage.
        A = max(aligned_P, aligned_C)
        BATCH = P
        n_scratch = (AHD + A) * A + BATCH * AHD
        self.LM_BIAS = alloc(n_bias, "lm.bias")
        self.LM_SCRATCH = alloc(n_scratch, "lm.attn_scratch")
        self._lm_zero_sizes = dict(
            kv=NL * KVH * C * AHD, hm=QH * self.LM_HEAD_ROWS * AHD,
            bias=n_bias, scratch=n_scratch)
        # Guard between the attention scratch and IDENTITY. The kernel's scratch
        # extent is derived from its own arguments, so a sizing mistake here is
        # silent -- it lands on whatever is allocated next. IDENTITY was that
        # neighbour twice.
        self._lm_scratch_guard = alloc(64 * 1024, "lm.scratch_guard")
        self.LM_IDENTITY = alloc(UE_VECTOR_SIZE * UE_VECTOR_SIZE, "lm.identity")
        # One row per position; every head reads the same rows.
        self.LM_ROPE_PRE = alloc(P * 2 * AHD, "lm.rope_prefill")
        self.LM_ROPE_DEC = alloc(2 * AHD, "lm.rope_decode")

        # KV cache: [layer][kv_head][C][AHD], head-major so decode attention
        # reads it in place -- no per-step marshalling.
        self.KV_STRIDE_HEAD = C * AHD * bpe
        self.KV_STRIDE_LAYER = KVH * self.KV_STRIDE_HEAD
        self.LM_K_CACHE = alloc(NL * KVH * C * AHD, "lm.k_cache")
        self.LM_V_CACHE = alloc(NL * KVH * C * AHD, "lm.v_cache")

        end = self.get_tensor_dram_addr()
        if end > self.TENSOR_LIMIT:
            raise MemoryError(
                f"LM tensors need {(end - self._tensor_dram_base) / 2**20:.1f} MiB "
                f"but the tensor region is "
                f"{(self.TENSOR_LIMIT - self._tensor_dram_base) / 2**20:.1f} MiB. "
                f"Lower MAX_CONTEXT_SIZE / PREFILL_MAX_SEQ_LEN, or move "
                f"TENSOR_BASE down.")
        self.dma_to_accelerator_memory(
            self.LM_IDENTITY, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))
        self.dma_to_accelerator_memory(
            self.PENALTY_BIAS, torch.zeros(d["VOCAB"], dtype=torch.bfloat16))
        self._loud(f"  [LM] tensors: {self.get_tensor_dram_usage() / 2**20:.1f} MiB "
                   f"(KV {2 * NL * KVH * C * AHD * bpe / 2**20:.1f} MiB at ctx {C})")
        self.lm_reset_attention_state()

    def lm_reset_attention_state(self) -> None:
        """Zero EVERY buffer unified_attention_core reads, over its WHOLE span.

        Not an optimisation -- correctness. The kernel always runs the 64-aligned
        length, so rows and columns past the live sequence are multiplied on
        every call and discarded only afterwards by the -inf bias, which needs
        them FINITE. The pre-run DRAM clean fills 0xFF, which is NaN in bf16,
        and -inf + NaN = NaN: one stale pad element takes out an entire softmax
        row, and nothing shows it until the tokens come out wrong.
        gemma4_e2b does the same before every prefill, for the same reason.
        """
        n = self._lm_zero_sizes
        kv = torch.zeros(n["kv"], dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LM_K_CACHE, kv)
        self.dma_to_accelerator_memory(self.LM_V_CACHE, kv)
        hm = torch.zeros(n["hm"], dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LM_Q_HM, hm)
        self.dma_to_accelerator_memory(self.LM_ATTN_HM, hm)
        self.dma_to_accelerator_memory(
            self.LM_SCRATCH, torch.zeros(n["scratch"], dtype=torch.bfloat16))
        # The bias region past the live length is written ONLY here, and that is
        # exactly the region the kernel reads as padding.
        self.dma_to_accelerator_memory(
            self.LM_BIAS, torch.full((n["bias"],), float("-inf"),
                                     dtype=torch.bfloat16))
        # Re-write IDENTITY last: the attention V-transpose reads it, and it is
        # the buffer an over-long neighbouring fill would land on.
        self.dma_to_accelerator_memory(
            self.LM_IDENTITY, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))
        self._loud(f"  [LM] attention state zeroed "
                   f"(KV {2 * kv.numel() * 2 / 2**20:.0f} MiB + scratch + bias)")

    # ---- program emission --------------------------------------------------

    def _kv_addr(self, cache_base: int, layer: int, kv_head: int) -> int:
        return cache_base + layer * self.KV_STRIDE_LAYER + kv_head * self.KV_STRIDE_HEAD

    def _emit_layer(self, li: int, M: int, *, decode: bool, m_reg: int,
                    aligned_kv: int, in_addr: int, out_addr: int,
                    rope_base: int, aligned_kv_reg: int = None) -> int:
        """One decoder layer. Shared by prefill (M=seq) and decode (M=1)."""
        d = self._lm_dims()
        H, AHD, KVH, QH, G, MLP = (d["H"], d["AHD"], d["KVH"], d["QH"],
                                   d["G"], d["MLP"])
        bpe = self.bytes_per_element
        la = self.lm_layer_addrs[li]
        flops = 0

        def mm(K, N, A, tag, OUT, *, quant=True, bias=None, silu=False):
            """One projection.

            DECODE USES THE M=1 GEMV KERNEL for every IF4 weight. At M=1 the
            general matmat kernel pays for a row tile it does not fill;
            quantized_matmat_core streams the packed weights straight through
            DOT_PRODUCT instead. It accepts IF4/IF8/TQ4 only, so v_proj and
            o_proj -- the two the weight bin deliberately keeps BF16 for
            attention accuracy, with no .if4 bytes to give it -- stay on
            matmat_mul_core. Prefill has real rows to fill and stays on the
            general kernel throughout.
            """
            if decode and quant:
                return self.quantized_matmat_core(
                    M=M, K=K, N=N, A_DRAM_ADDR=A,
                    B_DRAM_ADDR=la[f"{tag}_data"], OUTPUT_DRAM_ADDR=OUT,
                    SCALE_DRAM_ADDR=la[f"{tag}_scale"], data_type=TYPE.IF4,
                    C_DRAM_ADDR=bias,
                    bias_mode="broadcast_N" if bias is not None else "broadcast_N",
                    silu_enable=silu) or 0
            kw = dict(M=M, K=K, N=N, A_DRAM_ADDR=A, OUTPUT_DRAM_ADDR=OUT,
                      silu_enable=silu, gpr_M_reg=m_reg)
            if bias is not None:
                kw.update(C_DRAM_ADDR=bias, bias_mode="broadcast_N")
            if quant:
                kw.update(B_DRAM_ADDR=la[f"{tag}_data"], is_B_quantized=True,
                          data_type=TYPE.IF4, SCALE_DRAM_ADDR=la[f"{tag}_scale"])
            else:
                kw.update(B_DRAM_ADDR=la[f"{tag}_weight"])
            return self.matmat_mul_core(**kw) or 0

        flops += self.rms_norm_core_dram(
            M=M, N=H, A_DRAM_ADDR=in_addr, OUTPUT_DRAM_ADDR=self.LM_PRE_NORM,
            GAMMA_DRAM_ADDR=la["ln1"], gpr_M_reg=m_reg) or 0
        flops += mm(H, QH * AHD, self.LM_PRE_NORM, "q", self.LM_Q, bias=la["q_bias"])
        flops += mm(H, KVH * AHD, self.LM_PRE_NORM, "k", self.LM_K, bias=la["k_bias"])
        flops += mm(H, KVH * AHD, self.LM_PRE_NORM, "v", self.LM_V,
                    quant=False, bias=la["v_bias"])

        rope_cos, rope_sin = rope_base, rope_base + AHD * bpe
        # ATTENTION BATCH MUST BE 64-ALIGNED. unified_attention_core consumes Q
        # in 64-row tiles, so a batch of 37 rounds DOWN to zero tiles and the
        # kernel writes nothing at all -- output stays whatever was there, which
        # read back as exact zeros. The head planes are therefore strided by the
        # aligned row count, not M, and rows M..head_rows-1 stay zero (they are
        # zeroed once by lm_reset_attention_state and never written), so the
        # extra tile computes finite garbage that nothing reads.
        # ONLY aligned_seq_len needs 64-alignment; batch does not. Head planes
        # are therefore packed at the live row count and attention runs exactly
        # M query rows -- no padding tile. (An earlier batch=37 run produced
        # zero output and I read that as an alignment rule; the real cause was
        # the under-sized scratch fixed above.)
        head_rows = M

        # Q token-major [M, QH, AHD] -> head-major [QH, M, AHD].
        self.bf16_permute_dram_core(QH, M, AHD, self.LM_Q, self.LM_Q_HM,
                                    write_grouped=True, group_stride_rows=head_rows)
        # RoPE PER HEAD, M live rows each. The head planes are head_rows apart,
        # so they are not contiguous and one M=QH*head_rows call would run over
        # the padding -- which is exactly how head 1 filled with NaN when the
        # table was tiled at the live length but the stride was larger. Per-head
        # calls also mean the table needs no per-head tiling at all: every head
        # reads rows 0..M-1 for positions 0..M-1.
        for qh in range(QH):
            flops += self.rope_hf_core_dram(
                M=M, N=AHD,
                input_dram_addr=self.LM_Q_HM + qh * head_rows * AHD * bpe,
                output_dram_addr=self.LM_Q_HM + qh * head_rows * AHD * bpe,
                cos_dram_addr=rope_cos, sin_dram_addr=rope_sin,
                gpr_M_reg=m_reg) or 0
        self.generate_instruction_add_set(m_reg, M)

        # K/V straight into the cache at their head planes. In prefill the
        # permute writes rows 0..M-1 of each head and leaves the rest of the
        # C-row plane untouched, which is exactly what group_stride_rows is for.
        k_base = self._kv_addr(self.LM_K_CACHE, li, 0)
        v_base = self._kv_addr(self.LM_V_CACHE, li, 0)
        if decode:
            # Rotate K BEFORE storing: the cache row address is computed at
            # runtime from gf_seq_len, so it cannot be a RoPE operand.
            # PER HEAD, one row each. Both KV heads share the single decode
            # position, so both must read table row 0. A single M=KVH call would
            # walk to row 1 -- past the end of a one-row table -- and rotate K
            # head 1 with whatever bytes follow it.
            for h in range(KVH):
                flops += self.rope_hf_core_dram(
                    M=1, N=AHD,
                    input_dram_addr=self.LM_K + h * AHD * bpe,
                    output_dram_addr=self.LM_K + h * AHD * bpe,
                    cos_dram_addr=rope_cos, sin_dram_addr=rope_sin,
                    gpr_M_reg=m_reg) or 0
            # One token: write at row gf_seq_len, computed at runtime.
            for h in range(KVH):
                for src, base, sram in ((self.LM_K, k_base, 0x10000),
                                        (self.LM_V, v_base, 0x20000)):
                    self.accelerator_memory_to_sram(src + h * AHD * bpe, sram, AHD)
                    self.generate_instruction_reg_mul_imm(
                        self.TMP_REG, self.gf_seq_len,
                        ue_35bit_addr_shifter(AHD * bpe))
                    self.generate_instruction_add_imm(
                        self.TMP_REG,
                        ue_35bit_addr_shifter(base + h * self.KV_STRIDE_HEAD),
                        self.TMP_REG)
                    self.sram_to_accelerator_memory(
                        sram_address=sram, accelerator_dram_address=0,
                        element_size=AHD, general_reg_src=self.TMP_REG)
        else:
            self.bf16_permute_dram_core(
                KVH, M, AHD, self.LM_K, k_base,
                write_grouped=True, group_stride_rows=self.MAX_CONTEXT_SIZE)
            self.bf16_permute_dram_core(
                KVH, M, AHD, self.LM_V, v_base,
                write_grouped=True, group_stride_rows=self.MAX_CONTEXT_SIZE)
            # Rotate K inside the cache, ONE HEAD AT A TIME. The heads' planes
            # are MAX_CONTEXT_SIZE rows apart (that is what reserves room for
            # decode to append), so they are NOT contiguous and a single
            # M=KVH*M call would walk off head 0's plane into unwritten rows.
            for h in range(KVH):
                flops += self.rope_hf_core_dram(
                    M=M, N=AHD,
                    input_dram_addr=k_base + h * self.KV_STRIDE_HEAD,
                    output_dram_addr=k_base + h * self.KV_STRIDE_HEAD,
                    cos_dram_addr=rope_cos, sin_dram_addr=rope_sin,
                    gpr_M_reg=m_reg) or 0

        # GQA attention.
        #
        # DECODE GROUPS THE 8 Q HEADS THAT SHARE A KV HEAD INTO ONE CALL. At
        # M=1 the head planes are one row each, so LM_Q_HM is already
        # [QH, AHD] contiguous and group g's heads are adjacent -- the packing
        # is free, no gather. That cuts the per-layer calls from QH to KVH and,
        # with them, the V.T transpose and K/V plane reads that every head in a
        # group would otherwise repeat identically (8x redundant).
        #
        # PREFILL STAYS PER-HEAD. Grouping there means packing Q as
        # [seq*G, AHD], which makes the causal bias (seq*G)^2 -- 64x larger,
        # the blow-up the head-major layout exists to avoid.
        if decode:
            groups = [(kv, kv * G * AHD * bpe, G) for kv in range(KVH)]
        else:
            groups = [(qh // G, qh * head_rows * AHD * bpe, M) for qh in range(QH)]
        self.LM_BATCH_ROWS = G if decode else M
        for kv_h, plane_off, batch in groups:
            batch_reg = self.alloc_isa_reg()
            self.generate_instruction_add_set(batch_reg, batch)
            # aligned_seq_len is a COMPILE-TIME bound -- it sizes the scratch
            # sub-buffers -- while the runtime KV length rides in a register.
            # Decode passes gf_aligned_seq_len, primed per step by the preamble;
            # prefill's length is fixed at compile time so it primes its own.
            if aligned_kv_reg is None:
                aligned_reg = self.alloc_isa_reg()
                self.generate_instruction_add_set(aligned_reg, aligned_kv)
            else:
                aligned_reg = aligned_kv_reg
            f = self.unified_attention_core(
                batch=batch, aligned_seq_len=aligned_kv, head_dim=AHD,
                Q_DRAM_ADDR=self.LM_Q_HM + plane_off,
                K_DRAM_ADDR=k_base + kv_h * self.KV_STRIDE_HEAD,
                V_DRAM_ADDR=v_base + kv_h * self.KV_STRIDE_HEAD,
                BIAS_DRAM_ADDR=self.LM_BIAS,
                OUTPUT_DRAM_ADDR=self.LM_ATTN_HM + plane_off,
                SCRATCH_DRAM_ADDR=self.LM_SCRATCH,
                IDENTITY_DRAM_ADDR=self.LM_IDENTITY,
                gpr_batch_reg=batch_reg, gpr_aligned_seq_len_reg=aligned_reg)
            self.release_isa_reg()
            if aligned_kv_reg is None:
                self.release_isa_reg()
            f = f if isinstance(f, (int, float)) else 0
            flops += f
            # Billed separately: the core returns FLOPs for the COMPILE-TIME
            # aligned_seq_len, which for decode is the MAX_CONTEXT_SIZE bound
            # that sizes the scratch -- not the live KV length the step runs.
            # Counting it as-is reported 45 GFLOP/token and a 194%-of-peak
            # throughput. run_decoder rescales it by the actual length.
            self._emit_attn_flops += f

        self.bf16_permute_dram_core(QH, M, AHD, self.LM_ATTN_HM,
                                    self.LM_ATTN_RESULT, write_grouped=False,
                                    group_stride_rows=head_rows)
        flops += mm(QH * AHD, H, self.LM_ATTN_RESULT, "o", self.LM_ATTN_PROJ,
                    quant=False)
        flops += self.eltwise_core_dram(
            M=M, N=H, dram_a=in_addr, dram_b=self.LM_ATTN_PROJ,
            dram_out=self.LM_RESIDUAL, mode=UE_MODE.ELTWISE_ADD,
            gpr_M_reg=m_reg) or 0
        flops += self.rms_norm_core_dram(
            M=M, N=H, A_DRAM_ADDR=self.LM_RESIDUAL,
            OUTPUT_DRAM_ADDR=self.LM_MLP_NORM, GAMMA_DRAM_ADDR=la["ln2"],
            gpr_M_reg=m_reg) or 0
        flops += mm(H, MLP, self.LM_MLP_NORM, "gate", self.LM_MLP_GATE, silu=True)
        flops += mm(H, MLP, self.LM_MLP_NORM, "up", self.LM_MLP_UP)
        flops += self.eltwise_core_dram(
            M=M, N=MLP, dram_a=self.LM_MLP_GATE, dram_b=self.LM_MLP_UP,
            dram_out=self.LM_MLP_MULT, mode=UE_MODE.ELTWISE_MUL,
            gpr_M_reg=m_reg) or 0
        flops += mm(MLP, H, self.LM_MLP_MULT, "down", self.LM_MLP_DOWN)
        flops += self.eltwise_core_dram(
            M=M, N=H, dram_a=self.LM_RESIDUAL, dram_b=self.LM_MLP_DOWN,
            dram_out=out_addr, mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=m_reg) or 0
        return flops

    def compile_prefill(self, seq_len: int, layer_size: int = None) -> int:
        """Emit a prefill program for exactly ``seq_len`` tokens.

        Compiled per prompt rather than made length-agnostic: the head-major
        permutes take compile-time shapes, and compiling 36 layers costs about a
        second. gemma4_e2b takes the same approach.
        """
        d = self._lm_dims()
        if seq_len > self.PREFILL_MAX_SEQ_LEN:
            raise ValueError(
                f"prompt is {seq_len} tokens, PREFILL_MAX_SEQ_LEN is "
                f"{self.PREFILL_MAX_SEQ_LEN}; tensors are sized for the latter")
        aligned = ((seq_len + 63) // 64) * 64
        t0 = time.perf_counter()
        self.reset_program_dram_addr()
        base = self.get_program_dram_addr()
        self.clear_inst_id()
        self.clear_capture_buffer()
        self.start_capture()
        prev = self._set_silent(True)
        self._emit_attn_flops = 0

        m_reg = self.gf_seq_len
        self.generate_instruction_add_set(m_reg, seq_len)
        flops = 0
        nl = d["NL"] if layer_size is None else layer_size
        for li in range(nl):
            in_addr = self.LM_IO_A if li % 2 == 0 else self.LM_IO_B
            out_addr = self.LM_IO_B if li % 2 == 0 else self.LM_IO_A
            flops += self._emit_layer(
                li, seq_len, decode=False, m_reg=m_reg, aligned_kv=aligned,
                in_addr=in_addr, out_addr=out_addr,
                rope_base=self.LM_ROPE_PRE)
        self.generate_instruction_halt()
        self._set_silent(prev)
        self.stop_capture()

        blob = bytearray()
        for inst in self.capture_buffer:
            blob.extend(inst.get_bytes())
        self.clear_capture_buffer()
        self._prefill_program = (base, bytes(blob))
        self._prefill_flops = int(flops)
        self._prefill_seq_len = seq_len
        self._prefill_layers = nl
        # Buffer the last emitted layer wrote (ping-pong: even count -> IO_A).
        self.LM_PREFILL_OUT = self.LM_IO_A if nl % 2 == 0 else self.LM_IO_B
        self.allocate_program_dram(len(blob))
        if base + len(blob) > self.DRAM_END:
            raise MemoryError(
                f"LM prefill program overruns the ISA region: "
                f"0x{base + len(blob):X} > 0x{self.DRAM_END:X}. Prefill and "
                f"decoder share it; shorten the prompt or enlarge the region.")
        self._loud(f"  [LM] prefill compiled for {seq_len} tokens: "
                   f"{len(blob) / 2**20:.2f} MiB at 0x{base:X}, "
                   f"{flops / 1e9:.1f} GFLOP, {time.perf_counter() - t0:.1f}s")
        return base

    def compile_decoder(self, layer_size: int = None) -> int:
        """Emit ONE position-agnostic decode program.

        Everything that depends on the step is a register: gf_seq_len is the KV
        write row, gf_aligned_seq_len bounds attention, and the RoPE table is
        re-uploaded per step. So this compiles once and every token jumps to it.
        """
        d = self._lm_dims()
        t0 = time.perf_counter()
        base = self.get_program_dram_addr()
        self.clear_inst_id()
        self.clear_capture_buffer()
        self.start_capture()
        prev = self._set_silent(True)
        self._emit_attn_flops = 0

        m_reg = self.gf_one
        self.generate_instruction_add_set(m_reg, 1)
        flops = 0
        nl = d["NL"] if layer_size is None else layer_size
        for li in range(nl):
            in_addr = self.LM_IO_A if li % 2 == 0 else self.LM_IO_B
            out_addr = self.LM_IO_B if li % 2 == 0 else self.LM_IO_A
            flops += self._emit_layer(
                li, 1, decode=True, m_reg=m_reg,
                # aligned_seq_len is DYNAMIC: MAX_CONTEXT_SIZE is only the
                # compile-time bound that sizes the scratch, while
                # gf_aligned_seq_len carries the live KV length each step, so a
                # short context does not pay for a full one.
                aligned_kv=self.MAX_CONTEXT_SIZE,
                aligned_kv_reg=self.gf_aligned_seq_len, in_addr=in_addr,
                out_addr=out_addr, rope_base=self.LM_ROPE_DEC)

        final_buf = self.LM_IO_A if nl % 2 == 0 else self.LM_IO_B
        self.LM_DECODE_OUT = final_buf
        flops += self.rms_norm_core_dram(
            M=1, N=d["H"], A_DRAM_ADDR=final_buf,
            OUTPUT_DRAM_ADDR=self.LM_OUT_NORM,
            GAMMA_DRAM_ADDR=self.final_norm_addr, gpr_M_reg=m_reg) or 0
        # LM head with the penalty vector as its bias term: the HW argmax of
        # (logits + bias) is the answer, so write_back_disable keeps the 151936
        # logits off the bus entirely. An all-zero bias is plain greedy.
        flops += self.quantized_matmat_core(
            M=1, K=d["H"], N=d["VOCAB"], A_DRAM_ADDR=self.LM_OUT_NORM,
            B_DRAM_ADDR=self.lm_head_data, OUTPUT_DRAM_ADDR=self.LOGITS,
            SCALE_DRAM_ADDR=self.lm_head_scale, data_type=TYPE.IF4,
            C_DRAM_ADDR=self.PENALTY_BIAS, bias_mode="broadcast_N",
            write_back_disable=True) or 0
        self.generate_instruction_add_inc(self.gf_seq_len)
        self.generate_instruction_halt()
        self._set_silent(prev)
        self.stop_capture()

        blob = bytearray()
        for inst in self.capture_buffer:
            blob.extend(inst.get_bytes())
        self.clear_capture_buffer()
        self._decoder_program = (base, bytes(blob))
        self._decoder_flops = int(flops)
        # Split so a step can be priced at its real KV length: everything except
        # attention is fixed per token, and attention scales linearly with
        # aligned_seq_len.
        self._decoder_flops_fixed = int(flops - self._emit_attn_flops)
        self._decoder_attn_per_aligned = (
            self._emit_attn_flops / self.MAX_CONTEXT_SIZE
            if self.MAX_CONTEXT_SIZE else 0.0)
        self.allocate_program_dram(len(blob))
        self._decoder_preamble = self.allocate_program_dram(64 * 8)
        if base + len(blob) > self.DRAM_END:
            raise MemoryError(
                f"LM decoder program overruns the ISA region: "
                f"0x{base + len(blob):X} > 0x{self.DRAM_END:X}. Prefill and "
                f"decoder share it; shorten the prompt or enlarge the region.")
        self._loud(f"  [LM] decoder compiled: {len(blob) / 2**20:.2f} MiB at "
                   f"0x{base:X}, {self._decoder_flops_fixed / 1e9:.2f} GFLOP/token "
                   f"+ attention scaled by context, "
                   f"{time.perf_counter() - t0:.1f}s")
        return base

    # ---- execution ---------------------------------------------------------

    def _upload(self, program) -> int:
        addr, blob = program
        self._next_program_dram_addr = addr
        self.dma_write(DMA_DEVICE_H2C, addr, blob, len(blob))
        return addr

    def run_prefill(self, tokens, image_embeddings=None, positions=None) -> None:
        """Embed the prompt (splicing image tokens if given), then run prefill."""
        d = self._lm_dims()
        seq_len = len(tokens)
        aligned = ((seq_len + 63) // 64) * 64
        self.seq_len = seq_len

        emb = self.get_embedding_for_tokens(tokens)
        if image_embeddings is not None:
            slots = [i for i, t in enumerate(tokens) if t == 151655]
            n = min(len(slots), image_embeddings.shape[0])
            for i in range(n):
                emb[slots[i]] = image_embeddings[i]
            self._loud(f"  [LM] spliced {n} image embeddings at {slots[:4]}"
                       f"{'...' if len(slots) > 4 else ''}")
        self.dma_to_accelerator_memory(self.LM_IO_A, emb.flatten())

        self.load_rope_for_positions(
            positions if positions is not None else torch.arange(seq_len))
        # Causal mask over the aligned square; columns past the real prompt are
        # masked too, so the alignment padding cannot be attended to.
        bias = torch.full((aligned, aligned), float("-inf"), dtype=torch.bfloat16)
        bias.masked_fill_(torch.tril(torch.ones(aligned, aligned, dtype=torch.bool)), 0.0)
        bias[:, seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LM_BIAS, bias)

        addr = self._upload(self._prefill_program)
        t0 = time.perf_counter()
        self.start_execute_from_dram(addr)
        self.wait_queue(180.0)
        us = self.report_latency_in_us()
        # Prefill's compile-time shapes ARE what runs -- it is compiled for this
        # exact seq_len -- so the cores' FLOP sum needs no rescaling, unlike
        # decode's. The guard is here anyway: >100% of peak is impossible and
        # means the count is billed at shapes the hardware did not execute.
        gflops = self._prefill_flops / (us * 1e-6) / 1e9 if us else 0.0
        peak = self.vis_peak_gflops()
        warn = (f"  [warn] {100 * gflops / peak:.0f}% of peak is impossible"
                if peak and gflops > peak * 1.005 else "")
        self._loud(f"  [LM] prefill {seq_len} tokens: {us / 1e6:.2f}s HW, "
                   f"{self._prefill_flops / 1e9:.1f} GFLOP, {gflops:.1f} GFLOPS"
                   f"{f' = {100 * gflops / peak:.0f}% of peak' if peak else ''} "
                   f"({time.perf_counter() - t0:.2f}s wall){warn}")
        self._latency_prefill_us = us
        self._prefill_gflops = gflops
        self._prefill_wall_s = time.perf_counter() - t0
        self._prefill_seq_len_run = seq_len

    def run_decoder(self, first_token: int, max_new_tokens: int = 256) -> tuple[int, str]:
        """Greedy decode until EOS, ``max_new_tokens``, or the context fills.

        The cap is NOT cosmetic. If anything upstream makes the argmax garbage
        the sampled tokens never hit a stop id, and an uncapped loop then runs
        thousands of FPGA steps -- which looks exactly like a hung board. Killing
        the host mid-step then leaves the engine executing with its queue busy,
        so the NEXT run looks hung too.
        """
        d = self._lm_dims()
        stop = {151643, 151645, self._end_of_turn_token_id}
        addr, _ = self._decoder_program
        self._upload(self._decoder_program)

        token, out = first_token, []
        total_us = 0.0
        step_flops = 0.0
        self._decode_step_us = []
        t0 = time.perf_counter()

        # Live status bar: pin the bottom terminal row via an ANSI scroll region
        # so generated tokens stream above it while the counter refreshes in
        # place. Everything is on stdout -- tokens scroll inside rows 1..rows-1,
        # the status writes row `rows` with cursor save/restore -- so it never
        # clobbers the streamed text. TTY only; skipped when piped or redirected,
        # which is why the escape codes never reach a log file.
        import shutil
        start_seq = self.seq_len
        use_status = sys.stdout.isatty()

        def _status_setup():
            rows = shutil.get_terminal_size().lines
            sys.stdout.write(f"\033[1;{rows - 1}r")    # scroll region = rows 1..rows-1
            sys.stdout.write(f"\033[{rows - 1};1H")    # park cursor at its bottom
            sys.stdout.flush()

        def _status_update():
            rows = shutil.get_terminal_size().lines
            n_done = self.seq_len - start_seq
            elapsed = time.perf_counter() - t0
            rate = n_done / elapsed if elapsed > 0 else 0.0
            hw = (1e6 / (total_us / n_done)) if n_done and total_us else 0.0
            sys.stdout.write("\0337")                  # save cursor
            sys.stdout.write(f"\033[{rows};1H\033[2K")  # bottom row, clear it
            sys.stdout.write(
                f" decoding… {n_done} tokens  (pos {self.seq_len}/{self.MAX_CONTEXT_SIZE})"
                f"  {elapsed:.1f}s  {rate:.2f} tok/s  (HW {hw:.2f} tok/s)")
            sys.stdout.write("\0338")                  # restore cursor
            sys.stdout.flush()

        def _status_teardown():
            rows = shutil.get_terminal_size().lines
            sys.stdout.write("\033[r")                 # reset scroll region
            sys.stdout.write(f"\033[{rows};1H\033[2K")  # clear the status row
            sys.stdout.flush()

        if use_status:
            _status_setup()
        limit = self.MAX_CONTEXT_SIZE
        if max_new_tokens is not None:
            limit = min(limit, self.seq_len + max_new_tokens)
        prev_silent = self._set_silent(True)
        while self.seq_len < limit:
            step_pos = self.seq_len            # KV row this step writes
            self.seq_len += 1
            aligned = ((self.seq_len + 63) // 64) * 64

            self.dma_to_accelerator_memory(
                self.LM_IO_A, self.get_embedding_for_tokens([token]).flatten())
            # After an image the next text position is NOT step_pos: the image
            # occupied max(h, w) positions, not one per token, so mrope_delta
            # carries the gap. All three components advance together for text.
            pos = step_pos + getattr(self, "_rope_offset", 0)
            self.load_rope_for_positions(
                torch.tensor([[pos, pos, pos]]), decode=True)
            # Mask everything past the tokens actually written to the cache.
            # Cover EVERY row the kernel reads: batch is the 64-aligned tile,
            # not QH. Rows past the live query are unread, but an all -inf row
            # softmaxes to NaN, so give them the same mask as row 0.
            # ROW STRIDE IS THE RUNTIME aligned LENGTH, not the compile-time
            # bound. Every grouped query row needs its own mask row, and the
            # kernel indexes them by the live aligned_seq_len; laying them out
            # at MAX_CONTEXT_SIZE made row 0 correct and rows 1..G-1 read the
            # wrong slice -- which showed up as the first head of each KV group
            # being right and the other seven wrong.
            bias = torch.full((self.LM_BATCH_ROWS, aligned), float("-inf"),
                              dtype=torch.bfloat16)
            bias[:, :self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.LM_BIAS, bias)

            # Per-step preamble: prime the position registers, jump to the body.
            self.clear_inst_id()
            self.start_capture()
            self.generate_instruction_add_set(self.gf_seq_len, step_pos)
            self.generate_instruction_add_set(self.gf_aligned_seq_len, aligned)
            self.generate_instruction_jump_abs(ue_35bit_addr_shifter(addr))
            self.stop_capture()
            self.write_captured_instructions_to_dram(self._decoder_preamble)
            self.clear_capture_buffer()

            self.start_execute_from_dram(self._decoder_preamble)
            self.wait_queue(30.0)
            step_us = self.report_latency_in_us()
            total_us += step_us
            # Per-step latencies: the FIRST is the peak-speed datapoint (shortest
            # KV history), and the spread across steps shows the context growth.
            self._decode_step_us.append(step_us)
            step_flops += (self._decoder_flops_fixed
                           + self._decoder_attn_per_aligned * aligned)

            token = self.get_arg_max_index()
            if token in stop:
                if use_status:
                    _status_teardown()
                break
            piece = self.tokenizer.decode([token])
            out.append(piece)
            self._set_silent(False)
            print(piece, end="", flush=True)
            self._set_silent(True)
            if use_status:
                _status_update()
        else:
            # Loop ran to the token cap or the context end rather than breaking.
            if use_status:
                _status_teardown()
        self._set_silent(prev_silent)

        wall = time.perf_counter() - t0
        n = max(1, len(out))
        gflops = (step_flops / (total_us * 1e-6) / 1e9) if total_us else 0.0
        peak = self.vis_peak_gflops()
        note = ""
        if peak and gflops > peak * 1.005:
            # Cannot happen physically; means the FLOP count is billed at
            # shapes the hardware did not run.
            note = f"  [warn] {100 * gflops / peak:.0f}% of peak is impossible"
        # Emitted BEFORE the [LM] line on purpose: model_auto_test slices the
        # generated text from "--- Decode run ---" up to "\nDecoder done in",
        # so anything printed in between lands inside what it scores as model
        # output. Keeping the marker here leaves that region pure token stream.
        self._loud(f"\nDecoder done in {wall:.2f} seconds, "
                   f"speed: {n / wall:.2f} tokens/s, total {self.seq_len} tokens.")

        # First-token speed from the HW counter is the PEAK: the KV history is
        # shortest on step 1, so every later step attends further and is slower.
        first_toks = (1e6 / self._decode_step_us[0]) if self._decode_step_us else 0.0
        self._loud(f"\n  [LM] {n} tokens in {wall:.2f}s = {n / wall:.2f} tok/s "
                   f"(1st decode token: {first_toks:.2f} tok/s, "
                   f"{step_flops / n / 1e9:.2f} GFLOP/token, {gflops:.1f} GFLOPS"
                   f"{f' = {100 * gflops / peak:.0f}% of peak' if peak else ''})"
                   f"{note}")
        self._decode_total_us = total_us
        self._decode_step_flops = step_flops
        self._decode_wall_s = wall
        self._decode_n = n
        self._decode_gflops = gflops
        self._decoded_text = "".join(out)
        self._prompt_tokens = getattr(self, "_prompt_tokens", None)
        return self.seq_len, self._decoded_text
