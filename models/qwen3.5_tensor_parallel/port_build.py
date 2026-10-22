"""Build a patched copy of the stock Qwen3.5-2B port in the work directory ($APEX_TP_WORK/models/<name>/).

The stock port in models/qwen3.5_2b is never modified. Each change is a named text patch:

  silu     SiLU tiled as 64-wide rows instead of a full-width identity matmul
  kpad     padded key buffer built with one transpose instead of per-head spreader matmuls
  attn1    decode-time attention computed for the single query row (1 x T) with a one-row mask
  noeye    do not upload the conv_dim x conv_dim identity (only the old SiLU needed it)
  heads    allow more value heads than key heads in the linear-attention layer (9B: 16/32, 27B: 16/48)
  shard    let one engine hold a contiguous range of layers, with the LM head optional
  recgrp   state update in groups of heads (Q35_REC_GROUP) instead of one wide matmul
  tp       tensor-parallel mode: halt at the two residual additions of every layer
  abfuse   one projection for both gate vectors of the linear-attention layer
  vtshare  transpose V once per KV head instead of once per query head

Usage:
  python port_build.py --name q35_9b  --cfg9b silu kpad attn1 noeye heads shard recgrp tp abfuse vtshare
  python port_build.py --name q38_27b --hfcfg <model dir>/config.json silu kpad attn1 noeye heads shard recgrp tp abfuse vtshare
"""
import argparse
import json
import os
import re

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E = os.path.expanduser(os.environ.get("APEX_TP_WORK", "~/apex_tp"))
SRC_DIR = os.path.join(R, "models", "qwen3.5_2b")


def replace_once(text, old, new, what):
    if text.count(old) != 1:
        raise SystemExit("%s: expected exactly one match, found %d for: %r" % (what, text.count(old), old[:80]))
    return text.replace(old, new)


def p_silu(t):
    return replace_once(
        t,
        "    M = numel // N\n"
        "    OUT = _alloc_tensor(ue, numel)\n"
        "    I_DRAM = _get_identity_dram(ue, N)\n",
        "    N = UE_VECTOR_SIZE   # opt: tile as vector-width rows (was the full row width)\n"
        "    assert numel % N == 0\n"
        "    M = numel // N\n"
        "    OUT = _alloc_tensor(ue, numel)\n"
        "    I_DRAM = ue._identity_dram_addr\n", "silu")


def p_kpad(t):
    a = t.index("    head_stride_bytes  = Dk * BF16")
    end = "                           is_B_quantized=False)\n"
    b = t.index(end, a) + len(end)
    new = (
        "    nelem = num_heads * Dk                     # opt: one transpose instead of per-head spreader matmuls\n"
        "    stage = getattr(ue, '_k_stage_dram', None)\n"
        "    if stage is None:\n"
        "        stage = _upload(ue, torch.zeros(N * nelem, dtype=torch.float32))\n"
        "        ue._k_stage_dram = stage\n"
        "    ue.accelerator_memcpy(K_L2_DRAM, stage, nelem * BF16)\n"
        "    ue.bf16_transpose_core(M=N, N=nelem, INPUT_DRAM_ADDR=stage, OUTPUT_DRAM_ADDR=k_pad_dram)\n")
    return t[:a] + new + t[b:]


def p_attn1(t):
    t = replace_once(
        t,
        "        seq_len = int(kwargs.pop(\"seq_len\"))\n"
        "        head_dim = int(kwargs[\"head_dim\"])\n"
        "        kwargs.setdefault(\"IDENTITY_DRAM_ADDR\", self._identity_dram_addr)\n"
        "        return self.unified_attention_core(\n"
        "            batch=seq_len,\n",
        "        seq_len = int(kwargs.pop(\"seq_len\"))\n"
        "        batch = int(kwargs.pop(\"batch\", seq_len))   # opt: explicit query-row count\n"
        "        head_dim = int(kwargs[\"head_dim\"])\n"
        "        kwargs.setdefault(\"IDENTITY_DRAM_ADDR\", self._identity_dram_addr)\n"
        "        return self.unified_attention_core(\n"
        "            batch=batch,\n", "attn1/core")
    t = replace_once(
        t,
        "                    ue.flash_attention_core_cached(\n"
        "                        head_dim=full_head_dim, seq_len=T_aligned,\n"
        "                        Q_DRAM_ADDR=Q_FLASH_DRAM + h * head_stride_bytes,\n"
        "                        K_DRAM_ADDR=K_cache + h * head_stride_bytes,\n"
        "                        V_DRAM_ADDR=V_cache + h * head_stride_bytes,\n"
        "                        OUTPUT_DRAM_ADDR=FLASH_OUT_DRAM + h * head_stride_bytes,\n",
        "                    ue.flash_attention_core_cached(\n"
        "                        head_dim=full_head_dim, seq_len=T_aligned, batch=1,   # opt: 1 x T\n"
        "                        Q_DRAM_ADDR=Q_ROPE_DRAM + h * full_head_dim * BF16,\n"
        "                        K_DRAM_ADDR=K_cache + h * head_stride_bytes,\n"
        "                        V_DRAM_ADDR=V_cache + h * head_stride_bytes,\n"
        "                        OUTPUT_DRAM_ADDR=FLASH_OUT_DRAM + h * full_head_dim * BF16,\n", "attn1/emit")
    t = replace_once(
        t,
        "            attn_row_base = FLASH_OUT_DRAM + h * head_stride_bytes\n",
        "            attn_row_base = FLASH_OUT_DRAM + h * full_head_dim * BF16   # opt: one row per head\n", "attn1/row")
    t = replace_once(
        t,
        "            if ue._compile_mode:\n"
        "                ue.generate_instruction_add_imm(ue._ISA_POS_K_REG, ue_35bit_addr_shifter(attn_row_base), ue._ISA_TMP_REG)\n"
        "                ue.accelerator_memory_to_sram(attn_row_base, SA_ATTN, full_head_dim, general_reg_src=ue._ISA_TMP_REG)\n"
        "            else:\n"
        "                ue.accelerator_memory_to_sram(attn_row_base + pos_start * row_bytes, SA_ATTN, full_head_dim)\n",
        "            ue.accelerator_memory_to_sram(attn_row_base, SA_ATTN, full_head_dim)\n", "attn1/load")
    # Replace the per-step full [T, T] mask upload with a single mask row at the three T=1 sites.
    lines = t.split("\n")
    out, i, n = [], 0, 0
    while i < len(lines):
        m = re.match(r"^(\s+)bias_host = torch\.full\(\((T_al|T_aligned), (?:T_al|T_aligned)\), float\(\"-inf\"\),$", lines[i])
        if m and len(m.group(1)) != 4:      # indent 4 is the T>1 prefill path: keep it
            indent, tname = m.group(1), m.group(2)
            j = i
            while "* BF16)" not in lines[j]:
                j += 1
            obj = "self" if "self.dma_write" in "\n".join(lines[i:j + 1]) else "ue"
            out.append("%sbias_host = torch.full((1, %s), float(\"-inf\"), dtype=torch.bfloat16)   # opt: one mask row" % (indent, tname))
            out.append("%sbias_host[:, :end_pos] = 0.0" % indent)
            out.append("%s%s.dma_write(DMA_DEVICE_H2C, %s._bias_dram, _bf(bias_host), %s * BF16)" % (indent, obj, obj, tname))
            i = j + 1
            n += 1
            continue
        out.append(lines[i])
        i += 1
    if n != 3:
        raise SystemExit("attn1/bias: expected 3 sites, found %d" % n)
    return "\n".join(out)


def p_noeye(t):
    return replace_once(
        t,
        "        conv_dim = self.lin_conv_dim\n"
        "        eye_bytes = conv_dim * conv_dim * BF16\n"
        "        eye_addr = self.allocate_tensor_dram(eye_bytes)\n"
        "        self.dma_write(DMA_DEVICE_H2C, eye_addr,\n"
        "                       _bf(torch.eye(conv_dim, dtype=torch.float32)),\n"
        "                       eye_bytes)\n"
        "        _IDENTITY_DRAM_CACHE[conv_dim] = eye_addr\n",
        "        # opt: the conv_dim x conv_dim identity is no longer needed (SiLU is tiled)\n", "noeye")


def p_heads(t):
    t = replace_once(
        t,
        "    assert num_vh == num_kh, (\n"
        "        \"layer split assumes num_v_heads == num_k_heads (Qwen3.5-2B invariant)\")\n",
        "    assert T == 1 or num_vh == num_kh, \"T>1 path needs num_v_heads == num_k_heads\"\n", "heads/assert")
    t = replace_once(
        t,
        "    Q_L2_DRAM = run_rms_norm_dram(ue, M=T * num_vh, N=Dk,\n"
        "                                  X_DRAM=Q_SPLIT_DRAM,\n"
        "                                  GAMMA_DRAM=ue._gamma_q_l2, batched=True)\n"
        "    K_L2_DRAM = run_rms_norm_dram(ue, M=T * num_vh, N=Dk,\n"
        "                                  X_DRAM=K_SPLIT_DRAM,\n"
        "                                  GAMMA_DRAM=ue._gamma_k_l2, batched=True)\n",
        "    Q_L2_DRAM = run_rms_norm_dram(ue, M=T * num_kh, N=Dk,\n"
        "                                  X_DRAM=Q_SPLIT_DRAM,\n"
        "                                  GAMMA_DRAM=ue._gamma_q_l2, batched=True)\n"
        "    K_L2_DRAM = run_rms_norm_dram(ue, M=T * num_kh, N=Dk,\n"
        "                                  X_DRAM=K_SPLIT_DRAM,\n"
        "                                  GAMMA_DRAM=ue._gamma_k_l2, batched=True)\n"
        "    if num_vh != num_kh:\n"
        "        # opt: each key head serves num_vh/num_kh value heads (repeat_interleave on the card)\n"
        "        _rep = num_vh // num_kh\n"
        "        _q_exp = _alloc_tensor(ue, num_vh * Dk)\n"
        "        _k_exp = _alloc_tensor(ue, num_vh * Dk)\n"
        "        for _src, _dst in ((Q_L2_DRAM, _q_exp), (K_L2_DRAM, _k_exp)):\n"
        "            for _kh in range(num_kh):\n"
        "                ue.accelerator_memory_to_sram(_src + _kh * Dk * BF16, 0x00000, Dk)\n"
        "                for _r in range(_rep):\n"
        "                    ue.sram_to_accelerator_memory(0x00000, _dst + (_kh * _rep + _r) * Dk * BF16, Dk)\n"
        "        Q_L2_DRAM, K_L2_DRAM = _q_exp, _k_exp\n", "heads/expand")
    t = replace_once(
        t,
        "            chunk_elems = (num_heads * Dk * Dv) // 2          # 131072 elts\n"
        "            chunk_bytes = chunk_elems * BPE                    # 262144 B\n"
        "            for chunk in range(2):\n",
        "            total_elems = num_heads * Dk * Dv\n"
        "            n_chunks = max(2, -(-total_elems // 131072))       # opt: each chunk <= half a URAM bank\n"
        "            chunk_elems = total_elems // n_chunks\n"
        "            chunk_bytes = chunk_elems * BPE\n"
        "            for chunk in range(n_chunks):\n", "heads/phaseE")
    return t


def p_shard(t):
    # LM head is optional: a shard without it neither uploads nor emits it.
    t = replace_once(
        t,
        "        if \"lm_head_data\" in weights and \"lm_head_scale\" in weights:\n",
        "        self._shard_has_head = not weights.get(\"no_head\", False)\n"
        "        if not self._shard_has_head:\n"
        "            _lm_pair = None\n"
        "        elif \"lm_head_data\" in weights and \"lm_head_scale\" in weights:\n", "shard/head1")
    t = replace_once(
        t,
        "        self._lm_head_scale_dram, self._lm_head_data_dram = _upload_fp4_pair(self, _lm_pair)\n",
        "        if _lm_pair is None:\n"
        "            self._lm_head_scale_dram = self._lm_head_data_dram = 0\n"
        "        else:\n"
        "            self._lm_head_scale_dram, self._lm_head_data_dram = _upload_fp4_pair(self, _lm_pair)\n", "shard/head2")
    # Record where the residual stream leaves the last layer; skip the final norm without a head.
    t = replace_once(
        t,
        "    with _quiet():\n"
        "        FINAL_NORM_DRAM = run_rms_norm_dram(ue, M=T, N=H,\n",
        "    ue._decoder_last_hidden_dram = cur_dram\n"
        "    if not getattr(ue, \"_shard_has_head\", True):\n"
        "        ue._decoder_final_norm_dram = cur_dram\n"
        "        return None\n"
        "    with _quiet():\n"
        "        FINAL_NORM_DRAM = run_rms_norm_dram(ue, M=T, N=H,\n", "shard/forward")
    a = t.index("            self.quantized_matmat_core(\n                M=1, K=self.hidden_size, N=self.VOCAB,")
    end = "                write_back_disable=True)\n"
    b = t.index(end, a) + len(end)
    block = t[a:b]
    block = "            if getattr(self, \"_shard_has_head\", True):\n" + "".join(
        "    " + ln + "\n" for ln in block.rstrip("\n").split("\n"))
    return t[:a] + block + t[b:]


def p_recgrp(t):
    """State update: the stock code computes k_all @ S and q_all @ S as ONE wide matmul over all
    heads ([H, H*Dv] output) and uses only the diagonal blocks, so its cost grows with H^2.
    Split it into groups of Q35_REC_GROUP heads (default: all heads = stock behaviour). The
    diagonal blocks keep a uniform stride, so everything downstream addresses them unchanged."""
    t = replace_once(
        t,
        "        WIDE_HEAD_STRIDE = (num_heads + 1) * Dv * BPE   # diag offset of m_wide / y_wide\n",
        "        REC_G = int(os.environ.get(\"Q35_REC_GROUP\", \"0\")) or num_heads   # opt: heads per wide matmul\n"
        "        assert num_heads % REC_G == 0, \"Q35_REC_GROUP must divide the head count\"\n"
        "        WIDE_HEAD_STRIDE = (REC_G + 1) * Dv * BPE       # diag offset of m_wide / y_wide\n"
        "        WIDE_GROUP_BYTES = REC_G * (REC_G + 1) * Dv * BPE\n", "recgrp/stride")
    for base, buf, what in (("K_BASE", "self._m_wide_dram", "recgrp/B"), ("Q_BASE", "self._y_wide_dram", "recgrp/D")):
        t = replace_once(
            t,
            "            self.matmat_mul_core(M=num_heads, K=Dk, N=num_heads * Dv,\n"
            "                                 A_DRAM_ADDR=%s,\n"
            "                                 B_DRAM_ADDR=S_DRAM_ADDR,\n"
            "                                 OUTPUT_DRAM_ADDR=%s,\n"
            "                                 is_B_quantized=False)\n" % (base, buf),
            "            for _g in range(num_heads // REC_G):\n"
            "                self.matmat_mul_core(M=REC_G, K=Dk, N=REC_G * Dv,\n"
            "                                     A_DRAM_ADDR=%s + _g * REC_G * Dk * BPE,\n"
            "                                     B_DRAM_ADDR=S_DRAM_ADDR + _g * REC_G * S_HEAD_BYTES,\n"
            "                                     OUTPUT_DRAM_ADDR=%s + _g * WIDE_GROUP_BYTES,\n"
            "                                     is_B_quantized=False)\n" % (base, buf), what)
    return t


def p_tp(t):
    """Tensor-parallel mode (ue._tp_mode): the card holds a slice of every layer, so the two
    residual additions per layer cannot happen on the card. At each of them the program halts;
    the host sums the cards' partial outputs with the residual and writes the result back into
    the input buffer, then starts the next segment. Inert unless ue._tp_mode is set. Needs shard."""
    helper = (
        "def _tp_cut(ue, base_dram: int, part_dram: int, T: int, H: int) -> int:\n"
        "    if not getattr(ue, \"_tp_mode\", False):\n"
        "        return run_eltwise_op_dram(ue, base_dram, part_dram,\n"
        "                                   numel=T * H, op=\"add\", batched=True)\n"
        "    ue.generate_instruction_halt()          # also pads to a 64-byte boundary\n"
        "    ue._tp_segments.append((ue.capture_count, part_dram))\n"
        "    return ue._decoder_X_dram\n"
        "\n\n")
    t = replace_once(t, "def _run_linear_attn_layer(ue, X_DRAM: int, layer_idx: int, T: int,",
                     helper + "def _run_linear_attn_layer(ue, X_DRAM: int, layer_idx: int, T: int,", "tp/helper")
    for pat, new, what in (
            (r"RESIDUAL_DRAM = run_eltwise_op_dram\(ue, X_DRAM, OUT_PROJ_DRAM,\s+numel=T \* H, op=\"add\", batched=True\)",
             "RESIDUAL_DRAM = _tp_cut(ue, X_DRAM, OUT_PROJ_DRAM, T, H)", "tp/residual"),
            (r"FINAL_DRAM = run_eltwise_op_dram\(ue, RESIDUAL_DRAM, DOWN_DRAM,\s+numel=T \* H, op=\"add\", batched=True\)",
             "FINAL_DRAM = _tp_cut(ue, RESIDUAL_DRAM, DOWN_DRAM, T, H)", "tp/final")):
        t, n = re.subn(pat, new, t)
        if n != 3:
            raise SystemExit("%s: expected 3 sites, found %d" % (what, n))
    # The LM head slice must write its logits back: the host compares the cards' winners by value.
    a = t.index("M=1, K=self.hidden_size, N=self.VOCAB,")
    old = "write_back_disable=True)"
    b = t.index(old, a)
    return t[:b] + "write_back_disable=not getattr(self, \"_tp_mode\", False))" + t[b + len(old):]


def p_abfuse(t):
    """Decode: the per-head gate projections a and b are two [64, H] bf16 matmuls of which only the
    first num_heads rows are real. With <= 32 heads they fit one [64, H] matmul (b in rows 0..31,
    a in rows 32..63), halving the bytes read for them."""
    t = replace_once(
        t,
        "        W_A_PAD=_upload(ue, ex[\"W_A_PAD\"]),\n",
        "        W_A_PAD=_upload(ue, ex[\"W_A_PAD\"]),\n"
        "        W_AB_PAD=_upload(ue, torch.cat([ex[\"W_B_PAD\"][:UE_VECTOR_SIZE // 2],\n"
        "                                        ex[\"W_A_PAD\"][:UE_VECTOR_SIZE // 2]], 0)),   # opt: fused a/b\n"
        "        A_PAD_BUF=_upload(ue, torch.zeros(UE_VECTOR_SIZE)),\n", "abfuse/upload")
    old = (
        "    BETA_PAD_DRAM = run_bf16_matmul(ue, M=T, K=H, N=UE_VECTOR_SIZE,\n"
        "                                    A_DRAM=PRENORM_DRAM, B_DRAM=w[\"W_B_PAD\"],\n"
        "                                    sigmoid=(T != 1), batched=True)\n"
        "    A_PAD_DRAM = run_bf16_matmul(ue, M=T, K=H, N=UE_VECTOR_SIZE,\n"
        "                                 A_DRAM=PRENORM_DRAM, B_DRAM=w[\"W_A_PAD\"],\n"
        "                                 batched=True)\n")
    new = (
        "    if T == 1 and num_vh <= UE_VECTOR_SIZE // 2 and os.environ.get(\"Q35_AB_FUSE\", \"1\") != \"0\":\n"
        "        # opt: one projection for both gates; a is copied out to its own zero-padded row\n"
        "        BETA_PAD_DRAM = run_bf16_matmul(ue, M=T, K=H, N=UE_VECTOR_SIZE,\n"
        "                                        A_DRAM=PRENORM_DRAM, B_DRAM=w[\"W_AB_PAD\"],\n"
        "                                        batched=True)\n"
        "        A_PAD_DRAM = w[\"A_PAD_BUF\"]\n"
        "        ue.accelerator_memory_to_sram(BETA_PAD_DRAM + (UE_VECTOR_SIZE // 2) * BF16, 0x00000, UE_VECTOR_SIZE // 2)\n"
        "        ue.sram_to_accelerator_memory(0x00000, A_PAD_DRAM, UE_VECTOR_SIZE // 2)\n"
        "    else:\n" + "".join("    " + ln + "\n" for ln in old.rstrip("\n").split("\n")))
    return replace_once(t, old, new, "abfuse/layer")


def p_vtshare(t):
    """Decode attention: the engine's attention kernel transposes V on every call, i.e. once per
    QUERY head, although query heads of one group share the same V. Transpose once per KV head
    and feed the kernel's three remaining steps the ready V^T. Needs attn1."""
    t = replace_once(
        t,
        "        kwargs.setdefault(\"IDENTITY_DRAM_ADDR\", self._identity_dram_addr)\n"
        "        return self.unified_attention_core(\n",
        "        kwargs.setdefault(\"IDENTITY_DRAM_ADDR\", self._identity_dram_addr)\n"
        "        v_t = kwargs.pop(\"V_T_DRAM_ADDR\", None)\n"
        "        if v_t is not None:\n"
        "            # opt: same steps as unified_attention_core_legacy minus its V transpose\n"
        "            score = kwargs[\"SCRATCH_DRAM_ADDR\"] + head_dim * seq_len * BF16\n"
        "            scaled_q = score + seq_len * seq_len * BF16\n"
        "            self.eltwise_core_dram(M=batch, N=head_dim, dram_a=kwargs[\"Q_DRAM_ADDR\"], dram_b=None,\n"
        "                                   dram_out=scaled_q, mode=UE_MODE.MUL_BROADCAST,\n"
        "                                   scalar=1.0 / (head_dim ** 0.5))\n"
        "            self.matmat_mul_core(M=batch, K=head_dim, N=seq_len, A_DRAM_ADDR=scaled_q,\n"
        "                                 B_DRAM_ADDR=kwargs[\"K_DRAM_ADDR\"], OUTPUT_DRAM_ADDR=score,\n"
        "                                 softmax_enable=True, C_DRAM_ADDR=kwargs[\"BIAS_DRAM_ADDR\"],\n"
        "                                 bias_mode=\"full_matrix\")\n"
        "            return self.matmat_mul_core(M=batch, K=seq_len, N=head_dim, A_DRAM_ADDR=score,\n"
        "                                        B_DRAM_ADDR=v_t, OUTPUT_DRAM_ADDR=kwargs[\"OUTPUT_DRAM_ADDR\"])\n"
        "        return self.unified_attention_core(\n", "vtshare/core")
    t = replace_once(
        t,
        "            key_fa = (\"flash_T1\", layer_idx, Q_FLASH_DRAM, K_cache, V_cache,\n",
        "            VT_DRAM = _alloc_tensor(ue, num_kv_heads * T_aligned * full_head_dim)   # opt: V^T per KV head\n"
        "            key_fa = (\"flash_T1\", layer_idx, Q_FLASH_DRAM, K_cache, V_cache,\n", "vtshare/alloc")
    return replace_once(
        t,
        "                for h in range(num_q_heads):\n"
        "                    ue.flash_attention_core_cached(\n"
        "                        head_dim=full_head_dim, seq_len=T_aligned, batch=1,   # opt: 1 x T\n",
        "                for _kh in range(num_kv_heads):\n"
        "                    ue.bf16_transpose_core(M=T_aligned, N=full_head_dim,\n"
        "                                           INPUT_DRAM_ADDR=V_cache + _kh * group_sz * head_stride_bytes,\n"
        "                                           OUTPUT_DRAM_ADDR=VT_DRAM + _kh * head_stride_bytes,\n"
        "                                           IDENTITY_DRAM_ADDR=ue._identity_dram_addr)\n"
        "                for h in range(num_q_heads):\n"
        "                    ue.flash_attention_core_cached(\n"
        "                        V_T_DRAM_ADDR=VT_DRAM + (h // group_sz) * head_stride_bytes,\n"
        "                        head_dim=full_head_dim, seq_len=T_aligned, batch=1,   # opt: 1 x T\n", "vtshare/emit")


PATCHES = {"silu": p_silu, "kpad": p_kpad, "attn1": p_attn1, "noeye": p_noeye, "heads": p_heads, "shard": p_shard,
           "recgrp": p_recgrp, "tp": p_tp, "abfuse": p_abfuse, "vtshare": p_vtshare}

CFG_9B = {
    "num_layers": 32, "hidden_size": 4096, "embedding_vocab": 248320, "mlp_elements": 12288,
    "full_attn_head_dim": 256, "full_attn_num_heads": 16, "full_attn_num_kv_heads": 4,
    "full_attn_q_size": 4096, "full_attn_kv_size": 1024, "full_attn_rotary_dim": 64, "full_attention_interval": 4,
    "linear_attn_num_v_heads": 32, "linear_attn_num_k_heads": 16,
    "linear_attn_head_k_dim": 128, "linear_attn_head_v_dim": 128,
    "linear_attn_key_dim": 2048, "linear_attn_value_dim": 4096, "linear_attn_conv_dim": 8192,
    "linear_attn_conv_kernel_size": 4, "linear_attn_num_ab_heads": 32,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--link2b", action="store_true", help="share the 2B weights and model directory")
    ap.add_argument("--cfg9b", action="store_true", help="write a Qwen3.5-9B configuration")
    ap.add_argument("--hfcfg", default="", help="derive the configuration from a Hugging Face config.json (any Qwen3.5-family size)")
    ap.add_argument("patches", nargs="*")
    a = ap.parse_args()
    M = os.path.join(E, "models", a.name)
    os.makedirs(os.path.join(M, "qwen3.5_2b_bin"), exist_ok=True)
    for f in ("user_dma_core.py", "quant_lib.py", "andromeda_hw_info.py", "nn_lib.py", "test_samples", "read_trace.py"):
        dst = os.path.join(E, f)
        if not os.path.lexists(dst) and os.path.exists(os.path.join(R, f)):
            os.symlink(os.path.join(R, f), dst)
    cfg = json.load(open(os.path.join(SRC_DIR, "qwen3.5_2b_config.json")))
    if a.cfg9b:
        cfg["file_info"].update(CFG_9B)
        n = CFG_9B["num_layers"]
        cfg["model"]["full_attn_layer_indices"] = [i for i in range(n) if i % 4 == 3]
        cfg["model"]["linear_attn_layer_indices"] = [i for i in range(n) if i % 4 != 3]
        cfg["special"]["embedding"]["vocab_size"] = CFG_9B["embedding_vocab"]
        cfg["_template_note"] = "Qwen3.5-9B configuration derived from the stock 2B port."
    if a.hfcfg:
        tc = json.load(open(a.hfcfg))["text_config"]
        n, hd = tc["num_hidden_layers"], tc["head_dim"]
        nk, nv = tc["linear_num_key_heads"], tc["linear_num_value_heads"]
        dk, dv = tc["linear_key_head_dim"], tc["linear_value_head_dim"]
        cfg["file_info"].update({
            "num_layers": n, "hidden_size": tc["hidden_size"], "embedding_vocab": tc["vocab_size"],
            "mlp_elements": tc["intermediate_size"], "full_attn_head_dim": hd,
            "full_attn_num_heads": tc["num_attention_heads"], "full_attn_num_kv_heads": tc["num_key_value_heads"],
            "full_attn_q_size": tc["num_attention_heads"] * hd, "full_attn_kv_size": tc["num_key_value_heads"] * hd,
            "full_attn_rotary_dim": int(hd * tc["partial_rotary_factor"]),
            "full_attention_interval": tc["full_attention_interval"],
            "linear_attn_num_v_heads": nv, "linear_attn_num_k_heads": nk, "linear_attn_head_k_dim": dk,
            "linear_attn_head_v_dim": dv, "linear_attn_key_dim": nk * dk, "linear_attn_value_dim": nv * dv,
            "linear_attn_conv_dim": 2 * nk * dk + nv * dv, "linear_attn_conv_kernel_size": tc["linear_conv_kernel_dim"],
            "linear_attn_num_ab_heads": nv})
        cfg["model"]["full_attn_layer_indices"] = [i for i, t in enumerate(tc["layer_types"]) if t == "full_attention"]
        cfg["model"]["linear_attn_layer_indices"] = [i for i, t in enumerate(tc["layer_types"]) if t != "full_attention"]
        cfg["special"]["embedding"]["vocab_size"] = tc["vocab_size"]
        print("rope theta: template %s, model %s | rms eps model %s | tied output layer: %s" % (
            cfg["special"]["rope"]["theta_full"], tc["rope_parameters"]["rope_theta"], tc["rms_norm_eps"],
            tc["tie_word_embeddings"]))
        cfg["special"]["rope"]["theta_full"] = tc["rope_parameters"]["rope_theta"]
        cfg["_template_note"] = "Configuration derived from %s on top of the stock 2B port." % a.hfcfg
    json.dump(cfg, open(os.path.join(M, "qwen3.5_2b_config.json"), "w"), indent=2)
    if a.link2b:
        for f in ("params.bin", "params.json", "Qwen3.5-2B"):
            dst = os.path.join(M, "qwen3.5_2b_bin", f)
            if not os.path.lexists(dst):
                os.symlink(os.path.join(SRC_DIR, "qwen3.5_2b_bin", f), dst)
    text = open(os.path.join(SRC_DIR, "qwen3.5_2b_test.py")).read()
    for name in a.patches:
        text = PATCHES[name](text)
        print("applied:", name)
    out = os.path.join(M, "qwen3.5_2b_test.py")
    open(out, "w").write(text)
    compile(text, out, "exec")
    print("written:", out)


if __name__ == "__main__":
    main()
