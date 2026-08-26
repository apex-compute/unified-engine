"""
Kokoro FPGA forward pass, built section-by-section on the UnifiedEngine
accelerator. This module is meant to fully replace KokoroModel.forward's
CUDA/CPU path once every section below is ported -- not a permanent parallel
path. Sections not yet ported here fall back to calling the corresponding
CPU submodule from the already-loaded `model` (a KokoroModel instance from
kokoro_test.py), so the pipeline stays runnable end-to-end while individual
sections come online.

Section status:
  [x] Section 1: PL-BERT encoder (embedding + 12x shared Albert transformer
      layer + bert_encoder projection) -> d_en
  [x] Section 2: Prosody/duration path (DurationEncoder + predictor.lstm +
      duration_proj -> pred_dur). NOTE: still missing the pred_aln_trg
      (repeat_interleave + scatter) alignment-matrix construction that
      consumes pred_dur -- that lands with Section 3, since it's the bridge
      into F0Ntrain's `en = d.transpose(-1,-2) @ pred_aln_trg`.
  [x] Section 3: F0/N prediction (pred_aln_trg construction + predictor.shared
      LSTM + AdainResBlk1d stacks) -> F0_pred, N_pred. First section to
      introduce Conv1d/AdaIN1d/depthwise-ConvTranspose1d/nearest-upsample
      into the actual model (each individually validated earlier in the
      op-gap hardware proofs, composed here for the first time).
  [ ] Section 4: TextEncoder (phoneme embed + Conv1d stack + LSTM)
  [ ] Section 5: ISTFTNet Decoder (AdainResBlk1d stack + Generator/vocoder)

Weights are pulled directly from the loaded CPU `model`'s state dict at call
time (no bin-dump step yet -- that's a later optimization once the full
pipeline is ported, following the pattern in models/parakeet).

ALBERT weight sharing: the checkpoint only has ONE transformer layer's
weights (`encoder.albert_layer_groups.0.albert_layers.0.*`, num_hidden_groups=1),
applied 12 times -- confirmed by inspecting the state dict keys. So Section 1
only DMAs a single layer's weights and loops the same hardware program 12x.

nn.Linear weight layout ([out_features, in_features]) already matches
matmat_mul_core's B operand layout ([N, K], see its docstring at
user_dma_core.py:5234-5254 -- "A @ B^T"), so every Linear weight below is
uploaded as-is, no transpose needed.
"""
import torch

from user_dma_core import UnifiedEngine, UE_MODE, UE_VECTOR_SIZE, set_dma_device


def _round_up(n: int, mult: int) -> int:
    return ((n + mult - 1) // mult) * mult


def _bf16(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.bfloat16).contiguous()


class PLBertFPGA:
    """Section 1: PL-BERT (Albert) encoder + bert_encoder projection, run on the
    UnifiedEngine accelerator.
    """

    NEG_INF = -1.0e9  # masking value for padded key positions in the attention bias

    def __init__(self, model, ue: UnifiedEngine):
        self.model = model
        self.ue = ue
        albert = model.bert.albert
        cfg = albert.config
        sd = albert.state_dict()

        self.E = cfg.embedding_size          # 128
        self.H = cfg.hidden_size             # 768
        self.NH = cfg.num_attention_heads    # 12
        self.HD = self.H // self.NH          # 64
        self.NL = cfg.num_hidden_layers      # 12
        self.FFN = cfg.intermediate_size     # 2048

        g = lambda k: _bf16(sd[k])
        self.word_emb = g("embeddings.word_embeddings.weight")           # [178, 128]
        self.pos_emb = g("embeddings.position_embeddings.weight")        # [512, 128]
        self.type_emb0 = g("embeddings.token_type_embeddings.weight")[0]  # [128] (token_type_id always 0)
        self.emb_ln_w = g("embeddings.LayerNorm.weight")
        self.emb_ln_b = g("embeddings.LayerNorm.bias")
        self.map_in_w = g("encoder.embedding_hidden_mapping_in.weight")  # [768, 128]
        self.map_in_b = g("encoder.embedding_hidden_mapping_in.bias")

        p = "encoder.albert_layer_groups.0.albert_layers.0."
        self.q_w, self.q_b = g(p + "attention.query.weight"), g(p + "attention.query.bias")
        self.k_w, self.k_b = g(p + "attention.key.weight"), g(p + "attention.key.bias")
        self.v_w, self.v_b = g(p + "attention.value.weight"), g(p + "attention.value.bias")
        self.attn_out_w, self.attn_out_b = g(p + "attention.dense.weight"), g(p + "attention.dense.bias")
        self.attn_ln_w, self.attn_ln_b = g(p + "attention.LayerNorm.weight"), g(p + "attention.LayerNorm.bias")
        self.ffn_w, self.ffn_b = g(p + "ffn.weight"), g(p + "ffn.bias")
        self.ffn_out_w, self.ffn_out_b = g(p + "ffn_output.weight"), g(p + "ffn_output.bias")
        self.out_ln_w, self.out_ln_b = g(p + "full_layer_layer_norm.weight"), g(p + "full_layer_layer_norm.bias")

        self.bert_encoder_w = _bf16(model.bert_encoder.weight)  # [512, 768]
        self.bert_encoder_b = _bf16(model.bert_encoder.bias)

    def _up(self, tensor: torch.Tensor) -> int:
        a = self.ue.allocate_tensor_dram(tensor.numel() * 2)
        self.ue.dma_to_accelerator_memory(a, tensor.reshape(-1))
        return a

    def _linear(self, x_dram, M, K, N, w_dram, b_dram, out_dram, gelu=False):
        return self.ue.matmat_mul_core(
            M=M, K=K, N=N, A_DRAM_ADDR=x_dram, B_DRAM_ADDR=w_dram, OUTPUT_DRAM_ADDR=out_dram,
            C_DRAM_ADDR=b_dram, bias_mode="broadcast_N", gelu_enable=gelu,
        )

    _GELU_NEW_C0 = 0.7978845608028654  # sqrt(2/pi)
    _GELU_NEW_C1 = 0.044715

    def _gelu_new(self, x_dram, numel, identity_dram, tmp1, tmp2, tmp3, out_dram):
        """Exact HF `gelu_new` (tanh-approximation GELU), matching Albert's actual configured
        hidden_act -- NOT matmat_mul_core's fused gelu_enable epilogue, which computes a different
        (sigmoid-based) approximation and was the source of a ~15dB SNR loss compounding across
        Kokoro's 12 shared transformer layers.

        gelu_new(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))

        tanh is derived EXACTLY from the native sigmoid activation via tanh(x) = 2*sigmoid(2x) - 1
        (the same identity validated at 40+dB in the LSTM hardware proof) -- built entirely from
        eltwise_core_dram (mul/add/broadcast) + activation_core(sigmoid), no new hardware op.
        numel must be a multiple of UE_VECTOR_SIZE; identity_dram must be a UE_VECTOR_SIZE-square
        bf16 identity matrix.
        """
        ue = self.ue
        M, N = numel // UE_VECTOR_SIZE, UE_VECTOR_SIZE
        mul = lambda a, b, o: ue.eltwise_core_dram(M, N, a, b, o, mode=UE_MODE.ELTWISE_MUL)
        add = lambda a, b, o: ue.eltwise_core_dram(M, N, a, b, o, mode=UE_MODE.ELTWISE_ADD)
        mulb = lambda a, s, o: ue.eltwise_core_dram(M, N, a, None, o, mode=UE_MODE.MUL_BROADCAST, scalar=s)
        addb = lambda a, s, o: ue.eltwise_core_dram(M, N, a, None, o, mode=UE_MODE.ADD_BROADCAST, scalar=s)

        mul(x_dram, x_dram, tmp1)                       # tmp1 = x^2
        mul(tmp1, x_dram, tmp1)                          # tmp1 = x^3
        mulb(tmp1, self._GELU_NEW_C1, tmp1)               # tmp1 = 0.044715*x^3
        add(x_dram, tmp1, tmp1)                           # tmp1 = x + 0.044715*x^3
        mulb(tmp1, self._GELU_NEW_C0, tmp1)               # tmp1 = u = sqrt(2/pi)*(...)
        mulb(tmp1, 2.0, tmp2)                             # tmp2 = 2u
        ue.activation_core(M=M, N=N, A_DRAM_ADDR=tmp2, OUTPUT_DRAM_ADDR=tmp2,
                            IDENTITY_DRAM_ADDR=identity_dram, activation="sigmoid")
        mulb(tmp2, 2.0, tmp2)                             # tmp2 = 2*sigmoid(2u)
        addb(tmp2, -1.0, tmp2)                            # tmp2 = tanh(u) = 2*sigmoid(2u) - 1
        mul(x_dram, tmp2, tmp3)                           # tmp3 = x*tanh(u)
        add(x_dram, tmp3, tmp3)                           # tmp3 = x + x*tanh(u) = x*(1+tanh(u))
        mulb(tmp3, 0.5, out_dram)                         # out = 0.5*x*(1+tanh(u))

    def _run(self, ue):
        ue.stop_capture()
        ue.generate_instruction_halt()
        prog = ue.get_program_dram_addr()
        ue.write_captured_instructions_to_dram(prog)
        ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())
        ue.start_execute_from_dram(prog)
        ue.wait_queue(30.0)
        ue.clear_capture_buffer(); ue.reset_program_dram_addr()

    def forward(self, input_ids: torch.LongTensor, debug_cpu_ref=None) -> torch.Tensor:
        """input_ids: LongTensor [T]. Returns d_en: FloatTensor [512, T], matching
        KokoroModel.forward_with_tokens's ``d_en = bert_encoder(bert(...)).transpose(-1,-2)``.
        """
        ue = self.ue
        T = input_ids.shape[0]
        T_pad = _round_up(T, UE_VECTOR_SIZE)
        E, H, NH, HD, FFN = self.E, self.H, self.NH, self.HD, self.FFN

        # ---- upload weights (once per call; bin-dump caching is a later optimization) ----
        w_word = self._up(self.word_emb)
        w_map_in, b_map_in = self._up(self.map_in_w), self._up(self.map_in_b)
        w_emb_ln, b_emb_ln = self._up(self.emb_ln_w), self._up(self.emb_ln_b)
        w_q, b_q = self._up(self.q_w), self._up(self.q_b)
        w_k, b_k = self._up(self.k_w), self._up(self.k_b)
        w_v, b_v = self._up(self.v_w), self._up(self.v_b)
        w_attn_out, b_attn_out = self._up(self.attn_out_w), self._up(self.attn_out_b)
        w_attn_ln, b_attn_ln = self._up(self.attn_ln_w), self._up(self.attn_ln_b)
        w_ffn, b_ffn = self._up(self.ffn_w), self._up(self.ffn_b)
        w_ffn_out, b_ffn_out = self._up(self.ffn_out_w), self._up(self.ffn_out_b)
        w_out_ln, b_out_ln = self._up(self.out_ln_w), self._up(self.out_ln_b)
        w_bert_enc, b_bert_enc = self._up(self.bert_encoder_w), self._up(self.bert_encoder_b)
        identity_dram = self._up(torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))

        # Attention padding mask: same [T_pad] key-mask row broadcast across all T_pad query rows
        # (host-precomputed constant, not a runtime hardware op) -- 0 for real keys (j<T), -inf for
        # padded keys (j>=T), so padded positions contribute nothing to the softmax.
        key_mask_row = torch.zeros(T_pad, dtype=torch.float32)
        key_mask_row[T:] = self.NEG_INF
        attn_bias = key_mask_row.unsqueeze(0).expand(T_pad, T_pad).contiguous().to(torch.bfloat16)
        bias_dram = self._up(attn_bias)

        # ---- embeddings: word (indexed gather, ids known at capture time), position (contiguous
        #      rows 0..T_pad-1 of the table), token_type (row 0 broadcast to every position) ----
        word_dram = ue.allocate_tensor_dram(T_pad * E * 2)
        row_bytes_e = E * 2
        ue.start_capture()
        for i in range(T):
            tok_id = int(input_ids[i].item())
            ue.accelerator_memory_to_sram(accelerator_dram_address=w_word + tok_id * row_bytes_e,
                                           sram_address=0x00000, element_size=E)
            ue.sram_to_accelerator_memory(sram_address=0x00000,
                                           accelerator_dram_address=word_dram + i * row_bytes_e,
                                           element_size=E)
        for i in range(T, T_pad):
            # Padding rows: content doesn't matter (attention bias masks these positions out, and
            # only the first T output rows are ever read back), reuse row 0 for a cheap fill.
            ue.accelerator_memory_to_sram(accelerator_dram_address=w_word,
                                           sram_address=0x00000, element_size=E)
            ue.sram_to_accelerator_memory(sram_address=0x00000,
                                           accelerator_dram_address=word_dram + i * row_bytes_e,
                                           element_size=E)
        ue.stop_capture()
        ue.generate_instruction_halt()
        prog = ue.get_program_dram_addr()
        ue.write_captured_instructions_to_dram(prog)
        ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())
        ue.start_execute_from_dram(prog)
        ue.wait_queue(10.0)
        ue.clear_capture_buffer(); ue.reset_program_dram_addr()

        pos_dram = self._up(self.pos_emb[:T_pad])
        type_dram = self._up(self.type_emb0.unsqueeze(0).expand(T_pad, E).contiguous())

        emb_sum_dram = ue.allocate_tensor_dram(T_pad * E * 2)
        emb_ln_dram = ue.allocate_tensor_dram(T_pad * E * 2)
        hidden_dram = ue.allocate_tensor_dram(T_pad * H * 2)

        q_dram = ue.allocate_tensor_dram(T_pad * H * 2)
        k_dram = ue.allocate_tensor_dram(T_pad * H * 2)
        v_dram = ue.allocate_tensor_dram(T_pad * H * 2)
        q_heads_dram = ue.allocate_tensor_dram(T_pad * H * 2)
        k_heads_dram = ue.allocate_tensor_dram(T_pad * H * 2)
        v_heads_dram = ue.allocate_tensor_dram(T_pad * H * 2)
        attn_heads_out_dram = ue.allocate_tensor_dram(T_pad * H * 2)
        attn_merged_dram = ue.allocate_tensor_dram(T_pad * H * 2)
        attn_proj_dram = ue.allocate_tensor_dram(T_pad * H * 2)
        resid1_dram = ue.allocate_tensor_dram(T_pad * H * 2)
        attn_ln_dram = ue.allocate_tensor_dram(T_pad * H * 2)
        ffn_mid_dram = ue.allocate_tensor_dram(T_pad * FFN * 2)
        gelu_tmp1_dram = ue.allocate_tensor_dram(T_pad * FFN * 2)
        gelu_tmp2_dram = ue.allocate_tensor_dram(T_pad * FFN * 2)
        gelu_tmp3_dram = ue.allocate_tensor_dram(T_pad * FFN * 2)
        ffn_out_dram = ue.allocate_tensor_dram(T_pad * H * 2)
        resid2_dram = ue.allocate_tensor_dram(T_pad * H * 2)
        scratch_dram = ue.allocate_tensor_dram((HD + T_pad) * T_pad * 2 + T_pad * HD * 2)
        d_en_dram = ue.allocate_tensor_dram(T_pad * 512 * 2)

        from user_dma_core import calculate_snr

        # Buffer bisect results instead of printing inline -- each stage's capture/execute emits a
        # wall of M_chunk/URAM-usage compile logs that would otherwise bury the SNR lines between
        # every layer. Printed as one clean table at the end of forward().
        debug_log = []

        def _check(stage_name, dram_addr, shape, cpu_ref):
            if debug_cpu_ref is None:
                return
            got = ue.dma_from_accelerator_memory(dram_addr, shape)[:T].float()
            snr_db = calculate_snr(cpu_ref.detach().float().reshape(-1), got.reshape(-1))
            debug_log.append((stage_name, snr_db))

        # --- Embedding stage (own capture/execute so we can read back + compare before continuing) ---
        ue.start_capture()
        ue.eltwise_core_dram(T_pad, E, word_dram, pos_dram, emb_sum_dram, mode=UE_MODE.ELTWISE_ADD)
        ue.eltwise_core_dram(T_pad, E, emb_sum_dram, type_dram, emb_sum_dram, mode=UE_MODE.ELTWISE_ADD)
        ue.layer_norm_core_dram(M=T_pad, N=E, A_DRAM_ADDR=emb_sum_dram, OUTPUT_DRAM_ADDR=emb_ln_dram,
                                 GAMMA_DRAM_ADDR=w_emb_ln, BETA_DRAM_ADDR=b_emb_ln)
        self._linear(emb_ln_dram, T_pad, E, H, w_map_in, b_map_in, hidden_dram)
        self._run(ue)

        if debug_cpu_ref is not None:
            emb_cpu = debug_cpu_ref["embeddings"](input_ids=input_ids.unsqueeze(0))
            hidden_cpu = debug_cpu_ref["map_in"](emb_cpu)  # [1, T, H] -- keep the batch dim; AlbertLayer requires 3D input
            _check("embedding+map_in", hidden_dram, (T_pad, H), hidden_cpu)

        for layer_idx in range(self.NL):
            ue.start_capture()
            self._linear(hidden_dram, T_pad, H, H, w_q, b_q, q_dram)
            self._linear(hidden_dram, T_pad, H, H, w_k, b_k, k_dram)
            self._linear(hidden_dram, T_pad, H, H, w_v, b_v, v_dram)

            # Interleaved [T_pad, NH, HD] -> grouped [NH, T_pad, HD] (per-head contiguous), via
            # native strided DMA -- no per-row Python loop needed.
            ue.bf16_permute_dram_core(num_groups=NH, group_rows=T_pad, row_width=HD,
                                       in_dram=q_dram, out_dram=q_heads_dram, write_grouped=True)
            ue.bf16_permute_dram_core(num_groups=NH, group_rows=T_pad, row_width=HD,
                                       in_dram=k_dram, out_dram=k_heads_dram, write_grouped=True)
            ue.bf16_permute_dram_core(num_groups=NH, group_rows=T_pad, row_width=HD,
                                       in_dram=v_dram, out_dram=v_heads_dram, write_grouped=True)

            head_stride = T_pad * HD * 2
            for h in range(NH):
                off = h * head_stride
                ue.unified_attention_core(
                    batch=T_pad, aligned_seq_len=T_pad, head_dim=HD,
                    Q_DRAM_ADDR=q_heads_dram + off, K_DRAM_ADDR=k_heads_dram + off, V_DRAM_ADDR=v_heads_dram + off,
                    BIAS_DRAM_ADDR=bias_dram, OUTPUT_DRAM_ADDR=attn_heads_out_dram + off,
                    SCRATCH_DRAM_ADDR=scratch_dram, IDENTITY_DRAM_ADDR=identity_dram,
                )

            # Grouped [NH, T_pad, HD] (attention's per-head output) -> interleaved [T_pad, NH, HD]
            # == [T_pad, H] (write_grouped=False: in=grouped -> out=interleaved).
            ue.bf16_permute_dram_core(num_groups=NH, group_rows=T_pad, row_width=HD,
                                       in_dram=attn_heads_out_dram, out_dram=attn_merged_dram,
                                       write_grouped=False)

            self._linear(attn_merged_dram, T_pad, H, H, w_attn_out, b_attn_out, attn_proj_dram)
            ue.eltwise_core_dram(T_pad, H, hidden_dram, attn_proj_dram, resid1_dram, mode=UE_MODE.ELTWISE_ADD)
            ue.layer_norm_core_dram(M=T_pad, N=H, A_DRAM_ADDR=resid1_dram, OUTPUT_DRAM_ADDR=attn_ln_dram,
                                     GAMMA_DRAM_ADDR=w_attn_ln, BETA_DRAM_ADDR=b_attn_ln)

            self._linear(attn_ln_dram, T_pad, H, FFN, w_ffn, b_ffn, ffn_mid_dram, gelu=False)
            self._gelu_new(ffn_mid_dram, T_pad * FFN, identity_dram,
                            gelu_tmp1_dram, gelu_tmp2_dram, gelu_tmp3_dram, ffn_mid_dram)
            self._linear(ffn_mid_dram, T_pad, FFN, H, w_ffn_out, b_ffn_out, ffn_out_dram)
            ue.eltwise_core_dram(T_pad, H, attn_ln_dram, ffn_out_dram, resid2_dram, mode=UE_MODE.ELTWISE_ADD)
            ue.layer_norm_core_dram(M=T_pad, N=H, A_DRAM_ADDR=resid2_dram, OUTPUT_DRAM_ADDR=hidden_dram,
                                     GAMMA_DRAM_ADDR=w_out_ln, BETA_DRAM_ADDR=b_out_ln)
            self._run(ue)

            if debug_cpu_ref is not None:
                # Bisect within the layer: Q/K/V projections -> attention block (attn + dense +
                # residual + LN) -> full layer (+ FFN + residual + LN). Each stage is checked
                # against the SAME hidden_cpu input this hardware iteration used, so whichever
                # stage's SNR craters first is exactly where the bug lives.
                attn_mod = debug_cpu_ref["layer"].attention
                q_cpu = attn_mod.query(hidden_cpu)
                k_cpu = attn_mod.key(hidden_cpu)
                v_cpu = attn_mod.value(hidden_cpu)
                _check(f"layer {layer_idx} Q proj", q_dram, (T_pad, H), q_cpu)
                _check(f"layer {layer_idx} K proj", k_dram, (T_pad, H), k_cpu)
                _check(f"layer {layer_idx} V proj", v_dram, (T_pad, H), v_cpu)

                attn_block_cpu = attn_mod(hidden_cpu, attention_mask=debug_cpu_ref["ext_mask"])[0]
                _check(f"layer {layer_idx} attn block (post-LN)", attn_ln_dram, (T_pad, H), attn_block_cpu)

                # AlbertLayer.forward returns hidden_states directly (a plain Tensor, despite its
                # tuple[...] type hint) in this transformers version -- confirmed via direct
                # inspection, so no [0]/tuple-unpack here.
                hidden_cpu = debug_cpu_ref["layer"](hidden_cpu, attention_mask=debug_cpu_ref["ext_mask"])
                _check(f"layer {layer_idx} full (post-FFN)", hidden_dram, (T_pad, H), hidden_cpu)

        ue.start_capture()
        self._linear(hidden_dram, T_pad, H, 512, w_bert_enc, b_bert_enc, d_en_dram)
        self._run(ue)
        ue.report_timing_and_instruction_count()

        d_en = ue.dma_from_accelerator_memory(d_en_dram, (T_pad, 512))[:T]  # [T, 512]
        ue.reset_tensor_dram_addr()

        if debug_log:
            print("\n[fpga][debug] --- Section 1 bisect (SNR vs CPU, most-recent layer last) ---")
            for stage_name, snr_db in debug_log:
                flag = "  <-- LOW" if snr_db < 30.0 else ""
                print(f"[fpga][debug] {stage_name:38s} {snr_db:8.2f} dB{flag}")
            print("[fpga][debug] --- end bisect ---\n")

        return d_en.float().T  # [512, T]


class ProsodyDurationFPGA:
    """Section 2: the prosody/duration path -- DurationEncoder (3x BiLSTM + AdaLayerNorm) +
    predictor.lstm + duration_proj -> per-phoneme durations -> the frame alignment matrix.

    Confirmed from the checkpoint's actual weight shapes (not just source inspection):
    - Every one of the 4 BiLSTM instances (3 inside DurationEncoder + predictor.lstm) takes a
      640-wide input (weight_ih_l0 is [1024,640]): d_model(512) + style_dim(128) concatenated,
      because DurationEncoder re-concatenates style after EVERY AdaLayerNorm block, including the
      last -- so its returned `d` is itself 640-wide, matching predictor.lstm's expected input.
    - hidden=256 per direction (weight_hh_l0 is [1024,256], 4*256=1024 for the 4 gates), bf16
      throughout, no padding-to-64 needed anywhere here: LSTM steps are all M=1 matmuls (K/N
      already 64-aligned: 640, 256, 1024), so T itself never needs rounding the way the T_pad
      attention/matmul tiling in Section 1 did.

    nn.Linear/nn.LSTM weight layout already matches matmat_mul_core's B=[N,K] convention (see
    Section 1's header note), so every weight here is uploaded as-is, no transpose needed.
    """

    def __init__(self, model, ue: UnifiedEngine):
        self.model = model
        self.ue = ue
        # NOT reused from Section 1: PLBertFPGA.forward() calls reset_tensor_dram_addr() at the end
        # of its own run, which invalidates every DRAM address it allocated. Uploaded fresh in
        # forward() below (after this constructor's weight uploads, which happen before any capture).
        self.identity_dram = None
        self.style_dim = 128
        self.C = 512   # d_model
        self.H = 256   # per-direction LSTM hidden

        pred = model.predictor
        te = pred.text_encoder

        def lstm_w(m):
            return dict(
                Wx_f=_bf16(m.weight_ih_l0), Wh_f=_bf16(m.weight_hh_l0),
                b_f=_bf16(m.bias_ih_l0 + m.bias_hh_l0),
                Wx_b=_bf16(m.weight_ih_l0_reverse), Wh_b=_bf16(m.weight_hh_l0_reverse),
                b_b=_bf16(m.bias_ih_l0_reverse + m.bias_hh_l0_reverse),
            )

        def adaln_w(m):
            return dict(fc_w=_bf16(m.fc.weight), fc_b=_bf16(m.fc.bias))  # [1024,128], [1024]

        self.lstm_blocks = [lstm_w(te.lstms[0]), lstm_w(te.lstms[2]), lstm_w(te.lstms[4])]
        self.adaln_blocks = [adaln_w(te.lstms[1]), adaln_w(te.lstms[3]), adaln_w(te.lstms[5])]
        self.pred_lstm_w = lstm_w(pred.lstm)

        # duration_proj's output width is max_dur=50, NOT a multiple of UE_VECTOR_SIZE(64) -- unlike
        # every other matmul dim in this pipeline. Pad N to 64 with zero rows/entries (the padding
        # columns' garbage never gets read back; we always slice to :50 after readback) rather than
        # run matmat_mul_core with an unaligned N, which produced -inf dB (likely NaN/Inf from
        # unwritten padding in the hardware's internal N-chunk tiling).
        self.dur_n = 50
        self.dur_n_pad = _round_up(self.dur_n, UE_VECTOR_SIZE)
        dur_w_raw = _bf16(pred.duration_proj.linear_layer.weight)  # [50, 512]
        dur_b_raw = _bf16(pred.duration_proj.linear_layer.bias)    # [50]
        self.dur_w = torch.zeros(self.dur_n_pad, self.C, dtype=torch.bfloat16)
        self.dur_w[:self.dur_n] = dur_w_raw
        self.dur_b = torch.zeros(self.dur_n_pad, dtype=torch.bfloat16)
        self.dur_b[:self.dur_n] = dur_b_raw

    def _up(self, tensor: torch.Tensor) -> int:
        a = self.ue.allocate_tensor_dram(tensor.numel() * 2)
        self.ue.dma_to_accelerator_memory(a, tensor.reshape(-1))
        return a

    def _run(self, ue):
        """Flush the current captured instruction stream to the device and execute it. dma_to/from_
        accelerator_memory are direct host<->device PCIe DMA calls (confirmed via user_dma_core.py:
        dma_write/dma_read on the raw xdma char devices) -- completely separate from this
        capture/execute queue mechanism, so _up() weight uploads are safe to interleave anywhere.
        Only the compute calls (matmat_mul_core, eltwise_core_dram, etc.) need this bracket.
        """
        ue.stop_capture()
        ue.generate_instruction_halt()
        prog = ue.get_program_dram_addr()
        ue.write_captured_instructions_to_dram(prog)
        ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())
        ue.start_execute_from_dram(prog)
        ue.wait_queue(30.0)
        ue.clear_capture_buffer(); ue.reset_program_dram_addr()

    def _lstm_bidir(self, x_dram, T, Cin, w, out_dram):
        """Runs both directions of one BiLSTM over T (unrolled at capture time, same per-timestep
        decomposition validated at 40.5dB in the standalone LSTM hardware proof: batched gate
        matmul + native sigmoid + tanh(x)=2*sigmoid(2x)-1 + elementwise cell/hidden update).
        Writes directly into out_dram's [T, 2H] layout -- forward half at column 0, backward half
        at column H -- since every per-timestep write is M=1 (a single [1,H] vector), the
        destination address can be any byte offset, no separate merge/permute step needed.
        """
        ue, H = self.ue, self.H
        G = 4 * H

        def run_direction(reverse, Wx, Wh, b, col_off_bytes):
            wx_dram, wh_dram, b_dram = self._up(Wx), self._up(Wh), self._up(b)
            gates_x = ue.allocate_tensor_dram(G * 2)
            gates_h = ue.allocate_tensor_dram(G * 2)
            gates = ue.allocate_tensor_dram(G * 2)
            i_d, f_d, g_d, o_d = (ue.allocate_tensor_dram(H * 2) for _ in range(4))
            g_tmp, c_tmp, tanh_c, fc_, ig_ = (ue.allocate_tensor_dram(H * 2) for _ in range(5))
            h_bufs = [ue.allocate_tensor_dram(H * 2), ue.allocate_tensor_dram(H * 2)]
            c_bufs = [ue.allocate_tensor_dram(H * 2), ue.allocate_tensor_dram(H * 2)]
            ue.dma_to_accelerator_memory(h_bufs[0], torch.zeros(H, dtype=torch.bfloat16))
            ue.dma_to_accelerator_memory(c_bufs[0], torch.zeros(H, dtype=torch.bfloat16))

            M_H = H // UE_VECTOR_SIZE

            def sigmoid_ip(src, dst):
                ue.activation_core(M=M_H, N=UE_VECTOR_SIZE, A_DRAM_ADDR=src, OUTPUT_DRAM_ADDR=dst,
                                    IDENTITY_DRAM_ADDR=self.identity_dram, activation="sigmoid")

            def tanh_via_sigmoid(src, tmp, dst):
                ue.eltwise_core_dram(1, H, src, None, tmp, mode=UE_MODE.MUL_BROADCAST, scalar=2.0)
                sigmoid_ip(tmp, tmp)
                ue.eltwise_core_dram(1, H, tmp, None, tmp, mode=UE_MODE.MUL_BROADCAST, scalar=2.0)
                ue.eltwise_core_dram(1, H, tmp, None, dst, mode=UE_MODE.ADD_BROADCAST, scalar=-1.0)

            time_range = range(T - 1, -1, -1) if reverse else range(T)
            h_prev, c_prev = h_bufs[0], c_bufs[0]
            for step, t in enumerate(time_range):
                x_t = x_dram + t * Cin * 2
                h_cur, c_cur = h_bufs[(step + 1) % 2], c_bufs[(step + 1) % 2]

                ue.matmat_mul_core(M=1, K=Cin, N=G, A_DRAM_ADDR=x_t, B_DRAM_ADDR=wx_dram,
                                    OUTPUT_DRAM_ADDR=gates_x, C_DRAM_ADDR=b_dram, bias_mode="broadcast_N")
                ue.matmat_mul_core(M=1, K=H, N=G, A_DRAM_ADDR=h_prev, B_DRAM_ADDR=wh_dram, OUTPUT_DRAM_ADDR=gates_h)
                ue.eltwise_core_dram(1, G, gates_x, gates_h, gates, mode=UE_MODE.ELTWISE_ADD)

                i_raw, f_raw, g_raw, o_raw = (gates + k * H * 2 for k in range(4))
                sigmoid_ip(i_raw, i_d); sigmoid_ip(f_raw, f_d); sigmoid_ip(o_raw, o_d)
                tanh_via_sigmoid(g_raw, g_tmp, g_d)

                ue.eltwise_core_dram(1, H, f_d, c_prev, fc_, mode=UE_MODE.ELTWISE_MUL)
                ue.eltwise_core_dram(1, H, i_d, g_d, ig_, mode=UE_MODE.ELTWISE_MUL)
                ue.eltwise_core_dram(1, H, fc_, ig_, c_cur, mode=UE_MODE.ELTWISE_ADD)

                tanh_via_sigmoid(c_cur, c_tmp, tanh_c)
                ue.eltwise_core_dram(1, H, o_d, tanh_c, h_cur, mode=UE_MODE.ELTWISE_MUL)
                out_addr = out_dram + t * (2 * H) * 2 + col_off_bytes
                ue.eltwise_core_dram(1, H, o_d, tanh_c, out_addr, mode=UE_MODE.ELTWISE_MUL)

                h_prev, c_prev = h_cur, c_cur

        run_direction(False, w["Wx_f"], w["Wh_f"], w["b_f"], 0)
        run_direction(True, w["Wx_b"], w["Wh_b"], w["b_b"], H * 2)

    def _concat_columns(self, a_dram, a_width, b_dram, b_width, T, out_dram):
        """out[T, a_width+b_width] = concat([a[T,a_width], b[T,b_width]], dim=-1), via per-row
        copies (T is small here, <200 typically -- same technique as Section 1's embedding gather).
        """
        ue = self.ue
        out_w_bytes = (a_width + b_width) * 2
        a_row_bytes, b_row_bytes = a_width * 2, b_width * 2
        for t in range(T):
            ue.accelerator_memory_to_sram(accelerator_dram_address=a_dram + t * a_row_bytes,
                                           sram_address=0x00000, element_size=a_width)
            ue.sram_to_accelerator_memory(sram_address=0x00000,
                                           accelerator_dram_address=out_dram + t * out_w_bytes,
                                           element_size=a_width)
            ue.accelerator_memory_to_sram(accelerator_dram_address=b_dram + t * b_row_bytes,
                                           sram_address=0x00000, element_size=b_width)
            ue.sram_to_accelerator_memory(sram_address=0x00000,
                                           accelerator_dram_address=out_dram + t * out_w_bytes + a_row_bytes,
                                           element_size=b_width)

    def _adaln(self, x_dram, T, C, style_dram, w, out_dram):
        """AdaLayerNorm: plain (no-affine) LayerNorm over the channel axis, then a style-conditioned
        affine (1+gamma)*x_hat + beta where [gamma;beta] = style @ fc_w^T + fc_b. gamma/beta are the
        SAME for every T position (style doesn't vary by time), so -- as in Section 1's InstanceNorm
        proof -- they're computed once on hardware (a tiny M=1 matmul), read back, host-tiled to
        [T,C], and applied via plain eltwise (no per-channel-vector broadcast primitive exists).

        SELF-CONTAINED capture/execute: the gamma/beta readback is a genuine host round-trip (we
        need the actual computed values on the host to build the tiled broadcast buffers), so this
        method brackets its own two capture/execute cycles rather than assuming the caller has one
        open -- reading back mid-capture would read stale/uninitialized DRAM, since a captured
        instruction has only been *recorded* at that point, not yet run via start_execute_from_dram
        (this was the actual bug behind Section 2's first-pass catastrophic SNR).
        """
        ue = self.ue
        gb_dram = ue.allocate_tensor_dram(2 * C * 2)
        ue.start_capture()
        ue.matmat_mul_core(M=1, K=self.style_dim, N=2 * C, A_DRAM_ADDR=style_dram, B_DRAM_ADDR=self._up(w["fc_w"]),
                            OUTPUT_DRAM_ADDR=gb_dram, C_DRAM_ADDR=self._up(w["fc_b"]), bias_mode="broadcast_N")
        self._run(ue)

        gb = ue.dma_from_accelerator_memory(gb_dram, (2 * C,))
        gamma, beta = gb[:C], gb[C:]
        gamma_tiled = self._up(gamma.unsqueeze(0).expand(T, C).contiguous())
        beta_tiled = self._up(beta.unsqueeze(0).expand(T, C).contiguous())

        normed_dram = ue.allocate_tensor_dram(T * C * 2)
        ng_dram = ue.allocate_tensor_dram(T * C * 2)
        tmp_dram = ue.allocate_tensor_dram(T * C * 2)
        ue.start_capture()
        ue.layer_norm_core_dram(M=T, N=C, A_DRAM_ADDR=x_dram, OUTPUT_DRAM_ADDR=normed_dram)
        ue.eltwise_core_dram(T, C, normed_dram, gamma_tiled, ng_dram, mode=UE_MODE.ELTWISE_MUL)   # normed*gamma
        ue.eltwise_core_dram(T, C, normed_dram, ng_dram, tmp_dram, mode=UE_MODE.ELTWISE_ADD)        # normed*(1+gamma)
        ue.eltwise_core_dram(T, C, tmp_dram, beta_tiled, out_dram, mode=UE_MODE.ELTWISE_ADD)         # + beta
        self._run(ue)

    def forward(self, d_en: torch.Tensor, style_vec: torch.Tensor, T: int, debug_cpu_ref=None):
        """d_en: [512, T] (Section 1's output). style_vec: [128] (ref_s[:,128:] squeezed).
        Returns (d [T,640], pred_dur [T] LongTensor) matching KokoroModel.forward_with_tokens's
        `d` and `pred_dur` at this point in the pipeline.
        """
        ue = self.ue
        C, style_dim = self.C, self.style_dim
        debug_log = []

        def _check(name, dram_addr, shape, cpu_ref, cols=None):
            """shape is the buffer's ACTUAL (possibly padded) layout in DRAM -- always read that,
            then slice to the first `cols` real columns if the buffer is wider than the logical
            (unpadded) data (e.g. duration_proj's N=50 padded to 64). Reading a shape narrower than
            the true row stride would misread strided/shifted data, not just extra padding.
            """
            if debug_cpu_ref is None:
                return
            got = ue.dma_from_accelerator_memory(dram_addr, shape).float()
            if cols is not None:
                got = got[:, :cols]
            from user_dma_core import calculate_snr
            snr_db = calculate_snr(cpu_ref.detach().float().reshape(-1), got.reshape(-1))
            debug_log.append((name, snr_db))

        self.identity_dram = self._up(torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))

        style_dram = self._up(_bf16(style_vec))
        style_tiled_dram = self._up(_bf16(style_vec).unsqueeze(0).expand(T, style_dim).contiguous())

        x_dram = ue.allocate_tensor_dram(T * C * 2)
        ue.dma_to_accelerator_memory(x_dram, _bf16(d_en.T))  # [T, 512]

        # --- Stage A: 3x (BiLSTM + AdaLayerNorm + style-concat) -> d ---
        # _adaln is self-contained (it needs its own host round-trip for gamma/beta, see its
        # docstring), so it CANNOT be called from inside another active capture -- LSTM and concat
        # each get their own bracket instead of one big Stage-A capture.
        ue.start_capture()
        cur_dram = ue.allocate_tensor_dram(T * (C + style_dim) * 2)
        self._concat_columns(x_dram, C, style_tiled_dram, style_dim, T, cur_dram)  # [T, 640]
        self._run(ue)

        for lw, aw in zip(self.lstm_blocks, self.adaln_blocks):
            ue.start_capture()
            lstm_out_dram = ue.allocate_tensor_dram(T * C * 2)
            self._lstm_bidir(cur_dram, T, C + style_dim, lw, lstm_out_dram)  # [T, 512]
            self._run(ue)

            adaln_out_dram = ue.allocate_tensor_dram(T * C * 2)
            self._adaln(lstm_out_dram, T, C, style_dram, aw, adaln_out_dram)  # [T, 512]

            ue.start_capture()
            cur_dram = ue.allocate_tensor_dram(T * (C + style_dim) * 2)
            self._concat_columns(adaln_out_dram, C, style_tiled_dram, style_dim, T, cur_dram)  # [T, 640]
            self._run(ue)
        d_dram = cur_dram  # [T, 640] -- matches KokoroModel.forward_with_tokens's `d`
        _check("d (DurationEncoder output)", d_dram, (T, C + style_dim), debug_cpu_ref["d"] if debug_cpu_ref else None)

        # --- Stage B: predictor.lstm ---
        ue.start_capture()
        pred_lstm_out_dram = ue.allocate_tensor_dram(T * C * 2)
        self._lstm_bidir(d_dram, T, C + style_dim, self.pred_lstm_w, pred_lstm_out_dram)  # [T, 512]
        self._run(ue)
        _check("predictor.lstm output", pred_lstm_out_dram, (T, C), debug_cpu_ref["lstm_out"] if debug_cpu_ref else None)

        # --- Stage C: duration_proj (N padded to 64, see __init__'s dur_w/dur_b comment) ---
        ue.start_capture()
        dur_dram = ue.allocate_tensor_dram(T * self.dur_n_pad * 2)
        ue.matmat_mul_core(M=T, K=C, N=self.dur_n_pad, A_DRAM_ADDR=pred_lstm_out_dram, B_DRAM_ADDR=self._up(self.dur_w),
                            OUTPUT_DRAM_ADDR=dur_dram, C_DRAM_ADDR=self._up(self.dur_b), bias_mode="broadcast_N")
        self._run(ue)
        _check("duration_proj (pre-sigmoid)", dur_dram, (T, self.dur_n_pad),
                debug_cpu_ref["duration_raw"] if debug_cpu_ref else None, cols=self.dur_n)

        if debug_cpu_ref is not None:
            print("\n[fpga][debug] --- Section 2 bisect (SNR vs CPU) ---")
            for name, snr_db in debug_log:
                flag = "  <-- LOW" if snr_db < 30.0 else ""
                print(f"[fpga][debug] {name:38s} {snr_db:8.2f} dB{flag}")
            print("[fpga][debug] --- end bisect ---\n")

        duration_raw = ue.dma_from_accelerator_memory(dur_dram, (T, self.dur_n_pad)).float()[:, :self.dur_n]
        duration = torch.sigmoid(duration_raw).sum(axis=-1)
        pred_dur = torch.round(duration).clamp(min=1).long()
        d_out = ue.dma_from_accelerator_memory(d_dram, (T, C + style_dim)).float()
        ue.reset_tensor_dram_addr()
        return d_out, pred_dur


def _wn_conv(module):
    """Materialize a torch.nn.utils.weight_norm-wrapped Conv1d/ConvTranspose1d's effective weight
    from its (weight_g, weight_v) parametrization: w = g * v / ||v||_{dim=0} (the ATen op PyTorch's
    weight_norm hook computes internally). Falls back to a plain `.weight` for un-wrapped modules
    (e.g. F0_proj/N_proj, confirmed via the checkpoint's state dict to have no weight_g/weight_v).
    Computed directly from (g, v) rather than relying on the module's forward-hook side effect,
    since some of these conv modules may not have been run yet at the point we read their weight.
    """
    if hasattr(module, "weight_v"):
        w = torch._weight_norm(module.weight_v, module.weight_g, 0)
    else:
        w = module.weight
    b = module.bias if module.bias is not None else None
    return _bf16(w), (_bf16(b) if b is not None else None)


class F0NPredictionFPGA:
    """Section 3: predictor.shared LSTM + pred_aln_trg (phoneme->frame alignment) construction +
    F0Ntrain's two AdainResBlk1d stacks (F0 and N branches) -> F0_pred, N_pred.

    New op types introduced here (all individually validated in the earlier op-gap hardware proofs,
    being composed into the model for the first time):
      - Conv1d (k=3, pad=1, regular + k=1 pointwise) -> shifted-matmul-accumulate / plain matmul
        (conv1d_shifted_matmul_test, both at 40-51dB on real hardware).
      - AdaIN1d (InstanceNorm1d + style-conditioned per-channel affine) -> layer_norm_core_dram on
        a transposed [C,T] view (InstanceNorm1d proof, 46.5dB) + the same style-affine round-trip
        pattern as Section 2's AdaLayerNorm.
      - Depthwise ConvTranspose1d (the `pool` in upsampling blocks) -> zero-insertion (native
        bf16_permute_dram_core strided scatter) + depthwise shifted-eltwise-MAC with the flipped
        kernel (conv_transpose1d_zero_insert_test + depthwise_conv1d_eltwise_test, 49-53dB).
      - Nearest 2x upsample (shortcut path) -> exact row-duplicate (interpolate_fixed_matmul_test's
        nearest mode was bit-exact / inf dB).
      - LeakyReLU(0.2) -> relu(x) - 0.2*relu(-x), derived exactly from the native `clamp` activation
        (clamp_min=0, clamp_max=inf IS relu), no new primitive.

    Internal layout convention: [T, C] row-major (T=frames as rows) throughout, matching Sections
    1-2 and the conv1d_shifted_matmul_test proof (shifted reads are row offsets). AdaIN1d needs the
    OPPOSITE layout ([C, T], channels as rows) to use layer_norm_core_dram's per-row reduction --
    bf16_transpose_core (a validated native primitive) converts between the two as needed.
    """

    def __init__(self, model, ue: UnifiedEngine):
        self.model = model
        self.ue = ue
        self.style_dim = 128
        self.H = 256  # predictor.shared per-direction hidden
        self.identity_dram = None  # uploaded fresh in forward(), same reasoning as Section 2

        pred = model.predictor

        def lstm_w(m):
            return dict(
                Wx_f=_bf16(m.weight_ih_l0), Wh_f=_bf16(m.weight_hh_l0),
                b_f=_bf16(m.bias_ih_l0 + m.bias_hh_l0),
                Wx_b=_bf16(m.weight_ih_l0_reverse), Wh_b=_bf16(m.weight_hh_l0_reverse),
                b_b=_bf16(m.bias_ih_l0_reverse + m.bias_hh_l0_reverse),
            )

        def block_w(m, has_upsample):
            w = dict(
                conv1_w=None, conv1_b=None, conv2_w=None, conv2_b=None,
                norm1_fc_w=_bf16(m.norm1.fc.weight), norm1_fc_b=_bf16(m.norm1.fc.bias),
                norm2_fc_w=_bf16(m.norm2.fc.weight), norm2_fc_b=_bf16(m.norm2.fc.bias),
            )
            w["conv1_w"], w["conv1_b"] = _wn_conv(m.conv1)
            w["conv2_w"], w["conv2_b"] = _wn_conv(m.conv2)
            if m.learned_sc:
                w["conv1x1_w"], _ = _wn_conv(m.conv1x1)
            if has_upsample:
                w["pool_w"], w["pool_b"] = _wn_conv(m.pool)
            return w

        self.shared_w = lstm_w(pred.shared)
        self.F0_blocks = [block_w(pred.F0[i], i == 1) for i in range(3)]
        self.N_blocks = [block_w(pred.N[i], i == 1) for i in range(3)]

        self.proj_n = 1
        self.proj_n_pad = _round_up(self.proj_n, UE_VECTOR_SIZE)
        f0_proj_w, f0_proj_b = pred.F0_proj.weight, pred.F0_proj.bias  # [1,256,1], [1] -- not weight_norm wrapped
        n_proj_w, n_proj_b = pred.N_proj.weight, pred.N_proj.bias
        self.F0_proj_w = torch.zeros(self.proj_n_pad, 256, dtype=torch.bfloat16)
        self.F0_proj_w[:1] = _bf16(f0_proj_w.squeeze(-1))
        self.F0_proj_b = torch.zeros(self.proj_n_pad, dtype=torch.bfloat16); self.F0_proj_b[:1] = _bf16(f0_proj_b)
        self.N_proj_w = torch.zeros(self.proj_n_pad, 256, dtype=torch.bfloat16)
        self.N_proj_w[:1] = _bf16(n_proj_w.squeeze(-1))
        self.N_proj_b = torch.zeros(self.proj_n_pad, dtype=torch.bfloat16); self.N_proj_b[:1] = _bf16(n_proj_b)

    def _up(self, tensor: torch.Tensor) -> int:
        a = self.ue.allocate_tensor_dram(tensor.numel() * 2)
        self.ue.dma_to_accelerator_memory(a, tensor.reshape(-1))
        return a

    def _run(self, ue):
        ue.stop_capture()
        ue.generate_instruction_halt()
        prog = ue.get_program_dram_addr()
        ue.write_captured_instructions_to_dram(prog)
        ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())
        ue.start_execute_from_dram(prog)
        ue.wait_queue(30.0)
        ue.clear_capture_buffer(); ue.reset_program_dram_addr()

    def _lstm_bidir(self, x_dram, T, Cin, w, out_dram):
        """Identical to ProsodyDurationFPGA's (see its docstring for the full derivation) --
        duplicated rather than shared across classes to keep each section's file self-contained."""
        ue, H = self.ue, self.H
        G = 4 * H

        def run_direction(reverse, Wx, Wh, b, col_off_bytes):
            wx_dram, wh_dram, b_dram = self._up(Wx), self._up(Wh), self._up(b)
            gates_x = ue.allocate_tensor_dram(G * 2)
            gates_h = ue.allocate_tensor_dram(G * 2)
            gates = ue.allocate_tensor_dram(G * 2)
            i_d, f_d, g_d, o_d = (ue.allocate_tensor_dram(H * 2) for _ in range(4))
            g_tmp, c_tmp, tanh_c, fc_, ig_ = (ue.allocate_tensor_dram(H * 2) for _ in range(5))
            h_bufs = [ue.allocate_tensor_dram(H * 2), ue.allocate_tensor_dram(H * 2)]
            c_bufs = [ue.allocate_tensor_dram(H * 2), ue.allocate_tensor_dram(H * 2)]
            ue.dma_to_accelerator_memory(h_bufs[0], torch.zeros(H, dtype=torch.bfloat16))
            ue.dma_to_accelerator_memory(c_bufs[0], torch.zeros(H, dtype=torch.bfloat16))
            M_H = H // UE_VECTOR_SIZE

            def sigmoid_ip(src, dst):
                ue.activation_core(M=M_H, N=UE_VECTOR_SIZE, A_DRAM_ADDR=src, OUTPUT_DRAM_ADDR=dst,
                                    IDENTITY_DRAM_ADDR=self.identity_dram, activation="sigmoid")

            def tanh_via_sigmoid(src, tmp, dst):
                ue.eltwise_core_dram(1, H, src, None, tmp, mode=UE_MODE.MUL_BROADCAST, scalar=2.0)
                sigmoid_ip(tmp, tmp)
                ue.eltwise_core_dram(1, H, tmp, None, tmp, mode=UE_MODE.MUL_BROADCAST, scalar=2.0)
                ue.eltwise_core_dram(1, H, tmp, None, dst, mode=UE_MODE.ADD_BROADCAST, scalar=-1.0)

            time_range = range(T - 1, -1, -1) if reverse else range(T)
            h_prev, c_prev = h_bufs[0], c_bufs[0]
            for step, t in enumerate(time_range):
                x_t = x_dram + t * Cin * 2
                h_cur, c_cur = h_bufs[(step + 1) % 2], c_bufs[(step + 1) % 2]
                ue.matmat_mul_core(M=1, K=Cin, N=G, A_DRAM_ADDR=x_t, B_DRAM_ADDR=wx_dram,
                                    OUTPUT_DRAM_ADDR=gates_x, C_DRAM_ADDR=b_dram, bias_mode="broadcast_N")
                ue.matmat_mul_core(M=1, K=H, N=G, A_DRAM_ADDR=h_prev, B_DRAM_ADDR=wh_dram, OUTPUT_DRAM_ADDR=gates_h)
                ue.eltwise_core_dram(1, G, gates_x, gates_h, gates, mode=UE_MODE.ELTWISE_ADD)
                i_raw, f_raw, g_raw, o_raw = (gates + k * H * 2 for k in range(4))
                sigmoid_ip(i_raw, i_d); sigmoid_ip(f_raw, f_d); sigmoid_ip(o_raw, o_d)
                tanh_via_sigmoid(g_raw, g_tmp, g_d)
                ue.eltwise_core_dram(1, H, f_d, c_prev, fc_, mode=UE_MODE.ELTWISE_MUL)
                ue.eltwise_core_dram(1, H, i_d, g_d, ig_, mode=UE_MODE.ELTWISE_MUL)
                ue.eltwise_core_dram(1, H, fc_, ig_, c_cur, mode=UE_MODE.ELTWISE_ADD)
                tanh_via_sigmoid(c_cur, c_tmp, tanh_c)
                ue.eltwise_core_dram(1, H, o_d, tanh_c, h_cur, mode=UE_MODE.ELTWISE_MUL)
                out_addr = out_dram + t * (2 * H) * 2 + col_off_bytes
                ue.eltwise_core_dram(1, H, o_d, tanh_c, out_addr, mode=UE_MODE.ELTWISE_MUL)
                h_prev, c_prev = h_cur, c_cur

        run_direction(False, w["Wx_f"], w["Wh_f"], w["b_f"], 0)
        run_direction(True, w["Wx_b"], w["Wh_b"], w["b_b"], H * 2)

    def _device_row_copy(self, src_dram, dst_dram, num_rows, row_width):
        """Pure on-device row-by-row copy: CAPTURED instructions (accelerator_memory_to_sram ->
        sram_to_accelerator_memory, same technique as Section 1's embedding gather), NOT a host
        round-trip. Safe to call from inside an already-open capture block -- unlike reading a
        same-capture-block predecessor's output back to the host, which would read stale/
        uninitialized DRAM (that instruction hasn't executed yet, only been recorded). This is the
        device-to-device copy this class uses everywhere it needs to restage data (padding,
        zero-insertion, etc).
        """
        ue = self.ue
        row_bytes = row_width * 2
        for r in range(num_rows):
            ue.accelerator_memory_to_sram(accelerator_dram_address=src_dram + r * row_bytes,
                                           sram_address=0x00000, element_size=row_width)
            ue.sram_to_accelerator_memory(sram_address=0x00000,
                                           accelerator_dram_address=dst_dram + r * row_bytes, element_size=row_width)

    def _conv1d(self, x_dram, T, Cin, Cout, W, b, out_dram, kernel_size, pad):
        """Conv1d via K shifted-matmul-accumulate (conv1d_shifted_matmul_test's validated pattern).
        W: [Cout, Cin, kernel_size] bf16. kernel_size=1,pad=0 degenerates to a single plain matmul
        (used for the pointwise conv1x1/F0_proj/N_proj cases).
        """
        ue = self.ue
        if kernel_size == 1:
            ue.matmat_mul_core(M=T, K=Cin, N=Cout, A_DRAM_ADDR=x_dram, B_DRAM_ADDR=self._up(W.squeeze(-1)),
                                OUTPUT_DRAM_ADDR=out_dram,
                                C_DRAM_ADDR=(self._up(b) if b is not None else None),
                                bias_mode="broadcast_N")
            return
        x_pad_dram = ue.allocate_tensor_dram((T + 2 * pad) * Cin * 2)
        ue.dma_to_accelerator_memory(
            x_pad_dram, torch.zeros((T + 2 * pad) * Cin, dtype=torch.bfloat16))  # constant zero-fill, safe direct DMA
        self._device_row_copy(x_dram, x_pad_dram + pad * Cin * 2, T, Cin)  # on-device, NOT a host round-trip
        w_tap_dram = [self._up(W[:, :, k].contiguous()) for k in range(kernel_size)]
        acc_a = ue.allocate_tensor_dram(T * Cout * 2)
        acc_b = ue.allocate_tensor_dram(T * Cout * 2)
        for k in range(kernel_size):
            A_DRAM_ADDR = x_pad_dram + k * Cin * 2
            prev_acc = acc_a if k % 2 == 0 else acc_b
            cur_acc = acc_b if k % 2 == 0 else acc_a
            bias_addr = self._up(b) if (b is not None and k == 0) else None
            ue.matmat_mul_core(
                M=T, K=Cin, N=Cout, A_DRAM_ADDR=A_DRAM_ADDR, B_DRAM_ADDR=w_tap_dram[k], OUTPUT_DRAM_ADDR=cur_acc,
                C_DRAM_ADDR=(bias_addr if k == 0 else (prev_acc if k > 0 else None)),
                bias_mode=("broadcast_N" if (k == 0 and bias_addr is not None) else "full_matrix"),
            )
        final_acc = acc_b if (kernel_size - 1) % 2 == 0 else acc_a
        if final_acc != out_dram:
            self._device_row_copy(final_acc, out_dram, T, Cout)  # on-device, NOT a host round-trip

    def _leaky_relu(self, x_dram, T, C, out_dram):
        ue = self.ue
        numel = T * C
        M, N = numel // UE_VECTOR_SIZE, UE_VECTOR_SIZE
        neg_dram = ue.allocate_tensor_dram(numel * 2)
        relu_dram = ue.allocate_tensor_dram(numel * 2)
        relu_neg_dram = ue.allocate_tensor_dram(numel * 2)
        ue.activation_core(M=M, N=N, A_DRAM_ADDR=x_dram, OUTPUT_DRAM_ADDR=relu_dram,
                            IDENTITY_DRAM_ADDR=self.identity_dram, activation="clamp")
        ue.eltwise_core_dram(M, N, x_dram, None, neg_dram, mode=UE_MODE.MUL_BROADCAST, scalar=-1.0)
        ue.activation_core(M=M, N=N, A_DRAM_ADDR=neg_dram, OUTPUT_DRAM_ADDR=relu_neg_dram,
                            IDENTITY_DRAM_ADDR=self.identity_dram, activation="clamp")
        ue.eltwise_core_dram(M, N, relu_neg_dram, None, relu_neg_dram, mode=UE_MODE.MUL_BROADCAST, scalar=0.2)
        ue.eltwise_core_dram(M, N, relu_dram, relu_neg_dram, out_dram, mode=UE_MODE.ELTWISE_SUB)

    def _nearest_upsample2x(self, x_dram, T, C, out_dram):
        """Exact row-duplicate: out[2t]=out[2t+1]=x[t]. Bit-exact (interpolate_fixed_matmul_test's
        nearest mode measured inf dB), implemented as a per-row copy loop (same technique as
        Section 1's embedding gather)."""
        ue = self.ue
        row_bytes = C * 2
        for t in range(T):
            ue.accelerator_memory_to_sram(accelerator_dram_address=x_dram + t * row_bytes,
                                           sram_address=0x00000, element_size=C)
            ue.sram_to_accelerator_memory(sram_address=0x00000,
                                           accelerator_dram_address=out_dram + (2 * t) * row_bytes, element_size=C)
            ue.sram_to_accelerator_memory(sram_address=0x00000,
                                           accelerator_dram_address=out_dram + (2 * t + 1) * row_bytes, element_size=C)

    def _depthwise_convtranspose_upsample2x(self, x_dram, T, C, W, b, out_dram):
        """Depthwise ConvTranspose1d(C,C,kernel=3,stride=2,padding=1,output_padding=1,groups=C):
        zero-insertion dilate (native bf16_permute_dram_core strided scatter, num_groups=2 so group1
        is an all-zeros buffer) + pad(left=1,right=2, derived in the conversation from the standard
        conv/conv_transpose duality: left=right_base=kernel-1-padding=1, right=right_base+output_
        padding=2) + depthwise 3-tap shifted-eltwise-MAC with the FLIPPED kernel taps. Output length
        (T-1)*stride-2*padding+kernel+output_padding = 2T (verified against the same formula).
        """
        ue = self.ue
        dilated_len = (T - 1) * 2 + 1
        # Grouped [2,T,C] buffer (group0=real x, group1=constant zeros), built via ONE on-device
        # copy (x_dram -> group0; group1 is a constant zero-fill, a safe direct DMA since it doesn't
        # depend on any same-capture-block predecessor) -- then scattered into the interleaved
        # [T,2,C] == dilated [2T-1,C] layout via bf16_permute_dram_core's native strided write.
        grouped_dram = ue.allocate_tensor_dram(2 * T * C * 2)
        ue.dma_to_accelerator_memory(grouped_dram + T * C * 2, torch.zeros(T * C, dtype=torch.bfloat16))
        self._device_row_copy(x_dram, grouped_dram, T, C)  # on-device, NOT a host round-trip
        dilated_dram = ue.allocate_tensor_dram((dilated_len + 1) * C * 2)
        ue.bf16_permute_dram_core(num_groups=2, group_rows=T, row_width=C,
                                   in_dram=grouped_dram, out_dram=dilated_dram, write_grouped=False)

        pad_l, pad_r = 1, 2
        L_pad = pad_l + dilated_len + pad_r
        x_dilated_pad = ue.allocate_tensor_dram(L_pad * C * 2)
        ue.dma_to_accelerator_memory(x_dilated_pad, torch.zeros(L_pad * C, dtype=torch.bfloat16))  # constant, safe
        self._device_row_copy(dilated_dram, x_dilated_pad + pad_l * C * 2, dilated_len, C)  # on-device

        T_out = 2 * T
        assert L_pad - 3 + 1 == T_out, (L_pad, T_out)
        w_tap_tiled = [W[:, 0, 2 - k].unsqueeze(0).expand(T_out, C).contiguous() for k in range(3)]  # flipped taps
        w_tap_dram = [self._up(w_tap_tiled[k]) for k in range(3)]
        tmp = [ue.allocate_tensor_dram(T_out * C * 2) for _ in range(3)]
        acc_a = ue.allocate_tensor_dram(T_out * C * 2)
        acc_b = ue.allocate_tensor_dram(T_out * C * 2)
        running = None
        for k in range(3):
            shifted_x = x_dilated_pad + k * C * 2
            ue.eltwise_core_dram(T_out, C, shifted_x, w_tap_dram[k], tmp[k], mode=UE_MODE.ELTWISE_MUL)
            if k == 0:
                running = tmp[0]
                continue
            cur = acc_a if k % 2 == 1 else acc_b
            ue.eltwise_core_dram(T_out, C, tmp[k], running, cur, mode=UE_MODE.ELTWISE_ADD)
            running = cur
        b_tiled_dram = self._up(b.unsqueeze(0).expand(T_out, C).contiguous())
        ue.eltwise_core_dram(T_out, C, running, b_tiled_dram, out_dram, mode=UE_MODE.ELTWISE_ADD)

    def _adain1d(self, x_dram, T, C, style_dram, fc_w, fc_b, out_dram):
        """AdaIN1d: InstanceNorm1d (no affine) + style-conditioned per-channel affine. SELF-CONTAINED
        capture/execute (needs its own host round-trip for gamma/beta, same reasoning as Section 2's
        AdaLayerNorm) -- see instance_norm1d_via_layernorm_test: InstanceNorm1d == layer_norm_core_
        dram on a [C,T] (channels-as-rows) view, no permute needed once there, just the OPPOSITE of
        this class's usual [T,C] convention -- so we transpose in, normalize, transpose back out.
        """
        ue = self.ue
        gb_dram = ue.allocate_tensor_dram(2 * C * 2)
        ue.start_capture()
        ue.matmat_mul_core(M=1, K=self.style_dim, N=2 * C, A_DRAM_ADDR=style_dram, B_DRAM_ADDR=self._up(fc_w),
                            OUTPUT_DRAM_ADDR=gb_dram, C_DRAM_ADDR=self._up(fc_b), bias_mode="broadcast_N")
        x_ct_dram = ue.allocate_tensor_dram(T * C * 2)
        ue.bf16_transpose_core(M=T, N=C, INPUT_DRAM_ADDR=x_dram, OUTPUT_DRAM_ADDR=x_ct_dram,
                                IDENTITY_DRAM_ADDR=self.identity_dram)
        self._run(ue)

        gb = ue.dma_from_accelerator_memory(gb_dram, (2 * C,))
        gamma, beta = gb[:C], gb[C:]
        gamma_tiled = self._up(gamma.unsqueeze(1).expand(C, T).contiguous())
        beta_tiled = self._up(beta.unsqueeze(1).expand(C, T).contiguous())

        normed_ct = ue.allocate_tensor_dram(C * T * 2)
        ng_ct = ue.allocate_tensor_dram(C * T * 2)
        tmp_ct = ue.allocate_tensor_dram(C * T * 2)
        out_ct = ue.allocate_tensor_dram(C * T * 2)
        ue.start_capture()
        ue.layer_norm_core_dram(M=C, N=T, A_DRAM_ADDR=x_ct_dram, OUTPUT_DRAM_ADDR=normed_ct)
        ue.eltwise_core_dram(C, T, normed_ct, gamma_tiled, ng_ct, mode=UE_MODE.ELTWISE_MUL)
        ue.eltwise_core_dram(C, T, normed_ct, ng_ct, tmp_ct, mode=UE_MODE.ELTWISE_ADD)
        ue.eltwise_core_dram(C, T, tmp_ct, beta_tiled, out_ct, mode=UE_MODE.ELTWISE_ADD)
        ue.bf16_transpose_core(M=C, N=T, INPUT_DRAM_ADDR=out_ct, OUTPUT_DRAM_ADDR=out_dram,
                                IDENTITY_DRAM_ADDR=self.identity_dram)
        self._run(ue)

    def _adain_res_blk(self, x_dram, T, Cin, Cout, upsample, w, style_dram):
        """One AdainResBlk1d. Returns (out_dram, T_out)."""
        ue = self.ue
        T_out = 2 * T if upsample else T
        learned_sc = Cin != Cout

        h1_dram = ue.allocate_tensor_dram(T * Cin * 2)
        self._adain1d(x_dram, T, Cin, style_dram, w["norm1_fc_w"], w["norm1_fc_b"], h1_dram)

        ue.start_capture()
        act1_dram = ue.allocate_tensor_dram(T * Cin * 2)
        self._leaky_relu(h1_dram, T, Cin, act1_dram)
        if upsample:
            pooled_dram = ue.allocate_tensor_dram(T_out * Cin * 2)
            self._depthwise_convtranspose_upsample2x(act1_dram, T, Cin, w["pool_w"], w["pool_b"], pooled_dram)
        else:
            pooled_dram = act1_dram
        conv1_out_dram = ue.allocate_tensor_dram(T_out * Cout * 2)
        self._conv1d(pooled_dram, T_out, Cin, Cout, w["conv1_w"], w["conv1_b"], conv1_out_dram, kernel_size=3, pad=1)
        self._run(ue)

        h2_dram = ue.allocate_tensor_dram(T_out * Cout * 2)
        self._adain1d(conv1_out_dram, T_out, Cout, style_dram, w["norm2_fc_w"], w["norm2_fc_b"], h2_dram)

        ue.start_capture()
        act2_dram = ue.allocate_tensor_dram(T_out * Cout * 2)
        self._leaky_relu(h2_dram, T_out, Cout, act2_dram)
        residual_dram = ue.allocate_tensor_dram(T_out * Cout * 2)
        self._conv1d(act2_dram, T_out, Cout, Cout, w["conv2_w"], w["conv2_b"], residual_dram, kernel_size=3, pad=1)

        if upsample:
            sc_dram = ue.allocate_tensor_dram(T_out * Cin * 2)
            self._nearest_upsample2x(x_dram, T, Cin, sc_dram)
        else:
            sc_dram = x_dram
        if learned_sc:
            sc_proj_dram = ue.allocate_tensor_dram(T_out * Cout * 2)
            self._conv1d(sc_dram, T_out, Cin, Cout, w["conv1x1_w"], None, sc_proj_dram, kernel_size=1, pad=0)
            sc_dram = sc_proj_dram

        out_dram = ue.allocate_tensor_dram(T_out * Cout * 2)
        sum_dram = ue.allocate_tensor_dram(T_out * Cout * 2)
        ue.eltwise_core_dram(T_out, Cout, residual_dram, sc_dram, sum_dram, mode=UE_MODE.ELTWISE_ADD)
        ue.eltwise_core_dram(T_out, Cout, sum_dram, None, out_dram, mode=UE_MODE.MUL_BROADCAST, scalar=0.7071067811865476)
        self._run(ue)
        return out_dram, T_out

    def forward(self, d: torch.Tensor, pred_dur: torch.LongTensor, style_vec: torch.Tensor, T: int,
                debug_cpu_ref=None):
        """d: [T,640] (Section 2's output). pred_dur: [T] LongTensor (Section 2's output, exact).
        style_vec: [128]. Returns (F0_pred [n_frames], N_pred [n_frames]) FloatTensors.
        """
        ue = self.ue
        debug_log = []

        def _check(name, dram_addr, shape, cpu_ref, cols=None):
            if debug_cpu_ref is None:
                return
            got = ue.dma_from_accelerator_memory(dram_addr, shape).float()
            if cols is not None:
                got = got[:, :cols]
            from user_dma_core import calculate_snr
            snr_db = calculate_snr(cpu_ref.detach().float().reshape(-1), got.reshape(-1))
            debug_log.append((name, snr_db))

        self.identity_dram = self._up(torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))
        style_dram = self._up(_bf16(style_vec))

        # --- pred_aln_trg (host-built: pred_dur is already exact, this is pure data staging, same
        #     spirit as Section 1's contiguous position-embedding slice) + en = d^T @ pred_aln_trg ---
        n_frames = int(pred_dur.sum().item())
        frame_idx = torch.repeat_interleave(torch.arange(T), pred_dur)  # [n_frames], phoneme index per frame
        pred_aln_trg_T = torch.zeros(n_frames, T, dtype=torch.bfloat16)
        pred_aln_trg_T[torch.arange(n_frames), frame_idx] = 1.0

        ue.start_capture()
        en_dram = ue.allocate_tensor_dram(n_frames * 640 * 2)
        ue.matmat_mul_core(M=n_frames, K=T, N=640, A_DRAM_ADDR=self._up(pred_aln_trg_T),
                            B_DRAM_ADDR=self._up(_bf16(d).T.contiguous()), OUTPUT_DRAM_ADDR=en_dram)
        self._run(ue)

        # --- predictor.shared LSTM ---
        ue.start_capture()
        shared_out_dram = ue.allocate_tensor_dram(n_frames * 512 * 2)
        self._lstm_bidir(en_dram, n_frames, 640, self.shared_w, shared_out_dram)
        self._run(ue)
        _check("shared LSTM output", shared_out_dram, (n_frames, 512),
               debug_cpu_ref["shared_out"] if debug_cpu_ref else None)

        def run_branch(blocks, proj_w, proj_b, proj_n_pad, branch_name, cpu_refs):
            cur_dram, cur_T, cur_C = shared_out_dram, n_frames, 512
            for i, (blk_w, upsample) in enumerate(zip(blocks, (False, True, False))):
                out_C = 512 if i == 0 else 256
                cur_dram, cur_T = self._adain_res_blk(cur_dram, cur_T, cur_C, out_C, upsample, blk_w, style_dram)
                cur_C = out_C
                if cpu_refs is not None:
                    _check(f"{branch_name} block {i}", cur_dram, (cur_T, cur_C), cpu_refs[i])

            ue.start_capture()
            proj_dram = ue.allocate_tensor_dram(cur_T * proj_n_pad * 2)
            ue.matmat_mul_core(M=cur_T, K=cur_C, N=proj_n_pad, A_DRAM_ADDR=cur_dram, B_DRAM_ADDR=self._up(proj_w),
                                OUTPUT_DRAM_ADDR=proj_dram, C_DRAM_ADDR=self._up(proj_b), bias_mode="broadcast_N")
            self._run(ue)
            if cpu_refs is not None:
                _check(f"{branch_name}_proj", proj_dram, (cur_T, proj_n_pad), cpu_refs[3], cols=1)
            out = ue.dma_from_accelerator_memory(proj_dram, (cur_T, proj_n_pad)).float()[:, 0]
            return out

        f0_refs = debug_cpu_ref["F0_blocks"] + [debug_cpu_ref["F0_pred_pre_squeeze"]] if debug_cpu_ref else None
        n_refs = debug_cpu_ref["N_blocks"] + [debug_cpu_ref["N_pred_pre_squeeze"]] if debug_cpu_ref else None
        F0_pred = run_branch(self.F0_blocks, self.F0_proj_w, self.F0_proj_b, self.proj_n_pad, "F0", f0_refs)
        N_pred = run_branch(self.N_blocks, self.N_proj_w, self.N_proj_b, self.proj_n_pad, "N", n_refs)

        if debug_cpu_ref is not None:
            print("\n[fpga][debug] --- Section 3 bisect (SNR vs CPU) ---")
            for name, snr_db in debug_log:
                flag = "  <-- LOW" if snr_db < 30.0 else ""
                print(f"[fpga][debug] {name:38s} {snr_db:8.2f} dB{flag}")
            print("[fpga][debug] --- end bisect ---\n")

        ue.reset_tensor_dram_addr()
        return F0_pred, N_pred


def run_fpga_forward(model, phonemes: str, ref_s: torch.FloatTensor, speed: float = 1.0, dev: str = "xdma0"):
    """Entry point called from kokoro_test.py --fpga. Only runs the section(s) currently ported to
    hardware (Section 1: PL-BERT) and reports their SNR against the CPU reference, with a bisect
    down to sub-stage granularity within each layer. Deliberately does NOT fall back to running the
    rest of the model on CPU -- --fpga means "run and validate what's actually on the FPGA today",
    not "run the full pipeline with FPGA sprinkled in". Returns None; callers should not expect
    audio until later sections are ported (see the section checklist at the top of this file).
    """
    set_dma_device(dev)
    ue = UnifiedEngine()

    input_ids_list = [model.vocab[p] for p in phonemes if p in model.vocab]
    input_ids = torch.LongTensor([0, *input_ids_list, 0])
    T = input_ids.shape[0]
    assert T + 0 <= model.context_length, (T, model.context_length)

    # Bring-up instrumentation: exact HF module references for the embedding stage and the single
    # shared transformer layer, so PLBertFPGA.forward can print per-stage SNR and pinpoint exactly
    # where a divergence starts, instead of only seeing one aggregate end-to-end number.
    albert = model.bert.albert
    debug_cpu_ref = {
        "embeddings": albert.embeddings,
        "map_in": albert.encoder.embedding_hidden_mapping_in,
        "layer": albert.encoder.albert_layer_groups[0].albert_layers[0],
        # No padding in this debug reference (full, unpadded T), so omitting the mask entirely is
        # mathematically identical to an all-zero additive mask -- and sidesteps a transformers-
        # version-specific internal reshape in AlbertLayer's SDPA path when a pre-extended 4D mask
        # is passed directly to a single layer (bypassing AlbertModel.forward's own mask handling).
        "ext_mask": None,
    }

    print(f"[fpga] Section 1 (PL-BERT): T={T} phoneme tokens ...")
    plbert = PLBertFPGA(model, ue)
    with torch.no_grad():
        d_en = plbert.forward(input_ids, debug_cpu_ref=debug_cpu_ref)  # [512, T]

        input_ids_b = input_ids.unsqueeze(0)
        text_mask = torch.zeros(1, T, dtype=torch.bool)
        bert_dur_cpu = model.bert(input_ids_b, attention_mask=(~text_mask).int())
        d_en_cpu = model.bert_encoder(bert_dur_cpu).transpose(-1, -2).squeeze(0)  # [512, T]
        from user_dma_core import calculate_snr
        snr_db = calculate_snr(d_en_cpu.reshape(-1), d_en.reshape(-1))
        print(f"[fpga] Section 1 (PL-BERT) end-to-end SNR vs CPU: {snr_db:.2f} dB")

        # ---- Section 2: prosody/duration path ----
        # Seeded from d_en_cpu (the CLEAN CPU value), not Section 1's own FPGA output -- this
        # isolates Section 2's own correctness instead of compounding Section 1's ~25dB error into
        # a confusing combined number. Sections get validated independently; wiring them together
        # end-to-end is a later step once each section is individually trusted.
        style_vec = ref_s.reshape(-1)[128:256]
        input_lengths = torch.full((1,), T, dtype=torch.long)
        ref_s_b = ref_s.reshape(1, -1)
        s_cpu = ref_s_b[:, 128:]
        d_cpu = model.predictor.text_encoder(d_en_cpu.unsqueeze(0), s_cpu, input_lengths, text_mask)  # [1,T,640]
        lstm_out_cpu, _ = model.predictor.lstm(d_cpu)  # [1,T,512]
        duration_raw_cpu = model.predictor.duration_proj(lstm_out_cpu)  # [1,T,50]

        print(f"\n[fpga] Section 2 (prosody/duration): T={T} ...")
        prosody = ProsodyDurationFPGA(model, ue)
        debug_cpu_ref2 = {
            "d": d_cpu.squeeze(0), "lstm_out": lstm_out_cpu.squeeze(0), "duration_raw": duration_raw_cpu.squeeze(0),
        }
        d_fpga, pred_dur_fpga = prosody.forward(d_en_cpu, style_vec, T, debug_cpu_ref=debug_cpu_ref2)

        duration_cpu = torch.sigmoid(duration_raw_cpu.squeeze(0)).sum(axis=-1)
        pred_dur_cpu = torch.round(duration_cpu).clamp(min=1).long()
        pred_dur_snr = calculate_snr(pred_dur_cpu.float(), pred_dur_fpga.float())
        print(f"[fpga] Section 2 (prosody/duration) end-to-end SNR (d) vs CPU: "
              f"{calculate_snr(d_cpu.squeeze(0).reshape(-1), d_fpga.reshape(-1)):.2f} dB")
        print(f"[fpga] Section 2 pred_dur SNR vs CPU (post-round, should be near-exact): {pred_dur_snr:.2f} dB")
        n_mismatch = (pred_dur_cpu != pred_dur_fpga).sum().item()
        print(f"[fpga] Section 2 pred_dur exact-match: {T - n_mismatch}/{T} positions "
              f"(mismatches usually a rounding tie at a .5 boundary, not a correctness bug)")

        # ---- Section 3: F0/N prediction ----
        # Seeded from d_cpu/pred_dur_cpu (clean CPU values), same isolation philosophy as Section 2.
        indices_cpu = torch.repeat_interleave(torch.arange(T), pred_dur_cpu)
        pred_aln_trg_cpu = torch.zeros((T, indices_cpu.shape[0]))
        pred_aln_trg_cpu[indices_cpu, torch.arange(indices_cpu.shape[0])] = 1
        pred_aln_trg_cpu = pred_aln_trg_cpu.unsqueeze(0)
        en_cpu = d_cpu.transpose(-1, -2) @ pred_aln_trg_cpu  # [1,640,n_frames]
        shared_lstm_out_cpu, _ = model.predictor.shared(en_cpu.transpose(-1, -2))  # [1,n_frames,512]

        def run_branch_cpu(blocks, proj):
            x = shared_lstm_out_cpu.transpose(-1, -2)  # [1,512,n_frames], channels-first (CPU conv layout)
            block_outs = []
            for blk in blocks:
                x = blk(x, s_cpu)
                block_outs.append(x.squeeze(0).transpose(-1, -2))  # store as [T,C] to match our device layout
            proj_out = proj(x)  # [1,1,n_frames_final]
            return block_outs, proj_out.squeeze(1).squeeze(0)  # [n_frames_final]

        F0_block_outs_cpu, F0_pred_cpu = run_branch_cpu(model.predictor.F0, model.predictor.F0_proj)
        N_block_outs_cpu, N_pred_cpu = run_branch_cpu(model.predictor.N, model.predictor.N_proj)

        n_frames = int(pred_dur_cpu.sum().item())
        print(f"\n[fpga] Section 3 (F0/N prediction): n_frames={n_frames} ...")
        f0n = F0NPredictionFPGA(model, ue)
        debug_cpu_ref3 = {
            "shared_out": shared_lstm_out_cpu.squeeze(0),
            "F0_blocks": F0_block_outs_cpu, "F0_pred_pre_squeeze": F0_pred_cpu,
            "N_blocks": N_block_outs_cpu, "N_pred_pre_squeeze": N_pred_cpu,
        }
        F0_pred_fpga, N_pred_fpga = f0n.forward(d_cpu.squeeze(0), pred_dur_cpu, style_vec, T,
                                                 debug_cpu_ref=debug_cpu_ref3)

        f0_snr = calculate_snr(F0_pred_cpu.reshape(-1), F0_pred_fpga.reshape(-1))
        n_snr = calculate_snr(N_pred_cpu.reshape(-1), N_pred_fpga.reshape(-1))
        print(f"[fpga] Section 3 F0_pred end-to-end SNR vs CPU: {f0_snr:.2f} dB")
        print(f"[fpga] Section 3 N_pred end-to-end SNR vs CPU: {n_snr:.2f} dB")

    print("[fpga] Stopping after Section 3 -- sections 4-5 aren't ported yet, no audio produced.")
    return None
