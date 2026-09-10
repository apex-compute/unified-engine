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
  [x] Section 4: TextEncoder (phoneme embed + 3x weight-norm Conv1d k=5 +
      LayerNorm + LeakyReLU, then BiLSTM) -> t_en. NOT a transformer -- the
      only transformer in Kokoro is Section 1's PL-BERT. Reuses Section 3's
      _conv1d/_leaky_relu/_lstm_bidir directly; the only new mechanics are
      k=5/pad=2 and the channels-last LayerNorm, which is a plain
      layer_norm_core_dram(M=T, N=C) in our [T, C] convention.
  [x] Section 5a: Decoder front (F0_conv/N_conv + encode + 4x decode
      AdainResBlk1d) -> the Generator's input. First section where the CHANNEL
      axis needs 64-alignment (514 -> 576, 1090 -> 1152); padding is applied to
      the conv weights' input-channel axis, not by scrubbing activations.
  [ ] Section 5b-5d: ISTFTNet Generator (SourceModule/SineGen + STFT, the
      ConvTranspose1d ups, dilated AdaINResBlock1 stacks with Snake1D,
      conv_post + iSTFT). Op gaps already proven in user_hw_test.py; the open
      question is SineGen's unbounded phase vs the [-1,1] Taylor fit.

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
import builtins
import time
from contextlib import contextmanager

import torch

from user_dma_core import (UnifiedEngine, UE_MODE, UE_VECTOR_SIZE, set_dma_device,
                           ue_35bit_addr_shifter, INSTRUCTION_SIZE_BYTES)

# ---------------------------------------------------------------------------
# Compile-print suppression + per-section timing
#
# The cores print their tiling decisions (M_chunk/N_chunk, URAM occupancy, FLOP
# counts, capture/DMA sizes) on every call, which buries this module's own
# output. Same treatment every other model in the repo uses: swap builtins.print
# for a gated one (see models/llama3.2_1b/llama3.2_1b_test.py:68-76) and keep an
# unsuppressed `report` for our own lines.
# ---------------------------------------------------------------------------
_original_print = print
_SILENT_MODE = False


def quiet_print(*args, **kwargs):
    """Suppress prints when _SILENT_MODE is True; otherwise print normally."""
    if _SILENT_MODE:
        return
    _original_print(*args, **kwargs)


builtins.print = quiet_print
report = _original_print          # this module's own output, never suppressed
_DEBUG_SNR = [False]              # per-section SNR bisects: opt in with --debug-snr


def report_snr(*args, **kwargs):
    """Accuracy diagnostics. Off by default -- they are bring-up instrumentation, and computing
    the CPU references for them costs real time on top of the printing."""
    if _DEBUG_SNR[0]:
        _original_print(*args, **kwargs)

_STATS: "dict[str, list]" = {}    # name -> [compile_s, exec_s, runs, instructions]
_CUR = [None]
_T_CAPTURE = [None]


def _begin(name):
    """Bill every subsequent _timed_run to `name`, until the next _begin."""
    _STATS.setdefault(name, [0.0, 0.0, 0, 0])
    _CUR[0] = name


@contextmanager
def _section(name):
    prev = _CUR[0]
    _STATS.setdefault(name, [0.0, 0.0, 0, 0])
    _CUR[0] = name
    try:
        yield
    finally:
        _CUR[0] = prev


def _instrument(ue):
    """Timestamp every start_capture so _timed_run can bill the emission phase.
    Wrapping the instance rather than editing ~40 call sites also covers captures
    opened inside helpers."""
    if getattr(ue, "_kokoro_instrumented", False):
        return
    orig = ue.start_capture

    def wrapped(*a, **k):
        _T_CAPTURE[0] = time.perf_counter()
        return orig(*a, **k)

    ue.start_capture = wrapped

    # Permanently reserve GPR_M/GPR_K/GPR_N (see their definition). Never released.
    reserved = [ue.alloc_isa_reg() for _ in range(3)]
    assert reserved == [GPR_M, GPR_K, GPR_N], (
        f"ISA register reservation got {reserved}, expected {[GPR_M, GPR_K, GPR_N]} -- something "
        f"allocated a register before _instrument() ran")
    ue._kokoro_instrumented = True


def _timed_run(ue):
    """Shared by every section class. COMPILE = instruction emission (the Python
    capture pass) plus writing the program to DRAM. HW EXEC = the accelerator
    running it: start_execute_from_dram -> wait_queue."""
    t0 = time.perf_counter()
    ue.stop_capture()
    ue.generate_instruction_halt()
    prog = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(prog)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())
    n_inst = ue.get_capture_instruction_size_bytes() // INSTRUCTION_SIZE_BYTES
    t1 = time.perf_counter()
    ue.start_execute_from_dram(prog)
    ue.wait_queue(30.0)
    t2 = time.perf_counter()
    ue.clear_capture_buffer()
    ue.reset_program_dram_addr()
    if _CUR[0] is not None:
        st = _STATS[_CUR[0]]
        st[0] += (t0 - (_T_CAPTURE[0] or t0)) + (t1 - t0)
        st[1] += (t2 - t1)
        st[2] += 1
        st[3] += n_inst
    _T_CAPTURE[0] = None


def _report_timings():
    if not _STATS:
        return
    report("")
    report(f"{'section':<30}{'compile':>10}{'hw exec':>10}{'runs':>6}{'instructions':>14}")
    report("-" * 70)
    tc = te = 0.0; ti = 0
    for name, (c, e, n, ni) in _STATS.items():
        report(f"{name:<30}{c:>9.3f}s{e:>9.3f}s{n:>6d}{ni:>14,d}")
        tc += c; te += e; ti += ni
    report("-" * 70)
    report(f"{'TOTAL':<30}{tc:>9.3f}s{te:>9.3f}s{'':>6}{ti:>14,d}")



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
        return _dyn_matmul(
            self.ue,
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
        mul = lambda a, b, o: _dyn_eltwise(ue, M, N, a, b, o, mode=UE_MODE.ELTWISE_MUL)
        add = lambda a, b, o: _dyn_eltwise(ue, M, N, a, b, o, mode=UE_MODE.ELTWISE_ADD)
        mulb = lambda a, s, o: _dyn_eltwise(ue, M, N, a, None, o, mode=UE_MODE.MUL_BROADCAST, scalar=s)
        addb = lambda a, s, o: _dyn_eltwise(ue, M, N, a, None, o, mode=UE_MODE.ADD_BROADCAST, scalar=s)

        mul(x_dram, x_dram, tmp1)                       # tmp1 = x^2
        mul(tmp1, x_dram, tmp1)                          # tmp1 = x^3
        mulb(tmp1, self._GELU_NEW_C1, tmp1)               # tmp1 = 0.044715*x^3
        add(x_dram, tmp1, tmp1)                           # tmp1 = x + 0.044715*x^3
        mulb(tmp1, self._GELU_NEW_C0, tmp1)               # tmp1 = u = sqrt(2/pi)*(...)
        mulb(tmp1, 2.0, tmp2)                             # tmp2 = 2u
        _dyn_activation(ue, M=M, N=N, A_DRAM_ADDR=tmp2, OUTPUT_DRAM_ADDR=tmp2,
                            IDENTITY_DRAM_ADDR=identity_dram, activation="sigmoid")
        mulb(tmp2, 2.0, tmp2)                             # tmp2 = 2*sigmoid(2u)
        addb(tmp2, -1.0, tmp2)                            # tmp2 = tanh(u) = 2*sigmoid(2u) - 1
        mul(x_dram, tmp2, tmp3)                           # tmp3 = x*tanh(u)
        add(x_dram, tmp3, tmp3)                           # tmp3 = x + x*tanh(u) = x*(1+tanh(u))
        mulb(tmp3, 0.5, out_dram)                         # out = 0.5*x*(1+tanh(u))

    def _run(self, ue):
        _timed_run(ue)

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
        _dyn_eltwise(ue, T_pad, E, word_dram, pos_dram, emb_sum_dram, mode=UE_MODE.ELTWISE_ADD)
        _dyn_eltwise(ue, T_pad, E, emb_sum_dram, type_dram, emb_sum_dram, mode=UE_MODE.ELTWISE_ADD)
        _dyn_layernorm(ue, M=T_pad, N=E, A_DRAM_ADDR=emb_sum_dram, OUTPUT_DRAM_ADDR=emb_ln_dram,
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
            _dyn_eltwise(ue, T_pad, H, hidden_dram, attn_proj_dram, resid1_dram, mode=UE_MODE.ELTWISE_ADD)
            _dyn_layernorm(ue, M=T_pad, N=H, A_DRAM_ADDR=resid1_dram, OUTPUT_DRAM_ADDR=attn_ln_dram,
                                     GAMMA_DRAM_ADDR=w_attn_ln, BETA_DRAM_ADDR=b_attn_ln)

            self._linear(attn_ln_dram, T_pad, H, FFN, w_ffn, b_ffn, ffn_mid_dram, gelu=False)
            self._gelu_new(ffn_mid_dram, T_pad * FFN, identity_dram,
                            gelu_tmp1_dram, gelu_tmp2_dram, gelu_tmp3_dram, ffn_mid_dram)
            self._linear(ffn_mid_dram, T_pad, FFN, H, w_ffn_out, b_ffn_out, ffn_out_dram)
            _dyn_eltwise(ue, T_pad, H, attn_ln_dram, ffn_out_dram, resid2_dram, mode=UE_MODE.ELTWISE_ADD)
            _dyn_layernorm(ue, M=T_pad, N=H, A_DRAM_ADDR=resid2_dram, OUTPUT_DRAM_ADDR=hidden_dram,
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
            report_snr("\n[fpga][debug] --- Section 1 bisect (SNR vs CPU, most-recent layer last) ---")
            for stage_name, snr_db in debug_log:
                flag = "  <-- LOW" if snr_db < 30.0 else ""
                report_snr(f"[fpga][debug] {stage_name:38s} {snr_db:8.2f} dB{flag}")
            report_snr("[fpga][debug] --- end bisect ---\n")

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
        _timed_run(ue)

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
                _dyn_activation(ue, M=M_H, N=UE_VECTOR_SIZE, A_DRAM_ADDR=src, OUTPUT_DRAM_ADDR=dst,
                                    IDENTITY_DRAM_ADDR=self.identity_dram, activation="sigmoid")

            def tanh_via_sigmoid(src, tmp, dst):
                _dyn_eltwise(ue, 1, H, src, None, tmp, mode=UE_MODE.MUL_BROADCAST, scalar=2.0)
                sigmoid_ip(tmp, tmp)
                _dyn_eltwise(ue, 1, H, tmp, None, tmp, mode=UE_MODE.MUL_BROADCAST, scalar=2.0)
                _dyn_eltwise(ue, 1, H, tmp, None, dst, mode=UE_MODE.ADD_BROADCAST, scalar=-1.0)

            time_range = range(T - 1, -1, -1) if reverse else range(T)
            h_prev, c_prev = h_bufs[0], c_bufs[0]
            for step, t in enumerate(time_range):
                x_t = x_dram + t * Cin * 2
                h_cur, c_cur = h_bufs[(step + 1) % 2], c_bufs[(step + 1) % 2]

                _dyn_matmul(ue, M=1, K=Cin, N=G, A_DRAM_ADDR=x_t, B_DRAM_ADDR=wx_dram,
                                    OUTPUT_DRAM_ADDR=gates_x, C_DRAM_ADDR=b_dram, bias_mode="broadcast_N")
                _dyn_matmul(ue, M=1, K=H, N=G, A_DRAM_ADDR=h_prev, B_DRAM_ADDR=wh_dram, OUTPUT_DRAM_ADDR=gates_h)
                _dyn_eltwise(ue, 1, G, gates_x, gates_h, gates, mode=UE_MODE.ELTWISE_ADD)

                i_raw, f_raw, g_raw, o_raw = (gates + k * H * 2 for k in range(4))
                sigmoid_ip(i_raw, i_d); sigmoid_ip(f_raw, f_d); sigmoid_ip(o_raw, o_d)
                tanh_via_sigmoid(g_raw, g_tmp, g_d)

                _dyn_eltwise(ue, 1, H, f_d, c_prev, fc_, mode=UE_MODE.ELTWISE_MUL)
                _dyn_eltwise(ue, 1, H, i_d, g_d, ig_, mode=UE_MODE.ELTWISE_MUL)
                _dyn_eltwise(ue, 1, H, fc_, ig_, c_cur, mode=UE_MODE.ELTWISE_ADD)

                tanh_via_sigmoid(c_cur, c_tmp, tanh_c)
                _dyn_eltwise(ue, 1, H, o_d, tanh_c, h_cur, mode=UE_MODE.ELTWISE_MUL)
                out_addr = out_dram + t * (2 * H) * 2 + col_off_bytes
                _dyn_eltwise(ue, 1, H, o_d, tanh_c, out_addr, mode=UE_MODE.ELTWISE_MUL)

                h_prev, c_prev = h_cur, c_cur

        run_direction(False, w["Wx_f"], w["Wh_f"], w["b_f"], 0)
        run_direction(True, w["Wx_b"], w["Wh_b"], w["b_b"], H * 2)

    def _concat_columns(self, a_dram, a_width, b_dram, b_width, T, out_dram, gpr_rows=None):
        """out[T, a_width+b_width] = concat([a[T,a_width], b[T,b_width]], dim=-1), via per-row
        copies (T is small here, <200 typically -- same technique as Section 1's embedding gather).
        """
        ue = self.ue
        out_w_bytes = (a_width + b_width) * 2
        a_row_bytes, b_row_bytes = a_width * 2, b_width * 2
        # Both halves staged in the same iteration, at distinct SRAM offsets so one loop body
        # covers the read-read-write-write sequence the unrolled version did per row.
        _pbi_row_loop(ue, T,
                      reads=[(a_dram, a_row_bytes, a_width, 0),
                             (b_dram, b_row_bytes, b_width, a_row_bytes)],
                      writes=[(out_dram, out_w_bytes, a_width, 0),
                              (out_dram + a_row_bytes, out_w_bytes, b_width, a_row_bytes)],
                      gpr_rows=gpr_rows)

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
        _dyn_matmul(ue, M=1, K=self.style_dim, N=2 * C, A_DRAM_ADDR=style_dram, B_DRAM_ADDR=self._up(w["fc_w"]),
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
        _dyn_layernorm(ue, M=T, N=C, A_DRAM_ADDR=x_dram, OUTPUT_DRAM_ADDR=normed_dram)
        _dyn_eltwise(ue, T, C, normed_dram, gamma_tiled, ng_dram, mode=UE_MODE.ELTWISE_MUL)   # normed*gamma
        _dyn_eltwise(ue, T, C, normed_dram, ng_dram, tmp_dram, mode=UE_MODE.ELTWISE_ADD)        # normed*(1+gamma)
        _dyn_eltwise(ue, T, C, tmp_dram, beta_tiled, out_dram, mode=UE_MODE.ELTWISE_ADD)         # + beta
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
        _dyn_matmul(ue, M=T, K=C, N=self.dur_n_pad, A_DRAM_ADDR=pred_lstm_out_dram, B_DRAM_ADDR=self._up(self.dur_w),
                            OUTPUT_DRAM_ADDR=dur_dram, C_DRAM_ADDR=self._up(self.dur_b), bias_mode="broadcast_N")
        self._run(ue)
        _check("duration_proj (pre-sigmoid)", dur_dram, (T, self.dur_n_pad),
                debug_cpu_ref["duration_raw"] if debug_cpu_ref else None, cols=self.dur_n)

        if debug_cpu_ref is not None:
            report_snr("\n[fpga][debug] --- Section 2 bisect (SNR vs CPU) ---")
            for name, snr_db in debug_log:
                flag = "  <-- LOW" if snr_db < 30.0 else ""
                report_snr(f"[fpga][debug] {name:38s} {snr_db:8.2f} dB{flag}")
            report_snr("[fpga][debug] --- end bisect ---\n")

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
        _timed_run(ue)

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
                _dyn_activation(ue, M=M_H, N=UE_VECTOR_SIZE, A_DRAM_ADDR=src, OUTPUT_DRAM_ADDR=dst,
                                    IDENTITY_DRAM_ADDR=self.identity_dram, activation="sigmoid")

            def tanh_via_sigmoid(src, tmp, dst):
                _dyn_eltwise(ue, 1, H, src, None, tmp, mode=UE_MODE.MUL_BROADCAST, scalar=2.0)
                sigmoid_ip(tmp, tmp)
                _dyn_eltwise(ue, 1, H, tmp, None, tmp, mode=UE_MODE.MUL_BROADCAST, scalar=2.0)
                _dyn_eltwise(ue, 1, H, tmp, None, dst, mode=UE_MODE.ADD_BROADCAST, scalar=-1.0)

            time_range = range(T - 1, -1, -1) if reverse else range(T)
            h_prev, c_prev = h_bufs[0], c_bufs[0]
            for step, t in enumerate(time_range):
                x_t = x_dram + t * Cin * 2
                h_cur, c_cur = h_bufs[(step + 1) % 2], c_bufs[(step + 1) % 2]
                _dyn_matmul(ue, M=1, K=Cin, N=G, A_DRAM_ADDR=x_t, B_DRAM_ADDR=wx_dram,
                                    OUTPUT_DRAM_ADDR=gates_x, C_DRAM_ADDR=b_dram, bias_mode="broadcast_N")
                _dyn_matmul(ue, M=1, K=H, N=G, A_DRAM_ADDR=h_prev, B_DRAM_ADDR=wh_dram, OUTPUT_DRAM_ADDR=gates_h)
                _dyn_eltwise(ue, 1, G, gates_x, gates_h, gates, mode=UE_MODE.ELTWISE_ADD)
                i_raw, f_raw, g_raw, o_raw = (gates + k * H * 2 for k in range(4))
                sigmoid_ip(i_raw, i_d); sigmoid_ip(f_raw, f_d); sigmoid_ip(o_raw, o_d)
                tanh_via_sigmoid(g_raw, g_tmp, g_d)
                _dyn_eltwise(ue, 1, H, f_d, c_prev, fc_, mode=UE_MODE.ELTWISE_MUL)
                _dyn_eltwise(ue, 1, H, i_d, g_d, ig_, mode=UE_MODE.ELTWISE_MUL)
                _dyn_eltwise(ue, 1, H, fc_, ig_, c_cur, mode=UE_MODE.ELTWISE_ADD)
                tanh_via_sigmoid(c_cur, c_tmp, tanh_c)
                _dyn_eltwise(ue, 1, H, o_d, tanh_c, h_cur, mode=UE_MODE.ELTWISE_MUL)
                out_addr = out_dram + t * (2 * H) * 2 + col_off_bytes
                _dyn_eltwise(ue, 1, H, o_d, tanh_c, out_addr, mode=UE_MODE.ELTWISE_MUL)
                h_prev, c_prev = h_cur, c_cur

        run_direction(False, w["Wx_f"], w["Wh_f"], w["b_f"], 0)
        run_direction(True, w["Wx_b"], w["Wh_b"], w["b_b"], H * 2)

    def _device_row_copy(self, src_dram, dst_dram, num_rows, row_width, gpr_rows=None):
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
        _pbi_row_loop(ue, num_rows,
                      reads=[(src_dram, row_bytes, row_width, 0)],
                      writes=[(dst_dram, row_bytes, row_width, 0)],
                      gpr_rows=gpr_rows)

    def _conv1d(self, x_dram, T, Cin, Cout, W, b, out_dram, kernel_size, pad, real_T=None):
        """Conv1d via K shifted-matmul-accumulate (conv1d_shifted_matmul_test's validated pattern).
        W: [Cout, Cin, kernel_size] bf16. kernel_size=1,pad=0 degenerates to a single plain matmul
        (used for the pointwise conv1x1/F0_proj/N_proj cases).

        ``real_T`` is the UNPADDED frame count when ``T`` has been rounded up to a multiple of 64
        for SRAM alignment. It matters: this conv's zero-padding must sit at the REAL sequence
        boundary, exactly where CPU Conv1d puts it. Rows real_T..T hold whatever the previous
        AdaIN left there (layer_norm of a zero row is not zero, and the preceding conv's bias
        lands there too), so without this the last real frame gets convolved against garbage
        instead of against zero -- one badly-wrong row in real_T, i.e. an SNR ceiling of
        10*log10(real_T) ~= 24 dB at 253 frames. Every other row is unaffected, which is why the
        symptom looks like a single low block that the next InstanceNorm partly renormalizes away.
        """
        ue = self.ue
        if kernel_size == 1:
            _dyn_matmul(ue, M=T, K=Cin, N=Cout, A_DRAM_ADDR=x_dram, B_DRAM_ADDR=self._up(W.squeeze(-1)),
                                OUTPUT_DRAM_ADDR=out_dram,
                                C_DRAM_ADDR=(self._up(b) if b is not None else None),
                                bias_mode="broadcast_N")
            return
        x_pad_dram = ue.allocate_tensor_dram((T + 2 * pad) * Cin * 2)
        ue.dma_to_accelerator_memory(
            x_pad_dram, torch.zeros((T + 2 * pad) * Cin, dtype=torch.bfloat16))  # constant zero-fill, safe direct DMA
        self._device_row_copy(x_dram, x_pad_dram + pad * Cin * 2, T, Cin)  # on-device, NOT a host round-trip
        if real_T is not None and real_T < T:
            # Re-zero the alignment padding so the conv's boundary zero lands at the real end of
            # the sequence.
            #
            # This MUST be an on-device (captured) copy, not a dma_to_accelerator_memory. Host DMA
            # runs IMMEDIATELY at emit time, while _device_row_copy above is a CAPTURED instruction
            # that runs later during _run() -- so a host zero-fill here would be overwritten by that
            # copy when the program actually executes, and the change would be a silent no-op.
            # (The memset above survives only at rows 0 and T+1, which the copy never touches.)
            n_pad_rows = T - real_T
            zeros_dram = ue.allocate_tensor_dram(n_pad_rows * Cin * 2)
            ue.dma_to_accelerator_memory(zeros_dram, torch.zeros(n_pad_rows * Cin, dtype=torch.bfloat16))
            self._device_row_copy(zeros_dram, x_pad_dram + (pad + real_T) * Cin * 2, n_pad_rows, Cin)
        w_tap_dram = [self._up(W[:, :, k].contiguous()) for k in range(kernel_size)]
        acc_a = ue.allocate_tensor_dram(T * Cout * 2)
        acc_b = ue.allocate_tensor_dram(T * Cout * 2)
        for k in range(kernel_size):
            A_DRAM_ADDR = x_pad_dram + k * Cin * 2
            prev_acc = acc_a if k % 2 == 0 else acc_b
            cur_acc = acc_b if k % 2 == 0 else acc_a
            bias_addr = self._up(b) if (b is not None and k == 0) else None
            _dyn_matmul(ue, 
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
        _dyn_activation(ue, M=M, N=N, A_DRAM_ADDR=x_dram, OUTPUT_DRAM_ADDR=relu_dram,
                            IDENTITY_DRAM_ADDR=self.identity_dram, activation="clamp")
        _dyn_eltwise(ue, M, N, x_dram, None, neg_dram, mode=UE_MODE.MUL_BROADCAST, scalar=-1.0)
        _dyn_activation(ue, M=M, N=N, A_DRAM_ADDR=neg_dram, OUTPUT_DRAM_ADDR=relu_neg_dram,
                            IDENTITY_DRAM_ADDR=self.identity_dram, activation="clamp")
        _dyn_eltwise(ue, M, N, relu_neg_dram, None, relu_neg_dram, mode=UE_MODE.MUL_BROADCAST, scalar=0.2)
        _dyn_eltwise(ue, M, N, relu_dram, relu_neg_dram, out_dram, mode=UE_MODE.ELTWISE_SUB)

    def _nearest_upsample2x(self, x_dram, T, C, out_dram, gpr_rows=None):
        """Exact row-duplicate: out[2t]=out[2t+1]=x[t]. Bit-exact (interpolate_fixed_matmul_test's
        nearest mode measured inf dB), implemented as a per-row copy loop (same technique as
        Section 1's embedding gather)."""
        ue = self.ue
        row_bytes = C * 2
        # out[2t] = out[2t+1] = x[t]: one read, two writes whose destination stride is 2*row_bytes,
        # the second offset one row further in. Still bit-exact -- it is a pure row duplicate.
        _pbi_row_loop(ue, T,
                      reads=[(x_dram, row_bytes, C, 0)],
                      writes=[(out_dram, 2 * row_bytes, C, 0),
                              (out_dram + row_bytes, 2 * row_bytes, C, 0)],
                      gpr_rows=gpr_rows)

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
            _dyn_eltwise(ue, T_out, C, shifted_x, w_tap_dram[k], tmp[k], mode=UE_MODE.ELTWISE_MUL)
            if k == 0:
                running = tmp[0]
                continue
            cur = acc_a if k % 2 == 1 else acc_b
            _dyn_eltwise(ue, T_out, C, tmp[k], running, cur, mode=UE_MODE.ELTWISE_ADD)
            running = cur
        b_tiled_dram = self._up(b.unsqueeze(0).expand(T_out, C).contiguous())
        _dyn_eltwise(ue, T_out, C, running, b_tiled_dram, out_dram, mode=UE_MODE.ELTWISE_ADD)

    def _adain1d(self, x_dram, T, C, style_dram, fc_w, fc_b, out_dram, real_T=None):
        """AdaIN1d: InstanceNorm1d (no affine) + style-conditioned per-channel affine. SELF-CONTAINED
        capture/execute (needs its own host round-trip for gamma/beta, same reasoning as Section 2's
        AdaLayerNorm) -- see instance_norm1d_via_layernorm_test: InstanceNorm1d == layer_norm_core_
        dram on a [C,T] (channels-as-rows) view, no permute needed once there, just the OPPOSITE of
        this class's usual [T,C] convention -- so we transpose in, normalize, transpose back out.
        """
        ue = self.ue
        gb_dram = ue.allocate_tensor_dram(2 * C * 2)
        ue.start_capture()
        pad_k = 1.0
        if real_T is not None and real_T < T:
            # EXACT handling of the TIME alignment padding. This norm is an InstanceNorm -- it
            # takes mean/variance per channel ACROSS TIME -- so the padded frames are samples in
            # that statistic and their value matters. Zeros are catastrophic for a channel with a
            # large nonzero mean (the decoder's F0 sits near 200 Hz). Tiling real frames is far
            # better but still biased, and the bias scales with the padding FRACTION: n_frames=253
            # -> 256 is 1.2% padding and cost ~1 dB, but n_frames=56 -> 64 is 12.5% and cost ~20
            # dB. That made output quality depend on `n_frames mod 64`, which is intolerable.
            #
            # Padding at the per-channel MEAN is exact rather than approximate: those samples sit
            # exactly at the mean, so they shift it not at all and add nothing to the sum of
            # squares -- the variance is scaled by precisely real_T/T, a known constant, undone by
            # pad_k below. Verified at ~140 dB (float64 noise floor) on host.
            #
            # Cheap because the padding is whole ROWS here and the mean is a [1, C] row: a row
            # broadcast (src stride 0 re-reads the same row every iteration), not the strided
            # per-channel scatter it would be in the [C, T] view.
            row_b = C * 2
            half_a = ue.allocate_tensor_dram(((T + 1) // 2) * row_b)
            half_b = ue.allocate_tensor_dram(((T + 1) // 2) * row_b)
            sum_dram = _col_sum_rows(ue, x_dram, real_T, C, half_a, half_b)
            mean_dram = ue.allocate_tensor_dram(row_b)
            _dyn_eltwise(ue, 1, C, sum_dram, None, mean_dram,
                                 mode=UE_MODE.MUL_BROADCAST, scalar=1.0 / real_T)
            _pbi_row_loop(ue, T - real_T,
                          reads=[(mean_dram, 0, C, 0)],
                          writes=[(x_dram + real_T * row_b, row_b, C, 0)])
            pad_k = (real_T / T) ** 0.5
        _dyn_matmul(ue, M=1, K=self.style_dim, N=2 * C, A_DRAM_ADDR=style_dram, B_DRAM_ADDR=self._up(fc_w),
                            OUTPUT_DRAM_ADDR=gb_dram, C_DRAM_ADDR=self._up(fc_b), bias_mode="broadcast_N")
        x_ct_dram = ue.allocate_tensor_dram(T * C * 2)
        _dyn_transpose(ue, M=T, N=C, INPUT_DRAM_ADDR=x_dram, OUTPUT_DRAM_ADDR=x_ct_dram,
                                IDENTITY_DRAM_ADDR=self.identity_dram)
        self._run(ue)

        gb = ue.dma_from_accelerator_memory(gb_dram, (2 * C,))
        gamma, beta = gb[:C], gb[C:]
        if pad_k != 1.0:
            # The device normalized over T frames though only real_T carry signal, so its output
            # is normed_real * sqrt(T/real_T). AdaIN computes normed*(1+gamma)+beta, so the
            # correction folds in as gamma' = pad_k*(1+gamma) - 1. Free: gb is already host-side.
            gamma = ((1.0 + gamma.float()) * pad_k - 1.0).to(gamma.dtype)
        gamma_tiled = self._up(gamma.unsqueeze(1).expand(C, T).contiguous())
        beta_tiled = self._up(beta.unsqueeze(1).expand(C, T).contiguous())

        normed_ct = ue.allocate_tensor_dram(C * T * 2)
        ng_ct = ue.allocate_tensor_dram(C * T * 2)
        tmp_ct = ue.allocate_tensor_dram(C * T * 2)
        out_ct = ue.allocate_tensor_dram(C * T * 2)
        ue.start_capture()
        _dyn_layernorm(ue, M=C, N=T, A_DRAM_ADDR=x_ct_dram, OUTPUT_DRAM_ADDR=normed_ct)
        _dyn_eltwise(ue, C, T, normed_ct, gamma_tiled, ng_ct, mode=UE_MODE.ELTWISE_MUL)
        _dyn_eltwise(ue, C, T, normed_ct, ng_ct, tmp_ct, mode=UE_MODE.ELTWISE_ADD)
        _dyn_eltwise(ue, C, T, tmp_ct, beta_tiled, out_ct, mode=UE_MODE.ELTWISE_ADD)
        _dyn_transpose(ue, M=C, N=T, INPUT_DRAM_ADDR=out_ct, OUTPUT_DRAM_ADDR=out_dram,
                                IDENTITY_DRAM_ADDR=self.identity_dram)
        self._run(ue)

    def _adain_res_blk(self, x_dram, T, Cin, Cout, upsample, w, style_dram, real_T=None):
        """One AdainResBlk1d. ``T`` is the 64-aligned row count the device operates on; ``real_T``
        is the unpadded frame count, forwarded to _conv1d so its zero-padding lands at the real
        sequence boundary. Returns (out_dram, T_out)."""
        ue = self.ue
        T_out = 2 * T if upsample else T
        real_T_out = None if real_T is None else (2 * real_T if upsample else real_T)
        learned_sc = Cin != Cout

        h1_dram = ue.allocate_tensor_dram(T * Cin * 2)
        self._adain1d(x_dram, T, Cin, style_dram, w["norm1_fc_w"], w["norm1_fc_b"], h1_dram,
                      real_T=real_T)

        ue.start_capture()
        act1_dram = ue.allocate_tensor_dram(T * Cin * 2)
        self._leaky_relu(h1_dram, T, Cin, act1_dram)
        if upsample:
            pooled_dram = ue.allocate_tensor_dram(T_out * Cin * 2)
            self._depthwise_convtranspose_upsample2x(act1_dram, T, Cin, w["pool_w"], w["pool_b"], pooled_dram)
        else:
            pooled_dram = act1_dram
        conv1_out_dram = ue.allocate_tensor_dram(T_out * Cout * 2)
        self._conv1d(pooled_dram, T_out, Cin, Cout, w["conv1_w"], w["conv1_b"], conv1_out_dram, kernel_size=3, pad=1,
                     real_T=real_T_out)
        self._run(ue)

        h2_dram = ue.allocate_tensor_dram(T_out * Cout * 2)
        self._adain1d(conv1_out_dram, T_out, Cout, style_dram, w["norm2_fc_w"], w["norm2_fc_b"], h2_dram,
                      real_T=real_T_out)

        ue.start_capture()
        act2_dram = ue.allocate_tensor_dram(T_out * Cout * 2)
        self._leaky_relu(h2_dram, T_out, Cout, act2_dram)
        residual_dram = ue.allocate_tensor_dram(T_out * Cout * 2)
        self._conv1d(act2_dram, T_out, Cout, Cout, w["conv2_w"], w["conv2_b"], residual_dram, kernel_size=3, pad=1,
                     real_T=real_T_out)

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
        _dyn_eltwise(ue, T_out, Cout, residual_dram, sc_dram, sum_dram, mode=UE_MODE.ELTWISE_ADD)
        _dyn_eltwise(ue, T_out, Cout, sum_dram, None, out_dram, mode=UE_MODE.MUL_BROADCAST, scalar=0.7071067811865476)
        self._run(ue)
        return out_dram, T_out

    def forward(self, d: torch.Tensor, pred_dur: torch.LongTensor, style_vec: torch.Tensor, T: int,
                debug_cpu_ref=None):
        """d: [T,640] (Section 2's output). pred_dur: [T] LongTensor (Section 2's output, exact).
        style_vec: [128]. Returns (F0_pred [n_frames], N_pred [n_frames]) FloatTensors.
        """
        ue = self.ue
        debug_log = []

        def _check(name, dram_addr, shape, cpu_ref, cols=None, rows=None):
            """`shape` is the buffer's ACTUAL (frame-padded) layout in DRAM -- always read that,
            then slice to the `rows` REAL frames before comparing, since the CPU reference is built
            at the unpadded frame count. Same rule as Section 2's `cols` for N-padded buffers."""
            if debug_cpu_ref is None:
                return
            got = ue.dma_from_accelerator_memory(dram_addr, shape).float()
            if rows is not None:
                got = got[:rows]
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

        # T is the CONTRACTION dim of the `en` matmul below, and matmat_mul_core requires
        # K % UE_VECTOR_SIZE == 0 (user_dma_core.py:2814) -- the SRAM write-back offset is
        # m_take*K*bpe, which must land on a 128B boundary (user_dma_core.py:1685). A raw
        # T=95 gives 255*95*2 = 48450, i.e. 66 mod 128, and asserts. Pad K to T_pad on BOTH
        # operands; this is exact, not approximate: the padded rows of the one-hot
        # pred_aln_trg_T are all-zero and the padded columns of d^T are never selected, so
        # they contribute nothing to the contraction. Same treatment Section 1 gives T
        # (see PLBertFPGA.forward's T_pad).
        T_pad = _round_up(T, UE_VECTOR_SIZE)
        # The FRAME COUNT is itself a 64-constrained dimension downstream: _adain1d puts it on
        # bf16_transpose_core's M, and on layer_norm_core_dram's / eltwise_core_dram's N. Every
        # SRAM row is 64 bf16 elements (128B), so any offset computed as rows*dim*2 needs dim%64==0.
        # Pad it ONCE here: the only length transform in Section 3 is the x2 upsample, so every
        # downstream T_out stays a multiple of 64 for free.
        nf_pad = _round_up(n_frames, UE_VECTOR_SIZE)

        pred_aln_trg_T = torch.zeros(nf_pad, T_pad, dtype=torch.bfloat16)
        pred_aln_trg_T[torch.arange(n_frames), frame_idx] = 1.0   # rows >= n_frames stay all-zero
        d_T = torch.zeros(640, T_pad, dtype=torch.bfloat16)
        d_T[:, :T] = _bf16(d).T

        ue.start_capture()
        en_dram = ue.allocate_tensor_dram(nf_pad * 640 * 2)
        _dyn_matmul(ue, M=nf_pad, K=T_pad, N=640, A_DRAM_ADDR=self._up(pred_aln_trg_T),
                            B_DRAM_ADDR=self._up(d_T.contiguous()), OUTPUT_DRAM_ADDR=en_dram)
        self._run(ue)

        # --- predictor.shared LSTM ---
        shared_out_dram = ue.allocate_tensor_dram(nf_pad * 512 * 2)
        # Zero the whole buffer up front: the LSTM below writes only the first n_frames rows (it is
        # a SEQUENTIAL bidirectional pass -- running it over the padding would have the backward
        # direction start inside the pad and corrupt every real timestep), so rows n_frames..nf_pad
        # would otherwise be stale DRAM rather than a defined value.
        ue.dma_to_accelerator_memory(shared_out_dram, torch.zeros(nf_pad * 512, dtype=torch.bfloat16))
        ue.start_capture()
        self._lstm_bidir(en_dram, n_frames, 640, self.shared_w, shared_out_dram)
        self._run(ue)
        _check("shared LSTM output", shared_out_dram, (nf_pad, 512),
               debug_cpu_ref["shared_out"] if debug_cpu_ref else None, rows=n_frames)

        def run_branch(blocks, proj_w, proj_b, proj_n_pad, branch_name, cpu_refs):
            # cur_T is the PADDED row count the device actually operates on; real_T tracks the
            # unpadded frame count, which is what the CPU references are sized at. Both double
            # on the upsample block, so cur_T stays a multiple of 64 throughout.
            cur_dram, cur_T, cur_C = shared_out_dram, nf_pad, 512
            real_T = n_frames
            for i, (blk_w, upsample) in enumerate(zip(blocks, (False, True, False))):
                out_C = 512 if i == 0 else 256
                cur_dram, cur_T = self._adain_res_blk(cur_dram, cur_T, cur_C, out_C, upsample, blk_w,
                                                     style_dram, real_T=real_T)
                real_T = 2 * real_T if upsample else real_T
                cur_C = out_C
                if cpu_refs is not None:
                    _check(f"{branch_name} block {i}", cur_dram, (cur_T, cur_C), cpu_refs[i], rows=real_T)

            ue.start_capture()
            proj_dram = ue.allocate_tensor_dram(cur_T * proj_n_pad * 2)
            _dyn_matmul(ue, M=cur_T, K=cur_C, N=proj_n_pad, A_DRAM_ADDR=cur_dram, B_DRAM_ADDR=self._up(proj_w),
                                OUTPUT_DRAM_ADDR=proj_dram, C_DRAM_ADDR=self._up(proj_b), bias_mode="broadcast_N")
            self._run(ue)
            if cpu_refs is not None:
                _check(f"{branch_name}_proj", proj_dram, (cur_T, proj_n_pad), cpu_refs[3], cols=1, rows=real_T)
            out = ue.dma_from_accelerator_memory(proj_dram, (cur_T, proj_n_pad)).float()[:real_T, 0]
            return out

        f0_refs = debug_cpu_ref["F0_blocks"] + [debug_cpu_ref["F0_pred_pre_squeeze"]] if debug_cpu_ref else None
        n_refs = debug_cpu_ref["N_blocks"] + [debug_cpu_ref["N_pred_pre_squeeze"]] if debug_cpu_ref else None
        F0_pred = run_branch(self.F0_blocks, self.F0_proj_w, self.F0_proj_b, self.proj_n_pad, "F0", f0_refs)
        N_pred = run_branch(self.N_blocks, self.N_proj_w, self.N_proj_b, self.proj_n_pad, "N", n_refs)

        if debug_cpu_ref is not None:
            report_snr("\n[fpga][debug] --- Section 3 bisect (SNR vs CPU) ---")
            for name, snr_db in debug_log:
                flag = "  <-- LOW" if snr_db < 30.0 else ""
                report_snr(f"[fpga][debug] {name:38s} {snr_db:8.2f} dB{flag}")
            report_snr("[fpga][debug] --- end bisect ---\n")

        ue.reset_tensor_dram_addr()
        return F0_pred, N_pred


class TextEncoderFPGA:
    """Section 4: TextEncoder (phoneme embedding + Conv1d stack + BiLSTM) -> t_en [C, T].

    This is NOT a transformer -- Kokoro's only transformer is Section 1's PL-BERT. This is
    StyleTTS2's text encoder (kokoro_test.py:76-110):

        embedding(n_token, C) -> [T, C]
        3 x ( weight_norm Conv1d(C, C, kernel_size=5, padding=2) -> LayerNorm(C) -> LeakyReLU(0.2) )
        BiLSTM(C -> C/2 per direction, concat -> C)

    Two layout notes that make this cheaper than it looks:

    * The CPU module works channels-FIRST ([B, C, T]) and its `LayerNorm` (kokoro_test.py:62-73)
      transposes to channels-LAST, normalizes over C, and transposes back. Our convention is
      already [T, C], so that is a plain `layer_norm_core_dram(M=T, N=C)` -- no transpose at all,
      unlike Section 3's AdaIN which genuinely needed the [C, T] view for InstanceNorm.
    * `nn.Dropout(0.2)` is a no-op in eval.

    The CPU forward also `masked_fill_`s padded positions to 0 after every conv. With a single
    unbatched utterance the CPU mask is all-False, so there is nothing to mask there -- but OUR
    T is padded to a multiple of 64 for SRAM alignment, and those padded rows must be zero at the
    conv boundary. That is exactly what `_conv1d`'s `real_T` does, so it is threaded through every
    conv here. See [[hw-sram-alignment-rules]] reasoning in _conv1d's docstring.
    """

    # Borrowed wholesale from Section 3 rather than duplicated a third time -- these are plain
    # functions of (self.ue, self.H, self._up, self.identity_dram), all of which this class
    # provides. Section 2/3 keep their own copies; nothing there changes.
    _lstm_bidir = F0NPredictionFPGA._lstm_bidir
    _conv1d = F0NPredictionFPGA._conv1d
    _leaky_relu = F0NPredictionFPGA._leaky_relu
    _device_row_copy = F0NPredictionFPGA._device_row_copy
    _up = F0NPredictionFPGA._up
    _run = F0NPredictionFPGA._run

    def __init__(self, model, ue: UnifiedEngine):
        self.model = model
        self.ue = ue
        te = model.text_encoder

        self.C = te.embedding.embedding_dim            # 512
        self.H = self.C // 2                           # 256 per LSTM direction
        self.kernel_size = te.cnn[0][0].kernel_size[0] # 5
        self.pad = (self.kernel_size - 1) // 2         # 2
        self.depth = len(te.cnn)                       # 3

        self.emb_w = _bf16(te.embedding.weight)        # [n_token, C]
        self.cnn_blocks = []
        for blk in te.cnn:
            conv, ln = blk[0], blk[1]
            w, b = _wn_conv(conv)                      # materialize weight_norm (g, v) -> w
            self.cnn_blocks.append(dict(
                conv_w=w, conv_b=b,
                ln_g=_bf16(ln.gamma), ln_b=_bf16(ln.beta),
            ))
        self.lstm_w = dict(
            Wx_f=_bf16(te.lstm.weight_ih_l0), Wh_f=_bf16(te.lstm.weight_hh_l0),
            b_f=_bf16(te.lstm.bias_ih_l0 + te.lstm.bias_hh_l0),
            Wx_b=_bf16(te.lstm.weight_ih_l0_reverse), Wh_b=_bf16(te.lstm.weight_hh_l0_reverse),
            b_b=_bf16(te.lstm.bias_ih_l0_reverse + te.lstm.bias_hh_l0_reverse),
        )

    def forward(self, input_ids: torch.Tensor, T: int, debug_cpu_ref=None):
        """input_ids: [T] LongTensor of phoneme ids. Returns t_en [C, T] (channels-first, matching
        KokoroModel.forward_with_tokens's `t_en`, which Section 5 consumes as `t_en @ pred_aln_trg`).
        """
        ue = self.ue
        C, T_pad = self.C, _round_up(T, UE_VECTOR_SIZE)
        row_bytes = C * 2
        debug_log = []

        def _check(name, dram_addr, shape, cpu_ref, rows=None):
            from user_dma_core import calculate_snr
            if debug_cpu_ref is None or cpu_ref is None:
                return
            got = ue.dma_from_accelerator_memory(dram_addr, shape).float()
            if rows is not None:
                got = got[:rows]
            debug_log.append((name, calculate_snr(cpu_ref.detach().float().reshape(-1), got.reshape(-1))))

        self.identity_dram = self._up(torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))
        emb_dram = self._up(self.emb_w)

        # ---- phoneme embedding gather (same row-copy technique as Section 1's) ----
        # Zero-fill first, at emit time: the captured gather below writes ONLY rows 0..T-1, and no
        # other captured op touches rows T..T_pad, so the host zeros survive to execution. (Where a
        # captured op DOES rewrite the region, a host DMA would be silently clobbered -- see
        # _conv1d's real_T handling.) Zero is also what the CPU mask puts in padded positions.
        x_dram = ue.allocate_tensor_dram(T_pad * C * 2)
        ue.dma_to_accelerator_memory(x_dram, torch.zeros(T_pad * C, dtype=torch.bfloat16))
        ue.start_capture()
        for i in range(T):
            tok_id = int(input_ids[i].item())
            ue.accelerator_memory_to_sram(accelerator_dram_address=emb_dram + tok_id * row_bytes,
                                          sram_address=0x00000, element_size=C)
            ue.sram_to_accelerator_memory(sram_address=0x00000,
                                          accelerator_dram_address=x_dram + i * row_bytes,
                                          element_size=C)
        self._run(ue)
        _check("embedding", x_dram, (T_pad, C),
               debug_cpu_ref["embedding"] if debug_cpu_ref else None, rows=T)

        # ---- 3 x (Conv1d k=5 -> LayerNorm -> LeakyReLU(0.2)) ----
        for i, w in enumerate(self.cnn_blocks):
            ue.start_capture()
            conv_dram = ue.allocate_tensor_dram(T_pad * C * 2)
            self._conv1d(x_dram, T_pad, C, C, w["conv_w"], w["conv_b"], conv_dram,
                         kernel_size=self.kernel_size, pad=self.pad, real_T=T)
            ln_dram = ue.allocate_tensor_dram(T_pad * C * 2)
            _dyn_layernorm(ue, M=T_pad, N=C, A_DRAM_ADDR=conv_dram, OUTPUT_DRAM_ADDR=ln_dram,
                                    GAMMA_DRAM_ADDR=self._up(w["ln_g"]), BETA_DRAM_ADDR=self._up(w["ln_b"]))
            act_dram = ue.allocate_tensor_dram(T_pad * C * 2)
            self._leaky_relu(ln_dram, T_pad, C, act_dram)
            self._run(ue)
            x_dram = act_dram
            _check(f"cnn block {i}", x_dram, (T_pad, C),
                   debug_cpu_ref["cnn"][i] if debug_cpu_ref else None, rows=T)

        # ---- BiLSTM (C -> 2 * C/2 = C) ----
        # Run over the REAL T only: this is a sequential bidirectional pass, so starting the
        # backward direction inside the alignment padding would corrupt every real timestep.
        out_dram = ue.allocate_tensor_dram(T_pad * C * 2)
        ue.dma_to_accelerator_memory(out_dram, torch.zeros(T_pad * C, dtype=torch.bfloat16))
        ue.start_capture()
        self._lstm_bidir(x_dram, T, C, self.lstm_w, out_dram)
        self._run(ue)
        _check("lstm output", out_dram, (T_pad, C),
               debug_cpu_ref["lstm"] if debug_cpu_ref else None, rows=T)

        if debug_cpu_ref is not None:
            report_snr("\n[fpga][debug] --- Section 4 bisect (SNR vs CPU) ---")
            for name, snr_db in debug_log:
                flag = "  <-- LOW" if snr_db < 30.0 else ""
                report_snr(f"[fpga][debug] {name:38s} {snr_db:8.2f} dB{flag}")
            report_snr("[fpga][debug] --- end bisect ---\n")

        # CPU returns [B, C, T] (channels-first); transpose our [T, C] to match.
        t_en = ue.dma_from_accelerator_memory(out_dram, (T_pad, C)).float()[:T].T.contiguous()
        ue.reset_tensor_dram_addr()
        return t_en




# Reserved ISA registers for the dynamic (PBI) kernel dimensions.
#
# gpr_M_reg is a PBI LOOP-COUNT field and is validated to 1..15 (user_dma_core.py:2255, 2482,
# 3007); only gpr_N_reg/gpr_K_reg and the address registers accept 1..63. So these must be low --
# and low is exactly where alloc_isa_reg() hands registers out from. _instrument() therefore burns
# the first three allocations permanently, so every later alloc starts at 4 and can never collide.
# Safe because nothing in the repo calls reset_isa_reg_counter() and start_capture() leaves the
# counter alone, so the reservation survives every capture in the run.
GPR_M, GPR_K, GPR_N = 1, 2, 3

# Below this many output rows the dynamic path is a net loss: it costs 3 ADD_SETs to prime the
# dimension registers, while the legacy path emits one matvec per row. The LSTM's per-timestep
# gate matmuls are all M=1, so they stay on the legacy path.
_DYN_M_MIN = 8


def _dyn_matmul(ue, M, K, N, **kw):
    """matmat_mul_core with the dimensions sourced from ISA registers.

    WHY: the legacy kernel emits ONE matvec instruction per output row
    (user_dma_core.py:5473, `for output_row in range(m_take)`), so a matmul over M=320 frames
    costs 320 instructions to emit. That is kokoro's dominant compile cost -- and it is also why
    the program is prompt-specific, since M is baked in. matmat_mul_core dispatches to
    matmat_mul_core_dynamic as soon as a dimension GPR is set (user_dma_core.py:5234), turning the
    software tile loops into ISA while-loops: O(1) instructions regardless of M.

    The dynamic kernel requires ALL THREE dims as registers (user_dma_core.py:5657), so K and N
    are primed too even though they are compile-time constants here.
    """
    if M < _DYN_M_MIN:
        return ue.matmat_mul_core(M=M, K=K, N=N, **kw)
    ue.generate_instruction_add_set(GPR_M, M)
    ue.generate_instruction_add_set(GPR_K, K)
    ue.generate_instruction_add_set(GPR_N, N)
    return ue.matmat_mul_core(M=M, K=K, N=N,
                              gpr_M_reg=GPR_M, gpr_K_reg=GPR_K, gpr_N_reg=GPR_N, **kw)



def _dyn_eltwise(ue, M, *args, **kw):
    """eltwise_core_dram with a runtime row register.

    eltwise_core_dram_legacy Python-unrolls its vertical tiles, so emission is O(M). That bites
    hardest in _leaky_relu, which flattens to M = numel/64 -- at the decoder's T=384, C=1152 that
    is M=6912, i.e. thousands of instructions for one activation. The dynamic core turns the tile
    loop into an ISA loop; passing gpr_M_reg alone is enough, the entrypoint allocates and seeds
    any other register it needs (user_dma_core.py:2049).
    """
    if M >= _DYN_M_MIN and kw.get("gpr_M_reg") is None:
        ue.generate_instruction_add_set(GPR_M, M)
        kw["gpr_M_reg"] = GPR_M
    return ue.eltwise_core_dram(M, *args, **kw)


def _dyn_layernorm(ue, *args, **kw):
    """layer_norm_core_dram with a runtime row register -- its legacy path emits one instruction
    per row (user_dma_core.py:3973). Missing M/N/sqrt(N) registers are seeded by the entrypoint."""
    M = kw.get("M", args[0] if args else None)
    if M is not None and M >= _DYN_M_MIN and kw.get("gpr_M_reg") is None:
        ue.generate_instruction_add_set(GPR_M, M)
        kw["gpr_M_reg"] = GPR_M
    return ue.layer_norm_core_dram(*args, **kw)


def _dyn_transpose(ue, *args, **kw):
    """bf16_transpose_core with runtime dims -- legacy emits one matvec per output row
    (user_dma_core.py:6581)."""
    M = kw.get("M", args[0] if args else None)
    if M is not None and M >= _DYN_M_MIN and kw.get("gpr_M_reg") is None:
        ue.generate_instruction_add_set(GPR_M, M)
        kw["gpr_M_reg"] = GPR_M
    return ue.bf16_transpose_core(*args, **kw)


def _dyn_activation(ue, *args, **kw):
    """activation_core with a runtime row register (it applies the activation through the same
    identity-matmul path, so it inherits the per-row emission)."""
    M = kw.get("M", args[0] if args else None)
    if M is not None and M >= _DYN_M_MIN and kw.get("gpr_M_reg") is None:
        ue.generate_instruction_add_set(GPR_M, M)
        kw["gpr_M_reg"] = GPR_M
    return ue.activation_core(*args, **kw)


def _pbi_row_loop(ue, n_rows, reads, writes, gpr_rows=None, sram_addr=0x00000):
    """One HARDWARE loop over `n_rows` rows instead of a Python-unrolled emission.

    `reads` / `writes` are lists of ``(dram_base, row_bytes, elems, sram_off)``. Each iteration
    stages every read into SRAM at its ``sram_off``, then writes every entry back out; both the
    source and destination addresses are recomputed per iteration into a scratch register, so the
    strides are independent and arbitrary.

    WHY: the Python loops this replaces emit 2-4 instructions PER ROW, which is what makes a
    kokoro program prompt-specific and slow to compile -- Section 4 emitted 18,430 instructions
    for 49 phonemes, almost all of it LSTM/row-copy unroll, and compile time runs ~6x the
    accelerator's own execution time. A hardware loop emits a fixed ~8 instructions regardless of
    row count. Same mechanism gemma3 uses (see _emit_strided_copy_pbi, gemma3_test.py:570).

    ``gpr_rows``: GPR index holding the trip count at RUNTIME. When given, the emitted program is
    independent of the row count and one bin serves any prompt. When ``None``, the count is baked
    (still a hardware loop, so the instruction-count win holds, but the bin stays prompt-specific).

    Addresses are computed in UE word units (byte >> 3), so every ``row_bytes`` must be a multiple
    of 8 -- true here since row widths are 64-aligned bf16, i.e. multiples of 128 bytes.

    ``relative=True`` (loop_start's default) closes with a backward RELA_JNZ, which is bounded by
    a 512-instruction i-cache window; these bodies are under a dozen instructions, well inside it.
    """
    for base, row_bytes, _elems, _off in list(reads) + list(writes):
        assert row_bytes % 8 == 0, f"row_bytes={row_bytes} must be a multiple of 8 (word-addressed)"
        assert base % 8 == 0, f"dram base 0x{base:X} must be 8-byte aligned"
    i_reg = ue.alloc_isa_reg()
    t_reg = ue.alloc_isa_reg()
    ue.generate_instruction_add_set(i_reg, 0)
    ue.loop_start(loop_cnt=n_rows, gpr_loop_cnt=gpr_rows)
    for base, row_bytes, elems, off in reads:
        ue.generate_instruction_reg_mul_imm(t_reg, i_reg, ue_35bit_addr_shifter(row_bytes))
        ue.generate_instruction_add_imm(t_reg, ue_35bit_addr_shifter(base), t_reg)
        ue.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=sram_addr + off,
                                      element_size=elems, general_reg_src=t_reg)
    for base, row_bytes, elems, off in writes:
        ue.generate_instruction_reg_mul_imm(t_reg, i_reg, ue_35bit_addr_shifter(row_bytes))
        ue.generate_instruction_add_imm(t_reg, ue_35bit_addr_shifter(base), t_reg)
        ue.sram_to_accelerator_memory(sram_address=sram_addr + off, accelerator_dram_address=0,
                                      element_size=elems, general_reg_src=t_reg)
    ue.generate_instruction_add_inc(i_reg)
    ue.loop_end()
    ue.release_isa_reg()   # t_reg
    ue.release_isa_reg()   # i_reg



def _col_sum_rows(ue, src_dram, n_rows, C, tmp_a, tmp_b):
    """Sum the first `n_rows` rows of a [*, C] bf16 buffer -> a single [1, C] row.

    Pairwise tree reduction: add the top half to the bottom half, halving each step (folding in
    the odd row when the count is odd). ~log2(n) eltwise ops on shrinking buffers, versus a matmul
    that would need the [C, T] transpose we do not have yet at this point. Alternates two scratch
    buffers so no step reads the buffer it is writing. Returns the [1, C] sum's DRAM address.
    """
    row_b = C * 2
    cur, n, dst_a = src_dram, n_rows, True
    while n > 1:
        h, odd = n // 2, n % 2
        dst = tmp_a if dst_a else tmp_b
        _dyn_eltwise(ue, h, C, cur, cur + h * row_b, dst, mode=UE_MODE.ELTWISE_ADD)
        if odd:
            _dyn_eltwise(ue, 1, C, dst, cur + (n - 1) * row_b, dst, mode=UE_MODE.ELTWISE_ADD)
        cur, n, dst_a = dst, h, not dst_a
    return cur


def _pad_dim(t: torch.Tensor, dim: int, target: int) -> torch.Tensor:
    """Zero-pad `t` along `dim` up to `target`. Used to bring a CHANNEL count up to a multiple of
    UE_VECTOR_SIZE. Padding a conv weight's INPUT-channel axis is exact: those taps multiply the
    padded (don't-care) channels by zero, so whatever garbage sits there contributes nothing --
    which is why the activation buffers themselves never need re-zeroing on the channel axis.
    """
    if t.shape[dim] == target:
        return t
    assert t.shape[dim] < target, (t.shape, dim, target)
    shape = list(t.shape)
    shape[dim] = target - t.shape[dim]
    return torch.cat([t, torch.zeros(*shape, dtype=t.dtype)], dim=dim)


def _tile_pad_channels(x: torch.Tensor, c_real: int) -> None:
    """In-place: fill x[:, c_real:]'s alignment-padding channels by tiling the real ones.

    They must not be left at zero. AdaIN normalizes per channel over time, so an all-zero channel
    has zero variance, and the hardware RSQRT has no epsilon -- `0 * rsqrt(0)` = NaN, which then
    survives the zero conv taps that were supposed to discard the channel. Tiling gives the same
    finite statistics a real channel has; the zero taps still make the contribution exactly zero,
    so this changes nothing numerically about the real output.
    """
    c_pad = x.shape[1]
    if c_pad <= c_real:
        return
    n = c_pad - c_real
    reps = (n + c_real - 1) // c_real
    x[:, c_real:] = x[:, :c_real].repeat(1, reps)[:, :n]


class DecoderFPGA:
    """Section 5a: the Decoder FRONT -- everything up to (not including) the ISTFTNet Generator.

        F0 = F0_conv(F0_curve);  N = N_conv(N_curve)          (1->1 ch, k=3, stride=2)
        x  = encode( cat([asr, F0, N]) )                       AdainResBlk1d 514 -> 1024
        for 4 decode blocks:
            x = block( cat([x, asr_res, F0, N]) )              AdainResBlk1d 1090 -> 1024 (x3),
                                                               then 1090 -> 512 with upsample
        audio = generator(x, s, F0_curve)                      <-- still CPU, see below

    Every block here is the SAME AdainResBlk1d that Section 3 already runs, so this reuses
    `_adain_res_blk` and friends outright; the new mechanics are all layout:

    CHANNEL-AXIS ALIGNMENT (first time this bites -- Sections 1-4 only had to pad the TIME axis).
    The concatenations produce 512+1+1 = 514 and 1024+64+1+1 = 1090 channels, neither a multiple
    of 64. Channels land on `_conv1d`'s K and on `_adain1d`'s transpose M, so they must be padded
    (-> 576 and 1152). Padding is applied to the WEIGHTS' input-channel axis, so a zero tap
    discards each don't-care channel's contribution.

    That is necessary but NOT sufficient: the padded channels must also carry FINITE values.
    AdaIN normalizes per channel over time, so an all-zero padded channel has zero variance, and
    the hardware RSQRT path has no epsilon (user_dma_core.py:3864) -- `0 * rsqrt(0)` = `0 * inf`
    = NaN. NaN times a zero tap is still NaN, so weight padding alone cannot rescue it and the
    whole section reads -inf dB. The padded channels are therefore TILED from real channels:
    nonzero variance keeps the normalizer finite, and the zero taps still discard the result.

    STILL ON CPU, deliberately:
      * `F0_conv` / `N_conv` -- 1-in/1-out channel, k=3, stride=2 over ~506 samples. K=1 would have
        to be padded to 64 to reach the accelerator, i.e. 64x waste on a rounding-error amount of
        compute, and F0/N already arrive as host tensors from Section 3, so this costs no extra
        round trip.
      * `asr = t_en @ pred_aln_trg` and `asr_res` -- same reasoning, both operands already host-side.
      * The GENERATOR (Section 5b-5d). Not a blocker, just not written yet: the op gaps are already
        proven (`sincos_bounded_poly_test` for Snake1D/phase, `exp_via_sigmoid_test` for `spec`,
        `conv_transpose1d_zero_insert_test` for `ups`). The one genuinely open question is
        SineGen's UNBOUNDED accumulating phase -- the Taylor fit is proven on [-1,1] only.
    """

    _adain1d = F0NPredictionFPGA._adain1d
    _adain_res_blk = F0NPredictionFPGA._adain_res_blk
    _conv1d = F0NPredictionFPGA._conv1d
    _leaky_relu = F0NPredictionFPGA._leaky_relu
    _nearest_upsample2x = F0NPredictionFPGA._nearest_upsample2x
    _depthwise_convtranspose_upsample2x = F0NPredictionFPGA._depthwise_convtranspose_upsample2x
    _concat_columns = ProsodyDurationFPGA._concat_columns   # lives on Section 2, not Section 3
    _device_row_copy = F0NPredictionFPGA._device_row_copy
    _up = F0NPredictionFPGA._up
    _run = F0NPredictionFPGA._run

    def __init__(self, model, ue: UnifiedEngine):
        self.model = model
        self.ue = ue
        dec = model.decoder
        self.dec = dec
        self.style_dim = dec.encode.norm1.fc.in_features       # 128

        self.C_asr = dec.asr_res[0].in_channels                # 512
        self.C_res = dec.asr_res[0].out_channels               # 64
        self.C_enc_in = self.C_asr + 2                         # 514
        self.C_enc_out = dec.encode.conv1.out_channels         # 1024
        self.C_dec_in = self.C_enc_out + self.C_res + 2        # 1090
        self.C_enc_in_pad = _round_up(self.C_enc_in, UE_VECTOR_SIZE)   # 576
        self.C_dec_in_pad = _round_up(self.C_dec_in, UE_VECTOR_SIZE)   # 1152

        def adain_w(m, C_real, C_pad):
            """AdaIN1d fc -> [2*C_pad, style_dim], laid out gamma-block then beta-block so that
            _adain1d's `gb[:C]` / `gb[C:]` split still works at the PADDED width."""
            w, b = m.fc.weight, m.fc.bias
            g_w, be_w = w[:C_real], w[C_real:]
            g_b, be_b = b[:C_real], b[C_real:]
            pad_w = torch.zeros(C_pad - C_real, w.shape[1], dtype=w.dtype)
            pad_b = torch.zeros(C_pad - C_real, dtype=b.dtype)
            return (_bf16(torch.cat([g_w, pad_w, be_w, pad_w], dim=0)),
                    _bf16(torch.cat([g_b, pad_b, be_b, pad_b], dim=0)))

        def block_w(m, C_in_real, C_in_pad):
            w = {}
            c1w, c1b = _wn_conv(m.conv1)
            c2w, c2b = _wn_conv(m.conv2)
            w["conv1_w"] = _bf16(_pad_dim(c1w, 1, C_in_pad))   # [Cout, Cin_pad, 3]
            w["conv1_b"] = _bf16(c1b)
            w["conv2_w"], w["conv2_b"] = _bf16(c2w), _bf16(c2b)
            w["norm1_fc_w"], w["norm1_fc_b"] = adain_w(m.norm1, C_in_real, C_in_pad)
            w["norm2_fc_w"], w["norm2_fc_b"] = adain_w(m.norm2, m.conv2.out_channels,
                                                        m.conv2.out_channels)
            if m.learned_sc:
                sc_w, _ = _wn_conv(m.conv1x1)
                w["conv1x1_w"] = _bf16(_pad_dim(sc_w, 1, C_in_pad))
            if m.upsample_type != "none":
                pw, pb = _wn_conv(m.pool)                       # depthwise: [Cin, 1, 3]
                w["pool_w"] = _bf16(_pad_dim(pw, 0, C_in_pad))
                w["pool_b"] = _bf16(_pad_dim(pb, 0, C_in_pad))
            return w

        self.encode_w = block_w(dec.encode, self.C_enc_in, self.C_enc_in_pad)
        self.decode_w = [block_w(b, self.C_dec_in, self.C_dec_in_pad) for b in dec.decode]

    def forward(self, asr: torch.Tensor, F0_curve: torch.Tensor, N_curve: torch.Tensor,
                s: torch.Tensor, debug_cpu_ref=None):
        """asr: [C_asr, T] host. F0_curve/N_curve: [2T] host (Section 3's output, pre-F0_conv).
        s: [style_dim] host. Returns x [C_out, 2T] host -- the Generator's input.
        """
        ue = self.ue
        dec = self.dec
        debug_log = []

        def _check(name, dram_addr, shape, cpu_ref, rows=None):
            from user_dma_core import calculate_snr
            if debug_cpu_ref is None or cpu_ref is None:
                return
            got = ue.dma_from_accelerator_memory(dram_addr, shape).float()
            if rows is not None:
                got = got[:rows]
            debug_log.append((name, calculate_snr(cpu_ref.detach().float().reshape(-1), got.reshape(-1))))

        with torch.no_grad():
            F0 = dec.F0_conv(F0_curve.unsqueeze(0).unsqueeze(1)).squeeze(0)   # [1, T]
            Nc = dec.N_conv(N_curve.unsqueeze(0).unsqueeze(1)).squeeze(0)     # [1, T]
            asr_res = dec.asr_res(asr.unsqueeze(0)).squeeze(0)                # [C_res, T]

        T = asr.shape[-1]
        T_pad = _round_up(T, UE_VECTOR_SIZE)
        self.identity_dram = self._up(torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))
        style_dram = self._up(_bf16(s))

        # ---- encode: x0 = cat([asr, F0, N]) built host-side in one upload ----
        x0 = torch.zeros(T_pad, self.C_enc_in_pad, dtype=torch.bfloat16)
        x0[:T, :self.C_asr] = _bf16(asr.T)
        x0[:T, self.C_asr] = _bf16(F0[0])
        x0[:T, self.C_asr + 1] = _bf16(Nc[0])
        _tile_pad_channels(x0, self.C_enc_in)
        x_dram = self._up(x0)

        cur_dram, cur_C = self._adain_res_blk(
            x_dram, T_pad, self.C_enc_in_pad, self.C_enc_out, False, self.encode_w,
            style_dram, real_T=T)[0], self.C_enc_out
        _check("encode", cur_dram, (T_pad, self.C_enc_out),
               debug_cpu_ref["encode"] if debug_cpu_ref else None, rows=T)

        # The host-side half of every decode concat: [asr_res | F0 | N | zero-pad], one upload,
        # laid out so its width (C_dec_in_pad - C_enc_out) is itself 64-aligned and _concat_columns
        # can stitch it onto the device-resident x without any sub-64-wide copies.
        side_w = self.C_dec_in_pad - self.C_enc_out
        side = torch.zeros(T_pad, side_w, dtype=torch.bfloat16)
        side[:T, :self.C_res] = _bf16(asr_res.T)
        side[:T, self.C_res] = _bf16(F0[0])
        side[:T, self.C_res + 1] = _bf16(Nc[0])
        _tile_pad_channels(side, self.C_res + 2)
        side_dram = self._up(side)

        cur_T, real_T = T_pad, T
        res = True
        for i, (blk, w) in enumerate(zip(dec.decode, self.decode_w)):
            if res:
                ue.start_capture()
                cat_dram = ue.allocate_tensor_dram(cur_T * self.C_dec_in_pad * 2)
                self._concat_columns(cur_dram, self.C_enc_out, side_dram, side_w, cur_T, cat_dram)
                self._run(ue)
                cur_dram, cur_C = cat_dram, self.C_dec_in_pad
            up = blk.upsample_type != "none"
            cur_dram, cur_T = self._adain_res_blk(
                cur_dram, cur_T, cur_C, blk.conv2.out_channels, up, w, style_dram, real_T=real_T)
            cur_C = blk.conv2.out_channels
            real_T = 2 * real_T if up else real_T
            if up:
                res = False
            _check(f"decode {i}", cur_dram, (cur_T, cur_C),
                   debug_cpu_ref["decode"][i] if debug_cpu_ref else None, rows=real_T)

        if debug_cpu_ref is not None:
            report_snr("\n[fpga][debug] --- Section 5a bisect (SNR vs CPU) ---")
            for name, snr_db in debug_log:
                flag = "  <-- LOW" if snr_db < 30.0 else ""
                report_snr(f"[fpga][debug] {name:38s} {snr_db:8.2f} dB{flag}")
            report_snr("[fpga][debug] --- end bisect ---\n")

        x_out = ue.dma_from_accelerator_memory(cur_dram, (cur_T, cur_C)).float()[:real_T].T.contiguous()
        ue.reset_tensor_dram_addr()
        return x_out


def run_fpga_forward(model, phonemes: str, ref_s: torch.FloatTensor, speed: float = 1.0,
                     dev: str = "xdma0", debug: bool = False):
    """Entry point called from kokoro_test.py --fpga. Only runs the section(s) currently ported to
    hardware (Section 1: PL-BERT) and reports their SNR against the CPU reference, with a bisect
    down to sub-stage granularity within each layer. Deliberately does NOT fall back to running the
    rest of the model on CPU -- --fpga means "run and validate what's actually on the FPGA today",
    not "run the full pipeline with FPGA sprinkled in". Returns None; callers should not expect
    audio until later sections are ported (see the section checklist at the top of this file).
    """
    set_dma_device(dev)
    _DEBUG_SNR[0] = debug
    ue = UnifiedEngine()
    _instrument(ue)
    global _SILENT_MODE
    _SILENT_MODE = True   # hide the cores' per-call tiling/FLOP prints

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

    _begin("1: PL-BERT encoder")
    report(f"[fpga] Section 1 (PL-BERT): T={T} phoneme tokens ...")
    plbert = PLBertFPGA(model, ue)
    with torch.no_grad():
        d_en = plbert.forward(input_ids, debug_cpu_ref=(debug_cpu_ref if debug else None))  # [512, T]

        input_ids_b = input_ids.unsqueeze(0)
        text_mask = torch.zeros(1, T, dtype=torch.bool)
        bert_dur_cpu = model.bert(input_ids_b, attention_mask=(~text_mask).int())
        d_en_cpu = model.bert_encoder(bert_dur_cpu).transpose(-1, -2).squeeze(0)  # [512, T]
        from user_dma_core import calculate_snr
        snr_db = calculate_snr(d_en_cpu.reshape(-1), d_en.reshape(-1))
        report_snr(f"[fpga] Section 1 (PL-BERT) end-to-end SNR vs CPU: {snr_db:.2f} dB")

        # ---- Section 2: prosody/duration path ----
        # WIRED: fed from Section 1's own FPGA output (`d_en`), not the clean CPU value. The
        # per-section SNRs printed below are therefore CUMULATIVE (they include every upstream
        # section's error), not isolated. To bisect a regression back to a single section, swap
        # the input back to the corresponding *_cpu value and compare.
        style_vec = ref_s.reshape(-1)[128:256]
        input_lengths = torch.full((1,), T, dtype=torch.long)
        ref_s_b = ref_s.reshape(1, -1)
        s_cpu = ref_s_b[:, 128:]
        d_cpu = model.predictor.text_encoder(d_en_cpu.unsqueeze(0), s_cpu, input_lengths, text_mask)  # [1,T,640]
        lstm_out_cpu, _ = model.predictor.lstm(d_cpu)  # [1,T,512]
        duration_raw_cpu = model.predictor.duration_proj(lstm_out_cpu)  # [1,T,50]

        _begin("2: Prosody / duration")
        report(f"\n[fpga] Section 2 (prosody/duration): T={T} ...")
        prosody = ProsodyDurationFPGA(model, ue)
        debug_cpu_ref2 = {
            "d": d_cpu.squeeze(0), "lstm_out": lstm_out_cpu.squeeze(0), "duration_raw": duration_raw_cpu.squeeze(0),
        }
        d_fpga, pred_dur_fpga = prosody.forward(d_en, style_vec, T,
                                                debug_cpu_ref=(debug_cpu_ref2 if debug else None))

        duration_cpu = torch.sigmoid(duration_raw_cpu.squeeze(0)).sum(axis=-1)
        pred_dur_cpu = torch.round(duration_cpu).clamp(min=1).long()
        pred_dur_snr = calculate_snr(pred_dur_cpu.float(), pred_dur_fpga.float())
        report_snr(f"[fpga] Section 2 (prosody/duration) end-to-end SNR (d) vs CPU: "
              f"{calculate_snr(d_cpu.squeeze(0).reshape(-1), d_fpga.reshape(-1)):.2f} dB")
        report_snr(f"[fpga] Section 2 pred_dur SNR vs CPU (post-round, should be near-exact): {pred_dur_snr:.2f} dB")
        n_mismatch = (pred_dur_cpu != pred_dur_fpga).sum().item()
        report_snr(f"[fpga] Section 2 pred_dur exact-match: {T - n_mismatch}/{T} positions "
              f"(mismatches usually a rounding tie at a .5 boundary, not a correctness bug)")

        # ---- Section 3: F0/N prediction ----
        # WIRED: fed from Section 2's own FPGA outputs (d_fpga / pred_dur_fpga).
        #
        # The CPU reference for this section is built from the FPGA's OWN upstream values, not
        # from d_cpu/pred_dur_cpu. Two reasons, and the second is not optional:
        #   1. It makes the SNRs below measure Section 3's own arithmetic, given the inputs it
        #      actually received -- rather than smearing upstream error into every stage number.
        #   2. pred_dur sets a data-dependent SHAPE (n_frames = pred_dur.sum()). Once chaining
        #      makes pred_dur_fpga differ from pred_dur_cpu by even one rounding tie, the two
        #      pipelines produce different-length tensors and an end-to-end comparison is not
        #      merely noisy, it is undefined -- calculate_snr raises on the shape mismatch.
        # The upstream divergence is still reported: see the pred_dur exact-match line above.
        indices_cpu = torch.repeat_interleave(torch.arange(T), pred_dur_fpga)
        pred_aln_trg_cpu = torch.zeros((T, indices_cpu.shape[0]))
        pred_aln_trg_cpu[indices_cpu, torch.arange(indices_cpu.shape[0])] = 1
        pred_aln_trg_cpu = pred_aln_trg_cpu.unsqueeze(0)
        en_cpu = d_fpga.unsqueeze(0).float().transpose(-1, -2) @ pred_aln_trg_cpu  # [1,640,n_frames]
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

        n_frames = int(pred_dur_fpga.sum().item())
        n_frames_cpu = int(pred_dur_cpu.sum().item())
        if n_frames != n_frames_cpu:
            report_snr(f"[fpga][note] n_frames: fpga={n_frames} vs pure-CPU={n_frames_cpu} "
                  f"({n_frames - n_frames_cpu:+d} frames from pred_dur rounding ties upstream). "
                  f"Section 3's reference is built from the FPGA alignment, so the SNRs below "
                  f"remain valid measures of Section 3 itself.")
        _begin("3: F0 / N prediction")
        report(f"\n[fpga] Section 3 (F0/N prediction): n_frames={n_frames} ...")
        f0n = F0NPredictionFPGA(model, ue)
        debug_cpu_ref3 = {
            "shared_out": shared_lstm_out_cpu.squeeze(0),
            "F0_blocks": F0_block_outs_cpu, "F0_pred_pre_squeeze": F0_pred_cpu,
            "N_blocks": N_block_outs_cpu, "N_pred_pre_squeeze": N_pred_cpu,
        }
        F0_pred_fpga, N_pred_fpga = f0n.forward(d_fpga, pred_dur_fpga, style_vec, T,
                                                 debug_cpu_ref=(debug_cpu_ref3 if debug else None))

        f0_snr = calculate_snr(F0_pred_cpu.reshape(-1), F0_pred_fpga.reshape(-1))
        n_snr = calculate_snr(N_pred_cpu.reshape(-1), N_pred_fpga.reshape(-1))
        report_snr(f"[fpga] Section 3 F0_pred end-to-end SNR vs CPU: {f0_snr:.2f} dB")
        report_snr(f"[fpga] Section 3 N_pred end-to-end SNR vs CPU: {n_snr:.2f} dB")

        # ---- Section 4: TextEncoder (phoneme embed + Conv1d k=5 stack + BiLSTM) ----
        # Independent of Sections 2/3: it consumes the raw phoneme ids, not the prosody path, so
        # it is fed the real input rather than any upstream FPGA output. Its result joins the
        # pipeline in Section 5, as `asr = t_en @ pred_aln_trg`.
        _begin("4: TextEncoder")
        report(f"\n[fpga] Section 4 (TextEncoder): T={T} phoneme tokens ...")
        te = model.text_encoder
        emb_cpu = te.embedding(input_ids_b).squeeze(0)                 # [T, C]
        xc = emb_cpu.transpose(0, 1).unsqueeze(0)                      # [1, C, T] channels-first
        cnn_cpu = []
        for blk in te.cnn:
            xc = blk(xc)
            cnn_cpu.append(xc.squeeze(0).transpose(0, 1))              # store [T, C] to match device
        lstm_cpu, _ = te.lstm(xc.transpose(-1, -2))                    # [1, T, C]
        t_en_cpu = te(input_ids_b, input_lengths, text_mask).squeeze(0)  # [C, T]

        textenc = TextEncoderFPGA(model, ue)
        t_en_fpga = textenc.forward(input_ids, T, debug_cpu_ref=({
            "embedding": emb_cpu, "cnn": cnn_cpu, "lstm": lstm_cpu.squeeze(0),
        } if debug else None))
        report_snr(f"[fpga] Section 4 t_en end-to-end SNR vs CPU: "
              f"{calculate_snr(t_en_cpu.reshape(-1), t_en_fpga.reshape(-1)):.2f} dB")

        # ---- Section 5a: Decoder front (encode + 4 decode AdainResBlk1d) ----
        # Wired from Sections 3 and 4's own FPGA outputs: asr from t_en_fpga, F0/N from Section 3.
        s_dec = ref_s.reshape(1, -1)[:, :128]                      # decoder style (predictor uses [128:])
        asr_fpga = t_en_fpga @ pred_aln_trg_cpu.squeeze(0)          # [C, n_frames]
        dec = model.decoder
        with torch.no_grad():
            F0c = dec.F0_conv(F0_pred_fpga.unsqueeze(0).unsqueeze(1)).squeeze(0)
            Nc_ = dec.N_conv(N_pred_fpga.unsqueeze(0).unsqueeze(1)).squeeze(0)
            asr_res_cpu = dec.asr_res(asr_fpga.unsqueeze(0))
            enc_cpu = dec.encode(torch.cat([asr_fpga.unsqueeze(0), F0c.unsqueeze(0), Nc_.unsqueeze(0)], axis=1), s_dec)
            dec_cpu, xc, res_ = [], enc_cpu, True
            for blk in dec.decode:
                if res_:
                    xc = torch.cat([xc, asr_res_cpu, F0c.unsqueeze(0), Nc_.unsqueeze(0)], axis=1)
                xc = blk(xc, s_dec)
                if blk.upsample_type != "none":
                    res_ = False
                dec_cpu.append(xc.squeeze(0).transpose(0, 1))       # [T, C] to match device layout

        _begin("5a: Decoder front")
        report(f"\n[fpga] Section 5a (Decoder front): asr={tuple(asr_fpga.shape)} ...")
        decoder = DecoderFPGA(model, ue)
        x_gen = decoder.forward(asr_fpga, F0_pred_fpga, N_pred_fpga, s_dec.reshape(-1),
                                debug_cpu_ref=({"encode": enc_cpu.squeeze(0).transpose(0, 1),
                                                "decode": dec_cpu} if debug else None))
        report_snr(f"[fpga] Section 5a decoder-front SNR vs CPU: "
              f"{calculate_snr(dec_cpu[-1].reshape(-1), x_gen.T.reshape(-1)):.2f} dB")

        # ---- Section 5b-5d: ISTFTNet Generator -- STILL ON CPU ----
        # Not blocked, just not written: the op gaps are already proven in user_hw_test.py
        # (sincos_bounded_poly_test for Snake1D/phase, exp_via_sigmoid_test for `spec`,
        # conv_transpose1d_zero_insert_test for `ups`). SineGen's unbounded accumulating phase is
        # the one genuinely open question -- the Taylor fit is proven on [-1,1] only.
        report("[fpga] Section 5b-5d (ISTFTNet generator): CPU fallback -- not ported yet.")
        with torch.no_grad():
            audio = dec.generator(x_gen.unsqueeze(0), s_dec, F0_pred_fpga.unsqueeze(0)).squeeze()

        _CUR[0] = None
        report(f"[fpga] Pipeline complete: sections 1-5a on hardware, generator on CPU.")
        _report_timings()
        _SILENT_MODE = False     # hand normal printing back to the caller
        return audio

    return None
