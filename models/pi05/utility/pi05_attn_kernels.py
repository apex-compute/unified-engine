"""pi05-local attention kernel override (user_dma_core.py stays untouched).

``attention_split_hd`` is ``UnifiedEngine.unified_attention_core_dynamic`` with the
one head_dim split in two:

    qk_dim  the CONTRACTION dim of Q @ K^T. Must be a multiple of 64 (the matmul
            asserts K % 64), so SigLIP's 72 is padded to 128 here, as before.
    v_dim   the OUTPUT width of P @ V. P @ V contracts over the keys, so head_dim
            is its N, and N has no 64-rule outside softmax. v_dim may be the real
            72 (or any width <= qk_dim).

and one addition:

    out_row_stride  (elements) row pitch of the destination. With it, P @ V writes
            each head's (batch, v_dim) block straight into its columns of a wider
            token-major buffer, e.g. (S, 16 * 72): base + h * v_dim, stride 16 * 72.
            This is what removes the separate unpad step.

Steps, identical to the stock core except where marked:
  1. V^T   bf16_transpose_core over the FULL qk_dim-wide V -> (qk_dim, S). Its
           first v_dim rows are contiguous, and they are the only rows P @ V reads.
  2. Q *= scale        (skipped when q_pre_scaled)
  3. P = softmax(Q @ K^T + bias)         K = qk_dim
  4. OUT = P @ V^T[:v_dim]^T             N = v_dim, row pitch out_row_stride  [CHANGED]

Scratch layout is the stock one: V^T (qk_dim*S) | P (S*S) | scaled Q (batch*qk_dim).
"""
import math

from user_dma_core import UE_VECTOR_SIZE, UE_MODE


def attention_split_hd(ue, batch, aligned_seq_len, qk_dim, v_dim,
                       Q_DRAM_ADDR, K_DRAM_ADDR, V_DRAM_ADDR, BIAS_DRAM_ADDR,
                       OUTPUT_DRAM_ADDR, SCRATCH_DRAM_ADDR, IDENTITY_DRAM_ADDR,
                       out_row_stride=None, q_scale=None, q_pre_scaled=False,
                       V_T_DRAM_ADDR=None):
    bpe = 2
    S = aligned_seq_len
    if batch > S:
        raise ValueError(f"attention_split_hd: batch={batch} > aligned_seq_len={S}")
    if S % UE_VECTOR_SIZE or qk_dim % UE_VECTOR_SIZE:
        raise ValueError(f"attention_split_hd: aligned_seq_len={S} and qk_dim={qk_dim} "
                         f"must be multiples of {UE_VECTOR_SIZE}")
    if not 0 < v_dim <= qk_dim:
        raise ValueError(f"attention_split_hd: v_dim={v_dim} must be in (0, qk_dim={qk_dim}]")
    if OUTPUT_DRAM_ADDR % 8:
        raise ValueError("attention_split_hd: OUTPUT_DRAM_ADDR must be 8-byte aligned (PBI word)")

    regs = []

    def _reg(val):
        r = ue.alloc_isa_reg()
        regs.append(r)
        ue.generate_instruction_add_set(r, val)
        return r

    batch_reg = _reg(batch)
    seq_reg = _reg(S)
    qk_reg = _reg(qk_dim)
    v_reg = _reg(v_dim)
    stride_reg = _reg(out_row_stride) if out_row_stride is not None else None

    v_t = SCRATCH_DRAM_ADDR
    score = v_t + qk_dim * S * bpe
    scaled_q = score + S * S * bpe

    # 1. V^T over the padded width; only rows [0, v_dim) are consumed below.
    if V_T_DRAM_ADDR is not None:
        v_t = V_T_DRAM_ADDR
    else:
        ue.bf16_transpose_core(M=S, N=qk_dim, INPUT_DRAM_ADDR=V_DRAM_ADDR,
                               OUTPUT_DRAM_ADDR=v_t, IDENTITY_DRAM_ADDR=IDENTITY_DRAM_ADDR,
                               gpr_M_reg=seq_reg)
    # 2. scale Q
    flops = 0
    if not q_pre_scaled:
        flops += ue.eltwise_core_dram(
            M=batch, N=qk_dim, dram_a=Q_DRAM_ADDR, dram_b=None, dram_out=scaled_q,
            mode=UE_MODE.MUL_BROADCAST,
            scalar=(q_scale if q_scale is not None else 1.0 / math.sqrt(qk_dim)),
            gpr_M_reg=batch_reg)
    q_src = Q_DRAM_ADDR if q_pre_scaled else scaled_q
    # 3. scores + softmax (contraction over qk_dim)
    flops += ue.matmat_mul_core(
        M=batch, K=qk_dim, N=S, A_DRAM_ADDR=q_src, B_DRAM_ADDR=K_DRAM_ADDR,
        OUTPUT_DRAM_ADDR=score, softmax_enable=True,
        C_DRAM_ADDR=BIAS_DRAM_ADDR, bias_mode="full_matrix",
        gpr_M_reg=batch_reg, gpr_K_reg=qk_reg, gpr_N_reg=seq_reg)
    # 4. context: N = v_dim, written at the caller's row pitch
    flops += ue.matmat_mul_core(
        M=batch, K=S, N=v_dim, A_DRAM_ADDR=score, B_DRAM_ADDR=v_t,
        OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
        gpr_M_reg=batch_reg, gpr_K_reg=seq_reg, gpr_N_reg=v_reg,
        gpr_out_row_stride_reg=stride_reg)
    for _ in regs:
        ue.release_isa_reg()
    return flops
