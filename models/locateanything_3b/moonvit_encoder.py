#!/usr/bin/env python3
"""MoonViT-SO-400M vision encoder as an FPGA execution flow (LocateAnything-3B).

Stitches the existing UnifiedEngine / nn_lib ops into the MoonViT transformer stack so
the encoder runs on the board instead of the GPU. The output [N_merged, 4608] lands in
ue.VIS_MERGED_DRAM, exactly where the already-working FPGA connector (run_connector in
locateanything_3b_test.py) reads it — so the rest of the pipeline is unchanged.

Structure mirrors the Qwen2.5-VL vision encoder (models/qwen2.5_vl_3b compile_encoder):
a 2D-RoPE ViT with head_dim padded to 128 + per-head flash attention. MoonViT specifics:
  * LayerNorm (gamma+beta), NOT RMSNorm.
  * 2-layer GELU-tanh MLP (fc0 -> gelu -> fc1), like SigLIP. Engine GELU is the
    sigmoid approximation (already tolerated downstream in the connector).
  * head_dim 72 -> padded to 128 for rope + flash; trimmed back to 72 after attention.
  * intermediate 4304 -> padded to 4352 (next multiple of 64).
  * full bidirectional attention (no causal mask, no GQA).

NO new kernels: every op here already exists in user_dma_core.py / nn_lib.py.

--- RoPE convention note (the one delicate part) -------------------------------------
MoonViT's reference applies *interleaved* (GPT-J / view_as_complex) 2D RoPE: it rotates
adjacent pairs (x_{2j}, x_{2j+1}) by angle phi_j. The engine's rope_hf_core_dram does
*rotate_half* (GPT-NeoX): it rotates (x_i, x_{i+half}). These are equivalent if the
head_dim of Q/K is reordered evens-then-odds: position i<half <- original 2i, position
i+half <- original 2i+1. After that reorder, rotate_half(reordered) == interleaved(orig),
just with the head_dim permuted. Because Q.K^T is invariant to a *consistent* permutation
of head_dim on both Q and K, attention is unchanged and we never have to un-permute. So we
bake the evens-odds reorder into the q/k projection weights on the host; V keeps natural
order (it is not roped), so the attention output is already in natural head_dim order for
the o-projection. This reuses the validated pad-72->128 rope path (user_hw_test.py
vision_rope_hf_core_dram_test) with zero new kernels.

The flash core scales scores by 1/sqrt(128) (padded head_dim), but the reference uses
1/sqrt(72). We correct by pre-scaling the q weight+bias by sqrt(128/72); rope is a linear
rotation so the scale commutes through it.
"""
import math
import os

import numpy as np
import torch

from user_dma_core import DMA_DEVICE_H2C, TYPE, UE_VECTOR_SIZE
try:
    from user_dma_core import UE_FMAX_CONTEXT_SIZE
except Exception:
    UE_FMAX_CONTEXT_SIZE = 128
from nn_lib import (
    store_weight,
    store_quantized_weight,
    smart_bf16_permute_core,
    eltwise_add_core_dram,
    layer_norm_core_dram_post_add,
)
from quant_lib import quantize_q4_64

VD_PAD = 128  # padded head_dim (72 -> 128), must be a multiple of 64


def _vis_dims(cfg):
    v = cfg["vision_config"]
    return dict(
        depth=v["num_hidden_layers"],          # 27
        VH=v["hidden_size"],                    # 1152
        VN=v["num_attention_heads"],            # 16
        VD=v["head_dim"],                       # 72
        VI=v["intermediate_size"],              # 4304
        theta=v["rope_2d_theta"],               # 10000
        merge=v["merge_kernel_size"][0],        # 2
    )


# =====================================================================================
# Host-side weight preparation (padding + evens-odds reorder + q score-scale)
# =====================================================================================

def _qk_pad_reorder(W, b, VN, VD, scale):
    """[VN*VD, VH] q or k weight -> [VN*VD_PAD, VH] in the pad-128 rotate_half layout.

    rotate_half pairs lane i with lane i+64 (half of VD_PAD=128). The rope cos/sin table
    has the real angles at lanes [0:36] (lo) and [64:100] (hi). So the EVEN head_dim
    indices (the real part of each interleaved pair) go to lanes 0..35 and the ODD
    indices (imag part) go to lanes 64..99 — NOT contiguous 0..71. Pad lanes (36..63,
    100..127) stay zero. `scale` folds the sqrt(128/72) flash-scale correction into q."""
    VH = W.shape[1]
    half = VD // 2                       # 36
    hp = VD_PAD // 2                     # 64
    W = W.reshape(VN, VD, VH)
    b = b.reshape(VN, VD)
    W_even = W[:, 0:VD:2, :] * scale      # [VN, 36, VH] -> lo half lanes 0..35
    W_odd = W[:, 1:VD:2, :] * scale       # [VN, 36, VH] -> hi half lanes 64..99
    b_even = b[:, 0:VD:2] * scale
    b_odd = b[:, 1:VD:2] * scale
    Wp = torch.zeros(VN, VD_PAD, VH, dtype=torch.bfloat16)
    bp = torch.zeros(VN, VD_PAD, dtype=torch.bfloat16)
    Wp[:, :half, :] = W_even.to(torch.bfloat16)
    Wp[:, hp:hp + half, :] = W_odd.to(torch.bfloat16)
    bp[:, :half] = b_even.to(torch.bfloat16)
    bp[:, hp:hp + half] = b_odd.to(torch.bfloat16)
    return Wp.reshape(VN * VD_PAD, VH).contiguous(), bp.reshape(-1).contiguous()


def _o_pad(Wo, VN, VD):
    """o-proj weight [VH, VN*VD] -> [VH, VN*VD_PAD]: each head's VD input cols placed in
    the first VD of its 128-block, pad cols zero. Lets o-proj consume the 128-padded
    attention output directly (so no separate 72-trim / unaligned permute is needed)."""
    VH = Wo.shape[0]
    Wo = Wo.reshape(VH, VN, VD)
    Wp = torch.zeros(VH, VN, VD_PAD, dtype=torch.bfloat16)
    Wp[:, :, :VD] = Wo.to(torch.bfloat16)
    return Wp.reshape(VH, VN * VD_PAD).contiguous()


def _v_pad(W, b, VN, VD):
    """[VN*VD, VH] v weight -> [VN*VD_PAD, VH] natural order, zero-padded 72->128."""
    VH = W.shape[1]
    W = W.reshape(VN, VD, VH)
    b = b.reshape(VN, VD)
    Wp = torch.zeros(VN, VD_PAD, VH, dtype=torch.bfloat16)
    bp = torch.zeros(VN, VD_PAD, dtype=torch.bfloat16)
    Wp[:, :VD, :] = W.to(torch.bfloat16)
    bp[:, :VD] = b.to(torch.bfloat16)
    return Wp.reshape(VN * VD_PAD, VH).contiguous(), bp.reshape(-1).contiguous()


def _row_pad(W, b, target_rows):
    """Pad a [out, in] linear's OUTPUT rows (and bias) up to target_rows with zeros."""
    out, inn = W.shape
    Wp = torch.zeros(target_rows, inn, dtype=torch.bfloat16)
    Wp[:out] = W.to(torch.bfloat16)
    bp = torch.zeros(target_rows, dtype=torch.bfloat16)
    bp[:out] = b.to(torch.bfloat16)
    return Wp.contiguous(), bp.contiguous()


def _col_pad(W, target_cols):
    """Pad a [out, in] linear's INPUT cols (K dim) up to target_cols with zeros."""
    out, inn = W.shape
    Wp = torch.zeros(out, target_cols, dtype=torch.bfloat16)
    Wp[:, :inn] = W.to(torch.bfloat16)
    return Wp.contiguous()


def _store_proj(ue, la, name, W, precision):
    """Store one projection weight: quantized (if4) -> la[name_scale],la[name_data];
    or bf16 -> la[name_weight]. K (last dim) is a multiple of 64 for every MoonViT
    projection, so q4_64's 64-blocks never straddle rows."""
    if precision == "if4":
        packed, _ = quantize_q4_64(W)
        la[f'{name}_scale'], la[f'{name}_data'] = store_quantized_weight(ue, packed)
    else:
        la[f'{name}_weight'] = store_weight(ue, W.to(torch.bfloat16).contiguous())


def moonvit_load_weights(ue, model, cfg, precision="if4"):
    """Store all MoonViT transformer weights to params DRAM. `model` is a built
    la.LocateAnything (its .vision_model is the MoonVit module). Big matmul weights are
    if4-quantized by default (~0.28 GB vs ~0.98 GB bf16 — fits the params window);
    biases + LayerNorm gamma/beta stay bf16. Populates ue.mv_layer_addrs / ue.mv_*."""
    d = _vis_dims(cfg)
    VN, VD, VH, VI = d["VN"], d["VD"], d["VH"], d["VI"]
    VIS_MLP_PAD = ((VI + 63) // 64) * 64           # 4352
    q_scale = math.sqrt(VD_PAD / VD)                # sqrt(128/72) flash-scale correction
    ue.mv_precision = precision

    enc = model.vision_model.encoder
    ue.mv_layer_addrs = []
    for blk in enc.blocks:
        la = {}
        # wqkv: Linear(VH, 3*VH) -> rows [q(VH) | k(VH) | v(VH)], each [VN*VD]
        Wqkv = blk.wqkv.weight.data.float()         # [3*VH, VH]
        bqkv = blk.wqkv.bias.data.float()           # [3*VH]
        qW, kW, vW = Wqkv[:VH], Wqkv[VH:2 * VH], Wqkv[2 * VH:]
        qb, kb, vb = bqkv[:VH], bqkv[VH:2 * VH], bqkv[2 * VH:]
        qWp, qbp = _qk_pad_reorder(qW, qb, VN, VD, scale=q_scale)
        kWp, kbp = _qk_pad_reorder(kW, kb, VN, VD, scale=1.0)
        vWp, vbp = _v_pad(vW, vb, VN, VD)
        _store_proj(ue, la, 'q', qWp, precision); la['q_bias'] = store_weight(ue, qbp)
        _store_proj(ue, la, 'k', kWp, precision); la['k_bias'] = store_weight(ue, kbp)
        _store_proj(ue, la, 'v', vWp, precision); la['v_bias'] = store_weight(ue, vbp)
        # o proj: Linear(VH, VH); expand input dim VN*VD -> VN*VD_PAD to consume padded attn
        _store_proj(ue, la, 'o', _o_pad(blk.wo.weight.data.float(), VN, VD), precision)
        la['o_bias'] = store_weight(ue, blk.wo.bias.data.to(torch.bfloat16).contiguous())
        # norms (LayerNorm gamma+beta) — always bf16
        la['norm0_w'] = store_weight(ue, blk.norm0.weight.data.to(torch.bfloat16).contiguous())
        la['norm0_b'] = store_weight(ue, blk.norm0.bias.data.to(torch.bfloat16).contiguous())
        la['norm1_w'] = store_weight(ue, blk.norm1.weight.data.to(torch.bfloat16).contiguous())
        la['norm1_b'] = store_weight(ue, blk.norm1.bias.data.to(torch.bfloat16).contiguous())
        # MLP: fc0 (VH->VI) pad rows VI->VIS_MLP_PAD; fc1 (VI->VH) pad cols VI->VIS_MLP_PAD
        fc0W, fc0b = _row_pad(blk.mlp.fc0.weight.data.float(), blk.mlp.fc0.bias.data.float(), VIS_MLP_PAD)
        fc1W = _col_pad(blk.mlp.fc1.weight.data.float(), VIS_MLP_PAD)
        _store_proj(ue, la, 'fc0', fc0W, precision); la['fc0_bias'] = store_weight(ue, fc0b)
        _store_proj(ue, la, 'fc1', fc1W, precision)
        la['fc1_bias'] = store_weight(ue, blk.mlp.fc1.bias.data.to(torch.bfloat16).contiguous())
        ue.mv_layer_addrs.append(la)

    fln = enc.final_layernorm
    ue.mv_final_ln_w = store_weight(ue, fln.weight.data.to(torch.bfloat16).contiguous())
    ue.mv_final_ln_b = store_weight(ue, fln.bias.data.to(torch.bfloat16).contiguous())
    print(f"  MoonViT weights loaded: {len(ue.mv_layer_addrs)} layers ({precision}); "
          f"params DRAM usage: {ue.get_params_dram_usage()/1048576:.0f} MB")


# =====================================================================================
# DRAM region allocation (sized for a specific token count N)
# =====================================================================================

def moonvit_setup_dram(ue, N, cfg):
    """Allocate MoonViT activation regions for N patch tokens. The sequence length is
    padded up to a multiple of 64 (SKILL rule #1 — flash attention's seq_len drives the
    score-matrix row stride, which must be 128-byte aligned). Padded query rows are
    ignored at the merge; padded KEY columns are masked via the flash BIAS. Recompiled
    per image grid (N varies), so sizing to the actual N keeps the footprint small."""
    d = _vis_dims(cfg)
    VN, VD, VH, VI, merge = d["VN"], d["VD"], d["VH"], d["VI"], d["merge"]
    VIS_MLP_PAD = ((VI + 63) // 64) * 64
    bpe = 2
    N_real = N
    # Pad seq to a multiple of 128 (not just 64): flash stores partial scores with a
    # row stride of N elements, and the tiling appears to want 128-element alignment.
    N = ((N + 127) // 128) * 128                     # padded seq len (128-aligned)
    N_merged = N_real // (merge * merge)
    merge_dim = merge * merge * VH                   # 4*1152 = 4608

    ue.MV_N = N                                      # padded seq len (strides / M / flash)
    ue.MV_N_REAL = N_real                            # real patch count (merge reads these)
    ue.MV_N_MERGED = N_merged
    ue.MV_IO_A = ue.allocate_tensor_dram(N * VH * bpe)
    ue.MV_IO_B = ue.allocate_tensor_dram(N * VH * bpe)
    # zero the layer I/O so padded seq rows start clean (LayerNorm of a zero row -> beta,
    # no NaN; they stay isolated from real rows thereafter).
    io_zeros = torch.zeros(N * VH, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(ue.MV_IO_A, io_zeros)
    ue.dma_to_accelerator_memory(ue.MV_IO_B, io_zeros)
    ue.MV_NORM_OUT = ue.allocate_tensor_dram(N * VH * bpe)
    ue.MV_Q = ue.allocate_tensor_dram(N * VN * VD_PAD * bpe)
    ue.MV_K = ue.allocate_tensor_dram(N * VN * VD_PAD * bpe)
    ue.MV_V = ue.allocate_tensor_dram(N * VN * VD_PAD * bpe)
    ue.MV_Q_PAD = ue.allocate_tensor_dram(VN * N * VD_PAD * bpe)
    ue.MV_K_PAD = ue.allocate_tensor_dram(VN * N * VD_PAD * bpe)
    ue.MV_V_PAD = ue.allocate_tensor_dram(VN * N * VD_PAD * bpe)
    zeros_pad = torch.zeros(VN * N * VD_PAD, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(ue.MV_Q_PAD, zeros_pad)
    ue.dma_to_accelerator_memory(ue.MV_K_PAD, zeros_pad)
    ue.dma_to_accelerator_memory(ue.MV_V_PAD, zeros_pad)
    ue.MV_ATTN_OUT = ue.allocate_tensor_dram(VN * N * VD_PAD * bpe)
    # zero so a NaN read here means flash COMPUTED NaN, not that it failed to write
    ue.dma_to_accelerator_memory(ue.MV_ATTN_OUT, torch.zeros(VN * N * VD_PAD, dtype=torch.bfloat16))
    # debug tap: softmax(QK^T) dump [N, N] (flash debug_mode), zeroed for clean read
    ue.MV_SM_DEBUG = ue.allocate_tensor_dram(N * N * bpe)
    ue.dma_to_accelerator_memory(ue.MV_SM_DEBUG, torch.zeros(N * N, dtype=torch.bfloat16))
    ue.MV_ATTN_SCRATCH = ue.allocate_tensor_dram(
        (max(VD_PAD, UE_FMAX_CONTEXT_SIZE) * N * 2 + VD_PAD * N) * bpe)
    # attention result kept in padded head layout [N, VN*VD_PAD]; o-proj weight zero-pads
    ue.MV_ATTN_RESULT = ue.allocate_tensor_dram(N * VN * VD_PAD * bpe)
    ue.MV_O_PROJ = ue.allocate_tensor_dram(N * VH * bpe)
    ue.MV_RESIDUAL = ue.allocate_tensor_dram(N * VH * bpe)
    ue.MV_MLP_INTER = ue.allocate_tensor_dram(N * VIS_MLP_PAD * bpe)
    ue.MV_MLP_OUT = ue.allocate_tensor_dram(N * VH * bpe)
    ue.MV_POST_NORM = ue.allocate_tensor_dram(N * VH * bpe)
    ue.MV_ROPE = ue.allocate_tensor_dram(N * 2 * VD_PAD * bpe)  # [cos(128)||sin(128)] per token
    ue.MV_ROPE_COS = ue.MV_ROPE
    ue.MV_ROPE_SIN = ue.MV_ROPE + VD_PAD * bpe
    ue.MV_PERM_TEMP = ue.allocate_tensor_dram(VN * N * VD_PAD * bpe * 2)
    # Flash BIAS [N, N] row-major (added to QK^T before softmax): mask padded KEY
    # columns (>= N_real) with a large negative so real queries ignore them. Same mask
    # for every head/layer (depends only on key index).
    ue.MV_ATTN_BIAS = ue.allocate_tensor_dram(N * N * bpe)
    bias = torch.zeros(N, N, dtype=torch.bfloat16)
    if N_real < N:
        bias[:, N_real:] = -1e38   # proven mask value (qwen/smolvlm flash bias); NOT -30000
    ue.dma_to_accelerator_memory(ue.MV_ATTN_BIAS, bias.flatten())
    # identity for the permute core (reuse inherited VIS_PERMUTE_PARAMS_DRAM if present)
    if not hasattr(ue, "VIS_PERMUTE_PARAMS_DRAM"):
        permute_size = UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe
        ue.VIS_PERMUTE_PARAMS_DRAM = ue.get_params_dram_addr()
        ue.dma_write(DMA_DEVICE_H2C, ue.VIS_PERMUTE_PARAMS_DRAM,
                     torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16), permute_size)
        ue.allocate_params_dram(permute_size)
    # merged output [N_merged, 4608] -> overwrite VIS_MERGED_DRAM so run_connector picks it up
    ue.MV_MERGED = ue.allocate_tensor_dram(N_merged * merge_dim * bpe)
    ue.VIS_MERGED_DRAM = ue.MV_MERGED
    print(f"  MoonViT DRAM allocated for N={N} (N_merged={N_merged}, merge_dim={merge_dim})")


# =====================================================================================
# Host-side input prep: patch embedding (+pos emb) and the 2D-RoPE cos/sin table
# =====================================================================================

def moonvit_prepare_input(ue, model, pixel_values, grid_hw, cfg, device="cuda"):
    """Run patch_embed (conv + learnable 2D-interp pos-emb) and build the rope table on
    host; DMA both to the board. Returns N (token count)."""
    d = _vis_dims(cfg)
    VD = d["VD"]
    N_real = grid_hw[0] * grid_hw[1]
    N = ue.MV_N                       # padded (64-aligned) seq len; rope/patch sized to it
    half = VD // 2  # 36

    with torch.no_grad():
        # patch_embed already adds the interpolated learnable pos-emb -> [N_real, VH]
        patch = model.vision_model.patch_embed(
            pixel_values.to(device, dtype=torch.bfloat16), grid_hw).to(torch.bfloat16).cpu()
        assert patch.shape[0] == N_real, f"patch tokens {patch.shape[0]} != N_real={N_real}"
        # 2D RoPE freqs_cis: [N_real, half] complex (interleaved x/y per the reference)
        fc = model.vision_model.encoder.rope_2d.freqs_cis(grid_hw, torch.device(device))
        fc = fc.cpu()  # [N_real, half] complex64
        cos_raw = fc.real.to(torch.bfloat16)  # [N_real, half]
        sin_raw = fc.imag.to(torch.bfloat16)  # [N_real, half]

    # Fill the padded seq rows with copies of real patch rows, NOT zeros: a zero row has
    # zero variance, so LayerNorm does (0-0)/sqrt(0+eps) -> 0*inf = NaN, which then
    # poisons q/k/v -> V[pad] -> the whole attention output. Pad rows are masked out as
    # keys, so their (finite) content never affects the real tokens.
    pad = N - N_real
    if pad > 0:
        reps = (pad + N_real - 1) // N_real
        patch_full = torch.cat([patch] + [patch[:N_real]] * reps, dim=0)[:N]
    else:
        patch_full = patch
    ue.dma_to_accelerator_memory(ue.MV_IO_A, patch_full.contiguous().flatten())

    # Pad-128 rotate_half table (matches the validated vision_rope_hf_core_dram_test):
    # cos lanes [0:half] and [64:64+half] (gaps = 1.0 identity); sin lo pre-negated. Pad
    # seq rows (N_real..N) get identity rope (cos=1, sin=0) — they are masked-out keys.
    rope_table = torch.zeros(N, 2, VD_PAD, dtype=torch.bfloat16)
    cos_t = rope_table[:, 0, :]
    sin_t = rope_table[:, 1, :]
    cos_t.fill_(1.0)
    cos_t[:N_real, :half] = cos_raw
    cos_t[:N_real, 64:64 + half] = cos_raw
    sin_t[:N_real, :half] = -sin_raw
    sin_t[:N_real, 64:64 + half] = sin_raw
    ue.dma_to_accelerator_memory(ue.MV_ROPE, rope_table.contiguous().flatten())
    print(f"  MoonViT patch[{N_real},{patch.shape[1]}] + rope[{N},2,{VD_PAD}] "
          f"(seq padded {N_real}->{N}) DMA'd")
    return N


# =====================================================================================
# Compile the encoder program (single capture stream)
# =====================================================================================

def moonvit_compile_encoder(ue, cfg, grid_hw):
    """Capture the full MoonViT transformer stack + final norm + 2x2 merge into one
    instruction stream. Returns the program DRAM address. Static M=N (recompiled per
    grid), so no PBI M-register games — correctness first."""
    d = _vis_dims(cfg)
    depth, VN, VD, VH, VI, merge = d["depth"], d["VN"], d["VD"], d["VH"], d["VI"], d["merge"]
    VIS_MLP_PAD = ((VI + 63) // 64) * 64
    bpe = 2
    N = ue.MV_N
    N_merged = ue.MV_N_MERGED
    head_stride_pad = N * VD_PAD * bpe
    VN_VD_PAD = VN * VD_PAD

    from user_dma_core import TYPE
    prec = getattr(ue, "mv_precision", "if4")

    def _mm(M, K, Nn, A, name, la, OUT, bias=None, gelu=False):
        """Precision-aware matmul: if4 (quantized B + scale) or bf16."""
        if prec == "if4":
            ue.matmat_mul_core(M=M, K=K, N=Nn, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f'{name}_data'],
                               OUTPUT_DRAM_ADDR=OUT, C_DRAM_ADDR=bias, bias_mode="broadcast_N",
                               is_B_quantized=True, data_type=TYPE.IF4,
                               SCALE_DRAM_ADDR=la[f'{name}_scale'], gelu_enable=gelu)
        else:
            ue.matmat_mul_core(M=M, K=K, N=Nn, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f'{name}_weight'],
                               OUTPUT_DRAM_ADDR=OUT, C_DRAM_ADDR=bias, bias_mode="broadcast_N",
                               gelu_enable=gelu)

    ue.start_capture()

    # Runtime row-count register for the PBI rope core only (matmul/norm/flash tile or
    # unroll fine at static M, but legacy rope unrolls PER ROW -> instruction blow-up for
    # N*VN*2*depth calls). Primed once with N; the N>=128 PBI rope is the validated path
    # (user_hw_test.py vision_rope_hf_core_dram_test).
    ue.reset_isa_reg_counter()
    vis_M_reg = ue.alloc_isa_reg()
    ue.generate_instruction_add_set(vis_M_reg, N)

    for li in range(depth):
        la = ue.mv_layer_addrs[li]
        h_in = ue.MV_IO_A if li % 2 == 0 else ue.MV_IO_B
        h_out = ue.MV_IO_B if li % 2 == 0 else ue.MV_IO_A

        # --- norm0 (LayerNorm) ---
        ue.layer_norm_core_dram(M=N, N=VH, A_DRAM_ADDR=h_in, OUTPUT_DRAM_ADDR=ue.MV_NORM_OUT,
                                GAMMA_DRAM_ADDR=la['norm0_w'], BETA_DRAM_ADDR=la['norm0_b'])

        # --- q/k/v projections into per-head-128-padded layout [N, VN*128] ---
        for proj, dst in [('q', ue.MV_Q), ('k', ue.MV_K), ('v', ue.MV_V)]:
            _mm(N, VH, VN_VD_PAD, ue.MV_NORM_OUT, proj, la, dst, bias=la[f'{proj}_bias'])

        # --- permute [N, VN, 128] -> [VN, N, 128] for each of q/k/v ---
        for src, dst in [(ue.MV_Q, ue.MV_Q_PAD), (ue.MV_K, ue.MV_K_PAD), (ue.MV_V, ue.MV_V_PAD)]:
            smart_bf16_permute_core(ue, dims=[N, VN, VD_PAD], permute_indices=[1, 0, 2],
                                    input_dram_addr=src, output_dram_addr=dst,
                                    params_dram_addr=ue.VIS_PERMUTE_PARAMS_DRAM,
                                    temp_dram_start=ue.MV_PERM_TEMP)

        # --- 2D RoPE on q and k (pad-128 rotate_half kernel; cos/sin shared across heads) ---
        for h in range(VN):
            for buf in (ue.MV_Q_PAD, ue.MV_K_PAD):
                addr = buf + h * head_stride_pad
                ue.rope_hf_core_dram(M=N, N=VD_PAD, input_dram_addr=addr, output_dram_addr=addr,
                                     cos_dram_addr=ue.MV_ROPE_COS, sin_dram_addr=ue.MV_ROPE_SIN,
                                     gpr_M_reg=vis_M_reg)

        # --- per-head full bidirectional flash attention (head_dim=128) ---
        for h in range(VN):
            ue._flash_attention_core_cached(
                head_dim=VD_PAD, seq_len=N,
                Q_DRAM_ADDR=ue.MV_Q_PAD + h * head_stride_pad,
                K_DRAM_ADDR=ue.MV_K_PAD + h * head_stride_pad,
                V_DRAM_ADDR=ue.MV_V_PAD + h * head_stride_pad,
                OUTPUT_DRAM_ADDR=ue.MV_ATTN_OUT + h * head_stride_pad,
                SCRATCH_DRAM_ADDR=ue.MV_ATTN_SCRATCH,
                BIAS_DRAM_ADDR=ue.MV_ATTN_BIAS)  # mask padded key columns

        # --- inverse permute [VN, N, 128] -> [N, VN*128] (padded; aligned last dim) ---
        smart_bf16_permute_core(ue, dims=[VN, N, VD_PAD], permute_indices=[1, 0, 2],
                                input_dram_addr=ue.MV_ATTN_OUT, output_dram_addr=ue.MV_ATTN_RESULT,
                                params_dram_addr=ue.VIS_PERMUTE_PARAMS_DRAM,
                                temp_dram_start=ue.MV_PERM_TEMP)

        # --- o proj (K=VN*128, pad cols zeroed in weight) + residual + norm1 fused ---
        _mm(N, VN_VD_PAD, VH, ue.MV_ATTN_RESULT, 'o', la, ue.MV_O_PROJ, bias=la['o_bias'])
        layer_norm_core_dram_post_add(ue, M=N, N=VH,
                                      A_DRAM_ADDR=h_in, B_DRAM_ADDR=ue.MV_O_PROJ,
                                      ADDOUTPUT_DRAM_ADDR=ue.MV_RESIDUAL,
                                      NORMOUTPUT_DRAM_ADDR=ue.MV_NORM_OUT,
                                      GAMMA_DRAM_ADDR=la['norm1_w'], BETA_DRAM_ADDR=la['norm1_b'])

        # --- MLP: fc0 + gelu, fc1, residual add ---
        _mm(N, VH, VIS_MLP_PAD, ue.MV_NORM_OUT, 'fc0', la, ue.MV_MLP_INTER,
            bias=la['fc0_bias'], gelu=True)
        _mm(N, VIS_MLP_PAD, VH, ue.MV_MLP_INTER, 'fc1', la, ue.MV_MLP_OUT, bias=la['fc1_bias'])
        eltwise_add_core_dram(ue, size=N * VH, A_DRAM_ADDR=ue.MV_RESIDUAL,
                              B_DRAM_ADDR=ue.MV_MLP_OUT, OUTPUT_DRAM_ADDR=h_out)

    # --- final layernorm ---
    final_vis = ue.MV_IO_A if depth % 2 == 0 else ue.MV_IO_B
    ue.layer_norm_core_dram(M=N, N=VH, A_DRAM_ADDR=final_vis, OUTPUT_DRAM_ADDR=ue.MV_POST_NORM,
                            GAMMA_DRAM_ADDR=ue.mv_final_ln_w, BETA_DRAM_ADDR=ue.mv_final_ln_b)

    # --- 2x2 patch merge [gh,gw,VH] -> [N_merged, 4*VH] ---
    nh, nw = grid_hw[0] // merge, grid_hw[1] // merge
    smart_bf16_permute_core(ue, dims=[nh, merge, nw, merge, VH], permute_indices=[0, 2, 1, 3, 4],
                            input_dram_addr=ue.MV_POST_NORM, output_dram_addr=ue.MV_MERGED,
                            params_dram_addr=ue.VIS_PERMUTE_PARAMS_DRAM,
                            temp_dram_start=ue.MV_PERM_TEMP)

    ue.stop_capture()
    ue.generate_instruction_halt()
    prog = bytearray()
    for inst in ue.capture_buffer:
        prog.extend(inst.get_bytes())
    program_addr = ue.get_program_dram_addr()
    ue.dma_write(DMA_DEVICE_H2C, program_addr, prog, len(prog))
    ue.allocate_program_dram(len(prog))
    ue.clear_capture_buffer()
    print(f"  MoonViT encoder compiled: {len(prog)} bytes, {depth} layers (single stream)")
    return program_addr


def moonvit_run_encoder(ue, program_addr, timeout_s=300.0):
    """Execute the compiled MoonViT program. Output [N_merged, 4608] lands in
    ue.VIS_MERGED_DRAM, ready for run_connector."""
    ue.start_execute_from_dram(program_addr)
    ue.wait_queue(timeout_s)
    print(f"  MoonViT encoder executed -> VIS_MERGED_DRAM 0x{ue.VIS_MERGED_DRAM:X}")


# =====================================================================================
# Staged (op-by-op) debug runner — execute one op, read it back, NaN-check, repeat.
# =====================================================================================

def _step(ue, label, emit, check_addr=None, check_shape=None, real_rows=None,
          stop=False, check=True, timeout_s=120.0):
    """Capture exactly one op group, execute it on hardware immediately, then read the
    result back and assert it is NaN/Inf-free. The NaN assert IS the movable stop: it
    halts at the first bad stage, so you fix it and re-run to land on the next one.
    Pass stop=True (or --stop-after <label>) to force a halt after a known-good stage."""
    ue.clear_inst_id()
    ue.start_capture()
    emit()
    ue.stop_capture()
    ue.generate_instruction_halt()
    prog = ue.get_program_dram_addr()          # transient: reuse same region each step
    ue.write_captured_instructions_to_dram(prog)
    ue.clear_capture_buffer()
    ue.start_execute_from_dram(prog)
    ue.wait_queue(timeout_s)

    if check and check_addr is not None:
        t = ue.dma_from_accelerator_memory(check_addr, check_shape).float()
        view = t[:real_rows] if real_rows is not None else t
        n = int(torch.isnan(view).sum()); i = int(torch.isinf(view).sum())
        mx = float(view.abs().max()) if view.numel() else 0.0
        mean = float(view.abs().mean()) if view.numel() else 0.0
        print(f"  [{label:18s}] {str(tuple(check_shape)):>16s}  nan={n:<6d} inf={i:<4d} "
              f"absmax={mx:.4g} absmean={mean:.4g}")
        assert n == 0 and i == 0, f"NaN/Inf detected at stage '{label}' — fix here"
    else:
        print(f"  [{label:18s}] executed")
    if stop:
        assert False, f"FORCED STOP after stage '{label}' (--stop-after)"


def moonvit_run_staged(ue, cfg, grid_hw, stop_after=None, check=True, timeout_s=120.0,
                       mask_pad=True):
    """Run the encoder ONE op at a time (separate capture+execute per op), NaN-checking
    each output. Mirrors moonvit_compile_encoder's op chain exactly. Output ends in
    ue.VIS_MERGED_DRAM (same as the single-stream path) if it runs to completion.

    Walk it down: leave stop_after=None and it halts at the first NaN/Inf; fix that op,
    re-run, it advances to the next. Or set stop_after='L0.flash' to force-stop earlier."""
    d = _vis_dims(cfg)
    depth, VN, VD, VH, VI, merge = d["depth"], d["VN"], d["VD"], d["VH"], d["VI"], d["merge"]
    VIS_MLP_PAD = ((VI + 63) // 64) * 64
    bpe = 2
    N = ue.MV_N
    N_real = ue.MV_N_REAL
    head_stride_pad = N * VD_PAD * bpe
    VN_VD_PAD = VN * VD_PAD
    from user_dma_core import TYPE
    prec = getattr(ue, "mv_precision", "if4")

    def _mm_emit(M, K, Nn, A, name, la, OUT, bias=None, gelu=False):
        def f():
            if prec == "if4":
                ue.matmat_mul_core(M=M, K=K, N=Nn, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f'{name}_data'],
                                   OUTPUT_DRAM_ADDR=OUT, C_DRAM_ADDR=bias, bias_mode="broadcast_N",
                                   is_B_quantized=True, data_type=TYPE.IF4,
                                   SCALE_DRAM_ADDR=la[f'{name}_scale'], gelu_enable=gelu)
            else:
                ue.matmat_mul_core(M=M, K=K, N=Nn, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f'{name}_weight'],
                                   OUTPUT_DRAM_ADDR=OUT, C_DRAM_ADDR=bias, bias_mode="broadcast_N",
                                   gelu_enable=gelu)
        return f

    def step(label, emit, addr=None, shape=None):
        _step(ue, label, emit, addr, shape, real_rows=N_real,
              stop=(label == stop_after), check=check, timeout_s=timeout_s)

    print(f"=== MoonViT staged run (N={N_real}->pad {N}, prec={prec}, "
          f"stop_after={stop_after}) ===")
    for li in range(depth):
        la = ue.mv_layer_addrs[li]
        h_in = ue.MV_IO_A if li % 2 == 0 else ue.MV_IO_B
        h_out = ue.MV_IO_B if li % 2 == 0 else ue.MV_IO_A
        L = f"L{li}"

        step(f"{L}.norm0",
             lambda h_in=h_in, la=la: ue.layer_norm_core_dram(
                 M=N, N=VH, A_DRAM_ADDR=h_in, OUTPUT_DRAM_ADDR=ue.MV_NORM_OUT,
                 GAMMA_DRAM_ADDR=la['norm0_w'], BETA_DRAM_ADDR=la['norm0_b']),
             ue.MV_NORM_OUT, (N, VH))

        for proj, dst in [('q', ue.MV_Q), ('k', ue.MV_K), ('v', ue.MV_V)]:
            step(f"{L}.{proj}proj",
                 _mm_emit(N, VH, VN_VD_PAD, ue.MV_NORM_OUT, proj, la, dst, bias=la[f'{proj}_bias']),
                 dst, (N, VN_VD_PAD))

        for proj, src, dst in [('q', ue.MV_Q, ue.MV_Q_PAD), ('k', ue.MV_K, ue.MV_K_PAD),
                               ('v', ue.MV_V, ue.MV_V_PAD)]:
            step(f"{L}.perm_{proj}",
                 lambda src=src, dst=dst: smart_bf16_permute_core(
                     ue, dims=[N, VN, VD_PAD], permute_indices=[1, 0, 2],
                     input_dram_addr=src, output_dram_addr=dst,
                     params_dram_addr=ue.VIS_PERMUTE_PARAMS_DRAM, temp_dram_start=ue.MV_PERM_TEMP),
                 dst, (VN * N, VD_PAD))

        def rope_emit(la=la):
            ue.reset_isa_reg_counter()
            r = ue.alloc_isa_reg()
            ue.generate_instruction_add_set(r, N)
            for h in range(VN):
                for buf in (ue.MV_Q_PAD, ue.MV_K_PAD):
                    a = buf + h * head_stride_pad
                    ue.rope_hf_core_dram(M=N, N=VD_PAD, input_dram_addr=a, output_dram_addr=a,
                                         cos_dram_addr=ue.MV_ROPE_COS, sin_dram_addr=ue.MV_ROPE_SIN,
                                         gpr_M_reg=r)
        step(f"{L}.rope", rope_emit, ue.MV_Q_PAD, (VN * N, VD_PAD))

        # Flash PER HEAD, head 0 with the debug tap: dump softmax(QK^T) so we can see
        # whether the NaN is born in QK^T/softmax (SM dump NaN) or in the ×V stage
        # (SM finite, output NaN).
        bias_addr = ue.MV_ATTN_BIAS if mask_pad else None
        for h in range(VN):
            dbg = (h == 0)

            def flash_emit(h=h, dbg=dbg):
                ue._flash_attention_core_cached(
                    head_dim=VD_PAD, seq_len=N,
                    Q_DRAM_ADDR=ue.MV_Q_PAD + h * head_stride_pad,
                    K_DRAM_ADDR=ue.MV_K_PAD + h * head_stride_pad,
                    V_DRAM_ADDR=ue.MV_V_PAD + h * head_stride_pad,
                    OUTPUT_DRAM_ADDR=ue.MV_ATTN_OUT + h * head_stride_pad,
                    SCRATCH_DRAM_ADDR=ue.MV_ATTN_SCRATCH, BIAS_DRAM_ADDR=bias_addr,
                    debug_mode=dbg, SM_OUTPUT_DRAM_ADDR=(ue.MV_SM_DEBUG if dbg else None))

            if dbg:
                # run flash, then check the softmax dump BEFORE the output, no auto-assert,
                # so we see both numbers regardless of which is NaN.
                _step(ue, f"{L}.flash_h{h}", flash_emit, check=False, timeout_s=timeout_s)
                sm = ue.dma_from_accelerator_memory(ue.MV_SM_DEBUG, (N, N)).float()[:N_real, :N_real]
                out = ue.dma_from_accelerator_memory(ue.MV_ATTN_OUT, (N, VD_PAD)).float()[:N_real]
                # V^T sits at the start of SCRATCH after the first (I@V^T) matmul: [head_dim, seq]
                vt = ue.dma_from_accelerator_memory(ue.MV_ATTN_SCRATCH, (VD_PAD, N)).float()
                print(f"  [V^T (scratch) ] {tuple(vt.shape)}  nan={int(torch.isnan(vt).sum())} "
                      f"inf={int(torch.isinf(vt).sum())} absmax={float(vt.abs().max()):.4g}")
                print(f"  [softmax(QK^T)] {tuple(sm.shape)}  nan={int(torch.isnan(sm).sum())} "
                      f"inf={int(torch.isinf(sm).sum())} absmax={float(sm.abs().max()):.4g} "
                      f"rowsum0={float(sm[0].sum()):.4g}")
                print(f"  [attn ×V out  ] {tuple(out.shape)}  nan={int(torch.isnan(out).sum())} "
                      f"inf={int(torch.isinf(out).sum())} absmax={float(out.abs().max()):.4g}")
                assert not (torch.isnan(sm).any() or torch.isnan(out).any()), \
                    "flash NaN — see which of softmax/×V above is bad"
            else:
                step(f"{L}.flash_h{h}", flash_emit, ue.MV_ATTN_OUT + h * head_stride_pad, (N, VD_PAD))

        step(f"{L}.inv_perm",
             lambda: smart_bf16_permute_core(
                 ue, dims=[VN, N, VD_PAD], permute_indices=[1, 0, 2],
                 input_dram_addr=ue.MV_ATTN_OUT, output_dram_addr=ue.MV_ATTN_RESULT,
                 params_dram_addr=ue.VIS_PERMUTE_PARAMS_DRAM, temp_dram_start=ue.MV_PERM_TEMP),
             ue.MV_ATTN_RESULT, (N, VN_VD_PAD))

        step(f"{L}.oproj",
             _mm_emit(N, VN_VD_PAD, VH, ue.MV_ATTN_RESULT, 'o', la, ue.MV_O_PROJ, bias=la['o_bias']),
             ue.MV_O_PROJ, (N, VH))

        step(f"{L}.norm1_postadd",
             lambda h_in=h_in, la=la: layer_norm_core_dram_post_add(
                 ue, M=N, N=VH, A_DRAM_ADDR=h_in, B_DRAM_ADDR=ue.MV_O_PROJ,
                 ADDOUTPUT_DRAM_ADDR=ue.MV_RESIDUAL, NORMOUTPUT_DRAM_ADDR=ue.MV_NORM_OUT,
                 GAMMA_DRAM_ADDR=la['norm1_w'], BETA_DRAM_ADDR=la['norm1_b']),
             ue.MV_NORM_OUT, (N, VH))

        step(f"{L}.fc0_gelu",
             _mm_emit(N, VH, VIS_MLP_PAD, ue.MV_NORM_OUT, 'fc0', la, ue.MV_MLP_INTER,
                      bias=la['fc0_bias'], gelu=True),
             ue.MV_MLP_INTER, (N, VIS_MLP_PAD))

        step(f"{L}.fc1",
             _mm_emit(N, VIS_MLP_PAD, VH, ue.MV_MLP_INTER, 'fc1', la, ue.MV_MLP_OUT, bias=la['fc1_bias']),
             ue.MV_MLP_OUT, (N, VH))

        step(f"{L}.residual",
             lambda h_out=h_out: eltwise_add_core_dram(
                 ue, size=N * VH, A_DRAM_ADDR=ue.MV_RESIDUAL, B_DRAM_ADDR=ue.MV_MLP_OUT,
                 OUTPUT_DRAM_ADDR=h_out),
             h_out, (N, VH))

    final_vis = ue.MV_IO_A if depth % 2 == 0 else ue.MV_IO_B
    step("final_norm",
         lambda: ue.layer_norm_core_dram(
             M=N, N=VH, A_DRAM_ADDR=final_vis, OUTPUT_DRAM_ADDR=ue.MV_POST_NORM,
             GAMMA_DRAM_ADDR=ue.mv_final_ln_w, BETA_DRAM_ADDR=ue.mv_final_ln_b),
         ue.MV_POST_NORM, (N, VH))

    nh, nw = grid_hw[0] // merge, grid_hw[1] // merge
    step("merge",
         lambda: smart_bf16_permute_core(
             ue, dims=[nh, merge, nw, merge, VH], permute_indices=[0, 2, 1, 3, 4],
             input_dram_addr=ue.MV_POST_NORM, output_dram_addr=ue.MV_MERGED,
             params_dram_addr=ue.VIS_PERMUTE_PARAMS_DRAM, temp_dram_start=ue.MV_PERM_TEMP),
         ue.MV_MERGED, (ue.MV_N_MERGED, merge * merge * VH))
    print("  MoonViT staged run complete — no NaN/Inf through all stages.")
