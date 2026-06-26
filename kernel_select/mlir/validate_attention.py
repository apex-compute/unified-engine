#!/usr/bin/env python3
"""validate_attention.py — execute the WINDOWED-ATTENTION emitter on the FPGA and
SNR it against torch. This closes the one gap in the Swin emitter body: the flash
+ permute cluster the walker still _DEFER's.

The emitter `emit_windowed_attention()` is the reusable compiler stage — it is
parameterized by AttentionSpec (NOT Swin-hardcoded), so the same staging skeleton
will serve the LLM path with different knobs (causal/kv-cache/GQA). It ports the
proven golden sequence (swin_test.py steps 4-9): qkv proj -> pad+permute to heads
-> Q prescale -> batched flash (PBI subroutine) -> inverse permute -> unpad -> out
proj. All helpers already exist; this is assembly, not new kernels.

Run:  python validate_attention.py
"""
import os
import sys
import math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))               # kernel_select/
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "models", "swin"))

import torch
import torch.nn.functional as F

import nn_lib
# golden staging helpers (module-level fns in swin_test.py)
from nn_lib import (multihead_pad_and_permute, dram_zero_fill, pbi_strided_copy)
from user_dma_core import UE_VECTOR_SIZE, UE_FMAX_CONTEXT_SIZE
from loom_plan import AttentionSpec


# =============================================================================
# small replicas of the Swin model's address/gpr helpers, on a bare engine
# =============================================================================
def _alloc(ue, n_elems):
    return ue.allocate_tensor_dram(n_elems * 2)


def _alloc_param(ue, t):
    t = t.to(torch.bfloat16).contiguous()
    addr = ue.get_params_dram_addr()
    ue.allocate_params_dram(t.numel() * 2)
    ue.dma_to_accelerator_memory(addr, t)
    return addr


def _pbi_set(ue, gpr, val):
    ue._isa_reg_counter = gpr + 1
    ue.reset_inst_ptr_counter()
    ue.generate_instruction_add_set(dst_reg_idx=gpr, immediate_value=val)


# =============================================================================
# THE EMITTER — windowed attention, parameterized by AttentionSpec
# =============================================================================
def emit_windowed_attention(ue, spec, *, in_addr, w, bias_addr, identity_addr,
                            gpr_mm, gpr_pbi, alloc):
    """Emit steps 4-9 of windowed attention into the current capture.

    in_addr : windowed tokens (num_windows*wa, dim) already partitioned
    w       : dict of weight addrs: q/k/v_weight, q/k/v_bias, unpad_weight,
              out_weight, out_bias
    returns : out_addr holding (num_windows*wa, dim) = attention block output
    """
    nh, hd = spec.num_q_heads, spec.head_dim
    hd_pad = spec.head_dim_pad                  # 32 -> 64
    wa = spec.seq_len                           # window area (144)
    wa_pad = spec.seq_pad                       # 192
    nw = spec.batch                             # num_windows
    dim = nh * hd
    M_win = nw * wa
    total_batches = nw * nh

    # scratch buffers (exact sizes from swin_test alloc block)
    q_proj = alloc(M_win * dim); k_proj = alloc(M_win * dim); v_proj = alloc(M_win * dim)
    q_heads = alloc(total_batches * wa_pad * hd_pad)
    k_heads = alloc(total_batches * wa_pad * hd_pad)
    v_heads = alloc(total_batches * wa_pad * hd_pad)
    attn_output = alloc(total_batches * wa_pad * hd_pad)
    attn_scratch = alloc((hd_pad + UE_FMAX_CONTEXT_SIZE) * wa_pad)
    attn_p = alloc(wa_pad * wa_pad)
    fs_q = alloc(wa_pad * hd_pad); fs_k = alloc(wa_pad * hd_pad)
    fs_v = alloc(wa_pad * hd_pad); fs_o = alloc(wa_pad * hd_pad)
    fs_bias = alloc(wa_pad * wa_pad)
    permute_temp = alloc(2 * wa * nh * hd_pad)
    attn_permuted = alloc(M_win * nh * hd_pad)
    attn_unpadded = alloc(M_win * dim)
    out_addr = alloc(M_win * dim)

    # --- 4. Q/K/V projections (windowed flat) ---
    for nm, dst in [("q", q_proj), ("k", k_proj), ("v", v_proj)]:
        _pbi_set(ue, gpr_mm, M_win)
        ue.matmat_mul_core(M=M_win, K=dim, N=dim,
                           A_DRAM_ADDR=in_addr, B_DRAM_ADDR=w[f"{nm}_weight"],
                           OUTPUT_DRAM_ADDR=dst, C_DRAM_ADDR=w[f"{nm}_bias"],
                           bias_mode="broadcast_N", gpr_M_reg=gpr_mm)

    # --- 5. zero-fill + pad + permute Q/K/V to multi-head (total_batches, wa_pad, hd_pad) ---
    for proj, heads in [(q_proj, q_heads), (k_proj, k_heads), (v_proj, v_heads)]:
        dram_zero_fill(ue, heads, total_batches * wa_pad * hd_pad)
        multihead_pad_and_permute(ue, INPUT_DRAM_ADDR=proj, OUTPUT_DRAM_ADDR=heads,
                                  TEMP_DRAM_ADDR=permute_temp, num_windows=nw, wa=wa,
                                  num_heads=nh, head_dim=hd, head_dim_pad=hd_pad,
                                  wa_pad=wa_pad, gpr_pbi=gpr_pbi)

    # --- 5b. pre-scale Q so flash's /sqrt(hd_pad) becomes /sqrt(hd) ---
    scale = spec.q_prescale
    total_q = total_batches * wa_pad * hd_pad
    CHUNK = 262080
    for off in range(0, total_q, CHUNK):
        take = min(CHUNK, total_q - off)
        ue.accelerator_memory_to_sram(accelerator_dram_address=q_heads + off * 2,
                                      sram_address=0x00000, element_size=take)
        ue.broadcast_mul(scalar=scale, sram_start_addr=0x00000,
                         sram_wb_addr=0x00000, element_size=take)
        ue.sram_to_accelerator_memory(sram_address=0x00000,
                                      accelerator_dram_address=q_heads + off * 2,
                                      element_size=take)

    # --- 6. batched flash attention (static subroutine + per-window staging) ---
    dram_zero_fill(ue, attn_output, total_batches * wa_pad * hd_pad)
    nn_lib.flash_attention_batched_pbi(ue, num_batches=total_batches,
        head_dim=hd_pad, seq_len=wa_pad,
        Q_DRAM_ADDR=q_heads, K_DRAM_ADDR=k_heads, V_DRAM_ADDR=v_heads,
        OUTPUT_DRAM_ADDR=attn_output, SCRATCH_DRAM_ADDR=attn_scratch,
        ATTN_P_DRAM_ADDR=attn_p, IDENTITY_TRANSPOSE_DRAM_ADDR=identity_addr,
        BIAS_DRAM_ADDR=bias_addr,
        STAGE_Q_DRAM_ADDR=fs_q, STAGE_K_DRAM_ADDR=fs_k, STAGE_V_DRAM_ADDR=fs_v,
        STAGE_O_DRAM_ADDR=fs_o, STAGE_BIAS_DRAM_ADDR=fs_bias,
        _silent=True, gpr_M_reg=gpr_mm)

    # --- 7. inverse permute: (nh, wa_pad, hd_pad) -> token-major (wa, nh, hd_pad), drop wa_pad ---
    head_bytes = hd_pad * 2
    row_stride = nh * hd_pad * 2
    for wi in range(nw):
        src_w = attn_output + wi * nh * wa_pad * hd_pad * 2
        dst_w = attn_permuted + wi * wa * nh * hd_pad * 2
        for h in range(nh):
            pbi_strided_copy(ue, gpr_pbi,
                             src_base=src_w + h * wa_pad * hd_pad * 2,
                             dst_base=dst_w + h * head_bytes,
                             count=wa, copy_bytes=head_bytes,
                             src_stride=head_bytes, dst_stride=row_stride)

    # --- 8. unpad matmul (nh*hd_pad -> dim, selects real columns) ---
    _pbi_set(ue, gpr_mm, M_win)
    ue.matmat_mul_core(M=M_win, K=nh * hd_pad, N=dim,
                       A_DRAM_ADDR=attn_permuted, B_DRAM_ADDR=w["unpad_weight"],
                       OUTPUT_DRAM_ADDR=attn_unpadded, gpr_M_reg=gpr_mm)

    # --- 9. output projection ---
    _pbi_set(ue, gpr_mm, M_win)
    ue.matmat_mul_core(M=M_win, K=dim, N=dim,
                       A_DRAM_ADDR=attn_unpadded, B_DRAM_ADDR=w["out_weight"],
                       OUTPUT_DRAM_ADDR=out_addr, C_DRAM_ADDR=w["out_bias"],
                       bias_mode="broadcast_N", gpr_M_reg=gpr_mm)
    return out_addr


# =============================================================================
# torch reference — plain windowed MHSA (zero bias), window dim as batch
# =============================================================================
def torch_ref(x, wt, nh, hd):
    B, wa, dim = x.shape
    def proj(W, b):
        return (x @ W.T + b).view(B, wa, nh, hd).permute(0, 2, 1, 3)
    q, k, v = proj(wt["q_w"], wt["q_b"]), proj(wt["k_w"], wt["k_b"]), proj(wt["v_w"], wt["v_b"])
    a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(hd), dim=-1) @ v   # [B,nh,wa,hd]
    a = a.permute(0, 2, 1, 3).reshape(B, wa, dim)
    return a @ wt["o_w"].T + wt["o_b"]


def snr_db(ref, hw):
    diff = (ref - hw).abs()
    return float(20 * torch.log10(ref.abs().mean() / (diff.mean() + 1e-10)))


def validate(nh=6, hd=32, wa=144, nw=1):
    torch.manual_seed(0)
    dim = nh * hd
    hd_pad = ((hd + 63) // 64) * 64
    spec = AttentionSpec(kind="windowed", num_q_heads=nh, num_kv_heads=nh,
                         head_dim=hd, seq_len=wa, batch=nw, bias="none")
    print(f"[attn] {spec}")

    # torch weights (separate q/k/v to match the staging)
    x = torch.randn(nw, wa, dim)
    wt = {f"{p}_{s}": (torch.randn(dim, dim) * 0.05 if s == "w" else torch.zeros(dim))
          for p in ("q", "k", "v", "o") for s in ("w", "b")}
    ref = torch_ref(x, wt, nh, hd)

    from user_dma_core import UnifiedEngine
    ue = UnifiedEngine()

    # --- load weights + constants ---
    w = {}
    for p, key in [("q", "q"), ("k", "k"), ("v", "v"), ("o", "out")]:
        w[f"{key}_weight"] = _alloc_param(ue, wt[f"{p}_w"])
        w[f"{key}_bias"] = _alloc_param(ue, wt[f"{p}_b"])
    # unpad selection: [dim, nh*hd_pad], picks real hd columns out of each hd_pad block
    unpad = torch.zeros(dim, nh * hd_pad)
    for h in range(nh):
        for d in range(hd):
            unpad[h * hd + d, h * hd_pad + d] = 1.0
    w["unpad_weight"] = _alloc_param(ue, unpad)
    identity_addr = _alloc_param(ue, torch.eye(UE_VECTOR_SIZE))
    # mask bias: padded key positions (wa..wa_pad) get -inf so softmax kills them.
    # Without this they each add exp(0)=1 to the denominator -> output * wa/wa_pad.
    wa_pad = ((wa + 63) // 64) * 64
    mask = torch.zeros(nw * nh, wa_pad, wa_pad)
    mask[:, :, wa:] = -1e4
    bias_addr = _alloc_param(ue, mask)
    in_addr = _alloc_param(ue, x.reshape(nw * wa, dim))

    gpr_mm = ue.alloc_isa_reg()
    gpr_pbi = ue.alloc_isa_reg()

    # --- compile: capture the program ---
    prog_addr = ue.get_program_dram_addr()
    ue.clear_inst_id()
    ue.start_capture()
    out_addr = emit_windowed_attention(ue, spec, in_addr=in_addr, w=w,
                                       bias_addr=bias_addr, identity_addr=identity_addr,
                                       gpr_mm=gpr_mm, gpr_pbi=gpr_pbi, alloc=lambda n: _alloc(ue, n))
    ue.generate_instruction_halt()
    ue.stop_capture()
    size = ue.write_captured_instructions_to_dram(prog_addr)
    ue.allocate_program_dram(size)
    print(f"[attn] captured program: {size/1024:.1f} KB")

    # --- execute + read back ---
    ue.program_execute(prog_addr)
    hw = ue.dma_from_accelerator_memory(out_addr, (nw * wa, dim)).float().reshape(nw, wa, dim)

    s = snr_db(ref, hw)
    print(f"[attn] output {tuple(ref.shape)}  SNR = {s:.2f} dB  [{'PASS' if s > 19 else 'FAIL'}]")
    print(f"       ref[0,0,:4] = {ref[0,0,:4].tolist()}")
    print(f"       hw [0,0,:4] = {hw[0,0,:4].tolist()}")
    return s


if __name__ == "__main__":
    validate()
