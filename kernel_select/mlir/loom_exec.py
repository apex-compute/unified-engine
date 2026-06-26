#!/usr/bin/env python3
"""loom_exec.py — the EXECUTOR: ue MLIR op stream -> one captured FPGA program ->
run -> read back hidden state. This is the back half of the compiler (the front
half is torch_to_ue.lower). It is HOST/DEVICE aware: every op is classified
DEVICE (a ue kernel), VIEW (free reshape/select alias), MACRO (the attention
cluster -> emit_windowed_attention), or HOST (no kernel -> stays on CPU/torch).

A full Swin block (the 22-op body) runs here end-to-end on hardware, reusing the
validated windowed-attention emitter + the golden norm/mlp/residual primitives.
"""
import os
import sys
import math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "models", "swin"))

import torch

from validate_attention import emit_windowed_attention, _alloc, _alloc_param, _pbi_set, snr_db
from nn_lib import window_partition_dram, window_reverse_dram, pbi_strided_copy
from loom_plan import AttentionSpec
from user_dma_core import URAM_NEAR_FULL_ELEMENTS, UE_VECTOR_SIZE


# =============================================================================
# host/device classification — the compiler is AWARE of what runs where
# =============================================================================
DEVICE_OPS = {"layer_norm", "matmul", "eltwise", "window_partition",
              "window_reverse", "patching", "layer_norm_post_add"}
VIEW_OPS = {"reshape", "select", "permute"}          # free / absorbed
MACRO_OPS = {"flash_attn_pf"}                          # attention cluster anchor


def classify(ops):
    """Return {ssa: role}. role in DEVICE|VIEW|MACRO|HOST. HOST = no kernel ->
    the executor leaves it on the CPU (torch) and DMAs across the boundary."""
    roles = {}
    for o in ops:
        op = o["op"]
        if op in DEVICE_OPS:
            roles[o["res"]] = "DEVICE"
        elif op in MACRO_OPS:
            roles[o["res"]] = "MACRO"
        elif op in VIEW_OPS:
            roles[o["res"]] = "VIEW"
        else:
            roles[o["res"]] = "HOST"          # unsupported on device -> CPU fallback
    return roles


def partition_report(ops, roles):
    from collections import Counter
    c = Counter(roles[o["res"]] for o in ops)
    print(f"[exec] host/device partition: "
          f"DEVICE={c['DEVICE']}  MACRO={c['MACRO']}  VIEW={c['VIEW']}  HOST={c['HOST']}")
    if c["HOST"]:
        hosts = sorted({o["op"] for o in ops if roles[o["res"]] == "HOST"})
        print(f"[exec]   HOST (run on CPU/torch, not device): {hosts}")
    return c


# =============================================================================
# device primitives (raw ue, mirroring the golden swin_test sequence)
# =============================================================================
def _ln_chunks(M, N):
    ops_per_row = 6
    ideal = min(URAM_NEAR_FULL_ELEMENTS // N, M, (256 - 4) // ops_per_row)
    cs = ideal
    while M % cs != 0:
        cs -= 1
    return M // cs


def _sram_add(ue, a_addr, b_addr, out_addr, n):
    chunk = min(URAM_NEAR_FULL_ELEMENTS, n)
    for off in range(0, n, chunk):
        take = min(chunk, n - off)
        ue.accelerator_memory_to_sram(accelerator_dram_address=a_addr + off * 2,
                                      sram_address=0x00000, element_size=take)
        ue.accelerator_memory_to_sram(accelerator_dram_address=b_addr + off * 2,
                                      sram_address=0x80000, element_size=take)
        ue.eltwise_add_core(vector_A_sram_start_addr=0x00000,
                            vector_B_sram_start_addr=0x80000,
                            vector_C_sram_wb_addr=0x00000, element_size=take)
        ue.sram_to_accelerator_memory(sram_address=0x00000,
                                      accelerator_dram_address=out_addr + off * 2,
                                      element_size=take)


# =============================================================================
# emit ONE full Swin block (norm -> partition -> attn macro -> reverse ->
# residual -> norm -> mlp -> residual) into the current capture.
# =============================================================================
def emit_swin_block(ue, W, spec, *, x_addr, H, Wd, dim, mlp_dim, ws,
                    gpr_mm, gpr_ln, gpr_pbi, alloc, consts):
    M = H * Wd
    # 1. norm1
    ln1 = alloc(M * dim)
    _pbi_set(ue, gpr_ln, _ln_chunks(M, dim))
    ue.layer_norm_core_dram(M=M, N=dim, A_DRAM_ADDR=x_addr, OUTPUT_DRAM_ADDR=ln1,
                            GAMMA_DRAM_ADDR=W["n1_g"], BETA_DRAM_ADDR=W["n1_b"],
                            gpr_M_reg=gpr_ln)
    # 2. window partition
    windowed = alloc(spec.batch * spec.seq_len * dim)
    window_partition_dram(ue, INPUT_DRAM_ADDR=ln1, OUTPUT_DRAM_ADDR=windowed,
                          H=H, W=Wd, C=dim, window_size=ws)
    # 3. attention macro (validated windowed-attention emitter)
    attn = emit_windowed_attention(ue, spec, in_addr=windowed, w=W["attn"],
                                   bias_addr=consts["bias"], identity_addr=consts["identity"],
                                   gpr_mm=gpr_mm, gpr_pbi=gpr_pbi, alloc=alloc)
    # 4. window reverse
    rev = alloc(M * dim)
    window_reverse_dram(ue, INPUT_DRAM_ADDR=attn, OUTPUT_DRAM_ADDR=rev,
                        H=H, W=Wd, C=dim, window_size=ws)
    # 5. residual 1
    r1 = alloc(M * dim)
    _sram_add(ue, x_addr, rev, r1, M * dim)
    # 6. norm2
    ln2 = alloc(M * dim)
    _pbi_set(ue, gpr_ln, _ln_chunks(M, dim))
    ue.layer_norm_core_dram(M=M, N=dim, A_DRAM_ADDR=r1, OUTPUT_DRAM_ADDR=ln2,
                            GAMMA_DRAM_ADDR=W["n2_g"], BETA_DRAM_ADDR=W["n2_b"],
                            gpr_M_reg=gpr_ln)
    # 7. fc1 + gelu
    mid = alloc(M * mlp_dim)
    _pbi_set(ue, gpr_mm, M)
    ue.matmat_mul_core(M=M, K=dim, N=mlp_dim, A_DRAM_ADDR=ln2, B_DRAM_ADDR=W["fc1_w"],
                       OUTPUT_DRAM_ADDR=mid, C_DRAM_ADDR=W["fc1_b"],
                       bias_mode="broadcast_N", gelu_enable=True, gpr_M_reg=gpr_mm)
    # 8. fc2
    mout = alloc(M * dim)
    _pbi_set(ue, gpr_mm, M)
    ue.matmat_mul_core(M=M, K=mlp_dim, N=dim, A_DRAM_ADDR=mid, B_DRAM_ADDR=W["fc2_w"],
                       OUTPUT_DRAM_ADDR=mout, C_DRAM_ADDR=W["fc2_b"],
                       bias_mode="broadcast_N", gpr_M_reg=gpr_mm)
    # 9. residual 2
    out = alloc(M * dim)
    _sram_add(ue, r1, mout, out, M * dim)
    return out


# =============================================================================
# patch-merging: 2x2 spatial gather -> LayerNorm(4C) -> Linear(4C->2C). Downsamples
# between stages (H,W halve; C doubles). The gather is the HF cat order
# [x0=(er,ec), x1=(or,ec), x2=(er,oc), x3=(or,oc)] via per-row strided copies.
# =============================================================================
def emit_patch_merge(ue, x_addr, H, Wd, C, *, norm_g, norm_b, reduction_w,
                     gpr_mm, gpr_ln, gpr_pbi, alloc):
    Hh, Wh = H // 2, Wd // 2
    n_out = Hh * Wh
    C4 = 4 * C
    Cb = C * 2                                   # bytes for one C-vector
    merged = alloc(n_out * C4)
    # gather: for each output row i and each 2x2 sub-position, copy Wh C-vectors
    for i in range(Hh):
        for (dr, dc, blk) in [(0, 0, 0), (1, 0, 1), (0, 1, 2), (1, 1, 3)]:
            src = x_addr + ((2 * i + dr) * Wd + dc) * Cb
            dst = merged + (i * Wh) * C4 * 2 + blk * Cb
            pbi_strided_copy(ue, gpr_pbi, src_base=src, dst_base=dst, count=Wh,
                             copy_bytes=Cb, src_stride=2 * Cb, dst_stride=C4 * 2)
    # LayerNorm(4C)
    normed = alloc(n_out * C4)
    _pbi_set(ue, gpr_ln, _ln_chunks(n_out, C4))
    ue.layer_norm_core_dram(M=n_out, N=C4, A_DRAM_ADDR=merged, OUTPUT_DRAM_ADDR=normed,
                            GAMMA_DRAM_ADDR=norm_g, BETA_DRAM_ADDR=norm_b, gpr_M_reg=gpr_ln)
    # reduction Linear(4C -> 2C), no bias
    out = alloc(n_out * 2 * C)
    _pbi_set(ue, gpr_mm, n_out)
    ue.matmat_mul_core(M=n_out, K=C4, N=2 * C, A_DRAM_ADDR=normed, B_DRAM_ADDR=reduction_w,
                       OUTPUT_DRAM_ADDR=out, gpr_M_reg=gpr_mm)
    return out, Hh, Wh


# =============================================================================
# load a torch SwinBlock's weights to device + build constants
# =============================================================================
def _load_block_weights(ue, blk, spec, dim):
    nh, hd, hd_pad = spec.num_q_heads, spec.head_dim, spec.head_dim_pad
    sd = blk.state_dict()
    qkv_w, qkv_b = sd["qkv.weight"], sd["qkv.bias"]          # [3dim,dim],[3dim]
    W = {
        "n1_g": _alloc_param(ue, sd["norm1.weight"]), "n1_b": _alloc_param(ue, sd["norm1.bias"]),
        "n2_g": _alloc_param(ue, sd["norm2.weight"]), "n2_b": _alloc_param(ue, sd["norm2.bias"]),
        "fc1_w": _alloc_param(ue, sd["fc1.weight"]), "fc1_b": _alloc_param(ue, sd["fc1.bias"]),
        "fc2_w": _alloc_param(ue, sd["fc2.weight"]), "fc2_b": _alloc_param(ue, sd["fc2.bias"]),
    }
    # split fused qkv into separate q/k/v projections (what the emitter wants)
    attn = {}
    for i, nm in enumerate(("q", "k", "v")):
        attn[f"{nm}_weight"] = _alloc_param(ue, qkv_w[i * dim:(i + 1) * dim])
        attn[f"{nm}_bias"] = _alloc_param(ue, qkv_b[i * dim:(i + 1) * dim])
    attn["out_weight"] = _alloc_param(ue, sd["proj.weight"])
    attn["out_bias"] = _alloc_param(ue, sd["proj.bias"])
    # unpad selection [dim, nh*hd_pad]: pick real hd columns out of each hd_pad block
    unpad = torch.zeros(dim, nh * hd_pad)
    for h in range(nh):
        for d in range(hd):
            unpad[h * hd + d, h * hd_pad + d] = 1.0
    attn["unpad_weight"] = _alloc_param(ue, unpad)
    W["attn"] = attn
    # constants: identity (transpose) + padded-key softmax mask
    wa_pad = spec.seq_pad
    mask = torch.zeros(spec.batch * nh, wa_pad, wa_pad)
    mask[:, :, spec.seq_len:] = -1e4
    consts = {"identity": _alloc_param(ue, torch.eye(UE_VECTOR_SIZE)),
              "bias": _alloc_param(ue, mask)}
    return W, consts


# =============================================================================
# top-level: compile + execute a Swin block, return (hidden_state, snr)
# =============================================================================
def run_swin_block(dim=192, heads=6, ws=12, H=12, banner=True):
    import torch_to_ue
    torch.manual_seed(0)
    Wd = H
    mlp_dim = dim * 4
    blk = torch_to_ue.SwinBlock(dim=dim, heads=heads, ws=ws).eval()
    x = torch.randn(1, H, Wd, dim)
    ref = blk(x).reshape(H * Wd, dim)                       # torch hidden state

    nw = (H // ws) ** 2
    spec = AttentionSpec(kind="windowed", num_q_heads=heads, num_kv_heads=heads,
                         head_dim=dim // heads, seq_len=ws * ws, batch=nw, bias="none")

    from user_dma_core import UnifiedEngine
    ue = UnifiedEngine()
    W, consts = _load_block_weights(ue, blk, spec, dim)
    x_addr = _alloc_param(ue, x.reshape(H * Wd, dim))
    gpr_mm, gpr_ln, gpr_pbi = ue.alloc_isa_reg(), ue.alloc_isa_reg(), ue.alloc_isa_reg()

    prog = ue.get_program_dram_addr()
    ue.clear_inst_id(); ue.start_capture()
    out_addr = emit_swin_block(ue, W, spec, x_addr=x_addr, H=H, Wd=Wd, dim=dim,
                               mlp_dim=mlp_dim, ws=ws, gpr_mm=gpr_mm, gpr_ln=gpr_ln,
                               gpr_pbi=gpr_pbi, alloc=lambda n: _alloc(ue, n), consts=consts)
    ue.generate_instruction_halt(); ue.stop_capture()
    size = ue.write_captured_instructions_to_dram(prog)
    ue.allocate_program_dram(size)
    if banner:
        print(f"[exec] captured full Swin block: {size/1024:.1f} KB")
    ue.program_execute(prog)
    hw = ue.dma_from_accelerator_memory(out_addr, (H * Wd, dim)).float()
    return ref, hw, snr_db(ref, hw)


# =============================================================================
# stack `depth` blocks = a full Swin STAGE (the block_stack PBI archetype).
# Each block's output chains into the next; weights differ per block.
# =============================================================================
def run_swin_stack(dim=192, heads=6, ws=12, H=12, depth=2, banner=True):
    import torch_to_ue
    torch.manual_seed(0)
    Wd = H
    mlp_dim = dim * 4
    blocks = [torch_to_ue.SwinBlock(dim=dim, heads=heads, ws=ws).eval() for _ in range(depth)]
    x = torch.randn(1, H, Wd, dim)
    # torch reference: chain all blocks
    ref = x
    for b in blocks:
        ref = b(ref)
    ref = ref.reshape(H * Wd, dim)

    nw = (H // ws) ** 2
    spec = AttentionSpec(kind="windowed", num_q_heads=heads, num_kv_heads=heads,
                         head_dim=dim // heads, seq_len=ws * ws, batch=nw, bias="none")

    from user_dma_core import UnifiedEngine
    ue = UnifiedEngine()
    perblk = [_load_block_weights(ue, b, spec, dim) for b in blocks]
    x_addr = _alloc_param(ue, x.reshape(H * Wd, dim))
    gpr_mm, gpr_ln, gpr_pbi = ue.alloc_isa_reg(), ue.alloc_isa_reg(), ue.alloc_isa_reg()

    prog = ue.get_program_dram_addr()
    ue.clear_inst_id(); ue.start_capture()
    cur = x_addr
    for (W, consts) in perblk:
        cur = emit_swin_block(ue, W, spec, x_addr=cur, H=H, Wd=Wd, dim=dim,
                              mlp_dim=mlp_dim, ws=ws, gpr_mm=gpr_mm, gpr_ln=gpr_ln,
                              gpr_pbi=gpr_pbi, alloc=lambda n: _alloc(ue, n), consts=consts)
    ue.generate_instruction_halt(); ue.stop_capture()
    size = ue.write_captured_instructions_to_dram(prog)
    ue.allocate_program_dram(size)
    if banner:
        print(f"[exec] captured {depth}-block stage: {size/1024:.1f} KB")
    ue.program_execute(prog)
    hw = ue.dma_from_accelerator_memory(cur, (H * Wd, dim)).float()
    return ref, hw, snr_db(ref, hw)


# =============================================================================
# MULTI-STAGE: chain stages with patch-merge between them (the full encoder body).
# Driven by a stage list [(dim, heads, depth, spatial)]; merges halve spatial,
# double dim. torch reference uses the same blocks + an equivalent PatchMerging.
# =============================================================================
import torch.nn as nn


class _TorchPatchMerging(nn.Module):
    """HF-equivalent: 2x2 cat -> LayerNorm(4C) -> Linear(4C,2C,bias=False)."""
    def __init__(self, C):
        super().__init__()
        self.norm = nn.LayerNorm(4 * C)
        self.reduction = nn.Linear(4 * C, 2 * C, bias=False)

    def forward(self, x):                        # x: [B, H, W, C]
        x0 = x[:, 0::2, 0::2, :]; x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]; x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)      # [B, H/2, W/2, 4C]
        B, Hh, Wh, _ = x.shape
        x = self.norm(x.reshape(B, Hh * Wh, 4 * x0.shape[-1]))
        return self.reduction(x).reshape(B, Hh, Wh, -1)


def run_swin_multistage(stages, ws=12, banner=True):
    """stages: [(dim, heads, depth, spatial)]. Returns (ref, hw, snr) for the
    final-stage hidden state."""
    import torch_to_ue
    torch.manual_seed(0)
    # build torch modules
    blocks_per = []
    merges = []
    for si, (dim, heads, depth, sp) in enumerate(stages):
        blocks_per.append([torch_to_ue.SwinBlock(dim=dim, heads=heads, ws=ws).eval()
                           for _ in range(depth)])
        if si < len(stages) - 1:
            merges.append(_TorchPatchMerging(dim).eval())

    dim0, _, _, sp0 = stages[0]
    x = torch.randn(1, sp0, sp0, dim0)
    # torch reference
    ref = x
    for si, blks in enumerate(blocks_per):
        for b in blks:
            ref = b(ref)
        if si < len(merges):
            ref = merges[si](ref)
    dim_last, _, _, _ = stages[-1]
    sp_last = stages[-1][3]
    ref = ref.reshape(sp_last * sp_last, dim_last)

    from user_dma_core import UnifiedEngine
    ue = UnifiedEngine()
    gpr_mm, gpr_ln, gpr_pbi = ue.alloc_isa_reg(), ue.alloc_isa_reg(), ue.alloc_isa_reg()

    # load all weights
    dev_blocks, dev_merges = [], []
    for si, (dim, heads, depth, sp) in enumerate(stages):
        nw = (sp // ws) ** 2
        spec = AttentionSpec(kind="windowed", num_q_heads=heads, num_kv_heads=heads,
                             head_dim=dim // heads, seq_len=ws * ws, batch=nw, bias="none")
        dev_blocks.append((spec, [_load_block_weights(ue, b, spec, dim) for b in blocks_per[si]]))
        if si < len(merges):
            m = merges[si].state_dict()
            dev_merges.append({"n_g": _alloc_param(ue, m["norm.weight"]),
                               "n_b": _alloc_param(ue, m["norm.bias"]),
                               "red": _alloc_param(ue, m["reduction.weight"])})
    x_addr = _alloc_param(ue, x.reshape(sp0 * sp0, dim0))

    prog = ue.get_program_dram_addr()
    ue.clear_inst_id(); ue.start_capture()
    cur = x_addr
    H = Wd = sp0
    for si, (dim, heads, depth, sp) in enumerate(stages):
        spec, perblk = dev_blocks[si]
        for (W, consts) in perblk:
            cur = emit_swin_block(ue, W, spec, x_addr=cur, H=H, Wd=Wd, dim=dim,
                                  mlp_dim=dim * 4, ws=ws, gpr_mm=gpr_mm, gpr_ln=gpr_ln,
                                  gpr_pbi=gpr_pbi, alloc=lambda n: _alloc(ue, n), consts=consts)
        if si < len(merges):
            dm = dev_merges[si]
            cur, H, Wd = emit_patch_merge(ue, cur, H, Wd, dim, norm_g=dm["n_g"],
                                          norm_b=dm["n_b"], reduction_w=dm["red"],
                                          gpr_mm=gpr_mm, gpr_ln=gpr_ln, gpr_pbi=gpr_pbi,
                                          alloc=lambda n: _alloc(ue, n))
    ue.generate_instruction_halt(); ue.stop_capture()
    size = ue.write_captured_instructions_to_dram(prog)
    ue.allocate_program_dram(size)
    if banner:
        print(f"[exec] captured {len(stages)}-stage encoder: {size/1024:.1f} KB "
              f"(final {H}x{Wd}x{dim_last})")
    ue.program_execute(prog)
    hw = ue.dma_from_accelerator_memory(cur, (sp_last * sp_last, dim_last)).float()
    return ref, hw, snr_db(ref, hw)


class _TinySwin(nn.Module):
    """Full Swin-shaped model: patch-embed -> stages(+merges) -> norm -> pool -> head."""
    def __init__(self, in_ch, patch, stages, num_classes, ws):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, stages[0][0], patch, patch)
        self.blocks = nn.ModuleList()
        self.merges = nn.ModuleList()
        import torch_to_ue
        for si, (dim, heads, depth, sp) in enumerate(stages):
            self.blocks.append(nn.ModuleList(
                [torch_to_ue.SwinBlock(dim=dim, heads=heads, ws=ws) for _ in range(depth)]))
            if si < len(stages) - 1:
                self.merges.append(_TorchPatchMerging(dim))
        self.norm = nn.LayerNorm(stages[-1][0])
        self.head = nn.Linear(stages[-1][0], num_classes)

    def embed(self, img):
        x = self.proj(img)                       # [B, embed, sp, sp]
        B, C, Hh, Wd = x.shape
        return x.flatten(2).transpose(1, 2).reshape(B, Hh, Wd, C)

    def encode(self, x):
        for si, blks in enumerate(self.blocks):
            for b in blks:
                x = b(x)
            if si < len(self.merges):
                x = self.merges[si](x)
        return x                                 # [B, h, w, dim_last]

    def head_fwd(self, x):                        # x: [B, h, w, dim] -> logits
        B = x.shape[0]
        x = self.norm(x.reshape(B, -1, x.shape[-1]))
        return self.head(x.mean(1))


def _load_image(path, size):
    """Load a real image -> normalized [1,3,size,size] tensor (ImageNet stats)."""
    from PIL import Image
    im = Image.open(path).convert("RGB").resize((size, size))
    t = torch.tensor(list(im.getdata()), dtype=torch.float32).reshape(size, size, 3)
    t = t.permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0)


def run_swin_model(stages, ws=12, in_ch=3, patch=4, num_classes=1000, banner=True,
                   image_path=None):
    """FULL model: HOST patch-embed -> DEVICE encoder(stages+merges) -> HOST head.
    Returns (logits_ref, logits_hw, hidden_snr). The compiler's host/device split:
    heavy transformer on the FPGA, embed + pool + classify on the CPU.
    image_path: a real image to classify (else random input). NOTE: weights are
    random here, so the class is only meaningful once real checkpoint weights load."""
    torch.manual_seed(0)
    sp0 = stages[0][3]
    image = sp0 * patch
    model = _TinySwin(in_ch, patch, stages, num_classes, ws).eval()
    if image_path:
        img = _load_image(image_path, image)
        print(f"[exec] INPUT  real image: {image_path}  -> {tuple(img.shape)}")
    else:
        img = torch.randn(1, in_ch, image, image)
        print(f"[exec] INPUT  random tensor {tuple(img.shape)} (no image passed)")

    # ---- torch reference (full forward) ----
    with torch.no_grad():
        x_ref = model.embed(img)
        hid_ref = model.encode(x_ref)
        logits_ref = model.head_fwd(hid_ref)
    dim_last = stages[-1][0]; sp_last = stages[-1][3]
    hid_ref_flat = hid_ref.reshape(sp_last * sp_last, dim_last)

    # ---- HOST: patch embedding (run on CPU, DMA tokens to device) ----
    with torch.no_grad():
        tokens = model.embed(img).reshape(sp0 * sp0, stages[0][0])
    print(f"[exec] HOST  patch-embed: image {tuple(img.shape)} -> tokens {tuple(tokens.shape)}")

    from user_dma_core import UnifiedEngine
    ue = UnifiedEngine()
    gpr_mm, gpr_ln, gpr_pbi = ue.alloc_isa_reg(), ue.alloc_isa_reg(), ue.alloc_isa_reg()

    dev_blocks, dev_merges = [], []
    for si, (dim, heads, depth, sp) in enumerate(stages):
        nw = (sp // ws) ** 2
        spec = AttentionSpec(kind="windowed", num_q_heads=heads, num_kv_heads=heads,
                             head_dim=dim // heads, seq_len=ws * ws, batch=nw, bias="none")
        dev_blocks.append((spec, [_load_block_weights(ue, b, spec, dim) for b in model.blocks[si]]))
        if si < len(model.merges):
            m = model.merges[si].state_dict()
            dev_merges.append({"n_g": _alloc_param(ue, m["norm.weight"]),
                               "n_b": _alloc_param(ue, m["norm.bias"]),
                               "red": _alloc_param(ue, m["reduction.weight"])})
    x_addr = _alloc_param(ue, tokens)

    print(f"[exec] DEVICE encoder: {len(stages)} stages, "
          f"{sum(len(model.blocks[i]) for i in range(len(stages)))} blocks, "
          f"{len(model.merges)} patch-merges")
    prog = ue.get_program_dram_addr()
    ue.clear_inst_id(); ue.start_capture()
    cur = x_addr; H = Wd = sp0
    for si, (dim, heads, depth, sp) in enumerate(stages):
        spec, perblk = dev_blocks[si]
        for (W, consts) in perblk:
            cur = emit_swin_block(ue, W, spec, x_addr=cur, H=H, Wd=Wd, dim=dim,
                                  mlp_dim=dim * 4, ws=ws, gpr_mm=gpr_mm, gpr_ln=gpr_ln,
                                  gpr_pbi=gpr_pbi, alloc=lambda n: _alloc(ue, n), consts=consts)
        if si < len(model.merges):
            dm = dev_merges[si]
            cur, H, Wd = emit_patch_merge(ue, cur, H, Wd, dim, norm_g=dm["n_g"],
                                          norm_b=dm["n_b"], reduction_w=dm["red"],
                                          gpr_mm=gpr_mm, gpr_ln=gpr_ln, gpr_pbi=gpr_pbi,
                                          alloc=lambda n: _alloc(ue, n))
    ue.generate_instruction_halt(); ue.stop_capture()
    size = ue.write_captured_instructions_to_dram(prog)
    ue.allocate_program_dram(size)
    print(f"[exec] DEVICE program: {size/1024:.1f} KB")
    ue.program_execute(prog)
    hid_hw = ue.dma_from_accelerator_memory(cur, (sp_last * sp_last, dim_last)).float()

    # ---- HOST: final norm + mean-pool + classifier ----
    with torch.no_grad():
        logits_hw = model.head_fwd(hid_hw.reshape(1, sp_last, sp_last, dim_last))
    print(f"[exec] HOST  head: hidden {tuple(hid_hw.shape)} -> logits {tuple(logits_hw.shape)}")
    return logits_ref[0], logits_hw[0], snr_db(hid_ref_flat, hid_hw)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Swin executor on the FPGA")
    ap.add_argument("--mode", choices=["model", "multistage", "stage", "block"],
                    default="model", help="what to run (default: full model)")
    ap.add_argument("--image", default=None, help="path to an image to classify (model mode)")
    ap.add_argument("--depth", type=int, default=2, help="blocks to stack (stage mode)")
    args = ap.parse_args()

    if args.mode == "model":
        stages = [(192, 6, 1, 24), (384, 12, 1, 12)]
        lref, lhw, hsnr = run_swin_model(stages, image_path=args.image)
        top_ref, top_hw = int(lref.argmax()), int(lhw.argmax())
        print(f"[exec] hidden-state SNR (device vs torch) = {hsnr:.2f} dB")
        print(f"[exec] logit argmax: torch={top_ref}  hw={top_hw}  "
              f"[{'MATCH' if top_ref == top_hw else 'MISMATCH'}]")
        print(f"       logits_ref[:5] = {[round(v,3) for v in lref[:5].tolist()]}")
        print(f"       logits_hw [:5] = {[round(v,3) for v in lhw[:5].tolist()]}")
    elif args.mode == "multistage":
        ref, hw, s = run_swin_multistage([(192, 6, 1, 24), (384, 12, 1, 12)])
        print(f"[exec] 2-stage encoder hidden state {tuple(ref.shape)}  SNR = {s:.2f} dB  "
              f"[{'PASS' if s > 19 else 'FAIL'}]")
    elif args.mode == "block":
        ref, hw, s = run_swin_block()
        print(f"[exec] Swin block hidden state {tuple(ref.shape)}  SNR = {s:.2f} dB  "
              f"[{'PASS' if s > 19 else 'FAIL'}]")
    else:   # stage
        ref, hw, s = run_swin_stack(depth=args.depth)
        print(f"[exec] {args.depth}-block stage hidden state {tuple(ref.shape)}  SNR = {s:.2f} dB  "
              f"[{'PASS' if s > 19 else 'FAIL'}]")
