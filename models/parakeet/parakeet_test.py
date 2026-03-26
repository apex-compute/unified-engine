#!/usr/bin/env python3
"""
Parakeet-TDT-0.6B inference on accelerator.

  - Config from parakeet_config.json; weights from NeMo checkpoint.
  - Mel spectrogram computed on host CPU.
  - Encoder (subsampling + 24x conformer) compiled and run on accelerator.
  - TDT decode loop (LSTM predictor + joint network) run on accelerator.

Usage:
  python parakeet_test.py
  python parakeet_test.py --audio test.wav
  python parakeet_test.py --dev xdma0 [--cycle 5.63]

Fixed layout: parakeet_test.py, parakeet_config.json, and parakeet_bin/ live in the same folder.
  user_dma_core.py is at the repo root (two folders up); that directory is added to sys.path.
"""

import json
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

import numpy as np
import torch
import torch.nn.functional as F

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, DMA_DEVICE_C2H, DRAM_INSTRUCTION_ADDR, TYPE, UE_VECTOR_SIZE,
    URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS, UnifiedEngine, set_dma_device,
    UE_MODE, BROADCAST_MODE, LALU_MODE, MEMCPY_TYPE, URAM_SECTION, UE_ARGMAX1_INDEX, UE_ARGMAX2_INDEX
)

URAM_A_BASE = 0x00000
URAM_B_BASE = 0x80000
EPS = 1e-5

# Quantization support (FP4 / INT4)
FP4_WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "parakeet_bin", "encoder_fp4.bin")
FP4_MANIFEST_PATH = os.path.join(SCRIPT_DIR, "parakeet_bin", "encoder_fp4.json")
INT4_WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "parakeet_bin", "encoder_int4.bin")
INT4_MANIFEST_PATH = os.path.join(SCRIPT_DIR, "parakeet_bin", "encoder_int4.json")

_FP4_E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.bfloat16)

def quantize_fp4_64(tensor):
    """FP4 E2M1 quantization with 64-element blocks. Returns (packed_bytes, n_blocks)."""
    x = tensor.to(torch.bfloat16).cpu().flatten()
    n_blocks = int(np.ceil(x.numel() / 64))
    if x.numel() % 64 != 0:
        x = torch.nn.functional.pad(x, (0, n_blocks * 64 - x.numel()))
    blocks = x.view(n_blocks, 64)
    fp4_max = torch.tensor(6.0, dtype=torch.bfloat16)
    scales = (blocks.abs().max(dim=1).values / fp4_max).clamp(min=1e-8).to(torch.bfloat16)
    scaled = (blocks / scales[:, None]).to(torch.bfloat16)
    codes = torch.argmin(torch.abs(scaled.unsqueeze(-1) - _FP4_E2M1_TABLE), dim=-1).to(torch.uint8)
    codes_np = codes.numpy().flatten()
    if len(codes_np) % 2 != 0:
        codes_np = np.pad(codes_np, (0, 1))
    packed = (codes_np[0::2] & 0x0F) | ((codes_np[1::2] & 0x0F) << 4)
    scales_np = scales.view(torch.uint16).numpy()
    return np.frombuffer(scales_np.tobytes() + packed.tobytes(), dtype=np.uint8), n_blocks

def quantize_q4_64(tensor):
    """INT4 quantization with 64-element blocks. Returns (packed_bytes, n_blocks)."""
    data = tensor.flatten().cpu().float().numpy()
    n_blocks = int(np.ceil(len(data) / 64))
    padded = np.pad(data, (0, n_blocks * 64 - len(data)))
    blocks = padded.reshape(n_blocks, 64)
    scales = np.max(np.abs(blocks), axis=1)
    scales[scales == 0] = 1.0
    scales /= 7.0
    quantized = np.clip(np.round(blocks / scales[:, None]), -8, 7).astype(np.int8)
    pairs = (quantized.astype(np.uint8) & 0x0F).reshape(n_blocks, 32, 2)
    packed = pairs[:, :, 0] | (pairs[:, :, 1] << 4)
    scale_bytes = torch.tensor(scales, dtype=torch.float32).to(torch.bfloat16).view(torch.uint16).numpy()
    return np.frombuffer(scale_bytes.tobytes() + packed.tobytes(), dtype=np.uint8), n_blocks

def _encoder_weight_entries(sd, layer_idx, FF, D):
    """Return dict of {key: tensor} for the 16 large encoder weights of a layer."""
    pfx = f"encoder.layers.{layer_idx}"
    return {
        "FF1_W1_LO": sd[f"{pfx}.feed_forward1.linear1.weight"][:FF//2, :].contiguous(),
        "FF1_W1_HI": sd[f"{pfx}.feed_forward1.linear1.weight"][FF//2:, :].contiguous(),
        "FF1_W2_LO": sd[f"{pfx}.feed_forward1.linear2.weight"][:, :FF//2].contiguous(),
        "FF1_W2_HI": sd[f"{pfx}.feed_forward1.linear2.weight"][:, FF//2:].contiguous(),
        "FF2_W1_LO": sd[f"{pfx}.feed_forward2.linear1.weight"][:FF//2, :].contiguous(),
        "FF2_W1_HI": sd[f"{pfx}.feed_forward2.linear1.weight"][FF//2:, :].contiguous(),
        "FF2_W2_LO": sd[f"{pfx}.feed_forward2.linear2.weight"][:, :FF//2].contiguous(),
        "FF2_W2_HI": sd[f"{pfx}.feed_forward2.linear2.weight"][:, FF//2:].contiguous(),
        "ATTN_Q_W": sd[f"{pfx}.self_attn.linear_q.weight"],
        "ATTN_K_W": sd[f"{pfx}.self_attn.linear_k.weight"],
        "ATTN_V_W": sd[f"{pfx}.self_attn.linear_v.weight"],
        "ATTN_POS_W": sd[f"{pfx}.self_attn.linear_pos.weight"],
        "ATTN_OUT_W": sd[f"{pfx}.self_attn.linear_out.weight"],
        "CONV_PW1A_W": sd[f"{pfx}.conv.pointwise_conv1.weight"].squeeze(-1)[:D].contiguous(),
        "CONV_PW1B_W": sd[f"{pfx}.conv.pointwise_conv1.weight"].squeeze(-1)[D:].contiguous(),
        "CONV_PW2_W": sd[f"{pfx}.conv.pointwise_conv2.weight"].squeeze(-1),
    }

def generate_quantized_weights(quant_mode):
    """Quantize large encoder weights and write bin + json manifest.
    quant_mode: 'fp4' or 'int4'."""
    qfn = quantize_fp4_64 if quant_mode == 'fp4' else quantize_q4_64
    bin_path = FP4_WEIGHTS_PATH if quant_mode == 'fp4' else INT4_WEIGHTS_PATH
    json_path = FP4_MANIFEST_PATH if quant_mode == 'fp4' else INT4_MANIFEST_PATH
    print(f"Generating {quant_mode.upper()} encoder weights...")
    sd = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    FF, D = 4096, 1024
    manifest = {}
    with open(bin_path, 'wb') as f:
        for i in range(24):
            entries = _encoder_weight_entries(sd, i, FF, D)
            for key, tensor in entries.items():
                name = f"layer.{i}.{key}"
                data, _ = qfn(tensor)
                offset = f.tell()
                raw = data.tobytes()
                f.write(raw)
                manifest[name] = {"offset": offset, "size": len(raw)}
            print(f"\r  Layer {i:2d}/23 quantized", end="", flush=True)
    print()
    with open(json_path, 'w') as f:
        json.dump(manifest, f)
    print(f"  Written: {bin_path} ({os.path.getsize(bin_path)/1048576:.1f} MB)")
    print(f"  Manifest: {json_path} ({len(manifest)} tensors)")

def load_quantized_weight_cache(quant_mode):
    """Load quantized bin + json manifest. Returns {tensor_name: raw_numpy_data}."""
    json_path = FP4_MANIFEST_PATH if quant_mode == 'fp4' else INT4_MANIFEST_PATH
    bin_path = FP4_WEIGHTS_PATH if quant_mode == 'fp4' else INT4_WEIGHTS_PATH
    with open(json_path) as f:
        manifest = json.load(f)
    with open(bin_path, 'rb') as f:
        raw = f.read()
    cache = {}
    for name, meta in manifest.items():
        cache[name] = np.frombuffer(raw[meta['offset']:meta['offset'] + meta['size']], dtype=np.uint8).copy()
    return cache

def store_quantized_weight(ue, raw_data):
    """Store FP4_64 quantized data as separate scale + packed in DRAM. Returns (scale_addr, data_addr)."""
    raw_bytes = raw_data.tobytes() if hasattr(raw_data, 'tobytes') else bytes(raw_data)
    n_blocks = len(raw_bytes) // 34  # 34 = 2 (scale) + 32 (packed codes for 64 elements)
    scales_size = n_blocks * 2
    data_size = n_blocks * 32
    scales_np = np.frombuffer(raw_bytes[:scales_size], dtype=np.uint16).copy()
    scale_tensor = torch.from_numpy(scales_np).view(torch.bfloat16)
    scale_addr = ue.get_params_dram_addr()
    ue.dma_write(DMA_DEVICE_H2C, scale_addr, scale_tensor, scales_size)
    ue.allocate_params_dram(scales_size)
    data_np = np.frombuffer(raw_bytes[scales_size:scales_size + data_size], dtype=np.uint8).copy()
    data_tensor = torch.from_numpy(data_np)
    data_addr = ue.get_params_dram_addr()
    ue.dma_write(DMA_DEVICE_H2C, data_addr, data_tensor, data_size)
    ue.allocate_params_dram(data_size)
    return scale_addr, data_addr

WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "parakeet_bin", "model_weights.ckpt")
TOKENIZER_PATH = os.path.join(SCRIPT_DIR, "parakeet_bin", "tokenizer.model")


# ---------------------------------------------------------------------------
# Core functions: custom ops for Parakeet Conformer
# ---------------------------------------------------------------------------
def batch_norm_fuse_params(ue: UnifiedEngine, bn_weight, bn_bias, bn_mean, bn_var, eps=EPS):
    """Pre-fuse BN params to scale+shift, write to DRAM. Call once at weight load.
    Returns (scale_dram_addr, shift_dram_addr, C).
    """
    scale = (bn_weight.float() / torch.sqrt(bn_var.float() + eps)).to(torch.bfloat16).contiguous()
    shift = (bn_bias.float() - bn_mean.float() * scale.float()).to(torch.bfloat16).contiguous()
    C = scale.shape[0]
    scale_addr = ue.get_params_dram_addr()
    ue.allocate_params_dram(C * 2)
    ue.dma_write(DMA_DEVICE_H2C, scale_addr, scale, C * 2)
    shift_addr = ue.get_params_dram_addr()
    ue.allocate_params_dram(C * 2)
    ue.dma_write(DMA_DEVICE_H2C, shift_addr, shift, C * 2)
    return scale_addr, shift_addr, C
def batch_norm_prepare_tiled(ue, C, L, SCALE_DRAM_ADDR, SHIFT_DRAM_ADDR):
    """Pre-tile BN scale/shift vectors into (C, L) matrices for bulk ops.
    Returns (tiled_scale_addr, tiled_shift_addr).
    Call once before compile, not inside instruction capture.
    """
    bpe = 2
    # Read scale and shift from DRAM
    scale_host = torch.zeros(C, dtype=torch.bfloat16)
    shift_host = torch.zeros(C, dtype=torch.bfloat16)
    ue.dma_read(DMA_DEVICE_C2H, SCALE_DRAM_ADDR, scale_host, C * bpe)
    ue.dma_read(DMA_DEVICE_C2H, SHIFT_DRAM_ADDR, shift_host, C * bpe)
    # Tile: each scalar repeated L times per row
    scale_tiled = scale_host.unsqueeze(1).expand(C, L).contiguous()
    shift_tiled = shift_host.unsqueeze(1).expand(C, L).contiguous()
    # Write to DRAM
    scale_tiled_addr = ue.get_params_dram_addr()
    ue.allocate_params_dram(C * L * bpe)
    ue.dma_to_accelerator_memory(scale_tiled_addr, scale_tiled)
    shift_tiled_addr = ue.get_params_dram_addr()
    ue.allocate_params_dram(C * L * bpe)
    ue.dma_to_accelerator_memory(shift_tiled_addr, shift_tiled)
    return scale_tiled_addr, shift_tiled_addr
def batch_norm_core_dram(ue: UnifiedEngine, C: int, L: int,
                         A_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                         SCALE_DRAM_ADDR: int, SHIFT_DRAM_ADDR: int,
                         tiled_scale_addr: int = None,
                         tiled_shift_addr: int = None) -> None:
    """Emit instructions for fused eval-mode batch norm on (C, L) tensor.
    If tiled_scale_addr and tiled_shift_addr are provided, uses bulk ops (6 instructions).
    Otherwise falls back to per-channel loop.
    """
    assert L % UE_VECTOR_SIZE == 0, f"L={L} must be a multiple of {UE_VECTOR_SIZE}"
    if tiled_scale_addr is not None and tiled_shift_addr is not None:
        total_elems = C * L
        # Load input, load tiled scale, multiply
        ue.accelerator_memory_to_sram(A_DRAM_ADDR, URAM_A_BASE, total_elems)
        ue.accelerator_memory_to_sram(tiled_scale_addr, URAM_B_BASE, total_elems)
        ue.eltwise_mul_core(vector_A_sram_start_addr=URAM_A_BASE,
                           vector_B_sram_start_addr=URAM_B_BASE,
                           vector_C_sram_wb_addr=URAM_A_BASE, element_size=total_elems)
        # Load tiled shift, add
        ue.accelerator_memory_to_sram(tiled_shift_addr, URAM_B_BASE, total_elems)
        ue.eltwise_add_core(vector_A_sram_start_addr=URAM_A_BASE,
                           vector_B_sram_start_addr=URAM_B_BASE,
                           vector_C_sram_wb_addr=URAM_A_BASE, element_size=total_elems)
        # Write result
        ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR, total_elems)
        return
    # Fallback: per-channel (original behavior)
    row_bytes = L * 2
    scale_host = torch.zeros(C, dtype=torch.bfloat16)
    shift_host = torch.zeros(C, dtype=torch.bfloat16)
    ue.dma_read(DMA_DEVICE_C2H, SCALE_DRAM_ADDR, scale_host, C * 2)
    ue.dma_read(DMA_DEVICE_C2H, SHIFT_DRAM_ADDR, shift_host, C * 2)
    for c in range(C):
        ue.accelerator_memory_to_sram(A_DRAM_ADDR + c * row_bytes, URAM_A_BASE, L)
        ue.broadcast_mul(scalar=scale_host[c].float().item(), sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=L)
        ue.broadcast_add(scalar=shift_host[c].float().item(), sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=L)
        ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR + c * row_bytes, L)
def allocate_identity(ue: UnifiedEngine, N: int):
    """Allocate and write an (N, N) bf16 identity matrix to DRAM. Call once at init.
    Returns DRAM address. Reused by tanh_core_dram and glu_core_dram.
    """
    assert N % UE_VECTOR_SIZE == 0, f"N={N} must be a multiple of {UE_VECTOR_SIZE}"
    identity = torch.eye(N, dtype=torch.bfloat16).contiguous()
    addr = ue.get_params_dram_addr()
    ue.allocate_params_dram(N * N * 2)
    ue.dma_write(DMA_DEVICE_H2C, addr, identity, N * N * 2)
    return addr
def tanh_core_dram(ue: UnifiedEngine, M: int, N: int,
                   A_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                   IDENTITY_DRAM_ADDR: int) -> None:
    """Emit instructions for tanh(x) = 2*sigmoid(2x) - 1 on (M, N) tensor.
    Steps:
        1. MUL_BROADCAST scalar=2.0 on each row         →  2x
        2. matmat_mul_core (M,N)@(N,N) sigmoid_enable   →  sigmoid(2x)
        3. MUL_BROADCAST scalar=2.0 on each row         →  2*sigmoid(2x)
        4. ADD_BROADCAST scalar=-1.0 on each row        →  tanh(x)
    """
    assert N % UE_VECTOR_SIZE == 0, f"N={N} must be a multiple of {UE_VECTOR_SIZE}"
    row_bytes = N * 2
    # Step 1: 2x → OUTPUT as temp
    for m in range(M):
        ue.accelerator_memory_to_sram(A_DRAM_ADDR + m * row_bytes, URAM_A_BASE, N)
        ue.broadcast_mul(scalar=2.0, sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=N)
        ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR + m * row_bytes, N)
    # Step 2: sigmoid(2x) via identity matmul with LALU sigmoid
    ue.matmat_mul_core(M=M, K=N, N=N,
                       A_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                       B_DRAM_ADDR=IDENTITY_DRAM_ADDR,
                       OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                       sigmoid_enable=True)
    # Steps 3+4: 2*sigmoid(2x) - 1
    for m in range(M):
        ue.accelerator_memory_to_sram(OUTPUT_DRAM_ADDR + m * row_bytes, URAM_A_BASE, N)
        ue.broadcast_mul(scalar=2.0, sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=N)
        ue.broadcast_add(scalar=-1.0, sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=N)
        ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR + m * row_bytes, N)
def silu_core_dram(ue: UnifiedEngine, M: int, N: int,
                   A_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                   IDENTITY_DRAM_ADDR: int) -> None:
    """Emit instructions for standalone SiLU: output = x * sigmoid(x) on (M, N) tensor.
    Steps:
        1. Copy input to OUTPUT as temp (preserve x for step 3)
        2. matmat_mul_core (M,N)@I(N,N) sigmoid_enable on temp  → sigmoid(x)
        3. eltwise_mul row-by-row: x * sigmoid(x) → OUTPUT
    """
    assert N % UE_VECTOR_SIZE == 0, f"N={N} must be a multiple of {UE_VECTOR_SIZE}"
    total_elems = M * N
    # Step 1: bulk copy input to OUTPUT (so we can sigmoid in-place there)
    ue.accelerator_memory_to_sram(A_DRAM_ADDR, URAM_A_BASE, total_elems)
    ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR, total_elems)
    # Step 2: sigmoid(x) via identity matmul with LALU sigmoid → OUTPUT in-place
    ue.matmat_mul_core(M=M, K=N, N=N, A_DRAM_ADDR=OUTPUT_DRAM_ADDR, B_DRAM_ADDR=IDENTITY_DRAM_ADDR, OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR, sigmoid_enable=True)
    # Step 3: bulk x * sigmoid(x)
    ue.accelerator_memory_to_sram(A_DRAM_ADDR, URAM_A_BASE, total_elems)
    ue.accelerator_memory_to_sram(OUTPUT_DRAM_ADDR, URAM_B_BASE, total_elems)
    ue.eltwise_mul_core(vector_A_sram_start_addr=URAM_A_BASE, vector_B_sram_start_addr=URAM_B_BASE, vector_C_sram_wb_addr=URAM_A_BASE, element_size=total_elems)
    ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR, total_elems)
def glu_core_dram(ue: UnifiedEngine, M: int, C: int, A_DRAM_ADDR: int, B_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, IDENTITY_DRAM_ADDR: int) -> None:
    """Emit instructions for GLU: output = a * sigmoid(b).
    Assumes PW Conv1 was split into two matmuls writing a and b separately:
        PW Conv1a: (M, D) @ W_a(D, C) → a at A_DRAM_ADDR     (M, C)
        PW Conv1b: (M, D) @ W_b(D, C) → b at B_DRAM_ADDR     (M, C)
    Steps:
        1. matmat_mul_core (M,C)@I(C,C) sigmoid_enable on b  → sigmoid(b) in-place
        2. eltwise_mul row-by-row: a * sigmoid(b) → OUTPUT
    """
    assert C % UE_VECTOR_SIZE == 0, f"C={C} must be a multiple of {UE_VECTOR_SIZE}"
    total_elems = M * C
    # Step 1: sigmoid(b) via identity matmul
    ue.matmat_mul_core(M=M, K=C, N=C, A_DRAM_ADDR=B_DRAM_ADDR, B_DRAM_ADDR=IDENTITY_DRAM_ADDR, OUTPUT_DRAM_ADDR=B_DRAM_ADDR, sigmoid_enable=True)
    # Step 2: bulk a * sigmoid(b)
    ue.accelerator_memory_to_sram(A_DRAM_ADDR, URAM_A_BASE, total_elems)
    ue.accelerator_memory_to_sram(B_DRAM_ADDR, URAM_B_BASE, total_elems)
    ue.eltwise_mul_core(vector_A_sram_start_addr=URAM_A_BASE, vector_B_sram_start_addr=URAM_B_BASE, vector_C_sram_wb_addr=URAM_A_BASE, element_size=total_elems)
    ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR, total_elems)
def rel_shift_core_dram(ue: UnifiedEngine, L: int, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                        input_row_stride: int = None) -> None:
    """Emit instructions for rel_shift: extract (L, L) from (L, P_pad) positional scores.
    Row i of output = input[i, (L-1-i) : (L-1-i)+L]
    Each is a contiguous L-element DMA copy at a computed source offset.
    No arithmetic — pure memory rearrangement.
    Args:
        ue: UnifiedEngine instance
        L: sequence length (L_pad, bucketed, must be multiple of UE_VECTOR_SIZE)
        INPUT_DRAM_ADDR: (L, input_row_stride) bf16 positional score matrix
        OUTPUT_DRAM_ADDR: (L, L) bf16 output
        input_row_stride: actual number of elements per row in input (P_pad from matmul).
                          Defaults to 2*L-1 for backwards compatibility.
    """
    assert L % UE_VECTOR_SIZE == 0, f"L={L} must be a multiple of {UE_VECTOR_SIZE}"
    P_stride = input_row_stride if input_row_stride is not None else (2 * L - 1)
    bpe = 2          # bf16
    for i in range(L):
        src = INPUT_DRAM_ADDR + (i * P_stride + (L - 1 - i)) * bpe
        dst = OUTPUT_DRAM_ADDR + i * L * bpe
        ue.accelerator_memory_to_sram(src, URAM_A_BASE, L)
        ue.sram_to_accelerator_memory(URAM_A_BASE, dst, L)
def chunked_transpose_core_dram(ue: UnifiedEngine, M: int, N: int,
                                 input_dram_addr: int, output_dram_addr: int,
                                 identity_dram_addr: int, temp_dram_addr: int) -> None:
    """Transpose (M, N) -> (N, M) by processing N in UE_VECTOR_SIZE-column chunks.

    bf16_smart_permute_core's dot-product transpose accumulates across
    N_transpose // UE_VECTOR_SIZE groups, which corrupts results when
    N > UE_VECTOR_SIZE.  This helper splits into chunks where each sub-transpose
    has N_transpose = UE_VECTOR_SIZE (=64), avoiding the bug.

    Input at input_dram_addr: (M, N) contiguous bf16.
    Output at output_dram_addr: (N, M_aligned) contiguous bf16,
        where M_aligned = pad_to_multiple(M, UE_VECTOR_SIZE).
    """
    bpe = 2
    VS = UE_VECTOR_SIZE  # 64
    M_aligned = ((M - 1) // VS + 1) * VS
    n_chunks = (N + VS - 1) // VS

    for c in range(n_chunks):
        col_start = c * VS
        col_end = min(col_start + VS, N)
        chunk_cols = col_end - col_start
        chunk_cols_pad = ((chunk_cols - 1) // VS + 1) * VS  # = VS for full chunks

        # Extract columns [col_start:col_end] from each of M rows via strided DMA
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=input_dram_addr + col_start * bpe,
            sram_address=URAM_A_BASE,
            element_size=M * chunk_cols,
            stride_bytes_per_chunk=chunk_cols * bpe,
            stride_jump_bytes=N * bpe)
        ue.sram_to_accelerator_memory(
            sram_address=URAM_A_BASE,
            accelerator_dram_address=temp_dram_addr,
            element_size=M * chunk_cols_pad)

        # Transpose (M, chunk_cols_pad) -> (chunk_cols_pad, M_aligned)
        # N_transpose = chunk_cols_pad = VS, so dot product has 1 group. Safe.
        bf16_smart_permute_core(ue,
            dims=[M, chunk_cols_pad], permute_indices=[1, 0],
            input_dram_addr=temp_dram_addr,
            output_dram_addr=output_dram_addr + col_start * M_aligned * bpe,
            params_dram_addr=identity_dram_addr,
            temp_dram_start=temp_dram_addr + M * chunk_cols_pad * bpe)
def half_step_residual_core_dram(ue: UnifiedEngine, M: int, N: int,
                                 RESIDUAL_DRAM_ADDR: int, FF_DRAM_ADDR: int,
                                 OUTPUT_DRAM_ADDR: int) -> None:
    """Emit instructions for half-step residual: output = residual + 0.5 * ff_output.
    Args:
        ue: UnifiedEngine instance
        M: number of rows
        N: vector dimension (must be multiple of UE_VECTOR_SIZE)
        RESIDUAL_DRAM_ADDR: (M, N) bf16 — original input x
        FF_DRAM_ADDR: (M, N) bf16 — feed-forward output, modified in-place
        OUTPUT_DRAM_ADDR: (M, N) bf16 — result
    """
    assert N % UE_VECTOR_SIZE == 0, f"N={N} must be a multiple of {UE_VECTOR_SIZE}"
    total_elems = M * N
    # Load FF output, scale by 0.5
    ue.accelerator_memory_to_sram(FF_DRAM_ADDR, URAM_A_BASE, total_elems)
    ue.broadcast_mul(scalar=0.5, sram_start_addr=URAM_A_BASE,
                     sram_wb_addr=URAM_A_BASE, element_size=total_elems)
    # Load residual, add
    ue.accelerator_memory_to_sram(RESIDUAL_DRAM_ADDR, URAM_B_BASE, total_elems)
    ue.eltwise_add_core(vector_A_sram_start_addr=URAM_A_BASE,
                        vector_B_sram_start_addr=URAM_B_BASE,
                        vector_C_sram_wb_addr=URAM_A_BASE,
                        element_size=total_elems)
    # Write result
    ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR, total_elems)
def bf16_smart_permute_core(ue, dims, permute_indices, input_dram_addr, output_dram_addr,
                             params_dram_addr, temp_dram_start):
    """Permute via DMA gather + batched transpose decomposition."""
    from user_dma_core import (UE_VECTOR_SIZE, UE_MODE, URAM_FULL_ELEMENTS,
                               URAM_NEAR_FULL_ELEMENTS, URAM_HALF_ELEMENTS,
                               URAM_SECTION, URAM_WRITE_SRC, URAM_START_ADDR, LALU_MODE)
    n = len(dims) - 1
    bpe = 2
    total_elements = 1
    for d in dims:
        total_elements *= d
    k = permute_indices[n]
    inst_id = 0

    # Case 1: last dim stays fixed (P[n] == n) — pure DMA gather
    if k == n:
        last_dim = dims[n]
        output_shape = tuple(dims[permute_indices[i]] for i in range(len(dims)))
        permute_a = torch.arange(total_elements, dtype=torch.int32).reshape(*dims)
        permute_a = permute_a.permute(*permute_indices).contiguous().flatten()

        if last_dim < UE_VECTOR_SIZE:
            for j in range(total_elements // last_dim):
                src_idx = permute_a[j * last_dim].item()
                ue.ue_memcpy_from_dram(input_dram_addr + src_idx * bpe, last_dim * bpe, 0,
                    URAM_START_ADDR, URAM_SECTION.URAM_A.value, inst_id)
                ue.wait_queue(); inst_id += 1
                ue.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, URAM_START_ADDR,
                    output_dram_addr + j * last_dim * bpe, last_dim * bpe, inst_id)
                ue.wait_queue(); inst_id += 1
            return (1, output_shape)

        out_addr = output_dram_addr
        remaining = total_elements
        aligned = (URAM_NEAR_FULL_ELEMENTS // (UE_VECTOR_SIZE * last_dim)) * UE_VECTOR_SIZE * last_dim
        i = 0
        while remaining > 0:
            cur = min(aligned, remaining)
            n_blocks = cur // last_dim
            for j in range(n_blocks):
                src_idx = permute_a[i + j * last_dim].item()
                ue.ue_memcpy_from_dram(input_dram_addr + src_idx * bpe, last_dim * bpe, 0,
                    URAM_START_ADDR + (j * last_dim) // UE_VECTOR_SIZE,
                    URAM_SECTION.URAM_A.value, inst_id)
                ue.wait_queue(); inst_id += 1
            ue.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, URAM_START_ADDR,
                out_addr, n_blocks * last_dim * bpe, inst_id)
            ue.wait_queue(); inst_id += 1
            remaining -= cur; out_addr += n_blocks * last_dim * bpe; i += cur
        return (1, output_shape)

    # Case 2: last dim changes — Q1 + transpose + Q3
    remaining_for_q1 = [i for i in range(n + 1) if i != k and i != n]
    q1 = remaining_for_q1 + [k, n]
    q1_is_identity = all(q1[i] == i for i in range(n + 1))
    dims_after_q1 = [dims[q1[i]] for i in range(n + 1)]

    M_transpose = dims_after_q1[n - 1]
    N_transpose = dims_after_q1[n]
    M_aligned = ((M_transpose - 1) // UE_VECTOR_SIZE + 1) * UE_VECTOR_SIZE
    batch_size = 1
    for i in range(n - 1):
        batch_size *= dims_after_q1[i]
    dims_after_transpose = list(dims_after_q1[:n - 1]) + [N_transpose, M_aligned]

    current_dim_at_pos = list(q1[:n - 1]) + [n, k]
    pos_of_orig_dim = [0] * (n + 1)
    for pos, orig_dim in enumerate(current_dim_at_pos):
        pos_of_orig_dim[orig_dim] = pos
    q3 = [pos_of_orig_dim[permute_indices[i]] for i in range(n + 1)]
    q3_is_identity = all(q3[i] == i for i in range(n + 1))
    output_shape = tuple(dims_after_transpose[q3[i]] for i in range(n + 1))

    transposed_total = batch_size * N_transpose * M_aligned
    safe_temp = temp_dram_start
    if q1_is_identity:
        p2_in = input_dram_addr
    else:
        p2_in = safe_temp; safe_temp += total_elements * bpe
    if q3_is_identity:
        p2_out = output_dram_addr
    else:
        p2_out = safe_temp

    # Phase 1: Q1 permute (last-dim-fixed DMA gather)
    if not q1_is_identity:
        q1_pa = torch.arange(total_elements, dtype=torch.int32).reshape(*dims).permute(*q1).contiguous().flatten()
        last_dim = dims[n]
        out_addr = p2_in; remaining = total_elements
        aligned = (URAM_NEAR_FULL_ELEMENTS // (UE_VECTOR_SIZE * last_dim)) * UE_VECTOR_SIZE * last_dim
        i = 0
        while remaining > 0:
            cur = min(aligned, remaining); n_blocks = cur // last_dim
            for j in range(n_blocks):
                ue.ue_memcpy_from_dram(input_dram_addr + q1_pa[i + j * last_dim].item() * bpe,
                    last_dim * bpe, 0, URAM_START_ADDR + (j * last_dim) // UE_VECTOR_SIZE,
                    URAM_SECTION.URAM_A.value, inst_id)
                ue.wait_queue(); inst_id += 1
            ue.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, URAM_START_ADDR,
                out_addr, n_blocks * last_dim * bpe, inst_id)
            ue.wait_queue(); inst_id += 1
            remaining -= cur; out_addr += n_blocks * last_dim * bpe; i += cur

    # Phase 2: Batched transpose (identity dot-product)
    input_uram_addr = URAM_START_ADDR
    ue.ue_memcpy_from_dram(params_dram_addr, UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe,
        0, input_uram_addr, URAM_SECTION.URAM_A.value, inst_id)
    ue.wait_queue(); inst_id += 1

    max_N_chunk = min(((URAM_NEAR_FULL_ELEMENTS // N_transpose) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE, M_aligned)
    max_M_chunk = min(N_transpose, URAM_HALF_ELEMENTS // N_transpose, URAM_HALF_ELEMENTS // max_N_chunk)
    in_stride = M_transpose * N_transpose * bpe
    out_stride = M_aligned * N_transpose * bpe

    for batch in range(batch_size):
        cur_in = p2_in + batch * in_stride
        cur_out = p2_out + batch * out_stride
        remaining_M = N_transpose; start_vec = 0; out_chunk = cur_out

        while remaining_M > 0:
            cur_M = min(max_M_chunk, remaining_M)
            output_uram = UE_VECTOR_SIZE; remaining_N = M_aligned
            weight_addr = cur_in; out_offset = out_chunk

            while remaining_N > 0:
                cur_N = min(max_N_chunk, remaining_N)
                ue.ue_memcpy_from_dram(weight_addr, cur_N * N_transpose * bpe,
                    0, URAM_START_ADDR, URAM_SECTION.URAM_B.value, inst_id)
                ue.wait_queue(); inst_id += 1

                for i in range(cur_M):
                    abs_row = start_vec + i
                    vec_idx = abs_row % UE_VECTOR_SIZE
                    col_block = abs_row // UE_VECTOR_SIZE
                    ue.start_queue(0, 0, N_transpose // UE_VECTOR_SIZE, LALU_MODE.BYPASS.value, 0, 0,
                        URAM_SECTION.URAM_A.value, 0, 0, output_uram, URAM_WRITE_SRC.URAM_WRITE_BACK.value,
                        UE_MODE.BF16_DOT_PRODUCT, 0, input_uram_addr + vec_idx, URAM_START_ADDR + col_block,
                        1, 0, cur_N * N_transpose, cur_N, inst_id)
                    inst_id += 1; ue.wait_queue()
                    ue.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, output_uram,
                        out_offset + i * M_aligned * bpe, cur_N * bpe, inst_id)
                    ue.wait_queue(); inst_id += 1

                remaining_N -= cur_N; out_offset += cur_N * bpe; weight_addr += cur_N * N_transpose * bpe
            out_chunk += cur_M * M_aligned * bpe; remaining_M -= cur_M; start_vec += cur_M

    # Phase 3: Q3 permute (last-dim-fixed DMA gather)
    if not q3_is_identity:
        q3_pa = torch.arange(transposed_total, dtype=torch.int32).reshape(*dims_after_transpose).permute(*q3).contiguous().flatten()
        last_dim = M_aligned
        out_addr = output_dram_addr; remaining = transposed_total
        aligned = (URAM_NEAR_FULL_ELEMENTS // (UE_VECTOR_SIZE * last_dim)) * UE_VECTOR_SIZE * last_dim
        i = 0
        while remaining > 0:
            cur = min(aligned, remaining); n_blocks = cur // last_dim
            for j in range(n_blocks):
                ue.ue_memcpy_from_dram(p2_out + q3_pa[i + j * last_dim].item() * bpe,
                    last_dim * bpe, 0, URAM_START_ADDR + (j * last_dim) // UE_VECTOR_SIZE,
                    URAM_SECTION.URAM_A.value, inst_id)
                ue.wait_queue(); inst_id += 1
            ue.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, URAM_START_ADDR,
                out_addr, n_blocks * last_dim * bpe, inst_id)
            ue.wait_queue(); inst_id += 1
            remaining -= cur; out_addr += n_blocks * last_dim * bpe; i += cur

    return (2, output_shape)
# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def load_config(config_path=None):
    """Load parakeet_config.json."""
    if config_path is None:
        config_path = os.path.join(SCRIPT_DIR, "parakeet_config.json")
    with open(config_path, "r") as f:
        return json.load(f)
def pad_to_multiple(n, multiple):
    return ((n + multiple - 1) // multiple) * multiple
def conv2d_outsize(H, k=3, s=2, p=1):
    """Output spatial dim for conv2d/pool with given kernel, stride, padding."""
    return (H + 2 * p - k) // s + 1
def hw_vs_cpu(name, hw_tensor, ref_tensor, cos_threshold=0.99):
    """Compare HW result to CPU reference. Asserts on NaN or cosine failure."""
    h = hw_tensor.float().flatten()
    r = ref_tensor.float().flatten()
    assert not torch.isnan(h).any(), f"{name}: HW output contains NaN"
    assert not torch.isnan(r).any(), f"{name}: CPU reference contains NaN"
    n = min(len(h), len(r))
    cos = F.cosine_similarity(h[:n].unsqueeze(0), r[:n].unsqueeze(0)).item()
    mae = (h[:n] - r[:n]).abs().mean().item()
    max_err = (h[:n] - r[:n]).abs().max().item()
    print(f"  [{name}] cos={cos:.6f}  mae={mae:.6f}  max_err={max_err:.6f}  "
          f"hw_norm={h[:n].norm():.4f}  ref_norm={r[:n].norm():.4f}")
    assert cos >= cos_threshold, (
        f"MISMATCH {name}: cos={cos:.6f} < {cos_threshold}  "
        f"mae={mae:.6f}  max_err={max_err:.6f}")
def rel_shift(x):
    """Relative position shift (skew trick): (B, H, T, 2T-1) -> (B, H, T, T)."""
    B, H, T, P = x.shape
    zero_pad = torch.zeros(B, H, T, 1, device=x.device, dtype=x.dtype)
    x_padded = torch.cat([zero_pad, x], dim=-1)
    x_padded = x_padded.view(B, H, P + 1, T)
    x = x_padded[:, :, 1:].reshape(B, H, T, P)
    x = x[:, :, :, :P // 2 + 1]
    return x
def compile_and_run(engine, emit_fn):
    """Compile a block of HW instructions and execute it."""
    engine.clear_capture_buffer()
    engine.start_capture()
    engine.generate_instruction_flag_clear()
    emit_fn()
    engine.stop_capture()
    engine.generate_instruction_halt()
    prog = engine.get_program_dram_addr()
    engine.write_captured_instructions_to_dram(prog)
    engine.allocate_program_dram(engine.get_capture_instruction_size_bytes())
    engine.program_execute(prog)
def read_dram(engine, addr, numel):
    """Read bf16 tensor from accelerator DRAM."""
    buf = torch.zeros(numel, dtype=torch.bfloat16)
    engine.dma_read(DMA_DEVICE_C2H, addr, buf, numel * engine.bytes_per_element)
    return buf
# ---------------------------------------------------------------------------
# Host-side mel spectrogram (runs on CPU, not accelerator)
# ---------------------------------------------------------------------------
def compute_mel_spectrogram(waveform, cfg, ckpt_sd=None):
    """waveform: (B, samples) float32 → (B, T_mel, 128) bf16.

    Args:
        ckpt_sd: checkpoint state_dict — used to load the mel filterbank
                 and STFT window from 'preprocessor.featurizer.fb' and
                 'preprocessor.featurizer.window'.
    """
    pre = cfg["preprocessing"]
    n_fft = pre["n_fft"]
    hop_length = pre["hop_length"]
    win_length = pre["win_length"]
    if ckpt_sd is not None:
        fb = ckpt_sd["preprocessor.featurizer.fb"].float()       # (1, 128, 257)
        window = ckpt_sd["preprocessor.featurizer.window"].float()  # (400,)
    else:
        raise RuntimeError("Checkpoint state_dict required for mel filterbank and window")
    stft = torch.stft(waveform.float(), n_fft, hop_length, win_length,
                       window=window, center=True, pad_mode="reflect",
                       return_complex=True)
    mag = stft.abs()
    power = mag * mag
    mel = torch.matmul(fb, power)                      # (B, 128, T)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    mean = mel.mean(dim=-1, keepdim=True)
    var = mel.var(dim=-1, keepdim=True, unbiased=True)
    std = torch.clamp(torch.sqrt(var), min=1e-5)
    mel = (mel - mean) / std
    return mel.transpose(1, 2).to(torch.bfloat16)      # (B, T_mel, 128)
# ---------------------------------------------------------------------------
# Parakeet Unified Engine
# ---------------------------------------------------------------------------
# Parakeet DRAM partition over 4GB address space:
#   2.5 GB params / 750 MB tensors / 750 MB programs
PARAKEET_PARAMS_BASE  = 0x00000000   # 3 GB for weights + identities + Toeplitz DW conv matrices
PARAKEET_TENSOR_BASE  = 0xC0000000   # 512 MB for intermediate activations
PARAKEET_PROGRAM_BASE = 0xE0000000   # 512 MB for compiled instruction programs
class Parakeet_UnifiedEngine(UnifiedEngine):
    """UnifiedEngine subclass for Parakeet-TDT-0.6B."""

    def __init__(self, script_dir=None, clock_period_ns=None, fp4=False, int4=False):
        super().__init__(BASE_ADDR=user_dma_core.UE_0_BASE_ADDR, params_dram_base=PARAKEET_PARAMS_BASE, tensor_dram_base=PARAKEET_TENSOR_BASE, program_dram_base=PARAKEET_PROGRAM_BASE, clock_period_ns=clock_period_ns)
        self.script_dir = script_dir or SCRIPT_DIR
        self.fp4 = fp4
        self.int4 = int4
        # Unified quant mode: None, 'fp4', or 'int4'
        self.quant = 'fp4' if fp4 else ('int4' if int4 else None)
        # Hang prevention: stop stale execution, write HALT to program base
        self.dram_inst_running(False)
        self.start_capture()
        self.generate_instruction_halt()
        self.stop_capture()
        halt_bytes = bytearray()
        for inst in self.capture_buffer:
            halt_bytes.extend(inst.get_bytes())
        self.dma_write(DMA_DEVICE_H2C, PARAKEET_PROGRAM_BASE, halt_bytes, len(halt_bytes))
        self.clear_capture_buffer()
        self._cfg = load_config()
        enc = self._cfg["encoder"]
        pred = self._cfg["predictor"]
        jnt = self._cfg["joint"]
        hw = self._cfg["hardware"]
        self.d_model = enc["d_model"]           # 1024
        self.num_layers = enc["num_layers"]     # 24
        self.num_heads = enc["num_heads"]       # 8
        self.head_dim = enc["head_dim"]         # 128
        self.ff_dim = enc["ff_dim"]             # 4096
        self.conv_kernel = enc["conv_kernel"]   # 9
        self.conv_pad = enc["conv_pad"]         # 4
        self.sub_channels = enc["sub_channels"] # 256
        self.n_mels = enc["n_mels"]             # 128
        self.pred_hidden = pred["hidden_size"]  # 640
        self.vocab_size = pred["vocab_size"]    # 8193
        self.joint_hidden = jnt["hidden_size"]  # 640
        self.joint_output_padded = jnt["output_size_padded"]
        self.blank_id = jnt["blank_id"]
        self.tdt_durations = jnt["tdt_durations"]
        self.max_symbols_per_step = jnt["max_symbols_per_step"]
        self.block_size = hw["block_size"]      # 64
        self.bytes_per_element = enc["bytes_per_element"]
    # XDMA driver has a transfer size limit (~16-64MB per os.write call).
    # Chunk large DMA uploads to avoid silent failures.
    DMA_CHUNK_BYTES = 1 * 1024 * 1024  # 1 MB per chunk
    def dma_to_accelerator_memory(self, dma_address, data):
        """Chunked DMA write that handles large transfers."""
        assert data.dtype == torch.bfloat16, "Data must be in bf16 format"
        flat = data.contiguous().flatten()
        total_bytes = flat.numel() * 2
        if total_bytes <= self.DMA_CHUNK_BYTES:
            self.dma_write(DMA_DEVICE_H2C, dma_address, flat, total_bytes)
            return
        elems_per_chunk = self.DMA_CHUNK_BYTES // 2
        offset = 0
        while offset < flat.numel():
            chunk_elems = min(elems_per_chunk, flat.numel() - offset)
            chunk = flat[offset:offset + chunk_elems].contiguous()
            addr = dma_address + offset * 2
            self.dma_write(DMA_DEVICE_H2C, addr, chunk, chunk_elems * 2)
            offset += chunk_elems
    def _alloc_write(self, tensor):
        """Allocate params DRAM, write bf16 tensor via chunked DMA. Returns DRAM address."""
        t = tensor.to(torch.bfloat16).contiguous()
        addr = self.get_params_dram_addr()
        nbytes = t.numel() * 2
        self.allocate_params_dram(nbytes)
        self.dma_to_accelerator_memory(addr, t)
        return addr
    def _stage_dw_conv1d(self, weight):
        """(C,1,K) -> (C,64) im2col padded."""
        C, _, K = weight.shape
        padded_K = pad_to_multiple(K, self.block_size)
        w_flat = weight.reshape(C, K).to(torch.bfloat16)
        w_padded = torch.zeros(C, padded_K, dtype=torch.bfloat16)
        w_padded[:, :K] = w_flat
        return self._alloc_write(w_padded)
    def _split_pw_conv1(self, weight):
        """(2D,D,1) -> (addr_a, addr_b) each (D,D)."""
        w = weight.squeeze(-1).to(torch.bfloat16)
        D = w.shape[0] // 2
        return self._alloc_write(w[:D].contiguous()), self._alloc_write(w[D:].contiguous())
    def _stage_sub_conv2d(self, weight, out_ch, in_ch, k=3):
        """Stage 2D conv weight for im2col: (out_ch, in_ch, k, k) -> (out_ch, k*k*in_ch padded to 64)."""
        w = weight.to(torch.bfloat16).reshape(out_ch, -1)
        flat_k = w.shape[1]
        padded = pad_to_multiple(flat_k, self.block_size)
        w_pad = torch.zeros(out_ch, padded, dtype=torch.bfloat16)
        w_pad[:, :flat_k] = w
        return self._alloc_write(w_pad), flat_k, padded
    @staticmethod
    def ensure_model_files():
        """Download Parakeet-TDT-0.6B from HuggingFace if not present."""
        if os.path.exists(WEIGHTS_PATH) and os.path.exists(TOKENIZER_PATH):
            return
        model_dir = os.path.dirname(WEIGHTS_PATH)
        os.makedirs(model_dir, exist_ok=True)
        print(f"Model files not found, downloading to {model_dir} ...")
        from huggingface_hub import hf_hub_download
        import tarfile
        nemo_path = hf_hub_download(
            repo_id="nvidia/parakeet-tdt-0.6b-v3",
            filename="parakeet-tdt-0.6b-v3.nemo",
            cache_dir=model_dir)
        print(f"  Extracting from {nemo_path} ...")
        with tarfile.open(nemo_path, "r") as tar:
            for member in tar.getmembers():
                if member.name.endswith("model_weights.ckpt"):
                    member.name = "model_weights.ckpt"
                    tar.extract(member, model_dir)
                elif "tokenizer.model" in member.name:
                    member.name = "tokenizer.model"
                    tar.extract(member, model_dir)
        if not os.path.exists(WEIGHTS_PATH):
            sys.exit(f"Failed to extract model_weights.ckpt")
        if not os.path.exists(TOKENIZER_PATH):
            sys.exit(f"Failed to extract tokenizer.model")
        print("  Download complete.")
    def weight_init(self):
        """Load checkpoint, stage all weights to DRAM."""
        self.ensure_model_files()
        ckpt_path = WEIGHTS_PATH
        print(f"Loading weights from {ckpt_path} ...")
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        print(f"  {sum(v.numel() for v in sd.values() if hasattr(v, 'numel')):,} parameters")
        D, FF, H = self.d_model, self.ff_dim, self.pred_hidden
        self.w = {}

        # Identity matrices for LALU activation (sigmoid, softmax, etc.)
        self.w["IDENTITY_1024"] = allocate_identity(self, D)
        self.w["IDENTITY_4096"] = allocate_identity(self, FF)
        self.w["IDENTITY_640"] = allocate_identity(self, self.joint_hidden)
        # 64x64 identity for bf16_smart_permute_core transpose operations.
        self.w["IDENTITY_64"] = allocate_identity(self, UE_VECTOR_SIZE)

        # Subsampling weights for hardware im2col
        # Stage 0: Conv2d(1->256, k=3, s=2, p=1)
        self.w["SUB_CONV0_W"], self.sub0_flat_k, self.sub0_padded_k = \
            self._stage_sub_conv2d(sd["encoder.pre_encode.conv.0.weight"], 256, 1, 3)
        self.w["SUB_CONV0_B"] = self._alloc_write(sd["encoder.pre_encode.conv.0.bias"])
        # Stage 1: DW Conv2d(256, k=3, s=2, p=1) + PW Conv2d(256->256)
        self.w["SUB_DW1_W"] = self._stage_dw_conv1d(
            sd["encoder.pre_encode.conv.2.weight"].reshape(256, 1, 9))
        self.w["SUB_DW1_B"] = self._alloc_write(sd["encoder.pre_encode.conv.2.bias"])
        self.w["SUB_PW1_W"] = self._alloc_write(sd["encoder.pre_encode.conv.3.weight"].reshape(256, 256))
        self.w["SUB_PW1_B"] = self._alloc_write(sd["encoder.pre_encode.conv.3.bias"])
        # Stage 2: DW Conv2d(256, k=3, s=2, p=1) + PW Conv2d(256->256)
        self.w["SUB_DW2_W"] = self._stage_dw_conv1d(
            sd["encoder.pre_encode.conv.5.weight"].reshape(256, 1, 9))
        self.w["SUB_DW2_B"] = self._alloc_write(sd["encoder.pre_encode.conv.5.bias"])
        self.w["SUB_PW2_W"] = self._alloc_write(sd["encoder.pre_encode.conv.6.weight"].reshape(256, 256))
        self.w["SUB_PW2_B"] = self._alloc_write(sd["encoder.pre_encode.conv.6.bias"])
        # Final linear
        self.w["SUB_OUT_W"] = self._alloc_write(sd["encoder.pre_encode.out.weight"])
        self.w["SUB_OUT_B"] = self._alloc_write(sd["encoder.pre_encode.out.bias"])

        # Per-layer conformer weights
        # Expected (N, K) element counts for each quantized weight key
        _QUANT_EXPECTED = {
            "FF1_W1_LO": FF//2 * D, "FF1_W1_HI": FF//2 * D,
            "FF1_W2_LO": D * FF//2, "FF1_W2_HI": D * FF//2,
            "FF2_W1_LO": FF//2 * D, "FF2_W1_HI": FF//2 * D,
            "FF2_W2_LO": D * FF//2, "FF2_W2_HI": D * FF//2,
            "ATTN_Q_W": D*D, "ATTN_K_W": D*D, "ATTN_V_W": D*D,
            "ATTN_POS_W": D*D, "ATTN_OUT_W": D*D,
            "CONV_PW1A_W": D*D, "CONV_PW1B_W": D*D, "CONV_PW2_W": D*D,
        }
        _QUANT_KEYS = set(_QUANT_EXPECTED.keys())
        quant_cache = None
        if self.quant:
            bin_path = FP4_WEIGHTS_PATH if self.quant == 'fp4' else INT4_WEIGHTS_PATH
            if not os.path.exists(bin_path):
                print(f"  {self.quant.upper()} weights not found, generating...")
                generate_quantized_weights(self.quant)
            print(f"  Loading {self.quant.upper()} encoder weights from {bin_path}...")
            quant_cache = load_quantized_weight_cache(self.quant)
            print(f"  {self.quant.upper()} cache: {len(quant_cache)} tensors loaded")
        self.layer_addrs = []
        for i in range(self.num_layers):
            la = {}
            pfx = f"encoder.layers.{i}"
            ln_map = {"LN_FF1": "norm_feed_forward1", "LN_ATTN": "norm_self_att",
                      "LN_CONV": "norm_conv", "LN_FF2": "norm_feed_forward2", "LN_OUT": "norm_out"}
            for our_key, nemo_key in ln_map.items():
                la[f"{our_key}_WEIGHT"] = self._alloc_write(sd[f"{pfx}.{nemo_key}.weight"])
                la[f"{our_key}_BIAS"] = self._alloc_write(sd[f"{pfx}.{nemo_key}.bias"])
            if self.quant:
                # Load pre-quantized weights from bin file
                for key in _QUANT_KEYS:
                    cache_key = f"layer.{i}.{key}"
                    raw = quant_cache[cache_key]
                    n_blocks = len(raw) // 34
                    n_elements = n_blocks * 64
                    expected = _QUANT_EXPECTED[key]
                    assert n_elements == expected, (
                        f"{self.quant.upper()} shape mismatch: layer {i} {key}: got {n_elements} elements, "
                        f"expected {expected}. Delete encoder_{self.quant}.bin and regenerate.")
                    la[key] = store_quantized_weight(self, raw)
            else:
                # BF16 weights from checkpoint
                w1 = sd[f"{pfx}.feed_forward1.linear1.weight"]  # (4096, 1024)
                la["FF1_W1_LO"] = self._alloc_write(w1[:FF//2, :].contiguous())
                la["FF1_W1_HI"] = self._alloc_write(w1[FF//2:, :].contiguous())
                w2 = sd[f"{pfx}.feed_forward1.linear2.weight"]  # (1024, 4096)
                la["FF1_W2_LO"] = self._alloc_write(w2[:, :FF//2].contiguous())
                la["FF1_W2_HI"] = self._alloc_write(w2[:, FF//2:].contiguous())
                w1 = sd[f"{pfx}.feed_forward2.linear1.weight"]
                la["FF2_W1_LO"] = self._alloc_write(w1[:FF//2, :].contiguous())
                la["FF2_W1_HI"] = self._alloc_write(w1[FF//2:, :].contiguous())
                w2 = sd[f"{pfx}.feed_forward2.linear2.weight"]
                la["FF2_W2_LO"] = self._alloc_write(w2[:, :FF//2].contiguous())
                la["FF2_W2_HI"] = self._alloc_write(w2[:, FF//2:].contiguous())
                la["ATTN_Q_W"] = self._alloc_write(sd[f"{pfx}.self_attn.linear_q.weight"])
                la["ATTN_K_W"] = self._alloc_write(sd[f"{pfx}.self_attn.linear_k.weight"])
                la["ATTN_V_W"] = self._alloc_write(sd[f"{pfx}.self_attn.linear_v.weight"])
                la["ATTN_POS_W"] = self._alloc_write(sd[f"{pfx}.self_attn.linear_pos.weight"])
                la["ATTN_OUT_W"] = self._alloc_write(sd[f"{pfx}.self_attn.linear_out.weight"])
                pw1_a, pw1_b = self._split_pw_conv1(sd[f"{pfx}.conv.pointwise_conv1.weight"])
                la["CONV_PW1A_W"], la["CONV_PW1B_W"] = pw1_a, pw1_b
                la["CONV_PW2_W"] = self._alloc_write(sd[f"{pfx}.conv.pointwise_conv2.weight"].squeeze(-1))
            la["ATTN_BIAS_U"] = self._alloc_write(sd[f"{pfx}.self_attn.pos_bias_u"].reshape(-1))
            la["ATTN_BIAS_V"] = self._alloc_write(sd[f"{pfx}.self_attn.pos_bias_v"].reshape(-1))
            la["CONV_DW_W"] = self._stage_dw_conv1d(sd[f"{pfx}.conv.depthwise_conv.weight"])
            bn_w = sd[f"{pfx}.conv.batch_norm.weight"]
            bn_b = sd[f"{pfx}.conv.batch_norm.bias"]
            bn_m = sd[f"{pfx}.conv.batch_norm.running_mean"]
            bn_v = sd[f"{pfx}.conv.batch_norm.running_var"]
            s_addr, sh_addr, _ = batch_norm_fuse_params(self, bn_w, bn_b, bn_m, bn_v)
            la["CONV_BN_SCALE"], la["CONV_BN_SHIFT"] = s_addr, sh_addr
            self.layer_addrs.append(la)
            print(f"\r  Layer {i:2d}/{self.num_layers - 1} staged", end="", flush=True)

        print()
        if self.quant:
            q_count = sum(1 for la in self.layer_addrs for v in la.values() if isinstance(v, tuple))
            print(f"  Encoder: {q_count} {self.quant.upper()} weight tensors loaded to DRAM")
        # Predictor weights
        self.w["EMBED"] = self._alloc_write(sd["decoder.prediction.embed.weight"])
        for i in range(2):
            self.w[f"LSTM_WIH{i}"] = self._alloc_write(sd[f"decoder.prediction.dec_rnn.lstm.weight_ih_l{i}"])
            self.w[f"LSTM_WHH{i}"] = self._alloc_write(sd[f"decoder.prediction.dec_rnn.lstm.weight_hh_l{i}"])
            self.w[f"LSTM_BIH{i}"] = self._alloc_write(sd[f"decoder.prediction.dec_rnn.lstm.bias_ih_l{i}"])
            self.w[f"LSTM_BHH{i}"] = self._alloc_write(sd[f"decoder.prediction.dec_rnn.lstm.bias_hh_l{i}"])

        # Joint weights
        self.w["JOINT_ENC_W"] = self._alloc_write(sd["joint.enc.weight"])
        self.w["JOINT_ENC_B"] = self._alloc_write(sd["joint.enc.bias"])
        self.w["JOINT_PRED_W"] = self._alloc_write(sd["joint.pred.weight"])
        self.w["JOINT_PRED_B"] = self._alloc_write(sd["joint.pred.bias"])

        # Split joint output: token (8256x640) + duration (64x640)
        out_w = sd["joint.joint_net.2.weight"].to(torch.bfloat16)
        out_b = sd["joint.joint_net.2.bias"].to(torch.bfloat16)
        N_tok, N_tok_pad = self.vocab_size, pad_to_multiple(self.vocab_size, self.block_size)
        N_dur, N_dur_pad = len(self.tdt_durations), pad_to_multiple(len(self.tdt_durations), self.block_size)
        w_tok = torch.zeros(N_tok_pad, self.joint_hidden, dtype=torch.bfloat16)
        w_tok[:N_tok] = out_w[:N_tok]
        # Padding biases must be -inf so padding positions never win argmax.
        b_tok = torch.full((N_tok_pad,), -1e4, dtype=torch.bfloat16)
        b_tok[:N_tok] = out_b[:N_tok]
        self.w["JOINT_OUT_TOK_W"] = self._alloc_write(w_tok)
        self.w["JOINT_OUT_TOK_B"] = self._alloc_write(b_tok)
        w_dur = torch.zeros(N_dur_pad, self.joint_hidden, dtype=torch.bfloat16)
        w_dur[:N_dur] = out_w[N_tok:N_tok + N_dur]
        b_dur = torch.full((N_dur_pad,), -1e4, dtype=torch.bfloat16)
        b_dur[:N_dur] = out_b[N_tok:N_tok + N_dur]
        self.w["JOINT_OUT_DUR_W"] = self._alloc_write(w_dur)
        self.w["JOINT_OUT_DUR_B"] = self._alloc_write(b_dur)
        # Store checkpoint for mel spectrogram (filterbank + window)
        self._ckpt_sd = sd
        params_used = self.get_params_dram_usage()
        params_limit = self._tensor_dram_base - self._params_dram_base
        print(f"  Weight staging complete: {params_used / 1024**2:.1f} MB in DRAM "
              f"(budget: {params_limit / 1024**2:.0f} MB)")
        assert params_used <= params_limit, (
            f"PARAMS DRAM OVERFLOW: {params_used/1024**2:.1f} MB used > "
            f"{params_limit/1024**2:.0f} MB budget. "
            f"Params bleed into tensor region — corrupts activations!"
        )
    def tensor_init(self, L_pad):
        """Allocate intermediate DRAM buffers."""
        # Zero entire tensor DRAM region up front to prevent stale NaN.
        tensor_budget = self._program_dram_base - self._tensor_dram_base
        print(f"  Zeroing tensor DRAM ({tensor_budget / 1024**2:.0f} MB budget)...")
        ZERO_CHUNK = 512 * 1024  # 512K elements = 1MB per chunk
        offset = 0
        total_elems = tensor_budget // 2
        while offset < total_elems:
            chunk_elems = min(ZERO_CHUNK, total_elems - offset)
            z = torch.zeros(chunk_elems, dtype=torch.bfloat16)
            self.dma_to_accelerator_memory(self._tensor_dram_base + offset * 2, z)
            offset += chunk_elems
        print(f"  Tensor DRAM zeroed.")

        bpe = self.bytes_per_element
        D, FF, H = self.d_model, self.ff_dim, self.pred_hidden
        dk = self.head_dim
        SC = self.sub_channels  # 256
        P_pad = pad_to_multiple(2 * L_pad - 1, self.block_size)
        N_tok_pad = pad_to_multiple(self.vocab_size, self.block_size)
        N_dur_pad = pad_to_multiple(len(self.tdt_durations), self.block_size)
        # Encoder intermediates
        self.INPUT_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.LN_OUT_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.FF_MID_DRAM = self.allocate_tensor_dram(L_pad * FF * bpe)
        self.FF_OUT_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.RESIDUAL_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.Q_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.K_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.V_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.POS_PROJ_DRAM = self.allocate_tensor_dram(P_pad * D * bpe)  # P_pad rows, not L_pad
        self.SCORE_DRAM = self.allocate_tensor_dram(L_pad * L_pad * bpe)
        self.ATTN_OUT_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.POS_EMB_DRAM = self.allocate_tensor_dram(P_pad * D * bpe)
        self.REL_SHIFT_DRAM = self.allocate_tensor_dram(L_pad * L_pad * bpe)
        self.CONV_A_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.CONV_B_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.CONV_T_DRAM = self.allocate_tensor_dram(D * L_pad * bpe)
        self.CONV_DW_DRAM = self.allocate_tensor_dram(D * L_pad * bpe)
        self.CONV_OUT_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.ATTN_VT_DRAM = self.allocate_tensor_dram(dk * L_pad * bpe)
        self.ATTN_MASK_DRAM = self.allocate_tensor_dram(L_pad * L_pad * bpe)
        self.PERMUTE_TEMP_DRAM = self.allocate_tensor_dram(D * L_pad * bpe)
        self.ENC_OUT_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        # L_pad-sized identity for softmax and SiLU in conv module
        self.IDENTITY_LPAD_DRAM = allocate_identity(self, L_pad)
        # Subsampling intermediates (host builds im2col patches, HW runs matmuls)
        T_mel_max = L_pad * 8
        H0, W0 = T_mel_max // 2, self.n_mels // 2  # after stage 0
        H1, W1 = H0 // 2, W0 // 2                  # after stage 1
        N0 = H0 * W0
        N1 = H1 * W1
        N1_pad = pad_to_multiple(N1, self.block_size)
        H2, W2_tmp = H1 // 2, W1 // 2
        N2 = H2 * W2_tmp
        N2_pad = pad_to_multiple(N2, self.block_size)
        N_dw_max_pad = max(N1_pad, N2_pad)
        max_patch_size = max(N0 * 64, SC * N_dw_max_pad * 64)
        self.SUB_PATCH_DRAM = self.allocate_tensor_dram(max_patch_size * bpe)
        self.SUB_OUT0_DRAM = self.allocate_tensor_dram(N0 * SC * bpe)
        self.SUB_DW_OUT_DRAM = self.allocate_tensor_dram(SC * N_dw_max_pad * bpe)
        self.SUB_PW_IN_DRAM = self.allocate_tensor_dram(N_dw_max_pad * SC * bpe)
        self.SUB_FLAT_DRAM = self.allocate_tensor_dram(L_pad * 4096 * bpe)
        # Decoder intermediates
        self.PRED_EMB_DRAM = self.allocate_tensor_dram(H * bpe)
        self.PRED_GATES_DRAM = self.allocate_tensor_dram(4 * H * bpe)
        self.PRED_GATES2_DRAM = self.allocate_tensor_dram(4 * H * bpe)
        self.PRED_H0_DRAM = self.allocate_tensor_dram(H * bpe)
        self.PRED_C0_DRAM = self.allocate_tensor_dram(H * bpe)
        self.PRED_H1_DRAM = self.allocate_tensor_dram(H * bpe)
        self.PRED_C1_DRAM = self.allocate_tensor_dram(H * bpe)
        self.PRED_OUT_DRAM = self.allocate_tensor_dram(H * bpe)
        self.JOINT_ENC_DRAM = self.allocate_tensor_dram(D * bpe)
        self.JOINT_PRED_DRAM = self.allocate_tensor_dram(self.joint_hidden * bpe)
        self.JOINT_SUM_DRAM = self.allocate_tensor_dram(self.joint_hidden * bpe)
        self.JOINT_TOK_DRAM = self.allocate_tensor_dram(N_tok_pad * bpe)
        self.JOINT_DUR_DRAM = self.allocate_tensor_dram(N_dur_pad * bpe)
        tensor_used = self.get_tensor_dram_usage()
        tensor_limit = self._program_dram_base - self._tensor_dram_base
        print(f"  Tensor alloc: {tensor_used / 1024**2:.1f} MB "
              f"(budget: {tensor_limit / 1024**2:.0f} MB)")
        assert tensor_used <= tensor_limit, (
            f"TENSOR DRAM OVERFLOW: {tensor_used/1024**2:.1f} MB used > "
            f"{tensor_limit/1024**2:.0f} MB budget. "
            f"Tensors bleed into program region!"
        )
    def compile_sub_stage0(self, N0, padded_k, SC):
        """HW program: im2col patches(N0,64) @ W(64,256) + bias, ReLU. Patches pre-built on host."""
        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()
        self.matmat_mul_core(M=N0, K=padded_k, N=SC,
            A_DRAM_ADDR=self.SUB_PATCH_DRAM, B_DRAM_ADDR=self.w["SUB_CONV0_W"],
            OUTPUT_DRAM_ADDR=self.SUB_OUT0_DRAM,
            C_DRAM_ADDR=self.w["SUB_CONV0_B"], relu_enable=True)
        self.stop_capture()
        self.generate_instruction_halt()
        prog = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(prog)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        return prog, 2 * N0 * padded_k * SC
    def compile_sub_stage_dw_pw(self, N_in, N_out, SC, dw_patch_addr, dw_w_key, dw_b_key, pw_w_key, pw_b_key):
        """HW program: DW patches(N_out,64) @ kernels via per-channel matmul, then PW(N_out,256)@(256,256)+bias+ReLU.
        DW im2col patches pre-built on host at dw_patch_addr as (SC, N_out_pad, 64).
        N_out is padded to N_out_pad (next multiple of block_size) for aligned matmuls."""
        bpe = self.bytes_per_element
        N_out_pad = pad_to_multiple(N_out, self.block_size)
        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()
        flops = 0
        # DW: per channel, kernel(1,64) @ patches(N_out_pad,64)^T = (1, N_out_pad)
        for ch in range(SC):
            kernel_addr = self.w[dw_w_key] + ch * 64 * bpe
            patch_addr = dw_patch_addr + ch * N_out_pad * 64 * bpe
            out_addr = self.SUB_DW_OUT_DRAM + ch * N_out_pad * bpe
            self.matmat_mul_core(M=1, K=64, N=N_out_pad,
                A_DRAM_ADDR=kernel_addr, B_DRAM_ADDR=patch_addr, OUTPUT_DRAM_ADDR=out_addr)
            flops += 2 * 64 * N_out_pad
        # DW bias: broadcast_add per channel
        dw_bias = torch.zeros(SC, dtype=torch.bfloat16)
        self.dma_read(DMA_DEVICE_C2H, self.w[dw_b_key], dw_bias, SC * bpe)
        for ch in range(SC):
            out_addr = self.SUB_DW_OUT_DRAM + ch * N_out_pad * bpe
            self.accelerator_memory_to_sram(out_addr, URAM_A_BASE, N_out_pad)
            self.broadcast_add(scalar=dw_bias[ch].float().item(),
                sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=N_out_pad)
            self.sram_to_accelerator_memory(URAM_A_BASE, out_addr, N_out_pad)
        # Chunked on-device transpose: (SC, N_out_pad) -> (N_out_pad, SC)
        chunk = SC  # 256
        for c_start in range(0, N_out_pad, chunk):
            c_end = min(c_start + chunk, N_out_pad)
            c_len = c_end - c_start
            c_len_pad = pad_to_multiple(c_len, self.block_size)
            for ch in range(SC):
                src = self.SUB_DW_OUT_DRAM + (ch * N_out_pad + c_start) * bpe
                dst = self.PERMUTE_TEMP_DRAM + ch * c_len_pad * bpe
                self.accelerator_memory_to_sram(src, URAM_A_BASE, c_len)
                self.sram_to_accelerator_memory(URAM_A_BASE, dst, c_len)
            bf16_smart_permute_core(self,
                dims=[SC, c_len_pad], permute_indices=[1, 0],
                input_dram_addr=self.PERMUTE_TEMP_DRAM,
                output_dram_addr=self.SUB_PW_IN_DRAM + c_start * SC * bpe,
                params_dram_addr=self.w["IDENTITY_64"],
                temp_dram_start=self.PERMUTE_TEMP_DRAM + SC * c_len_pad * bpe)
        # PW: (N_out_pad, 256) @ (256, 256) + bias, ReLU
        self.matmat_mul_core(M=N_out_pad, K=SC, N=SC,
            A_DRAM_ADDR=self.SUB_PW_IN_DRAM, B_DRAM_ADDR=self.w[pw_w_key],
            OUTPUT_DRAM_ADDR=self.SUB_PW_IN_DRAM,
            C_DRAM_ADDR=self.w[pw_b_key], relu_enable=True)
        flops += 2 * N_out * SC * SC
        self.stop_capture()
        self.generate_instruction_halt()
        prog = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(prog)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        return prog, flops
    def compile_sub_linear(self, L_pad):
        """HW program: flatten linear (L_pad, 4096) @ (4096, 1024) + bias -> INPUT_DRAM."""
        D = self.d_model
        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()
        self.matmat_mul_core(M=L_pad, K=4096, N=D,
            A_DRAM_ADDR=self.SUB_FLAT_DRAM, B_DRAM_ADDR=self.w["SUB_OUT_W"],
            OUTPUT_DRAM_ADDR=self.INPUT_DRAM, C_DRAM_ADDR=self.w["SUB_OUT_B"])
        self.stop_capture()
        self.generate_instruction_halt()
        prog = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(prog)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        return prog, 2 * L_pad * 4096 * D
    def _im2col_conv2d(self, input_tensor, H_in, W_in, stride=2, padding=1):
        """Host-side: build im2col patch matrix for 3x3 conv2d using F.unfold (vectorized).
        input_tensor: (H_in, W_in) or (C, H_in, W_in) bf16.
        Single-channel: returns (N_out, 64) bf16.  Depthwise: returns (C, N_out_pad, 64)
        where N_out_pad is padded to the next multiple of block_size for aligned matmuls.
        """
        k = 3
        if input_tensor.dim() == 2:
            x = input_tensor.float().unsqueeze(0).unsqueeze(0)
            cols = F.unfold(x, k, padding=padding, stride=stride)
            H_out = (H_in + 2*padding - k) // stride + 1
            W_out = (W_in + 2*padding - k) // stride + 1
            N = cols.shape[2]
            patches = torch.zeros(N, 64, dtype=torch.bfloat16)
            patches[:, :9] = cols[0].t().to(torch.bfloat16)
            return patches, H_out, W_out
        else:
            C = input_tensor.shape[0]
            x = input_tensor.float().unsqueeze(1)
            cols = F.unfold(x.reshape(C, 1, H_in, W_in), k, padding=padding, stride=stride)
            H_out = (H_in + 2*padding - k) // stride + 1
            W_out = (W_in + 2*padding - k) // stride + 1
            N = cols.shape[2]
            N_pad = pad_to_multiple(N, self.block_size)
            patches = torch.zeros(C, N_pad, 64, dtype=torch.bfloat16)
            patches[:, :N, :9] = cols.permute(0, 2, 1).to(torch.bfloat16)
            return patches, H_out, W_out
    def prepare_attention_tiled_biases(self, L_pad):
        """Pre-tile attention bias_u and bias_v for bulk eltwise_add.
        For each layer and head, creates (L_pad, dk) tiled bias in DRAM.
        Must be called after weight_init() and before compile_encoder().
        """
        dk = self.head_dim  # 128
        bpe = self.bytes_per_element
        for layer_idx in range(self.num_layers):
            la = self.layer_addrs[layer_idx]
            for bias_key in ("ATTN_BIAS_U", "ATTN_BIAS_V"):
                # Read the full (H_heads * dk) bias from DRAM
                full_bias = torch.zeros(self.num_heads * dk, dtype=torch.bfloat16)
                self.dma_read(DMA_DEVICE_C2H, la[bias_key], full_bias, self.num_heads * dk * bpe)
                tiled_key = f"{bias_key}_TILED"
                head_addrs = []
                for h in range(self.num_heads):
                    head_bias = full_bias[h * dk:(h + 1) * dk]
                    # Tile: repeat dk-element bias L_pad times
                    tiled = head_bias.unsqueeze(0).expand(L_pad, dk).contiguous()
                    addr = self._alloc_write(tiled)
                    head_addrs.append(addr)
                la[tiled_key] = head_addrs
    _enc_matmul_quant_count = 0
    _enc_matmul_bf16_count = 0
    def _enc_matmul(self, M, K, N, A_DRAM_ADDR, w_addr, OUTPUT_DRAM_ADDR, **kw):
        """Encoder matmul: dispatches bf16 or quantized based on self.quant.
        w_addr is a single DRAM addr (bf16) or (scale_addr, data_addr) tuple (quantized)."""
        if self.quant and isinstance(w_addr, tuple):
            scale_addr, data_addr = w_addr
            Parakeet_UnifiedEngine._enc_matmul_quant_count += 1
            dtype = TYPE.FP4 if self.quant == 'fp4' else TYPE.INT4
            self.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=A_DRAM_ADDR,
                B_DRAM_ADDR=data_addr, OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                is_B_quantized=True, data_type=dtype, SCALE_DRAM_ADDR=scale_addr, **kw)
        else:
            Parakeet_UnifiedEngine._enc_matmul_bf16_count += 1
            self.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=A_DRAM_ADDR,
                B_DRAM_ADDR=w_addr, OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR, **kw)
    def compile_encoder(self, L_pad, toeplitz_addrs, bn_tiled):
        """Compile full 24-layer conformer encoder. Returns (program_addr, program_bytes).
        Must call prepare_attention_tiled_biases(L_pad) before this.
        Args:
            L_pad: padded sequence length
            toeplitz_addrs: list of 24 DRAM addresses for Toeplitz DW conv matrices
            bn_tiled: list of 24 (tiled_scale_addr, tiled_shift_addr) tuples
        """
        D, FF, H_heads, dk = self.d_model, self.ff_dim, self.num_heads, self.head_dim
        bpe = self.bytes_per_element
        P = 2 * L_pad - 1
        P_pad = pad_to_multiple(P, self.block_size)

        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()
        total_flops = 0

        print(f"  Compiling {self.num_layers} conformer layers")
        for layer_idx in range(self.num_layers):
            la = self.layer_addrs[layer_idx]

            # ===== FF1 (half-step residual, K-split) =====
            FF_HALF = FF // 2
            self.layer_norm_core_dram(M=L_pad, N=D,A_DRAM_ADDR=self.INPUT_DRAM, OUTPUT_DRAM_ADDR=self.LN_OUT_DRAM,GAMMA_DRAM_ADDR=la["LN_FF1_WEIGHT"], BETA_DRAM_ADDR=la["LN_FF1_BIAS"])
            # Up-proj split: two (L_pad, D) @ (D, FF/2) = (L_pad, FF/2) with SiLU
            self._enc_matmul(L_pad, D, FF_HALF, self.LN_OUT_DRAM, la["FF1_W1_LO"], self.FF_MID_DRAM, silu_enable=True)
            self._enc_matmul(L_pad, D, FF_HALF, self.LN_OUT_DRAM, la["FF1_W1_HI"], self.FF_MID_DRAM + L_pad * FF_HALF * bpe, silu_enable=True)
            total_flops += 2 * L_pad * D * FF
            # Down-proj split: two (L_pad, FF/2) @ (FF/2, D) + accumulate
            self._enc_matmul(L_pad, FF_HALF, D, self.FF_MID_DRAM, la["FF1_W2_LO"], self.FF_OUT_DRAM)
            self._enc_matmul(L_pad, FF_HALF, D, self.FF_MID_DRAM + L_pad * FF_HALF * bpe, la["FF1_W2_HI"], self.LN_OUT_DRAM)
            self.accelerator_memory_to_sram(self.FF_OUT_DRAM, URAM_A_BASE, L_pad * D)
            self.accelerator_memory_to_sram(self.LN_OUT_DRAM, URAM_B_BASE, L_pad * D)
            self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * D)
            self.sram_to_accelerator_memory(URAM_A_BASE, self.FF_OUT_DRAM, L_pad * D)
            total_flops += 2 * L_pad * FF * D
            half_step_residual_core_dram(self, M=L_pad, N=D,
                RESIDUAL_DRAM_ADDR=self.INPUT_DRAM, FF_DRAM_ADDR=self.FF_OUT_DRAM,
                OUTPUT_DRAM_ADDR=self.INPUT_DRAM)
            total_flops += 2 * L_pad * D

            # ===== Self-attention =====
            self.layer_norm_core_dram(M=L_pad, N=D, A_DRAM_ADDR=self.INPUT_DRAM, OUTPUT_DRAM_ADDR=self.LN_OUT_DRAM, GAMMA_DRAM_ADDR=la["LN_ATTN_WEIGHT"], BETA_DRAM_ADDR=la["LN_ATTN_BIAS"])
            self._enc_matmul(L_pad, D, D, self.LN_OUT_DRAM, la["ATTN_Q_W"], self.Q_DRAM)
            self._enc_matmul(L_pad, D, D, self.LN_OUT_DRAM, la["ATTN_K_W"], self.K_DRAM)
            self._enc_matmul(L_pad, D, D, self.LN_OUT_DRAM, la["ATTN_V_W"], self.V_DRAM)
            total_flops += 3 * 2 * L_pad * D * D
            self._enc_matmul(P_pad, D, D, self.POS_EMB_DRAM, la["ATTN_POS_W"], self.POS_PROJ_DRAM)
            total_flops += 2 * P_pad * D * D

            for h in range(H_heads):
                h_off = h * dk * bpe
                # Q_h + bias_u (TILED)
                self.accelerator_memory_to_sram(self.Q_DRAM + h_off, URAM_A_BASE, L_pad * dk,
                    stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                self.accelerator_memory_to_sram(la["ATTN_BIAS_U_TILED"][h], URAM_B_BASE, L_pad * dk)
                self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * dk)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_A_DRAM, L_pad * dk)
                # K_h
                self.accelerator_memory_to_sram(self.K_DRAM + h_off, URAM_A_BASE, L_pad * dk,
                    stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_B_DRAM, L_pad * dk)
                # Content scores
                self.matmat_mul_core(M=L_pad, K=dk, N=L_pad,
                    A_DRAM_ADDR=self.CONV_A_DRAM, B_DRAM_ADDR=self.CONV_B_DRAM,
                    OUTPUT_DRAM_ADDR=self.SCORE_DRAM)
                total_flops += 2 * L_pad * dk * L_pad
                # Positional: Q_h + bias_v (TILED)
                self.accelerator_memory_to_sram(self.Q_DRAM + h_off, URAM_A_BASE, L_pad * dk, stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                self.accelerator_memory_to_sram(la["ATTN_BIAS_V_TILED"][h], URAM_B_BASE, L_pad * dk)
                self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * dk)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_A_DRAM, L_pad * dk)
                # P_h
                self.accelerator_memory_to_sram(self.POS_PROJ_DRAM + h_off, URAM_A_BASE, P_pad * dk, stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_B_DRAM, P_pad * dk)
                self.matmat_mul_core(M=L_pad, K=dk, N=P_pad, A_DRAM_ADDR=self.CONV_A_DRAM, B_DRAM_ADDR=self.CONV_B_DRAM, OUTPUT_DRAM_ADDR=self.CONV_OUT_DRAM)
                total_flops += 2 * L_pad * dk * P_pad
                rel_shift_core_dram(self, L=L_pad, INPUT_DRAM_ADDR=self.CONV_OUT_DRAM, OUTPUT_DRAM_ADDR=self.REL_SHIFT_DRAM, input_row_stride=P_pad)
                # Combine content + pos
                score_elems = L_pad * L_pad
                self.accelerator_memory_to_sram(self.SCORE_DRAM, URAM_A_BASE, score_elems)
                self.accelerator_memory_to_sram(self.REL_SHIFT_DRAM, URAM_B_BASE, score_elems)
                self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, score_elems)
                # Scale (data already in URAM_A from eltwise_add above)
                inv_sqrt_dk = 1.0 / math.sqrt(dk)
                self.broadcast_mul(scalar=inv_sqrt_dk, sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=score_elems)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.SCORE_DRAM, score_elems)
                # Softmax + mask
                self.matmat_mul_core(M=L_pad, K=L_pad, N=L_pad, A_DRAM_ADDR=self.SCORE_DRAM, B_DRAM_ADDR=self.IDENTITY_LPAD_DRAM, OUTPUT_DRAM_ADDR=self.SCORE_DRAM, softmax_enable=True, C_DRAM_ADDR=self.ATTN_MASK_DRAM, bias_mode="full_matrix")
                total_flops += 2 * L_pad * L_pad * L_pad  # softmax identity matmul
                # V_h -> transpose -> attn @ V_h
                self.accelerator_memory_to_sram(self.V_DRAM + h_off, URAM_A_BASE, L_pad * dk, stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_B_DRAM, L_pad * dk)
                chunked_transpose_core_dram(self, M=L_pad, N=dk, input_dram_addr=self.CONV_B_DRAM, output_dram_addr=self.ATTN_VT_DRAM, identity_dram_addr=self.w["IDENTITY_64"], temp_dram_addr=self.PERMUTE_TEMP_DRAM)
                total_flops += (dk // UE_VECTOR_SIZE) * dk * 2 * UE_VECTOR_SIZE * pad_to_multiple(L_pad, UE_VECTOR_SIZE)  # V transpose dot-products
                self.matmat_mul_core(M=L_pad, K=L_pad, N=dk, A_DRAM_ADDR=self.SCORE_DRAM, B_DRAM_ADDR=self.ATTN_VT_DRAM, OUTPUT_DRAM_ADDR=self.CONV_A_DRAM)
                total_flops += 2 * L_pad * L_pad * dk
                # Write head output (strided)
                self.accelerator_memory_to_sram(self.CONV_A_DRAM, URAM_A_BASE, L_pad * dk)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.ATTN_OUT_DRAM + h_off, L_pad * dk, stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
            # Output projection + residual
            self._enc_matmul(L_pad, D, D, self.ATTN_OUT_DRAM, la["ATTN_OUT_W"], self.FF_OUT_DRAM)
            total_flops += 2 * L_pad * D * D
            self.accelerator_memory_to_sram(self.INPUT_DRAM, URAM_A_BASE, L_pad * D)
            self.accelerator_memory_to_sram(self.FF_OUT_DRAM, URAM_B_BASE, L_pad * D)
            self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * D)
            self.sram_to_accelerator_memory(URAM_A_BASE, self.INPUT_DRAM, L_pad * D)
            total_flops += L_pad * D

            # ===== Conv module =====
            self.layer_norm_core_dram(M=L_pad, N=D, A_DRAM_ADDR=self.INPUT_DRAM, OUTPUT_DRAM_ADDR=self.LN_OUT_DRAM, GAMMA_DRAM_ADDR=la["LN_CONV_WEIGHT"], BETA_DRAM_ADDR=la["LN_CONV_BIAS"])
            self._enc_matmul(L_pad, D, D, self.LN_OUT_DRAM, la["CONV_PW1A_W"], self.CONV_A_DRAM)
            self._enc_matmul(L_pad, D, D, self.LN_OUT_DRAM, la["CONV_PW1B_W"], self.CONV_B_DRAM, sigmoid_enable=True)
            total_flops += 2 * 2 * L_pad * D * D
            # GLU: a * sigmoid(b) — sigmoid fused into PW1b above
            self.accelerator_memory_to_sram(self.CONV_A_DRAM, URAM_A_BASE, L_pad * D)
            self.accelerator_memory_to_sram(self.CONV_B_DRAM, URAM_B_BASE, L_pad * D)
            self.eltwise_mul_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * D)
            self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_A_DRAM, L_pad * D)
            # Transpose (L_pad, D) -> (D, L_pad) for channel-first DW conv.
            chunked_transpose_core_dram(self, M=L_pad, N=D, input_dram_addr=self.CONV_A_DRAM, output_dram_addr=self.CONV_T_DRAM, identity_dram_addr=self.w["IDENTITY_64"], temp_dram_addr=self.PERMUTE_TEMP_DRAM)
            total_flops += (D // UE_VECTOR_SIZE) * D * 2 * UE_VECTOR_SIZE * pad_to_multiple(L_pad, UE_VECTOR_SIZE)  # transpose dot-products
            # DW Conv1d via Toeplitz: per-channel matmul(1, L_pad) @ T(L_pad, L_pad)
            t_addr = toeplitz_addrs[layer_idx]
            for ch in range(D):
                self.matmat_mul_core(M=1, K=L_pad, N=L_pad, A_DRAM_ADDR=self.CONV_T_DRAM + ch * L_pad * bpe, B_DRAM_ADDR=t_addr + ch * L_pad * L_pad * bpe, OUTPUT_DRAM_ADDR=self.CONV_DW_DRAM + ch * L_pad * bpe)
            total_flops += D * 2 * 1 * L_pad * L_pad  # actual HW compute (full Toeplitz matmul, not just 9-tap)
            batch_norm_core_dram(self, C=D, L=L_pad, A_DRAM_ADDR=self.CONV_DW_DRAM, OUTPUT_DRAM_ADDR=self.CONV_DW_DRAM, SCALE_DRAM_ADDR=la["CONV_BN_SCALE"], SHIFT_DRAM_ADDR=la["CONV_BN_SHIFT"], tiled_scale_addr=bn_tiled[layer_idx][0], tiled_shift_addr=bn_tiled[layer_idx][1])
            silu_core_dram(self, M=D, N=L_pad, A_DRAM_ADDR=self.CONV_DW_DRAM, OUTPUT_DRAM_ADDR=self.CONV_OUT_DRAM, IDENTITY_DRAM_ADDR=self.IDENTITY_LPAD_DRAM)
            total_flops += 2 * D * L_pad * L_pad  # SiLU identity matmul
            # Transpose (D, L_pad) -> (L_pad, D) from SiLU output.
            chunked_transpose_core_dram(self, M=D, N=L_pad, input_dram_addr=self.CONV_OUT_DRAM, output_dram_addr=self.CONV_T_DRAM, identity_dram_addr=self.w["IDENTITY_64"], temp_dram_addr=self.PERMUTE_TEMP_DRAM)
            total_flops += (L_pad // UE_VECTOR_SIZE) * L_pad * 2 * UE_VECTOR_SIZE * pad_to_multiple(D, UE_VECTOR_SIZE)  # transpose dot-products
            self._enc_matmul(L_pad, D, D, self.CONV_T_DRAM, la["CONV_PW2_W"], self.CONV_OUT_DRAM)
            total_flops += 2 * L_pad * D * D
            self.accelerator_memory_to_sram(self.INPUT_DRAM, URAM_A_BASE, L_pad * D)
            self.accelerator_memory_to_sram(self.CONV_OUT_DRAM, URAM_B_BASE, L_pad * D)
            self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * D)
            self.sram_to_accelerator_memory(URAM_A_BASE, self.INPUT_DRAM, L_pad * D)
            total_flops += L_pad * D

            # ===== FF2 (half-step residual, K-split) =====
            self.layer_norm_core_dram(M=L_pad, N=D, A_DRAM_ADDR=self.INPUT_DRAM, OUTPUT_DRAM_ADDR=self.LN_OUT_DRAM, GAMMA_DRAM_ADDR=la["LN_FF2_WEIGHT"], BETA_DRAM_ADDR=la["LN_FF2_BIAS"])
            # Up-proj split
            self._enc_matmul(L_pad, D, FF_HALF, self.LN_OUT_DRAM, la["FF2_W1_LO"], self.FF_MID_DRAM, silu_enable=True)
            self._enc_matmul(L_pad, D, FF_HALF, self.LN_OUT_DRAM, la["FF2_W1_HI"], self.FF_MID_DRAM + L_pad * FF_HALF * bpe, silu_enable=True)
            total_flops += 2 * L_pad * D * FF
            # Down-proj split + accumulate
            self._enc_matmul(L_pad, FF_HALF, D, self.FF_MID_DRAM, la["FF2_W2_LO"], self.FF_OUT_DRAM)
            self._enc_matmul(L_pad, FF_HALF, D, self.FF_MID_DRAM + L_pad * FF_HALF * bpe, la["FF2_W2_HI"], self.LN_OUT_DRAM)
            self.accelerator_memory_to_sram(self.FF_OUT_DRAM, URAM_A_BASE, L_pad * D)
            self.accelerator_memory_to_sram(self.LN_OUT_DRAM, URAM_B_BASE, L_pad * D)
            self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * D)
            self.sram_to_accelerator_memory(URAM_A_BASE, self.FF_OUT_DRAM, L_pad * D)
            total_flops += 2 * L_pad * FF * D
            half_step_residual_core_dram(self, M=L_pad, N=D,
                RESIDUAL_DRAM_ADDR=self.INPUT_DRAM, FF_DRAM_ADDR=self.FF_OUT_DRAM,
                OUTPUT_DRAM_ADDR=self.INPUT_DRAM)
            total_flops += 2 * L_pad * D
            # ===== Final LayerNorm =====
            self.layer_norm_core_dram(M=L_pad, N=D, A_DRAM_ADDR=self.INPUT_DRAM, OUTPUT_DRAM_ADDR=self.INPUT_DRAM, GAMMA_DRAM_ADDR=la["LN_OUT_WEIGHT"], BETA_DRAM_ADDR=la["LN_OUT_BIAS"])
            print(f"\r    Layer {layer_idx:2d}/{self.num_layers - 1} compiled", end="", flush=True)

        print()
        q_n = Parakeet_UnifiedEngine._enc_matmul_quant_count
        bf16_n = Parakeet_UnifiedEngine._enc_matmul_bf16_count
        q_label = self.quant.upper() if self.quant else "QUANT"
        print(f"  Encoder matmul dispatch: {q_n} {q_label}, {bf16_n} BF16  (expected: {384 if self.quant else 0} quantized)")
        # Copy final encoder output
        self.accelerator_memory_to_sram(self.INPUT_DRAM, URAM_A_BASE, L_pad * D)
        self.sram_to_accelerator_memory(URAM_A_BASE, self.ENC_OUT_DRAM, L_pad * D)

        self.stop_capture()
        self.generate_instruction_halt()
        program_addr = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(program_addr)
        program_bytes = self.get_capture_instruction_size_bytes()
        self.allocate_program_dram(program_bytes)
        return program_addr, program_bytes
    def compile_encoder_profile(self, L_pad, toeplitz_addrs, bn_tiled):
        """Compile each encoder sub-block as a separate program for profiling.
        Returns list of (layer, block_name, dims_str, prog_addr, flops) tuples."""
        D, FF, H_heads, dk = self.d_model, self.ff_dim, self.num_heads, self.head_dim
        bpe = self.bytes_per_element
        P_pad = pad_to_multiple(2 * L_pad - 1, self.block_size)
        FF_HALF = FF // 2
        programs = []

        def _compile(emit_fn):
            self.clear_capture_buffer()
            self.start_capture()
            self.generate_instruction_flag_clear()
            flops = emit_fn()
            self.stop_capture()
            self.generate_instruction_halt()
            addr = self.get_program_dram_addr()
            self.write_captured_instructions_to_dram(addr)
            self.allocate_program_dram(self.get_capture_instruction_size_bytes())
            return addr, flops

        for layer_idx in range(self.num_layers):
            la = self.layer_addrs[layer_idx]

            # --- FF1 ---
            def emit_ff1(la=la):
                flops = 0
                self.layer_norm_core_dram(M=L_pad, N=D, A_DRAM_ADDR=self.INPUT_DRAM, OUTPUT_DRAM_ADDR=self.LN_OUT_DRAM, GAMMA_DRAM_ADDR=la["LN_FF1_WEIGHT"], BETA_DRAM_ADDR=la["LN_FF1_BIAS"])
                self.matmat_mul_core(M=L_pad, K=D, N=FF_HALF, A_DRAM_ADDR=self.LN_OUT_DRAM, B_DRAM_ADDR=la["FF1_W1_LO"], OUTPUT_DRAM_ADDR=self.FF_MID_DRAM, silu_enable=True)
                self.matmat_mul_core(M=L_pad, K=D, N=FF_HALF, A_DRAM_ADDR=self.LN_OUT_DRAM, B_DRAM_ADDR=la["FF1_W1_HI"], OUTPUT_DRAM_ADDR=self.FF_MID_DRAM + L_pad * FF_HALF * bpe, silu_enable=True)
                flops += 2 * L_pad * D * FF
                self.matmat_mul_core(M=L_pad, K=FF_HALF, N=D, A_DRAM_ADDR=self.FF_MID_DRAM, B_DRAM_ADDR=la["FF1_W2_LO"], OUTPUT_DRAM_ADDR=self.FF_OUT_DRAM)
                self.matmat_mul_core(M=L_pad, K=FF_HALF, N=D, A_DRAM_ADDR=self.FF_MID_DRAM + L_pad * FF_HALF * bpe, B_DRAM_ADDR=la["FF1_W2_HI"], OUTPUT_DRAM_ADDR=self.LN_OUT_DRAM)
                flops += 2 * L_pad * FF * D
                self.accelerator_memory_to_sram(self.FF_OUT_DRAM, URAM_A_BASE, L_pad * D)
                self.accelerator_memory_to_sram(self.LN_OUT_DRAM, URAM_B_BASE, L_pad * D)
                self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * D)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.FF_OUT_DRAM, L_pad * D)
                half_step_residual_core_dram(self, M=L_pad, N=D, RESIDUAL_DRAM_ADDR=self.INPUT_DRAM, FF_DRAM_ADDR=self.FF_OUT_DRAM, OUTPUT_DRAM_ADDR=self.INPUT_DRAM)
                flops += 2 * L_pad * D
                return flops
            addr, flops = _compile(emit_ff1)
            dims = f"LN({L_pad},{D}) matmul({L_pad},{D},{FF_HALF})x2+SiLU matmul({L_pad},{FF_HALF},{D})x2"
            programs.append((layer_idx, "FF1", dims, addr, flops))

            # --- SelfAttn ---
            def emit_self_attn(la=la):
                flops = 0
                self.layer_norm_core_dram(M=L_pad, N=D, A_DRAM_ADDR=self.INPUT_DRAM, OUTPUT_DRAM_ADDR=self.LN_OUT_DRAM, GAMMA_DRAM_ADDR=la["LN_ATTN_WEIGHT"], BETA_DRAM_ADDR=la["LN_ATTN_BIAS"])
                self.matmat_mul_core(M=L_pad, K=D, N=D, A_DRAM_ADDR=self.LN_OUT_DRAM, B_DRAM_ADDR=la["ATTN_Q_W"], OUTPUT_DRAM_ADDR=self.Q_DRAM)
                self.matmat_mul_core(M=L_pad, K=D, N=D, A_DRAM_ADDR=self.LN_OUT_DRAM, B_DRAM_ADDR=la["ATTN_K_W"], OUTPUT_DRAM_ADDR=self.K_DRAM)
                self.matmat_mul_core(M=L_pad, K=D, N=D, A_DRAM_ADDR=self.LN_OUT_DRAM, B_DRAM_ADDR=la["ATTN_V_W"], OUTPUT_DRAM_ADDR=self.V_DRAM)
                flops += 3 * 2 * L_pad * D * D
                self.matmat_mul_core(M=P_pad, K=D, N=D, A_DRAM_ADDR=self.POS_EMB_DRAM, B_DRAM_ADDR=la["ATTN_POS_W"], OUTPUT_DRAM_ADDR=self.POS_PROJ_DRAM)
                flops += 2 * P_pad * D * D
                for h in range(H_heads):
                    h_off = h * dk * bpe
                    self.accelerator_memory_to_sram(self.Q_DRAM + h_off, URAM_A_BASE, L_pad * dk, stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                    self.accelerator_memory_to_sram(la["ATTN_BIAS_U_TILED"][h], URAM_B_BASE, L_pad * dk)
                    self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * dk)
                    self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_A_DRAM, L_pad * dk)
                    self.accelerator_memory_to_sram(self.K_DRAM + h_off, URAM_A_BASE, L_pad * dk, stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                    self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_B_DRAM, L_pad * dk)
                    self.matmat_mul_core(M=L_pad, K=dk, N=L_pad, A_DRAM_ADDR=self.CONV_A_DRAM, B_DRAM_ADDR=self.CONV_B_DRAM, OUTPUT_DRAM_ADDR=self.SCORE_DRAM)
                    flops += 2 * L_pad * dk * L_pad
                    self.accelerator_memory_to_sram(self.Q_DRAM + h_off, URAM_A_BASE, L_pad * dk, stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                    self.accelerator_memory_to_sram(la["ATTN_BIAS_V_TILED"][h], URAM_B_BASE, L_pad * dk)
                    self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * dk)
                    self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_A_DRAM, L_pad * dk)
                    self.accelerator_memory_to_sram(self.POS_PROJ_DRAM + h_off, URAM_A_BASE, P_pad * dk, stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                    self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_B_DRAM, P_pad * dk)
                    self.matmat_mul_core(M=L_pad, K=dk, N=P_pad, A_DRAM_ADDR=self.CONV_A_DRAM, B_DRAM_ADDR=self.CONV_B_DRAM, OUTPUT_DRAM_ADDR=self.CONV_OUT_DRAM)
                    flops += 2 * L_pad * dk * P_pad
                    rel_shift_core_dram(self, L=L_pad, INPUT_DRAM_ADDR=self.CONV_OUT_DRAM, OUTPUT_DRAM_ADDR=self.REL_SHIFT_DRAM, input_row_stride=P_pad)
                    score_elems = L_pad * L_pad
                    self.accelerator_memory_to_sram(self.SCORE_DRAM, URAM_A_BASE, score_elems)
                    self.accelerator_memory_to_sram(self.REL_SHIFT_DRAM, URAM_B_BASE, score_elems)
                    self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, score_elems)
                    self.broadcast_mul(scalar=1.0 / math.sqrt(dk), sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=score_elems)
                    self.sram_to_accelerator_memory(URAM_A_BASE, self.SCORE_DRAM, score_elems)
                    self.matmat_mul_core(M=L_pad, K=L_pad, N=L_pad, A_DRAM_ADDR=self.SCORE_DRAM, B_DRAM_ADDR=self.IDENTITY_LPAD_DRAM, OUTPUT_DRAM_ADDR=self.SCORE_DRAM, softmax_enable=True, C_DRAM_ADDR=self.ATTN_MASK_DRAM, bias_mode="full_matrix")
                    flops += 2 * L_pad * L_pad * L_pad  # softmax identity matmul
                    self.accelerator_memory_to_sram(self.V_DRAM + h_off, URAM_A_BASE, L_pad * dk, stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                    self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_B_DRAM, L_pad * dk)
                    chunked_transpose_core_dram(self, M=L_pad, N=dk, input_dram_addr=self.CONV_B_DRAM, output_dram_addr=self.ATTN_VT_DRAM, identity_dram_addr=self.w["IDENTITY_64"], temp_dram_addr=self.PERMUTE_TEMP_DRAM)
                    flops += (dk // UE_VECTOR_SIZE) * dk * 2 * UE_VECTOR_SIZE * pad_to_multiple(L_pad, UE_VECTOR_SIZE)  # V transpose
                    self.matmat_mul_core(M=L_pad, K=L_pad, N=dk, A_DRAM_ADDR=self.SCORE_DRAM, B_DRAM_ADDR=self.ATTN_VT_DRAM, OUTPUT_DRAM_ADDR=self.CONV_A_DRAM)
                    flops += 2 * L_pad * L_pad * dk
                    self.accelerator_memory_to_sram(self.CONV_A_DRAM, URAM_A_BASE, L_pad * dk)
                    self.sram_to_accelerator_memory(URAM_A_BASE, self.ATTN_OUT_DRAM + h_off, L_pad * dk, stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                self.matmat_mul_core(M=L_pad, K=D, N=D, A_DRAM_ADDR=self.ATTN_OUT_DRAM, B_DRAM_ADDR=la["ATTN_OUT_W"], OUTPUT_DRAM_ADDR=self.FF_OUT_DRAM)
                flops += 2 * L_pad * D * D
                self.accelerator_memory_to_sram(self.INPUT_DRAM, URAM_A_BASE, L_pad * D)
                self.accelerator_memory_to_sram(self.FF_OUT_DRAM, URAM_B_BASE, L_pad * D)
                self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * D)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.INPUT_DRAM, L_pad * D)
                flops += L_pad * D
                return flops
            addr, flops = _compile(emit_self_attn)
            dims = f"QKV({L_pad},{D},{D})x3 pos({P_pad},{D},{D}) heads={H_heads}x[({L_pad},{dk},{L_pad}) softmax attn] out({L_pad},{D},{D})"
            programs.append((layer_idx, "SelfAttn", dims, addr, flops))

            # --- ConvModule ---
            def emit_conv_module(la=la, layer_idx=layer_idx):
                flops = 0
                self.layer_norm_core_dram(M=L_pad, N=D, A_DRAM_ADDR=self.INPUT_DRAM, OUTPUT_DRAM_ADDR=self.LN_OUT_DRAM, GAMMA_DRAM_ADDR=la["LN_CONV_WEIGHT"], BETA_DRAM_ADDR=la["LN_CONV_BIAS"])
                self.matmat_mul_core(M=L_pad, K=D, N=D, A_DRAM_ADDR=self.LN_OUT_DRAM, B_DRAM_ADDR=la["CONV_PW1A_W"], OUTPUT_DRAM_ADDR=self.CONV_A_DRAM)
                self.matmat_mul_core(M=L_pad, K=D, N=D, A_DRAM_ADDR=self.LN_OUT_DRAM, B_DRAM_ADDR=la["CONV_PW1B_W"], OUTPUT_DRAM_ADDR=self.CONV_B_DRAM, sigmoid_enable=True)
                flops += 2 * 2 * L_pad * D * D
                # GLU: a * sigmoid(b) — sigmoid fused into PW1b above
                self.accelerator_memory_to_sram(self.CONV_A_DRAM, URAM_A_BASE, L_pad * D)
                self.accelerator_memory_to_sram(self.CONV_B_DRAM, URAM_B_BASE, L_pad * D)
                self.eltwise_mul_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * D)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_A_DRAM, L_pad * D)
                chunked_transpose_core_dram(self, M=L_pad, N=D, input_dram_addr=self.CONV_A_DRAM, output_dram_addr=self.CONV_T_DRAM, identity_dram_addr=self.w["IDENTITY_64"], temp_dram_addr=self.PERMUTE_TEMP_DRAM)
                flops += (D // UE_VECTOR_SIZE) * D * 2 * UE_VECTOR_SIZE * pad_to_multiple(L_pad, UE_VECTOR_SIZE)  # transpose dot-products
                t_addr = toeplitz_addrs[layer_idx]
                for ch in range(D):
                    self.matmat_mul_core(M=1, K=L_pad, N=L_pad, A_DRAM_ADDR=self.CONV_T_DRAM + ch * L_pad * bpe, B_DRAM_ADDR=t_addr + ch * L_pad * L_pad * bpe, OUTPUT_DRAM_ADDR=self.CONV_DW_DRAM + ch * L_pad * bpe)
                flops += D * 2 * 1 * L_pad * L_pad  # actual HW compute (full Toeplitz matmul)
                batch_norm_core_dram(self, C=D, L=L_pad, A_DRAM_ADDR=self.CONV_DW_DRAM, OUTPUT_DRAM_ADDR=self.CONV_DW_DRAM, SCALE_DRAM_ADDR=la["CONV_BN_SCALE"], SHIFT_DRAM_ADDR=la["CONV_BN_SHIFT"], tiled_scale_addr=bn_tiled[layer_idx][0], tiled_shift_addr=bn_tiled[layer_idx][1])
                silu_core_dram(self, M=D, N=L_pad, A_DRAM_ADDR=self.CONV_DW_DRAM, OUTPUT_DRAM_ADDR=self.CONV_OUT_DRAM, IDENTITY_DRAM_ADDR=self.IDENTITY_LPAD_DRAM)
                flops += 2 * D * L_pad * L_pad  # SiLU identity matmul
                chunked_transpose_core_dram(self, M=D, N=L_pad, input_dram_addr=self.CONV_OUT_DRAM, output_dram_addr=self.CONV_T_DRAM, identity_dram_addr=self.w["IDENTITY_64"], temp_dram_addr=self.PERMUTE_TEMP_DRAM)
                flops += (L_pad // UE_VECTOR_SIZE) * L_pad * 2 * UE_VECTOR_SIZE * pad_to_multiple(D, UE_VECTOR_SIZE)  # transpose dot-products
                self.matmat_mul_core(M=L_pad, K=D, N=D, A_DRAM_ADDR=self.CONV_T_DRAM, B_DRAM_ADDR=la["CONV_PW2_W"], OUTPUT_DRAM_ADDR=self.CONV_OUT_DRAM)
                flops += 2 * L_pad * D * D
                self.accelerator_memory_to_sram(self.INPUT_DRAM, URAM_A_BASE, L_pad * D)
                self.accelerator_memory_to_sram(self.CONV_OUT_DRAM, URAM_B_BASE, L_pad * D)
                self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * D)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.INPUT_DRAM, L_pad * D)
                flops += L_pad * D
                return flops
            addr, flops = _compile(emit_conv_module)
            dims = f"PW1({L_pad},{D},{D})x2+GLU DW({D}x(1,{L_pad},{L_pad})) BN+SiLU PW2({L_pad},{D},{D})"
            programs.append((layer_idx, "ConvModule", dims, addr, flops))

            # --- FF2 ---
            def emit_ff2(la=la):
                flops = 0
                self.layer_norm_core_dram(M=L_pad, N=D, A_DRAM_ADDR=self.INPUT_DRAM, OUTPUT_DRAM_ADDR=self.LN_OUT_DRAM, GAMMA_DRAM_ADDR=la["LN_FF2_WEIGHT"], BETA_DRAM_ADDR=la["LN_FF2_BIAS"])
                self.matmat_mul_core(M=L_pad, K=D, N=FF_HALF, A_DRAM_ADDR=self.LN_OUT_DRAM, B_DRAM_ADDR=la["FF2_W1_LO"], OUTPUT_DRAM_ADDR=self.FF_MID_DRAM, silu_enable=True)
                self.matmat_mul_core(M=L_pad, K=D, N=FF_HALF, A_DRAM_ADDR=self.LN_OUT_DRAM, B_DRAM_ADDR=la["FF2_W1_HI"], OUTPUT_DRAM_ADDR=self.FF_MID_DRAM + L_pad * FF_HALF * bpe, silu_enable=True)
                flops += 2 * L_pad * D * FF
                self.matmat_mul_core(M=L_pad, K=FF_HALF, N=D, A_DRAM_ADDR=self.FF_MID_DRAM, B_DRAM_ADDR=la["FF2_W2_LO"], OUTPUT_DRAM_ADDR=self.FF_OUT_DRAM)
                self.matmat_mul_core(M=L_pad, K=FF_HALF, N=D, A_DRAM_ADDR=self.FF_MID_DRAM + L_pad * FF_HALF * bpe, B_DRAM_ADDR=la["FF2_W2_HI"], OUTPUT_DRAM_ADDR=self.LN_OUT_DRAM)
                flops += 2 * L_pad * FF * D
                self.accelerator_memory_to_sram(self.FF_OUT_DRAM, URAM_A_BASE, L_pad * D)
                self.accelerator_memory_to_sram(self.LN_OUT_DRAM, URAM_B_BASE, L_pad * D)
                self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * D)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.FF_OUT_DRAM, L_pad * D)
                half_step_residual_core_dram(self, M=L_pad, N=D, RESIDUAL_DRAM_ADDR=self.INPUT_DRAM, FF_DRAM_ADDR=self.FF_OUT_DRAM, OUTPUT_DRAM_ADDR=self.INPUT_DRAM)
                flops += 2 * L_pad * D
                return flops
            addr, flops = _compile(emit_ff2)
            dims = f"LN({L_pad},{D}) matmul({L_pad},{D},{FF_HALF})x2+SiLU matmul({L_pad},{FF_HALF},{D})x2"
            programs.append((layer_idx, "FF2", dims, addr, flops))

            # --- FinalLN ---
            def emit_final_ln(la=la):
                self.layer_norm_core_dram(M=L_pad, N=D, A_DRAM_ADDR=self.INPUT_DRAM, OUTPUT_DRAM_ADDR=self.INPUT_DRAM, GAMMA_DRAM_ADDR=la["LN_OUT_WEIGHT"], BETA_DRAM_ADDR=la["LN_OUT_BIAS"])
                return 2 * L_pad * D
            addr, flops = _compile(emit_final_ln)
            dims = f"LN({L_pad},{D})"
            programs.append((layer_idx, "FinalLN", dims, addr, flops))

            print(f"\r    Layer {layer_idx:2d}/{self.num_layers - 1} compiled (5 sub-blocks)", end="", flush=True)

        print()
        return programs
    def compile_attn_decomposed(self, layer_idx, L_pad, toeplitz_addrs, bn_tiled):
        """Compile SelfAttn for one layer as 3 separate programs for debugging.
        Returns list of (name, prog_addr) tuples:
          1. 'Attn:LN+Proj'  — LayerNorm + QKV + Pos projections
          2. 'Attn:Heads'    — 8-head attention loop (scores, softmax, attn@V)
          3. 'Attn:Out+Res'  — Output projection + residual add
        """
        D, H_heads, dk = self.d_model, self.num_heads, self.head_dim
        bpe = self.bytes_per_element
        P_pad = pad_to_multiple(2 * L_pad - 1, self.block_size)
        la = self.layer_addrs[layer_idx]
        programs = []

        def _compile(emit_fn):
            self.clear_capture_buffer()
            self.start_capture()
            self.generate_instruction_flag_clear()
            emit_fn()
            self.stop_capture()
            self.generate_instruction_halt()
            addr = self.get_program_dram_addr()
            self.write_captured_instructions_to_dram(addr)
            self.allocate_program_dram(self.get_capture_instruction_size_bytes())
            return addr

        # --- Part 1: LN + QKV + Pos projections ---
        def emit_proj():
            self.layer_norm_core_dram(M=L_pad, N=D, A_DRAM_ADDR=self.INPUT_DRAM, OUTPUT_DRAM_ADDR=self.LN_OUT_DRAM, GAMMA_DRAM_ADDR=la["LN_ATTN_WEIGHT"], BETA_DRAM_ADDR=la["LN_ATTN_BIAS"])
            self.matmat_mul_core(M=L_pad, K=D, N=D, A_DRAM_ADDR=self.LN_OUT_DRAM, B_DRAM_ADDR=la["ATTN_Q_W"], OUTPUT_DRAM_ADDR=self.Q_DRAM)
            self.matmat_mul_core(M=L_pad, K=D, N=D, A_DRAM_ADDR=self.LN_OUT_DRAM, B_DRAM_ADDR=la["ATTN_K_W"], OUTPUT_DRAM_ADDR=self.K_DRAM)
            self.matmat_mul_core(M=L_pad, K=D, N=D, A_DRAM_ADDR=self.LN_OUT_DRAM, B_DRAM_ADDR=la["ATTN_V_W"], OUTPUT_DRAM_ADDR=self.V_DRAM)
            self.matmat_mul_core(M=P_pad, K=D, N=D, A_DRAM_ADDR=self.POS_EMB_DRAM, B_DRAM_ADDR=la["ATTN_POS_W"], OUTPUT_DRAM_ADDR=self.POS_PROJ_DRAM)
        programs.append(("Attn:LN+Proj", _compile(emit_proj)))

        # --- Part 2: Per-head attention loop ---
        def emit_heads():
            for h in range(H_heads):
                h_off = h * dk * bpe
                self.accelerator_memory_to_sram(self.Q_DRAM + h_off, URAM_A_BASE, L_pad * dk, stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                self.accelerator_memory_to_sram(la["ATTN_BIAS_U_TILED"][h], URAM_B_BASE, L_pad * dk)
                self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * dk)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_A_DRAM, L_pad * dk)
                self.accelerator_memory_to_sram(self.K_DRAM + h_off, URAM_A_BASE, L_pad * dk, stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_B_DRAM, L_pad * dk)
                self.matmat_mul_core(M=L_pad, K=dk, N=L_pad, A_DRAM_ADDR=self.CONV_A_DRAM, B_DRAM_ADDR=self.CONV_B_DRAM, OUTPUT_DRAM_ADDR=self.SCORE_DRAM)
                self.accelerator_memory_to_sram(self.Q_DRAM + h_off, URAM_A_BASE, L_pad * dk, stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                self.accelerator_memory_to_sram(la["ATTN_BIAS_V_TILED"][h], URAM_B_BASE, L_pad * dk)
                self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * dk)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_A_DRAM, L_pad * dk)
                self.accelerator_memory_to_sram(self.POS_PROJ_DRAM + h_off, URAM_A_BASE, P_pad * dk, stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_B_DRAM, P_pad * dk)
                self.matmat_mul_core(M=L_pad, K=dk, N=P_pad, A_DRAM_ADDR=self.CONV_A_DRAM, B_DRAM_ADDR=self.CONV_B_DRAM, OUTPUT_DRAM_ADDR=self.CONV_OUT_DRAM)
                rel_shift_core_dram(self, L=L_pad, INPUT_DRAM_ADDR=self.CONV_OUT_DRAM, OUTPUT_DRAM_ADDR=self.REL_SHIFT_DRAM, input_row_stride=P_pad)
                score_elems = L_pad * L_pad
                self.accelerator_memory_to_sram(self.SCORE_DRAM, URAM_A_BASE, score_elems)
                self.accelerator_memory_to_sram(self.REL_SHIFT_DRAM, URAM_B_BASE, score_elems)
                self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, score_elems)
                self.broadcast_mul(scalar=1.0 / math.sqrt(dk), sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=score_elems)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.SCORE_DRAM, score_elems)
                self.matmat_mul_core(M=L_pad, K=L_pad, N=L_pad, A_DRAM_ADDR=self.SCORE_DRAM, B_DRAM_ADDR=self.IDENTITY_LPAD_DRAM, OUTPUT_DRAM_ADDR=self.SCORE_DRAM, softmax_enable=True, C_DRAM_ADDR=self.ATTN_MASK_DRAM, bias_mode="full_matrix")
                self.accelerator_memory_to_sram(self.V_DRAM + h_off, URAM_A_BASE, L_pad * dk, stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_B_DRAM, L_pad * dk)
                chunked_transpose_core_dram(self, M=L_pad, N=dk, input_dram_addr=self.CONV_B_DRAM, output_dram_addr=self.ATTN_VT_DRAM, identity_dram_addr=self.w["IDENTITY_64"], temp_dram_addr=self.PERMUTE_TEMP_DRAM)
                self.matmat_mul_core(M=L_pad, K=L_pad, N=dk, A_DRAM_ADDR=self.SCORE_DRAM, B_DRAM_ADDR=self.ATTN_VT_DRAM, OUTPUT_DRAM_ADDR=self.CONV_A_DRAM)
                self.accelerator_memory_to_sram(self.CONV_A_DRAM, URAM_A_BASE, L_pad * dk)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.ATTN_OUT_DRAM + h_off, L_pad * dk, stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
        programs.append(("Attn:Heads", _compile(emit_heads)))

        # --- Part 3: Output projection + residual ---
        def emit_out():
            self.matmat_mul_core(M=L_pad, K=D, N=D, A_DRAM_ADDR=self.ATTN_OUT_DRAM, B_DRAM_ADDR=la["ATTN_OUT_W"], OUTPUT_DRAM_ADDR=self.FF_OUT_DRAM)
            self.accelerator_memory_to_sram(self.INPUT_DRAM, URAM_A_BASE, L_pad * D)
            self.accelerator_memory_to_sram(self.FF_OUT_DRAM, URAM_B_BASE, L_pad * D)
            self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * D)
            self.sram_to_accelerator_memory(URAM_A_BASE, self.INPUT_DRAM, L_pad * D)
        programs.append(("Attn:Out+Res", _compile(emit_out)))

        return programs
    def run_encoder_profile(self, programs):
        """Execute each sub-block program and print profiling results.
        Args: programs from compile_encoder_profile."""
        import time as _time
        import user_dma_core as _udc

        block_names = ["FF1", "SelfAttn", "ConvModule", "FF2", "FinalLN"]

        print(f"\n{'='*120}")
        print(f"  ENCODER PROFILING: {self.num_layers} layers x 5 sub-blocks")
        print(f"{'='*120}")
        print(f"{'Layer':>5}  {'Block':<12}  {'Instrs':>8}  {'HW(us)':>12}  {'Wall(ms)':>10}  {'GFLOP':>8}  {'GFLOP/s(hw)':>11}  {'GFLOP/s(w)':>10}  Dimensions")
        print("-" * 120)

        results = []
        total_t0 = _time.perf_counter()

        for layer_idx, block_name, dims, prog_addr, flops in programs:
            t0 = _time.perf_counter()

            # Phase 1: Dispatch (write instruction address register)
            t_dispatch_start = _time.perf_counter()
            self.start_execute_from_dram(prog_addr)
            t_dispatch_end = _time.perf_counter()

            # Phase 2: Poll wait (time spent waiting for HW to finish)
            poll_count = 0
            t_poll_start = _time.perf_counter()
            while self.is_queue_busy():
                _time.sleep(0.01)
                poll_count += 1
            t_poll_end = _time.perf_counter()

            # Phase 3: Register reads (latency + instruction count)
            t_read_start = _time.perf_counter()
            hw_us = self.report_latency_in_us()
            instr_count = self.read_reg32(_udc.UE_INSTRUCTION_CTL_ADDR)
            t_read_end = _time.perf_counter()

            wall_s = _time.perf_counter() - t0

            dispatch_ms = (t_dispatch_end - t_dispatch_start) * 1000
            poll_ms = (t_poll_end - t_poll_start) * 1000
            read_ms = (t_read_end - t_read_start) * 1000
            gap_ms = wall_s * 1000 - dispatch_ms - poll_ms - read_ms

            gflops = flops / 1e9
            wall_rate = gflops / wall_s if wall_s > 0 else 0

            hw_rate = gflops / (hw_us * 1e-6) if hw_us > 0 else 0

            results.append({
                "layer": layer_idx, "block": block_name,
                "instr_count": instr_count, "hw_us": hw_us,
                "wall_s": wall_s, "flops": flops,
                "gflops": gflops, "hw_rate": hw_rate, "wall_rate": wall_rate,
                "dispatch_ms": dispatch_ms, "poll_ms": poll_ms,
                "poll_count": poll_count, "read_ms": read_ms,
                "gap_ms": gap_ms,
            })
            print(f"{layer_idx:>5}  {block_name:<12}  {instr_count:>8}  {hw_us:>12.2f}  {wall_s*1000:>10.2f}  {gflops:>8.3f}  {hw_rate:>9.2f}  {wall_rate:>8.2f}  {dims}")

        total_wall = _time.perf_counter() - total_t0
        total_hw_us = sum(r["hw_us"] for r in results)
        total_gflops = sum(r["gflops"] for r in results)
        total_instrs = sum(r["instr_count"] for r in results)

        print(f"\n{'='*120}")
        print(f"  SUMMARY")
        print(f"{'='*120}")
        print(f"  Total HW latency:        {total_hw_us:>12.2f} us  ({total_hw_us/1e6:.3f} s)")
        print(f"  Total wall-clock:        {total_wall:>12.2f} s")
        print(f"  Total instructions:      {total_instrs:>12}")
        print(f"  Total GFLOP:             {total_gflops:>12.3f}")
        if total_hw_us > 0:
            print(f"  GFLOP/s (HW counter):    {total_gflops / (total_hw_us * 1e-6) :>12.2f}")
        print(f"  GFLOP/s (wall-clock):    {total_gflops / total_wall :>12.2f}")

        # Host-side overhead decomposition
        total_dispatch = sum(r["dispatch_ms"] for r in results)
        total_poll = sum(r["poll_ms"] for r in results)
        total_read = sum(r["read_ms"] for r in results)
        total_gap = sum(r["gap_ms"] for r in results)
        total_polls = sum(r["poll_count"] for r in results)
        print(f"\n  HOST-SIDE TIME DECOMPOSITION:")
        print(f"  {'-'*70}")
        print(f"  Dispatch (write_reg32):  {total_dispatch:>10.1f} ms  ({100*total_dispatch/(total_wall*1000):.1f}%)")
        print(f"  Poll wait (busy loop):   {total_poll:>10.1f} ms  ({100*total_poll/(total_wall*1000):.1f}%)  [{total_polls} polls total]")
        print(f"  Register reads:          {total_read:>10.1f} ms  ({100*total_read/(total_wall*1000):.1f}%)")
        print(f"  Unaccounted (perf_ctr):  {total_gap:>10.1f} ms  ({100*total_gap/(total_wall*1000):.1f}%)")
        print(f"  {'-'*70}")
        print(f"  Total wall:              {total_wall*1000:>10.1f} ms")
        print(f"  HW counter total:        {total_hw_us/1000:>10.1f} ms")
        print(f"  Wall minus HW:           {total_wall*1000 - total_hw_us/1000:>10.1f} ms  (pure overhead)")
        poll_overshoot = total_poll - total_hw_us / 1000
        print(f"  Poll time minus HW:      {poll_overshoot:>10.1f} ms  (poll overshoot + host reg access during poll)")

        print(f"\n  By sub-block type:")
        print(f"  {'Block':<12}  {'Total HW(us)':>14}  {'Total Wall(s)':>13}  {'% HW':>7}  {'% Wall':>7}  "
              f"{'Disp(ms)':>9}  {'Poll(ms)':>9}  {'Read(ms)':>9}  {'Total GFLOP':>12}  {'GFLOP/s(hw)':>11}  {'GFLOP/s(wall)':>13}")
        print(f"  {'-'*130}")
        for bname in block_names:
            br = [r for r in results if r["block"] == bname]
            bk_hw = sum(r["hw_us"] for r in br)
            bk_wall = sum(r["wall_s"] for r in br)
            bk_gflops = sum(r["gflops"] for r in br)
            bk_hw_pct = 100 * bk_hw / total_hw_us if total_hw_us > 0 else 0
            bk_wall_pct = 100 * bk_wall / total_wall if total_wall > 0 else 0
            bk_hw_rate = bk_gflops / (bk_hw * 1e-6) if bk_hw > 0 else 0
            bk_wall_rate = bk_gflops / bk_wall if bk_wall > 0 else 0
            bk_disp = sum(r["dispatch_ms"] for r in br)
            bk_poll = sum(r["poll_ms"] for r in br)
            bk_read = sum(r["read_ms"] for r in br)
            print(f"  {bname:<12}  {bk_hw:>14.2f}  {bk_wall:>13.3f}  {bk_hw_pct:>6.1f}%  {bk_wall_pct:>6.1f}%  "
                  f"{bk_disp:>9.1f}  {bk_poll:>9.1f}  {bk_read:>9.1f}  {bk_gflops:>12.3f}  {bk_hw_rate:>11.2f}  {bk_wall_rate:>13.2f}")

        print(f"\n  Per-layer total:")
        print(f"  {'Layer':>7}  {'HW(us)':>14}  {'Wall(ms)':>12}  {'Disp(ms)':>9}  {'Poll(ms)':>9}  {'Read(ms)':>9}")
        print(f"  {'-'*68}")
        for li in range(self.num_layers):
            lr = [r for r in results if r["layer"] == li]
            lhw = sum(r["hw_us"] for r in lr)
            lwall = sum(r["wall_s"] for r in lr) * 1000
            ldisp = sum(r["dispatch_ms"] for r in lr)
            lpoll = sum(r["poll_ms"] for r in lr)
            lread = sum(r["read_ms"] for r in lr)
            print(f"  {li:>7}  {lhw:>14.2f}  {lwall:>12.2f}  {ldisp:>9.2f}  {lpoll:>9.2f}  {lread:>9.2f}")
        print(f"\n{'='*120}")

        return results
    def compile_decoder(self):
        """Compile predictor LSTM + split joint. Returns (pred_prog, tok_prog, dur_prog, flops)."""
        H = self.pred_hidden
        D = self.d_model
        bpe = self.bytes_per_element
        N_tok_pad = pad_to_multiple(self.vocab_size, self.block_size)
        N_dur_pad = pad_to_multiple(len(self.tdt_durations), self.block_size)
        total_flops = 0
        # --- Predictor program ---
        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()
        for i in range(2):
            h_addr = self.PRED_H0_DRAM if i == 0 else self.PRED_H1_DRAM
            c_addr = self.PRED_C0_DRAM if i == 0 else self.PRED_C1_DRAM
            x_addr = self.PRED_EMB_DRAM if i == 0 else self.PRED_H0_DRAM
            self.matmat_mul_core(M=1, K=H, N=4*H, A_DRAM_ADDR=x_addr, B_DRAM_ADDR=self.w[f"LSTM_WIH{i}"], OUTPUT_DRAM_ADDR=self.PRED_GATES_DRAM, C_DRAM_ADDR=self.w[f"LSTM_BIH{i}"])
            total_flops += 2 * H * 4 * H
            self.matmat_mul_core(M=1, K=H, N=4*H, A_DRAM_ADDR=h_addr, B_DRAM_ADDR=self.w[f"LSTM_WHH{i}"], OUTPUT_DRAM_ADDR=self.PRED_GATES2_DRAM, C_DRAM_ADDR=self.w[f"LSTM_BHH{i}"])
            total_flops += 2 * H * 4 * H
            self.accelerator_memory_to_sram(self.PRED_GATES_DRAM, URAM_A_BASE, 4*H)
            self.accelerator_memory_to_sram(self.PRED_GATES2_DRAM, URAM_B_BASE, 4*H)
            self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, 4*H)
            self.sram_to_accelerator_memory(URAM_A_BASE, self.PRED_GATES_DRAM, 4*H)
            # i sigmoid
            self.matmat_mul_core(M=1, K=H, N=H, A_DRAM_ADDR=self.PRED_GATES_DRAM, B_DRAM_ADDR=self.w["IDENTITY_640"], OUTPUT_DRAM_ADDR=self.PRED_GATES_DRAM, sigmoid_enable=True)
            # f sigmoid
            self.matmat_mul_core(M=1, K=H, N=H, A_DRAM_ADDR=self.PRED_GATES_DRAM + H * bpe, B_DRAM_ADDR=self.w["IDENTITY_640"], OUTPUT_DRAM_ADDR=self.PRED_GATES_DRAM + H * bpe, sigmoid_enable=True)
            # g tanh
            tanh_core_dram(self, M=1, N=H, A_DRAM_ADDR=self.PRED_GATES_DRAM + 2 * H * bpe, OUTPUT_DRAM_ADDR=self.PRED_GATES_DRAM + 2 * H * bpe, IDENTITY_DRAM_ADDR=self.w["IDENTITY_640"])
            # o sigmoid
            self.matmat_mul_core(M=1, K=H, N=H, A_DRAM_ADDR=self.PRED_GATES_DRAM + 3 * H * bpe, B_DRAM_ADDR=self.w["IDENTITY_640"], OUTPUT_DRAM_ADDR=self.PRED_GATES_DRAM + 3 * H * bpe, sigmoid_enable=True)
            # c_new = f*c + i*g
            self.accelerator_memory_to_sram(self.PRED_GATES_DRAM + H * bpe, URAM_A_BASE, H)
            self.accelerator_memory_to_sram(c_addr, URAM_B_BASE, H)
            self.eltwise_mul_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, H)
            self.sram_to_accelerator_memory(URAM_A_BASE, c_addr, H)
            self.accelerator_memory_to_sram(self.PRED_GATES_DRAM, URAM_A_BASE, H)
            self.accelerator_memory_to_sram(self.PRED_GATES_DRAM + 2 * H * bpe, URAM_B_BASE, H)
            self.eltwise_mul_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, H)
            self.accelerator_memory_to_sram(c_addr, URAM_B_BASE, H)
            self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, H)
            self.sram_to_accelerator_memory(URAM_A_BASE, c_addr, H)
            # h_new = o * tanh(c_new)
            tanh_core_dram(self, M=1, N=H, A_DRAM_ADDR=c_addr, OUTPUT_DRAM_ADDR=self.PRED_OUT_DRAM, IDENTITY_DRAM_ADDR=self.w["IDENTITY_640"])
            self.accelerator_memory_to_sram(self.PRED_GATES_DRAM + 3 * H * bpe, URAM_A_BASE, H)
            self.accelerator_memory_to_sram(self.PRED_OUT_DRAM, URAM_B_BASE, H)
            self.eltwise_mul_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, H)
            self.sram_to_accelerator_memory(URAM_A_BASE, h_addr, H)
        self.accelerator_memory_to_sram(self.PRED_H1_DRAM, URAM_A_BASE, H)
        self.sram_to_accelerator_memory(URAM_A_BASE, self.PRED_OUT_DRAM, H)
        self.stop_capture()
        self.generate_instruction_halt()
        pred_prog = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(pred_prog)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        # --- Joint token program (shared projections + token matmul → argmax) ---
        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()
        self.matmat_mul_core(M=1, K=D, N=H, A_DRAM_ADDR=self.JOINT_ENC_DRAM, B_DRAM_ADDR=self.w["JOINT_ENC_W"], OUTPUT_DRAM_ADDR=self.JOINT_ENC_DRAM, C_DRAM_ADDR=self.w["JOINT_ENC_B"])
        total_flops += 2 * D * H
        self.matmat_mul_core(M=1, K=H, N=H, A_DRAM_ADDR=self.JOINT_PRED_DRAM, B_DRAM_ADDR=self.w["JOINT_PRED_W"], OUTPUT_DRAM_ADDR=self.JOINT_PRED_DRAM, C_DRAM_ADDR=self.w["JOINT_PRED_B"])
        total_flops += 2 * H * H
        self.accelerator_memory_to_sram(self.JOINT_ENC_DRAM, URAM_A_BASE, H)
        self.accelerator_memory_to_sram(self.JOINT_PRED_DRAM, URAM_B_BASE, H)
        self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, H)
        self.sram_to_accelerator_memory(URAM_A_BASE, self.JOINT_SUM_DRAM, H)
        self.matmat_mul_core(M=1, K=H, N=H, A_DRAM_ADDR=self.JOINT_SUM_DRAM, B_DRAM_ADDR=self.w["IDENTITY_640"], OUTPUT_DRAM_ADDR=self.JOINT_SUM_DRAM, relu_enable=True)
        self.matmat_mul_core(M=1, K=H, N=N_tok_pad, A_DRAM_ADDR=self.JOINT_SUM_DRAM, B_DRAM_ADDR=self.w["JOINT_OUT_TOK_W"], OUTPUT_DRAM_ADDR=self.JOINT_TOK_DRAM, C_DRAM_ADDR=self.w["JOINT_OUT_TOK_B"])
        total_flops += 2 * H * N_tok_pad
        self.stop_capture()
        self.generate_instruction_halt()
        tok_prog = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(tok_prog)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        # --- Joint duration program (duration matmul → argmax, reuses JOINT_SUM_DRAM) ---
        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()
        self.matmat_mul_core(M=1, K=H, N=N_dur_pad, A_DRAM_ADDR=self.JOINT_SUM_DRAM, B_DRAM_ADDR=self.w["JOINT_OUT_DUR_W"], OUTPUT_DRAM_ADDR=self.JOINT_DUR_DRAM, C_DRAM_ADDR=self.w["JOINT_OUT_DUR_B"])
        total_flops += 2 * H * N_dur_pad
        self.stop_capture()
        self.generate_instruction_halt()
        dur_prog = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(dur_prog)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())

        return pred_prog, tok_prog, dur_prog, total_flops
    def program_execute(self, program_addr, timeout=50.0, flops=None):
        """Execute compiled program. Returns (latency_us, gflops)."""
        self.start_execute_from_dram(program_addr)
        latency_us, gflops = 0, 0
        if timeout > 0:
            self.wait_queue(timeout)
            latency_us = self.report_latency_in_us()
            if flops:
                gflops = self.report_flop_rate_gflops(flops)
        return latency_us, gflops
    def get_arg_max_index1(self):
        """Read hardware argmax1 register."""
        return self.read_reg32(user_dma_core.UE_ARGMAX1_INDEX)
    def get_arg_max_index2(self):
        """Read hardware argmax2 register."""
        return self.read_reg32(user_dma_core.UE_ARGMAX2_INDEX)
    def make_rel_pos_emb(self, seq_len):
        """Generate relative positional encoding: (2*seq_len-1, D) bf16."""
        D = self.d_model
        max_len = 2 * seq_len
        pe = torch.zeros(max_len, D)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, D, 2, dtype=torch.float32) * -(math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe_pos = pe[:seq_len]
        pe_neg = torch.flip(pe[1:seq_len], [0])
        return torch.cat([pe_neg, pe_pos], dim=0).to(torch.bfloat16)
    # -----------------------------------------------------------------------
    # Execute: decoder
    # -----------------------------------------------------------------------
    def run_decode(self, enc_out_addr, L):
        """TDT greedy decode. All programs already compiled. Returns token list."""
        bpe = self.bytes_per_element
        H = self.pred_hidden
        D = self.d_model
        pred_prog = self.progs["pred"][0]
        tok_prog = self.progs["joint_tok"][0]
        dur_prog = self.progs["joint_dur"][0]
        # Pre-zero all decoder intermediate buffers
        zeros_h = torch.zeros(H, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.PRED_H0_DRAM, zeros_h)
        self.dma_to_accelerator_memory(self.PRED_C0_DRAM, zeros_h)
        self.dma_to_accelerator_memory(self.PRED_H1_DRAM, zeros_h)
        self.dma_to_accelerator_memory(self.PRED_C1_DRAM, zeros_h)
        self.dma_to_accelerator_memory(self.PRED_EMB_DRAM, zeros_h)
        self.dma_to_accelerator_memory(self.PRED_OUT_DRAM, zeros_h)
        self.dma_to_accelerator_memory(self.PRED_GATES_DRAM, torch.zeros(4 * H, dtype=torch.bfloat16))
        self.dma_to_accelerator_memory(self.PRED_GATES2_DRAM, torch.zeros(4 * H, dtype=torch.bfloat16))
        self.dma_to_accelerator_memory(self.JOINT_ENC_DRAM, torch.zeros(D, dtype=torch.bfloat16))
        self.dma_to_accelerator_memory(self.JOINT_PRED_DRAM, zeros_h)
        self.dma_to_accelerator_memory(self.JOINT_SUM_DRAM, zeros_h)
        N_tok_pad = pad_to_multiple(self.vocab_size, self.block_size)
        N_dur_pad = pad_to_multiple(len(self.tdt_durations), self.block_size)
        self.dma_to_accelerator_memory(self.JOINT_TOK_DRAM, torch.zeros(N_tok_pad, dtype=torch.bfloat16))
        self.dma_to_accelerator_memory(self.JOINT_DUR_DRAM, torch.zeros(N_dur_pad, dtype=torch.bfloat16))
        tokens = []
        t = 0
        last_token = self.blank_id
        total_steps = 0
        while t < L:
            symbols = 0
            while symbols < self.max_symbols_per_step:
                # Save LSTM state before predictor (restore on blank)
                saved_h0 = torch.zeros(H, dtype=torch.bfloat16)
                saved_c0 = torch.zeros(H, dtype=torch.bfloat16)
                saved_h1 = torch.zeros(H, dtype=torch.bfloat16)
                saved_c1 = torch.zeros(H, dtype=torch.bfloat16)
                self.dma_read(DMA_DEVICE_C2H, self.PRED_H0_DRAM, saved_h0, H * bpe)
                self.dma_read(DMA_DEVICE_C2H, self.PRED_C0_DRAM, saved_c0, H * bpe)
                self.dma_read(DMA_DEVICE_C2H, self.PRED_H1_DRAM, saved_h1, H * bpe)
                self.dma_read(DMA_DEVICE_C2H, self.PRED_C1_DRAM, saved_c1, H * bpe)
                # Embedding lookup via host DMA
                emb_src = self.w["EMBED"] + last_token * H * bpe
                emb_buf = torch.zeros(H, dtype=torch.bfloat16)
                self.dma_read(DMA_DEVICE_C2H, emb_src, emb_buf, H * bpe)
                self.dma_to_accelerator_memory(self.PRED_EMB_DRAM, emb_buf)
                # Predictor
                self.program_execute(pred_prog)
                # Copy enc_out[t] and pred_out to joint inputs via HOST DMA
                enc_t_buf = torch.zeros(D, dtype=torch.bfloat16)
                self.dma_read(DMA_DEVICE_C2H, enc_out_addr + t * D * bpe, enc_t_buf, D * bpe)
                self.dma_to_accelerator_memory(self.JOINT_ENC_DRAM, enc_t_buf)
                pred_buf = torch.zeros(H, dtype=torch.bfloat16)
                self.dma_read(DMA_DEVICE_C2H, self.PRED_OUT_DRAM, pred_buf, H * bpe)
                self.dma_to_accelerator_memory(self.JOINT_PRED_DRAM, pred_buf)
                # Joint token -> hardware argmax
                self.program_execute(tok_prog)
                token_id = self.get_arg_max_index1()
                # Joint duration -> hardware argmax
                self.program_execute(dur_prog)
                dur_idx = self.get_arg_max_index1()
                dur = self.tdt_durations[dur_idx] if dur_idx < len(self.tdt_durations) else 0
                total_steps += 1
                if token_id == self.blank_id:
                    # Restore LSTM state — blank means no state update
                    self.dma_to_accelerator_memory(self.PRED_H0_DRAM, saved_h0)
                    self.dma_to_accelerator_memory(self.PRED_C0_DRAM, saved_c0)
                    self.dma_to_accelerator_memory(self.PRED_H1_DRAM, saved_h1)
                    self.dma_to_accelerator_memory(self.PRED_C1_DRAM, saved_c1)
                    t += max(dur, 1)
                    break
                else:
                    tokens.append(token_id)
                    last_token = token_id
                    symbols += 1
                    if dur > 0:
                        t += dur
                        break
            else:
                t += 1
        print(f"  Decode: {total_steps} joint steps, {len(tokens)} tokens emitted")
        return tokens
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parakeet-TDT-0.6B accelerator inference — step-by-step verification")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file (.wav, .flac, etc.)")
    parser.add_argument("--dev", type=str, default="xdma0", help="XDMA device")
    parser.add_argument("--cycle", type=float, default=5.63, help="Clock cycle in ns")
    parser.add_argument("--profile", action="store_true", help="Profile encoder sub-blocks instead of running full inference")
    parser.add_argument("--compare", action="store_true", help="Layer-by-layer encoder comparison against NeMo reference")
    parser.add_argument("--frames", type=int, default=None, help="Truncate mel spectrogram to this many frames (e.g. 300 for ~3s)")
    parser.add_argument("--fp4", action="store_true", help="Use FP4 quantized encoder weights (auto-generates if missing)")
    parser.add_argument("--int4", action="store_true", help="Use INT4 quantized encoder weights (auto-generates if missing)")
    args = parser.parse_args()
    assert not (args.fp4 and args.int4), "--fp4 and --int4 are mutually exclusive"

    cfg = load_config()
    set_dma_device(args.dev)
    audio_path = args.audio or os.path.join(SCRIPT_DIR, cfg["defaults"]["default_audio"])
    print(f"Parakeet-TDT-0.6B on {args.dev}")

    import torchaudio
    assert os.path.exists(audio_path), f"Audio file not found: {audio_path}"
    print(f"  Loading: {audio_path}")
    waveform, sr = torchaudio.load(audio_path)
    if sr != cfg["preprocessing"]["sample_rate"]:
        waveform = torchaudio.functional.resample(waveform, sr, cfg["preprocessing"]["sample_rate"])
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    print(f"  Audio: {waveform.shape[1]} samples ({waveform.shape[1]/cfg['preprocessing']['sample_rate']:.1f}s)")

    # --- Init engine, load weights ---
    engine = Parakeet_UnifiedEngine(clock_period_ns=args.cycle, fp4=args.fp4, int4=args.int4)
    if engine.quant:
        print(f"  {engine.quant.upper()} encoder weights enabled")
    engine.weight_init()
    sd = engine._ckpt_sd

    # --- Mel spectrogram (CPU) ---
    mel = compute_mel_spectrogram(waveform, cfg, ckpt_sd=sd)
    if args.frames is not None:
        mel = mel[:, :args.frames, :]
        print(f"  --frames {args.frames}: truncated mel to {mel.shape[1]} frames")
    T_mel = mel.shape[1]
    n_mels = cfg["encoder"]["n_mels"]  # 128

    # --- Compute correct output dimensions through 3 subsampling stages ---
    H0, W0 = conv2d_outsize(T_mel), conv2d_outsize(n_mels)      # stage 0
    H1, W1 = conv2d_outsize(H0), conv2d_outsize(W0)             # stage 1
    H2, W2 = conv2d_outsize(H1), conv2d_outsize(W1)             # stage 2
    N0 = H0 * W0
    L = H2
    L_pad = pad_to_multiple(L, cfg["hardware"]["block_size"])

    audio_dur = waveform.shape[1] / cfg["preprocessing"]["sample_rate"]
    print(f"  Mel: T_mel={T_mel} ({audio_dur:.1f}s audio → {T_mel} frames)")
    print(f"  Sub dims: H0={H0} W0={W0} | H1={H1} W1={W1} | H2={H2} W2={W2}")
    print(f"  Encoder seq_len: L={L} (actual), L_pad={L_pad} (padded to {cfg['hardware']['block_size']}), P_pad={pad_to_multiple(2*L_pad-1, cfg['hardware']['block_size'])} (pos emb)")
    print(f"  N0={N0}")

    # --- Allocate DRAM buffers ---
    engine.tensor_init(L_pad)

    SC = engine.sub_channels  # 256
    D = engine.d_model         # 1024
    bpe = engine.bytes_per_element
    FF = engine.ff_dim         # 4096
    H_heads = engine.num_heads # 8
    dk = engine.head_dim       # 128
    la = engine.layer_addrs[0] # layer 0 weights

    # ==================================================================
    # Pre-stage: position embedding, attention mask, Toeplitz, BN tiling
    # ==================================================================
    N1 = H1 * W1
    N1_pad = pad_to_multiple(N1, engine.block_size)
    N2 = H2 * W2
    N2_pad = pad_to_multiple(N2, engine.block_size)
    P_pad = pad_to_multiple(2 * L_pad - 1, engine.block_size)

    # Position embedding
    rel_pe = engine.make_rel_pos_emb(L_pad)
    if rel_pe.shape[0] < P_pad:
        pe_padded = torch.zeros(P_pad, D, dtype=torch.bfloat16)
        pe_padded[:rel_pe.shape[0], :] = rel_pe
        rel_pe = pe_padded
    engine.dma_to_accelerator_memory(engine.POS_EMB_DRAM, rel_pe.contiguous())

    # Attention mask
    mask = torch.full((L_pad, L_pad), -1e38, dtype=torch.bfloat16)
    mask[:L, :L] = 0.0
    mask[L:, 0] = 0.0
    engine.dma_to_accelerator_memory(engine.ATTN_MASK_DRAM, mask.contiguous())

    # Toeplitz matrices for DW conv (all 24 layers)
    toeplitz_addrs = []
    for li in range(engine.num_layers):
        print(f"\r  Staging Toeplitz DW conv matrices... layer {li:2d}/{engine.num_layers - 1}", end="", flush=True)
        la_i = engine.layer_addrs[li]
        k_flat = torch.zeros(D * 64, dtype=torch.bfloat16)
        engine.dma_read(DMA_DEVICE_C2H, la_i["CONV_DW_W"], k_flat, D * 64 * bpe)
        kernel = k_flat.reshape(D, 64)
        toeplitz = torch.zeros(D, L_pad, L_pad, dtype=torch.bfloat16)
        for k in range(9):
            offset = k - 4
            t_idx = torch.arange(max(0, -offset), min(L_pad, L_pad - offset))
            toeplitz[:, t_idx, t_idx + offset] = kernel[:, k:k+1].expand(-1, len(t_idx))
        addr = engine.get_params_dram_addr()
        engine.allocate_params_dram(D * L_pad * L_pad * bpe)
        engine.dma_to_accelerator_memory(addr, toeplitz.reshape(-1).contiguous())
        toeplitz_addrs.append(addr)
    print(f"\r  Toeplitz staged: {engine.get_params_dram_usage()/1024**2:.0f} MB params used          ")

    # Pre-tile BN params and attention biases for bulk ops
    bn_tiled = []
    for li in range(engine.num_layers):
        la_i = engine.layer_addrs[li]
        s_addr, sh_addr = batch_norm_prepare_tiled(engine, D, L_pad,
            la_i["CONV_BN_SCALE"], la_i["CONV_BN_SHIFT"])
        bn_tiled.append((s_addr, sh_addr))
    engine.prepare_attention_tiled_biases(L_pad)

    # ==================================================================
    # Compile
    # ==================================================================
    import time as _time
    import sentencepiece as spm

    print(f"\n{'='*60}")
    print(f"  COMPILING...")
    print(f"{'='*60}")

    # Subsampling programs
    prog_s0, _ = engine.compile_sub_stage0(N0, 64, SC)
    prog_s1, _ = engine.compile_sub_stage_dw_pw(N0, N1, SC,
        engine.SUB_PATCH_DRAM, "SUB_DW1_W", "SUB_DW1_B", "SUB_PW1_W", "SUB_PW1_B")
    prog_s2, _ = engine.compile_sub_stage_dw_pw(N1, N2, SC,
        engine.SUB_PATCH_DRAM, "SUB_DW2_W", "SUB_DW2_B", "SUB_PW2_W", "SUB_PW2_B")
    prog_lin, _ = engine.compile_sub_linear(L_pad)
    print(f"  Subsampling compiled (4 programs)")

    if args.compare:
        # --- Compare mode: only compile per-layer programs ---
        enc_layer_programs = engine.compile_encoder_profile(L_pad, toeplitz_addrs, bn_tiled)
        print(f"  Encoder per-layer compiled: {len(enc_layer_programs)} sub-block programs")
        print(f"\n  Compilation Finished")
    elif args.profile:
        # --- Profile mode: compile each sub-block separately ---
        enc_programs = engine.compile_encoder_profile(L_pad, toeplitz_addrs, bn_tiled)
        print(f"  Encoder profiled: {len(enc_programs)} sub-block programs")
        # Also compile single-program encoder for comparison
        enc_prog_addr, enc_prog_bytes = engine.compile_encoder(L_pad, toeplitz_addrs, bn_tiled)
        print(f"  Encoder single-program also compiled: {enc_prog_bytes:,} bytes")
        print(f"\n  Compilation Finished")
    else:
        # Encoder program (single instruction stream for 24 conformer layers)
        enc_prog_addr, enc_prog_bytes = engine.compile_encoder(L_pad, toeplitz_addrs, bn_tiled)
        print(f"  Encoder compiled: {enc_prog_bytes:,} bytes")
        # Decoder programs (3 small programs for host-driven decode loop)
        pred_prog, tok_prog, dur_prog, _ = engine.compile_decoder()
        engine.progs = {"pred": (pred_prog, 0), "joint_tok": (tok_prog, 0), "joint_dur": (dur_prog, 0)}
        print(f"  Decoder compiled (3 programs)")
        print(f"\n  Compilation Finished")

    # ==================================================================
    # Execute
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"  EXECUTING FORWARD PASS...")
    print(f"{'='*60}")

    t_start = _time.perf_counter()

    # --- Subsampling (im2col on host, matmuls on FPGA) ---
    mel_2d = mel.squeeze(0)
    patches0, H0_actual, W0_actual = engine._im2col_conv2d(mel_2d, T_mel, n_mels)
    assert H0_actual == H0 and W0_actual == W0

    # Stage 0: Conv2d(1->256, k=3, s=2) + ReLU
    engine.dma_to_accelerator_memory(engine.SUB_OUT0_DRAM, torch.zeros(N0 * SC, dtype=torch.bfloat16))
    engine.dma_to_accelerator_memory(engine.SUB_PATCH_DRAM, patches0.contiguous())
    engine.program_execute(prog_s0)

    # Stage 1: DWConv2d(256, k=3, s=2) + PWConv2d(256) + ReLU
    s0_out = read_dram(engine, engine.SUB_OUT0_DRAM, N0 * SC)
    s0_ch = s0_out.reshape(H0, W0, SC).permute(2, 0, 1).contiguous()
    dw1_p, _, _ = engine._im2col_conv2d(s0_ch, H0, W0)
    engine.dma_to_accelerator_memory(engine.SUB_PATCH_DRAM, dw1_p.contiguous())
    engine.program_execute(prog_s1)

    # Stage 2: DWConv2d(256, k=3, s=2) + PWConv2d(256) + ReLU
    s1_out = read_dram(engine, engine.SUB_PW_IN_DRAM, N1_pad * SC)
    s1_ch = s1_out.reshape(N1_pad, SC)[:N1].reshape(H1, W1, SC).permute(2, 0, 1).contiguous()
    dw2_p, _, _ = engine._im2col_conv2d(s1_ch, H1, W1)
    engine.dma_to_accelerator_memory(engine.SUB_PATCH_DRAM, dw2_p.contiguous())
    engine.program_execute(prog_s2)

    # Flatten + Linear(4096, 1024)
    s2_out = read_dram(engine, engine.SUB_PW_IN_DRAM, N2_pad * SC)
    fl = s2_out.reshape(N2_pad, SC)[:N2].reshape(H2, W2, SC).permute(0, 2, 1).reshape(H2, SC * W2).contiguous()
    if H2 < L_pad:
        fp = torch.zeros(L_pad, 4096, dtype=torch.bfloat16); fp[:H2, :] = fl; fl = fp
    engine.dma_to_accelerator_memory(engine.SUB_FLAT_DRAM, fl.contiguous())
    engine.dma_to_accelerator_memory(engine.INPUT_DRAM, torch.zeros(L_pad * D, dtype=torch.bfloat16))
    engine.program_execute(prog_lin)

    # Padding rows for LayerNorm stability
    if L_pad > H2:
        ef = torch.zeros((L_pad - H2) * D, dtype=torch.bfloat16)
        ef[0::2] = 0.1; ef[1::2] = -0.1
        engine.dma_to_accelerator_memory(engine.INPUT_DRAM + H2 * D * bpe, ef)

    # Save subsampling output (encoder input) for --compare mode
    sub_output = read_dram(engine, engine.INPUT_DRAM, L_pad * D)

    t_sub_done = _time.perf_counter()

    if args.compare:
        # --- Compare mode: HW vs CPU BF16 reference (same weights, same input) ---
        # Use the standalone parakeet_modules.py implementation
        sys.path.insert(0, "/home/rohit/apex-compute-ML/simple-llm/src/parakeet")
        from parakeet_modules import ParakeetTDT, make_rel_pos_emb as cpu_make_rel_pos_emb
        sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", ".."))  # restore

        # Build CPU reference model in BF16, load same checkpoint weights
        print(f"  Loading CPU reference model (BF16)...")
        cpu_model = ParakeetTDT()
        cpu_sd = engine._ckpt_sd

        # Load encoder layer weights into CPU model
        for i in range(engine.num_layers):
            layer = cpu_model.encoder.layers[i]
            pfx = f"encoder.layers.{i}"
            layer.ln_ff1.weight.data.copy_(cpu_sd[f"{pfx}.norm_feed_forward1.weight"])
            layer.ln_ff1.bias.data.copy_(cpu_sd[f"{pfx}.norm_feed_forward1.bias"])
            layer.ln_attn.weight.data.copy_(cpu_sd[f"{pfx}.norm_self_att.weight"])
            layer.ln_attn.bias.data.copy_(cpu_sd[f"{pfx}.norm_self_att.bias"])
            layer.ln_conv.weight.data.copy_(cpu_sd[f"{pfx}.norm_conv.weight"])
            layer.ln_conv.bias.data.copy_(cpu_sd[f"{pfx}.norm_conv.bias"])
            layer.ln_ff2.weight.data.copy_(cpu_sd[f"{pfx}.norm_feed_forward2.weight"])
            layer.ln_ff2.bias.data.copy_(cpu_sd[f"{pfx}.norm_feed_forward2.bias"])
            layer.ln_out.weight.data.copy_(cpu_sd[f"{pfx}.norm_out.weight"])
            layer.ln_out.bias.data.copy_(cpu_sd[f"{pfx}.norm_out.bias"])
            layer.ff1.linear1.weight.data.copy_(cpu_sd[f"{pfx}.feed_forward1.linear1.weight"])
            layer.ff1.linear2.weight.data.copy_(cpu_sd[f"{pfx}.feed_forward1.linear2.weight"])
            layer.ff2.linear1.weight.data.copy_(cpu_sd[f"{pfx}.feed_forward2.linear1.weight"])
            layer.ff2.linear2.weight.data.copy_(cpu_sd[f"{pfx}.feed_forward2.linear2.weight"])
            layer.self_attn.linear_q.weight.data.copy_(cpu_sd[f"{pfx}.self_attn.linear_q.weight"])
            layer.self_attn.linear_k.weight.data.copy_(cpu_sd[f"{pfx}.self_attn.linear_k.weight"])
            layer.self_attn.linear_v.weight.data.copy_(cpu_sd[f"{pfx}.self_attn.linear_v.weight"])
            layer.self_attn.linear_out.weight.data.copy_(cpu_sd[f"{pfx}.self_attn.linear_out.weight"])
            layer.self_attn.linear_pos.weight.data.copy_(cpu_sd[f"{pfx}.self_attn.linear_pos.weight"])
            layer.self_attn.pos_bias_u.data.copy_(cpu_sd[f"{pfx}.self_attn.pos_bias_u"])
            layer.self_attn.pos_bias_v.data.copy_(cpu_sd[f"{pfx}.self_attn.pos_bias_v"])
            layer.conv.pw_conv1_w.data.copy_(cpu_sd[f"{pfx}.conv.pointwise_conv1.weight"])
            layer.conv.dw_conv_w.data.copy_(cpu_sd[f"{pfx}.conv.depthwise_conv.weight"])
            layer.conv.pw_conv2_w.data.copy_(cpu_sd[f"{pfx}.conv.pointwise_conv2.weight"])
            layer.conv.bn_weight.data.copy_(cpu_sd[f"{pfx}.conv.batch_norm.weight"])
            layer.conv.bn_bias.data.copy_(cpu_sd[f"{pfx}.conv.batch_norm.bias"])
            layer.conv.bn_mean.copy_(cpu_sd[f"{pfx}.conv.batch_norm.running_mean"])
            layer.conv.bn_var.copy_(cpu_sd[f"{pfx}.conv.batch_norm.running_var"])

        cpu_model.bfloat16()
        cpu_model.eval()
        cpu_model.stage_for_hardware()

        # Run CPU reference layer by layer from same subsampling output
        cpu_x = sub_output.reshape(1, L_pad, D).to(torch.bfloat16)
        pos_emb = cpu_make_rel_pos_emb(L_pad).to(torch.bfloat16)

        # Reset INPUT_DRAM with saved subsampling output for HW
        engine.dma_to_accelerator_memory(engine.INPUT_DRAM, sub_output.contiguous())

        def _compare(hw_tensor, ref_tensor, T):
            """Compute cos sim, SNR, max/mean error between HW and CPU reference."""
            hw = hw_tensor[:T].flatten()
            ref = ref_tensor[:T].flatten()
            cos = F.cosine_similarity(hw, ref, dim=0).item()
            err = (hw_tensor[:T] - ref_tensor[:T]).abs()
            ref_pow = (ref_tensor[:T] ** 2).mean()
            err_pow = ((hw_tensor[:T] - ref_tensor[:T]) ** 2).mean()
            snr = 10 * torch.log10(ref_pow / err_pow).item() if err_pow > 0 else float('inf')
            status = "OK" if cos > 0.99 else "WARN" if cos > 0.95 else "BAD"
            return cos, snr, err.max().item(), err.mean().item(), status

        print(f"\n{'='*90}")
        print(f"  PER-SUB-BLOCK ENCODER COMPARISON (same input + weights, HW vs CPU BF16)")
        print(f"{'='*90}")
        print(f"  {'Layer':>5} {'Block':<12}  {'Cos Sim':>10}  {'SNR(dB)':>10}  {'Max Err':>10}  {'Mean Err':>10}  Status")
        print(f"  {'-'*75}")

        block_names = ["FF1", "SelfAttn", "ConvModule", "FF2", "FinalLN"]
        T = L  # compare only valid positions

        # Also compile decomposed attention for layer 0
        P_pad = pad_to_multiple(2 * L_pad - 1, engine.block_size)
        print(f"  Compiling decomposed attention for layer 0...")
        attn_decomp = engine.compile_attn_decomposed(0, L_pad, toeplitz_addrs, bn_tiled)

        with torch.no_grad():
            for layer_idx in range(engine.num_layers):
                layer_progs = [(n, b, d, a, f) for (n, b, d, a, f) in enc_layer_programs if n == layer_idx]
                cpu_layer = cpu_model.encoder.layers[layer_idx]

                for block_idx, (_, block_name, _, prog_addr, _) in enumerate(layer_progs):
                    if block_name == "SelfAttn" and layer_idx == 0:
                        # --- Decomposed attention comparison for layer 0 ---
                        # CPU: compute intermediate attention states
                        cpu_ln = cpu_layer.ln_attn(cpu_x)
                        cpu_q = cpu_layer.self_attn.linear_q(cpu_ln)
                        cpu_k = cpu_layer.self_attn.linear_k(cpu_ln)
                        cpu_v = cpu_layer.self_attn.linear_v(cpu_ln)
                        cpu_p = cpu_layer.self_attn.linear_pos(pos_emb)

                        # HW: Part 1 — LN + QKV + Pos projections
                        attn_name, attn_addr = attn_decomp[0]
                        engine.start_execute_from_dram(attn_addr)
                        engine.wait_queue(120.0)

                        # Read HW Q, K, V, Pos from DRAM
                        hw_q = read_dram(engine, engine.Q_DRAM, L_pad * D).reshape(L_pad, D).float()
                        hw_k = read_dram(engine, engine.K_DRAM, L_pad * D).reshape(L_pad, D).float()
                        hw_v = read_dram(engine, engine.V_DRAM, L_pad * D).reshape(L_pad, D).float()
                        hw_p = read_dram(engine, engine.POS_PROJ_DRAM, P_pad * D).reshape(P_pad, D).float()
                        ref_q = cpu_q.squeeze(0).float()
                        ref_k = cpu_k.squeeze(0).float()
                        ref_v = cpu_v.squeeze(0).float()
                        ref_p = cpu_p.squeeze(0).float()

                        for name, hw_t, ref_t, sz in [("Q", hw_q, ref_q, L), ("K", hw_k, ref_k, L), ("V", hw_v, ref_v, L), ("Pos", hw_p, ref_p, min(ref_p.shape[0], P_pad))]:
                            cos, snr, mx, mn, st = _compare(hw_t, ref_t, sz)
                            print(f"  {layer_idx:>5} {name:<12}  {cos:>10.6f}  {snr:>10.2f}  {mx:>10.4f}  {mn:>10.4f}  {st}")

                        # Verify POS_EMB_DRAM content
                        hw_pe = read_dram(engine, engine.POS_EMB_DRAM, P_pad * D).reshape(P_pad, D).float()
                        cpu_pe = pos_emb.squeeze(0).float()
                        T_pe = min(cpu_pe.shape[0], P_pad)
                        pe_cos, pe_snr, pe_mx, pe_mn, pe_st = _compare(hw_pe, cpu_pe, T_pe)
                        print(f"  {layer_idx:>5} {'POS_EMB chk':<12}  {pe_cos:>10.6f}  {pe_snr:>10.2f}  {pe_mx:>10.4f}  {pe_mn:>10.4f}  {pe_st}")

                        # Split Pos projection by M_chunk to check if second tile is the problem
                        # M_chunk for M=P_pad, K=D: N_chunk=192, M_chunk=min(P_pad, 262144//(D+192))=215
                        m_chunk = min(P_pad, user_dma_core.URAM_FULL_ELEMENTS // (D + 192))
                        split = min(m_chunk, ref_p.shape[0])
                        if split < ref_p.shape[0]:
                            cos1, snr1, mx1, mn1, st1 = _compare(hw_p[:split], ref_p[:split], split)
                            rem = min(ref_p.shape[0], P_pad) - split
                            cos2, snr2, mx2, mn2, st2 = _compare(hw_p[split:split+rem], ref_p[split:split+rem], rem)
                            print(f"  {layer_idx:>5} {'Pos[:'+str(split)+']':<12}  {cos1:>10.6f}  {snr1:>10.2f}  {mx1:>10.4f}  {mn1:>10.4f}  {st1}")
                            print(f"  {layer_idx:>5} {'Pos['+str(split)+':]':<12}  {cos2:>10.6f}  {snr2:>10.2f}  {mx2:>10.4f}  {mn2:>10.4f}  {st2}")

                        # HW: Part 2 — Per-head attention loop
                        attn_name, attn_addr = attn_decomp[1]
                        engine.start_execute_from_dram(attn_addr)
                        engine.wait_queue(120.0)

                        hw_attn_out = read_dram(engine, engine.ATTN_OUT_DRAM, L_pad * D).reshape(L_pad, D).float()
                        # CPU: run full attention to get attn output (before out proj)
                        cpu_attn = cpu_layer.self_attn(cpu_ln, pos_emb)

                        # We can't easily get pre-out-proj from CPU module, so compare after out+residual
                        print(f"  {layer_idx:>5} {'Attn:Heads':<12}  (see ATTN_OUT below)")

                        # HW: Part 3 — Output proj + residual
                        attn_name, attn_addr = attn_decomp[2]
                        engine.start_execute_from_dram(attn_addr)
                        engine.wait_queue(120.0)

                        hw_out = read_dram(engine, engine.INPUT_DRAM, L_pad * D).reshape(L_pad, D).float()
                        # CPU: residual add
                        cpu_x = cpu_x + cpu_attn
                        ref = cpu_x.squeeze(0).float()
                        cos, snr, max_err, mean_err, status = _compare(hw_out, ref, T)
                        print(f"  {layer_idx:>5} {'SelfAttn':<12}  {cos:>10.6f}  {snr:>10.2f}  {max_err:>10.4f}  {mean_err:>10.4f}  {status}")
                        continue

                    # HW: execute this sub-block
                    engine.start_execute_from_dram(prog_addr)
                    engine.wait_queue(120.0)
                    hw_out = read_dram(engine, engine.INPUT_DRAM, L_pad * D).reshape(L_pad, D).float()

                    # CPU: execute matching sub-block
                    if block_name == "FF1":
                        cpu_x = cpu_x + 0.5 * cpu_layer.ff1(cpu_layer.ln_ff1(cpu_x))
                    elif block_name == "SelfAttn":
                        cpu_x = cpu_x + cpu_layer.self_attn(cpu_layer.ln_attn(cpu_x), pos_emb)
                    elif block_name == "ConvModule":
                        cpu_x = cpu_x + cpu_layer.conv(cpu_layer.ln_conv(cpu_x))
                    elif block_name == "FF2":
                        cpu_x = cpu_x + 0.5 * cpu_layer.ff2(cpu_layer.ln_ff2(cpu_x))
                    elif block_name == "FinalLN":
                        cpu_x = cpu_layer.ln_out(cpu_x)

                    ref = cpu_x.squeeze(0).float()
                    cos, snr, max_err, mean_err, status = _compare(hw_out, ref, T)
                    print(f"  {layer_idx:>5} {block_name:<12}  {cos:>10.6f}  {snr:>10.2f}  {max_err:>10.4f}  {mean_err:>10.4f}  {status}")

        print(f"\n{'='*90}")
        return

    if args.profile:
        # --- Profile mode: run each sub-block individually ---
        profile_results = engine.run_encoder_profile(enc_programs)

        # --- Single-program comparison (Approach 3) ---
        # Re-upload input data (profile runs may have modified INPUT_DRAM)
        dummy_input = torch.zeros(L_pad * D, dtype=torch.bfloat16)
        dummy_input[0::2] = 0.1; dummy_input[1::2] = -0.1
        engine.dma_to_accelerator_memory(engine.INPUT_DRAM, dummy_input)

        print(f"\n{'='*120}")
        print(f"  SINGLE-PROGRAM COMPARISON")
        print(f"{'='*120}")

        t_single_start = _time.perf_counter()
        engine.start_execute_from_dram(enc_prog_addr)
        engine.wait_queue(300.0)
        single_hw_us = engine.report_latency_in_us()
        single_instr = engine.read_reg32(user_dma_core.UE_INSTRUCTION_CTL_ADDR)
        t_single_end = _time.perf_counter()
        single_wall_s = t_single_end - t_single_start

        profile_total_hw = sum(r["hw_us"] for r in profile_results)
        profile_total_wall = sum(r["wall_s"] for r in profile_results)
        profile_total_instr = sum(r["instr_count"] for r in profile_results)

        print(f"  {'Metric':<30}  {'120 Dispatches':>16}  {'Single Program':>16}  {'Difference':>16}")
        print(f"  {'-'*82}")
        print(f"  {'HW latency (ms)':<30}  {profile_total_hw/1000:>16.2f}  {single_hw_us/1000:>16.2f}  {(single_hw_us - profile_total_hw)/1000:>+16.2f}")
        print(f"  {'Wall clock (ms)':<30}  {profile_total_wall*1000:>16.2f}  {single_wall_s*1000:>16.2f}  {(single_wall_s - profile_total_wall)*1000:>+16.2f}")
        print(f"  {'Instructions':<30}  {profile_total_instr:>16}  {single_instr:>16}  {single_instr - profile_total_instr:>+16}")
        print(f"  {'Wall / HW ratio':<30}  {profile_total_wall*1000/(profile_total_hw/1000):>16.2f}x  {single_wall_s*1000/(single_hw_us/1000):>16.2f}x")
        per_dispatch_overhead = (profile_total_wall - single_wall_s) / len(enc_programs)
        print(f"\n  Per-dispatch overhead:    {per_dispatch_overhead*1000:>10.2f} ms  (total {(profile_total_wall - single_wall_s)*1000:.1f} ms across {len(enc_programs)} dispatches)")
        print(f"  Single-program speedup:  {profile_total_wall / single_wall_s:>10.2f}x wall-clock")
        print(f"{'='*120}")
        return

    # --- Encoder (single instruction stream, no host intervention) ---
    engine.start_execute_from_dram(enc_prog_addr)
    engine.wait_queue(120.0)
    engine.report_timing_and_instruction_count()

    t_enc_done = _time.perf_counter()

    # --- Encoder diagnostic: print key info and stop ---
    enc_sample = read_dram(engine, engine.INPUT_DRAM, min(64, L_pad * D))
    vals = enc_sample[:8].tolist()
    mode_str = engine.quant.upper() if engine.quant else 'BF16'
    print(f"\n  === ENCODER DIAGNOSTIC ===")
    print(f"  Mode: {mode_str}")
    print(f"  Program bytes: {enc_prog_bytes:,}")
    print(f"  Encoder output first 8 values: {[f'{v:.6f}' for v in vals]}")
    print(f"  Matmul dispatch: {Parakeet_UnifiedEngine._enc_matmul_quant_count} {mode_str}, {Parakeet_UnifiedEngine._enc_matmul_bf16_count} BF16")
    assert False, f"DIAGNOSTIC STOP — compare BF16 vs --fp4 vs --int4 output above"

    # --- Decoder (host-driven greedy loop) ---
    hw_enc_out = read_dram(engine, engine.INPUT_DRAM, L_pad * D)
    engine.dma_to_accelerator_memory(engine.ENC_OUT_DRAM, hw_enc_out.contiguous())
    hw_tokens = engine.run_decode(engine.ENC_OUT_DRAM, L)

    t_dec_done = _time.perf_counter()

    # ==================================================================
    # CPU Decode (reference using HW encoder output)
    # ==================================================================
    print(f"\n  Running CPU decode on HW encoder output...")
    sd = engine._ckpt_sd

    # Read HW encoder output from accelerator DRAM
    hw_enc_out_flat = read_dram(engine, engine.ENC_OUT_DRAM, L_pad * D)
    hw_enc_out = hw_enc_out_flat.reshape(L_pad, D).float()  # (L_pad, D)

    # Load decoder weights from checkpoint
    embed_w = sd["decoder.prediction.embed.weight"].float()  # (vocab, H)
    lstm_wih = [sd[f"decoder.prediction.dec_rnn.lstm.weight_ih_l{i}"].float() for i in range(2)]
    lstm_whh = [sd[f"decoder.prediction.dec_rnn.lstm.weight_hh_l{i}"].float() for i in range(2)]
    lstm_bih = [sd[f"decoder.prediction.dec_rnn.lstm.bias_ih_l{i}"].float() for i in range(2)]
    lstm_bhh = [sd[f"decoder.prediction.dec_rnn.lstm.bias_hh_l{i}"].float() for i in range(2)]
    joint_enc_w = sd["joint.enc.weight"].float()    # (H, D)
    joint_enc_b = sd["joint.enc.bias"].float()       # (H,)
    joint_pred_w = sd["joint.pred.weight"].float()   # (H, H)
    joint_pred_b = sd["joint.pred.bias"].float()     # (H,)
    joint_out_w = sd["joint.joint_net.2.weight"].float()  # (vocab+dur, H)
    joint_out_b = sd["joint.joint_net.2.bias"].float()    # (vocab+dur,)
    N_tok = engine.vocab_size
    N_dur = len(engine.tdt_durations)

    H_pred = engine.pred_hidden  # 640
    h_states = [torch.zeros(H_pred), torch.zeros(H_pred)]
    c_states = [torch.zeros(H_pred), torch.zeros(H_pred)]

    cpu_tokens = []
    t_cpu = 0
    last_tok = engine.blank_id

    while t_cpu < L:
        symbols = 0
        while symbols < engine.max_symbols_per_step:
            saved_h = [h.clone() for h in h_states]
            saved_c = [c.clone() for c in c_states]

            # LSTM predictor (2 layers)
            x = embed_w[last_tok]  # (H,)
            for i in range(2):
                gates = x @ lstm_wih[i].T + lstm_bih[i] + h_states[i] @ lstm_whh[i].T + lstm_bhh[i]
                ig, fg, gg, og = gates.chunk(4)
                ig = torch.sigmoid(ig)
                fg = torch.sigmoid(fg)
                gg = torch.tanh(gg)
                og = torch.sigmoid(og)
                c_states[i] = fg * c_states[i] + ig * gg
                h_states[i] = og * torch.tanh(c_states[i])
                x = h_states[i]

            pred_out = h_states[1]  # (H,)

            # Joint network
            enc_proj = hw_enc_out[t_cpu] @ joint_enc_w.T + joint_enc_b    # (H,)
            pred_proj = pred_out @ joint_pred_w.T + joint_pred_b           # (H,)
            joint_hidden = torch.relu(enc_proj + pred_proj)                # (H,)
            logits = joint_hidden @ joint_out_w.T + joint_out_b            # (vocab+dur,)

            token_id = logits[:N_tok].argmax().item()
            dur_logits = logits[N_tok:N_tok + N_dur]
            dur_idx = dur_logits.argmax().item()
            dur = engine.tdt_durations[dur_idx] if dur_idx < len(engine.tdt_durations) else 0

            if token_id == engine.blank_id:
                h_states = saved_h
                c_states = saved_c
                t_cpu += max(dur, 1)
                break
            else:
                cpu_tokens.append(token_id)
                last_tok = token_id
                symbols += 1
                if dur > 0:
                    t_cpu += dur
                    break
        else:
            t_cpu += 1

    t_cpu_dec_done = _time.perf_counter()

    # ==================================================================
    # Results
    # ==================================================================
    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER_PATH)
    vocab_sz = sp.GetPieceSize()
    hw_text = sp.DecodeIds([t for t in hw_tokens if 0 <= t < vocab_sz])
    cpu_text = sp.DecodeIds([t for t in cpu_tokens if 0 <= t < vocab_sz])

    if args.frames is not None:
        audio_duration = T_mel * cfg["preprocessing"]["hop_length"] / cfg["preprocessing"]["sample_rate"]
    else:
        audio_duration = waveform.shape[1] / cfg["preprocessing"]["sample_rate"]
    total = t_dec_done - t_start

    print(f"\n  >>> HW decode:                {hw_text}")
    print(f"  >>> CPU decode (HW encoder):  {cpu_text}")
    if hw_text == cpu_text:
        print(f"  >>> MATCH")
    else:
        print(f"  >>> MISMATCH — encoder or decoder divergence")
        print(f"  >>> HW tokens ({len(hw_tokens)}): {hw_tokens[:20]}...")
        print(f"  >>> CPU tokens ({len(cpu_tokens)}): {cpu_tokens[:20]}...")
    print(f"\n  Timing:")
    print(f"    Subsampling:         {t_sub_done - t_start:.3f}s")
    print(f"    Encoder (24 layers): {t_enc_done - t_sub_done:.3f}s")
    print(f"    HW Decoder:          {t_dec_done - t_enc_done:.3f}s")
    print(f"    Total (HW path):     {total:.3f}s")
    print(f"    Audio duration:      {audio_duration:.1f}s")
    print(f"    Real-time factor:    {total / audio_duration:.2f}x")

    print(f"\n{'='*60}")
    print(f"  FULL PIPELINE COMPLETE")
    print(f"{'='*60}")

    # --- NeMo model inference (BF16) for transcript comparison ---
    import logging, warnings, io, contextlib
    logging.getLogger().setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore")

    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        import nemo.collections.asr as nemo_asr
        nemo_path = os.path.join(SCRIPT_DIR, "parakeet_bin", "models--nvidia--parakeet-tdt-0.6b-v3",
                                 "snapshots", "6d590f77001d318fb17a0b5bf7ee329a91b52598",
                                 "parakeet-tdt-0.6b-v3.nemo")
        model = nemo_asr.models.ASRModel.restore_from(nemo_path)
        model.bfloat16()
    t_nemo_start = _time.perf_counter()
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        transcript_bf16 = model.transcribe([audio_path])
    t_nemo_done = _time.perf_counter()
    print(f"\n  >>> NeMo BF16: {transcript_bf16[0].text}")
    print(f"  >>> NeMo BF16 wall clock: {t_nemo_done - t_nemo_start:.3f}s (RTF: {(t_nemo_done - t_nemo_start) / audio_duration:.2f}x)")

if __name__ == "__main__":
    main()