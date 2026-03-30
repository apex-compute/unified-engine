#!/usr/bin/env python3
"""
Swin-Large (patch4, window12, 384, in22k) inference on accelerator.

  - Config from swin_config.json; weights from a single bin (see below).
  - Forward pass: patch embedding → 4 encoder stages → layernorm → pool → classifier.
  - Each stage: N windowed self-attention layers + patch merging (except last stage).

Weights:
  - Default: swin_bin/weights_swin_hf.bin (generated from HF model if missing).

Usage:
  python swin_test.py
  python swin_test.py --image path/to/image.jpg
  python swin_test.py --dev xdma0 [--cycle 5.62]

Fixed layout: swin_test.py, swin_config.json, and swin_bin/ live in the same folder.
  user_dma_core.py is two folders up (repo root); that directory is added to sys.path.
"""

import json
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification
from huggingface_hub import snapshot_download
import time

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, DRAM_INSTRUCTION_ADDR, TYPE,
    UE_VECTOR_SIZE, URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS,
    set_dma_device, UnifiedEngine,
)

# ---------------------------------------------------------------------------
# Helper ops (built on top of UnifiedEngine primitives)
# ---------------------------------------------------------------------------

def bf16_patching_core(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                       weight_dram_addrs: list[int],
                       C: int = 3, H: int = 384, W: int = 384,
                       patch_h: int = 4, patch_w: int = 4,
                       K: int = 768, N: int = 192) -> int:
    """BF16 patching core: extract patches from CHW image and project with bf16 weights.

    Same gather logic as patching_core but uses bf16 dot product (weight in URAM_B)
    instead of quantized dot product (weight streamed from DRAM).

    The image is divided into (H/patch_h) x (W/patch_w) patches. Patches along W
    are grouped by len(weight_dram_addrs), each with its own (N, K) bf16 weight.
    patches_per_group * patch_w must equal UE_VECTOR_SIZE (64).

    Each group's input is C*patch_h URAM_A rows of 64 elements. Each patch in the
    group occupies patch_w=4 columns within those rows. The weight for patch p
    selects and projects those columns.

    Args:
        ue: UnifiedEngine instance (must be in capture mode)
        INPUT_DRAM_ADDR: CHW image in DRAM (C x H x W, bf16)
        OUTPUT_DRAM_ADDR: output (n_patches_h * n_patches_w, N, bf16)
        weight_dram_addrs: list of patches_per_group DRAM addrs, each (N, K) bf16
        C, H, W: image dimensions
        patch_h, patch_w: patch size
        K: weight inner dim = C * patch_h * UE_VECTOR_SIZE
        N: output dim per patch

    Returns total flops.
    """
    from user_dma_core import (URAM_START_ADDR, URAM_HALFWAY_ADDR, URAM_SECTION,
                                URAM_NEAR_FULL_ELEMENTS, MEMCPY_TYPE)

    bytes_per_element = 2
    patches_per_group = len(weight_dram_addrs)
    n_patches_h = H // patch_h
    n_patches_w = W // patch_w
    n_groups_w = n_patches_w // patches_per_group

    assert patches_per_group * patch_w == UE_VECTOR_SIZE, \
        f"patches_per_group*patch_w={patches_per_group * patch_w} must equal UE_VECTOR_SIZE={UE_VECTOR_SIZE}"
    assert n_patches_w % patches_per_group == 0
    assert K == C * patch_h * UE_VECTOR_SIZE

    input_bytes_per_uram_row = UE_VECTOR_SIZE * bytes_per_element  # 128 bytes
    output_row_uram_addr = URAM_HALFWAY_ADDR
    batch_output_dram_addr = OUTPUT_DRAM_ADDR

    total_flops = 0

    for p_i in range(n_patches_h):
        for p_j in range(n_groups_w):
            # Gather: load C*patch_h URAM_A rows from image
            patch_offset_bytes = (p_i * patch_h * W + p_j * patches_per_group * patch_w) * bytes_per_element

            for channel in range(C):
                for row_idx in range(patch_h):
                    uram_addr = URAM_START_ADDR + channel * patch_h + row_idx
                    src_dram = (INPUT_DRAM_ADDR
                                + channel * H * W * bytes_per_element
                                + row_idx * W * bytes_per_element
                                + patch_offset_bytes)
                    ue.ue_memcpy_from_dram(src_dram, input_bytes_per_uram_row, 0,
                                           uram_addr, URAM_SECTION.URAM_A.value, ue._inst_id)
                    ue._inst_id += 1

            # Project: for each patch in group, load weight to URAM_B, bf16 matvec
            output_sram_base = output_row_uram_addr * UE_VECTOR_SIZE * bytes_per_element
            for p in range(patches_per_group):
                # Load weight (N, K) into URAM_B
                ue.accelerator_memory_to_sram(
                    accelerator_dram_address=weight_dram_addrs[p],
                    sram_address=0x80000,  # URAM_B start
                    element_size=N * K,
                )

                # BF16 matvec: input vector in URAM_A (C*patch_h rows = K/64 rows),
                # weight matrix in URAM_B (N rows of K elements each).
                # Output N elements for this patch.
                out_sram = output_sram_base + p * N * bytes_per_element
                ue.start_queue_for_bf16_matvec_operation(
                    max_clear_en=0,
                    fmax_context_addr=0,
                    vector_sram_start_addr=0x00000,  # URAM_A start: input
                    matrix_sram_start_addr=0x80000,   # URAM_B start: weight
                    output_sram_wb_addr=out_sram,
                    K=K,
                    N=N,
                )
                total_flops += 2 * N * K

            # Write all patches_per_group * N outputs to DRAM
            ue.sram_to_accelerator_memory(
                sram_address=output_row_uram_addr * UE_VECTOR_SIZE * bytes_per_element,
                accelerator_dram_address=batch_output_dram_addr,
                element_size=patches_per_group * N,
            )
            batch_output_dram_addr += patches_per_group * N * bytes_per_element

    return total_flops


def flash_attention_batched(ue: UnifiedEngine, num_batches: int, head_dim: int, seq_len: int,
                            Q_DRAM_ADDR: int, K_DRAM_ADDR: int, V_DRAM_ADDR: int,
                            OUTPUT_DRAM_ADDR: int, SCRATCH_DRAM_ADDR: int,
                            BIAS_DRAM_ADDR: int = None) -> int:
    """Batched flash attention: loop over num_batches calling flash_attention_core.

    Each batch is one attention head. Q/K/V/OUTPUT are laid out contiguously
    in DRAM with stride (seq_len * head_dim) per batch. BIAS stride is
    (seq_len * seq_len) per batch if provided.

    Returns total flops.
    """
    bytes_per_element = 2
    qkv_batch_stride = seq_len * head_dim * bytes_per_element
    output_batch_stride = seq_len * head_dim * bytes_per_element
    scratch_batch_stride = head_dim * seq_len * bytes_per_element + seq_len * seq_len * bytes_per_element  # V^T + partial softmax
    bias_batch_stride = seq_len * seq_len * bytes_per_element if BIAS_DRAM_ADDR is not None else 0

    total_flops = 0
    for b in range(num_batches):
        bias_addr = BIAS_DRAM_ADDR + b * bias_batch_stride if BIAS_DRAM_ADDR is not None else None
        total_flops += ue.flash_attention_core(
            head_dim=head_dim,
            seq_len=seq_len,
            Q_DRAM_ADDR=Q_DRAM_ADDR + b * qkv_batch_stride,
            K_DRAM_ADDR=K_DRAM_ADDR + b * qkv_batch_stride,
            V_DRAM_ADDR=V_DRAM_ADDR + b * qkv_batch_stride,
            OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR + b * output_batch_stride,
            SCRATCH_DRAM_ADDR=SCRATCH_DRAM_ADDR + b * scratch_batch_stride,
            BIAS_DRAM_ADDR=bias_addr,
        )
    return total_flops
def window_partition_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                          H: int, W: int, C: int, window_size: int) -> None:
    """Rearrange (H, W, C) -> (num_windows, window_size*window_size, C) in DRAM.

    Each window is a contiguous (window_size*window_size, C) block in the output.
    Gathers rows from the spatial layout using strided DMA.

    Data layout assumption: input is row-major (H, W, C) in DRAM.
    """
    bytes_per_element = 2
    num_windows_h = H // window_size
    num_windows_w = W // window_size
    window_row_bytes = window_size * C * bytes_per_element  # one window-row: window_size pixels * C channels
    input_row_stride = W * C * bytes_per_element            # full spatial row in input
    window_elements = window_size * window_size * C         # total elements per window

    output_offset = 0
    for wh in range(num_windows_h):
        for ww in range(num_windows_w):
            # Top-left corner of this window in input DRAM
            window_src = INPUT_DRAM_ADDR + (wh * window_size * W + ww * window_size) * C * bytes_per_element

            # Strided read: gather window_size rows, each window_row_bytes long,
            # spaced input_row_stride apart in DRAM -> contiguous in SRAM
            ue.accelerator_memory_to_sram(
                accelerator_dram_address=window_src,
                sram_address=0x00000,
                element_size=window_elements,
                stride_bytes_per_chunk=window_row_bytes,
                stride_jump_bytes=input_row_stride,
            )
            # Contiguous write to output
            ue.sram_to_accelerator_memory(
                sram_address=0x00000,
                accelerator_dram_address=OUTPUT_DRAM_ADDR + output_offset,
                element_size=window_elements,
            )
            output_offset += window_elements * bytes_per_element
def window_reverse_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                        H: int, W: int, C: int, window_size: int) -> None:
    """Rearrange (num_windows, window_size*window_size, C) -> (H, W, C) in DRAM.

    Inverse of window_partition_dram. Reads each window contiguously from input,
    then strided-writes rows back to their spatial positions.
    """
    bytes_per_element = 2
    num_windows_h = H // window_size
    num_windows_w = W // window_size
    window_row_bytes = window_size * C * bytes_per_element
    output_row_stride = W * C * bytes_per_element
    window_elements = window_size * window_size * C

    input_offset = 0
    for wh in range(num_windows_h):
        for ww in range(num_windows_w):
            # Top-left corner of this window in output DRAM
            window_dst = OUTPUT_DRAM_ADDR + (wh * window_size * W + ww * window_size) * C * bytes_per_element

            # Contiguous read of full window from input
            ue.accelerator_memory_to_sram(
                accelerator_dram_address=INPUT_DRAM_ADDR + input_offset,
                sram_address=0x00000,
                element_size=window_elements,
            )
            # Strided write: scatter window_size rows, each window_row_bytes long,
            # spaced output_row_stride apart in DRAM
            ue.sram_to_accelerator_memory(
                sram_address=0x00000,
                accelerator_dram_address=window_dst,
                element_size=window_elements,
                stride_bytes_per_chunk=window_row_bytes,
                stride_jump_bytes=output_row_stride,
            )
            input_offset += window_elements * bytes_per_element
def cyclic_shift_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                      H: int, W: int, C: int, shift_h: int, shift_w: int) -> None:
    """Cyclic shift (torch.roll) of (H, W, C) tensor in DRAM.

    Shifts by (-shift_h, -shift_w): rows [shift_h:] go first, then rows [:shift_h].
    Within each row, columns [shift_w:] go first, then columns [:shift_w].

    Implemented as two contiguous copies per row (right part, then left part).
    """
    bytes_per_element = 2
    row_bytes = W * C * bytes_per_element
    right_cols = W - shift_w
    left_cols = shift_w
    right_bytes = right_cols * C * bytes_per_element
    left_bytes = left_cols * C * bytes_per_element

    for dst_row in range(H):
        src_row = (dst_row + shift_h) % H
        src_row_addr = INPUT_DRAM_ADDR + src_row * row_bytes

        dst_row_addr = OUTPUT_DRAM_ADDR + dst_row * row_bytes

        # Copy columns [shift_w:] -> columns [0:right_cols]
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=src_row_addr + shift_w * C * bytes_per_element,
            sram_address=0x00000,
            element_size=right_cols * C,
        )
        ue.sram_to_accelerator_memory(
            sram_address=0x00000,
            accelerator_dram_address=dst_row_addr,
            element_size=right_cols * C,
        )

        # Copy columns [0:shift_w] -> columns [right_cols:]
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=src_row_addr,
            sram_address=0x00000,
            element_size=left_cols * C,
        )
        ue.sram_to_accelerator_memory(
            sram_address=0x00000,
            accelerator_dram_address=dst_row_addr + right_bytes,
            element_size=left_cols * C,
        )
def patch_merging_gather_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                              H: int, W: int, C: int) -> None:
    """Gather 2x2 spatial neighbors and concatenate along C.

    Input:  (H, W, C) in DRAM
    Output: (H/2, W/2, 4*C) in DRAM

    For each 2x2 block, the 4 elements are concatenated:
      [x[0::2,0::2], x[1::2,0::2], x[0::2,1::2], x[1::2,1::2]] along last dim.
    """
    bytes_per_element = 2
    H2 = H // 2
    W2 = W // 2
    input_row_bytes = W * C * bytes_per_element
    output_row_bytes = W2 * 4 * C * bytes_per_element

    # Four quadrants: (row_offset, col_offset) within each 2x2 block
    quadrants = [(0, 0), (1, 0), (0, 1), (1, 1)]

    for out_row in range(H2):
        for q_idx, (dr, dc) in enumerate(quadrants):
            src_row = out_row * 2 + dr
            # Gather every other column from this row: columns dc, dc+2, dc+4, ...
            # Each element is C wide. Stride: read C elements, skip 2*C elements.
            src_addr = INPUT_DRAM_ADDR + src_row * input_row_bytes + dc * C * bytes_per_element

            ue.accelerator_memory_to_sram(
                accelerator_dram_address=src_addr,
                sram_address=0x00000,
                element_size=W2 * C,
                stride_bytes_per_chunk=C * bytes_per_element,
                stride_jump_bytes=2 * C * bytes_per_element,
            )

            # Write into output: each output position is 4*C wide, this quadrant fills
            # a C-sized slice at offset q_idx*C within each 4*C output element.
            # Output for this row: (W2, 4*C), we need to scatter C elements every 4*C.
            dst_addr = OUTPUT_DRAM_ADDR + out_row * output_row_bytes + q_idx * C * bytes_per_element

            ue.sram_to_accelerator_memory(
                sram_address=0x00000,
                accelerator_dram_address=dst_addr,
                element_size=W2 * C,
                stride_bytes_per_chunk=C * bytes_per_element,
                stride_jump_bytes=4 * C * bytes_per_element,
            )
def dram_zero_fill(ue: UnifiedEngine, DRAM_ADDR: int, num_elements: int) -> None:
    """Fill a DRAM region with zeros using SRAM as staging buffer."""
    from user_dma_core import URAM_NEAR_FULL_ELEMENTS, URAM_SECTION, URAM_START_ADDR, MEMCPY_TYPE
    bytes_per_element = 2

    # Write zeros to SRAM once (full URAM_A)
    chunk_elems = min(URAM_NEAR_FULL_ELEMENTS, num_elements)
    zeros = torch.zeros(chunk_elems, dtype=torch.bfloat16)
    zeros_dram = ue.get_params_dram_addr()
    ue.allocate_params_dram(chunk_elems * bytes_per_element)
    ue.dma_write(DMA_DEVICE_H2C, zeros_dram, zeros, chunk_elems * bytes_per_element)
    ue.accelerator_memory_to_sram(
        accelerator_dram_address=zeros_dram,
        sram_address=0x00000,
        element_size=chunk_elems,
    )

    # Write zeros from SRAM to destination DRAM in chunks
    offset = 0
    while offset < num_elements:
        take = min(chunk_elems, num_elements - offset)
        ue.sram_to_accelerator_memory(
            sram_address=0x00000,
            accelerator_dram_address=DRAM_ADDR + offset * bytes_per_element,
            element_size=take,
        )
        offset += take


def multihead_pad_and_permute(ue: UnifiedEngine,
                              INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                              TEMP_DRAM_ADDR: int,
                              num_windows: int, wa: int, num_heads: int,
                              head_dim: int, head_dim_pad: int, wa_pad: int) -> None:
    """Reshape Q/K/V from projection layout to multi-head attention layout.

    Input:  (num_windows * wa, num_heads * head_dim) in INPUT_DRAM_ADDR
            i.e. (num_windows, wa, num_heads, head_dim) row-major
    Output: (num_windows * num_heads, wa_pad, head_dim_pad) in OUTPUT_DRAM_ADDR
            i.e. per-head, per-window attention blocks with padding

    Steps per window:
      1. Zero-fill output region (already done before calling this)
      2. Strided copy: (wa, num_heads * head_dim) -> (wa, num_heads, head_dim_pad)
         in TEMP buffer — scatter head_dim real elements into head_dim_pad slots
      3. bf16_permute_core: (wa, num_heads, head_dim_pad) -> (num_heads, wa, head_dim_pad)
      4. Copy permuted data into output with wa_pad stride

    TEMP_DRAM_ADDR must hold at least (wa * num_heads * head_dim_pad) elements.
    OUTPUT_DRAM_ADDR must be pre-zeroed (call dram_zero_fill first).
    """
    bytes_per_element = 2
    dim = num_heads * head_dim
    input_window_stride = wa * dim * bytes_per_element
    temp_row_width = num_heads * head_dim_pad  # elements per row in padded layout
    output_head_stride = wa_pad * head_dim_pad * bytes_per_element  # per (window, head) block

    # Allocate a single zeros row in params DRAM for padding
    zeros_addr = ue.get_params_dram_addr()
    ue.allocate_params_dram(UE_VECTOR_SIZE * bytes_per_element)
    ue.dma_write(DMA_DEVICE_H2C, zeros_addr,
                 torch.zeros(UE_VECTOR_SIZE, dtype=torch.bfloat16),
                 UE_VECTOR_SIZE * bytes_per_element)

    for w in range(num_windows):
        input_base = INPUT_DRAM_ADDR + w * input_window_stride

        # Step 2: Zero-fill TEMP for this window, then scatter real data
        dram_zero_fill(ue, TEMP_DRAM_ADDR, wa * temp_row_width)

        for h in range(num_heads):
            for row in range(wa):
                src = input_base + (row * dim + h * head_dim) * bytes_per_element
                dst = TEMP_DRAM_ADDR + (row * temp_row_width + h * head_dim_pad) * bytes_per_element

                # Zero the SRAM row first
                ue.accelerator_memory_to_sram(
                    accelerator_dram_address=zeros_addr,
                    sram_address=0x00000,
                    element_size=UE_VECTOR_SIZE,
                )
                # Load 32 real elements into first half of the zeroed row
                ue.accelerator_memory_to_sram(
                    accelerator_dram_address=src,
                    sram_address=0x00000,
                    element_size=head_dim,
                )
                # Write full 64 elements (32 real + 32 zeros) to TEMP
                ue.sram_to_accelerator_memory(
                    sram_address=0x00000,
                    accelerator_dram_address=dst,
                    element_size=UE_VECTOR_SIZE,
                )

        # Step 3: Permute (wa, num_heads, head_dim_pad) -> (num_heads, wa, head_dim_pad)
        # Output goes to a temp region, then we copy to final output with wa_pad stride
        permute_out = TEMP_DRAM_ADDR + wa * temp_row_width * bytes_per_element  # second half of temp
        ue.bf16_permute_core(
            dim_0=wa, dim_1=num_heads, dim_2=head_dim_pad,
            INPUT_DRAM_ADDR=TEMP_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=permute_out,
        )

        # Step 4: Copy each head's (wa, head_dim_pad) block into output at (wa_pad, head_dim_pad)
        # with zero-padding rows wa..wa_pad-1 (already zeroed)
        for h in range(num_heads):
            src_head = permute_out + h * wa * head_dim_pad * bytes_per_element
            dst_head = OUTPUT_DRAM_ADDR + (w * num_heads + h) * output_head_stride

            # Copy wa rows of head_dim_pad elements each
            ue.accelerator_memory_to_sram(
                accelerator_dram_address=src_head,
                sram_address=0x00000,
                element_size=wa * head_dim_pad,
            )
            ue.sram_to_accelerator_memory(
                sram_address=0x00000,
                accelerator_dram_address=dst_head,
                element_size=wa * head_dim_pad,
            )


def build_avg_pool_weight(seq_len: int, seq_len_pad: int, N_pad: int = 64) -> torch.Tensor:
    """Build the averaging weight for adaptive avg pool via matmul.

    Returns a (N_pad, seq_len_pad) bf16 tensor with 1/seq_len in the first row
    for the real seq_len positions. Used as B in:
        matmat_mul_core(M=C_pad, K=seq_len_pad, N=N_pad, A=transposed_input, B=avg_weight)

    The transpose (seq_len, C) -> (C, seq_len_pad) must be done before the matmul
    using bf16_transpose_core.
    """
    W = torch.zeros(N_pad, seq_len_pad, dtype=torch.bfloat16)
    W[0, :seq_len] = 1.0 / seq_len
    return W
# ---------------------------------------------------------------------------
# Config / helpers
# ---------------------------------------------------------------------------
def _parse_offset(val) -> int:
    """Parse offset/size from JSON: int or hex string like '0x24000000'."""
    if isinstance(val, str):
        return int(val, 0)
    return int(val)
# ---------------------------------------------------------------------------
# Weight binary generation
# ---------------------------------------------------------------------------
def _ensure_hf_model(script_dir: str, cfg: dict):
    """Download HF model if not present. Returns (model, model_dir)."""
    paths = cfg["paths"]
    model_dir = os.path.join(script_dir, paths["hf_model_dir"])
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        snapshot_download(repo_id=paths["hf_model_repo"], local_dir=model_dir,
                          ignore_patterns=["*.h5", "*.ot", "*.msgpack"])
    model = AutoModelForImageClassification.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map=None)
    model.eval()
    return model, model_dir
def weight_bin_generate(output_path: str | None = None, config_path: str | None = None) -> str:
    """Generate weights_swin_hf.bin from HuggingFace model per swin_config.json layout.
    Returns the path to the written file."""
    cfg = Swin_UnifiedEngine.load_config(config_path=config_path, script_dir=SCRIPT_DIR)
    paths = cfg["paths"]
    out_path = output_path or os.path.join(SCRIPT_DIR, paths["weights_bin"])
    model, model_dir = _ensure_hf_model(SCRIPT_DIR, cfg)

    # TODO: extract and pack weights per config layout
    # - patch embedding: projection weight (192, 48) + bias (192,) + layernorm
    # - per stage, per layer:
    #     - attention: Q/K/V weights+biases, output projection weight+bias
    #     - relative position bias table
    #     - MLP: expand weight+bias (dim, dim*4), contract weight+bias (dim*4, dim)
    #     - layernorm: 2x gamma+beta per layer
    # - patch merging (3x): norm gamma+beta, reduction weight
    # - final layernorm gamma+beta
    # - classifier weight+bias
    # - unpad weights (constructed, not from HF): one per stage

    raise NotImplementedError("weight_bin_generate")


# ---------------------------------------------------------------------------
# Swin Unified Engine
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# DRAM partition over 4 GB address space:
#   2 GB params  /  1 GB activations  /  1 GB programs
# ---------------------------------------------------------------------------
SWIN_PARAMS_BASE  = 0x00000000   # 0.0 GB — 1 GB for weights, biases, norms, unpad/pool weights
SWIN_TENSOR_BASE  = 0x40000000   # 1.0 GB — 1 GB for intermediate activations, scratch buffers
SWIN_PROGRAM_BASE = 0x80000000   # 2.0 GB — 2 GB for compiled instruction programs


class Swin_UnifiedEngine(UnifiedEngine):
    """Swin-Large on Unified Engine FPGA accelerator."""

    # --- Architecture constants (Swin-Large patch4-window12-384-in22k) ---
    IMAGE_SIZE      = 384
    PATCH_SIZE      = 4
    NUM_CHANNELS    = 3
    EMBED_DIM       = 192
    DEPTHS          = [2, 2, 18, 2]          # layers per stage
    NUM_HEADS       = [6, 12, 24, 48]        # heads per stage
    WINDOW_SIZE     = 12
    MLP_RATIO       = 4.0
    HEAD_DIM        = 32                      # constant across all stages
    NUM_STAGES      = 4
    GRID_SIZE       = IMAGE_SIZE // PATCH_SIZE  # 96

    # Per-stage derived dims
    STAGE_DIMS      = [192, 384, 768, 1536]
    STAGE_SPATIAL   = [96, 48, 24, 12]
    WINDOW_AREA     = 144

    def __init__(self, script_dir: str = SCRIPT_DIR, weights_bin: str | None = None):
        super().__init__(params_dram_base=SWIN_PARAMS_BASE,
                         tensor_dram_base=SWIN_TENSOR_BASE,
                         program_dram_base=SWIN_PROGRAM_BASE)
        cfg = self.load_config(script_dir=script_dir)
        self.cfg = cfg
        self.script_dir = script_dir
        self.weights_bin = weights_bin or os.path.join(script_dir, cfg["paths"]["weights_bin"])

        self.weight_init()
        self.tensor_init()

    @staticmethod
    def load_config(config_path: str | None = None, script_dir: str | None = None) -> dict:
        """Load swin_config.json, resolve offsets into _weight_defs."""
        sd = script_dir or SCRIPT_DIR
        cp = config_path or os.path.join(sd, "swin_config.json")
        with open(cp) as f:
            cfg = json.load(f)

        # TODO: build _weight_defs from regions / non_layer_regions
        cfg["_weight_defs"] = {}
        return cfg

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    ALIGN = 64  # UE_VECTOR_SIZE

    @staticmethod
    def pad_dim(x: int) -> int:
        return ((x + 63) // 64) * 64

    DMA_CHUNK_BYTES = 1 * 1024 * 1024  # 1 MB chunks for large DMA writes

    def write_captured_instructions_to_dram(self, start_addr: int = DRAM_INSTRUCTION_ADDR) -> int:
        """Chunked override to avoid segfault on large instruction DMA writes."""
        if not self.capture_buffer or self.capture_count == 0:
            return 0

        total_bytes = self.capture_count * 32
        all_bytes = bytearray()
        for inst in self.capture_buffer:
            all_bytes.extend(inst.get_bytes())
        data = bytes(all_bytes)

        print(f"Writing {self.capture_count} captured instructions ({total_bytes} bytes) to DRAM at 0x{start_addr:x}...")
        offset = 0
        while offset < total_bytes:
            chunk = min(self.DMA_CHUNK_BYTES, total_bytes - offset)
            self.dma_write(DMA_DEVICE_H2C, start_addr + offset, data[offset:offset + chunk], chunk)
            offset += chunk
        print(f"Successfully wrote {total_bytes} bytes ({self.capture_count} instructions) to DRAM")
        return total_bytes

    def _alloc_param(self, tensor: torch.Tensor) -> int:
        """Allocate params DRAM, write bf16 tensor. Returns DRAM address."""
        t = tensor.to(torch.bfloat16).contiguous()
        addr = self.get_params_dram_addr()
        nbytes = t.numel() * 2
        self.allocate_params_dram(nbytes)
        self.dma_to_accelerator_memory(addr, t)
        return addr

    def _alloc_tensor(self, num_elements: int) -> int:
        """Allocate tensor DRAM buffer. Returns DRAM address."""
        return self.allocate_tensor_dram(num_elements * 2)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def weight_init(self) -> None:
        """Load HF model weights to DRAM."""
        model, model_dir = _ensure_hf_model(self.script_dir, self.cfg)
        swin = model.swin  # SwinModel inside SwinForImageClassification

        bpe = 2
        pad = self.pad_dim

        # ---- Pre-encoder: patch embedding via bf16_patching_core ----
        patch_proj = swin.embeddings.patch_embeddings.projection
        P = self.PATCH_SIZE       # 4
        C = self.NUM_CHANNELS     # 3
        N = self.EMBED_DIM        # 192
        patches_per_group = UE_VECTOR_SIZE // P  # 16
        K_patch = C * P * UE_VECTOR_SIZE          # 768 (weight inner dim)

        # Original weight: (N, C*P*P) = (192, 48)
        w = patch_proj.weight.data.to(torch.bfloat16)
        if w.dim() == 4:
            w = w.reshape(N, -1)  # (192, 48)

        # Build 16 sparse weight matrices, one per patch position in group.
        # Each is (N, K_patch) = (192, 768) bf16.
        # For patch p, the real data in each URAM row is at columns [p*P : p*P+P].
        # So weight_p[n, row*64 + p*P + pw] = original_w[n, row*P + pw]
        # where row = c*patch_h + ph, row*P + pw indexes into the 48-element patch.
        self.PATCH_EMBED_WEIGHT_ADDRS = []
        for p in range(patches_per_group):
            w_sparse = torch.zeros(N, K_patch, dtype=torch.bfloat16)
            for row in range(C * P):  # 12 URAM rows
                for pw in range(P):   # 4 elements per row per patch
                    # Source: original weight column = row * P + pw (0..47)
                    # Dest: sparse weight column = row * 64 + p * P + pw
                    w_sparse[:, row * UE_VECTOR_SIZE + p * P + pw] = w[:, row * P + pw]
            self.PATCH_EMBED_WEIGHT_ADDRS.append(self._alloc_param(w_sparse))

        # Bias: added after patching via matmul with bias, or separate broadcast
        # For now store it — we'll add it after patching_core output
        b_padded = torch.zeros(N, dtype=torch.bfloat16)
        if patch_proj.bias is not None:
            b_padded[:N] = patch_proj.bias.data.to(torch.bfloat16)
        self.PATCH_EMBED_BIAS = self._alloc_param(b_padded)

        # Patch embedding LayerNorm
        ln = swin.embeddings.norm
        self.PATCH_EMBED_LN_GAMMA = self._alloc_param(ln.weight.data)
        self.PATCH_EMBED_LN_BETA = self._alloc_param(ln.bias.data)

        # ---- Encoder: per-stage, per-layer weights ----
        # self.encoder_weights[stage][layer] = dict of DRAM addresses
        self.encoder_weights = []
        for s in range(self.NUM_STAGES):
            dim = self.STAGE_DIMS[s]
            num_heads = self.NUM_HEADS[s]
            head_dim = self.HEAD_DIM
            ws = self.WINDOW_SIZE
            window_area = ws * ws  # 144
            window_area_pad = pad(window_area)  # 192
            head_dim_pad = pad(head_dim)  # 64
            mlp_dim = int(dim * self.MLP_RATIO)

            stage_weights = []
            hf_stage = swin.encoder.layers[s]
            for l in range(self.DEPTHS[s]):
                block = hf_stage.blocks[l]
                lw = {}

                # Pre-attention LayerNorm
                lw['ln_before_gamma'] = self._alloc_param(block.layernorm_before.weight.data)
                lw['ln_before_beta'] = self._alloc_param(block.layernorm_before.bias.data)

                # Q/K/V weights + biases (dim, dim) — already 64-aligned for all stages
                attn = block.attention.self
                for name, proj in [('q', attn.query), ('k', attn.key), ('v', attn.value)]:
                    lw[f'{name}_weight'] = self._alloc_param(proj.weight.data)
                    lw[f'{name}_bias'] = self._alloc_param(proj.bias.data)

                # Relative position bias: precompute (num_heads, window_area, window_area)
                # then pad to (num_heads, window_area_pad, window_area_pad)
                # Tile for all windows: (num_windows * num_heads, wa_pad, wa_pad)
                # Layout matches flash_attention_batched: batch b = (w * num_heads + h)
                # gets bias at offset b * wa_pad * wa_pad. Since all windows share
                # the same per-head bias, we tile num_windows copies.
                rel_pos_bias_table = attn.relative_position_bias_table  # (529, num_heads)
                rel_pos_index = attn.relative_position_index  # (144, 144)
                rpb = rel_pos_bias_table[rel_pos_index.view(-1)].view(window_area, window_area, num_heads)
                rpb = rpb.permute(2, 0, 1).contiguous().to(torch.bfloat16)  # (num_heads, 144, 144)
                # Pad to (num_heads, 192, 192) with -100 in padding positions
                # so padded key positions get exp(-100) ≈ 0 after softmax
                rpb_padded = torch.full((num_heads, window_area_pad, window_area_pad), -100.0, dtype=torch.bfloat16)
                rpb_padded[:, :window_area, :window_area] = rpb
                # Tile: (num_windows * num_heads, wa_pad, wa_pad)
                nw = (self.STAGE_SPATIAL[s] // self.WINDOW_SIZE) ** 2
                rpb_tiled = rpb_padded.unsqueeze(0).expand(nw, -1, -1, -1).reshape(
                    nw * num_heads, window_area_pad, window_area_pad).contiguous()
                lw['rel_pos_bias'] = self._alloc_param(rpb_tiled)

                # Output projection (attention)
                out_proj = block.attention.output.dense
                lw['out_proj_weight'] = self._alloc_param(out_proj.weight.data)
                lw['out_proj_bias'] = self._alloc_param(out_proj.bias.data)

                # Pre-MLP LayerNorm
                lw['ln_after_gamma'] = self._alloc_param(block.layernorm_after.weight.data)
                lw['ln_after_beta'] = self._alloc_param(block.layernorm_after.bias.data)

                # MLP expand: (mlp_dim, dim)
                lw['mlp_expand_weight'] = self._alloc_param(block.intermediate.dense.weight.data)
                lw['mlp_expand_bias'] = self._alloc_param(block.intermediate.dense.bias.data)

                # MLP contract: (dim, mlp_dim)
                lw['mlp_contract_weight'] = self._alloc_param(block.output.dense.weight.data)
                lw['mlp_contract_bias'] = self._alloc_param(block.output.dense.bias.data)

                stage_weights.append(lw)
            self.encoder_weights.append(stage_weights)

        # ---- Encoder: unpad weights (constructed, one per stage) ----
        self.unpad_weight_addrs = []
        for s in range(self.NUM_STAGES):
            w = self.build_unpad_weight(self.NUM_HEADS[s], self.HEAD_DIM, pad(self.HEAD_DIM))
            self.unpad_weight_addrs.append(self._alloc_param(w))

        # ---- Encoder: patch merging weights ----
        # TODO: load norm gamma+beta, reduction weight for 3 merges

        # ---- Post-encoder ----
        # TODO: final layernorm, avg pool weight, classifier weight+bias

        del model  # free CPU memory

    def tensor_init(self) -> None:
        """Allocate intermediate DRAM buffers for forward pass."""
        bpe = 2
        pad = self.pad_dim

        # ---- Pre-encoder tensors ----
        M_patches = self.GRID_SIZE * self.GRID_SIZE  # 9216
        N_embed = self.EMBED_DIM  # 192
        C = self.NUM_CHANNELS
        H = W = self.IMAGE_SIZE  # 384

        self.IMAGE_DRAM = self._alloc_tensor(C * H * W)           # CHW input image
        self.PATCH_OUTPUT_DRAM = self._alloc_tensor(M_patches * N_embed)  # patching_core output
        self.EMBED_OUTPUT_DRAM = self._alloc_tensor(M_patches * N_embed)  # after LN

        # ---- Encoder tensors (per-stage scratch, reused across layers) ----
        self.stage_tensors = []
        for s in range(self.NUM_STAGES):
            dim = self.STAGE_DIMS[s]
            spatial = self.STAGE_SPATIAL[s]
            num_heads = self.NUM_HEADS[s]
            num_windows = (spatial // self.WINDOW_SIZE) ** 2
            wa = self.WINDOW_AREA         # 144
            wa_pad = pad(wa)              # 192
            hd_pad = pad(self.HEAD_DIM)   # 64
            mlp_dim = int(dim * self.MLP_RATIO)
            M = spatial * spatial   # total tokens
            total_batches = num_windows * num_heads

            st = {}
            # Layer input/output (spatial layout, reinterpreted as needed)
            st['layer_input'] = self._alloc_tensor(M * dim)
            st['ln_output'] = self._alloc_tensor(M * dim)

            # Windowed layout
            st['windowed'] = self._alloc_tensor(num_windows * wa * dim)
            st['shifted'] = self._alloc_tensor(M * dim)  # for cyclic shift

            # Q/K/V after projection (windowed flat)
            st['q_proj'] = self._alloc_tensor(num_windows * wa * dim)
            st['k_proj'] = self._alloc_tensor(num_windows * wa * dim)
            st['v_proj'] = self._alloc_tensor(num_windows * wa * dim)

            # Q/K/V after permute to multi-head: (total_batches, wa_pad, hd_pad)
            st['q_heads'] = self._alloc_tensor(total_batches * wa_pad * hd_pad)
            st['k_heads'] = self._alloc_tensor(total_batches * wa_pad * hd_pad)
            st['v_heads'] = self._alloc_tensor(total_batches * wa_pad * hd_pad)

            # Attention output: (total_batches, wa_pad, hd_pad)
            st['attn_output'] = self._alloc_tensor(total_batches * wa_pad * hd_pad)
            # Scratch for flash attention: per batch needs hd*seq + seq*seq
            scratch_per_batch = hd_pad * wa_pad + wa_pad * wa_pad
            st['attn_scratch'] = self._alloc_tensor(total_batches * scratch_per_batch)

            # After inverse permute: (num_windows * wa, num_heads * hd_pad)
            st['attn_permuted'] = self._alloc_tensor(num_windows * wa * num_heads * hd_pad)

            # Temp buffer for multihead_pad_and_permute: needs 2 * (wa * num_heads * hd_pad) for pad + permute
            st['permute_temp'] = self._alloc_tensor(2 * wa * num_heads * hd_pad)

            # After unpad: (num_windows * wa, dim)
            st['attn_unpadded'] = self._alloc_tensor(num_windows * wa * dim)

            # Output projection result
            st['out_proj'] = self._alloc_tensor(num_windows * wa * dim)

            # Residual / MLP
            st['residual1'] = self._alloc_tensor(M * dim)
            st['mlp_ln'] = self._alloc_tensor(M * dim)
            st['mlp_mid'] = self._alloc_tensor(M * mlp_dim)
            st['mlp_out'] = self._alloc_tensor(M * dim)

            self.stage_tensors.append(st)

        # ---- Post-encoder tensors ----
        # TODO: final norm output, transpose buffer, pool output, logits

    # ------------------------------------------------------------------
    # Unpad weight construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_unpad_weight(num_heads: int, head_dim: int = 32, head_dim_pad: int = 64) -> torch.Tensor:
        """Build the unpad selection matrix for a given stage.

        After attention, output is (M, num_heads * head_dim_pad). We need
        (M, num_heads * head_dim). This weight W has shape
        (num_heads * head_dim_pad, num_heads * head_dim) with identity blocks
        at the real-data positions and zeros at padding positions.

        matmul(padded_output, W) -> unpadded_output
        """
        K = num_heads * head_dim_pad   # input  dim (padded)
        N = num_heads * head_dim       # output dim (real)
        W = torch.zeros(K, N, dtype=torch.bfloat16)
        for h in range(num_heads):
            for d in range(head_dim):
                W[h * head_dim_pad + d, h * head_dim + d] = 1.0
        return W

    # ------------------------------------------------------------------
    # Program compilation
    # ------------------------------------------------------------------

    def compile_pre_encoder(self) -> int:
        """Compile everything before the encoder layers.

        Ops:
          1. bf16_patching_core: extract patches from CHW image + bf16 projection -> (9216, 192)
          2. Bias add (broadcast per row)
          3. LayerNorm(192)

        Input:  (3, 384, 384) CHW image in IMAGE_DRAM
        Output: (9216, 192) embedded + normalized in EMBED_OUTPUT_DRAM

        Returns program DRAM address.
        """
        M = self.GRID_SIZE * self.GRID_SIZE  # 9216
        N = self.EMBED_DIM  # 192
        C = self.NUM_CHANNELS
        H = W = self.IMAGE_SIZE
        P = self.PATCH_SIZE
        K_patch = C * P * UE_VECTOR_SIZE  # 768

        self.start_capture()

        # 1. Patch extraction + linear projection (fused, on device)
        bf16_patching_core(
            self,
            INPUT_DRAM_ADDR=self.IMAGE_DRAM,
            OUTPUT_DRAM_ADDR=self.PATCH_OUTPUT_DRAM,
            weight_dram_addrs=self.PATCH_EMBED_WEIGHT_ADDRS,
            C=C, H=H, W=W,
            patch_h=P, patch_w=P,
            K=K_patch, N=N,
        )

        # 2. Add bias: output += bias (broadcast N across M rows)
        # Use matmat_mul with identity-like approach, or just add bias via
        # loading bias to SRAM and broadcasting. Simplest: use the EMBED_OUTPUT
        # buffer and add bias row by row.
        # Actually, we can fold bias into LayerNorm: LN(x + b) != LN(x) + b,
        # so we must add bias before LN.
        # Use eltwise add: load each chunk of output + bias into SRAM, add, write back.
        # Bias is (N=192,) broadcast to each row.
        # Load bias to URAM_B once, then chunk rows through URAM_A.
        self.accelerator_memory_to_sram(
            accelerator_dram_address=self.PATCH_EMBED_BIAS,
            sram_address=0x80000,  # URAM_B
            element_size=N,
        )
        chunk_size = min(URAM_NEAR_FULL_ELEMENTS // N, M)
        for i, m_take in self.chunk_ranges(M, chunk_size):
            self.accelerator_memory_to_sram(
                accelerator_dram_address=self.PATCH_OUTPUT_DRAM + i * N * 2,
                sram_address=0x00000,
                element_size=m_take * N,
            )
            # Add bias to each row: for row j, add bias vector
            for j in range(m_take):
                self.eltwise_add_core(
                    vector_A_sram_start_addr=0x00000 + j * N * 2,
                    vector_B_sram_start_addr=0x80000,
                    vector_C_sram_wb_addr=0x00000 + j * N * 2,
                    element_size=N,
                )
            self.sram_to_accelerator_memory(
                sram_address=0x00000,
                accelerator_dram_address=self.PATCH_OUTPUT_DRAM + i * N * 2,
                element_size=m_take * N,
            )

        # 3. LayerNorm
        self.layer_norm_core_dram(
            M=M, N=N,
            A_DRAM_ADDR=self.PATCH_OUTPUT_DRAM,
            OUTPUT_DRAM_ADDR=self.EMBED_OUTPUT_DRAM,
            GAMMA_DRAM_ADDR=self.PATCH_EMBED_LN_GAMMA,
            BETA_DRAM_ADDR=self.PATCH_EMBED_LN_BETA,
        )

        self.stop_capture()
        self.generate_instruction_halt()

        prog_addr = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(prog_addr)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        self.clear_capture_buffer()

        return prog_addr

    def compile_encoder(self) -> list[int]:
        """Compile all encoder layers across all 4 stages.

        For each stage (s=0..3):
          dim = STAGE_DIMS[s], spatial = STAGE_SPATIAL[s], depth = DEPTHS[s]
          num_heads = NUM_HEADS[s], num_windows = (spatial/WINDOW_SIZE)^2

          For each layer (l=0..depth-1):
            shift_size = 0 if l is even, else WINDOW_SIZE//2

            Pre-attention memory ops:
              - cyclic_shift_dram (if shift_size > 0)
              - window_partition_dram: (spatial, spatial, dim) -> (num_windows, 144, dim)

            Attention compute:
              - layer_norm_core_dram (pre-attention)
              - matmat_mul_core x3: Q, K, V projections (M=num_windows*144, K=dim, N=dim) + bias
              - bf16_permute_core x3: reshape to multi-head (num_windows, 144, num_heads, 64)
                                      -> (num_windows*num_heads, 144, 64)
              - flash_attention_batched: num_windows*num_heads batches, head_dim=64, seq_len=192
              - bf16_permute_core: inverse permute back to (num_windows, 144, num_heads*64)
              - matmat_mul_core: unpad (num_heads*64 -> num_heads*32 = dim)
              - matmat_mul_core: output projection (M, dim, dim) + bias

            Post-attention:
              - window_reverse_dram: (num_windows, 144, dim) -> (spatial, spatial, dim)
              - cyclic_shift_dram (reverse, if shift_size > 0)
              - eltwise_add_core: residual 1
              - layer_norm_core_dram (pre-MLP)
              - matmat_mul_core: MLP expand (M, dim, dim*4) + bias + GELU
              - matmat_mul_core: MLP contract (M, dim*4, dim) + bias
              - eltwise_add_core: residual 2

          After all layers in stage (except stage 3):
            Patch merging:
              - patch_merging_gather_dram: (spatial, spatial, dim) -> (spatial/2, spatial/2, 4*dim)
              - layer_norm_core_dram(M=spatial/2*spatial/2, N=4*dim)
              - matmat_mul_core(M=spatial/2*spatial/2, K=4*dim, N=2*dim) (no bias)

        Returns list of program DRAM addresses (one per layer + one per merge).
        """
        pad = self.pad_dim
        program_addrs = []

        for s in range(self.NUM_STAGES):
            dim = self.STAGE_DIMS[s]
            spatial = self.STAGE_SPATIAL[s]
            num_heads = self.NUM_HEADS[s]
            num_windows = (spatial // self.WINDOW_SIZE) ** 2
            wa = self.WINDOW_AREA         # 144
            wa_pad = pad(wa)              # 192
            hd_pad = pad(self.HEAD_DIM)   # 64
            mlp_dim = int(dim * self.MLP_RATIO)
            M = spatial * spatial
            total_batches = num_windows * num_heads
            st = self.stage_tensors[s]

            for l in range(min(1, self.DEPTHS[s])):  # TODO: all layers. For now, layer 0 only.
                lw = self.encoder_weights[s][l]
                shift_size = 0 if l % 2 == 0 else self.WINDOW_SIZE // 2

                self.start_capture()

                # --- 1. Pre-attention LayerNorm ---
                # For first layer of first stage, read directly from embed output
                layer_input_addr = self.EMBED_OUTPUT_DRAM if (s == 0 and l == 0) else st['layer_input']
                self.layer_norm_core_dram(
                    M=M, N=dim,
                    A_DRAM_ADDR=layer_input_addr,
                    OUTPUT_DRAM_ADDR=st['ln_output'],
                    GAMMA_DRAM_ADDR=lw['ln_before_gamma'],
                    BETA_DRAM_ADDR=lw['ln_before_beta'],
                )

                # --- 2. Cyclic shift (odd layers only) ---
                if shift_size > 0:
                    cyclic_shift_dram(self,
                        INPUT_DRAM_ADDR=st['ln_output'],
                        OUTPUT_DRAM_ADDR=st['shifted'],
                        H=spatial, W=spatial, C=dim,
                        shift_h=shift_size, shift_w=shift_size)
                    partition_src = st['shifted']
                else:
                    partition_src = st['ln_output']

                # --- 3. Window partition ---
                window_partition_dram(self,
                    INPUT_DRAM_ADDR=partition_src,
                    OUTPUT_DRAM_ADDR=st['windowed'],
                    H=spatial, W=spatial, C=dim,
                    window_size=self.WINDOW_SIZE)

                # --- 4. Q/K/V projections ---
                M_win = num_windows * wa  # total windowed tokens
                for name, out_addr in [('q', st['q_proj']), ('k', st['k_proj']), ('v', st['v_proj'])]:
                    self.matmat_mul_core(
                        M=M_win, K=dim, N=dim,
                        A_DRAM_ADDR=st['windowed'],
                        B_DRAM_ADDR=lw[f'{name}_weight'],
                        OUTPUT_DRAM_ADDR=out_addr,
                        C_DRAM_ADDR=lw[f'{name}_bias'],
                        bias_mode="broadcast_N",
                    )

                # --- 5. Zero-fill + pad + permute Q/K/V to multi-head ---
                # (num_windows * wa, dim) -> (total_batches, wa_pad, hd_pad)
                head_dim = self.HEAD_DIM  # 32
                for proj_addr, heads_addr in [
                    (st['q_proj'], st['q_heads']),
                    (st['k_proj'], st['k_heads']),
                    (st['v_proj'], st['v_heads']),
                ]:
                    dram_zero_fill(self, heads_addr, total_batches * wa_pad * hd_pad)
                    multihead_pad_and_permute(self,
                        INPUT_DRAM_ADDR=proj_addr,
                        OUTPUT_DRAM_ADDR=heads_addr,
                        TEMP_DRAM_ADDR=st['permute_temp'],
                        num_windows=num_windows, wa=wa, num_heads=num_heads,
                        head_dim=head_dim, head_dim_pad=hd_pad, wa_pad=wa_pad)

                # --- 6. Flash attention batched ---
                dram_zero_fill(self, st['attn_output'], total_batches * wa_pad * hd_pad)
                flash_attention_batched(self,
                    num_batches=total_batches,
                    head_dim=hd_pad,       # 64
                    seq_len=wa_pad,        # 192
                    Q_DRAM_ADDR=st['q_heads'],
                    K_DRAM_ADDR=st['k_heads'],
                    V_DRAM_ADDR=st['v_heads'],
                    OUTPUT_DRAM_ADDR=st['attn_output'],
                    SCRATCH_DRAM_ADDR=st['attn_scratch'],
                    BIAS_DRAM_ADDR=lw['rel_pos_bias'],
                )

                # TODO: steps 7-17 (inverse permute, unpad, output proj, residuals, MLP)

                self.stop_capture()
                self.generate_instruction_halt()

                prog_addr = self.get_program_dram_addr()
                self.write_captured_instructions_to_dram(prog_addr)
                self.allocate_program_dram(self.get_capture_instruction_size_bytes())
                self.clear_capture_buffer()

                program_addrs.append(prog_addr)

        return program_addrs

    def compile_post_encoder(self) -> int:
        """Compile everything after the encoder layers.

        Ops:
          1. Final LayerNorm(1536)
          2. bf16_transpose_core(M=144, N=1536) -> (1536, 192 padded)
          3. matmat_mul_core(M=1536, K=192, N=64) with avg_pool_weight -> (1536, 64)
          4. matmat_mul_core(M=1, K=1536, N=21841) with classifier weight + bias

        Input:  (144, 1536) encoder output in DRAM
        Output: (1, 21841) logits in DRAM

        Returns program DRAM address.
        """
        # TODO: start_capture
        #   layer_norm_core_dram(M=144, N=1536, gamma, beta)
        #   bf16_transpose_core(M=144, N=1536)
        #   matmat_mul_core(M=1536, K=192, N=64, B=avg_pool_weight)
        #   matmat_mul_core(M=1, K=1536, N=21841, B=classifier_weight, C=classifier_bias)
        # stop_capture, write to program DRAM
        raise NotImplementedError("compile_post_encoder")

    def compile_forward(self) -> dict:
        """Compile full forward pass. Returns dict of program addresses."""
        pre_addr = self.compile_pre_encoder()
        encoder_addrs = self.compile_encoder()
        post_addr = self.compile_post_encoder()
        return {
            "pre_encoder": pre_addr,
            "encoder": encoder_addrs,
            "post_encoder": post_addr,
        }

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_pre_encoder(self, pixel_values: torch.Tensor, program_addr: int) -> None:
        """Run patch embedding + norm.

        1. DMA normalized CHW image (3, 384, 384) to device
        2. Execute pre_encoder program (patching + bias + layernorm, all on device)
        """
        # pixel_values: (1, 3, 384, 384) already normalized by preprocess_image
        image_chw = pixel_values.squeeze(0).to(torch.bfloat16)  # (3, 384, 384)
        self.dma_to_accelerator_memory(self.IMAGE_DRAM, image_chw.contiguous().flatten())
        self.start_execute_from_dram(program_addr)
        self.wait_queue()

    def run_encoder(self, program_addrs: list[int]) -> None:
        """Run encoder layer programs.

        For now: first layer reads directly from EMBED_OUTPUT_DRAM (compiled that way).
        """
        if program_addrs:
            self.start_execute_from_dram(program_addrs[0])
            self.wait_queue()

    def run_post_encoder(self, program_addr: int) -> torch.Tensor:
        """Run final norm + pool + classifier. Returns logits.

        1. Execute post_encoder program
        2. DMA logits back from device
        3. Return (1, num_labels) tensor
        """
        # TODO:
        #   start_execute_from_dram(program_addr)
        #   wait_queue()
        #   logits = dma_from_accelerator_memory(LOGITS_DRAM, (1, num_labels))
        #   return logits
        raise NotImplementedError("run_post_encoder")

    def run_forward(self, pixel_values: torch.Tensor, program_addrs: dict) -> torch.Tensor:
        """Run full forward pass on accelerator.

        Args:
            pixel_values: (1, 3, 384, 384) float32 input image tensor.
            program_addrs: dict from compile_forward().

        Returns:
            logits: (1, num_labels) bf16 tensor.
        """
        self.run_pre_encoder(pixel_values, program_addrs["pre_encoder"])
        self.run_encoder(program_addrs["encoder"])
        return self.run_post_encoder(program_addrs["post_encoder"])


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(image_path: str) -> torch.Tensor:
    """Load and preprocess image for Swin-Large-384. Returns (1, 3, 384, 384)."""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Swin-Large accelerator inference.")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--dev", type=str, default="xdma0", help="DMA device (default: xdma0)")
    parser.add_argument("--cycle", type=float, default=5.62, help="Clock cycle time in ns")
    args = parser.parse_args()

    set_dma_device(args.dev)
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle

    # Create engine (loads HF model weights to DRAM)
    ue = Swin_UnifiedEngine(script_dir=SCRIPT_DIR)

    # ---- Test pre_encoder against CPU reference ----
    print("=" * 60)
    print("Testing pre_encoder (patch embedding + layernorm)")
    print("=" * 60)

    # Load test image
    image_path = args.image or os.path.join(SCRIPT_DIR, "../../test_samples/vette.jpg")
    print(f"Using image: {image_path}")
    pixel_values = preprocess_image(image_path)

    # --- CPU reference: use HF model directly ---
    model, _ = _ensure_hf_model(SCRIPT_DIR, ue.cfg)
    swin = model.swin
    with torch.no_grad():
        # Run the actual HF patch embedding + layernorm
        embeddings_module = swin.embeddings
        # HF SwinEmbeddings.forward: patch_embeddings -> norm -> dropout
        ref_normed = embeddings_module(pixel_values.to(torch.bfloat16))[0]  # (1, 9216, 192)
        ref_normed = ref_normed.squeeze(0)  # (9216, 192)
    del model

    print(f"CPU reference: {ref_normed.shape}, range [{ref_normed.min():.4f}, {ref_normed.max():.4f}]")

    # --- Accelerator ---
    pre_addr = ue.compile_pre_encoder()
    ue.run_pre_encoder(pixel_values, pre_addr)

    # Read back result
    M = ue.GRID_SIZE * ue.GRID_SIZE  # 9216
    N = ue.EMBED_DIM  # 192
    hw_output = ue.dma_from_accelerator_memory(ue.EMBED_OUTPUT_DRAM, (M, N)).to(torch.bfloat16)

    print(f"HW output:     {hw_output.shape}, range [{hw_output.min():.4f}, {hw_output.max():.4f}]")

    # --- Compare ---
    diff = (ref_normed.float() - hw_output.float()).abs()
    mae = diff.mean().item()
    max_err = diff.max().item()

    # SNR
    signal_power = (ref_normed.float() ** 2).mean().item()
    noise_power = ((ref_normed.float() - hw_output.float()) ** 2).mean().item()
    snr_db = 10 * math.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

    print(f"MAE: {mae:.6f}")
    print(f"Max error: {max_err:.6f}")
    print(f"SNR: {snr_db:.2f} dB")

    if snr_db > 30:
        print("PASS: pre_encoder matches CPU reference")
    else:
        print("FAIL: pre_encoder does not match CPU reference")

    # ---- Test encoder layer 0 (stage 0): LN + window partition + QKV ----
    print()
    print("=" * 60)
    print("Testing encoder stage 0, layer 0 (LN + window partition + QKV)")
    print("=" * 60)

    # CPU reference for stage 0, layer 0 partial
    model2, _ = _ensure_hf_model(SCRIPT_DIR, ue.cfg)
    swin2 = model2.swin
    with torch.no_grad():
        # Get embedding output from HF
        embed_out = swin2.embeddings(pixel_values.to(torch.bfloat16))[0]  # (1, 9216, 192)
        embed_2d = embed_out.squeeze(0)  # (9216, 192)

        block = swin2.encoder.layers[0].blocks[0]  # stage 0, layer 0

        # 1. LayerNorm before attention
        ref_ln = block.layernorm_before(embed_2d.unsqueeze(0)).squeeze(0)  # (9216, 192)

        # 2. Window partition (no shift for layer 0)
        spatial = ue.STAGE_SPATIAL[0]  # 96
        ws = ue.WINDOW_SIZE  # 12
        ref_spatial = ref_ln.reshape(spatial, spatial, ue.EMBED_DIM)
        # Manual window partition
        num_wh = spatial // ws
        num_ww = spatial // ws
        ref_windows = ref_spatial.reshape(num_wh, ws, num_ww, ws, ue.EMBED_DIM)
        ref_windows = ref_windows.permute(0, 2, 1, 3, 4).contiguous().reshape(-1, ws * ws, ue.EMBED_DIM)
        # (64, 144, 192)

        # 3. QKV projections
        attn = block.attention.self
        ref_q = ref_windows.reshape(-1, ue.EMBED_DIM) @ attn.query.weight.data.to(torch.bfloat16).T + attn.query.bias.data.to(torch.bfloat16)
        ref_k = ref_windows.reshape(-1, ue.EMBED_DIM) @ attn.key.weight.data.to(torch.bfloat16).T + attn.key.bias.data.to(torch.bfloat16)
        ref_v = ref_windows.reshape(-1, ue.EMBED_DIM) @ attn.value.weight.data.to(torch.bfloat16).T + attn.value.bias.data.to(torch.bfloat16)
        # (9216, 192) each
    del model2

    # Run encoder (first layer)
    encoder_addrs = ue.compile_encoder()
    ue.run_encoder(encoder_addrs)

    # Read back Q projection
    st = ue.stage_tensors[0]
    num_windows = (ue.STAGE_SPATIAL[0] // ue.WINDOW_SIZE) ** 2  # 64
    M_win = num_windows * ue.WINDOW_AREA  # 9216
    dim = ue.STAGE_DIMS[0]  # 192

    # Intermediate check: LayerNorm output
    hw_ln = ue.dma_from_accelerator_memory(st['ln_output'], (M_win, dim)).to(torch.bfloat16)
    ln_diff = (ref_ln.float() - hw_ln.float()).abs()
    ln_signal = (ref_ln.float() ** 2).mean().item()
    ln_noise = ((ref_ln.float() - hw_ln.float()) ** 2).mean().item()
    ln_snr = 10 * math.log10(ln_signal / ln_noise) if ln_noise > 0 else float('inf')
    print(f"  LN: MAE={ln_diff.mean().item():.6f}, SNR={ln_snr:.2f} dB — {'PASS' if ln_snr > 30 else 'FAIL'}")

    # Intermediate check: Window partition output
    hw_win = ue.dma_from_accelerator_memory(st['windowed'], (M_win, dim)).to(torch.bfloat16)
    win_diff = (ref_windows.reshape(-1, dim).float() - hw_win.float()).abs()
    win_signal = (ref_windows.reshape(-1, dim).float() ** 2).mean().item()
    win_noise = ((ref_windows.reshape(-1, dim).float() - hw_win.float()) ** 2).mean().item()
    win_snr = 10 * math.log10(win_signal / win_noise) if win_noise > 0 else float('inf')
    print(f"  WindowPartition: MAE={win_diff.mean().item():.6f}, SNR={win_snr:.2f} dB — {'PASS' if win_snr > 30 else 'FAIL'}")

    # QKV check
    hw_q = ue.dma_from_accelerator_memory(st['q_proj'], (M_win, dim)).to(torch.bfloat16)
    hw_k = ue.dma_from_accelerator_memory(st['k_proj'], (M_win, dim)).to(torch.bfloat16)
    hw_v = ue.dma_from_accelerator_memory(st['v_proj'], (M_win, dim)).to(torch.bfloat16)

    for name, ref, hw in [('Q', ref_q, hw_q), ('K', ref_k, hw_k), ('V', ref_v, hw_v)]:
        diff = (ref.float() - hw.float()).abs()
        mae = diff.mean().item()
        signal = (ref.float() ** 2).mean().item()
        noise = ((ref.float() - hw.float()) ** 2).mean().item()
        snr = 10 * math.log10(signal / noise) if noise > 0 else float('inf')
        status = "PASS" if snr > 30 else "FAIL"
        print(f"  {name}: MAE={mae:.6f}, SNR={snr:.2f} dB — {status}")

    # Attention output check
    # CPU reference: run full self-attention on the windowed Q/K/V
    with torch.no_grad():
        model3, _ = _ensure_hf_model(SCRIPT_DIR, ue.cfg)
        block3 = model3.swin.encoder.layers[0].blocks[0]
        attn3 = block3.attention.self

        # Reconstruct Q/K/V in multi-head format from ref
        num_heads = ue.NUM_HEADS[0]  # 6
        head_dim = ue.HEAD_DIM  # 32
        wa = ue.WINDOW_AREA  # 144
        nw = (ue.STAGE_SPATIAL[0] // ue.WINDOW_SIZE) ** 2  # 64

        # ref_q is (9216, 192) = (64*144, 6*32)
        ref_q_mh = ref_q.reshape(nw, wa, num_heads, head_dim).permute(0, 2, 1, 3)  # (64, 6, 144, 32)
        ref_k_mh = ref_k.reshape(nw, wa, num_heads, head_dim).permute(0, 2, 1, 3)
        ref_v_mh = ref_v.reshape(nw, wa, num_heads, head_dim).permute(0, 2, 1, 3)

        # Attention scores: Q @ K^T / sqrt(head_dim)
        ref_scores = torch.matmul(ref_q_mh.float(), ref_k_mh.float().transpose(-1, -2))
        ref_scores = ref_scores / math.sqrt(head_dim)

        # Add relative position bias
        rpb_table = attn3.relative_position_bias_table  # (529, 6)
        rpb_index = attn3.relative_position_index  # (144, 144)
        rpb = rpb_table[rpb_index.view(-1)].view(wa, wa, num_heads)
        rpb = rpb.permute(2, 0, 1).contiguous()  # (6, 144, 144)
        ref_scores = ref_scores + rpb.float().unsqueeze(0)  # broadcast across windows

        # Softmax
        ref_probs = torch.nn.functional.softmax(ref_scores, dim=-1)

        # Context: probs @ V
        ref_ctx = torch.matmul(ref_probs, ref_v_mh.float())  # (64, 6, 144, 32)
        # Flatten to (64*6, 144, 32) = (384, 144, 32)
        ref_ctx_flat = ref_ctx.reshape(nw * num_heads, wa, head_dim).to(torch.bfloat16)

        del model3

    # Read back HW attention output: (total_batches, wa_pad, hd_pad) = (384, 192, 64)
    wa_pad = ue.pad_dim(wa)  # 192
    hd_pad = ue.pad_dim(head_dim)  # 64
    total_batches = nw * num_heads  # 384
    hw_attn = ue.dma_from_accelerator_memory(
        st['attn_output'], (total_batches * wa_pad, hd_pad)).to(torch.bfloat16)
    hw_attn = hw_attn.reshape(total_batches, wa_pad, hd_pad)

    # Extract real data: (384, 144, 32) from (384, 192, 64)
    hw_attn_real = hw_attn[:, :wa, :head_dim]

    attn_diff = (ref_ctx_flat.float() - hw_attn_real.float()).abs()
    attn_mae = attn_diff.mean().item()
    attn_signal = (ref_ctx_flat.float() ** 2).mean().item()
    attn_noise = ((ref_ctx_flat.float() - hw_attn_real.float()) ** 2).mean().item()
    attn_snr = 10 * math.log10(attn_signal / attn_noise) if attn_noise > 0 else float('inf')
    print(f"  Attention: MAE={attn_mae:.6f}, SNR={attn_snr:.2f} dB — {'PASS' if attn_snr > 30 else 'FAIL'}")

    assert False, "Encoder layer 0 test complete (LN + window partition + QKV + permute + attention)."


if __name__ == "__main__":
    main()
