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

import builtins
import json
import math
import os
import sys
import warnings
warnings.filterwarnings("ignore", message=".*torchao.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageClassification
from huggingface_hub import snapshot_download
import time

_original_print = builtins.print
_SILENT_MODE = False

def quiet_print(*args, **kwargs):
    if _SILENT_MODE:
        return
    _original_print(*args, **kwargs)

builtins.print = quiet_print

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, DMA_DEVICE_C2H, DRAM_INSTRUCTION_ADDR, TYPE,
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

    row_elements = window_size * C          # elements in one window row
    output_offset = 0
    for wh in range(num_windows_h):
        for ww in range(num_windows_w):
            # Top-left corner of this window in input DRAM
            window_src = INPUT_DRAM_ADDR + (wh * window_size * W + ww * window_size) * C * bytes_per_element

            # Read each window row individually into contiguous SRAM slots
            for row in range(window_size):
                row_src = window_src + row * input_row_stride
                row_sram = row * row_elements * bytes_per_element
                ue.accelerator_memory_to_sram(
                    accelerator_dram_address=row_src,
                    sram_address=row_sram,
                    element_size=row_elements,
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

    row_elements = window_size * C
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
            # Write each window row individually to its spatial position
            for row in range(window_size):
                row_sram = row * row_elements * bytes_per_element
                row_dst = window_dst + row * output_row_stride
                ue.sram_to_accelerator_memory(
                    sram_address=row_sram,
                    accelerator_dram_address=row_dst,
                    element_size=row_elements,
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
    model = AutoModelForImageClassification.from_pretrained(model_dir, dtype=torch.bfloat16, device_map=None)
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

        self.params_from_bin = self.load_params()
        if not self.params_from_bin:
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

        if not _SILENT_MODE:
            _original_print(f"Writing {self.capture_count:,} instructions ({total_bytes / 1024**2:.1f} MB) to DRAM at 0x{start_addr:x}...")
        offset = 0
        while offset < total_bytes:
            chunk = min(self.DMA_CHUNK_BYTES, total_bytes - offset)
            self.dma_write(DMA_DEVICE_H2C, start_addr + offset, data[offset:offset + chunk], chunk)
            offset += chunk
        if not _SILENT_MODE:
            _original_print(f"Successfully wrote {total_bytes / 1024**2:.1f} MB ({self.capture_count:,} instructions) to DRAM")
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
            # matmat_mul_core expects B as (N, K) since it computes A @ B^T
            self.unpad_weight_addrs.append(self._alloc_param(w.T.contiguous()))

        # ---- Encoder: patch merging weights (stages 0, 1, 2 only — no merge after stage 3) ----
        self.patch_merge_weights = []
        for s in range(self.NUM_STAGES - 1):
            dim = self.STAGE_DIMS[s]
            ds = swin.encoder.layers[s].downsample
            pmw = {}
            pmw['norm_gamma'] = self._alloc_param(ds.norm.weight.data)   # (4*dim,)
            pmw['norm_beta']  = self._alloc_param(ds.norm.bias.data)     # (4*dim,)
            pmw['reduction']  = self._alloc_param(ds.reduction.weight.data)  # (2*dim, 4*dim)
            self.patch_merge_weights.append(pmw)

        # ---- Post-encoder ----
        # Final LayerNorm weights → params DRAM
        self.POST_ENCODER_LN_GAMMA = self._alloc_param(swin.layernorm.weight.data)  # (1536,)
        self.POST_ENCODER_LN_BETA  = self._alloc_param(swin.layernorm.bias.data)    # (1536,)

        # Classifier weights → device DRAM
        # matmat_mul_core computes A @ B^T, so B is stored as (N, K) = (num_labels, 1536)
        # which is exactly the HF layout. Pad N to 64-aligned for AXI stride alignment.
        cls_w = model.classifier.weight.data.to(torch.bfloat16)  # (num_labels, 1536)
        cls_b = model.classifier.bias.data.to(torch.bfloat16)    # (num_labels,)
        self.NUM_LABELS = cls_w.shape[0]
        self.NUM_LABELS_PAD = self.pad_dim(self.NUM_LABELS)
        cls_w_pad = torch.zeros(self.NUM_LABELS_PAD, cls_w.shape[1], dtype=torch.bfloat16)
        cls_w_pad[:self.NUM_LABELS, :] = cls_w
        cls_b_pad = torch.zeros(self.NUM_LABELS_PAD, dtype=torch.bfloat16)
        cls_b_pad[:self.NUM_LABELS] = cls_b
        self.CLASSIFIER_WEIGHT_DRAM = self._alloc_param(cls_w_pad.contiguous())  # (num_labels_pad, 1536)
        self.CLASSIFIER_BIAS_DRAM = self._alloc_param(cls_b_pad)                 # (num_labels_pad,)

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
            st['shifted'] = self._alloc_tensor(M * dim)      # forward cyclic shift output (step 2)
            st['win_reverse'] = self._alloc_tensor(M * dim)  # window reverse output (step 10)

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

            # Patch merging scratch: (spatial/2 * spatial/2, 4*dim) — only for stages 0-2
            if s < self.NUM_STAGES - 1:
                next_M = (spatial // 2) * (spatial // 2)
                st['merge_buf'] = self._alloc_tensor(next_M * 4 * dim)

            self.stage_tensors.append(st)

        # ---- Post-encoder tensors ----
        final_M = self.STAGE_SPATIAL[3] ** 2  # 12*12 = 144
        final_C = self.STAGE_DIMS[3]           # 1536
        self.FINAL_LN_OUTPUT_DRAM = self._alloc_tensor(final_M * final_C)
        self.POOLED_OUTPUT_DRAM = self._alloc_tensor(1 * final_C)              # (1, 1536) avg pooled
        self.CLASSIFIER_OUTPUT_DRAM = self._alloc_tensor(1 * self.pad_dim(self.NUM_LABELS))  # (1, num_labels_pad)

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

    # ------------------------------------------------------------------
    # Bin dump / load (params + programs)
    # ------------------------------------------------------------------

    def dump_params(self):
        """Dump entire params DRAM to swin_bin/params.bin + params.json."""
        bin_dir = os.path.join(self.script_dir, "swin_bin")
        os.makedirs(bin_dir, exist_ok=True)
        bin_path = os.path.join(bin_dir, "params.bin")
        meta_path = os.path.join(bin_dir, "params.json")
        total = self.get_params_dram_usage()
        CHUNK = 1 * 1024 * 1024
        with open(bin_path, "wb") as f:
            offset = 0
            while offset < total:
                sz = min(CHUNK, total - offset)
                buf = bytearray(sz)
                self.dma_read(DMA_DEVICE_C2H, self._params_dram_base + offset, buf, sz)
                f.write(buf)
                offset += sz
        with open(meta_path, "w") as f:
            json.dump({"size": total, "num_labels": self.NUM_LABELS}, f)
        _original_print(f"  Params dumped: {total / 1024**2:.1f} MB → {bin_path}")
        model, _ = _ensure_hf_model(self.script_dir, self.cfg)
        labels_path = os.path.join(bin_dir, "labels.json")
        with open(labels_path, "w") as f:
            json.dump({str(k): v for k, v in model.config.id2label.items()}, f)
        del model
        _original_print(f"  Labels saved: {labels_path}")

    def load_params(self) -> bool:
        """Load params DRAM from bin. Returns True if loaded, False if not found."""
        bin_dir = os.path.join(self.script_dir, "swin_bin")
        bin_path = os.path.join(bin_dir, "params.bin")
        meta_path = os.path.join(bin_dir, "params.json")
        if not os.path.exists(bin_path) or not os.path.exists(meta_path):
            return False
        with open(meta_path) as f:
            meta = json.load(f)
        total = meta["size"]
        self.NUM_LABELS = meta["num_labels"]
        self.NUM_LABELS_PAD = self.pad_dim(self.NUM_LABELS)
        CHUNK = 1 * 1024 * 1024
        with open(bin_path, "rb") as f:
            offset = 0
            while offset < total:
                data = f.read(min(CHUNK, total - offset))
                self.dma_write(DMA_DEVICE_H2C, self._params_dram_base + offset, data, len(data))
                offset += len(data)
        self.allocate_params_dram(total)
        _original_print(f"  Params loaded: {total / 1024**2:.1f} MB from bin")
        return True

    def dump_programs(self, program_addr: int):
        """Dump compiled program to swin_bin/programs.bin + programs.json."""
        bin_dir = os.path.join(self.script_dir, "swin_bin")
        os.makedirs(bin_dir, exist_ok=True)
        bin_path = os.path.join(bin_dir, "programs.bin")
        meta_path = os.path.join(bin_dir, "programs.json")
        size = self.get_program_dram_usage()
        buf = bytearray(size)
        self.dma_read(DMA_DEVICE_C2H, program_addr, buf, size)
        with open(bin_path, "wb") as f:
            f.write(buf)
        with open(meta_path, "w") as f:
            json.dump({"size": size}, f)
        _original_print(f"  Program dumped: {size / 1024**2:.1f} MB → {bin_path}")

    def load_programs(self) -> int | None:
        """Load compiled program from bin. Returns program DRAM addr or None if not found."""
        bin_dir = os.path.join(self.script_dir, "swin_bin")
        bin_path = os.path.join(bin_dir, "programs.bin")
        meta_path = os.path.join(bin_dir, "programs.json")
        if not os.path.exists(bin_path) or not os.path.exists(meta_path):
            return None
        with open(bin_path, "rb") as f:
            data = f.read()
        addr = self.get_program_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, addr, data, len(data))
        self.allocate_program_dram(len(data))
        _original_print(f"  Program loaded: {len(data) / 1024**2:.1f} MB from bin")
        return addr

    # ------------------------------------------------------------------
    # Fully-fused: pre_encoder + encoder + post_encoder in one stream
    # ------------------------------------------------------------------

    def compile_full_fused(self) -> int:
        """Compile the entire forward pass as a single instruction stream.

        pre_encoder (patch embed + bias + LN)
        → encoder (all 4 stages, 24 layers + 3 patch merges)
        → post_encoder (final LN + avg pool matmul + classifier matmul w/ HW argmax)

        Returns a single program DRAM address.
        """
        pad = self.pad_dim
        final_M = self.STAGE_SPATIAL[3] ** 2   # 144
        final_C = self.STAGE_DIMS[3]            # 1536

        self.start_capture()

        # ================================================================
        # PRE-ENCODER: patch extraction + bias + LayerNorm
        # ================================================================
        M_pre = self.GRID_SIZE * self.GRID_SIZE  # 9216
        N_pre = self.EMBED_DIM                    # 192
        C = self.NUM_CHANNELS
        H = W = self.IMAGE_SIZE
        P = self.PATCH_SIZE
        K_patch = C * P * UE_VECTOR_SIZE          # 768

        bf16_patching_core(
            self,
            INPUT_DRAM_ADDR=self.IMAGE_DRAM,
            OUTPUT_DRAM_ADDR=self.PATCH_OUTPUT_DRAM,
            weight_dram_addrs=self.PATCH_EMBED_WEIGHT_ADDRS,
            C=C, H=H, W=W,
            patch_h=P, patch_w=P,
            K=K_patch, N=N_pre,
        )

        # Bias add
        self.accelerator_memory_to_sram(
            accelerator_dram_address=self.PATCH_EMBED_BIAS,
            sram_address=0x80000,
            element_size=N_pre,
        )
        chunk_size = min(URAM_NEAR_FULL_ELEMENTS // N_pre, M_pre)
        for i, m_take in self.chunk_ranges(M_pre, chunk_size):
            self.accelerator_memory_to_sram(
                accelerator_dram_address=self.PATCH_OUTPUT_DRAM + i * N_pre * 2,
                sram_address=0x00000,
                element_size=m_take * N_pre,
            )
            for j in range(m_take):
                self.eltwise_add_core(
                    vector_A_sram_start_addr=0x00000 + j * N_pre * 2,
                    vector_B_sram_start_addr=0x80000,
                    vector_C_sram_wb_addr=0x00000 + j * N_pre * 2,
                    element_size=N_pre,
                )
            self.sram_to_accelerator_memory(
                sram_address=0x00000,
                accelerator_dram_address=self.PATCH_OUTPUT_DRAM + i * N_pre * 2,
                element_size=m_take * N_pre,
            )

        # Patch embedding LayerNorm
        self.layer_norm_core_dram(
            M=M_pre, N=N_pre,
            A_DRAM_ADDR=self.PATCH_OUTPUT_DRAM,
            OUTPUT_DRAM_ADDR=self.EMBED_OUTPUT_DRAM,
            GAMMA_DRAM_ADDR=self.PATCH_EMBED_LN_GAMMA,
            BETA_DRAM_ADDR=self.PATCH_EMBED_LN_BETA,
        )

        # ================================================================
        # ENCODER: all stages, layers, and patch merges
        # ================================================================
        for s in range(self.NUM_STAGES):
            dim = self.STAGE_DIMS[s]
            spatial = self.STAGE_SPATIAL[s]
            num_heads = self.NUM_HEADS[s]
            num_windows = (spatial // self.WINDOW_SIZE) ** 2
            wa = self.WINDOW_AREA
            wa_pad = pad(wa)
            hd_pad = pad(self.HEAD_DIM)
            mlp_dim = int(dim * self.MLP_RATIO)
            M = spatial * spatial
            total_batches = num_windows * num_heads
            st = self.stage_tensors[s]

            for l in range(self.DEPTHS[s]):
                lw = self.encoder_weights[s][l]
                shift_size = 0 if l % 2 == 0 else self.WINDOW_SIZE // 2

                # --- 1. Pre-attention LayerNorm ---
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
                M_win = num_windows * wa
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
                head_dim = self.HEAD_DIM
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

                # --- 5b. Pre-scale Q ---
                scale_correction = math.sqrt(hd_pad) / math.sqrt(head_dim)
                total_q_elements = total_batches * wa_pad * hd_pad
                chunk_size_q = URAM_NEAR_FULL_ELEMENTS
                for offset in range(0, total_q_elements, chunk_size_q):
                    take = min(chunk_size_q, total_q_elements - offset)
                    self.accelerator_memory_to_sram(
                        accelerator_dram_address=st['q_heads'] + offset * 2,
                        sram_address=0x00000,
                        element_size=take,
                    )
                    self.broadcast_mul(
                        scalar=scale_correction,
                        sram_start_addr=0x00000,
                        sram_wb_addr=0x00000,
                        element_size=take,
                    )
                    self.sram_to_accelerator_memory(
                        sram_address=0x00000,
                        accelerator_dram_address=st['q_heads'] + offset * 2,
                        element_size=take,
                    )

                # --- 6. Flash attention batched ---
                dram_zero_fill(self, st['attn_output'], total_batches * wa_pad * hd_pad)
                flash_attention_batched(self,
                    num_batches=total_batches,
                    head_dim=hd_pad,
                    seq_len=wa_pad,
                    Q_DRAM_ADDR=st['q_heads'],
                    K_DRAM_ADDR=st['k_heads'],
                    V_DRAM_ADDR=st['v_heads'],
                    OUTPUT_DRAM_ADDR=st['attn_output'],
                    SCRATCH_DRAM_ADDR=st['attn_scratch'],
                    BIAS_DRAM_ADDR=lw['rel_pos_bias'],
                )

                # --- 7. Inverse permute ---
                for w in range(num_windows):
                    for h in range(num_heads):
                        src = st['attn_output'] + (w * num_heads + h) * wa_pad * hd_pad * 2
                        dst = st['permute_temp'] + h * wa * hd_pad * 2
                        self.accelerator_memory_to_sram(
                            accelerator_dram_address=src,
                            sram_address=0x00000,
                            element_size=wa * hd_pad,
                        )
                        self.sram_to_accelerator_memory(
                            sram_address=0x00000,
                            accelerator_dram_address=dst,
                            element_size=wa * hd_pad,
                        )

                    permute_out = st['permute_temp'] + num_heads * wa * hd_pad * 2
                    self.bf16_permute_core(
                        dim_0=num_heads, dim_1=wa, dim_2=hd_pad,
                        INPUT_DRAM_ADDR=st['permute_temp'],
                        OUTPUT_DRAM_ADDR=permute_out,
                    )

                    out_dst = st['attn_permuted'] + w * wa * num_heads * hd_pad * 2
                    row_elems = num_heads * hd_pad
                    rows_per_chunk = max(1, URAM_NEAR_FULL_ELEMENTS // row_elems)
                    for row_start in range(0, wa, rows_per_chunk):
                        row_take = min(rows_per_chunk, wa - row_start)
                        chunk_elems = row_take * row_elems
                        self.accelerator_memory_to_sram(
                            accelerator_dram_address=permute_out + row_start * row_elems * 2,
                            sram_address=0x00000,
                            element_size=chunk_elems,
                        )
                        self.sram_to_accelerator_memory(
                            sram_address=0x00000,
                            accelerator_dram_address=out_dst + row_start * row_elems * 2,
                            element_size=chunk_elems,
                        )

                # --- 8. Unpad matmul ---
                self.matmat_mul_core(
                    M=M_win, K=num_heads * hd_pad, N=dim,
                    A_DRAM_ADDR=st['attn_permuted'],
                    B_DRAM_ADDR=self.unpad_weight_addrs[s],
                    OUTPUT_DRAM_ADDR=st['attn_unpadded'],
                )

                # --- 9. Output projection ---
                self.matmat_mul_core(
                    M=M_win, K=dim, N=dim,
                    A_DRAM_ADDR=st['attn_unpadded'],
                    B_DRAM_ADDR=lw['out_proj_weight'],
                    OUTPUT_DRAM_ADDR=st['out_proj'],
                    C_DRAM_ADDR=lw['out_proj_bias'],
                    bias_mode="broadcast_N",
                )

                # --- 10. Window reverse ---
                window_reverse_dram(self,
                    INPUT_DRAM_ADDR=st['out_proj'],
                    OUTPUT_DRAM_ADDR=st['win_reverse'],
                    H=spatial, W=spatial, C=dim,
                    window_size=self.WINDOW_SIZE)

                # --- 11. Reverse cyclic shift (odd layers only) ---
                if shift_size > 0:
                    cyclic_shift_dram(self,
                        INPUT_DRAM_ADDR=st['win_reverse'],
                        OUTPUT_DRAM_ADDR=st['ln_output'],
                        H=spatial, W=spatial, C=dim,
                        shift_h=spatial - shift_size, shift_w=spatial - shift_size)

                # --- 12. Residual add 1 ---
                attn_spatial_addr = st['ln_output'] if shift_size > 0 else st['win_reverse']
                chunk_elems = min(URAM_NEAR_FULL_ELEMENTS, M * dim)
                for offset in range(0, M * dim, chunk_elems):
                    take = min(chunk_elems, M * dim - offset)
                    self.accelerator_memory_to_sram(
                        accelerator_dram_address=layer_input_addr + offset * 2,
                        sram_address=0x00000,
                        element_size=take,
                    )
                    self.accelerator_memory_to_sram(
                        accelerator_dram_address=attn_spatial_addr + offset * 2,
                        sram_address=0x80000,
                        element_size=take,
                    )
                    self.eltwise_add_core(
                        vector_A_sram_start_addr=0x00000,
                        vector_B_sram_start_addr=0x80000,
                        vector_C_sram_wb_addr=0x00000,
                        element_size=take,
                    )
                    self.sram_to_accelerator_memory(
                        sram_address=0x00000,
                        accelerator_dram_address=st['residual1'] + offset * 2,
                        element_size=take,
                    )

                # --- 13. Pre-MLP LayerNorm ---
                self.layer_norm_core_dram(
                    M=M, N=dim,
                    A_DRAM_ADDR=st['residual1'],
                    OUTPUT_DRAM_ADDR=st['mlp_ln'],
                    GAMMA_DRAM_ADDR=lw['ln_after_gamma'],
                    BETA_DRAM_ADDR=lw['ln_after_beta'],
                )

                # --- 14. MLP expand ---
                self.matmat_mul_core(
                    M=M, K=dim, N=mlp_dim,
                    A_DRAM_ADDR=st['mlp_ln'],
                    B_DRAM_ADDR=lw['mlp_expand_weight'],
                    OUTPUT_DRAM_ADDR=st['mlp_mid'],
                    C_DRAM_ADDR=lw['mlp_expand_bias'],
                    bias_mode="broadcast_N",
                    gelu_enable=True,
                )

                # --- 15. MLP contract ---
                self.matmat_mul_core(
                    M=M, K=mlp_dim, N=dim,
                    A_DRAM_ADDR=st['mlp_mid'],
                    B_DRAM_ADDR=lw['mlp_contract_weight'],
                    OUTPUT_DRAM_ADDR=st['mlp_out'],
                    C_DRAM_ADDR=lw['mlp_contract_bias'],
                    bias_mode="broadcast_N",
                )

                # --- 16. Residual add 2 ---
                layer_output_addr = st['layer_input']
                chunk_elems = min(URAM_NEAR_FULL_ELEMENTS, M * dim)
                for offset in range(0, M * dim, chunk_elems):
                    take = min(chunk_elems, M * dim - offset)
                    self.accelerator_memory_to_sram(
                        accelerator_dram_address=st['residual1'] + offset * 2,
                        sram_address=0x00000,
                        element_size=take,
                    )
                    self.accelerator_memory_to_sram(
                        accelerator_dram_address=st['mlp_out'] + offset * 2,
                        sram_address=0x80000,
                        element_size=take,
                    )
                    self.eltwise_add_core(
                        vector_A_sram_start_addr=0x00000,
                        vector_B_sram_start_addr=0x80000,
                        vector_C_sram_wb_addr=0x00000,
                        element_size=take,
                    )
                    self.sram_to_accelerator_memory(
                        sram_address=0x00000,
                        accelerator_dram_address=layer_output_addr + offset * 2,
                        element_size=take,
                    )

            # ---- Patch merging (after all layers in stages 0-2) ----
            if s < self.NUM_STAGES - 1:
                pmw = self.patch_merge_weights[s]
                next_spatial = spatial // 2
                next_M = next_spatial * next_spatial
                next_dim = dim * 2

                patch_merging_gather_dram(self,
                    INPUT_DRAM_ADDR=st['layer_input'],
                    OUTPUT_DRAM_ADDR=st['merge_buf'],
                    H=spatial, W=spatial, C=dim)

                self.layer_norm_core_dram(
                    M=next_M, N=4 * dim,
                    A_DRAM_ADDR=st['merge_buf'],
                    OUTPUT_DRAM_ADDR=st['merge_buf'],
                    GAMMA_DRAM_ADDR=pmw['norm_gamma'],
                    BETA_DRAM_ADDR=pmw['norm_beta'],
                )

                self.matmat_mul_core(
                    M=next_M, K=4 * dim, N=next_dim,
                    A_DRAM_ADDR=st['merge_buf'],
                    B_DRAM_ADDR=pmw['reduction'],
                    OUTPUT_DRAM_ADDR=self.stage_tensors[s + 1]['layer_input'],
                )

        # ================================================================
        # POST-ENCODER: final LN + avg pool + classifier (all on HW)
        # ================================================================

        # Final LayerNorm
        self.layer_norm_core_dram(
            M=final_M, N=final_C,
            A_DRAM_ADDR=self.stage_tensors[3]['layer_input'],
            OUTPUT_DRAM_ADDR=self.FINAL_LN_OUTPUT_DRAM,
            GAMMA_DRAM_ADDR=self.POST_ENCODER_LN_GAMMA,
            BETA_DRAM_ADDR=self.POST_ENCODER_LN_BETA,
        )

        # Avg pool via eltwise accumulation: sum all 144 rows, then scale by 1/144
        # (matmul can't be used here because LN output layout doesn't match B = (N, K))
        dram_zero_fill(self, self.POOLED_OUTPUT_DRAM, final_C)
        for row in range(final_M):
            self.accelerator_memory_to_sram(
                accelerator_dram_address=self.FINAL_LN_OUTPUT_DRAM + row * final_C * 2,
                sram_address=0x00000,
                element_size=final_C,
            )
            self.accelerator_memory_to_sram(
                accelerator_dram_address=self.POOLED_OUTPUT_DRAM,
                sram_address=0x80000,
                element_size=final_C,
            )
            self.eltwise_add_core(
                vector_A_sram_start_addr=0x00000,
                vector_B_sram_start_addr=0x80000,
                vector_C_sram_wb_addr=0x80000,
                element_size=final_C,
            )
            self.sram_to_accelerator_memory(
                sram_address=0x80000,
                accelerator_dram_address=self.POOLED_OUTPUT_DRAM,
                element_size=final_C,
            )
        # Scale by 1/144
        self.accelerator_memory_to_sram(
            accelerator_dram_address=self.POOLED_OUTPUT_DRAM,
            sram_address=0x00000,
            element_size=final_C,
        )
        self.broadcast_mul(
            scalar=1.0 / final_M,
            sram_start_addr=0x00000,
            sram_wb_addr=0x00000,
            element_size=final_C,
        )
        self.sram_to_accelerator_memory(
            sram_address=0x00000,
            accelerator_dram_address=self.POOLED_OUTPUT_DRAM,
            element_size=final_C,
        )

        # Classifier: (1, 1536) @ (num_labels_pad, 1536)^T + bias -> (1, num_labels_pad)
        # matmat_mul_core computes A @ B^T; B stored as (N, K) = (num_labels_pad, 1536)
        # HW argmax is computed automatically for M=1 matmul
        # Padded positions have zero weight+bias so they won't win argmax
        self.matmat_mul_core(
            M=1, K=final_C, N=self.NUM_LABELS_PAD,
            A_DRAM_ADDR=self.POOLED_OUTPUT_DRAM,
            B_DRAM_ADDR=self.CLASSIFIER_WEIGHT_DRAM,
            OUTPUT_DRAM_ADDR=self.CLASSIFIER_OUTPUT_DRAM,
            C_DRAM_ADDR=self.CLASSIFIER_BIAS_DRAM,
            bias_mode="broadcast_N",
        )

        self.stop_capture()
        self.generate_instruction_halt()

        prog_addr = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(prog_addr)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        self.clear_capture_buffer()

        return prog_addr

    def run_full_fused(self, pixel_values: torch.Tensor, program_addr: int, timeout: float = 120.0) -> int:
        """Run the fully-fused forward pass. Returns predicted class index (from HW argmax)."""
        image_chw = pixel_values.squeeze(0).to(torch.bfloat16)
        self.dma_to_accelerator_memory(self.IMAGE_DRAM, image_chw.contiguous().flatten())
        self.start_execute_from_dram(program_addr)
        self.wait_queue(timeout)
        return self.get_arg_max_index1()

    def get_arg_max_index1(self):
        """Read hardware argmax1 register."""
        return self.read_reg32(user_dma_core.UE_ARGMAX1_INDEX)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(image_path: str) -> torch.Tensor:
    """Load and preprocess image for Swin-Large-384. Returns (1, 3, 384, 384)."""
    image = Image.open(image_path).convert("RGB").resize((384, 384), Image.Resampling.BILINEAR)
    img_t = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img_t - mean) / std


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

    global _SILENT_MODE
    _SILENT_MODE = True

    set_dma_device(args.dev)
    # Refresh local bindings shadowed at import time so DMA goes to the right device
    import sys as _sys, user_dma_core as _udc
    _mod = _sys.modules[__name__]
    _mod.DMA_DEVICE_H2C = _udc.DMA_DEVICE_H2C
    _mod.DMA_DEVICE_C2H = _udc.DMA_DEVICE_C2H
    _mod.DMA_DEVICE_USER = _udc.DMA_DEVICE_USER
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle

    # Load image
    image_path = args.image or os.path.join(SCRIPT_DIR, "../../test_samples/vette.jpg")
    pixel_values = preprocess_image(image_path)

    _original_print(f"Swin-Large-384 on {args.dev}")

    # Create engine (loads weights from bin or HF model)
    import time as _time
    t0 = _time.perf_counter()
    ue = Swin_UnifiedEngine(script_dir=SCRIPT_DIR)
    ue.software_reset()
    t_weights = _time.perf_counter()
    _original_print(f"  Weights: {t_weights - t0:.3f}s")

    # Load or compile program
    import threading

    def _progress_timer(label, start_time, stop_event):
        while not stop_event.wait(1.0):
            elapsed = _time.perf_counter() - start_time
            _original_print(f"\r  {label} ({elapsed:.0f}s)", end="", flush=True)

    prog_addr = ue.load_programs()
    if prog_addr is None:
        if ue.params_from_bin:
            raise RuntimeError("Cannot compile without weight addresses — delete swin_bin/ and re-run")
        stop_c = threading.Event()
        timer_c = threading.Thread(target=_progress_timer,
            args=("Compiling (this will take some time, swin has a lot of batched operations)", t_weights, stop_c), daemon=True)
        timer_c.start()
        prog_addr = ue.compile_full_fused()
        stop_c.set()
        timer_c.join()
        ue.dump_params()
        ue.dump_programs(prog_addr)
    t_compile = _time.perf_counter()
    _original_print(f"\r  Compile: {t_compile - t_weights:.3f}s")

    t_exec_start = _time.perf_counter()
    stop = threading.Event()
    timer = threading.Thread(target=_progress_timer, args=("Executing", t_exec_start, stop), daemon=True)
    timer.start()

    predicted_idx = ue.run_full_fused(pixel_values, prog_addr)

    stop.set()
    timer.join()
    t_exec = _time.perf_counter()
    _original_print(f"\r  Executing: {t_exec - t_exec_start:.3f}s")

    # Look up label
    labels_path = os.path.join(SCRIPT_DIR, "swin_bin", "labels.json")
    with open(labels_path) as f:
        id2label = json.load(f)
    label = id2label.get(str(predicted_idx), str(predicted_idx))
    _original_print(f"\n  Image: {image_path}")
    _original_print(f"  Prediction: {label!r} (class {predicted_idx})")


if __name__ == "__main__":
    main()
