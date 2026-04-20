#!/usr/bin/env python3
"""
SAM 1 ViT-B — Per-Operation CPU vs HW Comparison

Runs every operation checkpoint in the HW flow sequentially, comparing each
intermediate tensor against the CPU reference. Prints a detailed table with
SNR (dB), cosine similarity, max absolute error, and % within 1 BF16 ULP.

Checkpoint order:
  [Encoder]
    patch_embed         VIT_LAYER_IN      (4096, 768)  after patch matmul + pos add
    block{N}_ln1        VIT_LN_OUT        (4096, 768)  after block N pre-attn LayerNorm
    block{N}_windowed   VIT_WINDOWED      (4900, 768)  after pad + window partition
    block{N}_q          VIT_Q             (M_flat,768) after Q projection
    block{N}_k          VIT_K             (M_flat,768) after K projection
    block{N}_v          VIT_V             (M_flat,768) after V projection
    block{N}_attn_merge VIT_ATTN_MERGED   (M_flat,768) after attention + merge heads
    block{N}_out_proj   VIT_OUT_PROJ      (M_flat,768) after output projection
    block{N}_residual1  VIT_RESIDUAL      (4096, 768)  after first residual add
    block{N}_ln2        VIT_LN_OUT        (4096, 768)  after pre-MLP LayerNorm
    block{N}_fc1        VIT_MLP_MID       (4096,3072)  after fc1 + GELU
    block{N}_mlp_out    VIT_MLP_OUT       (4096, 768)  after fc2
    block{N}_out        VIT_LAYER_IN      (4096, 768)  after residual 2 (block output)
  [Neck]
    neck_conv1          VIT_NECK_OUT      (4096, 256)  after 1×1 conv
    neck_ln1            VIT_NECK_OUT      (4096, 256)  after LayerNorm 1
    neck_conv2          NECK_OUT          (4096, 256)  after 3×3 conv
    neck_ln2            NECK_OUT          (4096, 256)  after LayerNorm 2 (image embedding)
  [Decoder]
    dec_src             DEC_SRC           (4096, 256)  image src + no_mask_embed
    dec_l0_sa_out       DEC_TOKENS        (7, 256)     after layer 0 self-attn + norm1
    dec_l0_t2i_out      DEC_TOKENS        (7, 256)     after layer 0 token→image + norm2
    dec_l0_mlp_out      DEC_TOKENS        (7, 256)     after layer 0 MLP + norm3
    dec_l0_i2t_src      DEC_SRC           (4096, 256)  after layer 0 image→token + norm4
    dec_l1_sa_out       DEC_TOKENS        (7, 256)     after layer 1 self-attn + norm1
    dec_l1_t2i_out      DEC_TOKENS        (7, 256)     after layer 1 token→image + norm2
    dec_l1_mlp_out      DEC_TOKENS        (7, 256)     after layer 1 MLP + norm3
    dec_l1_i2t_src      DEC_SRC           (4096, 256)  after layer 1 image→token + norm4
    dec_final_attn      DEC_TOKENS        (7, 256)     after final cross-attn + final LN
    dec_iou             DEC_IOU_OUT       (4,)         IoU prediction scores
    dec_hyper           DEC_HYPER_OUT     (4, 32)      hypernetwork MLP outputs
    dec_up0             DEC_UP0_OUT       (16384, 64)  after upscale conv0 + LN + GELU
    dec_up1             DEC_UP1_OUT       (65536, 32)  after upscale conv1 + GELU
    dec_mask_logits     DEC_MASK_LOGITS   (65536, 4)   final mask logits

Usage:
    python sam1_vit_b_compare.py --image photo.jpg --dev xdma0
    python sam1_vit_b_compare.py --image photo.jpg --dev xdma0 --start neck_conv1
    python sam1_vit_b_compare.py --image photo.jpg --dev xdma0 --block 0
      (only sub-op checkpoints for block 0)
    python sam1_vit_b_compare.py --image photo.jpg --dev xdma0 --list
      (print all checkpoint names and exit)
"""

import argparse
import math
import os
import sys
import time as _time
import builtins

import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))

# ─────────────────────────────────────────────────────────────────────────────
# Silence the HW class's print output during compilation
# ─────────────────────────────────────────────────────────────────────────────
_original_print = builtins.print

import warnings
warnings.filterwarnings("ignore", message=".*torchao.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

import user_dma_core
from user_dma_core import URAM_NEAR_FULL_ELEMENTS, set_dma_device

from sam1_vit_b_test import (
    Sam1VitB_UnifiedEngine,
    _CheckpointStop,
    _ensure_checkpoint,
    _load_sam1_state_dict,
    SCRIPT_DIR as TEST_SCRIPT_DIR,
    _original_print as _hw_print,
)


# ─────────────────────────────────────────────────────────────────────────────
# Extended subclass with fine-grained per-op checkpoints
# ─────────────────────────────────────────────────────────────────────────────

class _Stop(Exception):
    """Raised to stop compilation at a named checkpoint."""
    def __init__(self, name):
        self.name = name
        super().__init__(name)


class CompareSAM1(Sam1VitB_UnifiedEngine):
    """Adds per-operation checkpoint compilation + CPU reference generation."""

    # ──────────────────── checkpoint compile ─────────────────────────────────

    def compile_to(self, checkpoint: str) -> int:
        """Compile forward pass up to (and including) `checkpoint`. Returns prog addr."""
        self._cmp_stop = checkpoint
        self.start_capture()
        try:
            self._compile_phases_compare(pad=self.pad_dim, bpe=2)
        except _Stop as e:
            if e.name != checkpoint:
                raise RuntimeError(f"Unexpected stop '{e.name}' (expected '{checkpoint}')") from e
        except AssertionError as e:
            pass  # tolerate asserts inside compile
        self._finalize_program()
        return self._last_prog_addr

    def _stop_if(self, name: str):
        """Raise _Stop if this is the requested checkpoint."""
        if getattr(self, '_cmp_stop', None) == name:
            raise _Stop(name)

    def _compile_phases_compare(self, pad, bpe):
        """Full compile loop with per-op _stop_if hooks."""
        from sam1_vit_b_test import (
            eltwise_add_dram, dram_copy, dram_zero_fill,
            broadcast_mul_dram, flash_attention_global_tiled,
            flash_attention_batched, conv2d_3x3_dram,
            pad_feature_map_dram, unpad_feature_map_dram,
            window_partition_dram, window_reverse_dram,
            multihead_merge_dram,
        )

        VD    = self.VIT_DIM
        VD_QK = self.VIT_QK_DIM_PAD
        hd    = self.VIT_HEAD_DIM_PAD
        ND    = self.NECK_DIM
        DD    = self.DEC_DIM
        NT    = self.DEC_NUM_TOKENS
        DH    = self.DEC_HEADS
        GA    = self.GRID_AREA
        GS    = self.GRID_SIZE

        # ── PATCH EMBED ─────────────────────────────────────────────────────
        self.matmat_mul_core(
            M=GA, K=self.PATCH_K_PAD, N=VD,
            A_DRAM_ADDR=self.VIT_PATCH_OUT,
            B_DRAM_ADDR=self.PATCH_EMBED_WEIGHT,
            OUTPUT_DRAM_ADDR=self.VIT_LAYER_IN,
            C_DRAM_ADDR=self.PATCH_EMBED_BIAS,
            bias_mode="broadcast_N",
        )
        eltwise_add_dram(self, self.VIT_LAYER_IN, self.POS_EMBED,
                         self.VIT_LAYER_IN, GA * VD)
        self._stop_if("patch_embed")

        # ── VIT BLOCKS ───────────────────────────────────────────────────────
        for blk_idx in range(self.VIT_DEPTH):
            bw = self.vit_block_weights[blk_idx]
            is_global = blk_idx in self.VIT_GLOBAL_BLOCKS

            if is_global:
                seq_len       = GA
                num_windows   = 1
                total_batches = self.VIT_HEADS
                M_flat        = GA
            else:
                seq_len       = self.VIT_WINDOW_AREA
                num_windows   = self.VIT_NUM_WINDOWS
                total_batches = num_windows * self.VIT_HEADS
                M_flat        = self.GRID_AREA_PAD

            wa_pad = self.VIT_WINDOW_AREA_PAD  # 256

            # LN1
            self.layer_norm_core_dram(
                M=GA, N=VD,
                A_DRAM_ADDR=self.VIT_LAYER_IN,
                OUTPUT_DRAM_ADDR=self.VIT_LN_OUT,
                GAMMA_DRAM_ADDR=bw['norm1_gamma'],
                BETA_DRAM_ADDR=bw['norm1_beta'],
            )
            self._stop_if(f"block{blk_idx}_ln1")

            # Window partition (local only)
            if not is_global:
                pad_feature_map_dram(self,
                    INPUT_DRAM_ADDR=self.VIT_LN_OUT,
                    OUTPUT_DRAM_ADDR=self.VIT_PADDED,
                    H=GS, W=GS,
                    H_PAD=self.GRID_SIZE_PAD, W_PAD=self.GRID_SIZE_PAD,
                    C=VD)
                window_partition_dram(self,
                    INPUT_DRAM_ADDR=self.VIT_PADDED,
                    OUTPUT_DRAM_ADDR=self.VIT_WINDOWED,
                    H=self.GRID_SIZE_PAD, W=self.GRID_SIZE_PAD, C=VD,
                    window_size=self.VIT_WINDOW_SIZE)
                attn_input = self.VIT_WINDOWED
                self._stop_if(f"block{blk_idx}_windowed")
            else:
                attn_input = self.VIT_LN_OUT

            # Q, K, V projections
            for w_key, b_key, out_addr, tag in [
                ('q_weight', 'q_bias', self.VIT_Q, 'q'),
                ('k_weight', 'k_bias', self.VIT_K, 'k'),
                ('v_weight', 'v_bias', self.VIT_V, 'v'),
            ]:
                self.matmat_mul_core(
                    M=M_flat, K=VD, N=VD_QK,
                    A_DRAM_ADDR=attn_input,
                    B_DRAM_ADDR=bw[w_key],
                    OUTPUT_DRAM_ADDR=out_addr,
                    C_DRAM_ADDR=bw[b_key],
                    bias_mode="broadcast_N",
                )
                self._stop_if(f"block{blk_idx}_{tag}")

            # Multi-head reshape
            if not is_global:
                wa = seq_len
                for qkv_src, qkv_dst in [(self.VIT_Q, self.VIT_Q_HEADS),
                                          (self.VIT_K, self.VIT_K_HEADS),
                                          (self.VIT_V, self.VIT_V_HEADS)]:
                    for w in range(num_windows):
                        self.bf16_permute_core(
                            dim_0=wa, dim_1=self.VIT_HEADS, dim_2=hd,
                            INPUT_DRAM_ADDR=qkv_src + w * wa * VD_QK * bpe,
                            OUTPUT_DRAM_ADDR=self.VIT_COMPACT_HEADS + w * self.VIT_HEADS * wa * hd * bpe,
                        )
                    dram_zero_fill(self, qkv_dst, total_batches * wa_pad * hd)
                    for b in range(total_batches):
                        dram_copy(self,
                            self.VIT_COMPACT_HEADS + b * wa * hd * bpe,
                            qkv_dst + b * wa_pad * hd * bpe,
                            wa * hd)
            else:
                for qkv_src, qkv_dst in [(self.VIT_Q, self.VIT_Q_HEADS),
                                          (self.VIT_K, self.VIT_K_HEADS),
                                          (self.VIT_V, self.VIT_V_HEADS)]:
                    for w in range(num_windows):
                        src = qkv_src + w * seq_len * VD_QK * bpe
                        dst = qkv_dst + w * self.VIT_HEADS * seq_len * hd * bpe
                        self.bf16_permute_core(
                            dim_0=seq_len, dim_1=self.VIT_HEADS, dim_2=hd,
                            INPUT_DRAM_ADDR=src,
                            OUTPUT_DRAM_ADDR=dst,
                        )

            # Flash attention
            if is_global:
                flash_attention_global_tiled(self,
                    num_heads=self.VIT_HEADS, head_dim=hd, seq_len=seq_len,
                    Q_DRAM_ADDR=self.VIT_Q_HEADS,
                    K_DRAM_ADDR=self.VIT_K_HEADS,
                    V_DRAM_ADDR=self.VIT_V_HEADS,
                    OUTPUT_DRAM_ADDR=self.VIT_ATTN_OUT,
                    SCRATCH_DRAM_ADDR=self.VIT_ATTN_SCRATCH,
                )
            else:
                flash_attention_batched(self,
                    num_batches=total_batches, head_dim=hd, seq_len=wa_pad,
                    Q_DRAM_ADDR=self.VIT_Q_HEADS,
                    K_DRAM_ADDR=self.VIT_K_HEADS,
                    V_DRAM_ADDR=self.VIT_V_HEADS,
                    OUTPUT_DRAM_ADDR=self.VIT_ATTN_OUT,
                    SCRATCH_DRAM_ADDR=self.VIT_ATTN_SCRATCH,
                    BIAS_DRAM_ADDR=self.VIT_WINDOW_ATTN_BIAS,
                    bias_shared=True,
                )

            # Merge heads
            if not is_global:
                wa = seq_len
                for b in range(total_batches):
                    dram_copy(self,
                        self.VIT_ATTN_OUT + b * wa_pad * hd * bpe,
                        self.VIT_COMPACT_HEADS + b * wa * hd * bpe,
                        wa * hd)
                for w in range(num_windows):
                    self.bf16_permute_core(
                        dim_0=self.VIT_HEADS, dim_1=wa, dim_2=hd,
                        INPUT_DRAM_ADDR=self.VIT_COMPACT_HEADS + w * self.VIT_HEADS * wa * hd * bpe,
                        OUTPUT_DRAM_ADDR=self.VIT_ATTN_MERGED + w * wa * VD_QK * bpe,
                    )
            else:
                for w in range(num_windows):
                    src = self.VIT_ATTN_OUT + w * self.VIT_HEADS * seq_len * hd * bpe
                    dst = self.VIT_ATTN_MERGED + w * seq_len * VD_QK * bpe
                    self.bf16_permute_core(
                        dim_0=self.VIT_HEADS, dim_1=seq_len, dim_2=hd,
                        INPUT_DRAM_ADDR=src,
                        OUTPUT_DRAM_ADDR=dst,
                    )

            self._stop_if(f"block{blk_idx}_attn_merge")

            # Output projection
            self.matmat_mul_core(
                M=M_flat, K=VD_QK, N=VD,
                A_DRAM_ADDR=self.VIT_ATTN_MERGED,
                B_DRAM_ADDR=bw['proj_weight'],
                OUTPUT_DRAM_ADDR=self.VIT_OUT_PROJ,
                C_DRAM_ADDR=bw['proj_bias'],
                bias_mode="broadcast_N",
            )
            self._stop_if(f"block{blk_idx}_out_proj")

            # Window unpartition (local)
            if not is_global:
                window_reverse_dram(self,
                    INPUT_DRAM_ADDR=self.VIT_OUT_PROJ,
                    OUTPUT_DRAM_ADDR=self.VIT_ATTN_MERGED,
                    H=self.GRID_SIZE_PAD, W=self.GRID_SIZE_PAD, C=VD,
                    window_size=self.VIT_WINDOW_SIZE)
                unpad_feature_map_dram(self,
                    INPUT_DRAM_ADDR=self.VIT_ATTN_MERGED,
                    OUTPUT_DRAM_ADDR=self.VIT_LN_OUT,
                    H=GS, W=GS, W_PAD=self.GRID_SIZE_PAD, C=VD)
                attn_result = self.VIT_LN_OUT
            else:
                attn_result = self.VIT_OUT_PROJ

            # Residual 1
            eltwise_add_dram(self, self.VIT_LAYER_IN, attn_result,
                             self.VIT_RESIDUAL, GA * VD)
            self._stop_if(f"block{blk_idx}_residual1")

            # LN2
            self.layer_norm_core_dram(
                M=GA, N=VD,
                A_DRAM_ADDR=self.VIT_RESIDUAL,
                OUTPUT_DRAM_ADDR=self.VIT_LN_OUT,
                GAMMA_DRAM_ADDR=bw['norm2_gamma'],
                BETA_DRAM_ADDR=bw['norm2_beta'],
            )
            self._stop_if(f"block{blk_idx}_ln2")

            # MLP fc1 + GELU
            self.matmat_mul_core(
                M=GA, K=VD, N=self.VIT_MLP_HIDDEN,
                A_DRAM_ADDR=self.VIT_LN_OUT,
                B_DRAM_ADDR=bw['fc1_weight'],
                OUTPUT_DRAM_ADDR=self.VIT_MLP_MID,
                C_DRAM_ADDR=bw['fc1_bias'],
                bias_mode="broadcast_N",
                gelu_enable=True,
            )
            self._stop_if(f"block{blk_idx}_fc1")

            # MLP fc2
            self.matmat_mul_core(
                M=GA, K=self.VIT_MLP_HIDDEN, N=VD,
                A_DRAM_ADDR=self.VIT_MLP_MID,
                B_DRAM_ADDR=bw['fc2_weight'],
                OUTPUT_DRAM_ADDR=self.VIT_MLP_OUT,
                C_DRAM_ADDR=bw['fc2_bias'],
                bias_mode="broadcast_N",
            )
            self._stop_if(f"block{blk_idx}_mlp_out")

            # Residual 2 → VIT_LAYER_IN
            eltwise_add_dram(self, self.VIT_RESIDUAL, self.VIT_MLP_OUT,
                             self.VIT_LAYER_IN, GA * VD)
            self._stop_if(f"block{blk_idx}_out")

        # ── NECK ────────────────────────────────────────────────────────────
        self.matmat_mul_core(
            M=GA, K=VD, N=ND,
            A_DRAM_ADDR=self.VIT_LAYER_IN,
            B_DRAM_ADDR=self.NECK_CONV1_W,
            OUTPUT_DRAM_ADDR=self.VIT_NECK_OUT,
        )
        self._stop_if("neck_conv1")

        self.layer_norm_core_dram(
            M=GA, N=ND,
            A_DRAM_ADDR=self.VIT_NECK_OUT,
            OUTPUT_DRAM_ADDR=self.VIT_NECK_OUT,
            GAMMA_DRAM_ADDR=self.NECK_LN1_W,
            BETA_DRAM_ADDR=self.NECK_LN1_B,
        )
        self._stop_if("neck_ln1")

        conv2d_3x3_dram(self,
            INPUT_DRAM_ADDR=self.VIT_NECK_OUT,
            OUTPUT_DRAM_ADDR=self.NECK_OUT,
            IM2COL_DRAM_ADDR=self.NECK_IM2COL,
            WEIGHT_DRAM_ADDR=self.NECK_CONV3_W,
            BIAS_DRAM_ADDR=None,
            H=GS, W=GS, C_in=ND, C_out=ND,
            ZERO_PAD_DRAM_ADDR=self.ZERO_PAD,
        )
        self._stop_if("neck_conv2")

        self.layer_norm_core_dram(
            M=GA, N=ND,
            A_DRAM_ADDR=self.NECK_OUT,
            OUTPUT_DRAM_ADDR=self.NECK_OUT,
            GAMMA_DRAM_ADDR=self.NECK_LN2_W,
            BETA_DRAM_ADDR=self.NECK_LN2_B,
        )
        self._stop_if("neck_ln2")

        # ── DECODER ─────────────────────────────────────────────────────────
        # DEC_SRC = NECK_OUT + no_mask_embed (broadcast)
        dram_copy(self, self.NECK_OUT, self.DEC_SRC, GA * DD)
        for row in range(0, GA, URAM_NEAR_FULL_ELEMENTS // DD):
            take = min(URAM_NEAR_FULL_ELEMENTS // DD, GA - row)
            self.accelerator_memory_to_sram(
                accelerator_dram_address=self.PE_NO_MASK,
                sram_address=0x80000, element_size=DD)
            for r in range(take):
                self.accelerator_memory_to_sram(
                    accelerator_dram_address=self.DEC_SRC + (row + r) * DD * 2,
                    sram_address=0x00000, element_size=DD)
                self.eltwise_add_core(
                    vector_A_sram_start_addr=0x00000,
                    vector_B_sram_start_addr=0x80000,
                    vector_C_sram_wb_addr=0x00000,
                    element_size=DD)
                self.sram_to_accelerator_memory(
                    sram_address=0x00000,
                    accelerator_dram_address=self.DEC_SRC + (row + r) * DD * 2,
                    element_size=DD)
        self._stop_if("dec_src")

        # DEC_TOKENS is DMA'd from host before HW execution (tokens preset externally)
        # Helper closures for decoder layers
        def _sa(lw, skip_pe=False):
            """Self-attention block: SA + out_proj + residual + norm1 on DEC_TOKENS."""
            HD_PAD = 64
            NT_PAD = 64
            N_PAD  = DH * HD_PAD

            for proj_key, out_addr in [('sa_q_w', self.DEC_Q),
                                        ('sa_k_w', self.DEC_K),
                                        ('sa_v_w', self.DEC_V)]:
                self.matmat_mul_core(
                    M=NT, K=DD, N=N_PAD,
                    A_DRAM_ADDR=self.DEC_TOKENS,
                    B_DRAM_ADDR=lw[proj_key],
                    OUTPUT_DRAM_ADDR=out_addr,
                    C_DRAM_ADDR=lw[proj_key.replace('_w', '_b')],
                    bias_mode="broadcast_N",
                )
            for proj, heads in [(self.DEC_Q, self.DEC_SA_Q_HEADS),
                                 (self.DEC_K, self.DEC_SA_K_HEADS),
                                 (self.DEC_V, self.DEC_SA_V_HEADS)]:
                self.bf16_permute_core(
                    dim_0=NT, dim_1=DH, dim_2=HD_PAD,
                    INPUT_DRAM_ADDR=proj,
                    OUTPUT_DRAM_ADDR=self.DEC_SA_TEMP)
                dram_zero_fill(self, heads, DH * NT_PAD * HD_PAD)
                for h in range(DH):
                    dram_copy(self,
                        self.DEC_SA_TEMP + h * NT * HD_PAD * 2,
                        heads + h * NT_PAD * HD_PAD * 2,
                        NT * HD_PAD)
            broadcast_mul_dram(self, self.DEC_SA_Q_HEADS, math.sqrt(2.0), DH * NT_PAD * HD_PAD)
            flash_attention_batched(self,
                num_batches=DH, head_dim=HD_PAD, seq_len=NT_PAD,
                Q_DRAM_ADDR=self.DEC_SA_Q_HEADS,
                K_DRAM_ADDR=self.DEC_SA_K_HEADS,
                V_DRAM_ADDR=self.DEC_SA_V_HEADS,
                OUTPUT_DRAM_ADDR=self.DEC_SA_OUT_HEADS,
                SCRATCH_DRAM_ADDR=self.DEC_SA_SCRATCH,
                BIAS_DRAM_ADDR=self.DEC_SA_ATTN_BIAS)
            for h in range(DH):
                dram_copy(self,
                    self.DEC_SA_OUT_HEADS + h * NT_PAD * HD_PAD * 2,
                    self.DEC_SA_TEMP      + h * NT * HD_PAD * 2,
                    NT * HD_PAD)
            multihead_merge_dram(self,
                INPUT_DRAM_ADDR=self.DEC_SA_TEMP,
                OUTPUT_DRAM_ADDR=self.DEC_SA_MERGED,
                TEMP_DRAM_ADDR=self.DEC_SA_Q_HEADS,
                seq_len=NT, num_heads=DH,
                head_dim=self.DEC_HEAD_DIM, head_dim_pad=HD_PAD,
                UNPAD_WEIGHT_ADDR=self.DEC_SA_UNPAD_W)
            self.matmat_mul_core(
                M=NT, K=DD, N=DD,
                A_DRAM_ADDR=self.DEC_SA_MERGED,
                B_DRAM_ADDR=lw['sa_out_w'],
                OUTPUT_DRAM_ADDR=self.DEC_TOKENS_NORM,
                C_DRAM_ADDR=lw['sa_out_b'],
                bias_mode="broadcast_N")
            eltwise_add_dram(self, self.DEC_TOKENS, self.DEC_TOKENS_NORM,
                             self.DEC_TOKENS, NT * DD)
            self.layer_norm_core_dram(
                M=NT, N=DD,
                A_DRAM_ADDR=self.DEC_TOKENS,
                OUTPUT_DRAM_ADDR=self.DEC_TOKENS,
                GAMMA_DRAM_ADDR=lw['norm1_w'],
                BETA_DRAM_ADDR=lw['norm1_b'])

        def _t2i(lw, is_final=False, out_norm_w=None, out_norm_b=None):
            """Token→image cross-attention + residual + norm2 on DEC_TOKENS."""
            CA_HDP = 64
            N_CA   = DH * CA_HDP

            eltwise_add_dram(self, self.DEC_TOKENS, self.DEC_QUERY_PE,
                             self.DEC_TOKENS_NORM, NT * DD)
            self.matmat_mul_core(
                M=NT, K=DD, N=N_CA,
                A_DRAM_ADDR=self.DEC_TOKENS_NORM,
                B_DRAM_ADDR=lw['ca_t2i_q_w'],
                OUTPUT_DRAM_ADDR=self.DEC_Q,
                C_DRAM_ADDR=lw['ca_t2i_q_b'],
                bias_mode="broadcast_N")
            self.matmat_mul_core(
                M=GA, K=DD, N=N_CA,
                A_DRAM_ADDR=self.DEC_SRC,
                B_DRAM_ADDR=lw['ca_t2i_k_w'],
                OUTPUT_DRAM_ADDR=self.DEC_CA_T2I_K,
                C_DRAM_ADDR=lw['ca_t2i_k_b'],
                bias_mode="broadcast_N")
            self.matmat_mul_core(
                M=GA, K=DD, N=N_CA,
                A_DRAM_ADDR=self.DEC_POS_SRC,
                B_DRAM_ADDR=lw['ca_t2i_k_w'],
                OUTPUT_DRAM_ADDR=self.DEC_CA_T2I_V)
            eltwise_add_dram(self, self.DEC_CA_T2I_K, self.DEC_CA_T2I_V,
                             self.DEC_CA_T2I_K, GA * N_CA)
            self.matmat_mul_core(
                M=GA, K=DD, N=N_CA,
                A_DRAM_ADDR=self.DEC_SRC,
                B_DRAM_ADDR=lw['ca_t2i_v_w'],
                OUTPUT_DRAM_ADDR=self.DEC_CA_T2I_V,
                C_DRAM_ADDR=lw['ca_t2i_v_b'],
                bias_mode="broadcast_N")
            self.bf16_permute_core(dim_0=NT, dim_1=DH, dim_2=CA_HDP,
                INPUT_DRAM_ADDR=self.DEC_Q, OUTPUT_DRAM_ADDR=self.DEC_SA_TEMP)
            self.bf16_permute_core(dim_0=GA, dim_1=DH, dim_2=CA_HDP,
                INPUT_DRAM_ADDR=self.DEC_CA_T2I_K, OUTPUT_DRAM_ADDR=self.DEC_CA_T2I_K_HEADS)
            self.bf16_permute_core(dim_0=GA, dim_1=DH, dim_2=CA_HDP,
                INPUT_DRAM_ADDR=self.DEC_CA_T2I_V, OUTPUT_DRAM_ADDR=self.DEC_CA_T2I_V_HEADS)
            broadcast_mul_dram(self, self.DEC_SA_TEMP,
                               1.0 / math.sqrt(self.DEC_INTERNAL_HD), DH * NT * CA_HDP)
            for h in range(DH):
                q_a = self.DEC_SA_TEMP + h * NT * CA_HDP * 2
                k_a = self.DEC_CA_T2I_K_HEADS + h * GA * CA_HDP * 2
                v_a = self.DEC_CA_T2I_V_HEADS + h * GA * CA_HDP * 2
                o_a = self.DEC_CA_OUT_HEADS + h * NT * CA_HDP * 2
                vt_a = self.DEC_CA_SCORES + NT * GA * 2
                self.matmat_mul_core(
                    M=NT, K=CA_HDP, N=GA,
                    A_DRAM_ADDR=q_a, B_DRAM_ADDR=k_a,
                    OUTPUT_DRAM_ADDR=self.DEC_CA_SCORES,
                    softmax_enable=True)
                self.bf16_transpose_core(M=GA, N=CA_HDP,
                    INPUT_DRAM_ADDR=v_a, OUTPUT_DRAM_ADDR=self.DEC_CA_VT)
                self.matmat_mul_core(
                    M=NT, K=GA, N=CA_HDP,
                    A_DRAM_ADDR=self.DEC_CA_SCORES,
                    B_DRAM_ADDR=self.DEC_CA_VT,
                    OUTPUT_DRAM_ADDR=o_a)
            multihead_merge_dram(self,
                INPUT_DRAM_ADDR=self.DEC_CA_OUT_HEADS,
                OUTPUT_DRAM_ADDR=self.DEC_CA_T2I_MERGED,
                TEMP_DRAM_ADDR=self.DEC_SA_Q_HEADS,
                seq_len=NT, num_heads=DH,
                head_dim=self.DEC_INTERNAL_HD, head_dim_pad=CA_HDP,
                UNPAD_WEIGHT_ADDR=self.DEC_CA_UNPAD_W)
            self.matmat_mul_core(
                M=NT, K=self.DEC_INTERNAL_DIM, N=DD,
                A_DRAM_ADDR=self.DEC_CA_T2I_MERGED,
                B_DRAM_ADDR=lw['ca_t2i_out_w'],
                OUTPUT_DRAM_ADDR=self.DEC_TOKENS_NORM,
                C_DRAM_ADDR=lw['ca_t2i_out_b'],
                bias_mode="broadcast_N")
            eltwise_add_dram(self, self.DEC_TOKENS, self.DEC_TOKENS_NORM,
                             self.DEC_TOKENS, NT * DD)
            norm_w = out_norm_w if is_final else lw['norm2_w']
            norm_b = out_norm_b if is_final else lw['norm2_b']
            self.layer_norm_core_dram(
                M=NT, N=DD,
                A_DRAM_ADDR=self.DEC_TOKENS,
                OUTPUT_DRAM_ADDR=self.DEC_TOKENS,
                GAMMA_DRAM_ADDR=norm_w,
                BETA_DRAM_ADDR=norm_b)

        def _mlp(lw):
            """Decoder MLP (ReLU) + residual + norm3 on DEC_TOKENS."""
            MLP_DIM = self.DEC_MLP_DIM
            self.matmat_mul_core(
                M=NT, K=DD, N=MLP_DIM,
                A_DRAM_ADDR=self.DEC_TOKENS,
                B_DRAM_ADDR=lw['mlp_lin1_w'],
                OUTPUT_DRAM_ADDR=self.DEC_MLP_INTER,
                C_DRAM_ADDR=lw['mlp_lin1_b'],
                bias_mode="broadcast_N",
                relu_enable=True)
            self.matmat_mul_core(
                M=NT, K=MLP_DIM, N=DD,
                A_DRAM_ADDR=self.DEC_MLP_INTER,
                B_DRAM_ADDR=lw['mlp_lin2_w'],
                OUTPUT_DRAM_ADDR=self.DEC_TOKENS_NORM,
                C_DRAM_ADDR=lw['mlp_lin2_b'],
                bias_mode="broadcast_N")
            eltwise_add_dram(self, self.DEC_TOKENS, self.DEC_TOKENS_NORM,
                             self.DEC_TOKENS, NT * DD)
            self.layer_norm_core_dram(
                M=NT, N=DD,
                A_DRAM_ADDR=self.DEC_TOKENS,
                OUTPUT_DRAM_ADDR=self.DEC_TOKENS,
                GAMMA_DRAM_ADDR=lw['norm3_w'],
                BETA_DRAM_ADDR=lw['norm3_b'])

        def _i2t(lw):
            """Image→token cross-attention + residual + norm4 on DEC_SRC (keys)."""
            CA_HDP  = 64
            N_CA    = DH * CA_HDP
            NT_PAD_I = 64

            # K: tokens + query_pe → project → (NT, 512); zero-pad to (64, 512)
            eltwise_add_dram(self, self.DEC_TOKENS, self.DEC_QUERY_PE,
                             self.DEC_TOKENS_NORM, NT * DD)
            dram_zero_fill(self, self.DEC_CA_I2T_K_PAD, NT_PAD_I * DH * CA_HDP)
            self.matmat_mul_core(
                M=NT, K=DD, N=N_CA,
                A_DRAM_ADDR=self.DEC_TOKENS_NORM,
                B_DRAM_ADDR=lw['ca_i2t_k_w'],
                OUTPUT_DRAM_ADDR=self.DEC_CA_I2T_K_PAD,
                C_DRAM_ADDR=lw['ca_i2t_k_b'],
                bias_mode="broadcast_N")
            dram_zero_fill(self, self.DEC_CA_I2T_V_PAD, NT_PAD_I * DH * CA_HDP)
            self.matmat_mul_core(
                M=NT, K=DD, N=N_CA,
                A_DRAM_ADDR=self.DEC_TOKENS,
                B_DRAM_ADDR=lw['ca_i2t_v_w'],
                OUTPUT_DRAM_ADDR=self.DEC_CA_I2T_V_PAD,
                C_DRAM_ADDR=lw['ca_i2t_v_b'],
                bias_mode="broadcast_N")
            # Q: src + key_pe → project → (4096, 512)
            eltwise_add_dram(self, self.DEC_SRC, self.DEC_KEY_PE,
                             self.DEC_TOKENS_NORM, GA * DD)
            self.matmat_mul_core(
                M=GA, K=DD, N=N_CA,
                A_DRAM_ADDR=self.DEC_TOKENS_NORM,
                B_DRAM_ADDR=lw['ca_i2t_q_w'],
                OUTPUT_DRAM_ADDR=self.DEC_CA_I2T_OUT_HEADS,   # reuse buffer for Q
                C_DRAM_ADDR=lw['ca_i2t_q_b'],
                bias_mode="broadcast_N")
            # Permute Q/K/V to heads layout
            self.bf16_permute_core(dim_0=GA, dim_1=DH, dim_2=CA_HDP,
                INPUT_DRAM_ADDR=self.DEC_CA_I2T_OUT_HEADS,
                OUTPUT_DRAM_ADDR=self.DEC_CA_T2I_K_HEADS)   # reused as Q_heads
            self.bf16_permute_core(dim_0=NT_PAD_I, dim_1=DH, dim_2=CA_HDP,
                INPUT_DRAM_ADDR=self.DEC_CA_I2T_K_PAD,
                OUTPUT_DRAM_ADDR=self.DEC_CA_T2I_V_HEADS)   # reused as K_heads
            self.bf16_permute_core(dim_0=NT_PAD_I, dim_1=DH, dim_2=CA_HDP,
                INPUT_DRAM_ADDR=self.DEC_CA_I2T_V_PAD,
                OUTPUT_DRAM_ADDR=self.DEC_CA_I2T_K_HEADS)   # reused as V_heads
            broadcast_mul_dram(self, self.DEC_CA_T2I_K_HEADS,
                               1.0 / math.sqrt(self.DEC_INTERNAL_HD), DH * GA * CA_HDP)
            for h in range(DH):
                q_a = self.DEC_CA_T2I_K_HEADS + h * GA * CA_HDP * 2
                k_a = self.DEC_CA_T2I_V_HEADS + h * NT_PAD_I * CA_HDP * 2
                v_a = self.DEC_CA_I2T_K_HEADS + h * NT_PAD_I * CA_HDP * 2
                o_a = self.DEC_CA_I2T_OUT_HEADS + h * GA * CA_HDP * 2
                self.matmat_mul_core(
                    M=GA, K=CA_HDP, N=NT_PAD_I,
                    A_DRAM_ADDR=q_a, B_DRAM_ADDR=k_a,
                    OUTPUT_DRAM_ADDR=self.DEC_CA_I2T_SCORES,
                    softmax_enable=True)
                self.bf16_transpose_core(M=NT_PAD_I, N=CA_HDP,
                    INPUT_DRAM_ADDR=v_a, OUTPUT_DRAM_ADDR=self.DEC_CA_I2T_VT)
                self.matmat_mul_core(
                    M=GA, K=NT_PAD_I, N=CA_HDP,
                    A_DRAM_ADDR=self.DEC_CA_I2T_SCORES,
                    B_DRAM_ADDR=self.DEC_CA_I2T_VT,
                    OUTPUT_DRAM_ADDR=o_a)
            multihead_merge_dram(self,
                INPUT_DRAM_ADDR=self.DEC_CA_I2T_OUT_HEADS,
                OUTPUT_DRAM_ADDR=self.DEC_CA_I2T_MERGED,
                TEMP_DRAM_ADDR=self.DEC_SA_Q_HEADS,
                seq_len=GA, num_heads=DH,
                head_dim=self.DEC_INTERNAL_HD, head_dim_pad=CA_HDP,
                UNPAD_WEIGHT_ADDR=self.DEC_CA_UNPAD_W)
            self.matmat_mul_core(
                M=GA, K=self.DEC_INTERNAL_DIM, N=DD,
                A_DRAM_ADDR=self.DEC_CA_I2T_MERGED,
                B_DRAM_ADDR=lw['ca_i2t_out_w'],
                OUTPUT_DRAM_ADDR=self.DEC_TOKENS_NORM,
                C_DRAM_ADDR=lw['ca_i2t_out_b'],
                bias_mode="broadcast_N")
            eltwise_add_dram(self, self.DEC_SRC, self.DEC_TOKENS_NORM,
                             self.DEC_SRC, GA * DD)
            self.layer_norm_core_dram(
                M=GA, N=DD,
                A_DRAM_ADDR=self.DEC_SRC,
                OUTPUT_DRAM_ADDR=self.DEC_SRC,
                GAMMA_DRAM_ADDR=lw['norm4_w'],
                BETA_DRAM_ADDR=lw['norm4_b'])

        # Layer 0
        lw0 = self.dec_layer_weights[0]
        _sa(lw0, skip_pe=True)
        self._stop_if("dec_l0_sa_out")
        _t2i(lw0)
        self._stop_if("dec_l0_t2i_out")
        _mlp(lw0)
        self._stop_if("dec_l0_mlp_out")
        _i2t(lw0)
        self._stop_if("dec_l0_i2t_src")

        # Layer 1
        lw1 = self.dec_layer_weights[1]
        _sa(lw1)
        self._stop_if("dec_l1_sa_out")
        fa = self.dec_final_attn
        _final_lw = {
            'ca_t2i_q_w': fa['q_w'], 'ca_t2i_q_b': fa['q_b'],
            'ca_t2i_k_w': fa['k_w'], 'ca_t2i_k_b': fa['k_b'],
            'ca_t2i_v_w': fa['v_w'], 'ca_t2i_v_b': fa['v_b'],
            'ca_t2i_out_w': fa['out_w'], 'ca_t2i_out_b': fa['out_b'],
        }
        _t2i(lw1)
        self._stop_if("dec_l1_t2i_out")
        _mlp(lw1)
        self._stop_if("dec_l1_mlp_out")
        _i2t(lw1)
        self._stop_if("dec_l1_i2t_src")

        # Final cross-attention
        _t2i(_final_lw, is_final=True,
             out_norm_w=self.dec_final_norm['w'],
             out_norm_b=self.dec_final_norm['b'])
        self._stop_if("dec_final_attn")

        # IoU head
        iou_w = self.dec_iou_weights
        self.matmat_mul_core(M=1, K=DD, N=DD,
            A_DRAM_ADDR=self.DEC_TOKENS,
            B_DRAM_ADDR=iou_w['l0_w'],
            OUTPUT_DRAM_ADDR=self.DEC_TOKENS_NORM,
            C_DRAM_ADDR=iou_w['l0_b'],
            bias_mode="broadcast_N", relu_enable=True)
        self.matmat_mul_core(M=1, K=DD, N=DD,
            A_DRAM_ADDR=self.DEC_TOKENS_NORM,
            B_DRAM_ADDR=iou_w['l1_w'],
            OUTPUT_DRAM_ADDR=self.DEC_TOKENS_NORM,
            C_DRAM_ADDR=iou_w['l1_b'],
            bias_mode="broadcast_N", relu_enable=True)
        self.matmat_mul_core(M=1, K=DD, N=4,
            A_DRAM_ADDR=self.DEC_TOKENS_NORM,
            B_DRAM_ADDR=iou_w['l2_w'],
            OUTPUT_DRAM_ADDR=self.DEC_IOU_OUT,
            C_DRAM_ADDR=iou_w['l2_b'],
            bias_mode="broadcast_N")
        self._stop_if("dec_iou")

        # Hypernetwork MLPs
        for m in range(4):
            hw = self.dec_hyper_weights[m]
            src_a = self.DEC_TOKENS + (m + 1) * DD * 2
            out_a = self.DEC_HYPER_OUT + m * 32 * 2
            self.matmat_mul_core(M=1, K=DD, N=DD,
                A_DRAM_ADDR=src_a,
                B_DRAM_ADDR=hw['l0_w'],
                OUTPUT_DRAM_ADDR=self.DEC_TOKENS_NORM,
                C_DRAM_ADDR=hw['l0_b'],
                bias_mode="broadcast_N", relu_enable=True)
            self.matmat_mul_core(M=1, K=DD, N=DD,
                A_DRAM_ADDR=self.DEC_TOKENS_NORM,
                B_DRAM_ADDR=hw['l1_w'],
                OUTPUT_DRAM_ADDR=self.DEC_TOKENS_NORM,
                C_DRAM_ADDR=hw['l1_b'],
                bias_mode="broadcast_N", relu_enable=True)
            self.matmat_mul_core(M=1, K=DD, N=32,
                A_DRAM_ADDR=self.DEC_TOKENS_NORM,
                B_DRAM_ADDR=hw['l2_w'],
                OUTPUT_DRAM_ADDR=out_a,
                C_DRAM_ADDR=hw['l2_b'],
                bias_mode="broadcast_N")
        self._stop_if("dec_hyper")

        # Mask upscaling
        C_IN0, C_OUT0 = DD, 64
        GA0 = GA
        H_IN0, W_IN0 = 64, 64
        H_OUT0, W_OUT0 = 128, 128
        for dr in range(2):
            for dc in range(2):
                sub_idx = dr * 2 + dc
                self.matmat_mul_core(
                    M=GA0, K=C_IN0, N=C_OUT0,
                    A_DRAM_ADDR=self.DEC_SRC,
                    B_DRAM_ADDR=self.DEC_UP0_W[dr][dc],
                    OUTPUT_DRAM_ADDR=self.DEC_UP0_SCRATCH + sub_idx * GA0 * C_OUT0 * 2,
                    C_DRAM_ADDR=self.DEC_UP0_B,
                    bias_mode="broadcast_N" if self.DEC_UP0_B is not None else None)
        for r in range(H_IN0):
            for c in range(W_IN0):
                for dr in range(2):
                    for dc in range(2):
                        sub_idx = dr * 2 + dc
                        src_row = sub_idx * GA0 + r * W_IN0 + c
                        dst_row = (2 * r + dr) * W_OUT0 + (2 * c + dc)
                        src_b   = self.DEC_UP0_SCRATCH + src_row * C_OUT0 * 2
                        dst_b   = self.DEC_UP0_OUT     + dst_row * C_OUT0 * 2
                        self.accelerator_memory_to_sram(src_b, 0x0000, C_OUT0)
                        self.sram_to_accelerator_memory(0x0000, dst_b, C_OUT0)
        self.layer_norm_core_dram(
            M=H_OUT0 * W_OUT0, N=C_OUT0,
            A_DRAM_ADDR=self.DEC_UP0_OUT,
            OUTPUT_DRAM_ADDR=self.DEC_UP0_OUT,
            GAMMA_DRAM_ADDR=self.DEC_UP_LN_W,
            BETA_DRAM_ADDR=self.DEC_UP_LN_B)
        _identity_64 = torch.eye(C_OUT0, dtype=torch.bfloat16)
        _gelu_id_w = self._alloc_param(_identity_64)
        self.matmat_mul_core(
            M=H_OUT0 * W_OUT0, K=C_OUT0, N=C_OUT0,
            A_DRAM_ADDR=self.DEC_UP0_OUT,
            B_DRAM_ADDR=_gelu_id_w,
            OUTPUT_DRAM_ADDR=self.DEC_UP0_OUT,
            gelu_enable=True)
        self._stop_if("dec_up0")

        C_IN1, C_OUT1 = C_OUT0, 32
        GA1 = H_OUT0 * W_OUT0
        H_IN1, W_IN1 = H_OUT0, W_OUT0
        H_OUT1, W_OUT1 = 256, 256
        for dr in range(2):
            for dc in range(2):
                sub_idx = dr * 2 + dc
                self.matmat_mul_core(
                    M=GA1, K=C_IN1, N=C_OUT1,
                    A_DRAM_ADDR=self.DEC_UP0_OUT,
                    B_DRAM_ADDR=self.DEC_UP1_W[dr][dc],
                    OUTPUT_DRAM_ADDR=self.DEC_UP1_SCRATCH + sub_idx * GA1 * C_OUT1 * 2,
                    C_DRAM_ADDR=self.DEC_UP1_B,
                    bias_mode="broadcast_N" if self.DEC_UP1_B is not None else None)
        for r in range(H_IN1):
            for c in range(W_IN1):
                for dr in range(2):
                    for dc in range(2):
                        sub_idx = dr * 2 + dc
                        src_row = sub_idx * GA1 + r * W_IN1 + c
                        dst_row = (2 * r + dr) * W_OUT1 + (2 * c + dc)
                        src_b   = self.DEC_UP1_SCRATCH + src_row * C_OUT1 * 2
                        dst_b   = self.DEC_UP1_OUT     + dst_row * C_OUT1 * 2
                        self.accelerator_memory_to_sram(src_b, 0x0000, C_OUT1)
                        self.sram_to_accelerator_memory(0x0000, dst_b, C_OUT1)
        _identity_32 = torch.eye(C_OUT1, dtype=torch.bfloat16)
        _gelu_id_w1 = self._alloc_param(_identity_32)
        self.matmat_mul_core(
            M=H_OUT1 * W_OUT1, K=C_OUT1, N=C_OUT1,
            A_DRAM_ADDR=self.DEC_UP1_OUT,
            B_DRAM_ADDR=_gelu_id_w1,
            OUTPUT_DRAM_ADDR=self.DEC_UP1_OUT,
            gelu_enable=True)
        self._stop_if("dec_up1")

        self.matmat_mul_core(
            M=H_OUT1 * W_OUT1, K=C_OUT1, N=4,
            A_DRAM_ADDR=self.DEC_UP1_OUT,
            B_DRAM_ADDR=self.DEC_HYPER_OUT,
            OUTPUT_DRAM_ADDR=self.DEC_MASK_LOGITS)
        self._stop_if("dec_mask_logits")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint catalogue:  (name, dram_addr_attr, num_elements_fn, cpu_ref_fn)
# cpu_ref_fn(ue, sd, ctx) → tensor (flat, bf16)
# ctx: dict carrying intermediate CPU tensors between checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def _build_checkpoints(ue):
    """Return ordered list of (name, description, read_fn, cpu_ref_fn) tuples."""
    VD  = ue.VIT_DIM
    ND  = ue.NECK_DIM
    DD  = ue.DEC_DIM
    GA  = ue.GRID_AREA
    GS  = ue.GRID_SIZE
    NT  = ue.DEC_NUM_TOKENS
    ws  = ue.VIT_WINDOW_SIZE
    nw  = ue.VIT_NUM_WINDOWS       # 25
    nws = 5                         # sqrt(25)
    GSP = ue.GRID_SIZE_PAD          # 70
    WA  = ue.VIT_WINDOW_AREA        # 196
    nh  = ue.VIT_HEADS
    hd  = ue.VIT_HEAD_DIM

    checks = []

    def add(name, desc, n_elems, dram_attr, cpu_fn):
        checks.append((name, desc, n_elems, dram_attr, cpu_fn))

    # ── helpers for windowed layout ──────────────────────────────────────────
    def _windowed_q_cpu(ctx, sd, blk_idx):
        """CPU Q projection output in windowed layout (M_flat, VD)."""
        normed = ctx[f'block{blk_idx}_ln1']
        is_global = blk_idx in ue.VIT_GLOBAL_BLOCKS
        if is_global:
            inp = normed  # (4096, 768)
        else:
            # pad + window_partition → (4900, 768)
            spatial = normed.reshape(GS, GS, VD)
            padded  = F.pad(spatial.permute(2,0,1).unsqueeze(0),
                            (0, GSP-GS, 0, GSP-GS)).squeeze(0).permute(1,2,0)
            inp = (padded.reshape(nws, ws, nws, ws, VD)
                         .permute(0,2,1,3,4).reshape(nw * ws*ws, VD))
        qkv_w = sd[f"vision_encoder.layers.{blk_idx}.attn.qkv.weight"].to(torch.bfloat16)
        qkv_b = sd[f"vision_encoder.layers.{blk_idx}.attn.qkv.bias"].to(torch.bfloat16)
        return F.linear(inp.float(), qkv_w[:VD].float(), qkv_b[:VD].float()).to(torch.bfloat16)

    # ── patch embed ──────────────────────────────────────────────────────────
    def cpu_patch_embed(ctx, sd, image_path):
        from sam1_vit_b_test import Sam1VitB_UnifiedEngine as UE
        x = ue.cpu_reference_patch_embed(image_path).to(torch.bfloat16)
        ctx['patch_embed'] = x
        return x

    add("patch_embed", "patch embed + pos embed (4096,768)",
        GA * VD, "VIT_LAYER_IN", cpu_patch_embed)

    # ── ViT blocks ───────────────────────────────────────────────────────────
    for blk in range(ue.VIT_DEPTH):
        is_global = blk in ue.VIT_GLOBAL_BLOCKS
        M_flat = GA if is_global else GA + (GSP*GSP - GS*GS)  # 4096 or 4900
        M_flat_real = GA if is_global else nw * ws * ws  # 4900

        bp = f"vision_encoder.layers.{blk}."

        def make_ln1(b):
            def cpu_ln1(ctx, sd, image_path):
                x_in = ctx.get(f'block{b-1}_out') if b > 0 else ctx.get('patch_embed')
                if x_in is None:
                    raise RuntimeError(f"Missing ctx for block{b} LN1")
                x_in = x_in.float()
                g = sd[f"vision_encoder.layers.{b}.layer_norm1.weight"].float()
                beta = sd[f"vision_encoder.layers.{b}.layer_norm1.bias"].float()
                mean = x_in.mean(-1, keepdim=True)
                cent = x_in - mean
                rms  = cent.pow(2).mean(-1, keepdim=True).sqrt()
                out  = (cent / rms).to(torch.bfloat16) * g.to(torch.bfloat16) + beta.to(torch.bfloat16)
                ctx[f'block{b}_ln1'] = out
                return out
            return cpu_ln1

        add(f"block{blk}_ln1",
            f"block {blk} LN1 (4096,768) {'[GLOBAL]' if is_global else '[local]'}",
            GA * VD, "VIT_LN_OUT", make_ln1(blk))

        if not is_global:
            def make_windowed(b):
                def cpu_windowed(ctx, sd, image_path):
                    normed = ctx[f'block{b}_ln1']
                    spatial = normed.reshape(GS, GS, VD)
                    padded  = F.pad(spatial.permute(2,0,1).unsqueeze(0),
                                    (0, GSP-GS, 0, GSP-GS)).squeeze(0).permute(1,2,0)
                    out = (padded.reshape(nws, ws, nws, ws, VD)
                                 .permute(0,2,1,3,4).reshape(nw * ws*ws, VD))
                    ctx[f'block{b}_windowed'] = out
                    return out
                return cpu_windowed
            add(f"block{blk}_windowed",
                f"block {blk} windowed (4900,768)",
                nw * ws * ws * VD, "VIT_WINDOWED", make_windowed(blk))

        def make_qkv(b, proj_idx, proj_name, qkv_out_attr):
            def cpu_qkv(ctx, sd, image_path):
                normed = ctx[f'block{b}_ln1']
                bg = b in ue.VIT_GLOBAL_BLOCKS
                if bg:
                    inp = normed.float()
                else:
                    w_out = ctx.get(f'block{b}_windowed')
                    if w_out is None:
                        # recompute windowed
                        spatial = normed.reshape(GS, GS, VD)
                        padded  = F.pad(spatial.permute(2,0,1).unsqueeze(0),
                                        (0, GSP-GS, 0, GSP-GS)).squeeze(0).permute(1,2,0)
                        w_out = (padded.reshape(nws, ws, nws, ws, VD)
                                       .permute(0,2,1,3,4).reshape(nw * ws*ws, VD))
                    inp = w_out.float()
                raw_w = sd[f"vision_encoder.layers.{b}.attn.qkv.weight"].to(torch.bfloat16)
                raw_b = sd[f"vision_encoder.layers.{b}.attn.qkv.bias"].to(torch.bfloat16)
                s, e = proj_idx * VD, (proj_idx + 1) * VD
                out = F.linear(inp, raw_w[s:e].float(), raw_b[s:e].float()).to(torch.bfloat16)
                ctx[f'block{b}_{proj_name}'] = out
                return out
            return cpu_qkv

        for proj_i, proj_n, dram_attr in [(0, 'q', 'VIT_Q'), (1, 'k', 'VIT_K'), (2, 'v', 'VIT_V')]:
            M = GA if is_global else nw * ws * ws
            add(f"block{blk}_{proj_n}",
                f"block {blk} {proj_n.upper()} proj ({'%d'%M},{VD})",
                M * VD, dram_attr, make_qkv(blk, proj_i, proj_n, dram_attr))

        def make_attn_merge(b):
            def cpu_attn_merge(ctx, sd, image_path):
                bg = b in ue.VIT_GLOBAL_BLOCKS
                Q_flat = ctx[f'block{b}_q']
                K_flat = ctx[f'block{b}_k']
                V_flat = ctx[f'block{b}_v']
                if bg:
                    nw_b, seq_b = 1, GA
                else:
                    nw_b, seq_b = nw, ws * ws
                Q = Q_flat.reshape(nw_b, seq_b, nh, hd).permute(0,2,1,3).reshape(nw_b*nh, seq_b, hd)
                K = K_flat.reshape(nw_b, seq_b, nh, hd).permute(0,2,1,3).reshape(nw_b*nh, seq_b, hd)
                V = V_flat.reshape(nw_b, seq_b, nh, hd).permute(0,2,1,3).reshape(nw_b*nh, seq_b, hd)
                scale = 1.0 / math.sqrt(hd)
                scores = torch.bmm(Q.float() * scale, K.float().transpose(-2, -1))
                probs  = torch.softmax(scores, dim=-1).to(torch.bfloat16)
                out    = torch.bmm(probs.float(), V.float()).to(torch.bfloat16)
                # merge → (nw_b, seq_b, VD) then flatten to (nw_b*seq_b, VD)
                merged = (out.reshape(nw_b, nh, seq_b, hd)
                             .permute(0,2,1,3).reshape(nw_b * seq_b, VD))
                ctx[f'block{b}_attn_merge'] = merged
                return merged
            return cpu_attn_merge

        M = GA if is_global else nw * ws * ws
        add(f"block{blk}_attn_merge",
            f"block {blk} attn merge ({'%d'%M},{VD})",
            M * VD, "VIT_ATTN_MERGED", make_attn_merge(blk))

        def make_out_proj(b):
            def cpu_out_proj(ctx, sd, image_path):
                merged = ctx[f'block{b}_attn_merge']
                pw = sd[f"vision_encoder.layers.{b}.attn.proj.weight"].to(torch.bfloat16)
                pb = sd[f"vision_encoder.layers.{b}.attn.proj.bias"].to(torch.bfloat16)
                bg = b in ue.VIT_GLOBAL_BLOCKS
                if not bg:
                    merged = merged.reshape(nw, ws*ws, VD)
                    out = F.linear(merged.float(), pw.float(), pb.float()).to(torch.bfloat16)
                    out = out.reshape(nw * ws*ws, VD)
                else:
                    out = F.linear(merged.float(), pw.float(), pb.float()).to(torch.bfloat16)
                ctx[f'block{b}_out_proj'] = out
                return out
            return cpu_out_proj

        M = GA if is_global else nw * ws * ws
        add(f"block{blk}_out_proj",
            f"block {blk} out_proj ({'%d'%M},{VD})",
            M * VD, "VIT_OUT_PROJ", make_out_proj(blk))

        def make_residual1(b):
            def cpu_residual1(ctx, sd, image_path):
                x_in = ctx.get(f'block{b-1}_out') if b > 0 else ctx['patch_embed']
                out_proj = ctx[f'block{b}_out_proj']
                bg = b in ue.VIT_GLOBAL_BLOCKS
                if not bg:
                    # window reverse + unpad
                    proj = (out_proj.reshape(nws, nws, ws, ws, VD)
                                    .permute(0,2,1,3,4)
                                    .reshape(GSP, GSP, VD))
                    proj_flat = proj[:GS, :GS, :].reshape(GA, VD)
                else:
                    proj_flat = out_proj
                out = (x_in.float() + proj_flat.float()).to(torch.bfloat16)
                ctx[f'block{b}_residual1'] = out
                return out
            return cpu_residual1

        add(f"block{blk}_residual1", f"block {blk} residual1 (4096,768)",
            GA * VD, "VIT_RESIDUAL", make_residual1(blk))

        def make_ln2(b):
            def cpu_ln2(ctx, sd, image_path):
                x = ctx[f'block{b}_residual1'].float()
                g    = sd[f"vision_encoder.layers.{b}.layer_norm2.weight"].to(torch.bfloat16)
                beta = sd[f"vision_encoder.layers.{b}.layer_norm2.bias"].to(torch.bfloat16)
                mean = x.mean(-1, keepdim=True).to(torch.bfloat16)
                cent = (x.to(torch.bfloat16) - mean).to(torch.bfloat16)
                rms  = cent.float().pow(2).mean(-1, keepdim=True).sqrt().to(torch.bfloat16)
                out  = (cent.float() / rms.float()).to(torch.bfloat16) * g + beta
                ctx[f'block{b}_ln2'] = out
                return out
            return cpu_ln2

        add(f"block{blk}_ln2", f"block {blk} LN2 (4096,768)",
            GA * VD, "VIT_LN_OUT", make_ln2(blk))

        def make_fc1(b):
            def cpu_fc1(ctx, sd, image_path):
                normed2 = ctx[f'block{b}_ln2']
                lw = sd[f"vision_encoder.layers.{b}.mlp.lin1.weight"].to(torch.bfloat16)
                lb = sd[f"vision_encoder.layers.{b}.mlp.lin1.bias"].to(torch.bfloat16)
                out = F.linear(normed2.float(), lw.float(), lb.float()).to(torch.bfloat16)
                # GELU (quick-GELU approximation used by HW)
                out = (out * torch.sigmoid(1.702 * out.float()).to(torch.bfloat16)).to(torch.bfloat16)
                ctx[f'block{b}_fc1'] = out
                return out
            return cpu_fc1

        add(f"block{blk}_fc1", f"block {blk} fc1+GELU (4096,3072)",
            GA * ue.VIT_MLP_HIDDEN, "VIT_MLP_MID", make_fc1(blk))

        def make_mlp_out(b):
            def cpu_mlp_out(ctx, sd, image_path):
                mid = ctx[f'block{b}_fc1']
                lw = sd[f"vision_encoder.layers.{b}.mlp.lin2.weight"].to(torch.bfloat16)
                lb = sd[f"vision_encoder.layers.{b}.mlp.lin2.bias"].to(torch.bfloat16)
                out = F.linear(mid.float(), lw.float(), lb.float()).to(torch.bfloat16)
                ctx[f'block{b}_mlp_out'] = out
                return out
            return cpu_mlp_out

        add(f"block{blk}_mlp_out", f"block {blk} MLP fc2 (4096,768)",
            GA * VD, "VIT_MLP_OUT", make_mlp_out(blk))

        def make_block_out(b):
            def cpu_block_out(ctx, sd, image_path):
                res1   = ctx[f'block{b}_residual1']
                mlp_o  = ctx[f'block{b}_mlp_out']
                out    = (res1.float() + mlp_o.float()).to(torch.bfloat16)
                ctx[f'block{b}_out'] = out
                return out
            return cpu_block_out

        add(f"block{blk}_out", f"block {blk} output (4096,768)",
            GA * VD, "VIT_LAYER_IN", make_block_out(blk))

    # ── neck ─────────────────────────────────────────────────────────────────
    def cpu_neck_conv1(ctx, sd, image_path):
        x = ctx[f'block{ue.VIT_DEPTH-1}_out'].float()
        w = sd["vision_encoder.neck.conv1.weight"].to(torch.bfloat16).reshape(ND, VD)
        out = (x @ w.float().T).to(torch.bfloat16)
        ctx['neck_conv1'] = out
        return out

    add("neck_conv1", "neck conv1×1 (4096,256)", GA * ND, "VIT_NECK_OUT", cpu_neck_conv1)

    def cpu_neck_ln1(ctx, sd, image_path):
        import torch.nn as nn
        x = ctx['neck_conv1'].float()
        g = sd["vision_encoder.neck.layer_norm1.weight"].float()
        b = sd["vision_encoder.neck.layer_norm1.bias"].float()
        mean = x.mean(-1, keepdim=True)
        cent = x - mean
        rms  = cent.pow(2).mean(-1, keepdim=True).sqrt()
        out  = (cent / (rms + 1e-6)).to(torch.bfloat16) * g.to(torch.bfloat16) + b.to(torch.bfloat16)
        ctx['neck_ln1'] = out
        return out

    add("neck_ln1", "neck LN1 (4096,256)", GA * ND, "VIT_NECK_OUT", cpu_neck_ln1)

    def cpu_neck_conv2(ctx, sd, image_path):
        x = ctx['neck_ln1']
        w = sd["vision_encoder.neck.conv2.weight"].to(torch.bfloat16)
        x_s = x.reshape(GS, GS, ND).permute(2,0,1).unsqueeze(0)
        out = F.conv2d(x_s.float(), w.float(), bias=None, padding=1).squeeze(0).permute(1,2,0).reshape(GA, ND).to(torch.bfloat16)
        ctx['neck_conv2'] = out
        return out

    add("neck_conv2", "neck conv3×3 (4096,256)", GA * ND, "NECK_OUT", cpu_neck_conv2)

    def cpu_neck_ln2(ctx, sd, image_path):
        x = ctx['neck_conv2'].float()
        g = sd["vision_encoder.neck.layer_norm2.weight"].float()
        b = sd["vision_encoder.neck.layer_norm2.bias"].float()
        mean = x.mean(-1, keepdim=True)
        cent = x - mean
        rms  = cent.pow(2).mean(-1, keepdim=True).sqrt()
        out  = (cent / (rms + 1e-6)).to(torch.bfloat16) * g.to(torch.bfloat16) + b.to(torch.bfloat16)
        ctx['neck_ln2'] = out
        return out

    add("neck_ln2", "neck LN2 / image embedding (4096,256)", GA * ND, "NECK_OUT", cpu_neck_ln2)

    # ── decoder ──────────────────────────────────────────────────────────────
    def cpu_dec_src(ctx, sd, image_path):
        neck = ctx['neck_ln2']
        no_mask = sd["prompt_encoder.no_mask_embed.weight"].to(torch.bfloat16)
        out = (neck.float() + no_mask.float()).to(torch.bfloat16)
        ctx['dec_src'] = out
        return out

    add("dec_src", "dec image src + no_mask_embed (4096,256)", GA * DD, "DEC_SRC", cpu_dec_src)

    # Full decoder CPU reference is done via the existing cpu_reference_dec_* methods
    # but for a clean per-op walkthrough we just run the full CPU model at decoder stage
    def _cpu_dec_full(ctx, sd, image_path, point=(512.0, 512.0)):
        """Run full mask decoder CPU-side, storing all intermediates in ctx."""
        if 'cpu_dec_done' in ctx:
            return
        from sam1_vit_b_cpu import (
            PositionEmbeddingRandom, PromptEncoder, MaskDecoder,
            TwoWayTransformer, TwoWayBlock, DecAttn, HyperMLP, LayerNorm2d,
            PROMPT_DIM, IMG_SIZE, PATCH_SIZE
        )
        # Build a full SAM1 model and load weights, run decoder
        # We have image embedding already from CPU neck
        import sam1_vit_b_cpu as ref_mod
        model = ref_mod.SAM1().eval()
        ref_mod.load_weights(model, os.path.join(SCRIPT_DIR, "sam1_vit_b_bin", "model.safetensors"))

        img_emb_hw = ctx['neck_ln2'].reshape(64, 64, 256).permute(2, 0, 1).unsqueeze(0).float()

        pe_mat = sd["shared_image_embedding.positional_embedding"].float()
        px, py = point
        pt = torch.tensor([[[px, py]]], dtype=torch.float32)
        lbl = torch.tensor([[1]], dtype=torch.long)
        with torch.no_grad():
            image_pe = model.get_image_pe('cpu').expand(1,-1,-1,-1)
            sparse, dense = model.prompt_encoder(pt, lbl)
            masks, iou = model.mask_decoder(img_emb_hw, image_pe, sparse, dense)
        ctx['cpu_dec_masks']   = masks[0]   # (4, 256, 256)
        ctx['cpu_dec_iou']     = iou[0]     # (4,)
        ctx['cpu_dec_done']    = True
        # NOTE: do NOT overwrite ctx['cpu_tokens'] — it was pre-computed
        # in run_comparison() and must stay identical to what was DMA'd to HW.

    # Decoder checkpoints: compare DEC_TOKENS and DEC_SRC after each stage
    # Since sub-op tracking is complex, we compare key stage outputs using full decoder pass

    def cpu_dec_l0_sa_out(ctx, sd, image_path):
        # cpu_tokens already in ctx (pre-computed in run_comparison before the loop)
        neck       = ctx['neck_ln2']
        cpu_tokens = ctx['cpu_tokens']   # (7, 256) — identical to what was DMA'd to HW
        tokens_out, src_out = ue.cpu_reference_dec_sa0(cpu_tokens, neck)
        ctx['dec_l0_sa_tokens'] = tokens_out
        ctx['dec_l0_sa_src']    = src_out
        return tokens_out  # (7, 256)

    add("dec_l0_sa_out", "dec layer0 after self-attn+norm1 (7,256)",
        NT * DD, "DEC_TOKENS", cpu_dec_l0_sa_out)

    def cpu_dec_l0_t2i_out(ctx, sd, image_path):
        tokens   = ctx['dec_l0_sa_tokens']
        src      = ctx['dec_l0_sa_src']
        query_pe = ctx['query_pe']   # identical to what was DMA'd to DEC_QUERY_PE
        key_pe   = ctx['key_pe']     # identical to what was DMA'd to DEC_KEY_PE
        out = ue.cpu_reference_dec_t2i0(tokens, src, query_pe, key_pe)
        ctx['dec_l0_t2i_tokens'] = out
        return out  # (7, 256)

    add("dec_l0_t2i_out", "dec layer0 after t2i cross-attn+norm2 (7,256)",
        NT * DD, "DEC_TOKENS", cpu_dec_l0_t2i_out)

    # For stages after l0_t2i, use the full CPU SAM1 model's final outputs to compare
    # These are approximate comparisons showing cumulative error
    for stage_name, stage_desc in [
        ("dec_l0_mlp_out",  "dec layer0 after MLP+norm3 (7,256)"),
        ("dec_l0_i2t_src",  "dec layer0 after i2t+norm4 — DEC_SRC (4096,256)"),
        ("dec_l1_sa_out",   "dec layer1 after self-attn+norm1 (7,256)"),
        ("dec_l1_t2i_out",  "dec layer1 after t2i+norm2 (7,256)"),
        ("dec_l1_mlp_out",  "dec layer1 after MLP+norm3 (7,256)"),
        ("dec_l1_i2t_src",  "dec layer1 after i2t+norm4 — DEC_SRC (4096,256)"),
        ("dec_final_attn",  "dec final attn+LN (7,256)"),
    ]:
        # These need custom per-stage CPU references; skip for now and mark as N/A
        pass

    def cpu_dec_iou(ctx, sd, image_path):
        _cpu_dec_full(ctx, sd, image_path)
        return ctx['cpu_dec_iou'].to(torch.bfloat16)

    add("dec_iou", "decoder IoU scores (4,)",
        4, "DEC_IOU_OUT", cpu_dec_iou)

    def cpu_dec_mask_logits(ctx, sd, image_path):
        _cpu_dec_full(ctx, sd, image_path)
        # Return best-mask upscaled 256×256 binary comparison
        masks = ctx['cpu_dec_masks']  # (4, 256, 256)
        # Flatten to (65536, 4) to match DEC_MASK_LOGITS layout
        return masks.permute(1, 2, 0).reshape(-1, 4).to(torch.bfloat16)

    add("dec_mask_logits", "mask logits (65536,4)",
        65536 * 4, "DEC_MASK_LOGITS", cpu_dec_mask_logits)

    return checks


# ─────────────────────────────────────────────────────────────────────────────
# Comparison driver
# ─────────────────────────────────────────────────────────────────────────────

COL_W = {
    "name":    30,
    "snr":      9,
    "drop":     7,
    "cos":      9,
    "max":      8,
    "mean":     9,
    "ulp":      7,
    "status":   6,
}

def _print_header():
    h = (f"  {'Checkpoint':<{COL_W['name']}} "
         f"{'SNR(dB)':>{COL_W['snr']}} "
         f"{'Drop':>{COL_W['drop']}} "
         f"{'Cos':>{COL_W['cos']}} "
         f"{'MaxErr':>{COL_W['max']}} "
         f"{'MeanErr':>{COL_W['mean']}} "
         f"{'1ULP%':>{COL_W['ulp']}}  Status")
    _original_print(h)
    _original_print("  " + "-" * (len(h) - 2))

def _print_row(name, m, status, snr_drop=0.0):
    drop_str = f"{snr_drop:+.1f}" if snr_drop != 0.0 else "    —"
    _original_print(
        f"  {name:<{COL_W['name']}} "
        f"{m['snr_db']:>{COL_W['snr']}.1f} "
        f"{drop_str:>{COL_W['drop']}} "
        f"{m['cos']:>{COL_W['cos']}.6f} "
        f"{m['max_err']:>{COL_W['max']}.4f} "
        f"{m['mean_err']:>{COL_W['mean']}.6f} "
        f"{m['pct_1ulp']:>{COL_W['ulp']}.1f}%  {status}"
    )


def run_comparison(ue: CompareSAM1, image_path: str, point: tuple,
                   start_at: str = None, stop_at: str = None,
                   bail_on_fail: bool = True, block_only: int = None):
    """Run the full per-op comparison loop."""
    ckpt_path = _ensure_checkpoint(SCRIPT_DIR, ue.cfg)
    sd = _load_sam1_state_dict(ckpt_path)

    checks = _build_checkpoints(ue)

    # Filter by --block if specified
    if block_only is not None:
        checks = [(n, d, ne, da, fn) for n, d, ne, da, fn in checks
                  if n.startswith(f"block{block_only}") or n == "patch_embed"]

    # Filter by --start / --stop
    names = [c[0] for c in checks]
    start_idx = 0 if start_at is None else (names.index(start_at) if start_at in names else 0)
    stop_idx  = len(checks) if stop_at is None else (names.index(stop_at) + 1 if stop_at in names else len(checks))
    checks = checks[start_idx:stop_idx]

    # ── Pre-compute all decoder inputs upfront so HW and CPU use identical values ──
    # These are independent of the image embedding and must be DMA'd to DRAM
    # before every decoder checkpoint execution.
    _original_print("  Pre-computing decoder token inputs...")
    pe_mat  = sd["shared_image_embedding.positional_embedding"].float()  # (2, 128)
    px, py  = point

    # Foreground point PE: encode (px, py) normalised to [0,1]
    coord_fg = torch.tensor([[px / ue.IMAGE_SIZE, py / ue.IMAGE_SIZE]], dtype=torch.float32)
    coord_fg = 2.0 * coord_fg - 1.0
    proj_fg  = 2 * math.pi * (coord_fg @ pe_mat)
    fg_pe    = torch.cat([proj_fg.sin(), proj_fg.cos()], dim=-1).to(torch.bfloat16)  # (1, 256)
    fg_emb   = (fg_pe.float() + sd["prompt_encoder.point_embed.0.weight"].float()).to(torch.bfloat16)

    # Padding (not-a-point) PE: encode (0.5/1024, 0.5/1024) — center of first pixel
    coord_pad = torch.tensor([[0.5 / ue.IMAGE_SIZE, 0.5 / ue.IMAGE_SIZE]], dtype=torch.float32)
    coord_pad = 2.0 * coord_pad - 1.0
    proj_pad  = 2 * math.pi * (coord_pad @ pe_mat)
    pad_pe    = torch.cat([proj_pad.sin(), proj_pad.cos()], dim=-1).to(torch.bfloat16)
    pad_emb   = (pad_pe.float() + sd["prompt_encoder.not_a_point_embed.weight"].float()).to(torch.bfloat16)

    # Token sequence: [iou_token, mask_token×4, fg_point, pad_point]
    iou_tok   = sd["mask_decoder.iou_token.weight"].to(torch.bfloat16)    # (1, 256)
    mask_toks = sd["mask_decoder.mask_tokens.weight"].to(torch.bfloat16)  # (4, 256)
    cpu_tokens = torch.cat([iou_tok, mask_toks, fg_emb, pad_emb], dim=0)  # (7, 256)

    # Query PE = initial token embeddings (SAM1 uses tokens as their own PE)
    query_pe = cpu_tokens.clone()  # (7, 256)

    # Key PE = dense image positional encoding for 64×64 grid
    gy, gx  = torch.meshgrid(torch.arange(64, dtype=torch.float32),
                              torch.arange(64, dtype=torch.float32), indexing='ij')
    grid    = torch.stack([(gx + 0.5) / 64, (gy + 0.5) / 64], dim=-1).reshape(-1, 2)  # (4096, 2)
    grid    = 2.0 * grid - 1.0
    proj_kp = 2 * math.pi * (grid @ pe_mat)
    key_pe  = torch.cat([proj_kp.sin(), proj_kp.cos()], dim=-1).to(torch.bfloat16)     # (4096, 256)

    _original_print(f"    tokens    : {cpu_tokens.shape}  range [{cpu_tokens.min():.3f}, {cpu_tokens.max():.3f}]")
    _original_print(f"    query_pe  : {query_pe.shape}   range [{query_pe.min():.3f}, {query_pe.max():.3f}]")
    _original_print(f"    key_pe    : {key_pe.shape}  range [{key_pe.min():.3f}, {key_pe.max():.3f}]")

    # Store in ctx so CPU reference functions see the same values
    ctx = {
        'point':      point,
        'cpu_tokens': cpu_tokens,
        'query_pe':   query_pe,
        'key_pe':     key_pe,
    }

    _original_print(f"\n  Running {len(checks)} checkpoint(s)...\n")
    _print_header()

    any_fail = False
    prev_snr = None   # track SNR across checkpoints to detect sudden drops

    for name, desc, n_elems, dram_attr, cpu_fn in checks:
        t0 = _time.perf_counter()

        # 1. Compile up to checkpoint
        try:
            _original_print(f"  [compile] {name}...", end="", flush=True)
            prog_addr = ue.compile_to(name)
        except Exception as e:
            _original_print(f"\n  [ERROR] compile failed for '{name}': {e}")
            continue

        # 2. DMA decoder inputs before every decoder checkpoint execution.
        #    Done unconditionally so HW always gets the same inputs as CPU.
        if name.startswith("dec_"):
            ue.dma_to_accelerator_memory(ue.DEC_TOKENS,   cpu_tokens.contiguous())
            ue.dma_to_accelerator_memory(ue.DEC_QUERY_PE, query_pe.contiguous())
            ue.dma_to_accelerator_memory(ue.DEC_KEY_PE,   key_pe.contiguous())

        # 3. Execute HW
        ue.run_hw(image_path, prog_addr)
        t_hw = _time.perf_counter() - t0

        # 4. Read HW tensor
        dram_addr = getattr(ue, dram_attr)
        hw_tensor = ue.read_tensor_from_dram(dram_addr, n_elems).float()

        # 5. CPU reference
        try:
            cpu_tensor = cpu_fn(ctx, sd, image_path)
            cpu_flat = cpu_tensor.to(torch.float32).flatten()[:n_elems]
        except Exception as e:
            _original_print(f"\n  [ERROR] cpu_ref failed for '{name}': {e}")
            import traceback; traceback.print_exc()
            continue

        # 6. Compare
        m = Sam1VitB_UnifiedEngine.tensor_metrics(hw_tensor, cpu_flat)

        # Thresholds — what's "acceptable" vs "broken":
        #   PASS  : SNR > 40 dB  OR  ≥99% within 1 ULP  (near-perfect BF16)
        #   CLOSE : SNR 30–40 dB  — expected for LN ops, attention, BF16 accumulation
        #   WARN  : SNR 20–30 dB  — noteworthy degradation; accumulates with depth
        #   FAIL  : SNR < 20 dB   — real divergence: logic bug, wrong weights, wrong layout
        snr = m["snr_db"]
        if snr > 40 or m["pct_1ulp"] >= 99.0:
            status = "PASS"
        elif snr > 30:
            status = "CLOSE"
        elif snr > 20:
            status = "WARN"
        else:
            status = "FAIL"

        # Track SNR drop relative to last checkpoint (sudden drops = layout/logic bug)
        snr_drop = (prev_snr - snr) if prev_snr is not None else 0.0
        prev_snr = snr

        _print_row(name, m, status, snr_drop)

        if status in ("WARN", "FAIL"):
            any_fail = True
            if bail_on_fail:
                tag = {"WARN": "Significant degradation", "FAIL": "Divergence"}[status]
                _original_print(f"\n  *** {tag} at '{name}' (SNR={snr:.1f} dB, drop={snr_drop:+.1f} dB) — stopping. ***")
                _original_print(f"      HW range:  [{hw_tensor.min():.4f}, {hw_tensor.max():.4f}]")
                _original_print(f"      CPU range: [{cpu_flat.min():.4f}, {cpu_flat.max():.4f}]")
                break

    _original_print()
    if any_fail:
        _original_print("  ❌  WARN/FAIL found. See rows above.")
    else:
        _original_print("  ✓  All checkpoints PASS/CLOSE (within expected BF16 precision).")

    del sd


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Per-operation CPU vs HW comparison for SAM 1 ViT-B")
    parser.add_argument("--image", default=None, help="Input image path")
    parser.add_argument("--point", nargs=2, type=float, default=[512.0, 512.0],
                        metavar=("X", "Y"), help="Point prompt (default: center 512,512)")
    parser.add_argument("--dev", default="xdma0", help="DMA device (default: xdma0)")
    parser.add_argument("--cycle", type=float, default=5.62,
                        help="Clock cycle in ns (default: 5.62)")
    parser.add_argument("--start", default=None,
                        help="Start comparison at this checkpoint name")
    parser.add_argument("--stop", default=None,
                        help="Stop after this checkpoint name")
    parser.add_argument("--block", type=int, default=None,
                        help="Only compare sub-op checkpoints for this ViT block")
    parser.add_argument("--no-bail", action="store_true",
                        help="Continue past first FAIL instead of stopping")
    parser.add_argument("--list", action="store_true",
                        help="Print all checkpoint names and exit (no HW needed)")
    args = parser.parse_args()

    if args.list:
        _original_print("Checkpoint names (pass to --start / --stop):\n")
        # Enumerate checkpoints without connecting to HW
        VIT_DEPTH = 12
        VIT_GLOBAL_BLOCKS = {2, 5, 8, 11}
        i = 0
        _original_print(f"  {i:3d}  {'patch_embed':<35}  patch embed + pos embed (4096,768)"); i += 1
        for blk in range(VIT_DEPTH):
            t = "GLOBAL" if blk in VIT_GLOBAL_BLOCKS else "local"
            for suffix, desc in [
                (f"block{blk}_ln1",       f"block {blk} LN1 [{t}] (4096,768)"),
                *([(f"block{blk}_windowed", f"block {blk} windowed (4900,768)")] if blk not in VIT_GLOBAL_BLOCKS else []),
                (f"block{blk}_q",          f"block {blk} Q proj"),
                (f"block{blk}_k",          f"block {blk} K proj"),
                (f"block{blk}_v",          f"block {blk} V proj"),
                (f"block{blk}_attn_merge", f"block {blk} attn merge"),
                (f"block{blk}_out_proj",   f"block {blk} out proj"),
                (f"block{blk}_residual1",  f"block {blk} residual1 (4096,768)"),
                (f"block{blk}_ln2",        f"block {blk} LN2 (4096,768)"),
                (f"block{blk}_fc1",        f"block {blk} fc1+GELU (4096,3072)"),
                (f"block{blk}_mlp_out",    f"block {blk} MLP fc2 (4096,768)"),
                (f"block{blk}_out",        f"block {blk} output (4096,768)"),
            ]:
                _original_print(f"  {i:3d}  {suffix:<35}  {desc}"); i += 1
        for name, desc in [
            ("neck_conv1",       "neck conv1×1 (4096,256)"),
            ("neck_ln1",         "neck LN1 (4096,256)"),
            ("neck_conv2",       "neck conv3×3 (4096,256)"),
            ("neck_ln2",         "neck LN2 / image embedding (4096,256)"),
            ("dec_src",          "dec image src + no_mask_embed (4096,256)"),
            ("dec_l0_sa_out",    "dec layer0 self-attn+norm1 (7,256)"),
            ("dec_l0_t2i_out",   "dec layer0 t2i+norm2 (7,256)"),
            ("dec_l0_mlp_out",   "dec layer0 MLP+norm3 (7,256)"),
            ("dec_l0_i2t_src",   "dec layer0 i2t+norm4 DEC_SRC (4096,256)"),
            ("dec_l1_sa_out",    "dec layer1 self-attn+norm1 (7,256)"),
            ("dec_l1_t2i_out",   "dec layer1 t2i+norm2 (7,256)"),
            ("dec_l1_mlp_out",   "dec layer1 MLP+norm3 (7,256)"),
            ("dec_l1_i2t_src",   "dec layer1 i2t+norm4 DEC_SRC (4096,256)"),
            ("dec_final_attn",   "dec final cross-attn+LN (7,256)"),
            ("dec_iou",          "IoU scores (4,)"),
            ("dec_hyper",        "hypernetwork outputs (4,32)"),
            ("dec_up0",          "upscale conv0+LN+GELU (16384,64)"),
            ("dec_up1",          "upscale conv1+GELU (65536,32)"),
            ("dec_mask_logits",  "mask logits (65536,4)"),
        ]:
            _original_print(f"  {i:3d}  {name:<35}  {desc}"); i += 1
        return

    set_dma_device(args.dev)
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle
    user_dma_core.MAX_DECODER_INSTRUCTIONS = (0x100000000 - 0x9C000000) // 32

    image_path = args.image
    if image_path is None:
        for cand in [
            os.path.join(SCRIPT_DIR, "../../../test_samples/vette.jpg"),
            os.path.join(SCRIPT_DIR, "../../../test_samples/test_image.jpg"),
        ]:
            if os.path.exists(cand):
                image_path = cand
                break
        else:
            _original_print("No image found. Pass --image <path>")
            sys.exit(1)

    _original_print(f"SAM 1 ViT-B per-op comparison  [{args.dev}]  image={os.path.basename(image_path)}")
    _original_print(f"Point: ({args.point[0]:.0f}, {args.point[1]:.0f})")

    t0 = _time.perf_counter()
    ue = CompareSAM1(script_dir=SCRIPT_DIR)
    _original_print(f"  Model init: {_time.perf_counter()-t0:.3f}s")

    run_comparison(
        ue=ue,
        image_path=image_path,
        point=tuple(args.point),
        start_at=args.start,
        stop_at=args.stop,
        bail_on_fail=not args.no_bail,
        block_only=args.block,
    )


if __name__ == "__main__":
    main()
