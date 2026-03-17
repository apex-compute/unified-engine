#!/usr/bin/env python3
"""
Gemma3 numeric verification: torch reference vs hardware prefill.

Inherits Gemma3_UnifiedEngine, runs prefill with FIXED_PREFILL, then
torch_ref_check_prefill() computes torch ref (same as compile_prefill),
reads intermediate tensors from hardware DRAM (LAYER0_PRE_NORM_DRAM to
LAYER0_OUTPUT_DRAM), and checks using calculate_snr.

Usage:
  source venv/bin/activate  # if using venv
  python gemma3_numeric.py --dev xdma0 [--cycle 5.62] [--layer-size 1]

Same layout as gemma3_test: this file lives next to gemma3_bin/, *.json; user_dma_core one level up.
"""

import json
import math
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in [os.path.join(_REPO_ROOT, "src"), _REPO_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

import user_dma_core
from user_dma_core import DMA_DEVICE_C2H, TYPE, set_dma_device, calculate_snr
from read_trace import generate_trace

import gemma3_test
from gemma3_test import (
    Gemma3_UnifiedEngine,
    _parse_offset,
    SCRIPT_DIR,
)

def _dequantize_int4_from_bin(
    weight_bin: bytes,
    scale_off: int,
    scale_sz: int,
    data_off: int,
    data_sz: int,
    N: int,
    K: int,
    block_size: int = 64,
) -> torch.Tensor:
    """
    Dequantize INT4 packed weight from bin. Scale per block of 64 along K.
    Same format as _quantize_bf16_to_int4_packed in gemma3_test.
    Returns (N, K) bfloat16 tensor.
    """
    assert K % block_size == 0
    num_blocks_k = K // block_size

    # Scale: (N, K/64) bf16
    scale_bytes = weight_bin[scale_off : scale_off + scale_sz]
    scale_elems = len(scale_bytes) // 2
    scale = torch.frombuffer(
        bytearray(scale_bytes), dtype=torch.bfloat16
    ).clone().reshape(N, num_blocks_k)

    # Data: packed (N, K/2) bytes, low nibble = even idx, high = odd
    data_bytes = weight_bin[data_off : data_off + data_sz]
    packed = np.frombuffer(data_bytes, dtype=np.uint8)
    packed = packed[: N * (K // 2)].reshape(N, K // 2)

    # Unpack: each byte -> two int4 values (signed: 0-7 -> 0-7, 8-15 -> -8 to -1)
    low = (packed & 0x0F).astype(np.int16)
    high = ((packed >> 4) & 0x0F).astype(np.int16)
    low = np.where(low >= 8, low - 16, low)
    high = np.where(high >= 8, high - 16, high)
    w_int8 = np.stack([low, high], axis=-1).reshape(N, K)

    # Dequantize: w_int8 * scale (scale broadcast per block)
    scale_expanded = scale.float().repeat_interleave(block_size, dim=1).to(torch.bfloat16)
    w_bf16 = torch.from_numpy(w_int8.astype(np.float32)).to(torch.bfloat16) * scale_expanded
    return w_bf16


def _rms_norm_torch(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """RMS norm: x * gamma / sqrt(mean(x^2)). Matches HW: mean = sum/N (no epsilon)."""
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True)).to(x.dtype)
    return (x / rms) * gamma


def _rope_hf_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    RoPE matching HW rope_hf_core: out = [x0*cos0 - x1*sin0, x0*sin1 + x1*cos1].
    x shape (..., head_dim), cos/sin (head_dim,) with cos=[cos0, cos1], sin=[sin0, sin1].
    """
    D = x.shape[-1] // 2
    x0 = x[..., :D]
    x1 = x[..., D:]
    return torch.cat(
        [x0 * cos[:D] - x1 * sin[:D], x0 * sin[D:] + x1 * cos[D:]],
        dim=-1,
    )


def _flash_attention_torch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float,
    causal_mask_upper: bool,
) -> torch.Tensor:
    """
    Scaled dot-product attention. Matches compile_prefill + flash_attention_core + run_prefill bias:
    - Scale Q by 1/sqrt(head_dim) then scores = Q @ K.T (same as HW broadcast_mul on Q then matvec).
    - Bias: same as run_prefill (valid_mask tril/triu, masked_fill 0, then add to scores).
    - softmax(dim=-1) then @ V.
    """
    q_scaled = Q * scale
    scores = q_scaled @ K.T
    nq, nk = scores.shape[0], scores.shape[1]
    # Bias identical to run_prefill: full(-inf), then 0 where valid (tril/triu), then add to scores
    bias = torch.full((nq, nk), float("-inf"), dtype=torch.bfloat16)
    if causal_mask_upper:
        valid_mask = torch.triu(torch.ones(nq, nk, dtype=torch.bool), diagonal=0)
    else:
        valid_mask = torch.tril(torch.ones(nq, nk, dtype=torch.bool), diagonal=0)
    bias.masked_fill_(valid_mask, 0.0)
    scores = scores + bias
    probs = torch.nn.functional.softmax(scores.float(), dim=-1).to(Q.dtype)
    return probs @ V


def decoder_attention_torch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Single-head decoder attention matching compile_decoder + decoder_attention_core exactly:
    - Q (1, head_dim), K (seq_len, head_dim), V (seq_len, head_dim).
    - Scale Q by 1/sqrt(head_dim) (HW broadcast_mul), scores = Q @ K.T, add bias, softmax, out = probs @ V.
    - Called once per head group g; HW loops over g and calls decoder_attention_core per g.
    """
    assert Q.shape[0] == 1 and K.shape[0] == V.shape[0] and K.shape[1] == Q.shape[1] == V.shape[1]
    seq_len = K.shape[0]
    q_scaled = Q * scale
    scores = q_scaled @ K.T
    if bias.shape[1] < seq_len:
        bias = torch.nn.functional.pad(bias, (0, seq_len - bias.shape[1]), value=float("-inf"))
    scores = scores + bias[:, :seq_len].to(scores.dtype)
    probs = torch.nn.functional.softmax(scores.float(), dim=-1).to(Q.dtype)
    return probs @ V


def _gelu_torch(x: torch.Tensor) -> torch.Tensor:
    """GELU approximation: x * sigmoid(1.702 * x)."""
    return x * torch.sigmoid(1.702 * x)


class Gemma3_NumericEngine(Gemma3_UnifiedEngine):
    """
    Gemma3_UnifiedEngine with torch reference computation for numeric verification.
    """

    def _load_layer_weights_torch(self, layer_idx: int) -> dict:
        """Load layer weights from weight_bin for torch ref (bf16, int4 dequantized)."""
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        layer_off = layer_idx * LAYER_WEIGHT_SIZE
        wd = self.weight_defs
        blk0 = self._cfg["layers"]["structure"]
        block_size = self._cfg["special"]["quantization"]["block_size"]

        weights = {}
        i = 0
        while i < len(blk0):
            key = blk0[i]["key"]
            off = wd[key] + layer_off
            sz = wd[f"{key}_SIZE"]
            ttype = blk0[i].get("type", "")
            if ttype == "bf16":
                raw = self.weight_bin[off : off + sz]
                t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).clone()
                weights[key] = t
                i += 1
            elif ttype == "bf16_scale" and i + 1 < len(blk0) and "DATA" in blk0[i + 1]["key"]:
                scale_key = key
                data_key = blk0[i + 1]["key"]
                scale_off = wd[scale_key] + layer_off
                scale_sz = wd[f"{scale_key}_SIZE"]
                data_off = wd[data_key] + layer_off
                data_sz = wd[f"{data_key}_SIZE"]
                if "Q_WEIGHT" in scale_key:
                    N, K = self.head_dim * self.group_size, self.vector_length
                elif "K_WEIGHT" in scale_key or "V_WEIGHT" in scale_key:
                    N, K = self.head_dim, self.vector_length
                elif "OUTPUT_WEIGHT" in scale_key:
                    N, K = self.vector_length, self.head_dim * self.group_size
                elif "UP_WEIGHT" in scale_key or "GATE_WEIGHT" in scale_key:
                    N, K = self.mlp_elements, self.vector_length
                elif "DOWN_WEIGHT" in scale_key:
                    N, K = self.vector_length, self.mlp_elements
                else:
                    N, K = self.mlp_elements, self.vector_length
                deq = _dequantize_int4_from_bin(
                    self.weight_bin, scale_off, scale_sz, data_off, data_sz, N, K, block_size
                )
                weights[data_key] = deq
                i += 2
            else:
                i += 1

        return weights

    def _compute_torch_prefill(
        self,
        prefill_seq: tuple,
        layer_size: int,
    ) -> tuple[dict, dict]:
        """
        Compute prefill using torch (same logic as compile_prefill).
        prefill_seq: tokens to process; seq_len = len(prefill_seq).
        Returns (last_intermediates, kv_cache).
        kv_cache[layer_idx] = {"k_rope": (seq_len, head_dim), "v": (seq_len, head_dim)} for flash attention.
        """
        seq_len = len(prefill_seq)
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        scale_attn = 1.0 / math.sqrt(self.head_dim)

        # Embedding input
        embedding = self.get_embedding_for_tokens(prefill_seq)
        x = embedding.clone()

        # Load RoPE from weight bin: same layout as _load_rope_host [cos0, cos1, -sin0, sin1] per position (D = head_dim//2).
        rope_cfg = self._cfg["special"]["rope"]
        num_positions = rope_cfg["num_positions"]
        rope_shape = (num_positions, 2 * self.head_dim)
        rope_tensors = {}
        for name, off_key, sz_key in [
            ("local", "ROPE_LOCAL", "ROPE_LOCAL_SIZE"),
            ("global", "ROPE_GLOBAL", "ROPE_GLOBAL_SIZE"),
        ]:
            off = self.weight_defs[off_key]
            sz = self.weight_defs[sz_key]
            raw = self.weight_bin[off : off + sz]
            rope = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).clone().reshape(rope_shape)
            # cos = [cos0, cos1], sin_block = [-sin0, sin1] (as stored)
            rope_tensors[f"cos_{name}"] = rope[:, : self.head_dim].clone()
            rope_tensors[f"sin_block_{name}"] = rope[:, self.head_dim : 2 * self.head_dim].clone()
        cos_local = rope_tensors["cos_local"]
        sin_block_local = rope_tensors["sin_block_local"]
        cos_global = rope_tensors["cos_global"]
        sin_block_global = rope_tensors["sin_block_global"]

        last_intermediates = {}
        kv_cache = {}  # layer_idx -> {"k_rope": (seq_len, head_dim), "v": (seq_len, head_dim)}

        for layer_idx in range(layer_size):
            w = self._load_layer_weights_torch(layer_idx)

            # RMS norm (gamma from bin = gamma+offset per weight_bin_generate, same as _check_result_with_ref)
            gamma_pre = w["BLK0_ATTN_NORM_WEIGHT"].reshape(self.vector_length)
            pre_norm = _rms_norm_torch(x, gamma_pre)
            if layer_idx == layer_size - 1:
                last_intermediates["LAYER0_PRE_NORM_DRAM"] = pre_norm.clone()

            # Q, K, V proj (input @ W^T)
            q_w = w["BLK0_ATTN_Q_WEIGHT_DATA"]
            k_w = w["BLK0_ATTN_K_WEIGHT_DATA"]
            v_w = w["BLK0_ATTN_V_WEIGHT_DATA"]
            q_proj = pre_norm @ q_w.T
            k_proj = pre_norm @ k_w.T
            v_proj = pre_norm @ v_w.T
            if layer_idx == layer_size - 1:
                last_intermediates["LAYER0_Q_DRAM"] = q_proj.reshape(seq_len * self.group_size, self.head_dim).clone()
                last_intermediates["LAYER0_K_DRAM"] = k_proj.clone()
                last_intermediates["LAYER0_V_DRAM"] = v_proj.clone()

            # K, Q norm
            gamma_k = w["BLK0_ATTN_K_NORM_WEIGHT"][: self.head_dim]
            gamma_q = w["BLK0_ATTN_Q_NORM_WEIGHT"][: self.head_dim]
            k_norm = _rms_norm_torch(k_proj, gamma_k)
            q_norm = _rms_norm_torch(
                q_proj.reshape(seq_len, self.group_size, self.head_dim),
                gamma_q,
            ).reshape(seq_len * self.group_size, self.head_dim)
            if layer_idx == layer_size - 1:
                last_intermediates["LAYER0_K_NORM_DRAM"] = k_norm.clone()
                last_intermediates["LAYER0_Q_NORM_DRAM"] = q_norm.clone()

            # RoPE: match HW rope_hf_core and _load_rope_host layout [cos0, cos1, -sin0, sin1].
            # HW computes out = x*cos + [x1*sin[:D], x0*sin[D:]] -> [x0*cos0 - x1*sin0, x1*cos1 + x0*sin1].
            # _rope_hf_torch(x, cos, sin) does x0'=x0*cos0-x1*sin0, x1'=x0*sin1+x1*cos1, so sin must be [sin0, sin1].
            use_global = layer_idx in self._rope_global_layers
            cos_all = cos_global if use_global else cos_local
            sin_block_all = sin_block_global if use_global else sin_block_local
            D = self.head_dim // 2
            k_rope = torch.stack(
                [
                    _rope_hf_torch(
                        k_norm[t],
                        cos_all[t],
                        torch.cat([-sin_block_all[t, :D], sin_block_all[t, D:]], dim=0),
                    )
                    for t in range(seq_len)
                ],
                dim=0,
            )
            q_rope = torch.stack(
                [
                    _rope_hf_torch(
                        q_norm[t * self.group_size + g],
                        cos_all[t],
                        torch.cat([-sin_block_all[t, :D], sin_block_all[t, D:]], dim=0),
                    )
                    for t in range(seq_len)
                    for g in range(self.group_size)
                ],
                dim=0,
            )
            if layer_idx == layer_size - 1:
                # K RoPE: (seq_len, head_dim). HW writes at LAYER0_K_ROPE_DRAM + layer_idx * MAX_CONTEXT_SIZE * k_size.
                last_intermediates["LAYER0_K_ROPE_DRAM"] = k_rope.clone()
                # Q RoPE (flash Q input): (seq_len * group_size, head_dim), layout row = t * group_size + g.
                last_intermediates["LAYER0_FLASH_Q_DRAM"] = q_rope.clone()

            # Per-layer K/V cache for decoder (same layout as HW: K at LAYER0_K_ROPE_DRAM + layer*..., V at LAYER0_V_DRAM + layer*...)
            kv_cache[layer_idx] = {"k_rope": k_rope.clone(), "v": v_proj.clone()}

            # Expand K, V for GQA
            k_flash = k_rope.unsqueeze(1).expand(
                -1, self.group_size, -1
            ).reshape(seq_len * self.group_size, self.head_dim)
            v_flash = v_proj.unsqueeze(1).expand(
                -1, self.group_size, -1
            ).reshape(seq_len * self.group_size, self.head_dim)

            # Flash attention (same causal + scale as gemma3_test _check_result_with_ref and HW bias)
            flash_out = _flash_attention_torch(
                q_rope, k_flash, v_flash, scale_attn, self.causal_mask_upper
            )
            if layer_idx == layer_size - 1:
                last_intermediates["LAYER0_FLASH_OUTPUT_DRAM"] = flash_out.clone()

            # O proj: flash_out (seq*group, head_dim) -> reshape to (seq, head_dim*group) for matmul
            o_w = w["BLK0_ATTN_OUTPUT_WEIGHT_DATA"]
            flash_out_flat = flash_out.reshape(seq_len, self.head_dim * self.group_size)
            attn_out = flash_out_flat @ o_w.T
            if layer_idx == layer_size - 1:
                last_intermediates["LAYER0_ATTN_PROJ_OUTPUT_DRAM"] = attn_out.clone()

            # Post attn norm
            gamma_post = w["BLK0_POST_ATTENTION_NORM_WEIGHT"].reshape(self.vector_length)
            post_attn_norm = _rms_norm_torch(attn_out, gamma_post)
            if layer_idx == layer_size - 1:
                last_intermediates["LAYER0_POST_ATTN_NORM_DRAM"] = post_attn_norm.clone()

            # Residual
            post_attn_residual = x + post_attn_norm
            if layer_idx == layer_size - 1:
                last_intermediates["LAYER0_POST_ATTN_RESIDUAL_DRAM"] = post_attn_residual.clone()

            # Pre MLP norm
            gamma_ffn = w["BLK0_FFN_NORM_WEIGHT"].reshape(self.vector_length)
            pre_mlp_norm = _rms_norm_torch(post_attn_residual, gamma_ffn)
            if layer_idx == layer_size - 1:
                last_intermediates["LAYER0_PRE_MLP_NORM_DRAM"] = pre_mlp_norm.clone()

            # Gate, Up, mult, Down
            gate_w = w["BLK0_FFN_GATE_WEIGHT_DATA"]
            up_w = w["BLK0_FFN_UP_WEIGHT_DATA"]
            down_w = w["BLK0_FFN_DOWN_WEIGHT_DATA"]
            gate_out = _gelu_torch(pre_mlp_norm @ gate_w.T)
            up_out = pre_mlp_norm @ up_w.T
            if layer_idx == layer_size - 1:
                last_intermediates["LAYER0_MLP_GATE_DRAM"] = gate_out.clone()
                last_intermediates["LAYER0_MLP_UP_DRAM"] = up_out.clone()
            mult = gate_out * up_out
            if layer_idx == layer_size - 1:
                last_intermediates["LAYER0_MLP_MULT_DRAM"] = mult.clone()
            down_out = mult @ down_w.T
            if layer_idx == layer_size - 1:
                last_intermediates["LAYER0_MLP_DOWN_DRAM"] = down_out.clone()

            # Post MLP norm
            gamma_post_ffn = w["BLK0_POST_FFW_NORM_WEIGHT"].reshape(self.vector_length)
            post_mlp_norm = _rms_norm_torch(down_out, gamma_post_ffn)
            if layer_idx == layer_size - 1:
                last_intermediates["LAYER0_POST_MLP_NORM_DRAM"] = post_mlp_norm.clone()

            # Output
            x = post_attn_residual + post_mlp_norm
            if layer_idx == layer_size - 1:
                last_intermediates["LAYER0_OUTPUT_DRAM"] = x.clone()

        return last_intermediates, kv_cache

    def _load_output_norm_and_lm_head_torch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Load OUTPUT_NORM gamma and LM_HEAD weight (bf16) from weight_bin for torch ref."""
        wd = self.weight_defs
        block_size = self._cfg["special"]["quantization"]["block_size"]
        # OUTPUT_NORM_WEIGHT: gamma (vector_length,)
        off = wd["OUTPUT_NORM_WEIGHT"]
        sz = wd["OUTPUT_NORM_WEIGHT_SIZE"]
        gamma = torch.frombuffer(bytearray(self.weight_bin[off : off + sz]), dtype=torch.bfloat16).clone()
        # LM_HEAD: (EMBEDDING_ELEMENTS, vector_length), int4 dequant
        scale_off = wd["LM_HEAD_WEIGHT_SCALE"]
        scale_sz = wd["LM_HEAD_WEIGHT_SCALE_SIZE"]
        data_off = wd["LM_HEAD_WEIGHT_DATA"]
        data_sz = wd["LM_HEAD_WEIGHT_DATA_SIZE"]
        N, K = self.EMBEDDING_ELEMENTS, self.vector_length
        lm_head_w = _dequantize_int4_from_bin(
            self.weight_bin, scale_off, scale_sz, data_off, data_sz, N, K, block_size
        )
        return gamma, lm_head_w

    def _compute_torch_decoder(
        self,
        last_token_id: int,
        kv_cache: dict,
        layer_size: int,
    ) -> dict:
        """
        One decoder step: input = last token embedding, use kv_cache in flash attention,
        then through all layers and last LM norm + LM head. Returns ref dict for 1-token
        last-layer intermediates (same keys as prefill) plus OUTPUT_NORM_DRAM, LOGITS_DRAM.
        """
        seq_len = kv_cache[0]["k_rope"].shape[0]
        scale_attn = 1.0 / math.sqrt(self.head_dim)
        D = self.head_dim // 2

        # RoPE for position seq_len (new token)
        rope_cfg = self._cfg["special"]["rope"]
        num_positions = rope_cfg["num_positions"]
        rope_shape = (num_positions, 2 * self.head_dim)
        rope_tensors = {}
        for name, off_key, sz_key in [
            ("local", "ROPE_LOCAL", "ROPE_LOCAL_SIZE"),
            ("global", "ROPE_GLOBAL", "ROPE_GLOBAL_SIZE"),
        ]:
            off = self.weight_defs[off_key]
            sz = self.weight_defs[sz_key]
            raw = self.weight_bin[off : off + sz]
            rope = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).clone().reshape(rope_shape)
            rope_tensors[f"cos_{name}"] = rope[:, : self.head_dim].clone()
            rope_tensors[f"sin_block_{name}"] = rope[:, self.head_dim : 2 * self.head_dim].clone()

        x = self.get_embedding_for_tokens([last_token_id]).clone()  # (1, vector_length)
        ref = {}

        for layer_idx in range(layer_size):
            w = self._load_layer_weights_torch(layer_idx)
            use_global = layer_idx in self._rope_global_layers
            cos_all = rope_tensors["cos_global"] if use_global else rope_tensors["cos_local"]
            sin_block_all = rope_tensors["sin_block_global"] if use_global else rope_tensors["sin_block_local"]

            gamma_pre = w["BLK0_ATTN_NORM_WEIGHT"].reshape(self.vector_length)
            pre_norm = _rms_norm_torch(x, gamma_pre)
            q_proj = pre_norm @ w["BLK0_ATTN_Q_WEIGHT_DATA"].T
            k_proj = pre_norm @ w["BLK0_ATTN_K_WEIGHT_DATA"].T
            v_proj = pre_norm @ w["BLK0_ATTN_V_WEIGHT_DATA"].T

            gamma_k = w["BLK0_ATTN_K_NORM_WEIGHT"][: self.head_dim]
            gamma_q = w["BLK0_ATTN_Q_NORM_WEIGHT"][: self.head_dim]
            k_norm = _rms_norm_torch(k_proj, gamma_k)
            q_norm = _rms_norm_torch(q_proj.reshape(1, self.group_size, self.head_dim), gamma_q).reshape(self.group_size, self.head_dim)

            # RoPE for position seq_len (k_norm is (1, head_dim), so k_rope_new (1, head_dim) to match cache (seq_len, head_dim))
            cos_t = cos_all[seq_len]
            sin_t = torch.cat([-sin_block_all[seq_len, :D], sin_block_all[seq_len, D:]], dim=0)
            k_rope_new = _rope_hf_torch(k_norm, cos_t, sin_t)  # (1, head_dim)
            q_rope = torch.stack(
                [_rope_hf_torch(q_norm[g], cos_t, sin_t) for g in range(self.group_size)],
                dim=0,
            )

            cache_k = kv_cache[layer_idx]["k_rope"]
            cache_v = kv_cache[layer_idx]["v"]
            K_full = torch.cat([cache_k, k_rope_new], dim=0)
            V_full = torch.cat([cache_v, v_proj], dim=0)
            # Match compile_decoder: one decoder_attention_core per head group g; same K,V (seq_len, head_dim), Q_g (1, head_dim)
            bias_decoder = torch.zeros(1, seq_len + 1, dtype=torch.bfloat16)
            flash_out_list = []
            for g in range(self.group_size):
                out_g = decoder_attention_torch(
                    q_rope[g : g + 1], K_full, V_full, scale_attn, bias_decoder
                )
                flash_out_list.append(out_g)
            flash_out = torch.cat(flash_out_list, dim=0)
            flash_out_flat = flash_out.reshape(1, self.head_dim * self.group_size)
            attn_out = flash_out_flat @ w["BLK0_ATTN_OUTPUT_WEIGHT_DATA"].T
            gamma_post = w["BLK0_POST_ATTENTION_NORM_WEIGHT"].reshape(self.vector_length)
            post_attn_norm = _rms_norm_torch(attn_out, gamma_post)
            post_attn_residual = x + post_attn_norm

            gamma_ffn = w["BLK0_FFN_NORM_WEIGHT"].reshape(self.vector_length)
            pre_mlp_norm = _rms_norm_torch(post_attn_residual, gamma_ffn)
            gate_out = _gelu_torch(pre_mlp_norm @ w["BLK0_FFN_GATE_WEIGHT_DATA"].T)
            up_out = pre_mlp_norm @ w["BLK0_FFN_UP_WEIGHT_DATA"].T
            mult = gate_out * up_out
            down_out = mult @ w["BLK0_FFN_DOWN_WEIGHT_DATA"].T
            gamma_post_ffn = w["BLK0_POST_FFW_NORM_WEIGHT"].reshape(self.vector_length)
            post_mlp_norm = _rms_norm_torch(down_out, gamma_post_ffn)
            x = post_attn_residual + post_mlp_norm

            if layer_idx == layer_size - 1:
                ref["LAYER0_PRE_NORM_DRAM"] = pre_norm.clone()
                ref["LAYER0_Q_DRAM"] = q_proj.reshape(self.group_size, self.head_dim).clone()
                ref["LAYER0_K_DRAM"] = k_proj.clone()
                ref["LAYER0_V_DRAM"] = v_proj.clone()
                ref["LAYER0_K_NORM_DRAM"] = k_norm.clone()
                ref["LAYER0_Q_NORM_DRAM"] = q_norm.clone()
                ref["LAYER0_K_ROPE_DRAM"] = k_rope_new.clone()
                ref["LAYER0_FLASH_Q_DRAM"] = q_rope.clone()
                ref["LAYER0_FLASH_OUTPUT_DRAM"] = flash_out.clone()
                ref["LAYER0_ATTN_PROJ_OUTPUT_DRAM"] = attn_out.clone()
                ref["LAYER0_POST_ATTN_NORM_DRAM"] = post_attn_norm.clone()
                ref["LAYER0_POST_ATTN_RESIDUAL_DRAM"] = post_attn_residual.clone()
                ref["LAYER0_PRE_MLP_NORM_DRAM"] = pre_mlp_norm.clone()
                ref["LAYER0_MLP_GATE_DRAM"] = gate_out.clone()
                ref["LAYER0_MLP_UP_DRAM"] = up_out.clone()
                ref["LAYER0_MLP_MULT_DRAM"] = mult.clone()
                ref["LAYER0_MLP_DOWN_DRAM"] = down_out.clone()
                ref["LAYER0_POST_MLP_NORM_DRAM"] = post_mlp_norm.clone()
                ref["LAYER0_OUTPUT_DRAM"] = x.clone()

        gamma_out, lm_head_w = self._load_output_norm_and_lm_head_torch()
        out_norm = _rms_norm_torch(x, gamma_out)
        logits = out_norm @ lm_head_w.T
        ref["OUTPUT_NORM_DRAM"] = out_norm
        ref["LOGITS_DRAM"] = logits

        return ref

    def torch_ref_check_prefill(self, layer_size: int) -> dict:
        """
        Compute torch ref for prefill (prefill_seq[:-1] tokens), read intermediates from HW DRAM,
        compare with calculate_snr. Returns kv_cache for decoder ref.
        Uses self.prefill_seq so torch ref matches the same sequence as the HW run.
        """
        assert self.prefill_seq is not None, "prefill_seq is not set"
        prefill_seq = self.prefill_seq
        ref, kv_cache = self._compute_torch_prefill(prefill_seq[:-1], layer_size)
        seq_len = len(prefill_seq) - 1

        tensor_specs = [
            ("LAYER0_PRE_NORM_DRAM", (seq_len, self.vector_length)),
            ("LAYER0_Q_DRAM", (seq_len * self.group_size, self.head_dim)),
            ("LAYER0_K_DRAM", (seq_len, self.head_dim)),
            ("LAYER0_V_DRAM", (seq_len, self.head_dim)),  # per-layer offset applied when reading
            ("LAYER0_K_NORM_DRAM", (seq_len, self.head_dim)),
            ("LAYER0_Q_NORM_DRAM", (seq_len * self.group_size, self.head_dim)),
            ("LAYER0_K_ROPE_DRAM", (seq_len, self.head_dim)),  # layer offset applied when reading
            ("LAYER0_FLASH_Q_DRAM", (seq_len * self.group_size, self.head_dim)),  # Q after RoPE, layout t*group_size+g
            ("LAYER0_FLASH_OUTPUT_DRAM", (seq_len * self.group_size, self.head_dim)),
            ("LAYER0_ATTN_PROJ_OUTPUT_DRAM", (seq_len, self.vector_length)),
            ("LAYER0_POST_ATTN_NORM_DRAM", (seq_len, self.vector_length)),
            ("LAYER0_POST_ATTN_RESIDUAL_DRAM", (seq_len, self.vector_length)),
            ("LAYER0_PRE_MLP_NORM_DRAM", (seq_len, self.vector_length)),
            ("LAYER0_MLP_GATE_DRAM", (seq_len, self.mlp_elements)),
            ("LAYER0_MLP_UP_DRAM", (seq_len, self.mlp_elements)),
            ("LAYER0_MLP_MULT_DRAM", (seq_len, self.mlp_elements)),
            ("LAYER0_MLP_DOWN_DRAM", (seq_len, self.vector_length)),
            ("LAYER0_POST_MLP_NORM_DRAM", (seq_len, self.vector_length)),
            ("LAYER0_OUTPUT_DRAM", (seq_len, self.vector_length)),
        ]

        print("\n--- Torch ref vs hardware SNR check ---")
        all_pass = True
        # Same offset as gemma3_test: K_DRAM_ADDR/V_DRAM_ADDR use layer_idx * MAX_CONTEXT_SIZE * k_size
        layer_idx = layer_size - 1
        layer_kv_offset = layer_idx * self.MAX_CONTEXT_SIZE * self.k_size
        for name, shape in tensor_specs:
            if name not in ref:
                continue
            addr = getattr(self, name, None)
            if addr is None:
                continue
            if name == "LAYER0_V_DRAM":
                addr = self.LAYER0_V_DRAM + layer_kv_offset
            if name == "LAYER0_K_ROPE_DRAM":
                addr = self.LAYER0_K_ROPE_DRAM + layer_kv_offset
            hw = self.dma_from_accelerator_memory(addr, shape)
            r = ref[name]
            assert hw.shape == r.shape, f"name: {name}, Shape mismatch: hw {hw.shape} != torch ref {r.shape}"
            # if name == "LAYER0_PRE_NORM_DRAM":
            #     print(f"hw result for {name} (bf16 from accelerator): {hw} dtype={hw.dtype}")
            #     print(f"ref result for {name} (torch ref, bf16): {r} dtype={r.dtype}")
            snr = calculate_snr(r, hw)
            status = "PASS" if snr >= 19 else "FAIL"
            if snr < 19:
                all_pass = False
            print(f"  {name}: SNR = {snr:.2f} dB [{status}]")
        if all_pass:
            print("All checks passed.")
        else:
            print("Some checks failed (SNR < 19 dB).")
        return kv_cache

    def torch_ref_check_decoder(self, layer_size: int, kv_cache: dict) -> None:
        """
        Torch ref for decoder: one step using kv_cache from prefill; consumes cache in flash
        attention, then LM norm + LM head. Compares all layer-0 intermediates and outputs
        with HW at same DRAM addrs (decoder writes 1 token to each buffer).
        Uses self.prefill_seq so torch ref matches the same sequence as the HW run.
        """
        assert self.prefill_seq is not None, "prefill_seq is not set"
        prefill_seq = self.prefill_seq
        last_token_id = prefill_seq[-1]
        ref = self._compute_torch_decoder(last_token_id, kv_cache, layer_size)

        # HW decoder writes 1 token to base of each buffer; same names as prefill, 1-token shapes
        tensor_specs = [
            ("LAYER0_PRE_NORM_DRAM", (1, self.vector_length)),
            ("LAYER0_Q_DRAM", (self.group_size, self.head_dim)),
            ("LAYER0_K_DRAM", (1, self.head_dim)),
            ("LAYER0_V_DRAM", (1, self.head_dim)),
            ("LAYER0_K_NORM_DRAM", (1, self.head_dim)),
            ("LAYER0_Q_NORM_DRAM", (self.group_size, self.head_dim)),
            ("LAYER0_K_ROPE_DRAM", (1, self.head_dim)),
            ("LAYER0_FLASH_Q_DRAM", (self.group_size, self.head_dim)),
            ("LAYER0_FLASH_OUTPUT_DRAM", (self.group_size, self.head_dim)),
            ("LAYER0_ATTN_PROJ_OUTPUT_DRAM", (1, self.vector_length)),
            ("LAYER0_POST_ATTN_NORM_DRAM", (1, self.vector_length)),
            ("LAYER0_POST_ATTN_RESIDUAL_DRAM", (1, self.vector_length)),
            ("LAYER0_PRE_MLP_NORM_DRAM", (1, self.vector_length)),
            ("LAYER0_MLP_GATE_DRAM", (1, self.mlp_elements)),
            ("LAYER0_MLP_UP_DRAM", (1, self.mlp_elements)),
            ("LAYER0_MLP_MULT_DRAM", (1, self.mlp_elements)),
            ("LAYER0_MLP_DOWN_DRAM", (1, self.vector_length)),
            ("LAYER0_POST_MLP_NORM_DRAM", (1, self.vector_length)),
            ("LAYER0_OUTPUT_DRAM", (1, self.vector_length)),
            ("OUTPUT_NORM_DRAM", (1, self.vector_length)),
            ("LOGITS_DRAM", (1, self.EMBEDDING_ELEMENTS)),
        ]

        print("\n--- Torch ref vs hardware SNR check (decoder, 1 token, all tensors) ---")
        all_pass = True        
        for name, shape in tensor_specs:
            if name not in ref:
                continue
            if name in ("OUTPUT_NORM_DRAM", "LOGITS_DRAM") and layer_size != self.LAYER_SIZE:
                continue
            addr = getattr(self, name, None)
            if addr is None:
                continue
            if name == "LAYER0_V_DRAM":
                addr = self.LAYER0_FLASH_V_DRAM
            if name == "LAYER0_K_ROPE_DRAM":
                addr = self.LAYER0_K_ROPE_DRAM + (layer_size - 1) * 512 * self.k_size + (self.seq_len - 1) * self.k_size
            hw = self.dma_from_accelerator_memory(addr, shape)
            r = ref[name]
            assert hw.shape == r.shape, f"{name}: shape mismatch hw {hw.shape} != ref {r.shape}"
            snr = calculate_snr(r, hw)
            status = "PASS" if snr >= 19 else "FAIL"
            if snr < 19:
                all_pass = False
            print(f"  {name}: SNR = {snr:.2f} dB [{status}]")
        if all_pass:
            print("All decoder checks passed.")
        else:
            print("Some decoder checks failed (SNR < 19 dB).")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Gemma3 layer-0 prefill: run on accelerator, verify with torch ref.")
    parser.add_argument("--layer-size",type=int,default=26,help="Number of layers to compile and verify (default: 1)")
    parser.add_argument("--prompt", type=str, default=None, help="User prompt for prefill; if not set, use config default_prefill_tokens")
    parser.add_argument("--local-weights", action="store_true", help="Use gemma3_bin/full_model_weights.bin instead of generated weights_gemma3_hf.bin")
    parser.add_argument("--dual-engine", action="store_true", help="Use dual engine")
    parser.add_argument('--dev', type=str, default='xdma0',help='DMA device name (e.g., xdma0, xdma1). Default: xdma0')
    parser.add_argument('--cycle', type=float, default=5.62,help='Clock cycle time in nanoseconds (default: 3.0, use 2.5 for alveo)')
    args = parser.parse_args()

    set_dma_device(args.dev)
    gemma3_test.DMA_DEVICE_H2C = user_dma_core.DMA_DEVICE_H2C
    gemma3_test.DMA_DEVICE_C2H = user_dma_core.DMA_DEVICE_C2H
    gemma3_test.DMA_DEVICE_USER = user_dma_core.DMA_DEVICE_USER
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle
    print(f"Using DMA device: {args.dev}")
    print(f"  H2C: {gemma3_test.DMA_DEVICE_H2C}")
    print(f"  C2H: {gemma3_test.DMA_DEVICE_C2H}")
    print(f"  USER: {gemma3_test.DMA_DEVICE_USER}")
    print(f"Setting CLOCK_CYCLE_TIME_NS = {user_dma_core.CLOCK_CYCLE_TIME_NS}")

    dual_engine = args.dual_engine
    ue = Gemma3_NumericEngine(local_weights=args.local_weights, dual_engine=dual_engine)
    ue.set_prefill_seq(args.prompt)

    if dual_engine:
        ue2 = Gemma3_NumericEngine(local_weights=args.local_weights, dual_engine=True, engine_slave=True)
        ue2.set_prefill_seq(args.prompt)

    print(f"\n--- Compiling prefill (layer_size={args.layer_size}) ---")
    prefill_program_addr, flops_prefill = ue.compile_prefill(seq_len=len(ue.prefill_seq), layer_size=args.layer_size)
    
    if dual_engine:
        print(f"\n--- Compiling prefill for dual engine (layer_size={args.layer_size}) ---")
        prefill_program_addr2, flops_prefill2 = ue2.compile_prefill(seq_len=len(ue2.prefill_seq), layer_size=args.layer_size)
        ue2.program_execute(prefill_program_addr2, timeout=0, flops=flops_prefill2)

    print(f"\n--- Running prefill ---")
    timer=time.perf_counter()
    latency_prefill, _ = ue.run_prefill(prefill_program_addr, flops=flops_prefill)
    print(f"Prefill done in {(time.perf_counter() - timer):.2f} seconds.")
    if dual_engine:
        print(f"Dual engine prefill gflops: {((flops_prefill + flops_prefill2) / (latency_prefill * 1e3)):.2f} GFLOPS")

    print(f"\n--- Torch ref check ---")
    kv_cache = ue.torch_ref_check_prefill(layer_size=args.layer_size)

    # generate_trace(ue, f"gemma3_prefill_trace.csv", clock_period_ns=args.cycle)

    print(f"\n--- Compiling decoder (.bin file)---")
    timer_dec = time.perf_counter()
    decoder_bin_path, decoder_program_sizes, flops_per_token = ue.compile_decoder(layer_size=args.layer_size)
    print(f"Decoder compile done in {time.perf_counter() - timer_dec:.2f} seconds.")
    decoder_base_addr, _ = ue.load_instructions(decoder_bin_path)

    print(f"\n--- Running decoder ---")
    ue.MAX_CONTEXT_SIZE = len(ue.prefill_seq)
    timer=time.perf_counter()
    token_cnt_decoded, latency_hw_decoder, flop_rate_hw_decoder = ue.run_decoder(decoder_program_sizes, decoder_base_addr, token_id=ue.prefill_seq[-1], flops_per_token=flops_per_token)
    print(f"\nDecoder done in {(time.perf_counter() - timer):.2f} seconds.")
    print(f"hw decoder latency: {latency_hw_decoder / 1e6:.2f} seconds, gflops: {flop_rate_hw_decoder / (token_cnt_decoded - len(ue.prefill_seq) + 1):.2f} GFLOPS, token/s: {(token_cnt_decoded - len(ue.prefill_seq) + 1) / (time.perf_counter() - timer):.2f}, total {token_cnt_decoded} tokens")

    print(f"\n--- Torch ref check decoder ---")
    ue.torch_ref_check_decoder(layer_size=args.layer_size, kv_cache=kv_cache)

    print("Gemma3 numeric verification done.")


if __name__ == "__main__":
    main()
