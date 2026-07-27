#!/usr/bin/env python3
"""
Gemma3 pure-host checkpoint reference generator.

Computes Gemma3 prefill and one decoder step on the host, then stores each
checkpoint tensor as a separate .pt file for comparison against hardware runs.
This script does not reset hardware, allocate accelerator DRAM, compile ISA,
execute queues, or use DMA.

Usage:
  python gemma3_numeric.py [--layer-size 26] [--prompt "..."]

Same layout as gemma3_test: this file lives next to gemma3_bin/ and *.json.
"""

import json
import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure pcie_utils is on path so shared host modules import from any cwd.
_pcie_utils_dir = os.path.dirname(os.path.dirname(_THIS_DIR))
if _pcie_utils_dir not in sys.path:
    sys.path.insert(0, _pcie_utils_dir)

import torch

from quant_lib import dequant

import gemma3_test
from gemma3_test import (
    Gemma3_UnifiedEngine,
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
    """Dequantize current Gemma3 IF4 weight format from the packed weight bin."""
    scale_bytes = weight_bin[scale_off : scale_off + scale_sz]
    data_bytes = weight_bin[data_off : data_off + data_sz]
    return dequant(gemma3_test.QUANT_PRECISION, data_bytes, scale_bytes, N, K, block_size)


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
    Pure host Gemma3 reference computation. This intentionally does not call
    UnifiedEngine.__init__(), reset hardware, allocate DRAM, compile, or DMA.
    """

    def __init__(self, script_dir: str | None = None, local_weights: bool = False):
        self.script_dir = script_dir or SCRIPT_DIR
        self._cfg = self.load_config(script_dir=self.script_dir)
        self.weight_defs = self._cfg["_weight_defs"]

        fi = self._cfg["file_info"]
        model = self._cfg["model"]
        paths = self._cfg["paths"]

        self.vector_length = fi["hidden_size"]
        self.head_dim = fi["head_dim"]
        self.bytes_per_element = fi["bytes_per_element"]
        self.group_size = fi["group_size"]
        self.mlp_elements = fi["mlp_elements"]
        self.q_size = self.head_dim * self.group_size * self.bytes_per_element
        self.k_size = self.head_dim * self.bytes_per_element
        self.MAX_CONTEXT_SIZE = model["max_context_size"]
        self.PREFILL_CONTEXT_SIZE = model["prefill_context_size"]
        self.PREFILL_MAX_SEQ_LEN = int(model["prefill_max_seq_len"])
        self.LAYER_SIZE = fi["num_layers"]
        self.EMBEDDING_ELEMENTS = fi["embedding_vocab"]
        self.causal_mask_upper = False
        self._rope_global_layers = set(model["rope_global_layers"])
        self._end_of_turn_token_id = model["end_of_turn_token_id"]
        self.prefill_seq = None

        self._weights_bin_rel = "gemma3_bin/full_model_weights.bin" if local_weights else paths["weights_bin"]
        self._load_host_weights()

    def _load_host_weights(self) -> None:
        full_path = os.path.join(self.script_dir, self._weights_bin_rel)
        if not os.path.exists(full_path):
            print(f"Weight bin not found, generating: {full_path}")
            gemma3_test.weight_bin_generate(output_path=full_path)
        with open(full_path, "rb") as f:
            self.weight_bin = f.read()

        model, model_dir = gemma3_test._ensure_hf_model(self.script_dir, self._cfg)
        embed = model.get_input_embeddings().weight.detach().cpu().to(torch.bfloat16)
        embedding_scale = model.config.hidden_size ** 0.5
        self.embedding_weight = (embed.float() * embedding_scale).to(torch.bfloat16)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

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
        new_kv_all_layers: dict[int, dict] = {}

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

            new_kv_all_layers[layer_idx] = {"k_rope": k_rope_new.clone(), "v": v_proj.clone()}

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

        return ref, new_kv_all_layers

def _save_checkpoint_tensors(root: str, stage: str, tensors: dict[str, torch.Tensor]) -> dict:
    stage_dir = os.path.join(root, stage)
    os.makedirs(stage_dir, exist_ok=True)
    manifest = {}
    for name, tensor in tensors.items():
        path = os.path.join(stage_dir, f"{name}.pt")
        t = tensor.detach().cpu().contiguous()
        torch.save(t, path)
        manifest[name] = {
            "path": os.path.relpath(path, root),
            "shape": list(t.shape),
            "dtype": str(t.dtype),
        }
    return manifest


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate pure-host Gemma3 checkpoint references.")
    parser.add_argument("--layer-size", type=int, default=26, help="Number of layers to reference.")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for prefill; default uses config tokens.")
    parser.add_argument("--local-weights", action="store_true", help="Use gemma3_bin/full_model_weights.bin.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "gemma3_bin", "host_ref_checkpoints"),
        help="Directory for host reference checkpoint .pt files.",
    )
    args = parser.parse_args()

    ue = Gemma3_NumericEngine(local_weights=args.local_weights)
    ue.set_prefill_seq(args.prompt)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating pure-host Gemma3 refs: layer_size={args.layer_size}")
    print(f"Output dir: {args.output_dir}")

    prefill_ref, kv_cache = ue._compute_torch_prefill(ue.prefill_seq[:-1], args.layer_size)
    decoder_ref, _ = ue._compute_torch_decoder(ue.prefill_seq[-1], kv_cache, args.layer_size)

    manifest = {
        "prompt_tokens": list(ue.prefill_seq),
        "prefill_tokens": list(ue.prefill_seq[:-1]),
        "decoder_token": int(ue.prefill_seq[-1]),
        "layer_size": args.layer_size,
        "weights_bin": ue._weights_bin_rel,
        "prefill": _save_checkpoint_tensors(args.output_dir, "prefill", prefill_ref),
        "decoder": _save_checkpoint_tensors(args.output_dir, "decoder", decoder_ref),
    }
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved {len(prefill_ref)} prefill checkpoints and {len(decoder_ref)} decoder checkpoints.")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
