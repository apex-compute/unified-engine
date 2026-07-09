#!/usr/bin/env python3
"""Qwen3-4B inference from pre-compiled bins (self-contained offline runner).

This script is self-contained: it does NOT import from qwen3_4b_test.py. It
ships with everything needed to load weights, initialize tensors, load a
pre-compiled instruction bin from disk, and run prefill + decode.

Requirements on disk (relative to this file):
  - ``qwen3_4b_bin/qwen3_4b_config.json``   model config
  - ``qwen3_4b_bin/params.bin`` quantized weight bin
  - ``qwen3_4b_bin/params.json`` weight-bin sidecar
  - ``qwen3_4b_bin/programs.bin`` pre-compiled program bin
  - ``qwen3_4b_bin/programs.json`` compile-meta sidecar
  - ``qwen3_4b_bin/Qwen3-4B/`` tokenizer files

If any of those are missing, exit early with a clear message. Generate them
on a build machine that has HF access by running qwen3_4b_test.py once.

Architecture notes:
  - 36 layers, 32 Q / 8 KV heads (group_size=4), actual head_dim=128.
  - hidden_size=2560, mlp_intermediate=9728, vocab=151936.
  - QK RMSNorm per head (gamma_offset=0.0). No post-attn/post-FFN norm.
  - SwiGLU activation. Single RoPE base theta=1_000_000.
  - Separate lm_head weight (not tied to embedding).
  - Last layer (L35) routed through bf16 to mitigate INT4 amplification.

Usage:
  python qwen3_4b_run_from_bin.py
  python qwen3_4b_run_from_bin.py --prompt "your prompt"
  python qwen3_4b_run_from_bin.py --dev xdma0 [--cycle 5.62]
  python qwen3_4b_run_from_bin.py --local-weights
"""

import json
import math
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import time

# This file's folder; user_dma_core.py is two folders up (repo root); that directory is added to sys.path.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

import user_dma_core
from user_dma_core import DMA_DEVICE_H2C, TYPE, UE_MODE, UE_VECTOR_SIZE, SCALE_BRAM_ELEMENTS, set_dma_device, ue_35bit_addr_shifter
from user_dma_core import UnifiedEngine

# --- BROAD PRINT SUPPRESSION FOR LIBRARIES ---
import builtins

_original_print = builtins.print
_SILENT_MODE = False

def quiet_print(*args, **kwargs):
    """Suppress prints when _SILENT_MODE is True; otherwise print normally."""
    if _SILENT_MODE:
        return
    _original_print(*args, **kwargs)

builtins.print = quiet_print
# ---------------------------------------------

# down_proj K=6144 ≤ SCALE_BRAM_ELEMENTS=8192 — no K-split needed.
# (Previously was 8960 which required splitting; mlp_elements corrected to 6144.)

def _parse_offset(val) -> int:
    """Parse offset/size from JSON: int or hex string like '0x24000000'."""
    if isinstance(val, str):
        return int(val, 0)
    return int(val)

def _quantize_bf16_to_int4_packed(weight_bf16: torch.Tensor, block_size: int = 64) -> tuple[bytes, bytes]:
    """Quantize bf16 weight (N_w, K_w) to INT4 packed + scale per block of 64 along K. Returns (data_bytes, scale_bytes).
    Scale is stored with negative sign so HW IF4 path dispatches as INT4 (sign(bf16 scale)=neg → INT4 codebook)."""
    w = weight_bf16.detach().cpu().float().reshape(-1)
    N_w, K_w = weight_bf16.shape
    assert K_w % block_size == 0
    w_blocks = w.reshape(N_w, K_w // block_size, block_size)
    scale = w_blocks.abs().amax(dim=-1).clamp(min=1e-8) / 7.0
    scale_bf16 = (-scale).to(torch.bfloat16)
    w_int8 = (w_blocks / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int8)
    w_nibbles = w_int8.numpy().astype(np.int16) & 0x0F
    low = w_nibbles[:, :, 0::2].reshape(N_w, -1)
    high = w_nibbles[:, :, 1::2].reshape(N_w, -1)
    packed = (high << 4) | low
    data_bytes = packed.astype(np.uint8).tobytes()
    scale_bytes = scale_bf16.contiguous().view(torch.uint8).numpy().tobytes()
    return (data_bytes, scale_bytes)

def weight_bin_generate(script_dir: str | None = None, output_path: str | None = None) -> str:
    """Generate params.bin from Hugging Face model per qwen3_4b_config.json layout.
    Returns the path to the written file. Use this bin to replace full_model_weights.bin."""
    script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
    cfg = _load_config(script_dir)
    weight_defs = cfg["_weight_defs"]
    paths = cfg["paths"]
    paths_full = os.path.join(script_dir, paths["weights_bin"])
    out_path = output_path or paths_full

    model, model_dir = _ensure_hf_model(script_dir, cfg)
    gamma_offset = cfg["special"]["rms_norm"]["gamma_offset"]  # 0.0 for Qwen3
    emb_cfg = cfg["special"]["embedding"]
    token_embd_offset = _parse_offset(emb_cfg["token_embd_offset"])
    token_embd_size = _parse_offset(emb_cfg["token_embd_size"])
    LAYER_WEIGHT_SIZE = weight_defs["LAYER_WEIGHT_SIZE"]
    base_layer0 = weight_defs["BLK0_ATTN_NORM_WEIGHT"]
    num_layers = cfg["file_info"]["num_layers"]
    actual_head_dim = cfg["file_info"]["actual_head_dim"]
    blk0_structure = cfg["layers"]["structure"]

    # Compute total file size: max(offset + size) over all regions
    max_end = 0
    for key, r in cfg.get("regions", {}).items():
        off = weight_defs[key]
        max_end = max(max_end, off + (num_layers - 1) * LAYER_WEIGHT_SIZE + r["size"])
    for key, r in cfg.get("non_layer_regions", {}).items():
        off = weight_defs[key]
        max_end = max(max_end, off + r["size"])
    max_end = max(max_end, token_embd_offset + token_embd_size)
    buf = bytearray(max_end)

    def write_at(offset: int, data: bytes) -> None:
        buf[offset : offset + len(data)] = data[: len(buf) - offset]

    # Embedding
    embed = model.get_input_embeddings().weight.detach().cpu().to(torch.bfloat16)
    raw_emb = embed.contiguous().view(torch.uint8).numpy().tobytes()
    write_at(token_embd_offset, raw_emb)

    # Layers
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn

        # Qwen3 norms: gamma_offset=0.0 (stored as-is)
        gamma_in = (layer.input_layernorm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
        q_w = attn.q_proj.weight.detach().cpu().to(torch.bfloat16)
        k_w = attn.k_proj.weight.detach().cpu().to(torch.bfloat16)
        v_w = attn.v_proj.weight.detach().cpu().to(torch.bfloat16)
        # Qwen3 has QK norm per actual head dim (128); store as (actual_head_dim,)
        gamma_q = (attn.q_norm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
        gamma_k = (attn.k_norm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
        o_w = attn.o_proj.weight.detach().cpu().to(torch.bfloat16)
        # Qwen3: no post-attention norm — write zero placeholder (pipeline skips this step)
        gamma_post = torch.zeros(cfg["file_info"]["hidden_size"], dtype=torch.bfloat16)
        # Qwen3: post_attention_layernorm IS the pre-FFN norm (same as LLaMA)
        gamma_ffn = (layer.post_attention_layernorm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
        gate_w = layer.mlp.gate_proj.weight.detach().cpu().to(torch.bfloat16)
        up_w = layer.mlp.up_proj.weight.detach().cpu().to(torch.bfloat16)
        down_w = layer.mlp.down_proj.weight.detach().cpu().to(torch.bfloat16)
        # Qwen3: no post-FFN norm — write zero placeholder (pipeline skips this step)
        gamma_post_ffn = torch.zeros(cfg["file_info"]["hidden_size"], dtype=torch.bfloat16)

        # K=6144 ≤ SCALE_BRAM_ELEMENTS=8192 — single down_proj (no split needed)
        region_writes = [
            (gamma_in, "bf16"),
            (q_w, "int4"),
            (k_w, "int4"),
            (v_w, "int4"),
            (gamma_q, "bf16"),
            (gamma_k, "bf16"),
            (o_w, "int4"),
            (gamma_post, "bf16"),
            (gamma_ffn, "bf16"),
            (up_w, "int4"),
            (gate_w, "int4"),
            (down_w, "int4"),
            (gamma_post_ffn, "bf16"),
        ]
        j = 0
        i = 0
        while i < len(blk0_structure):
            off_key = blk0_structure[i]["key"]
            sz_key = f"{off_key}_SIZE"
            off = weight_defs[off_key]
            sz = weight_defs[sz_key]
            file_off = off + layer_idx * LAYER_WEIGHT_SIZE
            tensor, kind = region_writes[j]
            if kind == "int4":
                next_key = blk0_structure[i + 1]["key"]
                data_sz = weight_defs[f"{next_key}_SIZE"]
                data_bytes, scale_bytes = _quantize_bf16_to_int4_packed(tensor)
                scale_padded = (scale_bytes + b"\x00" * sz)[:sz]
                data_padded = (data_bytes + b"\x00" * data_sz)[:data_sz]
                write_at(file_off, scale_padded)
                data_off = weight_defs[next_key] + layer_idx * LAYER_WEIGHT_SIZE
                write_at(data_off, data_padded)
                i += 2
            else:
                t = tensor.detach().cpu().to(torch.bfloat16).contiguous()
                raw = (t.view(torch.uint8).numpy().tobytes() + b"\x00" * sz)[:sz]
                write_at(file_off, raw)
                i += 1
            j += 1

    # ROPE: single base theta for all layers (rope_global_layers is empty)
    # Per-head frequencies: D_per_head = actual_head_dim // 2 = 64
    rope_cfg = cfg["special"]["rope"]
    theta = rope_cfg["theta"]
    num_positions = rope_cfg["num_positions"]
    D_per_head = actual_head_dim // 2  # 64
    inv_freq = 1.0 / (theta ** (torch.arange(D_per_head, dtype=torch.float32) / D_per_head))
    pos = torch.arange(num_positions, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)         # (num_positions, 64)
    cos_ = freqs.cos().to(torch.bfloat16)      # (num_positions, 64)
    sin_ = freqs.sin().to(torch.bfloat16)      # (num_positions, 64)
    # rope_hf_core(N=actual_head_dim=128) layout: [cos(64), cos(64), -sin(64), sin(64)]
    rope_tensor = torch.cat([cos_, cos_, -sin_, sin_], dim=1)  # (num_positions, 256)
    for off_key, sz_key in [("ROPE_LOCAL", "ROPE_LOCAL_SIZE"), ("ROPE_GLOBAL", "ROPE_GLOBAL_SIZE")]:
        sz = weight_defs[sz_key]
        raw = (rope_tensor.contiguous().view(torch.uint8).numpy().tobytes() + b"\x00" * sz)[:sz]
        write_at(weight_defs[off_key], raw)

    # OUTPUT_NORM
    out_norm = (model.model.norm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
    sz = weight_defs["OUTPUT_NORM_WEIGHT_SIZE"]
    raw = (out_norm.contiguous().view(torch.uint8).numpy().tobytes() + b"\x00" * sz)[:sz]
    write_at(weight_defs["OUTPUT_NORM_WEIGHT"], raw)

    # LM_HEAD: Qwen3 has a separate lm_head weight (NOT tied to embedding)
    lm_head_w = model.lm_head.weight.detach().cpu().to(torch.bfloat16)
    scale_sz = weight_defs["LM_HEAD_WEIGHT_SCALE_SIZE"]
    data_sz = weight_defs["LM_HEAD_WEIGHT_DATA_SIZE"]
    data_bytes, scale_bytes = _quantize_bf16_to_int4_packed(lm_head_w)
    scale_padded = (scale_bytes + b"\x00" * scale_sz)[:scale_sz]
    data_padded = (data_bytes + b"\x00" * data_sz)[:data_sz]
    write_at(weight_defs["LM_HEAD_WEIGHT_SCALE"], scale_padded)
    write_at(weight_defs["LM_HEAD_WEIGHT_DATA"], data_padded)

    with open(out_path, "wb") as f:
        f.write(buf)
    print(f"Generated weights bin: {out_path} ({len(buf)} bytes)")
    return out_path

def _ensure_hf_model(script_dir: str, cfg: dict):
    """Ensure HF model is downloaded and loaded. Returns (model, model_dir). Single place for download + load."""
    model_dir = os.path.join(script_dir, cfg["paths"]["hf_model_dir"])
    hf_repo = cfg["paths"]["hf_model_repo"]
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        _original_print(f"Downloading HF model {hf_repo} to {os.path.abspath(model_dir)} ...")
        snapshot_download(repo_id=hf_repo, local_dir=model_dir, local_dir_use_symlinks=False)
        _original_print("Download complete.")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map=None, trust_remote_code=True
    )
    return model, model_dir

def _load_config(script_dir: str) -> dict:
    """Load qwen3_4b_config.json and build weight_defs (offset/size dict) from regions."""
    config_path = os.path.join(script_dir, "qwen3_4b_config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)
    weight_defs = {"LAYER_WEIGHT_SIZE": cfg["file_info"]["layer_size"]}
    for key, r in cfg.get("regions", {}).items():
        weight_defs[key] = _parse_offset(r["offset"])
        weight_defs[f"{key}_SIZE"] = r["size"]
    for key, r in cfg.get("non_layer_regions", {}).items():
        weight_defs[key] = _parse_offset(r["offset"])
        weight_defs[f"{key}_SIZE"] = r["size"]
    cfg["_weight_defs"] = weight_defs
    return cfg

# -----------------------------------------------------------------------------
# Qwen3-4B unified engine
# -----------------------------------------------------------------------------
class Qwen3_4b_UnifiedEngine(UnifiedEngine):
    """UnifiedEngine for Qwen3-4B: loads config + weight bin, compile_prefill/compile_decoder, run_prefill/run_decoder.

    Key architectural differences from Gemma3:
      - QK RMSNorm per actual head dim (128), gamma_offset=0.0.
      - No post-attention norm: residual applied directly to o_proj output.
      - No post-FFN norm: residual applied directly to down_proj output.
      - post_attention_layernorm in HF IS the pre-FFN norm.
      - Embedding NOT scaled by sqrt(hidden_size).
      - Separate lm_head weight (not tied to embedding).
      - SwiGLU (SiLU gate): silu_enable=True on gate_proj.
      - Single RoPE theta=1_000_000 for all layers; rope applied per head (N=128).
      - Per-KV-head flash attention (8 KV heads x 128 dim); group_size=2.
    """

    def __init__(self, script_dir: str | None = None, hf_model_dir: str | None = None, weights_bin: str | None = None):
        # Qwen3-4B DRAM layout (weights ~2.15 GB → needs more params region than 1.7B):
        #   params:       0x00000000 – 0x90000000 (~2.25 GB)
        #   tensors:      0x90000000 – 0xE0000000 (~1.25 GB, KV cache + activations)
        #   instructions: 0xE0000000 – 0x100000000 (~512 MB, compiled bin)
        super().__init__(
            params_dram_base=0x00000000,
            tensor_dram_base=0x90000000,
            program_dram_base=0xE0000000,
        )
        self.script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
        self._cfg = _load_config(self.script_dir)
        self.weight_defs = self._cfg["_weight_defs"]

        fi = self._cfg["file_info"]
        model = self._cfg["model"]
        paths = self._cfg["paths"]

        self.vector_length = fi["hidden_size"]           # 2048
        self.head_dim = fi["head_dim"]                   # 1024 (= num_kv_heads * actual_head_dim)
        self.actual_head_dim = fi["actual_head_dim"]     # 128
        self.num_kv_heads = fi["num_kv_heads"]           # 8
        self.bytes_per_element = fi["bytes_per_element"] # 2
        self.group_size = fi["group_size"]               # 2 (Q heads per KV head)
        self.mlp_elements = fi["mlp_elements"]           # 8960
        self.hf_model_dir = hf_model_dir or os.path.join(self.script_dir, paths["hf_model_dir"])
        # q_size = total Q output dim * bpe = (num_kv_heads * actual_head_dim * group_size) * bpe
        self.q_size = self.head_dim * self.group_size * self.bytes_per_element   # 1024*2*2 = 4096
        self.k_size = self.head_dim * self.bytes_per_element                     # 1024*2 = 2048
        self.MAX_CONTEXT_SIZE = model["max_context_size"]
        self.PREFILL_MAX_SEQ_LEN = int(model.get("prefill_max_seq_len", 256))
        self.LAYER_SIZE = fi["num_layers"]
        self.EMBEDDING_ELEMENTS = fi["embedding_vocab"]
        # Dynamic-PBI GPR layout (see core_changes.md §3a):
        #   reg 1 = TMP_REG            — scratch for reg_mul_imm + add_imm address math
        #   reg 2 = GPR_SEQ_LEN_REG    — runtime row count for matmul/norm/rope (M=seq_len ops)
        #   reg 3 = GPR_Q_SEQ_LEN_REG  — for ops with M = seq_len * group_size (Q-side norms/rope)
        #   reg 4 = GPR_ALIGNED_SEQ_LEN_REG — 64-aligned seq_len for dynamic unified_attention_core
        # Dynamic GPR allocation (via alloc_isa_reg) starts at 5; PBI op-internal loop counters
        # consume from there and release back at loop_end.
        fixed = self._cfg.get("fixed_isa_regs", {})
        self.TMP_REG            = fixed["TMP_REG"]
        self.gpr_seq_len        = fixed["GPR_SEQ_LEN_REG"]
        self.gpr_q_seq_len      = fixed["GPR_Q_SEQ_LEN_REG"]
        self.gpr_aligned_seq_len = fixed["GPR_ALIGNED_SEQ_LEN_REG"]
        self._isa_reg_counter = 5
        self.causal_mask_upper = False
        self._end_of_turn_token_id = model["end_of_turn_token_id"]

        # Single shared identity matrix slot (UE_VECTOR_SIZE × UE_VECTOR_SIZE,
        # populated by tensor_init). Passed as IDENTITY_DRAM_ADDR= to every
        # attention/transpose kernel call site. See Trick 4 in
        # shared_design_notes.md.

        bin_path = weights_bin or paths["weights_bin"]
        full_path = os.path.join(self.script_dir, bin_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Bin file not found: {full_path}")
        with open(full_path, "rb") as f:
            self.weight_bin = f.read()
        self.weight_init()
        self.tensor_init()

    # ---- On-FPGA repetition penalty (the sole decode path; no host sampling). The penalty
    # is folded into the LM-head matmul as its per-vocab additive C bias, argmaxed ON CHIP —
    # the HW argmax register holds the penalized token id, no logit readback. ----
    def _structural_token_ids(self) -> set:
        """Token ids never repetition-penalized: punctuation, whitespace, newline, specials."""
        cached = getattr(self, "_struct_ids_cache", None)
        if cached is not None:
            return cached
        import string
        allowed = set(string.punctuation) | set(string.whitespace) | set("—–’‘“”…·•‹›«»¡¿")
        ids = set(int(i) for i in (getattr(self.tokenizer, "all_special_ids", []) or []))
        for i in range(self.EMBEDDING_ELEMENTS):
            s = self.tokenizer.decode([i]).strip()
            if s == "" or all(ch in allowed for ch in s):
                ids.add(i)
        self._struct_ids_cache = ids
        return ids

    def _structural_ids_tensor(self) -> torch.Tensor:
        t = getattr(self, "_struct_ids_tensor_cache", None)
        if t is None:
            t = torch.tensor(sorted(self._structural_token_ids()), dtype=torch.long)
            self._struct_ids_tensor_cache = t
        return t

    def _write_penalty_bias(self, prev_tokens) -> None:
        """bias[t] = clamp(−alpha·count[t], min=−cap) over the last rep_window tokens (structural
        tokens stay 0); DMA to PENALTY_BIAS_DRAM so the HW argmax of (logits+bias) is penalized."""
        vocab = self.EMBEDDING_ELEMENTS
        alpha = float(getattr(self, "pen_alpha", 1.0))
        cap = float(getattr(self, "pen_cap", 20.0))
        W = int(getattr(self, "rep_window", 256))
        window = prev_tokens[-W:]
        count = torch.zeros(vocab, dtype=torch.float32)
        if window:
            win = torch.tensor(window, dtype=torch.long)
            count.index_add_(0, win, torch.ones(win.numel(), dtype=torch.float32))
            count[self._structural_ids_tensor()] = 0.0
        bias = (-alpha * count).clamp(min=-cap).to(torch.bfloat16).view(1, vocab)
        self.dma_to_accelerator_memory(self.PENALTY_BIAS_DRAM, bias)

    def _load_last_layer_bf16(self) -> None:
        """Load the last layer's 7 matmul weights as bf16 from the HF model and
        DMA them to params DRAM. The prefill+decoder emit below routes layer
        L=LAYER_SIZE-1 matmuls to these bf16 addresses via the bf16 matmul path,
        eliminating the INT4 quantization noise that's amplified by L35's
        extreme RMSNorm gammas (which otherwise cascades into long-context
        filler-token lock-in past ~1500 decoded tokens).

        Disable via env ``QWEN3_L27_BF16=0`` (env name kept for cross-model
        consistency with qwen3_1.7b) if DRAM is constrained.
        """
        self._last_layer_bf16_addrs: dict[str, int] = {}
        if os.environ.get("QWEN3_L27_BF16", "1") in ("0", "false", "no"):
            print(f"  QWEN3_L27_BF16=0 → keeping L{self.LAYER_SIZE-1} as INT4 (long-context output may degrade).")
            return
        import time as _time
        print(f"  Loading HF model L{self.LAYER_SIZE-1} weights as bf16 (mitigates INT4 amplification) ...")
        _t0 = _time.perf_counter()
        hf_model, _ = _ensure_hf_model(self.script_dir, self._cfg)
        layer = hf_model.model.layers[self.LAYER_SIZE - 1]
        weights = {
            "q":    layer.self_attn.q_proj.weight.detach().cpu().to(torch.bfloat16).contiguous(),
            "k":    layer.self_attn.k_proj.weight.detach().cpu().to(torch.bfloat16).contiguous(),
            "v":    layer.self_attn.v_proj.weight.detach().cpu().to(torch.bfloat16).contiguous(),
            "o":    layer.self_attn.o_proj.weight.detach().cpu().to(torch.bfloat16).contiguous(),
            "gate": layer.mlp.gate_proj.weight.detach().cpu().to(torch.bfloat16).contiguous(),
            "up":   layer.mlp.up_proj.weight.detach().cpu().to(torch.bfloat16).contiguous(),
            "down": layer.mlp.down_proj.weight.detach().cpu().to(torch.bfloat16).contiguous(),
        }
        total_bytes = 0
        for name, w in weights.items():
            sz = w.numel() * 2  # bf16
            addr = self.allocate_params_dram(sz)
            raw = w.view(torch.uint8).numpy().tobytes()
            self.dma_write(DMA_DEVICE_H2C, addr, raw, sz)
            self._last_layer_bf16_addrs[name] = addr
            total_bytes += sz
        del hf_model
        print(f"  bf16 L{self.LAYER_SIZE-1} weights DMA'd: {total_bytes / 1024 / 1024:.1f} MB "
              f"in {_time.perf_counter() - _t0:.1f}s")

    def get_embedding_for_tokens(self, token_ids: list[int] | tuple) -> torch.Tensor:
        """Return (len(token_ids), vector_length) bfloat16 tensor from self.embedding_weight (no scaling)."""
        tid_t = torch.tensor(token_ids, dtype=torch.long)
        out = torch.zeros(len(token_ids), self.vector_length, dtype=torch.bfloat16)
        valid = tid_t < self.embedding_weight.shape[0]
        out[valid] = self.embedding_weight[tid_t[valid]]
        return out

    def _load_rope_host(self, rope_theta: float | None = None) -> None:
        """Generate per-head RoPE (cos, cos, -sin, sin) on host and write to DRAM.
        D_per_head = actual_head_dim // 2 = 64. Both ROPE_LOCAL and ROPE_GLOBAL use the
        same theta (rope_global_layers is empty for Qwen3)."""
        rope_cfg = self._cfg["special"]["rope"]
        theta = rope_theta if rope_theta is not None else rope_cfg["theta"]
        num_rope_positions = rope_cfg["num_positions"]
        D_per_head = self.actual_head_dim // 2   # 64
        inv_freq = 1.0 / (theta ** (torch.arange(D_per_head, dtype=torch.float32) / D_per_head))
        pos = torch.arange(num_rope_positions, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)                     # (num_positions, 64)
        cos_ = freqs.cos().to(torch.bfloat16)
        sin_ = freqs.sin().to(torch.bfloat16)
        # rope_hf_core(N=actual_head_dim=128): [cos(64), cos(64), -sin(64), sin(64)]
        rope_tensor = torch.cat([cos_, cos_, -sin_, sin_], dim=1)  # (num_positions, 256)
        # Single DMA: write ROPE_LOCAL and ROPE_GLOBAL back-to-back (same table, same theta)
        rope_raw = rope_tensor.contiguous().view(torch.uint8).numpy().tobytes()
        local_sz  = self.weight_defs["ROPE_LOCAL_SIZE"]
        global_sz = self.weight_defs["ROPE_GLOBAL_SIZE"]
        local_raw  = (rope_raw + b"\x00" * local_sz)[:local_sz]
        global_raw = (rope_raw + b"\x00" * global_sz)[:global_sz]
        rope_buf = local_raw + global_raw
        rope_base = self.allocate_params_dram(len(rope_buf))
        self.dma_write(DMA_DEVICE_H2C, rope_base, rope_buf, len(rope_buf))
        self.DRAM_ADDR_ROPE_LOCAL  = rope_base
        self.DRAM_ADDR_ROPE_GLOBAL = rope_base + local_sz

    def weight_init(self) -> None:
        """Initialize DRAM from weight bin: load embedding+tokenizer offline from
        cached files, layers from bin, host-computed RoPE, then OUTPUT_NORM/LM_HEAD
        from bin. The embedding is read straight out of ``self.weight_bin`` (no HF
        model needed at this stage); the tokenizer is loaded with
        ``local_files_only=True``. ``main()`` runs ``weight_bin_generate`` first
        on a fresh machine — it handles the HF download — so by the time we get
        here both the weight bin and the tokenizer files exist locally."""
        emb_cfg = self._cfg["special"]["embedding"]
        token_embd_offset = _parse_offset(emb_cfg["token_embd_offset"])
        vocab_size = emb_cfg["vocab_size"]
        emb_dim = emb_cfg["embedding_dim"]
        emb_bytes = vocab_size * emb_dim * self.bytes_per_element
        raw_emb = bytearray(self.weight_bin[token_embd_offset : token_embd_offset + emb_bytes])
        self.embedding_weight = torch.frombuffer(raw_emb, dtype=torch.bfloat16).reshape(vocab_size, emb_dim).clone()
        model_dir = os.path.join(self.script_dir, self._cfg["paths"]["hf_model_dir"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True, local_files_only=True,
        )

        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        base_layer0 = self.weight_defs["BLK0_ATTN_NORM_WEIGHT"]
        blk0_regions = [
            (s["key"], f"{s['key']}_SIZE", s["attr"])
            for s in self._cfg["layers"]["structure"]
        ]
        non_layer = [
            (s["key"], f"{s['key']}_SIZE", s["attr"])
            for s in self._cfg["layers"]["non_layer"]
            if s["key"] not in ("ROPE_LOCAL", "ROPE_GLOBAL")  # RoPE loaded via _load_rope_host()
        ]

        layer0_end = (self.weight_defs["BLK0_POST_FFW_NORM_WEIGHT"] - base_layer0
                      + self.weight_defs["BLK0_POST_FFW_NORM_WEIGHT_SIZE"])
        assert layer0_end <= LAYER_WEIGHT_SIZE, (
            f"Layer 0 size mismatch: computed {layer0_end} > LAYER_WEIGHT_SIZE {LAYER_WEIGHT_SIZE}"
        )

        import time as _time
        print(f"\n--- Weights DRAM allocation, start at DRAM address: {self.get_params_dram_addr()} ---")
        layers_total = self.LAYER_SIZE * LAYER_WEIGHT_SIZE
        layers_base_dram = self.allocate_params_dram(layers_total)

        # Single large DMA for all layer weights: the bin file stores layers at base_layer0
        # with the same LAYER_WEIGHT_SIZE stride as DRAM, so data maps directly.
        # This avoids 28*20=560 small DMA calls (each with PCIe round-trip overhead).
        bin_layers_start = base_layer0
        print(f"  DMA layers: {layers_total // 1024 // 1024} MB → DRAM 0x{layers_base_dram:X}...", flush=True)
        _t0 = _time.perf_counter()
        self.dma_write(DMA_DEVICE_H2C, layers_base_dram,
                       self.weight_bin[bin_layers_start : bin_layers_start + layers_total],
                       layers_total)
        print(f"  DMA layers done in {_time.perf_counter() - _t0:.1f}s")
        # Set layer-0 DRAM attribute addresses from known offsets (same for all layers)
        for off_key, sz_key, attr in blk0_regions:
            offset_in_layer = self.weight_defs[off_key] - base_layer0
            setattr(self, attr, layers_base_dram + offset_in_layer)
        print(f"Layers 0..{self.LAYER_SIZE - 1} loaded: 0x{layers_base_dram:X} size {layers_total} (LAYER_WEIGHT_SIZE={LAYER_WEIGHT_SIZE})")

        # Single large DMA for all non-layer weights (OUTPUT_NORM + LM_HEAD).
        # Assemble a contiguous buffer then transfer in one shot.
        nl_slices = [self.weight_bin[self.weight_defs[k] : self.weight_defs[k] + self.weight_defs[s]]
                     for k, s, _ in non_layer]
        nl_buf = b"".join(nl_slices)
        print(f"  DMA non-layer weights: {len(nl_buf) // 1024 // 1024} MB...", flush=True)
        _t0 = _time.perf_counter()
        nl_base_dram = self.allocate_params_dram(len(nl_buf))
        self.dma_write(DMA_DEVICE_H2C, nl_base_dram, nl_buf, len(nl_buf))
        print(f"  DMA non-layer done in {_time.perf_counter() - _t0:.1f}s")
        nl_offset = 0
        for off_key, sz_key, attr in non_layer:
            setattr(self, attr, nl_base_dram + nl_offset)
            nl_offset += self.weight_defs[sz_key]

        self._load_rope_host()

        # Load last-layer matmul weights as bf16 to mitigate INT4 amplification
        # at L35's extreme RMSNorm gammas (default-on; QWEN3_L27_BF16=0 to opt out).
        # NOTE: the bin's baked addresses + matmul instruction shape depend on
        # this toggle, so delete the cached *_instruction.bin if you flip it.
        self._load_last_layer_bf16()

        print(f"    Allocate weights end at DRAM address: 0x{self.get_params_dram_addr():X}, usage: {self.get_params_dram_usage()} bytes")
        print("Tokenizer loaded successfully.")

    def tensor_init(self) -> None:
        """Initialize hardware DRAM tensors for Qwen3-4B.

        KV cache layout (per layer, per KV head):
          LAYER0_V_DRAM[layer][kv_h][t]  (MAX_CONTEXT_SIZE * actual_head_dim per head)
          LAYER0_K_ROPE_DRAM[layer][kv_h][t]  (same shape)
        KV cache is placed last in the activation region so it can grow with max_context_size.
        """
        seq_len = self.MAX_CONTEXT_SIZE
        # Qwen3: q_seq_len = seq_len * group_size (2 Q heads per KV head)
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        ahd = self.actual_head_dim   # 128
        nkvh = self.num_kv_heads     # 8
        bpe = self.bytes_per_element

        print(f"Allocate tensor dram start at DRAM address: 0x{self.get_tensor_dram_addr():X}")

        # --- Fixed tensors (reused per layer, do not grow with KV cache) ---
        # Constant zero and identity buffers
        zero_add = torch.zeros(seq_len * self.head_dim * bpe, dtype=torch.bfloat16)
        self.ZERO_DRAM_ADDR = self.allocate_tensor_dram(seq_len * self.head_dim * bpe)
        self.dma_to_accelerator_memory(self.ZERO_DRAM_ADDR, zero_add)
        # Single UE_VECTOR_SIZE × UE_VECTOR_SIZE identity matrix reused by
        # unified_attention_core's V^T transpose. Passed as IDENTITY_DRAM_ADDR= at
        # every call site; bin bakes one address.
        self.IDENTITY_DRAM_ADDR = self.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe)
        self.dma_to_accelerator_memory(self.IDENTITY_DRAM_ADDR, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))
        self.PREFILL_ALIGNED_SEQ_LEN = ((self.PREFILL_MAX_SEQ_LEN * self.group_size + 63) // 64) * 64
        self.DECODER_ALIGNED_SEQ_LEN = ((self.MAX_CONTEXT_SIZE + 63) // 64) * 64
        aligned_q_max = self.PREFILL_ALIGNED_SEQ_LEN
        self.LAYER0_FLASH_ATTN_P_DRAM = self.allocate_tensor_dram(aligned_q_max * aligned_q_max * bpe)

        # Per-head flash attention buffers: one KV head at a time, reused across heads
        # FLASH_Q: (q_seq_len_aligned, ahd) for one KV head's Q group (group_size=2 Q heads)
        self.LAYER0_FLASH_Q_DRAM = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        self.LAYER0_FLASH_K_DRAM = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        self.LAYER0_FLASH_V_DRAM = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        zero_pad = torch.zeros(aligned_seq_len * ahd * bpe, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_Q_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_K_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_V_DRAM, zero_pad)

        # Per-head flash output (seq_len * group_size, ahd); reused across KV heads
        self.LAYER0_FLASH_OUT_HEAD_DRAM = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        # Final assembled flash output (seq_len, head_dim * group_size) = (seq_len, 2048)
        self.LAYER0_FLASH_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.head_dim * self.group_size * bpe)
        def _unified_scratch_elems(batch: int, aligned: int) -> int:
            return (ahd + aligned) * aligned + batch * ahd
        scratch_elems = max(
            _unified_scratch_elems(self.PREFILL_ALIGNED_SEQ_LEN, self.PREFILL_ALIGNED_SEQ_LEN),
            _unified_scratch_elems(self.group_size, self.DECODER_ALIGNED_SEQ_LEN),
        )
        self.LAYER0_FLASH_SCRATCH_DRAM = self.allocate_tensor_dram(scratch_elems * bpe)
        bias_elems = max(
            self.PREFILL_ALIGNED_SEQ_LEN * self.PREFILL_ALIGNED_SEQ_LEN,
            self.group_size * self.DECODER_ALIGNED_SEQ_LEN,
        )
        self.LAYER0_FLASH_BIAS_DRAM = self.allocate_tensor_dram(bias_elems * bpe)

        # Layer intermediate tensors
        self.LAYER0_INPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_Q_DRAM = self.allocate_tensor_dram(seq_len * self.q_size)
        self.LAYER0_K_DRAM = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_K_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_Q_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.q_size)
        # V proj temp buffer: standard layout (seq_len, head_dim) = (seq_len, nkvh * ahd)
        self.LAYER0_V_PROJ_TEMP = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_ATTN_PROJ_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        # POST_ATTN_NORM_DRAM: Qwen3 has no post-attn norm; allocated but unused in pipeline
        self.LAYER0_POST_ATTN_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_ATTN_RESIDUAL_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_MLP_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_MLP_GATE_DRAM = self.allocate_tensor_dram(seq_len * self.mlp_elements * 2)
        self.LAYER0_MLP_UP_DRAM = self.allocate_tensor_dram(seq_len * self.mlp_elements * 2)
        self.LAYER0_MLP_MULT_DRAM = self.allocate_tensor_dram(seq_len * self.mlp_elements * 2)
        self.LAYER0_MLP_DOWN_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        # K=6144 fits in SCALE_BRAM directly — no split buffers needed
        # POST_MLP_NORM_DRAM: Qwen3 has no post-FFN norm; allocated but unused in pipeline
        self.LAYER0_POST_MLP_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.OUTPUT_NORM_DRAM = self.allocate_tensor_dram(1 * self.vector_length * bpe)
        self.LOGITS_DRAM = self.allocate_tensor_dram(1 * self.EMBEDDING_ELEMENTS * bpe)

        # --- KV cache (at tail of activation region, grows with MAX_CONTEXT_SIZE) ---
        # Per-head contiguous layout: [layer][kv_head][position] = ahd elements per position
        kv_cache_total = self.LAYER_SIZE * nkvh * self.MAX_CONTEXT_SIZE * ahd * bpe
        self.LAYER0_V_DRAM = self.allocate_tensor_dram(kv_cache_total)
        self.LAYER0_K_ROPE_DRAM = self.allocate_tensor_dram(kv_cache_total)
        zero_kv = torch.zeros(kv_cache_total, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, zero_kv)
        self.dma_to_accelerator_memory(self.LAYER0_K_ROPE_DRAM, zero_kv)

        # On-FPGA repetition penalty bias (LM-head matmul C term). MUST be allocated LAST,
        # exactly as in qwen3_4b_test.py, so its address matches the one baked into the bin.
        self.PENALTY_BIAS_DRAM = self.allocate_tensor_dram(1 * self.EMBEDDING_ELEMENTS * self.bytes_per_element)
        self.dma_to_accelerator_memory(self.PENALTY_BIAS_DRAM,
                                       torch.zeros(1, self.EMBEDDING_ELEMENTS, dtype=torch.bfloat16))

        print(f"    Allocate tensor dram end at DRAM address: 0x{self.get_tensor_dram_addr():X}, usage: {self.get_tensor_dram_usage()} bytes")

    def run_prefill(self, prefill_program_addr: int, preamble_addr: int,
                    prefill_seq, gflops: int = None) -> dict:
        """Run prefill via dynamic-PBI runtime preamble.

        Emits a tiny preamble program at ``preamble_addr`` that primes the three
        runtime GPRs (gpr_seq_len, gpr_q_seq_len, gpr_aligned_seq_len) and then
        unconditional-jumps into the cached prefill program. Single cached bin
        handles any actual_seq_len ≤ PREFILL_MAX_SEQ_LEN — no padding needed.

        Args:
            prefill_program_addr: address of the cached prefill program in DRAM.
            preamble_addr: pre-reserved DRAM slot for the runtime preamble.
                Caller (compile_instructions or run_from_bin's loader) reserved
                this via ``allocate_program_dram(SMALL_SIZE)`` once at startup.
            prefill_seq: full tokenized prompt; last token is decoder seed.
            gflops: FLOPS estimate (from meta).
        """
        if prefill_seq is None:
            raise ValueError("run_prefill: prefill_seq is required (caller must tokenize via apply_chat_template).")
        if len(prefill_seq) <= 1:
            raise ValueError("Prefill sequence must have at least 2 tokens.")

        # Prefill processes all but the last token (last token seeds the decoder).
        prefill_seq = prefill_seq[:-1]
        actual_seq_len = len(prefill_seq)
        if actual_seq_len > self.PREFILL_MAX_SEQ_LEN:
            raise ValueError(
                f"Prompt too long: actual_seq_len={actual_seq_len} > PREFILL_MAX_SEQ_LEN={self.PREFILL_MAX_SEQ_LEN}. "
                f"Rebuild the bin with a larger prefill_max_seq_len in config."
            )
        self.seq_len = actual_seq_len

        q_seq_len = actual_seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64

        # Zero the intermediate buffers that the prefill scatter loop reads at
        # ``template_seq_len`` rows. With ``gpr_M_reg=gpr_seq_len``, ops compute only
        # ``actual_seq_len`` rows but the static-unrolled scatter still iterates
        # ``template_seq_len`` rows, so slots [actual_seq_len..template_seq_len) would
        # otherwise pick up stale DRAM (potentially NaN bits from prior runs) and
        # propagate NaN through K/V cache → attention → all layers.
        ahd = self.actual_head_dim
        nkvh = self.num_kv_heads
        qpkv = self.group_size
        template = self.PREFILL_MAX_SEQ_LEN
        # K_NORM, Q_NORM, V_PROJ_TEMP are sized for MAX_CONTEXT_SIZE × per_token_K/Q
        # but only the first `template * (nkvh|nkvh*qpkv)` rows are read by the scatter.
        # Zero those leading regions to defeat stale-NaN.
        zero_kvq = torch.zeros(template * nkvh * qpkv * ahd, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_K_NORM_DRAM, zero_kvq[: template * nkvh * ahd])
        self.dma_to_accelerator_memory(self.LAYER0_Q_NORM_DRAM, zero_kvq[: template * nkvh * qpkv * ahd])
        self.dma_to_accelerator_memory(self.LAYER0_V_PROJ_TEMP, zero_kvq[: template * nkvh * ahd])

        # DMA inputs: embedding (actual_seq_len rows) and bias mask.
        embedding_tensor = self.get_embedding_for_tokens(prefill_seq)
        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
        bias_one_group = torch.full((aligned_seq_len, aligned_seq_len), float("-inf"), dtype=torch.bfloat16)
        valid_mask = torch.tril(torch.ones(aligned_seq_len, aligned_seq_len, dtype=torch.bool), diagonal=0) if not self.causal_mask_upper else torch.triu(torch.ones(aligned_seq_len, aligned_seq_len, dtype=torch.bool), diagonal=0)
        bias_one_group.masked_fill_(valid_mask, 0.0)
        bias_one_group[:, q_seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_one_group)

        # Emit runtime preamble: ADD_SETs for the 3 GPRs + JUMP_ABS into prefill.
        # Same slot is reused across calls (overwritten each time).
        self.clear_inst_id()
        self.start_capture()
        self.generate_instruction_add_set(self.gpr_seq_len,         actual_seq_len)
        self.generate_instruction_add_set(self.gpr_q_seq_len,       q_seq_len)
        self.generate_instruction_add_set(self.gpr_aligned_seq_len, aligned_seq_len)
        self.generate_instruction_jump_abs(ue_35bit_addr_shifter(prefill_program_addr))
        self.stop_capture()
        self.write_captured_instructions_to_dram(preamble_addr)
        self.clear_capture_buffer()

        # Execute from the preamble — it jumps into the cached prefill, which halts.
        self.program_execute(preamble_addr, timeout=120.0, flops=gflops)

    def run_decoder(self, decoder_program_addr: int, preamble_addr: int,
                    token_id: int, gflops_per_token: int | None = None) -> dict:
        """Run decode loop via dynamic-PBI runtime preamble.

        Each step emits a small preamble at ``preamble_addr`` that primes
        gpr_seq_len (current decode position, ==previous token count) and
        gpr_aligned_seq_len (64-aligned KV context length), then jump_abs's into
        the cached decoder program. Same preamble slot is overwritten per token.

        Args:
            decoder_program_addr: address of the cached decoder program in DRAM.
            preamble_addr: pre-reserved DRAM slot for runtime preamble (same
                slot reused as run_prefill).
            token_id: seed token (last token of the prompt).
            gflops_per_token: single FLOPS estimate for the decoder program.
        """
        if token_id is None:
            print("No last token available for decode.")
            return {}

        # Qwen3 stop tokens: <|im_end|>=151645, <|endoftext|>=151643
        _qwen3_stop_tokens = {151643, 151645, self._end_of_turn_token_id}

        global _SILENT_MODE
        max_seq_len = self.MAX_CONTEXT_SIZE

        # On-FPGA repetition-penalty state (the LM-head matmul already adds PENALTY_BIAS_DRAM;
        # token selection is always the HW argmax — no host logit readback). Position-gated:
        # pure greedy for the first `greedy_until` decoded tokens, then the bias turns on.
        if not hasattr(self, "_generated_tokens"):
            self._generated_tokens = []
        _fpga_penalty = bool(getattr(self, "fpga_penalty", True))
        _greedy_until = int(getattr(self, "greedy_until", 512))
        _prompt_len = len(self._generated_tokens)
        self.dma_to_accelerator_memory(self.PENALTY_BIAS_DRAM,
                                       torch.zeros(1, self.EMBEDDING_ELEMENTS, dtype=torch.bfloat16))

        # Live decode status bar (llama3.2-style): pin the bottom terminal row as a status
        # line via an ANSI scroll region; tokens stream above it and the counter refreshes in
        # place. Only on a real TTY (skipped when piped/redirected, so logs stay clean).
        import shutil
        _dec_timer = time.perf_counter()
        _seq_len_start = self.seq_len
        _use_status = sys.stdout.isatty()
        def _status_setup():
            rows = shutil.get_terminal_size().lines
            sys.stdout.write(f"\033[1;{rows - 1}r")    # scroll region = rows 1..rows-1
            sys.stdout.write(f"\033[{rows - 1};1H")    # park cursor at bottom of region
            sys.stdout.flush()
        def _status_update():
            rows = shutil.get_terminal_size().lines
            n = self.seq_len - _seq_len_start
            elapsed = time.perf_counter() - _dec_timer
            rate = n / elapsed if elapsed > 0 else 0.0
            sys.stdout.write("\0337")                  # save cursor
            sys.stdout.write(f"\033[{rows};1H\033[2K") # bottom row, clear it
            sys.stdout.write(f" decoding… {n} tokens  (pos {self.seq_len}/{self.MAX_CONTEXT_SIZE})  "
                             f"{elapsed:.1f}s  {rate:.1f} tok/s")
            sys.stdout.write("\0338")                  # restore cursor
            sys.stdout.flush()
        def _status_teardown():
            rows = shutil.get_terminal_size().lines
            sys.stdout.write("\033[r")                 # reset scroll region
            sys.stdout.write(f"\033[{rows};1H\033[2K") # clear the status row
            sys.stdout.flush()
        if _use_status:
            _status_setup()

        while self.seq_len < max_seq_len:
            _SILENT_MODE = True
            # self.seq_len at entry is the count of K/V already in cache; the
            # current decode token will be written at that index.
            decode_pos = self.seq_len               # K/V cache write index for this token
            new_ctx_len = decode_pos + 1            # KV positions [0..decode_pos] inclusive
            aligned_ctx = ((new_ctx_len + 63) // 64) * 64
            embedding_tensor = self.get_embedding_for_tokens([token_id])
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
            # unified_attention_core uses full-matrix bias, one row per Q head in
            # the KV group. Mask positions past the live context so stale cache rows
            # never enter softmax.
            bias_host = torch.full((1, aligned_ctx), float("-inf"), dtype=torch.bfloat16)
            bias_host[0, :new_ctx_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_host.repeat(self.group_size, 1))

            # Decode preamble: prime gpr_aligned_seq_len only; gpr_seq_len is carried over
            # from prefill (or previous decode step) via the in-bin ADD_INC.
            self.clear_inst_id()
            self.start_capture()
            self.generate_instruction_add_set(self.gpr_aligned_seq_len, aligned_ctx)
            self.generate_instruction_jump_abs(ue_35bit_addr_shifter(decoder_program_addr))
            self.stop_capture()
            self.write_captured_instructions_to_dram(preamble_addr)
            self.clear_capture_buffer()

            # Refresh the per-vocab penalty bias (this step's LM-head C term) once past the
            # gate, BEFORE execute, so the HW argmax of (logits + bias) is already penalized.
            if _fpga_penalty and (len(self._generated_tokens) - _prompt_len) > _greedy_until:
                self._write_penalty_bias(self._generated_tokens)

            self.program_execute(preamble_addr, timeout=10.0, flops=gflops_per_token)
            # Read the HW argmax register — the LM-head matmul already added the penalty bias
            # on chip, so this is the penalized (or pure-greedy) token. No host logit readback.
            token_id = self.get_arg_max_index()
            self._generated_tokens.append(token_id)
            token_char = self.tokenizer.decode([token_id])
            _SILENT_MODE = False
            self.seq_len += 1
            if token_id in _qwen3_stop_tokens:
                if _use_status:
                    _status_teardown()
                print(f"\nStop token {token_id} reached.")
                break
            print(token_char, end="", flush=True)
            if _use_status:
                _status_update()
        else:
            if _use_status:
                _status_teardown()
        return self.seq_len

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Offline runner — no compile machinery. Reads meta JSON from disk, loads the
# pre-compiled bin to DRAM, runs prefill + decoder.
# -----------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3-4B inference from pre-compiled bins (offline)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt (default: from qwen3_4b_config.json default_prompt)")
    parser.add_argument("--local-weights", action="store_true",
                        help="Use qwen3_4b_bin/full_model_weights.bin instead of params.bin")
    parser.add_argument("--dev", type=str, default="xdma0",
                        help="DMA device name (default: xdma0)")
    parser.add_argument("--cycle", type=float, default=5.62,
                        help="Clock cycle time in ns (default: 5.62ns ≈ peak 22.8 GFLOPS)")
    # Deterministic on-FPGA decode: token selection is always the HW argmax of
    # (logits + penalty bias). No host sampling — the repetition penalty is folded into
    # the LM-head matmul bias (notes_repetition_penalty_fpga_bias.md).
    parser.add_argument('--pure-greedy', action='store_true',
                        help='Disable the on-FPGA repetition penalty — plain greedy '
                             '(bias stays all-zero). Penalty is ENABLED by default.')
    pen_group = parser.add_argument_group('on-FPGA repetition penalty (active unless --pure-greedy)')
    pen_group.add_argument('--greedy-until', type=int, default=512,
                        help='Pure greedy for the first N decoded tokens, then the penalty turns on. Default 512.')
    pen_group.add_argument('--pen-alpha', type=float, default=1.0,
                        help='bias[t] = -alpha*count[t] (logit units). Default 1.0.')
    pen_group.add_argument('--pen-cap', type=float, default=20.0,
                        help='max |bias| per token (floor on -alpha*count). Default 20.')
    pen_group.add_argument('--rep-window', type=int, default=256,
                        help='count tokens over the last N (structural tokens exempt). Default 256.')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    bin_dir = os.path.join(script_dir, "qwen3_4b_bin")

    weights_bin_rel = ("qwen3_4b_bin/full_model_weights.bin" if args.local_weights
                       else "qwen3_4b_bin/params.bin")
    weights_bin_full = os.path.join(script_dir, weights_bin_rel)

    # Hard-fail BEFORE any FPGA / HF touch if a required local file is missing.
    missing = []
    if not os.path.exists(weights_bin_full):
        missing.append(os.path.relpath(weights_bin_full, script_dir))
    for name in ("programs.bin", "programs.json"):
        if not os.path.exists(os.path.join(bin_dir, name)):
            missing.append(name)
    tokenizer_dir = os.path.join(bin_dir, "Qwen3-4B")
    if not (os.path.exists(os.path.join(tokenizer_dir, "tokenizer.json")) or
            os.path.exists(os.path.join(tokenizer_dir, "tokenizer_config.json"))):
        missing.append("Qwen3-4B/{tokenizer.json,tokenizer_config.json}")
    if missing:
        _original_print("Missing local files (run qwen3_4b_test.py first on a build machine with HF access):")
        for f in missing:
            _original_print(f"  {f}")
        sys.exit(1)

    set_dma_device(args.dev)
    # Mirror test.py: rebind the device-name module globals after set_dma_device
    # so sample_next_token's dma_read(DMA_DEVICE_C2H, ...) resolves (and tracks
    # the chosen --dev). Without this, DMA_DEVICE_C2H is undefined in this module.
    global DMA_DEVICE_H2C, DMA_DEVICE_C2H, DMA_DEVICE_USER
    DMA_DEVICE_H2C = user_dma_core.DMA_DEVICE_H2C
    DMA_DEVICE_C2H = user_dma_core.DMA_DEVICE_C2H
    DMA_DEVICE_USER = user_dma_core.DMA_DEVICE_USER
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle
    user_dma_core.UE_PEAK_GFLOPS = 0.128 / args.cycle
    _original_print(f"Setting CLOCK_CYCLE_TIME_NS = {user_dma_core.CLOCK_CYCLE_TIME_NS}, "
                    f"UE_PEAK_GFLOPS = {user_dma_core.UE_PEAK_GFLOPS:.4f}")

    global _SILENT_MODE
    _SILENT_MODE = True
    ue = Qwen3_4b_UnifiedEngine(script_dir=script_dir, weights_bin=weights_bin_rel)
    _SILENT_MODE = False

    cfg = ue._cfg
    user_prompt = args.prompt if args.prompt is not None else cfg.get("default_prompt", "What is 3 + 5?")
    system_prompt = cfg.get("default_system_prompt", "You are a helpful assistant.")
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    prompt_with_template = ue.tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True,
    )
    prefill_seq = tuple(ue.tokenizer.encode(prompt_with_template, add_special_tokens=False))
    _original_print(f"User prompt ({len(prefill_seq)} tokens): {user_prompt!r}")

    # Wire the on-FPGA penalty config onto the engine so run_decoder can read it.
    ue.fpga_penalty = not bool(args.pure_greedy)
    ue.greedy_until = int(args.greedy_until)
    ue.pen_alpha = float(args.pen_alpha)
    ue.pen_cap = float(args.pen_cap)
    ue.rep_window = int(args.rep_window)
    ue._generated_tokens = list(prefill_seq)   # seed the penalty window with the prompt
    if ue.fpga_penalty:
        _original_print(f"On-FPGA penalty: alpha={ue.pen_alpha} cap={ue.pen_cap} "
                        f"rep_window={ue.rep_window} greedy_until={ue.greedy_until}")
    else:
        _original_print("Pure greedy (on-FPGA penalty disabled).")

    # Read meta JSON directly from disk — no compile_instructions call.
    paths_cfg = cfg.get("paths", {})
    inst_bin_path = os.path.join(script_dir, paths_cfg.get("instruction_bin",
                                  "qwen3_4b_bin/programs.bin"))
    meta_path = os.path.splitext(inst_bin_path)[0] + ".json"
    with open(meta_path) as _f:
        inst_meta = json.load(_f)
    _original_print(f"  Loaded compile meta from {os.path.relpath(meta_path, script_dir)}")
    # The released bin must carry the penalty-bias LM head; a stale bin from an older
    # build (no bias / writeback-on) would silently decode without the penalty.
    if inst_meta.get("lm_head_sig") != "penalty_bias_argmax":
        raise SystemExit(
            f"Instruction bin lm_head_sig={inst_meta.get('lm_head_sig')!r} != 'penalty_bias_argmax'. "
            "Recompile the bin via qwen3_4b_test.py (it now folds the repetition penalty into "
            "the LM-head matmul bias).")

    _original_print(f"\n--- Loading unified instruction bin ---")
    timer = time.perf_counter()
    base_addr, total_size = ue.load_program_instructions_from_file(inst_bin_path)
    preamble_addr = ue.get_program_dram_addr()
    ue.allocate_program_dram(128)
    _original_print(f"  Loaded {total_size} B at 0x{base_addr:X}; preamble slot at 0x{preamble_addr:X} "
                    f"({time.perf_counter() - timer:.3f}s)")

    prefill_program_addr = _parse_offset(inst_meta["prefill_program_start_addr"])
    decoder_program_addr = _parse_offset(inst_meta["decoder_program_start_addr"])
    decoder_total_flops  = inst_meta["decoder_total_flops"]

    actual_seq_len = len(prefill_seq) - 1
    template_seq_len = int(inst_meta["prefill_template_seq_len"])
    gflops_prefill = inst_meta["prefill_template_flops"] * actual_seq_len // max(template_seq_len, 1)

    _original_print(f"\n--- Starting prefill (actual {actual_seq_len} tokens, dynamic seq_len) ---")
    timer = time.perf_counter()
    ue.run_prefill(prefill_program_addr, preamble_addr, prefill_seq=prefill_seq, gflops=gflops_prefill)
    latency_prefill = time.perf_counter() - timer
    _original_print(f"  Prefill done in {latency_prefill:.2f}s")

    _original_print(f"\n--- Starting decoder ---")
    timer = time.perf_counter()
    token_cnt = ue.run_decoder(decoder_program_addr, preamble_addr,
                               token_id=prefill_seq[-1], gflops_per_token=decoder_total_flops)
    latency_decoder = time.perf_counter() - timer
    decoded_tokens = max(token_cnt - len(prefill_seq) + 1, 1)
    _original_print(f"\nDecoder done in {latency_prefill + latency_decoder:.2f}s, "
                    f"speed: {decoded_tokens / latency_decoder:.2f} tokens/s, total {token_cnt} tokens.")


if __name__ == "__main__":
    main()
