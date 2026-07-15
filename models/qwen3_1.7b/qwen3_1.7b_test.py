#!/usr/bin/env python3
"""
Qwen3-1.7B inference on accelerator: prefill + decode.

  - Config from qwen3_1.7b_bin/qwen3_1.7b_config.json; weights from a single bin (see below).
  - Prefill: compiled each run. Decoder: if qwen3_1.7b_bin/programs.bin and
    qwen3_1.7b_bin/programs.json exist, skip decoder compile and load
    program sizes from meta; otherwise compile and write the bin + meta.
  - Run prefill then decode loop. For numeric verification use qwen3_1.7b_numeric.py.

Architecture notes vs Gemma3:
  - 28 layers, 16 Q heads / 8 KV heads, actual head_dim=128.
  - QK RMSNorm per head (gamma_offset=0.0, like Gemma3 but no +1).
  - No post-attention norm; residual applied directly to o_proj output (like LLaMA).
  - No post-FFN norm; residual applied directly to down_proj output (like LLaMA).
  - post_attention_layernorm in HF IS the pre-FFN norm (like LLaMA).
  - Embedding NOT scaled by sqrt(hidden_size) (like LLaMA).
  - Separate lm_head weight (NOT tied to embedding).
  - SwiGLU activation: silu_enable=True on gate_proj (like LLaMA).
  - Single RoPE base theta=1_000_000 (no local/global split).
  - RoPE applied per head (N=128) without lo|hi permutation.

Weights:
  - Default: qwen3_1.7b_bin/params.bin (generated from HF model if missing).
  - --local-weights: use qwen3_1.7b_bin/full_model_weights.bin instead.

Usage:
  python qwen3_1.7b_test.py
  python qwen3_1.7b_test.py --prompt "your prompt"
  python qwen3_1.7b_test.py --dev xdma0 [--cycle 5.88]
  python qwen3_1.7b_test.py --local-weights
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
from user_dma_core import DMA_DEVICE_H2C, TYPE, UE_FMAX_CONTEXT_SIZE, UE_MODE, UE_VECTOR_SIZE, SCALE_BRAM_ELEMENTS, INSTRUCTION_SIZE_BYTES, set_dma_device, ue_35bit_addr_shifter
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
    """Generate params.bin from Hugging Face model per qwen3_1.7b_config.json layout.
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
    """Load qwen3_1.7b_config.json and build weight_defs (offset/size dict) from regions."""
    config_path = os.path.join(script_dir, "qwen3_1.7b_config.json")
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
# Qwen3-1.7B unified engine
# -----------------------------------------------------------------------------
class Qwen3_1_7b_UnifiedEngine(UnifiedEngine):
    """UnifiedEngine for Qwen3-1.7B: loads config + weight bin, compile_prefill/compile_decoder, run_prefill/run_decoder.

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
        # Qwen3 DRAM layout starting from 0x00000000 (4 GB DRAM):
        #   params: 0x00000000 – 0x58000000 (~1.4 GB, covers layers + LM_HEAD + ROPE)
        #   tensors: 0x58000000 – 0x98000000 (~1 GB, intermediates + KV cache)
        #   instructions: 0x98000000 – 0xA0000000 (128 MB)
        super().__init__(
            params_dram_base=0x00000000,
            tensor_dram_base=0x58000000,
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
        #   reg 4 = GPR_ALIGNED_SEQ_LEN_REG — 64-aligned seq_len for the dynamic unified_attention_core
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

    # ---- On-FPGA repetition penalty (the sole decode path; no host sampling) ----
    # The penalty is folded into the LM-head matmul as its per-vocab additive C bias
    # (bias_mode="broadcast_N"): logits = W·h + bias, argmaxed ON CHIP, so the HW
    # argmax register already holds the penalized token id — no logit readback, no
    # host sort, fully deterministic. See notes_repetition_penalty_fpga_bias.md.
    def _structural_token_ids(self) -> set:
        """Token ids that must NEVER be repetition-penalized: punctuation, whitespace,
        newline, and special tokens. These 'glue' tokens recur constantly; penalizing
        them over a long generation starves the model of grammar → word-salad. Cached."""
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
        """1-D LongTensor of the structural/special token ids (cached)."""
        t = getattr(self, "_struct_ids_tensor_cache", None)
        if t is None:
            t = torch.tensor(sorted(self._structural_token_ids()), dtype=torch.long)
            self._struct_ids_tensor_cache = t
        return t

    def _write_penalty_bias(self, prev_tokens) -> None:
        """Build the per-vocab additive bias from the windowed token frequency and DMA it
        to PENALTY_BIAS_DRAM (the LM-head matmul's C term). bias[t] = clamp(−alpha·count[t],
        min=−cap); structural tokens stay 0. The HW argmax of (logits + bias) then returns
        the penalized token id — no logit readback. One full-buffer DMA per step."""
        vocab = self.EMBEDDING_ELEMENTS
        alpha = float(getattr(self, "pen_alpha", 1.0))
        cap = float(getattr(self, "pen_cap", 20.0))
        W = int(getattr(self, "rep_window", 256))
        window = prev_tokens[-W:]
        count = torch.zeros(vocab, dtype=torch.float32)
        if window:
            win = torch.tensor(window, dtype=torch.long)
            count.index_add_(0, win, torch.ones(win.numel(), dtype=torch.float32))
            count[self._structural_ids_tensor()] = 0.0   # never penalize punctuation/whitespace/specials
        bias = (-alpha * count).clamp(min=-cap).to(torch.bfloat16).view(1, vocab)
        self.dma_to_accelerator_memory(self.PENALTY_BIAS_DRAM, bias)

    def get_embedding_for_tokens(self, token_ids: list[int] | tuple) -> torch.Tensor:
        """Return (len(token_ids), vector_length) bfloat16 tensor from self.embedding_weight (no scaling)."""
        tid_t = torch.tensor(token_ids, dtype=torch.long)
        out = torch.zeros(len(token_ids), self.vector_length, dtype=torch.bfloat16)
        valid = tid_t < self.embedding_weight.shape[0]
        out[valid] = self.embedding_weight[tid_t[valid]]
        return out

    def _load_last_layer_bf16(self) -> None:
        """Load the last layer's 7 matmul weights as bf16 from the HF model and
        DMA them to params DRAM. The prefill+decoder emit below routes layer
        L=LAYER_SIZE-1 matmuls to these bf16 addresses via the bf16 matmul path,
        eliminating the INT4 quantization noise that's amplified by L27's
        extreme RMSNorm gammas (max=153 vs ~1 elsewhere) — which otherwise
        cascades into long-context "the the the..." / "and and and..." filler
        token lock-in past ~1500 decoded tokens.

        ~100 MB of params DRAM (50 M params × 2 bytes bf16). Disable via env
        ``QWEN3_L27_BF16=0`` if DRAM is constrained.
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
        # Free the HF model now that we've extracted what we need.
        del hf_model
        print(f"  bf16 L{self.LAYER_SIZE-1} weights DMA'd: {total_bytes / 1024 / 1024:.1f} MB "
              f"in {_time.perf_counter() - _t0:.1f}s")

    def _matmul_b_kwargs(self, layer_idx: int, proj: str) -> dict:
        """Return the matmat_mul_core B/scale/quant kwargs for a layer projection.
        Routes the last layer (L=LAYER_SIZE-1) through bf16 when QWEN3_L27_BF16
        is enabled and `_maybe_load_last_layer_bf16` has populated the addrs;
        all other layers go through the standard INT4 path.

        proj is one of: 'q', 'k', 'v', 'o', 'gate', 'up', 'down'.
        """
        if (layer_idx == self.LAYER_SIZE - 1
                and self._last_layer_bf16_addrs
                and proj in self._last_layer_bf16_addrs):
            return {
                "B_DRAM_ADDR":   self._last_layer_bf16_addrs[proj],
                "is_B_quantized": False,
            }
        # Default INT4 path
        proj_to_attrs = {
            "q":    ("DRAM_ADDR_LAYER0_Q_PROJ_QUANT",     "DRAM_ADDR_LAYER0_Q_PROJ_SCALE"),
            "k":    ("DRAM_ADDR_LAYER0_K_PROJ_QUANT",     "DRAM_ADDR_LAYER0_K_PROJ_SCALE"),
            "v":    ("DRAM_ADDR_LAYER0_V_PROJ_QUANT",     "DRAM_ADDR_LAYER0_V_PROJ_SCALE"),
            "o":    ("DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT",  "DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE"),
            "gate": ("DRAM_ADDR_LAYER0_MLP_GATE_QUANT",   "DRAM_ADDR_LAYER0_MLP_GATE_SCALE"),
            "up":   ("DRAM_ADDR_LAYER0_MLP_UP_QUANT",     "DRAM_ADDR_LAYER0_MLP_UP_SCALE"),
            "down": ("DRAM_ADDR_LAYER0_MLP_DOWN_QUANT",   "DRAM_ADDR_LAYER0_MLP_DOWN_SCALE"),
        }
        quant_attr, scale_attr = proj_to_attrs[proj]
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        layer_off = layer_idx * LAYER_WEIGHT_SIZE
        return {
            "B_DRAM_ADDR":     getattr(self, quant_attr) + layer_off,
            "is_B_quantized":  True,
            "data_type":       TYPE.IF4,
            "SCALE_DRAM_ADDR": getattr(self, scale_attr) + layer_off,
        }

    def _decode_matmul(self, layer_idx: int, proj: str, *,
                       M: int, K: int, N: int,
                       A_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                       gpr_M_reg: int = None, **extras) -> int:
        """Decode-path matmul dispatcher (Trick 7 in shared_design_notes.md).

        IF4 path → quantized_matmat_core (fused, ~2× at M=1) — BUT only when
        ``K <= SCALE_BRAM_ELEMENTS`` (=8192). For K > 8192 the kernel's chunk
        math collapses to ``SCALE_BRAM // K = 0`` → ``N_chunk = 0`` →
        ``chunk_ranges(N, 0)`` infinite loop. Fall back to ``matmat_mul_core``
        unfused for those (Qwen3-1.7B never trips this — max K = 6144 — but
        the guard is here for safety/portability).

        bf16 path (L27 mitigation) → matmat_mul_core(is_B_quantized=False).
        Returns total FLOPS for accounting.
        """
        b = self._matmul_b_kwargs(layer_idx, proj)
        if b.get("is_B_quantized") and K <= SCALE_BRAM_ELEMENTS:
            # Fused: kernel streams IF4 + scales through the dot-product pipeline.
            # M is static (M=1 in decode); quantized_matmat_core has no PBI variant.
            self.quantized_matmat_core(
                M=M, K=K, N=N,
                A_DRAM_ADDR=A_DRAM_ADDR,
                B_DRAM_ADDR=b["B_DRAM_ADDR"],
                OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                SCALE_DRAM_ADDR=b["SCALE_DRAM_ADDR"],
                data_type=b["data_type"],
                **extras)
            return 2 * M * K * N
        # Unfused (bf16 path or K > SCALE_BRAM_ELEMENTS): uses PBI gpr_M_reg.
        return self.matmat_mul_core(
            M=M, K=K, N=N,
            A_DRAM_ADDR=A_DRAM_ADDR, OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
            gpr_M_reg=gpr_M_reg,
            **b, **extras) or 0

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
        # at L27's extreme RMSNorm gammas (default-on; QWEN3_L27_BF16=0 to opt out).
        # NOTE: the bin's baked addresses + matmul instruction shape depend on
        # this toggle, so delete the cached *_instruction.bin if you flip it.
        self._load_last_layer_bf16()

        print(f"    Allocate weights end at DRAM address: 0x{self.get_params_dram_addr():X}, usage: {self.get_params_dram_usage()} bytes")
        print("Tokenizer loaded successfully.")

    def tensor_init(self) -> None:
        """Initialize hardware DRAM tensors for Qwen3-1.7B.

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
        # Single UE_VECTOR_SIZE × UE_VECTOR_SIZE identity matrix reused by every
        # attention/transpose kernel that needs one (Trick 4): unified_attention_core
        # forwards it to the V^T transpose. Passed as IDENTITY_DRAM_ADDR= at every call
        # site; bin bakes one address.
        self.IDENTITY_DRAM_ADDR = self.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe)
        self.dma_to_accelerator_memory(self.IDENTITY_DRAM_ADDR, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))

        # unified_attention_core geometry (replaces the old bucketed flash /
        # decoder-group kernels). Per KV head:
        #   Prefill: Q/K/V are the GQA-flattened FLASH buffers, so
        #            batch = aligned_seq_len = aligned(PREFILL_MAX_SEQ_LEN * group_size).
        #   Decoder: batch = group_size (qpkv Q heads), K/V read straight from the KV cache,
        #            aligned_seq_len = aligned(MAX_CONTEXT_SIZE).
        # The runtime GPR gpr_aligned_seq_len bounds the actual KV length each call; these
        # statics only size the SCRATCH/BIAS address layout, so they must be the maxima.
        self.PREFILL_ALIGNED_SEQ_LEN = ((self.PREFILL_MAX_SEQ_LEN * self.group_size + 63) // 64) * 64
        self.DECODER_ALIGNED_SEQ_LEN = ((self.MAX_CONTEXT_SIZE + 63) // 64) * 64
        # ATTN_P scratch retained for compatibility with the flash-buffer address map.
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
        # unified_attention_core SCRATCH layout: V.T [head_dim, aligned] + score/P
        # [aligned, aligned] + scaled_q [batch, head_dim]. Size for the larger of the
        # prefill and decoder geometries (decoder's aligned=aligned(MAX_CONTEXT) dominates).
        def _unified_scratch_elems(batch: int, aligned: int) -> int:
            return (ahd + aligned) * aligned + batch * ahd
        scratch_elems = max(
            _unified_scratch_elems(self.PREFILL_ALIGNED_SEQ_LEN, self.PREFILL_ALIGNED_SEQ_LEN),
            _unified_scratch_elems(self.group_size, self.DECODER_ALIGNED_SEQ_LEN),
        )
        self.LAYER0_FLASH_SCRATCH_DRAM = self.allocate_tensor_dram(scratch_elems * bpe)
        # Full-matrix bias [batch, aligned]. Prefill's [aligned, aligned] dominates.
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
        # kv_cache_total is in BYTES (bpe=2 baked in). torch.zeros(N, bf16) treats N
        # as element count, so we want N = kv_cache_total // 2 to get the right
        # numel × 2 = kv_cache_total bytes. Without the //2 the DMA wrote 2× the
        # reserved slot, overflowing 224 MB past K_ROPE end into instruction region
        # (harmless because bin load overwrites it, but sloppy).
        zero_kv = torch.zeros(kv_cache_total // 2, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, zero_kv)
        self.dma_to_accelerator_memory(self.LAYER0_K_ROPE_DRAM, zero_kv)

        # On-FPGA repetition penalty: per-vocab additive bias (the LM-head matmul's C term,
        # bias_mode="broadcast_N"). Allocated LAST so adding it never shifts the addresses
        # baked into the bin; run_from_bin's tensor_init allocates it at the same position.
        # All-zero = no penalty. The host refreshes it each penalized step (_write_penalty_bias).
        self.PENALTY_BIAS_DRAM = self.allocate_tensor_dram(1 * self.EMBEDDING_ELEMENTS * self.bytes_per_element)
        self.dma_to_accelerator_memory(self.PENALTY_BIAS_DRAM,
                                       torch.zeros(1, self.EMBEDDING_ELEMENTS, dtype=torch.bfloat16))

        print(f"    Allocate tensor dram end at DRAM address: 0x{self.get_tensor_dram_addr():X}, usage: {self.get_tensor_dram_usage()} bytes")

    # NOTE: alloc_isa_reg / reset_isa_reg_counter are inherited from UnifiedEngine
    # (base cap = 63 GPRs, the real hardware limit — see matmat_mul_dynamic_core).
    # This model reserves regs 1-4 (TMP/seq_len/q_seq_len/aligned_seq_len) by seeding
    # ``self._isa_reg_counter = 5`` in each emit function before dynamic allocation.

    def _emit_prefill_program(self, seq_len: int, layer_size: int) -> int:
        """Emit ONE seq_len-agnostic prefill program (no capture-session boundary).

        Caller wraps this in start_capture()/stop_capture(). The emitted program
        ends with a halt so it can be jumped to via a runtime preamble that
        primes gpr_seq_len / gpr_q_seq_len / gpr_aligned_seq_len GPRs.

        ``seq_len`` here is a **compile-time template** used only for FLOPS
        bookkeeping and as the static ``M=`` arg to PBI ops (overridden at
        runtime by ``gpr_M_reg``). K/V/Q scatter, FLASH_K/V duplication, and the
        FLASH_OUTPUT assembly are emitted as **PBI runtime loops** keyed off
        ``gpr_seq_len``, so the bin is truly seq_len-agnostic up to
        ``MAX_CONTEXT_SIZE`` regardless of the template value.

        All bulk ops (rms_norm, matmat, rope, eltwise) are PBI-dispatched via
        gpr_M_reg=self.gpr_seq_len. Per-token outer loops for matmul/norm/rope
        run gpr_seq_len iterations at runtime regardless of the template value.
        Attention is an inline unified_attention_core call per KV head, reading its
        live KV length from gpr_aligned_seq_len.
        """
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64

        ahd  = self.actual_head_dim   # 128
        nkvh = self.num_kv_heads      # 8
        qpkv = self.group_size        # 2 (Q heads per KV head)
        bpe  = self.bytes_per_element
        hd   = self.head_dim          # 1024
        ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE_LOCAL   # single base for all layers

        # Reset dynamic GPR allocator so PBI-op-internal loop counters don't
        # accumulate across consecutive layers. (Also done in compile_instructions
        # but defensively reset here too.)
        self._isa_reg_counter = 5

        # Attention is now an inline unified_attention_core call per (layer, kv_head)
        # — no shared subroutine, no return-address register, no jump patching.

        total_flops = 0
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        for layer_idx in range(layer_size):
            _original_print(f"    prefill layer {layer_idx + 1}/{layer_size}", end="\r", flush=True)
            layer_off = layer_idx * LAYER_WEIGHT_SIZE
            if layer_idx != 0:
                # Inter-layer copy: previous layer's OUTPUT → next layer's INPUT.
                # Per-token PBI loop (one row of vector_length per iter) to avoid the
                # URAM-A overflow that a single seq_len*vector_length SRAM stage hits
                # once seq_len * vector_length * bpe > 512 KB (URAM_A capacity).
                self._emit_pbi_scatter_per_token(
                    read_base=self.LAYER0_OUTPUT_DRAM,
                    read_stride_bytes=self.vector_length * bpe,
                    write_specs=[(self.LAYER0_INPUT_DRAM, self.vector_length * bpe)],
                    sram_byte_addr=0x10000,
                    element_count=self.vector_length,
                    gpr_seq_len=self.gpr_seq_len,
                    template_seq_len=seq_len,
                )

            # Pre-norm (input_layernorm) — M=seq_len rows at runtime via gpr_seq_len
            total_flops += (self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off,
                              gpr_M_reg=self.gpr_seq_len) or 0)

            # Q, K, V projections — M=seq_len rows at runtime
            total_flops += (self.matmat_mul_core(M=seq_len, K=self.vector_length, N=hd * qpkv,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                gpr_M_reg=self.gpr_seq_len, **self._matmul_b_kwargs(layer_idx, "q")) or 0)
            total_flops += (self.matmat_mul_core(M=seq_len, K=self.vector_length, N=hd,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                gpr_M_reg=self.gpr_seq_len, **self._matmul_b_kwargs(layer_idx, "k")) or 0)
            total_flops += (self.matmat_mul_core(M=seq_len, K=self.vector_length, N=hd,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, OUTPUT_DRAM_ADDR=self.LAYER0_V_PROJ_TEMP,
                gpr_M_reg=self.gpr_seq_len, **self._matmul_b_kwargs(layer_idx, "v")) or 0)

            # QK RMSNorm per head — M = seq_len * nkvh and M = seq_len * nkvh * qpkv.
            # Compute M for each into TMP_REG via reg_mul_imm(gpr_seq_len, multiplier).
            self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gpr_seq_len, nkvh)
            total_flops += (self.rms_norm_core_dram(M=seq_len * nkvh, N=ahd, A_DRAM_ADDR=self.LAYER0_K_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_K_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_NORM_GAMMA + layer_off,
                              gpr_M_reg=self.TMP_REG) or 0)
            self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gpr_seq_len, nkvh * qpkv)
            total_flops += (self.rms_norm_core_dram(M=seq_len * nkvh * qpkv, N=ahd, A_DRAM_ADDR=self.LAYER0_Q_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_Q_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_NORM_GAMMA + layer_off,
                              gpr_M_reg=self.TMP_REG) or 0)

            # RoPE per head per token via rope_hf_core_dram_gqa:
            # K_NORM layout per token: nkvh groups sharing one cos/sin row.
            # Q_NORM layout per token: (nkvh*qpkv) groups sharing one cos/sin row.
            # Runtime row count = gpr_seq_len tokens.
            total_flops += self.rope_hf_core_dram_gqa(
                M=seq_len, group_size=nkvh, N=ahd,
                input_dram_addr=self.LAYER0_K_NORM_DRAM,
                output_dram_addr=self.LAYER0_K_NORM_DRAM,
                cos_dram_addr=ROPE_WEIGHT_ADDR,
                sin_dram_addr=ROPE_WEIGHT_ADDR + ahd * bpe,
                gpr_M_reg=self.gpr_seq_len)
            total_flops += self.rope_hf_core_dram_gqa(
                M=seq_len, group_size=nkvh * qpkv, N=ahd,
                input_dram_addr=self.LAYER0_Q_NORM_DRAM,
                output_dram_addr=self.LAYER0_Q_NORM_DRAM,
                cos_dram_addr=ROPE_WEIGHT_ADDR,
                sin_dram_addr=ROPE_WEIGHT_ADDR + ahd * bpe,
                gpr_M_reg=self.gpr_seq_len)

            # Per-KV-head: scatter K/V to cache + flash buffers, scatter Q, then flash_attention.
            # All per-token scatters are PBI runtime loops (gpr_seq_len trips) — the bin is
            # truly seq_len-agnostic up to MAX_CONTEXT_SIZE; no PREFILL_MAX_SEQ_LEN cap.
            for kv_h in range(nkvh):
                k_cache_base = (self.LAYER0_K_ROPE_DRAM
                                + layer_idx * nkvh * self.MAX_CONTEXT_SIZE * ahd * bpe
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)
                v_cache_base = (self.LAYER0_V_DRAM
                                + layer_idx * nkvh * self.MAX_CONTEXT_SIZE * ahd * bpe
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)

                # K scatter: K_NORM[t][kv_h] → K cache[kv_h][t] + FLASH_K[t*qpkv+g] for g in 0..qpkv-1
                k_write_specs = [(k_cache_base, ahd * bpe)]
                for g in range(qpkv):
                    k_write_specs.append((self.LAYER0_FLASH_K_DRAM + g * ahd * bpe, qpkv * ahd * bpe))
                self._emit_pbi_scatter_per_token(
                    read_base=self.LAYER0_K_NORM_DRAM + kv_h * ahd * bpe,
                    read_stride_bytes=nkvh * ahd * bpe,
                    write_specs=k_write_specs,
                    sram_byte_addr=0x10000,
                    element_count=ahd,
                    gpr_seq_len=self.gpr_seq_len,
                    template_seq_len=seq_len,
                )

                # V scatter: V_PROJ_TEMP[t][kv_h] → V cache[kv_h][t] + FLASH_V[t*qpkv+g]
                v_write_specs = [(v_cache_base, ahd * bpe)]
                for g in range(qpkv):
                    v_write_specs.append((self.LAYER0_FLASH_V_DRAM + g * ahd * bpe, qpkv * ahd * bpe))
                self._emit_pbi_scatter_per_token(
                    read_base=self.LAYER0_V_PROJ_TEMP + kv_h * ahd * bpe,
                    read_stride_bytes=nkvh * ahd * bpe,
                    write_specs=v_write_specs,
                    sram_byte_addr=0x20000,
                    element_count=ahd,
                    gpr_seq_len=self.gpr_seq_len,
                    template_seq_len=seq_len,
                )

                # Q scatter: Q_NORM[t][kv_h*qpkv:kv_h*qpkv+qpkv] → FLASH_Q[t*qpkv:(t+1)*qpkv]
                # Per token, copy qpkv contiguous Q rows (qpkv*ahd elements) in one DMA.
                self._emit_pbi_scatter_per_token(
                    read_base=self.LAYER0_Q_NORM_DRAM + kv_h * qpkv * ahd * bpe,
                    read_stride_bytes=nkvh * qpkv * ahd * bpe,
                    write_specs=[(self.LAYER0_FLASH_Q_DRAM, qpkv * ahd * bpe)],
                    sram_byte_addr=0x30000,
                    element_count=qpkv * ahd,
                    gpr_seq_len=self.gpr_seq_len,
                    template_seq_len=seq_len,
                )

                # Inline attention for this KV head. Q and the GQA-duplicated K/V are the
                # flattened FLASH buffers, so batch = q_seq_len rows and the KV length is the
                # same q_seq_len (padded to aligned_seq_len). Dynamic path: gpr_q_seq_len
                # holds the runtime batch (= actual_seq_len * qpkv) and gpr_aligned_seq_len
                # the 64-aligned KV length. The static batch/aligned_seq_len only size the
                # SCRATCH address layout (template = prefill max).
                total_flops += (self.unified_attention_core(
                    batch=q_seq_len,
                    aligned_seq_len=aligned_seq_len,
                    head_dim=ahd,
                    Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
                    K_DRAM_ADDR=self.LAYER0_FLASH_K_DRAM,
                    V_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                    BIAS_DRAM_ADDR=self.LAYER0_FLASH_BIAS_DRAM,
                    OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUT_HEAD_DRAM,
                    SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                    IDENTITY_DRAM_ADDR=self.IDENTITY_DRAM_ADDR,
                    gpr_batch_reg=self.gpr_q_seq_len,
                    gpr_aligned_seq_len_reg=self.gpr_aligned_seq_len,
                ) or 0)

                # Assemble per-head output into FLASH_OUTPUT_DRAM. Per token, copy
                # qpkv contiguous rows from FLASH_OUT_HEAD[t*qpkv:(t+1)*qpkv] to
                # FLASH_OUTPUT[t][kv_h*qpkv:(kv_h+1)*qpkv]. PBI runtime loop.
                self._emit_pbi_scatter_per_token(
                    read_base=self.LAYER0_FLASH_OUT_HEAD_DRAM,
                    read_stride_bytes=qpkv * ahd * bpe,
                    write_specs=[(self.LAYER0_FLASH_OUTPUT_DRAM + kv_h * qpkv * ahd * bpe,
                                  hd * qpkv * bpe)],
                    sram_byte_addr=0x40000,
                    element_count=qpkv * ahd,
                    gpr_seq_len=self.gpr_seq_len,
                    template_seq_len=seq_len,
                )

            # o_proj
            total_flops += (self.matmat_mul_core(M=seq_len, K=hd * qpkv, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM, OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                gpr_M_reg=self.gpr_seq_len, **self._matmul_b_kwargs(layer_idx, "o")) or 0)

            # Qwen3: no post-attention norm; add residual directly to o_proj output.
            # PBI runtime loop (one row per iter, gpr_seq_len trips) so seq_len > URAM-A capacity is safe.
            self.eltwise_core_dram(
                M=seq_len, N=self.vector_length,
                dram_a=self.LAYER0_INPUT_DRAM,
                dram_b=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                dram_out=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                mode=UE_MODE.ELTWISE_ADD,
                gpr_M_reg=self.gpr_seq_len,
            )

            # Qwen3: post_attention_layernorm IS the pre-FFN norm
            total_flops += (self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off,
                              gpr_M_reg=self.gpr_seq_len) or 0)

            # MLP: gate_proj with SiLU, up_proj, gate x up element-wise, down_proj
            total_flops += (self.matmat_mul_core(M=seq_len, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                silu_enable=True, gpr_M_reg=self.gpr_seq_len,
                **self._matmul_b_kwargs(layer_idx, "gate")) or 0)
            total_flops += (self.matmat_mul_core(M=seq_len, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                gpr_M_reg=self.gpr_seq_len, **self._matmul_b_kwargs(layer_idx, "up")) or 0)

            # gate × up: PBI runtime loop, one row of mlp_elements per iter (gpr_seq_len trips).
            self.eltwise_core_dram(
                M=seq_len, N=self.mlp_elements,
                dram_a=self.LAYER0_MLP_GATE_DRAM,
                dram_b=self.LAYER0_MLP_UP_DRAM,
                dram_out=self.LAYER0_MLP_MULT_DRAM,
                mode=UE_MODE.ELTWISE_MUL,
                gpr_M_reg=self.gpr_seq_len,
            )

            # down_proj: K=6144 ≤ SCALE_BRAM_ELEMENTS=8192, single call
            total_flops += (self.matmat_mul_core(M=seq_len, K=self.mlp_elements, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                gpr_M_reg=self.gpr_seq_len, **self._matmul_b_kwargs(layer_idx, "down")) or 0)

            # Qwen3: no post-FFN norm; add residual directly to down_proj output
            # Post-MLP residual: layer_output = POST_ATTN_RESIDUAL + MLP_DOWN. PBI runtime loop.
            self.eltwise_core_dram(
                M=seq_len, N=self.vector_length,
                dram_a=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                dram_b=self.LAYER0_MLP_DOWN_DRAM,
                dram_out=self.LAYER0_OUTPUT_DRAM,
                mode=UE_MODE.ELTWISE_ADD,
                gpr_M_reg=self.gpr_seq_len,
            )

        # HALT ends the prefill program (attention is now inline, no trailing subroutine).
        self.generate_instruction_halt()
        return total_flops

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
        self.program_execute(preamble_addr, timeout=60.0, flops=gflops)

    def _emit_decoder_program(self, layer_size: int) -> int:
        """Emit ONE decode-position-agnostic decoder program.

        Caller wraps this in start_capture()/stop_capture(). The emitted program
        ends with a halt so it can be jumped to via a runtime preamble that
        primes gpr_aligned_seq_len (64-aligned KV context length). gpr_seq_len is
        primed once by the prefill preamble and auto-incremented at the end
        of every decoder execution via ADD_INC.

        Address math: K/V cache write addresses are computed at runtime as
        ``base + gpr_seq_len * (ahd*bpe)`` via reg_mul_imm + add_imm into TMP_REG,
        then passed as ``general_reg_src=TMP_REG`` to the scatter store.

        Decoder attention is an inline ``unified_attention_core`` call per KV head:
        batch = qpkv (the Q heads sharing this KV head), K/V read straight from the
        per-head KV cache, and gpr_aligned_seq_len bounds the active KV length so no
        stale-cache positions enter the softmax.

        All bulk ops (matmul, rms_norm) use ``gpr_M_reg`` GPRs (gpr_one for M=1).
        """
        ahd  = self.actual_head_dim   # 128
        nkvh = self.num_kv_heads      # 8
        qpkv = self.group_size        # 2
        bpe  = self.bytes_per_element
        hd   = self.head_dim          # 1024
        rope_row_bytes = ahd * 2 * bpe   # 512 B per token

        # Static aligned KV length used only to size the unified_attention_core SCRATCH
        # address layout; the runtime KV length comes from gpr_aligned_seq_len.
        decoder_aligned_seq_len = self.DECODER_ALIGNED_SEQ_LEN
        ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE_LOCAL

        # Two locals survive across the layer loop:
        #   gpr_one          (5) — gpr_M_reg for M=1 matmul/norm ops
        #   gpr_rope_cos_abs (6) — absolute cos-base addr for this step's rope; reused
        #                         across all rope_hf_core calls in all layers
        # unified_attention_core allocates its scratch GPRs above these (from counter 7)
        # and releases them all before returning, so gpr_one / gpr_rope_cos_abs stay live
        # across the inline attention call used later in each layer.
        self._isa_reg_counter = 5
        gpr_one          = self.alloc_isa_reg()
        gpr_rope_cos_abs = self.alloc_isa_reg()
        self.generate_instruction_add_set(gpr_one, 1)
        # gpr_rope_cos_abs = ROPE_LOCAL_BASE + gpr_seq_len * rope_row_bytes (word address)
        self.generate_instruction_reg_mul_imm(
            gpr_rope_cos_abs, self.gpr_seq_len, ue_35bit_addr_shifter(rope_row_bytes))
        self.generate_instruction_add_imm(
            gpr_rope_cos_abs, ue_35bit_addr_shifter(ROPE_WEIGHT_ADDR), gpr_rope_cos_abs)

        total_flops = 0
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        for layer_idx in range(layer_size):
            layer_off = layer_idx * LAYER_WEIGHT_SIZE
            if layer_idx != 0:
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, sram_address=0x10000, element_size=self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_INPUT_DRAM, element_size=self.vector_length)

            # Pre-norm — M=1 via gpr_one
            total_flops += (self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                          OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off,
                          gpr_M_reg=gpr_one) or 0)

            # Q, K, V projections — fused for IF4 layers, unfused for L27 bf16 (see Trick 7).
            total_flops += self._decode_matmul(layer_idx, "q",
                M=1, K=self.vector_length, N=hd * qpkv,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                gpr_M_reg=gpr_one)
            total_flops += self._decode_matmul(layer_idx, "k",
                M=1, K=self.vector_length, N=hd,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                gpr_M_reg=gpr_one)
            total_flops += self._decode_matmul(layer_idx, "v",
                M=1, K=self.vector_length, N=hd,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, OUTPUT_DRAM_ADDR=self.LAYER0_V_PROJ_TEMP,
                gpr_M_reg=gpr_one)

            # QK RMSNorm: M is a compile-time constant (nkvh for K, nkvh*qpkv for Q).
            # Pass static M; no gpr_M_reg needed.
            total_flops += (self.rms_norm_core_dram(M=nkvh, N=ahd, A_DRAM_ADDR=self.LAYER0_K_DRAM,
                          OUTPUT_DRAM_ADDR=self.LAYER0_K_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_NORM_GAMMA + layer_off) or 0)
            total_flops += (self.rms_norm_core_dram(M=nkvh * qpkv, N=ahd, A_DRAM_ADDR=self.LAYER0_Q_DRAM,
                          OUTPUT_DRAM_ADDR=self.LAYER0_Q_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_NORM_GAMMA + layer_off) or 0)

            # RoPE per head — use base-class rope_hf_core with gr_weight_dram (single GPR
            # holding absolute cos-base address). gpr_rope_cos_abs was primed once at the
            # start of the bin (= ROPE_LOCAL + gpr_seq_len * rope_row_bytes).
            for kv_h in range(nkvh):
                total_flops += user_dma_core.UnifiedEngine.rope_hf_core_decode(self,
                    N=ahd,
                    input_dram_addr=self.LAYER0_K_NORM_DRAM + kv_h * ahd * bpe,
                    output_dram_addr=self.LAYER0_K_NORM_DRAM + kv_h * ahd * bpe,
                    cos_dram_addr=ROPE_WEIGHT_ADDR,
                    gr_weight_dram=gpr_rope_cos_abs)
            for q_h in range(nkvh * qpkv):
                total_flops += user_dma_core.UnifiedEngine.rope_hf_core_decode(self,
                    N=ahd,
                    input_dram_addr=self.LAYER0_Q_NORM_DRAM + q_h * ahd * bpe,
                    output_dram_addr=self.LAYER0_Q_NORM_DRAM + q_h * ahd * bpe,
                    cos_dram_addr=ROPE_WEIGHT_ADDR,
                    gr_weight_dram=gpr_rope_cos_abs)

            # Per-KV-head: store new K/V to cache at decode position, then per-Q-head attention.
            # K cache write addr = k_cache_base + gpr_seq_len * (ahd*bpe); same for V.
            for kv_h in range(nkvh):
                k_cache_base = (self.LAYER0_K_ROPE_DRAM
                                + layer_idx * nkvh * self.MAX_CONTEXT_SIZE * ahd * bpe
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)
                v_cache_base = (self.LAYER0_V_DRAM
                                + layer_idx * nkvh * self.MAX_CONTEXT_SIZE * ahd * bpe
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)

                # K cache write
                self.accelerator_memory_to_sram(self.LAYER0_K_NORM_DRAM + kv_h * ahd * bpe, 0x10000, ahd)
                self.generate_instruction_reg_mul_imm(
                    self.TMP_REG, self.gpr_seq_len, ue_35bit_addr_shifter(ahd * bpe))
                self.generate_instruction_add_imm(
                    self.TMP_REG, ue_35bit_addr_shifter(k_cache_base), self.TMP_REG)
                self.sram_to_accelerator_memory(
                    sram_address=0x10000, accelerator_dram_address=0,
                    element_size=ahd, general_reg_src=self.TMP_REG)

                # V cache write
                self.accelerator_memory_to_sram(self.LAYER0_V_PROJ_TEMP + kv_h * ahd * bpe, 0x20000, ahd)
                self.generate_instruction_reg_mul_imm(
                    self.TMP_REG, self.gpr_seq_len, ue_35bit_addr_shifter(ahd * bpe))
                self.generate_instruction_add_imm(
                    self.TMP_REG, ue_35bit_addr_shifter(v_cache_base), self.TMP_REG)
                self.sram_to_accelerator_memory(
                    sram_address=0x20000, accelerator_dram_address=0,
                    element_size=ahd, general_reg_src=self.TMP_REG)

                # Gather this KV head's Q group (qpkv contiguous head_dim rows) into FLASH_Q.
                for q in range(qpkv):
                    q_src = self.LAYER0_Q_NORM_DRAM + (kv_h * qpkv + q) * ahd * bpe
                    flash_q_addr = self.LAYER0_FLASH_Q_DRAM + q * ahd * bpe
                    self.accelerator_memory_to_sram(q_src, 0x30000, ahd)
                    self.sram_to_accelerator_memory(0x30000, flash_q_addr, ahd)

                # Inline attention for this KV head. Q is the qpkv-row group just gathered;
                # K/V are read straight from this head's KV cache (no separate gather). batch
                # is the static qpkv Q heads; gpr_aligned_seq_len bounds the live KV length so
                # stale cache rows past the current position never enter the softmax.
                total_flops += (self.unified_attention_core(
                    batch=qpkv,
                    aligned_seq_len=decoder_aligned_seq_len,
                    head_dim=ahd,
                    Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
                    K_DRAM_ADDR=k_cache_base,
                    V_DRAM_ADDR=v_cache_base,
                    BIAS_DRAM_ADDR=self.LAYER0_FLASH_BIAS_DRAM,
                    OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUT_HEAD_DRAM,
                    SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                    IDENTITY_DRAM_ADDR=self.IDENTITY_DRAM_ADDR,
                    gpr_aligned_seq_len_reg=self.gpr_aligned_seq_len,
                ) or 0)
                self.accelerator_memory_to_sram(self.LAYER0_FLASH_OUT_HEAD_DRAM, 0x40000, qpkv * ahd)
                self.sram_to_accelerator_memory(
                    0x40000, self.LAYER0_FLASH_OUTPUT_DRAM + kv_h * qpkv * ahd * bpe, qpkv * ahd)

            # o_proj
            total_flops += self._decode_matmul(layer_idx, "o",
                M=1, K=hd * qpkv, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM, OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                gpr_M_reg=gpr_one)

            # Qwen3: no post-attention norm; residual direct on o_proj output
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_INPUT_DRAM, sram_address=0x10000, element_size=self.vector_length)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM, sram_address=0x90000, element_size=self.vector_length)
            self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, element_size=self.vector_length)

            # Qwen3: post_attention_layernorm IS the pre-FFN norm
            total_flops += (self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                          OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off,
                          gpr_M_reg=gpr_one) or 0)

            # MLP: SwiGLU
            total_flops += self._decode_matmul(layer_idx, "gate",
                M=1, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                silu_enable=True, gpr_M_reg=gpr_one)
            total_flops += self._decode_matmul(layer_idx, "up",
                M=1, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                gpr_M_reg=gpr_one)

            # gate x up (M=1: mlp_elements fits in SRAM in one shot)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_GATE_DRAM, sram_address=0x10000, element_size=self.mlp_elements)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_UP_DRAM, sram_address=0x90000, element_size=self.mlp_elements)
            self.eltwise_mul_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.mlp_elements)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_MLP_MULT_DRAM, element_size=self.mlp_elements)

            # down_proj: K=6144 ≤ SCALE_BRAM_ELEMENTS=8192, single call
            total_flops += self._decode_matmul(layer_idx, "down",
                M=1, K=self.mlp_elements, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                gpr_M_reg=gpr_one)

            # Qwen3: no post-FFN norm; residual direct on down_proj output
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, sram_address=0x10000, element_size=self.vector_length)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_DOWN_DRAM, sram_address=0x90000, element_size=self.vector_length)
            self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, element_size=self.vector_length)

        if layer_size == self.LAYER_SIZE:
            total_flops += (self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_OUTPUT_DRAM,
                OUTPUT_DRAM_ADDR=self.OUTPUT_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_OUTPUT_NORM_GAMMA,
                gpr_M_reg=gpr_one) or 0)
            # LM head with the on-FPGA repetition-penalty bias folded in: the matmul adds
            # PENALTY_BIAS_DRAM (its per-vocab C term, bias_mode="broadcast_N") before the
            # on-chip argmax, so the HW argmax register holds the PENALIZED token id directly.
            # write_back_disable=True: greedy/argmax needs no 304 KB LOGITS_DRAM writeback
            # (no host logit readback at all). The bias buffer is zero pre-gate / in
            # --pure-greedy, so the same bin serves plain greedy and penalty.
            total_flops += (self.matmat_mul_core(M=1, K=self.vector_length, N=self.EMBEDDING_ELEMENTS,
                A_DRAM_ADDR=self.OUTPUT_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD_QUANT, OUTPUT_DRAM_ADDR=self.LOGITS_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD_SCALE, gpr_M_reg=gpr_one,
                C_DRAM_ADDR=self.PENALTY_BIAS_DRAM, bias_mode="broadcast_N",
                write_back_disable=True) or 0)

        # Advance gpr_seq_len for next decode step (matches gemma3 convention).
        # Host preamble primes only gpr_aligned_seq_len between steps; gpr_seq_len carries
        # across via this in-bin increment.
        self.generate_instruction_add_inc(self.gpr_seq_len)

        # End-of-program halt so the runtime preamble's JUMP_ABS returns control after
        # execute (attention is now inline, no trailing subroutine).
        self.generate_instruction_halt()
        return total_flops

    def compile_instructions(self, layer_size: int | None = None) -> dict:
        """Compile a UNIFIED single-bin instruction image: ONE prefill program
        + ONE decoder program in one capture session. Writes
        ``programs.bin`` + matching ``programs.json`` meta to disk.

        Both programs are **seq_len-agnostic**: matmul/norm/rope row counts come
        from gpr_seq_len / gpr_q_seq_len GPRs, and unified_attention_core reads its
        live KV length from gpr_aligned_seq_len — all primed by the runtime preamble in
        run_prefill / run_decoder per call. The cached bin therefore works across any
        prompt length ≤ PREFILL_MAX_SEQ_LEN and any decode position < MAX_CONTEXT_SIZE.

        Cache-hit: if bin + meta already exist, returns the meta without
        recompile.

        Returns the meta dict:
          {
            "prefill_template_seq_len":  int,   # static template used at compile time
            "prefill_program_start_addr": "0x...",
            "prefill_program_size":      int,   # bytes
            "prefill_template_flops":    int,
            "decoder_program_start_addr": "0x...",
            "decoder_program_size":      int,
            "decoder_total_flops":       int,
          }
        """
        if layer_size is None:
            layer_size = self.LAYER_SIZE

        paths_cfg = self._cfg.get("paths", {})
        bin_rel  = paths_cfg.get("instruction_bin",  "qwen3_1.7b_bin/programs.bin")
        meta_rel = paths_cfg.get("instruction_meta", "qwen3_1.7b_bin/programs.json")
        bin_path  = os.path.join(self.script_dir, bin_rel)
        meta_path = os.path.join(self.script_dir, meta_rel)

        # The decode LM head is uniform now: write_back_disable=True + the penalty C bias
        # (PENALTY_BIAS_DRAM). Penalty vs --pure-greedy differ only in the runtime bias
        # buffer contents, NOT in the instructions, so one bin serves both. The cache is
        # keyed on this LM-head signature so a stale bin from the older sampling/greedy
        # builds (which lack the bias / had writeback on) is detected and recompiled.
        lm_head_sig = "penalty_bias_argmax"

        # Cache short-circuit: reuse the on-disk bin only if it exists AND was compiled
        # with the same LM-head signature. Identity-matrix DMAs happen in weight_init at
        # fixed addresses every invocation, so loading the bin off disk is otherwise
        # sufficient. A bin from an older build (missing/different key) is recompiled.
        bin_cached = os.path.exists(bin_path) and os.path.exists(meta_path)
        if bin_cached:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if meta.get("lm_head_sig") == lm_head_sig:
                print(f"compile_instructions: cache hit, skipping compile ({bin_path}).")
                return meta
            print(f"compile_instructions: cached bin LM-head sig "
                  f"({meta.get('lm_head_sig')}) != this run ({lm_head_sig}); recompiling.")

        os.makedirs(os.path.dirname(bin_path), exist_ok=True)
        # Template seq_len: drives static M= args (overridden at runtime by gpr_M_reg),
        # FLOPs estimate, and unrolled scatter-loop iteration count. Set to
        # PREFILL_MAX_SEQ_LEN so the bin can handle any actual_seq_len up to that.
        prefill_template_seq_len = self.PREFILL_MAX_SEQ_LEN

        global _SILENT_MODE
        _SILENT_MODE = True
        self.clear_inst_id()
        self.start_capture()

        # Track start address (= current program-DRAM allocator tail before any DMA).
        instruction_base_addr = self.get_program_dram_addr()

        _original_print(f"  Compiling prefill (template seq_len={prefill_template_seq_len}, {layer_size} layers)...")
        prefill_count_at_start = self.capture_count
        prefill_flops = self._emit_prefill_program(seq_len=prefill_template_seq_len, layer_size=layer_size)
        prefill_program_size = (self.capture_count - prefill_count_at_start) * 32

        _original_print(f"  Compiling decoder ({layer_size} layers)...")
        decoder_count_at_start = self.capture_count
        decoder_flops = self._emit_decoder_program(layer_size=layer_size)
        decoder_program_size = (self.capture_count - decoder_count_at_start) * 32

        self.stop_capture()
        _SILENT_MODE = False

        # Sanity check vs MAX_DECODER_INSTRUCTIONS cap.
        from user_dma_core import MAX_DECODER_INSTRUCTIONS
        if self.capture_count >= MAX_DECODER_INSTRUCTIONS:
            raise RuntimeError(
                f"Capture hit MAX_DECODER_INSTRUCTIONS cap "
                f"({MAX_DECODER_INSTRUCTIONS} instructions = "
                f"{MAX_DECODER_INSTRUCTIONS * 32 / 2**20:.0f} MiB). "
                f"Captured {self.capture_count} instructions = "
                f"{self.capture_count * 32 / 2**20:.0f} MiB. "
                f"Shrink PREFILL_MAX_SEQ_LEN or move more ops to PBI loop_start."
            )

        all_bytes = bytearray()
        for inst in self.capture_buffer:
            all_bytes.extend(inst.get_bytes())

        prefill_program_start_addr = instruction_base_addr
        decoder_program_start_addr = instruction_base_addr + prefill_program_size

        meta = {
            "instruction_base_addr":        f"0x{instruction_base_addr:X}",
            "instruction_total_size":       len(all_bytes),
            "prefill_template_seq_len":     prefill_template_seq_len,
            "prefill_program_start_addr":   f"0x{prefill_program_start_addr:X}",
            "prefill_program_size":         prefill_program_size,
            "prefill_template_flops":       prefill_flops,
            "decoder_program_start_addr":   f"0x{decoder_program_start_addr:X}",
            "decoder_program_size":         decoder_program_size,
            "decoder_total_flops":          decoder_flops,
            "lm_head_sig":                  lm_head_sig,
        }
        with open(bin_path, "wb") as f:
            f.write(all_bytes)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Instruction bin written: {bin_path} ({len(all_bytes)} bytes; "
              f"prefill {prefill_program_size} B + decoder {decoder_program_size} B)")
        self.clear_capture_buffer()
        return meta

    def run_decoder(self, decoder_program_addr: int, preamble_addr: int,
                    token_id: int, gflops_per_token: int | None = None) -> dict:
        """Run decode loop via dynamic-PBI runtime preamble.

        Each step emits a small preamble at ``preamble_addr`` that primes
        gpr_aligned_seq_len (64-aligned KV context length), then jump_abs's into
        the cached decoder program. gpr_seq_len (current decode position) carries
        across steps via the in-bin ADD_INC. Same preamble slot is overwritten per token.

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
        # unified_attention_core reads its live KV length from gpr_aligned_seq_len. The
        # decoder program was compiled with a static aligned_seq_len = aligned(MAX_CONTEXT),
        # which sizes the SCRATCH/bias layout; the bias row must span that full width.
        decoder_aligned_seq_len = ((self.MAX_CONTEXT_SIZE + 63) // 64) * 64
        max_seq_len = self.MAX_CONTEXT_SIZE

        # On-FPGA repetition-penalty state. _generated_tokens is seeded with the prompt by
        # main() (or run_from_bin); decoded ids are appended below. The penalty is folded
        # into the LM-head matmul bias, so token selection is always the HW argmax — no host
        # logit readback. Position-gated: pure greedy for the first `greedy_until` decoded
        # tokens (math/reasoning lands early), then the per-vocab bias turns on to break
        # long-context loops. --pure-greedy (fpga_penalty=False) leaves the bias all-zero.
        if not hasattr(self, "_generated_tokens"):
            self._generated_tokens = []
        _fpga_penalty = bool(getattr(self, "fpga_penalty", True))
        _greedy_until = int(getattr(self, "greedy_until", 512))
        _prompt_len = len(self._generated_tokens)
        # Zero the bias → pure greedy until the gate (and entirely in --pure-greedy).
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
            # unified_attention_core's dynamic path uses bias_mode="full_matrix" (one bias
            # row per Q head), so materialize the same causal-mask row for each of the qpkv
            # query heads. Row width = aligned_ctx (== gpr_aligned_seq_len); positions
            # [new_ctx_len..aligned_ctx) are -inf so stale KV-cache slots never enter softmax.
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

            # Refresh the per-vocab penalty bias (this step's LM-head matmul C term) once
            # past the gate, BEFORE execute, so the HW argmax of (logits + bias) returns the
            # penalized token directly. One full-buffer DMA per step.
            if _fpga_penalty and (len(self._generated_tokens) - _prompt_len) > _greedy_until:
                self._write_penalty_bias(self._generated_tokens)

            self.program_execute(preamble_addr, timeout=10.0, flops=gflops_per_token)
            # Token selection: read the HW argmax register. The LM-head matmul already added
            # the penalty bias on chip, so the register holds the penalized token id (or pure
            # greedy when the bias is zero). No host logit readback.
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
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3-1.7B prefill + decode on accelerator.")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt: tokenizer encodes this to prefill_seq (overrides default)")
    parser.add_argument("--local-weights", action="store_true", help="Use qwen3_1.7b_bin/full_model_weights.bin instead of generated params.bin")
    parser.add_argument('--dev', type=str, default='xdma0',
                        help='DMA device name (e.g., xdma0, xdma1). Default: xdma0')
    parser.add_argument('--cycle', type=float, default=5.62,
                        help='Clock cycle time in nanoseconds (default: 5.62ns ≈ peak 22.8 GFLOPS)')
    # Decode is deterministic, on-FPGA only: token selection is always the HW argmax of
    # (logits + penalty bias). No host sampling (temperature/top-k/top-p/multinomial) — the
    # repetition penalty is folded into the LM-head matmul bias (notes_repetition_penalty_fpga_bias.md).
    parser.add_argument('--pure-greedy', action='store_true',
                        help='Disable the on-FPGA repetition penalty — plain greedy decode '
                             '(the penalty bias stays all-zero). Penalty is ENABLED by default.')
    pen_group = parser.add_argument_group('on-FPGA repetition penalty (active unless --pure-greedy)')
    pen_group.add_argument('--greedy-until', type=int, default=512,
                        help='Pure greedy for the first N decoded tokens (math/reasoning lands '
                             'early), then the penalty turns on to break long-context loops. '
                             '0 = penalty from the start. Default 512.')
    pen_group.add_argument('--pen-alpha', type=float, default=1.0,
                        help='bias[t] = -alpha*count[t] (logit units). Default 1.0.')
    pen_group.add_argument('--pen-cap', type=float, default=20.0,
                        help='max |bias| per token (floor on -alpha*count). Default 20.')
    pen_group.add_argument('--rep-window', type=int, default=256,
                        help='count tokens over the last N (never penalizes punctuation/whitespace/'
                             'special tokens). Default 256.')
    args = parser.parse_args()

    if args.device == "efinix":
        print("ERROR: qwen3_1.7b requires ~1134 MB of DRAM for weights, exceeding the Efinix board's 1 GB limit.")
        raise SystemExit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.local_weights:
        weights_bin_rel = "qwen3_1.7b_bin/full_model_weights.bin"
    else:
        weights_bin_rel = "qwen3_1.7b_bin/params.bin"
        weights_bin_full = os.path.join(script_dir, weights_bin_rel)
        if not os.path.exists(weights_bin_full):
            weight_bin_generate(script_dir=script_dir, output_path=weights_bin_full)

    set_dma_device(args.dev)
    global DMA_DEVICE_H2C, DMA_DEVICE_C2H, DMA_DEVICE_USER
    DMA_DEVICE_H2C = user_dma_core.DMA_DEVICE_H2C
    DMA_DEVICE_C2H = user_dma_core.DMA_DEVICE_C2H
    DMA_DEVICE_USER = user_dma_core.DMA_DEVICE_USER
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle
    user_dma_core.UE_PEAK_GFLOPS = 0.128 / args.cycle
    print(f"Using DMA device: {args.dev}")
    print(f"  H2C: {DMA_DEVICE_H2C}")
    print(f"  C2H: {DMA_DEVICE_C2H}")
    print(f"  USER: {DMA_DEVICE_USER}")
    print(f"Setting CLOCK_CYCLE_TIME_NS = {user_dma_core.CLOCK_CYCLE_TIME_NS}, UE_PEAK_GFLOPS = {user_dma_core.UE_PEAK_GFLOPS:.4f}")

    ue = Qwen3_1_7b_UnifiedEngine(script_dir=script_dir, weights_bin=weights_bin_rel)
    cfg = _load_config(script_dir)
    # Always tokenize via apply_chat_template — no hardcoded ids in the config.
    user_prompt = args.prompt if args.prompt is not None else cfg.get("default_prompt", "What is 3 + 5?")
    system_prompt = cfg.get("default_system_prompt", "You are a helpful assistant.")
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    prompt_with_template = ue.tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    prefill_seq = tuple(ue.tokenizer.encode(prompt_with_template, add_special_tokens=False))
    print(f"User prompt ({len(prefill_seq)} tokens): {user_prompt!r}")
    print(f"Sequence ids: {prefill_seq}")

    # On-FPGA repetition-penalty config (deterministic; the sole decode path).
    ue.fpga_penalty = not bool(args.pure_greedy)
    ue.greedy_until = int(args.greedy_until)
    ue.pen_alpha = float(args.pen_alpha)
    ue.pen_cap = float(args.pen_cap)
    ue.rep_window = int(args.rep_window)
    ue._generated_tokens = list(prefill_seq)   # seed the penalty window with the prompt
    if ue.fpga_penalty:
        print(f"On-FPGA penalty: alpha={ue.pen_alpha} cap={ue.pen_cap} "
              f"rep_window={ue.rep_window} greedy_until={ue.greedy_until}")
    else:
        print("Pure greedy (on-FPGA penalty disabled).")

    print(f"\n--- Compiling unified instruction bin (1 prefill + 1 decoder, dynamic PBI) ---")
    timer = time.perf_counter()
    inst_meta = ue.compile_instructions()
    print(f"  compile_instructions done in {time.perf_counter() - timer:.2f}s")

    paths_cfg = cfg.get("paths", {})
    inst_bin_path = os.path.join(script_dir, paths_cfg.get("instruction_bin",
                                  "qwen3_1.7b_bin/programs.bin"))
    base_addr, _ = ue.load_program_instructions_from_file(inst_bin_path)
    # Reserve a slot AFTER the loaded bin for the runtime preamble. Prefill preamble
    # is 4 instructions (3 ADD_SET + JUMP_ABS) = 128 B; decode preamble is 2 instructions
    # (ADD_SET gpr_aligned_seq_len + JUMP_ABS) and overwrites the same slot.
    preamble_addr = ue.get_program_dram_addr()
    ue.allocate_program_dram(128)

    prefill_program_addr = _parse_offset(inst_meta["prefill_program_start_addr"])
    decoder_program_addr = _parse_offset(inst_meta["decoder_program_start_addr"])
    decoder_total_flops  = inst_meta["decoder_total_flops"]

    actual_seq_len = len(prefill_seq) - 1
    # Rescale prefill FLOPs from the compile-time template to the actual seq_len.
    # The meta records FLOPs for ``prefill_template_seq_len`` rows; runtime ops execute
    # ``actual_seq_len`` rows (via gpr_seq_len-driven PBI loops), so linear scaling gives
    # the real work count for the GFLOPS report.
    template_seq_len = int(inst_meta["prefill_template_seq_len"])
    gflops_prefill = inst_meta["prefill_template_flops"] * actual_seq_len // max(template_seq_len, 1)
    print(f"\n--- Starting prefill (actual {actual_seq_len} tokens, dynamic seq_len) ---")
    print(f"Prompt tokens ({len(prefill_seq)}): {prefill_seq}")
    print(f"Prompt text: {ue.tokenizer.decode(prefill_seq, skip_special_tokens=False)!r}")
    timer = time.perf_counter()
    ue.run_prefill(prefill_program_addr, preamble_addr, prefill_seq=prefill_seq,
                   gflops=gflops_prefill)
    latency_prefill = time.perf_counter() - timer
    print(f"Prefill execute done in {latency_prefill:.2f} seconds, start decoding...\n")

    print(f"\n--- Starting decoder ---")
    timer = time.perf_counter()
    token_cnt_decoded = ue.run_decoder(decoder_program_addr, preamble_addr,
                                       token_id=prefill_seq[-1],
                                       gflops_per_token=decoder_total_flops)
    latency_decoder = time.perf_counter() - timer
    decoded_tokens = max(token_cnt_decoded - len(prefill_seq) + 1, 1)
    print(f"\nDecoder done in {latency_prefill + latency_decoder:.2f}s, "
          f"speed: {decoded_tokens / latency_decoder:.2f} tokens/s, total {token_cnt_decoded} tokens.")
    print("Qwen3-1.7B test ends.")

if __name__ == "__main__":
    main()
