#!/usr/bin/env python3
"""
Qwen3-4B inference on accelerator: prefill + decode.

  - Config from qwen3_4b_bin/qwen3_4b_config.json; weights from a single bin (see below).
  - Prefill: compiled each run. Decoder: if qwen3_4b_bin/decoder_program.bin and
    qwen3_4b_bin/decoder_program.json exist, skip decoder compile and load
    program sizes from meta; otherwise compile and write the bin + meta.
  - Run prefill then decode loop. For numeric verification use qwen3_4b_numeric.py.

Architecture notes vs Gemma3:
  - 36 layers, 32 Q heads / 8 KV heads (group_size=4), actual head_dim=128.
  - hidden_size=2560, mlp_intermediate=9728, vocab=151936.
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
  - Default: qwen3_4b_bin/weights_qwen3_4b_hf.bin (generated from HF model if missing).
  - --local-weights: use qwen3_4b_bin/full_model_weights.bin instead.

Usage:
  python qwen3_4b_test.py
  python qwen3_4b_test.py --prompt "your prompt"
  python qwen3_4b_test.py --dev xdma0 [--cycle 5.88]
  python qwen3_4b_test.py --local-weights
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
from user_dma_core import DMA_DEVICE_H2C, TYPE, UE_FMAX_CONTEXT_SIZE, UE_MODE, UE_VECTOR_SIZE, UE_ARGMAX_INDEX, SCALE_BRAM_ELEMENTS, INSTRUCTION_SIZE_BYTES, set_dma_device, ue_35bit_addr_shifter
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
    """Generate weights_qwen3_4b_hf.bin from Hugging Face model per qwen3_4b_config.json layout.
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
        #   reg 1 = TMP_REG          — scratch for reg_mul_imm + add_imm address math
        #   reg 2 = GPR_SEQ_LEN_REG   — runtime row count for matmul/norm/rope (M=seq_len ops)
        #   reg 3 = GPR_Q_SEQ_LEN_REG — for ops with M = seq_len * group_size (Q-side norms/rope)
        #   reg 4 = GPR_BUCKET_IDX_REG— 1-based bucket selector for flash / group-attention kernels
        # Dynamic GPR allocation (via alloc_isa_reg) starts at 5; PBI op-internal loop counters
        # consume from there and release back at loop_end.
        fixed = self._cfg.get("fixed_isa_regs", {})
        self.TMP_REG       = fixed["TMP_REG"]
        self.gpr_seq_len    = fixed["GPR_SEQ_LEN_REG"]
        self.gpr_q_seq_len  = fixed["GPR_Q_SEQ_LEN_REG"]
        self.gpr_bucket_idx = fixed["GPR_BUCKET_IDX_REG"]
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

    def load_instructions(self, bin_path: str) -> tuple[int, int]:
        """Load decoder instruction bin from file into program DRAM. Returns (start_addr, total_size)."""
        with open(bin_path, "rb") as f:
            data = f.read()
        total_size = len(data)
        start_addr = self.allocate_program_dram(total_size)
        self.dma_write(DMA_DEVICE_H2C, start_addr, data, total_size)
        print(f"    Loaded {total_size} bytes from instruction.bin to DRAM at 0x{start_addr:x}")
        return start_addr, total_size

    # Overwrite UnifiedEngine allocate_params_dram
    def allocate_params_dram(self, size_bytes: int) -> int:
        """
        Allocate memory from the params DRAM region incrementally.

        Args:
            size_bytes: Number of bytes to allocate

        Returns:
            The DRAM address of the allocated block (address before increment).
        """
        params_dram_addr = self._next_params_dram_addr
        self._next_params_dram_addr += size_bytes
        return params_dram_addr

    def clear_inst_id(self) -> None:
        """Reset instruction ID counter for the next capture."""
        self._inst_id = 0


    def get_arg_max_index(self) -> int:
        """Get the arg max index from the Unified Engine"""
        return self.read_reg32(UE_ARGMAX_INDEX)

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

    def _matmul_b_kwargs(self, layer_idx: int, proj: str) -> dict:
        """Return matmat_mul_core B/scale/quant kwargs; routes the last layer
        through bf16 when ``_load_last_layer_bf16`` populated the addrs."""
        if (layer_idx == self.LAYER_SIZE - 1
                and self._last_layer_bf16_addrs
                and proj in self._last_layer_bf16_addrs):
            return {
                "B_DRAM_ADDR":   self._last_layer_bf16_addrs[proj],
                "is_B_quantized": False,
            }
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
        ``K <= SCALE_BRAM_ELEMENTS`` (=8192). For K > 8192 (e.g. Qwen3-4B
        ``down_proj`` with K=9728) the kernel's chunk math collapses to
        ``SCALE_BRAM // K = 0`` → ``N_chunk = 0`` → ``chunk_ranges(N, 0)``
        infinite loop. Fall back to ``matmat_mul_core`` (unfused) in that case;
        we lose the ~2× M=1 speedup on that one op but the rest of decode keeps
        the fused-kernel win.

        bf16 path (L35 mitigation) → matmat_mul_core(is_B_quantized=False).
        Returns total FLOPS for accounting.
        """
        b = self._matmul_b_kwargs(layer_idx, proj)
        if b.get("is_B_quantized") and K <= SCALE_BRAM_ELEMENTS:
            self.quantized_matmat_core(
                M=M, K=K, N=N,
                A_DRAM_ADDR=A_DRAM_ADDR,
                B_DRAM_ADDR=b["B_DRAM_ADDR"],
                OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                SCALE_DRAM_ADDR=b["SCALE_DRAM_ADDR"],
                data_type=b["data_type"],
                **extras)
            return 2 * M * K * N
        return self.matmat_mul_core(
            M=M, K=K, N=N,
            A_DRAM_ADDR=A_DRAM_ADDR, OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
            gpr_M_reg=gpr_M_reg,
            **b, **extras) or 0

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
        # Single UE_VECTOR_SIZE × UE_VECTOR_SIZE identity matrix reused by every
        # attention/transpose kernel that needs one (Trick 4):
        #   - flash_attention_core (forwarded to bf16_transpose_core_pbi for V^T)
        #   - decoder_group_attention_core (qwen3 decode)
        # Passed as IDENTITY_DRAM_ADDR= at every call site; bin bakes one address.
        self.IDENTITY_DRAM_ADDR = self.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe)
        self.dma_to_accelerator_memory(self.IDENTITY_DRAM_ADDR, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))
        # Flash-attention bucket-dispatcher scratch buffer. Sized for the
        # largest aligned_seq_len_q this engine instance might see:
        # PREFILL_MAX_SEQ_LEN * group_size, rounded up to UE_VECTOR_SIZE.
        aligned_q_max = ((self.PREFILL_MAX_SEQ_LEN * self.group_size + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
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
        self.LAYER0_FLASH_SCRATCH_DRAM = self.allocate_tensor_dram(max(ahd, UE_FMAX_CONTEXT_SIZE) * aligned_seq_len * 2 + ahd * aligned_seq_len * 2)
        self.LAYER0_FLASH_BIAS_DRAM = self.allocate_tensor_dram(aligned_seq_len * aligned_seq_len * bpe)

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

        # On-FPGA repetition penalty: per-vocab additive bias (the LM-head matmul's C term,
        # bias_mode="broadcast_N"). Allocated LAST so adding it never shifts the addresses
        # baked into the bin; run_from_bin's tensor_init allocates it at the same position.
        # All-zero = no penalty. The host refreshes it each penalized step (_write_penalty_bias).
        self.PENALTY_BIAS_DRAM = self.allocate_tensor_dram(1 * self.EMBEDDING_ELEMENTS * self.bytes_per_element)
        self.dma_to_accelerator_memory(self.PENALTY_BIAS_DRAM,
                                       torch.zeros(1, self.EMBEDDING_ELEMENTS, dtype=torch.bfloat16))

        print(f"    Allocate tensor dram end at DRAM address: 0x{self.get_tensor_dram_addr():X}, usage: {self.get_tensor_dram_usage()} bytes")

    def program_execute(self, program_start_addr: int = user_dma_core.DRAM_INSTRUCTION_ADDR, timeout: float = 120.0, gflops: float = None) -> None:
        """Execute compiled program from DRAM instruction memory.
        """
        print(f"Execute program start at 0x{program_start_addr:X}")
        self.start_execute_from_dram(program_start_addr)
        self.wait_queue(timeout)
        latency = self.report_latency_in_us()
        print(f"    Total program execution latency = {latency} us")
        if gflops is not None:
            gflops_program, _ = self.report_flop_rate_gflops(gflops)
            # High-precision so callers can sanity-check vs peak ( = 128 / clock_ns ):
            # if num_flops or clock period is mis-set the reported GFLOPS will drift.
            print(f"Report FLOPS for program execution: {gflops_program:.4f} GFLOPS "
                  f"(num_flops={int(gflops)}, latency_us={latency:.1f}, clock_ns={self._clock_period_ns})")

    def reset_isa_reg_counter(self) -> None:
        """Reset PBI ISA-reg allocator back to 5 (regs 1-4 are fixed/reserved)."""
        self._isa_reg_counter = 5

    def alloc_isa_reg(self, reset: bool = False) -> int:
        """Allocate the next free ISA register. Used by PBI ops for loop counters."""
        if reset:
            self._isa_reg_counter = 5
        if self._isa_reg_counter > 15:
            raise ValueError("Exceeded available ISA registers (max 15)")
        reg_idx = self._isa_reg_counter
        self._isa_reg_counter += 1
        return reg_idx

    def _emit_pbi_scatter_per_token(self, *, read_base, read_stride_bytes,
                                    write_specs, sram_byte_addr, element_count,
                                    gpr_seq_len, template_seq_len):
        """Emit one PBI runtime loop that staged-copies one ``element_count``-row
        per outer iteration from ``read_base`` to each (base, stride) in
        ``write_specs``. The outer trip count is taken from GPR ``gpr_seq_len`` so
        the body executes exactly ``actual_seq_len`` times at runtime — making
        the captured bin truly seq_len-agnostic up to ``MAX_CONTEXT_SIZE``.

        Read side uses register-computed addresses (``reg_mul_imm`` + ``add_imm``
        + ``general_reg_src=TMP_REG``) — the same pattern used by the decoder
        bin. The write side uses gemma3-style ``pbi_init`` pointers + per-call
        DRAM-delta DMAs, which is the proven SRAM→DRAM PBI scatter shape.

        Per-iteration t-counter is a locally-allocated GPR that increments by 1
        at end-of-body via ``add_inc``. Released after the loop.
        """
        bpe = self.bytes_per_element
        bytes_per_call = element_count * bpe
        _, sram_words = self.sram_address_to_uram_address(sram_byte_addr)

        # Allocate write PBI pointers (one per destination stream).
        ptr_ws = [self.alloc_inst_ptr() for _ in write_specs]
        for ptr_w, (dst_base, _stride) in zip(ptr_ws, write_specs):
            self.generate_instruction_pbi_init(
                dram_shared_addr=dst_base,
                dma_length=bytes_per_call,
                uram_a_start_addr=sram_words,
                uram_b_start_addr=sram_words,
                inst_pointer_idx=ptr_w,
            )

        # Per-token counter t (0, 1, 2, ...) — used to compute read DRAM addr.
        t_reg = self.alloc_isa_reg()
        self.generate_instruction_add_set(t_reg, 0)

        self.loop_start(loop_cnt=template_seq_len, gpr_loop_cnt=gpr_seq_len)
        # Read DRAM addr = read_base + t * read_stride_bytes
        self.generate_instruction_reg_mul_imm(
            self.TMP_REG, t_reg, ue_35bit_addr_shifter(read_stride_bytes))
        self.generate_instruction_add_imm(
            self.TMP_REG, ue_35bit_addr_shifter(read_base), self.TMP_REG)
        # DRAM→SRAM with runtime-computed source DRAM addr (delivered via TMP_REG).
        self.accelerator_memory_to_sram(
            accelerator_dram_address=0,
            sram_address=sram_byte_addr,
            element_size=element_count,
            general_reg_src=self.TMP_REG,
        )
        # SRAM→DRAM via PBI pointers (each advances its DRAM addr by its stride).
        for ptr_w, (_base, dst_stride) in zip(ptr_ws, write_specs):
            self.sram_to_accelerator_memory(
                sram_address=0,
                accelerator_dram_address=dst_stride,
                element_size=element_count,
                inst_pointer_idx=ptr_w,
                memcpy_length_bytes=0,
            )
        # t += 1
        self.generate_instruction_add_inc(t_reg)
        self.loop_end()

        self.release_isa_reg()  # t_reg
        for ptr in reversed(ptr_ws):
            self.release_inst_ptr(ptr)

    def _emit_prefill_program(self, seq_len: int, layer_size: int) -> int:
        """Emit ONE seq_len-agnostic prefill program (no capture-session boundary).

        Caller wraps this in start_capture()/stop_capture(). The emitted program
        ends with a halt so it can be jumped to via a runtime preamble that
        primes gpr_seq_len / gpr_q_seq_len / gpr_bucket_idx GPRs.

        ``seq_len`` here is a **compile-time template** used only for FLOPS
        bookkeeping and as the static ``M=`` arg to PBI ops (overridden at
        runtime by ``gpr_M_reg``). K/V/Q scatter, FLASH_K/V duplication, and the
        FLASH_OUTPUT assembly are emitted as **PBI runtime loops** keyed off
        ``gpr_seq_len``, so the bin is truly seq_len-agnostic up to
        ``MAX_CONTEXT_SIZE`` regardless of the template value.

        All bulk ops (rms_norm, matmat, rope, eltwise) are PBI-dispatched via
        gpr_M_reg=self.gpr_seq_len. Per-token outer loops for matmul/norm/rope
        run gpr_seq_len iterations at runtime regardless of the template value.
        Flash attention dispatches on gpr_bucket_idx.
        """
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        # 1-based bucket selector at the static template; runtime overrides via gpr_bucket_idx.
        num_buckets_prefill = max(1, aligned_seq_len // UE_VECTOR_SIZE)

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

        # §7 shared-subroutine attention: flash_attention_core is compiled ONCE
        # after the program HALT; every (layer, kv_head) call site sets gpr_ret_id
        # to its return word address and jumps in, and the subroutine returns via
        # JUMP_REG_ABS(gpr_ret_id). This shrinks the prefill image ~Nkv*Nlayer×.
        # gpr_ret_id is held (reg 5) across the whole layer loop; per-layer PBI ops
        # alloc dynamic regs from 6 and release back, so it never gets clobbered.
        program_dram_base = self.get_program_dram_addr()
        gpr_ret_id = self.alloc_isa_reg()
        call_site_jump_capture_indices: list[int] = []

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

                # §7: call the shared flash-attention subroutine (compiled once
                # after HALT). Q/K/V are already marshaled into the fixed FLASH_Q/K/V
                # buffers by the scatters above, so this is a pure call-site jump:
                # set gpr_ret_id to the post-jump return address, record the jump
                # slot for back-patching, then JUMP_ABS (placeholder target=0).
                self.pad_capture_to_64b_boundary()
                return_word_addr = ue_35bit_addr_shifter(
                    program_dram_base + (self.capture_count + 2) * INSTRUCTION_SIZE_BYTES)
                self.generate_instruction_add_set(gpr_ret_id, return_word_addr)
                call_site_jump_capture_indices.append(self.capture_count)
                self.generate_instruction_jump_abs(target_instruction_word_addr=0)

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
            # DEBUG (task #45): pinpoint where 4B prefill emit hangs. Last-layer
            # body finishing should be visible; if it never prints, hang is
            # inside layer 35's body matmul/eltwise emit.
            if layer_idx == layer_size - 1:
                _original_print(f"\n    [debug] prefill layer {layer_idx} body emit completed", flush=True)

        # HALT ends the normal execution path; the flash-attention subroutine
        # follows and is reachable only via the call-site JUMP_ABS jumps above.
        self.generate_instruction_halt()

        # §7: compile flash_attention_core ONCE as a subroutine. Each bucket body
        # ends with JUMP_REG_ABS(gpr_ret_id) — returning to whichever call site
        # primed gpr_ret_id. Returns (subroutine_start_addr, per_bucket_flops).
        flash_sub_start, flash_flops = self.flash_attention_core(
            head_dim=ahd,
            seq_len=aligned_seq_len,
            Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
            K_DRAM_ADDR=self.LAYER0_FLASH_K_DRAM,
            V_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
            OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUT_HEAD_DRAM,
            SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
            BIAS_DRAM_ADDR=self.LAYER0_FLASH_BIAS_DRAM,
            ATTN_P_DRAM_ADDR=self.LAYER0_FLASH_ATTN_P_DRAM,
            gpr_bucket_idx=self.gpr_bucket_idx,
            num_buckets=num_buckets_prefill,
            IDENTITY_DRAM_ADDR=self.IDENTITY_DRAM_ADDR,
            gpr_ret_id=gpr_ret_id,
        )
        total_flops += flash_flops[num_buckets_prefill - 1] * layer_size * nkvh

        # Back-patch every call-site JUMP_ABS placeholder to the subroutine start.
        for jump_idx in call_site_jump_capture_indices:
            self._patch_jump_immediate(jump_idx, ue_35bit_addr_shifter(flash_sub_start))
        self.release_isa_reg()  # gpr_ret_id
        _original_print("    [debug] _emit_prefill_program: halt emitted, returning", flush=True)
        return total_flops

    def run_prefill(self, prefill_program_addr: int, preamble_addr: int,
                    prefill_seq, gflops: int = None) -> dict:
        """Run prefill via dynamic-PBI runtime preamble.

        Emits a tiny preamble program at ``preamble_addr`` that primes the three
        runtime GPRs (gpr_seq_len, gpr_q_seq_len, gpr_bucket_idx) and then
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
        bucket_idx = max(1, aligned_seq_len // UE_VECTOR_SIZE)

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
        self.generate_instruction_add_set(self.gpr_seq_len,    actual_seq_len)
        self.generate_instruction_add_set(self.gpr_q_seq_len,  q_seq_len)
        self.generate_instruction_add_set(self.gpr_bucket_idx, bucket_idx)
        self.generate_instruction_jump_abs(ue_35bit_addr_shifter(prefill_program_addr))
        self.stop_capture()
        self.write_captured_instructions_to_dram(preamble_addr)
        self.clear_capture_buffer()

        # Execute from the preamble — it jumps into the cached prefill, which halts.
        self.program_execute(preamble_addr, gflops=gflops)

    def _emit_decoder_program(self, layer_size: int) -> int:
        """Emit ONE decode-position-agnostic decoder program.

        Caller wraps this in start_capture()/stop_capture(). The emitted program
        ends with a halt so it can be jumped to via a runtime preamble that
        primes gpr_bucket_idx (1-based KV context bucket). gpr_seq_len is
        primed once by the prefill preamble and auto-incremented at the end
        of every decoder execution via ADD_INC.

        Address math: K/V cache write addresses are computed at runtime as
        ``base + gpr_seq_len * (ahd*bpe)`` via reg_mul_imm + add_imm into TMP_REG,
        then passed as ``general_reg_src=TMP_REG`` to the scatter store.

        Decoder attention uses the PBI-bucketed
        ``decoder_group_attention_core`` (one body per KV head, processing all
        ``qpkv`` Q heads at once), with bucket selection driven by
        ``gpr_bucket_idx``. Each bucket body covers exactly the active KV
        range — no stale-cache positions enter the softmax, eliminating the
        NaN cascade the legacy static-seq_len kernel produced.

        All bulk ops (matmul, rms_norm) use ``gpr_M_reg`` GPRs (gpr_one for M=1,
        gpr_nkvh for M=nkvh, gpr_nkvh_qpkv for M=nkvh*qpkv).
        """
        ahd  = self.actual_head_dim   # 128
        nkvh = self.num_kv_heads      # 8
        qpkv = self.group_size        # 2
        bpe  = self.bytes_per_element
        hd   = self.head_dim          # 1024
        rope_row_bytes = ahd * 2 * bpe   # 512 B per token

        # Decoder attention bucketing: each bucket body covers seq_len = i * UE_VECTOR_SIZE
        # for i = 1..num_buckets. num_buckets must cover MAX_CONTEXT_SIZE.
        num_buckets_decoder = (self.MAX_CONTEXT_SIZE + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
        ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE_LOCAL

        # Reset GPR allocator. Keep long-lived locals to a minimum — the PBI bucketed
        # decoder kernel internally allocates registers per bucket body (bucket_scratch,
        # m_reg, transpose's 4 + nested loop counters), so we have a tight budget.
        # Three locals survive across the layer loop:
        #   gpr_ret_id       (5) — §7 shared-subroutine return-address reg
        #   gpr_one          (6) — gpr_M_reg for M=1 matmul/norm ops
        #   gpr_rope_cos_abs (7) — absolute cos-base addr for this step's rope; reused
        #                         across all rope_hf_core calls in all layers
        # All three are LIVE when the shared decode-attention subroutine is called
        # mid-layer, so the subroutine must allocate its scratch ABOVE them (from
        # counter 8) — see the no-counter-lowering note at the post-HALT emit.
        self._isa_reg_counter = 5
        gpr_ret_id       = self.alloc_isa_reg()
        gpr_one          = self.alloc_isa_reg()
        gpr_rope_cos_abs = self.alloc_isa_reg()
        # §7: decoder_group_attention_core is compiled ONCE after the program HALT;
        # each (layer, kv_head) call site marshals its KV history into the fixed
        # FLASH_K/FLASH_V buffers, sets gpr_ret_id, and jumps in.
        program_dram_base = self.get_program_dram_addr()
        call_site_jump_capture_indices: list[int] = []
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

            # Q, K, V projections — fused for IF4 layers, unfused for L35 bf16 (see Trick 7).
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

                # §7: gather valid K history (incl. the just-written token) from the
                # per-head cache → fixed FLASH_K buffer that the shared subroutine
                # reads. Loop count = gpr_bucket_idx (number of 64-token blocks), so
                # only the active KV range is copied, not the full MAX_CONTEXT_SIZE.
                self._emit_pbi_scatter_per_token(
                    read_base=k_cache_base,
                    read_stride_bytes=UE_VECTOR_SIZE * ahd * bpe,
                    write_specs=[(self.LAYER0_FLASH_K_DRAM, UE_VECTOR_SIZE * ahd * bpe)],
                    sram_byte_addr=0,
                    element_count=UE_VECTOR_SIZE * ahd,
                    gpr_seq_len=self.gpr_bucket_idx,
                    template_seq_len=num_buckets_decoder,
                )

                # V cache write
                self.accelerator_memory_to_sram(self.LAYER0_V_PROJ_TEMP + kv_h * ahd * bpe, 0x20000, ahd)
                self.generate_instruction_reg_mul_imm(
                    self.TMP_REG, self.gpr_seq_len, ue_35bit_addr_shifter(ahd * bpe))
                self.generate_instruction_add_imm(
                    self.TMP_REG, ue_35bit_addr_shifter(v_cache_base), self.TMP_REG)
                self.sram_to_accelerator_memory(
                    sram_address=0x20000, accelerator_dram_address=0,
                    element_size=ahd, general_reg_src=self.TMP_REG)

                # §7: gather valid V history → fixed FLASH_V buffer (qwen3 decode puts
                # v_proj in V_PROJ_TEMP, not FLASH_V, so FLASH_V base is free — no
                # k_size offset needed, unlike llama3.2).
                self._emit_pbi_scatter_per_token(
                    read_base=v_cache_base,
                    read_stride_bytes=UE_VECTOR_SIZE * ahd * bpe,
                    write_specs=[(self.LAYER0_FLASH_V_DRAM, UE_VECTOR_SIZE * ahd * bpe)],
                    sram_byte_addr=0,
                    element_count=UE_VECTOR_SIZE * ahd,
                    gpr_seq_len=self.gpr_bucket_idx,
                    template_seq_len=num_buckets_decoder,
                )

                # Gather this KV head's Q group (qpkv contiguous head_dim rows) into FLASH_Q.
                for q in range(qpkv):
                    q_src = self.LAYER0_Q_NORM_DRAM + (kv_h * qpkv + q) * ahd * bpe
                    flash_q_addr = self.LAYER0_FLASH_Q_DRAM + q * ahd * bpe
                    self.accelerator_memory_to_sram(q_src, 0x30000, ahd)
                    self.sram_to_accelerator_memory(0x30000, flash_q_addr, ahd)

                # §7: call the shared decoder-attention subroutine (compiled once
                # after HALT). Q/K/V are now in the fixed FLASH_Q/K/V buffers; the
                # subroutine writes this head's output to the fixed FLASH_OUT_HEAD,
                # which we then copy into this head's slot of FLASH_OUTPUT.
                self.pad_capture_to_64b_boundary()
                return_word_addr = ue_35bit_addr_shifter(
                    program_dram_base + (self.capture_count + 2) * INSTRUCTION_SIZE_BYTES)
                self.generate_instruction_add_set(gpr_ret_id, return_word_addr)
                call_site_jump_capture_indices.append(self.capture_count)
                self.generate_instruction_jump_abs(target_instruction_word_addr=0)
                # Copy per-head output FLASH_OUT_HEAD → FLASH_OUTPUT[kv_h slot].
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
        # Host preamble primes only gpr_bucket_idx between steps; gpr_seq_len carries
        # across via this in-bin increment.
        self.generate_instruction_add_inc(self.gpr_seq_len)

        # End-of-program halt so the runtime preamble's JUMP_ABS returns control after execute.
        self.generate_instruction_halt()

        # §7: compile decoder_group_attention_core ONCE as a subroutine after HALT.
        # Do NOT lower _isa_reg_counter: the subroutine is CALLED mid-layer (before
        # o_proj/MLP, which use gpr_one as M=1), so its scratch must allocate ABOVE
        # gpr_ret_id(5)/gpr_one(6)/gpr_rope_cos_abs(7) — i.e. from counter 8. Lowering
        # it makes the kernel reuse reg 6/7, corrupting gpr_one/gpr_rope_cos_abs on
        # every call → wrong-M matmuls → garbage/hang. The kernel is shallow, so
        # base 8 stays well under the 15-GPR ceiling.
        self.pad_capture_to_64b_boundary()
        dec_sub_start, dec_attn_flops = self.decoder_group_attention_core(
            group_size=qpkv,
            head_dim=ahd,
            seq_len=self.MAX_CONTEXT_SIZE,
            Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
            K_DRAM_ADDR=self.LAYER0_FLASH_K_DRAM,
            V_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
            OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUT_HEAD_DRAM,
            IDENTITY_DRAM_ADDR=self.IDENTITY_DRAM_ADDR,
            SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
            BIAS_DRAM_ADDR=self.LAYER0_FLASH_BIAS_DRAM,
            gpr_bucket_idx=self.gpr_bucket_idx,
            num_buckets=num_buckets_decoder,
            gpr_ret_id=gpr_ret_id,
        )
        total_flops += dec_attn_flops[-1] * layer_size * nkvh

        # Back-patch every call-site JUMP_ABS placeholder to the subroutine start.
        for jump_idx in call_site_jump_capture_indices:
            self._patch_jump_immediate(jump_idx, ue_35bit_addr_shifter(dec_sub_start))
        return total_flops

    def compile_instructions(self, layer_size: int | None = None) -> dict:
        """Compile a UNIFIED single-bin instruction image: ONE prefill program
        + ONE decoder program in one capture session. Writes
        ``qwen3_4b_instruction.bin`` + matching ``.json`` meta to disk.

        Both programs are **seq_len-agnostic**: matmul/norm/rope row counts come
        from gpr_seq_len / gpr_q_seq_len GPRs, and flash attention dispatches on
        gpr_bucket_idx — all primed by the runtime preamble in run_prefill /
        run_decoder per call. The cached bin therefore works across any prompt
        length ≤ PREFILL_MAX_SEQ_LEN and any decode position < MAX_CONTEXT_SIZE.

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
        bin_rel  = paths_cfg.get("instruction_bin",  "qwen3_4b_bin/qwen3_4b_instruction.bin")
        meta_rel = paths_cfg.get("instruction_meta", "qwen3_4b_bin/qwen3_4b_instruction.json")
        bin_path  = os.path.join(self.script_dir, bin_rel)
        meta_path = os.path.join(self.script_dir, meta_rel)

        # The decode LM head is uniform now: write_back_disable=True + the penalty C bias
        # (PENALTY_BIAS_DRAM). Penalty vs --pure-greedy differ only in the runtime bias
        # buffer contents, NOT in the instructions, so one bin serves both. The cache is
        # keyed on this signature so a stale bin from the older sampling/greedy builds is
        # detected and recompiled.
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
        gpr_seq_len (current decode position, ==previous token count) and
        gpr_bucket_idx (1-based KV context bucket), then jump_abs's into the
        cached decoder program. Same preamble slot is overwritten per token.

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
        # Decoder PBI buckets cover seq_len = i * UE_VECTOR_SIZE for i = 1..num_buckets,
        # spanning MAX_CONTEXT_SIZE. Bias buffer must match the maximum bucket size
        # since each bucket body reads its own static seq_len of bias rows.
        num_buckets_decoder = (self.MAX_CONTEXT_SIZE + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
        max_bucket_seq_len = num_buckets_decoder * UE_VECTOR_SIZE
        max_seq_len = self.MAX_CONTEXT_SIZE

        # On-FPGA repetition-penalty state. _generated_tokens is seeded with the prompt by
        # main() (or run_from_bin); decoded ids are appended below. The penalty is folded
        # into the LM-head matmul bias, so token selection is always the HW argmax — no host
        # logit readback. Position-gated: pure greedy for the first `greedy_until` decoded
        # tokens, then the per-vocab bias turns on. --pure-greedy leaves the bias all-zero.
        if not hasattr(self, "_generated_tokens"):
            self._generated_tokens = []
        _fpga_penalty = bool(getattr(self, "fpga_penalty", True))
        _greedy_until = int(getattr(self, "greedy_until", 512))
        _prompt_len = len(self._generated_tokens)
        self.dma_to_accelerator_memory(self.PENALTY_BIAS_DRAM,
                                       torch.zeros(1, self.EMBEDDING_ELEMENTS, dtype=torch.bfloat16))

        while self.seq_len < max_seq_len:
            _SILENT_MODE = True
            # self.seq_len at entry is the count of K/V already in cache; the
            # current decode token will be written at that index.
            decode_pos = self.seq_len               # K/V cache write index for this token
            new_ctx_len = decode_pos + 1            # KV positions [0..decode_pos] inclusive
            aligned_ctx = ((new_ctx_len + 63) // 64) * 64
            bucket_idx = max(1, aligned_ctx // UE_VECTOR_SIZE)

            embedding_tensor = self.get_embedding_for_tokens([token_id])
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
            # Bias sized for the maximum decoder bucket; positions
            # [new_ctx_len..max_bucket_seq_len-1] are -inf so the selected bucket
            # body never sums over uninitialized KV-cache slots.
            bias_host = torch.full((1, max_bucket_seq_len), -1e36, dtype=torch.bfloat16)
            bias_host[0, :new_ctx_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_host)

            # Decode preamble: prime gpr_bucket_idx only; gpr_seq_len is carried over
            # from prefill (or previous decode step) via the in-bin ADD_INC.
            self.clear_inst_id()
            self.start_capture()
            self.generate_instruction_add_set(self.gpr_bucket_idx, bucket_idx)
            self.generate_instruction_jump_abs(ue_35bit_addr_shifter(decoder_program_addr))
            self.stop_capture()
            self.write_captured_instructions_to_dram(preamble_addr)
            self.clear_capture_buffer()

            # Refresh the per-vocab penalty bias (this step's LM-head matmul C term) once
            # past the gate, BEFORE execute, so the HW argmax of (logits + bias) returns the
            # penalized token directly. One full-buffer DMA per step.
            if _fpga_penalty and (len(self._generated_tokens) - _prompt_len) > _greedy_until:
                self._write_penalty_bias(self._generated_tokens)

            self.start_execute_from_dram(preamble_addr)
            self.wait_queue(10.0)
            # Read the HW argmax register — the LM-head matmul already added the penalty bias
            # on chip, so this is the penalized (or pure-greedy) token. No host logit readback.
            token_id = self.get_arg_max_index()
            self._generated_tokens.append(token_id)
            token_char = self.tokenizer.decode([token_id])
            _SILENT_MODE = False
            self.seq_len += 1
            if token_id in _qwen3_stop_tokens:
                print(f"\nStop token {token_id} reached.")
                break
            print(token_char, end="", flush=True)
        return self.seq_len

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3-4B prefill + decode on accelerator.")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt: tokenizer encodes this to prefill_seq (overrides default)")
    parser.add_argument("--local-weights", action="store_true", help="Use qwen3_4b_bin/full_model_weights.bin instead of generated weights_qwen3_4b_hf.bin")
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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.local_weights:
        weights_bin_rel = "qwen3_4b_bin/full_model_weights.bin"
    else:
        weights_bin_rel = "qwen3_4b_bin/weights_qwen3_4b_hf.bin"
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

    ue = Qwen3_4b_UnifiedEngine(script_dir=script_dir, weights_bin=weights_bin_rel)
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
                                  "qwen3_4b_bin/qwen3_4b_instruction.bin"))
    base_addr, _ = ue.load_instructions(inst_bin_path)
    # Reserve a slot AFTER the loaded bin for the runtime preamble. Prefill preamble
    # is 4 instructions (3 ADD_SET + JUMP_ABS) = 128 B; decode preamble is 2 instructions
    # (ADD_SET gpr_bucket_idx + JUMP_ABS) and overwrites the same slot.
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
    ue.clear_dram()
    print("Qwen3-4B test ends.")

if __name__ == "__main__":
    main()
