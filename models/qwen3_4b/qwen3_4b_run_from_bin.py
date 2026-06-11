#!/usr/bin/env python3
"""Qwen3-4B inference from pre-compiled bins (self-contained offline runner).

This script is self-contained: it does NOT import from qwen3_4b_test.py. It
ships with everything needed to load weights, initialize tensors, load a
pre-compiled instruction bin from disk, and run prefill + decode.

Requirements on disk (relative to this file):
  - ``qwen3_4b_bin/qwen3_4b_config.json``   model config
  - ``qwen3_4b_bin/weights_qwen3_4b_hf.bin`` quantized weight bin
  - ``qwen3_4b_bin/qwen3_4b_instruction.bin`` pre-compiled program bin
  - ``qwen3_4b_bin/qwen3_4b_instruction.json`` compile-meta sidecar
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
  python qwen3_4b_run_from_bin.py --dev xdma0 [--device kintex7] [--cycle 5.15]
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
from user_dma_core import DMA_DEVICE_H2C, TYPE, UE_FMAX_CONTEXT_SIZE, UE_MODE, UE_VECTOR_SIZE, UE_ARGMAX_INDEX, SCALE_BRAM_ELEMENTS, set_dma_device, ue_35bit_addr_shifter
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
        # build_API_for_FPGAcomponents.md.

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

    def sample_next_token(self, prev_tokens: list[int]) -> int:
        """Read full logits from LOGITS_DRAM and sample using temperature /
        top_k / top_p / repetition_penalty. Greedy (argmax) when temperature==0."""
        vocab = self.EMBEDDING_ELEMENTS
        bpe = self.bytes_per_element
        buf = torch.empty(vocab, dtype=torch.bfloat16)
        self.dma_read(DMA_DEVICE_C2H, self.LOGITS_DRAM, buf, vocab * bpe)
        logits = buf.float()
        rep_pen = float(getattr(self, "repetition_penalty", 1.0))
        if rep_pen != 1.0 and prev_tokens:
            seen = torch.tensor(list(set(prev_tokens)), dtype=torch.long)
            v = logits[seen]
            logits[seen] = torch.where(v > 0, v / rep_pen, v * rep_pen)
        temp = float(getattr(self, "temperature", 1.0))
        if temp <= 0:
            return int(logits.argmax().item())
        logits = logits / temp
        top_k = int(getattr(self, "top_k", 0) or 0)
        if top_k > 0 and top_k < vocab:
            kth = torch.topk(logits, top_k).values[-1]
            logits[logits < kth] = float("-inf")
        top_p = float(getattr(self, "top_p", 1.0))
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumprob = torch.cumsum(probs, dim=-1)
            keep = cumprob <= top_p
            keep[0] = True
            drop_idx = sorted_idx[~keep]
            logits[drop_idx] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

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

            self.start_execute_from_dram(preamble_addr)
            self.wait_queue(10.0)
            if not hasattr(self, "_generated_tokens"):
                self._generated_tokens = []
            if float(getattr(self, "temperature", 0.0)) > 0:
                token_id = self.sample_next_token(self._generated_tokens)
            else:
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

# -----------------------------------------------------------------------------
# Offline runner — no compile machinery. Reads meta JSON from disk, loads the
# pre-compiled bin to DRAM, runs prefill + decoder.
# -----------------------------------------------------------------------------
def _clock_ns_default_for_device(device: str) -> float:
    """Return default clock period (ns) for FPGA type — mirrors user_hw_test.py."""
    if device == "kintex7":                       return 5.1594
    if device in ("rk", "puzhi"):                 return 3.0
    if device in ("bittware", "bittware_256"):     return 3.3333
    if device == "alveo":                          return 4.0
    return 10.0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3-4B inference from pre-compiled bins (offline)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt (default: from qwen3_4b_config.json default_prompt)")
    parser.add_argument("--local-weights", action="store_true",
                        help="Use qwen3_4b_bin/full_model_weights.bin instead of weights_qwen3_4b_hf.bin")
    parser.add_argument("--dev", type=str, default="xdma0",
                        help="DMA device name (default: xdma0)")
    parser.add_argument("--cycle", type=float, default=None, help='Clock cycle time in ns. Overrides --device default.')
    parser.add_argument("--device", type=str, default="kintex7", help='FPGA board profile (kintex7, rk, puzhi, bittware, bittware_256, alveo).')
    # Sampling DEFAULTS to the validated long-context config (temp 0.7 / top-p 0.9
    # / rep-penalty 1.2). --temperature 0 forces greedy fast path.
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature (0=greedy via HW argmax; >0 enables SW sampling). Default 0.7.')
    parser.add_argument('--top-k', type=int, default=0,
                        help='Top-k filter (0=disabled). Active only when --temperature > 0.')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='Top-p (nucleus) filter. Default 0.9.')
    parser.add_argument('--repetition-penalty', type=float, default=1.2,
                        help='HF-style repetition penalty (>1 down-weights repeats). Default 1.2.')
    parser.add_argument('--seed', type=int, default=None,
                        help='RNG seed for reproducible sampling (only when --temperature > 0).')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    bin_dir = os.path.join(script_dir, "qwen3_4b_bin")

    weights_bin_rel = ("qwen3_4b_bin/full_model_weights.bin" if args.local_weights
                       else "qwen3_4b_bin/weights_qwen3_4b_hf.bin")
    weights_bin_full = os.path.join(script_dir, weights_bin_rel)

    # Hard-fail BEFORE any FPGA / HF touch if a required local file is missing.
    missing = []
    if not os.path.exists(weights_bin_full):
        missing.append(os.path.relpath(weights_bin_full, script_dir))
    for name in ("qwen3_4b_instruction.bin", "qwen3_4b_instruction.json"):
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
    axi_width_bits = 512 if args.device in ("bittware", "rk") else 256
    os.environ["UE_AXI_DATA_WIDTH_BITS"] = str(axi_width_bits)
    user_dma_core.UE_AXI_DATA_WIDTH_BITS = axi_width_bits
    clock = args.cycle if args.cycle is not None else _clock_ns_default_for_device(args.device)
    user_dma_core.CLOCK_CYCLE_TIME_NS = clock
    user_dma_core.UE_PEAK_GFLOPS = 0.128 / clock
    _original_print(f"FPGA profile: device={args.device}, clock={clock:.4f} ns, UE_AXI_DATA_WIDTH_BITS={axi_width_bits}")

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

    # Sampling config. NOTE: the LM-head writeback mode is baked into the
    # pre-compiled bin at compile time (qwen3_4b_test.py). Sampling needs logits
    # in DRAM (writeback ON), which is the default since test.py defaults to
    # temperature>0. If the bin was compiled with --temperature 0 (greedy,
    # writeback OFF), sampling here would read stale logits — recompile via
    # test.py with sampling defaults first.
    ue.temperature = float(args.temperature)
    ue.top_k = int(args.top_k)
    ue.top_p = float(args.top_p)
    ue.repetition_penalty = float(args.repetition_penalty)
    ue._generated_tokens = list(prefill_seq)
    if args.seed is not None and ue.temperature > 0:
        torch.manual_seed(args.seed)
    if ue.temperature > 0:
        _original_print(f"Sampling: temperature={ue.temperature}  top_k={ue.top_k}  "
                        f"top_p={ue.top_p}  repetition_penalty={ue.repetition_penalty}  seed={args.seed}")

    # Read meta JSON directly from disk — no compile_instructions call.
    paths_cfg = cfg.get("paths", {})
    inst_bin_path = os.path.join(script_dir, paths_cfg.get("instruction_bin",
                                  "qwen3_4b_bin/qwen3_4b_instruction.bin"))
    meta_path = os.path.splitext(inst_bin_path)[0] + ".json"
    with open(meta_path) as _f:
        inst_meta = json.load(_f)
    _original_print(f"  Loaded compile meta from {os.path.relpath(meta_path, script_dir)}")

    _original_print(f"\n--- Loading unified instruction bin ---")
    timer = time.perf_counter()
    base_addr, total_size = ue.load_instructions(inst_bin_path)
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
    _original_print(f"\nDecoder done in {latency_prefill + latency_decoder:.2f}s, total {token_cnt} tokens, "
                    f"decode speed: {decoded_tokens / latency_decoder:.2f} tokens/s ({decoded_tokens} decoded tokens / {latency_decoder:.2f}s).")


if __name__ == "__main__":
    main()
