#!/usr/bin/env python3
"""Qwen3-0.6B inference from pre-compiled bins (self-contained offline runner).

This script is self-contained: it does NOT import from qwen3_0.6b_test.py. It
ships with everything needed to load weights, initialize tensors, load a
pre-compiled instruction bin from disk, and run prefill + decode.

Requirements on disk (relative to this file):
  - ``qwen3_0.6b_config.json`` model config (beside this script)
  - ``qwen3_0.6b_bin/params.bin`` quantized weight bin
  - ``qwen3_0.6b_bin/params.json`` weight-bin sidecar
  - ``qwen3_0.6b_bin/programs.bin`` pre-compiled program bin
  - ``qwen3_0.6b_bin/programs.json`` compile-meta sidecar
  - ``qwen3_0.6b_bin/Qwen3-0.6B/`` tokenizer files

If any of those are missing, exit early with a clear message. Generate them
on a build machine that has HF access by running qwen3_0.6b_test.py once.

Architecture notes:
  - 28 layers, 16 Q / 8 KV heads (group_size=2), actual head_dim=128.
  - hidden_size=1024, mlp_intermediate=3072, vocab=151936.
  - QK RMSNorm per head (gamma_offset=0.0). No post-attn/post-FFN norm.
  - SwiGLU activation. Single RoPE base theta=1_000_000.
  - Tied HF embedding/lm_head, with an exact BF16 device embedding table and a
    separately quantized accelerator LM head.
  - Adaptive IF4 is the default for every layer; packaged BF16 L27 weights remain
    available through ``--bf16-last-layer``.

Usage:
  python qwen3_0.6b_run_from_bin.py
  python qwen3_0.6b_run_from_bin.py --prompt "your prompt"
  python qwen3_0.6b_run_from_bin.py --dev xdma0 [--cycle 5.62]
  python qwen3_0.6b_run_from_bin.py --local-weights
"""

import hashlib
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
from quant_lib import quantize

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

# down_proj K=3072 ≤ SCALE_BRAM_ELEMENTS=8192 — no K-split needed.

_DEVICE_DRAM_LIMIT = 0x100000000
_PROGRAM_DRAM_BASE = 0xE0000000
_RUNTIME_PREAMBLE_BYTES = 128
_EMBEDDING_DISPATCH_STRIDE_BYTES = 64

def _parse_offset(val) -> int:
    """Parse offset/size from JSON: int or hex string like '0x24000000'."""
    if isinstance(val, str):
        return int(val, 0)
    return int(val)

def _model_layout_signature(cfg: dict) -> str:
    """Stable signature for config fields that determine packed/program addresses."""
    public_cfg = {key: value for key, value in cfg.items() if not key.startswith("_")}
    payload = json.dumps(public_cfg, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()

def _instruction_compiler_fingerprint(script_dir: str, execution_sig: str) -> str:
    """Hash every source that can change emitted accelerator instructions."""
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    sources = (
        ("model_codegen", os.path.join(script_dir, "qwen3_0.6b_test.py")),
        ("engine_codegen", os.path.join(repo_root, "user_dma_core.py")),
        ("model_config", os.path.join(script_dir, "qwen3_0.6b_config.json")),
    )
    digest = hashlib.sha256()
    digest.update(b"qwen3_0.6b-program-abi-v1\0")
    digest.update(execution_sig.encode())
    for label, path in sources:
        digest.update(b"\0" + label.encode() + b"\0")
        with open(path, "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()

def _weight_artifact_fingerprint(cfg: dict, script_dir: str) -> str:
    """Fingerprint the packed-weight ABI and canonical IF4 codec."""
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    digest = hashlib.sha256()
    digest.update(b"qwen3_0.6b-weight-pack-v2\0")
    digest.update(_model_layout_signature(cfg).encode())
    with open(os.path.join(repo_root, "quant_lib.py"), "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _expected_weight_bin_size(cfg: dict) -> int:
    """Return the exact byte length implied by all configured file regions."""
    embedding = cfg["special"]["embedding"]
    ends = [_parse_offset(embedding["token_embd_offset"]) + _parse_offset(embedding["token_embd_size"])]
    num_layers = cfg["file_info"]["num_layers"]
    layer_size = cfg["file_info"]["layer_size"]
    ends.extend(
        _parse_offset(region["offset"]) + (num_layers - 1) * layer_size + region["size"]
        for region in cfg.get("regions", {}).values()
    )
    ends.extend(
        _parse_offset(region["offset"]) + region["size"]
        for region in cfg.get("non_layer_regions", {}).values()
    )
    return max(ends)

def _device_bf16_embedding_layout(cfg: dict) -> tuple[int, int]:
    """Return the fixed top-down device-DRAM slot for the exact BF16 table."""
    embedding = cfg["special"]["embedding"]
    size = (int(embedding["vocab_size"]) * int(embedding["embedding_dim"])
            * int(cfg["file_info"]["bytes_per_element"]))
    configured_size = _parse_offset(embedding["token_embd_size"])
    if configured_size != size:
        raise ValueError(
            f"Embedding layout mismatch: config region is {configured_size} bytes, "
            f"but vocab/dimension require {size}."
        )
    base = _DEVICE_DRAM_LIMIT - size
    if base < _PROGRAM_DRAM_BASE or base % 64:
        raise ValueError(
            f"BF16 embedding slot 0x{base:X}..0x{_DEVICE_DRAM_LIMIT:X} is invalid."
        )
    return base, size

def _validate_program_embedding_separation(
        program_size: int, embedding_base: int, dispatch_size: int) -> None:
    """Keep programs, runtime preamble, and token dispatch below the embedding."""
    aligned_program_size = (int(program_size) + 63) & ~63
    program_end = (_PROGRAM_DRAM_BASE + aligned_program_size
                   + _RUNTIME_PREAMBLE_BYTES + int(dispatch_size))
    if program_end > embedding_base:
        raise ValueError(
            f"programs.bin, runtime preamble, and token dispatch end at "
            f"0x{program_end:X}, "
            f"overlapping the BF16 embedding table at 0x{embedding_base:X}."
        )

def _quantize_bf16_to_if4_packed(weight_bf16: torch.Tensor, block_size: int = 64) -> tuple[bytes, bytes]:
    """Pack BF16 weights as adaptive IF4, one scale per 64-value K block.

    The canonical codec picks INT4 or FP4 independently per block by minimum
    reconstruction MSE. The BF16 scale sign carries the hardware dispatch bit,
    so storage and accelerator kernels are unchanged from forced INT4.
    """
    return quantize("if4", weight_bf16, block_size=block_size)

def _validate_hf_model_layout(model, cfg: dict) -> None:
    """Reject a checkpoint whose architecture cannot match the packed layout."""
    fi = cfg["file_info"]
    hf_cfg = model.config
    expected_fields = {
        "model_type": "qwen3",
        "hidden_size": fi["hidden_size"],
        "intermediate_size": fi["mlp_elements"],
        "num_hidden_layers": fi["num_layers"],
        "num_attention_heads": fi["num_kv_heads"] * fi["group_size"],
        "num_key_value_heads": fi["num_kv_heads"],
        "head_dim": fi["actual_head_dim"],
        "vocab_size": fi["embedding_vocab"],
        "rope_theta": cfg["special"]["rope"]["theta"],
        "tie_word_embeddings": True,
    }
    mismatches = []
    for field, expected in expected_fields.items():
        actual = getattr(hf_cfg, field, None)
        if field == "rope_theta" and actual is None:
            rope_parameters = getattr(hf_cfg, "rope_parameters", None) or {}
            actual = rope_parameters.get("rope_theta")
        if actual != expected:
            mismatches.append(f"config.{field}={actual!r} (expected {expected!r})")
    if getattr(hf_cfg, "attention_bias", False):
        mismatches.append("config.attention_bias=True (bias tensors are not supported)")
    if mismatches:
        raise ValueError("Checkpoint architecture does not match Qwen3-0.6B layout: " + "; ".join(mismatches))

    hidden = fi["hidden_size"]
    q_width = fi["head_dim"] * fi["group_size"]
    kv_width = fi["head_dim"]
    intermediate = fi["mlp_elements"]
    vocab = fi["embedding_vocab"]

    def require_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
        actual = tuple(tensor.shape)
        if actual != expected:
            raise ValueError(f"{name} has shape {actual}; expected {expected}")

    embedding = model.get_input_embeddings().weight
    lm_head = model.get_output_embeddings().weight
    require_shape("model.embed_tokens.weight", embedding, (vocab, hidden))
    require_shape("lm_head.weight", lm_head, (vocab, hidden))
    require_shape("model.norm.weight", model.model.norm.weight, (hidden,))
    if embedding.data_ptr() != lm_head.data_ptr() and not torch.equal(embedding, lm_head):
        raise ValueError("Checkpoint declares tied embeddings but input and output weights differ")

    if len(model.model.layers) != fi["num_layers"]:
        raise ValueError(f"Checkpoint has {len(model.model.layers)} layers; expected {fi['num_layers']}")
    for layer_idx, layer in enumerate(model.model.layers):
        tensors = {
            "input_layernorm.weight": (layer.input_layernorm.weight, (hidden,)),
            "post_attention_layernorm.weight": (layer.post_attention_layernorm.weight, (hidden,)),
            "self_attn.q_proj.weight": (layer.self_attn.q_proj.weight, (q_width, hidden)),
            "self_attn.k_proj.weight": (layer.self_attn.k_proj.weight, (kv_width, hidden)),
            "self_attn.v_proj.weight": (layer.self_attn.v_proj.weight, (kv_width, hidden)),
            "self_attn.o_proj.weight": (layer.self_attn.o_proj.weight, (hidden, q_width)),
            "self_attn.q_norm.weight": (layer.self_attn.q_norm.weight, (fi["actual_head_dim"],)),
            "self_attn.k_norm.weight": (layer.self_attn.k_norm.weight, (fi["actual_head_dim"],)),
            "mlp.gate_proj.weight": (layer.mlp.gate_proj.weight, (intermediate, hidden)),
            "mlp.up_proj.weight": (layer.mlp.up_proj.weight, (intermediate, hidden)),
            "mlp.down_proj.weight": (layer.mlp.down_proj.weight, (hidden, intermediate)),
        }
        for name, (tensor, expected) in tensors.items():
            require_shape(f"model.layers.{layer_idx}.{name}", tensor, expected)
        biased = [
            name for name, module in (
                ("q_proj", layer.self_attn.q_proj), ("k_proj", layer.self_attn.k_proj),
                ("v_proj", layer.self_attn.v_proj), ("o_proj", layer.self_attn.o_proj),
                ("gate_proj", layer.mlp.gate_proj), ("up_proj", layer.mlp.up_proj),
                ("down_proj", layer.mlp.down_proj),
            ) if getattr(module, "bias", None) is not None
        ]
        if biased:
            raise ValueError(f"model.layers.{layer_idx} has unsupported projection biases: {', '.join(biased)}")

def weight_bin_generate(script_dir: str | None = None, output_path: str | None = None) -> str:
    """Generate params.bin from Hugging Face model per qwen3_0.6b_config.json layout.
    Returns the path to the written file. Use this bin to replace full_model_weights.bin."""
    script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
    cfg = _load_config(script_dir)
    weight_defs = cfg["_weight_defs"]
    paths = cfg["paths"]
    paths_full = os.path.join(script_dir, paths["weights_bin"])
    out_path = output_path or paths_full

    model, model_dir = _ensure_hf_model(script_dir, cfg)
    _validate_hf_model_layout(model, cfg)
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
        end = offset + len(data)
        if offset < 0 or end > len(buf):
            raise ValueError(f"Write [{offset}, {end}) exceeds params.bin size {len(buf)}")
        buf[offset:end] = data

    def pad_region(data: bytes, size: int, label: str) -> bytes:
        if len(data) > size:
            raise ValueError(f"{label} has {len(data)} bytes; region allows {size}")
        return data + b"\x00" * (size - len(data))

    # Embedding
    embed = model.get_input_embeddings().weight.detach().cpu().to(torch.bfloat16)
    raw_emb = embed.contiguous().view(torch.uint8).numpy().tobytes()
    if len(raw_emb) != token_embd_size:
        raise ValueError(f"Embedding has {len(raw_emb)} bytes; config expects {token_embd_size}")
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

        # K=3072 ≤ SCALE_BRAM_ELEMENTS=8192 — single down_proj (no split needed)
        region_writes = [
            (gamma_in, "bf16"),
            (q_w, "if4"),
            (k_w, "if4"),
            (v_w, "if4"),
            (gamma_q, "bf16"),
            (gamma_k, "bf16"),
            (o_w, "if4"),
            (gamma_post, "bf16"),
            (gamma_ffn, "bf16"),
            (up_w, "if4"),
            (gate_w, "if4"),
            (down_w, "if4"),
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
            if kind == "if4":
                next_key = blk0_structure[i + 1]["key"]
                data_sz = weight_defs[f"{next_key}_SIZE"]
                data_bytes, scale_bytes = _quantize_bf16_to_if4_packed(tensor)
                scale_padded = pad_region(scale_bytes, sz, f"layer {layer_idx} {off_key}")
                data_padded = pad_region(data_bytes, data_sz, f"layer {layer_idx} {next_key}")
                write_at(file_off, scale_padded)
                data_off = weight_defs[next_key] + layer_idx * LAYER_WEIGHT_SIZE
                write_at(data_off, data_padded)
                i += 2
            else:
                t = tensor.detach().cpu().to(torch.bfloat16).contiguous()
                raw = pad_region(t.view(torch.uint8).numpy().tobytes(), sz,
                                 f"layer {layer_idx} {off_key}")
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
        raw = pad_region(rope_tensor.contiguous().view(torch.uint8).numpy().tobytes(), sz, off_key)
        write_at(weight_defs[off_key], raw)

    # OUTPUT_NORM
    out_norm = (model.model.norm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
    sz = weight_defs["OUTPUT_NORM_WEIGHT_SIZE"]
    raw = pad_region(out_norm.contiguous().view(torch.uint8).numpy().tobytes(), sz,
                     "OUTPUT_NORM_WEIGHT")
    write_at(weight_defs["OUTPUT_NORM_WEIGHT"], raw)

    # LM_HEAD is tied to the embedding in HF. Materialize a separately quantized
    # copy for the accelerator's output projection.
    lm_head_w = model.lm_head.weight.detach().cpu().to(torch.bfloat16)
    scale_sz = weight_defs["LM_HEAD_WEIGHT_SCALE_SIZE"]
    data_sz = weight_defs["LM_HEAD_WEIGHT_DATA_SIZE"]
    data_bytes, scale_bytes = _quantize_bf16_to_if4_packed(lm_head_w)
    scale_padded = pad_region(scale_bytes, scale_sz, "LM_HEAD_WEIGHT_SCALE")
    data_padded = pad_region(data_bytes, data_sz, "LM_HEAD_WEIGHT_DATA")
    write_at(weight_defs["LM_HEAD_WEIGHT_SCALE"], scale_padded)
    write_at(weight_defs["LM_HEAD_WEIGHT_DATA"], data_padded)

    # Package the final layer's BF16 projection weights in params.bin so this
    # runner does not need the full HF checkpoint after artifacts are built.
    last_layer = model.model.layers[num_layers - 1]
    last_layer_bf16 = {
        "LAST_LAYER_Q_WEIGHT_BF16": last_layer.self_attn.q_proj.weight,
        "LAST_LAYER_K_WEIGHT_BF16": last_layer.self_attn.k_proj.weight,
        "LAST_LAYER_V_WEIGHT_BF16": last_layer.self_attn.v_proj.weight,
        "LAST_LAYER_O_WEIGHT_BF16": last_layer.self_attn.o_proj.weight,
        "LAST_LAYER_GATE_WEIGHT_BF16": last_layer.mlp.gate_proj.weight,
        "LAST_LAYER_UP_WEIGHT_BF16": last_layer.mlp.up_proj.weight,
        "LAST_LAYER_DOWN_WEIGHT_BF16": last_layer.mlp.down_proj.weight,
    }
    for key, weight in last_layer_bf16.items():
        raw = weight.detach().cpu().to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()
        expected_size = weight_defs[f"{key}_SIZE"]
        if len(raw) != expected_size:
            raise ValueError(f"{key} has {len(raw)} bytes; config expects {expected_size}")
        write_at(weight_defs[key], raw)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(buf)
    meta_rel = paths.get("weights_meta", "qwen3_0.6b_bin/params.json")
    meta_path = os.path.join(script_dir, meta_rel)
    with open(meta_path, "w") as f:
        json.dump({
            "size": len(buf),
            "model_layout_sig": _model_layout_signature(cfg),
            "weight_fingerprint": _weight_artifact_fingerprint(cfg, script_dir),
        }, f)
    print(f"Generated weights bin: {out_path} ({len(buf)} bytes)")
    return out_path

def _ensure_hf_model(script_dir: str, cfg: dict):
    """Ensure HF model is downloaded and loaded. Returns (model, model_dir). Single place for download + load."""
    model_dir = os.path.join(script_dir, cfg["paths"]["hf_model_dir"])
    hf_repo = cfg["paths"]["hf_model_repo"]
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        _original_print(f"Downloading HF model {hf_repo} to {os.path.abspath(model_dir)} ...")
        # huggingface_hub 1.6 can race while finalizing large local-dir downloads
        # with multiple workers. A single worker is reliable and still resumes.
        snapshot_download(repo_id=hf_repo, local_dir=model_dir, max_workers=1)
        _original_print("Download complete.")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map=None, trust_remote_code=True
    )
    return model, model_dir

def _load_config(script_dir: str) -> dict:
    """Load qwen3_0.6b_config.json and build weight_defs (offset/size dict) from regions."""
    config_path = os.path.join(script_dir, "qwen3_0.6b_config.json")
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
# Qwen3-0.6B unified engine
# -----------------------------------------------------------------------------
class Qwen3_0_6b_UnifiedEngine(UnifiedEngine):
    """UnifiedEngine for Qwen3-0.6B: load offline artifacts and run prefill/decode.

    Key architectural differences from Gemma3:
      - QK RMSNorm per actual head dim (128), gamma_offset=0.0.
      - No post-attention norm: residual applied directly to o_proj output.
      - No post-FFN norm: residual applied directly to down_proj output.
      - post_attention_layernorm in HF IS the pre-FFN norm.
      - Embedding NOT scaled by sqrt(hidden_size).
      - Tied HF embedding/lm_head: exact BF16 embedding rows live in device DRAM,
        alongside a separately quantized accelerator LM head.
      - SwiGLU (SiLU gate): silu_enable=True on gate_proj.
      - Single RoPE theta=1_000_000 for all layers; rope applied per head (N=128).
      - Per-KV-head flash attention (8 KV heads x 128 dim); group_size=2.
    """

    def __init__(self, script_dir: str | None = None, hf_model_dir: str | None = None,
                 weights_bin: str | None = None, use_bf16_last_layer: bool = False):
        # Qwen3-0.6B DRAM layout in the 2 GiB accelerator window:
        #   params:       0x80000000 – 0xA0000000 (512 MB)
        #   tensors:      0xA0000000 – 0xE0000000 (KV cache + activations)
        #   programs:     0xE0000000 – 0xED740000 (bin, preamble, dispatch)
        #   embedding:    0xED740000 – 0x100000000 (exact BF16 rows)
        super().__init__(
            params_dram_base=0x80000000,
            tensor_dram_base=0xA0000000,
            program_dram_base=0xE0000000,
        )
        self.script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
        self.use_bf16_last_layer = bool(use_bf16_last_layer)
        self._cfg = _load_config(self.script_dir)
        self.weight_defs = self._cfg["_weight_defs"]

        fi = self._cfg["file_info"]
        model = self._cfg["model"]
        paths = self._cfg["paths"]

        self.vector_length = fi["hidden_size"]           # 1024
        self.head_dim = fi["head_dim"]                   # 1024 (= num_kv_heads * actual_head_dim)
        self.actual_head_dim = fi["actual_head_dim"]     # 128
        self.num_kv_heads = fi["num_kv_heads"]           # 8
        self.bytes_per_element = fi["bytes_per_element"] # 2
        self.group_size = fi["group_size"]               # 2 (Q heads per KV head)
        self.mlp_elements = fi["mlp_elements"]           # 3072
        self.hf_model_dir = hf_model_dir or os.path.join(self.script_dir, paths["hf_model_dir"])
        # q_size = total Q output dim * bpe = (num_kv_heads * actual_head_dim * group_size) * bpe
        self.q_size = self.head_dim * self.group_size * self.bytes_per_element   # 1024*2*2 = 4096
        self.k_size = self.head_dim * self.bytes_per_element                     # 1024*2 = 2048
        self.MAX_CONTEXT_SIZE = model["max_context_size"]
        self.PREFILL_MAX_SEQ_LEN = int(model.get("prefill_max_seq_len", 256))
        self.LAYER_SIZE = fi["num_layers"]
        self.EMBEDDING_ELEMENTS = fi["embedding_vocab"]
        (self.DRAM_ADDR_TOKEN_EMBEDDING,
         self.DEVICE_EMBEDDING_TABLE_SIZE) = _device_bf16_embedding_layout(self._cfg)
        self.EMBEDDING_ROW_BYTES = self.vector_length * self.bytes_per_element
        self.EMBEDDING_DISPATCH_STRIDE_BYTES = _EMBEDDING_DISPATCH_STRIDE_BYTES
        self.EMBEDDING_DISPATCH_TABLE_SIZE = (
            self.EMBEDDING_ELEMENTS * self.EMBEDDING_DISPATCH_STRIDE_BYTES)
        # Dynamic-PBI GPR layout:
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
        # attention/transpose kernel call site.

        bin_path = weights_bin or paths["weights_bin"]
        full_path = os.path.join(self.script_dir, bin_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Bin file not found: {full_path}")
        with open(full_path, "rb") as f:
            self.weight_bin = f.read()
        expected_size = _expected_weight_bin_size(self._cfg)
        if len(self.weight_bin) != expected_size:
            raise ValueError(
                f"Weight bin has {len(self.weight_bin)} bytes; config requires {expected_size}: {full_path}"
            )
        self.weight_init()
        self.tensor_init()

    # XDMA drivers/bitstreams commonly cap one os.write() well below the large
    # contiguous model regions. Chunk every large H2C transfer so failures are
    # reported immediately instead of silently leaving partial weights in DRAM.
    DMA_CHUNK_BYTES = 1 * 1024 * 1024

    def dma_write(self, device: str, address: int, buffer, size: int) -> int:
        if size <= self.DMA_CHUNK_BYTES:
            written = super().dma_write(device, address, buffer, size)
            if written != size:
                raise OSError(f"DMA write at 0x{address:X}: wrote {written} of {size} bytes")
            return written

        if isinstance(buffer, torch.Tensor):
            tensor = buffer.detach().cpu().contiguous()
            if tensor.dtype == torch.bfloat16:
                raw = tensor.view(torch.uint16).numpy().tobytes()[:size]
            else:
                raw = tensor.numpy().tobytes()[:size]
        elif isinstance(buffer, (bytes, bytearray, memoryview)):
            raw = memoryview(buffer).cast("B")[:size]
        else:
            raw = bytes(buffer)[:size]
        if len(raw) != size:
            raise ValueError(f"DMA source has {len(raw)} bytes; requested {size}")

        total = 0
        while total < size:
            chunk_size = min(self.DMA_CHUNK_BYTES, size - total)
            written = super().dma_write(
                device, address + total, raw[total : total + chunk_size], chunk_size,
            )
            if written != chunk_size:
                raise OSError(
                    f"DMA write at 0x{address + total:X}: wrote {written} of {chunk_size} bytes"
                )
            total += written
        return total

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
        """DMA the packaged final-layer BF16 projections into params DRAM.

        The seven tensors live in ``params.bin``, so this offline runner does
        not need to load or download the full Hugging Face checkpoint.
        """
        self._last_layer_bf16_addrs: dict[str, int] = {}
        if not self.use_bf16_last_layer:
            print(f"  Fast IF4 path selected for L{self.LAYER_SIZE-1}; "
                  "packaged BF16 fallback remains in params.bin")
            return
        regions = {
            "q": "LAST_LAYER_Q_WEIGHT_BF16",
            "k": "LAST_LAYER_K_WEIGHT_BF16",
            "v": "LAST_LAYER_V_WEIGHT_BF16",
            "o": "LAST_LAYER_O_WEIGHT_BF16",
            "gate": "LAST_LAYER_GATE_WEIGHT_BF16",
            "up": "LAST_LAYER_UP_WEIGHT_BF16",
            "down": "LAST_LAYER_DOWN_WEIGHT_BF16",
        }
        total_bytes = 0
        for name, key in regions.items():
            offset = self.weight_defs[key]
            sz = self.weight_defs[f"{key}_SIZE"]
            raw = self.weight_bin[offset : offset + sz]
            if len(raw) != sz:
                raise ValueError(f"params.bin is missing {key}: got {len(raw)} of {sz} bytes")
            addr = self.allocate_params_dram(sz)
            self.dma_write(DMA_DEVICE_H2C, addr, raw, sz)
            self._last_layer_bf16_addrs[name] = addr
            total_bytes += sz
        print(f"  Packaged BF16 L{self.LAYER_SIZE-1} weights DMA'd: "
              f"{total_bytes / 1024 / 1024:.1f} MB")

    def get_embedding_for_tokens(self, token_ids: list[int] | tuple) -> torch.Tensor:
        """Return (len(token_ids), vector_length) bfloat16 tensor from self.embedding_weight (no scaling)."""
        tid_t = torch.tensor(token_ids, dtype=torch.long)
        out = torch.zeros(len(token_ids), self.vector_length, dtype=torch.bfloat16)
        valid = tid_t < self.embedding_weight.shape[0]
        out[valid] = self.embedding_weight[tid_t[valid]]
        return out

    def get_device_embedding_row_addr(self, token_id: int) -> int:
        """Return the byte address of one exact BF16 embedding row in device DRAM."""
        token_id = int(token_id)
        if not 0 <= token_id < self.EMBEDDING_ELEMENTS:
            raise ValueError(
                f"token_id={token_id} is outside embedding vocabulary "
                f"[0, {self.EMBEDDING_ELEMENTS})."
            )
        return self.DRAM_ADDR_TOKEN_EMBEDDING + token_id * self.EMBEDDING_ROW_BYTES

    def get_embedding_dispatch_entry_addr(self, token_id: int) -> int:
        """Return the executable dispatch-entry address for one token id."""
        self.get_device_embedding_row_addr(token_id)  # validates the id
        if not hasattr(self, "embedding_dispatch_addr"):
            raise RuntimeError("Embedding dispatch table was not built before decode.")
        return (self.embedding_dispatch_addr
                + int(token_id) * self.EMBEDDING_DISPATCH_STRIDE_BYTES)

    def build_embedding_dispatch_table(self, preamble_addr: int) -> int:
        """Build one two-instruction, 64-byte dispatch entry per token.

        The host launches the entry selected by the latest token id. The entry
        places that token's exact BF16 row word address in ``TMP_REG`` and jumps
        through the bucket-cached decoder preamble. This removes both embedding
        H2C DMA and instruction H2C DMA from the token-to-token hot path.
        """
        dispatch_addr = self.get_program_dram_addr()
        dispatch_end = dispatch_addr + self.EMBEDDING_DISPATCH_TABLE_SIZE
        if preamble_addr % 64 or dispatch_addr % 64:
            raise ValueError("Runtime preamble and embedding dispatch must be 64-byte aligned.")
        if dispatch_end > self.DRAM_ADDR_TOKEN_EMBEDDING:
            raise ValueError(
                f"Embedding dispatch ends at 0x{dispatch_end:X}, overlapping "
                f"the BF16 embedding table at 0x{self.DRAM_ADDR_TOKEN_EMBEDDING:X}."
            )

        print(f"  Building {self.EMBEDDING_ELEMENTS:,}-entry embedding dispatch "
              f"({self.EMBEDDING_DISPATCH_TABLE_SIZE / 2**20:.2f} MiB)...", flush=True)
        global _SILENT_MODE
        previous_silent_mode = _SILENT_MODE
        _SILENT_MODE = True
        try:
            self.start_capture()
            for token_id in range(self.EMBEDDING_ELEMENTS):
                entry_start = self.capture_count
                self.clear_inst_id()
                self.generate_instruction_add_set(
                    self.TMP_REG,
                    ue_35bit_addr_shifter(self.get_device_embedding_row_addr(token_id)),
                )
                self.generate_instruction_jump_abs(ue_35bit_addr_shifter(preamble_addr))
                entry_size = self.capture_count - entry_start
                if entry_size != 2:
                    raise RuntimeError(
                        f"Embedding dispatch entry {token_id} has {entry_size} instructions; "
                        "expected exactly 2."
                    )
            self.stop_capture()
            dispatch_size = self.get_capture_instruction_size_bytes()
            if dispatch_size != self.EMBEDDING_DISPATCH_TABLE_SIZE:
                raise RuntimeError(
                    f"Embedding dispatch is {dispatch_size} bytes; expected "
                    f"{self.EMBEDDING_DISPATCH_TABLE_SIZE}."
                )
            written = self.write_captured_instructions_to_dram(dispatch_addr)
            if written != dispatch_size:
                raise RuntimeError(
                    f"Embedding dispatch DMA wrote {written} of {dispatch_size} bytes."
                )
            allocated_addr = self.allocate_program_dram(dispatch_size)
            if allocated_addr != dispatch_addr:
                raise RuntimeError(
                    f"Embedding dispatch allocation moved from 0x{dispatch_addr:X} "
                    f"to 0x{allocated_addr:X}."
                )
        finally:
            if self.is_capture_on:
                self.stop_capture()
            self.clear_capture_buffer()
            _SILENT_MODE = previous_silent_mode

        self.embedding_dispatch_addr = dispatch_addr
        print(f"  Embedding dispatch ready at 0x{dispatch_addr:X}")
        return dispatch_addr

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
        raw_emb = memoryview(self.weight_bin)[token_embd_offset : token_embd_offset + emb_bytes]
        if len(raw_emb) != self.DEVICE_EMBEDDING_TABLE_SIZE:
            raise ValueError(
                f"BF16 embedding slice has {len(raw_emb)} bytes; "
                f"expected {self.DEVICE_EMBEDDING_TABLE_SIZE}."
            )
        print(f"  DMA exact BF16 embedding: {emb_bytes / 1024 / 1024:.2f} MB "
              f"→ DRAM 0x{self.DRAM_ADDR_TOKEN_EMBEDDING:X}...", flush=True)
        _embedding_dma_start = time.perf_counter()
        self.dma_write(
            DMA_DEVICE_H2C, self.DRAM_ADDR_TOKEN_EMBEDDING, raw_emb, emb_bytes)
        print(f"  BF16 embedding DMA done in "
              f"{time.perf_counter() - _embedding_dma_start:.1f}s")
        # Prefill still gathers prompt rows on the host. Keep one mutable copy
        # backing the torch view, but avoid another 296.75-MiB clone at startup.
        self._embedding_host_storage = bytearray(raw_emb)
        self.embedding_weight = torch.frombuffer(
            self._embedding_host_storage, dtype=torch.bfloat16).reshape(vocab_size, emb_dim)
        del raw_emb
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

        # Load the packaged final-layer BF16 matmul weights at the same addresses
        # used by the builder when it compiled the instruction image.
        self._load_last_layer_bf16()

        # Every artifact slice needed at runtime has now been copied to host
        # tensors or device DRAM. Drop the 645 MiB source blob before allocating
        # activation tensors.
        self.weight_bin = b""

        print(f"    Allocate weights end at DRAM address: 0x{self.get_params_dram_addr():X}, usage: {self.get_params_dram_usage()} bytes")
        print("Tokenizer loaded successfully.")

    def tensor_init(self) -> None:
        """Initialize hardware DRAM tensors for Qwen3-0.6B.

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
        zero_add = torch.zeros(seq_len * self.head_dim, dtype=torch.bfloat16)
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
        zero_pad = torch.zeros(aligned_seq_len * ahd, dtype=torch.bfloat16)
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
        # K=3072 fits in SCALE_BRAM directly — no split buffers needed
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
        zero_kv = torch.zeros(kv_cache_total // bpe, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, zero_kv)
        self.dma_to_accelerator_memory(self.LAYER0_K_ROPE_DRAM, zero_kv)

        # On-FPGA repetition penalty bias (LM-head matmul C term). MUST be allocated LAST,
        # exactly as in qwen3_0.6b_test.py, so its address matches the one baked into the bin.
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
        qpkv = self.group_size

        # DMA inputs: embedding (actual_seq_len rows) and bias mask.
        embedding_tensor = self.get_embedding_for_tokens(prefill_seq)
        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
        bias_one_group = torch.full((aligned_seq_len, aligned_seq_len), float("-inf"), dtype=torch.bfloat16)
        # FLASH_K/V repeat every token once per Q head. Expand causality at the
        # token level so all heads for token t can attend every duplicate of
        # tokens <= t (a scalar-row tril would hide part of the current token).
        token_rows = torch.ones(actual_seq_len, actual_seq_len, dtype=torch.bool)
        token_mask = (torch.triu(token_rows, diagonal=0) if self.causal_mask_upper
                      else torch.tril(token_rows, diagonal=0))
        valid_mask = token_mask.repeat_interleave(qpkv, dim=0).repeat_interleave(qpkv, dim=1)
        bias_one_group[:q_seq_len, :q_seq_len].masked_fill_(valid_mask, 0.0)
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

        The host starts the 64-byte dispatch entry selected by ``token_id``. That
        entry primes ``TMP_REG`` with the exact BF16 embedding-row word address
        and jumps through the bucket-cached preamble at ``preamble_addr``. The
        preamble primes gpr_aligned_seq_len and jumps into the cached decoder.
        gpr_seq_len carries across steps via the in-bin ADD_INC.

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
        max_new_tokens = int(getattr(self, "max_new_tokens", 0))
        decoded_new_tokens = 0

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

        preamble_aligned_ctx = None
        decoder_hw_latency_us = 0.0
        while self.seq_len < max_seq_len:
            _SILENT_MODE = True
            # self.seq_len at entry is the count of K/V already in cache; the
            # current decode token will be written at that index.
            decode_pos = self.seq_len               # K/V cache write index for this token
            new_ctx_len = decode_pos + 1            # KV positions [0..decode_pos] inclusive
            aligned_ctx = ((new_ctx_len + 63) // 64) * 64
            # unified_attention_core uses full-matrix bias, one row per Q head in
            # the KV group. Mask positions past the live context so stale cache rows
            # never enter softmax.
            bias_host = torch.full((1, aligned_ctx), float("-inf"), dtype=torch.bfloat16)
            bias_host[0, :new_ctx_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_host.repeat(self.group_size, 1))

            # The two-instruction preamble depends only on the 64-token attention
            # bucket. The token-specific BF16 row address comes from the prebuilt
            # dispatch entry, so this H2C write remains once per bucket.
            if aligned_ctx != preamble_aligned_ctx:
                self.clear_inst_id()
                self.start_capture()
                self.generate_instruction_add_set(self.gpr_aligned_seq_len, aligned_ctx)
                self.generate_instruction_jump_abs(ue_35bit_addr_shifter(decoder_program_addr))
                self.stop_capture()
                self.write_captured_instructions_to_dram(preamble_addr)
                self.clear_capture_buffer()
                preamble_aligned_ctx = aligned_ctx

            # Refresh the per-vocab penalty bias (this step's LM-head C term) once past the
            # gate, BEFORE execute, so the HW argmax of (logits + bias) is already penalized.
            if _fpga_penalty and (len(self._generated_tokens) - _prompt_len) >= _greedy_until:
                self._write_penalty_bias(self._generated_tokens)

            dispatch_entry_addr = self.get_embedding_dispatch_entry_addr(token_id)
            step_latency_us, _ = self.program_execute(
                dispatch_entry_addr, timeout=10.0, flops=gflops_per_token)
            decoder_hw_latency_us += float(step_latency_us)
            # Read the HW argmax register — the LM-head matmul already added the penalty bias
            # on chip, so this is the penalized (or pure-greedy) token. No host logit readback.
            token_id = self.get_arg_max_index()
            self._generated_tokens.append(token_id)
            token_char = self.tokenizer.decode([token_id])
            _SILENT_MODE = False
            self.seq_len += 1
            decoded_new_tokens += 1
            if token_id in _qwen3_stop_tokens:
                if _use_status:
                    _status_teardown()
                print(f"\nStop token {token_id} reached.")
                break
            print(token_char, end="", flush=True)
            if max_new_tokens > 0 and decoded_new_tokens >= max_new_tokens:
                if _use_status:
                    _status_teardown()
                print(f"\nMaximum new-token limit ({max_new_tokens}) reached.")
                break
            if _use_status:
                _status_update()
        else:
            if _use_status:
                _status_teardown()
        self.last_decoder_hw_latency_us = decoder_hw_latency_us
        self.last_decoder_tokens = decoded_new_tokens
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
    parser = argparse.ArgumentParser(description="Qwen3-0.6B inference from pre-compiled bins (offline)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt (default: from qwen3_0.6b_config.json default_prompt)")
    parser.add_argument("--local-weights", action="store_true",
                        help="Use qwen3_0.6b_bin/full_model_weights.bin instead of params.bin")
    parser.add_argument("--dev", type=str, default="xdma0",
                        help="DMA device name (default: xdma0)")
    parser.add_argument("--cycle", type=float, default=5.62,
                        help="Clock cycle time in ns (default: 5.62ns ≈ peak 22.8 GFLOPS)")
    parser.add_argument("--device", type=str, default="kintex7",
                        help="FPGA board profile (kintex7, rk, puzhi, bittware, bittware_256, alveo, efinix).")
    # Deterministic on-FPGA decode: token selection is always the HW argmax of
    # (logits + penalty bias). No host sampling — the repetition penalty is folded into
    # the LM-head matmul bias.
    parser.add_argument('--pure-greedy', action='store_true',
                        help='Disable the on-FPGA repetition penalty — plain greedy '
                             '(bias stays all-zero). Penalty is ENABLED by default.')
    parser.add_argument('--max-new-tokens', type=int, default=0,
                        help='Stop after N decoded tokens. Default 0 means decode until a stop token '
                             'or the context limit.')
    parser.add_argument('--bf16-last-layer', action='store_true',
                        help='Use the packaged BF16 projections for transformer layer 27. '
                             'The default fused IF4 path is faster; this flag keeps the '
                             'previous quality-first execution mode.')
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
    cfg_on_disk = _load_config(script_dir)
    paths_cfg = cfg_on_disk["paths"]

    weights_bin_rel = (paths_cfg["local_weights_bin"] if args.local_weights
                       else paths_cfg["weights_bin"])
    weights_bin_full = os.path.join(script_dir, weights_bin_rel)
    params_meta_path = os.path.join(script_dir, paths_cfg["weights_meta"])
    inst_bin_path = os.path.join(script_dir, paths_cfg["instruction_bin"])
    meta_path = os.path.join(script_dir, paths_cfg["instruction_meta"])
    tokenizer_dir = os.path.join(script_dir, paths_cfg["hf_model_dir"])

    # Hard-fail BEFORE any FPGA / HF touch if a required local file is missing.
    missing = []
    if not os.path.exists(weights_bin_full):
        missing.append(os.path.relpath(weights_bin_full, script_dir))
    if not args.local_weights and not os.path.exists(params_meta_path):
        missing.append(os.path.relpath(params_meta_path, script_dir))
    for artifact_path in (inst_bin_path, meta_path):
        if not os.path.exists(artifact_path):
            missing.append(os.path.relpath(artifact_path, script_dir))
    if not (os.path.exists(os.path.join(tokenizer_dir, "tokenizer.json")) or
            os.path.exists(os.path.join(tokenizer_dir, "tokenizer_config.json"))):
        missing.append(
            f"{os.path.relpath(tokenizer_dir, script_dir)}/"
            "{tokenizer.json,tokenizer_config.json}"
        )
    if missing:
        _original_print("Missing local files (run qwen3_0.6b_test.py first on a build machine with HF access):")
        for f in missing:
            _original_print(f"  {f}")
        sys.exit(1)

    # Validate weight metadata before touching the FPGA. This catches truncated
    # or layout-incompatible artifacts with a useful rebuild instruction.
    actual_weight_size = os.path.getsize(weights_bin_full)
    expected_weight_size = _expected_weight_bin_size(cfg_on_disk)
    if args.local_weights:
        if actual_weight_size != expected_weight_size:
            raise SystemExit(
                f"{weights_bin_rel} has {actual_weight_size} bytes; "
                f"qwen3_0.6b_config.json requires {expected_weight_size}."
            )
    else:
        with open(params_meta_path) as f:
            params_meta = json.load(f)
        expected_layout_sig = _model_layout_signature(cfg_on_disk)
        if (params_meta.get("size") != actual_weight_size
                or actual_weight_size != expected_weight_size
                or params_meta.get("model_layout_sig") != expected_layout_sig
                or params_meta.get("weight_fingerprint")
                    != _weight_artifact_fingerprint(cfg_on_disk, script_dir)):
            raise SystemExit(
                "params.bin/params.json do not match qwen3_0.6b_config.json; "
                "rerun qwen3_0.6b_test.py to rebuild them."
            )

    with open(meta_path) as f:
        inst_meta = json.load(f)
    expected_layout_sig = _model_layout_signature(cfg_on_disk)
    expected_last_layer_sig = "bf16_l27" if args.bf16_last_layer else "if4_l27"
    expected_lm_head_sig = (
        f"fused_lm_shared_rope_qscale_bf16_embed_dispatch_v3:{expected_last_layer_sig}")
    expected_compiler_fingerprint = _instruction_compiler_fingerprint(
        script_dir, expected_lm_head_sig)
    expected_embedding_base, expected_embedding_size = _device_bf16_embedding_layout(cfg_on_disk)
    expected_dispatch_size = (
        int(cfg_on_disk["file_info"]["embedding_vocab"])
        * _EMBEDDING_DISPATCH_STRIDE_BYTES)
    if (inst_meta.get("lm_head_sig") != expected_lm_head_sig
            or inst_meta.get("model_layout_sig") != expected_layout_sig
            or inst_meta.get("compiler_fingerprint") != expected_compiler_fingerprint
            or inst_meta.get("instruction_base_addr") != f"0x{_PROGRAM_DRAM_BASE:X}"
            or inst_meta.get("instruction_total_size") != os.path.getsize(inst_bin_path)
            or inst_meta.get("prefill_program_start_addr") != f"0x{_PROGRAM_DRAM_BASE:X}"
            or _parse_offset(inst_meta.get("decoder_program_start_addr", -1))
                != _PROGRAM_DRAM_BASE + int(inst_meta.get("prefill_program_size", -1))
            or (int(inst_meta.get("prefill_program_size", -1))
                + int(inst_meta.get("decoder_program_size", -1)))
                != os.path.getsize(inst_bin_path)
            or inst_meta.get("device_embedding_base") != f"0x{expected_embedding_base:X}"
            or inst_meta.get("device_embedding_size") != expected_embedding_size
            or inst_meta.get("runtime_preamble_size") != _RUNTIME_PREAMBLE_BYTES
            or inst_meta.get("embedding_dispatch_stride")
                != _EMBEDDING_DISPATCH_STRIDE_BYTES
            or inst_meta.get("embedding_dispatch_size") != expected_dispatch_size):
        raise SystemExit(
            "programs.bin/programs.json do not match qwen3_0.6b_config.json; "
            "rerun qwen3_0.6b_test.py to recompile them."
        )
    try:
        _validate_program_embedding_separation(
            os.path.getsize(inst_bin_path), expected_embedding_base, expected_dispatch_size)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    set_dma_device("efinix" if args.device == "efinix" else args.dev)
    # Mirror test.py: rebind the device-name module globals after set_dma_device
    # so sample_next_token's dma_read(DMA_DEVICE_C2H, ...) resolves (and tracks
    # the chosen --dev). Without this, DMA_DEVICE_C2H is undefined in this module.
    global DMA_DEVICE_H2C, DMA_DEVICE_C2H, DMA_DEVICE_USER
    DMA_DEVICE_H2C = user_dma_core.DMA_DEVICE_H2C
    DMA_DEVICE_C2H = user_dma_core.DMA_DEVICE_C2H
    DMA_DEVICE_USER = user_dma_core.DMA_DEVICE_USER
    clock = 4.0 if args.device == "efinix" and args.cycle == 5.62 else args.cycle
    user_dma_core.CLOCK_CYCLE_TIME_NS = clock
    user_dma_core.UE_PEAK_GFLOPS = 0.128 / clock
    _original_print(f"Setting CLOCK_CYCLE_TIME_NS = {user_dma_core.CLOCK_CYCLE_TIME_NS}, "
                    f"UE_PEAK_GFLOPS = {user_dma_core.UE_PEAK_GFLOPS:.4f}")

    # Clear stale queue/PBI state before model weights are loaded. The reset's
    # DRAM self-test touches the params base, so it must happen before creating
    # Qwen3_0_6b_UnifiedEngine (same boot sequence as Gemma/Llama runners).
    boot_engine = user_dma_core.UnifiedEngine(BASE_ADDR=user_dma_core.UE_0_BASE_ADDR)
    boot_engine.software_reset()
    del boot_engine

    global _SILENT_MODE
    _SILENT_MODE = True
    ue = Qwen3_0_6b_UnifiedEngine(
        script_dir=script_dir,
        weights_bin=weights_bin_rel,
        use_bf16_last_layer=args.bf16_last_layer,
    )
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
    ue.max_new_tokens = max(0, int(args.max_new_tokens))
    ue._generated_tokens = list(prefill_seq)   # seed the penalty window with the prompt
    if ue.fpga_penalty:
        _original_print(f"On-FPGA penalty: alpha={ue.pen_alpha} cap={ue.pen_cap} "
                        f"rep_window={ue.rep_window} greedy_until={ue.greedy_until}")
    else:
        _original_print("Pure greedy (on-FPGA penalty disabled).")

    # Metadata was validated before device setup; no compile machinery is used.
    _original_print(f"  Loaded compile meta from {os.path.relpath(meta_path, script_dir)}")

    _original_print(f"\n--- Loading unified instruction bin ---")
    timer = time.perf_counter()
    base_addr, total_size = ue.load_program_instructions_from_file(inst_bin_path)
    if base_addr != _PROGRAM_DRAM_BASE:
        raise RuntimeError(
            f"Program loaded at 0x{base_addr:X}; expected 0x{_PROGRAM_DRAM_BASE:X}."
        )
    preamble_addr = ue.get_program_dram_addr()
    ue.allocate_program_dram(_RUNTIME_PREAMBLE_BYTES)
    ue.build_embedding_dispatch_table(preamble_addr)
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
    if ue.last_decoder_hw_latency_us > 0:
        _original_print(f"Decoder hardware: {ue.last_decoder_hw_latency_us / 1000:.2f} ms total, "
                        f"{ue.last_decoder_hw_latency_us / 1000 / decoded_tokens:.2f} ms/token, "
                        f"{decoded_tokens * 1e6 / ue.last_decoder_hw_latency_us:.2f} tokens/s.")


if __name__ == "__main__":
    main()
