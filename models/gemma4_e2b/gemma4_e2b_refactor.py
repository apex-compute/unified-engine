#!/usr/bin/env python3
"""
Gemma4 E2B LM inference on the Apex accelerator: prefill + decode.

Pure LM path, refactored out of gemma4_e2b_test.py into a clean, single-purpose
module. The vision and audio encoders are intentionally NOT here yet -- they
will be layered back on top of this base once each is cleaned up. For the
current combined LM + vision + audio implementation see gemma4_e2b_test.py.

  - Config from gemma4_e2b_config.json; weights from a single combined bin
    (gemma4_e2b_bin/params.bin), generated from the HF model if missing.
  - compile_gemma4() captures the prefill (at the real prompt seq_len) and the
    decoder into one combined image on disk (gemma4_e2b_bin/programs.bin);
    run_gemma4() loads it into program DRAM and dispatches prefill, then the
    decode loop, through a shared preamble (gemma3-style compile → run split).

Usage:
  python gemma4_e2b_refactor.py
  python gemma4_e2b_refactor.py --prompt "your prompt"
  python gemma4_e2b_refactor.py --dev xdma0 [--cycle 5.62]
  python gemma4_e2b_refactor.py --local-weights

Fixed layout: this script, gemma4_e2b_config.json, and gemma4_e2b_bin/ live in
the same folder; user_dma_core.py is two folders up (repo root), added to
sys.path.
"""

import json
import math
import os
import sys

# This file's folder: gemma4_e2b_bin/ and *.json live here. user_dma_core is
# two levels up (repo root).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

# We run on FPGA + CPU only; disable CUDA before importing torch so PyTorch
# doesn't probe the GPU driver (avoids a noisy "Error 804" warning on hosts
# whose CUDA driver/runtime doesn't match the installed GPU).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoTokenizer
from huggingface_hub import snapshot_download
import time
import user_dma_core
from user_dma_core import DMA_DEVICE_H2C, DRAM_INSTRUCTION_ADDR, TYPE, UE_FMAX_CONTEXT_SIZE, UE_VECTOR_SIZE, UE_ARGMAX_INDEX, URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS, URAM_NEAR_FULL_SIZE, URAM_START_ADDR, URAM_SECTION, set_dma_device
from user_dma_core import UnifiedEngine
from user_dma_core import ue_35bit_addr_shifter
from user_dma_core import INSTRUCTION_SIZE_BYTES
from user_dma_core import UE_MODE

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

def _parse_offset(val) -> int:
    """Parse offset/size from JSON: int or hex string like '0x24000000'."""
    if isinstance(val, str):
        return int(val, 0)
    return int(val)

def weight_bin_generate(output_path: str | None = None, config_path: str | None = None) -> str:
    """Generate params.bin from Hugging Face model per gemma4_e2b_config.json layout.
    Returns the path to the written file."""
    # Quantization setup (LM matmuls use IF4). The canonical INT4/FP4/IF4 codec
    # lives in quant_lib.quantize; it's pure-CPU and stateless, so many tensors
    # quantize concurrently via a thread pool (torch ops release the GIL; a
    # thread pool avoids the multi-GB pickle cost of a process pool).
    from quant_lib import quantize as _qs_quantize
    from concurrent.futures import ThreadPoolExecutor as _QuantPool
    _QUANT_WORKERS = max(1, (os.cpu_count() or 4) - 1)
    LM_QUANT_PRECISION = "if4"

    def _parallel_quantize(precision, tensors, block_size=64):
        """Quantize a list of [N, K] bf16 tensors in parallel; returns a parallel
        list of (data_bytes, scale_bytes). Order is preserved."""
        if len(tensors) == 0:
            return []
        if len(tensors) == 1:
            return [_qs_quantize(precision, tensors[0], block_size=block_size)]
        with _QuantPool(max_workers=_QUANT_WORKERS) as ex:
            return list(ex.map(
                lambda t: _qs_quantize(precision, t, block_size=block_size),
                tensors,
            ))

    def _ensure_hf_model(script_dir: str, cfg: dict):
        """Ensure HF model is downloaded and loaded. Returns (model, model_dir). Single place for download + load."""
        model_dir = os.path.join(script_dir, cfg["paths"]["hf_model_dir"])
        hf_repo = cfg["paths"]["hf_model_repo"]
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            _original_print(f"Downloading HF model {hf_repo} to {os.path.abspath(model_dir)} ...")
            snapshot_download(repo_id=hf_repo, local_dir=model_dir)
            _original_print("Download complete.")
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map=None, trust_remote_code=True
        )
        return model, model_dir

    def _build_host_section_bytes(text_model, cfg) -> tuple[bytes, dict]:
        """Build host-side tensor section bytes + manifest. The few tensors
        `_compute_per_layer_inputs` needs (per-layer embedding table, projection,
        projection norm) are concatenated as bf16 bytes; per-layer scalars and
        the KV-shared-layer map travel in the manifest dict.

        The big tensor (`embed_tokens_per_layer`, ~4.5 GB bf16) is laid down
        first so the run-time loader can mmap exactly its offset and pull rows
        on demand — RSS stays low even on a 16 GB Pi.
        """
        file_info = cfg["file_info"]
        num_layers = file_info["num_layers"]
        per_layer_input_dim = file_info["per_layer_input_dim"]

        # 1. per_layer_embed_tokens, pre-scaled by sqrt(per_layer_input_dim).
        # Chunked scale-and-cast so we don't materialize the full fp32 tensor
        # (would be 9.4 GB for vocab=262144, dim=8960 — blows past 16 GB).
        src = text_model.embed_tokens_per_layer.weight.detach().cpu().to(torch.bfloat16)
        per_layer_embed_scale = per_layer_input_dim ** 0.5
        embed_bf16 = torch.empty_like(src)
        chunk = 8192
        for i in range(0, src.shape[0], chunk):
            embed_bf16[i:i+chunk] = (src[i:i+chunk].float() * per_layer_embed_scale).to(torch.bfloat16)
        del src

        # 2. per_layer_model_proj_weight  [8960, 1536]
        proj_bf16 = text_model.per_layer_model_projection.weight.detach().cpu().to(torch.bfloat16).contiguous()
        # 3. per_layer_proj_norm_weight   [256]  (raw, no gamma_offset — host-side norm wants raw w)
        norm_bf16 = text_model.per_layer_projection_norm.weight.detach().cpu().to(torch.bfloat16).contiguous()

        # 4. Scalars + KV-shared map.
        # transformers' Gemma4TextAttention has no `kv_shared_layer_index` attribute
        # (only `layer_type` and `is_kv_shared_layer`) — at runtime a shared layer
        # reads `shared_kv_states[self.layer_type]`, populated by whichever earlier
        # (non-shared) layer is the LAST occurrence of that layer_type. Reproduce
        # that mapping here: for each shared layer, the reference is the last
        # non-shared layer with the same layer_type.
        layer_scalars = []
        kv_shared_map: dict[int, int] = {}
        last_layer_by_type: dict[str, int] = {}
        for layer_idx in range(num_layers):
            layer = text_model.layers[layer_idx]
            layer_scalars.append(float(layer.layer_scalar.item()))
            attn = layer.self_attn
            if attn.is_kv_shared_layer:
                kv_shared_map[layer_idx] = last_layer_by_type[attn.layer_type]
            else:
                last_layer_by_type[attn.layer_type] = layer_idx

        embed_b = embed_bf16.contiguous().view(torch.uint8).numpy().tobytes()
        proj_b  = proj_bf16.view(torch.uint8).numpy().tobytes()
        norm_b  = norm_bf16.view(torch.uint8).numpy().tobytes()

        embed_off = 0
        proj_off  = embed_off + len(embed_b)
        norm_off  = proj_off  + len(proj_b)
        total     = norm_off  + len(norm_b)

        out = embed_b + proj_b + norm_b
        manifest = {
            "embed_tokens_per_layer": {"offset": embed_off, "size": len(embed_b), "shape": list(embed_bf16.shape)},
            "per_layer_model_proj":   {"offset": proj_off,  "size": len(proj_b),  "shape": list(proj_bf16.shape)},
            "per_layer_proj_norm":    {"offset": norm_off,  "size": len(norm_b),  "shape": list(norm_bf16.shape)},
            "layer_scalars": layer_scalars,
            # JSON keys must be strings — convert int → str.
            "kv_shared_map": {str(k): v for k, v in kv_shared_map.items()},
        }
        print(f"  Host section: {total/1024**3:.2f} GiB, 3 tensors + scalars + kv_shared_map")
        return out, manifest

    def tokenizer_subset_extract(bin_dir: str, cfg: dict) -> str:
        """Copy the minimal tokenizer / processor files needed by
        gemma4_e2b_run_from_bin.py into `<bin_dir>/tokenizer/`. The full HF
        model directory (with the multi-GB .safetensors checkpoint) is not
        required at run time once this subset is in place.

        Returns the destination directory path.
        """
        import shutil
        hf_dir = os.path.join(SCRIPT_DIR, cfg["paths"]["hf_model_dir"])
        dst    = os.path.join(bin_dir, "tokenizer")
        os.makedirs(dst, exist_ok=True)
        # Required for AutoTokenizer (text) and AutoProcessor (image / audio).
        # Everything else (model.safetensors, config.json, .gitattributes, etc.)
        # is intentionally skipped.
        wanted = [
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "special_tokens_map.json",
            "processor_config.json",
        ]
        copied = []
        for name in wanted:
            src = os.path.join(hf_dir, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dst, name))
                copied.append(name)
        print(f"Bundled tokenizer subset to {dst} ({len(copied)} files: {', '.join(copied)})")
        return dst

    cfg = Gemma4_UnifiedEngine.load_config(config_path=config_path, script_dir=SCRIPT_DIR)
    weight_defs = cfg["_weight_defs"]
    paths = cfg["paths"]
    paths_full = os.path.join(SCRIPT_DIR, paths["weights_bin"])
    out_path = output_path or paths_full

    model, model_dir = _ensure_hf_model(SCRIPT_DIR, cfg)
    text_model = model.model.language_model
    gamma_offset = cfg["special"]["rms_norm"]["gamma_offset"]
    emb_cfg = cfg["special"]["embedding"]
    token_embd_offset = _parse_offset(emb_cfg["token_embd_offset"])
    token_embd_size = _parse_offset(emb_cfg["token_embd_size"])
    LAYER_WEIGHT_SIZE = weight_defs["LAYER_WEIGHT_SIZE"]
    base_layer0 = weight_defs["BLK0_ATTN_NORM_WEIGHT"]
    num_layers = cfg["file_info"]["num_layers"]
    head_dim = cfg["file_info"]["head_dim"]  # 512 (max / full attention)
    head_dim_sliding = cfg["file_info"]["head_dim_sliding"]  # 256
    hidden_size = cfg["file_info"]["hidden_size"]
    group_size = cfg["file_info"]["group_size"]
    full_attention_layers = set(cfg["model"]["full_attention_layers"])
    mlp_elements_wide = cfg["file_info"].get("mlp_elements_wide", cfg["file_info"]["mlp_elements"])
    blk0_structure = cfg["layers"]["structure"]

    # Compute total file size: max(offset + size) over all regions (layer regions use last layer)
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

    # Embedding: scale by sqrt(hidden_size)
    embed = text_model.embed_tokens.weight.detach().cpu().to(torch.bfloat16)
    embedding_scale = hidden_size ** 0.5
    emb_scaled = (embed.float() * embedding_scale).to(torch.bfloat16)
    raw_emb = emb_scaled.contiguous().view(torch.uint8).numpy().tobytes()
    write_at(token_embd_offset, raw_emb)

    # Layers
    for layer_idx in range(num_layers):
        layer = text_model.layers[layer_idx]
        attn = layer.self_attn
        is_full = layer_idx in full_attention_layers
        cur_head_dim = head_dim if is_full else head_dim_sliding
        cur_q_size = cur_head_dim * group_size
        cur_k_size = cur_head_dim

        gamma_in = (layer.input_layernorm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)

        # Q/K/V weights: actual sizes differ per layer, zero-pad to max (full attention) sizes.
        # KV-shared layers (attn.is_kv_shared_layer) have no k_proj/v_proj/k_norm at all —
        # they read another layer's KV cache at runtime (see _build_host_section_bytes's
        # kv_shared_map). Write zero/neutral placeholders into their (unused) K/V weight
        # region so the fixed per-layer layout and quantization pipeline stay uniform;
        # compile_prefill/compile_decoder never read these for shared layers.
        is_kv_shared = attn.is_kv_shared_layer
        q_w_actual = attn.q_proj.weight.detach().cpu().to(torch.bfloat16)  # [cur_q_size, hidden_size]
        if is_kv_shared:
            k_w_actual = torch.zeros(cur_k_size, hidden_size, dtype=torch.bfloat16)
            v_w_actual = torch.zeros(cur_k_size, hidden_size, dtype=torch.bfloat16)
        else:
            k_w_actual = attn.k_proj.weight.detach().cpu().to(torch.bfloat16)  # [cur_k_size, hidden_size]
            v_w_actual = attn.v_proj.weight.detach().cpu().to(torch.bfloat16)  # [cur_k_size, hidden_size]
        o_w_actual = attn.o_proj.weight.detach().cpu().to(torch.bfloat16)  # [hidden_size, cur_q_size]

        # Pad Q/K/V rows to max sizes (N dimension padding — contiguous rows, safe for sub-N matmul).
        # O weight: do NOT pad K dimension — quantize with actual K so scale/data blocks align correctly.
        max_q_size = head_dim * group_size  # 4096
        max_k_size = head_dim  # 512
        q_w = torch.zeros(max_q_size, hidden_size, dtype=torch.bfloat16)
        q_w[:cur_q_size, :] = q_w_actual
        k_w = torch.zeros(max_k_size, hidden_size, dtype=torch.bfloat16)
        k_w[:cur_k_size, :] = k_w_actual
        v_w = torch.zeros(max_k_size, hidden_size, dtype=torch.bfloat16)
        v_w[:cur_k_size, :] = v_w_actual
        # O weight: use actual dimensions (no column padding) to keep INT4 scale blocks aligned
        o_w = o_w_actual  # [hidden_size, cur_q_size]

        # Q/K norm: pad to max head_dim. KV-shared layers have no k_norm either.
        gamma_q_actual = (attn.q_norm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
        gamma_q = torch.ones(head_dim, dtype=torch.bfloat16)  # default 1.0 (gamma_offset already applied)
        gamma_q[:cur_head_dim] = gamma_q_actual[:cur_head_dim]
        gamma_k = torch.ones(head_dim, dtype=torch.bfloat16)
        if not is_kv_shared:
            gamma_k_actual = (attn.k_norm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
            gamma_k[:cur_head_dim] = gamma_k_actual

        gamma_post = (layer.post_attention_layernorm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
        gamma_ffn = (layer.pre_feedforward_layernorm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
        gate_w_actual = layer.mlp.gate_proj.weight.detach().cpu().to(torch.bfloat16)
        up_w_actual = layer.mlp.up_proj.weight.detach().cpu().to(torch.bfloat16)
        down_w = layer.mlp.down_proj.weight.detach().cpu().to(torch.bfloat16)
        # Pad gate/up rows to max MLP width (N dimension padding, safe for sub-N matmul)
        cur_mlp = gate_w_actual.shape[0]
        gate_w = torch.zeros(mlp_elements_wide, hidden_size, dtype=torch.bfloat16)
        gate_w[:cur_mlp, :] = gate_w_actual
        up_w = torch.zeros(mlp_elements_wide, hidden_size, dtype=torch.bfloat16)
        up_w[:cur_mlp, :] = up_w_actual
        # Down weight: use actual K (no padding) — quantize as-is so scale/data blocks align
        gamma_post_ffn = (layer.post_feedforward_layernorm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)

        # Per-layer weights (BF16, no quantization)
        per_layer_input_gate_w = layer.per_layer_input_gate.weight.detach().cpu().to(torch.bfloat16)  # [256, 1536]
        per_layer_projection_w = layer.per_layer_projection.weight.detach().cpu().to(torch.bfloat16)  # [1536, 256]
        gamma_post_per_layer_input_norm = (layer.post_per_layer_input_norm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)  # [1536]
        layer_scalar_val = layer.layer_scalar.detach().cpu().to(torch.bfloat16)  # scalar
        # Pad layer_scalar to 64 bytes (32 bf16 elements)
        layer_scalar_tensor = torch.zeros(32, dtype=torch.bfloat16)
        layer_scalar_tensor[0] = layer_scalar_val

        # O weight: quantize with actual K (unpadded) so INT4 scale/data stride matches K
        region_writes = [
            (gamma_in, "bf16"),
            (q_w, "int4"),
            (k_w, "int4"),
            (v_w, "int4"),
            (gamma_q, "bf16"),
            (gamma_k, "bf16"),
            (o_w, "int4"),  # [hidden_size, cur_q_size] — actual dimensions
            (gamma_post, "bf16"),
            (gamma_ffn, "bf16"),
            (up_w, "int4"),
            (gate_w, "int4"),
            (down_w, "int4"),
            (gamma_post_ffn, "bf16"),
            (per_layer_input_gate_w, "bf16"),
            (per_layer_projection_w, "bf16"),
            (gamma_post_per_layer_input_norm, "bf16"),
            (layer_scalar_tensor, "bf16"),
        ]
        # Two passes: collect every IF4-quant job for this layer + write BF16
        # regions inline (cheap), then parallel-quantize the IF4 batch and
        # serially copy the results into the buffer.
        quant_jobs: list[tuple[torch.Tensor, int, int, int, int]] = []
        # (tensor, scale_off, scale_sz, data_off, data_sz)
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
                data_off = weight_defs[next_key] + layer_idx * LAYER_WEIGHT_SIZE
                quant_jobs.append((tensor, file_off, sz, data_off, data_sz))
                i += 2
            else:
                t = tensor.detach().cpu().to(torch.bfloat16).contiguous()
                raw = (t.view(torch.uint8).numpy().tobytes() + b"\x00" * sz)[:sz]
                write_at(file_off, raw)
                i += 1
            j += 1
        # Parallel quant of this layer's IF4 tensors via the canonical wrapper.
        quant_results = _parallel_quantize(LM_QUANT_PRECISION,
                                           [t for t, *_ in quant_jobs])
        for (data_bytes, scale_bytes), (_t, scale_off, sz, data_off, data_sz) in zip(
                quant_results, quant_jobs):
            scale_padded = (scale_bytes + b"\x00" * sz)[:sz]
            data_padded = (data_bytes + b"\x00" * data_sz)[:data_sz]
            write_at(scale_off, scale_padded)
            write_at(data_off, data_padded)

    # ROPE: two tables with different dimensions
    rope_cfg = cfg["special"]["rope"]
    theta = rope_cfg["theta"]
    local_base = rope_cfg["local_base"]
    num_positions = rope_cfg["num_positions"]
    partial_rotary_factor = rope_cfg["partial_rotary_factor_global"]

    # LOCAL RoPE: head_dim=256, full rotation, D=128
    D_local = head_dim_sliding // 2  # 128
    inv_freq_local = 1.0 / (local_base ** (torch.arange(D_local, dtype=torch.float32) / D_local))
    pos = torch.arange(num_positions, dtype=torch.float32)
    freqs_local = torch.outer(pos, inv_freq_local)
    cos_local = freqs_local.cos().to(torch.bfloat16)
    sin_local = freqs_local.sin().to(torch.bfloat16)
    rope_local = torch.cat([cos_local, cos_local, -sin_local, sin_local], dim=1)
    sz = weight_defs["ROPE_LOCAL_SIZE"]
    raw = (rope_local.contiguous().view(torch.uint8).numpy().tobytes() + b"\x00" * sz)[:sz]
    write_at(weight_defs["ROPE_LOCAL"], raw)

    # GLOBAL RoPE: head_dim=512, partial_rotary_factor=0.25, rotary_dims=128, D=64
    rotary_dims = int(head_dim * partial_rotary_factor)  # 128
    D_global = rotary_dims // 2  # 64
    inv_freq_global = 1.0 / (theta ** (torch.arange(D_global, dtype=torch.float32) / D_global))
    freqs_global = torch.outer(pos, inv_freq_global)
    cos_global = freqs_global.cos().to(torch.bfloat16)
    sin_global = freqs_global.sin().to(torch.bfloat16)
    rope_global = torch.cat([cos_global, cos_global, -sin_global, sin_global], dim=1)
    sz = weight_defs["ROPE_GLOBAL_SIZE"]
    raw = (rope_global.contiguous().view(torch.uint8).numpy().tobytes() + b"\x00" * sz)[:sz]
    write_at(weight_defs["ROPE_GLOBAL"], raw)

    # OUTPUT_NORM
    out_norm = text_model.norm.weight.detach().cpu().to(torch.bfloat16)
    sz = weight_defs["OUTPUT_NORM_WEIGHT_SIZE"]
    raw = (out_norm.contiguous().view(torch.uint8).numpy().tobytes() + b"\x00" * sz)[:sz]
    write_at(weight_defs["OUTPUT_NORM_WEIGHT"], raw)

    # PER_LAYER_MODEL_PROJ_WEIGHT: [1536, 8960] from model.model.language_model.per_layer_model_projection
    per_layer_model_proj_w = text_model.per_layer_model_projection.weight.detach().cpu().to(torch.bfloat16)
    sz = weight_defs["PER_LAYER_MODEL_PROJ_WEIGHT_SIZE"]
    raw = (per_layer_model_proj_w.contiguous().view(torch.uint8).numpy().tobytes() + b"\x00" * sz)[:sz]
    write_at(weight_defs["PER_LAYER_MODEL_PROJ_WEIGHT"], raw)

    # PER_LAYER_PROJ_NORM_WEIGHT: [256] from model.model.language_model.per_layer_projection_norm
    per_layer_proj_norm_w = (text_model.per_layer_projection_norm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
    sz = weight_defs["PER_LAYER_PROJ_NORM_WEIGHT_SIZE"]
    raw = (per_layer_proj_norm_w.contiguous().view(torch.uint8).numpy().tobytes() + b"\x00" * sz)[:sz]
    write_at(weight_defs["PER_LAYER_PROJ_NORM_WEIGHT"], raw)

    # LM_HEAD: tied with embed_tokens, so clone. Quantized via the canonical
    # wrapper alongside the rest of the LM matmuls.
    lm_head_w = model.lm_head.weight.detach().clone().cpu().to(torch.bfloat16)
    scale_sz = weight_defs["LM_HEAD_WEIGHT_SCALE_SIZE"]
    data_sz = weight_defs["LM_HEAD_WEIGHT_DATA_SIZE"]
    data_bytes, scale_bytes = _qs_quantize(LM_QUANT_PRECISION, lm_head_w)
    scale_padded = (scale_bytes + b"\x00" * scale_sz)[:scale_sz]
    data_padded = (data_bytes + b"\x00" * data_sz)[:data_sz]
    write_at(weight_defs["LM_HEAD_WEIGHT_SCALE"], scale_padded)
    write_at(weight_defs["LM_HEAD_WEIGHT_DATA"], data_padded)

    # Build the host side-cache bytes in memory (no separate files) and
    # concatenate into ONE weights bin: [LM | host]. A single master manifest
    # JSON holds each section's offset + its per-tensor sub-manifest.
    #
    # NOTE (LM-only refactor): the vision and audio sections are intentionally
    # omitted here. weight_init reads section offsets from this manifest, so an
    # existing combined [LM | vision | audio | host] params.bin still loads
    # correctly for the LM path; only a from-scratch regeneration produces the
    # slimmer [LM | host] layout. Add the vision/audio sections back when those
    # encoders land in this module.
    host_bytes, host_manifest = _build_host_section_bytes(text_model, cfg)

    lm_size   = len(buf)
    host_size = len(host_bytes)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(buf)
        f.write(host_bytes)
    total = lm_size + host_size
    print(f"Generated weights bin: {out_path} ({total/1024**3:.2f} GiB total; "
          f"LM {lm_size/1024**3:.2f} GiB + host {host_size/1024**3:.2f} GiB)")

    master_meta_path = out_path.rsplit(".", 1)[0] + ".json"
    master = {
        "compile_version": "v1",
        "total_size": total,
        "lm_section":   {"offset": 0,       "size": lm_size},
        "host_section": {"offset": lm_size, "size": host_size, "manifest": host_manifest},
    }
    with open(master_meta_path, "w") as f:
        json.dump(master, f, indent=2)
    print(f"Generated weights manifest: {master_meta_path}")

    # Remove any stale standalone side-cache files from the previous layout
    # so users don't confuse them with the new combined bin.
    for stale in ("host_weights.bin", "host_weights.json",
                   "vision_weights.bin", "vision_weights.json"):
        sp = os.path.join(os.path.dirname(out_path), stale)
        if os.path.exists(sp):
            os.remove(sp)
            print(f"  removed legacy side-cache: {sp}")

    # Bundle just the tokenizer / processor files into gemma4_e2b_bin/tokenizer/
    # so a deploy host can run inference without the full HF model directory
    # (model.safetensors is ~10 GB and not needed at run time). This lets
    # gemma4_e2b_run_from_bin.py work on a stripped deploy.
    tokenizer_subset_extract(os.path.dirname(out_path), cfg)
    return out_path




class Gemma4_UnifiedEngine(UnifiedEngine):
    """UnifiedEngine specialized to Gemma4 E2B (LM path): loads config + weight
    bin, compiles prefill/decoder into one bin, runs prefill + decode. Numeric
    checks live in gemma4_e2b_numeric.py."""

    def __init__(self, script_dir: str | None = None, local_weights: bool = False):
        engine_base = user_dma_core.UE_0_BASE_ADDR
        # Gemma4 FIXED DRAM layout (FULL 4 GB; see notes/notes_gemma4_e2b_vision.md
        # "Master layout table"). All addresses below 0x100000000 (DMA-mapped DRAM
        # is 0x00000000 – 0xFFFFFFFF — same as qwen2.5; old DRAM_START_ADDR=0x80000000
        # only used the upper 2 GB and wasted the rest).
        #   Weight LM     : 0x00000000 – 0x64000000  (1600 MB)
        #   Weight Vision : 0x64000000 – 0x6c000000  (128 MB)
        #   Weight Audio  : 0x6c000000 – 0x78000000  (192 MB)
        #   Act. Scratch  : 0x78000000 – 0x88000000  (256 MB) ← tensor_base default
        #   Act. KV cache : 0x88000000 – 0x98000000  (256 MB; tail of activation region)
        #   ISA Audio     : 0x98000000 – 0xa0000000  (128 MB)
        #   ISA Unified   : 0xa0000000 – 0x100000000 (1.5 GB) ← program_base default
        #     (formerly split into vision/prefill/decoder regions; collapsed
        #     into one contiguous region for the combined program image
        #     (LM prefill+decode ISA) plus the dispatch preamble past it.
        #     Total fits comfortably under 4 GB; the prior base 0xC0000000
        #     caused the image to overflow 4 GB once vision (~385 MB) was
        #     appended to LM (~661 MB) in the full test.py build.)
        _params_base  = 0x00000000   # Weight region start
        _tensor_base  = 0x78000000   # Activation region start (stage scratch)
        _program_base = 0xa0000000   # unified bin base, gives 1.5 GB headroom
        super().__init__(BASE_ADDR=engine_base,
                          params_dram_base=_params_base,
                          program_dram_base=_program_base,
                          tensor_dram_base=_tensor_base)
        self.script_dir = script_dir or SCRIPT_DIR
        self._cfg = self.load_config(script_dir=self.script_dir)
        self.weight_defs = self._cfg["_weight_defs"]

        fi = self._cfg["file_info"]
        model = self._cfg["model"]
        paths = self._cfg["paths"]

        self.vector_length = fi["hidden_size"]
        self.head_dim = fi["head_dim"]  # 512 (max, for uniform sizing)
        self.head_dim_sliding = fi["head_dim_sliding"]  # 256
        self.per_layer_input_dim = fi["per_layer_input_dim"]  # 256
        self.bytes_per_element = fi["bytes_per_element"]
        self.group_size = fi["group_size"]
        self.mlp_elements = fi["mlp_elements"]
        self.mlp_elements_wide = fi.get("mlp_elements_wide", fi["mlp_elements"])
        self.q_size = self.head_dim * self.group_size * self.bytes_per_element
        self.k_size = self.head_dim * self.bytes_per_element
        self.MAX_CONTEXT_SIZE = model["max_context_size"]
        self.LAYER_SIZE = fi["num_layers"]
        self.EMBEDDING_ELEMENTS = fi["embedding_vocab"]
        self._isa_reg_counter = 1
        fixed = self._cfg.get("fixed_isa_regs", {})
        # Dynamic-PBI register binding (gemma3-style). Config keys:
        #   TMP_REG (scratch for reg_mul_imm + add_imm address math)
        #   GPR_SEQ_LEN_REG     — runtime seq_len / decode_pos
        #   GPR_Q_SEQ_LEN_REG   — seq_len × group_size (Q-side row count)
        #   GPR_BUCKET_IDX_REG  — retired bucket selector, now a general scratch reg
        #       (reused by vision RoPE loops). Bound to self.gpr_scratch below. The
        #       JSON key keeps its legacy name for gemma4_e2b_run_from_bin.py, which
        #       still reads it; rename the key when that file is migrated.
        #   GPR_ALIGNED_SEQ_LEN_REG — feeds unified_attention_core's dynamic aligned_seq_len
        self.TMP_REG       = fixed["TMP_REG"]
        self.gpr_seq_len    = fixed["GPR_SEQ_LEN_REG"]
        self.gpr_q_seq_len  = fixed["GPR_Q_SEQ_LEN_REG"]
        self.gpr_scratch    = fixed["GPR_BUCKET_IDX_REG"]  # retired bucket reg → scratch
        self.gpr_aligned_seq_len = fixed["GPR_ALIGNED_SEQ_LEN_REG"]
        # Dynamic GPR allocation starts past the five reserved regs. The ISA
        # register file holds 63 GPRs total (see user_dma_core.py's
        # matmat_mul_dynamic_core), so this leaves ample headroom.
        self._isa_reg_counter = 6
        self._isa_reg_base = 6  # one-shot mode resets the allocator to this base per sub-op
        self.causal_mask_upper = False
        self._rope_global_layers = set(model["rope_global_layers"])
        self._full_attention_layers = set(model["full_attention_layers"])
        self._double_wide_mlp_first = model.get("double_wide_mlp_first_layer", fi["num_layers"])
        self._end_of_turn_token_id = model["end_of_turn_token_id"]
        # Sliding-attention window. Sliding layers (i.e. layers NOT in
        # full_attention_layers) are limited to attending the last
        # `sliding_window` tokens. Default to MAX_CONTEXT_SIZE so older configs
        # without this key keep their old behaviour (no real windowing).
        self.sliding_window = model.get("sliding_window", model["max_context_size"])
        # max_prefill_seq_len caps the LM prefill attention shapes independently
        # of max_context_size. Decode may still need MAX_CONTEXT_SIZE KV rows, so
        # tensor_init sizes shared attention buffers for the larger aligned shape.
        self.max_prefill_seq_len = model.get("max_prefill_seq_len", model["max_context_size"])
        # KV sharing map: built from HF model during weight_init
        self._kv_shared_map = {}  # layer_idx -> reference_layer_idx (populated in weight_init)
        self._gamma_bin_offset = self._cfg["special"]["rms_norm"]["gamma_offset"]
        self._per_layer_model_proj_scale = model["per_layer_model_proj_scale"]
        self._per_layer_input_scale = model["per_layer_input_scale"]
        self.prefill_seq = None

        self._weights_bin_rel = "gemma4_e2b_bin/params.bin" if local_weights else paths["weights_bin"]
        self.weight_init()
        self.tensor_init()

    def _emit_sram_copy_chunked(self, src_addr: int, dst_addr: int,
                                 num_elements: int, chunk: int = 131072) -> None:
        """Emit a chunked DRAM→SRAM→DRAM copy into the capture buffer. Used to
        shuttle large activations when a direct DRAM-to-DRAM path isn't handy."""
        bpe = self.bytes_per_element
        for off in range(0, num_elements, chunk):
            n = min(chunk, num_elements - off)
            self.accelerator_memory_to_sram(
                accelerator_dram_address=src_addr + off * bpe,
                sram_address=0x10000, element_size=n)
            self.sram_to_accelerator_memory(
                sram_address=0x10000,
                accelerator_dram_address=dst_addr + off * bpe,
                element_size=n)

    @staticmethod
    def load_config(config_path: str | None = None, script_dir: str | None = None) -> dict:
        """Load gemma4_e2b_config.json and build weight_defs (offset/size dict) from regions."""
        if config_path is None:
            script_dir = script_dir or SCRIPT_DIR
            config_path = os.path.join(script_dir, "gemma4_e2b_config.json")
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

    def set_prefill_seq(self, prompt: str | None = None) -> None:
        """Set self.prefill_seq from a text prompt (tokenize with chat template) or from config default."""
        if prompt is not None:
            conversation = [{"role": "user", "content": prompt}]
            prompt_with_template = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            self.prefill_seq = tuple(self.tokenizer.encode(prompt_with_template, add_special_tokens=True))
            print(f"Prefill from prompt ({len(self.prefill_seq)} tokens): {prompt!r}")
        else:
            self.prefill_seq = tuple(self._cfg["default_prefill_tokens"])
            decoded = self.tokenizer.decode(list(self.prefill_seq), skip_special_tokens=True)
            print(f"Prefill from default ({len(self.prefill_seq)} tokens): {decoded!r}")

    def _structural_token_ids(self) -> set:
        """Token ids never repetition-penalized (punctuation/whitespace/special);
        exempting these 'glue' tokens stops penalized text collapsing. Cached."""
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
        """Build the per-vocab additive bias from the windowed token frequency and
        DMA it to PENALTY_BIAS_DRAM. bias[t] = clamp(-alpha*count[t], min=-cap);
        structural tokens stay 0."""
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
        # --- anti-loop hard ban (overrides the structural exemption above) ---
        # The structural exemption keeps glue tokens ("|", newline, space) out of
        # the soft frequency penalty so penalized text doesn't turn to word-salad
        # -- but that is exactly why the penalty alone can't break a degenerate
        # collapse whose tokens ARE structural (e.g. a "|"<->"\n" 2-cycle, which
        # naive consecutive-run detection would miss). Instead: over the last
        # `recent_w` generated tokens, hard-ban any token that fills >= `loop_thr`
        # of them (single-token run = all; 2-cycle = ~half each). No coherent text
        # fills a third of a short window with one token, so it never fires on
        # real output. (E2B VLM is empirically immune to the E4B image cycle, but
        # the penalty path is shared; harmless here, present for symmetry.)
        # Tunable: GEMMA4_PEN_LOOP_RECENT (window, 0=off), GEMMA4_PEN_LOOP_THR.
        recent_w = int(getattr(self, "pen_loop_recent", 24))
        loop_thr = int(getattr(self, "pen_loop_thr", 8))
        if recent_w > 0 and len(prev_tokens) >= recent_w:
            from collections import Counter
            _cnt = Counter(int(t) for t in prev_tokens[-recent_w:])
            _ban = [tok for tok, c in _cnt.items() if c >= loop_thr]
            if _ban:
                bias[0, torch.tensor(_ban, dtype=torch.long)] = -1e9  # finite, bf16-safe
        self.dma_to_accelerator_memory(self.PENALTY_BIAS_DRAM, bias)

    def get_embedding_for_tokens(self, token_ids: list[int] | tuple) -> torch.Tensor:
        """Return (len(token_ids), vector_length) bfloat16 tensor from self.embedding_weight (HF, scale applied)."""
        tid_t = torch.tensor(token_ids, dtype=torch.long)
        out = torch.zeros(len(token_ids), self.vector_length, dtype=torch.bfloat16)
        valid = tid_t < self.embedding_weight.shape[0]
        out[valid] = self.embedding_weight[tid_t[valid]]
        return out

    def _load_rope_host(self, rope_theta: float | None = None, rope_local_base: float | None = None) -> None:
        """Generate RoPE (cos, cos, -sin, sin) on host and write to DRAM. Uses config for sizes and num_positions.
        LOCAL: head_dim=256, full rotation. GLOBAL: head_dim=512, partial rotation (first 128 dims)."""
        rope_cfg = self._cfg["special"]["rope"]
        theta = rope_theta if rope_theta is not None else rope_cfg["theta"]
        local_base = rope_local_base if rope_local_base is not None else rope_cfg["local_base"]
        num_rope_positions = rope_cfg["num_positions"]
        partial_rotary_factor = rope_cfg["partial_rotary_factor_global"]

        # LOCAL RoPE: head_dim_sliding=256, full rotation, D=128
        D_local = self.head_dim_sliding // 2  # 128
        inv_freq_local = 1.0 / (local_base ** (torch.arange(D_local, dtype=torch.float32) / D_local))
        pos = torch.arange(num_rope_positions, dtype=torch.float32)
        freqs_local = torch.outer(pos, inv_freq_local)
        cos_local = freqs_local.cos().to(torch.bfloat16)
        sin_local = freqs_local.sin().to(torch.bfloat16)
        rope_local = torch.cat([cos_local, cos_local, -sin_local, sin_local], dim=1)
        sz = self.weight_defs["ROPE_LOCAL_SIZE"]
        raw = rope_local.contiguous().view(torch.uint8).numpy().tobytes()
        raw = (raw + b"\x00" * sz)[:sz]
        addr = self.allocate_params_dram(sz)
        self.dma_write(DMA_DEVICE_H2C, addr, raw, sz)
        self.DRAM_ADDR_ROPE_LOCAL = addr

        # GLOBAL RoPE: head_dim=512, partial_rotary_factor=0.25, rotary_dims=128, D=64
        rotary_dims = int(self.head_dim * partial_rotary_factor)  # 128
        D_global = rotary_dims // 2  # 64
        inv_freq_global = 1.0 / (theta ** (torch.arange(D_global, dtype=torch.float32) / D_global))
        freqs_global = torch.outer(pos, inv_freq_global)
        cos_global = freqs_global.cos().to(torch.bfloat16)
        sin_global = freqs_global.sin().to(torch.bfloat16)
        rope_global = torch.cat([cos_global, cos_global, -sin_global, sin_global], dim=1)
        sz = self.weight_defs["ROPE_GLOBAL_SIZE"]
        raw = rope_global.contiguous().view(torch.uint8).numpy().tobytes()
        raw = (raw + b"\x00" * sz)[:sz]
        addr = self.allocate_params_dram(sz)
        self.dma_write(DMA_DEVICE_H2C, addr, raw, sz)
        self.DRAM_ADDR_ROPE_GLOBAL = addr

    def _load_host_weights_from_combined_bin(self, host_section: dict, base_offset: int) -> None:
        """mmap the combined weights bin and create read-only torch tensor
        views over the host section. Zero-copy: nothing materializes in
        RSS until a row is indexed.

        `host_section["manifest"]` gives tensor offsets RELATIVE to the
        host section start; we add `base_offset` (the host section's
        absolute file offset) when creating the view.
        """
        sub = host_section["manifest"]

        def _view(section_name: str) -> torch.Tensor:
            s = sub[section_name]
            shape = tuple(s["shape"])
            n_elems = 1
            for d in shape:
                n_elems *= d
            return torch.frombuffer(
                self.weight_bin,
                dtype=torch.bfloat16,
                count=n_elems,
                offset=base_offset + s["offset"],
            ).reshape(shape)

        self.embed_tokens_per_layer_weight = _view("embed_tokens_per_layer")
        self.per_layer_model_proj_weight   = _view("per_layer_model_proj")
        self.per_layer_proj_norm_weight    = _view("per_layer_proj_norm")
        self._layer_scalars = list(sub["layer_scalars"])
        self._kv_shared_map = {int(k): int(v) for k, v in sub.get("kv_shared_map", {}).items()}
        print(f"[weight_init] host section mmap'd at file offset 0x{base_offset:X}: "
              f"embed_tokens_per_layer={tuple(self.embed_tokens_per_layer_weight.shape)} bf16 "
              f"({sub['embed_tokens_per_layer']['size']/1024**3:.2f} GiB, page-cached on demand)")

    def weight_init(self) -> None:
        """Ensure weight bin exists (generate from HF if missing), then mmap it
        and initialize FPGA DRAM: embedding, layers from bin, RoPE, OUTPUT_NORM/LM_HEAD.

        Host-side tensors needed for per-layer-input computation
        (per_layer_embed_tokens, per_layer_model_proj, per_layer_proj_norm,
        layer_scalars, kv_shared_map) come from `host_weights.bin` if it
        exists, mmap'd so RSS stays minimal. Otherwise we fall back to
        loading the HF model — which costs 6-12 GB host RAM and is OOM on
        a 16 GB Raspberry Pi. The first run on a beefier machine should
        generate the side-cache so subsequent runs anywhere can skip the
        HF model entirely.
        """
        import mmap as _mmap

        full_path = os.path.join(self.script_dir, self._weights_bin_rel)
        if os.path.exists(full_path):
            print(f"Weight bin exists, skip generation: {full_path}")
        else:
            print(f"Weight bin not found, generating: {full_path}")
            weight_bin_generate(output_path=full_path)

        # mmap the weight bin — read-only, OS pages in only what's touched.
        # Replaces a 2.4 GB f.read() that pinned the whole bin in RSS.
        self._weight_bin_fp = open(full_path, "rb")
        self.weight_bin = _mmap.mmap(self._weight_bin_fp.fileno(), 0,
                                     prot=_mmap.PROT_READ)

        # Master manifest: one JSON per combined weight bin. Holds the
        # offsets/sizes of the three sections (lm, vision, host) plus each
        # section's sub-manifest. If missing, the bin was generated with
        # the old multi-file layout and we need to regenerate.
        master_meta_path = full_path.rsplit(".", 1)[0] + ".json"
        if not os.path.exists(master_meta_path):
            raise RuntimeError(
                f"weights master manifest missing: {master_meta_path}\n"
                f"This bin was produced by the old multi-file layout. "
                f"Delete {full_path} (and any stale host_weights.bin / "
                f"vision_weights.bin) and re-run gemma4_e2b_test.py to "
                f"regenerate the combined bin.")
        with open(master_meta_path, "r") as f:
            self._weights_master = json.load(f)

        # Embedding: a zero-copy mmap view directly into the weight bin.
        # No 770 MB host allocation; only the touched rows (one per decode
        # token, ~3 KB) cost RSS. Read-only is fine because we only do
        # `embedding_weight[token_ids]` lookups.
        emb_cfg = self._cfg["special"]["embedding"]
        token_embd_offset = _parse_offset(emb_cfg["token_embd_offset"])
        vocab_size  = self.EMBEDDING_ELEMENTS         # 262144
        emb_dim     = self.vector_length              # 1536
        self.embedding_weight = torch.frombuffer(
            self.weight_bin,
            dtype=torch.bfloat16,
            count=vocab_size * emb_dim,
            offset=token_embd_offset,
        ).reshape(vocab_size, emb_dim)

        host_section = self._weights_master["host_section"]
        self._load_host_weights_from_combined_bin(host_section, host_section["offset"])

        # Tokenizer: from the bundled subset; full HF model not needed.
        tok_subset = os.path.join(self.script_dir, "gemma4_e2b_bin", "tokenizer")
        if os.path.exists(os.path.join(tok_subset, "tokenizer.json")):
            tok_dir = tok_subset
        else:
            tok_dir = os.path.join(self.script_dir, self._cfg["paths"]["hf_model_dir"])
        self.tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)

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

        last_structure_key = self._cfg["layers"]["structure"][-1]["key"]
        layer0_end = (self.weight_defs[last_structure_key] - base_layer0
                      + self.weight_defs[f"{last_structure_key}_SIZE"])
        assert layer0_end <= LAYER_WEIGHT_SIZE, (
            f"Layer 0 size overflow: computed {layer0_end} > LAYER_WEIGHT_SIZE {LAYER_WEIGHT_SIZE}"
        )

        print(f"\n--- Loading weights to DRAM ---")
        layers_total = self.LAYER_SIZE * LAYER_WEIGHT_SIZE
        layers_base_dram = self.allocate_params_dram(layers_total)
        load_t0 = time.perf_counter()
        for layer_idx in range(self.LAYER_SIZE):
            if layer_idx > 0 and layer_idx % 10 == 0:
                print(f"    layer {layer_idx}/{self.LAYER_SIZE} loaded ({time.perf_counter()-load_t0:.1f}s)")
            for off_key, sz_key, attr in blk0_regions:
                off = self.weight_defs[off_key]
                sz = self.weight_defs[sz_key]
                bin_off = off + layer_idx * LAYER_WEIGHT_SIZE
                raw = self.weight_bin[bin_off : bin_off + sz]
                offset_in_layer = off - base_layer0
                dram_addr = layers_base_dram + layer_idx * LAYER_WEIGHT_SIZE + offset_in_layer
                self.dma_write(DMA_DEVICE_H2C, dram_addr, raw, sz)
            if layer_idx == 0:
                for off_key, sz_key, attr in blk0_regions:
                    off = self.weight_defs[off_key]
                    offset_in_layer = off - base_layer0
                    setattr(self, attr, layers_base_dram + offset_in_layer)
        print(f"  Loaded {self.LAYER_SIZE} layers ({layers_total/(1024*1024):.1f} MB)")

        for off_key, sz_key, attr in non_layer:
            off = self.weight_defs[off_key]
            sz = self.weight_defs[sz_key]
            raw = self.weight_bin[off : off + sz]
            addr = self.allocate_params_dram(sz)
            self.dma_write(DMA_DEVICE_H2C, addr, raw, sz)
            setattr(self, attr, addr)

        self._load_rope_host()
        print(f"  Total weight DRAM: {self.get_params_dram_usage()/(1024*1024):.1f} MB")
        print("Tokenizer loaded.")

    def tensor_init(self) -> None:
        """Initialize hardware DRAM for gemma4 E2B model (layer-wise overlap except for kv cache).
        KV cache uses max head_dim (512) for uniform sizing."""
        seq_len = self.MAX_CONTEXT_SIZE
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        # Unified attention scratch/bias buffers are largest during prefill,
        # but decode can still require MAX_CONTEXT_SIZE KV rows. Size the
        # shared attention buffers for the larger aligned dimension.
        prefill_seq_len = min(self.max_prefill_seq_len, self.MAX_CONTEXT_SIZE)
        prefill_q_seq_len = prefill_seq_len * self.group_size
        prefill_aligned_seq_len = ((prefill_q_seq_len + 63) // 64) * 64
        decode_aligned_seq_len = ((self.MAX_CONTEXT_SIZE + 63) // 64) * 64
        attention_aligned_seq_len = max(prefill_aligned_seq_len, decode_aligned_seq_len)

        # Build compact KV slot map: only layers that own KV state get a slot.
        # KV-shared layers point at their reference layer's slot, so L15-34 do not
        # consume cache space (saves ~40 MB at MAX_CONTEXT_SIZE=1024).
        non_shared_layers = [l for l in range(self.LAYER_SIZE) if l not in self._kv_shared_map]
        self._kv_slot_for_layer = {}
        for slot, l in enumerate(non_shared_layers):
            self._kv_slot_for_layer[l] = slot
        for shared_l, ref_l in self._kv_shared_map.items():
            self._kv_slot_for_layer[shared_l] = self._kv_slot_for_layer[ref_l]
        self._num_kv_slots = len(non_shared_layers)
        _kv_saved = (self.LAYER_SIZE - self._num_kv_slots) * self.MAX_CONTEXT_SIZE * self.k_size * 2  # K+V
        print(f"KV cache: {self._num_kv_slots} unique slots (of {self.LAYER_SIZE} layers), saved {_kv_saved / (1024*1024):.1f} MB via KV sharing")
        # Allocate shared memory for k v cache (k rope and v projection) and zero pad for decoder use:
        # Uses max head_dim (512) = self.k_size for uniform sizing
        self.LAYER0_V_DRAM = self.allocate_tensor_dram(self._num_kv_slots * self.MAX_CONTEXT_SIZE * self.k_size)
        self.LAYER0_K_ROPE_DRAM = self.allocate_tensor_dram(self._num_kv_slots * self.MAX_CONTEXT_SIZE * self.k_size)
        zero_pad = torch.zeros(self._num_kv_slots * self.MAX_CONTEXT_SIZE * self.k_size, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_K_ROPE_DRAM, zero_pad)
        # Allocate memory for constant zero tensor, identity matrix, and bias:
        zero_add = torch.zeros(seq_len * self.head_dim * self.bytes_per_element, dtype=torch.bfloat16)
        self.ZERO_DRAM_ADDR = self.allocate_tensor_dram(seq_len * self.head_dim * self.bytes_per_element)
        self.dma_to_accelerator_memory(self.ZERO_DRAM_ADDR, zero_add)
        self.IDENTITY_DRAM_ADDR = self.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * self.bytes_per_element)
        self.dma_to_accelerator_memory(self.IDENTITY_DRAM_ADDR, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))
        # Allocate memory for attention and zero pad. Prefill uses
        # seq_len*group_size rows; decode uses MAX_CONTEXT_SIZE KV rows, so size
        # the shared buffers for the larger aligned dimension.
        self.LAYER0_FLASH_Q_DRAM = self.allocate_tensor_dram(attention_aligned_seq_len * self.head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_K_DRAM = self.allocate_tensor_dram(attention_aligned_seq_len * self.head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_V_DRAM = self.allocate_tensor_dram(attention_aligned_seq_len * self.head_dim * self.bytes_per_element)
        zero_pad = torch.zeros(attention_aligned_seq_len * self.head_dim * self.bytes_per_element, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_Q_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_K_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_V_DRAM, zero_pad)
        # Allocate memory for layer intermediate tensors:
        self.LAYER0_INPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_Q_DRAM = self.allocate_tensor_dram(seq_len * self.q_size)
        self.LAYER0_K_DRAM = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_K_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_Q_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.q_size)
        self.LAYER0_FLASH_OUTPUT_DRAM = self.allocate_tensor_dram(attention_aligned_seq_len * self.head_dim * self.bytes_per_element)
        # unified_attention_core scratch layout:
        #   V.T [HD, S] + scores [S, S] + scaled_q [batch, HD].
        # Worst case batch <= S, so allocate S*S + 2*HD*S elements.
        self.LAYER0_FLASH_SCRATCH_DRAM = self.allocate_tensor_dram(
            (attention_aligned_seq_len * attention_aligned_seq_len
             + 2 * self.head_dim * attention_aligned_seq_len) * self.bytes_per_element)
        # Two full-matrix bias buffers: full-attention layers attend to
        # the entire causal window, sliding-attention layers are limited to
        # `sliding_window` tokens. compile_prefill / compile_decoder pick the
        # right address per layer; run_prefill / run_decoder upload both.
        self.LAYER0_FLASH_BIAS_FULL_DRAM = self.allocate_tensor_dram(attention_aligned_seq_len * attention_aligned_seq_len * self.bytes_per_element)
        self.LAYER0_FLASH_BIAS_SLIDING_DRAM = self.allocate_tensor_dram(attention_aligned_seq_len * attention_aligned_seq_len * self.bytes_per_element)
        # Backwards-compat alias (older callers use the singular name).
        self.LAYER0_FLASH_BIAS_DRAM = self.LAYER0_FLASH_BIAS_FULL_DRAM
        self.LAYER0_ATTN_PROJ_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_ATTN_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_ATTN_RESIDUAL_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_MLP_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        mlp_max = max(self.mlp_elements, self.mlp_elements_wide)
        self.LAYER0_MLP_GATE_DRAM = self.allocate_tensor_dram(seq_len * mlp_max * 2)
        self.LAYER0_MLP_UP_DRAM = self.allocate_tensor_dram(seq_len * mlp_max * 2)
        self.LAYER0_MLP_MULT_DRAM = self.allocate_tensor_dram(seq_len * mlp_max * 2)
        self.LAYER0_MLP_DOWN_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_MLP_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.OUTPUT_NORM_DRAM = self.allocate_tensor_dram(1 * self.vector_length * self.bytes_per_element)
        self.LOGITS_DRAM = self.allocate_tensor_dram(1 * self.EMBEDDING_ELEMENTS * self.bytes_per_element)
        # On-FPGA repetition-penalty bias: the LM-head matmul's C term
        # (bias_mode="broadcast_N"). Allocated immediately after LOGITS_DRAM (and at
        # the SAME point in gemma4_e2b_run_from_bin.py) so its baked address matches.
        # All-zero == no penalty (pure greedy); _write_penalty_bias() fills it only
        # when GEMMA4_PENALTY=1. The HW argmax of (logits + bias) then returns the
        # penalized token with NO logit readback — identical mechanism to
        # llama3.2_1b (notes_repetition_penalty_fpga_bias.md).
        self.PENALTY_BIAS_DRAM = self.allocate_tensor_dram(1 * self.EMBEDDING_ELEMENTS * self.bytes_per_element)

        # Per-layer input injection buffers
        # PER_LAYER_INPUTS_DRAM: holds per_layer_inputs for all layers: MAX_CONTEXT_SIZE x 35 x 256 x 2 bytes
        self.PER_LAYER_INPUTS_DRAM = self.allocate_tensor_dram(self.MAX_CONTEXT_SIZE * self.LAYER_SIZE * self.per_layer_input_dim * self.bytes_per_element)
        # Intermediate DRAMs for per-layer injection
        self.LAYER0_PER_LAYER_GATE_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.per_layer_input_dim * self.bytes_per_element)
        self.LAYER0_PER_LAYER_PROJ_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * self.bytes_per_element)

        print(f"    Tensor DRAM usage: {self.get_tensor_dram_usage()/(1024*1024):.1f} MB")

    # Per-layer dim resolution for Gemma4's heterogeneous stack. Two independent
    # axes → 4 layer buckets. Values below are for the current config (35 layers,
    # full_attention_layers={4,9,14,19,24,29,34}, double_wide_mlp_first_layer=15,
    # group_size=8, partial_rotary_factor_global=0.25):
    #
    #   attention type (every 5th layer is full/global, rest sliding-window):
    #     full    : head_dim=512, q_size=4096, k_size=512, rope_N=128 (partial)
    #     sliding : head_dim=256, q_size=2048, k_size=256, rope_N=256 (full)
    #   MLP width (doubles from layer 15 onward, the KV-shared layers):
    #     layer <15 : mlp=6144        layer >=15 : mlp=12288
    #
    #   layers 0-3,5-8,10-13         -> (256,2048,256) rope 256 mlp 6144
    #   layers 4,9,14                -> (512,4096,512) rope 128 mlp 6144
    #   layers 15-18,20-23,25-28,30-33 -> (256,2048,256) rope 256 mlp 12288
    #   layers 19,24,29,34           -> (512,4096,512) rope 128 mlp 12288
    def _get_layer_attention_dims(self, layer_idx: int) -> tuple[int, int, int]:
        """Return (cur_head_dim, cur_q_size, cur_k_size) for a given layer index."""
        if layer_idx in self._full_attention_layers:
            cur_head_dim = self.head_dim  # 512
            cur_q_size = cur_head_dim * self.group_size  # 4096
            cur_k_size = cur_head_dim  # 512
        else:
            cur_head_dim = self.head_dim_sliding  # 256
            cur_q_size = cur_head_dim * self.group_size  # 2048
            cur_k_size = cur_head_dim  # 256
        return cur_head_dim, cur_q_size, cur_k_size

    def _get_rope_dims(self, layer_idx: int) -> int:
        """Return the number of dims to apply RoPE on for a given layer.
        Sliding layers: full rotation on head_dim_sliding=256, so N=256.
        Full attention layers: partial rotation, only first 128 dims of head_dim=512, so N=128."""
        if layer_idx in self._full_attention_layers:
            partial_rotary_factor = self._cfg["special"]["rope"]["partial_rotary_factor_global"]
            return int(self.head_dim * partial_rotary_factor)  # 128
        else:
            return self.head_dim_sliding  # 256

    def _get_mlp_elements(self, layer_idx: int) -> int:
        """Return MLP intermediate size for a given layer (wide for KV-shared layers)."""
        if layer_idx >= self._double_wide_mlp_first:
            return self.mlp_elements_wide
        return self.mlp_elements

    def _compile_per_layer_injection(self, layer_idx: int, layer_off: int, seq_len: int) -> int:
        """Compile per-layer input injection block. Returns flops added."""
        total_flops = 0
        # gate = gelu(per_layer_input_gate @ hidden_state): Linear(1536->256) + GELU
        total_flops += self.matmat_mul_core(M=seq_len, K=self.vector_length, N=self.per_layer_input_dim,
            A_DRAM_ADDR=self.LAYER0_OUTPUT_DRAM,
            B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PER_LAYER_GATE + layer_off,
            OUTPUT_DRAM_ADDR=self.LAYER0_PER_LAYER_GATE_OUTPUT_DRAM,
            gelu_enable=True)
        # gated = gate * per_layer_input[layer_idx]  (small: seq_len * 256)
        per_layer_input_addr = self.PER_LAYER_INPUTS_DRAM + layer_idx * seq_len * self.per_layer_input_dim * self.bytes_per_element
        self.eltwise_core_dram(
            M=seq_len, N=self.per_layer_input_dim,
            dram_a=self.LAYER0_PER_LAYER_GATE_OUTPUT_DRAM,
            dram_b=per_layer_input_addr,
            dram_out=self.LAYER0_PER_LAYER_GATE_OUTPUT_DRAM,
            mode=UE_MODE.ELTWISE_MUL)
        # projected = per_layer_projection @ gated: Linear(256->1536)
        total_flops += self.matmat_mul_core(M=seq_len, K=self.per_layer_input_dim, N=self.vector_length,
            A_DRAM_ADDR=self.LAYER0_PER_LAYER_GATE_OUTPUT_DRAM,
            B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PER_LAYER_PROJ + layer_off,
            OUTPUT_DRAM_ADDR=self.LAYER0_PER_LAYER_PROJ_OUTPUT_DRAM)

        # FUSED: rms_norm(projected) + (hidden + normed) + layer_scalar
        # Previously: rms_norm_core_dram wrote the normed output to DRAM,
        # then a chunked SRAM pass read it back for the residual add. This
        # DRAM write+read pair is eliminated by running the per-row RMS norm
        # on projection data that's ALREADY in URAM_A, then immediately
        # loading the residual hidden-state chunk into URAM_B and doing the
        # add + scalar-mul before the single final writeback.
        #
        # SRAM layout (chunked by row):
        #   URAM_A [0x10000..0x80000]: projection chunk in-place (M_chunk rows × N)
        #   URAM_B [0x80000..0x80000+N*bpe]: gamma (loaded once outside the loop)
        #   URAM_B [0x80000+N*bpe..]: hidden-state chunk (M_chunk rows × N)
        bpe = self.bytes_per_element
        N = self.vector_length  # 1536
        gamma_addr = self.DRAM_ADDR_LAYER0_POST_PER_LAYER_NORM_GAMMA + layer_off
        gamma_sram = 0x80000
        hidden_sram_base = 0x80000 + N * bpe  # leave room for gamma
        proj_sram_base = 0x10000
        # Max rows per chunk is bounded by whichever URAM region is smaller
        # after reserving Q (0x00000-0x10000), gamma, and accounting for
        # two-row-aligned layout.
        uram_a_free_elements = (0x80000 - proj_sram_base) // bpe   # 229376
        uram_b_free_elements = (0x100000 - hidden_sram_base) // bpe  # 259072
        max_rows_per_chunk = min(uram_a_free_elements, uram_b_free_elements) // N
        if max_rows_per_chunk < 1:
            max_rows_per_chunk = 1
        # Upload gamma once for all chunks
        self.accelerator_memory_to_sram(
            accelerator_dram_address=gamma_addr,
            sram_address=gamma_sram, element_size=N)
        for m_off in range(0, seq_len, max_rows_per_chunk):
            m_take = min(max_rows_per_chunk, seq_len - m_off)
            chunk_bytes_offset = m_off * N * bpe
            # Load projection chunk into URAM_A
            self.accelerator_memory_to_sram(
                accelerator_dram_address=self.LAYER0_PER_LAYER_PROJ_OUTPUT_DRAM + chunk_bytes_offset,
                sram_address=proj_sram_base, element_size=m_take * N)
            # Per-row RMS norm in-place with the URAM_B-resident gamma
            for row in range(m_take):
                row_sram = proj_sram_base + row * N * bpe
                self.rms_norm_core(row_sram, row_sram, N, gamma_sram)
            # Load hidden-state chunk into URAM_B (past gamma)
            self.accelerator_memory_to_sram(
                accelerator_dram_address=self.LAYER0_OUTPUT_DRAM + chunk_bytes_offset,
                sram_address=hidden_sram_base, element_size=m_take * N)
            # Residual add: normed (URAM_A) + hidden (URAM_B) → URAM_A
            self.eltwise_add_core(
                vector_A_sram_start_addr=proj_sram_base,
                vector_B_sram_start_addr=hidden_sram_base,
                vector_C_sram_wb_addr=proj_sram_base,
                element_size=m_take * N)
            # Multiply by layer_scalar in-place
            self.broadcast_mul(
                scalar=self._layer_scalars[layer_idx],
                sram_start_addr=proj_sram_base,
                sram_wb_addr=proj_sram_base,
                element_size=m_take * N)
            # Single writeback per chunk
            self.sram_to_accelerator_memory(
                sram_address=proj_sram_base,
                accelerator_dram_address=self.LAYER0_OUTPUT_DRAM + chunk_bytes_offset,
                element_size=m_take * N)
        return total_flops

    def _compile_per_layer_injection_prefill(self, layer_idx: int, layer_off: int) -> int:
        """Seq_len-agnostic per-layer input injection for PREFILL.

        Same math as _compile_per_layer_injection (which decode still uses with
        seq_len=1), but the row count is applied indirectly through gpr_seq_len
        at runtime, so a single captured program is valid for any prompt length:
          * gate / projection matmuls read M from gpr_seq_len (M= is template);
          * per_layer_input is packed [layer, token, dim] with a seq_len row
            stride, so its per-layer base is computed from gpr_seq_len and the
            gate*per_layer_input multiply runs as a per-row ISA loop;
          * the fused rms_norm + residual + layer_scalar step is a plain per-row
            ISA loop — one row resident at a time, no chunking / SRAM reuse.
        """
        total_flops = 0
        bpe = self.bytes_per_element
        dim = self.per_layer_input_dim   # 256
        N   = self.vector_length         # 1536
        M_tmpl = self.seq_len            # compile-time template (FLOPs / loop_cnt only)

        # gate = gelu(hidden @ per_layer_input_gate): [S,1536] @ [1536,256] -> [S,256]
        total_flops += self.matmat_mul_core(M=M_tmpl, K=N, N=dim,
            A_DRAM_ADDR=self.LAYER0_OUTPUT_DRAM,
            B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PER_LAYER_GATE + layer_off,
            OUTPUT_DRAM_ADDR=self.LAYER0_PER_LAYER_GATE_OUTPUT_DRAM,
            gelu_enable=True, gpr_M_reg=self.gpr_seq_len)

        # gated = gate * per_layer_input[layer_idx], per-row (runtime trip count).
        # per_layer_input[layer] base = PER_LAYER_INPUTS_DRAM + layer*seq_len*dim*bpe,
        # computed from gpr_seq_len; a running pointer walks one row per iteration.
        gate_dram = self.LAYER0_PER_LAYER_GATE_OUTPUT_DRAM
        # eltwise operands must live in different URAM banks (split at 0x80000):
        # A in URAM_A (<0x80000), B in URAM_B (>=0x80000).
        sram_a, sram_b = 0x10000, 0x80000
        _pli = self.alloc_isa_reg()
        self.generate_instruction_reg_mul_imm(_pli, self.gpr_seq_len,
            ue_35bit_addr_shifter(layer_idx * dim * bpe))
        self.generate_instruction_add_imm(_pli,
            ue_35bit_addr_shifter(self.PER_LAYER_INPUTS_DRAM), _pli)
        _r = self.alloc_isa_reg()
        self.generate_instruction_add_set(_r, 0)
        self.loop_start(loop_cnt=M_tmpl, gpr_loop_cnt=self.gpr_seq_len)
        # gate row -> SRAM_A
        self.generate_instruction_reg_mul_imm(self.TMP_REG, _r, ue_35bit_addr_shifter(dim * bpe))
        self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(gate_dram), self.TMP_REG)
        self.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=sram_a,
                                        element_size=dim, general_reg_src=self.TMP_REG)
        # per_layer_input row -> SRAM_B (running pointer)
        self.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=sram_b,
                                        element_size=dim, general_reg_src=_pli)
        # gated = gate * per_layer_input -> SRAM_A
        self.eltwise_mul_core(vector_A_sram_start_addr=sram_a, vector_B_sram_start_addr=sram_b,
                              vector_C_sram_wb_addr=sram_a, element_size=dim)
        # SRAM_A -> gate row (in place)
        self.generate_instruction_reg_mul_imm(self.TMP_REG, _r, ue_35bit_addr_shifter(dim * bpe))
        self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(gate_dram), self.TMP_REG)
        self.sram_to_accelerator_memory(sram_address=sram_a, accelerator_dram_address=0,
                                        element_size=dim, general_reg_src=self.TMP_REG)
        # advance per_layer_input pointer by one row
        self.generate_instruction_add_imm(_pli, ue_35bit_addr_shifter(dim * bpe), _pli)
        self.generate_instruction_add_inc(_r)
        self.loop_end()
        self.release_isa_reg()  # _r
        self.release_isa_reg()  # _pli

        # projected = gated @ per_layer_projection: [S,256] @ [256,1536] -> [S,1536]
        total_flops += self.matmat_mul_core(M=M_tmpl, K=dim, N=N,
            A_DRAM_ADDR=self.LAYER0_PER_LAYER_GATE_OUTPUT_DRAM,
            B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PER_LAYER_PROJ + layer_off,
            OUTPUT_DRAM_ADDR=self.LAYER0_PER_LAYER_PROJ_OUTPUT_DRAM,
            gpr_M_reg=self.gpr_seq_len)

        # FUSED per-row loop: for each of gpr_seq_len rows,
        #   normed = rms_norm(projected_row, gamma)
        #   LAYER0_OUTPUT_row = (hidden_row + normed) * layer_scalar
        # One row resident per iteration; no chunking, no cross-op SRAM reuse.
        proj_dram   = self.LAYER0_PER_LAYER_PROJ_OUTPUT_DRAM
        hidden_dram = self.LAYER0_OUTPUT_DRAM
        gamma_addr  = self.DRAM_ADDR_LAYER0_POST_PER_LAYER_NORM_GAMMA + layer_off
        proj_sram, gamma_sram, hidden_sram = 0x10000, 0x80000, 0x90000
        # gamma resident across all rows
        self.accelerator_memory_to_sram(accelerator_dram_address=gamma_addr,
                                        sram_address=gamma_sram, element_size=N)
        _r = self.alloc_isa_reg()
        self.generate_instruction_add_set(_r, 0)
        self.loop_start(loop_cnt=M_tmpl, gpr_loop_cnt=self.gpr_seq_len)
        # projected row -> proj_sram
        self.generate_instruction_reg_mul_imm(self.TMP_REG, _r, ue_35bit_addr_shifter(N * bpe))
        self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(proj_dram), self.TMP_REG)
        self.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=proj_sram,
                                        element_size=N, general_reg_src=self.TMP_REG)
        # rms_norm(projected) in place with resident gamma
        self.rms_norm_core(proj_sram, proj_sram, N, gamma_sram)
        # hidden row -> hidden_sram
        self.generate_instruction_reg_mul_imm(self.TMP_REG, _r, ue_35bit_addr_shifter(N * bpe))
        self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(hidden_dram), self.TMP_REG)
        self.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=hidden_sram,
                                        element_size=N, general_reg_src=self.TMP_REG)
        # residual add: normed + hidden -> proj_sram
        self.eltwise_add_core(vector_A_sram_start_addr=proj_sram, vector_B_sram_start_addr=hidden_sram,
                              vector_C_sram_wb_addr=proj_sram, element_size=N)
        # * layer_scalar
        self.broadcast_mul(scalar=self._layer_scalars[layer_idx],
                           sram_start_addr=proj_sram, sram_wb_addr=proj_sram, element_size=N)
        # store -> LAYER0_OUTPUT row
        self.generate_instruction_reg_mul_imm(self.TMP_REG, _r, ue_35bit_addr_shifter(N * bpe))
        self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(self.LAYER0_OUTPUT_DRAM), self.TMP_REG)
        self.sram_to_accelerator_memory(sram_address=proj_sram, accelerator_dram_address=0,
                                        element_size=N, general_reg_src=self.TMP_REG)
        self.generate_instruction_add_inc(_r)
        self.loop_end()
        self.release_isa_reg()  # _r
        return total_flops

    def _emit_gqa_duplicate_pbi(self, src_dram_base: int, dst_dram_base: int,
                                cur_head_dim: int, template_seq_len: int,
                                gpr_seq_len: int, sram_addr: int = 0x10000,
                                src_row_bytes: int = None) -> None:
        """§4.4 PBI hardware loop replacing the static per-token GQA replication.
        Per token (outer loop, runtime trip count = gpr_seq_len) read its one
        cur_head_dim row from the KV cache (src stride src_row_bytes, default one
        head; test.py passes k_size) into a fixed SRAM slot, then scatter
        group_size contiguous copies into the flat FLASH buffer via a pbi
        write-pointer that auto-advances one head per call. SRAM-safe (one row
        resident). Output layout: FLASH row (i*group_size + g)."""
        bpe = self.bytes_per_element
        row_bytes = cur_head_dim * bpe          # dst scatter stride (= one head)
        src_stride = src_row_bytes if src_row_bytes is not None else row_bytes
        _, sram_words = self.sram_address_to_uram_address(sram_addr)
        ptr = self.alloc_inst_ptr()
        self.generate_instruction_pbi_init(
            dram_shared_addr=dst_dram_base, dma_length=row_bytes,
            output_size=0, uram_length=0,
            uram_a_start_addr=sram_words, uram_b_start_addr=sram_words,
            uram_wb_addr=0, uram_dst_addr=0, fmax_context_addr=0,
            inst_pointer_idx=ptr)
        t_reg = self.alloc_isa_reg()
        self.generate_instruction_add_set(t_reg, 0)
        self.loop_start(loop_cnt=template_seq_len, gpr_loop_cnt=gpr_seq_len)
        self.generate_instruction_reg_mul_imm(
            self.TMP_REG, t_reg, ue_35bit_addr_shifter(src_stride))
        self.generate_instruction_add_imm(
            self.TMP_REG, ue_35bit_addr_shifter(src_dram_base), self.TMP_REG)
        self.accelerator_memory_to_sram(
            accelerator_dram_address=0, sram_address=sram_addr,
            element_size=cur_head_dim, general_reg_src=self.TMP_REG)
        self.loop_start(self.group_size)
        self.sram_to_accelerator_memory(
            sram_address=0, accelerator_dram_address=row_bytes,
            element_size=cur_head_dim, inst_pointer_idx=ptr,
            memcpy_length_bytes=0)
        self.loop_end()
        self.generate_instruction_add_inc(t_reg)
        self.loop_end()
        self.release_isa_reg()       # t_reg
        self.release_inst_ptr(ptr)

    def _emit_strided_copy_pbi(self, src_base: int, dst_base: int, copy_elems: int,
                               src_row_bytes: int, dst_row_bytes: int,
                               n_template: int, gpr_loop: int,
                               sram_addr: int = 0x10000) -> None:
        """§4.4 PBI hardware loop: for n_template rows (runtime trip count =
        gpr_loop) copy copy_elems bf16 from src to dst, advancing the source DRAM
        addr by src_row_bytes and the dest by dst_row_bytes each iteration (both
        register-computed → arbitrary strides). Used for the partial-rotary
        gather/scatter, the non-rotated-dim pass-through, and spreading a
        contiguous K rope result into the k_size-strided KV cache."""
        i_reg = self.alloc_isa_reg()
        self.generate_instruction_add_set(i_reg, 0)
        self.loop_start(loop_cnt=n_template, gpr_loop_cnt=gpr_loop)
        self.generate_instruction_reg_mul_imm(
            self.TMP_REG, i_reg, ue_35bit_addr_shifter(src_row_bytes))
        self.generate_instruction_add_imm(
            self.TMP_REG, ue_35bit_addr_shifter(src_base), self.TMP_REG)
        self.accelerator_memory_to_sram(
            accelerator_dram_address=0, sram_address=sram_addr,
            element_size=copy_elems, general_reg_src=self.TMP_REG)
        self.generate_instruction_reg_mul_imm(
            self.TMP_REG, i_reg, ue_35bit_addr_shifter(dst_row_bytes))
        self.generate_instruction_add_imm(
            self.TMP_REG, ue_35bit_addr_shifter(dst_base), self.TMP_REG)
        self.sram_to_accelerator_memory(
            sram_address=sram_addr, accelerator_dram_address=0,
            element_size=copy_elems, general_reg_src=self.TMP_REG)
        self.generate_instruction_add_inc(i_reg)
        self.loop_end()
        self.release_isa_reg()       # i_reg

    def compile_prefill(self, seq_len: int, layer_size: int = 35, profile: bool = False) -> tuple[None, int]:
        """Emit ISA for prefill into the currently open capture buffer, sized
        for the ACTUAL prompt ``seq_len`` — no fixed template, no padding.

        The prefill program is compiled per prompt (see compile_gemma4),
        so the per-token Python loops (V-norm, RoPE, K/V dup, partial-rotary
        copies, broadcast_mul) iterate exactly ``seq_len`` times. Bulk PBI ops
        (matmat / rms_norm / eltwise) still read M from gpr_seq_len at execute
        time; run_prefill primes gpr_seq_len / gpr_q_seq_len / gpr_aligned_seq_len
        with this same ``seq_len`` before launching.

        ``profile``: mirror compile_decoder — emit a HALT at each per-layer phase
        boundary and record the resume address (the next instruction), so
        run_gemma4_profile can time each phase's HW latency. Checkpoints are
        placed only at UNCONDITIONAL points (never inside a loop_start/loop_end
        or a per-layer kv-shared branch), so every layer contributes one sample
        per phase. Stored in self._prefill_checkpoints (empty when not profiling).
        A profile-compiled prefill can only be run segment-by-segment (each HALT
        stops the FPGA), never in one shot.
        """
        self.seq_len = seq_len
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        global _SILENT_MODE
        _SILENT_MODE = True
        total_flops = 0
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        _original_print(f"  Emitting prefill: {layer_size} layers, seq_len={seq_len}, attention=unified-inline"
                        + (" (+profile checkpoints)" if profile else ""))
        checkpoints: list[list] = []
        def _checkpoint(name: str) -> None:
            if not profile:
                return
            self.generate_instruction_halt()
            resume = self.get_program_dram_addr() + self.capture_count * INSTRUCTION_SIZE_BYTES
            checkpoints.append([name, f"0x{resume:X}"])
        prefill_t0 = time.perf_counter()
        for layer_idx in range(layer_size):
            if layer_idx > 0 and layer_idx % 10 == 0:
                _original_print(f"    prefill layer {layer_idx}/{layer_size} ({time.perf_counter()-prefill_t0:.1f}s)")
            layer_off = layer_idx * LAYER_WEIGHT_SIZE
            cur_head_dim, cur_q_size, cur_k_size = self._get_layer_attention_dims(layer_idx)
            cur_mlp = self._get_mlp_elements(layer_idx)
            rope_n = self._get_rope_dims(layer_idx)

            # Layer-input source (mirrors compile_decoder): layer 0 reads the
            # uploaded LAYER0_INPUT; layer i>0 reads the previous layer's
            # LAYER0_OUTPUT directly — no seq_len-sized copy. LAYER0_OUTPUT is
            # overwritten only at this layer's final MLP+injection residual,
            # AFTER it is consumed here and as the attention-residual source.
            layer_input_addr = self.LAYER0_INPUT_DRAM if layer_idx == 0 else self.LAYER0_OUTPUT_DRAM
            total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=layer_input_addr,
                                OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off,
                                                gpr_M_reg=self.gpr_seq_len)
            # Q projection: N = cur_q_size (actual per-layer Q output dim)
            total_flops += self.matmat_mul_core(M=seq_len, K=self.vector_length, N=cur_q_size,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                is_B_quantized=True,
                data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off,
                gpr_M_reg=self.gpr_seq_len,
                )
            if layer_idx not in self._kv_shared_map:
                # Non-shared layer: compute K/V projections normally.
                # Shared layers skip entirely — their attention reads K/V directly
                # from the reference layer's slot via _kv_slot_for_layer.
                # K projection: N = cur_k_size (actual per-layer K output dim)
                total_flops += self.matmat_mul_core(M=seq_len, K=self.vector_length, N=cur_k_size,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off,
                    gpr_M_reg=self.gpr_seq_len,
                    )
                # V projection: write to temp buffer first, then scatter to KV cache at k_size stride
                total_flops += self.matmat_mul_core(M=seq_len, K=self.vector_length, N=cur_k_size,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,  # temp buffer
                    is_B_quantized=True,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off,
                    gpr_M_reg=self.gpr_seq_len,
                    )
                # V norm + scatter to KV cache at k_size stride — §4.4 PBI loop
                # (was a per-token Python unroll). rms_norm_core works on a fixed
                # SRAM slot; the per-token read (FLASH_V, cur_k_size stride) and
                # write (KV cache, k_size stride) addrs are register-computed, so
                # the body is emitted once and hardware-looped over gpr_seq_len.
                v_cache_base = self.LAYER0_V_DRAM + self._kv_slot_for_layer[layer_idx] * self.MAX_CONTEXT_SIZE * self.k_size
                _vi = self.alloc_isa_reg()
                self.generate_instruction_add_set(_vi, 0)
                self.loop_start(loop_cnt=seq_len, gpr_loop_cnt=self.gpr_seq_len)
                self.generate_instruction_reg_mul_imm(self.TMP_REG, _vi, ue_35bit_addr_shifter(cur_k_size * self.bytes_per_element))
                self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(self.LAYER0_FLASH_V_DRAM), self.TMP_REG)
                self.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=0x10000, element_size=cur_k_size, general_reg_src=self.TMP_REG)
                self.rms_norm_core(0x10000, 0x10000, cur_k_size)  # no gamma
                self.generate_instruction_reg_mul_imm(self.TMP_REG, _vi, ue_35bit_addr_shifter(self.k_size))
                self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(v_cache_base), self.TMP_REG)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=0, element_size=cur_k_size, general_reg_src=self.TMP_REG)
                self.generate_instruction_add_inc(_vi)
                self.loop_end()
                self.release_isa_reg()  # _vi

            # Q norm always needed (Q is always computed fresh)
            # Q-norm: M = seq_len * group_size → use gpr_q_seq_len
            total_flops += self.rms_norm_core_dram(M=seq_len * self.group_size, N=cur_head_dim, A_DRAM_ADDR=self.LAYER0_Q_DRAM,
                            OUTPUT_DRAM_ADDR=self.LAYER0_Q_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_NORM_GAMMA + layer_off,
                                            gpr_M_reg=self.gpr_q_seq_len)
            _checkpoint(f"L{layer_idx}_qkv_vproj")

            ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE_GLOBAL if layer_idx in self._rope_global_layers else self.DRAM_ADDR_ROPE_LOCAL

            # §4.4 PBI-loop RoPE + GQA replication (replaces the per-token /
            # per-token×group Python unrolls — the dominant prefill-bin bloat).
            # cos/sin layout (per-token contiguous, sin half follows cos, stride
            # 2*rope_n*bpe) already matches rope_*_pbi. K rope result is built
            # CONTIGUOUS (cur_head_dim stride) in MLP scratch, then spread into
            # the k_size-strided KV cache; Q goes straight to cur_head_dim-
            # contiguous FLASH_Q. Sliding layers are full-rotary (one rope call);
            # global layers are partial-rotary (gather→rope→scatter→copy).
            bpe = self.bytes_per_element
            head_bytes = cur_head_dim * bpe
            rope_bytes = rope_n * bpe
            sin_addr = ROPE_WEIGHT_ADDR + rope_n * bpe
            q_rows = seq_len * self.group_size
            kv_slot_off = self._kv_slot_for_layer[layer_idx] * self.MAX_CONTEXT_SIZE * self.k_size
            k_cache_base = self.LAYER0_K_ROPE_DRAM + kv_slot_off
            v_cache_base = self.LAYER0_V_DRAM + kv_slot_off
            K_TMP = self.LAYER0_MLP_MULT_DRAM     # contiguous K rope scratch
            tmp_in = self.LAYER0_MLP_GATE_DRAM    # gather/rope scratch (dead during attn,
            tmp_out = self.LAYER0_MLP_UP_DRAM     #  overwritten by the real MLP later)
            non_shared = layer_idx not in self._kv_shared_map
            if non_shared:
                # K norm (non-shared layers own their KV slot).
                total_flops += self.rms_norm_core_dram(M=seq_len, N=cur_head_dim, A_DRAM_ADDR=self.LAYER0_K_DRAM,
                                OUTPUT_DRAM_ADDR=self.LAYER0_K_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_NORM_GAMMA + layer_off,
                                gpr_M_reg=self.gpr_seq_len)
            if rope_n == cur_head_dim:
                # Full rotary: single PBI rope call (input/output contiguous).
                if non_shared:
                    total_flops += self.rope_hf_core_dram(M=seq_len, N=rope_n,
                        input_dram_addr=self.LAYER0_K_NORM_DRAM, output_dram_addr=K_TMP,
                        cos_dram_addr=ROPE_WEIGHT_ADDR, sin_dram_addr=sin_addr, gpr_M_reg=self.gpr_seq_len)
                total_flops += self.rope_hf_core_dram_gqa(M=seq_len, group_size=self.group_size, N=rope_n,
                    input_dram_addr=self.LAYER0_Q_NORM_DRAM, output_dram_addr=self.LAYER0_FLASH_Q_DRAM,
                    cos_dram_addr=ROPE_WEIGHT_ADDR, sin_dram_addr=sin_addr, gpr_M_reg=self.gpr_seq_len)
            else:
                # Partial rotary: rotate first rope_n dims, copy the rest through.
                non_rot = cur_head_dim - rope_n
                if non_shared:
                    self._emit_strided_copy_pbi(self.LAYER0_K_NORM_DRAM, tmp_in, rope_n, head_bytes, rope_bytes, seq_len, self.gpr_seq_len)
                    total_flops += self.rope_hf_core_dram(M=seq_len, N=rope_n, input_dram_addr=tmp_in, output_dram_addr=tmp_out, cos_dram_addr=ROPE_WEIGHT_ADDR, sin_dram_addr=sin_addr, gpr_M_reg=self.gpr_seq_len)
                    self._emit_strided_copy_pbi(tmp_out, K_TMP, rope_n, rope_bytes, head_bytes, seq_len, self.gpr_seq_len)
                    self._emit_strided_copy_pbi(self.LAYER0_K_NORM_DRAM + rope_bytes, K_TMP + rope_bytes, non_rot, head_bytes, head_bytes, seq_len, self.gpr_seq_len)
                self._emit_strided_copy_pbi(self.LAYER0_Q_NORM_DRAM, tmp_in, rope_n, head_bytes, rope_bytes, q_rows, self.gpr_q_seq_len)
                total_flops += self.rope_hf_core_dram_gqa(M=seq_len, group_size=self.group_size, N=rope_n, input_dram_addr=tmp_in, output_dram_addr=tmp_out, cos_dram_addr=ROPE_WEIGHT_ADDR, sin_dram_addr=sin_addr, gpr_M_reg=self.gpr_seq_len)
                self._emit_strided_copy_pbi(tmp_out, self.LAYER0_FLASH_Q_DRAM, rope_n, rope_bytes, head_bytes, q_rows, self.gpr_q_seq_len)
                self._emit_strided_copy_pbi(self.LAYER0_Q_NORM_DRAM + rope_bytes, self.LAYER0_FLASH_Q_DRAM + rope_bytes, non_rot, head_bytes, head_bytes, q_rows, self.gpr_q_seq_len)
            # Spread the contiguous K rope result into the k_size-strided KV cache.
            if non_shared:
                self._emit_strided_copy_pbi(K_TMP, k_cache_base, cur_head_dim, head_bytes, self.k_size, seq_len, self.gpr_seq_len)
            _checkpoint(f"L{layer_idx}_rope")
            # GQA dup: read KV cache (k_size stride) → scatter group_size copies to FLASH (cur_head_dim stride).
            self._emit_gqa_duplicate_pbi(k_cache_base, self.LAYER0_FLASH_K_DRAM, cur_head_dim, seq_len, self.gpr_seq_len, src_row_bytes=self.k_size)
            self._emit_gqa_duplicate_pbi(v_cache_base, self.LAYER0_FLASH_V_DRAM, cur_head_dim, seq_len, self.gpr_seq_len, src_row_bytes=self.k_size)
            _checkpoint(f"L{layer_idx}_kv_gather")

            # Gemma4 uses scaling=1.0 (no 1/sqrt(d) in attention scores), so pass
            # q_scale=1.0 below; no Q pre-scale needed.

            # Pick the per-layer bias: full attention layers see the
            # entire causal window; sliding-attention layers are limited
            # to `sliding_window` tokens (run_prefill builds both biases).
            bias_addr_layer = (self.LAYER0_FLASH_BIAS_FULL_DRAM
                               if layer_idx in self._full_attention_layers
                               else self.LAYER0_FLASH_BIAS_SLIDING_DRAM)
            # unified_attention_core uses bias_mode="full_matrix" internally.
            # Prefill bias is [aligned_q, aligned_q], while the dynamic batch
            # GPR limits the live rows to q_seq_len.
            total_flops += self.unified_attention_core(
                batch=aligned_seq_len,
                aligned_seq_len=aligned_seq_len,
                head_dim=cur_head_dim,
                Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
                K_DRAM_ADDR=self.LAYER0_FLASH_K_DRAM,
                V_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                BIAS_DRAM_ADDR=bias_addr_layer,
                OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                IDENTITY_DRAM_ADDR=self.IDENTITY_DRAM_ADDR,
                gpr_batch_reg=self.gpr_q_seq_len,
                gpr_aligned_seq_len_reg=self.gpr_aligned_seq_len,
                q_scale=1.0,
            )
            _checkpoint(f"L{layer_idx}_attention")
            # O projection: INT4, K=cur_q_size
            total_flops += self.matmat_mul_core(M=seq_len, K=cur_q_size, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                is_B_quantized=True,
                data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off,
                gpr_M_reg=self.gpr_seq_len,
                )
            total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                            OUTPUT_DRAM_ADDR=self.LAYER0_POST_ATTN_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_NORM_GAMMA + layer_off,
                                            gpr_M_reg=self.gpr_seq_len)
            self.eltwise_core_dram(
                M=seq_len, N=self.vector_length,
                dram_a=layer_input_addr, dram_b=self.LAYER0_POST_ATTN_NORM_DRAM,
                dram_out=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=self.gpr_seq_len)
            _checkpoint(f"L{layer_idx}_o_proj")
            total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                            OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off,
                                            gpr_M_reg=self.gpr_seq_len)
            # MLP gate (fused GELU) + up projections.
            total_flops += self.matmat_mul_core(M=seq_len, K=self.vector_length, N=cur_mlp,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                is_B_quantized=True,
                data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off,
                gelu_enable=True,
                gpr_M_reg=self.gpr_seq_len,
                )
            total_flops += self.matmat_mul_core(M=seq_len, K=self.vector_length, N=cur_mlp,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                is_B_quantized=True,
                data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off,
                gpr_M_reg=self.gpr_seq_len,
                )
            self.eltwise_core_dram(
                M=seq_len, N=cur_mlp,
                dram_a=self.LAYER0_MLP_GATE_DRAM, dram_b=self.LAYER0_MLP_UP_DRAM,
                dram_out=self.LAYER0_MLP_MULT_DRAM,
                mode=UE_MODE.ELTWISE_MUL, gpr_M_reg=self.gpr_seq_len)
            # MLP down projection.
            total_flops += self.matmat_mul_core(M=seq_len, K=cur_mlp, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                is_B_quantized=True,
                data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off,
                gpr_M_reg=self.gpr_seq_len,
                )
            total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                            OUTPUT_DRAM_ADDR=self.LAYER0_POST_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_FFW_NORM_GAMMA + layer_off,
                                            gpr_M_reg=self.gpr_seq_len)
            self.eltwise_core_dram(
                M=seq_len, N=self.vector_length,
                dram_a=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, dram_b=self.LAYER0_POST_MLP_NORM_DRAM,
                dram_out=self.LAYER0_OUTPUT_DRAM,
                mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=self.gpr_seq_len)
            _checkpoint(f"L{layer_idx}_mlp")

            # Per-layer input injection (NEW for Gemma4 E2B) — seq_len-agnostic
            # (gpr_seq_len-driven) prefill variant; decode uses the seq_len=1 one.
            total_flops += self._compile_per_layer_injection_prefill(layer_idx, layer_off)
            _checkpoint(f"L{layer_idx}_inject")

        self.generate_instruction_halt()
        self._prefill_checkpoints = checkpoints
        _SILENT_MODE = False
        return None, total_flops

    def _compute_per_layer_inputs(self, token_ids, embedding_tensor: torch.Tensor) -> torch.Tensor:
        """Compute per-layer inputs on host side.
        Args:
            token_ids: token id sequence (list or tuple)
            embedding_tensor: (seq_len, hidden_size) bf16 tensor (already scaled)
        Returns:
            per_layer_inputs: (seq_len, LAYER_SIZE, per_layer_input_dim) bf16 tensor
        """
        def _host_rms_norm(x, gamma, eps=1e-6):
            """RMS norm on host (for the per-layer projection norm)."""
            x_f = x.float()
            rms = (x_f ** 2).mean(dim=-1, keepdim=True).add(eps).sqrt()
            return ((x_f / rms) * gamma.float()).to(x.dtype)

        seq_len = len(token_ids)
        tid_t = torch.tensor(token_ids, dtype=torch.long)

        # Multimodal mode: image/audio soft-token rows already carry the real
        # modality embedding in embedding_tensor. Use pad_token_id for the
        # per-layer token lookup so placeholder IDs do not inject extra text
        # token-specific per-layer embeddings.
        if hasattr(self, '_mm_types') and self._mm_types is not None:
            mm_mask = torch.tensor(self._mm_types[:len(token_ids)])
            tid_t_for_pli = tid_t.clone()
            tid_t_for_pli[(mm_mask == 1) | (mm_mask == 3)] = 0  # pad_token_id
        else:
            tid_t_for_pli = tid_t

        # per_layer_embed: lookup from embed_tokens_per_layer [262144, 8960] -> [seq_len, 8960] -> [seq_len, 35, 256]
        per_layer_embed = self.embed_tokens_per_layer_weight[tid_t_for_pli]  # [seq_len, 8960]
        per_layer_embed = per_layer_embed.reshape(seq_len, self.LAYER_SIZE, self.per_layer_input_dim)  # [seq_len, 35, 256]

        # per_layer_proj: (per_layer_model_projection @ embedding.T).T
        # per_layer_model_proj_weight is [8960, 1536], embedding_tensor is [seq_len, 1536]
        # We want [seq_len, 8960] = embedding_tensor @ per_layer_model_proj_weight.T
        # But need to undo the embedding scale first: use the unscaled embedding
        # Actually the spec says to use the embedding_tensor (already scaled), and apply proj_scale
        per_layer_proj = (embedding_tensor.float() @ self.per_layer_model_proj_weight.float().T)  # [seq_len, 8960]
        per_layer_proj = (per_layer_proj * self._per_layer_model_proj_scale).to(torch.bfloat16)
        per_layer_proj = per_layer_proj.reshape(seq_len, self.LAYER_SIZE, self.per_layer_input_dim)  # [seq_len, 35, 256]

        # rms_norm per_layer_proj along last dim with per_layer_proj_norm_weight
        per_layer_proj = _host_rms_norm(per_layer_proj, self.per_layer_proj_norm_weight)

        # per_layer_inputs = (per_layer_proj + per_layer_embed) * per_layer_input_scale
        per_layer_inputs = ((per_layer_proj.float() + per_layer_embed.float()) * self._per_layer_input_scale).to(torch.bfloat16)

        return per_layer_inputs  # [seq_len, 35, 256]

    def run_prefill(self, prefill_program_addr: int, prefill_seq=None, flops: int = None,
                    profile_checkpoints: list | None = None):
        """
        Run prefill for the actual prompt — single entry, no bucket/padding.

        The prefill program at ``prefill_program_addr`` must have been compiled
        for this exact prompt length via compile_gemma4 and loaded into program
        DRAM by run_gemma4. This method restores clean FPGA state, uploads the
        prompt's embeddings / per-layer inputs / attention bias, primes the
        dynamic gpr registers, and launches the program.

        Args:
            prefill_program_addr: DRAM address of the compiled prefill program.
            prefill_seq: Full prompt token tuple (incl. the final token, which
                is NOT processed by prefill); if None, uses self.prefill_seq.
            flops: FLOP count for the HW rate counter.
            profile_checkpoints: when given (from a profile-compiled image), run
                the prefill segment-by-segment through its HALT checkpoints and
                return the per-segment [(name, ms)] HW latencies INSTEAD of a
                one-shot execute. The KV cache is still fully populated. Used by
                run_gemma4_profile.

        Returns:
            (latency, flop_rate) from the accelerator, or — when
            ``profile_checkpoints`` is given — the per-segment latency list.
        """
        if prefill_seq is None:
            prefill_seq = self.prefill_seq
        if prefill_seq is None:
            prefill_seq = tuple(self._cfg["default_prefill_tokens"])
        if len(prefill_seq) < 2:
            raise ValueError("Prefill sequence must have at least 2 tokens.")
        # Prefill processes all but the last token.
        prefill_seq = tuple(prefill_seq[:-1])
        seq_len = len(prefill_seq)
        assert seq_len <= self.max_prefill_seq_len, (
            f"Prefill length {seq_len} exceeds max_prefill_seq_len {self.max_prefill_seq_len}. "
            f"Bump 'max_prefill_seq_len' in gemma4_e2b_config.json (and raise _tensor_estimate "
            f"in __init__ accordingly) to support longer prompts."
        )
        # The compiled program's static per-token loops were emitted at this
        # seq_len; keep self.seq_len consistent for the dynamic-gpr preamble and
        # for run_decoder, which resumes from this position.
        self.seq_len = seq_len
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64

        # Restore clean FPGA state before this prefill (formerly in
        # run_prefill_bucketed): zero the entire K/V cache so decode's
        # bias-masked reads past seq_len see clean zeros, zero the attention
        # Q/K/V gather buffers, and re-upload the IDENTITY matrix (read by the
        # attention I @ V^T step). Idempotent for LM-only runs.
        from user_dma_core import UE_VECTOR_SIZE as _UE_VS
        num_slots = getattr(self, "_num_kv_slots", self.LAYER_SIZE)
        kv_slot_elems = num_slots * self.MAX_CONTEXT_SIZE * self.head_dim
        kv_zero_pad = torch.zeros(kv_slot_elems, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, kv_zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_K_ROPE_DRAM, kv_zero_pad)
        _pre_align = ((self.max_prefill_seq_len * self.group_size + 63) // 64) * 64
        flash_qkv_elems = _pre_align * self.head_dim
        _flash_zero = torch.zeros(flash_qkv_elems, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_Q_DRAM, _flash_zero)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_K_DRAM, _flash_zero)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_V_DRAM, _flash_zero)
        self.dma_to_accelerator_memory(self.IDENTITY_DRAM_ADDR,
                                       torch.eye(_UE_VS, dtype=torch.bfloat16))
        print(f"[Prefill] LM-state restored ({num_slots} KV slots zeroed, IDENTITY re-uploaded)")

        # --- prefill profiler: two parts, wall-clock only. host_prepare covers
        # all host-side work (embedding lookup, per-layer inputs, bias build)
        # plus the DMAs that feed the FPGA; fpga_execute is the accelerator run.
        # Summary printed at the end; set GEMMA4_PROFILE=0 to silence it.
        _host_prepare_w0 = time.perf_counter()

        print(f"[Prefill] [host] looking up token embeddings for {seq_len} tokens...", flush=True)
        embedding_tensor = self.get_embedding_for_tokens(prefill_seq)

        # Multimodal merge: replace image/audio placeholder embeddings with
        # encoder-produced soft-token features. Uses mm_token_type_ids where
        # 1=image, 3=audio (HF convention, see transformers/processing_utils.py
        # create_mm_token_type_ids).
        if hasattr(self, '_mm_types') and self._mm_types is not None:
            mm_types = torch.tensor(self._mm_types[:len(prefill_seq)])
            if hasattr(self, '_image_features') and self._image_features is not None:
                image_mask = (mm_types == 1)
                embedding_tensor[image_mask] = self._image_features[:image_mask.sum()].to(embedding_tensor.dtype)
                print(f"[Prefill] merged {image_mask.sum().item()} image features into embeddings")
            if hasattr(self, '_audio_features') and self._audio_features is not None:
                audio_mask = (mm_types == 3)
                embedding_tensor[audio_mask] = self._audio_features[:audio_mask.sum()].to(embedding_tensor.dtype)
                print(f"[Prefill] merged {audio_mask.sum().item()} audio features into embeddings")

        print(f"[Prefill] uploading embeddings to FPGA DRAM...", flush=True)
        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)

        # Compute per-layer inputs on host and DMA to FPGA
        print(f"[Prefill] [host] computing per-layer inputs ({seq_len} tokens x {self.LAYER_SIZE} layers)...", flush=True)
        per_layer_inputs = self._compute_per_layer_inputs(prefill_seq, embedding_tensor)  # [seq_len, 35, 256]
        # Permute to [35, seq_len, 256] so each layer's data is contiguous in DRAM
        per_layer_inputs_flat = per_layer_inputs.permute(1, 0, 2).contiguous()  # [35, seq_len, 256]
        print(f"[Prefill] uploading per-layer inputs to FPGA DRAM...", flush=True)
        self.dma_to_accelerator_memory(self.PER_LAYER_INPUTS_DRAM, per_layer_inputs_flat)

        # Clear multimodal state now that prefill's per-layer-inputs have been computed.
        # Decode reuses _compute_per_layer_inputs with seq_len=1, and if _mm_types
        # is still set it would incorrectly treat every decode token as a
        # multimodal position (replacing its ID with pad_token_id=0 for
        # per_layer_embed lookup),
        # producing garbage per-layer injection and all-pad output. Mirror the
        # compare script's pattern: clear right after use.
        self._mm_types = None
        self._image_features = None
        self._audio_features = None

        # Build BOTH prefill bias matrices: full (causal) for full-attention
        # layers, and sliding (causal AND within `sliding_window` tokens) for
        # sliding-attention layers. compile_prefill picks per-layer.
        # Both biases are in q_seq_len space (each token has group_size query
        # heads, K is GQA-duplicated to match), so the window is converted
        # from token space to q-position space by multiplying by group_size.
        full_bias = torch.full((aligned_seq_len, aligned_seq_len), float("-inf"), dtype=torch.bfloat16)
        # Q rows are laid out token*group_size + head, and K is GQA-duplicated
        # to the same layout, so each attn head must attend ONLY its own head
        # slot (its KV head), causal in tokens. A flat q-space causal tril
        # (j<=i) additionally allows CROSS-HEAD attention (head h attending
        # earlier heads 0..h within a token). For E2B (num_kv=1) this is one KV
        # group so it doesn't corrupt the VALUE, but it still over-weights
        # earlier tokens (each contributes group_size duplicate slots vs the
        # current token's h+1), so the correct mask is same head AND
        # token-causal. (For E4B num_kv=2 the flat tril was catastrophic — it
        # let heads attend the WRONG KV group; see gemma4_e4b_test.py.)
        _gs = self.group_size
        _i = torch.arange(aligned_seq_len).unsqueeze(1)
        _j = torch.arange(aligned_seq_len).unsqueeze(0)
        _same_head = (_i % _gs) == (_j % _gs)
        _tok_causal = ((_j // _gs) <= (_i // _gs)) if not self.causal_mask_upper else ((_j // _gs) >= (_i // _gs))
        valid_mask = _same_head & _tok_causal
        full_bias.masked_fill_(valid_mask, 0.0)
        full_bias[:, q_seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_FULL_DRAM, full_bias)

        # Sliding bias is identical to full when seq_len ≤ sliding_window;
        # otherwise it additionally masks anything older than the window.
        if seq_len <= self.sliding_window:
            sliding_bias = full_bias
        else:
            window_q = self.sliding_window * self.group_size
            sliding_bias = torch.full((aligned_seq_len, aligned_seq_len), float("-inf"), dtype=torch.bfloat16)
            i_idx = torch.arange(aligned_seq_len).unsqueeze(1)
            j_idx = torch.arange(aligned_seq_len).unsqueeze(0)
            i_token = i_idx // self.group_size
            j_token = j_idx // self.group_size
            in_window = (i_token - j_token) < self.sliding_window
            sliding_mask = valid_mask & in_window
            sliding_bias.masked_fill_(sliding_mask, 0.0)
            sliding_bias[:, q_seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_SLIDING_DRAM, sliding_bias)

        # End of host prep — everything above ran on the host (CPU + DMA to feed
        # the FPGA); everything below is the accelerator run.
        host_prepare_s = time.perf_counter() - _host_prepare_w0

        # Profiling path: run the prefill through its per-phase HALT checkpoints
        # and return per-segment HW latencies. Populates the KV cache exactly
        # like a straight run (segments tile the whole program). The dynamic-PBI
        # preamble primes the same three GPRs as the one-shot dispatch below.
        if profile_checkpoints is not None:
            print(f"[Prefill] [profile] running {len(profile_checkpoints)} segments "
                  f"(host prep {host_prepare_s:.2f}s)...", flush=True)
            return self._profile_execute(
                [(self.gpr_seq_len,         seq_len),
                 (self.gpr_q_seq_len,       q_seq_len),
                 (self.gpr_aligned_seq_len, aligned_seq_len)],
                prefill_program_addr, profile_checkpoints, tail_name="tail_halt")

        print(f"[Prefill] [exec] launching prefill program on FPGA ({seq_len} tokens, {self.LAYER_SIZE} layers)...", flush=True)
        # Heartbeat thread: program_execute blocks until the FPGA halts, with no
        # intermediate visibility. Print elapsed seconds every 10s so the user
        # sees liveness during the ~30-60s prefill execution.
        import threading
        _pf_t0 = time.perf_counter()
        _pf_stop = threading.Event()
        def _pf_hb():
            while not _pf_stop.wait(10):
                print(f"[Prefill] [exec]   ... still running on FPGA ({time.perf_counter()-_pf_t0:.0f}s elapsed)", flush=True)
        _pf_th = threading.Thread(target=_pf_hb, daemon=True)
        _pf_th.start()
        try:
            # Dynamic-PBI dispatch: a single preamble primes seq_len / Q rows /
            # aligned attention length, then jumps into the cached prefill
            # program (gemma3 pattern). Building this at the fixed _preamble_addr
            # (past every program) is what keeps the gpr priming from clobbering
            # the prefill body.
            latency, flop_rate_program = self._dispatch_program(
                [(self.gpr_seq_len,         seq_len),
                 (self.gpr_q_seq_len,       q_seq_len),
                 (self.gpr_aligned_seq_len, aligned_seq_len)],
                prefill_program_addr, timeout=300.0, flops=flops)
        finally:
            _pf_stop.set()
            _pf_th.join(timeout=1.0)
        fpga_execute_s = time.perf_counter() - _pf_t0

        # Profile summary: two parts, wall-clock only. host_prepare is host-side
        # work (torch compute + DMA); fpga_execute is the accelerator run.
        # Set GEMMA4_PROFILE=0 to silence.
        if os.environ.get("GEMMA4_PROFILE", "1") == "1":
            total_s = host_prepare_s + fpga_execute_s
            print("[Prefill] [profile] wall-clock time:", flush=True)
            print(f"[Prefill] [profile]   host_prepare  {host_prepare_s:8.3f}s")
            print(f"[Prefill] [profile]   fpga_execute  {fpga_execute_s:8.3f}s")
            print(f"[Prefill] [profile]   TOTAL         {total_s:8.3f}s", flush=True)
        return latency, flop_rate_program

    def compile_decoder(self, layer_size: int = 35, profile: bool = False) -> tuple[None, list[int], list[int]]:
        """Compile a single decoder program with dynamic PBI.

        DYNAMIC PBI (see notes_gemma4_e2b.md): one captured
        program handles all decode positions. Per-token KV/RoPE addresses
        are computed at execute time via reg_mul_imm(gpr_seq_len, stride) +
        add_imm(base) → TMP_REG. Decoder attention calls unified_attention_core
        inline with batch=group_size and dynamic aligned KV length. End of
        program issues add_inc(gpr_seq_len) so subsequent decode steps advance
        automatically.

        Returns (None, [program_size_bytes], [total_flops]) — backward-compat
        single-element lists; caller uses [0] index.
        """
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]

        global _SILENT_MODE
        _SILENT_MODE = True
        _original_print(f"  Emitting dynamic-PBI decoder: 1 segment x {layer_size} layers, attention=unified-inline")
        seg_t0 = time.perf_counter()
        count_at_start = self.capture_count
        total_flops = 0

        # Optional per-phase profiling (see run_gemma4_profile / --profile).
        # _checkpoint emits a HALT at a phase boundary and records the resume
        # address (the next instruction). At runtime the profiler runs each
        # segment to its HALT, reads the HW latency counter, then resumes. Only
        # placed at UNCONDITIONAL points so every layer contributes one sample
        # per phase (never inside a loop_start/loop_end or a per-layer branch).
        checkpoints: list[list] = []
        def _checkpoint(name: str) -> None:
            if not profile:
                return
            self.generate_instruction_halt()
            resume = self.get_program_dram_addr() + self.capture_count * INSTRUCTION_SIZE_BYTES
            checkpoints.append([name, f"0x{resume:X}"])
        # gpr_one holds the constant 1 — used as gpr_M_reg for all M=1 ops.
        gpr_one = self.alloc_isa_reg()
        self.generate_instruction_add_set(gpr_one, 1)
        gpr_group_size = self.alloc_isa_reg()
        self.generate_instruction_add_set(gpr_group_size, self.group_size)
        # Iterate once (no bucket loop)
        for _bi_unused in [0]:
            seq_len = self.MAX_CONTEXT_SIZE  # template only — for FLOPs
            for layer_idx in range(layer_size):
                layer_off = layer_idx * LAYER_WEIGHT_SIZE
                cur_head_dim, cur_q_size, cur_k_size = self._get_layer_attention_dims(layer_idx)
                cur_mlp = self._get_mlp_elements(layer_idx)
                rope_n = self._get_rope_dims(layer_idx)

                # Layer-input source:
                #   layer 0: LAYER0_INPUT_DRAM (uploaded by run_decoder each step)
                #   layer i>0: LAYER0_OUTPUT_DRAM (written by the previous layer's
                #     per_layer_injection). No copy needed — LAYER0_OUTPUT_DRAM is
                #     only overwritten at the end of the current layer (in the
                #     MLP residual add at line ~2955), which happens AFTER we
                #     consume it as the attention-residual source. So reading it
                #     here and for the attention residual below is safe.
                layer_input_addr = self.LAYER0_INPUT_DRAM if layer_idx == 0 else self.LAYER0_OUTPUT_DRAM
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=layer_input_addr,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off,
                              gpr_M_reg=gpr_one)
                # Q/K/V projections: use per-layer dims
                total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=cur_q_size,
                                                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                                                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off,
                                                    OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                                                    data_type=TYPE.IF4,
                                                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off,
                                                    )
                if layer_idx in self._kv_shared_map:
                    ref_layer = self._kv_shared_map[layer_idx]
                    kv_layer_for_attn = ref_layer  # read from reference layer's KV cache
                else:
                    kv_layer_for_attn = layer_idx  # read from own KV cache
                    # K projection
                    total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=cur_k_size,
                        A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                        B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off,
                        OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                        data_type=TYPE.IF4,
                        SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off,
                        )
                    # V projection
                    total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=cur_k_size,
                        A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                        B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off,
                        OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                        data_type=TYPE.IF4,
                        SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off,
                        )
                    self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_FLASH_V_DRAM, sram_address=0x10000, element_size=cur_k_size)
                    # V norm (Gemma4: normalize V without learnable scale)
                    self.rms_norm_core(0x10000, 0x10000, cur_k_size)  # no gamma
                    # V scatter to V cache at decode_pos via reg_mul_imm + add_imm.
                    # Address = LAYER0_V_DRAM + slot * MAX_CTX * k_size + gpr_seq_len * k_size.
                    _v_slot_base = self.LAYER0_V_DRAM + self._kv_slot_for_layer[layer_idx] * self.MAX_CONTEXT_SIZE * self.k_size
                    self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gpr_seq_len, ue_35bit_addr_shifter(self.k_size))
                    self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(_v_slot_base), self.TMP_REG)
                    self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=0, element_size=cur_k_size, general_reg_src=self.TMP_REG)
                    # RMS norm on K
                    total_flops += self.rms_norm_core_dram(M=1, N=cur_head_dim, A_DRAM_ADDR=self.LAYER0_K_DRAM,
                                  OUTPUT_DRAM_ADDR=self.LAYER0_K_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_NORM_GAMMA + layer_off,
                                  gpr_M_reg=gpr_one)

                _checkpoint(f"L{layer_idx}_qkv_vproj")

                # Q norm: M = group_size (compile-time constant). Use legacy
                # static-M path (no gpr_M_reg) since group_size doesn't vary.
                total_flops += self.rms_norm_core_dram(M=self.group_size, N=cur_head_dim, A_DRAM_ADDR=self.LAYER0_Q_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_Q_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_NORM_GAMMA + layer_off)

                ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE_GLOBAL if layer_idx in self._rope_global_layers else self.DRAM_ADDR_ROPE_LOCAL
                rope_row = 2 * rope_n * self.bytes_per_element  # full cos+sin pair stride per token (rope_n*2 values * 2 bytes)

                kv_slot_off_local = self._kv_slot_for_layer[layer_idx] * self.MAX_CONTEXT_SIZE * self.k_size
                k_rope_base = self.LAYER0_K_ROPE_DRAM + kv_slot_off_local

                if layer_idx not in self._kv_shared_map:
                    # K-RoPE at decode_pos: cos/sin = ROPE_WEIGHT_ADDR + gpr_seq_len * rope_row.
                    # IMPORTANT: write RoPE output to LAYER0_K_DRAM (scratch buffer), NOT to
                    # k_rope_base (= cache position 0). Writing to position 0 would corrupt
                    # the first prefill token's K every decode step.
                    self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gpr_seq_len, ue_35bit_addr_shifter(rope_row))
                    self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(ROPE_WEIGHT_ADDR), self.TMP_REG)
                    total_flops += self.rope_hf_core_decode(
                        N=rope_n,
                        input_dram_addr=self.LAYER0_K_NORM_DRAM,
                        output_dram_addr=self.LAYER0_K_DRAM,         # scratch, not cache
                        gr_weight_dram=self.TMP_REG)
                    # Copy rotated dims from scratch into K cache at decode_pos.
                    self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gpr_seq_len, ue_35bit_addr_shifter(self.k_size))
                    self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(k_rope_base), self.TMP_REG)
                    self.accelerator_memcpy(self.LAYER0_K_DRAM, 0, rope_n * self.bytes_per_element, gr_dst_addr=self.TMP_REG)

                # Q-RoPE: same cos/sin address for all group_size heads (same decode_pos)
                self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gpr_seq_len, ue_35bit_addr_shifter(rope_row))
                self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(ROPE_WEIGHT_ADDR), self.TMP_REG)
                for g in range(self.group_size):
                    total_flops += self.rope_hf_core_decode(
                        N=rope_n,
                        input_dram_addr=self.LAYER0_Q_NORM_DRAM + g * cur_head_dim * self.bytes_per_element,
                        output_dram_addr=self.LAYER0_FLASH_Q_DRAM + g * cur_head_dim * self.bytes_per_element,
                        gr_weight_dram=self.TMP_REG)

                # Partial-rotary non-rotated dims (full-attention layers only).
                if layer_idx in self._full_attention_layers and rope_n < cur_head_dim:
                    remaining = cur_head_dim - rope_n
                    # Q non-rotated dims: static addresses (per-group).
                    for g in range(self.group_size):
                        src = self.LAYER0_Q_NORM_DRAM + g * cur_head_dim * self.bytes_per_element + rope_n * self.bytes_per_element
                        dst = self.LAYER0_FLASH_Q_DRAM + g * cur_head_dim * self.bytes_per_element + rope_n * self.bytes_per_element
                        self.accelerator_memory_to_sram(src, 0x10000, remaining)
                        self.sram_to_accelerator_memory(0x10000, dst, remaining)
                    if layer_idx not in self._kv_shared_map:
                        # K non-rotated dims at decode_pos: cache_addr = k_rope_base + gpr_seq_len * k_size + rope_n_bytes
                        src = self.LAYER0_K_NORM_DRAM + rope_n * self.bytes_per_element
                        k_cache_nrot_base = k_rope_base + rope_n * self.bytes_per_element
                        self.accelerator_memory_to_sram(src, 0x10000, remaining)
                        self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gpr_seq_len, ue_35bit_addr_shifter(self.k_size))
                        self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(k_cache_nrot_base), self.TMP_REG)
                        self.sram_to_accelerator_memory(0x10000, 0, remaining, general_reg_src=self.TMP_REG)

                # Gemma4 uses scaling=1.0; q_scale=1.0 on the attention call means no
                # Q pre-scale is needed here.

                _checkpoint(f"L{layer_idx}_rope")

                # K/V cache reads — KV-shared layers point at the source layer's cache.
                kv_slot_off_read = self._kv_slot_for_layer[kv_layer_for_attn] * self.MAX_CONTEXT_SIZE * self.k_size
                kv_k_base = self.LAYER0_K_ROPE_DRAM + kv_slot_off_read
                kv_v_base = self.LAYER0_V_DRAM + kv_slot_off_read

                # Gather this layer's K/V into the FIXED FLASH_K/V buffers. Runtime
                # trip count is the aligned decode context length; cache rows past
                # decode_pos are zero and the full-matrix bias masks them.
                _dhb = cur_head_dim * self.bytes_per_element
                self._emit_strided_copy_pbi(
                    kv_k_base, self.LAYER0_FLASH_K_DRAM, cur_head_dim,
                    self.k_size, _dhb, self.MAX_CONTEXT_SIZE, self.gpr_aligned_seq_len, sram_addr=0x10000)
                self._emit_strided_copy_pbi(
                    kv_v_base, self.LAYER0_FLASH_V_DRAM, cur_head_dim,
                    self.k_size, _dhb, self.MAX_CONTEXT_SIZE, self.gpr_aligned_seq_len, sram_addr=0x20000)

                _checkpoint(f"L{layer_idx}_kv_gather")

                # unified_attention_core uses bias_mode="full_matrix"; run_decoder
                # uploads group_size identical bias rows for this call.
                bias_addr_layer = (self.LAYER0_FLASH_BIAS_FULL_DRAM
                                   if layer_idx in self._full_attention_layers
                                   else self.LAYER0_FLASH_BIAS_SLIDING_DRAM)
                total_flops += self.unified_attention_core(
                    batch=self.group_size,
                    aligned_seq_len=seq_len,
                    head_dim=cur_head_dim,
                    Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
                    K_DRAM_ADDR=self.LAYER0_FLASH_K_DRAM,
                    V_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                    BIAS_DRAM_ADDR=bias_addr_layer,
                    OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                    SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                    IDENTITY_DRAM_ADDR=self.IDENTITY_DRAM_ADDR,
                    gpr_batch_reg=gpr_group_size,
                    gpr_aligned_seq_len_reg=self.gpr_aligned_seq_len,
                    q_scale=1.0,
                )
                _checkpoint(f"L{layer_idx}_attention")

                # O projection: INT4, K=cur_q_size (actual per-layer attention output dim)
                total_flops += self.quantized_matmat_core(M=1, K=cur_q_size, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off,
                    )
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_POST_ATTN_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_NORM_GAMMA + layer_off,
                              gpr_M_reg=gpr_one)

                # Attention residual: use layer_input_addr (LAYER0_OUTPUT_DRAM
                # for layers > 0, LAYER0_INPUT_DRAM for layer 0) — same source
                # as the pre-norm above. This avoids the LAYER0_OUTPUT → LAYER0_INPUT
                # copy that used to run at the top of every layer.
                self.accelerator_memory_to_sram(accelerator_dram_address=layer_input_addr, sram_address=0x10000, element_size=self.vector_length)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_NORM_DRAM, sram_address=0x90000, element_size=self.vector_length)
                self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, element_size=self.vector_length)

                _checkpoint(f"L{layer_idx}_o_proj")

                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off,
                              gpr_M_reg=gpr_one)

                total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=cur_mlp,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off,
                    gelu_enable=True,
                    )
                total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=cur_mlp,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off,
                    )

                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_GATE_DRAM, sram_address=0x10000, element_size=cur_mlp)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_UP_DRAM, sram_address=0x90000, element_size=cur_mlp)
                self.eltwise_mul_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=cur_mlp)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_MLP_MULT_DRAM, element_size=cur_mlp)

                total_flops += self.matmat_mul_core(M=1, K=cur_mlp, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off,
                        gpr_M_reg=gpr_one,
                    )
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_POST_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_FFW_NORM_GAMMA + layer_off,
                              gpr_M_reg=gpr_one)

                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, sram_address=0x10000, element_size=self.vector_length)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_MLP_NORM_DRAM, sram_address=0x90000, element_size=self.vector_length)
                self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, element_size=self.vector_length)

                _checkpoint(f"L{layer_idx}_mlp")

                # Per-layer input injection (NEW for Gemma4 E2B) - decoder uses seq_len=1
                total_flops += self._compile_per_layer_injection(layer_idx, layer_off, 1)

                _checkpoint(f"L{layer_idx}_inject")

            if layer_size == self.LAYER_SIZE:
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_OUTPUT_DRAM,
                    OUTPUT_DRAM_ADDR=self.OUTPUT_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_OUTPUT_NORM_GAMMA,
                    gpr_M_reg=gpr_one)
                total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=self.EMBEDDING_ELEMENTS,
                    A_DRAM_ADDR=self.OUTPUT_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD_QUANT,
                    OUTPUT_DRAM_ADDR=self.LOGITS_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD_SCALE,
                    # On-FPGA repetition penalty (llama3.2-style): add the per-vocab
                    # bias as the matmul C term so the HW argmax already returns the
                    # penalized token. PENALTY_BIAS_DRAM is all-zero unless
                    # GEMMA4_PENALTY=1, so greedy decode is bit-identical.
                    C_DRAM_ADDR=self.PENALTY_BIAS_DRAM,
                    bias_mode="broadcast_N",
                    gpr_M_reg=gpr_one,
                    )
                _checkpoint("lm_head")

            # Advance decode_pos for next token. The host's preamble only
            # sets gpr_seq_len once at the very first decode step; each
            # subsequent step the program self-increments.
            self.generate_instruction_add_inc(self.gpr_seq_len)
            self.generate_instruction_halt()
            self.release_isa_reg()  # gpr_group_size
            self.release_isa_reg()  # gpr_one
            instr_count = self.capture_count - count_at_start
            _original_print(f"    decoder segment ({instr_count} instr) done in {time.perf_counter()-seg_t0:.1f}s")
        program_sizes = [instr_count * 32]
        total_flops_list = [total_flops]
        self._decoder_checkpoints = checkpoints
        _SILENT_MODE = False
        return None, program_sizes, total_flops_list

    def _dispatch_program(self, gpr_sets: list[tuple[int, int]],
                          target_addr: int | None,
                          timeout: float = 50.0, flops: float | None = None):
        """Build and execute a one-shot dispatch preamble at the fixed scratch
        slot ``self._preamble_addr`` (which sits past every cached program).

        The preamble sets each (reg, value) in ``gpr_sets`` via add_set, then:
          - if ``target_addr`` is given, jump_abs into that cached program
            (which ends in its own HALT); or
          - if ``target_addr`` is None, HALT immediately (used to prime a gpr
            with no program to run).

        This is the gemma3 dispatch idiom: the preamble is always rewritten to
        the SAME address and never advances the program cursor, so it can never
        clobber the prefill or decoder programs. Returns program_execute's
        (latency, flop_rate).
        """
        self.clear_inst_id()
        self.start_capture()
        for reg, val in gpr_sets:
            self.generate_instruction_add_set(reg, val)
        if target_addr is not None:
            self.generate_instruction_jump_abs(ue_35bit_addr_shifter(target_addr))
        else:
            self.generate_instruction_halt()
        self.stop_capture()
        self.write_captured_instructions_to_dram(self._preamble_addr)
        self.clear_capture_buffer()
        return self.program_execute(self._preamble_addr, timeout=timeout, flops=flops)

    def run_decoder(self, decoder_program_sizes: list[int], decoder_base_addr: int, token_id: int, flops_per_token: list[int] | None = None) -> dict:
        """Run decode loop with dynamic PBI.

        Single decoder program — same address every token. gpr_seq_len is
        primed ONCE before the first decode step to the current decode_pos
        (= prompt length); the captured program's trailing add_inc(gpr_seq_len)
        advances it automatically for subsequent tokens. gpr_aligned_seq_len
        is re-set each step because K context length grows by 1 token and may
        cross a UE_VECTOR_SIZE boundary.
        """
        if token_id is None:
            print("No last token available for decode.")
            return {}

        global _SILENT_MODE
        max_seq_len = self.MAX_CONTEXT_SIZE
        _maxdec = os.environ.get("GEMMA4_MAX_DECODE")   # benchmark cap (e.g. 128); default off
        if _maxdec:
            max_seq_len = min(max_seq_len, self.seq_len + int(_maxdec))
        total_latency, total_flop_rate = 0, 0
        # Single program (dynamic PBI). Ignore decoder_program_sizes length.
        prog_addr = decoder_base_addr
        flops_per_token_scalar = flops_per_token[0] if flops_per_token else None

        # On-FPGA repetition penalty (llama3.2_1b mechanism), DEFAULT OFF. Gemma4
        # decodes greedily by default — the HW argmax of the LM-head logits (no
        # readback). When GEMMA4_PENALTY=1, each decode step refreshes a per-vocab
        # additive bias (PENALTY_BIAS_DRAM, the LM-head matmul C term) so the HW
        # argmax of (logits + bias) returns the penalized token. Bias all-zero
        # (penalty off) makes greedy decode bit-identical. Same on-FPGA penalty as
        # llama3.2_1b / qwen2.5-vl / qwen3 (kept as an opt-in loop-breaker backup).
        _pen_off        = os.environ.get("GEMMA4_PENALTY", "0") != "1"
        self.pen_alpha  = float(os.environ.get("GEMMA4_PEN_ALPHA", "1.0"))
        self.pen_cap    = float(os.environ.get("GEMMA4_PEN_CAP", "20.0"))
        self.rep_window = int(os.environ.get("GEMMA4_REP_WINDOW", "256"))
        _greedy_until   = int(os.environ.get("GEMMA4_GREEDY_UNTIL", "0"))
        self.pen_loop_recent = int(os.environ.get("GEMMA4_PEN_LOOP_RECENT", "24"))  # anti-loop window (0=off)
        self.pen_loop_thr    = int(os.environ.get("GEMMA4_PEN_LOOP_THR", "8"))       # ban tok at >= thr of last RECENT
        _gen_tokens: list[int] = []
        _n_generated = 0
        self.dma_to_accelerator_memory(
            self.PENALTY_BIAS_DRAM,
            torch.zeros(1, self.EMBEDDING_ELEMENTS, dtype=torch.bfloat16))
        if not _pen_off:
            print(f"[decode] on-FPGA repetition penalty ON "
                  f"(alpha={self.pen_alpha} cap={self.pen_cap} window={self.rep_window} "
                  f"greedy_until={_greedy_until} loop={self.pen_loop_recent}/{self.pen_loop_thr}); "
                  f"unset GEMMA4_PENALTY for pure greedy")

        # Prime gpr_seq_len to the current decode_pos (= prompt length, since
        # self.seq_len reflects the prompt at this point). Subsequent steps
        # rely on the program's add_inc to advance it. A HALT-terminated
        # preamble (no program to run) just latches the register on the HW.
        self._dispatch_program([(self.gpr_seq_len, self.seq_len)], None, timeout=10.0)
        print("\n------------------------------ DECODE START ------------------------------\n", flush=True)

        # Live decode status bar (mirrors llama3.2_1b / gemma4_e4b): pin the bottom
        # terminal row via an ANSI scroll region; generated tokens stream above it
        # while a tokens/s counter refreshes in place. All output is on stdout
        # (tokens scroll inside rows 1..rows-1; the status writes row `rows` with
        # cursor save/restore), so nothing clobbers the streamed text. TTY-only
        # (skipped when piped/redirected).
        import shutil
        _dec_start_seq = self.seq_len
        _dec_timer = time.perf_counter()
        _first_tok_dt = None   # wall-clock of the 1st decoded token → peak tok/s
        _decoded_n = 0         # number of decode steps (for average tok/s)
        _use_status = sys.stdout.isatty()
        def _status_setup():
            rows = shutil.get_terminal_size().lines
            sys.stdout.write(f"\033[1;{rows - 1}r")   # scroll region = rows 1..rows-1
            sys.stdout.write(f"\033[{rows - 1};1H")   # park cursor at bottom of region
            sys.stdout.flush()
        def _status_update():
            rows = shutil.get_terminal_size().lines
            n = self.seq_len - _dec_start_seq
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
            _tok_t0 = time.perf_counter()               # per-token wall-clock start
            self.seq_len += 1
            decode_pos = self.seq_len - 1               # 0-based pos of token now being computed
            aligned_seq_len = ((self.seq_len + 63) // 64) * 64

            embedding_tensor = self.get_embedding_for_tokens([token_id])
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
            per_layer_inputs = self._compute_per_layer_inputs([token_id], embedding_tensor)
            self.dma_to_accelerator_memory(self.PER_LAYER_INPUTS_DRAM, per_layer_inputs.permute(1, 0, 2).contiguous())

            # Build BOTH decode bias matrices. unified_attention_core uses
            # bias_mode="full_matrix", and decoder batch is group_size Q heads.
            full_bias_row = torch.full((self.group_size, aligned_seq_len), -1e36, dtype=torch.bfloat16)
            full_bias_row[:, :self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_FULL_DRAM, full_bias_row)
            if self.seq_len <= self.sliding_window:
                sliding_bias_row = full_bias_row
            else:
                sliding_bias_row = torch.full((self.group_size, aligned_seq_len), -1e36, dtype=torch.bfloat16)
                window_start = self.seq_len - self.sliding_window
                sliding_bias_row[:, window_start:self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_SLIDING_DRAM, sliding_bias_row)

            # On-FPGA penalty: refresh THIS step's per-vocab bias (the LM-head
            # matmul's C term) once past the greedy gate, so the HW argmax of
            # (logits + bias) returns the penalized token. No logit readback.
            if not _pen_off and _n_generated >= _greedy_until:
                self._write_penalty_bias(_gen_tokens)

            # Dynamic-PBI dispatch: re-set the attention length (K context grows
            # each step, may cross a 64-align boundary), then jump into the
            # cached decoder program. gpr_seq_len was primed once above and is
            # advanced by the decoder's trailing add_inc.
            latency, flop_rate_program = self._dispatch_program(
                [(self.gpr_aligned_seq_len, aligned_seq_len)],
                prog_addr, timeout=300.0, flops=flops_per_token_scalar)
            total_latency += latency
            total_flop_rate += flop_rate_program
            # HW argmax of (logits + penalty bias): greedy when the bias is zero;
            # penalized token when GEMMA4_PENALTY=1 (bias added in the LM-head).
            token_id = self.get_arg_max_index()
            _gen_tokens.append(int(token_id))
            _n_generated += 1
            token_char = self.tokenizer.decode([token_id])
            _SILENT_MODE = False

            _tok_dt = time.perf_counter() - _tok_t0
            if _first_tok_dt is None:
                _first_tok_dt = _tok_dt                  # 1st token (shortest context) → peak
            _decoded_n += 1

            if token_id in [1, self._end_of_turn_token_id]:
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
        # Decode-speed report (matches the Qwen comparison table format).
        _elapsed = time.perf_counter() - _dec_timer
        _peak = (1.0 / _first_tok_dt) if _first_tok_dt else 0.0
        _avg = (_decoded_n / _elapsed) if _elapsed > 0 else 0.0
        print(f"\nDecode speed: peak (1st token) {_peak:.2f} tok/s, "
              f"average {_avg:.2f} tok/s  ({_decoded_n} tokens in {_elapsed:.2f}s)")
        return self.seq_len, total_latency, total_flop_rate

    def _program_image_paths(self, profile: bool = False) -> tuple[str, str]:
        bin_dir = os.path.join(self.script_dir, "gemma4_e2b_bin")
        stem = "programs_profile" if profile else "programs"
        return (os.path.join(bin_dir, stem + ".bin"),
                os.path.join(bin_dir, stem + ".json"))

    def compile_gemma4(self, layer_size: int = 35, profile: bool = False) -> None:
        """Capture [prefill][decoder] into one combined image and write it to
        disk (gemma4_e2b_bin/programs.bin + programs.json). No DRAM is touched —
        run_gemma4() loads it.

        Both programs are seq_len-agnostic (dynamic-PBI: row counts come from
        gpr_seq_len / gpr_q_seq_len / gpr_aligned_seq_len at runtime), so the
        cached image is valid for ANY prompt length up to max_prefill_seq_len.
        Reused as-is whenever programs.bin exists — delete it to force a rebuild.

        set_prefill_seq() MUST have been called first (prefill needs the prompt).
        """
        assert self.prefill_seq is not None, (
            "call set_prefill_seq() before compile_gemma4()")
        prefill_seq_len = len(self.prefill_seq) - 1
        bin_path, meta_path = self._program_image_paths(profile)
        os.makedirs(os.path.dirname(bin_path), exist_ok=True)
        if os.path.exists(bin_path) and os.path.exists(meta_path):
            print(f"[compile] reusing cached {os.path.basename(bin_path)} "
                  f"(seq_len-agnostic; delete to force recompile).")
            return

        print(f"[compile] building combined [prefill@{prefill_seq_len}][decoder] image...")
        global _SILENT_MODE
        _SILENT_MODE = True
        _orig_builtin_print = builtins.print
        builtins.print = lambda *a, **kw: None
        try:
            self.clear_inst_id()
            self.clear_capture_buffer()
            self.start_capture()
            # Leading flag-clear (instruction 0); both program entries land past
            # it, matching the standalone prefill/decoder layouts they replace.
            self.generate_instruction_flag_clear()
            instruction_base_addr = self.get_program_dram_addr()

            prefill_count_at_start = self.capture_count          # 1 (after flag-clear)
            _, prefill_total_flops = self.compile_prefill(seq_len=prefill_seq_len,
                                                          layer_size=layer_size,
                                                          profile=profile)
            prefill_program_addr = instruction_base_addr + prefill_count_at_start * INSTRUCTION_SIZE_BYTES
            prefill_size_bytes = (self.capture_count - prefill_count_at_start) * INSTRUCTION_SIZE_BYTES

            decoder_count_at_start = self.capture_count
            _, decoder_program_sizes, decoder_total_flops = self.compile_decoder(
                layer_size=layer_size, profile=profile)
            decoder_program_addr = instruction_base_addr + decoder_count_at_start * INSTRUCTION_SIZE_BYTES
            decoder_size_bytes = decoder_program_sizes[0]

            self.stop_capture()
            image_bytes = bytearray()
            for inst in self.capture_buffer:
                image_bytes.extend(inst.get_bytes())
            self.clear_capture_buffer()
        finally:
            _SILENT_MODE = False
            builtins.print = _orig_builtin_print
            # Capture must leave the program cursor untouched at the base so a
            # subsequent load lands the image where its jumps were baked.
            self._next_program_dram_addr = instruction_base_addr

        manifest = {
            "instruction_bin": os.path.relpath(bin_path, self.script_dir),
            "instruction_base_addr": f"0x{instruction_base_addr:X}",
            "instruction_total_size": len(image_bytes),
            "prefill_seq_len": prefill_seq_len,
            "prefill_program_start_addr": f"0x{prefill_program_addr:X}",
            "prefill_program_size": prefill_size_bytes,
            "prefill_total_flops": prefill_total_flops,
            "decoder_program_start_addr": f"0x{decoder_program_addr:X}",
            "decoder_program_size": decoder_size_bytes,
            "decoder_total_flops": decoder_total_flops[0],
            "layer_size": layer_size,
        }
        if profile:
            manifest["prefill_profile_checkpoints"] = self._prefill_checkpoints
            manifest["decoder_profile_checkpoints"] = self._decoder_checkpoints
        bin_tmp, meta_tmp = bin_path + ".tmp", meta_path + ".tmp"
        with open(bin_tmp, "wb") as f:
            f.write(bytes(image_bytes)); f.flush(); os.fsync(f.fileno())
        with open(meta_tmp, "w") as f:
            json.dump(manifest, f, indent=2); f.flush(); os.fsync(f.fileno())
        os.rename(bin_tmp, bin_path); os.rename(meta_tmp, meta_path)

        print(f"[compile] wrote {len(image_bytes)/1024:.1f} KB -> {os.path.basename(bin_path)}")
        print(f"[compile]   prefill @ 0x{prefill_program_addr:X} ({prefill_size_bytes/1024:.1f} KB), "
              f"decoder @ 0x{decoder_program_addr:X} ({decoder_size_bytes/1024:.1f} KB)")

    def run_gemma4(self) -> tuple[int, float, float, float, float, float]:
        """Load the combined image from disk into program DRAM, then run one
        prefill pass and decode to the stop token / context limit — the gemma4
        analogue of run_gemma3() (load .bin, preamble-dispatch prefill + decode).
        Requires a prior compile_gemma4().

        Returns (token_cnt_decoded, latency_hw_prefill, latency_hw_decoder,
        flop_rate_hw_decoder, wallclock_prefill_s, wallclock_decoder_s).
        """
        bin_path, meta_path = self._program_image_paths()
        with open(meta_path, "r") as f:
            meta = json.load(f)
        base_addr = int(meta["instruction_base_addr"], 16)
        prefill_program_addr = int(meta["prefill_program_start_addr"], 16)
        decoder_program_addr = int(meta["decoder_program_start_addr"], 16)

        # Load the combined [prefill][decoder] image into program DRAM at the
        # baked base (PBI jump targets resolve against it), then park the single
        # dispatch preamble just past the whole image.
        self._next_program_dram_addr = base_addr
        start_addr, prog_size = self.load_program_instructions_from_file(bin_path)
        assert start_addr == base_addr, (
            f"combined image loaded at 0x{start_addr:X}, expected baked base 0x{base_addr:X}")
        self._preamble_addr = self.get_program_dram_addr()
        print(f"[run] loaded combined image at 0x{base_addr:X} ({prog_size/1024:.1f} KB); "
              f"dispatch preamble @ 0x{self._preamble_addr:X}")

        print(f"\n--- Starting prefill ---")
        print(f"--- Prompt begin ({len(self.prefill_seq)} tokens) ---")
        print(f"  [{', '.join(str(t) for t in self.prefill_seq)}]")
        try:
            _prompt_text = self.tokenizer.decode(list(self.prefill_seq), skip_special_tokens=False)
            print(f"  text: {_prompt_text!r}")
        except Exception as _e:
            print(f"  text: (decode failed: {_e})")
        print(f"--- Prompt end ---")

        timer = time.perf_counter()
        latency_hw_prefill, _flop_rate_hw_prefill = self.run_prefill(
            prefill_program_addr, flops=meta["prefill_total_flops"])
        latency_prefill = time.perf_counter() - timer
        print(f"Prefill execute done in {latency_prefill:.2f} seconds, start decoding...\n", flush=True)

        timer = time.perf_counter()
        token_cnt_decoded, latency_hw_decoder, flop_rate_hw_decoder = self.run_decoder(
            [meta["decoder_program_size"]], decoder_program_addr,
            token_id=self.prefill_seq[-1], flops_per_token=[meta["decoder_total_flops"]])
        latency_decoder = time.perf_counter() - timer
        return (token_cnt_decoded, latency_hw_prefill, latency_hw_decoder,
                flop_rate_hw_decoder, latency_prefill, latency_decoder)

    def _profile_execute(self, gpr_sets: list[tuple[int, int]], target_addr: int,
                         checkpoints: list, tail_name: str, timeout: float = 120.0) -> list:
        """Run a checkpointed program segment-by-segment, returning [(name, ms)]
        HW latency per segment. A preamble at self._preamble_addr primes each
        (reg, value) in ``gpr_sets`` then jumps into ``target_addr``; each
        checkpoint HALT stops the FPGA so the per-segment latency counter can be
        read before resuming from the recorded address. The final ``tail_name``
        segment covers everything after the last checkpoint up to the program's
        terminal HALT. Because the resume addresses are the instruction right
        after each HALT, the segments tile the whole program with no gaps, so the
        summed latencies cover all FPGA execution (see run_gemma4_profile)."""
        self.clear_inst_id()
        self.start_capture()
        for reg, val in gpr_sets:
            self.generate_instruction_add_set(reg, val)
        self.generate_instruction_jump_abs(ue_35bit_addr_shifter(target_addr))
        self.stop_capture()
        self.write_captured_instructions_to_dram(self._preamble_addr)
        self.clear_capture_buffer()

        results = []
        self.start_execute_from_dram(self._preamble_addr)
        for name, resume_hex in checkpoints:
            self.wait_queue(timeout)
            results.append((name, self.report_latency_in_us() / 1e3))   # ms
            self.start_execute_from_dram(int(resume_hex, 16))
        self.wait_queue(timeout)   # tail segment: everything up to the terminal HALT
        results.append((tail_name, self.report_latency_in_us() / 1e3))
        return results

    def _decode_profile_execute(self, decoder_addr: int, aligned_seq_len: int,
                                checkpoints: list, timeout: float = 120.0) -> list:
        """One decode step through the profile-bin's HALT checkpoints. Preamble
        primes gpr_seq_len (=decode_pos) / gpr_aligned_seq_len; the tail segment
        is the add_inc(gpr_seq_len) + terminal HALT."""
        return self._profile_execute(
            [(self.gpr_seq_len, self.seq_len - 1),
             (self.gpr_aligned_seq_len, aligned_seq_len)],
            decoder_addr, checkpoints, tail_name="tail_addinc", timeout=timeout)

    def _print_phase_breakdown(self, title: str, results: list, per_token: bool) -> None:
        """Aggregate per-layer checkpoints ("L<idx>_<phase>") into phase totals
        and print a table. ``per_token`` adds a decode throughput line."""
        import re
        from collections import defaultdict
        phase_ms: dict = defaultdict(float)
        phase_n: dict = defaultdict(int)
        for name, ms in results:
            m = re.match(r"L\d+_(.+)", name)
            phase = m.group(1) if m else name
            phase_ms[phase] += ms
            phase_n[phase] += 1
        total_ms = sum(phase_ms.values())

        print(f"\n=== {title} per-phase HW latency ===")
        print(f"{'phase':<16}{'total ms':>10}{'%':>7}{'n':>5}{'ms/layer':>10}")
        print("-" * 48)
        for phase, ms in sorted(phase_ms.items(), key=lambda kv: -kv[1]):
            n = phase_n[phase]
            pct = ms / total_ms * 100 if total_ms else 0.0
            print(f"{phase:<16}{ms:>10.3f}{pct:>6.1f}%{n:>5}{ms/n:>10.4f}")
        print("-" * 48)
        print(f"{'TOTAL':<16}{total_ms:>10.3f}{100.0:>6.1f}%")
        if per_token and total_ms:
            print(f"Decode HW throughput: {1000.0/total_ms:.2f} tok/s ({total_ms:.1f} ms/token)")

    def run_gemma4_profile(self) -> None:
        """Load the profile image (programs_profile.bin, built with per-phase
        HALT checkpoints), run a profiled prefill and one profiled decode step,
        and print per-phase HW-latency breakdowns aggregated across all layers.
        Use to find where prefill / decode time goes. Requires
        compile_gemma4(profile=True)."""
        bin_path, meta_path = self._program_image_paths(profile=True)
        with open(meta_path, "r") as f:
            meta = json.load(f)
        base_addr = int(meta["instruction_base_addr"], 16)
        prefill_program_addr = int(meta["prefill_program_start_addr"], 16)
        decoder_program_addr = int(meta["decoder_program_start_addr"], 16)
        prefill_checkpoints = meta.get("prefill_profile_checkpoints", [])
        decoder_checkpoints = meta["decoder_profile_checkpoints"]

        self._next_program_dram_addr = base_addr
        _, prog_size = self.load_program_instructions_from_file(bin_path)
        self._preamble_addr = self.get_program_dram_addr()
        print(f"[profile] loaded profile image at 0x{base_addr:X} ({prog_size/1024:.1f} KB), "
              f"{len(prefill_checkpoints)} prefill + {len(decoder_checkpoints)} decode checkpoints")

        # --- Profiled prefill (segmented). Populates the KV cache AND sets
        # self.seq_len exactly like a straight run, and returns per-phase HW
        # latencies for the whole prefill. ---
        print(f"\n--- Profiling prefill (seq_len={len(self.prefill_seq) - 1}) ---", flush=True)
        prefill_results = self.run_prefill(
            prefill_program_addr, flops=meta["prefill_total_flops"],
            profile_checkpoints=prefill_checkpoints)
        self._print_phase_breakdown("PREFILL", prefill_results, per_token=False)

        # Host prep for ONE decode step (mirrors run_decoder's per-step work).
        token_id = self.prefill_seq[-1]
        self.dma_to_accelerator_memory(
            self.PENALTY_BIAS_DRAM,
            torch.zeros(1, self.EMBEDDING_ELEMENTS, dtype=torch.bfloat16))
        self.seq_len += 1
        aligned_seq_len = ((self.seq_len + 63) // 64) * 64
        emb = self.get_embedding_for_tokens([token_id])
        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, emb)
        pli = self._compute_per_layer_inputs([token_id], emb)
        self.dma_to_accelerator_memory(self.PER_LAYER_INPUTS_DRAM, pli.permute(1, 0, 2).contiguous())
        full_bias = torch.full((self.group_size, aligned_seq_len), -1e36, dtype=torch.bfloat16)
        full_bias[:, :self.seq_len] = 0.0
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_FULL_DRAM, full_bias)
        if self.seq_len <= self.sliding_window:
            sliding_bias = full_bias
        else:
            sliding_bias = torch.full((self.group_size, aligned_seq_len), -1e36, dtype=torch.bfloat16)
            sliding_bias[:, self.seq_len - self.sliding_window:self.seq_len] = 0.0
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_SLIDING_DRAM, sliding_bias)

        print(f"\n--- Profiling one decode step (pos {self.seq_len - 1}) ---", flush=True)
        decode_results = self._decode_profile_execute(decoder_program_addr, aligned_seq_len, decoder_checkpoints)
        self._print_phase_breakdown("DECODE (1 token)", decode_results, per_token=True)



def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Gemma4 E2B LM prefill + decode on the accelerator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python gemma4_e2b_refactor.py                          # default prompt
  python gemma4_e2b_refactor.py --prompt "your prompt"   # custom prompt
  python gemma4_e2b_refactor.py --dev xdma1 --cycle 5.62

default prompt: "x+3=5, what is x?"
                (pre-tokenized as default_prefill_tokens in gemma4_e2b_config.json)""")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt. Default is the built-in test question.")
    parser.add_argument("--local-weights", action="store_true",
                        help="Use gemma4_e2b_bin/params.bin instead of the configured weights bin.")
    parser.add_argument('--dev', type=str, default='xdma0',
                        help='DMA device name (e.g., xdma0, xdma1). Default: xdma0')
    parser.add_argument('--cycle', type=float, default=5.62,
                        help='Clock cycle time in nanoseconds (default: 5.62)')
    parser.add_argument('--profile', action='store_true',
                        help='Compile a profile bin with per-phase HALT checkpoints and run one '
                             'profiled decode step; print a per-phase HW-latency breakdown.')
    args = parser.parse_args()

    set_dma_device(args.dev)
    global DMA_DEVICE_H2C, DMA_DEVICE_C2H, DMA_DEVICE_USER
    DMA_DEVICE_H2C = user_dma_core.DMA_DEVICE_H2C
    DMA_DEVICE_C2H = user_dma_core.DMA_DEVICE_C2H
    DMA_DEVICE_USER = user_dma_core.DMA_DEVICE_USER
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle
    print(f"Using DMA device: {args.dev}")
    print(f"  H2C: {DMA_DEVICE_H2C}")
    print(f"  C2H: {DMA_DEVICE_C2H}")
    print(f"  USER: {DMA_DEVICE_USER}")
    print(f"Setting CLOCK_CYCLE_TIME_NS = {user_dma_core.CLOCK_CYCLE_TIME_NS}")

    ue = Gemma4_UnifiedEngine(local_weights=args.local_weights)

    # Prompt first — the prefill program is compiled for its exact length.
    if args.prompt:
        print(f"[Mode] LM -- prompt: {args.prompt!r}")
        ue.set_prefill_seq(args.prompt)
    else:
        print(f"[Mode] LM -- default prompt")
        ue.set_prefill_seq()

    # --profile: build the instrumented bin and print a per-phase decode
    # breakdown instead of the normal generation run.
    if args.profile:
        print(f"\n--- Compiling profile bin (per-phase checkpoints) ---")
        timer = time.perf_counter()
        ue.compile_gemma4(profile=True)
        print(f"Profile compile done in {time.perf_counter() - timer:.2f} seconds")
        ue.run_gemma4_profile()
        print("Gemma4 E2B decode profile ends.")
        return

    # gemma3-style workflow: compile (decoder + real-seq_len prefill) then run
    # (prefill + decode). See Gemma4_UnifiedEngine.compile_gemma4 / run_gemma4.
    print(f"\n--- Compiling ---")
    timer = time.perf_counter()
    ue.compile_gemma4()
    print(f"Compile done in {time.perf_counter() - timer:.2f} seconds")

    (token_cnt_decoded, latency_hw_prefill, latency_hw_decoder,
     flop_rate_hw_decoder, latency_prefill, latency_decoder) = ue.run_gemma4()

    print(f"\nDecoder done in {latency_prefill + latency_decoder:.2f} seconds, "
          f"speed: {(token_cnt_decoded - len(ue.prefill_seq) + 1) / latency_decoder:.2f} tokens/s, "
          f"total {token_cnt_decoded} tokens.")
    print(f"HW counter: Latency: {(latency_hw_prefill + latency_hw_decoder) / 1e6:.2f} seconds, "
          f"decoder average Gflops: {flop_rate_hw_decoder / (token_cnt_decoded - len(ue.prefill_seq) + 1):.2f} Gflops")
    print("Gemma4 E2B LM test ends.")


if __name__ == "__main__":
    main()

