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
  python gemma4_e2b_test.py
  python gemma4_e2b_test.py --prompt "your prompt"
  python gemma4_e2b_test.py --dev xdma0 [--cycle 5.042]

Fixed layout: this script, gemma4_e2b_config.json, and gemma4_e2b_bin/ live in
the same folder; user_dma_core.py is two folders up (repo root), added to
sys.path.
"""

import gc
import json
import math
import os
import sys

# Disable the HF Hub Xet backend during weight-bin generation (avoids its extra
# memory/process overhead; see gemma4 params.bin memory-optimization work).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# This file's folder: gemma4_e2b_bin/ and *.json live here. user_dma_core is
# two levels up (repo root).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

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
from multi_engine_shard import MultiEngineScheduler

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


# Shipped sample image for VLM mode (same as gemma4_e2b_test.py's DEFAULT_IMAGE):
# repo test_samples/yosemite.jpg, two folders up from this script.
DEFAULT_IMAGE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "test_samples", "yosemite.jpg"))
DEFAULT_AUDIO = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "test_samples", "apex.wav"))


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
    _QUANT_WORKERS = max(1, int(os.environ.get("GEMMA4_QUANT_WORKERS", "2")))
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
        # Re-download if config exists but the actual weight files don't (handles a
        # partial/interrupted snapshot_download).
        has_checkpoint = False
        if os.path.isdir(model_dir):
            for _root, _dirs, _files in os.walk(model_dir):
                if any(name.endswith(".safetensors")
                       or name in ("pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")
                       for name in _files):
                    has_checkpoint = True
                    break
        if not os.path.exists(config_path) or not has_checkpoint:
            _original_print(f"Downloading HF model {hf_repo} to {os.path.abspath(model_dir)} ...")
            snapshot_download(repo_id=hf_repo, local_dir=model_dir)
            _original_print("Download complete.")
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map=None, trust_remote_code=True
        )
        return model, model_dir

    def _write_host_section(f, text_model, cfg) -> tuple[int, dict]:
        """Stream host-side tensors into the open combined params bin.

        Same manifest layout as the old _build_host_section_bytes(), but the huge
        `embed_tokens_per_layer` table (~4.5 GB bf16) is written in chunks instead
        of materializing a multi-GiB Python bytes object. Laid down first so the
        run-time loader can mmap exactly its offset and pull rows on demand.
        """
        file_info = cfg["file_info"]
        num_layers = file_info["num_layers"]
        per_layer_input_dim = file_info["per_layer_input_dim"]
        section_start = f.tell()

        # 1. per_layer_embed_tokens, pre-scaled by sqrt(per_layer_input_dim).
        # Chunked scale-and-cast, written straight to disk (never materialize the
        # full fp32/bf16 tensor — would be ~9.4 GB).
        src = text_model.embed_tokens_per_layer.weight.detach().cpu()
        per_layer_embed_scale = per_layer_input_dim ** 0.5
        embed_off = 0
        embed_shape = list(src.shape)
        embed_size = src.numel() * torch.tensor([], dtype=torch.bfloat16).element_size()
        chunk = 1024
        for i in range(0, src.shape[0], chunk):
            scaled = (src[i:i+chunk].float() * per_layer_embed_scale).to(torch.bfloat16).contiguous()
            f.write(scaled.view(torch.uint8).numpy().tobytes())
        del src

        # 2. per_layer_model_proj_weight  [8960, 1536]
        proj_off = embed_off + embed_size
        proj_bf16 = text_model.per_layer_model_projection.weight.detach().cpu().to(torch.bfloat16).contiguous()
        proj_b = proj_bf16.view(torch.uint8).numpy().tobytes()
        f.write(proj_b)
        proj_size = len(proj_b)
        proj_shape = list(proj_bf16.shape)
        del proj_b, proj_bf16

        # 3. per_layer_proj_norm_weight   [256]  (raw, no gamma_offset — host-side norm wants raw w)
        norm_off = proj_off + proj_size
        norm_bf16 = text_model.per_layer_projection_norm.weight.detach().cpu().to(torch.bfloat16).contiguous()
        norm_b = norm_bf16.view(torch.uint8).numpy().tobytes()
        f.write(norm_b)
        norm_size = len(norm_b)
        norm_shape = list(norm_bf16.shape)
        del norm_b, norm_bf16
        total = f.tell() - section_start

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

        manifest = {
            "embed_tokens_per_layer": {"offset": embed_off, "size": embed_size, "shape": embed_shape},
            "per_layer_model_proj":   {"offset": proj_off,  "size": proj_size,  "shape": proj_shape},
            "per_layer_proj_norm":    {"offset": norm_off,  "size": norm_size,  "shape": norm_shape},
            "layer_scalars": layer_scalars,
            # JSON keys must be strings — convert int → str.
            "kv_shared_map": {str(k): v for k, v in kv_shared_map.items()},
        }
        print(f"  Host section: {total/1024**3:.2f} GiB, 3 tensors + scalars + kv_shared_map")
        return total, manifest

    def _build_vision_section_bytes(hf_model) -> tuple[bytes, dict]:
        """Pre-quantize the Gemma4 vision-tower weights once and return
        (section_bytes, section_manifest). Ported from gemma4_e2b_test.py.

        The manifest holds tensor offsets RELATIVE to the vision section start
        (vision_weight_init adds the section's absolute file offset) plus the
        scalars the loader needs (VIS_* dims, VIS_ROPE_PERM, clip ranges). Q/K/V
        output rows and O input columns are permuted by VIS_ROPE_PERM (the
        interleaved→head-major 2D-RoPE layout the FPGA attention expects); the
        Q/K norms get the same permutation. Projection matmuls are IF4; norms and
        the position-embedding table stay BF16.
        """
        vt = hf_model.model.vision_tower
        ev = hf_model.model.embed_vision
        H, MLP, HD, NH = 768, 3072, 64, 12
        L_count = 16

        VIS_ROPE_PERM = torch.cat([
            torch.arange(0, 16), torch.arange(32, 48),
            torch.arange(16, 32), torch.arange(48, 64),
        ])

        def _perm_qkv(w: torch.Tensor) -> torch.Tensor:
            return w.reshape(NH, HD, -1)[:, VIS_ROPE_PERM, :].reshape(NH * HD, -1).contiguous()
        def _perm_o(w: torch.Tensor) -> torch.Tensor:
            return w.reshape(-1, NH, HD)[:, :, VIS_ROPE_PERM].reshape(-1, NH * HD).contiguous()

        sections: list[tuple[str, bytes, list[int], str]] = []

        def _add_bf16(key: str, w: torch.Tensor) -> None:
            wc = w.contiguous()
            sections.append((key, wc.view(torch.uint8).numpy().tobytes(), list(wc.shape), "bf16"))

        # Per-layer weights.
        for li in range(L_count):
            L = vt.encoder.layers[li]
            pre = f"layer{li}"
            for n in ["input_layernorm", "post_attention_layernorm",
                      "pre_feedforward_layernorm", "post_feedforward_layernorm"]:
                _add_bf16(f"{pre}.{n}", getattr(L, n).weight.detach().cpu().to(torch.bfloat16))
            for n in ["q_norm", "k_norm"]:
                w = getattr(L.self_attn, n).weight.detach().cpu().to(torch.bfloat16)
                _add_bf16(f"{pre}.{n}", w[VIS_ROPE_PERM])
            q4_names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            q4_tensors: list[torch.Tensor] = []
            for proj in ["q_proj", "k_proj", "v_proj"]:
                w = getattr(L.self_attn, proj).linear.weight.detach().cpu().to(torch.bfloat16)
                q4_tensors.append(_perm_qkv(w))
            q4_tensors.append(_perm_o(L.self_attn.o_proj.linear.weight.detach().cpu().to(torch.bfloat16)))
            for mlp in ["gate_proj", "up_proj", "down_proj"]:
                q4_tensors.append(getattr(L.mlp, mlp).linear.weight.detach().cpu().to(torch.bfloat16))
            qs = _parallel_quantize(VISION_QUANT_PRECISION, [t.contiguous() for t in q4_tensors])
            for name, w, (data_b, scale_b) in zip(q4_names, q4_tensors, qs):
                sections.append((f"{pre}.{name}.scale", scale_b, list(w.shape), f"{VISION_QUANT_PRECISION}_scale"))
                sections.append((f"{pre}.{name}.data",  data_b,  list(w.shape), f"{VISION_QUANT_PRECISION}_data"))

        # Non-layer weights.
        pe = vt.patch_embedder
        _add_bf16("pos_embedding_table",
                  pe.position_embedding_table.detach().cpu().to(torch.bfloat16))
        _add_bf16("embed_norm_gamma", torch.ones(H, dtype=torch.bfloat16))
        _add_bf16("v_norm_ones_gamma", torch.ones(HD, dtype=torch.bfloat16))

        w_patch = pe.input_proj.weight.detach().cpu().to(torch.bfloat16).contiguous()
        w_embed = ev.embedding_projection.weight.detach().cpu().to(torch.bfloat16).contiguous()
        pq = _parallel_quantize(VISION_QUANT_PRECISION, [w_patch, w_embed])
        for name, w, (d, s) in zip(["patch_proj", "embed_proj"], [w_patch, w_embed], pq):
            sections.append((f"{name}.scale", s, list(w.shape), f"{VISION_QUANT_PRECISION}_scale"))
            sections.append((f"{name}.data",  d, list(w.shape), f"{VISION_QUANT_PRECISION}_data"))

        # Clip ranges from the Gemma4ClippableLinear wrappers (JSON can't hold inf).
        def _finite(x):
            if x == float("inf"):  return "inf"
            if x == -float("inf"): return "-inf"
            return float(x)
        clip_ranges = []
        for li in range(L_count):
            L = vt.encoder.layers[li]
            cr: dict = {}
            for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"]:
                proj = (getattr(L.self_attn, proj_name) if proj_name in
                        ("q_proj", "k_proj", "v_proj", "o_proj") else getattr(L.mlp, proj_name))
                if proj.use_clipped_linears:
                    cr[proj_name] = {
                        "input":  [_finite(proj.input_min.item()),  _finite(proj.input_max.item())],
                        "output": [_finite(proj.output_min.item()), _finite(proj.output_max.item())],
                    }
                else:
                    cr[proj_name] = {"input": ["-inf", "inf"], "output": ["-inf", "inf"]}
            clip_ranges.append(cr)

        out = bytearray()
        section_meta: dict = {}
        cur = 0
        for key, b, shape, dtype in sections:
            section_meta[key] = {"offset": cur, "size": len(b), "shape": shape, "dtype": dtype}
            out.extend(b)
            cur += len(b)
        manifest = {
            "vision_quant_precision": VISION_QUANT_PRECISION,
            "num_layers": L_count,
            "VIS_H": H, "VIS_MLP": MLP, "VIS_HEAD_DIM": HD, "VIS_HEADS": NH,
            "VIS_POOL_K": int(vt.config.pooling_kernel_size),
            "VIS_TEXT_H": int(ev.embedding_projection.weight.shape[0]),
            "VIS_ROPE_PERM": VIS_ROPE_PERM.tolist(),
            "sections": section_meta,
            "clip_ranges": clip_ranges,
        }
        print(f"  Vision section: {cur/1024**2:.1f} MiB, {len(section_meta)} tensors")
        return bytes(out), manifest

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
        # they read another layer's KV cache at runtime (see _write_host_section's
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
        # Free this layer's large tensors + quant byte-strings before the next one
        # (keeps peak RSS bounded during bin generation).
        del quant_results, quant_jobs, region_writes
        del q_w, k_w, v_w, gate_w, up_w
        del data_bytes, scale_bytes, scale_padded, data_padded, _t
        gc.collect()

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

    # LM_HEAD is tied with embed_tokens. Quantization is read-only, so avoid a
    # full clone of this large matrix.
    lm_head_w = model.lm_head.weight.detach().cpu().to(torch.bfloat16)
    scale_sz = weight_defs["LM_HEAD_WEIGHT_SCALE_SIZE"]
    data_sz = weight_defs["LM_HEAD_WEIGHT_DATA_SIZE"]
    data_bytes, scale_bytes = _qs_quantize(LM_QUANT_PRECISION, lm_head_w)
    scale_padded = (scale_bytes + b"\x00" * scale_sz)[:scale_sz]
    data_padded = (data_bytes + b"\x00" * data_sz)[:data_sz]
    write_at(weight_defs["LM_HEAD_WEIGHT_SCALE"], scale_padded)
    write_at(weight_defs["LM_HEAD_WEIGHT_DATA"], data_padded)
    del lm_head_w, data_bytes, scale_bytes, scale_padded, data_padded
    gc.collect()

    # Build the vision + host sections in memory (no separate files) and
    # concatenate into ONE weights bin: [LM | vision | audio | host].
    # manifest JSON holds each section's offset + its per-tensor sub-manifest.
    vision_bytes, vision_manifest = _build_vision_section_bytes(model)
    audio_bytes, audio_manifest = build_audio_weight_section(
        model, _parallel_quantize)

    lm_size     = len(buf)
    vision_size = len(vision_bytes)
    audio_size  = len(audio_bytes)
    vision_off  = lm_size
    audio_off   = vision_off + vision_size
    host_off    = audio_off + audio_size

    # Write to a temp file, freeing each in-memory section as it lands and
    # streaming the multi-GiB host section straight to disk, then atomically
    # rename. Keeps peak RSS bounded during generation.
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_out_path = out_path + ".tmp"
    with open(tmp_out_path, "wb") as f:
        f.write(buf)
        del buf
        gc.collect()
        f.write(vision_bytes)
        del vision_bytes
        gc.collect()
        f.write(audio_bytes)
        del audio_bytes
        gc.collect()
        host_size, host_manifest = _write_host_section(f, text_model, cfg)
    os.replace(tmp_out_path, out_path)
    total = lm_size + vision_size + audio_size + host_size
    print(f"Generated weights bin: {out_path} ({total/1024**3:.2f} GiB total; "
          f"LM {lm_size/1024**3:.2f} GiB + vision {vision_size/1024**2:.1f} MiB + "
          f"audio {audio_size/1024**2:.1f} MiB + "
          f"host {host_size/1024**3:.2f} GiB)")

    master_meta_path = out_path.rsplit(".", 1)[0] + ".json"
    master = {
        "compile_version": "v1",
        "total_size": total,
        "lm_section":     {"offset": 0,          "size": lm_size},
        "vision_section": {"offset": vision_off, "size": vision_size, "manifest": vision_manifest},
        "audio_section":  {"offset": audio_off,  "size": audio_size, "manifest": audio_manifest},
        "host_section":   {"offset": host_off,   "size": host_size,   "manifest": host_manifest},
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


def _ensure_hf_model(script_dir: str, cfg: dict):
    """Module-level HF VLM loader for the runtime vision path (processor +
    2D-RoPE tables + vision-tower config). Downloads on first use. Returns
    (model, model_dir)."""
    model_dir = os.path.join(script_dir, cfg["paths"]["hf_model_dir"])
    hf_repo = cfg["paths"]["hf_model_repo"]
    config_path = os.path.join(model_dir, "config.json")
    # Re-download if config exists but the actual weight files don't (handles a
    # partial/interrupted snapshot_download).
    has_checkpoint = False
    if os.path.isdir(model_dir):
        for _root, _dirs, _files in os.walk(model_dir):
            if any(name.endswith(".safetensors")
                   or name in ("pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")
                   for name in _files):
                has_checkpoint = True
                break
    if not os.path.exists(config_path) or not has_checkpoint:
        _original_print(f"Downloading HF model {hf_repo} to {os.path.abspath(model_dir)} ...")
        snapshot_download(repo_id=hf_repo, local_dir=model_dir)
        _original_print("Download complete.")
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map=None, trust_remote_code=True)
    return model, model_dir


# Vision + LM method groups live in sibling files and are mixed into the
# concrete engine below (see their module docstrings). They import nothing
# from this module, so the 3-file split has no import cycle.
from gemma4_e2b_vision import (Gemma4VisionMixin, VISION_QUANT_PRECISION,
                               VISION_CANONICAL_SIZE, VISION_FIXED_NUM_PATCHES)
from gemma4_e2b_lm import Gemma4LMMixin
from gemma4_e2b_audio import (Gemma4AudioMixin, AUDIO_FIXED_SOFT_TOKENS,
                              build_audio_weight_section)


class Gemma4_UnifiedEngine(Gemma4LMMixin, Gemma4VisionMixin,
                           Gemma4AudioMixin, UnifiedEngine):
    """UnifiedEngine specialized to Gemma4 E2B (LM path): loads config + weight
    bin, compiles prefill/decoder into one bin, runs prefill + decode. Numeric
    checks live in gemma4_e2b_numeric.py."""

    def _ensure_stage_scheduler(self, stage: str, worker_dram_base: int):
        """Return the shared two-engine scheduler configuration for one stage."""
        if self.multi_core == 1:
            return None
        scheduler = self._multi_core_schedulers.get(stage)
        if scheduler is None:
            # Two-core keeps its single worker at the supplied stage address
            # (huge stride; only the base is used). >2-core packs the extra
            # workers into the dedicated MULTICORE_WORKER_ISA arena. Vision and
            # prefill are sequential, so both stages reuse this worker-ISA arena.
            worker_stride = 0x10000000
            if self.multi_core > 2:
                worker_dram_base = self.MULTICORE_WORKER_ISA_BASE
                worker_stride = self.MULTICORE_WORKER_ISA_STRIDE
            scheduler = MultiEngineScheduler(
                self, num_engines=self.multi_core,
                engine_base_stride=0x00010000,
                worker_dram_base=worker_dram_base,
                worker_dram_stride=worker_stride,
                worker_tensor_offset=0,
                worker_program_offset=0,
                barrier_margin_nops=32,
                allow_unaligned_rows=True,
                allow_more_than_two_engines=self.multi_core > 2)
            self._multi_core_schedulers[stage] = scheduler
        return scheduler

    def __init__(self, script_dir: str | None = None,
                 vision_kernel: str = "matmatmul", prefill_kernel: str = "streaming",
                 decode_kernel: str = "streaming", multi_core: int = 1,
                 bin_reuse: bool = False):
        for stage, kernel in (("vision", vision_kernel), ("prefill", prefill_kernel),
                              ("decode", decode_kernel)):
            if kernel not in ("streaming", "matmatmul"):
                raise ValueError(f"unsupported {stage} kernel: {kernel}")
        if vision_kernel == "streaming":
            raise NotImplementedError(
                "--vision-kernel streaming is not supported: the vision patch-embed "
                "projection passes clamp_min/clamp_max, but quantized_matmat_core "
                "(the streaming path) lacks clamp support. Use --vision-kernel "
                "matmatmul (the default) until clamp is added to quantized_matmat_core.")
        if os.environ.get("GEMMA4_PENALTY", "0") == "1":
            raise RuntimeError(
                "GEMMA4_PENALTY=1 is temporarily unsupported: the dynamic streaming "
                "quantized_matmat_core must gain broadcast-bias support before the "
                "on-FPGA penalty can be re-enabled.")
        if not 1 <= multi_core <= 8:
            raise ValueError(f"multi_core must be between 1 and 8, got {multi_core}")
        if multi_core > 1 and vision_kernel != "matmatmul":
            raise ValueError("multi-engine vision projection sharding requires --vision-kernel matmatmul")
        if multi_core > 1:
            prefill_kernel = "matmatmul"
        self.vision_kernel = vision_kernel
        self.prefill_kernel = prefill_kernel
        self.decode_kernel = decode_kernel
        self.multi_core = multi_core
        # Program-image reuse: OFF by default (recompile programs.bin fresh every
        # run). --bin-reuse opts into reusing a matching cached section instead.
        self.bin_reuse = bin_reuse
        self._multi_core_schedulers = {}
        self._prefill_shard_m_regs = None
        engine_base = user_dma_core.UE_0_BASE_ADDR
        # Gemma4 DRAM layout. SINGLE-CORE keeps the original upper-2 GB window
        # (matching the Llama-3.2-1B path); its 16 MiB ISA region is tight but
        # sufficient for one engine. MULTI-CORE re-bases to 0x0 with the full
        # 4 GB budget so the per-engine worker programs (incl. sharded vision
        # RoPE) and future parallelism add-ons have room, and so the vision
        # tensor arena is large enough for N-engine scratch.
        #
        # SINGLE-CORE (upper 2 GB):
        #   PARAMS  weights   : 0x80000000 – 0xE1000000  (1552 MiB)
        #   TENSOR  acts/scr  : 0xE1000000 – 0xFF000000  (480 MiB, LM<->vision)
        #   ISA vision core0  : 0xFF000000 – 0xFF400000  (4 MiB)
        #   ISA vision core1  : 0xFF400000 – 0xFF620000  (2.125 MiB, unused 1-core)
        #   ISA LM            : 0xFF620000 – 0x100000000 (9.875 MiB)
        #
        # MULTI-CORE (full 4 GB, base 0x0):
        #   PARAMS  weights   : 0x00000000 – 0x80000000  (2 GiB;  LM ~1540 MB)
        #   TENSOR  acts/scr  : 0x80000000 – 0xC0000000  (1 GiB;  LM + vision,
        #                        incl. up-to-8-engine vision scratch + top wts)
        #   ISA vision core0  : 0xC0000000 – 0xC8000000  (128 MiB, master)
        #   ISA vision core1  : 0xC8000000 – 0xE0000000  (384 MiB, 2-core worker)
        #   ISA LM            : 0xE0000000 – 0xF0000000  (256 MiB)
        #   ISA >2-core workrs: 0xF0000000 – 0x100000000 (256 MiB, 32 MiB/worker)
        # Vision and LM programs remain resident at disjoint addresses.
        self.DRAM_END = 0x100000000
        if self.multi_core == 1:
            _params_base  = 0x80000000
            _tensor_base  = 0xE1000000
            self.VISION_ISA_BASE             = 0xFF000000
            self.VISION_WORKER_ISA_BASE      = 0xFF400000
            self.LM_ISA_BASE                 = 0xFF620000
            self.MULTICORE_WORKER_ISA_BASE   = 0xFC000000    # unused (1-core)
            self.MULTICORE_WORKER_ISA_STRIDE = 0x00240000    # unused (1-core)
        else:
            _params_base  = 0x00000000
            _tensor_base  = 0x80000000
            self.VISION_ISA_BASE             = 0xC0000000
            self.VISION_WORKER_ISA_BASE      = 0xC8000000
            self.LM_ISA_BASE                 = 0xE0000000
            self.MULTICORE_WORKER_ISA_BASE   = 0xF0000000    # >2-core worker arena
            self.MULTICORE_WORKER_ISA_STRIDE = 0x02000000    # 32 MiB / worker
        # Top of the vision tensor arena (vision weights are top-placed against
        # it; scratch stays below). In the multi-core layout every worker ISA
        # lives in the dedicated ISA region, so the tensor arena simply runs up
        # to VISION_ISA_BASE for any engine count.
        self.VISION_ARENA_TOP = self.VISION_ISA_BASE
        _program_base = self.LM_ISA_BASE
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

        self._weights_bin_rel = paths["weights_bin"]
        self.weight_init()
        self.tensor_init()

        # Vision-attention IDENTITY: unified_attention_core needs a
        # UE_VECTOR_SIZE² bf16 identity for its V-transpose. The LM one
        # (IDENTITY_DRAM_ADDR) lives in the tensor region that vision scratch
        # overwrites, so keep a copy in the PARAMS region (untouched by vision
        # scratch @ tensor DRAM and vision ISA @ 0xa0000000).
        self._vis_identity_dram = self.allocate_params_dram(
            UE_VECTOR_SIZE * UE_VECTOR_SIZE * self.bytes_per_element)
        self.dma_to_accelerator_memory(
            self._vis_identity_dram, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))
        # Snapshot the LM weight-DRAM high-water mark now, before any vision run
        # advances the params cursor (its cursor isn't restored). Used by
        # write_run_summary as the "total weight DRAM" figure (~1540.4 MB).
        self._lm_weight_dram_bytes = self.get_params_dram_usage()
        if self.get_params_dram_addr() > self._tensor_dram_base:
            raise MemoryError(
                f"Gemma4 weights overflow the 2-GB layout: "
                f"end=0x{self.get_params_dram_addr():X}, "
                f"tensor_start=0x{self._tensor_dram_base:X}")
        if self.get_tensor_dram_addr() > self.VISION_ISA_BASE:
            raise MemoryError(
                f"Gemma4 tensors overflow the 2-GB layout: "
                f"end=0x{self.get_tensor_dram_addr():X}, "
                f"vision_program_start=0x{self.VISION_ISA_BASE:X}")

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

    # --- shared plumbing (used by the vision/LM method groups) ---------------
    # These stay on the control class so the vision (gemma4_e2b_lm.py) and LM
    # (gemma4_e2b_vision.py) method groups reach them through `self` (MRO) and
    # never import from this module -- which keeps the split import-cycle free.
    def _set_silent(self, on: bool) -> bool:
        """Toggle library-print suppression (see module-level quiet_print) and
        return the previous state so callers can restore it. Encapsulates the
        module global so method groups in the sibling files never need a
        `global _SILENT_MODE` of their own."""
        global _SILENT_MODE
        prev = _SILENT_MODE
        _SILENT_MODE = on
        return prev

    def _loud(self, *args, **kwargs) -> None:
        """Progress print that bypasses suppression (the saved original print)."""
        _original_print(*args, **kwargs)

    def _ensure_model_loaded(self):
        """Load (downloading if missing) the HF model once and cache it on self.
        Centralizes the download in this control file so the vision/LM groups
        just read self.hf_model / self.hf_model_dir."""
        if getattr(self, "hf_model", None) is None:
            self.hf_model, self.hf_model_dir = _ensure_hf_model(self.script_dir, self._cfg)
            self.hf_model.eval()
        return self.hf_model, self.hf_model_dir

    def set_prefill_seq_vlm(self, image_path: str, prompt: str | None = None,
                            profile: bool = False, vision_host: bool = False) -> None:
        """VLM entry: run the vision encoder on FPGA or host, then stash the prompt token
        ids + image soft-token features so run_prefill's merge hooks splice them
        into the prompt embeddings (mm_token_type_ids == 1 → image positions).
        ``profile`` forwards to the encoder's per-phase FPGA-latency breakdown."""
        if prompt is None:
            prompt = "Describe this image in detail."
        if vision_host:
            image_features, token_ids, mm_types = self._run_vision_encoder_host(
                image_path, prompt)
        else:
            image_features, token_ids, mm_types = self._run_vision_encoder_fpga(
                image_path, prompt, profile=profile)
        self.prefill_seq = tuple(token_ids)
        self._image_features = image_features   # [N_soft, VIS_TEXT_H]
        self._mm_types = mm_types
        _n_img = int((torch.tensor(mm_types) == 1).sum().item())
        print(f"VLM prefill: {len(token_ids)} tokens ({_n_img} image, {len(token_ids) - _n_img} text)")

    def set_prefill_seq_audio(self, audio_path: str,
                              prompt: str | None = None) -> None:
        """Run the FPGA audio encoder and stage its soft tokens for LM prefill."""
        features, token_ids, mm_types = self._run_audio_encoder_fpga(
            audio_path, prompt or "Describe what you hear.")
        self.prefill_seq = tuple(token_ids)
        self._audio_features = features
        self._mm_types = mm_types
        print(f"Audio prefill: {len(token_ids)} tokens, "
              f"{sum(kind == 3 for kind in mm_types)} audio tokens")

    def _program_image_paths(self, profile: bool = False) -> tuple[str, str]:
        bin_dir = os.path.join(self.script_dir, "gemma4_e2b_bin")
        stem = "programs_profile" if profile else "programs"
        return (os.path.join(bin_dir, stem + ".bin"),
                os.path.join(bin_dir, stem + ".json"))

    # ------------------------------------------------------------------
    # Combined program image: ONE programs.bin holds every ISA section (vision
    # @ 0xa0000000, LM @ 0xc0000000, ...), each at its own baked DRAM base. The
    # bytes and addresses of each section are identical to a standalone build —
    # only the on-disk packaging is shared. programs.json carries a `sections`
    # map {name: {file_offset, size, dram_base, ...extra}}. Compiles merge their
    # section in (preserving others); loaders DMA a section to its dram_base.
    # ------------------------------------------------------------------
    def _read_program_sections(self, profile: bool = False) -> tuple[dict, bytes]:
        """Return ({name: section_meta}, whole_bin_bytes). Empty dict if none."""
        bin_path, meta_path = self._program_image_paths(profile)
        if not (os.path.exists(bin_path) and os.path.exists(meta_path)):
            return {}, b""
        with open(meta_path) as f:
            sections = json.load(f).get("sections", {})
        with open(bin_path, "rb") as f:
            data = f.read()
        return sections, data

    def _get_program_section(self, name: str, profile: bool = False):
        """Return (section_meta, section_bytes) for `name`, or (None, None)."""
        sections, data = self._read_program_sections(profile)
        s = sections.get(name)
        if s is None:
            return None, None
        return s, data[s["file_offset"]: s["file_offset"] + s["size"]]

    def _store_program_section(self, name: str, dram_base: int, section_bytes: bytes,
                               extra_meta: dict, profile: bool = False) -> None:
        """Merge one section into the combined programs bin, preserving every
        other section written by THIS run. Rewrites the file so offsets stay
        contiguous; sections are ordered by dram_base (vision 0xa0 before LM 0xc0).

        Fresh-compile hygiene: on a fresh run (``bin_reuse`` False) the FIRST
        section stored to a given image path in this process starts from an EMPTY
        image, so sections a previous run left behind are dropped — e.g. the
        vision_worker*/prefill_worker* sections from an earlier multi-core run
        when the current run is single-core. Without this, a fresh single-core
        image only overwrites its own vision/lm sections and the stale worker
        sections keep inflating the file (and the reported programs.bin size).
        Later stores in the same run merge normally, so this run's vision + LM
        (+ its own workers) all land. With ``--bin-reuse`` nothing is dropped:
        cached sections are preserved and only the recompiled one is replaced.
        """
        bin_path, meta_path = self._program_image_paths(profile)
        started = getattr(self, "_programs_bin_started", None)
        if started is None:
            started = self._programs_bin_started = set()
        if (not getattr(self, "bin_reuse", False)) and (bin_path not in started):
            sections, data = {}, b""          # fresh run: first store starts clean
        else:
            sections, data = self._read_program_sections(profile)
        started.add(bin_path)
        blobs = {k: data[s["file_offset"]: s["file_offset"] + s["size"]] for k, s in sections.items()}
        metas = {k: {kk: vv for kk, vv in s.items() if kk not in ("file_offset", "size")}
                 for k, s in sections.items()}
        blobs[name] = bytes(section_bytes)
        metas[name] = {"dram_base": f"0x{dram_base:X}", **extra_meta}
        order = sorted(blobs, key=lambda k: int(metas[k]["dram_base"], 16))
        out = bytearray()
        new_sections = {}
        for k in order:
            b = blobs[k]
            new_sections[k] = {"file_offset": len(out), "size": len(b), **metas[k]}
            out.extend(b)
        os.makedirs(os.path.dirname(bin_path), exist_ok=True)
        bin_tmp, meta_tmp = bin_path + ".tmp", meta_path + ".tmp"
        with open(bin_tmp, "wb") as f:
            f.write(bytes(out)); f.flush(); os.fsync(f.fileno())
        with open(meta_tmp, "w") as f:
            json.dump({"sections": new_sections}, f, indent=2); f.flush(); os.fsync(f.fileno())
        os.rename(bin_tmp, bin_path); os.rename(meta_tmp, meta_path)

    def _load_program_section(self, name: str, profile: bool = False) -> dict:
        """DMA the named section's bytes to its baked DRAM base and advance the
        program cursor past it. Returns the section meta (with int dram_base).
        Raises if the section is absent."""
        meta, sect_bytes = self._get_program_section(name, profile)
        if meta is None:
            raise FileNotFoundError(f"program section {name!r} not found in combined programs bin")
        base = int(meta["dram_base"], 16)
        self._next_program_dram_addr = base
        written = self.dma_write(DMA_DEVICE_H2C, base, sect_bytes, len(sect_bytes))
        if written != len(sect_bytes):
            raise RuntimeError(f"section {name}: DMA wrote {written} of {len(sect_bytes)} bytes")
        self.allocate_program_dram(len(sect_bytes))
        return {**meta, "_dram_base_int": base}

    def compile_gemma4(self, layer_size: int = 35, profile: bool = False) -> None:
        """Capture [prefill][decoder] into one combined image and write it to
        disk (gemma4_e2b_bin/programs.bin + programs.json). No DRAM is touched —
        run_gemma4() loads it.

        Both decoder and prefill are sequence-length agnostic. Prefill is
        captured once with runtime GPR dimensions. FLOP accounting uses the
        current prompt length; the sole legacy preparation matmul remains at
        the configured template maximum.

        set_prefill_seq() MUST have been called first (prefill needs the prompt).
        """
        assert self.prefill_seq is not None, (
            "call set_prefill_seq() before compile_gemma4()")
        prefill_template_seq_len = int(self._cfg["model"].get(
            "prefill_max_seq_len", self.max_prefill_seq_len))
        prefill_flops_seq_len = len(self.prefill_seq) - 1
        if self.multi_core > 1 and prefill_flops_seq_len < self.multi_core:
            raise ValueError(
                f"{self.multi_core}-engine fixed prefill requires at least "
                f"{self.multi_core} prefill rows")
        prefill_scheduler = self._ensure_prefill_scheduler()
        _lm_meta, _ = self._get_program_section("lm", profile)
        _prefill_worker_metas = [
            self._get_program_section(f"prefill_worker{i}", profile)[0]
            for i in range(1, self.multi_core)
        ]
        if (self.bin_reuse
                and _lm_meta is not None
                and _lm_meta.get("prefill_flops_seq_len") == prefill_flops_seq_len
                and _lm_meta.get("prefill_kernel") == self.prefill_kernel
                and _lm_meta.get("decode_kernel") == self.decode_kernel
                and _lm_meta.get("multi_core", 1) == self.multi_core
                and all(meta is not None
                        and meta.get("prefill_seq_len") == prefill_flops_seq_len
                        and meta.get("prefill_kernel") == self.prefill_kernel
                        for meta in _prefill_worker_metas)):
            bin_path, _ = self._program_image_paths(profile)
            print(f"[compile] reusing existing instruction image at {bin_path}")
            print(f"  delete {bin_path} (or make clean) to force recompile.")
            self._lm_compile_reused = True
            return
        self._lm_compile_reused = False

        print(f"[compile] building combined [prefill-template@{prefill_template_seq_len}, "
              f"kernel={self.prefill_kernel}][decoder, kernel={self.decode_kernel}] image...")
        self._set_silent(True)
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
            if prefill_scheduler is not None:
                prefill_scheduler.begin_program()
                self._prefill_shard_m_regs = [self.alloc_isa_reg()]
                self._prefill_shard_m_regs.extend(
                    worker.alloc_isa_reg() for worker in prefill_scheduler.workers)
                self._active_prefill_scheduler = prefill_scheduler
            _, prefill_total_flops = self.compile_prefill(seq_len=prefill_flops_seq_len,
                                                          layer_size=layer_size,
                                                          profile=profile)
            prefill_program_addr = instruction_base_addr + prefill_count_at_start * INSTRUCTION_SIZE_BYTES
            prefill_size_bytes = (self.capture_count - prefill_count_at_start) * INSTRUCTION_SIZE_BYTES
            prefill_worker_addrs = (prefill_scheduler.finalize()
                                    if prefill_scheduler is not None else [])
            self._active_prefill_scheduler = None
            if prefill_scheduler is not None:
                for worker in reversed(prefill_scheduler.workers):
                    worker.release_isa_reg()
                self.release_isa_reg()
                self._prefill_shard_m_regs = None

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
            self._set_silent(False)
            builtins.print = _orig_builtin_print
            # Capture must leave the program cursor untouched at the base so a
            # subsequent load lands the image where its jumps were baked.
            self._next_program_dram_addr = instruction_base_addr

        lm_meta = {
            "instruction_base_addr": f"0x{instruction_base_addr:X}",
            "prefill_template_seq_len": prefill_template_seq_len,
            "prefill_flops_seq_len": prefill_flops_seq_len,
            "prefill_program_start_addr": f"0x{prefill_program_addr:X}",
            "prefill_program_size": prefill_size_bytes,
            "prefill_total_flops": prefill_total_flops,
            "decoder_program_start_addr": f"0x{decoder_program_addr:X}",
            "decoder_program_size": decoder_size_bytes,
            "decoder_total_flops": decoder_total_flops[0],
            "layer_size": layer_size,
            "prefill_kernel": self.prefill_kernel,
            "multi_core": self.multi_core,
            "decode_kernel": self.decode_kernel,
        }
        if profile:
            lm_meta["prefill_profile_checkpoints"] = self._prefill_checkpoints
            lm_meta["decoder_profile_checkpoints"] = self._decoder_checkpoints
        # Merge the LM section into the combined programs bin (vision stays intact).
        self._store_program_section("lm", instruction_base_addr, image_bytes, lm_meta, profile=profile)
        if prefill_scheduler is not None:
            for engine_idx, (worker, worker_addr) in enumerate(
                    zip(prefill_scheduler.workers, prefill_worker_addrs), start=1):
                worker_bytes = bytearray()
                for inst in worker.capture_buffer:
                    worker_bytes.extend(inst.get_bytes())
                worker_limit = (worker_addr + self.MULTICORE_WORKER_ISA_STRIDE
                                if self.multi_core > 2 else self.LM_ISA_BASE)
                if worker_addr + len(worker_bytes) > worker_limit:
                    raise RuntimeError(
                        f"prefill core{engine_idx} ISA overflow: "
                        f"0x{worker_addr + len(worker_bytes):X} > 0x{worker_limit:X}")
                self._store_program_section(
                    f"prefill_worker{engine_idx}", worker_addr, worker_bytes,
                    {"parent": "lm", "engine_idx": engine_idx,
                     "multi_core": self.multi_core,
                     "prefill_seq_len": prefill_flops_seq_len,
                     "prefill_kernel": self.prefill_kernel}, profile=profile)

        print(f"[compile] stored LM section ({len(image_bytes)/1024:.1f} KB @ 0x{instruction_base_addr:X}); "
              f"prefill @ 0x{prefill_program_addr:X} ({prefill_size_bytes/1024:.1f} KB), "
              f"decoder @ 0x{decoder_program_addr:X} ({decoder_size_bytes/1024:.1f} KB)")

    def run_gemma4(self) -> tuple[int, float, float, float, float, float]:
        """Load the combined image from disk into program DRAM, then run one
        prefill pass and decode to the stop token / context limit — the gemma4
        analogue of run_gemma3() (load .bin, preamble-dispatch prefill + decode).
        Requires a prior compile_gemma4().

        Returns (token_cnt_decoded, latency_hw_prefill, latency_hw_decoder,
        flop_rate_hw_decoder, wallclock_prefill_s, wallclock_decoder_s).
        """
        # Load the LM section ([prefill][decoder]) from the combined programs bin
        # into program DRAM at its baked base (PBI jump targets resolve against
        # it), then park the single dispatch preamble just past it.
        meta = self._load_program_section("lm")
        base_addr = meta["_dram_base_int"]
        prefill_program_addr = int(meta["prefill_program_start_addr"], 16)
        decoder_program_addr = int(meta["decoder_program_start_addr"], 16)
        self._preamble_addr = self.get_program_dram_addr()
        print(f"[run] loaded LM section at 0x{base_addr:X} ({meta['prefill_program_size']/1024:.1f} + "
              f"{meta['decoder_program_size']/1024:.1f} KB); dispatch preamble @ 0x{self._preamble_addr:X}")

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
        # Stash prefill metrics for the run-summary writer (write_run_summary).
        self._prefill_seq_len = self.seq_len
        self._prefill_latency_hw_us = latency_hw_prefill
        self._prefill_flop_rate_gflops = _flop_rate_hw_prefill
        self._prefill_total_flops = meta["prefill_total_flops"]
        self._prefill_e2e_s = latency_prefill
        print(f"Prefill execute done in {latency_prefill:.2f} seconds, start decoding...\n", flush=True)

        timer = time.perf_counter()
        token_cnt_decoded, latency_hw_decoder, flop_rate_hw_decoder = self.run_decoder(
            [meta["decoder_program_size"]], decoder_program_addr,
            token_id=self.prefill_seq[-1], flops_per_token=[meta["decoder_total_flops"]])
        latency_decoder = time.perf_counter() - timer
        return (token_cnt_decoded, latency_hw_prefill, latency_hw_decoder,
                flop_rate_hw_decoder, latency_prefill, latency_decoder)

    def write_run_summary(self, out_path: str, args) -> str:
        """Write a per-run Markdown summary (weights/program sizes, per-stage HW
        latency + FLOPS, decode throughput, and the prompt/decoded text) built
        from the metrics stashed on ``self`` during compile_gemma4 / run_gemma4
        (and the vision encoder in --image mode). ``args`` is the parsed CLI
        namespace. Returns the written path.

        Everything read here is either an already-computed attribute or cheap
        host-side bookkeeping (file sizes, program-section metadata, a register
        read) — no FPGA program is launched, so calling this after a run is free.
        """
        clock_ns = self._clock_period_ns
        freq_mhz = 1000.0 / clock_ns if clock_ns else 0.0
        cores = self.multi_core
        peak_gflops = freq_mhz * 0.128 * cores   # freq(MHz) * 128 MAC/cyc/1000 * cores
        try:
            hw_version = self.user_read_reg32(user_dma_core.UE_FPGA_VERSION_ADDR) & 0xFFFFFFFF
            hw_version_str = f"0x{hw_version:08x}"
        except Exception as e:
            hw_version_str = f"(read failed: {e})"

        # Weights.
        weight_bin_path = os.path.join(self.script_dir, self._weights_bin_rel)
        weight_bin_size = os.path.getsize(weight_bin_path) if os.path.exists(weight_bin_path) else 0
        total_weight_dram_mb = getattr(
            self, "_lm_weight_dram_bytes", self.get_params_dram_usage()) / (1024 * 1024)

        # Program image + per-section sizes.
        sections, prog_data = self._read_program_sections()
        prog_bin_path, _ = self._program_image_paths()
        prog_bin_size = os.path.getsize(prog_bin_path) if os.path.exists(prog_bin_path) else 0
        vision_sect = sections.get("vision")
        lm_sect = sections.get("lm")
        vision_prog_size = vision_sect["size"] if vision_sect else None
        prefill_prog_size = lm_sect.get("prefill_program_size") if lm_sect else None
        decoder_prog_size = lm_sect.get("decoder_program_size") if lm_sect else None

        def _mb(n):
            return f"{n / (1024 * 1024):.2f} MB" if n is not None else "n/a"
        def _kb(n):
            return f"{n / 1024:.1f} KB" if n is not None else "n/a"

        is_image = bool(getattr(args, "image", None))
        is_audio = bool(getattr(args, "audio", None))

        lines = []
        lines.append(f"# gemma4_e2b_test run summary")
        lines.append("")
        lines.append(f"- **HW version:** {hw_version_str}")
        lines.append(f"- **--dev:** {args.dev}")
        lines.append(f"- **--device:** {args.device}")
        lines.append(f"- **Clock / frequency:** {clock_ns:.4f} ns ({freq_mhz:.1f} MHz)")
        lines.append(f"- **Cores (--multi-core):** {cores}")
        lines.append(f"- **Peak throughput:** {peak_gflops:.2f} GFLOPS "
                     f"({freq_mhz:.1f} MHz × 128 × {cores} core(s))")
        lines.append("")

        # --- Weights ---------------------------------------------------------
        lines.append(f"## Weights")
        lines.append("")
        lines.append(f"- **Weight bin:** `{os.path.basename(weight_bin_path)}` — {_mb(weight_bin_size)}")
        lines.append(f"- **Total weight DRAM (quantized, on FPGA):** {total_weight_dram_mb:.1f} MB")
        lines.append("")

        # --- Program image ---------------------------------------------------
        lines.append(f"## Program image (`{os.path.basename(prog_bin_path)}`)")
        lines.append("")
        lines.append(f"- **Total programs.bin size:** {_mb(prog_bin_size)}")
        if is_image:
            lines.append(f"- **Vision section:** {_mb(vision_prog_size)}")
        lines.append(f"- **Prefill program:** {_kb(prefill_prog_size)}")
        lines.append(f"- **Decoder program:** {_kb(decoder_prog_size)}")
        reuse_bits = []
        if is_image and getattr(self, "_vision_compile_reused", None) is not None:
            reuse_bits.append(f"vision {'reused' if self._vision_compile_reused else 'fresh'}")
        if getattr(self, "_lm_compile_reused", None) is not None:
            reuse_bits.append(f"prefill/decode {'reused' if self._lm_compile_reused else 'fresh'}")
        lines.append(f"- **Program image:** {', '.join(reuse_bits) if reuse_bits else 'n/a'}")
        lines.append("")

        # --- Vision ----------------------------------------------------------
        if is_image:
            vis_kernel = "host" if getattr(args, "vision_host", False) else self.vision_kernel
            vis_lat_us = getattr(self, "_vis_last_latency_us", None)
            vis_gflops = getattr(self, "_vis_last_gflops", None)
            lines.append(f"## Vision")
            lines.append("")
            lines.append(f"- **Vision kernel:** {vis_kernel}")
            lines.append(f"- **Vision tokens (soft tokens):** {getattr(self, '_vis_num_soft_tokens', 'n/a')}")
            if vis_lat_us is not None:
                lines.append(f"- **Vision FPGA run time (HW latency):** {vis_lat_us / 1e3:.2f} ms "
                             f"({vis_lat_us:.0f} us)")
            else:
                lines.append(f"- **Vision FPGA run time (HW latency):** n/a (host vision)")
            if vis_gflops is not None:
                lines.append(f"- **Vision reported FLOPS:** {vis_gflops:.2f} GFLOPS")
            else:
                lines.append(f"- **Vision reported FLOPS:** n/a (host vision)")
            _vis_e2e = getattr(self, "_vis_e2e_s", None)
            lines.append(f"- **Vision end-to-end (CPU timer):** "
                         f"{_vis_e2e:.2f} s" if _vis_e2e is not None else
                         "- **Vision end-to-end (CPU timer):** n/a")
            lines.append("")

        # --- Prefill ---------------------------------------------------------
        pf_seq = getattr(self, "_prefill_seq_len", None)
        pf_lat_us = getattr(self, "_prefill_latency_hw_us", None)
        pf_gflops = getattr(self, "_prefill_flop_rate_gflops", None)
        pf_e2e = getattr(self, "_prefill_e2e_s", None)
        lines.append(f"## Prefill")
        lines.append("")
        lines.append(f"- **Prefill seq_len:** {pf_seq if pf_seq is not None else 'n/a'}")
        if pf_lat_us is not None:
            lines.append(f"- **Prefill FPGA run time (HW latency):** {pf_lat_us / 1e3:.2f} ms "
                         f"({pf_lat_us:.0f} us)")
        if pf_gflops is not None:
            lines.append(f"- **Prefill reported FLOPS:** {pf_gflops:.2f} GFLOPS")
        if pf_e2e is not None:
            lines.append(f"- **Prefill end-to-end (CPU timer):** {pf_e2e:.2f} s")
        lines.append("")

        # --- Decode ----------------------------------------------------------
        gen_n = getattr(self, "_decode_generated_n", None)
        total_tok = self.seq_len
        peak_toks = getattr(self, "_decode_peak_toks", None)
        avg_toks = getattr(self, "_decode_avg_toks", None)
        dec_flop_rate = getattr(self, "_decode_total_flop_rate", None)
        avg_gflops = (dec_flop_rate / gen_n) if (dec_flop_rate is not None and gen_n) else None
        lines.append(f"## Decode")
        lines.append("")
        lines.append(f"- **Decoded tokens:** {gen_n if gen_n is not None else 'n/a'} generated "
                     f"(sequence total {total_tok})")
        if peak_toks is not None:
            lines.append(f"- **First-token speed (peak):** {peak_toks:.2f} tok/s")
        if avg_toks is not None:
            lines.append(f"- **Average speed:** {avg_toks:.2f} tok/s")
        if avg_gflops is not None:
            lines.append(f"- **Average FLOPS:** {avg_gflops:.2f} GFLOPS")
        lines.append("")

        # --- Text ------------------------------------------------------------
        try:
            prompt_text = self.tokenizer.decode(list(self.prefill_seq), skip_special_tokens=False)
        except Exception:
            prompt_text = "(decode failed)"
        decoded_text = getattr(self, "_decoded_text", None) or "(none)"
        lines.append(f"## Prompt & output")
        lines.append("")
        lines.append(f"### Full prefill prompt")
        lines.append("")
        lines.append("```")
        lines.append(prompt_text)
        lines.append("```")
        lines.append("")
        lines.append(f"### Decoded text")
        lines.append("")
        lines.append("```")
        lines.append(decoded_text)
        lines.append("```")
        lines.append("")

        with open(out_path, "w") as f:
            f.write("\n".join(lines))
        return out_path

    def _profile_execute(self, gpr_sets: list[tuple[int, int]], target_addr: int,
                         checkpoints: list, tail_name: str, timeout: float = 120.0,
                         worker_scheduler=None, worker_addrs=None) -> list:
        """Run a checkpointed program segment-by-segment, returning [(name, ms)]
        HW latency per segment. A preamble at self._preamble_addr primes each
        (reg, value) in ``gpr_sets`` then jumps into ``target_addr``; each
        checkpoint HALT stops the FPGA so the per-segment latency counter can be
        read before resuming from the recorded address. The final ``tail_name``
        segment covers everything after the last checkpoint up to the program's
        terminal HALT. Because the resume addresses are the instruction right
        after each HALT, the segments tile the whole program with no gaps, so the
        summed latencies cover all FPGA execution (see run_gemma4_profile).

        Two-engine (``worker_scheduler``/``worker_addrs``): the worker runs its
        own continuous shard stream once. Master checkpoints sit at the sharded-
        region boundaries, so a master HALT lands while the worker is parked at the
        next region's entry flag — the master's per-segment counter then measures
        each region's fork-to-join wall-time and the master-only phases directly."""
        self.clear_inst_id()
        self.start_capture()
        for reg, val in gpr_sets:
            self.generate_instruction_add_set(reg, val)
        self.generate_instruction_jump_abs(ue_35bit_addr_shifter(target_addr))
        self.stop_capture()
        self.write_captured_instructions_to_dram(self._preamble_addr)
        self.clear_capture_buffer()

        results = []
        if worker_scheduler is not None:
            worker_scheduler.preclear_flags()
            worker_scheduler.start_workers(worker_addrs or [])
        self.start_execute_from_dram(self._preamble_addr)
        for name, resume_hex in checkpoints:
            self.wait_queue(timeout)
            results.append((name, self.report_latency_in_us() / 1e3))   # ms
            self.start_execute_from_dram(int(resume_hex, 16))
        self.wait_queue(timeout)   # tail segment: everything up to the terminal HALT
        results.append((tail_name, self.report_latency_in_us() / 1e3))
        for worker in (worker_scheduler.workers if worker_scheduler is not None else []):
            worker.wait_queue(timeout)
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
        meta = self._load_program_section("lm", profile=True)
        base_addr = meta["_dram_base_int"]
        prefill_program_addr = int(meta["prefill_program_start_addr"], 16)
        decoder_program_addr = int(meta["decoder_program_start_addr"], 16)
        prefill_checkpoints = meta.get("prefill_profile_checkpoints", [])
        decoder_checkpoints = meta["decoder_profile_checkpoints"]

        self._preamble_addr = self.get_program_dram_addr()
        print(f"[profile] loaded LM profile section at 0x{base_addr:X}, "
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
        pli_embed = self._lookup_per_layer_embeddings([token_id])
        self.dma_to_accelerator_memory(
            self.PER_LAYER_EMBED_DRAM,
            pli_embed)
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

    @staticmethod
    def _parse_offset(val) -> int:
        """Delegate to the module-level helper so the LM method group (in a
        sibling file) reaches it via self without importing this module."""
        return _parse_offset(val)

    @staticmethod
    def _weight_bin_generate(**kwargs) -> str:
        """Delegate to the module-level params.bin generator (see _parse_offset)."""
        return weight_bin_generate(**kwargs)


def _clock_ns_default_for_device(device: str) -> float:
    """Return the board-profile clock period, matching user_hw_test/Llama."""
    if device == "kintex7":
        return 1000 / (1066 / 5.375)
    if device == "kintex7_systolic":
        return 1000 / 149.61403
    if device in ("rk", "rk_256", "puzhi"):
        return 3.0
    if device in ("bittware", "bittware_256"):
        return 3.3333
    if device == "alveo":
        return 1000 / 366.666666
    if device == "alveo_u55c":
        return 3.3333333
    if device == "efinix":
        return 4.0
    return 10.0


# --- Shared engine CLI/config -------------------------------------------------
# add_engine_args + resolve_engine_config are the SINGLE source of truth for the
# kernel/device knobs, their validation, and the device->AXI->clock wiring.
# gemma4_e2b_numeric.py imports this module and calls both, so anything changed
# here is reflected there automatically (no second arg-parser to keep in sync).
def add_engine_args(parser) -> None:
    """Register the kernel + device knobs shared by every gemma4_e2b entrypoint."""
    parser.add_argument("--vision-kernel", choices=("streaming", "matmatmul"),
                        default="matmatmul",
                        help="Quantized projection kernel for FPGA vision (default: matmatmul).")
    parser.add_argument("--prefill-kernel", choices=("streaming", "matmatmul"),
                        default="streaming",
                        help="Quantized projection kernel for LM prefill (default: streaming).")
    parser.add_argument("--decode-kernel", choices=("streaming", "matmatmul"),
                        default="streaming",
                        help="Quantized projection kernel for LM decode, including LM head "
                             "(default: streaming).")
    parser.add_argument("--multi-core", nargs="?", const=2, default=1, type=int,
                        help="Enable multi-engine vision and LM prefill. Bare --multi-core "
                             "selects 2 engines; --multi-core 8 selects 8 (Alveo only). "
                             "Multicore prefill always uses matmatmul.")
    parser.add_argument("--dev", type=str, default="xdma0",
                        help="DMA device name (e.g., xdma0, xdma1). Default: xdma0")
    parser.add_argument("--device", type=str, default="kintex7",
                        help="FPGA board profile (kintex7, kintex7_systolic, rk, "
                             "rk_256, puzhi, bittware, bittware_256, alveo, "
                             "alveo_u55c, efinix).")
    parser.add_argument("--cycle", type=float, default=None,
                        help="Clock cycle time in nanoseconds. Overrides --device default.")
    parser.add_argument("--bin-reuse", action="store_true",
                        help="Reuse a matching cached program image (programs.bin) if it "
                             "exists instead of recompiling. Default: OFF (fresh compile "
                             "every run).")


def resolve_engine_config(parser, args) -> dict:
    """Validate the shared knobs, wire device->AXI->clock, sync DMA globals, and
    return the kwargs dict for ``Gemma4_UnifiedEngine(...)``.

    Entrypoint-only cross-checks (``--profile`` vs ``--multi-core``,
    ``--vision-host`` requires ``--image``, etc.) stay in each ``main()``, before
    this runs. Every loaded ``gemma4_e2b_*`` module binds ``DMA_DEVICE_*`` by
    value at import, so all of them are refreshed after ``set_dma_device()``.
    """
    if args.multi_core not in range(1, 9):
        parser.error("--multi-core must be between 1 and 8")
    if args.multi_core > 2 and args.device != "alveo" and args.device != "alveo_u55c":
        parser.error("--multi-core values above 2 are currently supported only on Alveo and Alveo U55C")
    if args.multi_core > 1:
        # Multicore prefill only has a matmatmul (two-pass) shard path.
        args.prefill_kernel = "matmatmul"
    if args.multi_core > 1 and args.vision_kernel != "matmatmul":
        parser.error("--multi-core currently requires --vision-kernel matmatmul")

    axi_width_bits = 512 if args.device in ("bittware", "rk") else 256
    # Gemma4 vision has K <= 3072, so its two-pass kernel remains valid on
    # 512-bit AXI. LM prefill and decode include wide MLP-down K=12288 and
    # therefore cannot select matmatmul on that hardware profile.
    if axi_width_bits == 512 and (
            args.prefill_kernel == "matmatmul" or args.decode_kernel == "matmatmul"):
        requested = []
        if args.prefill_kernel == "matmatmul":
            requested.append("--prefill-kernel matmatmul")
        if args.decode_kernel == "matmatmul":
            requested.append("--decode-kernel matmatmul")
        parser.error(
            f"{' and '.join(requested)} unsupported: matmatmul is not supported "
            "on the 512-bit AXI data path; use streaming.")
    if os.environ.get("GEMMA4_PENALTY", "0") == "1":
        parser.error(
            "GEMMA4_PENALTY=1 is temporarily unsupported; dynamic streaming "
            "quantized_matmat_core needs broadcast-bias support first")

    dma_name = "efinix" if args.device == "efinix" else args.dev
    set_dma_device(dma_name)
    # Refresh DMA_DEVICE_* on every loaded gemma4_e2b_* module (they import the
    # names by value). Covers test/lm/vision/audio without a hand-kept list.
    for _name, _mod in list(sys.modules.items()):
        if _name.startswith("gemma4_e2b_") and _mod is not None:
            for _attr in ("DMA_DEVICE_H2C", "DMA_DEVICE_C2H", "DMA_DEVICE_USER"):
                if hasattr(_mod, _attr):
                    setattr(_mod, _attr, getattr(user_dma_core, _attr))

    os.environ["UE_AXI_DATA_WIDTH_BITS"] = str(axi_width_bits)
    user_dma_core.UE_AXI_DATA_WIDTH_BITS = axi_width_bits
    clock = args.cycle if args.cycle is not None else _clock_ns_default_for_device(args.device)
    user_dma_core.CLOCK_CYCLE_TIME_NS = clock
    user_dma_core.UE_PEAK_GFLOPS = 0.128 / clock

    print(f"FPGA profile: device={args.device}, clock={clock:.4f} ns, "
          f"UE_AXI_DATA_WIDTH_BITS={axi_width_bits}")
    print(f"Using DMA device: {dma_name}")
    print(f"  H2C: {user_dma_core.DMA_DEVICE_H2C}")
    print(f"  C2H: {user_dma_core.DMA_DEVICE_C2H}")
    print(f"  USER: {user_dma_core.DMA_DEVICE_USER}")
    print(f"Setting CLOCK_CYCLE_TIME_NS = {user_dma_core.CLOCK_CYCLE_TIME_NS}")
    print(f"Kernels: vision={args.vision_kernel}, prefill={args.prefill_kernel}, "
          f"engines={args.multi_core}, decode={args.decode_kernel}")
    print(f"Program image: {'reuse cached if present' if args.bin_reuse else 'fresh compile every run'}"
          f" (--bin-reuse {'on' if args.bin_reuse else 'off'})")

    return dict(
        vision_kernel=args.vision_kernel,
        prefill_kernel=args.prefill_kernel,
        decode_kernel=args.decode_kernel,
        multi_core=args.multi_core,
        bin_reuse=args.bin_reuse,
    )


def build_arg_parser():
    """The full gemma4_e2b CLI parser, shared as the SINGLE source of truth.

    gemma4_e2b_numeric.py takes the exact same args by calling this — it just
    runs a numeric SNR check instead of a decode, so any flag added here is
    available there automatically.
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="Gemma4 E2B LM prefill + decode on the accelerator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python gemma4_e2b_test.py                          # default prompt
  python gemma4_e2b_test.py --prompt "your prompt"   # custom prompt
  python gemma4_e2b_test.py --dev xdma1 --cycle 5.042

default prompt: "x+3=5, what is x?"
                (pre-tokenized as default_prefill_tokens in gemma4_e2b_config.json)""")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt. Default is the built-in test question.")
    parser.add_argument("--image", type=str, nargs="?", const=DEFAULT_IMAGE, default=None,
                        help="VLM mode: run the vision encoder on the FPGA and merge image "
                             f"soft-tokens into the prompt. Bare --image uses the default "
                             f"({os.path.basename(DEFAULT_IMAGE)}); omit it for LM-only mode.")
    parser.add_argument("--audio", type=str, nargs="?", const=DEFAULT_AUDIO, default=None,
                        help="Audio mode: run the FPGA audio encoder and merge its "
                             "soft tokens into LM prefill.")
    parser.add_argument("--vision-host", action="store_true",
                        help="With --image, generate image soft-token embeddings on the host "
                             "with the HF vision tower; FPGA prefill/decode are unchanged.")
    add_engine_args(parser)   # --vision-kernel/--prefill-kernel/--decode-kernel/--multi-core/--dev/--device/--cycle
    parser.add_argument('--profile', action='store_true',
                        help='Compile a profile bin with per-phase HALT checkpoints and run one '
                             'profiled decode step; print a per-phase HW-latency breakdown.')
    return parser


def run_summary_filename(args) -> str:
    """Build the per-run summary .md filename encoding the full CLI config, e.g.
    ``--dev xdma0 --device alveo --image --multi-core 2`` ->
    ``gemma4_e2b_test_xdma0_alveo_image_multi-core_2.md``.

    dev / device / mode are always present (in that order); every other engine
    knob is appended only when it differs from its default, so a plain LM run
    stays short. Call this BEFORE resolve_engine_config(), which mutates
    args.prefill_kernel for multi-core — otherwise an auto-forced kernel would
    leak into the name.
    """
    if args.audio:
        mode = "audio"
    elif args.image:
        mode = "image"
    else:
        mode = "lm"
    tokens = [args.dev, args.device, mode]
    if args.image and args.vision_host:
        tokens.append("vision-host")
    if args.multi_core != 1:
        tokens.append(f"multi-core_{args.multi_core}")
    if args.vision_kernel != "matmatmul":
        tokens.append(f"vision-kernel_{args.vision_kernel}")
    if args.prefill_kernel != "streaming":
        tokens.append(f"prefill-kernel_{args.prefill_kernel}")
    if args.decode_kernel != "streaming":
        tokens.append(f"decode-kernel_{args.decode_kernel}")
    if args.bin_reuse:
        tokens.append("bin-reuse")
    if args.cycle is not None:
        tokens.append(f"cycle_{args.cycle}")
    return "gemma4_e2b_test_" + "_".join(str(t) for t in tokens) + ".md"


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    # Entrypoint-only cross-checks (reference test.py-specific flags); the shared
    # kernel/device validation + wiring lives in resolve_engine_config().
    if args.vision_host and not args.image:
        parser.error("--vision-host requires --image")
    if args.image and args.audio:
        parser.error("--image and --audio are mutually exclusive")
    # Capture the summary filename now, before resolve_engine_config() mutates
    # args.prefill_kernel (multi-core forces matmatmul) — keeps auto-forced
    # kernels out of the name.
    _summary_name = run_summary_filename(args)
    engine_kwargs = resolve_engine_config(parser, args)
    ue = Gemma4_UnifiedEngine(**engine_kwargs)

    # Prompt first — the prefill program is compiled for its exact length.
    # VLM mode (--image): run the vision encoder on the FPGA now (separate bin
    # @ VISION_ISA_BASE); it sets prefill_seq + image soft-tokens that run_prefill
    # merges. The LM prefill/decoder then compiles at LM_ISA_BASE.
    if args.audio:
        if not os.path.isfile(args.audio):
            candidate = os.path.join(
                os.path.dirname(DEFAULT_AUDIO), os.path.basename(args.audio))
            if os.path.isfile(candidate):
                args.audio = candidate
        if not os.path.isfile(args.audio):
            raise SystemExit(f"--audio: file not found: {args.audio!r}")
        print(f"[Mode] audio (FPGA) -- file: {args.audio!r}")
        ue.set_prefill_seq_audio(args.audio, args.prompt)
    elif args.image:
        # Resolve a bare filename (e.g. --image people.jpg) against the shipped
        # test_samples dir where the default lives; existing/absolute paths pass
        # through unchanged.
        if not os.path.isfile(args.image):
            _cand = os.path.join(os.path.dirname(DEFAULT_IMAGE), os.path.basename(args.image))
            if os.path.isfile(_cand):
                args.image = _cand
        if not os.path.isfile(args.image):
            raise SystemExit(f"--image: file not found: {args.image!r} "
                             f"(also tried {os.path.dirname(DEFAULT_IMAGE)}/)")
        _vision_device = "host" if args.vision_host else "FPGA"
        print(f"[Mode] VLM ({_vision_device} vision) -- image: {args.image!r} "
              f"prompt: {args.prompt!r}")
        ue.set_prefill_seq_vlm(
            args.image, args.prompt, profile=args.profile, vision_host=args.vision_host)
    elif args.prompt:
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

    # Per-run Markdown summary, named for the full CLI config (see
    # run_summary_filename), written next to this script.
    _summary_path = os.path.join(SCRIPT_DIR, _summary_name)
    try:
        ue.write_run_summary(_summary_path, args)
        print(f"Wrote run summary: {_summary_path}")
    except Exception as _e:
        print(f"[warn] failed to write run summary: {_e}")
    print("Gemma4 E2B LM test ends.")


if __name__ == "__main__":
    main()
