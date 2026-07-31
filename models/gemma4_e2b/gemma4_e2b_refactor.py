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
  python gemma4_e2b_refactor.py --dev xdma0 [--cycle 5.042]
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

# Vision encoder matmuls (Q/K/V/O proj, MLP gate/up/down, patch/embed proj) are
# IF4-quantized, same codec as the LM path. Norms stay BF16. Kept module-level so
# a numeric harness (gemma4_e2b_numeric-style) can import it for HW-mimicking refs.
VISION_QUANT_PRECISION = "if4"

# Fixed canonical vision input: every image is resized to VISION_CANONICAL_SIZE
# before the HF processor so num_patches is always VISION_FIXED_NUM_PATCHES, which
# lets the vision encoder bin be compiled once for a fixed shape.
VISION_CANONICAL_SIZE = (896, 896)   # (width, height) for PIL.Image.resize
VISION_FIXED_NUM_PATCHES = 2520
LM_PROGRAM_CACHE_VERSION = 10

# Shipped sample image for VLM mode (same as gemma4_e2b_test.py's DEFAULT_IMAGE):
# repo test_samples/yosemite.jpg, two folders up from this script.
DEFAULT_IMAGE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "test_samples", "yosemite.jpg"))


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

    # Build the vision + host sections in memory (no separate files) and
    # concatenate into ONE weights bin: [LM | vision | host]. A single master
    # manifest JSON holds each section's offset + its per-tensor sub-manifest.
    # (Audio is still omitted — add an audio_section between vision and host when
    # that encoder lands.) weight_init reads section offsets from this manifest,
    # so a combined [LM | vision | audio | host] bin also loads for the LM path.
    vision_bytes, vision_manifest = _build_vision_section_bytes(model)
    host_bytes, host_manifest = _build_host_section_bytes(text_model, cfg)

    lm_size     = len(buf)
    vision_size = len(vision_bytes)
    host_size   = len(host_bytes)
    vision_off  = lm_size
    host_off    = lm_size + vision_size

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(buf)
        f.write(vision_bytes)
        f.write(host_bytes)
    total = lm_size + vision_size + host_size
    print(f"Generated weights bin: {out_path} ({total/1024**3:.2f} GiB total; "
          f"LM {lm_size/1024**3:.2f} GiB + vision {vision_size/1024**2:.1f} MiB + "
          f"host {host_size/1024**3:.2f} GiB)")

    master_meta_path = out_path.rsplit(".", 1)[0] + ".json"
    master = {
        "compile_version": "v1",
        "total_size": total,
        "lm_section":     {"offset": 0,          "size": lm_size},
        "vision_section": {"offset": vision_off, "size": vision_size, "manifest": vision_manifest},
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
    if not os.path.exists(config_path):
        _original_print(f"Downloading HF model {hf_repo} to {os.path.abspath(model_dir)} ...")
        snapshot_download(repo_id=hf_repo, local_dir=model_dir)
        _original_print("Download complete.")
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map=None, trust_remote_code=True)
    return model, model_dir


class Gemma4_UnifiedEngine(UnifiedEngine):
    """UnifiedEngine specialized to Gemma4 E2B (LM path): loads config + weight
    bin, compiles prefill/decoder into one bin, runs prefill + decode. Numeric
    checks live in gemma4_e2b_numeric.py."""

    # Vision encoder (SigLIP-style tower) constants — Gemma4 E2B.
    VIS_H = 768          # hidden size
    VIS_HEADS = 12       # attention heads
    VIS_HEAD_DIM = 64    # head_dim = 768 / 12
    VIS_MLP = 3072       # intermediate size
    VIS_LAYERS = 16      # num_hidden_layers
    VIS_ROPE_DIM = 32    # half of head_dim for 2D RoPE (64 / 2)

    def __init__(self, script_dir: str | None = None, local_weights: bool = False,
                 prefill_kernel: str = "streaming", decode_kernel: str = "streaming"):
        if prefill_kernel not in ("streaming", "matmatmul"):
            raise ValueError(f"unsupported prefill kernel: {prefill_kernel}")
        if decode_kernel not in ("streaming", "matmatmul"):
            raise ValueError(f"unsupported decode kernel: {decode_kernel}")
        if os.environ.get("GEMMA4_PENALTY", "0") == "1":
            raise RuntimeError(
                "GEMMA4_PENALTY=1 is temporarily unsupported: the dynamic streaming "
                "quantized_matmat_core must gain broadcast-bias support before the "
                "on-FPGA penalty can be re-enabled.")
        self.prefill_kernel = prefill_kernel
        self.decode_kernel = decode_kernel
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
        #   ISA Vision    : 0xa0000000 – 0xc0000000  (512 MB) ← vision_tensor_init
        #   ISA LM        : 0xc0000000 – 0x100000000 (1 GB)   ← program_base default
        #     Vision (a separate encoder bin) and the LM combined image live in
        #     DISJOINT program regions so both stay resident: vision runs first
        #     and produces soft-token features, then LM prefill+decode runs.
        #     The refactor's LM image is small (~7 MB, seq_len-agnostic dynamic
        #     PBI), so 0xc0000000 has ~1 GB headroom — no overflow (that only bit
        #     test.py's old monolithic LM+vision image).
        _params_base  = 0x00000000   # Weight region start
        _tensor_base  = 0x78000000   # Activation region start (stage scratch)
        _program_base = 0xc0000000   # LM ISA base; vision ISA sits at 0xa0000000
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

        # Vision-attention IDENTITY: unified_attention_core needs a
        # UE_VECTOR_SIZE² bf16 identity for its V-transpose. The LM one
        # (IDENTITY_DRAM_ADDR) lives in the tensor region that vision scratch
        # overwrites, so keep a copy in the PARAMS region (untouched by vision
        # scratch @ tensor DRAM and vision ISA @ 0xa0000000).
        self._vis_identity_dram = self.allocate_params_dram(
            UE_VECTOR_SIZE * UE_VECTOR_SIZE * self.bytes_per_element)
        self.dma_to_accelerator_memory(
            self._vis_identity_dram, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))

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
        """Initialize hardware DRAM for Gemma4 E2B model.

        Unique KV slots are packed using their owning layer's actual head
        dimension. Sliding-attention rows therefore use 256 elements and
        global-attention rows use 512 elements, matching unified attention's
        contiguous K/V layout directly.
        """
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
        self._kv_offset_for_layer = {}
        self._kv_row_bytes_for_layer = {}
        _kv_offset = 0
        for slot, l in enumerate(non_shared_layers):
            self._kv_slot_for_layer[l] = slot
            _head_dim, _, _ = self._get_layer_attention_dims(l)
            _row_bytes = _head_dim * self.bytes_per_element
            self._kv_offset_for_layer[l] = _kv_offset
            self._kv_row_bytes_for_layer[l] = _row_bytes
            _kv_offset += self.MAX_CONTEXT_SIZE * _row_bytes
        for shared_l, ref_l in self._kv_shared_map.items():
            self._kv_slot_for_layer[shared_l] = self._kv_slot_for_layer[ref_l]
            self._kv_offset_for_layer[shared_l] = self._kv_offset_for_layer[ref_l]
            self._kv_row_bytes_for_layer[shared_l] = self._kv_row_bytes_for_layer[ref_l]
            _shared_dim, _, _ = self._get_layer_attention_dims(shared_l)
            if _shared_dim * self.bytes_per_element != self._kv_row_bytes_for_layer[ref_l]:
                raise ValueError(
                    f"KV-shared layer {shared_l} head_dim={_shared_dim} does not match "
                    f"reference layer {ref_l} row size")
        self._num_kv_slots = len(non_shared_layers)
        self._kv_cache_bytes = _kv_offset
        _uniform_bytes = self._num_kv_slots * self.MAX_CONTEXT_SIZE * self.k_size
        _compact_saved = 2 * (_uniform_bytes - self._kv_cache_bytes)
        print(
            f"KV cache: {self._num_kv_slots} unique compact slots "
            f"({self._kv_cache_bytes * 2 / (1024*1024):.1f} MB K+V, "
            f"saved {_compact_saved / (1024*1024):.1f} MB vs padded slots)")
        self.LAYER0_V_DRAM = self.allocate_tensor_dram(self._kv_cache_bytes)
        self.LAYER0_K_ROPE_DRAM = self.allocate_tensor_dram(self._kv_cache_bytes)
        zero_pad = torch.zeros(self._kv_cache_bytes // self.bytes_per_element, dtype=torch.bfloat16)
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
        # Reserved for a future streaming LM-head penalty implementation.
        # GEMMA4_PENALTY is temporarily rejected.
        self.PENALTY_BIAS_DRAM = self.allocate_tensor_dram(1 * self.EMBEDDING_ELEMENTS * self.bytes_per_element)

        # Per-layer input injection buffers
        # PER_LAYER_INPUTS_DRAM: holds per_layer_inputs for all layers: MAX_CONTEXT_SIZE x 35 x 256 x 2 bytes
        self.PER_LAYER_INPUTS_DRAM = self.allocate_tensor_dram(self.MAX_CONTEXT_SIZE * self.LAYER_SIZE * self.per_layer_input_dim * self.bytes_per_element)
        # Intermediate DRAMs for per-layer injection
        self.LAYER0_PER_LAYER_GATE_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.per_layer_input_dim * self.bytes_per_element)
        self.LAYER0_PER_LAYER_PROJ_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * self.bytes_per_element)

        print(f"    Tensor DRAM usage: {self.get_tensor_dram_usage()/(1024*1024):.1f} MB")

    # ------------------------------------------------------------------
    # Vision encoder (S1): weight + tensor init. Ported from
    # gemma4_e2b_test.py. Vision runs before LM prefill and produces soft-token
    # features that run_prefill merges into the prompt embeddings.
    # ------------------------------------------------------------------
    def _vision_weight_init_from_combined_bin(self, weights_bin_path: str, vision_section: dict) -> None:
        """Load pre-quantized vision weights from the combined weights bin and
        DMA to FPGA (no HF model, no quantization). ``vision_section['manifest']
        ['sections']`` gives tensor offsets RELATIVE to the vision section start;
        we add ``vision_section['offset']`` (absolute file offset) when seeking.
        Allocation order MUST match _build_vision_section_bytes so DRAM addresses
        line up with the addresses baked into the vision ISA."""
        base_offset = int(vision_section["offset"])
        meta = vision_section["manifest"]
        if meta.get("vision_quant_precision") != VISION_QUANT_PRECISION:
            raise RuntimeError(
                f"vision section quant precision mismatch (disk: {meta.get('vision_quant_precision')!r}, "
                f"expected {VISION_QUANT_PRECISION!r}). Regenerate the weights bin.")
        self.VIS_ROPE_PERM = torch.tensor(meta["VIS_ROPE_PERM"], dtype=torch.long)
        self.VIS_POOL_K    = int(meta["VIS_POOL_K"])
        self.VIS_TEXT_H    = int(meta["VIS_TEXT_H"])
        L_count = int(meta["num_layers"])
        sections = meta["sections"]

        def _str_to_float(v):
            if v == "inf":  return float("inf")
            if v == "-inf": return -float("inf")
            return float(v)
        self._vis_clip_ranges = []
        for cr in meta["clip_ranges"]:
            row = {}
            for k, v in cr.items():
                row[k] = {
                    "input":  (_str_to_float(v["input"][0]),  _str_to_float(v["input"][1])),
                    "output": (_str_to_float(v["output"][0]), _str_to_float(v["output"][1])),
                }
            self._vis_clip_ranges.append(row)

        print(f"\n[Vision] Loading pre-quantized vision weights from combined bin "
              f"({VISION_QUANT_PRECISION.upper()} block=64 + BF16 norms) ...")
        with open(weights_bin_path, "rb") as f:
            def _dma_section(key: str) -> int:
                s = sections[key]
                f.seek(base_offset + s["offset"])
                bts = f.read(s["size"])
                if len(bts) != s["size"]:
                    raise RuntimeError(f"truncated section read {key} at offset {base_offset + s['offset']}")
                addr = self.allocate_tensor_dram(s["size"])
                self.dma_write(DMA_DEVICE_H2C, addr, bts, s["size"])
                return addr

            layer_weight_addrs = []
            for li in range(L_count):
                pre = f"layer{li}"
                addrs = {}
                for n in ["input_layernorm", "post_attention_layernorm",
                          "pre_feedforward_layernorm", "post_feedforward_layernorm",
                          "q_norm", "k_norm"]:
                    addrs[n] = _dma_section(f"{pre}.{n}")
                for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]:
                    scale_addr = _dma_section(f"{pre}.{proj_name}.scale")
                    data_addr  = _dma_section(f"{pre}.{proj_name}.data")
                    addrs[proj_name] = {
                        "data": data_addr, "scale": scale_addr,
                        "shape": tuple(sections[f"{pre}.{proj_name}.data"]["shape"]),
                    }
                layer_weight_addrs.append(addrs)
            self._vis_weight_addrs = layer_weight_addrs

            # Position embedding table stays on host (one-time gather in
            # vision_patch_embed). Allocation order below mirrors the bin builder.
            s = sections["pos_embedding_table"]
            f.seek(base_offset + s["offset"])
            bts = f.read(s["size"])
            self._vis_pos_embed_table = torch.frombuffer(
                bytearray(bts), dtype=torch.bfloat16).reshape(*s["shape"]).clone()

            self.VIS_EMBED_NORM_HAS_SCALE = False
            self.VIS_EMBED_NORM_GAMMA = _dma_section("embed_norm_gamma")

            for nm in ("patch_proj", "embed_proj"):
                scale_addr = _dma_section(f"{nm}.scale")
                data_addr  = _dma_section(f"{nm}.data")
                info = {"data": data_addr, "scale": scale_addr,
                        "shape": tuple(sections[f"{nm}.data"]["shape"])}
                if nm == "patch_proj":
                    self.VIS_PATCH_PROJ_INFO = info
                else:
                    self.VIS_EMBED_PROJ_INFO = info

            self.VIS_V_NORM_ONES_GAMMA = _dma_section("v_norm_ones_gamma")
        self._vis_weight_end = self.get_tensor_dram_addr()
        print(f"  Vision weights loaded. Tensor DRAM usage: "
              f"{self.get_tensor_dram_usage()/(1024*1024):.1f} MB")

    def vision_weight_init(self, hf_model=None) -> None:
        """Upload vision encoder weights to FPGA DRAM from the combined bin's
        pre-quantized vision section. Idempotent. Unlike test.py there is no
        HF-model fallback — the refactor's weight bin always carries the vision
        section (regenerate params.bin if this raises)."""
        if getattr(self, "_vision_weight_init_done", False):
            return
        master = getattr(self, "_weights_master", None)
        if not (master and "vision_section" in master
                and "manifest" in master["vision_section"]):
            raise RuntimeError(
                "vision section missing from the weights bin. Delete "
                "gemma4_e2b_bin/params.bin and re-run so weight_bin_generate "
                "emits the [LM | vision | host] layout.")
        weights_bin_path = os.path.join(self.script_dir, self._weights_bin_rel)
        self._vision_weight_init_from_combined_bin(weights_bin_path, master["vision_section"])
        self._vision_weight_init_done = True

    def vision_tensor_init(self, num_patches: int, *, program_base: int | None = None) -> None:
        """Allocate DRAM for vision encoder intermediate tensors. Reuses the LM
        tensor DRAM region (vision and LM never run simultaneously). Vision ISA
        goes to VISION_ISA_BASE unless ``program_base`` overrides it."""
        bpe = self.bytes_per_element
        H, HD, NH, MLP = self.VIS_H, self.VIS_HEAD_DIM, self.VIS_HEADS, self.VIS_MLP
        S = num_patches

        self.reset_tensor_dram_addr()
        VISION_ISA_BASE = 0xa0000000
        base = VISION_ISA_BASE if program_base is None else program_base
        self._next_program_dram_addr = base
        self._program_dram_base = base
        print(f"\n[Vision] Allocating vision tensor DRAM for {S} patches ...")

        # Layer I/O (double-buffered).
        self.VIS_IO_A = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_IO_B = self.allocate_tensor_dram(S * H * bpe)

        # Layer-0 checkpoint snapshot buffers (GEMMA4_VIS_L0_CKPT=1): the layer-0
        # ISA copies before-attn / per-stage / end-of-layer tensors here so the
        # numeric harness can read them back after the run.
        if os.environ.get("GEMMA4_VIS_L0_CKPT") == "1":
            aS = ((S + 63) // 64) * 64
            self.VIS_L0_BEFORE_ATTN = self.allocate_tensor_dram(S * H * bpe)
            self.VIS_L0_QPROJ       = self.allocate_tensor_dram(S * H * bpe)
            self.VIS_L0_KPROJ       = self.allocate_tensor_dram(S * H * bpe)
            self.VIS_L0_VPROJ       = self.allocate_tensor_dram(S * H * bpe)
            self.VIS_L0_QPRE        = self.allocate_tensor_dram(S * H * bpe)
            self.VIS_L0_KPRE        = self.allocate_tensor_dram(S * H * bpe)
            self.VIS_L0_VNORM       = self.allocate_tensor_dram(S * H * bpe)
            self.VIS_L0_ROPE_Q      = self.allocate_tensor_dram(S * H * bpe)
            self.VIS_L0_ROPE_K      = self.allocate_tensor_dram(S * H * bpe)
            self.VIS_L0_Q_HM        = self.allocate_tensor_dram(NH * aS * HD * bpe)
            self.VIS_L0_K_HM        = self.allocate_tensor_dram(NH * aS * HD * bpe)
            self.VIS_L0_V_HM        = self.allocate_tensor_dram(NH * aS * HD * bpe)
            self.VIS_L0_ATTN_OUT_HM = self.allocate_tensor_dram(NH * aS * HD * bpe)
            self.VIS_L0_ATTN_CORE   = self.allocate_tensor_dram(S * H * bpe)
            self.VIS_L0_AFTER_ATTN  = self.allocate_tensor_dram(S * H * bpe)
            self.VIS_L0_END         = self.allocate_tensor_dram(S * H * bpe)
            self.VIS_L0H0_Q   = self.allocate_tensor_dram(aS * HD * bpe)
            self.VIS_L0H0_K   = self.allocate_tensor_dram(aS * HD * bpe)
            self.VIS_L0H0_V   = self.allocate_tensor_dram(aS * HD * bpe)
            self.VIS_L0H0_OUT = self.allocate_tensor_dram(aS * HD * bpe)

        # Intermediates.
        self.VIS_NORM_OUT = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_Q_DRAM = self.allocate_tensor_dram(S * NH * HD * bpe)
        self.VIS_K_DRAM = self.allocate_tensor_dram(S * NH * HD * bpe)
        self.VIS_V_DRAM = self.allocate_tensor_dram(S * NH * HD * bpe)
        self.VIS_Q_NORM = self.allocate_tensor_dram(S * NH * HD * bpe)
        self.VIS_K_NORM = self.allocate_tensor_dram(S * NH * HD * bpe)
        self.VIS_ATTN_OUT = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_POST_ATTN_NORM = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_POST_ATTN_RES = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_PRE_FFN_NORM = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_MLP_GATE = self.allocate_tensor_dram(S * MLP * bpe)
        self.VIS_MLP_UP = self.allocate_tensor_dram(S * MLP * bpe)
        self.VIS_MLP_MULT = self.allocate_tensor_dram(S * MLP * bpe)
        self.VIS_MLP_DOWN = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_POST_FFN_NORM = self.allocate_tensor_dram(S * H * bpe)

        # Per-head unified-attention buffers (heads processed sequentially).
        aligned_S = ((S + 63) // 64) * 64
        self.VIS_FLASH_Q = self.allocate_tensor_dram(aligned_S * HD * bpe)
        self.VIS_FLASH_K = self.allocate_tensor_dram(aligned_S * HD * bpe)
        self.VIS_FLASH_V = self.allocate_tensor_dram(aligned_S * HD * bpe)
        self.VIS_FLASH_OUT = self.allocate_tensor_dram(aligned_S * HD * bpe)
        # BIAS before SCRATCH, guard-padded both sides (see test.py note re: PBI
        # flash scratch sizing / adjacent-buffer corruption).
        BIAS_GUARD_BYTES = 4 * 1024 * 1024
        self._vis_bias_guard_pre  = self.allocate_tensor_dram(BIAS_GUARD_BYTES)
        self.VIS_FLASH_BIAS = self.allocate_tensor_dram(aligned_S * aligned_S * bpe)
        self._vis_bias_guard_post = self.allocate_tensor_dram(BIAS_GUARD_BYTES)
        self.VIS_FLASH_SCRATCH = self.allocate_tensor_dram(
            (aligned_S * aligned_S + 2 * HD * aligned_S) * 2)
        # Unused by unified_attention_core but kept to keep the allocation order
        # (and every baked address after it) stable.
        self.VIS_ATTN_P = self.allocate_tensor_dram(aligned_S * aligned_S * bpe)

        # 64×64 identity for FPGA clamp passes.
        self.VIS_IDENTITY_64 = self.allocate_tensor_dram(64 * 64 * bpe)
        self.dma_to_accelerator_memory(self.VIS_IDENTITY_64, torch.eye(64, dtype=torch.bfloat16))
        self.VIS_INPUT_CLIP_H_SCRATCH   = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_INPUT_CLIP_MLP_SCRATCH = self.allocate_tensor_dram(S * MLP * bpe)

        # Head-major Q/K/V/OUT buffers (NH, aligned_S, HD) — FPGA transposes fill
        # these; the per-head attention loop reads offsets within them. Pre-zero
        # so padding rows [S, aligned_S) stay zero.
        hm_bytes = NH * aligned_S * HD * bpe
        self.VIS_FLASH_Q_HM   = self.allocate_tensor_dram(hm_bytes)
        self.VIS_FLASH_K_HM   = self.allocate_tensor_dram(hm_bytes)
        self.VIS_FLASH_V_HM   = self.allocate_tensor_dram(hm_bytes)
        self.VIS_FLASH_OUT_HM = self.allocate_tensor_dram(hm_bytes)
        zeros_hm = torch.zeros(NH * aligned_S * HD, dtype=torch.bfloat16)
        for a in (self.VIS_FLASH_Q_HM, self.VIS_FLASH_K_HM,
                  self.VIS_FLASH_V_HM, self.VIS_FLASH_OUT_HM):
            self.dma_to_accelerator_memory(a, zeros_hm)

        # Bidirectional bias: zeros with alignment-padding columns masked. Real
        # image padding patches (position_ids == -1) are masked later by
        # set_vision_attention_bias.
        bias = torch.zeros(aligned_S, aligned_S, dtype=torch.bfloat16)
        bias[:, S:] = float("-inf")
        self.dma_to_accelerator_memory(self.VIS_FLASH_BIAS, bias)

        # Legacy RoPE tables (kept for address stability).
        self.VIS_ROPE_COS = self.allocate_tensor_dram(S * HD * bpe)
        self.VIS_ROPE_SIN = self.allocate_tensor_dram(S * HD * bpe)
        # Three 32-wide (HD//2) RoPE tables for the FPGA 2D-RoPE split-64 path.
        self.VIS_ROPE_COS_PAD_TILED     = self.allocate_tensor_dram(S * (HD // 2) * bpe)
        self.VIS_ROPE_NEG_SIN_PAD_TILED = self.allocate_tensor_dram(S * (HD // 2) * bpe)
        self.VIS_ROPE_SIN_HI_PAD_TILED  = self.allocate_tensor_dram(S * (HD // 2) * bpe)

        # HD×HD identity for transpose/matmul helpers.
        self.VIS_IDENTITY = self.allocate_tensor_dram(HD * HD * bpe)
        self.dma_to_accelerator_memory(self.VIS_IDENTITY, torch.eye(HD, dtype=torch.bfloat16))

        # Embed-vision (pooler tail) scratch. N_soft is image-dependent but
        # bounded by S; size at S to be safe.
        text_h = getattr(self, "VIS_TEXT_H", 1536)
        self.VIS_EMBED_POOL = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_EMBED_NORMED = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_EMBED_OUT = self.allocate_tensor_dram(S * text_h * bpe)

        self._vis_num_patches = S
        self._vis_aligned_S = aligned_S
        self._vis_padding_mask = None
        print(f"  Vision tensor DRAM allocated. Total usage: {self.get_tensor_dram_usage()} bytes")

    # ------------------------------------------------------------------
    # Vision execution infra (S2): single-op compile/run, chunked eltwise,
    # matmul M-priming. The non-encoder vision ops (patch embed, projection,
    # clamps, eltwise) each run as their own tiny FPGA program; the 16-layer
    # encoder is captured as ONE program via _oneshot_mode (see S3).
    # ------------------------------------------------------------------
    def _prime_M(self, M: int) -> int:
        """Emit ADD_SET gpr_seq_len <- M and return that register index, for use
        as gpr_M_reg on a vision matmul / rms_norm. Folds the compile-time M
        tiling into one runtime ISA loop (bit-exact to the legacy static unroll).
        gpr_seq_len is LM-only, free in the vision program. Emits an instruction,
        so it must be evaluated inside the active capture."""
        self.generate_instruction_add_set(self.gpr_seq_len, M)
        return self.gpr_seq_len

    def _vis_mm_gpr(self, M: int):
        """gpr_M_reg for a vision matmul/norm: PBI M-loop by default (one-shot bin
        shrink), VIS_MATMUL_LEGACY=1 forces the static unroll. Returns None when
        emission is suppressed (no active capture)."""
        if os.environ.get("VIS_MATMUL_LEGACY") == "1":
            return None
        if self.capture_buffer is None:
            return None
        return self._prime_M(M)

    def _vis_flops_add(self, v) -> None:
        """Accumulate an op's returned FLOP count into self._vis_flops (matmuls,
        rms-norm, and attention return numeric FLOPs; other ops return None).
        No-op unless a caller (compile_vision_encoder_bin) is accumulating."""
        if isinstance(v, (int, float)) and getattr(self, "_vis_flops", None) is not None:
            self._vis_flops += v

    def _compile_and_run_single(self, label: str, compile_fn) -> None:
        """Capture ONE vision op's ISA, DMA it to program DRAM, execute, wait.
        Reuses the same program slot each call (reset_program_dram_addr).

        Oneshot mode (self._oneshot_mode=True): emit into the caller's already-open
        capture instead — used to build the 16-layer encoder as one program.
        Resets the inst-pointer + ISA-register allocators to their base first, so
        the per-op PBI cores (which alloc pointers/regs and never release them)
        can't climb across sub-ops and exhaust the pool."""
        if getattr(self, "_oneshot_mode", False):
            self.reset_inst_ptr_counter()
            self._isa_reg_counter = self._isa_reg_base
            self._vis_flops_add(compile_fn())
            return
        if os.environ.get("TRACE_SUBOP"):
            _original_print(f"      [subop] {label} ...", flush=True)
        global _SILENT_MODE
        _prev_silent = _SILENT_MODE
        _SILENT_MODE = True
        try:
            self.reset_program_dram_addr()
            self.start_capture()
            self._vis_flops_add(compile_fn())
            self.stop_capture()
            self.generate_instruction_halt()
            prog_addr = self.get_program_dram_addr()
            self.write_captured_instructions_to_dram(prog_addr)
            self.allocate_program_dram(self.get_capture_instruction_size_bytes())
            self.clear_capture_buffer()
        finally:
            _SILENT_MODE = _prev_silent
        self.start_execute_from_dram(prog_addr)
        self.wait_queue(120.0)

    def _run_eltwise_add_chunked(self, a_addr: int, b_addr: int, out_addr: int, num_elements: int) -> None:
        """Element-wise add two DRAM tensors, one SRAM-sized chunk per program."""
        CHUNK = 65536  # 128 KB/buffer — fits the SRAM gap 0x10000..0x90000
        bpe = self.bytes_per_element
        for off in range(0, num_elements, CHUNK):
            n = min(CHUNK, num_elements - off)
            def _fn(a=a_addr + off * bpe, b=b_addr + off * bpe, o=out_addr + off * bpe, sz=n):
                self.accelerator_memory_to_sram(a, 0x10000, sz)
                self.accelerator_memory_to_sram(b, 0x90000, sz)
                self.eltwise_add_core(0x10000, 0x90000, 0x10000, sz)
                self.sram_to_accelerator_memory(0x10000, o, sz)
            self._compile_and_run_single("eltwise_add_chunk", _fn)

    def _run_eltwise_mul_chunked(self, a_addr: int, b_addr: int, out_addr: int, num_elements: int) -> None:
        """Element-wise multiply two DRAM tensors, one SRAM-sized chunk per program."""
        CHUNK = 65536
        bpe = self.bytes_per_element
        for off in range(0, num_elements, CHUNK):
            n = min(CHUNK, num_elements - off)
            def _fn(a=a_addr + off * bpe, b=b_addr + off * bpe, o=out_addr + off * bpe, sz=n):
                self.accelerator_memory_to_sram(a, 0x10000, sz)
                self.accelerator_memory_to_sram(b, 0x90000, sz)
                self.eltwise_mul_core(0x10000, 0x90000, 0x10000, sz)
                self.sram_to_accelerator_memory(0x10000, o, sz)
            self._compile_and_run_single("eltwise_mul_chunk", _fn)

    def vision_patch_embed(self, pixel_values: torch.Tensor,
                           pixel_position_ids: torch.Tensor,
                           padding_positions: torch.Tensor) -> torch.Tensor:
        """Gemma4 vision patch embedder on FPGA. Mirrors
        Gemma4VisionPatchEmbedder.forward:
          scaled = 2*(pixel_values - 0.5)
          hidden = input_proj(scaled)                 # FPGA, IF4
          pos    = pos_table[0,x] + pos_table[1,y]    # host gather
          pos[padding] = 0
          return hidden + pos                          # FPGA, chunked eltwise add
        Uses VIS_IO_B (pixel scratch) + VIS_NORM_OUT (pos staging); output lands
        in VIS_IO_A. Returns the [S, H] patch embeddings (read back for numeric)."""
        S, H = self._vis_num_patches, self.VIS_H
        pv = pixel_values.squeeze(0) if (pixel_values.dim() == 3 and pixel_values.shape[0] == 1) else pixel_values
        assert pv.shape == (S, H), f"pixels shape {pv.shape}, expected ({S}, {H})"
        pids = pixel_position_ids.squeeze(0) if pixel_position_ids.dim() == 3 else pixel_position_ids
        pad = padding_positions.squeeze(0) if padding_positions.dim() == 2 else padding_positions

        scaled = (2.0 * (pv.float() - 0.5)).to(torch.bfloat16).contiguous()
        self.dma_to_accelerator_memory(self.VIS_IO_B, scaled)

        w = self.VIS_PATCH_PROJ_INFO
        self._compile_and_run_single("patch_input_proj", lambda: self.matmat_mul_core(
            M=S, K=H, N=H,
            A_DRAM_ADDR=self.VIS_IO_B, B_DRAM_ADDR=w["data"], OUTPUT_DRAM_ADDR=self.VIS_IO_A,
            is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=w["scale"],
            gpr_M_reg=self._vis_mm_gpr(M=S)))

        table = self._vis_pos_embed_table.float()             # [2, P, H]
        clamped = pids.clamp(min=0).long().cpu()              # [S, 2]
        pe_sum = table[0, clamped[:, 0]] + table[1, clamped[:, 1]]
        pe_sum[pad.cpu()] = 0.0
        self.dma_to_accelerator_memory(self.VIS_NORM_OUT, pe_sum.to(torch.bfloat16).contiguous())
        self._run_eltwise_add_chunked(self.VIS_IO_A, self.VIS_NORM_OUT, self.VIS_IO_A, S * H)

        return self.dma_from_accelerator_memory(self.VIS_IO_A, (S, H)).cpu()

    # ------------------------------------------------------------------
    # Vision encoder ISA primitives (S2b/S3): 2-D RoPE, Q/K/V head-major
    # transposes, PBI rms-norm, FPGA clamp. Ported from gemma4_e2b_test.py —
    # these are validated standalone (see test.py's test_vision_* helpers).
    # ------------------------------------------------------------------
    def _load_or_build_vision_rope_pads(self, hf_model, pixel_position_ids, num_patches):
        """Compute the three FPGA-ready 2D-RoPE tables (cos / neg_sin / sin_hi),
        each (num_patches, HD//2) bf16, for the canonical image grid. One row per
        PATCH (2D-RoPE is head-independent) and 32-wide (only half_rot is read)."""
        HD = self.VIS_HEAD_DIM
        print(f"  [Vision] [host] computing 2D-RoPE pads...", flush=True)
        t0 = time.perf_counter()
        with torch.no_grad():
            vt = hf_model.model.vision_tower
            patch_embeds_dummy = torch.zeros(
                1, num_patches, vt.config.hidden_size,
                dtype=torch.bfloat16, device=pixel_position_ids.device)
            cos_table, sin_table = vt.encoder.rotary_emb(patch_embeds_dummy, pixel_position_ids)
        cos_2d = cos_table.squeeze(0).cpu().to(torch.bfloat16)
        sin_2d = sin_table.squeeze(0).cpu().to(torch.bfloat16)
        first_half_idx = self.VIS_ROPE_PERM[:HD // 2]
        cos_tbl     = cos_2d[:, first_half_idx].contiguous()
        neg_sin_tbl = (-sin_2d[:, first_half_idx]).contiguous()
        sin_hi_tbl  = sin_2d[:, first_half_idx].contiguous()
        print(f"  [Vision] [host] 2D-RoPE pads computed in {time.perf_counter()-t0:.2f}s "
              f"(per-patch 32-wide)", flush=True)
        return cos_tbl, neg_sin_tbl, sin_hi_tbl

    def _emit_vision_rope_2d(self, src_dram: int, out_dram: int,
                             cos_pad_dram: int, neg_sin_pad_dram: int,
                             sin_hi_pad_dram: int, M: int) -> None:
        """Apply vision 2D RoPE to M consecutive HD=64 rows via the split-64
        permuted layout. cos/neg_sin/sin_hi hold ONE 32-wide row per PATCH,
        shared by that patch's VIS_HEADS heads (patch-outer / head-inner loop).
        Only the 32 valid cols are touched (SRAM [32:64] is never read)."""
        rot_dim = 64
        half_rot = rot_dim // 2
        BF16 = 2
        row_bytes = rot_dim * BF16
        half_bytes = half_rot * BF16

        SA_X_LO, SA_X_HI = 0x40000, 0x40080
        SA_OUT_LO, SA_OUT_HI = 0x40100, 0x40180
        SA_TMP_A = 0x40200
        SB_COS, SB_NEG_SIN, SB_SIN_HI, SB_TMP_B = 0x80000, 0x80080, 0x80100, 0x80180

        rope_reads = [(cos_pad_dram, SB_COS, half_rot),
                      (neg_sin_pad_dram, SB_NEG_SIN, half_rot),
                      (sin_hi_pad_dram, SB_SIN_HI, half_rot)]
        src_reads = [(src_dram, SA_X_LO, half_rot),
                     (src_dram + half_bytes, SA_X_HI, half_rot)]
        writes = [(SA_OUT_LO, out_dram, half_rot),
                  (SA_OUT_HI, out_dram + half_bytes, half_rot)]

        t_reg = self.gpr_seq_len      # flat per-row Q/K address
        off_reg = self.gpr_q_seq_len  # scratch for reg_mul_imm + add_imm address math
        patch_reg = self.gpr_scratch  # per-patch rope address
        S = M // self.VIS_HEADS
        self.generate_instruction_add_set(t_reg, 0)
        self.generate_instruction_add_set(patch_reg, 0)
        self.loop_start(loop_cnt=S)                              # OUTER: patches
        self.generate_instruction_reg_mul_imm(off_reg, patch_reg, ue_35bit_addr_shifter(half_bytes))
        for base, sram, elems in rope_reads:
            self.generate_instruction_add_imm(off_reg, ue_35bit_addr_shifter(base), self.TMP_REG)
            self.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=sram,
                                            element_size=elems, general_reg_src=self.TMP_REG)
        self.loop_start(loop_cnt=self.VIS_HEADS)                 # INNER: heads
        self.generate_instruction_reg_mul_imm(off_reg, t_reg, ue_35bit_addr_shifter(row_bytes))
        for base, sram, elems in src_reads:
            self.generate_instruction_add_imm(off_reg, ue_35bit_addr_shifter(base), self.TMP_REG)
            self.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=sram,
                                            element_size=elems, general_reg_src=self.TMP_REG)
        self.eltwise_mul_core(SA_X_LO, SB_COS, SB_TMP_B, half_rot)
        self.eltwise_mul_core(SA_X_HI, SB_NEG_SIN, SA_TMP_A, half_rot)
        self.eltwise_add_core(SA_TMP_A, SB_TMP_B, SA_OUT_LO, half_rot)
        self.eltwise_mul_core(SA_X_HI, SB_COS, SB_TMP_B, half_rot)
        self.eltwise_mul_core(SA_X_LO, SB_SIN_HI, SA_TMP_A, half_rot)
        self.eltwise_add_core(SA_TMP_A, SB_TMP_B, SA_OUT_HI, half_rot)
        for sram, base, elems in writes:
            self.generate_instruction_add_imm(off_reg, ue_35bit_addr_shifter(base), self.TMP_REG)
            self.sram_to_accelerator_memory(sram_address=sram, accelerator_dram_address=0,
                                            element_size=elems, general_reg_src=self.TMP_REG)
        self.generate_instruction_add_inc(t_reg)
        self.loop_end()                                          # end INNER
        self.generate_instruction_add_inc(patch_reg)
        self.loop_end()                                          # end OUTER

    def _emit_qkv_transpose_to_hm(self, src_dram: int, dst_dram: int,
                                  S: int, dst_aligned_S: int) -> None:
        """Transpose vision Q/K/V (S*NH, HD) interleaved → (NH, aligned_S, HD)
        head-major. Only the first S rows of each head are written."""
        NH, HD, BF16 = self.VIS_HEADS, self.VIS_HEAD_DIM, 2
        row_bytes = HD * BF16
        src_jump_bytes = NH * row_bytes
        dst_head_bytes = dst_aligned_S * row_bytes
        chunk_S = max(1, URAM_NEAR_FULL_ELEMENTS // HD)
        SA_BUF = 0x00000
        for h in range(NH):
            src_base = src_dram + h * row_bytes
            dst_base = dst_dram + h * dst_head_bytes
            s = 0
            while s < S:
                this_chunk = min(chunk_S, S - s)
                self.accelerator_memory_to_sram(
                    accelerator_dram_address=src_base + s * src_jump_bytes, sram_address=SA_BUF,
                    element_size=this_chunk * HD, stride_bytes_per_chunk=row_bytes,
                    stride_jump_bytes=src_jump_bytes)
                self.sram_to_accelerator_memory(
                    sram_address=SA_BUF, accelerator_dram_address=dst_base + s * row_bytes,
                    element_size=this_chunk * HD)
                s += this_chunk

    def _emit_attn_out_transpose_to_interleaved(self, src_dram: int, dst_dram: int,
                                                S: int, src_aligned_S: int) -> None:
        """Inverse transpose attn output (NH, aligned_S, HD) head-major →
        (S, NH*HD) interleaved. Reads only the first S rows of each head."""
        NH, HD, BF16 = self.VIS_HEADS, self.VIS_HEAD_DIM, 2
        row_bytes = HD * BF16
        dst_jump_bytes = NH * row_bytes
        src_head_bytes = src_aligned_S * row_bytes
        chunk_S = max(1, URAM_NEAR_FULL_ELEMENTS // HD)
        SA_BUF = 0x00000
        for h in range(NH):
            src_base = src_dram + h * src_head_bytes
            dst_base = dst_dram + h * row_bytes
            s = 0
            while s < S:
                this_chunk = min(chunk_S, S - s)
                self.accelerator_memory_to_sram(
                    accelerator_dram_address=src_base + s * row_bytes, sram_address=SA_BUF,
                    element_size=this_chunk * HD)
                self.sram_to_accelerator_memory(
                    sram_address=SA_BUF, accelerator_dram_address=dst_base + s * dst_jump_bytes,
                    element_size=this_chunk * HD, stride_bytes_per_chunk=row_bytes,
                    stride_jump_bytes=dst_jump_bytes)
                s += this_chunk

    def _rms_norm_dram_pbi(self, M: int, N: int, A_DRAM_ADDR: int,
                           OUTPUT_DRAM_ADDR: int, GAMMA_DRAM_ADDR: int) -> None:
        """rms_norm_core_dram via the runtime ISA loop (gpr_M_reg) to shrink the
        captured bin — captures one single-row body regardless of M. Bit-exact to
        the static unroll (VIS_RMS_LEGACY=1 forces the unroll)."""
        if os.environ.get("VIS_RMS_LEGACY") == "1":
            return self.rms_norm_core_dram(M=M, N=N, A_DRAM_ADDR=A_DRAM_ADDR,
                                    OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR, GAMMA_DRAM_ADDR=GAMMA_DRAM_ADDR)
        self.generate_instruction_add_set(self.gpr_seq_len, M)
        return self.rms_norm_core_dram(M=M, N=N, A_DRAM_ADDR=A_DRAM_ADDR,
                                OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR, GAMMA_DRAM_ADDR=GAMMA_DRAM_ADDR,
                                gpr_M_reg=self.gpr_seq_len)

    def _emit_clamp_dram_to_dram(self, src_dram: int, dst_dram: int, num_elements: int,
                                 clamp_min: float, clamp_max: float,
                                 *, identity_addr: int | None = None, use_pbi: bool = True) -> None:
        """FPGA DRAM→DRAM clamp via matmul-with-identity + fused clamp (HW routes
        CLAMP through LALU only during DOT_PRODUCT). A=src (M=N/64, K=64) × a bf16
        64×64 identity = passthrough, then the fused clamp applies."""
        assert num_elements % 64 == 0, f"num_elements ({num_elements}) must be a multiple of 64"
        if identity_addr is None:
            identity_addr = self.VIS_IDENTITY_64
        _mm_kw = {}
        if use_pbi and os.environ.get("CLAMP_NO_GPRM") != "1":
            _mm_kw["gpr_M_reg"] = self._prime_M(num_elements // 64)
        self.matmat_mul_core(
            M=num_elements // 64, K=64, N=64,
            A_DRAM_ADDR=src_dram, B_DRAM_ADDR=identity_addr, OUTPUT_DRAM_ADDR=dst_dram,
            is_B_quantized=False, clamp_enable=True, clamp_min=clamp_min, clamp_max=clamp_max, **_mm_kw)

    def _matmul_with_output_clamp(self, *, clamp_min: float, clamp_max: float, **mm_kwargs) -> None:
        """matmat_mul_core with fused output-clamp bounds."""
        return self.matmat_mul_core(clamp_enable=True, clamp_min=clamp_min,
                                    clamp_max=clamp_max, **mm_kwargs)

    # ------------------------------------------------------------------
    # Vision encoder layers (S3): one SigLIP layer = pre_norm + Q/K/V proj +
    # Q/K norm (part A) → V-norm + 2D RoPE + head-major transpose → per-head
    # attention (_vis_s7) → O proj + post-attn norm + residual + MLP (part C).
    # Every Gemma4ClippableLinear clips input (FPGA clamp into scratch) and
    # output (fused clamp). Captured one-shot into the vision encoder bin.
    # ------------------------------------------------------------------
    def set_vision_attention_bias(self, padding_positions: torch.Tensor) -> None:
        """Rebuild VIS_FLASH_BIAS so attention masks BOTH real padding patches
        (position_ids == -1) and the alignment padding at the end. Call after
        vision_tensor_init and before running attention."""
        S, aligned_S = self._vis_num_patches, self._vis_aligned_S
        pad = (padding_positions[0] if padding_positions.dim() == 2 else padding_positions).cpu().bool()
        assert pad.shape[0] == S, f"padding mask has {pad.shape[0]} entries, expected {S}"
        self._vis_padding_mask = pad
        col_mask = torch.zeros(aligned_S, dtype=torch.bool)
        col_mask[:S] = pad
        col_mask[S:] = True
        bias = torch.zeros(aligned_S, aligned_S, dtype=torch.bfloat16)
        bias[:, col_mask] = float("-inf")
        self.dma_to_accelerator_memory(self.VIS_FLASH_BIAS, bias)

    def compile_vision_layer(self, layer_idx: int) -> int:
        """Layer part A: pre_norm + Q/K/V projections (clipped) + Q/K norms."""
        S, H, HD, NH = self._vis_num_patches, self.VIS_H, self.VIS_HEAD_DIM, self.VIS_HEADS
        w = self._vis_weight_addrs[layer_idx]
        clips = self._vis_clip_ranges[layer_idx]
        INPUT_DRAM = self.VIS_IO_A if layer_idx % 2 == 0 else self.VIS_IO_B

        self._compile_and_run_single("pre_norm", lambda: self._rms_norm_dram_pbi(
            M=S, N=H, A_DRAM_ADDR=INPUT_DRAM,
            OUTPUT_DRAM_ADDR=self.VIS_NORM_OUT, GAMMA_DRAM_ADDR=w["input_layernorm"]))

        for proj, out_dram in (("q_proj", self.VIS_Q_DRAM), ("k_proj", self.VIS_K_DRAM),
                               ("v_proj", self.VIS_V_DRAM)):
            c = clips[proj]
            self._compile_and_run_single(f"clip_in_{proj}", lambda c=c: self._emit_clamp_dram_to_dram(
                src_dram=self.VIS_NORM_OUT, dst_dram=self.VIS_INPUT_CLIP_H_SCRATCH,
                num_elements=S * H, clamp_min=c["input"][0], clamp_max=c["input"][1]))
            self._compile_and_run_single(proj, lambda proj=proj, out_dram=out_dram, c=c:
                self._matmul_with_output_clamp(
                    M=S, K=H, N=NH * HD,
                    A_DRAM_ADDR=self.VIS_INPUT_CLIP_H_SCRATCH,
                    B_DRAM_ADDR=w[proj]["data"], OUTPUT_DRAM_ADDR=out_dram,
                    is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=w[proj]["scale"],
                    clamp_min=c["output"][0], clamp_max=c["output"][1],
                    gpr_M_reg=self._vis_mm_gpr(M=S)))

        for norm, src, dst in (("q_norm", self.VIS_Q_DRAM, self.VIS_Q_NORM),
                               ("k_norm", self.VIS_K_DRAM, self.VIS_K_NORM)):
            self._compile_and_run_single(norm, lambda norm=norm, src=src, dst=dst:
                self._rms_norm_dram_pbi(M=S * NH, N=HD, A_DRAM_ADDR=src,
                                        OUTPUT_DRAM_ADDR=dst, GAMMA_DRAM_ADDR=w[norm]))
        return 0

    def host_vision_v_norm_rope_gather(self, layer_idx: int) -> None:
        """V-norm (FPGA) + 2D RoPE on Q/K (FPGA) + per-head transpose of Q/K/V
        into head-major VIS_FLASH_*_HM (FPGA). No Q pre-scale (attention passes
        q_scale=1.0 — Gemma vision folds the scaling into q_norm)."""
        S, HD, NH = self._vis_num_patches, self.VIS_HEAD_DIM, self.VIS_HEADS
        aligned_S = self._vis_aligned_S

        self._compile_and_run_single("v_norm", lambda: self._rms_norm_dram_pbi(
            M=S * NH, N=HD, A_DRAM_ADDR=self.VIS_V_DRAM,
            OUTPUT_DRAM_ADDR=self.VIS_V_DRAM, GAMMA_DRAM_ADDR=self.VIS_V_NORM_ONES_GAMMA))

        for label, dram in (("rope_q", self.VIS_Q_NORM), ("rope_k", self.VIS_K_NORM)):
            self._compile_and_run_single(label, lambda dram=dram: self._emit_vision_rope_2d(
                src_dram=dram, out_dram=dram,
                cos_pad_dram=self.VIS_ROPE_COS_PAD_TILED,
                neg_sin_pad_dram=self.VIS_ROPE_NEG_SIN_PAD_TILED,
                sin_hi_pad_dram=self.VIS_ROPE_SIN_HI_PAD_TILED, M=S * NH))

        for label, src, dst in (("transpose_q", self.VIS_Q_NORM, self.VIS_FLASH_Q_HM),
                                ("transpose_k", self.VIS_K_NORM, self.VIS_FLASH_K_HM),
                                ("transpose_v", self.VIS_V_DRAM, self.VIS_FLASH_V_HM)):
            self._compile_and_run_single(label, lambda src=src, dst=dst:
                self._emit_qkv_transpose_to_hm(src_dram=src, dst_dram=dst,
                                               S=S, dst_aligned_S=aligned_S))

    def _vis_emit_attn(self, Q: int, K: int, V: int, O: int, HD: int, aligned_S: int) -> int:
        """Emit one head's vision attention via unified_attention_core. batch =
        aligned_seq_len = aligned_S (every patch is a query), full-matrix bias
        (VIS_FLASH_BIAS), Vᵀ/score/scaled_q folded into VIS_FLASH_SCRATCH.
        q_scale=1.0 → no QKᵀ scaling (folded into q_norm)."""
        _batch_reg = self.alloc_isa_reg()
        self.generate_instruction_add_set(_batch_reg, aligned_S)
        _aligned_reg = self.alloc_isa_reg()
        self.generate_instruction_add_set(_aligned_reg, aligned_S)
        total = self.unified_attention_core(
            batch=aligned_S, aligned_seq_len=aligned_S, head_dim=HD,
            Q_DRAM_ADDR=Q, K_DRAM_ADDR=K, V_DRAM_ADDR=V,
            BIAS_DRAM_ADDR=self.VIS_FLASH_BIAS, OUTPUT_DRAM_ADDR=O,
            SCRATCH_DRAM_ADDR=self.VIS_FLASH_SCRATCH,
            IDENTITY_DRAM_ADDR=self._vis_identity_dram,
            gpr_batch_reg=_batch_reg, gpr_aligned_seq_len_reg=_aligned_reg, q_scale=1.0)
        self.release_isa_reg()  # _aligned_reg
        self.release_isa_reg()  # _batch_reg
        return total

    def run_vision_attention_all_heads(self, layer_idx: int) -> None:
        """Per-head attention (_vis_s7 path): marshal each head's head-major
        Q/K/V slice into the FIXED VIS_FLASH_Q/K/V via an SRAM bounce, run
        _vis_emit_attn, bounce OUT back to head-major OUT_HM; then inverse-
        transpose OUT_HM → interleaved VIS_Q_DRAM for o_proj."""
        bpe = self.bytes_per_element
        S, HD, NH = self._vis_num_patches, self.VIS_HEAD_DIM, self.VIS_HEADS
        aligned_S = self._vis_aligned_S
        head_stride_bytes = aligned_S * HD * bpe
        _elems = aligned_S * HD
        _l0h0 = (layer_idx == 0 and os.environ.get("GEMMA4_VIS_L0_CKPT") == "1")

        def _bounce(_src, _dst):
            for _o in range(0, _elems, 131072):
                _n = min(131072, _elems - _o)
                self.accelerator_memory_to_sram(
                    accelerator_dram_address=_src + _o * bpe, sram_address=0x10000, element_size=_n)
                self.sram_to_accelerator_memory(
                    sram_address=0x10000, accelerator_dram_address=_dst + _o * bpe, element_size=_n)

        for h in range(NH):
            base = h * head_stride_bytes
            _bounce(self.VIS_FLASH_Q_HM + base, self.VIS_FLASH_Q)
            _bounce(self.VIS_FLASH_K_HM + base, self.VIS_FLASH_K)
            _bounce(self.VIS_FLASH_V_HM + base, self.VIS_FLASH_V)
            if _l0h0 and h == 0:
                self._emit_sram_copy_chunked(self.VIS_FLASH_Q, self.VIS_L0H0_Q, _elems)
                self._emit_sram_copy_chunked(self.VIS_FLASH_K, self.VIS_L0H0_K, _elems)
                self._emit_sram_copy_chunked(self.VIS_FLASH_V, self.VIS_L0H0_V, _elems)
            self._vis_flops_add(self._vis_emit_attn(
                self.VIS_FLASH_Q, self.VIS_FLASH_K,
                self.VIS_FLASH_V, self.VIS_FLASH_OUT, HD, aligned_S))
            if _l0h0 and h == 0:
                self._emit_sram_copy_chunked(self.VIS_FLASH_OUT, self.VIS_L0H0_OUT, _elems)
            _bounce(self.VIS_FLASH_OUT, self.VIS_FLASH_OUT_HM + base)

        self._compile_and_run_single("transpose_attn_out", lambda:
            self._emit_attn_out_transpose_to_interleaved(
                src_dram=self.VIS_FLASH_OUT_HM, dst_dram=self.VIS_Q_DRAM,
                S=S, src_aligned_S=aligned_S))

    def compile_vision_layer_post_attn(self, layer_idx: int) -> int:
        """Layer part C: O proj (clipped) + post_attn norm + residual + pre-FFN
        norm + MLP (gate/GELU + up, clipped; gate*up; down, clipped) + post-FFN
        norm + residual → OUTPUT buffer."""
        S, H, HD, NH, MLP = (self._vis_num_patches, self.VIS_H, self.VIS_HEAD_DIM,
                             self.VIS_HEADS, self.VIS_MLP)
        w = self._vis_weight_addrs[layer_idx]
        clips = self._vis_clip_ranges[layer_idx]
        INPUT_DRAM = self.VIS_IO_A if layer_idx % 2 == 0 else self.VIS_IO_B
        OUTPUT_DRAM = self.VIS_IO_B if layer_idx % 2 == 0 else self.VIS_IO_A
        sz_h = S * H

        co = clips["o_proj"]
        self._compile_and_run_single("clip_in_o", lambda: self._emit_clamp_dram_to_dram(
            src_dram=self.VIS_Q_DRAM, dst_dram=self.VIS_INPUT_CLIP_H_SCRATCH,
            num_elements=S * NH * HD, clamp_min=co["input"][0], clamp_max=co["input"][1]))
        self._compile_and_run_single("o_proj", lambda: self._matmul_with_output_clamp(
            M=S, K=NH * HD, N=H,
            A_DRAM_ADDR=self.VIS_INPUT_CLIP_H_SCRATCH,
            B_DRAM_ADDR=w["o_proj"]["data"], OUTPUT_DRAM_ADDR=self.VIS_ATTN_OUT,
            is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=w["o_proj"]["scale"],
            clamp_min=co["output"][0], clamp_max=co["output"][1], gpr_M_reg=self._vis_mm_gpr(M=S)))

        self._compile_and_run_single("post_attn_norm", lambda: self._rms_norm_dram_pbi(
            M=S, N=H, A_DRAM_ADDR=self.VIS_ATTN_OUT,
            OUTPUT_DRAM_ADDR=self.VIS_POST_ATTN_NORM, GAMMA_DRAM_ADDR=w["post_attention_layernorm"]))
        self._run_eltwise_add_chunked(INPUT_DRAM, self.VIS_POST_ATTN_NORM, self.VIS_POST_ATTN_RES, sz_h)
        self._compile_and_run_single("pre_ffn_norm", lambda: self._rms_norm_dram_pbi(
            M=S, N=H, A_DRAM_ADDR=self.VIS_POST_ATTN_RES,
            OUTPUT_DRAM_ADDR=self.VIS_PRE_FFN_NORM, GAMMA_DRAM_ADDR=w["pre_feedforward_layernorm"]))

        cg = clips["gate_proj"]
        self._compile_and_run_single("clip_in_gate", lambda: self._emit_clamp_dram_to_dram(
            src_dram=self.VIS_PRE_FFN_NORM, dst_dram=self.VIS_INPUT_CLIP_H_SCRATCH,
            num_elements=S * H, clamp_min=cg["input"][0], clamp_max=cg["input"][1]))
        self._compile_and_run_single("gate_proj", lambda: self.matmat_mul_core(
            M=S, K=H, N=MLP, A_DRAM_ADDR=self.VIS_INPUT_CLIP_H_SCRATCH,
            B_DRAM_ADDR=w["gate_proj"]["data"], OUTPUT_DRAM_ADDR=self.VIS_MLP_GATE,
            is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=w["gate_proj"]["scale"],
            gelu_enable=True, gpr_M_reg=self._vis_mm_gpr(M=S)))
        self._compile_and_run_single("clip_out_gate", lambda: self._emit_clamp_dram_to_dram(
            src_dram=self.VIS_MLP_GATE, dst_dram=self.VIS_MLP_GATE,
            num_elements=S * MLP, clamp_min=cg["output"][0], clamp_max=cg["output"][1]))

        cu = clips["up_proj"]
        self._compile_and_run_single("clip_in_up", lambda: self._emit_clamp_dram_to_dram(
            src_dram=self.VIS_PRE_FFN_NORM, dst_dram=self.VIS_INPUT_CLIP_H_SCRATCH,
            num_elements=S * H, clamp_min=cu["input"][0], clamp_max=cu["input"][1]))
        self._compile_and_run_single("up_proj", lambda: self._matmul_with_output_clamp(
            M=S, K=H, N=MLP, A_DRAM_ADDR=self.VIS_INPUT_CLIP_H_SCRATCH,
            B_DRAM_ADDR=w["up_proj"]["data"], OUTPUT_DRAM_ADDR=self.VIS_MLP_UP,
            is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=w["up_proj"]["scale"],
            clamp_min=cu["output"][0], clamp_max=cu["output"][1], gpr_M_reg=self._vis_mm_gpr(M=S)))

        self._run_eltwise_mul_chunked(self.VIS_MLP_GATE, self.VIS_MLP_UP, self.VIS_MLP_MULT, S * MLP)

        cd = clips["down_proj"]
        self._compile_and_run_single("clip_in_down", lambda: self._emit_clamp_dram_to_dram(
            src_dram=self.VIS_MLP_MULT, dst_dram=self.VIS_INPUT_CLIP_MLP_SCRATCH,
            num_elements=S * MLP, clamp_min=cd["input"][0], clamp_max=cd["input"][1]))
        self._compile_and_run_single("down_proj", lambda: self._matmul_with_output_clamp(
            M=S, K=MLP, N=H, A_DRAM_ADDR=self.VIS_INPUT_CLIP_MLP_SCRATCH,
            B_DRAM_ADDR=w["down_proj"]["data"], OUTPUT_DRAM_ADDR=self.VIS_MLP_DOWN,
            is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=w["down_proj"]["scale"],
            clamp_min=cd["output"][0], clamp_max=cd["output"][1], gpr_M_reg=self._vis_mm_gpr(M=S)))

        self._compile_and_run_single("post_ffn_norm", lambda: self._rms_norm_dram_pbi(
            M=S, N=H, A_DRAM_ADDR=self.VIS_MLP_DOWN,
            OUTPUT_DRAM_ADDR=self.VIS_POST_FFN_NORM, GAMMA_DRAM_ADDR=w["post_feedforward_layernorm"]))
        self._run_eltwise_add_chunked(self.VIS_POST_ATTN_RES, self.VIS_POST_FFN_NORM, OUTPUT_DRAM, sz_h)
        return 0

    def run_vision_layer(self, layer_idx: int) -> int:
        """Run one full vision encoder layer: part A → V-norm/RoPE/transpose →
        per-head attention → part C. Input in VIS_IO_A (even layer) / VIS_IO_B
        (odd); output lands in the other buffer (address returned)."""
        self.compile_vision_layer(layer_idx)
        self.host_vision_v_norm_rope_gather(layer_idx)
        self.run_vision_attention_all_heads(layer_idx)
        self.compile_vision_layer_post_attn(layer_idx)
        return self.VIS_IO_B if layer_idx % 2 == 0 else self.VIS_IO_A

    def compile_vision_encoder_bin(self, num_patches: int, profile: bool = False) -> None:
        """Capture the 16-layer vision encoder ISA one-shot into a bin at
        VISION_ISA_BASE (0xa0000000). Pure host emission, no FPGA activity.
        Skips if cached. Optional layer-0 checkpoint snapshots (GEMMA4_VIS_L0_CKPT=1)
        for the numeric harness.

        ``profile``: emit a HALT at each major per-layer boundary (proj /
        rope_gather / attention / post_attn) and record the resume address, so
        execute_vision_encoder_bin can time each phase's FPGA latency (like the
        LM per-phase profile). The checkpoints go in the meta; a profile-compiled
        bin can only be run segment-by-segment."""
        L = self.VIS_LAYERS
        _vmeta, _ = self._get_program_section("vision", profile)
        if _vmeta is not None and _vmeta.get("num_patches") == num_patches:
            print(f"  [Vision] encoder section cached (num_patches={num_patches}).", flush=True)
            return
        print(f"  [Vision] compiling {L} vision layers one-shot"
              f"{' (+profile checkpoints)' if profile else ''} ...", flush=True)
        t0 = time.perf_counter()
        _l0_ckpt = os.environ.get("GEMMA4_VIS_L0_CKPT") == "1"
        enc_checkpoints: list[list] = []
        def _checkpoint(name: str) -> None:
            if not profile:
                return
            self.generate_instruction_halt()
            resume = self.get_program_dram_addr() + self.capture_count * INSTRUCTION_SIZE_BYTES
            enc_checkpoints.append([name, f"0x{resume:X}"])
        _l0_elems = self._vis_num_patches * self.VIS_H
        global _SILENT_MODE
        _prev_silent = _SILENT_MODE
        self.reset_program_dram_addr()
        base_addr = self.get_program_dram_addr()
        self.clear_inst_id()
        self.clear_capture_buffer()
        self.start_capture()
        self._oneshot_mode = True
        self._vis_s7 = True
        self._vis_flops = 0    # accumulated via _vis_flops_add during emission
        _SILENT_MODE = True
        try:
            for li in range(L):
                self.compile_vision_layer(li)
                _checkpoint(f"L{li}_proj")
                if li == 0 and _l0_ckpt:
                    self._emit_sram_copy_chunked(self.VIS_NORM_OUT, self.VIS_L0_BEFORE_ATTN, _l0_elems)
                    self._emit_sram_copy_chunked(self.VIS_Q_DRAM, self.VIS_L0_QPROJ, _l0_elems)
                    self._emit_sram_copy_chunked(self.VIS_K_DRAM, self.VIS_L0_KPROJ, _l0_elems)
                    self._emit_sram_copy_chunked(self.VIS_V_DRAM, self.VIS_L0_VPROJ, _l0_elems)
                    self._emit_sram_copy_chunked(self.VIS_Q_NORM, self.VIS_L0_QPRE, _l0_elems)
                    self._emit_sram_copy_chunked(self.VIS_K_NORM, self.VIS_L0_KPRE, _l0_elems)
                self.host_vision_v_norm_rope_gather(li)
                _checkpoint(f"L{li}_rope_gather")
                if li == 0 and _l0_ckpt:
                    _hm = self.VIS_HEADS * self._vis_aligned_S * self.VIS_HEAD_DIM
                    self._emit_sram_copy_chunked(self.VIS_V_DRAM, self.VIS_L0_VNORM, _l0_elems)
                    self._emit_sram_copy_chunked(self.VIS_Q_NORM, self.VIS_L0_ROPE_Q, _l0_elems)
                    self._emit_sram_copy_chunked(self.VIS_K_NORM, self.VIS_L0_ROPE_K, _l0_elems)
                    self._emit_sram_copy_chunked(self.VIS_FLASH_Q_HM, self.VIS_L0_Q_HM, _hm)
                    self._emit_sram_copy_chunked(self.VIS_FLASH_K_HM, self.VIS_L0_K_HM, _hm)
                    self._emit_sram_copy_chunked(self.VIS_FLASH_V_HM, self.VIS_L0_V_HM, _hm)
                self.run_vision_attention_all_heads(li)
                _checkpoint(f"L{li}_attention")
                if li == 0 and _l0_ckpt:
                    _hm = self.VIS_HEADS * self._vis_aligned_S * self.VIS_HEAD_DIM
                    self._emit_sram_copy_chunked(self.VIS_FLASH_OUT_HM, self.VIS_L0_ATTN_OUT_HM, _hm)
                    self._emit_sram_copy_chunked(self.VIS_Q_DRAM, self.VIS_L0_ATTN_CORE, _l0_elems)
                self.compile_vision_layer_post_attn(li)
                _checkpoint(f"L{li}_post_attn")
                if li == 0 and _l0_ckpt:
                    self._emit_sram_copy_chunked(self.VIS_ATTN_OUT, self.VIS_L0_AFTER_ATTN, _l0_elems)
                    self._emit_sram_copy_chunked(self.VIS_IO_B, self.VIS_L0_END, _l0_elems)
            self.generate_instruction_halt()
        finally:
            self._oneshot_mode = False
            self._vis_s7 = False
            _SILENT_MODE = _prev_silent
        self.stop_capture()
        enc_bytes = bytearray()
        for inst in self.capture_buffer:
            enc_bytes.extend(inst.get_bytes())
        self.clear_capture_buffer()
        total_flops = int(self._vis_flops)
        self._vis_flops = None    # stop accumulating outside the encoder capture

        _vis_meta = {"num_patches": num_patches, "vis_layers": L, "total_flops": total_flops}
        if profile:
            _vis_meta["profile_checkpoints"] = enc_checkpoints
        # Merge the vision section into the combined programs bin (LM stays intact).
        self._store_program_section("vision", base_addr, enc_bytes, _vis_meta, profile=profile)
        print(f"  [Vision] encoder section stored ({len(enc_bytes)/1024/1024:.1f} MB @ 0x{base_addr:X}, "
              f"{time.perf_counter()-t0:.1f}s)", flush=True)

    def execute_vision_encoder_bin(self, num_patches: int, profile: bool = False):
        """Load the cached encoder bin, DMA it to program DRAM at its baked base,
        and execute. Returns elapsed seconds.

        ``profile``: run the profile-compiled bin **segment-by-segment** through
        its per-phase HALT checkpoints, timing each with the HW latency counter,
        and return the per-segment [(name, ms)] list instead. The encoder still
        computes its full output (segments tile the whole program). No preamble
        is needed — the vision ops prime their own GPRs inline."""
        import threading
        meta, enc_bytes = self._get_program_section("vision", profile)
        if meta is None:
            raise FileNotFoundError("vision section not found in combined programs bin")
        program_addr = int(meta["dram_base"], 16)
        self._next_program_dram_addr = program_addr
        self.dma_write(DMA_DEVICE_H2C, program_addr, enc_bytes, len(enc_bytes))
        self.allocate_program_dram(len(enc_bytes))
        print(f"  [Vision] launching encoder ({len(enc_bytes)/1024/1024:.1f} MB) at 0x{program_addr:X}"
              f"{' [profiled]' if profile else ''} ...", flush=True)
        t0 = time.perf_counter()

        if profile:
            checkpoints = meta.get("profile_checkpoints", [])
            results = []
            self.start_execute_from_dram(program_addr)
            for name, resume_hex in checkpoints:
                self.wait_queue(180.0)
                results.append((name, self.report_latency_in_us() / 1e3))   # ms
                self.start_execute_from_dram(int(resume_hex, 16))
            self.wait_queue(180.0)   # tail: final HALT after the last checkpoint
            results.append(("tail", self.report_latency_in_us() / 1e3))
            return results

        _anchor = getattr(self, "_vis_fpga_t0", t0)
        _stop = threading.Event()
        def _hb():
            while not _stop.wait(10):
                _original_print(f"  [Vision] ... running on FPGA ({time.perf_counter()-_anchor:.0f}s)", flush=True)
        _th = threading.Thread(target=_hb, daemon=True); _th.start()
        try:
            self.start_execute_from_dram(program_addr)
            self.wait_queue(180.0)
        finally:
            _stop.set(); _th.join(timeout=1.0)
        elapsed = time.perf_counter() - t0
        # Report like the LM's program_execute: HW latency + FLOP rate.
        latency_us = self.report_latency_in_us()
        print(f"    Total program execution latency = {latency_us} us")
        _flops = meta.get("total_flops")
        if _flops:
            gflops, _ = self.report_flop_rate_gflops(_flops)
            print(f"Report FLOPS for program execution: {gflops:.2f} GFLOPS")
        print(f"Vision encoder execute done in {elapsed:.2f} seconds.", flush=True)
        return elapsed

    # ------------------------------------------------------------------
    # Vision projection + driver (S4): pooler tail, full encoder run, and the
    # set_prefill_seq_vlm entry that feeds image soft-tokens into run_prefill's
    # existing merge hooks.
    # ------------------------------------------------------------------
    def vision_embed_project(self, hidden_states: torch.Tensor,
                             pixel_position_ids: torch.Tensor,
                             padding_positions: torch.Tensor) -> torch.Tensor:
        """Gemma4 vision pooler + embed_vision tail. Host: mask padding, spatial
        avg-pool by position, scale by sqrt(H), strip masked rows. FPGA: RMSNorm
        (gamma=ones) + embedding_projection (IF4). Returns image_features
        [N_final, VIS_TEXT_H]."""
        H, text_h, pool_k = self.VIS_H, self.VIS_TEXT_H, self.VIS_POOL_K
        hidden_states = hidden_states.detach().cpu()
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        S = hidden_states.shape[1]
        output_length = S // (pool_k * pool_k)
        pids = pixel_position_ids.detach().cpu()
        if pids.dim() == 2:
            pids = pids.unsqueeze(0)
        pad = padding_positions.detach().cpu()
        if pad.dim() == 1:
            pad = pad.unsqueeze(0)

        with torch.no_grad():
            h = hidden_states.float().clone()
            h.masked_fill_(pad.unsqueeze(-1), 0.0)
            input_seq_len = h.shape[1]
            k = int((input_seq_len // output_length) ** 0.5)
            k_squared = k * k
            if k_squared * output_length != input_seq_len:
                raise ValueError(
                    f"Cannot pool {h.shape} to {output_length}: k={k}^2 × {output_length} "
                    f"must equal {input_seq_len}.")
            clamped = pids.clamp(min=0)
            max_x = clamped[..., 0].max(dim=-1, keepdim=True)[0] + 1
            kernel_idxs = torch.div(clamped, k, rounding_mode="floor")
            kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
            weights = F.one_hot(kernel_idxs.long(), output_length).float() / k_squared
            pooled = weights.transpose(1, 2) @ h
            pooler_mask = torch.logical_not((weights == 0).all(dim=1))
            pooled = (pooled * (H ** 0.5))[pooler_mask]
        N_final = int(pooled.shape[0])
        assert N_final <= S, f"pooler produced {N_final} rows, scratch sized for {S}"
        self.dma_to_accelerator_memory(self.VIS_EMBED_POOL, pooled.to(torch.bfloat16).contiguous())

        self._compile_and_run_single("embed_pre_norm", lambda: self.rms_norm_core_dram(
            M=N_final, N=H, A_DRAM_ADDR=self.VIS_EMBED_POOL,
            OUTPUT_DRAM_ADDR=self.VIS_EMBED_NORMED, GAMMA_DRAM_ADDR=self.VIS_EMBED_NORM_GAMMA))
        w = self.VIS_EMBED_PROJ_INFO
        self._compile_and_run_single("embed_projection", lambda: self.matmat_mul_core(
            M=N_final, K=H, N=text_h, A_DRAM_ADDR=self.VIS_EMBED_NORMED,
            B_DRAM_ADDR=w["data"], OUTPUT_DRAM_ADDR=self.VIS_EMBED_OUT,
            is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=w["scale"],
            gpr_M_reg=self._vis_mm_gpr(M=N_final)))
        return self.dma_from_accelerator_memory(self.VIS_EMBED_OUT, (N_final, text_h)).cpu()

    def _run_vision_encoder_host(self, image_path: str, prompt: str):
        """Run the HF vision tower on the host and return the same image soft
        tokens and prompt metadata consumed by the FPGA LM prefill path."""
        from PIL import Image
        from transformers import AutoProcessor

        host_t0 = time.perf_counter()
        hf_model, model_dir = _ensure_hf_model(self.script_dir, self._cfg)
        processor = AutoProcessor.from_pretrained(model_dir)
        hf_model.eval()

        image = Image.open(image_path).convert("RGB").resize(
            VISION_CANONICAL_SIZE, Image.BICUBIC)
        conversation = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text_prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text_prompt], images=[[image]], return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        pixel_position_ids = inputs["image_position_ids"]
        token_ids = inputs["input_ids"][0].tolist()
        mm_types = inputs["mm_token_type_ids"][0].tolist()

        forward_t0 = time.perf_counter()
        with torch.no_grad():
            out = hf_model.model.get_image_features(
                pixel_values=pixel_values.to(torch.bfloat16),
                image_position_ids=pixel_position_ids)
        forward_s = time.perf_counter() - forward_t0
        image_features = getattr(out, "pooler_output", out).detach().cpu()
        if image_features.dim() == 3 and image_features.shape[0] == 1:
            image_features = image_features.squeeze(0)

        n_image_tokens = sum(t == 1 for t in mm_types)
        if image_features.shape[0] != n_image_tokens:
            raise RuntimeError(
                f"Host vision produced {image_features.shape[0]} embeddings, "
                f"but the prompt contains {n_image_tokens} image-token positions.")
        total_s = time.perf_counter() - host_t0
        print(f"[Vision] host encoder done: {image_features.shape[0]} soft tokens "
              f"in {total_s:.2f}s (model forward {forward_s:.2f}s).", flush=True)
        return image_features, token_ids, mm_types

    def _run_vision_encoder_fpga(self, image_path: str, prompt: str, profile: bool = False):
        """Full vision encoder on FPGA (separate bin @ 0xa0000000): preprocess →
        patch embed → 16-layer encoder → pooler + projection. Returns
        (image_features, token_ids, mm_types). Stashes FPGA readbacks in
        self._vis_ckpt for the numeric harness. Restores the LM allocator state
        on exit so LM prefill/decode still work.

        ``profile``: run the encoder through per-phase HALT checkpoints and print
        a major-step FPGA-latency breakdown (proj / rope_gather / attention /
        post_attn, aggregated across layers), same style as the LM profile."""
        from PIL import Image
        from transformers import AutoProcessor

        fpga_total_t0 = time.perf_counter()
        hf_model, model_dir = _ensure_hf_model(self.script_dir, self._cfg)
        processor = AutoProcessor.from_pretrained(model_dir)
        hf_model.eval()

        image = Image.open(image_path).convert("RGB").resize(VISION_CANONICAL_SIZE, Image.BICUBIC)
        conversation = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text_prompt], images=[[image]], return_tensors="pt")
        pixel_values = inputs['pixel_values']                       # [1, S, 768]
        pixel_position_ids = inputs['image_position_ids']            # [1, S, 2]
        padding_positions = (pixel_position_ids == -1).all(dim=-1)   # [1, S]
        num_patches = pixel_values.shape[1]
        token_ids = inputs['input_ids'][0].tolist()
        mm_types = inputs['mm_token_type_ids'][0].tolist()
        assert num_patches == VISION_FIXED_NUM_PATCHES, (
            f"Vision shape contract broken: num_patches={num_patches}, expected "
            f"{VISION_FIXED_NUM_PATCHES}. Check the {VISION_CANONICAL_SIZE} → patches mapping.")

        # Save LM allocator state to restore after vision clobbers the shared
        # tensor DRAM region.
        _tensor_save = self._tensor_dram_addr
        _prog_addr_save = self._next_program_dram_addr
        _prog_base_save = self._program_dram_base

        global _SILENT_MODE
        _prev_silent = _SILENT_MODE
        _SILENT_MODE = True
        try:
            self.vision_weight_init(hf_model)        # weights ABOVE LM tensors (no reset)
            self.vision_tensor_init(num_patches)     # scratch reuses LM tensor region; ISA base 0xa0
            self.set_vision_attention_bias(padding_positions)
            cos_p, neg_sin_p, sin_hi_p = self._load_or_build_vision_rope_pads(
                hf_model, pixel_position_ids, num_patches)
            self.dma_to_accelerator_memory(self.VIS_ROPE_COS_PAD_TILED, cos_p)
            self.dma_to_accelerator_memory(self.VIS_ROPE_NEG_SIN_PAD_TILED, neg_sin_p)
            self.dma_to_accelerator_memory(self.VIS_ROPE_SIN_HI_PAD_TILED, sin_hi_p)
        finally:
            _SILENT_MODE = _prev_silent

        self._vis_fpga_t0 = time.perf_counter()
        patch_embeds = self.vision_patch_embed(
            pixel_values.cpu(), pixel_position_ids.cpu(), padding_positions.cpu())
        self.compile_vision_encoder_bin(num_patches, profile=profile)
        enc_result = self.execute_vision_encoder_bin(num_patches, profile=profile)

        # 16 layers (even) → final output in VIS_IO_A (patch embed → IO_A → IO_B
        # → ... → IO_A). Odd layer counts would land in VIS_IO_B.
        final_buf = self.VIS_IO_A if self.VIS_LAYERS % 2 == 0 else self.VIS_IO_B
        encoder_out = self.dma_from_accelerator_memory(final_buf, (num_patches, self.VIS_H)).cpu()
        image_features = self.vision_embed_project(
            encoder_out, pixel_position_ids.cpu(), padding_positions.cpu())

        # Numeric-harness checkpoints (SNR-compared vs HF in the numeric script).
        self._vis_ckpt = {
            "patch_embed": patch_embeds,
            "encoder_out": encoder_out,
            "image_features": image_features,
        }

        # Restore LM allocator state (vision scratch overwrote LM tensor DATA;
        # run_prefill re-uploads/zeros it, so only the cursors need restoring).
        self._tensor_dram_addr = _tensor_save
        self._next_program_dram_addr = _prog_addr_save
        self._program_dram_base = _prog_base_save

        # Major-step FPGA-latency breakdown of the encoder (profile mode only).
        if profile and isinstance(enc_result, list):
            self._print_phase_breakdown("VISION ENCODER", enc_result, per_token=False)
        fpga_total_s = time.perf_counter() - fpga_total_t0
        print(f"[Vision] FPGA encoder done: {image_features.shape[0]} soft tokens "
              f"in {fpga_total_s:.2f}s total.", flush=True)
        return image_features, token_ids, mm_types

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
        def _projection_core(**kwargs) -> int:
            # The streaming core cannot hold a 64-column strip's scales when
            # K=12,288. Gemma4's wide MLP-down projection therefore retains the
            # two-pass compatibility core even in streaming mode.
            if (self.prefill_kernel == "matmatmul"
                    or kwargs["K"] == self.mlp_elements_wide):
                return self.matmat_mul_core(**kwargs)
            kwargs.pop("is_B_quantized", None)
            return self.quantized_matmat_core(**kwargs)
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
            total_flops += _projection_core(M=seq_len, K=self.vector_length, N=cur_q_size,
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
                total_flops += _projection_core(M=seq_len, K=self.vector_length, N=cur_k_size,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off,
                    gpr_M_reg=self.gpr_seq_len,
                    )
                # V projection: write to temp buffer first, then scatter to KV cache at k_size stride
                total_flops += _projection_core(M=seq_len, K=self.vector_length, N=cur_k_size,
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
                kv_row_bytes = self._kv_row_bytes_for_layer[layer_idx]
                v_cache_base = self.LAYER0_V_DRAM + self._kv_offset_for_layer[layer_idx]
                _vi = self.alloc_isa_reg()
                self.generate_instruction_add_set(_vi, 0)
                self.loop_start(loop_cnt=seq_len, gpr_loop_cnt=self.gpr_seq_len)
                self.generate_instruction_reg_mul_imm(self.TMP_REG, _vi, ue_35bit_addr_shifter(cur_k_size * self.bytes_per_element))
                self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(self.LAYER0_FLASH_V_DRAM), self.TMP_REG)
                self.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=0x10000, element_size=cur_k_size, general_reg_src=self.TMP_REG)
                self.rms_norm_core(0x10000, 0x10000, cur_k_size)  # no gamma
                self.generate_instruction_reg_mul_imm(self.TMP_REG, _vi, ue_35bit_addr_shifter(kv_row_bytes))
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
            kv_slot_off = self._kv_offset_for_layer[layer_idx]
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
                self._emit_strided_copy_pbi(K_TMP, k_cache_base, cur_head_dim, head_bytes, head_bytes, seq_len, self.gpr_seq_len)
            _checkpoint(f"L{layer_idx}_rope")
            # GQA dup: read KV cache (k_size stride) → scatter group_size copies to FLASH (cur_head_dim stride).
            self._emit_gqa_duplicate_pbi(k_cache_base, self.LAYER0_FLASH_K_DRAM, cur_head_dim, seq_len, self.gpr_seq_len, src_row_bytes=head_bytes)
            self._emit_gqa_duplicate_pbi(v_cache_base, self.LAYER0_FLASH_V_DRAM, cur_head_dim, seq_len, self.gpr_seq_len, src_row_bytes=head_bytes)
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
            total_flops += _projection_core(M=seq_len, K=cur_q_size, N=self.vector_length,
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
            total_flops += _projection_core(M=seq_len, K=self.vector_length, N=cur_mlp,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                is_B_quantized=True,
                data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off,
                gelu_enable=True,
                gpr_M_reg=self.gpr_seq_len,
                )
            total_flops += _projection_core(M=seq_len, K=self.vector_length, N=cur_mlp,
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
            total_flops += _projection_core(M=seq_len, K=cur_mlp, N=self.vector_length,
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
        kv_cache_bytes = getattr(
            self, "_kv_cache_bytes",
            num_slots * self.MAX_CONTEXT_SIZE * self.head_dim * self.bytes_per_element)
        kv_zero_pad = torch.zeros(
            kv_cache_bytes // self.bytes_per_element, dtype=torch.bfloat16)
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

        # Time host-side prep (embedding lookup, per-layer inputs, bias build,
        # DMAs) — reported only in the segmented-profile path's status line.
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
        def _projection_core(**kwargs) -> int:
            if self.decode_kernel == "matmatmul":
                kwargs.setdefault("is_B_quantized", True)
                return self.matmat_mul_core(**kwargs)
            if kwargs["K"] == self.mlp_elements_wide:
                # K=12,288 wide MLP-down uses the legacy streaming core's
                # verified N=32 partial-width scale-BRAM tiling. Drop the
                # dimension GPR so quantized_matmat_core dispatches legacy;
                # the dynamic core still supports only K <= 8,192.
                kwargs.pop("gpr_M_reg", None)
                kwargs.pop("gpr_K_reg", None)
                kwargs.pop("gpr_N_reg", None)
            kwargs.pop("is_B_quantized", None)
            return self.quantized_matmat_core(**kwargs)
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
                total_flops += _projection_core(M=1, K=self.vector_length, N=cur_q_size,
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
                    total_flops += _projection_core(M=1, K=self.vector_length, N=cur_k_size,
                        A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                        B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off,
                        OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                        data_type=TYPE.IF4,
                        SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off,
                        )
                    # V projection
                    total_flops += _projection_core(M=1, K=self.vector_length, N=cur_k_size,
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
                    # Compact cache row stride equals this layer's head dimension.
                    _kv_row_bytes = self._kv_row_bytes_for_layer[layer_idx]
                    _v_slot_base = self.LAYER0_V_DRAM + self._kv_offset_for_layer[layer_idx]
                    self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gpr_seq_len, ue_35bit_addr_shifter(_kv_row_bytes))
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

                kv_slot_off_local = self._kv_offset_for_layer[layer_idx]
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
                    self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gpr_seq_len, ue_35bit_addr_shifter(self._kv_row_bytes_for_layer[layer_idx]))
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
                        self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gpr_seq_len, ue_35bit_addr_shifter(self._kv_row_bytes_for_layer[layer_idx]))
                        self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(k_cache_nrot_base), self.TMP_REG)
                        self.sram_to_accelerator_memory(0x10000, 0, remaining, general_reg_src=self.TMP_REG)

                # Gemma4 uses scaling=1.0; q_scale=1.0 on the attention call means no
                # Q pre-scale is needed here.

                _checkpoint(f"L{layer_idx}_rope")

                # K/V cache reads — KV-shared layers point at the source layer's cache.
                kv_slot_off_read = self._kv_offset_for_layer[kv_layer_for_attn]
                kv_k_base = self.LAYER0_K_ROPE_DRAM + kv_slot_off_read
                kv_v_base = self.LAYER0_V_DRAM + kv_slot_off_read

                # Compact per-slot rows already satisfy unified attention's
                # [aligned_seq_len, head_dim] contract. Read the cache directly;
                # zero-initialized future rows cover alignment padding.
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
                    K_DRAM_ADDR=kv_k_base,
                    V_DRAM_ADDR=kv_v_base,
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
                total_flops += _projection_core(M=1, K=cur_q_size, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off,
                    )
                # Decode-only fused attention residual and FFN pre-normalization.
                _vec_a = 0x10000
                _vec_b = 0x90000
                self.accelerator_memory_to_sram(
                    self.LAYER0_ATTN_PROJ_OUTPUT_DRAM, _vec_a, self.vector_length)
                self.accelerator_memory_to_sram(
                    self.DRAM_ADDR_LAYER0_POST_NORM_GAMMA + layer_off,
                    _vec_b, self.vector_length)
                self.rms_norm_core(_vec_a, _vec_a, self.vector_length, _vec_b)
                total_flops += 4 * self.vector_length

                # Attention residual: use layer_input_addr (LAYER0_OUTPUT_DRAM
                # for layers > 0, LAYER0_INPUT_DRAM for layer 0) — same source
                # as the pre-norm above. This avoids the LAYER0_OUTPUT → LAYER0_INPUT
                # copy that used to run at the top of every layer.
                self.accelerator_memory_to_sram(
                    layer_input_addr, _vec_b, self.vector_length)
                self.eltwise_add_core(_vec_a, _vec_b, _vec_a, self.vector_length)
                self.sram_to_accelerator_memory(
                    _vec_a, self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                    self.vector_length)
                self.accelerator_memory_to_sram(
                    self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off,
                    _vec_b, self.vector_length)
                self.rms_norm_core(_vec_a, _vec_a, self.vector_length, _vec_b)
                total_flops += 4 * self.vector_length
                self.sram_to_accelerator_memory(
                    _vec_a, self.LAYER0_PRE_MLP_NORM_DRAM,
                    self.vector_length)

                _checkpoint(f"L{layer_idx}_o_proj")

                total_flops += _projection_core(M=1, K=self.vector_length, N=cur_mlp,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off,
                    gelu_enable=True,
                    )
                total_flops += _projection_core(M=1, K=self.vector_length, N=cur_mlp,
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

                total_flops += _projection_core(M=1, K=cur_mlp, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off,
                    gpr_M_reg=gpr_one,
                    )
                # Decode-only fused MLP post-normalization + residual. Keep the
                # normalized MLP-down vector in SRAM through the residual add,
                # removing the post-MLP-norm DRAM write/read pair.
                self.accelerator_memory_to_sram(
                    self.LAYER0_MLP_DOWN_DRAM, _vec_a, self.vector_length)
                self.accelerator_memory_to_sram(
                    self.DRAM_ADDR_LAYER0_POST_FFW_NORM_GAMMA + layer_off,
                    _vec_b, self.vector_length)
                self.rms_norm_core(_vec_a, _vec_a, self.vector_length, _vec_b)
                total_flops += 4 * self.vector_length
                self.accelerator_memory_to_sram(
                    self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                    _vec_b, self.vector_length)
                self.eltwise_add_core(_vec_a, _vec_b, _vec_a, self.vector_length)
                self.sram_to_accelerator_memory(
                    _vec_a, self.LAYER0_OUTPUT_DRAM, self.vector_length)

                _checkpoint(f"L{layer_idx}_mlp")

                # Per-layer input injection (NEW for Gemma4 E2B) - decoder uses seq_len=1
                total_flops += self._compile_per_layer_injection(layer_idx, layer_off, 1)

                _checkpoint(f"L{layer_idx}_inject")

            if layer_size == self.LAYER_SIZE:
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_OUTPUT_DRAM,
                    OUTPUT_DRAM_ADDR=self.OUTPUT_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_OUTPUT_NORM_GAMMA,
                    gpr_M_reg=gpr_one)
                total_flops += _projection_core(M=1, K=self.vector_length, N=self.EMBEDDING_ELEMENTS,
                    A_DRAM_ADDR=self.OUTPUT_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD_QUANT,
                    OUTPUT_DRAM_ADDR=self.LOGITS_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD_SCALE,
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

        # Pure greedy decode. GEMMA4_PENALTY=1 is rejected in __init__ until
        # dynamic streaming quantized_matmat_core supports broadcast bias.

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

            # Dynamic-PBI dispatch: re-set the attention length (K context grows
            # each step, may cross a 64-align boundary), then jump into the
            # cached decoder program. gpr_seq_len was primed once above and is
            # advanced by the decoder's trailing add_inc.
            latency, flop_rate_program = self._dispatch_program(
                [(self.gpr_aligned_seq_len, aligned_seq_len)],
                prog_addr, timeout=300.0, flops=flops_per_token_scalar)
            total_latency += latency
            total_flop_rate += flop_rate_program
            # HW argmax of the streaming LM-head logits.
            token_id = self.get_arg_max_index()
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
        other section. Rewrites the file so offsets stay contiguous; sections are
        ordered by dram_base (vision 0xa0 before LM 0xc0)."""
        bin_path, meta_path = self._program_image_paths(profile)
        sections, data = self._read_program_sections(profile)
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

        Both programs are seq_len-agnostic (dynamic-PBI: row counts come from
        gpr_seq_len / gpr_q_seq_len / gpr_aligned_seq_len at runtime), so the
        cached image is valid for ANY prompt length up to max_prefill_seq_len.
        Reused as-is whenever programs.bin exists — delete it to force a rebuild.

        set_prefill_seq() MUST have been called first (prefill needs the prompt).
        """
        assert self.prefill_seq is not None, (
            "call set_prefill_seq() before compile_gemma4()")
        prefill_seq_len = len(self.prefill_seq) - 1
        # Reuse the cached LM section only if it was baked at the current program
        # base — jump/scratch addresses are absolute, so a base change (e.g. the
        # LM 0xa0→0xc0 relocation for vision coexistence) forces a rebuild.
        _lm_meta, _ = self._get_program_section("lm", profile)
        if _lm_meta is not None:
            _cached_base = int(_lm_meta["dram_base"], 16)
            _cached_prefill_kernel = _lm_meta.get("prefill_kernel")
            _cached_decode_kernel = _lm_meta.get("decode_kernel")
            _cached_version = _lm_meta.get("kernel_cache_version")
            if (_cached_base == self._program_dram_base
                    and _cached_prefill_kernel == self.prefill_kernel
                    and _cached_decode_kernel == self.decode_kernel
                    and _cached_version == LM_PROGRAM_CACHE_VERSION):
                print("[compile] reusing cached LM section "
                      f"(prefill={self.prefill_kernel}, decode={self.decode_kernel}; "
                      "seq_len-agnostic).")
                return
            print("[compile] cached LM section does not match requested configuration "
                  f"(base=0x{_cached_base:X}, prefill={_cached_prefill_kernel}, "
                  f"decode={_cached_decode_kernel}, version={_cached_version}) — rebuilding.")

        print(f"[compile] building combined [prefill@{prefill_seq_len}, "
              f"kernel={self.prefill_kernel}][decoder, kernel={self.decode_kernel}] image...")
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

        lm_meta = {
            "instruction_base_addr": f"0x{instruction_base_addr:X}",
            "prefill_seq_len": prefill_seq_len,
            "prefill_program_start_addr": f"0x{prefill_program_addr:X}",
            "prefill_program_size": prefill_size_bytes,
            "prefill_total_flops": prefill_total_flops,
            "decoder_program_start_addr": f"0x{decoder_program_addr:X}",
            "decoder_program_size": decoder_size_bytes,
            "decoder_total_flops": decoder_total_flops[0],
            "layer_size": layer_size,
            "prefill_kernel": self.prefill_kernel,
            "decode_kernel": self.decode_kernel,
            "kernel_cache_version": LM_PROGRAM_CACHE_VERSION,
        }
        if profile:
            lm_meta["prefill_profile_checkpoints"] = self._prefill_checkpoints
            lm_meta["decoder_profile_checkpoints"] = self._decoder_checkpoints
        # Merge the LM section into the combined programs bin (vision stays intact).
        self._store_program_section("lm", instruction_base_addr, image_bytes, lm_meta, profile=profile)

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
  python gemma4_e2b_refactor.py --dev xdma1 --cycle 5.042

default prompt: "x+3=5, what is x?"
                (pre-tokenized as default_prefill_tokens in gemma4_e2b_config.json)""")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt. Default is the built-in test question.")
    parser.add_argument("--image", type=str, nargs="?", const=DEFAULT_IMAGE, default=None,
                        help="VLM mode: run the vision encoder on the FPGA and merge image "
                             f"soft-tokens into the prompt. Bare --image uses the default "
                             f"({os.path.basename(DEFAULT_IMAGE)}); omit it for LM-only mode.")
    parser.add_argument("--vision-host", action="store_true",
                        help="With --image, generate image soft-token embeddings on the host "
                             "with the HF vision tower; FPGA prefill/decode are unchanged.")
    parser.add_argument("--prefill-kernel", choices=("streaming", "matmatmul"),
                        default="streaming",
                        help="Quantized projection kernel for LM prefill (default: streaming).")
    parser.add_argument("--decode-kernel", choices=("streaming", "matmatmul"),
                        default="streaming",
                        help="Quantized projection kernel for LM decode, including LM head "
                             "(default: streaming).")
    parser.add_argument("--local-weights", action="store_true",
                        help="Use gemma4_e2b_bin/params.bin instead of the configured weights bin.")
    parser.add_argument('--dev', type=str, default='xdma0',
                        help='DMA device name (e.g., xdma0, xdma1). Default: xdma0')
    parser.add_argument('--cycle', type=float, default=1000 / 198.3256,
                        help='Clock cycle time in nanoseconds '
                             '(default: 1000/198.3256 ≈ 5.042, i.e. 198.3256 MHz)')
    parser.add_argument('--profile', action='store_true',
                        help='Compile a profile bin with per-phase HALT checkpoints and run one '
                             'profiled decode step; print a per-phase HW-latency breakdown.')
    args = parser.parse_args()
    if args.vision_host and not args.image:
        parser.error("--vision-host requires --image")
    if os.environ.get("GEMMA4_PENALTY", "0") == "1":
        parser.error(
            "GEMMA4_PENALTY=1 is temporarily unsupported; dynamic streaming "
            "quantized_matmat_core needs broadcast-bias support first")

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

    print(f"LM kernels: prefill={args.prefill_kernel}, decode={args.decode_kernel}")
    ue = Gemma4_UnifiedEngine(
        local_weights=args.local_weights,
        prefill_kernel=args.prefill_kernel,
        decode_kernel=args.decode_kernel)

    # Prompt first — the prefill program is compiled for its exact length.
    # VLM mode (--image): run the vision encoder on the FPGA now (separate bin
    # @ 0xa0000000); it sets prefill_seq + image soft-tokens that run_prefill
    # merges. The LM prefill/decoder then compiles at 0xc0000000 as usual.
    if args.image:
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
    print("Gemma4 E2B LM test ends.")


if __name__ == "__main__":
    main()
