"""model_lib_core.py

Shared helpers extracted from per-model bring-up files. Every public op uses the
``_core_dram`` naming convention for DRAM-in/DRAM-out wrappers around the SRAM
primitives in ``user_dma_core.UnifiedEngine``.

Functions here are intentionally thin — state lives on the ``ue`` (UnifiedEngine)
instance. This file never mutates ``user_dma_core``; callers still construct a
``UnifiedEngine`` subclass and pass it in.

Grouping:
  §1 Weight staging / DRAM loaders
  §2 DRAM elementwise wrappers
  §3 Fused post-add norm variants (residual + norm in one pass)
  §4 Quantization helpers (bf16 → INT4 packed, Q4_64, FP4 E2M1)
  §5 Weight-bin plumbing (manifest writer, name mapping hook)
  §6 Simple-LM boilerplate (config loading, HF download, quiet print)
"""
from __future__ import annotations

import builtins
import json
import os
import struct
from typing import Callable

import numpy as np
import torch

from user_dma_core import (
    DMA_DEVICE_H2C,
    MEMCPY_TYPE,
    UE_VECTOR_SIZE,
    URAM_NEAR_FULL_ELEMENTS,
)

# =============================================================================
# §1 Weight staging / DRAM loaders
# =============================================================================
def store_weight(ue, tensor: torch.Tensor, padded_shape=None) -> int:
    """Pad, cast to bf16, DMA to params DRAM. Returns the DRAM address."""
    bf16 = tensor.to(torch.bfloat16)
    if padded_shape is not None:
        padded = torch.zeros(padded_shape, dtype=torch.bfloat16)
        slices = tuple(slice(0, s) for s in bf16.shape)
        padded[slices] = bf16
        bf16 = padded
    bf16 = bf16.contiguous()
    nbytes = bf16.numel() * 2
    addr = ue.get_params_dram_addr()
    ue.dma_write(DMA_DEVICE_H2C, addr, bf16.flatten(), nbytes)
    ue.allocate_params_dram(nbytes)
    return addr


def store_quantized_weight(ue, raw_data) -> tuple[int, int]:
    """Store Q4_64 raw GGUF data as (scales, packed) in DRAM. Returns (scale_addr, data_addr)."""
    raw_bytes = raw_data.tobytes() if hasattr(raw_data, "tobytes") else bytes(raw_data)
    n_blocks = len(raw_bytes) // 34
    scales_size = n_blocks * 2
    data_size = n_blocks * 32

    scales_np = np.frombuffer(raw_bytes[:scales_size], dtype=np.uint16).copy()
    scale_tensor = torch.from_numpy(scales_np).view(torch.bfloat16)
    scale_addr = ue.get_params_dram_addr()
    ue.dma_write(DMA_DEVICE_H2C, scale_addr, scale_tensor, scales_size)
    ue.allocate_params_dram(scales_size)

    data_np = np.frombuffer(raw_bytes[scales_size:scales_size + data_size], dtype=np.uint8).copy()
    data_tensor = torch.from_numpy(data_np)
    data_addr = ue.get_params_dram_addr()
    ue.dma_write(DMA_DEVICE_H2C, data_addr, data_tensor, data_size)
    ue.allocate_params_dram(data_size)

    return scale_addr, data_addr


def load_weight_cache(bin_path: str) -> dict:
    """Load bin + sibling json manifest. Returns ``{tensor_name: raw_uint8_numpy}``."""
    json_path = bin_path.rsplit(".", 1)[0] + ".json"
    with open(json_path) as f:
        manifest = json.load(f)
    with open(bin_path, "rb") as f:
        raw = f.read()
    cache = {}
    for name, meta in manifest.items():
        cache[name] = np.frombuffer(
            raw[meta["offset"]:meta["offset"] + meta["size"]], dtype=np.uint8
        ).copy()
    return cache


def store_identity_matrix(ue) -> int:
    """Write a UE_VECTOR_SIZE × UE_VECTOR_SIZE bf16 identity into params DRAM. Returns its address."""
    bpe = 2
    size = UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe
    addr = ue.get_params_dram_addr()
    ue.dma_write(DMA_DEVICE_H2C, addr, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16), size)
    ue.allocate_params_dram(size)
    return addr


# =============================================================================
# §2 DRAM elementwise wrappers
# =============================================================================
def eltwise_add_core_dram(ue, size: int, A_DRAM_ADDR: int, B_DRAM_ADDR: int,
                          OUTPUT_DRAM_ADDR: int) -> int:
    """OUTPUT = A + B over ``size`` bf16 elements, streamed through URAM A/B."""
    bytes_per_element = 2
    uram_a_addr = 0x00000
    uram_b_addr = 0x80000
    chunk_size = min(URAM_NEAR_FULL_ELEMENTS, size)

    for start, take in ue.chunk_ranges(size, chunk_size):
        offset_bytes = start * bytes_per_element
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=A_DRAM_ADDR + offset_bytes,
            sram_address=uram_a_addr, element_size=take)
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=B_DRAM_ADDR + offset_bytes,
            sram_address=uram_b_addr, element_size=take)
        ue.eltwise_add_core(
            vector_A_sram_start_addr=uram_a_addr,
            vector_B_sram_start_addr=uram_b_addr,
            vector_C_sram_wb_addr=uram_a_addr, element_size=take)
        ue.sram_to_accelerator_memory(
            sram_address=uram_a_addr,
            accelerator_dram_address=OUTPUT_DRAM_ADDR + offset_bytes,
            element_size=take)
    return size


def eltwise_mul_core_dram(ue, size: int, A_DRAM_ADDR: int, B_DRAM_ADDR: int,
                          OUTPUT_DRAM_ADDR: int) -> int:
    """OUTPUT = A * B over ``size`` bf16 elements, streamed through URAM A/B."""
    bytes_per_element = 2
    uram_a_addr = 0x00000
    uram_b_addr = 0x80000
    chunk_size = min(URAM_NEAR_FULL_ELEMENTS, size)

    for start, take in ue.chunk_ranges(size, chunk_size):
        offset_bytes = start * bytes_per_element
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=A_DRAM_ADDR + offset_bytes,
            sram_address=uram_a_addr, element_size=take)
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=B_DRAM_ADDR + offset_bytes,
            sram_address=uram_b_addr, element_size=take)
        ue.eltwise_mul_core(
            vector_A_sram_start_addr=uram_a_addr,
            vector_B_sram_start_addr=uram_b_addr,
            vector_C_sram_wb_addr=uram_a_addr, element_size=take)
        ue.sram_to_accelerator_memory(
            sram_address=uram_a_addr,
            accelerator_dram_address=OUTPUT_DRAM_ADDR + offset_bytes,
            element_size=take)
    return size


# =============================================================================
# §3 Fused post-add norm variants
# =============================================================================
def rms_norm_core_dram_post_add(ue, M: int, N: int,
                                A_DRAM_ADDR: int, B_DRAM_ADDR: int,
                                ADDOUTPUT_DRAM_ADDR: int, NORMOUTPUT_DRAM_ADDR: int,
                                GAMMA_DRAM_ADDR: int | None = None) -> int:
    """rms_norm(A + B). Writes both the pre-norm sum (residual) and the normed output."""
    gamma_sram_addr = 0x80000
    params_sram_addr = gamma_sram_addr

    if GAMMA_DRAM_ADDR is not None:
        ue.accelerator_memory_to_sram(accelerator_dram_address=GAMMA_DRAM_ADDR,
                                      sram_address=gamma_sram_addr, element_size=N)
        params_sram_addr += N * 2
    else:
        gamma_sram_addr = None

    vector_A_sram_addr = 0x00000
    vector_B_sram_addr = params_sram_addr
    uram_b_remaining_elements = URAM_NEAR_FULL_ELEMENTS - (params_sram_addr - 0x80000) // 2
    chunk_size = min(URAM_NEAR_FULL_ELEMENTS // N, uram_b_remaining_elements // N, M)
    assert chunk_size >= 1 and chunk_size <= M

    for i, m_take in ue.chunk_ranges(M, chunk_size):
        chunk_elements = m_take * N
        ue.accelerator_memory_to_sram(accelerator_dram_address=A_DRAM_ADDR + i * N * 2,
                                      sram_address=vector_A_sram_addr, element_size=chunk_elements)
        ue.accelerator_memory_to_sram(accelerator_dram_address=B_DRAM_ADDR + i * N * 2,
                                      sram_address=vector_B_sram_addr, element_size=chunk_elements)
        for j in range(m_take):
            ue.eltwise_add_core(vector_A_sram_addr + j * N * 2, vector_B_sram_addr + j * N * 2,
                                vector_A_sram_addr + j * N * 2, N)
        ue.sram_to_accelerator_memory(sram_address=vector_A_sram_addr,
                                      accelerator_dram_address=ADDOUTPUT_DRAM_ADDR + i * N * 2,
                                      element_size=chunk_elements)
        for j in range(m_take):
            ue.rms_norm_core(vector_A_sram_addr + j * N * 2, vector_A_sram_addr + j * N * 2,
                             N, gamma_sram_addr)
        ue.sram_to_accelerator_memory(sram_address=vector_A_sram_addr,
                                      accelerator_dram_address=NORMOUTPUT_DRAM_ADDR + i * N * 2,
                                      element_size=chunk_elements)

    total_flops = M * N + 3 * M * N
    if gamma_sram_addr is not None:
        total_flops += M * N
    return total_flops


def layer_norm_core_dram_post_add(ue, M: int, N: int,
                                  A_DRAM_ADDR: int, B_DRAM_ADDR: int,
                                  ADDOUTPUT_DRAM_ADDR: int, NORMOUTPUT_DRAM_ADDR: int,
                                  GAMMA_DRAM_ADDR: int | None = None,
                                  BETA_DRAM_ADDR: int | None = None) -> int:
    """layer_norm(A + B). Writes both the pre-norm sum and the normed output."""
    zeros_sram_addr = 0x80000

    zeros_dram_addr = ue.get_params_dram_addr()
    ue.allocate_params_dram(N * 2)
    ue.dma_write(DMA_DEVICE_H2C, zeros_dram_addr, torch.zeros(N, dtype=torch.bfloat16), N * 2)

    ue.accelerator_memory_to_sram(accelerator_dram_address=zeros_dram_addr,
                                  sram_address=zeros_sram_addr, element_size=N)
    params_sram_addr = zeros_sram_addr + N * 2

    gamma_sram_addr = None
    beta_sram_addr = None

    if GAMMA_DRAM_ADDR is not None:
        gamma_sram_addr = params_sram_addr
        ue.accelerator_memory_to_sram(accelerator_dram_address=GAMMA_DRAM_ADDR,
                                      sram_address=gamma_sram_addr, element_size=N)
        params_sram_addr += N * 2

    if BETA_DRAM_ADDR is not None:
        beta_sram_addr = params_sram_addr
        ue.accelerator_memory_to_sram(accelerator_dram_address=BETA_DRAM_ADDR,
                                      sram_address=beta_sram_addr, element_size=N)
        params_sram_addr += N * 2

    vector_A_sram_addr = 0x00000
    vector_B_sram_addr = params_sram_addr
    uram_b_remaining_elements = URAM_NEAR_FULL_ELEMENTS - (params_sram_addr - 0x80000) // 2
    chunk_size = min(URAM_NEAR_FULL_ELEMENTS // N, uram_b_remaining_elements // N, M)
    assert chunk_size >= 1 and chunk_size <= M

    for i, m_take in ue.chunk_ranges(M, chunk_size):
        chunk_elements = m_take * N
        ue.accelerator_memory_to_sram(accelerator_dram_address=A_DRAM_ADDR + i * N * 2,
                                      sram_address=vector_A_sram_addr, element_size=chunk_elements)
        ue.accelerator_memory_to_sram(accelerator_dram_address=B_DRAM_ADDR + i * N * 2,
                                      sram_address=vector_B_sram_addr, element_size=chunk_elements)
        for j in range(m_take):
            ue.eltwise_add_core(vector_A_sram_addr + j * N * 2, vector_B_sram_addr + j * N * 2,
                                vector_A_sram_addr + j * N * 2, N)
        ue.sram_to_accelerator_memory(sram_address=vector_A_sram_addr,
                                      accelerator_dram_address=ADDOUTPUT_DRAM_ADDR + i * N * 2,
                                      element_size=chunk_elements)
        for j in range(m_take):
            ue.layer_norm_core(vector_A_sram_addr + j * N * 2, vector_A_sram_addr + j * N * 2,
                               N, zeros_sram_addr, gamma_sram_addr, beta_sram_addr)
        ue.sram_to_accelerator_memory(sram_address=vector_A_sram_addr,
                                      accelerator_dram_address=NORMOUTPUT_DRAM_ADDR + i * N * 2,
                                      element_size=chunk_elements)

    total_flops = M * N + 5 * M * N
    if gamma_sram_addr is not None:
        total_flops += M * N
    if beta_sram_addr is not None:
        total_flops += M * N
    return total_flops


# =============================================================================
# §4 Quantization helpers
# =============================================================================
# FP4 E2M1 lookup table (16 values, codes 0-7 positive, 8-15 negative).
# Kept here as the canonical default; callers may override by passing their own
# table as ``fp4_table`` to ``quantize_fp4_64``.
_FP4_E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.bfloat16,
)


def quantize_bf16_to_int4_packed(weight_bf16: torch.Tensor,
                                 block_size: int = 64) -> tuple[bytes, bytes]:
    """Quantize bf16 (N_w, K_w) to INT4 packed + per-block bf16 scale along K.

    Returns ``(data_bytes, scale_bytes)``. ``K_w`` must be a multiple of ``block_size``.
    """
    w = weight_bf16.detach().cpu().float().reshape(-1)
    N_w, K_w = weight_bf16.shape
    assert K_w % block_size == 0
    w_blocks = w.reshape(N_w, K_w // block_size, block_size)
    scale = w_blocks.abs().amax(dim=-1).clamp(min=1e-8) / 7.0
    scale_bf16 = scale.to(torch.bfloat16)
    w_int8 = (w_blocks / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int8)
    w_nibbles = w_int8.numpy().astype(np.int16) & 0x0F
    low = w_nibbles[:, :, 0::2].reshape(N_w, -1)
    high = w_nibbles[:, :, 1::2].reshape(N_w, -1)
    packed = (high << 4) | low
    data_bytes = packed.astype(np.uint8).tobytes()
    scale_bytes = scale_bf16.contiguous().view(torch.uint8).numpy().tobytes()
    return data_bytes, scale_bytes


def quantize_q4_64(tensor: torch.Tensor) -> tuple[np.ndarray, int]:
    """INT4 quantization with 64-element blocks. Returns (packed_uint8_array, n_blocks).

    Layout: [bf16 scales per block][packed int4 data]. Matches GGUF Q4_64.
    """
    data = tensor.flatten().cpu().float().numpy()
    n_blocks = int(np.ceil(len(data) / 64))
    padded = np.pad(data, (0, n_blocks * 64 - len(data)))
    blocks = padded.reshape(n_blocks, 64)
    scales = np.max(np.abs(blocks), axis=1)
    scales[scales == 0] = 1.0
    scales /= 7.0
    quantized = np.clip(np.round(blocks / scales[:, None]), -8, 7).astype(np.int8)
    pairs = (quantized.astype(np.uint8) & 0x0F).reshape(n_blocks, 32, 2)
    packed = pairs[:, :, 0] | (pairs[:, :, 1] << 4)
    scale_bytes = torch.tensor(scales, dtype=torch.float32).to(torch.bfloat16).view(torch.uint16).numpy()
    return np.frombuffer(scale_bytes.tobytes() + packed.tobytes(), dtype=np.uint8), n_blocks


def quantize_fp4_64(tensor: torch.Tensor,
                    fp4_table: torch.Tensor | None = None) -> tuple[np.ndarray, int]:
    """FP4 E2M1 quantization with 64-element blocks. Returns (packed_uint8_array, n_blocks)."""
    table = fp4_table if fp4_table is not None else _FP4_E2M1_TABLE
    x = tensor.to(torch.bfloat16).cpu().flatten()
    n_blocks = int(np.ceil(x.numel() / 64))
    if x.numel() % 64 != 0:
        x = torch.nn.functional.pad(x, (0, n_blocks * 64 - x.numel()))
    blocks = x.view(n_blocks, 64)
    fp4_max = torch.tensor(6.0, dtype=torch.bfloat16)
    scales = (blocks.abs().max(dim=1).values / fp4_max).clamp(min=1e-8).to(torch.bfloat16)
    scaled = (blocks / scales[:, None]).to(torch.bfloat16)
    codes = torch.argmin(torch.abs(scaled.unsqueeze(-1) - table), dim=-1).to(torch.uint8)
    codes_np = codes.numpy().flatten()
    if len(codes_np) % 2 != 0:
        codes_np = np.pad(codes_np, (0, 1))
    packed = (codes_np[0::2] & 0x0F) | ((codes_np[1::2] & 0x0F) << 4)
    scales_np = scales.view(torch.uint16).numpy()
    return np.frombuffer(scales_np.tobytes() + packed.tobytes(), dtype=np.uint8), n_blocks


# =============================================================================
# §5 Weight-bin plumbing
# =============================================================================
def write_weight_bin(bin_path: str,
                     model,
                     param_filter: Callable[[str], bool],
                     mode: str,
                     qtype: str,
                     qfn: Callable,
                     weight_key_fn: Callable[[str, str], str],
                     quant_suffixes: dict[str, set[str]],
                     transform_fn: Callable[[str, torch.Tensor], torch.Tensor] | None = None) -> None:
    """Write (possibly quantized) weights to ``bin_path`` with a sibling ``.json`` manifest.

    ``weight_key_fn(hf_name, mode)`` returns the short manifest key.
    ``quant_suffixes[qtype]`` is the set of HF-name suffixes that get quantized via ``qfn``.
    ``transform_fn`` optionally rewrites a tensor before bf16 packing (e.g. transpose).
    """
    json_path = bin_path.rsplit(".", 1)[0] + ".json"
    manifest: dict = {}
    count = 0
    with open(bin_path, "wb") as f:
        for pname, param in model.named_parameters():
            if not param_filter(pname):
                continue
            key = weight_key_fn(pname, mode)
            t = param.data
            if any(pname.endswith(s) for s in quant_suffixes[qtype]):
                data, _ = qfn(t)
                raw = data.tobytes()
                key = f"{key}.{qtype}"
            else:
                if transform_fn is not None:
                    t = transform_fn(pname, t)
                raw = t.to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().tobytes()
            offset = f.tell()
            f.write(raw)
            manifest[key] = {"offset": offset, "size": len(raw)}
            count += 1
    with open(json_path, "w") as f:
        json.dump(manifest, f)
    print(f"Weights: {count} tensors, {os.path.getsize(bin_path)/1048576:.1f} MB → {bin_path}")


# =============================================================================
# §6 Simple-LM boilerplate
# =============================================================================
_ORIGINAL_PRINT = builtins.print
_SILENT_MODE = False


def set_silent(silent: bool) -> None:
    """Toggle the global silent mode used by ``quiet_print``."""
    global _SILENT_MODE
    _SILENT_MODE = silent


def quiet_print(*args, **kwargs) -> None:
    """``print`` replacement that respects ``set_silent``."""
    if _SILENT_MODE:
        return
    _ORIGINAL_PRINT(*args, **kwargs)


def install_quiet_print() -> None:
    """Monkeypatch ``builtins.print`` with ``quiet_print`` (call once at module import)."""
    builtins.print = quiet_print


def parse_offset(val) -> int:
    """Parse offset/size JSON field: int or hex string like ``'0x24000000'``."""
    if isinstance(val, str):
        return int(val, 0)
    return int(val)


def ensure_hf_model(script_dir: str, cfg: dict, model_cls):
    """Download (if missing) and load an HF causal-LM. Returns ``(model, model_dir)``.

    ``model_cls`` is the class to call ``.from_pretrained`` on (e.g.
    ``transformers.AutoModelForCausalLM``). Kept as a parameter so this module
    does not depend on ``transformers`` at import time.
    """
    from huggingface_hub import snapshot_download

    model_dir = os.path.join(script_dir, cfg["paths"]["hf_model_dir"])
    hf_repo = cfg["paths"]["hf_model_repo"]
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        _ORIGINAL_PRINT(f"Downloading HF model {hf_repo} to {os.path.abspath(model_dir)} ...")
        snapshot_download(repo_id=hf_repo, local_dir=model_dir, local_dir_use_symlinks=False)
        _ORIGINAL_PRINT("Download complete.")
    model = model_cls.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map=None, trust_remote_code=True
    )
    return model, model_dir


def load_config_with_weight_defs(config_path: str) -> dict:
    """Load a per-model config JSON and derive the ``_weight_defs`` offset/size dict.

    Expects the config to have ``file_info.layer_size`` and ``regions`` /
    ``non_layer_regions`` maps of ``{key: {offset, size}}``. Attaches the flat
    ``weight_defs`` dict to the returned config under key ``_weight_defs``.
    """
    with open(config_path) as f:
        cfg = json.load(f)
    weight_defs = {"LAYER_WEIGHT_SIZE": cfg["file_info"]["layer_size"]}
    for key, r in cfg.get("regions", {}).items():
        weight_defs[key] = parse_offset(r["offset"])
        weight_defs[f"{key}_SIZE"] = r["size"]
    for key, r in cfg.get("non_layer_regions", {}).items():
        weight_defs[key] = parse_offset(r["offset"])
        weight_defs[f"{key}_SIZE"] = r["size"]
    cfg["_weight_defs"] = weight_defs
    return cfg
