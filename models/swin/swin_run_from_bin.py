#!/usr/bin/env python3
"""Swin-Large-384 inference from pre-compiled bin files.

Compile first (auto-skipped if already compiled):
    python swin_test.py

Then run:
    python swin_run_from_bin.py --image path/to/image.jpg
"""

import builtins
import json
import os
import sys
import threading
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

_original_print = builtins.print
_SILENT_MODE = False

def _quiet_print(*args, **kwargs):
    if not _SILENT_MODE:
        _original_print(*args, **kwargs)

builtins.print = _quiet_print

import numpy as np
import torch
from PIL import Image

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, DMA_DEVICE_C2H, DRAM_INSTRUCTION_ADDR,
    UE_VECTOR_SIZE, set_dma_device, UnifiedEngine,
)

# ---------------------------------------------------------------------------
# DRAM partition
# ---------------------------------------------------------------------------
SWIN_PARAMS_BASE  = 0x00000000
SWIN_TENSOR_BASE  = 0x40000000
SWIN_PROGRAM_BASE = 0x80000000


def _parse_offset(val) -> int:
    if isinstance(val, str):
        return int(val, 0)
    return int(val)


# ---------------------------------------------------------------------------
# Swin Unified Engine (inference-only — no weight_init, no compile)
# ---------------------------------------------------------------------------

class Swin_UnifiedEngine(UnifiedEngine):

    IMAGE_SIZE   = 384
    PATCH_SIZE   = 4
    NUM_CHANNELS = 3
    EMBED_DIM    = 192
    DEPTHS       = [2, 2, 18, 2]
    NUM_HEADS    = [6, 12, 24, 48]
    WINDOW_SIZE  = 12
    MLP_RATIO    = 4.0
    HEAD_DIM     = 32
    NUM_STAGES   = 4
    GRID_SIZE    = IMAGE_SIZE // PATCH_SIZE  # 96

    STAGE_DIMS    = [192, 384, 768, 1536]
    STAGE_SPATIAL = [96, 48, 24, 12]
    WINDOW_AREA   = 144

    def __init__(self, script_dir: str = SCRIPT_DIR):
        super().__init__(params_dram_base=SWIN_PARAMS_BASE,
                         tensor_dram_base=SWIN_TENSOR_BASE,
                         program_dram_base=SWIN_PROGRAM_BASE)
        cfg = self.load_config(script_dir=script_dir)
        self.cfg = cfg
        self.script_dir = script_dir

        if not self.load_params():
            raise RuntimeError("params.bin not found — run python swin_test.py first.")
        self.tensor_init()

    @staticmethod
    def load_config(config_path: str | None = None, script_dir: str | None = None) -> dict:
        sd = script_dir or SCRIPT_DIR
        cp = config_path or os.path.join(sd, "swin_config.json")
        with open(cp) as f:
            cfg = json.load(f)
        cfg["_weight_defs"] = {}
        return cfg

    ALIGN = 64
    DMA_CHUNK_BYTES = 1 * 1024 * 1024

    @staticmethod
    def pad_dim(x: int) -> int:
        return ((x + 63) // 64) * 64

    def write_captured_instructions_to_dram(self, start_addr: int = DRAM_INSTRUCTION_ADDR) -> int:
        if not self.capture_buffer or self.capture_count == 0:
            return 0
        total_bytes = self.capture_count * 32
        all_bytes = bytearray()
        for inst in self.capture_buffer:
            all_bytes.extend(inst.get_bytes())
        data = bytes(all_bytes)
        offset = 0
        while offset < total_bytes:
            chunk = min(self.DMA_CHUNK_BYTES, total_bytes - offset)
            self.dma_write(DMA_DEVICE_H2C, start_addr + offset, data[offset:offset + chunk], chunk)
            offset += chunk
        return total_bytes

    def _alloc_param(self, tensor: torch.Tensor) -> int:
        t = tensor.to(torch.bfloat16).contiguous()
        addr = self.get_params_dram_addr()
        nbytes = t.numel() * 2
        self.allocate_params_dram(nbytes)
        self.dma_to_accelerator_memory(addr, t)
        return addr

    def _alloc_tensor(self, num_elements: int) -> int:
        return self.allocate_tensor_dram(num_elements * 2)

    def load_params(self) -> bool:
        bin_dir = os.path.join(self.script_dir, "swin_bin")
        bin_path = os.path.join(bin_dir, "params.bin")
        meta_path = os.path.join(bin_dir, "params.json")
        if not os.path.exists(bin_path) or not os.path.exists(meta_path):
            return False
        with open(meta_path) as f:
            meta = json.load(f)
        total = meta["size"]
        self.NUM_LABELS = meta["num_labels"]
        self.NUM_LABELS_PAD = self.pad_dim(self.NUM_LABELS)
        CHUNK = 1 * 1024 * 1024
        with open(bin_path, "rb") as f:
            offset = 0
            while offset < total:
                data = f.read(min(CHUNK, total - offset))
                self.dma_write(DMA_DEVICE_H2C, self._params_dram_base + offset, data, len(data))
                offset += len(data)
        self.allocate_params_dram(total)
        _original_print(f"  Params: {total / 1024**2:.1f} MB from bin")
        return True

    def load_programs(self) -> int | None:
        bin_dir = os.path.join(self.script_dir, "swin_bin")
        bin_path = os.path.join(bin_dir, "programs.bin")
        meta_path = os.path.join(bin_dir, "programs.json")
        if not os.path.exists(bin_path) or not os.path.exists(meta_path):
            return None
        with open(bin_path, "rb") as f:
            data = f.read()
        addr = self.get_program_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, addr, data, len(data))
        self.allocate_program_dram(len(data))
        _original_print(f"  Program: {len(data) / 1024**2:.1f} MB from bin")
        return addr

    def tensor_init(self) -> None:
        pad = self.pad_dim

        M_patches = self.GRID_SIZE * self.GRID_SIZE
        N_embed = self.EMBED_DIM
        C = self.NUM_CHANNELS
        H = W = self.IMAGE_SIZE

        self.IMAGE_DRAM = self._alloc_tensor(C * H * W)
        self.PATCH_OUTPUT_DRAM = self._alloc_tensor(M_patches * N_embed)
        self.EMBED_OUTPUT_DRAM = self._alloc_tensor(M_patches * N_embed)

        self.stage_tensors = []
        for s in range(self.NUM_STAGES):
            dim = self.STAGE_DIMS[s]
            spatial = self.STAGE_SPATIAL[s]
            num_heads = self.NUM_HEADS[s]
            num_windows = (spatial // self.WINDOW_SIZE) ** 2
            wa = self.WINDOW_AREA
            wa_pad = pad(wa)
            hd_pad = pad(self.HEAD_DIM)
            mlp_dim = int(dim * self.MLP_RATIO)
            M = spatial * spatial
            total_batches = num_windows * num_heads

            st = {}
            st['layer_input'] = self._alloc_tensor(M * dim)
            st['ln_output'] = self._alloc_tensor(M * dim)
            st['windowed'] = self._alloc_tensor(num_windows * wa * dim)
            st['shifted'] = self._alloc_tensor(M * dim)
            st['win_reverse'] = self._alloc_tensor(M * dim)
            st['q_proj'] = self._alloc_tensor(num_windows * wa * dim)
            st['k_proj'] = self._alloc_tensor(num_windows * wa * dim)
            st['v_proj'] = self._alloc_tensor(num_windows * wa * dim)
            st['q_heads'] = self._alloc_tensor(total_batches * wa_pad * hd_pad)
            st['k_heads'] = self._alloc_tensor(total_batches * wa_pad * hd_pad)
            st['v_heads'] = self._alloc_tensor(total_batches * wa_pad * hd_pad)
            st['attn_output'] = self._alloc_tensor(total_batches * wa_pad * hd_pad)
            scratch_per_batch = hd_pad * wa_pad + wa_pad * wa_pad
            st['attn_scratch'] = self._alloc_tensor(total_batches * scratch_per_batch)
            st['attn_permuted'] = self._alloc_tensor(num_windows * wa * num_heads * hd_pad)
            st['permute_temp'] = self._alloc_tensor(2 * wa * num_heads * hd_pad)
            st['attn_unpadded'] = self._alloc_tensor(num_windows * wa * dim)
            st['out_proj'] = self._alloc_tensor(num_windows * wa * dim)
            st['residual1'] = self._alloc_tensor(M * dim)
            st['mlp_ln'] = self._alloc_tensor(M * dim)
            st['mlp_mid'] = self._alloc_tensor(M * mlp_dim)
            st['mlp_out'] = self._alloc_tensor(M * dim)
            if s < self.NUM_STAGES - 1:
                next_M = (spatial // 2) * (spatial // 2)
                st['merge_buf'] = self._alloc_tensor(next_M * 4 * dim)
            self.stage_tensors.append(st)

        final_M = self.STAGE_SPATIAL[3] ** 2
        final_C = self.STAGE_DIMS[3]
        self.FINAL_LN_OUTPUT_DRAM = self._alloc_tensor(final_M * final_C)
        self.POOLED_OUTPUT_DRAM = self._alloc_tensor(1 * final_C)
        self.CLASSIFIER_OUTPUT_DRAM = self._alloc_tensor(1 * self.pad_dim(self.NUM_LABELS))

    def run_full_fused(self, pixel_values: torch.Tensor, program_addr: int, timeout: float = 120.0) -> int:
        image_chw = pixel_values.squeeze(0).to(torch.bfloat16)
        self.dma_to_accelerator_memory(self.IMAGE_DRAM, image_chw.contiguous().flatten())
        self.start_execute_from_dram(program_addr)
        self.wait_queue(timeout)
        return self.get_arg_max_index1()

    def get_arg_max_index1(self):
        return self.read_reg32(user_dma_core.UE_ARGMAX1_INDEX)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(image_path: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB").resize((384, 384), Image.Resampling.BILINEAR)
    img_t = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img_t - mean) / std


def check_bins_early(script_dir):
    bin_dir = os.path.join(script_dir, "swin_bin")
    missing = []
    for name in ["params.bin", "params.json", "programs.bin", "programs.json", "labels.json"]:
        p = os.path.join(bin_dir, name)
        if not os.path.exists(p):
            missing.append(os.path.join("swin_bin", name))
    return missing


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Swin-Large-384 inference from pre-compiled bins")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--dev", type=str, default="xdma0")
    parser.add_argument("--cycle", type=float, default=5.62)
    args = parser.parse_args()

    missing = check_bins_early(SCRIPT_DIR)
    if missing:
        _original_print("Missing bin files (run python swin_test.py first):")
        for f in missing:
            _original_print(f"  {f}")
        return

    image_path = args.image or os.path.join(SCRIPT_DIR, "../../test_samples/vette.jpg")
    pixel_values = preprocess_image(image_path)

    set_dma_device(args.dev)
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle

    _original_print(f"Swin-Large-384 on {args.dev}")

    t0 = time.perf_counter()
    ue = Swin_UnifiedEngine(script_dir=SCRIPT_DIR)
    ue.software_reset()
    _original_print(f"  Weights: {time.perf_counter() - t0:.3f}s")

    prog_addr = ue.load_programs()
    if prog_addr is None:
        _original_print("programs.bin not found — run python swin_test.py first.")
        return

    def _progress_timer(start_time, stop_event):
        while not stop_event.wait(1.0):
            _original_print(f"\r  Executing ({time.perf_counter() - start_time:.0f}s)", end="", flush=True)

    t_exec = time.perf_counter()
    stop = threading.Event()
    timer = threading.Thread(target=_progress_timer, args=(t_exec, stop), daemon=True)
    timer.start()

    predicted_idx = ue.run_full_fused(pixel_values, prog_addr)

    stop.set()
    timer.join()
    _original_print(f"\r  Executing: {time.perf_counter() - t_exec:.3f}s")

    labels_path = os.path.join(SCRIPT_DIR, "swin_bin", "labels.json")
    with open(labels_path) as f:
        id2label = json.load(f)
    label = id2label.get(str(predicted_idx), str(predicted_idx))

    _original_print(f"\n  Image: {image_path}")
    _original_print(f"  Prediction: {label!r} (class {predicted_idx})")


if __name__ == "__main__":
    main()
