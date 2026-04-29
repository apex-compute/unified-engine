#!/usr/bin/env python3
"""MobileSAM inference from pre-compiled bin files. Run mobilesam_test.py --point to generate bins first.

Usage:
  python mobilesam_run_from_bin.py --point 512 512
  python mobilesam_run_from_bin.py --point 512 512 --dev xdma0

Save phase (run from mobilesam_test.py): see dump_bins() in that file.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from PIL import Image as _PIL_Image
from huggingface_hub import hf_hub_download

_original_print = print

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, DMA_DEVICE_C2H, DRAM_INSTRUCTION_ADDR,
    UE_VECTOR_SIZE, URAM_NEAR_FULL_ELEMENTS,
    set_dma_device, UnifiedEngine,
)

BIN_DIR = os.path.join(SCRIPT_DIR, "mobilesam_bin")

# ---------------------------------------------------------------------------
# Architecture constants (match mobilesam_test.py)
# ---------------------------------------------------------------------------
ENC_IN_H = ENC_IN_W = 1024
IMG_H = IMG_W = 64
GA = IMG_H * IMG_W  # 4096
DEC_DIM = 256
NT = 7
NT_PAD = 64
BPE = 2
DEC_HEADS = 8
DEC_HD_PAD = 64
DEC_SA_HD = 32
DEC_CA_HD = 16
DEC_MLP_DIM = 2048
DEC_LAYERS = 2

_MASK_THRESHOLD = 0.0


class MobileSAM_UE_Run(UnifiedEngine):
    """MobileSAM engine that loads pre-compiled bins (no weight unpacking, no compilation)."""

    def __init__(self, clock_period_ns=None):
        super().__init__(clock_period_ns=clock_period_ns)
        self.script_dir = SCRIPT_DIR
        # Hang prevention
        self.dram_inst_running(False)
        self.start_capture()
        self.generate_instruction_halt()
        self.stop_capture()
        halt_bytes = bytearray()
        for inst in self.capture_buffer:
            halt_bytes.extend(inst.get_bytes())
        self.dma_write(DMA_DEVICE_H2C, self._program_dram_base, halt_bytes, len(halt_bytes))
        self.clear_capture_buffer()

    # ------------------------------------------------------------------
    # Load from bin
    # ------------------------------------------------------------------

    def load_params(self) -> bool:
        """Load params DRAM from bin. Returns True if loaded."""
        bin_path = os.path.join(BIN_DIR, "params.bin")
        meta_path = os.path.join(BIN_DIR, "params.json")
        if not os.path.exists(bin_path) or not os.path.exists(meta_path):
            return False
        with open(meta_path) as f:
            meta = json.load(f)
        total = meta["size"]
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

    def load_programs(self) -> dict | None:
        """Load compiled programs from bin. Returns dict of {name: dram_addr} or None."""
        bin_path = os.path.join(BIN_DIR, "programs.bin")
        meta_path = os.path.join(BIN_DIR, "programs.json")
        if not os.path.exists(bin_path) or not os.path.exists(meta_path):
            return None
        with open(meta_path) as f:
            manifest = json.load(f)
        with open(bin_path, "rb") as f:
            all_bytes = f.read()
        addrs = {}
        for name, meta in manifest["programs"].items():
            data = all_bytes[meta["offset"]:meta["offset"] + meta["size"]]
            addr = self.get_program_dram_addr()
            self.dma_write(DMA_DEVICE_H2C, addr, data, len(data))
            self.allocate_program_dram(len(data))
            addrs[name] = addr
        _original_print(f"  Programs: {len(all_bytes)} bytes from bin")
        return addrs

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute_encoder(self, prog: int, image_t: torch.Tensor) -> None:
        """Execute pre-compiled encoder program on input image."""
        img_hwc = image_t[0].permute(1, 2, 0).contiguous()
        img_pad = torch.zeros(ENC_IN_H * ENC_IN_W, 64, dtype=torch.bfloat16)
        img_pad[:, :3] = img_hwc.reshape(-1, 3)
        self.dma_to_accelerator_memory(self.pe_in_dram, img_pad.reshape(-1))
        self.start_execute_from_dram(prog)
        self.wait_queue(120.0)

    def dma_to_accelerator_memory(self, dma_address, data):
        """Chunked DMA write with flat bf16 handling."""
        if isinstance(data, torch.Tensor):
            data = data.contiguous().flatten()
        flat = data
        total_bytes = flat.numel() * 2 if hasattr(flat, 'numel') else len(data)
        CHUNK = 1 * 1024 * 1024
        if total_bytes <= CHUNK:
            self.dma_write(DMA_DEVICE_H2C, dma_address, flat, total_bytes)
            return
        elems_per_chunk = CHUNK // 2
        offset = 0
        n = flat.numel() if hasattr(flat, 'numel') else len(data)
        while offset < n:
            chunk_elems = min(elems_per_chunk, n - offset)
            chunk = flat[offset:offset + chunk_elems]
            if hasattr(chunk, 'contiguous'):
                chunk = chunk.contiguous()
            self.dma_write(DMA_DEVICE_H2C, dma_address + offset * 2, chunk, chunk_elems * 2)
            offset += chunk_elems

    def dma_from_accelerator_memory(self, dram_addr, shape):
        """Read bf16 tensor from accelerator DRAM."""
        numel = 1
        for d in shape:
            numel *= d
        buf = torch.zeros(numel, dtype=torch.bfloat16)
        self.dma_read(DMA_DEVICE_C2H, dram_addr, buf, numel * 2)
        return buf.reshape(shape)

    def run_decoder(self, prog: int, tokens_t: torch.Tensor,
                    image_emb_t: torch.Tensor, image_pe_t: torch.Tensor,
                    dense_t: torch.Tensor, tensor_addrs: dict,
                    timeout: float = 120.0):
        """Run the compiled mask decoder program.

        Returns (masks_hw, iou_hw) where masks_hw is (4, 256, 256) and iou_hw is (4,).
        """
        src_t = (image_emb_t + dense_t).to(torch.bfloat16).contiguous()

        self.dma_to_accelerator_memory(tensor_addrs["tokens"], tokens_t.flatten())
        self.dma_to_accelerator_memory(tensor_addrs["tokens_pe"], tokens_t.flatten())
        self.dma_to_accelerator_memory(tensor_addrs["src"], src_t.flatten())
        self.dma_to_accelerator_memory(tensor_addrs["key_pe"], image_pe_t.flatten())

        self.start_execute_from_dram(prog)
        self.wait_queue(timeout)

        mask_out = tensor_addrs["mask_out"]
        iou_out = tensor_addrs["iou_out"]
        masks_hw = self.dma_from_accelerator_memory(mask_out, (4, 256 * 256)).float().reshape(4, 256, 256)
        iou_hw = self.dma_from_accelerator_memory(iou_out, (64,)).float()[:4]
        return masks_hw, iou_hw


def main():
    parser = argparse.ArgumentParser(description="MobileSAM inference from pre-compiled bins")
    parser.add_argument("--point", type=int, nargs=2, metavar=("X", "Y"), default=[512, 512],
                        help="Single-point inference at (x, y) (default: 512 512)")
    parser.add_argument("--dev", type=str, default="xdma0")
    parser.add_argument("--cycle", type=float, default=None)
    args = parser.parse_args()

    # Check bins exist
    missing = []
    for f in ("params.bin", "programs.bin"):
        if not os.path.exists(os.path.join(BIN_DIR, f)):
            missing.append(f)
    if missing:
        _original_print("Missing bin files. Run mobilesam_test.py first to generate them.")
        for f in missing:
            _original_print(f"  {os.path.join(BIN_DIR, f)}")
        return

    # Load manifests
    with open(os.path.join(BIN_DIR, "programs.json")) as f:
        prog_meta = json.load(f)
    with open(os.path.join(BIN_DIR, "params.json")) as f:
        param_meta = json.load(f)

    set_dma_device(args.dev)
    if args.cycle is not None:
        user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle

    # Load image
    img_path = os.path.join(SCRIPT_DIR, "../../../test_samples/vette.jpg")
    if not os.path.exists(img_path):
        _original_print(f"Test image not found: {img_path}")
        return
    img = _PIL_Image.open(img_path).convert("RGB")
    img = img.resize((ENC_IN_W, ENC_IN_H), _PIL_Image.BILINEAR)
    img_arr = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    image_t = img_arr.unsqueeze(0).bfloat16()

    _original_print(f"MobileSAM on {args.dev} (from pre-compiled bins)\n")

    # Build engine
    engine = MobileSAM_UE_Run(clock_period_ns=args.cycle)
    engine.load_params()

    # Allocate tensor DRAM from saved manifest — use absolute offsets from compilation
    tensor_addrs = {}
    max_end = 0
    for name, ofs in param_meta["tensors"].items():
        addr = engine._tensor_dram_base + ofs
        tensor_addrs[name] = addr
        end = addr + param_meta["tensor_sizes"][name]
        if end > max_end:
            max_end = end
    # Advance the tensor allocator to protect the region
    if max_end > engine._tensor_dram_addr:
        engine._tensor_dram_addr = max_end
    engine.pe_in_dram = tensor_addrs.get("pe_in")

    # Load programs
    progs = engine.load_programs()
    if progs is None:
        _original_print("Failed to load programs")
        return
    enc_prog = progs.get("encoder")
    dec_prog = progs.get("decoder")
    if enc_prog is None or dec_prog is None:
        _original_print("Missing encoder or decoder program in manifest")
        return

    px, py = args.point
    _original_print(f"Single-point inference at ({px}, {py})")

    # Build sparse prompt tokens
    # We need a MobileSAM prompt encoder — load from HF checkpoint
    from mobilesam_test import MobileSAM, load_weights, _assemble_tokens
    hf_repo = "apexcompute/mobile-sam"
    hf_filename = "mobile_sam.pt"
    ckpt_path = os.path.join(BIN_DIR, hf_filename)
    if not os.path.exists(ckpt_path):
        _original_print(f"  Downloading HF weights {hf_repo}/{hf_filename} …")
        ckpt_path = hf_hub_download(repo_id=hf_repo, filename=hf_filename, local_dir=BIN_DIR)

    # CPU reference for sparse prompt encoding
    raw = MobileSAM().bfloat16().eval()
    load_weights(raw, ckpt_path)

    with torch.no_grad():
        coord = torch.tensor([[float(px), float(py)]])
        pts = coord.reshape(1, 1, 2).bfloat16()
        lbs = torch.ones(1, 1, dtype=torch.long)
        sparse, _ = raw.prompt_encoder(pts, lbs)
        sparse_tok = sparse[0].bfloat16()
        image_pe_t = raw.pe_layer.forward_grid((IMG_H, IMG_W))[0] \
                        .permute(1, 2, 0).reshape(GA, DEC_DIM).bfloat16()
        dense_t = raw.prompt_encoder.no_mask_embed.weight \
                      .reshape(1, DEC_DIM).expand(GA, -1).bfloat16().contiguous()

    tokens_t = _assemble_tokens(ckpt_path, sparse_tok)

    # CPU reference
    _original_print("  Running CPU reference …")
    with torch.no_grad():
        cpu_enc_out = raw.image_encoder(image_t.bfloat16())
        cpu_img_emb = cpu_enc_out[0].permute(1, 2, 0).reshape(GA, DEC_DIM).bfloat16()
        masks_cpu, iou_cpu = raw.mask_decoder(
            cpu_img_emb.reshape(1, IMG_H, IMG_W, DEC_DIM).permute(0, 3, 1, 2),
            image_pe_t.reshape(1, IMG_H, IMG_W, DEC_DIM).permute(0, 3, 1, 2),
            sparse, dense_t.reshape(1, IMG_H, IMG_W, DEC_DIM).permute(0, 3, 1, 2))
        masks_cpu = masks_cpu[0]; iou_cpu = iou_cpu[0]
    best_cpu = iou_cpu.argmax().item()
    _original_print(f"  CPU IOU: {[round(x,4) for x in iou_cpu.tolist()]}  best={best_cpu}")

    # HW encoder
    t_enc = time.perf_counter()
    engine.execute_encoder(enc_prog, image_t)
    _original_print(f"  HW encoder: {time.perf_counter() - t_enc:.2f}s")
    image_emb_t = engine.dma_from_accelerator_memory(
        tensor_addrs["neck_out"], (GA, DEC_DIM)).bfloat16()

    # HW decoder
    t_dec = time.perf_counter()
    masks_hw, iou_hw = engine.run_decoder(
        dec_prog, tokens_t, image_emb_t, image_pe_t, dense_t, tensor_addrs)
    _original_print(f"  HW decoder: {time.perf_counter() - t_dec:.3f}s")
    best = iou_hw.argmax().item()
    _original_print(f"  HW  IOU: {[round(x,4) for x in iou_hw.tolist()]}  best={best}")

    # Save overlay images
    def _save_overlay(mask_256, path, color):
        from PIL import ImageDraw
        mask_1024 = np.array(_PIL_Image.fromarray(mask_256).resize((1024, 1024), _PIL_Image.NEAREST))
        img_np = image_t[0].float().permute(1, 2, 0).numpy()
        img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6) * 255).astype(np.uint8)
        overlay = img_np.copy().astype(np.float32)
        overlay[mask_1024] = overlay[mask_1024] * 0.4 + np.array(color, dtype=np.float32) * 0.6
        overlay = overlay.astype(np.uint8)
        for dy in range(-6, 7):
            for dx in range(-6, 7):
                if abs(dy) + abs(dx) <= 6:
                    ry, rx = np.clip(py + dy, 0, 1023), np.clip(px + dx, 0, 1023)
                    overlay[ry, rx] = [255, 50, 50]
        _PIL_Image.fromarray(overlay).save(path)
        _original_print(f"  Saved {path}")

    _save_overlay((masks_cpu[best_cpu] > _MASK_THRESHOLD).cpu().numpy(),
                   "mask_point_cpu.png", [0, 200, 255])
    _save_overlay((masks_hw[best] > _MASK_THRESHOLD).cpu().numpy(),
                   "mask_point.png", [0, 255, 100])


if __name__ == "__main__":
    main()
