#!/usr/bin/env python3
"""MobileSAM inference from pre-compiled bin files.

Compile first (auto-skipped if already compiled):
    python mobilesam_test.py

Then run:
    python mobilesam_run_from_bin.py --point 512 512
"""
import argparse
import json
import math
import os
import sys
import threading
import time

import numpy as np
import torch
from PIL import Image as _PIL_Image

_original_print = print

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, SCRIPT_DIR)

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


def _amg_encode_point(coord_xy: torch.Tensor, pw: dict) -> torch.Tensor:
    """Build (2, 256) bf16 tokens for one foreground point."""
    pts = coord_xy.reshape(1, 1, 2).float() + 0.5
    lbs = torch.ones(1, 1, dtype=torch.long)
    pad_pt = torch.zeros(1, 1, 2)
    pad_lb = -torch.ones(1, 1, dtype=torch.long)
    pts = torch.cat([pts, pad_pt], dim=1)
    lbs = torch.cat([lbs, pad_lb], dim=1)
    coords = pts.clone()
    coords[:, :, 0] /= 1024.0
    coords[:, :, 1] /= 1024.0
    coords = 2 * coords - 1
    pe = coords @ pw["gauss_matrix"]
    pe = 2 * math.pi * pe
    point_embedding = torch.cat([torch.sin(pe), torch.cos(pe)], dim=-1)
    nap = (lbs == -1)[0]
    fg  = (lbs == 1)[0]
    point_embedding[:, nap, :] = 0.0
    point_embedding[:, nap, :] += pw["not_a_point"]
    point_embedding[:, fg, :]  += pw["point_1"]
    return point_embedding[0].bfloat16()


def _assemble_tokens(checkpoint_path: str, sparse_t: torch.Tensor) -> torch.Tensor:
    """Build (NT_PAD, 256) padded token tensor from iou+mask weights + sparse prompt."""
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    iou_tok   = sd["mask_decoder.iou_token.weight"].to(torch.bfloat16)
    mask_toks = sd["mask_decoder.mask_tokens.weight"].to(torch.bfloat16)
    tokens = torch.zeros(NT_PAD, DEC_DIM, dtype=torch.bfloat16)
    tokens[0]   = iou_tok[0]
    tokens[1:5] = mask_toks
    tokens[5:7] = sparse_t.to(torch.bfloat16).reshape(2, DEC_DIM)
    return tokens


def main():
    parser = argparse.ArgumentParser(description="MobileSAM inference from pre-compiled bins")
    parser.add_argument("--point", type=int, nargs=2, metavar=("X", "Y"), default=[512, 512],
                        help="Single-point inference at (x, y) (default: 512 512)")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--dev", type=str, default="xdma0")
    parser.add_argument("--cycle", type=float, default=None)
    args = parser.parse_args()

    missing = [f for f in ("params.bin", "params.json", "programs.bin", "programs.json")
               if not os.path.exists(os.path.join(BIN_DIR, f))]
    if missing:
        _original_print("Missing bin files (run python mobilesam_test.py first):")
        for f in missing:
            _original_print(f"  {os.path.join(BIN_DIR, f)}")
        return

    # Download checkpoint if needed (for prompt encoding — CPU-side only)
    ckpt_path = os.path.join(BIN_DIR, "mobile_sam.pt")
    if not os.path.exists(ckpt_path):
        import urllib.request
        _original_print("  Downloading mobile_sam.pt …")
        os.makedirs(BIN_DIR, exist_ok=True)
        urllib.request.urlretrieve(
            "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
            ckpt_path,
        )

    set_dma_device(args.dev)
    # Refresh local bindings shadowed at import time so DMA goes to the right device
    import sys as _sys, user_dma_core as _udc
    _mod = _sys.modules[__name__]
    _mod.DMA_DEVICE_H2C = _udc.DMA_DEVICE_H2C
    _mod.DMA_DEVICE_C2H = _udc.DMA_DEVICE_C2H
    _mod.DMA_DEVICE_USER = _udc.DMA_DEVICE_USER
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

    set_dma_device(args.dev)
    if args.cycle is not None:
        user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle

    _original_print(f"MobileSAM on {args.dev} (from pre-compiled bins)")

    engine = MobileSAM_UE_Run(clock_period_ns=args.cycle)
    engine.load_params()

    with open(os.path.join(BIN_DIR, "params.json")) as f:
        param_meta = json.load(f)

    tensor_addrs = {}
    max_end = 0
    for name, ofs in param_meta["tensors"].items():
        addr = engine._tensor_dram_base + ofs
        tensor_addrs[name] = addr
        end = addr + param_meta["tensor_sizes"][name]
        if end > max_end:
            max_end = end
    if max_end > engine._tensor_dram_addr:
        engine._tensor_dram_addr = max_end
    engine.pe_in_dram = tensor_addrs["pe_in"]

    progs = engine.load_programs()
    if progs is None:
        _original_print("Failed to load programs.")
        return
    enc_prog = progs["encoder"]
    dec_prog = progs["decoder"]

    # Build prompt inputs from raw checkpoint weights (no full model needed)
    _sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    _gauss = _sd["prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"]
    dense_t = _sd["prompt_encoder.no_mask_embed.weight"].reshape(1, DEC_DIM).expand(GA, -1).bfloat16().contiguous()
    _tmp = torch.ones(IMG_H, IMG_W)
    _gy = (_tmp.cumsum(dim=0) - 0.5) / IMG_H
    _gx = (_tmp.cumsum(dim=1) - 0.5) / IMG_W
    _coords = 2 * torch.stack([_gx, _gy], dim=-1) - 1
    _pe_raw = 2 * math.pi * (_coords @ _gauss)
    _pe = torch.cat([torch.sin(_pe_raw), torch.cos(_pe_raw)], dim=-1)
    image_pe_t = _pe.permute(2, 0, 1)[None][0].permute(1, 2, 0).reshape(GA, DEC_DIM).bfloat16()

    px, py = args.point
    _original_print(f"  Point: ({px}, {py})")

    pw = {
        "gauss_matrix": _gauss,
        "not_a_point": _sd["prompt_encoder.not_a_point_embed.weight"],
        "point_1": _sd["prompt_encoder.point_embeddings.1.weight"],
    }
    coord = torch.tensor([[float(px), float(py)]])
    sparse_tok = _amg_encode_point(coord, pw)
    tokens_t = _assemble_tokens(ckpt_path, sparse_tok)
    del _sd

    def _progress_timer(label, start_time, stop_event):
        while not stop_event.wait(1.0):
            _original_print(f"\r  {label} ({time.perf_counter() - start_time:.0f}s)", end="", flush=True)

    t_enc = time.perf_counter()
    stop = threading.Event()
    timer = threading.Thread(target=_progress_timer, args=("Encoder running…", t_enc, stop), daemon=True)
    timer.start()
    engine.execute_encoder(enc_prog, image_t)
    stop.set(); timer.join()
    _original_print(f"\r  HW encoder: {time.perf_counter() - t_enc:.2f}s")
    image_emb_t = engine.dma_from_accelerator_memory(tensor_addrs["neck_out"], (GA, DEC_DIM)).bfloat16()

    t_dec = time.perf_counter()
    stop = threading.Event()
    timer = threading.Thread(target=_progress_timer, args=("Decoder running…", t_dec, stop), daemon=True)
    timer.start()
    masks_hw, iou_hw = engine.run_decoder(dec_prog, tokens_t, image_emb_t, image_pe_t, dense_t, tensor_addrs)
    stop.set(); timer.join()
    _original_print(f"\r  HW decoder: {time.perf_counter() - t_dec:.3f}s")
    best = iou_hw.argmax().item()
    _original_print(f"  HW  IOU: {[round(x,4) for x in iou_hw.tolist()]}  best={best}")

    def _save_overlay(mask_256, path, color):
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

    out_path = os.path.join(SCRIPT_DIR, "mask_point.png")
    _save_overlay((masks_hw[best] > _MASK_THRESHOLD).cpu().numpy(), out_path, [0, 255, 100])


if __name__ == "__main__":
    main()
