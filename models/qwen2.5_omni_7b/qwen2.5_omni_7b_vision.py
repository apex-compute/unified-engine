#!/usr/bin/env python3
"""Qwen2.5-Omni vision adapter over the Qwen2.5-VL accelerator tower."""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from user_dma_core import DMA_DEVICE_H2C, TYPE


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VL_DIR = os.path.join(os.path.dirname(_THIS_DIR), "qwen2.5_vl_3b")


def _load_vl_vision():
    name = "qwen2_5_vl_3b_vision_for_omni"
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(_VL_DIR, "qwen2.5_vl_3b_vision.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Qwen2.5-VL vision mixin from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_vl_vision = _load_vl_vision()


class Qwen25OmniVisionMixin(_vl_vision.Qwen25VLVisionMixin):
    """Omni's 1280-wide NaViT tower with a width-3584 patch merger."""

    def _read_vision_region(self) -> dict:
        return self._read_params_region("vision")

    def vision_weight_init(self) -> None:
        """Load the complete vision tower, including the FPGA patch matrix."""
        if (
            getattr(self, "_vision_weight_init_done", False)
            and hasattr(self, "patch_bf16_weight")
        ):
            return
        region = self._read_vision_region()
        key = "visual.patch_embed.proj.weight.bf16"
        section = region["sections"].get(key)
        if section is None:
            raise KeyError(
                f"vision region has no {key!r}; regenerate params.bin with the "
                "current Omni exporter"
            )
        d = self._vision_dims()
        patch_k = (
            3
            * int(self._cfg["vision"]["temporal_patch_size"])
            * d["PATCH_SIZE"] ** 2
        )
        patch_k_padded = ((patch_k + 63) // 64) * 64
        disk_shape = tuple(int(value) for value in section.get("shape", ()))
        if disk_shape != (d["VH"], patch_k_padded):
            raise ValueError(
                f"{key} shape {disk_shape} != ({d['VH']}, {patch_k_padded}); "
                "regenerate params.bin so the FPGA patch projection has a "
                "64-lane padded reduction axis"
            )
        self._invalidate_decode_overlay()
        super().vision_weight_init()
        try:
            with open(region["bin_path"], "rb") as file_obj:
                self.patch_bf16_weight = self._dma_bf16(
                    file_obj,
                    section,
                    int(region["base_offset"]),
                    "patch_embed.bf16",
                )
            self._vis_weight_end = self.get_params_dram_addr()
            if self._vis_weight_end > self.PARAMS_LIMIT:
                raise MemoryError(
                    "vision weights including the BF16 patch projection overflow "
                    f"params DRAM: 0x{self._vis_weight_end:X} > "
                    f"0x{self.PARAMS_LIMIT:X}"
                )
        except Exception:
            self._vision_weight_init_done = False
            raise
        self._lm_weight_init_done = False
        self._audio_weight_init_done = False

    @staticmethod
    def _position_ids(grid_thw: torch.Tensor, merge: int) -> torch.Tensor:
        positions = []
        for t, h, w in torch.as_tensor(grid_thw, dtype=torch.long).tolist():
            if h % merge or w % merge:
                raise ValueError(
                    f"vision grid {(t, h, w)} must be divisible by merge size {merge}")
            hp = torch.arange(h).unsqueeze(1).expand(-1, w)
            hp = hp.reshape(h // merge, merge, w // merge, merge)
            hp = hp.permute(0, 2, 1, 3).flatten()
            wp = torch.arange(w).unsqueeze(0).expand(h, -1)
            wp = wp.reshape(h // merge, merge, w // merge, merge)
            wp = wp.permute(0, 2, 1, 3).flatten()
            positions.append(torch.stack((hp, wp), dim=-1).repeat(t, 1))
        return torch.cat(positions, dim=0)

    @staticmethod
    def _window_index(
        grid_thw: torch.Tensor, merge: int, window_size: int, patch_size: int
    ) -> tuple[torch.Tensor, list[int]]:
        indices: list[torch.Tensor] = []
        cumulative = [0]
        index_base = 0
        merger_window = window_size // merge // patch_size
        if merger_window <= 0:
            raise ValueError("vision window is smaller than one merged patch")
        for grid_t, grid_h, grid_w in torch.as_tensor(grid_thw, dtype=torch.long).tolist():
            mh, mw = grid_h // merge, grid_w // merge
            index = torch.arange(grid_t * mh * mw).reshape(grid_t, mh, mw)
            # This deliberately mirrors the upstream implementation, including
            # one all-padding window when a side divides evenly.
            pad_h = merger_window - mh % merger_window
            pad_w = merger_window - mw % merger_window
            nw_h = (mh + pad_h) // merger_window
            nw_w = (mw + pad_w) // merger_window
            padded = F.pad(index, (0, pad_w, 0, pad_h), value=-100)
            padded = padded.reshape(
                grid_t, nw_h, merger_window, nw_w, merger_window
            ).permute(0, 1, 3, 2, 4)
            lengths = (padded != -100).sum((3, 4)).reshape(-1)
            flattened = padded.reshape(-1)
            valid = flattened[flattened != -100]
            indices.append(valid + index_base)
            cumulative.extend(
                (lengths.cumsum(0) * merge * merge + cumulative[-1]).tolist()
            )
            index_base += grid_t * mh * mw
        return torch.cat(indices), cumulative

    def prepare_encoder_input(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor = None
    ) -> dict:
        """Prepare/reorder pixels; the learned patch projection runs on U55."""
        d = self._vision_dims()
        if image_grid_thw is None:
            pixel_values, image_grid_thw = self._hf_preprocess_image(pixel_values)
        pixels = torch.as_tensor(pixel_values).detach().cpu()
        if pixels.ndim > 2:
            pixels = pixels.reshape(pixels.shape[0], -1)
        k = 3 * self._cfg["vision"]["temporal_patch_size"] * d["PATCH_SIZE"] ** 2
        if pixels.ndim != 2 or pixels.shape[1] != k:
            raise ValueError(
                f"processed image patches must have shape [patches, {k}], "
                f"got {tuple(pixels.shape)}")
        if pixels.shape[0] != d["VS"]:
            raise ValueError(
                f"pixel patch count {pixels.shape[0]} != {d['VS']}; only the "
                "canonical 336x336 program is supported"
            )

        grid = torch.as_tensor(image_grid_thw, dtype=torch.long)
        pos_ids = self._position_ids(grid, d["VMERGE"])
        rotary_dim = d["VD"] // 2
        inv = 1.0 / (
            float(self._cfg["vision"].get("rope_theta", 10000.0))
            ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        rotary = (pos_ids.float().unsqueeze(-1) * inv).flatten(1)
        window_index, cu = self._window_index(
            grid, d["VMERGE"], d["WINDOW_SIZE"], d["PATCH_SIZE"]
        )
        unit = d["VMERGE"] ** 2
        pixels = pixels.to(torch.bfloat16).contiguous()
        pixels = pixels.reshape(d["VS"] // unit, unit, k)[window_index]
        pixels = pixels.reshape(d["VS"], k)
        k_padded = ((k + 63) // 64) * 64
        patch_input = torch.zeros(d["VS"], k_padded, dtype=torch.bfloat16)
        patch_input[:, :k] = pixels
        rotary = rotary.reshape(d["VS"] // unit, unit, -1)[window_index]
        rotary = rotary.reshape(d["VS"], -1)

        self._vis_reverse_index = torch.argsort(window_index)
        self._cu_window_seqlens = torch.tensor(cu).unique_consecutive().tolist()
        self._vis_patch_input = patch_input.contiguous()
        self._vis_patch_k = k_padded
        self._vis_rotary_pos_emb = rotary
        self._image_grid_thw = grid
        self._vis_hf_pixels = pixels
        self._loud(
            f"    FPGA patch projection [{d['VS']}, {k_padded}] -> "
            f"[{d['VS']}, {d['VH']}], "
            f"{len(self._cu_window_seqlens) - 1} attention windows")
        return {
            "patch_input": patch_input,
            "rotary_pos_emb": rotary,
            "image_grid_thw": grid,
        }

    def vision_tensor_init(self) -> None:
        """Allocate the raw-patch input while reusing the inherited ViT arena."""
        if not hasattr(self, "_vis_patch_input"):
            raise RuntimeError("prepare_encoder_input() must run first")
        d = self._vision_dims()
        # The shared implementation allocates every transformer buffer and
        # ordinarily queues host-created patch embeddings into VIS_IO_A. Give
        # it a shape-only placeholder, then replace that DMA with raw pixels.
        self._vis_patch_embeds = torch.zeros(
            d["VS"], d["VH"], dtype=torch.bfloat16
        )
        super().vision_tensor_init()
        self._vis_pending_dmas = [
            item for item in self._vis_pending_dmas if item[0] != self.VIS_IO_A
        ]
        self.VIS_PATCH_INPUT = self.allocate_tensor_dram(
            self._vis_patch_input.numel() * self.bytes_per_element,
            label="vis.patch_input",
        )
        self._vis_pending_dmas.append(
            (self.VIS_PATCH_INPUT, self._vis_patch_input.flatten())
        )
        end = self.get_tensor_dram_addr()
        if end > self.TENSOR_LIMIT:
            raise MemoryError(
                f"vision tensors including raw patches overflow tensor DRAM: "
                f"0x{end:X} > 0x{self.TENSOR_LIMIT:X}"
            )

    def compile_vision_encoder(self, profile: bool = False) -> int:
        """Compile the ViT plus a U55-only BF16 patch-projection program."""
        encoder_addr = super().compile_vision_encoder(profile=profile)
        encoder_blob = self._vis_program_bytes
        patch_addr = encoder_addr + len(encoder_blob)
        reg_before = self._isa_reg_counter
        ptr_before = self._inst_ptr_counter
        self._next_program_dram_addr = patch_addr
        self.clear_inst_id()
        self.clear_capture_buffer()
        self.start_capture()
        previous_silent = self._set_silent(True)
        try:
            rows_reg = self.alloc_isa_reg()
            self.generate_instruction_add_set(rows_reg, self._vision_dims()["VS"])
            patch_flops = self.matmat_mul_core(
                M=self._vision_dims()["VS"],
                K=self._vis_patch_k,
                N=self._vision_dims()["VH"],
                A_DRAM_ADDR=self.VIS_PATCH_INPUT,
                B_DRAM_ADDR=self.patch_bf16_weight,
                OUTPUT_DRAM_ADDR=self.VIS_IO_A,
                is_B_quantized=False,
                gpr_M_reg=rows_reg,
            ) or 0
            self.release_isa_reg()
            self.generate_instruction_halt()
            self.stop_capture()
            patch_blob = b"".join(
                instruction.get_bytes() for instruction in self.capture_buffer
            )
        except Exception:
            if getattr(self, "is_capture_on", False):
                self.stop_capture()
            self.clear_capture_buffer()
            self._isa_reg_counter = reg_before
            self._inst_ptr_counter = ptr_before
            self._vis_program_bytes = encoder_blob
            raise
        finally:
            self._set_silent(previous_silent)
        self.clear_capture_buffer()
        if not patch_blob:
            raise RuntimeError("vision patch projection capture produced no ISA")
        self._vis_encoder_program_bytes = encoder_blob
        self._vis_patch_program_addr = patch_addr
        self._vis_patch_program_bytes = patch_blob
        # One contiguous upload keeps ISA accounting and bounds checks exact;
        # the encoder HALT prevents fall-through into the appended patch image.
        self._vis_program_bytes = encoder_blob + patch_blob
        self._vis_total_flops += int(patch_flops)
        if encoder_addr + len(self._vis_program_bytes) > self.MASTER_ISA_LIMIT:
            raise MemoryError(
                "vision encoder plus patch projection exceeds the master ISA "
                f"reserve at 0x{self.MASTER_ISA_LIMIT:X}"
            )
        self._loud(
            f"  [Vision] FPGA patch projection compiled: "
            f"{len(patch_blob) / 2**20:.2f} MiB at 0x{patch_addr:X}"
        )
        return encoder_addr

    def run_vision_encoder(
        self, timeout_s: float = 600.0, profile: bool = False
    ) -> torch.Tensor:
        """Run patch projection on U55, then launch the eight-engine ViT."""
        if not hasattr(self, "_vis_patch_program_addr"):
            raise RuntimeError("compile_vision_encoder() must run first")
        pending = list(self._vis_pending_dmas)
        for address, tensor in pending:
            self.dma_to_accelerator_memory(address, tensor.contiguous())
        composite = self._vis_program_bytes
        written = self.dma_write(
            DMA_DEVICE_H2C,
            self._vis_program_addr,
            composite,
            len(composite),
        )
        if written != len(composite):
            raise IOError(
                f"vision combined ISA DMA wrote {written} of {len(composite)} bytes"
            )
        started = time.perf_counter()
        self.start_execute_from_dram(self._vis_patch_program_addr)
        self.wait_queue(float(timeout_s))
        if self.is_queue_busy():
            raise TimeoutError(
                f"vision patch projection is still busy after {timeout_s:.1f}s"
            )
        patch_latency_us = float(self.report_latency_in_us())
        patch_wall_s = time.perf_counter() - started

        # Inputs are already resident, and VIS_IO_A now contains the device
        # projection. Keep pristine host inputs for retry only.
        self._vis_pending_dmas = []
        self._vis_program_bytes = self._vis_encoder_program_bytes
        try:
            output = super().run_vision_encoder(timeout_s=timeout_s, profile=profile)
        except Exception:
            self._vis_pending_dmas = pending
            raise
        finally:
            self._vis_program_bytes = composite
        self._vis_latency_us += patch_latency_us
        self._vis_wall_s += patch_wall_s
        self._vis_gflops = (
            self._vis_total_flops / (self._vis_latency_us * 1e-6) / 1e9
            if self._vis_latency_us > 0
            else 0.0
        )
        return output


VISION_QUANT_PRECISION = _vl_vision.VISION_QUANT_PRECISION

__all__ = ["Qwen25OmniVisionMixin", "VISION_QUANT_PRECISION"]
