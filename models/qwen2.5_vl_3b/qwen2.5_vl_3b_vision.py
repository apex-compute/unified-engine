#!/usr/bin/env python3
"""Qwen2.5-VL-3B vision-encoder method group (ViT tower with window attention).

``Qwen25VLVisionMixin`` carries the vision methods and is mixed into
``Qwen25VL_UnifiedEngine`` in qwen2.5_vl_3b_test.py; it is never
instantiated on its own. Everything shared -- the config on ``self._cfg``, the
DRAM allocators, ``self._loud`` -- resolves through the concrete class, so this
module imports nothing from the test module (keeps the split cycle-free).
This mirrors the gemma4_e2b_vision.py arrangement.

WEIGHTS ARE TRANSIENT. The encoder runs exactly once, before LM prefill, so its
weights and the LM's occupy the SAME params window at 0x8000_0000: vision loads
first, the encoder runs, its 144 output embeddings are copied out, and the LM
weights are then loaded over the top. Only the encoder output survives the
handover -- see the DRAM map in qwen2.5_vl_3b_test.py.
"""
import json
import math
import os
import sys
import time

_SD = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SD)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SD)))

import numpy as np
import torch

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, INSTRUCTION_SIZE_BYTES, TYPE, UE_MODE, UE_VECTOR_SIZE,
    URAM_NEAR_FULL_ELEMENTS)

# Vision matmuls are IF4-quantized; norms and biases stay BF16. Module-level so
# a numeric harness can import it without constructing an engine.
VISION_QUANT_PRECISION = "if4"

# IF4 on-disk block layout: 64 elements per block, stored as a 2-byte BF16 scale
# followed by 32 bytes of packed nibbles. The two halves are split into separate
# DRAM buffers at load time because the engine takes scale and data as distinct
# operands (SCALE_DRAM_ADDR / B_DRAM_ADDR).
IF4_BLOCK_ELEMS = 64
IF4_SCALE_BYTES = 2
IF4_DATA_BYTES = 32
IF4_BLOCK_BYTES = IF4_SCALE_BYTES + IF4_DATA_BYTES     # 34


class Qwen25VLVisionMixin:
    """Vision-encoder methods for Qwen25VL_UnifiedEngine (see module docstring)."""

    # ---- weight loading ---------------------------------------------------

    def _vision_dims(self) -> dict:
        """Vision geometry from the config, with the on-chip padding applied.

        head_dim 80 is not a multiple of the 64-element quantization block, and
        the attention core needs a 64-aligned head stride, so weights are stored
        pre-padded to 128. Likewise the FFN 3420 -> 3456. The padding is baked
        into params.bin by the weight generator; these numbers only have to
        agree with it.
        """
        v = self._cfg["vision"]
        VD_PAD = 128
        VI_PAD = ((v["intermediate_size"] + UE_VECTOR_SIZE - 1)
                  // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
        return dict(
            VS=v["num_patches"], VH=v["hidden_size"], VN=v["num_heads"],
            VD=v["head_dim"], VD_PAD=VD_PAD,
            VI=v["intermediate_size"], VI_PAD=VI_PAD,
            VL=v["depth"], VMERGE=v["spatial_merge_size"],
            VH_OUT=v["out_hidden_size"],
            NUM_MERGED_TOKENS=v["num_merged_tokens"],
            FULL_ATTN_LAYERS=set(v["fullatt_block_indexes"]),
            WINDOW_SIZE=v["window_size"], PATCH_SIZE=v["patch_size"],
        )

    def _read_vision_region(self) -> dict:
        """Slice the vision region out of params.bin and return its manifest.

        params.bin is [ lm | vision ]; params.json carries a ``regions`` map with
        each region's absolute file offset plus a sub-manifest of region-relative
        tensor offsets. Only the vision region is read here -- the LM region is
        2.4 GB and this phase never touches it.
        """
        bin_path = os.path.join(self.script_dir, self._cfg["paths"]["params"])
        json_path = bin_path.rsplit(".", 1)[0] + ".json"
        if not (os.path.exists(bin_path) and os.path.exists(json_path)):
            raise FileNotFoundError(
                f"weight bin not found: {bin_path}. Generate it with the sibling "
                f"models/qwen2.5_vl_3b build, or point paths.params elsewhere.")
        with open(json_path) as f:
            manifest = json.load(f)
        regions = manifest.get("regions") or {}
        if "vision" not in regions:
            raise KeyError(
                f"no 'vision' region in {json_path} (regions={list(regions)}). "
                f"The bin predates the [lm | vision] layout; regenerate it.")
        r = regions["vision"]
        return dict(bin_path=bin_path, base_offset=int(r["offset"]),
                    size=int(r["size"]), sections=r["manifest"])

    def _dma_bf16(self, f, section: dict, base_offset: int, what: str) -> int:
        """Copy one BF16 tensor from the bin straight into params DRAM."""
        f.seek(base_offset + section["offset"])
        blob = f.read(section["size"])
        if len(blob) != section["size"]:
            raise RuntimeError(f"truncated read for {what}")
        addr = self.allocate_params_dram(section["size"], label=what)
        self.dma_write(DMA_DEVICE_H2C, addr, blob, section["size"])
        return addr

    def _dma_if4(self, f, section: dict, base_offset: int, what: str) -> tuple[int, int]:
        """Split one IF4 tensor into (scale, data) DRAM buffers.

        The bin stores all scales first, then all packed data -- the same layout
        store_quantized_weight() used in the original build -- so the split is a
        single slice, not a de-interleave.
        """
        f.seek(base_offset + section["offset"])
        blob = f.read(section["size"])
        if len(blob) != section["size"]:
            raise RuntimeError(f"truncated read for {what}")
        if len(blob) % IF4_BLOCK_BYTES:
            raise ValueError(
                f"{what}: {len(blob)} bytes is not a multiple of the "
                f"{IF4_BLOCK_BYTES}-byte IF4 block")
        n_blocks = len(blob) // IF4_BLOCK_BYTES
        scale_bytes = n_blocks * IF4_SCALE_BYTES
        data_bytes = n_blocks * IF4_DATA_BYTES

        scale_addr = self.allocate_params_dram(scale_bytes, label=f"{what}.scale")
        self.dma_write(DMA_DEVICE_H2C, scale_addr, blob[:scale_bytes], scale_bytes)
        data_addr = self.allocate_params_dram(data_bytes, label=f"{what}.data")
        self.dma_write(DMA_DEVICE_H2C, data_addr,
                       blob[scale_bytes:scale_bytes + data_bytes], data_bytes)
        return scale_addr, data_addr

    def vision_weight_init(self) -> None:
        """Load the vision encoder's weights into the params window at
        VISION_WEIGHT_BASE (0x8000_0000). Idempotent.

        Records every DRAM address on ``self.vis_layer_addrs`` (per layer) plus
        the merger / patch-embed attributes, in the order the encoder program
        will reference them. Nothing is compiled here -- this is the weight
        phase only.
        """
        if getattr(self, "_vision_weight_init_done", False):
            return
        d = self._vision_dims()
        region = self._read_vision_region()
        sections = region["sections"]
        sfx = VISION_QUANT_PRECISION

        # Vision owns the params window from its base. reset_params_dram_addr()
        # rewinds the cursor so the LM can later be loaded over these same
        # addresses (see the module docstring); nothing else may allocate params
        # DRAM between here and the encoder run.
        self.reset_params_dram_addr()
        start_addr = self.get_params_dram_addr()
        if start_addr != self.VISION_WEIGHT_BASE:
            raise AssertionError(
                f"vision weights must start at 0x{self.VISION_WEIGHT_BASE:X}, "
                f"params cursor is at 0x{start_addr:X}")

        self._loud(f"\n[Vision] Loading {d['VL']} encoder layers + merger "
                   f"({sfx.upper()} block=64, BF16 norms/biases) at "
                   f"0x{start_addr:X} ...")

        def need(key: str) -> dict:
            if key not in sections:
                raise KeyError(f"vision weight {key!r} missing from params.bin")
            return sections[key]

        with open(region["bin_path"], "rb") as f:
            base = region["base_offset"]
            layer_addrs = []
            for li in range(d["VL"]):
                pre = f"visual.blocks.{li}"
                la = {}
                # Fused QK (pre-padded 80 -> 128 per head) and V, then o_proj.
                for tag, key in (("qk", "attn.qk_padded"), ("v", "attn.v_padded"),
                                 ("o", "attn.proj")):
                    la[f"{tag}_scale"], la[f"{tag}_data"] = self._dma_if4(
                        f, need(f"{pre}.{key}.weight.{sfx}"), base, f"{pre}.{key}")
                    la[f"{tag}_bias"] = self._dma_bf16(
                        f, need(f"{pre}.{key}.bias"), base, f"{pre}.{key}.bias")
                # SwiGLU MLP (pre-padded 3420 -> 3456), all three with bias.
                for tag in ("gate", "up", "down"):
                    key = f"mlp.{tag}_proj"
                    la[f"{tag}_scale"], la[f"{tag}_data"] = self._dma_if4(
                        f, need(f"{pre}.{key}.weight.{sfx}"), base, f"{pre}.{key}")
                    la[f"{tag}_bias"] = self._dma_bf16(
                        f, need(f"{pre}.{key}.bias"), base, f"{pre}.{key}.bias")
                # RMSNorm gammas (no bias).
                la["norm1_weight"] = self._dma_bf16(
                    f, need(f"{pre}.norm1.weight"), base, f"{pre}.norm1")
                la["norm2_weight"] = self._dma_bf16(
                    f, need(f"{pre}.norm2.weight"), base, f"{pre}.norm2")
                layer_addrs.append(la)
                if (li + 1) % 8 == 0 or li == d["VL"] - 1:
                    self._loud(f"    layer {li + 1}/{d['VL']} loaded")
            self.vis_layer_addrs = layer_addrs

            # Patch embed (Conv3d folded to a matmul) — quantized like the rest.
            self.patch_weight_scale, self.patch_weight_data = self._dma_if4(
                f, need(f"visual.patch_embed.proj.weight.{sfx}"), base, "patch_embed")

            # Merger: RMSNorm(ln_q) then 5120 -> 5120 (GELU) -> 2048.
            self.merger_ln_q_weight = self._dma_bf16(
                f, need("visual.merger.ln_q.weight"), base, "merger.ln_q")
            self.merger_mlp0_scale, self.merger_mlp0_data = self._dma_if4(
                f, need(f"visual.merger.mlp.0.weight.{sfx}"), base, "merger.mlp0")
            self.merger_mlp0_bias = self._dma_bf16(
                f, need("visual.merger.mlp.0.bias"), base, "merger.mlp0.bias")
            self.merger_mlp2_scale, self.merger_mlp2_data = self._dma_if4(
                f, need(f"visual.merger.mlp.2.weight.{sfx}"), base, "merger.mlp2")
            self.merger_mlp2_bias = self._dma_bf16(
                f, need("visual.merger.mlp.2.bias"), base, "merger.mlp2.bias")

        # unified_attention_core needs a UE_VECTOR_SIZE^2 BF16 identity for its
        # V-transpose. It is an input, not scratch, so it belongs with the
        # weights in the same transient window.
        self._vis_identity_dram = self.allocate_params_dram(
            UE_VECTOR_SIZE * UE_VECTOR_SIZE * self.bytes_per_element,
            label="vision.identity")
        self.dma_to_accelerator_memory(
            self._vis_identity_dram,
            torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))

        self._vis_weight_start = start_addr
        self._vis_weight_end = self.get_params_dram_addr()
        used = self._vis_weight_end - start_addr
        if self._vis_weight_end > self.PARAMS_LIMIT:
            raise MemoryError(
                f"vision weights overflow the params window: end "
                f"0x{self._vis_weight_end:X} > limit 0x{self.PARAMS_LIMIT:X}")
        self._vision_weight_init_done = True
        self._loud(f"  Vision weights loaded: {used / 2**20:.1f} MiB, "
                   f"0x{start_addr:X}..0x{self._vis_weight_end:X} "
                   f"({100 * used / (self.PARAMS_LIMIT - start_addr):.1f}% of the "
                   f"params window)")

    def vision_weight_summary(self) -> str:
        """One-line-per-group breakdown of what vision_weight_init() placed."""
        if not getattr(self, "_vision_weight_init_done", False):
            return "  (vision weights not loaded)"
        d = self._vision_dims()
        sections = self._read_vision_region()["sections"]
        groups = {
            "qk (fused, padded)": ".attn.qk_padded.weight",
            "v (padded)":         ".attn.v_padded.weight",
            "o_proj":             ".attn.proj.weight",
            "gate_proj":          ".mlp.gate_proj.weight",
            "up_proj":            ".mlp.up_proj.weight",
            "down_proj":          ".mlp.down_proj.weight",
        }
        lines = [f"  vision weights @ 0x{self._vis_weight_start:X} "
                 f"({d['VL']} layers + merger):"]
        for label, suffix in groups.items():
            total = sum(s["size"] for k, s in sections.items()
                        if suffix in k and k.startswith("visual.blocks."))
            lines.append(f"    {label:<20} {total / 2**20:7.1f} MiB  IF4")
        merger = sum(s["size"] for k, s in sections.items()
                     if k.startswith("visual.merger."))
        patch = sections[f"visual.patch_embed.proj.weight.{VISION_QUANT_PRECISION}"]["size"]
        lines.append(f"    {'merger MLP':<20} {merger / 2**20:7.1f} MiB  IF4")
        lines.append(f"    {'patch_embed':<20} {patch / 2**20:7.1f} MiB  IF4")
        lines.append(f"    {'TOTAL':<20} "
                     f"{(self._vis_weight_end - self._vis_weight_start) / 2**20:7.1f} MiB")
        return "\n".join(lines)

    # ---- host-side preprocessing ------------------------------------------

    def prepare_encoder_input(self, pixel_values: torch.Tensor,
                              image_grid_thw: torch.Tensor = None) -> dict:
        """Run the host half of the encoder: patch embed, window reorder, RoPE.

        WHY PATCH EMBED STAYS ON HOST. It is a Conv3d over raw pixels with
        K = 3*2*14*14 = 1176, which is not a multiple of the 64-element IF4
        block, so it does not lower cleanly to the quantized matmul path. It is
        also only 1.5 M params / 1.7 GFLOP against the encoder's 748 GFLOP, and
        the window reordering it feeds is host-side regardless (it depends on
        image_grid_thw). The IF4 patch weight is still loaded to DRAM by
        vision_weight_init so an FPGA patch embed can be dropped in later.
        """
        vis = self._vision_dims()
        VS, VH = vis["VS"], vis["VH"]
        model_dir = os.path.join(self.script_dir, self._cfg["paths"]["hf_model_dir"])

        if not hasattr(self, "_hf_model"):
            self._loud("    loading HF model for patch embedding ...")
            from transformers import Qwen2_5_VLForConditionalGeneration
            m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_dir, torch_dtype=torch.bfloat16)
            if not hasattr(m, "visual") and hasattr(m, "model"):
                m.visual = m.model.visual
            m.eval()
            self._hf_model = m
        visual = self._hf_model.visual

        with torch.no_grad():
            if image_grid_thw is None:
                pixel_values, image_grid_thw = self._hf_preprocess_image(pixel_values)
            self._vis_hf_pixels = pixel_values
            patch_embeds = visual.patch_embed(pixel_values.to(torch.bfloat16))
            if patch_embeds.shape != (VS, VH):
                raise ValueError(
                    f"patch embed shape {tuple(patch_embeds.shape)} != ({VS}, {VH}); "
                    f"the encoder program is compiled for a fixed {VS}-patch input")
            rotary_pos_emb = visual.rot_pos_emb(image_grid_thw)
            window_index, cu_window_seqlens = visual.get_window_index(image_grid_thw)

            # Reorder patches so each attention window is a contiguous slice.
            # The reorder works on 2x2 merge units, which is also why the merger
            # later needs no gather: 4 consecutive rows are one merged token.
            unit = visual.spatial_merge_size ** 2
            patch_embeds = patch_embeds.reshape(VS // unit, unit, -1)[window_index]
            patch_embeds = patch_embeds.reshape(VS, -1)
            rotary_pos_emb = rotary_pos_emb.reshape(VS // unit, unit, -1)[window_index]
            rotary_pos_emb = rotary_pos_emb.reshape(VS, -1)
            self._vis_reverse_index = torch.argsort(window_index)

            if isinstance(cu_window_seqlens, torch.Tensor):
                cu = cu_window_seqlens.unique_consecutive().tolist()
            else:
                cu = [cu_window_seqlens[0]] + [
                    c for i, c in enumerate(cu_window_seqlens[1:], 1)
                    if c != cu_window_seqlens[i - 1]]
            self._cu_window_seqlens = cu

        self._vis_patch_embeds = patch_embeds
        self._vis_rotary_pos_emb = rotary_pos_emb
        self._image_grid_thw = image_grid_thw
        self._loud(f"    patch embed [{VS}, {VH}], "
                   f"{len(cu) - 1} windows {cu[:4]}{'...' if len(cu) > 5 else ''}")
        return dict(patch_embeds=patch_embeds, rotary_pos_emb=rotary_pos_emb,
                    image_grid_thw=image_grid_thw)

    def _hf_preprocess_image(self, pixel_values: torch.Tensor):
        """Normalized CHW tensor -> (HF pixel_values, image_grid_thw)."""
        from transformers import Qwen2VLImageProcessor
        from PIL import Image
        vis_cfg = self._cfg["vision"]
        img_cfg = self._cfg.get("image_processing", {})
        mean = img_cfg.get("normalize_mean", [0.48145466, 0.4578275, 0.40821073])
        std = img_cfg.get("normalize_std", [0.26862954, 0.26130258, 0.27577711])
        arr = pixel_values.float().permute(1, 2, 0).numpy()
        for c in range(3):
            arr[:, :, c] = arr[:, :, c] * std[c] + mean[c]
        img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
        size = vis_cfg.get("image_size", 336)
        img = img.resize((size, size), Image.Resampling.BILINEAR)
        proc = Qwen2VLImageProcessor.from_pretrained(
            os.path.join(self.script_dir, self._cfg["paths"]["hf_model_dir"]),
            trust_remote_code=True)
        out = proc(images=[img], return_tensors="pt")
        return out["pixel_values"].to(torch.bfloat16), out["image_grid_thw"]

    def _build_rope_tables(self) -> torch.Tensor:
        """[VN*VS, 2, VD_PAD] interleaved cos/sin rows for rope_hf_core_dram.

        head_dim 80 is zero-padded to 128 in the layout
        [d0..d39 | pad(24) | d40..d79 | pad(24)], so rotate-half at width 128
        pairs d_i with d_{i+40} exactly as it would at width 80. cos is 1.0 in
        the pad gaps (identity) and sin is 0.0 there; sin's lower half is
        pre-negated so the core's two elementwise passes are add-only.
        Tiled VN times so one M=VN*VS call covers every head -- cos/sin are
        position-only and shared across heads.
        """
        vis = self._vision_dims()
        VS, VN, VD, VD_PAD = vis["VS"], vis["VN"], vis["VD"], vis["VD_PAD"]
        half, quarter = VD // 2, VD_PAD // 2
        with torch.no_grad():
            cos_raw = self._vis_rotary_pos_emb.cos().to(torch.bfloat16)   # [VS, 40]
            sin_raw = self._vis_rotary_pos_emb.sin().to(torch.bfloat16)
            table = torch.empty(VS, 2, VD_PAD, dtype=torch.bfloat16)
            cos_t, sin_t = table[:, 0, :], table[:, 1, :]
            cos_t.fill_(1.0)
            cos_t[:, :half] = cos_raw
            cos_t[:, quarter:quarter + half] = cos_raw
            sin_t.zero_()
            sin_t[:, :half] = -sin_raw
            sin_t[:, quarter:quarter + half] = sin_raw
            return table.repeat(VN, 1, 1).contiguous()

    def _build_attention_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(full, windowed) additive attention biases, both [VS, VS] bf16.

        Window attention is expressed as a BLOCK-DIAGONAL MASK over the full
        sequence rather than as one attention call per (head, window). The two
        are mathematically identical once patches are in window order -- a token
        can only see its own window either way -- but the mask form means every
        layer issues the same 16 calls and differs only in which bias it reads.
        The alternative, 16 heads x 9 windows = 144 calls per windowed layer,
        is what made the original build's encoder image enormous.

        It costs FLOPs: full 576x576 attention on all 32 layers is 54.4 G against
        12.1 G for true windowing, i.e. +42 G on a 748 G encoder (+5.6%).
        """
        vis = self._vision_dims()
        VS = vis["VS"]
        full = torch.zeros(VS, VS, dtype=torch.bfloat16)
        windowed = torch.full((VS, VS), float("-inf"), dtype=torch.bfloat16)
        cu = getattr(self, "_cu_window_seqlens", [0, VS])
        for start, end in zip(cu[:-1], cu[1:]):
            if end > start:
                windowed[start:end, start:end] = 0.0
        return full, windowed

    # ---- tensor allocation -------------------------------------------------

    def vision_tensor_init(self) -> None:
        """Allocate the encoder's intermediates in the tensor region and queue
        the host-built inputs for upload at run time."""
        vis = self._vision_dims()
        VS, VH, VN, VD, VD_PAD = (vis["VS"], vis["VH"], vis["VN"],
                                  vis["VD"], vis["VD_PAD"])
        VI_PAD, VMERGE, VH_OUT = vis["VI_PAD"], vis["VMERGE"], vis["VH_OUT"]
        T = vis["NUM_MERGED_TOKENS"]
        bpe = self.bytes_per_element
        aligned_S = ((VS + 63) // 64) * 64
        self._vis_aligned_S = aligned_S
        self._vis_pending_dmas: list[tuple[int, torch.Tensor]] = []

        def alloc(n_elems, what):
            return self.allocate_tensor_dram(n_elems * bpe, label=what)

        # Layer I/O, double-buffered; the patch embeddings land in IO_A.
        self.VIS_IO_A = alloc(VS * VH, "vis.io_a")
        self.VIS_IO_B = alloc(VS * VH, "vis.io_b")
        self.VIS_NORM_OUT = alloc(VS * VH, "vis.norm_out")

        # Projection outputs. QK is the fused [VS, 2*VN*VD_PAD] result; V is
        # [VS, VN*VD_PAD]. Both are token-major and get permuted to head-major.
        self.VIS_QK = alloc(VS * VN * VD_PAD * 2, "vis.qk")
        self.VIS_V = alloc(VS * VN * VD_PAD, "vis.v")

        # Head-major [n_groups, aligned_S, VD_PAD]. QK_HM holds Q's VN heads
        # followed by K's, so ONE permute produces both.
        hm = aligned_S * VD_PAD
        self.VIS_QK_HM = alloc(2 * VN * hm, "vis.qk_hm")
        self.VIS_Q_HM = self.VIS_QK_HM
        self.VIS_K_HM = self.VIS_QK_HM + VN * hm * bpe
        self.VIS_V_HM = alloc(VN * hm, "vis.v_hm")
        self.VIS_OUT_HM = alloc(VN * hm, "vis.out_hm")
        zeros_hm = torch.zeros(VN * hm, dtype=torch.bfloat16)
        for addr in (self.VIS_Q_HM, self.VIS_K_HM, self.VIS_V_HM, self.VIS_OUT_HM):
            self._vis_pending_dmas.append((addr, zeros_hm))

        # Attention: two static biases (see _build_attention_bias) and scratch.
        self.VIS_BIAS_FULL = alloc(aligned_S * aligned_S, "vis.bias_full")
        self.VIS_BIAS_WINDOW = alloc(aligned_S * aligned_S, "vis.bias_window")
        scratch_elems = (VD_PAD + aligned_S) * aligned_S + aligned_S * VD_PAD
        self.VIS_ATTN_SCRATCH = alloc(scratch_elems, "vis.attn_scratch")
        # Head-sharded attention needs ONE PRIVATE scratch per engine -- the
        # heads run concurrently and each writes its own V.T / scores / scaled-Q.
        # Core 0 keeps the tensor-arena buffer above; workers get theirs from
        # their private low-DRAM window, which cannot alias the model map
        # however the model's cursors move.
        self.VIS_ATTN_SCRATCH_PER_ENGINE = [self.VIS_ATTN_SCRATCH]
        if getattr(self, "multi_core", 1) > 1:
            scratch_bytes = scratch_elems * bpe
            self.VIS_ATTN_SCRATCH_PER_ENGINE.extend(
                self.mc_arena.alloc_tensor(e, scratch_bytes, "vision attn scratch")
                for e in range(1, self.multi_core))
            spans = sorted((a, a + scratch_bytes)
                           for a in self.VIS_ATTN_SCRATCH_PER_ENGINE)
            for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
                assert prev_end <= next_start, "per-engine attention scratch overlaps"
            self._loud(f"  Vision attention scratch: {self.multi_core} private "
                       f"buffer(s), {scratch_bytes / 2**20:.2f} MiB each")

        # Attention output: token-major at the padded width, then trimmed.
        self.VIS_ATTN_PAD = alloc(VS * VN * VD_PAD, "vis.attn_pad")
        self.VIS_ATTN_RESULT = alloc(VS * VH, "vis.attn_result")

        self.VIS_O_PROJ = alloc(VS * VH, "vis.o_proj")
        self.VIS_RESIDUAL = alloc(VS * VH, "vis.residual")
        self.VIS_MLP_GATE = alloc(VS * VI_PAD, "vis.mlp_gate")
        self.VIS_MLP_UP = alloc(VS * VI_PAD, "vis.mlp_up")
        self.VIS_MLP_MULT = alloc(VS * VI_PAD, "vis.mlp_mult")
        self.VIS_MLP_DOWN = alloc(VS * VH, "vis.mlp_down")

        # Merger. The 2x2 spatial merge is a pure reinterpretation of
        # POST_NORM [VS, VH] as [VS/4, 4*VH] -- window reordering already put
        # each merge unit's 4 patches in consecutive rows -- so the merger
        # matmul reads POST_NORM directly and no gather is emitted.
        self.VIS_POST_NORM = alloc(VS * VH, "vis.post_norm")
        self.VIS_MERGER_INTER = alloc(T * VMERGE * VMERGE * VH, "vis.merger_inter")
        self.VIS_ENCODER_OUT = alloc(T * VH_OUT, "vis.encoder_out")

        # RoPE table, tiled per head (see _build_rope_tables).
        self.VIS_ROPE = alloc(VN * VS * 2 * VD_PAD, "vis.rope")
        self.VIS_ROPE_COS = self.VIS_ROPE
        self.VIS_ROPE_SIN = self.VIS_ROPE + VD_PAD * bpe

        self._vis_pending_dmas.append((self.VIS_ROPE, self._build_rope_tables().flatten()))
        bias_full, bias_window = self._build_attention_bias()
        self._vis_pending_dmas.append((self.VIS_BIAS_FULL, bias_full))
        self._vis_pending_dmas.append((self.VIS_BIAS_WINDOW, bias_window))
        self._vis_pending_dmas.append(
            (self.VIS_IO_A, self._vis_patch_embeds.to(torch.bfloat16).contiguous().flatten()))

        end = self.get_tensor_dram_addr()
        if end > self.TENSOR_LIMIT:
            raise MemoryError(
                f"vision tensors overflow the tensor region: end 0x{end:X} > "
                f"limit 0x{self.TENSOR_LIMIT:X}")
        self._loud(f"  Vision tensors: {self.get_tensor_dram_usage() / 2**20:.1f} MiB "
                   f"at 0x{self._tensor_dram_base:X}..0x{end:X}")

    # ---- compile -----------------------------------------------------------

    def _prime_M(self, M: int) -> int:
        """ADD_SET gf_seq_len <- M; returns the register for gpr_M_reg.

        Folds a compile-time M into one runtime PBI loop instead of unrolling
        the op M times. gf_seq_len is an LM register, free in the vision program.
        Emits an instruction, so it is only valid inside an active capture.
        """
        self.generate_instruction_add_set(self.gf_seq_len, M)
        return self.gf_seq_len

    def _vis_trim_pad_lanes(self, src_tokens: int, dst_tokens: int) -> None:
        """[VS, VN*VD_PAD] -> [VS, VN*VD], dropping each head's 48 pad lanes.

        o_proj contracts over VN*VD = 1280, not the padded VN*VD_PAD = 2048, so
        the zero lanes have to come out first.

        The gather is PERIODIC, which is what makes it cheap: every head is a
        VD-wide chunk on a VD_PAD stride, and a token row (VN*VD_PAD) is an exact
        multiple of that stride, so one strided DMA walks straight across many
        consecutive tokens instead of restarting per token. SRAM capacity sets
        the batch. Writes are contiguous -- a multi-chunk strided WRITE returns
        wrong data on this device (verified in the previous build), so only the
        read side is strided.
        """
        vis = self._vision_dims()
        VS, VN, VD, VD_PAD = vis["VS"], vis["VN"], vis["VD"], vis["VD_PAD"]
        bpe = self.bytes_per_element
        out_row, in_row = VN * VD, VN * VD_PAD
        tokens_per_chunk = max(1, URAM_NEAR_FULL_ELEMENTS // out_row)
        for t0 in range(0, VS, tokens_per_chunk):
            take = min(tokens_per_chunk, VS - t0)
            self.accelerator_memory_to_sram(
                src_tokens + t0 * in_row * bpe, 0x00000, take * out_row,
                stride_bytes_per_chunk=VD * bpe,
                stride_jump_bytes=VD_PAD * bpe)
            self.sram_to_accelerator_memory(
                0x00000, dst_tokens + t0 * out_row * bpe, take * out_row)

    def compile_vision_encoder(self, profile: bool = False) -> int:
        """Capture the whole encoder -- 32 layers + merger -- as one program at
        the ISA base. Host emission only; nothing touches the device.

        ``profile``: emit a HALT at every phase boundary and record the resume
        address, so run_vision_encoder can time each phase's HW latency
        separately. A profile-compiled bin can ONLY be run segment-by-segment
        (each phase ends in a HALT), and the extra stop/restart round trips
        inflate the wall total -- read the SHARE column, not the absolute times.

        Returns the program's DRAM base address.
        """
        vis = self._vision_dims()
        VS, VH, VN, VD, VD_PAD = (vis["VS"], vis["VH"], vis["VN"],
                                  vis["VD"], vis["VD_PAD"])
        VI_PAD, VL, VMERGE, VH_OUT = (vis["VI_PAD"], vis["VL"],
                                      vis["VMERGE"], vis["VH_OUT"])
        T, FULL = vis["NUM_MERGED_TOKENS"], vis["FULL_ATTN_LAYERS"]
        bpe = self.bytes_per_element
        aligned_S = self._vis_aligned_S
        head_stride = aligned_S * VD_PAD * bpe
        merge_dim = VMERGE * VMERGE * VH

        # Softmax scale. head_dim is 80; the on-chip 128 is zero padding, which
        # contributes nothing to the dot product but WOULD change the default
        # 1/sqrt(head_dim) the core bakes in, so it is overridden here to match
        # HuggingFace exactly.
        # REMINDER: the previous build (models/qwen2.5_vl_3b) deliberately left
        # this at 1/sqrt(128), claiming the softer attention tolerated IF4 noise
        # better end-to-end. If captions come out wrong under IF4, try
        # q_scale = 1.0 / math.sqrt(VD_PAD) before hunting elsewhere.
        q_scale = 1.0 / math.sqrt(VD)

        sched = self._ensure_stage_scheduler("vision")
        self._loud(f"  [Vision] compiling {VL} layers + merger "
                   f"(block-diagonal window mask, q_scale=1/sqrt({VD}), "
                   f"{getattr(self, 'multi_core', 1)} engine(s)) ...")
        t0 = time.perf_counter()
        self.reset_program_dram_addr()
        base_addr = self.get_program_dram_addr()
        self.clear_inst_id()
        self.clear_capture_buffer()
        self.start_capture()
        gate_m_regs = None
        if sched is not None:
            sched.register_per_engine_addrs("vision_attn_scratch",
                                            self.VIS_ATTN_SCRATCH_PER_ENGINE)
            sched.begin_program()
            # Engine 0 reuses the model's row-count GPR; plain UnifiedEngine
            # workers have no such model attribute, so each allocates its own
            # after begin_program() resets its allocator.
            gate_m_regs = [self.gf_seq_len]
            gate_m_regs.extend(wk.alloc_isa_reg() for wk in sched.workers)
        prev_silent = self._set_silent(True)
        flops = 0
        self._vis_checkpoints = []
        _ckpt_flops = [0]

        def _ckpt(name: str) -> None:
            """End a profile phase: HALT, then record where to resume."""
            if not profile:
                return
            self.generate_instruction_halt()
            resume = (self.get_program_dram_addr()
                      + self.capture_count * INSTRUCTION_SIZE_BYTES)
            self._vis_checkpoints.append(
                [name, resume, int(flops - _ckpt_flops[0])])
            _ckpt_flops[0] = int(flops)

        def matmul(M, K, N, A, w, tag, OUT, *, silu=False, gelu=False, bias=True):
            return self.matmat_mul_core(
                M=M, K=K, N=N, A_DRAM_ADDR=A,
                B_DRAM_ADDR=w[f"{tag}_data"], OUTPUT_DRAM_ADDR=OUT,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=w[f"{tag}_scale"],
                C_DRAM_ADDR=w[f"{tag}_bias"] if bias else None,
                bias_mode="broadcast_N" if bias else None,
                silu_enable=silu, gelu_enable=gelu,
                gpr_M_reg=self._prime_M(M)) or 0

        def sh_matmul(ctx, m_reg, K, N, A, A_row, w, tag, OUT, OUT_row,
                      *, silu=False, gelu=False):
            """One engine's row-block of a projection. Bias is per-COLUMN
            (broadcast_N), so it is shared, not sliced."""
            ctx.ue.generate_instruction_add_set(m_reg, ctx.rows)
            return ctx.ue.matmat_mul_core(
                M=ctx.rows, K=K, N=N,
                A_DRAM_ADDR=ctx.rows_addr(A, A_row),
                B_DRAM_ADDR=w[f"{tag}_data"],
                OUTPUT_DRAM_ADDR=ctx.rows_addr(OUT, OUT_row),
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=w[f"{tag}_scale"],
                C_DRAM_ADDR=w[f"{tag}_bias"], bias_mode="broadcast_N",
                silu_enable=silu, gelu_enable=gelu, gpr_M_reg=m_reg) or 0

        h_row = VH * bpe
        for li in range(VL):
            w = self.vis_layer_addrs[li]
            IN = self.VIS_IO_A if li % 2 == 0 else self.VIS_IO_B
            OUT = self.VIS_IO_B if li % 2 == 0 else self.VIS_IO_A

            # --- attention ---
            if sched is None:
                flops += self.rms_norm_core_dram(
                    M=VS, N=VH, A_DRAM_ADDR=IN, OUTPUT_DRAM_ADDR=self.VIS_NORM_OUT,
                    GAMMA_DRAM_ADDR=w["norm1_weight"], gpr_M_reg=self._prime_M(VS)) or 0
                flops += matmul(VS, VH, 2 * VN * VD_PAD, self.VIS_NORM_OUT, w, "qk", self.VIS_QK)
                flops += matmul(VS, VH, VN * VD_PAD, self.VIS_NORM_OUT, w, "v", self.VIS_V)
            else:
                # Row-shard over tokens: norm1 and both projections are
                # independent per token row.
                acc = [0]

                def _proj(ctx, w=w, IN=IN, acc=acc):
                    m = gate_m_regs[ctx.engine_idx]
                    ctx.ue.generate_instruction_add_set(m, ctx.rows)
                    acc[0] += ctx.ue.rms_norm_core_dram(
                        M=ctx.rows, N=VH,
                        A_DRAM_ADDR=ctx.rows_addr(IN, h_row),
                        OUTPUT_DRAM_ADDR=ctx.rows_addr(self.VIS_NORM_OUT, h_row),
                        GAMMA_DRAM_ADDR=w["norm1_weight"], gpr_M_reg=m) or 0
                    acc[0] += sh_matmul(ctx, m, VH, 2 * VN * VD_PAD,
                                        self.VIS_NORM_OUT, h_row, w, "qk",
                                        self.VIS_QK, 2 * VN * VD_PAD * bpe)
                    acc[0] += sh_matmul(ctx, m, VH, VN * VD_PAD,
                                        self.VIS_NORM_OUT, h_row, w, "v",
                                        self.VIS_V, VN * VD_PAD * bpe)

                sched.sharded_region(VS, _proj)
                flops += acc[0]
            _ckpt(f"L{li}:qkv_proj")

            # Token-major -> head-major. Viewing the fused QK output as
            # [VS, 2*VN, VD_PAD] makes Q's heads groups 0..VN-1 and K's heads
            # groups VN..2VN-1, so one permute lands both, already split.
            self.bf16_permute_dram_core(2 * VN, VS, VD_PAD, self.VIS_QK, self.VIS_QK_HM,
                                        write_grouped=True, group_stride_rows=aligned_S)
            self.bf16_permute_dram_core(VN, VS, VD_PAD, self.VIS_V, self.VIS_V_HM,
                                        write_grouped=True, group_stride_rows=aligned_S)
            _ckpt(f"L{li}:permute_qkv")

            # RoPE over all heads at once; the table is tiled VN times so row
            # h*VS+t reads position t's cos/sin.
            if sched is None:
                for buf in (self.VIS_Q_HM, self.VIS_K_HM):
                    flops += self.rope_hf_core_dram(
                        M=VN * VS, N=VD_PAD, input_dram_addr=buf, output_dram_addr=buf,
                        cos_dram_addr=self.VIS_ROPE_COS, sin_dram_addr=self.VIS_ROPE_SIN,
                        gpr_M_reg=self._prime_M(VN * VS)) or 0
            else:
                # RoPE is row-independent over its flattened M = VN*VS, but
                # rope_hf_core_dram is NOT in the shard proxy's allowlist, so it
                # goes through ctx.unsafe_ue with explicitly sliced addresses --
                # the same escape hatch gemma4_e2b uses for its RoPE.
                # TWO STRIDES: the Q/K rows are VD_PAD wide, while the cos/sin
                # table is interleaved [cos||sin] and so strides 2*VD_PAD per row.
                rope_acc = [0]
                rope_row = VD_PAD * bpe
                table_row = 2 * VD_PAD * bpe

                def _rope(ctx, buf, acc=rope_acc):
                    m = gate_m_regs[ctx.engine_idx]
                    ctx.ue.generate_instruction_add_set(m, ctx.rows)
                    src = ctx.rows_addr(buf, rope_row)
                    acc[0] += ctx.unsafe_ue.rope_hf_core_dram(
                        M=ctx.rows, N=VD_PAD,
                        input_dram_addr=src, output_dram_addr=src,
                        cos_dram_addr=ctx.rows_addr(self.VIS_ROPE_COS, table_row),
                        sin_dram_addr=ctx.rows_addr(self.VIS_ROPE_SIN, table_row),
                        gpr_M_reg=m) or 0

                for buf in (self.VIS_Q_HM, self.VIS_K_HM):
                    sched.sharded_region(VN * VS,
                                         lambda ctx, b=buf: _rope(ctx, b))
                flops += rope_acc[0]
            _ckpt(f"L{li}:rope")

            bias = self.VIS_BIAS_FULL if li in FULL else self.VIS_BIAS_WINDOW

            def _attn_kernel(ue, head_dim, seq_len, _acc=None, **kwargs):
                """One head through unified_attention_core.

                Vision is MHA -- one call per head -- so the GQA fan-out
                argument the scheduler passes for the LM is dropped.
                """
                kwargs.pop("num_q_heads", None)
                batch_reg = ue.alloc_isa_reg()
                ue.generate_instruction_add_set(batch_reg, seq_len)
                aligned_reg = ue.alloc_isa_reg()
                ue.generate_instruction_add_set(aligned_reg, seq_len)
                f = ue.unified_attention_core(
                    batch=seq_len, aligned_seq_len=seq_len, head_dim=head_dim,
                    gpr_batch_reg=batch_reg, gpr_aligned_seq_len_reg=aligned_reg,
                    q_scale=q_scale, **kwargs)
                ue.release_isa_reg()
                ue.release_isa_reg()
                if _acc is not None and isinstance(f, (int, float)):
                    _acc[0] += f

            if sched is None:
                for h in range(VN):
                    off = h * head_stride
                    _acc = [0]
                    _attn_kernel(
                        self, VD_PAD, aligned_S, _acc=_acc,
                        Q_DRAM_ADDR=self.VIS_Q_HM + off,
                        K_DRAM_ADDR=self.VIS_K_HM + off,
                        V_DRAM_ADDR=self.VIS_V_HM + off,
                        BIAS_DRAM_ADDR=bias,
                        OUTPUT_DRAM_ADDR=self.VIS_OUT_HM + off,
                        SCRATCH_DRAM_ADDR=self.VIS_ATTN_SCRATCH,
                        IDENTITY_DRAM_ADDR=self._vis_identity_dram)
                    flops += _acc[0]
            else:
                # 16 heads over N engines; each engine writes only its own
                # heads' planes of OUT_HM and uses its own private scratch.
                attn_acc = [0]
                sched.head_sharded_attention(
                    VN, aligned_S, VD_PAD,
                    Q_addr=self.VIS_Q_HM, K_addr=self.VIS_K_HM,
                    V_addr=self.VIS_V_HM, OUT_addr=self.VIS_OUT_HM,
                    IDENTITY_addr=self._vis_identity_dram,
                    bias_addr=bias, bias_per_head=False,
                    scratch_name="vision_attn_scratch",
                    kernel=lambda ue, head_dim, seq_len, **kw: _attn_kernel(
                        ue, head_dim, seq_len, _acc=attn_acc, **kw))
                flops += attn_acc[0]
            _ckpt(f"L{li}:attention")

            # Head-major -> token-major at the PADDED width (the permute core
            # requires a 64-multiple row_width, and VD=80 is not one), then drop
            # the pad lanes in a second pass.
            self.bf16_permute_dram_core(VN, VS, VD_PAD, self.VIS_OUT_HM,
                                        self.VIS_ATTN_PAD, write_grouped=False,
                                        group_stride_rows=aligned_S)
            self._vis_trim_pad_lanes(self.VIS_ATTN_PAD, self.VIS_ATTN_RESULT)
            _ckpt(f"L{li}:unpermute+trim")
            if sched is None:
                flops += matmul(VS, VH, VH, self.VIS_ATTN_RESULT, w, "o", self.VIS_O_PROJ)
                flops += self.eltwise_core_dram(
                    M=VS, N=VH, dram_a=IN, dram_b=self.VIS_O_PROJ,
                    dram_out=self.VIS_RESIDUAL, mode=UE_MODE.ELTWISE_ADD,
                    gpr_M_reg=self._prime_M(VS)) or 0


                # --- SwiGLU MLP ---
                flops += self.rms_norm_core_dram(
                    M=VS, N=VH, A_DRAM_ADDR=self.VIS_RESIDUAL,
                    OUTPUT_DRAM_ADDR=self.VIS_NORM_OUT,
                    GAMMA_DRAM_ADDR=w["norm2_weight"], gpr_M_reg=self._prime_M(VS)) or 0
                flops += matmul(VS, VH, VI_PAD, self.VIS_NORM_OUT, w, "gate",
                                self.VIS_MLP_GATE, silu=True)
                flops += matmul(VS, VH, VI_PAD, self.VIS_NORM_OUT, w, "up", self.VIS_MLP_UP)
                flops += self.eltwise_core_dram(
                    M=VS, N=VI_PAD, dram_a=self.VIS_MLP_GATE, dram_b=self.VIS_MLP_UP,
                    dram_out=self.VIS_MLP_MULT, mode=UE_MODE.ELTWISE_MUL,
                    gpr_M_reg=self._prime_M(VS)) or 0
                flops += matmul(VS, VI_PAD, VH, self.VIS_MLP_MULT, w, "down",
                                self.VIS_MLP_DOWN)
                flops += self.eltwise_core_dram(
                    M=VS, N=VH, dram_a=self.VIS_RESIDUAL, dram_b=self.VIS_MLP_DOWN,
                    dram_out=OUT, mode=UE_MODE.ELTWISE_ADD,
                    gpr_M_reg=self._prime_M(VS)) or 0
                # ONE phase, matching the sharded path's single region: the
                # multi-core body fuses o_proj through the MLP so only its entry
                # and exit need a rendezvous, and a checkpoint inside a region
                # would strand the workers. Naming the serial path the same way
                # is what makes the 1-core and N-core tables comparable.
                _ckpt(f"L{li}:o_proj+mlp")
            else:
                # o_proj through the second residual is one region: within an
                # engine's rows the chain is sequential, so only region entry
                # and exit need a rendezvous.
                post_acc = [0]
                mlp_row = VI_PAD * bpe

                def _post(ctx, w=w, IN=IN, OUT=OUT, acc=post_acc):
                    m = gate_m_regs[ctx.engine_idx]
                    acc[0] += sh_matmul(ctx, m, VH, VH, self.VIS_ATTN_RESULT,
                                        h_row, w, "o", self.VIS_O_PROJ, h_row)
                    ctx.ue.generate_instruction_add_set(m, ctx.rows)
                    acc[0] += ctx.ue.eltwise_core_dram(
                        M=ctx.rows, N=VH,
                        dram_a=ctx.rows_addr(IN, h_row),
                        dram_b=ctx.rows_addr(self.VIS_O_PROJ, h_row),
                        dram_out=ctx.rows_addr(self.VIS_RESIDUAL, h_row),
                        mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=m) or 0
                    ctx.ue.generate_instruction_add_set(m, ctx.rows)
                    acc[0] += ctx.ue.rms_norm_core_dram(
                        M=ctx.rows, N=VH,
                        A_DRAM_ADDR=ctx.rows_addr(self.VIS_RESIDUAL, h_row),
                        OUTPUT_DRAM_ADDR=ctx.rows_addr(self.VIS_NORM_OUT, h_row),
                        GAMMA_DRAM_ADDR=w["norm2_weight"], gpr_M_reg=m) or 0
                    acc[0] += sh_matmul(ctx, m, VH, VI_PAD, self.VIS_NORM_OUT,
                                        h_row, w, "gate", self.VIS_MLP_GATE,
                                        mlp_row, silu=True)
                    acc[0] += sh_matmul(ctx, m, VH, VI_PAD, self.VIS_NORM_OUT,
                                        h_row, w, "up", self.VIS_MLP_UP, mlp_row)
                    ctx.ue.generate_instruction_add_set(m, ctx.rows)
                    acc[0] += ctx.ue.eltwise_core_dram(
                        M=ctx.rows, N=VI_PAD,
                        dram_a=ctx.rows_addr(self.VIS_MLP_GATE, mlp_row),
                        dram_b=ctx.rows_addr(self.VIS_MLP_UP, mlp_row),
                        dram_out=ctx.rows_addr(self.VIS_MLP_MULT, mlp_row),
                        mode=UE_MODE.ELTWISE_MUL, gpr_M_reg=m) or 0
                    acc[0] += sh_matmul(ctx, m, VI_PAD, VH, self.VIS_MLP_MULT,
                                        mlp_row, w, "down", self.VIS_MLP_DOWN, h_row)
                    ctx.ue.generate_instruction_add_set(m, ctx.rows)
                    acc[0] += ctx.ue.eltwise_core_dram(
                        M=ctx.rows, N=VH,
                        dram_a=ctx.rows_addr(self.VIS_RESIDUAL, h_row),
                        dram_b=ctx.rows_addr(self.VIS_MLP_DOWN, h_row),
                        dram_out=ctx.rows_addr(OUT, h_row),
                        mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=m) or 0

                sched.sharded_region(VS, _post)
                flops += post_acc[0]
                # o_proj through the MLP is ONE region here, so it is one phase.
                # Without this the sharded path emits no checkpoint at all and
                # the work silently folds into the next phase's sample.
                _ckpt(f"L{li}:o_proj+mlp")

        # --- merger ---
        final = self.VIS_IO_A if VL % 2 == 0 else self.VIS_IO_B
        flops += self.rms_norm_core_dram(
            M=VS, N=VH, A_DRAM_ADDR=final, OUTPUT_DRAM_ADDR=self.VIS_POST_NORM,
            GAMMA_DRAM_ADDR=self.merger_ln_q_weight,
            gpr_M_reg=self._prime_M(VS)) or 0
        # POST_NORM [VS, VH] reinterpreted as [T, 4*VH] -- no data movement.
        flops += self.matmat_mul_core(
            M=T, K=merge_dim, N=merge_dim, A_DRAM_ADDR=self.VIS_POST_NORM,
            B_DRAM_ADDR=self.merger_mlp0_data, OUTPUT_DRAM_ADDR=self.VIS_MERGER_INTER,
            is_B_quantized=True, data_type=TYPE.IF4,
            SCALE_DRAM_ADDR=self.merger_mlp0_scale,
            C_DRAM_ADDR=self.merger_mlp0_bias, bias_mode="broadcast_N",
            gelu_enable=True, gpr_M_reg=self._prime_M(T)) or 0
        flops += self.matmat_mul_core(
            M=T, K=merge_dim, N=VH_OUT, A_DRAM_ADDR=self.VIS_MERGER_INTER,
            B_DRAM_ADDR=self.merger_mlp2_data, OUTPUT_DRAM_ADDR=self.VIS_ENCODER_OUT,
            is_B_quantized=True, data_type=TYPE.IF4,
            SCALE_DRAM_ADDR=self.merger_mlp2_scale,
            C_DRAM_ADDR=self.merger_mlp2_bias, bias_mode="broadcast_N",
            gpr_M_reg=self._prime_M(T)) or 0
        _ckpt("merger")

        self.generate_instruction_halt()
        worker_addrs = sched.finalize() if sched is not None else []
        if sched is not None:
            for wk in reversed(sched.workers):
                wk.release_isa_reg()
        self._set_silent(prev_silent)
        self.stop_capture()

        enc = bytearray()
        for inst in self.capture_buffer:
            enc.extend(inst.get_bytes())
        self.clear_capture_buffer()

        # Worker images live in each engine's private ISA slice. An overrun
        # would not fault -- it would scribble over that engine's own scratch or
        # the next engine's window -- so the arena checks it where it is written.
        self._vis_worker_programs = []
        if sched is not None:
            for idx, (wk, addr) in enumerate(zip(sched.workers, worker_addrs), start=1):
                blob = bytearray()
                for inst in wk.capture_buffer:
                    blob.extend(inst.get_bytes())
                self.mc_arena.check_isa_fits(idx, addr, len(blob))
                self._note_worker_isa(idx, "vision", len(blob))
                self._vis_worker_programs.append((idx, wk, addr, bytes(blob)))
        if base_addr + len(enc) > self.DRAM_END:
            raise MemoryError(
                f"vision ISA overflow: 0x{base_addr + len(enc):X} > "
                f"0x{self.DRAM_END:X} ({(self.DRAM_END - base_addr) / 2**20:.0f} MiB "
                f"ISA region). Enlarge the ISA region or shrink the program.")
        self._vis_program_bytes = bytes(enc)
        self._vis_program_addr = base_addr
        self._vis_total_flops = int(flops)
        wk_note = ""
        if self._vis_worker_programs:
            wk_mib = sum(len(b) for *_, b in self._vis_worker_programs) / 2**20
            wk_note = (f", {len(self._vis_worker_programs)} worker image(s) "
                       f"{wk_mib:.2f} MiB")
        self._loud(f"  [Vision] encoder compiled: {len(enc) / 2**20:.2f} MiB at "
                   f"0x{base_addr:X}{wk_note}, {flops / 1e9:.1f} GFLOP, "
                   f"{time.perf_counter() - t0:.1f}s")
        return base_addr

    # ---- run ---------------------------------------------------------------

    def vis_peak_gflops(self) -> float:
        """Engine peak throughput, straight from HW_INFO.

        128 flops per cycle per core: freq_MHz x 0.128 GFLOPS, times the engine
        count in play. Same expression gemma4_e2b uses.

        NOT user_dma_core.UE_PEAK_GFLOPS -- despite the name that global is
        ``0.128 / clock_ns``, which is TFLOPS (0.047 on this board, not 46.9).
        Using it directly reports % of peak 1000x too high.
        """
        clock_ns = (getattr(self, "_clock_period_ns", None)
                    or user_dma_core.CLOCK_CYCLE_TIME_NS)
        if not clock_ns:
            return 0.0
        return (1000.0 / clock_ns) * 0.128 * getattr(self, "multi_core", 1)

    # Phases that stay on core 0 whatever --multi-core says: the strided-DMA
    # permutes, the pad-lane trim, and the merger tail. RoPE is NOT here -- it is
    # outside multi_engine_shard.SHARDED_OP_ALLOWLIST but still row-independent,
    # so it is sharded through ctx.unsafe_ue like gemma4_e2b does.
    # These bound what sharding can win, and their utilisation must be measured
    # against ONE core's peak, not the aggregate.
    _VIS_SERIAL_PHASES = ("permute_qkv", "unpermute+trim", "merger")

    def _aggregate_vis_profile(self, results):
        """Fold per-layer samples into one row per phase, preserving order."""
        peak = self.vis_peak_gflops()
        agg, order = {}, []
        for name, ms, ph_flops in results:
            phase = name.split(":", 1)[1] if ":" in name else name
            if phase not in agg:
                agg[phase] = dict(phase=phase, ms=0.0, flops=0, n=0)
                order.append(phase)
            row = agg[phase]
            row["ms"] += float(ms)
            row["n"] += 1
            row["flops"] += int(ph_flops)
        cores = getattr(self, "multi_core", 1)
        rows = []
        for phase in order:
            row = agg[phase]
            row["gflops"] = (row["flops"] / (row["ms"] * 1e6)) if row["ms"] else 0.0
            # A phase that runs on core 0 alone can only ever reach one core's
            # peak; charging it the aggregate would understate it N-fold.
            row["serial"] = phase in self._VIS_SERIAL_PHASES
            phase_peak = peak / cores if row["serial"] else peak
            row["util_pct"] = (100.0 * row["gflops"] / phase_peak) if phase_peak else 0.0
            # A phase cannot exceed the peak of the engines actually running it,
            # so >100% means the DENOMINATOR is wrong, not that the hardware
            # overachieved -- almost always _VIS_SERIAL_PHASES disagreeing with
            # what compile_vision_encoder really shards (charging a sharded
            # phase one core's peak inflates it by up to the core count).
            if row["util_pct"] > 100.5:
                row["suspect"] = True
                self._loud(
                    f"  [warn] {phase}: {row['util_pct']:.1f}% of peak is "
                    f"impossible -- the FLOP count is billed at shapes the "
                    f"hardware did not run, or the phase's share of the total "
                    f"is misattributed"
                    f"{' (also check _VIS_SERIAL_PHASES)' if row['serial'] else ''}.")
            else:
                row["suspect"] = False
            rows.append(row)
        return rows

    def _vis_profile_table(self, rows, total_ms):
        """The profile table as markdown rows, shared by the terminal and the .md."""
        out = []
        for r in rows:
            mark = " *" if r["serial"] else ""
            share = 100.0 * r["ms"] / total_ms if total_ms else 0.0
            out.append((r["phase"] + mark, r["n"], r["ms"], share,
                        r["flops"] / 1e9, r["gflops"], r["util_pct"]))
        return out

    def print_profile_table(self, title, results, note=None) -> None:
        """Per-phase breakdown, printed for any stage.

        Absolute times include one HALT + restart round trip per phase, so the
        SHARE column is the number to act on -- it says which phase to shard
        first, which is the whole reason to run this.
        """
        rows = self._aggregate_vis_profile(results)
        total_ms = sum(r["ms"] for r in rows) or 1.0
        peak = self.vis_peak_gflops()
        self._loud(f"\n  === {title} ===" + (f"  {note}" if note else ""))
        print(f"  peak {peak:.1f} GFLOPS ({getattr(self, 'multi_core', 1)} core(s))")
        print(f"\n  {'phase':<18}{'calls':>6}{'total ms':>10}{'share':>8}"
              f"{'GFLOP':>9}{'GFLOPS':>9}{'% peak':>9}")
        print(f"  {'-' * 69}")
        for name, n, ms, share, gf, gfs, util in self._vis_profile_table(rows, total_ms):
            print(f"  {name:<18}{n:>6}{ms:>10.1f}{share:>7.1f}%"
                  f"{gf:>9.1f}{gfs:>9.1f}{util:>8.1f}%")
        print(f"  {'-' * 69}")
        tot_gf = sum(r["flops"] for r in rows) / 1e9
        tot_gfs = tot_gf / (total_ms / 1e3) if total_ms else 0.0
        print(f"  {'TOTAL':<18}{len(results):>6}{total_ms:>10.1f}{100.0:>7.1f}%"
              f"{tot_gf:>9.1f}{tot_gfs:>9.1f}{(100 * tot_gfs / peak if peak else 0):>8.1f}%")
        blocked = sum(r["ms"] for r in rows if r["serial"])
        if not blocked:
            return
        frac = blocked / total_ms if total_ms else 0.0
        cores = getattr(self, "multi_core", 1)
        print(f"\n  * runs on core 0 only (strided-DMA permutes, pad-lane trim, "
              f"merger): {100 * frac:.1f}% of HW time; % peak is vs ONE core.")
        if cores == 1:
            # The Amdahl form takes SINGLE-core phase times. Applying it to an
            # already-sharded run would treat times that never shrink as though
            # they still could, and understate the ceiling.
            print(f"    Amdahl ceiling with the rest sharded: "
                  f"{1.0 / (frac + (1 - frac) / 8):.2f}x at 8 cores.")
        else:
            print(f"    These do not shrink with more cores; at {cores} cores they "
                  f"are already {100 * frac:.1f}% of the total.")

    def run_vision_encoder(self, timeout_s: float = 600.0,
                           profile: bool = False) -> torch.Tensor:
        """Upload the queued inputs and the program, execute, and return the
        [NUM_MERGED_TOKENS, out_hidden_size] embeddings in raster order."""
        if not hasattr(self, "_vis_program_bytes"):
            raise RuntimeError("compile_vision_encoder() must run first")
        vis = self._vision_dims()
        T, VH_OUT = vis["NUM_MERGED_TOKENS"], vis["VH_OUT"]

        for addr, tensor in self._vis_pending_dmas:
            self.dma_to_accelerator_memory(addr, tensor)
        self._vis_pending_dmas = []

        addr = self._vis_program_addr
        self._next_program_dram_addr = addr
        self.dma_write(DMA_DEVICE_H2C, addr, self._vis_program_bytes,
                       len(self._vis_program_bytes))
        self.allocate_program_dram(len(self._vis_program_bytes))

        sched = self._ensure_stage_scheduler("vision")
        worker_addrs = []
        for idx, wk, wk_addr, blob in getattr(self, "_vis_worker_programs", []):
            wk._next_program_dram_addr = wk_addr
            wk.dma_write(DMA_DEVICE_H2C, wk_addr, blob, len(blob))
            wk.allocate_program_dram(len(blob))
            worker_addrs.append(wk_addr)
        if sched is not None:
            # A flag left set by an earlier program would make the first CHECK
            # pass spuriously, so clear every engine's flag before the run.
            sched.preclear_flags()

        self._loud(f"  [Vision] launching encoder "
                   f"({len(self._vis_program_bytes) / 2**20:.2f} MiB) at 0x{addr:X}"
                   f"{' [profiled]' if profile else ''} ...")
        t0 = time.perf_counter()
        if profile:
            checkpoints = getattr(self, "_vis_checkpoints", [])
            if not checkpoints:
                raise RuntimeError(
                    "profile run needs a profile-compiled bin; call "
                    "compile_vision_encoder(profile=True) first")
            # Each phase ends in a HALT, so the encoder is driven forward one
            # segment at a time. It still computes the full output -- the
            # segments tile the whole program.
            results = []
            # Workers are launched ONCE and run their whole program; they simply
            # block longer at each rendezvous while the master is stopped at a
            # checkpoint. This works only because every checkpoint sits OUTSIDE
            # a sharded region -- a HALT inside one would strand the workers.
            if sched is not None:
                sched.start_workers(worker_addrs)
            self.start_execute_from_dram(addr)
            for name, resume, ph_flops in checkpoints:
                self.wait_queue(timeout_s)
                results.append((name, self.report_latency_in_us() / 1e3, ph_flops))
                self.start_execute_from_dram(resume)
            self.wait_queue(timeout_s)
            for wk in (sched.workers if sched is not None else []):
                wk.wait_queue(timeout_s)
            latency_us = sum(r[1] for r in results) * 1e3
            self._vis_profile = results
        else:
            # Workers first: they park on their first rendezvous and wait for
            # the master to enter the region.
            if sched is not None:
                sched.start_workers(worker_addrs)
            self.start_execute_from_dram(addr)
            self.wait_queue(timeout_s)
            for wk in (sched.workers if sched is not None else []):
                wk.wait_queue(timeout_s)
            latency_us = self.report_latency_in_us()
        wall = time.perf_counter() - t0
        gflops = (self._vis_total_flops / (latency_us * 1e-6) / 1e9
                  if latency_us > 0 else 0.0)
        self._vis_latency_us = latency_us
        self._vis_gflops = gflops
        self._vis_wall_s = wall
        self._loud(f"  [Vision] done: {wall:.2f}s wall, {latency_us / 1e6:.2f}s HW, "
                   f"{gflops:.1f} GFLOPS")

        out = self.dma_from_accelerator_memory(self.VIS_ENCODER_OUT, (T, VH_OUT)).cpu()
        # Undo the window reordering so the tokens are back in raster order,
        # which is what the LM prompt expects at the <|image_pad|> positions.
        if hasattr(self, "_vis_reverse_index"):
            out = out[self._vis_reverse_index].contiguous()
        self._vis_embeddings = out
        self._vis_num_tokens = T
        self._loud(f"  [Vision] {T} image embeddings ready")
        return out
