#!/usr/bin/env python3
"""Qwen2.5-Omni vision-encoder method group (ViT tower with window attention).

``Qwen25OmniVisionMixin`` carries every vision method for Omni's 1280-wide
NaViT tower with a width-3584 patch merger. Owned entirely by
qwen2.5_omni_7b -- NOT shared with qwen2.5_vl_3b (which has its own,
independently editable, near-identical file). The two started from the
same dataflow but a change here must never be assumed to apply there, or
vice versa; qwen2.5_omni_7b must depend only on shared libraries, never on
another model's own files.

Everything shared -- the config on ``self._cfg``, the DRAM allocators,
``self._loud`` -- resolves through the concrete class, so this module
imports nothing from the test module (keeps the split cycle-free). This
mirrors the gemma4_e2b_vision.py arrangement.

WEIGHTS ARE TRANSIENT. The encoder runs exactly once, before LM prefill, so its
weights and the LM's occupy the SAME params window at 0x8000_0000: vision loads
first, the encoder runs, its 144 output embeddings are copied out, and the LM
weights are then loaded over the top. Only the encoder output survives the
handover -- see the DRAM map in qwen2.5_omni_7b_test.py.
"""
from __future__ import annotations

import math
import os
import sys
import time

_SD = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SD)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SD)))

import numpy as np
import torch
import torch.nn.functional as F

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_C2H, DMA_DEVICE_H2C, INSTRUCTION_SIZE_BYTES, TYPE, UE_MODE, UE_VECTOR_SIZE,
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


class Qwen25OmniVisionMixin:
    """Vision-encoder methods for Qwen25Omni_UnifiedEngine (see module docstring).

    _base_* methods are the inherited-tower baseline (weight init, tensor
    alloc, compile, run); the public methods of the same un-prefixed name
    are Omni's own overrides, which call the _base_ version for the parts
    that don't change (weight DMA layout, program capture bookkeeping) and
    add Omni-specific behavior around it (the FPGA patch-embed matmul,
    raw-pixel tensor init, the combined encoder+patch program image).
    """

    def _vision_dims(self) -> dict:
        """Vision geometry from the config, with the on-chip padding applied.

        head_dim 80 is not a multiple of the 64-element quantization block, and
        the attention core needs a 64-aligned head stride, so the ATTENTION-FACING
        Q/K/V buffers (VIS_QK/VIS_V) are stored/padded to VD_PAD=128 per head.
        Likewise the FFN 3420 -> 3456.

        VD_HALF describes RoPE's split, not a storage width: the rotate-half
        circuit pairs element i with i+VD_PAD/2=64 in hardware, unconditionally,
        so each head's real 80 dims must land in the padded buffer as two
        40-wide halves 64 apart (not one 80-wide run) -- see
        _build_rope_tables. When self._vis_qk_compact (see vision_weight_init),
        Q/K/V weights are stored fully COMPACT (real width, no padding baked in
        at all) and the qkv_proj emission writes each head (V) or head-half
        (QK) DIRECTLY to its final padded position via the matmul's own
        gpr_out_row_stride_reg -- no scatter DMA, no intermediate buffer.
        """
        v = self._cfg["vision"]
        VD_PAD = 128
        VD_HALF = v["head_dim"] // 2
        VI_PAD = ((v["intermediate_size"] + UE_VECTOR_SIZE - 1)
                  // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
        return dict(
            VS=v["num_patches"], VH=v["hidden_size"], VN=v["num_heads"],
            VD=v["head_dim"], VD_PAD=VD_PAD, VD_HALF=VD_HALF,
            VI=v["intermediate_size"], VI_PAD=VI_PAD,
            VL=v["depth"], VMERGE=v["spatial_merge_size"],
            VH_OUT=v["out_hidden_size"],
            NUM_MERGED_TOKENS=v["num_merged_tokens"],
            FULL_ATTN_LAYERS=set(v["fullatt_block_indexes"]),
            WINDOW_SIZE=v["window_size"], PATCH_SIZE=v["patch_size"],
        )

    def _dma_bf16(self, f, section: dict, base_offset: int, what: str) -> int:
        """Copy one BF16 tensor from the bin straight into params DRAM."""
        f.seek(base_offset + section["offset"])
        blob = f.read(section["size"])
        if len(blob) != section["size"]:
            raise RuntimeError(f"truncated read for {what}")
        addr = self.allocate_params_dram(section["size"], label=what)
        written = self.dma_write(DMA_DEVICE_H2C, addr, blob, section["size"])
        if written != section["size"]:
            raise IOError(
                f"{what}: params DMA wrote {written} of {section['size']} bytes")
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
        written = self.dma_write(
            DMA_DEVICE_H2C, scale_addr, blob[:scale_bytes], scale_bytes)
        if written != scale_bytes:
            raise IOError(
                f"{what}.scale: params DMA wrote {written} of {scale_bytes} bytes")
        data_addr = self.allocate_params_dram(data_bytes, label=f"{what}.data")
        written = self.dma_write(
            DMA_DEVICE_H2C,
            data_addr,
            blob[scale_bytes:scale_bytes + data_bytes],
            data_bytes,
        )
        if written != data_bytes:
            raise IOError(
                f"{what}.data: params DMA wrote {written} of {data_bytes} bytes")
        return scale_addr, data_addr

    def _base_vision_weight_init(self) -> None:
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
        # Loading from this point overwrites any previous LM image. Clear both
        # phase-cache flags before the first DMA so a partial transfer cannot
        # make a retry trust corrupt shared-window contents.
        self._vision_weight_init_done = False
        self._lm_weight_init_done = False
        if hasattr(self, "_audio_weight_init_done"):
            self._audio_weight_init_done = False
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

        # attn.qk / attn.v (compact, real width, no per-head padding baked in
        # -- the runtime scatters into padded per-head storage itself, see
        # compile_vision_encoder_bin's qkv_proj emission) vs the older
        # attn.qk_padded / attn.v_padded (padding baked into the quantized
        # weight at conversion time). Detected from whichever key this
        # params.bin actually has, so an old, not-yet-reconverted params.bin
        # for THIS model keeps working unchanged via the padded path.
        self._vis_qk_compact = f"visual.blocks.0.attn.qk.weight.{sfx}" in sections
        qk_key = "attn.qk" if self._vis_qk_compact else "attn.qk_padded"
        v_key = "attn.v" if self._vis_qk_compact else "attn.v_padded"

        with open(region["bin_path"], "rb") as f:
            base = region["base_offset"]
            layer_addrs = []
            for li in range(d["VL"]):
                pre = f"visual.blocks.{li}"
                la = {}
                for tag, key in (("qk", qk_key), ("v", v_key),
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
        qk_suffix = (".attn.qk.weight" if getattr(self, "_vis_qk_compact", False)
                    else ".attn.qk_padded.weight")
        v_suffix = (".attn.v.weight" if getattr(self, "_vis_qk_compact", False)
                   else ".attn.v_padded.weight")
        groups = {
            "qk (fused)": qk_suffix,
            "v":          v_suffix,
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

    def _build_attention_bias(self) -> torch.Tensor:
        """All-zero [VS, VS] additive attention bias for the FULL_ATTN_LAYERS
        (unmasked full-sequence attention -- there is nothing to mask).

        Windowed layers do NOT use this: they run TRUE per-window attention
        (one call per (window, head) at aligned_seq_len=window_tokens, see
        _compile_vision_encoder_impl), which needs only a small all-zero
        [window_tokens, window_tokens] bias (VIS_BIAS_WIN_ZERO, built in
        vision_tensor_init) shared by every window/head/layer -- a token
        genuinely only ever attends within its own call's window, so there is
        no cross-window term left to mask out. This replaces an earlier
        design that computed full 576x576 attention for EVERY layer and used
        a block-diagonal -inf mask to zero the cross-window terms after the
        fact: mathematically identical output, but 54.4 GFLOP issued for 12.1
        GFLOP of real work (4.5x) on every windowed layer, all 28 of them.
        """
        vis = self._vision_dims()
        VS = vis["VS"]
        return torch.zeros(VS, VS, dtype=torch.bfloat16)

    def _alloc_per_engine_attn_scratch(self, engine_idx: int, size_bytes: int) -> int:
        """One engine's private vision attention scratch.

        Default: the per-engine tensor slice, which is where it has always come
        from. A model whose scratch outgrows that slice overrides this -- the
        buffer scales as aligned_S^2, so at a large patch count it is tens of
        MiB and no longer belongs in a fixed-size per-core slice.
        """
        return self.mc_arena.alloc_tensor(
            engine_idx, size_bytes, "vision attn scratch")

    def _base_vision_tensor_init(self) -> None:
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
        # Padded per head (VD_PAD, not the real VD) because permute_qkv/RoPE/
        # attention downstream all address one head at a fixed VD_PAD stride.
        # When self._vis_qk_compact, the qkv_proj emission writes each head's
        # (V) or head-half's (QK, RoPE) REAL data directly into its final
        # position here via the matmul's own gpr_out_row_stride_reg -- no
        # intermediate compact buffer, no scatter DMA.
        self.VIS_QK = alloc(VS * VN * VD_PAD * 2, "vis.qk")
        self.VIS_V = alloc(VS * VN * VD_PAD, "vis.v")
        if getattr(self, "_vis_qk_compact", False):
            # ZERO ONCE. Each per-head(-half) matmul only writes its REAL
            # columns (QK: [0:40)+[64:104) of each 128-wide head slot; V:
            # [0:80)) -- unlike the old fully-padded matmul, which wrote the
            # WHOLE VD_PAD-wide row (zero rows baked into the weight) every
            # single layer, nothing ever writes the remaining gap lanes (QK:
            # [40:64)+[104:128); V: [80:128)) on any layer. Those lanes have
            # to start zero here and then, since nothing ever touches them
            # again, they simply stay zero -- the same self-annihilating-
            # padding pattern pi05's vision encoder uses. Without this,
            # VIS_QK/VIS_V start as uninitialized DRAM and every layer's
            # attention/RoPE reads garbage in those lanes.
            self._vis_pending_dmas.append(
                (self.VIS_QK, torch.zeros(VS * VN * VD_PAD * 2, dtype=torch.bfloat16)))
            self._vis_pending_dmas.append(
                (self.VIS_V, torch.zeros(VS * VN * VD_PAD, dtype=torch.bfloat16)))

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

        # Attention: full-sequence bias for the 4 FULL_ATTN_LAYERS (see
        # _build_attention_bias) and a small all-zero bias for the 28 windowed
        # layers, which now run TRUE per-window attention (one call per
        # (window, head), aligned_seq_len=window_tokens, no cross-window mask
        # needed at all -- see the qkv_proj-style rationale in
        # _compile_vision_encoder_impl's attention emission for why a
        # block-diagonal mask over the full 576 was replaced).
        self.VIS_BIAS_FULL = alloc(aligned_S * aligned_S, "vis.bias_full")
        cu = getattr(self, "_cu_window_seqlens", None)
        if cu is None or len(cu) < 2:
            raise RuntimeError("prepare_encoder_input() must run before vision_tensor_init()")
        window_sizes = [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]
        if len(set(window_sizes)) != 1:
            raise ValueError(
                f"per-window attention requires uniform window token counts, got {window_sizes}")
        self._vis_window_tokens = window_sizes[0]
        self._vis_num_windows = len(window_sizes)
        self.VIS_BIAS_WIN_ZERO = alloc(self._vis_window_tokens * self._vis_window_tokens,
                                       "vis.bias_win_zero")
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
                self._alloc_per_engine_attn_scratch(e, scratch_bytes)
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
        self._vis_pending_dmas.append((self.VIS_BIAS_FULL, self._build_attention_bias()))
        self._vis_pending_dmas.append(
            (self.VIS_BIAS_WIN_ZERO,
             torch.zeros(self._vis_window_tokens * self._vis_window_tokens, dtype=torch.bfloat16)))
        self._vis_pending_dmas.append(
            (self.VIS_IO_A, self._vis_patch_embeds.to(torch.bfloat16).contiguous().flatten()))

        end = self.get_tensor_dram_addr()
        if end > self.TENSOR_LIMIT:
            raise MemoryError(
                f"vision tensors overflow the tensor region: end 0x{end:X} > "
                f"limit 0x{self.TENSOR_LIMIT:X}")
        # Report bytes, not an address range: a model whose tensors are carved
        # per buffer from a pool has no single extent to name, and printing the
        # accounting origin as though it were a base address is worse than
        # printing nothing.
        self._loud(f"  Vision tensors: {self.get_tensor_dram_usage() / 2**20:.1f} MiB "
                   f"in {len(self._vis_pending_dmas)} staged buffer(s)")

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

    def _base_compile_vision_encoder(self, profile: bool = False) -> int:
        """Capture the encoder and restore primary/worker state on failure."""
        previous_silent = self._set_silent(False)
        self._set_silent(previous_silent)
        reg_counter_before = self._isa_reg_counter
        inst_ptr_counter_before = self._inst_ptr_counter
        program_cursor_before = self.get_program_dram_addr()
        scheduler_before = getattr(self, "_multi_core_schedulers", {}).get("vision")
        worker_cursors_before = (
            [worker.get_program_dram_addr() for worker in scheduler_before.workers]
            if scheduler_before is not None else None
        )
        try:
            return self._compile_vision_encoder_impl(profile)
        except Exception:
            scheduler = getattr(self, "_multi_core_schedulers", {}).get("vision")
            if scheduler is not None:
                scheduler.abort_program()
                if worker_cursors_before is None:
                    for worker in scheduler.workers:
                        worker.reset_program_dram_addr()
                else:
                    for worker, cursor in zip(
                        scheduler.workers, worker_cursors_before
                    ):
                        worker._next_program_dram_addr = cursor
            if getattr(self, "is_capture_on", False):
                self.stop_capture()
            self.clear_capture_buffer()
            self._isa_reg_counter = reg_counter_before
            self._inst_ptr_counter = inst_ptr_counter_before
            self._next_program_dram_addr = program_cursor_before
            raise
        finally:
            self._set_silent(previous_silent)

    def _compile_vision_encoder_impl(self, profile: bool = False) -> int:
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

        # TRUE per-window attention for the 28 windowed layers (see
        # _build_attention_bias): each (window, head) is an independent
        # attention call at aligned_seq_len=window_tokens, addressed as one
        # flat unit of a [VN*num_windows, window_tokens, VD_PAD] blob. This is
        # exact, not an approximation, because the head-major buffers are
        # ALREADY window-ordered along the token axis (prepare_encoder_input's
        # window reorder) and head_stride is an exact multiple of one window's
        # bytes (aligned_S = window_tokens * num_windows) -- so head h's
        # windows sit back-to-back immediately followed by head h+1's, making
        # the whole [VN, num_windows] grid one uniformly-strided flat axis.
        window_tokens = self._vis_window_tokens
        num_windows = self._vis_num_windows
        if aligned_S != window_tokens * num_windows:
            raise ValueError(
                f"windowed attention needs aligned_S={aligned_S} == "
                f"window_tokens({window_tokens}) * num_windows({num_windows})")
        window_bytes = window_tokens * VD_PAD * bpe
        H_WIN = VN * num_windows

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
        qk_compact = getattr(self, "_vis_qk_compact", False)
        qk_out_stride_regs = v_out_stride_regs = None
        if sched is not None:
            sched.register_per_engine_addrs("vision_attn_scratch",
                                            self.VIS_ATTN_SCRATCH_PER_ENGINE)
            sched.begin_program()
            # Engine 0 reuses the model's row-count GPR; plain UnifiedEngine
            # workers have no such model attribute, so each allocates its own
            # after begin_program() resets its allocator.
            gate_m_regs = [self.gf_seq_len]
            gate_m_regs.extend(wk.alloc_isa_reg() for wk in sched.workers)
            if qk_compact:
                # gpr_out_row_stride_reg constants (see the qkv_proj emission
                # below): fixed for the whole program, one register per engine,
                # primed once here instead of every one of the 80 per-head(-half)
                # matmul_slot calls per layer.
                qk_out_stride_regs, v_out_stride_regs = [], []
                for eng in (self, *sched.workers):
                    r = eng.alloc_isa_reg()
                    eng.generate_instruction_add_set(r, 2 * VN * VD_PAD)
                    qk_out_stride_regs.append(r)
                    r2 = eng.alloc_isa_reg()
                    eng.generate_instruction_add_set(r2, VN * VD_PAD)
                    v_out_stride_regs.append(r2)
        elif qk_compact:
            qk_out_stride_regs = [self.alloc_isa_reg()]
            self.generate_instruction_add_set(qk_out_stride_regs[0], 2 * VN * VD_PAD)
            v_out_stride_regs = [self.alloc_isa_reg()]
            self.generate_instruction_add_set(v_out_stride_regs[0], VN * VD_PAD)
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

        def matmul_slot(M, K, N, A, data_addr, scale_addr, bias_addr, OUT,
                        *, stride_reg):
            """One head's (or RoPE head-half's) REAL-width projection,
            written straight to its final padded position by the matmul's
            OWN output addressing (gpr_out_row_stride_reg) -- see the
            qkv_proj emission below for why this replaces a DMA scatter
            entirely. data_addr/scale_addr/bias_addr are the weight blob's
            base address PLUS this slice's own row offset -- IF4 blocks are
            along K, not N, so an arbitrary N-row slice is pure address
            arithmetic, no re-quantization."""
            return self.matmat_mul_core(
                M=M, K=K, N=N, A_DRAM_ADDR=A,
                B_DRAM_ADDR=data_addr, OUTPUT_DRAM_ADDR=OUT,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=scale_addr,
                C_DRAM_ADDR=bias_addr, bias_mode="broadcast_N",
                gpr_M_reg=self._prime_M(M),
                gpr_out_row_stride_reg=stride_reg) or 0

        def sh_matmul_slot(ctx, m_reg, K, N, A, A_row, data_addr, scale_addr,
                           bias_addr, OUT, OUT_row_bytes, *, stride_reg):
            """Sharded counterpart of matmul_slot: this engine's row-block of
            one head's (or head-half's) REAL-width projection, written
            straight to its row-sharded slice of the final padded position."""
            return ctx.ue.matmat_mul_core(
                M=ctx.rows, K=K, N=N,
                A_DRAM_ADDR=ctx.rows_addr(A, A_row),
                B_DRAM_ADDR=data_addr,
                OUTPUT_DRAM_ADDR=ctx.rows_addr(OUT, OUT_row_bytes),
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=scale_addr,
                C_DRAM_ADDR=bias_addr, bias_mode="broadcast_N",
                gpr_M_reg=m_reg, gpr_out_row_stride_reg=stride_reg) or 0

        h_row = VH * bpe
        VD_HALF = vis["VD_HALF"]
        # IF4 blocks are along K (=VH here), one full block-row per output
        # feature -- so every head's (or half's) N-slice is these many bytes,
        # regardless of which head/half it is. Shared by QK and V (both
        # contract over VH).
        if qk_compact:
            data_row_bytes = VH // 2
            scale_row_bytes = (VH // 64) * 2
            qk_row_bytes = 2 * VN * VD_PAD * bpe
            v_row_bytes = VN * VD_PAD * bpe
        for li in range(VL):
            w = self.vis_layer_addrs[li]
            IN = self.VIS_IO_A if li % 2 == 0 else self.VIS_IO_B
            OUT = self.VIS_IO_B if li % 2 == 0 else self.VIS_IO_A

            # --- attention ---
            if sched is None:
                flops += self.rms_norm_core_dram(
                    M=VS, N=VH, A_DRAM_ADDR=IN, OUTPUT_DRAM_ADDR=self.VIS_NORM_OUT,
                    GAMMA_DRAM_ADDR=w["norm1_weight"], gpr_M_reg=self._prime_M(VS)) or 0
                if qk_compact:
                    # Each head (V) / head-half (QK, RoPE) is its own REAL-
                    # width matmul, writing straight to its final padded slot
                    # via gpr_out_row_stride_reg -- no scatter, no compact
                    # staging buffer. See matmul_slot's docstring and
                    # _vision_dims for the RoPE split-half rationale. Weight
                    # rows are addressed directly (IF4 blocks are along K,
                    # not N, so any N-row slice is free address arithmetic).
                    for proj in range(2):          # 0 = Q, 1 = K
                        for h in range(VN):
                            for half in range(2):
                                n0 = proj * VH + h * VD + half * VD_HALF
                                OUT_addr = (self.VIS_QK + (proj * VN + h) * VD_PAD * bpe
                                           + half * (VD_PAD // 2) * bpe)
                                flops += matmul_slot(
                                    VS, VH, VD_HALF, self.VIS_NORM_OUT,
                                    w["qk_data"] + n0 * data_row_bytes,
                                    w["qk_scale"] + n0 * scale_row_bytes,
                                    w["qk_bias"] + n0 * bpe,
                                    OUT_addr, stride_reg=qk_out_stride_regs[0])
                    for h in range(VN):
                        n0 = h * VD
                        OUT_addr = self.VIS_V + h * VD_PAD * bpe
                        flops += matmul_slot(
                            VS, VH, VD, self.VIS_NORM_OUT,
                            w["v_data"] + n0 * data_row_bytes,
                            w["v_scale"] + n0 * scale_row_bytes,
                            w["v_bias"] + n0 * bpe,
                            OUT_addr, stride_reg=v_out_stride_regs[0])
                else:
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
                    if qk_compact:
                        sreg_qk = qk_out_stride_regs[ctx.engine_idx]
                        sreg_v = v_out_stride_regs[ctx.engine_idx]
                        for proj in range(2):
                            for h in range(VN):
                                for half in range(2):
                                    n0 = proj * VH + h * VD + half * VD_HALF
                                    OUT_addr = (self.VIS_QK + (proj * VN + h) * VD_PAD * bpe
                                               + half * (VD_PAD // 2) * bpe)
                                    acc[0] += sh_matmul_slot(
                                        ctx, m, VH, VD_HALF, self.VIS_NORM_OUT, h_row,
                                        w["qk_data"] + n0 * data_row_bytes,
                                        w["qk_scale"] + n0 * scale_row_bytes,
                                        w["qk_bias"] + n0 * bpe,
                                        OUT_addr, qk_row_bytes, stride_reg=sreg_qk)
                        for h in range(VN):
                            n0 = h * VD
                            OUT_addr = self.VIS_V + h * VD_PAD * bpe
                            acc[0] += sh_matmul_slot(
                                ctx, m, VH, VD, self.VIS_NORM_OUT, h_row,
                                w["v_data"] + n0 * data_row_bytes,
                                w["v_scale"] + n0 * scale_row_bytes,
                                w["v_bias"] + n0 * bpe,
                                OUT_addr, v_row_bytes, stride_reg=sreg_v)
                    else:
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

            is_full = li in FULL

            def _attn_kernel(ue, head_dim, seq_len, _acc=None, **kwargs):
                """One head (or one window-head unit) through
                unified_attention_core.

                Vision is MHA -- one call per unit -- so the GQA fan-out
                argument the scheduler passes for the LM is dropped.

                pv_head_dim=VD (real 80, not the padded head_dim=128): the
                score matmul (Q@K^T) and the V-transpose still need the full
                padded head_dim -- both have a genuine 64-alignment
                requirement -- but the final P@V matmul's OUTPUT width has
                none (no softmax on that one), so it writes only the real 80
                columns per row. VIS_OUT_HM's [80:128) pad columns are
                already zeroed once in vision_tensor_init and untouched by
                anything else, so they simply stay zero -- same
                self-annihilating-padding pattern as VIS_QK/VIS_V.
                """
                kwargs.pop("num_q_heads", None)
                batch_reg = ue.alloc_isa_reg()
                ue.generate_instruction_add_set(batch_reg, seq_len)
                aligned_reg = ue.alloc_isa_reg()
                ue.generate_instruction_add_set(aligned_reg, seq_len)
                f = ue.unified_attention_core(
                    batch=seq_len, aligned_seq_len=seq_len, head_dim=head_dim,
                    gpr_batch_reg=batch_reg, gpr_aligned_seq_len_reg=aligned_reg,
                    q_scale=q_scale, pv_head_dim=VD, **kwargs)
                ue.release_isa_reg()
                ue.release_isa_reg()
                if _acc is not None and isinstance(f, (int, float)):
                    _acc[0] += f

            if sched is None:
                if is_full:
                    for h in range(VN):
                        off = h * head_stride
                        _acc = [0]
                        _attn_kernel(
                            self, VD_PAD, aligned_S, _acc=_acc,
                            Q_DRAM_ADDR=self.VIS_Q_HM + off,
                            K_DRAM_ADDR=self.VIS_K_HM + off,
                            V_DRAM_ADDR=self.VIS_V_HM + off,
                            BIAS_DRAM_ADDR=self.VIS_BIAS_FULL,
                            OUTPUT_DRAM_ADDR=self.VIS_OUT_HM + off,
                            SCRATCH_DRAM_ADDR=self.VIS_ATTN_SCRATCH,
                            IDENTITY_DRAM_ADDR=self._vis_identity_dram)
                        flops += _acc[0]
                else:
                    for u in range(H_WIN):
                        off = u * window_bytes
                        _acc = [0]
                        _attn_kernel(
                            self, VD_PAD, window_tokens, _acc=_acc,
                            Q_DRAM_ADDR=self.VIS_Q_HM + off,
                            K_DRAM_ADDR=self.VIS_K_HM + off,
                            V_DRAM_ADDR=self.VIS_V_HM + off,
                            BIAS_DRAM_ADDR=self.VIS_BIAS_WIN_ZERO,
                            OUTPUT_DRAM_ADDR=self.VIS_OUT_HM + off,
                            SCRATCH_DRAM_ADDR=self.VIS_ATTN_SCRATCH,
                            IDENTITY_DRAM_ADDR=self._vis_identity_dram)
                        flops += _acc[0]
            else:
                attn_acc = [0]
                if is_full:
                    # 16 heads over N engines; each engine writes only its own
                    # heads' planes of OUT_HM and uses its own private scratch.
                    sched.head_sharded_attention(
                        VN, aligned_S, VD_PAD,
                        Q_addr=self.VIS_Q_HM, K_addr=self.VIS_K_HM,
                        V_addr=self.VIS_V_HM, OUT_addr=self.VIS_OUT_HM,
                        IDENTITY_addr=self._vis_identity_dram,
                        bias_addr=self.VIS_BIAS_FULL, bias_per_head=False,
                        scratch_name="vision_attn_scratch",
                        kernel=lambda ue, head_dim, seq_len, **kw: _attn_kernel(
                            ue, head_dim, seq_len, _acc=attn_acc, **kw))
                else:
                    # VN*num_windows (144) independent (window, head) units
                    # over N engines -- more, smaller, fully independent
                    # attention calls than the 16-head full case, so this
                    # shards far more evenly (144/8=18 units/engine vs 16/8=2
                    # heads/engine) as well as doing 4.5x less issued work.
                    # Addressed exactly like the H=VN case (see H_WIN/
                    # window_bytes above): head_sharded_attention's per-unit
                    # stride (seq_len*head_dim*bpe = window_bytes here) needs
                    # no extra plumbing, just smaller seq_len and more units.
                    sched.head_sharded_attention(
                        H_WIN, window_tokens, VD_PAD,
                        Q_addr=self.VIS_Q_HM, K_addr=self.VIS_K_HM,
                        V_addr=self.VIS_V_HM, OUT_addr=self.VIS_OUT_HM,
                        IDENTITY_addr=self._vis_identity_dram,
                        bias_addr=self.VIS_BIAS_WIN_ZERO, bias_per_head=False,
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
            if qk_compact:
                # Release in exact reverse of the per-engine alloc order
                # (qk_out_stride_regs then v_out_stride_regs, self then
                # workers) -- alloc_isa_reg/release_isa_reg is a LIFO counter
                # per engine, not a general allocator.
                for wk, qreg, vreg in reversed(
                    list(zip(sched.workers, qk_out_stride_regs[1:], v_out_stride_regs[1:]))
                ):
                    wk.release_isa_reg()
                    wk.release_isa_reg()
                self.release_isa_reg()
                self.release_isa_reg()
            for wk in reversed(sched.workers):
                wk.release_isa_reg()
        elif qk_compact:
            self.release_isa_reg()
            self.release_isa_reg()
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

    # Phases that stay on core 0 whatever --multi-core says: the strided-DMA
    # permutes, the pad-lane trim, and the merger tail. RoPE is NOT here -- it is
    # outside multi_engine_shard.SHARDED_OP_ALLOWLIST but still row-independent,
    # so it is sharded through ctx.unsafe_ue like gemma4_e2b does.
    # These bound what sharding can win, and their utilisation must be measured
    # against ONE core's peak, not the aggregate.
    _VIS_SERIAL_PHASES = ("permute_qkv", "unpermute+trim", "merger")

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

    def _base_run_vision_encoder(self, timeout_s: float = 600.0,
                           profile: bool = False) -> torch.Tensor:
        """Upload the queued inputs and the program, execute, and return the
        [NUM_MERGED_TOKENS, out_hidden_size] embeddings in raster order."""
        if not hasattr(self, "_vis_program_bytes"):
            raise RuntimeError("compile_vision_encoder() must run first")
        if not math.isfinite(float(timeout_s)) or timeout_s <= 0:
            raise ValueError(f"timeout_s must be finite and positive, got {timeout_s!r}")
        vis = self._vision_dims()
        T, VH_OUT = vis["NUM_MERGED_TOKENS"], vis["VH_OUT"]

        for addr, tensor in self._vis_pending_dmas:
            self.dma_to_accelerator_memory(addr, tensor)

        addr = self._vis_program_addr
        self._next_program_dram_addr = addr
        written = self.dma_write(
            DMA_DEVICE_H2C, addr, self._vis_program_bytes,
            len(self._vis_program_bytes))
        if written != len(self._vis_program_bytes):
            raise IOError(
                f"vision master ISA DMA wrote {written} of "
                f"{len(self._vis_program_bytes)} bytes")
        self.allocate_program_dram(len(self._vis_program_bytes))

        sched = self._ensure_stage_scheduler("vision")
        worker_addrs = []
        for idx, wk, wk_addr, blob in getattr(self, "_vis_worker_programs", []):
            wk._next_program_dram_addr = wk_addr
            written = wk.dma_write(DMA_DEVICE_H2C, wk_addr, blob, len(blob))
            if written != len(blob):
                raise IOError(
                    f"vision worker {idx} ISA DMA wrote {written} of "
                    f"{len(blob)} bytes")
            wk.allocate_program_dram(len(blob))
            worker_addrs.append(wk_addr)
        if sched is not None and not sched.host_segmented:
            # A flag left set by an earlier program would make the first CHECK
            # pass spuriously, so clear every engine's flag before the run.
            sched.preclear_flags()

        self._loud(f"  [Vision] launching encoder "
                   f"({len(self._vis_program_bytes) / 2**20:.2f} MiB) at 0x{addr:X}"
                   f"{' [profiled]' if profile else ''} ...")
        t0 = time.perf_counter()
        if profile:
            if sched is not None and sched.host_segmented:
                raise RuntimeError(
                    "profile checkpoints cannot be combined with the installed "
                    "FPGA build's host-segmented rendezvous")
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
                if self.is_queue_busy():
                    raise TimeoutError(
                        f"vision profile phase {name} is still busy after "
                        f"{timeout_s:.1f}s")
                results.append((name, self.report_latency_in_us() / 1e3, ph_flops))
                self.start_execute_from_dram(resume)
            self.wait_queue(timeout_s)
            if self.is_queue_busy():
                raise TimeoutError(
                    f"vision profile tail is still busy after {timeout_s:.1f}s")
            for idx, wk in enumerate(
                sched.workers if sched is not None else [], start=1
            ):
                wk.wait_queue(timeout_s)
                if wk.is_queue_busy():
                    raise TimeoutError(
                        f"vision worker {idx} is still busy after {timeout_s:.1f}s")
            latency_us = sum(r[1] for r in results) * 1e3
            self._vis_profile = results
        else:
            # Workers first: they park on their first rendezvous and wait for
            # the master to enter the region.
            if sched is not None and sched.host_segmented:
                latency_us = sched.run_host_segmented(
                    addr, worker_addrs, timeout_seconds=timeout_s)
            elif sched is not None:
                sched.start_workers(worker_addrs)
            if sched is None or not sched.host_segmented:
                self.start_execute_from_dram(addr)
                self.wait_queue(timeout_s)
                if self.is_queue_busy():
                    raise TimeoutError(
                        f"vision master is still busy after {timeout_s:.1f}s")
                for idx, wk in enumerate(
                    sched.workers if sched is not None else [], start=1
                ):
                    wk.wait_queue(timeout_s)
                    if wk.is_queue_busy():
                        raise TimeoutError(
                            f"vision worker {idx} is still busy after "
                            f"{timeout_s:.1f}s")
                latency_us = self.report_latency_in_us()
        wall = time.perf_counter() - t0
        gflops = (self._vis_total_flops / (latency_us * 1e-6) / 1e9
                  if latency_us > 0 else 0.0)
        self._vis_latency_us = latency_us
        self._vis_gflops = gflops
        self._vis_wall_s = wall
        self._loud(f"  [Vision] done: {wall:.2f}s wall, {latency_us / 1e6:.2f}s HW, "
                   f"{gflops:.1f} GFLOPS")

        out = torch.zeros(T * VH_OUT, dtype=torch.bfloat16)
        read = self.dma_read(
            DMA_DEVICE_C2H, self.VIS_ENCODER_OUT, out, out.numel() * 2)
        if read != out.numel() * 2:
            raise IOError(
                f"vision output DMA read {read} of {out.numel() * 2} bytes")
        out = out.reshape(T, VH_OUT).cpu()
        # Undo the window reordering so the tokens are back in raster order,
        # which is what the LM prompt expects at the <|image_pad|> positions.
        if hasattr(self, "_vis_reverse_index"):
            out = out[self._vis_reverse_index].contiguous()
        self._vis_embeddings = out
        self._vis_num_tokens = T
        # The encoder ping-pongs through its original input buffers. Retain the
        # pristine host copies until output validation succeeds so a failed run
        # can re-upload them before retrying on this same engine instance.
        self._vis_pending_dmas = []
        self._loud(f"  [Vision] {T} image embeddings ready")
        return out

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
        self._base_vision_weight_init()
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
        self._base_vision_tensor_init()
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
        encoder_addr = self._base_compile_vision_encoder(profile=profile)
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
            output = self._base_run_vision_encoder(timeout_s=timeout_s, profile=profile)
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

__all__ = ["Qwen25OmniVisionMixin", "VISION_QUANT_PRECISION"]
