#!/usr/bin/env python3
"""Gemma4 E2B vision-encoder method group (SigLIP-style tower; FPGA + host tail).

Split out of gemma4_e2b_test.py. ``Gemma4VisionMixin`` carries the vision
methods and is mixed into ``Gemma4_UnifiedEngine`` there; it is never
instantiated on its own. All shared plumbing -- ``self._set_silent`` /
``self._loud`` / ``self._print_phase_breakdown`` and the HF model on
``self.hf_model`` -- resolves through the concrete class, so this module imports
nothing from gemma4_e2b_test (keeps the 3-file split import-cycle free).
"""
import builtins
import gc
import json
import math
import os
import sys
import time

_SD = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SD)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SD)))

import torch
import torch.nn.functional as F
import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, DRAM_INSTRUCTION_ADDR, TYPE, UE_FMAX_CONTEXT_SIZE,
    UE_VECTOR_SIZE, UE_ARGMAX_INDEX, URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS,
    URAM_NEAR_FULL_SIZE, URAM_START_ADDR, URAM_SECTION, set_dma_device,
    ue_35bit_addr_shifter, INSTRUCTION_SIZE_BYTES, UE_MODE)

# Vision matmuls are IF4-quantized (norms stay BF16). Kept module-level so a
# numeric harness can import it; gemma4_e2b_test.py imports it back for the
# combined-bin generator.
VISION_QUANT_PRECISION = "if4"

# Fixed canonical vision input: every image is resized to VISION_CANONICAL_SIZE
# before the HF processor so num_patches is always VISION_FIXED_NUM_PATCHES, which
# lets the vision encoder bin be compiled once for a fixed shape.
VISION_CANONICAL_SIZE = (896, 896)   # (width, height) for PIL.Image.resize
VISION_FIXED_NUM_PATCHES = 2520


class Gemma4VisionMixin:
    """Vision-encoder methods for Gemma4_UnifiedEngine (see module docstring)."""


    # Vision encoder (SigLIP-style tower) constants — Gemma4 E2B.
    VIS_H = 768          # hidden size
    VIS_HEADS = 12       # attention heads
    VIS_HEAD_DIM = 64    # head_dim = 768 / 12
    VIS_MLP = 3072       # intermediate size
    VIS_LAYERS = 16      # num_hidden_layers
    VIS_ROPE_DIM = 32    # half of head_dim for 2D RoPE (64 / 2)

    def _ensure_vision_gate_scheduler(self):
        """Return the shared multicore configuration for the vision stage."""
        return self._ensure_stage_scheduler(
            "vision", self.VISION_WORKER_ISA_BASE)


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
            # patch-embedding position gather). Allocation order below mirrors
            # the bin builder.
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

    def vision_weight_init(self) -> None:
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

    def vision_tensor_init(self, *, program_base: int | None = None) -> None:
        """Allocate DRAM for vision encoder intermediate tensors. Reuses the LM
        tensor DRAM region (vision and LM never run simultaneously). Vision ISA
        goes to the engine's vision ISA base unless ``program_base`` overrides it."""
        bpe = self.bytes_per_element
        H, HD, NH, MLP = self.VIS_H, self.VIS_HEAD_DIM, self.VIS_HEADS, self.VIS_MLP
        S = self.vision_num_patches
        hf_model = self.vision_hf_model
        pixel_position_ids = self.vision_pixel_position_ids
        padding_positions = self.vision_padding_positions

        # Reuse only the SCRATCH sub-region of the tensor space. Persistent LM
        # tensors (KV cache, constants, attention bias, per-layer injection) live
        # below _scratch_dram_base and must NOT be clobbered by vision scratch
        # (incl. the pooler's VIS_POOL_P/OUT). Falls back to _tensor_base if the
        # LM tensor_init has not run (should never happen in practice).
        self._tensor_dram_addr = getattr(self, "_scratch_dram_base", self._tensor_dram_base)
        # Deferred DMAs: this method only ALLOCATES + builds host tensors; the
        # actual uploads are queued here and flushed at run time by
        # execute_vision_encoder_bin (right before the encoder launches).
        self._vis_pending_dmas: list[tuple[int, torch.Tensor]] = []
        base = self.VISION_ISA_BASE if program_base is None else program_base
        self._next_program_dram_addr = base
        self._program_dram_base = base
        print(f"\n[Vision] Allocating vision tensor DRAM for {S} patches ...")

        # Layer I/O (double-buffered). Patch embedding writes directly to IO_A.
        self.VIS_IO_A = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_IO_B = self.allocate_tensor_dram(S * H * bpe)

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

        # Patch-embed runtime inputs. Host does only the input scaling and
        # position-table gather; projection and positional addition are emitted
        # at the beginning of the single vision FPGA program.
        pixel_values = self.vision_pixel_values
        pv = (pixel_values.squeeze(0)
              if pixel_values.dim() == 3 and pixel_values.shape[0] == 1
              else pixel_values)
        assert pv.shape == (S, H), f"pixels shape {pv.shape}, expected ({S}, {H})"
        scaled = (2.0 * (pv.float() - 0.5)).to(torch.bfloat16).contiguous()
        self._vis_pending_dmas.append((self.VIS_IO_B, scaled))

        pids = (pixel_position_ids.squeeze(0)
                if pixel_position_ids.dim() == 3 else pixel_position_ids)
        pad_rows = (padding_positions.squeeze(0)
                    if padding_positions.dim() == 2 else padding_positions)
        table = self._vis_pos_embed_table.float()
        clamped = pids.clamp(min=0).long().cpu()
        pe_sum = table[0, clamped[:, 0]] + table[1, clamped[:, 1]]
        pe_sum[pad_rows.cpu()] = 0.0
        self._vis_pending_dmas.append(
            (self.VIS_NORM_OUT, pe_sum.to(torch.bfloat16).contiguous()))

        # Per-head attention operates in place on the head-major buffers
        # (VIS_FLASH_*_HM below) at each head's offset — no fixed staging buffers.
        aligned_S = ((S + 63) // 64) * 64
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
        self._vis_pending_dmas.append((self.VIS_IDENTITY_64, torch.eye(64, dtype=torch.bfloat16)))
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
            self._vis_pending_dmas.append((a, zeros_hm))

        # Bidirectional attention bias: mask both real image-padding patches
        # (position_ids == -1) and alignment padding at the end.
        pad = (padding_positions[0] if padding_positions.dim() == 2
               else padding_positions).detach().cpu().bool()
        assert pad.shape[0] == S, (
            f"padding mask has {pad.shape[0]} entries, expected {S}")
        self._vis_padding_mask = pad
        col_mask = torch.zeros(aligned_S, dtype=torch.bool)
        col_mask[:S] = pad
        col_mask[S:] = True
        bias = torch.zeros(aligned_S, aligned_S, dtype=torch.bfloat16)
        bias[:, col_mask] = float("-inf")
        self._vis_pending_dmas.append((self.VIS_FLASH_BIAS, bias))

        # Legacy RoPE tables (kept for address stability).
        self.VIS_ROPE_COS = self.allocate_tensor_dram(S * HD * bpe)
        self.VIS_ROPE_SIN = self.allocate_tensor_dram(S * HD * bpe)
        # Three 32-wide (HD//2) RoPE tables for the FPGA 2D-RoPE split-64 path.
        self.VIS_ROPE_COS_PAD_TILED     = self.allocate_tensor_dram(S * (HD // 2) * bpe)
        self.VIS_ROPE_NEG_SIN_PAD_TILED = self.allocate_tensor_dram(S * (HD // 2) * bpe)
        self.VIS_ROPE_SIN_HI_PAD_TILED  = self.allocate_tensor_dram(S * (HD // 2) * bpe)

        # Build the image-specific FPGA-ready 2D-RoPE tables now that their
        # tensor addresses exist. One 32-wide row per patch is shared by all
        # vision heads.
        print("  [Vision] [host] computing 2D-RoPE pads...", flush=True)
        rope_t0 = time.perf_counter()
        with torch.no_grad():
            vision_tower = hf_model.model.vision_tower
            patch_embeds_dummy = torch.zeros(
                1, S, vision_tower.config.hidden_size,
                dtype=torch.bfloat16, device=pixel_position_ids.device)
            cos_table, sin_table = vision_tower.encoder.rotary_emb(
                patch_embeds_dummy, pixel_position_ids)
        cos_2d = cos_table.squeeze(0).cpu().to(torch.bfloat16)
        sin_2d = sin_table.squeeze(0).cpu().to(torch.bfloat16)
        first_half_idx = self.VIS_ROPE_PERM[:HD // 2]
        cos_p = cos_2d[:, first_half_idx].contiguous()
        neg_sin_p = (-sin_2d[:, first_half_idx]).contiguous()
        sin_hi_p = sin_2d[:, first_half_idx].contiguous()
        self._vis_pending_dmas.append((self.VIS_ROPE_COS_PAD_TILED, cos_p))
        self._vis_pending_dmas.append((self.VIS_ROPE_NEG_SIN_PAD_TILED, neg_sin_p))
        self._vis_pending_dmas.append((self.VIS_ROPE_SIN_HI_PAD_TILED, sin_hi_p))
        print(f"  [Vision] [host] 2D-RoPE pads computed in "
              f"{time.perf_counter()-rope_t0:.2f}s (per-patch 32-wide)", flush=True)

        # HD×HD identity for transpose/matmul helpers.
        self.VIS_IDENTITY = self.allocate_tensor_dram(HD * HD * bpe)
        self._vis_pending_dmas.append((self.VIS_IDENTITY, torch.eye(HD, dtype=torch.bfloat16)))

        # Embed-vision (pooler tail) scratch. N_soft is image-dependent but
        # bounded by S; size at S to be safe.
        text_h = getattr(self, "VIS_TEXT_H", 1536)
        self.VIS_EMBED_POOL = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_EMBED_NORMED = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_EMBED_OUT = self.allocate_tensor_dram(S * text_h * bpe)

        # ---- Pooler tail on FPGA (spatial average-pool + embed_vision) ----
        # The host used to average-pool the [S, H] encoder output down to N_SOFT
        # soft tokens, then run embed_vision (RMSNorm + 768->text_h projection).
        # We now do all of that on FPGA. The k x k spatial average is expressed as
        # a matmul P[N_SOFT, nV] @ hiddenᵀ[H, nV] so the MAC array accumulates each
        # cell sum in its bf20-wide accumulator (see compile_vision_encoder_bin).
        # For the fixed canonical input the valid patches form a Wg x Hg raster
        # grid (row = grid_row*Wg + col) followed by trailing zero-padding, so cell
        # (cy,cx) sums the k*k rows {b, b+1, b+2, b+Wg, .., b+2Wg+2}, b = k*cx+k*Wg*cy.
        pool_k = self.VIS_POOL_K
        pool_pids = (pixel_position_ids.squeeze(0)
                     if pixel_position_ids.dim() == 3 else pixel_position_ids)
        pool_pad = (padding_positions.squeeze(0)
                    if padding_positions.dim() == 2 else padding_positions).cpu().bool()
        clamped = pool_pids.clamp(min=0).long().cpu()
        Wg = int(clamped[:, 0].max()) + 1                 # valid grid width (48)
        Hg = int(clamped[:, 1].max()) + 1                 # valid grid height (48)
        n_valid = Wg * Hg
        # The pooling matrix assumes the canonical raster layout: the first n_valid
        # rows are the grid in row-major order, the rest are padding.
        raster = torch.arange(n_valid)
        layout_ok = (not bool(pool_pad[:n_valid].any())
                     and bool(pool_pad[n_valid:].all())
                     and bool((clamped[:n_valid, 1] * Wg + clamped[:n_valid, 0]
                               == raster).all()))
        assert layout_ok, ("vision pooler expects a raster grid + trailing padding; "
                           "got a different patch layout.")
        self.VIS_POOL_WG = Wg
        self.VIS_POOL_NVALID = n_valid                    # valid grid patches (48*48=2304)
        self.VIS_POOL_OUT_W = Wg // pool_k                # 16 output cols
        self.VIS_POOL_OUT_H = Hg // pool_k                # 16 output cell-rows
        self.VIS_POOL_N_SOFT = self.VIS_POOL_OUT_W * self.VIS_POOL_OUT_H   # 256
        # P[N_SOFT, nV]: sparse pooling matrix, weight 1.0 at each cell's k*k patch
        # positions; VIS_HIDDEN_T holds the transposed valid grid [H, nV] the matmul
        # contracts against; VIS_POOL_OUT holds the pooled soft tokens. Weight = 1.0
        # (cell SUM), not 1/k^2: the 1/k^2 scale is cancelled by the scale-invariant
        # RMSNorm that follows, and bf16(1/9)'s +0.2% quant error pushes outlier-
        # channel values across bf16 boundaries (tips greedy decode); 1.0 is exact.
        N_SOFT = self.VIS_POOL_N_SOFT
        self.VIS_POOL_OUT = self.allocate_tensor_dram(N_SOFT * H * bpe)
        self.VIS_POOL_P   = self.allocate_tensor_dram(N_SOFT * n_valid * bpe)
        self.VIS_HIDDEN_T = self.allocate_tensor_dram(H * n_valid * bpe)
        Pm = torch.zeros(N_SOFT, n_valid, dtype=torch.bfloat16)
        for cy in range(self.VIS_POOL_OUT_H):
            for cx in range(self.VIS_POOL_OUT_W):
                base = pool_k * (Wg * cy + cx)
                cell = self.VIS_POOL_OUT_W * cy + cx
                for ky in range(pool_k):
                    for kx in range(pool_k):
                        Pm[cell, base + ky * Wg + kx] = 1.0
        self._vis_pending_dmas.append((self.VIS_POOL_P, Pm))

        if self.get_tensor_dram_addr() > self.VISION_ISA_BASE:
            raise MemoryError(
                f"Gemma4 vision tensors overflow the 2-GB layout: "
                f"end=0x{self.get_tensor_dram_addr():X}, "
                f"vision_program_start=0x{self.VISION_ISA_BASE:X}")

        self._vis_num_patches = S
        self._vis_aligned_S = aligned_S
        print(f"  Vision tensor DRAM allocated. Total usage: {self.get_tensor_dram_usage()} bytes")

    # ------------------------------------------------------------------
    # Vision execution infra (S2): patch embedding and the 16-layer encoder are
    # captured as one program; pooler-tail helpers remain standalone.
    # ------------------------------------------------------------------
    def _prime_M(self, M: int) -> int:
        """Emit ADD_SET gpr_seq_len <- M and return that register index, for use
        as gpr_M_reg on a vision matmul / rms_norm. Folds the compile-time M
        tiling into one runtime ISA loop (bit-exact to the legacy static unroll).
        gpr_seq_len is LM-only, free in the vision program. Emits an instruction,
        so it must be evaluated inside the active capture."""
        self.generate_instruction_add_set(self.gpr_seq_len, M)
        return self.gpr_seq_len

    # ------------------------------------------------------------------
    # Vision encoder ISA primitives (S2b/S3): 2-D RoPE, Q/K/V head-major
    # transposes, PBI rms-norm, FPGA clamp. Ported from gemma4_e2b_test.py —
    # these are validated standalone (see test.py's test_vision_* helpers).
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Vision encoder layers (S3): one SigLIP layer = pre_norm + Q/K/V proj +
    # Q/K norm (part A) → V-norm + 2D RoPE + head-major transpose → per-head
    # attention → O proj + post-attn norm + residual + MLP (part C).
    # Every Gemma4ClippableLinear clips input (FPGA clamp into scratch) and
    # output (fused clamp). Captured one-shot into the vision encoder bin.
    # ------------------------------------------------------------------
    def compile_vision_encoder_bin(self, num_patches: int, profile: bool = False) -> None:
        """Capture patch embedding plus the 16-layer vision encoder as one
        FPGA program at the engine's VISION_ISA_BASE. Pure host emission, no FPGA activity.
        Skips if the cached vision section exists — delete
        gemma4_e2b_bin/programs*.bin (make clean) after changing the FPGA compile
        so it recaptures fresh.

        ``profile``: emit a HALT at each major per-layer boundary (proj /
        rope_gather / attention / post_attn) and record the resume address, so
        execute_vision_encoder_bin can time each phase's FPGA latency (like the
        LM per-phase profile). The checkpoints go in the meta; a profile-compiled
        bin can only be run segment-by-segment."""
        L = self.VIS_LAYERS
        _vmeta, _ = self._get_program_section("vision", profile)
        _worker_meta, _ = self._get_program_section("vision_worker1", profile)
        if (_vmeta is not None
                and _vmeta.get("vision_kernel") == self.vision_kernel
                and _vmeta.get("multi_core", 1) == self.multi_core
                and (self.multi_core == 1 or _worker_meta is not None)):
            self._ensure_vision_gate_scheduler()
            bin_path, _ = self._program_image_paths(profile)
            print(f"  [Vision] reusing existing instruction image at {bin_path}", flush=True)
            print(f"    delete {bin_path} (or make clean) to force recompile.", flush=True)
            return
        print(f"  [Vision] compiling patch embed + {L} vision layers one-shot"
              f" ({self.multi_core} engine(s); projection phases row-sharded)"
              f"{' (+profile checkpoints)' if profile else ''} ...", flush=True)
        t0 = time.perf_counter()
        enc_checkpoints: list[list] = []
        def _checkpoint(name: str) -> None:
            if not profile:
                return
            self.generate_instruction_halt()
            resume = self.get_program_dram_addr() + self.capture_count * INSTRUCTION_SIZE_BYTES
            enc_checkpoints.append([name, f"0x{resume:X}"])
        _prev_silent = self._set_silent(False)
        self.reset_program_dram_addr()
        base_addr = self.get_program_dram_addr()
        self.clear_inst_id()
        self.clear_capture_buffer()
        self.start_capture()
        gate_scheduler = self._ensure_vision_gate_scheduler()
        gate_m_regs = None
        if gate_scheduler is not None:
            gate_scheduler.begin_program()
            # Engine 0 uses Gemma's reserved sequence-length GPR. Plain
            # UnifiedEngine workers do not have that model-level attribute, so
            # allocate one worker GPR after begin_program() resets its allocator.
            gate_m_regs = [self.gpr_seq_len]
            gate_m_regs.extend(worker.alloc_isa_reg()
                               for worker in gate_scheduler.workers)
        self._vis_flops = 0
        self._set_silent(True)

        # Emit each layer in hardware execution order. Reset the instruction
        # pointer and ISA-register allocator before every PBI core invocation.
        S, H, HD, NH, MLP = (self._vis_num_patches, self.VIS_H, self.VIS_HEAD_DIM,
                             self.VIS_HEADS, self.VIS_MLP)
        aligned_S, bpe = self._vis_aligned_S, self.bytes_per_element

        def _projection_core(**kwargs) -> int:
            """Dynamic IF4 projection selected for the entire vision stage."""
            if self.vision_kernel == "matmatmul":
                kwargs.setdefault("is_B_quantized", True)
                return self.matmat_mul_core(**kwargs)
            kwargs.pop("is_B_quantized", None)
            return self.quantized_matmat_core(**kwargs)

        # Patch embedder: IF4 input projection followed by the gathered 2D
        # positional embedding add. Both runtime inputs were queued by
        # vision_tensor_init; no separate FPGA launch is needed.
        self._vis_flops += _projection_core(M=S, K=H, N=H, A_DRAM_ADDR=self.VIS_IO_B, B_DRAM_ADDR=self.VIS_PATCH_PROJ_INFO['data'], OUTPUT_DRAM_ADDR=self.VIS_IO_A, is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=self.VIS_PATCH_PROJ_INFO['scale'], gelu_enable=False, clamp_enable=(None is not None), clamp_min=(0.0 if None is None else None), clamp_max=(float('inf') if None is None else None), gpr_M_reg=self._prime_M(S))
        self._vis_flops += self.eltwise_core_dram(M=S, N=S * H // S, dram_a=self.VIS_IO_A, dram_b=self.VIS_NORM_OUT, dram_out=self.VIS_IO_A, mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=self._prime_M(S))
        _checkpoint("patch_embed")
        for li in range(L):
            w, clips = self._vis_weight_addrs[li], self._vis_clip_ranges[li]
            IN = self.VIS_IO_A if li % 2 == 0 else self.VIS_IO_B
            # q/k/v share one input clip range -> clamp VIS_NORM_OUT once.
            qkv_in = [clips[p]["input"] for p in ("q_proj", "k_proj", "v_proj")]
            share = all(x == qkv_in[0] for x in qkv_in)
            if gate_scheduler is None:
                self._vis_flops += self.rms_norm_core_dram(M=S, N=H, A_DRAM_ADDR=IN, OUTPUT_DRAM_ADDR=self.VIS_NORM_OUT, GAMMA_DRAM_ADDR=w['input_layernorm'], gpr_M_reg=self._prime_M(S))
                if share:
                    self._vis_flops += self.activation_core(M=S * H // 64, N=64, A_DRAM_ADDR=self.VIS_NORM_OUT, OUTPUT_DRAM_ADDR=self.VIS_INPUT_CLIP_H_SCRATCH, IDENTITY_DRAM_ADDR=self.VIS_IDENTITY_64, activation="clamp", clamp_min=qkv_in[0][0], clamp_max=qkv_in[0][1], gpr_M_reg=self._prime_M(S * H // 64))
                for proj, out in (("q_proj", self.VIS_Q_DRAM), ("k_proj", self.VIS_K_DRAM),
                                  ("v_proj", self.VIS_V_DRAM)):
                    c = clips[proj]
                    if not share:
                        self._vis_flops += self.activation_core(M=S * H // 64, N=64, A_DRAM_ADDR=self.VIS_NORM_OUT, OUTPUT_DRAM_ADDR=self.VIS_INPUT_CLIP_H_SCRATCH, IDENTITY_DRAM_ADDR=self.VIS_IDENTITY_64, activation="clamp", clamp_min=c['input'][0], clamp_max=c['input'][1], gpr_M_reg=self._prime_M(S * H // 64))
                    self._vis_flops += _projection_core(M=S, K=H, N=NH * HD, A_DRAM_ADDR=self.VIS_INPUT_CLIP_H_SCRATCH, B_DRAM_ADDR=w[proj]['data'], OUTPUT_DRAM_ADDR=out, is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=w[proj]['scale'], gelu_enable=False, clamp_enable=(c['output'][0] is not None), clamp_min=(0.0 if c['output'][0] is None else c['output'][0]), clamp_max=(float('inf') if c['output'][1] is None else c['output'][1]), gpr_M_reg=self._prime_M(S))
                self._vis_flops += self.rms_norm_core_dram(M=S * NH, N=HD, A_DRAM_ADDR=self.VIS_Q_DRAM, OUTPUT_DRAM_ADDR=self.VIS_Q_NORM, GAMMA_DRAM_ADDR=w['q_norm'], gpr_M_reg=self._prime_M(S * NH))
                self._vis_flops += self.rms_norm_core_dram(M=S * NH, N=HD, A_DRAM_ADDR=self.VIS_K_DRAM, OUTPUT_DRAM_ADDR=self.VIS_K_NORM, GAMMA_DRAM_ADDR=w['k_norm'], gpr_M_reg=self._prime_M(S * NH))
            else:
                # Complete pre-attention projection phase only. RoPE, layout
                # permutation, and attention remain outside this region.
                shard_flops = [0]
                def _emit_projection_shard(ctx):
                    m_reg = gate_m_regs[ctx.engine_idx]
                    clamp_rows = ctx.rows * H // 64
                    qk_norm_rows = ctx.rows * NH
                    ctx.ue.generate_instruction_add_set(m_reg, ctx.rows)
                    shard_flops[0] += ctx.ue.rms_norm_core_dram(M=ctx.rows, N=H, A_DRAM_ADDR=ctx.rows_addr(IN, H * bpe), OUTPUT_DRAM_ADDR=ctx.rows_addr(self.VIS_NORM_OUT, H * bpe), GAMMA_DRAM_ADDR=w['input_layernorm'], gpr_M_reg=m_reg)
                    if share:
                        ctx.ue.generate_instruction_add_set(m_reg, clamp_rows)
                        shard_flops[0] += ctx.ue.activation_core(M=clamp_rows, N=64, A_DRAM_ADDR=ctx.rows_addr(self.VIS_NORM_OUT, H * bpe), OUTPUT_DRAM_ADDR=ctx.rows_addr(self.VIS_INPUT_CLIP_H_SCRATCH, H * bpe), IDENTITY_DRAM_ADDR=self.VIS_IDENTITY_64, activation="clamp", clamp_min=qkv_in[0][0], clamp_max=qkv_in[0][1], gpr_M_reg=m_reg)
                    for proj, out in (("q_proj", self.VIS_Q_DRAM), ("k_proj", self.VIS_K_DRAM),
                                      ("v_proj", self.VIS_V_DRAM)):
                        c = clips[proj]
                        if not share:
                            ctx.ue.generate_instruction_add_set(m_reg, clamp_rows)
                            shard_flops[0] += ctx.ue.activation_core(M=clamp_rows, N=64, A_DRAM_ADDR=ctx.rows_addr(self.VIS_NORM_OUT, H * bpe), OUTPUT_DRAM_ADDR=ctx.rows_addr(self.VIS_INPUT_CLIP_H_SCRATCH, H * bpe), IDENTITY_DRAM_ADDR=self.VIS_IDENTITY_64, activation="clamp", clamp_min=c['input'][0], clamp_max=c['input'][1], gpr_M_reg=m_reg)
                        ctx.ue.generate_instruction_add_set(m_reg, ctx.rows)
                        shard_flops[0] += ctx.ue.matmat_mul_core(M=ctx.rows, K=H, N=NH * HD, A_DRAM_ADDR=ctx.rows_addr(self.VIS_INPUT_CLIP_H_SCRATCH, H * bpe), B_DRAM_ADDR=w[proj]['data'], OUTPUT_DRAM_ADDR=ctx.rows_addr(out, NH * HD * bpe), is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=w[proj]['scale'], gelu_enable=False, clamp_enable=(c['output'][0] is not None), clamp_min=(0.0 if c['output'][0] is None else c['output'][0]), clamp_max=(float('inf') if c['output'][1] is None else c['output'][1]), gpr_M_reg=m_reg)
                    ctx.ue.generate_instruction_add_set(m_reg, qk_norm_rows)
                    shard_flops[0] += ctx.ue.rms_norm_core_dram(M=qk_norm_rows, N=HD, A_DRAM_ADDR=ctx.rows_addr(self.VIS_Q_DRAM, NH * HD * bpe), OUTPUT_DRAM_ADDR=ctx.rows_addr(self.VIS_Q_NORM, NH * HD * bpe), GAMMA_DRAM_ADDR=w['q_norm'], gpr_M_reg=m_reg)
                    ctx.ue.generate_instruction_add_set(m_reg, qk_norm_rows)
                    shard_flops[0] += ctx.ue.rms_norm_core_dram(M=qk_norm_rows, N=HD, A_DRAM_ADDR=ctx.rows_addr(self.VIS_K_DRAM, NH * HD * bpe), OUTPUT_DRAM_ADDR=ctx.rows_addr(self.VIS_K_NORM, NH * HD * bpe), GAMMA_DRAM_ADDR=w['k_norm'], gpr_M_reg=m_reg)
                gate_scheduler.sharded_region(S, _emit_projection_shard)
                self._vis_flops += shard_flops[0]
            _checkpoint(f"L{li}_proj")
            self._vis_flops += self.rms_norm_core_dram(M=S * NH, N=HD, A_DRAM_ADDR=self.VIS_V_DRAM, OUTPUT_DRAM_ADDR=self.VIS_V_DRAM, GAMMA_DRAM_ADDR=self.VIS_V_NORM_ONES_GAMMA, gpr_M_reg=self._prime_M(S * NH))
            for dram in (self.VIS_Q_NORM, self.VIS_K_NORM):
                self._emit_vision_rope_2d(src_dram=dram, out_dram=dram, cos_pad_dram=self.VIS_ROPE_COS_PAD_TILED, neg_sin_pad_dram=self.VIS_ROPE_NEG_SIN_PAD_TILED, sin_hi_pad_dram=self.VIS_ROPE_SIN_HI_PAD_TILED, M=S * NH)
            for src, dst in ((self.VIS_Q_NORM, self.VIS_FLASH_Q_HM),
                             (self.VIS_K_NORM, self.VIS_FLASH_K_HM),
                             (self.VIS_V_DRAM, self.VIS_FLASH_V_HM)):
                # interleaved [S, NH, HD] -> head-major [NH, aligned_S, HD]
                self.bf16_permute_dram_core(NH, S, HD, src, dst,
                                            write_grouped=True, group_stride_rows=aligned_S)
            _checkpoint(f"L{li}_rope_gather")
            # Per-head attention reads Q/K/V and writes OUT DIRECTLY at each head's
            # offset in the head-major buffers (no per-head staging copy — the core
            # takes arbitrary DRAM addresses).
            head_stride = aligned_S * HD * bpe
            for h in range(NH):
                base = h * head_stride
                batch_reg = self.alloc_isa_reg()
                self.generate_instruction_add_set(batch_reg, aligned_S)
                aligned_seq_reg = self.alloc_isa_reg()
                self.generate_instruction_add_set(aligned_seq_reg, aligned_S)
                flops = self.unified_attention_core(
                    batch=aligned_S, aligned_seq_len=aligned_S, head_dim=HD,
                    Q_DRAM_ADDR=self.VIS_FLASH_Q_HM + base, K_DRAM_ADDR=self.VIS_FLASH_K_HM + base,
                    V_DRAM_ADDR=self.VIS_FLASH_V_HM + base, BIAS_DRAM_ADDR=self.VIS_FLASH_BIAS,
                    OUTPUT_DRAM_ADDR=self.VIS_FLASH_OUT_HM + base,
                    SCRATCH_DRAM_ADDR=self.VIS_FLASH_SCRATCH,
                    IDENTITY_DRAM_ADDR=self._vis_identity_dram,
                    gpr_batch_reg=batch_reg,
                    gpr_aligned_seq_len_reg=aligned_seq_reg, q_scale=1.0)
                self.release_isa_reg()
                self.release_isa_reg()
                if isinstance(flops, (int, float)):
                    self._vis_flops += flops
            # head-major attn output [NH, aligned_S, HD] -> interleaved VIS_Q_DRAM [S, NH, HD]
            self.bf16_permute_dram_core(NH, S, HD, self.VIS_FLASH_OUT_HM, self.VIS_Q_DRAM,
                                        write_grouped=False, group_stride_rows=aligned_S)
            _checkpoint(f"L{li}_attention")
            OUT = self.VIS_IO_B if li % 2 == 0 else self.VIS_IO_A
            co, cg, cu, cd = (clips["o_proj"], clips["gate_proj"],
                              clips["up_proj"], clips["down_proj"])
            if gate_scheduler is None:
                self._vis_flops += self.activation_core(M=S * NH * HD // 64, N=64, A_DRAM_ADDR=self.VIS_Q_DRAM, OUTPUT_DRAM_ADDR=self.VIS_INPUT_CLIP_H_SCRATCH, IDENTITY_DRAM_ADDR=self.VIS_IDENTITY_64, activation="clamp", clamp_min=co['input'][0], clamp_max=co['input'][1], gpr_M_reg=self._prime_M(S * NH * HD // 64))
                self._vis_flops += _projection_core(M=S, K=NH * HD, N=H, A_DRAM_ADDR=self.VIS_INPUT_CLIP_H_SCRATCH, B_DRAM_ADDR=w['o_proj']['data'], OUTPUT_DRAM_ADDR=self.VIS_ATTN_OUT, is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=w['o_proj']['scale'], gelu_enable=False, clamp_enable=(co['output'][0] is not None), clamp_min=(0.0 if co['output'][0] is None else co['output'][0]), clamp_max=(float('inf') if co['output'][1] is None else co['output'][1]), gpr_M_reg=self._prime_M(S))
                self._vis_flops += self.rms_norm_core_dram(M=S, N=H, A_DRAM_ADDR=self.VIS_ATTN_OUT, OUTPUT_DRAM_ADDR=self.VIS_POST_ATTN_NORM, GAMMA_DRAM_ADDR=w['post_attention_layernorm'], gpr_M_reg=self._prime_M(S))
                self._vis_flops += self.eltwise_core_dram(M=S, N=S * H // S, dram_a=IN, dram_b=self.VIS_POST_ATTN_NORM, dram_out=self.VIS_POST_ATTN_RES, mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=self._prime_M(S))
                self._vis_flops += self.rms_norm_core_dram(M=S, N=H, A_DRAM_ADDR=self.VIS_POST_ATTN_RES, OUTPUT_DRAM_ADDR=self.VIS_PRE_FFN_NORM, GAMMA_DRAM_ADDR=w['pre_feedforward_layernorm'], gpr_M_reg=self._prime_M(S))
                self._vis_flops += self.activation_core(M=S * H // 64, N=64, A_DRAM_ADDR=self.VIS_PRE_FFN_NORM, OUTPUT_DRAM_ADDR=self.VIS_INPUT_CLIP_H_SCRATCH, IDENTITY_DRAM_ADDR=self.VIS_IDENTITY_64, activation="clamp", clamp_min=cg['input'][0], clamp_max=cg['input'][1], gpr_M_reg=self._prime_M(S * H // 64))
                self._vis_flops += _projection_core(M=S, K=H, N=MLP, A_DRAM_ADDR=self.VIS_INPUT_CLIP_H_SCRATCH, B_DRAM_ADDR=w['gate_proj']['data'], OUTPUT_DRAM_ADDR=self.VIS_MLP_GATE, is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=w['gate_proj']['scale'], gelu_enable=True, clamp_enable=(None is not None), clamp_min=(0.0 if None is None else None), clamp_max=(float('inf') if None is None else None), gpr_M_reg=self._prime_M(S))
                self._vis_flops += self.activation_core(M=S * MLP // 64, N=64, A_DRAM_ADDR=self.VIS_MLP_GATE, OUTPUT_DRAM_ADDR=self.VIS_MLP_GATE, IDENTITY_DRAM_ADDR=self.VIS_IDENTITY_64, activation="clamp", clamp_min=cg['output'][0], clamp_max=cg['output'][1], gpr_M_reg=self._prime_M(S * MLP // 64))
                # up shares gate's input clamp (VIS_INPUT_CLIP_H_SCRATCH still valid).
                if cu["input"] != cg["input"]:
                    self._vis_flops += self.activation_core(M=S * H // 64, N=64, A_DRAM_ADDR=self.VIS_PRE_FFN_NORM, OUTPUT_DRAM_ADDR=self.VIS_INPUT_CLIP_H_SCRATCH, IDENTITY_DRAM_ADDR=self.VIS_IDENTITY_64, activation="clamp", clamp_min=cu['input'][0], clamp_max=cu['input'][1], gpr_M_reg=self._prime_M(S * H // 64))
                self._vis_flops += _projection_core(M=S, K=H, N=MLP, A_DRAM_ADDR=self.VIS_INPUT_CLIP_H_SCRATCH, B_DRAM_ADDR=w['up_proj']['data'], OUTPUT_DRAM_ADDR=self.VIS_MLP_UP, is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=w['up_proj']['scale'], gelu_enable=False, clamp_enable=(cu['output'][0] is not None), clamp_min=(0.0 if cu['output'][0] is None else cu['output'][0]), clamp_max=(float('inf') if cu['output'][1] is None else cu['output'][1]), gpr_M_reg=self._prime_M(S))
                self._vis_flops += self.eltwise_core_dram(M=S, N=S * MLP // S, dram_a=self.VIS_MLP_GATE, dram_b=self.VIS_MLP_UP, dram_out=self.VIS_MLP_MULT, mode=UE_MODE.ELTWISE_MUL, gpr_M_reg=self._prime_M(S))
                self._vis_flops += self.activation_core(M=S * MLP // 64, N=64, A_DRAM_ADDR=self.VIS_MLP_MULT, OUTPUT_DRAM_ADDR=self.VIS_INPUT_CLIP_MLP_SCRATCH, IDENTITY_DRAM_ADDR=self.VIS_IDENTITY_64, activation="clamp", clamp_min=cd['input'][0], clamp_max=cd['input'][1], gpr_M_reg=self._prime_M(S * MLP // 64))
                self._vis_flops += _projection_core(M=S, K=MLP, N=H, A_DRAM_ADDR=self.VIS_INPUT_CLIP_MLP_SCRATCH, B_DRAM_ADDR=w['down_proj']['data'], OUTPUT_DRAM_ADDR=self.VIS_MLP_DOWN, is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=w['down_proj']['scale'], gelu_enable=False, clamp_enable=(cd['output'][0] is not None), clamp_min=(0.0 if cd['output'][0] is None else cd['output'][0]), clamp_max=(float('inf') if cd['output'][1] is None else cd['output'][1]), gpr_M_reg=self._prime_M(S))
                self._vis_flops += self.rms_norm_core_dram(M=S, N=H, A_DRAM_ADDR=self.VIS_MLP_DOWN, OUTPUT_DRAM_ADDR=self.VIS_POST_FFN_NORM, GAMMA_DRAM_ADDR=w['post_feedforward_layernorm'], gpr_M_reg=self._prime_M(S))
                self._vis_flops += self.eltwise_core_dram(M=S, N=S * H // S, dram_a=self.VIS_POST_ATTN_RES, dram_b=self.VIS_POST_FFN_NORM, dram_out=OUT, mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=self._prime_M(S))
            else:
                # One row-sharded region covers the complete post-attention
                # projection/MLP phase. Dependencies are sequential within each
                # core's rows; only the region entry and exit need rendezvous.
                shard_flops = [0]
                def _emit_post_attention_shard(ctx):
                    m_reg = gate_m_regs[ctx.engine_idx]
                    clamp_rows_h = ctx.rows * H // 64
                    clamp_rows_mlp = ctx.rows * MLP // 64
                    ctx.ue.generate_instruction_add_set(m_reg, clamp_rows_h)
                    shard_flops[0] += ctx.ue.activation_core(M=clamp_rows_h, N=64, A_DRAM_ADDR=ctx.rows_addr(self.VIS_Q_DRAM, H * bpe), OUTPUT_DRAM_ADDR=ctx.rows_addr(self.VIS_INPUT_CLIP_H_SCRATCH, H * bpe), IDENTITY_DRAM_ADDR=self.VIS_IDENTITY_64, activation="clamp", clamp_min=co['input'][0], clamp_max=co['input'][1], gpr_M_reg=m_reg)
                    ctx.ue.generate_instruction_add_set(m_reg, ctx.rows)
                    shard_flops[0] += ctx.ue.matmat_mul_core(M=ctx.rows, K=NH * HD, N=H, A_DRAM_ADDR=ctx.rows_addr(self.VIS_INPUT_CLIP_H_SCRATCH, H * bpe), B_DRAM_ADDR=w['o_proj']['data'], OUTPUT_DRAM_ADDR=ctx.rows_addr(self.VIS_ATTN_OUT, H * bpe), is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=w['o_proj']['scale'], gelu_enable=False, clamp_enable=(co['output'][0] is not None), clamp_min=(0.0 if co['output'][0] is None else co['output'][0]), clamp_max=(float('inf') if co['output'][1] is None else co['output'][1]), gpr_M_reg=m_reg)
                    ctx.ue.generate_instruction_add_set(m_reg, ctx.rows)
                    shard_flops[0] += ctx.ue.rms_norm_core_dram(M=ctx.rows, N=H, A_DRAM_ADDR=ctx.rows_addr(self.VIS_ATTN_OUT, H * bpe), OUTPUT_DRAM_ADDR=ctx.rows_addr(self.VIS_POST_ATTN_NORM, H * bpe), GAMMA_DRAM_ADDR=w['post_attention_layernorm'], gpr_M_reg=m_reg)
                    ctx.ue.generate_instruction_add_set(m_reg, ctx.rows)
                    shard_flops[0] += ctx.ue.eltwise_core_dram(M=ctx.rows, N=H, dram_a=ctx.rows_addr(IN, H * bpe), dram_b=ctx.rows_addr(self.VIS_POST_ATTN_NORM, H * bpe), dram_out=ctx.rows_addr(self.VIS_POST_ATTN_RES, H * bpe), mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=m_reg)
                    ctx.ue.generate_instruction_add_set(m_reg, ctx.rows)
                    shard_flops[0] += ctx.ue.rms_norm_core_dram(M=ctx.rows, N=H, A_DRAM_ADDR=ctx.rows_addr(self.VIS_POST_ATTN_RES, H * bpe), OUTPUT_DRAM_ADDR=ctx.rows_addr(self.VIS_PRE_FFN_NORM, H * bpe), GAMMA_DRAM_ADDR=w['pre_feedforward_layernorm'], gpr_M_reg=m_reg)
                    ctx.ue.generate_instruction_add_set(m_reg, clamp_rows_h)
                    shard_flops[0] += ctx.ue.activation_core(M=clamp_rows_h, N=64, A_DRAM_ADDR=ctx.rows_addr(self.VIS_PRE_FFN_NORM, H * bpe), OUTPUT_DRAM_ADDR=ctx.rows_addr(self.VIS_INPUT_CLIP_H_SCRATCH, H * bpe), IDENTITY_DRAM_ADDR=self.VIS_IDENTITY_64, activation="clamp", clamp_min=cg['input'][0], clamp_max=cg['input'][1], gpr_M_reg=m_reg)
                    ctx.ue.generate_instruction_add_set(m_reg, ctx.rows)
                    shard_flops[0] += ctx.ue.matmat_mul_core(M=ctx.rows, K=H, N=MLP, A_DRAM_ADDR=ctx.rows_addr(self.VIS_INPUT_CLIP_H_SCRATCH, H * bpe), B_DRAM_ADDR=w['gate_proj']['data'], OUTPUT_DRAM_ADDR=ctx.rows_addr(self.VIS_MLP_GATE, MLP * bpe), is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=w['gate_proj']['scale'], gelu_enable=True, clamp_enable=False, clamp_min=0.0, clamp_max=float('inf'), gpr_M_reg=m_reg)
                    ctx.ue.generate_instruction_add_set(m_reg, clamp_rows_mlp)
                    shard_flops[0] += ctx.ue.activation_core(M=clamp_rows_mlp, N=64, A_DRAM_ADDR=ctx.rows_addr(self.VIS_MLP_GATE, MLP * bpe), OUTPUT_DRAM_ADDR=ctx.rows_addr(self.VIS_MLP_GATE, MLP * bpe), IDENTITY_DRAM_ADDR=self.VIS_IDENTITY_64, activation="clamp", clamp_min=cg['output'][0], clamp_max=cg['output'][1], gpr_M_reg=m_reg)
                    if cu["input"] != cg["input"]:
                        ctx.ue.generate_instruction_add_set(m_reg, clamp_rows_h)
                        shard_flops[0] += ctx.ue.activation_core(M=clamp_rows_h, N=64, A_DRAM_ADDR=ctx.rows_addr(self.VIS_PRE_FFN_NORM, H * bpe), OUTPUT_DRAM_ADDR=ctx.rows_addr(self.VIS_INPUT_CLIP_H_SCRATCH, H * bpe), IDENTITY_DRAM_ADDR=self.VIS_IDENTITY_64, activation="clamp", clamp_min=cu['input'][0], clamp_max=cu['input'][1], gpr_M_reg=m_reg)
                    ctx.ue.generate_instruction_add_set(m_reg, ctx.rows)
                    shard_flops[0] += ctx.ue.matmat_mul_core(M=ctx.rows, K=H, N=MLP, A_DRAM_ADDR=ctx.rows_addr(self.VIS_INPUT_CLIP_H_SCRATCH, H * bpe), B_DRAM_ADDR=w['up_proj']['data'], OUTPUT_DRAM_ADDR=ctx.rows_addr(self.VIS_MLP_UP, MLP * bpe), is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=w['up_proj']['scale'], gelu_enable=False, clamp_enable=(cu['output'][0] is not None), clamp_min=(0.0 if cu['output'][0] is None else cu['output'][0]), clamp_max=(float('inf') if cu['output'][1] is None else cu['output'][1]), gpr_M_reg=m_reg)
                    ctx.ue.generate_instruction_add_set(m_reg, ctx.rows)
                    shard_flops[0] += ctx.ue.eltwise_core_dram(M=ctx.rows, N=MLP, dram_a=ctx.rows_addr(self.VIS_MLP_GATE, MLP * bpe), dram_b=ctx.rows_addr(self.VIS_MLP_UP, MLP * bpe), dram_out=ctx.rows_addr(self.VIS_MLP_MULT, MLP * bpe), mode=UE_MODE.ELTWISE_MUL, gpr_M_reg=m_reg)
                    ctx.ue.generate_instruction_add_set(m_reg, clamp_rows_mlp)
                    shard_flops[0] += ctx.ue.activation_core(M=clamp_rows_mlp, N=64, A_DRAM_ADDR=ctx.rows_addr(self.VIS_MLP_MULT, MLP * bpe), OUTPUT_DRAM_ADDR=ctx.rows_addr(self.VIS_INPUT_CLIP_MLP_SCRATCH, MLP * bpe), IDENTITY_DRAM_ADDR=self.VIS_IDENTITY_64, activation="clamp", clamp_min=cd['input'][0], clamp_max=cd['input'][1], gpr_M_reg=m_reg)
                    ctx.ue.generate_instruction_add_set(m_reg, ctx.rows)
                    shard_flops[0] += ctx.ue.matmat_mul_core(M=ctx.rows, K=MLP, N=H, A_DRAM_ADDR=ctx.rows_addr(self.VIS_INPUT_CLIP_MLP_SCRATCH, MLP * bpe), B_DRAM_ADDR=w['down_proj']['data'], OUTPUT_DRAM_ADDR=ctx.rows_addr(self.VIS_MLP_DOWN, H * bpe), is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=w['down_proj']['scale'], gelu_enable=False, clamp_enable=(cd['output'][0] is not None), clamp_min=(0.0 if cd['output'][0] is None else cd['output'][0]), clamp_max=(float('inf') if cd['output'][1] is None else cd['output'][1]), gpr_M_reg=m_reg)
                    ctx.ue.generate_instruction_add_set(m_reg, ctx.rows)
                    shard_flops[0] += ctx.ue.rms_norm_core_dram(M=ctx.rows, N=H, A_DRAM_ADDR=ctx.rows_addr(self.VIS_MLP_DOWN, H * bpe), OUTPUT_DRAM_ADDR=ctx.rows_addr(self.VIS_POST_FFN_NORM, H * bpe), GAMMA_DRAM_ADDR=w['post_feedforward_layernorm'], gpr_M_reg=m_reg)
                    ctx.ue.generate_instruction_add_set(m_reg, ctx.rows)
                    shard_flops[0] += ctx.ue.eltwise_core_dram(M=ctx.rows, N=H, dram_a=ctx.rows_addr(self.VIS_POST_ATTN_RES, H * bpe), dram_b=ctx.rows_addr(self.VIS_POST_FFN_NORM, H * bpe), dram_out=ctx.rows_addr(OUT, H * bpe), mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=m_reg)
                gate_scheduler.sharded_region(S, _emit_post_attention_shard)
                self._vis_flops += shard_flops[0]
            _checkpoint(f"L{li}_post_attn")

        # ---- Pooler tail: wide-acc matmul pool + embed_vision (all on FPGA) ----
        # final_buf holds the [S, H] encoder output. Express the k x k spatial
        # average as a matmul P[N_SOFT, nV] @ hiddenᵀ[H, nV] so the MAC array
        # accumulates each cell sum in its bf20-wide accumulator, then the
        # Gemma4MultimodalEmbedder tail: RMSNorm (ones gamma) -> IF4 projection.
        # The matmul core computes A @ Bᵀ with A=P[N_SOFT,nV] and B=hiddenᵀ[H,nV],
        # so the valid grid is transposed [nV,H] -> [H,nV] first. P carries weight
        # 1.0 (cell SUM); the 1/k^2 average and the host sqrt(H) pre-scale are both
        # cancelled by the scale-invariant RMSNorm next.
        final_buf = self.VIS_IO_A if self.VIS_LAYERS % 2 == 0 else self.VIS_IO_B
        N_SOFT = self.VIS_POOL_N_SOFT
        nV = self.VIS_POOL_NVALID
        self.bf16_transpose_core(
            M=nV, N=H, INPUT_DRAM_ADDR=final_buf,
            OUTPUT_DRAM_ADDR=self.VIS_HIDDEN_T,
            IDENTITY_DRAM_ADDR=self.VIS_IDENTITY_64, gpr_M_reg=self._prime_M(nV))
        self._vis_flops += self.matmat_mul_core(
            M=N_SOFT, K=nV, N=H, A_DRAM_ADDR=self.VIS_POOL_P,
            B_DRAM_ADDR=self.VIS_HIDDEN_T, OUTPUT_DRAM_ADDR=self.VIS_POOL_OUT,
            is_B_quantized=False, gpr_M_reg=self._prime_M(N_SOFT))
        # embed_vision pre-projection RMSNorm (with_scale=False -> ones gamma).
        self._vis_flops += self.rms_norm_core_dram(M=N_SOFT, N=H, A_DRAM_ADDR=self.VIS_POOL_OUT, OUTPUT_DRAM_ADDR=self.VIS_EMBED_NORMED, GAMMA_DRAM_ADDR=self.VIS_EMBED_NORM_GAMMA, gpr_M_reg=self._prime_M(N_SOFT))
        # embed_vision projection: IF4 768 -> text_h.
        self._vis_flops += self.matmat_mul_core(M=N_SOFT, K=H, N=self.VIS_TEXT_H, A_DRAM_ADDR=self.VIS_EMBED_NORMED, B_DRAM_ADDR=self.VIS_EMBED_PROJ_INFO['data'], OUTPUT_DRAM_ADDR=self.VIS_EMBED_OUT, is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=self.VIS_EMBED_PROJ_INFO['scale'], gpr_M_reg=self._prime_M(N_SOFT))
        _checkpoint("pooler_tail")

        self.generate_instruction_halt()
        worker_addrs = gate_scheduler.finalize() if gate_scheduler is not None else []
        if gate_scheduler is not None:
            for worker in reversed(gate_scheduler.workers):
                worker.release_isa_reg()
        self._set_silent(_prev_silent)
        self.stop_capture()
        enc_bytes = bytearray()
        for inst in self.capture_buffer:
            enc_bytes.extend(inst.get_bytes())
        self.clear_capture_buffer()
        total_flops = int(self._vis_flops)
        self._vis_flops = None    # stop accumulating outside the encoder capture

        _vis_meta = {"num_patches": num_patches, "vis_layers": L,
                     "includes_patch_embed": True,
                     "vision_kernel": self.vision_kernel,
                     "multi_core": self.multi_core,
                     "total_flops": total_flops}
        if profile:
            _vis_meta["profile_checkpoints"] = enc_checkpoints
        # Merge the vision section into the combined programs bin (LM stays intact).
        primary_limit = (self.VISION_WORKER_ISA_BASE
                         if gate_scheduler is not None else self.LM_ISA_BASE)
        if base_addr + len(enc_bytes) > primary_limit:
            raise RuntimeError(
                f"vision core0 ISA overflow: 0x{base_addr + len(enc_bytes):X} > "
                f"0x{primary_limit:X}")
        self._store_program_section("vision", base_addr, enc_bytes, _vis_meta, profile=profile)
        if gate_scheduler is not None:
            worker = gate_scheduler.workers[0]
            worker_bytes = bytearray()
            for inst in worker.capture_buffer:
                worker_bytes.extend(inst.get_bytes())
            worker_addr = worker_addrs[0]
            if worker_addr + len(worker_bytes) > self.LM_ISA_BASE:
                raise RuntimeError(
                    f"vision core1 ISA overflow: 0x{worker_addr + len(worker_bytes):X} > "
                    f"0x{self.LM_ISA_BASE:X}")
            self._store_program_section(
                "vision_worker1", worker_addr, worker_bytes,
                {"parent": "vision", "engine_idx": 1,
                 "multi_core": self.multi_core}, profile=profile)
        print(f"  [Vision] encoder section stored ({len(enc_bytes)/1024/1024:.1f} MB @ 0x{base_addr:X}, "
              f"{time.perf_counter()-t0:.1f}s)", flush=True)

    def execute_vision_encoder_bin(self, num_patches: int, profile: bool = False):
        """Flush queued input DMAs (pixels/positions/identity/zeros/bias/RoPE),
        load the cached patch+encoder
        bin to program DRAM at its baked base, and execute. Returns elapsed
        seconds.

        ``profile``: run the profile-compiled bin **segment-by-segment** through
        its per-phase HALT checkpoints, timing each with the HW latency counter,
        and return the per-segment [(name, ms)] list instead. The encoder still
        computes its full output (segments tile the whole program). No preamble
        is needed — the vision ops prime their own GPRs inline."""
        import threading
        # Runtime uploads: everything vision_tensor_init allocated + built on
        # host is DMA'd here, right before the encoder runs.
        for _addr, _t in getattr(self, "_vis_pending_dmas", []):
            self.dma_to_accelerator_memory(_addr, _t)
        self._vis_pending_dmas = []
        meta, enc_bytes = self._get_program_section("vision", profile)
        if meta is None:
            raise FileNotFoundError("vision section not found in combined programs bin")
        program_addr = int(meta["dram_base"], 16)
        self._next_program_dram_addr = program_addr
        self.dma_write(DMA_DEVICE_H2C, program_addr, enc_bytes, len(enc_bytes))
        self.allocate_program_dram(len(enc_bytes))
        gate_scheduler = self._ensure_vision_gate_scheduler()
        worker_addrs = []
        if gate_scheduler is not None:
            worker_meta, worker_bytes = self._get_program_section("vision_worker1", profile)
            if worker_meta is None:
                raise FileNotFoundError("vision_worker1 section not found in combined programs bin")
            worker = gate_scheduler.workers[0]
            worker_addr = int(worker_meta["dram_base"], 16)
            worker._next_program_dram_addr = worker_addr
            worker.dma_write(DMA_DEVICE_H2C, worker_addr, worker_bytes, len(worker_bytes))
            worker.allocate_program_dram(len(worker_bytes))
            worker_addrs = [worker_addr]
            gate_scheduler.preclear_flags()
        print(f"  [Vision] launching patch+encoder ({len(enc_bytes)/1024/1024:.1f} MB) at 0x{program_addr:X}"
              f"{' [profiled]' if profile else ''} ...", flush=True)
        t0 = time.perf_counter()

        if profile:
            checkpoints = meta.get("profile_checkpoints", [])
            results = []
            if gate_scheduler is not None:
                gate_scheduler.start_workers(worker_addrs)
            self.start_execute_from_dram(program_addr)
            for name, resume_hex in checkpoints:
                self.wait_queue(180.0)
                results.append((name, self.report_latency_in_us() / 1e3))   # ms
                self.start_execute_from_dram(int(resume_hex, 16))
            self.wait_queue(180.0)   # tail: final HALT after the last checkpoint
            for worker in gate_scheduler.workers if gate_scheduler is not None else []:
                worker.wait_queue(180.0)
            results.append(("tail", self.report_latency_in_us() / 1e3))
            return results

        _anchor = getattr(self, "_vis_fpga_t0", t0)
        _stop = threading.Event()
        def _hb():
            while not _stop.wait(10):
                self._loud(f"  [Vision] ... running on FPGA ({time.perf_counter()-_anchor:.0f}s)", flush=True)
        _th = threading.Thread(target=_hb, daemon=True); _th.start()
        try:
            if gate_scheduler is not None:
                gate_scheduler.start_workers(worker_addrs)
            self.start_execute_from_dram(program_addr)
            self.wait_queue(180.0)
            for worker in gate_scheduler.workers if gate_scheduler is not None else []:
                worker.wait_queue(180.0)
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
    # Vision driver (S4): pooler tail, full encoder run, and the
    # set_prefill_seq_vlm entry that feeds image soft-tokens into run_prefill's
    # existing merge hooks.
    # ------------------------------------------------------------------
    def _run_vision_encoder_host(self, image_path: str, prompt: str):
        """Run the HF vision tower on the host and return the same image soft
        tokens and prompt metadata consumed by the FPGA LM prefill path."""
        from PIL import Image
        from transformers import AutoProcessor

        host_t0 = time.perf_counter()
        hf_model, model_dir = self._ensure_model_loaded()
        processor = AutoProcessor.from_pretrained(model_dir)

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

        # Per-stage host wall-clock timing (see summary table at the end). Each
        # entry is (label, seconds); _mark() closes the stage started at _t[0].
        _stage: list[tuple[str, float]] = []
        _t = [time.perf_counter()]
        def _mark(label: str) -> None:
            _now = time.perf_counter()
            _stage.append((label, _now - _t[0]))
            _t[0] = _now

        hf_model, model_dir = self._ensure_model_loaded()
        processor = AutoProcessor.from_pretrained(model_dir)
        _mark("HF model load")
        fpga_total_t0 = time.perf_counter()

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
        self.vision_hf_model = hf_model
        self.vision_pixel_values = pixel_values
        self.vision_pixel_position_ids = pixel_position_ids
        self.vision_padding_positions = padding_positions
        self.vision_num_patches = num_patches
        _mark("preprocess (HF processor)")

        # Save LM allocator state to restore after vision clobbers the shared
        # tensor DRAM region.
        _tensor_save = self._tensor_dram_addr
        _prog_addr_save = self._next_program_dram_addr
        _prog_base_save = self._program_dram_base

        _prev_silent = self._set_silent(True)
        try:
            self.vision_weight_init()                # weights ABOVE LM tensors (no reset)
            self.vision_tensor_init()                # scratch reuses LM tensor region; ISA base 0xa0
        finally:
            self._set_silent(_prev_silent)
        _mark("setup (weights/scratch alloc + host RoPE/bias)")

        self.compile_vision_encoder_bin(num_patches, profile=profile)
        _mark("patch+encoder compile (host emit)")
        self._vis_fpga_t0 = time.perf_counter()
        enc_result = self.execute_vision_encoder_bin(num_patches, profile=profile)
        _mark("patch+encoder execute (FPGA)")

        # The pooler tail (wide-acc matmul pool + embed_vision) runs on FPGA as
        # part of the one-shot encoder program (see compile_vision_encoder_bin);
        # the host only reads back the results.
        final_buf = self.VIS_IO_A if self.VIS_LAYERS % 2 == 0 else self.VIS_IO_B
        N_SOFT = self.VIS_POOL_N_SOFT
        image_features = self.dma_from_accelerator_memory(
            self.VIS_EMBED_OUT, (N_SOFT, self.VIS_TEXT_H)).float().cpu()
        encoder_out = self.dma_from_accelerator_memory(
            final_buf, (num_patches, self.VIS_H)).cpu()
        _mark("pooler tail readback (DMA)")

        # Numeric-harness checkpoints (SNR-compared vs HF in the numeric script).
        self._vis_ckpt = {
            "encoder_out": encoder_out,
            "pool_out": self.dma_from_accelerator_memory(
                self.VIS_POOL_OUT, (N_SOFT, self.VIS_H)).cpu(),
            "embed_normed": self.dma_from_accelerator_memory(
                self.VIS_EMBED_NORMED, (N_SOFT, self.VIS_H)).cpu(),
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
        # Per-stage host wall-clock breakdown of the whole vision path (--profile only).
        if profile:
            _tot = sum(s for _, s in _stage)
            print("\n=== VISION PATH host wall-clock breakdown ===", flush=True)
            print(f"{'stage':<38}{'sec':>9}{'%':>7}")
            print("-" * 54)
            for _lbl, _s in _stage:
                print(f"{_lbl:<38}{_s:>9.2f}{(_s / _tot * 100 if _tot else 0):>6.1f}%")
            print("-" * 54)
            print(f"{'TOTAL':<38}{_tot:>9.2f}{100.0:>6.1f}%")
        print(f"[Vision] FPGA encoder done: {image_features.shape[0]} soft tokens "
              f"in {fpga_total_s:.2f}s total.", flush=True)
        return image_features, token_ids, mm_types
