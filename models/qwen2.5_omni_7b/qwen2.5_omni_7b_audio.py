#!/usr/bin/env python3
"""Qwen2.5-Omni-7B Thinker audio encoder for UnifiedEngine.

``Qwen25OmniAudioMixin`` is mixed into the model's concrete engine class; it
is not instantiated by itself. The two convolutional front-end layers and
their GELUs, the 32 Transformer blocks, pooling, final LayerNorm, and
1280 -> 3584 projection all execute on FPGA. Front-end matrix rows and every
block's 20 independent attention heads are distributed over all eight U55
engines; serial data-layout operations execute on the primary between
rendezvous.

The implementation intentionally follows the upstream packing rules.  Each
audio item is split into 200-mel-frame chunks *before* convolution, the
stride-2 convolution reduces a chunk to at most 100 states, and all valid
states are packed back together.  A block-diagonal additive bias makes one
dense accelerator attention call mathematically equivalent to independent
attention within those chunks.

Weights are transient.  ``audio_weight_init`` rewinds the shared params
allocator and loads the ``audio`` region, allowing the LM weights to replace
the audio weights after the embeddings have been produced.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from user_dma_core import (
    DMA_DEVICE_C2H,
    DMA_DEVICE_H2C,
    INSTRUCTION_SIZE_BYTES,
    TYPE,
    UE_MODE,
    UE_VECTOR_SIZE,
    URAM_NEAR_FULL_ELEMENTS,
)


AUDIO_QUANT_PRECISION = "if4"
AUDIO_IF4_BLOCK_ELEMENTS = 64
AUDIO_IF4_SCALE_BYTES = 2
AUDIO_IF4_DATA_BYTES = 32
AUDIO_IF4_BLOCK_BYTES = AUDIO_IF4_SCALE_BYTES + AUDIO_IF4_DATA_BYTES
FLAG_PRECLEAR_PROGRAM_BYTES = 2 * INSTRUCTION_SIZE_BYTES


class Qwen25OmniAudioMixin:
    """Thinker audio-encoder methods for the Qwen2.5-Omni concrete engine."""

    # ------------------------------------------------------------------
    # Configuration and manifest validation
    # ------------------------------------------------------------------

    def _audio_dims(self) -> dict[str, int]:
        """Return and validate the fixed Qwen2.5-Omni-7B audio geometry."""
        cfg = self._cfg.get("audio", {})

        def value(*names: str, default: int) -> int:
            for name in names:
                if name in cfg:
                    return int(cfg[name])
            return int(default)

        dims = {
            "MELS": value("num_mel_bins", "in_channels", default=128),
            "H": value("d_model", "hidden_size", default=1280),
            "HEADS": value("encoder_attention_heads", "num_heads", default=20),
            "FFN": value("encoder_ffn_dim", "intermediate_size", default=5120),
            "LAYERS": value(
                "encoder_layers", "num_hidden_layers", "depth", default=32
            ),
            "N_WINDOW": value("n_window", "attention_window", default=100),
            "OUT": value("output_dim", "out_hidden_size", default=3584),
            "MAX_POS": value(
                "max_source_positions", "max_positions", default=1500
            ),
            "MAX_MEL_FRAMES": value("max_mel_frames", default=3000),
            "MAX_ENCODER_TOKENS": value("max_encoder_tokens", default=1500),
            "MAX_OUTPUT_TOKENS": value("max_output_tokens", default=750),
            "POSITION_THETA": value("position_theta", default=10000),
        }
        expected = {
            "MELS": 128,
            "H": 1280,
            "HEADS": 20,
            "FFN": 5120,
            "LAYERS": 32,
            "N_WINDOW": 100,
            "OUT": 3584,
        }
        mismatches = [
            f"{key}={dims[key]} (expected {want})"
            for key, want in expected.items()
            if dims[key] != want
        ]
        if mismatches:
            raise ValueError(
                "unsupported audio geometry for Qwen2.5-Omni-7B: "
                + ", ".join(mismatches)
            )
        if dims["H"] % dims["HEADS"]:
            raise ValueError("audio hidden size must be divisible by head count")
        dims["HEAD_DIM"] = dims["H"] // dims["HEADS"]
        dims["MEL_CHUNK"] = 2 * dims["N_WINDOW"]
        configured_head_dim = value("head_dim", default=dims["HEAD_DIM"])
        if configured_head_dim != dims["HEAD_DIM"]:
            raise ValueError(
                f"audio head_dim={configured_head_dim} disagrees with "
                f"hidden_size/num_heads={dims['HEAD_DIM']}"
            )
        for limit in (
            "MAX_POS",
            "MAX_MEL_FRAMES",
            "MAX_ENCODER_TOKENS",
            "MAX_OUTPUT_TOKENS",
            "POSITION_THETA",
        ):
            if dims[limit] < 1:
                raise ValueError(f"audio {limit.lower()} must be positive")
        activation = str(cfg.get("activation_function", cfg.get("activation", "gelu")))
        if activation != "gelu":
            raise ValueError(f"unsupported audio activation {activation!r}; expected 'gelu'")
        precision = str(
            self._cfg.get("precision", {}).get("audio", AUDIO_QUANT_PRECISION)
        )
        if precision != AUDIO_QUANT_PRECISION:
            raise ValueError(
                f"unsupported audio precision {precision!r}; expected "
                f"{AUDIO_QUANT_PRECISION!r}"
            )
        if dims["HEAD_DIM"] != UE_VECTOR_SIZE:
            raise ValueError(
                f"audio head_dim={dims['HEAD_DIM']} must equal the accelerator "
                f"vector width {UE_VECTOR_SIZE}"
            )
        return dims

    def _check_audio_execution_mode(self) -> None:
        """Validate the topology used by the head-sharded audio program."""
        cores = int(getattr(self, "multi_core", 1))
        if cores < 1:
            raise ValueError(f"multi_core must be positive, got {cores}")

    def _audio_expected_sections(self) -> dict[str, tuple[str, tuple[int, ...]]]:
        """Manifest contract: section name -> (wire format, logical shape)."""
        d = self._audio_dims()
        h, ff, layers, out = d["H"], d["FFN"], d["LAYERS"], d["OUT"]
        expected: dict[str, tuple[str, tuple[int, ...]]] = {
            "audio.conv1.weight": ("bf16", (h, 3 * d["MELS"])),
            "audio.conv1.bias": ("bf16", (h,)),
            "audio.conv2.weight": ("bf16", (h, 3 * h)),
            "audio.conv2.bias": ("bf16", (h,)),
            "audio.positional_embedding": ("bf16", (d["N_WINDOW"], h)),
        }
        for li in range(layers):
            pre = f"audio.layers.{li}"
            expected[f"{pre}.ln1.weight"] = ("bf16", (h,))
            expected[f"{pre}.ln1.bias"] = ("bf16", (h,))
            for proj in ("q", "k", "v", "o"):
                expected[f"{pre}.{proj}.weight.{AUDIO_QUANT_PRECISION}"] = (
                    AUDIO_QUANT_PRECISION,
                    (h, h),
                )
            for proj in ("q", "v", "o"):
                expected[f"{pre}.{proj}.bias"] = ("bf16", (h,))
            expected[f"{pre}.ln2.weight"] = ("bf16", (h,))
            expected[f"{pre}.ln2.bias"] = ("bf16", (h,))
            expected[f"{pre}.fc1.weight.{AUDIO_QUANT_PRECISION}"] = (
                AUDIO_QUANT_PRECISION,
                (ff, h),
            )
            expected[f"{pre}.fc1.bias"] = ("bf16", (ff,))
            expected[f"{pre}.fc2.weight.{AUDIO_QUANT_PRECISION}"] = (
                AUDIO_QUANT_PRECISION,
                (h, ff),
            )
            expected[f"{pre}.fc2.bias"] = ("bf16", (h,))
        expected["audio.ln_post.weight"] = ("bf16", (h,))
        expected["audio.ln_post.bias"] = ("bf16", (h,))
        expected[f"audio.proj.weight.{AUDIO_QUANT_PRECISION}"] = (
            AUDIO_QUANT_PRECISION,
            (out, h),
        )
        expected["audio.proj.bias"] = ("bf16", (out,))
        return expected

    def _audio_region(self) -> dict[str, Any]:
        region = getattr(self, "_audio_region_cache", None)
        if region is None:
            region = self._read_params_region("audio")
            required = {"bin_path", "base_offset", "size", "sections"}
            missing = required.difference(region)
            if missing:
                raise KeyError(
                    f"audio params region is missing fields: {sorted(missing)}"
                )
            self._audio_region_cache = region
        return region

    def _validate_audio_manifest(self, region: dict[str, Any]) -> None:
        sections = region["sections"]
        region_size = int(region["size"])
        for name, (wire_dtype, shape) in self._audio_expected_sections().items():
            if name not in sections:
                raise KeyError(f"audio weight {name!r} is missing from params.bin")
            section = sections[name]
            offset, size = int(section["offset"]), int(section["size"])
            if offset < 0 or size < 0 or offset + size > region_size:
                raise ValueError(
                    f"audio section {name!r} lies outside its region: "
                    f"offset={offset}, size={size}, region_size={region_size}"
                )
            elements = math.prod(shape)
            if wire_dtype == "bf16":
                expected_size = elements * 2
            else:
                if elements % AUDIO_IF4_BLOCK_ELEMENTS:
                    raise AssertionError(
                        f"internal error: {name} has an unaligned IF4 shape {shape}"
                    )
                expected_size = (
                    elements // AUDIO_IF4_BLOCK_ELEMENTS * AUDIO_IF4_BLOCK_BYTES
                )
            if size != expected_size:
                raise ValueError(
                    f"audio section {name!r} has {size} bytes; expected "
                    f"{expected_size} for {wire_dtype} {shape}"
                )
            disk_shape = section.get("shape")
            if disk_shape is not None and tuple(int(x) for x in disk_shape) != shape:
                raise ValueError(
                    f"audio section {name!r} shape {tuple(disk_shape)} != {shape}"
                )

    # ------------------------------------------------------------------
    # Host layout-only preparation for the FPGA convolutional front end
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prepare_audio_input(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
    ) -> dict[str, Any]:
        """Build packed im2col values and metadata for the U55 front end.

        Args:
            input_features: canonical processor output ``[B, 128, T]``.
            feature_attention_mask: ``[B, T]`` with exactly 0/1 entries.

        This method performs validation, masking, padding, and reordering only.
        Both learned convolutions and both GELUs execute in the captured FPGA
        program.
        """
        self._check_audio_execution_mode()
        if getattr(self, "_audio_tensor_init_done", False):
            raise RuntimeError(
                "prepare_audio_input() cannot replace audio after its shape-specific "
                "tensor arena was allocated; construct a fresh engine"
            )
        d = self._audio_dims()

        if not isinstance(input_features, torch.Tensor):
            raise TypeError("input_features must be a torch.Tensor")
        if not isinstance(feature_attention_mask, torch.Tensor):
            raise TypeError("feature_attention_mask must be a torch.Tensor")
        if input_features.ndim != 3:
            raise ValueError(
                f"input_features must have shape [B, {d['MELS']}, T], got "
                f"{tuple(input_features.shape)}"
            )
        batch, mels, frames = (int(x) for x in input_features.shape)
        if batch < 1 or mels != d["MELS"] or frames < 1:
            raise ValueError(
                f"input_features must have shape [B>=1, {d['MELS']}, T>=1], "
                f"got {tuple(input_features.shape)}"
            )
        if tuple(feature_attention_mask.shape) != (batch, frames):
            raise ValueError(
                "feature_attention_mask shape "
                f"{tuple(feature_attention_mask.shape)} != ({batch}, {frames})"
            )

        features = input_features.detach().cpu()
        mask_raw = feature_attention_mask.detach().cpu()
        if features.is_complex():
            raise TypeError("input_features must be real-valued")
        if not torch.isfinite(features.float()).all().item():
            raise ValueError("input_features contains NaN or Inf")
        if mask_raw.is_floating_point() and not torch.isfinite(mask_raw).all().item():
            raise ValueError("feature_attention_mask contains NaN or Inf")
        if not torch.all((mask_raw == 0) | (mask_raw == 1)).item():
            raise ValueError("feature_attention_mask entries must be exactly 0 or 1")
        mask = mask_raw.bool()
        feature_lens = mask.sum(dim=1, dtype=torch.long)
        if torch.any(feature_lens < 3).item():
            raise ValueError(
                "each audio item needs at least 3 valid mel frames so the "
                "post-encoder stride-2 pool has one output"
            )
        longest = int(feature_lens.max().item())
        if longest > d["MAX_MEL_FRAMES"]:
            raise ValueError(
                f"audio has {longest} valid mel frames; configured maximum is "
                f"{d['MAX_MEL_FRAMES']}"
            )

        # Match upstream's boolean pack followed by split(200): holes in a
        # processor mask are removed, not treated as zero-valued time steps.
        chunks: list[torch.Tensor] = []
        chunk_lengths: list[int] = []
        mel_chunk = d["MEL_CHUNK"]
        for bi in range(batch):
            sample = features[bi, :, mask[bi]].to(torch.bfloat16).contiguous()
            for start in range(0, sample.shape[1], mel_chunk):
                chunk = sample[:, start : start + mel_chunk]
                chunks.append(chunk)
                chunk_lengths.append(int(chunk.shape[1]))

        # Conv1 input is an im2col view with kernel-major lanes. ``unfold`` is
        # only a byte-layout transform; no learned arithmetic runs here.
        conv1_rows: list[torch.Tensor] = []
        for chunk, length in zip(chunks, chunk_lengths):
            padded = torch.zeros(
                d["MELS"], length + 2, dtype=torch.bfloat16
            )
            padded[:, 1 : length + 1] = chunk
            windows = padded.unfold(1, 3, 1).permute(1, 2, 0)
            conv1_rows.append(windows.reshape(length, 3 * d["MELS"]))
        conv1_input = torch.cat(conv1_rows, dim=0).contiguous()
        if conv1_input.shape[0] != sum(chunk_lengths):
            raise AssertionError("conv1 im2col rows do not cover all valid frames")

        after_chunk = [(length - 1) // 2 + 1 for length in chunk_lengths]
        max_after = max(after_chunk)
        if max_after > d["MAX_POS"]:
            raise ValueError(
                f"audio chunk produces {max_after} positions, above MAX_POS={d['MAX_POS']}"
            )
        encoder_rows = sum(after_chunk)
        if encoder_rows > d["MAX_ENCODER_TOKENS"]:
            raise ValueError(
                f"audio produces {encoder_rows} packed encoder states; "
                f"configured maximum is {d['MAX_ENCODER_TOKENS']}"
            )

        # 200 is even, therefore summing ceil(chunk_len/2) across an audio item
        # equals ceil(total_feature_len/2), exactly as upstream aftercnn_lens.
        aftercnn_lens = ((feature_lens - 1) // 2 + 1).tolist()
        if sum(int(x) for x in aftercnn_lens) != encoder_rows:
            raise AssertionError("packed convolution lengths do not add up")
        output_lens = [((int(length) - 2) // 2 + 1) for length in aftercnn_lens]
        if any(length < 1 for length in output_lens):
            raise ValueError("stride-2 audio pooling would produce an empty output")
        if sum(output_lens) > d["MAX_OUTPUT_TOKENS"]:
            raise ValueError(
                f"audio produces {sum(output_lens)} pooled embeddings; configured "
                f"maximum is {d['MAX_OUTPUT_TOKENS']}"
            )

        cu = [0]
        for length in after_chunk:
            cu.append(cu[-1] + int(length))

        self._audio_conv1_input = conv1_input
        self._audio_feature_lens = [int(x) for x in feature_lens.tolist()]
        self._audio_chunk_feature_lens = list(chunk_lengths)
        self._audio_aftercnn_lens = [int(x) for x in aftercnn_lens]
        self._audio_output_lengths = [int(x) for x in output_lens]
        self._audio_chunk_aftercnn_lens = [int(x) for x in after_chunk]
        self._audio_cu_seqlens = cu
        self._audio_input_generation = getattr(self, "_audio_input_generation", 0) + 1

        self._loud(
            f"  [Audio] layout prep: {batch} item(s), "
            f"{sum(self._audio_feature_lens)} mel frames -> {encoder_rows} "
            f"encoder states in {len(after_chunk)} chunk(s) -> "
            f"{sum(output_lens)} LM embeddings; conv1/conv2/GELU run on U55"
        )
        return {
            "frontend_input": conv1_input,
            "feature_lengths": list(self._audio_feature_lens),
            "aftercnn_lengths": list(self._audio_aftercnn_lens),
            "output_lengths": list(self._audio_output_lengths),
            "cu_seqlens": torch.tensor(cu, dtype=torch.int32),
        }

    # ------------------------------------------------------------------
    # Weight and tensor lifecycle
    # ------------------------------------------------------------------

    def audio_weight_init(self) -> None:
        """Load IF4 encoder weights and BF16 norm/bias tensors.

        The method is idempotent only while the params cursor still identifies
        the audio allocation.  LM and vision phases rewind and overwrite that
        same window; after either phase, calling this method reloads audio at
        the original deterministic addresses so an existing capture remains
        valid.
        """
        self._check_audio_execution_mode()
        already_loaded = getattr(self, "_audio_weight_init_done", False)
        if (
            already_loaded
            and hasattr(self, "_audio_weight_end")
            and self.get_params_dram_addr() == self._audio_weight_end
        ):
            return
        old_span = (
            (self._audio_weight_start, self._audio_weight_end)
            if already_loaded
            else None
        )

        region = self._audio_region()
        self._validate_audio_manifest(region)  # fail before modifying DRAM cursors
        sections = region["sections"]
        d = self._audio_dims()

        def need(name: str) -> dict[str, Any]:
            return sections[name]

        # Audio and LM deliberately share the transient params window.
        self._invalidate_decode_overlay()
        self.reset_params_dram_addr()
        self._audio_weight_init_done = False
        # Rewinding immediately destroys whichever phase owned this window.
        # Clear its optimistic idempotency flag even if a later DMA fails, so
        # no caller can silently reuse the now-partial LM/vision image.
        self._lm_weight_init_done = False
        self._vision_weight_init_done = False
        start = self.get_params_dram_addr()
        self._loud(
            f"  [Audio] loading {d['LAYERS']} layers "
            f"({AUDIO_QUANT_PRECISION.upper()} matrices, BF16 norms/biases) "
            f"at 0x{start:X} ..."
        )

        with open(region["bin_path"], "rb") as f:
            base = int(region["base_offset"])
            self.audio_conv1_weight = self._dma_bf16(
                f, need("audio.conv1.weight"), base, "audio.conv1.weight"
            )
            self.audio_conv1_bias = self._dma_bf16(
                f, need("audio.conv1.bias"), base, "audio.conv1.bias"
            )
            self.audio_conv2_weight = self._dma_bf16(
                f, need("audio.conv2.weight"), base, "audio.conv2.weight"
            )
            self.audio_conv2_bias = self._dma_bf16(
                f, need("audio.conv2.bias"), base, "audio.conv2.bias"
            )
            self.audio_positional_embedding = self._dma_bf16(
                f,
                need("audio.positional_embedding"),
                base,
                "audio.positional_embedding",
            )
            layer_addrs: list[dict[str, int]] = []
            for li in range(d["LAYERS"]):
                pre = f"audio.layers.{li}"
                la: dict[str, int] = {}
                la["ln1_weight"] = self._dma_bf16(
                    f, need(f"{pre}.ln1.weight"), base, f"{pre}.ln1.weight"
                )
                la["ln1_bias"] = self._dma_bf16(
                    f, need(f"{pre}.ln1.bias"), base, f"{pre}.ln1.bias"
                )
                for proj in ("q", "k", "v", "o"):
                    la[f"{proj}_scale"], la[f"{proj}_data"] = self._dma_if4(
                        f,
                        need(f"{pre}.{proj}.weight.{AUDIO_QUANT_PRECISION}"),
                        base,
                        f"{pre}.{proj}",
                    )
                for proj in ("q", "v", "o"):
                    la[f"{proj}_bias"] = self._dma_bf16(
                        f, need(f"{pre}.{proj}.bias"), base, f"{pre}.{proj}.bias"
                    )
                la["ln2_weight"] = self._dma_bf16(
                    f, need(f"{pre}.ln2.weight"), base, f"{pre}.ln2.weight"
                )
                la["ln2_bias"] = self._dma_bf16(
                    f, need(f"{pre}.ln2.bias"), base, f"{pre}.ln2.bias"
                )
                for fc in ("fc1", "fc2"):
                    la[f"{fc}_scale"], la[f"{fc}_data"] = self._dma_if4(
                        f,
                        need(f"{pre}.{fc}.weight.{AUDIO_QUANT_PRECISION}"),
                        base,
                        f"{pre}.{fc}",
                    )
                    la[f"{fc}_bias"] = self._dma_bf16(
                        f, need(f"{pre}.{fc}.bias"), base, f"{pre}.{fc}.bias"
                    )
                layer_addrs.append(la)
                if (li + 1) % 8 == 0 or li == d["LAYERS"] - 1:
                    self._loud(f"    audio layer {li + 1}/{d['LAYERS']} loaded")
            self.audio_layer_addrs = layer_addrs

            self.audio_ln_post_weight = self._dma_bf16(
                f, need("audio.ln_post.weight"), base, "audio.ln_post.weight"
            )
            self.audio_ln_post_bias = self._dma_bf16(
                f, need("audio.ln_post.bias"), base, "audio.ln_post.bias"
            )
            self.audio_proj_scale, self.audio_proj_data = self._dma_if4(
                f,
                need(f"audio.proj.weight.{AUDIO_QUANT_PRECISION}"),
                base,
                "audio.proj",
            )
            self.audio_proj_bias = self._dma_bf16(
                f, need("audio.proj.bias"), base, "audio.proj.bias"
            )

        bpe = self.bytes_per_element
        self._audio_identity = self.allocate_params_dram(
            UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe, label="audio.identity64"
        )
        self.dma_to_accelerator_memory(
            self._audio_identity,
            torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16).contiguous(),
        )
        # Reused by every dynamic LayerNorm; providing these explicitly avoids
        # allocating another constant vector during each of 65 norm calls.
        self._audio_ln_zeros = self.allocate_params_dram(
            d["H"] * bpe, label="audio.layernorm.zeros"
        )
        self._audio_ln_inv_n = self.allocate_params_dram(
            d["H"] * bpe, label="audio.layernorm.inv_n"
        )
        self.dma_to_accelerator_memory(
            self._audio_ln_zeros, torch.zeros(d["H"], dtype=torch.bfloat16)
        )
        self.dma_to_accelerator_memory(
            self._audio_ln_inv_n,
            torch.full((d["H"],), 1.0 / d["H"], dtype=torch.bfloat16),
        )

        end = self.get_params_dram_addr()
        if end > self.PARAMS_LIMIT:
            raise MemoryError(
                f"audio weights overflow params DRAM: 0x{end:X} > "
                f"0x{self.PARAMS_LIMIT:X}"
            )
        self._audio_weight_start = start
        self._audio_weight_end = end
        self._audio_weight_init_done = True
        # Allocation is deterministic after reset_params_dram_addr().  Should a
        # concrete engine change alignment/base policy between phase loads,
        # force recapture instead of launching code with stale weight literals.
        if old_span is not None and old_span != (start, end):
            for attr in (
                "_audio_program_addr",
                "_audio_program_bytes",
                "_audio_compiled_generation",
            ):
                if hasattr(self, attr):
                    delattr(self, attr)
        self._loud(
            f"  [Audio] weights loaded: {(end - start) / 2**20:.1f} MiB, "
            f"0x{start:X}..0x{end:X}"
        )

    def _build_audio_attention_bias(self, seq_len: int, aligned_seq_len: int) -> torch.Tensor:
        """Dense BF16 block-diagonal mask for independent 200-frame chunks."""
        bias = torch.full(
            (aligned_seq_len, aligned_seq_len),
            float("-inf"),
            dtype=torch.bfloat16,
        )
        cu = self._audio_cu_seqlens
        if cu[0] != 0 or cu[-1] != seq_len:
            raise AssertionError(f"invalid audio chunk boundaries {cu}")
        for start, end in zip(cu[:-1], cu[1:]):
            if not 0 <= start < end <= seq_len:
                raise AssertionError(f"invalid audio chunk [{start}, {end})")
            bias[start:end, start:end] = 0
        # Valid queries may never see padding.  Padded query rows are ignored
        # after attention, but giving them finite logits avoids an all--Inf
        # softmax row and the resulting NaNs in scratch/output buffers.
        if aligned_seq_len > seq_len:
            bias[seq_len:, :] = 0
        return bias.contiguous()

    def audio_tensor_init(self) -> None:
        """Allocate intermediates for the currently prepared audio input."""
        self._check_audio_execution_mode()
        if not getattr(self, "_audio_weight_init_done", False):
            raise RuntimeError("audio_weight_init() must run before audio_tensor_init()")
        if not hasattr(self, "_audio_conv1_input"):
            raise RuntimeError("prepare_audio_input() must run before audio_tensor_init()")

        generation = self._audio_input_generation
        if getattr(self, "_audio_tensor_init_done", False):
            if self._audio_tensor_generation == generation:
                return
            raise RuntimeError(
                "audio tensors were already allocated for another input; construct "
                "a fresh engine before changing the audio shape"
            )

        d = self._audio_dims()
        h, heads, head_dim, ff = d["H"], d["HEADS"], d["HEAD_DIM"], d["FFN"]
        bpe = self.bytes_per_element
        conv1_rows = int(self._audio_conv1_input.shape[0])
        seq_len = sum(self._audio_chunk_aftercnn_lens)
        aligned = ((seq_len + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
        pooled = sum(self._audio_output_lengths)
        if seq_len < 1 or pooled < 1:
            raise ValueError("audio encoder and pooled lengths must both be positive")
        self._audio_seq_len = seq_len
        self._audio_aligned_seq_len = aligned
        self._audio_num_tokens = pooled
        self._audio_pending_dmas: list[tuple[int, torch.Tensor]] = []

        def alloc(elements: int, label: str) -> int:
            if elements < 1:
                raise ValueError(f"cannot allocate empty audio tensor {label}")
            return self.allocate_tensor_dram(elements * bpe, label=label)

        # Device convolution front end. Conv1 im2col arrives from host as a
        # layout-only transform. Conv2 im2col and the positional rows are
        # assembled in DRAM by the captured program from device-resident data.
        self.AUDIO_CONV1_INPUT = alloc(
            conv1_rows * 3 * d["MELS"], "audio.conv1_input"
        )
        self.AUDIO_CONV1_OUT = alloc(conv1_rows * h, "audio.conv1_out")
        self.AUDIO_CONV2_INPUT = alloc(seq_len * 3 * h, "audio.conv2_input")
        self.AUDIO_POSITION = alloc(seq_len * h, "audio.position")

        # Transformer I/O and projections.  The aligned tail exists only so
        # head-major attention planes have a 64-row stride; projections and
        # residuals execute over ``seq_len`` real rows.
        self.AUDIO_IO_A = alloc(aligned * h, "audio.io_a")
        self.AUDIO_IO_B = alloc(aligned * h, "audio.io_b")
        self.AUDIO_NORM = alloc(aligned * h, "audio.norm")
        self.AUDIO_RESIDUAL = alloc(aligned * h, "audio.residual")
        self.AUDIO_Q = alloc(aligned * h, "audio.q")
        self.AUDIO_K = alloc(aligned * h, "audio.k")
        self.AUDIO_V = alloc(aligned * h, "audio.v")
        self.AUDIO_Q_HM = alloc(heads * aligned * head_dim, "audio.q_head_major")
        self.AUDIO_K_HM = alloc(heads * aligned * head_dim, "audio.k_head_major")
        self.AUDIO_V_HM = alloc(heads * aligned * head_dim, "audio.v_head_major")
        self.AUDIO_ATTN_OUT_HM = alloc(
            heads * aligned * head_dim, "audio.attn_out_head_major"
        )
        self.AUDIO_ATTN_RESULT = alloc(aligned * h, "audio.attn_result")
        self.AUDIO_PROJ_OUT = alloc(aligned * h, "audio.o_proj")
        self.AUDIO_FFN = alloc(aligned * ff, "audio.ffn")
        self.AUDIO_FFN_OUT = alloc(aligned * h, "audio.ffn_out")

        self.AUDIO_ATTN_BIAS = alloc(aligned * aligned, "audio.attn_bias")
        scratch_elements = (head_dim + aligned) * aligned + aligned * head_dim
        self.AUDIO_ATTN_SCRATCH = alloc(scratch_elements, "audio.attn_scratch")
        # Attention heads execute concurrently.  The kernel writes V.T,
        # softmax, and scaled-Q scratch, so sharing this buffer across engines
        # would create a finite-but-corrupt race.  Core 0 keeps the tensor-map
        # allocation; each worker receives a private low-DRAM copy.
        self.AUDIO_ATTN_SCRATCH_PER_ENGINE = [self.AUDIO_ATTN_SCRATCH]
        if getattr(self, "multi_core", 1) > 1:
            scratch_bytes = scratch_elements * bpe
            self.AUDIO_ATTN_SCRATCH_PER_ENGINE.extend(
                self.mc_arena.alloc_tensor(
                    engine_idx, scratch_bytes, "audio attention scratch"
                )
                for engine_idx in range(1, self.multi_core)
            )
            spans = sorted(
                (address, address + scratch_bytes)
                for address in self.AUDIO_ATTN_SCRATCH_PER_ENGINE
            )
            for (_, previous_end), (next_start, _) in zip(spans, spans[1:]):
                assert previous_end <= next_start, (
                    "per-engine audio attention scratch overlaps"
                )
            self._loud(
                f"  Audio attention scratch: {self.multi_core} private "
                f"buffer(s), {scratch_bytes / 2**20:.2f} MiB each"
            )

        # Pooling is performed before ln_post in the released architecture.
        self.AUDIO_POOL_EVEN = alloc(pooled * h, "audio.pool_even")
        self.AUDIO_POOL_ODD = alloc(pooled * h, "audio.pool_odd")
        self.AUDIO_POOL = alloc(pooled * h, "audio.pool")
        self.AUDIO_ENCODER_OUT = alloc(pooled * d["OUT"], "audio.encoder_out")

        self._audio_pending_dmas.append(
            (self.AUDIO_CONV1_INPUT, self._audio_conv1_input.flatten())
        )
        # Conv2 gather deliberately skips left/right padding at chunk edges,
        # and the transformer uses an aligned tail. Seed both regions to zero.
        self._audio_pending_dmas.append(
            (
                self.AUDIO_CONV2_INPUT,
                torch.zeros(seq_len * 3 * h, dtype=torch.bfloat16),
            )
        )
        self._audio_pending_dmas.append(
            (self.AUDIO_IO_A, torch.zeros(aligned * h, dtype=torch.bfloat16))
        )
        self._audio_pending_dmas.append(
            (
                self.AUDIO_ATTN_BIAS,
                self._build_audio_attention_bias(seq_len, aligned),
            )
        )
        # bf16_permute writes only real rows.  Seed head-major tails once; they
        # remain zero because no layer ever writes projection data into them.
        head_zeros = torch.zeros(heads * aligned * head_dim, dtype=torch.bfloat16)
        for addr in (self.AUDIO_Q_HM, self.AUDIO_K_HM, self.AUDIO_V_HM):
            self._audio_pending_dmas.append((addr, head_zeros))

        end = self.get_tensor_dram_addr()
        if end > self.TENSOR_LIMIT:
            raise MemoryError(
                f"audio tensors overflow tensor DRAM: 0x{end:X} > "
                f"0x{self.TENSOR_LIMIT:X}"
            )
        self._audio_tensor_generation = generation
        self._audio_tensor_init_done = True
        self._loud(
            f"  [Audio] tensors: S={seq_len} (aligned {aligned}), pooled={pooled}, "
            f"end=0x{end:X}"
        )

    # ------------------------------------------------------------------
    # FPGA program capture
    # ------------------------------------------------------------------

    def _emit_audio_conv2_im2col(self) -> None:
        """Build stride-2 Conv2 windows from Conv1 activations on device."""
        h = self._audio_dims()["H"]
        bpe = self.bytes_per_element
        input_offset = 0
        output_offset = 0
        for chunk_len, output_len in zip(
            self._audio_chunk_feature_lens,
            self._audio_chunk_aftercnn_lens,
        ):
            for local_out in range(output_len):
                center = 2 * local_out
                for kernel_slot, local_input in enumerate(
                    (center - 1, center, center + 1)
                ):
                    if not 0 <= local_input < chunk_len:
                        continue
                    self.accelerator_memory_to_sram(
                        accelerator_dram_address=(
                            self.AUDIO_CONV1_OUT
                            + (input_offset + local_input) * h * bpe
                        ),
                        sram_address=0x00000,
                        element_size=h,
                    )
                    self.sram_to_accelerator_memory(
                        sram_address=0x00000,
                        accelerator_dram_address=(
                            self.AUDIO_CONV2_INPUT
                            + ((output_offset + local_out) * 3 * h + kernel_slot * h)
                            * bpe
                        ),
                        element_size=h,
                    )
            input_offset += chunk_len
            output_offset += output_len
        if input_offset != self._audio_conv1_input.shape[0]:
            raise AssertionError("conv2 gather did not consume every Conv1 row")
        if output_offset != self._audio_seq_len:
            raise AssertionError("conv2 gather did not produce every encoder row")

    def _emit_audio_position_gathers(self) -> None:
        """Repeat position rows 0..chunk_len-1 from params into packed order."""
        h = self._audio_dims()["H"]
        bpe = self.bytes_per_element
        rows_per_transfer = max(1, URAM_NEAR_FULL_ELEMENTS // h)
        output_offset = 0
        for chunk_len in self._audio_chunk_aftercnn_lens:
            for start in range(0, chunk_len, rows_per_transfer):
                take = min(rows_per_transfer, chunk_len - start)
                self.accelerator_memory_to_sram(
                    accelerator_dram_address=(
                        self.audio_positional_embedding + start * h * bpe
                    ),
                    sram_address=0x00000,
                    element_size=take * h,
                )
                self.sram_to_accelerator_memory(
                    sram_address=0x00000,
                    accelerator_dram_address=(
                        self.AUDIO_POSITION + (output_offset + start) * h * bpe
                    ),
                    element_size=take * h,
                )
            output_offset += chunk_len
        if output_offset != self._audio_seq_len:
            raise AssertionError("position gathers do not cover the packed sequence")

    def _emit_audio_pool_gathers(self, source: int) -> None:
        """Gather even/odd rows per original audio item for AvgPool1d(2, 2)."""
        h = self._audio_dims()["H"]
        bpe = self.bytes_per_element
        rows_per_transfer = max(1, URAM_NEAR_FULL_ELEMENTS // h)
        input_offset = 0
        output_offset = 0
        for aftercnn_len, output_len in zip(
            self._audio_aftercnn_lens, self._audio_output_lengths
        ):
            for local_out in range(0, output_len, rows_per_transfer):
                take = min(rows_per_transfer, output_len - local_out)
                src_row = input_offset + 2 * local_out
                dst_row = output_offset + local_out
                for parity, dst in (
                    (0, self.AUDIO_POOL_EVEN),
                    (1, self.AUDIO_POOL_ODD),
                ):
                    self.accelerator_memory_to_sram(
                        accelerator_dram_address=(
                            source + (src_row + parity) * h * bpe
                        ),
                        sram_address=0x00000,
                        element_size=take * h,
                        stride_bytes_per_chunk=h * bpe,
                        stride_jump_bytes=2 * h * bpe,
                    )
                    self.sram_to_accelerator_memory(
                        sram_address=0x00000,
                        accelerator_dram_address=dst + dst_row * h * bpe,
                        element_size=take * h,
                    )
            input_offset += aftercnn_len
            output_offset += output_len
        if input_offset != self._audio_seq_len or output_offset != self._audio_num_tokens:
            raise AssertionError("audio pooling offsets do not cover the packed sequence")

    def compile_audio_encoder(self) -> int:
        """Compile audio as a transaction, including post-finalize checks."""
        previous_silent = self._set_silent(False)
        self._set_silent(previous_silent)
        reg_counter_before = self._isa_reg_counter
        inst_ptr_counter_before = self._inst_ptr_counter
        program_cursor_before = self.get_program_dram_addr()
        scheduler_before = getattr(self, "_multi_core_schedulers", {}).get("audio")
        worker_cursors_before = (
            [worker.get_program_dram_addr() for worker in scheduler_before.workers]
            if scheduler_before is not None else None
        )
        try:
            return self._compile_audio_encoder_impl()
        except Exception:
            scheduler = getattr(self, "_multi_core_schedulers", {}).get("audio")
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

    def _compile_audio_encoder_impl(self) -> int:
        """Capture all 32 blocks plus pool/ln_post/proj as one FPGA program."""
        self._check_audio_execution_mode()
        if not getattr(self, "_audio_tensor_init_done", False):
            raise RuntimeError("audio_tensor_init() must run before compile_audio_encoder()")
        if self._audio_tensor_generation != self._audio_input_generation:
            raise RuntimeError("audio tensors do not match the prepared input")

        d = self._audio_dims()
        conv1_rows = int(self._audio_conv1_input.shape[0])
        s, s_aligned, h = (
            self._audio_seq_len,
            self._audio_aligned_seq_len,
            d["H"],
        )
        heads, head_dim, ff = d["HEADS"], d["HEAD_DIM"], d["FFN"]
        pooled = self._audio_num_tokens
        bpe = self.bytes_per_element

        self._loud(
            f"  [Audio] compiling FPGA Conv1/Conv2 + {d['LAYERS']} layers, S={s}, "
            f"{heads}x{head_dim} attention (200-frame block mask) ..."
        )
        started = time.perf_counter()
        scheduler = self._ensure_stage_scheduler("audio")
        scheduler.register_per_engine_addrs(
            "audio_attention_scratch", self.AUDIO_ATTN_SCRATCH_PER_ENGINE
        )
        self.reset_program_dram_addr()
        program_addr = self.get_program_dram_addr()
        self.clear_inst_id()
        self.clear_capture_buffer()
        self.start_capture()
        previous_silent = self._set_silent(True)
        reg_counter_before = self._isa_reg_counter
        inst_ptr_counter_before = self._inst_ptr_counter
        flops = 0

        def layer_norm(rows: int, rows_reg: int, src: int, dst: int, weight: int, bias: int) -> int:
            return self.layer_norm_core_dram(
                M=rows,
                N=h,
                A_DRAM_ADDR=src,
                OUTPUT_DRAM_ADDR=dst,
                GAMMA_DRAM_ADDR=weight,
                BETA_DRAM_ADDR=bias,
                ZEROS_DRAM_ADDR=self._audio_ln_zeros,
                INV_N_DRAM_ADDR=self._audio_ln_inv_n,
                gpr_M_reg=rows_reg,
            ) or 0

        def matmul(
            rows: int,
            rows_reg: int,
            k: int,
            n: int,
            src: int,
            weights: dict[str, int] | None,
            tag: str,
            dst: int,
            *,
            bias: bool = True,
            gelu: bool = False,
            scale_addr: int | None = None,
            data_addr: int | None = None,
            bias_addr: int | None = None,
        ) -> int:
            if weights is not None:
                scale_addr = weights[f"{tag}_scale"]
                data_addr = weights[f"{tag}_data"]
                bias_addr = weights.get(f"{tag}_bias") if bias else None
            if scale_addr is None or data_addr is None:
                raise AssertionError(f"missing IF4 addresses for audio {tag}")
            return self.matmat_mul_core(
                M=rows,
                K=k,
                N=n,
                A_DRAM_ADDR=src,
                B_DRAM_ADDR=data_addr,
                OUTPUT_DRAM_ADDR=dst,
                is_B_quantized=True,
                data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=scale_addr,
                C_DRAM_ADDR=bias_addr,
                bias_mode="broadcast_N" if bias_addr is not None else None,
                gelu_enable=gelu,
                gpr_M_reg=rows_reg,
            ) or 0

        def sharded_bf16_frontend(
            rows: int,
            k: int,
            src: int,
            weight: int,
            bias: int,
            dst: int,
        ) -> int:
            """Row-shard one BF16 convolution-as-matmul over all engines."""
            total = [0]

            def body(ctx) -> None:
                value = ctx.ue.matmat_mul_core(
                    M=ctx.rows,
                    K=k,
                    N=h,
                    A_DRAM_ADDR=ctx.rows_addr(src, k * bpe),
                    B_DRAM_ADDR=weight,
                    OUTPUT_DRAM_ADDR=ctx.rows_addr(dst, h * bpe),
                    C_DRAM_ADDR=bias,
                    bias_mode="broadcast_N",
                    is_B_quantized=False,
                    gelu_enable=True,
                    gpr_M_reg=ctx.m_reg,
                )
                if isinstance(value, (int, float)):
                    total[0] += int(value)

            scheduler.sharded_region(rows, body)
            return total[0]

        try:
            scheduler.begin_program()

            # These row-count registers let the large DRAM kernels emit compact
            # PBI loops instead of Python-unrolling once per audio state.
            seq_reg = self.alloc_isa_reg()
            pool_reg = self.alloc_isa_reg()
            self.generate_instruction_add_set(seq_reg, s)
            self.generate_instruction_add_set(pool_reg, pooled)

            # Learned audio front end: BF16 Conv1/GELU and Conv2/GELU are
            # matmuls over kernel-major im2col rows. Both projections are
            # row-sharded across all eight U55 engines.
            flops += sharded_bf16_frontend(
                conv1_rows,
                3 * d["MELS"],
                self.AUDIO_CONV1_INPUT,
                self.audio_conv1_weight,
                self.audio_conv1_bias,
                self.AUDIO_CONV1_OUT,
            )
            self._emit_audio_conv2_im2col()
            flops += sharded_bf16_frontend(
                s,
                3 * h,
                self.AUDIO_CONV2_INPUT,
                self.audio_conv2_weight,
                self.audio_conv2_bias,
                self.AUDIO_IO_A,
            )
            self._emit_audio_position_gathers()
            flops += self.eltwise_core_dram(
                M=s,
                N=h,
                dram_a=self.AUDIO_IO_A,
                dram_b=self.AUDIO_POSITION,
                dram_out=self.AUDIO_IO_A,
                mode=UE_MODE.ELTWISE_ADD,
                gpr_M_reg=seq_reg,
            ) or 0

            for li, weights in enumerate(self.audio_layer_addrs):
                layer_in = self.AUDIO_IO_A if li % 2 == 0 else self.AUDIO_IO_B
                layer_out = self.AUDIO_IO_B if li % 2 == 0 else self.AUDIO_IO_A

                # Pre-norm MHA: q/v/o carry bias, k intentionally does not.
                flops += layer_norm(
                    s,
                    seq_reg,
                    layer_in,
                    self.AUDIO_NORM,
                    weights["ln1_weight"],
                    weights["ln1_bias"],
                )
                flops += matmul(s, seq_reg, h, h, self.AUDIO_NORM, weights, "q", self.AUDIO_Q)
                flops += matmul(
                    s,
                    seq_reg,
                    h,
                    h,
                    self.AUDIO_NORM,
                    weights,
                    "k",
                    self.AUDIO_K,
                    bias=False,
                )
                flops += matmul(s, seq_reg, h, h, self.AUDIO_NORM, weights, "v", self.AUDIO_V)

                self.bf16_permute_dram_core(
                    heads,
                    s,
                    head_dim,
                    self.AUDIO_Q,
                    self.AUDIO_Q_HM,
                    write_grouped=True,
                    group_stride_rows=s_aligned,
                )
                self.bf16_permute_dram_core(
                    heads,
                    s,
                    head_dim,
                    self.AUDIO_K,
                    self.AUDIO_K_HM,
                    write_grouped=True,
                    group_stride_rows=s_aligned,
                )
                self.bf16_permute_dram_core(
                    heads,
                    s,
                    head_dim,
                    self.AUDIO_V,
                    self.AUDIO_V_HM,
                    write_grouped=True,
                    group_stride_rows=s_aligned,
                )

                attention_flops = [0]

                def attention_kernel(
                    engine, kernel_head_dim: int, kernel_seq_len: int, **kwargs
                ) -> None:
                    # Audio is MHA, so the scheduler's GQA fan-out parameter is
                    # always one and unified_attention_core does not consume it.
                    kwargs.pop("num_q_heads", None)
                    batch_reg = engine.alloc_isa_reg()
                    aligned_reg = engine.alloc_isa_reg()
                    engine.generate_instruction_add_set(batch_reg, kernel_seq_len)
                    engine.generate_instruction_add_set(aligned_reg, kernel_seq_len)
                    value = engine.unified_attention_core(
                        batch=kernel_seq_len,
                        aligned_seq_len=kernel_seq_len,
                        head_dim=kernel_head_dim,
                        q_scale=1.0 / math.sqrt(kernel_head_dim),
                        gpr_batch_reg=batch_reg,
                        gpr_aligned_seq_len_reg=aligned_reg,
                        **kwargs,
                    )
                    engine.release_isa_reg()
                    engine.release_isa_reg()
                    if isinstance(value, (int, float)):
                        attention_flops[0] += value

                scheduler.head_sharded_attention(
                    heads,
                    s_aligned,
                    head_dim,
                    Q_addr=self.AUDIO_Q_HM,
                    K_addr=self.AUDIO_K_HM,
                    V_addr=self.AUDIO_V_HM,
                    OUT_addr=self.AUDIO_ATTN_OUT_HM,
                    IDENTITY_addr=self._audio_identity,
                    bias_addr=self.AUDIO_ATTN_BIAS,
                    bias_per_head=False,
                    scratch_name="audio_attention_scratch",
                    kernel=attention_kernel,
                )
                flops += attention_flops[0]

                self.bf16_permute_dram_core(
                    heads,
                    s,
                    head_dim,
                    self.AUDIO_ATTN_OUT_HM,
                    self.AUDIO_ATTN_RESULT,
                    write_grouped=False,
                    group_stride_rows=s_aligned,
                )
                flops += matmul(
                    s,
                    seq_reg,
                    h,
                    h,
                    self.AUDIO_ATTN_RESULT,
                    weights,
                    "o",
                    self.AUDIO_PROJ_OUT,
                )
                flops += self.eltwise_core_dram(
                    M=s,
                    N=h,
                    dram_a=layer_in,
                    dram_b=self.AUDIO_PROJ_OUT,
                    dram_out=self.AUDIO_RESIDUAL,
                    mode=UE_MODE.ELTWISE_ADD,
                    gpr_M_reg=seq_reg,
                ) or 0

                # Pre-norm GELU FFN and the second residual.
                flops += layer_norm(
                    s,
                    seq_reg,
                    self.AUDIO_RESIDUAL,
                    self.AUDIO_NORM,
                    weights["ln2_weight"],
                    weights["ln2_bias"],
                )
                flops += matmul(
                    s,
                    seq_reg,
                    h,
                    ff,
                    self.AUDIO_NORM,
                    weights,
                    "fc1",
                    self.AUDIO_FFN,
                    gelu=True,
                )
                flops += matmul(
                    s,
                    seq_reg,
                    ff,
                    h,
                    self.AUDIO_FFN,
                    weights,
                    "fc2",
                    self.AUDIO_FFN_OUT,
                )
                flops += self.eltwise_core_dram(
                    M=s,
                    N=h,
                    dram_a=self.AUDIO_RESIDUAL,
                    dram_b=self.AUDIO_FFN_OUT,
                    dram_out=layer_out,
                    mode=UE_MODE.ELTWISE_ADD,
                    gpr_M_reg=seq_reg,
                ) or 0

            final_states = (
                self.AUDIO_IO_A if d["LAYERS"] % 2 == 0 else self.AUDIO_IO_B
            )
            self._emit_audio_pool_gathers(final_states)
            flops += self.eltwise_core_dram(
                M=pooled,
                N=h,
                dram_a=self.AUDIO_POOL_EVEN,
                dram_b=self.AUDIO_POOL_ODD,
                dram_out=self.AUDIO_POOL,
                mode=UE_MODE.ELTWISE_ADD,
                gpr_M_reg=pool_reg,
            ) or 0
            flops += self.eltwise_core_dram(
                M=pooled,
                N=h,
                dram_a=self.AUDIO_POOL,
                dram_b=None,
                dram_out=self.AUDIO_POOL,
                mode=UE_MODE.MUL_BROADCAST,
                scalar=0.5,
                gpr_M_reg=pool_reg,
            ) or 0
            flops += layer_norm(
                pooled,
                pool_reg,
                self.AUDIO_POOL,
                self.AUDIO_NORM,
                self.audio_ln_post_weight,
                self.audio_ln_post_bias,
            )
            flops += matmul(
                pooled,
                pool_reg,
                h,
                d["OUT"],
                self.AUDIO_NORM,
                None,
                "proj",
                self.AUDIO_ENCODER_OUT,
                scale_addr=self.audio_proj_scale,
                data_addr=self.audio_proj_data,
                bias_addr=self.audio_proj_bias,
            )

            self.release_isa_reg()  # pool_reg
            self.release_isa_reg()  # seq_reg
            self.generate_instruction_halt()
            worker_addresses = scheduler.finalize()
            self.stop_capture()
        except Exception:
            # Restore the LIFO allocator counters as well as capture state so a
            # corrected retry does not inherit registers/pointers leaked by the
            # failing emitter.
            scheduler.abort_program()
            if getattr(self, "is_capture_on", False):
                self.stop_capture()
            self.clear_capture_buffer()
            self._isa_reg_counter = reg_counter_before
            self._inst_ptr_counter = inst_ptr_counter_before
            raise
        finally:
            self._set_silent(previous_silent)

        encoded = bytearray()
        for instruction in self.capture_buffer:
            encoded.extend(instruction.get_bytes())
        self.clear_capture_buffer()
        program = bytes(encoded)
        if not program:
            raise RuntimeError("audio encoder capture produced an empty program")
        master_limit = getattr(
            self, "WORKER_ISA_BASE", getattr(self, "DRAM_END", None)
        )
        master_end = program_addr + len(program) + FLAG_PRECLEAR_PROGRAM_BYTES
        if master_limit is not None and master_end > master_limit:
            raise MemoryError(
                f"audio ISA plus flag-preclear overflow: 0x{master_end:X} > "
                f"0x{master_limit:X}"
            )
        self._audio_worker_programs = []
        for engine_idx, (worker, address) in enumerate(
            zip(scheduler.workers, worker_addresses), start=1
        ):
            worker_blob = bytearray()
            for instruction in worker.capture_buffer:
                worker_blob.extend(instruction.get_bytes())
            self.mc_arena.check_isa_fits(
                engine_idx,
                address,
                len(worker_blob) + FLAG_PRECLEAR_PROGRAM_BYTES,
            )
            self._note_worker_isa(engine_idx, "audio", len(worker_blob))
            self._audio_worker_programs.append(
                (engine_idx, worker, address, bytes(worker_blob))
            )
        self._audio_program_addr = program_addr
        self._audio_program_bytes = program
        self._audio_total_flops = int(flops)
        self._audio_compiled_generation = self._audio_input_generation
        self._loud(
            f"  [Audio] compiled: {len(program) / 2**20:.2f} MiB at "
            f"0x{program_addr:X}, 20 heads across "
            f"{getattr(self, 'multi_core', 1)} engines, {flops / 1e9:.1f} GFLOP, "
            f"{time.perf_counter() - started:.1f}s"
        )
        return program_addr

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_audio_encoder(self, timeout_s: float = 600.0) -> torch.Tensor:
        """Upload inputs/program, execute, and return packed LM embeddings."""
        self._check_audio_execution_mode()
        if not hasattr(self, "_audio_program_bytes"):
            raise RuntimeError("compile_audio_encoder() must run before run_audio_encoder()")
        if self._audio_compiled_generation != self._audio_input_generation:
            raise RuntimeError("compiled audio program does not match the prepared input")
        if not math.isfinite(float(timeout_s)) or timeout_s <= 0:
            raise ValueError(f"timeout_s must be finite and positive, got {timeout_s!r}")

        for address, tensor in self._audio_pending_dmas:
            if not torch.isfinite(tensor.float()).all().item():
                # The block mask intentionally contains -Inf; all other queued
                # tensors must be finite.
                if address != self.AUDIO_ATTN_BIAS:
                    raise ValueError("queued audio input contains NaN or Inf")
                if torch.isnan(tensor.float()).any().item():
                    raise ValueError("audio attention bias contains NaN")
            self.dma_to_accelerator_memory(address, tensor.contiguous())

        address = self._audio_program_addr
        self._next_program_dram_addr = address
        written = self.dma_write(
            DMA_DEVICE_H2C,
            address,
            self._audio_program_bytes,
            len(self._audio_program_bytes),
        )
        if written != len(self._audio_program_bytes):
            raise IOError(
                f"audio master ISA DMA wrote {written} of "
                f"{len(self._audio_program_bytes)} bytes")
        self.allocate_program_dram(len(self._audio_program_bytes))

        scheduler = self._ensure_stage_scheduler("audio")
        worker_addresses = []
        for engine_idx, worker, worker_address, worker_blob in self._audio_worker_programs:
            worker._next_program_dram_addr = worker_address
            written = worker.dma_write(
                DMA_DEVICE_H2C,
                worker_address,
                worker_blob,
                len(worker_blob),
            )
            if written != len(worker_blob):
                raise IOError(
                    f"audio worker {engine_idx} ISA DMA wrote {written} of "
                    f"{len(worker_blob)} bytes")
            worker.allocate_program_dram(len(worker_blob))
            worker_addresses.append(worker_address)
        if not scheduler.host_segmented:
            scheduler.preclear_flags()

        self._loud(
            f"  [Audio] launching encoder ({len(self._audio_program_bytes) / 2**20:.2f} MiB) "
            f"at 0x{address:X} ..."
        )
        started = time.perf_counter()
        # Workers park on their first rendezvous until core 0 releases the
        # first layer's head-sharded attention region.
        if scheduler.host_segmented:
            latency_us = scheduler.run_host_segmented(
                address, worker_addresses, timeout_seconds=timeout_s)
        else:
            scheduler.start_workers(worker_addresses)
            self.start_execute_from_dram(address)
            self.wait_queue(float(timeout_s))
            # wait_queue logs and returns on timeout; it deliberately does not
            # raise. Never read partially-written embeddings as though they were
            # a successful encoder result.
            if self.is_queue_busy():
                raise TimeoutError(
                    f"audio FPGA queue is still busy after {timeout_s:.1f}s"
                )
            for engine_idx, worker in enumerate(scheduler.workers, start=1):
                worker.wait_queue(float(timeout_s))
                if worker.is_queue_busy():
                    raise TimeoutError(
                        f"audio FPGA worker {engine_idx} is still busy after "
                        f"{timeout_s:.1f}s"
                    )
            latency_us = float(self.report_latency_in_us())
        wall_s = time.perf_counter() - started

        d = self._audio_dims()
        output = torch.zeros(
            self._audio_num_tokens * d["OUT"], dtype=torch.bfloat16
        )
        read = self.dma_read(
            DMA_DEVICE_C2H,
            self.AUDIO_ENCODER_OUT,
            output,
            output.numel() * 2,
        )
        if read != output.numel() * 2:
            raise IOError(
                f"audio output DMA read {read} of {output.numel() * 2} bytes")
        output = output.reshape(self._audio_num_tokens, d["OUT"]).cpu()
        if tuple(output.shape) != (self._audio_num_tokens, d["OUT"]):
            raise RuntimeError(
                f"audio output shape {tuple(output.shape)} != "
                f"({self._audio_num_tokens}, {d['OUT']})"
            )
        if not torch.isfinite(output.float()).all().item():
            raise FloatingPointError("audio encoder returned NaN or Inf")

        self._audio_embeddings = output.contiguous()
        self._audio_latency_us = latency_us
        self._audio_wall_s = wall_s
        self._audio_gflops = (
            self._audio_total_flops / (latency_us * 1e-6) / 1e9
            if latency_us > 0
            else 0.0
        )
        # Transformer layers overwrite the input ping-pong buffer. Keep the
        # prepared host tensors until output validation succeeds, allowing a
        # failed launch/wait/read to retry from pristine inputs.
        self._audio_pending_dmas = []
        self._loud(
            f"  [Audio] done: {wall_s:.2f}s wall, {latency_us / 1e6:.2f}s HW, "
            f"{self._audio_gflops:.1f} GFLOPS; {self._audio_num_tokens} embeddings"
        )
        return self._audio_embeddings
