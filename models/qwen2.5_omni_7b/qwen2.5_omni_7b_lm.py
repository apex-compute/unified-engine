#!/usr/bin/env python3
"""Qwen2.5-Omni Thinker LM adapter over the Qwen2.5-VL emitters."""

from __future__ import annotations

import importlib.util
import os
import sys

import torch


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VL_DIR = os.path.join(os.path.dirname(_THIS_DIR), "qwen2.5_vl_3b")


def _load_vl_lm():
    name = "qwen2_5_vl_3b_lm_for_omni"
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(_VL_DIR, "qwen2.5_vl_3b_lm.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Qwen2.5-VL LM mixin from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_vl_lm = _load_vl_lm()


class Qwen25OmniLMMixin(_vl_lm.Qwen25VLLMMixin):
    """28-layer, width-3584 Thinker using the shared Qwen GQA dataflow.

    Shapes and the projection policy come from the Omni config. V remains BF16
    for attention accuracy, following Qwen2.5-VL. Prefill uses IF4 Q/K/O and
    MLP projections; after full prefill, decode replaces O with a time-shared
    BF16 overlay while the other projections keep their prefill precision.
    Only region discovery differs from the shared runtime; the emitted RMSNorm,
    biased Q/K/V, GQA attention, SwiGLU and multimodal RoPE operations are the
    same architecture.
    """

    def compile_decoder(self, layer_size: int = None,
                        profile: bool = False) -> int:
        """Compile decode with a mandatory FPGA-wide sharded-head argmax.

        The shared Qwen runtime normally compares the eight shard winners on
        the host because each engine's MMIO argmax register exposes an index,
        not its corresponding value.  Omni is stricter: ``_dec_round`` appends
        an engine-0 reduction over the already contiguous logits in board DRAM,
        so the host reads only the final global hardware index.
        """
        master_cursor = self.get_program_dram_addr()
        scheduler_before = getattr(self, "_multi_core_schedulers", {}).get(
            "decode"
        )
        worker_cursors = (
            [worker.get_program_dram_addr() for worker in scheduler_before.workers]
            if scheduler_before is not None
            else None
        )
        self._fpga_global_argmax_emitted = False
        try:
            base = super().compile_decoder(
                layer_size=layer_size, profile=profile
            )
            if not self._fpga_global_argmax_emitted:
                raise RuntimeError(
                    "Omni decoder compiled without its FPGA global-argmax tail"
                )
            # Decoder column shards are allocated first. The IF8 embedding then
            # fills only the audited remainder of each private weight window.
            self._ensure_fpga_embedding()
            return base
        except Exception:
            # The shared compiler's transaction ends before the embedding
            # upload. Roll ISA cursors back across both operations so a short
            # table DMA can be retried without appending another full decoder
            # image to every master/worker slice. Decode weight shards remain
            # cached and are deliberately reused on that retry.
            self._next_program_dram_addr = master_cursor
            scheduler = getattr(self, "_multi_core_schedulers", {}).get("decode")
            if scheduler is not None:
                if worker_cursors is None:
                    for worker in scheduler.workers:
                        worker.reset_program_dram_addr()
                else:
                    for worker, cursor in zip(scheduler.workers, worker_cursors):
                        worker._next_program_dram_addr = cursor
            raise

    def _dec_round(self, dec_sched, ops, master_emit, worker_extra=None) -> int:
        """Append the device-wide reduction immediately after the head join."""
        flops = super()._dec_round(
            dec_sched, ops, master_emit, worker_extra=worker_extra)
        if (
            len(ops) == 1
            and getattr(ops[0][0], "name", None) == "lm_head"
        ):
            if self._fpga_global_argmax_emitted:
                raise RuntimeError("FPGA global argmax emitted more than once")
            flops += self._emit_fpga_global_argmax()
            self._fpga_global_argmax_emitted = True
        return flops

    def _emit_fpga_global_argmax(self) -> int:
        """Reduce all eight LM-head shards without host logit inspection.

        The sharded head writes one contiguous BF16 vocabulary row to shared
        board DRAM.  Engine 0 then streams that row in 64-value tiles through
        the existing 64x64 identity matrix.  ``max_clear_en`` is asserted only
        for the first tile, so the hardware max/index tracker spans the entire
        vocabulary and its final argmax register contains the global token ID.

        This extra identity pass is deliberately simple: it adds no parameter
        storage, transfers no logits over PCIe, and relies on the same
        cross-strip max accumulation already used by wide quantized matmuls.
        """
        d = self._lm_dims()
        vocab = int(d["VOCAB"])
        tile = int(_vl_lm.UE_VECTOR_SIZE)
        if vocab <= 0 or vocab % tile:
            raise ValueError(
                f"FPGA global argmax requires a positive {tile}-aligned "
                f"vocabulary, got {vocab}")

        shard = getattr(self, "_decode_lm_shard", None)
        if shard is None or len(shard.shards) != 8:
            raise RuntimeError(
                "FPGA global argmax requires the LM head on all eight engines")
        cursor = 0
        for engine_idx, part in enumerate(shard.shards):
            if part.col_offset != cursor or part.cols <= 0:
                raise RuntimeError(
                    f"LM-head shard {engine_idx} does not form a contiguous "
                    f"vocabulary row at column {cursor}")
            cursor += part.cols
        if cursor != vocab:
            raise RuntimeError(
                f"LM-head shards cover {cursor} columns, expected {vocab}")

        vector_sram = 0x00000
        identity_sram = 0x80000
        self.accelerator_memory_to_sram(
            accelerator_dram_address=self.LM_IDENTITY,
            sram_address=identity_sram,
            element_size=tile * tile,
        )
        for column in range(0, vocab, tile):
            self.accelerator_memory_to_sram(
                accelerator_dram_address=(
                    self.LOGITS + column * self.bytes_per_element
                ),
                sram_address=vector_sram,
                element_size=tile,
            )
            self.start_queue_for_bf16_matvec_operation(
                max_clear_en=int(column == 0),
                fmax_context_addr=0,
                vector_sram_start_addr=vector_sram,
                matrix_sram_start_addr=identity_sram,
                output_sram_wb_addr=vector_sram,
                K=tile,
                N=tile,
            )
        return 2 * vocab * tile

    def _decode_token(self) -> int:
        """Return only the FPGA's global token index; never read logits."""
        if not getattr(self, "_fpga_global_argmax_emitted", False):
            raise RuntimeError(
                "refusing host token selection: FPGA global argmax is absent")
        token = int(self.get_arg_max_index())
        self._fpga_decode_token_ids.append(token)
        return token

    def _decode_stop_token_ids(self) -> set[int]:
        """Use Omni's declared EOS; its distinct padding ID is not an EOS."""
        eos = int(self._cfg["tokens"]["eos_token_id"])
        if eos != int(self._end_of_turn_token_id):
            raise ValueError(
                f"configured Omni EOS {eos} differs from tokenizer turn end "
                f"{self._end_of_turn_token_id}"
            )
        return {eos}

    def run_decoder(self, first_token: int,
                    max_new_tokens: int = 256) -> tuple[int, str]:
        """Reset the audit trail of FPGA-selected token IDs for this request."""
        self._require_decode_overlay()
        self._fpga_decode_token_ids = []
        return super().run_decoder(first_token, max_new_tokens)

    def run_decode_step_profiled(self, token: int, program, checkpoints,
                                 workers=None, timeout_s: float = 60.0):
        """Apply the same phase guard to the direct profiled-step API."""
        self._require_decode_overlay()
        return super().run_decode_step_profiled(
            token, program, checkpoints, workers=workers, timeout_s=timeout_s
        )

    def _require_decode_overlay(self) -> None:
        if not getattr(self, "_decode_bf16_o_loaded", False):
            raise RuntimeError(
                "decode refused before the BF16 O shared phase is active"
            )

    def _invalidate_decode_overlay(self) -> None:
        """Forget every fact made stale when another params phase is loaded."""
        self._decode_bf16_o_planned = False
        self._decode_bf16_o_loaded = False
        self._decode_o_plan = None
        # A completed prefill belongs to the exact LM image that produced its
        # KV cache.  It cannot authorize a later overlay after that image has
        # been replaced by vision, audio, or a fresh LM load.
        self._prefill_seq_len_run = None

    def _read_lm_region(self) -> dict:
        return self._read_params_region("lm")

    def lm_weight_init(self) -> None:
        """Load the Thinker and invalidate encoders sharing its DRAM window."""
        if getattr(self, "_lm_weight_init_done", False):
            return
        # Clear the optimistic overlay state before the first possible LM DMA.
        # A failed reload must never leave decode authorized against a partial
        # or newly replaced params image.
        self._invalidate_decode_overlay()
        super().lm_weight_init()
        # The inherited loader already clears the vision flag.  Audio occupies
        # the same transient params window and must be treated the same way.
        self._audio_weight_init_done = False

    def lm_tensor_init(self) -> None:
        """Allocate LM state and exclude padded head rows from FPGA argmax."""
        super().lm_tensor_init()
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "Qwen2.5-Omni requires its tokenizer before LM tensor setup"
            )
        valid_vocab = len(tokenizer)
        padded_vocab = int(self._lm_dims()["VOCAB"])
        if valid_vocab <= 0 or valid_vocab > padded_vocab:
            raise ValueError(
                f"tokenizer exposes {valid_vocab} token(s), outside the padded "
                f"LM-head width 1..{padded_vocab}"
            )
        if valid_vocab == padded_vocab:
            return

        # The released head has 399 alignment-only rows after the tokenizer's
        # final added token. Keep the full 64-aligned matmul/argmax width, but
        # make those rows ineligible in the device-side broadcast bias.
        penalty = torch.zeros(padded_vocab, dtype=torch.bfloat16)
        penalty[valid_vocab:] = torch.finfo(torch.bfloat16).min
        self.dma_to_accelerator_memory(self.PENALTY_BIAS, penalty)
        self._valid_token_count = valid_vocab
        self._loud(
            f"  [LM] FPGA argmax masks {padded_vocab - valid_vocab} padded "
            f"LM-head row(s); valid token IDs 0..{valid_vocab - 1}"
        )

    def _device_embedding_enabled(self) -> bool:
        return self._cfg.get("precision", {}).get("embedding") in {"if4", "if8"}

    def _decode_bf16_projections(self) -> set[str]:
        values = set(
            self._cfg.get("precision", {}).get("decode_bf16_projections", ())
        )
        if values != {"o"}:
            raise ValueError(
                "Qwen2.5-Omni decode phase requires BF16 O projections"
            )
        return values

    def _decode_projection_is_quantized(self, tag: str) -> bool:
        if tag in self._decode_bf16_projections():
            return False
        return super()._decode_projection_is_quantized(tag)

    def _decode_projection_should_shard(self, tag: str, layer: int) -> bool:
        # BF16 O is loaded into the shared params window after all remaining
        # projections have been sharded. Running this narrow M=1 projection on
        # the master avoids another 686 MiB of private duplication.
        return tag not in self._decode_bf16_projections()

    def _decode_projection_uses_static_bf16(self, tag: str) -> bool:
        """Keep the phase-shared BF16 O projection on the proven M=1 tiler."""
        return tag in self._decode_bf16_projections()

    def _prepare_decode_shared_weights(self, layer_size: int) -> None:
        """Plan BF16 O addresses and preserve auxiliaries without touching prefill.

        Decoder compilation happens before prefill because device-side token
        embedding is installed with the decode shards. It may assign addresses
        for the future overlay, but it must not overwrite the resident IF4
        prefill image. ``activate_decode_shared_weights`` performs that phase
        transition only after prefill has populated the KV cache.
        """
        if getattr(self, "_decode_bf16_o_planned", False):
            return
        d = self._lm_dims()
        if layer_size != d["NL"]:
            raise ValueError(
                f"decode BF16 O region covers {d['NL']} layers, got {layer_size}"
            )

        lm_region = self._read_lm_region()
        decode_region = self._read_params_region("decode_o")
        lm_sections = lm_region["sections"]
        decode_sections = decode_region["sections"]
        h = int(d["H"])
        expected_o_bytes = h * h * self.bytes_per_element

        if len(self.lm_layer_addrs) < layer_size:
            raise RuntimeError(
                f"only {len(self.lm_layer_addrs)} LM layer address sets are "
                f"available for {layer_size} decode layers"
            )

        auxiliary: list[tuple[dict, str, str, str, int]] = []
        for li, la in enumerate(self.lm_layer_addrs[:layer_size]):
            prefix = f"language_model.layers.{li}"
            for attr, section_name, elements in (
                ("q_bias", f"{prefix}.self_attn.q_proj.bias", d["QH"] * d["AHD"]),
                ("k_bias", f"{prefix}.self_attn.k_proj.bias", d["KVH"] * d["AHD"]),
                ("v_bias", f"{prefix}.self_attn.v_proj.bias", d["KVH"] * d["AHD"]),
                ("ln1", f"{prefix}.input_layernorm.weight", h),
                ("ln2", f"{prefix}.post_attention_layernorm.weight", h),
            ):
                if section_name not in lm_sections:
                    raise KeyError(f"decode auxiliary {section_name!r} is missing")
                section = lm_sections[section_name]
                if tuple(section.get("shape", ())) != (elements,):
                    raise ValueError(
                        f"{section_name} shape {section.get('shape')} is not "
                        f"[{elements}]"
                    )
                if int(section["size"]) != elements * self.bytes_per_element:
                    raise ValueError(
                        f"{section_name} has {section['size']} bytes, expected "
                        f"{elements * self.bytes_per_element}"
                    )
                auxiliary.append(
                    (la, attr, section_name, f"decode.{section_name}", elements)
                )
        final_name = "language_model.norm.weight"
        if final_name not in lm_sections:
            raise KeyError(f"decode auxiliary {final_name!r} is missing")
        final_section = lm_sections[final_name]
        if tuple(final_section.get("shape", ())) != (h,):
            raise ValueError(
                f"{final_name} shape {final_section.get('shape')} is not [{h}]"
            )
        if int(final_section["size"]) != h * self.bytes_per_element:
            raise ValueError(
                f"{final_name} has {final_section['size']} bytes, expected "
                f"{h * self.bytes_per_element}"
            )

        o_plan = []
        for li in range(layer_size):
            name = f"language_model.layers.{li}.self_attn.o_proj.weight"
            section = decode_sections.get(name)
            if section is None:
                raise KeyError(f"decode BF16 O tensor {name!r} is missing")
            if tuple(section.get("shape", ())) != (h, h):
                raise ValueError(
                    f"{name} shape {section.get('shape')} is not [{h}, {h}]"
                )
            if int(section["size"]) != expected_o_bytes:
                raise ValueError(
                    f"{name} has {section['size']} bytes, expected {expected_o_bytes}"
                )
            o_plan.append((name, section))
        if int(decode_region["size"]) > self.PARAMS_LIMIT - self.PARAMS_BASE:
            raise MemoryError(
                f"decode BF16 O region is {decode_region['size']} bytes, larger "
                "than the shared params window"
            )

        # Validate the complete tensor allocation before the first write so a
        # geometry error cannot partially destroy the resident prefill weights.
        tensor_cursor = self.get_tensor_dram_addr()
        for _, _, section_name, _, _ in auxiliary:
            tensor_cursor = (tensor_cursor + 63) & ~63
            tensor_cursor += int(lm_sections[section_name]["size"])
        tensor_cursor = (tensor_cursor + 63) & ~63
        tensor_cursor += int(lm_sections[final_name]["size"])
        if tensor_cursor > self.TENSOR_LIMIT:
            raise MemoryError(
                f"decode auxiliaries end at 0x{tensor_cursor:X}, beyond tensor "
                f"limit 0x{self.TENSOR_LIMIT:X}"
            )

        def read_exact(file_obj, region: dict, section: dict, label: str) -> bytes:
            file_obj.seek(int(region["base_offset"]) + int(section["offset"]))
            blob = file_obj.read(int(section["size"]))
            if len(blob) != int(section["size"]):
                raise RuntimeError(f"truncated artifact read for {label}")
            return blob

        tensor_cursor_before = self.get_tensor_dram_addr()
        dram_addresses_before = dict(self._dram_addresses)
        pointer_values_before = [
            (la, attr, la.get(attr)) for la, attr, _, _, _ in auxiliary
        ]
        final_norm_before = getattr(self, "final_norm_addr", None)
        try:
            with open(lm_region["bin_path"], "rb") as file_obj:
                for la, attr, section_name, label, _ in auxiliary:
                    section = lm_sections[section_name]
                    blob = read_exact(file_obj, lm_region, section, label)
                    address = self.allocate_tensor_dram(len(blob), label=label)
                    written = self.dma_write(
                        _vl_lm.DMA_DEVICE_H2C, address, blob, len(blob)
                    )
                    if written != len(blob):
                        raise IOError(
                            f"{label}: tensor DMA wrote {written} of {len(blob)} bytes"
                        )
                    la[attr] = address
                final_section = lm_sections[final_name]
                blob = read_exact(file_obj, lm_region, final_section, final_name)
                address = self.allocate_tensor_dram(
                    len(blob), label="decode.final_norm"
                )
                written = self.dma_write(
                    _vl_lm.DMA_DEVICE_H2C, address, blob, len(blob)
                )
                if written != len(blob):
                    raise IOError(
                        f"decode.final_norm DMA wrote {written} of {len(blob)} bytes"
                    )
                self.final_norm_addr = address
        except Exception:
            self._tensor_dram_addr = tensor_cursor_before
            self._dram_addresses.clear()
            self._dram_addresses.update(dram_addresses_before)
            for la, attr, old_value in pointer_values_before:
                if old_value is None:
                    la.pop(attr, None)
                else:
                    la[attr] = old_value
            if final_norm_before is None:
                if hasattr(self, "final_norm_addr"):
                    del self.final_norm_addr
            else:
                self.final_norm_addr = final_norm_before
            self._decode_bf16_o_planned = False
            self._decode_bf16_o_loaded = False
            self._decode_o_plan = None
            raise

        for li, (_name, section) in enumerate(o_plan):
            self.lm_layer_addrs[li]["o_weight"] = (
                self.PARAMS_BASE + int(section["offset"])
            )
        self._decode_o_plan = (decode_region, tuple(o_plan))
        self._decode_bf16_o_planned = True
        self._loud(
            f"  [Decode] BF16 O phase planned: {decode_region['size'] / 2**20:.1f} "
            "MiB will replace shared IF4 prefill weights after prefill"
        )

    def activate_decode_shared_weights(self) -> None:
        """Atomically transition the shared params window from prefill to decode."""
        if getattr(self, "_decode_bf16_o_loaded", False):
            return
        if not getattr(self, "_decode_bf16_o_planned", False):
            raise RuntimeError("BF16 decode O was not planned during decoder compile")
        if not getattr(self, "_lm_weight_init_done", False):
            raise RuntimeError(
                "BF16 decode O activation requires the complete LM prefill "
                "weight image to be resident"
            )
        d = self._lm_dims()
        compiled_seq_len = getattr(self, "_prefill_seq_len", None)
        completed_seq_len = getattr(self, "_prefill_seq_len_run", None)
        if (
            compiled_seq_len is None
            or completed_seq_len != compiled_seq_len
            or getattr(self, "seq_len", None) != compiled_seq_len
        ):
            raise RuntimeError(
                "BF16 decode O activation requires a successfully completed "
                "prefill for the currently compiled prompt"
            )
        if int(getattr(self, "_prefill_layers", -1)) != int(d["NL"]):
            raise RuntimeError(
                "BF16 decode O activation requires a full-layer prefill"
            )
        decode_region, o_plan = self._decode_o_plan

        self.reset_params_dram_addr()
        try:
            with open(decode_region["bin_path"], "rb") as file_obj:
                for li, (name, section) in enumerate(o_plan):
                    file_obj.seek(
                        int(decode_region["base_offset"]) + int(section["offset"])
                    )
                    blob = file_obj.read(int(section["size"]))
                    if len(blob) != int(section["size"]):
                        raise RuntimeError(f"truncated artifact read for {name}")
                    address = self.allocate_params_dram(
                        len(blob), label=f"decode.{name}"
                    )
                    expected = int(self.lm_layer_addrs[li]["o_weight"])
                    if address != expected:
                        raise AssertionError(
                            f"{name} allocated at 0x{address:X}, decoder expects "
                            f"0x{expected:X}"
                        )
                    written = self.dma_write(
                        _vl_lm.DMA_DEVICE_H2C, address, blob, len(blob)
                    )
                    if written != len(blob):
                        raise IOError(
                            f"{name}: params DMA wrote {written} of {len(blob)} bytes"
                        )
            expected_end = self.PARAMS_BASE + int(decode_region["size"])
            if self.get_params_dram_addr() != expected_end:
                raise AssertionError(
                    f"decode BF16 O ends at 0x{self.get_params_dram_addr():X}, "
                    f"expected 0x{expected_end:X}"
                )
        except Exception:
            # A short overlay DMA destroys the prefill image without producing
            # a complete decode image, so neither phase may be retried in place.
            self._lm_weight_init_done = False
            self._decode_bf16_o_loaded = False
            raise

        self._decode_bf16_o_loaded = True
        self._lm_weight_init_done = False
        self._loud(
            f"  [Decode] BF16 O phase active: {decode_region['size'] / 2**20:.1f} "
            "MiB in shared params DRAM; IF4 prefill weights reclaimed"
        )

    def _prefill_use_streaming_quantized_projection(self, tag: str) -> bool:
        """Stream Omni's large-K MLP down projection on every FPGA engine.

        Omni has K=18944 here, beyond the general dynamic matmat tiler's
        12-bit Z-row field.  Gemma4 E2B's runtime-row, one-pass IF4 path supports
        its 16-column strips without expanding one copy per prompt row, and
        keeps all learned arithmetic on the FPGA.
        """
        return tag == "down"

    def _prefill_execution_rows(self, seq_len: int) -> int:
        """Choose the eight-engine prefill tile for the active U55 image.

        A 31-token prompt splits as seven four-row shards and one unsupported
        three-row MLP shard. Current images safely execute eight rows per engine,
        so prompts round to 64 global rows. The legacy host-segmented image uses
        one full 64-row block per engine (512 global rows). Padding keys are
        masked, preserving the logical prompt in either case.
        """
        if int(getattr(self, "fpga_build", -1)) == 0xE7AC2CAF:
            rows = 8 * 64
        else:
            rows = ((int(seq_len) + 63) // 64) * 64
        if int(seq_len) > rows:
            raise ValueError(
                f"{seq_len} live token(s) exceed the {rows}-row eight-engine "
                "prefill tile")
        if rows > self.PREFILL_MAX_SEQ_LEN:
            raise ValueError(
                f"{seq_len} live token(s) require {rows} FPGA rows, beyond the "
                f"{self.PREFILL_MAX_SEQ_LEN}-row prefill allocation")
        return rows

    def _ensure_fpga_embedding(self) -> None:
        """Upload eight quantized embedding row shards into private DRAM."""
        if getattr(self, "_fpga_embedding_loaded", False):
            return
        if not self._device_embedding_enabled():
            raise RuntimeError(
                "Qwen2.5-Omni requires a quantized device embedding"
            )
        if not hasattr(self, "_decode_shards"):
            raise RuntimeError(
                "FPGA embeddings must be loaded after decoder weight sharding"
            )
        artifact = getattr(self, "_embedding_artifact", None)
        if artifact is None:
            raise RuntimeError(
                "the LM artifact has no quantized embedding descriptor"
            )

        d = self._lm_dims()
        section = artifact["section"]
        precision = artifact.get("precision")
        configured_precision = self._cfg.get("precision", {}).get("embedding")
        if precision != configured_precision or precision not in {"if4", "if8"}:
            raise ValueError(
                f"embedding artifact precision {precision!r} does not match "
                f"configured device precision {configured_precision!r}"
            )
        if section.get("precision") != precision:
            raise ValueError(
                f"embedding manifest precision {section.get('precision')!r} "
                f"does not match {precision!r}"
            )
        shape = tuple(int(value) for value in section.get("shape", ()))
        expected_shape = (d["VOCAB"], d["H"])
        if shape != expected_shape:
            raise ValueError(
                f"embedding shape {shape} does not match {expected_shape}"
            )
        if section.get("layout") != "padded_row_scales_then_data":
            raise ValueError(f"unsupported embedding layout {section.get('layout')!r}")
        if int(section.get("block_size", 0)) != 64:
            raise ValueError("FPGA embedding quantization block size must be 64")
        blocks_per_row = d["H"] // 64
        scale_row_bytes = int(section.get("scale_row_bytes", 0))
        data_row_bytes = int(section.get("data_row_bytes", 0))
        if int(section.get("scale_values_per_row", 0)) != blocks_per_row:
            raise ValueError("embedding scale count does not match hidden width")
        if scale_row_bytes < blocks_per_row * 2 or scale_row_bytes % 64:
            raise ValueError("embedding scale rows must be full 64-byte AXI beats")
        expected_data_row_bytes = d["H"] // 2 if precision == "if4" else d["H"]
        if data_row_bytes != expected_data_row_bytes or data_row_bytes % 64:
            raise ValueError(
                f"embedding {precision.upper()} data rows are not "
                "AXI-beat aligned"
            )
        expected_size = d["VOCAB"] * (scale_row_bytes + data_row_bytes)
        if int(section["size"]) != expected_size:
            raise ValueError(
                f"embedding section is {section['size']} bytes, expected {expected_size}"
            )

        engines = int(getattr(self, "multi_core", 1))
        if engines != 8:
            raise ValueError(f"FPGA embedding layout requires eight engines, got {engines}")
        base_rows, extra = divmod(d["VOCAB"], engines)
        row_splits = []
        row_start = 0
        for engine_idx in range(engines):
            rows = base_rows + (1 if engine_idx < extra else 0)
            row_splits.append((engine_idx, row_start, rows))
            row_start += rows
        if row_start != d["VOCAB"]:
            raise AssertionError("embedding row split did not cover the vocabulary")

        arena = self.mc_arena
        weight_cursors = list(arena._weight_cursor)
        program_cursor = self.get_program_dram_addr()
        self._fpga_embedding_loaded = False
        try:
            # Allocate all shards before the first DMA so an arena overflow
            # cannot leave a half-populated table accepted as valid.
            shards = []
            for engine_idx, first_row, rows in row_splits:
                scale_bytes = rows * scale_row_bytes
                data_bytes = rows * data_row_bytes
                scale_addr = arena.alloc_weights(
                    engine_idx, scale_bytes, "token embedding scales"
                )
                data_addr = arena.alloc_weights(
                    engine_idx, data_bytes,
                    f"token embedding {precision.upper()} rows",
                )
                shards.append(
                    {
                        "engine": engine_idx,
                        "first_row": first_row,
                        "rows": rows,
                        "scale_addr": scale_addr,
                        "data_addr": data_addr,
                    }
                )

            # One reusable master-ISA slot holds at most three descriptors per
            # prefill row (scale load, dequantize, BF16 writeback), plus HALT.
            max_instructions = self.PREFILL_MAX_SEQ_LEN * 3 + 2
            program_bytes = ((max_instructions * 32 + 63) // 64) * 64
            lookup_program_addr = self.allocate_program_dram(
                program_bytes, label="lm.embedding_lookup"
            )
            if lookup_program_addr + program_bytes > self.WORKER_ISA_BASE:
                raise MemoryError(
                    "embedding lookup program reserve exceeds the 40-MiB master ISA slice"
                )

            table_start = int(artifact["file_offset"])
            all_scale_bytes = d["VOCAB"] * scale_row_bytes

            def upload_range(file_obj, source: int, destination: int,
                             size: int, label: str) -> None:
                # Keep host staging bounded; this is an uninterpreted byte copy,
                # not model arithmetic. Every chunk remains AXI-beat aligned.
                chunk_bytes = 8 * 2**20
                copied = 0
                while copied < size:
                    take = min(chunk_bytes, size - copied)
                    file_obj.seek(source + copied)
                    blob = file_obj.read(take)
                    if len(blob) != take:
                        raise RuntimeError(f"truncated params read for {label}")
                    written = self.dma_write(
                        _vl_lm.DMA_DEVICE_H2C, destination + copied, blob, take
                    )
                    if written != take:
                        raise IOError(
                            f"{label}: DMA wrote {written} of {take} bytes"
                        )
                    copied += take

            with open(artifact["bin_path"], "rb") as file_obj:
                for shard in shards:
                    first_row = shard["first_row"]
                    rows = shard["rows"]
                    upload_range(
                        file_obj,
                        table_start + first_row * scale_row_bytes,
                        shard["scale_addr"],
                        rows * scale_row_bytes,
                        f"embedding core {shard['engine']} scales",
                    )
                    upload_range(
                        file_obj,
                        table_start + all_scale_bytes + first_row * data_row_bytes,
                        shard["data_addr"],
                        rows * data_row_bytes,
                        f"embedding core {shard['engine']} "
                        f"{precision.upper()} rows",
                    )

            self._fpga_embedding_shards = shards
            self._fpga_embedding_scale_row_bytes = scale_row_bytes
            self._fpga_embedding_data_row_bytes = data_row_bytes
            self._fpga_embedding_blocks_per_row = blocks_per_row
            self._fpga_embedding_data_type = (
                _vl_lm.TYPE.IF4 if precision == "if4" else _vl_lm.TYPE.IF8
            )
            self._fpga_embedding_precision = precision
            self._fpga_embedding_program_addr = lookup_program_addr
            self._fpga_embedding_program_bytes = program_bytes
            self._fpga_embedding_loaded = True
        except Exception:
            arena._weight_cursor[:] = weight_cursors
            self._next_program_dram_addr = program_cursor
            for name in (
                "_fpga_embedding_shards",
                "_fpga_embedding_scale_row_bytes",
                "_fpga_embedding_data_row_bytes",
                "_fpga_embedding_blocks_per_row",
                "_fpga_embedding_data_type",
                "_fpga_embedding_precision",
                "_fpga_embedding_program_addr",
                "_fpga_embedding_program_bytes",
            ):
                if hasattr(self, name):
                    delattr(self, name)
            raise

        used = arena.usage()
        remaining = [
            arena.region(i).weight_limit - arena._weight_cursor[i]
            for i in range(engines)
        ]
        self._loud(
            f"  [LM] FPGA embedding: {expected_size / 2**20:.2f} MiB "
            f"{precision.upper()} "
            f"across 8 private windows ({expected_size / engines / 2**20:.2f} "
            f"MiB/core); private usage "
            f"{', '.join(f'{value / 2**20:.1f}' for value in used)} MiB, "
            f"minimum margin {min(remaining) / 2**20:.1f} MiB"
        )

    def _embedding_row_addresses(self, token: int) -> tuple[int, int]:
        self._ensure_fpga_embedding()
        token = int(token)
        d = self._lm_dims()
        if token < 0 or token >= d["VOCAB"]:
            raise ValueError(
                f"token ID {token} is outside embedding vocabulary 0..{d['VOCAB'] - 1}"
            )
        for shard in self._fpga_embedding_shards:
            local_row = token - shard["first_row"]
            if 0 <= local_row < shard["rows"]:
                return (
                    shard["scale_addr"]
                    + local_row * self._fpga_embedding_scale_row_bytes,
                    shard["data_addr"]
                    + local_row * self._fpga_embedding_data_row_bytes,
                )
        raise AssertionError(f"no FPGA embedding shard owns token {token}")

    def _emit_fpga_embedding_row(self, token: int, output_dram_addr: int) -> None:
        """Emit quantized row dequantization and BF16 DRAM writeback."""
        scale_addr, data_addr = self._embedding_row_addresses(token)
        uc = _vl_lm.user_dma_core
        blocks = self._fpga_embedding_blocks_per_row
        # Load the padded scale row (128 B for H=3584), not merely its 112
        # live bytes, so source, length, and the following row stay aligned on
        # both supported 256- and 512-bit AXI images. The DEQUANTIZE descriptor
        # consumes only `blocks` entries.
        self.accelerator_memory_to_scale_sram(
            scale_addr, self._fpga_embedding_scale_row_bytes // 2
        )
        output_type, output_row = self.sram_address_to_uram_address(0)
        self.ue_arithmetic_op(
            0,
            0,
            1,
            0,
            0,
            uc.LALU_MODE.BYPASS.value,
            0,
            output_type.value,
            0,
            output_row,
            uc.URAM_WRITE_SRC.URAM_WRITE_BACK.value,
            uc.UE_MODE.DEQUANTIZE,
            self._fpga_embedding_data_type.value,
            0,
            0,
            blocks,
            data_addr,
            self._fpga_embedding_data_row_bytes,
            blocks,
        )
        self.sram_to_accelerator_memory(
            sram_address=0,
            accelerator_dram_address=output_dram_addr,
            element_size=self._lm_dims()["H"],
        )

    def _load_device_embeddings(self, token_ids, output_dram_addr: int) -> None:
        """Execute prompt embedding lookup without materializing rows on host."""
        self._ensure_fpga_embedding()
        tokens = [int(token) for token in token_ids]
        if len(tokens) > self.PREFILL_MAX_SEQ_LEN:
            raise ValueError(
                f"embedding request has {len(tokens)} rows; limit is "
                f"{self.PREFILL_MAX_SEQ_LEN}"
            )
        self.clear_inst_id()
        self.clear_capture_buffer()
        self.start_capture()
        try:
            row_bytes = self._lm_dims()["H"] * 2
            for row, token in enumerate(tokens):
                self._emit_fpga_embedding_row(
                    token, output_dram_addr + row * row_bytes
                )
            self.generate_instruction_halt()
            self.stop_capture()
            expected = self.get_capture_instruction_size_bytes()
            if expected > self._fpga_embedding_program_bytes:
                raise MemoryError(
                    f"embedding lookup program needs {expected} bytes, reserved "
                    f"{self._fpga_embedding_program_bytes}"
                )
            written = self.write_captured_instructions_to_dram(
                self._fpga_embedding_program_addr
            )
            if written != expected:
                raise IOError(
                    f"embedding ISA DMA wrote {written} of {expected} bytes"
                )
        finally:
            if getattr(self, "is_capture_on", False):
                self.stop_capture()
            self.clear_capture_buffer()
        self.start_execute_from_dram(self._fpga_embedding_program_addr)
        self._wait_lm_queue(self, 30.0, "FPGA prompt embedding lookup")

    def _emit_device_decode_embedding(self, token: int,
                                      output_dram_addr: int) -> None:
        """Inline one FPGA lookup into the decoder's per-token preamble."""
        self._emit_fpga_embedding_row(token, output_dram_addr)

    def lm_reset_attention_state(self) -> None:
        """Clear primary and per-engine attention scratch after DRAM poison."""
        super().lm_reset_attention_state()
        if getattr(self, "multi_core", 1) <= 1:
            return
        d = self._lm_dims()
        aligned = (
            (self.PREFILL_MAX_SEQ_LEN + 63) // 64
        ) * 64
        elements = (
            (d["AHD"] + aligned) * aligned + aligned * d["AHD"]
        )
        zeros = torch.zeros(elements, dtype=torch.bfloat16)
        for address in self.LM_ATTN_SCRATCH_PER_ENGINE[1:]:
            self.dma_to_accelerator_memory(address, zeros)
        self._loud(
            f"  [LM] zeroed {len(self.LM_ATTN_SCRATCH_PER_ENGINE) - 1} "
            "private prefill scratch buffer(s)"
        )

    def _ensure_tokenizer(self):
        if not hasattr(self, "tokenizer"):
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(self.script_dir, self._cfg["paths"]["hf_model_dir"]),
                trust_remote_code=True,
            )
        return self.tokenizer


LM_QUANT_PRECISION = _vl_lm.LM_QUANT_PRECISION

__all__ = ["LM_QUANT_PRECISION", "Qwen25OmniLMMixin"]
