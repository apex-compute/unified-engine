#!/usr/bin/env python3
"""Gemma4 E2B LM method group (prefill + decode: compile + run, plus LM setup).

Split out of gemma4_e2b_test.py. ``Gemma4LMMixin`` carries the LM prefill/
decode kernels and is mixed into ``Gemma4_UnifiedEngine`` there; it is never
instantiated on its own. The top-level orchestration (compile_gemma4/run_gemma4)
lives in gemma4_e2b_test.py and calls these through self. Shared plumbing
resolves through the concrete class, so this module imports nothing from
gemma4_e2b_test (keeps the 3-file split import-cycle free).
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
from transformers import AutoTokenizer


class Gemma4LMMixin:
    """LM prefill/decode methods for Gemma4_UnifiedEngine (see module docstring)."""


    def set_prefill_seq(self, prompt: str | None = None) -> None:
        """Set self.prefill_seq from a text prompt (tokenize with chat template) or from config default."""
        if prompt is not None:
            conversation = [{"role": "user", "content": prompt}]
            prompt_with_template = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            self.prefill_seq = tuple(self.tokenizer.encode(prompt_with_template, add_special_tokens=True))
            print(f"Prefill from prompt ({len(self.prefill_seq)} tokens): {prompt!r}")
        else:
            self.prefill_seq = tuple(self._cfg["default_prefill_tokens"])
            decoded = self.tokenizer.decode(list(self.prefill_seq), skip_special_tokens=True)
            print(f"Prefill from default ({len(self.prefill_seq)} tokens): {decoded!r}")

    def _structural_token_ids(self) -> set:
        """Token ids never repetition-penalized (punctuation/whitespace/special);
        exempting these 'glue' tokens stops penalized text collapsing. Cached."""
        cached = getattr(self, "_struct_ids_cache", None)
        if cached is not None:
            return cached
        import string
        allowed = set(string.punctuation) | set(string.whitespace) | set("—–’‘“”…·•‹›«»¡¿")
        ids = set(int(i) for i in (getattr(self.tokenizer, "all_special_ids", []) or []))
        for i in range(self.EMBEDDING_ELEMENTS):
            s = self.tokenizer.decode([i]).strip()
            if s == "" or all(ch in allowed for ch in s):
                ids.add(i)
        self._struct_ids_cache = ids
        return ids

    def _structural_ids_tensor(self) -> torch.Tensor:
        t = getattr(self, "_struct_ids_tensor_cache", None)
        if t is None:
            t = torch.tensor(sorted(self._structural_token_ids()), dtype=torch.long)
            self._struct_ids_tensor_cache = t
        return t

    def _write_penalty_bias(self, prev_tokens) -> None:
        """Build the per-vocab additive bias from the windowed token frequency and
        DMA it to PENALTY_BIAS_DRAM. bias[t] = clamp(-alpha*count[t], min=-cap);
        structural tokens stay 0."""
        vocab = self.EMBEDDING_ELEMENTS
        alpha = float(getattr(self, "pen_alpha", 1.0))
        cap = float(getattr(self, "pen_cap", 20.0))
        W = int(getattr(self, "rep_window", 256))
        window = prev_tokens[-W:]
        count = torch.zeros(vocab, dtype=torch.float32)
        if window:
            win = torch.tensor(window, dtype=torch.long)
            count.index_add_(0, win, torch.ones(win.numel(), dtype=torch.float32))
            count[self._structural_ids_tensor()] = 0.0
        bias = (-alpha * count).clamp(min=-cap).to(torch.bfloat16).view(1, vocab)
        # --- anti-loop hard ban (overrides the structural exemption above) ---
        # The structural exemption keeps glue tokens ("|", newline, space) out of
        # the soft frequency penalty so penalized text doesn't turn to word-salad
        # -- but that is exactly why the penalty alone can't break a degenerate
        # collapse whose tokens ARE structural (e.g. a "|"<->"\n" 2-cycle, which
        # naive consecutive-run detection would miss). Instead: over the last
        # `recent_w` generated tokens, hard-ban any token that fills >= `loop_thr`
        # of them (single-token run = all; 2-cycle = ~half each). No coherent text
        # fills a third of a short window with one token, so it never fires on
        # real output. (E2B VLM is empirically immune to the E4B image cycle, but
        # the penalty path is shared; harmless here, present for symmetry.)
        # Tunable: GEMMA4_PEN_LOOP_RECENT (window, 0=off), GEMMA4_PEN_LOOP_THR.
        recent_w = int(getattr(self, "pen_loop_recent", 24))
        loop_thr = int(getattr(self, "pen_loop_thr", 8))
        if recent_w > 0 and len(prev_tokens) >= recent_w:
            from collections import Counter
            _cnt = Counter(int(t) for t in prev_tokens[-recent_w:])
            _ban = [tok for tok, c in _cnt.items() if c >= loop_thr]
            if _ban:
                bias[0, torch.tensor(_ban, dtype=torch.long)] = -1e9  # finite, bf16-safe
        self.dma_to_accelerator_memory(self.PENALTY_BIAS_DRAM, bias)

    def get_embedding_for_tokens(self, token_ids: list[int] | tuple) -> torch.Tensor:
        """Return (len(token_ids), vector_length) bfloat16 tensor from self.embedding_weight (HF, scale applied)."""
        tid_t = torch.tensor(token_ids, dtype=torch.long)
        out = torch.zeros(len(token_ids), self.vector_length, dtype=torch.bfloat16)
        valid = tid_t < self.embedding_weight.shape[0]
        out[valid] = self.embedding_weight[tid_t[valid]]
        return out

    def _load_rope_host(self, rope_theta: float | None = None, rope_local_base: float | None = None) -> None:
        """Generate RoPE (cos, cos, -sin, sin) on host and write to DRAM. Uses config for sizes and num_positions.
        LOCAL: head_dim=256, full rotation. GLOBAL: head_dim=512, partial rotation (first 128 dims)."""
        rope_cfg = self._cfg["special"]["rope"]
        theta = rope_theta if rope_theta is not None else rope_cfg["theta"]
        local_base = rope_local_base if rope_local_base is not None else rope_cfg["local_base"]
        # RoPE must cover every position up to the context limit. Derive the
        # count from MAX_CONTEXT_SIZE (falling back to the config value when it is
        # larger) so raising the context never silently runs past the table.
        num_rope_positions = max(int(rope_cfg["num_positions"]), int(self.MAX_CONTEXT_SIZE))
        partial_rotary_factor = rope_cfg["partial_rotary_factor_global"]

        # LOCAL RoPE: head_dim_sliding=256, full rotation, D=128
        D_local = self.head_dim_sliding // 2  # 128
        inv_freq_local = 1.0 / (local_base ** (torch.arange(D_local, dtype=torch.float32) / D_local))
        pos = torch.arange(num_rope_positions, dtype=torch.float32)
        freqs_local = torch.outer(pos, inv_freq_local)
        cos_local = freqs_local.cos().to(torch.bfloat16)
        sin_local = freqs_local.sin().to(torch.bfloat16)
        rope_local = torch.cat([cos_local, cos_local, -sin_local, sin_local], dim=1)
        # Size the table to the actual data (num_rope_positions x head_dim_sliding*2
        # bf16); the baked ROPE_LOCAL_SIZE region only fits 1024 positions and would
        # truncate anything larger.
        raw = rope_local.contiguous().view(torch.uint8).numpy().tobytes()
        sz = len(raw)
        addr = self.allocate_params_dram(sz)
        self.dma_write(DMA_DEVICE_H2C, addr, raw, sz)
        self.DRAM_ADDR_ROPE_LOCAL = addr

        # GLOBAL RoPE: head_dim=512, partial_rotary_factor=0.25, rotary_dims=128, D=64
        rotary_dims = int(self.head_dim * partial_rotary_factor)  # 128
        D_global = rotary_dims // 2  # 64
        inv_freq_global = 1.0 / (theta ** (torch.arange(D_global, dtype=torch.float32) / D_global))
        freqs_global = torch.outer(pos, inv_freq_global)
        cos_global = freqs_global.cos().to(torch.bfloat16)
        sin_global = freqs_global.sin().to(torch.bfloat16)
        rope_global = torch.cat([cos_global, cos_global, -sin_global, sin_global], dim=1)
        # Size to the actual data (num_rope_positions x rotary_dims*2 bf16); see the
        # ROPE_LOCAL note above on why the baked region size would truncate.
        raw = rope_global.contiguous().view(torch.uint8).numpy().tobytes()
        sz = len(raw)
        addr = self.allocate_params_dram(sz)
        self.dma_write(DMA_DEVICE_H2C, addr, raw, sz)
        self.DRAM_ADDR_ROPE_GLOBAL = addr

    def _load_host_weights_from_combined_bin(self, host_section: dict, base_offset: int) -> None:
        """mmap the combined weights bin and create read-only torch tensor
        views over the host section. Zero-copy: nothing materializes in
        RSS until a row is indexed.

        `host_section["manifest"]` gives tensor offsets RELATIVE to the
        host section start; we add `base_offset` (the host section's
        absolute file offset) when creating the view.
        """
        sub = host_section["manifest"]

        def _view(section_name: str) -> torch.Tensor:
            s = sub[section_name]
            shape = tuple(s["shape"])
            n_elems = 1
            for d in shape:
                n_elems *= d
            return torch.frombuffer(
                self.weight_bin,
                dtype=torch.bfloat16,
                count=n_elems,
                offset=base_offset + s["offset"],
            ).reshape(shape)

        self.embed_tokens_per_layer_weight = _view("embed_tokens_per_layer")
        self._layer_scalars = list(sub["layer_scalars"])
        self._kv_shared_map = {int(k): int(v) for k, v in sub.get("kv_shared_map", {}).items()}
        print(f"[weight_init] host section mmap'd at file offset 0x{base_offset:X}: "
              f"embed_tokens_per_layer={tuple(self.embed_tokens_per_layer_weight.shape)} bf16 "
              f"({sub['embed_tokens_per_layer']['size']/1024**3:.2f} GiB, page-cached on demand)")

    def weight_init(self) -> None:
        """Ensure weight bin exists (generate from HF if missing), then mmap it
        and initialize FPGA DRAM: embedding, layers from bin, RoPE, OUTPUT_NORM/LM_HEAD.

        Host-side tensors needed for per-layer-input computation
        (per_layer_embed_tokens, per_layer_model_proj, per_layer_proj_norm,
        layer_scalars, kv_shared_map) come from `host_weights.bin` if it
        exists, mmap'd so RSS stays minimal. Otherwise we fall back to
        loading the HF model — which costs 6-12 GB host RAM and is OOM on
        a 16 GB Raspberry Pi. The first run on a beefier machine should
        generate the side-cache so subsequent runs anywhere can skip the
        HF model entirely.
        """
        import mmap as _mmap

        full_path = os.path.join(self.script_dir, self._weights_bin_rel)
        if os.path.exists(full_path):
            print(f"Weight bin exists, skip generation: {full_path}")
        else:
            print(f"Weight bin not found, generating: {full_path}")
            self._weight_bin_generate(output_path=full_path)

        # mmap the weight bin — read-only, OS pages in only what's touched.
        # Replaces a 2.4 GB f.read() that pinned the whole bin in RSS.
        self._weight_bin_fp = open(full_path, "rb")
        self.weight_bin = _mmap.mmap(self._weight_bin_fp.fileno(), 0,
                                     prot=_mmap.PROT_READ)

        # Master manifest: one JSON per combined weight bin. Holds the
        # offsets/sizes of the three sections (lm, vision, host) plus each
        # section's sub-manifest. If missing, the bin was generated with
        # the old multi-file layout and we need to regenerate.
        master_meta_path = full_path.rsplit(".", 1)[0] + ".json"
        if not os.path.exists(master_meta_path):
            raise RuntimeError(
                f"weights master manifest missing: {master_meta_path}\n"
                f"This bin was produced by the old multi-file layout. "
                f"Delete {full_path} (and any stale host_weights.bin / "
                f"vision_weights.bin) and re-run gemma4_e2b_test.py to "
                f"regenerate the combined bin.")
        with open(master_meta_path, "r") as f:
            self._weights_master = json.load(f)

        # Embedding: a zero-copy mmap view directly into the weight bin.
        # No 770 MB host allocation; only the touched rows (one per decode
        # token, ~3 KB) cost RSS. Read-only is fine because we only do
        # `embedding_weight[token_ids]` lookups.
        emb_cfg = self._cfg["special"]["embedding"]
        token_embd_offset = self._parse_offset(emb_cfg["token_embd_offset"])
        vocab_size  = self.EMBEDDING_ELEMENTS         # 262144
        emb_dim     = self.vector_length              # 1536
        self.embedding_weight = torch.frombuffer(
            self.weight_bin,
            dtype=torch.bfloat16,
            count=vocab_size * emb_dim,
            offset=token_embd_offset,
        ).reshape(vocab_size, emb_dim)

        host_section = self._weights_master["host_section"]
        self._load_host_weights_from_combined_bin(host_section, host_section["offset"])

        # Tokenizer: from the bundled subset; full HF model not needed.
        tok_subset = os.path.join(self.script_dir, "gemma4_e2b_bin", "tokenizer")
        if os.path.exists(os.path.join(tok_subset, "tokenizer.json")):
            tok_dir = tok_subset
        else:
            tok_dir = os.path.join(self.script_dir, self._cfg["paths"]["hf_model_dir"])
        self.tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)

        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        base_layer0 = self.weight_defs["BLK0_ATTN_NORM_WEIGHT"]
        blk0_regions = [
            (s["key"], f"{s['key']}_SIZE", s["attr"])
            for s in self._cfg["layers"]["structure"]
        ]
        non_layer = [
            (s["key"], f"{s['key']}_SIZE", s["attr"])
            for s in self._cfg["layers"]["non_layer"]
            if s["key"] not in ("ROPE_LOCAL", "ROPE_GLOBAL")  # loaded via _load_rope_host()
        ]

        last_structure_key = self._cfg["layers"]["structure"][-1]["key"]
        layer0_end = (self.weight_defs[last_structure_key] - base_layer0
                      + self.weight_defs[f"{last_structure_key}_SIZE"])
        assert layer0_end <= LAYER_WEIGHT_SIZE, (
            f"Layer 0 size overflow: computed {layer0_end} > LAYER_WEIGHT_SIZE {LAYER_WEIGHT_SIZE}"
        )

        print(f"\n--- Loading weights to DRAM ---")
        layers_total = self.LAYER_SIZE * LAYER_WEIGHT_SIZE
        layers_base_dram = self.allocate_params_dram(layers_total)
        load_t0 = time.perf_counter()
        for layer_idx in range(self.LAYER_SIZE):
            if layer_idx > 0 and layer_idx % 10 == 0:
                print(f"    layer {layer_idx}/{self.LAYER_SIZE} loaded ({time.perf_counter()-load_t0:.1f}s)")
            for off_key, sz_key, attr in blk0_regions:
                off = self.weight_defs[off_key]
                sz = self.weight_defs[sz_key]
                bin_off = off + layer_idx * LAYER_WEIGHT_SIZE
                raw = self.weight_bin[bin_off : bin_off + sz]
                offset_in_layer = off - base_layer0
                dram_addr = layers_base_dram + layer_idx * LAYER_WEIGHT_SIZE + offset_in_layer
                self.dma_write(DMA_DEVICE_H2C, dram_addr, raw, sz)
            if layer_idx == 0:
                for off_key, sz_key, attr in blk0_regions:
                    off = self.weight_defs[off_key]
                    offset_in_layer = off - base_layer0
                    setattr(self, attr, layers_base_dram + offset_in_layer)
        print(f"  Loaded {self.LAYER_SIZE} layers ({layers_total/(1024*1024):.1f} MB)")

        for off_key, sz_key, attr in non_layer:
            off = self.weight_defs[off_key]
            sz = self.weight_defs[sz_key]
            raw = self.weight_bin[off : off + sz]
            addr = self.allocate_params_dram(sz)
            self.dma_write(DMA_DEVICE_H2C, addr, raw, sz)
            setattr(self, attr, addr)

        self._load_rope_host()
        print(f"  Total weight DRAM: {self.get_params_dram_usage()/(1024*1024):.1f} MB")
        print("Tokenizer loaded.")

    def tensor_init(self) -> None:
        """Initialize hardware DRAM for Gemma4 E2B model.

        Unique KV slots are packed using their owning layer's actual head
        dimension. Sliding-attention rows therefore use 256 elements and
        global-attention rows use 512 elements, matching unified attention's
        contiguous K/V layout directly.
        """
        # Per-token work length. Every LM intermediate (norms, projections, MLP,
        # attention staging, per-layer input prep) only ever holds up to
        # `prefill_seq_len` rows: prefill processes at most `max_prefill_seq_len`
        # tokens per pass (run_prefill asserts this) and decode processes one.
        # Sizing these buffers by the work length instead of MAX_CONTEXT_SIZE keeps
        # the tensor-DRAM footprint flat as the context grows (e.g. 1024 -> 4096);
        # only the KV cache (below) genuinely scales with MAX_CONTEXT_SIZE, and the
        # attention scratch/bias buffers scale with `attention_aligned_seq_len`.
        prefill_seq_len = min(self.max_prefill_seq_len, self.MAX_CONTEXT_SIZE)
        seq_len = prefill_seq_len
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        # Unified attention scratch/bias buffers are largest during prefill, but
        # decode can still require MAX_CONTEXT_SIZE KV rows. Size the shared
        # attention buffers for the larger aligned dimension.
        prefill_q_seq_len = prefill_seq_len * self.group_size
        prefill_aligned_seq_len = ((prefill_q_seq_len + 63) // 64) * 64
        decode_aligned_seq_len = ((self.MAX_CONTEXT_SIZE + 63) // 64) * 64
        attention_aligned_seq_len = max(prefill_aligned_seq_len, decode_aligned_seq_len)

        # Build compact KV slot map: only layers that own KV state get a slot.
        # KV-shared layers point at their reference layer's slot, so L15-34 do not
        # consume cache space (saves ~40 MB at MAX_CONTEXT_SIZE=1024).
        non_shared_layers = [l for l in range(self.LAYER_SIZE) if l not in self._kv_shared_map]
        self._kv_slot_for_layer = {}
        self._kv_offset_for_layer = {}
        self._kv_row_bytes_for_layer = {}
        _kv_offset = 0
        for slot, l in enumerate(non_shared_layers):
            self._kv_slot_for_layer[l] = slot
            _head_dim, _, _ = self._get_layer_attention_dims(l)
            _row_bytes = _head_dim * self.bytes_per_element
            self._kv_offset_for_layer[l] = _kv_offset
            self._kv_row_bytes_for_layer[l] = _row_bytes
            _kv_offset += self.MAX_CONTEXT_SIZE * _row_bytes
        for shared_l, ref_l in self._kv_shared_map.items():
            self._kv_slot_for_layer[shared_l] = self._kv_slot_for_layer[ref_l]
            self._kv_offset_for_layer[shared_l] = self._kv_offset_for_layer[ref_l]
            self._kv_row_bytes_for_layer[shared_l] = self._kv_row_bytes_for_layer[ref_l]
            _shared_dim, _, _ = self._get_layer_attention_dims(shared_l)
            if _shared_dim * self.bytes_per_element != self._kv_row_bytes_for_layer[ref_l]:
                raise ValueError(
                    f"KV-shared layer {shared_l} head_dim={_shared_dim} does not match "
                    f"reference layer {ref_l} row size")
        self._num_kv_slots = len(non_shared_layers)
        self._kv_cache_bytes = _kv_offset
        _uniform_bytes = self._num_kv_slots * self.MAX_CONTEXT_SIZE * self.k_size
        _compact_saved = 2 * (_uniform_bytes - self._kv_cache_bytes)
        print(
            f"KV cache: {self._num_kv_slots} unique compact slots "
            f"({self._kv_cache_bytes * 2 / (1024*1024):.1f} MB K+V, "
            f"saved {_compact_saved / (1024*1024):.1f} MB vs padded slots)")
        self.LAYER0_V_DRAM = self.allocate_tensor_dram(self._kv_cache_bytes)
        self.LAYER0_K_ROPE_DRAM = self.allocate_tensor_dram(self._kv_cache_bytes)
        zero_pad = torch.zeros(self._kv_cache_bytes // self.bytes_per_element, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_K_ROPE_DRAM, zero_pad)
        # Allocate memory for constant zero tensor, identity matrix, and bias:
        zero_add = torch.zeros(seq_len * self.head_dim * self.bytes_per_element, dtype=torch.bfloat16)
        self.ZERO_DRAM_ADDR = self.allocate_tensor_dram(seq_len * self.head_dim * self.bytes_per_element)
        self.dma_to_accelerator_memory(self.ZERO_DRAM_ADDR, zero_add)
        self.IDENTITY_DRAM_ADDR = self.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * self.bytes_per_element)
        self.dma_to_accelerator_memory(self.IDENTITY_DRAM_ADDR, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))
        # Allocate memory for attention and zero pad. Prefill uses
        # seq_len*group_size rows; decode uses MAX_CONTEXT_SIZE KV rows, so size
        # the shared buffers for the larger aligned dimension.
        self.LAYER0_FLASH_Q_DRAM = self.allocate_tensor_dram(attention_aligned_seq_len * self.head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_K_DRAM = self.allocate_tensor_dram(attention_aligned_seq_len * self.head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_V_DRAM = self.allocate_tensor_dram(attention_aligned_seq_len * self.head_dim * self.bytes_per_element)
        zero_pad = torch.zeros(attention_aligned_seq_len * self.head_dim * self.bytes_per_element, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_Q_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_K_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_V_DRAM, zero_pad)
        # Allocate memory for layer intermediate tensors:
        self.LAYER0_INPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_Q_DRAM = self.allocate_tensor_dram(seq_len * self.q_size)
        self.LAYER0_K_DRAM = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_K_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_Q_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.q_size)
        self.LAYER0_FLASH_OUTPUT_DRAM = self.allocate_tensor_dram(attention_aligned_seq_len * self.head_dim * self.bytes_per_element)
        # unified_attention_core scratch layout:
        #   V.T [HD, S] + scores [S, S] + scaled_q [batch, HD].
        # Worst case batch <= S, so allocate S*S + 2*HD*S elements.
        self.LAYER0_FLASH_SCRATCH_DRAM = self.allocate_tensor_dram(
            (attention_aligned_seq_len * attention_aligned_seq_len
             + 2 * self.head_dim * attention_aligned_seq_len) * self.bytes_per_element)
        # Two full-matrix bias buffers: full-attention layers attend to
        # the entire causal window, sliding-attention layers are limited to
        # `sliding_window` tokens. compile_prefill / compile_decoder pick the
        # right address per layer; run_prefill / run_decoder upload both.
        self.LAYER0_FLASH_BIAS_FULL_DRAM = self.allocate_tensor_dram(attention_aligned_seq_len * attention_aligned_seq_len * self.bytes_per_element)
        self.LAYER0_FLASH_BIAS_SLIDING_DRAM = self.allocate_tensor_dram(attention_aligned_seq_len * attention_aligned_seq_len * self.bytes_per_element)
        # Backwards-compat alias (older callers use the singular name).
        self.LAYER0_FLASH_BIAS_DRAM = self.LAYER0_FLASH_BIAS_FULL_DRAM
        self.LAYER0_ATTN_PROJ_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_ATTN_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_ATTN_RESIDUAL_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_MLP_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        mlp_max = max(self.mlp_elements, self.mlp_elements_wide)
        self.LAYER0_MLP_GATE_DRAM = self.allocate_tensor_dram(seq_len * mlp_max * 2)
        self.LAYER0_MLP_UP_DRAM = self.allocate_tensor_dram(seq_len * mlp_max * 2)
        self.LAYER0_MLP_MULT_DRAM = self.allocate_tensor_dram(seq_len * mlp_max * 2)
        self.LAYER0_MLP_DOWN_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_MLP_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.OUTPUT_NORM_DRAM = self.allocate_tensor_dram(1 * self.vector_length * self.bytes_per_element)
        self.LOGITS_DRAM = self.allocate_tensor_dram(1 * self.EMBEDDING_ELEMENTS * self.bytes_per_element)
        # Reserved for a future streaming LM-head penalty implementation.
        # GEMMA4_PENALTY is temporarily rejected.
        self.PENALTY_BIAS_DRAM = self.allocate_tensor_dram(1 * self.EMBEDDING_ELEMENTS * self.bytes_per_element)

        # Per-layer input preparation + injection buffers.  The host uploads
        # only token-indexed rows from embed_tokens_per_layer; FPGA performs
        # model projection, RMSNorm, add and scaling into PER_LAYER_INPUTS_DRAM.
        # All three are per-pass work buffers: prefill fills up to prefill_seq_len
        # tokens ([token, layer, dim]); decode recomputes the single current token
        # at the base each step (see the decode per-layer prep, which writes with
        # OUTPUT_DRAM_ADDR=PER_LAYER_INPUTS_DRAM). So size by the work length, not
        # MAX_CONTEXT_SIZE.
        pli_elements = seq_len * self.LAYER_SIZE * self.per_layer_input_dim
        self.PER_LAYER_EMBED_DRAM = self.allocate_tensor_dram(
            pli_elements * self.bytes_per_element)
        self.PER_LAYER_MODEL_PROJ_OUTPUT_DRAM = self.allocate_tensor_dram(
            pli_elements * self.bytes_per_element)
        # Final layout is [token, layer, dim], consumed by injection blocks.
        self.PER_LAYER_INPUTS_DRAM = self.allocate_tensor_dram(pli_elements * self.bytes_per_element)
        # Intermediate DRAMs for per-layer injection
        self.LAYER0_PER_LAYER_GATE_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.per_layer_input_dim * self.bytes_per_element)
        self.LAYER0_PER_LAYER_PROJ_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * self.bytes_per_element)

        print(f"    Tensor DRAM usage: {self.get_tensor_dram_usage()/(1024*1024):.1f} MB")

    # Per-layer dim resolution for Gemma4's heterogeneous stack. Two independent
    # axes → 4 layer buckets. Values below are for the current config (35 layers,
    # full_attention_layers={4,9,14,19,24,29,34}, double_wide_mlp_first_layer=15,
    # group_size=8, partial_rotary_factor_global=0.25):
    #
    #   attention type (every 5th layer is full/global, rest sliding-window):
    #     full    : head_dim=512, q_size=4096, k_size=512, rope_N=128 (partial)
    #     sliding : head_dim=256, q_size=2048, k_size=256, rope_N=256 (full)
    #   MLP width (doubles from layer 15 onward, the KV-shared layers):
    #     layer <15 : mlp=6144        layer >=15 : mlp=12288
    #
    #   layers 0-3,5-8,10-13         -> (256,2048,256) rope 256 mlp 6144
    #   layers 4,9,14                -> (512,4096,512) rope 128 mlp 6144
    #   layers 15-18,20-23,25-28,30-33 -> (256,2048,256) rope 256 mlp 12288
    #   layers 19,24,29,34           -> (512,4096,512) rope 128 mlp 12288
    def _get_layer_attention_dims(self, layer_idx: int) -> tuple[int, int, int]:
        """Return (cur_head_dim, cur_q_size, cur_k_size) for a given layer index."""
        if layer_idx in self._full_attention_layers:
            cur_head_dim = self.head_dim  # 512
            cur_q_size = cur_head_dim * self.group_size  # 4096
            cur_k_size = cur_head_dim  # 512
        else:
            cur_head_dim = self.head_dim_sliding  # 256
            cur_q_size = cur_head_dim * self.group_size  # 2048
            cur_k_size = cur_head_dim  # 256
        return cur_head_dim, cur_q_size, cur_k_size

    def _get_rope_dims(self, layer_idx: int) -> int:
        """Return the number of dims to apply RoPE on for a given layer.
        Sliding layers: full rotation on head_dim_sliding=256, so N=256.
        Full attention layers: partial rotation, only first 128 dims of head_dim=512, so N=128."""
        if layer_idx in self._full_attention_layers:
            partial_rotary_factor = self._cfg["special"]["rope"]["partial_rotary_factor_global"]
            return int(self.head_dim * partial_rotary_factor)  # 128
        else:
            return self.head_dim_sliding  # 256

    def _get_mlp_elements(self, layer_idx: int) -> int:
        """Return MLP intermediate size for a given layer (wide for KV-shared layers)."""
        if layer_idx >= self._double_wide_mlp_first:
            return self.mlp_elements_wide
        return self.mlp_elements

    def _emit_gqa_duplicate_pbi(self, src_dram_base: int, dst_dram_base: int,
                                cur_head_dim: int, template_seq_len: int,
                                gpr_seq_len: int, sram_addr: int = 0x10000,
                                src_row_bytes: int = None) -> None:
        """§4.4 PBI hardware loop replacing the static per-token GQA replication.
        Per token (outer loop, runtime trip count = gpr_seq_len) read its one
        cur_head_dim row from the KV cache (src stride src_row_bytes, default one
        head; test.py passes k_size) into a fixed SRAM slot, then scatter
        group_size contiguous copies into the flat FLASH buffer via a pbi
        write-pointer that auto-advances one head per call. SRAM-safe (one row
        resident). Output layout: FLASH row (i*group_size + g)."""
        bpe = self.bytes_per_element
        row_bytes = cur_head_dim * bpe          # dst scatter stride (= one head)
        src_stride = src_row_bytes if src_row_bytes is not None else row_bytes
        _, sram_words = self.sram_address_to_uram_address(sram_addr)
        ptr = self.alloc_inst_ptr()
        self.generate_instruction_pbi_init(
            dram_shared_addr=dst_dram_base, dma_length=row_bytes,
            output_size=0, uram_length=0,
            uram_a_start_addr=sram_words, uram_b_start_addr=sram_words,
            uram_wb_addr=0, uram_dst_addr=0, fmax_context_addr=0,
            inst_pointer_idx=ptr)
        t_reg = self.alloc_isa_reg()
        self.generate_instruction_add_set(t_reg, 0)
        self.loop_start(loop_cnt=template_seq_len, gpr_loop_cnt=gpr_seq_len)
        self.generate_instruction_reg_mul_imm(
            self.TMP_REG, t_reg, ue_35bit_addr_shifter(src_stride))
        self.generate_instruction_add_imm(
            self.TMP_REG, ue_35bit_addr_shifter(src_dram_base), self.TMP_REG)
        self.accelerator_memory_to_sram(
            accelerator_dram_address=0, sram_address=sram_addr,
            element_size=cur_head_dim, general_reg_src=self.TMP_REG)
        self.loop_start(self.group_size)
        self.sram_to_accelerator_memory(
            sram_address=0, accelerator_dram_address=row_bytes,
            element_size=cur_head_dim, inst_pointer_idx=ptr,
            memcpy_length_bytes=0)
        self.loop_end()
        self.generate_instruction_add_inc(t_reg)
        self.loop_end()
        self.release_isa_reg()       # t_reg
        self.release_inst_ptr(ptr)

    def _emit_strided_copy_pbi(self, src_base: int, dst_base: int, copy_elems: int,
                               src_row_bytes: int, dst_row_bytes: int,
                               n_template: int, gpr_loop: int,
                               sram_addr: int = 0x10000) -> None:
        """§4.4 PBI hardware loop: for n_template rows (runtime trip count =
        gpr_loop) copy copy_elems bf16 from src to dst, advancing the source DRAM
        addr by src_row_bytes and the dest by dst_row_bytes each iteration (both
        register-computed → arbitrary strides). Used for the partial-rotary
        gather/scatter and the non-rotated-dim pass-through."""
        i_reg = self.alloc_isa_reg()
        self.generate_instruction_add_set(i_reg, 0)
        self.loop_start(loop_cnt=n_template, gpr_loop_cnt=gpr_loop)
        self.generate_instruction_reg_mul_imm(
            self.TMP_REG, i_reg, ue_35bit_addr_shifter(src_row_bytes))
        self.generate_instruction_add_imm(
            self.TMP_REG, ue_35bit_addr_shifter(src_base), self.TMP_REG)
        self.accelerator_memory_to_sram(
            accelerator_dram_address=0, sram_address=sram_addr,
            element_size=copy_elems, general_reg_src=self.TMP_REG)
        self.generate_instruction_reg_mul_imm(
            self.TMP_REG, i_reg, ue_35bit_addr_shifter(dst_row_bytes))
        self.generate_instruction_add_imm(
            self.TMP_REG, ue_35bit_addr_shifter(dst_base), self.TMP_REG)
        self.sram_to_accelerator_memory(
            sram_address=sram_addr, accelerator_dram_address=0,
            element_size=copy_elems, general_reg_src=self.TMP_REG)
        self.generate_instruction_add_inc(i_reg)
        self.loop_end()
        self.release_isa_reg()       # i_reg

    def compile_prefill(self, seq_len: int, layer_size: int = 35, profile: bool = False) -> tuple[None, int]:
        """Emit one runtime-length prefill program into the capture buffer.

        ``seq_len`` is the current prompt length used for dynamic-core defaults
        and FLOP accounting. Runtime GPRs still supply the live row counts, so
        the captured program serves every prompt up to ``prefill_max_seq_len``.
        Only the non-dynamic per-layer preparation projection is emitted at the
        configured maximum length.

        ``profile``: mirror compile_decoder — emit a HALT at each per-layer phase
        boundary and record the resume address (the next instruction), so
        run_gemma4_profile can time each phase's HW latency. Checkpoints are
        placed only at UNCONDITIONAL points (never inside a loop_start/loop_end
        or a per-layer kv-shared branch), so every layer contributes one sample
        per phase. Stored in self._prefill_checkpoints (empty when not profiling).
        A profile-compiled prefill can only be run segment-by-segment (each HALT
        stops the FPGA), never in one shot.
        """
        seq_len = int(seq_len)
        template_seq_len = int(self._cfg["model"].get(
            "prefill_max_seq_len", self.max_prefill_seq_len))
        self.seq_len = seq_len
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        self._set_silent(True)
        total_flops = 0
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        self._loud(f"  Emitting dynamic prefill: {layer_size} layers, accounting_seq={seq_len}, attention=unified-inline"
                        + (" (+profile checkpoints)" if profile else ""))
        checkpoints: list[list] = []
        def _checkpoint(name: str) -> None:
            if not profile:
                return
            self.generate_instruction_halt()
            resume = self.get_program_dram_addr() + self.capture_count * INSTRUCTION_SIZE_BYTES
            checkpoints.append([name, f"0x{resume:X}"])
        def _projection_core(**kwargs) -> int:
            """Dynamic quantized projection selected for the whole prefill stage."""
            kwargs.setdefault("gpr_M_reg", self.gpr_seq_len)
            if self.prefill_kernel == "matmatmul":
                kwargs.setdefault("is_B_quantized", True)
                return self.matmat_mul_core(**kwargs)
            kwargs.pop("is_B_quantized", None)
            return self.quantized_matmat_core(**kwargs)
        prefill_t0 = time.perf_counter()

        # Per-layer input preparation (prefill): project each token to all 35 layer slices, then normalize, add its per-layer embedding, and scale. The projection uses template M; the row-wise stages use live seq_len.
        per_layer_dim = self.per_layer_input_dim
        per_layer_rows = seq_len * self.LAYER_SIZE
        # TODO: Fix the non-quantized BF16 dynamic matmul path, then use
        # matmat_mul_core with gpr_M_reg=self.gpr_seq_len here.
        total_flops += self.matmat_mul_core_legacy(
            M=template_seq_len, K=self.vector_length,
            N=self.LAYER_SIZE * per_layer_dim,
            A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
            B_DRAM_ADDR=self.DRAM_ADDR_PER_LAYER_MODEL_PROJ,
            OUTPUT_DRAM_ADDR=self.PER_LAYER_MODEL_PROJ_OUTPUT_DRAM)
        per_layer_rows_reg = self.alloc_isa_reg()
        self.generate_instruction_reg_mul_imm(per_layer_rows_reg, self.gpr_seq_len, self.LAYER_SIZE)
        total_flops += self.rms_norm_core_dram(
            M=per_layer_rows, N=per_layer_dim,
            A_DRAM_ADDR=self.PER_LAYER_MODEL_PROJ_OUTPUT_DRAM,
            OUTPUT_DRAM_ADDR=self.PER_LAYER_INPUTS_DRAM,
            GAMMA_DRAM_ADDR=self.DRAM_ADDR_PER_LAYER_PROJ_NORM,
            gpr_M_reg=per_layer_rows_reg)
        total_flops += self.eltwise_core_dram(
            M=per_layer_rows, N=per_layer_dim,
            dram_a=self.PER_LAYER_INPUTS_DRAM,
            dram_b=self.PER_LAYER_EMBED_DRAM,
            dram_out=self.PER_LAYER_INPUTS_DRAM,
            mode=UE_MODE.ELTWISE_ADD,
            gpr_M_reg=per_layer_rows_reg)
        total_flops += self.eltwise_core_dram(
            M=per_layer_rows, N=per_layer_dim,
            dram_a=self.PER_LAYER_INPUTS_DRAM,
            dram_b=None,
            dram_out=self.PER_LAYER_INPUTS_DRAM,
            mode=UE_MODE.MUL_BROADCAST,
            scalar=self._per_layer_input_scale,
            gpr_M_reg=per_layer_rows_reg)
        self.release_isa_reg()
        _checkpoint("per_layer_prepare")

        for layer_idx in range(layer_size):
            if layer_idx > 0 and layer_idx % 10 == 0:
                self._loud(f"    prefill layer {layer_idx}/{layer_size} ({time.perf_counter()-prefill_t0:.1f}s)")
            layer_off = layer_idx * LAYER_WEIGHT_SIZE
            cur_head_dim, cur_q_size, cur_k_size = self._get_layer_attention_dims(layer_idx)
            cur_mlp = self._get_mlp_elements(layer_idx)
            rope_n = self._get_rope_dims(layer_idx)

            # Layer-input source (mirrors compile_decoder): layer 0 reads the
            # uploaded LAYER0_INPUT; layer i>0 reads the previous layer's
            # LAYER0_OUTPUT directly — no seq_len-sized copy. LAYER0_OUTPUT is
            # overwritten only at this layer's final MLP+injection residual,
            # AFTER it is consumed here and as the attention-residual source.
            layer_input_addr = self.LAYER0_INPUT_DRAM if layer_idx == 0 else self.LAYER0_OUTPUT_DRAM
            total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=layer_input_addr,
                                OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off,
                                                gpr_M_reg=self.gpr_seq_len)
            # Q projection: N = cur_q_size (actual per-layer Q output dim)
            total_flops += _projection_core(M=seq_len, K=self.vector_length, N=cur_q_size,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                is_B_quantized=True,
                data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off,
                gpr_M_reg=self.gpr_seq_len,
                )
            if layer_idx not in self._kv_shared_map:
                # Non-shared layer: compute K/V projections normally.
                # Shared layers skip entirely — their attention reads K/V directly
                # from the reference layer's slot via _kv_slot_for_layer.
                # K projection: N = cur_k_size (actual per-layer K output dim)
                total_flops += _projection_core(M=seq_len, K=self.vector_length, N=cur_k_size,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off,
                    gpr_M_reg=self.gpr_seq_len,
                    )
                # V projection: write to temp buffer first, then scatter to KV cache at k_size stride
                total_flops += _projection_core(M=seq_len, K=self.vector_length, N=cur_k_size,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,  # temp buffer
                    is_B_quantized=True,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off,
                    gpr_M_reg=self.gpr_seq_len,
                    )
                # V norm + scatter to KV cache at k_size stride — §4.4 PBI loop
                # (was a per-token Python unroll). rms_norm_core works on a fixed
                # SRAM slot; the per-token read (FLASH_V, cur_k_size stride) and
                # write (KV cache, k_size stride) addrs are register-computed, so
                # the body is emitted once and hardware-looped over gpr_seq_len.
                kv_row_bytes = self._kv_row_bytes_for_layer[layer_idx]
                v_cache_base = self.LAYER0_V_DRAM + self._kv_offset_for_layer[layer_idx]
                _vi = self.alloc_isa_reg()
                self.generate_instruction_add_set(_vi, 0)
                self.loop_start(loop_cnt=seq_len, gpr_loop_cnt=self.gpr_seq_len)
                self.generate_instruction_reg_mul_imm(self.TMP_REG, _vi, ue_35bit_addr_shifter(cur_k_size * self.bytes_per_element))
                self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(self.LAYER0_FLASH_V_DRAM), self.TMP_REG)
                self.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=0x10000, element_size=cur_k_size, general_reg_src=self.TMP_REG)
                self.rms_norm_core(0x10000, 0x10000, cur_k_size)  # no gamma
                self.generate_instruction_reg_mul_imm(self.TMP_REG, _vi, ue_35bit_addr_shifter(kv_row_bytes))
                self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(v_cache_base), self.TMP_REG)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=0, element_size=cur_k_size, general_reg_src=self.TMP_REG)
                self.generate_instruction_add_inc(_vi)
                self.loop_end()
                self.release_isa_reg()  # _vi

            # Q norm always needed (Q is always computed fresh)
            # Q-norm: M = seq_len * group_size → use gpr_q_seq_len
            total_flops += self.rms_norm_core_dram(M=seq_len * self.group_size, N=cur_head_dim, A_DRAM_ADDR=self.LAYER0_Q_DRAM,
                            OUTPUT_DRAM_ADDR=self.LAYER0_Q_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_NORM_GAMMA + layer_off,
                                            gpr_M_reg=self.gpr_q_seq_len)
            _checkpoint(f"L{layer_idx}_qkv_vproj")

            ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE_GLOBAL if layer_idx in self._rope_global_layers else self.DRAM_ADDR_ROPE_LOCAL

            # §4.4 PBI-loop RoPE + GQA replication (replaces the per-token /
            # per-token×group Python unrolls — the dominant prefill-bin bloat).
            # cos/sin layout (per-token contiguous, sin half follows cos, stride
            # 2*rope_n*bpe) already matches rope_*_pbi. K rope result is built
            # directly in the compact KV cache; Q goes straight to
            # cur_head_dim-contiguous FLASH_Q. Sliding layers are full-rotary
            # (one rope call);
            # global layers are partial-rotary (gather→rope→scatter→copy).
            bpe = self.bytes_per_element
            head_bytes = cur_head_dim * bpe
            rope_bytes = rope_n * bpe
            sin_addr = ROPE_WEIGHT_ADDR + rope_n * bpe
            q_rows = seq_len * self.group_size
            kv_slot_off = self._kv_offset_for_layer[layer_idx]
            k_cache_base = self.LAYER0_K_ROPE_DRAM + kv_slot_off
            v_cache_base = self.LAYER0_V_DRAM + kv_slot_off
            tmp_in = self.LAYER0_MLP_GATE_DRAM    # gather/rope scratch (dead during attn,
            tmp_out = self.LAYER0_MLP_UP_DRAM     #  overwritten by the real MLP later)
            non_shared = layer_idx not in self._kv_shared_map
            if non_shared:
                # K norm (non-shared layers own their KV slot).
                total_flops += self.rms_norm_core_dram(M=seq_len, N=cur_head_dim, A_DRAM_ADDR=self.LAYER0_K_DRAM,
                                OUTPUT_DRAM_ADDR=self.LAYER0_K_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_NORM_GAMMA + layer_off,
                                gpr_M_reg=self.gpr_seq_len)
            if rope_n == cur_head_dim:
                # Full rotary: single PBI rope call (input/output contiguous).
                if non_shared:
                    total_flops += self.rope_hf_core_dram(M=seq_len, N=rope_n,
                        input_dram_addr=self.LAYER0_K_NORM_DRAM, output_dram_addr=k_cache_base,
                        cos_dram_addr=ROPE_WEIGHT_ADDR, sin_dram_addr=sin_addr, gpr_M_reg=self.gpr_seq_len)
                total_flops += self.rope_hf_core_dram_gqa(M=seq_len, group_size=self.group_size, N=rope_n,
                    input_dram_addr=self.LAYER0_Q_NORM_DRAM, output_dram_addr=self.LAYER0_FLASH_Q_DRAM,
                    cos_dram_addr=ROPE_WEIGHT_ADDR, sin_dram_addr=sin_addr, gpr_M_reg=self.gpr_seq_len)
            else:
                # Partial rotary: rotate first rope_n dims, copy the rest through.
                non_rot = cur_head_dim - rope_n
                if non_shared:
                    self._emit_strided_copy_pbi(self.LAYER0_K_NORM_DRAM, tmp_in, rope_n, head_bytes, rope_bytes, seq_len, self.gpr_seq_len)
                    total_flops += self.rope_hf_core_dram(M=seq_len, N=rope_n, input_dram_addr=tmp_in, output_dram_addr=tmp_out, cos_dram_addr=ROPE_WEIGHT_ADDR, sin_dram_addr=sin_addr, gpr_M_reg=self.gpr_seq_len)
                    self._emit_strided_copy_pbi(tmp_out, k_cache_base, rope_n, rope_bytes, head_bytes, seq_len, self.gpr_seq_len)
                    self._emit_strided_copy_pbi(self.LAYER0_K_NORM_DRAM + rope_bytes, k_cache_base + rope_bytes, non_rot, head_bytes, head_bytes, seq_len, self.gpr_seq_len)
                self._emit_strided_copy_pbi(self.LAYER0_Q_NORM_DRAM, tmp_in, rope_n, head_bytes, rope_bytes, q_rows, self.gpr_q_seq_len)
                total_flops += self.rope_hf_core_dram_gqa(M=seq_len, group_size=self.group_size, N=rope_n, input_dram_addr=tmp_in, output_dram_addr=tmp_out, cos_dram_addr=ROPE_WEIGHT_ADDR, sin_dram_addr=sin_addr, gpr_M_reg=self.gpr_seq_len)
                self._emit_strided_copy_pbi(tmp_out, self.LAYER0_FLASH_Q_DRAM, rope_n, rope_bytes, head_bytes, q_rows, self.gpr_q_seq_len)
                self._emit_strided_copy_pbi(self.LAYER0_Q_NORM_DRAM + rope_bytes, self.LAYER0_FLASH_Q_DRAM + rope_bytes, non_rot, head_bytes, head_bytes, q_rows, self.gpr_q_seq_len)
            _checkpoint(f"L{layer_idx}_rope")
            # GQA dup: read KV cache (k_size stride) → scatter group_size copies to FLASH (cur_head_dim stride).
            self._emit_gqa_duplicate_pbi(k_cache_base, self.LAYER0_FLASH_K_DRAM, cur_head_dim, seq_len, self.gpr_seq_len, src_row_bytes=head_bytes)
            self._emit_gqa_duplicate_pbi(v_cache_base, self.LAYER0_FLASH_V_DRAM, cur_head_dim, seq_len, self.gpr_seq_len, src_row_bytes=head_bytes)
            _checkpoint(f"L{layer_idx}_kv_gather")

            # Gemma4 uses scaling=1.0 (no 1/sqrt(d) in attention scores), so pass
            # q_scale=1.0 below; no Q pre-scale needed.

            # Pick the per-layer bias: full attention layers see the
            # entire causal window; sliding-attention layers are limited
            # to `sliding_window` tokens (run_prefill builds both biases).
            bias_addr_layer = (self.LAYER0_FLASH_BIAS_FULL_DRAM
                               if layer_idx in self._full_attention_layers
                               else self.LAYER0_FLASH_BIAS_SLIDING_DRAM)
            # unified_attention_core uses bias_mode="full_matrix" internally.
            # Prefill bias is [aligned_q, aligned_q], while the dynamic batch
            # GPR limits the live rows to q_seq_len.
            total_flops += self.unified_attention_core(
                batch=aligned_seq_len,
                aligned_seq_len=aligned_seq_len,
                head_dim=cur_head_dim,
                Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
                K_DRAM_ADDR=self.LAYER0_FLASH_K_DRAM,
                V_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                BIAS_DRAM_ADDR=bias_addr_layer,
                OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                IDENTITY_DRAM_ADDR=self.IDENTITY_DRAM_ADDR,
                gpr_batch_reg=self.gpr_q_seq_len,
                gpr_aligned_seq_len_reg=self.gpr_aligned_seq_len,
                q_scale=1.0,
            )
            _checkpoint(f"L{layer_idx}_attention")
            # O projection: INT4, K=cur_q_size
            total_flops += _projection_core(M=seq_len, K=cur_q_size, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                is_B_quantized=True,
                data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off,
                gpr_M_reg=self.gpr_seq_len,
                )
            total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                            OUTPUT_DRAM_ADDR=self.LAYER0_POST_ATTN_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_NORM_GAMMA + layer_off,
                                            gpr_M_reg=self.gpr_seq_len)
            self.eltwise_core_dram(
                M=seq_len, N=self.vector_length,
                dram_a=layer_input_addr, dram_b=self.LAYER0_POST_ATTN_NORM_DRAM,
                dram_out=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=self.gpr_seq_len)
            _checkpoint(f"L{layer_idx}_o_proj")
            total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                            OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off,
                                            gpr_M_reg=self.gpr_seq_len)
            # MLP gate (fused GELU) + up projections.
            total_flops += _projection_core(M=seq_len, K=self.vector_length, N=cur_mlp,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                is_B_quantized=True,
                data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off,
                gelu_enable=True,
                gpr_M_reg=self.gpr_seq_len,
                )
            total_flops += _projection_core(M=seq_len, K=self.vector_length, N=cur_mlp,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                is_B_quantized=True,
                data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off,
                gpr_M_reg=self.gpr_seq_len,
                )
            self.eltwise_core_dram(
                M=seq_len, N=cur_mlp,
                dram_a=self.LAYER0_MLP_GATE_DRAM, dram_b=self.LAYER0_MLP_UP_DRAM,
                dram_out=self.LAYER0_MLP_MULT_DRAM,
                mode=UE_MODE.ELTWISE_MUL, gpr_M_reg=self.gpr_seq_len)
            # MLP down projection.
            total_flops += _projection_core(M=seq_len, K=cur_mlp, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                is_B_quantized=True,
                data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off,
                gpr_M_reg=self.gpr_seq_len,
                )
            total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                            OUTPUT_DRAM_ADDR=self.LAYER0_POST_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_FFW_NORM_GAMMA + layer_off,
                                            gpr_M_reg=self.gpr_seq_len)
            self.eltwise_core_dram(
                M=seq_len, N=self.vector_length,
                dram_a=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, dram_b=self.LAYER0_POST_MLP_NORM_DRAM,
                dram_out=self.LAYER0_OUTPUT_DRAM,
                mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=self.gpr_seq_len)
            _checkpoint(f"L{layer_idx}_mlp")

            # Per-layer input injection (NEW for Gemma4 E2B) — seq_len-agnostic
            # (gpr_seq_len-driven) prefill variant; decode uses the seq_len=1 one.
            # Per-layer input injection. The prepared input is token-major, so
            # selecting one layer requires a strided row loop. RMSNorm,
            # residual addition and scaling use the standard DRAM cores.
            dim, N = self.per_layer_input_dim, self.vector_length
            total_flops += self.matmat_mul_core(
                M=seq_len, K=N, N=dim,
                A_DRAM_ADDR=self.LAYER0_OUTPUT_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PER_LAYER_GATE + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_PER_LAYER_GATE_OUTPUT_DRAM,
                gelu_enable=True, gpr_M_reg=self.gpr_seq_len)
            gate_dram = self.LAYER0_PER_LAYER_GATE_OUTPUT_DRAM
            sram_a, sram_b = 0x10000, 0x80000
            pli_reg = self.alloc_isa_reg()
            self.generate_instruction_add_set(pli_reg, ue_35bit_addr_shifter(self.PER_LAYER_INPUTS_DRAM + layer_idx * dim * self.bytes_per_element))
            row_reg = self.alloc_isa_reg()
            self.generate_instruction_add_set(row_reg, 0)
            self.loop_start(loop_cnt=seq_len, gpr_loop_cnt=self.gpr_seq_len)
            self.generate_instruction_reg_mul_imm(self.TMP_REG, row_reg, ue_35bit_addr_shifter(dim * self.bytes_per_element))
            self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(gate_dram), self.TMP_REG)
            self.accelerator_memory_to_sram(0, sram_a, dim, general_reg_src=self.TMP_REG)
            self.accelerator_memory_to_sram(0, sram_b, dim, general_reg_src=pli_reg)
            self.eltwise_mul_core(sram_a, sram_b, sram_a, dim)
            self.generate_instruction_reg_mul_imm(self.TMP_REG, row_reg, ue_35bit_addr_shifter(dim * self.bytes_per_element))
            self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(gate_dram), self.TMP_REG)
            self.sram_to_accelerator_memory(sram_a, 0, dim, general_reg_src=self.TMP_REG)
            self.generate_instruction_add_imm(pli_reg, ue_35bit_addr_shifter(self.LAYER_SIZE * dim * self.bytes_per_element), pli_reg)
            self.generate_instruction_add_inc(row_reg)
            self.loop_end()
            self.release_isa_reg()
            self.release_isa_reg()
            total_flops += seq_len * dim

            total_flops += self.matmat_mul_core(
                M=seq_len, K=dim, N=N,
                A_DRAM_ADDR=self.LAYER0_PER_LAYER_GATE_OUTPUT_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PER_LAYER_PROJ + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_PER_LAYER_PROJ_OUTPUT_DRAM,
                gpr_M_reg=self.gpr_seq_len)
            total_flops += self.rms_norm_core_dram(
                M=seq_len, N=N,
                A_DRAM_ADDR=self.LAYER0_PER_LAYER_PROJ_OUTPUT_DRAM,
                OUTPUT_DRAM_ADDR=self.LAYER0_PER_LAYER_PROJ_OUTPUT_DRAM,
                GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_PER_LAYER_NORM_GAMMA + layer_off,
                gpr_M_reg=self.gpr_seq_len)
            total_flops += self.eltwise_core_dram(
                M=seq_len, N=N,
                dram_a=self.LAYER0_OUTPUT_DRAM,
                dram_b=self.LAYER0_PER_LAYER_PROJ_OUTPUT_DRAM,
                dram_out=self.LAYER0_OUTPUT_DRAM,
                mode=UE_MODE.ELTWISE_ADD,
                gpr_M_reg=self.gpr_seq_len)
            total_flops += self.eltwise_core_dram(
                M=seq_len, N=N,
                dram_a=self.LAYER0_OUTPUT_DRAM,
                dram_b=None,
                dram_out=self.LAYER0_OUTPUT_DRAM,
                mode=UE_MODE.MUL_BROADCAST,
                scalar=self._layer_scalars[layer_idx],
                gpr_M_reg=self.gpr_seq_len)
            _checkpoint(f"L{layer_idx}_inject")

        self.generate_instruction_halt()
        self._prefill_checkpoints = checkpoints
        self._set_silent(False)
        return None, total_flops

    def _lookup_per_layer_embeddings(self, token_ids) -> torch.Tensor:
        """Return lightweight token-indexed rows as [token, layer, dim]."""
        tid_t = torch.tensor(token_ids, dtype=torch.long)
        if hasattr(self, '_mm_types') and self._mm_types is not None:
            mm_mask = torch.tensor(self._mm_types[:len(token_ids)])
            tid_t = tid_t.clone()
            tid_t[(mm_mask == 1) | (mm_mask == 3)] = 0
        return self.embed_tokens_per_layer_weight[tid_t].reshape(
            len(token_ids), self.LAYER_SIZE, self.per_layer_input_dim).contiguous()

    def run_prefill(self, prefill_program_addr: int, prefill_seq=None, flops: int = None,
                    profile_checkpoints: list | None = None):
        """
        Run prefill for the actual prompt — single entry, no bucket/padding.

        The prefill program at ``prefill_program_addr`` is shared by all prompt
        lengths up to ``max_prefill_seq_len``. This method restores clean FPGA state, uploads the
        prompt's embeddings / per-layer inputs / attention bias, primes the
        dynamic gpr registers, and launches the program.

        Args:
            prefill_program_addr: DRAM address of the compiled prefill program.
            prefill_seq: Full prompt token tuple (incl. the final token, which
                is NOT processed by prefill); if None, uses self.prefill_seq.
            flops: FLOP count for the HW rate counter.
            profile_checkpoints: when given (from a profile-compiled image), run
                the prefill segment-by-segment through its HALT checkpoints and
                return the per-segment [(name, ms)] HW latencies INSTEAD of a
                one-shot execute. The KV cache is still fully populated. Used by
                run_gemma4_profile.

        Returns:
            (latency, flop_rate) from the accelerator, or — when
            ``profile_checkpoints`` is given — the per-segment latency list.
        """
        if prefill_seq is None:
            prefill_seq = self.prefill_seq
        if prefill_seq is None:
            prefill_seq = tuple(self._cfg["default_prefill_tokens"])
        if len(prefill_seq) < 2:
            raise ValueError("Prefill sequence must have at least 2 tokens.")
        # Prefill processes all but the last token.
        prefill_seq = tuple(prefill_seq[:-1])
        seq_len = len(prefill_seq)
        assert seq_len <= self.max_prefill_seq_len, (
            f"Prefill length {seq_len} exceeds max_prefill_seq_len {self.max_prefill_seq_len}. "
            f"Bump 'max_prefill_seq_len' in gemma4_e2b_config.json (and raise _tensor_estimate "
            f"in __init__ accordingly) to support longer prompts."
        )
        # Runtime GPRs select the real row counts; run_decoder resumes from this
        # prompt position after prefill.
        self.seq_len = seq_len
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64

        # Restore clean FPGA state before this prefill (formerly in
        # run_prefill_bucketed): zero the entire K/V cache so decode's
        # bias-masked reads past seq_len see clean zeros, zero the attention
        # Q/K/V gather buffers, and re-upload the IDENTITY matrix (read by the
        # attention I @ V^T step). Idempotent for LM-only runs.
        from user_dma_core import UE_VECTOR_SIZE as _UE_VS
        num_slots = getattr(self, "_num_kv_slots", self.LAYER_SIZE)
        kv_cache_bytes = getattr(
            self, "_kv_cache_bytes",
            num_slots * self.MAX_CONTEXT_SIZE * self.head_dim * self.bytes_per_element)
        kv_zero_pad = torch.zeros(
            kv_cache_bytes // self.bytes_per_element, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, kv_zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_K_ROPE_DRAM, kv_zero_pad)
        _pre_align = ((self.max_prefill_seq_len * self.group_size + 63) // 64) * 64
        flash_qkv_elems = _pre_align * self.head_dim
        _flash_zero = torch.zeros(flash_qkv_elems, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_Q_DRAM, _flash_zero)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_K_DRAM, _flash_zero)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_V_DRAM, _flash_zero)
        self.dma_to_accelerator_memory(self.IDENTITY_DRAM_ADDR,
                                       torch.eye(_UE_VS, dtype=torch.bfloat16))
        print(f"[Prefill] LM-state restored ({num_slots} KV slots zeroed, IDENTITY re-uploaded)")

        # Time host-side prep (embedding lookup, per-layer inputs, bias build,
        # DMAs) — reported only in the segmented-profile path's status line.
        _host_prepare_w0 = time.perf_counter()

        print(f"[Prefill] [host] looking up token embeddings for {seq_len} tokens...", flush=True)
        embedding_tensor = self.get_embedding_for_tokens(prefill_seq)

        # Multimodal merge: replace image/audio placeholder embeddings with
        # encoder-produced soft-token features. Uses mm_token_type_ids where
        # 1=image, 3=audio (HF convention, see transformers/processing_utils.py
        # create_mm_token_type_ids).
        if hasattr(self, '_mm_types') and self._mm_types is not None:
            mm_types = torch.tensor(self._mm_types[:len(prefill_seq)])
            if hasattr(self, '_image_features') and self._image_features is not None:
                image_mask = (mm_types == 1)
                embedding_tensor[image_mask] = self._image_features[:image_mask.sum()].to(embedding_tensor.dtype)
                print(f"[Prefill] merged {image_mask.sum().item()} image features into embeddings")
            if hasattr(self, '_audio_features') and self._audio_features is not None:
                audio_mask = (mm_types == 3)
                embedding_tensor[audio_mask] = self._audio_features[:audio_mask.sum()].to(embedding_tensor.dtype)
                print(f"[Prefill] merged {audio_mask.sum().item()} audio features into embeddings")

        print(f"[Prefill] uploading embeddings to FPGA DRAM...", flush=True)
        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)

        # Host performs only the runtime token-indexed lookup. Projection,
        # normalization, add and scaling run on FPGA.
        print(f"[Prefill] [host] looking up per-layer embedding rows...", flush=True)
        per_layer_embed = self._lookup_per_layer_embeddings(prefill_seq)
        print(f"[Prefill] uploading per-layer embedding rows to FPGA DRAM...", flush=True)
        self.dma_to_accelerator_memory(
            self.PER_LAYER_EMBED_DRAM,
            per_layer_embed)

        # Clear multimodal state after the prefill per-layer lookup. Decode uses
        # _lookup_per_layer_embeddings with one token; retaining _mm_types would
        # incorrectly treat that token as multimodal and replace its ID with 0,
        # producing garbage per-layer injection and all-pad output. Mirror the
        # compare script's pattern: clear right after use.
        self._mm_types = None
        self._image_features = None
        self._audio_features = None

        # Build BOTH prefill bias matrices: full (causal) for full-attention
        # layers, and sliding (causal AND within `sliding_window` tokens) for
        # sliding-attention layers. compile_prefill picks per-layer.
        # Both biases are in q_seq_len space (each token has group_size query
        # heads, K is GQA-duplicated to match), so the window is converted
        # from token space to q-position space by multiplying by group_size.
        full_bias = torch.full((aligned_seq_len, aligned_seq_len), float("-inf"), dtype=torch.bfloat16)
        # Q rows are laid out token*group_size + head, and K is GQA-duplicated
        # to the same layout, so each attn head must attend ONLY its own head
        # slot (its KV head), causal in tokens. A flat q-space causal tril
        # (j<=i) additionally allows CROSS-HEAD attention (head h attending
        # earlier heads 0..h within a token). For E2B (num_kv=1) this is one KV
        # group so it doesn't corrupt the VALUE, but it still over-weights
        # earlier tokens (each contributes group_size duplicate slots vs the
        # current token's h+1), so the correct mask is same head AND
        # token-causal. (For E4B num_kv=2 the flat tril was catastrophic — it
        # let heads attend the WRONG KV group; see gemma4_e4b_test.py.)
        _gs = self.group_size
        _i = torch.arange(aligned_seq_len).unsqueeze(1)
        _j = torch.arange(aligned_seq_len).unsqueeze(0)
        _same_head = (_i % _gs) == (_j % _gs)
        _tok_causal = ((_j // _gs) <= (_i // _gs)) if not self.causal_mask_upper else ((_j // _gs) >= (_i // _gs))
        valid_mask = _same_head & _tok_causal
        full_bias.masked_fill_(valid_mask, 0.0)
        full_bias[:, q_seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_FULL_DRAM, full_bias)

        # Sliding bias is identical to full when seq_len ≤ sliding_window;
        # otherwise it additionally masks anything older than the window.
        if seq_len <= self.sliding_window:
            sliding_bias = full_bias
        else:
            window_q = self.sliding_window * self.group_size
            sliding_bias = torch.full((aligned_seq_len, aligned_seq_len), float("-inf"), dtype=torch.bfloat16)
            i_idx = torch.arange(aligned_seq_len).unsqueeze(1)
            j_idx = torch.arange(aligned_seq_len).unsqueeze(0)
            i_token = i_idx // self.group_size
            j_token = j_idx // self.group_size
            in_window = (i_token - j_token) < self.sliding_window
            sliding_mask = valid_mask & in_window
            sliding_bias.masked_fill_(sliding_mask, 0.0)
            sliding_bias[:, q_seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_SLIDING_DRAM, sliding_bias)

        host_prepare_s = time.perf_counter() - _host_prepare_w0

        # Profiling path: run the prefill through its per-phase HALT checkpoints
        # and return per-segment HW latencies. Populates the KV cache exactly
        # like a straight run (segments tile the whole program). The dynamic-PBI
        # preamble primes the same three GPRs as the one-shot dispatch below.
        if profile_checkpoints is not None:
            print(f"[Prefill] [profile] running {len(profile_checkpoints)} segments "
                  f"(host prep {host_prepare_s:.2f}s)...", flush=True)
            return self._profile_execute(
                [(self.gpr_seq_len,         seq_len),
                 (self.gpr_q_seq_len,       q_seq_len),
                 (self.gpr_aligned_seq_len, aligned_seq_len)],
                prefill_program_addr, profile_checkpoints, tail_name="tail_halt")

        print(f"[Prefill] [exec] launching prefill program on FPGA ({seq_len} tokens, {self.LAYER_SIZE} layers)...", flush=True)
        # Heartbeat thread: program_execute blocks until the FPGA halts, with no
        # intermediate visibility. Print elapsed seconds every 10s so the user
        # sees liveness during the ~30-60s prefill execution.
        import threading
        _pf_t0 = time.perf_counter()
        _pf_stop = threading.Event()
        def _pf_hb():
            while not _pf_stop.wait(10):
                print(f"[Prefill] [exec]   ... still running on FPGA ({time.perf_counter()-_pf_t0:.0f}s elapsed)", flush=True)
        _pf_th = threading.Thread(target=_pf_hb, daemon=True)
        _pf_th.start()
        try:
            # Dynamic-PBI dispatch: a single preamble primes seq_len / Q rows /
            # aligned attention length, then jumps into the cached prefill
            # program (gemma3 pattern). Building this at the fixed _preamble_addr
            # (past every program) is what keeps the gpr priming from clobbering
            # the prefill body.
            latency, flop_rate_program = self._dispatch_program(
                [(self.gpr_seq_len,         seq_len),
                 (self.gpr_q_seq_len,       q_seq_len),
                 (self.gpr_aligned_seq_len, aligned_seq_len)],
                prefill_program_addr, timeout=300.0, flops=flops)
        finally:
            _pf_stop.set()
            _pf_th.join(timeout=1.0)
        return latency, flop_rate_program

    def compile_decoder(self, layer_size: int = 35, profile: bool = False) -> tuple[None, list[int], list[int]]:
        """Compile a single decoder program with dynamic PBI.

        DYNAMIC PBI (see notes_gemma4_e2b.md): one captured
        program handles all decode positions. Per-token KV/RoPE addresses
        are computed at execute time via reg_mul_imm(gpr_seq_len, stride) +
        add_imm(base) → TMP_REG. Decoder attention calls unified_attention_core
        inline with batch=group_size and dynamic aligned KV length. End of
        program issues add_inc(gpr_seq_len) so subsequent decode steps advance
        automatically.

        Returns (None, [program_size_bytes], [total_flops]) — backward-compat
        single-element lists; caller uses [0] index.
        """
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]

        self._set_silent(True)
        self._loud(f"  Emitting dynamic-PBI decoder: 1 segment x {layer_size} layers, attention=unified-inline")
        seg_t0 = time.perf_counter()
        count_at_start = self.capture_count
        total_flops = 0

        # Optional per-phase profiling (see run_gemma4_profile / --profile).
        # _checkpoint emits a HALT at a phase boundary and records the resume
        # address (the next instruction). At runtime the profiler runs each
        # segment to its HALT, reads the HW latency counter, then resumes. Only
        # placed at UNCONDITIONAL points so every layer contributes one sample
        # per phase (never inside a loop_start/loop_end or a per-layer branch).
        checkpoints: list[list] = []
        def _checkpoint(name: str) -> None:
            if not profile:
                return
            self.generate_instruction_halt()
            resume = self.get_program_dram_addr() + self.capture_count * INSTRUCTION_SIZE_BYTES
            checkpoints.append([name, f"0x{resume:X}"])
        def _projection_core(**kwargs) -> int:
            """Uniform kernel selection for configurable decode IF4 projections."""
            if self.decode_kernel == "matmatmul":
                kwargs.setdefault("is_B_quantized", True)
                kwargs.setdefault("gpr_M_reg", gpr_one)
                return self.matmat_mul_core(**kwargs)
            # TODO: Resolve the decoder quantized_matmat_core dynamic-path
            # numerical bug, then stop removing the dimension GPRs here. All
            # configurable IF4 projection callsites (Q/K/V, O, MLP gate/up/down,
            # and LM head) are temporarily forced through the legacy path.
            kwargs.pop("gpr_M_reg", None)
            kwargs.pop("gpr_K_reg", None)
            kwargs.pop("gpr_N_reg", None)
            kwargs.pop("is_B_quantized", None)
            return self.quantized_matmat_core(**kwargs)
        # gpr_one holds the constant 1 — used as gpr_M_reg for all M=1 ops.
        gpr_one = self.alloc_isa_reg()
        self.generate_instruction_add_set(gpr_one, 1)
        gpr_group_size = self.alloc_isa_reg()
        self.generate_instruction_add_set(gpr_group_size, self.group_size)

        # Per-layer input preparation (decode): project this token to all 35 layer slices, then normalize, add its per-layer embedding, and scale.
        per_layer_dim = self.per_layer_input_dim
        # TODO: Fix the non-quantized BF16 dynamic matmul path, then use
        # matmat_mul_core with gpr_M_reg=gpr_one here.
        total_flops += self.matmat_mul_core_legacy(
            M=1, K=self.vector_length,
            N=self.LAYER_SIZE * per_layer_dim,
            A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
            B_DRAM_ADDR=self.DRAM_ADDR_PER_LAYER_MODEL_PROJ,
            OUTPUT_DRAM_ADDR=self.PER_LAYER_MODEL_PROJ_OUTPUT_DRAM)
        per_layer_rows_reg = self.alloc_isa_reg()
        self.generate_instruction_reg_mul_imm(
            per_layer_rows_reg, gpr_one, self.LAYER_SIZE)
        total_flops += self.rms_norm_core_dram(
            M=self.LAYER_SIZE, N=per_layer_dim,
            A_DRAM_ADDR=self.PER_LAYER_MODEL_PROJ_OUTPUT_DRAM,
            OUTPUT_DRAM_ADDR=self.PER_LAYER_INPUTS_DRAM,
            GAMMA_DRAM_ADDR=self.DRAM_ADDR_PER_LAYER_PROJ_NORM,
            gpr_M_reg=per_layer_rows_reg)
        total_flops += self.eltwise_core_dram(
            M=self.LAYER_SIZE, N=per_layer_dim,
            dram_a=self.PER_LAYER_INPUTS_DRAM,
            dram_b=self.PER_LAYER_EMBED_DRAM,
            dram_out=self.PER_LAYER_INPUTS_DRAM,
            mode=UE_MODE.ELTWISE_ADD,
            gpr_M_reg=per_layer_rows_reg)
        total_flops += self.eltwise_core_dram(
            M=self.LAYER_SIZE, N=per_layer_dim,
            dram_a=self.PER_LAYER_INPUTS_DRAM,
            dram_b=None,
            dram_out=self.PER_LAYER_INPUTS_DRAM,
            mode=UE_MODE.MUL_BROADCAST,
            scalar=self._per_layer_input_scale,
            gpr_M_reg=per_layer_rows_reg)
        self.release_isa_reg()
        _checkpoint("per_layer_prepare")

        # Iterate once (no bucket loop)
        for _bi_unused in [0]:
            seq_len = self.MAX_CONTEXT_SIZE  # template only — for FLOPs
            for layer_idx in range(layer_size):
                layer_off = layer_idx * LAYER_WEIGHT_SIZE
                cur_head_dim, cur_q_size, cur_k_size = self._get_layer_attention_dims(layer_idx)
                cur_mlp = self._get_mlp_elements(layer_idx)
                rope_n = self._get_rope_dims(layer_idx)

                # Layer-input source:
                #   layer 0: LAYER0_INPUT_DRAM (uploaded by run_decoder each step)
                #   layer i>0: LAYER0_OUTPUT_DRAM (written by the previous layer's
                #     per_layer_injection). No copy needed — LAYER0_OUTPUT_DRAM is
                #     only overwritten at the end of the current layer (in the
                #     MLP residual add at line ~2955), which happens AFTER we
                #     consume it as the attention-residual source. So reading it
                #     here and for the attention residual below is safe.
                layer_input_addr = self.LAYER0_INPUT_DRAM if layer_idx == 0 else self.LAYER0_OUTPUT_DRAM
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=layer_input_addr,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off,
                              gpr_M_reg=gpr_one)
                # Q/K/V projections: use per-layer dims
                total_flops += _projection_core(M=1, K=self.vector_length, N=cur_q_size,
                                                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                                                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off,
                                                    OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                                                    data_type=TYPE.IF4,
                                                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off,
                                                    )
                if layer_idx in self._kv_shared_map:
                    ref_layer = self._kv_shared_map[layer_idx]
                    kv_layer_for_attn = ref_layer  # read from reference layer's KV cache
                else:
                    kv_layer_for_attn = layer_idx  # read from own KV cache
                    # K projection
                    total_flops += _projection_core(M=1, K=self.vector_length, N=cur_k_size,
                        A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                        B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off,
                        OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                        data_type=TYPE.IF4,
                        SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off,
                        )
                    # V projection
                    total_flops += _projection_core(M=1, K=self.vector_length, N=cur_k_size,
                        A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                        B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off,
                        OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                        data_type=TYPE.IF4,
                        SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off,
                        )
                    self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_FLASH_V_DRAM, sram_address=0x10000, element_size=cur_k_size)
                    # V norm (Gemma4: normalize V without learnable scale)
                    self.rms_norm_core(0x10000, 0x10000, cur_k_size)  # no gamma
                    # V scatter to V cache at decode_pos via reg_mul_imm + add_imm.
                    # Compact cache row stride equals this layer's head dimension.
                    _kv_row_bytes = self._kv_row_bytes_for_layer[layer_idx]
                    _v_slot_base = self.LAYER0_V_DRAM + self._kv_offset_for_layer[layer_idx]
                    self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gpr_seq_len, ue_35bit_addr_shifter(_kv_row_bytes))
                    self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(_v_slot_base), self.TMP_REG)
                    self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=0, element_size=cur_k_size, general_reg_src=self.TMP_REG)
                    # RMS norm on K
                    total_flops += self.rms_norm_core_dram(M=1, N=cur_head_dim, A_DRAM_ADDR=self.LAYER0_K_DRAM,
                                  OUTPUT_DRAM_ADDR=self.LAYER0_K_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_NORM_GAMMA + layer_off,
                                  gpr_M_reg=gpr_one)

                _checkpoint(f"L{layer_idx}_qkv_vproj")

                # Q norm: M = group_size (compile-time constant). Use legacy
                # static-M path (no gpr_M_reg) since group_size doesn't vary.
                total_flops += self.rms_norm_core_dram(M=self.group_size, N=cur_head_dim, A_DRAM_ADDR=self.LAYER0_Q_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_Q_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_NORM_GAMMA + layer_off)

                ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE_GLOBAL if layer_idx in self._rope_global_layers else self.DRAM_ADDR_ROPE_LOCAL
                rope_row = 2 * rope_n * self.bytes_per_element  # full cos+sin pair stride per token (rope_n*2 values * 2 bytes)

                kv_slot_off_local = self._kv_offset_for_layer[layer_idx]
                k_rope_base = self.LAYER0_K_ROPE_DRAM + kv_slot_off_local

                if layer_idx not in self._kv_shared_map:
                    # K-RoPE at decode_pos: cos/sin = ROPE_WEIGHT_ADDR + gpr_seq_len * rope_row.
                    # IMPORTANT: write RoPE output to LAYER0_K_DRAM (scratch buffer), NOT to
                    # k_rope_base (= cache position 0). Writing to position 0 would corrupt
                    # the first prefill token's K every decode step.
                    self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gpr_seq_len, ue_35bit_addr_shifter(rope_row))
                    self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(ROPE_WEIGHT_ADDR), self.TMP_REG)
                    total_flops += self.rope_hf_core_decode(
                        N=rope_n,
                        input_dram_addr=self.LAYER0_K_NORM_DRAM,
                        output_dram_addr=self.LAYER0_K_DRAM,         # scratch, not cache
                        gr_weight_dram=self.TMP_REG)
                    # Copy rotated dims from scratch into K cache at decode_pos.
                    self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gpr_seq_len, ue_35bit_addr_shifter(self._kv_row_bytes_for_layer[layer_idx]))
                    self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(k_rope_base), self.TMP_REG)
                    self.accelerator_memcpy(self.LAYER0_K_DRAM, 0, rope_n * self.bytes_per_element, gr_dst_addr=self.TMP_REG)

                # Q-RoPE: same cos/sin address for all group_size heads (same decode_pos)
                self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gpr_seq_len, ue_35bit_addr_shifter(rope_row))
                self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(ROPE_WEIGHT_ADDR), self.TMP_REG)
                for g in range(self.group_size):
                    total_flops += self.rope_hf_core_decode(
                        N=rope_n,
                        input_dram_addr=self.LAYER0_Q_NORM_DRAM + g * cur_head_dim * self.bytes_per_element,
                        output_dram_addr=self.LAYER0_FLASH_Q_DRAM + g * cur_head_dim * self.bytes_per_element,
                        gr_weight_dram=self.TMP_REG)

                # Partial-rotary non-rotated dims (full-attention layers only).
                if layer_idx in self._full_attention_layers and rope_n < cur_head_dim:
                    remaining = cur_head_dim - rope_n
                    # Q non-rotated dims: static addresses (per-group).
                    for g in range(self.group_size):
                        src = self.LAYER0_Q_NORM_DRAM + g * cur_head_dim * self.bytes_per_element + rope_n * self.bytes_per_element
                        dst = self.LAYER0_FLASH_Q_DRAM + g * cur_head_dim * self.bytes_per_element + rope_n * self.bytes_per_element
                        self.accelerator_memory_to_sram(src, 0x10000, remaining)
                        self.sram_to_accelerator_memory(0x10000, dst, remaining)
                    if layer_idx not in self._kv_shared_map:
                        # K non-rotated dims at decode_pos: cache_addr = k_rope_base + gpr_seq_len * k_size + rope_n_bytes
                        src = self.LAYER0_K_NORM_DRAM + rope_n * self.bytes_per_element
                        k_cache_nrot_base = k_rope_base + rope_n * self.bytes_per_element
                        self.accelerator_memory_to_sram(src, 0x10000, remaining)
                        self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gpr_seq_len, ue_35bit_addr_shifter(self._kv_row_bytes_for_layer[layer_idx]))
                        self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(k_cache_nrot_base), self.TMP_REG)
                        self.sram_to_accelerator_memory(0x10000, 0, remaining, general_reg_src=self.TMP_REG)

                # Gemma4 uses scaling=1.0; q_scale=1.0 on the attention call means no
                # Q pre-scale is needed here.

                _checkpoint(f"L{layer_idx}_rope")

                # K/V cache reads — KV-shared layers point at the source layer's cache.
                kv_slot_off_read = self._kv_offset_for_layer[kv_layer_for_attn]
                kv_k_base = self.LAYER0_K_ROPE_DRAM + kv_slot_off_read
                kv_v_base = self.LAYER0_V_DRAM + kv_slot_off_read

                # Compact per-slot rows already satisfy unified attention's
                # [aligned_seq_len, head_dim] contract. Read the cache directly;
                # zero-initialized future rows cover alignment padding.
                _checkpoint(f"L{layer_idx}_kv_gather")

                # unified_attention_core uses bias_mode="full_matrix"; run_decoder
                # uploads group_size identical bias rows for this call.
                bias_addr_layer = (self.LAYER0_FLASH_BIAS_FULL_DRAM
                                   if layer_idx in self._full_attention_layers
                                   else self.LAYER0_FLASH_BIAS_SLIDING_DRAM)
                total_flops += self.unified_attention_core(
                    batch=self.group_size,
                    aligned_seq_len=seq_len,
                    head_dim=cur_head_dim,
                    Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
                    K_DRAM_ADDR=kv_k_base,
                    V_DRAM_ADDR=kv_v_base,
                    BIAS_DRAM_ADDR=bias_addr_layer,
                    OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                    SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                    IDENTITY_DRAM_ADDR=self.IDENTITY_DRAM_ADDR,
                    gpr_batch_reg=gpr_group_size,
                    gpr_aligned_seq_len_reg=self.gpr_aligned_seq_len,
                    q_scale=1.0,
                )
                _checkpoint(f"L{layer_idx}_attention")

                # O projection: INT4, K=cur_q_size (actual per-layer attention output dim)
                total_flops += _projection_core(M=1, K=cur_q_size, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off,
                    )
                # Decode-only fused attention residual and FFN pre-normalization.
                _vec_a = 0x10000
                _vec_b = 0x90000
                self.accelerator_memory_to_sram(
                    self.LAYER0_ATTN_PROJ_OUTPUT_DRAM, _vec_a, self.vector_length)
                self.accelerator_memory_to_sram(
                    self.DRAM_ADDR_LAYER0_POST_NORM_GAMMA + layer_off,
                    _vec_b, self.vector_length)
                self.rms_norm_core(_vec_a, _vec_a, self.vector_length, _vec_b)
                total_flops += 4 * self.vector_length

                # Attention residual: use layer_input_addr (LAYER0_OUTPUT_DRAM
                # for layers > 0, LAYER0_INPUT_DRAM for layer 0) — same source
                # as the pre-norm above. This avoids the LAYER0_OUTPUT → LAYER0_INPUT
                # copy that used to run at the top of every layer.
                self.accelerator_memory_to_sram(
                    layer_input_addr, _vec_b, self.vector_length)
                self.eltwise_add_core(_vec_a, _vec_b, _vec_a, self.vector_length)
                self.sram_to_accelerator_memory(
                    _vec_a, self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                    self.vector_length)
                self.accelerator_memory_to_sram(
                    self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off,
                    _vec_b, self.vector_length)
                self.rms_norm_core(_vec_a, _vec_a, self.vector_length, _vec_b)
                total_flops += 4 * self.vector_length
                self.sram_to_accelerator_memory(
                    _vec_a, self.LAYER0_PRE_MLP_NORM_DRAM,
                    self.vector_length)

                _checkpoint(f"L{layer_idx}_o_proj")

                total_flops += _projection_core(M=1, K=self.vector_length, N=cur_mlp,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off,
                    gelu_enable=True,
                    )
                total_flops += _projection_core(M=1, K=self.vector_length, N=cur_mlp,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off,
                    )

                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_GATE_DRAM, sram_address=0x10000, element_size=cur_mlp)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_UP_DRAM, sram_address=0x90000, element_size=cur_mlp)
                self.eltwise_mul_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=cur_mlp)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_MLP_MULT_DRAM, element_size=cur_mlp)

                total_flops += _projection_core(M=1, K=cur_mlp, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off,
                    gpr_M_reg=gpr_one,
                    )
                # Decode-only fused MLP post-normalization + residual. Keep the
                # normalized MLP-down vector in SRAM through the residual add,
                # removing the post-MLP-norm DRAM write/read pair.
                self.accelerator_memory_to_sram(
                    self.LAYER0_MLP_DOWN_DRAM, _vec_a, self.vector_length)
                self.accelerator_memory_to_sram(
                    self.DRAM_ADDR_LAYER0_POST_FFW_NORM_GAMMA + layer_off,
                    _vec_b, self.vector_length)
                self.rms_norm_core(_vec_a, _vec_a, self.vector_length, _vec_b)
                total_flops += 4 * self.vector_length
                self.accelerator_memory_to_sram(
                    self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                    _vec_b, self.vector_length)
                self.eltwise_add_core(_vec_a, _vec_b, _vec_a, self.vector_length)
                self.sram_to_accelerator_memory(
                    _vec_a, self.LAYER0_OUTPUT_DRAM, self.vector_length)

                _checkpoint(f"L{layer_idx}_mlp")

                # Per-layer input injection (NEW for Gemma4 E2B) - decoder uses seq_len=1
                total_flops += self.matmat_mul_core(
                    M=1, K=self.vector_length, N=self.per_layer_input_dim,
                    A_DRAM_ADDR=self.LAYER0_OUTPUT_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PER_LAYER_GATE + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_PER_LAYER_GATE_OUTPUT_DRAM,
                    gelu_enable=True, gpr_M_reg=gpr_one)
                per_layer_input_addr = (
                    self.PER_LAYER_INPUTS_DRAM
                    + layer_idx * self.per_layer_input_dim * self.bytes_per_element)
                total_flops += self.eltwise_core_dram(
                    M=1, N=self.per_layer_input_dim,
                    dram_a=self.LAYER0_PER_LAYER_GATE_OUTPUT_DRAM,
                    dram_b=per_layer_input_addr,
                    dram_out=self.LAYER0_PER_LAYER_GATE_OUTPUT_DRAM,
                    mode=UE_MODE.ELTWISE_MUL, gpr_M_reg=gpr_one)
                total_flops += self.matmat_mul_core(
                    M=1, K=self.per_layer_input_dim, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_PER_LAYER_GATE_OUTPUT_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PER_LAYER_PROJ + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_PER_LAYER_PROJ_OUTPUT_DRAM,
                    gpr_M_reg=gpr_one)
                total_flops += self.rms_norm_core_dram(
                    M=1, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_PER_LAYER_PROJ_OUTPUT_DRAM,
                    OUTPUT_DRAM_ADDR=self.LAYER0_PER_LAYER_PROJ_OUTPUT_DRAM,
                    GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_PER_LAYER_NORM_GAMMA + layer_off,
                    gpr_M_reg=gpr_one)
                total_flops += self.eltwise_core_dram(
                    M=1, N=self.vector_length,
                    dram_a=self.LAYER0_OUTPUT_DRAM,
                    dram_b=self.LAYER0_PER_LAYER_PROJ_OUTPUT_DRAM,
                    dram_out=self.LAYER0_OUTPUT_DRAM,
                    mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=gpr_one)
                total_flops += self.eltwise_core_dram(
                    M=1, N=self.vector_length,
                    dram_a=self.LAYER0_OUTPUT_DRAM,
                    dram_b=None,
                    dram_out=self.LAYER0_OUTPUT_DRAM,
                    mode=UE_MODE.MUL_BROADCAST,
                    scalar=self._layer_scalars[layer_idx],
                    gpr_M_reg=gpr_one)

                _checkpoint(f"L{layer_idx}_inject")

            if layer_size == self.LAYER_SIZE:
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_OUTPUT_DRAM,
                    OUTPUT_DRAM_ADDR=self.OUTPUT_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_OUTPUT_NORM_GAMMA,
                    gpr_M_reg=gpr_one)
                total_flops += _projection_core(M=1, K=self.vector_length, N=self.EMBEDDING_ELEMENTS,
                    A_DRAM_ADDR=self.OUTPUT_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD_QUANT,
                    OUTPUT_DRAM_ADDR=self.LOGITS_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD_SCALE,
                    )
                _checkpoint("lm_head")

            # Advance decode_pos for next token. The host's preamble only
            # sets gpr_seq_len once at the very first decode step; each
            # subsequent step the program self-increments.
            self.generate_instruction_add_inc(self.gpr_seq_len)
            self.generate_instruction_halt()
            self.release_isa_reg()  # gpr_group_size
            self.release_isa_reg()  # gpr_one
            instr_count = self.capture_count - count_at_start
            self._loud(f"    decoder segment ({instr_count} instr) done in {time.perf_counter()-seg_t0:.1f}s")
        program_sizes = [instr_count * 32]
        total_flops_list = [total_flops]
        self._decoder_checkpoints = checkpoints
        self._set_silent(False)
        return None, program_sizes, total_flops_list

    def _dispatch_program(self, gpr_sets: list[tuple[int, int]],
                          target_addr: int | None,
                          timeout: float = 50.0, flops: float | None = None):
        """Build and execute a one-shot dispatch preamble at the fixed scratch
        slot ``self._preamble_addr`` (which sits past every cached program).

        The preamble sets each (reg, value) in ``gpr_sets`` via add_set, then:
          - if ``target_addr`` is given, jump_abs into that cached program
            (which ends in its own HALT); or
          - if ``target_addr`` is None, HALT immediately (used to prime a gpr
            with no program to run).

        This is the gemma3 dispatch idiom: the preamble is always rewritten to
        the SAME address and never advances the program cursor, so it can never
        clobber the prefill or decoder programs. Returns program_execute's
        (latency, flop_rate).
        """
        self.clear_inst_id()
        self.start_capture()
        for reg, val in gpr_sets:
            self.generate_instruction_add_set(reg, val)
        if target_addr is not None:
            self.generate_instruction_jump_abs(ue_35bit_addr_shifter(target_addr))
        else:
            self.generate_instruction_halt()
        self.stop_capture()
        self.write_captured_instructions_to_dram(self._preamble_addr)
        self.clear_capture_buffer()
        return self.program_execute(self._preamble_addr, timeout=timeout, flops=flops)

    def run_decoder(self, decoder_program_sizes: list[int], decoder_base_addr: int, token_id: int, flops_per_token: list[int] | None = None) -> dict:
        """Run decode loop with dynamic PBI.

        Single decoder program — same address every token. gpr_seq_len is
        primed ONCE before the first decode step to the current decode_pos
        (= prompt length); the captured program's trailing add_inc(gpr_seq_len)
        advances it automatically for subsequent tokens. gpr_aligned_seq_len
        is re-set each step because K context length grows by 1 token and may
        cross a UE_VECTOR_SIZE boundary.
        """
        if token_id is None:
            print("No last token available for decode.")
            return {}

        max_seq_len = self.MAX_CONTEXT_SIZE
   # benchmark cap (e.g. 128); default off
        total_latency, total_flop_rate = 0, 0
        # Single program (dynamic PBI). Ignore decoder_program_sizes length.
        prog_addr = decoder_base_addr
        flops_per_token_scalar = flops_per_token[0] if flops_per_token else None

        # Pure greedy decode. GEMMA4_PENALTY=1 is rejected in __init__ until
        # dynamic streaming quantized_matmat_core supports broadcast bias.

        # Prime gpr_seq_len to the current decode_pos (= prompt length, since
        # self.seq_len reflects the prompt at this point). Subsequent steps
        # rely on the program's add_inc to advance it. A HALT-terminated
        # preamble (no program to run) just latches the register on the HW.
        self._dispatch_program([(self.gpr_seq_len, self.seq_len)], None, timeout=10.0)
        print("\n------------------------------ DECODE START ------------------------------\n", flush=True)

        # Live decode status bar (mirrors llama3.2_1b / gemma4_e4b): pin the bottom
        # terminal row via an ANSI scroll region; generated tokens stream above it
        # while a tokens/s counter refreshes in place. All output is on stdout
        # (tokens scroll inside rows 1..rows-1; the status writes row `rows` with
        # cursor save/restore), so nothing clobbers the streamed text. TTY-only
        # (skipped when piped/redirected).
        import shutil
        _dec_start_seq = self.seq_len
        _dec_timer = time.perf_counter()
        _first_tok_dt = None   # wall-clock of the 1st decoded token → peak tok/s
        _decoded_n = 0         # number of decode steps (for average tok/s)
        _use_status = sys.stdout.isatty()
        def _status_setup():
            rows = shutil.get_terminal_size().lines
            sys.stdout.write(f"\033[1;{rows - 1}r")   # scroll region = rows 1..rows-1
            sys.stdout.write(f"\033[{rows - 1};1H")   # park cursor at bottom of region
            sys.stdout.flush()
        def _status_update():
            rows = shutil.get_terminal_size().lines
            n = self.seq_len - _dec_start_seq
            elapsed = time.perf_counter() - _dec_timer
            rate = n / elapsed if elapsed > 0 else 0.0
            sys.stdout.write("\0337")                  # save cursor
            sys.stdout.write(f"\033[{rows};1H\033[2K") # bottom row, clear it
            sys.stdout.write(f" decoding… {n} tokens  (pos {self.seq_len}/{self.MAX_CONTEXT_SIZE})  "
                             f"{elapsed:.1f}s  {rate:.1f} tok/s")
            sys.stdout.write("\0338")                  # restore cursor
            sys.stdout.flush()
        def _status_teardown():
            rows = shutil.get_terminal_size().lines
            sys.stdout.write("\033[r")                 # reset scroll region
            sys.stdout.write(f"\033[{rows};1H\033[2K") # clear the status row
            sys.stdout.flush()
        if _use_status:
            _status_setup()

        while self.seq_len < max_seq_len:
            self._set_silent(True)
            _tok_t0 = time.perf_counter()               # per-token wall-clock start
            self.seq_len += 1
            decode_pos = self.seq_len - 1               # 0-based pos of token now being computed
            aligned_seq_len = ((self.seq_len + 63) // 64) * 64

            embedding_tensor = self.get_embedding_for_tokens([token_id])
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
            per_layer_embed = self._lookup_per_layer_embeddings([token_id])
            self.dma_to_accelerator_memory(
                self.PER_LAYER_EMBED_DRAM,
                per_layer_embed)

            # Build BOTH decode bias matrices. unified_attention_core uses
            # bias_mode="full_matrix", and decoder batch is group_size Q heads.
            full_bias_row = torch.full((self.group_size, aligned_seq_len), -1e36, dtype=torch.bfloat16)
            full_bias_row[:, :self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_FULL_DRAM, full_bias_row)
            if self.seq_len <= self.sliding_window:
                sliding_bias_row = full_bias_row
            else:
                sliding_bias_row = torch.full((self.group_size, aligned_seq_len), -1e36, dtype=torch.bfloat16)
                window_start = self.seq_len - self.sliding_window
                sliding_bias_row[:, window_start:self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_SLIDING_DRAM, sliding_bias_row)

            # Dynamic-PBI dispatch: re-set the attention length (K context grows
            # each step, may cross a 64-align boundary), then jump into the
            # cached decoder program. gpr_seq_len was primed once above and is
            # advanced by the decoder's trailing add_inc.
            latency, flop_rate_program = self._dispatch_program(
                [(self.gpr_aligned_seq_len, aligned_seq_len)],
                prog_addr, timeout=300.0, flops=flops_per_token_scalar)
            total_latency += latency
            total_flop_rate += flop_rate_program
            # HW argmax of the streaming LM-head logits.
            token_id = self.get_arg_max_index()
            token_char = self.tokenizer.decode([token_id])
            self._set_silent(False)

            _tok_dt = time.perf_counter() - _tok_t0
            if _first_tok_dt is None:
                _first_tok_dt = _tok_dt                  # 1st token (shortest context) → peak
            _decoded_n += 1

            if token_id in [1, self._end_of_turn_token_id]:
                if _use_status:
                    _status_teardown()
                print(f"\nStop token {token_id} reached.")
                break
            print(token_char, end="", flush=True)
            if _use_status:
                _status_update()
        else:
            if _use_status:
                _status_teardown()
        # Decode-speed report (matches the Qwen comparison table format).
        _elapsed = time.perf_counter() - _dec_timer
        _peak = (1.0 / _first_tok_dt) if _first_tok_dt else 0.0
        _avg = (_decoded_n / _elapsed) if _elapsed > 0 else 0.0
        print(f"\nDecode speed: peak (1st token) {_peak:.2f} tok/s, "
              f"average {_avg:.2f} tok/s  ({_decoded_n} tokens in {_elapsed:.2f}s)")
        return self.seq_len, total_latency, total_flop_rate
