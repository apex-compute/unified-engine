#!/usr/bin/env python3
"""SmolVLM2-500M standalone inference from pre-compiled bin files.

Self-contained and OFFLINE: it loads the deployable artifacts under ``smolvlm2_bin/`` (params.bin,
programs.bin, and the shipped tokenizer dir) and runs prefill + decode with no compilation and no
network access. It does not import the build script and never downloads anything, so it can be shipped
to a customer on its own. Generate the bins once with smolvlm2_test.py (the build/offline-prep step)."""
import builtins
import hashlib
import json
import os
import sys
import time
import threading

# Hard offline: guarantee transformers / huggingface_hub never reach the network (set before import).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

# Suppress prints during decode (same as smolvlm2_test.py)
_original_print = builtins.print
_SILENT_MODE = False
def _quiet_print(*args, **kwargs):
    if not _SILENT_MODE:
        _original_print(*args, **kwargs)
builtins.print = _quiet_print

import torch
from transformers import AutoTokenizer

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, DMA_DEVICE_C2H, TYPE, UE_VECTOR_SIZE,
    DRAM_INSTRUCTION_ADDR,
    UnifiedEngine, set_dma_device, ue_35bit_addr_shifter,
)

# =============================================================================
# Config loading
# =============================================================================
def _load_smolvlm2_config(path: str = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smolvlm2_config.json")
    with open(path) as f:
        return json.load(f)

_SMOLVLM2_CFG = _load_smolvlm2_config()


class SmolVLM2RuntimeAttentionStateMixin:
    """Runtime helper mixin for volatile LM attention state."""

    def _zero_runtime_attention_state(self, seq_len: int | None = None, bucket_len: int | None = None,
                                      preserve_kv_prefix: bool = False) -> None:
        """Clear volatile tensor state used by LM attention.

        Runtime correctness must not depend on tensor_init() zeroing.
        This helper zeroes the decode attention workspace and the KV tail.
        """
        if seq_len is None:
            seq_len = getattr(self, "seq_len", None)
        if bucket_len is None:
            bucket_len = seq_len
        if seq_len is None:
            raise ValueError("seq_len is required")
        max_context = int(getattr(self, "max_seq_len", 0) or 0)
        if seq_len < 0 or seq_len > max_context:
            raise ValueError(f"seq_len={seq_len} is outside [0, {max_context}]")
        if bucket_len < seq_len:
            raise ValueError(f"bucket_len={bucket_len} must be >= seq_len={seq_len}")
        if bucket_len > max_context:
            bucket_len = max_context
        bpe = int(getattr(self, "bytes_per_element", 2) or 2)
        head_dim = int(getattr(self, "HEAD_DIM", 0) or 0)
        num_layers = int(getattr(self, "NUM_LAYERS", 0) or 0)
        num_kv_heads = int(getattr(self, "NUM_KV_HEADS", 0) or 0)
        kv_head_stride = int(getattr(self, "KV_HEAD_STRIDE", max_context * head_dim * bpe) or (max_context * head_dim * bpe))
        kv_layer_stride = int(getattr(self, "KV_LAYER_STRIDE", num_kv_heads * kv_head_stride) or (num_kv_heads * kv_head_stride))
        self._last_runtime_zero_tail_start = seq_len
        self._runtime_attention_zero_calls_total = getattr(self, "_runtime_attention_zero_calls_total", 0) + 1
        if getattr(self, "seq_len", None) == seq_len:
            self._runtime_attention_zero_decode_calls = getattr(self, "_runtime_attention_zero_decode_calls", 0) + 1
        else:
            self._runtime_attention_zero_decode_calls = 0

        if max_context > seq_len:
            zero_tail = torch.zeros((max_context - seq_len) * head_dim, dtype=torch.bfloat16)
            for layer_idx in range(num_layers):
                layer_k_base = self.LAYER0_K_DRAM + layer_idx * kv_layer_stride
                layer_v_base = self.LAYER0_V_DRAM + layer_idx * kv_layer_stride
                for head_idx in range(num_kv_heads):
                    self.dma_to_accelerator_memory(
                        layer_k_base + head_idx * kv_head_stride + seq_len * head_dim * bpe,
                        zero_tail,
                    )
                    self.dma_to_accelerator_memory(
                        layer_v_base + head_idx * kv_head_stride + seq_len * head_dim * bpe,
                        zero_tail,
                    )

        if hasattr(self, "FLASH_Q_DRAM") and hasattr(self, "FLASH_K_DRAM") and hasattr(self, "FLASH_V_DRAM"):
            flash_len = int(getattr(self, "PREFILL_QMAX", bucket_len) or bucket_len) * head_dim
            flash_q = torch.zeros((flash_len,), dtype=torch.bfloat16)
            flash_kv = torch.zeros((flash_len,), dtype=torch.bfloat16)
            self.dma_to_accelerator_memory(self.FLASH_Q_DRAM, flash_q)
            self.dma_to_accelerator_memory(self.FLASH_K_DRAM, flash_kv)
            self.dma_to_accelerator_memory(self.FLASH_V_DRAM, flash_kv)
            if hasattr(self, "FLASH_OUT_DRAM"):
                self.dma_to_accelerator_memory(self.FLASH_OUT_DRAM, torch.zeros_like(flash_q))
        if hasattr(self, "FLASH_BIAS_DRAM"):
            bias_len = int(getattr(self, "PREFILL_QMAX", bucket_len) or bucket_len)
            bias = torch.full((bias_len, bias_len), -1e38, dtype=torch.bfloat16)
            self.dma_to_accelerator_memory(self.FLASH_BIAS_DRAM, bias)
        if hasattr(self, "DECODE_BIAS_DRAM"):
            decode_bias = torch.full((int(getattr(self, "GROUP_SIZE", 1) or 1), max_context), -1e38, dtype=torch.bfloat16)
            decode_bias[:, :seq_len] = 0.0
            self.dma_to_accelerator_memory(self.DECODE_BIAS_DRAM, decode_bias)

# =============================================================================
# Helper functions
# =============================================================================
def init_hang_prevention(ue) -> None:
    """Stop stale execution and write HALT to instruction DRAM base."""
    print("[Init] Hang prevention: disabling instruction execution...")
    ue.start_capture()
    ue.generate_instruction_halt()
    ue.stop_capture()
    halt_bytes = bytearray()
    for inst in ue.capture_buffer:
        halt_bytes.extend(inst.get_bytes())
    ue.dma_write(DMA_DEVICE_H2C, DRAM_INSTRUCTION_ADDR, halt_bytes, len(halt_bytes))
    ue.clear_capture_buffer()
    print("[Init] HALT written to instruction DRAM base")



# =============================================================================
# SmolVLM2_UnifiedEngine class
# =============================================================================
class SmolVLM2_UnifiedEngine(SmolVLM2RuntimeAttentionStateMixin, UnifiedEngine):
    """SmolVLM2-500M accelerator engine: weight loading and inference."""
    # --- Model dimensions (SmolVLM2-500M) — loaded from smolvlm2_config.json ---
    _cfg = _SMOLVLM2_CFG
    HIDDEN_SIZE       = _cfg["lm"]["hidden_size"]
    NUM_LAYERS        = _cfg["lm"]["num_layers"]
    NUM_HEADS         = _cfg["lm"]["num_heads"]
    NUM_KV_HEADS      = _cfg["lm"]["num_kv_heads"]
    GROUP_SIZE        = _cfg["lm"]["group_size"]
    HEAD_DIM          = _cfg["lm"]["head_dim"]
    INTERMEDIATE_SIZE = _cfg["lm"]["intermediate_size"]
    VOCAB_SIZE        = _cfg["lm"]["vocab_size"]
    ROPE_THETA        = _cfg["lm"]["rope_theta"]
    RMS_NORM_EPS      = _cfg["lm"]["rms_norm_eps"]
    MAX_POSITION_EMBEDDINGS = _cfg["lm"]["max_position_embeddings"]
    # ISA register assignments (fixed, used by compile_decoder and run_decoder)
    V_CACHE_SIZE_REG = _cfg["fixed_isa_regs"]["V_CACHE_SIZE_REG"]
    ROPE_SIZE_REG    = _cfg["fixed_isa_regs"]["ROPE_SIZE_REG"]
    TMP_REG          = _cfg["fixed_isa_regs"]["TMP_REG"]
    GPR_SEQ_LEN_REG    = _cfg["fixed_isa_regs"]["GPR_SEQ_LEN_REG"]
    GPR_Q_SEQ_LEN_REG  = _cfg["fixed_isa_regs"]["GPR_Q_SEQ_LEN_REG"]
    GPR_ALIGNED_SEQ_LEN_REG = _cfg["fixed_isa_regs"]["GPR_ALIGNED_SEQ_LEN_REG"]
    PREFILL_MAX_SEQ_LEN = _cfg["model"]["prefill_max_seq_len"]

    def __init__(self, script_dir: str = None):
        self.script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
        self._cfg = _SMOLVLM2_CFG
        # Fixed precision scheme: bf16 vision + q4 LM → the bf16 DRAM layout.
        dl = self._cfg["dram_layout"]["bf16"]
        super().__init__(
            params_dram_base=int(dl["params_dram_base"], 16),
            tensor_dram_base=int(dl["tensor_dram_base"], 16),
            program_dram_base=int(dl["program_dram_base"], 16),
        )
        self._isa_reg_counter = 1
        self.gpr_seq_len = self.GPR_SEQ_LEN_REG       # primed to S in run_prefill / seq pos in decode
        self.gpr_q_seq_len = self.GPR_Q_SEQ_LEN_REG   # primed to S*GROUP_SIZE in run_prefill
        self.gpr_aligned_seq_len = self.GPR_ALIGNED_SEQ_LEN_REG
        self.vision_bf16 = True
        self.decode_matmat_mul_core_enable = False
        self.penalty_enable = False

    def _artifact_mode_suffix(self) -> str:
        # Only the decode linear kernel changes the compiled artifacts. The repetition penalty is a pure
        # RUNTIME tensor-DRAM write (PENALTY_BIAS_DRAM, always wired as the LM-head C term with zeros =
        # greedy), so it never affects params.bin or programs.bin and is not part of the suffix.
        if bool(getattr(self, "decode_matmat_mul_core_enable", False)):
            return "_decode_matmat_mul_core"
        return ""

    def _artifact_mode_meta(self) -> dict:
        return {
            "decode_matmat_mul_core_enable": bool(getattr(self, "decode_matmat_mul_core_enable", False)),
        }

    def _validate_artifact_mode(self, meta: dict, artifact_name: str) -> None:
        expected = self._artifact_mode_meta()
        missing = [k for k in expected if k not in meta]
        if missing:
            raise RuntimeError(
                f"[artifact-mode] {artifact_name} is missing metadata keys {missing}; rebuild it with "
                "smolvlm2_test.py"
            )
        mismatched = [f"{k}={meta.get(k)!r}" for k, v in expected.items() if bool(meta.get(k)) != bool(v)]
        if mismatched:
            exp = ", ".join(f"{k}={v}" for k, v in expected.items())
            got = ", ".join(mismatched)
            raise RuntimeError(
                f"[artifact-mode] {artifact_name} metadata mismatch: expected {exp}, got {got}"
            )

    def zero_dram(self, chunk_size_bytes: int = 64 * 1024 * 1024) -> None:
        """Zero-fill the DRAM working range [DRAM_START_ADDR..0xFFFFFFFF] with 0x00 (mirror of
        smolvlm2_test.zero_dram). The base clear_dram() fills 0xFF (NaN in bf16), which poisons any
        region read-before-write on the NEXT run — back-to-back runs then emit garbage / endoftext.
        run_from_bin calls this at the END of main() so consecutive load-only runs each start clean."""
        start = user_dma_core.DRAM_START_ADDR
        total = 0xFFFFFFFF - start + 1
        zeros = b"\x00" * chunk_size_bytes
        offset = 0
        while offset < total:
            n = min(chunk_size_bytes, total - offset)
            self.dma_write(DMA_DEVICE_H2C, start + offset, zeros[:n], n)
            offset += n

    def load_snapshot(self) -> bool:
        """Load params DRAM from snapshot bin + restore all address metadata. Returns True if loaded."""
        bin_dir = os.path.join(self.script_dir, "smolvlm2_bin")
        suffix = self._artifact_mode_suffix()
        bin_path = os.path.join(bin_dir, f"params{suffix}.bin")
        meta_path = os.path.join(bin_dir, f"params{suffix}.json")
        if not os.path.exists(bin_path) or not os.path.exists(meta_path):
            raise RuntimeError(
                f"[artifact-mode] missing snapshot artifact for current mode: "
                f"{os.path.basename(bin_path)} / {os.path.basename(meta_path)}"
            )
        with open(meta_path) as f:
            meta = json.load(f)
        self._validate_artifact_mode(meta, os.path.basename(meta_path))
        total = meta["params_size"]
        CHUNK = 1 * 1024 * 1024
        with open(bin_path, "rb") as f:
            offset = 0
            while offset < total:
                data = f.read(min(CHUNK, total - offset))
                self.dma_write(DMA_DEVICE_H2C, self._params_dram_base + offset, data, len(data))
                offset += len(data)
        self.allocate_params_dram(total)
        self.allocate_tensor_dram(meta["tensor_size"])
        for key, val in meta.items():
            if key not in ("params_size", "tensor_size"):
                setattr(self, key, val)
        model_dir = os.path.join(self.script_dir, "smolvlm2_bin", "SmolVLM2-500M-Video-Instruct")
        # Offline-only: load the shipped tokenizer files from disk; never reach the network. This script
        # is releasable standalone (bins + tokenizer dir) and must run without internet.
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
        _original_print(f"  Snapshot loaded: {total / 1024**2:.1f} MB")
        return True

    def load_all(self) -> dict:
        """Load the unified single instruction bin (encoder + decoder + prefill) — mirror of
        smolvlm2_test.compile_all's cache-hit path. DMAs each segment to its recorded absolute address
        and sets the program/preamble addrs the run_* methods read. Must be called AFTER load_snapshot
        (which restores params, including the shared vis_zeros LN buffer — Trick 9, no replay needed)."""
        bin_dir = os.path.join(self.script_dir, "smolvlm2_bin")
        suffix = self._artifact_mode_suffix()
        meta_path = os.path.join(bin_dir, f"programs{suffix}.json")
        bin_path = os.path.join(bin_dir, f"programs{suffix}.bin")
        with open(meta_path) as f:
            meta = json.load(f)
        self._validate_artifact_mode(meta, os.path.basename(meta_path))
        with open(bin_path, "rb") as f:
            raw = f.read()
        for entry in meta["segments"]:
            b = raw[entry["off"]:entry["off"] + entry["size"]]
            self.dma_write(DMA_DEVICE_H2C, entry["addr"], b, len(b))
        self._vis_program_addr = meta["encoder_addr"]
        self._decoder_program_addr = meta["decoder_addr"]
        self._decoder_preamble_addr = meta["decoder_preamble"]
        self._decoder_attention_use_pbi = meta.get("decoder_attention_use_pbi")
        self._decoder_attention_impl = meta.get("decoder_attention_impl")
        self._decoder_attention_shared_subroutine = meta.get("decoder_attention_shared_subroutine")
        self._prefill_addr = meta["prefill_addr"]
        self._prefill_preamble_addr = meta["prefill_preamble"]
        self._prefill_postamble_addr = meta["prefill_postamble"]
        self._prefill_final_buf = meta["prefill_final_buf"]
        self._next_program_dram_addr = meta["end_addr"]
        if (self._decoder_attention_impl is None or self._decoder_attention_use_pbi is None or
                self._decoder_attention_shared_subroutine is None):
            raise RuntimeError(
                "[decoder-attn-path] loaded programs.json lacks decoder-attention branch metadata; "
                "the artifact is stale/incompatible — rebuild it with smolvlm2_test.py")
        if bool(self._decoder_attention_use_pbi) is not False or self._decoder_attention_impl != "unified_attention_core":
            raise RuntimeError(
                "[decoder-attn-path] loaded artifact is not the required SmolVLM2 unified decoder-attention implementation")
        artifact_sha256 = hashlib.sha256(raw).hexdigest()
        self._loaded_artifact_sha256 = artifact_sha256
        # _original_print: this is a required runtime deliverable, so it must survive _SILENT_MODE
        # (load_all runs after _SILENT_MODE=True in main).
        _original_print(
            f"    [decoder-attn-path] model=smolvlm2 phase=lm_decode "
            f"shared_subroutine=no use_pbi=false implementation=unified_attention_core "
            f"artifact_sha256={artifact_sha256} "
            f"metadata_validated=yes"
        )
        print(f"    Loaded unified instruction bin ({len(raw)/1024/1024:.1f} MB, "
              f"{len(meta['segments'])} segments)")
        return meta

    # --- Run ---
    def run_encoder(self, encoder_addr: int, pixel_values) -> None:
        """Run vision encoder. Output lands in VIS_CONNECTOR_DRAM."""
        pixels_bf16 = pixel_values.to(torch.bfloat16).contiguous().flatten()
        self.dma_to_accelerator_memory(self.VIS_PIXEL_IN_DRAM, pixels_bf16)
        # Encoder FLOPs: patch proj + pos add + N layers + post LN + connector
        VS, VH, VD, VN, VI = 1024, 768, 64, 12, 3072
        n_vis = 12  # SigLIP-base vision encoder always has 12 layers
        attn_per_head = VS * VD + 2 * VS * VD * VS + VS * VS * 5 + 2 * VS * VS * VD
        per_layer = (7 * VS * VH + 3 * (2 * VS * VH * VH + VS * VH)
            + VN * attn_per_head + 2 * VS * VH * VH + VS * VH
            + 8 * VS * VH + 2 * VS * VH * VI + VS * VI
            + 2 * VS * VI * VH + VS * VH + VS * VH)
        enc_flops = (2 * VS * VH * VH + VS * VH + VS * VH
            + n_vis * per_layer + 7 * VS * VH
            + 2 * 64 * 12288 * self.HIDDEN_SIZE)
        self.program_execute(encoder_addr, timeout=30.0, flops=enc_flops)

    def run_prefill(self, token_ids, has_image: bool = False, total_flops: int = None) -> int:
        """Run the seq-len-agnostic prefill (dynamic-PBI / §7): host prep (epsilon pad + block bias),
        a preamble that does on-device embed/merge + primes dynamic sequence GPRs + jumps into the cached
        prefill, then a postamble that does final-norm + LM head on the runtime last token. Mirrors
        smolvlm2_test.run_prefill exactly (runtime-only; no compilation). Returns argmax."""
        from user_dma_core import TYPE
        seq_len = len(token_ids)
        self.seq_len = seq_len
        self._prefill_token_ids = list(token_ids)  # seeds the repetition-penalty window in run_decoder
        H, D, G, bpe = self.HIDDEN_SIZE, self.HEAD_DIM, self.GROUP_SIZE, 2
        PM = self.PREFILL_MAX_SEQ_LEN
        assert seq_len <= PM, f"prompt {seq_len} exceeds PREFILL_MAX_SEQ_LEN={PM}"
        qmax = self.PREFILL_QMAX
        qS = seq_len * G
        aligned_q = ((qS + 63) // 64) * 64
        embed_row_bytes = H * bpe

        # Vision/connector execution is complete before run_prefill. Clear all
        # volatile LM attention state before any prefill instruction can read it.
        self._zero_runtime_attention_state(seq_len=seq_len, bucket_len=aligned_q)
        # 1. epsilon-fill INPUT padding rows [seq_len:PM] (non-attention ops run static PM rows)
        if PM > seq_len:
            epsilon = torch.full(((PM - seq_len) * H,), 1e-6, dtype=torch.bfloat16)
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM + seq_len * embed_row_bytes, epsilon)
        # 2. block-diagonal causal bias [aligned_q, aligned_q]: allow (t,g)->(t',g') iff g==g' & t'<=t
        rq = torch.arange(aligned_q).unsqueeze(1)
        rk = torch.arange(aligned_q).unsqueeze(0)
        allow = (rq < qS) & (rk < qS) & (rq % G == rk % G) & (rk // G <= rq // G)
        bias = torch.where(allow, torch.tensor(0.0), torch.tensor(-1e38)).to(torch.bfloat16)
        bias[torch.arange(aligned_q) >= qS, 0] = 0.0  # padded query rows attend row 0 (discarded)
        self.dma_to_accelerator_memory(self.FLASH_BIAS_DRAM, bias)

        # 4. Preamble: on-device embed gather + vision merge + prime dynamic sequence GPRs + jump into prefill
        self.clear_inst_id()
        self.start_capture()
        for t in range(seq_len):
            src = self.embed_addr + token_ids[t] * embed_row_bytes
            self.accelerator_memory_to_sram(src, 0x00000, H)
            self.sram_to_accelerator_memory(0x00000, self.LAYER0_INPUT_DRAM + t * embed_row_bytes, H)
        if has_image:
            img_positions = [i for i, t in enumerate(token_ids) if t == IMAGE_TOKEN_ID]
            if len(img_positions) > 0:
                assert len(img_positions) == IMAGE_SEQ_LEN, \
                    f"Expected {IMAGE_SEQ_LEN} image tokens, got {len(img_positions)}"
                for i, pos in enumerate(img_positions):
                    self.accelerator_memory_to_sram(self.VIS_CONNECTOR_DRAM + i * embed_row_bytes, 0x00000, H)
                    self.sram_to_accelerator_memory(0x00000, self.LAYER0_INPUT_DRAM + pos * embed_row_bytes, H)
        self.generate_instruction_add_set(self.gpr_seq_len, seq_len)
        self.generate_instruction_add_set(self.gpr_q_seq_len, qS)
        self.generate_instruction_add_set(self.gpr_aligned_seq_len, aligned_q)
        self.generate_instruction_jump_abs(ue_35bit_addr_shifter(self._prefill_addr))
        self.stop_capture()
        self.write_captured_instructions_to_dram(self._prefill_preamble_addr)
        self.clear_capture_buffer()
        self.start_execute_from_dram(self._prefill_preamble_addr)
        self.wait_queue(180.0)
        # Static prefill writes epsilon-derived KV rows through PREFILL_MAX.
        # Retain real tokens and clear every remaining context row before decode.
        self._zero_runtime_attention_state(
            seq_len=seq_len, bucket_len=aligned_q, preserve_kv_prefix=True
        )

        # 5. Postamble: final norm over ALL tokens + LM head on the last real token.
        last_off = (seq_len - 1) * H * bpe
        self.clear_inst_id()
        self.start_capture()
        self.rms_norm_core_dram(M=seq_len, N=H, A_DRAM_ADDR=self._prefill_final_buf,
            OUTPUT_DRAM_ADDR=self.FINAL_NORM_DRAM, GAMMA_DRAM_ADDR=self.final_norm_addr)
        last_norm = self.FINAL_NORM_DRAM + last_off
        # Prefill LM-head stays fused in both modes (mirrors smolvlm2_test.run_prefill): quantized_matmat_core
        # is the optimized GEMV for the 49280-wide vocab; the deterministic flag only de-fuses per-layer linears.
        self.quantized_matmat_core(M=1, K=H, N=self.VOCAB_SIZE, A_DRAM_ADDR=last_norm,
            B_DRAM_ADDR=self.lm_head_data, OUTPUT_DRAM_ADDR=self.LOGITS_DRAM,
            SCALE_DRAM_ADDR=self.lm_head_scale, data_type=TYPE.IF4)
        self.generate_instruction_halt()
        self.stop_capture()
        self.write_captured_instructions_to_dram(self._prefill_postamble_addr)
        self.clear_capture_buffer()
        self.start_execute_from_dram(self._prefill_postamble_addr)
        self.wait_queue(30.0)
        if total_flops is not None:
            try:
                self._last_hw_gflops, _ = self.report_flop_rate_gflops(total_flops)
            except ZeroDivisionError:
                self._last_hw_gflops = None
            self._last_total_flops = total_flops
        else:
            self._last_hw_gflops = None
            self._last_total_flops = None
        return self.get_arg_max_index()

    def _structural_token_ids(self):
        """Token ids never repetition-penalized: punctuation, whitespace, and special tokens. Cached."""
        cached = getattr(self, "_struct_ids_cache", None)
        if cached is not None:
            return cached
        import string
        allowed = set(string.punctuation) | set(string.whitespace) | set("—–’‘“”…·•‹›«»¡¿")
        ids = set(int(i) for i in (getattr(self.tokenizer, "all_special_ids", []) or []))
        ids |= set(int(v) for v in _SMOLVLM2_CFG["special_tokens"].values())
        for i in range(self.VOCAB_SIZE):
            s = self.tokenizer.decode([i]).strip()
            if s == "" or all(ch in allowed for ch in s):
                ids.add(i)
        self._struct_ids_cache = ids
        return ids

    def _structural_ids_tensor(self):
        t = getattr(self, "_struct_ids_tensor_cache", None)
        if t is None:
            t = torch.tensor(sorted(self._structural_token_ids()), dtype=torch.long)
            self._struct_ids_tensor_cache = t
        return t

    def _write_penalty_bias(self, prev_tokens) -> None:
        """Per-vocab additive repetition bias → PENALTY_BIAS_DRAM (decode LM-head C term). bias[t] =
        clamp(−alpha·count[t], min=−cap); structural tokens stay 0. HW argmax of (logits+bias) penalizes."""
        vocab = self.VOCAB_SIZE
        alpha = float(getattr(self, "pen_alpha", 3.0))
        cap = float(getattr(self, "pen_cap", 100.0))
        W = int(getattr(self, "rep_window", 256))
        window = prev_tokens[-W:]
        count = torch.zeros(vocab, dtype=torch.float32)
        if window:
            win = torch.tensor(window, dtype=torch.long)
            count.index_add_(0, win, torch.ones(win.numel(), dtype=torch.float32))
            count[self._structural_ids_tensor()] = 0.0
        bias = (-alpha * count).clamp(min=-cap)
        # Short-window loop backstop (mirrors smolvlm2_test): hard-ban (-1e9) any NON-structural token
        # that fills >= pen_loop_thr of the last pen_loop_recent tokens — breaks 2-cycles / single-token
        # runs the soft count penalty cannot, pushing the HW argmax toward EOS / new content.
        loop_recent = int(getattr(self, "pen_loop_recent", 24))
        loop_thr = int(getattr(self, "pen_loop_thr", 6))
        recent = prev_tokens[-loop_recent:]
        if len(recent) >= loop_recent:
            rc = torch.zeros(vocab, dtype=torch.float32)
            rt = torch.tensor(recent, dtype=torch.long)
            rc.index_add_(0, rt, torch.ones(rt.numel(), dtype=torch.float32))
            rc[self._structural_ids_tensor()] = 0.0
            bias[rc >= loop_thr] = -1e9
        # Soft length nudge toward EOS (mirrors smolvlm2_test): past a soft token budget, grow a positive
        # bias on the end-of-utterance token so open-ended generations terminate instead of hitting the cap.
        soft = int(getattr(self, "pen_eos_soft", 96))
        n_gen = len(prev_tokens) - len(getattr(self, "_prefill_token_ids", []) or [])
        if n_gen > soft:
            eos = int(self._cfg["special_tokens"]["end_of_utterance_id"])
            bias[eos] = max(float(bias[eos]), float(n_gen - soft))
        bias = bias.to(torch.bfloat16).view(1, vocab)
        self.dma_to_accelerator_memory(self.PENALTY_BIAS_DRAM, bias)

    def run_decoder(
        self,
        token_id: int,
        max_new_tokens: int = 512,
    ) -> list:
        """Auto-regressive decode loop. Each step writes a tiny preamble that primes sequence GPRs."""
        global _SILENT_MODE
        generated = []
        H = self.HIDDEN_SIZE
        bpe = 2
        embed_row_bytes = H * bpe
        decoder_program_addr = self._decoder_program_addr
        preamble_addr = self._decoder_preamble_addr

        # On-FPGA repetition penalty (same as test.py): the decode LM-head always adds PENALTY_BIAS_DRAM;
        # zero it up front, then refresh each step past the greedy_until gate from the windowed frequency.
        _fpga_penalty = bool(getattr(self, "penalty_enable", False))
        _greedy_until = int(getattr(self, "greedy_until", 0))
        self._generated_tokens = list(getattr(self, "_prefill_token_ids", [])) + [token_id]
        self.dma_to_accelerator_memory(self.PENALTY_BIAS_DRAM,
                                       torch.zeros(1, self.VOCAB_SIZE, dtype=torch.bfloat16))
        if _fpga_penalty:
            self._structural_ids_tensor()  # warm the structural-token cache off the first decode token

        _original_print(self.tokenizer.decode([token_id]), end="", flush=True)
        while len(generated) < max_new_tokens and self.seq_len < self.max_seq_len:
            _SILENT_MODE = True
            self.seq_len += 1
            aligned_kv = ((self.seq_len + 63) // 64) * 64

            # Decode bias is compact [GROUP_SIZE, aligned_kv]; unified_attention_core's full-matrix
            # bias row stride is the dynamic aligned_seq_len, not max_seq_len.
            bias = torch.full((self.GROUP_SIZE, aligned_kv), -1e38, dtype=torch.bfloat16)
            bias[:, :self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.DECODE_BIAS_DRAM, bias)

            # Host-side embed DMA — avoids on-device C2H before decoder
            src_addr = self.embed_addr + token_id * embed_row_bytes
            embed_vec = self.dma_from_accelerator_memory(src_addr, torch.Size([H]))
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embed_vec)

            # Preamble: prime dynamic sequence GPRs + jump into decoder
            self.clear_inst_id()
            self.start_capture()
            self.generate_instruction_add_set(self.gpr_seq_len, self.seq_len - 1)
            self.generate_instruction_add_set(self.gpr_q_seq_len, self.seq_len)
            self.generate_instruction_add_set(self.gpr_aligned_seq_len, aligned_kv)
            self.generate_instruction_jump_abs(ue_35bit_addr_shifter(decoder_program_addr))
            self.stop_capture()
            self.write_captured_instructions_to_dram(preamble_addr)
            self.clear_capture_buffer()

            if _fpga_penalty and len(generated) >= _greedy_until:
                self._write_penalty_bias(self._generated_tokens)

            self.start_execute_from_dram(preamble_addr)
            self.wait_queue(30.0)

            # HW argmax: read the 4-byte (penalized) argmax-index register the LM-head left.
            token_id = self.get_arg_max_index()
            generated.append(token_id)
            self._generated_tokens.append(token_id)
            _SILENT_MODE = False
            if token_id in _SMOLVLM2_CFG["model"]["stop_token_ids"]:
                _original_print(f"\nStop token {token_id} reached.")
                break
            _original_print(self.tokenizer.decode([token_id]), end="", flush=True)
        else:
            # Loop ended without hitting an EOS stop token — report which cap stopped it.
            _reason = ("max decode tokens" if len(generated) >= max_new_tokens
                       else "max sequence length")
            _original_print(f"\nStopped: {_reason} ({len(generated)} tokens).")
        _SILENT_MODE = False
        return generated

# =============================================================================
# Image processing + input construction
# =============================================================================
IMAGE_TOKEN_ID = _SMOLVLM2_CFG["special_tokens"]["image_token_id"]
FAKE_TOKEN_ID = _SMOLVLM2_CFG["special_tokens"]["fake_token_id"]
GLOBAL_IMG_TOKEN_ID = _SMOLVLM2_CFG["special_tokens"]["global_img_token_id"]
IMAGE_SEQ_LEN = _SMOLVLM2_CFG["model"]["image_seq_len"]

def process_image(image_path: str, size: int = 512) -> torch.Tensor:
    """Load, resize, normalize image → [3, size, size] bf16."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # normalize to [-1, 1]
    return torch.from_numpy(arr).permute(2, 0, 1).to(torch.bfloat16)

def build_input_ids(tokenizer, prompt: str, has_image: bool = True) -> list:
    """Build token ID list with chat template and optional image token expansion."""
    if has_image:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    tokens = tokenizer.encode(prompt_text)
    if not has_image:
        return tokens
    # Expand single <image> token → <fake><global-img><image>×64<fake>
    expanded = []
    for t in tokens:
        if t == IMAGE_TOKEN_ID:
            expanded += [FAKE_TOKEN_ID, GLOBAL_IMG_TOKEN_ID] + [IMAGE_TOKEN_ID] * IMAGE_SEQ_LEN + [FAKE_TOKEN_ID]
        else:
            expanded.append(t)
    return expanded

# =============================================================================
# Main
# =============================================================================
def _clock_ns_default_for_device(device: str) -> float:
    """Return default clock period (ns) for FPGA type — mirrors user_hw_test.py."""
    if device == "kintex7":                       return 5.1594
    if device in ("rk", "puzhi"):                 return 3.0
    if device in ("bittware", "bittware_256"):     return 3.3333
    if device == "alveo":                          return 4.0
    return 10.0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SmolVLM2-500M inference from pre-compiled bins")
    _d = _SMOLVLM2_CFG["defaults"]
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _default_image = os.path.join(_root, _d["image"])
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt. Default: 'Describe this image.' (VLM) or a text question (--lm-enable).")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to an image for VLM. Default: the bundled sample image. Ignored with --lm-enable.")
    parser.add_argument("--lm-enable", action="store_true",
                        help="Pure language-model (text-only) mode — skip the vision encoder. Default is VLM (vision).")
    parser.add_argument("--dev", type=str, default=_d["dev"])
    parser.add_argument("--cycle", type=float, default=None, help='Clock cycle time in ns. Overrides --device default.')
    parser.add_argument("--device", type=str, default='kintex7', help='FPGA board profile (kintex7, rk, puzhi, bittware, bittware_256, alveo).')
    parser.add_argument("--max-seq", type=int, default=_d["max_seq"])
    parser.add_argument("--max-decode-tokens", type=int, default=None,
                        help="Cap the number of generated decode tokens.")
    parser.add_argument(
        "--decode-matmat_mul_core-enable",
        action="store_true",
        help="Use matmat_mul_core for decode linear layers instead of quantized_matmat_core.",
    )
    parser.add_argument(
        "--greedy-enable",
        action="store_true",
        help="Use pure greedy decoding. Default generation uses the on-FPGA repetition penalty.",
    )
    args = parser.parse_args()

    global _SILENT_MODE
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bin_dir = os.path.join(script_dir, "smolvlm2_bin")
    # Penalty is a pure runtime tensor write, so it never affects the artifact names; only the decode
    # linear kernel does (see _artifact_mode_suffix).
    snapshot_suffix = "_decode_matmat_mul_core" if args.decode_matmat_mul_core_enable else ""
    instruction_suffix = snapshot_suffix

    # Check all static bins before any hardware init. SmolVLM2 uses one program artifact:
    # programs.bin / programs.json.
    def _missing_static_bins():
        missing = []
        for name in (f"params{snapshot_suffix}.bin", f"params{snapshot_suffix}.json",
                     f"programs{instruction_suffix}.bin", f"programs{instruction_suffix}.json"):
            if not os.path.exists(os.path.join(bin_dir, name)):
                missing.append(name)
        return missing

    early_missing = _missing_static_bins()
    if early_missing:
        _original_print("Missing required bin files (these must be shipped in smolvlm2_bin/ alongside this script):")
        for f in early_missing:
            _original_print(f"  {os.path.join(bin_dir, f)}")
        return

    # Default is VLM (vision). The bin is always the full VLM bin, so the encoder is available either way;
    # this only decides whether vision RUNS. --lm-enable (or --image none) selects pure LM-only text mode.
    lm_only = bool(args.lm_enable) or (args.image is not None
                                       and str(args.image).strip().lower() in ("none", ""))
    vision_on = not lm_only
    if vision_on and (args.image is None or str(args.image).strip().lower() in ("none", "")):
        args.image = _default_image
    if args.prompt is None:           # mode-appropriate default prompt
        args.prompt = _d["lm_prompt"] if lm_only else _d["vlm_prompt"]
    has_image = vision_on

    set_dma_device(args.dev)
    axi_width_bits = 512 if args.device in ("bittware", "rk") else 256
    os.environ["UE_AXI_DATA_WIDTH_BITS"] = str(axi_width_bits)
    user_dma_core.UE_AXI_DATA_WIDTH_BITS = axi_width_bits
    clock = args.cycle if args.cycle is not None else _clock_ns_default_for_device(args.device)
    user_dma_core.CLOCK_CYCLE_TIME_NS = clock
    user_dma_core.UE_PEAK_GFLOPS = 0.128 / clock
    _original_print(f"FPGA profile: device={args.device}, clock={clock:.4f} ns, UE_AXI_DATA_WIDTH_BITS={axi_width_bits}")
    _SILENT_MODE = True

    ue = SmolVLM2_UnifiedEngine(script_dir=script_dir)
    ue.decode_matmat_mul_core_enable = bool(args.decode_matmat_mul_core_enable)
    ue.penalty_enable = not bool(args.greedy_enable)
    _original_print(f"decode_linear={ 'if4_matmat_mul_core' if ue.decode_matmat_mul_core_enable else 'quantized_matmat_core' }")
    _original_print(f"generation={ 'hardware_penalty' if ue.penalty_enable else 'greedy' }")
    init_hang_prevention(ue)
    ue.load_snapshot()

    token_ids = build_input_ids(ue.tokenizer, args.prompt, has_image=has_image)
    seq_len = len(token_ids)
    num_image_tokens = sum(t == IMAGE_TOKEN_ID for t in token_ids)
    num_text_tokens = seq_len - num_image_tokens

    if has_image:
        _original_print(f"Image: {os.path.abspath(args.image)}")
    _original_print(f"Prompt: {args.prompt!r} ({seq_len} tokens, image={'yes' if has_image else 'no'})")

    # Load the ONE unified instruction bin (encoder + decoder + prefill).
    timer = time.perf_counter()
    uni_meta = ue.load_all()
    enc_addr = uni_meta["encoder_addr"]
    _original_print(f"  Loaded from bin in {time.perf_counter() - timer:.2f}s")

    # Vision encoder
    if has_image:
        timer = time.perf_counter()
        pixel_values = process_image(args.image)
        stop_enc = threading.Event()
        def _enc_progress():
            while not stop_enc.wait(1.0):
                _original_print(f"\r  Vision encoder ({time.perf_counter() - timer:.0f}s)", end="", flush=True)
        threading.Thread(target=_enc_progress, daemon=True).start()
        ue.run_encoder(enc_addr, pixel_values)
        stop_enc.set()
        _original_print(f"\r  Vision encoder ({time.perf_counter() - timer:.0f}s) done")

    # Prefill
    timer = time.perf_counter()
    padded_seq = ((seq_len + 63) // 64) * 64
    H, D, I = ue.HIDDEN_SIZE, ue.HEAD_DIM, ue.INTERMEDIATE_SIZE
    KVH, QH, G = ue.NUM_KV_HEADS, ue.NUM_HEADS, ue.GROUP_SIZE
    S = padded_seq
    attn_per_group = (D * S + G * (S * D + 2 * S * D * S + S * S + S * S * 5 + 2 * S * S * D))
    per_layer = (4 * S * H + 2 * S * H * H + 2 * 2 * S * H * KVH * D
        + (QH + KVH) * S * D * 4 + KVH * attn_per_group
        + 2 * S * H * H + 5 * S * H + 2 * (2 * S * H * I) + S * I + 2 * S * I * H + S * H)
    prefill_flops = ue.NUM_LAYERS * per_layer + 4 * S * H + 2 * 1 * H * ue.VOCAB_SIZE
    stop_pf = threading.Event()
    def _pf_progress():
        while not stop_pf.wait(1.0):
            _original_print(f"\r  Prefill ({time.perf_counter() - timer:.0f}s)", end="", flush=True)
    threading.Thread(target=_pf_progress, daemon=True).start()
    hw_token = ue.run_prefill(token_ids, has_image=has_image, total_flops=prefill_flops)
    stop_pf.set()
    prefill_time = time.perf_counter() - timer
    _original_print(f"\r  Prefill ({prefill_time:.0f}s) done")

    _greedy_until = _d["vlm_greedy_until"] if vision_on else _d["lm_greedy_until"]
    ue.penalty_enable = not bool(args.greedy_enable)
    ue.greedy_until = _greedy_until
    max_new = args.max_decode_tokens if args.max_decode_tokens is not None else args.max_seq - seq_len
    _original_print(f"\nPrompt:   {args.prompt}")
    _original_print(f"Response: ", end="", flush=True)   # the generated answer streams right after this
    # run_decoder prints the seed token itself (mirrors smolvlm2_test.run_decoder), so don't pre-print.
    decode_timer = time.perf_counter()
    hw_tokens = ue.run_decoder(
        hw_token,
        max_new_tokens=max_new,
    )
    decode_time = time.perf_counter() - decode_timer
    _original_print(f"\nDecoder done in {prefill_time + decode_time:.2f}s total, {len(hw_tokens)} tokens.")
    _original_print(
        f"prefill {round(seq_len / prefill_time, 2) if prefill_time > 0 else 0.0} tok/s, "
        f"decode {round(len(hw_tokens) / decode_time, 2) if decode_time > 0 else 0.0} tok/s.")

    _SILENT_MODE = True
    ue.software_reset()


if __name__ == "__main__":
    main()
