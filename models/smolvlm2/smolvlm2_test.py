#!/usr/bin/env python3
"""SmolVLM2-500M on accelerator: bf16 (vision) & q4_64 (language). Use --vision-fp4 for FP4 vision."""
import builtins
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

# Suppress prints during decode/compile (same as Gemma3)
_original_print = builtins.print
_SILENT_MODE = False
def _quiet_print(*args, **kwargs):
    if not _SILENT_MODE:
        _original_print(*args, **kwargs)
builtins.print = _quiet_print
import torch
from huggingface_hub import snapshot_download

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, DMA_DEVICE_C2H, TYPE, UE_VECTOR_SIZE,
    URAM_NEAR_FULL_ELEMENTS,
    DRAM_INSTRUCTION_ADDR,
    UnifiedEngine, set_dma_device, ue_35bit_addr_shifter, UE_MODE,
)
from nn_lib import (
    smart_bf16_permute_core,
    store_weight, store_quantized_weight, store_identity_matrix,
    eltwise_add_core_dram, eltwise_mul_core_dram,
    rms_norm_core_dram_post_add,
)
from quant_lib import quantize_q4_64 as _mlc_quantize_q4_64
def _load_smolvlm2_config(path: str | None = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smolvlm2_config.json")
    with open(path) as f:
        return json.load(f)

_SMOLVLM2_CFG = _load_smolvlm2_config()
HF_MODEL_REPO = _SMOLVLM2_CFG["paths"]["hf_model_repo"]


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
# Helper Methods for SmolVLM2
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
# GGUF generation — quantization helpers
# =============================================================================
def quantize_q4_64(tensor):
    """Quantize an LM weight tensor to q4_64 (used inline by weight_init — no intermediate bin)."""
    return _mlc_quantize_q4_64(tensor)

def clean_bins(bin_dir: str, verbose: bool = True) -> list:
    """Delete every cached artifact in `bin_dir`, for ALL engine configurations.

    NAMES THE FILES; never rm -rf the directory. The downloaded HF checkpoint lives
    INSIDE the bin dir (smolvlm2_bin/SmolVLM2-500M-Video-Instruct/, see weight_init), so
    removing the directory would throw the weights away and force a re-download. Only
    params*.bin/.json and programs*.bin/.json are touched.
    """
    import glob
    victims = sorted(
        glob.glob(os.path.join(bin_dir, "params*.bin"))
        + glob.glob(os.path.join(bin_dir, "params*.json"))
        + glob.glob(os.path.join(bin_dir, "programs*.bin"))
        + glob.glob(os.path.join(bin_dir, "programs*.json")))
    freed = 0
    for p in victims:
        freed += os.path.getsize(p)
        os.remove(p)
    if verbose:
        if victims:
            print(f"[clean] removed {len(victims)} artifact(s), "
                  f"{freed / 1024**3:.2f} GB from {bin_dir}")
            for p in victims:
                print(f"    {os.path.basename(p)}")
        else:
            print(f"[clean] no artifacts in {bin_dir} (nothing to remove)")
    return victims


class SmolVLM2_UnifiedEngine(SmolVLM2RuntimeAttentionStateMixin, UnifiedEngine):
    """SmolVLM2-500M accelerator engine: weight loading, compile, inference."""
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
    # Prefill PBI (seq-len-agnostic, gemma3-style): three runtime GPRs primed by run_prefill.
    #   gpr_seq_len    — token count S (matmul/norm/rope/eltwise/gather row loops)
    #   gpr_q_seq_len  — S * GROUP_SIZE (token-major stacked Q rope row loop)
    #   gpr_aligned_seq_len — aligned attention K/V length for unified_attention_core
    GPR_SEQ_LEN_REG    = _cfg["fixed_isa_regs"]["GPR_SEQ_LEN_REG"]
    GPR_Q_SEQ_LEN_REG  = _cfg["fixed_isa_regs"]["GPR_Q_SEQ_LEN_REG"]
    GPR_ALIGNED_SEQ_LEN_REG = _cfg["fixed_isa_regs"]["GPR_ALIGNED_SEQ_LEN_REG"]
    # Max prompt length the compile-once prefill program supports (sets flash bucket count).
    PREFILL_MAX_SEQ_LEN = _cfg["model"]["prefill_max_seq_len"]
    # --- multi-engine vision encoder (--engines N) -------------------------------
    # Row-shards the ENTIRE vision-encoder layer body (LayerNorms, q/k/v, attention,
    # o, residuals, fc1, fc2) across N engines over the 1024 patch tokens. Only the
    # two permutes and the M=64 connector stay on the primary. See
    # compile_encoder_sharded.
    NUM_ENGINES = 2
    # Worker arenas, pi05 layout: one contiguous ARENA_BYTES slice per worker taken
    # from the TENSOR allocator, so they sit inside the 512 MB tensor region and the
    # model's own cursor moves past them. NOT multi_engine_shard's default
    # (DRAM_START_ADDR + 0x10000000 = 0x90000000), which lands inside smolvlm2's 1 GB
    # params window and would overwrite weights.
    #
    # Each worker needs ~2.4 MB of PER_ENGINE scratch (4 flash buffers + the 2.2 MB
    # attention scratch) inside the 16 MB params/tensor window, and its encoder
    # program in the 32 MB above it. 7 workers (--engines 8) take 336 MB of the
    # ~400 MB the tensor region has spare after tensor_init.
    VIS_WORKER_ARENA_BYTES    = 0x03000000   # 48 MB per worker
    VIS_WORKER_TENSOR_OFFSET  = 0x00100000   #  1 MB
    VIS_WORKER_PROGRAM_OFFSET = 0x01000000   # 16 MB
    # generate_instruction_flag_check accepts engine indices 0-15
    # (user_dma_core.py:9351); 8 is what the DRAM budget above supports.
    MAX_ENGINES = 8
    # Decode-side sharding is a SEPARATE knob (--decode-engines), default 1 = off.
    # It is an experiment, not a default: decode is a weight-streaming GEMV, so an
    # N-split halves the bytes EACH engine reads but leaves the TOTAL bytes moved
    # unchanged. It only wins if the current limit is per-engine issue rate rather
    # than the DRAM controller -- see compile_decoder_sharded.
    DECODE_NUM_ENGINES = 1
    # --- prefill sharding (follows --engines) ------------------------------------
    # Row-block granularity. Nothing in the prefill body consumes a row count as
    # anything but a runtime M -- rms_norm/matmul take gpr_M_reg, eltwise takes M/N,
    # and attention is GROUP-sharded at full S with every strided DMA left intact --
    # so this is a load-balance knob, not a correctness one. (pi05 uses 8 for its
    # prefix for the same reason.) The word-address arithmetic is exact at ANY
    # offset because every pitch is a multiple of 8.
    # Per-stage override; None -> follow NUM_ENGINES (pi05's VIS_NUM_ENGINES
    # convention). Prefill is M=PM with real weight reuse, so unlike decode it is
    # compute-bound and shards well: ~1.79x on two engines.
    PREFILL_NUM_ENGINES = None
    PREFILL_ROW_ALIGN = 8
    # Shard registers, primed per run by the preambles in run_prefill. 7..11 sit
    # above the fixed regs (1..6); scratch allocation starts past them.
    # NOTE gpr_M_reg must be 1..15 (user_dma_core eltwise/rms_norm validation), which
    # PF_GPR_SHARD_ROWS=7 satisfies; the address GPRs may be 1..63 but must be
    # distinct from it and from each other (_validate_addr_gprs).
    PF_GPR_SHARD_ROWS   = 7    # this engine's row count  -> gpr_M_reg
    PF_GPR_ROW_OFF      = 8    # this engine's first row  -> address arithmetic
    PF_GPR_ADDR_A       = 9    # scratch: computed A base   (word address)
    PF_GPR_ADDR_B       = 10   # scratch: computed B base   (eltwise only)
    PF_GPR_ADDR_OUT     = 11   # scratch: computed OUT base
    PF_ISA_SCRATCH_BASE = 12
    def __init__(self, script_dir: str = None):
        self.script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
        self._cfg = _SMOLVLM2_CFG
        # Fixed precision scheme: bf16 vision + q4 LM → the bf16 DRAM layout (params 1 GB / tensors /
        # instructions). No other precision options.
        dl = self._cfg["dram_layout"]["bf16"]
        super().__init__(
            params_dram_base=int(dl["params_dram_base"], 16),
            tensor_dram_base=int(dl["tensor_dram_base"], 16),
            program_dram_base=int(dl["program_dram_base"], 16),
        )
        self._isa_reg_counter = 1
        self.gpr_seq_len = self.GPR_SEQ_LEN_REG      # primed to S in run_prefill
        self.gpr_q_seq_len = self.GPR_Q_SEQ_LEN_REG  # primed to S*GROUP_SIZE in run_prefill
        self.gpr_aligned_seq_len = self.GPR_ALIGNED_SEQ_LEN_REG
        # Unified single-bin assembly (compile_all): when active, the three compile_* methods stash
        # their (program_addr, bytes) into these instead of writing separate per-program bins.
        self._unified_active = False
        self._seg_encoder = self._seg_prefill = self._seg_decoder = None
        self.decode_matmat_mul_core_enable = False
        self.penalty_enable = False
        self.vision_bf16 = True
        # Multi-engine vision encoder state (see compile_encoder_sharded).
        self._vis_sched = None
        self._vis_worker_prog_addrs = []
        self._vis_flags_precleared = False
        self._lm_sched = None
        self._pf_sched = None
        self._vis_worker_arena = None
        self._pf_worker_prog_addrs = []
        self._pf_worker_preamble_addrs = []
        self._lm_worker_prog_addrs = []
    # --- ISA register helpers (same as Gemma3) ---
    def _artifact_mode_suffix(self) -> str:
        # Only the decode linear kernel changes the compiled instruction stream (and, trivially, the
        # weight layout). The repetition penalty is a pure RUNTIME tensor-DRAM write (PENALTY_BIAS_DRAM,
        # always wired as the LM-head C term with zeros = greedy), so it never affects params.bin or
        # programs.bin and is deliberately NOT part of the artifact suffix/metadata.
        if bool(getattr(self, "decode_matmat_mul_core_enable", False)):
            return "_decode_matmat_mul_core"
        return ""

    def _artifact_engine_suffix(self) -> str:
        """Engine configuration, as a programs-file suffix.

        SEPARATE from _artifact_mode_suffix because params{suffix}.bin uses that one and
        the params snapshot is engine-count INVARIANT: sharding never writes the params
        region (every per-engine buffer is registered out of the worker TENSOR arena), so
        one params.bin serves every configuration. The programs are the opposite -- a
        sharded program bakes one rendezvous per engine and cannot be replayed at a
        different count -- so they get one file per (encoder, prefill, decode) triple.
        """
        return (f"_e{int(self.NUM_ENGINES)}_{self._prefill_engines()}"
                f"_{int(self.DECODE_NUM_ENGINES)}")

    def _programs_paths(self, bin_dir: str):
        suffix = self._artifact_mode_suffix() + self._artifact_engine_suffix()
        return (os.path.join(bin_dir, f"programs{suffix}.bin"),
                os.path.join(bin_dir, f"programs{suffix}.json"), suffix)

    def _record_worker_prog_sizes(self, attr: str, sched) -> None:
        """Snapshot each worker program's size right after finalize().

        Must happen HERE: finalize() flushes the capture but leaves the buffer in place,
        and the next stage's begin_program() clears it on the same shared workers. The
        size cannot be recovered by differencing worker program addresses either --
        allocate_program_dram rounds every allocation up to 64 B.
        """
        setattr(self, attr + "_sizes",
                [w.get_capture_instruction_size_bytes() for w in sched.workers])

    def _worker_pool(self):
        """The shared worker engines (owned by the encoder scheduler)."""
        return self._vis_sched.workers if self._vis_sched is not None else []

    # stage -> (worker program addr attr, worker size attr)
    _WORKER_STAGES = (("encoder", "_vis_worker_prog_addrs"),
                      ("prefill", "_pf_worker_prog_addrs"),
                      ("decode",  "_lm_worker_prog_addrs"))

    def _collect_worker_segments(self, raw: bytearray) -> tuple[list, dict]:
        """Read every worker program back and append it to `raw`.

        A sharded stage is not one program: each extra engine executes its own stream out
        of its own arena slice. Replaying only the primary's would leave the worker rows
        unwritten while the primary waits at a rendezvous nobody answers.
        """
        segs, addrs = [], {}
        pool = self._worker_pool()
        for stage, attr in self._WORKER_STAGES:
            waddrs = list(getattr(self, attr, None) or [])
            wsizes = list(getattr(self, attr + "_sizes", None) or [])
            addrs[stage] = waddrs
            assert len(wsizes) == len(waddrs), (
                f"{attr}: {len(waddrs)} worker address(es) but {len(wsizes)} size(s) -- "
                f"_record_worker_prog_sizes did not run for {stage!r}")
            for wi, (addr, size) in enumerate(zip(waddrs, wsizes)):
                buf = bytearray(size)
                w = pool[wi]
                got = w.dma_read(DMA_DEVICE_C2H, addr, buf, size)
                if got != size:
                    raise RuntimeError(
                        f"dma_read of {stage} worker {wi + 1} program at 0x{addr:X} "
                        f"returned {got}/{size} B -- refusing to write a bin holding a "
                        f"zero-filled program that would masquerade as real code")
                segs.append({"name": f"{stage}_w{wi + 1}", "addr": addr,
                             "off": len(raw), "size": size, "engine": wi + 1})
                raw.extend(buf)
        return segs, addrs

    def _restore_worker_programs(self, meta: dict, raw: bytes) -> None:
        """Rebuild the schedulers, DMA every worker program back, restore the addresses.

        The scheduler builders are replayed rather than persisted: every per-engine
        registration lives INSIDE them (_vis_make_scheduler / _pf_make_scheduler), and
        they allocate from the worker tensor arenas in a fixed order, so calling them in
        the compile order reproduces byte-identical addresses.
        """
        if not self._is_sharded():
            return
        self._vis_make_scheduler(int(self.NUM_ENGINES))     # owns the shared pool
        if int(self.DECODE_NUM_ENGINES) > 1:
            self._lm_make_scheduler(int(self.DECODE_NUM_ENGINES))
        if self._prefill_engines() > 1:
            self._pf_make_scheduler(self._prefill_engines())

        pool = self._worker_pool()
        for seg in meta.get("segments", []):
            e = seg.get("engine")
            if not e:
                continue                       # primary segment, already written
            w = pool[e - 1]
            b = raw[seg["off"]:seg["off"] + seg["size"]]
            w.dma_write(DMA_DEVICE_H2C, seg["addr"], b, len(b))

        wa = meta.get("worker_prog_addrs") or {}
        self._vis_worker_prog_addrs = list(wa.get("encoder") or [])
        self._pf_worker_prog_addrs = list(wa.get("prefill") or [])
        self._lm_worker_prog_addrs = list(wa.get("decode") or [])
        # Prefill preambles are RESERVED slots, not compiled code: run_prefill rewrites
        # them every run with that run's row split. Only the addresses need restoring.
        self._pf_worker_preamble_addrs = list(meta.get("pf_worker_preambles") or [])
        for w, end in zip(pool, meta.get("worker_prog_end") or []):
            w._next_program_dram_addr = int(end)

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
        """Zero-fill the DRAM working range [DRAM_START_ADDR..0xFFFFFFFF].

        NOTE: the base ``clear_dram()`` fills with 0xFF, which is NaN in bf16. That
        poisons any region read-before-write on the *next* run (back-to-back runs
        emit token 0 / endoftext immediately). Calling this at startup guarantees a
        clean (zeroed) DRAM regardless of what the previous run left behind.
        Temporary mitigation until the read-before-write gap is found and fixed.
        """
        start = user_dma_core.DRAM_START_ADDR
        total = 0xFFFFFFFF - start + 1
        zeros = b"\x00" * chunk_size_bytes
        offset = 0
        bar_width = 40
        _original_print(f"Zeroing DRAM [{hex(start)}..0xffffffff] ({total / 1024**3:.2f} GB)")
        while offset < total:
            n = min(chunk_size_bytes, total - offset)
            self.dma_write(DMA_DEVICE_H2C, start + offset, zeros[:n], n)
            offset += n
            pct = offset / total
            filled = int(bar_width * pct)
            bar = '█' * filled + '░' * (bar_width - filled)
            _original_print(f"\r  [{bar}] {pct*100:5.1f}%  {offset/1024**2:.0f}/{total/1024**2:.0f} MB",
                            end='', flush=True)
        _original_print()
    # --- Weight loading ---
    def weight_init(self) -> None:
        """Build ALL weights into params DRAM straight from the HF model — q4_64 LM (quantized on the
        fly) + bf16 vision. No intermediate weight bins: dump_snapshot then captures the assembled
        params DRAM into the single params.bin. (Fixed scheme: bf16 vision + q4 LM; no other options.)"""
        from transformers import AutoTokenizer, AutoModelForImageTextToText
        model_dir = os.path.join(self.script_dir, "smolvlm2_bin", "SmolVLM2-500M-Video-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        hf_model = AutoModelForImageTextToText.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map=None).eval()

        # --- Language model weights: q4_64 projections; bf16 embeddings/norms ---
        text_model = hf_model.model.text_model
        embed_bf16 = text_model.embed_tokens.weight.data.to(torch.bfloat16)
        self.embedding_weight = embed_bf16.clone()  # CPU copy for host-side token embedding lookup
        self.embed_addr = store_weight(self, embed_bf16)
        self.lm_layer_addrs = []
        for layer in text_model.layers:
            la = {}
            for proj, attr in [('q', 'q_proj'), ('k', 'k_proj'), ('v', 'v_proj'), ('o', 'o_proj'),
                               ('gate', 'gate_proj'), ('up', 'up_proj'), ('down', 'down_proj')]:
                module = layer.self_attn if proj in ('q', 'k', 'v', 'o') else layer.mlp
                weight = getattr(module, attr).weight.data
                data, _ = quantize_q4_64(weight)
                la[f'{proj}_scale'], la[f'{proj}_data'] = store_quantized_weight(self, data.tobytes())
                la[f'{proj}_bf16'] = None
            la['ln1_gamma'] = store_weight(self, layer.input_layernorm.weight.data.to(torch.bfloat16))
            la['ln2_gamma'] = store_weight(self, layer.post_attention_layernorm.weight.data.to(torch.bfloat16))
            self.lm_layer_addrs.append(la)
        self.final_norm_addr = store_weight(self, text_model.norm.weight.data.to(torch.bfloat16))
        data, _ = quantize_q4_64(hf_model.lm_head.weight.data)
        self.lm_head_scale, self.lm_head_data = store_quantized_weight(self, data.tobytes())
        print(f"LM weights loaded (Q4): {self.NUM_LAYERS} layers, params DRAM usage: {self.get_params_dram_usage()} bytes")
        # Identity matrix for decode attention
        self.identity_addr = store_identity_matrix(self)
        # Shared zeros buffer for the vision encoder's LayerNorms (Trick 9 — the released
        # layer_norm_core_dram self-allocates+dma_writes a zeros operand at COMPILE time, which the
        # bin-load path never re-emits → stale DRAM/NaN). Seeding it HERE (params DRAM, before
        # dump_snapshot) makes it part of the weights snapshot, so build + every load path get it with
        # no replay. All encoder LNs use N=768. Passed as ZEROS_DRAM_ADDR to each layer_norm call.
        _vz = 768  # vision hidden dim (all encoder LayerNorms use N=768)
        self.vis_zeros_addr = self.get_params_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, self.vis_zeros_addr, torch.zeros(_vz, dtype=torch.bfloat16), _vz * 2)
        self.allocate_params_dram(_vz * 2)
        # The synced dynamic LayerNorm core requires an explicit 1/N vector.
        # Store one shared copy in params DRAM so params.bin restores the exact
        # operand baked into the encoder program.
        self.vis_inv_n_addr = self.get_params_dram_addr()
        self.dma_write(
            DMA_DEVICE_H2C, self.vis_inv_n_addr,
            torch.full((_vz,), 1.0 / _vz, dtype=torch.bfloat16), _vz * 2)
        self.allocate_params_dram(_vz * 2)
        # Shared zero page for diagnostic device-side clears inside the decoder instruction stream.
        # Keep it in params DRAM so load_snapshot() restores it and the clear mode does not depend on
        # runtime tensor_init() side effects.
        self.device_attn_clear_zero_addr = self.get_params_dram_addr()
        self.device_attn_clear_zero_bytes = 0x24000
        self.dma_write(
            DMA_DEVICE_H2C,
            self.device_attn_clear_zero_addr,
            torch.zeros(self.device_attn_clear_zero_bytes // 2, dtype=torch.bfloat16),
            self.device_attn_clear_zero_bytes,
        )
        self.allocate_params_dram(self.device_attn_clear_zero_bytes)
        params_used = self.get_params_dram_usage()
        params_limit = self._tensor_dram_base - self._params_dram_base
        _original_print(f"  Params DRAM: {params_used/1024/1024:.1f} MB used / {params_limit/1024/1024:.0f} MB available"
                        + (" OVERFLOW!" if params_used > params_limit else ""))

        # --- Vision encoder weights (bf16) — always built so the program bin is the full VLM bin
        # (one bin, robust + easy to maintain); whether vision RUNS is a runtime decision (--image). ---
        NPS = 32  # num_patches_per_side = 512 / 16
        boundaries = torch.arange(1.0 / NPS, 1.0, 1.0 / NPS, dtype=torch.float32)
        frac = torch.arange(NPS, dtype=torch.float32) / NPS * (1 - 1e-6)
        buckets = torch.bucketize(frac, boundaries, right=True)
        vis_position_ids = (buckets[:, None] * NPS + buckets[None, :]).flatten()
        vis_enc = hf_model.model.vision_model
        self.vis_layer_addrs = []
        for layer in vis_enc.encoder.layers:
            la = {}
            for proj, attr in [('q', 'q_proj'), ('k', 'k_proj'), ('v', 'v_proj'), ('o', 'out_proj')]:
                linear = getattr(layer.self_attn, attr)
                la[f'{proj}_weight'] = store_weight(self, linear.weight.data)
                la[f'{proj}_bias'] = store_weight(self, linear.bias.data)
            for proj, attr in [('fc1', 'fc1'), ('fc2', 'fc2')]:
                linear = getattr(layer.mlp, attr)
                la[f'{proj}_weight'] = store_weight(self, linear.weight.data)
                la[f'{proj}_bias'] = store_weight(self, linear.bias.data)
            for ln, attr in [('ln1', 'layer_norm1'), ('ln2', 'layer_norm2')]:
                norm = getattr(layer, attr)
                la[f'{ln}_weight'] = store_weight(self, norm.weight.data)
                la[f'{ln}_bias'] = store_weight(self, norm.bias.data)
            self.vis_layer_addrs.append(la)
        vis_layers = len(vis_enc.encoder.layers)
        self.patch_weight_addr = store_weight(self, vis_enc.embeddings.patch_embedding.weight.data.reshape(768, 768))
        self.patch_bias_addr = store_weight(self, vis_enc.embeddings.patch_embedding.bias.data)
        pos_table = vis_enc.embeddings.position_embedding.weight.data  # [1024, 768]
        self.pos_embed_addr = store_weight(self, pos_table[vis_position_ids])
        self.vis_post_ln_weight = store_weight(self, vis_enc.post_layernorm.weight.data)
        self.vis_post_ln_bias = store_weight(self, vis_enc.post_layernorm.bias.data)
        self.connector_weight_addr = store_weight(self, hf_model.model.connector.modality_projection.proj.weight.data)
        del hf_model
        params_used = self.get_params_dram_usage()
        _original_print(f"  Vision BF16 loaded: {vis_layers} layers, total params: {params_used/1024/1024:.1f} MB")
    _SNAP_SCALARS = (int, float, bool, str, type(None))

    def _capture_tensor_attrs(self, before) -> dict:
        """Every JSON-safe attribute tensor_init published.

        Captured by DIFFING __dict__ rather than from the hand-written addr_attrs list.
        That list is what drifts: VIS_FLASH_Q/K/V/OUT_DRAM were added to tensor_init for
        the sharded encoder but never added to the list, so a snapshot-backed multi-engine
        run had no attribute for _vis_make_scheduler to register. It was invisible before
        only because 1-engine runs never build a scheduler and multi-engine runs never
        loaded a bin.
        """
        def ok(v, d=0):
            if d > 6:
                return False
            if isinstance(v, self._SNAP_SCALARS):
                return True
            if isinstance(v, (list, tuple)):
                return all(ok(x, d + 1) for x in v)
            if isinstance(v, dict):
                return all(isinstance(k, str) and ok(x, d + 1) for k, x in v.items())
            return False

        def norm(v):
            if isinstance(v, (list, tuple)):
                return [norm(x) for x in v]
            if isinstance(v, dict):
                return {k: norm(x) for k, x in v.items()}
            return v

        return {k: norm(self.__dict__[k])
                for k in sorted(set(self.__dict__) - set(before))
                if ok(self.__dict__[k])}

    def tensor_init(self, max_seq_len: int = 512) -> None:
        _snap_before = set(self.__dict__)
        try:
            return self._tensor_init_inner(max_seq_len)
        finally:
            self._tensor_attrs = self._capture_tensor_attrs(_snap_before)

    def _tensor_init_inner(self, max_seq_len: int = 512) -> None:
        """Allocate DRAM for activations, KV cache, masks, RoPE."""
        self.max_seq_len = max_seq_len
        seq_len = max_seq_len
        bpe = 2
        # KV cache offset constants (Gemma3-style flat layout: [layer, head, seq, dim])
        self.bytes_per_element = bpe
        self.k_size = self.HEAD_DIM * bpe                             # 128 bytes per position per head
        self.KV_HEAD_STRIDE = seq_len * self.k_size                   # one head, all positions
        self.KV_LAYER_STRIDE = self.NUM_KV_HEADS * self.KV_HEAD_STRIDE  # all heads, one layer

        print(f"Allocate tensor dram start at DRAM address: 0x{self.get_tensor_dram_addr():X}")
        # KV cache: [NUM_LAYERS, NUM_KV_HEADS, max_seq_len, HEAD_DIM], zero-padded
        kv_cache_total = self.NUM_LAYERS * self.KV_LAYER_STRIDE
        self.LAYER0_K_DRAM = self.allocate_tensor_dram(kv_cache_total)
        self.LAYER0_V_DRAM = self.allocate_tensor_dram(kv_cache_total)
        # Redundant build-path safety only. Runtime correctness must not depend
        # on tensor_init(): run_prefill initializes volatile attention state via
        # _zero_runtime_attention_state() on both fresh-build and load paths.
        kv_zeros = torch.zeros(self.NUM_LAYERS * self.NUM_KV_HEADS * seq_len * self.HEAD_DIM, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_K_DRAM, kv_zeros)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, kv_zeros)
        # Decoder layer intermediates (shared across 32 layers):
        self.LAYER0_INPUT_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LAYER0_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LAYER0_PRE_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LAYER0_Q_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LAYER0_K_PROJ_DRAM = self.allocate_tensor_dram(seq_len * self.NUM_KV_HEADS * self.HEAD_DIM * bpe)
        self.LAYER0_V_PROJ_DRAM = self.allocate_tensor_dram(seq_len * self.NUM_KV_HEADS * self.HEAD_DIM * bpe)
        self.LAYER0_Q_PERM_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LAYER0_ATTN_OUT_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LAYER0_ATTN_SCRATCH_DRAM = self.allocate_tensor_dram((self.HEAD_DIM * seq_len + max(self.HEAD_DIM, 64) * seq_len) * bpe)
        self.LAYER0_ATTN_RESULT_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LAYER0_O_PROJ_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LAYER0_RESIDUAL_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LAYER0_MLP_GATE_DRAM = self.allocate_tensor_dram(seq_len * self.INTERMEDIATE_SIZE * bpe)
        self.LAYER0_MLP_UP_DRAM = self.allocate_tensor_dram(seq_len * self.INTERMEDIATE_SIZE * bpe)
        self.LAYER0_MLP_MULT_DRAM = self.allocate_tensor_dram(seq_len * self.INTERMEDIATE_SIZE * bpe)
        self.LAYER0_MLP_DOWN_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.FINAL_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LOGITS_DRAM = self.allocate_tensor_dram(1 * self.VOCAB_SIZE * bpe)
        # Causal mask for prefill (written by compile_prefill/load_prefill before use)
        self.CAUSAL_MASK_DRAM = self.allocate_tensor_dram(seq_len * seq_len * bpe)
        # Decode bias: [GROUP_SIZE, max_seq_len] — written each decode step.
        self.DECODE_BIAS_DRAM = self.allocate_tensor_dram(self.GROUP_SIZE * seq_len * bpe)
        # --- stacked-GQA unified-attention buffers (seq-len-agnostic prefill/decode) ---
        # Prefill operates on aligned(S*GROUP_SIZE) stacked rows; decode reuses the same K/V buffers
        # with batch=GROUP_SIZE and aligned_seq_len=aligned(current KV length).
        G = self.GROUP_SIZE
        qmax = ((self.PREFILL_MAX_SEQ_LEN * G + 63) // 64) * 64
        self.PREFILL_QMAX = qmax
        D = self.HEAD_DIM
        self.FLASH_Q_DRAM = self.allocate_tensor_dram(qmax * D * bpe)       # token-major stacked Q
        self.FLASH_K_DRAM = self.allocate_tensor_dram(qmax * D * bpe)       # K duplicated token-major
        self.FLASH_V_DRAM = self.allocate_tensor_dram(qmax * D * bpe)       # V duplicated token-major
        self.FLASH_OUT_DRAM = self.allocate_tensor_dram(qmax * D * bpe)     # attention output (stacked)
        self.FLASH_SCRATCH_DRAM = self.allocate_tensor_dram(
            ((D + qmax) * qmax + qmax * D) * bpe)
        self.FLASH_BIAS_DRAM = self.allocate_tensor_dram(qmax * qmax * bpe)    # block-diagonal causal bias
        # RoPE cos/sin tables
        self._load_rope_tables()
        # Vision encoder intermediates (fixed seq=1024, hidden=768, intermediate=3072):
        VS, VH, VI = 1024, 768, 3072
        self.VIS_PIXEL_IN_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_PATCH_PERM_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_PATCH_PROJ_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_IO_A_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_IO_B_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_LN_OUT_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_Q_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_K_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_V_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_Q_PERM_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_K_PERM_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_V_PERM_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_ATTN_OUT_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_ATTN_SCRATCH_DRAM = self.allocate_tensor_dram(((64 + VS) * VS + VS * 64) * bpe)
        self.VIS_ATTN_RESULT_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_O_PROJ_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_RESIDUAL_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_MLP_INTER_DRAM = self.allocate_tensor_dram(VS * VI * bpe)
        self.VIS_MLP_OUT_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_POST_LN_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        # Connector: pixel shuffle [1024,768]→[64,12288] (scale_factor=4), linear→[64,960]
        self.VIS_SHUFFLED_DRAM = self.allocate_tensor_dram(64 * 12288 * bpe)
        self.VIS_CONNECTOR_DRAM = self.allocate_tensor_dram(64 * self.HIDDEN_SIZE * bpe)
        # Permute params (identity matrix + temp for bf16_smart_permute_core)
        permute_size = UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe
        self.PERMUTE_PARAMS_DRAM = self.get_params_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, self.PERMUTE_PARAMS_DRAM,
                       torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16), permute_size)
        self.allocate_params_dram(permute_size)
        self.PERMUTE_TEMP_DRAM = self.get_tensor_dram_addr()
        self.allocate_tensor_dram(VS * VH * bpe * 2)  # temp space for permute decomposition
        # Vision unified-attention single-head operand buffers ([VS, VD]).
        VD = self.HEAD_DIM
        self.VIS_FLASH_Q_DRAM = self.allocate_tensor_dram(VS * VD * bpe)
        self.VIS_FLASH_K_DRAM = self.allocate_tensor_dram(VS * VD * bpe)
        self.VIS_FLASH_V_DRAM = self.allocate_tensor_dram(VS * VD * bpe)
        self.VIS_FLASH_OUT_DRAM = self.allocate_tensor_dram(VS * VD * bpe)
        self.VIS_ATTN_BIAS_DRAM = self.allocate_tensor_dram(VS * VS * bpe)
        self.dma_to_accelerator_memory(self.VIS_ATTN_BIAS_DRAM, torch.zeros(VS, VS, dtype=torch.bfloat16))
        # Fixed scratch for the SHARED vision layer body: the small per-layer params (biases ≤ 3072,
        # layer_norm gamma/beta = 768) are marshalled here from their register-driven source address
        # so the matmul/layernorm kernels read a static address. (Big weights go through gpr_B_reg.)
        VI = 3072  # vision intermediate (largest bias = fc1)
        self.VIS_BIAS_SCRATCH = self.allocate_tensor_dram(VI * bpe)
        self.VIS_GAMMA_SCRATCH = self.allocate_tensor_dram(VH * bpe)
        self.VIS_BETA_SCRATCH = self.allocate_tensor_dram(VH * bpe)
        print(f"    Allocate tensor dram end at DRAM address: 0x{self.get_tensor_dram_addr():X}, usage: {self.get_tensor_dram_usage()} bytes")
    def _load_rope_tables(self) -> None:
        """Precompute cos/sin tables, pre-negate sin, DMA to device."""
        D, S, bpe = self.HEAD_DIM, self.max_seq_len, 2
        inv_freq = 1.0 / (self.ROPE_THETA ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
        freqs = torch.outer(torch.arange(S, dtype=torch.float32), inv_freq)  # [S, D/2]
        cos_full = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=1).to(torch.bfloat16).contiguous()
        sin_full = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=1).to(torch.bfloat16).contiguous()
        sin_full[:, :D // 2] = -sin_full[:, :D // 2]  # pre-negate for HW RoPE kernel
        table_size = S * D * bpe
        self.ROPE_COS_DRAM = self.get_params_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, self.ROPE_COS_DRAM, cos_full.flatten(), table_size)
        self.allocate_params_dram(table_size)
        self.ROPE_SIN_DRAM = self.get_params_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, self.ROPE_SIN_DRAM, sin_full.flatten(), table_size)
        self.allocate_params_dram(table_size)
        self._hw_cos, self._hw_sin = cos_full, sin_full  # CPU copies for decode position updates
        # Contiguous packed table for the prefill d64 PBI RoPE (rope_hf_core_dram_d64_pbi):
        # per token [cos(D) || sin(D)] = 2D elems, sin first half pre-negated. The d64 path
        # asserts sin_dram_addr == cos_dram_addr + D*bpe, so cos/sin must be interleaved per
        # token (NOT two separate [S,D] tables like ROPE_COS_DRAM/ROPE_SIN_DRAM above).
        packed = torch.cat([cos_full, sin_full], dim=1).contiguous()  # [S, 2D]
        self.ROPE_PACKED_DRAM = self.get_params_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, self.ROPE_PACKED_DRAM, packed.flatten(), S * 2 * D * bpe)
        self.allocate_params_dram(S * 2 * D * bpe)
        # Token-major-DUPLICATED packed table for the gemma3-style stacked-Q rope. The stacked Q
        # for one kv-group is [S, GROUP_SIZE, D] flattened to rows r = t*G + g; roping it as one
        # M=S*G d64 loop needs row r to use token (r//G)'s cos/sin. There is no d64 GQA rope, so
        # we pre-duplicate each token's [cos||sin] row GROUP_SIZE times → row r holds packed[r//G].
        G = self.GROUP_SIZE
        packed_gqa = packed.repeat_interleave(G, dim=0).contiguous()  # [G*S, 2D]
        self.ROPE_PACKED_GQA_DRAM = self.get_params_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, self.ROPE_PACKED_GQA_DRAM, packed_gqa.flatten(), G * S * 2 * D * bpe)
        self.allocate_params_dram(G * S * 2 * D * bpe)
        # Padded-split cos/sin for prefill RoPE (D<128 path):
        # Each 32-element half is zero-padded to 64 elements = 128 bytes = 1 URAM row
        half = D // 2
        if D < 128:
            cos_lo = cos_full[:, :half]  # [S, 32]
            cos_hi = cos_full[:, half:]  # [S, 32]
            sin_lo = sin_full[:, :half]  # [S, 32] (already pre-negated)
            sin_hi = sin_full[:, half:]  # [S, 32]
            pad_D = UE_VECTOR_SIZE  # 64 — pad each half to full URAM row
            for name, data in [('COS_LO', cos_lo), ('COS_HI', cos_hi),
                               ('SIN_LO', sin_lo), ('SIN_HI', sin_hi)]:
                padded = torch.zeros(S, pad_D, dtype=torch.bfloat16)
                padded[:, :half] = data
                padded = padded.contiguous()
                addr = self.get_params_dram_addr()
                self.dma_write(DMA_DEVICE_H2C, addr, padded.flatten(), S * pad_D * bpe)
                self.allocate_params_dram(S * pad_D * bpe)
                setattr(self, f'ROPE_{name}_PAD_DRAM', addr)
            print(f"    RoPE padded-split [{S}, {pad_D}] × 4 arrays DMA'd")
        print(f"    RoPE cos/sin [{S}, {D}] DMA'd, theta={self.ROPE_THETA}")

        # On-FPGA repetition-penalty per-vocab additive bias (the decode LM-head's C term, broadcast_N).
        # Allocated LAST so its presence never shifts any earlier baked tensor address; zeros = no penalty.
        self.PENALTY_BIAS_DRAM = self.allocate_tensor_dram(1 * self.VOCAB_SIZE * bpe)

    # --- Compile ---
    def compile_encoder(self) -> int:
        """Compile vision encoder program (PORTABLE — released API only). Returns DRAM address.

        12 structurally-identical SigLIP layers, UNROLLED, with every per-row op a PBI hardware loop so
        the captured bin is structure-bound, not M-bound. No ``gpr_B_reg`` / layer-body sharing (that
        needed a non-released ``matmat_mul_core`` extension — reverted). The synced dynamic cores all
        consume the actual runtime row count in ``gpr_M_reg``. §7 shared flash + the strided-DMA
        fused head transpose are kept as-is.
        """
        from user_dma_core import TYPE
        S, H, D, N_HEADS, I = 1024, 768, 64, 12, 3072
        bpe = 2

        self.start_capture()

        # Shared runtime row count for matmul, LayerNorm, and attention.
        vis_S_reg = self.alloc_isa_reg()
        self.generate_instruction_add_set(vis_S_reg, S)

        # === Patch embedding: pixels → [1024, 768] ===
        P = 16
        H_patches = 32  # 512 / 16
        smart_bf16_permute_core(self, dims=[3, H_patches, P, H_patches, P], permute_indices=[1, 3, 0, 2, 4],
            input_dram_addr=self.VIS_PIXEL_IN_DRAM, output_dram_addr=self.VIS_PATCH_PERM_DRAM,
            params_dram_addr=self.PERMUTE_PARAMS_DRAM, temp_dram_start=self.PERMUTE_TEMP_DRAM)
        self.matmat_mul_core(M=S, K=H, N=H, A_DRAM_ADDR=self.VIS_PATCH_PERM_DRAM, B_DRAM_ADDR=self.patch_weight_addr,
            OUTPUT_DRAM_ADDR=self.VIS_PATCH_PROJ_DRAM, C_DRAM_ADDR=self.patch_bias_addr, bias_mode="broadcast_N",
            gpr_M_reg=vis_S_reg)
        eltwise_add_core_dram(self, size=S * H, A_DRAM_ADDR=self.VIS_PATCH_PROJ_DRAM,
            B_DRAM_ADDR=self.pos_embed_addr, OUTPUT_DRAM_ADDR=self.VIS_IO_A_DRAM)

        def vis_matmul(M, K, N, A, la, proj, OUT, bias=None, **kw):
            self.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f'{proj}_weight'],
                OUTPUT_DRAM_ADDR=OUT, C_DRAM_ADDR=bias, bias_mode="broadcast_N", gpr_M_reg=vis_S_reg, **kw)

        for layer_idx, la in enumerate(self.vis_layer_addrs):
            h_in  = self.VIS_IO_A_DRAM if layer_idx % 2 == 0 else self.VIS_IO_B_DRAM
            h_out = self.VIS_IO_B_DRAM if layer_idx % 2 == 0 else self.VIS_IO_A_DRAM
            # LN1 (PBI)
            self.layer_norm_core_dram(M=S, N=H, A_DRAM_ADDR=h_in, OUTPUT_DRAM_ADDR=self.VIS_LN_OUT_DRAM,
                GAMMA_DRAM_ADDR=la['ln1_weight'], BETA_DRAM_ADDR=la['ln1_bias'], gpr_M_reg=vis_S_reg,
                ZEROS_DRAM_ADDR=self.vis_zeros_addr, INV_N_DRAM_ADDR=self.vis_inv_n_addr)
            # Q/K/V projections (PBI)
            for proj, dst in [('q', self.VIS_Q_DRAM), ('k', self.VIS_K_DRAM), ('v', self.VIS_V_DRAM)]:
                vis_matmul(S, H, H, self.VIS_LN_OUT_DRAM, la, proj, dst, bias=la[f'{proj}_bias'])
            # §7 attention with fused strided-DMA head transpose (per-head gather → flash → scatter).
            elems = S * D
            col_stride = D * bpe   # one head's column block width
            row_jump = H * bpe     # full [S, 768] row stride
            for h in range(N_HEADS):
                col = h * col_stride
                for src, dst in ((self.VIS_Q_DRAM + col, self.VIS_FLASH_Q_DRAM),
                                 (self.VIS_K_DRAM + col, self.VIS_FLASH_K_DRAM),
                                 (self.VIS_V_DRAM + col, self.VIS_FLASH_V_DRAM)):
                    self.accelerator_memory_to_sram(src, 0x00000, elems,
                        stride_bytes_per_chunk=col_stride, stride_jump_bytes=row_jump)
                    self.sram_to_accelerator_memory(0x00000, dst, elems)
                self.unified_attention_core(batch=S, aligned_seq_len=S, head_dim=D,
                    Q_DRAM_ADDR=self.VIS_FLASH_Q_DRAM, K_DRAM_ADDR=self.VIS_FLASH_K_DRAM,
                    V_DRAM_ADDR=self.VIS_FLASH_V_DRAM, BIAS_DRAM_ADDR=self.VIS_ATTN_BIAS_DRAM,
                    OUTPUT_DRAM_ADDR=self.VIS_FLASH_OUT_DRAM, SCRATCH_DRAM_ADDR=self.VIS_ATTN_SCRATCH_DRAM,
                    IDENTITY_DRAM_ADDR=self.identity_addr,
                    gpr_batch_reg=vis_S_reg,
                    gpr_aligned_seq_len_reg=vis_S_reg)
                self.accelerator_memory_to_sram(self.VIS_FLASH_OUT_DRAM, 0x00000, elems)
                self.sram_to_accelerator_memory(0x00000, self.VIS_ATTN_RESULT_DRAM + col, elems,
                    stride_bytes_per_chunk=col_stride, stride_jump_bytes=row_jump)
            # O projection + residual (eltwise) + LN2 (PBI) — split from the static post-add
            vis_matmul(S, H, H, self.VIS_ATTN_RESULT_DRAM, la, 'o', self.VIS_O_PROJ_DRAM, bias=la['o_bias'])
            eltwise_add_core_dram(self, size=S * H, A_DRAM_ADDR=h_in, B_DRAM_ADDR=self.VIS_O_PROJ_DRAM,
                OUTPUT_DRAM_ADDR=self.VIS_RESIDUAL_DRAM)
            self.layer_norm_core_dram(M=S, N=H, A_DRAM_ADDR=self.VIS_RESIDUAL_DRAM, OUTPUT_DRAM_ADDR=self.VIS_LN_OUT_DRAM,
                GAMMA_DRAM_ADDR=la['ln2_weight'], BETA_DRAM_ADDR=la['ln2_bias'], gpr_M_reg=vis_S_reg,
                ZEROS_DRAM_ADDR=self.vis_zeros_addr, INV_N_DRAM_ADDR=self.vis_inv_n_addr)
            # MLP: fc1 + GELU, fc2, residual
            vis_matmul(S, H, I, self.VIS_LN_OUT_DRAM, la, 'fc1', self.VIS_MLP_INTER_DRAM, bias=la['fc1_bias'], gelu_enable=True)
            vis_matmul(S, I, H, self.VIS_MLP_INTER_DRAM, la, 'fc2', self.VIS_MLP_OUT_DRAM, bias=la['fc2_bias'])
            eltwise_add_core_dram(self, size=S * H,
                A_DRAM_ADDR=self.VIS_RESIDUAL_DRAM, B_DRAM_ADDR=self.VIS_MLP_OUT_DRAM, OUTPUT_DRAM_ADDR=h_out)

        # Post-layernorm (PBI), pixel shuffle, connector (M=64, static).
        final_vis = self.VIS_IO_A_DRAM if len(self.vis_layer_addrs) % 2 == 0 else self.VIS_IO_B_DRAM
        self.layer_norm_core_dram(M=S, N=H, A_DRAM_ADDR=final_vis, OUTPUT_DRAM_ADDR=self.VIS_POST_LN_DRAM,
            GAMMA_DRAM_ADDR=self.vis_post_ln_weight, BETA_DRAM_ADDR=self.vis_post_ln_bias, gpr_M_reg=vis_S_reg,
            ZEROS_DRAM_ADDR=self.vis_zeros_addr, INV_N_DRAM_ADDR=self.vis_inv_n_addr)
        smart_bf16_permute_core(self, dims=[8, 4, 8, 4, H], permute_indices=[0, 2, 1, 3, 4],
            input_dram_addr=self.VIS_POST_LN_DRAM, output_dram_addr=self.VIS_SHUFFLED_DRAM,
            params_dram_addr=self.PERMUTE_PARAMS_DRAM, temp_dram_start=self.PERMUTE_TEMP_DRAM)
        self.matmat_mul_core(M=64, K=12288, N=self.HIDDEN_SIZE,
            A_DRAM_ADDR=self.VIS_SHUFFLED_DRAM, B_DRAM_ADDR=self.connector_weight_addr,
            OUTPUT_DRAM_ADDR=self.VIS_CONNECTOR_DRAM)

        self.generate_instruction_halt()
        self.release_isa_reg()  # vis_S_reg
        self.stop_capture()
        all_bytes = bytearray()
        for inst in self.capture_buffer:
            all_bytes.extend(inst.get_bytes())
        program_addr = self.get_program_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, program_addr, all_bytes, len(all_bytes))
        self.allocate_program_dram(len(all_bytes))
        self.clear_capture_buffer()
        if self._unified_active:
            self._seg_encoder = (program_addr, bytes(all_bytes))
        else:
            # Standalone encoder compile path is not used in production (main always goes through
            # compile_all). Keep the bytes in _seg_encoder so the unified programs.bin writer (if
            # ever invoked with a partial set) has the segment available.
            self._seg_encoder = (program_addr, bytes(all_bytes))
        self._vis_program_addr = program_addr
        print(f"    Vision encoder compiled: {len(all_bytes)} bytes at 0x{program_addr:X}")
        return program_addr


    # ---------------------------------------------------------------- multi-engine --
    def _vis_make_scheduler(self, num_engines: int):
        """Build (once) the MultiEngineScheduler used by compile_encoder_sharded.

        Note what is NOT here, compared with pi05's equivalent: no PER_ENGINE
        registrations and no host-side weight slicing. Only the six projections are
        sharded, and every one of them is a plain matmat_mul_core writing a disjoint
        row block of a buffer that already exists -- nothing sharded writes scratch,
        so vis_zeros / the flash buffers / the attention scratch stay shared and
        untouched. Weights, scales and biases are SHARED_FULL and pass through at
        their existing addresses; only A and OUTPUT carry a row offset.

        The workers are constructed BY the scheduler (workers=None) on purpose: that
        path runs its _save_dram_selftest_region guard. UnifiedEngine.__init__ runs a
        DRAM self-test that writes 16 KB to the hardcoded DRAM_START_ADDR, and
        smolvlm2's params_dram_base IS DRAM_START_ADDR -- so building a worker after
        weight_init() would silently shred the head of the weight blob. That surfaces
        as NaN vision tokens, not as an error.
        """
        from multi_engine_shard import MultiEngineScheduler
        if self._vis_sched is not None:
            assert self._vis_sched.num_engines == num_engines, (
                f"vision scheduler already built for {self._vis_sched.num_engines} "
                f"engine(s), cannot rebuild for {num_engines}")
            return self._vis_sched
        assert 1 <= num_engines <= self.MAX_ENGINES, (
            f"num_engines={num_engines} outside 1..{self.MAX_ENGINES} (the ISA's "
            f"FLAG_CHECK addresses engines 0-15)")
        # FAIL FAST ON MISSING SILICON. Asking for more engines than the fabric has
        # builds workers against AXI-Lite windows with nothing behind them; the first
        # FLAG_CHECK then spin-waits on a flag nobody will ever set, and FLAG_CHECK
        # has NO TIMEOUT. The symptom is a wedged board, not an error, so check here.
        cores = getattr(user_dma_core, 'ANDROMEDA_CORE_COUNT', None)
        if cores is not None and num_engines > int(cores):
            raise RuntimeError(
                f'--engines {num_engines} but this board reports {cores} core(s) '
                f'(HW_INFO). The extra engines do not exist: every barrier would '
                f'spin forever on a flag nobody sets, and FLAG_CHECK has no timeout.')
        # Carve the arenas out of the TENSOR allocator (pi05 layout). Taking them from
        # the model's own cursor is what keeps them from colliding with anything
        # allocated later -- the cursor simply moves past them.
        n_workers = num_engines - 1
        arena = self.get_tensor_dram_addr()
        if n_workers:
            need = n_workers * self.VIS_WORKER_ARENA_BYTES
            free = self._program_dram_base - arena
            assert need <= free, (
                f"--engines {num_engines} needs {need >> 20} MB of worker arenas but "
                f"only {free >> 20} MB of tensor DRAM is left at 0x{arena:X}")
            self.allocate_tensor_dram(need)
            self._vis_worker_arena = arena
        self._vis_sched = MultiEngineScheduler(
            self, num_engines=num_engines,
            worker_dram_base=arena,
            worker_dram_stride=self.VIS_WORKER_ARENA_BYTES,
            worker_tensor_offset=self.VIS_WORKER_TENSOR_OFFSET,
            worker_program_offset=self.VIS_WORKER_PROGRAM_OFFSET,
            row_align=64,
            # "blocks", not "even": an even split asserts unless num_engines divides
            # 1024/64 = 16 blocks exactly, which rules out 3, 5, 6, 7, 12... Block
            # splitting stays 64-aligned at any count and degrades to load imbalance
            # instead of a compile error. compile_encoder_sharded reports the cost.
            split_mode="blocks",
            # >2 is unverified in multi_engine_shard's own guard; --engines names the
            # count explicitly and the core-count check above is the real gate.
            allow_more_than_two_engines=True,
        )
        # PER_ENGINE: the flash operand buffers and the attention scratch are WRITTEN
        # as scratch, so each engine needs its own copy or two engines racing on one
        # buffer produce finite garbage. K and V are held FULL-SIZE in every copy --
        # only the QUERY rows are split, so each engine reads all 1024 keys.
        #
        # vis_zeros and vis_inv_n stay SHARED on purpose: the dynamic layer_norm core
        # only PRELOADS them DRAM->SRAM (user_dma_core.py:3811) and never writes them
        # back, so concurrent readers are safe. (pi05 duplicates its vis_zeros; that
        # is a different norm ABI, not a rule.)
        _S, _D, _bpe = 1024, 64, 2
        for _nm, _addr in (("flash_q", self.VIS_FLASH_Q_DRAM),
                           ("flash_k", self.VIS_FLASH_K_DRAM),
                           ("flash_v", self.VIS_FLASH_V_DRAM),
                           ("flash_out", self.VIS_FLASH_OUT_DRAM)):
            self._vis_sched.register_per_engine(_nm, _addr, _S * _D * _bpe)
        # Match tensor_init's own VIS_ATTN_SCRATCH_DRAM size exactly rather than
        # attn_scratch_bytes(), which omits the trailing S*64 term.
        self._vis_sched.register_per_engine(
            "attn_scratch", self.VIS_ATTN_SCRATCH_DRAM, ((64 + _S) * _S + _S * 64) * _bpe)
        return self._vis_sched

    def _is_sharded(self) -> bool:
        """True if ANY stage is multi-engine. Gates every programs.bin read and
        write: the manifest carries only the primary's segments, so a sharded build
        must neither load a single-engine bin nor publish one."""
        return (int(self.NUM_ENGINES) > 1 or int(self.DECODE_NUM_ENGINES) > 1
                or self._prefill_engines() > 1)

    def _lm_make_scheduler(self, num_engines: int):
        """Scheduler for the decode stage, SHARING the encoder's worker engines.

        The shared pool is a correctness requirement, not an optimization. Each
        worker's DRAM allocators live in the UnifiedEngine object; a second
        scheduler that CONSTRUCTS its own workers over the same arena restarts
        those allocators at offset 0, so the decode worker programs would be
        written on top of the encoder's, byte for byte, at the same addresses.
        The encoder has already cached those addresses, so its next run would
        launch decode code on the workers: different program, different barrier
        count, instant desync, and both engines spin on a FLAG_CHECK that has no
        timeout. Passing ``workers=`` keeps ONE allocator per engine, so decode's
        programs land after the encoder's.
        """
        from multi_engine_shard import MultiEngineScheduler
        if self._lm_sched is not None:
            return self._lm_sched
        # The encoder scheduler owns the pool and is always built first
        # (compile_all compiles encoder -> decoder -> prefill).
        enc = self._vis_make_scheduler(int(self.NUM_ENGINES))
        assert num_engines <= int(self.NUM_ENGINES), (
            f'--decode-engines {num_engines} exceeds --engines {self.NUM_ENGINES}: the\n'
            f'worker engines are built by the encoder stage and shared.')
        self._lm_sched = MultiEngineScheduler(
            self, num_engines=num_engines,
            workers=enc.workers,
            allow_more_than_two_engines=True,
        )
        return self._lm_sched

    def compile_encoder_sharded(self) -> int:
        """compile_encoder with the ENTIRE per-layer body row-sharded over
        ``NUM_ENGINES`` engines. A SEPARATE forward pass: compile_encoder above is
        untouched and remains the only path at --engines 1.

        WHAT IS SHARDED, and the region grouping
        ----------------------------------------
        The whole layer body -- both LayerNorms, q/k/v, all 12 heads of attention,
        o, both residual adds, fc1+GELU and fc2 -- runs sharded over a 512-row block
        of the 1024 patch tokens. Two regions per layer, so TWO barriers:

          region A  LN1, q, k, v
          ---- rendezvous ----   the ONE cross-shard read in the encoder
          region B  attention, o, residual, LN2, fc1, fc2, residual

        Only attention reads rows this engine did not write: it gathers K and V in
        FULL while gathering only its own rows of Q. Everything else is row-local,
        which is why region B needs no internal barrier -- fc1 writes this engine's
        rows of VIS_MLP_INTER and fc2 reads exactly those rows.

        There is no barrier at the LAYER boundary either: layer i writes its own
        rows of h_out and layer i+1 reads the same rows. Region B's closing
        rendezvous is the WAR barrier -- it stops an engine racing into the next
        layer's K/V write while a peer is still gathering K/V.

        Sharding attention REMOVED synchronization rather than adding it: the
        earlier projections-only version needed three regions (6 barriers/layer) to
        hand VIS_Q/K/V and VIS_ATTN_RESULT back and forth with a single-engine
        attention. 24 barriers for the encoder, down from 72.

        Primary-only, and not shardable: the patch permute and the pixel shuffle
        (strided permutes), and the connector matmul (M=64 is one row block).

        AT num_engines == 1 THIS EMITS THE SINGLE-ENGINE STREAM BY CONSTRUCTION, not
        by careful transcription: split_rows(1024) is [(0, 1024)], so every
        rows_addr() offset is 0 and every M is S, and barrier() returns without
        emitting an instruction. It is still not the default path -- compile_encoder
        is -- but that property is what makes the two comparable.
        """
        from user_dma_core import TYPE
        S, H, D, N_HEADS, I = 1024, 768, 64, 12, 3072
        bpe = 2
        ne = int(self.NUM_ENGINES)
        sched = self._vis_make_scheduler(ne)
        # S=1024 is 16 blocks of 64. Counts that divide 16 (1, 2, 4, 8) split evenly;
        # anything else leaves some engines with an extra block, and since every
        # barrier waits for the slowest engine the whole encoder runs at that pace.
        # Say so rather than quietly returning less than the engine count suggests.
        _splits = sched.split_rows(S)
        _rows = [c for _, c in _splits]
        if ne > 1 and max(_rows) != min(_rows):
            print(f"    [vis] WARNING: {ne} engines over {S // 64} blocks of 64 splits "
                  f"{_rows} -- unbalanced. Every barrier waits for the {max(_rows)}-row "
                  f"engine, so the ceiling is {S / max(_rows):.1f}x, not {ne}x. "
                  f"Engine counts that divide {S // 64} (1, 2, 4, 8) split evenly.")

        self.start_capture()
        sched.begin_program()

        # Shared runtime row count for the PRIMARY-ONLY ops (LN, patch embed): still
        # the full S, because those are not sharded.
        vis_S_reg = self.alloc_isa_reg()
        self.generate_instruction_add_set(vis_S_reg, S)

        # Per-engine row-count GPR, allocated ONCE for the whole program.
        # ctx.m_reg would allocate a fresh register per region (the scheduler clears
        # its cache on every begin_sharded), i.e. 36 registers per engine against the
        # ISA's 63 -- it would run out. Every region here splits the same S into the
        # same row counts, so one register per engine is correct for all of them.
        # s_regs holds the FULL key length S. Attention needs both: batch is this
        # engine's query rows (m_reg) while aligned_seq_len is all 1024 keys (s_reg).
        m_regs, s_regs = {}, {}
        for ctx in sched.begin_sharded(S):
            m_regs[ctx.engine_idx] = ctx.m_reg
            _r = ctx.ue.alloc_isa_reg()
            ctx.ue.generate_instruction_add_set(_r, S)
            s_regs[ctx.engine_idx] = _r
        sched.end_sharded(join=True)

        # === Patch embedding: pixels -> [1024, 768] === (primary only)
        P = 16
        H_patches = 32  # 512 / 16
        smart_bf16_permute_core(self, dims=[3, H_patches, P, H_patches, P], permute_indices=[1, 3, 0, 2, 4],
            input_dram_addr=self.VIS_PIXEL_IN_DRAM, output_dram_addr=self.VIS_PATCH_PERM_DRAM,
            params_dram_addr=self.PERMUTE_PARAMS_DRAM, temp_dram_start=self.PERMUTE_TEMP_DRAM)
        self.matmat_mul_core(M=S, K=H, N=H, A_DRAM_ADDR=self.VIS_PATCH_PERM_DRAM, B_DRAM_ADDR=self.patch_weight_addr,
            OUTPUT_DRAM_ADDR=self.VIS_PATCH_PROJ_DRAM, C_DRAM_ADDR=self.patch_bias_addr, bias_mode="broadcast_N",
            gpr_M_reg=vis_S_reg)
        eltwise_add_core_dram(self, size=S * H, A_DRAM_ADDR=self.VIS_PATCH_PROJ_DRAM,
            B_DRAM_ADDR=self.pos_embed_addr, OUTPUT_DRAM_ADDR=self.VIS_IO_A_DRAM)

        def sh_matmul(ctx, K, N, A_base, A_pitch, la, proj, OUT_base, OUT_pitch, bias=None, **kw):
            """One sharded projection. A and OUTPUT carry this engine's row offset;
            the weight and the broadcast_N bias do not (SHARED_FULL)."""
            ctx.ue.matmat_mul_core(
                M=ctx.rows, K=K, N=N,
                A_DRAM_ADDR=ctx.rows_addr(A_base, A_pitch),
                B_DRAM_ADDR=la[f'{proj}_weight'],
                OUTPUT_DRAM_ADDR=ctx.rows_addr(OUT_base, OUT_pitch),
                C_DRAM_ADDR=bias, bias_mode="broadcast_N",
                gpr_M_reg=m_regs[ctx.engine_idx], **kw)

        def sh_layer_norm(ctx, A_base, OUT_base, gamma, beta):
            ctx.ue.layer_norm_core_dram(
                M=ctx.rows, N=H,
                A_DRAM_ADDR=ctx.rows_addr(A_base, H * bpe),
                OUTPUT_DRAM_ADDR=ctx.rows_addr(OUT_base, H * bpe),
                GAMMA_DRAM_ADDR=gamma, BETA_DRAM_ADDR=beta,
                gpr_M_reg=m_regs[ctx.engine_idx],
                ZEROS_DRAM_ADDR=self.vis_zeros_addr, INV_N_DRAM_ADDR=self.vis_inv_n_addr)

        def sh_residual(ctx, A_base, B_base, OUT_base):
            # eltwise_add_core_dram is a flat helper, not an allowlisted method, so it
            # takes the raw engine. FLAT means element i of B pairs with element i of A:
            # every one of the three operands must carry the SAME row offset.
            eltwise_add_core_dram(
                ctx.unsafe_ue, size=ctx.elems(H),
                A_DRAM_ADDR=ctx.rows_addr(A_base, H * bpe),
                B_DRAM_ADDR=ctx.rows_addr(B_base, H * bpe),
                OUTPUT_DRAM_ADDR=ctx.rows_addr(OUT_base, H * bpe))

        col_stride = D * bpe   # one head's column block width (128 B -> AXI aligned)
        row_jump = H * bpe     # full [S, 768] row stride

        def sh_attention(ctx):
            """Per-head bidirectional flash attention over THIS engine's query rows.

            Splitting the QUERY axis needs no online-softmax merge: each query row's
            softmax is already over the full key set, which is why every engine gathers
            K and V in FULL (elems_kv = S*D) while gathering only its own rows of Q.
            That full-K/V read is the ONE place an engine touches rows it did not
            write, and it is what the barrier opening this region pays for.

            TRAP: the Q gather and the output scatter carry the row offset; the K/V
            gathers must NOT. Omitting it on the scatter makes both engines write rows
            0..511 -- finite, NaN-free, structurally plausible scrambled output.
            """
            ue = ctx.unsafe_ue          # strided DMA + attention are not allowlisted ops
            elems_q  = ctx.elems(D)     # THIS engine's query rows
            elems_kv = S * D            # FULL K/V -- every engine reads all keys
            q_rows  = ctx.rows_addr(self.VIS_Q_DRAM, row_jump)
            out_rows = ctx.rows_addr(self.VIS_ATTN_RESULT_DRAM, row_jump)
            FQ = ctx.per_engine("flash_q")
            FK = ctx.per_engine("flash_k")
            FV = ctx.per_engine("flash_v")
            FO = ctx.per_engine("flash_out")
            SCRATCH = ctx.per_engine("attn_scratch")
            for h in range(N_HEADS):
                col = h * col_stride
                assert (ctx.row_offset * row_jump + col) % 32 == 0, \
                    "strided DMA base must be AXI-beat aligned"
                for src, dst, n in ((q_rows + col,               FQ, elems_q),
                                    (self.VIS_K_DRAM + col,      FK, elems_kv),
                                    (self.VIS_V_DRAM + col,      FV, elems_kv)):
                    ue.accelerator_memory_to_sram(src, 0x00000, n,
                        stride_bytes_per_chunk=col_stride, stride_jump_bytes=row_jump)
                    ue.sram_to_accelerator_memory(0x00000, dst, n)
                ue.unified_attention_core(batch=ctx.rows, aligned_seq_len=S, head_dim=D,
                    Q_DRAM_ADDR=FQ, K_DRAM_ADDR=FK, V_DRAM_ADDR=FV,
                    BIAS_DRAM_ADDR=self.VIS_ATTN_BIAS_DRAM,
                    OUTPUT_DRAM_ADDR=FO, SCRATCH_DRAM_ADDR=SCRATCH,
                    IDENTITY_DRAM_ADDR=self.identity_addr,
                    gpr_batch_reg=m_regs[ctx.engine_idx],
                    gpr_aligned_seq_len_reg=s_regs[ctx.engine_idx])
                ue.accelerator_memory_to_sram(FO, 0x00000, elems_q)
                ue.sram_to_accelerator_memory(0x00000, out_rows + col, elems_q,
                    stride_bytes_per_chunk=col_stride, stride_jump_bytes=row_jump)

        n_layers = len(self.vis_layer_addrs)
        for layer_idx, la in enumerate(self.vis_layer_addrs):
            h_in  = self.VIS_IO_A_DRAM if layer_idx % 2 == 0 else self.VIS_IO_B_DRAM
            h_out = self.VIS_IO_B_DRAM if layer_idx % 2 == 0 else self.VIS_IO_A_DRAM

            # ---- REGION A: LN1 + q/k/v ----------------------------------------
            # Ends WITHOUT a join: begin_sharded(B) below emits the rendezvous that
            # region B's full-K/V read needs. join=True here would emit a second,
            # redundant one.
            for ctx in sched.begin_sharded(S):
                sh_layer_norm(ctx, h_in, self.VIS_LN_OUT_DRAM, la['ln1_weight'], la['ln1_bias'])
                for proj, dst in [('q', self.VIS_Q_DRAM), ('k', self.VIS_K_DRAM), ('v', self.VIS_V_DRAM)]:
                    sh_matmul(ctx, H, H, self.VIS_LN_OUT_DRAM, H * bpe, la, proj, dst, H * bpe,
                              bias=la[f'{proj}_bias'])
            sched.end_sharded(join=False)

            # ---- REGION B: attention + o + residual + LN2 + fc1 + fc2 + residual ----
            # Everything after attention is row-local, so it stays in lane with no
            # further exchange: fc1 writes THIS engine's rows of VIS_MLP_INTER and fc2
            # reads the same rows. The closing rendezvous doubles as the WAR barrier --
            # it is what stops an engine racing into the next layer's K/V write while a
            # peer is still gathering K/V here.
            for ctx in sched.begin_sharded(S):
                sh_attention(ctx)
                sh_matmul(ctx, H, H, self.VIS_ATTN_RESULT_DRAM, H * bpe, la, 'o',
                          self.VIS_O_PROJ_DRAM, H * bpe, bias=la['o_bias'])
                sh_residual(ctx, h_in, self.VIS_O_PROJ_DRAM, self.VIS_RESIDUAL_DRAM)
                sh_layer_norm(ctx, self.VIS_RESIDUAL_DRAM, self.VIS_LN_OUT_DRAM,
                              la['ln2_weight'], la['ln2_bias'])
                sh_matmul(ctx, H, I, self.VIS_LN_OUT_DRAM, H * bpe, la, 'fc1',
                          self.VIS_MLP_INTER_DRAM, I * bpe, bias=la['fc1_bias'], gelu_enable=True)
                sh_matmul(ctx, I, H, self.VIS_MLP_INTER_DRAM, I * bpe, la, 'fc2',
                          self.VIS_MLP_OUT_DRAM, H * bpe, bias=la['fc2_bias'])
                sh_residual(ctx, self.VIS_RESIDUAL_DRAM, self.VIS_MLP_OUT_DRAM, h_out)
            # Last layer joins: the primary-only post-LN below reads ALL rows.
            sched.end_sharded(join=(layer_idx == n_layers - 1))

        # Post-layernorm, pixel shuffle, connector (M=64, static). Primary only.
        final_vis = self.VIS_IO_A_DRAM if len(self.vis_layer_addrs) % 2 == 0 else self.VIS_IO_B_DRAM
        self.layer_norm_core_dram(M=S, N=H, A_DRAM_ADDR=final_vis, OUTPUT_DRAM_ADDR=self.VIS_POST_LN_DRAM,
            GAMMA_DRAM_ADDR=self.vis_post_ln_weight, BETA_DRAM_ADDR=self.vis_post_ln_bias, gpr_M_reg=vis_S_reg,
            ZEROS_DRAM_ADDR=self.vis_zeros_addr, INV_N_DRAM_ADDR=self.vis_inv_n_addr)
        smart_bf16_permute_core(self, dims=[8, 4, 8, 4, H], permute_indices=[0, 2, 1, 3, 4],
            input_dram_addr=self.VIS_POST_LN_DRAM, output_dram_addr=self.VIS_SHUFFLED_DRAM,
            params_dram_addr=self.PERMUTE_PARAMS_DRAM, temp_dram_start=self.PERMUTE_TEMP_DRAM)
        self.matmat_mul_core(M=64, K=12288, N=self.HIDDEN_SIZE,
            A_DRAM_ADDR=self.VIS_SHUFFLED_DRAM, B_DRAM_ADDR=self.connector_weight_addr,
            OUTPUT_DRAM_ADDR=self.VIS_CONNECTOR_DRAM)

        self.generate_instruction_halt()
        # Halts + flushes every WORKER program. Keep the addresses: run_encoder needs
        # them, and a later finalize() on this scheduler would overwrite the copy the
        # scheduler holds.
        self._vis_worker_prog_addrs = sched.finalize()
        self._record_worker_prog_sizes("_vis_worker_prog_addrs", sched)
        self.release_isa_reg()  # vis_S_reg
        self.stop_capture()

        all_bytes = bytearray()
        for inst in self.capture_buffer:
            all_bytes.extend(inst.get_bytes())
        program_addr = self.get_program_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, program_addr, all_bytes, len(all_bytes))
        self.allocate_program_dram(len(all_bytes))
        self.clear_capture_buffer()
        self._seg_encoder = (program_addr, bytes(all_bytes))
        self._vis_program_addr = program_addr
        worker_lens = [w.get_capture_instruction_size_bytes() for w in sched.workers]
        extra = f" + {len(worker_lens)} worker program(s) {worker_lens} bytes" if worker_lens else ""
        print(f"    Vision encoder compiled (sharded): {len(all_bytes)} bytes at "
              f"0x{program_addr:X}{extra} (num_engines={ne})")
        return program_addr

    def _prefill_engines(self) -> int:
        """Engine count for the prefill stage; None -> follow NUM_ENGINES."""
        v = self.PREFILL_NUM_ENGINES
        return int(self.NUM_ENGINES) if v is None else int(v)

    def _pf_split_rows(self, S: int, ne: int):
        """[(row_offset, rows)] per engine for a RUNTIME sequence length.

        Called by run_prefill, not by the compiler: the split is primed into GPRs
        each run, so one captured program serves any S. Whole ALIGN-row blocks
        first, then the ragged tail onto the last engine -- the tail is at most
        ALIGN-1 rows, so imbalance is bounded by that regardless of S.
        """
        A = self.PREFILL_ROW_ALIGN
        blocks = S // A
        base, rem = divmod(blocks, ne)
        counts = [A * (base + (1 if i < rem else 0)) for i in range(ne)]
        counts[-1] += S - sum(counts)          # ragged tail (S % ALIGN)
        # NEVER hand an engine zero rows. gpr_M_reg drives a HARDWARE loop; a count of
        # 0 is not a no-op, and the engine would still have to reach every rendezvous.
        # Short prompts (S < ne*ALIGN) land here, so redistribute rather than assert.
        for i in range(ne):
            if counts[i] == 0:
                j = counts.index(max(counts))
                assert counts[j] > 1, (
                    f'sequence length {S} cannot be split across {ne} engines')
                counts[j] -= 1
                counts[i] += 1
        assert sum(counts) == S and all(c >= 1 for c in counts), (counts, S, ne)
        offs = [sum(counts[:i]) for i in range(ne)]
        return list(zip(offs, counts))

    @staticmethod
    def _pf_split_groups(n_groups: int, ne: int):
        """[[kv_head, ...]] per engine. Attention is GROUP-sharded, not row-sharded."""
        base, rem = divmod(n_groups, ne)
        out, start = [], 0
        for i in range(ne):
            c = base + (1 if i < rem else 0)
            out.append(list(range(start, start + c)))
            start += c
        return out

    def _pf_make_scheduler(self, num_engines: int):
        """Prefill scheduler, sharing the encoder's worker engines (see
        _lm_make_scheduler for why the pool must be shared).

        Registers the flash operand buffers PER_ENGINE: attention is group-sharded,
        so two engines run DIFFERENT kv-groups at the same instant and would
        otherwise marshal their Q/K/V into the same FLASH_* buffers.
        """
        from multi_engine_shard import MultiEngineScheduler
        if self._pf_sched is not None:
            return self._pf_sched
        enc = self._vis_make_scheduler(int(self.NUM_ENGINES))
        self._pf_sched = MultiEngineScheduler(
            self, num_engines=num_engines,
            workers=enc.workers,
            allow_more_than_two_engines=True,
        )
        D, bpe = self.HEAD_DIM, 2
        qmax = self.PREFILL_QMAX
        for nm, addr in (("pf_flash_q", self.FLASH_Q_DRAM),
                         ("pf_flash_k", self.FLASH_K_DRAM),
                         ("pf_flash_v", self.FLASH_V_DRAM),
                         ("pf_flash_out", self.FLASH_OUT_DRAM)):
            self._pf_sched.register_per_engine(nm, addr, qmax * D * bpe)
        self._pf_sched.register_per_engine(
            "pf_flash_scratch", self.FLASH_SCRATCH_DRAM,
            ((D + qmax) * qmax + qmax * D) * bpe)
        # FLASH_BIAS stays SHARED: it is read-only and indexed by the FULL stacked
        # batch, which group sharding does not change.
        return self._pf_sched

    def compile_prefill(self) -> None:
        """Seq-len-agnostic gemma3-style prefill. Compiled ONCE for PREFILL_MAX; the runtime length
        is read by unified_attention_core via gpr_aligned_seq_len. Permutes/matmuls/norms run dynamic
        at PREFILL_MAX (shorter prompts padded with epsilon, masked by the block-diagonal bias);
        only the d64 RoPE (gpr) and the O(seq^2) flash (bucket) are runtime-sized.

        Per kv-group GQA into one flash: gather the 3 q-heads token-major [PM,G,D]; rope it (M=PM*G,
        token-major-duplicated table); rope the kv-head into the decoder KV cache then duplicate its
        rows xG token-major; run unified_attention_core; un-stack the output back into
        [PM,H]. Ends after the 32 layers with HALT (no final-norm/LM head — those depend on the
        runtime last-token offset and are emitted by run_prefill). Captured in place at a fixed
        program-DRAM address so the flash dispatcher's absolute jumps stay valid; run_prefill's
        preamble primes runtime sequence GPRs and jump_abs-es into this program.

        MULTI-ENGINE (--engines N > 1)
        ------------------------------
        THE PROBLEM THIS SOLVES. The encoder shards on a COMPILE-TIME row count
        (S=1024), so every engine's row offset folds into the addresses at capture
        time. Prefill cannot do that: every projection takes gpr_M_reg=gpr_seq_len
        and run_prefill primes that register to the ACTUAL prompt length, so the row
        count is a RUNTIME value. A compile-time split would hand engine 1 a block of
        pure epsilon padding on any prompt shorter than PREFILL_MAX.

        THE MECHANISM. matmat_mul_core / rms_norm_core_dram / eltwise_core_dram all
        accept gpr_a_addr / gpr_b_addr / gpr_out_addr on the dynamic path: the
        pointer's DRAM base is sourced from a GPR at execute time (word address,
        byte >> 3) via a one-shot PBI_MODE_REG override, while the in-loop row
        strides stay compile-time immediates (user_dma_core.py:1988). So the
        captured program refers to this engine's row block SYMBOLICALLY:

            PF_GPR_ROW_OFF     this engine's first row      } primed per run by the
            PF_GPR_SHARD_ROWS  this engine's row count      } preamble in run_prefill

        and before each sharded op we emit, into a scratch GPR,
        ``(base + ROW_OFF*pitch) >> 3`` with reg_mul_imm + add_imm -- the same idiom
        compile_decoder already uses for KV-cache addressing. ONE program, ANY
        sequence length. PREFILL_MAX and the seq-len-agnostic property are untouched.

        Every pitch here (H*2=1920, KV*2=640, I*2=5120) is a multiple of 8, so the
        word-address arithmetic is EXACT for any row offset -- PREFILL_ROW_ALIGN is
        about kernel efficiency, not correctness.

        THREE REGIONS PER LAYER:

          A  row-sharded   residual eltwise, input RMSNorm, q/k/v projections
          -- rendezvous --  region B reads ALL rows of Q/K/V
          B  GROUP-sharded  the engine owning kv-head h does its K copy + RoPE-K, its
                            V copy, then Q gather / RoPE-Q / duplicate / flash /
                            un-stack. FULL S rows, so NO row offsets: every strided
                            DMA and the static PM loop inside duplicate_gqa_rows keep
                            their single-engine form. Row-splitting attention instead
                            would need a per-engine slice of the block-diagonal
                            causal bias and a row offset on each strided marshalling
                            copy -- the failure class that yields finite, plausible,
                            scrambled output. pi05 head-shards its prefix for the
                            same reason.
          -- rendezvous --  region C reads ALL rows of ATTN_RESULT
          C  row-sharded   o projection, residual, post-attn RMSNorm, gate/up/mul/down
          -- rendezvous --

        The third rendezvous is CONSERVATIVE and could be dropped: C writes only this
        engine's rows and the next layer's A reads exactly those rows, and the h_in
        ping-pong alternates buffers between consecutive layers so a one-region
        runahead cannot collide. Keeping it costs ~1% of a 2.7 s stage; removing it
        should be a measured change, not an assumed one.

        BALANCE. Projections + MLP are ~93% of prefill MACs and split evenly; the 5
        kv-groups split 3/2 over two engines but are only ~7%, so the ceiling is
        ~1.97x.

        AT N == 1 THIS EMITS THE HISTORICAL STREAM BYTE FOR BYTE: `sharded` is False,
        so no address GPR is computed, no barrier is emitted, gpr_M_reg stays
        gpr_seq_len, every address is the literal, the flat eltwise helpers are used
        as before, and the region split is a pure reordering of nothing (A, B, C
        already ran in that order).
        """
        from user_dma_core import TYPE
        H, D, I = self.HIDDEN_SIZE, self.HEAD_DIM, self.INTERMEDIATE_SIZE
        KV = self.NUM_KV_HEADS * D
        G = self.GROUP_SIZE
        bpe = 2
        PM = self.PREFILL_MAX_SEQ_LEN          # static row count for non-attention ops
        qmax = self.PREFILL_QMAX                # aligned(PM*G) — stacked flash rows

        ne = self._prefill_engines()
        sharded = ne > 1
        sched = self._pf_make_scheduler(ne) if sharded else None
        engines = sched.engines if sharded else [self]
        groups = self._pf_split_groups(self.NUM_KV_HEADS, ne if sharded else 1)
        # Row count register: this engine's share when sharded, the full runtime
        # length otherwise. Region B always uses the FULL length (gpr_seq_len).
        Mreg = self.PF_GPR_SHARD_ROWS if sharded else self.gpr_seq_len

        def pf_addr(ue, dst_reg, base, pitch):
            """dst_reg <- (base + PF_GPR_ROW_OFF*pitch) >> 3, the PBI word-address format."""
            assert pitch % 8 == 0, f"pitch {pitch} must be a multiple of 8 for word addressing"
            ue.generate_instruction_reg_mul_imm(dst_reg, self.PF_GPR_ROW_OFF,
                                                ue_35bit_addr_shifter(pitch))
            ue.generate_instruction_add_imm(dst_reg, ue_35bit_addr_shifter(base), dst_reg)

        def sh_mm(ue, K, N, A, proj, la, OUT, a_pitch, out_pitch, **kw):
            ka = {}
            if sharded:
                pf_addr(ue, self.PF_GPR_ADDR_A, A, a_pitch)
                pf_addr(ue, self.PF_GPR_ADDR_OUT, OUT, out_pitch)
                ka = dict(gpr_a_addr=self.PF_GPR_ADDR_A, gpr_out_addr=self.PF_GPR_ADDR_OUT)
            ue.matmat_mul_core(M=PM, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f'{proj}_data'],
                OUTPUT_DRAM_ADDR=OUT, is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=la[f'{proj}_scale'], gpr_M_reg=Mreg, **ka, **kw)

        def sh_rms(ue, A, OUT, gamma):
            ka = {}
            if sharded:
                pf_addr(ue, self.PF_GPR_ADDR_A, A, H * bpe)
                pf_addr(ue, self.PF_GPR_ADDR_OUT, OUT, H * bpe)
                ka = dict(gpr_a_addr=self.PF_GPR_ADDR_A, gpr_out_addr=self.PF_GPR_ADDR_OUT)
            ue.rms_norm_core_dram(M=PM, N=H, A_DRAM_ADDR=A, OUTPUT_DRAM_ADDR=OUT,
                GAMMA_DRAM_ADDR=gamma, gpr_M_reg=Mreg, **ka)

        def sh_elt(ue, a, b, out, N, mode):
            """Sharded eltwise. All THREE operands take the row offset: this is a flat
            row-wise op, so element i of b pairs with element i of a."""
            pitch = N * bpe
            pf_addr(ue, self.PF_GPR_ADDR_A, a, pitch)
            pf_addr(ue, self.PF_GPR_ADDR_B, b, pitch)
            pf_addr(ue, self.PF_GPR_ADDR_OUT, out, pitch)
            ue.eltwise_core_dram(M=PM, N=N, dram_a=a, dram_b=b, dram_out=out, mode=mode,
                gpr_M_reg=Mreg, gpr_a_addr=self.PF_GPR_ADDR_A,
                gpr_b_addr=self.PF_GPR_ADDR_B, gpr_out_addr=self.PF_GPR_ADDR_OUT)

        def strided_copy(ue, src, src_jump, dst, dst_jump, rows, width):
            # static [rows, width] copy: strided gather -> contiguous SRAM -> strided scatter
            ue.accelerator_memory_to_sram(src, 0x00000, rows * width,
                stride_bytes_per_chunk=width * bpe, stride_jump_bytes=src_jump)
            ue.sram_to_accelerator_memory(0x00000, dst, rows * width,
                stride_bytes_per_chunk=width * bpe, stride_jump_bytes=dst_jump)

        def duplicate_gqa_rows(ue, src_sram_addr, dst_dram_addr):
            # token-major duplication: dst row t*G+g = src row t (kv head broadcast across the group)
            row_bytes = D * bpe
            row_uram_words = row_bytes // (UE_VECTOR_SIZE * bpe)
            _, src_uram_addr = ue.sram_address_to_uram_address(src_sram_addr)
            ptr = ue.alloc_inst_ptr()
            ue.generate_instruction_pbi_init(dram_shared_addr=dst_dram_addr, dma_length=row_bytes,
                output_size=0, uram_length=0, uram_a_start_addr=src_uram_addr,
                uram_b_start_addr=src_uram_addr, uram_wb_addr=0, uram_dst_addr=0,
                fmax_context_addr=0, inst_pointer_idx=ptr)
            ue.loop_start(loop_cnt=PM)
            ue.loop_start(G)
            ue.sram_to_accelerator_memory(sram_address=0, accelerator_dram_address=row_bytes,
                element_size=D, inst_pointer_idx=ptr, memcpy_length_bytes=0)
            ue.loop_end()
            ue.generate_instruction_pbi_inc(dram_shared_addr=0, dma_length=0, output_size=0,
                uram_length=0, uram_a_start_addr=row_uram_words, uram_b_start_addr=row_uram_words,
                uram_wb_addr=0, uram_dst_addr=0, fmax_context_addr=0, inst_pointer_idx=ptr)
            ue.loop_end()
            ue.release_inst_ptr(ptr)

        def barrier():
            if sharded:
                sched.barrier()

        # ---- region A: residual + input norm + q/k/v (ROW-sharded) ----
        def emit_A(ue, layer_idx, la, h_in):
            if layer_idx == 0:
                sh_rms(ue, h_in, self.LAYER0_PRE_NORM_DRAM, la['ln1_gamma'])
            else:
                if sharded:
                    sh_elt(ue, self.LAYER0_RESIDUAL_DRAM, self.LAYER0_MLP_DOWN_DRAM, h_in,
                           H, UE_MODE.ELTWISE_ADD)
                else:
                    ue.eltwise_core_dram(M=PM, N=H, dram_a=self.LAYER0_RESIDUAL_DRAM,
                        dram_b=self.LAYER0_MLP_DOWN_DRAM, dram_out=h_in, mode=UE_MODE.ELTWISE_ADD,
                        gpr_M_reg=Mreg)
                sh_rms(ue, h_in, self.LAYER0_PRE_NORM_DRAM, la['ln1_gamma'])
            sh_mm(ue, H, H,  self.LAYER0_PRE_NORM_DRAM, 'q', la, self.LAYER0_Q_DRAM,
                  H * bpe, H * bpe)
            sh_mm(ue, H, KV, self.LAYER0_PRE_NORM_DRAM, 'k', la, self.LAYER0_K_PROJ_DRAM,
                  H * bpe, KV * bpe)
            sh_mm(ue, H, KV, self.LAYER0_PRE_NORM_DRAM, 'v', la, self.LAYER0_V_PROJ_DRAM,
                  H * bpe, KV * bpe)

        # ---- region B: KV cache staging + GQA attention (GROUP-sharded, FULL S) ----
        def emit_B(ue, eidx, layer_idx):
            my = groups[eidx]
            FQ = sched.per_engine_addr("pf_flash_q", eidx) if sharded else self.FLASH_Q_DRAM
            FK = sched.per_engine_addr("pf_flash_k", eidx) if sharded else self.FLASH_K_DRAM
            FV = sched.per_engine_addr("pf_flash_v", eidx) if sharded else self.FLASH_V_DRAM
            FO = sched.per_engine_addr("pf_flash_out", eidx) if sharded else self.FLASH_OUT_DRAM
            FS = (sched.per_engine_addr("pf_flash_scratch", eidx) if sharded
                  else self.FLASH_SCRATCH_DRAM)
            # K per kv-head -> decoder KV cache (contiguous [PM,D]) + RoPE
            for h in my:
                k_cache = self.LAYER0_K_DRAM + layer_idx * self.KV_LAYER_STRIDE + h * self.KV_HEAD_STRIDE
                strided_copy(ue, self.LAYER0_K_PROJ_DRAM + h * D * bpe, KV * bpe, k_cache, D * bpe, PM, D)
                ue.rope_hf_core_dram(M=PM, N=D, input_dram_addr=k_cache, output_dram_addr=k_cache,
                    cos_dram_addr=self.ROPE_PACKED_DRAM, sin_dram_addr=self.ROPE_PACKED_DRAM + D * bpe,
                    gpr_M_reg=self.gpr_seq_len)
            # V per kv-head -> decoder KV cache (no RoPE)
            for h in my:
                v_cache = self.LAYER0_V_DRAM + layer_idx * self.KV_LAYER_STRIDE + h * self.KV_HEAD_STRIDE
                strided_copy(ue, self.LAYER0_V_PROJ_DRAM + h * D * bpe, KV * bpe, v_cache, D * bpe, PM, D)
            # Per kv-group GQA: stack Q token-major, rope, duplicate K/V, one bucket flash, un-stack
            for kv_b in my:
                k_cache = self.LAYER0_K_DRAM + layer_idx * self.KV_LAYER_STRIDE + kv_b * self.KV_HEAD_STRIDE
                v_cache = self.LAYER0_V_DRAM + layer_idx * self.KV_LAYER_STRIDE + kv_b * self.KV_HEAD_STRIDE
                for g in range(G):
                    strided_copy(ue, self.LAYER0_Q_DRAM + (kv_b * G + g) * D * bpe, H * bpe,
                                 FQ + g * D * bpe, G * D * bpe, PM, D)
                ue.rope_hf_core_dram(M=PM * G, N=D, input_dram_addr=FQ,
                    output_dram_addr=FQ, cos_dram_addr=self.ROPE_PACKED_GQA_DRAM,
                    sin_dram_addr=self.ROPE_PACKED_GQA_DRAM + D * bpe, gpr_M_reg=self.gpr_q_seq_len)
                ue.accelerator_memory_to_sram(k_cache, 0x10000, PM * D)
                duplicate_gqa_rows(ue, 0x10000, FK)
                ue.accelerator_memory_to_sram(v_cache, 0x20000, PM * D)
                duplicate_gqa_rows(ue, 0x20000, FV)
                ue.unified_attention_core(batch=PM * G, aligned_seq_len=qmax, head_dim=D,
                    Q_DRAM_ADDR=FQ, K_DRAM_ADDR=FK, V_DRAM_ADDR=FV,
                    BIAS_DRAM_ADDR=self.FLASH_BIAS_DRAM,
                    OUTPUT_DRAM_ADDR=FO, SCRATCH_DRAM_ADDR=FS,
                    IDENTITY_DRAM_ADDR=self.identity_addr,
                    gpr_batch_reg=self.gpr_q_seq_len,
                    gpr_aligned_seq_len_reg=self.gpr_aligned_seq_len)
                for g in range(G):
                    strided_copy(ue, FO + g * D * bpe, G * D * bpe,
                                 self.LAYER0_ATTN_RESULT_DRAM + (kv_b * G + g) * D * bpe, H * bpe, PM, D)

        # ---- region C: o proj + residual + post-norm + MLP (ROW-sharded) ----
        def emit_C(ue, layer_idx, la, h_in, h_out):
            sh_mm(ue, H, H, self.LAYER0_ATTN_RESULT_DRAM, 'o', la, self.LAYER0_O_PROJ_DRAM,
                  H * bpe, H * bpe)
            if sharded:
                sh_elt(ue, h_in, self.LAYER0_O_PROJ_DRAM, self.LAYER0_RESIDUAL_DRAM,
                       H, UE_MODE.ELTWISE_ADD)
            else:
                ue.eltwise_core_dram(M=PM, N=H, dram_a=h_in, dram_b=self.LAYER0_O_PROJ_DRAM,
                    dram_out=self.LAYER0_RESIDUAL_DRAM, mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=Mreg)
            sh_rms(ue, self.LAYER0_RESIDUAL_DRAM, self.LAYER0_PRE_NORM_DRAM, la['ln2_gamma'])
            sh_mm(ue, H, I, self.LAYER0_PRE_NORM_DRAM, 'gate', la, self.LAYER0_MLP_GATE_DRAM,
                  H * bpe, I * bpe, silu_enable=True)
            sh_mm(ue, H, I, self.LAYER0_PRE_NORM_DRAM, 'up', la, self.LAYER0_MLP_UP_DRAM,
                  H * bpe, I * bpe)
            if sharded:
                sh_elt(ue, self.LAYER0_MLP_GATE_DRAM, self.LAYER0_MLP_UP_DRAM,
                       self.LAYER0_MLP_MULT_DRAM, I, UE_MODE.ELTWISE_MUL)
            else:
                eltwise_mul_core_dram(ue, size=PM * I,
                    A_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM, B_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM)
            sh_mm(ue, I, H, self.LAYER0_MLP_MULT_DRAM, 'down', la, self.LAYER0_MLP_DOWN_DRAM,
                  I * bpe, H * bpe)
            # MLP residual: materialize the last layer's output (the inner layers fuse it into the
            # next layer's input norm; the last one is written out explicitly so the postamble can
            # read it).
            if layer_idx == self.NUM_LAYERS - 1:
                if sharded:
                    sh_elt(ue, self.LAYER0_RESIDUAL_DRAM, self.LAYER0_MLP_DOWN_DRAM, h_out,
                           H, UE_MODE.ELTWISE_ADD)
                else:
                    eltwise_add_core_dram(ue, size=PM * H,
                        A_DRAM_ADDR=self.LAYER0_RESIDUAL_DRAM, B_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                        OUTPUT_DRAM_ADDR=h_out)

        self.start_capture()
        if sharded:
            sched.begin_program()
        prefill_addr = self.get_program_dram_addr()
        # Scratch ISA regs start past the fixed gpr regs (1..6) so rope/flash scratch don't clobber
        # them; when sharded they must also clear the five shard regs (7..11).
        _scratch_base = (self.PF_ISA_SCRATCH_BASE if sharded
                         else max(self._cfg["fixed_isa_regs"].values()) + 1)
        for ue in engines:
            ue._isa_reg_counter = _scratch_base
        # ENTRY RENDEZVOUS -- NOT optional, and the reason prefill sharding first
        # produced garbage. run_prefill launches the workers BEFORE the primary runs its
        # preamble, and that preamble is what writes LAYER0_INPUT_DRAM (the on-device
        # embed gather + vision merge). Without this barrier every worker charges into
        # layer 0 region A and normalizes embeddings THAT HAVE NOT BEEN WRITTEN YET.
        #
        # It hid well: give engine 0 the full row range and the bug vanishes, because
        # engine 0 -- delayed by running the preamble -- overwrites every row the worker
        # got wrong. Only a real disjoint split leaves the worker's garbage in place.
        # The encoder never needed this: its input arrives by host DMA before launch and
        # its primary-only patch embed is followed by a rendezvous.
        barrier()
        for layer_idx in range(self.NUM_LAYERS):
            la = self.lm_layer_addrs[layer_idx]
            h_in  = self.LAYER0_INPUT_DRAM if layer_idx % 2 == 0 else self.LAYER0_OUTPUT_DRAM
            h_out = self.LAYER0_OUTPUT_DRAM if layer_idx % 2 == 0 else self.LAYER0_INPUT_DRAM
            for ue in engines:
                emit_A(ue, layer_idx, la, h_in)
            barrier()
            for eidx, ue in enumerate(engines):
                emit_B(ue, eidx, layer_idx)
            barrier()
            for ue in engines:
                emit_C(ue, layer_idx, la, h_in, h_out)
            barrier()
            # alloc_isa_reg/release_isa_reg are a bump allocator on a shared counter.
            # If any core released more than it allocated, the counter would drift DOWN
            # into 7..11 and a later scratch allocation would silently overwrite this
            # engine's row offset -- finite, plausible, wrong output. Check per layer,
            # where the message still names the layer.
            if sharded:
                for _e, _u in enumerate(engines):
                    assert _u._isa_reg_counter >= self.PF_ISA_SCRATCH_BASE, (
                        f'engine {_e} ISA counter drifted to {_u._isa_reg_counter} at layer '
                        f'{layer_idx}, below the shard registers '
                        f'({self.PF_ISA_SCRATCH_BASE}); scratch would clobber the row offset')
        self.generate_instruction_halt()
        if sharded:
            # Halts + flushes every WORKER program, then reserves each worker a preamble
            # slot. run_prefill rewrites those preambles per run with this run's row split
            # and jumps into the worker program -- that indirection is what lets ONE
            # compiled program serve any sequence length.
            self._pf_worker_prog_addrs = sched.finalize()
            self._record_worker_prog_sizes("_pf_worker_prog_addrs", sched)
            self._pf_worker_preamble_addrs = []
            for _wi, w in enumerate(sched.workers):
                _a = w.get_program_dram_addr()
                w.allocate_program_dram(256)
                self._pf_worker_preamble_addrs.append(_a)
                # Worker programs ACCUMULATE across stages in one shared allocator
                # (encoder, then prefill). Overflowing the arena slice would write this
                # program over the NEXT worker's -- a desync, and a hang at the first
                # rendezvous rather than an error.
                _slice_end = (self._vis_worker_arena
                              + (_wi + 1) * self.VIS_WORKER_ARENA_BYTES)
                assert w.get_program_dram_addr() <= _slice_end, (
                    f'worker {_wi} programs reach 0x{w.get_program_dram_addr():X}, past its '
                    f'arena slice end 0x{_slice_end:X}; raise VIS_WORKER_ARENA_BYTES')
        self.stop_capture()
        self.write_captured_instructions_to_dram(prefill_addr)
        size_bytes = self.get_capture_instruction_size_bytes()
        # Cache the full program (gemma3 convention) so later runs load in ~1s instead of recompiling.
        # Under compile_all, the bytes are stashed for the unified programs.bin instead of a
        # separate prefill_program.bin.
        raw = bytearray()
        for inst in self.capture_buffer:
            raw.extend(inst.get_bytes())
        self._seg_prefill = (prefill_addr, bytes(raw))
        self.allocate_program_dram(size_bytes)
        self.clear_capture_buffer()
        self._prefill_addr = prefill_addr
        # Reserve preamble (embed/merge + reg prime + jump) and postamble (final-norm + LM head)
        # scratch regions; both are captured fresh per run.
        self._prefill_preamble_addr = self.get_program_dram_addr()
        self.allocate_program_dram(PM * 2 * 32 + IMAGE_SEQ_LEN * 2 * 32 + 256)
        self._prefill_postamble_addr = self.get_program_dram_addr()
        self.allocate_program_dram(4096)
        # Final hidden lands in the buffer matching the layer ping-pong parity.
        self._prefill_final_buf = self.LAYER0_INPUT_DRAM if self.NUM_LAYERS % 2 == 0 else self.LAYER0_OUTPUT_DRAM
        _extra = ""
        if sharded:
            _wl = [w.get_capture_instruction_size_bytes() for w in sched.workers]
            _extra = (f" + {len(_wl)} worker program(s) {_wl} bytes, engines={ne}, "
                      f"kv-groups per engine {[len(g) for g in groups]}")
        print(f"    Prefill compiled @0x{prefill_addr:X}: {size_bytes} bytes, "
              f"unified_attention_core, PM={PM}, qmax={qmax}{_extra}")

    def _restore_unified_addrs(self, meta: dict) -> None:
        """Set the program/preamble addresses the run_* methods read, from the unified-bin meta."""
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
                "[decoder-attn-path] cached programs.json lacks decoder-attention branch metadata; "
                "the artifact is stale/incompatible — delete smolvlm2_bin and rebuild")
        if bool(self._decoder_attention_use_pbi) is not False or self._decoder_attention_impl != "unified_attention_core":
            raise RuntimeError(
                "[decoder-attn-path] loaded artifact is not the required SmolVLM2 unified decoder-attention implementation")

    def _print_decoder_attn_path_banner(self, artifact_sha256: str) -> None:
        """Required runtime banner. Uses _original_print so it survives main()'s redirect_stdout."""
        _original_print(
            f"    [decoder-attn-path] model=smolvlm2 phase=lm_decode "
            f"shared_subroutine=no use_pbi=false implementation=unified_attention_core "
            f"artifact_sha256={artifact_sha256} "
            f"metadata_validated=yes"
        )

    def compile_all(self) -> dict:
        """Build (or load) the COMPLETE single program bin — encoder + decoder + prefill in ONE
        programs.bin. Mirrors qwen2.5_vl compile_all. The compile order (encoder, decoder,
        prefill) matches the validated 3-bin order, so every §7 JUMP_ABS bakes against the exact
        program-DRAM address it did before; the bin just bundles the three segments, each recorded with
        its absolute load address (see fpga_pbi_jump_target_bake). Cache key = layout signature."""
        bin_dir = os.path.join(self.script_dir, "smolvlm2_bin")
        os.makedirs(bin_dir, exist_ok=True)
        uni_bin, uni_meta, artifact_suffix = self._programs_paths(bin_dir)
        sig = {"prefill_max_seq_len": self.PREFILL_MAX_SEQ_LEN, "num_layers": self.NUM_LAYERS,
               "num_vis_layers": len(self.vis_layer_addrs),
               "encoder_ln": "dynamic_rows_shared_zeros_inv_n_v2",
               "vision_attention_impl": "unified_attention_core_dynamic",
               "decoder_attention_use_pbi": False,
               "decoder_attention_impl": "unified_attention_core",
               "decoder_attention_shared_subroutine": False,
               # A sharded program bakes one rendezvous per engine, so a set built for a
               # different count cannot be replayed: the missing workers never answer the
               # first FLAG_CHECK, which has no timeout, and the run hangs instead of
               # failing. The filename separates configurations; this makes a stale file
               # under the right name refuse too.
               "engines": [int(self.NUM_ENGINES), self._prefill_engines(),
                           int(self.DECODE_NUM_ENGINES)],
               **self._artifact_mode_meta()}

        # ---- cache hit: replay the encoder LN params, DMA each program to its stored abs addr ----
        # Sharded builds load too: the bin carries one segment per WORKER program
        # alongside the primary's, and the file is keyed by the engine triple, so what is
        # replayed always matches the configuration that produced it.
        if os.path.exists(uni_bin) and os.path.exists(uni_meta):
            with open(uni_meta) as f:
                meta = json.load(f)
            if all(meta.get(k) == v for k, v in sig.items()) and "segments" in meta:
                # Shared LayerNorm zeros and 1/N are part of the weights snapshot,
                # so load_snapshot already restored both operands.
                with open(uni_bin, "rb") as f:
                    raw = f.read()
                self._loaded_artifact_sha256 = hashlib.sha256(raw).hexdigest()
                for seg in meta["segments"]:
                    if seg.get("engine"):
                        continue               # worker program; written below
                    b = raw[seg["off"]:seg["off"] + seg["size"]]
                    self.dma_write(DMA_DEVICE_H2C, seg["addr"], b, len(b))
                self._restore_unified_addrs(meta)
                self._restore_worker_programs(meta, raw)
                self._print_decoder_attn_path_banner(self._loaded_artifact_sha256)
                print(f"  Loaded unified instruction bin ({len(raw)/1024/1024:.1f} MB, "
                      f"{len(meta['segments'])} segments)")
                return meta
            print(f"  programs{artifact_suffix or ''}.bin layout signature "
                  f"{dict((k, meta.get(k)) for k in sig)} ≠ current {sig} — rebuilding.")

        # ---- cache miss: compile all three in ONE session (encoder, decoder, prefill) ----
        self._unified_active = True
        self._seg_encoder = self._seg_prefill = self._seg_decoder = None
        try:
            enc_addr = (self.compile_encoder_sharded() if int(self.NUM_ENGINES) > 1
                        else self.compile_encoder())
            self.compile_decoder()
            self.compile_prefill()
        finally:
            self._unified_active = False
        end_addr = self.get_program_dram_addr()
        # The worker arenas live in the TENSOR region and are taken from the model's
        # own allocator, so the primary's continuing program growth cannot reach them.
        assert end_addr <= 0x100000000, f'program DRAM overflowed at 0x{end_addr:X}'
        segments, raw = [], bytearray()
        for name, seg in (("encoder", self._seg_encoder), ("decoder", self._seg_decoder),
                          ("prefill", self._seg_prefill)):
            if seg is None:
                continue
            addr, b = seg
            segments.append({"name": name, "addr": addr, "off": len(raw), "size": len(b)})
            raw.extend(b)
        _wsegs, _waddrs = self._collect_worker_segments(raw)
        segments.extend(_wsegs)
        meta = {**sig, "segments": segments, "end_addr": end_addr,
                "worker_prog_addrs": _waddrs,
                "pf_worker_preambles": list(getattr(self, "_pf_worker_preamble_addrs",
                                                    []) or []),
                "worker_prog_end": [w.get_program_dram_addr()
                                    for w in self._worker_pool()],
                "encoder_addr": enc_addr,
                "decoder_addr": self._decoder_program_addr,
                "decoder_preamble": self._decoder_preamble_addr,
                "prefill_addr": self._prefill_addr,
                "prefill_preamble": self._prefill_preamble_addr,
                "prefill_postamble": self._prefill_postamble_addr,
                "prefill_final_buf": self._prefill_final_buf}
        with open(uni_bin, "wb") as f:
            f.write(raw)
        with open(uni_meta, "w") as f:
            json.dump(meta, f)
        self._loaded_artifact_sha256 = hashlib.sha256(bytes(raw)).hexdigest()
        self._print_decoder_attn_path_banner(self._loaded_artifact_sha256)
        _nprim = sum(1 for s in segments if not s.get("engine"))
        print(f"  Complete instruction bin: {len(raw)/1024/1024:.1f} MB, "
              f"{len(segments)} segments ({_nprim} primary + {len(segments) - _nprim} "
              f"worker) → {uni_bin}")
        return meta

    def run_prefill(self, token_ids, has_image: bool = False, total_flops: int = None) -> int:
        """Run the seq-len-agnostic prefill: host prep (epsilon pad + block bias), a preamble that
        does on-device embed/merge + primes dynamic sequence GPRs + jumps into the cached prefill, then a
        postamble that does final-norm + LM head on the runtime last token. Returns argmax."""
        from user_dma_core import TYPE
        seq_len = len(token_ids)
        self.seq_len = seq_len
        self._prefill_token_ids = list(token_ids)  # seeds the repetition-penalty window in run_decoder
        H, D, G, bpe = self.HIDDEN_SIZE, self.HEAD_DIM, self.GROUP_SIZE, 2
        PM = self.PREFILL_MAX_SEQ_LEN
        assert seq_len <= PM, f"prompt {seq_len} exceeds PREFILL_MAX_SEQ_LEN={PM}"
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

        # 4. Preamble: on-device embed gather + vision merge + prime dynamic GPRs + jump into prefill
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
        # THE RUNTIME ROW SPLIT. gpr_seq_len above stays the FULL length (region B is
        # group-sharded and runs every row); these two carry THIS engine's row block,
        # which is why one captured program serves any sequence length.
        _pf_sched = self._pf_sched
        _pf_multi = _pf_sched is not None and _pf_sched.num_engines > 1
        if _pf_multi:
            _split = self._pf_split_rows(seq_len, _pf_sched.num_engines)
            self.generate_instruction_add_set(self.PF_GPR_SHARD_ROWS, _split[0][1])
            self.generate_instruction_add_set(self.PF_GPR_ROW_OFF, _split[0][0])
        self.generate_instruction_jump_abs(ue_35bit_addr_shifter(self._prefill_addr))
        self.stop_capture()
        self.write_captured_instructions_to_dram(self._prefill_preamble_addr)
        self.clear_capture_buffer()
        if _pf_multi:
            # One preamble per worker, rewritten every run: prime the full-length
            # registers region B needs, this worker's row block, then jump into its
            # compiled program.
            for _wi, _w in enumerate(_pf_sched.workers):
                _off, _rows = _split[_wi + 1]
                _w.clear_inst_id()
                _w.start_capture()
                _w.generate_instruction_add_set(self.gpr_seq_len, seq_len)
                _w.generate_instruction_add_set(self.gpr_q_seq_len, qS)
                _w.generate_instruction_add_set(self.gpr_aligned_seq_len, aligned_q)
                _w.generate_instruction_add_set(self.PF_GPR_SHARD_ROWS, _rows)
                _w.generate_instruction_add_set(self.PF_GPR_ROW_OFF, _off)
                _w.generate_instruction_jump_abs(
                    ue_35bit_addr_shifter(self._pf_worker_prog_addrs[_wi]))
                _w.stop_capture()
                _w.write_captured_instructions_to_dram(self._pf_worker_preamble_addrs[_wi])
                _w.clear_capture_buffer()
            if not self._vis_flags_precleared:
                _pf_sched.preclear_flags()
                self._vis_flags_precleared = True
            _pf_sched.start_workers(self._pf_worker_preamble_addrs)
        self.start_execute_from_dram(self._prefill_preamble_addr)
        self.wait_queue(180.0)
        if _pf_multi:
            for _w in _pf_sched.workers:
                _w.wait_queue(180.0)
        # Static prefill writes epsilon-derived KV rows through PREFILL_MAX.
        # Retain real tokens and clear every remaining context row before decode.
        self._zero_runtime_attention_state(
            seq_len=seq_len, bucket_len=aligned_q, preserve_kv_prefix=True
        )
        # 5. Postamble: final norm over ALL tokens (identical to the proven old prefill) + LM head on
        # the last real token. (RMSNorm is per-row, so M=S then index == M=1 on the last row; using
        # M=S removes the M=1 path as a variable.)
        last_off = (seq_len - 1) * H * bpe
        self.clear_inst_id()
        self.start_capture()
        self.rms_norm_core_dram(M=seq_len, N=H, A_DRAM_ADDR=self._prefill_final_buf,
            OUTPUT_DRAM_ADDR=self.FINAL_NORM_DRAM, GAMMA_DRAM_ADDR=self.final_norm_addr)
        last_norm = self.FINAL_NORM_DRAM + last_off
        # Prefill LM-head stays fused in both modes (see the decode LM-head note): quantized_matmat_core
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

    def _emit_decode_gate_up(self, la, lm_matmul, H, I, bpe):
        """gate + up + the silu*up multiply, optionally N-split over DECODE_NUM_ENGINES.

        WHY THIS BLOCK AND NOTHING ELSE. Decode is M=1, so every linear is a GEMV
        with no weight reuse: cost is bytes streamed from DRAM, and gate+up are
        2.62 of the 5.24 MB a layer reads -- half the traffic. They are also the
        only decode linears whose N divides cleanly: I=2560 is 40 blocks of 64, so
        2/4/8 engines split exactly even. q (N=960) and k/v (N=320) are 15 and 5
        blocks and split unevenly; o and down are K-fat and would need split_k plus
        host-sliced weights.

        AN N-SPLIT DOES NOT REDUCE TOTAL DRAM TRAFFIC. Each engine reads half of B,
        so the bytes moved across the pair are identical to one engine reading all
        of it. This is only a win if the current ceiling is per-engine issue rate
        or AXI port, not the memory controller. Measured single-engine decode is
        ~192 MB/token at ~12 tok/s = ~2.3 GB/s against a 256-bit/198 MHz AXI
        (~6.35 GB/s), i.e. ~36% of peak -- suggestive but not proof. That is what
        this exists to measure.

        WHY NO PER-ENGINE OUTPUT BUFFERS. At M=1 a column block of a [1, N] buffer
        IS a contiguous byte range, and matmat's writeback stride is the N it was
        given, so each engine writes cols dense elements at OUT + col_offset*bpe --
        directly into the shared buffer, no alloc_col_output, no gather. The silu*up
        multiply is elementwise, so it stays in lane too: gate, up and the multiply
        run with ZERO barriers between them. The region rejoins only for `down`,
        which reads the full [1, I] intermediate.

        Cost: 2 rendezvous per layer x 32 layers = 64 per TOKEN, against ~2.6 ms of
        work per layer. That is a far higher barrier rate relative to work than the
        encoder's 24 per image, and it is the main reason this may not pay off.
        """
        ne = int(self.DECODE_NUM_ENGINES)
        if ne <= 1:
            # Byte-identical to the historical stream.
            lm_matmul(1, H, I, self.LAYER0_PRE_NORM_DRAM, 'gate', la, self.LAYER0_MLP_GATE_DRAM, silu_enable=True)
            lm_matmul(1, H, I, self.LAYER0_PRE_NORM_DRAM, 'up',   la, self.LAYER0_MLP_UP_DRAM)
            self.accelerator_memory_to_sram(self.LAYER0_MLP_GATE_DRAM, 0x10000, I)
            self.accelerator_memory_to_sram(self.LAYER0_MLP_UP_DRAM,   0x90000, I)
            self.eltwise_mul_core(0x10000, 0x90000, 0x10000, I)
            self.sram_to_accelerator_memory(0x10000, self.LAYER0_MLP_MULT_DRAM, I)
            return

        from user_dma_core import TYPE
        deterministic = bool(getattr(self, "decode_matmat_mul_core_enable", False))

        # DIAGNOSTIC: SMOLVLM2_DECODE_BARRIER_ONLY=1 makes every engine emit the FULL
        # gate/up/mul instead of its column slice -- same inputs, same weights, same
        # addresses, so the concurrent writes are byte-identical and benign. The work is
        # therefore redundant, not divided: the run pays every rendezvous and gains
        # nothing. Timing it against --decode-engines 1 isolates pure barrier cost, which
        # is what separates "the DRAM controller is the ceiling" from "the barriers ate
        # the win" -- two conclusions with opposite implications for decode sharding.
        barrier_only = os.environ.get("SMOLVLM2_DECODE_BARRIER_ONLY") == "1"

        def _body(ctx):
            # quantized_matmat_core and the SRAM ops are not on SHARDED_OP_ALLOWLIST
            # (which names a 'quantized_matmat_mul_core' that does not exist in
            # user_dma_core -- the real names are quantized_matmat_core{,_legacy,
            # _dynamic}). Use the documented escape hatch rather than editing a
            # library pi05 also depends on.
            ue = ctx.unsafe_ue
            nc = I if barrier_only else ctx.cols
            off = 0 if barrier_only else ctx.col_offset * bpe
            for proj, dst in (('gate', self.LAYER0_MLP_GATE_DRAM),
                              ('up',   self.LAYER0_MLP_UP_DRAM)):
                act = {'silu_enable': True} if proj == 'gate' else {}
                # b_addr/scale_addr own the IF4 offsets: B is nibble-packed so the
                # slice is (n0*K)//2 bytes in, while the scale blob is one bf16 per
                # 64 elements of the flattened (N,K) weight -> ((n0*K)//64)*2. K=960
                # is 15 whole 64-blocks, which is what makes the scale stride linear.
                B = (la[f'{proj}_data'] if barrier_only
                     else ctx.b_addr(la[f'{proj}_data'], H, data_type=TYPE.IF4))
                SC = (la[f'{proj}_scale'] if barrier_only
                      else ctx.scale_addr(la[f'{proj}_scale'], H))
                if deterministic:
                    ue.matmat_mul_core(M=1, K=H, N=nc, A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                        B_DRAM_ADDR=B, OUTPUT_DRAM_ADDR=dst + off, is_B_quantized=True,
                        data_type=TYPE.IF4, SCALE_DRAM_ADDR=SC, **act)
                else:
                    ue.quantized_matmat_core(M=1, K=H, N=nc, A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                        B_DRAM_ADDR=B, OUTPUT_DRAM_ADDR=dst + off, SCALE_DRAM_ADDR=SC,
                        data_type=TYPE.IF4, **act)
            # Elementwise, so it stays in this engine's column lane. Each engine has
            # its own SRAM, so the fixed 0x10000/0x90000 staging addresses do not
            # collide across engines.
            ue.accelerator_memory_to_sram(self.LAYER0_MLP_GATE_DRAM + off, 0x10000, nc)
            ue.accelerator_memory_to_sram(self.LAYER0_MLP_UP_DRAM + off, 0x90000, nc)
            ue.eltwise_mul_core(0x10000, 0x90000, 0x10000, nc)
            ue.sram_to_accelerator_memory(0x10000, self.LAYER0_MLP_MULT_DRAM + off, nc)

        self._lm_sched.col_sharded_region(I, _body, join=True)

    def compile_decoder(self) -> None:
        """Compile single decoder program with runtime KV length via unified_attention_core."""
        from user_dma_core import TYPE
        H, D, I = self.HIDDEN_SIZE, self.HEAD_DIM, self.INTERMEDIATE_SIZE
        bpe = 2
        G = self.GROUP_SIZE
        def lm_matmul(M, K, N, A, proj, la, OUT, **kw):
            if bool(getattr(self, "decode_matmat_mul_core_enable", False)):
                self.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f'{proj}_data'],
                    OUTPUT_DRAM_ADDR=OUT, is_B_quantized=True, data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=la[f'{proj}_scale'], **kw)
            else:
                self.quantized_matmat_core(M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f'{proj}_data'],
                    OUTPUT_DRAM_ADDR=OUT, SCALE_DRAM_ADDR=la[f'{proj}_scale'], data_type=TYPE.IF4, **kw)

        # Guarantee counter starts at 1 regardless of prior compilations.
        self.reset_isa_reg_counter()
        # Reserve fixed ISA registers (1-6) so alloc_isa_reg() inside
        # unified_attention_core / matmat_mul_core never clobbers
        # V_CACHE_SIZE_REG(1), ROPE_SIZE_REG(2), TMP_REG(3),
        # gpr_seq_len(4), gpr_q_seq_len(5), gpr_aligned_seq_len(6) at runtime.
        _NUM_FIXED_REGS = 6
        for _ in range(_NUM_FIXED_REGS):
            self.alloc_isa_reg()

        global _SILENT_MODE
        _SILENT_MODE = True
        self.clear_inst_id()
        self.start_capture()
        ne_dec = int(self.DECODE_NUM_ENGINES)
        lm_sched = self._lm_make_scheduler(ne_dec) if ne_dec > 1 else None
        if lm_sched is not None:
            lm_sched.begin_program()
        for layer_idx in range(self.NUM_LAYERS):
            la = self.lm_layer_addrs[layer_idx]
            h_in  = self.LAYER0_INPUT_DRAM if layer_idx % 2 == 0 else self.LAYER0_OUTPUT_DRAM
            h_out = self.LAYER0_OUTPUT_DRAM if layer_idx % 2 == 0 else self.LAYER0_INPUT_DRAM

            # RMS norm (input_layernorm)
            self.rms_norm_core_dram(M=1, N=H, A_DRAM_ADDR=h_in,
                OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=la['ln1_gamma'])
            # Q/K/V projections
            lm_matmul(1, H, H,                     self.LAYER0_PRE_NORM_DRAM, 'q', la, self.LAYER0_Q_DRAM)
            lm_matmul(1, H, self.NUM_KV_HEADS * D, self.LAYER0_PRE_NORM_DRAM, 'k', la, self.LAYER0_K_PROJ_DRAM)
            lm_matmul(1, H, self.NUM_KV_HEADS * D, self.LAYER0_PRE_NORM_DRAM, 'v', la, self.LAYER0_V_PROJ_DRAM)

            # Position-based byte offsets are gpr_seq_len * stride. Following gemma3's decode
            # convention, recompute the offset register(s) immediately before each use rather
            # than once per layer, so no intervening op can leave a stale value in the offset
            # register between the compute site and the consuming DMA/RoPE.
            #   V_CACHE_SIZE_REG = gpr_seq_len * k_size   (byte offset into KV cache)
            #   ROPE_SIZE_REG    = gpr_seq_len * D * bpe   (byte offset into cos/sin tables)

            # Write V to KV cache at current position
            for h in range(self.NUM_KV_HEADS):
                v_base = self.LAYER0_V_DRAM + layer_idx * self.KV_LAYER_STRIDE + h * self.KV_HEAD_STRIDE
                self.accelerator_memory_to_sram(self.LAYER0_V_PROJ_DRAM + h * D * bpe, 0x10000, D)
                # Compute full destination address into TMP_REG immediately before the store.
                self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gpr_seq_len, ue_35bit_addr_shifter(self.k_size))
                self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(v_base), self.TMP_REG)
                self.sram_to_accelerator_memory(0x10000, 0, D, general_reg_src=self.TMP_REG)

            # RoPE K: d64 packed-table rope, written straight into the KV cache at the current
            # position. Same convention as prefill's rope_hf_core_dram_d64_pbi (packed [cos||sin],
            # sin first-half pre-negated, every URAM slice row-aligned). Table + output offsets are
            # derived from gpr_seq_len inside the helper via register-addressed DMAs.
            for h in range(self.NUM_KV_HEADS):
                k_base = self.LAYER0_K_DRAM + layer_idx * self.KV_LAYER_STRIDE + h * self.KV_HEAD_STRIDE
                self.rope_hf_core_decode_d64(N=D,
                    input_dram_addr=self.LAYER0_K_PROJ_DRAM + h * D * bpe,
                    output_dram_addr=k_base, output_pos_strided=True,
                    packed_table_addr=self.ROPE_PACKED_DRAM,
                    pos_reg=self.gpr_seq_len, tmp_reg=self.TMP_REG)

            # RoPE Q: same d64 rope, output to the fixed Q_PERM slot (attention reads from there).
            for h in range(self.NUM_HEADS):
                self.rope_hf_core_decode_d64(N=D,
                    input_dram_addr=self.LAYER0_Q_DRAM + h * D * bpe,
                    output_dram_addr=self.LAYER0_Q_PERM_DRAM + h * D * bpe, output_pos_strided=False,
                    packed_table_addr=self.ROPE_PACKED_DRAM,
                    pos_reg=self.gpr_seq_len, tmp_reg=self.TMP_REG)
            # GQA decode attention (§7): per kv-group, marshal K/V history + Q into the fixed FLASH
            # buffers, jump into the shared decoder-attention subroutine, copy the head output back.
            for kv_b in range(self.NUM_KV_HEADS):
                q_start = kv_b * G * D * bpe
                k_cache_base = self.LAYER0_K_DRAM + layer_idx * self.KV_LAYER_STRIDE + kv_b * self.KV_HEAD_STRIDE
                v_cache_base = self.LAYER0_V_DRAM + layer_idx * self.KV_LAYER_STRIDE + kv_b * self.KV_HEAD_STRIDE
                # Gather the valid K/V history into fixed FLASH_K/FLASH_V buffers.
                self._emit_pbi_scatter_per_token(
                    read_base=k_cache_base, read_stride_bytes=D * bpe,
                    write_specs=[(self.FLASH_K_DRAM, D * bpe)],
                    sram_byte_addr=0x50000, element_count=D,
                    gpr_seq_len=self.gpr_q_seq_len, template_seq_len=self.max_seq_len)
                self._emit_pbi_scatter_per_token(
                    read_base=v_cache_base, read_stride_bytes=D * bpe,
                    write_specs=[(self.FLASH_V_DRAM, D * bpe)],
                    sram_byte_addr=0x60000, element_count=D,
                    gpr_seq_len=self.gpr_q_seq_len, template_seq_len=self.max_seq_len)
                # Scatter the group's Q [G, D] into the fixed FLASH_Q base (drop the per-group offset).
                self.accelerator_memory_to_sram(self.LAYER0_Q_PERM_DRAM + q_start, 0x30000, G * D)
                self.sram_to_accelerator_memory(0x30000, self.FLASH_Q_DRAM, G * D)
                self.unified_attention_core(batch=G, aligned_seq_len=self.max_seq_len, head_dim=D,
                    Q_DRAM_ADDR=self.FLASH_Q_DRAM, K_DRAM_ADDR=self.FLASH_K_DRAM,
                    V_DRAM_ADDR=self.FLASH_V_DRAM, BIAS_DRAM_ADDR=self.DECODE_BIAS_DRAM,
                    OUTPUT_DRAM_ADDR=self.FLASH_OUT_DRAM, SCRATCH_DRAM_ADDR=self.FLASH_SCRATCH_DRAM,
                    IDENTITY_DRAM_ADDR=self.identity_addr,
                    gpr_aligned_seq_len_reg=self.gpr_aligned_seq_len)
                # Copy the group output [G, D] back to this group's columns in ATTN_OUT.
                self.accelerator_memory_to_sram(self.FLASH_OUT_DRAM, 0x40000, G * D)
                self.sram_to_accelerator_memory(0x40000, self.LAYER0_ATTN_OUT_DRAM + q_start, G * D)
            # O projection
            lm_matmul(1, H, H, self.LAYER0_ATTN_OUT_DRAM, 'o', la, self.LAYER0_O_PROJ_DRAM)
            # Post-attn residual + RMS norm
            rms_norm_core_dram_post_add(self, M=1, N=H,
                A_DRAM_ADDR=h_in, B_DRAM_ADDR=self.LAYER0_O_PROJ_DRAM,
                ADDOUTPUT_DRAM_ADDR=self.LAYER0_RESIDUAL_DRAM,
                NORMOUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                GAMMA_DRAM_ADDR=la['ln2_gamma'])
            # MLP
            self._emit_decode_gate_up(la, lm_matmul, H, I, bpe)
            lm_matmul(1, I, H, self.LAYER0_MLP_MULT_DRAM, 'down', la, self.LAYER0_MLP_DOWN_DRAM)
            # MLP residual
            self.accelerator_memory_to_sram(self.LAYER0_RESIDUAL_DRAM,  0x10000, H)
            self.accelerator_memory_to_sram(self.LAYER0_MLP_DOWN_DRAM,  0x90000, H)
            self.eltwise_add_core(0x10000, 0x90000, 0x10000, H)
            self.sram_to_accelerator_memory(0x10000, h_out, H)

        # Final norm + LM head
        final_buf = self.LAYER0_INPUT_DRAM if self.NUM_LAYERS % 2 == 0 else self.LAYER0_OUTPUT_DRAM
        self.rms_norm_core_dram(M=1, N=H, A_DRAM_ADDR=final_buf,
            OUTPUT_DRAM_ADDR=self.FINAL_NORM_DRAM, GAMMA_DRAM_ADDR=self.final_norm_addr)
        # The LM-head always wires PENALTY_BIAS_DRAM as its C term (zeros = pure greedy).
        lm_head_kwargs = {
            "C_DRAM_ADDR": self.PENALTY_BIAS_DRAM,
            "bias_mode": "broadcast_N",
        }
        # The LM-head GEMV stays on the fused quantized_matmat_core in BOTH modes — it is NOT gated by
        # --decode-matmat_mul_core-enable. quantized_matmat_core is the optimized M=1 GEMV for the
        # 49280-wide vocab with on-chip HW-argmax + write_back_disable (no logit readback). The
        # deterministic flag de-fuses the per-layer decoder/prefill linears; the validated deterministic
        # path keeps this fused LM-head (run-to-run repeatable on clean DRAM — see the run_from_bin
        # end-of-run zero_dram, which is what makes consecutive load-only runs reproducible).
        self.quantized_matmat_core(M=1, K=H, N=self.VOCAB_SIZE, A_DRAM_ADDR=self.FINAL_NORM_DRAM,
            B_DRAM_ADDR=self.lm_head_data, OUTPUT_DRAM_ADDR=self.LOGITS_DRAM,
            SCALE_DRAM_ADDR=self.lm_head_scale, data_type=TYPE.IF4,
            write_back_disable=True, **lm_head_kwargs)

        # Advance token position on-device so next decode step writes to the right KV slot
        self.generate_instruction_add_inc(self.gpr_seq_len)
        self.generate_instruction_halt()
        if lm_sched is not None:
            self._lm_worker_prog_addrs = lm_sched.finalize()
            self._record_worker_prog_sizes("_lm_worker_prog_addrs", lm_sched)
        self.stop_capture()
        for _ in range(_NUM_FIXED_REGS):
            self.release_isa_reg()
        _SILENT_MODE = False

        raw = bytearray()
        for inst in self.capture_buffer:
            raw.extend(inst.get_bytes())
        self.clear_capture_buffer()

        bin_dir = os.path.join(self.script_dir, "smolvlm2_bin")
        # Load into program DRAM for immediate use (same session)
        self._decoder_program_addr = self.get_program_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, self._decoder_program_addr, bytes(raw), len(raw))
        self.allocate_program_dram(len(raw))
        self._decoder_preamble_addr = self.get_program_dram_addr()
        self.allocate_program_dram(256)
        self._decoder_attention_use_pbi = False
        self._decoder_attention_impl = "unified_attention_core"
        self._decoder_attention_shared_subroutine = False
        if self._unified_active:
            self._seg_decoder = (self._decoder_program_addr, bytes(raw))
        else:
            with open(os.path.join(bin_dir, "decoder_program.bin"), "wb") as f:
                f.write(bytes(raw))
            with open(os.path.join(bin_dir, "decoder_program.json"), "w") as f:
                json.dump({
                    "decoder_program_size": len(raw),
                    "max_seq_len": self.max_seq_len,
                    "version": 2,
                    "decoder_attention_use_pbi": self._decoder_attention_use_pbi,
                    "decoder_attention_impl": self._decoder_attention_impl,
                    "decoder_attention_shared_subroutine": self._decoder_attention_shared_subroutine,
                }, f)
        print(
            f"    Decoder compiled: single program {len(raw)} bytes at 0x{self._decoder_program_addr:X}, "
            "unified_attention_core"
        )
        artifact_sha256 = hashlib.sha256(bytes(raw)).hexdigest()

    def dump_snapshot(self) -> None:
        """Dump params DRAM + all runtime address metadata to smolvlm2_bin/params.bin + params.json."""
        bin_dir = os.path.join(self.script_dir, "smolvlm2_bin")
        suffix = self._artifact_mode_suffix()
        bin_path = os.path.join(bin_dir, f"params{suffix}.bin")
        meta_path = os.path.join(bin_dir, f"params{suffix}.json")
        total = self.get_params_dram_usage()
        CHUNK = 1 * 1024 * 1024
        with open(bin_path, "wb") as f:
            offset = 0
            while offset < total:
                sz = min(CHUNK, total - offset)
                buf = bytearray(sz)
                self.dma_read(DMA_DEVICE_C2H, self._params_dram_base + offset, buf, sz)
                f.write(buf)
                offset += sz
        addr_attrs = [
            "embed_addr", "ROPE_COS_DRAM", "ROPE_SIN_DRAM", "PERMUTE_PARAMS_DRAM",
            "LAYER0_K_DRAM", "LAYER0_V_DRAM", "LAYER0_INPUT_DRAM", "LAYER0_OUTPUT_DRAM",
            "LAYER0_PRE_NORM_DRAM", "LAYER0_Q_DRAM", "LAYER0_K_PROJ_DRAM", "LAYER0_V_PROJ_DRAM",
            "LAYER0_Q_PERM_DRAM", "LAYER0_ATTN_OUT_DRAM", "LAYER0_ATTN_SCRATCH_DRAM",
            "LAYER0_ATTN_RESULT_DRAM", "LAYER0_O_PROJ_DRAM", "LAYER0_RESIDUAL_DRAM",
            "LAYER0_MLP_GATE_DRAM", "LAYER0_MLP_UP_DRAM", "LAYER0_MLP_MULT_DRAM",
            "LAYER0_MLP_DOWN_DRAM", "FINAL_NORM_DRAM", "LOGITS_DRAM", "PENALTY_BIAS_DRAM",
            "CAUSAL_MASK_DRAM", "DECODE_BIAS_DRAM",
            "vis_zeros_addr", "vis_inv_n_addr",
            "device_attn_clear_zero_addr", "device_attn_clear_zero_bytes",
            "VIS_PIXEL_IN_DRAM", "VIS_PATCH_PERM_DRAM", "VIS_PATCH_PROJ_DRAM",
            "VIS_IO_A_DRAM", "VIS_IO_B_DRAM", "VIS_LN_OUT_DRAM",
            "VIS_Q_DRAM", "VIS_K_DRAM", "VIS_V_DRAM",
            "VIS_Q_PERM_DRAM", "VIS_K_PERM_DRAM", "VIS_V_PERM_DRAM",
            "VIS_ATTN_OUT_DRAM", "VIS_ATTN_SCRATCH_DRAM", "VIS_ATTN_RESULT_DRAM",
            "VIS_O_PROJ_DRAM", "VIS_RESIDUAL_DRAM", "VIS_MLP_INTER_DRAM", "VIS_MLP_OUT_DRAM",
            "VIS_POST_LN_DRAM", "VIS_SHUFFLED_DRAM", "VIS_CONNECTOR_DRAM", "PERMUTE_TEMP_DRAM",
            # Prefill (gemma3-style) RoPE tables + stacked-GQA flash buffers
            "ROPE_PACKED_DRAM", "ROPE_PACKED_GQA_DRAM",
            "FLASH_Q_DRAM", "FLASH_K_DRAM", "FLASH_V_DRAM", "FLASH_OUT_DRAM",
            "FLASH_SCRATCH_DRAM", "FLASH_BIAS_DRAM", "VIS_ATTN_BIAS_DRAM",
        ]
        meta = {
            "params_size": total,
            "tensor_size": self.get_tensor_dram_usage(),
            "max_seq_len": self.max_seq_len,
            "k_size": self.k_size,
            "bytes_per_element": self.bytes_per_element,
            "KV_HEAD_STRIDE": self.KV_HEAD_STRIDE,
            "KV_LAYER_STRIDE": self.KV_LAYER_STRIDE,
            "vision_bf16": self.vision_bf16,
            "_num_vis_layers": len(self.vis_layer_addrs),
            "PREFILL_QMAX": self.PREFILL_QMAX,
            **self._artifact_mode_meta(),
            # LM weight addresses: run_prefill re-captures the decoder + LM postamble fresh
            # each run, so it needs the real per-layer/final/lm-head addresses (NOT just a count).
            "lm_layer_addrs": self.lm_layer_addrs,
            "final_norm_addr": self.final_norm_addr,
        }
        meta["lm_head_data"] = self.lm_head_data
        meta["lm_head_scale"] = self.lm_head_scale
        # Superset of addr_attrs, captured automatically (see _capture_tensor_attrs).
        meta["tensor_attrs"] = getattr(self, "_tensor_attrs", None) or {}
        for attr in addr_attrs:
            meta[attr] = getattr(self, attr)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        _original_print(f"  Snapshot dumped: {total / 1024**2:.1f} MB → {bin_path}")

    def load_snapshot(self) -> bool:
        """Load params DRAM from snapshot bin + restore all address metadata. Returns True if loaded."""
        bin_dir = os.path.join(self.script_dir, "smolvlm2_bin")
        suffix = self._artifact_mode_suffix()
        bin_path = os.path.join(bin_dir, f"params{suffix}.bin")
        meta_path = os.path.join(bin_dir, f"params{suffix}.json")
        if not os.path.exists(bin_path) or not os.path.exists(meta_path):
            return False
        with open(meta_path) as f:
            meta = json.load(f)
        try:
            self._validate_artifact_mode(meta, os.path.basename(meta_path))
        except RuntimeError as exc:
            _original_print(f"  Snapshot mode mismatch ({exc}) — recompiling.")
            return False
        # Self-heal stale snapshots: ones dumped before LM weight addresses were saved lack
        # these keys and would crash run_prefill. Force a recompile + fresh dump instead.
        if ("lm_layer_addrs" not in meta or "final_norm_addr" not in meta
                or "vis_inv_n_addr" not in meta or "tensor_attrs" not in meta):
            _original_print("  Snapshot stale (missing required model addresses) — recompiling.")
            return False
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
        # tensor_init's full published state first; the explicit meta keys below carry
        # the same values and simply win on any overlap.
        for key, val in (meta.get("tensor_attrs") or {}).items():
            setattr(self, key, val)
        for key, val in meta.items():
            if key not in ("params_size", "tensor_size", "tensor_attrs"):
                setattr(self, key, val)
        self.bytes_per_element = 2  # always bf16; not in snapshot, needed by attention PBI body
        # load_encoder/run_encoder only need len(vis_layer_addrs) (the per-layer weight addresses
        # are already baked into the precompiled encoder program in programs.bin). Restore a length-correct
        # placeholder list from the saved layer count so those len() calls work post-snapshot.
        self.vis_layer_addrs = [None] * self._num_vis_layers
        from transformers import AutoTokenizer
        model_dir = os.path.join(self.script_dir, "smolvlm2_bin", "SmolVLM2-500M-Video-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        _original_print(f"  Snapshot loaded: {total / 1024**2:.1f} MB")
        return True

    # --- Run ---
    def run_encoder(self, encoder_addr: int, pixel_values) -> None:
        """Run vision encoder. Output lands in VIS_CONNECTOR_DRAM."""
        pixels_bf16 = pixel_values.to(torch.bfloat16).contiguous().flatten()
        self.dma_to_accelerator_memory(self.VIS_PIXEL_IN_DRAM, pixels_bf16)
        # Encoder FLOPs: patch proj + pos add + N layers + post LN + connector
        VS, VH, VD, VN, VI = 1024, 768, 64, 12, 3072
        n_vis = len(self.vis_layer_addrs)
        attn_per_head = VS * VD + 2 * VS * VD * VS + VS * VS * 5 + 2 * VS * VS * VD
        per_layer = (7 * VS * VH + 3 * (2 * VS * VH * VH + VS * VH)
            + VN * attn_per_head + 2 * VS * VH * VH + VS * VH
            + 8 * VS * VH + 2 * VS * VH * VI + VS * VI
            + 2 * VS * VI * VH + VS * VH + VS * VH)
        enc_flops = (2 * VS * VH * VH + VS * VH + VS * VH
            + n_vis * per_layer + 7 * VS * VH
            + 2 * 64 * 12288 * self.HIDDEN_SIZE)
        sched = self._vis_sched
        multi = sched is not None and sched.num_engines > 1
        if multi:
            # Flags survive a killed run. Clear every engine's once per process
            # before the first rendezvous, or engine 0 sails through a stale SET.
            if not self._vis_flags_precleared:
                sched.preclear_flags()
                self._vis_flags_precleared = True
            sched.start_workers(self._vis_worker_prog_addrs)
        self.program_execute(encoder_addr, timeout=30.0, flops=enc_flops)
        if multi:
            # The primary retires its halt as soon as the last rendezvous clears; a
            # worker may still be draining its (write-free) margin NOPs. Wait rather
            # than assert-not-busy, which would be flaky.
            for w in sched.workers:
                w.wait_queue(60.0)

    def _structural_token_ids(self):
        """Token ids that must NEVER be repetition-penalized: punctuation, whitespace, and special
        tokens (image/fake/global/end-of-utterance). These 'glue' tokens recur in any text; penalizing
        them over a long generation is what starves a small model of grammar and yields word-salad.
        Computed once from the vocab + config special tokens, cached."""
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
        """Build the per-vocab additive repetition bias from windowed token frequency and DMA it to
        PENALTY_BIAS_DRAM (the decode LM-head's C term, broadcast_N). bias[t] = clamp(−alpha·count[t],
        min=−cap); structural tokens stay 0. The HW argmax of (logits + bias) returns the penalized
        token — no logit readback. One full-buffer DMA per step (incremental writes pay more in device
        open/close than the tiny transfer saves)."""
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
        # Short-window loop backstop: hard-ban (-1e9) any NON-structural token that fills >= pen_loop_thr
        # of the last pen_loop_recent generated tokens. The soft count penalty above cannot break a
        # 2-cycle (both tokens get an equal penalty) or a strong single-token run (the cap is intentionally
        # soft); this forces the HW argmax off a loop-dominating token toward EOS / new content.
        loop_recent = int(getattr(self, "pen_loop_recent", 24))
        loop_thr = int(getattr(self, "pen_loop_thr", 6))
        recent = prev_tokens[-loop_recent:]
        if len(recent) >= loop_recent:
            rc = torch.zeros(vocab, dtype=torch.float32)
            rt = torch.tensor(recent, dtype=torch.long)
            rc.index_add_(0, rt, torch.ones(rt.numel(), dtype=torch.float32))
            rc[self._structural_ids_tensor()] = 0.0
            bias[rc >= loop_thr] = -1e9
        # Soft length nudge toward EOS: past a soft token budget, give the end-of-utterance token a
        # growing positive bias so open-ended generations terminate cleanly instead of rambling to the
        # hard max. Inactive below the budget, so short/normal answers are unaffected.
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
        """Auto-regressive decode loop. Returns generated token IDs."""
        global _SILENT_MODE
        generated = []
        H = self.HIDDEN_SIZE
        bpe = 2
        embed_row_bytes = H * bpe
        decoder_program_addr = self._decoder_program_addr
        preamble_addr = self._decoder_preamble_addr
        self._runtime_attention_zero_decode_calls = 0

        # On-FPGA repetition penalty: track every token seen (prompt + the prefill seed token + decoded)
        # for the windowed bias. The decode LM-head always adds PENALTY_BIAS_DRAM as its C term, so zero
        # it up front (→ pure greedy until the greedy_until gate), then refresh per step past the gate.
        _fpga_penalty = bool(getattr(self, "penalty_enable", False))
        _greedy_until = int(getattr(self, "greedy_until", 0))
        self._generated_tokens = list(getattr(self, "_prefill_token_ids", [])) + [token_id]
        self.dma_to_accelerator_memory(self.PENALTY_BIAS_DRAM,
                                       torch.zeros(1, self.VOCAB_SIZE, dtype=torch.bfloat16))
        if _fpga_penalty:
            self._structural_ids_tensor()  # warm the structural-token cache off the first decode token

        # Live t/s counter (same as gemma3): pin the bottom terminal row as a status line
        # via an ANSI scroll region; tokens stream above it and the rate refreshes in place.
        # Only when stdout is a real TTY (skip when piped/redirected).
        import shutil
        timer = time.perf_counter()
        _use_status = sys.stdout.isatty()
        def _status_setup():
            rows = shutil.get_terminal_size().lines
            sys.stdout.write(f"\033[1;{rows - 1}r")    # scroll region = rows 1..rows-1
            sys.stdout.write(f"\033[{rows - 1};1H")    # park cursor at bottom of region
            sys.stdout.flush()
        def _status_update():
            rows = shutil.get_terminal_size().lines
            elapsed = time.perf_counter() - timer
            rate = len(generated) / elapsed if elapsed > 0 else 0.0
            sys.stdout.write("\0337")                   # save cursor
            sys.stdout.write(f"\033[{rows};1H\033[2K")   # bottom row, clear it
            sys.stdout.write(f" decoding… {len(generated)} tokens  (pos {self.seq_len}/{self.max_seq_len})  "
                             f"{elapsed:.1f}s  {rate:.1f} tok/s")
            sys.stdout.write("\0338")                   # restore cursor
            sys.stdout.flush()
        def _status_teardown():
            rows = shutil.get_terminal_size().lines
            sys.stdout.write("\033[r")                  # reset scroll region
            sys.stdout.write(f"\033[{rows};1H\033[2K")   # clear the status row
            sys.stdout.flush()
        if _use_status:
            _status_setup()
        # Emit the prefill seed token now (after scroll-region setup, so it isn't clobbered).
        _original_print(self.tokenizer.decode([token_id]), end="", flush=True)

        _lm_sched = self._lm_sched
        _lm_multi = _lm_sched is not None and _lm_sched.num_engines > 1
        if _lm_multi:
            # Flags survive a killed run, and with --lm-enable the encoder never ran
            # to clear them. Clear once before the first token's rendezvous.
            # Under _SILENT_MODE: preclear_flags captures and runs a 2-instruction
            # program per engine, and builtins.print is globally routed through
            # _quiet_print -- so without this the core's capture/DMA chatter lands in
            # the middle of the generated text.
            _SILENT_MODE = True
            _lm_sched.preclear_flags()
            _SILENT_MODE = False
        while len(generated) < max_new_tokens and self.seq_len < self.max_seq_len:
            _SILENT_MODE = True
            self.seq_len += 1
            aligned_kv = ((self.seq_len + 63) // 64) * 64

            # Decode bias is compact [GROUP_SIZE, aligned_kv]; unified_attention_core's full-matrix
            # bias row stride is the dynamic aligned_seq_len, not max_seq_len.
            bias = torch.full((self.GROUP_SIZE, aligned_kv), -1e38, dtype=torch.bfloat16)
            bias[:, :self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.DECODE_BIAS_DRAM, bias)

            # Host-side embed DMA (same as gemma3) — avoids on-device C2H before decoder
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

            # Refresh the per-vocab penalty bias (this step's LM-head C term) from the windowed token
            # frequency once past the greedy_until gate, so the HW argmax of (logits + bias) returns the
            # penalized token directly. Before the gate the bias stays zero → pure greedy.
            if _fpga_penalty and len(generated) >= _greedy_until:
                self._write_penalty_bias(self._generated_tokens)

            if _lm_multi:
                # The worker program halts at the end of every token, so it has to be
                # relaunched each step -- one register write per worker per token.
                _lm_sched.start_workers(self._lm_worker_prog_addrs)
            self.start_execute_from_dram(preamble_addr)
            self.wait_queue(30.0)
            if _lm_multi:
                for _w in _lm_sched.workers:
                    _w.wait_queue(30.0)

            # Greedy decode reads the HW argmax register, updated from the live LM-head output (no logit
            # readback).
            token_id = self.get_arg_max_index()

            generated.append(token_id)
            self._generated_tokens.append(token_id)
            _SILENT_MODE = False
            if token_id in _SMOLVLM2_CFG["model"]["stop_token_ids"]:
                if _use_status:
                    _status_teardown()
                _original_print(f"\nStop token {token_id} reached.")
                break
            _original_print(self.tokenizer.decode([token_id]), end="", flush=True)
            if _use_status:
                _status_update()
        else:
            # Loop ended without hitting an EOS stop token — report which cap stopped it.
            if _use_status:
                _status_teardown()
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
def main():
    import argparse

    parser = argparse.ArgumentParser(description="SmolVLM2-500M on accelerator (bf16 vision + q4 LM)")
    _d = _SMOLVLM2_CFG["defaults"]
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _default_image = os.path.join(_root, _d["image"])
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt. Default: 'Describe this image.' (VLM) or a text question (--lm-enable).")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to an image for VLM. Default: the bundled sample image. Ignored with --lm-enable.")
    parser.add_argument("--lm-enable", action="store_true",
                        help="Pure language-model (text-only) mode — skip the vision encoder. Default is VLM (vision).")
    parser.add_argument("--dev", type=str, default=_d["dev"], help="DMA device name")
    parser.add_argument("--device", type=str, default='kintex7', help='FPGA board profile (kintex7, rk, puzhi, bittware, bittware_256, alveo, efinix).')
    parser.add_argument("--max-seq", type=int, default=_d["max_seq"], help="Max sequence length")
    parser.add_argument("--max-decode-tokens", type=int, default=None,
                        help="Cap the number of generated decode tokens.")
    parser.add_argument(
        "--decode-matmat_mul_core-enable",
        action="store_true",
        help="Use matmat_mul_core for decode linear layers instead of quantized_matmat_core.",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="delete every cached artifact (params*.bin/json and programs*.bin/json, all "
             "engine configurations) before running, so this run rebuilds them. The "
             "downloaded checkpoint lives in the same directory and is left alone.")
    parser.add_argument(
        "--greedy-enable",
        action="store_true",
        help="Use pure greedy decoding. Default generation uses the on-FPGA repetition penalty.",
    )
    _MAX_ENGINES = SmolVLM2_UnifiedEngine.MAX_ENGINES
    def _engines_arg(value):
        """--engines N, 1..MAX_ENGINES. Whether the board HAS N engines is checked at
        compile time against HW_INFO -- see _vis_make_scheduler."""
        try:
            n = int(value)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(f"--engines takes an integer, got {value!r}")
        if not 1 <= n <= _MAX_ENGINES:
            raise argparse.ArgumentTypeError(
                f"--engines must be 1-{_MAX_ENGINES}, got {n}.")
        return n
    parser.add_argument(
        "--prefill-engines", type=_engines_arg, default=None, metavar="N",
        help="Row-shard the prefill LM across N engines. Defaults to --engines. Each "
             "layer runs row-sharded residual/RMSNorm/q/k/v, then kv-GROUP-sharded "
             "attention at full sequence length, then row-sharded o/MLP. The row split "
             "is a RUNTIME value primed into GPRs, so one compiled program still serves "
             "any sequence length. Must be <= --engines.",
    )
    parser.add_argument(
        "--decode-engines", type=_engines_arg, default=1, metavar="N",
        help="EXPERIMENT (default 1 = off). N-split the decode MLP's gate and up "
             "projections across N engines -- half the per-layer weight bytes, and "
             "the only decode linears whose N (2560) splits evenly. Decode is a "
             "weight-streaming GEMV, so this reduces bytes read PER ENGINE but not "
             "in TOTAL; it only wins if the ceiling is issue rate rather than the "
             "DRAM controller. Must be <= --engines (workers are shared).",
    )
    parser.add_argument(
        "--engines", type=_engines_arg, default=2, metavar="N",
        help="Row-shard the vision encoder's whole layer body -- LayerNorms, q/k/v, "
             "attention, o, residuals, fc1, fc2 -- across N engines (1-8, default 2). "
             "Only the two permutes and the M=64 connector stay on the primary. The "
             "1024 patch tokens are 16 blocks of 64, so 1/2/4/8 split evenly. N>1 "
             "always COMPILES and never reads or writes programs.bin, which holds "
             "only the primary's programs; use --engines 1 to build a bin.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, _SMOLVLM2_CFG["paths"]["hf_model_dir"])
    # Download HF model if needed (the only weight source — weight_init quantizes/loads from it,
    # producing the single params.bin snapshot; there are no intermediate weight bins).
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        print(f"Downloading {HF_MODEL_REPO} ...")
        snapshot_download(repo_id=HF_MODEL_REPO, local_dir=model_dir, local_dir_use_symlinks=False, ignore_patterns=["onnx/*"])

    # --- Hardware inference ---
    set_dma_device("efinix" if args.device == "efinix" else args.dev)
    global DMA_DEVICE_H2C, DMA_DEVICE_C2H, DMA_DEVICE_USER
    DMA_DEVICE_H2C = user_dma_core.DMA_DEVICE_H2C
    DMA_DEVICE_C2H = user_dma_core.DMA_DEVICE_C2H
    DMA_DEVICE_USER = user_dma_core.DMA_DEVICE_USER
    user_dma_core.configure_clock_from_hardware()
    print(f"FPGA profile: device={args.device}")
    print(user_dma_core.hardware_info_summary())
    global _SILENT_MODE
    _SILENT_MODE = True
    ue = SmolVLM2_UnifiedEngine(script_dir=script_dir)
    # Software-reset the engine, then clear DRAM to 0 before anything else.
    # SmolVLM2 has a read-before-write defect and depends on clean, zero-filled
    # DRAM. The harness now poisons DRAM (0xFF) before SmolVLM2 like every other
    # model, so the model must reset and zero it itself before loading weights /
    # running. Remove once the gap is fixed.
    ue.software_reset()
    ue.zero_dram()
    ue.decode_matmat_mul_core_enable = bool(args.decode_matmat_mul_core_enable)
    ue.NUM_ENGINES = int(args.engines)
    ue.DECODE_NUM_ENGINES = int(args.decode_engines)
    ue.PREFILL_NUM_ENGINES = (None if args.prefill_engines is None
                              else int(args.prefill_engines))
    ue.penalty_enable = not bool(args.greedy_enable)
    _original_print(f"decode_linear={ 'if4_matmat_mul_core' if ue.decode_matmat_mul_core_enable else 'quantized_matmat_core' }")
    _original_print(f"generation={ 'hardware_penalty' if ue.penalty_enable else 'greedy' }")
    _original_print(f"vision_encoder_engines={ue.NUM_ENGINES}"
                    + (" (full layer body row-sharded; compile-only, no bin)"
                       if ue.NUM_ENGINES > 1 else ""))
    _original_print(f"decode_engines={ue.DECODE_NUM_ENGINES}"
                    + (" (gate/up N-split)" if ue.DECODE_NUM_ENGINES > 1 else ""))
    _original_print(f"prefill_engines={ue._prefill_engines()}"
                    + (" (rows + kv-groups sharded)" if ue._prefill_engines() > 1 else ""))
    # Default is VLM (vision). The instruction bin is ALWAYS the full VLM bin (encoder+decoder+prefill) —
    # built once; this flag only decides whether the vision encoder actually RUNS this invocation.
    # --lm-enable (or --image none) selects pure LM-only text mode.
    lm_only = bool(args.lm_enable) or (args.image is not None
                                       and str(args.image).strip().lower() in ("none", ""))
    vision_on = not lm_only
    if vision_on and (args.image is None or str(args.image).strip().lower() in ("none", "")):
        args.image = _default_image   # VLM default with no --image → use the bundled sample image
    if args.prompt is None:           # mode-appropriate default prompt
        args.prompt = _d["lm_prompt"] if lm_only else _d["vlm_prompt"]
    if args.clean:
        clean_bins(os.path.join(script_dir, "smolvlm2_bin"))
    init_hang_prevention(ue)
    _artifact_suffix = ue._artifact_mode_suffix()
    # Load the snapshot (params.bin) only when the program bin for THIS engine
    # configuration also exists. NUM_ENGINES no longer gates it: the bin carries one
    # segment per worker program and is keyed by the (encoder, prefill, decode) triple.
    _bin_dir = os.path.join(script_dir, "smolvlm2_bin")
    _pbin, _pmeta, _ = ue._programs_paths(_bin_dir)
    _have_instr_bin = os.path.exists(_pbin) and os.path.exists(_pmeta)
    loaded_snapshot = _have_instr_bin and ue.load_snapshot()
    if not loaded_snapshot:
        ue.weight_init()
        ue.tensor_init(max_seq_len=args.max_seq)
        ue.dump_snapshot()
    has_image = vision_on
    token_ids = build_input_ids(ue.tokenizer, args.prompt, has_image=has_image)
    seq_len = len(token_ids)
    num_image_tokens = sum(t == IMAGE_TOKEN_ID for t in token_ids)
    num_text_tokens = seq_len - num_image_tokens
    _SILENT_MODE = False
    if has_image:
        image_path = os.path.abspath(args.image)
        _original_print(f"Image: {image_path}")
    _original_print(f"Prompt: {args.prompt!r} ({seq_len} tokens, image={'yes' if has_image else 'no'})")
    # --- Compile (or load from bin) ---
    _SILENT_MODE = True
    timer = time.perf_counter()
    bin_dir = os.path.join(script_dir, "smolvlm2_bin")
    S = ((seq_len + 63) // 64) * 64
    # ONE unified instruction bin (encoder + decoder + prefill). compile_all() builds it on the
    # first run and loads it (cache key = layout signature) on subsequent runs.
    _ub, _um, _ = ue._programs_paths(bin_dir)
    use_bin = os.path.exists(_ub) and os.path.exists(_um)
    import threading, io, contextlib
    _real_out = sys.stdout  # spinner writes here even while stdout is redirected
    stop_compile = threading.Event()
    def _compile_progress():
        while not stop_compile.wait(1.0):
            elapsed = time.perf_counter() - timer
            _original_print(f"\r  Compiling ({elapsed:.0f}s)", end="", flush=True, file=_real_out)
    if not use_bin:
        t = threading.Thread(target=_compile_progress, daemon=True)
        t.start()
    # Hard-silence the core's compile/capture chatter (M_chunk/URAM/Capture stopped/…),
    # which leaks past _SILENT_MODE. The live spinner above writes to _real_out, so it survives.
    with contextlib.redirect_stdout(io.StringIO()):
        uni_meta = ue.compile_all()
        enc_addr = uni_meta["encoder_addr"]
    stop_compile.set()
    _SILENT_MODE = False
    elapsed = time.perf_counter() - timer
    if use_bin:
        _original_print(f"  Loaded from bin in {elapsed:.2f}s")
    else:
        _original_print(f"\r  Compiling ({elapsed:.0f}s) done")
    # Required runtime banner — compile_all's own emission above is swallowed by redirect_stdout, so
    # re-print it here (outside the redirect) using the state compile_all just set on ue.
    ue._print_decoder_attn_path_banner(ue._loaded_artifact_sha256)
    # --- Vision encoder ---
    if has_image:
        timer = time.perf_counter()
        pixel_values = process_image(args.image)
        stop_enc = threading.Event()
        def _enc_progress():
            while not stop_enc.wait(1.0):
                _original_print(f"\r  Vision encoder ({time.perf_counter() - timer:.0f}s)", end="", flush=True)
        t_enc = threading.Thread(target=_enc_progress, daemon=True)
        t_enc.start()
        _SILENT_MODE = True
        ue.run_encoder(enc_addr, pixel_values)
        _SILENT_MODE = False
        stop_enc.set()
        t_enc.join()
        enc_time = time.perf_counter() - timer
        _original_print(f"\r  Vision encoder ({enc_time:.2f}s) done")
    # --- Prefill ---
    timer = time.perf_counter()
    padded_seq = ((seq_len + 63) // 64) * 64
    S = padded_seq
    H, D, I = ue.HIDDEN_SIZE, ue.HEAD_DIM, ue.INTERMEDIATE_SIZE
    KVH, QH, G = ue.NUM_KV_HEADS, ue.NUM_HEADS, ue.GROUP_SIZE
    KV = KVH * D
    attn_per_group = (D * S + G * (S * D + 2 * S * D * S + S * S + S * S * 5 + 2 * S * S * D))
    per_layer = (4 * S * H + 2 * S * H * H + 2 * 2 * S * H * KV
        + (QH + KVH) * S * D * 4 + KVH * attn_per_group
        + 2 * S * H * H + 5 * S * H + 2 * (2 * S * H * I) + S * I + 2 * S * I * H + S * H)
    prefill_flops = ue.NUM_LAYERS * per_layer + 4 * S * H + 2 * 1 * H * ue.VOCAB_SIZE
    stop_pf = threading.Event()
    def _pf_progress():
        while not stop_pf.wait(1.0):
            _original_print(f"\r  Prefill ({time.perf_counter() - timer:.0f}s)", end="", flush=True)
    t_pf = threading.Thread(target=_pf_progress, daemon=True)
    t_pf.start()
    _SILENT_MODE = True
    hw_token = ue.run_prefill(token_ids, has_image=has_image, total_flops=prefill_flops)
    _SILENT_MODE = False
    stop_pf.set()
    t_pf.join()
    prefill_time = time.perf_counter() - timer
    _original_print(f"\r  Prefill ({prefill_time:.0f}s) done")
    # --- Decode (on-device embed fused with decoder, single dispatch per token) ---
    _greedy_until = _d["vlm_greedy_until"] if vision_on else _d["lm_greedy_until"]
    ue.penalty_enable = not bool(args.greedy_enable)
    ue.greedy_until = _greedy_until
    max_new = args.max_decode_tokens if args.max_decode_tokens is not None else args.max_seq - seq_len
    _original_print(f"\nPrompt:   {args.prompt}")
    _original_print(f"Response: ", end="", flush=True)   # the generated answer streams right after this
    decode_timer = time.perf_counter()
    hw_tokens = ue.run_decoder(hw_token, max_new_tokens=max_new)
    decode_time = time.perf_counter() - decode_timer
    total_time = prefill_time + decode_time
    n_generated = len(hw_tokens)
    _original_print(f"\nDecoder done in {total_time:.2f} seconds, total {n_generated} tokens.")
    _original_print(
        f"SmolVLM2 test ends. prefill {round(seq_len / prefill_time, 2) if prefill_time > 0 else 0.0} tok/s, "
        f"decode {round(n_generated / decode_time, 2) if decode_time > 0 else 0.0} tok/s.")

    _SILENT_MODE = True
    ue.software_reset()

if __name__ == "__main__":
    main()
