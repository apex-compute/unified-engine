"""
Qwen3.5-2B LM-only inference & generation on FPGA.

All 24 layers (18 Gated DeltaNet + 6 full-attention) run end-to-end on the
accelerator.  HF is used only as a weight store and tokenizer.  The pipeline
mirrors the validated implementation in compare/compare_qwen3.5_2b.py.

Key architectural specifics (see notes_qwen3.5_2b.md for rationale):
  - Hybrid layer types: [0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22]
    are Gated DeltaNet; [3, 7, 11, 15, 19, 23] are gated full attention.
  - Qwen3_5RMSNorm uses (1+w) gamma; Qwen3_5RMSNormGated uses plain w.
  - Full-attn q_proj is DOUBLE-WIDE: first half = query, second half = gate.
    Attention output is multiplied by sigmoid(gate) before o_proj.
  - Partial rotary (first 64 of 256 head dims), theta = 10_000_000.
  - FP4_64 quantization for all large projections; BF16 kept for norms / SSM
    scalars (in_proj_b, in_proj_a, conv1d).
  - FPGA rms_norm_core has no eps — overridden with a padding trick so the
    numerics match HF.
  - Identity matrix used by bf16_transpose_core / flash_attention_core is
    cached globally; also one torch.eye(N) per N is cached for SiLU via
    identity-matmul.
  - Depthwise causal conv1d uses 4-tap shifted-eltwise (NOT Toeplitz).

Usage:
    python3 qwen3.5_2b_test.py                                    # LM-only, default seed
    python3 qwen3.5_2b_test.py --prompt "Once upon a time"
    python3 qwen3.5_2b_test.py --vision-enable                    # VLM, HF vision encoder (default)
    python3 qwen3.5_2b_test.py --vision-enable --vision-on-hardware   # VLM, FPGA vision encoder
    python3 qwen3.5_2b_test.py --image my.jpg --prompt "What is in this image?"

Generation uses persistent Gated-DeltaNet, convolution, and attention-cache
state. Prefill replays the compiled decoder once per prompt token; each decode
token then runs one incremental FPGA decoder step.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# user_dma_core lives at src/template/ — add that to path.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[2]))

import user_dma_core                                  # noqa: E402
# Match Gemma3's deployed FPGA clock period (5.62 ns ≈ 178 MHz).  The
# user_dma_core default is 3 ns (333 MHz), which makes
# `report_latency_in_us()` under-report HW time by ~1.87× and inflates
# our GFLOP/s figure by the same factor.  Set BEFORE constructing the
# engine so `self._clock_period_ns` picks it up.
user_dma_core.CLOCK_CYCLE_TIME_NS = 5.62

from user_dma_core import (                          # noqa: E402
    UnifiedEngine, DMA_DEVICE_H2C, DMA_DEVICE_C2H,
    UE_VECTOR_SIZE, TYPE,
    LALU_MODE, UE_MODE, URAM_SECTION, URAM_WRITE_SRC, BROADCAST_MODE,
    MEMCPY_TYPE, URAM_START_ADDR, URAM_NEAR_FULL_ELEMENTS,
    DRAM_ACTIVATION_ADDR,
    LALU_CLAMP_RELU_A, LALU_CLAMP_RELU_B,
    ue_35bit_addr_shifter, set_dma_device,
)

BF16 = 2
MODEL_PATH = "/srv/model_files/Qwen3.5-2B-ModelFiles/Qwen3.5-2B"
CONFIG_PATH = _THIS.parent / "qwen3.5_2b_config.json"


# ============================================================================
# FP4_64 quantization (E2M1, block_size=64)
# ============================================================================

_FP4_E2M1_TABLE = torch.tensor(
    [ 0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.bfloat16)


def quantize_fp4_64(tensor: torch.Tensor):
    """FP4 E2M1 quantization, 64-element block size.
    Returns (scales [n_blocks] bf16, packed_data [n_blocks*32] uint8)."""
    x = tensor.to(torch.bfloat16).cpu().flatten()
    n_blocks = int(np.ceil(x.numel() / 64))
    if x.numel() % 64 != 0:
        x = F.pad(x, (0, n_blocks * 64 - x.numel()))
    blocks = x.view(n_blocks, 64)
    fp4_max = torch.tensor(6.0, dtype=torch.bfloat16)
    scales = (blocks.abs().max(dim=1).values / fp4_max).clamp(min=1e-8).to(torch.bfloat16)
    scaled = (blocks / scales[:, None]).to(torch.bfloat16)
    codes = torch.argmin(torch.abs(scaled.unsqueeze(-1) - _FP4_E2M1_TABLE),
                         dim=-1).to(torch.uint8).numpy().flatten()
    if len(codes) % 2:
        codes = np.pad(codes, (0, 1))
    packed = (codes[0::2] & 0x0F) | ((codes[1::2] & 0x0F) << 4)
    return scales, torch.from_numpy(packed.astype(np.uint8))


def quantize_lm_head_fp4(embed_w: torch.Tensor, rows_per_chunk: int = 4096):
    """Quantize the (tied) embedding ``[vocab, H]`` to FP4_64 for use as the
    on-chip LM-head weight (B = ``[N=vocab, K=H]`` for ``quantized_matmat_core``).

    Done chunked-by-rows because ``quantize_fp4_64`` materializes a
    ``[n_blocks, 64, 16]`` tensor for the codebook argmin — for the full
    ~500 M-element table that is ~16 GB and OOMs.  ``H`` is a multiple of 64,
    so a row boundary is always a 64-block boundary ⇒ chunked quantization is
    bit-identical to quantizing the whole tensor at once.  Returns
    ``(scales_bf16, packed_u8)``, the same pair shape as ``quantize_fp4_64``."""
    assert embed_w.shape[1] % 64 == 0, "H must be a multiple of the FP4 block size (64)"
    s_list, p_list = [], []
    for i in range(0, embed_w.shape[0], rows_per_chunk):
        s, p = quantize_fp4_64(embed_w[i:i + rows_per_chunk])
        s_list.append(s)
        p_list.append(p)
    return torch.cat(s_list), torch.cat(p_list)


# ============================================================================
# Stdout noise filter (blanks out user_dma_core chatter during capture/exec)
# ============================================================================

class _NoiseFilterStdout:
    NOISE = (
        "M_chunk", "N_chunk", "While-loop body size",
        "Layer norm PBI outer loop body size",
        "URAM_A usage", "URAM_B usage", "Total Theoretical FLOPS",
        "Chunk size:", "Capture ", "Writing ", "Successfully wrote",
        "Capture buffer cleared", "init_unified_engine", "Unified Engine init",
        "FPGA DRAM read/write", "Dram read/write test",
        "Software reset complete", "address 0x", "HW version",
        "/dev/xdma0", "register access",
    )

    def __init__(self, dest):
        self.dest, self.buf = dest, ""

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if not any(tok in line for tok in self.NOISE):
                self.dest.write(line + "\n")

    def flush(self):
        self.dest.flush()


@contextlib.contextmanager
def _quiet():
    saved = sys.stdout
    sys.stdout = _NoiseFilterStdout(saved)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stdout = saved


# ============================================================================
# Qwen3_5_2b_UnifiedEngine — merges identity caching and eps-padded RMSNorm
# ============================================================================

class Qwen3_5_2b_UnifiedEngine(UnifiedEngine):
    """FPGA engine for Qwen3.5-2B LM.

    Overrides:
      - bf16_transpose_core / flash_attention_core_cached: reuse one cached
        8-KB identity slot instead of allocating per call.
      - dma_write: silently drop duplicate writes to the identity slot.
      - rms_norm_core_dram: pad the row with sqrt(N·eps) so the HW RSQRT gives
        1/sqrt(mean(x²) + eps) — HF-exact numerics without touching the HW API.
    """

    _IDENTITY_MAT_BYTES = UE_VECTOR_SIZE * UE_VECTOR_SIZE * 2   # 8192 B

    def __init__(self, *args, config_path: str = None, **kwargs):
        self._identity_dram_addr = None
        self._identity_dram_written = False
        self._eps_tail_cache: Dict[Tuple[int, float], Tuple[int, int]] = {}
        # Cache for recurrent_gated_delta_rule_core: each entry stores the
        # Instruction list emitted during the first (recording) call for a
        # given key, plus a list of scalar patch points so subsequent calls
        # can just mutate α/β/k scalar bits and splice the instructions into
        # the capture buffer — skipping ~37K Python broadcast_mul calls.
        self._rec_cache: Dict[tuple, dict] = {}
        # Cache for every other primitive emission: across decode steps every
        # `run_*` helper is called with byte-identical addresses/args (tensor
        # DRAM alloc is deterministic after reset, params DRAM is pre-written),
        # so the instructions they emit are byte-identical too.  First call
        # records the span; subsequent calls splice it.
        self._primitive_cache: Dict[tuple, list] = {}
        # Per-capture DRAM-write cache: avoids re-uploading byte-identical
        # instruction buffers to the FPGA.  Keyed by the _exec_captured call
        # index within a decode step.  `_current_capture_has_patches` is set
        # by _replay_recurrence and tells _exec_captured to re-write instead
        # of reusing the cached DRAM image.
        self._exec_cache: Dict[int, int] = {}
        # Pre-serialized byte-buffer cache for patched captures.  Captured on
        # first execution, then mutated in place per step (skipping the
        # Instruction.get_bytes() serialization loop — ~60 ms/capture).
        # Entry: {bytes: bytearray, scalar_byte_patches: [(byte_offset, kind,
        # t, h, i_or_-1), ...]}.  Populated by the recurrence replay path.
        self._patched_bytes_cache: Dict[int, dict] = {}
        self._exec_idx = 0
        self._current_capture_has_patches = False
        # Pending scalar-patch batch for the active capture: list of
        # (byte_offset, kind, t, h, i_or_-1, alpha_ref, beta_ref, k_ref).
        # Enqueued by _replay_recurrence and applied by _exec_captured.
        self._pending_scalar_patches: list = []
        # Accumulated HW latency (µs) across all captures of the current
        # prefill/decode step.  Reset by prefill() / decode_step(); read by
        # generate() to compute FPGA-accurate GFLOP/s.
        self._step_hw_latency_us = 0.0
        # Option B: single-hardware-trigger compile mode (Gemma3-compatible API).
        self._compile_mode: bool = False
        self._ISA_POS_K_REG    = 1   # reg 1: V_CACHE_SIZE_REG — pos_start * row_bytes
        self._ISA_TMP_REG      = 2   # reg 2: TMP_REG — destination of add_imm / overwrite source
        self._ISA_POS_ROPE_REG = 3   # reg 3: ROPE_SIZE_REG — pos_start * rot_dim * BF16
        self._decoder_prog_addr: int = 0
        self._decoder_prog_size: int = 0
        self._decoder_X_dram: int = 0
        self._decoder_final_norm_dram: int = 0
        # Full-4 GB DRAM layout (like gemma4): params 0x0–0xB0000000 = 2.75 GB.
        # The kernel default base (0x80000000) gives only 768 MB of params, which
        # the layer weights (~706 MB) + the on-chip FP4 LM head (~258 MB) = ~964 MB
        # overflow into tensor DRAM (0xB0000000) — corrupting the KV/S caches →
        # NaN.  Tensor/program bases stay at the kernel defaults (0xB0000000 /
        # 0xD0000000); the small unified bin lives at 0xD0000000.
        kwargs.setdefault("params_dram_base", 0x00000000)
        super().__init__(*args, **kwargs)
        with open(config_path or CONFIG_PATH) as f:
            self._cfg = json.load(f)
        fi = self._cfg["file_info"]
        self.num_layers       = fi["num_layers"]
        self.hidden_size      = fi["hidden_size"]
        self.mlp_dim          = fi["mlp_elements"]
        self.full_head_dim    = fi["full_attn_head_dim"]
        self.full_num_heads   = fi["full_attn_num_heads"]
        self.full_num_kv_heads= fi["full_attn_num_kv_heads"]
        self.full_rotary_dim  = fi["full_attn_rotary_dim"]
        self.lin_num_v_heads  = fi["linear_attn_num_v_heads"]
        self.lin_num_k_heads  = fi["linear_attn_num_k_heads"]
        self.lin_head_k_dim   = fi["linear_attn_head_k_dim"]
        self.lin_head_v_dim   = fi["linear_attn_head_v_dim"]
        self.lin_key_dim      = fi["linear_attn_key_dim"]
        self.lin_value_dim    = fi["linear_attn_value_dim"]
        self.lin_conv_dim     = fi["linear_attn_conv_dim"]
        mcfg = self._cfg["model"]
        self.linear_attn_layers = set(mcfg["linear_attn_layer_indices"])
        self.full_attn_layers   = set(mcfg["full_attn_layer_indices"])
        self.rope_theta = self._cfg["special"]["rope"]["theta_full"]

        # ── Optional on-FPGA repetition penalty ──────────────────────────────
        # The LM head matmul always carries the PENALTY_BIAS_DRAM C term + HW
        # argmax (write_back_disable). The default zero bias is bit-identical
        # greedy. Q35_REPETITION_PENALTY=1 enables the optional host-refreshed
        # bias; Q35_PURE_GREEDY remains an explicit override.
        _envb = lambda k: os.environ.get(k, "0") not in ("0", "", "false", "False", "no")
        self.fpga_penalty  = (_envb("Q35_REPETITION_PENALTY")
                              and not _envb("Q35_PURE_GREEDY"))
        self.pen_alpha     = float(os.environ.get("PEN_ALPHA", "3.0"))   # bias = -alpha*count
        self.pen_cap       = float(os.environ.get("PEN_CAP", "20.0"))    # clamp(min=-cap)
        self.rep_window    = int(os.environ.get("REP_WINDOW", "256"))    # freq over last N decoded
        self.greedy_until  = int(os.environ.get("GREEDY_UNTIL", "8"))    # pure greedy first N tokens
        self.pen_ban_count = int(os.environ.get("PEN_BAN_COUNT", "0"))   # ≥N times in window ⇒ hard ban (0=off; backstop)
        self.pen_loop_run  = int(os.environ.get("PEN_LOOP_RUN", "4"))    # ≥N in a row ⇒ hard ban (-1e9)
        self.tokenizer = None       # set by generate()/run_decoder for the structural scan
        self._struct_ids_cache = None
        self._struct_ids_tensor_cache = None

    def _preallocate_identity_matrix(self) -> None:
        if self._identity_dram_addr is not None:
            return
        self._identity_dram_addr = super().allocate_params_dram(self._IDENTITY_MAT_BYTES)
        eye = torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16)
        super().dma_write(DMA_DEVICE_H2C, self._identity_dram_addr,
                          eye, self._IDENTITY_MAT_BYTES)
        self._identity_dram_written = True

    # ── On-FPGA repetition penalty helpers ───────────────────────────────────
    def _structural_token_ids(self) -> set:
        """Token ids that must NEVER be repetition-penalized: punctuation,
        whitespace, newline, and special tokens.  Scanned once from the tokenizer
        vocab and cached.  These 'glue' tokens recur in any text; penalizing them
        over a long generation starves a small model of grammar (word-salad)."""
        if self._struct_ids_cache is not None:
            return self._struct_ids_cache
        import string
        allowed = set(string.punctuation) | set(string.whitespace) | set("—–’‘“”…·•‹›«»¡¿")
        ids = set(int(i) for i in (getattr(self.tokenizer, "all_special_ids", []) or []))
        for i in range(self.VOCAB):
            s = self.tokenizer.decode([i]).strip()
            if s == "" or all(ch in allowed for ch in s):
                ids.add(i)
        self._struct_ids_cache = ids
        return ids

    def _structural_ids_tensor(self) -> torch.Tensor:
        """1-D LongTensor of the structural ids (cached) for vectorized exemption."""
        if self._struct_ids_tensor_cache is None:
            self._struct_ids_tensor_cache = torch.tensor(
                sorted(self._structural_token_ids()), dtype=torch.long)
        return self._struct_ids_tensor_cache

    def _write_penalty_bias(self, prev_tokens) -> None:
        """Build the per-vocab additive repetition-penalty bias from windowed
        token frequency and DMA it to PENALTY_BIAS_DRAM (the LM-head matmul's C
        term, bias_mode="broadcast_N"): bias[t] = clamp(-alpha*count[t], min=-cap);
        structural tokens stay 0.  The HW argmax of (logits + bias) then returns
        the penalized token id — no logit readback.  A token emitted
        >= pen_loop_run times IN A ROW gets a hard ban (-1e9) that OVERRIDES the
        structural exemption, so the argmax is forced off a stuck loop the
        frequency penalty alone can't escape (e.g. a punctuation token).  One
        full-buffer DMA per step; all-zero = no penalty = bit-identical greedy."""
        vocab = self.VOCAB
        alpha, cap, W = float(self.pen_alpha), float(self.pen_cap), int(self.rep_window)
        window = prev_tokens[-W:] if W > 0 else list(prev_tokens)
        count = torch.zeros(vocab, dtype=torch.float32)
        if window:
            win = torch.tensor(window, dtype=torch.long)
            count.index_add_(0, win, torch.ones(win.numel(), dtype=torch.float32))
            count[self._structural_ids_tensor()] = 0.0          # never penalize glue tokens
        bias = (-alpha * count).clamp(min=-cap)
        # Windowed count ban: any non-structural token seen >= pen_ban_count times
        # in the window is banned outright (-1e9), even if not consecutive — this
        # breaks loops that intersperse EXEMPT structural tokens (e.g.
        # "* 1876 \n * 1876"), which the gentle frequency penalty and the
        # consecutive-run ban both miss.
        B = int(self.pen_ban_count)
        if B > 0:
            bias[count >= B] = -1e9
        # Consecutive-run ban: a token emitted >= pen_loop_run times in a row.
        R = int(self.pen_loop_run)
        if R > 0 and len(prev_tokens) >= R:
            last = int(prev_tokens[-1])
            if all(int(prev_tokens[-1 - j]) == last for j in range(R)):
                bias[last] = -1e9
        self.dma_write(DMA_DEVICE_H2C, self.PENALTY_BIAS_DRAM,
                       _bf(bias.view(1, vocab)), vocab * BF16)

    def bf16_transpose_core(self, M, N, INPUT_DRAM_ADDR, OUTPUT_DRAM_ADDR, **kwargs):
        saved = self._next_params_dram_addr
        self._next_params_dram_addr = self._identity_dram_addr
        super().bf16_transpose_core(M=M, N=N,
                                    INPUT_DRAM_ADDR=INPUT_DRAM_ADDR,
                                    OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                                    **kwargs)
        self._next_params_dram_addr = saved

    def flash_attention_core_cached(self, **kwargs):
        kwargs.pop("ATTN_P_DRAM_ADDR", None)
        kwargs.pop("gpr_bucket_idx", None)
        kwargs.pop("num_buckets", None)
        kwargs.pop("gpr_ret_id", None)
        seq_len = int(kwargs.pop("seq_len"))
        head_dim = int(kwargs["head_dim"])
        kwargs.setdefault("IDENTITY_DRAM_ADDR", self._identity_dram_addr)
        return self.unified_attention_core(
            batch=seq_len,
            aligned_seq_len=seq_len,
            head_dim=head_dim,
            Q_DRAM_ADDR=kwargs["Q_DRAM_ADDR"],
            K_DRAM_ADDR=kwargs["K_DRAM_ADDR"],
            V_DRAM_ADDR=kwargs["V_DRAM_ADDR"],
            BIAS_DRAM_ADDR=kwargs["BIAS_DRAM_ADDR"],
            OUTPUT_DRAM_ADDR=kwargs["OUTPUT_DRAM_ADDR"],
            SCRATCH_DRAM_ADDR=kwargs["SCRATCH_DRAM_ADDR"],
            IDENTITY_DRAM_ADDR=kwargs["IDENTITY_DRAM_ADDR"],
        )

    def dma_write(self, device, addr, data, size):
        if (self._identity_dram_written
                and self._identity_dram_addr is not None
                and addr == self._identity_dram_addr
                and size == self._IDENTITY_MAT_BYTES):
            return size
        return super().dma_write(device, addr, data, size)

    def wait_queue(self, timeout_seconds: float = 5.0) -> None:
        """Busy-wait override.  The base class sleeps 10 ms between polls,
        and even an exponential backoff leaves ~10 ms of slack per capture
        when the FPGA completes mid-quantum.  At 60+ captures per decode
        step that's ~700 ms of pure sleep on the critical path.  Busy-
        checking burns one CPU core but is_queue_busy() is a single PCIe
        register read (~1 µs), so we poll at near-zero slack overhead."""
        start = time.time()
        while self.is_queue_busy():
            if time.time() - start >= timeout_seconds:
                print(f"wait_queue timed out after {timeout_seconds:.1f}s")
                return

    # ---- Compile-mode capture overrides (Option B) -------------------------
    def start_capture(self):
        if self._compile_mode:
            return  # global capture already open; don't reset the buffer
        super().start_capture()

    def stop_capture(self):
        if self._compile_mode:
            return  # keep accumulating; closed explicitly in compile_decoder
        super().stop_capture()

    def isa_add_set_core(self, reg: int, value: int) -> None:
        """Compile and execute a mini-program that sets ISA register `reg`
        to `value`.  The register persists until the next set call, so call
        once per decode step before `start_execute_from_dram`.
        Only valid when NOT in compile mode (_compile_mode must be False)."""
        prog_addr = self.get_program_dram_addr()
        self.clear_capture_buffer()
        super().start_capture()
        self.generate_instruction_add_set(reg, value)
        self.generate_instruction_halt()
        super().stop_capture()
        self.write_captured_instructions_to_dram(prog_addr)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        self.start_execute_from_dram(prog_addr)
        self.wait_queue(2.0)
        self.clear_capture_buffer()

    # ---- RMSNorm with eps via padding trick --------------------------------
    def _build_eps_tail(self, N: int, eps: float) -> Tuple[int, int]:
        key = (N, eps)
        if key in self._eps_tail_cache:
            return self._eps_tail_cache[key]
        row_size = (N + 1 + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
        tail_len = row_size * UE_VECTOR_SIZE - N
        tail = torch.zeros(tail_len, dtype=torch.float32)
        tail[0] = math.sqrt(N * eps)
        tail_bf = tail.to(torch.bfloat16).contiguous().cpu()
        addr = super().allocate_params_dram(tail_len * 2)
        super().dma_write(DMA_DEVICE_H2C, addr, tail_bf, tail_len * 2)
        self._eps_tail_cache[key] = (addr, tail_len)
        return self._eps_tail_cache[key]

    def rms_norm_core_dram(self, M: int, N: int,
                           A_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                           GAMMA_DRAM_ADDR: int, eps: float = 1e-6) -> None:
        bpe = 2
        tail_addr, tail_len = self._build_eps_tail(N, eps)
        N_pad = N + 1
        padded_row_size = (N_pad + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
        vector_sram = 0x00000
        gamma_sram  = 0x80000

        self.accelerator_memory_to_sram(GAMMA_DRAM_ADDR, gamma_sram, N)
        # Hoist the tail-pad load out of the per-row loop: the tail is the
        # same constant [sqrt(N*eps), 0, 0, ...] for every row of this call,
        # and the per-row DMA only writes positions 0..N-1 so the pad at
        # positions N..N+tail_len-1 is preserved across iterations.
        self.accelerator_memory_to_sram(
            tail_addr, vector_sram + N * bpe, tail_len)
        for i in range(M):
            self.accelerator_memory_to_sram(
                A_DRAM_ADDR + i * N * bpe, vector_sram, N)
            vu_type, vu_addr = self.sram_address_to_uram_address(vector_sram)
            self.ue_arithmetic_op(
                0, 0, 1, 0, 0,
                LALU_MODE.MODE_RSQRT.value,
                self.float_to_bf19(float(math.sqrt(N))),
                vu_type, 0, 0,
                URAM_WRITE_SRC.URAM_WB_DISABLE.value,
                UE_MODE.RMS, 0, vu_addr, 0, padded_row_size, 0, 0, 0,
            )
            self.start_queue_broadcast(
                UE_MODE.MUL_BROADCAST, BROADCAST_MODE.LALU_RESULT,
                vector_sram, vector_sram, N)
            self.eltwise_mul_core(vector_sram, gamma_sram, vector_sram, N)
            self.sram_to_accelerator_memory(
                vector_sram, OUTPUT_DRAM_ADDR + i * N * bpe, N)

    # ---- Cached recurrence: emit ONCE, patch scalars per step --------------
    def recurrent_gated_delta_rule_core(self, T, num_heads, Dk, Dv,
                                        Q_DRAM_ADDR, K_DRAM_ADDR, V_DRAM_ADDR,
                                        S_DRAM_ADDR, OUT_DRAM_ADDR,
                                        SCRATCH_DRAM_ADDR,
                                        alpha_host, beta_host, k_host,
                                        alpha_on_fpga: bool = False,
                                        beta_on_fpga: bool = False,
                                        k_on_fpga: bool = False):
        """Drop-in replacement for UnifiedEngine.recurrent_gated_delta_rule_core.

        For T=1 (decode hot path), α/β come from DRAM via RMS+RSQRT loader
        (host uploads y=1/s; HW computes 1/|y|=s → LALU; broadcast_mul reads
        LALU_RESULT / LALU_RESULT_NEGATE).  k[i] stays baked.
        For T>1 (prefill, one-shot), everything stays baked — no need to
        pay the per-(t,h) RMS+RSQRT overhead for a throwaway emission.

        alpha_on_fpga=True: α values were written to alpha_dram by the FPGA
        pipeline above; skip the host α upload.
        beta_on_fpga=True: β values were written to beta_dram by the FPGA
        pipeline above; skip the host β upload.
        k_on_fpga=True: k values were written to k_pad_dram by the FPGA
        pipeline above; skip the host k upload.
        """
        key = (T, num_heads, Dk, Dv,
               Q_DRAM_ADDR, K_DRAM_ADDR, V_DRAM_ADDR,
               S_DRAM_ADDR, OUT_DRAM_ADDR, SCRATCH_DRAM_ADDR)
        entry = self._rec_cache.get(key)
        if entry is None:
            start_in_capture = self.capture_count
            self._rec_cache[key] = self._record_recurrence(
                T, num_heads, Dk, Dv,
                Q_DRAM_ADDR, K_DRAM_ADDR, V_DRAM_ADDR,
                S_DRAM_ADDR, OUT_DRAM_ADDR, SCRATCH_DRAM_ADDR,
                alpha_host, beta_host, k_host)
            if self._rec_cache[key].get('runtime_ab'):
                if alpha_on_fpga and beta_on_fpga:
                    pass  # both α and β written by FPGA pipelines
                elif alpha_on_fpga:
                    self._upload_runtime_beta_only(self._rec_cache[key], beta_host)
                else:
                    self._upload_runtime_ab(self._rec_cache[key],
                                            alpha_host, beta_host)
            if self._rec_cache[key].get('runtime_k') and not k_on_fpga:
                self._upload_runtime_k(self._rec_cache[key], k_host)
            patches = self._rec_cache[key]['patches']
            if patches:
                self._pending_scalar_patches.append(
                    ('rec', start_in_capture, patches,
                     alpha_host, beta_host, k_host))
                self._current_capture_has_patches = True
            return
        self._replay_recurrence(entry, alpha_host, beta_host, k_host,
                                alpha_on_fpga=alpha_on_fpga,
                                beta_on_fpga=beta_on_fpga,
                                k_on_fpga=k_on_fpga)

    def _upload_runtime_ab(self, entry, alpha_host, beta_host) -> None:
        """Compute y = 1/s bf16-padded into [T, num_heads, 64] blocks (only
        position 0 populated) and DMA-upload to the entry's α/β DRAM slots."""
        T = entry['T']; H = entry['num_heads']
        ab_bytes = T * H * UE_VECTOR_SIZE * 2
        alpha_np = alpha_host.detach().float().cpu().numpy()
        beta_np  = beta_host.detach().float().cpu().numpy()
        eps = 1e-6
        y_alpha = (1.0 / np.clip(alpha_np, eps, None)).astype(np.float32)
        y_beta  = (1.0 / np.clip(beta_np,  eps, None)).astype(np.float32)
        pad_a = np.zeros((T, H, UE_VECTOR_SIZE), dtype=np.float32)
        pad_b = np.zeros((T, H, UE_VECTOR_SIZE), dtype=np.float32)
        pad_a[:, :, 0] = y_alpha
        pad_b[:, :, 0] = y_beta
        pad_a_bf16 = torch.from_numpy(pad_a).to(torch.bfloat16).contiguous()
        pad_b_bf16 = torch.from_numpy(pad_b).to(torch.bfloat16).contiguous()
        self.dma_write(DMA_DEVICE_H2C, entry['alpha_dram'], pad_a_bf16, ab_bytes)
        self.dma_write(DMA_DEVICE_H2C, entry['beta_dram'],  pad_b_bf16, ab_bytes)

    def _upload_runtime_beta_only(self, entry, beta_host) -> None:
        """Upload only β to its DRAM slots (α already written by FPGA pipeline)."""
        T = entry['T']; H = entry['num_heads']
        ab_bytes = T * H * UE_VECTOR_SIZE * 2
        beta_np = beta_host.detach().float().cpu().numpy()
        eps = 1e-6
        y_beta = (1.0 / np.clip(beta_np, eps, None)).astype(np.float32)
        pad_b = np.zeros((T, H, UE_VECTOR_SIZE), dtype=np.float32)
        pad_b[:, :, 0] = y_beta
        pad_b_bf16 = torch.from_numpy(pad_b).to(torch.bfloat16).contiguous()
        self.dma_write(DMA_DEVICE_H2C, entry['beta_dram'], pad_b_bf16, ab_bytes)

    def _upload_runtime_k(self, entry, k_host) -> None:
        """Pad k to [T, num_heads, Dk, 64] with col 0 = k value, upload
        per-(t,h) head slabs to K_PAD_DRAM.  matmat_mul_core then reads
        A[Dk, 64] for each head from this buffer."""
        T = entry['T']; H = entry['num_heads']; Dk = entry['Dk']
        N = UE_VECTOR_SIZE
        k_np = k_host.detach().float().cpu().numpy()  # [T, H, Dk]
        pad_k = np.zeros((H, Dk, N), dtype=np.float32)
        # For T=1 decode, collapse the T dim; for T>1 we'd fall back to baked.
        pad_k[:, :, 0] = k_np[0]
        pad_k_bf16 = torch.from_numpy(pad_k).to(torch.bfloat16).contiguous()
        self.dma_write(DMA_DEVICE_H2C, entry['k_pad_dram'],
                       pad_k_bf16, H * Dk * N * 2)

    def _record_recurrence(self, T, num_heads, Dk, Dv,
                           Q_DRAM_ADDR, K_DRAM_ADDR, V_DRAM_ADDR,
                           S_DRAM_ADDR, OUT_DRAM_ADDR, SCRATCH_DRAM_ADDR,
                           alpha_host, beta_host, k_host) -> dict:
        """For T=1 (decode): emit runtime α/β recurrence using per-layer
        preallocated DRAM slots (see prepare_inference).  For T>1 (prefill):
        delegate to super() (baked scalars)."""
        if T == 1:
            return self._record_recurrence_runtime_ab(
                T, num_heads, Dk, Dv,
                Q_DRAM_ADDR, K_DRAM_ADDR, V_DRAM_ADDR,
                S_DRAM_ADDR, OUT_DRAM_ADDR, SCRATCH_DRAM_ADDR,
                alpha_host, beta_host, k_host)

        start_idx = self.capture_count
        super().recurrent_gated_delta_rule_core(
            T=T, num_heads=num_heads, Dk=Dk, Dv=Dv,
            Q_DRAM_ADDR=Q_DRAM_ADDR, K_DRAM_ADDR=K_DRAM_ADDR,
            V_DRAM_ADDR=V_DRAM_ADDR,
            S_DRAM_ADDR=S_DRAM_ADDR, OUT_DRAM_ADDR=OUT_DRAM_ADDR,
            SCRATCH_DRAM_ADDR=SCRATCH_DRAM_ADDR,
            alpha_host=alpha_host, beta_host=beta_host, k_host=k_host)
        end_idx = self.capture_count
        span = list(self.capture_buffer[start_idx:end_idx])
        scalar_inst_indices = [i for i, inst in enumerate(span)
                               if ((inst.words[5] >> 12) & 0xF) == UE_MODE.MUL_BROADCAST.value
                               and ((inst.words[7] >> 5) & 0x3) == 3]
        expected = T * num_heads * (3 + Dk)
        if len(scalar_inst_indices) != expected:
            raise RuntimeError(
                f"scalar count mismatch: {len(scalar_inst_indices)} vs {expected}")
        patches = []
        idx_iter = iter(scalar_inst_indices)
        for t in range(T):
            for h in range(num_heads):
                patches.append((next(idx_iter), 'alpha',    t, h, -1))
                patches.append((next(idx_iter), 'neg_beta', t, h, -1))
                patches.append((next(idx_iter), 'beta',     t, h, -1))
                for i in range(Dk):
                    patches.append((next(idx_iter), 'k', t, h, i))
        return {'instructions': span, 'patches': patches,
                'T': T, 'num_heads': num_heads, 'runtime_ab': False}

    def _record_recurrence_runtime_ab(self, T, num_heads, Dk, Dv,
                                      Q_DRAM_ADDR, K_DRAM_ADDR, V_DRAM_ADDR,
                                      S_DRAM_ADDR, OUT_DRAM_ADDR,
                                      SCRATCH_DRAM_ADDR,
                                      alpha_host, beta_host, k_host) -> dict:
        """Emit the T=1 Gated DeltaNet recurrence with runtime α/β/k.

        Pipeline overview (one decode token):
          Phase A (per head)    : αS := α · S_prev,  written in-place to S_DRAM.
          Phase B (batched)     : m_wide := k_all @ S_cache    (one wide matmul).
          Phase C (per head)    : δ, outer(δ,k), S_new := αS + outer.
                                  Step-4 transpose+matmul are SRAM-fused, no
                                  DRAM round-trip through b_pad/outer scratch.
          Phase D (batched)     : y_wide := q_all @ S_cache    (one wide matmul);
                                  diagonal blocks are extracted into OUT_DRAM
                                  via one strided gather DMA.

        Scalars are runtime-loaded into LALU via an RMS+RSQRT trick (host pre-
        uploads y = 1/x; HW computes 1/|y| = x back).  k is runtime-scattered
        into k_pad_dram by `_emit_k_pad_fpga` upstream.
        """
        bytes_per_element = 2
        N = UE_VECTOR_SIZE
        del SCRATCH_DRAM_ADDR  # unused in T=1 (Phase 1+2 dropped αS scratch)

        # SRAM layout used across the recurrence body:
        SA_MAT            = 0x00000   # URAM_A: S/m/etc loads
        SA_V              = 0x00400   # URAM_A: v load (step 3)
        SA_DELTA          = 0x00800   # URAM_A: δ (step 3 output / step 4 input)
        SA_OUTER          = 0x10000   # URAM_A: legacy slot (no longer used for T=1)
        SA_SCALAR_SCRATCH = 0x60000   # URAM_A: scalar→LALU staging
        SB_AS             = 0x80000   # URAM_B: αS load for step 5
        SB_NBM            = 0x90000   # URAM_B: -β·m intermediate (step 3)

        # Per-layer pre-allocated DRAM slots (indexed by S_DRAM_ADDR).
        alpha_dram = self._alpha_dram_by_s[S_DRAM_ADDR]
        beta_dram  = self._beta_dram_by_s[S_DRAM_ADDR]
        k_pad_dram = self._k_pad_dram_by_s[S_DRAM_ADDR]
        d_pre_dram = self._d_pre_dram_by_s[S_DRAM_ADDR]
        b_pad_dram = self._b_pad_dram_by_s[S_DRAM_ADDR]

        # SRAM addresses for the dot-product scalar-extraction trick.
        # Phase B/D matmul (M=16, K=128, N=2048) fills URAM_B rows 0..3967 with
        # B-chunks; SB_E0 must live above that so e_0 survives across phases
        # without needing per-phase reloads.  Row 4032 = byte 0xFE000 fits.
        SB_E0 = 0xFE000   # URAM_B row 4032: e_0 = [1, 0, ..., 0] selector

        def _emit_scalar_via_dot_product(slot_dram_addr: int) -> None:
            """Extract α[h] from a 64-element slot (slot[0]=α, rest zeros)
            into the LALU register via BF16_DOT_PRODUCT(slot, e_0) + LALU RELU.

            RELU(α) = α since α > 0, populating the LALU register so that the
            next broadcast_mul with BROADCAST_MODE.LALU_RESULT consumes it
            directly.  Replaces the DMA + RSQRT (UE_MODE.RMS + LALU RSQRT)
            scalar-load — same net effect (α in LALU register) on the
            matmul-engine path instead of the RMS pipeline.
            Requires the α-on-FPGA pipeline to write α directly (not 1/α);
            see `_emit_alpha_fpga_to_slots` for the pipeline change.
            Pre-condition: e_0 = [1, 0, ..., 0] in URAM_B at SB_E0 (loaded
            once per recurrence call by _phase_A_alpha_S).
            """
            self.accelerator_memory_to_sram(
                accelerator_dram_address=slot_dram_addr,
                sram_address=SA_SCALAR_SCRATCH,
                element_size=N)
            v_type, v_addr = self.sram_address_to_uram_address(SA_SCALAR_SCRATCH)
            m_type, m_addr = self.sram_address_to_uram_address(SB_E0)
            assert v_type == URAM_SECTION.URAM_A
            assert m_type == URAM_SECTION.URAM_B
            self.ue_arithmetic_op(
                0,                                    # broadcast_mode
                0,                                    # max_clear_en
                1,                                    # stride_z (rows)
                LALU_CLAMP_RELU_A, LALU_CLAMP_RELU_B, # clamp(α, 0, +inf) = ReLU(α)
                LALU_MODE.CLAMP.value,                # CLAMP(α,0,+inf)=α for α>0; populates LALU reg
                0,                                    # scalar (unused for CLAMP)
                URAM_SECTION.URAM_A.value,            # uram_section (irrelevant when WB disabled)
                0,                                    # uram_dst_addr
                0,                                    # uram_wb_addr
                URAM_WRITE_SRC.URAM_WB_DISABLE.value, # keep result in LALU register
                UE_MODE.BF16_DOT_PRODUCT,
                0,                                    # data_type
                v_addr,                               # uram_a_start_addr (slot)
                m_addr,                               # uram_b_start_addr (e_0 row)
                1,                                    # K/UE_VECTOR_SIZE = 64/64 = 1
                0,                                    # dma_start_addr
                64,                                   # dma_length (K*N)
                1,                                    # output_size (N=1)
            )

        start_idx = self.capture_count

        # S layout invariant: stored TRANSPOSED as [Dv, Dk] per head (so it's
        # actually S^T relative to the math).  This lets steps 2 and 6 read
        # B = S directly with no transpose, and step 4 computes outer(δ, k)
        # [Dv, Dk] (A/B swapped) so the step-5 add lines up element-wise.
        BPE = bytes_per_element
        S_HEAD_BYTES = Dk * Dv * BPE                # one head's S block
        WIDE_HEAD_STRIDE = (num_heads + 1) * Dv * BPE   # diag offset of m_wide / y_wide

        # ---- Phase A: per-head step 1 -- αS computed in-place into S_DRAM ---
        # α now lives in alpha_dram as raw α (was 1/α); per-head extraction is
        # via BF16_DOT_PRODUCT(slot, e_0) → LALU register, replacing the prior
        # DMA + RSQRT dance.  e_0 is pre-loaded to URAM_B SB_E0 once per call.
        def _phase_A_alpha_S(t: int) -> None:
            self.accelerator_memory_to_sram(self._e0_dram, SB_E0, N)
            for h in range(num_heads):
                S_h        = S_DRAM_ADDR + h * S_HEAD_BYTES
                alpha_slot = alpha_dram  + (t * num_heads + h) * N * BPE
                self.accelerator_memory_to_sram(S_h, SA_MAT, Dk * Dv)
                _emit_scalar_via_dot_product(alpha_slot)
                self.start_queue_broadcast(
                    UE_MODE.MUL_BROADCAST, BROADCAST_MODE.LALU_RESULT,
                    SA_MAT, SA_MAT, Dk * Dv)
                self.sram_to_accelerator_memory(SA_MAT, S_h, Dk * Dv)

        # ---- Phase B: batched step 2 -- m_wide = k_all @ αS_cache ----------
        # S_DRAM now holds αS for every head in contiguous [H*Dv, Dk] layout.
        # One wide matmul computes all 16 per-head m[h] on the m_wide diagonal.
        def _phase_B_m_wide(t: int) -> None:
            K_BASE = K_DRAM_ADDR + t * num_heads * Dk * BPE
            self.matmat_mul_core(M=num_heads, K=Dk, N=num_heads * Dv,
                                 A_DRAM_ADDR=K_BASE,
                                 B_DRAM_ADDR=S_DRAM_ADDR,
                                 OUTPUT_DRAM_ADDR=self._m_wide_dram,
                                 is_B_quantized=False)

        # ---- Phase C: per-head steps 3+4 (SRAM-fused) + spill outer to DRAM
        # step 3 : δ = β·(v − m)                      (LALU β reused across 2 muls)
        # step 4 : outer(δ, k)  =  transpose(d_pre) × k_pad       SRAM-fused —
        #          transpose's output sits at SRAM 0x2000, the inline matmul
        #          reads A from there directly and leaves output at SRAM 0x6000.
        # spill  : outer is then DMA'd to _outer_wide_dram[h] so Phase E can
        #          batch step 5 across all heads.  Note: spill must come AFTER
        #          a URAM_A read (here, the matmul's last matvec) — bare
        #          DMA-after-matmul races with the matmul's URAM writeback.
        SA_BPAD_FROM_TRANSPOSE = 0x02000   # where bf16_transpose_core leaves output
        SA_OUTER_FROM_MATMUL   = 0x06000   # where _emit_step4_matmul_sram_A leaves output

        def _phase_C_per_head(t: int, h: int) -> None:
            v_addr     = V_DRAM_ADDR + (t * num_heads + h) * Dv * BPE
            S_h        = S_DRAM_ADDR + h * S_HEAD_BYTES
            beta_slot  = beta_dram   + (t * num_heads + h) * N * BPE
            m_h_addr   = self._m_wide_dram + h * WIDE_HEAD_STRIDE  # diag block
            head_k_pad = k_pad_dram + h * Dk * N * BPE

            # step 3: δ = β*v + (-β)*m  (β extracted via dot_product+RELU,
            # uniform with α; β slot now holds β directly after the per-head
            # invert in _emit_inv_beta_fpga_to_slots step 7.)
            self.accelerator_memory_to_sram(m_h_addr, SA_MAT, Dv)
            self.accelerator_memory_to_sram(v_addr,   SA_V,   Dv)
            _emit_scalar_via_dot_product(beta_slot)
            self.start_queue_broadcast(
                UE_MODE.MUL_BROADCAST, BROADCAST_MODE.LALU_RESULT,
                SA_V, SA_DELTA, Dv)
            self.start_queue_broadcast(
                UE_MODE.MUL_BROADCAST, BROADCAST_MODE.LALU_RESULT_NEGATE,
                SA_MAT, SB_NBM, Dv)
            self.eltwise_add_core(SA_DELTA, SB_NBM, SA_DELTA, Dv)

            # step 4: outer(δ, k) -- transpose + custom SRAM-fused matmul
            self.sram_to_accelerator_memory(SA_DELTA, d_pre_dram, Dv)
            self.bf16_transpose_core(
                M=N, N=Dv,
                INPUT_DRAM_ADDR=d_pre_dram,
                OUTPUT_DRAM_ADDR=b_pad_dram)   # b_pad_dram unused; matmul reads SRAM
            _emit_step4_matmul_sram_A(
                self,
                B_DRAM=head_k_pad,
                A_sram_addr=SA_BPAD_FROM_TRANSPOSE,
                out_sram_addr=SA_OUTER_FROM_MATMUL,
                M=Dv, K=N, N=Dk)

            # spill outer[h] → _outer_wide_dram[h] for batched Phase E.
            outer_h_dram = self._outer_wide_dram + h * Dk * Dv * BPE
            self.sram_to_accelerator_memory(SA_OUTER_FROM_MATMUL,
                                            outer_h_dram, Dk * Dv)

        # ---- Phase E: batched step 5 -- S_new = αS + outer over all heads --
        # S_DRAM holds αS for all H heads (from Phase A), _outer_wide_dram
        # holds outer for all H heads (spilled per-head in Phase C).  Total
        # numel is H·Dk·Dv = 262144 bf16 elements; split into 2 chunks of
        # 131072 elements each (8 heads' worth, fits half a URAM bank).
        # 2 × (1 DMA load + 1 DMA load + 1 eltwise + 1 DMA store) = 8 instr,
        # vs. the previous 16 × (load αS + add + store) = 48 instructions.
        def _phase_E_S_new_batched(t: int) -> None:
            chunk_elems = (num_heads * Dk * Dv) // 2          # 131072 elts
            chunk_bytes = chunk_elems * BPE                    # 262144 B
            for chunk in range(2):
                off_bytes = chunk * chunk_bytes
                self.accelerator_memory_to_sram(
                    self._outer_wide_dram + off_bytes,
                    SA_MAT, chunk_elems)
                self.accelerator_memory_to_sram(
                    S_DRAM_ADDR + off_bytes,
                    SB_AS, chunk_elems)
                self.eltwise_add_core(SA_MAT, SB_AS, SA_MAT, chunk_elems)
                self.sram_to_accelerator_memory(
                    SA_MAT, S_DRAM_ADDR + off_bytes, chunk_elems)

        # ---- Phase D: batched step 6 -- y_wide = q_all @ S_new_cache --------
        # All 16 S_new are in S_DRAM after Phase E.  One wide matmul produces
        # y_wide [H, H·Dv]; diagonal blocks (h, h·Dv..(h+1)·Dv) carry y[h].
        # A single strided-gather DMA collapses the diagonals into OUT_DRAM.
        def _phase_D_y_wide(t: int) -> None:
            Q_BASE = Q_DRAM_ADDR  + t * num_heads * Dk * BPE
            O_BASE = OUT_DRAM_ADDR + t * num_heads * Dv * BPE
            self.matmat_mul_core(M=num_heads, K=Dk, N=num_heads * Dv,
                                 A_DRAM_ADDR=Q_BASE,
                                 B_DRAM_ADDR=S_DRAM_ADDR,
                                 OUTPUT_DRAM_ADDR=self._y_wide_dram,
                                 is_B_quantized=False)
            self.accelerator_memory_to_sram(
                accelerator_dram_address=self._y_wide_dram,
                sram_address=0x00000,
                element_size=num_heads * Dv,
                stride_bytes_per_chunk=Dv * BPE,
                stride_jump_bytes=WIDE_HEAD_STRIDE)
            self.sram_to_accelerator_memory(
                sram_address=0x00000,
                accelerator_dram_address=O_BASE,
                element_size=num_heads * Dv)

        # ---- Drive the five phases (one decode token per outer iteration) --
        for t in range(T):
            _phase_A_alpha_S(t)
            _phase_B_m_wide(t)
            for h in range(num_heads):
                _phase_C_per_head(t, h)
            _phase_E_S_new_batched(t)
            _phase_D_y_wide(t)

        end_idx = self.capture_count
        span = list(self.capture_buffer[start_idx:end_idx])

        # No baked scalar broadcasts remain — α/β come from LALU (runtime),
        # k comes from DRAM (host-uploaded padded layout, consumed by matmul).
        residual_scalar_broadcasts = [i for i, inst in enumerate(span)
            if ((inst.words[5] >> 12) & 0xF) == UE_MODE.MUL_BROADCAST.value
            and ((inst.words[7] >> 5) & 0x3) == 3]
        if residual_scalar_broadcasts:
            raise RuntimeError(
                f"Unexpected {len(residual_scalar_broadcasts)} SCALAR_IN_REG "
                "broadcast_muls — runtime-ab recurrence should have none")
        return {'instructions': span, 'patches': [],
                'alpha_dram': alpha_dram, 'beta_dram': beta_dram,
                'k_pad_dram': k_pad_dram, 'S_DRAM_ADDR': S_DRAM_ADDR,
                'T': T, 'num_heads': num_heads, 'Dk': Dk,
                'runtime_ab': True, 'runtime_k': True}

    def _replay_recurrence(self, entry, alpha_host, beta_host, k_host,
                           alpha_on_fpga: bool = False,
                           beta_on_fpga: bool = False,
                           k_on_fpga: bool = False) -> None:
        insts = entry['instructions']
        self.capture_buffer.extend(insts)
        self.capture_count += len(insts)
        self._inst_id += len(insts)
        start_in_capture = self.capture_count - len(insts)
        if entry.get('runtime_ab'):
            if alpha_on_fpga and beta_on_fpga:
                pass  # both written by FPGA pipelines; skip host uploads
            elif alpha_on_fpga:
                # α already written to alpha_dram by the FPGA pipeline;
                # only upload β (host still computes sigmoid(beta_proj)).
                self._upload_runtime_beta_only(entry, beta_host)
            else:
                self._upload_runtime_ab(entry, alpha_host, beta_host)
        if entry.get('runtime_k') and not k_on_fpga:
            self._upload_runtime_k(entry, k_host)
        if entry['patches']:
            self._pending_scalar_patches.append(
                ('rec', start_in_capture, entry['patches'],
                 alpha_host, beta_host, k_host))
            self._current_capture_has_patches = True

    # ---- One-time inference setup: caches + all layer weights --------------
    def prepare_inference(self, weights: dict, max_context: int = 256) -> None:
        """Allocate persistent S/KV caches and upload every layer's weights
        from a pre-extracted dict (produced by _extract_all_weights or loaded
        from qwen3.5_2b_bin/params.bin).  `weights` must have
        keys: 'layers' (list of per-layer extracted dicts), 'final_norm_gamma'
        (bf16 tensor), 'embed_weight' (bf16 tensor).

        After this call:
          - `self._s_dram[idx]`: S state per linear-attn layer (zeroed)
          - `self._k_cache[idx]`, `self._v_cache[idx]`: KV cache per full-attn
            layer, laid out [num_q_heads, T_aligned, head_dim] bf16.
          - `self._layer_weights[idx]`: dict of DRAM addresses + host tensors
          - `self._conv_state[idx]`: host-side [K-1, conv_dim] fp32 tensor
          - `self._cache_pos`: number of tokens currently in the cache.
        """
        self.max_context = max_context
        # Decoder binary paths (set here so compile_decoder / load_instructions
        # can be called without passing paths explicitly, matching Gemma3 API).
        _script_dir = str(Path(__file__).parent)
        _cfg_paths  = self._cfg.get("paths", {})
        # Transient compile artifacts — folded into programs.bin and removed by
        # the caller. Kept under qwen3.5_2b_bin/ but hidden (leading dot) so the
        # final output directory only contains params.{bin,json} + programs.{bin,json}.
        self._decoder_bin_path  = os.path.join(
            _script_dir, "qwen3.5_2b_bin", ".decoder_compile.bin")
        self._decoder_meta_path = os.path.join(
            _script_dir, "qwen3.5_2b_bin", ".decoder_compile.json")
        T_aligned = max_context
        if T_aligned < UE_VECTOR_SIZE:
            T_aligned = UE_VECTOR_SIZE
        if T_aligned % UE_VECTOR_SIZE:
            T_aligned = ((T_aligned // UE_VECTOR_SIZE) + 1) * UE_VECTOR_SIZE
        self.max_context_aligned = T_aligned

        # Build eps tails for every N we'll use (must sit below layer-weight
        # region so per-layer pointer resets don't orphan them — but we
        # upload weights only once now, so this ordering is just defensive).
        self._build_eps_tail(self.hidden_size, 1e-6)
        self._build_eps_tail(self.full_head_dim, 1e-6)
        self._build_eps_tail(self.lin_head_k_dim, 1e-6)

        # L2 gammas for linear-attn Q/K normalization.
        Dk = self.lin_head_k_dim
        self._gamma_q_l2 = _upload(self,
            torch.full((Dk,), 1.0 / Dk, dtype=torch.float32))
        self._gamma_k_l2 = _upload(self,
            torch.full((Dk,), 1.0 / (Dk ** 0.5), dtype=torch.float32))

        # ------------------------------------------------------------------
        # Caches live in TENSOR DRAM's low region (not params DRAM) so the
        # 768 MB params budget stays free for the 440+ MB of layer weights.
        # After all cache allocations, we bump `self._tensor_dram_base` so
        # `reset_tensor_dram_addr()` (called before every layer) preserves
        # them — only activations above the bumped base get recycled.
        # Sizes at max_context=256:
        #   identity eye(6144):  72 MB   (for silu-via-identity-matmul)
        #   S cache (18 layers):  9 MB
        #   KV cache (6 layers): 12 MB
        #   bias:               ~66 KB
        # Total reserved: ~93 MB.  Activation region: ~400 MB free.
        # ------------------------------------------------------------------

        # eye(conv_dim) for the post-conv1d SiLU (run_silu_dram).  Populate
        # the module-level _IDENTITY_DRAM_CACHE so later lookups skip upload.
        conv_dim = self.lin_conv_dim
        eye_bytes = conv_dim * conv_dim * BF16
        eye_addr = self.allocate_tensor_dram(eye_bytes)
        self.dma_write(DMA_DEVICE_H2C, eye_addr,
                       _bf(torch.eye(conv_dim, dtype=torch.float32)),
                       eye_bytes)
        _IDENTITY_DRAM_CACHE[conv_dim] = eye_addr

        # S cache: one [num_vh, Dk, Dv] bf16 buffer per linear-attn layer.
        num_vh = self.lin_num_v_heads
        Dv = self.lin_head_v_dim
        s_elems = num_vh * Dk * Dv
        zeros_s = _bf(torch.zeros(s_elems))
        self._s_dram: Dict[int, int] = {}
        # Decode-time α/β slots per linear-attn layer: T=1 * num_vh heads,
        # padded to a 64-bf16 block each (only position 0 carries y=1/s).
        # Indexed by S_DRAM_ADDR — _record_recurrence_runtime_ab looks up
        # the right slot for the layer via its S address.
        self._alpha_dram_by_s: Dict[int, int] = {}
        self._beta_dram_by_s:  Dict[int, int] = {}
        # Per-layer scratch DRAM for α-on-FPGA pipeline intermediates:
        #   alpha_x_dram:    [UE_VECTOR_SIZE] raw x = a_raw + dt_bias
        #   alpha_rx_dram:   [UE_VECTOR_SIZE] relu(x) via identity matmul
        #   alpha_y_dram:    [UE_VECTOR_SIZE] α output (scattered to alpha_dram slots)
        self._alpha_x_dram_by_s:  Dict[int, int] = {}
        self._alpha_rx_dram_by_s: Dict[int, int] = {}
        self._alpha_y_dram_by_s:  Dict[int, int] = {}
        # Decode-time padded-matmul scratch for outer(k, δ):
        #   K_PAD: [num_vh, Dk, 64] — host uploads col-0=k per step.
        #   D_PRE: [64, Dv]         — pre-zeroed, row 0 ← δ per (t,h); rest stays 0.
        #   B_PAD: [Dv, 64]         — transpose(D_PRE), col 0 = δ.
        #   OUTER: [Dk, Dv]         — matmul(K_PAD[h], B_PAD) = outer(k, δ).
        self._k_pad_dram_by_s: Dict[int, int] = {}
        self._d_pre_dram_by_s: Dict[int, int] = {}
        self._b_pad_dram_by_s: Dict[int, int] = {}
        self._outer_dram_by_s: Dict[int, int] = {}
        ab_decode_bytes   = num_vh * UE_VECTOR_SIZE * BF16
        alpha_scratch_bytes = UE_VECTOR_SIZE * BF16
        zeros_ab          = _bf(torch.zeros(num_vh * UE_VECTOR_SIZE))
        k_pad_bytes       = num_vh * Dk * UE_VECTOR_SIZE * BF16
        d_pre_bytes       = UE_VECTOR_SIZE * Dv * BF16
        b_pad_bytes       = Dv * UE_VECTOR_SIZE * BF16
        outer_bytes       = Dk * Dv * BF16
        zeros_d_pre       = _bf(torch.zeros(UE_VECTOR_SIZE * Dv))
        zeros_k_pad = _bf(torch.zeros(num_vh * Dk * UE_VECTOR_SIZE))
        beta_y_bytes = UE_VECTOR_SIZE * BF16
        zeros_beta_y = _bf(torch.zeros(UE_VECTOR_SIZE))
        self._beta_y_dram_by_s: Dict[int, int] = {}

        # Batched step-6 output buffer: y_wide [H, H*Dv] bf16 (~64 KB for
        # H=16, Dv=128). One wide matmul y_wide = q_all @ S_cache replaces
        # the per-head step-6 loop; only the diagonal blocks
        # y_wide[h, h*Dv..(h+1)*Dv] carry the per-head result, extracted via
        # a strided DMA into OUT_DRAM.  Shared across all linear-attn layers
        # since only one layer is active at any time during decode.
        y_wide_bytes = num_vh * num_vh * Dv * BF16
        self._y_wide_dram = self.allocate_tensor_dram(y_wide_bytes)
        self.dma_write(DMA_DEVICE_H2C, self._y_wide_dram,
                       _bf(torch.zeros(num_vh * num_vh * Dv)), y_wide_bytes)
        # Batched step-2 output buffer: m_wide [H, H*Dv] bf16 (~64 KB).
        # Same pattern as y_wide — one wide matmul m_wide = k_all @ αS_cache
        # replaces 16 per-head matmuls; diagonals m_wide[h, h*Dv..] = k[h]@αS[h].
        # Step 3 reads m[h] directly from m_wide diagonal (no extraction DMA
        # needed since step 3 is still per-head, so per-head addressing is free).
        m_wide_bytes = num_vh * num_vh * Dv * BF16
        self._m_wide_dram = self.allocate_tensor_dram(m_wide_bytes)
        self.dma_write(DMA_DEVICE_H2C, self._m_wide_dram,
                       _bf(torch.zeros(num_vh * num_vh * Dv)), m_wide_bytes)
        # Batched step-5 staging buffer: outer_wide [H, Dk, Dv] bf16 (~512 KB).
        # Phase C spills each head's `outer` (rank-1 result) here so a single
        # batched eltwise_add (Phase E) can compute S_new = αS + outer for
        # all 16 heads at once, replacing 16 per-head eltwise_add_core calls.
        outer_wide_bytes = num_vh * Dk * Dv * BF16
        self._outer_wide_dram = self.allocate_tensor_dram(outer_wide_bytes)
        self.dma_write(DMA_DEVICE_H2C, self._outer_wide_dram,
                       _bf(torch.zeros(num_vh * Dk * Dv)), outer_wide_bytes)
        for idx in sorted(self.linear_attn_layers):
            addr = self.allocate_tensor_dram(s_elems * BF16)
            self.dma_write(DMA_DEVICE_H2C, addr, zeros_s, s_elems * BF16)
            self._s_dram[idx] = addr
            a_addr = self.allocate_tensor_dram(ab_decode_bytes)
            self.dma_write(DMA_DEVICE_H2C, a_addr, zeros_ab, ab_decode_bytes)
            b_addr = self.allocate_tensor_dram(ab_decode_bytes)
            self.dma_write(DMA_DEVICE_H2C, b_addr, zeros_ab, ab_decode_bytes)
            self._alpha_dram_by_s[addr] = a_addr
            self._beta_dram_by_s[addr]  = b_addr
            ax_addr  = self.allocate_tensor_dram(alpha_scratch_bytes)
            arx_addr = self.allocate_tensor_dram(alpha_scratch_bytes)
            ay_addr  = self.allocate_tensor_dram(alpha_scratch_bytes)
            self._alpha_x_dram_by_s[addr]  = ax_addr
            self._alpha_rx_dram_by_s[addr] = arx_addr
            self._alpha_y_dram_by_s[addr]  = ay_addr
            # β-on-FPGA scratch: [UE_VECTOR_SIZE] bf16.
            by_addr = self.allocate_tensor_dram(beta_y_bytes)
            self.dma_write(DMA_DEVICE_H2C, by_addr, zeros_beta_y, beta_y_bytes)
            self._beta_y_dram_by_s[addr] = by_addr
            k_addr = self.allocate_tensor_dram(k_pad_bytes)
            # Pre-zero k_pad_dram so non-col-0 entries remain zero.
            self.dma_write(DMA_DEVICE_H2C, k_addr, zeros_k_pad, k_pad_bytes)
            d_addr = self.allocate_tensor_dram(d_pre_bytes)
            # Pre-zero D_PRE so rows 1..63 stay at zero across all (t,h);
            # only row 0 gets overwritten each step.
            self.dma_write(DMA_DEVICE_H2C, d_addr, zeros_d_pre, d_pre_bytes)
            p_addr = self.allocate_tensor_dram(b_pad_bytes)
            o_addr = self.allocate_tensor_dram(outer_bytes)
            self._k_pad_dram_by_s[addr] = k_addr
            self._d_pre_dram_by_s[addr] = d_addr
            self._b_pad_dram_by_s[addr] = p_addr
            self._outer_dram_by_s[addr] = o_addr

        # KV cache: [num_q_heads, T_aligned, full_head_dim] bf16 per layer.
        # Stored POST-GQA-expansion so each head's buffer is a direct
        # flash_attention_core input (no on-the-fly expansion at run time).
        kv_elems = self.full_num_heads * T_aligned * self.full_head_dim
        zeros_kv = _bf(torch.zeros(kv_elems))
        self._k_cache: Dict[int, int] = {}
        self._v_cache: Dict[int, int] = {}
        for idx in sorted(self.full_attn_layers):
            k_addr = self.allocate_tensor_dram(kv_elems * BF16)
            self.dma_write(DMA_DEVICE_H2C, k_addr, zeros_kv, kv_elems * BF16)
            self._k_cache[idx] = k_addr
            v_addr = self.allocate_tensor_dram(kv_elems * BF16)
            self.dma_write(DMA_DEVICE_H2C, v_addr, zeros_kv, kv_elems * BF16)
            self._v_cache[idx] = v_addr

        # Bias buffer: causal lower-tri over T_aligned, -inf above diagonal.
        # Rewritten each full-attn call (cheap: <64 KB) to mask cols beyond
        # the current cache position.
        self._bias_dram = self.allocate_tensor_dram(T_aligned * T_aligned * BF16)
        self._upload_causal_bias(self.max_context_aligned)

        # Constant ones vector for β-on-FPGA: UE_VECTOR_SIZE bf16 ones in params DRAM.
        ones_vec = torch.ones(UE_VECTOR_SIZE, dtype=torch.bfloat16)
        ones_bytes = UE_VECTOR_SIZE * BF16
        self._ones_dram = super().allocate_params_dram(ones_bytes)
        super().dma_write(DMA_DEVICE_H2C, self._ones_dram, ones_vec, ones_bytes)

        # Selector e_0 = [1, 0, 0, ..., 0] (UE_VECTOR_SIZE bf16) for the
        # BF16_DOT_PRODUCT scalar-extraction trick: dot(α_slot, e_0) = α[h]
        # routed straight to LALU register (URAM_WB_DISABLE), letting the
        # subsequent broadcast_mul consume it via BROADCAST_MODE.LALU_RESULT.
        # This replaces the older DMA + RSQRT scalar-load dance.
        e0_vec = torch.zeros(UE_VECTOR_SIZE, dtype=torch.bfloat16)
        e0_vec[0] = 1.0
        self._e0_dram = super().allocate_params_dram(ones_bytes)
        super().dma_write(DMA_DEVICE_H2C, self._e0_dram, e0_vec, ones_bytes)

        # Spreader matrix for α/β scatter.  matmat_mul_core does A @ B^T where
        # B is stored as [N_out, K].  We want output[0, h*N] = y[h] (else 0),
        # which requires B[h*N, h] = 1 (else 0), so B shape = [num_vh*N, N].
        # Same matrix works for both α and β (same num_vh and N).
        N_us = UE_VECTOR_SIZE
        alpha_spreader = torch.zeros(num_vh * N_us, N_us, dtype=torch.bfloat16)
        for h in range(num_vh):
            alpha_spreader[h * N_us, h] = 1.0
        alpha_spreader_bytes = num_vh * N_us * N_us * BF16
        self._alpha_spreader_dram = super().allocate_params_dram(alpha_spreader_bytes)
        super().dma_write(DMA_DEVICE_H2C, self._alpha_spreader_dram,
                          _bf(alpha_spreader.contiguous()), alpha_spreader_bytes)

        # Spreader for k-pad scatter.  Per-head: A=k_h [1, Dk], B=[Dk*N, Dk]
        # with B[dk*N, dk] = 1.  Output[0, dk*N] = k_h[dk].
        k_spreader = torch.zeros(Dk * N_us, Dk, dtype=torch.bfloat16)
        for dk in range(Dk):
            k_spreader[dk * N_us, dk] = 1.0
        k_spreader_bytes = Dk * N_us * Dk * BF16
        self._k_spreader_dram = super().allocate_params_dram(k_spreader_bytes)
        super().dma_write(DMA_DEVICE_H2C, self._k_spreader_dram,
                          _bf(k_spreader.contiguous()), k_spreader_bytes)

        # Pre-zero k_pad_dram: already allocated per-layer below; zeros written there.

        # Full-attn: RoPE cos/sin cache for all positions.
        # rope_core_dram cannot be used for rot_dim=64 (N=64 makes half_N=32
        # elements = 64 bytes; eltwise_mul inside rope_core requires 128-byte
        # SRAM alignment, which is impossible at +64-byte offset).  Instead we
        # store three split-padded buffers — each [max_context, rot_dim] bf16,
        # with only the first half_rot positions non-zero — and implement RoPE
        # manually via 6 eltwise ops using 128-byte-aligned full-width slots.
        rot_dim = self.full_rotary_dim  # 64
        half_rot = rot_dim // 2          # 32
        rope_theta_val = self.rope_theta
        inv_freq = 1.0 / (rope_theta_val ** (
            torch.arange(0, rot_dim, 2, dtype=torch.float32) / rot_dim))  # [32]
        # cos_pad[pos, 0:32]=cos, cos_pad[pos, 32:64]=0
        # neg_sin_pad[pos, 0:32]=-sin (for x_hi cross-term), [32:64]=0
        # sin_hi_pad[pos, 0:32]=sin (for x_lo cross-term), [32:64]=0
        cos_pad_buf     = torch.zeros(max_context, rot_dim, dtype=torch.bfloat16)
        neg_sin_pad_buf = torch.zeros(max_context, rot_dim, dtype=torch.bfloat16)
        sin_hi_pad_buf  = torch.zeros(max_context, rot_dim, dtype=torch.bfloat16)
        for pos in range(max_context):
            freqs = pos * inv_freq  # keep float32 for trig precision
            c_f32, s_f32 = freqs.cos(), freqs.sin()
            cos_pad_buf[pos, :half_rot]     = c_f32.to(torch.bfloat16)
            neg_sin_pad_buf[pos, :half_rot] = (-s_f32).to(torch.bfloat16)
            sin_hi_pad_buf[pos, :half_rot]  = s_f32.to(torch.bfloat16)
        rope_pad_bytes = max_context * rot_dim * BF16
        self._fa_cos_pad_dram     = self.allocate_tensor_dram(rope_pad_bytes)
        self._fa_neg_sin_pad_dram = self.allocate_tensor_dram(rope_pad_bytes)
        self._fa_sin_hi_pad_dram  = self.allocate_tensor_dram(rope_pad_bytes)
        self.dma_write(DMA_DEVICE_H2C, self._fa_cos_pad_dram,     cos_pad_buf,     rope_pad_bytes)
        self.dma_write(DMA_DEVICE_H2C, self._fa_neg_sin_pad_dram, neg_sin_pad_buf, rope_pad_bytes)
        self.dma_write(DMA_DEVICE_H2C, self._fa_sin_hi_pad_dram,  sin_hi_pad_buf,  rope_pad_bytes)

        # Full-attn: persistent Q_FLASH buffers (zeroed once per layer).
        self._fa_q_flash_dram: Dict[int, int] = {}
        fa_head_dim = self.full_head_dim
        fa_num_q = self.full_num_heads
        q_flash_elems = fa_num_q * T_aligned * fa_head_dim
        q_flash_zeros = _bf(torch.zeros(q_flash_elems))
        q_flash_bytes = q_flash_elems * BF16
        for idx in sorted(self.full_attn_layers):
            addr = self.allocate_tensor_dram(q_flash_bytes)
            self.dma_write(DMA_DEVICE_H2C, addr, q_flash_zeros, q_flash_bytes)
            self._fa_q_flash_dram[idx] = addr

        # §7 shared full-attn flash subroutine: ONE set of fixed operand buffers
        # reused by every (layer, head) call site.  The bucketized flash body is
        # compiled ONCE after the decoder HALT (see compile_decoder); each call
        # site marshals its head's Q/K/V into these fixed buffers, jumps into the
        # shared body, and copies the per-head result out.  Sizes match the
        # largest bucket (full max_context_aligned).  Keeping the exact "full
        # T_aligned flash + causal bias mask" numerics as the prior inline path
        # (bucket selector = T_aligned/UE_VECTOR_SIZE), so output stays
        # bit-identical; only the instruction-bin shrinks (48 inline bodies → 4
        # shared bucket bodies).
        s7_buf_elems = T_aligned * fa_head_dim
        s7_zeros = _bf(torch.zeros(s7_buf_elems))
        self._s7_flash_q   = self.allocate_tensor_dram(s7_buf_elems * BF16)
        self._s7_flash_k   = self.allocate_tensor_dram(s7_buf_elems * BF16)
        self._s7_flash_v   = self.allocate_tensor_dram(s7_buf_elems * BF16)
        self._s7_flash_out = self.allocate_tensor_dram(s7_buf_elems * BF16)
        for _z in (self._s7_flash_q, self._s7_flash_k,
                   self._s7_flash_v, self._s7_flash_out):
            self.dma_write(DMA_DEVICE_H2C, _z, s7_zeros, s7_buf_elems * BF16)
        # PBI bucket dispatcher scratch (seq_len*seq_len BF16) + flash scratch
        # (V^T + partial softmax), sized for the max bucket.
        self._s7_attn_p = self.allocate_tensor_dram(T_aligned * T_aligned * BF16)
        self._s7_flash_scratch = self.allocate_tensor_dram(
            T_aligned * T_aligned * 4 + T_aligned * fa_head_dim * 4)
        # Active only during compile_decoder's capture (set/cleared there).  The
        # load_instructions tensor-layout restore pass runs _forward_range with
        # this False so it takes the legacy inline-flash emission (its capture is
        # discarded; only the identical _alloc_tensor sequence matters).
        self._s7_full_attn_active = False
        self._s7_flash_call_sites: list = []

        # Full-attn: eye(head_dim) for sigmoid gate computation in params DRAM.
        fa_head_dim = self.full_head_dim
        eye_head = _bf(torch.eye(fa_head_dim, dtype=torch.float32))
        eye_head_bytes = fa_head_dim * fa_head_dim * BF16
        self._fa_eye_head_dram = super().allocate_params_dram(eye_head_bytes)
        super().dma_write(DMA_DEVICE_H2C, self._fa_eye_head_dram, eye_head, eye_head_bytes)

        # Per-linear-attn-layer conv1d state ring + scratch tap buffers.
        # State ring: K-1 slots of [conv_dim], slot s holds x[t-(K-1)+s].  New
        # tokens roll in at slot K-2 via FPGA memcpy chain at end of each
        # decode step's conv emit.
        K = self._cfg["file_info"]["linear_attn_conv_kernel_size"]
        conv_dim = self.lin_conv_dim
        self._conv_K = K
        self._conv_state_dram: Dict[int, int] = {}
        self._conv_weight_tiles_dram: Dict[int, int] = {}
        self._conv_tap_dram: Dict[int, int] = {}
        for idx in sorted(self.linear_attn_layers):
            state_bytes = (K - 1) * conv_dim * BF16
            state_addr = self.allocate_tensor_dram(state_bytes)
            self.dma_write(DMA_DEVICE_H2C, state_addr,
                           _bf(torch.zeros((K - 1) * conv_dim)), state_bytes)
            self._conv_state_dram[idx] = state_addr
            # K scratch tap buffers (one per tap) for the T=1 conv decode path.
            tap_bytes = K * conv_dim * BF16
            tap_addr = self.allocate_tensor_dram(tap_bytes)
            self._conv_tap_dram[idx] = tap_addr
            # Weight tiles: [K, conv_dim] row-major.  tap s reads from
            # wt_addr + s * conv_dim * BF16.  Data is filled AFTER the weight-
            # upload loop below (which is when W_conv1d becomes available).
            wt_bytes = K * conv_dim * BF16
            wt_addr = self.allocate_tensor_dram(wt_bytes)
            self._conv_weight_tiles_dram[idx] = wt_addr

        # On-chip LM head outputs (permanent tensors — must survive the per-layer
        # reset_tensor_dram_addr()).  LOGITS_DRAM is the LM-head matmul target
        # (never actually written: write_back_disable=True → HW argmax only).
        # PENALTY_BIAS_DRAM is the matmul's per-vocab C bias (bias_mode="broadcast_N")
        # for the on-FPGA repetition penalty; the host refreshes it each penalized
        # step (_write_penalty_bias).  Allocated LAST + seeded to zero (= no penalty
        # = bit-identical greedy) so adding it never shifts other addresses.
        self.VOCAB = int(weights["embed_weight"].shape[0])
        self.LOGITS_DRAM = self.allocate_tensor_dram(self.VOCAB * BF16)
        self.PENALTY_BIAS_DRAM = self.allocate_tensor_dram(self.VOCAB * BF16)
        self.dma_write(DMA_DEVICE_H2C, self.PENALTY_BIAS_DRAM,
                       _bf(torch.zeros(self.VOCAB, dtype=torch.bfloat16)),
                       self.VOCAB * BF16)

        # FREEZE the tensor-DRAM base: everything allocated above is now
        # permanent; reset_tensor_dram_addr() will only reclaim allocations
        # made above this point.
        self._tensor_dram_base = self._tensor_dram_addr
        reserved_mb = (self._tensor_dram_base - DRAM_ACTIVATION_ADDR) / (1024 * 1024)
        print(f"  Tensor DRAM reserved for caches: {reserved_mb:.0f} MB")

        # Final norm gamma (Qwen3_5RMSNorm: (1+w) already folded in by extractor).
        self._final_norm_gamma = _upload(self, weights["final_norm_gamma"])

        # Embedding table — kept on host (~1 GB bf16 too big for params DRAM).
        self._embed_weight = weights["embed_weight"]

        # On-chip LM head: FP4-quantized tied embedding, B=[N=vocab, K=H].  Loaded
        # from the weights bin when present (compile-free load path); for a stale
        # bin without it, quantize on the fly (chunked).  quantized_matmat_core
        # takes (B_DRAM_ADDR=data, SCALE_DRAM_ADDR=scale).
        if "lm_head_data" in weights and "lm_head_scale" in weights:
            _lm_pair = (weights["lm_head_scale"], weights["lm_head_data"])
        else:
            print("  (weights bin lacks lm_head_* — quantizing tied embed on the fly)",
                  flush=True)
            _lm_pair = quantize_lm_head_fp4(self._embed_weight)
        self._lm_head_scale_dram, self._lm_head_data_dram = _upload_fp4_pair(self, _lm_pair)

        # Per-layer weight upload from pre-extracted dicts.  Each layer's dict
        # in weights["layers"] holds bf16 tensors + (scales, packed) FP4 pairs;
        # the _upload_*_weights helpers turn them into FPGA DRAM addresses.
        self._layer_weights: Dict[int, Dict] = {}
        print(f"  Uploading {self.num_layers} layers' weights ...", flush=True)
        t0 = time.time()
        layer_ex_list = weights["layers"]
        for idx in range(self.num_layers):
            ex = layer_ex_list[idx]
            if idx in self.linear_attn_layers:
                self._layer_weights[idx] = _upload_linear_attn_weights(self, ex)
            else:
                self._layer_weights[idx] = _upload_full_attn_weights(self, ex)
            if (idx + 1) % 6 == 0 or idx == self.num_layers - 1:
                used_mb = (self._next_params_dram_addr
                           - self._params_dram_base) / (1024 * 1024)
                print(f"    {idx+1}/{self.num_layers}  "
                      f"params DRAM used: {used_mb:.0f} MB")
        print(f"  Weight upload done ({time.time()-t0:.1f}s).")

        # Fill conv weight tiles now that W_conv1d is captured on host.
        # Layout: [K, conv_dim] row-major, tap s at offset s*conv_dim*BF16.
        # For causal conv at position t: out[t] = sum_s w[:, K-1-s] * x[t-s].
        # Tap s=0 corresponds to x[t] (most recent / current token), so its
        # weight row uses w[:, K-1]; tap s=K-1 uses w[:, 0] (oldest history).
        for idx in sorted(self.linear_attn_layers):
            w_host = self._layer_weights[idx]["W_conv1d"].view(conv_dim, K).float()
            w_tiles = torch.stack([w_host[:, K - 1 - s] for s in range(K)], 0)
            wt_bytes = K * conv_dim * BF16
            self.dma_write(DMA_DEVICE_H2C,
                           self._conv_weight_tiles_dram[idx],
                           _bf(w_tiles), wt_bytes)

        # Host-side conv state: last K-1 rows of in_proj_qkv output per layer.
        # Initialized to zeros (matches causal-conv zero-padding at t<0).
        # This mirrors the FPGA state buffer; prefill updates both so that
        # the first decode step finds consistent history on both sides.
        self._conv_state: Dict[int, torch.Tensor] = {}
        for idx in sorted(self.linear_attn_layers):
            self._conv_state[idx] = torch.zeros(K - 1, self.lin_conv_dim,
                                                dtype=torch.float32)

        self._cache_pos = 0

    def _upload_causal_bias(self, T_aligned: int) -> None:
        """(Re)upload a [T_aligned × T_aligned] causal lower-tri mask into
        self._bias_dram.  Called once at prepare_inference; masking of
        positions beyond cache_pos is applied via a per-step narrower slice."""
        bias = torch.full((T_aligned, T_aligned), float("-inf"),
                          dtype=torch.bfloat16)
        bias.masked_fill_(
            torch.tril(torch.ones(T_aligned, T_aligned, dtype=torch.bool)), 0.0)
        self.dma_write(DMA_DEVICE_H2C, self._bias_dram, _bf(bias),
                       T_aligned * T_aligned * BF16)

    # ------------------------------------------------------------------ #
    # Gemma3-compatible public decoder API                               #
    # ------------------------------------------------------------------ #

    def compile_decoder(self) -> tuple:
        """Compile the full 24-layer decode pass into a single FPGA binary.

        Uses Gemma3's single-hardware-trigger model: one ``start_capture``,
        all 24 layers emitted in compile mode (position-dependent DMA
        addresses register-patched inline), one ``generate_instruction_halt``,
        then written to program DRAM and saved to disk.

        Returns
        -------
        decoder_bin_path : str
        program_sizes    : list[int]   – [total byte size of binary]
        total_flops      : list[int]   – [FLOPs per decode step]
        """
        bin_path  = self._decoder_bin_path
        meta_path = self._decoder_meta_path

        if os.path.exists(bin_path) and os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            return bin_path, meta["program_sizes"], meta["total_flops"]

        print("  compile_decoder: emitting single-binary decoder (Option B) ...",
              flush=True)
        t0 = time.time()

        # Open ONE global capture and enable compile mode so all inner
        # start_capture / stop_capture / _exec_captured calls become no-ops.
        self._primitive_cache.clear()
        self._rec_cache.clear()
        self.clear_capture_buffer()
        # Directly set capture state (bypass the overridden start_capture
        # which would no-op because _compile_mode isn't set yet).
        self.capture_buffer = []
        self.capture_count  = 0
        self.is_capture_on  = True
        self._compile_mode  = True

        # Old S7 shared full-attn flash subroutine used the removed
        # flash_attention_core/gpr_bucket_idx path. Keep full attention inline so
        # every call goes through flash_attention_core_cached -> unified_attention_core.
        self._s7_full_attn_active = False
        self._s7_flash_call_sites = []
        s7_enabled = False
        if s7_enabled:
            self._isa_reg_counter = 4
            self._s7_gpr_ret_id = self.alloc_isa_reg()   # reg 4
            self._s7_gpr_bucket = self.alloc_isa_reg()   # reg 5
            self._s7_program_base = self.get_program_dram_addr()
            self._s7_num_buckets = self.max_context_aligned // UE_VECTOR_SIZE
            self._s7_full_attn_active = True
            self.generate_instruction_add_set(self._s7_gpr_bucket,
                                               self._s7_num_buckets)

        dummy_tok = torch.tensor([0], dtype=torch.long)
        with _quiet():
            _forward_range(self, dummy_tok, pos_start=0, zero_s=False)

            # On-chip LM head (M=1 GEMV): logits = final_norm @ embedᵀ + penalty,
            # argmaxed ON CHIP (write_back_disable=True → no logit readback; the
            # host reads only the winning index via get_arg_max_index).  The
            # penalty C bias (PENALTY_BIAS_DRAM, bias_mode="broadcast_N") is zero
            # by default = bit-identical greedy; the host refreshes it per
            # penalized decode step.  M=1 ⇒ the fast quantized GEMV path (same as
            # llama3.2/qwen3).  Prefill replays this program per token, so its
            # last-token argmax is the first generated token.
            self.quantized_matmat_core(
                M=1, K=self.hidden_size, N=self.VOCAB,
                A_DRAM_ADDR=self._decoder_final_norm_dram,
                B_DRAM_ADDR=self._lm_head_data_dram,
                OUTPUT_DRAM_ADDR=self.LOGITS_DRAM,
                SCALE_DRAM_ADDR=self._lm_head_scale_dram,
                data_type=TYPE.IF4,
                C_DRAM_ADDR=self.PENALTY_BIAS_DRAM, bias_mode="broadcast_N",
                write_back_disable=True)

        self.generate_instruction_halt()

        self._s7_full_attn_active = False

        self._compile_mode = False
        self.is_capture_on = False

        # Write the compiled binary to program DRAM and record addr + size.
        prog_addr = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(prog_addr)
        prog_size = self.get_capture_instruction_size_bytes()
        self.allocate_program_dram(prog_size)
        self._decoder_prog_addr = prog_addr
        self._decoder_prog_size = prog_size

        # Persist to disk (raw instruction bytes + Gemma3-style JSON).
        bin_bytes = bytearray()
        for inst in self.capture_buffer:
            bin_bytes.extend(inst.get_bytes())
        n_insts = len(self.capture_buffer)
        self.clear_capture_buffer()

        flops = _theoretical_flops_per_step(self, 1)
        meta = {
            "instruction_counts": [n_insts],
            "program_sizes":      [len(bin_bytes)],
            "total_flops":        [int(flops)],
        }
        with open(bin_path, "wb") as f:
            f.write(bin_bytes)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=0)

        bin_mb  = os.path.getsize(bin_path) / (1024 * 1024)
        meta_kb = os.path.getsize(meta_path) / 1024
        print(f"  compile_decoder: {n_insts} instructions, "
              f"{bin_mb:.1f} MB bin + {meta_kb:.0f} KB meta "
              f"({time.time()-t0:.1f}s).")
        return bin_path, [len(bin_bytes)], [int(flops)]

    def load_instructions(self, bin_path: str) -> tuple:
        """Load a compiled decoder binary into FPGA program DRAM.

        Reads the raw instruction bytes from disk, writes them to a fresh
        program DRAM segment, and records ``_decoder_prog_addr`` so
        ``run_decoder`` can trigger execution.  Follows Gemma3's convention:
        returns ``(base_addr, total_byte_size)``.

        Also re-runs compile mode for one step if the in-memory
        ``_decoder_prog_addr`` is 0 (first call after a fresh engine init).
        """
        meta_path = self._decoder_meta_path
        if self._decoder_prog_addr != 0:
            # Already compiled into program DRAM this session.
            return self._decoder_prog_addr, self._decoder_prog_size

        with open(meta_path) as f:
            meta = json.load(f)
        with open(bin_path, "rb") as f:
            bin_bytes = f.read()

        prog_addr = self.get_program_dram_addr()
        # Write raw bytes to program DRAM via a temporary capture.
        insts = _bytes_to_insts(bin_bytes)
        self.capture_buffer = list(insts)
        self.capture_count  = len(insts)
        self.write_captured_instructions_to_dram(prog_addr)
        prog_size = len(bin_bytes)
        self.allocate_program_dram(prog_size)
        self._decoder_prog_addr = prog_addr
        self._decoder_prog_size = prog_size
        self.clear_capture_buffer()

        # Restore tensor DRAM layout so _decoder_X_dram / _decoder_final_norm_dram
        # point to the correct addresses (same allocation sequence as compile).
        if self._decoder_X_dram == 0 or self._decoder_final_norm_dram == 0:
            self._compile_mode = True
            self.capture_buffer = []; self.capture_count = 0; self.is_capture_on = True
            with _quiet():
                dummy_tok = torch.tensor([0], dtype=torch.long)
                _forward_range(self, dummy_tok, pos_start=0, zero_s=False)
            self._compile_mode = False; self.is_capture_on = False
            self.clear_capture_buffer()

        return prog_addr, prog_size

    def run_decoder(self, tokenizer, first_token_id: int,
                    max_new_tokens: int = 128,
                    temperature: float = 0.0,
                    top_k: int = 0,
                    verbose: bool = True,
                    one_shot: bool = True) -> dict:
        """Run the decode loop.

        When `one_shot=True` (default), each token executes as ONE FPGA
        trigger from the precompiled binary at self._decoder_prog_addr.  Host
        only provides per-step input data (X embedding, causal bias, ISA
        position registers) and reads FINAL_NORM_DRAM afterward.

        When `one_shot=False`, falls back to the per-layer `_forward_range`
        path with host-captured α/β/k uploads — slower (~1.5 s/step) but
        useful as a correctness reference while validating new pipelines.
        """
        eot      = self._cfg["model"].get("end_of_turn_token_id", None)
        flops    = _theoretical_flops_per_step(self, 1)
        H        = self.hidden_size
        T_al     = self.max_context_aligned
        row_byt  = self.full_head_dim * BF16
        rope_byt = self.full_rotary_dim * BF16
        X_DRAM   = self._decoder_X_dram
        FN_DRAM  = self._decoder_final_norm_dram
        embed_w  = self._embed_weight.to(torch.float32)

        out_tokens  = [first_token_id]
        total_dt    = 0.0
        total_hw_us = 0.0
        step_dts: list = []   # per-token wall time → peak (1st token) + average

        # On-FPGA penalty setup: bind the tokenizer (structural scan needs it),
        # reset the per-vocab C bias to zero (clean state across generate calls),
        # and precompute the structural-exemption set ONCE here (a full vocab
        # scan — not mid-decode).  Penalty off ⇒ bias stays zero ⇒ pure greedy.
        self.tokenizer = tokenizer
        self.dma_write(DMA_DEVICE_H2C, self.PENALTY_BIAS_DRAM,
                       _bf(torch.zeros(self.VOCAB, dtype=torch.bfloat16)),
                       self.VOCAB * BF16)
        if verbose:
            if self.fpga_penalty:
                _n_struct = len(self._structural_token_ids())
                print(f"  penalty ON  (alpha={self.pen_alpha} cap={self.pen_cap} "
                      f"window={self.rep_window} greedy_until={self.greedy_until} "
                      f"ban_count={self.pen_ban_count} loop_run={self.pen_loop_run}; "
                      f"{_n_struct} structural tokens exempt)")
            else:
                print("  penalty OFF (pure greedy)")

        if verbose:
            print()
            print("  " + "-" * 74)
            mode_str = "one-shot" if one_shot else "per-layer"
            print(f"  DECODE   (T=1 {mode_str}, up to {max_new_tokens} tokens)")
            print("  " + "-" * 74)
            # Stream generated tokens inline as they're produced.  The first
            # token came from prefill logits; subsequent tokens append below.
            first_txt = tokenizer.decode([first_token_id], skip_special_tokens=False)
            print(f"  Generated: {first_txt}", end="", flush=True)

        # §8a: live decode status bar pinned to the bottom terminal row via an
        # ANSI scroll region (TTY only; piped/redirected runs stay clean).
        _use_status = verbose and sys.stdout.isatty()
        _decode_t0 = time.time()

        def _status_setup():
            rows = shutil.get_terminal_size().lines
            sys.stdout.write(f"\033[1;{rows-1}r")
            sys.stdout.write(f"\033[{rows-1};1H"); sys.stdout.flush()

        def _status_update():
            rows = shutil.get_terminal_size().lines
            nn = len(out_tokens) - 1
            el = time.time() - _decode_t0
            sys.stdout.write("\0337")
            sys.stdout.write(f"\033[{rows};1H\033[2K")
            sys.stdout.write(f" decoding… {nn} tokens  {el:.1f}s  "
                             f"{nn/max(el,1e-9):.2f} tok/s")
            sys.stdout.write("\0338"); sys.stdout.flush()

        def _status_teardown():
            rows = shutil.get_terminal_size().lines
            sys.stdout.write("\033[r")
            sys.stdout.write(f"\033[{rows};1H\033[2K"); sys.stdout.flush()

        if _use_status:
            _status_setup()

        for step in range(1, max_new_tokens):
            if eot is not None and out_tokens[-1] == eot:
                break
            cur_tok = out_tokens[-1]
            pos     = self._cache_pos
            end_pos = pos + 1

            t0 = time.time()
            self._step_hw_latency_us = 0.0

            # Suppress user_dma_core chatter (capture start/stop, DRAM write
            # status lines) emitted by the two isa_add_set_core mini-programs
            # each decode step.  The outer `generate()` doesn't wrap the decode
            # loop in _quiet(), so we do it here at the per-step granularity.
            with _quiet():
                if one_shot:
                    # (1) Causal bias: 0 for j<=i and j<=end_pos, -inf elsewhere.
                    bias_host = torch.full((T_al, T_al), float("-inf"),
                                           dtype=torch.bfloat16)
                    bias_host.masked_fill_(
                        torch.tril(torch.ones(T_al, T_al, dtype=torch.bool)), 0.0)
                    bias_host[:, end_pos:] = float("-inf")
                    self.dma_write(DMA_DEVICE_H2C, self._bias_dram,
                                   _bf(bias_host), T_al * T_al * BF16)

                    # (2) Embedding → X_DRAM.
                    x_host = embed_w[cur_tok].unsqueeze(0)
                    self.reset_tensor_dram_addr()
                    self.dma_write(DMA_DEVICE_H2C, X_DRAM, _bf(x_host), H * BF16)

                    # (3) Set ISA registers (per-token position offsets).
                    self.isa_add_set_core(self._ISA_POS_K_REG,    ue_35bit_addr_shifter(pos * row_byt))
                    self.isa_add_set_core(self._ISA_POS_ROPE_REG, ue_35bit_addr_shifter(pos * rope_byt))

                    # (3.5) Optional on-FPGA repetition penalty: refresh the
                    # per-vocab C bias before the trigger. It remains zero for
                    # the default pure-greedy path.
                    if self.fpga_penalty and step >= self.greedy_until:
                        self._write_penalty_bias(out_tokens)

                    # (4) ONE FPGA trigger: 24 layers + final_norm + LM head, with
                    # on-chip argmax of (logits + penalty bias); write_back_disable
                    # ⇒ no logit readback.
                    self.start_execute_from_dram(self._decoder_prog_addr)
                    # Sleep 1 ms between polls — burns ~1000 PCIe reads per step
                    # instead of ~240 k, freeing the CPU core without changing
                    # wall time (FPGA queue_busy is independent of polling rate).
                    t_wait = time.time()
                    while self.is_queue_busy():
                        time.sleep(0.001)
                        if time.time() - t_wait >= 300.0:
                            print("  wait_queue timed out")
                            break
                    self._step_hw_latency_us = self.report_latency_in_us()

                    # (5) Read ONLY the winning token index (HW argmax).
                    nxt = self.get_arg_max_index(rank=1)
                else:
                    # Per-layer reference path (one_shot=False): host LM head +
                    # host argmax (numeric reference only; not a shipping path).
                    self._exec_idx = 0
                    dbg_tok = torch.tensor([cur_tok], dtype=torch.long)
                    logits_row_all = _forward_range(
                        self, dbg_tok, pos_start=pos, zero_s=False)
                    nxt = int(torch.argmax(logits_row_all[-1]).item())
            step_dt = time.time() - t0
            step_dts.append(step_dt)
            step_hw_us = self._step_hw_latency_us
            total_dt += step_dt
            total_hw_us += step_hw_us
            out_tokens.append(nxt)
            self._cache_pos += 1

            if verbose:
                tok_txt = tokenizer.decode([nxt], skip_special_tokens=False)
                print(tok_txt, end="", flush=True)
                if _use_status:
                    _status_update()

        # §8a: tear down the scroll region on BOTH exits (EOT break + natural
        # max-length end) so it never leaks into the shell.
        if _use_status:
            _status_teardown()

        n = max(len(out_tokens) - 1, 1)
        avg_hw_us  = total_hw_us / n if total_hw_us > 0 else 0.0
        avg_gflops = (flops / 1e9 / max(avg_hw_us * 1e-6, 1e-9)
                      if avg_hw_us > 0 else 0.0)
        if verbose and n > 0:
            # Close the streaming line before the summary block.
            print()
            print()
            print(f"  Decode summary: {n} tokens in {total_dt:.2f}s wall "
                  f"({n/max(total_dt,1e-9):.2f} tok/s, "
                  f"{total_dt/n:.2f}s/token)")
            if step_dts:
                _peak = 1.0 / max(min(step_dts), 1e-9)        # fastest token (1st, smallest context)
                _avg  = n / max(total_dt, 1e-9)
                print(f"                  peak (1st token): {1.0/max(step_dts[0],1e-9):.2f} tok/s, "
                      f"fastest: {_peak:.2f} tok/s, average: {_avg:.2f} tok/s")
            if avg_hw_us > 0:
                print(f"                  HW: {flops*n/1e9:.1f} GFLOP  "
                      f"{avg_gflops:.1f} GFLOP/s")

        # §8b: clear tensor DRAM at the end of the run so the next process that
        # reads a scratch buffer before overwriting it doesn't see stale/NaN data.
        with _quiet():
            self.clear_dram()

        generated = tokenizer.decode(out_tokens, skip_special_tokens=True)
        return {
            "token_ids":      out_tokens,
            "generated_text": generated,
            "n_tokens":       n,
            "wall_time_s":    total_dt,
            "tok_per_s":      n / max(total_dt, 1e-9),
            "hw_gflops_s":    avg_gflops,
        }

    def reset_state(self) -> None:
        """Zero the S cache, the FPGA-resident conv state, the host-side
        conv state mirror, and the cache position counter.  Call before each
        fresh prompt.  (KV cache contents don't need zeroing — the effective
        mask in the bias already restricts attention to cols ≤ cache_pos.)"""
        s_elems = (self.lin_num_v_heads * self.lin_head_k_dim
                   * self.lin_head_v_dim)
        zeros_s = _bf(torch.zeros(s_elems))
        for addr in self._s_dram.values():
            self.dma_write(DMA_DEVICE_H2C, addr, zeros_s, s_elems * BF16)
        K = self._conv_K
        conv_dim = self.lin_conv_dim
        conv_state_bytes = (K - 1) * conv_dim * BF16
        zeros_conv = _bf(torch.zeros((K - 1) * conv_dim))
        for idx, addr in self._conv_state_dram.items():
            self.dma_write(DMA_DEVICE_H2C, addr, zeros_conv, conv_state_bytes)
        for idx in self._conv_state:
            self._conv_state[idx].zero_()
        self._cache_pos = 0


# ============================================================================
# Custom matvec kernels
# ============================================================================

def _emit_step4_matmul_sram_A(ue, B_DRAM: int, A_sram_addr: int,
                              out_sram_addr: int,
                              M: int, K: int, N: int) -> None:
    """Custom step-4 matmul for the Gated DeltaNet outer product, fused with
    the preceding bf16_transpose_core.

    Behavior matches matmat_mul_core(M, K, N, A_DRAM=b_pad_dram, B_DRAM=k_pad_h,
    OUTPUT_DRAM=outer_dram) for Qwen3.5-2B's recurrence shape (M=Dv=128, K=N=64,
    N=Dk=128), but:
      * reads A directly from SRAM at `A_sram_addr` (where bf16_transpose_core
        left its output at the canonical 0x2000 address) — no DMA round-trip
        through `b_pad_dram`.
      * leaves the output in SRAM at `out_sram_addr`; step 5's `eltwise_add_core`
        reads it from SRAM too, skipping the SRAM→DRAM→SRAM hop through
        `outer_dram`.

    Assumes single-chunk execution (M_chunk == M, N_chunk == N).  Verified for
    Qwen3.5-2B's (M=128, K=64, N=128): M_chunk = min(128, URAM_FULL//(K+N_chunk))
    = 128; N_chunk = min(128, URAM_NEAR_FULL//K//64*64) = 128.

    Saves 2 DMAs per head (A-load + output-store) ≈ 576 instructions/token.
    """
    bpe = 2
    # Load B into URAM_B (matches matmat_mul_core's convention).
    ue.accelerator_memory_to_sram(
        accelerator_dram_address=B_DRAM,
        sram_address=0x80000,
        element_size=N * K)
    # Emit M matvecs — A read directly from SRAM, no DMA.
    clear_en = 1
    for output_row in range(M):
        ue.start_queue_for_bf16_matvec_operation(
            max_clear_en=clear_en,
            fmax_context_addr=output_row,
            vector_sram_start_addr=A_sram_addr + output_row * K * bpe,
            matrix_sram_start_addr=0x80000,
            output_sram_wb_addr=out_sram_addr + output_row * N * bpe,
            K=K, N=N,
            bias_enable=False,
            lalu_mode=LALU_MODE.BYPASS,
            lalu_a=0, lalu_b=0)
        clear_en = 0


# ============================================================================
# DMA / capture helpers
# ============================================================================

def _bf(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.bfloat16).contiguous().cpu()

def _pad_to_vec(t: torch.Tensor) -> torch.Tensor:
    """Pad a 1-D bf16 tensor to UE_VECTOR_SIZE (64) elements."""
    assert t.ndim == 1
    n = t.numel()
    if n >= UE_VECTOR_SIZE:
        return t[:UE_VECTOR_SIZE].contiguous()
    out = torch.zeros(UE_VECTOR_SIZE, dtype=torch.bfloat16)
    out[:n] = t
    return out

def _upload(ue, t: torch.Tensor) -> int:
    nb = t.numel() * BF16
    addr = ue.allocate_params_dram(nb)
    ue.dma_write(DMA_DEVICE_H2C, addr, _bf(t), nb)
    return addr

def _upload_fp4(ue, w: torch.Tensor):
    scales, packed = quantize_fp4_64(w)
    s_bytes, d_bytes = scales.numel() * BF16, packed.numel()
    scale_addr = ue.allocate_params_dram(s_bytes)
    ue.dma_write(DMA_DEVICE_H2C, scale_addr, scales, s_bytes)
    data_addr = ue.allocate_params_dram(d_bytes)
    ue.dma_write(DMA_DEVICE_H2C, data_addr, packed, d_bytes)
    return scale_addr, data_addr

def _alloc_tensor(ue, numel: int) -> int:
    return ue.allocate_tensor_dram(numel * BF16)

def _read_bf(ue, addr: int, shape) -> torch.Tensor:
    n = 1
    for d in shape:
        n *= d
    t = torch.zeros(n, dtype=torch.bfloat16)
    ue.dma_read(DMA_DEVICE_C2H, addr, t, n * BF16)
    return t.reshape(*shape).float()

def _emit_alpha_fpga_to_slots(ue, A_RAW_DRAM: int, DT_BIAS_DRAM: int,
                              C_DRAM: int, alpha_dram: int,
                              x_scratch: int, rx_scratch: int, y_scratch: int,
                              num_heads: int) -> None:
    """Emit α-on-FPGA pipeline into the active capture.

    Computes α[h] = exp(-c[h] · softplus(x[h])) for each head, then scatters
    each value into the padded alpha_dram slot so the recurrence's
    scalar-load reads the correct value without a host round-trip.

    Inputs (pre-uploaded to params DRAM, 64-element padded bf16 vectors):
      A_RAW_DRAM   : a_raw from A_pad projection (first num_heads elems valid)
      DT_BIAS_DRAM : dt_bias constants
      C_DRAM       : c = exp(A_log) constants
      alpha_dram   : pre-zeroed slot array [num_heads, 64] bf16; position 0 of
                     each 64-element block receives α[h]

    Scratch DRAM (temporary, UE_VECTOR_SIZE elements each):
      x_scratch, rx_scratch, y_scratch

    softplus is computed via the native identity:
        softplus(x) = relu(x) + log(1 + exp(-|x|))
    using LALU's CLAMP (=ReLU with [0, +inf]) and LOG (=log·clamp(1e-3, +inf))
    modes. The add and log must be emitted separately; see _emit_log1p_add.
    """
    N = UE_VECTOR_SIZE

    # SRAM scratch layout (re-uses slots freed after each sub-op completes;
    # safe because FPGA executes sequentially and recurrence runs AFTER us).
    SA_X     = 0x00000  # x = a_raw + dt_bias                  (URAM_A)
    SB_BIAS  = 0x80000  # dt_bias constant                      (URAM_B)
    SA_RX    = 0x00200  # relu(x)                              (URAM_A)
    SA_NX    = 0x00400  # -x                                   (URAM_A)
    SB_2RX   = 0x80200  # 2·relu(x)                            (URAM_B)
    SA_ABSX  = 0x00600  # |x| = 2·relu(x) - x                  (URAM_A)
    SA_NABSX = 0x00800  # -|x|                                 (URAM_A)
    SA_E     = 0x00A00  # exp(-|x|)                            (URAM_A)
    SB_L1P   = 0x80400  # log(1 + exp(-|x|))                 (URAM_B; eltwise_add needs cross-bank operands)
    SA_SP    = 0x01400  # softplus = relu(x) + log1p           (URAM_A)
    SB_C     = 0x80800  # c = exp(A_log)                        (URAM_B)
    SA_CS    = 0x01600  # c · softplus                         (URAM_A)
    SA_Y     = 0x01800  # y = exp(-c·sp) = α                   (URAM_A)
    SA_TMP   = 0x01A00  # 1-element scatter scratch            (URAM_A)

    def _emit_exp(src_sram, dst_sram, n):
        src_type, src_addr = ue.sram_address_to_uram_address(src_sram)
        dst_type, dst_addr = ue.sram_address_to_uram_address(dst_sram)
        rs = (n + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
        ue.ue_arithmetic_op(
            BROADCAST_MODE.SCALAR_IN_REG.value, 0, 1, 0, 0,
            LALU_MODE.BYPASS.value,
            ue.float_to_bf16(0.0),
            dst_type.value, 0, dst_addr,
            URAM_WRITE_SRC.URAM_WRITE_BACK.value,
            UE_MODE.EXP, 0,
            src_addr, 0, rs,
            0, 0, 0,
        )

    def _emit_log1p_add(src_sram, dst_sram, n):
        """Compute dst = log(src + 1.0) using supported, explicit stages."""
        src_type, src_addr = ue.sram_address_to_uram_address(src_sram)
        dst_type, dst_addr = ue.sram_address_to_uram_address(dst_sram)
        rs = (n + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
        # Do not fuse ADD_BROADCAST(+1) with LALU.LOG here. On hardware that
        # path applied the +1 but dropped LOG, turning log1p(x) into 1+x and
        # causing DeltaNet alpha over-decay. Use the tested
        # matmat_mul_core(log_enable=True) path instead.
        ue.ue_arithmetic_op(
            BROADCAST_MODE.SCALAR_IN_REG.value, 0, 1,
            0, 0,
            LALU_MODE.BYPASS.value,
            ue.float_to_bf16(1.0),
            dst_type.value, 0, dst_addr,
            URAM_WRITE_SRC.URAM_WRITE_BACK.value,
            UE_MODE.ADD_BROADCAST, 0,
            src_addr, 0, rs,
            0, 0, 0,
        )
        # Spill src+1 to scratch, apply log through the tested identity-matmul
        # path, then restore relu(x), because matmat_mul_core clobbers SRAM.
        ue.sram_to_accelerator_memory(dst_sram, y_scratch, n)
        ue.matmat_mul_core(M=1, K=n, N=n,
                           A_DRAM_ADDR=y_scratch,
                           B_DRAM_ADDR=ue._identity_dram_addr,
                           OUTPUT_DRAM_ADDR=y_scratch,
                           log_enable=True)
        ue.accelerator_memory_to_sram(y_scratch, dst_sram, n)
        ue.accelerator_memory_to_sram(rx_scratch, SA_RX, n)

    # 1. Load dt_bias and a_raw; x = a_raw + dt_bias
    ue.accelerator_memory_to_sram(DT_BIAS_DRAM, SB_BIAS, N)
    ue.accelerator_memory_to_sram(A_RAW_DRAM,   SA_X,    N)
    ue.eltwise_add_core(SA_X, SB_BIAS, SA_X, N)

    # 2. relu(x) via identity matmul + LALU clamp(0, +inf).  v1.2 dropped
    # `relu_enable`; `clamp_enable=True` configures the LALU with
    # LALU_CLAMP_RELU_A=0, LALU_CLAMP_RELU_B=+inf which is exactly ReLU.
    ue.sram_to_accelerator_memory(SA_X, x_scratch, N)
    ue.matmat_mul_core(M=1, K=N, N=N,
                       A_DRAM_ADDR=x_scratch,
                       B_DRAM_ADDR=ue._identity_dram_addr,
                       OUTPUT_DRAM_ADDR=rx_scratch,
                       clamp_enable=True)
    ue.accelerator_memory_to_sram(rx_scratch, SA_RX, N)
    ue.accelerator_memory_to_sram(x_scratch,  SA_X,  N)  # matmul clobbered SRAM

    # 3. |x| = 2·relu(x) - x
    ue.broadcast_mul(scalar=-1.0, sram_start_addr=SA_X,  sram_wb_addr=SA_NX,  element_size=N)
    ue.broadcast_mul(scalar= 2.0, sram_start_addr=SA_RX, sram_wb_addr=SB_2RX, element_size=N)
    ue.eltwise_add_core(SB_2RX, SA_NX, SA_ABSX, N)

    # 4. Native softplus: softplus(x) = relu(x) + log(1 + exp(-|x|)).
    #    The +1 and LOG are intentionally separate; see _emit_log1p_add.
    ue.broadcast_mul(scalar=-1.0, sram_start_addr=SA_ABSX, sram_wb_addr=SA_NABSX, element_size=N)
    _emit_exp(SA_NABSX, SA_E, N)              # exp(-|x|)
    _emit_log1p_add(SA_E, SB_L1P, N)          # log(1 + exp(-|x|)) → URAM_B
    ue.eltwise_add_core(SA_RX, SB_L1P, SA_SP, N)  # softplus = relu(x) + log1p

    # 5. α = exp(-c · softplus)   (was: y = 1/α = exp(+c · softplus))
    # Recurrence consumes α directly via BF16_DOT_PRODUCT extraction + LALU RELU
    # — RELU(α)=α since α>0, populating LALU register without any 1/x recovery.
    ue.accelerator_memory_to_sram(C_DRAM, SB_C, N)
    ue.eltwise_mul_core(SA_SP, SB_C, SA_CS, N)
    ue.broadcast_mul(scalar=-1.0, sram_start_addr=SA_CS,
                     sram_wb_addr=SA_CS, element_size=N)
    _emit_exp(SA_CS, SA_Y, N)

    # 6. Write y to DRAM (aligned 64-element block).
    ue.sram_to_accelerator_memory(SA_Y, y_scratch, N)

    # 7. Spread y [N] → alpha_dram [num_heads, N] via matmul with a sparse
    #    selector `ue._alpha_spreader_dram` of shape [N, num_heads*N] where
    #    S[h, h*N] = 1 and everything else is 0.  Result:
    #        out[0, h*N]     = sum_k y[k] * S[k, h*N] = y[h]
    #        out[0, h*N + t] = 0   for t > 0
    #    which matches the recurrence's per-head [N]-block RSQRT load layout.
    #
    #    Per-element scatter via `accelerator_memory_to_sram(y_scratch+h*BF16, ...)`
    #    cannot work because DMA requires 128-byte aligned source addresses
    #    (h*2 bytes is unaligned for h>0 and silently reads y_scratch[0]).
    ue.matmat_mul_core(M=1, K=N, N=num_heads * N,
                       A_DRAM_ADDR=y_scratch,
                       B_DRAM_ADDR=ue._alpha_spreader_dram,
                       OUTPUT_DRAM_ADDR=alpha_dram,
                       is_B_quantized=False)


def _emit_inv_beta_fpga_to_slots(ue, BETA_RAW_DRAM: int, beta_dram: int,
                                 y_scratch: int, ones_dram: int,
                                 num_heads: int) -> None:
    """Emit β-on-FPGA pipeline: compute 1/β = 1 + exp(-raw_beta) and scatter
    each value into the padded beta_dram slot so the recurrence scalar-load
    reads the correct 1/β without a host round-trip.

    1/β = sigmoid(raw_beta)^-1 = 1 + exp(-raw_beta).

    SRAM scratch layout (does not conflict with α pipeline at 0x00000..0x01A00):
      SA_BRAW  = 0x02000  raw beta values (num_heads valid, rest zero)
      SA_NEG_B = 0x02200  -raw_beta
      SA_EXPNB = 0x02400  exp(-raw_beta)
      SB_ONES  = 0x82000  constant 1.0 vector
      SA_INV_B = 0x02600  1 + exp(-raw_beta) = 1/β
      SA_TMP   = 0x02800  1-element scatter scratch
    """
    N = UE_VECTOR_SIZE

    SA_BRAW  = 0x02000
    SA_NEG_B = 0x02200
    SA_EXPNB = 0x02400
    SB_ONES  = 0x82000
    SA_INV_B = 0x02600
    SA_TMP   = 0x02800

    def _emit_exp(src_sram, dst_sram, n):
        src_type, src_addr = ue.sram_address_to_uram_address(src_sram)
        dst_type, dst_addr = ue.sram_address_to_uram_address(dst_sram)
        rs = (n + N - 1) // N
        ue.ue_arithmetic_op(
            BROADCAST_MODE.SCALAR_IN_REG.value, 0, 1, 0, 0,
            LALU_MODE.BYPASS.value,
            ue.float_to_bf16(0.0),
            dst_type.value, 0, dst_addr,
            URAM_WRITE_SRC.URAM_WRITE_BACK.value,
            UE_MODE.EXP, 0,
            src_addr, 0, rs,
            0, 0, 0,
        )

    # 1. Load raw beta and ones constant.
    ue.accelerator_memory_to_sram(BETA_RAW_DRAM, SA_BRAW, N)
    ue.accelerator_memory_to_sram(ones_dram,     SB_ONES, N)

    # 2. -raw_beta
    ue.broadcast_mul(scalar=-1.0, sram_start_addr=SA_BRAW, sram_wb_addr=SA_NEG_B,
                     element_size=N)

    # 3. exp(-raw_beta)
    _emit_exp(SA_NEG_B, SA_EXPNB, N)

    # 4. 1 + exp(-raw_beta) = 1/β
    ue.eltwise_add_core(SA_EXPNB, SB_ONES, SA_INV_B, N)

    # 5. Write 1/β to DRAM (aligned).
    ue.sram_to_accelerator_memory(SA_INV_B, y_scratch, N)

    # 6. Spread via matmul with the same spreader as α (identical M=N=num_heads
    #    pattern).  See `_emit_alpha_fpga_to_slots` for the rationale.
    ue.matmat_mul_core(M=1, K=N, N=num_heads * N,
                       A_DRAM_ADDR=y_scratch,
                       B_DRAM_ADDR=ue._alpha_spreader_dram,
                       OUTPUT_DRAM_ADDR=beta_dram,
                       is_B_quantized=False)

    # 7. Per-head invert each slot in-place: 1/β at slot[h, 0] → β at slot[h, 0].
    # The slot has 1/β at pos 0, zeros elsewhere (post-spreader).
    #   (a) RMS over slot gives sum(x²) = (1/β)²;
    #   (b) LALU_RSQRT(scalar=1.0) gives β = 1/sqrt((1/β)²) — held in LALU register;
    #   (c) broadcast_mul reads a pre-loaded e_0=[1,0,...,0] vector and multiplies
    #       it by the LALU value, materialising [β, 0, ..., 0] in SRAM;
    #   (d) DMA that back to the slot — slot now has β at position 0.
    # After this the recurrence consumes β directly via dot_product+RELU,
    # uniform with α (no RSQRT recovery in the recurrence body).
    SA_BETA_INV = 0x02A00
    SA_E0_LOCAL = 0x02C00
    inv_type, inv_addr = ue.sram_address_to_uram_address(SA_BETA_INV)
    e0_type, e0_addr = ue.sram_address_to_uram_address(SA_E0_LOCAL)
    # Load e_0 once for the whole β pipeline call (reused across 16 heads).
    ue.accelerator_memory_to_sram(ue._e0_dram, SA_E0_LOCAL, N)
    for h in range(num_heads):
        slot_addr = beta_dram + h * N * 2
        ue.accelerator_memory_to_sram(slot_addr, SA_BETA_INV, N)
        # RMS+RSQRT — β = 1/(1/β) into LALU register, no URAM writeback.
        ue.ue_arithmetic_op(
            0, 0, 1, 0, 0,
            LALU_MODE.MODE_RSQRT.value,
            ue.float_to_bf19(1.0),
            inv_type.value, 0, 0,
            URAM_WRITE_SRC.URAM_WB_DISABLE.value,
            UE_MODE.RMS, 0, inv_addr, 0, 1, 0, 0, 0,
        )
        # broadcast_mul: e_0 (URAM_A) * β (LALU) → [β,0,...,0] at SA_BETA_INV.
        ue.start_queue_broadcast(
            UE_MODE.MUL_BROADCAST, BROADCAST_MODE.LALU_RESULT,
            SA_E0_LOCAL, SA_BETA_INV, N)
        ue.sram_to_accelerator_memory(SA_BETA_INV, slot_addr, N)


def _emit_k_pad_fpga(ue, K_L2_DRAM: int, k_pad_dram: int,
                     num_heads: int, Dk: int, N: int) -> None:
    """Emit k-pad-on-FPGA pipeline: scatter K_L2[h, dk] into k_pad_dram col-0
    slots at stride N=64 bf16 per element.

    k_pad_dram layout: [num_heads, Dk, N] bf16, col-0 at element 0 of each
    N-element row.  Only col 0 is written; the rest stay at zero (pre-zeroed
    in prepare_inference).

    Same alignment gotcha as α/β — can't do 1-element unaligned DMA.  Per-head
    we run matmul(k_h [1, Dk], k_spreader [Dk, Dk*N]) = [1, Dk*N], which places
    each k[h, dk] at stride-N position with zeros in between.  K_L2_DRAM is
    itself contiguous and 128-byte-aligned at head boundaries (h*Dk*BF16 =
    h*256, multiple of 128).
    """
    head_stride_bytes  = Dk * BF16           # 256 bytes for Dk=128
    out_stride_bytes   = Dk * N * BF16       # 16384 bytes for Dk=128, N=64
    for h in range(num_heads):
        ue.matmat_mul_core(M=1, K=Dk, N=Dk * N,
                           A_DRAM_ADDR=K_L2_DRAM + h * head_stride_bytes,
                           B_DRAM_ADDR=ue._k_spreader_dram,
                           OUTPUT_DRAM_ADDR=k_pad_dram + h * out_stride_bytes,
                           is_B_quantized=False)


def _emit_rope_split_64(
        ue,
        src_dram: int,
        out_dram: int,
        cos_pad_dram: int,
        neg_sin_pad_dram: int,
        sin_hi_pad_dram: int,
        rot_dim: int,
        pass_len: int,
        load_cos_sin: bool = True,
) -> None:
    """Manual HF RoPE for rot_dim=64 without rope_core_dram.

    rope_core_dram is unusable for N=64 because internally it accesses
    x_sram_addr + half_N*2 = addr+64, which violates the 128-byte SRAM
    alignment required by eltwise_mul_core.

    Layout:  half_rot = rot_dim//2 = 32 elements = 64 bytes.
    Stores x_lo and x_hi each in a 128-byte-aligned 64-element SRAM slot,
    with the upper 32 positions left as garbage (multiplied by zero-padded
    cos/sin, so they vanish in the result).

    SRAM map (all 128-byte aligned):
      URAM_A: SA_X_LO=0x40000  SA_X_HI=0x40080
              SA_OUT_LO=0x40100 SA_OUT_HI=0x40180
              SA_TMP_A=0x40200  SA_PASS=0x40280
      URAM_B: SB_COS=0x80000   SB_NEG_SIN=0x80080
              SB_SIN_HI=0x80100 SB_TMP_B=0x80180

    cos_pad, neg_sin_pad, sin_hi_pad are each rot_dim bf16 elements with
    only positions [0:half_rot] populated (the rest are zero).
    """
    half_rot = rot_dim // 2
    SA_X_LO   = 0x40000
    SA_X_HI   = 0x40080
    SA_OUT_LO  = 0x40100
    SA_OUT_HI  = 0x40180
    SA_TMP_A   = 0x40200
    SA_PASS    = 0x40280
    SB_COS     = 0x80000
    SB_NEG_SIN = 0x80080
    SB_SIN_HI  = 0x80100
    SB_TMP_B   = 0x80180  # bank B temp for eltwise_mul output (add needs A+B inputs)

    if load_cos_sin:
        if getattr(ue, '_compile_mode', False):
            # Register-patch each cos/sin load: TMP_REG = POS_ROPE_REG + base_addr,
            # then overwrite the following DMA instruction's DRAM addr with TMP_REG.
            ue.generate_instruction_add_imm(ue._ISA_POS_ROPE_REG, ue_35bit_addr_shifter(cos_pad_dram),     ue._ISA_TMP_REG)
            ue.accelerator_memory_to_sram(cos_pad_dram,     SB_COS,     rot_dim)
            ue.overwrite_instruction_with_general_register(ue._ISA_TMP_REG)
            ue.generate_instruction_add_imm(ue._ISA_POS_ROPE_REG, ue_35bit_addr_shifter(neg_sin_pad_dram), ue._ISA_TMP_REG)
            ue.accelerator_memory_to_sram(neg_sin_pad_dram, SB_NEG_SIN, rot_dim)
            ue.overwrite_instruction_with_general_register(ue._ISA_TMP_REG)
            ue.generate_instruction_add_imm(ue._ISA_POS_ROPE_REG, ue_35bit_addr_shifter(sin_hi_pad_dram),  ue._ISA_TMP_REG)
            ue.accelerator_memory_to_sram(sin_hi_pad_dram,  SB_SIN_HI,  rot_dim)
            ue.overwrite_instruction_with_general_register(ue._ISA_TMP_REG)
        else:
            ue.accelerator_memory_to_sram(cos_pad_dram,     SB_COS,     rot_dim)
            ue.accelerator_memory_to_sram(neg_sin_pad_dram, SB_NEG_SIN, rot_dim)
            ue.accelerator_memory_to_sram(sin_hi_pad_dram,  SB_SIN_HI,  rot_dim)

    # Load x_lo (first half_rot elements) and x_hi (second half_rot elements).
    # Each occupies a full 128-byte aligned slot; upper 32 positions are
    # garbage but cancel out when multiplied with zero-padded cos/sin.
    ue.accelerator_memory_to_sram(src_dram,                    SA_X_LO, half_rot)
    ue.accelerator_memory_to_sram(src_dram + half_rot * BF16,  SA_X_HI, half_rot)

    # out_lo = x_lo*cos_lo + x_hi*neg_sin_lo
    ue.eltwise_mul_core(SA_X_LO, SB_COS,     SB_TMP_B,  rot_dim)
    ue.eltwise_mul_core(SA_X_HI, SB_NEG_SIN, SA_TMP_A,  rot_dim)
    ue.eltwise_add_core(SA_TMP_A, SB_TMP_B,  SA_OUT_LO, rot_dim)

    # out_hi = x_hi*cos_hi + x_lo*sin_hi  (cos_lo==cos_hi for HF repeated-half)
    ue.eltwise_mul_core(SA_X_HI, SB_COS,    SB_TMP_B,  rot_dim)
    ue.eltwise_mul_core(SA_X_LO, SB_SIN_HI, SA_TMP_A,  rot_dim)
    ue.eltwise_add_core(SA_TMP_A, SB_TMP_B, SA_OUT_HI, rot_dim)

    # Write first half_rot → out_dram[0:half_rot]
    ue.sram_to_accelerator_memory(SA_OUT_LO, out_dram,                    half_rot)
    # Write second half_rot → out_dram[half_rot:rot_dim]
    ue.sram_to_accelerator_memory(SA_OUT_HI, out_dram + half_rot * BF16,  half_rot)

    # Passthrough: x[rot_dim:] → out_dram[rot_dim:]
    if pass_len > 0:
        ue.accelerator_memory_to_sram(src_dram + rot_dim * BF16, SA_PASS, pass_len)
        ue.sram_to_accelerator_memory(SA_PASS, out_dram + rot_dim * BF16, pass_len)


def _exec_captured(ue, timeout: float = 60.0):
    if getattr(ue, '_compile_mode', False):
        # In compile mode: leave instructions in the global capture buffer.
        # Do NOT halt, flush, or execute — compile_decoder does that at the end.
        ue._current_capture_has_patches = False
        ue._pending_scalar_patches.clear()
        return
    ue.generate_instruction_halt()
    ue.stop_capture()
    idx = ue._exec_idx
    ue._exec_idx += 1
    cached_addr = ue._exec_cache.get(idx)
    if cached_addr is None:
        # First sighting of this capture slot — serialize, upload, record.
        # If the capture carries scalar patches (recurrence record OR replay
        # both queue them), mutate Instruction.words before serialization so
        # the written DRAM bytes reflect this step's α/β/k exactly.
        if ue._current_capture_has_patches:
            _apply_word_patches(ue)
        prog_addr = ue.get_program_dram_addr()
        ue.write_captured_instructions_to_dram(prog_addr)
        ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())
        ue._exec_cache[idx] = prog_addr
        if ue._current_capture_has_patches:
            # Snapshot the bytes + precomputed patch offsets/indices.  From
            # the second step onward, per-step scalar patching runs in
            # vectorized numpy over this cached bytearray — skipping the
            # Instruction.get_bytes() serialization loop (~60 ms/capture).
            ue._patched_bytes_cache[idx] = _build_patched_bytes_cache(ue)
    elif ue._current_capture_has_patches:
        _apply_scalar_patches_and_write(ue, idx, cached_addr)
        prog_addr = cached_addr
    else:
        # Byte-identical to the image already in DRAM — skip the DMA write.
        prog_addr = cached_addr
    ue.start_execute_from_dram(prog_addr)
    ue.wait_queue(timeout)
    # HW latency counter (µs) accumulates over all captures in a step —
    # generate() divides theoretical FLOPs by it to report GFLOP/s.
    ue._step_hw_latency_us += ue.report_latency_in_us()
    ue.clear_capture_buffer()
    ue._current_capture_has_patches = False
    ue._pending_scalar_patches.clear()


def _apply_word_patches(ue) -> None:
    """Mutate Instruction.words for every pending scalar patch so a
    subsequent write_captured_instructions_to_dram emits correct bytes.
    Only called on the FIRST write of a patched capture (i.e., the first
    step that flows through this capture); subsequent steps bypass
    Instruction serialization entirely via _apply_scalar_patches_and_write."""
    _f2bf16 = ue.float_to_bf16
    for _, start_inst_idx, patches, alpha, beta, k in ue._pending_scalar_patches:
        alpha_np = alpha.detach().float().cpu().numpy()
        beta_np  = beta.detach().float().cpu().numpy()
        k_np     = k.detach().float().cpu().numpy()
        for inst_idx, kind, t, h, i in patches:
            if kind == 'alpha':
                val = float(alpha_np[t, h])
            elif kind == 'neg_beta':
                val = -float(beta_np[t, h])
            elif kind == 'beta':
                val = float(beta_np[t, h])
            else:  # 'k'
                val = float(k_np[t, h, i])
            enc = _f2bf16(val) & 0x1FFFFF
            w = ue.capture_buffer[start_inst_idx + inst_idx].words
            w[5] = (w[5] & ~(0x3F << 26)) | ((enc & 0x3F) << 26)
            w[6] = (w[6] & ~0x7FFF) | ((enc >> 6) & 0x7FFF)


def _build_patched_bytes_cache(ue) -> dict:
    """Called on the FIRST execute of a patched capture.  Serializes the
    current capture_buffer into a single bytearray, and prebuilds numpy
    index arrays so per-step scalar patching runs in vectorized numpy (vs
    ~40 µs/scalar Python loops).  The scalar field is bf16 (16 bits) spread
    across three bytes at byte offsets (+23, +24, +25) of each instruction:
        byte+23 bits [2:8] ← bf16[0:6]
        byte+24 bits [0:8] ← bf16[6:14]
        byte+25 bits [0:2] ← bf16[14:16]
    """
    inst_bytes = bytearray()
    for inst in ue.capture_buffer:
        inst_bytes.extend(inst.get_bytes())

    # Concatenate patches from every recurrence replay in this capture.
    all_byte_offs, all_kinds, all_t, all_h, all_i = [], [], [], [], []
    kind_to_int = {'alpha': 0, 'neg_beta': 1, 'beta': 2, 'k': 3}
    for _, start_inst_idx, patches, _, _, _ in ue._pending_scalar_patches:
        for inst_idx, scalar_kind, t, h, i in patches:
            all_byte_offs.append((start_inst_idx + inst_idx) * 32)
            all_kinds.append(kind_to_int[scalar_kind])
            all_t.append(t)
            all_h.append(h)
            all_i.append(i if i >= 0 else 0)
    return {
        "bytes":     inst_bytes,
        "byte_offs": np.asarray(all_byte_offs, dtype=np.int64),
        "kinds":     np.asarray(all_kinds, dtype=np.int8),
        "tt":        np.asarray(all_t, dtype=np.int64),
        "hh":        np.asarray(all_h, dtype=np.int64),
        "ii":        np.asarray(all_i, dtype=np.int64),
    }


def _apply_scalar_patches_and_write(ue, idx: int, prog_addr: int) -> None:
    """Apply α/β/k scalar mutations in-place on the cached byte buffer for
    capture `idx`, then DMA the buffer to DRAM.  Vectorized via numpy —
    ~37K scalar writes per step complete in a few ms instead of ~60 ms each
    in a Python loop.

    Scalar encoding in each instruction's 32-byte record (little-endian):
      byte +23, bits [2:8]  ← bf16 bits [0:6]
      byte +24, bits [0:8]  ← bf16 bits [6:14]
      byte +25, bits [0:2]  ← bf16 bits [14:16]
    """
    cache = ue._patched_bytes_cache.get(idx)
    if cache is None:
        ue.write_captured_instructions_to_dram(prog_addr)
        return
    buf: bytearray = cache["bytes"]
    byte_offs = cache["byte_offs"]
    kinds     = cache["kinds"]
    tt        = cache["tt"]
    hh        = cache["hh"]
    ii        = cache["ii"]

    pending = ue._pending_scalar_patches
    if len(pending) != 1:
        raise RuntimeError(
            f"unexpected number of recurrence replays in capture {idx}: "
            f"{len(pending)}; expected exactly 1")
    _, _, _, alpha_host, beta_host, k_host = pending[0]
    alpha_np = alpha_host.detach().float().cpu().numpy()
    beta_np  = beta_host.detach().float().cpu().numpy()
    k_np     = k_host.detach().float().cpu().numpy()

    # Gather per-patch values in one vectorized pass.
    vals = np.empty(byte_offs.shape, dtype=np.float32)
    m_alpha    = (kinds == 0)
    m_neg_beta = (kinds == 1)
    m_beta     = (kinds == 2)
    m_k        = (kinds == 3)
    vals[m_alpha]    = alpha_np[tt[m_alpha], hh[m_alpha]]
    vals[m_neg_beta] = -beta_np[tt[m_neg_beta], hh[m_neg_beta]]
    vals[m_beta]     = beta_np[tt[m_beta], hh[m_beta]]
    vals[m_k]        = k_np[tt[m_k], hh[m_k], ii[m_k]]

    # fp32 → bf16 via round-to-nearest-even on the high 16 bits.
    as_u32  = vals.view(np.uint32)
    rounded = as_u32 + ((as_u32 >> 16) & 1) + 0x7FFF
    enc16   = (rounded >> 16).astype(np.uint16)
    lo6  = (enc16 & 0x3F).astype(np.uint8)
    mid8 = ((enc16 >> 6) & 0xFF).astype(np.uint8)
    hi2  = ((enc16 >> 14) & 0x03).astype(np.uint8)

    # Scatter bf16 bits into bytes +23/+24/+25 of each instruction.
    buf_arr = np.frombuffer(buf, dtype=np.uint8)
    b23_off = byte_offs + 23
    b24_off = byte_offs + 24
    b25_off = byte_offs + 25
    buf_arr[b23_off] = (buf_arr[b23_off] & 0x03) | (lo6 << 2)
    buf_arr[b24_off] = mid8
    buf_arr[b25_off] = (buf_arr[b25_off] & 0xFC) | hi2

    ue.dma_write(DMA_DEVICE_H2C, prog_addr, bytes(buf), len(buf))


def _cached_emit(ue, key, emit_fn) -> None:
    """Emit via `emit_fn`, caching the instruction span it appends to
    `ue.capture_buffer` keyed by `key`.  On subsequent calls with the same
    key, splice the cached Instruction list into the capture buffer instead
    of running emit_fn.  emit_fn must only append to capture_buffer (no
    other side effects).  Used to eliminate the ~50 µs/instruction Python
    emission cost from decode-step hot paths."""
    cached = ue._primitive_cache.get(key)
    if cached is None:
        start = ue.capture_count
        emit_fn()
        end = ue.capture_count
        ue._primitive_cache[key] = list(ue.capture_buffer[start:end])
        return
    ue.capture_buffer.extend(cached)
    ue.capture_count += len(cached)
    ue._inst_id += len(cached)


# ============================================================================
# Per-layer weight extraction + upload
# ============================================================================
# The flow is split into two phases so the expensive FP4 quantization runs
# once and the results are cached in a binary file under qwen3.5_2b_bin/:
#
#   extract(hf_model) → dict of tensors / (scales, packed) tuples     [~20 s]
#   torch.save(extracted, weights_bin)                                [~2 s]
#   upload(extracted → FPGA DRAM)                                      [~5 s]
#
# Subsequent runs: torch.load(weights_bin) + upload, skipping the HF model
# load + quantization entirely.

def _pad_rows(W: torch.Tensor, target_rows: int) -> torch.Tensor:
    pad = target_rows - W.shape[0]
    if pad <= 0:
        return W
    return torch.cat([W, torch.zeros(pad, W.shape[1], dtype=W.dtype)], 0)


def _extract_linear_attn_weights(hf_layer) -> Dict:
    """Quantize + materialize one Gated-DeltaNet layer's weights on host
    (no FPGA calls).  FP4 projections become (scales_bf16, packed_u8) pairs."""
    la = hf_layer.linear_attn
    lyr = hf_layer
    return dict(
        GAMMA_INPUT=(lyr.input_layernorm.weight.float() + 1.0).to(torch.bfloat16),
        GAMMA_POST =(lyr.post_attention_layernorm.weight.float() + 1.0).to(torch.bfloat16),
        GAMMA_GATED=la.norm.weight.float().to(torch.bfloat16),
        Q_QKV =quantize_fp4_64(la.in_proj_qkv.weight.float()),
        Q_Z   =quantize_fp4_64(la.in_proj_z.weight.float()),
        Q_OUT =quantize_fp4_64(la.out_proj.weight.float()),
        Q_GATE=quantize_fp4_64(lyr.mlp.gate_proj.weight.float()),
        Q_UP  =quantize_fp4_64(lyr.mlp.up_proj.weight.float()),
        Q_DOWN=quantize_fp4_64(lyr.mlp.down_proj.weight.float()),
        W_B_PAD=_pad_rows(la.in_proj_b.weight.float(), UE_VECTOR_SIZE).to(torch.bfloat16),
        W_A_PAD=_pad_rows(la.in_proj_a.weight.float(), UE_VECTOR_SIZE).to(torch.bfloat16),
        # Host-resident at decode time (used by α computation + causal conv):
        A_log   =la.A_log.float().clone(),
        dt_bias =la.dt_bias.float().clone(),
        W_conv1d=la.conv1d.weight.float().clone(),
    )


def _extract_full_attn_weights(hf_layer) -> Dict:
    """Quantize + materialize one full-attn layer's weights on host."""
    attn = hf_layer.self_attn
    lyr = hf_layer
    return dict(
        GAMMA_INPUT=(lyr.input_layernorm.weight.float() + 1.0).to(torch.bfloat16),
        GAMMA_POST =(lyr.post_attention_layernorm.weight.float() + 1.0).to(torch.bfloat16),
        GAMMA_Q_N  =(attn.q_norm.weight.float() + 1.0).to(torch.bfloat16),
        GAMMA_K_N  =(attn.k_norm.weight.float() + 1.0).to(torch.bfloat16),
        Q_Q =quantize_fp4_64(attn.q_proj.weight.float()),
        Q_K =quantize_fp4_64(attn.k_proj.weight.float()),
        Q_V =quantize_fp4_64(attn.v_proj.weight.float()),
        Q_O =quantize_fp4_64(attn.o_proj.weight.float()),
        Q_G =quantize_fp4_64(lyr.mlp.gate_proj.weight.float()),
        Q_U =quantize_fp4_64(lyr.mlp.up_proj.weight.float()),
        Q_D =quantize_fp4_64(lyr.mlp.down_proj.weight.float()),
    )


def _upload_fp4_pair(ue, pair) -> Tuple[int, int]:
    """Upload a pre-quantized (scales_bf16, packed_u8) pair to FPGA params
    DRAM.  Returns (scale_addr, data_addr)."""
    scales, packed = pair
    s_bytes = scales.numel() * BF16
    d_bytes = packed.numel()
    scale_addr = ue.allocate_params_dram(s_bytes)
    ue.dma_write(DMA_DEVICE_H2C, scale_addr, scales, s_bytes)
    data_addr = ue.allocate_params_dram(d_bytes)
    ue.dma_write(DMA_DEVICE_H2C, data_addr, packed, d_bytes)
    return scale_addr, data_addr


def _upload_fp4_pair_fused(ue, pairs) -> Tuple[int, int]:
    """Upload a list of (scales, packed) FP4 pairs as ONE contiguous weight,
    concatenated along the output-row dim (i.e. along the flattened-block
    dimension — since `quantize_fp4_64` flattens row-major and blocks never
    span rows, cat([scales_A, scales_B]) and cat([packed_A, packed_B]) is
    exactly what you get from `quantize_fp4_64(cat([W_A, W_B], dim=0))`).

    Returns one (scale_addr, data_addr) pair addressing the fused weight, so
    a single `run_fp4_matmul` with N=sum(N_i) replaces N separate matmuls."""
    scales_all = torch.cat([p[0] for p in pairs], dim=0).contiguous()
    packed_all = torch.cat([p[1] for p in pairs], dim=0).contiguous()
    s_bytes = scales_all.numel() * BF16
    d_bytes = packed_all.numel()
    scale_addr = ue.allocate_params_dram(s_bytes)
    ue.dma_write(DMA_DEVICE_H2C, scale_addr, scales_all, s_bytes)
    data_addr = ue.allocate_params_dram(d_bytes)
    ue.dma_write(DMA_DEVICE_H2C, data_addr, packed_all, d_bytes)
    return scale_addr, data_addr


def _upload_linear_attn_weights(ue, ex: Dict) -> Dict:
    """DMA a per-extracted linear-attn layer to FPGA; returns DRAM-address
    dict consumed by _run_linear_attn_layer."""
    return dict(
        GAMMA_INPUT=_upload(ue, ex["GAMMA_INPUT"]),
        GAMMA_POST =_upload(ue, ex["GAMMA_POST"]),
        GAMMA_GATED=_upload(ue, ex["GAMMA_GATED"]),
        Q_QKV =_upload_fp4_pair(ue, ex["Q_QKV"]),
        Q_Z   =_upload_fp4_pair(ue, ex["Q_Z"]),
        Q_OUT =_upload_fp4_pair(ue, ex["Q_OUT"]),
        Q_GATE=_upload_fp4_pair(ue, ex["Q_GATE"]),
        Q_UP  =_upload_fp4_pair(ue, ex["Q_UP"]),
        Q_DOWN=_upload_fp4_pair(ue, ex["Q_DOWN"]),
        W_B_PAD=_upload(ue, ex["W_B_PAD"]),
        W_A_PAD=_upload(ue, ex["W_A_PAD"]),
        A_log   =ex["A_log"],
        dt_bias =ex["dt_bias"],
        W_conv1d=ex["W_conv1d"],
        # α-on-FPGA: dt_bias and c=exp(A_log) pre-uploaded padded to UE_VECTOR_SIZE.
        # Stored in params DRAM so they survive reset_tensor_dram_addr().
        DT_BIAS_FPGA=_upload(ue, _pad_to_vec(ex["dt_bias"].to(torch.bfloat16))),
        C_FPGA      =_upload(ue, _pad_to_vec(ex["A_log"].exp().to(torch.bfloat16))),
    )


def _upload_full_attn_weights(ue, ex: Dict) -> Dict:
    """DMA a per-extracted full-attn layer to FPGA.

    K and V projections share the same input and row-major FP4 block layout,
    so we upload them in one contiguous chunk (`Q_KV`).  For the T=1 decode
    path, one `run_fp4_matmul(N=2*kv_size)` replaces two separate matmuls.
    The T>1 prefill path still uses Q_K / Q_V separately — those are just
    subpointers into the same fused chunk (zero extra DRAM).
    """
    # Upload K and V as one block, then expose the K-only and V-only subviews.
    Q_KV = _upload_fp4_pair_fused(ue, [ex["Q_K"], ex["Q_V"]])
    k_scales_bytes = ex["Q_K"][0].numel() * BF16
    k_packed_bytes = ex["Q_K"][1].numel()
    Q_K = (Q_KV[0], Q_KV[1])
    Q_V = (Q_KV[0] + k_scales_bytes, Q_KV[1] + k_packed_bytes)
    return dict(
        GAMMA_INPUT=_upload(ue, ex["GAMMA_INPUT"]),
        GAMMA_POST =_upload(ue, ex["GAMMA_POST"]),
        GAMMA_Q_N  =_upload(ue, ex["GAMMA_Q_N"]),
        GAMMA_K_N  =_upload(ue, ex["GAMMA_K_N"]),
        Q_Q =_upload_fp4_pair(ue, ex["Q_Q"]),
        Q_K =Q_K,  Q_V=Q_V,  Q_KV=Q_KV,
        Q_O =_upload_fp4_pair(ue, ex["Q_O"]),
        Q_G =_upload_fp4_pair(ue, ex["Q_G"]),
        Q_U =_upload_fp4_pair(ue, ex["Q_U"]),
        Q_D =_upload_fp4_pair(ue, ex["Q_D"]),
    )


# ============================================================================
# HF model resolution + weight-bin caching
# ============================================================================

def _ensure_hf_model(script_dir: str, cfg: dict) -> str:
    """Ensure the HF model lives under qwen3.5_2b_bin/Qwen3.5-2B/.  Preference
    order: (1) it already exists there; (2) symlink from a pre-downloaded
    fallback path (e.g. /srv/model_files/...); (3) snapshot_download from the
    hub.  Returns the resolved local model dir.

    Matches gemma3/gemma4's pattern of keeping the HF snapshot alongside
    the weight + instruction bins in a single <model>_bin/ folder."""
    paths = cfg["paths"]
    model_dir = os.path.join(script_dir, paths["hf_model_dir"])
    config_json = os.path.join(model_dir, "config.json")
    if os.path.exists(config_json):
        return model_dir

    fallback = paths.get("hf_model_fallback_dir")
    if fallback and os.path.exists(os.path.join(fallback, "config.json")):
        os.makedirs(os.path.dirname(model_dir), exist_ok=True)
        if os.path.islink(model_dir) or os.path.exists(model_dir):
            os.remove(model_dir) if os.path.islink(model_dir) else None
        os.symlink(os.path.abspath(fallback), model_dir)
        print(f"  Symlinked {model_dir} -> {fallback}")
        return model_dir

    print(f"  HF model not found at {model_dir}, downloading from "
          f"{paths['hf_model_repo']} ...")
    from huggingface_hub import snapshot_download
    os.makedirs(model_dir, exist_ok=True)
    snapshot_download(repo_id=paths["hf_model_repo"], local_dir=model_dir,
                      local_dir_use_symlinks=False)
    return model_dir


def _extract_all_weights(text_model, linear_attn_layers: set) -> Dict:
    """Extract + quantize every layer's weights + final norm + embedding
    into a single nested dict suitable for torch.save()."""
    layers = []
    for idx, hf_layer in enumerate(text_model.layers):
        if idx in linear_attn_layers:
            layers.append(_extract_linear_attn_weights(hf_layer))
        else:
            layers.append(_extract_full_attn_weights(hf_layer))
    embed_w = text_model.embed_tokens.weight.detach().to(torch.bfloat16)
    # On-chip LM head = FP4-quantized tied embedding (B=[N=vocab, K=H]).  Pre-
    # quantized here so the load path (test.py reruns + run_from_bin) stays
    # compile-free; the bf16 table is kept too for the host input-embed gather.
    lm_scale, lm_data = quantize_lm_head_fp4(embed_w)
    return {
        "layers": layers,
        "final_norm_gamma": (text_model.norm.weight.float() + 1.0).to(torch.bfloat16),
        "embed_weight": embed_w,
        "lm_head_scale": lm_scale,
        "lm_head_data":  lm_data,
    }


# ============================================================================
# Instruction-bin save/load
# ----------------------------------------------------------------------------
# After the first decode step populates the instruction caches, we serialize
# them into the unified qwen3.5_2b_bin/programs.bin (decoder section). On subsequent runs we load the
# caches back and DMA the recorded bytes to the same prog_addrs — skipping
# the ~12 s of Python instruction emission on the first decode step.
#
# Instruction objects are serialized as raw 32-byte blobs (their in-memory
# representation is just .words = list of 8 uint32).  On load we rebuild the
# objects via `Instructions()` + struct.unpack.
# ============================================================================

from user_dma_core import Instructions   # noqa: E402


def _bytes_to_insts(b) -> list:
    """Decode a raw decoder-binary blob (32-byte instructions) back into a
    list of Instructions objects.  Used by `load_instructions` to rehydrate
    the pre-compiled decoder binary from disk on repeat runs."""
    import struct as _struct
    out = []
    for off in range(0, len(b), 32):
        inst = Instructions()
        for j in range(8):
            inst.words[j] = _struct.unpack_from("<I", b, off + j * 4)[0]
        out.append(inst)
    return out


# ============================================================================
# FPGA primitive wrappers (one capture each)
# ============================================================================

# Every run_* helper below accepts `batched=False`.  When `batched=True`,
# the helper assumes the caller has already called `ue.start_capture()` and
# will call `_exec_captured(ue)` itself — the helper just emits instructions.
# This lets the layer functions bundle many primitives into one capture and
# one DMA-write/execute round trip, instead of paying that cost per primitive.

def run_rms_norm_dram(ue, M: int, N: int, X_DRAM: int, GAMMA_DRAM: int,
                      batched: bool = False) -> int:
    OUT = _alloc_tensor(ue, M * N)
    if not batched:
        ue.start_capture()
    key = ("rms_norm", M, N, X_DRAM, OUT, GAMMA_DRAM)
    _cached_emit(ue, key, lambda: ue.rms_norm_core_dram(
        M=M, N=N, A_DRAM_ADDR=X_DRAM,
        OUTPUT_DRAM_ADDR=OUT, GAMMA_DRAM_ADDR=GAMMA_DRAM))
    if not batched:
        _exec_captured(ue)
    return OUT


_IDENTITY_DRAM_CACHE: Dict[int, int] = {}


def _get_identity_dram(ue, N: int) -> int:
    if N in _IDENTITY_DRAM_CACHE:
        return _IDENTITY_DRAM_CACHE[N]
    addr = _upload(ue, torch.eye(N, dtype=torch.float32))
    _IDENTITY_DRAM_CACHE[N] = addr
    return addr


def run_silu_dram(ue, X_DRAM: int, numel: int, N: int,
                  batched: bool = False) -> int:
    M = numel // N
    OUT = _alloc_tensor(ue, numel)
    I_DRAM = _get_identity_dram(ue, N)
    if not batched:
        ue.start_capture()
    key = ("silu", M, N, X_DRAM, I_DRAM, OUT)
    _cached_emit(ue, key, lambda: ue.matmat_mul_core(
        M=M, K=N, N=N, A_DRAM_ADDR=X_DRAM, B_DRAM_ADDR=I_DRAM,
        OUTPUT_DRAM_ADDR=OUT, is_B_quantized=False, silu_enable=True))
    if not batched:
        _exec_captured(ue)
    return OUT


def run_eltwise_op_dram(ue, A_DRAM: int, B_DRAM: int, numel: int,
                        op: str = "mul", batched: bool = False) -> int:
    OUT = _alloc_tensor(ue, numel)
    nbytes = numel * BF16
    if not batched:
        ue.start_capture()
    if op not in ("mul", "add"):
        raise ValueError(f"op must be 'mul' or 'add', got {op!r}")
    key = ("eltwise", op, numel, A_DRAM, B_DRAM, OUT)

    def _emit():
        ue.ue_memcpy_from_dram(A_DRAM, nbytes, MEMCPY_TYPE.URAM.value,
                               URAM_START_ADDR, URAM_SECTION.URAM_A.value)
        ue.ue_memcpy_from_dram(B_DRAM, nbytes, MEMCPY_TYPE.URAM.value,
                               URAM_START_ADDR, URAM_SECTION.URAM_B.value)
        if op == "mul":
            ue.eltwise_mul_core(0x00000, 0x80000, 0x00000, numel)
        else:
            ue.eltwise_add_core(0x00000, 0x80000, 0x00000, numel)
        ue.ue_memcpy_to_dram(MEMCPY_TYPE.URAM.value, URAM_SECTION.URAM_A.value,
                             URAM_START_ADDR, OUT, nbytes)

    _cached_emit(ue, key, _emit)
    if not batched:
        _exec_captured(ue)
    return OUT


def run_bf16_matmul(ue, M, K, N, A_DRAM, B_DRAM,
                    silu=False, sigmoid=False, gelu=False,
                    batched: bool = False) -> int:
    OUT = _alloc_tensor(ue, M * N)
    if not batched:
        ue.start_capture()
    key = ("bf16_mm", M, K, N, A_DRAM, B_DRAM, OUT, silu, sigmoid, gelu)
    _cached_emit(ue, key, lambda: ue.matmat_mul_core(
        M=M, K=K, N=N, A_DRAM_ADDR=A_DRAM, B_DRAM_ADDR=B_DRAM,
        OUTPUT_DRAM_ADDR=OUT, is_B_quantized=False,
        silu_enable=silu, sigmoid_enable=sigmoid, gelu_enable=gelu))
    if not batched:
        _exec_captured(ue)
    return OUT


def run_fp4_matmul(ue, M, K, N, A_DRAM, B_SCALE, B_DATA,
                   silu=False, sigmoid=False, gelu=False,
                   batched: bool = False) -> int:
    OUT = _alloc_tensor(ue, M * N)
    if not batched:
        ue.start_capture()
    # §3h (core_changes.md): at M=1 (decode GEMV) use the dedicated
    # quantized_matmat_core — it streams IF4 weights+scales through the
    # dot-product pipeline in N//256 chunks (~24-72 instrs for the big
    # matmuls), vs matmat_mul_core's per-N-tile (N_chunk=32) dequantize+matvec
    # GEMM unroll (~hundreds of instrs/matmul). Same IF4 numerics, far fewer
    # instructions AND faster on HW. This is the single biggest decoder-bin
    # contributor (notes: "MLP gate/up/down FP4 matmuls dominate").
    # Guard K>SCALE_BRAM_ELEMENTS: quantized_matmat_core's
    # N_chunk=(SCALE_BRAM_ELEMENTS//K)*UE_VECTOR_SIZE goes to 0 there and the
    # compile loops forever — fall back to the GEMM kernel. All qwen3.5_2b
    # matmuls have K<=6144<8192 so decode always takes the fast GEMV path;
    # the T>1 prefill/compare path (M>1) stays on the GEMM kernel.
    use_gemv = (M == 1 and K <= user_dma_core.SCALE_BRAM_ELEMENTS)
    if use_gemv:
        key = ("fp4_gemv", M, K, N, A_DRAM, B_SCALE, B_DATA, OUT, silu, sigmoid, gelu)
        _cached_emit(ue, key, lambda: ue.quantized_matmat_core(
            M=M, K=K, N=N, A_DRAM_ADDR=A_DRAM, B_DRAM_ADDR=B_DATA,
            OUTPUT_DRAM_ADDR=OUT, SCALE_DRAM_ADDR=B_SCALE, data_type=TYPE.IF4,
            silu_enable=silu, sigmoid_enable=sigmoid, gelu_enable=gelu))
    else:
        key = ("fp4_mm", M, K, N, A_DRAM, B_SCALE, B_DATA, OUT, silu, sigmoid, gelu)
        _cached_emit(ue, key, lambda: ue.matmat_mul_core(
            M=M, K=K, N=N, A_DRAM_ADDR=A_DRAM, B_DRAM_ADDR=B_DATA,
            OUTPUT_DRAM_ADDR=OUT, is_B_quantized=True,
            SCALE_DRAM_ADDR=B_SCALE, data_type=TYPE.IF4,
            silu_enable=silu, sigmoid_enable=sigmoid, gelu_enable=gelu))
    if not batched:
        _exec_captured(ue)
    return OUT


def run_causal_conv1d_4tap(ue, X_DRAM: int, W_host: torch.Tensor,
                           seq_len: int, conv_dim: int,
                           history: torch.Tensor = None,
                           batched: bool = False
                           ) -> Tuple[int, torch.Tensor]:
    """K-tap causal depthwise conv1d as K shifted eltwise_muls + K-1 adds.

    Args:
        history: optional [K-1, conv_dim] fp32 tensor of the most-recent K-1
            tokens that preceded this slice (i.e. x[-(K-1)], ..., x[-1]).
            Used so decode steps can continue the convolution across calls.
            If None, zeros are used (correct for prefill starting at pos 0).
        batched: if True, caller owns start_capture / _exec_captured and this
            helper emits instructions only.  NOTE: when batched=True, caller
            must have already executed any upstream primitives that produce
            X_DRAM — this helper does a C2H DMA read of X_DRAM to host to
            build the shifted tiles, which requires the data to already be
            materialized on the FPGA.

    Returns:
        (OUT_DRAM addr, new_history [K-1, conv_dim] fp32) — new_history holds
        the last K-1 rows of the concatenated input, ready for the next call.
    """
    T, C = seq_len, conv_dim
    K = W_host.shape[-1]
    elems = T * C
    nbytes = elems * BF16

    x_bf = torch.zeros(elems, dtype=torch.bfloat16)
    ue.dma_read(DMA_DEVICE_C2H, X_DRAM, x_bf, nbytes)
    x_host = x_bf.reshape(T, C).float()

    if history is None:
        history = torch.zeros(K - 1, C, dtype=torch.float32)
    assert history.shape == (K - 1, C)
    x_ext = torch.cat([history, x_host], 0)   # [K-1+T, C]
    w = W_host.view(C, K).float()

    TAP = [_alloc_tensor(ue, elems) for _ in range(K)]
    SHIFT_DRAM, WTILE_DRAM = [], []
    for s in range(K):
        k_idx = (K - 1) - s
        # shifted[t] = x_ext[t + (K-1) - s] for t = 0..T-1.
        start_idx = (K - 1) - s
        x_shifted = x_ext[start_idx:start_idx + T].contiguous()
        saddr = _alloc_tensor(ue, elems)
        ue.dma_write(DMA_DEVICE_H2C, saddr, _bf(x_shifted), nbytes)
        SHIFT_DRAM.append(saddr)
        wt = w[:, k_idx].unsqueeze(0).expand(T, C).contiguous()
        waddr = _alloc_tensor(ue, elems)
        ue.dma_write(DMA_DEVICE_H2C, waddr, _bf(wt), nbytes)
        WTILE_DRAM.append(waddr)

    rows_per_chunk = max(1, URAM_NEAR_FULL_ELEMENTS // C)
    if rows_per_chunk > T:
        rows_per_chunk = T

    if not batched:
        ue.start_capture()
    # Phase 1: K eltwise_muls, tap s → TAP[s].
    for s in range(K):
        for t_start in range(0, T, rows_per_chunk):
            rows = min(rows_per_chunk, T - t_start)
            chunk_elems = rows * C
            chunk_bytes = chunk_elems * BF16
            off_bytes = t_start * C * BF16
            ue.ue_memcpy_from_dram(SHIFT_DRAM[s] + off_bytes, chunk_bytes,
                                   MEMCPY_TYPE.URAM.value,
                                   URAM_START_ADDR, URAM_SECTION.URAM_A.value)
            ue.ue_memcpy_from_dram(WTILE_DRAM[s] + off_bytes, chunk_bytes,
                                   MEMCPY_TYPE.URAM.value,
                                   URAM_START_ADDR, URAM_SECTION.URAM_B.value)
            ue.eltwise_mul_core(0x00000, 0x80000, 0x00000, chunk_elems)
            ue.ue_memcpy_to_dram(MEMCPY_TYPE.URAM.value,
                                 URAM_SECTION.URAM_A.value,
                                 URAM_START_ADDR,
                                 TAP[s] + off_bytes, chunk_bytes)
    # Phase 2: accumulate into TAP[0].  Merged into the same capture.
    for s in range(1, K):
        for t_start in range(0, T, rows_per_chunk):
            rows = min(rows_per_chunk, T - t_start)
            chunk_elems = rows * C
            chunk_bytes = chunk_elems * BF16
            off_bytes = t_start * C * BF16
            ue.ue_memcpy_from_dram(TAP[0] + off_bytes, chunk_bytes,
                                   MEMCPY_TYPE.URAM.value,
                                   URAM_START_ADDR, URAM_SECTION.URAM_A.value)
            ue.ue_memcpy_from_dram(TAP[s] + off_bytes, chunk_bytes,
                                   MEMCPY_TYPE.URAM.value,
                                   URAM_START_ADDR, URAM_SECTION.URAM_B.value)
            ue.eltwise_add_core(0x00000, 0x80000, 0x00000, chunk_elems)
            ue.ue_memcpy_to_dram(MEMCPY_TYPE.URAM.value,
                                 URAM_SECTION.URAM_A.value,
                                 URAM_START_ADDR,
                                 TAP[0] + off_bytes, chunk_bytes)
    if not batched:
        _exec_captured(ue)
    new_history = x_ext[-(K - 1):].clone() if K > 1 else None
    return TAP[0], new_history


def _emit_conv1d_decode_T1(ue, layer_idx: int, QKV_PROJ_DRAM: int) -> int:
    """Emit the T=1 causal conv1d step directly into the active capture.

    No host round-trip: weights + state + scratch taps live at fixed DRAM
    addresses set up by prepare_inference.  At position t (with t-1, t-2, t-3
    already stored in the K-1 state slots):

        out = sum_{s=0..K-1} w[:, K-1-s] * x_ext[K-1-s]

    where `x_ext[K-1]` is the current token (QKV_PROJ_DRAM) and `x_ext[i<K-1]`
    is state slot `i`.  After computing out, we roll the state ring
    (state[0]←state[1], ..., state[K-2]←x_current) so the next step inherits
    the correct history.

    Returns the DRAM address holding the conv output [conv_dim] bf16.
    """
    C = ue.lin_conv_dim
    K = ue._conv_K
    nbytes = C * BF16
    state_addr   = ue._conv_state_dram[layer_idx]
    weights_addr = ue._conv_weight_tiles_dram[layer_idx]
    tap_addr     = ue._conv_tap_dram[layer_idx]

    def _input_addr(s):
        # s=0 → current token; s>=1 → state slot (K-1-s)
        return QKV_PROJ_DRAM if s == 0 else state_addr + (K - 1 - s) * nbytes

    def _emit():
        for s in range(K):
            ue.ue_memcpy_from_dram(_input_addr(s), nbytes,
                                   MEMCPY_TYPE.URAM.value,
                                   URAM_START_ADDR, URAM_SECTION.URAM_A.value)
            ue.ue_memcpy_from_dram(weights_addr + s * nbytes, nbytes,
                                   MEMCPY_TYPE.URAM.value,
                                   URAM_START_ADDR, URAM_SECTION.URAM_B.value)
            ue.eltwise_mul_core(0x00000, 0x80000, 0x00000, C)
            ue.ue_memcpy_to_dram(MEMCPY_TYPE.URAM.value, URAM_SECTION.URAM_A.value,
                                 URAM_START_ADDR,
                                 tap_addr + s * nbytes, nbytes)
        for s in range(1, K):
            ue.ue_memcpy_from_dram(tap_addr + 0 * nbytes, nbytes,
                                   MEMCPY_TYPE.URAM.value,
                                   URAM_START_ADDR, URAM_SECTION.URAM_A.value)
            ue.ue_memcpy_from_dram(tap_addr + s * nbytes, nbytes,
                                   MEMCPY_TYPE.URAM.value,
                                   URAM_START_ADDR, URAM_SECTION.URAM_B.value)
            ue.eltwise_add_core(0x00000, 0x80000, 0x00000, C)
            ue.ue_memcpy_to_dram(MEMCPY_TYPE.URAM.value, URAM_SECTION.URAM_A.value,
                                 URAM_START_ADDR,
                                 tap_addr + 0 * nbytes, nbytes)
        for s in range(K - 2):
            ue.ue_memcpy_from_dram(state_addr + (s + 1) * nbytes, nbytes,
                                   MEMCPY_TYPE.URAM.value,
                                   URAM_START_ADDR, URAM_SECTION.URAM_A.value)
            ue.ue_memcpy_to_dram(MEMCPY_TYPE.URAM.value, URAM_SECTION.URAM_A.value,
                                 URAM_START_ADDR,
                                 state_addr + s * nbytes, nbytes)
        ue.ue_memcpy_from_dram(QKV_PROJ_DRAM, nbytes,
                               MEMCPY_TYPE.URAM.value,
                               URAM_START_ADDR, URAM_SECTION.URAM_A.value)
        ue.ue_memcpy_to_dram(MEMCPY_TYPE.URAM.value, URAM_SECTION.URAM_A.value,
                             URAM_START_ADDR,
                             state_addr + (K - 2) * nbytes, nbytes)

    key = ("conv1d_T1", layer_idx, QKV_PROJ_DRAM)
    _cached_emit(ue, key, _emit)
    return tap_addr  # tap_buf_0 — accumulated conv output


def _apply_partial_rotary(x: torch.Tensor, rope_theta: float,
                          rot_dim: int, pos_start: int = 0) -> torch.Tensor:
    """Apply RoPE to the first rot_dim dims of each head at absolute positions
    [pos_start, pos_start+T).  Remaining head dims pass through unchanged."""
    T, Hh, D = x.shape
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rot_dim, 2,
                                                   dtype=torch.float32) / rot_dim))
    pos = torch.arange(pos_start, pos_start + T, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    cos = freqs.cos().unsqueeze(1)
    sin = freqs.sin().unsqueeze(1)
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    half = rot_dim // 2
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rotated = torch.cat([x1 * cos - x2 * sin,
                         x1 * sin + x2 * cos], dim=-1)
    return torch.cat([rotated, x_pass], dim=-1)


# ============================================================================
# Per-layer cascaded execution (ports from compare_qwen3.5_2b.py)
# ============================================================================

def _run_linear_attn_layer(ue, X_DRAM: int, layer_idx: int, T: int,
                           zero_s: bool) -> int:
    """One Gated DeltaNet (linear-attention) layer; returns OUT_DRAM.

    Uses `ue._layer_weights[layer_idx]` (pre-uploaded), `ue._s_dram[layer_idx]`
    (persistent S state — updated in place by the recurrence core), and
    `ue._conv_state[layer_idx]` (last K-1 rows of the in_proj_qkv output,
    needed to continue the causal conv1d across calls).

    `zero_s=True` should be passed on the first (prefill) call so the S state
    is re-zeroed; decode steps pass `zero_s=False` to preserve it.
    """
    w = ue._layer_weights[layer_idx]

    H        = ue.hidden_size
    conv_dim = ue.lin_conv_dim
    value_dim= ue.lin_value_dim
    key_dim  = ue.lin_key_dim
    num_vh   = ue.lin_num_v_heads
    num_kh   = ue.lin_num_k_heads
    Dk       = ue.lin_head_k_dim
    Dv       = ue.lin_head_v_dim
    mlp_dim  = ue.mlp_dim

    if T == 1:
        # === DECODE FAST PATH (T=1): single capture covers prenorm + qkv +
        # conv + silu + Q/K/V norms + projections + α + β + k + recurrence +
        # normed + gated + out_proj + residual + MLP + residual.
        # Zero host barriers for the entire layer.
        ue.start_capture()
        PRENORM_DRAM  = run_rms_norm_dram(ue, M=T, N=H, X_DRAM=X_DRAM,
                                          GAMMA_DRAM=w["GAMMA_INPUT"],
                                          batched=True)
        QKV_PROJ_DRAM = run_fp4_matmul(ue, M=T, K=H, N=conv_dim,
                                       A_DRAM=PRENORM_DRAM,
                                       B_SCALE=w["Q_QKV"][0], B_DATA=w["Q_QKV"][1],
                                       batched=True)
        CONV_OUT_DRAM = _emit_conv1d_decode_T1(ue, layer_idx, QKV_PROJ_DRAM)
        POST_SILU_DRAM = run_silu_dram(ue, CONV_OUT_DRAM,
                                       numel=T * conv_dim, N=conv_dim,
                                       batched=True)
        # Continue without closing capture — fold everything into one exec.
    else:
        # === PREFILL PATH (T>1): keep the vectorized host-tile conv path
        # since it processes T rows per eltwise op in parallel. ===
        ue.start_capture()
        PRENORM_DRAM  = run_rms_norm_dram(ue, M=T, N=H, X_DRAM=X_DRAM,
                                          GAMMA_DRAM=w["GAMMA_INPUT"],
                                          batched=True)
        QKV_PROJ_DRAM = run_fp4_matmul(ue, M=T, K=H, N=conv_dim,
                                       A_DRAM=PRENORM_DRAM,
                                       B_SCALE=w["Q_QKV"][0], B_DATA=w["Q_QKV"][1],
                                       batched=True)
        _exec_captured(ue)

        ue.start_capture()
        CONV_OUT_DRAM, new_hist = run_causal_conv1d_4tap(
            ue, X_DRAM=QKV_PROJ_DRAM, W_host=w["W_conv1d"],
            seq_len=T, conv_dim=conv_dim,
            history=ue._conv_state[layer_idx], batched=True)
        ue._conv_state[layer_idx] = new_hist
        # Sync the new history to the FPGA state ring so the next (decode)
        # step's FPGA-resident conv picks up the right K-1 trailing tokens.
        K_conv = ue._conv_K
        state_bytes = (K_conv - 1) * conv_dim * BF16
        ue.dma_write(DMA_DEVICE_H2C, ue._conv_state_dram[layer_idx],
                     _bf(new_hist.contiguous()), state_bytes)
        POST_SILU_DRAM = run_silu_dram(ue, CONV_OUT_DRAM,
                                       numel=T * conv_dim, N=conv_dim,
                                       batched=True)

    # Q, K, V split.
    # T=1 fast path: POST_SILU_DRAM is a single [conv_dim] row, so Q/K/V are
    #   just fixed byte offsets into the same buffer — no reshape needed.
    # T>1 prefill: [T, conv_dim] row-major means Q for all T tokens is NOT a
    #   contiguous block (it's column slices across rows), so we must read to
    #   host, rearrange, and upload contiguous Q/K/V buffers.
    assert num_vh == num_kh, (
        "layer split assumes num_v_heads == num_k_heads (Qwen3.5-2B invariant)")
    if T == 1:
        Q_SPLIT_DRAM = POST_SILU_DRAM + 0
        K_SPLIT_DRAM = POST_SILU_DRAM + key_dim * BF16
        V_SPLIT_DRAM = POST_SILU_DRAM + 2 * key_dim * BF16
    else:
        _exec_captured(ue)
        post_silu_hw = _read_bf(ue, POST_SILU_DRAM, (T, conv_dim))
        q_flat = post_silu_hw[:, :key_dim].contiguous().reshape(
            T * num_vh, Dk).contiguous()
        k_flat = post_silu_hw[:, key_dim:2 * key_dim].contiguous().reshape(
            T * num_vh, Dk).contiguous()
        v_flat = post_silu_hw[:, 2 * key_dim:].contiguous().reshape(
            T * num_vh, Dv).contiguous()
        Q_SPLIT_DRAM = _alloc_tensor(ue, T * num_vh * Dk)
        K_SPLIT_DRAM = _alloc_tensor(ue, T * num_vh * Dk)
        V_SPLIT_DRAM = _alloc_tensor(ue, T * num_vh * Dv)
        ue.dma_write(DMA_DEVICE_H2C, Q_SPLIT_DRAM, _bf(q_flat),
                     T * num_vh * Dk * BF16)
        ue.dma_write(DMA_DEVICE_H2C, K_SPLIT_DRAM, _bf(k_flat),
                     T * num_vh * Dk * BF16)
        ue.dma_write(DMA_DEVICE_H2C, V_SPLIT_DRAM, _bf(v_flat),
                     T * num_vh * Dv * BF16)
        ue.start_capture()

    Q_L2_DRAM = run_rms_norm_dram(ue, M=T * num_vh, N=Dk,
                                  X_DRAM=Q_SPLIT_DRAM,
                                  GAMMA_DRAM=ue._gamma_q_l2, batched=True)
    K_L2_DRAM = run_rms_norm_dram(ue, M=T * num_vh, N=Dk,
                                  X_DRAM=K_SPLIT_DRAM,
                                  GAMMA_DRAM=ue._gamma_k_l2, batched=True)
    # For T=1: BETA_PAD uses sigmoid=False (raw projection) so FPGA can compute
    # 1/β = 1 + exp(-raw_beta) without a host round-trip.
    # For T>1: use sigmoid=True (host reads sigmoid output directly).
    BETA_PAD_DRAM = run_bf16_matmul(ue, M=T, K=H, N=UE_VECTOR_SIZE,
                                    A_DRAM=PRENORM_DRAM, B_DRAM=w["W_B_PAD"],
                                    sigmoid=(T != 1), batched=True)
    A_PAD_DRAM = run_bf16_matmul(ue, M=T, K=H, N=UE_VECTOR_SIZE,
                                 A_DRAM=PRENORM_DRAM, B_DRAM=w["W_A_PAD"],
                                 batched=True)
    SILU_Z_DRAM = run_fp4_matmul(ue, M=T, K=H, N=value_dim,
                                 A_DRAM=PRENORM_DRAM,
                                 B_SCALE=w["Q_Z"][0], B_DATA=w["Q_Z"][1],
                                 silu=True, batched=True)
    # Persistent S buffer.  Re-zero only on explicit request (first prefill).
    S_DRAM = ue._s_dram[layer_idx]
    if zero_s:
        ue.dma_write(DMA_DEVICE_H2C, S_DRAM,
                     _bf(torch.zeros(num_vh * Dk * Dv)),
                     num_vh * Dk * Dv * BF16)
    CORE_OUT_DRAM = _alloc_tensor(ue, T * num_vh * Dv)
    SCRATCH = _alloc_tensor(ue, num_vh * (2 * Dk * Dv + Dv))

    if T == 1:
        # α, β, k all computed on FPGA — zero host barriers inside the layer.
        # Both the per-step run_decoder path (compile_mode=False) and the
        # single-trigger compile_decoder path (compile_mode=True) share this
        # branch, so switching between them is a one-line capture flag.  The
        # scatter matmul uses a spreader matrix to avoid the unaligned-DMA
        # alignment pitfall (see `_emit_alpha_fpga_to_slots` docstring).
        alpha_dram = ue._alpha_dram_by_s[S_DRAM]
        beta_dram  = ue._beta_dram_by_s[S_DRAM]
        k_pad_dram = ue._k_pad_dram_by_s[S_DRAM]

        _emit_alpha_fpga_to_slots(
            ue,
            A_RAW_DRAM=A_PAD_DRAM,
            DT_BIAS_DRAM=w["DT_BIAS_FPGA"],
            C_DRAM=w["C_FPGA"],
            alpha_dram=alpha_dram,
            x_scratch=ue._alpha_x_dram_by_s[S_DRAM],
            rx_scratch=ue._alpha_rx_dram_by_s[S_DRAM],
            y_scratch=ue._alpha_y_dram_by_s[S_DRAM],
            num_heads=num_vh)
        _emit_inv_beta_fpga_to_slots(
            ue,
            BETA_RAW_DRAM=BETA_PAD_DRAM,
            beta_dram=beta_dram,
            y_scratch=ue._beta_y_dram_by_s[S_DRAM],
            ones_dram=ue._ones_dram,
            num_heads=num_vh)
        _emit_k_pad_fpga(
            ue,
            K_L2_DRAM=K_L2_DRAM,
            k_pad_dram=k_pad_dram,
            num_heads=num_vh,
            Dk=Dk,
            N=UE_VECTOR_SIZE)
        alpha_dummy = torch.ones(T, num_vh, dtype=torch.bfloat16)
        beta_dummy  = torch.ones(T, num_vh, dtype=torch.bfloat16)
        k_dummy     = torch.zeros(T, num_vh, Dk, dtype=torch.bfloat16)
        ue.recurrent_gated_delta_rule_core(
            T=T, num_heads=num_vh, Dk=Dk, Dv=Dv,
            Q_DRAM_ADDR=Q_L2_DRAM, K_DRAM_ADDR=K_L2_DRAM,
            V_DRAM_ADDR=V_SPLIT_DRAM,
            S_DRAM_ADDR=S_DRAM, OUT_DRAM_ADDR=CORE_OUT_DRAM,
            SCRATCH_DRAM_ADDR=SCRATCH,
            alpha_host=alpha_dummy,
            beta_host=beta_dummy,
            k_host=k_dummy,
            alpha_on_fpga=True,
            beta_on_fpga=True,
            k_on_fpga=True)
    else:
        # === PREFILL: host barrier for α/β/k (T>1 path unchanged). ===
        _exec_captured(ue)
        k_l2_hw  = _read_bf(ue, K_L2_DRAM, (T * num_vh, Dk))
        beta_hw  = _read_bf(ue, BETA_PAD_DRAM, (T, UE_VECTOR_SIZE))[:, :num_vh]
        a_raw_hw = _read_bf(ue, A_PAD_DRAM, (T, UE_VECTOR_SIZE))[:, :num_vh]
        alpha_hw = (-w["A_log"].exp() * F.softplus(a_raw_hw + w["dt_bias"])).exp()

        ue.start_capture()
        ue.recurrent_gated_delta_rule_core(
            T=T, num_heads=num_vh, Dk=Dk, Dv=Dv,
            Q_DRAM_ADDR=Q_L2_DRAM, K_DRAM_ADDR=K_L2_DRAM,
            V_DRAM_ADDR=V_SPLIT_DRAM,
            S_DRAM_ADDR=S_DRAM, OUT_DRAM_ADDR=CORE_OUT_DRAM,
            SCRATCH_DRAM_ADDR=SCRATCH,
            alpha_host=alpha_hw.to(torch.bfloat16),
            beta_host=beta_hw.to(torch.bfloat16),
            k_host=k_l2_hw.reshape(T, num_vh, Dk).to(torch.bfloat16))
    NORMED_DRAM = run_rms_norm_dram(ue, M=T * num_vh, N=Dv,
                                    X_DRAM=CORE_OUT_DRAM,
                                    GAMMA_DRAM=w["GAMMA_GATED"], batched=True)
    GATED_DRAM = run_eltwise_op_dram(ue, NORMED_DRAM, SILU_Z_DRAM,
                                     numel=T * value_dim, op="mul",
                                     batched=True)
    OUT_PROJ_DRAM = run_fp4_matmul(ue, M=T, K=value_dim, N=H,
                                   A_DRAM=GATED_DRAM,
                                   B_SCALE=w["Q_OUT"][0], B_DATA=w["Q_OUT"][1],
                                   batched=True)
    RESIDUAL_DRAM = run_eltwise_op_dram(ue, X_DRAM, OUT_PROJ_DRAM,
                                        numel=T * H, op="add", batched=True)
    POST_LN_DRAM = run_rms_norm_dram(ue, M=T, N=H,
                                     X_DRAM=RESIDUAL_DRAM,
                                     GAMMA_DRAM=w["GAMMA_POST"], batched=True)
    GATE_DRAM = run_fp4_matmul(ue, M=T, K=H, N=mlp_dim,
                               A_DRAM=POST_LN_DRAM,
                               B_SCALE=w["Q_GATE"][0], B_DATA=w["Q_GATE"][1],
                               silu=True, batched=True)
    UP_DRAM = run_fp4_matmul(ue, M=T, K=H, N=mlp_dim,
                             A_DRAM=POST_LN_DRAM,
                             B_SCALE=w["Q_UP"][0], B_DATA=w["Q_UP"][1],
                             batched=True)
    MULT_DRAM = run_eltwise_op_dram(ue, GATE_DRAM, UP_DRAM,
                                    numel=T * mlp_dim, op="mul", batched=True)
    DOWN_DRAM = run_fp4_matmul(ue, M=T, K=mlp_dim, N=H,
                               A_DRAM=MULT_DRAM,
                               B_SCALE=w["Q_DOWN"][0], B_DATA=w["Q_DOWN"][1],
                               batched=True)
    FINAL_DRAM = run_eltwise_op_dram(ue, RESIDUAL_DRAM, DOWN_DRAM,
                                     numel=T * H, op="add", batched=True)
    _exec_captured(ue, timeout=180.0)
    return FINAL_DRAM


def _run_full_attn_layer(ue, X_DRAM: int, layer_idx: int, T: int,
                         pos_start: int) -> int:
    """One gated full-attention layer using a persistent KV cache.

    Cache layout per layer: [num_q_heads, max_context_aligned, full_head_dim]
    bf16, already GQA-expanded + RoPE-applied.  On each call the new T tokens
    are normalized, RoPE'd at absolute positions [pos_start, pos_start+T), GQA-
    expanded, and written into the cache at rows [pos_start:pos_start+T].
    Flash attention then runs over the entire max_context_aligned region with
    a causal bias that masks out cols beyond `pos_start+T`.
    """
    w = ue._layer_weights[layer_idx]
    H            = ue.hidden_size
    mlp_dim      = ue.mlp_dim
    full_head_dim= ue.full_head_dim
    num_q_heads  = ue.full_num_heads
    num_kv_heads = ue.full_num_kv_heads
    rotary_dim   = ue.full_rotary_dim
    rope_theta   = ue.rope_theta
    T_aligned    = ue.max_context_aligned

    q_size_flat = num_q_heads * full_head_dim
    q_proj_out  = num_q_heads * full_head_dim * 2
    kv_size     = num_kv_heads * full_head_dim
    group_sz    = num_q_heads // num_kv_heads
    end_pos     = pos_start + T
    head_stride_bytes = T_aligned * full_head_dim * BF16
    row_bytes   = full_head_dim * BF16
    assert end_pos <= ue.max_context, (
        f"pos_start+T={end_pos} exceeds max_context={ue.max_context}")

    K_cache = ue._k_cache[layer_idx]
    V_cache = ue._v_cache[layer_idx]

    if T == 1:
        # ================================================================
        # === DECODE FAST PATH (T=1): ONE capture, ONE exec, zero host barriers.
        # ================================================================
        # SRAM scratch (must not conflict with flash attention internals).
        # 0x40000-0x40400: used by _emit_rope_split_64 (reused sequentially).
        SA_QTMP  = 0x40000  # Q de-interleave scratch           (URAM_A, reused)
        SA_KV_K  = 0x40600  # K for GQA expand                 (URAM_A)
        SA_KV_V  = 0x40800  # V for GQA expand                 (URAM_A)
        SA_ATTN  = 0x40A00  # flash output row extract          (URAM_A)
        SB_GATE  = 0x8C000  # gate scratch                      (URAM_B)

        rot_dim = rotary_dim  # 64

        Q_FLASH_DRAM = ue._fa_q_flash_dram[layer_idx]

        FLASH_OUT_DRAM = _alloc_tensor(ue, num_q_heads * T_aligned * full_head_dim)
        SCRATCH_DRAM   = _alloc_tensor(ue,
                                       T_aligned * T_aligned * 4
                                       + T_aligned * full_head_dim * 4)
        ATTN_CONCAT_DRAM = _alloc_tensor(ue, T * q_size_flat)

        # Rebuild bias with cols beyond end_pos masked (host side, cheap DMA).
        bias_host = torch.full((T_aligned, T_aligned), float("-inf"),
                               dtype=torch.bfloat16)
        bias_host.masked_fill_(
            torch.tril(torch.ones(T_aligned, T_aligned, dtype=torch.bool)), 0.0)
        bias_host[:, end_pos:] = float("-inf")
        ue.dma_write(DMA_DEVICE_H2C, ue._bias_dram, _bf(bias_host),
                     T_aligned * T_aligned * BF16)

        # Single capture starts here — covers everything through MLP.
        ue.start_capture()

        PRENORM_DRAM = run_rms_norm_dram(ue, M=T, N=H,
                                         X_DRAM=X_DRAM,
                                         GAMMA_DRAM=w["GAMMA_INPUT"],
                                         batched=True)
        Q_PROJ = run_fp4_matmul(ue, M=T, K=H, N=q_proj_out,
                                A_DRAM=PRENORM_DRAM,
                                B_SCALE=w["Q_Q"][0], B_DATA=w["Q_Q"][1],
                                batched=True)
        # Fused K+V matmul: output [T, 2*kv_size], first kv_size bytes = K,
        # next kv_size bytes = V (per row).  Splitting is a pointer offset.
        KV_PROJ = run_fp4_matmul(ue, M=T, K=H, N=2 * kv_size,
                                 A_DRAM=PRENORM_DRAM,
                                 B_SCALE=w["Q_KV"][0], B_DATA=w["Q_KV"][1],
                                 batched=True)
        K_PROJ = KV_PROJ
        V_PROJ = KV_PROJ + kv_size * BF16
        # Q_PROJ layout: head h occupies [h*head_dim*2 : h*head_dim*2+head_dim]
        # as query, [+head_dim : +2*head_dim] as gate.
        Q_QUERY_DRAM = _alloc_tensor(ue, num_q_heads * full_head_dim)
        for h in range(num_q_heads):
            ue.accelerator_memory_to_sram(
                Q_PROJ + h * full_head_dim * 2 * BF16, SA_QTMP, full_head_dim)
            ue.sram_to_accelerator_memory(
                SA_QTMP, Q_QUERY_DRAM + h * full_head_dim * BF16, full_head_dim)
        Q_NORM_DRAM = run_rms_norm_dram(ue, M=T * num_q_heads, N=full_head_dim,
                                        X_DRAM=Q_QUERY_DRAM,
                                        GAMMA_DRAM=w["GAMMA_Q_N"], batched=True)
        K_NORM_DRAM = run_rms_norm_dram(ue, M=T * num_kv_heads, N=full_head_dim,
                                        X_DRAM=K_PROJ,
                                        GAMMA_DRAM=w["GAMMA_K_N"], batched=True)
        Q_ROPE_DRAM = _alloc_tensor(ue, num_q_heads * full_head_dim)
        pass_len = full_head_dim - rot_dim
        COS_PAD_POS     = ue._fa_cos_pad_dram     + pos_start * rot_dim * BF16
        NEG_SIN_PAD_POS = ue._fa_neg_sin_pad_dram + pos_start * rot_dim * BF16
        SIN_HI_PAD_POS  = ue._fa_sin_hi_pad_dram  + pos_start * rot_dim * BF16
        # In compile mode the add_imm uses these as immediates: pass the base
        # (position 0) so that at runtime TMP = ISA_POS_ROPE_REG + base, which
        # resolves to the correct per-step position.  Non-compile mode uses the
        # position-indexed address directly for the DMA load.
        if ue._compile_mode:
            _cos_arg, _neg_sin_arg, _sin_hi_arg = (
                ue._fa_cos_pad_dram, ue._fa_neg_sin_pad_dram, ue._fa_sin_hi_pad_dram)
        else:
            _cos_arg, _neg_sin_arg, _sin_hi_arg = (
                COS_PAD_POS, NEG_SIN_PAD_POS, SIN_HI_PAD_POS)
        for h in range(num_q_heads):
            src_h = Q_NORM_DRAM + h * full_head_dim * BF16
            out_h = Q_ROPE_DRAM + h * full_head_dim * BF16
            _emit_rope_split_64(ue, src_h, out_h,
                                _cos_arg, _neg_sin_arg, _sin_hi_arg,
                                rot_dim, pass_len, load_cos_sin=(h == 0))

        K_ROPE_DRAM = _alloc_tensor(ue, num_kv_heads * full_head_dim)
        for kh in range(num_kv_heads):
            src_k = K_NORM_DRAM + kh * full_head_dim * BF16
            out_k = K_ROPE_DRAM + kh * full_head_dim * BF16
            _emit_rope_split_64(ue, src_k, out_k,
                                _cos_arg, _neg_sin_arg, _sin_hi_arg,
                                rot_dim, pass_len, load_cos_sin=(kh == 0))
        for kh in range(num_kv_heads):
            src_k = K_ROPE_DRAM + kh * full_head_dim * BF16
            src_v = V_PROJ + kh * full_head_dim * BF16
            ue.accelerator_memory_to_sram(src_k, SA_KV_K, full_head_dim)
            ue.accelerator_memory_to_sram(src_v, SA_KV_V, full_head_dim)
            for rep in range(group_sz):
                qh = kh * group_sz + rep
                dst_k_base = K_cache + qh * head_stride_bytes
                dst_v_base = V_cache + qh * head_stride_bytes
                if ue._compile_mode:
                    ue.generate_instruction_add_imm(ue._ISA_POS_K_REG, ue_35bit_addr_shifter(dst_k_base), ue._ISA_TMP_REG)
                    ue.sram_to_accelerator_memory(SA_KV_K, dst_k_base, full_head_dim)
                    ue.overwrite_instruction_with_general_register(ue._ISA_TMP_REG)
                    ue.generate_instruction_add_imm(ue._ISA_POS_K_REG, ue_35bit_addr_shifter(dst_v_base), ue._ISA_TMP_REG)
                    ue.sram_to_accelerator_memory(SA_KV_V, dst_v_base, full_head_dim)
                    ue.overwrite_instruction_with_general_register(ue._ISA_TMP_REG)
                else:
                    ue.sram_to_accelerator_memory(SA_KV_K, dst_k_base + pos_start * row_bytes, full_head_dim)
                    ue.sram_to_accelerator_memory(SA_KV_V, dst_v_base + pos_start * row_bytes, full_head_dim)

        for h in range(num_q_heads):
            src = Q_ROPE_DRAM + h * full_head_dim * BF16
            dst_base = Q_FLASH_DRAM + h * head_stride_bytes
            ue.accelerator_memory_to_sram(src, SA_QTMP, full_head_dim)
            if ue._compile_mode:
                ue.generate_instruction_add_imm(ue._ISA_POS_K_REG, ue_35bit_addr_shifter(dst_base), ue._ISA_TMP_REG)
                ue.sram_to_accelerator_memory(SA_QTMP, dst_base, full_head_dim)
                ue.overwrite_instruction_with_general_register(ue._ISA_TMP_REG)
            else:
                ue.sram_to_accelerator_memory(SA_QTMP, dst_base + pos_start * row_bytes, full_head_dim)
        if getattr(ue, "_s7_full_attn_active", False):
            # §7 shared-subroutine flash: instead of emitting a full flash body
            # per head (8 heads × 6 layers = 48 inline bodies), marshal each
            # head's Q/K/V into the fixed shared buffers and jump into the ONE
            # bucketized flash body compiled after the decoder HALT.  The full
            # per-head [T_aligned, head_dim] block fits one URAM_A DMA (65536 ≤
            # URAM_FULL_ELEMENTS), so each marshal/out-copy is one load+store.
            head_elems = T_aligned * full_head_dim
            S7_STAGE = 0x00000   # URAM_A start — free before the shared body runs
            for h in range(num_q_heads):
                off = h * head_stride_bytes
                for src, dst in ((Q_FLASH_DRAM + off, ue._s7_flash_q),
                                 (K_cache + off,      ue._s7_flash_k),
                                 (V_cache + off,      ue._s7_flash_v)):
                    ue.accelerator_memory_to_sram(src, S7_STAGE, head_elems)
                    ue.sram_to_accelerator_memory(S7_STAGE, dst, head_elems)
                ue.pad_capture_to_64b_boundary()
                _ret = ue_35bit_addr_shifter(
                    ue._s7_program_base
                    + (ue.capture_count + 2) * user_dma_core.INSTRUCTION_SIZE_BYTES)
                ue.generate_instruction_add_set(ue._s7_gpr_ret_id, _ret)
                ue._s7_flash_call_sites.append(ue.capture_count)
                ue.generate_instruction_jump_abs(target_instruction_word_addr=0)
                ue.accelerator_memory_to_sram(ue._s7_flash_out, S7_STAGE, head_elems)
                ue.sram_to_accelerator_memory(
                    S7_STAGE, FLASH_OUT_DRAM + off, head_elems)
        else:
            key_fa = ("flash_T1", layer_idx, Q_FLASH_DRAM, K_cache, V_cache,
                      FLASH_OUT_DRAM, SCRATCH_DRAM, ue._bias_dram, T_aligned)

            def _emit_flash_T1():
                for h in range(num_q_heads):
                    ue.flash_attention_core_cached(
                        head_dim=full_head_dim, seq_len=T_aligned,
                        Q_DRAM_ADDR=Q_FLASH_DRAM + h * head_stride_bytes,
                        K_DRAM_ADDR=K_cache + h * head_stride_bytes,
                        V_DRAM_ADDR=V_cache + h * head_stride_bytes,
                        OUTPUT_DRAM_ADDR=FLASH_OUT_DRAM + h * head_stride_bytes,
                        SCRATCH_DRAM_ADDR=SCRATCH_DRAM,
                        BIAS_DRAM_ADDR=ue._bias_dram,
                    )
            _cached_emit(ue, key_fa, _emit_flash_T1)
        SIG_GATE_DRAM = _alloc_tensor(ue, num_q_heads * full_head_dim)
        for h in range(num_q_heads):
            gate_h = Q_PROJ + h * full_head_dim * 2 * BF16 + full_head_dim * BF16
            sig_out_h = SIG_GATE_DRAM + h * full_head_dim * BF16
            # sigmoid(gate[h]) via matmul with eye(head_dim)
            key_sig = ("sig_gate_T1", layer_idx, h, gate_h, sig_out_h)
            _cached_emit(ue, key_sig, lambda _g=gate_h, _o=sig_out_h: ue.matmat_mul_core(
                M=1, K=full_head_dim, N=full_head_dim,
                A_DRAM_ADDR=_g,
                B_DRAM_ADDR=ue._fa_eye_head_dram,
                OUTPUT_DRAM_ADDR=_o,
                is_B_quantized=False,
                sigmoid_enable=True))

        for h in range(num_q_heads):
            attn_row_base = FLASH_OUT_DRAM + h * head_stride_bytes
            sig_h    = SIG_GATE_DRAM + h * full_head_dim * BF16
            out_h    = ATTN_CONCAT_DRAM + h * full_head_dim * BF16
            if ue._compile_mode:
                ue.generate_instruction_add_imm(ue._ISA_POS_K_REG, ue_35bit_addr_shifter(attn_row_base), ue._ISA_TMP_REG)
                ue.accelerator_memory_to_sram(attn_row_base, SA_ATTN, full_head_dim)
                ue.overwrite_instruction_with_general_register(ue._ISA_TMP_REG)
            else:
                ue.accelerator_memory_to_sram(attn_row_base + pos_start * row_bytes, SA_ATTN, full_head_dim)
            ue.accelerator_memory_to_sram(sig_h, SB_GATE, full_head_dim)
            ue.eltwise_mul_core(SA_ATTN, SB_GATE, SA_ATTN, full_head_dim)
            ue.sram_to_accelerator_memory(SA_ATTN, out_h, full_head_dim)
        OUT_PROJ_DRAM = run_fp4_matmul(ue, M=T, K=q_size_flat, N=H,
                                       A_DRAM=ATTN_CONCAT_DRAM,
                                       B_SCALE=w["Q_O"][0], B_DATA=w["Q_O"][1],
                                       batched=True)
        RESIDUAL_DRAM = run_eltwise_op_dram(ue, X_DRAM, OUT_PROJ_DRAM,
                                            numel=T * H, op="add", batched=True)
        POST_LN_DRAM = run_rms_norm_dram(ue, M=T, N=H,
                                         X_DRAM=RESIDUAL_DRAM,
                                         GAMMA_DRAM=w["GAMMA_POST"], batched=True)
        GATE_DRAM = run_fp4_matmul(ue, M=T, K=H, N=mlp_dim,
                                   A_DRAM=POST_LN_DRAM,
                                   B_SCALE=w["Q_G"][0], B_DATA=w["Q_G"][1],
                                   silu=True, batched=True)
        UP_DRAM = run_fp4_matmul(ue, M=T, K=H, N=mlp_dim,
                                 A_DRAM=POST_LN_DRAM,
                                 B_SCALE=w["Q_U"][0], B_DATA=w["Q_U"][1],
                                 batched=True)
        MULT_DRAM = run_eltwise_op_dram(ue, GATE_DRAM, UP_DRAM,
                                        numel=T * mlp_dim, op="mul", batched=True)
        DOWN_DRAM = run_fp4_matmul(ue, M=T, K=mlp_dim, N=H,
                                   A_DRAM=MULT_DRAM,
                                   B_SCALE=w["Q_D"][0], B_DATA=w["Q_D"][1],
                                   batched=True)
        FINAL_DRAM = run_eltwise_op_dram(ue, RESIDUAL_DRAM, DOWN_DRAM,
                                         numel=T * H, op="add", batched=True)
        _exec_captured(ue, timeout=300.0)
        return FINAL_DRAM

    # ================================================================
    # === PREFILL PATH (T>1): unchanged multi-capture with host barriers ===
    # ================================================================

    # === Capture A: prenorm + Q/K/V projections ===
    ue.start_capture()
    PRENORM_DRAM = run_rms_norm_dram(ue, M=T, N=H,
                                     X_DRAM=X_DRAM, GAMMA_DRAM=w["GAMMA_INPUT"],
                                     batched=True)
    Q_PROJ = run_fp4_matmul(ue, M=T, K=H, N=q_proj_out,
                            A_DRAM=PRENORM_DRAM,
                            B_SCALE=w["Q_Q"][0], B_DATA=w["Q_Q"][1],
                            batched=True)
    K_PROJ = run_fp4_matmul(ue, M=T, K=H, N=kv_size,
                            A_DRAM=PRENORM_DRAM,
                            B_SCALE=w["Q_K"][0], B_DATA=w["Q_K"][1],
                            batched=True)
    V_PROJ = run_fp4_matmul(ue, M=T, K=H, N=kv_size,
                            A_DRAM=PRENORM_DRAM,
                            B_SCALE=w["Q_V"][0], B_DATA=w["Q_V"][1],
                            batched=True)
    _exec_captured(ue)

    # === Host: split query+gate halves, upload query half ===
    q_raw = _read_bf(ue, Q_PROJ, (T, num_q_heads, full_head_dim * 2))
    q_host_preN, gate_host = q_raw[..., :full_head_dim], q_raw[..., full_head_dim:]
    gate_host_flat = gate_host.reshape(T, q_size_flat)
    Q_PRE_NORM_DRAM = _alloc_tensor(ue, T * num_q_heads * full_head_dim)
    ue.dma_write(DMA_DEVICE_H2C, Q_PRE_NORM_DRAM,
                 _bf(q_host_preN.contiguous()),
                 T * num_q_heads * full_head_dim * BF16)

    # === Capture B: Q/K norms (read their outputs to host next) ===
    ue.start_capture()
    Q_NORM = run_rms_norm_dram(ue, M=T * num_q_heads, N=full_head_dim,
                               X_DRAM=Q_PRE_NORM_DRAM,
                               GAMMA_DRAM=w["GAMMA_Q_N"], batched=True)
    K_NORM = run_rms_norm_dram(ue, M=T * num_kv_heads, N=full_head_dim,
                               X_DRAM=K_PROJ, GAMMA_DRAM=w["GAMMA_K_N"],
                               batched=True)
    _exec_captured(ue)

    # === Host: apply partial RoPE, GQA-duplicate K/V, assemble flash input ===
    q_host = _read_bf(ue, Q_NORM, (T, num_q_heads, full_head_dim))
    k_host = _read_bf(ue, K_NORM, (T, num_kv_heads, full_head_dim))
    v_host = _read_bf(ue, V_PROJ, (T, num_kv_heads, full_head_dim))
    q_host = _apply_partial_rotary(q_host, rope_theta, rotary_dim,
                                   pos_start=pos_start)
    k_host = _apply_partial_rotary(k_host, rope_theta, rotary_dim,
                                   pos_start=pos_start)
    k_host = k_host.repeat_interleave(group_sz, dim=1)
    v_host = v_host.repeat_interleave(group_sz, dim=1)

    q_flash_full = torch.zeros(num_q_heads, T_aligned, full_head_dim,
                               dtype=q_host.dtype)
    q_flash_full[:, pos_start:end_pos, :] = q_host.permute(1, 0, 2)
    Q_FLASH_DRAM = _alloc_tensor(ue, num_q_heads * T_aligned * full_head_dim)
    ue.dma_write(DMA_DEVICE_H2C, Q_FLASH_DRAM, _bf(q_flash_full),
                 num_q_heads * T_aligned * full_head_dim * BF16)

    k_new = k_host.permute(1, 0, 2).contiguous()
    v_new = v_host.permute(1, 0, 2).contiguous()
    for h in range(num_q_heads):
        off = h * head_stride_bytes + pos_start * row_bytes
        ue.dma_write(DMA_DEVICE_H2C, K_cache + off,
                     _bf(k_new[h]), T * row_bytes)
        ue.dma_write(DMA_DEVICE_H2C, V_cache + off,
                     _bf(v_new[h]), T * row_bytes)

    bias_host = torch.full((T_aligned, T_aligned), float("-inf"),
                           dtype=torch.bfloat16)
    bias_host.masked_fill_(
        torch.tril(torch.ones(T_aligned, T_aligned, dtype=torch.bool)), 0.0)
    bias_host[:, end_pos:] = float("-inf")
    ue.dma_write(DMA_DEVICE_H2C, ue._bias_dram, _bf(bias_host),
                 T_aligned * T_aligned * BF16)

    FLASH_OUT_DRAM = _alloc_tensor(ue, num_q_heads * T_aligned * full_head_dim)
    SCRATCH_DRAM   = _alloc_tensor(ue,
                                   T_aligned * T_aligned * 4
                                   + T_aligned * full_head_dim * 4)

    # === Capture C: flash attention per head ===
    ue.start_capture()
    key = ("flash", layer_idx, Q_FLASH_DRAM, K_cache, V_cache, FLASH_OUT_DRAM,
           SCRATCH_DRAM, ue._bias_dram, T_aligned)

    def _emit_flash():
        for h in range(num_q_heads):
            ue.flash_attention_core_cached(
                head_dim=full_head_dim, seq_len=T_aligned,
                Q_DRAM_ADDR=Q_FLASH_DRAM + h * head_stride_bytes,
                K_DRAM_ADDR=K_cache + h * head_stride_bytes,
                V_DRAM_ADDR=V_cache + h * head_stride_bytes,
                OUTPUT_DRAM_ADDR=FLASH_OUT_DRAM + h * head_stride_bytes,
                SCRATCH_DRAM_ADDR=SCRATCH_DRAM,
                BIAS_DRAM_ADDR=ue._bias_dram,
            )

    _cached_emit(ue, key, _emit_flash)
    _exec_captured(ue, timeout=300.0)

    # === Host: extract attn output, apply sigmoid(gate), upload ===
    attn_host = _read_bf(ue, FLASH_OUT_DRAM,
                         (num_q_heads, T_aligned, full_head_dim))
    attn_host = (attn_host[:, pos_start:end_pos, :]
                 .permute(1, 0, 2).contiguous()
                 .reshape(T, q_size_flat))
    attn_host = attn_host * torch.sigmoid(gate_host_flat.float())
    ATTN_CONCAT_DRAM = _alloc_tensor(ue, T * q_size_flat)
    ue.dma_write(DMA_DEVICE_H2C, ATTN_CONCAT_DRAM, _bf(attn_host),
                 T * q_size_flat * BF16)

    # === Capture D: out_proj + residual + post_ln + MLP + residual ===
    ue.start_capture()
    OUT_PROJ_DRAM = run_fp4_matmul(ue, M=T, K=q_size_flat, N=H,
                                   A_DRAM=ATTN_CONCAT_DRAM,
                                   B_SCALE=w["Q_O"][0], B_DATA=w["Q_O"][1],
                                   batched=True)
    RESIDUAL_DRAM = run_eltwise_op_dram(ue, X_DRAM, OUT_PROJ_DRAM,
                                        numel=T * H, op="add", batched=True)
    POST_LN_DRAM = run_rms_norm_dram(ue, M=T, N=H,
                                     X_DRAM=RESIDUAL_DRAM,
                                     GAMMA_DRAM=w["GAMMA_POST"], batched=True)
    GATE_DRAM = run_fp4_matmul(ue, M=T, K=H, N=mlp_dim,
                               A_DRAM=POST_LN_DRAM,
                               B_SCALE=w["Q_G"][0], B_DATA=w["Q_G"][1],
                               silu=True, batched=True)
    UP_DRAM = run_fp4_matmul(ue, M=T, K=H, N=mlp_dim,
                             A_DRAM=POST_LN_DRAM,
                             B_SCALE=w["Q_U"][0], B_DATA=w["Q_U"][1],
                             batched=True)
    MULT_DRAM = run_eltwise_op_dram(ue, GATE_DRAM, UP_DRAM,
                                    numel=T * mlp_dim, op="mul", batched=True)
    DOWN_DRAM = run_fp4_matmul(ue, M=T, K=mlp_dim, N=H,
                               A_DRAM=MULT_DRAM,
                               B_SCALE=w["Q_D"][0], B_DATA=w["Q_D"][1],
                               batched=True)
    FINAL_DRAM = run_eltwise_op_dram(ue, RESIDUAL_DRAM, DOWN_DRAM,
                                     numel=T * H, op="add", batched=True)
    _exec_captured(ue)
    return FINAL_DRAM


# ============================================================================
# HF model loader (patches CUDA/Triton kernels with pure-pytorch equivalents)
# ============================================================================

def _load_hf_model(model_path: str = MODEL_PATH):
    """Load Qwen3.5-2B in bf16 on CPU with all Triton/CUDA kernels replaced by
    their torch equivalents (so HF weights can still be used on CPU host for
    weight extraction — the actual forward runs on FPGA)."""
    from transformers import AutoModelForImageTextToText
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5RMSNormGated, torch_recurrent_gated_delta_rule,
    )
    hf = AutoModelForImageTextToText.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16).eval()
    for path_try in [("model", "language_model"), ("model",), ("language_model",)]:
        obj = hf
        try:
            for p in path_try:
                obj = getattr(obj, p)
            if hasattr(obj, "layers") and len(obj.layers):
                text_model = obj
                break
        except AttributeError:
            continue
    for lyr in text_model.layers:
        la_m = getattr(lyr, "linear_attn", None) or getattr(lyr, "self_attn", None)
        if la_m is None:
            continue
        if hasattr(la_m, "norm") and type(la_m.norm).__name__ != "Qwen3_5RMSNormGated":
            n = Qwen3_5RMSNormGated(la_m.head_v_dim, eps=la_m.layer_norm_epsilon)
            n.weight.data = la_m.norm.weight.data.clone()
            la_m.norm = n
        if hasattr(la_m, "causal_conv1d_fn"):
            la_m.causal_conv1d_fn = None
        if hasattr(la_m, "chunk_gated_delta_rule"):
            la_m.chunk_gated_delta_rule = torch_recurrent_gated_delta_rule
        if hasattr(la_m, "recurrent_gated_delta_rule"):
            la_m.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
    return hf, text_model


# ============================================================================
# Top-level forward: embed → 24 layers → final norm → LM head
# ============================================================================

def _forward_range(ue: Qwen3_5_2b_UnifiedEngine, token_ids: torch.Tensor,
                   pos_start: int, zero_s: bool) -> torch.Tensor:
    """Run the 24-layer stack over T tokens starting at absolute position
    `pos_start`, using pre-uploaded weights and persistent caches.  Returns
    [T, vocab] logits.

    Tensor DRAM is reset before every layer (transient activations only);
    params DRAM cursor is NOT touched (weights live permanently).
    """
    assert token_ids.ndim == 1
    T = int(token_ids.numel())
    H = ue.hidden_size

    # Embedding lookup on host (bf16 table gather).
    x_host = ue._embed_weight[token_ids].to(torch.float32)   # [T, H]

    with _quiet():
        ue.reset_tensor_dram_addr()
        X_DRAM = _alloc_tensor(ue, T * H)
        ue._decoder_X_dram = X_DRAM   # record for compile_decoder / run_decoder
        if not getattr(ue, '_compile_mode', False):
            ue.dma_write(DMA_DEVICE_H2C, X_DRAM, _bf(x_host), T * H * BF16)

    cur_dram = X_DRAM
    for idx in range(ue.num_layers):
        with _quiet():
            if T > 1:
                # Prefill: read activation to host, reset scratch, re-upload.
                x_prev = _read_bf(ue, cur_dram, (T, H))
                ue.reset_tensor_dram_addr()
                NEXT_X = _alloc_tensor(ue, T * H)
                ue.dma_write(DMA_DEVICE_H2C, NEXT_X, _bf(x_prev), T * H * BF16)
            else:
                # T=1 decode: pass FPGA-side result directly — zero host round-trip.
                # Scratch accumulates across layers (24 layers × small T=1 scratch ≪ DRAM cap).
                NEXT_X = cur_dram
            if idx in ue.linear_attn_layers:
                cur_dram = _run_linear_attn_layer(
                    ue, NEXT_X, idx, T, zero_s=zero_s)
            else:
                cur_dram = _run_full_attn_layer(
                    ue, NEXT_X, idx, T, pos_start=pos_start)

    with _quiet():
        FINAL_NORM_DRAM = run_rms_norm_dram(ue, M=T, N=H,
                                            X_DRAM=cur_dram,
                                            GAMMA_DRAM=ue._final_norm_gamma)
        ue._decoder_final_norm_dram = FINAL_NORM_DRAM  # record for compile_decoder / run_decoder
        if getattr(ue, '_compile_mode', False):
            return None  # logits computed on host in run_decoder() after binary execution
        final_norm_hw = _read_bf(ue, FINAL_NORM_DRAM, (T, H))

    # LM head: tied to embedding weight (tie_word_embeddings=True).
    logits = final_norm_hw.float() @ ue._embed_weight.to(torch.float32).t()
    return logits


def prefill_via_decode(ue: Qwen3_5_2b_UnifiedEngine,
                       token_ids: torch.Tensor,
                       verbose: bool = True,
                       vision_tokens: torch.Tensor = None,
                       image_token_id: int = None) -> torch.Tensor:
    """Fast prefill: replay the one-shot decoder binary T_prompt times from
    zero state, letting S/KV/conv build up naturally.

    Why this works: the decoder binary is already prompt-agnostic (position
    via ISA register, token via X_DRAM, causal mask via bias DMA), so running
    it from `pos=0` with zeroed S/KV/conv caches is an arithmetic prefill.
    It trades the current host-barrier-heavy T>1 path (host RoPE, GQA-expand,
    host α/β computation, ~1 M Python `Instruction` objects emitted per run)
    for T separate FPGA triggers at ~1.19 s each.

    Caller must have already compiled+loaded the decoder binary so that
    `ue._decoder_prog_addr` / `_decoder_X_dram` / `_decoder_final_norm_dram`
    are set.  Caller is also responsible for bumping `ue._cache_pos` to
    `T_prompt` after this returns — this function doesn't touch it because
    the loop already advances `cache_pos` per step.

    Returns the first generated token id (int): the on-chip LM head + HW argmax
    runs for every prompt token, and this is the last token's argmax.
    """
    T_prompt = int(token_ids.numel())
    H        = ue.hidden_size
    T_al     = ue.max_context_aligned
    row_byt  = ue.full_head_dim * BF16
    rope_byt = ue.full_rotary_dim * BF16
    X_DRAM   = ue._decoder_X_dram
    FN_DRAM  = ue._decoder_final_norm_dram
    embed_w  = ue._embed_weight.to(torch.float32)

    # Zero S, conv state, cache_pos.  KV cache and Q_FLASH start at
    # prepare_inference's zeros and only get overwritten at the positions
    # we actually visit, so residual old values outside [0, T_prompt) are
    # masked by the causal bias anyway.
    ue.reset_state()
    total_hw_us = 0.0
    t_wall_start = time.time()

    if verbose:
        print(f"  Running fast prefill ({T_prompt} tokens via decoder binary) ",
              end="", flush=True)

    image_idx = 0
    with _quiet():
        for i in range(T_prompt):
            tok = int(token_ids[i].item())
            pos = ue._cache_pos
            end_pos = pos + 1

            # Causal bias for current position.
            bias_host = torch.full((T_al, T_al), float("-inf"),
                                   dtype=torch.bfloat16)
            bias_host.masked_fill_(
                torch.tril(torch.ones(T_al, T_al, dtype=torch.bool)), 0.0)
            bias_host[:, end_pos:] = float("-inf")
            ue.dma_write(DMA_DEVICE_H2C, ue._bias_dram, _bf(bias_host),
                         T_al * T_al * BF16)

            # Embedding for current token → X_DRAM.
            # VLM substitution: at image-token positions (tok == image_token_id),
            # use the corresponding row from the host-computed vision-encoder
            # output instead of the embed table row. Phase 5 of the FPGA-VLM
            # port replaces this host-side substitution with on-device DMA.
            if (vision_tokens is not None and image_token_id is not None
                    and tok == image_token_id):
                x_host = vision_tokens[image_idx].unsqueeze(0)
                image_idx += 1
            else:
                x_host = embed_w[tok].unsqueeze(0)
            ue.reset_tensor_dram_addr()
            ue.dma_write(DMA_DEVICE_H2C, X_DRAM, _bf(x_host), H * BF16)

            # Position-dependent ISA registers.
            ue.isa_add_set_core(ue._ISA_POS_K_REG,    ue_35bit_addr_shifter(pos * row_byt))
            ue.isa_add_set_core(ue._ISA_POS_ROPE_REG, ue_35bit_addr_shifter(pos * rope_byt))

            # ONE FPGA trigger per prompt token.
            ue.start_execute_from_dram(ue._decoder_prog_addr)
            t_wait = time.time()
            while ue.is_queue_busy():
                time.sleep(0.001)
                if time.time() - t_wait >= 300.0:
                    break
            total_hw_us += ue.report_latency_in_us()
            ue._cache_pos += 1

            if verbose and (i + 1) % 4 == 0:
                print(".", end="", flush=True)

    # The compiled program ran the on-chip LM head + HW argmax for EVERY prompt
    # token; the LAST token's argmax IS the first generated token (no logit
    # readback — write_back_disable).  The penalty bias is zero throughout
    # prefill (the prompt is never penalized), so this is plain greedy.
    first_token_id = ue.get_arg_max_index(rank=1)

    # Surface timing through the same field the caller's pretty-printer uses.
    ue._step_hw_latency_us = total_hw_us
    if verbose:
        dt = time.time() - t_wall_start
        flops = _theoretical_flops_per_step(ue, 1) * T_prompt
        gflops_hw = (flops / 1e9) / max(total_hw_us * 1e-6, 1e-9)
        print(f" done ({dt:.1f}s, {flops/1e9:.1f} GFLOP, "
              f"{gflops_hw:.1f} GFLOP/s HW)")
    return first_token_id


# ============================================================================
# Sampling + generation
# ============================================================================

def _sample_next(logits_row: torch.Tensor, temperature: float, top_k: int) -> int:
    """logits_row: [vocab] float tensor.  Returns int token id.  Host-side
    reference only — the production decode/prefill paths sample via the on-FPGA
    LM-head argmax with an optional repetition penalty."""
    if temperature <= 0.0:
        return int(torch.argmax(logits_row).item())
    logits = logits_row.float() / temperature
    if top_k and top_k > 0:
        topv, topi = torch.topk(logits, min(top_k, logits.numel()))
        mask = torch.full_like(logits, float("-inf"))
        mask[topi] = topv
        logits = mask
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def generate(ue: Qwen3_5_2b_UnifiedEngine, tokenizer,
             prompt: str, max_new_tokens: int = 128,
             temperature: float = 0.0, top_k: int = 0,
             verbose: bool = True,
             processor=None, image=None,
             model_dir: str = None,
             precomputed_vision_tokens: torch.Tensor = None,
             decoder_preloaded: bool = False) -> str:
    """Prefill-then-decode greedy/top-k sampling.

    Prefill replays the decoder binary T_prompt times from zero state so the
    S / KV / conv caches build up as an exact equivalent of a T>1 forward
    pass, but with zero host barriers inside the layer — see
    `prefill_via_decode()` and the "Fast prefill via decoder-binary replay"
    section in `template/notes/notes_qwen3.5_2b.md`.  After prefill, each
    decode step is one FPGA trigger.
    """
    # Wrap the user prompt in the chat template (user role + assistant
    # generation prompt) — same convention Gemma4 uses for its LM prompt path.
    # VLM mode (processor + image given): tokenize via HF processor so
    # `<|image_pad|>` expands to the right number of image tokens for the
    # input image's grid, and run the host-side vision encoder.
    ids, hf_inputs = _tokenize_with_chat_template(
        tokenizer, prompt, processor=processor, image=image)
    vision_tokens = None
    image_token_id = None
    if hf_inputs is not None:
        image_token_id = int(processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        if precomputed_vision_tokens is not None:
            vision_tokens = precomputed_vision_tokens
            if verbose:
                print(f"  Using FPGA-computed vision tokens "
                      f"({tuple(vision_tokens.shape)} merged tokens).")
        else:
            if verbose:
                print(f"  Running host-side HF vision encoder ...", end=" ", flush=True)
            t_v = time.time()
            vision_tokens = _run_hf_vision_host(model_dir, hf_inputs)
            if verbose:
                print(f"done ({time.time()-t_v:.1f}s, {tuple(vision_tokens.shape)} merged tokens).")
    T_prompt = int(ids.numel())
    # max_new_tokens=0 means "run until EOT or the cache is full".  Expand
    # it to the remaining KV-cache budget; the decode loop short-circuits on
    # EOT so the actual length will usually be much shorter.
    if max_new_tokens <= 0:
        max_new_tokens = ue.max_context - T_prompt
    assert T_prompt + max_new_tokens <= ue.max_context, (
        f"prompt ({T_prompt}) + new ({max_new_tokens}) exceeds "
        f"max_context={ue.max_context}; pass --max-context")

    if verbose:
        print()
        print("  " + "-" * 74)
        print(f"  PREFILL  (T={T_prompt} prompt tokens → seeds S cache + KV cache)")
        print("  " + "-" * 74)
        print(f"  Prompt:         {prompt!r}")
        print(f"  Prompt tokens:  {ids.tolist()}")

    # Load the single-binary decoder up-front (prefill replays it per prompt
    # token, so it must be resident first).  ``decoder_preloaded`` skips this when
    # the caller already DMA'd the decoder section from the unified bin (the
    # load-only path) — the decoder is read from programs.bin's decoder section.
    if not decoder_preloaded:
        if verbose:
            print(f"  Compiling / loading decoder binary ...", end=" ", flush=True)
        t_c0 = time.time()
        with _quiet():
            bin_path, _sizes, _flops_list = ue.compile_decoder()
            ue.load_instructions(bin_path)
        if verbose:
            bin_mb = os.path.getsize(bin_path) / (1024 * 1024)
            print(f"done ({time.time()-t_c0:.1f}s, {bin_mb:.1f} MB binary).")

    # Token-by-token replay of the decoder binary from zero state. This
    # produces S directly in [Dv, Dk] decode form.
    nxt = prefill_via_decode(ue, ids, verbose=verbose,
                             vision_tokens=vision_tokens,
                             image_token_id=image_token_id)

    eot = ue._cfg["model"].get("end_of_turn_token_id", None)

    # Clear instruction-emission caches before switching to the T=1 path.
    ue._primitive_cache.clear()
    ue._rec_cache.clear()
    ue._exec_cache.clear()
    ue._patched_bytes_cache.clear()
    ue._exec_idx = 0

    # Delegate to run_decoder (one-shot: one FPGA trigger per token).
    result = ue.run_decoder(
        tokenizer, nxt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        verbose=verbose,
    )
    return result["generated_text"]


# ============================================================================
# Vision encoder (merged from vision_encoder_fpga.py). COMPACT single-capture
# §3b PBI + §7 shared flash (default, ~0.82 MB, FPGA-validated); VIS_LEGACY=1
# falls back to the unrolled multi-capture cascade. See notes_qwen3.5_2b.md.
# Reuses this module's _bf / _upload_fp4 / quantize_fp4_64.
# ============================================================================
def _ln_pbi_nchunks(M: int, N: int, has_gamma: bool = True, has_beta: bool = True) -> int:
    """Replicate layer_norm_core_dram_pbi's chunk math. The PBI layernorm's
    gpr_M_reg must be primed with the CHUNK COUNT (M // chunk_size), NOT M
    (unlike matmat_mul_core whose gpr_M_reg is the row count). Getting this
    wrong over-runs the row loop and produces garbage."""
    ops_per_row = 4 + (1 if has_gamma else 0) + (1 if has_beta else 0)
    max_chunk_from_icache = (256 - 4) // ops_per_row
    chunk_size = min(URAM_NEAR_FULL_ELEMENTS // N, M, max_chunk_from_icache)
    while M % chunk_size != 0:
        chunk_size -= 1
    return M // chunk_size

def _read_dram_bf16(ue, dram_addr: int, shape) -> torch.Tensor:
    nelem = 1
    for d in shape:
        nelem *= d
    out = torch.zeros(nelem, dtype=torch.bfloat16)
    ue.dma_read(DMA_DEVICE_C2H, dram_addr, out, nelem * BF16)
    return out.reshape(*shape)

def _upload_bf16(ue, t: torch.Tensor, nelem: int) -> int:
    addr = ue.allocate_params_dram(nelem * BF16)
    ue.dma_write(DMA_DEVICE_H2C, addr, _bf(t), nelem * BF16)
    return addr

def run_fpga_vision_encoder(ue, model_dir: str, hf_inputs, setup_only: bool = False,
                            build_only: bool = False):
    """Run the compact single-capture vision encoder STANDALONE (build + execute
    at program base) and return the merged ``[N_merged, fc2_dim]`` tokens; the
    encoder bytes are stashed on ``ue._vis_encoder_bytes`` for the unified bin.
    ``setup_only=True`` only uploads the vision weights + per-image inputs for
    the bin-load path and returns None."""
    if os.environ.get("VIS_LEGACY"):
        return _run_fpga_vision_encoder_legacy(ue, model_dir, hf_inputs)
    return _run_fpga_vision_encoder_compact(ue, model_dir, hf_inputs, setup_only, build_only)


UNIFIED_BIN_NAME  = "programs.bin"
UNIFIED_META_NAME = "programs.json"


def write_unified_bin(ue, bin_dir: str, encoder_bytes: bytes, vis_meta: dict,
                      decoder_bin_path: str, flops) -> dict:
    """Assemble the UNIFIED programs bin + manifest = ``[encoder][decoder]``,
    both baked for the SAME program base (0xD0000000) and run SEQUENTIALLY with a
    DRAM reset between them (vision encoder → reset → LM decoder) — never resident
    together, so no base-shift / coexistence issue.

    The encoder's only compile-time constant (the LayerNorm zeros base) is a shared
    ``VIS_ZEROS`` buffer seeded in ``setup_only`` (via the kernel's ``ZEROS_DRAM_ADDR``
    arg), so it is re-created on the bin-load path with no extra bin section.

    IMPORTANT: ``prepare_inference`` must be called EXACTLY ONCE per engine and
    AFTER the vision encoder runs — a second prepare (or vision executed after
    prepare) silently corrupts the LM decode.
    """
    uni_bin  = os.path.join(bin_dir, UNIFIED_BIN_NAME)
    uni_meta = os.path.join(bin_dir, UNIFIED_META_NAME)
    decoder_bytes = open(decoder_bin_path, "rb").read()
    decoder_size  = len(decoder_bytes)
    encoder_size  = len(encoder_bytes)

    raw = bytearray(encoder_bytes)
    raw.extend(decoder_bytes)
    programs = {}
    if encoder_size:
        programs["encoder"] = {"offset": 0, "size": encoder_size}
    programs["decoder"] = {"offset": encoder_size, "size": decoder_size}
    meta = {
        "layout": "encoder|decoder @base",
        "programs": programs,
        # Back-compat duplicate fields (offset/size pairs) so callers using the
        # flat keys keep working until they migrate to ``programs[...]``.
        "encoder_off": 0,            "encoder_size": encoder_size,
        "decoder_off": encoder_size, "decoder_size": decoder_size,
        "decoder_flops": int(flops[0]),
        "decoder_x_dram": ue._decoder_X_dram,
        "decoder_final_norm_dram": ue._decoder_final_norm_dram,
        "max_context": ue.max_context,
        "n_patches": vis_meta.get("n_patches"),
        "mg_out":    vis_meta.get("mg_out"),
        "n_merged":  vis_meta.get("n_merged"),
        "fc2_dim":   vis_meta.get("fc2_dim"),
    }
    with open(uni_bin, "wb") as f:
        f.write(bytes(raw))
    with open(uni_meta, "w") as f:
        json.dump(meta, f, indent=0)
    print(f"  unified bin {len(raw)/1024/1024:.2f} MB "
          f"(encoder {encoder_size/1024/1024:.2f} + decoder {decoder_size/1024/1024:.2f}) "
          f"→ {UNIFIED_BIN_NAME}")
    return meta


def run_vision_from_bin(ue, raw: bytes, meta: dict, model_dir: str, hf_inputs) -> torch.Tensor:
    """Execute the vision-encoder SECTION of a loaded unified bin (no rebuild):
    upload vision weights/inputs incl. the shared VIS_ZEROS buffer (setup_only) →
    DMA the encoder program to base 0xD0000000 → trigger → read merged tokens.
    Must run BEFORE the single prepare_inference (encoder owns params at base)."""
    run_fpga_vision_encoder(ue, model_dir, hf_inputs, setup_only=True)
    ue.reset_program_dram_addr()
    enc_addr = ue.get_program_dram_addr()
    ue.dma_write(DMA_DEVICE_H2C, enc_addr,
                 raw[meta["encoder_off"]:meta["encoder_off"] + meta["encoder_size"]],
                 meta["encoder_size"])
    ue.allocate_program_dram(meta["encoder_size"])
    ue.start_execute_from_dram(enc_addr)
    ue.wait_queue(120.0)
    return _read_dram_bf16(ue, ue._vis_encoder_out_dram, (ue._vis_n_merged, ue._vis_fc2_dim))


def load_decoder_from_bin(ue, raw: bytes, meta: dict) -> None:
    """DMA the LM-decoder SECTION of a loaded unified bin to program base
    0xD0000000 (no recompilation) and restore the transient decode I/O addresses
    so prefill/decode run exactly as a compiled decoder would. Call AFTER the
    single prepare_inference; pair with ``generate(..., decoder_preloaded=True)``."""
    ue.reset_program_dram_addr()
    dec_addr = ue.get_program_dram_addr()
    ue.dma_write(DMA_DEVICE_H2C, dec_addr,
                 raw[meta["decoder_off"]:meta["decoder_off"] + meta["decoder_size"]],
                 meta["decoder_size"])
    ue.allocate_program_dram(meta["decoder_size"])
    ue._decoder_prog_addr        = dec_addr
    ue._decoder_X_dram           = meta["decoder_x_dram"]
    ue._decoder_final_norm_dram  = meta["decoder_final_norm_dram"]


def _run_fpga_vision_encoder_compact(ue, model_dir: str, hf_inputs, setup_only: bool = False,
                                     build_only: bool = False):
    from transformers import AutoModelForImageTextToText
    print(f"  [vision-compact] Loading HF visual ...", end=" ", flush=True)
    t_load = time.time()
    hf = AutoModelForImageTextToText.from_pretrained(
        model_dir, dtype=torch.bfloat16, local_files_only=True).eval()
    visual = hf.model.visual
    print(f"({time.time()-t_load:.1f}s)")

    pixel_values = hf_inputs["pixel_values"].to(torch.bfloat16)
    grid_thw     = hf_inputs["image_grid_thw"]
    N_P    = pixel_values.shape[0]
    HIDDEN, K_PIXEL = 1024, 1536
    N_HEADS, HEAD_D = 16, 64
    QKV_DIM, MLP_INT = 3 * HIDDEN, 4096
    DEPTH = len(visual.blocks)
    assert N_P % UE_VECTOR_SIZE == 0, (
        f"compact vision path requires N_P ({N_P}) to be a multiple of "
        f"{UE_VECTOR_SIZE} (no attention bias mask; pad upstream or use VIS_LEGACY=1)")
    half_d = HEAD_D // 2

    # Patch embed (Conv3d-as-matmul) + pos embed (host-computed).
    pe_w = visual.patch_embed.proj.weight.detach().view(HIDDEN, K_PIXEL).to(torch.bfloat16)
    pe_b = visual.patch_embed.proj.bias.detach().to(torch.bfloat16)
    with torch.no_grad():
        pos_embeds = visual.fast_pos_embed_interpolate(grid_thw).to(torch.bfloat16)

    # --- RoPE table: rotate-half, [N_P, 2, HEAD_D] (cos||sin), tiled N_HEADS ---
    # Matches qwen2.5_vl_3b (FPGA-proven): row t = [cos(HEAD_D) || sin(HEAD_D)]
    # with cos=[cos,cos], sin=[-sin,+sin]; tiled per head so a single
    # M=N_HEADS*N_P rope_hf_core_dram covers every head's token t.
    with torch.no_grad():
        rot = visual.rot_pos_emb(grid_thw).reshape(N_P, -1)          # [N_P, half_d]
        cos_raw = rot.cos().to(torch.bfloat16)                        # [N_P, half_d]
        sin_raw = rot.sin().to(torch.bfloat16)
        rope_table = torch.empty(N_P, 2, HEAD_D, dtype=torch.bfloat16)
        ct, st = rope_table[:, 0, :], rope_table[:, 1, :]
        ct[:, :half_d] = cos_raw;  ct[:, half_d:HEAD_D] = cos_raw
        st[:, :half_d] = -sin_raw; st[:, half_d:HEAD_D] = sin_raw
        rope_tiled = rope_table.repeat(N_HEADS, 1, 1).flatten()       # [N_HEADS*N_P*2*HEAD_D]

    print(f"  [vision-compact] Uploading weights (FP4_64) ...", end=" ", flush=True)
    t_w = time.time()
    # --- Tensor DRAM: intermediates + fixed §7 flash buffers + rope table ---
    PIXEL          = ue.allocate_tensor_dram(N_P * K_PIXEL * BF16)
    PATCH_OUT      = ue.allocate_tensor_dram(N_P * HIDDEN * BF16)
    BLOCK_OUT      = ue.allocate_tensor_dram(N_P * HIDDEN * BF16)   # layer I/O (also patch+pos)
    LN1_OUT        = ue.allocate_tensor_dram(N_P * HIDDEN * BF16)
    QKV_OUT        = ue.allocate_tensor_dram(N_P * QKV_DIM * BF16)
    LN2_OUT        = ue.allocate_tensor_dram(N_P * HIDDEN * BF16)
    head_bytes     = N_P * HEAD_D * BF16
    Q_PERM         = ue.allocate_tensor_dram(N_HEADS * head_bytes)  # head-major [N_HEADS,N_P,HEAD_D]
    K_PERM         = ue.allocate_tensor_dram(N_HEADS * head_bytes)
    V_PERM         = ue.allocate_tensor_dram(N_HEADS * head_bytes)
    ATTN_OUT       = ue.allocate_tensor_dram(N_HEADS * head_bytes)
    ATTN_CONCAT    = ue.allocate_tensor_dram(N_P * HIDDEN * BF16)
    ATTN_PROJ      = ue.allocate_tensor_dram(N_P * HIDDEN * BF16)
    POST_ATTN      = ue.allocate_tensor_dram(N_P * HIDDEN * BF16)
    FC1_GELU_OUT   = ue.allocate_tensor_dram(N_P * MLP_INT * BF16)
    FC2_OUT        = ue.allocate_tensor_dram(N_P * HIDDEN * BF16)
    # Shared flash fixed operand buffers (sized for the full N_P segment).
    VIS_FLASH_Q    = ue.allocate_tensor_dram(N_P * HEAD_D * BF16)
    VIS_FLASH_K    = ue.allocate_tensor_dram(N_P * HEAD_D * BF16)
    VIS_FLASH_V    = ue.allocate_tensor_dram(N_P * HEAD_D * BF16)
    VIS_FLASH_OUT  = ue.allocate_tensor_dram(N_P * HEAD_D * BF16)
    ATTN_SCRATCH   = ue.allocate_tensor_dram((HEAD_D + N_P) * N_P * BF16 + N_P * HEAD_D * BF16)
    VIS_ATTN_P     = ue.allocate_tensor_dram(N_P * N_P * BF16)
    VIS_BIAS       = ue.allocate_tensor_dram(N_P * N_P * BF16)
    VIS_EYE        = ue.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * BF16)
    VIS_ZEROS      = ue.allocate_tensor_dram(HIDDEN * BF16)   # shared LayerNorm zeros base
    VIS_ROPE       = ue.allocate_tensor_dram(N_HEADS * N_P * 2 * HEAD_D * BF16)
    VIS_ROPE_COS   = VIS_ROPE
    VIS_ROPE_SIN   = VIS_ROPE + HEAD_D * BF16
    # Merger intermediates.
    N_M     = N_P // 4
    K_FC    = HIDDEN * 4    # 4096
    FC2_DIM = 2048
    MG_LN_OUT  = ue.allocate_tensor_dram(N_P * HIDDEN * BF16)
    MG_FC1_OUT = ue.allocate_tensor_dram(N_M * K_FC * BF16)
    MG_OUT     = ue.allocate_tensor_dram(N_M * FC2_DIM * BF16)

    # --- Params DRAM: BF16 (norms/biases/patch/pos) + FP4 weights ---
    PATCH_W = _upload_bf16(ue, pe_w, HIDDEN * K_PIXEL)
    PATCH_B = _upload_bf16(ue, pe_b, HIDDEN)
    POS     = _upload_bf16(ue, pos_embeds, N_P * HIDDEN)
    layer_addrs = []
    for li in range(DEPTH):
        bi = visual.blocks[li]
        L = {
            "LN1_W": _upload_bf16(ue, bi.norm1.weight, HIDDEN),
            "LN1_B": _upload_bf16(ue, bi.norm1.bias,   HIDDEN),
            "QKV_B": _upload_bf16(ue, bi.attn.qkv.bias, QKV_DIM),
            "PROJ_B": _upload_bf16(ue, bi.attn.proj.bias, HIDDEN),
            "LN2_W": _upload_bf16(ue, bi.norm2.weight, HIDDEN),
            "LN2_B": _upload_bf16(ue, bi.norm2.bias,   HIDDEN),
            "FC1_B": _upload_bf16(ue, bi.mlp.linear_fc1.bias, MLP_INT),
            "FC2_B": _upload_bf16(ue, bi.mlp.linear_fc2.bias, HIDDEN),
        }
        L["QKV_W_SCALE"], L["QKV_W_DATA"]   = _upload_fp4(ue, bi.attn.qkv.weight.detach().float())
        L["PROJ_W_SCALE"], L["PROJ_W_DATA"] = _upload_fp4(ue, bi.attn.proj.weight.detach().float())
        L["FC1_W_SCALE"], L["FC1_W_DATA"]   = _upload_fp4(ue, bi.mlp.linear_fc1.weight.detach().float())
        L["FC2_W_SCALE"], L["FC2_W_DATA"]   = _upload_fp4(ue, bi.mlp.linear_fc2.weight.detach().float())
        layer_addrs.append(L)
    mg = visual.merger
    MG_LN_W  = _upload_bf16(ue, mg.norm.weight, HIDDEN)
    MG_LN_B  = _upload_bf16(ue, mg.norm.bias,   HIDDEN)
    MG_FC1_B = _upload_bf16(ue, mg.linear_fc1.bias, K_FC)
    MG_FC2_B = _upload_bf16(ue, mg.linear_fc2.bias, FC2_DIM)
    MG_FC1_S, MG_FC1_D = _upload_fp4(ue, mg.linear_fc1.weight.detach().float())
    MG_FC2_S, MG_FC2_D = _upload_fp4(ue, mg.linear_fc2.weight.detach().float())

    # Inputs.
    ue.dma_write(DMA_DEVICE_H2C, PIXEL, _bf(pixel_values), N_P * K_PIXEL * BF16)
    ue.dma_write(DMA_DEVICE_H2C, VIS_EYE,
                 _bf(torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16)),
                 UE_VECTOR_SIZE * UE_VECTOR_SIZE * BF16)
    ue.dma_write(DMA_DEVICE_H2C, VIS_BIAS,
                 _bf(torch.zeros(N_P, N_P, dtype=torch.bfloat16)),
                 N_P * N_P * BF16)
    ue.dma_write(DMA_DEVICE_H2C, VIS_ZEROS,
                 _bf(torch.zeros(HIDDEN, dtype=torch.bfloat16)), HIDDEN * BF16)
    ue.dma_write(DMA_DEVICE_H2C, VIS_ROPE, _bf(rope_tiled),
                 N_HEADS * N_P * 2 * HEAD_D * BF16)
    del hf
    print(f"({time.time()-t_w:.1f}s)")

    # setup_only (bin-load path in run_from_bin): vision weights +
    # per-image inputs are now resident at the same deterministic addresses the
    # prebuilt encoder section was baked against. No emit/execute — the caller
    # DMAs the encoder bytes from the bin and triggers them. Record the readback
    # location for the merged tokens.
    if setup_only:
        ue._vis_encoder_out_dram = MG_OUT
        ue._vis_n_merged = N_M
        ue._vis_fc2_dim = FC2_DIM
        ue._vis_n_patches = N_P
        return None

    # ------------------------------------------------------------------
    # ONE captured program.
    # ------------------------------------------------------------------
    print(f"  [vision-compact] Emitting single-capture program ({DEPTH} layers) ...",
          end=" ", flush=True)
    t_e = time.time()
    ue.reset_program_dram_addr()       # encoder owns program DRAM (base 0xD0000000)
    prog_base = ue.get_program_dram_addr()
    ue.reset_isa_reg_counter()
    vis_M      = ue.alloc_isa_reg()   # gpr_M_reg for matmul/norm row count

    head_stride = head_bytes          # per-head block in Q_PERM/K_PERM/V_PERM/ATTN_OUT
    chunk_elems = 65536
    elems_total = N_P * HIDDEN
    # PBI LayerNorm's gpr_M_reg = CHUNK COUNT (M//chunk_size), NOT M. All vision
    # LNs are M=N_P, N=HIDDEN, gamma+beta → one constant n_chunks. Prime a
    # dedicated reg (priming vis_M=N_P here would over-run the row loop → garbage).
    vis_ln_chunks = ue.alloc_isa_reg()
    ln_nchunks = _ln_pbi_nchunks(N_P, HIDDEN)
    # Fallback: Q35_VIS_LN_STATIC=1 → static (legacy) LayerNorm (no gpr_M_reg).
    ln_static = bool(os.environ.get("Q35_VIS_LN_STATIC"))
    ln_reg = None if ln_static else vis_ln_chunks

    def _emit_residual_add(a_dram, b_dram, out_dram):
        done = 0
        while done < elems_total:
            c = min(chunk_elems, elems_total - done)
            off = done * BF16
            ue.accelerator_memory_to_sram(a_dram + off, 0x00000, c)
            ue.accelerator_memory_to_sram(b_dram + off, 0x80000, c)
            ue.eltwise_add_core(0x00000, 0x80000, 0x00000, c)
            ue.sram_to_accelerator_memory(0x00000, out_dram + off, c)
            done += c

    def _vis_flash_call(q_src, k_src, v_src, out_dst):
        elems = N_P * HEAD_D
        for src, dst in ((q_src, VIS_FLASH_Q), (k_src, VIS_FLASH_K), (v_src, VIS_FLASH_V)):
            ue.accelerator_memory_to_sram(src, 0x00000, elems)
            ue.sram_to_accelerator_memory(0x00000, dst, elems)
        ue.flash_attention_core_cached(
            head_dim=HEAD_D, seq_len=N_P,
            Q_DRAM_ADDR=VIS_FLASH_Q, K_DRAM_ADDR=VIS_FLASH_K, V_DRAM_ADDR=VIS_FLASH_V,
            OUTPUT_DRAM_ADDR=VIS_FLASH_OUT, SCRATCH_DRAM_ADDR=ATTN_SCRATCH,
            IDENTITY_DRAM_ADDR=VIS_EYE, BIAS_DRAM_ADDR=VIS_BIAS)
        ue.accelerator_memory_to_sram(VIS_FLASH_OUT, 0x00000, elems)
        ue.sram_to_accelerator_memory(0x00000, out_dst, elems)

    ue.start_capture()
    ue.generate_instruction_add_set(vis_M, N_P)
    if not ln_static:
        ue.generate_instruction_add_set(vis_ln_chunks, ln_nchunks)

    # Phase 2: patch_embed + pos.
    ue.matmat_mul_core(M=N_P, K=K_PIXEL, N=HIDDEN,
        A_DRAM_ADDR=PIXEL, B_DRAM_ADDR=PATCH_W, OUTPUT_DRAM_ADDR=PATCH_OUT,
        C_DRAM_ADDR=PATCH_B, bias_mode="broadcast_N", is_B_quantized=False,
        gpr_M_reg=vis_M)
    _emit_residual_add(PATCH_OUT, POS, BLOCK_OUT)

    # Phase 3: 24 layers (BLOCK_OUT is the running residual stream).
    for li in range(DEPTH):
        L = layer_addrs[li]
        ue.layer_norm_core_dram(M=N_P, N=HIDDEN, A_DRAM_ADDR=BLOCK_OUT,
            OUTPUT_DRAM_ADDR=LN1_OUT, GAMMA_DRAM_ADDR=L["LN1_W"],
            BETA_DRAM_ADDR=L["LN1_B"], gpr_M_reg=ln_reg, ZEROS_DRAM_ADDR=VIS_ZEROS)
        ue.matmat_mul_core(M=N_P, K=HIDDEN, N=QKV_DIM,
            A_DRAM_ADDR=LN1_OUT, B_DRAM_ADDR=L["QKV_W_DATA"], OUTPUT_DRAM_ADDR=QKV_OUT,
            C_DRAM_ADDR=L["QKV_B"], bias_mode="broadcast_N",
            is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=L["QKV_W_SCALE"],
            gpr_M_reg=vis_M)
        # Permute Q,K,V (interleaved [N_P, 3*HIDDEN]) → head-major [N_HEADS, N_P, HEAD_D].
        for h in range(N_HEADS):
            for srcbase, dst in ((0,        Q_PERM),
                                 (HIDDEN,   K_PERM),
                                 (2*HIDDEN, V_PERM)):
                ue.ue_memcpy_from_dram(
                    QKV_OUT + (srcbase + h * HEAD_D) * BF16,
                    N_P * HEAD_D * BF16, MEMCPY_TYPE.URAM.value, URAM_START_ADDR,
                    URAM_SECTION.URAM_A.value,
                    stride_bytes_per_chunk=HEAD_D * BF16,
                    stride_jump_bytes=QKV_DIM * BF16)
                ue.ue_memcpy_to_dram(MEMCPY_TYPE.URAM.value, URAM_SECTION.URAM_A.value,
                    URAM_START_ADDR, dst + h * head_stride, N_P * HEAD_D * BF16)
        # Batched rope on head-major Q,K: one PBI call over all N_HEADS*N_P rows.
        ue.generate_instruction_add_set(vis_M, N_HEADS * N_P)
        for buf in (Q_PERM, K_PERM):
            ue.rope_hf_core_dram(M=N_HEADS * N_P, N=HEAD_D,
                input_dram_addr=buf, output_dram_addr=buf,
                cos_dram_addr=VIS_ROPE_COS, sin_dram_addr=VIS_ROPE_SIN,
                gpr_M_reg=vis_M)
        ue.generate_instruction_add_set(vis_M, N_P)
        # §7 shared flash, per head.
        for h in range(N_HEADS):
            _vis_flash_call(Q_PERM + h * head_stride, K_PERM + h * head_stride,
                            V_PERM + h * head_stride, ATTN_OUT + h * head_stride)
        # Concat head-major ATTN_OUT → token-major ATTN_CONCAT.
        for h in range(N_HEADS):
            ue.ue_memcpy_from_dram(ATTN_OUT + h * head_stride,
                N_P * HEAD_D * BF16, MEMCPY_TYPE.URAM.value, URAM_START_ADDR,
                URAM_SECTION.URAM_A.value)
            ue.ue_memcpy_to_dram(MEMCPY_TYPE.URAM.value, URAM_SECTION.URAM_A.value,
                URAM_START_ADDR, ATTN_CONCAT + h * HEAD_D * BF16, N_P * HEAD_D * BF16,
                stride_bytes_per_chunk=HEAD_D * BF16, stride_jump_bytes=HIDDEN * BF16)
        ue.matmat_mul_core(M=N_P, K=HIDDEN, N=HIDDEN,
            A_DRAM_ADDR=ATTN_CONCAT, B_DRAM_ADDR=L["PROJ_W_DATA"], OUTPUT_DRAM_ADDR=ATTN_PROJ,
            C_DRAM_ADDR=L["PROJ_B"], bias_mode="broadcast_N",
            is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=L["PROJ_W_SCALE"],
            gpr_M_reg=vis_M)
        _emit_residual_add(BLOCK_OUT, ATTN_PROJ, POST_ATTN)
        ue.layer_norm_core_dram(M=N_P, N=HIDDEN, A_DRAM_ADDR=POST_ATTN,
            OUTPUT_DRAM_ADDR=LN2_OUT, GAMMA_DRAM_ADDR=L["LN2_W"],
            BETA_DRAM_ADDR=L["LN2_B"], gpr_M_reg=ln_reg, ZEROS_DRAM_ADDR=VIS_ZEROS)
        ue.matmat_mul_core(M=N_P, K=HIDDEN, N=MLP_INT,
            A_DRAM_ADDR=LN2_OUT, B_DRAM_ADDR=L["FC1_W_DATA"], OUTPUT_DRAM_ADDR=FC1_GELU_OUT,
            C_DRAM_ADDR=L["FC1_B"], bias_mode="broadcast_N",
            is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=L["FC1_W_SCALE"],
            gelu_enable=True, gpr_M_reg=vis_M)
        ue.matmat_mul_core(M=N_P, K=MLP_INT, N=HIDDEN,
            A_DRAM_ADDR=FC1_GELU_OUT, B_DRAM_ADDR=L["FC2_W_DATA"], OUTPUT_DRAM_ADDR=FC2_OUT,
            C_DRAM_ADDR=L["FC2_B"], bias_mode="broadcast_N",
            is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=L["FC2_W_SCALE"],
            gpr_M_reg=vis_M)
        _emit_residual_add(POST_ATTN, FC2_OUT, BLOCK_OUT)

    # Phase 4: merger (spatial-merge implicit via M=N_M, K=K_FC reshape).
    ue.layer_norm_core_dram(M=N_P, N=HIDDEN, A_DRAM_ADDR=BLOCK_OUT,
        OUTPUT_DRAM_ADDR=MG_LN_OUT, GAMMA_DRAM_ADDR=MG_LN_W, BETA_DRAM_ADDR=MG_LN_B,
        gpr_M_reg=ln_reg, ZEROS_DRAM_ADDR=VIS_ZEROS)
    ue.generate_instruction_add_set(vis_M, N_M)
    ue.matmat_mul_core(M=N_M, K=K_FC, N=K_FC,
        A_DRAM_ADDR=MG_LN_OUT, B_DRAM_ADDR=MG_FC1_D, OUTPUT_DRAM_ADDR=MG_FC1_OUT,
        C_DRAM_ADDR=MG_FC1_B, bias_mode="broadcast_N",
        is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=MG_FC1_S,
        gelu_enable=True, gpr_M_reg=vis_M)
    ue.matmat_mul_core(M=N_M, K=K_FC, N=FC2_DIM,
        A_DRAM_ADDR=MG_FC1_OUT, B_DRAM_ADDR=MG_FC2_D, OUTPUT_DRAM_ADDR=MG_OUT,
        C_DRAM_ADDR=MG_FC2_B, bias_mode="broadcast_N",
        is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=MG_FC2_S,
        gpr_M_reg=vis_M)

    ue.generate_instruction_halt()
    ue.stop_capture()

    prog_size = ue.get_capture_instruction_size_bytes()
    print(f"({time.time()-t_e:.1f}s, {prog_size/1024/1024:.2f} MB, "
          f"{ue.capture_count} instr)")

    # Capture the encoder bytes (baked for prog_base) + meta so the caller can
    # write them into the unified bin — the encoder runs STANDALONE (vision
    # weights + program at base), the layout the unified-bin VLM run replays.
    enc_bytes = bytearray()
    for inst in ue.capture_buffer:
        enc_bytes.extend(inst.get_bytes())
    ue._vis_encoder_bytes = bytes(enc_bytes)
    ue._vis_encoder_addr  = prog_base
    ue._vis_encoder_out_dram = MG_OUT
    ue._vis_n_merged = N_M
    ue._vis_fc2_dim  = FC2_DIM
    ue._vis_n_patches = N_P

    if build_only:
        # Comprehensive-bin build with no vision tokens needed (LM-only / host
        # vision): the captured bytes are already stashed for the unified bin —
        # do NOT execute the encoder.
        ue.clear_capture_buffer()
        return None

    ue.write_captured_instructions_to_dram(prog_base)
    ue.allocate_program_dram(prog_size)
    ue.clear_capture_buffer()
    ue.start_execute_from_dram(prog_base)
    ue.wait_queue(120.0)

    return _read_dram_bf16(ue, MG_OUT, (N_M, FC2_DIM))

def _vis_exec_capture(ue):
    ue.generate_instruction_halt()
    ue.stop_capture()
    prog_addr = ue.get_program_dram_addr()
    prog_size = ue.get_capture_instruction_size_bytes()
    ue.write_captured_instructions_to_dram(prog_addr)
    ue.allocate_program_dram(prog_size)
    ue.start_execute_from_dram(prog_addr)
    ue.wait_queue(60.0)
    ue.clear_capture_buffer()
    ue.reset_program_dram_addr()

def _emit_rotary_per_row(ue, src_dram, out_dram,
                         cos_pad_dram, neg_sin_pad_dram, sin_hi_pad_dram,
                         load_cos_sin: bool):
    rot_dim, half_rot = 64, 32
    SA_X_LO,   SA_X_HI    = 0x40000, 0x40080
    SA_OUT_LO, SA_OUT_HI  = 0x40100, 0x40180
    SA_TMP_A              = 0x40200
    SB_COS, SB_NEG_SIN    = 0x80000, 0x80080
    SB_SIN_HI, SB_TMP_B   = 0x80100, 0x80180
    if load_cos_sin:
        ue.accelerator_memory_to_sram(cos_pad_dram,     SB_COS,     rot_dim)
        ue.accelerator_memory_to_sram(neg_sin_pad_dram, SB_NEG_SIN, rot_dim)
        ue.accelerator_memory_to_sram(sin_hi_pad_dram,  SB_SIN_HI,  rot_dim)
    ue.accelerator_memory_to_sram(src_dram,                    SA_X_LO, half_rot)
    ue.accelerator_memory_to_sram(src_dram + half_rot * BF16,  SA_X_HI, half_rot)
    ue.eltwise_mul_core(SA_X_LO, SB_COS,     SB_TMP_B,  rot_dim)
    ue.eltwise_mul_core(SA_X_HI, SB_NEG_SIN, SA_TMP_A,  rot_dim)
    ue.eltwise_add_core(SA_TMP_A, SB_TMP_B,  SA_OUT_LO, rot_dim)
    ue.eltwise_mul_core(SA_X_HI, SB_COS,    SB_TMP_B,  rot_dim)
    ue.eltwise_mul_core(SA_X_LO, SB_SIN_HI, SA_TMP_A,  rot_dim)
    ue.eltwise_add_core(SA_TMP_A, SB_TMP_B, SA_OUT_HI, rot_dim)
    ue.sram_to_accelerator_memory(SA_OUT_LO, out_dram,                    half_rot)
    ue.sram_to_accelerator_memory(SA_OUT_HI, out_dram + half_rot * BF16,  half_rot)

def _run_fpga_vision_encoder_legacy(ue, model_dir: str, hf_inputs) -> torch.Tensor:
    from transformers import AutoModelForImageTextToText
    print(f"  [vision-legacy] Loading HF visual ...", end=" ", flush=True)
    t_load = time.time()
    hf = AutoModelForImageTextToText.from_pretrained(
        model_dir, dtype=torch.bfloat16, local_files_only=True).eval()
    visual = hf.model.visual
    print(f"({time.time()-t_load:.1f}s)")

    pixel_values = hf_inputs["pixel_values"].to(torch.bfloat16)
    grid_thw     = hf_inputs["image_grid_thw"]
    N_P    = pixel_values.shape[0]
    HIDDEN, K_PIXEL = 1024, 1536
    N_HEADS, HEAD_D = 16, 64
    QKV_DIM, MLP_INT = 3 * HIDDEN, 4096
    DEPTH = len(visual.blocks)

    pe_w = visual.patch_embed.proj.weight.detach().view(HIDDEN, K_PIXEL).to(torch.bfloat16)
    pe_b = visual.patch_embed.proj.bias.detach().to(torch.bfloat16)
    with torch.no_grad():
        pos_embeds = visual.fast_pos_embed_interpolate(grid_thw).to(torch.bfloat16)
    with torch.no_grad():
        rot = visual.rot_pos_emb(grid_thw).reshape(N_P, -1)
        rot_emb = torch.cat((rot, rot), dim=-1)
        rot_cos = rot_emb.cos()[:, :HEAD_D].to(torch.float32)
        rot_sin = rot_emb.sin()[:, :HEAD_D].to(torch.float32)
    half_rot = HEAD_D // 2
    cos_pad     = torch.zeros(N_P, HEAD_D, dtype=torch.float32)
    neg_sin_pad = torch.zeros_like(cos_pad)
    sin_hi_pad  = torch.zeros_like(cos_pad)
    cos_pad[:,     :half_rot] = rot_cos[:, :half_rot]
    neg_sin_pad[:, :half_rot] = -rot_sin[:, :half_rot]
    sin_hi_pad[:,  :half_rot] = rot_sin[:, half_rot:]

    print(f"  [vision-legacy] Uploading weights (FP4_64) ...", end=" ", flush=True)
    t_w = time.time()
    PIXEL          = ue.allocate_tensor_dram(N_P * K_PIXEL * BF16)
    PATCH_OUT      = ue.allocate_tensor_dram(N_P * HIDDEN * BF16)
    PATCH_PLUS_POS = ue.allocate_tensor_dram(N_P * HIDDEN * BF16)
    LN1_OUT        = ue.allocate_tensor_dram(N_P * HIDDEN * BF16)
    QKV_OUT        = ue.allocate_tensor_dram(N_P * QKV_DIM * BF16)
    LN2_OUT        = ue.allocate_tensor_dram(N_P * HIDDEN * BF16)
    BLOCK_OUT      = ue.allocate_tensor_dram(N_P * HIDDEN * BF16)
    Q_ROT          = ue.allocate_tensor_dram(N_P * HIDDEN * BF16)
    K_ROT          = ue.allocate_tensor_dram(N_P * HIDDEN * BF16)
    head_bytes     = N_P * HEAD_D * BF16
    Q_PERM         = ue.allocate_tensor_dram(N_HEADS * head_bytes)
    K_PERM         = ue.allocate_tensor_dram(N_HEADS * head_bytes)
    V_PERM         = ue.allocate_tensor_dram(N_HEADS * head_bytes)
    ATTN_OUT       = ue.allocate_tensor_dram(N_HEADS * head_bytes)
    ATTN_CONCAT    = ue.allocate_tensor_dram(N_P * HIDDEN * BF16)
    ATTN_PROJ      = ue.allocate_tensor_dram(N_P * HIDDEN * BF16)
    POST_ATTN      = ue.allocate_tensor_dram(N_P * HIDDEN * BF16)
    FC1_GELU_OUT   = ue.allocate_tensor_dram(N_P * MLP_INT * BF16)
    FC2_OUT        = ue.allocate_tensor_dram(N_P * HIDDEN * BF16)
    ATTN_SCRATCH   = ue.allocate_tensor_dram((HEAD_D + N_P) * N_P * BF16 + N_P * HEAD_D * BF16)
    VIS_BIAS = ue.allocate_tensor_dram(N_P * N_P * BF16)
    VIS_EYE = ue.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * BF16)
    ue.dma_write(DMA_DEVICE_H2C, VIS_EYE,
                 _bf(torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16)),
                 UE_VECTOR_SIZE * UE_VECTOR_SIZE * BF16)
    ue.dma_write(DMA_DEVICE_H2C, VIS_BIAS,
                 _bf(torch.zeros(N_P, N_P, dtype=torch.bfloat16)),
                 N_P * N_P * BF16)
    N_M     = N_P // 4
    K_FC    = HIDDEN * 4
    FC2_DIM = 2048
    MG_LN_OUT  = ue.allocate_tensor_dram(N_P * HIDDEN * BF16)
    MG_FC1_OUT = ue.allocate_tensor_dram(N_M * K_FC * BF16)
    MG_OUT     = ue.allocate_tensor_dram(N_M * FC2_DIM * BF16)
    rot_bytes = N_P * HEAD_D * BF16
    COS_PAD     = ue.allocate_tensor_dram(rot_bytes)
    NEG_SIN_PAD = ue.allocate_tensor_dram(rot_bytes)
    SIN_HI_PAD  = ue.allocate_tensor_dram(rot_bytes)

    PATCH_W = _upload_bf16(ue, pe_w, HIDDEN * K_PIXEL)
    PATCH_B = _upload_bf16(ue, pe_b, HIDDEN)
    POS     = _upload_bf16(ue, pos_embeds, N_P * HIDDEN)
    layer_addrs = []
    for li in range(DEPTH):
        bi = visual.blocks[li]
        ln1_w = _upload_bf16(ue, bi.norm1.weight, HIDDEN)
        ln1_b = _upload_bf16(ue, bi.norm1.bias,   HIDDEN)
        qkv_b = _upload_bf16(ue, bi.attn.qkv.bias, QKV_DIM)
        proj_b = _upload_bf16(ue, bi.attn.proj.bias, HIDDEN)
        ln2_w = _upload_bf16(ue, bi.norm2.weight, HIDDEN)
        ln2_b = _upload_bf16(ue, bi.norm2.bias,   HIDDEN)
        fc1_b = _upload_bf16(ue, bi.mlp.linear_fc1.bias, MLP_INT)
        fc2_b = _upload_bf16(ue, bi.mlp.linear_fc2.bias, HIDDEN)
        qkv_s, qkv_d   = _upload_fp4(ue, bi.attn.qkv.weight.detach().float())
        proj_s, proj_d = _upload_fp4(ue, bi.attn.proj.weight.detach().float())
        fc1_s, fc1_d   = _upload_fp4(ue, bi.mlp.linear_fc1.weight.detach().float())
        fc2_s, fc2_d   = _upload_fp4(ue, bi.mlp.linear_fc2.weight.detach().float())
        layer_addrs.append({
            "LN1_W": ln1_w, "LN1_B": ln1_b, "LN2_W": ln2_w, "LN2_B": ln2_b,
            "QKV_B": qkv_b, "PROJ_B": proj_b, "FC1_B": fc1_b, "FC2_B": fc2_b,
            "QKV_W_DATA": qkv_d, "QKV_W_SCALE": qkv_s,
            "PROJ_W_DATA": proj_d, "PROJ_W_SCALE": proj_s,
            "FC1_W_DATA": fc1_d, "FC1_W_SCALE": fc1_s,
            "FC2_W_DATA": fc2_d, "FC2_W_SCALE": fc2_s,
        })
    mg = visual.merger
    MG_LN_W  = _upload_bf16(ue, mg.norm.weight, HIDDEN)
    MG_LN_B  = _upload_bf16(ue, mg.norm.bias,   HIDDEN)
    MG_FC1_B = _upload_bf16(ue, mg.linear_fc1.bias, K_FC)
    MG_FC2_B = _upload_bf16(ue, mg.linear_fc2.bias, FC2_DIM)
    MG_FC1_S, MG_FC1_D = _upload_fp4(ue, mg.linear_fc1.weight.detach().float())
    MG_FC2_S, MG_FC2_D = _upload_fp4(ue, mg.linear_fc2.weight.detach().float())

    ue.dma_write(DMA_DEVICE_H2C, PIXEL, _bf(pixel_values), N_P * K_PIXEL * BF16)
    ue.dma_write(DMA_DEVICE_H2C, COS_PAD,
                 _bf(cos_pad.to(torch.bfloat16).contiguous()),     rot_bytes)
    ue.dma_write(DMA_DEVICE_H2C, NEG_SIN_PAD,
                 _bf(neg_sin_pad.to(torch.bfloat16).contiguous()), rot_bytes)
    ue.dma_write(DMA_DEVICE_H2C, SIN_HI_PAD,
                 _bf(sin_hi_pad.to(torch.bfloat16).contiguous()),  rot_bytes)
    del hf
    print(f"({time.time()-t_w:.1f}s)")

    ue.start_capture()
    ue.matmat_mul_core(M=N_P, K=K_PIXEL, N=HIDDEN,
        A_DRAM_ADDR=PIXEL, B_DRAM_ADDR=PATCH_W, OUTPUT_DRAM_ADDR=PATCH_OUT,
        C_DRAM_ADDR=PATCH_B, bias_mode="broadcast_N", is_B_quantized=False)
    URAM_A_BASE, URAM_B_BASE = 0x00000, 0x80000
    chunk_elems = 65536
    elems_total = N_P * HIDDEN
    elems_done = 0
    while elems_done < elems_total:
        c = min(chunk_elems, elems_total - elems_done)
        off = elems_done * BF16
        ue.accelerator_memory_to_sram(PATCH_OUT + off, URAM_A_BASE, c)
        ue.accelerator_memory_to_sram(POS       + off, URAM_B_BASE, c)
        ue.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, c)
        ue.sram_to_accelerator_memory(URAM_A_BASE, PATCH_PLUS_POS + off, c)
        elems_done += c
    _vis_exec_capture(ue)

    print(f"  [vision-legacy] Running {DEPTH} layers ...", end=" ", flush=True)
    t_l = time.time()
    for li in range(DEPTH):
        in_dram = PATCH_PLUS_POS if li == 0 else BLOCK_OUT
        L = layer_addrs[li]
        ue.start_capture()
        ue.layer_norm_core_dram(M=N_P, N=HIDDEN,
            A_DRAM_ADDR=in_dram, OUTPUT_DRAM_ADDR=LN1_OUT,
            GAMMA_DRAM_ADDR=L["LN1_W"], BETA_DRAM_ADDR=L["LN1_B"])
        ue.matmat_mul_core(M=N_P, K=HIDDEN, N=QKV_DIM,
            A_DRAM_ADDR=LN1_OUT, B_DRAM_ADDR=L["QKV_W_DATA"], OUTPUT_DRAM_ADDR=QKV_OUT,
            C_DRAM_ADDR=L["QKV_B"], bias_mode="broadcast_N",
            is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=L["QKV_W_SCALE"])
        _vis_exec_capture(ue)
        ue.start_capture()
        for t in range(N_P):
            cos_a = COS_PAD     + t * HEAD_D * BF16
            nsin_a = NEG_SIN_PAD + t * HEAD_D * BF16
            sin_a  = SIN_HI_PAD  + t * HEAD_D * BF16
            for h in range(N_HEADS):
                _emit_rotary_per_row(ue, QKV_OUT + (t * QKV_DIM + h * HEAD_D) * BF16,
                    Q_ROT + (t * HIDDEN + h * HEAD_D) * BF16,
                    cos_a, nsin_a, sin_a, load_cos_sin=(h == 0))
        _vis_exec_capture(ue)
        ue.start_capture()
        for t in range(N_P):
            cos_a = COS_PAD     + t * HEAD_D * BF16
            nsin_a = NEG_SIN_PAD + t * HEAD_D * BF16
            sin_a  = SIN_HI_PAD  + t * HEAD_D * BF16
            for h in range(N_HEADS):
                _emit_rotary_per_row(ue, QKV_OUT + (t * QKV_DIM + HIDDEN + h * HEAD_D) * BF16,
                    K_ROT + (t * HIDDEN + h * HEAD_D) * BF16,
                    cos_a, nsin_a, sin_a, load_cos_sin=(h == 0))
        _vis_exec_capture(ue)
        ue.start_capture()
        for h in range(N_HEADS):
            ue.ue_memcpy_from_dram(Q_ROT + h * HEAD_D * BF16,
                N_P * HEAD_D * BF16, MEMCPY_TYPE.URAM.value, URAM_START_ADDR,
                URAM_SECTION.URAM_A.value,
                stride_bytes_per_chunk=HEAD_D * BF16, stride_jump_bytes=HIDDEN * BF16)
            ue.ue_memcpy_to_dram(MEMCPY_TYPE.URAM.value, URAM_SECTION.URAM_A.value,
                URAM_START_ADDR, Q_PERM + h * head_bytes, N_P * HEAD_D * BF16)
            ue.ue_memcpy_from_dram(K_ROT + h * HEAD_D * BF16,
                N_P * HEAD_D * BF16, MEMCPY_TYPE.URAM.value, URAM_START_ADDR,
                URAM_SECTION.URAM_A.value,
                stride_bytes_per_chunk=HEAD_D * BF16, stride_jump_bytes=HIDDEN * BF16)
            ue.ue_memcpy_to_dram(MEMCPY_TYPE.URAM.value, URAM_SECTION.URAM_A.value,
                URAM_START_ADDR, K_PERM + h * head_bytes, N_P * HEAD_D * BF16)
            ue.ue_memcpy_from_dram(QKV_OUT + (2 * HIDDEN + h * HEAD_D) * BF16,
                N_P * HEAD_D * BF16, MEMCPY_TYPE.URAM.value, URAM_START_ADDR,
                URAM_SECTION.URAM_A.value,
                stride_bytes_per_chunk=HEAD_D * BF16, stride_jump_bytes=QKV_DIM * BF16)
            ue.ue_memcpy_to_dram(MEMCPY_TYPE.URAM.value, URAM_SECTION.URAM_A.value,
                URAM_START_ADDR, V_PERM + h * head_bytes, N_P * HEAD_D * BF16)
        for h in range(N_HEADS):
            ue.flash_attention_core_cached(head_dim=HEAD_D, seq_len=N_P,
                Q_DRAM_ADDR=Q_PERM + h * head_bytes, K_DRAM_ADDR=K_PERM + h * head_bytes,
                V_DRAM_ADDR=V_PERM + h * head_bytes, OUTPUT_DRAM_ADDR=ATTN_OUT + h * head_bytes,
                SCRATCH_DRAM_ADDR=ATTN_SCRATCH, IDENTITY_DRAM_ADDR=VIS_EYE,
                BIAS_DRAM_ADDR=VIS_BIAS)
        for h in range(N_HEADS):
            ue.ue_memcpy_from_dram(ATTN_OUT + h * head_bytes,
                N_P * HEAD_D * BF16, MEMCPY_TYPE.URAM.value, URAM_START_ADDR,
                URAM_SECTION.URAM_A.value)
            ue.ue_memcpy_to_dram(MEMCPY_TYPE.URAM.value, URAM_SECTION.URAM_A.value,
                URAM_START_ADDR, ATTN_CONCAT + h * HEAD_D * BF16, N_P * HEAD_D * BF16,
                stride_bytes_per_chunk=HEAD_D * BF16, stride_jump_bytes=HIDDEN * BF16)
        ue.matmat_mul_core(M=N_P, K=HIDDEN, N=HIDDEN,
            A_DRAM_ADDR=ATTN_CONCAT, B_DRAM_ADDR=L["PROJ_W_DATA"], OUTPUT_DRAM_ADDR=ATTN_PROJ,
            C_DRAM_ADDR=L["PROJ_B"], bias_mode="broadcast_N",
            is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=L["PROJ_W_SCALE"])
        _vis_exec_capture(ue)
        ue.start_capture()
        elems_done = 0
        while elems_done < elems_total:
            c = min(chunk_elems, elems_total - elems_done)
            off = elems_done * BF16
            ue.accelerator_memory_to_sram(in_dram   + off, URAM_A_BASE, c)
            ue.accelerator_memory_to_sram(ATTN_PROJ + off, URAM_B_BASE, c)
            ue.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, c)
            ue.sram_to_accelerator_memory(URAM_A_BASE, POST_ATTN + off, c)
            elems_done += c
        ue.layer_norm_core_dram(M=N_P, N=HIDDEN,
            A_DRAM_ADDR=POST_ATTN, OUTPUT_DRAM_ADDR=LN2_OUT,
            GAMMA_DRAM_ADDR=L["LN2_W"], BETA_DRAM_ADDR=L["LN2_B"])
        ue.matmat_mul_core(M=N_P, K=HIDDEN, N=MLP_INT,
            A_DRAM_ADDR=LN2_OUT, B_DRAM_ADDR=L["FC1_W_DATA"], OUTPUT_DRAM_ADDR=FC1_GELU_OUT,
            C_DRAM_ADDR=L["FC1_B"], bias_mode="broadcast_N",
            is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=L["FC1_W_SCALE"],
            gelu_enable=True)
        _vis_exec_capture(ue)
        ue.start_capture()
        ue.matmat_mul_core(M=N_P, K=MLP_INT, N=HIDDEN,
            A_DRAM_ADDR=FC1_GELU_OUT, B_DRAM_ADDR=L["FC2_W_DATA"], OUTPUT_DRAM_ADDR=FC2_OUT,
            C_DRAM_ADDR=L["FC2_B"], bias_mode="broadcast_N",
            is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=L["FC2_W_SCALE"])
        elems_done = 0
        while elems_done < elems_total:
            c = min(chunk_elems, elems_total - elems_done)
            off = elems_done * BF16
            ue.accelerator_memory_to_sram(POST_ATTN + off, URAM_A_BASE, c)
            ue.accelerator_memory_to_sram(FC2_OUT   + off, URAM_B_BASE, c)
            ue.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, c)
            ue.sram_to_accelerator_memory(URAM_A_BASE, BLOCK_OUT + off, c)
            elems_done += c
        _vis_exec_capture(ue)
    print(f"({time.time()-t_l:.1f}s)")

    ue.start_capture()
    ue.layer_norm_core_dram(M=N_P, N=HIDDEN,
        A_DRAM_ADDR=BLOCK_OUT, OUTPUT_DRAM_ADDR=MG_LN_OUT,
        GAMMA_DRAM_ADDR=MG_LN_W, BETA_DRAM_ADDR=MG_LN_B)
    ue.matmat_mul_core(M=N_M, K=K_FC, N=K_FC,
        A_DRAM_ADDR=MG_LN_OUT, B_DRAM_ADDR=MG_FC1_D, OUTPUT_DRAM_ADDR=MG_FC1_OUT,
        C_DRAM_ADDR=MG_FC1_B, bias_mode="broadcast_N",
        is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=MG_FC1_S, gelu_enable=True)
    ue.matmat_mul_core(M=N_M, K=K_FC, N=FC2_DIM,
        A_DRAM_ADDR=MG_FC1_OUT, B_DRAM_ADDR=MG_FC2_D, OUTPUT_DRAM_ADDR=MG_OUT,
        C_DRAM_ADDR=MG_FC2_B, bias_mode="broadcast_N",
        is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=MG_FC2_S)
    _vis_exec_capture(ue)

    return _read_dram_bf16(ue, MG_OUT, (N_M, FC2_DIM))

def main():
    ap = argparse.ArgumentParser(description="Qwen3.5-2B LM inference/generation on FPGA")
    ap.add_argument("--prompt", type=str, default=None,
                    help="text prompt for the model to answer/continue. "
                         "Defaults: VLM mode (--vision-enable or --image) → "
                         "'Describe what you see in this image.'; LM-only → "
                         "'Tell me about the Eiffel Tower. What year was it built?'.")
    ap.add_argument("--max-new-tokens", type=int, default=128,
                    help="max tokens to generate (default: 128). "
                         "Pass 0 to run until EOT or the KV cache is full "
                         "(`max_context - prompt_len`).")
    ap.add_argument("--dev", type=str, default="xdma0",
                    help="DMA device name (default: xdma0).")
    # VLM opt-in (gemma4 pattern). Default mode is pure LM; vision activates
    # only when --image PATH or --vision-enable is given.  Vision encoder
    # runs on FPGA (Phase 4) by default; the host-side HF path (Phase 1) is
    # available via --no-fpga-vision.  See notes_qwen3.5_2b.md "Vision
    # Encoder Implementation Notes".
    ap.add_argument("--image", type=str, default=None,
                    help="Path to image file for VLM inference. Implies --vision-enable.")
    ap.add_argument("--vision-enable", action="store_true",
                    help="Run as VLM using the default sample image "
                         "(src/template/test_samples/yosemite.jpg). "
                         "Ignored if --image is also given.")
    ap.add_argument("--vision-on-hardware", action="store_true",
                    help="Run the vision encoder on FPGA. By default the vision "
                         "encoder runs through Hugging Face on the host. "
                         "Only meaningful with "
                         "--vision-enable / --image.")
    args = ap.parse_args()
    if args.vision_on_hardware and not (args.vision_enable or args.image):
        ap.error("--vision-on-hardware requires --vision-enable or --image")

    set_dma_device(args.dev)
    global DMA_DEVICE_H2C, DMA_DEVICE_C2H
    DMA_DEVICE_H2C = user_dma_core.DMA_DEVICE_H2C
    DMA_DEVICE_C2H = user_dma_core.DMA_DEVICE_C2H

    # Hard-coded inference settings (formerly CLI flags, now defaults).
    MAX_CONTEXT = 256
    TEMPERATURE = 0.0
    TOP_K       = 0

    # VLM mode resolution (gemma4 pattern).  Default sample image lives in
    # the shared test_samples directory at the template level.
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _TEST_SAMPLES = os.path.normpath(os.path.join(_HERE, "..", "..", "test_samples"))
    DEFAULT_IMAGE = os.path.join(_TEST_SAMPLES, "yosemite.jpg")
    vision_on = bool(args.image) or args.vision_enable
    image_path = args.image or (DEFAULT_IMAGE if args.vision_enable else None)
    if vision_on and image_path and not os.path.exists(image_path):
        raise SystemExit(f"Image file not found: {image_path}")

    # Default prompt depends on mode: LM-only gets a directed factual prompt
    # that resists greedy-decode looping; VLM gets an image-grounded
    # instruction matched to the sample image.
    if args.prompt is None:
        args.prompt = ("Describe what you see in this image."
                       if vision_on else
                       "Tell me about the Eiffel Tower. What year was it built?")

    print("=" * 78)
    if vision_on:
        if args.vision_on_hardware:
            print("  Qwen3.5-2B VLM inference on FPGA (vision + LM, full FPGA)")
        else:
            print("  Qwen3.5-2B VLM inference on FPGA (host-side vision, Phase 1)")
    else:
        print("  Qwen3.5-2B LM inference on FPGA")
    print("=" * 78)
    if vision_on:
        print(f"  [Mode] VLM — image: {image_path}")

    script_dir = str(_THIS.parent)
    cfg_path = _THIS.parent / "qwen3.5_2b_config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    bin_dir = os.path.join(script_dir, "qwen3.5_2b_bin")
    os.makedirs(bin_dir, exist_ok=True)

    # Resolve HF model dir (symlink from /srv fallback if needed, else download).
    model_dir = _ensure_hf_model(script_dir, cfg)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    # VLM mode also needs the multimodal processor (image preprocessing +
    # chat-template image-pad expansion).  Loaded lazily so LM-only runs
    # don't pull in image-processing dependencies.
    processor = None
    image = None
    if vision_on:
        from transformers import AutoProcessor
        from PIL import Image
        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        image = Image.open(image_path).convert("RGB")

    # Load (or generate) the cached quantized-weights bin.  First run does HF
    # load + quantize + torch.save (~25 s total); subsequent runs do only
    # torch.load (~2 s).
    weights_bin = os.path.join(script_dir, cfg["paths"]["weights_bin"])
    weights_meta = os.path.join(
        script_dir, cfg["paths"].get("weights_meta", "qwen3.5_2b_bin/params.json"))
    if os.path.exists(weights_bin):
        print(f"  Loading weights from {weights_bin} ...", flush=True)
        t0 = time.time()
        weights = torch.load(weights_bin, weights_only=False,
                             map_location="cpu")
        print(f"  Weights loaded ({time.time()-t0:.1f}s).")
        if not os.path.exists(weights_meta):
            with open(weights_meta, "w") as _wm:
                json.dump({
                    "format": "torch.save",
                    "size": os.path.getsize(weights_bin),
                    "hf_model_repo": cfg["paths"].get("hf_model_repo"),
                    "num_layers": cfg["file_info"]["num_layers"],
                }, _wm)
    else:
        print(f"  Weight bin not found at {weights_bin}")
        print(f"  Loading HF model from {model_dir} ...", flush=True)
        with _quiet():
            hf, text_model = _load_hf_model(model_dir)
        print(f"  Loaded: {_model_info(text_model)}")
        print(f"  Extracting + quantizing weights ...", flush=True)
        t0 = time.time()
        linear_attn_layers = set(cfg["model"]["linear_attn_layer_indices"])
        weights = _extract_all_weights(text_model, linear_attn_layers)
        print(f"  Extraction done ({time.time()-t0:.1f}s). Saving to bin ...",
              flush=True)
        t0 = time.time()
        torch.save(weights, weights_bin)
        sz_mb = os.path.getsize(weights_bin) / (1024 * 1024)
        with open(weights_meta, "w") as _wm:
            json.dump({
                "format": "torch.save",
                "size": os.path.getsize(weights_bin),
                "hf_model_repo": cfg["paths"].get("hf_model_repo"),
                "num_layers": cfg["file_info"]["num_layers"],
            }, _wm)
        print(f"  Wrote {weights_bin} ({sz_mb:.0f} MB, {time.time()-t0:.1f}s).")
        # Free the HF model — all the tensors we need are in `weights` now.
        del hf, text_model

    print("  FPGA init ...", end=" ", flush=True)
    with _quiet():
        ue = Qwen3_5_2b_UnifiedEngine(device='cpu')
        ue.software_reset()
    print("ok.")

    # VLM-on-hardware: the vision encoder runs FIRST — BEFORE the LM is prepared.
    # Ordering is critical: `prepare_inference` must be called EXACTLY ONCE per
    # engine and AFTER all vision work.  A second prepare, OR executing the vision
    # encoder after prepare, silently corrupts the LM decode (→ all-`!`) and is
    # NOT recoverable by software_reset / clear_dram.  The encoder uses
    # params/program at base 0xD0000000; we reset the bump pointers afterward so
    # the LM gets that same base.
    #
    # THE SINGLE comprehensive programs bin (programs.bin + programs.json manifest)
    # holds [encoder?][decoder] — vision encoder (if built via a --vision-on-hardware
    # run) + LM decoder, both baked for program base 0xD0000000.  There is no
    # separate persisted decoder bin: the transient compile output is folded into
    # programs.bin and removed.  FIRST run BUILDS + writes the bin; EVERY later run
    # (test.py or run_from_bin) LOADS it — no recompilation.
    uni_bin       = os.path.join(bin_dir, UNIFIED_BIN_NAME)
    uni_meta_path = os.path.join(bin_dir, UNIFIED_META_NAME)
    _raw = _meta = None
    if os.path.exists(uni_bin) and os.path.exists(uni_meta_path):
        with open(uni_bin, "rb") as f:
            _raw = f.read()
        with open(uni_meta_path) as f:
            _meta = json.load(f)
    want_vision = bool(vision_on and args.vision_on_hardware)
    # Build when there is no bin yet, or when an on-FPGA VLM run finds a cached
    # bin that has no encoder section (e.g. an earlier LM-only build).
    need_build = (_meta is None) or (want_vision and _meta.get("encoder_size", 0) == 0)

    precomputed_vision_tokens = None
    vis_encoder_bytes = None
    vis_meta = None
    if need_build:
        # ALWAYS build the vision encoder into the comprehensive bin — even on an
        # LM-only run — so the single bin holds the full model.  Use the run's
        # image if given, else the bundled sample (the encoder bakes that image's
        # fixed patch count, N_P=576).  EXECUTE it only when this run actually
        # needs vision tokens (on-FPGA VLM); otherwise just capture its bytes
        # (build_only) — loading the HF visual model + uploading vision weights is
        # a one-time first-build cost.
        from transformers import AutoProcessor as _AutoProc
        from PIL import Image as _Image
        _proc = processor if processor is not None else \
            _AutoProc.from_pretrained(model_dir, local_files_only=True)
        _img  = image if image is not None else _Image.open(DEFAULT_IMAGE).convert("RGB")
        _msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": args.prompt}]}]
        _txt  = _proc.apply_chat_template(_msgs, tokenize=False, add_generation_prompt=True)
        hf_inputs = _proc(text=[_txt], images=[_img], return_tensors="pt")
        print(f"  Building vision encoder (first run → comprehensive bin) ...", flush=True)
        t_v = time.time()
        with _quiet():
            _tok = run_fpga_vision_encoder(ue, model_dir, hf_inputs, build_only=not want_vision)
            if want_vision:
                precomputed_vision_tokens = _tok
            vis_encoder_bytes = ue._vis_encoder_bytes
            vis_meta = {"n_patches": ue._vis_n_patches, "mg_out": ue._vis_encoder_out_dram,
                        "n_merged": ue._vis_n_merged, "fc2_dim": ue._vis_fc2_dim}
            ue.reset_params_dram_addr()
            ue.reset_tensor_dram_addr()
            ue.reset_program_dram_addr()
        if want_vision:
            print(f"  vision encoder built + run ({time.time()-t_v:.1f}s, "
                  f"{tuple(precomputed_vision_tokens.shape)} merged tokens).")
        else:
            print(f"  vision encoder built into bin ({time.time()-t_v:.1f}s, bytes only).")
    elif want_vision:
        # LOAD path, on-FPGA VLM: replay the encoder section from the bin → tokens.
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": args.prompt}]}]
        text_for_proc = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        hf_inputs = processor(text=[text_for_proc], images=[image], return_tensors="pt")
        print(f"  Loading vision encoder from unified bin ...", flush=True)
        t_v = time.time()
        with _quiet():
            precomputed_vision_tokens = run_vision_from_bin(ue, _raw, _meta, model_dir, hf_inputs)
            ue.reset_params_dram_addr()
            ue.reset_tensor_dram_addr()
            ue.reset_program_dram_addr()
        print(f"  vision (from bin) done ({time.time()-t_v:.1f}s, "
              f"{tuple(precomputed_vision_tokens.shape)} merged tokens).")

    # SINGLE prepare_inference — after any vision work, never twice.
    with _quiet():
        ue._preallocate_identity_matrix()
        ue.prepare_inference(weights, max_context=MAX_CONTEXT)

    # LM decoder: BUILD (compile → write the unified bin → drop the transient
    # transient .decoder_compile.bin) on the first run, else LOAD the decoder section from the
    # unified bin.  Either way the decoder ends up resident in DRAM, so generate()
    # is told it is preloaded and skips compilation.
    if need_build:
        with _quiet():
            _bp, _, _flops = ue.compile_decoder()
            ue.load_instructions(_bp)
        write_unified_bin(ue, bin_dir, vis_encoder_bytes or b"", vis_meta or {},
                          decoder_bin_path=_bp, flops=_flops)
        for _p in (ue._decoder_bin_path, ue._decoder_meta_path):
            try:
                os.remove(_p)
            except OSError:
                pass
    else:
        with _quiet():
            load_decoder_from_bin(ue, _raw, _meta)

    generated = generate(ue, tokenizer,
                         prompt=args.prompt,
                         max_new_tokens=args.max_new_tokens,
                         temperature=TEMPERATURE,
                         top_k=TOP_K,
                         processor=processor,
                         image=image,
                         model_dir=model_dir,
                         precomputed_vision_tokens=precomputed_vision_tokens,
                         decoder_preloaded=True)
    # Generated text is already streamed inline inside run_decoder() and the
    # decode-summary block closes the run — no need to repeat the prompt /
    # generation / full text here.  Silence main()-side epilogue to avoid
    # dumping the whole generation twice.
    del generated


def _model_info(text_model) -> str:
    n_layers = len(text_model.layers)
    n_linear = sum(1 for l in text_model.layers if hasattr(l, "linear_attn"))
    n_full   = n_layers - n_linear
    return f"{n_layers} layers ({n_linear} linear-attn, {n_full} full-attn)"



def _theoretical_flops_per_step(ue: Qwen3_5_2b_UnifiedEngine, T: int) -> int:
    """Analytical FLOP count for one forward pass of `T` tokens on Qwen3.5-2B
    LM.  FP4 / BF16 matmuls count 2 FLOPs per MAC (1 mul + 1 add); rms_norm
    counts ~4·N FLOPs per row (sum-of-squares + rsqrt + normalize + gamma).
    Flash attention uses T_aligned = ue.max_context_aligned for key/value
    length (we always run it over the full cache region; early positions are
    bias-masked out).  Useful for reporting GFLOP/s; not an exact instruction
    count but matches the dominant work terms."""
    H            = ue.hidden_size
    conv_dim     = ue.lin_conv_dim
    value_dim    = ue.lin_value_dim
    key_dim      = ue.lin_key_dim
    num_vh       = ue.lin_num_v_heads
    Dk           = ue.lin_head_k_dim
    Dv           = ue.lin_head_v_dim
    mlp_dim      = ue.mlp_dim
    full_head_d  = ue.full_head_dim
    num_q        = ue.full_num_heads
    num_kv       = ue.full_num_kv_heads
    rotary_dim   = ue.full_rotary_dim
    q_proj_out   = 2 * num_q * full_head_d
    kv_size      = num_kv * full_head_d
    q_size_flat  = num_q * full_head_d
    T_aligned    = ue.max_context_aligned

    def rms(T_, N): return 4 * T_ * N
    def mm(T_, K, N): return 2 * T_ * K * N

    # ---- linear-attn layer (Gated DeltaNet) ----
    flops_lin = 0
    flops_lin += rms(T, H)                                   # input_layernorm
    flops_lin += mm(T, H, conv_dim)                          # in_proj_qkv (FP4)
    flops_lin += 2 * T * 4 * conv_dim                        # causal conv1d 4-tap
    flops_lin += mm(T, conv_dim, conv_dim)                   # silu-via-eye matmul (big!)
    flops_lin += rms(T * num_vh, Dk) * 2                     # L2-norm Q and K
    flops_lin += mm(T, H, UE_VECTOR_SIZE) * 2                # β and A projections
    flops_lin += mm(T, H, value_dim)                         # in_proj_z (silu-fused)
    # recurrence: per (t, h) → α·S + 2 matvecs + outer-product + αS+outer
    rec_per_head = 4 * Dk * Dv + 4 * Dk * Dv                 # ≈ 8·Dk·Dv (matvec + outer + sum)
    flops_lin += T * num_vh * rec_per_head
    flops_lin += rms(T * num_vh, Dv)                         # normed rms (RMSNormGated)
    flops_lin += T * value_dim                               # gated eltwise_mul
    flops_lin += mm(T, value_dim, H)                         # out_proj (FP4)
    flops_lin += T * H                                       # residual add
    flops_lin += rms(T, H)                                   # post_attention_layernorm
    flops_lin += mm(T, H, mlp_dim) * 2                       # MLP gate + up (FP4, silu fused on gate)
    flops_lin += T * mlp_dim                                 # gate ⊙ up
    flops_lin += mm(T, mlp_dim, H)                           # MLP down (FP4)
    flops_lin += T * H                                       # residual add

    # ---- full-attn layer ----
    flops_full = 0
    flops_full += rms(T, H)
    flops_full += mm(T, H, q_proj_out)                       # Q proj (FP4, gated double-wide)
    flops_full += mm(T, H, kv_size) * 2                      # K, V projs
    flops_full += rms(T * num_q, full_head_d)                # q_norm
    flops_full += rms(T * num_kv, full_head_d)               # k_norm
    # flash attention per head: Q·Kᵀ (2·T_aligned²·head_dim) + softmax (small) + attn·V
    flops_full += num_q * (4 * T_aligned * T_aligned * full_head_d + 5 * T_aligned * T_aligned)
    flops_full += T * q_size_flat                            # sigmoid(gate) × attn
    flops_full += mm(T, q_size_flat, H)                      # O proj (FP4)
    flops_full += T * H                                      # residual add
    flops_full += rms(T, H)
    flops_full += mm(T, H, mlp_dim) * 2                      # MLP gate + up
    flops_full += T * mlp_dim
    flops_full += mm(T, mlp_dim, H)                          # MLP down
    flops_full += T * H

    # ---- final norm + LM head ----
    vocab = ue._cfg["special"]["embedding"]["vocab_size"]
    flops_tail = rms(T, H) + mm(T, H, vocab)

    n_lin = len(ue.linear_attn_layers)
    n_full = len(ue.full_attn_layers)
    return n_lin * flops_lin + n_full * flops_full + flops_tail


def _tokenize_with_chat_template(tokenizer, prompt: str,
                                 processor=None, image=None) -> torch.Tensor:
    """Wrap `prompt` in the model's chat template (user role + generation
    prompt) and tokenize, matching the conversational-inference pattern the
    model was instruction-tuned with.

    Two modes:
      * Text-only (processor/image=None): same as before — tokenizer + chat
        template. Unchanged from the LM-only path.
      * VLM (processor + PIL.Image given): use HF processor so it correctly
        expands `<|image_pad|>` into the right number of image tokens for
        the given image's grid_thw. Returns the same flat input_ids tensor
        as the text path; the VLM-aware caller separately handles the
        pixel_values / grid_thw needed by the host-side vision encoder.
    """
    if processor is not None and image is not None:
        conversation = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        return inputs.input_ids[0], inputs
    conversation = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(conversation, tokenize=False,
                                         add_generation_prompt=True)
    ids = tokenizer.encode(text, add_special_tokens=True)
    return torch.tensor(ids, dtype=torch.long), None


def _run_hf_vision_host(model_dir: str, hf_inputs) -> torch.Tensor:
    """Run the HF vision encoder + merger host-side and return the merged
    `[N_image_tokens, lm_hidden=2048]` bf16 tensor.

    Phase 1 of the FPGA-VLM port: the LM runs on FPGA but the vision encoder
    runs on host (HF AutoModelForImageTextToText.visual).  Subsequent phases
    move patch embed → ViT layers → spatial merge / connector onto FPGA.

    `hf_inputs` is the BatchEncoding returned by `processor(text=..., images=...)`
    and must contain `pixel_values` and `image_grid_thw`.
    """
    from transformers import AutoModelForImageTextToText
    hf = AutoModelForImageTextToText.from_pretrained(
        model_dir, dtype=torch.bfloat16, local_files_only=True)
    hf.eval()
    with torch.no_grad():
        out = hf.model.visual(hf_inputs["pixel_values"].to(torch.bfloat16),
                              grid_thw=hf_inputs["image_grid_thw"])
    # `out.last_hidden_state` is the pre-merger ViT output [N_patches, 1024];
    # `out.pooler_output` is the post-merger output [N_merged, 2048] which
    # matches the LM's hidden_size and is what gets substituted at <|image_pad|>
    # token positions during prefill.
    merged = out.pooler_output.detach().cpu()
    del hf
    return merged


if __name__ == "__main__":
    main()
