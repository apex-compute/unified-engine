#!/usr/bin/env python3
"""
Moonshine-base ASR on the UnifiedEngine FPGA accelerator.

This is the ACCELERATOR path (every <model>_test.py in the repo is). The CPU/CUDA
reference lives in moonshine_cpu_reference.py and is imported here only for the
weights, tokenizer, audio and the per-stage SNR oracle.

STATUS: written against the nn_lib / user_dma_core API, NOT yet run on hardware.

Layout conventions (all bf16, all row-major, every innermost dim a multiple of 64):
  hidden      416 -> 448. Pad lanes of every residual-stream tensor hold the ROW MEAN,
              not zero, so the hardware LayerNorm over 448 lanes equals the HF LayerNorm
              over 416 lanes (sigma_448 = sqrt(416/448)*sigma_416, folded into gamma).
              The invariant is maintained for free by left-multiplying every projection
              that writes the residual stream (o_proj, fc2, embedding) by W_mean.
  head_dim    52 -> 64 per head. q/k lanes are PERMUTED so Moonshine's interleaved
              partial RoPE (pairs (2i,2i+1), i<16, lanes 32..51 pass-through) becomes
              the half-split rotation rope_hf_core_dram implements: new lane i <- 2i,
              new lane 32+i <- 2i+1 (i<16), pass-through lanes at 16..31 and 48..51,
              lanes 52..63 zero. cos=1/sin=0 on non-rotary lanes. Attention is invariant
              to a consistent lane permutation of q and k; v/o are not permuted.
  scaling     the flash cores hard-code head_dim**-0.5 = 1/8; HF uses 52**-0.5, so
              q_proj weights carry sqrt(64/52).
  attention   per-head blocks [T, 64] (prefill) / caches [MAX_STEPS, 64] (decode),
              one flash call per head (num_q_heads=1). Padded encoder frames are masked
              with a -inf bias row.
  conv stem   im2col is done ON DEVICE with strided DRAM->URAM reads (chunk = one
              receptive-field row, jump = stride*row_pitch), then plain matmuls.
              GroupNorm(1 group) is done on device: flat sum / sum-of-squares via matmul
              reductions against a ones matrix, rsqrt via Newton iterations on a
              64-lane vector, then two 448x448 matmuls apply (x-mu)*rstd*gamma + beta.
  GELU        encoder/conv GELU uses the fused LALU gelu (x*sigmoid(1.702x)); HF uses
              erf-GELU. Expect a few dB on the encoder; measure before replacing.
  decode      ONE body program for all 194 steps (llama pattern). Runtime registers:
              GPR_ALIGNED_SEQ_LEN_REG = ceil64(step+1) drives unified_attention_core's dynamic
              K/V length over the [256, 64] per-head caches; V_CACHE_SIZE_REG = step*128B (word
              addr) is the cache-row offset for the K/V scatter (add_imm + general_reg_src DMA).
              A 194-entry dispatch table (add_set, add_set, jump, nop) primes both and jumps into
              the body; the host launches entry `step`. Per-step host data: embedding row (896 B),
              the three RoPE table rows for position `step`, one causal bias row [1, 256].
              With --engines N the step is sharded in master/worker rounds: self- and
              cross-attention head-split, fc1x/fc1g N-split straight into the shared M=1 row,
              LM head N-split with per-engine argmax candidates compared on the host.
              BANDWIDTH: an M=1 step is weight-streaming bound, and on the U50 more than two
              engines reading one 1 GiB HBM switch region halve each other (multi_engine_shard
              notes). So with --engines N the model map moves to the second HBM stack
              (model_multicore_layout) and every decode projection's column shard is
              MATERIALIZED into the owning engine's controller-private window
              (shard_bf16_weight, the gemma3 scheme): q/k/v/attention per head on the
              owning engine, o_proj/cross o_proj/fc1x/fc1g/fc2/LM head N-split from private
              shards. The primary is left with the LayerNorms and the rendezvous.
  multi-engine (--engines N, default 1 = exact single-engine emission):
              stem + 8 encoder layers + cross K/V are ONE primary program plus one program
              per worker (multi_engine_shard.MultiEngineScheduler). conv1/tanh/GroupNorm
              partial sums: row-split in 64-row blocks with per-engine reduction arenas;
              stats/rsqrt/D_gamma on the primary; GN apply, conv2, conv3, W_mean+LN,
              encoder LN/fc1/fc2 and cross K/V: row-split; encoder attention: head-split
              (q/k/v/RoPE/flash/o_proj partial per head on the owning engine, then
              reduce_add). Decode stays single-engine (M=1).
  shapes      everything is compiled ONCE at the max_audio_seconds cap geometry. Per-run
              data = zero-padded PCM, the frame masks (-inf bias for frames >= real T3) and
              the GroupNorm 1/count vector, so one frozen image serves every clip <= cap.
              Real output frames depend only on real input frames (convs are local), so
              this is exact w.r.t. HF on the unpadded clip.
  bins        after a compile run, params.bin / programs.bin / programs.json are dumped to
              moonshine_bin/ (mobilesam pattern) with a signature (code sha1, weights sha1,
              cap geometry, DRAM bases). The NEXT run finds a matching image and replays it:
              no weight upload, no compilation. A signature mismatch (code/config/weights/cap
              changed) silently falls back to a fresh compile + re-dump; --clean forces it.

Usage (repo root):
  python models/moonshine/moonshine_test.py --dev xdma0 [--audio x.wav] [--no-snr] [--clean] [--verbose]
"""
import argparse
import builtins
import hashlib
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))  # repo root, for user_dma_core
sys.path.insert(0, SCRIPT_DIR)

# Silence the cores' per-op tiling prints (same builtins.print swap every model in the repo uses).
# This module's own output goes through `report`, which is never suppressed.
_original_print = builtins.print
_SILENT_MODE = [True]


def quiet_print(*args, **kwargs):
    if not _SILENT_MODE[0]:
        _original_print(*args, **kwargs)


builtins.print = quiet_print
report = _original_print

import user_dma_core
from user_dma_core import (
    UE_VECTOR_SIZE, DRAM_INSTRUCTION_ADDR, UnifiedEngine, ue_35bit_addr_shifter,
)
from multi_engine_shard import MultiEngineScheduler, PrivateArena, model_multicore_layout
from nn_lib import (
    URAM_A_BASE, URAM_B_BASE, URAM_NEAR_FULL_ELEMENTS,
    prefill_flash_attention_core, decode_flash_attention_core,
    layer_norm_core_dram_post_add, rope_hf_core_dram, tanh_core_dram,
    eltwise_add_core_dram, eltwise_mul_core_dram,
    store_weight, store_identity_matrix,
)

from moonshine_cpu_reference import (  # host-side plumbing + oracle; no hardware in there
    CFG, BIN_DIR, SAMPLE_RATE, DEFAULT_AUDIO,
    MoonshineReference, load_state_dict, ensure_weights, load_tokenizer, load_audio, snr_db,
)

ENC = CFG["encoder"]
DEC = CFG["decoder"]
STEM = CFG["model"]["conv_stem"]
BPE = 2                       # bytes per bf16
V = UE_VECTOR_SIZE            # 64
RED_K = 4096                  # K of the reduction ones-matrix (64 x 4096)
NEWTON_ITERS = 32
ARTIFACT_VERSION = 1
BIN_FILES = ("params.bin", "params.json", "programs.bin", "programs.json", "workers.bin", "shards.bin")


def _ceil(a, b):
    return (a + b - 1) // b * b


def _parse_addr(s):
    return int(s, 16) if isinstance(s, str) else int(s)


def init_hang_prevention(ue):
    """5-step hardware clear. MUST stay identical to moonshine_run_from_bin.py (skill gotchas)."""
    halt = bytes(64)
    ue.dma_write(ue.h2c_device, DRAM_INSTRUCTION_ADDR, halt, len(halt))
    report("[Init] HALT written to instruction DRAM base")


# ---------------------------------------------------------------------------
# host-side weight preparation (padding, permutation, mean-fold)
# ---------------------------------------------------------------------------

class WeightPrep:
    """Builds every padded/permuted bf16 tensor the device needs from the HF state dict."""

    def __init__(self, sd):
        self.sd = {k: v.float() for k, v in sd.items()}
        self.H = ENC["num_heads"]
        self.D = ENC["head_dim"]                 # 52
        self.DP = ENC["head_dim_padded"]         # 64
        self.C = ENC["hidden_size"]              # 416
        self.CP = ENC["hidden_size_padded"]      # 448
        self.I = ENC["intermediate_size"]        # 1664
        self.RD = ENC["rotary_dim"]              # 32
        self.q_scale = math.sqrt(self.DP / self.D)
        self.ln_scale = math.sqrt(self.C / self.CP)
        # W_mean: identity on real lanes, pad rows = mean of the real lanes
        wm = torch.zeros(self.CP, self.CP)
        wm[: self.C, : self.C] = torch.eye(self.C)
        wm[self.C:, : self.C] = 1.0 / self.C
        self.w_mean = wm
        # q/k lane permutation: new_lane -> orig_lane (-1 = zero pad)
        perm = [-1] * self.DP
        half = self.RD // 2                      # 16 rotary pairs
        for i in range(half):
            perm[i] = 2 * i
            perm[32 + i] = 2 * i + 1
        n_pass = self.D - self.RD                # 20 pass-through lanes
        pass_lanes = list(range(self.RD, self.D))
        for j, lane in enumerate(pass_lanes[:16]):
            perm[16 + j] = lane
        for j, lane in enumerate(pass_lanes[16:]):
            perm[48 + j] = lane
        assert sorted(l for l in perm if l >= 0) == list(range(self.D))
        self.perm = perm

    def p(self, name):
        return self.sd[name]

    # -- generic pads -------------------------------------------------------
    def pad_rows_cols(self, w, rows, cols):
        out = torch.zeros(rows, cols)
        out[: w.shape[0], : w.shape[1]] = w
        return out

    def pad_vec(self, b, n):
        out = torch.zeros(n)
        out[: b.shape[0]] = b
        return out

    def mean_fold_rows(self, w):
        """w: [416, K] projection writing the residual stream -> [448, K] with pad rows = mean."""
        return self.w_mean @ self.pad_rows_cols(w, self.CP, w.shape[1])[:, :]

    def mean_fold_bias(self, b):
        return self.w_mean @ self.pad_vec(b, self.CP)

    def ln_gamma(self, name):
        return self.pad_vec(self.p(name) * self.ln_scale, self.CP)

    # -- attention ------------------------------------------------------------
    def head_proj(self, w, h, permute, scale=1.0):
        """w: [8*52, 416] q/k/v projection -> head h as [64, 448] (B operand, N x K)."""
        wh = w[h * self.D:(h + 1) * self.D, :] * scale        # [52, 416]
        out = torch.zeros(self.DP, self.CP)
        if permute:
            for new, orig in enumerate(self.perm):
                if orig >= 0:
                    out[new, : self.C] = wh[orig]
        else:
            out[: self.D, : self.C] = wh
        return out

    def o_proj_full(self, w):
        """w: [416, 8*52] -> [448, 512] mean-folded, head blocks padded 52->64 on K."""
        out = torch.zeros(self.C, self.H * self.DP)
        for h in range(self.H):
            out[:, h * self.DP: h * self.DP + self.D] = w[:, h * self.D:(h + 1) * self.D]
        return self.mean_fold_rows(out)

    def o_proj_heads(self, w):
        """Per-head [448, 64] column blocks of o_proj_full (for the prefill accumulate)."""
        full = self.o_proj_full(w)
        return [full[:, h * self.DP:(h + 1) * self.DP].contiguous() for h in range(self.H)]

    # -- conv stem --------------------------------------------------------------
    def conv1(self):
        w = self.p("model.encoder.conv1.weight")              # [416, 1, 127]
        out = torch.zeros(self.CP, 128)
        out[: self.C, :127] = w[:, 0, :]
        return out

    def conv2(self):
        w = self.p("model.encoder.conv2.weight")              # [832, 416, 7]
        out = torch.zeros(2 * self.C, 7 * self.CP)
        for tap in range(7):
            out[:, tap * self.CP: tap * self.CP + self.C] = w[:, :, tap]
        return out

    def conv3(self):
        w = self.p("model.encoder.conv3.weight")              # [416, 832, 3]
        out = torch.zeros(self.CP, 3 * 2 * self.C)
        for tap in range(3):
            out[: self.C, tap * 2 * self.C:(tap + 1) * 2 * self.C] = w[:, :, tap]
        return out

    # -- embeddings ------------------------------------------------------------
    def embed_in(self):
        e = self.p("model.decoder.embed_tokens.weight")         # [32768, 416]
        out = self.pad_rows_cols(e, e.shape[0], self.CP)
        out[:, self.C:] = e.mean(dim=1, keepdim=True)
        return out

    def embed_out(self):
        e = self.sd.get("proj_out.weight", self.p("model.decoder.embed_tokens.weight"))
        return self.pad_rows_cols(e, e.shape[0], self.CP)


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------

class MoonshineEngine(UnifiedEngine):
    H        = ENC["num_heads"]
    DP       = ENC["head_dim_padded"]
    CP       = ENC["hidden_size_padded"]
    C        = ENC["hidden_size"]
    I        = ENC["intermediate_size"]
    N_ENC    = ENC["num_layers"]
    N_DEC    = DEC["num_layers"]
    VOCAB    = DEC["vocab_size"]
    MAX_STEPS = DEC["max_length"]
    MAX_STEPS_PAD = _ceil(DEC["max_length"], V)      # decode K/V window: kv_len must be a 64-multiple
    REGS = CFG["fixed_isa_regs"]
    R_VCACHE  = REGS["V_CACHE_SIZE_REG"]             # runtime cache-row word offset (step * 128 B >> 3)
    R_TMP     = REGS["TMP_REG"]
    R_ALIGNED = REGS["GPR_ALIGNED_SEQ_LEN_REG"]      # runtime aligned kv length for unified_attention_core
    FIRST_FREE_REG = max(REGS.values()) + 1
    THETA    = ENC["rope_theta"]
    RD       = ENC["rotary_dim"]
    GN_EPS   = STEM["groupnorm"]["eps"]
    BOS      = DEC["decoder_start_token_id"]
    EOS      = DEC["eos_token_id"]
    PRIMARY_PROGRAM_LIMIT = 0xD8000000   # single-engine map; shifted with the model map when sharded

    def __init__(self, ref: MoonshineReference = None, dev=None, max_seconds=None, from_bin=False, engines=1):
        """ref=None + from_bin=True: load params/programs from moonshine_bin/ (no HF weights,
        no compile). Otherwise upload weights from `ref` and compile everything at the cap."""
        lay = CFG["dram_layout"]
        bases = [_parse_addr(lay["params_dram_base"]), _parse_addr(lay["tensor_dram_base"]),
                 _parse_addr(lay["program_dram_base"]), self.PRIMARY_PROGRAM_LIMIT]
        self.windows = None
        if engines > 1:
            # Board map: private 512 MiB windows per engine (consecutive from 0 on interleaved
            # DRAM), and the model's own 2 GiB map placed disjoint from every window.
            if dev:
                user_dma_core.set_dma_device(dev)
            windows, model_base = model_multicore_layout(engines, model_bytes=2 << 30)
            shift = model_base - bases[0]
            bases = [b + shift for b in bases]
            self.windows = windows
        self.PRIMARY_PROGRAM_LIMIT = bases[3]
        kw = dict(params_dram_base=bases[0], tensor_dram_base=bases[1], program_dram_base=bases[2])
        if dev:
            kw["device_name"] = dev
        super().__init__(**kw)
        init_hang_prevention(self)
        self.ref = ref
        self.max_seconds = max_seconds or CFG["defaults"]["max_audio_seconds"]
        self._set_caps(int(self.max_seconds * SAMPLE_RATE))
        self.w = {}
        self.buf = {}
        self.prog = {}          # name -> (addr, size)
        self.prog_order = []
        self.NE = engines
        self.worker_progs, self.dec_worker_progs = [], []
        if engines > 1:
            # HW_INFO is the authority on how many cores the loaded bitstream has; a worker that does
            # not exist never raises its flag and the primary spins forever at the first barrier.
            user_dma_core.configure_clock_from_hardware()
            cores = user_dma_core.ANDROMEDA_CORE_COUNT
            report(user_dma_core.hardware_info_summary())
            assert cores is not None and engines <= cores, \
                f"--engines {engines} but the bitstream reports {cores} core(s)"
        # Scheduler FIRST: constructing worker engines stomps 16 KB at DRAM_START (guarded inside,
        # but nothing of ours is uploaded yet either way). num_engines=1 is an exact passthrough.
        self.arena = PrivateArena(engines, windows=self.windows) if engines > 1 else None
        self.sched = MultiEngineScheduler(
            self, num_engines=engines, arena=self.arena,
            # master/worker rendezvous + four-phase handshake: the shape HW-verified at 8 engines (ACT uses
            # it). Symmetric+four_phase deadlocked in the first barrier (primary retired 15 instructions).
            allow_unaligned_rows=True, handshake="four_phase", region_rendezvous="master_worker",
            allow_more_than_two_engines=True)
        if self.arena is not None:
            report(self.arena.describe())
            report(f"[layout] model map at 0x{bases[0]:X} (params) / 0x{bases[1]:X} (tensors) / 0x{bases[2]:X} (programs)")
        self.shards = {}
        self._isa_reg_counter = self.FIRST_FREE_REG      # dynamic kernels allocate GPRs above the fixed ones
        self._allocate_buffers()
        self._register_per_engine()
        self._rope_host = self._rope_tables_host(self.CAP_POS)
        self.from_bin = False
        if from_bin and self.bins_match(BIN_DIR):
            self.load_bins(BIN_DIR)
            self.from_bin = True
        else:
            assert ref is not None, "no usable frozen image and no weights to compile from"
            self.prep = WeightPrep(ref.w)
            self._upload_weights()
            self._materialize_shards()
            self.compile_all()

    # ------------------------------------------------------------------ geometry
    @staticmethod
    def frames(n_samples):
        t1 = (n_samples - 127) // 64 + 1
        t2 = (t1 - 7) // 3 + 1
        t3 = (t2 - 3) // 2 + 1
        return t1, t2, t3

    def _set_caps(self, n_cap):
        """Compile geometry: the programs are emitted for exactly these frame counts."""
        t1, t2, t3 = self.frames(n_cap)
        self.T1, self.T2, self.T3 = t1, t2, t3
        self.T3P = _ceil(t3, V)
        self.CAP_T1 = _ceil(t1, V)
        self.CAP_T2 = _ceil(t2, V) + 128        # + zero slack rows read by the conv3 im2col (<= 2*63+2)
        self.CAP_T3 = self.T3P
        self.CAP_N = self.CAP_T1 * 64 + 128
        self.CAP_POS = max(self.CAP_T3, self.MAX_STEPS_PAD)
        self.N_CAP = n_cap

    # ------------------------------------------------------------------ helpers
    def _store(self, name, t):
        self.w[name] = store_weight(self, t.to(torch.bfloat16))
        return self.w[name]

    def _tensor(self, name, n_elems, zero=False):
        a = self.allocate_tensor_dram(n_elems * BPE, label=name)
        self.buf[name] = (a, n_elems)
        if zero:
            self._zero(a, n_elems)
        return a

    def _zero(self, addr, n_elems):
        self.dma_to_accelerator_memory(addr, torch.zeros(n_elems, dtype=torch.bfloat16))

    def _write(self, addr, t):
        self.dma_to_accelerator_memory(addr, t.to(torch.bfloat16).contiguous().reshape(-1))

    def _read(self, addr, shape):
        return self.dma_from_accelerator_memory(addr, torch.Size(shape)).float()

    def _mm(self, M, K, N, A, B, OUT, bias=None, bias_mode="broadcast_N", ue=None, **kw):
        (ue or self).matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=B, OUTPUT_DRAM_ADDR=OUT,
                                     C_DRAM_ADDR=bias, bias_mode=bias_mode, **kw)

    def _pe(self, name, e):
        """Per-engine scratch address (engine 0 = the primary's own buffer)."""
        return self.sched.per_engine_addr(name, e)

    def _compile(self, name, emit):
        """capture -> HALT -> write to the next program slot. Programs are bump-allocated and
        never reset, so the whole set can be read back and frozen into programs.bin."""
        t0 = time.time()
        self.start_capture()
        emit()
        self.stop_capture()
        self.generate_instruction_halt()
        size = self.get_capture_instruction_size_bytes()
        addr = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(addr)
        self.allocate_program_dram(size, label=name)
        self.clear_capture_buffer()
        assert self.get_program_dram_addr() <= self.PRIMARY_PROGRAM_LIMIT, "primary program region overran the worker arenas"
        self.prog[name] = (addr, size)
        self.prog_order.append(name)
        print(f"[compile {name}] {size // 32} instr @0x{addr:X}, {time.time() - t0:.2f}s")

    def _exec(self, name, timeout=300.0):
        addr, _ = self.prog[name]
        t0 = time.time()
        self.start_execute_from_dram(addr)
        try:
            self.wait_queue(timeout)
        finally:
            if self.NE > 1:
                busy = [i for i, e in enumerate(self.sched.engines) if e.is_queue_busy()]
                if busy:
                    report(f"[exec {name}] engines still busy after {time.time() - t0:.1f}s: {busy} "
                           f"(barrier desync or missing worker)")
        dt = time.time() - t0
        print(f"[exec {name}] hw {dt:.3f}s")
        self._hw_time += dt
        return dt

    @staticmethod
    def _report(*a):
        report(*a, flush=True)

    # -- 64-lane vector ops on DRAM (A operand in URAM A, B in URAM B, result in URAM A) ----
    def _vload(self, ue, a, b, n):
        ue.accelerator_memory_to_sram(a, URAM_A_BASE, n)
        if b is not None:
            ue.accelerator_memory_to_sram(b, URAM_B_BASE, n)

    def vmul(self, a, b, o, n=V, ue=None):
        ue = ue or self
        self._vload(ue, a, b, n); ue.eltwise_mul_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, n)
        ue.sram_to_accelerator_memory(URAM_A_BASE, o, n)

    def vadd(self, a, b, o, n=V, ue=None):
        ue = ue or self
        self._vload(ue, a, b, n); ue.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, n)
        ue.sram_to_accelerator_memory(URAM_A_BASE, o, n)

    def vsub(self, a, b, o, n=V, ue=None):
        ue = ue or self
        self._vload(ue, a, b, n); ue.eltwise_sub_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, n)
        ue.sram_to_accelerator_memory(URAM_A_BASE, o, n)

    def vscale(self, a, s, o, n=V, ue=None):
        ue = ue or self
        self._vload(ue, a, None, n)
        ue.broadcast_mul(scalar=float(s), sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=n)
        ue.sram_to_accelerator_memory(URAM_A_BASE, o, n)

    def vshift(self, a, s, o, n=V, ue=None):
        ue = ue or self
        self._vload(ue, a, None, n)
        ue.broadcast_add(scalar=float(s), sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=n)
        ue.sram_to_accelerator_memory(URAM_A_BASE, o, n)

    # ------------------------------------------------------------------ weights
    def _upload_weights(self):
        P = self.prep
        t0 = time.time()
        self.w["identity64"] = store_identity_matrix(self)
        self._store("identity448", torch.eye(self.CP))
        # constants for GroupNorm / reductions
        ones_red = torch.zeros(V, RED_K); ones_red[0] = 1.0
        self._store("ones_red", ones_red)                                   # [64, 4096]
        bcol = torch.zeros(self.CP, V); bcol[: self.C, 0] = 1.0
        self._store("bcol", bcol)                                           # [448, 64]
        self._store("ones448x64", torch.ones(self.CP, V))
        self._store("e0", torch.eye(V)[0])                                  # [64] lane-0 mask
        self._store("ones64", torch.ones(V))
        self._store("gn_diag_gamma", torch.diag(P.pad_vec(P.p("model.encoder.groupnorm.weight"), self.CP)))
        self._store("gn_beta", P.pad_vec(P.p("model.encoder.groupnorm.bias"), self.CP))
        self._store("w_mean", P.w_mean)
        # conv stem
        self._store("conv1_w", P.conv1())
        self._store("conv2_w", P.conv2())
        self._store("conv2_b", P.p("model.encoder.conv2.bias"))
        self._store("conv3_w", P.conv3())
        self._store("conv3_b", P.pad_vec(P.p("model.encoder.conv3.bias"), self.CP))
        # rope tables (shared by encoder positions and decoder positions)
        self._store_rope_tables(self.CAP_POS)
        # encoder layers
        for i in range(self.N_ENC):
            p = f"model.encoder.layers.{i}."
            for h in range(self.H):
                self._store(f"enc{i}.q{h}", P.head_proj(P.p(p + "self_attn.q_proj.weight"), h, True, P.q_scale))
                self._store(f"enc{i}.k{h}", P.head_proj(P.p(p + "self_attn.k_proj.weight"), h, True))
                self._store(f"enc{i}.v{h}", P.head_proj(P.p(p + "self_attn.v_proj.weight"), h, False))
            for h, blk in enumerate(P.o_proj_heads(P.p(p + "self_attn.o_proj.weight"))):
                self._store(f"enc{i}.o{h}", blk)
            self._store(f"enc{i}.ln1", P.ln_gamma(p + "input_layernorm.weight"))
            self._store(f"enc{i}.ln2", P.ln_gamma(p + "post_attention_layernorm.weight"))
            self._store(f"enc{i}.fc1_w", P.pad_rows_cols(P.p(p + "mlp.fc1.weight"), self.I, self.CP))
            self._store(f"enc{i}.fc1_b", P.p(p + "mlp.fc1.bias"))
            self._store(f"enc{i}.fc2_w", P.mean_fold_rows(P.p(p + "mlp.fc2.weight")))
            self._store(f"enc{i}.fc2_b", P.mean_fold_bias(P.p(p + "mlp.fc2.bias")))
        self._store("enc.ln_final", P.ln_gamma("model.encoder.layer_norm.weight"))
        # decoder layers
        for i in range(self.N_DEC):
            p = f"model.decoder.layers.{i}."
            # self-attn: q as ONE [512, 448] matmul (M=1 output is already [8,64] contiguous)
            q_full = torch.cat([P.head_proj(P.p(p + "self_attn.q_proj.weight"), h, True, P.q_scale)
                                for h in range(self.H)], dim=0)
            self._store(f"dec{i}.q", q_full)
            for h in range(self.H):
                self._store(f"dec{i}.k{h}", P.head_proj(P.p(p + "self_attn.k_proj.weight"), h, True))
                self._store(f"dec{i}.v{h}", P.head_proj(P.p(p + "self_attn.v_proj.weight"), h, False))
            self._store(f"dec{i}.o", P.o_proj_full(P.p(p + "self_attn.o_proj.weight")))
            # cross-attn: no rope -> no permutation
            qx = torch.cat([P.head_proj(P.p(p + "encoder_attn.q_proj.weight"), h, False, P.q_scale)
                            for h in range(self.H)], dim=0)
            self._store(f"dec{i}.xq", qx)
            for h in range(self.H):
                self._store(f"dec{i}.xk{h}", P.head_proj(P.p(p + "encoder_attn.k_proj.weight"), h, False))
                self._store(f"dec{i}.xv{h}", P.head_proj(P.p(p + "encoder_attn.v_proj.weight"), h, False))
            self._store(f"dec{i}.xo", P.o_proj_full(P.p(p + "encoder_attn.o_proj.weight")))
            self._store(f"dec{i}.ln1", P.ln_gamma(p + "input_layernorm.weight"))
            self._store(f"dec{i}.ln2", P.ln_gamma(p + "post_attention_layernorm.weight"))
            self._store(f"dec{i}.ln3", P.ln_gamma(p + "final_layernorm.weight"))
            fc1_w = P.p(p + "mlp.fc1.weight")                   # [3328, 416]: x half then gate half
            fc1_b = P.p(p + "mlp.fc1.bias")
            self._store(f"dec{i}.fc1x_w", P.pad_rows_cols(fc1_w[: self.I], self.I, self.CP))
            self._store(f"dec{i}.fc1x_b", fc1_b[: self.I])
            self._store(f"dec{i}.fc1g_w", P.pad_rows_cols(fc1_w[self.I:], self.I, self.CP))
            self._store(f"dec{i}.fc1g_b", fc1_b[self.I:])
            self._store(f"dec{i}.fc2_w", P.mean_fold_rows(P.p(p + "mlp.fc2.weight")))
            self._store(f"dec{i}.fc2_b", P.mean_fold_bias(P.p(p + "mlp.fc2.bias")))
        self._store("dec.ln_final", P.ln_gamma("model.decoder.norm.weight"))
        causal = torch.zeros(self.MAX_STEPS, self.MAX_STEPS_PAD)
        for st in range(self.MAX_STEPS):
            causal[st, st + 1:] = float("-inf")
        self._store("dec_causal_bias", causal)            # [194, 256]
        self._store("embed_in", P.embed_in())
        self._store("embed_out", P.embed_out())
        used = self.get_params_dram_addr() - self._params_dram_base
        self._report(f"[weights] {len(self.w)} tensors, {used / 2**20:.1f} MB params, {time.time() - t0:.1f}s")

    def _rope_tables_host(self, n_pos):
        """Three [n_pos, 64] tables for rope_hf_core_dram on the permuted 64-lane head.
        Lanes 0..15 of each half carry rotary pair i; the rest are identity (cos 1, sin 0).
        The core computes lo = x_lo*cos_lo + x_hi*sin_lo, hi = x_hi*cos_hi + x_lo*sin_hi,
        so sin_lo is pre-negated. cos serves both halves."""
        half = self.RD // 2
        inv_freq = 1.0 / (self.THETA ** (torch.arange(0, self.RD, 2).float() / self.RD))   # [16]
        pos = torch.arange(n_pos).float()
        fr = torch.outer(pos, inv_freq)                                                      # [n_pos, 16]
        cos = torch.ones(n_pos, V); sin_lo = torch.zeros(n_pos, V); sin_hi = torch.zeros(n_pos, V)
        cos[:, :half] = fr.cos()
        sin_lo[:, :half] = -fr.sin()
        sin_hi[:, :half] = fr.sin()
        return {"cos": cos.to(torch.bfloat16), "sin_lo": sin_lo.to(torch.bfloat16), "sin_hi": sin_hi.to(torch.bfloat16)}

    # ------------------------------------------------------------------ private weight shards
    # (name, K, N) of every decode projection that is N-split from controller-private copies
    def _shard_specs(self):
        CP, D, H, I = self.CP, self.DP, self.H, self.I
        specs = [("embed_out", CP, self.VOCAB)]
        for l in range(self.N_DEC):
            specs += [(f"dec{l}.o", H * D, CP), (f"dec{l}.xo", H * D, CP),
                      (f"dec{l}.fc1x_w", CP, I), (f"dec{l}.fc1g_w", CP, I), (f"dec{l}.fc2_w", I, CP)]
        return specs

    def _materialize_shards(self):
        """Copy each engine's column block of every decode projection into its private window, then
        park the workers' own params allocators past the shards (their LN kernels allocate zero rows
        through ue.allocate_params_dram, which would otherwise land on top of the copies)."""
        if self.NE == 1:
            return
        t0 = time.time()
        for name, K, N in self._shard_specs():
            # trailing remainder: the primary (engine 0) also runs the LNs and the rendezvous, so it
            # gets the smallest block; narrow projections (448 cols = 7 blocks) use 7 of 8 engines.
            self.shards[name] = self.sched.shard_bf16_weight(
                name, self.w[name], K=K, N=N, layers=1, main_layer_stride=0,
                remainder="trailing", max_engines=min(self.NE, N // V), verbose=False)
        self._park_worker_params()
        used = [f"{u / 2**20:.1f}" for u in self.sched.private_usage()]
        self._report(f"[shards] {len(self.shards)} projections materialized; private MB per engine: {used} "
                     f"({time.time() - t0:.1f}s)")

    def _park_worker_params(self):
        for e, wk in enumerate(self.sched.workers, start=1):
            used = self.arena.usage()[e]
            if wk.get_params_dram_usage() < used:
                wk.allocate_params_dram(used - wk.get_params_dram_usage())

    def _rebuild_shard_table(self):
        """Recreate ShardedWeight objects (addresses only) in the exact allocation order of
        _materialize_shards, without copying: the shards.bin restore already holds the bytes."""
        from multi_engine_shard import ShardedWeight, WeightShard, DENSE_BF16
        cursors = [self.arena.region(e).weight_base for e in range(self.NE)]
        for name, K, N in self._shard_specs():
            sw = ShardedWeight(name=name, K=K, N=N, layers=1, data_type=DENSE_BF16)
            for e, (off, cols) in enumerate(self.sched.split_cols(N, remainder="trailing", max_engines=min(self.NE, N // V))):
                addr = (cursors[e] + 63) & ~63
                sw.shards.append(WeightShard(engine_idx=e, col_offset=off, cols=cols, weight_addr=addr,
                                             scale_addr=0, layer_stride=cols * K * BPE, scale_layer_stride=0))
                cursors[e] = addr + cols * K * BPE
            self.shards[name] = sw

    def _shard_addr(self, name, e):
        """(weight_addr, col_offset, cols) of engine e's block of a materialized projection, or None."""
        sh = self.shards[name].shard_or_none(e)
        return None if sh is None else (sh.weight_addr, sh.col_offset, sh.cols)

    def _store_rope_tables(self, n_pos):
        t = self._rope_tables_host(n_pos)
        self._store("rope_cos", t["cos"])
        self._store("rope_sin_lo", t["sin_lo"])
        self._store("rope_sin_hi", t["sin_hi"])

    def _rope(self, x, out, rows, pos0, ue=None, tables=None):
        """In-place-capable RoPE over `rows` consecutive [64] rows starting at position pos0.
        `tables` = (cos, sin_lo, sin_hi) row-0 addresses to use instead of the static tables at pos0
        (the decode step passes its per-step host-written rows)."""
        ue = ue or self
        row_b = V * BPE
        for r0, take in self.chunk_ranges(rows, 256):            # 4 tables * 256 rows * 128B = 128KB of URAM B
            t = (pos0 + r0) * row_b
            cos, slo, shi = tables if tables is not None else (
                self.w["rope_cos"] + t, self.w["rope_sin_lo"] + t, self.w["rope_sin_hi"] + t)
            rope_hf_core_dram(ue, M=take, D=self.DP,
                              X_DRAM_ADDR=x + r0 * row_b, OUTPUT_DRAM_ADDR=out + r0 * row_b,
                              COS_DRAM_ADDR=cos, SIN_DRAM_ADDR=slo,
                              COS_LO_PAD_ADDR=cos, COS_HI_PAD_ADDR=cos,
                              SIN_LO_PAD_ADDR=slo, SIN_HI_PAD_ADDR=shi)

    # ------------------------------------------------------------------ buffers
    def _allocate_buffers(self):
        T1, T2, T3, CP = self.CAP_T1, self.CAP_T2, self.CAP_T3, self.CAP_POS
        H, D = self.H, self.DP
        # conv stem
        self._tensor("pcm", self.CAP_N, zero=True)
        self._tensor("im2col1", T1 * 128)
        self._tensor("conv1_out", _ceil(T1 * self.CP, RED_K), zero=True)       # flat-reducible
        self._tensor("conv1_sq", _ceil(T1 * self.CP, RED_K), zero=True)
        self._tensor("conv1_c", T1 * self.CP)
        self._tensor("gn_out", T1 * self.CP)                                   # conv2 im2col stays within T1 rows
        self._tensor("im2col2", T2 * 7 * self.CP)
        self._tensor("conv2_out", T2 * 2 * self.C, zero=True)                  # slack rows stay zero
        self._tensor("im2col3", T3 * 3 * 2 * self.C)
        self._tensor("conv3_out", T3 * self.CP)
        # reductions / stats (64-lane vectors, 128B slots)
        # reduction arena: every stage of every reduction gets a fresh zeroed slice (sizes shrink 64x
        # per stage, so 2 reductions * ~1.02 * n/64 elements fit in 3 * ceil(n/64))
        self._tensor("red_arena", 3 * (_ceil(T1 * self.CP // V, RED_K) + RED_K), zero=True)
        for n in ("sum", "sqsum", "mean", "ex2", "m2", "var", "y", "t", "rstd", "invcnt"):
            self._tensor("v_" + n, V, zero=True)
        self._tensor("rstd64", V * V, zero=True)
        self._tensor("mu448", self.CP); self._tensor("negmu448", self.CP)
        self._tensor("rstd_col", self.CP * V); self._tensor("rstd_mat", self.CP * self.CP)
        self._tensor("d_gamma", self.CP * self.CP)
        # encoder
        self._tensor("x_a", T3 * self.CP); self._tensor("x_b", T3 * self.CP)
        self._tensor("h", T3 * self.CP)
        self._tensor("zeros", T3 * self.CP, zero=True)
        self._tensor("q", H * T3 * D); self._tensor("k", H * T3 * D); self._tensor("v", H * T3 * D)
        self._tensor("attn", H * T3 * D)
        self._tensor("acc_a", T3 * self.CP); self._tensor("acc_b", T3 * self.CP)
        self._tensor("op0", T3 * self.CP); self._tensor("op1", T3 * self.CP)          # o_proj partials (per engine)
        self._tensor("mlp", T3 * self.I)
        self._tensor("flash_scratch", D * T3 + T3 * T3)
        self._tensor("enc_bias", T3 * T3)
        # decoder
        self._tensor("xk", self.N_DEC * H * T3 * D); self._tensor("xv", self.N_DEC * H * T3 * D)
        self._tensor("kcache", self.N_DEC * H * self.MAX_STEPS_PAD * D, zero=True)
        self._tensor("vcache", self.N_DEC * H * self.MAX_STEPS_PAD * D, zero=True)
        self._tensor("dx_a", self.CP); self._tensor("dx_b", self.CP); self._tensor("dh", self.CP)
        self._tensor("dq", H * D); self._tensor("dattn", H * D); self._tensor("dy", self.CP)
        self._tensor("dmlp_x", self.I); self._tensor("dmlp_g", self.I)
        self._tensor("dzeros", self.CP, zero=True)
        self._tensor("dec_bias_row", T3)                                       # cross-attn frame mask [1, T3P]
        self._tensor("dec_self_bias", self.MAX_STEPS_PAD)                       # causal row [1, 256], host per step
        self._tensor("rope_cur", 3 * V)                                         # cos | sin_lo | sin_hi rows for `step`
        self._tensor("k_tmp", D); self._tensor("v_tmp", D)                     # one head's new K/V row before scatter
        self._tensor("dec_scratch", D * T3 + T3)                                # decode_flash scratch (cross-attn)
        self._tensor("ua_scratch", (D + self.MAX_STEPS_PAD) * self.MAX_STEPS_PAD + D)   # unified_attention scratch
        self._tensor("logits", self.VOCAB)
        used = self.get_tensor_dram_addr() - self._tensor_dram_base
        assert self.get_tensor_dram_addr() <= self._program_dram_base, "tensor region overran program base"
        self._report(f"[buffers] {used / 2**20:.1f} MB tensors")

    def _register_per_engine(self):
        """Scratch that is WRITTEN inside sharded regions must be private per engine. Engine 0 keeps
        the model's buffers; workers get copies from their own tensor allocators (deterministic, so
        the from_bin path reproduces the same addresses by calling this in the same order)."""
        S = self.sched
        S.alloc_attn_scratch("attn_scratch", self.T3P, self.DP, self.A("flash_scratch"))
        S.register_per_engine("dec_scratch", self.A("dec_scratch"), self.buf["dec_scratch"][1] * BPE)
        for name in ("k_tmp", "v_tmp"):
            S.register_per_engine(name, self.A(name), self.DP * BPE)
        S.register_per_engine("ua_scratch", self.A("ua_scratch"), self.buf["ua_scratch"][1] * BPE)
        for name in ("op0", "op1", "v_sum", "v_sqsum"):
            a, n = self.buf[name]
            S.register_per_engine(name, a, n * BPE)
        a, n = self.buf["red_arena"]
        S.register_per_engine("red_arena", a, n * BPE, init_tensor=torch.zeros(n, dtype=torch.bfloat16))

    def _region(self, M, body, mode="even"):
        """Row-split region: body(ue, row_offset, rows, engine_idx) replayed per engine between barriers."""
        S = self.sched
        prev, S.split_mode = S.split_mode, mode
        try:
            S.sharded_region(M, lambda ctx: body(ctx.unsafe_ue, ctx.row_offset, ctx.rows, ctx.engine_idx))
        finally:
            S.split_mode = prev

    def _enc_head(self, name, h):
        a, _ = self.buf[name]
        return a + h * self.CAP_T3 * self.DP * BPE

    def _xkv(self, name, layer, h):
        a, _ = self.buf[name]
        return a + (layer * self.H + h) * self.CAP_T3 * self.DP * BPE

    def _cache(self, name, layer, h):
        a, _ = self.buf[name]
        return a + (layer * self.H + h) * self.MAX_STEPS_PAD * self.DP * BPE

    def A(self, name):
        return self.buf[name][0]

    # ------------------------------------------------------------------ conv stem
    def _im2col(self, src, dst, rows, row_elems, src_pitch_bytes, ue=None):
        """dst[r, :] = src[r*pitch : r*pitch + row_elems] via strided DRAM->URAM reads.
        chunk = row_elems*2 bytes and pitch must both be 32B multiples (AXI beat rule)."""
        ue = ue or self
        chunk = row_elems * BPE
        assert chunk % 32 == 0 and src_pitch_bytes % 32 == 0, (chunk, src_pitch_bytes)
        per_pass = max(1, URAM_NEAR_FULL_ELEMENTS // row_elems)
        for r0, take in self.chunk_ranges(rows, per_pass):
            ue.accelerator_memory_to_sram(src + r0 * src_pitch_bytes, URAM_A_BASE, take * row_elems,
                                          stride_bytes_per_chunk=chunk, stride_jump_bytes=src_pitch_bytes)
            ue.sram_to_accelerator_memory(URAM_A_BASE, dst + r0 * chunk, take * row_elems)

    def _reduce_sum(self, src, n, out_vec, ue, arena):
        """out_vec[0] = sum(src[0:n]). n must be a 4096 multiple at the first stage (so no neighbour
        rows are read) and the arena must be zero beyond what each stage writes.
        Each stage: view as [rows, 4096] @ ones_red[64, 4096]^T -> [rows, 64] (lane 0 = partials)."""
        assert n % RED_K == 0, f"first reduction stage needs n % 4096 == 0, got {n}"
        cur, cur_n = src, n
        while True:
            rows = (cur_n + RED_K - 1) // RED_K
            if rows == 1:
                self._mm(1, RED_K, V, cur, self.w["ones_red"], out_vec, ue=ue)
                return
            dst = self._red_alloc(arena, rows * V)
            self._mm(rows, RED_K, V, cur, self.w["ones_red"], dst, ue=ue)
            cur, cur_n = dst, rows * V

    @staticmethod
    def _red_alloc(arena, n_elems):
        """Bump-allocate a 4096-aligned slice of a zeroed reduction arena dict {base, cap, cur}."""
        take = _ceil(n_elems, RED_K)
        assert arena["cur"] + take <= arena["cap"], "reduction arena exhausted"
        a = arena["base"] + arena["cur"] * BPE
        arena["cur"] += take
        return a

    def _emit_conv_stem(self):
        """Stem at the compiled cap geometry. Regions (each = entry barrier + body per engine + exit barrier):
        A conv1+tanh+x^2+partial sums (64-row blocks) | primary: stats | B GN apply | C conv2 | D conv3+W_mean+LN0."""
        CP, C, w = self.CP, self.C, self.w
        T1c, T2, T3p = self.CAP_T1, self.T2, self.T3P
        NE = self.NE
        A = self.A

        def body_a(ue, r0, rows, e):
            im = A("im2col1") + r0 * 128 * BPE
            x = A("conv1_out") + r0 * CP * BPE
            sq = A("conv1_sq") + r0 * CP * BPE
            self._im2col(A("pcm") + r0 * 64 * BPE, im, rows, 128, 64 * BPE, ue=ue)
            self._mm(rows, 128, CP, im, w["conv1_w"], x, ue=ue)
            tanh_core_dram(ue, rows, CP, x, x, w["identity448"])
            n = rows * CP
            eltwise_mul_core_dram(ue, n, x, x, sq)
            arena = {"base": self._pe("red_arena", e), "cap": self.buf["red_arena"][1], "cur": 0}
            self._reduce_sum(x, n, self._pe("v_sum", e), ue, arena)
            self._reduce_sum(sq, n, self._pe("v_sqsum", e), ue, arena)
        self._region(T1c, body_a, mode="blocks")          # rows per engine are 64-multiples -> n % 4096 == 0

        # ---- primary only: combine partials, stats, rsqrt, D_gamma ----
        for e in range(1, NE):
            self.vadd(A("v_sum"), self._pe("v_sum", e), A("v_sum"))
            self.vadd(A("v_sqsum"), self._pe("v_sqsum", e), A("v_sqsum"))
        # 1/(T1_real*416) is per-run data (host-uploaded 64-vector), so the program is clip-independent
        self.vmul(A("v_sum"), A("v_invcnt"), A("v_mean"))
        self.vmul(A("v_sqsum"), A("v_invcnt"), A("v_ex2"))
        self.vmul(A("v_mean"), A("v_mean"), A("v_m2"))
        self.vsub(A("v_ex2"), A("v_m2"), A("v_var"))
        self.vshift(A("v_var"), self.GN_EPS, A("v_var"))
        # rstd = 1/sqrt(var) by Newton: y <- y * (1.5 - 0.5*var*y^2), y0 = 1 (converges for var <= 1)
        y, t, var = A("v_y"), A("v_t"), A("v_var")
        self.vscale(w["ones64"], 1.0, y)
        for _ in range(NEWTON_ITERS):
            self.vmul(y, y, t)
            self.vmul(t, var, t)
            self.vscale(t, -0.5, t)
            self.vshift(t, 1.5, t)
            self.vmul(y, t, y)
        self.vmul(y, w["e0"], A("rstd64"))                     # row 0 of the 64x64 zero block
        self._mm(1, V, CP, A("v_mean"), w["bcol"], A("mu448"))
        self.vscale(A("mu448"), -1.0, A("negmu448"), n=CP)
        self._mm(CP, V, V, w["ones448x64"], A("rstd64"), A("rstd_col"))
        self._mm(CP, V, CP, A("rstd_col"), w["bcol"], A("rstd_mat"))
        eltwise_mul_core_dram(self, CP * CP, A("rstd_mat"), w["gn_diag_gamma"], A("d_gamma"))

        def body_b(ue, r0, rows, e):      # GN apply: (x - mu) then * rstd*gamma + beta
            x = A("conv1_out") + r0 * CP * BPE
            xc = A("conv1_c") + r0 * CP * BPE
            self._mm(rows, CP, CP, x, w["identity448"], xc, bias=A("negmu448"), ue=ue)
            self._mm(rows, CP, CP, xc, A("d_gamma"), A("gn_out") + r0 * CP * BPE, bias=w["gn_beta"], ue=ue)
        self._region(T1c, body_b)

        def body_c(ue, r0, rows, e):      # conv2 k=7 s=3 (+gelu): im2col rows are contiguous 7*448 windows
            im = A("im2col2") + r0 * 7 * CP * BPE
            self._im2col(A("gn_out") + 3 * r0 * CP * BPE, im, rows, 7 * CP, 3 * CP * BPE, ue=ue)
            self._mm(rows, 7 * CP, 2 * C, im, w["conv2_w"], A("conv2_out") + r0 * 2 * C * BPE,
                     bias=w["conv2_b"], gelu_enable=True, ue=ue)
        self._region(T2, body_c)

        def body_d(ue, r0, rows, e):      # conv3 k=3 s=2 (+gelu) on T3p rows, W_mean invariant, h = LN_in0(x)
            im = A("im2col3") + r0 * 3 * 2 * C * BPE
            self._im2col(A("conv2_out") + 2 * r0 * 2 * C * BPE, im, rows, 3 * 2 * C, 2 * 2 * C * BPE, ue=ue)
            c3 = A("conv3_out") + r0 * CP * BPE
            self._mm(rows, 3 * 2 * C, CP, im, w["conv3_w"], c3, bias=w["conv3_b"], gelu_enable=True, ue=ue)
            off = r0 * CP * BPE
            self._mm(rows, CP, CP, c3, w["w_mean"], A("x_a") + off, ue=ue)
            layer_norm_core_dram_post_add(ue, rows, CP, A("x_a") + off, A("zeros") + off,
                                          A("x_b") + off, A("h") + off, GAMMA_DRAM_ADDR=w["enc0.ln1"])
        self._region(T3p, body_d)

    # ------------------------------------------------------------------ encoder
    def _emit_encoder_layer(self, i, x_in, x_out):
        """x_in: residual (pad lanes = mean); h = LN_in_i(x_in) already. Writes x_out and
        h = LN_in_{i+1}(x_out) (or the final encoder LN after the last layer).
        Attention is head-split (one head's q/k/v/RoPE/flash/o_proj-partial per engine), the rest row-split."""
        CP, D, H, T3p, w, A = self.CP, self.DP, self.H, self.T3P, self.w, self.A
        S = self.sched
        hbuf = A("h")
        final_partial = {}

        def attn_body(ctx):
            ue, e = ctx.ue, ctx.engine_idx
            heads = [h for (h0, _kv, n_q) in ctx.call_runs() for h in range(h0, h0 + n_q)]
            for h in heads:
                q, k, v, o = (self._enc_head(n, h) for n in ("q", "k", "v", "attn"))
                self._mm(T3p, CP, D, hbuf, w[f"enc{i}.q{h}"], q, ue=ue)
                self._mm(T3p, CP, D, hbuf, w[f"enc{i}.k{h}"], k, ue=ue)
                self._mm(T3p, CP, D, hbuf, w[f"enc{i}.v{h}"], v, ue=ue)
                self._rope(q, q, T3p, 0, ue=ue)
                self._rope(k, k, T3p, 0, ue=ue)
                prefill_flash_attention_core(ue, D, T3p, q, k, v, o, ctx.scratch("attn_scratch"),
                                             w["identity64"], BIAS_DRAM_ADDR=A("enc_bias"), num_q_heads=1)
            # this engine's o_proj partial = sum over its heads of attn_h @ Wo_h^T (full_matrix chain, ping-pong)
            parts = [self._pe("op0", e), self._pe("op1", e)]
            for j, h in enumerate(heads):
                self._mm(T3p, D, CP, self._enc_head("attn", h), w[f"enc{i}.o{h}"], parts[j % 2],
                         bias=None if j == 0 else parts[(j - 1) % 2], bias_mode="full_matrix", ue=ue)
            final_partial[e] = parts[(len(heads) - 1) % 2]
        S.head_sharded_region(H, T3p, D, attn_body)
        partials = [final_partial[e] for e in range(self.NE)]
        if self.NE == 1:
            attn_out = partials[0]
        else:
            S.reduce_add(partials, A("acc_a"), T3p, CP, parallel=True)   # cross-engine sum -> acc_a
            attn_out = A("acc_a")

        def ln1_body(ue, r0, rows, e):   # x_mid = x_in + attn ; h = LN_post(x_mid)   (x_out holds x_mid)
            off = r0 * CP * BPE
            layer_norm_core_dram_post_add(ue, rows, CP, x_in + off, attn_out + off, x_out + off, hbuf + off,
                                          GAMMA_DRAM_ADDR=w[f"enc{i}.ln2"])
        self._region(T3p, ln1_body)

        next_gamma = w[f"enc{i + 1}.ln1"] if i + 1 < self.N_ENC else w["enc.ln_final"]

        def mlp_body(ue, r0, rows, e):   # fc1 (gelu fused) -> fc2 (mean-folded) ; x_out += mlp ; h = LN_next(x_out)
            off = r0 * CP * BPE
            m = A("mlp") + r0 * self.I * BPE
            self._mm(rows, CP, self.I, hbuf + off, w[f"enc{i}.fc1_w"], m, bias=w[f"enc{i}.fc1_b"], gelu_enable=True, ue=ue)
            self._mm(rows, self.I, CP, m, w[f"enc{i}.fc2_w"], A("acc_b") + off, bias=w[f"enc{i}.fc2_b"], ue=ue)
            layer_norm_core_dram_post_add(ue, rows, CP, x_out + off, A("acc_b") + off, x_out + off, hbuf + off,
                                          GAMMA_DRAM_ADDR=next_gamma)
        self._region(T3p, mlp_body)

    def _emit_cross_kv(self):
        """K/V for all decoder cross-attention layers from the final encoder output (h = enc LN); row-split."""
        T3p, CP, D, w, A = self.T3P, self.CP, self.DP, self.w, self.A

        def body(ue, r0, rows, e):
            for l in range(self.N_DEC):
                for h in range(self.H):
                    self._mm(rows, CP, D, A("h") + r0 * CP * BPE, w[f"dec{l}.xk{h}"],
                             self._xkv("xk", l, h) + r0 * D * BPE, ue=ue)
                    self._mm(rows, CP, D, A("h") + r0 * CP * BPE, w[f"dec{l}.xv{h}"],
                             self._xkv("xv", l, h) + r0 * D * BPE, ue=ue)
        self._region(T3p, body)

    # ------------------------------------------------------------------ decoder
    def _scatter_row(self, ue, src, cache_base):
        """Copy one [64] row from `src` to cache_base + V_CACHE_SIZE_REG (runtime row offset)."""
        ue.accelerator_memory_to_sram(src, 0x10000, self.DP)
        ue.generate_instruction_add_imm(self.R_VCACHE, ue_35bit_addr_shifter(cache_base), self.R_TMP)
        ue.sram_to_accelerator_memory(0x10000, 0, self.DP, general_reg_src=self.R_TMP)

    def _round(self, body):
        """One master/worker round: body(ue, engine_idx) on every engine. Every engine takes part in
        every round (an engine with nothing to do still has to meet the rendezvous)."""
        S = self.sched
        if self.NE == 1:
            body(self, 0)
            return
        S.release()
        body(self, 0)
        for e in S.worker_indices():
            S.begin_worker_round(e)
            body(S.engines[e], e)
            S.end_worker_round(e)
        S.join()

    def _col_mm(self, name, K, N, A_addr, out_row, bias=None, **kw):
        """N-split M=1 projection: each engine multiplies its PRIVATE column shard and writes its
        columns straight into the shared output row (M=1 makes the column block contiguous).
        Single-engine: the plain matmul on the primary."""
        if self.NE == 1:
            self._mm(1, K, N, A_addr, self.w[name], out_row, bias=bias, **kw)
            return

        def body(ue, e):
            sh = self._shard_addr(name, e)
            if sh is None:
                return
            w_addr, off, cols = sh
            self._mm(1, K, cols, A_addr, w_addr, out_row + off * BPE,
                     bias=None if bias is None else bias + off * BPE, ue=ue, **kw)
        self._round(body)

    def _emit_decode_body(self):
        """One token for ANY step. dx_a holds the embedding row (pad lanes = mean), rope_cur the
        three table rows for this position, dec_self_bias the causal row; R_ALIGNED / R_VCACHE are
        primed by the dispatch entry (primary) and the worker preamble (workers).
        Per layer: [head round: q_h/k_h/v_h, RoPE, K/V scatter, self-attn] -> o_proj (N-split) -> LN
                   [head round: xq_h, cross-attn] -> xo (N-split) -> LN
                   fc1x/fc1g+silu (N-split, per-lane mul) -> fc2 (N-split) -> LN ; LM head (N-split)."""
        CP, D, H, I, T3p = self.CP, self.DP, self.H, self.I, self.T3P
        w, A, S = self.w, self.A, self.sched
        x = [A("dx_a"), A("dx_b")]
        hbuf, dq, dattn, dy = A("dh"), A("dq"), A("dattn"), A("dy")
        row_b = D * BPE
        rope_cur = (A("rope_cur"), A("rope_cur") + V * BPE, A("rope_cur") + 2 * V * BPE)
        kvp = self.MAX_STEPS_PAD
        layer_norm_core_dram_post_add(self, 1, CP, x[0], A("dzeros"), x[1], hbuf, GAMMA_DRAM_ADDR=w["dec0.ln1"])
        cur = 0
        for l in range(self.N_DEC):
            def self_attn(ctx, l=l):
                ue, e = ctx.ue, ctx.engine_idx
                ktmp, vtmp = ctx.scratch("k_tmp"), ctx.scratch("v_tmp")
                for h0, _kv, n_q in ctx.call_runs():
                    for h in range(h0, h0 + n_q):
                        q = dq + h * row_b
                        self._mm(1, CP, D, hbuf, w[f"dec{l}.q"] + h * D * CP * BPE, q, ue=ue)
                        self._rope(q, q, 1, 0, tables=rope_cur, ue=ue)
                        self._mm(1, CP, D, hbuf, w[f"dec{l}.k{h}"], ktmp, ue=ue)
                        self._rope(ktmp, ktmp, 1, 0, tables=rope_cur, ue=ue)
                        self._scatter_row(ue, ktmp, self._cache("kcache", l, h))
                        self._mm(1, CP, D, hbuf, w[f"dec{l}.v{h}"], vtmp, ue=ue)
                        self._scatter_row(ue, vtmp, self._cache("vcache", l, h))
                        ue.unified_attention_core(
                            batch=1, aligned_seq_len=kvp, head_dim=D,
                            Q_DRAM_ADDR=q,
                            K_DRAM_ADDR=self._cache("kcache", l, h), V_DRAM_ADDR=self._cache("vcache", l, h),
                            BIAS_DRAM_ADDR=A("dec_self_bias"), OUTPUT_DRAM_ADDR=dattn + h * row_b,
                            SCRATCH_DRAM_ADDR=ctx.scratch("ua_scratch"), IDENTITY_DRAM_ADDR=w["identity64"],
                            gpr_aligned_seq_len_reg=self.R_ALIGNED)
            S.head_sharded_region(H, kvp, D, self_attn)
            self._col_mm(f"dec{l}.o", H * D, CP, dattn, dy)
            layer_norm_core_dram_post_add(self, 1, CP, x[cur], dy, x[1 - cur], hbuf, GAMMA_DRAM_ADDR=w[f"dec{l}.ln2"])
            cur = 1 - cur

            def cross_attn(ctx, l=l):
                ue = ctx.ue
                for h0, _kv, n_q in ctx.call_runs():
                    for h in range(h0, h0 + n_q):
                        q = dq + h * row_b
                        self._mm(1, CP, D, hbuf, w[f"dec{l}.xq"] + h * D * CP * BPE, q, ue=ue)
                        decode_flash_attention_core(ue, D, T3p, q,
                                                    self._xkv("xk", l, h), self._xkv("xv", l, h),
                                                    dattn + h * row_b, ctx.scratch("dec_scratch"),
                                                    IDENTITY_DRAM_ADDR=w["identity64"],
                                                    BIAS_DRAM_ADDR=A("dec_bias_row"), num_q_heads=1, bias_row_stride=T3p)
            S.head_sharded_region(H, T3p, D, cross_attn)
            self._col_mm(f"dec{l}.xo", H * D, CP, dattn, dy)
            layer_norm_core_dram_post_add(self, 1, CP, x[cur], dy, x[1 - cur], hbuf, GAMMA_DRAM_ADDR=w[f"dec{l}.ln3"])
            cur = 1 - cur

            if self.NE == 1:
                self._mm(1, CP, I, hbuf, w[f"dec{l}.fc1x_w"], A("dmlp_x"), bias=w[f"dec{l}.fc1x_b"])
                self._mm(1, CP, I, hbuf, w[f"dec{l}.fc1g_w"], A("dmlp_g"), bias=w[f"dec{l}.fc1g_b"], silu_enable=True)
                eltwise_mul_core_dram(self, I, A("dmlp_x"), A("dmlp_g"), A("dmlp_g"))
            else:
                def fc1(ue, e, l=l):
                    wx, off, cols = self._shard_addr(f"dec{l}.fc1x_w", e)
                    wg, _, _ = self._shard_addr(f"dec{l}.fc1g_w", e)
                    self._mm(1, CP, cols, hbuf, wx, A("dmlp_x") + off * BPE, bias=w[f"dec{l}.fc1x_b"] + off * BPE, ue=ue)
                    self._mm(1, CP, cols, hbuf, wg, A("dmlp_g") + off * BPE, bias=w[f"dec{l}.fc1g_b"] + off * BPE,
                             silu_enable=True, ue=ue)
                    eltwise_mul_core_dram(ue, cols, A("dmlp_x") + off * BPE, A("dmlp_g") + off * BPE, A("dmlp_g") + off * BPE)
                self._round(fc1)
            self._col_mm(f"dec{l}.fc2_w", I, CP, A("dmlp_g"), dy, bias=w[f"dec{l}.fc2_b"])
            next_gamma = w[f"dec{l + 1}.ln1"] if l + 1 < self.N_DEC else w["dec.ln_final"]
            layer_norm_core_dram_post_add(self, 1, CP, x[cur], dy, x[1 - cur], hbuf, GAMMA_DRAM_ADDR=next_gamma)
            cur = 1 - cur
        # LM head (tied embedding), N-split from private shards; every engine's argmax register = local candidate
        self._col_mm("embed_out", CP, self.VOCAB, hbuf, A("logits"))

    def _emit_decode_program(self):
        S = self.sched
        S.begin_program()
        for wk in S.workers:
            wk._isa_reg_counter = self.FIRST_FREE_REG
        self._emit_decode_body()
        self.dec_worker_progs = S.finalize()

    def _emit_decode_dispatch(self):
        """194 entries x 4 instructions (128 B): prime R_ALIGNED + R_VCACHE, jump into the decode body."""
        body = self.prog["decode"][0]
        for step in range(self.MAX_STEPS):
            self.clear_inst_id()
            self.generate_instruction_add_set(self.R_ALIGNED, _ceil(step + 1, V))
            self.generate_instruction_add_set(self.R_VCACHE, ue_35bit_addr_shifter(step * self.DP * BPE))
            self.generate_instruction_jump_abs(ue_35bit_addr_shifter(body))
            self.generate_instruction_nop()
        assert self.capture_count == 4 * self.MAX_STEPS

    def _decode_gpr_sets(self, step):
        return [(self.R_ALIGNED, _ceil(step + 1, V)), (self.R_VCACHE, ue_35bit_addr_shifter(step * self.DP * BPE))]

    def _global_argmax(self):
        """Per-engine rank-1 candidates from the N-split LM head, compared by value on the host."""
        if self.NE == 1:
            return self.get_arg_max_index(rank=1)
        best, best_val = None, None
        for sh in self.shards["embed_out"].shards:
            e, off, cols = sh.engine_idx, sh.col_offset, sh.cols
            local = self.sched.engines[e].get_arg_max_index(rank=1)
            assert 0 <= local < cols, f"engine {e} argmax {local} outside its {cols}-column shard"
            g = off + local
            val = self.sched._read_bf16(self.A("logits") + g * BPE)
            if best_val is None or val > best_val:
                best, best_val = g, val
        return best

    # ------------------------------------------------------------------ compile (once, at cap)
    def _emit_encoder_program(self):
        """stem -> 8 layers -> cross K/V, one primary program + one program per worker."""
        self.sched.begin_program()
        self._emit_conv_stem()
        x = [self.A("x_a"), self.A("x_b")]
        for i in range(self.N_ENC):
            self._emit_encoder_layer(i, x[i % 2], x[(i + 1) % 2])
        self._emit_cross_kv()
        self.worker_progs = self.sched.finalize()

    def compile_all(self):
        t0 = time.time()
        T3p = self.T3P
        self._compile("encoder", self._emit_encoder_program)
        if self.NE > 1:
            self._report("[compile] worker programs: " +
                         ", ".join(f"{w.get_program_dram_usage() / 2**20:.1f} MB" for w in self.sched.workers))
        self._compile("decode", self._emit_decode_program)
        self._compile("decode_dispatch", self._emit_decode_dispatch)
        if self.NE > 1:
            self._report("[compile] decode worker programs: " +
                         ", ".join(f"{w.get_program_dram_usage() / 2**20:.1f} MB" for w in self.sched.workers))
        total = self.get_program_dram_usage()
        self._report(f"[compile] {len(self.prog)} programs, {total / 2**20:.1f} MB instructions, "
                     f"{time.time() - t0:.1f}s")

    # ------------------------------------------------------------------ execute (per clip)
    def execute(self, pcm: np.ndarray, max_new_tokens: int, debug_snr: bool = True):
        n = len(pcm)
        assert n <= self.N_CAP, f"audio {n / SAMPLE_RATE:.1f}s longer than the {self.max_seconds}s cap"
        T1, T2, T3 = self.frames(n)          # REAL frame counts (masks + GN count)
        T3p = self.T3P                        # compiled geometry
        self._report(f"[geometry] samples={n} real T1={T1} T2={T2} T3={T3}; compiled T1={self.T1} T3p={T3p}")
        self._hw_time = 0.0
        t_e2e = time.time()

        # ---- per-run host uploads (data only; programs are fixed) ----
        pcm_t = torch.zeros(self.CAP_N); pcm_t[:n] = torch.from_numpy(pcm)
        self._write(self.A("pcm"), pcm_t)
        for name in ("conv1_out", "conv1_sq", "red_arena", "conv2_out", "rstd64"):
            self._zero(*self.buf[name])
        invcnt = torch.zeros(V); invcnt[0] = 1.0 / (T1 * self.C)
        self._write(self.A("v_invcnt"), invcnt)
        bias = torch.zeros(T3p, T3p); bias[:, T3:] = float("-inf")
        self._write(self.A("enc_bias"), bias)
        row = torch.zeros(T3p); row[T3:] = float("-inf")
        self._write(self.A("dec_bias_row"), row)

        ref_dump = {}
        if debug_snr and self.ref is not None:
            enc_ref = self.ref.encode(torch.from_numpy(pcm)[None], dump=ref_dump)
            caches = ([dict() for _ in range(self.N_DEC)], [dict() for _ in range(self.N_DEC)])
            ref_dump["logits_step0"] = self.ref.decode_step(torch.tensor([[self.BOS]]), enc_ref, 0, *caches)[0, -1]
        snr = (lambda key, got: self._report(f"  SNR {key}: {snr_db(ref_dump[key][0], got):.1f} dB")) if ref_dump else (lambda *a: None)

        # ---- stem + encoder + cross K/V: one program (workers must be launched first, every run) ----
        if self.NE > 1:
            if not getattr(self, "_flags_cleared", False):
                self.sched.preclear_flags()      # recovers engines a killed run left spinning; clears stale flags
                self._flags_cleared = True
            self.sched.start_workers(self.worker_progs)
        self._exec("encoder", timeout=60.0)
        snr("conv_stem", self._read(self.A("conv3_out"), (T3p, self.CP))[:T3, : self.C])
        x = [self.A("x_a"), self.A("x_b")]
        snr("enc_layer_6", self._read(x[(6 + 1) % 2], (T3p, self.CP))[:T3, : self.C])    # ping-pong survivors
        snr("enc_layer_7", self._read(x[(7 + 1) % 2], (T3p, self.CP))[:T3, : self.C])
        snr("encoder_out", self._read(self.A("h"), (T3p, self.CP))[:T3, : self.C])   # final LN sits in h

        # ---- greedy decode ----
        tok, out = self.BOS, []
        t_dec = time.time()
        dispatch = self.prog["decode_dispatch"][0]
        rt = self._rope_host
        for step in range(min(max_new_tokens, self.MAX_STEPS)):
            # per-step host data: embedding row (device params -> host -> dx_a), rope rows, causal bias row
            row = self.dma_from_accelerator_memory(self.embed_in_addr + tok * self.CP * BPE, torch.Size((self.CP,)))
            self.dma_to_accelerator_memory(self.A("dx_a"), row)
            self.dma_to_accelerator_memory(self.A("rope_cur"),
                                           torch.cat([rt["cos"][step], rt["sin_lo"][step], rt["sin_hi"][step]]))
            bias = torch.full((self.MAX_STEPS_PAD,), -1e36); bias[: step + 1] = 0.0
            self.dma_to_accelerator_memory(self.A("dec_self_bias"), bias.to(torch.bfloat16))
            if self.NE > 1:
                self.sched.start_workers(self.dec_worker_progs,
                                         gpr_sets_by_worker=[self._decode_gpr_sets(step)] * (self.NE - 1))
            t0 = time.time()
            self.start_execute_from_dram(dispatch + step * 4 * 32)
            self.wait_queue(60.0)
            self._hw_time += time.time() - t0
            nxt = self._global_argmax()
            if ref_dump and step == 0:
                lg = self._read(self.A("logits"), (self.VOCAB,))
                self._report(f"  SNR logits_step0: {snr_db(ref_dump['logits_step0'], lg):.1f} dB "
                             f"(hw argmax {nxt}, host argmax {int(lg.argmax())}, ref {int(ref_dump['logits_step0'].argmax())})")
            if nxt == self.EOS:
                break
            out.append(nxt)
            tok = nxt
        audio_s = n / SAMPLE_RATE
        self._report(f"[decode] {len(out)} tokens in {time.time() - t_dec:.2f}s")
        self._report(f"[e2e] {time.time() - t_e2e:.2f}s wall ({self._hw_time:.2f}s on hardware) for "
                     f"{audio_s:.1f}s audio -> RTF {(time.time() - t_e2e) / audio_s:.3f}")
        return out

    @property
    def embed_in_addr(self):
        return self.w["embed_in"]

    # ------------------------------------------------------------------ bins (mobilesam pattern)
    def _signature(self):
        h = hashlib.sha1()
        for f in (__file__, os.path.join(SCRIPT_DIR, "moonshine_config.json")):
            with open(f, "rb") as fh:
                h.update(fh.read())
        wpath = os.path.join(BIN_DIR, "model.safetensors")
        wsig = None
        if os.path.exists(wpath):
            wh = hashlib.sha1()
            with open(wpath, "rb") as fh:
                wh.update(fh.read(4 << 20))
            wsig = f"{os.path.getsize(wpath)}:{wh.hexdigest()}"
        return {"artifact_version": ARTIFACT_VERSION, "code_sha1": h.hexdigest(), "weights": wsig, "engines": self.NE,
                "windows": self.windows,
                "max_seconds": self.max_seconds, "T1": self.T1, "T2": self.T2, "T3P": self.T3P,
                "params_base": self._params_dram_base, "tensor_base": self._tensor_dram_base,
                "program_base": self._program_dram_base}

    def _dma_read_chunks(self, addr, size, f):
        CHUNK = 1 << 20
        off = 0
        while off < size:
            sz = min(CHUNK, size - off)
            buf = bytearray(sz)
            self.dma_read(self.c2h_device, addr + off, buf, sz)
            f.write(buf)
            off += sz

    def dump_bins(self, bin_dir=BIN_DIR):
        os.makedirs(bin_dir, exist_ok=True)
        sig = self._signature()
        total = self.get_params_dram_usage()
        with open(os.path.join(bin_dir, "params.bin"), "wb") as f:
            self._dma_read_chunks(self._params_dram_base, total, f)
        with open(os.path.join(bin_dir, "params.json"), "w") as f:
            json.dump({"signature": sig, "size": total,
                       "weights": {k: v - self._params_dram_base for k, v in self.w.items()}}, f)
        manifest = {"signature": sig, "programs": {}}
        with open(os.path.join(bin_dir, "programs.bin"), "wb") as f:
            for name in self.prog_order:
                addr, size = self.prog[name]
                manifest["programs"][name] = {"offset": f.tell(), "size": size, "addr": addr}
                self._dma_read_chunks(addr, size, f)
            psize = f.tell()
        manifest["order"] = self.prog_order
        # worker programs + the consts their kernels allocated (LN zero rows), ACT pattern
        # worker ISA (encoder program + decode program, contiguous in each worker's program region) + the
        # consts their kernels allocated (LN zero rows), ACT pattern. Entry addresses of both programs kept.
        manifest["workers"] = []
        with open(os.path.join(bin_dir, "workers.bin"), "wb") as f:
            for i, wk in enumerate(self.sched.workers):
                psize, csize = wk.get_program_dram_usage(), wk.get_params_dram_usage()
                entry = {"prog_base": wk._program_dram_base, "prog_offset": f.tell(), "prog_size": psize,
                         "enc_entry": self.worker_progs[i], "dec_entry": self.dec_worker_progs[i]}
                self._dma_read_chunks(wk._program_dram_base, psize, f)
                entry.update({"consts_addr": wk._params_dram_base, "consts_offset": f.tell(), "consts_size": csize})
                self._dma_read_chunks(wk._params_dram_base, csize, f)
                manifest["workers"].append(entry)
        # materialized private shards: each engine's weight arena [weight_base, weight_base+usage)
        manifest["shards"] = []
        with open(os.path.join(bin_dir, "shards.bin"), "wb") as f:
            if self.arena is not None:
                for e, used in enumerate(self.arena.usage()):
                    base = self.arena.region(e).weight_base
                    manifest["shards"].append({"engine": e, "base": base, "offset": f.tell(), "size": used})
                    self._dma_read_chunks(base, used, f)
        with open(os.path.join(bin_dir, "programs.json"), "w") as f:
            json.dump(manifest, f)
        self._report(f"[bins] params {total / 2**20:.1f} MB + programs {psize / 2**20:.1f} MB -> {bin_dir}")

    def bins_match(self, bin_dir=BIN_DIR):
        """True iff a complete frozen image exists whose signature matches this code/config/weights/cap."""
        if any(not os.path.exists(os.path.join(bin_dir, f)) for f in BIN_FILES):
            return False
        sig = self._signature()
        for name in ("params.json", "programs.json"):
            with open(os.path.join(bin_dir, name)) as f:
                got = json.load(f)["signature"]
            bad = [k for k in sig if sig[k] != got.get(k) and not (k == "weights" and sig[k] is None)]
            if bad:
                self._report(f"[bins] image stale on {bad} (image {[got.get(k) for k in bad]} vs now "
                             f"{[sig[k] for k in bad]}); recompiling")
                return False
        return True

    def load_bins(self, bin_dir=BIN_DIR):
        with open(os.path.join(bin_dir, "params.json")) as f:
            pmeta = json.load(f)
        with open(os.path.join(bin_dir, "programs.json")) as f:
            manifest = json.load(f)
        t0 = time.time()
        CHUNK = 1 << 20
        with open(os.path.join(bin_dir, "params.bin"), "rb") as f:
            off = 0
            while off < pmeta["size"]:
                data = f.read(min(CHUNK, pmeta["size"] - off))
                self.dma_write(self.h2c_device, self._params_dram_base + off, data, len(data))
                off += len(data)
        self.allocate_params_dram(pmeta["size"])
        self.w = {k: v + self._params_dram_base for k, v in pmeta["weights"].items()}
        with open(os.path.join(bin_dir, "programs.bin"), "rb") as f:
            blob = f.read()
        end = self._program_dram_base
        for name in manifest["order"]:
            m = manifest["programs"][name]
            data = blob[m["offset"]: m["offset"] + m["size"]]
            self.dma_write(self.h2c_device, m["addr"], data, len(data))     # absolute addrs are baked in
            self.prog[name] = (m["addr"], m["size"])
            self.prog_order.append(name)
            end = max(end, m["addr"] + m["size"])
        self.allocate_program_dram(end - self._program_dram_base)
        if manifest.get("shards"):
            with open(os.path.join(bin_dir, "shards.bin"), "rb") as f:
                sblob = f.read()
            for m in manifest["shards"]:
                self.dma_write(self.h2c_device, m["base"], sblob[m["offset"]: m["offset"] + m["size"]], m["size"])
                self.arena.alloc_weights(m["engine"], m["size"], "restored shards")
            # rebuild the shard address table the argmax readout needs (same deterministic split)
            self._rebuild_shard_table()
            self._park_worker_params()
        self.worker_progs, self.dec_worker_progs = [], []
        if manifest.get("workers"):
            with open(os.path.join(bin_dir, "workers.bin"), "rb") as f:
                wblob = f.read()
            assert len(manifest["workers"]) == len(self.sched.workers), "image built for a different --engines"
            for wk, m in zip(self.sched.workers, manifest["workers"]):
                assert wk.get_program_dram_addr() == m["prog_base"], "worker program base moved"
                self.dma_write(self.h2c_device, m["prog_base"],
                               wblob[m["prog_offset"]: m["prog_offset"] + m["prog_size"]], m["prog_size"])
                wk.allocate_program_dram(m["prog_size"])
                self.dec_worker_progs.append(m["dec_entry"])
                if m["consts_size"]:
                    self.dma_write(self.h2c_device, m["consts_addr"],
                                   wblob[m["consts_offset"]: m["consts_offset"] + m["consts_size"]], m["consts_size"])
                    wk.allocate_params_dram(m["consts_size"])
                self.worker_progs.append(m["enc_entry"])
        self._report(f"[bins] loaded {pmeta['size'] / 2**20:.1f} MB params + {len(blob) / 2**20:.1f} MB programs "
                     f"({len(self.prog)} programs) in {time.time() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser(description="Moonshine-base ASR on the UnifiedEngine FPGA accelerator")
    ap.add_argument("--audio", default=DEFAULT_AUDIO)
    ap.add_argument("--dev", default=CFG["defaults"]["dev"])
    ap.add_argument("--max-new-tokens", type=int, default=CFG["defaults"]["max_new_tokens"])
    ap.add_argument("--max-seconds", type=float, default=CFG["defaults"]["max_audio_seconds"])
    ap.add_argument("--no-snr", action="store_false", dest="debug_snr",
                    help="skip per-stage SNR against the CPU reference (default: on)")
    ap.add_argument("--clean", action="store_true",
                    help="ignore + delete the frozen image: recompile and re-dump (weights are kept)")
    ap.add_argument("--verbose", action="store_true", help="show the cores' compile/tiling prints")
    ap.add_argument("--engines", type=int, default=1, help="engines for the stem/encoder (1 = single-engine)")
    ap.set_defaults(debug_snr=True)
    args = ap.parse_args()
    _SILENT_MODE[0] = not args.verbose

    if args.clean:
        for f in BIN_FILES:
            p = os.path.join(BIN_DIR, f)
            if os.path.exists(p):
                os.remove(p)
    files = ensure_weights()
    tok = load_tokenizer(files["tokenizer.json"])
    pcm = load_audio(args.audio, args.max_seconds)
    ref = MoonshineReference(load_state_dict(files["model.safetensors"]), CFG, device="cpu")

    t0 = time.time()
    ue = MoonshineEngine(ref, dev=args.dev, max_seconds=args.max_seconds, from_bin=not args.clean, engines=args.engines)
    report(f"[init] {'replayed frozen image' if ue.from_bin else 'uploaded weights + compiled'} in {time.time() - t0:.1f}s")
    ids = ue.execute(pcm, args.max_new_tokens, debug_snr=args.debug_snr)
    if not ue.from_bin:
        ue.dump_bins(BIN_DIR)
    report(f"\n>>> {tok.decode(ids, skip_special_tokens=True)}")
    if args.debug_snr:
        ref_ids = ref.generate(torch.from_numpy(pcm)[None], max_new_tokens=args.max_new_tokens)
        report(f"[tokens] match={ids == ref_ids}  hw={len(ids)} ref={len(ref_ids)}")
        if ids != ref_ids:
            report(f"  hw : {tok.decode(ids, skip_special_tokens=True)}\n  ref: {tok.decode(ref_ids, skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
