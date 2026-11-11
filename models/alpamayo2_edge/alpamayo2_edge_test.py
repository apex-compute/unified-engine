#!/usr/bin/env python3
"""Alpamayo 2 Edge on the accelerator (FPGA path), 1..N engines.

Whole model on the device: SigLIP2 vision encoder -> projector -> Cosmos3-Edge LM prefill ->
reasoning decode -> flow-matching action expert. Host side: tokenizer / embedding lookup, token
pick from the logits, Fourier features + Euler update of the action, unicycle integration.

Vision encoder
  * Every image is its own attention sequence (22 images, 2 shapes: 20x36 = 720 patches,
    14x26 = 364). Per image: patch embed + position table -> 27 layers -> post-layernorm ->
    its rows of VIS_OUT, the packed (11568, 1152) tensor in the processor's 2x2 block-major
    order (the projector's 2x2 merge is a reshape of four adjacent rows).
  * Padding obligations (config hw_notes):
      seq       720 -> 768, 364 -> 384 (128-multiple). Pad rows carry copies of real patch rows
                (a zero row makes LayerNorm emit NaN); pad KEY columns are masked with -1e38.
      head_dim  72 -> 128 for Q.K^T: q/k/v are projected at the real 1152 and the matmul's strip
                write-back drops each head's 72 lanes at the start of a 128-lane slot. q carries
                sqrt(128/72) so the core's 1/sqrt(128) score scale is right. P.V runs at the real
                72 straight into the token-major result (no inverse permute, no trim).
      mlp       4304 -> 4352: zero rows in fc1 (+bias), zero columns in fc2.
  * GELU: the hardware activation is x*sigmoid(1.702x); the model uses tanh-GELU. The gate
    therefore compares against the reference run with hw_gelu=True (implementation check) and
    reports the distance to the published model separately.

Multi-engine (--engines, default: every core the bitstream reports)
  Engines meet in master/worker rounds (MultiEngineScheduler release / join, the moonshine / ACT
  recipe); between two rounds the primary may run alone. Nothing an engine reads inside a round
  is written by another engine in that round.
    vision     WHOLE IMAGES per engine: every engine runs the full single-engine pipeline on its
               own images with private scratch; one round for the stage.
    projector  token rows split across engines; one round.
    prefill    two rounds per layer: [row shard: the previous layer's o/MLP tail + this layer's
               norm, q/k/v, RoPE] then [KV-head shard: cache store, Q gather, attention].
    decode,    M = 1 and M = 64 cannot be row-split, so the matmuls split their OUTPUT COLUMNS
    expert     (a column block of B is a contiguous row block: no weight copy) and attention splits
               by KV head; the one-row norms / RoPE / adds run on the primary between rounds.
  The model map sits above the engines' private 512 MiB windows (model_multicore_layout); written
  scratch (vision buffers, attention V^T / scores / scaled Q) lives in each engine's window.

Usage (from models/):
  python alpamayo2_edge/alpamayo2_edge_test.py                                    # everything
  python alpamayo2_edge/alpamayo2_edge_test.py --engines 1                        # single engine
  python alpamayo2_edge/alpamayo2_edge_test.py --host-vision                      # LM + decode + expert
  python alpamayo2_edge/alpamayo2_edge_test.py --host-vision --stop-at prefill    # LM prefill only
  python alpamayo2_edge/alpamayo2_edge_test.py --stop-at vision --images 2        # first 2 images
  python alpamayo2_edge/alpamayo2_edge_test.py --staged vision --image-index 3    # op by op (1 engine)
  python alpamayo2_edge/alpamayo2_edge_test.py --host-vision --staged prefill --stop-after P0.out
"""
import argparse
import builtins
import json
import math
import os
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

_original_print = builtins.print
_SILENT_MODE = False


def _quiet_print(*args, **kwargs):
    if not _SILENT_MODE:
        _original_print(*args, **kwargs)


builtins.print = _quiet_print

import torch
from safetensors.torch import load_file

import user_dma_core
from user_dma_core import DRAM_INSTRUCTION_ADDR, TYPE, UE_MODE, UE_VECTOR_SIZE, UnifiedEngine
from multi_engine_shard import MultiEngineScheduler, PrivateArena, model_multicore_layout
from nn_lib import eltwise_add_core_dram, layer_norm_core_dram_post_add

from alpamayo2_edge_cpu_reference import (  # host oracle + weight loading; no hardware in there
    AIP, EXP, LM, PROJ, STAT_KEYS, VIS, Alpamayo2EdgeRef, KVCache, cos_sim, resolve_model_dir, snr_db, synthetic_inputs,
)

with open(os.path.join(_HERE, "alpamayo2_edge_config.json")) as _f:
    CFG = json.load(_f)

V = CFG["vision"]
VH, VN, VD, VDP = V["hidden_size"], V["num_heads"], V["head_dim"], V["head_dim_padded"]
VI, VIP, VDEPTH, PATCH_DIM = V["intermediate_size"], V["intermediate_size_padded"], V["num_layers"], V["patch_dim"]
C = CFG["connector"]
PJ_IN, PJ_HID, PJ_OUT, MERGE2 = C["input_size"], C["hidden_size"], C["output_size"], C["pixel_shuffle_scale_factor"] ** 2
T, X = CFG["lm"], CFG["action_expert"]
LH, LML, QH, KVH, HD, LI, VOCAB = (T["hidden_size"], T["num_layers"], T["num_heads"], T["num_kv_heads"],
                                   T["head_dim"], T["intermediate_size"], T["vocab_size"])
LQW, KVW = QH * HD, KVH * HD            # 2048 query lanes, 1024 key / value lanes (LM and expert alike)
EH, EI, NWAY = X["hidden_size"], X["intermediate_size"], X["num_action_tokens"]
BPE = 2
SEQ_ALIGN = 128            # flash score rows want a 128-element pitch (LocateAnything MoonViT)
MASK_VALUE = -1e38         # proven flash-bias mask value

# Multi-engine DRAM map: N private 512 MiB windows from 0, then this model's contiguous map.
MODEL_BYTES = 4 << 30              # params | tensors | programs, the single-engine offsets rebased
PRIVATE_TENSOR_BYTES = 128 << 20   # per engine: vision scratch (~43 MB) + attention scratch (~24 MB)
PRIVATE_ISA_BYTES = 256 << 20      # per engine: its share of the unrolled vision program dominates
ZERO_CHUNK = 1 << 26               # elements per zero-fill DMA


def init_hang_prevention(ue):
    """5-step hardware clear. MUST stay identical to run_from_bin.py (see skill gotchas)."""
    halt = bytes(64)
    ue.dma_write(ue.h2c_device, DRAM_INSTRUCTION_ADDR, halt, len(halt))
    print("[Init] HALT written to instruction DRAM base")


def _pad_seq(n):
    return ((n + SEQ_ALIGN - 1) // SEQ_ALIGN) * SEQ_ALIGN


class Eng:
    """Emit context of one engine: the engine object, its index, the registers its current
    round primed (r_M = the row count its row-wise ops run at, r_row = vision q/k/v row pitch)."""

    def __init__(self, ue, e):
        self.ue, self.e, self.r_M, self.r_row = ue, e, None, None


class Alpamayo2Edge_UnifiedEngine(UnifiedEngine):
    def __init__(self, dev=None, engines=1):
        lay = CFG["dram_layout"]
        bases = {k: int(lay[k], 16) for k in ("params_dram_base", "tensor_dram_base", "program_dram_base")}
        self.NE = engines
        windows = None
        if engines > 1:
            windows, model_base = model_multicore_layout(engines, model_bytes=MODEL_BYTES)
            bases = {k: v + model_base for k, v in bases.items()}
        kw = dict(bases)
        if dev:
            kw["device_name"] = dev
        # Built before any weight upload: the constructor's DRAM self-test stomps 16 KB at DRAM start.
        super().__init__(**kw)
        init_hang_prevention(self)
        self.tensor_base, self.tensor_limit = bases["tensor_dram_base"], bases["program_dram_base"]
        self.program_base, self.program_limit = bases["program_dram_base"], bases["params_dram_base"] + MODEL_BYTES
        self.arena = self.sched = None
        if engines > 1:
            # Scheduler (and its worker engines) FIRST, flags cleared FIRST: both can reset an engine,
            # and a reset re-runs the DRAM self-test over whatever sits at DRAM_START_ADDR.
            self.arena = PrivateArena(engines, windows=windows, isa_bytes=PRIVATE_ISA_BYTES,
                                      tensor_bytes=PRIVATE_TENSOR_BYTES)
            self.sched = MultiEngineScheduler(
                self, num_engines=engines, arena=self.arena, allow_unaligned_rows=True,
                handshake="four_phase", region_rendezvous="master_worker", allow_more_than_two_engines=True)
            self.sched.preclear_flags()
            for w in self.sched.workers:        # keep kernel-allocated constants off each window's first 64 KB
                w.allocate_params_dram(64 << 10)    # (DRAM_START_ADDR, the self-test's target, starts a window)
            print(self.arena.describe())
            print(f"[layout] {engines} engines; model map params 0x{bases['params_dram_base']:X} / "
                  f"tensors 0x{self.tensor_base:X} / programs 0x{self.program_base:X}")
        self.eng = [Eng(ue, e) for e, ue in enumerate(self.sched.engines if self.sched else [self])]
        self.buf = {}                                   # shared buffers (model tensor arena)
        self.pbuf = [dict() for _ in range(engines)]    # written scratch, one copy per engine
        self.prog, self.wprog, self.hw_us = {}, {}, {}
        # matmul FLOPs (2*M*K*N, attention's two matmuls included) each program EXECUTES, all engines
        self._flops_acc, self.flops, self.done_flops = 0, {}, {}
        self._attn_parts = ("store", "vt", "attn")      # --profile narrows this to time one part
        self._vattn_parts = ("vt", "attn")              # same, vision attention (--profile-vision)

    # ---------------------------------------------------------------- plumbing

    def _store(self, t):
        t = t.to(torch.bfloat16).contiguous().flatten()
        addr = self.get_params_dram_addr()
        self.dma_to_accelerator_memory(addr, t)
        self.allocate_params_dram(t.numel() * BPE)
        return addr

    def _zero(self, addr, elements):
        for off in range(0, elements, ZERO_CHUNK):
            n = min(ZERO_CHUNK, elements - off)
            self.dma_to_accelerator_memory(addr + off * BPE, torch.zeros(n, dtype=torch.bfloat16))

    def _alloc(self, elements, zero=False):
        addr = self.allocate_tensor_dram(elements * BPE)
        assert self.get_tensor_dram_addr() <= self.tensor_limit, "tensor arena ran into the program arena"
        if zero:
            self._zero(addr, elements)
        return addr

    def _palloc(self, name, elements):
        """Scratch an op WRITES: one zero-filled copy per engine (fresh private DRAM holds NaN
        patterns). Single-engine: the model's own tensor arena."""
        self.private_bytes = getattr(self, "private_bytes", 0) + elements * BPE
        for e in range(self.NE):
            if self.NE == 1:
                addr = self._alloc(elements, zero=True)
            else:
                addr = self.arena.alloc_tensor(e, elements * BPE, name)
                self._zero(addr, elements)
            self.pbuf[e][name] = addr

    def _split(self, total, align=UE_VECTOR_SIZE):
        """[(offset, count)] per engine, in whole `align` blocks; an engine may get 0."""
        assert total % align == 0, f"{total} is not a multiple of {align}"
        base, rem = divmod(total // align, self.NE)
        counts = [(base + (i < rem)) * align for i in range(self.NE)]
        return [(sum(counts[:i]), counts[i]) for i in range(self.NE)]

    def _round(self, body):
        """One master/worker round: body(E) on every engine, fenced on both sides. Every engine
        takes part in every round (one with nothing to do still meets the rendezvous)."""
        if self.NE == 1:
            body(self.eng[0])
            return
        S = self.sched
        S.release()
        body(self.eng[0])
        for e in S.worker_indices():
            S.begin_worker_round(e)
            body(self.eng[e])
            S.end_worker_round(e)
        S.join()

    def _compile(self, name, emit, parallel=False):
        """capture -> HALT -> next program slot. parallel: the workers get a program of their own."""
        global _SILENT_MODE
        t0 = time.time()
        par = parallel and self.NE > 1
        flops0 = self._flops_acc
        _SILENT_MODE = True
        try:
            self.start_capture()
            if par:
                self.sched.begin_program()
            emit()
            if par:
                self.wprog[name] = self.sched.finalize()
            self.stop_capture()
            self.generate_instruction_halt()
            size = self.get_capture_instruction_size_bytes()
            addr = self.get_program_dram_addr()
            self.write_captured_instructions_to_dram(addr)
            self.allocate_program_dram(size)
            self.clear_capture_buffer()
        finally:
            _SILENT_MODE = False
        assert self.get_program_dram_addr() <= self.program_limit, "program arena overflow"
        self.prog[name] = addr
        self.flops[name] = self._flops_acc - flops0
        if size > (1 << 20):
            print(f"[compile {name}] primary {size // 32} instr ({size / 2**20:.1f} MB) @0x{addr:X}, {time.time() - t0:.1f}s")
        return addr

    def _run_transient(self, emit, timeout):
        """Capture one op group into the (unallocated) top of the program arena and run it now."""
        global _SILENT_MODE
        _SILENT_MODE = True
        try:
            self.clear_inst_id()
            self.start_capture()
            emit()
            self.stop_capture()
            self.generate_instruction_halt()
            addr = self.get_program_dram_addr()
            self.write_captured_instructions_to_dram(addr)
            self.clear_capture_buffer()
        finally:
            _SILENT_MODE = False
        return self._exec_addr(addr, timeout)

    def _exec_addr(self, addr, timeout=300.0):
        t0 = time.perf_counter()
        self.start_execute_from_dram(addr)
        self.wait_queue(timeout)
        wall = time.perf_counter() - t0
        us = self.report_latency_in_us()
        wrap = (2 ** 32) * self._clock_period_ns / 1e3      # 32-bit cycle counter wraps (~24 s)
        while us + wrap / 2 < wall * 1e6:
            us += wrap
        return us

    def _exec(self, name, timeout=300.0):
        if name in self.wprog:                  # workers first, every run: they halt with the program
            self.sched.start_workers(self.wprog[name])
        us = self._exec_addr(self.prog[name], timeout)
        self.hw_us[name] = us
        self.done_flops[name] = self.done_flops.get(name, 0) + self.flops.get(name, 0)
        return us

    def _report_programs(self, what, t0):
        used = self.get_program_dram_addr() - self.program_base
        line = f"{what} compiled in {time.time() - t0:.1f}s, primary program DRAM {used / 2**20:.1f} MB"
        if self.NE > 1:
            line += "; workers " + ", ".join(f"{w.get_program_dram_usage() / 2**20:.1f}" for w in self.sched.workers) + " MB"
        print(line)

    # ---------------------------------------------------------------- vision: weights

    def vision_weight_init(self, ref, shapes):
        """Upload the SigLIP2 tower in bf16. `shapes`: unique (grid_h, grid_w) of the input."""
        w = {k: v.float().cpu() for k, v in ref.w.items() if k.startswith(VIS)}
        q_scale = math.sqrt(VDP / VD)
        st = self._store

        def rows(t, n):       # pad a Linear's output rows / a bias
            out = torch.zeros((n,) + tuple(t.shape[1:]))
            out[: t.shape[0]] = t
            return out

        def cols(t, n):       # pad a Linear's input columns
            out = torch.zeros(t.shape[0], n)
            out[:, : t.shape[1]] = t
            return out

        self.vw = {"patch_w": st(w[VIS + "embeddings.patch_embedding.weight"]),
                   "patch_b": st(w[VIS + "embeddings.patch_embedding.bias"]),
                   "post_w": st(w[VIS + "post_layernorm.weight"]),
                   "post_b": st(w[VIS + "post_layernorm.bias"]),
                   "layers": [], "pos": {}}
        for li in range(VDEPTH):
            p = f"{VIS}encoder.layers.{li}."
            a = p + "self_attn."
            self.vw["layers"].append({
                "ln1_w": st(w[p + "layer_norm1.weight"]), "ln1_b": st(w[p + "layer_norm1.bias"]),
                "q": st(w[a + "q_proj.weight"] * q_scale), "q_b": st(w[a + "q_proj.bias"] * q_scale),
                "k": st(w[a + "k_proj.weight"]), "k_b": st(w[a + "k_proj.bias"]),
                "v": st(w[a + "v_proj.weight"]), "v_b": st(w[a + "v_proj.bias"]),
                "o": st(w[a + "out_proj.weight"]), "o_b": st(w[a + "out_proj.bias"]),
                "ln2_w": st(w[p + "layer_norm2.weight"]), "ln2_b": st(w[p + "layer_norm2.bias"]),
                "fc1": st(rows(w[p + "mlp.fc1.weight"], VIP)), "fc1_b": st(rows(w[p + "mlp.fc1.bias"], VIP)),
                "fc2": st(cols(w[p + "mlp.fc2.weight"], VIP)), "fc2_b": st(w[p + "mlp.fc2.bias"]),
            })
        # Learned 16x16 position grid, resized per shape on the host (a constant of the shape).
        for gh, gw in shapes:
            pos = ref.vision_pos_embed(torch.tensor([[1, gh, gw]])).float().cpu()
            self.vw["pos"][(gh, gw)] = st(rows(pos, _pad_seq(gh * gw)))
        print(f"vision weights stored (bf16): params DRAM {self.get_params_dram_usage() / 2**20:.0f} MB")

    # ---------------------------------------------------------------- vision: tensors

    def vision_tensor_init(self, grids):
        """grids: [(grid_h, grid_w)] per image, in prompt order."""
        self.images = []
        pix_off = out_off = 0
        for gh, gw in grids:
            n = gh * gw
            self.images.append(dict(key=(gh, gw), n=n, N=_pad_seq(n), pix_off=pix_off, out_off=out_off))
            pix_off += _pad_seq(n)
            out_off += n
        self.total_real = out_off
        NM = max(im["N"] for im in self.images)
        b = self.buf
        b["IDENTITY"] = self._alloc(UE_VECTOR_SIZE * UE_VECTOR_SIZE)
        self.dma_to_accelerator_memory(b["IDENTITY"], torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16).flatten())
        b["PIX"] = self._alloc(pix_off * PATCH_DIM)
        b["OUT"] = self._alloc(out_off * VH, zero=True)
        for name in ("EMB", "IO_A", "IO_B", "NORM", "O_PROJ", "RESID", "MLP_OUT", "POST"):
            self._palloc(name, NM * VH)
        self._palloc("MLP", NM * VIP)
        # q/k/v token-major [N, VN*128] and head-major [VN, N, 128]. Only the 72 real lanes of a
        # slot are ever written, at shape-independent offsets, so the pad lanes are zeroed once.
        for name in ("Q", "K", "V", "Q_HM", "K_HM", "V_HM"):
            self._palloc(name, NM * VN * VDP)
        self._palloc("ATTN", NM * VN * VD)
        self._palloc("ATTN_SCRATCH", (VDP + NM) * NM + NM * VDP)
        self.mask = {}
        for im in self.images:
            if im["key"] in self.mask:
                continue
            N, n = im["N"], im["n"]
            bias = torch.zeros(N, N, dtype=torch.bfloat16)
            bias[:, n:] = MASK_VALUE
            self.mask[im["key"]] = self._alloc(N * N)
            self.dma_to_accelerator_memory(self.mask[im["key"]], bias.flatten())
        print(f"vision tensors: {(self.get_tensor_dram_addr() - self.tensor_base) / 2**20:.0f} MB shared "
              f"({len(self.images)} images, {self.total_real} patches)")

    def vision_upload_pixels(self, pixel_values):
        """pixel_values (total_real, 768) -> per-image padded slots; pad rows repeat real rows."""
        for im in self.images:
            px = pixel_values[im["out_off"] : im["out_off"] + im["n"]]
            reps = -(-im["N"] // im["n"])
            slot = px.repeat(reps, 1)[: im["N"]].to(torch.bfloat16).contiguous()
            self.dma_to_accelerator_memory(self.buf["PIX"] + im["pix_off"] * PATCH_DIM * BPE, slot.flatten())

    # ---------------------------------------------------------------- vision: op emitters

    def _prime(self, E, N):
        """Runtime registers of one image's ops on engine E."""
        ue = E.ue
        ue.reset_isa_reg_counter()
        E.r_M = ue.alloc_isa_reg()
        ue.generate_instruction_add_set(E.r_M, N)
        E.r_row = ue.alloc_isa_reg()                          # row pitch of the head-padded q/k/v
        ue.generate_instruction_add_set(E.r_row, VN * VDP)

    def _mm(self, E, M, K, N, A, W, OUT, bias, gelu=False, strip=False):
        kw = dict(strip_cols=VD, strip_out_stride=VDP, gpr_out_row_stride_reg=E.r_row) if strip else {}
        self._flops_acc += 2 * M * K * N
        E.ue.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=W, OUTPUT_DRAM_ADDR=OUT,
                             C_DRAM_ADDR=bias, bias_mode="broadcast_N", gelu_enable=gelu,
                             gpr_M_reg=E.r_M, **kw)

    @staticmethod
    def _transpose_10(ue, src, dst, d0, d1, last):
        """[d0, d1, last] -> [d1, d0, last] by strided gather (strided scatter is wrong on device)."""
        for i1 in range(d1):
            ue.accelerator_memory_to_sram(
                accelerator_dram_address=src + i1 * last * BPE, sram_address=0x00000,
                element_size=d0 * last, stride_bytes_per_chunk=last * BPE, stride_jump_bytes=d1 * last * BPE)
            ue.sram_to_accelerator_memory(
                sram_address=0x00000, accelerator_dram_address=dst + i1 * d0 * last * BPE,
                element_size=d0 * last)

    def _attention_head(self, E, n, N, q, k, v, bias, out, out_row_stride):
        """One head: softmax(Q K^T / sqrt(128) + bias) V. Q.K^T contracts over the padded 128,
        P.V writes the real 72 lanes at the caller's row pitch (LocateAnything's split_hd).
        n real query rows against N (128-aligned) keys: pad rows are keys only, and masked."""
        ue, regs = E.ue, []

        def reg(val):
            r = ue.alloc_isa_reg()
            regs.append(r)
            ue.generate_instruction_add_set(r, val)
            return r

        batch_reg, seq_reg, qk_reg, v_reg, stride_reg = reg(n), reg(N), reg(VDP), reg(VD), reg(out_row_stride)
        v_t = self.pbuf[E.e]["ATTN_SCRATCH"]
        self._flops_acc += 2 * n * VDP * N + 2 * n * N * VD
        score = v_t + VDP * N * BPE
        scaled_q = score + N * N * BPE
        if "vt" in self._vattn_parts:
            ue.bf16_transpose_core(M=N, N=VDP, INPUT_DRAM_ADDR=v, OUTPUT_DRAM_ADDR=v_t,
                                   IDENTITY_DRAM_ADDR=self.buf["IDENTITY"], gpr_M_reg=seq_reg)
        if "attn" in self._vattn_parts:
            ue.eltwise_core_dram(M=n, N=VDP, dram_a=q, dram_b=None, dram_out=scaled_q,
                                 mode=UE_MODE.MUL_BROADCAST, scalar=1.0 / math.sqrt(VDP), gpr_M_reg=batch_reg)
            ue.matmat_mul_core(M=n, K=VDP, N=N, A_DRAM_ADDR=scaled_q, B_DRAM_ADDR=k, OUTPUT_DRAM_ADDR=score,
                               softmax_enable=True, C_DRAM_ADDR=bias, bias_mode="full_matrix",
                               gpr_M_reg=batch_reg, gpr_K_reg=qk_reg, gpr_N_reg=seq_reg)
            ue.matmat_mul_core(M=n, K=N, N=VD, A_DRAM_ADDR=score, B_DRAM_ADDR=v_t, OUTPUT_DRAM_ADDR=out,
                               gpr_M_reg=batch_reg, gpr_K_reg=seq_reg, gpr_N_reg=v_reg,
                               gpr_out_row_stride_reg=stride_reg)
        for _ in regs:
            ue.release_isa_reg()

    def _encoder_steps(self, E, key, n, N):
        """The encoder for one shape on engine E's private buffers, as
        [(label, emit, check_addr, check_shape)]. The compiled stream and the staged runner both
        replay this list, so they cannot drift apart.
        Every row-wise op runs on the n real rows only; the pad rows [n, N) exist as attention
        KEYS alone (zero / stale-finite K and V rows behind a -1e38 bias)."""
        b, ue, steps = self.pbuf[E.e], E.ue, []
        head_bytes = N * VDP * BPE

        def add(label, emit, addr=None, shape=None):
            steps.append((label, emit, addr, shape))

        for li in range(VDEPTH):
            la = self.vw["layers"][li]
            h_in, h_out = (b["IO_A"], b["IO_B"]) if li % 2 == 0 else (b["IO_B"], b["IO_A"])
            L = f"L{li}"
            add(f"{L}.norm1", lambda h_in=h_in, la=la: ue.layer_norm_core_dram(
                M=n, N=VH, A_DRAM_ADDR=h_in, OUTPUT_DRAM_ADDR=b["NORM"],
                GAMMA_DRAM_ADDR=la["ln1_w"], BETA_DRAM_ADDR=la["ln1_b"]), b["NORM"], (n, VH))
            for pj in "qkv":
                dst = b[pj.upper()]
                add(f"{L}.{pj}_proj", lambda pj=pj, la=la, dst=dst: self._mm(
                    E, n, VH, VN * VD, b["NORM"], la[pj], dst, la[pj + "_b"], strip=True), dst, (n, VN * VDP))
            for pj in "QKV":
                add(f"{L}.perm_{pj.lower()}", lambda pj=pj: self._transpose_10(
                    ue, b[pj], b[pj + "_HM"], d0=N, d1=VN, last=VDP), b[pj + "_HM"], (VN * N, VDP))
            for h in range(VN):
                last = h == VN - 1
                add(f"{L}.attn_h{h}", lambda h=h: self._attention_head(
                    E, n, N, b["Q_HM"] + h * head_bytes, b["K_HM"] + h * head_bytes, b["V_HM"] + h * head_bytes,
                    self.mask[key], b["ATTN"] + h * VD * BPE, out_row_stride=VN * VD),
                    b["ATTN"] if last else None, (n, VN * VD) if last else None)
            add(f"{L}.o_proj", lambda la=la: self._mm(E, n, VN * VD, VH, b["ATTN"], la["o"], b["O_PROJ"], la["o_b"]),
                b["O_PROJ"], (n, VH))
            add(f"{L}.norm2", lambda h_in=h_in, la=la: layer_norm_core_dram_post_add(
                ue, M=n, N=VH, A_DRAM_ADDR=h_in, B_DRAM_ADDR=b["O_PROJ"],
                ADDOUTPUT_DRAM_ADDR=b["RESID"], NORMOUTPUT_DRAM_ADDR=b["NORM"],
                GAMMA_DRAM_ADDR=la["ln2_w"], BETA_DRAM_ADDR=la["ln2_b"]), b["NORM"], (n, VH))
            add(f"{L}.fc1", lambda la=la: self._mm(E, n, VH, VIP, b["NORM"], la["fc1"], b["MLP"], la["fc1_b"], gelu=True),
                b["MLP"], (n, VIP))
            add(f"{L}.fc2", lambda la=la: self._mm(E, n, VIP, VH, b["MLP"], la["fc2"], b["MLP_OUT"], la["fc2_b"]),
                b["MLP_OUT"], (n, VH))
            add(f"{L}.out", lambda h_out=h_out: eltwise_add_core_dram(
                ue, size=n * VH, A_DRAM_ADDR=b["RESID"], B_DRAM_ADDR=b["MLP_OUT"], OUTPUT_DRAM_ADDR=h_out),
                h_out, (n, VH))
        final = b["IO_A"] if VDEPTH % 2 == 0 else b["IO_B"]
        add("post_norm", lambda: ue.layer_norm_core_dram(
            M=n, N=VH, A_DRAM_ADDR=final, OUTPUT_DRAM_ADDR=b["POST"],
            GAMMA_DRAM_ADDR=self.vw["post_w"], BETA_DRAM_ADDR=self.vw["post_b"]), b["POST"], (n, VH))
        return steps

    def _emit_pre(self, E, im):
        """Patch embed (768 -> 1152, bias) + learned position table -> layer-0 input."""
        b, N = self.pbuf[E.e], im["n"]            # real rows only
        self._prime(E, N)
        self._mm(E, N, PATCH_DIM, VH, self.buf["PIX"] + im["pix_off"] * PATCH_DIM * BPE,
                 self.vw["patch_w"], b["EMB"], self.vw["patch_b"])
        eltwise_add_core_dram(E.ue, size=N * VH, A_DRAM_ADDR=b["EMB"],
                              B_DRAM_ADDR=self.vw["pos"][im["key"]], OUTPUT_DRAM_ADDR=b["IO_A"])

    def _emit_post(self, E, im, rows_per_chunk=128):
        """POST[:n] -> the image's rows of VIS_OUT (whole rows; 1152 lanes = 18 SRAM rows each)."""
        src, dst = self.pbuf[E.e]["POST"], self.buf["OUT"] + im["out_off"] * VH * BPE
        for r0 in range(0, im["n"], rows_per_chunk):
            take = min(rows_per_chunk, im["n"] - r0)
            E.ue.accelerator_memory_to_sram(accelerator_dram_address=src + r0 * VH * BPE,
                                            sram_address=0x00000, element_size=take * VH)
            E.ue.sram_to_accelerator_memory(sram_address=0x00000,
                                            accelerator_dram_address=dst + r0 * VH * BPE, element_size=take * VH)

    @staticmethod
    def _vkind(label):
        tail = label.split(".")[-1]
        if "norm" in tail:
            return "norm"
        if tail.startswith("perm"):
            return "perm"
        if tail.startswith("attn"):
            return "attn"
        return "elt" if tail == "out" else "mm"

    def _emit_image(self, E, i, only=None):
        """only (profiling): emit just the ops of one kind -- "io" | "norm" | "mm" | "perm" | "attn" | "elt"."""
        im = self.images[i]
        if only in (None, "io"):
            self._emit_pre(E, im)
        else:
            self._prime(E, im["n"])
        for label, op, _, _ in self._encoder_steps(E, im["key"], im["n"], im["N"]):
            if only is None or self._vkind(label) == only:
                op()
        if only in (None, "io"):
            self._emit_post(E, im)

    def profile_vision(self):
        """Where the vision stage spends its time: re-run it with only one kind of op emitted (same
        images per engine). Timing only -- it leaves garbage behind, so call it after the real run."""
        full = self.hw_us["vision"]
        variants = [("patch embed + copy out", "io", None), ("layer norms (norm2 adds the residual)", "norm", None),
                    ("matmuls (q/k/v/o/fc1/fc2)", "mm", None), ("q/k/v head permutes", "perm", None),
                    ("attn: V^T transposes", "attn", ("vt",)), ("attn: scale + scores + P.V", "attn", ("attn",)),
                    ("final residual add", "elt", None)]
        print(f"\nprofile: vision = {full / 1e6:.1f} s on {self.NE} engine{'s' if self.NE > 1 else ''} "
              f"(each line re-runs the stage with one kind of op)")
        for label, only, parts in variants:
            self._vattn_parts = parts or ("vt", "attn")
            name = f"prof_vision_{label}"
            self._compile(name, lambda only=only: self._round(
                lambda E: [self._emit_image(E, i, only) for i in self.assign[E.e]]), parallel=True)
            us = self._exec(name, timeout=3600.0)
            print(f"  {label:38s} {us / 1e6:6.2f} s  ({100 * us / (full or 1):4.0f}% of the stage)")
        self._vattn_parts = ("vt", "attn")

    # ---------------------------------------------------------------- vision: compile / run

    def vision_compile(self, image_ids):
        """Whole images per engine, longest first onto the least-loaded engine. Cost ~ N^1.1
        (measured single-engine: 20.0 s at 768 rows, 9.3 s at 384)."""
        t0 = time.time()
        load, self.assign = [0.0] * self.NE, [[] for _ in range(self.NE)]
        for i in sorted(image_ids, key=lambda i: -self.images[i]["N"]):
            e = min(range(self.NE), key=lambda e: load[e])
            self.assign[e].append(i)
            load[e] += self.images[i]["N"] ** 1.1
        self._compile("vision", lambda: self._round(lambda E: [self._emit_image(E, i) for i in self.assign[E.e]]),
                      parallel=True)
        if self.NE > 1:
            print("vision images per engine: " + " | ".join(",".join(map(str, a)) or "-" for a in self.assign))
        self._report_programs("vision", t0)

    def vision_read(self, i):
        im = self.images[i]
        return self.dma_from_accelerator_memory(self.buf["OUT"] + im["out_off"] * VH * BPE, (im["n"], VH)).float()

    # ---------------------------------------------------------------- projector

    def projector_init(self, ref):
        w, st = ref.w, self._store
        self.pw = {"ln_w": st(w[PROJ + "norm.weight"]), "ln_b": st(w[PROJ + "norm.bias"]),
                   "fc1": st(w[PROJ + "linear_fc1.weight"]), "fc1_b": st(w[PROJ + "linear_fc1.bias"]),
                   "fc2": st(w[PROJ + "linear_fc2.weight"]), "fc2_b": st(w[PROJ + "linear_fc2.bias"])}
        tokens = self.total_real // MERGE2
        self.buf["PJ_LN"] = self._alloc(self.total_real * VH)
        self.buf["PJ_HID"] = self._alloc(tokens * PJ_HID)
        self.buf["IMG"] = self._alloc(tokens * PJ_OUT)
        print(f"projector stored: params DRAM {self.get_params_dram_usage() / 2**20:.0f} MB")

    def projector_compile(self, rows):
        """rows: leading patch rows of VIS_OUT to project (a multiple of 4). The 2x2 merge is the
        reinterpretation of (rows, 1152) as (rows / 4, 4608): block-major patch order. Tokens are
        split evenly across engines (every op here is row-wise)."""
        b, pw, tokens = self.buf, self.pw, rows // MERGE2
        base, rem = divmod(tokens, self.NE)
        counts = [base + (i < rem) for i in range(self.NE)]
        shards = [(sum(counts[:i]), counts[i]) for i in range(self.NE)]

        def body(E):
            t0, n = shards[E.e]
            if n == 0:
                return
            ue = E.ue
            self._flops_acc += 2 * n * PJ_IN * PJ_HID + 2 * n * PJ_HID * PJ_OUT
            ue.reset_isa_reg_counter()
            r_tok = ue.alloc_isa_reg()
            ue.generate_instruction_add_set(r_tok, n)
            ue.layer_norm_core_dram(M=n * MERGE2, N=VH, A_DRAM_ADDR=b["OUT"] + t0 * PJ_IN * BPE,
                                    OUTPUT_DRAM_ADDR=b["PJ_LN"] + t0 * PJ_IN * BPE,
                                    GAMMA_DRAM_ADDR=pw["ln_w"], BETA_DRAM_ADDR=pw["ln_b"])
            ue.matmat_mul_core(M=n, K=PJ_IN, N=PJ_HID, A_DRAM_ADDR=b["PJ_LN"] + t0 * PJ_IN * BPE, B_DRAM_ADDR=pw["fc1"],
                               OUTPUT_DRAM_ADDR=b["PJ_HID"] + t0 * PJ_HID * BPE, C_DRAM_ADDR=pw["fc1_b"],
                               bias_mode="broadcast_N", gelu_enable=True, gpr_M_reg=r_tok)
            ue.matmat_mul_core(M=n, K=PJ_HID, N=PJ_OUT, A_DRAM_ADDR=b["PJ_HID"] + t0 * PJ_HID * BPE, B_DRAM_ADDR=pw["fc2"],
                               OUTPUT_DRAM_ADDR=b["IMG"] + t0 * PJ_OUT * BPE, C_DRAM_ADDR=pw["fc2_b"],
                               bias_mode="broadcast_N", gpr_M_reg=r_tok)
        self._compile("projector", lambda: self._round(body), parallel=True)

    def projector_read(self, rows):
        return self.dma_from_accelerator_memory(self.buf["IMG"], (rows // MERGE2, PJ_OUT)).float()

    # ---------------------------------------------------------------- staged runner (1 engine)

    def _staged(self, steps, prime, n_real, n_pad, ref_for, stop_after=None, timeout=300.0):
        """One op per program: NaN-check every op that names an output, SNR-check where
        `ref_for(label)` returns a reference. Pad rows are excluded from every statistic."""
        assert self.NE == 1, "staged runs are single-engine"
        for label, op, addr, shape in steps:
            def emit(op=op):
                prime()
                op()
            self._run_transient(emit, timeout)
            if addr is not None:
                t = self.dma_from_accelerator_memory(addr, shape).float()
                real = t[:n_real] if shape[0] == n_pad else t
                nan, inf = int(torch.isnan(real).sum()), int(torch.isinf(real).sum())
                line = f"  [{label:14s}] {str(tuple(shape)):>14s} nan={nan} inf={inf} absmax={float(real.abs().max()):.4g}"
                ref = ref_for(label)
                if ref is not None:
                    line += f"  snr={snr_db(ref, real):6.1f} dB cos={cos_sim(ref, real):.5f}"
                print(line)
                assert nan == 0 and inf == 0, f"NaN/Inf at '{label}'"
            if label == stop_after:
                print(f"stopped after {label}")
                return False
        return True

    def vision_run_staged(self, i, ref_embed, ref_layers, ref_out, stop_after=None):
        """One image, op by op. ref_*: the host reference with hw_gelu=True for this image."""
        im, E = self.images[i], self.eng[0]
        N, n = im["N"], im["n"]
        self._run_transient(lambda: self._emit_pre(E, im), 120.0)
        emb = self.dma_from_accelerator_memory(self.pbuf[0]["IO_A"], (n, VH)).float()
        print(f"  [embed] snr={snr_db(ref_embed, emb):.1f} dB cos={cos_sim(ref_embed, emb):.5f}")

        def ref_for(label):
            if label.endswith(".out"):
                return ref_layers[int(label[1:].split(".")[0])]
            return ref_out if label == "post_norm" else None

        if self._staged(self._encoder_steps(E, im["key"], n, N), lambda: self._prime(E, n), n, N, ref_for, stop_after):
            print("staged run complete")

    # ================================================================ LM + action expert
    #
    # KV cache: [layer][kv_head][KV_CAP][128], keys already rotated. Slot map (KV_CAP = S + 128):
    #   [0, P)            prompt (prefill; P = prompt length - 1)
    #   [P, S)            decode tokens, appended one per step (S = P + 1 + max_new, 128-aligned)
    #   [S, S + 64)       the 64 action tokens of the current denoise step (never kept)
    #   KV_CAP - 1        the token being decoded: its K/V are written to this FIXED slot so the
    #                     decode program has no position-dependent address; a tiny per-position
    #                     program copies it to its real slot after the step.
    # Attention always spans a fixed, 128-aligned KV length and validity is a host-written bias
    # (-1e38 on disabled slots), so slot order never matters (keys carry their RoPE already).
    # One op table serves all three modes: prefill (M = S rows, IF4, row shards), decode (M = 1,
    # IF4, column shards), expert (M = 64, hidden 512, bf16, column shards).

    def _qstore(self, pack):
        """(data_bytes, scale_bytes, N, K) IF4 pack -> (data_addr, scale_addr)."""
        data, scale = pack[0], pack[1]
        s_addr = self.get_params_dram_addr()
        self.dma_write(self.h2c_device, s_addr, scale, len(scale))
        self.allocate_params_dram(len(scale))
        d_addr = self.get_params_dram_addr()
        self.dma_write(self.h2c_device, d_addr, torch.frombuffer(bytearray(data), dtype=torch.uint8), len(data))
        self.allocate_params_dram(len(data))
        return (d_addr, s_addr)

    def lm_weight_init(self, ref, packs):
        w, st, q = ref.w, self._store, self._qstore
        self.lw = {"norm": st(w[LM + "norm.weight"]), "head": q(packs["head"]), "layers": []}
        for li in range(LML):
            p = f"{LM}layers.{li}."
            self.lw["layers"].append({
                "ln1": st(w[p + "input_layernorm.weight"]), "ln2": st(w[p + "post_attention_layernorm.weight"]),
                "q": q(packs[f"{li}.q"]), "k": q(packs[f"{li}.k"]), "v": q(packs[f"{li}.v"]), "o": q(packs[f"{li}.o"]),
                "fc1": [q(packs[f"{li}.fc1.{i}"]) for i in range(2)],
                "fc2": [q(packs[f"{li}.fc2.{i}"]) for i in range(2)],
            })
        print(f"LM weights stored (IF4): params DRAM {self.get_params_dram_usage() / 2**20:.0f} MB")

    def expert_weight_init(self, ref):
        w, st = ref.w, self._store

        def pad(t, shape):
            out = torch.zeros(shape)
            out[tuple(slice(0, s) for s in t.shape)] = t.float().cpu()
            return out

        self.ew = {"norm": st(w[EXP + "norm.weight"]),
                   "out": st(pad(w["expert.action_out_proj.weight"], (UE_VECTOR_SIZE, EH))),
                   "out_b": st(pad(w["expert.action_out_proj.bias"], (UE_VECTOR_SIZE,))),
                   "t0": st(pad(w[AIP + "encoder.trunk.0.weight"], (EH, UE_VECTOR_SIZE))),
                   "t0_b": st(w[AIP + "encoder.trunk.0.bias"]), "t2": st(w[AIP + "encoder.trunk.2.weight"]),
                   "t3": st(w[AIP + "encoder.trunk.3.weight"]), "t3_b": st(w[AIP + "encoder.trunk.3.bias"]),
                   "t5": st(w[AIP + "encoder.trunk.5.weight"]),
                   "t6": st(w[AIP + "encoder.trunk.6.weight"]), "t6_b": st(w[AIP + "encoder.trunk.6.bias"]),
                   "ln_w": st(w[AIP + "norm.weight"]), "ln_b": st(w[AIP + "norm.bias"]), "layers": []}
        for li in range(LML):
            p = f"{EXP}layers.{li}."
            a = p + "self_attn."
            self.ew["layers"].append({
                "ln1": st(w[p + "input_layernorm.weight"]), "ln2": st(w[p + "post_attention_layernorm.weight"]),
                "q": st(w[a + "q_proj.weight"]), "k": st(w[a + "k_proj.weight"]), "v": st(w[a + "v_proj.weight"]),
                "o": st(w[a + "o_proj.weight"]), "fc1": [st(w[p + "mlp.fc1.weight"])], "fc2": [st(w[p + "mlp.fc2.weight"])],
            })
        end = self.get_params_dram_addr()
        assert end <= self.tensor_base, f"params 0x{end:X} ran into the tensor arena"
        print(f"expert weights stored (bf16): params DRAM {self.get_params_dram_usage() / 2**20:.0f} MB")

    def lm_tensor_init(self, P, max_new):
        self.P = P
        S = self.S = _pad_seq(P + 1 + max_new)
        CAP = self.KV_CAP = S + 128
        self.E0, self.CUR = S, CAP - 1
        b = self.buf
        for name in ("X_A", "X_B", "LNORM", "LO", "RES", "MA", "MB", "LQ", "ATT"):
            b[name] = self._alloc(S * LH)
        b["LK"], b["LV"] = self._alloc(S * KVW), self._alloc(S * KVW)
        b["Q_HM"] = self._alloc(S * LH)                 # head-major; each head's block has one writer
        b["HA"], b["HB"] = self._alloc(S * LI // 2), self._alloc(S * LI // 2)
        b["MASK_PRE"] = self._alloc(S * S)
        b["MASK_DEC"] = self._alloc(CAP)
        b["MASK_EXP"] = self._alloc(NWAY * CAP)
        b["ROPE_PRE"], b["ROPE_CUR"], b["ROPE_EXP"] = self._alloc(S * 2 * HD), self._alloc(2 * HD), self._alloc(NWAY * 2 * HD)
        b["HID"], b["LOGITS"] = self._alloc(LH), self._alloc(VOCAB)
        b["FEAT"], b["E1"], b["E2"], b["VOUT"] = (self._alloc(NWAY * UE_VECTOR_SIZE), self._alloc(NWAY * EH),
                                                  self._alloc(NWAY * EH), self._alloc(NWAY * UE_VECTOR_SIZE))
        # attention scratch is written by every engine at once: V^T, scores / probabilities, scaled Q
        self._palloc("V_T", HD * CAP)
        self._palloc("SCORE", S * S)
        self._palloc("SQ", S * HD)
        # Zeroed: slots no stage ever writes are masked, but still flow through V^T and P.V.
        kv = LML * KVH * CAP * HD
        b["KC"], b["VC"] = self._alloc(kv, zero=True), self._alloc(kv, zero=True)
        # Persistent V^T [layer][kv_head][128][KV_CAP] for decode / expert: built once after prefill,
        # then only the 64-slot block a step wrote is refreshed (a full re-transpose was 62% of a token).
        b["VT"] = self._alloc(kv)
        b["EYE"] = self._alloc(HD * HD)
        self.dma_to_accelerator_memory(b["EYE"], torch.eye(HD, dtype=torch.bfloat16).flatten())
        mask = torch.full((S, S), MASK_VALUE).triu(1).to(torch.bfloat16)      # causal
        self.dma_to_accelerator_memory(b["MASK_PRE"], mask.flatten())
        line = f"LM tensors: S={S} KV_CAP={CAP}, shared tensor arena {(self.get_tensor_dram_addr() - self.tensor_base) / 2**20:.0f} MB"
        if self.NE > 1:
            line += f", private {self.private_bytes / 2**20:.0f} MB / engine"
        print(line)

    def _kv(self, name, layer, kvh, slot=0):
        return self.buf[name] + ((layer * KVH + kvh) * self.KV_CAP + slot) * HD * BPE

    def _vt(self, layer, kvh):
        return self.buf["VT"] + (layer * KVH + kvh) * HD * self.KV_CAP * BPE

    def _vt_refresh(self, E, layer, kvh, slot):
        """V^T columns of the 64-slot block holding `slot` <- that block of V: I(128) . V_blk^T
        written as 128 rows of 64 at the V^T row pitch (the output-stride path column shards use)."""
        ue, blk = E.ue, slot // UE_VECTOR_SIZE * UE_VECTOR_SIZE
        regs = [ue.alloc_isa_reg() for _ in range(4)]
        for r, val in zip(regs, (HD, HD, UE_VECTOR_SIZE, self.KV_CAP)):
            ue.generate_instruction_add_set(r, val)
        ue.matmat_mul_core(M=HD, K=HD, N=UE_VECTOR_SIZE, A_DRAM_ADDR=self.buf["EYE"],
                           B_DRAM_ADDR=self._kv("VC", layer, kvh, blk), OUTPUT_DRAM_ADDR=self._vt(layer, kvh) + blk * BPE,
                           gpr_M_reg=regs[0], gpr_K_reg=regs[1], gpr_N_reg=regs[2], gpr_out_row_stride_reg=regs[3])
        for _ in regs:
            ue.release_isa_reg()

    def lm_write_rope(self, name, cos, sin):
        """cos / sin (rows, 128) from the reference -> engine rows [cos(128) | -sin(64) | sin(64)]."""
        cos, sin = cos.float().cpu(), sin.float().cpu()
        half = HD // 2
        table = torch.cat([cos, -sin[:, :half], sin[:, half:]], -1).to(torch.bfloat16)
        self.dma_to_accelerator_memory(self.buf[name], table.flatten())

    def lm_write_mask(self, name, rows, valid):
        """Bias rows over the KV slots: 0 on `valid` (bool, KV_CAP), -1e38 elsewhere."""
        row = torch.where(valid, torch.tensor(0.0), torch.tensor(MASK_VALUE)).to(torch.bfloat16)
        self.dma_to_accelerator_memory(self.buf[name], row.repeat(rows))

    # ---------------------------------------------------------------- LM: op emitters

    @staticmethod
    def _prime_rows(E, M):
        """Start of a round body / a solo run: r_M = the rows this engine's row-wise ops cover."""
        E.ue.reset_isa_reg_counter()
        E.r_M = E.ue.alloc_isa_reg()
        E.ue.generate_instruction_add_set(E.r_M, M)

    @staticmethod
    def _wshard(W, K, c0):
        """Weight rows [c0, ...) = output columns [c0, ...): a contiguous row block of B (N x K)."""
        if isinstance(W, tuple):                        # IF4: two nibbles per element, one scale per 64
            return (W[0] + (c0 * K) // 2, W[1] + ((c0 * K) // UE_VECTOR_SIZE) * BPE)
        return W + c0 * K * BPE

    def _mmw(self, E, M, K, N, A, W, OUT, bias=None, out_stride=None, **act):
        """Matmul against an IF4 pack (data_addr, scale_addr) or a bf16 weight address.
        out_stride: full output row width when this call writes a column block of wider rows."""
        ue = E.ue
        kw = dict(act)
        r = None
        self._flops_acc += 2 * M * K * N
        if out_stride is not None:
            r = ue.alloc_isa_reg()
            ue.generate_instruction_add_set(r, out_stride)
            kw["gpr_out_row_stride_reg"] = r
        if isinstance(W, tuple) and M == 1 and bias is None:
            # Decode: the fused 1-pass streaming quantized dot (HW-validated at M == 1; the kernel
            # qwen3 / gemma3 / moonshine decode with). matmat_mul_core's IF4 path is 2-pass
            # (dequantize, then multiply), which doubles the work on a weight-streaming-bound row.
            ue.quantized_matmat_core(M=1, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=W[0], OUTPUT_DRAM_ADDR=OUT,
                                     SCALE_DRAM_ADDR=W[1], data_type=TYPE.IF4, **kw)
        elif isinstance(W, tuple):
            ue.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=W[0], OUTPUT_DRAM_ADDR=OUT,
                               C_DRAM_ADDR=bias, bias_mode="broadcast_N", is_B_quantized=True,
                               data_type=TYPE.IF4, SCALE_DRAM_ADDR=W[1], gpr_M_reg=E.r_M, **kw)
        else:
            ue.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=W, OUTPUT_DRAM_ADDR=OUT,
                               C_DRAM_ADDR=bias, bias_mode="broadcast_N", gpr_M_reg=E.r_M, **kw)
        if r is not None:
            ue.release_isa_reg()

    @staticmethod
    def _gather(ue, src, col, ncols, rows, dst, rows_per_chunk=1024):
        """Column `col` (128 lanes) of a token-major [rows, ncols*128] buffer -> contiguous [rows, 128]."""
        row_bytes = HD * BPE
        for r0 in range(0, rows, rows_per_chunk):
            take = min(rows_per_chunk, rows - r0)
            stride = dict(stride_bytes_per_chunk=row_bytes, stride_jump_bytes=ncols * row_bytes) if take > 1 else {}
            ue.accelerator_memory_to_sram(accelerator_dram_address=src + (r0 * ncols + col) * row_bytes,
                                          sram_address=0x00000, element_size=take * HD, **stride)
            ue.sram_to_accelerator_memory(sram_address=0x00000, accelerator_dram_address=dst + r0 * row_bytes,
                                          element_size=take * HD)

    def _lm_attention(self, E, M, kv_len, q, k, v_t, mask, out, parts=("attn",)):
        """One query head: softmax(Q K^T / sqrt(128) + mask) V over kv_len slots, written into its
        128 columns of the token-major [M, 2048] result. v_t: this KV head's V^T [128, kv_len]."""
        ue, p, regs = E.ue, self.pbuf[E.e], []

        def reg(val):
            r = ue.alloc_isa_reg()
            regs.append(r)
            ue.generate_instruction_add_set(r, val)
            return r

        batch_reg, seq_reg, hd_reg, stride_reg = reg(M), reg(kv_len), reg(HD), reg(LQW)
        if "attn" in parts:
            self._flops_acc += 4 * M * HD * kv_len
        if "attn" in parts:
            ue.eltwise_core_dram(M=M, N=HD, dram_a=q, dram_b=None, dram_out=p["SQ"], mode=UE_MODE.MUL_BROADCAST,
                                 scalar=1.0 / math.sqrt(HD), gpr_M_reg=batch_reg)
            ue.matmat_mul_core(M=M, K=HD, N=kv_len, A_DRAM_ADDR=p["SQ"], B_DRAM_ADDR=k, OUTPUT_DRAM_ADDR=p["SCORE"],
                               softmax_enable=True, C_DRAM_ADDR=mask, bias_mode="full_matrix",
                               gpr_M_reg=batch_reg, gpr_K_reg=hd_reg, gpr_N_reg=seq_reg)
            ue.matmat_mul_core(M=M, K=kv_len, N=HD, A_DRAM_ADDR=p["SCORE"], B_DRAM_ADDR=v_t, OUTPUT_DRAM_ADDR=out,
                               gpr_M_reg=batch_reg, gpr_K_reg=seq_reg, gpr_N_reg=hd_reg,
                               gpr_out_row_stride_reg=stride_reg)
        for _ in regs:
            ue.release_isa_reg()

    def _layer_ops(self, tag, layer, M, H, wl, x_in, x_out, rope, slot0, kv_len, mask):
        """One decoder layer (LM or expert) as an ordered op table:
            (label, kind, fn(E, r0, n, c0, cn), check_addr, check_shape)
        kind ("row",)    row-wise op on rows [r0, r0 + n)
             ("mm", N)   matmul on rows [r0, r0 + n) producing output columns [c0, c0 + cn) of N
             ("heads",)  attention for the KV heads engine E owns (reads every row)
        Consecutive "mm" ops never read each other's output, so a column round may hold a run of
        them. This call's K/V rows go to KV slots [slot0, slot0 + M); attention spans kv_len."""
        b, ops = self.buf, []
        L = f"{tag}{layer}"
        parts = len(wl["fc1"])
        I = (LI if H == LH else EI) // parts
        hbuf, mbuf = (b["HA"], b["HB"]), (b["MA"], b["MB"])
        group = QH // KVH

        def row(label, fn, addr, shape):
            ops.append((f"{L}.{label}", ("row",), lambda E, r0, n, c0, cn: fn(E, r0, n), addr, shape))

        def mm(label, K, N, A, W, OUT, **act):
            def fn(E, r0, n, c0, cn):
                self._mmw(E, n, K, cn, A + r0 * K * BPE, self._wshard(W, K, c0), OUT + (r0 * N + c0) * BPE,
                          out_stride=N if (cn < N and n > 1) else None, **act)
            ops.append((f"{L}.{label}", ("mm", N), fn, OUT, (M, N)))

        def norm(label, src, dst, gamma):
            row(label, lambda E, r0, n: E.ue.rms_norm_core_dram(
                M=n, N=H, A_DRAM_ADDR=src + r0 * H * BPE, OUTPUT_DRAM_ADDR=dst + r0 * H * BPE,
                GAMMA_DRAM_ADDR=gamma, gpr_M_reg=E.r_M), dst, (M, H))

        def eltwise(label, a, c, out, mode, W):
            row(label, lambda E, r0, n: E.ue.eltwise_core_dram(
                M=n, N=W, dram_a=a + r0 * W * BPE, dram_b=c + r0 * W * BPE, dram_out=out + r0 * W * BPE,
                mode=mode, gpr_M_reg=E.r_M), out, (M, W))

        def rope_op(label, name, heads):
            row(label, lambda E, r0, n: E.ue.rope_hf_core_dram_gqa(
                M=n, group_size=heads, N=HD, input_dram_addr=b[name] + r0 * heads * HD * BPE,
                output_dram_addr=b[name] + r0 * heads * HD * BPE, cos_dram_addr=rope + r0 * 2 * HD * BPE,
                sin_dram_addr=rope + r0 * 2 * HD * BPE + HD * BPE, gpr_M_reg=E.r_M), b[name], (M, heads * HD))

        def attention(E, r0, n, c0, cn):
            ue = E.ue
            for kvh in range(KVH):
                if kvh % self.NE != E.e:
                    continue
                parts = self._attn_parts
                if "store" in parts:
                    self._gather(ue, b["LK"], kvh, KVH, M, self._kv("KC", layer, kvh, slot0))
                    self._gather(ue, b["LV"], kvh, KVH, M, self._kv("VC", layer, kvh, slot0))
                persistent = kv_len == self.KV_CAP          # decode / expert; prefill transposes its S slots
                v_t = self._vt(layer, kvh) if persistent else self.pbuf[E.e]["V_T"]
                if "vt" in parts and persistent:
                    self._vt_refresh(E, layer, kvh, slot0)
                elif "vt" in parts:
                    r = ue.alloc_isa_reg()
                    ue.generate_instruction_add_set(r, kv_len)
                    ue.bf16_transpose_core(M=kv_len, N=HD, INPUT_DRAM_ADDR=self._kv("VC", layer, kvh), OUTPUT_DRAM_ADDR=v_t,
                                           IDENTITY_DRAM_ADDR=b["IDENTITY"], gpr_M_reg=r)
                    ue.release_isa_reg()
                for g in range(group):
                    h = kvh * group + g
                    q = b["Q_HM"] + h * M * HD * BPE
                    if "store" in parts:
                        self._gather(ue, b["LQ"], h, QH, M, q)
                    self._lm_attention(E, M, kv_len, q, self._kv("KC", layer, kvh), v_t,
                                       mask, b["ATT"] + h * HD * BPE, parts=parts)

        norm("norm1", x_in, b["LNORM"], wl["ln1"])
        mm("q", H, LQW, b["LNORM"], wl["q"], b["LQ"])
        mm("k", H, KVW, b["LNORM"], wl["k"], b["LK"])
        mm("v", H, KVW, b["LNORM"], wl["v"], b["LV"])
        rope_op("rope_q", "LQ", QH)
        rope_op("rope_k", "LK", KVH)
        ops.append((f"{L}.attn", ("heads",), attention, b["ATT"], (M, LQW)))
        mm("o", LQW, H, b["ATT"], wl["o"], b["LO"])
        eltwise("res", x_in, b["LO"], b["RES"], UE_MODE.ELTWISE_ADD, H)
        norm("norm2", b["RES"], b["LNORM"], wl["ln2"])
        for i in range(parts):        # relu^2 MLP; the LM's 9216 runs as two 4608 halves (IF4 K <= 8192)
            mm(f"fc1_{i}", H, I, b["LNORM"], wl["fc1"][i], hbuf[i], clamp_enable=True, clamp_min=0.0, clamp_max=float("inf"))
        for i in range(parts):
            eltwise(f"sq_{i}", hbuf[i], hbuf[i], hbuf[i], UE_MODE.ELTWISE_MUL, I)
        for i in range(parts):
            mm(f"fc2_{i}", I, H, hbuf[i], wl["fc2"][i], mbuf[i])
        if parts == 2:
            eltwise("fc2_sum", b["MA"], b["MB"], b["MA"], UE_MODE.ELTWISE_ADD, H)
        eltwise("out", b["RES"], b["MA"], x_out, UE_MODE.ELTWISE_ADD, H)
        return ops

    def _stack_ops(self, tag, M, H, weights, rope, slot0, kv_len, mask):
        """All 28 layers, x ping-ponging X_A -> X_B -> ...; returns (ops, final x address)."""
        b, ops, x = self.buf, [], self.buf["X_A"]
        for li in range(LML):
            nxt = b["X_B"] if x == b["X_A"] else b["X_A"]
            ops += self._layer_ops(tag, li, M, H, weights[li], x, nxt, rope, slot0, kv_len, mask)
            x = nxt
        return ops, x

    def _emit_ops(self, ops, M, mode, only=None):
        """Schedule an op table onto the engines.
        mode "rows": every run of row / mm ops is ONE row-sharded round (an engine's rows never
                     leave its lane, so a layer's tail and the next layer's head share a round).
        mode "cols": every run of mm ops is one column-sharded round; row ops between them are
                     row-sharded (M / engines rows each) when M divides, else run on the primary.
                     "solo" ops (whole-buffer emitters) always run on the primary.
        Attention is always its own round, split by KV head.
        only (profiling): emit the full round structure but only the ops of that kind --
        "row" | "mm" | "heads" | "none" (rendezvous alone)."""
        keep = lambda kind: only is None or only == kind
        i = 0
        while i < len(ops):
            kind = ops[i][1][0]
            j = i + 1
            if kind == "heads":
                def body(E, op=ops[i]):
                    self._prime_rows(E, M)
                    if keep("heads"):
                        op[2](E, 0, M, None, None)
                self._round(body)
            elif mode == "rows":
                while j < len(ops) and ops[j][1][0] != "heads":
                    j += 1
                shards = self._split(M, 1)      # balanced; a row count need not be a 64-multiple

                def body(E, run=ops[i:j], shards=shards):
                    r0, n = shards[E.e]
                    if n == 0:
                        return
                    self._prime_rows(E, n)
                    for _, k, fn, _, _ in run:
                        if keep(k[0]):
                            fn(E, r0, n, 0, k[1] if k[0] == "mm" else None)
                self._round(body)
            elif kind == "row" and self.NE > 1 and M > 1 and M % self.NE == 0:
                while j < len(ops) and ops[j][1][0] == "row":       # expert: M / NE rows per engine
                    j += 1

                def body(E, run=ops[i:j], n=M // self.NE):
                    self._prime_rows(E, n)
                    for _, _, fn, _, _ in run:
                        if keep("row"):
                            fn(E, E.e * n, n, None, None)
                self._round(body)
            elif kind in ("row", "solo"):                           # primary alone, every row
                while j < len(ops) and ops[j][1][0] == kind:
                    j += 1
                E0 = self.eng[0]
                self._prime_rows(E0, M)
                for _, _, fn, _, _ in ops[i:j]:
                    if keep("row"):
                        fn(E0, 0, M, None, None)
            else:
                while j < len(ops) and ops[j][1][0] == "mm":
                    j += 1

                def body(E, run=ops[i:j]):
                    self._prime_rows(E, M)
                    for _, k, fn, _, _ in run:
                        c0, cn = self._split(k[1])[E.e]
                        if cn and keep("mm"):
                            fn(E, 0, M, c0, cn)
                self._round(body)
            i = j

    def _ops_as_steps(self, ops, M):
        """The op table as single-engine staged steps (full rows, full columns)."""
        E0 = self.eng[0]
        return [(label, lambda fn=fn, k=k: fn(E0, 0, M, 0, k[1] if k[0] == "mm" else None), addr, shape)
                for label, k, fn, addr, shape in ops]

    # ---------------------------------------------------------------- LM: compile

    def lm_compile(self, image_runs, max_new):
        """image_runs: [(prompt row, first image-embed row, count)] of the image placeholders."""
        b, S, CAP = self.buf, self.S, self.KV_CAP
        t0 = time.time()

        def scatter():        # projector output -> the image rows of the prompt embedding
            for row, src, n in image_runs:
                for r0 in range(0, n, 64):
                    take = min(64, n - r0)
                    self.accelerator_memory_to_sram(accelerator_dram_address=b["IMG"] + (src + r0) * LH * BPE,
                                                    sram_address=0x00000, element_size=take * LH)
                    self.sram_to_accelerator_memory(sram_address=0x00000,
                                                    accelerator_dram_address=b["X_A"] + (row + r0) * LH * BPE,
                                                    element_size=take * LH)
        self._compile("scatter_img", scatter)

        # Real prompt rows only: the pad rows [P, S) stay in the KV length as masked keys (slots the
        # decode tokens later fill), but nothing is computed for them.
        P = self.P
        ops, self.prefill_out = self._stack_ops("P", P, LH, self.lw["layers"], b["ROPE_PRE"], 0, S, b["MASK_PRE"])
        self._compile("prefill", lambda ops=ops: self._emit_ops(ops, P, "rows"), parallel=True)

        def build_vt():       # every KV head's whole V -> the persistent V^T, once per inference
            def body(E):
                self._prime_rows(E, 1)
                r = E.ue.alloc_isa_reg()
                E.ue.generate_instruction_add_set(r, CAP)
                for layer in range(LML):
                    for kvh in range(KVH):
                        if kvh % self.NE == E.e:
                            E.ue.bf16_transpose_core(M=CAP, N=HD, INPUT_DRAM_ADDR=self._kv("VC", layer, kvh),
                                                     OUTPUT_DRAM_ADDR=self._vt(layer, kvh),
                                                     IDENTITY_DRAM_ADDR=b["IDENTITY"], gpr_M_reg=r)
            self._round(body)
        self._compile("build_vt", build_vt, parallel=True)

        ops, x = self._stack_ops("D", 1, LH, self.lw["layers"], b["ROPE_CUR"], self.CUR, CAP, b["MASK_DEC"])
        ops.append(("D.norm", ("row",), lambda E, r0, n, c0, cn, x=x: E.ue.rms_norm_core_dram(
            M=1, N=LH, A_DRAM_ADDR=x, OUTPUT_DRAM_ADDR=b["HID"], GAMMA_DRAM_ADDR=self.lw["norm"], gpr_M_reg=E.r_M),
            b["HID"], (1, LH)))
        ops.append(("D.head", ("mm", VOCAB), lambda E, r0, n, c0, cn: self._mmw(
            E, 1, LH, cn, b["HID"], self._wshard(self.lw["head"], LH, c0), b["LOGITS"] + c0 * BPE), b["LOGITS"], (1, VOCAB)))
        self._compile("decode", lambda ops=ops: self._emit_ops(ops, 1, "cols"), parallel=True)
        self.stage_ops = {"decode": (ops, 1)}

        for slot in range(self.P, self.P + 1 + max_new):      # CUR slot -> the token's real slot, by KV head
            def append(slot=slot):
                def body(E):
                    self._prime_rows(E, 1)
                    for layer in range(LML):
                        for kvh in range(KVH):
                            if kvh % self.NE != E.e:
                                continue
                            for name in ("KC", "VC"):
                                E.ue.accelerator_memory_to_sram(accelerator_dram_address=self._kv(name, layer, kvh, self.CUR),
                                                                sram_address=0x00000, element_size=HD)
                                E.ue.sram_to_accelerator_memory(sram_address=0x00000,
                                                                accelerator_dram_address=self._kv(name, layer, kvh, slot),
                                                                element_size=HD)
                            self._vt_refresh(E, layer, kvh, slot)
                self._round(body)
            self._compile(f"append_{slot}", append, parallel=True)

        # action expert: action_in_proj trunk (primary) -> 28 expert layers -> norm -> velocity head
        e, n = self.ew, NWAY
        solo = lambda label, fn, addr, shape: (label, ("solo",), lambda E, r0, nn, c0, cn: fn(E), addr, shape)
        rms = lambda E, src, dst, gamma: E.ue.rms_norm_core_dram(M=n, N=EH, A_DRAM_ADDR=src, OUTPUT_DRAM_ADDR=dst,
                                                                GAMMA_DRAM_ADDR=gamma, gpr_M_reg=E.r_M)
        V64 = UE_VECTOR_SIZE
        ops = [
            solo("A.t0", lambda E: self._mmw(E, n, V64, EH, b["FEAT"], e["t0"], b["E1"], e["t0_b"], silu_enable=True), b["E1"], (n, EH)),
            solo("A.t2", lambda E: rms(E, b["E1"], b["E2"], e["t2"]), b["E2"], (n, EH)),
            solo("A.t3", lambda E: self._mmw(E, n, EH, EH, b["E2"], e["t3"], b["E1"], e["t3_b"], silu_enable=True), b["E1"], (n, EH)),
            solo("A.t5", lambda E: rms(E, b["E1"], b["E2"], e["t5"]), b["E2"], (n, EH)),
            solo("A.t6", lambda E: self._mmw(E, n, EH, EH, b["E2"], e["t6"], b["E1"], e["t6_b"]), b["E1"], (n, EH)),
            solo("A.ln", lambda E: E.ue.layer_norm_core_dram(M=n, N=EH, A_DRAM_ADDR=b["E1"], OUTPUT_DRAM_ADDR=b["X_A"],
                                                           GAMMA_DRAM_ADDR=e["ln_w"], BETA_DRAM_ADDR=e["ln_b"]), b["X_A"], (n, EH)),
        ]
        layers, x = self._stack_ops("E", n, EH, e["layers"], b["ROPE_EXP"], self.E0, CAP, b["MASK_EXP"])
        ops += layers
        ops += [
            solo("E.norm", lambda E, x=x: rms(E, x, b["E1"], e["norm"]), b["E1"], (n, EH)),
            solo("E.vel", lambda E: self._mmw(E, n, EH, V64, b["E1"], e["out"], b["VOUT"], e["out_b"]), b["VOUT"], (n, V64)),
        ]
        self._compile("expert", lambda ops=ops: self._emit_ops(ops, n, "cols"), parallel=True)
        self.stage_ops["expert"] = (ops, n)
        self._report_programs("LM + expert", t0)

    def profile(self, stage):
        """Where one decode step / one denoise step spends its time: re-run the stage's program
        with only one kind of op emitted (same rounds, same shards). Timing only -- the data
        these partial programs leave behind is meaningless, so call this after the real run."""
        ops, M = self.stage_ops[stage]
        variants = [("rendezvous only", "none", None), ("row ops", "row", None), ("matmuls", "mm", None),
                    ("attn: K/V/Q store + gather", "heads", ("store",)), ("attn: V^T block update", "heads", ("vt",)),
                    ("attn: scale + scores + P.V", "heads", ("attn",))]
        full = self._exec(stage)
        print(f"\nprofile: one {stage} step = {full / 1e3:.1f} ms on {self.NE} engine{'s' if self.NE > 1 else ''} "
              f"(each line re-runs the step with one kind of op; rendezvous is in every line)")
        base = None
        for label, only, parts in variants:
            self._attn_parts = parts or ("store", "vt", "attn")
            name = f"prof_{stage}_{label}"
            self._compile(name, lambda only=only: self._emit_ops(ops, M, "cols", only=only), parallel=True)
            us = self._exec(name)
            base = us if base is None else base
            net = us if only == "none" else us - base
            print(f"  {label:28s} {us / 1e3:8.1f} ms   net of rendezvous {net / 1e3:8.1f} ms  ({100 * net / (full or 1):4.0f}% of the step)")
        self._attn_parts = ("store", "vt", "attn")

    def lm_read_kv(self, name, layer, n):
        return self.dma_from_accelerator_memory(self._kv(name, layer, 0), (KVH, self.KV_CAP, HD))[:, :n].float()


# ----------------------------------------------------------------------------- host weight prep

def prepare_lm_if4(ref):
    """IF4 packs of the LM matmuls and the lm_head (cached on disk), and the SAME dequantized
    weights installed into the reference so the oracle predicts the device."""
    import quant_lib
    bin_dir = Path(_HERE) / "alpamayo2_edge_bin"
    path = bin_dir / "lm_if4.pt"
    half = LI // 2

    def sources():
        yield "head", "vlm.lm_head.weight", None
        for li in range(LML):
            p = f"{LM}layers.{li}."
            for short, name in (("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"), ("v", "self_attn.v_proj"), ("o", "self_attn.o_proj")):
                yield f"{li}.{short}", p + name + ".weight", None
            for i in range(2):      # fc1 by output rows, fc2 by input columns (64-aligned: blocks intact)
                yield f"{li}.fc1.{i}", p + "mlp.fc1.weight", (slice(i * half, (i + 1) * half), slice(None))
                yield f"{li}.fc2.{i}", p + "mlp.fc2.weight", (slice(None), slice(i * half, (i + 1) * half))

    if path.exists():
        packs = torch.load(path, weights_only=False)
        print(f"IF4 packs loaded from {path}")
    else:
        t0, packs = time.time(), {}
        for key, name, sl in sources():
            t = ref.w[name] if sl is None else ref.w[name][sl]
            data, scale = quant_lib.quantize("if4", t.float().cpu().to(torch.bfloat16).contiguous(), block_size=64)
            packs[key] = (data, scale, t.shape[0], t.shape[1])
        bin_dir.mkdir(exist_ok=True)
        torch.save(packs, path)
        print(f"IF4 packs built in {time.time() - t0:.0f}s -> {path}")
    for key, name, sl in sources():
        data, scale, n, k = packs[key]
        deq = quant_lib.dequant("if4", data, scale, n, k).to(ref.w[name].device, ref.w[name].dtype)
        if sl is None:
            ref.w[name] = deq
        else:
            ref.w[name][sl] = deq
    return packs


# ----------------------------------------------------------------------------- driver

def pick_sample(path):
    """A real-clip golden if one has been captured, else the synthetic one (real photographs
    through NVIDIA's processor at the driving resolutions)."""
    if path == "none":
        return None
    if path:
        return Path(path).expanduser()
    root = Path("~/a2e_golden").expanduser()
    have = sorted(d for d in root.glob("*") if (d / "packed.safetensors").exists())
    real = [d for d in have if not d.name.startswith("synthetic")]
    return (real or have)[0] if have else None      # None: the reference's built-in synthetic input


def utilization(ue, stage, programs, us):
    """One line: executed matmul FLOPs of a stage against the raw peak of the engines in use."""
    flops = sum(ue.done_flops.get(p, 0) for p in programs)
    peak = ue.NE * 128 * 1e3 / ue._clock_period_ns / 1e3                 # GFLOPS
    rate = flops / us / 1e3 if us else 0.0                               # FLOP / us -> GFLOPS
    ue.util = getattr(ue, "util", {})
    ue.util[stage] = (flops, us, rate, rate / peak)
    print(f"  [util] {stage}: {flops / 1e12:.2f} TFLOP in {us / 1e6:.1f} s = {rate:.0f} GFLOPS, "
          f"{100 * rate / peak:.0f}% of the {peak:.0f} GFLOPS peak ({ue.NE} engine{'s' if ue.NE > 1 else ''})")


def report(name, ref, got, extra=""):
    s, c = snr_db(ref, got), cos_sim(ref, got)
    print(f"  {name:22s} {str(tuple(got.shape)):>16s}  snr {s:6.1f} dB  cos {c:.5f}  {extra}")
    return s, c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", default=CFG["defaults"]["dev"])
    ap.add_argument("--engines", default="max", help="engine count, or 'max' = every core the bitstream reports")
    ap.add_argument("--golden", default=None, help="sample dir with packed.safetensors (default: a real clip if captured, else synthetic)")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--ref-device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--stop-at", choices=["vision", "prefill", "decode", "expert"], default="expert",
                    help="last stage to run on the device")
    ap.add_argument("--host-vision", action="store_true",
                    help="skip the device vision encoder + projector; upload the reference's image embeddings")
    ap.add_argument("--images", type=int, default=None, help="with --stop-at vision: run only the first K images")
    ap.add_argument("--staged", choices=["vision", "prefill"], default=None,
                    help="op by op with NaN + per-layer SNR checks, single engine (vision: one image)")
    ap.add_argument("--image-index", type=int, default=0, help="image for --staged vision")
    ap.add_argument("--stop-after", default=None, help="--staged: stop after this label, e.g. L0.attn_h0 / P0.out")
    ap.add_argument("--teacher", action="store_true",
                    help="decode NVIDIA's golden tokens instead of picking greedily, so the trajectory "
                         "compares like for like with the golden (leaves only the 4-bit / GELU effects)")
    ap.add_argument("--no-profile", dest="profile", action="store_false",
                    help="skip the per-op-kind timing of one decode step and one denoise step that follows the run")
    ap.add_argument("--profile-vision", action="store_true",
                    help="after the run, time the vision stage by op kind (recompiles it 7 times; adds a few minutes)")
    ap.add_argument("--max-new", type=int, default=CFG["model"]["max_new_tokens"])
    ap.add_argument("--steps", type=int, default=CFG["action_expert"]["flow_steps"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-snr", type=float, default=20.0, help="vision / projector gate vs the hw-gelu reference (dB)")
    ap.add_argument("--min-cos", type=float, default=0.99)
    ap.add_argument("--max-traj-err", type=float, default=0.5, help="gate: max |xyz error| vs the device-equivalent reference (m)")
    args = ap.parse_args()
    vision_only = args.stop_at == "vision" or args.staged == "vision"

    sample = pick_sample(args.golden)
    if sample is None:      # no captured sample (capture needs NVIDIA's code): small built-in input, no NVIDIA comparison
        packed, golden = synthetic_inputs(Alpamayo2EdgeRef(resolve_model_dir(args.model_dir), prefixes=("none",))), None
    else:
        packed = load_file(str(sample / "packed.safetensors"))
        golden = load_file(str(sample / "golden.safetensors")) if (sample / "golden.safetensors").exists() else None
    pixels = packed["tokenized_data.pixel_values"].float()
    grid = packed["tokenized_data.image_grid_thw"].long()
    grids = [(int(h), int(w)) for _, h, w in grid.tolist()]
    sizes = [h * w for h, w in grids]
    starts = [sum(sizes[:i]) for i in range(len(sizes))]
    print(f"sample {sample}: {len(grids)} images, {sum(sizes)} patches")

    # Oracle = the host reference made device-equivalent: hw GELU in vision / projector, IF4 LM.
    dtype = torch.bfloat16 if args.ref_device.startswith("cuda") else torch.float32
    ref = Alpamayo2EdgeRef(resolve_model_dir(args.model_dir), device=args.ref_device, dtype=dtype,
                           prefixes=(VIS, PROJ) if vision_only else None)
    if golden is not None and not vision_only:      # the constants NVIDIA's model really held
        ref.freqs = golden["freqs"].float().to(ref.dev)
        ref.stats = dict(zip(STAT_KEYS, golden["act_stats"].float().tolist()))
    image_ids = [args.image_index] if args.staged == "vision" else list(range(len(grids)))[: args.images if vision_only else None]

    def reference(i, hw_gelu):
        stages, layers = {}, []
        out = ref.vision_encode(pixels[starts[i] : starts[i] + sizes[i]], grid[i : i + 1],
                                stages=stages, hw_gelu=hw_gelu, layer_outs=layers)
        return stages["vision_embed"], layers, out

    user_dma_core.set_dma_device(args.dev)
    user_dma_core.configure_clock_from_hardware()
    cores = getattr(user_dma_core, "ANDROMEDA_CORE_COUNT", None) or 1
    engines = 1 if args.staged else (cores if args.engines == "max" else int(args.engines))
    assert 1 <= engines <= cores, f"--engines {engines} but the bitstream reports {cores} core(s)"
    print(f"engines: {engines} of {cores}")
    ue = Alpamayo2Edge_UnifiedEngine(dev=args.dev, engines=engines)
    mhz = 1e3 / ue._clock_period_ns
    print(f"clock {mhz:.1f} MHz -> raw peak {engines * 128 * mhz / 1e3:.1f} GFLOPS ({engines} x 128 FLOP/cycle)")
    ue.vision_weight_init(ref, sorted(set(grids)))
    ue.vision_tensor_init(grids)
    ue.vision_upload_pixels(pixels)

    if args.staged == "vision":
        with torch.no_grad():
            emb, layers, out = reference(args.image_index, hw_gelu=True)
        ue.vision_run_staged(args.image_index, emb, layers, out, stop_after=args.stop_after)
        return

    ok, hw_ms = True, {}
    rows = sum(sizes[i] for i in image_ids)
    ue.projector_init(ref)

    # ------------------------------------------------------------ vision + projector
    if args.host_vision:
        with torch.no_grad():
            img_ref = ref.project(ref.vision_encode(pixels, grid, hw_gelu=True), hw_gelu=True)
        ue.dma_to_accelerator_memory(ue.buf["IMG"], img_ref.to(torch.bfloat16).cpu().contiguous().flatten())
        img = img_ref.float().cpu()
        print("vision + projector: SKIPPED on the device (--host-vision), reference image embeddings uploaded")
    else:
        ue.vision_compile(image_ids)
        ue.projector_compile(rows)
        hw_ms["vision"] = ue._exec("vision", timeout=3600.0) / 1e3
        hw_ms["projector"] = ue._exec("projector") / 1e3
        utilization(ue, "vision", ["vision"], hw_ms["vision"] * 1e3)
        utilization(ue, "projector", ["projector"], hw_ms["projector"] * 1e3)
        print(f"\n{'image':>5} {'grid':>6} | vs hw-gelu ref: {'snr dB':>7} {'cos':>8} | vs published: {'snr dB':>7} {'cos':>8}")
        hw, pub, got = [], [], []
        with torch.no_grad():
            for i in image_ids:
                dev_out = ue.vision_read(i)
                r_hw, r_pub = reference(i, True)[2].float().cpu(), reference(i, False)[2].float().cpu()
                hw.append(r_hw); pub.append(r_pub); got.append(dev_out)
                s, c = snr_db(r_hw, dev_out), cos_sim(r_hw, dev_out)
                bad = bool(torch.isnan(dev_out).any()) or s < args.min_snr or c < args.min_cos
                ok &= not bad
                print(f"{i:5d} {grids[i][0]:>3}x{grids[i][1]:<2} | {s:23.1f} {c:8.5f} | "
                      f"{snr_db(r_pub, dev_out):21.1f} {cos_sim(r_pub, dev_out):8.5f}{'  FAIL' if bad else ''}")
            hw, pub, got = torch.cat(hw), torch.cat(pub), torch.cat(got)
            print(f"\nall {len(image_ids)} images: vs hw-gelu ref {snr_db(hw, got):.1f} dB cos {cos_sim(hw, got):.5f} | "
                  f"vs published {snr_db(pub, got):.1f} dB cos {cos_sim(pub, got):.5f} | hw {hw_ms['vision'] / 1e3:.1f} s")
            # Projector: isolated (reference projector fed the DEVICE vision output) and end to end.
            img = ue.projector_read(rows)
            on = lambda t: t.to(ref.dev, ref.dt)
            iso, e2e_hw, e2e_pub = (ref.project(on(got), hw_gelu=True).float().cpu(),
                                    ref.project(on(hw), hw_gelu=True).float().cpu(), ref.project(on(pub)).float().cpu())
        s, c = snr_db(iso, img), cos_sim(iso, img)
        proj_bad = bool(torch.isnan(img).any()) or s < args.min_snr or c < args.min_cos
        ok &= not proj_bad
        print(f"projector {tuple(img.shape)} hw {hw_ms['projector'] / 1e3:.1f} s: isolated {s:.1f} dB cos {c:.5f}{'  FAIL' if proj_bad else ''} | "
              f"end to end vs hw-gelu ref {snr_db(e2e_hw, img):.1f} dB cos {cos_sim(e2e_hw, img):.5f} | "
              f"vs published {snr_db(e2e_pub, img):.1f} dB cos {cos_sim(e2e_pub, img):.5f}")
    if vision_only:
        print(f"hw ms: {hw_ms}")
        print("PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)

    # ------------------------------------------------------------ LM setup
    ids = packed["tokenized_data.input_ids"][0]
    types = packed["tokenized_data.mm_token_type_ids"][0]
    L = ids.shape[0]
    P = L - 1                                   # the last prompt token is the first decode step
    packs = prepare_lm_if4(ref)
    ue.lm_weight_init(ref, packs)
    del packs
    ue.expert_weight_init(ref)
    ue.lm_tensor_init(P, args.max_new)
    S, CAP = ue.S, ue.KV_CAP
    pos, delta = ref.mrope_positions(types, grid)
    emb_table = ref.w[LM + "embed_tokens.weight"]

    # prompt embeddings: text rows from the host table, image rows scattered on the device from
    # the projector output. Pad rows [P, S) repeat text rows (finite, never attended by real rows).
    is_img = ids == ref.image_token
    x = emb_table[ids.to(ref.dev)].float().cpu()
    x[is_img] = 0
    text_rows = torch.nonzero(~is_img[:P]).flatten()
    x_pad = torch.cat([x[:P], x[text_rows[torch.arange(S - P) % len(text_rows)]]])
    ue.dma_to_accelerator_memory(ue.buf["X_A"], x_pad.to(torch.bfloat16).contiguous().flatten())
    runs, src, i = [], 0, 0
    while i < P:
        if is_img[i]:
            j = i
            while j < P and is_img[j]:
                j += 1
            runs.append((i, src, j - i))
            src, i = src + j - i, j
        else:
            i += 1
    assert src == img.shape[0], f"{src} image placeholders vs {img.shape[0]} image embeddings"
    slot_pos = torch.cat([pos[:, :P], (torch.arange(P, S) + delta).expand(3, -1)], 1)
    ue.lm_write_rope("ROPE_PRE", *ref.rope_cos_sin(slot_pos))
    ue.lm_compile(runs, args.max_new)

    # ------------------------------------------------------------ prefill
    with torch.no_grad():
        embeds = ref.embed_prompt(ids, img.to(ref.dev, ref.dt))
        cache = KVCache(ref.n_lm, ref.kv_heads, L + args.max_new + 1, ref.hd, ref.dev, ref.dt)
        ref_layers = []
        ref.lm_forward(embeds[:P], pos[:, :P], cache, layer_outs=ref_layers)
    ue._exec("scatter_img")
    emb_dev = ue.dma_from_accelerator_memory(ue.buf["X_A"], (S, LH)).float()[:P]
    print("\nLM (reference = IF4 weights, fed the device's image embeddings)")
    report("prompt embeddings", embeds[:P].float().cpu(), emb_dev)

    if args.staged == "prefill":
        ops, _ = ue._stack_ops("P", P, LH, ue.lw["layers"], ue.buf["ROPE_PRE"], 0, S, ue.buf["MASK_PRE"])
        ref_for = lambda label: ref_layers[int(label[1:].split(".")[0])].float().cpu() if label.endswith(".out") else None
        if ue._staged(ue._ops_as_steps(ops, P), lambda: ue._prime_rows(ue.eng[0], P), P, S, ref_for, args.stop_after, timeout=600.0):
            print("staged prefill complete")
        return

    hw_ms["prefill"] = (ue._exec("prefill", timeout=3600.0) + ue._exec("build_vt")) / 1e3
    x_dev = ue.dma_from_accelerator_memory(ue.prefill_out, (S, LH)).float()[:P]
    s, c = report("prefill last layer", ref_layers[-1].float().cpu(), x_dev, f"hw {hw_ms['prefill'] / 1e3:.1f} s")
    ok &= not bool(torch.isnan(x_dev).any())
    utilization(ue, "prefill", ["prefill"], hw_ms["prefill"] * 1e3)
    for layer in (0, LML - 1):
        for name, store in (("KC", cache.K), ("VC", cache.V)):
            report(f"{name[0].lower()} cache layer {layer}", store[layer, :, :P].float().cpu(), ue.lm_read_kv(name, layer, P))
    if args.stop_at == "prefill":
        print(f"hw ms: {hw_ms}")
        print("PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)

    # ------------------------------------------------------------ decode (greedy on device logits)
    head = ref.w["vlm.lm_head.weight"]
    tok, index, out, agree, dec_us = int(ids[-1]), P, [], 0, 0.0
    teacher = golden["generated_ids"].tolist() if (args.teacher and golden is not None) else None
    assert not args.teacher or teacher is not None, "--teacher needs a golden.safetensors with generated_ids"
    slots = torch.arange(CAP)
    print("\ndecode")
    while True:
        ue.dma_to_accelerator_memory(ue.buf["X_A"], emb_table[tok].to(torch.bfloat16).cpu().contiguous())
        ue.lm_write_rope("ROPE_CUR", *ref.rope_cos_sin(torch.full((3, 1), index + delta)))
        ue.lm_write_mask("MASK_DEC", 1, (slots < index) | (slots == ue.CUR))
        dec_us += ue._exec("decode") + ue._exec(f"append_{index}")
        with torch.no_grad():
            h = ref.lm_forward(emb_table[tok][None], torch.full((3, 1), index + delta), cache)
        if out and out[-1] == ref.anchor:
            break
        logits = ue.dma_from_accelerator_memory(ue.buf["LOGITS"], (VOCAB,)).float()
        ref_logits = torch.nn.functional.linear(h[0], head).float().cpu()
        nxt, ref_nxt = ref.pick(logits, "greedy", None, 0.6, 0.98), ref.pick(ref_logits, "greedy", None, 0.6, 0.98)
        agree += nxt == ref_nxt
        if teacher is not None:
            assert len(out) < len(teacher), "the golden token list ended without the anchor"
            nxt = teacher[len(out)]
        print(f"  step {len(out):2d} token {nxt:6d} (reference {ref_nxt:6d})  logits snr {snr_db(ref_logits, logits):5.1f} dB")
        ok &= not bool(torch.isnan(logits).any())
        if len(out) == args.max_new - 1 and nxt != ref.anchor:
            print(f"  no <|traj_future_start|> within {args.max_new} tokens: forcing it so the expert still runs")
            nxt = ref.anchor
        out.append(nxt)
        tok, index = nxt, index + 1
    n_cached = index + 1
    hw_ms["decode"] = dec_us / 1e3
    utilization(ue, "decode", ["decode"], dec_us)
    print(f"  {len(out)} tokens{' (teacher-forced: NVIDIA golden)' if teacher else ''}, device pick agrees with the "
          f"reference's pick on {agree}/{len(out)}, hw {dec_us / 1e6:.1f} s")
    if golden is not None:
        print(f"  NVIDIA golden tokens: {golden['generated_ids'].tolist()}\n  device tokens:        {out}")
    try:
        from transformers import AutoTokenizer
        print("  reasoning:", AutoTokenizer.from_pretrained(str(ref.dir)).decode(out[:-1]))
    except Exception as exc:        # the text is cosmetic; never fail the run on it
        print(f"  (reasoning text not decoded: {exc})")
    if args.stop_at == "decode":
        print(f"hw ms: {hw_ms}")
        print("PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)

    # ------------------------------------------------------------ action expert
    # Host per Euler step: Fourier features of the current action (no sin/cos core) and the update.
    n = NWAY
    ue.lm_write_rope("ROPE_EXP", *ref.rope_cos_sin((n_cached + delta + torch.arange(n)).expand(3, n)))
    ue.lm_write_mask("MASK_EXP", n, (slots < n_cached) | ((slots >= ue.E0) & (slots < ue.E0 + n)))
    if golden is not None and "x_steps" in golden:
        x0 = golden["x_steps"][0].float()
    else:
        x0 = torch.randn(n, 2, generator=torch.Generator().manual_seed(args.seed))
    ts = torch.linspace(0.0, 1.0, args.steps + 1)
    xa, exp_us, v_dev = x0.clone(), 0.0, []
    for i in range(args.steps):
        xd = xa.to(ref.dev)
        f = torch.cat([ref.fourier(xd[:, 0]), ref.fourier(xd[:, 1]), ref.fourier(ts[i].to(ref.dev)).expand(n, -1)], -1)
        feat = torch.zeros(n, UE_VECTOR_SIZE)
        feat[:, : f.shape[1]] = f.float().cpu()
        ue.dma_to_accelerator_memory(ue.buf["FEAT"], feat.to(torch.bfloat16).flatten())
        exp_us += ue._exec("expert")
        v = ue.dma_from_accelerator_memory(ue.buf["VOUT"], (n, UE_VECTOR_SIZE)).float()[:, :2]
        ok &= not bool(torch.isnan(v).any())
        v_dev.append(v)
        xa = xa + (ts[i + 1] - ts[i]) * v
    hw_ms["expert"] = exp_us / 1e3
    with torch.no_grad():
        st = {}
        cache.n = n_cached
        act_ref = ref.denoise(cache, delta, x0, args.steps, st).float().cpu()
    hist_xyz = packed["traj_data.ego_history_xyz"].reshape(-1, 3)
    hist_rot = packed["traj_data.ego_history_rot"].reshape(-1, 3, 3)
    xyz_dev, _ = ref.action_to_traj(xa, hist_xyz, hist_rot)
    xyz_ref, _ = ref.action_to_traj(act_ref, hist_xyz, hist_rot)
    print("\naction expert")
    v_ref = st["v_steps"].float().cpu()
    print("  v per step dB:", " ".join(f"{snr_db(v_ref[i], v_dev[i]):.0f}" for i in range(args.steps)))
    report("action", act_ref, xa, f"hw {exp_us / 1e6:.1f} s")
    utilization(ue, "expert", ["expert"], exp_us)
    err = float((xyz_dev - xyz_ref).abs().max())
    report("trajectory xyz", xyz_ref, xyz_dev, f"max |err| {err:.3f} m vs device-equivalent reference")
    ok &= err <= args.max_traj_err
    if golden is not None and "pred_xyz" in golden:
        g = golden["pred_xyz"].float().reshape(-1, 3)
        print(f"  vs NVIDIA golden (teacher noise; tokens {'equal' if out == golden['generated_ids'].tolist() else 'DIFFER'}): "
              f"max |err| {float((xyz_dev - g).abs().max()):.3f} m, snr {snr_db(g, xyz_dev):.1f} dB")
    if "gt_xyz" in (golden or {}):
        gt = golden["gt_xyz"].float().reshape(-1, 3)
        k = min(50, gt.shape[0], xyz_dev.shape[0])
        print(f"  ADE@5s vs logged future: {float((xyz_dev[:k, :2] - gt[:k, :2]).norm(dim=-1).mean()):.3f} m")

    print(f"\nhw time ({engines} engine{'s' if engines > 1 else ''}): " + ", ".join(f"{k} {v / 1e3:.1f} s" for k, v in hw_ms.items())
          + f" | total {sum(hw_ms.values()) / 1e3:.1f} s")
    print(f"\n{'stage':10s} {'TFLOP':>8} {'time s':>8} {'GFLOPS':>8} {'of peak':>8}")
    for stage, (fl, us, rate, frac) in ue.util.items():
        print(f"{stage:10s} {fl / 1e12:8.2f} {us / 1e6:8.1f} {rate:8.0f} {100 * frac:7.0f}%")
    fl, us = sum(v[0] for v in ue.util.values()), sum(v[1] for v in ue.util.values())
    peak = ue.NE * 128 * 1e3 / ue._clock_period_ns / 1e3
    print(f"{'total':10s} {fl / 1e12:8.2f} {us / 1e6:8.1f} {fl / us / 1e3 if us else 0:8.0f} {100 * (fl / us / 1e3 if us else 0) / peak:7.0f}%   "
          f"(matmul FLOPs incl. attention; norms, RoPE and elementwise ops are not counted)")
    if args.profile:
        ue.profile("decode")
        ue.profile("expert")
    if args.profile_vision and "vision" in ue.hw_us:
        ue.profile_vision()
    print("PASS" if ok else f"FAIL (gates: vision/projector snr >= {args.min_snr} dB, no NaN, trajectory max err <= {args.max_traj_err} m)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
