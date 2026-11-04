#!/usr/bin/env python3
"""FLUX.2 VAE decoder on the Unified Engine (phase 3), precompiled.

``AutoencoderKLFlux2`` decode path as a sequence of prebuilt per-engine
programs, compiled once (offline, shape-only) by ``flux2_klein_4b_test.py
--compile-only`` into the same ``programs/<config>_e<N>.{bin,json}`` as the
text encoder and the DiT, and replayed at run time without any instruction
capture:

  conv (k3/k1, bias, IF8 block scales)   one tile-loop program per engine (tile ranges sharded)
                                         preceded by an on-engine weight-stream expansion
  GroupNorm(32, affine, eps 1e-6)        stats + apply programs, pixel chunks sharded
  SiLU                                   rounds sharded
  residual add                           rounds sharded
  nearest 2x upsample                    strided DMA passes (engine 0)
  mid-block single-head attention        unified_attention_core, query rows sharded

Weights: the conv walker re-reads its weight stream once per output pixel of a
tile, so the resident stream of one conv is ``tile_pixels x`` its block of
codes (3.5 GB for the whole decoder).  Only the unrepeated blocks (~50 MB, one
per OC chunk, each under one URAM bank) are DMA'd to the board, once per run
as the VAE constant image; before every conv a prebuilt program on each engine
expands its chunks into the staging region (DRAM -> URAM once, URAM -> DRAM
``tile_pixels`` times).  Per layer the host only packs/moves the activation
map, as every conv model in this repo does.

DRAM during the phase (all baked into the program metadata):
  params arena   [const_base, const_end)       weight blocks, scales, biases, GN zero block,
                                               attention identity + zero bias  (loaded once)
                 [staging_base, staging_end)   per-layer host-written inputs + expanded streams
  tensor arena   [VAE_TENSOR_BASE, ...)        per-layer outputs / scratch
  program slots  after each engine's TE / DiT images

The host reference with the same IF8 weights (``reference_decode``) is the
oracle for this phase.
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import user_dma_core as udc  # noqa: E402
from user_dma_core import TYPE, UE_VECTOR_SIZE, INSTRUCTION_SIZE_BYTES  # noqa: E402
import flux2_klein_4b_weights as W  # noqa: E402

LINE = UE_VECTOR_SIZE * 2          # one 64-lane bf16 URAM line
URAM_B = 0x80000                   # URAM bank B byte offset


def emitter_hash() -> str:
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


def _split(n: int, parts: int, unit: int = 1) -> list[tuple[int, int]]:
    """Even [start, stop) split of ``n`` items (in multiples of ``unit``) over ``parts``."""
    assert n % unit == 0
    blocks = n // unit
    base, rem = divmod(blocks, parts)
    out, t = [], 0
    for e in range(parts):
        cnt = (base + (1 if e < rem else 0)) * unit
        out.append((t, t + cnt))
        t += cnt
    return out


def _i16bytes(t: torch.Tensor) -> bytes:
    return t.contiguous().view(torch.int16).numpy().tobytes()


class Flux2VaeFPGA:
    """Compiler (``compile()``, offline) and runtime (``decode()``) of the VAE phase."""

    def __init__(self, ue, weights: dict | None = None, cfg: dict | None = None, *, meta: dict | None = None,
                 progress: bool = True, timeout_s: float = 900.0):
        self.ue = ue
        self.engines = list(getattr(ue, "E", [ue]))
        self.ne = len(self.engines)
        self.w = weights
        self.cfg = cfg["vae"]
        self.groups = self.cfg["norm_num_groups"]
        self.gn_eps = self.cfg.get("gn_eps", 1e-6)
        self.block_out = list(self.cfg["block_out_channels"])
        self.n_res = self.cfg["layers_per_block"] + 1
        self.progress = progress
        self.timeout_s = timeout_s
        self.layer_times: list[tuple[str, float]] = []
        self.meta = meta                      # None while compiling
        self.compiling = meta is None
        self.steps: list[dict] = []
        self.images: list[tuple[int, int, bytes]] = []     # (engine, addr, bytes) while compiling
        self.const_base = ue._params_dram_base
        self.tensor_base = getattr(ue, "VAE_TENSOR_BASE", ue._tensor_dram_base)
        self._const = self.const_base
        self._cursor = 0
        self._dry = False
        self.staging_end = self.tensor_end = 0

    # ------------------------------------------------------------------ allocators
    def _calloc(self, n: int) -> int:
        self._const = (self._const + 63) & ~63
        a = self._const
        self._const += n
        return a

    def _reset(self) -> None:
        self.ue._next_params_dram_addr = self.staging_base
        self.ue._tensor_dram_addr = self.tensor_base

    def _salloc(self, n: int) -> int:
        a = self.ue.allocate_params_dram(n)
        self.staging_end = max(self.staging_end, self.ue._next_params_dram_addr)
        return a

    def _talloc(self, n: int) -> int:
        a = self.ue.allocate_tensor_dram(n)
        self.tensor_end = max(self.tensor_end, self.ue._tensor_dram_addr)
        return a

    # ------------------------------------------------------------------ programs
    def _capture(self, e: int, emit) -> dict:
        """Capture one standalone program on engine ``e`` at its next program address."""
        eng = self.engines[e]
        eng.clear_inst_id()
        eng.reset_isa_reg_counter()
        eng.reset_inst_ptr_counter()
        eng.start_capture()
        emit(eng)
        eng.stop_capture()
        eng.generate_instruction_halt()
        eng.pad_capture_to_64b_boundary()
        addr = eng.get_program_dram_addr()
        data = b"".join(inst.get_bytes() for inst in eng.capture_buffer)
        eng.allocate_program_dram(len(data))
        eng.clear_capture_buffer()
        self.images.append((e, addr, data))
        return {"engine": e, "addr": addr, "size": len(data)}

    def _launch(self, progs: list[dict]) -> None:
        for p in progs:
            self.engines[p["engine"]].start_execute_from_dram(p["addr"])
        for p in progs:
            self.engines[p["engine"]].wait_queue(self.timeout_s)

    def _step(self, kind: str, name: str, shape_in, shape_out, addrs: dict, build) -> dict:
        """Compile: record the step (programs from ``build()``). Run: fetch and check it."""
        addrs = {k: (list(v) if isinstance(v, (tuple, list)) else v) for k, v in addrs.items()}
        if self.compiling:
            if self._dry:                      # sizing pass: addresses only, no programs
                return None
            st = {"kind": kind, "name": name, "in": list(shape_in), "out": list(shape_out),
                  "addrs": addrs, "progs": build()}
            self.steps.append(st)
            return st
        st = self.meta["vae"]["steps"][self._cursor]
        self._cursor += 1
        assert st["kind"] == kind and st["name"] == name and st["in"] == list(shape_in), \
            f"VAE step {self._cursor - 1}: programs were compiled for {st['kind']} {st['name']} {st['in']}, " \
            f"running {kind} {name} {list(shape_in)}"
        assert st["addrs"] == addrs, f"VAE step {name}: baked addresses differ from this layout (recompile)"
        return st

    def _timed(self, name: str, shape, fn):
        if self.compiling:
            return fn()
        t0 = time.time()
        out = fn()
        dt = time.time() - t0
        self.layer_times.append((name, dt))
        if self.progress:
            print(f"    [vae] {name:52s} {tuple(shape)} {dt:6.2f}s", flush=True)
        return out

    @staticmethod
    def _shape(x):
        return tuple(x.shape) if isinstance(x, torch.Tensor) else tuple(x)

    def _dma(self, addr: int, t: torch.Tensor) -> None:
        self.ue.dma_write(self.ue.h2c_device, addr, t, t.numel() * t.element_size())

    # ------------------------------------------------------------------ conv
    def _conv_geometry(self, C: int, H: int, Wd: int, OC: int, kh: int, kw: int, pad: int) -> dict:
        ct = (C + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
        taps = kh * kw * ct
        patch = kh * kw * C
        chunks = (patch + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
        can_gather = C <= 255 and patch <= 256
        use_gather = can_gather and max(kh * kw * ((C + 3) // 4), OC * chunks) < OC * taps
        out_h, out_w, oc_chunk, tiles = udc.plan_conv2d_layer_tiles(
            c_in=C, oc_count=OC, in_h=H, in_w=Wd, kernel_h=kh, kernel_w=kw, stride_s=1,
            pad=pad, pad_h=pad, dilation=1, gather=use_gather, bias_enabled=True)
        groups, act_bytes, chunk_out = udc.conv2d_tiled_dram_layout(tiles, ct, oc_chunk)
        sg = max(groups, key=lambda g: g[2] * g[3])
        return dict(ct=ct, taps=taps, chunks=chunks, use_gather=use_gather, out_h=out_h, out_w=out_w,
                    oc_chunk=oc_chunk, n_chunks=OC // oc_chunk, tiles=tiles, groups=groups,
                    act_bytes=act_bytes, chunk_out=chunk_out, npix=sg[2] * sg[3],
                    blocks_per_oc=chunks if use_gather else taps)

    def _conv_shape(self, key: str) -> tuple[int, int, int, int, int]:
        """(OC, C, kh, kw, pad) of a decoder conv from the config alone (compile needs no weights)."""
        if self.w is not None:
            c = self.w["convs"][key]
            OC, C, kh, kw = c["codes"].shape
            return OC, C, kh, kw, c["pad"]
        return self._conv_shapes[key]

    def conv(self, x, key: str):
        C, H, Wd = self._shape(x)
        OC, wc_c, kh, kw, pad = self._conv_shape(key)
        assert wc_c == C, f"{key}: weight C_in {wc_c} != activation C {C}"
        g = self._conv_geometry(C, H, Wd, OC, kh, kw, pad)
        self._reset()
        n_oc, nck = g["oc_chunk"], g["n_chunks"]
        block_bytes = n_oc * g["blocks_per_oc"] * UE_VECTOR_SIZE          # IF8: 64 B per block
        assert block_bytes <= udc.URAM_NEAR_FULL_SIZE, f"{key}: weight block {block_bytes} B exceeds URAM_A"
        ACT = self._salloc(g["act_bytes"])
        OUT = self._talloc(nck * g["chunk_out"])
        WBLK = [self._calloc(block_bytes) for _ in range(nck)]
        SA = [self._calloc(n_oc * g["blocks_per_oc"] * 2) for _ in range(nck)]
        BA = [self._calloc(n_oc * 2) for _ in range(nck)]
        WEXP = [self._salloc(block_bytes * g["npix"]) for _ in range(nck)]
        addrs = dict(ACT=ACT, OUT=OUT, WBLK=WBLK, SA=SA, BA=BA, WEXP=WEXP, block_bytes=block_bytes,
                     npix=g["npix"], oc_chunk=n_oc, gather=g["use_gather"])
        lalu_mode, lalu_a, lalu_b = udc._conv_fused_lalu(False, False, False)

        def build():
            # weight-stream expansion: chunk ci on engine ci % ne
            expand = []
            for e in range(self.ne):
                mine = [ci for ci in range(nck) if ci % self.ne == e]
                if not mine:
                    continue

                def emit_expand(eng, mine=mine):
                    for ci in mine:
                        eng.accelerator_memory_to_sram(accelerator_dram_address=WBLK[ci], sram_address=0,
                                                       element_size=0, memcpy_length_bytes=block_bytes)
                        reg = eng.alloc_isa_reg()
                        eng.generate_instruction_add_set(reg, WEXP[ci] >> 3)
                        start = eng.get_program_dram_addr()
                        eng.generate_instruction_jump_abs(udc.ue_35bit_addr_shifter(
                            start + (eng.capture_count + 1) * INSTRUCTION_SIZE_BYTES))
                        eng.loop_start(loop_cnt=g["npix"])
                        eng.sram_to_accelerator_memory(sram_address=0, accelerator_dram_address=0, element_size=0,
                                                       memcpy_length_bytes=block_bytes, general_reg_src=reg)
                        eng.generate_instruction_add_imm(src_reg_idx=reg, immediate_value=block_bytes >> 3)
                        eng.loop_end()
                        eng.release_isa_reg()
                expand.append(self._capture(e, emit_expand))
            # tile loops: every (chunk, geometry group) tile range split evenly across engines
            per_engine = [[] for _ in range(self.ne)]
            for ci in range(nck):
                for gi, grp in enumerate(g["groups"]):
                    t = grp[0]
                    for e, (a, b) in enumerate(_split(grp[1] - grp[0], self.ne)):
                        if b > a:
                            per_engine[e].append((ci, gi, t + a, t + b))
            conv = []
            for e in range(self.ne):
                if not per_engine[e]:
                    continue

                def emit_conv(eng, work=per_engine[e]):
                    for ci, gi, t0, t1 in work:
                        (start, stop, th, tw, _wh, win_w, act_off, act_b, out_off, out_b, _rl) = g["groups"][gi]
                        rel = t0 - start
                        eng._capture_conv2d_tile_loop(
                            n_tiles=t1 - t0,
                            act_base=ACT + act_off + rel * act_b, act_bytes=act_b,
                            weights_dram_addr=WEXP[ci],
                            out_base=OUT + ci * g["chunk_out"] + out_off + rel * out_b, out_bytes=out_b,
                            kernel_w=kw, kernel_h=kh, ct=g["ct"], oc_count=n_oc, out_w=tw, out_h=th,
                            w_pad=win_w, stride_s=1, dilation=1, data_type=TYPE.IF8,
                            scale_dram_addr=SA[ci], scale_count=n_oc * g["blocks_per_oc"],
                            bias_dram_addr=BA[ci], results=th * tw * n_oc,
                            lalu_mode=lalu_mode, lalu_a=lalu_a, lalu_b=lalu_b,
                            gather=g["use_gather"], c_in=C)
                conv.append(self._capture(e, emit_conv))
            return {"expand": expand, "conv": conv}

        st = self._step("conv", key, (C, H, Wd), (OC, g["out_h"], g["out_w"]), addrs, build)
        if self.compiling:
            return (OC, g["out_h"], g["out_w"])

        def run():
            staged = udc.conv2d_pack_activation_tiles(x, g["tiles"], pad=pad, pad_h=pad)
            assert staged.numel() * 2 == g["act_bytes"]
            self._dma(ACT, staged)
            self._launch(st["progs"]["expand"])
            self._launch(st["progs"]["conv"])
            big = self.ue.dma_from_accelerator_memory(OUT, (nck * g["chunk_out"] // 2,))
            return udc.conv2d_unpack_tiled_result(big, g["tiles"], g["out_h"], g["out_w"], OC, n_oc)
        return self._timed(key, (OC, g["out_h"], g["out_w"]), run)

    # ------------------------------------------------------------------ GroupNorm
    def gn(self, x, key: str):
        C, H, Wd = self._shape(x)
        (ct, slots, slot_lines, n_full, tail, flush_every, _nf) = udc.plan_group_norm(C, H, Wd, self.groups)
        pixels = H * Wd
        chunk_bytes = slot_lines * LINE
        n_chunks = n_full + (1 if tail else 0)
        assert 2 * slot_lines <= 4096 and 3 * slot_lines <= 4096
        self._reset()
        ACT = self._salloc(pixels * ct * LINE)
        AREP = self._salloc(chunk_bytes)
        BREP = self._salloc(chunk_bytes)
        OUT = self._talloc(pixels * ct * LINE)
        ranges = _split(n_chunks, self.ne)
        n_fl = [-(-(b - a) // flush_every) if b > a else 0 for a, b in ranges]
        SUM = [self._talloc(chunk_bytes * n) if n else 0 for n in n_fl]
        SQ = [self._talloc(chunk_bytes * n) if n else 0 for n in n_fl]
        addrs = dict(ACT=ACT, AREP=AREP, BREP=BREP, OUT=OUT, SUM=SUM, SQ=SQ, ZERO=self.ZERO,
                     ranges=[list(r) for r in ranges])
        A_CHUNK, A_SQUARE = 0, slot_lines
        B_COPY, B_SUM, B_SQ = 0, slot_lines, 2 * slot_lines
        a_sram = lambda ln: ln * LINE                 # noqa: E731
        b_sram = lambda ln: URAM_B + ln * LINE        # noqa: E731

        def build():
            stats, apply = [], []
            for e, (c0, c1) in enumerate(ranges):
                if c1 <= c0:
                    continue

                def emit_stats(eng, c0=c0, c1=c1, e=e):
                    def prime():
                        for dst in (B_SUM, B_SQ):
                            eng.accelerator_memory_to_sram(accelerator_dram_address=self.ZERO, sram_address=b_sram(dst),
                                                           element_size=0, memcpy_length_bytes=chunk_bytes)

                    def flush(f):
                        for src, base in ((B_SUM, SUM[e]), (B_SQ, SQ[e])):
                            eng.sram_to_accelerator_memory(sram_address=b_sram(src), accelerator_dram_address=base + f * chunk_bytes,
                                                           element_size=0, memcpy_length_bytes=chunk_bytes)
                    prime()
                    for i in range(c0, c1):
                        take_lines = (slots if i < n_full else tail) * ct
                        take_bytes = take_lines * LINE
                        src = ACT + i * chunk_bytes
                        eng.accelerator_memory_to_sram(accelerator_dram_address=src, sram_address=a_sram(A_CHUNK),
                                                       element_size=0, memcpy_length_bytes=take_bytes)
                        eng.accelerator_memory_to_sram(accelerator_dram_address=src, sram_address=b_sram(B_COPY),
                                                       element_size=0, memcpy_length_bytes=take_bytes)
                        eng.eltwise_add_core(a_sram(A_CHUNK), b_sram(B_SUM), b_sram(B_SUM), take_lines * UE_VECTOR_SIZE)
                        eng.eltwise_mul_core(a_sram(A_CHUNK), b_sram(B_COPY), a_sram(A_SQUARE), take_lines * UE_VECTOR_SIZE)
                        eng.eltwise_add_core(a_sram(A_SQUARE), b_sram(B_SQ), b_sram(B_SQ), take_lines * UE_VECTOR_SIZE)
                        j = i - c0
                        if (j + 1) % flush_every == 0 or i == c1 - 1:
                            flush(j // flush_every)
                            if i != c1 - 1:
                                prime()
                stats.append(self._capture(e, emit_stats))

                def emit_apply(eng, c0=c0, c1=c1):
                    B_A, B_B = 0, slot_lines
                    eng.accelerator_memory_to_sram(accelerator_dram_address=AREP, sram_address=b_sram(B_A),
                                                   element_size=0, memcpy_length_bytes=chunk_bytes)
                    eng.accelerator_memory_to_sram(accelerator_dram_address=BREP, sram_address=b_sram(B_B),
                                                   element_size=0, memcpy_length_bytes=chunk_bytes)
                    for i in range(c0, c1):
                        take_lines = (slots if i < n_full else tail) * ct
                        take_bytes = take_lines * LINE
                        off = i * chunk_bytes
                        eng.accelerator_memory_to_sram(accelerator_dram_address=ACT + off, sram_address=a_sram(A_CHUNK),
                                                       element_size=0, memcpy_length_bytes=take_bytes)
                        eng.eltwise_mul_core(a_sram(A_CHUNK), b_sram(B_A), a_sram(A_CHUNK), take_lines * UE_VECTOR_SIZE)
                        eng.eltwise_add_core(a_sram(A_CHUNK), b_sram(B_B), a_sram(A_CHUNK), take_lines * UE_VECTOR_SIZE)
                        eng.sram_to_accelerator_memory(sram_address=a_sram(A_CHUNK), accelerator_dram_address=OUT + off,
                                                       element_size=0, memcpy_length_bytes=take_bytes)
                apply.append(self._capture(e, emit_apply))
            return {"stats": stats, "apply": apply}

        st = self._step("gn", key, (C, H, Wd), (C, H, Wd), addrs, build)
        if self.compiling:
            return (C, H, Wd)
        n = self.w["norms"][key]

        def run():
            self._dma(ACT, udc.conv2d_pack_activation_map(x.to(torch.bfloat16), 0))
            self._launch(st["progs"]["stats"])
            sum_c = torch.zeros(ct * UE_VECTOR_SIZE, dtype=torch.float64)
            sumsq_c = torch.zeros(ct * UE_VECTOR_SIZE, dtype=torch.float64)
            for e in range(self.ne):
                if n_fl[e]:
                    for base, acc in ((SUM[e], sum_c), (SQ[e], sumsq_c)):
                        blocks = self.ue.dma_from_accelerator_memory(base, (n_fl[e] * slot_lines * UE_VECTOR_SIZE,))
                        acc += blocks.reshape(-1, ct * UE_VECTOR_SIZE).to(torch.float64).sum(dim=0)
            A_vec, B_vec = udc.group_norm_fold_affine(sum_c[:C], sumsq_c[:C], (C // self.groups) * H * Wd,
                                                      self.groups, n["gamma"], n["beta"], self.gn_eps)

            def rep(vec):
                line = torch.zeros(ct * UE_VECTOR_SIZE, dtype=torch.bfloat16)
                line[:C] = vec
                return line.reshape(ct, UE_VECTOR_SIZE).repeat(slots, 1).contiguous()
            self._dma(AREP, rep(A_vec))
            self._dma(BREP, rep(B_vec))
            self._launch(st["progs"]["apply"])
            flat = self.ue.dma_from_accelerator_memory(OUT, (pixels * ct * UE_VECTOR_SIZE,))
            return udc.packed_map_to_chw(flat, H, Wd, C)
        return self._timed(key, (C, H, Wd), run)

    # ------------------------------------------------------------------ SiLU
    def silu(self, x, key: str):
        C, H, Wd = self._shape(x)
        ct = (C + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
        total_lines = H * Wd * ct
        round_lines = min(total_lines, 0xFFF // 4, (udc.URAM_NEAR_FULL_SIZE // LINE) // 2)
        n_rounds = -(-total_lines // round_lines)
        chunk_bytes = round_lines * LINE
        pair_bytes = 2 * chunk_bytes
        elements = round_lines * UE_VECTOR_SIZE
        coeffs = udc.SILU_RECIP_POLY
        self._reset()
        PAIRS = self._salloc(n_rounds * pair_bytes)
        OUT = self._talloc(n_rounds * chunk_bytes)
        ranges = _split(n_rounds, self.ne)
        SCR = [[self._talloc(chunk_bytes) for _ in range(3)] if b > a else [0, 0, 0] for a, b in ranges]
        addrs = dict(PAIRS=PAIRS, OUT=OUT, SCR=SCR, ranges=[list(r) for r in ranges])

        def build():
            progs = []
            for e, (r0, r1) in enumerate(ranges):
                if r1 <= r0:
                    continue

                def emit(eng, r0=r0, r1=r1, e=e):
                    XP, XN, WQ = SCR[e]
                    pair_reg = eng.alloc_isa_reg()
                    out_reg = eng.alloc_isa_reg()
                    eng.generate_instruction_add_set(pair_reg, (PAIRS + r0 * pair_bytes) >> 3)
                    eng.generate_instruction_add_set(out_reg, (OUT + r0 * chunk_bytes) >> 3)
                    start = eng.get_program_dram_addr()
                    eng.generate_instruction_jump_abs(udc.ue_35bit_addr_shifter(
                        start + (eng.capture_count + 1) * INSTRUCTION_SIZE_BYTES))
                    eng.loop_start(loop_cnt=r1 - r0)
                    eng.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=0, element_size=0,
                                                   memcpy_length_bytes=pair_bytes, general_reg_src=pair_reg)
                    eng.start_queue_for_maxpool2d_operation(act_sram_start_addr=0, output_sram_wb_addr=URAM_B,
                                                            kernel_w=2, kernel_h=1, out_w=round_lines, out_h=1,
                                                            w_pad=2 * round_lines, stride_s=2)
                    eng.sram_to_accelerator_memory(sram_address=URAM_B, accelerator_dram_address=XP, element_size=0,
                                                   memcpy_length_bytes=chunk_bytes)
                    eng.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=0, element_size=0,
                                                   memcpy_length_bytes=chunk_bytes, stride_bytes_per_chunk=LINE,
                                                   stride_jump_bytes=2 * LINE, general_reg_src=pair_reg)
                    eng.eltwise_sub_core(0, URAM_B, 0, elements)
                    eng.sram_to_accelerator_memory(sram_address=0, accelerator_dram_address=XN, element_size=0,
                                                   memcpy_length_bytes=chunk_bytes)
                    eng.eltwise_sub_core(0, URAM_B, 0, elements)
                    eng.exp_core(0, 0, elements)
                    eng.sram_to_accelerator_memory(sram_address=0, accelerator_dram_address=WQ, element_size=0,
                                                   memcpy_length_bytes=chunk_bytes)
                    eng.accelerator_memory_to_sram(accelerator_dram_address=WQ, sram_address=URAM_B, element_size=0,
                                                   memcpy_length_bytes=chunk_bytes)
                    eng.broadcast_mul(float(coeffs[-1]), 0, 0, elements)
                    eng.broadcast_add(float(coeffs[-2]), 0, 0, elements)
                    for coeff in reversed(coeffs[:-2]):
                        eng.eltwise_mul_core(0, URAM_B, 0, elements)
                        eng.broadcast_add(float(coeff), 0, 0, elements)
                    eng.eltwise_mul_core(0, URAM_B, URAM_B, elements)
                    eng.sram_to_accelerator_memory(sram_address=URAM_B, accelerator_dram_address=WQ, element_size=0,
                                                   memcpy_length_bytes=chunk_bytes)
                    eng.accelerator_memory_to_sram(accelerator_dram_address=XP, sram_address=URAM_B, element_size=0,
                                                   memcpy_length_bytes=chunk_bytes)
                    eng.eltwise_mul_core(0, URAM_B, 0, elements)
                    eng.sram_to_accelerator_memory(sram_address=0, accelerator_dram_address=XP, element_size=0,
                                                   memcpy_length_bytes=chunk_bytes)
                    eng.accelerator_memory_to_sram(accelerator_dram_address=XN, sram_address=0, element_size=0,
                                                   memcpy_length_bytes=chunk_bytes)
                    eng.accelerator_memory_to_sram(accelerator_dram_address=WQ, sram_address=URAM_B, element_size=0,
                                                   memcpy_length_bytes=chunk_bytes)
                    eng.eltwise_mul_core(0, URAM_B, 0, elements)
                    eng.accelerator_memory_to_sram(accelerator_dram_address=XP, sram_address=URAM_B, element_size=0,
                                                   memcpy_length_bytes=chunk_bytes)
                    eng.eltwise_add_core(0, URAM_B, 0, elements)
                    eng.sram_to_accelerator_memory(sram_address=0, accelerator_dram_address=0, element_size=0,
                                                   memcpy_length_bytes=chunk_bytes, general_reg_src=out_reg)
                    eng.generate_instruction_add_imm(src_reg_idx=pair_reg, immediate_value=pair_bytes >> 3)
                    eng.generate_instruction_add_imm(src_reg_idx=out_reg, immediate_value=chunk_bytes >> 3)
                    body = eng.loop_end()
                    assert body <= 256
                    eng.release_isa_reg()
                    eng.release_isa_reg()
                progs.append(self._capture(e, emit))
            return {"silu": progs}

        st = self._step("silu", key, (C, H, Wd), (C, H, Wd), addrs, build)
        if self.compiling:
            return (C, H, Wd)

        def run():
            paired, shape = udc.UnifiedEngine._pack_relu_pairs(x)
            assert shape == (C, H, Wd, total_lines, round_lines, n_rounds)
            self._dma(PAIRS, paired)
            self._launch(st["progs"]["silu"])
            flat = self.ue.dma_from_accelerator_memory(OUT, (n_rounds * round_lines * UE_VECTOR_SIZE,))
            return udc.packed_map_to_chw(flat[:total_lines * UE_VECTOR_SIZE], H, Wd, C)
        return self._timed(key + ".silu", (C, H, Wd), run)

    # ------------------------------------------------------------------ residual add
    def add(self, x, y, key: str):
        C, H, Wd = self._shape(x)
        assert self._shape(y) == (C, H, Wd)
        ct = (C + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
        total_lines = H * Wd * ct
        round_lines = min(udc.URAM_NEAR_FULL_SIZE // LINE, 4096)
        n_rounds = -(-total_lines // round_lines)
        self._reset()
        XB = self._salloc(total_lines * LINE)
        YB = self._salloc(total_lines * LINE)
        OUT = self._talloc(total_lines * LINE)
        ranges = _split(n_rounds, self.ne)
        addrs = dict(XB=XB, YB=YB, OUT=OUT, ranges=[list(r) for r in ranges])

        def build():
            progs = []
            for e, (r0, r1) in enumerate(ranges):
                if r1 <= r0:
                    continue

                def emit(eng, r0=r0, r1=r1):
                    for r in range(r0, r1):
                        done = r * round_lines
                        take = min(round_lines, total_lines - done)
                        off = done * LINE
                        nbytes = take * LINE
                        eng.accelerator_memory_to_sram(accelerator_dram_address=XB + off, sram_address=0,
                                                       element_size=0, memcpy_length_bytes=nbytes)
                        eng.accelerator_memory_to_sram(accelerator_dram_address=YB + off, sram_address=URAM_B,
                                                       element_size=0, memcpy_length_bytes=nbytes)
                        eng.eltwise_add_core(0, URAM_B, 0, take * UE_VECTOR_SIZE)
                        eng.sram_to_accelerator_memory(sram_address=0, accelerator_dram_address=OUT + off,
                                                       element_size=0, memcpy_length_bytes=nbytes)
                progs.append(self._capture(e, emit))
            return {"add": progs}

        st = self._step("add", key, (C, H, Wd), (C, H, Wd), addrs, build)
        if self.compiling:
            return (C, H, Wd)

        def run():
            self._dma(XB, udc.conv2d_pack_activation_map(x.to(torch.bfloat16), 0))
            self._dma(YB, udc.conv2d_pack_activation_map(y.to(torch.bfloat16), 0))
            self._launch(st["progs"]["add"])
            flat = self.ue.dma_from_accelerator_memory(OUT, (total_lines * UE_VECTOR_SIZE,))
            return udc.packed_map_to_chw(flat, H, Wd, C)
        return self._timed(key + ".residual", (C, H, Wd), run)

    # ------------------------------------------------------------------ nearest 2x upsample
    def upsample(self, x, key: str):
        C, H, Wd = self._shape(x)
        out_h, out_w, ct, passes = udc.plan_nn_upsample_2x(H, Wd, C)
        max_chunk = max(p["chunk_bytes"] for p in passes)
        assert max_chunk <= udc.URAM_NEAR_FULL_SIZE, f"{key}: one input row ({max_chunk} B) exceeds the staging budget"
        self._reset()
        IN = self._salloc(H * Wd * ct * LINE)
        TMP = self._talloc(2 * H * Wd * ct * LINE)
        OUT = self._talloc(out_h * out_w * ct * LINE)
        addrs = dict(IN=IN, TMP=TMP, OUT=OUT)

        def build():
            def emit(eng):
                src_of = {"in": IN, "tmp": TMP}
                for p in passes:
                    src = src_of[p["src"]]
                    dst = TMP if p["kind"] == "v" else OUT
                    chunk, jump = p["chunk_bytes"], p["jump_bytes"]
                    strided = chunk <= udc.UE_STRIDE_CHUNK_MAX_BYTES and jump <= udc.UE_STRIDE_JUMP_MAX_BYTES
                    round_bytes = (udc.URAM_NEAR_FULL_SIZE // chunk) * chunk
                    done_bytes = chunks_done = 0
                    while done_bytes < p["total_bytes"]:
                        take = min(round_bytes, p["total_bytes"] - done_bytes)
                        eng.accelerator_memory_to_sram(accelerator_dram_address=src + done_bytes, sram_address=0,
                                                       element_size=0, memcpy_length_bytes=take)
                        base = dst + p["dst_off"] + chunks_done * jump
                        if strided:
                            eng.sram_to_accelerator_memory(sram_address=0, accelerator_dram_address=base, element_size=0,
                                                           memcpy_length_bytes=take, stride_bytes_per_chunk=chunk,
                                                           stride_jump_bytes=jump)
                        else:
                            for i in range(take // chunk):
                                eng.sram_to_accelerator_memory(sram_address=i * chunk, accelerator_dram_address=base + i * jump,
                                                               element_size=0, memcpy_length_bytes=chunk)
                        done_bytes += take
                        chunks_done += take // chunk
            return {"up": [self._capture(0, emit)]}

        st = self._step("upsample", key, (C, H, Wd), (C, out_h, out_w), addrs, build)
        if self.compiling:
            return (C, out_h, out_w)

        def run():
            self._dma(IN, udc.conv2d_pack_activation_map(x.to(torch.bfloat16), 0))
            self._launch(st["progs"]["up"])
            flat = self.ue.dma_from_accelerator_memory(OUT, (out_h * out_w * ct * UE_VECTOR_SIZE,))
            return udc.nn_upsample_2x_unpack_result(flat, out_h, out_w, C)
        return self._timed(key + ".nearest2x", (C, out_h, out_w), run)

    # ------------------------------------------------------------------ mid-block attention
    def attention_core(self, q, k, v, key: str):
        """q/k/v (C, H, W) -> softmax(q k^T / sqrt(C)) v as (C, H, W); query rows sharded."""
        C, H, Wd = self._shape(q)
        seq = H * Wd
        assert C % UE_VECTOR_SIZE == 0 and seq % UE_VECTOR_SIZE == 0
        self._reset()
        bpe = 2
        Q, K, V = (self._talloc(seq * C * bpe) for _ in range(3))
        OUT = self._talloc(seq * C * bpe)
        ranges = _split(seq, self.ne, unit=UE_VECTOR_SIZE) if seq % (self.ne * UE_VECTOR_SIZE) == 0 \
            else [(0, seq)] + [(0, 0)] * (self.ne - 1)
        SCRATCH = [self._talloc((C * seq + seq * seq + seq * C) * bpe) if b > a else 0 for a, b in ranges]
        BIAS = self.ATT_BIAS[seq]
        addrs = dict(Q=Q, K=K, V=V, OUT=OUT, SCRATCH=SCRATCH, BIAS=BIAS, IDENT=self.IDENT,
                     ranges=[list(r) for r in ranges])

        def build():
            progs = []
            for e, (r0, r1) in enumerate(ranges):
                if r1 <= r0:
                    continue

                def emit(eng, r0=r0, r1=r1, e=e):
                    eng.unified_attention_core(batch=r1 - r0, aligned_seq_len=seq, head_dim=C,
                                               Q_DRAM_ADDR=Q + r0 * C * bpe, K_DRAM_ADDR=K, V_DRAM_ADDR=V,
                                               BIAS_DRAM_ADDR=BIAS + r0 * seq * bpe,
                                               OUTPUT_DRAM_ADDR=OUT + r0 * C * bpe, SCRATCH_DRAM_ADDR=SCRATCH[e],
                                               IDENTITY_DRAM_ADDR=self.IDENT)
                progs.append(self._capture(e, emit))
            return {"attn": progs}

        st = self._step("attention", key, (C, H, Wd), (C, H, Wd), addrs, build)
        if self.compiling:
            return (C, H, Wd)

        def run():
            rows = torch.cat([t.reshape(C, seq).t() for t in (q, k, v)], 0).contiguous()   # (3*seq, C)
            self._dma(Q, rows)                                                              # Q, K, V are consecutive
            self._launch(st["progs"]["attn"])
            attn = self.ue.dma_from_accelerator_memory(OUT, (seq, C))
            return attn.t().reshape(C, H, Wd).contiguous()
        return self._timed(key + ".attention", (seq, C), run)

    # ------------------------------------------------------------------ graph
    def attention(self, x, p: str):
        h = self.gn(x, p + ".group_norm")
        q = self.conv(h, p + ".to_q")
        k = self.conv(h, p + ".to_k")
        v = self.conv(h, p + ".to_v")
        o = self.attention_core(q, k, v, p)
        o = self.conv(o, p + ".to_out.0")
        return self.add(x, o, p)

    def resnet(self, x, p: str, cin: int, cout: int):
        h = self.conv(self.silu(self.gn(x, p + ".norm1"), p + ".norm1"), p + ".conv1")
        h = self.conv(self.silu(self.gn(h, p + ".norm2"), p + ".norm2"), p + ".conv2")
        if cin != cout:
            x = self.conv(x, p + ".conv_shortcut")
        return self.add(x, h, p)

    def _graph(self, x):
        x = self.conv(x, "post_quant_conv")
        x = self.conv(x, "decoder.conv_in")
        c = self.block_out[-1]
        x = self.resnet(x, "decoder.mid_block.resnets.0", c, c)
        x = self.attention(x, "decoder.mid_block.attentions.0")
        x = self.resnet(x, "decoder.mid_block.resnets.1", c, c)
        rev = list(reversed(self.block_out))
        for i, cout in enumerate(rev):
            for j in range(self.n_res):
                x = self.resnet(x, f"decoder.up_blocks.{i}.resnets.{j}", c if j == 0 else cout, cout)
            c = cout
            if i < len(rev) - 1:
                x = self.upsample(x, f"decoder.up_blocks.{i}.upsamplers.0")
                x = self.conv(x, f"decoder.up_blocks.{i}.upsamplers.0.conv")
        x = self.silu(self.gn(x, "decoder.conv_norm_out"), "decoder.conv_norm_out")
        return self.conv(x, "decoder.conv_out")

    def conv_shapes(self) -> dict[str, tuple[int, int, int, int, int]]:
        """(OC, C, kh, kw, pad) of every decoder conv from the config (the shape-only compile)."""
        lat = self.cfg["latent_channels"]
        s = {"post_quant_conv": (lat, lat, 1, 1, 0)}
        c = self.block_out[-1]
        s["decoder.conv_in"] = (c, lat, 3, 3, 1)

        def resnet(p, cin, cout):
            s[p + ".conv1"] = (cout, cin, 3, 3, 1)
            s[p + ".conv2"] = (cout, cout, 3, 3, 1)
            if cin != cout:
                s[p + ".conv_shortcut"] = (cout, cin, 1, 1, 0)
        resnet("decoder.mid_block.resnets.0", c, c)
        for nm in ("to_q", "to_k", "to_v", "to_out.0"):
            s[f"decoder.mid_block.attentions.0.{nm}"] = (c, c, 1, 1, 0)
        resnet("decoder.mid_block.resnets.1", c, c)
        rev = list(reversed(self.block_out))
        for i, cout in enumerate(rev):
            for j in range(self.n_res):
                resnet(f"decoder.up_blocks.{i}.resnets.{j}", c if j == 0 else cout, cout)
            c = cout
            if i < len(rev) - 1:
                s[f"decoder.up_blocks.{i}.upsamplers.0.conv"] = (c, c, 3, 3, 1)
        s["decoder.conv_out"] = (self.cfg.get("out_channels", 3), c, 3, 3, 1)
        return s

    def _const_layout(self, lat_h: int, lat_w: int) -> None:
        """Shared constants at the head of the const image (same in both modes)."""
        self._const = self.const_base
        self.ZERO = self._calloc(udc.GROUP_NORM_SLOT_LINES * LINE)
        self.IDENT = self._calloc(UE_VECTOR_SIZE * UE_VECTOR_SIZE * 2)
        seq = lat_h * lat_w                       # the mid block runs at latent resolution
        self.ATT_BIAS = {seq: self._calloc(seq * seq * 2)}

    # ------------------------------------------------------------------ compile
    def compile(self, lat_h: int, lat_w: int) -> dict:
        """Shape-only pass: plans every address, captures every program. Returns the
        ``vae`` metadata; the program bytes are in ``self.images``."""
        assert self.compiling
        lat = self.cfg["latent_channels"]
        self._conv_shapes = self.conv_shapes()
        # pass 1 (dry): walk the graph for the const image size (set by the conv blocks)
        self._dry = True
        self._const_layout(lat_h, lat_w)
        self.staging_base = self.tensor_base
        self._graph((lat, lat_h, lat_w))
        const_end = (self._const + 63) & ~63
        # pass 2: the same walk with the staging region after the const image, capturing programs
        self._dry = False
        self._const_layout(lat_h, lat_w)
        self.staging_base = (const_end + 4095) & ~4095
        self.staging_end = self.tensor_end = 0
        out = self._graph((lat, lat_h, lat_w))
        assert ((self._const + 63) & ~63) == const_end, "const image layout drifted between the two walks"
        assert self.staging_end <= self.ue._tensor_dram_base, \
            f"VAE staging {self.staging_end:#x} runs into the tensor arena {self.ue._tensor_dram_base:#x}"
        assert self.tensor_end <= self.ue._program_dram_base, \
            f"VAE tensors {self.tensor_end:#x} run into the program arena {self.ue._program_dram_base:#x}"
        return {"latent": [lat, lat_h, lat_w], "out": list(out),
                "const_base": self.const_base, "const_size": const_end - self.const_base,
                "staging_base": self.staging_base, "staging_end": self.staging_end,
                "tensor_base": self.tensor_base, "tensor_end": self.tensor_end,
                "zero": self.ZERO, "ident": self.IDENT, "att_bias": {str(k): v for k, v in self.ATT_BIAS.items()},
                "engines": self.ne, "steps": self.steps,
                "program_bytes": sum(len(b) for _, _, b in self.images)}

    # ------------------------------------------------------------------ run
    def _load_const_image(self) -> None:
        """Pack the IF8 conv blocks / scales / biases at their baked addresses; one DMA."""
        vm = self.meta["vae"]
        img = bytearray(vm["const_size"])
        base = vm["const_base"]

        def put(addr, data: bytes):
            img[addr - base:addr - base + len(data)] = data
        put(vm["ident"], _i16bytes(torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16)))
        for st in vm["steps"]:
            if st["kind"] != "conv":
                continue
            c = self.w["convs"][st["name"]]
            a = st["addrs"]
            n_oc, gather = a["oc_chunk"], a["gather"]
            OC, C, kh, kw = c["codes"].shape
            ct = (C + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
            taps, chunks = kh * kw * ct, (kh * kw * C + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
            bs = c["block_scales"].to(torch.bfloat16)
            for ci, oc0 in enumerate(range(0, OC, n_oc)):
                wc = c["codes"][oc0:oc0 + n_oc]
                sc = udc._conv2d_chunk_scale(bs, 1.0, oc0=oc0, n_oc=n_oc)
                if gather:
                    blk = udc.conv2d_pack_weight_stream_gather(wc, 1, 1, TYPE.IF8)
                    ss = udc.conv2d_pack_scale_stream_gather(sc, n_oc, chunks)
                else:
                    blk = udc.conv2d_pack_weight_stream(wc, 1, 1, TYPE.IF8)
                    ss = udc.conv2d_pack_scale_stream(sc, n_oc, taps, 1, 1)
                assert blk.numel() == a["block_bytes"]
                put(a["WBLK"][ci], blk.numpy().tobytes())
                put(a["SA"][ci], _i16bytes(ss))
                put(a["BA"][ci], _i16bytes(udc.conv2d_pack_bias_stream(c["bias"][oc0:oc0 + n_oc], 1, 1)))
        n = self.ue.dma_write(self.ue.h2c_device, base, bytes(img), len(img))
        assert n == len(img), f"VAE const image: DMA wrote {n} of {len(img)} bytes"

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """(32, h, w) bf16 latent (already BN de-normalised) -> (3, 8h, 8w) bf16 in [-1, 1]."""
        assert not self.compiling and "vae" in self.meta, "programs have no VAE phase: recompile"
        vm = self.meta["vae"]
        assert list(latent.shape) == vm["latent"], f"programs compiled for latent {vm['latent']}, got {list(latent.shape)}"
        assert vm["engines"] == self.ne
        self.staging_base = vm["staging_base"]
        self._const_layout(latent.shape[1], latent.shape[2])
        self._cursor = 0
        t0 = time.time()
        self._load_const_image()
        self.layer_times.append(("const_image", time.time() - t0))
        if self.progress:
            print(f"    [vae] const image {vm['const_size'] / 1e6:.1f} MB -> 0x{vm['const_base']:X} "
                  f"{time.time() - t0:6.2f}s", flush=True)
        out = self._graph(latent.to(torch.bfloat16))
        assert self._cursor == len(vm["steps"])
        return out


# ----------------------------------------------------------------------------
# Host oracle with the same IF8 weights
# ----------------------------------------------------------------------------

def reference_decode(weights: dict, cfg: dict, latent: torch.Tensor, dtype=torch.float32,
                     device: str = "cpu") -> torch.Tensor:
    """Same graph on the host from the dequantized IF8 codes (bias in bf16)."""
    v = cfg["vae"]
    groups, eps = v["norm_num_groups"], v.get("gn_eps", 1e-6)
    convs, norms = weights["convs"], weights["norms"]
    x = latent[None].to(device=device, dtype=dtype)

    def conv(x, key):
        c = convs[key]
        w = W.dequantize_conv_if8(c["codes"], c["block_scales"]).to(device=device, dtype=dtype)
        return F.conv2d(x, w, c["bias"].to(device=device, dtype=dtype), padding=c["pad"])

    def gn(x, key):
        n = norms[key]
        return F.group_norm(x, groups, n["gamma"].to(device=device, dtype=dtype),
                            n["beta"].to(device=device, dtype=dtype), eps)

    def resnet(x, p, cin, cout):
        h = conv(F.silu(gn(x, p + ".norm1")), p + ".conv1")
        h = conv(F.silu(gn(h, p + ".norm2")), p + ".conv2")
        if cin != cout:
            x = conv(x, p + ".conv_shortcut")
        return x + h

    def attn(x, p):
        B, C, H, Wd = x.shape
        h = gn(x, p + ".group_norm")
        q, k, vv = (conv(h, p + f".{n}").reshape(B, C, H * Wd).transpose(1, 2) for n in ("to_q", "to_k", "to_v"))
        o = F.scaled_dot_product_attention(q[:, None], k[:, None], vv[:, None])[:, 0]
        o = conv(o.transpose(1, 2).reshape(B, C, H, Wd), p + ".to_out.0")
        return x + o

    x = conv(x, "post_quant_conv")
    x = conv(x, "decoder.conv_in")
    c = v["block_out_channels"][-1]
    x = resnet(x, "decoder.mid_block.resnets.0", c, c)
    x = attn(x, "decoder.mid_block.attentions.0")
    x = resnet(x, "decoder.mid_block.resnets.1", c, c)
    rev = list(reversed(v["block_out_channels"]))
    for i, cout in enumerate(rev):
        for j in range(v["layers_per_block"] + 1):
            x = resnet(x, f"decoder.up_blocks.{i}.resnets.{j}", c if j == 0 else cout, cout)
        c = cout
        if i < len(rev) - 1:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            x = conv(x, f"decoder.up_blocks.{i}.upsamplers.0.conv")
    x = F.silu(gn(x, "decoder.conv_norm_out"))
    return conv(x, "decoder.conv_out")[0].cpu()


def decoder_cost(cfg: dict, latent_h: int, latent_w: int) -> dict:
    """MACs per primitive for a (32, latent_h, latent_w) decode (host estimate)."""
    v = cfg["vae"]
    rev = list(reversed(v["block_out_channels"]))
    macs = 0
    ops = 0

    def conv(cin, cout, h, w, k):
        nonlocal macs, ops
        macs += h * w * cout * k * k * cin
        ops += 1
    h, w = latent_h, latent_w
    c = rev[0]
    conv(32, 32, h, w, 1); conv(32, c, h, w, 3)

    def resnet(cin, cout):
        conv(cin, cout, h, w, 3); conv(cout, cout, h, w, 3)
        if cin != cout:
            conv(cin, cout, h, w, 1)
    resnet(c, c)
    for _ in range(4):
        conv(c, c, h, w, 1)
    macs += 2 * (h * w) ** 2 * c
    resnet(c, c)
    for i, cout in enumerate(rev):
        for j in range(v["layers_per_block"] + 1):
            resnet(c if j == 0 else cout, cout)
        c = cout
        if i < len(rev) - 1:
            h, w = 2 * h, 2 * w
            conv(c, c, h, w, 3)
    conv(c, 3, h, w, 3)
    return {"macs": macs, "convs": ops, "out": (3, h, w)}
