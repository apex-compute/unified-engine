#!/usr/bin/env python3
"""Bring-up probes: every primitive the FLUX.2 compiler emits, run on the
board as a tiny program on the model's own DRAM buffers and real weights,
scored against torch (SNR dB). Localises a bad op in seconds instead of
bisecting a 4-step program.

    python models/flux2_klein_4b/flux2_klein_4b_probe.py [--height 256 --width 256 --max-seq-len 128] [--only ln,matmul_if4,...]
"""
from __future__ import annotations

import argparse
import json
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

import quant_lib  # noqa: E402
import user_dma_core  # noqa: E402
from user_dma_core import UE_MODE  # noqa: E402
import flux2_klein_4b_ref as ref  # noqa: E402
import flux2_klein_4b_test as T  # noqa: E402

BPE = 2


def snr(a, b):
    return T.snr_db(a, b)


class Probe:
    program = "dit"

    def __init__(self, ue: T.Flux2KleinEngine, paths: dict, meta_dit: dict):
        self.ue = ue
        self.paths = paths
        self.meta = meta_dit
        with open(paths["params_dit_bin"], "rb") as f:
            self.dit_bin = f.read()
        self.results = {}
        torch.manual_seed(0)

    # ---- weights straight from the packed bin (what the board multiplies by)
    def region(self, name_off: int, regions: dict, key: str) -> bytes:
        r = regions[key]
        off = name_off + r["offset"]
        return self.dit_bin[off:off + r["size"]], r

    def dbl_weight(self, i: int, name: str) -> torch.Tensor:
        dl = self.meta["double"]
        base = dl["base"] + i * dl["stride"]
        data, rd = self.region(base, dl["regions"], name + "_data")
        scale, _ = self.region(base, dl["regions"], name + "_scale")
        N, K = rd["shape"]
        return quant_lib.dequant(self.meta["precision"], data, scale, N, K, block_size=self.meta["block_size"])

    def dbl_vec(self, i: int, name: str) -> torch.Tensor:
        dl = self.meta["double"]
        data, rd = self.region(dl["base"] + i * dl["stride"], dl["regions"], name)
        return torch.frombuffer(bytearray(data), dtype=torch.bfloat16).clone()

    def nl_bf16(self, name: str) -> torch.Tensor:
        data, rd = self.region(0, self.meta["nonlayer_regions"], name)
        return torch.frombuffer(bytearray(data), dtype=torch.bfloat16).clone().reshape(rd["shape"])

    # ---- one captured program, emitted into EVERY engine's stream (sharded)
    def run(self, name: str, emit, timeout=300.0):
        ue = self.ue
        ue.clear_inst_id()
        ue.start_capture()
        ue.reset_inst_ptr_counter()
        if ue.sched is not None:
            ue.sched.begin_program()
        for e, eng in enumerate(ue.E):
            ue.cur, ue.e = eng, e
            ue._prime_regs(self.program)
            emit()
            if e == 0:
                eng.generate_instruction_halt()
        ue.cur, ue.e = ue, 0
        ue.stop_capture()
        prog = ue.get_program_dram_addr()
        n = ue.get_capture_instruction_size_bytes()
        ue.write_captured_instructions_to_dram(prog)
        ue.allocate_program_dram(n)
        ue.clear_capture_buffer()
        waddrs = ue.sched.finalize() if ue.sched is not None else []
        t0 = time.time()
        if ue.sched is not None:
            if not getattr(ue, "_flags_precleared", False):
                ue.sched.preclear_flags()
                ue._flags_precleared = True
            ue.sched.start_workers(waddrs)
        ue.start_execute_from_dram(prog)
        ue.wait_queue(timeout)
        for w in ue.workers:
            w.wait_queue(60.0)
        return time.time() - t0, n

    def report(self, name: str, got: torch.Tensor, want: torch.Tensor, secs: float, nbytes: int, gate: float = 25.0):
        finite = bool(torch.isfinite(got.float()).all())
        s = snr(want, got) if finite else float("-inf")
        ok = finite and s >= gate
        self.results[name] = {"snr_db": s, "finite": finite, "ok": ok, "secs": secs, "inst_bytes": nbytes}
        print(f"  {'PASS' if ok else 'FAIL'}  {name:22s} SNR {s:7.1f} dB  finite={finite}  {secs:6.2f}s  {nbytes / 1e3:7.1f} KB", flush=True)
        if not ok:
            g, w = got.float().flatten(), want.float().flatten()
            print(f"        got[:6]={g[:6].tolist()}\n        want[:6]={w[:6].tolist()}  |want|max={w.abs().max():.3f} |got|max={g.abs().max():.3f}")
        return ok

    # ------------------------------------------------------------ probes
    def p_dram_copy(self):
        ue = self.ue
        x = torch.randn(ue.L, ue.D).to(torch.bfloat16)
        ue.dma_to_accelerator_memory(ue.TXT0, x)
        secs, n = self.run("dram_copy", lambda: ue._dram_copy(ue.TXT0, ue.H[0], ue.L, ue.D))
        got = ue.dma_from_accelerator_memory(ue.H[0], (ue.L, ue.D))
        self.report("dram_copy", got, x, secs, n, gate=100)

    def p_broadcast_row(self):
        ue = self.ue
        row = torch.randn(ue.D).to(torch.bfloat16)
        ue.dma_to_accelerator_memory(ue.mod_addr("img", 0, 2), row)
        secs, n = self.run("broadcast_row", lambda: ue._broadcast_row(ue.mod_addr("img", 0, 2), ue.GATE_B["img_msa"], ue.NI, ue.D))
        got = ue.dma_from_accelerator_memory(ue.GATE_B["img_msa"], (ue.NI, ue.D))
        self.report("broadcast_row", got, row[None].expand(ue.NI, -1), secs, n, gate=100)

    def p_eltwise(self):
        ue = self.ue
        a = torch.randn(ue.NI, ue.D).to(torch.bfloat16)
        b = torch.randn(ue.NI, ue.D).to(torch.bfloat16)
        ue.dma_to_accelerator_memory(ue.AO, a)
        ue.dma_to_accelerator_memory(ue.GATE_B["img_msa"], b)
        def emit():
            ue._elt(ue.NI, ue.D, ue.AO, ue.GATE_B["img_msa"], ue.G, UE_MODE.ELTWISE_MUL, "IMG")
            ue._elt(ue.NI, ue.D, ue.AO, ue.G, ue.Y, UE_MODE.ELTWISE_ADD, "IMG")
            ue._elt(ue.NI, ue.D, ue.Y, None, ue.MO, UE_MODE.MUL_BROADCAST, "IMG", scalar=-0.25)
            ue._elt(ue.NI, ue.D, ue.MO, None, ue.ATT, UE_MODE.ADD_BROADCAST, "IMG", scalar=1.0)
        secs, n = self.run("eltwise", emit)
        want = (((a.float() * b.float()).to(torch.bfloat16).float() + a.float()).to(torch.bfloat16).float() * -0.25).to(torch.bfloat16).float() + 1.0
        got = ue.dma_from_accelerator_memory(ue.ATT, (ue.NI, ue.D))
        self.report("eltwise_mul_add_bcast", got, want, secs, n, gate=35)

    def p_ln_mod(self):
        ue = self.ue
        x = (torch.randn(ue.NI, ue.D) * 3).to(torch.bfloat16)
        scale = (torch.randn(ue.D) * 0.5).to(torch.bfloat16)
        shift = (torch.randn(ue.D) * 0.5).to(torch.bfloat16)
        ue.dma_to_accelerator_memory(ue.H[0], x)
        ue.dma_to_accelerator_memory(ue.mod_addr("img", 0, 1, True), (1 + scale.float()).to(torch.bfloat16))
        ue.dma_to_accelerator_memory(ue.mod_addr("img", 0, 0), shift)
        secs, n = self.run("ln_mod", lambda: ue._ln_mod(ue.NI, ue.H[0], ue.HN, ue.mod_addr("img", 0, 1, True), ue.mod_addr("img", 0, 0), "IMG"))
        want = F.layer_norm(x.float(), (ue.D,), eps=1e-6) * (1 + scale.float()) + shift.float()
        got = ue.dma_from_accelerator_memory(ue.HN, (ue.NI, ue.D))
        self.report("ln_mod", got, want, secs, n)

    def p_matmul_if4(self):
        ue = self.ue
        x = torch.randn(ue.NI, ue.D).to(torch.bfloat16)
        ue.dma_to_accelerator_memory(ue.HN, x)
        secs, n = self.run("matmul_if4", lambda: ue._mm(ue.NI, ue.D, ue.D, ue.HN, ue.Q, "IMG", ue.dbl_B(0, "img_q")))
        w = self.dbl_weight(0, "img_q")
        want = x.float() @ w.float().T
        got = ue.dma_from_accelerator_memory(ue.Q, (ue.NI, ue.D))
        self.report("matmul_if4_3072", got, want, secs, n)
        # wide N (ff_gate, N=9216) with fused SiLU and K=9216 (ff_out)
        secs, n = self.run("matmul_if4_silu", lambda: ue._mm(ue.NI, ue.D, ue.Dm, ue.HN, ue.MG, "IMG", ue.dbl_B(0, "img_ff_gate"), silu_enable=True))
        w = self.dbl_weight(0, "img_ff_gate")
        want = F.silu(x.float() @ w.float().T)
        got = ue.dma_from_accelerator_memory(ue.MG, (ue.NI, ue.Dm))
        self.report("matmul_if4_silu_9216", got, want, secs, n)
        y = torch.randn(ue.NI, ue.Dm).to(torch.bfloat16)
        ue.dma_to_accelerator_memory(ue.MM, y)
        secs, n = self.run("matmul_if4_k9216", lambda: ue._mm(ue.NI, ue.Dm, ue.D, ue.MM, ue.MO, "IMG", ue.dbl_B(0, "img_ff_out")))
        w = self.dbl_weight(0, "img_ff_out")
        want = y.float() @ w.float().T
        got = ue.dma_from_accelerator_memory(ue.MO, (ue.NI, ue.D))
        self.report("matmul_if4_K9216", got, want, secs, n)

    def p_matmul_bf16(self):
        ue = self.ue
        x = torch.randn(ue.NI, ue.CIN).to(torch.bfloat16)
        ue.dma_to_accelerator_memory(ue.X_LAT[0], x)
        secs, n = self.run("matmul_bf16", lambda: ue._mm(ue.NI, ue.CIN, ue.D, ue.X_LAT[0], ue.HN_IMG, "IMG", ue.nl_B("x_embedder")))
        want = x.float() @ self.nl_bf16("x_embedder").float().T
        got = ue.dma_from_accelerator_memory(ue.HN_IMG, (ue.NI, ue.D))
        self.report("matmul_bf16_K128", got, want, secs, n)
        t = torch.randn(ue.STEPS, ue.D).to(torch.bfloat16)
        ue.dma_to_accelerator_memory(ue.TEMB_S, t)
        secs, n = self.run("matmul_bf16_mod", lambda: ue._mm(ue.STEPS, ue.D, 6 * ue.D, ue.TEMB_S, ue.MOD["img"], "STEPS", ue.nl_B("mod_img")))
        want = t.float() @ self.nl_bf16("mod_img").float().T
        got = ue.dma_from_accelerator_memory(ue.MOD["img"], (ue.STEPS, 6 * ue.D))
        self.report("matmul_bf16_M4_N18432", got, want, secs, n)

    def p_rms_heads(self):
        ue = self.ue
        x = torch.randn(ue.NI, ue.D).to(torch.bfloat16)
        ue.dma_to_accelerator_memory(ue.Q, x)
        secs, n = self.run("rms_heads", lambda: ue._rms(ue.NI * ue.NH, ue.HD, ue.Q, ue.QN, ue.dbl_w(0, "img_norm_q"), "MH_IMG"))
        g = self.dbl_vec(0, "img_norm_q")
        want = ref._rms_norm(x.view(ue.NI, ue.NH, ue.HD), g, 1e-6).reshape(ue.NI, ue.D)
        got = ue.dma_from_accelerator_memory(ue.QN, (ue.NI, ue.D))
        self.report("rms_norm_heads", got, want, secs, n)

    def p_rope(self):
        ue = self.ue
        x = torch.randn(ue.S, ue.D).to(torch.bfloat16)
        ue.dma_to_accelerator_memory(ue.QN, x)
        secs, n = self.run("rope", lambda: ue._rope(ue.S, ue.NH, ue.HD, ue.QN, ue.DIT_ROPE, "S"))
        tab = ue.dma_from_accelerator_memory(ue.DIT_ROPE, (ue.S, 2 * ue.HD)).float()
        c, s = tab[:, :ue.HD], tab[:, ue.HD:]
        xv = x.float().view(ue.S, ue.NH, ue.HD)
        h = ue.HD // 2
        a = xv * c[:, None, :]
        b = torch.cat([xv[..., h:] * s[:, None, :h], xv[..., :h] * s[:, None, h:]], -1)
        want = (a + b).reshape(ue.S, ue.D)
        got = ue.dma_from_accelerator_memory(ue.QN, (ue.S, ue.D))
        self.report("rope_gqa_joint", got, want, secs, n)

    def p_attention(self):
        ue = self.ue
        q = torch.randn(ue.S, ue.D).to(torch.bfloat16)
        k = torch.randn(ue.S, ue.D).to(torch.bfloat16)
        v = torch.randn(ue.S, ue.D).to(torch.bfloat16)
        ue.dma_to_accelerator_memory(ue.QN, q)
        ue.dma_to_accelerator_memory(ue.KN, k)
        ue.dma_to_accelerator_memory(ue.V, v)
        secs, n = self.run("attention", ue._joint_attention, timeout=600)
        qh = q.float().view(1, ue.S, ue.NH, ue.HD).transpose(1, 2)
        kh = k.float().view(1, ue.S, ue.NH, ue.HD).transpose(1, 2)
        vh = v.float().view(1, ue.S, ue.NH, ue.HD).transpose(1, 2)
        want = F.scaled_dot_product_attention(qh, kh, vh).transpose(1, 2).reshape(ue.S, ue.D)
        got = ue.dma_from_accelerator_memory(ue.ATT, (ue.S, ue.D))
        self.report("attention_24_heads", got, want, secs, n, gate=20)
        qh_dev = ue.dma_from_accelerator_memory(ue.QH, (ue.NH, ue.S, ue.HD))
        self.report("permute_head_major", qh_dev, q.view(ue.S, ue.NH, ue.HD).permute(1, 0, 2), 0, 0, gate=100)

    def p_silu(self):
        ue = self.ue
        t = (torch.randn(ue.STEPS, ue.D) * 3).to(torch.bfloat16)
        ue.dma_to_accelerator_memory(ue.TEMB, t)
        secs, n = self.run("silu", lambda: ue._activation(ue.STEPS * ue.D // 64, ue.TEMB, ue.TEMB_S, "silu", "SILU"))
        got = ue.dma_from_accelerator_memory(ue.TEMB_S, (ue.STEPS, ue.D))
        self.report("activation_silu", got, F.silu(t.float()), secs, n)

    def p_scatter(self):
        ue = self.ue
        x = torch.randn(ue.L, ue.TE_H).to(torch.bfloat16)
        ue.dma_to_accelerator_memory(ue.TE_X[1], x)
        ue.dma_to_accelerator_memory(ue.TXT_EMB_IN, torch.zeros(ue.L, ue.J, dtype=torch.bfloat16))
        self.program = "te"
        secs, n = self.run("scatter", lambda: ue._scatter_rows(
            ue.TE_X[1], ue.TE_H * BPE, ue.TXT_EMB_IN + 1 * ue.TE_H * BPE, ue.J * BPE, ue.L, ue.TE_H, "TE"))
        self.program = "dit"
        got = ue.dma_from_accelerator_memory(ue.TXT_EMB_IN, (ue.L, ue.J))
        want = torch.zeros(ue.L, ue.J, dtype=torch.bfloat16)
        want[:, ue.TE_H:2 * ue.TE_H] = x
        self.report("pbi_scatter_tap", got, want, secs, n, gate=100)

    def p_conditioning(self):
        ue = self.ue
        secs, n = self.run("conditioning", ue.emit_conditioning)
        state = {k: self.nl_bf16(n2) for n2, k in (
            ("time_l1", "time_guidance_embed.timestep_embedder.linear_1.weight"),
            ("time_l2", "time_guidance_embed.timestep_embedder.linear_2.weight"),
            ("mod_img", "double_stream_modulation_img.linear.weight"),
            ("mod_txt", "double_stream_modulation_txt.linear.weight"),
            ("mod_single", "single_stream_modulation.linear.weight"),
            ("norm_out", "norm_out.linear.weight"))}
        dit = ref.Flux2Transformer(state, ue.geo)
        want = {k: [] for k in ("img", "txt", "single", "out")}
        for s in range(ue.STEPS):
            mod = dit.modulation(dit.temb(float(ue.sigmas[s])))
            for k in want:
                want[k].append(torch.cat(mod[k], dim=1))
        ok = True
        for k in want:
            w = torch.cat(want[k], 0)
            got = ue.dma_from_accelerator_memory(ue.MOD[k], (ue.STEPS, ue.MOD_W[k]))
            ok &= self.report(f"modulation_{k}", got, w, secs, n, gate=20)
            got1 = ue.dma_from_accelerator_memory(ue.MOD_P1[k], (ue.STEPS, ue.MOD_W[k]))
            ok &= self.report(f"modulation_{k}_p1", got1, w.float() + 1.0, secs, n, gate=20)

    def p_text_layer_rope(self):
        ue = self.ue
        x = torch.randn(ue.L, ue.TE_NH * ue.TE_HD).to(torch.bfloat16)
        ue.dma_to_accelerator_memory(ue.TE_QN, x)
        self.program = "te"
        secs, n = self.run("te_rope", lambda: ue._rope(ue.L, ue.TE_NH, ue.TE_HD, ue.TE_QN, ue.TE_ROPE, "TE"))
        self.program = "dit"
        tab = ue.dma_from_accelerator_memory(ue.TE_ROPE, (ue.L, 2 * ue.TE_HD)).float()
        c, s = tab[:, :ue.TE_HD], tab[:, ue.TE_HD:]
        xv = x.float().view(ue.L, ue.TE_NH, ue.TE_HD)
        h = ue.TE_HD // 2
        want = (xv * c[:, None, :] + torch.cat([xv[..., h:] * s[:, None, :h], xv[..., :h] * s[:, None, h:]], -1)).reshape(ue.L, -1)
        got = ue.dma_from_accelerator_memory(ue.TE_QN, (ue.L, ue.TE_NH * ue.TE_HD))
        self.report("te_rope_gqa32", got, want, secs, n)

    def p_mm_then_rows(self):
        """Column-sharded matmul -> barrier -> row-split consumer reading every column."""
        ue = self.ue
        x = torch.randn(ue.NI, ue.D).to(torch.bfloat16)
        ue.dma_to_accelerator_memory(ue.HN, x)
        ue.dma_to_accelerator_memory(ue.QN, torch.zeros(ue.NI, ue.D, dtype=torch.bfloat16))
        def emit():
            ue._mm(ue.NI, ue.D, ue.D, ue.HN, ue.Q, "IMG", ue.dbl_B(0, "img_q"))
            ue._elt(ue.NI, ue.D, ue.Q, None, ue.QN, UE_MODE.MUL_BROADCAST, "IMG", scalar=1.0)
            ue._rms(ue.NI * ue.NH, ue.HD, ue.Q, ue.KN, ue.dbl_w(0, "img_norm_q"), "MH_IMG")
        secs, n = self.run("mm_then_rows", emit)
        w = self.dbl_weight(0, "img_q")
        want = x.float() @ w.float().T
        got = ue.dma_from_accelerator_memory(ue.QN, (ue.NI, ue.D))
        self.report("mm_then_rows_eltwise", got, want, secs, n)
        g = self.dbl_vec(0, "img_norm_q")
        want2 = ref._rms_norm(want.to(torch.bfloat16).view(ue.NI, ue.NH, ue.HD), g, 1e-6).reshape(ue.NI, ue.D)
        got2 = ue.dma_from_accelerator_memory(ue.KN, (ue.NI, ue.D))
        self.report("mm_then_rows_rms", got2, want2, secs, n)

    def p_rows_then_mm(self):
        """Row-split producer (LN) -> barrier -> column-sharded matmul reading every row."""
        ue = self.ue
        x = (torch.randn(ue.NI, ue.D) * 2).to(torch.bfloat16)
        ue.dma_to_accelerator_memory(ue.H[0], x)
        ue.dma_to_accelerator_memory(ue.mod_addr("img", 0, 1, True), torch.ones(ue.D, dtype=torch.bfloat16))
        ue.dma_to_accelerator_memory(ue.mod_addr("img", 0, 0), torch.zeros(ue.D, dtype=torch.bfloat16))
        def emit():
            ue._ln_mod(ue.NI, ue.H[0], ue.HN, ue.mod_addr("img", 0, 1, True), ue.mod_addr("img", 0, 0), "IMG")
            ue._mm(ue.NI, ue.D, ue.D, ue.HN, ue.Q, "IMG", ue.dbl_B(0, "img_q"))
        secs, n = self.run("rows_then_mm", emit)
        w = self.dbl_weight(0, "img_q")
        xn = F.layer_norm(x.float(), (ue.D,), eps=1e-6).to(torch.bfloat16)
        want = xn.float() @ w.float().T
        got = ue.dma_from_accelerator_memory(ue.Q, (ue.NI, ue.D))
        self.report("rows_then_mm", got, want, secs, n)

    def p_double_block(self):
        """One double block on random inputs vs the torch reference (same IF4 weights)."""
        ue = self.ue
        torch.manual_seed(3)
        L, NI, S, D = ue.L, ue.NI, ue.S, ue.D
        h = torch.randn(S, D).to(torch.bfloat16)
        mod = {k: (torch.randn(ue.STEPS, ue.MOD_W[k]) * 0.3).to(torch.bfloat16) for k in ("img", "txt", "single", "out")}
        ue.dma_to_accelerator_memory(ue.H[0], h)
        for k, t in mod.items():
            ue.dma_to_accelerator_memory(ue.MOD[k], t)
            ue.dma_to_accelerator_memory(ue.MOD_P1[k], (t.float() + 1).to(torch.bfloat16))
        def emit():
            ue.emit_gates(0)
            ue.emit_double_block(0, 0, 0)
        secs, n = self.run("double_block", emit, timeout=600)
        # torch reference from the packed weights
        dl = self.meta["double"]
        state = {}
        for k in dl["regions"]:
            if k.endswith("_data"):
                state[k[:-5]] = self.dbl_weight(0, k[:-5])
            elif not k.endswith("_scale"):
                state[k] = self.dbl_vec(0, k)
        geo = ue.geo
        ids = torch.cat([ref.text_position_ids(L), ref.image_position_ids(geo)], 0)
        cos, sin = ref.rope_cos_sin(ids, geo.axes_dims_rope, geo.rope_theta)
        tab = ref.rope_table_rotate_half(cos, sin).float()
        c_t, s_t = tab[:, :ue.HD], tab[:, ue.HD:]
        def rot(xh):   # rotate-half on (S, H, D) with the engine table
            half = ue.HD // 2
            a = xh * c_t[:, None, :]
            b = torch.cat([xh[..., half:] * s_t[:, None, :half], xh[..., :half] * s_t[:, None, half:]], -1)
            return a + b
        img, txt = h[L:].float(), h[:L].float()
        m = {k: [mod[k][0, i * D:(i + 1) * D].float() for i in range(ue.MOD_W[k] // D)] for k in mod}
        def ln(x):
            return F.layer_norm(x, (D,), eps=1e-6)
        def proj(x, name):
            return x @ state[name].float().T
        def qkn(x, name):
            return ref._rms_norm(x.view(-1, ue.NH, ue.HD), state[name], 1e-6).float()
        xs = {"txt": ln(txt) * (1 + m["txt"][1]) + m["txt"][0], "img": ln(img) * (1 + m["img"][1]) + m["img"][0]}
        q = torch.cat([qkn(proj(xs["txt"], "txt_q"), "txt_norm_q"), qkn(proj(xs["img"], "img_q"), "img_norm_q")], 0)
        k = torch.cat([qkn(proj(xs["txt"], "txt_k"), "txt_norm_k"), qkn(proj(xs["img"], "img_k"), "img_norm_k")], 0)
        v = torch.cat([proj(xs["txt"], "txt_v"), proj(xs["img"], "img_v")], 0)
        q, k = rot(q), rot(k)
        o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.view(S, ue.NH, ue.HD).transpose(0, 1)[None])[0]
        o = o.transpose(0, 1).reshape(S, D)
        outs = {}
        for sname, r0, M, hh in (("txt", 0, L, txt), ("img", L, NI, img)):
            a = proj(o[r0:r0 + M], f"{sname}_o")
            hh = hh + m[sname][2] * a
            x2 = ln(hh) * (1 + m[sname][4]) + m[sname][3]
            ff = proj(F.silu(proj(x2, f"{sname}_ff_gate")) * proj(x2, f"{sname}_ff_up"), f"{sname}_ff_out")
            outs[sname] = hh + m[sname][5] * ff
        want = torch.cat([outs["txt"], outs["img"]], 0)
        got = ue.dma_from_accelerator_memory(ue.H[0], (S, D))
        self.report("double_block_0", got, want, secs, n, gate=20)

    def sgl_weight(self, i: int, name: str) -> torch.Tensor:
        dl = self.meta["single"]
        base = dl["base"] + i * dl["stride"]
        data, rd = self.region(base, dl["regions"], name + "_data")
        scale, _ = self.region(base, dl["regions"], name + "_scale")
        N, K = rd["shape"]
        return quant_lib.dequant(self.meta["precision"], data, scale, N, K, block_size=self.meta["block_size"])

    def sgl_vec(self, i: int, name: str) -> torch.Tensor:
        dl = self.meta["single"]
        data, rd = self.region(dl["base"] + i * dl["stride"], dl["regions"], name)
        return torch.frombuffer(bytearray(data), dtype=torch.bfloat16).clone()

    def _rope_tab(self):
        ue = self.ue
        ids = torch.cat([ref.text_position_ids(ue.L), ref.image_position_ids(ue.geo)], 0)
        cos, sin = ref.rope_cos_sin(ids, ue.geo.axes_dims_rope, ue.geo.rope_theta)
        tab = ref.rope_table_rotate_half(cos, sin).float()
        c_t, s_t = tab[:, :ue.HD], tab[:, ue.HD:]
        half = ue.HD // 2
        def rot(xh):
            a = xh * c_t[:, None, :]
            b = torch.cat([xh[..., half:] * s_t[:, None, :half], xh[..., :half] * s_t[:, None, half:]], -1)
            return a + b
        return rot

    def p_single_block(self):
        ue = self.ue
        torch.manual_seed(4)
        S, D, Dm = ue.S, ue.D, ue.Dm
        h = torch.randn(S, D).to(torch.bfloat16)
        mod = (torch.randn(ue.STEPS, 3 * D) * 0.3).to(torch.bfloat16)
        ue.dma_to_accelerator_memory(ue.H[0], h)
        ue.dma_to_accelerator_memory(ue.MOD["single"], mod)
        ue.dma_to_accelerator_memory(ue.MOD_P1["single"], (mod.float() + 1).to(torch.bfloat16))
        def emit():
            ue._broadcast_row(ue.mod_addr("single", 0, 2), ue.GATE_B["single"], S, D)
            ue.emit_single_block(0, 0, 0)
        secs, n = self.run("single_block", emit, timeout=600)
        st = {k: (self.sgl_weight(0, k[:-5]) if k.endswith("_data") else self.sgl_vec(0, k))
              for k in self.meta["single"]["regions"] if not k.endswith("_scale")}
        w = {k[:-5] if k.endswith("_data") else k: v for k, v in st.items()}
        m = [mod[0, i * D:(i + 1) * D].float() for i in range(3)]
        x = F.layer_norm(h.float(), (D,), eps=1e-6) * (1 + m[1]) + m[0]
        proj = lambda a, name: a @ w[name].float().T
        qkn = lambda a, name: ref._rms_norm(a.view(S, ue.NH, ue.HD), w[name], 1e-6).float()
        rot = self._rope_tab()
        q, k = rot(qkn(proj(x, "q"), "norm_q")), rot(qkn(proj(x, "k"), "norm_k"))
        v = proj(x, "v").view(S, ue.NH, ue.HD)
        o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None])[0]
        o = o.transpose(0, 1).reshape(S, D)
        mlp = F.silu(proj(x, "mlp_gate")) * proj(x, "mlp_up")
        y = proj(o, "out_attn") + proj(mlp, "out_mlp")
        want = h.float() + m[2] * y
        got = ue.dma_from_accelerator_memory(ue.H[1], (S, D))
        self.report("single_block_0", got, want, secs, n, gate=20)

    def p_tail(self):
        """norm_out (image rows) -> proj_out -> Euler update -> BN de-normalisation."""
        ue = self.ue
        torch.manual_seed(5)
        L, NI, S, D, C = ue.L, ue.NI, ue.S, ue.D, ue.CIN
        h = torch.randn(S, D).to(torch.bfloat16)
        x0 = torch.randn(NI, C).to(torch.bfloat16)
        mod = (torch.randn(ue.STEPS, 2 * D) * 0.3).to(torch.bfloat16)
        ue.dma_to_accelerator_memory(ue.H[0], h)
        ue.dma_to_accelerator_memory(ue.X_LAT[0], x0)
        ue.dma_to_accelerator_memory(ue.MOD["out"], mod)
        ue.dma_to_accelerator_memory(ue.MOD_P1["out"], (mod.float() + 1).to(torch.bfloat16))
        dt = float(ue.sigmas[1] - ue.sigmas[0])
        def emit():
            ue._ln_mod(NI, ue.H[0] + L * D * BPE, ue.HN_IMG, ue.mod_addr("out", 0, 0, True), ue.mod_addr("out", 0, 1), "IMG")
            ue._mm(NI, D, C, ue.HN_IMG, ue.V_PRED, "IMG", ue.nl_B("proj_out"))
            ue._elt(NI, C, ue.V_PRED, None, ue.V_SCALED, UE_MODE.MUL_BROADCAST, "IMG", scalar=dt)
            ue._elt(NI, C, ue.X_LAT[0], ue.V_SCALED, ue.X_LAT[1], UE_MODE.ELTWISE_ADD, "IMG")
            ue._elt(NI, C, ue.X_LAT[1], ue.BN_STD_B, ue.V_SCALED, UE_MODE.ELTWISE_MUL, "IMG")
            ue._elt(NI, C, ue.V_SCALED, ue.BN_MEAN_B, ue.X_DENORM, UE_MODE.ELTWISE_ADD, "IMG")
        secs, n = self.run("tail", emit)
        scale, shift = mod[0, :D].float(), mod[0, D:].float()
        hn = F.layer_norm(h[L:].float(), (D,), eps=1e-6) * (1 + scale) + shift
        v = hn @ self.nl_bf16("proj_out").float().T
        got_v = ue.dma_from_accelerator_memory(ue.V_PRED, (NI, C))
        self.report("tail_proj_out", got_v, v, secs, n)
        x1 = x0.float() + dt * v
        got_x1 = ue.dma_from_accelerator_memory(ue.X_LAT[1], (NI, C))
        self.report("tail_euler", got_x1, x1, secs, n)
        std = ue.dma_from_accelerator_memory(ue.BN_STD_B, (NI, C)).float()
        mean = ue.dma_from_accelerator_memory(ue.BN_MEAN_B, (NI, C)).float()
        got_d = ue.dma_from_accelerator_memory(ue.X_DENORM, (NI, C))
        self.report("tail_denorm", got_d, x1 * std + mean, secs, n)

    def p_full_step(self):
        """The whole DiT stream (STEPS steps) vs the reference, stage by stage."""
        ue = self.ue
        cfg = T.load_config()
        snap = ref.snapshot_dir(cfg["paths"]["hf_model_repo"])
        geo = ue.geo
        q = cfg["quantization"]
        state = ref.load_safetensors_dir(os.path.join(snap, "transformer"))
        qkeys = [k for k, v in state.items() if v.dim() == 2 and k not in set(q["bf16_keys"])]
        state = ref.fake_quantize_state(state, qkeys, q["precision"], q["block_size"])
        dit = ref.Flux2Transformer(state, geo)
        torch.manual_seed(6)
        emb = (torch.randn(1, ue.L, ue.J) * 0.5).to(torch.bfloat16)
        x0 = ref.initial_noise(geo, 0)
        ue.dma_to_accelerator_memory(ue.TXT_EMB_IN, emb[0])
        ue.dma_to_accelerator_memory(ue.X_LAT[0], x0[0])
        secs, n = self.run("full_step", ue.emit_dit_stream, timeout=1200)
        ids = torch.cat([ref.text_position_ids(ue.L), ref.image_position_ids(geo)], 0)
        cos, sin = ref.rope_cos_sin(ids, geo.axes_dims_rope, geo.rope_theta)
        tr = {}
        v = dit.forward(x0, emb, float(ue.sigmas[0]), cos, sin, trace=tr)
        L, S, D = ue.L, ue.S, ue.D
        self.report("fs_temb", ue.dma_from_accelerator_memory(ue.TEMB, (ue.STEPS, D))[:1], tr["temb"], secs, n, gate=20)
        for k in ("img", "txt", "single", "out"):
            got = ue.dma_from_accelerator_memory(ue.MOD[k], (ue.STEPS, ue.MOD_W[k]))[:1]
            self.report(f"fs_mod_{k}", got, torch.cat(tr["mod"][k], 1), secs, n, gate=20)
        self.report("fs_txt_embed", ue.dma_from_accelerator_memory(ue.TXT0, (L, D)), tr["txt_embed"][0], secs, n, gate=20)
        self.report("fs_norm_out", ue.dma_from_accelerator_memory(ue.HN_IMG, (ue.NI, D)), tr["norm_out"][0], secs, n, gate=15)
        self.report("fs_velocity", ue.dma_from_accelerator_memory(ue.V_PRED, (ue.NI, ue.CIN)), v[0], secs, n, gate=15)
        # the residual stream after the last block (steps==1: H[cur] with cur = N_SINGLE % 2)
        cur = ue.N_SINGLE % 2
        hfin = ue.dma_from_accelerator_memory(ue.H[cur], (S, D))
        self.report("fs_h_final", hfin, tr[f"single_{ue.N_SINGLE - 1}"][0], secs, n, gate=15)
        self.report("fs_h_final_txt_rows", hfin[:L], tr[f"single_{ue.N_SINGLE - 1}"][0, :L], secs, n, gate=15)


PROBES = ["dram_copy", "broadcast_row", "eltwise", "ln_mod", "matmul_bf16", "matmul_if4", "rms_heads",
          "rope", "attention", "silu", "scatter", "conditioning", "text_layer_rope", "mm_then_rows", "rows_then_mm", "double_block", "single_block", "tail", "full_step"]


def main():
    cfg = T.load_config()
    m, dfl = cfg["model"], cfg["defaults"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--steps", type=int, default=m["num_inference_steps"])
    ap.add_argument("--dev", default=dfl["dev"])
    ap.add_argument("--only", default=None, help="comma-separated probe names")
    ap.add_argument("--engines", type=int, default=1, help="engine count: every probe runs through the sharded streams")
    args = ap.parse_args()
    geo = ref.Flux2Geometry(args.height, args.width, args.max_seq_len, args.steps)
    paths = T._paths(cfg)
    user_dma_core.set_dma_device(args.dev)
    user_dma_core.configure_clock_from_hardware()
    ue = T.Flux2KleinEngine(cfg, geo, num_engines=args.engines)
    with open(paths["params_dit_meta"]) as f:
        meta = json.load(f)
    vae_w = torch.load(paths["vae_weights"], map_location="cpu")
    ue.load_constants(vae_w)
    ue._dma_file(paths["params_dit_bin"], ue.params_base, "DiT weights")
    pr = Probe(ue, paths, meta)
    names = args.only.split(",") if args.only else PROBES
    for nm in names:
        print(f"[probe] {nm}", flush=True)
        try:
            getattr(pr, "p_" + nm)()
        except Exception as e:  # keep going; report at the end
            print(f"  ERROR {nm}: {e!r}", flush=True)
            if ue.sched is not None and getattr(ue.sched, "_program_open", False):
                ue.sched.abort_program()
            if ue.is_capture_on:
                ue.stop_capture()
                ue.clear_capture_buffer()
            pr.results[nm] = {"ok": False, "error": repr(e)}
    bad = [k for k, v in pr.results.items() if not v.get("ok")]
    print("PROBE_RESULT: " + json.dumps({"failed": bad, "results": pr.results}))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
