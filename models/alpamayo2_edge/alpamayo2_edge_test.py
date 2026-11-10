#!/usr/bin/env python3
"""Alpamayo 2 Edge on the accelerator (FPGA path).

Implemented so far: the SigLIP2 vision encoder (27 layers x 1152, 16 heads x 72) and the
projector (LayerNorm -> 2x2 merge -> 4608 -> 11520 -> GELU -> 2048). LM prefill / decode and the
action expert are not ported yet.

Vision encoder layout
  * Every image is its own attention sequence. The driving input has 22 images in 2 shapes
    (20x36 = 720 patches, 14x26 = 364), so there is ONE compiled encoder program per shape,
    working on fixed scratch buffers, plus two tiny programs per image:
      pre_i   patch embed (768 -> 1152, bias) + learned position table -> layer-0 input
      post_i  copy the post-layernorm rows into the image's slot of VIS_OUT
    VIS_OUT is the packed (11568, 1152) tensor in the processor's 2x2 block-major order, so the
    projector's 2x2 merge is a plain reshape of four adjacent rows.
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

Usage (from models/):
  python alpamayo2_edge/alpamayo2_edge_test.py --staged --image-index 3   # op by op, one image
  python alpamayo2_edge/alpamayo2_edge_test.py --images 2                 # first 2 images
  python alpamayo2_edge/alpamayo2_edge_test.py                            # all 22 images
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
from user_dma_core import DRAM_INSTRUCTION_ADDR, UE_MODE, UE_VECTOR_SIZE, UnifiedEngine
from nn_lib import eltwise_add_core_dram, layer_norm_core_dram_post_add

from alpamayo2_edge_cpu_reference import (  # host oracle + weight loading; no hardware in there
    PROJ, VIS, Alpamayo2EdgeRef, cos_sim, resolve_model_dir, snr_db,
)

with open(os.path.join(_HERE, "alpamayo2_edge_config.json")) as _f:
    CFG = json.load(_f)

V = CFG["vision"]
VH, VN, VD, VDP = V["hidden_size"], V["num_heads"], V["head_dim"], V["head_dim_padded"]
VI, VIP, VDEPTH, PATCH_DIM = V["intermediate_size"], V["intermediate_size_padded"], V["num_layers"], V["patch_dim"]
C = CFG["connector"]
PJ_IN, PJ_HID, PJ_OUT, MERGE2 = C["input_size"], C["hidden_size"], C["output_size"], C["pixel_shuffle_scale_factor"] ** 2
BPE = 2
SEQ_ALIGN = 128            # flash score rows want a 128-element pitch (LocateAnything MoonViT)
MASK_VALUE = -1e38         # proven flash-bias mask value


def init_hang_prevention(ue):
    """5-step hardware clear. MUST stay identical to run_from_bin.py (see skill gotchas)."""
    halt = bytes(64)
    ue.dma_write(ue.h2c_device, DRAM_INSTRUCTION_ADDR, halt, len(halt))
    print("[Init] HALT written to instruction DRAM base")


def _pad_seq(n):
    return ((n + SEQ_ALIGN - 1) // SEQ_ALIGN) * SEQ_ALIGN


class Alpamayo2Edge_UnifiedEngine(UnifiedEngine):
    def __init__(self, dev=None):
        lay = CFG["dram_layout"]
        kw = {k: int(lay[k], 16) for k in ("params_dram_base", "tensor_dram_base", "program_dram_base")}
        if dev:
            kw["device_name"] = dev
        # Built before any weight upload: the constructor's DRAM self-test stomps 16 KB at DRAM start.
        super().__init__(**kw)
        init_hang_prevention(self)
        self.tensor_limit = kw["program_dram_base"]
        self.prog = {}
        self.hw_us = {}

    # ---------------------------------------------------------------- plumbing

    def _store(self, t):
        t = t.to(torch.bfloat16).contiguous().flatten()
        addr = self.get_params_dram_addr()
        self.dma_to_accelerator_memory(addr, t)
        self.allocate_params_dram(t.numel() * BPE)
        return addr

    def _alloc(self, elements, zero=False):
        addr = self.allocate_tensor_dram(elements * BPE)
        assert self.get_tensor_dram_addr() <= self.tensor_limit, "tensor arena ran into the program arena"
        if zero:
            self.dma_to_accelerator_memory(addr, torch.zeros(elements, dtype=torch.bfloat16))
        return addr

    def _compile(self, name, emit):
        global _SILENT_MODE
        t0 = time.time()
        _SILENT_MODE = True
        try:
            self.start_capture()
            emit()
            self.stop_capture()
            self.generate_instruction_halt()
            size = self.get_capture_instruction_size_bytes()
            addr = self.get_program_dram_addr()
            self.write_captured_instructions_to_dram(addr)
            self.allocate_program_dram(size)
            self.clear_capture_buffer()
        finally:
            _SILENT_MODE = False
        self.prog[name] = addr
        if size > (1 << 20):
            print(f"[compile {name}] {size // 32} instr ({size / 2**20:.1f} MB) @0x{addr:X}, {time.time() - t0:.1f}s")
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
        us = self._exec_addr(self.prog[name], timeout)
        self.hw_us[name] = us
        return us

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
        reinterpretation of (rows, 1152) as (rows / 4, 4608): block-major patch order."""
        b, pw, tokens = self.buf, self.pw, rows // MERGE2

        def emit():
            self.reset_isa_reg_counter()
            r_tok = self.alloc_isa_reg()
            self.generate_instruction_add_set(r_tok, tokens)
            self.layer_norm_core_dram(M=rows, N=VH, A_DRAM_ADDR=b["OUT"], OUTPUT_DRAM_ADDR=b["PJ_LN"],
                                      GAMMA_DRAM_ADDR=pw["ln_w"], BETA_DRAM_ADDR=pw["ln_b"])
            self.matmat_mul_core(M=tokens, K=PJ_IN, N=PJ_HID, A_DRAM_ADDR=b["PJ_LN"], B_DRAM_ADDR=pw["fc1"],
                                 OUTPUT_DRAM_ADDR=b["PJ_HID"], C_DRAM_ADDR=pw["fc1_b"], bias_mode="broadcast_N",
                                 gelu_enable=True, gpr_M_reg=r_tok)
            self.matmat_mul_core(M=tokens, K=PJ_HID, N=PJ_OUT, A_DRAM_ADDR=b["PJ_HID"], B_DRAM_ADDR=pw["fc2"],
                                 OUTPUT_DRAM_ADDR=b["IMG"], C_DRAM_ADDR=pw["fc2_b"], bias_mode="broadcast_N",
                                 gpr_M_reg=r_tok)
        self._compile("projector", emit)

    def projector_read(self, rows):
        return self.dma_from_accelerator_memory(self.buf["IMG"], (rows // MERGE2, PJ_OUT)).float()

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
        b = self.buf = {}
        b["IDENTITY"] = self._alloc(UE_VECTOR_SIZE * UE_VECTOR_SIZE)
        self.dma_to_accelerator_memory(b["IDENTITY"], torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16).flatten())
        b["PIX"] = self._alloc(pix_off * PATCH_DIM)
        b["OUT"] = self._alloc(out_off * VH, zero=True)
        for name in ("EMB", "IO_A", "IO_B", "NORM", "O_PROJ", "RESID", "MLP_OUT", "POST"):
            b[name] = self._alloc(NM * VH)
        b["MLP"] = self._alloc(NM * VIP)
        # q/k/v token-major [N, VN*128] and head-major [VN, N, 128]. Only the 72 real lanes of a
        # slot are ever written, at shape-independent offsets, so the pad lanes are zeroed once.
        for name in ("Q", "K", "V", "Q_HM", "K_HM", "V_HM"):
            b[name] = self._alloc(NM * VN * VDP, zero=True)
        b["ATTN"] = self._alloc(NM * VN * VD, zero=True)
        b["ATTN_SCRATCH"] = self._alloc((VDP + NM) * NM + NM * VDP)
        self.mask = {}
        for im in self.images:
            if im["key"] in self.mask:
                continue
            N, n = im["N"], im["n"]
            bias = torch.zeros(N, N, dtype=torch.bfloat16)
            bias[:, n:] = MASK_VALUE
            self.mask[im["key"]] = self._alloc(N * N)
            self.dma_to_accelerator_memory(self.mask[im["key"]], bias.flatten())
        print(f"vision tensors: {(self.get_tensor_dram_addr() - int(CFG['dram_layout']['tensor_dram_base'], 16)) / 2**20:.0f} MB "
              f"({len(self.images)} images, {self.total_real} patches)")

    def vision_upload_pixels(self, pixel_values):
        """pixel_values (total_real, 768) -> per-image padded slots; pad rows repeat real rows."""
        for im in self.images:
            px = pixel_values[im["out_off"] : im["out_off"] + im["n"]]
            reps = -(-im["N"] // im["n"])
            slot = px.repeat(reps, 1)[: im["N"]].to(torch.bfloat16).contiguous()
            self.dma_to_accelerator_memory(self.buf["PIX"] + im["pix_off"] * PATCH_DIM * BPE, slot.flatten())

    # ---------------------------------------------------------------- vision: op emitters

    def _prime(self, N):
        """Runtime registers every vision program starts with."""
        self.reset_isa_reg_counter()
        self.r_M = self.alloc_isa_reg()
        self.generate_instruction_add_set(self.r_M, N)
        self.r_row = self.alloc_isa_reg()                     # row pitch of the head-padded q/k/v
        self.generate_instruction_add_set(self.r_row, VN * VDP)

    def _mm(self, M, K, N, A, W, OUT, bias, gelu=False, strip=False):
        kw = dict(strip_cols=VD, strip_out_stride=VDP, gpr_out_row_stride_reg=self.r_row) if strip else {}
        self.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=W, OUTPUT_DRAM_ADDR=OUT,
                             C_DRAM_ADDR=bias, bias_mode="broadcast_N", gelu_enable=gelu,
                             gpr_M_reg=self.r_M, **kw)

    def _transpose_10(self, src, dst, d0, d1, last):
        """[d0, d1, last] -> [d1, d0, last] by strided gather (strided scatter is wrong on device)."""
        for i1 in range(d1):
            self.accelerator_memory_to_sram(
                accelerator_dram_address=src + i1 * last * BPE, sram_address=0x00000,
                element_size=d0 * last, stride_bytes_per_chunk=last * BPE, stride_jump_bytes=d1 * last * BPE)
            self.sram_to_accelerator_memory(
                sram_address=0x00000, accelerator_dram_address=dst + i1 * d0 * last * BPE,
                element_size=d0 * last)

    def _attention_head(self, N, q, k, v, bias, out, out_row_stride):
        """One head: softmax(Q K^T / sqrt(128) + bias) V. Q.K^T contracts over the padded 128,
        P.V writes the real 72 lanes at the caller's row pitch (LocateAnything's split_hd)."""
        regs = []

        def reg(val):
            r = self.alloc_isa_reg()
            regs.append(r)
            self.generate_instruction_add_set(r, val)
            return r

        batch_reg, seq_reg, qk_reg, v_reg, stride_reg = reg(N), reg(N), reg(VDP), reg(VD), reg(out_row_stride)
        v_t = self.buf["ATTN_SCRATCH"]
        score = v_t + VDP * N * BPE
        scaled_q = score + N * N * BPE
        self.bf16_transpose_core(M=N, N=VDP, INPUT_DRAM_ADDR=v, OUTPUT_DRAM_ADDR=v_t,
                                 IDENTITY_DRAM_ADDR=self.buf["IDENTITY"], gpr_M_reg=seq_reg)
        self.eltwise_core_dram(M=N, N=VDP, dram_a=q, dram_b=None, dram_out=scaled_q,
                               mode=UE_MODE.MUL_BROADCAST, scalar=1.0 / math.sqrt(VDP), gpr_M_reg=batch_reg)
        self.matmat_mul_core(M=N, K=VDP, N=N, A_DRAM_ADDR=scaled_q, B_DRAM_ADDR=k, OUTPUT_DRAM_ADDR=score,
                             softmax_enable=True, C_DRAM_ADDR=bias, bias_mode="full_matrix",
                             gpr_M_reg=batch_reg, gpr_K_reg=qk_reg, gpr_N_reg=seq_reg)
        self.matmat_mul_core(M=N, K=N, N=VD, A_DRAM_ADDR=score, B_DRAM_ADDR=v_t, OUTPUT_DRAM_ADDR=out,
                             gpr_M_reg=batch_reg, gpr_K_reg=seq_reg, gpr_N_reg=v_reg,
                             gpr_out_row_stride_reg=stride_reg)
        for _ in regs:
            self.release_isa_reg()

    def _encoder_steps(self, key, N):
        """The encoder for one shape as [(label, emit, check_addr, check_shape)]. The compiled
        program and the staged runner both replay this list, so they cannot drift apart."""
        b, steps = self.buf, []
        head_bytes = N * VDP * BPE

        def add(label, emit, addr=None, shape=None):
            steps.append((label, emit, addr, shape))

        for li in range(VDEPTH):
            la = self.vw["layers"][li]
            h_in, h_out = (b["IO_A"], b["IO_B"]) if li % 2 == 0 else (b["IO_B"], b["IO_A"])
            L = f"L{li}"
            add(f"{L}.norm1", lambda h_in=h_in, la=la: self.layer_norm_core_dram(
                M=N, N=VH, A_DRAM_ADDR=h_in, OUTPUT_DRAM_ADDR=b["NORM"],
                GAMMA_DRAM_ADDR=la["ln1_w"], BETA_DRAM_ADDR=la["ln1_b"]), b["NORM"], (N, VH))
            for pj in "qkv":
                dst = b[pj.upper()]
                add(f"{L}.{pj}_proj", lambda pj=pj, la=la, dst=dst: self._mm(
                    N, VH, VN * VD, b["NORM"], la[pj], dst, la[pj + "_b"], strip=True), dst, (N, VN * VDP))
            for pj in "QKV":
                add(f"{L}.perm_{pj.lower()}", lambda pj=pj: self._transpose_10(
                    b[pj], b[pj + "_HM"], d0=N, d1=VN, last=VDP), b[pj + "_HM"], (VN * N, VDP))
            for h in range(VN):
                last = h == VN - 1
                add(f"{L}.attn_h{h}", lambda h=h: self._attention_head(
                    N, b["Q_HM"] + h * head_bytes, b["K_HM"] + h * head_bytes, b["V_HM"] + h * head_bytes,
                    self.mask[key], b["ATTN"] + h * VD * BPE, out_row_stride=VN * VD),
                    b["ATTN"] if last else None, (N, VN * VD) if last else None)
            add(f"{L}.o_proj", lambda la=la: self._mm(N, VN * VD, VH, b["ATTN"], la["o"], b["O_PROJ"], la["o_b"]),
                b["O_PROJ"], (N, VH))
            add(f"{L}.norm2", lambda h_in=h_in, la=la: layer_norm_core_dram_post_add(
                self, M=N, N=VH, A_DRAM_ADDR=h_in, B_DRAM_ADDR=b["O_PROJ"],
                ADDOUTPUT_DRAM_ADDR=b["RESID"], NORMOUTPUT_DRAM_ADDR=b["NORM"],
                GAMMA_DRAM_ADDR=la["ln2_w"], BETA_DRAM_ADDR=la["ln2_b"]), b["NORM"], (N, VH))
            add(f"{L}.fc1", lambda la=la: self._mm(N, VH, VIP, b["NORM"], la["fc1"], b["MLP"], la["fc1_b"], gelu=True),
                b["MLP"], (N, VIP))
            add(f"{L}.fc2", lambda la=la: self._mm(N, VIP, VH, b["MLP"], la["fc2"], b["MLP_OUT"], la["fc2_b"]),
                b["MLP_OUT"], (N, VH))
            add(f"{L}.out", lambda h_out=h_out: eltwise_add_core_dram(
                self, size=N * VH, A_DRAM_ADDR=b["RESID"], B_DRAM_ADDR=b["MLP_OUT"], OUTPUT_DRAM_ADDR=h_out),
                h_out, (N, VH))
        final = b["IO_A"] if VDEPTH % 2 == 0 else b["IO_B"]
        add("post_norm", lambda: self.layer_norm_core_dram(
            M=N, N=VH, A_DRAM_ADDR=final, OUTPUT_DRAM_ADDR=b["POST"],
            GAMMA_DRAM_ADDR=self.vw["post_w"], BETA_DRAM_ADDR=self.vw["post_b"]), b["POST"], (N, VH))
        return steps

    def _emit_pre(self, im):
        b, N = self.buf, im["N"]
        self._prime(N)
        self._mm(N, PATCH_DIM, VH, b["PIX"] + im["pix_off"] * PATCH_DIM * BPE,
                 self.vw["patch_w"], b["EMB"], self.vw["patch_b"])
        eltwise_add_core_dram(self, size=N * VH, A_DRAM_ADDR=b["EMB"],
                              B_DRAM_ADDR=self.vw["pos"][im["key"]], OUTPUT_DRAM_ADDR=b["IO_A"])

    def _emit_post(self, im, rows_per_chunk=128):
        """POST[:n] -> the image's rows of VIS_OUT (whole rows; 1152 lanes = 18 SRAM rows each)."""
        dst = self.buf["OUT"] + im["out_off"] * VH * BPE
        for r0 in range(0, im["n"], rows_per_chunk):
            take = min(rows_per_chunk, im["n"] - r0)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.buf["POST"] + r0 * VH * BPE,
                                            sram_address=0x00000, element_size=take * VH)
            self.sram_to_accelerator_memory(sram_address=0x00000,
                                            accelerator_dram_address=dst + r0 * VH * BPE, element_size=take * VH)

    # ---------------------------------------------------------------- vision: compile / run

    def vision_compile(self, image_ids):
        t0 = time.time()
        for i in image_ids:
            im = self.images[i]
            name = f"enc_{im['key'][0]}x{im['key'][1]}"
            if name not in self.prog:
                def emit(im=im):
                    self._prime(im["N"])
                    for _, op, _, _ in self._encoder_steps(im["key"], im["N"]):
                        op()
                self._compile(name, emit)
            self._compile(f"pre_{i}", lambda im=im: self._emit_pre(im))
            self._compile(f"post_{i}", lambda im=im: self._emit_post(im))
        used = self.get_program_dram_addr() - int(CFG["dram_layout"]["program_dram_base"], 16)
        print(f"vision compiled in {time.time() - t0:.1f}s, program DRAM {used / 2**20:.1f} MB")

    def vision_run(self, image_ids):
        """Returns {image: hw_us}. Output rows land in buf['OUT']."""
        per_image = {}
        for i in image_ids:
            im = self.images[i]
            us = self._exec(f"pre_{i}") + self._exec(f"enc_{im['key'][0]}x{im['key'][1]}") + self._exec(f"post_{i}")
            per_image[i] = us
            print(f"  image {i:2d} {im['key'][0]}x{im['key'][1]}  hw {us / 1e3:8.1f} ms")
        return per_image

    def vision_read(self, i):
        im = self.images[i]
        return self.dma_from_accelerator_memory(self.buf["OUT"] + im["out_off"] * VH * BPE, (im["n"], VH)).float()

    def vision_run_staged(self, i, ref_embed, ref_layers, ref_out, stop_after=None, timeout=120.0):
        """One image, one op per program: NaN-check every op, SNR-check every layer output.
        ref_*: the host reference with hw_gelu=True for this image (real rows only)."""
        im = self.images[i]
        N, n = im["N"], im["n"]

        def check(label, addr, shape, ref=None):
            t = self.dma_from_accelerator_memory(addr, shape).float()
            real = t[:n] if shape[0] == N else t      # pad rows are excluded from every statistic
            nan, inf = int(torch.isnan(real).sum()), int(torch.isinf(real).sum())
            line = f"  [{label:14s}] {str(tuple(shape)):>13s} nan={nan} inf={inf} absmax={float(real.abs().max()):.4g}"
            if ref is not None:
                line += f"  snr={snr_db(ref, real):6.1f} dB cos={cos_sim(ref, real):.5f}"
            print(line)
            assert nan == 0 and inf == 0, f"NaN/Inf at '{label}'"

        self._run_transient(lambda: self._emit_pre(im), timeout)
        check("embed", self.buf["IO_A"], (N, VH), ref_embed)
        for label, op, addr, shape in self._encoder_steps(im["key"], N):
            def emit(op=op):
                self._prime(N)
                op()
            self._run_transient(emit, timeout)
            if addr is not None:
                ref = None
                if label.endswith(".out"):
                    ref = ref_layers[int(label[1:].split(".")[0])]
                elif label == "post_norm":
                    ref = ref_out
                check(label, addr, shape, ref)
            if label == stop_after:
                print(f"stopped after {label}")
                return
        print("staged run complete")


# ----------------------------------------------------------------------------- driver

def pick_sample(path):
    """A real-clip golden if one has been captured, else the synthetic one (real photographs
    through NVIDIA's processor at the driving resolutions)."""
    if path:
        return Path(path).expanduser()
    root = Path("~/a2e_golden").expanduser()
    have = sorted(d for d in root.glob("*") if (d / "packed.safetensors").exists())
    real = [d for d in have if not d.name.startswith("synthetic")]
    assert have, f"no golden sample under {root}; run utility/capture_golden.py first"
    return (real or have)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", default=CFG["defaults"]["dev"])
    ap.add_argument("--golden", default=None, help="sample dir with packed.safetensors (default: newest real clip, else synthetic)")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--images", type=int, default=None, help="run only the first K images")
    ap.add_argument("--staged", action="store_true", help="one image, op by op, with NaN + per-layer SNR checks")
    ap.add_argument("--image-index", type=int, default=0, help="image for --staged")
    ap.add_argument("--stop-after", default=None, help="--staged: stop after this label, e.g. L0.attn_h0")
    ap.add_argument("--min-snr", type=float, default=20.0, help="gate vs the hw-gelu reference (dB)")
    ap.add_argument("--min-cos", type=float, default=0.99)
    args = ap.parse_args()

    sample = pick_sample(args.golden)
    packed = load_file(str(sample / "packed.safetensors"))
    pixels = packed["tokenized_data.pixel_values"].float()
    grid = packed["tokenized_data.image_grid_thw"].long()
    grids = [(int(h), int(w)) for _, h, w in grid.tolist()]
    sizes = [h * w for h, w in grids]
    starts = [sum(sizes[:i]) for i in range(len(sizes))]
    print(f"sample {sample}: {len(grids)} images, {sum(sizes)} patches")

    ref = Alpamayo2EdgeRef(resolve_model_dir(args.model_dir), prefixes=(VIS, PROJ))
    image_ids = [args.image_index] if args.staged else list(range(len(grids)))[: args.images]

    def reference(i, hw_gelu):
        stages, layers = {}, []
        out = ref.vision_encode(pixels[starts[i] : starts[i] + sizes[i]], grid[i : i + 1],
                                stages=stages, hw_gelu=hw_gelu, layer_outs=layers)
        return stages["vision_embed"], layers, out

    user_dma_core.set_dma_device(args.dev)
    user_dma_core.configure_clock_from_hardware()
    ue = Alpamayo2Edge_UnifiedEngine(dev=args.dev)
    ue.vision_weight_init(ref, sorted(set(grids)))
    ue.vision_tensor_init(grids)
    ue.vision_upload_pixels(pixels)

    if args.staged:
        with torch.no_grad():
            emb, layers, out = reference(args.image_index, hw_gelu=True)
        ue.vision_run_staged(args.image_index, emb, layers, out, stop_after=args.stop_after)
        return

    rows = sum(sizes[i] for i in image_ids)
    ue.projector_init(ref)
    ue.vision_compile(image_ids)
    ue.projector_compile(rows)
    per_image = ue.vision_run(image_ids)
    proj_us = ue._exec("projector")

    print(f"\n{'image':>5} {'grid':>6} {'hw ms':>8} | vs hw-gelu ref: {'snr dB':>7} {'cos':>8} | vs published: {'snr dB':>7} {'cos':>8}")
    ok, hw, pub, got = True, [], [], []
    with torch.no_grad():
        for i in image_ids:
            dev_out = ue.vision_read(i)
            r_hw, r_pub = reference(i, True)[2].float(), reference(i, False)[2].float()
            hw.append(r_hw); pub.append(r_pub); got.append(dev_out)
            s, c = snr_db(r_hw, dev_out), cos_sim(r_hw, dev_out)
            bad = bool(torch.isnan(dev_out).any()) or s < args.min_snr or c < args.min_cos
            ok &= not bad
            print(f"{i:5d} {grids[i][0]:>3}x{grids[i][1]:<2} {per_image[i] / 1e3:8.1f} | {s:23.1f} {c:8.5f} | "
                  f"{snr_db(r_pub, dev_out):21.1f} {cos_sim(r_pub, dev_out):8.5f}{'  FAIL' if bad else ''}")
    hw, pub, got = torch.cat(hw), torch.cat(pub), torch.cat(got)
    print(f"\nall {len(image_ids)} images: vs hw-gelu ref {snr_db(hw, got):.1f} dB cos {cos_sim(hw, got):.5f} | "
          f"vs published {snr_db(pub, got):.1f} dB cos {cos_sim(pub, got):.5f} | "
          f"hw-gelu ref vs published {snr_db(pub, hw):.1f} dB")
    print(f"vision encoder hw time: {sum(per_image.values()) / 1e3:.1f} ms")

    # Projector: isolated (reference projector fed the DEVICE vision output) and end to end.
    with torch.no_grad():
        img = ue.projector_read(rows)
        iso, e2e_hw, e2e_pub = ref.project(got, hw_gelu=True).float(), ref.project(hw, hw_gelu=True).float(), ref.project(pub).float()
    s, c = snr_db(iso, img), cos_sim(iso, img)
    proj_bad = bool(torch.isnan(img).any()) or s < args.min_snr or c < args.min_cos
    ok &= not proj_bad
    print(f"projector {tuple(img.shape)} hw {proj_us / 1e3:.1f} ms: isolated {s:.1f} dB cos {c:.5f}{'  FAIL' if proj_bad else ''} | "
          f"end to end vs hw-gelu ref {snr_db(e2e_hw, img):.1f} dB cos {cos_sim(e2e_hw, img):.5f} | "
          f"vs published {snr_db(e2e_pub, img):.1f} dB cos {cos_sim(e2e_pub, img):.5f}")
    print("PASS" if ok else f"FAIL (gate: snr >= {args.min_snr} dB, cos >= {args.min_cos} vs hw-gelu reference)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
