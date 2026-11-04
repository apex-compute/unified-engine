#!/usr/bin/env python3
"""FLUX.2 [klein] 4B text-to-image on the Apex Unified Engine.

Build + run entry point (the ``qwen3_4b_test.py`` of this model):

  1. export weights   (flux2_klein_4b_weights.py: IF4 Qwen3 + DiT images, IF8 VAE)
  2. compile programs (this file: ONE text-encoder program + ONE 4-step DiT
                       program per engine, captured once into programs.bin)
  3. run              (TE phase -> DiT phase -> VAE phase), compare against the
                       host reference, write the PNG, print TEST_RESULT.

Three phases, N engines:

  TE   Qwen3-4B, 27 layers at M = max_sequence_length (padded, right-pad key
       mask + causal bias). Per layer: RMSNorm -> IF4 q/k/v -> per-head QK
       RMSNorm -> rotate-half RoPE (GQA table) -> head-major permute -> one
       unified_attention_core per query head -> permute back -> o_proj ->
       residual -> RMSNorm -> SwiGLU (silu fused on gate) -> down -> residual.
       The raw residual stream after layers 9/18/27 is scattered straight into
       the DiT's 7680-wide text input buffer (token-major [hs9 | hs18 | hs27]).

  DiT  The 4 Euler steps are statically unrolled (the proven pi05 pattern).
       Per step: context/x embedders, modulated LayerNorm (gamma = 1+scale,
       beta = shift -- the LN core's affine vectors ARE the adaLN modulation,
       no broadcast needed), IF4 projections, per-head QK RMSNorm, joint
       [text; image] RoPE + attention (24 heads), gated residuals (gate
       materialized once per step with a row broadcast), SwiGLU FFs, then
       norm_out / proj_out and the Euler update with the baked dt. All
       timestep / modulation vectors for the 4 steps are computed on-device
       up front from a host sin/cos table (M = 4 matmuls).

  VAE  AutoencoderKLFlux2 decoder as ONE program, the third phase of the
       set (flux2_klein_4b_vae.py): 3x3 convs straight from the padded
       channels-in-lanes maps, 1x1 convs as quantized-B matmuls, GroupNorm
       statistics reduced on the ladder and folded on device, SiLU,
       residual adds, upsample and the mid-block attention, all sharded
       over the engines with flag barriers between ops. One DMA in (latent
       map + constant image), one DMA out (the image). Needs the CONV2D
       queue-config image; ``--vae host`` decodes on the host instead.

Multi-engine (``--engines N``): every op is emitted into each engine's own
program with that engine's share -- matmuls are split over output columns
(each engine reads the same IF4 weight image and writes its column strip into
the full output with the row-stride override), norms / RoPE / eltwise / copies
are split over rows, attention over heads -- and a four-phase flag barrier
(``MultiEngineScheduler.barrier_slice``, the pi05 prefix protocol) follows
every op. With one engine the stream is byte-for-byte the single-engine one.

DRAM: the TE and DiT weight images time-share the params arena (both are
DMA'd in at their phase), tensors persist across phases (prompt embedding and
latents never leave DRAM), per-engine program slots follow the tensor arena,
all addresses stay below the 4 GiB DMA limit.

Usage (repo root):
    python models/flux2_klein_4b/flux2_klein_4b_test.py --prompt "..." [--height 512 --width 512 --max-seq-len 512] [--engines max]
    python models/flux2_klein_4b/flux2_klein_4b_test.py --compile-only      # no device needed
    python models/flux2_klein_4b/flux2_klein_4b_run_from_bin.py --prompt "..."
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import user_dma_core  # noqa: E402
from user_dma_core import (  # noqa: E402
    TYPE, UE_MODE, UE_VECTOR_SIZE, URAM_NEAR_FULL_ELEMENTS, UnifiedEngine,
    CONV_GEOMETRY_QUEUE_CONFIG,
)
import flux2_klein_4b_ref as ref  # noqa: E402
import flux2_klein_4b_weights as W  # noqa: E402
import flux2_klein_4b_vae  # noqa: E402

BPE = 2  # bf16

# The engine cores print a line per emitted op; a 4-step DiT program is ~10^5
# ops, so printing is switched off while compiling (qwen3's _SILENT_MODE).
import builtins  # noqa: E402

_original_print = builtins.print
_SILENT = False


def _quiet_print(*a, **k):
    if not _SILENT:
        _original_print(*a, **k)


builtins.print = _quiet_print


def _log(msg: str) -> None:
    _original_print(msg, flush=True)


def snr_db(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    err = (a - b).pow(2).sum()
    if err == 0:
        return float("inf")
    return float(10 * torch.log10(a.pow(2).sum() / err))


def psnr_u8(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a.float() - b.float()).pow(2).mean()
    return float("inf") if mse == 0 else float(10 * math.log10(255.0 ** 2 / mse))


# ============================================================================
# Engine objects
# ============================================================================

class _OfflineMixin:
    """``offline=True`` turns every DMA / register access into a no-op so the
    programs can be captured without a board."""
    offline = False

    def dma_write(self, device, address, buffer, size):
        if self.offline:
            return size
        return super().dma_write(device, address, buffer, size)

    def dma_read(self, device, address, buffer, size):
        if self.offline:
            return size
        return super().dma_read(device, address, buffer, size)

    def user_write_reg32(self, address, value):
        if self.offline:
            return None
        return super().user_write_reg32(address, value)

    def user_read_reg32(self, address):
        if self.offline:
            return 0
        return super().user_read_reg32(address)

    def write_reg32(self, address, value):
        if self.offline:
            return None
        return super().write_reg32(address, value)

    def read_reg32(self, address):
        if self.offline:
            return 0
        return super().read_reg32(address)

    def start_execute_from_dram(self, instruction_addr=None):
        if self.offline:
            return None
        return super().start_execute_from_dram(instruction_addr)

    def wait_queue(self, timeout_seconds=5.0, poll_interval_seconds=0.001):
        if self.offline:
            return None
        return super().wait_queue(timeout_seconds, poll_interval_seconds)

    def is_queue_busy(self):
        if self.offline:
            return False
        return super().is_queue_busy()


class _WorkerEngine(_OfflineMixin, UnifiedEngine):
    """Engine e > 0: same DRAM, its own register block and program slot."""

    def __init__(self, e: int, params_base: int, tensor_base: int, program_base: int,
                 offline: bool, clock_period_ns: float | None):
        self.offline = offline
        super().__init__(BASE_ADDR=user_dma_core.UE_0_BASE_ADDR + e * 0x10000,
                         params_dram_base=params_base, tensor_dram_base=tensor_base,
                         program_dram_base=program_base, clock_period_ns=clock_period_ns,
                         conv_geometry_mode=CONV_GEOMETRY_QUEUE_CONFIG)
        self.bytes_per_element = BPE
        self.TMP_REG = 1


class Flux2KleinEngine(_OfflineMixin, UnifiedEngine):
    """Compiler + runtime for the three phases (engine 0 / primary)."""

    PROGRAM_SLOT = 0x04000000      # per-engine program slot (64 MiB), 8 slots end at 4 GiB
    FIRST_DYNAMIC_REG = 15

    def __init__(self, cfg: dict, geo: ref.Flux2Geometry, *, num_engines: int = 1,
                 offline: bool = False, clock_period_ns: float | None = None):
        lay = cfg["dram_layout"]
        self.offline = offline
        if offline and user_dma_core.UE_AXI_DATA_WIDTH_BITS is None:
            # No board: emulate the U50 HW_INFO so the AXI-beat / clock helpers the
            # cores assert on have values (queue mode, 256-bit AXI, 8 GiB, 8 engines,
            # 64 lanes, 300 MHz). Only alignment rules depend on it.
            from andromeda_hw_info import decode_hardware_info
            raw = (1 << 31) | (0 << 29) | (2 << 26) | (8 << 21) | (0 << 20) | int(300.0 * 256)
            user_dma_core._apply_hardware_info(decode_hardware_info(raw))
        clk = clock_period_ns if clock_period_ns is not None else (3.333 if offline else None)
        super().__init__(
            params_dram_base=int(lay["params_dram_base"], 16),
            tensor_dram_base=int(lay["tensor_dram_base"], 16),
            program_dram_base=int(lay["program_dram_base"], 16),
            clock_period_ns=clk,
            conv_geometry_mode=CONV_GEOMETRY_QUEUE_CONFIG,
        )
        self.cfg = cfg
        self.geo = geo
        self.bytes_per_element = BPE
        d, te = cfg["dit"], cfg["text_encoder"]
        self.D, self.Dm, self.NH, self.HD = d["hidden"], d["mlp_hidden"], d["heads"], d["head_dim"]
        self.J, self.CIN, self.TC = d["joint_dim"], d["in_channels"], d["time_channels"]
        self.N_DOUBLE, self.N_SINGLE = d["num_double_blocks"], d["num_single_blocks"]
        self.TE_H, self.TE_NH, self.TE_NKV, self.TE_HD = te["hidden"], te["heads"], te["kv_heads"], te["head_dim"]
        self.TE_I, self.TE_L = te["intermediate"], te["num_layers_used"]
        self.TE_TAPS = list(te["tap_layers"])
        self.L, self.NI, self.S = geo.n_txt, geo.n_img, geo.n_joint
        self.STEPS = geo.num_inference_steps
        assert self.L % 64 == 0 and self.NI % 64 == 0, "text and image token counts must be multiples of 64"
        self.TMP_REG = cfg["fixed_isa_regs"]["TMP_REG"]
        self.ne = int(num_engines)
        assert self.ne >= 1
        for nm, v in (("text tokens", self.L), ("image tokens", self.NI), ("joint tokens", self.S),
                      ("DiT heads", self.NH), ("TE q heads", self.TE_NH), ("TE kv heads", self.TE_NKV)):
            assert v % self.ne == 0, f"{nm} ({v}) must be divisible by --engines {self.ne}"
        assert (self.STEPS * self.D // 64) % self.ne == 0
        self.sigmas = ref.flow_match_sigmas(self.NI, self.STEPS)
        self.te_layout = W.plan_te_layout(cfg)
        self.dit_layout = W.plan_dit_layout(cfg)
        self.params_base = self._params_dram_base
        self.flops = {"te": 0, "dit": 0}
        self._flops_acc = 0
        self.timings = {}
        self.weight_init()
        self.tensor_init()
        # worker engines: shared DRAM, own program slot after the primary's
        slot = self.PROGRAM_SLOT
        assert self._program_dram_base + self.ne * slot <= (1 << 32), "program slots exceed the 4 GiB window"
        self.workers = [
            _WorkerEngine(e, self.params_base, self.VAE_TENSOR_BASE,
                          self._program_dram_base + e * slot, offline, clk)
            for e in range(1, self.ne)]
        self.E = [self] + self.workers
        self.sched = None
        if self.ne > 1:
            from multi_engine_shard import MultiEngineScheduler
            self.sched = MultiEngineScheduler(self, num_engines=self.ne, workers=self.workers,
                                              allow_more_than_two_engines=True)
        # emission context
        self.cur = self
        self.e = 0
        self.R: dict[str, int] = {}

    # ---------------- weights (address bookkeeping only) ----------------
    def weight_init(self) -> None:
        """Both weight images start at the params base (phased layout)."""
        base = self.params_base
        self.te_addr = {k: base + v["offset"] for k, v in self.te_layout["layer_regions"].items()}
        self.te_stride = self.te_layout["layer_stride"]
        dl = self.dit_layout
        self.dit_nl = {k: base + v["offset"] for k, v in dl["nonlayer_regions"].items()}
        self.dbl_base, self.dbl_stride = base + dl["double"]["base"], dl["double"]["stride"]
        self.dbl_off = {k: v["offset"] for k, v in dl["double"]["regions"].items()}
        self.sgl_base, self.sgl_stride = base + dl["single"]["base"], dl["single"]["stride"]
        self.sgl_off = {k: v["offset"] for k, v in dl["single"]["regions"].items()}
        self.params_image_size = max(self.te_layout["size"], dl["size"])
        limit = self._tensor_dram_base - self.params_base
        assert self.params_image_size <= limit, (
            f"params image {self.params_image_size / 1e9:.2f} GB exceeds the params arena {limit / 1e9:.2f} GB")
        self.allocate_params_dram(self.params_image_size)

    def te_w(self, layer: int, name: str) -> int:
        return self.te_addr[name] + layer * self.te_stride

    def dbl_w(self, i: int, name: str) -> int:
        return self.dbl_base + i * self.dbl_stride + self.dbl_off[name]

    def sgl_w(self, i: int, name: str) -> int:
        return self.sgl_base + i * self.sgl_stride + self.sgl_off[name]

    def _bq(self, scale_addr: int, data_addr: int) -> dict:
        prec = self.cfg["quantization"]["precision"]
        dtype = {"if4": TYPE.IF4, "int4": TYPE.IF4, "fp4": TYPE.IF4,
                 "if8": TYPE.IF8, "int8": TYPE.IF8, "fp8": TYPE.IF8}[prec]
        return {"B_DRAM_ADDR": data_addr, "is_B_quantized": True, "data_type": dtype,
                "SCALE_DRAM_ADDR": scale_addr}

    def te_B(self, layer: int, name: str) -> dict:
        return self._bq(self.te_w(layer, name + "_scale"), self.te_w(layer, name + "_data"))

    def dbl_B(self, i: int, name: str) -> dict:
        return self._bq(self.dbl_w(i, name + "_scale"), self.dbl_w(i, name + "_data"))

    def sgl_B(self, i: int, name: str) -> dict:
        return self._bq(self.sgl_w(i, name + "_scale"), self.sgl_w(i, name + "_data"))

    def nl_B(self, name: str) -> dict:
        regs = self.dit_layout["nonlayer_regions"]
        if name in regs:                               # bf16 B operand
            return {"B_DRAM_ADDR": self.dit_nl[name], "is_B_quantized": False}
        return self._bq(self.dit_nl[name + "_scale"], self.dit_nl[name + "_data"])

    # ---------------- tensors ----------------
    def tensor_init(self) -> None:
        """Deterministic tensor-arena allocation (compile and run-from-bin
        execute exactly the same sequence so every baked address matches)."""
        D, Dm, L, NI, S, ne = self.D, self.Dm, self.L, self.NI, self.S, self.ne
        A = lambda n_elems: self.allocate_tensor_dram(n_elems * BPE, align_bytes=4096)
        # constants
        self.ZERO = A(max(S, L) * 64)
        self.IDENTITY = A(64 * 64)
        self.LN_ONES = A(D)
        self.LN_INV_N = A(D)
        self.LN_ZEROS = [A(D) for _ in range(ne)]           # per engine (kernel scratch)
        self.TE_ROPE = A(L * 2 * self.TE_HD)
        self.DIT_ROPE = A(S * 2 * self.HD)
        self.SINCOS = A(self.STEPS * self.TC)
        self.BN_STD_B = A(NI * self.CIN)
        self.BN_MEAN_B = A(NI * self.CIN)
        self.TE_BIAS = A(L * L)
        self.DIT_BIAS = A(S * S)
        # text encoder
        TH, TQ, TKV, TI = self.TE_H, self.TE_NH * self.TE_HD, self.TE_NKV * self.TE_HD, self.TE_I
        self.TE_X = [A(L * TH), A(L * TH)]
        self.TE_XN = A(L * TH)
        self.TE_Q, self.TE_K, self.TE_V = A(L * TQ), A(L * TKV), A(L * TKV)
        self.TE_QN, self.TE_KN = A(L * TQ), A(L * TKV)
        self.TE_QH, self.TE_KH, self.TE_VH, self.TE_AH = A(L * TQ), A(L * TKV), A(L * TKV), A(L * TQ)
        self.TE_ATT, self.TE_O, self.TE_RES, self.TE_XN2 = A(L * TQ), A(L * TH), A(L * TH), A(L * TH)
        self.TE_GATE, self.TE_UP, self.TE_MULT, self.TE_DOWN = A(L * TI), A(L * TI), A(L * TI), A(L * TH)
        self.TE_SCRATCH = [A((self.TE_HD + L) * L + L * self.TE_HD) for _ in range(ne)]
        self.TXT_EMB_IN = A(L * self.J)
        # DiT conditioning
        self.T1, self.TEMB, self.TEMB_S = A(self.STEPS * D), A(self.STEPS * D), A(self.STEPS * D)
        self.MOD = {"img": A(self.STEPS * 6 * D), "txt": A(self.STEPS * 6 * D),
                    "single": A(self.STEPS * 3 * D), "out": A(self.STEPS * 2 * D)}
        self.MOD_P1 = {"img": A(self.STEPS * 6 * D), "txt": A(self.STEPS * 6 * D),
                       "single": A(self.STEPS * 3 * D), "out": A(self.STEPS * 2 * D)}
        self.MOD_W = {"img": 6 * D, "txt": 6 * D, "single": 3 * D, "out": 2 * D}
        self.GATE_B = {"img_msa": A(NI * D), "img_mlp": A(NI * D), "txt_msa": A(L * D),
                       "txt_mlp": A(L * D), "single": A(S * D)}
        # DiT activations
        self.TXT0 = A(L * D)
        self.H = [A(S * D), A(S * D)]
        self.HN = A(S * D)
        self.Q, self.K, self.V = A(S * D), A(S * D), A(S * D)
        self.QN, self.KN = A(S * D), A(S * D)
        self.QH, self.KH, self.VH, self.AH = A(S * D), A(S * D), A(S * D), A(S * D)
        self.ATT, self.AO, self.G, self.Y = A(S * D), A(S * D), A(S * D), A(S * D)
        self.MG, self.MU, self.MM, self.MO = A(S * Dm), A(S * Dm), A(S * Dm), A(S * D)
        self.DIT_SCRATCH = [A((self.HD + S) * S + S * self.HD) for _ in range(ne)]
        self.HN_IMG = A(NI * D)
        self.X_LAT = [A(NI * self.CIN) for _ in range(self.STEPS + 1)]
        self.V_PRED, self.V_SCALED, self.X_DENORM = A(NI * self.CIN), A(NI * self.CIN), A(NI * self.CIN)
        self.VAE_TENSOR_BASE = self._align_up(self.get_tensor_dram_addr(), 4096)
        used = self.get_tensor_dram_usage()
        limit = self._program_dram_base - self._tensor_dram_base
        assert used <= limit, f"tensor arena overflow: {used / 1e6:.0f} MB > {limit / 1e6:.0f} MB"
        self.tensor_usage = used

    def load_constants(self, vae_weights: dict | None = None) -> None:
        """Host-built constant tensors (geometry-dependent, independent of the prompt)."""
        D, L, NI, S = self.D, self.L, self.NI, self.S
        self.dma_to_accelerator_memory(self.ZERO, torch.zeros(max(S, L) * 64, dtype=torch.bfloat16))
        self.dma_to_accelerator_memory(self.IDENTITY, torch.eye(64, dtype=torch.bfloat16))
        self.dma_to_accelerator_memory(self.LN_ONES, torch.ones(D, dtype=torch.bfloat16))
        self.dma_to_accelerator_memory(self.LN_INV_N, torch.full((D,), 1.0 / D, dtype=torch.bfloat16))
        for z in self.LN_ZEROS:
            self.dma_to_accelerator_memory(z, torch.zeros(D, dtype=torch.bfloat16))
        # Qwen3 rotate-half table: [cos | cos | -sin | sin] per position
        te = self.cfg["text_encoder"]
        hd = self.TE_HD
        inv = 1.0 / (te["rope_theta"] ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
        ang = torch.outer(torch.arange(L, dtype=torch.float32), inv)
        c, s = ang.cos(), ang.sin()
        self.dma_to_accelerator_memory(self.TE_ROPE, torch.cat([c, c, -s, s], 1).to(torch.bfloat16))
        # FLUX 4-axis rope (text ids then image ids), interleaved -> rotate-half
        ids = torch.cat([ref.text_position_ids(L), ref.image_position_ids(self.geo)], 0)
        cos, sin = ref.rope_cos_sin(ids, self.geo.axes_dims_rope, self.geo.rope_theta)
        self.dma_to_accelerator_memory(self.DIT_ROPE, ref.rope_table_rotate_half(cos, sin))
        # timestep sinusoids for the steps
        sc = torch.cat([ref.timestep_sinusoid(ref.dit_timestep_bf16(float(self.sigmas[i])), self.TC)
                        for i in range(self.STEPS)], 0)
        self.dma_to_accelerator_memory(self.SINCOS, sc)
        # attention biases: DiT fully bidirectional (zeros); TE filled per prompt
        self.dma_to_accelerator_memory(self.DIT_BIAS, torch.zeros(S, S, dtype=torch.bfloat16))
        # BN de-normalisation broadcast rows (from the VAE statistics)
        if vae_weights is not None:
            bn = vae_weights["bn"]
            eps = self.cfg["vae"]["batch_norm_eps"]
            std = torch.sqrt(bn["running_var"].float() + eps).to(torch.bfloat16)
            self.dma_to_accelerator_memory(self.BN_STD_B, std[None, :].expand(NI, -1).contiguous())
            self.dma_to_accelerator_memory(self.BN_MEAN_B, bn["running_mean"][None, :].expand(NI, -1).contiguous())

    # ---------------- sharding helpers ----------------
    # bring-up knobs: FLUX2_SHARD_ROWS/COLS/HEADS=0 forces that split back onto engine 0
    _SHARD = {k: os.environ.get(f"FLUX2_SHARD_{k}", "1") != "0" for k in ("ROWS", "COLS", "HEADS")}

    def _rows(self, M: int) -> tuple[int, int]:
        """Row block of the current engine: even split when divisible, else engine 0 does all."""
        if self._SHARD["ROWS"] and M % self.ne == 0:
            m = M // self.ne
            return self.e * m, m
        return (0, M) if self.e == 0 else (0, 0)

    def _cols(self, N: int) -> tuple[int, int]:
        n = N // self.ne
        if self._SHARD["COLS"] and N % self.ne == 0 and n % UE_VECTOR_SIZE == 0:
            return self.e * n, n
        return (0, N) if self.e == 0 else (0, 0)

    def _heads(self, H: int) -> tuple[int, int]:
        if self._SHARD["HEADS"] and H % self.ne == 0:
            h = H // self.ne
            return self.e * h, h
        return (0, H) if self.e == 0 else (0, 0)

    def _reg_rows(self, kind: str, M: int) -> int:
        """GPR holding this engine's row count for a row-split op over M rows."""
        if self._SHARD["ROWS"] and M % self.ne == 0 and (kind + "_e") in self.R:
            return self.R[kind + "_e"]
        return self.R[kind]

    def B(self) -> None:
        """Rendezvous after an op whose output other engines read next."""
        if self.sched is not None:
            self.sched.barrier_slice(self.e)

    def _prime_regs(self, program: str) -> None:
        """Fixed GPR map of one program, primed on the CURRENT engine with its own
        row counts (per-engine kinds end in _e; the others hold the full count)."""
        ne, e = self.ne, self.e
        if program == "dit":
            vals = [("TXT_e", self.L // ne), ("IMG_e", self.NI // ne), ("S_e", self.S // ne),
                    ("MH_TXT_e", self.L // ne * self.NH), ("MH_IMG_e", self.NI // ne * self.NH),
                    ("MH_S_e", self.S // ne * self.NH), ("STEPS", self.STEPS),
                    ("SILU_e", self.STEPS * self.D // 64 // ne), ("TXT", self.L), ("IMG", self.NI),
                    ("S", self.S), ("S_B", self.S), ("S_AL", self.S)]
        else:
            vals = [("TE_e", self.L // ne), ("MH_TE_Q_e", self.L // ne * self.TE_NH),
                    ("MH_TE_KV_e", self.L // ne * self.TE_NKV), ("TE", self.L), ("TE_B", self.L), ("TE_AL", self.L)]
        self.R = {"TMP": self.TMP_REG}
        for i, (name, val) in enumerate(vals, start=2):
            self.R[name] = i
            self.cur.generate_instruction_add_set(i, val)
        assert len(self.R) <= self.FIRST_DYNAMIC_REG
        self.cur._isa_reg_counter = self.FIRST_DYNAMIC_REG
        self.cur.reset_inst_ptr_counter()

    # ---------------- op wrappers (all shard-aware, all end with a barrier) ----------------
    def _dram_copy(self, src: int, dst: int, rows: int, width: int) -> None:
        r0, m = self._rows(rows)
        if m:
            n = m * width
            off = r0 * width * BPE
            chunk = min(URAM_NEAR_FULL_ELEMENTS, n)
            for start, take in self.cur.chunk_ranges(n, chunk):
                self.cur.accelerator_memory_to_sram(src + off + start * BPE, 0x00000, take)
                self.cur.sram_to_accelerator_memory(0x00000, dst + off + start * BPE, take)
        self.B()

    def _broadcast_row(self, src: int, dst: int, rows: int, width: int) -> None:
        """Replicate one (1, width) row to ``rows`` consecutive rows (pi05's
        _ae_broadcast_row): one SRAM load, ``rows`` SRAM->DRAM writes."""
        r0, m = self._rows(rows)
        if m:
            self.cur.accelerator_memory_to_sram(src, 0x00000, width)
            for r in range(r0, r0 + m):
                self.cur.sram_to_accelerator_memory(0x00000, dst + r * width * BPE, width)
        self.B()

    def _mm(self, M, K, N, A, OUT, mkind: str, Bw: dict, **kw) -> None:
        """A[M,K] @ B[N,K]^T -> OUT[M,N], split over output columns."""
        c0, n = self._cols(N)
        if n:
            Bw = dict(Bw)
            if Bw.get("is_B_quantized"):
                bits = 4 if Bw["data_type"] == TYPE.IF4 else 8
                Bw["B_DRAM_ADDR"] += (c0 * K * bits) // 8
                Bw["SCALE_DRAM_ADDR"] += (c0 * K // 64) * BPE
            else:
                Bw["B_DRAM_ADDR"] += c0 * K * BPE
            stride_reg = None
            if n != N:
                stride_reg = self.cur.alloc_isa_reg()
                self.cur.generate_instruction_add_set(stride_reg, N)
            f = self.cur.matmat_mul_core(M=M, K=K, N=n, A_DRAM_ADDR=A, OUTPUT_DRAM_ADDR=OUT + c0 * BPE,
                                         gpr_M_reg=self.R[mkind], gpr_out_row_stride_reg=stride_reg,
                                         **Bw, **kw) or 0
            if stride_reg is not None:
                self.cur.release_isa_reg()
            self._flops_acc += f if f else 2 * M * K * n
        self.B()

    def _elt(self, M, N, a, b, out, mode, kind: str, scalar=None) -> None:
        r0, m = self._rows(M)
        if m:
            off = r0 * N * BPE
            self.cur.eltwise_core_dram(M=m, N=N, dram_a=a + off, dram_b=(None if b is None else b + off),
                                       dram_out=out + off, mode=mode, gpr_M_reg=self._reg_rows(kind, M), scalar=scalar)
        self.B()

    def _rms(self, M, N, A, OUT, gamma, kind: str) -> None:
        r0, m = self._rows(M)
        if m:
            off = r0 * N * BPE
            self.cur.rms_norm_core_dram(M=m, N=N, A_DRAM_ADDR=A + off, OUTPUT_DRAM_ADDR=OUT + off,
                                        GAMMA_DRAM_ADDR=gamma, gpr_M_reg=self._reg_rows(kind, M))
        self.B()

    def _ln_mod(self, M, A, OUT, gamma, beta, kind: str) -> None:
        """LayerNorm(no affine) * (1 + scale) + shift == LayerNorm with
        gamma = (1 + scale) vector, beta = shift vector."""
        r0, m = self._rows(M)
        if m:
            off = r0 * self.D * BPE
            self.cur.layer_norm_core_dram(M=m, N=self.D, A_DRAM_ADDR=A + off, OUTPUT_DRAM_ADDR=OUT + off,
                                          GAMMA_DRAM_ADDR=gamma, BETA_DRAM_ADDR=beta, gpr_M_reg=self._reg_rows(kind, M),
                                          ZEROS_DRAM_ADDR=self.LN_ZEROS[self.e], INV_N_DRAM_ADDR=self.LN_INV_N)
        self.B()

    def _rope(self, M, group: int, hd: int, buf: int, table: int, kind: str) -> None:
        r0, m = self._rows(M)
        if m:
            off = r0 * group * hd * BPE
            tab = table + r0 * 2 * hd * BPE
            self.cur.rope_hf_core_dram_gqa(M=m, group_size=group, N=hd, input_dram_addr=buf + off, output_dram_addr=buf + off,
                                           cos_dram_addr=tab, sin_dram_addr=tab + hd * BPE, gpr_M_reg=self._reg_rows(kind, M))
        self.B()

    def _activation(self, M64, A, OUT, act: str, kind: str) -> None:
        """Pointwise activation over a buffer viewed as M64 rows of 64 (identity matmul)."""
        r0, m = self._rows(M64)
        if m:
            off = r0 * 64 * BPE
            self.cur.activation_core(M=m, N=64, A_DRAM_ADDR=A + off, OUTPUT_DRAM_ADDR=OUT + off,
                                     IDENTITY_DRAM_ADDR=self.IDENTITY, activation=act, gpr_M_reg=self._reg_rows(kind, M64))
        self.B()

    def _scatter_rows(self, src, src_row_bytes, dst, dst_row_bytes, rows, elems, kind: str) -> None:
        """Per-token copy of one row into a strided destination (the TE taps)."""
        r0, m = self._rows(rows)
        if m:
            self.cur._emit_pbi_scatter_per_token(
                read_base=src + r0 * src_row_bytes, read_stride_bytes=src_row_bytes,
                write_specs=[(dst + r0 * dst_row_bytes, dst_row_bytes)],
                sram_byte_addr=0x10000, element_count=elems,
                gpr_seq_len=self._reg_rows(kind, rows), template_seq_len=m)
        self.B()

    def _permute_heads(self, n_total: int, g0: int, n: int, rows: int, width: int,
                       token_major: int, head_major: int, to_head_major: bool) -> None:
        """Heads [g0, g0+n) of a token-major [rows, n_total*width] buffer <-> the
        head-major [n_total, rows, width] buffer (bf16_permute_dram_core's strided
        DMA, with the token-major stride taken from the TOTAL head count so a
        shard of heads can be moved on its own)."""
        ue = self.cur
        row_bytes = width * BPE
        inter_jump = n_total * row_bytes
        chunk_rows = max(1, URAM_NEAR_FULL_ELEMENTS // width)
        for g in range(g0, g0 + n):
            inter_base = token_major + g * row_bytes
            group_base = head_major + g * rows * row_bytes
            r = 0
            while r < rows:
                take = min(chunk_rows, rows - r)
                if to_head_major:
                    ue.accelerator_memory_to_sram(accelerator_dram_address=inter_base + r * inter_jump, sram_address=0,
                                                  element_size=take * width, stride_bytes_per_chunk=row_bytes,
                                                  stride_jump_bytes=inter_jump)
                    ue.sram_to_accelerator_memory(sram_address=0, accelerator_dram_address=group_base + r * row_bytes,
                                                  element_size=take * width)
                else:
                    ue.accelerator_memory_to_sram(accelerator_dram_address=group_base + r * row_bytes, sram_address=0,
                                                  element_size=take * width)
                    ue.sram_to_accelerator_memory(sram_address=0, accelerator_dram_address=inter_base + r * inter_jump,
                                                  element_size=take * width, stride_bytes_per_chunk=row_bytes,
                                                  stride_jump_bytes=inter_jump)
                r += take

    def _attention_heads(self, *, seq: int, n_q_heads: int, n_kv_heads: int, hd: int,
                         Q, K, V, QH, KH, VH, AH, OUT, BIAS, SCRATCH, batch_reg: str, aligned_reg: str) -> None:
        """Token-major Q (seq, nq*hd) / K,V (seq, nkv*hd) -> head-major -> one
        unified_attention_core per q head -> token-major OUT (seq, nq*hd).
        Heads are split across engines; K/V permutes split over kv heads."""
        ue = self.cur
        head_bytes = seq * hd * BPE
        q0, nq = self._heads(n_q_heads)
        k0, nk = self._heads(n_kv_heads)
        self._permute_heads(n_q_heads, q0, nq, seq, hd, Q, QH, True)
        self._permute_heads(n_kv_heads, k0, nk, seq, hd, K, KH, True)
        self._permute_heads(n_kv_heads, k0, nk, seq, hd, V, VH, True)
        self.B()
        group = n_q_heads // n_kv_heads
        for h in range(q0, q0 + nq):
            kv = h // group
            f = ue.unified_attention_core(
                batch=seq, aligned_seq_len=seq, head_dim=hd,
                Q_DRAM_ADDR=QH + h * head_bytes, K_DRAM_ADDR=KH + kv * head_bytes,
                V_DRAM_ADDR=VH + kv * head_bytes, BIAS_DRAM_ADDR=BIAS,
                OUTPUT_DRAM_ADDR=AH + h * head_bytes, SCRATCH_DRAM_ADDR=SCRATCH[self.e],
                IDENTITY_DRAM_ADDR=self.IDENTITY,
                gpr_batch_reg=self.R[batch_reg], gpr_aligned_seq_len_reg=self.R[aligned_reg]) or 0
            self._flops_acc += f if f else 4 * seq * seq * hd
        # own heads only: no barrier needed before the permute back
        self._permute_heads(n_q_heads, q0, nq, seq, hd, OUT, AH, False)
        self.B()

    # ============================================================
    # Program 1: text encoder (one engine's stream)
    # ============================================================
    def emit_text_encoder_stream(self) -> None:
        L, H, hd, nh, nkv, I = self.L, self.TE_H, self.TE_HD, self.TE_NH, self.TE_NKV, self.TE_I
        self._prime_regs("te")
        cur, nxt = 0, 1
        for li in range(self.TE_L):
            if self.e == 0 and li % 9 == 0:
                _log(f"    TE layer {li + 1}/{self.TE_L}")
            X, XO = self.TE_X[cur], self.TE_X[nxt]
            self._rms(L, H, X, self.TE_XN, self.te_w(li, "attn_norm"), "TE")
            self._mm(L, H, nh * hd, self.TE_XN, self.TE_Q, "TE", self.te_B(li, "q"))
            self._mm(L, H, nkv * hd, self.TE_XN, self.TE_K, "TE", self.te_B(li, "k"))
            self._mm(L, H, nkv * hd, self.TE_XN, self.TE_V, "TE", self.te_B(li, "v"))
            self._rms(L * nh, hd, self.TE_Q, self.TE_QN, self.te_w(li, "q_norm"), "MH_TE_Q")
            self._rms(L * nkv, hd, self.TE_K, self.TE_KN, self.te_w(li, "k_norm"), "MH_TE_KV")
            self._rope(L, nh, hd, self.TE_QN, self.TE_ROPE, "TE")
            self._rope(L, nkv, hd, self.TE_KN, self.TE_ROPE, "TE")
            self._attention_heads(seq=L, n_q_heads=nh, n_kv_heads=nkv, hd=hd,
                                  Q=self.TE_QN, K=self.TE_KN, V=self.TE_V,
                                  QH=self.TE_QH, KH=self.TE_KH, VH=self.TE_VH, AH=self.TE_AH, OUT=self.TE_ATT,
                                  BIAS=self.TE_BIAS, SCRATCH=self.TE_SCRATCH, batch_reg="TE_B", aligned_reg="TE_AL")
            self._mm(L, nh * hd, H, self.TE_ATT, self.TE_O, "TE", self.te_B(li, "o"))
            self._elt(L, H, X, self.TE_O, self.TE_RES, UE_MODE.ELTWISE_ADD, "TE")
            self._rms(L, H, self.TE_RES, self.TE_XN2, self.te_w(li, "ffn_norm"), "TE")
            self._mm(L, H, I, self.TE_XN2, self.TE_GATE, "TE", self.te_B(li, "gate"), silu_enable=True)
            self._mm(L, H, I, self.TE_XN2, self.TE_UP, "TE", self.te_B(li, "up"))
            self._elt(L, I, self.TE_GATE, self.TE_UP, self.TE_MULT, UE_MODE.ELTWISE_MUL, "TE")
            self._mm(L, I, H, self.TE_MULT, self.TE_DOWN, "TE", self.te_B(li, "down"))
            self._elt(L, H, self.TE_RES, self.TE_DOWN, XO, UE_MODE.ELTWISE_ADD, "TE")
            if (li + 1) in self.TE_TAPS:
                k = self.TE_TAPS.index(li + 1)
                self._scatter_rows(XO, H * BPE, self.TXT_EMB_IN + k * H * BPE, self.J * BPE, L, H, "TE")
            cur, nxt = nxt, cur

    # ============================================================
    # Program 2: DiT, steps unrolled (one engine's stream)
    # ============================================================
    def mod_addr(self, which: str, step: int, chunk: int, plus_one: bool = False) -> int:
        base = self.MOD_P1[which] if plus_one else self.MOD[which]
        return base + (step * self.MOD_W[which] + chunk * self.D) * BPE

    def emit_conditioning(self) -> None:
        """Timestep MLP + shared modulation for all steps at once (M = STEPS)."""
        D, T = self.D, self.TC
        self._mm(self.STEPS, T, D, self.SINCOS, self.T1, "STEPS", self.nl_B("time_l1"), silu_enable=True)
        self._mm(self.STEPS, D, D, self.T1, self.TEMB, "STEPS", self.nl_B("time_l2"))
        self._activation(self.STEPS * D // 64, self.TEMB, self.TEMB_S, "silu", "SILU")
        for which, wname in (("img", "mod_img"), ("txt", "mod_txt"), ("single", "mod_single"), ("out", "norm_out")):
            self._mm(self.STEPS, D, self.MOD_W[which], self.TEMB_S, self.MOD[which], "STEPS", self.nl_B(wname))
            self._elt(self.STEPS, self.MOD_W[which], self.MOD[which], None, self.MOD_P1[which],
                      UE_MODE.ADD_BROADCAST, "STEPS", scalar=1.0)

    def emit_gates(self, step: int) -> None:
        D = self.D
        self._broadcast_row(self.mod_addr("img", step, 2), self.GATE_B["img_msa"], self.NI, D)
        self._broadcast_row(self.mod_addr("img", step, 5), self.GATE_B["img_mlp"], self.NI, D)
        self._broadcast_row(self.mod_addr("txt", step, 2), self.GATE_B["txt_msa"], self.L, D)
        self._broadcast_row(self.mod_addr("txt", step, 5), self.GATE_B["txt_mlp"], self.L, D)
        self._broadcast_row(self.mod_addr("single", step, 2), self.GATE_B["single"], self.S, D)

    def _joint_attention(self) -> None:
        self._attention_heads(seq=self.S, n_q_heads=self.NH, n_kv_heads=self.NH, hd=self.HD,
                              Q=self.QN, K=self.KN, V=self.V, QH=self.QH, KH=self.KH, VH=self.VH,
                              AH=self.AH, OUT=self.ATT, BIAS=self.DIT_BIAS, SCRATCH=self.DIT_SCRATCH,
                              batch_reg="S_B", aligned_reg="S_AL")

    def _rope_joint(self) -> None:
        self._rope(self.S, self.NH, self.HD, self.QN, self.DIT_ROPE, "S")
        self._rope(self.S, self.NH, self.HD, self.KN, self.DIT_ROPE, "S")

    def emit_double_block(self, i: int, step: int, cur: int) -> int:
        D, Dm, L, NI, hd, nh = self.D, self.Dm, self.L, self.NI, self.HD, self.NH
        H, H2 = self.H[cur], self.H[1 - cur]
        streams = (("txt", 0, L, "TXT", "MH_TXT"), ("img", L, NI, "IMG", "MH_IMG"))
        rb = lambda r: r * D * BPE
        for s, r0, M, kM, kMH in streams:
            self._ln_mod(M, H + rb(r0), self.HN + rb(r0), self.mod_addr(s, step, 1, True), self.mod_addr(s, step, 0), kM)
            self._mm(M, D, D, self.HN + rb(r0), self.Q + rb(r0), kM, self.dbl_B(i, f"{s}_q"))
            self._mm(M, D, D, self.HN + rb(r0), self.K + rb(r0), kM, self.dbl_B(i, f"{s}_k"))
            self._mm(M, D, D, self.HN + rb(r0), self.V + rb(r0), kM, self.dbl_B(i, f"{s}_v"))
            self._rms(M * nh, hd, self.Q + rb(r0), self.QN + rb(r0), self.dbl_w(i, f"{s}_norm_q"), kMH)
            self._rms(M * nh, hd, self.K + rb(r0), self.KN + rb(r0), self.dbl_w(i, f"{s}_norm_k"), kMH)
        self._rope_joint()
        self._joint_attention()
        for s, r0, M, kM, kMH in streams:
            self._mm(M, D, D, self.ATT + rb(r0), self.AO + rb(r0), kM, self.dbl_B(i, f"{s}_o"))
            self._elt(M, D, self.AO + rb(r0), self.GATE_B[f"{s}_msa"], self.G + rb(r0), UE_MODE.ELTWISE_MUL, kM)
            self._elt(M, D, H + rb(r0), self.G + rb(r0), H2 + rb(r0), UE_MODE.ELTWISE_ADD, kM)
            self._ln_mod(M, H2 + rb(r0), self.HN + rb(r0), self.mod_addr(s, step, 4, True), self.mod_addr(s, step, 3), kM)
            self._mm(M, D, Dm, self.HN + rb(r0), self.MG + r0 * Dm * BPE, kM, self.dbl_B(i, f"{s}_ff_gate"), silu_enable=True)
            self._mm(M, D, Dm, self.HN + rb(r0), self.MU + r0 * Dm * BPE, kM, self.dbl_B(i, f"{s}_ff_up"))
            self._elt(M, Dm, self.MG + r0 * Dm * BPE, self.MU + r0 * Dm * BPE, self.MM + r0 * Dm * BPE, UE_MODE.ELTWISE_MUL, kM)
            self._mm(M, Dm, D, self.MM + r0 * Dm * BPE, self.MO + rb(r0), kM, self.dbl_B(i, f"{s}_ff_out"))
            self._elt(M, D, self.MO + rb(r0), self.GATE_B[f"{s}_mlp"], self.G + rb(r0), UE_MODE.ELTWISE_MUL, kM)
            self._elt(M, D, H2 + rb(r0), self.G + rb(r0), H + rb(r0), UE_MODE.ELTWISE_ADD, kM)
        return cur

    def emit_single_block(self, i: int, step: int, cur: int) -> int:
        D, Dm, S, hd, nh = self.D, self.Dm, self.S, self.HD, self.NH
        H, H2 = self.H[cur], self.H[1 - cur]
        self._ln_mod(S, H, self.HN, self.mod_addr("single", step, 1, True), self.mod_addr("single", step, 0), "S")
        self._mm(S, D, D, self.HN, self.Q, "S", self.sgl_B(i, "q"))
        self._mm(S, D, D, self.HN, self.K, "S", self.sgl_B(i, "k"))
        self._mm(S, D, D, self.HN, self.V, "S", self.sgl_B(i, "v"))
        self._mm(S, D, Dm, self.HN, self.MG, "S", self.sgl_B(i, "mlp_gate"), silu_enable=True)
        self._mm(S, D, Dm, self.HN, self.MU, "S", self.sgl_B(i, "mlp_up"))
        self._rms(S * nh, hd, self.Q, self.QN, self.sgl_w(i, "norm_q"), "MH_S")
        self._rms(S * nh, hd, self.K, self.KN, self.sgl_w(i, "norm_k"), "MH_S")
        self._rope_joint()
        self._joint_attention()
        self._elt(S, Dm, self.MG, self.MU, self.MM, UE_MODE.ELTWISE_MUL, "S")
        self._mm(S, D, D, self.ATT, self.AO, "S", self.sgl_B(i, "out_attn"))
        self._mm(S, Dm, D, self.MM, self.MO, "S", self.sgl_B(i, "out_mlp"))
        self._elt(S, D, self.AO, self.MO, self.Y, UE_MODE.ELTWISE_ADD, "S")
        self._elt(S, D, self.Y, self.GATE_B["single"], self.G, UE_MODE.ELTWISE_MUL, "S")
        self._elt(S, D, H, self.G, H2, UE_MODE.ELTWISE_ADD, "S")
        return 1 - cur

    def emit_dit_stream(self) -> None:
        D, L, NI, C = self.D, self.L, self.NI, self.CIN
        self._prime_regs("dit")
        self.emit_conditioning()
        self._mm(L, self.J, D, self.TXT_EMB_IN, self.TXT0, "TXT", self.nl_B("context_embedder"))
        for step in range(self.STEPS):
            if self.e == 0:
                _log(f"    DiT step {step + 1}/{self.STEPS}")
            cur = 0
            self._dram_copy(self.TXT0, self.H[cur], L, D)
            self._mm(NI, C, D, self.X_LAT[step], self.H[cur] + L * D * BPE, "IMG", self.nl_B("x_embedder"))
            self.emit_gates(step)
            for i in range(self.N_DOUBLE):
                cur = self.emit_double_block(i, step, cur)
            for i in range(self.N_SINGLE):
                cur = self.emit_single_block(i, step, cur)
            self._ln_mod(NI, self.H[cur] + L * D * BPE, self.HN_IMG, self.mod_addr("out", step, 0, True),
                         self.mod_addr("out", step, 1), "IMG")
            self._mm(NI, D, C, self.HN_IMG, self.V_PRED, "IMG", self.nl_B("proj_out"))
            dt = float(self.sigmas[step + 1] - self.sigmas[step])
            self._elt(NI, C, self.V_PRED, None, self.V_SCALED, UE_MODE.MUL_BROADCAST, "IMG", scalar=dt)
            self._elt(NI, C, self.X_LAT[step], self.V_SCALED, self.X_LAT[step + 1], UE_MODE.ELTWISE_ADD, "IMG")
        self._elt(NI, C, self.X_LAT[self.STEPS], self.BN_STD_B, self.V_SCALED, UE_MODE.ELTWISE_MUL, "IMG")
        self._elt(NI, C, self.V_SCALED, self.BN_MEAN_B, self.X_DENORM, UE_MODE.ELTWISE_ADD, "IMG")

    # ============================================================
    # Compile / load
    # ============================================================
    @staticmethod
    def _emitter_hash() -> str:
        """Hash of this compiler's source: a code change must never replay a stale bin."""
        with open(os.path.abspath(__file__), "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()[:12]

    def program_signature(self) -> str:
        g = self.geo
        key = json.dumps({"h": g.height, "w": g.width, "L": g.max_sequence_length, "steps": g.num_inference_steps,
                          "sigmas": [round(float(x), 7) for x in self.sigmas], "layout": self.cfg["dram_layout"],
                          "q": self.cfg["quantization"], "engines": self.ne, "emitter": self._emitter_hash(),
                          "vae_emitter": flux2_klein_4b_vae.emitter_hash(), "v": 4},
                         sort_keys=True)
        return hashlib.sha1(key.encode()).hexdigest()[:12]

    def _emit_phase(self, name: str, stream_fn) -> tuple[int, int, int, list[int], list[bytes]]:
        """Emit one phase into every engine's stream. The primary's program is a
        segment of its capture; the workers' programs are finalized by the
        scheduler. Returns (primary start, primary size, flops, worker addrs, worker blobs)."""
        base = self.get_program_dram_addr() + self.get_capture_instruction_size_bytes()
        self._flops_acc = 0
        if self.sched is not None:
            self.sched.begin_program()
        for e, ue in enumerate(self.E):
            self.cur, self.e = ue, e
            stream_fn()
            if e == 0:
                ue.generate_instruction_halt()
        self.cur, self.e = self, 0
        flops = self._flops_acc
        size = self.get_program_dram_addr() + self.get_capture_instruction_size_bytes() - base
        self.pad_capture_to_64b_boundary()
        w_addrs, w_blobs = [], []
        if self.sched is not None:
            w_addrs = self.sched.finalize()
            w_blobs = [b"".join(inst.get_bytes() for inst in w.capture_buffer) for w in self.workers]
        self.flops[name] = flops
        return base, size, flops, w_addrs, w_blobs

    def compile_programs(self, bin_path: str, meta_path: str) -> dict:
        global _SILENT
        t0 = time.time()
        self.clear_inst_id()
        self.start_capture()
        self.reset_inst_ptr_counter()
        _SILENT = True
        try:
            _log(f"  compiling text-encoder program ({self.ne} engine(s))...")
            te = self._emit_phase("te", self.emit_text_encoder_stream)
            _log(f"  compiling DiT program ({self.ne} engine(s))...")
            dit = self._emit_phase("dit", self.emit_dit_stream)
            _log(f"  compiling VAE program ({self.ne} engine(s))...")
            vae_plan = self.vae_planner().plan(self.geo.latent_h, self.geo.latent_w)
            vae = self._emit_phase("vae", self.vae.emit)
        finally:
            _SILENT = False
        self.stop_capture()
        assert self.capture_count <= user_dma_core.MAX_DECODER_INSTRUCTIONS
        primary = b"".join(inst.get_bytes() for inst in self.capture_buffer)
        images = [{"engine": 0, "phase": "primary", "addr": self._program_dram_base, "size": len(primary)}]
        blob = bytearray(primary)
        for name, ph in (("te", te), ("dit", dit), ("vae", vae)):
            for wi, (addr, wb) in enumerate(zip(ph[3], ph[4]), start=1):
                images.append({"engine": wi, "phase": name, "addr": addr, "size": len(wb), "offset": len(blob)})
                blob.extend(wb)
        images[0]["offset"] = 0
        self.clear_capture_buffer()
        _SILENT = True
        try:
            self._set_program_cursors(images)
            # the scheduler's per-engine flag-clear program (run once before the first phase),
            # prebuilt so nothing is captured at run time
            flag_clear = []
            for e, eng in enumerate(self.E):
                eng.start_capture()
                eng.generate_instruction_flag_clear()
                eng.generate_instruction_halt()
                eng.stop_capture()
                addr = eng.get_program_dram_addr()
                data = b"".join(inst.get_bytes() for inst in eng.capture_buffer)
                eng.allocate_program_dram(len(data))
                eng.clear_capture_buffer()
                images.append({"engine": e, "phase": "flag_clear", "addr": addr, "size": len(data), "offset": len(blob)})
                blob.extend(data)
                flag_clear.append(addr)
        finally:
            _SILENT = False
        os.makedirs(os.path.dirname(bin_path), exist_ok=True)
        with open(bin_path, "wb") as f:
            f.write(blob)
        meta = {
            "signature": self.program_signature(),
            "engines": self.ne,
            "geometry": {"height": self.geo.height, "width": self.geo.width,
                         "max_sequence_length": self.geo.max_sequence_length, "steps": self.STEPS},
            "sigmas": [float(x) for x in self.sigmas],
            "images": images,
            "te_program_start_addr": te[0], "te_program_size": te[1], "te_flops": te[2], "te_worker_prog_addrs": te[3],
            "dit_program_start_addr": dit[0], "dit_program_size": dit[1], "dit_flops": dit[2], "dit_worker_prog_addrs": dit[3],
            "vae_program_start_addr": vae[0], "vae_program_size": vae[1], "vae_flops": vae[2], "vae_worker_prog_addrs": vae[3],
            "tensor_dram_usage": self.tensor_usage, "params_image_size": self.params_image_size,
            "vae": vae_plan,
            "flag_clear_addrs": flag_clear,
            "compile_seconds": round(time.time() - t0, 1),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=1)
        per_engine = (max((im["size"] for im in images[1:] if im["phase"] in ("te", "dit")), default=0)) / 1e6
        _log(f"  programs: TE {te[1] / 1e6:.1f} MB ({te[2] / 1e12:.2f} TFLOP), DiT {dit[1] / 1e6:.1f} MB "
             f"({dit[2] / 1e12:.2f} TFLOP), worker programs up to {per_engine:.1f} MB, "
             f"VAE {vae[1] / 1e6:.1f} MB ({vae[2] / 1e12:.2f} TFLOP, {len(vae_plan['steps'])} steps, "
             f"const image {vae_plan['const_size'] / 1e6:.0f} MB, arena {(vae_plan['arena_end'] - vae_plan['const_base']) / 1e6:.0f} MB), "
             f"bin {len(blob) / 1e6:.1f} MB, compiled in {meta['compile_seconds']}s")
        return meta

    def vae_planner(self):
        """The VAE phase planner for this geometry (deterministic: compile and run agree)."""
        self.vae = flux2_klein_4b_vae.Flux2VaeOneShot(self, self.cfg)
        return self.vae

    def run_vae(self, meta: dict, latent: torch.Tensor, vae_weights: dict) -> torch.Tensor:
        """(32, h, w) latent -> (3, 8h, 8w) image: one DMA in (constant image + latent map),
        the VAE program, one DMA out."""
        vm = meta["vae"]
        plan = self.vae_planner().plan(self.geo.latent_h, self.geo.latent_w)
        for k in ("const_base", "const_size", "input", "output", "output_bytes", "arena_end"):
            assert plan[k] == vm[k], f"VAE plan {k} differs from the compiled programs (recompile)"
        t0 = time.time()
        img = self.vae.const_image(vae_weights)
        n = self.dma_write(self.h2c_device, vm["const_base"], img, len(img))
        assert n == len(img)
        lat = self.vae.pack_latent(latent)
        assert lat.numel() * 2 == vm["input_bytes"]
        self.dma_to_accelerator_memory(vm["input"], lat)
        self.timings["vae_load_s"] = time.time() - t0
        t0 = time.time()
        self._execute_phase(meta, "vae", getattr(self, "phase_timeout", 4 * 3600.0))
        self.timings["vae_s"] = time.time() - t0
        flat = self.dma_from_accelerator_memory(vm["output"], (vm["output_bytes"] // 2,))
        return self.vae.unpack_output(flat)

    def _set_program_cursors(self, images: list[dict]) -> None:
        """Every engine's next program address = just past its last image."""
        ends = {}
        for im in images:
            ends[im["engine"]] = max(ends.get(im["engine"], 0), im["addr"] + im["size"])
        for e, end in ends.items():
            self.E[e]._next_program_dram_addr = self._align_up(end, 64)

    def load_programs(self, bin_path: str, meta_path: str) -> dict:
        with open(meta_path) as f:
            meta = json.load(f)
        with open(bin_path, "rb") as f:
            blob = f.read()
        return self.load_program_images(blob, meta)

    def load_program_images(self, blob, meta: dict, check_signature: bool = True) -> dict:
        """DMA every program image of ``meta`` (the primary's and the workers')
        from ``blob`` (bytes-like / ndarray) to its baked address."""
        if check_signature and meta.get("signature") != self.program_signature():
            raise RuntimeError("program signature does not match this geometry/config/engine count/compiler; "
                               "recompile (flux2_klein_4b_test.py --compile-only ... --force-compile)")
        for im in meta["images"]:
            ue = self.E[im["engine"]]
            data = blob[im["offset"]:im["offset"] + im["size"]]
            assert len(data) == im["size"]
            n = ue.dma_write(ue.h2c_device, im["addr"], data, len(data))
            assert n == len(data), f"program image for engine {im['engine']}: DMA wrote {n} of {len(data)} bytes"
        self._set_program_cursors(meta["images"])
        for e, addr in enumerate(meta.get("flag_clear_addrs", [])):
            self.E[e]._flag_clear_addr = addr            # MultiEngineScheduler.preclear_flags launches it by address
        _log(f"    loaded {len(meta['images'])} program image(s), {sum(im['size'] for im in meta['images']) / 1e6:.1f} MB")
        return meta

    # ============================================================
    # Run
    # ============================================================
    def _dma_blob(self, data, addr: int, label: str) -> None:
        """DMA a weight image given as a path, bytes-like or uint8 ndarray."""
        t0 = time.time()
        if isinstance(data, str):
            with open(data, "rb") as f:
                data = f.read()
        n = self.dma_write(self.h2c_device, addr, data, len(data))
        assert n == len(data), f"{label}: DMA wrote {n} of {len(data)} bytes"
        _log(f"  [{label}] {len(data) / 1e9:.2f} GB -> 0x{addr:X} in {time.time() - t0:.1f}s")

    _dma_file = _dma_blob

    def _execute_phase(self, meta: dict, name: str, timeout: float) -> None:
        if self.sched is not None:
            if not getattr(self, "_flags_precleared", False):
                self.sched.preclear_flags()
                self._flags_precleared = True
            self.sched.start_workers(meta[f"{name}_worker_prog_addrs"])
        self.program_execute(meta[f"{name}_program_start_addr"], timeout=timeout, flops=meta[f"{name}_flops"])
        if self.sched is not None:
            for w in self.workers:
                w.wait_queue(120.0)

    def run_text_encoder(self, meta: dict, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                         embed_tokens: torch.Tensor, params_te_bin: str) -> torch.Tensor:
        L = self.L
        assert input_ids.shape[-1] == L
        self._dma_blob(params_te_bin, self.params_base, "TE weights")
        emb = embed_tokens[input_ids.reshape(-1)].to(torch.bfloat16)
        self.dma_to_accelerator_memory(self.TE_X[0], emb)
        valid = int(attention_mask.sum())
        bias = torch.full((L, L), float("-inf"), dtype=torch.bfloat16)
        bias.masked_fill_(torch.tril(torch.ones(L, L, dtype=torch.bool)), 0.0)
        bias[:, valid:] = float("-inf")
        self.dma_to_accelerator_memory(self.TE_BIAS, bias)
        self.dma_to_accelerator_memory(self.TXT_EMB_IN, torch.zeros(L, self.J, dtype=torch.bfloat16))
        t0 = time.time()
        self._execute_phase(meta, "te", min(3600.0, getattr(self, "phase_timeout", 3600.0)))
        self.timings["te_s"] = time.time() - t0
        return self.dma_from_accelerator_memory(self.TXT_EMB_IN, (1, L, self.J))

    def run_dit(self, meta: dict, noise: torch.Tensor, params_dit_bin: str,
                prompt_embeds: torch.Tensor | None = None) -> torch.Tensor:
        self._dma_blob(params_dit_bin, self.params_base, "DiT weights")
        if prompt_embeds is not None:                       # bypass the TE phase (cached / host embeds)
            self.dma_to_accelerator_memory(self.TXT_EMB_IN, prompt_embeds.reshape(self.L, self.J).to(torch.bfloat16))
        self.dma_to_accelerator_memory(self.X_LAT[0], noise.reshape(self.NI, self.CIN).to(torch.bfloat16))
        t0 = time.time()
        self._execute_phase(meta, "dit", getattr(self, "phase_timeout", 4 * 3600.0))
        self.timings["dit_s"] = time.time() - t0
        # every step's latent buffer persists: x_0 (noise) .. x_STEPS
        self.step_latents = [self.dma_from_accelerator_memory(self.X_LAT[s], (1, self.NI, self.CIN))
                             for s in range(self.STEPS + 1)]
        self.latents_raw = self.step_latents[-1]
        return self.dma_from_accelerator_memory(self.X_DENORM, (1, self.NI, self.CIN))


# ============================================================================
# Host side
# ============================================================================

def load_config() -> dict:
    with open(os.path.join(HERE, "flux2_klein_4b_config.json")) as f:
        return json.load(f)


def _paths(cfg: dict) -> dict:
    return {k: os.path.join(HERE, v) for k, v in cfg["paths"].items()
            if k.endswith(("_bin", "_meta", "vae_weights"))}


def load_embed_tokens(cfg: dict, paths: dict) -> torch.Tensor:
    te = cfg["text_encoder"]
    with open(paths["embed_tokens_bin"], "rb") as f:
        raw = f.read()
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(te["vocab"], te["hidden"])


def tokenize(snap: str, prompt: str, L: int):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(snap, "tokenizer"))
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=L)
    return enc["input_ids"], enc["attention_mask"]


def quantized_reference(cfg: dict, geo: ref.Flux2Geometry, snap: str, prompt: str, seed: int, device: str,
                        prompt_embeds: torch.Tensor | None = None, keep_steps: bool = True) -> ref.ReferenceResult:
    """Host run with the SAME IF4 weights the board uses (fake-quantized) --
    the numeric oracle the FPGA output is scored against."""
    q = cfg["quantization"]
    if prompt_embeds is None:
        # Text encoder first (8 GB bf16): fake-quantize the projections IN PLACE on the
        # CPU, then move to the device, and free it before the DiT weights go there.
        te = ref.Qwen3TextEncoder(snap, device="cpu", num_layers=geo.text_layers_used)
        with torch.no_grad():
            for n, p in te.model.named_parameters():
                if n.endswith(".weight") and "proj" in n and p.dim() == 2:
                    p.copy_(ref.fake_quantize_tensor(p, q["precision"], q["block_size"]))
        te.model.to(device)
        te.device = device
        prompt_embeds = te(prompt, geo.max_sequence_length, geo.text_tap_layers)
        del te
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    state = ref.load_safetensors_dir(os.path.join(snap, "transformer"))
    qkeys = [k for k, v in state.items() if v.dim() == 2 and k not in set(q["bf16_keys"])]
    state = ref.fake_quantize_state(state, qkeys, q["precision"], q["block_size"])
    state = {k: v.to(device) for k, v in state.items()}
    vae_state = ref.load_safetensors_dir(os.path.join(snap, "vae"))
    return ref.run_reference(prompt, geo, seed, snap=snap, device=device, transformer_state=state,
                             vae_state=vae_state, prompt_embeds=prompt_embeds, keep_steps=keep_steps)


def config_key(geo: ref.Flux2Geometry) -> str:
    return f"{geo.height}x{geo.width}x{geo.max_sequence_length}x{geo.num_inference_steps}"


def program_paths(cfg: dict, geo: ref.Flux2Geometry, engines: int) -> tuple[str, str]:
    """Where one configuration's compiled programs live:
    flux2_klein_4b_bin/programs/<HxWxTOKENSxSTEPS>_e<engines>.{bin,json}."""
    d = os.path.join(HERE, cfg["paths"]["bin_dir"], "programs")
    os.makedirs(d, exist_ok=True)
    stem = os.path.join(d, f"{config_key(geo)}_e{engines}")
    return stem + ".bin", stem + ".json"


def ensure_programs(ue: "Flux2KleinEngine", cfg: dict, *, allow_compile: bool = True, force: bool = False) -> dict:
    """Return the program metadata for ``ue``'s configuration, compiling (offline-capable)
    when the files are missing or their signature is stale."""
    bin_p, meta_p = program_paths(cfg, ue.geo, ue.ne)
    if not force and os.path.exists(bin_p) and os.path.exists(meta_p):
        with open(meta_p) as f:
            meta = json.load(f)
        if meta.get("signature") == ue.program_signature():
            return meta
    if not allow_compile:
        raise RuntimeError(f"programs for {config_key(ue.geo)} / {ue.ne} engine(s) are missing or stale: "
                           f"compile them with flux2_klein_4b_test.py --compile-only")
    return ue.compile_programs(bin_p, meta_p)


def resolve_engines(spec: str, offline: bool) -> int:
    if spec == "max":
        cores = getattr(user_dma_core, "ANDROMEDA_CORE_COUNT", None)
        return int(cores) if cores else (8 if offline else 1)
    return max(1, int(spec))


def main(argv=None) -> int:
    cfg = load_config()
    m, dfl = cfg["model"], cfg["defaults"]
    ap = argparse.ArgumentParser(description="FLUX.2 klein 4B on the Unified Engine")
    ap.add_argument("--prompt", default=dfl["prompt"])
    ap.add_argument("--seed", type=int, default=dfl["seed"])
    ap.add_argument("--height", type=int, default=m["height"])
    ap.add_argument("--width", type=int, default=m["width"])
    ap.add_argument("--max-seq-len", default=str(m["max_sequence_length"]),
                    help="text tokens (padded like the pipeline), or 'auto' = the prompt's length rounded up to 64")
    ap.add_argument("--steps", type=int, default=m["num_inference_steps"])
    ap.add_argument("--dev", default=dfl["dev"])
    ap.add_argument("--engines", default="max", help="engine count or 'max' (HW_INFO core count)")
    ap.add_argument("--compile-only", action="store_true", help="build programs.bin without a board")
    ap.add_argument("--no-compile", action="store_true", help="run-from-bin: never compile")
    ap.add_argument("--force-compile", action="store_true")
    ap.add_argument("--vae", choices=["fpga", "host"], default="fpga")
    ap.add_argument("--phase-timeout", type=float, default=4 * 3600.0, help="wait timeout (s) for the TE / DiT programs")
    ap.add_argument("--text-encoder", choices=["fpga", "host"], default="fpga",
                    help="host: compute the prompt embedding with transformers and skip the TE phase")
    ap.add_argument("--no-reference", action="store_true", help="skip the host IF4 reference comparison")
    ap.add_argument("--ref-device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="output PNG (default flux2_klein_4b_bin/flux2_<seed>.png)")
    args = ap.parse_args(argv)

    if args.max_seq_len == "auto":
        snap0 = ref.snapshot_dir(cfg["paths"]["hf_model_repo"])
        n_real = int(tokenize(snap0, args.prompt, 4096)[1].sum())
        max_seq_len = max(64, -(-n_real // 64) * 64)
        _log(f"  prompt is {n_real} tokens -> max_sequence_length {max_seq_len}")
    else:
        max_seq_len = int(args.max_seq_len)
    geo = ref.Flux2Geometry(args.height, args.width, max_seq_len, args.steps)
    paths = _paths(cfg)
    os.makedirs(os.path.join(HERE, cfg["paths"]["bin_dir"]), exist_ok=True)

    if not args.compile_only:
        for k in ("params_te_bin", "params_dit_bin", "vae_weights", "embed_tokens_bin"):
            if not os.path.exists(paths[k]):
                if args.no_compile:
                    raise FileNotFoundError(paths[k])
                _log("  exporting weights (one-time)...")
                W.build_all(cfg)
                break
        user_dma_core.set_dma_device(args.dev)
        user_dma_core.configure_clock_from_hardware()
    ne = resolve_engines(args.engines, args.compile_only)
    _log(f"FLUX.2 klein 4B | {geo.height}x{geo.width} | text {geo.n_txt} + image {geo.n_img} tokens "
         f"(joint {geo.n_joint}) | {geo.num_inference_steps} steps | {ne} engine(s)")

    ue = Flux2KleinEngine(cfg, geo, num_engines=ne, offline=args.compile_only)
    ue.phase_timeout = args.phase_timeout
    _log(f"  tensor arena {ue.tensor_usage / 1e6:.0f} MB, params image {ue.params_image_size / 1e9:.2f} GB")

    meta = ensure_programs(ue, cfg, allow_compile=not args.no_compile, force=args.force_compile)
    if args.compile_only:
        _log(f"compile-only done: {program_paths(cfg, geo, ne)[0]}")
        return 0
    meta = ue.load_programs(*program_paths(cfg, geo, ne))
    ue.timings = {}

    snap = ref.snapshot_dir(cfg["paths"]["hf_model_repo"])
    vae_w = torch.load(paths["vae_weights"], map_location="cpu")
    ue.load_constants(vae_w)

    # ---------------- phase 1: text encoder ----------------
    input_ids, attention_mask = tokenize(snap, args.prompt, geo.n_txt)
    host_embeds = None
    if args.text_encoder == "host":
        te = ref.Qwen3TextEncoder(snap, device=args.ref_device, num_layers=geo.text_layers_used)
        host_embeds = te(args.prompt, geo.n_txt)
        del te
        fpga_embeds = host_embeds
    else:
        embed_tokens = load_embed_tokens(cfg, paths)
        fpga_embeds = ue.run_text_encoder(meta, input_ids, attention_mask, embed_tokens, paths["params_te_bin"])
        _log(f"  TE phase {ue.timings['te_s']:.1f}s; embeds finite={bool(torch.isfinite(fpga_embeds.float()).all())}")

    # ---------------- phase 2: DiT ----------------
    noise = ref.initial_noise(geo, args.seed)
    lat_denorm = ue.run_dit(meta, noise, paths["params_dit_bin"], prompt_embeds=host_embeds)
    _log(f"  DiT phase {ue.timings['dit_s']:.1f}s; latents finite={bool(torch.isfinite(lat_denorm.float()).all())}")

    # ---------------- phase 3: VAE ----------------
    vae_cfg = ref.load_json(os.path.join(snap, "vae", "config.json"))
    latent = ref.unpatchify_latents(ref.unpack_latents(lat_denorm, geo.token_h, geo.token_w))  # (1, 32, h, w)
    if args.vae == "fpga":
        img = ue.run_vae(meta, latent[0].to(torch.bfloat16), vae_w)[None]
    else:
        t0 = time.time()
        vae = ref.Flux2VaeDecoder(ref.load_safetensors_dir(os.path.join(snap, "vae")), vae_cfg, device=args.ref_device)
        img = vae.decode(latent)
        ue.timings["vae_s"] = time.time() - t0
    image = ref.to_uint8_image(img)
    out = args.out or os.path.join(HERE, cfg["paths"]["bin_dir"], f"flux2_{args.seed}.png")
    ref.save_image(image, out)
    _log(f"  VAE phase ({args.vae}) {ue.timings['vae_s']:.1f}s -> {out}")
    vae_psnr = None
    if args.vae == "fpga" and not args.no_reference:
        # the decoder alone: same latent, same IF8 weights, host arithmetic (CPU)
        t0 = time.time()
        oracle = flux2_klein_4b_vae.reference_decode(vae_w, cfg, latent[0])[None]
        vae_psnr = psnr_u8(ref.to_uint8_image(oracle), image)
        _log(f"  FPGA decoder vs IF8 host oracle: image PSNR {vae_psnr:.1f} dB, "
             f"latent SNR {snr_db(oracle, img):.1f} dB (oracle {time.time() - t0:.1f}s on the host)")

    # ---------------- scoring ----------------
    result = {"height": geo.height, "width": geo.width, "max_sequence_length": geo.n_txt, "steps": geo.num_inference_steps,
              "seed": args.seed, "engines": ne, "image": out, "vae": args.vae, "text_encoder": args.text_encoder,
              "finite": bool(torch.isfinite(lat_denorm.float()).all()) and bool(torch.isfinite(img.float()).all()),
              "te_s": ue.timings.get("te_s"), "dit_s": ue.timings.get("dit_s"), "vae_s": ue.timings.get("vae_s"),
              "vae_psnr_db": vae_psnr}
    if not args.no_reference:
        _log("  host IF4 reference...")
        rr = quantized_reference(cfg, geo, snap, args.prompt, args.seed, args.ref_device,
                                 prompt_embeds=host_embeds)
        ref.save_image(rr.image, out.replace(".png", "_ref.png"))
        result["embeds_snr_db"] = snr_db(rr.prompt_embeds, fpga_embeds)
        result["latents_snr_db"] = snr_db(rr.latents, ue.latents_raw)
        # Per-step agreement; step 0 (x1 = x0 + dt0*v0) is one full DiT forward on identical
        # inputs -- the metric that isolates engine arithmetic from the sampler's chaotic
        # amplification of per-step noise.
        result["step_latents_snr_db"] = [snr_db(x, ue.step_latents[s + 1]) for s, (i, sg, v, x) in enumerate(rr.steps)]
        dt0 = float(rr.sigmas[1] - rr.sigmas[0])
        v0_fpga = (ue.step_latents[1].float() - ue.step_latents[0].float()) / dt0
        result["velocity0_snr_db"] = snr_db(rr.steps[0][2], v0_fpga)
        result["image_psnr_db"] = psnr_u8(rr.image, image)
        _log(f"  vs IF4 host reference: embeds {result['embeds_snr_db']:.1f} dB, step-0 velocity "
             f"{result['velocity0_snr_db']:.1f} dB, per-step latents {[round(x, 1) for x in result['step_latents_snr_db']]} dB, "
             f"image PSNR {result['image_psnr_db']:.1f} dB")
    print("TEST_RESULT: " + json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
