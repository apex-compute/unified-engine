#!/usr/bin/env python3
"""
ACT (Action Chunking Transformer, LeRobot) inference on the Unified Engine.

Inference graph (VAE encoder is train-only; latent z = 0):
  per camera: ResNet-18 (BN folded) -> 15x20x512 feature map -> 1x1 proj -> 300 tokens
  tokens = [latent_proj(0), state_proj(state), cam0 (300), cam1 (300)]      S = 602 -> 640
  4 x post-norm encoder layers (8 heads x 64, FFN 3200 ReLU)
  1 x post-norm decoder layer over 100 (-> 128) zero queries + learned query pos
  final LayerNorm -> action head 512 -> ACTION_DIM                           -> (100, ACTION_DIM)

Inputs come from act_bin/ (produced by act_export.py in the lerobot env):
  act_weights.pt  fp32 state_dict + host-precomputed positional tables
  reference.npz   the exact model inputs used for the golden run + CPU intermediates

Usage:
  python act_test.py --engines 8        # first run: compile + cache; later runs: load bins + run
  python act_test.py --engines 8 --profile   # per-section ms / GFLOP / utilization
  python act_test.py --cleanup          # drop cached params/programs
  python act_test.py --dev xdma0 --device kintex7

Bins (act_bin/): params.bin is the weight blob (engine-count independent); programs_e<N>/ holds
the primary + worker instruction streams and the compile-time constants for N engines. The cache
key is the weights file + a hash of this source, so stale bins are never loaded.

Execution model: one instruction stream per engine, one host launch per inference. Every stage
is a sharded region replayed on all engines (MultiEngineScheduler, master/worker four-phase
barrier). The two camera backbones run concurrently on NE/2 engines each.

Layout / kernel conventions:
  * conv activations are HWC "padded grids" [(H+2)*(W+2), C] with a zero border that is never
    written (only real pixels are computed).
  * every 3x3 conv is the on-chip window kernel (conv3x3_rows): input rows + halo resident in
    URAM A, the 3x3 window assembled per pixel by SRAM copies, one matvec per (N-strip, K-group)
    against resident weights. Bias rides in a ones-column block of the weights; residuals and
    K-group partials go through the bias BRAM. Big-weight convs shard by output channel.
  * stem 7x7/s2: host space-to-depth(4) of the image (48 -> 64 ch) turns it into a 3x3 window
    conv over the 120x160 super-grid, one per sub-pixel plane; maxpool folds the 9 plane taps
    on-chip with max(a, b) = b + relu(a - b) (ReLU = identity-matvec clamp).
  * transformer: token rows sharded across engines, attention sharded by head; fused Q/K/V and
    o_proj with head scatter/gather via strided DMA.
"""
import builtins
import json
import math
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

import numpy as np
import torch

_original_print = builtins.print
_SILENT_MODE = False


def quiet_print(*args, **kwargs):
    if _SILENT_MODE:
        return
    _original_print(*args, **kwargs)


builtins.print = quiet_print

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, DMA_DEVICE_C2H, DRAM_INSTRUCTION_ADDR,
    UE_VECTOR_SIZE, URAM_NEAR_FULL_ELEMENTS,
    set_dma_device, UnifiedEngine, UE_MODE,
)

_BPE = 2
NEG_INF = float("-inf")

ACT_PARAMS_BASE = 0x00000000
ACT_TENSOR_BASE = 0x40000000
ACT_PROGRAM_BASE = 0x80000000


def pad64(x: int) -> int:
    return ((x + 63) // 64) * 64


def snr_db(ref: np.ndarray, hw: np.ndarray) -> float:
    ref = ref.astype(np.float64).ravel()
    err = ref - hw.astype(np.float64).ravel()
    den = np.linalg.norm(err)
    if den == 0:
        return float("inf")
    return 20.0 * math.log10(np.linalg.norm(ref) / den)


# ---------------------------------------------------------------------------
# Geometry of a padded HWC grid
# ---------------------------------------------------------------------------

class Grid:
    """Zero-bordered HWC activation: rows = (H+2)*(W+2), each C wide."""

    def __init__(self, H: int, W: int, C: int):
        self.H, self.W, self.C = H, W, C
        self.Hp, self.Wp = H + 2, W + 2

    @property
    def rows(self) -> int:
        # + one slack row: the conv window of the last output row reads one grid row past H+1
        return self.Hp * self.Wp + self.Wp + 2

    @property
    def numel(self) -> int:
        return self.rows * self.C

    def idx(self, h: int, w: int) -> int:
        """Grid row index of interior pixel (h, w), 0-based interior coords."""
        return (h + 1) * self.Wp + (w + 1)

    @property
    def conv_M(self) -> int:
        """Grid rows spanned by the interior (used by the dense 1x1 downsample matmul)."""
        return self.H * self.Wp


# ---------------------------------------------------------------------------
# Engine subclass
# ---------------------------------------------------------------------------

class ACT_UnifiedEngine(UnifiedEngine):
    D, NH, HD, FF = 512, 8, 64, 3200
    N_ENC, N_DEC = 4, 1
    CHUNK, CHP = 100, 128           # decoder queries, padded
    N_CAMS = 2
    IMG_H, IMG_W = 480, 640
    STATE_DIM = ACTION_DIM = 6   # 5 arm joints + gripper
    LATENT = 32
    FEAT_H, FEAT_W = 15, 20
    S = 2 + N_CAMS * FEAT_H * FEAT_W  # 602
    SP = pad64(S)                    # 640
    BN_EPS = 1e-5

    # ResNet-18 stage table (after the stem+pool at 120x160x64)
    STAGES = [  # (name, C_out, stride, H_out, W_out)
        ("layer1", 64, 1, 120, 160),
        ("layer2", 128, 2, 60, 80),
        ("layer3", 256, 2, 30, 40),
        ("layer4", 512, 2, 15, 20),
    ]

    BIN_SUBDIR = "act_bin"
    USE_BIN_CACHE = True    # params.bin once + programs per engine count (act_bin/programs_e<N>/)
    DMA_CHUNK_BYTES = 1 * 1024 * 1024

    # Worker engine arenas (absolute DRAM): above the primary's program region, 128 MB each.
    WORKER_ARENA_BASE = 0x90000000
    WORKER_ARENA_STRIDE = 0x08000000
    WORKER_TENSOR_OFF = 0x01000000
    WORKER_PROGRAM_OFF = 0x04000000

    def __init__(self, script_dir: str = SCRIPT_DIR, num_engines: int = 1):
        super().__init__(params_dram_base=ACT_PARAMS_BASE,
                         tensor_dram_base=ACT_TENSOR_BASE,
                         program_dram_base=ACT_PROGRAM_BASE)
        self.NE = num_engines
        self.SNE = num_engines          # engines in the current shard group (backbone: NE/2 per camera)
        self.EOFF = 0                   # global index of the group's first engine (per-engine scratch)
        self.sched = None
        self.worker_progs = []
        if num_engines > 1:
            from multi_engine_shard import MultiEngineScheduler
            self.sched = MultiEngineScheduler(
                self, num_engines=num_engines,
                worker_dram_base=self.WORKER_ARENA_BASE, worker_dram_stride=self.WORKER_ARENA_STRIDE,
                worker_tensor_offset=self.WORKER_TENSOR_OFF, worker_program_offset=self.WORKER_PROGRAM_OFF,
                allow_unaligned_rows=True, handshake="four_phase", region_rendezvous="master_worker")
        self.script_dir = script_dir
        with open(os.path.join(script_dir, "act_config.json")) as f:
            self.cfg = json.load(f)
        self.bin_dir = os.path.join(script_dir, self.BIN_SUBDIR)

        self.cache_meta = self.load_params() if self.USE_BIN_CACHE else None
        self.params_from_bin = self.cache_meta is not None
        if not self.params_from_bin:
            self.weight_init()
        self.tensor_init()

    # ------------------------------------------------------------------
    # DRAM helpers
    # ------------------------------------------------------------------

    def _alloc_param(self, t: torch.Tensor) -> int:
        t = t.to(torch.bfloat16).contiguous()
        addr = self.get_params_dram_addr()
        self.allocate_params_dram(t.numel() * _BPE)
        self.dma_to_accelerator_memory(addr, t)
        return addr

    def _alloc_tensor(self, numel: int, zero: bool = False) -> int:
        addr = self.allocate_tensor_dram(numel * _BPE)
        if zero:
            self._zero_fill(addr, numel)
        return addr

    def _zero_fill(self, addr: int, numel: int) -> None:
        chunk = self.DMA_CHUNK_BYTES // _BPE
        z = torch.zeros(min(chunk, numel), dtype=torch.bfloat16)
        off = 0
        while off < numel:
            n = min(chunk, numel - off)
            self.dma_to_accelerator_memory(addr + off * _BPE, z[:n].contiguous())
            off += n

    def _write_bf16(self, addr: int, t: torch.Tensor) -> None:
        self.dma_to_accelerator_memory(addr, t.to(torch.bfloat16).contiguous())

    def _read_bf16(self, addr: int, numel: int) -> torch.Tensor:
        buf = bytearray(numel * _BPE)
        self.dma_read(DMA_DEVICE_C2H, addr, buf, numel * _BPE)
        return torch.frombuffer(bytes(buf), dtype=torch.bfloat16).float()

    def write_captured_instructions_to_dram(self, start_addr: int = DRAM_INSTRUCTION_ADDR) -> int:
        """Chunked override (large instruction streams segfault the single-shot DMA)."""
        if not self.capture_buffer or self.capture_count == 0:
            return 0
        total = self.capture_count * 32
        data = bytes(b"".join(inst.get_bytes() for inst in self.capture_buffer))
        off = 0
        while off < total:
            n = min(self.DMA_CHUNK_BYTES, total - off)
            self.dma_write(DMA_DEVICE_H2C, start_addr + off, data[off:off + n], n)
            off += n
        return total

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    def _fold_bn(self, sd: dict, conv: str, bn: str):
        w = sd[conv + ".weight"]
        g, b = sd[bn + ".weight"], sd[bn + ".bias"]
        m, v = sd[bn + ".running_mean"], sd[bn + ".running_var"]
        s = g / torch.sqrt(v + self.BN_EPS)
        return w * s[:, None, None, None], b - m * s

    def _alloc_stem(self, w7: torch.Tensor, b7: torch.Tensor) -> list[dict]:
        """7x7/s2 stem on a space-to-depth(4) image, one 3x3 window conv per sub-pixel plane.

        Output pixel (2H+a', 2W+b') reads image rows 4(H+dH)+a with ky = 4*dH + a - 2*a' + 3
        (0..6 valid), same for columns -> a 3x3 tap-conv over the 120x160 super-grid whose tap
        (dH,dW) weight is (64 out, 64 s2d-channels ((a*4+b)*3 + c)), zero outside the kernel.
        """
        Cout = w7.shape[0]
        planes = []
        for ap in range(2):
            for bp in range(2):
                w9 = torch.zeros(Cout, 64, 3, 3)
                for dH in (-1, 0, 1):
                    for dW in (-1, 0, 1):
                        for a in range(4):
                            ky = 4 * dH + a - 2 * ap + 3
                            if not 0 <= ky <= 6:
                                continue
                            for b in range(4):
                                kx = 4 * dW + b - 2 * bp + 3
                                if not 0 <= kx <= 6:
                                    continue
                                for c in range(3):
                                    w9[:, (a * 4 + b) * 3 + c, dH + 1, dW + 1] = w7[:, c, ky, kx]
                planes.append(self._alloc_conv3x3(w9, b7))
        return planes

    def _alloc_linear(self, sd: dict, prefix: str, K_pad: int | None = None, N_pad: int | None = None):
        w, b = sd[prefix + ".weight"], sd[prefix + ".bias"]
        if w.dim() == 4:  # 1x1 conv
            w = w[:, :, 0, 0]
        N, K = w.shape
        Kp, Np = K_pad or K, N_pad or N
        wp = torch.zeros(Np, Kp)
        wp[:N, :K] = w
        bp = torch.zeros(Np)
        bp[:N] = b
        return self._alloc_param(wp), self._alloc_param(bp)

    def _alloc_mha(self, sd: dict, prefix: str) -> dict:
        """nn.MultiheadAttention -> full-width Q/K/V weights (N=512 each, split by head after
        the matmul) and the o_proj weight (K=512, heads gathered before the matmul)."""
        D = self.D
        w_in, b_in = sd[prefix + ".in_proj_weight"], sd[prefix + ".in_proj_bias"]
        wo, bo = sd[prefix + ".out_proj.weight"], sd[prefix + ".out_proj.bias"]
        out = {}
        for name, base in (("q", 0), ("k", D), ("v", 2 * D)):
            out[name] = (self._alloc_param(w_in[base:base + D]), self._alloc_param(b_in[base:base + D]))
        out["o"] = (self._alloc_param(wo.contiguous()), self._alloc_param(bo))
        return out

    def _alloc_ln(self, sd: dict, prefix: str):
        return self._alloc_param(sd[prefix + ".weight"]), self._alloc_param(sd[prefix + ".bias"])

    def weight_init(self) -> None:
        sd = torch.load(os.path.join(self.script_dir, self.cfg["paths"]["weights_pt"]), weights_only=False)
        dims = sd.pop("_dims")
        assert tuple(dims["feat_hw"]) == (self.FEAT_H, self.FEAT_W), dims["feat_hw"]
        self.weights_source = dims["checkpoint"]

        # --- constants ---
        self.ZERO_ROW = self._alloc_param(torch.zeros(3 * 512))         # >= 3*Cin (SRAM conv copies), >= D
        ones_blk = torch.zeros(64); ones_blk[0] = 1.0
        self.ONES_BLOCK = self._alloc_param(ones_blk)                    # staging-row bias block (1.0 in column 0)
        self.IDENTITY_64 = self._alloc_param(torch.eye(64))
        self.IDENTITY_128 = self._alloc_param(torch.eye(128))
        self.LATENT_IN = self._alloc_param(torch.zeros(64))              # z = 0, K padded 32 -> 64
        enc_pos = torch.zeros(self.SP, self.D)
        enc_pos[:self.S] = sd["_enc_pos"]
        self.ENC_POS = self._alloc_param(enc_pos)
        dec_pos = torch.zeros(self.CHP, self.D)
        dec_pos[:self.CHUNK] = sd["_dec_pos"]
        self.DEC_POS = self._alloc_param(dec_pos)
        enc_mask = torch.zeros(self.SP, self.SP)
        enc_mask[:, self.S:] = NEG_INF
        self.ENC_MASK = self._alloc_param(enc_mask)
        dec_mask = torch.zeros(self.CHP, self.CHP)
        dec_mask[:, self.CHUNK:] = NEG_INF
        self.DEC_MASK = self._alloc_param(dec_mask)

        # --- backbone (BN folded) ---
        w7, b7 = self._fold_bn(sd, "backbone.conv1", "backbone.bn1")
        self.STEM = self._alloc_stem(w7, b7)
        self.BB = []
        for name, cout, stride, _, _ in self.STAGES:
            blocks = []
            for i in range(2):
                p = f"backbone.{name}.{i}"
                w1, b1 = self._fold_bn(sd, p + ".conv1", p + ".bn1")
                w2, b2 = self._fold_bn(sd, p + ".conv2", p + ".bn2")
                blk = {"conv1": self._alloc_conv3x3(w1, b1), "conv2": self._alloc_conv3x3(w2, b2)}
                if i == 0 and stride == 2:
                    wd, bd = self._fold_bn(sd, p + ".downsample.0", p + ".downsample.1")
                    blk["down"] = self._alloc_param(wd[:, :, 0, 0])
                    blk["bd"] = self._alloc_param(bd)
                blocks.append(blk)
            self.BB.append(blocks)

        # --- token projections ---
        self.LATENT_PROJ = self._alloc_linear(sd, "encoder_latent_input_proj", K_pad=64)
        self.STATE_PROJ = self._alloc_linear(sd, "encoder_robot_state_input_proj", K_pad=64)
        self.IMG_PROJ = self._alloc_linear(sd, "encoder_img_feat_input_proj")

        # --- transformer ---
        self.ENC = []
        for i in range(self.N_ENC):
            p = f"encoder.layers.{i}"
            self.ENC.append({
                "attn": self._alloc_mha(sd, p + ".self_attn"),
                "ffn1": self._alloc_linear(sd, p + ".linear1"),
                "ffn2": self._alloc_linear(sd, p + ".linear2"),
                "ln1": self._alloc_ln(sd, p + ".norm1"),
                "ln2": self._alloc_ln(sd, p + ".norm2"),
            })
        self.DEC = []
        for i in range(self.N_DEC):
            p = f"decoder.layers.{i}"
            self.DEC.append({
                "self": self._alloc_mha(sd, p + ".self_attn"),
                "cross": self._alloc_mha(sd, p + ".multihead_attn"),
                "ffn1": self._alloc_linear(sd, p + ".linear1"),
                "ffn2": self._alloc_linear(sd, p + ".linear2"),
                "ln1": self._alloc_ln(sd, p + ".norm1"),
                "ln2": self._alloc_ln(sd, p + ".norm2"),
                "ln3": self._alloc_ln(sd, p + ".norm3"),
            })
        self.DEC_NORM = self._alloc_ln(sd, "decoder.norm")
        self.ACTION_HEAD = self._alloc_linear(sd, "action_head", N_pad=64)

    # ------------------------------------------------------------------
    # Activations
    # ------------------------------------------------------------------

    def tensor_init(self) -> None:
        SP, CHP, D, FF, HD = self.SP, self.CHP, self.D, self.FF, self.HD
        # Backbone grids (zeroed: the program only ever re-zeroes borders it dirties)
        self.IMG = Grid(120, 160, 64)
        self.IMG_DRAM = [self._alloc_tensor(self.IMG.numel, zero=True) for _ in range(self.N_CAMS)]
        # per camera so both backbones can run concurrently
        self.STEM_PLANES = [[self._alloc_tensor(self.IMG.numel, zero=True) for _ in range(4)] for _ in range(self.N_CAMS)]
        self.POOL_G = Grid(120, 160, 64)
        self.POOL_DRAM = [self._alloc_tensor(self.POOL_G.numel, zero=True) for _ in range(self.N_CAMS)]
        self.POOL_TMP = [self._alloc_tensor(self.POOL_G.numel, zero=True) for _ in range(self.N_CAMS)]
        self.STAGE_G = [Grid(H, W, cout) for name, cout, stride, H, W in self.STAGES]
        self.STAGE_BUF = [[{k: self._alloc_tensor(g.numel, zero=True) for k in ("A", "B", "T", "R")}
                           for g in self.STAGE_G] for _ in range(self.N_CAMS)]
        # stride-2 tap gather scratch: padded output grid x C_in, worst case over stages
        gather_elems = max(Grid(H, W, 1).rows * self.STAGE_G[i - 1].C
                           for i, (_, _, st, H, W) in enumerate(self.STAGES) if st == 2)
        self.GATHER_DRAM_ALL = [self._alloc_tensor(gather_elems, zero=True) for _ in range(self.N_CAMS)]
        self.GATHER_DRAM = self.GATHER_DRAM_ALL[0]
        self.FEAT_DENSE = [self._alloc_tensor(self.FEAT_H * self.FEAT_W * D) for _ in range(self.N_CAMS)]

        # Tokens / encoder
        self.X = self._alloc_tensor(SP * D, zero=True)       # residual stream (pad rows stay 0)
        self.QK = self._alloc_tensor(SP * D)                  # x + pos
        self.X1 = self._alloc_tensor(SP * D)
        self.O = self._alloc_tensor(SP * D)
        self.H1 = self._alloc_tensor(SP * FF)
        self.QH = [self._alloc_tensor(SP * HD) for _ in range(self.NH)]
        self.KH = [self._alloc_tensor(SP * HD) for _ in range(self.NH)]
        self.VH = [self._alloc_tensor(SP * HD) for _ in range(self.NH)]
        self.AH = [self._alloc_tensor(SP * HD) for _ in range(self.NH)]
        self.ATT_SCRATCH = [self._alloc_tensor(HD * SP + SP * SP + SP * HD) for _ in range(self.NE)]  # per engine
        self.PF = self._alloc_tensor(SP * D)     # fused projection staging [rows, 512] (row-disjoint per engine)
        # K-split partial spill: one row per pixel of the largest K-split chunk (layer4: <=15x20 px x 128)
        self.KSPLIT_SPILL = [self._alloc_tensor(self.FEAT_H * self.FEAT_W * 128) for _ in range(self.NE)]   # per engine
        self.STATE_IN = self._alloc_tensor(64, zero=True)

        # Decoder
        self.DIN = self._alloc_tensor(CHP * D, zero=True)     # zero queries (constant)
        self.DQK = self._alloc_tensor(CHP * D)
        self.D1 = self._alloc_tensor(CHP * D)
        self.D2 = self._alloc_tensor(CHP * D)
        self.DO = self._alloc_tensor(CHP * D)
        self.DH1 = self._alloc_tensor(CHP * FF)
        self.DQH = [self._alloc_tensor(CHP * HD) for _ in range(self.NH)]
        self.DKH = [self._alloc_tensor(CHP * HD) for _ in range(self.NH)]
        self.DVH = [self._alloc_tensor(CHP * HD) for _ in range(self.NH)]
        self.DAH = [self._alloc_tensor(CHP * HD) for _ in range(self.NH)]
        self.D3 = self._alloc_tensor(CHP * D)                 # last decoder layer out (pre final norm)
        self.DEC_OUT = self._alloc_tensor(CHP * D)
        self.ACT_OUT = self._alloc_tensor(CHP * 64)

    # ------------------------------------------------------------------
    # Bin cache
    # ------------------------------------------------------------------

    # Layout: act_bin/params.bin + params.json (weights, engine-count independent) and
    # act_bin/programs_e<NE>/{primary.bin, worker<i>.bin, meta.json} per engine count. Tensor
    # addresses are a deterministic function of NE (same allocation order), so a cached program
    # is valid whenever weights (mtime) and NE match.

    def _weights_stamp(self) -> str:
        """Cache key: weights file (size+mtime) and a hash of this source file, so a program
        compiled by older emitter code is never loaded against new buffer layouts."""
        import hashlib
        wp = os.path.join(self.script_dir, self.cfg["paths"]["weights_pt"])
        st = os.stat(wp)
        with open(os.path.abspath(__file__), "rb") as f:
            code = hashlib.sha1(f.read()).hexdigest()[:10]
        return f"{st.st_size}:{int(st.st_mtime)}:{code}"

    def _prog_dir(self) -> str:
        return os.path.join(self.bin_dir, f"programs_e{self.NE}")

    def _dma_dump(self, addr: int, size: int, path: str) -> None:
        with open(path, "wb") as f:
            off = 0
            while off < size:
                n = min(self.DMA_CHUNK_BYTES, size - off)
                buf = bytearray(n)
                self.dma_read(DMA_DEVICE_C2H, addr + off, buf, n)
                f.write(buf)
                off += n

    def _dma_load(self, addr: int, path: str) -> int:
        with open(path, "rb") as f:
            off = 0
            while True:
                data = f.read(self.DMA_CHUNK_BYTES)
                if not data:
                    break
                self.dma_write(DMA_DEVICE_H2C, addr + off, data, len(data))
                off += len(data)
        return off

    def dump_params(self):
        os.makedirs(self.bin_dir, exist_ok=True)
        total = getattr(self, "_params_weights_end", None) or self.get_params_dram_usage()
        self._dma_dump(self._params_dram_base, total, os.path.join(self.bin_dir, "params.bin"))
        with open(os.path.join(self.bin_dir, "params.json"), "w") as f:
            json.dump({"size": total, "source": self.weights_source, "stamp": self._weights_stamp(),
                       }, f)
        _original_print(f"  Params dumped: {total / 1024**2:.1f} MB -> act_bin/params.bin")

    def load_params(self):
        """Load params.bin if present and matching the weights file; returns its meta or None."""
        bp, mp = (os.path.join(self.bin_dir, n) for n in ("params.bin", "params.json"))
        if not (os.path.exists(bp) and os.path.exists(mp)):
            return None
        with open(mp) as f:
            meta = json.load(f)
        if meta.get("stamp") != self._weights_stamp():
            _original_print("  cached bins are stale (weights or emitter source changed) -> recompiling")
            return None
        n = self._dma_load(self._params_dram_base, bp)
        self.allocate_params_dram(n)
        self.weights_source = meta.get("source", "?")
        _original_print(f"  Params loaded: {n / 1024**2:.1f} MB from act_bin/params.bin ({self.weights_source})")
        return meta

    def dump_programs(self, program_addr: int):
        d = self._prog_dir()
        os.makedirs(d, exist_ok=True)
        size = self.get_program_dram_usage()
        self._dma_dump(program_addr, size, os.path.join(d, "primary.bin"))
        # compile-time constants the kernels allocated: primary tail past the weights, worker arenas
        tail0, tail1 = self._params_weights_end, self.get_params_dram_usage()
        if tail1 > tail0:
            self._dma_dump(self._params_dram_base + tail0, tail1 - tail0, os.path.join(d, "primary_consts.bin"))
        workers = []
        if self.sched is not None:
            for i, w in enumerate(self.sched.workers):
                wsize = w.get_program_dram_usage()
                self._dma_dump(self.worker_progs[i], wsize, os.path.join(d, f"worker{i}.bin"))
                pused = w.get_params_dram_usage()
                if pused:
                    self._dma_dump(w._params_dram_base, pused, os.path.join(d, f"worker{i}_consts.bin"))
                workers.append({"addr": self.worker_progs[i], "size": wsize, "params_used": pused})
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({"engines": self.NE, "primary_addr": program_addr, "primary_size": size,
                       "num_instructions": self.last_num_instructions, "workers": workers,
                       "consts_offset": tail0, "consts_size": tail1 - tail0,
                       "stamp": self._weights_stamp()}, f)
        tot = size + sum(w["size"] for w in workers)
        _original_print(f"  Programs dumped: {tot / 1024**2:.1f} MB -> act_bin/programs_e{self.NE}/")

    def load_programs(self) -> int | None:
        d = self._prog_dir()
        mp = os.path.join(d, "meta.json")
        if not os.path.exists(mp):
            return None
        with open(mp) as f:
            meta = json.load(f)
        if meta.get("stamp") != self._weights_stamp() or meta.get("engines") != self.NE:
            return None
        if meta.get("consts_size", 0):
            n = self._dma_load(self._params_dram_base + meta["consts_offset"], os.path.join(d, "primary_consts.bin"))
            self.allocate_params_dram(n)
        addr = self.get_program_dram_addr()
        assert addr == meta["primary_addr"], (addr, meta["primary_addr"])
        n = self._dma_load(addr, os.path.join(d, "primary.bin"))
        self.allocate_program_dram(n)
        self.last_num_instructions = meta.get("num_instructions", 0)
        self.worker_progs = []
        if self.sched is not None:
            for i, w in enumerate(self.sched.workers):
                wa = w.get_program_dram_addr()
                assert wa == meta["workers"][i]["addr"]
                wn = self._dma_load(wa, os.path.join(d, f"worker{i}.bin"))
                w.allocate_program_dram(wn)
                if meta["workers"][i].get("params_used", 0):
                    pn = self._dma_load(w._params_dram_base, os.path.join(d, f"worker{i}_consts.bin"))
                    w.allocate_params_dram(pn)
                self.worker_progs.append(wa)
        tot = n + sum(w["size"] for w in meta["workers"])
        _original_print(f"  Programs loaded: {tot / 1024**2:.1f} MB from act_bin/programs_e{self.NE}/")
        return addr

    # ------------------------------------------------------------------
    # Multi-engine plumbing
    #
    # Every emit helper takes the emitting engine `ue` explicitly (the primary is `self`,
    # workers are plain UnifiedEngines sharing the same DRAM). Work is split by GRID ROWS
    # (backbone) or TOKEN ROWS (transformer) across NE engines; attention is split by head.
    # `self.region(body)` replays body(ue, engine_idx) once per engine inside one barrier
    # -free region; `join=False` chains regions in the same lane without a rendezvous.
    # ------------------------------------------------------------------

    def _split(self, n: int, i: int) -> tuple[int, int]:
        """Contiguous slice i of SNE (the current shard group's engine count) over n items."""
        base, rem = divmod(n, self.SNE)
        start = i * base + min(i, rem)
        return start, start + base + (1 if i < rem else 0)

    def region(self, body, join: bool = True) -> None:
        if self.sched is None:
            body(self, 0)
            return
        self.sched.sharded_region(64 * self.NE, lambda ctx: body(ctx.unsafe_ue, ctx.engine_idx), join=join)

    # ------------------------------------------------------------------
    # Kernel wrappers
    # ------------------------------------------------------------------

    def _prime_M(self, ue, M: int) -> int:
        """Runtime-M PBI: re-arm the scratch allocators above the held M-reg and set M."""
        r = getattr(ue, "_pbi_M_reg", None)
        if r is None:
            r = ue._pbi_M_reg = ue.alloc_isa_reg()
        ue._isa_reg_counter = r + 1
        ue.reset_inst_ptr_counter()
        ue.generate_instruction_add_set(dst_reg_idx=r, immediate_value=M)
        return r

    def mm(self, ue, M, K, N, A, B, OUT, C=None, mode="broadcast_N", relu=False):
        """OUT = A[M,K] @ B[N,K]^T (+ C) (ReLU). C: bias[N] or full [M,N]."""
        if M <= 0:
            return
        self._flops += 2 * M * K * N
        r = self._prime_M(ue, M) if M > 1 else None   # M=1 token projections stay on the legacy path
        ue.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=B, OUTPUT_DRAM_ADDR=OUT,
                           C_DRAM_ADDR=C, bias_mode=mode, clamp_enable=relu, gpr_M_reg=r)

    def add(self, ue, M, N, A, B, OUT):
        if M > 0:
            ue.eltwise_core_dram(M, N, A, B, OUT, UE_MODE.ELTWISE_ADD)

    def sub(self, ue, M, N, A, B, OUT):
        if M > 0:
            ue.eltwise_core_dram(M, N, A, B, OUT, UE_MODE.ELTWISE_SUB)

    def relu(self, ue, numel, A, OUT):
        """Standalone ReLU: identity-matmul clamp on a flat 64-wide view (activation_core gotcha)."""
        assert numel % 64 == 0
        if numel == 0:
            return
        r = self._prime_M(ue, numel // 64)
        ue.activation_core(M=numel // 64, N=64, A_DRAM_ADDR=A, OUTPUT_DRAM_ADDR=OUT,
                           IDENTITY_DRAM_ADDR=self.IDENTITY_64, activation="clamp", gpr_M_reg=r)

    def ln(self, ue, M, A, OUT, gb):
        if M > 0:
            r = self._prime_M(ue, M)
            ue.layer_norm_core_dram(M=M, N=self.D, A_DRAM_ADDR=A, OUTPUT_DRAM_ADDR=OUT,
                                    GAMMA_DRAM_ADDR=gb[0], BETA_DRAM_ADDR=gb[1], ZEROS_DRAM_ADDR=self.ZERO_ROW,
                                    gpr_M_reg=r)

    def memcpy(self, ue, SRC, DST, numel):
        off = 0
        while off < numel:
            n = min(URAM_NEAR_FULL_ELEMENTS, numel - off)
            ue.accelerator_memory_to_sram(SRC + off * _BPE, 0x00000, n)
            ue.sram_to_accelerator_memory(0x00000, DST + off * _BPE, n)
            off += n

    def attention(self, ue, ei, q_rows, kv_rows, Q, K, V, MASK, OUT):
        self._flops += 2 * 2 * q_rows * kv_rows * self.HD          # QK^T and PV
        ue.unified_attention_core(batch=q_rows, aligned_seq_len=kv_rows, head_dim=self.HD,
                                  Q_DRAM_ADDR=Q, K_DRAM_ADDR=K, V_DRAM_ADDR=V,
                                  BIAS_DRAM_ADDR=MASK, OUTPUT_DRAM_ADDR=OUT,
                                  SCRATCH_DRAM_ADDR=self.ATT_SCRATCH[ei],
                                  IDENTITY_DRAM_ADDR=self.IDENTITY_64)

    # ------------------------------------------------------------------
    # Conv building blocks (padded-grid layout, sharded by grid rows)
    # ------------------------------------------------------------------

    def zero_borders(self, ue, buf: int, g: Grid, h0: int, h1: int) -> None:
        """Re-zero grid columns 0 and W+1 for output rows h0..h1-1 plus grid row h1+1: the dense
        downsample matmul writes the border cells of its row span (and the (h1+1, 0) cell), which
        must read back as zero padding. Shards may overlap on row h1+1; both write zeros."""
        ue.accelerator_memory_to_sram(self.ZERO_ROW, 0x00000, g.C)
        for r in range(h0 + 1, min(h1 + 2, g.Hp)):
            for c in (0, g.Wp - 1):
                ue.sram_to_accelerator_memory(0x00000, buf + (r * g.Wp + c) * g.C * _BPE, g.C)

    # ------------------------------------------------------------------
    # SRAM-window conv (all 3x3 convs of the net)
    # ------------------------------------------------------------------
    SRAM_STRIP_MAX = 200 * 1024      # weight-strip elements (N_s x K_g) allowed in URAM B
    COL_SHARD_MIN_WEIGHT_BYTES = 1 << 20   # convs with more weight than this shard by output channel

    def _conv_plan(self, conv: dict, Cin: int, Cout: int):
        """Choose K-groups (tap subsets) and the N-strip width so one weight strip fits URAM B.
        Returns (groups, N_s) with groups = [(taps, k_cols_in_group)]; the bias ones-column rides
        in the LAST group (its K is +64)."""
        taps_all = [t for g in conv["groups"] for t in g[0]]
        K1 = len(taps_all) * Cin + 64
        for N_s in (Cout, 512, 256, 128, 64):
            if N_s <= Cout and Cout % N_s == 0 and N_s * K1 <= self.SRAM_STRIP_MAX:
                return [(taps_all, K1)], N_s
        # K-split by dy row (3 taps each); pick the widest strip that fits the largest group
        rows = [[t for t in taps_all if t // 3 == dy] for dy in range(3)]
        rows = [r for r in rows if r]
        groups = [(r, len(r) * Cin + (64 if i == len(rows) - 1 else 0)) for i, r in enumerate(rows)]
        Kmax = max(k for _, k in groups)
        for N_s in (256, 128, 64):
            if N_s <= Cout and Cout % N_s == 0 and N_s * Kmax <= self.SRAM_STRIP_MAX:
                return groups, N_s
        raise AssertionError(f"no SRAM conv plan for Cin={Cin} Cout={Cout}")

    def _alloc_conv3x3(self, w: torch.Tensor, b: torch.Tensor) -> dict:
        """w: (Cout, Cin, 3, 3) -> weights packed per SRAM-conv plan.

        Weight matrix of group g is [Cout, K_g] = its taps' (Cout, Cin) slices concatenated along
        K in tap order; the LAST group carries an extra 64-wide block whose column 0 is the bias
        (the staging row has a matching 1.0 there). Taps whose weights are identically zero
        (stem sub-pixel planes) are dropped. Returns {"groups": [(taps, addr, K_g)], "N_s": N_s}.
        """
        Cout, Cin = w.shape[:2]
        active = [t for t in range(9) if bool(w[:, :, t // 3, t % 3].abs().sum() > 0)]
        plan, N_s = self._conv_plan({"groups": [(active, None, None)]}, Cin, Cout)
        groups = []
        for gi, (taps, Kg) in enumerate(plan):
            blocks = [w[:, :, t // 3, t % 3] for t in taps]
            if gi == len(plan) - 1:
                bias_blk = torch.zeros(Cout, 64)
                bias_blk[:, 0] = b
                blocks.append(bias_blk)
            B = torch.cat(blocks, dim=1).contiguous()
            assert B.shape[1] == Kg
            groups.append((taps, self._alloc_param(B), Kg))
        return {"groups": groups, "N_s": N_s, "Cin": Cin, "Cout": Cout}

    def _conv_shard(self, ei: int, conv: dict, go: Grid):
        """(output rows h0..h1, N-strip indices) this engine owns. Big-weight convs split output
        channels across engines (each streams only its strips' weights) and rows across the rest;
        small-weight convs split rows only."""
        Cout, N_s = go.C, conv["N_s"]
        n_strips = Cout // N_s
        w_elems = sum(Cout * Kg for _, _, Kg in conv["groups"])
        n_col = 1
        if w_elems * _BPE > self.COL_SHARD_MIN_WEIGHT_BYTES:
            n_col = max(d for d in (1, 2, 4, 8) if d <= self.SNE and self.SNE % d == 0 and n_strips % d == 0)
        n_row = self.SNE // n_col
        cs, rs = ei % n_col, ei // n_col
        base, rem = divmod(go.H, n_row)
        h0 = rs * base + min(rs, rem)
        h1 = h0 + base + (1 if rs < rem else 0)
        return h0, h1, range(cs * n_strips // n_col, (cs + 1) * n_strips // n_col)

    @staticmethod
    def _tap_runs(taps: list[int], Cin: int):
        """Consecutive dx taps of one dy row are ONE contiguous input run -> one SRAM copy each.
        Returns [(dy, dx_first, run_len, k_col)]."""
        out, j = [], 0
        while j < len(taps):
            t, run = taps[j], 1
            while j + run < len(taps) and taps[j + run] == t + run and (t + run) // 3 == t // 3:
                run += 1
            out.append((t // 3 - 1, t % 3 - 1, run, j * Cin))
            j += run
        return out

    def conv3x3_rows(self, ue, ei, in_buf, out_buf, g: Grid, go: Grid, conv: dict, relu: bool,
                     stride: int = 1, residual: int | None = None) -> None:
        """3x3 conv (pad 1, stride 1|2) for this engine's shard, entirely on-chip.

        Per chunk of output rows: the input grid rows those outputs read are DMA'd into URAM A
        once; for each output PIXEL the 3x3 window is assembled in a staging row by SRAM copies
        (eltwise add against a zero row in URAM B -- the only on-chip copy primitive), then one
        matvec per (N-strip, K-group) against the resident weight strip. Bias rides in the
        weights' ones-column block. The bias BRAM carries the residual (group 0) and the K-group
        partials (spilled per pixel to a per-engine DRAM row and reloaded), so the last group's
        matvec sees conv + b + residual and applies the ReLU in its clamp epilogue. Only real
        pixels are computed and written back (strided DMA per row per strip): borders stay zero.
        """
        from user_dma_core import LALU_MODE
        Cin, Cout, Wp_in, W = g.C, go.C, g.Wp, go.W
        groups, N_s = conv["groups"], conv["N_s"]
        h0, h1, strips = self._conv_shard(ei, conv, go)
        if h1 <= h0:
            return
        Kmax = max(Kg for _, _, Kg in groups)
        Kst = sum(Kg for _, _, Kg in groups)                # staging row holds every group's block
        g_runs = [self._tap_runs(taps, Cin) for taps, _, _ in groups]
        g_kofs = [sum(Kg for _, _, Kg in groups[:i]) for i in range(len(groups))]
        spill = self.KSPLIT_SPILL[self.EOFF + ei]
        lalu_a, lalu_b = ue.float_to_bf16(0.0), ue.float_to_bf16(float("inf"))

        # URAM B: weight strip + zero row.  URAM A: input rows, staging row, output chunk.
        B_W, B_Z = 0x80000, 0x80000 + N_s * Kmax * _BPE
        assert N_s * Kmax + 3 * Cin <= URAM_NEAR_FULL_ELEMENTS

        def a_need(rows):
            nin = (rows + 2 if stride == 1 else 2 * rows + 2) * Wp_in * Cin
            return nin + Kst + rows * W * N_s
        rows_per = h1 - h0
        while rows_per > 1 and a_need(rows_per) > URAM_NEAR_FULL_ELEMENTS:
            rows_per -= 1
        assert a_need(rows_per) <= URAM_NEAR_FULL_ELEMENTS, (rows_per, a_need(rows_per))
        ue.accelerator_memory_to_sram(self.ZERO_ROW, B_Z, 3 * Cin)

        for c0 in range(h0, h1, rows_per):
            rows = min(h1, c0 + rows_per) - c0
            m = rows * W
            in_row0 = c0 if stride == 1 else 2 * c0
            nin_rows = rows + 2 if stride == 1 else 2 * rows + 2
            A_IN = 0x00000
            A_ST = nin_rows * Wp_in * Cin * _BPE
            A_OUT = A_ST + ((Kst * _BPE + 127) // 128) * 128
            ue.accelerator_memory_to_sram(in_buf + in_row0 * Wp_in * Cin * _BPE, A_IN, nin_rows * Wp_in * Cin)
            ue.accelerator_memory_to_sram(self.ONES_BLOCK, A_ST + (Kst - 64) * _BPE, 64)
            for si in strips:
                for gi, (taps, B_addr, Kg) in enumerate(groups):
                    last_g = gi == len(groups) - 1
                    ue.accelerator_memory_to_sram(B_addr + si * N_s * Kg * _BPE, B_W, N_s * Kg)
                    for i in range(m):
                        h, w = c0 + i // W, i % W
                        for dy, dx, run, kcol in g_runs[gi]:
                            if stride == 1:
                                pix = (h - c0 + 1 + dy) * Wp_in + (w + 1 + dx)
                            else:
                                pix = (2 * (h - c0) + 1 + dy) * Wp_in + (2 * w + 1 + dx)
                            ue.eltwise_add_core(A_IN + pix * Cin * _BPE, B_Z, A_ST + (g_kofs[gi] + kcol) * _BPE, run * Cin)
                        use_bias = True
                        if gi == 0 and residual is not None:
                            ue.accelerator_memory_to_bias_sram(residual + (go.idx(h, w) * Cout + si * N_s) * _BPE, N_s)
                        elif gi > 0:
                            ue.accelerator_memory_to_bias_sram(spill + i * N_s * _BPE, N_s)
                        else:
                            use_bias = False
                        self._flops += 2 * Kg * N_s
                        ue.start_queue_for_bf16_matvec_operation(
                            max_clear_en=0, fmax_context_addr=0,
                            vector_sram_start_addr=A_ST + g_kofs[gi] * _BPE, matrix_sram_start_addr=B_W,
                            output_sram_wb_addr=A_OUT + i * N_s * _BPE, K=Kg, N=N_s,
                            bias_enable=use_bias,
                            lalu_mode=LALU_MODE.CLAMP if (relu and last_g) else LALU_MODE.BYPASS,
                            lalu_a=lalu_a, lalu_b=lalu_b)
                        if not last_g:
                            ue.sram_to_accelerator_memory(A_OUT + i * N_s * _BPE, spill + i * N_s * _BPE, N_s)
                for r in range(rows):       # write back this strip's rows, real pixels only
                    ue.sram_to_accelerator_memory(A_OUT + r * W * N_s * _BPE,
                                                  out_buf + (go.idx(c0 + r, 0) * Cout + si * N_s) * _BPE, W * N_s,
                                                  stride_bytes_per_chunk=N_s * _BPE, stride_jump_bytes=Cout * _BPE)

    def conv1x1_s2_rows(self, ue, ei, in_buf, out_buf, g: Grid, go: Grid, w, bias) -> None:
        """Downsample 1x1/s2 for this engine's output rows: strided gather -> dense scratch -> matmul."""
        h0, h1 = self._split(go.H, ei)
        if h1 <= h0:
            return
        for h in range(h0, h1):
            src = in_buf + g.idx(2 * h, 0) * g.C * _BPE
            ue.accelerator_memory_to_sram(src, 0x00000, go.W * g.C,
                                          stride_bytes_per_chunk=g.C * _BPE, stride_jump_bytes=2 * g.C * _BPE)
            ue.sram_to_accelerator_memory(0x00000, self.GATHER_DRAM + go.idx(h, 0) * g.C * _BPE, go.W * g.C)
        r0, M = h0 * go.Wp, (h1 - h0) * go.Wp
        self.mm(ue, M, g.C, go.C, self.GATHER_DRAM + (go.idx(0, 0) + r0) * g.C * _BPE, w,
                out_buf + (go.idx(0, 0) + r0) * go.C * _BPE, C=bias)
        self.zero_borders(ue, out_buf, go, h0, h1)

    def maxpool_rows(self, ue, ei, planes: list[int], out_buf: int, tmp_buf: int, g: Grid) -> None:
        """3x3/s2/pad1 max over the 240x320 stem output stored as 4 sub-pixel planes (plane
        a'*2+b' holds pixel (2H+a', 2W+b')), for this engine's output rows. Output (h, w) reads
        pixel rows 2h+dy: dy=-1 -> plane row 1 at super-row h-1; dy=0/1 -> plane 0/1 at h.
        Border planes are zero and inputs are post-ReLU so a zero pad equals -inf pad.

        SRAM-resident: the running max lives in URAM B for a chunk of rows; each tap is DMA'd
        into URAM A and folded with max(a, out) = out + relu(a - out), where the ReLU is the
        identity-matvec clamp (LALU CLAMP only exists on the matvec path). DRAM traffic is
        one read per tap + one write, instead of 24 full passes."""
        h0, h1 = self._split(g.H, ei)
        if h1 <= h0:
            return
        C = g.C
        assert C == 64
        r0, M = h0 * g.Wp, (h1 - h0) * g.Wp
        base = g.idx(0, 0) + r0
        taps = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                pa, oh = (1, -1) if dy == -1 else (dy, 0)
                pb, ow = (1, -1) if dx == -1 else (dx, 0)
                taps.append(planes[pa * 2 + pb] + (base + oh * g.Wp + ow) * C * _BPE)
        ROWB = C * _BPE
        CH = 1024                                   # rows per chunk
        A_TAP, A_TMP, A_RELU = 0x00000, CH * ROWB, 2 * CH * ROWB
        B_OUT, B_EYE = 0x80000, 0x80000 + CH * ROWB
        ue.accelerator_memory_to_sram(self.IDENTITY_64, B_EYE, 64 * 64)
        lalu_a, lalu_b = ue.float_to_bf16(0.0), ue.float_to_bf16(float("inf"))
        from user_dma_core import LALU_MODE
        for c0 in range(0, M, CH):
            m = min(CH, M - c0)
            ue.accelerator_memory_to_sram(taps[0] + c0 * ROWB, B_OUT, m * C)
            for a in taps[1:]:
                ue.accelerator_memory_to_sram(a + c0 * ROWB, A_TAP, m * C)
                ue.eltwise_sub_core(A_TAP, B_OUT, A_TMP, m * C)              # tmp = a - out
                for i in range(m):                                           # relu(tmp) row by row
                    ue.start_queue_for_bf16_matvec_operation(
                        max_clear_en=0, fmax_context_addr=0,
                        vector_sram_start_addr=A_TMP + i * ROWB, matrix_sram_start_addr=B_EYE,
                        output_sram_wb_addr=A_RELU + i * ROWB, K=64, N=64,
                        lalu_mode=LALU_MODE.CLAMP, lalu_a=lalu_a, lalu_b=lalu_b)
                ue.eltwise_add_core(A_RELU, B_OUT, B_OUT, m * C)              # out += relu(a - out)
            ue.sram_to_accelerator_memory(B_OUT, out_buf + (base + c0) * ROWB, m * C)
        self.zero_borders(ue, out_buf, g, h0, h1)

    def _block_stages(self, x_in, bufs, g_in: Grid, g_out: Grid, w, stride):
        """ResNet BasicBlock as two stages (callables of (ue, ei)): conv1 (+downsample) | conv2+res."""
        if stride == 1:
            s1 = lambda ue, ei: self.conv3x3_rows(ue, ei, x_in, bufs["T"], g_in, g_out, w["conv1"], True)
        else:
            def s1(ue, ei):
                self.conv3x3_rows(ue, ei, x_in, bufs["T"], g_in, g_out, w["conv1"], True, stride=2)
                self.conv1x1_s2_rows(ue, ei, x_in, bufs["R"], g_in, g_out, w["down"], w["bd"])
        res = x_in if stride == 1 else bufs["R"]
        s2 = lambda ue, ei: self.conv3x3_rows(ue, ei, bufs["T"], bufs["out"], g_out, g_out, w["conv2"], True, residual=res)
        return [s1, s2], bufs["out"]

    def backbone_stages(self, cam: int) -> list:
        """ResNet-18 + token projection for one camera as an ordered list of stage callables.
        Stage k only reads buffers written by stages < k, so consecutive stages need a barrier
        and nothing inside a stage does. Per-camera buffers keep the two cameras independent."""
        g = self.IMG
        st = [lambda ue, ei: [self.conv3x3_rows(ue, ei, self.IMG_DRAM[cam], self.STEM_PLANES[cam][sub], g, g, self.STEM[sub], True)
                              for sub in range(4)],
              lambda ue, ei: self.maxpool_rows(ue, ei, self.STEM_PLANES[cam], self.POOL_DRAM[cam], self.POOL_TMP[cam], self.POOL_G)]
        x, g_in = self.POOL_DRAM[cam], self.POOL_G
        for si, (name, cout, stride, H, W) in enumerate(self.STAGES):
            g_out, sb, wb = self.STAGE_G[si], self.STAGE_BUF[cam][si], self.BB[si]
            s_, x = self._block_stages(x, {"T": sb["T"], "R": sb["R"], "out": sb["A"]}, g_in, g_out, wb[0], stride)
            st += s_
            s_, x = self._block_stages(x, {"T": sb["T"], "R": sb["R"], "out": sb["B"]}, g_out, g_out, wb[1], 1)
            st += s_
            g_in = g_out
        D, g4, feat = self.D, self.STAGE_G[-1], x

        def tokens(ue, ei):
            # padded grid -> dense (h w) token rows -> 1x1 projection straight into X
            h0, h1 = self._split(self.FEAT_H, ei)
            for h in range(h0, h1):
                self.memcpy(ue, feat + g4.idx(h, 0) * D * _BPE, self.FEAT_DENSE[cam] + h * self.FEAT_W * D * _BPE,
                            self.FEAT_W * D)
            n = (h1 - h0) * self.FEAT_W
            self.mm(ue, n, D, D, self.FEAT_DENSE[cam] + h0 * self.FEAT_W * D * _BPE, self.IMG_PROJ[0],
                    self.X + (2 + cam * self.FEAT_H * self.FEAT_W + h0 * self.FEAT_W) * D * _BPE, C=self.IMG_PROJ[1])
        st.append(tokens)
        return st

    def backbones(self, stop_after_stem: bool = False) -> None:
        """Both cameras' backbones concurrently, each on its own engine group (NE/2 engines):
        halves the barrier count and keeps every engine busy on every stage. (A one-stage stagger
        to overlap DRAM-bound and compute-bound stages was measured slower: sum of pairwise maxima
        exceeds the sum of stages.) With NE < 2 the cameras run sequentially on all engines."""
        stages = [self.backbone_stages(c) for c in range(self.N_CAMS)]
        if stop_after_stem:
            stages = [st[:1] for st in stages]
        if self.NE < 2:
            for cam in range(self.N_CAMS):
                self.GATHER_DRAM = self.GATHER_DRAM_ALL[cam]
                for stage in stages[cam]:
                    self.region(stage)
            return
        half = self.NE // 2
        n = len(stages[0])
        for r in range(n):                          # region r: stage r of both cameras

            def body(ue, ei, r=r):
                cam, local = ei // half, ei % half
                if local < half:
                    self.SNE, self.GATHER_DRAM, self.EOFF = half, self.GATHER_DRAM_ALL[cam], cam * half
                    try:
                        stages[cam][r](ue, local)
                    finally:
                        self.SNE, self.GATHER_DRAM, self.EOFF = self.NE, self.GATHER_DRAM_ALL[0], 0
            self.region(body)

    # ------------------------------------------------------------------
    # Transformer (token rows sharded; attention sharded by head)
    # ------------------------------------------------------------------

    def _rows(self, ei, total, width):
        """(row0, nrows, byte offset) of this engine's token-row shard of a [total, width] buffer."""
        r0, r1 = self._split(total, ei)
        return r0, r1 - r0, r0 * width * _BPE

    def _heads_proj(self, ue, ei, total, SRC_QK, SRC_V, w, QH, KH, VH, want=("q", "k", "v")):
        """Q/K/V for this engine's token rows: one full-width matmul each into PF, then the 8
        head column blocks are scattered to the per-head [rows, 64] buffers with strided reads."""
        D, HD = self.D, self.HD
        r0, n, offD = self._rows(ei, total, D)
        offH = r0 * HD * _BPE
        for name, src, dst in (("q", SRC_QK, QH), ("k", SRC_QK, KH), ("v", SRC_V, VH)):
            if name not in want:
                continue
            wf, bf = w[name]
            self.mm(ue, n, D, D, src + offD, wf, self.PF + offD, C=bf)
            for h in range(self.NH):
                ue.accelerator_memory_to_sram(self.PF + offD + h * HD * _BPE, 0x00000, n * HD,
                                              stride_bytes_per_chunk=HD * _BPE, stride_jump_bytes=D * _BPE)
                ue.sram_to_accelerator_memory(0x00000, dst[h] + offH, n * HD)

    def _out_proj(self, ue, ei, total, w, AH, OUT):
        """Gather the 8 per-head [rows, 64] outputs into PF [rows, 512], then one K=512 matmul."""
        D, HD = self.D, self.HD
        r0, n, offD = self._rows(ei, total, D)
        offH = r0 * HD * _BPE
        for h in range(self.NH):
            ue.accelerator_memory_to_sram(AH[h] + offH, 0x00000, n * HD)
            ue.sram_to_accelerator_memory(0x00000, self.PF + offD + h * HD * _BPE, n * HD,
                                          stride_bytes_per_chunk=HD * _BPE, stride_jump_bytes=D * _BPE)
        self.mm(ue, n, D, D, self.PF + offD, w["o"][0], OUT + offD, C=w["o"][1])

    def _heads_attention(self, ue, ei, q_rows, kv_rows, QH, KH, VH, MASK, AH):
        """Head h on engine h % NE (8 heads over 8 engines = one head each)."""
        for h in range(self.NH):
            if h % self.NE == ei:
                self.attention(ue, ei, q_rows, kv_rows, QH[h], KH[h], VH[h], MASK, AH[h])

    def encoder_layer(self, w):
        SP, D, FF = self.SP, self.D, self.FF

        def proj(ue, ei):
            r0, n, offD = self._rows(ei, SP, D)
            self.add(ue, n, D, self.X + offD, self.ENC_POS + offD, self.QK + offD)       # q = k = x + pos
            self._heads_proj(ue, ei, SP, self.QK, self.X, w["attn"], self.QH, self.KH, self.VH)
        self.region(proj)
        self.region(lambda ue, ei: self._heads_attention(ue, ei, SP, SP, self.QH, self.KH, self.VH, self.ENC_MASK, self.AH))

        def post(ue, ei):
            r0, n, offD = self._rows(ei, SP, D)
            offF = r0 * FF * _BPE
            self._out_proj(ue, ei, SP, w["attn"], self.AH, self.O)
            self.add(ue, n, D, self.O + offD, self.X + offD, self.O + offD)
            self.ln(ue, n, self.O + offD, self.X1 + offD, w["ln1"])
            self.mm(ue, n, D, FF, self.X1 + offD, w["ffn1"][0], self.H1 + offF, C=w["ffn1"][1], relu=True)
            self.mm(ue, n, FF, D, self.H1 + offF, w["ffn2"][0], self.O + offD, C=w["ffn2"][1])
            self.add(ue, n, D, self.O + offD, self.X1 + offD, self.O + offD)
            self.ln(ue, n, self.O + offD, self.X + offD, w["ln2"])
        self.region(post)

    def decoder_layer(self, w, X_IN, X_OUT):
        SP, CHP, D, FF = self.SP, self.CHP, self.D, self.FF

        def self_proj(ue, ei):
            r0, n, offD = self._rows(ei, CHP, D)
            self.add(ue, n, D, X_IN + offD, self.DEC_POS + offD, self.DQK + offD)
            self._heads_proj(ue, ei, CHP, self.DQK, X_IN, w["self"], self.DQH, self.DKH, self.DVH)
        self.region(self_proj)
        self.region(lambda ue, ei: self._heads_attention(ue, ei, CHP, CHP, self.DQH, self.DKH, self.DVH, self.DEC_MASK, self.DAH))

        def cross_proj(ue, ei):
            r0, n, offD = self._rows(ei, CHP, D)
            self._out_proj(ue, ei, CHP, w["self"], self.DAH, self.DO)
            self.add(ue, n, D, self.DO + offD, X_IN + offD, self.DO + offD)
            self.ln(ue, n, self.DO + offD, self.D1 + offD, w["ln1"])
            # cross-attention inputs: q = d1 + dec_pos (own rows), k = enc + enc_pos, v = enc (token rows)
            self.add(ue, n, D, self.D1 + offD, self.DEC_POS + offD, self.DQK + offD)
            self._heads_proj(ue, ei, CHP, self.DQK, None, w["cross"], self.DQH, None, None, want=("q",))
            s0, sn, soffD = self._rows(ei, SP, D)
            self.add(ue, sn, D, self.X + soffD, self.ENC_POS + soffD, self.QK + soffD)
            self._heads_proj(ue, ei, SP, self.QK, self.X, w["cross"], None, self.KH, self.VH, want=("k", "v"))
        self.region(cross_proj)
        self.region(lambda ue, ei: self._heads_attention(ue, ei, CHP, SP, self.DQH, self.KH, self.VH, self.ENC_MASK, self.DAH))

        def post(ue, ei):
            r0, n, offD = self._rows(ei, CHP, D)
            offF = r0 * FF * _BPE
            self._out_proj(ue, ei, CHP, w["cross"], self.DAH, self.DO)
            self.add(ue, n, D, self.DO + offD, self.D1 + offD, self.DO + offD)
            self.ln(ue, n, self.DO + offD, self.D2 + offD, w["ln2"])
            self.mm(ue, n, D, FF, self.D2 + offD, w["ffn1"][0], self.DH1 + offF, C=w["ffn1"][1], relu=True)
            self.mm(ue, n, FF, D, self.DH1 + offF, w["ffn2"][0], self.DO + offD, C=w["ffn2"][1])
            self.add(ue, n, D, self.DO + offD, self.D2 + offD, self.DO + offD)
            self.ln(ue, n, self.DO + offD, X_OUT + offD, w["ln3"])
        self.region(post)

    # ------------------------------------------------------------------
    # Whole program
    # ------------------------------------------------------------------

    def compile_full_fused(self, stop_after: str | None = None) -> int:
        """stop_after (profiling): "stem" | "backbone0" | "backbone" | "encoder" -> HALT early."""
        self._flops = 0            # model FLOPs emitted (matmuls, attention, SRAM-conv matvecs)
        self._params_weights_end = self.get_params_dram_usage()   # kernels may allocate constants past here
        self.start_capture()
        self._pbi_M_reg = self.alloc_isa_reg()
        if self.sched is not None:
            self.sched.begin_program()
        D, SP = self.D, self.SP
        g4 = self.STAGE_G[-1]

        if stop_after == "stem":
            self.backbones(stop_after_stem=True)
            return self._finish_program()
        if stop_after == "backbone0":
            stop_after = "backbone"          # cameras run concurrently now; no cam0-only prefix
        self.backbones()
        if stop_after == "backbone":
            return self._finish_program()
        self.mm(self, 1, 64, D, self.LATENT_IN, self.LATENT_PROJ[0], self.X, C=self.LATENT_PROJ[1])
        self.mm(self, 1, 64, D, self.STATE_IN, self.STATE_PROJ[0], self.X + D * _BPE, C=self.STATE_PROJ[1])

        for w in self.ENC:
            self.encoder_layer(w)

        if stop_after == "encoder":
            return self._finish_program()
        x_in = self.DIN
        for i, w in enumerate(self.DEC):
            self.decoder_layer(w, x_in, self.D3)
            x_in = self.D3

        def head(ue, ei):
            r0, n, offD = self._rows(ei, self.CHP, D)
            self.ln(ue, n, self.D3 + offD, self.DEC_OUT + offD, self.DEC_NORM)
            self.mm(ue, n, D, 64, self.DEC_OUT + offD, self.ACTION_HEAD[0], self.ACT_OUT + r0 * 64 * _BPE,
                    C=self.ACTION_HEAD[1])
        self.region(head)
        return self._finish_program()

    def _finish_program(self) -> int:
        if self.sched is not None:
            self.worker_progs = self.sched.finalize()
        self.stop_capture()
        self.generate_instruction_halt()
        self.last_num_instructions = self.capture_count
        prog_addr = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(prog_addr)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        self.clear_capture_buffer()
        return prog_addr

    # ------------------------------------------------------------------
    # Host-side I/O
    # ------------------------------------------------------------------

    def image_to_grid(self, img: np.ndarray) -> torch.Tensor:
        """(3, 480, 640) -> space-to-depth(4) padded grid (122*162, 64) bf16, channel (a*4+b)*3+c."""
        t = torch.from_numpy(np.ascontiguousarray(img)).float()
        C, H, W = t.shape
        s2d = t.reshape(C, H // 4, 4, W // 4, 4).permute(1, 3, 2, 4, 0).reshape(H // 4, W // 4, 48)
        g = self.IMG
        grid = torch.zeros(g.Hp, g.Wp, 64)
        grid[1:-1, 1:-1, :48] = s2d
        grid = torch.cat([grid.reshape(-1, 64), torch.zeros(g.rows - g.Hp * g.Wp, 64)])  # + slack rows
        return grid.to(torch.bfloat16)

    def grid_to_chw(self, buf: int, g: Grid) -> np.ndarray:
        t = self._read_bf16(buf, g.Hp * g.Wp * g.C).reshape(g.Hp, g.Wp, g.C)
        return t[1:-1, 1:-1].permute(2, 0, 1).numpy()

    def run(self, images: np.ndarray, state: np.ndarray, program_addr: int, timeout: float = 600.0):
        for cam in range(self.N_CAMS):
            self._write_bf16(self.IMG_DRAM[cam], self.image_to_grid(images[cam]))
        s = torch.zeros(64)
        s[:self.STATE_DIM] = torch.from_numpy(state).float()
        self._write_bf16(self.STATE_IN, s)
        t0 = time.perf_counter()
        if self.sched is not None:
            self.sched.start_workers(self.worker_progs)
        self.start_execute_from_dram(program_addr)
        self.wait_queue(timeout)
        self.last_inference_seconds = time.perf_counter() - t0
        out = self._read_bf16(self.ACT_OUT, self.CHP * 64).reshape(self.CHP, 64)
        return out[:self.CHUNK, :self.ACTION_DIM].numpy()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="ACT accelerator inference + timing.")
    ap.add_argument("--dev", default="xdma0")
    ap.add_argument("--device", default="kintex7")
    ap.add_argument("--cleanup", action="store_true", help="delete cached params/programs in act_bin/")
    ap.add_argument("--rows", type=int, default=5, help="action-chunk rows to print")
    ap.add_argument("--runs", type=int, default=3, help="timed HW runs after the first")
    ap.add_argument("--engines", type=int, default=2, help="number of engines to shard across (pass 8 for the full box)")
    ap.add_argument("--profile", action="store_true", help="time stem / backbone / encoder prefixes")
    ap.add_argument("--debug", action="store_true", help="per-stage SNRs vs the CPU reference + NaN localizer")
    args = ap.parse_args()

    bin_dir = os.path.join(SCRIPT_DIR, ACT_UnifiedEngine.BIN_SUBDIR)
    if args.cleanup:
        import shutil
        for n in ("params.bin", "params.json"):
            p = os.path.join(bin_dir, n)
            if os.path.exists(p):
                os.remove(p)
        for d in [x for x in os.listdir(bin_dir) if x.startswith("programs_e")] if os.path.isdir(bin_dir) else []:
            shutil.rmtree(os.path.join(bin_dir, d))
        _original_print("[cleanup] removed cached params/programs")

    ref_path = os.path.join(bin_dir, "reference.npz")
    if not os.path.exists(ref_path):
        raise SystemExit(f"missing {ref_path}: run act_export.py in the lerobot env first")
    ref = np.load(ref_path)

    global _SILENT_MODE
    _SILENT_MODE = True
    set_dma_device("efinix" if args.device == "efinix" else args.dev)
    global DMA_DEVICE_H2C, DMA_DEVICE_C2H
    DMA_DEVICE_H2C = user_dma_core.DMA_DEVICE_H2C
    DMA_DEVICE_C2H = user_dma_core.DMA_DEVICE_C2H
    user_dma_core.configure_clock_from_hardware()
    _original_print(user_dma_core.hardware_info_summary())

    t0 = time.perf_counter()
    ue = ACT_UnifiedEngine(num_engines=args.engines)
    ue.software_reset()
    if ue.sched is not None:
        ue.sched.preclear_flags()
    t1 = time.perf_counter()
    _original_print(f"  Weights+tensors: {t1 - t0:.2f}s  (params {ue.get_params_dram_usage() / 1024**2:.1f} MB, "
                    f"tensors {ue.get_tensor_dram_usage() / 1024**2:.1f} MB)")

    if args.profile:
        if ue.params_from_bin:
            # profiling always compiles: rebuild the weight addresses (same order -> same DRAM)
            ue.reset_params_dram_addr()
            ue.weight_init()
        # peak = engines x 64 MACs x 2 FLOP x clock
        clk_ghz = 1.0 / (ue._clock_period_ns or 2.727)
        peak = args.engines * 64 * 2 * clk_ghz          # GFLOPS
        _original_print(f"  [profile] peak {peak:.1f} GFLOPS ({args.engines} engines x 64 MAC @ {clk_ghz*1e3:.0f} MHz)")
        _original_print(f"  {'section':12s} {'ms':>8s} {'GFLOP':>8s} {'GFLOPS':>8s} {'util':>6s}")
        prev_t, prev_f, last = 0.0, 0, "start"
        for stage in ("stem", "backbone", "encoder", None):
            ue.reset_isa_reg_counter(); ue.reset_inst_ptr_counter(); ue._pbi_M_reg = None
            if ue.sched is not None:
                for w_ in ue.sched.workers:
                    w_._pbi_M_reg = None
            pa = ue.compile_full_fused(stop_after=stage)
            ue.run(ref["images"], ref["state"], pa); ue.run(ref["images"], ref["state"], pa)
            t, f = ue.last_inference_seconds * 1000, ue._flops
            dt, df = t - prev_t, f - prev_f
            name = {"stem": "stem x2cam", "backbone": "resnet x2cam", "encoder": "encoder x4",
                    None: "decoder+head"}[stage]
            gf = df / 1e9
            _original_print(f"  {name:12s} {dt:8.1f} {gf:8.2f} {gf / (dt / 1e3):8.1f} {100 * gf / (dt / 1e3) / peak:5.0f}%")
            prev_t, prev_f = t, f
        gf = prev_f / 1e9
        _original_print(f"  {'TOTAL':12s} {prev_t:8.1f} {gf:8.2f} {gf / (prev_t / 1e3):8.1f} {100 * gf / (prev_t / 1e3) / peak:5.0f}%")
        return
    prog_addr = ue.load_programs() if ue.USE_BIN_CACHE else None
    if prog_addr is None:
        if ue.params_from_bin:
            # weights are in DRAM but their compile-time addresses are not: rebuild them
            ue.reset_params_dram_addr()
            ue.weight_init()
        prog_addr = ue.compile_full_fused()
        t2 = time.perf_counter()
        _original_print(f"  Compile: {t2 - t1:.2f}s  ({ue.last_num_instructions:,} primary instructions, "
                        f"{ue.get_program_dram_usage() / 1024**2:.1f} MB; {args.engines} engine(s))")
        if ue.USE_BIN_CACHE:
            if not ue.params_from_bin:
                ue.dump_params()
            ue.dump_programs(prog_addr)
    else:
        t2 = time.perf_counter()
        _original_print(f"  Program load: {t2 - t1:.2f}s  ({ue.last_num_instructions:,} primary instructions; "
                        f"{args.engines} engine(s), cached)")

    actions = ue.run(ref["images"], ref["state"], prog_addr)
    times = [ue.last_inference_seconds]
    for _ in range(args.runs):
        ue.run(ref["images"], ref["state"], prog_addr)
        times.append(ue.last_inference_seconds)
    _original_print(f"  Inference (pure HW): first {times[0] * 1000:.1f} ms, "
                    f"steady {np.median(times[1:]) * 1000 if len(times) > 1 else float('nan'):.1f} ms")

    results = {"actions_snr_db": snr_db(ref["actions"], actions),
               "hw_ms": float(np.median(times[1:]) * 1000 if len(times) > 1 else times[0] * 1000)}
    _original_print(f"  actions SNR: {results['actions_snr_db']:.1f} dB")

    if args.debug:
        g4 = ue.STAGE_G[-1]
        feat_hw = ue.grid_to_chw(ue.STAGE_BUF[-1][-1]["B"], g4)
        results["feat_cam1_snr_db"] = snr_db(ref["feat"][-1], feat_hw)
        results["pool_cam1_snr_db"] = snr_db(ref["pool"][-1], ue.grid_to_chw(ue.POOL_DRAM[-1], ue.POOL_G))
        for si, name in enumerate(("layer1", "layer2", "layer3")):
            results[f"{name}_cam1_snr_db"] = snr_db(ref[name][-1], ue.grid_to_chw(ue.STAGE_BUF[-1][si]["B"], ue.STAGE_G[si]))
        enc_hw = ue._read_bf16(ue.X, ue.SP * ue.D).reshape(ue.SP, ue.D)[:ue.S].numpy()
        results["enc_out_snr_db"] = snr_db(ref["enc_out"], enc_hw)
        dec_hw = ue._read_bf16(ue.DEC_OUT, ue.CHP * ue.D).reshape(ue.CHP, ue.D)[:ue.CHUNK].numpy()
        results["dec_out_snr_db"] = snr_db(ref["dec_out"], dec_hw)  # both post decoder.norm
        for k, v in results.items():
            if k.endswith("_snr_db"):
                _original_print(f"  {k}: {v:.1f} dB")
        # NaN localizer: first bad buffer per stage (T = conv1 out, R = downsample, A/B = block0/1 out)
        _original_print("  non-finite counts per stage buffer (last cam):")
        for si, (name, *_ ) in enumerate(ue.STAGES):
            g = ue.STAGE_G[si]
            cnt = {k: int((~torch.isfinite(ue._read_bf16(ue.STAGE_BUF[-1][si][k], g.Hp * g.Wp * g.C))).sum())
                   for k in ("T", "R", "A", "B")}
            _original_print(f"    {name} {g.H}x{g.W}x{g.C}: {cnt}")

    np.set_printoptions(precision=4, suppress=True, linewidth=140)
    _original_print(f"\n  Action chunk ({ue.CHUNK} x {ue.ACTION_DIM}), first {args.rows} rows, CPU reference | HW:")
    for t in range(min(args.rows, ue.CHUNK)):
        _original_print(f"  t={t:3d}  {ref['actions'][t]}  |  {actions[t]}")
    _original_print(f"  ... max |CPU-HW| over the chunk: {float(np.abs(ref['actions'] - actions).max()):.4f}")
    _original_print("TEST_RESULT:" + json.dumps(results))


if __name__ == "__main__":
    main()
