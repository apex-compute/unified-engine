#!/usr/bin/env python3
"""The Talker on the accelerator: staging, program emission, and the decode loop.

THE TALKER IS A QWEN2 DECODER AND NOTHING MORE, which is what makes this a port
rather than a new kernel set. Per layer: RMSNorm, GQA q/k/v/o where q/k/v carry
biases, RMSNorm, and a gate/up/down MLP. 24 layers at H=896, 12 query heads over
4 KV heads of 128, MLP width 18944.

What surrounds those layers is the only unusual part, and it is two lines of
arithmetic rather than a new datapath:

    step:  x = codec_embed[id] + thinker_reply_part[i]      # 3584, eltwise add
           x = thinker_to_talker_proj(x) + bias             # 3584 -> 896
           ... 24 layers ...
           logits = codec_head(model.norm(x))               # 896 -> 8448

The Thinker's reply is consumed one row per step -- the Talker is predicting
speech for text the Thinker has already produced -- so the "dual stream" is a
vector add of two 3584-element rows before the projection.

WHAT STAYS ON THE HOST, and why it is small: the codec embedding lookup and
that add (one 3584 row, 7 KiB DMA'd in), and sampling from the 8448 codec
logits (17 KiB read back). The reference samples with top_k/top_p/temperature,
which the device decode path does not do -- it is argmax only -- and matching
the reference matters more here than saving a 17 KiB readback. Everything with
FLOPs in it, the 1.35 B parameters, runs on the board.

WEIGHTS ARE SHARED, NOT PRIVATE, deliberately. The Thinker already holds
5659 MiB of private shards and core 0's window has no room left, so 644 MiB of
IF4 Talker weights cannot be copied per core. They go in the shared pool as
per-layer sections, and the engines take zero-copy column blocks of them --
ColumnShardContext.b_addr is exactly that, a shifted base into a shared weight
rather than a duplicated one.
"""

from __future__ import annotations

import json
import os
import time

import torch

import quant_lib
import user_dma_core
from user_dma_core import DMA_DEVICE_H2C, TYPE, UE_MODE

TALKER_PREFIX = "talker."
BLOCK = 64

# Matrices worth quantizing. The norms and biases are tiny and stay bf16, where
# their precision is free.
_IF4_SUFFIXES = (
    "codec_head.weight",
    "self_attn.q_proj.weight", "self_attn.k_proj.weight",
    "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
)


class TalkerWeights:
    """Talker parameters, quantized once and resident in accelerator DRAM."""

    def __init__(self, ue, model_dir: str, *, verbose: bool = True):
        self.ue = ue
        self.model_dir = model_dir
        self.verbose = verbose
        index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
        self._map = {n: s for n, s in index["weight_map"].items()
                     if n.startswith(TALKER_PREFIX)}
        self.addr: dict[str, int] = {}     # tensor name -> DRAM address
        self.scale: dict[str, int] = {}    # tensor name -> scale blob address
        self.shape: dict[str, tuple] = {}
        self._staged_bytes = 0

    def _tensor(self, name: str) -> torch.Tensor:
        from safetensors import safe_open
        with safe_open(os.path.join(self.model_dir, self._map[name]),
                       framework="pt") as f:
            return f.get_tensor(name)

    def _put_bf16(self, name: str, t: torch.Tensor) -> None:
        blob = t.to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()
        a = self.ue.allocate_params_dram(len(blob), label=f"talker.{name}")
        if self.ue.dma_write(DMA_DEVICE_H2C, a, blob, len(blob)) != len(blob):
            raise IOError(f"talker {name}: short DMA")
        self.addr[name] = a
        self.shape[name] = tuple(t.shape)
        self._staged_bytes += len(blob)

    def _put_if4(self, name: str, t: torch.Tensor) -> None:
        data, scales = quant_lib.quantize("if4", t.float(), block_size=BLOCK)
        a = self.ue.allocate_params_dram(len(data), label=f"talker.{name}.data")
        if self.ue.dma_write(DMA_DEVICE_H2C, a, data, len(data)) != len(data):
            raise IOError(f"talker {name}: short data DMA")
        s = self.ue.allocate_params_dram(len(scales), label=f"talker.{name}.scale")
        if self.ue.dma_write(DMA_DEVICE_H2C, s, scales, len(scales)) != len(scales):
            raise IOError(f"talker {name}: short scale DMA")
        self.addr[name] = a
        self.scale[name] = s
        self.shape[name] = tuple(t.shape)
        self._staged_bytes += len(data) + len(scales)

    def stage(self, layers: int = 24) -> "TalkerWeights":
        t0 = time.perf_counter()
        for name in sorted(self._map):
            short = name[len(TALKER_PREFIX):]
            if ".layers." in short:
                li = int(short.split(".layers.")[1].split(".")[0])
                if li >= layers:
                    continue
            if short == "model.embed_tokens.weight":
                # Host-side lookup: the codec embedding is one 3584 row per
                # step, added to the reply row before the projection. Staging
                # the 57.8 MiB table on the device would buy nothing.
                continue
            t = self._tensor(name)
            if short.endswith(_IF4_SUFFIXES):
                self._put_if4(short, t)
            else:
                self._put_bf16(short, t)
            del t
        if self.verbose:
            print(f"  [Talker] staged {len(self.addr)} tensors, "
                  f"{self._staged_bytes / 2**20:.1f} MiB (IF4 matrices, bf16 norms "
                  f"and biases) in {time.perf_counter() - t0:.1f}s")
        return self


class TalkerRunner:
    """Emits and runs the Talker's layers on one engine.

    Single-engine first, and deliberately: the Talker's correctness is the hard
    part, and a wrong column split looks exactly like a wrong kernel. Once the
    numbers match the reference, the same bodies take a ColumnShardContext and
    the weights stay where they are -- they are shared, so an engine's block is
    a shifted base, not a copy.
    """

    H = 896
    QH, KVH, AHD = 12, 4, 128
    Q_SIZE, KV_SIZE = 1536, 512
    MLP = 18944
    VOCAB = 8448
    THINKER_H = 3584

    def __init__(self, ue, weights: TalkerWeights, *, max_ctx: int = 1024):
        self.ue = ue
        self.w = weights
        self.max_ctx = max_ctx
        self._alloc_tensors()

    def _alloc(self, n_elem: int, label: str) -> int:
        return self.ue.allocate_tensor_dram(n_elem * 2, label=label)

    def _alloc_tensors(self) -> None:
        H, C = self.H, self.max_ctx
        self.IO_A = self._alloc(H, "talker.io_a")
        self.IO_B = self._alloc(H, "talker.io_b")
        self.NORM = self._alloc(H, "talker.norm")
        self.Q = self._alloc(self.Q_SIZE, "talker.q")
        self.K = self._alloc(self.KV_SIZE, "talker.k")
        self.V = self._alloc(self.KV_SIZE, "talker.v")
        self.ATTN = self._alloc(self.Q_SIZE, "talker.attn")
        self.PROJ = self._alloc(H, "talker.proj")
        self.RESID = self._alloc(H, "talker.resid")
        self.GATE = self._alloc(self.MLP, "talker.gate")
        self.UP = self._alloc(self.MLP, "talker.up")
        self.DOWN = self._alloc(H, "talker.down")
        self.IN_3584 = self._alloc(self.THINKER_H, "talker.in3584")
        self.LOGITS = self._alloc(self.VOCAB, "talker.logits")
        # KV cache: one plane per layer, max_ctx rows of KV_SIZE.
        self.K_CACHE = self._alloc(24 * C * self.KV_SIZE, "talker.k_cache")
        self.V_CACHE = self._alloc(24 * C * self.KV_SIZE, "talker.v_cache")

    # -- emission ---------------------------------------------------------
    def _proj(self, *, M, K, N, src, wname, dst, bias=True, silu=False):
        """One IF4 projection through the STREAMING kernel.

        Not matmat_mul_core: its two-pass path dequantizes B into URAM, so it
        refuses K=18944 (down_proj) against URAM_NEAR_FULL_ELEMENTS=262080.
        quantized_matmat_core streams B instead and has no such ceiling, which
        is exactly why the Thinker's own multi-core decode requires
        --decode-kernel streaming. At M=1 it is also the faster of the two.
        """
        w = self.w
        bias_addr = w.addr.get(wname.replace(".weight", ".bias")) if bias else None
        return self.ue.quantized_matmat_core(
            M=M, K=K, N=N, A_DRAM_ADDR=src,
            B_DRAM_ADDR=w.addr[wname], OUTPUT_DRAM_ADDR=dst,
            SCALE_DRAM_ADDR=w.scale[wname], data_type=TYPE.IF4,
            C_DRAM_ADDR=bias_addr,
            bias_mode="broadcast_N" if bias_addr is not None else None,
            silu_enable=silu) or 0

    def _rms(self, *, M, N, src, dst, gamma):
        return self.ue.rms_norm_core_dram(
            M=M, N=N, A_DRAM_ADDR=src, OUTPUT_DRAM_ADDR=dst,
            GAMMA_DRAM_ADDR=self.w.addr[gamma]) or 0

    def emit_input_projection(self) -> int:
        """[1, 3584] -> [1, 896]: the Thinker-space input into Talker space."""
        return self.ue.matmat_mul_core(
            M=1, K=self.THINKER_H, N=self.H, A_DRAM_ADDR=self.IN_3584,
            B_DRAM_ADDR=self.w.addr["thinker_to_talker_proj.weight"],
            OUTPUT_DRAM_ADDR=self.IO_A, is_B_quantized=False,
            C_DRAM_ADDR=self.w.addr["thinker_to_talker_proj.bias"],
            bias_mode="broadcast_N") or 0

    def emit_head(self, src: int) -> int:
        """[1, 896] -> [1, 8448] codec logits, through the final norm."""
        f = self._rms(M=1, N=self.H, src=src, dst=self.NORM, gamma="model.norm.weight")
        return f + (self.ue.matmat_mul_core(
            M=1, K=self.H, N=self.VOCAB, A_DRAM_ADDR=self.NORM,
            B_DRAM_ADDR=self.w.addr["codec_head.weight"],
            OUTPUT_DRAM_ADDR=self.LOGITS, is_B_quantized=True,
            data_type=TYPE.IF4,
            SCALE_DRAM_ADDR=self.w.scale["codec_head.weight"]) or 0)

    def emit_mlp(self, li: int, src: int, dst: int) -> int:
        """Post-attention norm, gated SiLU MLP, and the residual.

        Qwen2's gate is SiLU, which the LALU has natively (silu_enable), so the
        activation fuses into the gate projection's writeback rather than
        costing a second pass.
        """
        pre = f"model.layers.{li}."
        f = self._rms(M=1, N=self.H, src=src, dst=self.NORM,
                      gamma=pre + "post_attention_layernorm.weight")
        f += self._proj(M=1, K=self.H, N=self.MLP, src=self.NORM,
                        wname=pre + "mlp.gate_proj.weight", dst=self.GATE,
                        bias=False, silu=True)
        f += self._proj(M=1, K=self.H, N=self.MLP, src=self.NORM,
                        wname=pre + "mlp.up_proj.weight", dst=self.UP, bias=False)
        self.ue.eltwise_core_dram(M=1, N=self.MLP, dram_a=self.GATE,
                                  dram_b=self.UP, dram_out=self.GATE,
                                  mode=UE_MODE.ELTWISE_MUL)
        f += self._proj(M=1, K=self.MLP, N=self.H, src=self.GATE,
                        wname=pre + "mlp.down_proj.weight", dst=self.DOWN, bias=False)
        self.ue.eltwise_core_dram(M=1, N=self.H, dram_a=src, dram_b=self.DOWN,
                                  dram_out=dst, mode=UE_MODE.ELTWISE_ADD)
        return f

    # -- attention --------------------------------------------------------
    KV_HEADS, GQA = 4, 3          # 12 query heads over 4 KV heads

    def _kv_plane(self, base: int, li: int, h: int) -> int:
        """Address of layer ``li``, KV head ``h``'s [max_ctx, 128] plane."""
        return base + ((li * self.KV_HEADS + h) * self.max_ctx) * self.AHD * 2

    def alloc_attention_scratch(self, aligned: int) -> None:
        """V^T + scores + scaled_q, sized for the longest KV this run allows.

        The score plane is [aligned, aligned], so this grows as the square of
        the codec context -- the same term that dominates the Thinker's map.
        """
        self.ALIGNED = aligned
        n = (self.AHD + aligned) * aligned + self.GQA * self.AHD
        self._scratch_bytes = n * 2
        self.SCRATCH = self._alloc(n, "talker.attn_scratch")
        self.BIAS = self._alloc(max(aligned * aligned, self.GQA * aligned),
                                "talker.attn_bias")
        self.IDENT = self._alloc(64 * 64, "talker.identity")
        self.ue.dma_to_accelerator_memory(
            self.IDENT, torch.eye(64, dtype=torch.bfloat16))

    def zero_state(self, layers: int = 24) -> None:
        """Zero the KV planes and the attention scratch.

        Not hygiene -- correctness. The kernel reads the whole 64-aligned KV
        tile, so rows past the live context are read whatever they contain, and
        fresh DRAM is full of 0xFFFF, which is bf16 NaN. The bias masks the
        SCORES, but V^T is built from every row, and 0 * NaN is NaN, so an
        unzeroed cache poisons the output even at positions the mask excludes.
        """
        import numpy as np
        for base in (self.K_CACHE, self.V_CACHE):
            n = layers * self.max_ctx * self.KV_SIZE
            self.ue.dma_write(DMA_DEVICE_H2C, base, bytes(n * 2), n * 2)
        self.ue.dma_write(DMA_DEVICE_H2C, self.SCRATCH,
                          bytes(self._scratch_bytes), self._scratch_bytes)

    def write_kv(self, li: int, pos: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Place one step's K/V into their per-head planes.

        The four KV heads live in separate planes so decode can append to each
        without walking off the end of head 0, so a step is four small writes
        rather than one contiguous 512-element store.
        """
        for h in range(self.KV_HEADS):
            off = pos * self.AHD * 2
            sl = slice(h * self.AHD, (h + 1) * self.AHD)
            self.ue.dma_to_accelerator_memory(
                self._kv_plane(self.K_CACHE, li, h) + off, k[sl].to(torch.bfloat16))
            self.ue.dma_to_accelerator_memory(
                self._kv_plane(self.V_CACHE, li, h) + off, v[sl].to(torch.bfloat16))

    def set_causal_bias(self, live: int) -> None:
        """Zero for positions the step may attend to, -inf past the live KV.

        Every row the kernel reads gets the same mask, not just the live query:
        an all -inf row softmaxes to NaN, and the tile is 64-aligned.
        """
        a = self.ALIGNED
        bias = torch.full((self.GQA, a), float("-inf"), dtype=torch.bfloat16)
        bias[:, :live] = 0.0
        self.ue.dma_to_accelerator_memory(self.BIAS, bias.reshape(-1))

    def emit_attention(self, li: int, live_aligned: int) -> int:
        """GQA over the codec KV cache, one group at a time.

        unified_attention_core does NOT apply the 1/sqrt(head_dim) softmax
        scale -- its callers pre-scale Q -- so that happens here first.
        """
        import math as _math
        self.ue.eltwise_core_dram(
            M=1, N=self.Q_SIZE, dram_a=self.Q, dram_b=None, dram_out=self.Q,
            mode=UE_MODE.MUL_BROADCAST, scalar=1.0 / _math.sqrt(self.AHD))
        f = 0
        for g in range(self.KV_HEADS):
            q_off = g * self.GQA * self.AHD * 2      # 3 contiguous heads at M=1
            out = self.ue.unified_attention_core(
                batch=self.GQA, aligned_seq_len=live_aligned, head_dim=self.AHD,
                Q_DRAM_ADDR=self.Q + q_off,
                K_DRAM_ADDR=self._kv_plane(self.K_CACHE, li, g),
                V_DRAM_ADDR=self._kv_plane(self.V_CACHE, li, g),
                BIAS_DRAM_ADDR=self.BIAS,
                OUTPUT_DRAM_ADDR=self.ATTN + q_off,
                SCRATCH_DRAM_ADDR=self.SCRATCH,
                IDENTITY_DRAM_ADDR=self.IDENT)
            f += out if isinstance(out, (int, float)) else 0
        return f

    # -- rotary -----------------------------------------------------------
    ROPE_THETA = 1_000_000.0

    def build_rope_table(self) -> None:
        """Host-generated cos/sin for the Talker's own theta.

        Layout per position is [cos(half) | cos(half) | -sin(half) | sin(half)],
        which is what rope_hf_core_decode expects: the duplicated cos covers
        both halves and the signed sin carries the rotation's sign, so the
        kernel needs no branch.
        """
        half = self.AHD // 2
        inv = 1.0 / (self.ROPE_THETA ** (torch.arange(half, dtype=torch.float32) / half))
        pos = torch.arange(self.max_ctx, dtype=torch.float32)
        f = torch.outer(pos, inv)
        cos, sin = f.cos().to(torch.bfloat16), f.sin().to(torch.bfloat16)
        table = torch.cat([cos, cos, -sin, sin], dim=1)          # [max_ctx, 2*AHD]
        self.ROPE = self._alloc(self.max_ctx * 2 * self.AHD, "talker.rope")
        self.ue.dma_to_accelerator_memory(self.ROPE, table.reshape(-1))
        self._rope_row_bytes = 2 * self.AHD * 2

    def emit_rope_k(self, pos_reg: int, tmp_reg: int) -> int:
        """Rotate the four key heads in the staging buffer, before caching."""
        f = 0
        for h in range(self.KV_HEADS):
            f += self.ue.rope_hf_core_decode(
                N=self.AHD, input_dram_addr=self.K + h * self.AHD * 2,
                output_dram_addr=self.K + h * self.AHD * 2,
                cos_dram_addr=self.ROPE, sin_dram_addr=self.ROPE + self.AHD * 2,
                rope_size_reg=pos_reg, tmp_reg=tmp_reg) or 0
        return f

    def emit_rope_q(self, pos_reg: int, tmp_reg: int) -> int:
        """Rotate every query head at the current position."""
        f = 0
        for h in range(self.QH):
            f += self.ue.rope_hf_core_decode(
                N=self.AHD, input_dram_addr=self.Q + h * self.AHD * 2,
                output_dram_addr=self.Q + h * self.AHD * 2,
                cos_dram_addr=self.ROPE, sin_dram_addr=self.ROPE + self.AHD * 2,
                rope_size_reg=pos_reg, tmp_reg=tmp_reg) or 0
        return f

    KV_SRAM = 0x10000

    def emit_kv_to_cache(self, li: int, kv_off_reg: int, addr_reg: int) -> int:
        """Copy this step's K and V heads into their cache planes, on device.

        ONE MATMUL CANNOT SCATTER: k_proj writes [1, 512] contiguously while the
        four KV heads live in four separate planes. Rather than route that
        through the host -- 48 round trips per codec token at 24 layers -- each
        head takes an SRAM round to its plane, with the position carried in a
        register. The same pattern the per-layer injection loop uses.

        K is copied AFTER RoPE, so the cache holds rotated keys and attention
        reads them directly.
        """
        for src, cache in ((self.K, self.K_CACHE), (self.V, self.V_CACHE)):
            for h in range(self.KV_HEADS):
                self.ue.accelerator_memory_to_sram(
                    src + h * self.AHD * 2, self.KV_SRAM, self.AHD)
                # add_imm is (SRC, IMM, DST): addr = kv_off + plane_base.
                self.ue.generate_instruction_add_imm(
                    kv_off_reg,
                    user_dma_core.ue_35bit_addr_shifter(self._kv_plane(cache, li, h)),
                    addr_reg)
                self.ue.sram_to_accelerator_memory(
                    self.KV_SRAM, 0, self.AHD, general_reg_src=addr_reg)
        return 0

    # -- a whole layer ----------------------------------------------------
    def emit_layer(self, li: int, src: int, dst: int, *, live_aligned: int,
                   pos_reg: int, kv_off_reg: int, addr_reg: int,
                   tmp_reg: int) -> int:
        """One Talker decoder layer at M=1, as a single uninterrupted program.

        K and V land in their cache planes directly, so nothing leaves the
        device between the projections and the attention that consumes them.
        RoPE is applied to Q here and to K inside the cache, at the row this
        step just wrote.
        """
        pre = f"model.layers.{li}."
        f = self._rms(M=1, N=self.H, src=src, dst=self.NORM,
                      gamma=pre + "input_layernorm.weight")
        f += self._proj(M=1, K=self.H, N=self.Q_SIZE, src=self.NORM,
                        wname=pre + "self_attn.q_proj.weight", dst=self.Q)
        f += self._proj(M=1, K=self.H, N=self.KV_SIZE, src=self.NORM,
                        wname=pre + "self_attn.k_proj.weight", dst=self.K)
        f += self._proj(M=1, K=self.H, N=self.KV_SIZE, src=self.NORM,
                        wname=pre + "self_attn.v_proj.weight", dst=self.V)
        f += self.emit_rope_q(pos_reg, tmp_reg)
        f += self.emit_rope_k(pos_reg, tmp_reg)
        f += self.emit_kv_to_cache(li, kv_off_reg, addr_reg)
        f += self.emit_attention(li, live_aligned)
        f += self._proj(M=1, K=self.Q_SIZE, N=self.H, src=self.ATTN,
                        wname=pre + "self_attn.o_proj.weight", dst=self.PROJ,
                        bias=False)
        self.ue.eltwise_core_dram(M=1, N=self.H, dram_a=src, dram_b=self.PROJ,
                                  dram_out=self.RESID, mode=UE_MODE.ELTWISE_ADD)
        f += self.emit_mlp(li, self.RESID, dst)
        return f

    def emit_step(self, *, layers: int, live_aligned: int, pos_reg: int,
                  kv_off_reg: int, addr_reg: int, tmp_reg: int) -> int:
        """The whole Talker for one codec step: projection, layers, head."""
        f = self.emit_input_projection()
        src, dst = self.IO_A, self.IO_B
        for li in range(layers):
            f += self.emit_layer(li, src, dst, live_aligned=live_aligned,
                                 pos_reg=pos_reg, kv_off_reg=kv_off_reg,
                                 addr_reg=addr_reg, tmp_reg=tmp_reg)
            src, dst = dst, src
        f += self.emit_head(src)
        return f
