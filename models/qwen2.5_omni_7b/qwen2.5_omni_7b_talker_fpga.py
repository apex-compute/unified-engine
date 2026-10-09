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

In multi-engine mode, matrix output columns and their weights are partitioned
across private engine regions. Small norms and biases remain shared.
"""

from __future__ import annotations

import json
import os
import time

import torch

import quant_lib
import user_dma_core
from user_dma_core import DMA_DEVICE_H2C, TYPE, UE_MODE
from multi_engine_shard import DENSE_BF16

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

    def __init__(self, ue, model_dir: str, *, scheduler=None, verbose: bool = True):
        self.ue = ue
        self.scheduler = scheduler
        self.model_dir = model_dir
        self.verbose = verbose
        index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
        self._map = {n: s for n, s in index["weight_map"].items()
                     if n.startswith(TALKER_PREFIX)}
        self.addr: dict[str, int] = {}     # tensor name -> DRAM address
        self.scale: dict[str, int] = {}    # tensor name -> scale blob address
        self.shape: dict[str, tuple] = {}
        self.shards: dict[str, object] = {}
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

    def _put_sharded(self, name: str, t: torch.Tensor, *, if4: bool) -> None:
        """Stage only the materialized column blocks, never a full matrix."""
        if t.ndim != 2:
            raise ValueError(f"{name}: expected a matrix, got {tuple(t.shape)}")
        n, k = map(int, t.shape)
        if if4:
            data, scales = quant_lib.quantize("if4", t.float(), block_size=BLOCK)
            dtype = TYPE.IF4
        else:
            data = t.to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()
            scales = None
            dtype = DENSE_BF16
        self.shards[name] = self.scheduler.shard_quantized_weight_from_bytes(
            name=f"talker.{name}", weight_bytes=data, scale_bytes=scales,
            K=k, N=n, layers=1, layer_stride_bytes=0, data_type=dtype,
            verbose=False)
        self.shape[name] = (n, k)
        self._staged_bytes += len(data) + (len(scales) if scales is not None else 0)

    def _selected(self, layers: int):
        """The tensors that are staged on the device, in staging order."""
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
            yield name, short

    def params_items(self, layers: int = 24):
        """Yield ``(key, dtype, shape, bytes)`` for params.bin's talker region.

        Quantization happens here, once, when the region is built; staging then
        only slices these bytes into the engines' shards.
        """
        for name, short in self._selected(layers):
            t = self._tensor(name)
            if t.ndim == 2 and short.endswith(_IF4_SUFFIXES):
                data, scales = quant_lib.quantize("if4", t.float(), block_size=BLOCK)
                yield f"{short}.data", "if4_data", tuple(t.shape), bytes(data)
                yield f"{short}.scale", "if4_scale", tuple(t.shape), bytes(scales)
            else:
                raw = t.to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()
                yield short, "bf16", tuple(t.shape), raw
            del t

    def stage(self, layers: int = 24, region: dict | None = None) -> "TalkerWeights":
        """Place the Talker in accelerator DRAM.

        With ``region`` (params.bin's talker region) the bytes come from the file;
        without it they are converted from the checkpoint, as before.
        """
        t0 = time.perf_counter()
        handle = open(region["bin_path"], "rb") if region is not None else None
        try:
            def blob(key: str) -> bytes:
                section = region["sections"][key]
                handle.seek(region["base_offset"] + int(section["offset"]))
                data = handle.read(int(section["size"]))
                if len(data) != int(section["size"]):
                    raise IOError(f"params.bin truncated in talker section {key}")
                return data

            for name, short in self._selected(layers):
                if region is not None:
                    sections = region["sections"]
                    if f"{short}.data" in sections:                # IF4 matrix
                        n, k = (int(x) for x in sections[f"{short}.data"]["shape"])
                        self._shard(short, n, k, blob(f"{short}.data"),
                                    blob(f"{short}.scale"), TYPE.IF4)
                    else:
                        shape = tuple(int(x) for x in sections[short]["shape"])
                        raw = blob(short)
                        if len(shape) == 2 and self.scheduler is not None:
                            self._shard(short, shape[0], shape[1], raw, None, DENSE_BF16)
                        else:
                            self._put_bf16_bytes(short, shape, raw)
                    continue
                t = self._tensor(name)
                if self.scheduler is not None and t.ndim == 2:
                    self._put_sharded(short, t, if4=short.endswith(_IF4_SUFFIXES))
                elif short.endswith(_IF4_SUFFIXES):
                    self._put_if4(short, t)
                else:
                    self._put_bf16(short, t)
                del t
        finally:
            if handle is not None:
                handle.close()
        if self.verbose:
            print(f"  [Talker] staged {len(self.addr) + len(self.shards)} tensors, "
                  f"{self._staged_bytes / 2**20:.1f} MiB (IF4 matrices, bf16 norms "
                  f"and biases) in {time.perf_counter() - t0:.1f}s"
                  + ("" if region is None else " from params.bin")
                  + (f"; private usage: " + ", ".join(
                      f"{x / 2**20:.1f} MiB" for x in self.scheduler.private_usage())
                     if self.scheduler is not None else ""))
        return self

    def _shard(self, short: str, n: int, k: int, data: bytes, scales, dtype) -> None:
        self.shards[short] = self.scheduler.shard_quantized_weight_from_bytes(
            name=f"talker.{short}", weight_bytes=data, scale_bytes=scales,
            K=k, N=n, layers=1, layer_stride_bytes=0, data_type=dtype, verbose=False)
        self.shape[short] = (n, k)
        self._staged_bytes += len(data) + (len(scales) if scales is not None else 0)

    def _put_bf16_bytes(self, short: str, shape: tuple, blob: bytes) -> None:
        a = self.ue.allocate_params_dram(len(blob), label=f"talker.{short}")
        if self.ue.dma_write(DMA_DEVICE_H2C, a, blob, len(blob)) != len(blob):
            raise IOError(f"talker {short}: short DMA")
        self.addr[short] = a
        self.shape[short] = tuple(shape)
        self._staged_bytes += len(blob)


class TalkerRunner:
    """Emit Talker steps on one engine or private, column-sharded engines."""

    H = 896
    QH, KVH, AHD = 12, 4, 128
    Q_SIZE, KV_SIZE = 1536, 512
    MLP = 18944
    VOCAB = 8448
    THINKER_H = 3584

    def __init__(self, ue, weights: TalkerWeights, *, max_ctx: int = 1024,
                 scheduler=None):
        self.ue = ue
        self.w = weights
        self.scheduler = scheduler or weights.scheduler
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

    def _emit_shard(self, engine_idx: int, wname: str, src: int, dst: int,
                    *, bias: bool = False, silu: bool = False) -> int:
        sw = self.w.shards[wname]
        sh = sw.shard_or_none(engine_idx)
        if sh is None:
            return 0
        off = sh.col_offset * 2
        assert off % 128 == 0
        kw = dict(M=1, K=sw.K, N=sh.cols, A_DRAM_ADDR=src,
                  B_DRAM_ADDR=sh.weight_addr, OUTPUT_DRAM_ADDR=dst + off)
        bias_addr = self.w.addr.get(wname.replace(".weight", ".bias")) if bias else None
        if bias_addr is not None:
            kw.update(C_DRAM_ADDR=bias_addr + off, bias_mode="broadcast_N")
        if silu:
            kw["silu_enable"] = True
        engine = self.scheduler.engines[engine_idx]
        if sw.data_type is DENSE_BF16:
            return engine.matmat_mul_core(**kw) or 0
        kw.update(SCALE_DRAM_ADDR=sh.scale_addr, data_type=TYPE.IF4)
        return engine.quantized_matmat_core(**kw) or 0

    def _sharded_round(self, ops) -> int:
        """Run independent projections together; join before any consumer."""
        sched = self.scheduler
        sched.release()
        flops = sum(self._emit_shard(0, *op[:3], bias=op[3], silu=op[4])
                    for op in ops)
        for e in sched.worker_indices():
            sched.begin_worker_round(e)
            for wname, src, dst, bias, silu in ops:
                # Every engine's own shard counts: the work issued is the sum over engines.
                flops += self._emit_shard(e, wname, src, dst, bias=bias, silu=silu)
            sched.end_worker_round(e)
        sched.join()
        return flops

    def _rms(self, *, M, N, src, dst, gamma):
        return self.ue.rms_norm_core_dram(
            M=M, N=N, A_DRAM_ADDR=src, OUTPUT_DRAM_ADDR=dst,
            GAMMA_DRAM_ADDR=self.w.addr[gamma]) or 0

    def emit_input_projection(self) -> int:
        """[1, 3584] -> [1, 896]: the Thinker-space input into Talker space."""
        if self.scheduler is not None:
            return self._sharded_round([("thinker_to_talker_proj.weight",
                                         self.IN_3584, self.IO_A, True, False)])
        return self.ue.matmat_mul_core(
            M=1, K=self.THINKER_H, N=self.H, A_DRAM_ADDR=self.IN_3584,
            B_DRAM_ADDR=self.w.addr["thinker_to_talker_proj.weight"],
            OUTPUT_DRAM_ADDR=self.IO_A, is_B_quantized=False,
            C_DRAM_ADDR=self.w.addr["thinker_to_talker_proj.bias"],
            bias_mode="broadcast_N") or 0

    def emit_head(self, src: int) -> int:
        """[1, 896] -> [1, 8448] codec logits, through the final norm."""
        f = self._rms(M=1, N=self.H, src=src, dst=self.NORM, gamma="model.norm.weight")
        if self.scheduler is not None:
            return f + self._sharded_round([("codec_head.weight", self.NORM,
                                             self.LOGITS, False, False)])
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
        if self.scheduler is not None:
            f += self._sharded_round([
                (pre + "mlp.gate_proj.weight", self.NORM, self.GATE, False, True),
                (pre + "mlp.up_proj.weight", self.NORM, self.UP, False, False)])
            self.ue.eltwise_core_dram(M=1, N=self.MLP, dram_a=self.GATE,
                                      dram_b=self.UP, dram_out=self.GATE,
                                      mode=UE_MODE.ELTWISE_MUL)
            f += self._sharded_round([(pre + "mlp.down_proj.weight", self.GATE,
                                       self.DOWN, False, False)])
            self.ue.eltwise_core_dram(M=1, N=self.H, dram_a=src,
                                      dram_b=self.DOWN, dram_out=dst,
                                      mode=UE_MODE.ELTWISE_ADD)
            return f
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

    def set_causal_bias(self, live: int, *, aligned: int | None = None) -> None:
        """Zero for positions the step may attend to, -inf past the live KV.

        Every row the kernel reads gets the same mask, not just the live query:
        an all -inf row softmaxes to NaN, and the tile is 64-aligned.
        """
        a = aligned or self.ALIGNED
        if a > self.ALIGNED or a % 64 or not 1 <= live <= a:
            raise ValueError(f"invalid Talker bias shape: live={live}, aligned={a}, "
                             f"capacity={self.ALIGNED}")
        bias = torch.full((self.GQA, a), float("-inf"), dtype=torch.bfloat16)
        bias[:, :live] = 0.0
        self.ue.dma_to_accelerator_memory(self.BIAS, bias.reshape(-1))

    def emit_attention(self, li: int, live_aligned: int,
                       aligned_reg: int | None = None) -> int:
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
                IDENTITY_DRAM_ADDR=self.IDENT,
                gpr_aligned_seq_len_reg=aligned_reg,
                q_pre_scaled=aligned_reg is not None)
            f += out if isinstance(out, (int, float)) else 0
        self._attn_flops_emitted = getattr(self, "_attn_flops_emitted", 0) + f
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
                   tmp_reg: int, aligned_reg: int | None = None) -> int:
        """One Talker decoder layer at M=1, as a single uninterrupted program.

        K and V land in their cache planes directly, so nothing leaves the
        device between the projections and the attention that consumes them.
        RoPE is applied to Q here and to K inside the cache, at the row this
        step just wrote.
        """
        pre = f"model.layers.{li}."
        f = self._rms(M=1, N=self.H, src=src, dst=self.NORM,
                      gamma=pre + "input_layernorm.weight")
        if self.scheduler is not None:
            f += self._sharded_round([
                (pre + "self_attn.q_proj.weight", self.NORM, self.Q, True, False),
                (pre + "self_attn.k_proj.weight", self.NORM, self.K, True, False),
                (pre + "self_attn.v_proj.weight", self.NORM, self.V, True, False)])
            f += self.emit_rope_q(pos_reg, tmp_reg)
            f += self.emit_rope_k(pos_reg, tmp_reg)
            f += self.emit_kv_to_cache(li, kv_off_reg, addr_reg)
            f += self.emit_attention(li, live_aligned, aligned_reg=aligned_reg)
            f += self._sharded_round([(pre + "self_attn.o_proj.weight", self.ATTN,
                                       self.PROJ, False, False)])
            self.ue.eltwise_core_dram(M=1, N=self.H, dram_a=src,
                                      dram_b=self.PROJ, dram_out=self.RESID,
                                      mode=UE_MODE.ELTWISE_ADD)
            return f + self.emit_mlp(li, self.RESID, dst)
        f += self._proj(M=1, K=self.H, N=self.Q_SIZE, src=self.NORM,
                        wname=pre + "self_attn.q_proj.weight", dst=self.Q)
        f += self._proj(M=1, K=self.H, N=self.KV_SIZE, src=self.NORM,
                        wname=pre + "self_attn.k_proj.weight", dst=self.K)
        f += self._proj(M=1, K=self.H, N=self.KV_SIZE, src=self.NORM,
                        wname=pre + "self_attn.v_proj.weight", dst=self.V)
        f += self.emit_rope_q(pos_reg, tmp_reg)
        f += self.emit_rope_k(pos_reg, tmp_reg)
        f += self.emit_kv_to_cache(li, kv_off_reg, addr_reg)
        f += self.emit_attention(li, live_aligned, aligned_reg=aligned_reg)
        f += self._proj(M=1, K=self.Q_SIZE, N=self.H, src=self.ATTN,
                        wname=pre + "self_attn.o_proj.weight", dst=self.PROJ,
                        bias=False)
        self.ue.eltwise_core_dram(M=1, N=self.H, dram_a=src, dram_b=self.PROJ,
                                  dram_out=self.RESID, mode=UE_MODE.ELTWISE_ADD)
        f += self.emit_mlp(li, self.RESID, dst)
        return f

    def emit_step(self, *, layers: int, live_aligned: int, pos_reg: int,
                  kv_off_reg: int, addr_reg: int, tmp_reg: int,
                  aligned_reg: int | None = None) -> int:
        """The whole Talker for one codec step: projection, layers, head."""
        f = self.emit_input_projection()
        src, dst = self.IO_A, self.IO_B
        for li in range(layers):
            f += self.emit_layer(li, src, dst, live_aligned=live_aligned,
                                 pos_reg=pos_reg, kv_off_reg=kv_off_reg,
                                 addr_reg=addr_reg, tmp_reg=tmp_reg,
                                 aligned_reg=aligned_reg)
            src, dst = dst, src
        f += self.emit_head(src)
        return f

    def compile_reusable_step(self, *, layers: int = 24) -> None:
        """Compile one dynamic-length body; runtime preambles set position/length."""
        if self.scheduler is None:
            raise ValueError("reusable Talker requires a multi-engine scheduler")
        ue, sched = self.ue, self.scheduler
        ue.reset_isa_reg_counter()
        ue.reset_inst_ptr_counter()
        self._pos_reg = ue.alloc_isa_reg()
        self._kv_off_reg = ue.alloc_isa_reg()
        self._addr_reg = ue.alloc_isa_reg()
        self._tmp_reg = ue.alloc_isa_reg()
        self._aligned_reg = ue.alloc_isa_reg()
        ue.clear_inst_id()
        ue.clear_capture_buffer()
        self._program_addr = ue.get_program_dram_addr()
        ue.start_capture()
        try:
            sched.begin_program()
            self._attn_flops_emitted = 0
            total = self.emit_step(
                layers=layers, live_aligned=64, pos_reg=self._pos_reg,
                kv_off_reg=self._kv_off_reg, addr_reg=self._addr_reg,
                tmp_reg=self._tmp_reg, aligned_reg=self._aligned_reg)
            # FLOPs the emitted program issues per step, split like the Thinker's: everything
            # except attention is fixed per step, attention scales with the aligned KV length
            # (the template was emitted at live_aligned=64).
            self.step_flops_fixed = int(total - self._attn_flops_emitted)
            self.attn_flops_per_aligned = self._attn_flops_emitted / 64.0
            ue.generate_instruction_halt()
            self._worker_addrs = sched.finalize()
            ue.stop_capture()
        except Exception:
            sched.abort_program()
            if ue.is_capture_on:
                ue.stop_capture()
            ue.clear_capture_buffer()
            raise
        body_bytes = ue.get_capture_instruction_size_bytes()
        self._preamble_addr = self._program_addr + body_bytes
        # The 4-instruction preamble is padded to 128 bytes; neither it nor the
        # body may consume the protected tail above the master's ISA slice.
        limit = getattr(ue, "MASTER_ISA_LIMIT", None)
        if limit is not None and self._preamble_addr + 128 > limit:
            raise MemoryError("Talker program and preamble exceed master ISA")
        written = ue.write_captured_instructions_to_dram(self._program_addr)
        if written != body_bytes:
            raise IOError("Talker body DMA short write")
        ue.allocate_program_dram(body_bytes + 128)
        ue.clear_capture_buffer()
        self._build_launch_table()

    def _build_launch_table(self) -> None:
        """Precompile the per-position launch entry for every position.

        An entry primes the position, KV-offset and aligned-length registers and
        jumps into the body, so a step is started by launching one address; the
        host writes no instructions while the Talker runs.
        """
        ue = self.ue
        entries = []
        for pos in range(self.max_ctx):
            aligned = ((pos + 64) // 64) * 64
            ue.clear_inst_id()
            ue.clear_capture_buffer()
            ue.start_capture()
            ue.generate_instruction_add_set(
                self._pos_reg, user_dma_core.ue_35bit_addr_shifter(
                    pos * self._rope_row_bytes))
            ue.generate_instruction_add_set(
                self._kv_off_reg, user_dma_core.ue_35bit_addr_shifter(pos * self.AHD * 2))
            ue.generate_instruction_add_set(self._aligned_reg, aligned)
            ue.generate_instruction_jump_abs(
                user_dma_core.ue_35bit_addr_shifter(self._program_addr))
            ue.stop_capture()
            entries.append(b"".join(i.get_bytes() for i in ue.capture_buffer))
            ue.clear_capture_buffer()
        stride = (max(len(e) for e in entries) + 63) // 64 * 64
        blob = b"".join(e.ljust(stride, b"\0") for e in entries)
        limit = getattr(ue, "MASTER_ISA_LIMIT", None)
        base = ue.get_program_dram_addr()
        if limit is not None and base + len(blob) > limit:
            raise MemoryError("Talker launch table exceeds master ISA")
        if ue.dma_write(user_dma_core.DMA_DEVICE_H2C, base, blob, len(blob)) != len(blob):
            raise IOError("Talker launch table DMA short write")
        ue.allocate_program_dram(len(blob))
        self._launch_base, self._launch_stride = base, stride
        self._launch_end = base + len(blob)

    def run_step(self, x: torch.Tensor, pos: int, *, timeout_s: float = 30.0):
        """Execute one codec position and return logits plus core-0 HW time."""
        if not hasattr(self, "_program_addr"):
            raise RuntimeError("compile_reusable_step must run before run_step")
        if not 0 <= pos < self.max_ctx:
            raise ValueError(f"Talker position {pos} exceeds max_ctx={self.max_ctx}")
        ue, sched = self.ue, self.scheduler
        aligned = ((pos + 64) // 64) * 64
        self.set_causal_bias(pos + 1, aligned=aligned)
        ue.dma_to_accelerator_memory(
            self.IN_3584, x.reshape(-1).to(torch.bfloat16))
        sched.start_workers(self._worker_addrs)
        ue.start_execute_from_dram(self._launch_base + pos * self._launch_stride)
        ue.wait_queue(timeout_s)
        if ue.is_queue_busy():
            raise TimeoutError(f"Talker position {pos}: master remained busy")
        for idx, worker in enumerate(sched.workers, start=1):
            worker.wait_queue(timeout_s)
            if worker.is_queue_busy():
                raise TimeoutError(f"Talker position {pos}: worker {idx} remained busy")
        hw_us = ue.report_latency_in_us()
        buf = bytearray(self.VOCAB * 2)
        read = ue.dma_read(ue.c2h_device, self.LOGITS, buf, len(buf))
        if read != len(buf):
            raise IOError(f"Talker position {pos}: logits DMA short read")
        logits = torch.frombuffer(buf, dtype=torch.bfloat16).clone()
        return logits, hw_us
