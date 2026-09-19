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
