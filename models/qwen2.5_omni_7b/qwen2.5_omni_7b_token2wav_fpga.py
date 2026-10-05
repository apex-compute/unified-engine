#!/usr/bin/env python3
"""Qwen2.5-Omni Token2Wav DiT on the accelerator (bf16 weights).

What runs on the board and what does not, and why:

ON THE BOARD, every learned matmul of the 36 DiT evaluations: the input
projection, all 22 transformer blocks (LayerNorm + AdaLN modulation, Q/K/V,
block-local attention with RoPE on head 0, output projection, GELU MLP, gated
residuals) and the final modulated projection to 80 mel bins.

ON THE HOST, only things that do not depend on the evolving mel state:
  * the ECAPA speaker encoder output (a function of the speaker's reference mel
    only) and the codec embedding lookup, folded with the static columns of the
    input projection into one [2T, 1024] "static" tensor per utterance;
  * the AdaLN shift/scale/gate vectors, which depend only on the diffusion time
    value (36 known scalars) -- a handful of 1024x6144 matvecs per evaluation;
  * the Runge-Kutta combine and classifier-free-guidance blend on the [T, 80]
    state (a few KB per evaluation).

Layout: dense ops run on a [2*Tp, 1024] token-major buffer (the guided and the
unguided branch stacked, Tp = T rounded up to the 24-frame block size times the
attention query chunk). Attention is block-diagonal (24-frame blocks; layer 0
and 20 also see the previous block, layer 10 the next one), so each query chunk
only reads a short key window.
"""

from __future__ import annotations

import math
import time

import torch

from user_dma_core import DMA_DEVICE_H2C, UE_MODE

BLOCK = 24
HEADS = 16
HD = 64
HID = 1024
FF = 2048
MEL = 80
MEL_PAD = 128
LAYERS = 22
GATE_TILE_ROWS = 8               # rows per SRAM tile of the gated-residual chain


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a



class DiTFpga:
    """DiT forward over the engines of a ``Cores`` runner (bf16 weights).

    Every layer is two regions. Region A is row-sharded: each engine takes a
    contiguous slice of the [2*Tp] stacked rows and does the previous layer's
    output projection / gated residual / LayerNorm / MLP and this layer's
    LayerNorm and Q/K/V. Region B is plane-sharded: engine e takes the
    (head, branch) planes p with p % n == e and does the head split, RoPE on
    head 0, block-local attention and the merge back to token-major rows.
    """

    def __init__(self, cores, dit, *, frames: int, layers: int = LAYERS,
                 query_chunk_blocks: int = 4, verbose: bool = True):
        self.cores = cores
        self.n = cores.n
        self.dit = dit
        self.T = frames
        self.layers = layers
        self.verbose = verbose
        self.QC = BLOCK * query_chunk_blocks
        self.Tp = _align(frames, self.QC)
        self.rows = 2 * self.Tp
        if self.rows % (self.n * GATE_TILE_ROWS):
            raise ValueError(f"{self.rows} rows do not split into {self.n} engines")
        self.rs = self.rows // self.n
        self.look = [(1 if i in dit.config.look_backward_layers else 0,
                      1 if i in dit.config.look_ahead_layers else 0)
                     for i in range(LAYERS)]
        self.w: dict[str, int] = {}
        self._zero_tiles: dict[int, int] = {}
        self._mregs: dict[int, int] = {}
        self._next_reg = 40
        self._stage_weights()
        self._alloc_tensors()
        self._stage_constants()

    # ------------------------------------------------------------------ memory
    def _put(self, name: str, t: torch.Tensor) -> int:
        addr = self.cores.put_weight(t)
        self.w[name] = addr
        return addr

    def _tensor(self, n_elem: int) -> int:
        return self.cores.alloc(n_elem * 2)

    def _write(self, addr: int, t: torch.Tensor) -> None:
        self.cores.write(addr, t)

    def _read(self, addr: int, n_elem: int) -> torch.Tensor:
        return self.cores.read(addr, n_elem)

    def _stage_weights(self) -> None:
        t0 = time.perf_counter()
        dit = self.dit
        proj = dit.input_embed.proj
        wm = torch.zeros(HID, MEL_PAD)
        wm[:, :MEL] = proj.weight[:, :MEL]
        self._put("in.mel", wm)
        self.static_weight = proj.weight[:, MEL:].detach().float()
        self.static_bias = proj.bias.detach().float()
        for i in range(self.layers):
            b = dit.transformer_blocks[i]
            a = b.attn
            self._put(f"{i}.q", a.to_q.weight); self._put(f"{i}.qb", a.to_q.bias)
            self._put(f"{i}.k", a.to_k.weight); self._put(f"{i}.kb", a.to_k.bias)
            self._put(f"{i}.v", a.to_v.weight); self._put(f"{i}.vb", a.to_v.bias)
            self._put(f"{i}.o", a.to_out[0].weight); self._put(f"{i}.ob", a.to_out[0].bias)
            self._put(f"{i}.f1", b.ff.ff[0].weight); self._put(f"{i}.f1b", b.ff.ff[0].bias)
            self._put(f"{i}.f2", b.ff.ff[3].weight); self._put(f"{i}.f2b", b.ff.ff[3].bias)
        wp = torch.zeros(MEL_PAD, HID)
        wp[:MEL] = dit.proj_out.weight
        bp = torch.zeros(MEL_PAD)
        bp[:MEL] = dit.proj_out.bias
        self._put("out.w", wp); self._put("out.b", bp)
        if self.verbose:
            print(f"  [T2W] staged DiT weights: "
                  f"{self.cores.weights.get_params_dram_usage() / 2**20:.1f} MiB bf16 "
                  f"in {time.perf_counter() - t0:.1f}s", flush=True)

    def _alloc_tensors(self) -> None:
        R, Tp = self.rows, self.Tp
        c = self.cores
        self.pool_mark = c.pool_mark()
        self.X = self._tensor(R * HID)
        self.XN = self._tensor(R * HID)
        self.Q = self._tensor(R * HID)
        self.K = self._tensor(R * HID)
        self.V = self._tensor(R * HID)
        self.QH = self._tensor(R * HID)                    # [h][b][Tp][64]
        self.KH = self._tensor(R * HID + 512 * HD)
        self.VH = self._tensor(R * HID + 512 * HD)
        self.AH = self._tensor(R * HID)
        self.ATT = self._tensor(R * HID)
        self.Y = self._tensor(R * HID)
        self.H1 = self._tensor(R * FF)
        self.MELIN = self._tensor(R * MEL_PAD)
        self.STATIC = self._tensor(R * HID)
        self.OUT = self._tensor(R * MEL_PAD)
        self.MOD = self._tensor((LAYERS * 6 + 2) * HID)
        self.GATES = self._tensor(LAYERS * 2 * GATE_TILE_ROWS * HID)
        aligned = _align(self.QC + BLOCK * 2, 64)
        self.ALIGNED = aligned
        n = (HD + aligned) * aligned + self.QC * HD
        self._scratch_bytes = n * 2
        self.priv = []
        for ue in c.engines:                              # per-engine private scratch
            self.priv.append({
                "scratch": ue.allocate_tensor_dram(n * 2),
                "rot": ue.allocate_tensor_dram(Tp * HD * 2),
                "a": ue.allocate_tensor_dram(Tp * HD * 2),
                "b": ue.allocate_tensor_dram(Tp * HD * 2),
            })
            ue.dma_write(DMA_DEVICE_H2C, self.priv[-1]["scratch"],
                         bytes(self._scratch_bytes), self._scratch_bytes)
        for base in (self.KH, self.VH):
            nb = (R * HID + 512 * HD) * 2
            c.zero(base, nb)

    def _stage_constants(self) -> None:
        Tp = self.Tp
        self.IDENT = self._put("ident", torch.eye(64))
        pos = torch.arange(Tp, dtype=torch.float32)
        inv = 1.0 / (10000.0 ** (torch.arange(0, HD, 2, dtype=torch.float32) / HD))
        fr = torch.outer(pos, inv)
        emb = torch.cat([fr, fr], dim=-1)
        self.ROPE_COS = self._put("rope.cos", emb.cos())
        self.ROPE_SIN = self._put("rope.sin", emb.sin())
        rot = torch.zeros(HD, HD)
        # The DiT uses the interleaved rotate_half_codec: (x0, x1) -> (-x1, x0).
        for i in range(HD // 2):
            rot[2 * i, 2 * i + 1] = -1.0
            rot[2 * i + 1, 2 * i] = 1.0
        self.ROT = self._put("rope.rot", rot)
        self.bias_addr = {}
        for lb, la in {(l, a) for l, a in self.look[: self.layers]}:
            for c in range(self.Tp // self.QC):
                self.bias_addr[(lb, la, c)] = self._put(
                    f"bias.{lb}{la}.{c}", self._attn_bias(c * self.QC, lb, la))

    # --------------------------------------------------------------- attention
    def _window(self, q0: int, lb: int) -> int:
        return max(q0 - BLOCK * lb, 0)

    def _attn_bias(self, q0: int, lb: int, la: int) -> torch.Tensor:
        s0 = self._window(q0, lb)
        bias = torch.full((self.QC, self.ALIGNED), float("-inf"))
        for r in range(self.QC):
            q = q0 + r
            qb = q // BLOCK
            for j in range(self.ALIGNED):
                f = s0 + j
                if f < self.T and -lb <= f // BLOCK - qb <= la:
                    bias[r, j] = 0.0
            if not torch.isfinite(bias[r]).any():
                bias[r, 0] = 0.0
        return bias

    # ---------------------------------------------------------------- emission
    def _begin(self, ue) -> None:
        self._mregs = {}
        self._next_reg = 40

    def _mreg(self, ue, M: int) -> int:
        r = self._mregs.get(M)
        if r is None:
            r = self._next_reg
            self._next_reg += 1
            ue.generate_instruction_add_set(r, M)
            self._mregs[M] = r
        return r

    def _mm(self, ue, M, K, N, a, w, out, bias=None, gelu=False):
        return ue.matmat_mul_core(
            M=M, K=K, N=N, A_DRAM_ADDR=a, B_DRAM_ADDR=w, OUTPUT_DRAM_ADDR=out,
            C_DRAM_ADDR=bias, bias_mode="broadcast_N" if bias is not None else None,
            is_B_quantized=False, gelu_enable=gelu, gpr_M_reg=self._mreg(ue, M)) or 0

    def _ln(self, ue, rows, src, dst, gamma, beta):
        # Statically unrolled LayerNorm: the dynamic (hardware-loop) path hung the board
        # at N=1024 in testing, so only the static path is used.
        ue.layer_norm_core_dram(M=rows, N=HID, A_DRAM_ADDR=src, OUTPUT_DRAM_ADDR=dst,
                                GAMMA_DRAM_ADDR=gamma, BETA_DRAM_ADDR=beta)

    def _gate_tile(self, li: int, which: int) -> int:
        return self.GATES + ((li * 2 + which) * GATE_TILE_ROWS * HID) * 2

    def _zero_tile(self, n: int) -> int:
        if n not in self._zero_tiles:
            self._zero_tiles[n] = self._put(f"zero{n}", torch.zeros(n))
        return self._zero_tiles[n]

    def _gated_residual(self, ue, x: int, y: int, gate_tile: int, rows: int) -> None:
        """x[rows, HID] += gate * y[rows, HID] as one hardware tile loop."""
        kk = _kokoro()
        ch = kk.SramChain(ue, rows, HID, tile_rows=GATE_TILE_ROWS)
        ch.inputs["x"] = x
        ch.inputs["y"] = y
        ch.consts["g"] = gate_tile
        ch.consts["__zero__"] = self._zero_tile(GATE_TILE_ROWS * HID)
        ch.outputs["o"] = x
        ch.mul("t", "y", "g")
        ch.add("o", "x", "t")
        ch.emit()

    def _mod(self, li: int, k: int) -> int:
        return self.MOD + (li * 6 + k) * HID * 2

    def _rope(self, ue, e: int, plane: int) -> None:
        """RoPE on one [Tp, 64] plane in place: x*cos + (x @ P^T)*sin."""
        Tp, pv = self.Tp, self.priv[e]
        self._mm(ue, Tp, HD, HD, plane, self.ROT, pv["rot"])
        ue.eltwise_core_dram(M=Tp, N=HD, dram_a=plane, dram_b=self.ROPE_COS,
                             dram_out=pv["a"], mode=UE_MODE.ELTWISE_MUL)
        ue.eltwise_core_dram(M=Tp, N=HD, dram_a=pv["rot"], dram_b=self.ROPE_SIN,
                             dram_out=pv["b"], mode=UE_MODE.ELTWISE_MUL)
        ue.eltwise_core_dram(M=Tp, N=HD, dram_a=pv["a"], dram_b=pv["b"],
                             dram_out=plane, mode=UE_MODE.ELTWISE_ADD)

    def _split_plane(self, ue, src: int, dst: int, h: int, b: int) -> None:
        """Token-major [2*Tp, HID] head h of branch b -> contiguous plane [Tp, 64]."""
        Tp = self.Tp
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=src + (b * Tp * HID + h * HD) * 2,
            sram_address=0x00000, element_size=Tp * HD,
            stride_bytes_per_chunk=HD * 2, stride_jump_bytes=HID * 2)
        ue.sram_to_accelerator_memory(0x00000, dst + ((h * 2 + b) * Tp * HD) * 2, Tp * HD)

    def _merge_plane(self, ue, src: int, dst: int, h: int, b: int) -> None:
        Tp = self.Tp
        ue.accelerator_memory_to_sram(src + ((h * 2 + b) * Tp * HD) * 2, 0x00000, Tp * HD)
        ue.sram_to_accelerator_memory(
            0x00000, dst + (b * Tp * HID + h * HD) * 2, Tp * HD,
            stride_bytes_per_chunk=HD * 2, stride_jump_bytes=HID * 2)

    def _attention_plane(self, ue, e: int, li: int, h: int, b: int) -> None:
        Tp, QC = self.Tp, self.QC
        lb, la = self.look[li]
        plane = ((h * 2 + b) * Tp) * HD * 2
        for c in range(Tp // QC):
            q0 = c * QC
            s0 = self._window(q0, lb)
            ue.unified_attention_core(
                batch=QC, aligned_seq_len=self.ALIGNED, head_dim=HD,
                Q_DRAM_ADDR=self.QH + plane + q0 * HD * 2,
                K_DRAM_ADDR=self.KH + plane + s0 * HD * 2,
                V_DRAM_ADDR=self.VH + plane + s0 * HD * 2,
                BIAS_DRAM_ADDR=self.bias_addr[(lb, la, c)],
                OUTPUT_DRAM_ADDR=self.AH + plane + q0 * HD * 2,
                SCRATCH_DRAM_ADDR=self.priv[e]["scratch"],
                IDENTITY_DRAM_ADDR=self.IDENT)

    def _pre(self, ue, e: int, li: int) -> None:
        """LayerNorm1 + Q/K/V for this engine's rows of layer li."""
        w, rs, r0 = self.w, self.rs, e * self.rs
        row = lambda base, width: base + r0 * width * 2
        self._ln(ue, rs, row(self.X, HID), row(self.XN, HID),
                 self._mod(li, 1), self._mod(li, 0))
        for name, dst in (("q", self.Q), ("k", self.K), ("v", self.V)):
            self._mm(ue, rs, HID, HID, row(self.XN, HID), w[f"{li}.{name}"],
                     row(dst, HID), w[f"{li}.{name}b"])

    def _tail(self, ue, e: int, li: int) -> None:
        """Output projection, gated residual, LayerNorm2, MLP, gated residual."""
        w, rs, r0 = self.w, self.rs, e * self.rs
        row = lambda base, width: base + r0 * width * 2
        self._mm(ue, rs, HID, HID, row(self.ATT, HID), w[f"{li}.o"], row(self.Y, HID),
                 w[f"{li}.ob"])
        self._gated_residual(ue, row(self.X, HID), row(self.Y, HID), self._gate_tile(li, 0), rs)
        self._ln(ue, rs, row(self.X, HID), row(self.XN, HID),
                 self._mod(li, 4), self._mod(li, 3))
        self._mm(ue, rs, HID, FF, row(self.XN, HID), w[f"{li}.f1"], row(self.H1, FF),
                 w[f"{li}.f1b"], gelu=True)
        self._mm(ue, rs, FF, HID, row(self.H1, FF), w[f"{li}.f2"], row(self.Y, HID),
                 w[f"{li}.f2b"])
        self._gated_residual(ue, row(self.X, HID), row(self.Y, HID), self._gate_tile(li, 1), rs)

    # ------------------------------------------------------------------ regions
    def compile(self) -> list[int]:
        """Capture the regions of one DiT evaluation; returns their ids in order."""
        c = self.cores
        t0 = time.perf_counter()
        self.sequence: list[int] = []
        rs = self.rs

        def embed(e, ue):
            self._begin(ue)
            r0 = e * rs
            self._mm(ue, rs, MEL_PAD, HID, self.MELIN + r0 * MEL_PAD * 2, self.w["in.mel"],
                     self.X + r0 * HID * 2)
            ue.eltwise_core_dram(M=rs, N=HID, dram_a=self.X + r0 * HID * 2,
                                 dram_b=self.STATIC + r0 * HID * 2,
                                 dram_out=self.X + r0 * HID * 2, mode=UE_MODE.ELTWISE_ADD)
            self._pre(ue, e, 0)
        self.sequence.append(c.region("embed+pre0", embed))

        for li in range(self.layers):
            def attn(e, ue, li=li):
                self._begin(ue)
                for p in range(e, HEADS * 2, self.n):
                    h, b = divmod(p, 2)
                    self._split_plane(ue, self.Q, self.QH, h, b)
                    self._split_plane(ue, self.K, self.KH, h, b)
                    self._split_plane(ue, self.V, self.VH, h, b)
                    if h == 0:
                        self._rope(ue, e, self.QH + ((h * 2 + b) * self.Tp) * HD * 2)
                        self._rope(ue, e, self.KH + ((h * 2 + b) * self.Tp) * HD * 2)
                    self._attention_plane(ue, e, li, h, b)
                    self._merge_plane(ue, self.AH, self.ATT, h, b)
            self.sequence.append(c.region(f"attn{li}", attn))

            def tail(e, ue, li=li):
                self._begin(ue)
                self._tail(ue, e, li)
                if li + 1 < self.layers:
                    self._pre(ue, e, li + 1)
                else:
                    r0 = e * rs
                    self._ln(ue, rs, self.X + r0 * HID * 2, self.XN + r0 * HID * 2,
                             self.MOD + (LAYERS * 6 + 1) * HID * 2,
                             self.MOD + (LAYERS * 6) * HID * 2)
                    self._mm(ue, rs, HID, MEL_PAD, self.XN + r0 * HID * 2, self.w["out.w"],
                             self.OUT + r0 * MEL_PAD * 2, self.w["out.b"])
            self.sequence.append(c.region(f"tail{li}", tail))
        if self.verbose:
            print(f"  [T2W] DiT: {len(self.sequence)} regions x {self.n} engines, "
                  f"{c.program_bytes() / 2**20:.1f} MiB of programs, compiled in "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)
        return self.sequence

    # ------------------------------------------------------------- host inputs
    def set_static(self, conditioning: torch.Tensor, ref_mel: torch.Tensor,
                   codes: torch.Tensor) -> None:
        """Per-utterance constants: static input columns for both branches."""
        dit, T, Tp = self.dit, self.T, self.Tp
        with torch.no_grad():
            spk = conditioning.float().reshape(1, -1)
            cond_vec = dit.input_embed.spk_encoder(ref_mel.float()).reshape(1, -1)
            uncond_vec = dit.input_embed.spk_encoder(torch.zeros_like(ref_mel)).reshape(1, -1)
            ce = dit.text_embed(codes, drop_code=False)[0]
            ceu = dit.text_embed(codes, drop_code=True)[0]
            guided = torch.cat([cond_vec.expand(T, -1), ce, spk.expand(T, -1)], dim=-1)
            null = torch.cat([uncond_vec.expand(T, -1), ceu, torch.zeros(T, spk.shape[1])],
                             dim=-1)
            st = torch.zeros(2, Tp, HID)
            for b, v in enumerate((guided, null)):
                st[b, :T] = v @ self.static_weight.T + self.static_bias
                gen = torch.Generator().manual_seed(7 + b)
                st[b, T:] = torch.randn(Tp - T, HID, generator=gen) * 0.1
        self._write(self.STATIC, st.reshape(-1))

    def modulation(self, t: float) -> torch.Tensor:
        dit = self.dit
        with torch.no_grad():
            emb = dit.time_embed(torch.tensor([t], dtype=torch.float32))
            rows = []
            for i in range(self.layers):
                blk = dit.transformer_blocks[i]
                v = blk.attn_norm.linear(torch.nn.functional.silu(emb))[0]
                sh_a, sc_a, g_a, sh_m, sc_m, g_m = v.chunk(6)
                rows += [sh_a, 1 + sc_a, g_a, sh_m, 1 + sc_m, g_m]
            v = dit.norm_out.linear(torch.nn.functional.silu(emb))[0]
            sc, sh = v.chunk(2)
            full = torch.zeros((LAYERS * 6 + 2), HID)
            full[: len(rows)] = torch.stack(rows)
            full[LAYERS * 6] = sh
            full[LAYERS * 6 + 1] = 1 + sc
        return full.reshape(-1)

    def run(self, mel_state: torch.Tensor, t: float) -> torch.Tensor:
        """One DiT evaluation. mel_state [T, 80] -> predictions [2, T, 80]."""
        T, Tp = self.T, self.Tp
        mel = torch.zeros(Tp, MEL_PAD)
        mel[:T, :MEL] = mel_state
        gen = torch.Generator().manual_seed(11)
        mel[T:, :MEL] = torch.randn(Tp - T, MEL, generator=gen) * 0.1
        self._write(self.MELIN, mel.repeat(2, 1).reshape(-1))
        mod = self.modulation(t)
        self._write(self.MOD, mod)
        rows = mod.reshape(-1, HID)
        tiles = torch.stack([rows[li * 6 + g].expand(GATE_TILE_ROWS, HID)
                             for li in range(LAYERS) for g in (2, 5)])
        self._write(self.GATES, tiles.reshape(-1))
        self.cores.run(self.sequence)
        out = self._read(self.OUT, self.rows * MEL_PAD).float().reshape(2, Tp, MEL_PAD)
        return out[:, :T, :MEL]

    # ----------------------------------------------------------------- sampler
    def sample(self, noise: torch.Tensor, *, num_steps: int = 10,
               guidance_scale: float = 0.5, sway: float = -1.0,
               progress: bool = False) -> torch.Tensor:
        """Classifier-free-guided RK4 integration, [T, 80] noise -> [80, T] mel.

        Mirrors Qwen2_5OmniToken2WavDiTModel.sample: the board evaluates the
        guided and unguided branches together; the host only blends them and
        advances the [T, 80] state.
        """
        times = torch.linspace(0.0, 1.0, num_steps)
        times = times + sway * (torch.cos(torch.pi / 2 * times) - 1 + times)
        evals = 0

        def f(t: float, x: torch.Tensor) -> torch.Tensor:
            nonlocal evals
            pred = self.run(x, float(t))
            evals += 1
            if not torch.isfinite(pred).all():
                bad = (~torch.isfinite(pred)).nonzero()
                raise FloatingPointError(
                    f"DiT evaluation {evals} (t={t:.4f}) produced "
                    f"{bad.shape[0]} non-finite values; first at {bad[0].tolist()}, "
                    f"input state max |x| {x.abs().max().item():.2f}")
            if progress:
                print(f"    [T2W] DiT evaluation {evals}/{(num_steps - 1) * 4}", flush=True)
            return pred[0] + (pred[0] - pred[1]) * guidance_scale

        x = noise.clone().float()
        for t0, t1 in zip(times[:-1].tolist(), times[1:].tolist()):
            dt = t1 - t0
            k1 = f(t0, x)
            k2 = f(t0 + dt / 3, x + dt * k1 / 3)
            k3 = f(t0 + dt * 2 / 3, x + dt * (k2 - k1 / 3))
            k4 = f(t1, x + dt * (k1 - k2 + k3))
            x = x + (k1 + 3 * (k2 + k3) + k4) * dt / 8
        return x.t().contiguous()


# =============================================================================
# BigVGAN vocoder
# =============================================================================
# [T, C] layout (rows = time), channels padded to a multiple of 64. Every op is a
# whole-tensor program step; the convolutions are shifted matmuls (one per tap,
# accumulated through the matmul's full-matrix bias), the transposed convolutions
# are polyphase (no zero-insert waste), and the anti-aliased SnakeBeta activation
# is: polyphase up-FIR -> Snake on both phases -> down-FIR over the phases.
# Kokoro's SramChain supplies the sine (range-reduced Taylor), unchanged.
import importlib.util as _ilu
import os as _os
import sys as _sys

HALO = 160                       # rows of margin above and below every buffer


def _kokoro():
    mod = _sys.modules.get("kokoro_fpga")
    if mod is None:
        path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                             "..", "kokoro", "kokoro_fpga.py")
        spec = _ilu.spec_from_file_location("kokoro_fpga", path)
        mod = _ilu.module_from_spec(spec)
        _sys.modules["kokoro_fpga"] = mod
        spec.loader.exec_module(mod)
    return mod


def _cpad(c: int) -> int:
    return _align(c, 64)


class BigVGANFpga:
    """Mel [80, T] -> waveform over the engines of a ``Cores`` runner.

    Every op is sharded by TIME: an engine owns a contiguous slice of rows
    (a multiple of 128, so tile loops never write into a neighbour's rows) and
    reads whatever neighbouring rows its taps need. Reads of other engines'
    rows are safe because a region boundary is a barrier. The anti-aliased
    activation is two regions (up-FIR + Snake, then down-FIR) because the
    down-FIR needs neighbours' Snake output; a convolution is one region.
    """

    SLICE_ALIGN = 128

    def __init__(self, cores, big, *, frames: int, verbose: bool = True):
        self.cores = cores
        self.n = cores.n
        self.big = big
        self.T = frames
        self.verbose = verbose
        self.kk = _kokoro()
        self.cfg = big.config
        self.strides = list(self.cfg.upsample_rates)
        self.lens = [frames]
        for st in self.strides:
            self.lens.append(self.lens[-1] * st)
        act = big.resblocks[0].activations[0]
        self.fup = act.upsample.filter.reshape(-1).float()
        self.fdn = act.downsample.filter.reshape(-1).float()
        self._zero_rows: dict[int, int] = {}
        self._memo: dict = {}
        self._mregs: dict[int, int] = {}
        self._next_reg = 40
        self.sequence: list[int] = []
        self.marks: dict = {}

    # ------------------------------------------------------------------ memory
    def _put_new(self, t: torch.Tensor) -> int:
        return self.cores.put_weight(t)

    def _memoized(self, key, make):
        if key not in self._memo:
            self._memo[key] = make()
        return self._memo[key]

    def _buf(self, rows: int, C: int) -> int:
        """A shared [rows, C] buffer with HALO spare rows either side; returns data row 0."""
        return self.cores.alloc((rows + 2 * HALO) * C * 2) + HALO * C * 2

    def _zero_row(self, C: int) -> int:
        return self._memoized(("zr", C), lambda: self._put_new(torch.zeros(C)))

    def _zero_tile(self, n: int) -> int:
        return self._memoized(("zt", n), lambda: self._put_new(torch.zeros(n)))

    def _slice(self, rows: int, e: int) -> tuple[int, int]:
        q = -(-rows // self.n)
        q = -(-q // self.SLICE_ALIGN) * self.SLICE_ALIGN
        s0 = e * q
        return (s0, min(rows, s0 + q)) if s0 < rows else (rows, rows)

    # --------------------------------------------------------------------- ops
    def _begin(self) -> None:
        self._mregs = {}
        self._next_reg = 40

    def _mreg(self, ue, M: int) -> int:
        r = self._mregs.get(M)
        if r is None:
            r = self._next_reg
            self._next_reg += 1
            ue.generate_instruction_add_set(r, M)
            self._mregs[M] = r
        return r

    def _ew(self, ue, mode, a, b, out, rows, C, scalar=None):
        ue.eltwise_core_dram(M=rows, N=C, dram_a=a, dram_b=b, dram_out=out,
                             mode=mode, scalar=scalar)

    def _mm(self, ue, M, K, N, a, w, out, bias=None, mode="broadcast_N"):
        return ue.matmat_mul_core(
            M=M, K=K, N=N, A_DRAM_ADDR=a, B_DRAM_ADDR=w, OUTPUT_DRAM_ADDR=out,
            C_DRAM_ADDR=bias, bias_mode=mode if bias is not None else None,
            is_B_quantized=False, gpr_M_reg=self._mreg(ue, M)) or 0

    def _row_copy(self, ue, src, dst, C):
        ue.accelerator_memcpy(src, dst, C * 2)

    def _zero_halo(self, ue, buf, rows, C, pad, s0, s1):
        """Zero the halo rows a conv reads past the global ends (only the end engines)."""
        z, rb = self._zero_row(C), C * 2
        if s0 == 0:
            for i in range(1, pad + 1):
                self._row_copy(ue, z, buf - i * rb, C)
        if s1 == rows:
            for i in range(pad):
                self._row_copy(ue, z, buf + (rows + i) * rb, C)

    def _bias(self, b, Cout_p: int) -> int:
        def make():
            t = torch.zeros(Cout_p)
            if b is not None:
                t[: b.shape[0]] = b.detach().float()
            return self._put_new(t)
        return self._memoized(("bias", id(b), Cout_p), make)

    def _tap(self, conv, kk: int, Cin_p: int, Cout_p: int, transposed: bool = False) -> int:
        def make():
            w = conv.weight.detach().float()
            t = torch.zeros(Cout_p, Cin_p)
            if transposed:                       # [Cin, Cout, k] -> [Cout, Cin]
                t[: w.shape[1], : w.shape[0]] = w[:, :, kk].T
            else:                                # [Cout, Cin, k] -> [Cout, Cin]
                t[: w.shape[0], : w.shape[1]] = w[:, :, kk]
            return self._put_new(t)
        return self._memoized(("tap", id(conv), kk, Cin_p, Cout_p), make)

    def conv1d(self, ue, x, rows, s0, s1, conv, Cin_p, Cout_p, out, dil=1):
        """This engine's rows [s0, s1) of a 'same' Conv1d; x has a zero halo at the ends."""
        m = s1 - s0
        if m <= 0:
            return
        k = conv.weight.shape[2]
        pad = dil * (k - 1) // 2
        bias = self._bias(conv.bias, Cout_p)
        acc = [ue.allocate_tensor_dram(m * Cout_p * 2) for _ in range(2)] if k > 1 else []
        ob = Cout_p * 2
        for kk in range(k):
            last = kk == k - 1
            dst = out + s0 * ob if last else acc[kk % 2]
            prev = None if kk == 0 else acc[(kk - 1) % 2]
            self._mm(ue, m, Cin_p, Cout_p, x + (s0 + kk * dil - pad) * Cin_p * 2,
                     self._tap(conv, kk, Cin_p, Cout_p), dst,
                     bias if kk == 0 else prev, "broadcast_N" if kk == 0 else "full_matrix")

    def conv_transpose(self, ue, x, T_in, s0, s1, conv, Cin_p, Cout_p, stride, out):
        """This engine's INPUT rows [s0, s1) of a polyphase ConvTranspose1d."""
        m = s1 - s0
        if m <= 0:
            return
        k = conv.weight.shape[2]
        pad = conv.padding[0]
        bias = self._bias(conv.bias, Cout_p)
        phase = ue.allocate_tensor_dram(m * Cout_p * 2)
        accs = [ue.allocate_tensor_dram(m * Cout_p * 2) for _ in range(2)]
        for r in range(stride):
            taps = [kk for kk in range(k) if (kk - r - pad) % stride == 0]
            for n, kk in enumerate(taps):
                d = (kk - r - pad) // stride
                last = n == len(taps) - 1
                dst = phase if last else accs[n % 2]
                prev = None if n == 0 else accs[(n - 1) % 2]
                self._mm(ue, m, Cin_p, Cout_p, x + (s0 - d) * Cin_p * 2,
                         self._tap(conv, kk, Cin_p, Cout_p, transposed=True), dst,
                         bias if n == 0 else prev, "broadcast_N" if n == 0 else "full_matrix")
            self._interleave(ue, phase, out, s0, m, Cout_p, stride, r)

    def _interleave(self, ue, src, dst, j0, m, C, stride, r):
        """dst rows stride*(j0 + j) + r <- src rows j (strided store)."""
        per = max(1, 262000 // C)
        j = 0
        while j < m:
            c = min(per, m - j)
            ue.accelerator_memory_to_sram(src + j * C * 2, 0x00000, c * C)
            ue.sram_to_accelerator_memory(
                0x00000, dst + ((stride * (j0 + j) + r) * C) * 2, c * C,
                stride_bytes_per_chunk=C * 2, stride_jump_bytes=stride * C * 2)
            j += c

    # ------------------------------------------------------------- activation
    @staticmethod
    def _fused_rows(C_p: int) -> int:
        """Tile rows for the fused FIR+Snake chains: small tiles so ~20 live tiles fit a bank."""
        R = 128
        while R * C_p > 4096 and R > 8:
            R //= 2
        return R

    def _snake_consts(self, snake, C_p):
        R = self._fused_rows(C_p)

        def make():
            a = torch.exp(snake.alpha.detach().float())
            b = torch.exp(snake.beta.detach().float())
            api = torch.zeros(C_p)
            api[: a.numel()] = a / math.pi
            inv2b = torch.zeros(C_p)
            inv2b[: b.numel()] = 1.0 / (2.0 * (b + 1e-9))
            tile = lambda v: self._put_new(v.reshape(1, C_p).expand(R, C_p).contiguous())
            return tile(api), tile(inv2b)
        api, inv2b = self._memoized(("snake", id(snake), C_p), make)
        return R, api, inv2b

    @staticmethod
    def _snake_body(ch, x, out):
        """Kokoro's SnakeBeta recipe (range-reduced degree-9 sine) on named tiles."""
        ch.mul("p", x, "api"); ch.adds("p", "p", 0.25)
        ch.round("n", "p"); ch.sub("p", "p", "n"); ch.adds("p", "p", 1.0)
        ch.step("s", "p", 1.0); ch.sub("p", "p", "s")
        ch.step("s1", "p", 0.25); ch.step("s2", "p", 0.75)
        ch.muls("v", "p", -2.0); ch.adds("v", "v", 0.5); ch.mul("v", "v", "s1"); ch.add("q", "p", "v")
        ch.muls("v", "p", 2.0); ch.adds("v", "v", -1.5); ch.mul("v", "v", "s2"); ch.add("q", "q", "v")
        ch.muls("q", "q", 2.0 * math.pi)
        ch.mul("x2", "q", "q")
        coef = [((-1) ** ((k - 1) // 2)) / math.factorial(k) for k in (9, 7, 5, 3, 1)]
        ch.muls("tm", "x2", coef[0])
        for c in coef[1:]:
            ch.adds("tm", "tm", c)
            if c != coef[-1]:
                ch.mul("tm", "tm", "x2")
        ch.mul("tm", "tm", "q")
        ch.muls("tm", "tm", -1.0); ch.adds("tm", "tm", 1.0); ch.mul("tm", "tm", "inv2a")
        ch.add(out, x, "tm")

    def act_up(self, ue, x, E, O, rows, s0, s1, snake, C_p):
        """Rows [s0, s1): up-FIR phases of x, then SnakeBeta, in one tile loop."""
        m = s1 - s0
        if m <= 0:
            return
        kk, rb = self.kk, C_p * 2
        if s0 == 0:                                      # replicate the global edges
            for i in range(1, 6):
                self._row_copy(ue, x, x - i * rb, C_p)
        if s1 == rows:
            for i in range(5):
                self._row_copy(ue, x + (rows - 1) * rb, x + (rows + i) * rb, C_p)
        f, base = self.fup, s0 * rb
        R, api, inv2b = self._snake_consts(snake, C_p)
        # phase 0 taps f[15-2d], d=2..7 (shift d-5); phase 1 taps f[16-2d], d=3..8.
        # One chain per phase: reusing temp names inside one chain gave wrong odd-phase values.
        for name, ds, off, dst in (("e", range(2, 8), 15, E), ("o", range(3, 9), 16, O)):
            ch = kk.SramChain(ue, m, C_p, tile_rows=R)
            for d in ds:
                ch.inputs[f"x{d - 5}"] = x + base + (d - 5) * rb
            ch.consts["api"], ch.consts["inv2a"] = api, inv2b
            ch.consts["__zero__"] = self._zero_tile(R * C_p)
            for n, d in enumerate(ds):
                src = f"x{d - 5}"
                if n == 0:
                    ch.muls(name, src, 2.0 * f[off - 2 * d])
                else:
                    ch.muls("t", src, 2.0 * f[off - 2 * d])
                    ch.add(name, name, "t")
            self._snake_body(ch, name, name)
            ch.outputs[name] = dst + base
            ch.emit()

    def act_down(self, ue, E, O, out, rows, s0, s1, C_p):
        """Rows [s0, s1) of the down-FIR over the (E, O) phases, in one tile loop."""
        m = s1 - s0
        if m <= 0:
            return
        kk, rb = self.kk, C_p * 2
        # replicate halos in the interleaved index space:
        #   E above = O above = E[0];  E below = O below = O[T-1]
        if s0 == 0:
            for i in range(1, 4):
                self._row_copy(ue, E, E - i * rb, C_p)
                self._row_copy(ue, E, O - i * rb, C_p)
        if s1 == rows:
            for i in range(3):
                self._row_copy(ue, O + (rows - 1) * rb, E + (rows + i) * rb, C_p)
                self._row_copy(ue, O + (rows - 1) * rb, O + (rows + i) * rb, C_p)
        base = s0 * rb
        R = self._fused_rows(C_p)
        ch = kk.SramChain(ue, m, C_p, tile_rows=R)
        ch.consts["__zero__"] = self._zero_tile(R * C_p)
        for k in range(12):
            sh = (k - 5) // 2
            src = (E if (k - 5) % 2 == 0 else O) + base + sh * rb
            ch.inputs[f"y{k}"] = src
            if k == 0:
                ch.muls("acc", f"y{k}", float(self.fdn[k]))
            else:
                ch.muls("t", f"y{k}", float(self.fdn[k]))
                ch.add("acc", "acc", "t")
        ch.outputs["acc"] = out + base
        ch.emit()

    # ----------------------------------------------------------------- regions
    def _region(self, name, emit):
        def wrapped(e, ue):
            self._begin()
            emit(e, ue)
        rid = self.cores.region(name, wrapped)
        self.sequence.append(rid)
        return rid

    def _activation(self, name, x, rows, snake, C_p, out):
        E = self._buf(rows, C_p)
        O = self._buf(rows, C_p)
        self._region(f"{name}.up", lambda e, ue: self.act_up(
            ue, x, E, O, rows, *self._slice(rows, e), snake, C_p))
        self._region(f"{name}.down", lambda e, ue: self.act_down(
            ue, E, O, out, rows, *self._slice(rows, e), C_p))

    def _conv_region(self, name, x, rows, conv, Cin_p, Cout_p, out, dil, pad, post=None):
        def emit(e, ue):
            s0, s1 = self._slice(rows, e)
            if s1 <= s0:
                return
            self._zero_halo(ue, x, rows, Cin_p, pad, s0, s1)
            self.conv1d(ue, x, rows, s0, s1, conv, Cin_p, Cout_p, out, dil)
            if post is not None:
                post(ue, s0, s1)
        self._region(name, emit)

    def amp_block(self, name, x, rows, amp, C_p, out):
        """AMPBlock: three (act, conv, act, conv, residual) rounds; writes out."""
        cur = x
        for i in range(3):
            nxt = out if i == 2 else self._buf(rows, C_p)
            mark = self.cores.pool_mark()
            a1, c1, a2, c2 = (self._buf(rows, C_p) for _ in range(4))
            dil = amp.convs1[i].dilation[0]
            k = amp.convs1[i].kernel_size[0]
            nm = f"{name}.r{i}"
            self._activation(f"{nm}.a1", cur, rows, amp.activations[2 * i].act, C_p, a1)
            self._conv_region(f"{nm}.c1", a1, rows, amp.convs1[i], C_p, C_p, c1, dil,
                              dil * (k - 1) // 2)

            def add(ue, s0, s1, cur=cur, c2=c2, nxt=nxt):
                rb = C_p * 2
                self._ew(ue, UE_MODE.ELTWISE_ADD, cur + s0 * rb, c2 + s0 * rb,
                         nxt + s0 * rb, s1 - s0, C_p)
            self._activation(f"{nm}.a2", c1, rows, amp.activations[2 * i + 1].act, C_p, a2)
            self._conv_region(f"{nm}.c2", a2, rows, amp.convs2[i], C_p, C_p, c2, 1,
                              (k - 1) // 2, post=add)
            self.cores.pool_rewind(mark if i < 2 else mark)
            cur = nxt

    def build_pre(self, mel_addr):
        conv = self.big.conv_pre
        C1 = _cpad(conv.out_channels)
        out = self._buf(self.T, C1)
        self._conv_region("conv_pre", mel_addr, self.T, conv, 128, C1, out, 1, 3)
        return out

    def build_stage(self, i, x_in):
        conv = self.big.ups[i][0]
        stride = self.strides[i]
        T_in, rows = self.lens[i], self.lens[i + 1]
        Cin_p, Cout_p = _cpad(conv.in_channels), _cpad(conv.out_channels)
        S = self._buf(rows, Cout_p)                    # stage output, allocated first
        mark = self.cores.pool_mark()
        zpad = -(-conv.kernel_size[0] // stride) + 1
        xs = self._buf(rows, Cout_p)

        def up(e, ue):
            s0, s1 = self._slice(T_in, e)
            if s1 <= s0:
                return
            self._zero_halo(ue, x_in, T_in, Cin_p, zpad, s0, s1)
            self.conv_transpose(ue, x_in, T_in, s0, s1, conv, Cin_p, Cout_p, stride, xs)
        self._region(f"stage{i}.convT", up)
        outs = [self._buf(rows, Cout_p) for _ in range(3)]
        for j in range(3):
            self.amp_block(f"stage{i}.amp{j}", xs, rows, self.big.resblocks[i * 3 + j],
                           Cout_p, outs[j])

        def combine(e, ue):
            s0, s1 = self._slice(rows, e)
            if s1 <= s0:
                return
            rb = Cout_p * 2
            o = s0 * rb
            self._ew(ue, UE_MODE.ELTWISE_ADD, outs[0] + o, outs[1] + o, S + o, s1 - s0, Cout_p)
            self._ew(ue, UE_MODE.ELTWISE_ADD, S + o, outs[2] + o, S + o, s1 - s0, Cout_p)
            self._ew(ue, UE_MODE.MUL_BROADCAST, S + o, None, S + o, s1 - s0, Cout_p,
                     scalar=1.0 / 3.0)
        self._region(f"stage{i}.combine", combine)
        self.cores.pool_rewind(mark)
        return S

    def build_post(self, x):
        rows = self.lens[-1]
        C_p = _cpad(self.big.conv_post.in_channels)
        out = self._buf(rows, 64)
        mark = self.cores.pool_mark()
        a = self._buf(rows, C_p)
        self._activation("post.act", x, rows, self.big.activation_post.act, C_p, a)
        self._conv_region("post.conv", a, rows, self.big.conv_post, C_p, 64, out, 1, 3)
        self.cores.pool_rewind(mark)
        return out

    def compile(self) -> None:
        t0 = time.perf_counter()
        c = self.cores
        self.mark0 = c.pool_mark()
        self.mel_addr = self._buf(self.T, 128)
        x = self.build_pre(self.mel_addr)
        self.marks["pre"] = (len(self.sequence), x, 0)
        for i in range(len(self.strides)):
            x = self.build_stage(i, x)
            self.marks[i] = (len(self.sequence), x, i + 1)
        self.out_addr = self.build_post(x)
        self.marks["post"] = (len(self.sequence), self.out_addr, None)
        if self.verbose:
            print(f"  [T2W] BigVGAN: {len(self.sequence)} regions x {self.n} engines, "
                  f"{c.program_bytes() / 2**20:.1f} MiB of programs in total, compiled in "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)

    def preprocess(self, mel: torch.Tensor) -> torch.Tensor:
        """The model's mel normalisation (exp -> dB -> clamp), on the [80, T] mel."""
        return self.big.process_mel_spectrogram(mel.unsqueeze(0))[0]

    def load_mel(self, mel: torch.Tensor) -> None:
        x = torch.zeros(self.T, 128)
        x[:, :MEL] = self.preprocess(mel).t()
        self.cores.write(self.mel_addr, x.reshape(-1))

    def run(self, mel: torch.Tensor) -> torch.Tensor:
        """mel [80, T] -> waveform [L]."""
        self.load_mel(mel)
        self.cores.run(self.sequence)
        out = self.cores.read(self.out_addr, self.lens[-1] * 64).float().reshape(-1, 64)
        return out[:, 0].clamp(-1.0, 1.0)


def dit_flops(frames: int, Tp: int, QC: int, aligned: int, look, layers: int = LAYERS,
              evaluations: int = 36) -> tuple[int, int]:
    """(model, issued) matmul-class FLOPs of all DiT evaluations.

    Model = the work the network defines for ``frames`` frames and the two guidance
    branches. Issued = what the engines are asked to do: rows padded up to ``Tp``,
    the mel columns padded to 128, and attention over 128-row key windows.
    """
    def per_eval(rows, mel_k, mel_n, attn_per_layer):
        f = 2 * rows * mel_k * HID + 2 * rows * HID * mel_n
        f += layers * (rows * (4 * 2 * HID * HID + 2 * 2 * HID * FF))
        return f + attn_per_layer
    model_attn = sum(4 * (2 * frames) * HID * BLOCK * (1 + lb + la)
                     for lb, la in look[:layers])
    issued_attn = layers * 32 * (Tp // QC) * 4 * QC * aligned * HD
    model = per_eval(2 * frames, MEL, MEL, model_attn)
    issued = per_eval(2 * Tp, MEL_PAD, MEL_PAD, issued_attn)
    return model * evaluations, issued * evaluations


def bigvgan_flops(big, frames: int) -> tuple[int, int]:
    """(model, issued) matmul-class FLOPs of the BigVGAN (convolution taps as matmuls)."""
    cfg = big.config
    lens = [frames]
    for st in cfg.upsample_rates:
        lens.append(lens[-1] * st)

    def total(pad):
        c = (lambda n: _cpad(n)) if pad else (lambda n: n)
        f = 2 * frames * (128 if pad else MEL) * c(big.conv_pre.out_channels) * 7
        for i, st in enumerate(cfg.upsample_rates):
            up = big.ups[i][0]
            cin, cout, k = c(up.in_channels), c(up.out_channels), up.kernel_size[0]
            f += 2 * lens[i] * k * cin * cout
            for j in range(3):
                amp = big.resblocks[i * 3 + j]
                kk = amp.convs1[0].kernel_size[0]
                f += 6 * 2 * lens[i + 1] * cout * cout * kk
        post_in = c(big.conv_post.in_channels)
        f += 2 * lens[-1] * post_in * (64 if pad else 1) * 7
        return f
    return total(False), total(True)


class Token2WavFpga:
    """Codec IDs -> waveform with the DiT and the BigVGAN both on the accelerator."""

    def __init__(self, cores, t2w, *, codes: int, verbose: bool = True):
        self.cores = cores
        self.t2w = t2w
        self.frames = codes * t2w.code2wav_dit_model.repeats
        self.verbose = verbose
        self.dit = DiTFpga(cores, t2w.code2wav_dit_model, frames=self.frames, verbose=verbose)
        self.dit.compile()
        # BigVGAN reuses the activation pool the DiT used once the DiT is done with it.
        self.vocoder = BigVGANFpga(cores, t2w.code2wav_bigvgan_model, frames=self.frames,
                                   verbose=verbose)
        cores.pool_rewind(self.dit.pool_mark)
        self.vocoder.compile()
        cores.pool_rewind(self.dit.pool_mark)

    def synthesize(self, codes: torch.Tensor, conditioning: torch.Tensor,
                   reference_mel: torch.Tensor, noise: torch.Tensor | None = None,
                   *, guidance_scale: float = 0.5, num_steps: int = 10) -> tuple[torch.Tensor, dict]:
        t0 = time.perf_counter()
        dev0 = self.cores.device_s
        self.dit.set_static(conditioning, reference_mel, codes)
        if noise is None:
            noise = torch.randn(self.frames, MEL)
        mel = self.dit.sample(noise, num_steps=num_steps, guidance_scale=guidance_scale,
                              progress=self.verbose)
        t1 = time.perf_counter()
        dev1 = self.cores.device_s
        wav = self.vocoder.run(mel)
        t2 = time.perf_counter()
        dev2 = self.cores.device_s
        dm, di = dit_flops(self.frames, self.dit.Tp, self.dit.QC, self.dit.ALIGNED,
                           self.dit.look, evaluations=(num_steps - 1) * 4)
        bm, bi = bigvgan_flops(self.t2w.code2wav_bigvgan_model, self.frames)
        return wav, {"dit_s": t1 - t0, "bigvgan_s": t2 - t1, "mel": mel,
                     "dit_device_s": dev1 - dev0, "bigvgan_device_s": dev2 - dev1,
                     "dit_flops_model": dm, "dit_flops_issued": di,
                     "bigvgan_flops_model": bm, "bigvgan_flops_issued": bi,
                     "evaluations": (num_steps - 1) * 4, "mel_frames": self.frames}
