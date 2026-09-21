#!/usr/bin/env python3
"""Qwen2.5-VL-3B language-model method group (36-layer GQA decoder).

``Qwen25VLLMMixin`` carries the LM methods and is mixed into
``Qwen25VL_UnifiedEngine`` in qwen2.5_vl_3b_test.py; it is never
instantiated on its own. Like the vision mixin it imports nothing from the
test module, so the split stays cycle-free.

PHASE B OF THE SHARED PARAMS WINDOW. Vision weights and LM weights occupy the
SAME addresses from 0x8000_0000 -- they never run together. ``lm_weight_init``
rewinds the params cursor and loads over whatever the encoder left there, so
anything the encoder produced must already be on the host by then (the 144
image embeddings are, via ``run_vision_encoder``).

ATTENTION SHAPE. Q/K/V are laid out head-major and attention runs ONE
unified_attention_core call per Q head, reading its group's K/V head straight
out of the cache. The previous build instead packed Q as [seq*group_size,
head_dim] and duplicated K/V per group, which made the causal bias
(seq*8)^2 -- 64x larger, 2 GiB at a 4096 context. Per-head keeps it seq^2.
"""
import json
import math
import os
import sys
import time

_SD = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SD)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SD)))

import numpy as np
import torch

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, INSTRUCTION_SIZE_BYTES, TYPE, UE_MODE, UE_VECTOR_SIZE,
    ue_35bit_addr_shifter)

LM_QUANT_PRECISION = "if4"
def _weight_gen():
    """The sibling weight-bin generator, loaded by path ("2.5" is not an
    identifier). Imported lazily: it pulls in transformers and huggingface_hub,
    which a run with the bin already present has no reason to load."""
    name = "qwen2_5_vl_3b_weights"
    if name in sys.modules:
        return sys.modules[name]
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "qwen2.5_vl_3b_weights.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Qwen25VLLMMixin:
    """LM methods for Qwen25VL_UnifiedEngine (see module docstring)."""

    # ---- weights -----------------------------------------------------------

    def _lm_quantized_projections(self) -> set[str]:
        """Projection tags stored with the configured block-quantized codec.

        Qwen2.5-VL-3B keeps V/O in BF16 for accuracy. Larger compatible
        decoders can opt either projection into IF4 independently; notably,
        Qwen2.5-Omni-7B uses IF4 O during prefill, then switches to a shared
        BF16 O overlay for decode while retaining BF16 V in both phases.
        Keeping the choices phase-aware here makes the emitter reusable without
        changing the released 3B bin format.
        """
        default = ("q", "k", "gate", "up", "down")
        values = self._cfg.get("precision", {}).get(
            "lm_quantized_projections", default)
        valid = {"q", "k", "v", "o", "gate", "up", "down"}
        result = set(values)
        unknown = result - valid
        if unknown:
            raise ValueError(
                f"unknown LM quantized projection tag(s): {sorted(unknown)}")
        return result

    def _lm_projection_is_quantized(self, tag: str) -> bool:
        return tag in self._lm_quantized_projections()

    def _decode_projection_is_quantized(self, tag: str) -> bool:
        """Decode storage policy, overridable independently from prefill."""
        return self._lm_projection_is_quantized(tag)

    def _decode_projection_should_shard(self, tag: str, layer: int) -> bool:
        """Whether decode materializes this projection in private DRAM."""
        return True

    def _decode_projection_uses_static_bf16(self, tag: str) -> bool:
        """Whether an unsharded BF16 decode projection uses legacy tiling.

        The dynamic dense-BF16 matmul path is not yet a proven replacement for
        the compile-time tiler.  Models that phase-share an unsharded M=1
        matrix can opt into the same static path used by Gemma4 E2B.
        """
        return False

    def _decode_use_one_round_group_attention(self) -> bool:
        """Whether decode assigns each complete GQA group to one engine.

        The shared Qwen path keeps its established two-round transpose/PV
        split.  A model whose KV-group count fits the engine topology may opt
        into the one-round path: one engine owns each complete group while
        every otherwise-idle engine still participates in the rendezvous.
        """
        return False

    def _prepare_decode_shared_weights(self, layer_size: int) -> None:
        """Optional post-sharding hook for phase-shared decoder weights."""
        return None

    def _prefill_use_streaming_quantized_projection(self, tag: str) -> bool:
        """Whether a row-sharded prefill projection uses the streaming core.

        The released Qwen2.5-VL dimensions fit the general dynamic matmat
        tiler, so its established path remains the default.  Compatible models
        with a larger inner dimension can opt individual projections into the
        runtime-row, one-pass quantized core used by Gemma4 E2B's configurable
        prefill implementation.
        """
        return False

    def _prefill_mlp_k_lanes(self) -> int:
        """K-lane count for the prefill gated MLP (1 = one full-width chain).

        ``down_proj`` contracts over the MLP intermediate, so its K is the
        model's intermediate size.  ``matmat_mul_core_dynamic`` dequantizes a
        K x N_chunk strip of B into URAM, so a >=64-wide column strip needs
        ``K <= URAM_NEAR_FULL_ELEMENTS // 64`` (4095), and it refuses outright
        at ``K // 64 > 255``.  Past that the only kernel left is
        ``quantized_matmat_core``, whose M>1 path re-streams B once PER ROW --
        correct, but it turns down_proj into a weight-bandwidth wall.

        Models whose intermediate exceeds the cap split down's K into lanes
        instead.  gate/up are sliced along N -- a contiguous ROW slice of the
        (N, K) weight, so the along-K quant blocking is untouched and no
        repack is needed -- each lane's activation lands in its own dense
        [rows, LANE] plane, and the per-lane down partials are summed.  Only
        down_proj's weight needs a lane-major repack, because K is its inner
        dimension; :meth:`_prefill_down_lane` supplies it.

        Returning 1 keeps the historical single full-width chain byte for byte.
        """
        return 1

    def _prefill_down_lane(self, la: dict, lane: int, lanes: int) -> tuple:
        """(data, scale) DRAM addresses for one K-lane of down_proj.

        A K-lane is a COLUMN slice of the (N, K) weight, and no matmul kernel
        has a B row-stride override -- B's row stride is derived from K -- so
        the slice cannot be addressed inside the as-loaded image.  The lanes
        are therefore dense planes laid end to end over the SAME allocation,
        written by :meth:`repack_down_to_k_lanes`, and a lane is a plain
        offset.  The addresses are correct from compile time; only the BYTES
        behind them are, until the repack runs.  Compilation happens first by
        design, so the repack is ordered at prefill execution instead.
        """
        if lanes == 1:
            return la["down_data"], la["down_scale"]
        d = self._lm_dims()
        lane_k = d["MLP"] // lanes
        return (la["down_data"] + lane * d["H"] * (lane_k // 2),
                la["down_scale"] + lane * d["H"]
                * (lane_k // UE_VECTOR_SIZE) * self.bytes_per_element)

    def _prefill_mlp_lane_geometry(self) -> tuple:
        """(lanes, LANE, lane_plane_bytes) for the K-split prefill MLP.

        ``lane_plane_bytes`` is the stride between lane planes inside the
        existing [P, MLP] gate/up/mult buffers, so lane c of GATE lives at
        ``LM_MLP_GATE + c * lane_plane_bytes`` with a ``LANE * bpe`` row
        stride -- exactly what the matmul wants for an A operand of K == LANE.
        """
        lanes = int(self._prefill_mlp_k_lanes())
        MLP = self._lm_dims()["MLP"]
        if lanes < 1:
            raise ValueError(f"_prefill_mlp_k_lanes() returned {lanes}")
        if MLP % lanes or (MLP // lanes) % UE_VECTOR_SIZE:
            # A lane boundary off a UE_VECTOR_SIZE multiple would cut an IF4
            # scale block in half: the blocks run along K, which is what is
            # being sliced.
            raise ValueError(
                f"MLP intermediate {MLP} does not split into {lanes} lanes of "
                f"whole {UE_VECTOR_SIZE}-element blocks")
        LANE = MLP // lanes
        return lanes, LANE, self.PREFILL_MAX_SEQ_LEN * LANE * self.bytes_per_element

    def repack_down_to_k_lanes(self) -> None:
        """Rewrite every layer's down_proj image in place as dense K-lanes.

        The loaded image is (N, K) row major -- scales as (N, K/64) bf16, then
        nibbles as (N, K/2) -- so a K-lane is a column slice of each, byte
        aligned because a lane is a whole number of 64-element blocks.  This
        re-groups those same bytes into ``lanes`` dense (N, K/lanes) planes
        end to end, so lane c starts at a fixed offset and has a row stride
        equal to its own K.  Same allocation, same total bytes; no extra DRAM.

        IN PLACE AND ONE WAY.  Anything that reads down_proj as one full-width
        image -- notably the decode column shards, which copy card -> host ->
        card out of this exact allocation -- must already have run.  Call it
        once, immediately before prefill executes.
        """
        if self._prefill_mlp_tp_engines():
            # Tensor-parallel prefill never reads a shared down_proj image --
            # its K-shards were sliced into the private windows at load time --
            # so there is nothing here to repack, and doing it would rewrite
            # bytes the decode column shards still own.
            return
        lanes = self._prefill_mlp_k_lanes()
        if lanes == 1 or getattr(self, "_down_k_lanes_packed", False):
            return
        d = self._lm_dims()
        N, K = d["H"], d["MLP"]
        blocks_k = K // UE_VECTOR_SIZE
        scale_bytes, data_bytes = N * blocks_k * 2, N * (K // 2)
        t0 = time.perf_counter()
        for la in self.lm_layer_addrs:
            for addr, nbytes, dtype, width in (
                    (la["down_scale"], scale_bytes, "<u2", blocks_k),
                    (la["down_data"], data_bytes, np.uint8, K // 2)):
                # A bytearray keeps dma_read on its raw-bytes path; a typed
                # buffer would go through a NUMERIC cast instead (see
                # multi_engine_shard._save_dram_selftest_region).
                buf = bytearray(nbytes)
                got = self.dma_read(
                    user_dma_core.DMA_DEVICE_C2H, addr, buf, nbytes)
                if got != nbytes:
                    raise IOError(
                        f"down_proj repack read {got} of {nbytes} bytes")
                plane = np.frombuffer(bytes(buf), dtype=dtype).reshape(N, width)
                # (N, lanes, width/lanes) -> (lanes, N, width/lanes)
                packed = np.ascontiguousarray(
                    plane.reshape(N, lanes, width // lanes).transpose(1, 0, 2)
                ).tobytes()
                written = self.dma_write(
                    user_dma_core.DMA_DEVICE_H2C, addr, packed, nbytes)
                if written != nbytes:
                    raise IOError(
                        f"down_proj repack wrote {written} of {nbytes} bytes")
        self._down_k_lanes_packed = True
        self._loud(
            f"  [LM] down_proj repacked into {lanes} K-lanes of "
            f"{K // lanes} across {len(self.lm_layer_addrs)} layers "
            f"({time.perf_counter() - t0:.1f}s)")

    def _lm_dims(self) -> dict:
        fi = self._cfg["file_info"]
        qh = fi["num_kv_heads"] * fi["group_size"]
        return dict(H=fi["hidden_size"], AHD=fi["actual_head_dim"],
                    KVH=fi["num_kv_heads"], G=fi["group_size"], QH=qh,
                    MLP=fi["mlp_elements"], NL=fi["num_layers"],
                    VOCAB=fi["embedding_vocab"])

    def _read_lm_region(self) -> dict:
        # Generates params.bin + params.json from the HF checkpoint on a machine
        # that has neither. A no-op once they exist.
        bin_path = _weight_gen().ensure_params_bin(self.script_dir)
        json_path = bin_path.rsplit(".", 1)[0] + ".json"
        with open(json_path) as f:
            manifest = json.load(f)
        r = (manifest.get("regions") or {}).get("lm")
        if r is None:
            raise KeyError(f"no 'lm' region in {json_path}")
        return dict(bin_path=bin_path, base_offset=int(r["offset"]),
                    size=int(r["size"]), sections=r["manifest"])

    def lm_weight_init(self) -> None:
        """Load LM weights over the params window. Idempotent.

        Projection storage follows ``precision.lm_quantized_projections`` and
        the LM head is IF4. Embeddings default to the original host-side BF16
        gather. Models that set ``precision.embedding`` to ``if4`` or ``if8``
        retain only the artifact descriptor here and provide an accelerator
        lookup hook.
        """
        if getattr(self, "_lm_weight_init_done", False):
            return
        d = self._lm_dims()
        region = self._read_lm_region()
        sec, sfx = region["sections"], LM_QUANT_PRECISION

        if getattr(self, "_vision_weight_init_done", False):
            self._loud("  [LM] reclaiming the params window from vision weights")
        self.reset_params_dram_addr()
        # The first successful DMA below starts destroying every previous
        # occupant of this phase-shared window. Invalidate optimistic cache
        # flags before that can happen so a short write remains safely retryable.
        self._lm_weight_init_done = False
        self._vision_weight_init_done = False
        if hasattr(self, "_audio_weight_init_done"):
            self._audio_weight_init_done = False
        start = self.get_params_dram_addr()
        quantized = self._lm_quantized_projections()
        q_desc = "/".join(t.upper() for t in
                          ("q", "k", "v", "o", "gate", "up", "down")
                          if t in quantized)
        dense_desc = "/".join(t.upper() for t in
                              ("q", "k", "v", "o", "gate", "up", "down")
                              if t not in quantized)
        storage = f"{sfx.upper()} {q_desc}"
        if dense_desc:
            storage += f", BF16 {dense_desc}"
        self._loud(f"  [LM] loading {d['NL']} layers ({storage}) at "
                   f"0x{start:X} ...")

        def need(k):
            if k not in sec:
                raise KeyError(f"LM weight {k!r} missing from params.bin")
            return sec[k]

        with open(region["bin_path"], "rb") as f:
            base = region["base_offset"]
            self.lm_layer_addrs = []
            for i in range(d["NL"]):
                pre = f"language_model.layers.{i}"
                la = {}
                projections = (
                    ("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"),
                    ("v", "self_attn.v_proj"), ("o", "self_attn.o_proj"),
                    ("gate", "mlp.gate_proj"), ("up", "mlp.up_proj"),
                    ("down", "mlp.down_proj"),
                )
                for tag, key in projections:
                    if self._lm_projection_is_private(tag):
                        # Staged per-engine instead of into the shared image;
                        # the subclass owns the slicing and records its own
                        # addresses in `la`.
                        self._stage_private_lm_projection(
                            f, need(f"{pre}.{key}.weight.{sfx}"), base, la,
                            tag, i, f"{pre}.{key}")
                    elif tag in quantized:
                        la[f"{tag}_scale"], la[f"{tag}_data"] = self._dma_if4(
                            f, need(f"{pre}.{key}.weight.{sfx}"), base,
                            f"{pre}.{key}")
                    else:
                        la[f"{tag}_weight"] = self._dma_bf16(
                            f, need(f"{pre}.{key}.weight"), base,
                            f"{pre}.{key}")
                for tag, key in (("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"),
                                 ("v", "self_attn.v_proj")):
                    la[f"{tag}_bias"] = self._dma_bf16(
                        f, need(f"{pre}.{key}.bias"), base, f"{pre}.{key}.bias")
                la["ln1"] = self._dma_bf16(
                    f, need(f"{pre}.input_layernorm.weight"), base, f"{pre}.ln1")
                la["ln2"] = self._dma_bf16(
                    f, need(f"{pre}.post_attention_layernorm.weight"), base, f"{pre}.ln2")
                self.lm_layer_addrs.append(la)
                if (i + 1) % 8 == 0 or i == d["NL"] - 1:
                    self._loud(f"    layer {i + 1}/{d['NL']} loaded")

            self.final_norm_addr = self._dma_bf16(
                f, need("language_model.norm.weight"), base, "final_norm")
            self.lm_head_scale, self.lm_head_data = self._dma_if4(
                f, need(f"lm_head.weight.{sfx}"), base, "lm_head")

            embedding_precision = self._cfg.get("precision", {}).get(
                "embedding", "bf16"
            )
            if embedding_precision == "bf16":
                # Original Qwen2.5-VL path: retain the table in host RAM and
                # DMA only rows selected by token IDs.
                s = need("language_model.embed_tokens.weight")
                f.seek(base + s["offset"])
                raw = f.read(s["size"])
                if len(raw) != s["size"]:
                    raise RuntimeError("truncated host embedding table")
                self.embedding_weight = torch.frombuffer(
                    bytearray(raw), dtype=torch.bfloat16
                ).reshape(d["VOCAB"], d["H"])
                self._embedding_artifact = None
                embedding_desc = f"{len(raw) / 2**20:.1f} MiB kept on host"
            elif embedding_precision in ("if4", "if8"):
                s = need(
                    f"language_model.embed_tokens.weight.{embedding_precision}"
                )
                self._embedding_artifact = {
                    "bin_path": region["bin_path"],
                    "file_offset": base + int(s["offset"]),
                    "section": dict(s),
                    "precision": embedding_precision,
                }
                if hasattr(self, "embedding_weight"):
                    del self.embedding_weight
                embedding_desc = (
                    f"{int(s['size']) / 2**20:.1f} MiB {embedding_precision.upper()} "
                    "reserved for device lookup"
                )
            else:
                raise ValueError(
                    "precision.embedding must be 'bf16', 'if4', or 'if8', "
                    f"got {embedding_precision!r}"
                )

        self._lm_weight_end = self.get_params_dram_addr()
        used = self._lm_weight_end - start
        if self._lm_weight_end > self.PARAMS_LIMIT:
            raise MemoryError(
                f"LM weights overflow the params window: end "
                f"0x{self._lm_weight_end:X} > 0x{self.PARAMS_LIMIT:X}")
        self._lm_weight_init_done = True
        self._loud(f"  [LM] weights loaded: {used / 2**20:.1f} MiB "
                   f"(embedding {embedding_desc})")

        self._ensure_tokenizer()

    def _prefill_mlp_tp_engines(self) -> int:
        """Engines the prefill MLP is TENSOR-PARALLEL over, or 0 for row-shard.

        Row-sharding splits the sequence and needs every weight on every engine.
        Tensor-parallel splits the weights instead -- gate/up by output column,
        down by its contraction dim -- so each engine holds 1/n of the MLP and
        one cross-engine reduce_add per layer joins the result. A model whose
        map cannot hold a shared MLP copy returns its engine count here and
        stages the shards via :meth:`_lm_projection_is_private`.
        """
        return 0

    def _reuse_prefill_tp_tensor_scratch(self) -> bool:
        """Whether the TP down-output plane also owns transient LM storage.

        Opt in only when prefill always takes :meth:`_emit_prefill_mlp_tp`.
        """
        return False

    def _emit_prefill_mlp_tp(self, sched, la, M, in_addr, out_addr,
                             m_regs) -> int:
        """One layer's MLP, tensor-parallel. Returns FLOPs emitted.

        Four regions, and the boundaries are where the dataflow actually turns:

          1. residual1 + norm   ROW-sharded -- pure elementwise/row-wise, and it
                                must JOIN because step 2 reads every row.
          2. gate/up/mult/down  COLUMN-sharded -- engine e runs ALL M rows for
                                lane e alone. The SiLU-multiply never leaves the
                                lane, so the whole chain is one region.
          3. reduce_add         the only cross-engine arithmetic in the layer.
          4. residual2          ROW-sharded again over the reduced result.
        """
        d = self._lm_dims()
        H, MLP = d["H"], d["MLP"]
        ne = int(sched.num_engines)
        if MLP % ne:
            raise ValueError(f"MLP={MLP} does not split {ne} ways")
        LANE = MLP // ne
        if LANE % UE_VECTOR_SIZE:
            raise ValueError(f"MLP lane {LANE} is not {UE_VECTOR_SIZE}-aligned")
        bpe = self.bytes_per_element
        h_row = H * bpe
        lane_plane = M * LANE * bpe
        acc = [0]

        # The ping-pong destination is dead until this layer produces it, so
        # the scratch-sharing layout keeps residual1 there through residual2.
        residual_addr = out_addr if self._reuse_prefill_tp_tensor_scratch() else self.LM_RESIDUAL

        def _pre(ctx):
            m = m_regs[ctx.engine_idx]
            ctx.ue.generate_instruction_add_set(m, ctx.rows)
            acc[0] += ctx.ue.eltwise_core_dram(
                M=ctx.rows, N=H,
                dram_a=ctx.rows_addr(in_addr, h_row),
                dram_b=ctx.rows_addr(self.LM_ATTN_PROJ, h_row),
                dram_out=ctx.rows_addr(residual_addr, h_row),
                mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=m) or 0
            acc[0] += ctx.ue.rms_norm_core_dram(
                M=ctx.rows, N=H,
                A_DRAM_ADDR=ctx.rows_addr(residual_addr, h_row),
                OUTPUT_DRAM_ADDR=ctx.rows_addr(self.LM_MLP_NORM, h_row),
                GAMMA_DRAM_ADDR=la["ln2"], gpr_M_reg=m) or 0

        sched.sharded_region(M, _pre, join=True)

        def _tp(ctx):
            e = ctx.engine_idx
            m = m_regs[e]
            if ctx.cols != LANE:
                raise AssertionError(
                    f"column split gave {ctx.cols}, expected lane {LANE}")
            ctx.ue.generate_instruction_add_set(m, M)
            for tag, plane in (("gate", self.LM_MLP_GATE),
                               ("up", self.LM_MLP_UP)):
                data, scale = la[f"{tag}_tp"][e]
                acc[0] += ctx.ue.matmat_mul_core(
                    M=M, K=H, N=LANE, A_DRAM_ADDR=self.LM_MLP_NORM,
                    B_DRAM_ADDR=data, SCALE_DRAM_ADDR=scale,
                    is_B_quantized=True, data_type=TYPE.IF4,
                    OUTPUT_DRAM_ADDR=plane + e * lane_plane,
                    silu_enable=(tag == "gate"), gpr_M_reg=m) or 0
            acc[0] += ctx.ue.eltwise_core_dram(
                M=M, N=LANE,
                dram_a=self.LM_MLP_GATE + e * lane_plane,
                dram_b=self.LM_MLP_UP + e * lane_plane,
                dram_out=self.LM_MLP_MULT + e * lane_plane,
                mode=UE_MODE.ELTWISE_MUL, gpr_M_reg=m) or 0
            data, scale = la["down_tp"][e]
            acc[0] += ctx.ue.matmat_mul_core(
                M=M, K=LANE, N=H,
                A_DRAM_ADDR=self.LM_MLP_MULT + e * lane_plane,
                B_DRAM_ADDR=data, SCALE_DRAM_ADDR=scale,
                is_B_quantized=True, data_type=TYPE.IF4,
                OUTPUT_DRAM_ADDR=self.LM_MLP_DOWN_TP + e * M * h_row,
                gpr_M_reg=m) or 0

        sched.col_sharded_region(MLP, _tp, join=True)
        sched.reduce_add(
            [self.LM_MLP_DOWN_TP + e * M * h_row for e in range(ne)],
            self.LM_MLP_DOWN, M=M, N=H, parallel=True)

        def _post(ctx):
            m = m_regs[ctx.engine_idx]
            ctx.ue.generate_instruction_add_set(m, ctx.rows)
            acc[0] += ctx.ue.eltwise_core_dram(
                M=ctx.rows, N=H,
                dram_a=ctx.rows_addr(residual_addr, h_row),
                dram_b=ctx.rows_addr(self.LM_MLP_DOWN, h_row),
                dram_out=ctx.rows_addr(out_addr, h_row),
                mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=m) or 0

        sched.sharded_region(M, _post, join=True)
        return acc[0]

    def _decode_shard_override(self, op: str, layer: int):
        """A ShardedWeight already staged privately, or None to shard normally.

        Companion to :meth:`_lm_projection_is_private`: a projection that never
        entered the shared image has no source address for
        ``shard_quantized_weight`` to copy FROM, so the subclass hands over the
        shards it staged itself.
        """
        return None

    def _lm_projection_is_private(self, tag: str) -> bool:
        """Whether this projection is staged PER ENGINE instead of shared.

        Default: nothing is, so the shared params image is built exactly as
        before. A model whose map cannot hold a shared copy of a projection
        overrides this and :meth:`_stage_private_lm_projection` together.
        """
        return False

    def _stage_private_lm_projection(self, file_obj, section: dict,
                                     base_offset: int, la: dict, tag: str,
                                     layer: int, what: str) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} says {tag!r} is private but does not "
            f"implement _stage_private_lm_projection()")

    def _ensure_tokenizer(self):
        """Load the tokenizer on demand.

        The VLM path needs it BEFORE lm_weight_init -- it builds the prompt (and
        from it the mRoPE positions) while the vision weights are still resident
        in the params window.
        """
        if not hasattr(self, "tokenizer"):
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(self.script_dir, self._cfg["paths"]["hf_model_dir"]),
                trust_remote_code=True)
        return self.tokenizer

    def get_embedding_for_tokens(self, token_ids) -> torch.Tensor:
        if not hasattr(self, "embedding_weight"):
            raise RuntimeError(
                "this model keeps embeddings on the accelerator; use its "
                "device embedding loader"
            )
        return self.embedding_weight[torch.as_tensor(list(token_ids),
                                                     dtype=torch.long)].contiguous()

    def _device_embedding_enabled(self) -> bool:
        """Whether token rows are produced by an accelerator-specific hook."""
        return False

    def _prefill_execution_rows(self, seq_len: int) -> int:
        """Rows the FPGA executes for ``seq_len`` live prompt tokens.

        Most models execute the live row count directly.  A concrete model may
        round this up when its multi-engine kernels require a non-ragged shape;
        :meth:`run_prefill` keeps every extra row finite and masks its key
        column, so padded rows cannot affect live-token results.
        """
        return seq_len

    def _load_device_embeddings(self, token_ids, output_dram_addr: int) -> None:
        raise NotImplementedError("device embedding lookup is not implemented")

    def _emit_device_decode_embedding(self, token: int, output_dram_addr: int) -> None:
        raise NotImplementedError("device decode embedding lookup is not implemented")

    # ---- RoPE --------------------------------------------------------------

    def _mrope_dim_source(self) -> torch.Tensor:
        """Which of (t, h, w) each of the AHD/2 half-dims takes its position from.

        mrope_section [16, 24, 24] splits the 64 half-dims: the first 16 rotate
        by the temporal index, the next 24 by height, the last 24 by width. Text
        tokens set t = h = w, so the same table construction covers both cases.
        """
        sec = self._cfg["special"]["rope"]["mrope_section"]
        return torch.cat([torch.full((n,), i, dtype=torch.long)
                          for i, n in enumerate(sec)])

    def _build_rope_table(self, positions: torch.Tensor) -> torch.Tensor:
        """[n, 4*AHD/2] rows of [cos | cos | -sin | sin] for rope_hf_core.

        ``positions`` is either [n] (1-D, text-only) or [n, 3] carrying the
        mRoPE (t, h, w) triple per token -- image tokens occupy a 2-D grid, so
        their height and width indices differ from their sequence order.

        Half-width AHD/2 duplicated because rotate-half at width AHD pairs lane
        i with i+AHD/2; sin's lower half is pre-negated so the core's two
        elementwise passes are add-only.
        """
        d = self._lm_dims()
        half = d["AHD"] // 2
        theta = self._cfg["special"]["rope"]["theta"]
        inv = 1.0 / (theta ** (torch.arange(half, dtype=torch.float32) / half))
        if positions.dim() == 1:
            f = torch.outer(positions.float(), inv)
        else:
            # Per half-dim, take the position component that dim rotates by.
            f = positions.float()[:, self._mrope_dim_source()] * inv
        c, s = f.cos().to(torch.bfloat16), f.sin().to(torch.bfloat16)
        return torch.cat([c, c, -s, s], dim=1).contiguous()

    def load_rope_for_positions(self, positions: torch.Tensor, *,
                                decode: bool = False) -> None:
        """Upload the RoPE table: one row per position, NO per-head tiling.

        Every head's rope call reads rows 0..M-1, because the heads are rotated
        one at a time (their planes are head_rows apart and so not contiguous).
        """
        self.dma_to_accelerator_memory(
            self.LM_ROPE_DEC if decode else self.LM_ROPE_PRE,
            self._build_rope_table(positions).flatten())

    # ---- tensors -----------------------------------------------------------

    def lm_tensor_init(self) -> None:
        """Allocate LM activations and the KV cache.

        Vision tensors are NOT reused: vision runs to completion first and its
        output is on the host, so the whole tensor region is rewound and
        re-carved for the LM.
        """
        d = self._lm_dims()
        H, AHD, KVH, QH, G = (
            d["H"], d["AHD"], d["KVH"], d["QH"], d["G"]
        )
        MLP, NL = d["MLP"], d["NL"]
        C, P = self.MAX_CONTEXT_SIZE, self.PREFILL_MAX_SEQ_LEN
        bpe = self.bytes_per_element
        self.reset_tensor_dram_addr()

        def alloc(n, what):
            return self.allocate_tensor_dram(n * bpe, label=what)

        # Double-buffered layer I/O: layer li reads A and writes B when li is
        # even, and the reverse when odd, so no inter-layer copy is emitted.
        self.LM_IO_A = alloc(P * H, "lm.io_a")
        self.LM_IO_B = alloc(P * H, "lm.io_b")

        tp_ne = self._prefill_mlp_tp_engines()
        reuse_tp_scratch = self._reuse_prefill_tp_tensor_scratch()
        if reuse_tp_scratch and not tp_ne:
            raise RuntimeError("TP tensor scratch reuse requires a TP prefill MLP")

        if reuse_tp_scratch:
            # One physical plane has three non-overlapping lifetimes:
            #
            #   qkv:       pre_norm | q | k | v
            #   attention: result   | projected result
            #   MLP:       norm, then all eight down-projection partials
            #
            # Q_HM and ATTN_HM remain dedicated because attention addresses
            # their padded head planes. Bias, attention scratch and KV caches
            # are also deliberately outside this overlay.
            tp_elems = tp_ne * P * H
            self.LM_MLP_DOWN_TP = alloc(tp_elems, "lm.mlp_down_tp_scratch")
            scratch_base = self.LM_MLP_DOWN_TP
            scratch_end = scratch_base + tp_elems * bpe

            self.LM_PRE_NORM = scratch_base
            self.LM_Q = self.LM_PRE_NORM + P * H * bpe
            self.LM_K = self.LM_Q + P * QH * AHD * bpe
            self.LM_V = self.LM_K + P * KVH * AHD * bpe
            qkv_end = self.LM_V + P * KVH * AHD * bpe
            if qkv_end > scratch_end:
                raise MemoryError("Q/K/V transient layout exceeds TP scratch")

            self.LM_ATTN_RESULT = scratch_base
            self.LM_ATTN_PROJ = self.LM_ATTN_RESULT + P * QH * AHD * bpe
            if self.LM_ATTN_PROJ + P * H * bpe > scratch_end:
                raise MemoryError("attention transient layout exceeds TP scratch")
            self.LM_MLP_NORM = scratch_base
            for label, address in (
                ("lm.pre_norm", self.LM_PRE_NORM),
                ("lm.q", self.LM_Q),
                ("lm.k", self.LM_K),
                ("lm.v", self.LM_V),
                ("lm.attn_result", self.LM_ATTN_RESULT),
                ("lm.attn_proj", self.LM_ATTN_PROJ),
                ("lm.mlp_norm", self.LM_MLP_NORM),
            ):
                self._dram_addresses[label] = address
        else:
            self.LM_PRE_NORM = alloc(P * H, "lm.pre_norm")
            self.LM_Q = alloc(P * QH * AHD, "lm.q")
            self.LM_K = alloc(P * KVH * AHD, "lm.k")
            self.LM_V = alloc(P * KVH * AHD, "lm.v")
        # Sized for the largest program (prefill at PREFILL_MAX_SEQ_LEN); each
        # program strides these planes by its own M.
        head_rows = P
        self.LM_HEAD_ROWS = head_rows
        self.LM_Q_HM = alloc(QH * head_rows * AHD, "lm.q_hm")
        self.LM_ATTN_HM = alloc(QH * head_rows * AHD, "lm.attn_hm")
        if not reuse_tp_scratch:
            self.LM_ATTN_RESULT = alloc(P * QH * AHD, "lm.attn_result")
            self.LM_ATTN_PROJ = alloc(P * H, "lm.attn_proj")
            self.LM_RESIDUAL = alloc(P * H, "lm.residual")
            self.LM_MLP_NORM = alloc(P * H, "lm.mlp_norm")
        else:
            # Decode still needs one residual row. TP prefill keeps its
            # residual in the layer's ping-pong output buffer.
            self.LM_RESIDUAL = alloc(H, "lm.residual_decode")
        self.LM_MLP_GATE = alloc(P * MLP, "lm.mlp_gate")
        self.LM_MLP_UP = alloc(P * MLP, "lm.mlp_up")
        if reuse_tp_scratch:
            # eltwise_core_dram supports an output aliasing input A (the
            # existing down accumulator already relies on it). Once gate*up
            # is formed, UP is dead and becomes the down/reduce destination.
            self.LM_MLP_MULT = self.LM_MLP_GATE
            self.LM_MLP_DOWN = self.LM_MLP_UP
            self._dram_addresses["lm.mlp_mult"] = self.LM_MLP_MULT
            self._dram_addresses["lm.mlp_down"] = self.LM_MLP_DOWN
        else:
            self.LM_MLP_MULT = alloc(P * MLP, "lm.mlp_mult")
            self.LM_MLP_DOWN = alloc(P * H, "lm.mlp_down")
        # K-lane split only: one lane's down partial before it is summed into
        # LM_MLP_DOWN.  GATE/UP/MULT need no extra space -- the lanes are a
        # re-interpretation of the same [P, MLP] planes as lanes x [P, LANE].
        if self._prefill_mlp_k_lanes() > 1:
            self.LM_MLP_DOWN_PART = (None if reuse_tp_scratch else
                                     alloc(P * H, "lm.mlp_down_part"))
        if tp_ne and not reuse_tp_scratch:
            # Tensor-parallel down contracts over the LANE dim, so every engine
            # produces a FULL-width [P, H] partial and reduce_add sums all of
            # them. The K-lane accumulator above is the single-engine analogue.
            self.LM_MLP_DOWN_TP = alloc(tp_ne * P * H, "lm.mlp_down_tp")
        self.LM_OUT_NORM = alloc(H, "lm.out_norm")
        self.LOGITS = alloc(d["VOCAB"], "lm.logits")
        # Repetition-penalty bias: the LM-head matmul's C term, so the HW argmax
        # of (logits + bias) is the penalized token and no logits come back.
        self.PENALTY_BIAS = alloc(d["VOCAB"], "lm.penalty_bias")

        aligned_P = ((P + 63) // 64) * 64
        aligned_C = ((C + 63) // 64) * 64
        # SIZES ARE COMPUTED ONCE AND REUSED. lm_reset_attention_state has to
        # zero exactly these element counts; when it recomputed them from its
        # own expressions the two drifted, its scratch fill overran by 260 K
        # elements and wiped LM_IDENTITY (allocated right after) -- which made
        # the attention V-transpose produce zeros and every token garbage.
        n_bias = max(aligned_P * aligned_P, QH * aligned_C)
        # score/P inside the core is [aligned_seq, aligned_seq], so decode at a
        # long context -- not prefill -- sets the scratch size.
        # V.T [AHD, A] + score [A, A] + scaled_q [BATCH, AHD], where A is the
        # largest aligned_seq_len any call uses and BATCH the largest 64-aligned
        # batch. The scaled_q term is batch*AHD -- sizing it QH*AHD left the
        # buffer 6144 elements short at decode's A=2048, so the FIRST attention
        # call of each layer overran into LM_IDENTITY and every LATER call read
        # a corrupted identity: head 0 correct, heads 1..15 garbage.
        A = max(aligned_P, aligned_C)
        BATCH = P
        n_scratch = (AHD + A) * A + BATCH * AHD
        self.LM_BIAS = alloc(n_bias, "lm.bias")
        self.LM_SCRATCH = alloc(n_scratch, "lm.attn_scratch")
        self._lm_zero_sizes = dict(
            kv=NL * KVH * C * AHD, hm=QH * self.LM_HEAD_ROWS * AHD,
            bias=n_bias, scratch=n_scratch)
        # Guard between the attention scratch and IDENTITY. The kernel's scratch
        # extent is derived from its own arguments, so a sizing mistake here is
        # silent -- it lands on whatever is allocated next. IDENTITY was that
        # neighbour twice.
        self._lm_scratch_guard = alloc(64 * 1024, "lm.scratch_guard")
        # Head-sharded prefill attention needs ONE PRIVATE scratch per engine:
        # heads run concurrently and each builds its own V.T / scores / scaled_q.
        # Sized for the PREFILL shape, not the decode-worst-case LM_SCRATCH --
        # 480 KiB against 8.6 MiB, and 8 copies of the latter would be 69 MiB
        # for buffers prefill never fills.
        self.LM_ATTN_SCRATCH_PER_ENGINE = [self.LM_SCRATCH]
        if getattr(self, "multi_core", 1) > 1:
            n_pref = (AHD + aligned_P) * aligned_P + aligned_P * AHD
            if self._decode_use_one_round_group_attention():
                # Compact one-group decode scratch: V.T [AHD,A], probabilities
                # [G,A], and scaled Q [G,AHD].  Unlike unified_attention_core's
                # general scratch this does not reserve an unused [A,A] score
                # matrix when the live decode batch is only one GQA group.
                n_group_decode = AHD * aligned_C + G * aligned_C + G * AHD
                n_pref = max(n_pref, n_group_decode)
            self.LM_ATTN_SCRATCH_PER_ENGINE.extend(
                self.mc_arena.alloc_tensor(e, n_pref * bpe, "lm prefill attn scratch")
                for e in range(1, self.multi_core))
            self._lm_worker_attn_scratch_elements = n_pref
        self.LM_IDENTITY = alloc(UE_VECTOR_SIZE * UE_VECTOR_SIZE, "lm.identity")
        # One row per position; every head reads the same rows.
        self.LM_ROPE_PRE = alloc(P * 2 * AHD, "lm.rope_prefill")
        self.LM_ROPE_DEC = alloc(2 * AHD, "lm.rope_decode")

        # KV cache: [layer][kv_head][C][AHD], head-major so decode attention
        # reads it in place -- no per-step marshalling.
        self.KV_STRIDE_HEAD = C * AHD * bpe
        self.KV_STRIDE_LAYER = KVH * self.KV_STRIDE_HEAD
        self.LM_K_CACHE = alloc(NL * KVH * C * AHD, "lm.k_cache")
        self.LM_V_CACHE = alloc(NL * KVH * C * AHD, "lm.v_cache")

        end = self.get_tensor_dram_addr()
        if end > self.TENSOR_LIMIT:
            raise MemoryError(
                f"LM tensors need {(end - self._tensor_dram_base) / 2**20:.1f} MiB "
                f"but the tensor region is "
                f"{(self.TENSOR_LIMIT - self._tensor_dram_base) / 2**20:.1f} MiB. "
                f"Lower MAX_CONTEXT_SIZE / PREFILL_MAX_SEQ_LEN, or move "
                f"TENSOR_BASE down.")
        self.dma_to_accelerator_memory(
            self.LM_IDENTITY, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))
        self.dma_to_accelerator_memory(
            self.PENALTY_BIAS, torch.zeros(d["VOCAB"], dtype=torch.bfloat16))
        self._loud(f"  [LM] tensors: {self.get_tensor_dram_usage() / 2**20:.1f} MiB "
                   f"(KV {2 * NL * KVH * C * AHD * bpe / 2**20:.1f} MiB at ctx {C})")
        self.lm_reset_attention_state()

    def lm_reset_attention_state(self) -> None:
        """Zero EVERY buffer unified_attention_core reads, over its WHOLE span.

        Not an optimisation -- correctness. The kernel always runs the 64-aligned
        length, so rows and columns past the live sequence are multiplied on
        every call and discarded only afterwards by the -inf bias, which needs
        them FINITE. The pre-run DRAM clean fills 0xFF, which is NaN in bf16,
        and -inf + NaN = NaN: one stale pad element takes out an entire softmax
        row, and nothing shows it until the tokens come out wrong.
        gemma4_e2b does the same before every prefill, for the same reason.
        """
        n = self._lm_zero_sizes
        kv = torch.zeros(n["kv"], dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LM_K_CACHE, kv)
        self.dma_to_accelerator_memory(self.LM_V_CACHE, kv)
        hm = torch.zeros(n["hm"], dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LM_Q_HM, hm)
        self.dma_to_accelerator_memory(self.LM_ATTN_HM, hm)
        self.dma_to_accelerator_memory(
            self.LM_SCRATCH, torch.zeros(n["scratch"], dtype=torch.bfloat16))
        # The bias region past the live length is written ONLY here, and that is
        # exactly the region the kernel reads as padding.
        self.dma_to_accelerator_memory(
            self.LM_BIAS, torch.full((n["bias"],), float("-inf"),
                                     dtype=torch.bfloat16))
        # Re-write IDENTITY last: the attention V-transpose reads it, and it is
        # the buffer an over-long neighbouring fill would land on.
        self.dma_to_accelerator_memory(
            self.LM_IDENTITY, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))
        self._loud(f"  [LM] attention state zeroed "
                   f"(KV {2 * kv.numel() * 2 / 2**20:.0f} MiB + scratch + bias)")

    # ---- program emission --------------------------------------------------

    def _ensure_decode_shards(self, sched, layer_size: int) -> dict:
        """Build the complete decode shard set atomically."""
        if getattr(self, "_down_k_lanes_packed", False):
            # The shards are COLUMN blocks of the full-width (N, K) images.
            # After repack_down_to_k_lanes() down_proj is no longer full width,
            # so slicing it here would silently produce transposed garbage.
            raise RuntimeError(
                "decode shards must be built before repack_down_to_k_lanes(); "
                "the down_proj image is now lane-major")
        missing = object()
        arena_cursors_before = list(sched.arena._weight_cursor)
        scheduler_weights_before = dict(sched._weights)
        decode_shards_before = getattr(self, "_decode_shards", missing)
        lm_shard_before = getattr(self, "_decode_lm_shard", missing)
        try:
            return self._ensure_decode_shards_impl(sched, layer_size)
        except Exception:
            # A shard copy is card -> host -> card and may fail after earlier
            # projections were fully cached. Roll back the whole set so retry
            # neither collides with those names nor leaks private-arena space.
            sched.arena._weight_cursor[:] = arena_cursors_before
            sched._weights.clear()
            sched._weights.update(scheduler_weights_before)
            for attr, value in (
                ("_decode_shards", decode_shards_before),
                ("_decode_lm_shard", lm_shard_before),
            ):
                if value is missing:
                    if hasattr(self, attr):
                        delattr(self, attr)
                else:
                    setattr(self, attr, value)
            raise

    def _ensure_decode_shards_impl(self, sched, layer_size: int) -> dict:
        """Copy each engine's COLUMN block of the decode weights into its own
        private arena. Returns ``{(op, layer): ShardedWeight}``.

        WHY COLUMNS, AND WHY A COPY. Decode is M=1, so there are no rows to
        split -- the only parallel axis is the output width N. And decode is
        bandwidth-bound: a whole weight block is streamed per token, so if every
        engine read its block out of the ONE shared weight image their streams
        would contend and the speedup would cap however evenly N divides.
        Engine i therefore reads from ITS OWN window.

        SCOPE AT 8 ENGINES: q and o over all 8, k and v over 4. q is N=2048 -- 32 blocks of
        64, i.e. 4 blocks per engine. k is N=256, only 4 blocks, so it CANNOT
        reach 8 engines; rather than leave it full-width on the master it goes
        one block to each of engines 0-3 and engines 4-7 emit nothing for it.
        That is still the right trade: they are already stopped at this layer's
        rendezvous waiting for q, so k costs them nothing, and the master sheds
        three quarters of a projection. v shards the same 4 ways: it is BF16, so
        it carries no scale blob and lands on matmat_mul_core rather than the
        GEMV kernel, but a column block of a bf16 [N, K] blob is the same
        contiguous row block, so it materializes into the private arenas too.
        o is bf16 as well and N=H=2048, so it splits 8 ways like q -- but it
        consumes the ATTENTION output, so it cannot ride the qkv rendezvous and
        opens a second one after the permute. gate and up (N=11008, 172 blocks:
        21 per engine with the last four taking 22) read the same post-norm
        input and write disjoint buffers, so they share a third; down
        (N=H=2048) consumes their product and takes a fourth. Four rendezvous
        per layer is the minimum the dataflow allows -- each one separates a
        producer from its consumer.
        """
        cached = getattr(self, "_decode_shards", None)
        if cached is not None:
            return cached
        import multi_engine_shard as mes
        d = self._lm_dims()
        t0 = time.perf_counter()
        shards, skipped = {}, []
        for li in range(layer_size):
            la = self.lm_layer_addrs[li]
            for op, K, N in (("q", d["H"], d["QH"] * d["AHD"]),
                             ("k", d["H"], d["KVH"] * d["AHD"]),
                             ("v", d["H"], d["KVH"] * d["AHD"]),
                             ("o", d["QH"] * d["AHD"], d["H"]),
                             ("gate", d["H"], d["MLP"]),
                             ("up", d["H"], d["MLP"]),
                             ("down", d["MLP"], d["H"])):
                pre_staged = self._decode_shard_override(op, li)
                if pre_staged is not None:
                    # Already sliced into the private windows at weight-load
                    # time -- re-sharding it would copy the same bytes twice.
                    shards[(op, li)] = pre_staged
                    continue
                if not self._decode_projection_should_shard(op, li):
                    skipped.append((op, li))
                    continue
                quant = self._decode_projection_is_quantized(op)
                # An op narrower than one 64-column block per engine still
                # shards -- over as many engines as it fills.
                n_sh = min(sched.num_engines, mes.max_shards(N))
                if n_sh < 2:
                    skipped.append((op, li))
                    continue
                if quant:
                    sw = sched.shard_quantized_weight(
                        name=f"{op}_proj_L{li}",
                        main_weight_addr=la[f"{op}_data"],
                        main_scale_addr=la[f"{op}_scale"],
                        K=K, N=N, layers=1, main_layer_stride=0,
                        data_type=TYPE.IF4, max_engines=n_sh, verbose=False)
                else:
                    sw = sched.shard_bf16_weight(
                        name=f"{op}_proj_L{li}",
                        main_weight_addr=la[f"{op}_weight"],
                        K=K, N=N, layers=1, main_layer_stride=0,
                        max_engines=n_sh, verbose=False)
                shards[(op, li)] = sw
        # The head is not per-layer: one [151936, 2048] IF4 block, 2374 col
        # blocks, an even 296-297 per engine. It is ~8.7% of the bytes decode
        # streams per token, so it is worth a round of its own.
        self._decode_lm_shard = None
        if mes.max_shards(d["VOCAB"]) >= sched.num_engines:
            self._decode_lm_shard = sched.shard_quantized_weight(
                name="lm_head", main_weight_addr=self.lm_head_data,
                main_scale_addr=self.lm_head_scale,
                K=d["H"], N=d["VOCAB"], layers=1, main_layer_stride=0,
                data_type=TYPE.IF4, verbose=False)
        self._decode_shards = shards
        used = sched.private_usage()
        self._loud(f"  [Decode] sharded {len(shards)} projection(s) over "
                   f"{sched.num_engines} engines in {time.perf_counter() - t0:.1f}s"
                   f"{f'; {len(skipped)} left on the master' if skipped else ''}; "
                   f"private weight arenas: "
                   f"{', '.join(f'{u / 2**20:.1f} MiB' for u in used)}")
        return shards

    def _emit_dec_shard(self, ue, sw, e: int, out_base: int, a_addr: int,
                        bias_base: int = None, silu: bool = False) -> int:
        """Emit engine ``e``'s column block of one decode projection.

        B and the scales come from THIS engine's private arena; ``a_addr`` is the
        shared input every engine reads in full, and only the output slice --
        and the bias, which is per-column -- are per-engine.
        """
        import multi_engine_shard as mes
        sh = sw.shard(e)
        off = sh.col_offset * self.bytes_per_element
        # A shard is a whole multiple of 64 columns, so at bf16 the output
        # offset is a whole 128-byte SRAM row. Asserted, not assumed: a
        # misaligned writeback is finite-but-wrong data, not a fault.
        assert off % 128 == 0, f"shard output offset {off} is not a whole SRAM row"
        kw = dict(M=1, K=sw.K, N=sh.cols, A_DRAM_ADDR=a_addr,
                  B_DRAM_ADDR=sh.weight_addr, OUTPUT_DRAM_ADDR=out_base + off)
        if bias_base is not None:
            kw.update(C_DRAM_ADDR=bias_base + off, bias_mode="broadcast_N")
        if silu:
            # SwiGLU's gate half. The activation is elementwise, so it applies
            # to a column shard exactly as it does to the full width.
            kw.update(silu_enable=True)
        if sw.data_type is mes.DENSE_BF16:
            # bf16 weights have no scale blob and no GEMV kernel of their own;
            # the general matmat is what the unsharded path uses for them too.
            return ue.matmat_mul_core(**kw) or 0
        kw.update(SCALE_DRAM_ADDR=sh.scale_addr, data_type=TYPE.IF4)
        return ue.quantized_matmat_core(**kw) or 0

    # Decode attention is sharded two ways, both of which need values that only
    # exist at run time (the KV length grows every step). See _emit_layer.
    DEC_ATTN_REGS = ("row_off", "out_off", "rows", "stride", "aligned")

    def _ensure_decode_attn_regs(self, dec_sched) -> list:
        """One runtime register set per worker for the sharded decode attention.

        Allocated ONCE, before any layer is emitted, and reused by all 36 -- the
        split is the same in every layer, only the per-layer K/V and scratch
        bases differ, and those are literals.
        """
        regs = getattr(self, "_decode_attn_worker_regs", None)
        if regs is None or len(regs) != len(dec_sched.workers):
            regs = [{n: w.alloc_isa_reg() for n in self.DEC_ATTN_REGS}
                    for w in dec_sched.workers]
            self._decode_attn_worker_regs = regs
        else:
            # begin_program() deliberately resets worker allocators. A second
            # decoder compile (profiling or retry) reuses these stable runtime
            # IDs, so reserve them again before any temporary register is
            # allocated or it will alias row_off/out_off/... in the program.
            for worker, worker_regs in zip(dec_sched.workers, regs):
                next_free = max(worker_regs.values(), default=0) + 1
                if worker._isa_reg_counter < next_free:
                    worker._isa_reg_counter = next_free
        return regs

    def _decode_attn_worker_gpr_sets(self, dec_sched, aligned: int):
        """Per-step ``(register, value)`` pairs for the M-sharded V transpose.

        Splits the LIVE aligned KV length across the workers in whole 64-row
        blocks; the remainder goes to the leading workers so the slices stay
        contiguous and together cover exactly [0, aligned). 64 is not a choice:
        bf16_transpose_core writes each block as one strided DMA of 64 Y-rows
        whose URAM cursor steps by ceil(m_take/64) rows, so a non-multiple-of-64
        row count reads the DMA chunks at the wrong offsets.

        Fewer blocks than workers: the leading ones get a block each and the rest
        get ZERO, which they branch over. Never a 0-row transpose -- its row loop
        never terminates and the core hangs.

        The SAME split serves both KV groups; only the base addresses differ, and
        those are per-layer literals.
        """
        regs = getattr(self, "_decode_attn_worker_regs", [])
        if len(regs) != len(dec_sched.workers):
            raise RuntimeError(
                f"decode attention worker registers ({len(regs)}) do not match "
                f"the {len(dec_sched.workers)} worker engine(s); recompile")
        n = len(regs)
        bpe = self.bytes_per_element
        blocks = aligned // 64
        if blocks >= n:
            base, rem = divmod(blocks, n)
            counts = [64 * (base + (1 if i < rem else 0)) for i in range(n)]
        else:
            counts = [64 if i < blocks else 0 for i in range(n)]
        assert sum(counts) == aligned, (
            f"V^T shards {counts} do not cover aligned={aligned}")
        offsets = [sum(counts[:i]) for i in range(n)]
        return [[(r["row_off"], off), (r["out_off"], ue_35bit_addr_shifter(off * bpe)),
                 (r["rows"], cnt), (r["stride"], aligned * bpe),
                 (r["aligned"], aligned)]
                for r, off, cnt in zip(regs, offsets, counts)]

    def _start_decode_workers(self, dec_sched, worker_addrs,
                              aligned: int) -> list[int]:
        """THE only way to start the decode workers. Both the run loop and the
        profile path go through here so they cannot drift: a worker entering with
        a stale row count transposes the wrong rows. Return the runtime-preamble
        entries so a caller may relaunch those exact bytes while ``aligned`` is
        unchanged."""
        return dec_sched.start_workers(
            worker_addrs,
            gpr_sets_by_worker=self._decode_attn_worker_gpr_sets(dec_sched, aligned))

    def _decode_pv_shards(self, KVH: int, AHD: int, num_engines: int) -> list:
        """``[[(group, n_offset, columns), ...]]`` indexed by engine.

        P@V^T's N is AHD, so it splits into AHD/64 column blocks per KV group --
        2 x 2 = 4 blocks here, dealt round-robin. At 8 engines that leaves 4..7
        with nothing for this round; they still run its handshake.
        """
        blocks = [(g, n, 64) for g in range(KVH) for n in range(0, AHD, 64)]
        return [blocks[e::num_engines] for e in range(num_engines)]

    @staticmethod
    def _one_round_group_assignments(groups, num_engines: int) -> tuple:
        """Map one complete GQA group to each leading engine.

        ``None`` entries are intentional idle participants.  They must still
        emit the worker-side release/join protocol; dropping them would shift
        their flag stream by one round and deadlock the next rendezvous.
        """
        if num_engines < 1:
            raise ValueError(f"attention needs at least one engine, got {num_engines}")
        groups = tuple(groups)
        if not groups or len(groups) > num_engines:
            raise ValueError(
                f"cannot place {len(groups)} GQA group(s) on {num_engines} engine(s)"
            )
        return groups + (None,) * (num_engines - len(groups))

    def _emit_one_round_group_attention(
        self, ue, *, group, aligned_kv: int, aligned_kv_reg: int,
        k_base: int, v_base: int, scratch: int,
    ) -> tuple[int, int]:
        """Emit one complete decode GQA group on ``ue``.

        The general attention helper reserves an ``aligned_kv**2`` score
        matrix even though decode has only ``G`` query rows.  This compact form
        uses exactly V.T ``[AHD,A]``, score/P ``[G,A]``, and scaled-Q
        ``[G,AHD]``.  It returns ``(all_flops, kv_length_scaled_flops)`` so the
        decode reporter can keep the fixed Q scaling out of its context term.
        """
        d = self._lm_dims()
        AHD, G = int(d["AHD"]), int(d["G"])
        bpe = self.bytes_per_element
        kv_h, plane_off, batch = group
        if batch != G:
            raise ValueError(
                f"decode GQA group {kv_h} has batch {batch}, expected G={G}"
            )
        if aligned_kv_reg is None:
            raise ValueError("one-round decode attention needs runtime aligned KV")

        v_t = scratch
        score = v_t + AHD * aligned_kv * bpe
        scaled_q = score + G * aligned_kv * bpe
        batch_reg = ue.alloc_isa_reg()
        head_dim_reg = ue.alloc_isa_reg()
        ue.generate_instruction_add_set(batch_reg, G)
        ue.generate_instruction_add_set(head_dim_reg, AHD)
        try:
            ue.bf16_transpose_core(
                M=aligned_kv, N=AHD,
                INPUT_DRAM_ADDR=v_base + kv_h * self.KV_STRIDE_HEAD,
                OUTPUT_DRAM_ADDR=v_t,
                IDENTITY_DRAM_ADDR=self.LM_IDENTITY,
                gpr_M_reg=aligned_kv_reg,
            )
            fixed = ue.eltwise_core_dram(
                M=G, N=AHD,
                dram_a=self.LM_Q_HM + plane_off,
                dram_b=None, dram_out=scaled_q,
                mode=UE_MODE.MUL_BROADCAST,
                scalar=1.0 / math.sqrt(AHD),
                gpr_M_reg=batch_reg,
            ) or 0
            score_flops = ue.matmat_mul_core(
                M=G, K=AHD, N=aligned_kv,
                A_DRAM_ADDR=scaled_q,
                B_DRAM_ADDR=k_base + kv_h * self.KV_STRIDE_HEAD,
                OUTPUT_DRAM_ADDR=score,
                softmax_enable=True,
                C_DRAM_ADDR=self.LM_BIAS,
                bias_mode="full_matrix",
                gpr_M_reg=batch_reg,
                gpr_K_reg=head_dim_reg,
                gpr_N_reg=aligned_kv_reg,
            ) or 0
            pv_flops = ue.matmat_mul_core(
                M=G, K=aligned_kv, N=AHD,
                A_DRAM_ADDR=score,
                B_DRAM_ADDR=v_t,
                OUTPUT_DRAM_ADDR=self.LM_ATTN_HM + plane_off,
                gpr_M_reg=batch_reg,
                gpr_K_reg=aligned_kv_reg,
                gpr_N_reg=head_dim_reg,
            ) or 0
        finally:
            ue.release_isa_reg()  # head_dim_reg
            ue.release_isa_reg()  # batch_reg
        scaled = score_flops + pv_flops
        return fixed + scaled, scaled

    def _emit_one_round_group_attention_round(
        self, dec_sched, assignments, *, aligned_kv: int,
        aligned_kv_reg: int, k_base: int, v_base: int,
    ) -> tuple[int, int]:
        """Emit one complete eight-participant group-attention rendezvous."""
        if len(assignments) != dec_sched.num_engines:
            raise ValueError(
                f"attention assignment count {len(assignments)} does not match "
                f"{dec_sched.num_engines} engines"
            )
        total_flops = 0
        scaled_flops = 0
        dec_sched.release()
        total, scaled = self._emit_one_round_group_attention(
            self,
            group=assignments[0],
            aligned_kv=aligned_kv,
            aligned_kv_reg=aligned_kv_reg,
            k_base=k_base,
            v_base=v_base,
            scratch=self.LM_ATTN_SCRATCH_PER_ENGINE[0],
        )
        total_flops += total
        scaled_flops += scaled
        for e in dec_sched.worker_indices():
            dec_sched.begin_worker_round(e)
            group = assignments[e]
            if group is not None:
                total, scaled = self._emit_one_round_group_attention(
                    dec_sched.engines[e],
                    group=group,
                    aligned_kv=aligned_kv,
                    aligned_kv_reg=self._decode_attn_worker_regs[e - 1][
                        "aligned"
                    ],
                    k_base=k_base,
                    v_base=v_base,
                    scratch=self.LM_ATTN_SCRATCH_PER_ENGINE[e],
                )
                total_flops += total
                scaled_flops += scaled
            # All workers close the round, including deliberately idle ones.
            dec_sched.end_worker_round(e)
        dec_sched.join()
        return total_flops, scaled_flops

    def _dec_round(self, dec_sched, ops, master_emit, worker_extra=None) -> int:
        """One decode rendezvous: release, master's work, workers' rounds, join.

        ``ops`` is [(ShardedWeight, out_base, a_addr, bias, silu)] -- the ops
        every engine holding a shard re-emits against its own private block.
        ``master_emit`` emits engine 0's side: its own shard-0 blocks plus
        anything left unsharded, which then OVERLAPS the workers instead of
        serialising after them. ``worker_extra(ue, e)`` appends non-matmul work
        to each worker's round -- for an op that consumes only THIS engine's
        outputs of the ops above it, and so needs no barrier of its own (the
        master emits its own copy inside ``master_emit``).

        Every worker runs every round the master opens, even one where its
        engine holds no shard of a narrow weight -- a skipped rendezvous
        desynchronises the group permanently and the master waits forever.
        """
        if ops:
            dec_sched.release()
        flops = master_emit()
        if ops:
            for e in dec_sched.worker_indices():
                dec_sched.begin_worker_round(e)
                for sw, out_base, a_addr, bias, silu in ops:
                    if sw.shard_or_none(e) is not None:
                        self._emit_dec_shard(dec_sched.engines[e], sw, e,
                                             out_base, a_addr, bias, silu)
                if worker_extra is not None:
                    worker_extra(dec_sched.engines[e], e)
                dec_sched.end_worker_round(e)
            dec_sched.join()
        return flops

    def _decode_token(self) -> int:
        """The sampled token after one decode step.

        With a sharded head each engine argmaxes only its own column block, so
        the winner is whichever engine's candidate has the largest value --
        global_argmax reads those back and compares. Unsharded, the master's
        argmax register already holds the answer.
        """
        sw = getattr(self, "_decode_lm_shard", None)
        sched = self._multi_core_schedulers.get("decode") if sw is not None else None
        if sw is None or sched is None:
            return self.get_arg_max_index()
        return sched.global_argmax(sw, self.LOGITS)

    def _decode_stop_token_ids(self) -> set[int]:
        """Token IDs that terminate greedy generation for this model family."""
        return {151643, 151645, self._end_of_turn_token_id}

    def _kv_addr(self, cache_base: int, layer: int, kv_head: int) -> int:
        return cache_base + layer * self.KV_STRIDE_LAYER + kv_head * self.KV_STRIDE_HEAD

    def _emit_layer(self, li: int, M: int, *, decode: bool, m_reg: int,
                    aligned_kv: int, in_addr: int, out_addr: int,
                    rope_base: int, aligned_kv_reg: int = None,
                    ckpt=None, sched=None, gate_m_regs=None,
                    dec_sched=None, dec_shards=None) -> int:
        """One decoder layer. Shared by prefill (M=seq) and decode (M=1)."""
        d = self._lm_dims()
        H, AHD, KVH, QH, G, MLP = (d["H"], d["AHD"], d["KVH"], d["QH"],
                                   d["G"], d["MLP"])
        bpe = self.bytes_per_element
        la = self.lm_layer_addrs[li]
        flops = 0

        def mm(K, N, A, tag, OUT, *, quant=True, bias=None, silu=False):
            """One projection.

            DECODE USES THE M=1 GEMV KERNEL for every IF4 weight. At M=1 the
            general matmat kernel pays for a row tile it does not fill;
            quantized_matmat_core streams the packed weights straight through
            DOT_PRODUCT instead. It accepts IF4/IF8/TQ4 only, so projections
            configured as BF16 stay on matmat_mul_core. Prefill has real rows
            to fill and normally stays on the general kernel; compatible
            large-K models may opt a projection into the streaming path.
            """
            if decode and quant:
                return self.quantized_matmat_core(
                    M=M, K=K, N=N, A_DRAM_ADDR=A,
                    B_DRAM_ADDR=la[f"{tag}_data"], OUTPUT_DRAM_ADDR=OUT,
                    SCALE_DRAM_ADDR=la[f"{tag}_scale"], data_type=TYPE.IF4,
                    C_DRAM_ADDR=bias,
                    bias_mode="broadcast_N" if bias is not None else "broadcast_N",
                    silu_enable=silu) or 0
            if (
                decode
                and not quant
                and self._decode_projection_uses_static_bf16(tag)
            ):
                # Gemma4 E2B deliberately uses the legacy/static kernel for
                # dense BF16 M=1 projections.  Do not pass gpr_M_reg here: that
                # would dispatch to the unresolved dynamic dense-BF16 path.
                return self.matmat_mul_core_legacy(
                    M=M, K=K, N=N,
                    A_DRAM_ADDR=A,
                    B_DRAM_ADDR=la[f"{tag}_weight"],
                    OUTPUT_DRAM_ADDR=OUT,
                    C_DRAM_ADDR=bias,
                    bias_mode="broadcast_N",
                    silu_enable=silu,
                ) or 0
            kw = dict(M=M, K=K, N=N, A_DRAM_ADDR=A, OUTPUT_DRAM_ADDR=OUT,
                      silu_enable=silu, gpr_M_reg=m_reg)
            if bias is not None:
                kw.update(C_DRAM_ADDR=bias, bias_mode="broadcast_N")
            if quant:
                kw.update(B_DRAM_ADDR=la[f"{tag}_data"], is_B_quantized=True,
                          data_type=TYPE.IF4, SCALE_DRAM_ADDR=la[f"{tag}_scale"])
            else:
                kw.update(B_DRAM_ADDR=la[f"{tag}_weight"])
            return self.matmat_mul_core(**kw) or 0

        if sched is None:
            flops += self.rms_norm_core_dram(
                M=M, N=H, A_DRAM_ADDR=in_addr, OUTPUT_DRAM_ADDR=self.LM_PRE_NORM,
                GAMMA_DRAM_ADDR=la["ln1"], gpr_M_reg=m_reg) or 0

        if decode and dec_sched is not None:
            # DECODE: column shard, master/worker round. q, k and v all read
            # PRE_NORM and write disjoint buffers, so everything that shards
            # rides ONE rendezvous for the layer; an op that cannot split runs
            # full-width on the master inside that same round, overlapping the
            # workers rather than serialising after them.
            # At decode M=1, token-major [1, QH, AHD] and head-major
            # [QH, 1, AHD] have the exact same byte layout.  Land Q directly in
            # its attention input buffer so the serial token->head permute below
            # can be omitted without changing either values or ordering.
            projs = tuple(
                (tag, out, width, bias,
                 self._decode_projection_is_quantized(tag))
                for tag, out, width, bias in (
                    ("q", self.LM_Q_HM, QH * AHD, la["q_bias"]),
                    ("k", self.LM_K, KVH * AHD, la["k_bias"]),
                    ("v", self.LM_V, KVH * AHD, la["v_bias"]),
                ))
            round_ops = [(dec_shards[(tag, li)], out, self.LM_PRE_NORM, bias, False)
                         for tag, out, _, bias, _ in projs
                         if (tag, li) in dec_shards]

            def _master_qkv():
                f = 0
                for tag, out, N_op, bias, quant in projs:
                    sw = dec_shards.get((tag, li))
                    if sw is None:
                        f += mm(H, N_op, self.LM_PRE_NORM, tag, out,
                                quant=quant, bias=bias)
                    else:
                        # The master owns shard 0 and emits it inline; the rest
                        # are the workers' and their FLOPs come from
                        # worker_flops, which counts shards[1:] however many
                        # exist.
                        f += self._emit_dec_shard(self, sw, 0, out,
                                                  self.LM_PRE_NORM, bias)
                        f += dec_sched.worker_flops(sw)
                return f

            flops += self._dec_round(dec_sched, round_ops, _master_qkv)
        elif sched is None:
            q_out = self.LM_Q_HM if decode else self.LM_Q
            flops += mm(H, QH * AHD, self.LM_PRE_NORM, "q", q_out,
                        bias=la["q_bias"])
            flops += mm(H, KVH * AHD, self.LM_PRE_NORM, "k", self.LM_K, bias=la["k_bias"])
            flops += mm(
                H, KVH * AHD, self.LM_PRE_NORM, "v", self.LM_V,
                quant=(self._decode_projection_is_quantized("v") if decode
                       else self._lm_projection_is_quantized("v")),
                bias=la["v_bias"])
        else:
            # Row-shard the three projections over tokens: each engine reads its
            # own rows of LM_PRE_NORM and writes the matching rows of Q/K/V.
            # Weights and biases are shared read-only -- the bias is per-COLUMN
            # (broadcast_N), so it is not sliced.
            acc = [0]

            def _qkv(ctx, la=la, acc=acc, in_addr=in_addr):
                m = gate_m_regs[ctx.engine_idx]
                # norm1 folded in: it is row-independent and feeds the three
                # projections directly, so it rides the region already here
                # rather than paying its own entry/exit rendezvous.
                ctx.ue.generate_instruction_add_set(m, ctx.rows)
                acc[0] += ctx.ue.rms_norm_core_dram(
                    M=ctx.rows, N=H,
                    A_DRAM_ADDR=ctx.rows_addr(in_addr, H * bpe),
                    OUTPUT_DRAM_ADDR=ctx.rows_addr(self.LM_PRE_NORM, H * bpe),
                    GAMMA_DRAM_ADDR=la["ln1"], gpr_M_reg=m) or 0
                for tag, n_out, out, row in (
                        ("q", QH * AHD, self.LM_Q, QH * AHD * bpe),
                        ("k", KVH * AHD, self.LM_K, KVH * AHD * bpe),
                        ("v", KVH * AHD, self.LM_V, KVH * AHD * bpe)):
                    quant = self._lm_projection_is_quantized(tag)
                    ctx.ue.generate_instruction_add_set(m, ctx.rows)
                    kw = dict(M=ctx.rows, K=H, N=n_out,
                              A_DRAM_ADDR=ctx.rows_addr(self.LM_PRE_NORM, H * bpe),
                              OUTPUT_DRAM_ADDR=ctx.rows_addr(out, row),
                              C_DRAM_ADDR=la[f"{tag}_bias"],
                              bias_mode="broadcast_N", gpr_M_reg=m)
                    if quant:
                        kw.update(B_DRAM_ADDR=la[f"{tag}_data"], is_B_quantized=True,
                                  data_type=TYPE.IF4,
                                  SCALE_DRAM_ADDR=la[f"{tag}_scale"])
                    else:
                        kw.update(B_DRAM_ADDR=la[f"{tag}_weight"])
                    acc[0] += ctx.ue.matmat_mul_core(**kw) or 0

            sched.sharded_region(M, _qkv)
            flops += acc[0]
            # ENGINE 0's row register IS gf_seq_len, and the region left it
            # holding that engine's shard row count. Everything after here --
            # rope, o_proj, the eltwise ops -- reads it expecting the full
            # seq_len, so it has to be restored. (Vision never hits this because
            # its serial ops re-prime through _prime_M on every call.)
            self.generate_instruction_add_set(m_reg, M)

        ckpt = ckpt or (lambda name, f: None)
        rope_cos, rope_sin = rope_base, rope_base + AHD * bpe
        # ATTENTION BATCH MUST BE 64-ALIGNED. unified_attention_core consumes Q
        # in 64-row tiles, so a batch of 37 rounds DOWN to zero tiles and the
        # kernel writes nothing at all -- output stays whatever was there, which
        # read back as exact zeros. The head planes are therefore strided by the
        # aligned row count, not M, and rows M..head_rows-1 stay zero (they are
        # zeroed once by lm_reset_attention_state and never written), so the
        # extra tile computes finite garbage that nothing reads.
        # ONLY aligned_seq_len needs 64-alignment; batch does not. Head planes
        # are therefore packed at the live row count and attention runs exactly
        # M query rows -- no padding tile. (An earlier batch=37 run produced
        # zero output and I read that as an alignment rule; the real cause was
        # the under-sized scratch fixed above.)
        # Head-plane stride. Prefill uses the 64-ALIGNED length because
        # HeadShardContext.q_addr/out_addr index planes as head*seq_len*head_dim
        # -- a plane strided by M would put every head past 0 in the wrong
        # place. Decode keeps M=1 so its QH planes stay contiguous, which is
        # what makes the grouped decode call free.
        head_rows = M if decode else ((M + 63) // 64) * 64

        ckpt(f"L{li}:qkv_proj", flops)

        # Q token-major [M, QH, AHD] -> head-major [QH, M, AHD].  Decode's
        # M=1 projection already wrote LM_Q_HM because these layouts are then
        # byte-identical; prefill retains the real permutation and padded plane
        # stride it needs.
        if decode:
            if M != 1:
                raise ValueError(f"decode Q layout requires M=1, got M={M}")
        else:
            self.bf16_permute_dram_core(
                QH, M, AHD, self.LM_Q, self.LM_Q_HM,
                write_grouped=True, group_stride_rows=head_rows)
        # Prefill RoPE stays per-head: the planes are head_rows apart, so one
        # flattened call would run over their padding. Decode has no padding at
        # M=1 and may instead reuse its one table row across contiguous heads.
        grouped_decode_rope = decode and AHD >= 128 and AHD % 128 == 0
        if grouped_decode_rope:
            # Every decode head contains the same one position and the planes
            # are contiguous at M=1.  The static grouped primitive loads the
            # [cos|sin] row once and reuses it across all heads.  Passing no GPR
            # is intentional: it avoids emitting the general runtime-M/N RoPE
            # setup once per head in a position-agnostic decoder whose M is
            # nevertheless fixed at one.
            flops += self.rope_hf_core_dram_gqa(
                M=1, group_size=QH, N=AHD,
                input_dram_addr=self.LM_Q_HM,
                output_dram_addr=self.LM_Q_HM,
                cos_dram_addr=rope_cos, sin_dram_addr=rope_sin) or 0
        else:
            for qh in range(QH):
                flops += self.rope_hf_core_dram(
                    M=M, N=AHD,
                    input_dram_addr=self.LM_Q_HM + qh * head_rows * AHD * bpe,
                    output_dram_addr=self.LM_Q_HM + qh * head_rows * AHD * bpe,
                    cos_dram_addr=rope_cos, sin_dram_addr=rope_sin,
                    gpr_M_reg=m_reg) or 0
        self.generate_instruction_add_set(m_reg, M)

        # K/V straight into the cache at their head planes. In prefill the
        # permute writes rows 0..M-1 of each head and leaves the rest of the
        # C-row plane untouched, which is exactly what group_stride_rows is for.
        k_base = self._kv_addr(self.LM_K_CACHE, li, 0)
        v_base = self._kv_addr(self.LM_V_CACHE, li, 0)
        if decode:
            # Rotate K BEFORE storing: the cache row address is computed at
            # runtime from gf_seq_len, so it cannot be a RoPE operand.
            # All KV heads share the same single decode position. Grouped RoPE
            # broadcasts table row 0 across them; retain the general per-head
            # path for a future model whose head width cannot use static GQA.
            if grouped_decode_rope:
                flops += self.rope_hf_core_dram_gqa(
                    M=1, group_size=KVH, N=AHD,
                    input_dram_addr=self.LM_K,
                    output_dram_addr=self.LM_K,
                    cos_dram_addr=rope_cos, sin_dram_addr=rope_sin) or 0
            else:
                for h in range(KVH):
                    flops += self.rope_hf_core_dram(
                        M=1, N=AHD,
                        input_dram_addr=self.LM_K + h * AHD * bpe,
                        output_dram_addr=self.LM_K + h * AHD * bpe,
                        cos_dram_addr=rope_cos, sin_dram_addr=rope_sin,
                        gpr_M_reg=m_reg) or 0
            # One token: write at row gf_seq_len, computed at runtime.
            for h in range(KVH):
                for src, base, sram in ((self.LM_K, k_base, 0x10000),
                                        (self.LM_V, v_base, 0x20000)):
                    self.accelerator_memory_to_sram(src + h * AHD * bpe, sram, AHD)
                    self.generate_instruction_reg_mul_imm(
                        self.TMP_REG, self.gf_seq_len,
                        ue_35bit_addr_shifter(AHD * bpe))
                    self.generate_instruction_add_imm(
                        self.TMP_REG,
                        ue_35bit_addr_shifter(base + h * self.KV_STRIDE_HEAD),
                        self.TMP_REG)
                    self.sram_to_accelerator_memory(
                        sram_address=sram, accelerator_dram_address=0,
                        element_size=AHD, general_reg_src=self.TMP_REG)
        else:
            self.bf16_permute_dram_core(
                KVH, M, AHD, self.LM_K, k_base,
                write_grouped=True, group_stride_rows=self.MAX_CONTEXT_SIZE)
            self.bf16_permute_dram_core(
                KVH, M, AHD, self.LM_V, v_base,
                write_grouped=True, group_stride_rows=self.MAX_CONTEXT_SIZE)
            # Rotate K inside the cache, ONE HEAD AT A TIME. The heads' planes
            # are MAX_CONTEXT_SIZE rows apart (that is what reserves room for
            # decode to append), so they are NOT contiguous and a single
            # M=KVH*M call would walk off head 0's plane into unwritten rows.
            for h in range(KVH):
                flops += self.rope_hf_core_dram(
                    M=M, N=AHD,
                    input_dram_addr=k_base + h * self.KV_STRIDE_HEAD,
                    output_dram_addr=k_base + h * self.KV_STRIDE_HEAD,
                    cos_dram_addr=rope_cos, sin_dram_addr=rope_sin,
                    gpr_M_reg=m_reg) or 0

        ckpt(f"L{li}:rope+cache", flops)

        # GQA attention.
        #
        # DECODE GROUPS THE 8 Q HEADS THAT SHARE A KV HEAD INTO ONE CALL. At
        # M=1 the head planes are one row each, so LM_Q_HM is already
        # [QH, AHD] contiguous and group g's heads are adjacent -- the packing
        # is free, no gather. That cuts the per-layer calls from QH to KVH and,
        # with them, the V.T transpose and K/V plane reads that every head in a
        # group would otherwise repeat identically (8x redundant).
        #
        # PREFILL STAYS PER-HEAD. Grouping there means packing Q as
        # [seq*G, AHD], which makes the causal bias (seq*G)^2 -- 64x larger,
        # the blow-up the head-major layout exists to avoid.
        if decode:
            groups = [(kv, kv * G * AHD * bpe, G) for kv in range(KVH)]
        else:
            groups = [(qh // G, qh * head_rows * AHD * bpe, M) for qh in range(QH)]
        self.LM_BATCH_ROWS = G if decode else M
        if sched is not None and not decode:
            attn_acc = [0]

            def _attn(ctx, k_base=k_base, v_base=v_base, attn_acc=attn_acc):
                # Q/out come from the context accessors (address discipline);
                # K/V are computed here because the KV cache planes stride by
                # MAX_CONTEXT_SIZE, not by seq_len as ctx.kv_addr assumes.
                scratch = self.LM_ATTN_SCRATCH_PER_ENGINE[ctx.engine_idx]
                for qh in range(ctx.head_off, ctx.head_off + ctx.heads):
                    kv_h = qh // G
                    b_reg = ctx.ue.alloc_isa_reg()
                    ctx.ue.generate_instruction_add_set(b_reg, M)
                    a_reg = ctx.ue.alloc_isa_reg()
                    ctx.ue.generate_instruction_add_set(a_reg, aligned_kv)
                    f = ctx.ue.unified_attention_core(
                        batch=M, aligned_seq_len=aligned_kv, head_dim=AHD,
                        Q_DRAM_ADDR=ctx.q_addr(self.LM_Q_HM, qh),
                        K_DRAM_ADDR=k_base + kv_h * self.KV_STRIDE_HEAD,
                        V_DRAM_ADDR=v_base + kv_h * self.KV_STRIDE_HEAD,
                        BIAS_DRAM_ADDR=self.LM_BIAS,
                        OUTPUT_DRAM_ADDR=ctx.out_addr(self.LM_ATTN_HM, qh),
                        SCRATCH_DRAM_ADDR=scratch,
                        IDENTITY_DRAM_ADDR=self.LM_IDENTITY,
                        gpr_batch_reg=b_reg, gpr_aligned_seq_len_reg=a_reg)
                    ctx.ue.release_isa_reg()
                    ctx.ue.release_isa_reg()
                    attn_acc[0] += f if isinstance(f, (int, float)) else 0

            # mode="qheads": split all QH Q heads, not the KV groups -- there are
            # only KVH=2 groups, so mode="groups" would cap parallelism at 2.
            sched.head_sharded_region(QH, aligned_kv, AHD, _attn,
                                      gqa_ratio=G, mode="qheads")
            flops += attn_acc[0]
            self.generate_instruction_add_set(m_reg, M)   # restore gf_seq_len
            groups = []

        if (
            decode
            and dec_sched is not None
            and groups
            and self._decode_use_one_round_group_attention()
        ):
            # ONE ROUND, ONE COMPLETE GQA GROUP PER LEADING ENGINE.  This is
            # intentionally an opt-in model policy: it trades the shared path's
            # two fine-grained transpose/PV rounds for four independent full
            # attention pipelines.  Omni has four KV groups, so engines 0..3
            # each own one while 4..7 execute an idle handshake.  The latter is
            # part of the protocol, not optional work.
            if dec_sched.num_engines != 8:
                raise ValueError(
                    "one-round decode GQA currently requires eight engines, "
                    f"got {dec_sched.num_engines}"
                )
            expected_groups = [
                (kv, kv * G * AHD * bpe, G) for kv in range(KVH)
            ]
            if groups != expected_groups:
                raise AssertionError(
                    f"decode GQA layout changed: {groups!r} != "
                    f"{expected_groups!r}"
                )
            assignments = self._one_round_group_assignments(
                groups, dec_sched.num_engines
            )
            if len(self.LM_ATTN_SCRATCH_PER_ENGINE) != dec_sched.num_engines:
                raise RuntimeError(
                    "one-round decode GQA needs one scratch address per engine"
                )
            scratch_elems = AHD * aligned_kv + G * aligned_kv + G * AHD
            worker_capacity = int(getattr(
                self, "_lm_worker_attn_scratch_elements", 0
            ))
            if any(assignments[1:]) and worker_capacity < scratch_elems:
                raise MemoryError(
                    f"decode GQA needs {scratch_elems} worker scratch elements, "
                    f"allocated {worker_capacity}"
                )

            total, scaled = self._emit_one_round_group_attention_round(
                dec_sched,
                assignments,
                aligned_kv=aligned_kv,
                aligned_kv_reg=aligned_kv_reg,
                k_base=k_base,
                v_base=v_base,
            )
            flops += total
            self._emit_attn_flops += scaled
            groups = []

        if decode and dec_sched is not None and groups:
            # SHARDED DECODE ATTENTION. unified_attention_core is inlined so its
            # V transpose -- which is most of the step at a long context, and the
            # only part that grows with the KV length -- can be split across the
            # workers while engine 0 runs the parts that do not depend on it.
            #
            #   round 1   workers: V^T row slices, BOTH groups
            #             engine 0: Q scale + Q@K^T + bias + softmax, both groups
            #   round 2   all engines: P@V^T, column-sharded
            #
            # Q@K^T is NOT sharded: softmax is a row reduction over N, which is
            # exactly the axis an N-shard would split, and the hardware's fmax
            # context is per engine.
            #
            # Own scratch carve, not the core's: the core reserves score as
            # [aligned, aligned] because batch can reach aligned, but decode's
            # batch is G=8, so [batch, aligned] is 256x smaller and both groups
            # fit side by side inside LM_SCRATCH with room to spare.
            A = aligned_kv
            g_elems = AHD * A + G * A + G * AHD
            def _vt(g):  return self.LM_SCRATCH + g * g_elems * bpe
            def _sc(g):  return _vt(g) + AHD * A * bpe
            def _sq(g):  return _sc(g) + G * A * bpe
            attn_regs = self._decode_attn_worker_regs
            pv_by_engine = self._decode_pv_shards(KVH, AHD, dec_sched.num_engines)
            b_reg = self.alloc_isa_reg()
            hd_reg = self.alloc_isa_reg()
            self.generate_instruction_add_set(b_reg, G)
            self.generate_instruction_add_set(hd_reg, AHD)

            # ---- round 1: transpose (workers) || scale + scores (engine 0) ----
            dec_sched.release()
            for kv_h, plane_off, batch in groups:
                flops += self.eltwise_core_dram(
                    M=batch, N=AHD, dram_a=self.LM_Q_HM + plane_off, dram_b=None,
                    dram_out=_sq(kv_h), mode=UE_MODE.MUL_BROADCAST,
                    scalar=1.0 / math.sqrt(AHD), gpr_M_reg=b_reg) or 0
                f = self.matmat_mul_core(
                    M=batch, K=AHD, N=A,
                    A_DRAM_ADDR=_sq(kv_h),
                    B_DRAM_ADDR=k_base + kv_h * self.KV_STRIDE_HEAD,
                    OUTPUT_DRAM_ADDR=_sc(kv_h),
                    softmax_enable=True,
                    C_DRAM_ADDR=self.LM_BIAS, bias_mode="full_matrix",
                    gpr_M_reg=b_reg, gpr_K_reg=hd_reg,
                    gpr_N_reg=aligned_kv_reg) or 0
                flops += f
                self._emit_attn_flops += f
            for e in dec_sched.worker_indices():
                dec_sched.begin_worker_round(e)
                ue = dec_sched.engines[e]
                rg = attn_regs[e - 1]
                # Branch over the transpose when this worker drew no rows this
                # step -- M=0 hangs the core's row loop. The handshake still runs.
                jz_at = ue.capture_count
                ue.generate_instruction_jump_abs_jz(0, rg["rows"])   # patched below
                for kv_h, _plane_off, _batch in groups:
                    src = ue.alloc_isa_reg()
                    dst = ue.alloc_isa_reg()
                    ue.generate_instruction_reg_mul_imm(
                        src, rg["row_off"], ue_35bit_addr_shifter(AHD * bpe))
                    ue.generate_instruction_add_imm(
                        src_reg_idx=src,
                        immediate_value=ue_35bit_addr_shifter(
                            v_base + kv_h * self.KV_STRIDE_HEAD),
                        dst_reg_idx=src)
                    ue.generate_instruction_add_imm(
                        src_reg_idx=rg["out_off"],
                        immediate_value=ue_35bit_addr_shifter(_vt(kv_h)),
                        dst_reg_idx=dst)
                    ue.bf16_transpose_core(
                        M=A, N=AHD,
                        INPUT_DRAM_ADDR=v_base + kv_h * self.KV_STRIDE_HEAD,
                        OUTPUT_DRAM_ADDR=_vt(kv_h),
                        IDENTITY_DRAM_ADDR=self.LM_IDENTITY,
                        gpr_M_reg=rg["rows"], gpr_input_addr=src,
                        gpr_out_addr=dst,
                        gpr_out_row_stride_reg=rg["stride"])
                    ue.release_isa_reg()   # dst
                    ue.release_isa_reg()   # src
                ue._patch_jump_immediate(jz_at, ue_35bit_addr_shifter(
                    ue.get_program_dram_addr()
                    + ue.capture_count * INSTRUCTION_SIZE_BYTES))
                dec_sched.end_worker_round(e)
            dec_sched.join()

            # ---- round 2: P@V^T, column-sharded --------------------------
            # B is V^T, [AHD, aligned], so an output column block is a contiguous
            # ROW BLOCK of it. The OUTPUT block is strided though (batch > 1), so
            # each engine writes IN PLACE at row stride AHD instead of into a
            # private dense buffer -- no gather, and o_proj still reads one
            # [batch, AHD] plane per group.
            dec_sched.release()
            n_reg = self.alloc_isa_reg()
            self.generate_instruction_add_set(n_reg, 64)
            for g, n_off, cols in pv_by_engine[0]:
                f = self.matmat_mul_core(
                    M=G, K=A, N=cols,
                    A_DRAM_ADDR=_sc(g), B_DRAM_ADDR=_vt(g) + n_off * A * bpe,
                    OUTPUT_DRAM_ADDR=self.LM_ATTN_HM + g * G * AHD * bpe + n_off * bpe,
                    gpr_M_reg=b_reg, gpr_K_reg=aligned_kv_reg, gpr_N_reg=n_reg,
                    gpr_out_row_stride_reg=hd_reg) or 0
                flops += f
                self._emit_attn_flops += f
            self.release_isa_reg()   # n_reg
            for e in dec_sched.worker_indices():
                dec_sched.begin_worker_round(e)
                ue = dec_sched.engines[e]
                rg = attn_regs[e - 1]
                for g, n_off, cols in pv_by_engine[e]:
                    wb = ue.alloc_isa_reg()
                    wn = ue.alloc_isa_reg()
                    wm = ue.alloc_isa_reg()
                    ws = ue.alloc_isa_reg()
                    ue.generate_instruction_add_set(wn, cols)
                    ue.generate_instruction_add_set(wm, G)
                    ue.generate_instruction_add_set(ws, AHD)
                    # V^T row block = _vt(g) + n_off * aligned * bpe. Linear in
                    # the RUNTIME KV length, so it is derived from the primed
                    # register, never baked off the MAX_CONTEXT_SIZE bound.
                    ue.generate_instruction_reg_mul_imm(
                        wb, rg["aligned"], ue_35bit_addr_shifter(n_off * bpe))
                    ue.generate_instruction_add_imm(
                        src_reg_idx=wb,
                        immediate_value=ue_35bit_addr_shifter(_vt(g)),
                        dst_reg_idx=wb)
                    f = ue.matmat_mul_core(
                        M=G, K=A, N=cols,
                        A_DRAM_ADDR=_sc(g), B_DRAM_ADDR=_vt(g),
                        OUTPUT_DRAM_ADDR=(self.LM_ATTN_HM + g * G * AHD * bpe
                                          + n_off * bpe),
                        gpr_M_reg=wm, gpr_K_reg=rg["aligned"], gpr_N_reg=wn,
                        gpr_b_addr=wb, gpr_out_row_stride_reg=ws) or 0
                    for _ in range(4):
                        ue.release_isa_reg()
                    flops += f
                    self._emit_attn_flops += f
                dec_sched.end_worker_round(e)
            dec_sched.join()
            self.release_isa_reg()   # hd_reg
            self.release_isa_reg()   # b_reg
            groups = []

        for kv_h, plane_off, batch in groups:
            batch_reg = self.alloc_isa_reg()
            self.generate_instruction_add_set(batch_reg, batch)
            # aligned_seq_len is a COMPILE-TIME bound -- it sizes the scratch
            # sub-buffers -- while the runtime KV length rides in a register.
            # Decode passes gf_aligned_seq_len, primed per step by the preamble;
            # prefill's length is fixed at compile time so it primes its own.
            if aligned_kv_reg is None:
                aligned_reg = self.alloc_isa_reg()
                self.generate_instruction_add_set(aligned_reg, aligned_kv)
            else:
                aligned_reg = aligned_kv_reg
            f = self.unified_attention_core(
                batch=batch, aligned_seq_len=aligned_kv, head_dim=AHD,
                Q_DRAM_ADDR=self.LM_Q_HM + plane_off,
                K_DRAM_ADDR=k_base + kv_h * self.KV_STRIDE_HEAD,
                V_DRAM_ADDR=v_base + kv_h * self.KV_STRIDE_HEAD,
                BIAS_DRAM_ADDR=self.LM_BIAS,
                OUTPUT_DRAM_ADDR=self.LM_ATTN_HM + plane_off,
                SCRATCH_DRAM_ADDR=self.LM_SCRATCH,
                IDENTITY_DRAM_ADDR=self.LM_IDENTITY,
                gpr_batch_reg=batch_reg, gpr_aligned_seq_len_reg=aligned_reg)
            self.release_isa_reg()
            if aligned_kv_reg is None:
                self.release_isa_reg()
            f = f if isinstance(f, (int, float)) else 0
            flops += f
            # Billed separately: the core returns FLOPs for the COMPILE-TIME
            # aligned_seq_len, which for decode is the MAX_CONTEXT_SIZE bound
            # that sizes the scratch -- not the live KV length the step runs.
            # Counting it as-is reported 45 GFLOP/token and a 194%-of-peak
            # throughput. run_decoder rescales it by the actual length.
            self._emit_attn_flops += f

        ckpt(f"L{li}:attention", flops)

        # At decode M=1, [QH, 1, AHD] and [1, QH, AHD] are byte-identical.
        # Feed O directly from the head-major attention buffer and retain the
        # actual head->token permutation only for prefill.
        attn_result_addr = self.LM_ATTN_HM if decode else self.LM_ATTN_RESULT
        if not decode:
            self.bf16_permute_dram_core(
                QH, M, AHD, self.LM_ATTN_HM, self.LM_ATTN_RESULT,
                write_grouped=False, group_stride_rows=head_rows)
        ckpt(f"L{li}:attn_permute", flops)
        o_sw = dec_shards.get(("o", li)) if (decode and dec_shards) else None
        if o_sw is not None:
            # SECOND rendezvous of the layer: o_proj reads the attention result,
            # so it cannot ride the qkv round. bf16 and N=H=2048, i.e. 32 blocks
            # of 64 -- an even 4 blocks per engine at 8 cores. No bias.
            def _master_o(o_sw=o_sw):
                return (self._emit_dec_shard(self, o_sw, 0, self.LM_ATTN_PROJ,
                                             attn_result_addr)
                        + dec_sched.worker_flops(o_sw, M=1))

            flops += self._dec_round(
                dec_sched,
                [(o_sw, self.LM_ATTN_PROJ, attn_result_addr, None, False)],
                _master_o)
        elif sched is None:
            flops += mm(
                QH * AHD, H, attn_result_addr, "o", self.LM_ATTN_PROJ,
                quant=(self._decode_projection_is_quantized("o") if decode
                       else self._lm_projection_is_quantized("o")))
        else:
            # Same row shard as qkv: each engine takes its own tokens of the
            # attention result and writes the matching rows of the projection.
            # The storage policy is per-model; the 3B build keeps o_proj BF16,
            # while memory-constrained larger decoders opt it into IF4.
            o_acc = [0]

            def _o(ctx, la=la, o_acc=o_acc):
                m = gate_m_regs[ctx.engine_idx]
                ctx.ue.generate_instruction_add_set(m, ctx.rows)
                kw = dict(
                    M=ctx.rows, K=QH * AHD, N=H,
                    A_DRAM_ADDR=ctx.rows_addr(self.LM_ATTN_RESULT, QH * AHD * bpe),
                    OUTPUT_DRAM_ADDR=ctx.rows_addr(self.LM_ATTN_PROJ, H * bpe),
                    gpr_M_reg=m)
                if self._lm_projection_is_quantized("o"):
                    kw.update(B_DRAM_ADDR=la["o_data"], is_B_quantized=True,
                              data_type=TYPE.IF4,
                              SCALE_DRAM_ADDR=la["o_scale"])
                else:
                    kw.update(B_DRAM_ADDR=la["o_weight"])
                o_acc[0] += ctx.ue.matmat_mul_core(**kw) or 0

            sched.sharded_region(M, _o)
            flops += o_acc[0]
            self.generate_instruction_add_set(m_reg, M)   # restore gf_seq_len
        ckpt(f"L{li}:o_proj", flops)
        mlp_sw = ({t: dec_shards.get((t, li)) for t in ("gate", "up", "down")}
                  if (decode and dec_shards) else {})
        folded_resid2 = False
        if any(mlp_sw.values()):
            # The residual add, norm2 and the gate*up product stay on the
            # master: they are elementwise over a single row at decode, cheap
            # next to the projections, and each sits BETWEEN two rounds where
            # the workers are parked anyway.
            flops += self.eltwise_core_dram(
                M=M, N=H, dram_a=in_addr, dram_b=self.LM_ATTN_PROJ,
                dram_out=self.LM_RESIDUAL, mode=UE_MODE.ELTWISE_ADD,
                gpr_M_reg=m_reg) or 0
            flops += self.rms_norm_core_dram(
                M=M, N=H, A_DRAM_ADDR=self.LM_RESIDUAL,
                OUTPUT_DRAM_ADDR=self.LM_MLP_NORM, GAMMA_DRAM_ADDR=la["ln2"],
                gpr_M_reg=m_reg) or 0
            # Its own phase: resid+norm2 are master-serial, so folding them in
            # with the projections would report a diluted MLP speedup.
            ckpt(f"L{li}:mlp_norm", flops)

            # gate and up read the SAME post-norm row and write disjoint
            # buffers, so one rendezvous covers both.
            gu = (("gate", self.LM_MLP_GATE, True), ("up", self.LM_MLP_UP, False))
            gu_ops = [(mlp_sw[t], out, self.LM_MLP_NORM, None, silu)
                      for t, out, silu in gu if mlp_sw[t] is not None]

            # The SwiGLU product folds INTO that round rather than following
            # it: engine e produced gate[e] and up[e] over the SAME columns, so
            # multiplying them over those columns reads nothing another engine
            # wrote. It needs no barrier -- and it is the widest of the layer's
            # elementwise ops (N=11008), so leaving it on the master was
            # capping the MLP speedup.
            fold_mul = (mlp_sw["gate"] is not None and mlp_sw["up"] is not None
                        and len(mlp_sw["gate"].shards) == len(mlp_sw["up"].shards))

            def _mul_slice(ue, e):
                sh = mlp_sw["gate"].shard(e)
                off = sh.col_offset * bpe
                return ue.eltwise_core_dram(
                    M=1, N=sh.cols,
                    dram_a=self.LM_MLP_GATE + off, dram_b=self.LM_MLP_UP + off,
                    dram_out=self.LM_MLP_MULT + off,
                    mode=UE_MODE.ELTWISE_MUL) or 0

            def _master_gu():
                f = 0
                for t, out, silu in gu:
                    sw = mlp_sw[t]
                    if sw is None:
                        f += mm(H, MLP, self.LM_MLP_NORM, t, out, silu=silu)
                    else:
                        f += self._emit_dec_shard(self, sw, 0, out,
                                                  self.LM_MLP_NORM, None, silu)
                        f += dec_sched.worker_flops(sw)
                if fold_mul:
                    f += _mul_slice(self, 0)
                return f

            flops += self._dec_round(dec_sched, gu_ops, _master_gu,
                                     worker_extra=_mul_slice if fold_mul else None)
            if not fold_mul:
                flops += self.eltwise_core_dram(
                    M=M, N=MLP, dram_a=self.LM_MLP_GATE, dram_b=self.LM_MLP_UP,
                    dram_out=self.LM_MLP_MULT, mode=UE_MODE.ELTWISE_MUL,
                    gpr_M_reg=m_reg) or 0

            dn = mlp_sw["down"]
            # The layer-output residual folds into the down round for the same
            # reason the SwiGLU product folded into the gate/up one: down is
            # column-sharded over N=H, so engine e adds into exactly the slice
            # it just wrote.
            folded_resid2 = dn is not None

            def _resid2_slice(ue, e, dn=dn, out_addr=out_addr):
                sh = dn.shard(e)
                off = sh.col_offset * bpe
                return ue.eltwise_core_dram(
                    M=1, N=sh.cols,
                    dram_a=self.LM_RESIDUAL + off, dram_b=self.LM_MLP_DOWN + off,
                    dram_out=out_addr + off, mode=UE_MODE.ELTWISE_ADD) or 0

            def _master_down(dn=dn):
                if dn is None:
                    return mm(MLP, H, self.LM_MLP_MULT, "down", self.LM_MLP_DOWN)
                return (self._emit_dec_shard(self, dn, 0, self.LM_MLP_DOWN,
                                             self.LM_MLP_MULT)
                        + dec_sched.worker_flops(dn)
                        + _resid2_slice(self, 0))

            ckpt(f"L{li}:mlp_gate_up", flops)
            flops += self._dec_round(
                dec_sched,
                [(dn, self.LM_MLP_DOWN, self.LM_MLP_MULT, None, False)] if dn else [],
                _master_down,
                worker_extra=_resid2_slice if folded_resid2 else None)
        elif sched is None:
            flops += self.eltwise_core_dram(
                M=M, N=H, dram_a=in_addr, dram_b=self.LM_ATTN_PROJ,
                dram_out=self.LM_RESIDUAL, mode=UE_MODE.ELTWISE_ADD,
                gpr_M_reg=m_reg) or 0
            flops += self.rms_norm_core_dram(
                M=M, N=H, A_DRAM_ADDR=self.LM_RESIDUAL,
                OUTPUT_DRAM_ADDR=self.LM_MLP_NORM, GAMMA_DRAM_ADDR=la["ln2"],
                gpr_M_reg=m_reg) or 0
            flops += mm(H, MLP, self.LM_MLP_NORM, "gate", self.LM_MLP_GATE, silu=True)
            flops += mm(H, MLP, self.LM_MLP_NORM, "up", self.LM_MLP_UP)
            flops += self.eltwise_core_dram(
                M=M, N=MLP, dram_a=self.LM_MLP_GATE, dram_b=self.LM_MLP_UP,
                dram_out=self.LM_MLP_MULT, mode=UE_MODE.ELTWISE_MUL,
                gpr_M_reg=m_reg) or 0
            if self._prefill_mlp_k_lanes() > 1:
                # The lane split lives in the sharded path only; this
                # single-engine chain would read the repacked down_proj image
                # as though it were still full width.
                raise RuntimeError(
                    "the prefill MLP K-lane split requires the row-sharded "
                    "path; no scheduler was supplied")
            flops += mm(MLP, H, self.LM_MLP_MULT, "down", self.LM_MLP_DOWN)
        else:
            # The SwiGLU MLP has no biases, so only the activations are sliced.
            # Four-phase images keep the whole row-independent chain in one
            # region. A host-segmented legacy image places a HALT between its
            # kernels: besides matching that image's one-rendezvous-per-launch
            # contract, it prevents a very long resumed queue from spanning
            # several independent PBI loop nests.
            mlp_acc = [0]
            lanes, LANE, lane_plane = self._prefill_mlp_lane_geometry()

            def _mlp_step(ctx, step, la=la, mlp_acc=mlp_acc,
                          in_addr=in_addr, out_addr=out_addr):
                m = gate_m_regs[ctx.engine_idx]
                h_row, mlp_row = H * bpe, LANE * bpe
                lane = step[1] if isinstance(step, tuple) else 0
                step = step[0] if isinstance(step, tuple) else step
                ctx.ue.generate_instruction_add_set(m, ctx.rows)
                if step == "residual1":
                    mlp_acc[0] += ctx.ue.eltwise_core_dram(
                        M=ctx.rows, N=H,
                        dram_a=ctx.rows_addr(in_addr, h_row),
                        dram_b=ctx.rows_addr(self.LM_ATTN_PROJ, h_row),
                        dram_out=ctx.rows_addr(self.LM_RESIDUAL, h_row),
                        mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=m) or 0
                elif step == "norm":
                    mlp_acc[0] += ctx.ue.rms_norm_core_dram(
                        M=ctx.rows, N=H,
                        A_DRAM_ADDR=ctx.rows_addr(self.LM_RESIDUAL, h_row),
                        OUTPUT_DRAM_ADDR=ctx.rows_addr(self.LM_MLP_NORM, h_row),
                        GAMMA_DRAM_ADDR=la["ln2"], gpr_M_reg=m) or 0
                elif step in ("gate", "up"):
                    out = (self.LM_MLP_GATE if step == "gate"
                           else self.LM_MLP_UP) + lane * lane_plane
                    a = ctx.rows_addr(self.LM_MLP_NORM, h_row)
                    # A K-lane of down is an N-slice of gate/up, and N slices
                    # the (N, K) weight by whole rows -- contiguous, so the
                    # lane weight is pure address arithmetic on the unsliced
                    # blob (K/2 data bytes and K/64 bf16 scales per row).
                    mlp_acc[0] += ctx.ue.matmat_mul_core(
                        M=ctx.rows, K=H, N=LANE, A_DRAM_ADDR=a,
                        B_DRAM_ADDR=la[f"{step}_data"] + lane * LANE * (H // 2),
                        is_B_quantized=True,
                        data_type=TYPE.IF4,
                        SCALE_DRAM_ADDR=(la[f"{step}_scale"]
                                         + lane * LANE * (H // UE_VECTOR_SIZE) * bpe),
                        OUTPUT_DRAM_ADDR=ctx.rows_addr(out, mlp_row),
                        silu_enable=(step == "gate"), gpr_M_reg=m) or 0
                elif step == "multiply":
                    off = lane * lane_plane
                    mlp_acc[0] += ctx.ue.eltwise_core_dram(
                        M=ctx.rows, N=LANE,
                        dram_a=ctx.rows_addr(self.LM_MLP_GATE + off, mlp_row),
                        dram_b=ctx.rows_addr(self.LM_MLP_UP + off, mlp_row),
                        dram_out=ctx.rows_addr(self.LM_MLP_MULT + off, mlp_row),
                        mode=UE_MODE.ELTWISE_MUL, gpr_M_reg=m) or 0
                elif step == "down":
                    # Lane 0 writes the accumulator; every later lane writes a
                    # partial that is summed in.  An in-place full_matrix-bias
                    # accumulate would save the adds, but the adds are [rows, H]
                    # against a [rows, LANE] x [LANE, H] matmul -- under 1% --
                    # and this needs no aliasing assumption about C and OUT.
                    first = (lane == 0)
                    down_data, down_scale = self._prefill_down_lane(
                        la, lane, lanes)
                    dst = (self.LM_MLP_DOWN if first
                           else self.LM_MLP_DOWN_PART)
                    down_kw = dict(
                        M=ctx.rows, K=LANE, N=H,
                        A_DRAM_ADDR=ctx.rows_addr(
                            self.LM_MLP_MULT + lane * lane_plane, mlp_row),
                        B_DRAM_ADDR=down_data,
                        data_type=TYPE.IF4, SCALE_DRAM_ADDR=down_scale,
                        OUTPUT_DRAM_ADDR=ctx.rows_addr(dst, h_row),
                    )
                    if self._prefill_use_streaming_quantized_projection("down"):
                        # Keep M in the engine-local row-count register. This
                        # is Gemma4 E2B's compact streaming-prefill form: one
                        # captured loop serves text and multimodal row counts.
                        down_kw["gpr_M_reg"] = m
                        mlp_acc[0] += (
                            ctx.ue.quantized_matmat_core(**down_kw) or 0)
                    else:
                        down_kw.update(is_B_quantized=True, gpr_M_reg=m)
                        mlp_acc[0] += ctx.ue.matmat_mul_core(**down_kw) or 0
                    if not first:
                        mlp_acc[0] += ctx.ue.eltwise_core_dram(
                            M=ctx.rows, N=H,
                            dram_a=ctx.rows_addr(self.LM_MLP_DOWN, h_row),
                            dram_b=ctx.rows_addr(self.LM_MLP_DOWN_PART, h_row),
                            dram_out=ctx.rows_addr(self.LM_MLP_DOWN, h_row),
                            mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=m) or 0
                elif step == "residual2":
                    mlp_acc[0] += ctx.ue.eltwise_core_dram(
                        M=ctx.rows, N=H,
                        dram_a=ctx.rows_addr(self.LM_RESIDUAL, h_row),
                        dram_b=ctx.rows_addr(self.LM_MLP_DOWN, h_row),
                        dram_out=ctx.rows_addr(out_addr, h_row),
                        mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=m) or 0
                else:
                    raise AssertionError(f"unknown prefill MLP step {step!r}")

            # One lane's gate/up/multiply/down runs back to back so the lane's
            # [rows, LANE] intermediate is consumed while it is still the most
            # recently written thing in DRAM. At lanes == 1 this is the
            # historical step list, tuple wrappers aside.
            if self._prefill_mlp_tp_engines():
                flops += self._emit_prefill_mlp_tp(
                    sched, la, M, in_addr, out_addr, gate_m_regs)
                self.generate_instruction_add_set(m_reg, M)
                ckpt(f"L{li}:mlp_proj", flops)
                return flops
            mlp_steps = ["residual1", "norm"]
            for _lane in range(lanes):
                mlp_steps += [("gate", _lane), ("up", _lane),
                              ("multiply", _lane), ("down", _lane)]
            mlp_steps.append("residual2")
            mlp_steps = tuple(mlp_steps)
            if sched.host_segmented:
                for step in mlp_steps:
                    sched.sharded_region(
                        M, lambda ctx, step=step: _mlp_step(ctx, step))
            else:
                def _mlp(ctx):
                    for step in mlp_steps:
                        _mlp_step(ctx, step)

                sched.sharded_region(M, _mlp)
            flops += mlp_acc[0]
            self.generate_instruction_add_set(m_reg, M)   # restore gf_seq_len
        if sched is None and not folded_resid2:
            flops += self.eltwise_core_dram(
                M=M, N=H, dram_a=self.LM_RESIDUAL, dram_b=self.LM_MLP_DOWN,
                dram_out=out_addr, mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=m_reg) or 0
        ckpt(f"L{li}:mlp_proj", flops)
        return flops

    def _make_ckpt(self, profile: bool, base_ref, store):
        """Phase-boundary checkpoint emitter shared by prefill and decode.

        Each call ends the current phase with a HALT and records where to
        resume, so the run driver can time phases separately.

        THE RUNNING FLOP TOTAL IS PASSED IN, not read from an outer variable.
        _emit_layer accumulates into a local and only publishes it on return, so
        a checkpoint firing mid-layer would otherwise be billed the PREVIOUS
        layer's total -- which reported norm+qkv at 1348% of peak.
        ``base_ref[0]`` carries the total from layers already emitted.
        """
        last = [0]

        def ckpt(name: str, flops_in_layer: float = 0.0) -> None:
            if not profile:
                return
            self.generate_instruction_halt()
            resume = (self.get_program_dram_addr()
                      + self.capture_count * INSTRUCTION_SIZE_BYTES)
            total = base_ref[0] + flops_in_layer
            store.append([name, resume, int(total - last[0])])
            last[0] = int(total)

        return ckpt

    def _run_checkpointed(self, start_addr: int, checkpoints, timeout_s: float):
        """Drive a profile-compiled program segment by segment.

        Every phase ends in a HALT, so the program only advances when restarted
        at the recorded resume address. It still computes the full result --
        the segments tile the whole program.
        """
        results = []
        self.start_execute_from_dram(start_addr)
        for name, resume, ph_flops in checkpoints:
            self._wait_lm_queue(self, timeout_s, f"profile phase {name}")
            results.append((name, self.report_latency_in_us() / 1e3, ph_flops))
            self.start_execute_from_dram(resume)
        self._wait_lm_queue(self, timeout_s, "profile tail")
        return results

    @staticmethod
    def _wait_lm_queue(engine, timeout_s: float, what: str,
                       poll_interval_s: float | None = None) -> None:
        if not math.isfinite(float(timeout_s)) or timeout_s <= 0:
            raise ValueError(f"timeout_s must be finite and positive, got {timeout_s!r}")
        if poll_interval_s is None:
            engine.wait_queue(float(timeout_s))
        else:
            if (
                not math.isfinite(float(poll_interval_s))
                or poll_interval_s <= 0
            ):
                raise ValueError(
                    "poll_interval_s must be finite and positive, got "
                    f"{poll_interval_s!r}"
                )
            engine.wait_queue(
                timeout_seconds=float(timeout_s),
                poll_interval_seconds=float(poll_interval_s),
            )
        if engine.is_queue_busy():
            raise TimeoutError(f"{what} is still busy after {timeout_s:.1f}s")

    def _run_lm_compile_transaction(self, stage: str, compiler):
        """Run a capture compiler without leaking partial state on failure."""
        previous_silent = self._set_silent(False)
        self._set_silent(previous_silent)
        reg_counter_before = self._isa_reg_counter
        inst_ptr_counter_before = self._inst_ptr_counter
        program_cursor_before = self.get_program_dram_addr()
        scheduler_before = getattr(self, "_multi_core_schedulers", {}).get(stage)
        worker_cursors_before = (
            [worker.get_program_dram_addr() for worker in scheduler_before.workers]
            if scheduler_before is not None else None
        )
        try:
            return compiler()
        except Exception:
            scheduler = getattr(self, "_multi_core_schedulers", {}).get(stage)
            if scheduler is not None:
                scheduler.abort_program()
                if worker_cursors_before is None:
                    for worker in scheduler.workers:
                        worker.reset_program_dram_addr()
                else:
                    for worker, cursor in zip(
                        scheduler.workers, worker_cursors_before
                    ):
                        worker._next_program_dram_addr = cursor
            if getattr(self, "is_capture_on", False):
                self.stop_capture()
            self.clear_capture_buffer()
            self._isa_reg_counter = reg_counter_before
            self._inst_ptr_counter = inst_ptr_counter_before
            self._next_program_dram_addr = program_cursor_before
            raise
        finally:
            self._set_silent(previous_silent)

    def compile_prefill(self, seq_len: int, layer_size: int = None,
                        profile: bool = False) -> int:
        return self._run_lm_compile_transaction(
            "prefill",
            lambda: self._compile_prefill_impl(seq_len, layer_size, profile),
        )

    def _compile_prefill_impl(self, seq_len: int, layer_size: int = None,
                              profile: bool = False) -> int:
        """Emit a prefill program for exactly ``seq_len`` tokens.

        Compiled per prompt rather than made length-agnostic: the head-major
        permutes take compile-time shapes, and compiling 36 layers costs about a
        second. gemma4_e2b takes the same approach.
        """
        d = self._lm_dims()
        if seq_len > self.PREFILL_MAX_SEQ_LEN:
            raise ValueError(
                f"prompt is {seq_len} tokens, PREFILL_MAX_SEQ_LEN is "
                f"{self.PREFILL_MAX_SEQ_LEN}; tensors are sized for the latter")
        execution_rows = int(self._prefill_execution_rows(seq_len))
        if not seq_len <= execution_rows <= self.PREFILL_MAX_SEQ_LEN:
            raise ValueError(
                f"prefill execution rows must satisfy {seq_len} <= rows <= "
                f"{self.PREFILL_MAX_SEQ_LEN}, got {execution_rows}")
        aligned = ((execution_rows + 63) // 64) * 64
        t0 = time.perf_counter()
        self.reset_program_dram_addr()
        base = self.get_program_dram_addr()
        self.clear_inst_id()
        self.clear_capture_buffer()
        self.start_capture()
        prev = self._set_silent(True)
        self._emit_attn_flops = 0

        m_reg = self.gf_seq_len
        self.generate_instruction_add_set(m_reg, execution_rows)
        flops = 0
        flops_ref = [0]
        self._prefill_checkpoints = []
        ckpt = self._make_ckpt(profile, flops_ref, self._prefill_checkpoints)

        # Prefill gets its OWN scheduler but the SAME arena as vision, so the
        # two stages' worker programs never land on one another.
        sched = self._ensure_stage_scheduler("prefill")
        gate_m_regs = None
        if sched is not None:
            sched.begin_program()
            # Engine 0 reuses the model's row-count register; workers are plain
            # UnifiedEngines and allocate their own after begin_program() resets
            # their allocator.
            gate_m_regs = [self.gf_seq_len]
            gate_m_regs.extend(w.alloc_isa_reg() for w in sched.workers)
        nl = d["NL"] if layer_size is None else layer_size
        for li in range(nl):
            in_addr = self.LM_IO_A if li % 2 == 0 else self.LM_IO_B
            out_addr = self.LM_IO_B if li % 2 == 0 else self.LM_IO_A
            flops += self._emit_layer(
                li, execution_rows, decode=False, m_reg=m_reg, aligned_kv=aligned,
                in_addr=in_addr, out_addr=out_addr,
                rope_base=self.LM_ROPE_PRE, ckpt=ckpt,
                sched=sched, gate_m_regs=gate_m_regs)
            flops_ref[0] = flops
        self.generate_instruction_halt()
        worker_addrs = sched.finalize() if sched is not None else []
        if sched is not None:
            for w in reversed(sched.workers):
                w.release_isa_reg()
        self._set_silent(prev)
        self.stop_capture()

        blob = bytearray()
        for inst in self.capture_buffer:
            blob.extend(inst.get_bytes())
        self.clear_capture_buffer()

        self._prefill_workers = []
        if sched is not None:
            for idx, (w, addr) in enumerate(zip(sched.workers, worker_addrs), start=1):
                wb = bytearray()
                for inst in w.capture_buffer:
                    wb.extend(inst.get_bytes())
                self.mc_arena.check_isa_fits(idx, addr, len(wb))
                self._note_worker_isa(idx, "prefill", len(wb))
                self._prefill_workers.append((idx, w, addr, bytes(wb)))
        self._prefill_program = (base, bytes(blob))
        self._prefill_flops = int(flops)
        self._prefill_seq_len = seq_len
        self._prefill_execution_rows_compiled = execution_rows
        self._prefill_layers = nl
        # Buffer the last emitted layer wrote (ping-pong: even count -> IO_A).
        self.LM_PREFILL_OUT = self.LM_IO_A if nl % 2 == 0 else self.LM_IO_B
        self.allocate_program_dram(len(blob))
        if base + len(blob) > self.DRAM_END:
            raise MemoryError(
                f"LM prefill program overruns the ISA region: "
                f"0x{base + len(blob):X} > 0x{self.DRAM_END:X}. Prefill and "
                f"decoder share it; shorten the prompt or enlarge the region.")
        row_note = (
            f" ({execution_rows} FPGA execution rows)"
            if execution_rows != seq_len else ""
        )
        self._loud(f"  [LM] prefill compiled for {seq_len} tokens{row_note}: "
                   f"{len(blob) / 2**20:.2f} MiB at 0x{base:X}, "
                   f"{flops / 1e9:.1f} GFLOP, {time.perf_counter() - t0:.1f}s")
        return base

    def compile_decoder(self, layer_size: int = None,
                        profile: bool = False) -> int:
        return self._run_lm_compile_transaction(
            "decode",
            lambda: self._compile_decoder_impl(layer_size, profile),
        )

    def _compile_decoder_impl(self, layer_size: int = None,
                              profile: bool = False) -> int:
        """Emit ONE position-agnostic decode program.

        Everything that depends on the step is a register: gf_seq_len is the KV
        write row, gf_aligned_seq_len bounds attention, and the RoPE table is
        re-uploaded per step. So this compiles once and every token jumps to it.
        """
        d = self._lm_dims()
        t0 = time.perf_counter()
        base = self.get_program_dram_addr()
        self.clear_inst_id()
        self.clear_capture_buffer()
        self.start_capture()
        prev = self._set_silent(True)
        self._emit_attn_flops = 0

        m_reg = self.gf_one
        self.generate_instruction_add_set(m_reg, 1)
        flops = 0
        flops_ref = [0]
        self._decoder_checkpoints = []
        ckpt = self._make_ckpt(profile, flops_ref, self._decoder_checkpoints)

        # Decode gets its own scheduler (its own worker stream) but the SAME
        # arena, so its weight blocks and worker ISA cannot land on vision's or
        # prefill's.
        # Layer count first: the shard setup below allocates one weight block
        # per layer, so it needs nl.
        nl = d["NL"] if layer_size is None else layer_size
        dec_sched = self._ensure_stage_scheduler("decode")
        dec_shards = {}
        if dec_sched is not None:
            dec_sched.begin_program()
            dec_shards = self._ensure_decode_shards(dec_sched, nl)
            self._ensure_decode_attn_regs(dec_sched)
            # This runs only after every private shard has been copied out of
            # the shared params window. Larger models may now repurpose that
            # window for decode-only weights, following the same phase-sharing
            # discipline used by the media towers.
            self._prepare_decode_shared_weights(nl)
        for li in range(nl):
            in_addr = self.LM_IO_A if li % 2 == 0 else self.LM_IO_B
            out_addr = self.LM_IO_B if li % 2 == 0 else self.LM_IO_A
            flops += self._emit_layer(
                li, 1, decode=True, m_reg=m_reg,
                # aligned_seq_len is DYNAMIC: MAX_CONTEXT_SIZE is only the
                # compile-time bound that sizes the scratch, while
                # gf_aligned_seq_len carries the live KV length each step, so a
                # short context does not pay for a full one.
                aligned_kv=self.MAX_CONTEXT_SIZE,
                aligned_kv_reg=self.gf_aligned_seq_len, in_addr=in_addr,
                out_addr=out_addr, rope_base=self.LM_ROPE_DEC, ckpt=ckpt,
                dec_sched=dec_sched, dec_shards=dec_shards)
            flops_ref[0] = flops

        final_buf = self.LM_IO_A if nl % 2 == 0 else self.LM_IO_B
        self.LM_DECODE_OUT = final_buf
        flops += self.rms_norm_core_dram(
            M=1, N=d["H"], A_DRAM_ADDR=final_buf,
            OUTPUT_DRAM_ADDR=self.LM_OUT_NORM,
            GAMMA_DRAM_ADDR=self.final_norm_addr, gpr_M_reg=m_reg) or 0
        ckpt("final_norm", flops - flops_ref[0])

        head_sw = getattr(self, "_decode_lm_shard", None) if dec_sched else None
        if head_sw is not None:
            # SHARDED HEAD: writeback must be ENABLED, unlike the single-engine
            # path below. Each engine's argmax register holds a LOCAL index into
            # its own column block and the hardware exposes no max-VALUE
            # register, so the global winner is found by reading back the eight
            # candidates' values -- which requires them to be in DRAM.
            # That is 8 two-byte reads, not a 297 KiB readback.
            def _master_head():
                return (self._emit_dec_shard(self, head_sw, 0, self.LOGITS,
                                             self.LM_OUT_NORM, self.PENALTY_BIAS)
                        + dec_sched.worker_flops(head_sw))

            flops += self._dec_round(
                dec_sched,
                [(head_sw, self.LOGITS, self.LM_OUT_NORM, self.PENALTY_BIAS, False)],
                _master_head)
        else:
            # LM head with the penalty vector as its bias term: the HW argmax of
            # (logits + bias) is the answer, so write_back_disable keeps the
            # 151936 logits off the bus entirely. An all-zero bias is plain
            # greedy.
            flops += self.quantized_matmat_core(
                M=1, K=d["H"], N=d["VOCAB"], A_DRAM_ADDR=self.LM_OUT_NORM,
                B_DRAM_ADDR=self.lm_head_data, OUTPUT_DRAM_ADDR=self.LOGITS,
                SCALE_DRAM_ADDR=self.lm_head_scale, data_type=TYPE.IF4,
                C_DRAM_ADDR=self.PENALTY_BIAS, bias_mode="broadcast_N",
                write_back_disable=True) or 0
        ckpt("lm_head", flops - flops_ref[0])
        self.generate_instruction_add_inc(self.gf_seq_len)
        self.generate_instruction_halt()
        dec_worker_addrs = dec_sched.finalize() if dec_sched is not None else []
        self._set_silent(prev)
        self.stop_capture()

        blob = bytearray()
        for inst in self.capture_buffer:
            blob.extend(inst.get_bytes())
        self.clear_capture_buffer()
        self._decoder_workers = []
        if dec_sched is not None:
            for idx, (w, addr) in enumerate(zip(dec_sched.workers, dec_worker_addrs),
                                            start=1):
                wb = bytearray()
                for inst in w.capture_buffer:
                    wb.extend(inst.get_bytes())
                self.mc_arena.check_isa_fits(idx, addr, len(wb))
                self._note_worker_isa(idx, "decode", len(wb))
                self._decoder_workers.append((idx, w, addr, bytes(wb)))
        self._decoder_program = (base, bytes(blob))
        self._decoder_flops = int(flops)
        # Split so a step can be priced at its real KV length: everything except
        # attention is fixed per token, and attention scales linearly with
        # aligned_seq_len.
        self._decoder_flops_fixed = int(flops - self._emit_attn_flops)
        self._decoder_attn_per_aligned = (
            self._emit_attn_flops / self.MAX_CONTEXT_SIZE
            if self.MAX_CONTEXT_SIZE else 0.0)
        self.allocate_program_dram(len(blob))
        self._decoder_preamble = self.allocate_program_dram(64 * 8)
        if base + len(blob) > self.DRAM_END:
            raise MemoryError(
                f"LM decoder program overruns the ISA region: "
                f"0x{base + len(blob):X} > 0x{self.DRAM_END:X}. Prefill and "
                f"decoder share it; shorten the prompt or enlarge the region.")
        self._loud(f"  [LM] decoder compiled: {len(blob) / 2**20:.2f} MiB at "
                   f"0x{base:X}, {self._decoder_flops_fixed / 1e9:.2f} GFLOP/token "
                   f"+ attention scaled by context, "
                   f"{time.perf_counter() - t0:.1f}s")
        return base

    # ---- execution ---------------------------------------------------------

    def _upload(self, program) -> int:
        """Write a compiled program to its baked address and ADVANCE the cursor.

        The advance is not bookkeeping. Leaving the cursor on the program's own
        base means the next thing that emits a program there overwrites it --
        and at multi-core that is preclear_flags(), which drops a small
        flag-clearing program on every engine right before launch. The master
        then executes whatever is left and never halts, which is exactly how
        multi-core prefill hung while single-core was fine.
        """
        addr, blob = program
        self._next_program_dram_addr = addr
        written = self.dma_write(DMA_DEVICE_H2C, addr, blob, len(blob))
        if written != len(blob):
            raise IOError(
                f"LM master ISA DMA wrote {written} of {len(blob)} bytes")
        self.allocate_program_dram(len(blob))
        return addr

    def run_prefill(self, tokens, image_embeddings=None, positions=None,
                    profile: bool = False, audio_embeddings=None,
                    video_embeddings=None, modality_embeddings=None) -> None:
        """Embed a prompt, splice supplied modality rows, then run prefill.

        ``modality_embeddings`` may map ``image``/``audio``/``video`` to
        tensors and is merged with the explicit backwards-compatible keyword
        arguments.  Placeholder IDs come from the model config when present,
        so compatible Qwen multimodal decoders do not inherit this model's
        image-only assumptions.
        """
        # The K-lane repack rewrites the shared down_proj image in place, so
        # it must follow compile_decoder(), whose column shards copy that image
        # card -> host -> card, and precede any prefill read of a lane. Prefill
        # execution is the one point both hold; it is a no-op at lanes == 1 and
        # after the first call.
        self.repack_down_to_k_lanes()
        d = self._lm_dims()
        seq_len = len(tokens)
        compiled_seq_len = getattr(self, "_prefill_seq_len", None)
        if compiled_seq_len != seq_len:
            raise ValueError(
                f"prefill program was compiled for {compiled_seq_len} live "
                f"token(s), but run received {seq_len}")
        execution_rows = int(getattr(
            self, "_prefill_execution_rows_compiled", seq_len
        ))
        if not seq_len <= execution_rows <= self.PREFILL_MAX_SEQ_LEN:
            raise RuntimeError(
                f"invalid compiled prefill execution row count {execution_rows} "
                f"for {seq_len} live token(s)")
        aligned = ((execution_rows + 63) // 64) * 64

        device_embedding = self._device_embedding_enabled()
        emb = None if device_embedding else self.get_embedding_for_tokens(tokens)
        if device_embedding:
            # The model-specific hook emits IF4 row lookup + dequantization on
            # the accelerator and writes the resulting BF16 rows straight into
            # LM_IO_A.  The host handles token IDs and addresses only.
            self._load_device_embeddings(tokens, self.LM_IO_A)
            if execution_rows > seq_len:
                # Padding is data preparation, not learned inference. It must be
                # finite AND nonzero: this FPGA RMS reciprocal path cannot
                # normalize an all-zero row. Padding keys are masked from every
                # live query below, so the sentinel cannot affect live tokens.
                self.dma_to_accelerator_memory(
                    self.LM_IO_A + seq_len * d["H"] * self.bytes_per_element,
                    torch.ones(
                        (execution_rows - seq_len) * d["H"],
                        dtype=torch.bfloat16,
                    ),
                )
        elif execution_rows > seq_len:
            padded = torch.ones((execution_rows, d["H"]), dtype=emb.dtype)
            padded[:seq_len] = emb
            emb = padded
        supplied = dict(modality_embeddings or {})
        for name, value in (("image", image_embeddings),
                            ("audio", audio_embeddings),
                            ("video", video_embeddings)):
            if value is not None:
                if name in supplied:
                    raise ValueError(
                        f"{name} embeddings supplied both explicitly and in "
                        f"modality_embeddings")
                supplied[name] = value

        token_cfg = self._cfg.get("tokens", {})
        token_ids = {
            "image": int(token_cfg.get("image_token_id", 151655)),
            "audio": int(token_cfg.get("audio_token_id", 151646)),
            "video": int(token_cfg.get("video_token_id", 151656)),
        }
        for name, values in supplied.items():
            if name not in token_ids:
                raise ValueError(
                    f"unsupported modality {name!r}; expected one of "
                    f"{sorted(token_ids)}")
            if device_embedding:
                values = torch.as_tensor(values)
                if values.dtype != torch.bfloat16:
                    raise TypeError(
                        f"{name} device embeddings must already be BF16; got "
                        f"{values.dtype}. Host-side learned-data conversion is disabled."
                    )
            else:
                values = torch.as_tensor(values, dtype=emb.dtype)
            if values.ndim != 2 or values.shape[1] != d["H"]:
                raise ValueError(
                    f"{name} embeddings must have shape [tokens, {d['H']}], "
                    f"got {tuple(values.shape)}")
            slots = [i for i, token in enumerate(tokens)
                     if token == token_ids[name]]
            if len(slots) != values.shape[0]:
                raise ValueError(
                    f"{name} placeholder/embedding mismatch: prompt has "
                    f"{len(slots)} token(s) {token_ids[name]}, encoder returned "
                    f"{values.shape[0]} row(s)")
            if slots:
                if device_embedding:
                    # Encoder rows already came from accelerator execution.
                    # Copy contiguous placeholder runs back to their final LM
                    # addresses without evaluating learned arithmetic on host.
                    first = 0
                    while first < len(slots):
                        last = first + 1
                        while (
                            last < len(slots)
                            and slots[last] == slots[last - 1] + 1
                        ):
                            last += 1
                        rows = values[first:last].contiguous()
                        self.dma_to_accelerator_memory(
                            self.LM_IO_A + slots[first] * d["H"] * 2,
                            rows.flatten(),
                        )
                        first = last
                else:
                    emb[torch.tensor(slots, dtype=torch.long)] = values
            self._loud(
                f"  [LM] spliced {len(slots)} {name} embeddings at {slots[:4]}"
                f"{'...' if len(slots) > 4 else ''}")
        if not device_embedding:
            self.dma_to_accelerator_memory(self.LM_IO_A, emb.flatten())

        live_positions = torch.as_tensor(
            positions if positions is not None else torch.arange(seq_len)
        )
        if live_positions.ndim not in (1, 2) or live_positions.shape[0] != seq_len:
            raise ValueError(
                f"positions must have leading shape [{seq_len}], got "
                f"{tuple(live_positions.shape)}")
        if execution_rows > seq_len:
            pad_shape = (execution_rows - seq_len, *live_positions.shape[1:])
            live_positions = torch.cat(
                (live_positions,
                 torch.zeros(pad_shape, dtype=live_positions.dtype)),
                dim=0,
            )
        self.load_rope_for_positions(live_positions)
        # Causal mask over the aligned square; columns past the real prompt are
        # masked too, so the alignment padding cannot be attended to.
        bias = torch.full((aligned, aligned), float("-inf"), dtype=torch.bfloat16)
        bias.masked_fill_(torch.tril(torch.ones(aligned, aligned, dtype=torch.bool)), 0.0)
        bias[:, seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LM_BIAS, bias)

        addr = self._upload(self._prefill_program)
        sched = self._ensure_stage_scheduler("prefill")
        worker_addrs = []
        for idx, w, waddr, blob in getattr(self, "_prefill_workers", []):
            w._next_program_dram_addr = waddr
            written = w.dma_write(DMA_DEVICE_H2C, waddr, blob, len(blob))
            if written != len(blob):
                raise IOError(
                    f"prefill worker {idx} ISA DMA wrote {written} of "
                    f"{len(blob)} bytes")
            w.allocate_program_dram(len(blob))
            worker_addrs.append(waddr)
        if sched is not None and not sched.host_segmented:
            sched.preclear_flags()
        t0 = time.perf_counter()
        if profile:
            if sched is not None and sched.host_segmented:
                raise RuntimeError(
                    "profile checkpoints cannot be combined with the installed "
                    "FPGA build's host-segmented rendezvous")
            cps = getattr(self, "_prefill_checkpoints", [])
            if not cps:
                raise RuntimeError("profiled prefill needs compile_prefill(profile=True)")
            if sched is not None:
                sched.start_workers(worker_addrs)
            self._prefill_profile = self._run_checkpointed(addr, cps, 180.0)
            for idx, w in enumerate(
                sched.workers if sched is not None else [], start=1
            ):
                self._wait_lm_queue(w, 180.0, f"prefill worker {idx}")
            us = sum(r[1] for r in self._prefill_profile) * 1e3
        else:
            # Workers first: each parks on its first rendezvous until the master
            # enters the region.
            if sched is not None and sched.host_segmented:
                us = sched.run_host_segmented(
                    addr, worker_addrs, timeout_seconds=30.0)
            elif sched is not None:
                sched.start_workers(worker_addrs)
            if sched is None or not sched.host_segmented:
                self.start_execute_from_dram(addr)
                self._wait_lm_queue(self, 180.0, "prefill master")
                for idx, w in enumerate(
                    sched.workers if sched is not None else [], start=1
                ):
                    self._wait_lm_queue(w, 180.0, f"prefill worker {idx}")
                us = self.report_latency_in_us()
        # Expose the prompt length only after every engine has completed the
        # cache population. Failed uploads or launches leave host state at the
        # last known-complete sequence.
        self.seq_len = seq_len
        # Prefill's compile-time shapes ARE what runs -- it is compiled for this
        # exact seq_len -- so the cores' FLOP sum needs no rescaling, unlike
        # decode's. The guard is here anyway: >100% of peak is impossible and
        # means the count is billed at shapes the hardware did not execute.
        gflops = self._prefill_flops / (us * 1e-6) / 1e9 if us else 0.0
        peak = self.vis_peak_gflops()
        warn = (f"  [warn] {100 * gflops / peak:.0f}% of peak is impossible"
                if peak and gflops > peak * 1.005 else "")
        self._loud(f"  [LM] prefill {seq_len} tokens: {us / 1e6:.2f}s HW, "
                   f"{self._prefill_flops / 1e9:.1f} GFLOP, {gflops:.1f} GFLOPS"
                   f"{f' = {100 * gflops / peak:.0f}% of peak' if peak else ''} "
                   f"({time.perf_counter() - t0:.2f}s wall){warn}")
        self._latency_prefill_us = us
        self._prefill_gflops = gflops
        self._prefill_wall_s = time.perf_counter() - t0
        self._prefill_seq_len_run = seq_len

    def run_decode_step_profiled(self, token: int, program, checkpoints,
                                 workers=None, timeout_s: float = 60.0):
        """One profiled decode step at the CURRENT context length.

        Runs a real step -- it writes K/V at gf_seq_len and advances the
        position -- so calling it twice, once right after prefill and once after
        some tokens, gives the 1st-token and at-context breakdowns that bracket
        how attention grows while the projections stay fixed.
        """
        d = self._lm_dims()
        addr, _ = program
        self._upload(program)
        # The profiled decoder has its OWN worker images -- compile_decoder runs
        # twice under --profile and the second (plain) call overwrites
        # self._decoder_workers -- so they are passed in alongside the program.
        dec_sched = self._ensure_stage_scheduler("decode")
        worker_addrs = []
        for idx, w, waddr, wblob in (workers or []):
            w._next_program_dram_addr = waddr
            written = w.dma_write(DMA_DEVICE_H2C, waddr, wblob, len(wblob))
            if written != len(wblob):
                raise IOError(
                    f"profiled decode worker {idx} ISA DMA wrote {written} of "
                    f"{len(wblob)} bytes")
            w.allocate_program_dram(len(wblob))
            worker_addrs.append(waddr)
        if dec_sched is not None and not dec_sched.host_segmented:
            dec_sched.preclear_flags()
        step_pos = self.seq_len
        next_seq_len = step_pos + 1
        aligned = ((next_seq_len + 63) // 64) * 64

        if not self._device_embedding_enabled():
            self.dma_to_accelerator_memory(
                self.LM_IO_A, self.get_embedding_for_tokens([token]).flatten())
        pos = step_pos + getattr(self, "_rope_offset", 0)
        self.load_rope_for_positions(torch.tensor([[pos, pos, pos]]), decode=True)
        bias = torch.full((self.LM_BATCH_ROWS, aligned), float("-inf"),
                          dtype=torch.bfloat16)
        bias[:, :next_seq_len] = 0.0
        self.dma_to_accelerator_memory(self.LM_BIAS, bias)

        self.clear_inst_id()
        self.start_capture()
        if self._device_embedding_enabled():
            self._emit_device_decode_embedding(token, self.LM_IO_A)
        self.generate_instruction_add_set(self.gf_seq_len, step_pos)
        self.generate_instruction_add_set(self.gf_aligned_seq_len, aligned)
        self.generate_instruction_jump_abs(ue_35bit_addr_shifter(addr))
        self.stop_capture()
        try:
            written = self.write_captured_instructions_to_dram(
                self._decoder_preamble)
            expected = self.get_capture_instruction_size_bytes()
            if written != expected:
                raise IOError(
                    f"profiled decoder preamble DMA wrote {written} of "
                    f"{expected} bytes")
        finally:
            self.clear_capture_buffer()

        prev = self._set_silent(True)
        try:
            if dec_sched is not None and dec_sched.host_segmented:
                raise RuntimeError(
                    "profile checkpoints cannot be combined with the installed "
                    "FPGA build's host-segmented rendezvous")
            # Workers run their whole stream and block at each rendezvous; the
            # master's HALTs sit at phase boundaries OUTSIDE any round, so stopping
            # it between segments does not strand them.
            if dec_sched is not None:
                self._start_decode_workers(dec_sched, worker_addrs, aligned)
            results = self._run_checkpointed(
                self._decoder_preamble, checkpoints, timeout_s)
            for idx, w in enumerate(
                dec_sched.workers if dec_sched is not None else [], start=1
            ):
                self._wait_lm_queue(
                    w, timeout_s, f"profiled decode worker {idx}")
            # Commit the host position only after every participating engine
            # completed the step. Upload or launch failures leave it unchanged.
            self.seq_len = next_seq_len
            return results, self._decode_token(), aligned
        finally:
            self._set_silent(prev)

    def run_decoder(self, first_token: int,
                    max_new_tokens: int = 256) -> tuple[int, str]:
        """Run greedy decode and always restore process-wide output state."""
        previous_silent = self._set_silent(False)
        self._set_silent(previous_silent)
        try:
            return self._run_decoder_impl(first_token, max_new_tokens)
        except Exception:
            # The implementation installs a terminal scroll region only while
            # decoding. A DMA failure or queue timeout must not leave the
            # caller's terminal pinned after the exception propagates.
            if sys.stdout.isatty():
                import shutil
                rows = shutil.get_terminal_size().lines
                sys.stdout.write("\033[r")
                sys.stdout.write(f"\033[{rows};1H\033[2K")
                sys.stdout.flush()
            raise
        finally:
            self._set_silent(previous_silent)

    def _run_decoder_impl(self, first_token: int,
                          max_new_tokens: int = 256) -> tuple[int, str]:
        """Greedy decode until EOS, ``max_new_tokens``, or the context fills.

        The cap is NOT cosmetic. If anything upstream makes the argmax garbage
        the sampled tokens never hit a stop id, and an uncapped loop then runs
        thousands of FPGA steps -- which looks exactly like a hung board. Killing
        the host mid-step then leaves the engine executing with its queue busy,
        so the NEXT run looks hung too.
        """
        d = self._lm_dims()
        stop = self._decode_stop_token_ids()
        addr, _ = self._decoder_program
        self._upload(self._decoder_program)
        dec_sched = self._ensure_stage_scheduler("decode")
        dec_worker_addrs = []
        for idx, w, waddr, wblob in getattr(self, "_decoder_workers", []):
            w._next_program_dram_addr = waddr
            written = w.dma_write(DMA_DEVICE_H2C, waddr, wblob, len(wblob))
            if written != len(wblob):
                raise IOError(
                    f"decode worker {idx} ISA DMA wrote {written} of "
                    f"{len(wblob)} bytes")
            w.allocate_program_dram(len(wblob))
            dec_worker_addrs.append(waddr)
        if dec_sched is not None and not dec_sched.host_segmented:
            dec_sched.preclear_flags()

        token, out = first_token, []
        total_us = 0.0
        step_flops = 0.0
        self._decode_step_us = []
        t0 = time.perf_counter()

        # Worker preambles contain only runtime attention dimensions.  Those
        # dimensions change at a 64-token alignment boundary, not every token.
        # Keep this cache local to one decode run: while ``aligned`` is stable,
        # relaunch all seven workers from the exact preambles already written to
        # DRAM instead of rebuilding and DMA-writing seven identical images.
        # Every worker is still launched for every token and participates in
        # every rendezvous; only redundant host control traffic is removed.
        worker_preamble_aligned = None
        worker_preamble_entries = None

        # Live status bar: pin the bottom terminal row via an ANSI scroll region
        # so generated tokens stream above it while the counter refreshes in
        # place. Everything is on stdout -- tokens scroll inside rows 1..rows-1,
        # the status writes row `rows` with cursor save/restore -- so it never
        # clobbers the streamed text. TTY only; skipped when piped or redirected,
        # which is why the escape codes never reach a log file.
        import shutil
        start_seq = self.seq_len
        use_status = sys.stdout.isatty()

        def _status_setup():
            rows = shutil.get_terminal_size().lines
            sys.stdout.write(f"\033[1;{rows - 1}r")    # scroll region = rows 1..rows-1
            sys.stdout.write(f"\033[{rows - 1};1H")    # park cursor at its bottom
            sys.stdout.flush()

        def _status_update():
            rows = shutil.get_terminal_size().lines
            n_done = self.seq_len - start_seq
            elapsed = time.perf_counter() - t0
            rate = n_done / elapsed if elapsed > 0 else 0.0
            hw = (1e6 / (total_us / n_done)) if n_done and total_us else 0.0
            sys.stdout.write("\0337")                  # save cursor
            sys.stdout.write(f"\033[{rows};1H\033[2K")  # bottom row, clear it
            sys.stdout.write(
                f" decoding… {n_done} tokens  (pos {self.seq_len}/{self.MAX_CONTEXT_SIZE})"
                f"  {elapsed:.1f}s  {rate:.2f} tok/s  (HW {hw:.2f} tok/s)")
            sys.stdout.write("\0338")                  # restore cursor
            sys.stdout.flush()

        def _status_teardown():
            rows = shutil.get_terminal_size().lines
            sys.stdout.write("\033[r")                 # reset scroll region
            sys.stdout.write(f"\033[{rows};1H\033[2K")  # clear the status row
            sys.stdout.flush()

        if use_status:
            _status_setup()
        limit = self.MAX_CONTEXT_SIZE
        if max_new_tokens is not None:
            limit = min(limit, self.seq_len + max_new_tokens)
        prev_silent = self._set_silent(True)
        while self.seq_len < limit:
            step_pos = self.seq_len            # KV row this step writes
            next_seq_len = step_pos + 1
            aligned = ((next_seq_len + 63) // 64) * 64

            if not self._device_embedding_enabled():
                self.dma_to_accelerator_memory(
                    self.LM_IO_A, self.get_embedding_for_tokens([token]).flatten())
            # After an image the next text position is NOT step_pos: the image
            # occupied max(h, w) positions, not one per token, so mrope_delta
            # carries the gap. All three components advance together for text.
            pos = step_pos + getattr(self, "_rope_offset", 0)
            self.load_rope_for_positions(
                torch.tensor([[pos, pos, pos]]), decode=True)
            # Mask everything past the tokens actually written to the cache.
            # Cover EVERY row the kernel reads: batch is the 64-aligned tile,
            # not QH. Rows past the live query are unread, but an all -inf row
            # softmaxes to NaN, so give them the same mask as row 0.
            # ROW STRIDE IS THE RUNTIME aligned LENGTH, not the compile-time
            # bound. Every grouped query row needs its own mask row, and the
            # kernel indexes them by the live aligned_seq_len; laying them out
            # at MAX_CONTEXT_SIZE made row 0 correct and rows 1..G-1 read the
            # wrong slice -- which showed up as the first head of each KV group
            # being right and the other seven wrong.
            bias = torch.full((self.LM_BATCH_ROWS, aligned), float("-inf"),
                              dtype=torch.bfloat16)
            bias[:, :next_seq_len] = 0.0
            self.dma_to_accelerator_memory(self.LM_BIAS, bias)

            # Per-step preamble: prime the position registers, jump to the body.
            self.clear_inst_id()
            self.start_capture()
            if self._device_embedding_enabled():
                self._emit_device_decode_embedding(token, self.LM_IO_A)
            self.generate_instruction_add_set(self.gf_seq_len, step_pos)
            self.generate_instruction_add_set(self.gf_aligned_seq_len, aligned)
            self.generate_instruction_jump_abs(ue_35bit_addr_shifter(addr))
            self.stop_capture()
            try:
                written = self.write_captured_instructions_to_dram(
                    self._decoder_preamble)
                expected = self.get_capture_instruction_size_bytes()
                if written != expected:
                    raise IOError(
                        f"decoder preamble DMA wrote {written} of "
                        f"{expected} bytes")
            finally:
                self.clear_capture_buffer()

            # Workers are relaunched EVERY step: each decoder program ends with
            # its workers halted, so a step that did not start them would leave
            # the master waiting on a rendezvous that never arrives.
            if dec_sched is not None and dec_sched.host_segmented:
                step_us = dec_sched.run_host_segmented(
                    self._decoder_preamble,
                    dec_worker_addrs,
                    gpr_sets_by_worker=self._decode_attn_worker_gpr_sets(
                        dec_sched, aligned),
                    master_reserved_end=self._decoder_preamble + 64 * 8,
                    timeout_seconds=30.0,
                )
            else:
                if dec_sched is not None:
                    if (
                        worker_preamble_aligned == aligned
                        and worker_preamble_entries is not None
                    ):
                        if len(worker_preamble_entries) != len(dec_sched.workers):
                            raise RuntimeError(
                                "cached decode worker preambles no longer match "
                                "the scheduler topology"
                            )
                        for worker, entry in zip(
                            dec_sched.workers, worker_preamble_entries
                        ):
                            worker.start_execute_from_dram(entry)
                    else:
                        worker_preamble_entries = self._start_decode_workers(
                            dec_sched, dec_worker_addrs, aligned)
                        worker_preamble_aligned = aligned
                self.start_execute_from_dram(self._decoder_preamble)
                # Native decode is only about 100 ms/token.  The generic 1-ms
                # queue poll can therefore hide roughly half a percent of real
                # throughput after the FPGA has already halted.  A decode-only
                # 250-us sleep keeps polling bounded (no busy-spin) while
                # leaving all long-running stage waits at the global default.
                self._wait_lm_queue(
                    self, 30.0, "decode master", poll_interval_s=0.00025)
                for idx, w in enumerate(
                    dec_sched.workers if dec_sched is not None else [], start=1
                ):
                    self._wait_lm_queue(w, 30.0, f"decode worker {idx}")
                step_us = self.report_latency_in_us()
            # The FPGA step is now complete on every engine. Only now expose
            # its KV row as committed host state; failed uploads/launches leave
            # seq_len at the last known-complete token.
            self.seq_len = next_seq_len
            total_us += step_us
            # Per-step latencies: the FIRST is the peak-speed datapoint (shortest
            # KV history), and the spread across steps shows the context growth.
            self._decode_step_us.append(step_us)
            step_flops += (self._decoder_flops_fixed
                           + self._decoder_attn_per_aligned * aligned)

            token = self._decode_token()
            if token in stop:
                if use_status:
                    _status_teardown()
                break
            piece = self.tokenizer.decode([token])
            out.append(piece)
            self._set_silent(False)
            print(piece, end="", flush=True)
            self._set_silent(True)
            if use_status:
                _status_update()
        else:
            # Loop ran to the token cap or the context end rather than breaking.
            if use_status:
                _status_teardown()
        self._set_silent(prev_silent)

        wall = time.perf_counter() - t0
        n = max(1, len(out))
        gflops = (step_flops / (total_us * 1e-6) / 1e9) if total_us else 0.0
        peak = self.vis_peak_gflops()
        note = ""
        if peak and gflops > peak * 1.005:
            # Cannot happen physically; means the FLOP count is billed at
            # shapes the hardware did not run.
            note = f"  [warn] {100 * gflops / peak:.0f}% of peak is impossible"
        # Emitted BEFORE the [LM] line on purpose: model_auto_test slices the
        # generated text from "--- Decode run ---" up to "\nDecoder done in",
        # so anything printed in between lands inside what it scores as model
        # output. Keeping the marker here leaves that region pure token stream.
        self._loud(f"\nDecoder done in {wall:.2f} seconds, "
                   f"speed: {n / wall:.2f} tokens/s, total {self.seq_len} tokens.")

        # First-token speed from the HW counter is the PEAK: the KV history is
        # shortest on step 1, so every later step attends further and is slower.
        first_toks = (1e6 / self._decode_step_us[0]) if self._decode_step_us else 0.0
        self._loud(f"\n  [LM] {n} tokens in {wall:.2f}s = {n / wall:.2f} tok/s "
                   f"(1st decode token: {first_toks:.2f} tok/s, "
                   f"{step_flops / n / 1e9:.2f} GFLOP/token, {gflops:.1f} GFLOPS"
                   f"{f' = {100 * gflops / peak:.0f}% of peak' if peak else ''})"
                   f"{note}")
        self._decode_total_us = total_us
        self._decode_step_flops = step_flops
        self._decode_wall_s = wall
        self._decode_n = n
        self._decode_gflops = gflops
        self._decoded_text = "".join(out)
        self._prompt_tokens = getattr(self, "_prompt_tokens", None)
        return self.seq_len, self._decoded_text
