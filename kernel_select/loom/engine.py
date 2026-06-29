#!/usr/bin/env python3
"""loom.engine — the executor. Owns the single UnifiedEngine, lowers torch -> IR,
runs each Workflow, and dumps the ue-MLIR dialect after every run.

Design boundary (deliberate, to avoid reinventing validated silicon):
  - The NEW spine (core.py) owns detection, archetype, scheduling, and the dump.
  - DEVICE EMISSION is delegated to the proven, SNR-validated executors in
    kernel_select/mlir (IRExecutor for vision block-stacks). engine.lower() also
    builds a clean core.Graph from the same lowering so detect/dump run on the
    real graph — nothing here is model-specific.

After every run, engine writes kernel_select/loom_dumps/<model>.<region>.<arch>.mlir
with a numbers header (tokens/hidden/depth/program-MB/tensor-DRAM/SNR).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import core

# the validated frontend lives in kernel_select/mlir; the single kernel leaf
# (nn_lib) lives at the repo root — add it so `import nn_lib` always resolves to
# the one, complete copy regardless of cwd (there used to be a stale mlir dup).
_HERE = Path(__file__).resolve().parent
_KSEL = _HERE.parent
_REPO = _KSEL.parent
sys.path[:0] = [str(_KSEL / "mlir"), str(_KSEL), str(_REPO)]

DUMP_DIR = _KSEL / "loom_dumps"


# --------------------------------------------------------------------------- #
# torch -> core.Graph  (auto, model-agnostic)
# --------------------------------------------------------------------------- #
def _ops_to_graph(name, args, op_dicts, op_weights) -> core.Graph:
    """Convert torch_to_ue.lower's parsed op dicts (res/op/operands/attrs/out)
    into a canonical core.Graph, attaching bound weights so detect+dump see the
    real graph. Pure translation — no model knowledge."""
    g = core.Graph(name)
    for nm, shape in args:
        g.add_tensor(core.Tensor(nm, tuple(shape or ()), role="input"))
        g.inputs.append(nm)
    for o in op_dicts:
        res, shape = o["res"], tuple(o["out"] or ())
        g.add_tensor(core.Tensor(res, shape))
        weights = op_weights.get(res, {}) or {}
        g.add_op(core.Op(o["op"], list(o["operands"]), res, shape,
                         attrs=dict(o["attrs"]), weights=dict(weights)))
    if op_dicts:
        g.outputs.append(op_dicts[-1]["res"])
    return g


def lower(module, config, *, example_inputs=None, patch_spec=None) -> core.Graph:
    """Frontend hook injected into core.detect(): torch module -> core.Graph via
    the proven torch_to_ue.lower(). example_inputs is supplied by the caller that
    has the real pixel/token inputs (run.py); without it we cannot trace, so
    detect falls back to config-only classification (graph=None)."""
    if example_inputs is None:
        return None
    import torch_to_ue
    from loom_walker import parse_mlir
    mlir, em, _ = torch_to_ue.lower(module, example_inputs,
                                    patch_spec=patch_spec, strict=False)
    args, ops = parse_mlir(mlir)
    return _ops_to_graph(getattr(config, "model_type", "model"),
                         args, ops, em.op_weights)


# --------------------------------------------------------------------------- #
# Engine: schedule + run + dump
# --------------------------------------------------------------------------- #
class Engine:
    """Owns execution for one model run. Holds no UnifiedEngine of its own yet —
    the delegated IRExecutor constructs the single `ue` for the device span; as
    archetypes move inward, the ue lifetime hoists here. One Engine per process."""

    def __init__(self, model_id, *, model=None, verbose=True):
        self.model_id = model_id
        self.model = model                   # used to size the DRAM map to the footprint
        self.verbose = verbose
        self._reg_n = 0
        self._ue = None                      # the ONE UnifiedEngine (lazy: init on first use)

    # --- 4GB DRAM partitioning (offsets from 0x80000000; DMA adds the base) ---
    def _dram_bases(self):
        """Size the three arenas to the model, per the 4GB rule: params from
        offset 0x0, tensor ABOVE the largest single-workload params_end, program
        pinned just under the 4GB ceiling. Workloads reset between runs, so the
        peak params is the MAX of (bf16 vision encoder, q4 decoder) — not the sum.
        Fixed 2GB-era bases overflow big bf16 vision encoders (error 512)."""
        TOP = 0x100000000                    # 4GB offset ceiling (absolute 0x180000000)
        PROG = 0x10000000                    # 256MB program at the top
        MIN_TENSOR = 0x20000000              # >=512MB activations/KV to run
        align = lambda x, a=0x1000000: (x + a - 1) // a * a
        params_bytes = 0
        if self.model is not None:
            vis = dec = 0
            for n, p in self.model.named_parameters():
                nl = n.lower()
                if any(h in nl for h in ("vision", "visual", "image", "patch", "vit",
                                         "siglip", "connector")):
                    vis += p.numel() * 2                 # vision/connector stay bf16
                else:
                    dec += int(p.numel() * 0.5 * 1.18)   # decoder is q4 (+scale/pad)
            # The vision encoder ALLOCATES ~2.8x its weight bytes — the per-layer
            # attention emit (emit_windowed_attention) allocates scratch/identity/
            # bias per layer on top of the weights. Size to that real footprint so
            # params_end clears tensor_base (else overlap -> NaN). TODO: share those
            # per-layer attention buffers to drop the encoder back toward 1x.
            VIS_ALLOC = 2.8
            params_bytes = int(max(vis * VIS_ALLOC, dec) * 1.05)
        tensor_base = max(align(params_bytes), 0x40000000)   # >=1GB floor for params
        program_base = TOP - PROG
        if tensor_base + MIN_TENSOR > program_base:
            raise MemoryError(
                f"params footprint {params_bytes/1e9:.2f}GB leaves <"
                f"{MIN_TENSOR/1e6:.0f}MB for tensor before program; model too big "
                f"for the 4GB map (quantize the vision encoder).")
        return 0x0, tensor_base, program_base

    # --- the single shared engine --------------------------------------------
    @property
    def ue(self):
        """The one and only UnifiedEngine, created once (one FPGA init) and passed
        to every workload's executor. No workload constructs its own."""
        if self._ue is None:
            from user_dma_core import UnifiedEngine
            params_base, tensor_base, program_base = self._dram_bases()
            if self.verbose:
                print(f"[loom] DRAM map (offsets): params 0x0 / "
                      f"tensor 0x{tensor_base:08X} / program 0x{program_base:08X}")
            self._ue = UnifiedEngine(params_dram_base=params_base,
                                     tensor_dram_base=tensor_base,
                                     program_dram_base=program_base)
            self._clear_full_dram(self._ue)
        return self._ue

    def _clear_full_dram(self, ue, fill_byte=0x00, chunk=64 * 1024 * 1024):
        """Clear the ENTIRE physical 4GB DRAM, using the SAME absolute range as
        randomize_dram.py: raw addresses [0x0 .. 0x100000000). Raw dma_write
        takes the absolute physical address (no base added), so the range is
        0-based, NOT 0x80000000-based — writing past 0x100000000 returns
        'error 512' (out of range). Fill with 0x00 so any unwritten lane reads
        finite 0.0, not the 0xFF (=bf16 NaN) that stale DRAM holds. Uses only
        the public DMA API (no user_dma_core edits)."""
        from user_dma_core import DMA_DEVICE_H2C
        DRAM_BASE = 0x0
        DRAM_END  = 0x100000000                 # 4GB physical ceiling (0-based)
        total = DRAM_END - DRAM_BASE
        buf = bytes([fill_byte]) * chunk
        off = 0
        if self.verbose:
            print(f"[loom] clearing full 4GB DRAM "
                  f"[{DRAM_BASE:#x}..{DRAM_END:#x}) fill=0x{fill_byte:02x}")
        while off < total:
            n = min(chunk, total - off)
            ue.dma_write(DMA_DEVICE_H2C, DRAM_BASE + off, buf[:n], n)
            off += n
        return self._ue

    def reset_device(self):
        """Soft-reset between workloads: reclaim DRAM + ISA regs + capture state so
        the next workload reuses the same map — WITHOUT re-initializing the FPGA.
        Safe because activations thread host-side via io; never call it between
        prefill and decode (they share the on-device KV cache)."""
        if self._ue is None:
            return
        for fn in ("reset_tensor_dram_addr", "reset_program_dram_addr",
                   "reset_params_dram_addr", "reset_isa_reg_counter",
                   "reset_inst_ptr_counter", "clear_capture_buffer", "clear_inst_id"):
            getattr(self._ue, fn, lambda: None)()

    # --- loop-register handles (symbolic until the harness assigns real GPRs) -
    def alloc_loop_reg(self, name):
        self._reg_n += 1
        return f"{name}#{self._reg_n}"

    # --- capture: declare programs (delegation records the inputs for run) ----
    def capture_block_stack(self, graph, body, dims, loop_reg):
        return core.Program(name="forward", loop_reg=loop_reg)

    def capture_decode(self, graph, body, dims, loop_reg, phase):
        return core.Program(name=phase, loop_reg=loop_reg)

    def capture_static(self, graph, name):
        return core.Program(name=name)

    # --- the dump (after every run) ------------------------------------------
    def dump(self, workflow, graph, numbers: dict):
        DUMP_DIR.mkdir(exist_ok=True)
        slug = self.model_id.replace("/", "_")
        path = DUMP_DIR / f"{slug}.{workflow.region.name}.{workflow.archetype.kind}.mlir"
        d = workflow.dims
        head = [
            f"// loom dump — {self.model_id}",
            f"// region={workflow.region.name} role={workflow.region.role} "
            f"archetype={workflow.archetype.kind}",
            f"// depth={d.get('depth')} tokens={d.get('tokens')} "
            f"hidden={d.get('hidden')} ffn={d.get('ffn')} "
            f"gemms/blk={d.get('n_gemm')} attn={d.get('has_attn')}",
            "// numbers: " + "  ".join(f"{k}={v}" for k, v in numbers.items()),
            "",
        ]
        path.write_text("\n".join(head) + (graph.to_mlir() if graph else ""))
        if self.verbose:
            print(f"[loom] dumped {path}")
        return path

    # --- decode-layer graph for the dump (synthesized from spec) -------------
    def _decode_graph(self, name, spec, block):
        """Render ONE decode layer as ue-dialect for the dump. The decode path
        emits ISA directly (no ue-MLIR text), so we synthesize a faithful M=1
        layer skeleton from the extracted block + spec dims."""
        H, hd, nq, nkv = spec["H"], spec["hd"], spec["nq"], spec["nkv"]
        g = core.Graph(name)
        x = g.add_tensor(core.Tensor("%x", (1, H), role="input")); g.inputs.append("%x")
        n = [0]
        def op(opname, ins, shape, **attrs):
            n[0] += 1; r = f"%{n[0]}"
            g.add_tensor(core.Tensor(r, tuple(shape)))
            g.add_op(core.Op(opname, ins, r, tuple(shape), attrs=attrs)); return r
        h = op("rms_norm", ["%x"], (1, H))
        q = op("qmatmul", [h], (1, nq * hd), note="M=1")
        k = op("qmatmul", [h], (1, nkv * hd), note="M=1")
        v = op("qmatmul", [h], (1, nkv * hd), note="M=1")
        a = op("decode_attention", [q, k, v], (1, nq * hd),
               group=nq // nkv, pbi=(1 < nkv <= 8))
        o = op("qmatmul", [a], (1, H), note="M=1")
        r1 = op("eltwise", ["%x", o], (1, H), mode="add")
        h2 = op("rms_norm", [r1], (1, H))
        gt = op("qmatmul", [h2], (1, spec["I"]), note="M=1")
        up = op("qmatmul", [h2], (1, spec["I"]), note="M=1")
        ml = op("eltwise", [gt, up], (1, spec["I"]), mode="mul")
        dn = op("qmatmul", [ml], (1, H), note="M=1")
        r2 = op("eltwise", [r1, dn], (1, H), mode="add")
        g.outputs.append(r2)
        return g

    # --- prefill workload: build the decoder + populate the KV cache ---------
    def run_prefill(self, workload, *, model, tokenizer, prompt, io):
        """w(prefill): build k_select's ExtractedDecoder (which owns the proven
        M=1/PBI decode machinery), run the prompt prefill (M=tokens) to populate
        the KV cache, and stash the live decoder state in `io` for the peer decode
        workload. If io carries 'inputs_embeds' (image-grounded, from connect),
        prefill over those merged embeddings instead of raw text embeddings."""
        import torch
        import k_select as K

        dec, wpfx, rotary = K.locate_decoder(model)
        spec = K.derive_spec(model, dec)
        block = K.derive_block(model, dec)

        # VLM prefill reuses the processor input_ids (with image placeholders) so
        # the embed-scale probe and SEQ match the merged sequence; text-only models
        # build ids from the prompt + chat template.
        ids = io.get("input_ids")
        if ids is None:
            ids = tokenizer(prompt, return_tensors="pt").input_ids
            if getattr(tokenizer, "chat_template", None):
                chat = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True, tokenize=False)
                ids = tokenizer(chat, return_tensors="pt",
                                add_special_tokens=False).input_ids
        embed = dec.embed_tokens.weight.float()
        lmw = model.lm_head.weight.float()
        with torch.no_grad():
            x_input = io.get("inputs_embeds")               # merged image+text, if any
            if x_input is None:
                x_input = dec.embed_tokens(ids)[0].float()
            i0 = int(ids[0, 0].item())
            probe = dec.embed_tokens(torch.tensor([[i0]]))[0, 0].float()
            norm = (probe / embed[i0]).median().item()
            if spec["hd"] < 128:
                dvec = torch.zeros(1, 1, K.MAXPOS, spec["hd"])
                cf, sf = rotary(dvec, torch.arange(K.MAXPOS)[None])
                K.ExtractedDecoder.rope_cos = cf[0, :, :spec["hd"] // 2].float()
                K.ExtractedDecoder.rope_sin = sf[0, :, :spec["hd"] // 2].float()
        SEQ = x_input.shape[0]
        if SEQ >= K.MAXPOS:
            raise ValueError(
                f"prefill sequence {SEQ} >= MAXPOS {K.MAXPOS} (KV-cache capacity). "
                f"Use a smaller image/fewer frames, or raise k_select.MAXPOS "
                f"(grows KV cache + rope tables).")

        K.ExtractedDecoder._dram = K.decide_dram(model)
        K.ExtractedDecoder.spec = spec
        K.ExtractedDecoder.sd = model.state_dict()
        K.ExtractedDecoder.block = block
        K.ExtractedDecoder.wpfx = wpfx
        K.ExtractedDecoder.plan = K.decide_forward(spec, norm)
        K.ExtractedDecoder.SEQ = SEQ
        K.ExtractedDecoder.x_input = x_input
        m = K.ExtractedDecoder(ue=self.ue)                 # prep + build on the shared engine

        m.execute("prefill", bind={m.seq: SEQ, m.qseq: SEQ * m.GRP,
                                   m.bucket: max(1, (SEQ * m.GRP + 63) // 64),
                                   m.qall: SEQ * spec["nq"], m.kall: SEQ * spec["nkv"]})
        last = m.ue.dma_from_accelerator_memory(m.out.addr, (SEQ, m.H)).float()[-1]

        # hand the live decoder forward to the decode workload
        io["decoder"] = dict(m=m, spec=spec, block=block, embed=embed, lmw=lmw,
                             norm=norm, SEQ=SEQ, last=last)
        numbers = dict(layers=spec["layers"], hidden=spec["H"], heads=spec["nq"],
                       kv_heads=spec["nkv"], head_dim=spec["hd"], M_prefill=SEQ,
                       prompt_tokens=SEQ,
                       tensor_mb=round(m.ue.get_tensor_dram_usage() / 1e6, 1),
                       params_mb=round(m.ue.get_params_dram_usage() / 1e6, 1))
        self.dump(workload, self._decode_graph(
            self.model_id.replace("/", "_"), spec, block), numbers)
        return numbers

    # --- decode workload: the M=1 per-token loop -----------------------------
    def run_decode(self, workload, *, io, tokenizer, max_new_tokens=32, eos_ids=None):
        """w(decode): the peer of prefill. Consumes the live decoder state from io
        (set by run_prefill) and spins the M=1 loop, KV pointer advancing per
        token via the proven ExtractedDecoder path."""
        import sys
        import time
        import torch
        import k_select as K
        st = io["decoder"]
        m, spec, embed, lmw, norm, SEQ, last = (
            st["m"], st["spec"], st["embed"], st["lmw"], st["norm"], st["SEQ"], st["last"])

        # stream the generated tokens as they decode: print the detokenized delta
        # after each token, then report throughput (tok/s) over the decode loop.
        def stream(toks, shown):
            full = tokenizer.decode(toks, skip_special_tokens=True)
            if len(full) > len(shown):
                sys.stdout.write(full[len(shown):])
                sys.stdout.flush()
            return full

        eos = set(eos_ids or [])
        out = [int((last @ lmw.T).argmax().item())]
        gen = min(max_new_tokens, K.MAXPOS - SEQ)
        sys.stdout.write("\n[loom] decode> ")
        shown = stream(out, "")
        t0 = time.perf_counter()
        for step in range(gen - 1):
            if out[-1] in eos:
                break
            pos = SEQ + step
            m.ue.dma_to_accelerator_memory(
                m.dx.addr, (embed[out[-1]] * norm).view(1, m.H).to(torch.bfloat16))
            K._set_decode_bias(m, pos + 1)
            m.execute("decode", bind={m.bucket: max(1, (pos + 1 + 63) // 64)}, banner=False)
            out.append(m.ue.get_arg_max_index())
            shown = stream(out, shown)
        elapsed = time.perf_counter() - t0
        n_dec = max(0, len(out) - 1)
        rate = (n_dec / elapsed) if elapsed > 0 else 0.0
        sys.stdout.write("\n")
        print(f"[loom] decode throughput: {n_dec} tok / {elapsed:.2f}s = {rate:.1f} tok/s")

        pbi_decode = 1 < spec["nkv"] <= K.PBI_DECODE_MAX_NKV
        numbers = dict(M=1, kv_heads=spec["nkv"],
                       attn_path=("group_attn_pbi" if pbi_decode else
                                  "group_attn" if spec["nkv"] > 1 else "decode_attention(1kv)"),
                       pbi_decode=pbi_decode, generated=len(out),
                       decode_s=round(elapsed, 2), tok_per_s=round(rate, 1))
        self.dump(workload, self._decode_graph(
            self.model_id.replace("/", "_"), spec, st["block"]), numbers)
        text = tokenizer.decode(out, skip_special_tokens=True)
        return out, text, numbers

    # --- connect workload: project encoder output into decoder embedding space
    def run_connect(self, workload, *, model, io):
        """w(connect): the glue, ON DEVICE. Lowers the connector module (pixel-
        shuffle reshapes/permutes + projection matmul) through the generic compiler
        and runs it on the FPGA, then merges the image embeds into the text sequence
        at the <image> positions so prefill sees one fused sequence. The pixel-
        shuffle permutes map to smart_bf16_permute_core; the projection to matmul."""
        import torch
        conn = workload.region.module
        hidden = io.get("image_hidden")
        numbers = dict()
        if hidden is None:                       # nothing to project (text-only run)
            return dict(skipped="no image")

        # DEVICE: lower + run the connector through IRExecutor.
        from loom_ir import IRExecutor, snr_db
        from loom_walker import parse_mlir
        with torch.no_grad():
            ref = conn(hidden)
            ref = ref[0] if isinstance(ref, (tuple, list)) else ref
            ref = ref.reshape(-1, ref.shape[-1])
        ex = IRExecutor(conn, (hidden,), strict=False, ue=self.ue)
        hw, _ = ex.run()
        image_embeds = hw.reshape(-1, hw.shape[-1]).float()
        io["image_embeds"] = image_embeds
        numbers.update(ex.stats)
        numbers["image_tokens"] = image_embeds.shape[0]
        numbers["embed_dim"] = image_embeds.shape[-1]
        M, N = image_embeds.shape
        numbers["snr_db"] = round(float(snr_db(ref.reshape(M, N), image_embeds)), 2)
        args, ops = parse_mlir(ex.mlir)
        graph = _ops_to_graph(self.model_id.replace("/", "_"),
                              args, ops, ex.em.op_weights)
        self.dump(workload, graph, numbers)

        # MERGE: scatter the image embeds into the text-embedding sequence at the
        # <image> placeholder positions -> one fused sequence for prefill. Generic:
        # uses the model's own input embedding and the config image_token_id.
        ids = io.get("input_ids")
        img_tok = io.get("image_token_id")
        if ids is not None and img_tok is not None:
            with torch.no_grad():
                txt = model.get_input_embeddings()(ids)[0].float()   # [seq, H]
                mask = (ids[0] == img_tok)
                n_slots, n_img = int(mask.sum()), image_embeds.shape[0]
                if n_slots == n_img and n_img > 0:
                    txt[mask] = image_embeds.to(txt.dtype)
                    io["inputs_embeds"] = txt
                    numbers["merged"] = True
                    numbers["image_slots"] = n_slots
                else:
                    numbers["merged"] = False
                    numbers["slot_mismatch"] = f"{n_slots} slots vs {n_img} embeds"
        return numbers

    # --- run a ViT/SigLIP encoder STACK on device (embeddings -> device) -----
    def run_encoder_stack(self, workload, *, stack, embeds, reference=None):
        """Lower the encoder's transformer stack (the repeating layers) through the
        generic compiler and run it on device, with the host-computed embeddings as
        the DIRECT input — no conv, no encoder_span, no inject. This is the default
        path for single-stage ViT/SigLIP encoders (their first LN is layer0.norm1,
        so the Swin-shaped span mis-splits them). The IR is the same generic ue
        dialect; only the input boundary differs (embeddings instead of pixels)."""
        from loom_ir import IRExecutor, snr_db
        from loom_walker import parse_mlir

        # COMPILE-TIME LARGE-MAGNITUDE FLAG: the hardware LayerNorm computes
        # variance naively and overflows -> NaN on large-magnitude / outlier inputs
        # (SigLIP-SO400M vision: embeddings absmax ~169 vs the 500M's ~12). Detect it
        # up front so it's a clear flagged condition, not a silent device NaN.
        # DIAGNOSTIC: LOOM_EMB_SCALE=s divides the injected embedding by s before
        # lowering, to test whether op#0's LayerNorm NaN is magnitude-driven (op#0
        # input shrinks; if it goes finite, magnitude is the cause). Changes the
        # downstream result (residual), so it's a diagnostic, not a fix.
        _sc = float(os.environ.get("LOOM_EMB_SCALE", "1") or "1")
        if _sc != 1.0:
            import torch as _t
            embeds = (embeds.float() / _sc).to(embeds.dtype)
            print(f"[loom] DIAGNOSTIC: scaled embeds by 1/{_sc} -> absmax={float(embeds.abs().max()):.1f}")

        # DIAGNOSTIC/FIX: pad the TOKEN dim (M) to a multiple of 64 with zero rows.
        # 729 (27x27 patch grid) is not 64-aligned -> the row-wise kernels (LayerNorm
        # etc.) corrupt -> NaN. Padding M to pad64 fixes the alignment; the dummy
        # rows are zeros (harmless for per-row ops; attention masks padded keys).
        if os.environ.get("LOOM_PAD_M"):
            import torch as _t
            M0 = embeds.shape[-2]
            Mp = ((M0 + 63) // 64) * 64
            if Mp != M0:
                pad = _t.zeros(*embeds.shape[:-2], Mp - M0, embeds.shape[-1], dtype=embeds.dtype)
                embeds = _t.cat([embeds, pad], dim=-2)
                print(f"[loom] DIAGNOSTIC: padded token dim M {M0} -> {Mp} (64-aligned)")

        amax = float(embeds.abs().max())
        LN_SAFE_ABSMAX = 64.0
        if amax > LN_SAFE_ABSMAX:
            print(f"[loom] ⚠ LayerNorm-overflow risk: encoder input absmax={amax:.1f} "
                  f"> {LN_SAFE_ABSMAX:.0f} (outlier embeddings). The hardware LayerNorm "
                  f"can NaN on this magnitude — flagged at compile time.")

        # `stack` is expected to already be a clean forward(x)->tensor module (the
        # caller builds it). We lower it DIRECTLY — wrapping it again here would
        # double-nest the module and break torch.export (_export_root.m key error).
        ex = IRExecutor(stack, (embeds,), strict=False, ue=self.ue)
        hw, _ = ex.run()                                   # whole stack on device
        numbers = dict(ex.stats)
        if reference is not None:
            M, N = hw.shape
            numbers["snr_db"] = round(float(snr_db(reference.reshape(M, N), hw)), 2)
        args, ops = parse_mlir(ex.mlir)
        graph = _ops_to_graph(self.model_id.replace("/", "_"),
                              args, ops, ex.em.op_weights)
        self.dump(workload, graph, numbers)
        return hw, numbers

    # --- run a vision block-stack via the proven IRExecutor ------------------
    def run_block_stack(self, workflow, *, model, example_inputs, ws,
                        inject=None, stop_after=None, reference=None):
        """Delegate device emission to the validated IRExecutor (the 45 dB path),
        collect numbers, dump the ue-MLIR. `reference` (torch encoder output) gives
        the SNR; inject/stop_after carry the host patch-embed / head split."""
        from loom_ir import IRExecutor, snr_db
        ex = IRExecutor(model, example_inputs, ws=ws, strict=False, ue=self.ue)
        hw, last = ex.run(inject=inject, stop_after=stop_after)
        numbers = dict(ex.stats)
        if reference is not None:
            M, N = hw.shape
            numbers["snr_db"] = round(float(snr_db(reference.reshape(M, N), hw)), 2)
        # build the clean core.Graph from the SAME lowering for detect+dump
        from loom_walker import parse_mlir
        args, ops = parse_mlir(ex.mlir)
        graph = _ops_to_graph(self.model_id.replace("/", "_"), args, ops, ex.em.op_weights)
        self.dump(workflow, graph, numbers)
        return hw, numbers
