#!/usr/bin/env python3
"""Alpamayo 2 Edge on the accelerator, as a reusable backend: weights and programs go to the
board once, then every call runs one full inference (vision, projector, prefill, greedy decode,
action expert) there. It drives alpamayo2_edge_test.py's engine class -- the same programs and
the same host glue as that script's main(), without the reference comparisons.

The host keeps what the test keeps on the host: token embedding lookup, the next-token pick,
RoPE tables, the masks, Fourier features and the Euler update, and the unicycle integration.

The prompt length changes by a few tokens with the navigation text. The prefill program and the
per-slot append programs are built for one length, so each new length is compiled once (about
15 s) and kept.
"""
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

LM_PROGRAMS = ("scatter_img", "prefill", "build_vt", "decode", "expert")


class DeviceRunner:
    def __init__(self, model_dir, sample, ref_device="cuda", max_new=None, steps=None):
        """sample: one packed input (edge_inputs.EdgeInputBuilder.pack) -- fixes the image grids and
        the first prompt length to compile for."""
        import alpamayo2_edge_test as T
        import user_dma_core

        self.T = T
        dev = os.environ.get("A2E_DEV", T.CFG["defaults"]["dev"])
        self.max_new = max_new or T.CFG["model"]["max_new_tokens"]
        self.steps = steps or T.CFG["action_expert"]["flow_steps"]
        t0 = time.time()
        user_dma_core.set_dma_device(dev)
        user_dma_core.configure_clock_from_hardware()
        cores = getattr(user_dma_core, "ANDROMEDA_CORE_COUNT", None) or 1
        engines = int(os.environ.get("A2E_ENGINES", cores))
        ref_device = torch.device(ref_device)
        dtype = torch.bfloat16 if ref_device.type == "cuda" else torch.float32
        self.ref = ref = T.Alpamayo2EdgeRef(T.resolve_model_dir(model_dir), device=ref_device, dtype=dtype)
        grid = sample["tokenized_data.image_grid_thw"].long()
        self.grids = [(int(h), int(w)) for _, h, w in grid.tolist()]
        self.rows = sum(h * w for h, w in self.grids)

        ue = self.ue = T.Alpamayo2Edge_UnifiedEngine(dev=dev, engines=engines)
        ue.vision_weight_init(ref, sorted(set(self.grids)))
        ue.vision_tensor_init(self.grids)
        ue.projector_init(ref)
        ue.vision_compile(list(range(len(self.grids))))
        ue.projector_compile(self.rows)
        packs = T.prepare_lm_if4(ref)               # also makes the reference's LM the dequantized one
        ue.lm_weight_init(ref, packs)
        del packs
        ue.expert_weight_init(ref)
        self.emb_table = ref.w[T.LM + "embed_tokens.weight"]
        self.cache = {}
        self.cooldown, self.idle_from = float(os.environ.get("A2E_COOLDOWN", 0)), None   # idle seconds between inferences
        ids = sample["tokenized_data.input_ids"][0]
        ue.lm_tensor_init(ids.shape[0] - 1, self.max_new)
        self._programs(ids)
        print(f"[device] {engines} engines ready in {time.time() - t0:.0f} s (prompt {ids.shape[0]} tokens, S {ue.S})", flush=True)

    def _programs(self, ids):
        """Make the LM programs for this prompt current, compiling them the first time it is seen."""
        ue, T = self.ue, self.T
        P = ids.shape[0] - 1
        is_img = (ids == self.ref.image_token)[:P]
        runs, src, i = [], 0, 0
        while i < P:                                # [(prompt row, first image-embed row, count)]
            if is_img[i]:
                j = i
                while j < P and is_img[j]:
                    j += 1
                runs.append((i, src, j - i))
                src, i = src + j - i, j
            else:
                i += 1
        assert src == self.rows // T.MERGE2, f"{src} image placeholders vs {self.rows // T.MERGE2} image embeddings"
        key = (P, tuple(runs))
        if key not in self.cache:
            assert T._pad_seq(P + 1 + self.max_new) == ue.S, \
                f"a {P + 1}-token prompt does not fit the buffers sized at start-up (S = {ue.S})"
            t0 = time.time()
            ue.P = P
            ue.lm_compile(runs, self.max_new)
            names = LM_PROGRAMS + tuple(f"append_{s}" for s in range(P, P + 1 + self.max_new))
            self.cache[key] = ({n: ue.prog[n] for n in names}, {n: ue.wprog[n] for n in names if n in ue.wprog},
                               {n: ue.flops.get(n, 0) for n in names})
            print(f"[device] programs for a {P + 1}-token prompt compiled in {time.time() - t0:.0f} s", flush=True)
        prog, wprog, flops = self.cache[key]
        ue.P = P
        ue.prog.update(prog), ue.wprog.update(wprog), ue.flops.update(flops)

    @torch.no_grad()
    def run(self, packed, seed):
        """packed dict -> (xyz (64, 3), rot (64, 3, 3), generated token ids, anchor last)."""
        ue, T, ref = self.ue, self.T, self.ref
        b, LH, BPE = ue.buf, T.LH, T.BPE
        if self.idle_from is not None and self.cooldown:
            time.sleep(max(0.0, self.idle_from + self.cooldown - time.time()))
        t_wall, ms = time.time(), {}
        grid = packed["tokenized_data.image_grid_thw"].long()
        assert [(int(h), int(w)) for _, h, w in grid.tolist()] == self.grids, "image sizes changed since start-up"
        ids = packed["tokenized_data.input_ids"][0]
        types = packed["tokenized_data.mm_token_type_ids"][0]
        self._programs(ids)
        P, S, CAP = ue.P, ue.S, ue.KV_CAP

        # vision + projector
        ue.vision_upload_pixels(packed["tokenized_data.pixel_values"].float())
        ms["vision"] = ue._exec("vision", timeout=300.0) / 1e3
        ms["projector"] = ue._exec("projector") / 1e3

        # prompt: text rows from the host table, image rows scattered on the device; pad rows repeat text rows
        pos, delta = ref.mrope_positions(types, grid)
        is_img = ids == ref.image_token
        x = self.emb_table[ids.to(ref.dev)].float().cpu()
        x[is_img] = 0
        text_rows = torch.nonzero(~is_img[:P]).flatten()
        x_pad = torch.cat([x[:P], x[text_rows[torch.arange(S - P) % len(text_rows)]]])
        ue.dma_to_accelerator_memory(b["X_A"], x_pad.to(torch.bfloat16).contiguous().flatten())
        slot_pos = torch.cat([pos[:, :P], (torch.arange(P, S) + delta).expand(3, -1)], 1)
        ue.lm_write_rope("ROPE_PRE", *ref.rope_cos_sin(slot_pos))
        ue._exec("scatter_img")
        ms["prefill"] = (ue._exec("prefill", timeout=300.0) + ue._exec("build_vt")) / 1e3

        # greedy decode on the device logits, until the trajectory anchor
        tok, index, out, us = int(ids[-1]), P, [], 0.0
        slots = torch.arange(CAP)
        while True:
            ue.dma_to_accelerator_memory(b["X_A"], self.emb_table[tok].to(torch.bfloat16).cpu().contiguous())
            ue.lm_write_rope("ROPE_CUR", *ref.rope_cos_sin(torch.full((3, 1), index + delta)))
            ue.lm_write_mask("MASK_DEC", 1, (slots < index) | (slots == ue.CUR))
            us += ue._exec("decode") + ue._exec(f"append_{index}")
            if out and out[-1] == ref.anchor:
                break
            logits = ue.dma_from_accelerator_memory(b["LOGITS"], (T.VOCAB,)).float()
            assert not bool(torch.isnan(logits).any()), "NaN logits from the device"
            nxt = ref.pick(logits, "greedy", None, 0.6, 0.98)
            if len(out) == self.max_new - 1 and nxt != ref.anchor:      # no anchor in time: force it so the expert runs
                nxt = ref.anchor
            out.append(nxt)
            tok, index = nxt, index + 1
        n_cached = index + 1
        ms["decode"] = us / 1e3

        # action expert: host Fourier features + Euler update around one device step
        n, V64 = T.NWAY, T.UE_VECTOR_SIZE
        ue.lm_write_rope("ROPE_EXP", *ref.rope_cos_sin((n_cached + delta + torch.arange(n)).expand(3, n)))
        ue.lm_write_mask("MASK_EXP", n, (slots < n_cached) | ((slots >= ue.E0) & (slots < ue.E0 + n)))
        xa = torch.randn(n, 2, generator=torch.Generator().manual_seed(int(seed)))
        ts, us = torch.linspace(0.0, 1.0, self.steps + 1), 0.0
        for i in range(self.steps):
            xd = xa.to(ref.dev)
            f = torch.cat([ref.fourier(xd[:, 0]), ref.fourier(xd[:, 1]), ref.fourier(ts[i].to(ref.dev)).expand(n, -1)], -1)
            feat = torch.zeros(n, V64)
            feat[:, : f.shape[1]] = f.float().cpu()
            ue.dma_to_accelerator_memory(b["FEAT"], feat.to(torch.bfloat16).flatten())
            us += ue._exec("expert")
            v = ue.dma_from_accelerator_memory(b["VOUT"], (n, V64)).float()[:, :2]
            assert not bool(torch.isnan(v).any()), "NaN velocity from the device"
            xa = xa + (ts[i + 1] - ts[i]) * v
        ms["expert"] = us / 1e3
        xyz, rot = ref.action_to_traj(xa, packed["traj_data.ego_history_xyz"].reshape(-1, 3),
                                      packed["traj_data.ego_history_rot"].reshape(-1, 3, 3))
        print(f"[device] inference {sum(ms.values()) / 1e3:.1f} s on the board (" + ", ".join(f"{k} {v / 1e3:.1f}" for k, v in ms.items())
              + f"), {time.time() - t_wall:.0f} s wall, {len(out)} tokens", flush=True)
        self.idle_from = time.time()
        return xyz, rot, out


if __name__ == "__main__":      # repeat test without the simulator: N inferences back to back on one sample
    sys.path.insert(0, str(HERE))
    from edge_inputs import EdgeInputBuilder, default_model_dir, synthetic_raw

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    sample = EdgeInputBuilder(default_model_dir()).pack(**synthetic_raw(nav_text="Continue straight"))
    runner = DeviceRunner(default_model_dir(), sample)
    first = None
    if len(sys.argv) > 2 and sys.argv[2] == "vision":       # vision only, back to back
        pause = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0      # idle seconds between runs
        for i in range(n):
            time.sleep(pause if i else 0.0)
            runner.ue.vision_upload_pixels(sample["tokenized_data.pixel_values"].float())
            print(f"[device] vision run {i}: {runner.ue._exec('vision', timeout=120.0) / 1e6:.1f} s", flush=True)
        sys.exit(0)
    for i in range(n):
        xyz, _, out = runner.run(sample, 0)
        first = xyz if first is None else first
        print(f"[device] run {i}: {len(out)} tokens, max |xyz - run 0| {float((xyz - first).abs().max()):.4f} m", flush=True)
