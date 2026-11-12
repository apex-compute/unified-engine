#!/usr/bin/env python3
"""Alpamayo 2 Edge as a policy: raw sensor inputs in, 6.4 s trajectory out. Backend-agnostic.

  policy = EdgePolicy(HostReferenceBackend(device="cuda"))
  xyz, rot, text = policy.predict(frames, camera_ids, timestamps, hist_xyz, hist_rot, nav_text, seed)

xyz (64, 3) / rot (64, 3, 3) are the poses at t0 + 0.1 .. 6.4 s in the t0 ego frame (x forward,
y left). Reasoning is greedy, so a (inputs, seed) pair always gives the same plan; the seed
only draws the flow-matching noise.

  # one full host inference on the synthetic golden sample (loads the weights; bf16 on cuda, float32 on cpu)
  python models/alpamayo2_edge/sim/edge_policy.py --selfcheck --device cuda
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from edge_inputs import N_WAYPOINTS, EdgeInputBuilder, default_model_dir, synthetic_raw  # noqa: E402


class HostReferenceBackend:
    """Runs alpamayo2_edge_cpu_reference.Alpamayo2EdgeRef end to end (the flow of its run())."""

    def __init__(self, model_dir=None, device="cpu", dtype=None, steps=10, max_new=256):
        from alpamayo2_edge_cpu_reference import Alpamayo2EdgeRef

        device = torch.device(device)
        if dtype is None:
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.steps, self.max_new = steps, max_new
        self.m = Alpamayo2EdgeRef(model_dir or default_model_dir(), device, dtype)

    @torch.no_grad()
    def run(self, packed, seed):
        """packed dict -> (xyz (64, 3), rot (64, 3, 3), generated token ids, anchor last)."""
        from alpamayo2_edge_cpu_reference import KVCache

        m = self.m
        ids = packed["tokenized_data.input_ids"][0]
        types = packed["tokenized_data.mm_token_type_ids"][0]
        grid = packed["tokenized_data.image_grid_thw"]
        L = ids.shape[0]
        image_embeds = m.project(m.vision_encode(packed["tokenized_data.pixel_values"], grid))
        pos, delta = m.mrope_positions(types, grid)
        embeds = m.embed_prompt(ids, image_embeds)
        cache = KVCache(m.n_lm, m.kv_heads, L + self.max_new + 1, m.hd, m.dev, m.dt)
        m.lm_forward(embeds[: L - 1], pos[:, : L - 1], cache)
        tokens = m.decode(ids[-1], L - 1, delta, cache, "greedy", None, self.max_new)
        x0 = torch.randn(m.n_way, 2, generator=torch.Generator().manual_seed(int(seed)))
        action = m.denoise(cache, delta, x0, self.steps)
        xyz, rot = m.action_to_traj(
            action, packed["traj_data.ego_history_xyz"].reshape(-1, 3), packed["traj_data.ego_history_rot"].reshape(-1, 3, 3)
        )
        return xyz, rot, tokens


class DeviceBackend:
    """STUB -- the accelerator backend. Same interface as HostReferenceBackend.run."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("DeviceBackend: the accelerator hook is added later by the maintainer")

    def run(self, packed, seed):
        raise NotImplementedError("DeviceBackend: the accelerator hook is added later by the maintainer")


class EdgePolicy:
    def __init__(self, backend, model_dir=None):
        self.backend = backend
        self.inputs = EdgeInputBuilder(model_dir)

    def predict(self, frames, camera_ids, timestamps, ego_history_xyz, ego_history_rot, nav_text, seed=0):
        """frames: 22 HWC uint8 RGB; camera_ids: model ids (0, 1, 2, 3, 5, 6), 3 or 5 frames each;
        timestamps: capture times in us; ego_history_*: 20 poses at 10 Hz in the t0 ego frame.
        Returns (waypoints_xyz (64, 3) float32, rotations (64, 3, 3) float32, reasoning_text)."""
        packed = self.inputs.pack(frames, camera_ids, timestamps, ego_history_xyz, ego_history_rot, nav_text)
        xyz, rot, tokens = self.backend.run(packed, seed)
        xyz = np.asarray(torch.as_tensor(xyz).float().cpu(), dtype=np.float32)
        rot = np.asarray(torch.as_tensor(rot).float().cpu(), dtype=np.float32)
        if xyz.shape != (N_WAYPOINTS, 3) or rot.shape != (N_WAYPOINTS, 3, 3) or not (np.isfinite(xyz).all() and np.isfinite(rot).all()):
            raise RuntimeError(f"backend returned a bad trajectory: {xyz.shape} {rot.shape}")
        text = self.inputs.tokenizer.decode(list(tokens)[:-1], skip_special_tokens=False).strip()  # drop the anchor
        return xyz, rot, text


def selfcheck(args):
    """One host inference on the synthetic sample, compared with NVIDIA's free-running output.
    The plans are not expected to be identical: NVIDIA sampled the reasoning (T=0.6) and drew
    its noise on its own generator; this run is greedy with our seed."""
    import json

    from safetensors.torch import load_file

    gdir = Path(args.golden).expanduser()
    meta = json.loads((gdir / "meta.json").read_text())
    t = time.perf_counter()
    policy = EdgePolicy(HostReferenceBackend(args.model_dir, args.device, steps=args.steps), args.model_dir)
    t_load = time.perf_counter() - t
    raw = synthetic_raw(meta["t0_us"], meta["nav_text"])
    t = time.perf_counter()
    xyz, rot, text = policy.predict(raw["frames"], raw["camera_ids"], raw["timestamps_us"], raw["ego_history_xyz"], raw["ego_history_rot"], raw["nav_text"], seed=args.seed)
    t_run = time.perf_counter() - t
    xyz2, _, text2 = policy.predict(raw["frames"], raw["camera_ids"], raw["timestamps_us"], raw["ego_history_xyz"], raw["ego_history_rot"], raw["nav_text"], seed=args.seed)
    nv = load_file(str(gdir / "golden.safetensors"))["pred_xyz"].numpy()
    print(f"load {t_load:.1f}s, predict {t_run:.1f}s on {args.device}")
    print(f"reasoning (ours, greedy): {text!r}")
    print(f"reasoning (NVIDIA, sampled): {meta['cot']!r}")
    print(f"end point ours ({xyz[-1, 0]:.2f}, {xyz[-1, 1]:.2f}) m, NVIDIA ({nv[-1, 0]:.2f}, {nv[-1, 1]:.2f}) m; "
          f"mean xy distance {float(np.linalg.norm(xyz[:, :2] - nv[:, :2], axis=-1).mean()):.3f} m")
    repeat = float(np.abs(xyz - xyz2).max())
    print(f"same seed twice: max|diff| {repeat:.3g} m, text {'equal' if text == text2 else 'DIFFERENT'}")
    ok = repeat == 0.0 and text == text2
    print("selfcheck:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--golden", default="~/a2e_golden/synthetic_float32")
    ap.add_argument("--model-dir")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()
    if not args.selfcheck:
        ap.error("nothing to do: pass --selfcheck")
    sys.exit(0 if selfcheck(args) else 1)


if __name__ == "__main__":
    main()
