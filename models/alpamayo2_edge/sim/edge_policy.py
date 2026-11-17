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
import os
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
        # A2E_HOST_EQUIV=1: the accelerator's model on the host -- its 4-bit LM weights (dequantized
        # from the cached packs), its GELU, its 64-token reasoning cap. Separates "the device model
        # drives differently" from "the device computes it wrong".
        self.equiv = os.environ.get("A2E_HOST_EQUIV") == "1"
        if self.equiv:
            self.max_new = 64
            self._install_if4()

    def _install_if4(self):
        sys.path.insert(0, str(HERE.parent.parent.parent))      # repo root, for quant_lib
        import quant_lib
        from alpamayo2_edge_cpu_reference import LM

        w = self.m.w
        packs = torch.load(HERE.parent / "alpamayo2_edge_bin" / "lm_if4.pt", weights_only=False)
        for key, (data, scale, n, k) in packs.items():
            deq = quant_lib.dequant("if4", data, scale, n, k)
            part = key.split(".")
            if key == "head":
                name, sl = "vlm.lm_head.weight", None
            elif len(part) == 2:
                name, sl = f"{LM}layers.{part[0]}.self_attn.{part[1]}_proj.weight", None
            else:
                name, i = f"{LM}layers.{part[0]}.mlp.{part[1]}.weight", int(part[2])
                half = w[name].shape[0 if part[1] == "fc1" else 1] // 2
                rows = slice(i * half, (i + 1) * half)
                sl = (rows, slice(None)) if part[1] == "fc1" else (slice(None), rows)
            deq = deq.to(w[name].device, w[name].dtype)
            if sl is None:
                w[name] = deq
            else:
                w[name][sl] = deq
        print(f"[host] device-equivalent model: {len(packs)} IF4 tensors installed, hw GELU, max {self.max_new} tokens", flush=True)

    @torch.no_grad()
    def run(self, packed, seed):
        """packed dict -> (xyz (64, 3), rot (64, 3, 3), generated token ids, anchor last)."""
        from alpamayo2_edge_cpu_reference import KVCache

        m = self.m
        ids = packed["tokenized_data.input_ids"][0]
        types = packed["tokenized_data.mm_token_type_ids"][0]
        grid = packed["tokenized_data.image_grid_thw"]
        L = ids.shape[0]
        image_embeds = m.project(m.vision_encode(packed["tokenized_data.pixel_values"], grid, hw_gelu=self.equiv), hw_gelu=self.equiv)
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


RESUME_FILE = Path.home() / "a2e_resume"


def _device_worker(conn, model_dir, device):
    """Child process that owns the board: one DeviceRunner, one reply per request, exits on any error."""
    try:
        from edge_device import DeviceRunner

        sample = EdgeInputBuilder(model_dir).pack(**synthetic_raw(nav_text="Continue straight"))
        runner = DeviceRunner(model_dir, sample, ref_device=device)
        conn.send(("ok", None))
        while True:
            packed, seed = conn.recv()
            xyz, rot, tokens = runner.run(packed, seed)
            conn.send(("ok", (xyz.float().cpu(), rot.float().cpu(), list(tokens))))
    except EOFError:
        pass
    except BaseException as e:  # noqa: BLE001 -- a dead board shows up as TimeoutError / OSError
        conn.send(("error", f"{type(e).__name__}: {e}"))


class DeviceBackend:
    """The accelerator: same interface as HostReferenceBackend.run. Weights and programs are put on
    the board at construction (a few minutes), sized from a synthetic sample of the driving input.

    The board lives in a child process. When an inference fails (the board dies under sustained
    load), run() waits for the board to be reset -- `touch ~/a2e_resume` -- then starts a fresh
    child and repeats the same inference, so the caller only sees a slow plan."""

    def __init__(self, model_dir=None, device="cuda"):
        self.args = (model_dir or default_model_dir(), device)
        self.proc = None
        while not self._start():
            self._wait_for_reset(f"start-up failed: {self.err}")

    def _start(self):
        import multiprocessing as mp

        if self.proc is not None:
            self.proc.kill()
            self.proc.join()
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_device_worker, args=(child, *self.args), daemon=True)
        self.proc.start()
        child.close()
        kind, self.err = self._reply()
        return kind == "ok"

    def _reply(self):
        try:
            return self.conn.recv()
        except (EOFError, OSError) as e:
            return "error", f"device process died ({type(e).__name__})"

    def _wait_for_reset(self, why):
        RESUME_FILE.unlink(missing_ok=True)       # a stale file must not resume a dead board
        print(f"[device] {why}\n[device] reset the board, then run:  touch {RESUME_FILE}", flush=True)
        while not RESUME_FILE.exists():
            time.sleep(1.0)
        RESUME_FILE.unlink()
        print("[device] resuming: reloading the board", flush=True)

    def run(self, packed, seed):
        while True:
            try:
                self.conn.send((packed, seed))
                kind, val = self._reply()
            except OSError as e:
                kind, val = "error", f"device process died ({type(e).__name__})"
            if kind == "ok":
                return val
            self._wait_for_reset(f"inference failed: {val}")
            while not self._start():
                self._wait_for_reset(f"start-up failed: {self.err}")


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
        dump = os.environ.get("A2E_DUMP_DIR")       # every plan's input and output, for edge_device.py replay
        if dump:
            Path(dump).mkdir(parents=True, exist_ok=True)
            self.n_dumped = getattr(self, "n_dumped", 0) + 1
            torch.save({"packed": packed, "seed": int(seed), "xyz": torch.from_numpy(xyz), "tokens": list(tokens),
                        "nav_text": nav_text}, Path(dump) / f"plan_{self.n_dumped - 1:03d}.pt")
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
