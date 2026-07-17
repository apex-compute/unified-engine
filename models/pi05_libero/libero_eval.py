"""LIBERO evaluation for pi0.5 -- SINGLE PROCESS, swappable backend.

Steps the real LIBERO simulator, drives the robot with a pi0.5 action chunk, and
reports the success rate. Two interchangeable backends produce that chunk:

    --backend torch   pi05_torch_ref.run_pi05 on GPU/CPU  (the GOLDEN reference)
    --backend fpga    the UnifiedEngine on real hardware

Everything outside the backend -- sim, seeding, init states, preprocessing,
replan cadence, denoise noise -- is shared, so the two paths are directly
comparable EPISODE BY EPISODE. That pairing is the point: it turns
"is my silicon faithful?" into a per-episode yes/no instead of a fight between
two noisy success percentages with overlapping confidence intervals.

Setup once:
    conda create -n pi05_libero python=3.11 -y && conda activate pi05_libero
    pip install -r models/pi05_libero/requirements.txt
    pip install -e ~/apex-compute-ML/simple-llm/src/models/pi0_5/openpi_src/third_party/libero

Run:
    # golden reference on GPU -- start here
    MUJOCO_GL=egl python models/pi05_libero/libero_eval.py \
        --backend torch --tasks 1 --trials 1 --max-steps 60          # smoke test
    MUJOCO_GL=egl python models/pi05_libero/libero_eval.py \
        --backend torch --trials 5                                   # 10x5 = 50 episodes

    # same 50 episodes on hardware, then diff the two result JSONs
    MUJOCO_GL=egl python models/pi05_libero/libero_eval.py \
        --backend fpga --trials 5

Each run writes per-episode success bits to --results-out (JSON). Diff two of
them to get the paired FPGA-vs-reference agreement.

The backend predicts a 10x7 chunk; LIBERO executes --replan-steps of it before
re-querying. Success = the task's BDDL goal predicate fires (done=True) before
max_steps.

DETERMINISM: the denoise noise is pinned (numpy RandomState(0)) in BOTH backends,
so the policy is a deterministic function of the observation. Combined with fixed
init_states, an episode is bit-reproducible and the two backends see identical
inputs at every step. Note this is a deviation from real openpi, which samples
fresh noise per call -- fine for verification, but say so before quoting a number.
"""
import argparse
import collections
import functools
import gc
import math
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).parent
_OPENPI = pathlib.Path.home() / "apex-compute-ML" / "simple-llm" / "src" / "models" / "pi0_5" / "openpi_src"
sys.path.insert(0, str(_OPENPI / "packages" / "openpi-client" / "src"))   # image_tools
sys.path.insert(0, str(_OPENPI / "third_party" / "libero"))              # libero package
sys.path.insert(0, str(_HERE))                                           # engine module

# pi0.5 checkpoint assets (norm_stats + tokenizer) -- same paths as _test.py main().
_CKPT = pathlib.Path.home() / ".cache" / "openpi" / "openpi-assets" / "checkpoints" / "pi05_libero"
_NORM_STATS_PATH = _CKPT / "assets" / "physical-intelligence" / "libero" / "norm_stats.json"
_TOKENIZER_PATH = pathlib.Path.home() / ".cache" / "openpi" / "big_vision" / "paligemma_tokenizer.model"

LIBERO_ENV_RESOLUTION = 256   # sim render resolution
RESIZE = 224                  # model input resolution
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

_SUITE_MAX_STEPS = {
    "libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
    "libero_10": 520, "libero_90": 400,
}


# ---------------------------------------------------------------------------
# pi0.5 input construction (mirrors pi05_libero_test.main() exactly)
# ---------------------------------------------------------------------------
class _Pi05Pre:
    """Raw LIBERO obs -> (images, prompt_tokens) for the engine; un-normalizes
    the (10,7) output back to robot action space."""

    def __init__(self):
        import json
        import sentencepiece
        self.norm = json.loads(_NORM_STATS_PATH.read_text())["norm_stats"]
        self.sp = sentencepiece.SentencePieceProcessor(model_file=str(_TOKENIZER_PATH))
        self.s_q01 = np.array(self.norm["state"]["q01"], dtype=np.float32)
        self.s_q99 = np.array(self.norm["state"]["q99"], dtype=np.float32)
        self.a_q01 = np.array(self.norm["actions"]["q01"], dtype=np.float32)
        self.a_q99 = np.array(self.norm["actions"]["q99"], dtype=np.float32)

    def images(self, base_u8, wrist_u8):
        """3 slots [base, wrist, zero-pad] as (224,224,3) float32 in [-1,1].
        Inputs already resize_with_pad'd to 224. Zero 3rd slot triggers the
        engine's masked-vision skip."""
        base = np.asarray(base_u8, dtype=np.float32) / 127.5 - 1.0
        wrist = np.asarray(wrist_u8, dtype=np.float32) / 127.5 - 1.0
        return [base, wrist, np.zeros_like(base)]

    def prompt_tokens(self, state_8d, language):
        state = np.asarray(state_8d, dtype=np.float32)
        state_norm = np.clip(
            (state - self.s_q01) / (self.s_q99 - self.s_q01 + 1e-6) * 2.0 - 1.0, -1.0, 1.0)
        digitized = np.digitize(state_norm, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        state_str = " ".join(map(str, digitized))
        full = f"Task: {language}, State: {state_str};\nAction: "
        return np.array(self.sp.encode(full, add_bos=True), dtype=np.int64)

    def unnormalize(self, actions):  # (10,7) normalized [-1,1] -> robot space
        return (actions + 1.0) / 2.0 * (self.a_q99[:7] - self.a_q01[:7] + 1e-6) + self.a_q01[:7]


def _quat2axisangle(quat):
    q = np.array(quat, dtype=np.float32)
    q[3] = min(1.0, max(-1.0, q[3]))
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


# ---------------------------------------------------------------------------
# Backends: raw model inputs -> (10,7) NORMALIZED action chunk.
# ---------------------------------------------------------------------------
# The flow-matching denoise loop starts from noise. Both backends must start from
# the SAME noise or their chunks differ for reasons that have nothing to do with
# hardware fidelity. The FPGA seeds itself with numpy RandomState(0) (see
# pi05_libero_test.run_inference), so that is the shared source of truth here --
# NOT torch.manual_seed(0), which draws a completely different sequence.
AE_ACTION_HORIZON, AE_ACTION_DIM_PADDED = 10, 32


def _fixed_noise32():
    """(10, 32) denoise seed, identical in both backends. Cols 7:32 are padding."""
    n = np.zeros((AE_ACTION_HORIZON, AE_ACTION_DIM_PADDED), dtype=np.float32)
    n[:, :7] = np.random.RandomState(0).randn(AE_ACTION_HORIZON, 7)
    return n


class _TorchBackend:
    """Golden reference: pi05_torch_ref.run_pi05 on GPU (or CPU)."""

    def __init__(self, device, quant, fresh_noise=False):
        import torch
        import pi05_torch_ref as R
        self.torch, self.R = torch, R
        self.device = device
        self.fresh_noise = fresh_noise
        self.dtype = {"fp32": torch.float32, "bf16": torch.bfloat16,
                      "if4": torch.bfloat16}[quant]
        print(f"[eval] loading torch reference weights (device={device} quant={quant} "
              f"noise={'fresh' if fresh_noise else 'fixed'})...", flush=True)
        self.W = R.Weights(device=device, dtype=self.dtype, quant=quant)
        self.noise32 = torch.from_numpy(_fixed_noise32()).to(device)[None]  # (1,10,32)
        self._rng = np.random.RandomState(12345)

    def _noise(self):
        # Real openpi samples FRESH noise per call; pinning it (our default, for
        # FPGA-vs-ref determinism) makes the policy a deterministic function of the
        # observation. That is great for paired verification but it is ONE fixed draw
        # from a stochastic policy, so it is NOT the setting to quote a success rate from.
        if not self.fresh_noise:
            return self.noise32
        n = np.zeros((AE_ACTION_HORIZON, AE_ACTION_DIM_PADDED), dtype=np.float32)
        n[:, :7] = self._rng.randn(AE_ACTION_HORIZON, 7)
        return self.torch.from_numpy(n).to(self.device)[None]

    def infer(self, images, toks, state_8d):
        torch = self.torch
        # images: list of 3 (224,224,3) float32 in [-1,1], slot 2 all-zero.
        imgs = torch.from_numpy(np.stack(images)).to(self.device, self.dtype)
        # The FPGA skips all-zero image slots and records them in prefix_masked_cols
        # as [(i*256, (i+1)*256)]. The reference must mask the SAME columns, else its
        # RoPE positions (cumsum(mask)-1) diverge from the FPGA's for every text token
        # AND for every suffix token -- a silent, total mismatch.
        masked_cols = [(i * 256, (i + 1) * 256)
                       for i, im in enumerate(images) if not np.any(im)]
        out = self.R.run_pi05(
            imgs,
            torch.as_tensor(toks, device=self.device, dtype=torch.long)[None],
            torch.as_tensor(state_8d, device=self.device, dtype=torch.float32)[None],
            self.W, noise=self._noise(), masked_cols=masked_cols or None)
        return out[0, :, :7].float().cpu().numpy()


class _FpgaBackend:
    """The UnifiedEngine on real hardware."""

    def __init__(self):
        import pi05_libero_test as M
        print("[eval] building FPGA engine (weight_init + tensor_init)...", flush=True)
        self.ue = M.Pi05Libero_UnifiedEngine()
        M.init_hang_prevention(self.ue)
        self.ue.weight_init()
        self.ue.tensor_init(M._CFG["defaults"].get("max_seq", 512))

    def infer(self, images, toks, state_8d):
        # run_inference seeds its own noise from RandomState(0) when noise32 is None,
        # which is exactly _fixed_noise32() -- but pass it explicitly so the two
        # backends can never silently drift apart if that default ever changes.
        S = self.ue.AE_ACTION_HORIZON_PADDED
        noise32 = np.full((S, self.ue.AE_XT_WIDTH), 1e-6, dtype=np.float32)
        noise32[:AE_ACTION_HORIZON, :7] = _fixed_noise32()[:, :7]
        return self.ue.run_inference(images, toks, noise32=noise32)[:, :7]


_DIM_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "GRIPPER"]


def _diff_actions(path_a, path_b):
    """Compare two --dump-actions dumps. Inference #0 sees a bit-identical observation
    in both runs, so it is a CONTROLLED comparison: any difference there is the backend
    and nothing else. Reports per-dimension, because a single pooled SNR happily hides
    one inverted gripper behind six healthy dims."""
    A, B = np.load(path_a), np.load(path_b)
    na, nb = int(A["n"]), int(B["n"])
    print(f"\n{'='*68}\n  {pathlib.Path(path_a).name}  vs  {pathlib.Path(path_b).name}")
    print(f"  {na} vs {nb} inferences\n{'='*68}")

    # Inputs must match on inference 0, or the whole comparison is meaningless.
    print("\n--- inference #0 input check (must be identical) ---")
    ok = True
    for k in ("base_u8", "wrist_u8", "state", "tokens"):
        a, b = A[f"{k}_0"], B[f"{k}_0"]
        same = a.shape == b.shape and np.array_equal(a, b)
        ok &= same
        print(f"  {k:10s} {'identical' if same else '*** DIFFERS ***'}  shape={a.shape}")
    if not ok:
        print("\n  !! inputs differ -> the two runs did NOT see the same observation.")
        print("     Any action mismatch below is meaningless. Check --seed/--tasks/--trials.")
        return

    a, b = A["unnorm_0"], B["unnorm_0"]                     # (10,7) real robot units
    # A dim the robot is barely using (e.g. no rotation during a straight reach) has a
    # near-zero signal, so its SNR is a ratio of two tiny numbers -- meaningless, and it
    # reads as a scary low dB while the direction is perfect. Judge those on cosine only.
    # (Same trap as the masked-rows-poison-SNR bug: never SNR a signal that isn't there.)
    QUIET = 0.02   # RMS below this in robot units = dim is idle this chunk
    print(f"\n--- inference #0 action chunk, per-dimension (A={pathlib.Path(path_a).name}, "
          f"B={pathlib.Path(path_b).name}) ---")
    print(f"  {'dim':>8s} {'RMS':>8s} {'SNR dB':>8s} {'cosine':>8s} {'A mean':>9s} {'B mean':>9s}  verdict")
    for d in range(7):
        x, y = a[:, d], b[:, d]
        rms = float(np.sqrt(np.mean(x ** 2)))
        sig, noise = np.linalg.norm(x), np.linalg.norm(x - y)
        snr = 20 * np.log10(sig / max(noise, 1e-12)) if sig > 1e-12 else float("nan")
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        cos = float(x @ y / denom) if denom > 1e-12 else float("nan")
        # A sign-flipped dim is the classic silent killer: magnitudes look sane, the
        # robot drives the wrong way. cosine < 0 catches it; SNR alone does not.
        if cos < -0.5:
            verdict = "*** SIGN FLIP ***"
        elif rms < QUIET:
            verdict = f"idle (|x|={rms:.4f}, SNR n/a)" if cos > 0.8 else "idle but UNCORRELATED"
        elif abs(cos) < 0.5:
            verdict = "*** UNCORRELATED ***"
        elif snr > 20:
            verdict = "ok"
        else:
            verdict = "drift"
        print(f"  {_DIM_NAMES[d]:>8s} {rms:8.4f} {snr:8.1f} {cos:8.3f} "
              f"{x.mean():9.4f} {y.mean():9.4f}  {verdict}")

    sig, noise = np.linalg.norm(a), np.linalg.norm(a - b)
    print(f"\n  overall SNR: {20*np.log10(sig/max(noise,1e-12)):.1f} dB   "
          f"(~29 dB is the known-good FPGA-vs-CPU-IF4 level for this model)")
    print("  READ THE COSINE COLUMN, NOT THE SNR. An idle dim's SNR is a ratio of two")
    print("  tiny numbers and means nothing; cosine still tells you the direction is right.")

    n = min(na, nb)
    if n > 1:
        print(f"\n--- divergence across the first {n} inferences (unnorm chunk) ---")
        print("  chunk #0 is controlled; later chunks legitimately drift once the two")
        print("  backends execute different actions and the sim state separates.")
        for i in range(n):
            x, y = A[f"unnorm_{i}"], B[f"unnorm_{i}"]
            sig, noise = np.linalg.norm(x), np.linalg.norm(x - y)
            snr = 20 * np.log10(sig / max(noise, 1e-12))
            gsign = float(np.mean(np.sign(x[:, 6]) == np.sign(y[:, 6])))
            print(f"    #{i}  SNR {snr:7.1f} dB   gripper sign agreement {gsign*100:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-suite", default="libero_spatial", choices=list(_SUITE_MAX_STEPS))
    ap.add_argument("--tasks", type=int, default=None,
                    help="cap #tasks (default: None = all 10 tasks in the suite)")
    ap.add_argument("--task-start", type=int, default=0,
                    help="first task id to run. Lets you split a suite across processes: "
                         "MuJoCo's EGL renderer segfaults after ~6 OffScreenRenderEnv "
                         "creations when CUDA is also active (torch backend), so run the "
                         "GPU eval in chunks of <=5 tasks. The FPGA backend does no CUDA "
                         "compute and is not affected.")
    ap.add_argument("--trials", type=int, default=2,
                    help="rollouts per task (default 2 -> 20 total across 10 tasks, "
                         "a directional +/-20%% number; bump to 5 for a citable +/-13%%)")
    ap.add_argument("--max-steps", type=int, default=None, help="override suite default")
    ap.add_argument("--replan-steps", type=int, default=10,
                    help="execute this many actions of each 10-step chunk before re-querying "
                         "the FPGA (default 10 = use the whole chunk, ~half the inferences of 5)")
    ap.add_argument("--wait-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--video-out", default=str(_HERE / "data" / "libero" / "videos"))
    ap.add_argument("--backend", default="torch", choices=["torch", "fpga"],
                    help="torch = pi05_torch_ref golden reference; fpga = real hardware")
    ap.add_argument("--device", default="cuda", help="--backend torch only")
    ap.add_argument("--quant", default="if4", choices=["fp32", "bf16", "if4"],
                    help="--backend torch only; if4 matches what the FPGA actually runs")
    ap.add_argument("--results-out", default=None,
                    help="JSON of per-episode success bits (default: data/libero/"
                         "results_<backend>_<suite>.json). Diff two of these for the "
                         "paired backend comparison.")
    ap.add_argument("--no-video", action="store_true", help="skip mp4 writing")
    ap.add_argument("--fresh-noise", action="store_true",
                    help="--backend torch only: sample fresh denoise noise per inference, "
                         "like real openpi. Default is FIXED noise (RandomState(0)) so the "
                         "torch and fpga backends are bit-comparable -- but fixed noise is "
                         "one draw from a stochastic policy, so use --fresh-noise when "
                         "measuring a success rate, not when doing a paired FPGA diff.")
    ap.add_argument("--dump-actions", action="store_true",
                    help="save every inference's (images, state, tokens, chunk) to "
                         "data/libero/actions_<backend>_<suite>.npz. Run once per backend, "
                         "then: python libero_eval.py --diff-actions A.npz B.npz")
    ap.add_argument("--diff-actions", nargs=2, metavar=("A.npz", "B.npz"), default=None,
                    help="compare two --dump-actions files and exit (no sim, no model)")
    args = ap.parse_args()

    if args.diff_actions:
        return _diff_actions(*args.diff_actions)

    # torch.load: LIBERO's get_task_init_states() loads numpy-pickled files that
    # torch 2.x rejects under the weights_only=True default. These are trusted.
    import torch
    torch.load = functools.partial(torch.load, weights_only=False)

    import imageio
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools

    # --- build the backend ONCE (weights stay resident across all episodes) ---
    backend = (_TorchBackend(args.device, args.quant, args.fresh_noise)
               if args.backend == "torch" else _FpgaBackend())
    pre = _Pi05Pre()
    print(f"[eval] backend '{args.backend}' ready.", flush=True)

    # --dump-actions: record the exact (input, output) of every inference. Two runs
    # with different backends are directly diffable -- and inference #0 of an episode
    # is guaranteed to see a BIT-IDENTICAL observation in both (same init_state, same
    # wait-steps, no model action executed yet). So chunk #0 is a controlled
    # comparison; any mismatch there is purely the backend. Later chunks may diverge
    # legitimately, because a different action changes the next observation.
    dumps = []

    def infer(base_u8, wrist_u8, state_8d, language):
        images = pre.images(base_u8, wrist_u8)
        toks = pre.prompt_tokens(state_8d, language)
        actions = backend.infer(images, toks, state_8d)   # (10,7) normalized
        unnorm = pre.unnormalize(actions)
        if args.dump_actions:
            dumps.append({
                "norm": np.asarray(actions, dtype=np.float32),
                "unnorm": np.asarray(unnorm, dtype=np.float32),
                "state": np.asarray(state_8d, dtype=np.float32),
                "tokens": np.asarray(toks, dtype=np.int64),
                "base_u8": np.asarray(base_u8, dtype=np.uint8),
                "wrist_u8": np.asarray(wrist_u8, dtype=np.uint8),
            })
        return unnorm

    max_steps = args.max_steps or _SUITE_MAX_STEPS[args.task_suite]
    np.random.seed(args.seed)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    n_tasks = min(args.tasks, task_suite.n_tasks) if args.tasks else task_suite.n_tasks
    video_dir = pathlib.Path(args.video_out)
    video_dir.mkdir(parents=True, exist_ok=True)

    import json
    results_path = pathlib.Path(args.results_out) if args.results_out else (
        video_dir.parent / f"results_{args.backend}_{args.task_suite}.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    total_ep, total_succ = 0, 0
    episodes = []   # per-episode success bits -> the paired-comparison record

    def _save():
        # Written after EVERY episode, not just at the end: an FPGA suite run is many
        # hours, and a crash at task 5 must not throw away tasks 0-4.
        results_path.write_text(json.dumps({
            "backend": args.backend, "task_suite": args.task_suite, "seed": args.seed,
            "task_start": args.task_start, "tasks": n_tasks, "trials": args.trials,
            "max_steps": max_steps, "replan_steps": args.replan_steps,
            "quant": args.quant if args.backend == "torch" else "if4-hw",
            "noise": ("fresh" if (args.backend == "torch" and args.fresh_noise) else "fixed"),
            "successes": total_succ, "episodes_total": total_ep,
            "success_rate": total_succ / max(1, total_ep),
            "episodes": episodes,
        }, indent=2))

    for task_id in range(args.task_start, args.task_start + n_tasks):
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=LIBERO_ENV_RESOLUTION, camera_widths=LIBERO_ENV_RESOLUTION)
        env.seed(args.seed)

        task_ep, task_succ = 0, 0
        for trial in range(args.trials):
            env.reset()
            obs = env.set_init_state(init_states[trial])
            plan = collections.deque()
            replay, done, t = [], False, 0
            print(f"[eval] task {task_id} '{task.language}' trial {trial}", flush=True)
            while t < max_steps + args.wait_steps:
                if t < args.wait_steps:                    # let objects settle
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue
                # rotate 180 to match training, resize_with_pad -> 224 uint8
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE))
                wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, RESIZE, RESIZE))
                replay.append(img)
                if not plan:
                    state = np.concatenate((
                        obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"]))
                    chunk = infer(img, wrist, state, str(task.language))
                    plan.extend(chunk[: args.replan_steps])
                obs, _, done, _ = env.step(np.asarray(plan.popleft()).tolist())
                if done:
                    break
                t += 1
            task_ep += 1
            total_ep += 1
            done = bool(done)
            if done:
                task_succ += 1
                total_succ += 1
            episodes.append({"task_id": task_id, "trial": trial,
                             "language": str(task.language), "steps": t, "success": done})
            _save()   # checkpoint every episode -- a crash must not lose hours of work
            suffix = "success" if done else "failure"
            if not args.no_video:
                imageio.mimwrite(
                    video_dir / f"{args.backend}_{args.task_suite}_t{task_id}_e{trial}_{suffix}.mp4",
                    [np.asarray(x) for x in replay], fps=10)
            print(f"[eval]   -> {'SUCCESS' if done else 'failure'}  "
                  f"(running {total_succ}/{total_ep} = {100*total_succ/total_ep:.1f}%)", flush=True)
        print(f"[eval] task {task_id} success rate: {task_succ}/{task_ep}", flush=True)
        env.close()
        del env
        gc.collect()   # MuJoCo's EGL context is not freed by close() alone

    if args.dump_actions and dumps:
        flat = {"n": np.array(len(dumps))}
        for i, d in enumerate(dumps):
            for k, v in d.items():
                flat[f"{k}_{i}"] = v
        actions_path = video_dir.parent / f"actions_{args.backend}_{args.task_suite}.npz"
        np.savez_compressed(actions_path, **flat)
        print(f"[eval] dumped {len(dumps)} inferences -> {actions_path}", flush=True)

    _save()
    print("\n" + "=" * 56)
    print(f"  LIBERO {args.task_suite}  |  backend={args.backend}  "
          f"trials/task={args.trials}  tasks={n_tasks}")
    print(f"  SUCCESS RATE: {total_succ}/{total_ep} = {100*total_succ/max(1,total_ep):.1f}%")
    print(f"  per-episode results -> {results_path}")
    print("=" * 56)


if __name__ == "__main__":
    main()
