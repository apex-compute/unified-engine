"""LIBERO closed-loop evaluation for VeraPulse PulseVLA-LIBERO-0.5B -- SINGLE PROCESS.

Ported from models/pi05_libero/libero_eval.py, which is the proven harness for the same
class of model (VLA: vision + prefix + flow-matching action expert + action chunking) on
the same benchmark. Structure, cadence, checkpointing and failure handling are that
file's; only the model plumbing differs.

    reset -> settle -> observe -> infer a chunk -> execute --replan-steps of it -> re-plan

*** THE POLICY IS NOT VALIDATED YET. ***
As of this writing the action expert scores ~17 dB below its bf16 floor and the executed
actions sit at cos 0.97 against the CPU oracle. A success rate produced by this harness
today measures A MODEL WITH A KNOWN BUG, not the accelerator and not PulseVLA. The
harness exists so that the number can be taken the moment the expert bug is fixed. Every
run prints this warning and stamps `policy_validated: false` into the results JSON.

Run:
    # plumbing only -- no FPGA, no LIBERO, no simulator. Safe anywhere.
    python models/verapulse/libero_eval.py --dry-run --tasks 2 --trials 1 --max-steps 40

    # real: FPGA + LIBERO. MUJOCO_GL=egl is required for headless rendering.
    MUJOCO_GL=egl python models/verapulse/libero_eval.py \
        --task-suite libero_spatial --tasks 1 --trials 1 --max-steps 60   # smoke test
    MUJOCO_GL=egl python models/verapulse/libero_eval.py --trials 2       # 10x2 episodes

    # after a reboot (see below), resume where it stopped:
    MUJOCO_GL=egl python models/verapulse/libero_eval.py --trials 2 --resume

THREE CONSTRAINTS THIS FILE IS BUILT AROUND, all learned the expensive way:

1. COMPILE ONCE, EXECUTE MANY. The engine is constructed and weight_init/tensor_init are
   run EXACTLY ONCE for the whole eval; the three device programs are compiled on the
   first inference (via the engine's own _compile_once) and only re-EXECUTED afterwards.
   Recompiling advances the program-DRAM allocator every inference and marches it at the
   4 GB ceiling: pi05 died after 3 inferences that way and survived 252 once fixed. Any
   change here that re-enters weight_init or constructs a second engine per episode
   silently reintroduces that bug.

2. NEVER CONSTRUCT A SECOND UnifiedEngine after weight_init. Every ctor DMA-writes ~16 KB
   of noise to a hardcoded 0x80000000, and this model's params_dram_base is 0x00000000 --
   i.e. exactly there. The second engine would shred the first weights stored. There is
   one engine in this process, built once, at the top.

3. THE BOX HARD-REBOOTS under sustained FPGA load, with no kernel log. So results are
   checkpointed to --results-out AFTER EVERY EPISODE, and --resume reads that file back
   and skips the (task_id, trial) pairs already recorded. A reboot costs one episode, not
   a night. The resume command is printed after every checkpoint.

Everything is single-process on purpose (pi05's rule): the FPGA driver and MuJoCo are not
forked alongside each other.
"""
import argparse
import collections
import gc
import json
import math
import os
import pathlib
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))          # verapulse_test
sys.path.insert(0, str(_HERE.parent.parent))   # repo root (user_dma_core, engine libs)

# Headless rendering. MuJoCo picks its GL backend at IMPORT time, so this must be set
# before anything imports mujoco/robosuite -- setting it in the shell is the documented
# way (upstream's requirements.txt says so), and this is the belt-and-braces default for
# when someone forgets.
os.environ.setdefault("MUJOCO_GL", "egl")

LIBERO_ENV_RESOLUTION = 256      # what the sim renders
MODEL_IMAGE_SIZE = 512           # what the ViT consumes (config vision.image_size)
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

# Same per-suite step budgets pi05 uses; they come from the LIBERO paper's episode
# lengths, not from anything model-specific, so they carry over unchanged.
_SUITE_MAX_STEPS = {
    "libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
    "libero_10": 520, "libero_90": 400,
}

_POLICY_WARNING = (
    "  !! THE POLICY IS NOT NUMERICALLY VALIDATED. The action expert is ~17 dB below\n"
    "     its bf16 floor and the executed actions are at cos 0.97 vs the CPU oracle.\n"
    "     Any success rate below measures a KNOWN-BUGGY model. Do not quote it.")


# ---------------------------------------------------------------------------
# observation -> model inputs, and model output -> robot action
# ---------------------------------------------------------------------------
def _resize_with_pad(img_u8, size):
    """(H,W,3) uint8 -> (size,size,3) uint8, ASPECT PRESERVED with zero padding.

    LIBERO renders 256x256 (square) so the padding is inert today, but a
    plain stretch would silently distort the moment a non-square camera appears, and
    the training pipeline pads. Implemented locally with PIL rather than pulling in
    openpi's image_tools: this model has no openpi dependency and should not grow one.
    """
    from PIL import Image
    h, w = img_u8.shape[:2]
    if (h, w) == (size, size):
        return np.ascontiguousarray(img_u8)
    scale = min(size / h, size / w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    small = np.asarray(Image.fromarray(img_u8).resize((nw, nh), Image.BILINEAR))
    out = np.zeros((size, size, 3), dtype=np.uint8)
    t, l = (size - nh) // 2, (size - nw) // 2
    out[t:t + nh, l:l + nw] = small
    return out


def _quat2axisangle(quat):
    """robosuite's eef quaternion -> axis-angle, the form the state vector uses."""
    q = np.array(quat, dtype=np.float32)
    q[3] = min(1.0, max(-1.0, q[3]))
    den = math.sqrt(max(0.0, 1.0 - q[3] * q[3]))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


class _VeraPulsePre:
    """Raw LIBERO obs -> (images[2,512,512,3], token ids[48], mask[48], state[32]),
    and the model's normalized [10,7] chunk -> robot action space.

    Uses verapulse_test.tokenize() / load_norm_stats(), which is where those two stubs
    were implemented -- one source of truth shared with the _test.py entry point, so the
    eval and the bring-up runs can never tokenize or de-normalize differently.
    """

    def __init__(self, image_norm="pm1"):
        import verapulse_test as M
        self.M = M
        cfg = M._CFG
        self.text_len = cfg["lm"]["tokenizer_max_length"]         # 48
        self.state_dim = cfg["action_head"]["state_dim"]          # 32 (padded)
        self.action_dim = cfg["action_head"]["action_dim"]        # 7
        self.n_exec = cfg["action_head"]["n_action_steps"]        # 10
        self.slots = cfg["vision"]["num_image_slots"]             # 2
        ns = M.load_norm_stats()
        self.s_mean, self.s_std = (ns["state"][0].numpy(), ns["state"][1].numpy())
        self.a_mean, self.a_std = (ns["action"][0].numpy(), ns["action"][1].numpy())
        self.image_norm = image_norm

    # -- images -------------------------------------------------------------------
    def images(self, base_u8, wrist_u8):
        """[2,512,512,3] float32. Slot 0 = agentview ('image'), slot 1 = wrist
        ('image2') -- the two obs keys upstream's predict_example.py uses, in that order.

        ASSUMPTION (--image-norm): SigLIP/SmolVLM2 preprocessing is mean=std=0.5, i.e.
        pixels land in [-1,1]. verapulse_test.main() feeds /255 ([0,1]) for its synthetic
        smoke images, which is fine for an SNR gate (hardware and oracle see the same
        pixels either way) but is NOT the training distribution. Real rollouts use [-1,1];
        --image-norm unit switches back if the checkpoint turns out to disagree.
        """
        stack = np.stack([np.asarray(base_u8, dtype=np.float32),
                          np.asarray(wrist_u8, dtype=np.float32)])
        if self.image_norm == "pm1":
            return stack / 127.5 - 1.0
        return stack / 255.0

    # -- language -----------------------------------------------------------------
    def tokens(self, language):
        """(ids[48], mask[48]). The mask is the whole point: the prefix's valid length is
        1 + 128 + n_real_text, NOT the constant 177 in the config. Feeding 177 would let
        every pad slot of a short instruction attend as if it were language."""
        return self.M.tokenize(language, self.text_len, return_mask=True)

    # -- state --------------------------------------------------------------------
    def state(self, state_raw):
        """robot state (8: eef pos 3 + axis-angle 3 + gripper 2) -> normalized, zero-padded
        to max_state_dim=32.

        Normalization is (x - mean) / std from norm_stats -- the lerobot family transform,
        NOT pi05's q01/q99 quantile scaling. Padding is ZERO, and that is safe here (unlike
        the 1e-6 the prefix/x_t rows need) because this vector goes through state_proj, a
        plain matmul whose K-padded columns are zero weights: zeros in, zeros contributed.
        The epsilon-free RMSNorm never sees this vector directly."""
        s = np.asarray(state_raw, dtype=np.float32).reshape(-1)
        n = self.s_mean.shape[0]
        assert s.shape[0] == n, (
            f"norm_stats expects a {n}-dim robot state, got {s.shape[0]} -- do not "
            f"silently truncate; the mean/std would apply to the wrong joints")
        s = (s - self.s_mean) / np.maximum(self.s_std, 1e-6)
        out = np.zeros(self.state_dim, dtype=np.float32)
        out[:n] = s
        return out

    # -- output -------------------------------------------------------------------
    def unnormalize(self, actions):
        """[n,7] normalized -> robot action space: a * std + mean (inverse of the
        training-time (a - mean)/std). A wrong inverse here does not crash -- it makes a
        correct policy look broken -- so it lives beside load_norm_stats() and nowhere
        else."""
        a = np.asarray(actions, dtype=np.float32)
        return a * self.a_std[None, : a.shape[1]] + self.a_mean[None, : a.shape[1]]


# ---------------------------------------------------------------------------
# backends: model inputs -> (10,7) NORMALIZED action chunk
# ---------------------------------------------------------------------------
class _FpgaBackend:
    """The UnifiedEngine on real hardware. ONE engine, built ONCE (see constraints 1+2
    in the module docstring)."""

    def __init__(self, weights="real", seed=0, use_run_inference=False, snr=False,
                 strict_gates=False, fused_silu=True):
        import verapulse_test as M
        self.M = M
        self.use_run_inference = use_run_inference
        self.snr = snr
        self.strict_gates = strict_gates
        print("[eval] building FPGA engine (ctor + weight_init + tensor_init)...",
              flush=True)
        t0 = time.perf_counter()
        # THE ONLY UnifiedEngine construction in this process. Its ctor DMA-writes 16 KB
        # of noise to 0x80000000 == this model's params base, so a second one built later
        # would destroy the weights loaded below.
        self.ue = M.VeraPulse_UnifiedEngine()
        self.ue.PREFIX_FUSED_SILU = fused_silu
        self.ue.weight_init(dummy=(weights == "dummy"), seed=seed)
        self.ue.tensor_init()
        print(f"[eval] engine ready in {time.perf_counter() - t0:.1f}s", flush=True)
        # NOTE: no precompile_all() here. It is still NotImplementedError on this model,
        # so the three programs compile lazily inside the FIRST inference (each through
        # the engine's _compile_once) and every later inference is execute-only. That
        # makes inference #0 of the run structurally slower than the rest -- expected,
        # and NOT a per-episode cost. Wire precompile_all in here once it exists so run 0
        # stops being different from the others.
        self._first = True

    def infer(self, images, ids, mask, state32, noise=None):
        ue = self.ue
        if self._first:
            print("[eval] first inference: the encoder/prefix/denoise programs compile "
                  "here (once). Later inferences are execute-only.", flush=True)
        if self.use_run_inference:
            # Single entry point, but it cannot carry the text mask (run_inference does
            # not take one and is off-limits to edit), so the prefix falls back to
            # counting ALL 48 text slots as valid. Conservative -- it never masks a real
            # token -- but padded text then attends. Kept as an opt-in so the eval can be
            # cross-checked against exactly what verapulse_test.py --stage all runs.
            out = ue.run_inference(images, ids, state32, noise=noise, snr=self.snr,
                                   strict_gates=self.strict_gates)
            self._first = False
            return np.asarray(out, dtype=np.float32)
        # DEFAULT: the same three stages run_inference runs, in the same order, with the
        # real text mask plumbed into run_prefix -> build_attn_bias. The per-stage SNR
        # gates are deliberately NOT run per inference: each one builds/uses the torch
        # oracle from a 2.23 GB checkpoint, which is minutes of CPU per step. Gate with
        # `verapulse_test.py --stage all`, then roll out here.
        vision_tokens = ue.run_vision(images)
        ue.run_prefix(vision_tokens, ids, state32, text_mask=mask)
        chunk = ue.run_denoise(noise=noise)          # [50, 7] normalized
        self._first = False
        n_exec = self.M._CFG["action_head"]["n_action_steps"]
        return np.asarray(chunk[:n_exec], dtype=np.float32)


class _FakeBackend:
    """--dry-run: exercises the WHOLE harness with no FPGA and no engine construction.

    Deliberately not a mock of the engine API -- it is a stand-in for the one thing the
    harness consumes, a (10,7) normalized chunk. Its actions are small and smooth so a
    dry run steps the loop plausibly rather than flailing; it is plumbing, not a policy."""

    def __init__(self, n_exec=10, action_dim=7, seed=0):
        self.n_exec, self.action_dim = n_exec, action_dim
        self.rng = np.random.RandomState(seed)
        self.calls = 0

    def infer(self, images, ids, mask, state32, noise=None):
        self.calls += 1
        assert images.shape[0] == 2, "fake backend still checks the 2-camera contract"
        assert ids.shape[-1] == mask.shape[-1], "ids/mask length mismatch"
        return (self.rng.randn(self.n_exec, self.action_dim) * 0.05).astype(np.float32)


# ---------------------------------------------------------------------------
# fake environment for --dry-run
# ---------------------------------------------------------------------------
class _FakeTask:
    def __init__(self, tid):
        self.language = f"fake task {tid}: pick up the black bowl and place it on the plate"
        self.problem_folder, self.bddl_file = "fake", f"task{tid}.bddl"


class _FakeEnv:
    """Minimal stand-in for OffScreenRenderEnv: same obs keys, same step/reset contract,
    and it declares success on a fixed step so the bookkeeping/checkpoint/resume paths are
    all exercised (successes AND failures) with zero hardware and zero simulator."""

    def __init__(self, seed=0, succeed_after=None):
        self.rng = np.random.RandomState(seed)
        self.succeed_after = succeed_after
        self.t = 0

    def _obs(self):
        r = LIBERO_ENV_RESOLUTION
        return {
            "agentview_image": self.rng.randint(0, 256, (r, r, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": self.rng.randint(0, 256, (r, r, 3), dtype=np.uint8),
            "robot0_eef_pos": self.rng.randn(3).astype(np.float32) * 0.1,
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_gripper_qpos": self.rng.randn(2).astype(np.float32) * 0.01,
        }

    def seed(self, s):
        self.rng = np.random.RandomState(s)

    def reset(self):
        self.t = 0
        return self._obs()

    def set_init_state(self, st):
        return self._obs()

    def step(self, action):
        assert len(action) == 7, f"env expects a 7-dof action, got {len(action)}"
        assert np.all(np.isfinite(action)), "non-finite action reached the env"
        self.t += 1
        done = self.succeed_after is not None and self.t >= self.succeed_after
        return self._obs(), 0.0, bool(done), {}

    def close(self):
        pass


# ---------------------------------------------------------------------------
# results file: checkpoint + resume
# ---------------------------------------------------------------------------
def _load_previous(path):
    """Read a results JSON back for --resume. Returns (episodes, done_keys).

    Tolerant on purpose: a hard reboot can truncate the file mid-write, and losing the
    whole history to a JSONDecodeError would defeat the point of checkpointing. A
    corrupt file is reported and treated as empty."""
    p = pathlib.Path(path)
    if not p.exists():
        return [], set()
    try:
        blob = json.loads(p.read_text())
    except Exception as e:
        print(f"[eval] --resume: {p} is unreadable ({e!r}); starting fresh "
              f"(the old file is NOT overwritten until the first new episode).")
        return [], set()
    eps = blob.get("episodes", [])
    return eps, {(int(e["task_id"]), int(e["trial"])) for e in eps}


def main():
    ap = argparse.ArgumentParser(
        description="Closed-loop LIBERO evaluation for VeraPulse on the FPGA.")
    ap.add_argument("--task-suite", default="libero_spatial", choices=list(_SUITE_MAX_STEPS))
    ap.add_argument("--tasks", type=int, default=None,
                    help="cap #tasks (default: all 10 in the suite)")
    ap.add_argument("--task-start", type=int, default=0,
                    help="first task id. Lets a suite be split across processes.")
    ap.add_argument("--trials", type=int, default=2,
                    help="rollouts per task (2 -> 20 episodes, directional; 5 for a "
                         "citable number)")
    ap.add_argument("--max-steps", type=int, default=None, help="override the suite default")
    ap.add_argument("--replan-steps", type=int, default=10,
                    help="how many actions of each 10-step chunk to execute before "
                         "re-querying the model (default 10 = the whole chunk, i.e. the "
                         "fewest inferences; lower = tighter closed loop, more FPGA time)")
    ap.add_argument("--wait-steps", type=int, default=10,
                    help="dummy-action steps after reset so the objects settle")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--results-out", default=None,
                    help="per-episode results JSON (default: data/libero/"
                         "results_verapulse_<suite>.json). Written after EVERY episode.")
    ap.add_argument("--resume", action="store_true",
                    help="read --results-out back and SKIP the (task, trial) pairs it "
                         "already contains. This is the reboot recovery path.")
    ap.add_argument("--dry-run", action="store_true",
                    help="exercise the whole harness with a FAKE env and FAKE backend: "
                         "no FPGA, no engine construction, no LIBERO import, no MuJoCo. "
                         "This is the plumbing test.")
    ap.add_argument("--dry-succeed-every", type=int, default=3,
                    help="--dry-run only: make every Nth episode 'succeed' so the "
                         "success/failure bookkeeping is exercised in both directions")
    ap.add_argument("--weights", default="real", choices=["real", "dummy"],
                    help="which weights weight_init stores. dummy = shape-exact synthetic "
                         "(plumbing on hardware without the 2.23 GB checkpoint); it is NOT "
                         "a fidelity claim and the rollout means nothing.")
    ap.add_argument("--use-run-inference", action="store_true",
                    help="drive the model through run_inference() instead of the "
                         "vision/prefix/denoise sequence. Matches verapulse_test.py "
                         "--stage all exactly, but CANNOT pass the text mask, so the "
                         "prefix counts all 48 text slots as valid.")
    ap.add_argument("--snr", action="store_true",
                    help="--use-run-inference only: run the per-stage SNR gates on every "
                         "inference. Each gate runs the 2.23 GB torch oracle on CPU -- "
                         "minutes per step. For debugging one inference, not for a rollout.")
    ap.add_argument("--strict-gates", action="store_true")
    ap.add_argument("--no-fused-silu", dest="fused_silu", action="store_false",
                    help="prefix MLP: composed silu instead of the fused LALU path")
    ap.add_argument("--image-norm", default="pm1", choices=["pm1", "unit"],
                    help="pixel range fed to the ViT: pm1 = [-1,1] (SigLIP/SmolVLM2 "
                         "mean=std=0.5, the training preprocessing) or unit = [0,1]")
    ap.add_argument("--video-out", default=str(_HERE / "data" / "libero" / "videos"))
    ap.add_argument("--no-video", action="store_true", help="skip mp4 writing")
    args = ap.parse_args()

    print("=" * 74)
    print("  VeraPulse PulseVLA-LIBERO-0.5B  |  LIBERO closed-loop eval")
    print(_POLICY_WARNING)
    print("=" * 74, flush=True)

    max_steps = args.max_steps or _SUITE_MAX_STEPS[args.task_suite]
    np.random.seed(args.seed)

    results_path = pathlib.Path(args.results_out) if args.results_out else (
        _HERE / "data" / "libero" / f"results_verapulse_{args.task_suite}.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    episodes, done_keys = ([], set())
    if args.resume:
        episodes, done_keys = _load_previous(results_path)
        print(f"[eval] --resume: {len(done_keys)} episodes already in {results_path}; "
              f"they will be SKIPPED.", flush=True)

    # ---- environment + task list -------------------------------------------------
    # The LIBERO import is inside this branch so --dry-run never touches MuJoCo/robosuite.
    if args.dry_run:
        n_all_tasks = 10
        get_task = _FakeTask
        get_init_states = lambda tid: [None] * max(1, args.trials)
        make_env = lambda tid: _FakeEnv(
            seed=args.seed + tid,
            succeed_after=None)   # per-trial override below
    else:
        # LIBERO's benchmark.get_task_init_states() torch.load()s PICKLED NUMPY .init
        # files. torch >= 2.6 defaults weights_only=True and refuses them, so every task
        # dies before the sim starts. These files are plain data from the LIBERO clone;
        # restore the pre-2.6 default at this import site only (pi05's shim, same reason).
        import torch as _torch
        if not getattr(_torch.load, "_libero_weights_only_shim", False):
            _orig_load = _torch.load

            def _torch_load_compat(*a, **kw):
                kw.setdefault("weights_only", False)
                return _orig_load(*a, **kw)
            _torch_load_compat._libero_weights_only_shim = True
            _torch.load = _torch_load_compat

        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()[args.task_suite]()
        n_all_tasks = suite.n_tasks
        get_task = suite.get_task
        get_init_states = suite.get_task_init_states

        def make_env(tid):
            t = suite.get_task(tid)
            bddl = (pathlib.Path(get_libero_path("bddl_files"))
                    / t.problem_folder / t.bddl_file)
            e = OffScreenRenderEnv(bddl_file_name=str(bddl),
                                   camera_heights=LIBERO_ENV_RESOLUTION,
                                   camera_widths=LIBERO_ENV_RESOLUTION)
            e.seed(args.seed)
            return e

    n_tasks = min(args.tasks, n_all_tasks) if args.tasks else n_all_tasks

    # ---- backend + preprocessing, built ONCE -------------------------------------
    if args.dry_run:
        pre = None                       # verapulse_test import is FPGA-free, but the
        try:                             # checkpoint may not be present on a dev box;
            pre = _VeraPulsePre(args.image_norm)   # use it when it loads, fake it if not.
            print("[eval] dry-run: using the REAL tokenizer + norm_stats.", flush=True)
        except Exception as e:
            print(f"[eval] dry-run: real preprocessing unavailable ({e!r}); "
                  f"using synthetic tokens/stats. The loop is still exercised.", flush=True)
        backend = _FakeBackend(seed=args.seed)
    else:
        pre = _VeraPulsePre(args.image_norm)
        backend = _FpgaBackend(weights=args.weights, seed=args.seed,
                               use_run_inference=args.use_run_inference,
                               snr=args.snr, strict_gates=args.strict_gates,
                               fused_silu=args.fused_silu)
    print("[eval] backend ready.", flush=True)

    total_ep = sum(1 for _ in episodes)
    total_succ = sum(1 for e in episodes if e.get("success"))

    def _save():
        """Checkpoint AFTER EVERY EPISODE. The box hard-reboots under sustained FPGA
        load with no kernel log, so this file -- not the process -- is the run's memory.
        Written whole each time (tens of KB); a torn write is handled by _load_previous."""
        results_path.write_text(json.dumps({
            "model": "verapulse/pulsevla-libero-0.5b",
            "policy_validated": False,
            "warning": ("action expert ~17 dB below its bf16 floor, executed actions at "
                        "cos 0.97 vs the CPU oracle -- this success rate does NOT "
                        "characterize the model or the accelerator"),
            "backend": "dry-run" if args.dry_run else "fpga",
            "weights": args.weights, "task_suite": args.task_suite, "seed": args.seed,
            "task_start": args.task_start, "tasks": n_tasks, "trials": args.trials,
            "max_steps": max_steps, "replan_steps": args.replan_steps,
            "image_norm": args.image_norm,
            "entry": "run_inference" if args.use_run_inference else "vision+prefix+denoise",
            "successes": total_succ, "episodes_total": total_ep,
            "success_rate": total_succ / max(1, total_ep),
            "episodes": episodes,
        }, indent=2))

    def infer(base_u8, wrist_u8, state_raw, language):
        """One inference: preprocess -> backend -> de-normalize to robot units."""
        if pre is not None:
            images = pre.images(base_u8, wrist_u8)
            ids, mask = pre.tokens(language)
            state32 = pre.state(state_raw)
            chunk = backend.infer(images, ids, mask, state32)
            return pre.unnormalize(chunk)
        # dry-run without a checkpoint: shapes only, same contract.
        images = np.stack([base_u8, wrist_u8]).astype(np.float32) / 127.5 - 1.0
        ids = np.zeros(48, dtype=np.int64)
        mask = np.zeros(48, dtype=bool)
        mask[:12] = True
        return backend.infer(images, ids, mask, np.zeros(32, dtype=np.float32))

    t_run0 = time.perf_counter()
    for task_id in range(args.task_start, args.task_start + n_tasks):
        task = get_task(task_id)
        init_states = get_init_states(task_id)

        # Skip the whole task (and its env construction, which is expensive and leaks an
        # EGL context) when --resume shows every trial already done.
        if all((task_id, tr) in done_keys for tr in range(args.trials)):
            print(f"[eval] task {task_id}: all {args.trials} trials already done "
                  f"(--resume) -- skipping.", flush=True)
            continue

        env = make_env(task_id)
        task_ep, task_succ = 0, 0
        for trial in range(args.trials):
            if (task_id, trial) in done_keys:
                print(f"[eval] task {task_id} trial {trial}: already done -- skipping.",
                      flush=True)
                continue
            if args.dry_run and args.dry_succeed_every:
                # Force a mix of successes and failures so the bookkeeping, the video
                # naming and the checkpoint file are all exercised in both directions.
                env.succeed_after = (args.wait_steps + 25
                                     if (total_ep + 1) % args.dry_succeed_every == 0
                                     else None)

            plan = collections.deque()
            replay, done, t, errored, n_infer = [], False, 0, False, 0
            t_ep = time.perf_counter()
            print(f"[eval] task {task_id} '{task.language}' trial {trial}", flush=True)
            # Isolate the episode: one bad rollout (sim glitch, a one-off device error)
            # is recorded as a failure and the run CONTINUES. Aborting the job would
            # throw away everything a long FPGA run has earned so far.
            try:
                env.reset()
                obs = env.set_init_state(init_states[trial])
                while t < max_steps + args.wait_steps:
                    if t < args.wait_steps:              # let the scene settle
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue
                    # Rotate 180 to match the training-time camera convention, then
                    # resize 256 -> 512 (this model's ViT is 512, unlike pi05's 224).
                    base = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist = np.ascontiguousarray(
                        obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    base = _resize_with_pad(base, MODEL_IMAGE_SIZE)
                    wrist = _resize_with_pad(wrist, MODEL_IMAGE_SIZE)
                    if not args.no_video:
                        replay.append(base)
                    if not plan:
                        # ACTION CHUNKING: infer once, execute --replan-steps actions,
                        # then re-plan. This cadence (not per-step inference) is what
                        # makes a VLA tractable on hardware -- an inference is seconds.
                        state_raw = np.concatenate((
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]))
                        chunk = infer(base, wrist, state_raw, str(task.language))
                        n_infer += 1
                        plan.extend(np.asarray(chunk)[: args.replan_steps])
                    obs, _, done, _ = env.step(np.asarray(plan.popleft()).tolist())
                    if done:
                        break
                    t += 1
            except Exception as _e:
                import traceback
                errored, done = True, False
                print(f"[eval] !! task {task_id} trial {trial} ERRORED at step {t}: "
                      f"{_e!r} -- recording as failure, continuing", flush=True)
                traceback.print_exc()

            task_ep += 1
            total_ep += 1
            done = bool(done)
            if done:
                task_succ += 1
                total_succ += 1
            episodes.append({"task_id": task_id, "trial": trial,
                             "language": str(task.language), "steps": t,
                             "inferences": n_infer,
                             "seconds": round(time.perf_counter() - t_ep, 1),
                             "success": done, "errored": errored})
            _save()   # <- the reboot insurance. Never move this out of the trial loop.
            suffix = "error" if errored else ("success" if done else "failure")
            if not args.no_video and replay and not args.dry_run:
                try:
                    import imageio
                    vdir = pathlib.Path(args.video_out)
                    vdir.mkdir(parents=True, exist_ok=True)
                    imageio.mimwrite(
                        vdir / f"verapulse_{args.task_suite}_t{task_id}_e{trial}_{suffix}.mp4",
                        replay, fps=10)
                except Exception as _ve:
                    # A missing codec must not cost an episode that already ran.
                    print(f"[eval]   (video write skipped: {_ve!r})", flush=True)
            print(f"[eval]   -> {suffix.upper()}  {n_infer} inferences  "
                  f"(running {total_succ}/{total_ep} = "
                  f"{100 * total_succ / max(1, total_ep):.1f}%)", flush=True)
            print(f"[eval]   checkpointed -> {results_path}")
            print(f"[eval]   if the box reboots, resume with:  MUJOCO_GL=egl python "
                  f"{pathlib.Path(__file__).resolve()} --task-suite {args.task_suite} "
                  f"--trials {args.trials} --results-out {results_path} --resume",
                  flush=True)
            del replay, plan
            gc.collect()
        print(f"[eval] task {task_id} success rate: {task_succ}/{task_ep}", flush=True)
        env.close()
        del env
        gc.collect()     # MuJoCo's EGL context is not released by close() alone

    _save()

    # ---- summary ------------------------------------------------------------------
    per_task = collections.defaultdict(lambda: [0, 0])
    for e in episodes:
        per_task[e["task_id"]][1] += 1
        per_task[e["task_id"]][0] += int(bool(e.get("success")))
    print("\n" + "=" * 74)
    print(f"  LIBERO {args.task_suite}  |  backend="
          f"{'dry-run' if args.dry_run else 'fpga'}  tasks={n_tasks}  "
          f"trials/task={args.trials}")
    for tid in sorted(per_task):
        s, n = per_task[tid]
        print(f"    task {tid:>2d}: {s}/{n} = {100 * s / max(1, n):5.1f}%")
    print(f"  OVERALL: {total_succ}/{total_ep} = "
          f"{100 * total_succ / max(1, total_ep):.1f}%   "
          f"({time.perf_counter() - t_run0:.0f}s)")
    print(f"  per-episode results -> {results_path}")
    print(_POLICY_WARNING)
    print("=" * 74)


if __name__ == "__main__":
    main()
