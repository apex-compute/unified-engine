"""LIBERO evaluation for the pi0.5 UnifiedEngine (FPGA) -- SINGLE PROCESS.

Builds the FPGA engine once, steps the real LIBERO simulator, drives the robot
with the engine's action chunks, and reports the success rate. No socket, no
server/client -- the engine (user_dma_core) and the simulator (robosuite/mujoco)
live in the SAME merged env (see requirements.txt; numpy<2 lets them coexist).

Setup once:
    conda create -n pi05_libero python=3.11 -y && conda activate pi05_libero
    pip install -r models/pi05_libero/requirements.txt
    pip install -e ~/apex-compute-ML/simple-llm/src/models/pi0_5/openpi_src/third_party/libero

Run:
    MUJOCO_GL=egl python models/pi05_libero/libero_eval.py \
        --task-suite libero_spatial --tasks 1 --trials 1 --max-steps 60   # smoke test
    MUJOCO_GL=egl python models/pi05_libero/libero_eval.py \
        --task-suite libero_spatial --trials 10                            # real run

The engine predicts a 10x7 chunk; LIBERO replans every --replan-steps (5), so
only the first 5 rows of each chunk are executed before re-querying. Success =
the task's BDDL goal predicate fires (done=True) before max_steps.
"""
import argparse
import collections
import functools
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-suite", default="libero_spatial", choices=list(_SUITE_MAX_STEPS))
    ap.add_argument("--tasks", type=int, default=None,
                    help="cap #tasks (default: None = all 10 tasks in the suite)")
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
    args = ap.parse_args()

    # torch.load: LIBERO's get_task_init_states() loads numpy-pickled files that
    # torch 2.x rejects under the weights_only=True default. These are trusted.
    import torch
    torch.load = functools.partial(torch.load, weights_only=False)

    import imageio
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools

    # --- build the FPGA engine ONCE (weight_init + tensor_init) ---
    import pi05_libero_test as M
    print("[eval] building FPGA engine (weight_init + tensor_init)...", flush=True)
    ue = M.Pi05Libero_UnifiedEngine()
    M.init_hang_prevention(ue)
    ue.weight_init()
    ue.tensor_init(M._CFG["defaults"].get("max_seq", 512))
    pre = _Pi05Pre()
    print("[eval] engine ready.", flush=True)

    def infer(base_u8, wrist_u8, state_8d, language):
        images = pre.images(base_u8, wrist_u8)
        toks = pre.prompt_tokens(state_8d, language)
        actions = ue.run_inference(images, toks)          # (10,7) normalized
        return pre.unnormalize(actions[:, :7])

    max_steps = args.max_steps or _SUITE_MAX_STEPS[args.task_suite]
    np.random.seed(args.seed)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    n_tasks = min(args.tasks, task_suite.n_tasks) if args.tasks else task_suite.n_tasks
    video_dir = pathlib.Path(args.video_out)
    video_dir.mkdir(parents=True, exist_ok=True)

    total_ep, total_succ = 0, 0
    for task_id in range(n_tasks):
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
            if done:
                task_succ += 1
                total_succ += 1
            suffix = "success" if done else "failure"
            imageio.mimwrite(
                video_dir / f"{args.task_suite}_t{task_id}_e{trial}_{suffix}.mp4",
                [np.asarray(x) for x in replay], fps=10)
            print(f"[eval]   -> {'SUCCESS' if done else 'failure'}  "
                  f"(running {total_succ}/{total_ep} = {100*total_succ/total_ep:.1f}%)", flush=True)
        print(f"[eval] task {task_id} success rate: {task_succ}/{task_ep}", flush=True)

    print("\n" + "=" * 56)
    print(f"  LIBERO {args.task_suite}  |  trials/task={args.trials}  tasks={n_tasks}")
    print(f"  SUCCESS RATE: {total_succ}/{total_ep} = {100*total_succ/max(1,total_ep):.1f}%")
    print("=" * 56)


if __name__ == "__main__":
    main()
