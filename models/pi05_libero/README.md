# pi0.5 + LIBERO on the UnifiedEngine

Closed-loop robot-manipulation validation for the pi0.5 vision-language-action model,
running on the FPGA and paired against a GPU reference on identical episodes.

**Status:** the FPGA completes LIBERO-Spatial task 0 (`done` at step 91) matching the
GPU reference (step 96) on the identical episode, with 30.2 dB action-chunk agreement,
gripper cosine 1.000 and 100% sign agreement.

---

## 1. Setup

Two env files, pick one:

| file | installs | env name |
|---|---|---|
| `models/pi05_libero/pi05_env.yml` | conda, **GPU** torch (cu128) | `libero` |
| `models/pi05_libero/requirements.txt` | pip, **CPU** torch (the FPGA runs the model) | `pi05_libero` |

```bash
conda env create -f models/pi05_libero/pi05_env.yml        # creates env `libero`
conda activate libero

# libero is NOT on PyPI -- still an editable install by path:
pip install -e ~/apex-compute-ML/simple-llm/src/models/pi0_5/openpi_src/third_party/libero
```

`openpi_client` (the obs preprocessing helpers `libero_eval.py` imports) comes from
PyPI's real `openpi-client` package — no local checkout needed.

**`openpi` proper is NOT on PyPI.** The name resolves, but only to a 0.0.0 placeholder
stub ("Reserved package name for future use by Physical Intelligence") with no code in
it — a bare `openpi` requirement installs nothing and then fails confusingly at
`import openpi`. It must come from git:

```bash
pip install "openpi @ git+https://github.com/Physical-Intelligence/openpi"
```

Only the **weight export** prep step needs it (§1.1); the FPGA + sim runtime never
imports it. (`openpi-client` on PyPI *is* real, but it is only the websocket policy
client — it has neither `shared.download.maybe_download` nor
`models.model.restore_params`.)

Checkpoint assets are expected at `~/.cache/openpi/`:
- `openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json`
- `big_vision/paligemma_tokenizer.model`

### 1.1 Weights: `pi05_libero_bin/weights_export/` (prep phase, one-time)

`pi05_libero_bin/weights_export/` (51 tensors, ~13 GB of `.npy` + a `manifest.json`) is what
every backend
loads. It is **regenerable**, not an un-reproducible blob —
`models/pi05_libero/pi05_libero_export_weights.py` rebuilds it from the upstream openpi JAX
checkpoint (`gs://openpi-assets/checkpoints/pi05_libero`, declared in
`pi05_libero_config.json` under `paths.openpi_checkpoint`).

```bash
# needs openpi (jax/orbax) -- run it in a prep env, not the runtime env:
/home/rohit/miniconda3/envs/pi05/bin/python models/pi05_libero/pi05_libero_export_weights.py \
    --out pi05_libero_bin/weights_export_new
```

| flag | does |
|---|---|
| `--out DIR` | export destination (default `pi05_libero_bin/weights_export_new`) |
| `--verify` | read-only: restore the checkpoint and diff it tensor-by-tensor against `--reference` (default `pi05_libero_bin/weights_export`) |
| `--list` | print checkpoint leaf names + shapes, exit, write nothing |
| `--only NAME ...` | restrict export/verify to a subset (subset export skips `manifest.json`) |

Verified: `--verify` reproduces the existing export with **all 51 tensors bit-identical**.
The script **refuses `--out pi05_libero_bin/weights_export`** — the validated reference is not overwritable;
export to a new dir and compare. Weights are stored in the checkpoint's own JAX `(in, out)`
layout; the consumer transposes at load.

Two things to expect: `restore_params` pulls all ~13 GB into RAM before saving, and openpi
drags in jax/jaxlib/orbax (plus `jax-cuda12-plugin`'s own `nvidia-*` wheels, which can
collide with torch's cu128 set) — which is exactly why the FPGA + sim runtime env does not
have it and does not need it.

### Two pins are load-bearing — do not "upgrade" either

| pin | why |
|---|---|
| `numpy<2` (1.26.4) | robosuite/gym 0.25 break on numpy 2.x. This is what lets the **simulator and `user_dma_core` live in ONE process** — no socket, no server/client. |
| `torch==2.8.0+cu128` | the RTX 5070 is Blackwell (sm_120). Stock PyPI torch installs **CPU-only** and silently gives you `cuda=False`. |

### Verify the install

```bash
MUJOCO_GL=egl python models/pi05_libero/libero_eval.py --backend torch --tasks 1 --trials 3
# expect: 3/3 SUCCESS in ~36s, mp4s written to data/libero/videos/
```

`MUJOCO_GL=egl` is required — it renders headless. Without it MuJoCo wants a display.

---

## 2. What LIBERO is

**LIBERO** ("LIfelong learning BEnchmark for RObot manipulation", Liu et al. 2023) is a
simulated manipulation benchmark built on **robosuite → MuJoCo**, using a **Franka Emika
Panda** 7-DoF arm with a parallel-jaw gripper. It is the standard eval for VLA policies
(pi0, pi0.5, OpenVLA).

### The stack

```
libero          task registry, BDDL goal files, 50 preset init states per task
  └─ robosuite  robot + controller (OSC_POSE), camera rendering
      └─ mujoco physics
```

Nothing is replayed from a dataset. **MuJoCo renders every frame live.** The only fixed
input is the initial object layout (`init_states[trial]`); everything the model sees
afterward, it caused.

### Task suites (10 tasks each, 50 human demos per task)

| suite | what varies | tests | max_steps |
|---|---|---|---|
| `libero_spatial` | object placement | spatial reasoning | 220 |
| `libero_object` | which object | object grounding | 280 |
| `libero_goal` | the instruction | language grounding | 300 |
| `libero_10` | long-horizon, multi-step | temporal composition | 520 |

`max_steps` is a **timeout, not a prediction**. Nobody knows how long a task takes — the
episode ends the instant the goal predicate fires (task 0 finishes around step 91–96 of
its 220 budget), or is declared a failure when the budget runs out.

### The 7-D action

Actions go through robosuite's **OSC_POSE** controller — end-effector space, not joints.

| dim | meaning |
|---|---|
| 0–2 | **Δx, Δy, Δz** — EEF position delta |
| 3–5 | **Δroll, Δpitch, Δyaw** — EEF orientation delta (axis-angle) |
| 6 | **gripper** — single scalar, both fingers move symmetrically |

All roughly in `[-1, 1]`. The gripper is 1-D because the jaw is a 1-DoF mechanism; it's
continuous (a position/force command, held every step to keep holding an object), though
trained policies saturate it near ±1. **Negative = open, positive = close.**

Note the arm has **7 joints** but the end-effector has **6 DoF** — the arm is redundant,
and OSC picks a joint solution for you. `7 joints` and `7-D action` are unrelated; a
shape check of `(..., 7)` passes for both.

### Success

Each task ships a **BDDL** file declaring a symbolic goal predicate — e.g.
`(On akita_black_bowl_1 plate_1)`. robosuite evaluates it against MuJoCo state after
**every** `env.step()` and returns `done=True` the instant it holds.

Scoring is **binary per episode, no partial credit** — hovering the bowl 2cm above the
plate at timeout scores 0. Suite score = mean over 10 tasks. No time bonus, no smoothness
or collision penalty.

---

## 3. How an episode runs

```
env.set_init_state(init_states[trial])      deterministic starting layout
10 dummy steps                              let objects settle

loop until done or max_steps:
    agentview 256² + wrist 256²   ← rendered fresh by MuJoCo
      → rotate 180° → resize_with_pad → 224²
    state = eef_pos(3) + axis-angle(3) + gripper_qpos(2)     ← 8 numbers
    prompt = "Task: <lang>, State: <8 ints 0-255>;\nAction: "
      → model → (10,7) normalized → un-normalize via norm_stats q01/q99
    execute --replan-steps rows: env.step(a0), env.step(a1), ...
```

**Two different "10"s, and they are unrelated:**
- **10 denoise steps** — flow-matching Euler integration *inside* the model. The sim never
  sees these. Produces **one** action chunk.
- **10 = action horizon** — the chunk is `(10,7)`: ten consecutive robot commands ≈ 0.5s
  of motion.

Denoising is *sampling*, not *progress*. Reaching the goal takes ~91 timesteps ≈ **9 full
inferences**, each with its own 10 denoise steps, each looking at **new camera frames**.

**The model is memoryless.** No KV is carried between inferences; it has no idea it's on
chunk 5 of 9. The world carries the state — the arm being 3cm from the bowl *is* the
memory of the reaching it already did. pi0.5 also has no separate state tensor: the 8-D
proprioception is discretized to 0–255 ints and spliced into the prompt **text**.

### Closed-loop = the actual validation claim

The model's output becomes its own next input, through a system with state:

```
FPGA → actions → physics → new images → FPGA → ...   (×9)
```

This is strictly stronger than a vector/SNR test:
- **It integrates error.** A small per-inference error moves the arm slightly wrong, so the
  next inference sees a slightly wrong image. The loop has gain; a vector test does not.
- **Physics is the judge, not a golden vector.** "Does the DUT match the reference?" is
  worthless if both are wrong the same way (a gripper sign flip lives in shared
  preprocessing — DUT and golden agree at 50 dB while the robot never grasps). "Is the
  bowl on the plate?" cannot be argued with.
- **It probes off-distribution states** the reference trajectory never visits.

**The sim has no wall clock.** MuJoCo is turn-based: it advances exactly one tick per
`env.step()` and then waits, indefinitely. The FPGA taking 195s/inference produces a
*bit-identical trajectory* to a 5ms engine. Latency costs wall-clock and nothing else —
so correctness proven on slow silicon fully transfers. (This stops being true on a real
robot, where the world does not wait.)

---

## 4. Running it

### GPU reference (the oracle) — ~10s/episode

```bash
MUJOCO_GL=egl python models/pi05_libero/libero_eval.py \
    --backend torch --device cuda --quant if4 --tasks 1 --trials 3
```

### FPGA — ~195s/inference

```bash
screen -dmS pi05_fpga bash -c 'cd /home/rohit/unified-engine && \
  MUJOCO_GL=egl python models/pi05_libero/libero_eval.py \
    --backend fpga --tasks 1 --trials 1 --dump-actions 2>&1 \
    | tee models/pi05_libero/libero_fpga_full.log'
screen -r pi05_fpga     # attach; Ctrl-A D to detach
```

Always run the FPGA under `screen` — a several-hour run must survive the session.

### The paired diff — the real verification

Both backends share the sim, seeding, init states, preprocessing, replan cadence, and
denoise noise, so **inference #0 sees a bit-identical observation in both.** Any
difference there is the backend and nothing else.

```bash
python models/pi05_libero/libero_eval.py --diff-actions \
    data/libero/actions_torch_libero_spatial.npz \
    data/libero/actions_fpga_libero_spatial.npz
```

Reports per-dimension SNR, cosine, and gripper sign agreement.

**READ THE COSINE COLUMN, NOT THE SNR.** A dim the robot isn't using (no wrist rotation
during a straight reach) has a near-zero signal, so its SNR is a ratio of two tiny numbers
— meaningless, and it reads as an alarming low dB while the direction is perfect. Same
trap as the masked-rows-poison-SNR bug.

### Cost model

| | inferences | wall-clock |
|---|---|---|
| 1 episode, success (exits early) | ~9 | **~31 min** |
| 1 episode, failure (full 220 steps) | 22 | **~72 min** |
| 10 tasks × 1 trial | ~142 | **~8 h** |
| 10 tasks × 5 trials (citable) | ~710 | **~38 h** |

Successes are *cheaper* than failures — `done` fires and the loop breaks. Engine build
(~4 min) happens **once** per process; never re-launch per episode.

---

## 5. Gotchas that cost real time

**Fixed noise is the default, and it's a deviation.** Both backends pin the denoise seed
to `RandomState(0)`, making the policy a deterministic function of the observation. That is
what makes the paired FPGA-vs-torch diff valid. But it is **one draw from a stochastic
policy** — real openpi samples fresh noise per call. Use `--fresh-noise` when measuring a
success rate; use the default (fixed) when doing a paired diff. Never quote a number from a
fixed-noise run without saying so.

**Do not benchmark the FPGA against pi0.5's published ~98%.** Suite accuracy is a property
of *pi0.5*; equivalence is a property of *your silicon*. Get the suite number on GPU (8 min)
and spend FPGA time on paired episodes. Comparing two noisy percentages with overlapping
confidence intervals is how you lose a week.

**n=10 is almost all noise.** The 95% CI on 7/10 is roughly [39%, 89%]. Ablations at one
trial per task are not distinguishable from each other — measured: if4/replan10/fixed 6/10,
if4/replan5 7/10, bf16/replan5 7/10, bf16/replan5/fresh 8/10. All the same, statistically.
(Tasks 7 and 9 do fail in *every* config — that part is a real signal.)

**The `datasets path ... does not exist!` warning is benign.** That directory holds LIBERO's
human demo HDF5s. Rollouts don't need them — the sim generates everything. You would only
need them for a teacher-forced open-loop probe test.

**The mp4 is the model's own eye view.** MuJoCo renders `agentview_image` because that *is*
the model's input; the video is just those frames kept in a list. The `_success` /
`_failure` filename suffix comes from the BDDL `done` flag, so `ls` alone tells you the
verdict of a finished run.

**Only 1 of every 10 rendered frames reaches the model.** With `--replan-steps 10` the whole
chunk executes open-loop; the other 9 obs are discarded (they still go into the video).
`--replan-steps 5` doubles the feedback rate — and doubles FPGA cost.
