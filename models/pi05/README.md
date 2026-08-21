# pi0.5 + LIBERO on the UnifiedEngine

Closed-loop robot-manipulation validation for the pi0.5 vision-language-action model,
running on the FPGA and paired against a GPU reference on identical episodes.

**Status:** the FPGA completes LIBERO-Spatial task 0 (`done` at step 91) matching the
GPU reference (step 96) on the identical episode, with 30.2 dB action-chunk agreement,
gripper cosine 1.000 and 100% sign agreement.

## Quickstart — 5 commands

Everything below is run from the **repo root**. Commands 1-2 are one-time; 3 is the
single FPGA inference; 4-5 are the closed-loop LIBERO tests.

```bash
# 1. env: FPGA inference only (no simulator). Skip if the repo env is already active.
pip install -r models/pi05/pi05_requirements.txt

# 2. env: add the LIBERO simulator (conda env + robosuite/MuJoCo + OSMesa, no sudo).
#    ONLY needed for commands 4-5. Skip it if you just want command 3.
bash models/pi05/setup_env.sh && conda activate pi05_libero

# 3. SINGLE INFERENCE on the FPGA -- prints a (10,7) action chunk. ~14.5s.
#    First run also downloads the checkpoint and exports weights (~13 GB, once).
python models/pi05/pi05_test.py --engines max

# 4. CLOSED-LOOP episode on the FPGA, under screen (hours -- must survive a dropped
#    session). Writes a rollout video + per-episode results JSON under models/pi05/data/.
screen -dmS pi05 bash -c 'MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  LD_LIBRARY_PATH=$CONDA_PREFIX/lib python models/pi05/utility/libero_eval.py \
  --backend fpga --task-suite libero_spatial --task-start 0 --tasks 1 --trials 1 \
  --engines max --dump-actions 2>&1 | tee models/pi05/libero_fpga.log'

# 5. watch it (Ctrl-A D detaches without killing the run)
screen -r pi05
```

**Correctness gate.** Command 3 must print finite numbers -- `nan=False inf=False`.
If it prints NaN, run `--probe-step0`: it diffs every action-expert intermediate
against a torch reference fed the FPGA's own prefix K/V and prints an SNR per op, so
the first line that craters names the broken op.

```bash
python models/pi05/pi05_test.py --engines max --probe-step0
```

To compare an FPGA episode against the GPU reference on the identical episode, re-run
command 4 with `--backend torch`, then diff the two action dumps with
`utility/libero_eval.py --diff-actions` (§4).

---

### Layout

| path | what |
|---|---|
| `pi05_test.py` | the engine model — compile + single inference. The entry point. |
| `pi05_config.json` | model manifest (geometry, DRAM map, `paths.*`) |
| `pi05_sample_meta.json` | the sample observation's raw 8-D robot state (+ its task id) |
| `pi05_requirements.txt` | FPGA-inference deps (no simulator) |
| `libero_requirements.txt` | the above **plus** the LIBERO simulator stack |
| `utility/libero_eval.py` | closed-loop LIBERO rollouts (`--backend fpga\|torch`) + `--diff-actions` |
| `utility/pi05_torch_ref.py` | CPU/GPU reference implementation (the numeric oracle) |
| `utility/pi05_jax_oracle.py` | JAX/openpi oracle, for checking the reference itself |
| `utility/pi05_weight_export.py` | checkpoint → `pi05_bin/weights_export/` (prep phase) |
| `utility/pi05_ckpt_noopenpi.py` | orbax checkpoint reader, so the export works without openpi |
| `utility/compare_bf16_if4.py` | bf16-vs-IF4 quantization sweep on the reference |
| `setup_env.sh` | one-command conda env + LIBERO + OSMesa bring-up |
| `document/` | the writeup (`main.tex`) and the episode videos it cites |

The sample observation is two PNGs in the repo-root `test_samples/`
(`pi05_libero_base.png`, `pi05_libero_wrist.png`) plus `pi05_sample_meta.json` here.
That JSON is the **non-image half of one LIBERO observation**: `state_example`, the 8
raw proprioception numbers the sim reported on the frame those PNGs were captured on
(EEF position ×3, EEF orientation as axis-angle ×3, gripper finger qpos ×2), and
`task_index`, which task it came from. pi0.5 has no separate state input — the loader
normalizes these against the checkpoint's `norm_stats` q01/q99, discretizes them to
ints 0–255 and splices them into the prompt **text**, so without this file there is no
prompt to run. Together the three files are one complete observation, so
`pi05_test.py` runs a single inference with **no LIBERO, robosuite, or MuJoCo installed**.
Only `utility/libero_eval.py` needs the simulator.

---

## 1. One-time setup

### Which requirements file?

Two files, layered — there is no third:

| file | what it is | use it when |
|---|---|---|
| `pi05_requirements.txt` | FPGA inference only: engine base + the one-time weight export (`jax`/`orbax`/`flax`). No simulator. | running `pi05_test.py`. This is all it needs. |
| `libero_requirements.txt` | `-r pi05_requirements.txt` plus robosuite/mujoco/LIBERO and friends. | running `utility/libero_eval.py` (closed-loop episodes). Installed by `setup_env.sh`. |

`libero_requirements.txt` is **not sufficient on its own** — headless rendering needs
system GL libraries (`libOSMesa`, `mesalib`, `libglu`) that no requirements file can
install. That is what `setup_env.sh` is for.

**`pi05_test.py` alone needs far less.** It imports only `numpy<2`, `torch`,
`pillow`, `sentencepiece` and this repo's own modules, so `pi05_requirements.txt` (or
even the repo-root `requirements.txt`) already covers a single inference. The extra
weight in `libero_requirements.txt` is for two things you may not need:

- **the one-time weight export** — `jax` + `jaxlib` + `flax` + `orbax-checkpoint`,
  used once on first run and never imported again;
- **`utility/libero_eval.py`** — robosuite/MuJoCo/LIBERO/OSMesa, the whole simulator.

If you only want an action chunk out of the sample observation, `pip install -r
requirements.txt` plus the four jax packages is enough.

### One command

```bash
bash models/pi05/setup_env.sh
```

That creates the `pi05_libero` conda env (python 3.11 + mesalib + libglu), pip-installs
`libero_requirements.txt`, installs the one library neither can supply (`libOSMesa`), seeds LIBERO's interactive first-run prompt, and verifies
the whole thing renders. It is idempotent — safe to re-run, and it updates an existing
env rather than recreating it. **No sudo required.**

Then, in every shell you run from:

```bash
conda activate pi05_libero
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
```

A successful setup prints:

```
  libero      OK  /path/to/LIBERO/libero/libero/__init__.py
  benchmark   OK  libero_object has 10 tasks
  osmesa      OK  rendered (256, 256, 3), sim steps
  task 0      "pick up the alphabet soup and place it in the basket"
SETUP COMPLETE
```

Re-run just the checks any time with `setup_env.sh --verify-only`.

### What it is doing, and why you can't skip it

One conda env holds everything: the FPGA model **and** the LIBERO simulator run in the
same process — no socket, no server/client. CPU-only; no CUDA device required. The pins
in the yaml are load-bearing (`numpy<2` is what lets robosuite and `user_dma_core`
coexist in one interpreter). Three things are sharper than they look:

**LIBERO cannot be pip-installed normally.** `pip install git+…LIBERO` reports success
and installs a ~5.5 KB *metadata-only* wheel — LIBERO is a PEP 420 namespace package
with no `libero/__init__.py`, so its `setup.py`'s `find_packages()` returns nothing.
Plain `pip install -e` is *also* broken: PEP 660 generates a finder with
`MAPPING = {}`. The yaml therefore pins `-e … --config-settings editable_mode=compat`,
which falls back to the legacy `.pth` mechanism and puts the project root on
`sys.path`. **Never trust pip's success message here** — `import libero.libero` is the
only real check, which is why the script does exactly that.

**OSMesa is not on conda-forge.** `mesalib` ships GL/GLX/EGL and the llvmpipe software
rasterizer but no `libOSMesa` (verified absent in both 25.2.8 and 26.1.6), and there is
no standalone `osmesa` package. EGL is not a substitute: MuJoCo's own `Renderer` can go
surfaceless, but robosuite uses `EGLGLContext`, which needs the `PLATFORM_DEVICE`
extension and therefore `/dev/dri` — denied unless you are in the `render`/`video`
groups, and `LIBGL_ALWAYS_SOFTWARE` will not save you. The script pulls Ubuntu's
`libosmesa6` via `apt-get download` (a plain download, not an install), extracts it, and
drops `libOSMesa.so.8` into `$CONDA_PREFIX/lib`. No missing transitive deps.

> **If you have a usable render node** (`ls -l /dev/dri` shows `renderD128` and you are
> in the `render` group), skip OSMesa and use `MUJOCO_GL=egl` — it is faster. Add
> yourself with `sudo usermod -aG render,video $USER` and re-login.

**torch ≥ 2.6 breaks LIBERO's init states.** `benchmark.get_task_init_states()` loads
pickled numpy via `torch.load`, which now defaults to `weights_only=True` and refuses
them. `libero_eval.py` installs a scoped shim; standalone scripts need their own
(`torch.load = functools.partial(torch.load, weights_only=False)`).

### Weights

First run of `pi05_test.py` downloads the checkpoint and exports it — a one-time
cost of several minutes, cached into `models/pi05/pi05_bin/`.

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

### Single inference (fast)

One observation in, one `(10, 7)` action chunk out. No simulator. This is the loop you
want while iterating.

```bash
python models/pi05/pi05_test.py --engines max
```

`--engines max` applies each stage's own ceiling from `STAGE_MAX_ENGINES`, which is
**8/8/8 today** — vision reached 8 once it became 2D (4 row groups × 2 K lanes); it
capped at 4 while row-sharded only, since `S=256`/slot is just 4 blocks of 64. So `max`
and a flat `--engines 8` currently agree, but that is not guaranteed: any stage whose
ceiling drops below 8 makes them diverge, which is why `max` reads the dict. Prefer it.

Roughly 37 s of execution plus ~13 s of one-time compile. Single-engine
(`--engines 1`) is ~5× slower but is the control you need when a result looks wrong.

### Closed-loop episodes (the real thing)

```bash
screen -dmS pi05_obj bash -c 'cd /home/rohit/unified-engine && \
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  LD_LIBRARY_PATH=$CONDA_PREFIX/lib \
  python models/pi05/utility/libero_eval.py \
    --backend fpga --task-suite libero_object --tasks 1 --trials 1 \
    --engines max --dump-actions 2>&1 \
    | tee models/pi05/libero_fpga_object_t0.log'

screen -r pi05_obj      # attach; Ctrl-A D to detach
```

**Always run under `screen`.** A multi-episode run is hours and must survive the
session dropping.

#### Picking a task

`--task-start N --tasks 1` runs exactly one task. Every suite has 10 (ids 0–9).

**`libero_spatial`** (220 steps) — same bowl every time, only the *spatial phrase*
changes, so it isolates language grounding. **Task 0 is the one the 30.2 dB baseline
was measured on** (`done` at step 91), which makes it the best regression check.

```
0  pick up the black bowl between the plate and the ramekin and place it on the plate
1  pick up the black bowl next to the ramekin and place it on the plate
2  pick up the black bowl from table center and place it on the plate
3  pick up the black bowl on the cookie box and place it on the plate
4  pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate
5  pick up the black bowl on the ramekin and place it on the plate
6  pick up the black bowl next to the cookie box and place it on the plate
7  pick up the black bowl on the stove and place it on the plate
8  pick up the black bowl next to the plate and place it on the plate
9  pick up the black bowl on the wooden cabinet and place it on the plate
```

**`libero_object`** (280 steps) — same scene and motion, only the *target object*
changes, so it isolates object recognition. Most gripper-diagnostic: every task is a
grasp-and-place into a basket.

```
0  pick up the alphabet soup and place it in the basket
1  pick up the cream cheese and place it in the basket
2  pick up the salad dressing and place it in the basket
3  pick up the bbq sauce and place it in the basket
4  pick up the ketchup and place it in the basket
5  pick up the tomato sauce and place it in the basket
6  pick up the butter and place it in the basket
7  pick up the milk and place it in the basket
8  pick up the chocolate pudding and place it in the basket
9  pick up the orange juice and place it in the basket
```

**`libero_goal`** (300 steps) — same objects, different *goal*. The only suite with
non-grasp verbs (push, turn on, open), so it exercises behaviours the other three
never touch.

```
0  open the middle drawer of the cabinet
1  put the bowl on the stove
2  put the wine bottle on top of the cabinet
3  open the top drawer and put the bowl inside
4  put the bowl on top of the cabinet
5  push the plate to the front of the stove
6  put the cream cheese in the bowl
7  turn on the stove
8  put the bowl on the plate
9  put the wine bottle on the rack
```

**`libero_10`** (520 steps) — long-horizon, two objects or two stages per task. Budget
~32 min per episode at max engines; use these last.

```
0  put both the alphabet soup and the tomato sauce in the basket
1  put both the cream cheese box and the butter in the basket
2  turn on the stove and put the moka pot on it
3  put the black bowl in the bottom drawer of the cabinet and close it
4  put the white mug on the left plate and put the yellow and white mug on the right plate
5  pick up the book and place it in the back compartment of the caddy
6  put the white mug on the plate and put the chocolate pudding to the right of the plate
7  put both the alphabet soup and the cream cheese box in the basket
8  put both moka pots on the stove
9  put the yellow and white mug in the microwave and close it
```

#### Suggested runs

```bash
# regression check -- the task the 30.2 dB / step-91 baseline came from
--task-suite libero_spatial --task-start 0 --tasks 1 --trials 1

# gripper stress -- grasp and place into a basket
--task-suite libero_object --task-start 4 --tasks 1 --trials 1

# non-grasp behaviour -- no pick-and-place at all
--task-suite libero_goal --task-start 7 --tasks 1 --trials 1

# whole suite, one trial each (~10 episodes)
--task-suite libero_object --trials 1
```

There is no free-text prompt in the loop: the sim needs a BDDL scene with matching
objects, so the prompt comes from the benchmark. For arbitrary text use the
single-inference path (§2) and edit `defaults.prompt` in `pi05_config.json` —
you get an action chunk against the sample images, but no robot and no video.

#### Outputs

| what | where |
|---|---|
| video (one per episode) | `models/pi05/data/libero/videos/{backend}_{suite}_t{task}_e{trial}_{success\|failure\|error}.mp4` |
| results JSON (checkpointed every episode) | `models/pi05/data/libero/results_{backend}_{suite}.json` |
| action dumps (`--dump-actions`) | `models/pi05/data/libero/actions_{backend}_{suite}.npz` |

Videos carry a **prompt overlay in the top-left**: `[fpga/if4-hw] pick up the alphabet
soup and place it in the basket`, yellow on a translucent black bar, 2× upscaled for
legibility. The overlay is applied to copies at write time and never to the frames fed
to the model.

#### Cost

One inference produces 10 actions. The loop consumes them one `env.step()` at a time
and only re-queries when the queue empties, so **inferences = steps ÷ `--replan-steps`**.
An episode ends early the moment `done` fires, or runs the full step budget.

At `--engines max`, measured: **37.0 s** for the first inference's execution plus
**11.9 s** of one-time compile, then **36.9 s** each thereafter.

| | inferences | wall-clock |
|---|---|---|
| best case, `done` around step ~90 | ~9 | **~6 min** |
| full budget, `libero_spatial` (220) | 22 | ~14 min |
| full budget, `libero_object` (280) | 28 | **~17 min** |
| full budget, `libero_10` (520) | 52 | ~32 min |
| 10 tasks × 1 trial, `libero_object` | ≤280 | ≤3 h |
| 10 tasks × 5 trials (citable ±13%) | ≤1400 | ≤14 h |

Add ~2 min of engine build (`weight_init` unpacks ~1.6 GB of params) once per process,
never per episode. At `--engines 1`, multiply the inference numbers by ~5.

**Measured reference run** — `libero_object` task 0, `--engines max`: SUCCESS at step
280, 28 inferences, **~18 min** wall clock for the episode. The FPGA was ~97% of that;
all 280 sim steps plus both camera renders through OSMesa came to ~35 s total, so the
software renderer is effectively free.

Note a success is only cheaper than a failure if `done` fires early — that run
succeeded on the *last* step and cost the same as a failure would have.

> **The sim has no wall clock.** MuJoCo advances exactly one tick per `env.step()` and
> then waits. A 37 s/inference engine produces a *bit-identical trajectory* to a 5 ms
> one — latency costs wall-clock and nothing else, so correctness proven on slow
> silicon fully transfers. (This stops being true on a real robot.)

#### Comparing two runs — the real verification

Both backends share the sim, seeding, init states, preprocessing, replan cadence and
denoise noise, so **inference #0 sees a bit-identical observation in both**. Any
difference there is the backend and nothing else.

```bash
python models/pi05/utility/libero_eval.py --diff-actions \
    models/pi05/data/libero/actions_torch_libero_object.npz \
    models/pi05/data/libero/actions_fpga_libero_object.npz
```

**Read the cosine column, not the SNR.** A dimension the robot is barely using has a
near-zero signal, so its SNR is a ratio of two tiny numbers — meaningless, and it reads
as an alarming low dB while the direction is perfect.

---

## 5. Correctness checks

Ordered cheapest-first. Each isolates one stage.

```bash
# vision encoder vs CPU IF4 reference (~7s) -- per-slot SNR
python models/pi05/pi05_test.py --engines max --verify-vision

# full (10,7) action chunk vs pi05_torch_ref on CPU, matched quant/noise/inputs
# also prints prefix K/V SNR per layer. Slow: the reference runs on CPU.
python models/pi05/pi05_test.py --engines max --verify-denoise

# on-device timestep-embedding MLP vs exact host oracle
python models/pi05/pi05_test.py --engines max --check-cond-table

# localize the first diverging op in the action expert (~75s with --debug)
python models/pi05/pi05_test.py --debug --probe-step0
```

**Masked slots poison pooled SNR.** LIBERO supplies 2 real cameras; the 3rd slot is an
all-zero placeholder that is `-inf`-masked out of attention. Its HW rows are *not* an
encode — `run_vision` overwrites them with a finite `1e-6` placeholder — while any CPU
oracle encodes that image for real. The two are structurally incomparable and score
~0 dB by construction. `--verify-vision` reports them separately and excludes them from
the overall; the prefix K/V check does **not** yet, so its pooled number is pessimistic
by roughly the masked fraction (256 of 832 rows).

### Known-good baseline

| metric | good | notes |
|---|---|---|
| action-chunk agreement, FPGA vs GPU reference | **30.2 dB** | LIBERO-Spatial task 0 |
| gripper cosine / sign agreement | 1.000 / 100% | |
| episode completion | `done` @ step 91 (ref: 96) | identical episode |

If `--verify-denoise` reports substantially below 30 dB overall, something has
regressed — compare against these numbers rather than eyeballing the chunk.

---

## 6. Flag reference

### Engine sharding (both entry points)

| flag | effect |
|---|---|
| `--engines max` | each stage's own ceiling from `STAGE_MAX_ENGINES` (8/8/8 today). **The recommended way to run fully sharded.** |
| `--engines N` | flat N (1–8) for every stage. `N=1` is the single-engine control. |
| `--vis_4` / `--pref_8` / `--dns_8` | force one stage, composes with and overrides `--engines` |

Defaults differ deliberately: `pi05_test.py` defaults to **1** (so the
single-engine path stays the byte-identical baseline), `libero_eval.py` defaults to
**`max`** (nobody wants a 5× slower episode by accident).

Multi-engine runs cannot dump `.bin` program blobs — the bins hold only engine 0's
programs. Use `--engines 1` if you need bins.

### `libero_eval.py`

| flag | default | notes |
|---|---|---|
| `--backend` | `torch` | use `fpga` for hardware |
| `--task-suite` | `libero_spatial` | see the table above |
| `--tasks` / `--task-start` | all / 0 | cap and offset the task range |
| `--trials` | 2 | rollouts per task; 5 for a citable ±13% |
| `--replan-steps` | 10 | actions consumed per chunk before re-querying |
| `--dump-actions` | off | record every inference's exact (input, output) |
| `--no-video` | off | skip mp4 writing |
| `--quant` | `bf16` | `--backend torch` only; use `if4` to match the FPGA |

---

## 7. Gotchas that cost real time

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

---

## 8. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'libero'` after a successful `pip install` | `find_packages()` found nothing; the wheel is metadata-only | §1 — `editable_mode=compat`, not a plain install |
| `pip install -e` succeeds, import still fails | PEP 660 finder has `MAPPING = {}` | §1 — same fix |
| `AttributeError: 'NoneType' object has no attribute 'glGetError'` | `PYOPENGL_PLATFORM=osmesa` but no `libOSMesa` on the loader path | §1 — run `setup_env.sh`, set `LD_LIBRARY_PATH` |
| `libEGL warning: failed to open /dev/dri/card0: Permission denied` | not in the `render`/`video` groups | use OSMesa (§1), or `sudo usermod -aG render,video $USER` + re-login |
| `Cannot initialize a EGL device display … PLATFORM_DEVICE extension` | robosuite needs EGL device platform; surfaceless is not enough | same as above |
| `_pickle.UnpicklingError: Weights only load failed` | torch ≥2.6 flipped `weights_only` to True; LIBERO's init states are pickled numpy | already shimmed in `libero_eval.py`; add `torch.load = functools.partial(torch.load, weights_only=False)` for standalone scripts |
| `[Warning]: datasets path … does not exist!` | LIBERO's demo HDF5 datasets absent | **harmless** — evaluation needs only bddl files and init states |
| `Exception ignored in: <function GLContext.__del__>` at exit | interpreter-shutdown ordering | **harmless** — cosmetic, appears after results are written |
| vision asserts above its ceiling | row-sharding needs 64-row blocks; `S=256`/slot is 4 | use `--engines max` — it reads `STAGE_MAX_ENGINES` instead of assuming |
| overall SNR far below per-slot SNR | masked placeholder slot folded into the pool | see §4 — read per-slot numbers |
| `skipping bin dump: multi-engine run` | expected on any sharded run | `--engines 1` if you actually need bins |

### Splitting a long GPU run

MuJoCo's EGL renderer segfaults after ~6 `OffScreenRenderEnv` creations when CUDA is
also active, so run the **torch/GPU** eval in chunks of ≤5 tasks using `--task-start`.
The FPGA backend does no CUDA compute and is unaffected.
