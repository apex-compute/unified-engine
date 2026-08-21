# LIBERO evaluation guide (pi0.5 on the UnifiedEngine)

Long-form companion to [`../README.md`](../README.md). The README carries the 5-command
quickstart, the flag reference and the troubleshooting table; everything here is the
background you need only when you are running or interpreting closed-loop episodes.

---

## Environment bring-up, in detail

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
