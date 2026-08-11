# Running pi0.5 + LIBERO on the UnifiedEngine

Everything you need to go from a fresh machine to a closed-loop robot episode with a
video at the end. For *what the model is* and *why the validation is designed this
way*, see [README.md](README.md) — this file is only "how do I run it".

- [1. One-time setup](#1-one-time-setup) — one command
- [2. Single inference (fast)](#2-single-inference-fast)
- [3. Closed-loop episodes (the real thing)](#3-closed-loop-episodes-the-real-thing)
- [4. Correctness checks](#4-correctness-checks)
- [5. Flag reference](#5-flag-reference)
- [6. Troubleshooting](#6-troubleshooting)

---

## 1. One-time setup

### One command

```bash
bash models/pi05_libero/setup_env.sh
```

That creates the `pi05_libero` conda env from `pi05_libero_env.yml`, installs the one
library conda cannot supply, seeds LIBERO's interactive first-run prompt, and verifies
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

First run of `pi05_libero_test.py` downloads the checkpoint and exports it — a one-time
cost of several minutes, cached into `models/pi05_libero/pi05_libero_bin/`.

---

## 2. Single inference (fast)

One observation in, one `(10, 7)` action chunk out. No simulator. This is the loop you
want while iterating.

```bash
python models/pi05_libero/pi05_libero_test.py --engines max
```

`--engines max` applies each stage's own ceiling from `STAGE_MAX_ENGINES`, which is
**8/8/8 today** — vision reached 8 once it became 2D (4 row groups × 2 K lanes); it
capped at 4 while row-sharded only, since `S=256`/slot is just 4 blocks of 64. So `max`
and a flat `--engines 8` currently agree, but that is not guaranteed: any stage whose
ceiling drops below 8 makes them diverge, which is why `max` reads the dict. Prefer it.

Roughly 37 s of execution plus ~13 s of one-time compile. Single-engine
(`--engines 1`) is ~5× slower but is the control you need when a result looks wrong.

---

## 3. Closed-loop episodes (the real thing)

```bash
screen -dmS pi05_obj bash -c 'cd /home/rohit/unified-engine && \
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  LD_LIBRARY_PATH=$CONDA_PREFIX/lib \
  python models/pi05_libero/libero_eval.py \
    --backend fpga --task-suite libero_object --tasks 1 --trials 1 \
    --engines max --dump-actions 2>&1 \
    | tee models/pi05_libero/libero_fpga_object_t0.log'

screen -r pi05_obj      # attach; Ctrl-A D to detach
```

**Always run under `screen`.** A multi-episode run is hours and must survive the
session dropping.

### Picking a task

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

### Suggested runs

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
single-inference path (§2) and edit `defaults.prompt` in `pi05_libero_config.json` —
you get an action chunk against the sample images, but no robot and no video.

### Outputs

| what | where |
|---|---|
| video (one per episode) | `models/pi05_libero/data/libero/videos/{backend}_{suite}_t{task}_e{trial}_{success\|failure\|error}.mp4` |
| results JSON (checkpointed every episode) | `models/pi05_libero/data/libero/results_{backend}_{suite}.json` |
| action dumps (`--dump-actions`) | `models/pi05_libero/data/libero/actions_{backend}_{suite}.npz` |

Videos carry a **prompt overlay in the top-left**: `[fpga/if4-hw] pick up the alphabet
soup and place it in the basket`, yellow on a translucent black bar, 2× upscaled for
legibility. The overlay is applied to copies at write time and never to the frames fed
to the model.

### Cost

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

### Comparing two runs

Both backends share the sim, seeding, init states, preprocessing, replan cadence and
denoise noise, so **inference #0 sees a bit-identical observation in both**. Any
difference there is the backend and nothing else.

```bash
python models/pi05_libero/libero_eval.py --diff-actions \
    models/pi05_libero/data/libero/actions_torch_libero_object.npz \
    models/pi05_libero/data/libero/actions_fpga_libero_object.npz
```

**Read the cosine column, not the SNR.** A dimension the robot is barely using has a
near-zero signal, so its SNR is a ratio of two tiny numbers — meaningless, and it reads
as an alarming low dB while the direction is perfect.

---

## 4. Correctness checks

Ordered cheapest-first. Each isolates one stage.

```bash
# vision encoder vs CPU IF4 reference (~7s) -- per-slot SNR
python models/pi05_libero/pi05_libero_test.py --engines max --verify-vision

# full (10,7) action chunk vs pi05_torch_ref on CPU, matched quant/noise/inputs
# also prints prefix K/V SNR per layer. Slow: the reference runs on CPU.
python models/pi05_libero/pi05_libero_test.py --engines max --verify-denoise

# on-device timestep-embedding MLP vs exact host oracle
python models/pi05_libero/pi05_libero_test.py --engines max --check-cond-table

# localize the first diverging op in the action expert (~75s with --debug)
python models/pi05_libero/pi05_libero_test.py --debug --probe-step0
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

## 5. Flag reference

### Engine sharding (both entry points)

| flag | effect |
|---|---|
| `--engines max` | each stage's own ceiling from `STAGE_MAX_ENGINES` (8/8/8 today). **The recommended way to run fully sharded.** |
| `--engines N` | flat N (1–8) for every stage. `N=1` is the single-engine control. |
| `--vis_4` / `--pref_8` / `--dns_8` | force one stage, composes with and overrides `--engines` |

Defaults differ deliberately: `pi05_libero_test.py` defaults to **1** (so the
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

## 6. Troubleshooting

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
