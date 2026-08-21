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
`utility/libero_eval.py --diff-actions` -- see
[`document/libero_eval_guide.md`](document/libero_eval_guide.md).

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


## 1. Requirements & weights

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


### Weights

First run of `pi05_test.py` downloads the checkpoint and exports it — a one-time
cost of several minutes, cached into `models/pi05/pi05_bin/`.

For what `setup_env.sh` actually does (and why none of its steps can be skipped), what
LIBERO is, how an episode runs, and how to pick tasks / read the eval outputs, see
[`document/libero_eval_guide.md`](document/libero_eval_guide.md).

---

## 2. Correctness checks

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


## 3. Flag reference

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


## 4. Gotchas that cost real time

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


## 5. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'libero'` after a successful `pip install` | `find_packages()` found nothing; the wheel is metadata-only | `libero_requirements.txt` uses `editable_mode=compat`, not a plain install |
| `pip install -e` succeeds, import still fails | PEP 660 finder has `MAPPING = {}` | same fix — `editable_mode=compat` |
| `AttributeError: 'NoneType' object has no attribute 'glGetError'` | `PYOPENGL_PLATFORM=osmesa` but no `libOSMesa` on the loader path | run `setup_env.sh`, set `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` |
| `libEGL warning: failed to open /dev/dri/card0: Permission denied` | not in the `render`/`video` groups | use OSMesa (`setup_env.sh`), or `sudo usermod -aG render,video $USER` + re-login |
| `Cannot initialize a EGL device display … PLATFORM_DEVICE extension` | robosuite needs EGL device platform; surfaceless is not enough | same as above |
| `_pickle.UnpicklingError: Weights only load failed` | torch ≥2.6 flipped `weights_only` to True; LIBERO's init states are pickled numpy | already shimmed in `libero_eval.py`; add `torch.load = functools.partial(torch.load, weights_only=False)` for standalone scripts |
| `[Warning]: datasets path … does not exist!` | LIBERO's demo HDF5 datasets absent | **harmless** — evaluation needs only bddl files and init states |
| `Exception ignored in: <function GLContext.__del__>` at exit | interpreter-shutdown ordering | **harmless** — cosmetic, appears after results are written |
| vision asserts above its ceiling | row-sharding needs 64-row blocks; `S=256`/slot is 4 | use `--engines max` — it reads `STAGE_MAX_ENGINES` instead of assuming |
| overall SNR far below per-slot SNR | masked placeholder slot folded into the pool | read per-slot numbers, never the pool |
| `skipping bin dump: multi-engine run` | expected on any sharded run | `--engines 1` if you actually need bins |

### Splitting a long GPU run

MuJoCo's EGL renderer segfaults after ~6 `OffScreenRenderEnv` creations when CUDA is
also active, so run the **torch/GPU** eval in chunks of ≤5 tasks using `--task-start`.
The FPGA backend does no CUDA compute and is unaffected.
