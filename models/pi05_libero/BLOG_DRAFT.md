pi0.5 libero all performance metrics and unified engine throughput results.
With setup, concise walkthrough.

# Setup:

Create a new environment:
```
python -m venv pi
source ./pi/bin/activate
```

```
pip install -r models/pi05_libero/pi_requirements.txt
pip install -e ~/apex-compute-ML/simple-llm/src/models/pi0_5/openpi_src/third_party/libero
```
Check:
```
printf 'N\n' | python -c "import sys; sys.path.insert(0,'~/apex-compute-ML/simple-llm/src/models/pi0_5/openpi_src/third_party/libero'); import libero.libero"
```

# Closed Loop Benchmark Modes




# pi0.5 on the FPGA: closed-loop LIBERO benchmarking (draft)






## 1. Dependencies & Setup

pi0.5 runs in a single merged Python process that hosts both the LIBERO/robosuite
simulator and the FPGA inference path — no client/server split. LIBERO itself isn't
vendored; it's an editable install from a separate `openpi` checkout (`openpi` is
git-only — the PyPI package is an empty placeholder). The stack is
`libero -> robosuite -> mujoco`, plus `bddl` for task/goal definitions. One pin is
load-bearing: `numpy<2`, without which `robosuite`/`gym==0.25.2` break — and it's
specifically what lets the simulator and the hardware DMA core share one interpreter.
Model weights come from the upstream openpi JAX checkpoint (not HuggingFace), exported
once to a 13GB bf16 tensor dump that the FPGA build compiles from.

### Environment

One merged pip env runs both the sim and the FPGA driver:

```bash
conda create -n pi05_libero python=3.11 -y && conda activate pi05_libero
pip install -r models/pi05_libero/pi_requirements.txt

# LIBERO itself is not on PyPI -- editable install from the openpi checkout:
pip install -e ~/apex-compute-ML/simple-llm/src/models/pi0_5/openpi_src/third_party/libero

# one-time: seed LIBERO's config (answers its interactive first-run prompt)
printf 'N\n' | python -c "import sys; sys.path.insert(0,'~/apex-compute-ML/simple-llm/src/models/pi0_5/openpi_src/third_party/libero'); import libero.libero"
```

Two pins in `pi_requirements.txt` are load-bearing and must not be "upgraded":

- **`numpy<2`** — `robosuite`/`gym==0.25.2` break on numpy 2.x; this is what lets the
  simulator and `user_dma_core` coexist in one interpreter.
- **CPU `torch`** for the runtime env (the model runs on the FPGA, not the GPU). The
  GPU reference path (`--verify-denoise`, and `libero_eval.py --backend torch`) needs
  CUDA separately — on Blackwell (sm_120, e.g. RTX 5070) that means `torch==2.8.0+cu128`
  from `https://download.pytorch.org/whl/cu128`; stock PyPI torch is CPU-only there and
  silently reports `cuda=False`.

Headless rendering needs EGL: `export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`.

Weight export is a one-time prep step, run in a separate env that has `openpi`/jax
(kept out of the runtime env on purpose — jax-cuda12-plugin's nvidia-* wheels collide
with the cu128 torch set):

```bash
/home/rohit/miniconda3/envs/pi05/bin/python models/pi05_libero/pi05_libero_export_weights.py \
    --out pi05_libero_bin/weights_export_new
```

This pulls the checkpoint from `gs://openpi-assets/checkpoints/pi05_libero` and writes
51 tensors (~13GB of `.npy` + `manifest.json`) that every backend loads from thereafter
— `openpi` is never imported again after export.

Verify the install:

```bash
MUJOCO_GL=egl python models/pi05_libero/libero_eval.py --backend torch --tasks 1 --trials 3
# expect: 3/3 SUCCESS in ~36s, mp4s written to data/libero/videos/
```

Swapping the policy backend between CUDA and the FPGA accelerator is a single flag on
`libero_eval.py` (`--backend torch` vs `--backend fpga`), so the same closed-loop
episode can be replayed against either backend for direct comparison.

## 2. Model Architecture Walkthrough

pi0.5 is ~2.7B parameters across three stacks:

- **Vision tower — SigLIP So400m/14** (~0.41B params): 27-layer bidirectional
  transformer, hidden=1152, 16 heads (head_dim=72), 14x14 patches over three 224x224
  camera slots (base, wrist, and an unused third slot LIBERO doesn't populate),
  256 patch tokens per image, learned absolute position embeddings — no RoPE.
- **Language prefix — Gemma-2B** (~2.0B params): 18 layers, hidden=2048, 8 query
  heads / 1 shared KV head (MQA), head_dim=256, gated-SiLU MLP (16384 intermediate).
  Vision tokens are linearly projected 1152->2048 and concatenated with instruction
  tokens into one prefix sequence (~968 tokens, padded to 1024) that's processed once
  per inference and cached as KV.
- **Action expert — Gemma-300M** (~0.31B params): a *separate* 18-layer transformer
  (hidden=1024, 8 heads / 1 KV head) that cross-attends into the cached prefix KV and
  runs 10 Euler flow-matching steps to denoise a 10-step x 7-dim action chunk from
  noise, conditioned on the timestep via AdaRMSNorm.

Per-inference compute is dominated by the language prefix: ~2332 GFLOP of the ~3059
GFLOP effective total (vision ~656, action expert ~71). On the FPGA, hardware padding
(vision head_dim 72->128, action horizon 10->64 and width 7->64 for tiling) inflates
issued FLOPs to ~3466 GFLOP — about 13% overhead, concentrated almost entirely in the
action expert, whose padded FLOPs are ~6.5x its effective FLOPs.

## 3. Closed-Loop Policy Behavior

*(placeholder — needs the in-the-loop reaction narrative: replan cadence, how the
policy recovers/fails to recover mid-chunk, qualitative behaviors seen across
episodes/tasks)*

## 4. Divergent Behavior Under Quantization (bf16 vs IF4)

IF4 is a **DRAM-footprint enabler, not a speedup**: weights are dequantized to bf16
immediately before every multiply-accumulate, so compute is bf16 throughout regardless
of storage format. IF4 shrinks the checkpoint from ~13GB to ~1.56GB but was never
expected — and never observed — to run faster than bf16; no same-platform bf16-vs-IF4
timing anomaly turned up in the codebase or logs.

Where IF4 *does* diverge is accuracy, and unevenly across the pipeline: vision encoder
~25dB SNR, prefix K-cache ~16dB, prefix V-cache ~10.6dB (the weak point), denoise
velocity ~23dB, end-to-end action ~16.3dB. Per-dimension, this is lumpy — one action
dimension drops to ~1.5dB while the gripper dimension holds ~40dB — and SNR degrades
across the 10 denoise steps as errors compound (~45dB at step 0 down to ~16dB at the
final step). Separately, one IF4 matmul kernel was observed passing ~53dB in isolation
but failing catastrophically (-2.97dB) when chained after other quantized-weight
allocations in the same process — suspected DRAM allocator overlap in test sequencing,
not a model or op correctness issue.
