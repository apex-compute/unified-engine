"""Action-chunk ablation for VeraPulse PulseVLA-LIBERO-0.5B. GPU / pure torch. NO FPGA.

THE QUESTION
------------
The action head predicts a chunk of 50 actions; only the first 10 are executed before
re-planning. What happens if the action expert only ever sees the first 10 action tokens?

Two variants, which answer different questions:

  A ("truncate")  x_t is [10, 32] outright. The expert self-attends over 10 tokens, the
                  final projection gives [10, 7]. Fewer ROWS OF COMPUTE.
  B ("mask")      x_t stays [50, 32], but the expert's SELF-ATTENTION over the suffix is
                  masked so no suffix query may attend to suffix keys 10..49. Then the
                  first 10 rows are taken. Same rows of compute, LESS ATTENTION CONTEXT.

WHAT THE ARCHITECTURE ALREADY DOES (read smolvla/modeling.py before trusting a result)
--------------------------------------------------------------------------------------
`VLAFlowMatching.embed_suffix` returns `att = torch.ones(bsz, chunk)`, and
`make_att_2d_masks` turns a run of 1s into `cumsum[j] <= cumsum[i]`, i.e. a LOWER
TRIANGULAR mask. **The suffix self-attention is already causal.** Action token i attends
to the whole prefix plus suffix tokens 0..i -- never to i+1..49. The cross-attention
layers read only the frozen prefix KV cache, so they are per-token independent. Every
other op in the expert (MLP, residual, norms, the o_proj) is per-token.

Therefore token i's value depends only on suffix tokens 0..i, at every Euler step, and
so the first 10 rows are mathematically independent of rows 10..49. Both ablations are
expected to be EXACT NO-OPS on the executed actions. This script exists to check that
claim numerically instead of asserting it from a code read -- and the closed-loop mode
exists to confirm it survives contact with the simulator.

USAGE
-----
    PY=/home/rohit/miniconda3/envs/apex_libero/bin/python

    # 1. numerical fidelity + wall clock, no simulator needed
    $PY models/verapulse/utility/chunk_ablation.py --mode numeric --device cuda -n 8

    # 2. closed-loop LIBERO success rate, one variant per run
    MUJOCO_GL=egl $PY models/verapulse/utility/chunk_ablation.py --mode libero \
        --variant baseline --task-suite libero_spatial --tasks 10 --trials 2

Nothing here writes to verapulse_test.py or libero_eval.py; the ablation is installed by
rebinding VLAFlowMatching.denoise_step on the live module instance (the same
types.MethodType trick verapulse_test.py uses for _patch_upstream_quick_gelu).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import types

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent
_VP = _HERE.parent                      # models/verapulse
_BUNDLE = _VP / "verapulse_bin" / "verapulse__pulsevla-libero-0.5b"

# MuJoCo picks its GL backend at IMPORT time, so this has to be set before anything
# imports mujoco/robosuite. Same default libero_eval.py uses.
os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(_VP))            # libero_eval, verapulse_test
sys.path.insert(0, str(_VP.parent.parent))
sys.path.insert(0, str(_BUNDLE))        # the checkpoint's own `smolvla` package


# --------------------------------------------------------------------------- #
# the ablation itself
# --------------------------------------------------------------------------- #
def _denoise_step_masked(self, prefix_pad, kv, x_t, timestep, keep, invert=False):
    """VLAFlowMatching.denoise_step with the suffix self-attention keys truncated.

    Byte-for-byte upstream's denoise_step except for the `suffix_2d[:, :, keep:] = False`
    line. That column mask is what variant B is: it lands in `full_2d`, which the SELF-
    ATTENTION layers consume as their mask. The cross-attention layers slice
    `mask[:, -suffix_len:, :ek.shape[1]]` where `ek.shape[1] == prefix_len`, so they only
    ever see the prefix columns and are untouched -- which is what makes this an
    expert-self-attention ablation and not a prefix ablation.
    """
    from smolvla.modeling import make_att_2d_masks

    suffix_embs, suffix_pad, suffix_att = self.embed_suffix(x_t, timestep)
    suffix_len = suffix_pad.shape[1]
    bsz, prefix_len = prefix_pad.shape[0], prefix_pad.shape[1]
    prefix_2d = prefix_pad[:, None, :].expand(bsz, suffix_len, prefix_len)
    suffix_2d = make_att_2d_masks(suffix_pad, suffix_att)
    if keep is not None and keep < suffix_len:
        suffix_2d = suffix_2d.clone()
        if invert:
            suffix_2d[:, :, :keep] = False       # negative control, see install_variant
        else:
            suffix_2d[:, :, keep:] = False       # variant B: drop suffix keys 10..49
    full_2d = torch.cat([prefix_2d, suffix_2d], dim=2)
    offsets = torch.sum(prefix_pad, dim=-1)[:, None]
    pos_ids = offsets + torch.cumsum(suffix_pad, dim=1) - 1
    outs, _ = self.vlm_with_expert.forward(
        attention_mask=full_2d, position_ids=pos_ids, past_key_values=kv,
        inputs_embeds=[None, suffix_embs], use_cache=True, fill_kv_cache=False,
    )
    suffix_out = outs[1][:, -self.cfg.chunk_size:].to(torch.float32)
    return self.action_out_proj(suffix_out)


def install_variant(model, variant, keep=10):
    """Switch a live SmolVLA between baseline / A / B. Idempotent, and reversible by
    calling again with another variant -- baseline restores the shipped method.

    Returns the number of chunk ROWS the sampler should draw noise for.
    """
    fm = model.model                                     # VLAFlowMatching
    chunk = fm.cfg.chunk_size
    if variant == "baseline":
        # rebind the class's own function so repeated installs never stack patches
        fm.denoise_step = types.MethodType(type(fm).denoise_step, fm)
        return chunk
    if variant == "A":
        # Nothing to patch: a shorter x_t propagates through embed_suffix ->
        # suffix_len -> the 2d masks -> pos_ids on its own, and the
        # `outs[1][:, -chunk_size:]` slice is tolerant of a sequence shorter than 50.
        fm.denoise_step = types.MethodType(type(fm).denoise_step, fm)
        return keep
    if variant == "B":
        fm.denoise_step = types.MethodType(
            lambda s, pp, kv, x, t: _denoise_step_masked(s, pp, kv, x, t, keep), fm)
        return chunk
    if variant == "Bctl":
        # NEGATIVE CONTROL. Masks the OTHER end of the suffix key axis: keys 0..keep-1 go
        # away and 10..49 survive. Under the causal mask rows 0..9 only ever had keys
        # 0..i, so this strips every suffix key they had -- it MUST move the output. If
        # this reads zero divergence too, the mask is not reaching the attention at all
        # and B's clean result is an artefact of dead code, not of the architecture.
        fm.denoise_step = types.MethodType(
            lambda s, pp, kv, x, t: _denoise_step_masked(
                s, pp, kv, x, t, keep, invert=True), fm)
        return chunk
    raise ValueError(f"unknown variant {variant!r}")


# --------------------------------------------------------------------------- #
# model construction (mirrors libero_eval._OracleBackend, without importing it)
# --------------------------------------------------------------------------- #
def build_oracle(device="cuda", n_exec=10, seed=0):
    from safetensors.torch import load_file
    from smolvla import (SmolVLA, SmolVLAProcessor, Tokenizer,
                         load_lerobot_norm_stats, load_smolvla_config)

    if not _BUNDLE.is_dir():
        raise FileNotFoundError(f"checkpoint bundle missing at {_BUNDLE}")
    print(f"[abl] loading {_BUNDLE.name} on {device} ...", flush=True)
    t0 = time.perf_counter()
    cfg = load_smolvla_config(str(_BUNDLE / "config.json"), n_action_steps=n_exec)
    model = SmolVLA(cfg).float().to(device).eval()
    model.load_state_dict(load_file(str(_BUNDLE / "model.safetensors")))
    stats = load_lerobot_norm_stats(str(_BUNDLE / "norm_stats.safetensors"))
    tok = Tokenizer(str(_BUNDLE / "tokenizer.json"), max_length=cfg.tokenizer_max_length)
    proc = SmolVLAProcessor(cfg, tok, stats, device=device)
    torch.manual_seed(seed)
    print(f"[abl] ready in {time.perf_counter() - t0:.1f}s "
          f"({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)", flush=True)
    return model, proc, cfg


# --------------------------------------------------------------------------- #
# numeric mode: same prefix, same noise, three samplers
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def _prefix_pass(model, model_input):
    """Run the VLM once and return (prefix_pad, kv). SHARED by all three variants.

    Reusing one prefix KV cache is not just a speedup -- it removes the vision tower and
    the 32 VLM layers from the comparison entirely, so any divergence that shows up is
    the action expert's and nothing else's. The cache is only ever READ during denoising
    (forward_attn_layer/forward_cross_attn_layer take the fill=False branch), so sharing
    it across variants cannot cross-contaminate them.
    """
    from smolvla.modeling import make_att_2d_masks

    fm = model.model
    images, img_masks = model._images_list(model_input)
    pe, pp, pa = fm.embed_prefix(images, img_masks, model_input.lang_tokens,
                                model_input.lang_masks, model_input.state)
    _, kv = fm.vlm_with_expert.forward(
        attention_mask=make_att_2d_masks(pp, pa),
        position_ids=torch.cumsum(pp, dim=1) - 1,
        past_key_values=None, inputs_embeds=[pe, None],
        use_cache=True, fill_kv_cache=True,
    )
    return pp, kv


@torch.inference_mode()
def _sample_traced(model, prefix_pad, kv, noise):
    """Upstream's Euler sampler, unrolled so every intermediate x_t is kept.

    Identical arithmetic to VLAFlowMatching.sample_actions -- the only addition is the
    per-step snapshot, which is what lets us say WHERE a divergence appears rather than
    only that the endpoint moved.
    """
    fm = model.model
    bsz, device = prefix_pad.shape[0], prefix_pad.device
    num_steps = fm.cfg.num_steps
    dt = -1.0 / num_steps
    x_t = noise
    traj = [x_t.clone()]
    for step in range(num_steps):
        t = 1.0 + step * dt
        t_tensor = torch.tensor(t, dtype=torch.float32, device=device).expand(bsz)
        x_t = x_t + dt * fm.denoise_step(prefix_pad, kv, x_t, t_tensor)
        traj.append(x_t.clone())
    return x_t, traj


def _cos(a, b, eps=1e-12):
    """Row-wise cosine similarity between two (n, d) arrays."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return num / np.maximum(den, eps)


def _stats(ref, alt, label):
    d = np.asarray(alt, np.float64) - np.asarray(ref, np.float64)
    return {
        "label": label,
        "cos_mean": float(_cos(ref, alt).mean()),
        "cos_min": float(_cos(ref, alt).min()),
        "mse": float((d ** 2).mean()),
        "max_abs": float(np.abs(d).max()),
        "rel_l2": float(np.linalg.norm(d) / max(np.linalg.norm(ref), 1e-12)),
    }


def _synthetic_obs(n, cfg, device, seed=0):
    """Random images / plausible robot states. A mechanism check, not a policy check --
    the closed-loop mode is what exercises real on-policy observations."""
    from smolvla.types import Obs
    rng = np.random.RandomState(seed)
    tasks = ["pick up the black bowl and place it on the plate",
             "open the top drawer and put the bowl inside",
             "put the wine bottle on top of the cabinet",
             "push the plate to the front of the stove"]
    out = []
    for i in range(n):
        imgs = {k: torch.as_tensor(
            rng.rand(1, 3, 256, 256).astype(np.float32)).to(device) for k in ("image", "image2")}
        state = torch.as_tensor(
            np.concatenate([rng.randn(3) * 0.1, rng.randn(3) * 0.3,
                            rng.randn(2) * 0.01]).astype(np.float32))[None].to(device)
        out.append(Obs(images=imgs, state=state, task=[tasks[i % len(tasks)]]))
    return out


def _libero_obs(n, device, task_suite="libero_spatial", seed=7):
    """Real on-policy-ish observations: reset a LIBERO task, settle, and grab frames.

    Deliberately NOT policy-driven -- these are the observations the very first inference
    of an episode sees, which is enough for a fidelity check and costs one env
    construction instead of a rollout per sample.
    """
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from smolvla.types import Obs

    suite = benchmark.get_benchmark_dict()[task_suite]()
    out = []
    for i in range(n):
        tid = i % suite.n_tasks
        task = suite.get_task(tid)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                                 camera_heights=256, camera_widths=256)
        env.seed(seed)
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(tid)[i % 10])
        for _ in range(10):                       # let the scene settle
            obs, *_ = env.step([0.0] * 6 + [-1.0])
        base = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

        def _img(a):
            t = torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32)
            return (t.permute(2, 0, 1) / 255.0).unsqueeze(0).to(device)

        import libero_eval as LE
        state = np.concatenate((obs["robot0_eef_pos"],
                                LE._quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"])).astype(np.float32)
        out.append(Obs(images={"image": _img(base), "image2": _img(wrist)},
                       state=torch.as_tensor(state)[None].to(device),
                       task=[str(task.language)]))
        env.close()
        del env
    return out


def run_numeric(args):
    model, proc, cfg = build_oracle(args.device, n_exec=args.keep, seed=args.seed)
    device = args.device
    keep, chunk = args.keep, cfg.chunk_size
    print(f"[abl] chunk_size={chunk}  n_exec/keep={keep}  num_steps={cfg.num_steps}  "
          f"self_attn_every_n_layers={cfg.self_attn_every_n_layers}", flush=True)

    obs_list = (_libero_obs(args.n, device, args.task_suite, args.seed)
                if args.obs == "libero" else _synthetic_obs(args.n, cfg, device, args.seed))

    rows, per_step_rows = [], []
    for i, obs in enumerate(obs_list):
        mi = proc.to_model_input(obs)
        prefix_pad, kv = _prefix_pass(model, mi)

        # ONE noise draw, shared. A's noise is literally the first `keep` rows of the
        # baseline's, so the comparison is of the sampler, never of the RNG.
        g = torch.Generator(device=device).manual_seed(args.seed * 1000 + i)
        noise = torch.randn((1, chunk, cfg.max_action_dim), device=device, generator=g)

        install_variant(model, "baseline", keep)
        base_x, base_traj = _sample_traced(model, prefix_pad, kv, noise)

        install_variant(model, "A", keep)
        a_x, a_traj = _sample_traced(model, prefix_pad, kv, noise[:, :keep])

        install_variant(model, "B", keep)
        b_x, b_traj = _sample_traced(model, prefix_pad, kv, noise)

        install_variant(model, "Bctl", keep)
        c_x, c_traj = _sample_traced(model, prefix_pad, kv, noise)

        # robot units, which is what actually drives the arm
        r_base = proc.postprocess_action(base_x)[0, :keep].numpy()
        r_a = proc.postprocess_action(a_x)[0, :keep].numpy()
        r_b = proc.postprocess_action(b_x)[0, :keep].numpy()
        r_c = proc.postprocess_action(c_x)[0, :keep].numpy()
        # normalized model output, all 32 dims
        n_base = base_x[0, :keep].float().cpu().numpy()
        n_a = a_x[0, :keep].float().cpu().numpy()
        n_b = b_x[0, :keep].float().cpu().numpy()
        n_c = c_x[0, :keep].float().cpu().numpy()

        rows.append({
            "obs": i,
            "robot": [_stats(r_base, r_a, "A"), _stats(r_base, r_b, "B"),
                      _stats(r_base, r_c, "Bctl")],
            "norm": [_stats(n_base, n_a, "A"), _stats(n_base, n_b, "B"),
                     _stats(n_base, n_c, "Bctl")],
        })
        for s in range(len(base_traj)):
            pb = base_traj[s][0, :keep].float().cpu().numpy()
            per_step_rows.append({
                "obs": i, "step": s,
                "A": _stats(pb, a_traj[s][0, :keep].float().cpu().numpy(), "A"),
                "B": _stats(pb, b_traj[s][0, :keep].float().cpu().numpy(), "B"),
                "Bctl": _stats(pb, c_traj[s][0, :keep].float().cpu().numpy(), "Bctl"),
            })
        print(f"[abl] obs {i}: A cos={rows[-1]['robot'][0]['cos_mean']:.9f} "
              f"mse={rows[-1]['robot'][0]['mse']:.3e} | "
              f"B cos={rows[-1]['robot'][1]['cos_mean']:.9f} "
              f"mse={rows[-1]['robot'][1]['mse']:.3e} | "
              f"Bctl cos={rows[-1]['robot'][2]['cos_mean']:.6f} "
              f"mse={rows[-1]['robot'][2]['mse']:.3e}", flush=True)

    # ---- wall clock: the FULL inference each variant would really run --------------
    timing = {}
    if args.time_reps:
        mi = proc.to_model_input(obs_list[0])
        for variant in ("baseline", "A", "B"):
            rows_n = install_variant(model, variant, keep)
            g = torch.Generator(device=device).manual_seed(args.seed)
            noise = torch.randn((1, rows_n, cfg.max_action_dim), device=device, generator=g)

            def _one():
                """One whole inference: vision + 32 VLM layers, then 10 Euler steps."""
                pp, kv_ = _prefix_pass(model, mi)
                return _sample_traced(model, pp, kv_, noise)[0]

            for _ in range(2):                        # warm up cudnn/cublas autotune
                _one()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.time_reps):
                _one()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            full = (time.perf_counter() - t0) / args.time_reps

            # denoise-only: the part the ablation can actually make cheaper
            pp, kv_ = _prefix_pass(model, mi)
            for _ in range(2):
                _sample_traced(model, pp, kv_, noise)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.time_reps):
                _sample_traced(model, pp, kv_, noise)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            den = (time.perf_counter() - t0) / args.time_reps

            timing[variant] = {"full_s": full, "denoise_only_s": den,
                               "prefix_s": full - den, "chunk_rows": rows_n}
            print(f"[abl] timing {variant:>8s}: full {full * 1000:8.1f} ms  "
                  f"(prefix {(full - den) * 1000:7.1f} + denoise {den * 1000:6.1f})",
                  flush=True)

    _report_numeric(rows, per_step_rows, timing, args)
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(
            {"device": device, "n_obs": args.n, "obs_source": args.obs, "keep": keep,
             "chunk_size": chunk, "per_obs": rows, "per_step": per_step_rows,
             "timing": timing}, indent=2))
        print(f"[abl] json -> {args.out}")


def _report_numeric(rows, per_step_rows, timing, args):
    def agg(space, label):
        rs = [r for row in rows for r in row[space] if r["label"] == label]
        return (float(np.mean([r["cos_mean"] for r in rs])),
                float(np.min([r["cos_min"] for r in rs])),
                float(np.mean([r["mse"] for r in rs])),
                float(np.max([r["max_abs"] for r in rs])))

    print("\n" + "=" * 78)
    print(f"  ACTION FIDELITY vs baseline, first {args.keep} actions, "
          f"{len(rows)} observations ({args.obs})")
    print(f"  {'variant':<10}{'space':<10}{'cos mean':>14}{'cos min':>14}"
          f"{'MSE':>13}{'max|d|':>13}")
    for label in ("A", "B", "Bctl"):
        for space, nm in (("robot", "robot 7d"), ("norm", "norm 32d")):
            c, cm, m, mx = agg(space, label)
            print(f"  {label:<10}{nm:<10}{c:>14.9f}{cm:>14.9f}{m:>13.3e}{mx:>13.3e}")
    if per_step_rows:
        print(f"\n  worst per-Euler-step divergence (max over obs, of max|d| in x_t):")
        for label in ("A", "B", "Bctl"):
            worst = {}
            for r in per_step_rows:
                worst[r["step"]] = max(worst.get(r["step"], 0.0), r[label]["max_abs"])
            line = "  ".join(f"s{k}:{v:.2e}" for k, v in sorted(worst.items()))
            print(f"    {label}: {line}")
    if timing:
        print(f"\n  WALL CLOCK on {args.device}, mean of {args.time_reps} "
              f"(prefix VLM pass + {10} Euler steps)")
        print(f"  {'variant':<10}{'rows':>6}{'full ms':>12}{'prefix ms':>12}"
              f"{'denoise ms':>13}{'vs base':>10}")
        base = timing.get("baseline", {}).get("full_s")
        for v, t in timing.items():
            sp = f"{base / t['full_s']:.3f}x" if base else "-"
            print(f"  {v:<10}{t['chunk_rows']:>6}{t['full_s'] * 1000:>12.1f}"
                  f"{t['prefix_s'] * 1000:>12.1f}{t['denoise_only_s'] * 1000:>13.1f}{sp:>10}")
    print("=" * 78 + "\n")


# --------------------------------------------------------------------------- #
# audit mode: prove the attention geometry instead of asserting it
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def run_audit(args):
    """Instrument one denoise step and report what the expert's attention ACTUALLY sees.

    Three things get checked, because the whole ablation result hinges on them:
      1. the ORDER of the combined key axis on self-attn layers (prefix first or last?)
      2. whether the suffix block of the mask is causal or full
      3. whether cross-attn layers ever see a suffix key
    """
    model, proc, cfg = build_oracle(args.device, n_exec=args.keep, seed=args.seed)
    device, keep, chunk = args.device, args.keep, cfg.chunk_size
    obs = _synthetic_obs(1, cfg, device, args.seed)[0]
    mi = proc.to_model_input(obs)
    prefix_pad, kv = _prefix_pass(model, mi)
    prefix_len = prefix_pad.shape[1]
    fm, vwe = model.model, model.model.vlm_with_expert

    rec = []
    orig = vwe.eager_attention_forward

    def spy(attention_mask, bsz, head_dim, q, k, v):
        rec.append({"q_len": q.shape[1], "k_len": k.shape[1],
                    "mask": attention_mask.detach().clone(),
                    "k": k.detach()})
        return orig(attention_mask, bsz, head_dim, q, k, v)

    vwe.eager_attention_forward = spy
    try:
        noise = torch.randn((1, chunk, cfg.max_action_dim), device=device)
        fm.denoise_step(prefix_pad, kv,
                        noise, torch.tensor(1.0, device=device).expand(1))
    finally:
        vwe.eager_attention_forward = orig

    n_self = cfg.self_attn_every_n_layers
    print("\n" + "=" * 78)
    print(f"  ATTENTION AUDIT -- one denoise step, prefix_len={prefix_len}, "
          f"chunk={chunk}, {len(rec)} attention calls")
    print(f"  prefix valid (non-pad) keys: {int(prefix_pad.sum().item())} of {prefix_len}")

    self_idx = [i for i in range(len(rec)) if i % n_self == 0]
    cross_idx = [i for i in range(len(rec)) if i % n_self != 0]
    s0, c0 = rec[self_idx[0]], rec[cross_idx[0]]

    print(f"\n  SELF-ATTN layers (idx % {n_self} == 0, {len(self_idx)} of {len(rec)}):")
    print(f"    q_len={s0['q_len']}  k_len={s0['k_len']}  "
          f"(= prefix {prefix_len} + suffix {s0['k_len'] - prefix_len})")
    # 1. key ORDER: are the first prefix_len key rows literally the cached prefix K?
    cached_k = kv[self_idx[0]]["key_states"]
    head = s0["k"][:, :prefix_len]
    tail = s0["k"][:, prefix_len:]
    print(f"    key-axis order: k[:, :{prefix_len}] == cached prefix K ? "
          f"{bool(torch.equal(head, cached_k))}   "
          f"(so the {tail.shape[1]} SUFFIX keys sit at the END: "
          f"columns {prefix_len}..{s0['k_len'] - 1})")
    # 2. mask geometry
    m = s0["mask"][0]
    pref_block, suff_block = m[:, :prefix_len], m[:, prefix_len:]
    causal = torch.tril(torch.ones_like(suff_block, dtype=torch.bool))
    rows_same = bool((pref_block == pref_block[0:1]).all())
    print(f"    prefix block [{tuple(pref_block.shape)}]: every suffix query sees the "
          f"same prefix keys ? {rows_same}  ({int(pref_block[0].sum())} visible)")
    print(f"    suffix block [{tuple(suff_block.shape)}]: lower-triangular (CAUSAL) ? "
          f"{bool(torch.equal(suff_block, causal))}")
    vis = [int(suff_block[i].sum()) for i in (0, 1, 9, 10, 25, chunk - 1)]
    print(f"    suffix keys visible to query row  0,1,9,10,25,{chunk - 1}: {vis}")
    print(f"    -> action token i attends to suffix tokens 0..i ONLY. Rows 0..{keep - 1} "
          f"never see rows {keep}..{chunk - 1} even in the UNABLATED model.")

    print(f"\n  CROSS-ATTN layers ({len(cross_idx)} of {len(rec)}):")
    print(f"    q_len={c0['q_len']}  k_len={c0['k_len']}  "
          f"suffix keys present ? {c0['k_len'] > prefix_len}")
    print(f"    -> cross layers read the reprojected PREFIX K/V only; they are "
          f"per-action-token independent and the ablation leaves them untouched.")

    frac = (chunk - keep) / s0["k_len"]
    print(f"\n  SCALE OF THE ABLATION: dropping suffix keys {keep}..{chunk - 1} removes "
          f"{chunk - keep} of {s0['k_len']} keys ({100 * frac:.1f}%) on self-attn layers,")
    print(f"  and 0 of {c0['k_len']} on cross-attn layers. The observation prefix "
          f"dominates the key axis.")
    print("=" * 78 + "\n")


# --------------------------------------------------------------------------- #
# libero mode: drive libero_eval.py's loop with an ablated oracle
# --------------------------------------------------------------------------- #
def run_libero(args, extra_argv):
    """Import libero_eval, swap its _OracleBackend for an ablated one, and call main().

    libero_eval.main() looks _OracleBackend up as a module global, so rebinding the name
    is enough -- the file itself is never touched. Everything else (env construction,
    checkpointing, --resume, video) is the harness's, unchanged.
    """
    import libero_eval as LE

    variant, keep = args.variant, args.keep
    _Base = LE._OracleBackend

    class _AblatedOracle(_Base):
        takes_raw_obs = True

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.chunk_rows = install_variant(self.model, variant, keep)
            self._t_infer, self._n_infer = 0.0, 0
            self._eps_rng = np.random.RandomState(kw.get("seed", 0))
            print(f"[abl] variant={variant} keep={keep} -> sampler draws "
                  f"{self.chunk_rows} chunk rows", flush=True)

        def infer_raw(self, base_u8, wrist_u8, state_raw, language):
            def _img(a):
                t = torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32)
                return (t.permute(2, 0, 1) / 255.0).unsqueeze(0).to(self.device)

            obs = self.Obs(images={"image": _img(base_u8), "image2": _img(wrist_u8)},
                           state=torch.as_tensor(
                               np.asarray(state_raw, dtype=np.float32))[None].to(self.device),
                           task=[str(language)])
            mi = self.proc.to_model_input(obs)
            fm = self.model.model
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                # PAIRED NOISE. Always draw the FULL (1,50,32) tensor and slice it for
                # variant A, rather than drawing (1,10,32) directly. Both variants then
                # consume exactly 1600 normals per inference, so the global RNG stream
                # stays in lockstep across runs and A's chunk rows are literally
                # baseline's first 10 rows of noise. Without this, A and baseline would
                # desynchronise after inference 1 and any success-rate difference would
                # be confounded by a different noise draw -- which is the one thing this
                # experiment must not measure by accident.
                noise = torch.randn((mi.state.shape[0], fm.cfg.chunk_size,
                                     fm.cfg.max_action_dim), device=self.device)
                pp, kv = _prefix_pass(self.model, mi)
                x_t, _ = _sample_traced(self.model, pp, kv, noise[:, :self.chunk_rows])
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            self._t_infer += time.perf_counter() - t0
            self._n_infer += 1
            act = np.asarray(self.proc.postprocess_action(x_t)[0], dtype=np.float32)
            if args.eps:
                # CHAOS CONTROL (--eps). Perturb the executed action by a gaussian of the
                # SAME MAGNITUDE as variant A's float-rounding difference from baseline
                # (max|d| ~2.6e-6 in robot units). This variant is semantically the
                # baseline policy -- if its success rate moves as much as A's does, then
                # A's drop is the simulator amplifying float noise over a 200-step
                # rollout, not the ablation removing anything the policy needed.
                act = act + self._eps_rng.randn(*act.shape).astype(np.float32) * args.eps
            return act

    LE._OracleBackend = _AblatedOracle
    argv = ["libero_eval.py", "--backend", "oracle",
            "--oracle-device", args.device, "--replan-steps", str(keep)] + extra_argv
    old = sys.argv
    sys.argv = argv
    print(f"[abl] handing off to libero_eval.main() with: {' '.join(argv[1:])}", flush=True)
    holder = {}
    _orig_init = _AblatedOracle.__init__

    def _init(self, *a, **kw):
        _orig_init(self, *a, **kw)
        holder["backend"] = self
    _AblatedOracle.__init__ = _init
    try:
        LE.main()
    finally:
        sys.argv = old
        LE._OracleBackend = _Base
        b = holder.get("backend")
        if b is not None and b._n_infer:
            print(f"[abl] variant={variant}: {b._n_infer} inferences, "
                  f"{1000 * b._t_infer / b._n_infer:.1f} ms each "
                  f"(GPU, model only -- excludes preprocessing and the simulator)",
                  flush=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="numeric", choices=["numeric", "libero", "audit"])
    ap.add_argument("--variant", default="baseline",
                    choices=["baseline", "A", "B", "Bctl"],
                    help="--mode libero only; numeric mode always runs all three")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--keep", type=int, default=10,
                    help="how many action tokens survive the ablation (n_action_steps)")
    ap.add_argument("-n", "--n", type=int, default=8, help="numeric mode: observations")
    ap.add_argument("--obs", default="synthetic", choices=["synthetic", "libero"])
    ap.add_argument("--task-suite", default="libero_spatial")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--time-reps", type=int, default=5, help="0 disables timing")
    ap.add_argument("--eps", type=float, default=0.0,
                    help="--mode libero chaos control: add N(0, eps) to every executed "
                         "action. Set it to variant A's float-rounding magnitude (~2.6e-6) "
                         "to measure how much of A's success-rate change is the "
                         "simulator amplifying float noise rather than the ablation.")
    ap.add_argument("--out", default=None, help="numeric mode: write a JSON report")
    args, extra = ap.parse_known_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit(f"--device {args.device} but torch.cuda.is_available() is False "
                 f"(torch {torch.__version__}). Use the CUDA env.")
    if args.device.startswith("cuda"):
        print(f"[abl] {torch.cuda.get_device_name(0)}  torch {torch.__version__}", flush=True)

    if args.mode in ("numeric", "audit"):
        if extra:
            sys.exit(f"unknown args for --mode {args.mode}: {extra}")
        (run_numeric if args.mode == "numeric" else run_audit)(args)
    else:
        run_libero(args, extra)


if __name__ == "__main__":
    main()
