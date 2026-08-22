"""JAX/openpi GOLDEN ORACLE for one pi0.5 LIBERO observation.

This is level L0: the real openpi implementation running the real
gs://openpi-assets/checkpoints/pi05_libero checkpoint in JAX. It shares NO code
with either pi05_test.py (FPGA) or pi05_torch_ref.py (CPU), which is the
whole point -- those two are a hand-written reimplementation and the hardware
port OF that reimplementation, so they agree with each other even when both are
wrong. That is exactly how the missing sqrt(2048) text-embedding scale survived
a "matches CPU-IF4" 29dB sign-off.

Inputs are taken from the SAME helpers the FPGA path uses (_load_sample_images,
_load_sample_prompt_tokens, _load_sample_state), so the only difference between
this and the FPGA run is the implementation under test -- not the preprocessing.

Usage (needs the pi05 env: jax 0.5.3 + CUDA, and openpi on PYTHONPATH):
    conda activate pi05
    PYTHONPATH=<openpi_src>/src python pi05_jax_oracle.py --out oracle.npz
"""

import argparse
import os
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).parent          # <model>/utility/
_MODEL_DIR = _HERE.parent                      # <model>/
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_MODEL_DIR))            # pi05_test

# The checkpoint that pi05_bin/weights_export/*.npy was exported FROM.
CKPT = _MODEL_DIR / "pi05_bin" / "openpi-assets" / "checkpoints" / "pi05_libero"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_HERE / "jax_oracle_chunk.npz"))
    ap.add_argument("--steps", type=int, default=10, help="denoise steps (FPGA uses 10)")
    ap.add_argument("--dump-stages", default=None,
                    help="PREFIX for per-stage golden tensors. Writes <prefix>_vision_slot<i>.npy "
                         "(SigLIP image tokens, directly comparable to the FPGA's "
                         "enc_*_slot<i>_head_out.npy) and <prefix>_prefix_tokens.npy (the "
                         "embedded prefix BEFORE layer 0 -- vision rows + sqrt(width)-scaled "
                         "text rows, which is exactly what embed_and_concat_prefix builds).")
    args = ap.parse_args()

    # Import the FPGA test module ONLY for its input helpers -- never its model.
    # It pulls in torch/user_dma_core but touches no hardware at import time.
    import pi05_test as M

    images_hwc = M._load_sample_images()          # 3 x (224,224,3) float32 in [-1,1]
    prompt_tokens = M._load_sample_prompt_tokens()  # int64 token ids, BOS included
    state = np.asarray(M._load_sample_state(), dtype=np.float32)

    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.training import config as _config

    print(f"[oracle] jax devices: {jax.devices()}")
    cfg = _config.get_config("pi05_libero")
    print(f"[oracle] restoring params from {CKPT / 'params'} ...")
    model = cfg.model.load(_model.restore_params(CKPT / "params", dtype=jnp.bfloat16))
    print(f"[oracle] model loaded: action_horizon={model.action_horizon} "
          f"action_dim={model.action_dim}")

    # openpi's LIBERO observation carries base + wrist. Our 3rd slot is the
    # all-zero masked camera (image_mask=False), which is exactly how openpi
    # represents an absent camera -- so pass it through with mask False rather
    # than dropping it, matching the FPGA's keep-masked-slots default.
    names = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    images, image_masks = {}, {}
    for i, name in enumerate(names):
        img = np.asarray(images_hwc[i], dtype=np.float32)
        images[name] = jnp.asarray(img)[None]                      # (1,224,224,3)
        # A slot is masked iff it is the all-zero placeholder camera.
        image_masks[name] = jnp.asarray([bool(np.any(img))])

    tok = jnp.asarray(prompt_tokens, dtype=jnp.int32)[None]        # (1, L)
    obs = _model.Observation(
        images=images,
        image_masks=image_masks,
        state=jnp.asarray(state, dtype=jnp.float32)[None],
        tokenized_prompt=tok,
        tokenized_prompt_mask=jnp.ones_like(tok, dtype=bool),
    )

    # EXACT noise the FPGA seeds: run_inference builds an all-1e-6 buffer and
    # overwrites [:10,:7] with RandomState(0).randn(10,7). The 1e-6 (not 0)
    # matters -- rms_norm has no epsilon on the hardware -- and the oracle must
    # see the same values or the two denoise trajectories start apart.
    AH, AD = model.action_horizon, model.action_dim
    noise = np.full((1, AH, AD), 1e-6, dtype=np.float32)
    noise[0, :10, :7] = np.random.RandomState(0).randn(10, 7)

    if args.dump_stages:
        # STAGE 1 -- SigLIP image tokens, per slot. This is the FPGA's
        # VIS_HEAD_OUT buffer (run_encoder_only's <prefix>_slot<i>_head_out.npy):
        # 256 tokens x 2048, already projected to Gemma width. Vision does not
        # touch the text embedder, so an FPGA dump taken before the sqrt(width)
        # fix is still valid to score against this.
        for i, name in enumerate(names):
            toks, _ = model.PaliGemma.img(obs.images[name], train=False)
            t = np.asarray(jax.device_get(toks), dtype=np.float32)[0]
            np.save(f"{args.dump_stages}_vision_slot{i}.npy", t)
            print(f"[oracle] vision slot{i} ({name}): {t.shape} "
                  f"RMS={np.sqrt((t**2).mean()):.4f} absmax={np.abs(t).max():.3f}")
        # STAGE 2 -- the embedded prefix as it enters layer 0. embed_prefix
        # concatenates image tokens and sqrt(width)-scaled text tokens, which is
        # the exact tensor embed_and_concat_prefix DMAs to LAYER0_INPUT_DRAM.
        pre_tokens, pre_mask, _ = model.embed_prefix(
            _model.preprocess_observation(None, obs, train=False))
        pt = np.asarray(jax.device_get(pre_tokens), dtype=np.float32)[0]
        np.save(f"{args.dump_stages}_prefix_tokens.npy", pt)
        n_vis = 3 * 256
        print(f"[oracle] prefix tokens: {pt.shape} "
              f"vision_rows_RMS={np.sqrt((pt[:n_vis]**2).mean()):.4f} "
              f"text_rows_RMS={np.sqrt((pt[n_vis:]**2).mean()):.4f}")
        print(f"[oracle] prefix valid_len (mask sum) = "
              f"{int(np.asarray(jax.device_get(pre_mask)).sum())}")

    print(f"[oracle] sampling {args.steps} denoise steps ...")
    actions = model.sample_actions(
        jax.random.key(0), obs, noise=jnp.asarray(noise), num_steps=args.steps)
    actions = np.asarray(jax.device_get(actions), dtype=np.float32)[0]   # (AH, AD)

    # Unnormalize to real robot action space exactly as the FPGA path prints it
    # (openpi Unnormalize: (x+1)/2*(q99-q01)+q01), using the SAME norm stats.
    ns = M._load_norm_stats()
    q01 = np.array(ns["actions"]["q01"], dtype=np.float32)
    q99 = np.array(ns["actions"]["q99"], dtype=np.float32)
    chunk = actions[:10, :7]
    unnorm = (chunk + 1.0) / 2.0 * (q99[:7] - q01[:7] + 1e-6) + q01[:7]

    np.set_printoptions(precision=4, suppress=True)
    print("\n[oracle] action_chunk (real robot actions): (10, 7)")
    print(unnorm)
    print(f"[oracle] nan={np.isnan(unnorm).any()} inf={np.isinf(unnorm).any()} "
          f"min={unnorm.min():.4f} max={unnorm.max():.4f}")

    np.savez(args.out, normalized=chunk, unnormalized=unnorm,
             prompt_tokens=np.asarray(prompt_tokens))
    print(f"[oracle] wrote {args.out}")


if __name__ == "__main__":
    main()