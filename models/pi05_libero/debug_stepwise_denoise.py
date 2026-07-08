#!/usr/bin/env python3
"""Stepwise micro-op NaN isolation for the 10-step / 18-layer denoise stack
(compile_denoise_loop). Mirrors debug_stepwise_prefix.py's structure exactly.

The denoise NaN (denoise_step0 comes back 100% NaN) does NOT reproduce with a
zero-filled placeholder K/V cache -- only with the REAL prefix K/V cache. So
unlike debug_stepwise_prefix.py (which can skip straight to weight_init_lm_prefix),
every checkpoint here must run the full vision (3 slots) + prefix (18 layers)
pipeline first to populate the real K/V cache, THEN compile_denoise_loop up to
the requested checkpoint. That's ~190s of fixed setup per checkpoint -- there's
no way around it without changing what's being tested.

Usage:
    python debug_stepwise_denoise.py --list                # print all checkpoint names/indices
    python debug_stepwise_denoise.py --stop-after 0         # step0_action_in_proj only
    python debug_stepwise_denoise.py --stop-after 1 --dump  # step0_time_embed + dump buffer

Move --stop-after up by one at a time (or bisect via debug_stepwise_denoise_walk.py,
same pattern as debug_stepwise_prefix_walk.py) to find the first NaN checkpoint.
"""
import argparse
import sys

import numpy as np
import torch

from pi05_libero_test import Pi05Libero_UnifiedEngine, init_hang_prevention, _CFG
from user_dma_core import DMA_DEVICE_H2C, DMA_DEVICE_C2H


def checkpoint_names(num_steps, num_layers):
    """Names in the same order _debug_op emits them inside compile_denoise_loop,
    without running any HW -- just for --list."""
    names = []
    for s in range(num_steps):
        names.append(f"step{s}_action_in_proj")
        names.append(f"step{s}_time_embed")
        for l in range(num_layers):
            p = f"step{s}_layer{l}"
            names.append(f"{p}_preattn_norm")
            names.append(f"{p}_kv_proj")
            names.append(f"{p}_q_proj")
            names.append(f"{p}_flash_kv_staged")
            names.append(f"{p}_flash_attn_out")
            names.append(f"{p}_o_proj")
            names.append(f"{p}_attn_residual")
            names.append(f"{p}_preffw_norm")
            names.append(f"{p}_mlp_out")
            names.append(f"step{s}_suffix_layer{l}")
        names.append(f"step{s}_final_ada_rms_norm")
        names.append(f"step{s}_action_out_proj")
        names.append(f"step{s}_euler_update")
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop-after", type=int, default=0,
                    help="checkpoint index to halt+run at (see --list)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--timeout", type=float, default=250.0,
                     help="per-checkpoint heartbeat timeout in seconds before declaring a real hang")
    ap.add_argument("--dump", action="store_true",
                     help="after a clean (non-hang) run, DMA-read the checkpoint's output buffer "
                          "and report NaN/Inf/all-zero/all-identical sanity checks + a value sample "
                          "+ save the full array to dump_<name>.npy for offline inspection")
    args = ap.parse_args()

    if args.list:
        cfg_steps = Pi05Libero_UnifiedEngine.AE_NUM_DENOISE_STEPS
        cfg_layers = Pi05Libero_UnifiedEngine.AE_LAYERS
        names = checkpoint_names(cfg_steps, cfg_layers)
        for i, n in enumerate(names):
            print(f"{i:4d}  {n}")
        return

    ue = Pi05Libero_UnifiedEngine()
    names = checkpoint_names(ue.AE_NUM_DENOISE_STEPS, ue.AE_LAYERS)
    if not (0 <= args.stop_after < len(names)):
        print(f"--stop-after must be in [0, {len(names)-1}]", file=sys.stderr)
        sys.exit(1)

    init_hang_prevention(ue)
    ue.weight_init()
    ue.tensor_init(_CFG["defaults"].get("max_seq", 512))

    prompt_tokens = np.random.RandomState(1).randint(0, 257152, size=(16,))

    from pathlib import Path
    import torch.nn.functional as _F
    sample_dir = Path.home() / "apex-compute-ML" / "simple-llm" / "src" / "models" / "pi0_5" / "sample_data"
    img = np.load(sample_dir / "sample_0_image.npy").astype(np.float32) / 127.5 - 1.0
    wrist = np.load(sample_dir / "sample_0_wrist_image.npy").astype(np.float32) / 127.5 - 1.0
    pad = np.zeros_like(img)
    def _resize(a):
        t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).float()
        t = _F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
        return t.squeeze(0).permute(1, 2, 0).numpy()
    images = [_resize(img), _resize(wrist), _resize(pad)]

    print("running vision (3 slots)...")
    vision_tokens = ue.run_vision(images)

    seq_len = _CFG["model"]["prefill_max_seq_len"]
    valid_len = ue.embed_and_concat_prefix(prompt_tokens, vision_embeddings=vision_tokens, seq_len=seq_len)
    print(f"running prefix (valid_len={valid_len})...")
    ue._compile_and_run(lambda: ue.compile_prefix(seq_len, valid_len), label="prefix", timeout=args.timeout)

    ue.tensor_init_action_expert()
    S, AD = ue.AE_ACTION_HORIZON_PADDED, ue.AE_ACTION_DIM_PADDED
    noise32 = np.full((S, AD), 1e-6, dtype=np.float32)
    noise32[:10, :7] = np.random.RandomState(0).randn(10, 7)
    ue.dma_write(DMA_DEVICE_H2C, ue.AE_XT_DRAM, torch.from_numpy(noise32).to(torch.bfloat16), S * AD * 2)

    ue.DEBUG_STOP_AFTER = args.stop_after
    ue.start_capture()
    prog_addr = ue.compile_denoise_loop()
    info = ue._debug_halt_info
    assert info is not None and info["idx"] == args.stop_after, \
        "compile_denoise_loop did not halt at the requested checkpoint -- check _debug_op wiring"
    assert info["name"] == names[args.stop_after], \
        f"checkpoint name mismatch: got {info['name']} expected {names[args.stop_after]}"

    if ue.capture_buffer:
        all_bytes = bytearray()
        for inst in ue.capture_buffer:
            all_bytes.extend(inst.get_bytes())
        prog_addr = ue.get_program_dram_addr()
        ue._dma_write_retry(DMA_DEVICE_H2C, prog_addr, all_bytes, len(all_bytes))
        ue.allocate_program_dram(len(all_bytes))
        ue.clear_capture_buffer()
    ue.stop_capture()

    print(f"[{info['idx']}/{len(names)-1}] {info['name']}: compiled, executing (timeout={args.timeout}s)...")
    ue.start_execute_from_dram(prog_addr)
    try:
        ue._wait_with_heartbeat(f"denoise[{info['name']}]", timeout=args.timeout)
    except RuntimeError as e:
        print(f"HANG at checkpoint {args.stop_after}/{len(names)-1} ({info['name']}): {e}")
        print(f"  bug is between checkpoint {args.stop_after-1} "
              f"({names[args.stop_after-1] if args.stop_after else 'start'}) and {args.stop_after} ({info['name']})")
        sys.exit(1)

    print(f"PASS -- no hang at checkpoint {args.stop_after}/{len(names)-1} ({info['name']}). "
          f"Bump --stop-after to {args.stop_after + 1} "
          f"({names[args.stop_after+1] if args.stop_after+1 < len(names) else 'done, full denoise clean'}) next")

    buf = bytearray(info["numel"] * 2)
    ue.dma_read(DMA_DEVICE_C2H, info["addr"], buf, len(buf))
    out = torch.frombuffer(bytes(buf), dtype=torch.bfloat16).reshape(info["shape"]).clone().float()
    n_nan = torch.isnan(out).sum().item()
    n_inf = torch.isinf(out).sum().item()
    print(f"  nan={n_nan} inf={n_inf} min={out.min().item():.6f} max={out.max().item():.6f}")

    if args.dump:
        n_zero = (out == 0).sum().item()
        n_unique = out.flatten().unique().numel()
        print(f"  dump: shape={tuple(out.shape)} numel={out.numel()}")
        print(f"  zero={n_zero}/{out.numel()} unique_values={n_unique}")
        print(f"  mean={out.mean().item():.6f} std={out.std().item():.6f}")
        print(f"  sample [0, :8]  = {out[0, :8].tolist()}")
        print(f"  sample [-1, :8] = {out[-1, :8].tolist()}")
        fname = f"dump_{info['name']}.npy"
        np.save(fname, out.numpy())
        print(f"  full array saved to {fname}")


if __name__ == "__main__":
    main()
