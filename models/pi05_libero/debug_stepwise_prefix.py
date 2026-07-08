#!/usr/bin/env python3
"""Stepwise micro-op HANG isolation for the 18-layer prefix stack (compile_prefix).

Unlike debug_stepwise.py (vision), this does NOT compare against a CPU
reference -- the goal here is to find where a HANG originates, not a
numerics mismatch (flash attention itself, at this exact shape (bucket 16,
extreme uniform -1e36 masking), was already proven clean in isolation via
debug_prefix_attn_isolate.py). So each checkpoint just: compiles the program
up through that op, executes it, and reports whether it returned within the
timeout or hung for real (via the same honest _wait_with_heartbeat check
pi05_libero_test.py already uses -- wait_queue() alone does NOT raise on
timeout).

Usage:
    python debug_stepwise_prefix.py --list                # print all checkpoint names/indices
    python debug_stepwise_prefix.py --stop-after 0         # layer0_ln1 only
    python debug_stepwise_prefix.py --stop-after 10        # through layer0's last head-jump, etc.

Move --stop-after up by one at a time. Whichever index first fails to return
within --timeout is where the hang originates.
"""
import argparse
import sys

import numpy as np
import torch

from pi05_libero_test import Pi05Libero_UnifiedEngine, init_hang_prevention, _CFG
from user_dma_core import DMA_DEVICE_H2C, DMA_DEVICE_C2H


def checkpoint_names(num_layers, num_heads):
    """Names in the same order _debug_op emits them inside compile_prefix,
    without running any HW -- just for --list."""
    names = []
    for l in range(num_layers):
        names.append(f"layer{l}_ln1")
        names.append(f"layer{l}_qkv_proj")
        names.append(f"layer{l}_q_permute")
        names.append(f"layer{l}_kv_cache")
        for h in range(num_heads):
            names.append(f"layer{l}_head{h}")
        names.append(f"layer{l}_attn_permute")
        names.append(f"layer{l}_o_proj_residual")
        names.append(f"layer{l}_ln2")
        names.append(f"layer{l}_mlp_lo")
        names.append(f"layer{l}_mlp_hi")
        names.append(f"layer{l}_mlp_down")
        names.append(f"layer{l}_residual2")
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
                          "+ save the full array to <name>.npy for offline inspection")
    ap.add_argument("--smoke-test", action="store_true",
                     help="use the fast vision_embeddings=None/valid_len=16 smoke test instead of "
                          "real vision (the default). The smoke test did NOT reproduce the real "
                          "hang seen at valid_len=784 with real vision-encoder output -- only use "
                          "this for quick sanity checks, not to trust a 'clean' result.")
    args = ap.parse_args()
    args.real_vision = not args.smoke_test

    if args.list:
        # NUM_HEADS/NUM_LAYERS come from config -- print without touching hardware.
        cfg = _CFG["lm"]
        names = checkpoint_names(cfg["num_layers"], cfg["num_heads"])
        for i, n in enumerate(names):
            print(f"{i:4d}  {n}")
        return

    ue = Pi05Libero_UnifiedEngine()
    names = checkpoint_names(ue.NUM_LAYERS, ue.NUM_HEADS)
    if not (0 <= args.stop_after < len(names)):
        print(f"--stop-after must be in [0, {len(names)-1}]", file=sys.stderr)
        sys.exit(1)

    ue.DEBUG_STOP_AFTER = args.stop_after
    init_hang_prevention(ue)
    # NOTE: tried skipping to _weight_init_lm_prefix() only to cut per-run
    # overhead, but tensor_init() -> _tensor_init_vision() reads VIS_H/VIS_NH/
    # etc, which are only set as a side effect of _weight_init_vision() --
    # AttributeError without it. Full weight_init() it is; the ~269MB vision
    # DMA cost is real per-subprocess overhead but correctness > speed here.
    ue.weight_init()
    ue.tensor_init(_CFG["defaults"].get("max_seq", 512))

    prompt_tokens = np.random.RandomState(1).randint(0, 257152, size=(16,))
    seq_len = _CFG["model"]["prefill_max_seq_len"]

    if args.real_vision:
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
        vision_tokens = ue.run_vision(images)
        valid_len = ue.embed_and_concat_prefix(prompt_tokens, vision_embeddings=vision_tokens, seq_len=seq_len)
    else:
        # Text-only prefix (no vision) -- fast smoke test. Numerically meaningless
        # but exercises the same compiled program shape/instruction pattern.
        # NOTE: this did NOT reproduce the hang seen with real valid_len=784 --
        # use --real-vision to match the actual failing conditions.
        valid_len = ue.embed_and_concat_prefix(prompt_tokens, vision_embeddings=None, seq_len=seq_len)

    ue.start_capture()
    prog_addr = ue.compile_prefix(seq_len, valid_len)
    info = ue._debug_halt_info
    assert info is not None and info["idx"] == args.stop_after, \
        "compile_prefix did not halt at the requested checkpoint -- check _debug_op wiring"
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
        ue._wait_with_heartbeat(f"prefix[{info['name']}]", timeout=args.timeout)
    except RuntimeError as e:
        print(f"HANG at checkpoint {args.stop_after}/{len(names)-1} ({info['name']}): {e}")
        print(f"  bug is between checkpoint {args.stop_after-1} "
              f"({names[args.stop_after-1] if args.stop_after else 'start'}) and {args.stop_after} ({info['name']})")
        sys.exit(1)

    print(f"PASS -- no hang at checkpoint {args.stop_after}/{len(names)-1} ({info['name']}). "
          f"Bump --stop-after to {args.stop_after + 1} "
          f"({names[args.stop_after+1] if args.stop_after+1 < len(names) else 'done, full prefix clean'}) next")

    if args.dump:
        buf = bytearray(info["numel"] * 2)
        ue.dma_read(DMA_DEVICE_C2H, info["addr"], buf, len(buf))
        out = torch.frombuffer(bytes(buf), dtype=torch.bfloat16).reshape(info["shape"]).clone().float()
        n_nan = torch.isnan(out).sum().item()
        n_inf = torch.isinf(out).sum().item()
        n_zero = (out == 0).sum().item()
        n_unique = out.flatten().unique().numel()
        print(f"  dump: shape={tuple(out.shape)} numel={out.numel()}")
        print(f"  nan={n_nan} inf={n_inf} zero={n_zero}/{out.numel()} unique_values={n_unique}")
        print(f"  min={out.min().item():.6f} max={out.max().item():.6f} mean={out.mean().item():.6f} "
              f"std={out.std().item():.6f}")
        print(f"  sample [0, :8]  = {out[0, :8].tolist()}")
        print(f"  sample [-1, :8] = {out[-1, :8].tolist()}")
        fname = f"dump_{info['name']}.npy"
        np.save(fname, out.numpy())
        print(f"  full array saved to {fname}")


if __name__ == "__main__":
    main()
