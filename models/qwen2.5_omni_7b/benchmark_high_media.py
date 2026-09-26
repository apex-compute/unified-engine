#!/usr/bin/env python3
"""Run only Qwen2.5-Omni's real FPGA media encoders for the High benchmark.

Four same-shape vision invocations replay the bundled image; no LM is loaded.
The output JSON contains hardware-counter times and FLOPs, not a TTFT claim.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time


_MODEL_PATH = os.path.join(os.path.dirname(__file__), "qwen2.5_omni_7b_test.py")
_SPEC = importlib.util.spec_from_file_location("qwen25omni_benchmark_model", _MODEL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot import {_MODEL_PATH}")
model = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = model
_SPEC.loader.exec_module(model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", default="xdma0")
    parser.add_argument("--multi-core", type=int, default=8)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--audio-seconds", type=float, default=6.0)
    args = parser.parse_args()
    if args.frames < 1 or args.audio_seconds <= 0:
        parser.error("frames and audio-seconds must be positive")
    args.image = model._resolve_sample(model.DEFAULT_IMAGE, model.DEFAULT_IMAGE, "--image")
    args.audio = model._resolve_sample(model.DEFAULT_AUDIO, model.DEFAULT_AUDIO, "--audio")
    args.vision_res = "medium"  # 896 x 896, 4096 patches, 1024 soft tokens
    args.prompt = None
    args.target_prefill_tokens = None
    args.prompt_base = "Describe the four camera frames and transcribe the speech."
    args.audio_fraction = 1.0

    with model._exclusive_run_lock():
        engine_kwargs = model.resolve_engine_config(parser, args)
        cfg = model.apply_vision_resolution(
            model.Qwen25OmniUnifiedEngine.load_config(script_dir=model.SCRIPT_DIR),
            args.vision_res,
        )
        params_path = model._weight_mod.ensure_params_bin(model.SCRIPT_DIR)
        processor_dir = model._weight_mod.ensure_processor_bundle(
            model.SCRIPT_DIR, verbose=False)
        processor, processed, _tokens, _prompt, _rendered = (
            model._prepare_processor_inputs(args, cfg, processor_dir)
        )

        # The only software reset in this High suite: the parent benchmark
        # runs the LM op measurements after this subprocess exits cleanly.
        fpga_build = model.reset_selected_engines()
        ue = model.Qwen25OmniUnifiedEngine(
            script_dir=model.SCRIPT_DIR, fpga_build=fpga_build,
            vision_res=args.vision_res, **engine_kwargs,
        )
        ue.configure_runtime_artifacts(params_path, processor_dir)
        ue.tokenizer = processor.tokenizer
        ue.processor = processor

        vision_hw_us = 0.0
        vision_cpu_s = 0.0
        vision_flops = 0.0
        frame_times_us = []
        started = time.perf_counter()
        model._run_vision(ue, processed)
        vision_cpu_s += time.perf_counter() - started
        for frame in range(args.frames):
            if frame:
                started = time.perf_counter()
                ue.run_vision_encoder()
                vision_cpu_s += time.perf_counter() - started
            frame_us = float(ue._vis_latency_us)
            frame_times_us.append(frame_us)
            vision_hw_us += frame_us
            vision_flops += float(ue._vis_total_flops)
            print(f"HIGH frame {frame + 1}/{args.frames}: {frame_us / 1e3:.1f} ms HW", flush=True)
        dims = ue._vision_dims()
        per_frame_model_flops = ue._model_flops_vision(dims)
        if per_frame_model_flops is None:
            raise RuntimeError("vision model FLOP count is unavailable")
        vision_model_flops = float(per_frame_model_flops) * args.frames

        started = time.perf_counter()
        model._run_audio(ue, processed)
        audio_cpu_s = time.perf_counter() - started
        audio_model_flops = ue._model_flops_audio()
        if audio_model_flops is None:
            raise RuntimeError("audio model FLOP count is unavailable")
        peak_gflops = float(ue.vis_peak_gflops())
        result = {
            "hw_version": f"0x{fpga_build:08x}",
            "peak_gflops": peak_gflops,
            "vision": {
                "frames": args.frames,
                "frame_shape": "896x896",
                "patches_per_frame": int(dims["VS"]),
                "soft_tokens": args.frames * int(dims["NUM_MERGED_TOKENS"]),
                "frame_hw_us": frame_times_us,
                "hw_us": vision_hw_us,
                "flops": vision_flops,
                "model_flops": vision_model_flops,
                "cpu_stage_s": vision_cpu_s,
            },
            "audio": {
                "seconds": args.audio_seconds,
                "soft_tokens": int(ue._audio_num_tokens),
                "hw_us": float(ue._audio_latency_us),
                "flops": float(ue._audio_total_flops),
                "model_flops": float(audio_model_flops),
                "cpu_stage_s": audio_cpu_s,
            },
        }
        print("HIGH_MEDIA_RESULT: " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
