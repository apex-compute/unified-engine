"""Fixed-shape benchmark tiers for gemma4_e2b: a voice command (Low), a
single-camera query (Medium), and a multi-camera query (High).

The tiers follow qwen2.5_omni_7b's benchmark.py -- the same camera-frame
counts and total prompt lengths, audio within the same per-tier ranges -- so
the two models can be compared tier by tier:

  tier    frames (896x896)  spoken query  total prompt  decoded
  low     0                 2.55 s        850 tokens    32
  medium  1                 4.5 s         2048 tokens   64
  high    4                 6.0 s         6144 tokens   128

The total prompt length is the fixed part of the shape. Gemma4 E2B encodes a
896x896 frame into 256 soft tokens (Qwen-Omni: ~1024), so the text filler
makes up the rest: ~785 / ~1680 / ~4970 tokens of text. This script owns the shapes;
gemma4_e2b_test.py stays generic (it only knows "--target-prefill-tokens N",
"--prompt-base TEXT", "--audio-seconds S", "--frames N"). Every tier is one
real, continuous FPGA run: the script builds the argv and runs the main
script once, exactly as a user would from the command line.

High's 6144-token prompt is past the LM's 512-row prefill program, so it is
prefilled in chunks (gemma4_e2b_test.py prefill_chunks): each chunk's program
is compiled with its position offset and attends the whole KV-cache prefix,
inside the 8192-token context (MAX_CONTEXT_SIZE).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys

# The model this benchmark drives; gemma4_e4b/benchmark.py repoints these
# three at the E4B launcher and directory and reuses everything else.
MODEL_NAME = "gemma4_e2b"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(SCRIPT_DIR, "gemma4_e2b_test.py")


# ==========================================================================
# WORKLOAD SHAPES (shared with qwen2.5_omni_7b/benchmark.py)
# ==========================================================================
# Input: the whole prompt (media included) is fitted to target_prefill_tokens.
# Output: the prompts ask for more than max_new_tokens of answer, so decode
# runs to that cap rather than stopping early at the end-of-turn token -- the
# decode length is part of the shape (low ~32, medium ~64, high ~128).
PRESETS = {
    "low": {
        "label": "LOW -- voice command",
        "frames": 0,
        "audio_seconds": 2.55,
        "target_prefill_tokens": 850,
        "max_new_tokens": 32,
        "prompt_base": (
            "Transcribe the audio clip exactly. Then, in a separate "
            "paragraph, explain in two or three sentences what the speaker "
            "is talking about."
        ),
    },
    "medium": {
        "label": "MEDIUM -- single camera",
        "frames": 1,
        "audio_seconds": 4.5,
        "target_prefill_tokens": 2048,
        "max_new_tokens": 64,
        "prompt_base": (
            "Do two things, in this order. First, transcribe the speech "
            "exactly as it is spoken, preserving the wording and the order "
            "of the terms. Second, describe the image in detail, covering "
            "the lighting, the terrain, the vegetation and the sky. Both "
            "parts are required."
        ),
    },
    "high": {
        "label": "HIGH -- multi-camera",
        "frames": 4,
        "audio_seconds": 6.0,
        "target_prefill_tokens": 6144,
        "max_new_tokens": 128,
        "prompt_base": (
            "Four camera frames and one audio clip are provided. Do two "
            "things, in this order. First, transcribe the speech exactly "
            "as it is spoken, preserving the wording and the order of the "
            "terms. Second, describe each of the four camera frames in "
            "detail, covering the lighting, the terrain, the vegetation "
            "and the sky. All parts are required."
        ),
    },
}


def _argv_for(spec: dict, args, summary: str) -> list[str]:
    argv = [
        sys.executable, "-u", MAIN_SCRIPT,
        "--dev", args.dev, "--multi-core", str(args.multi_core),
        "--target-prefill-tokens", str(spec["target_prefill_tokens"]),
        "--prompt-base", spec["prompt_base"],
        "--audio", "--audio-seconds", str(spec["audio_seconds"]),
        "--max-new-tokens", str(args.max_new_tokens or spec["max_new_tokens"]),
        "--summary", summary,
    ]
    if spec["frames"]:
        argv += ["--image", "--frames", str(spec["frames"])]
    if args.heartbeat is not None:
        argv += ["--heartbeat"] + ([args.heartbeat] if args.heartbeat else [])
        argv += ["--heartbeat-interval", str(args.heartbeat_interval)]
        if args.heartbeat_precheck:
            argv += ["--heartbeat-precheck"]
    return argv


def _run(argv: list[str], label: str, result_prefix: str = "TEST_RESULT: ") -> dict | None:
    """Run the main script to completion, streaming its output, and return
    its TEST_RESULT payload."""
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}", flush=True)
    print(" ".join(argv), flush=True)
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    result = None
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            if line.startswith(result_prefix):
                result = json.loads(line[len(result_prefix):])
        returncode = proc.wait()
    finally:
        # A killed benchmark must not leave its run holding the board.
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    if returncode:
        raise RuntimeError(f"{label} exited with status {returncode}")
    return result


def _fmt(value, spec: str = ".2f") -> str:
    return "n/a" if value is None else format(value, spec)


def result_table(results: dict[str, dict]) -> list[str]:
    """Markdown table of the headline metrics, one column per tier."""
    tiers = [t for t in ("low", "medium", "high") if t in results]
    rows = [
        ("Camera frames / image tokens",
         lambda r: f"{r['image_frames']} / {r['image_tokens']}"),
        ("Audio (s) / audio tokens",
         lambda r: f"{_fmt(r['audio_seconds'])} / {r['audio_tokens']}"),
        ("Prompt tokens", lambda r: str(r["prompt_tokens"])),
        ("Prefill chunks", lambda r: str(r["prefill_chunks"])),
        ("Vision encoder, HW (s)", lambda r: _fmt(r["vision_hw_s"])),
        ("Audio encoder, HW (s)", lambda r: _fmt(r["audio_hw_s"])),
        ("Prefill, HW (s)", lambda r: _fmt(r["prefill_hw_s"])),
        ("Prefill speed, HW (tok/s)", lambda r: _fmt(r["prefill_hw_tok_s"], ".1f")),
        ("Prefill model GFLOPS (% peak)",
         lambda r: (f"{r['prefill_gflops_model']:.1f} "
                    f"({100 * r['prefill_gflops_model'] / r['peak_gflops']:.1f}%)"
                    if r.get("prefill_gflops_model") else "n/a")),
        ("TTFT, HW counters (s)", lambda r: _fmt(r["ttft_hw_s"])),
        ("TTFT, wall clock (s)", lambda r: _fmt(r["ttft_wall_s"])),
        ("  of which chunk program DMA (s)",
         lambda r: _fmt(r["prefill_program_dma_s"])),
        ("Decoded tokens", lambda r: str(r["decode_tokens"])),
        ("Decode first token, HW (tok/s)",
         lambda r: _fmt(r["decode_first_token_tok_s"], ".1f")),
        ("Decode average, HW (tok/s)", lambda r: _fmt(r["decode_hw_tok_s"], ".1f")),
        ("Decode average, wall clock (tok/s)", lambda r: _fmt(r["decode_avg_tok_s"], ".1f")),
        ("Decode model GFLOPS (% peak)",
         lambda r: (f"{r['decode_gflops_model']:.1f} "
                    f"({100 * r['decode_gflops_model'] / r['peak_gflops']:.1f}%)"
                    if r.get("decode_gflops_model") else "n/a")),
        ("Final context (tokens)", lambda r: str(r["final_seq_len"])),
    ]
    lines = ["| Metric | " + " | ".join(t.capitalize() for t in tiers) + " |",
             "|---|" + "---:|" * len(tiers)]
    for name, cell in rows:
        values = []
        for t in tiers:
            try:
                values.append(cell(results[t]))
            except (KeyError, TypeError, ZeroDivisionError):
                values.append("n/a")
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    return lines


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"{MODEL_NAME} fixed-shape benchmark: --low / --medium / --high / --all")
    parser.add_argument("--dev", default="xdma0")
    parser.add_argument("--multi-core", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="override every tier's decode budget (low 32, medium 64, high 128)")
    parser.add_argument("--summary", default=None, metavar="PATH",
                        help="per-run summary path (single tier only); default "
                             f"{MODEL_NAME}_test_<dev>_<content>_<tier>_multi-core_<N>.md")
    parser.add_argument("--heartbeat", nargs="?", default=None, const="", metavar="LOG",
                        help="forwarded to the main script: sample the engines during "
                             "every prefill program (optional log path)")
    parser.add_argument("--heartbeat-interval", type=float, default=1.0, metavar="S",
                        help="seconds between --heartbeat samples. Default: 1")
    parser.add_argument("--heartbeat-precheck", action="store_true",
                        help="forwarded: hw_check.py + temp_check.tcl before every "
                             "sampled FPGA program")
    parser.add_argument("--report", default=None, metavar="PATH",
                        help="also write the tier metric table (and results JSON) to PATH")
    tier = parser.add_mutually_exclusive_group(required=True)
    tier.add_argument("--low", action="store_true",
                      help="voice command: 2.55 s audio, 850-token prompt, 32 decoded")
    tier.add_argument("--medium", action="store_true",
                      help="single camera: 1 frame + 4.5 s audio, 2048-token prompt, 64 decoded")
    tier.add_argument("--high", action="store_true",
                      help="multi-camera: 4 frames + 6 s audio, 6144-token prompt, 128 decoded "
                           "(chunked prefill)")
    tier.add_argument("--all", action="store_true", help="low, medium and high, in that order")
    return parser


def main() -> None:
    # SIGTERM unwinds like Ctrl-C, so _run's cleanup stops the child run.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    args = build_arg_parser().parse_args()
    tiers = (["low", "medium", "high"] if args.all
             else [t for t in ("low", "medium", "high") if getattr(args, t)])
    if args.summary and len(tiers) > 1:
        raise SystemExit("--summary names one run's summary; use --report with --all")
    results = {}
    for t in tiers:
        spec = PRESETS[t]
        content = "image+audio" if spec["frames"] else "audio"
        summary = args.summary or os.path.join(
            SCRIPT_DIR, f"{MODEL_NAME}_test_{args.dev}_{content}_{t}_multi-core_{args.multi_core}.md")
        results[t] = _run(_argv_for(spec, args, summary), spec["label"])
    table = result_table({t: r for t, r in results.items() if r})
    print("\n" + "\n".join(table))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(f"# {MODEL_NAME} benchmark\n\n" + "\n".join(table) + "\n\n```json\n"
                         + json.dumps(results, indent=2, ensure_ascii=False) + "\n```\n")
        print(f"\nWrote benchmark report: {args.report}")


if __name__ == "__main__":
    main()
