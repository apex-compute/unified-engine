#!/usr/bin/env python3
"""
Kokoro-82M inference driver.

Entry point for both paths:
  - FPGA (default): kokoro_fpga.run_fpga_forward on the UnifiedEngine accelerator.
  - --cuda: the CPU/CUDA reference in kokoro_cpu_reference.py, a self-contained
    port of hexgrad/kokoro's model code so we no longer depend on the `kokoro`
    pip package.

This file keeps only the things around the model: weight/voice download, the
G2P frontend, and the CLI. The model definition lives in kokoro_cpu_reference.py;
the accelerator forward pass lives in kokoro_fpga.py.

`misaki` + espeak-ng are still a dependency for G2P (text -> IPA phonemes) --
a linguistic front-end, not model weights/forward-pass code, so it is kept as
a dependency rather than reimplemented.

Weights (kokoro-v1_0.pth + config.json) and voice packs are downloaded from
HF on first run and cached under kokoro_bin/.

Usage:
  python models/kokoro/kokoro_test.py
  python models/kokoro/kokoro_test.py --text "some text" --voice af_heart
  python models/kokoro/kokoro_test.py --cuda        # reference path, no hardware
"""
import argparse
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))  # repo root, for user_dma_core

from kokoro_cpu_reference import KokoroModel  # same folder; see the module docstring
BIN_DIR = os.path.join(SCRIPT_DIR, "kokoro_bin")
HF_REPO = "hexgrad/Kokoro-82M"
MODEL_FILENAME = "kokoro-v1_0.pth"

DEFAULT_TEXT = (
    "Hello, this is Kokoro, an open weight text to speech model with "
    "eighty two million parameters."
)


# ---------------------------------------------------------------------------
# Weight / voice download + G2P frontend + CLI driver
# ---------------------------------------------------------------------------

def ensure_weights():
    os.makedirs(BIN_DIR, exist_ok=True)
    model_path = os.path.join(BIN_DIR, MODEL_FILENAME)
    config_path = os.path.join(BIN_DIR, "config.json")
    if not (os.path.exists(model_path) and os.path.exists(config_path)):
        print(f"Model files not found, downloading {HF_REPO} to {BIN_DIR} ...")
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(repo_id=HF_REPO, filename=MODEL_FILENAME, local_dir=BIN_DIR)
        config_path = hf_hub_download(repo_id=HF_REPO, filename="config.json", local_dir=BIN_DIR)
    else:
        print(f"Found cached weights in {BIN_DIR}")
    return model_path, config_path


def ensure_voice(voice: str, device: str):
    voice_path = os.path.join(BIN_DIR, "voices", f"{voice}.pt")
    if not os.path.exists(voice_path):
        print(f"Voice '{voice}' not found, downloading ...")
        from huggingface_hub import hf_hub_download
        voice_path = hf_hub_download(repo_id=HF_REPO, filename=f"voices/{voice}.pt", local_dir=BIN_DIR)
    return torch.load(voice_path, weights_only=True).to(device)


def text_to_phonemes(text: str, british: bool = False) -> str:
    """G2P frontend: text -> IPA phoneme string, matching kokoro's American/British English path."""
    from misaki import en, espeak
    try:
        fallback = espeak.EspeakFallback(british=british)
    except Exception as e:
        print(f"WARNING: EspeakFallback not enabled ({e}); OOD words will be skipped")
        fallback = None
    g2p = en.G2P(trf=False, british=british, fallback=fallback, unk='')
    _, tokens = g2p(text)
    return ''.join(t.phonemes + (' ' if t.whitespace else '') for t in tokens if t.phonemes).strip()


def _shared_report(input_desc, engines, stages, decoded_tokens=None,
                   decode_first_tok_s=None, decode_avg_tok_s=None, peak_gflops=None) -> dict:
    """Shared TEST_RESULT keys that model_auto_test.py reads (_REPORT_REQUIRED / _REPORT_DECODER).

    ``stages`` is ``[(name, hw_seconds, gflop), ...]`` in run order.  ``hw_seconds`` is the
    device latency counter and ``gflop`` the compile-time theoretical FLOP count of what ran,
    in units of 1e9.  A value that was not measured is passed as None and stays null in the
    result -- nothing is estimated here.  ``peak_gflops`` defaults to the per-engine peak
    (128 FLOP per clock cycle per engine) times ``engines``.
    The decoder keys are only emitted when ``decoded_tokens`` is given.
    """
    import user_dma_core as _udc
    if peak_gflops is None:
        _hz = getattr(_udc, "CORE_CLOCK_FREQ_HZ", None)
        per_engine = _hz * 128 / 1e9 if _hz else None  # GFLOP/s per engine
        peak_gflops = per_engine * engines if (per_engine and engines) else None

    def _num(value, digits=4):
        return round(float(value), digits) if value is not None else None

    def _pct(rate):
        return _num(100.0 * rate / peak_gflops, 2) if (rate is not None and peak_gflops) else None

    rows = []
    for name, hw_s, gflop in stages:
        rate = gflop / hw_s if (gflop is not None and hw_s) else None
        rows.append({"name": name, "hw_s": _num(hw_s, 6), "gflop": _num(gflop, 6),
                     "gflops": _num(rate), "pct_peak": _pct(rate)})
    hw_all = [hw_s for _, hw_s, _ in stages]
    gflop_all = [gflop for _, _, gflop in stages]
    total_hw_s = sum(hw_all) if (hw_all and all(v is not None for v in hw_all)) else None
    total_gflop = sum(gflop_all) if (gflop_all and all(v is not None for v in gflop_all)) else None
    avg_gflops = total_gflop / total_hw_s if (total_gflop is not None and total_hw_s) else None
    report = {
        "input": input_desc,
        "engines": engines,
        "peak_gflops": _num(peak_gflops),
        "stages": rows,
        "total_hw_s": _num(total_hw_s, 6),
        "total_gflop": _num(total_gflop, 6),
        "avg_gflops": _num(avg_gflops),
        "avg_pct_peak": _pct(avg_gflops),
    }
    if decoded_tokens is not None:
        report.update({"decoded_tokens": decoded_tokens,
                       "decode_first_tok_s": _num(decode_first_tok_s, 3),
                       "decode_avg_tok_s": _num(decode_avg_tok_s, 3)})
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Kokoro-82M on the UnifiedEngine FPGA accelerator (default); "
                    "--cuda runs the stock CUDA/CPU reference instead.")
    parser.add_argument("--prompt", "--text", dest="prompt", type=str, default=DEFAULT_TEXT,
                         help="Text to speak. (--text is kept as an alias.)")
    parser.add_argument("--voice", type=str, default="af_heart")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None,
                         help="Torch device for the reference path (default: cuda if available, "
                              "else cpu). Ignored on the FPGA path, which always builds its "
                              "reference/comparison tensors on the host.")
    parser.add_argument("--out", type=str, default=os.path.join(SCRIPT_DIR, "kokoro_out.wav"))
    parser.add_argument("--cuda", action="store_true",
                         help="Run the stock CUDA/CPU KokoroModel reference path instead of the "
                              "FPGA accelerator. Use --device to pick cuda vs cpu.")
    parser.add_argument("--fpga", action="store_true",
                         help="Run on the UnifiedEngine FPGA accelerator. This is now the DEFAULT, "
                              "so the flag is only needed to be explicit (or to override nothing). "
                              "Built section-by-section (see kokoro_fpga.py); sections not yet "
                              "ported fall back to the CUDA/CPU path, compared via SNR against the "
                              "FPGA section's own output.")
    parser.add_argument("--dev", type=str, default="xdma0",
                         help="XDMA device name for the FPGA path (e.g. xdma0).")
    parser.add_argument("--dump-programs", type=str, default=None, dest="dump_programs",
                         help="Write per-program instruction fingerprints (section, count, sha1) "
                              "to this JSON path. Diff two prompt lengths to see which captured "
                              "programs are already prompt-independent.")
    parser.add_argument("--clean", action="store_true",
                         help="Delete the frozen instruction image (kokoro_bin/params.bin, "
                              "programs.bin, programs.json) before running, so this run "
                              "recompiles and saves a fresh one. Model weights and voices are kept.")
    parser.add_argument("--no-bin-cache", action="store_true", dest="no_bin_cache",
                         help="Ignore kokoro_bin/programs.bin and recompile the instruction "
                              "streams from scratch. The cache is keyed on the compiler "
                              "fingerprint and the capacity caps -- deliberately NOT on the "
                              "sequence length, since one image is meant to serve every prompt.")
    parser.add_argument("--engines", type=int, default=1,
                         help="Engine count. Only 1 is supported: see the check below.")
    parser.add_argument("--no-snr", action="store_false", dest="debug_snr",
                         help="Skip the per-section SNR bisects against the CPU reference. "
                              "They run by default; pass this to opt out.")
    parser.set_defaults(debug_snr=True)
    args = parser.parse_args()
    # Multi-engine is not supported. The sharded generator only exists on the fused path,
    # which the default per-section SNR checks bypass (they pass a CPU reference, so the
    # generator ran on one engine whatever --engines said); and on the fused path the
    # sharded AdaIN stats staging is sized for many engines -- at 2 it overflows at compile
    # time (_stats_sharded: 32896 rows per engine vs 8192 staged). Refuse rather than
    # silently run on one engine.
    if args.engines != 1:
        parser.error(f"--engines {args.engines}: kokoro supports a single engine only "
                     f"(the multi-engine generator does not work yet)")

    # FPGA is the default; --cuda opts out to the stock reference path.
    use_fpga = not args.cuda
    if args.cuda and args.fpga:
        parser.error("--cuda and --fpga are mutually exclusive")

    if use_fpga:
        args.device = "cpu"  # reference/comparison tensors still computed on host
    elif args.device is None:
        args.device = "cuda"

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    model_path, config_path = ensure_weights()

    print(f"Loading Kokoro model on {args.device} ...")
    model = KokoroModel(config_path)
    model.load_weights(model_path)
    model = model.to(args.device).eval()

    ref_s = ensure_voice(args.voice, args.device)

    print(f"Phonemizing text: {args.prompt!r}")
    phonemes = text_to_phonemes(args.prompt, british=args.voice.startswith("b"))
    print(f"Phonemes: {phonemes}")

    if use_fpga:
        from kokoro_fpga import run_fpga_forward, FROZEN_IMAGE_FILES
        if args.clean:
            for name in FROZEN_IMAGE_FILES:
                path = os.path.join(BIN_DIR, name)
                if os.path.exists(path):
                    os.remove(path)
            print(f"--clean: removed the frozen instruction image from {BIN_DIR}; "
                  f"this run recompiles and saves a new one")
        print("Running FPGA inference (only sections currently ported to hardware) ...")
        audio = run_fpga_forward(model, phonemes, ref_s[len(phonemes) - 1], speed=args.speed,
                                 dev=args.dev, debug=args.debug_snr,
                                 dump_programs=args.dump_programs,
                                 bin_cache=(None if args.no_bin_cache else BIN_DIR),
                                 engines=args.engines)
        if audio is None:
            return  # see kokoro_fpga.py's section checklist
        import soundfile as sf
        sf.write(args.out, audio.cpu().numpy(), 24000, subtype='PCM_16')
        print(f"\nWrote {len(audio) / 24000:.2f}s of audio to {os.path.abspath(args.out)}")

        # One TEST_RESULT line for model_auto_test.py. Per section: hw_s is the device latency
        # counter summed over the section's programs, gflop the theoretical FLOPs of what those
        # programs ran at this prompt's real lengths (see kokoro_fpga.py, "TEST_RESULT accounting").
        import json
        from kokoro_fpga import get_test_report
        rep = get_test_report()
        audio_s = len(audio) / 24000
        snrs = rep["snr_db"]
        worst = min(snrs, key=lambda sv: sv[1]) if snrs else None
        output = f"{audio_s:.2f} s of audio written ({len(audio)} samples at 24 kHz, {os.path.basename(args.out)})"
        if worst is not None:
            output += f"; worst of {len(snrs)} section SNRs vs CPU: {worst[1]:.2f} dB (section {worst[0]})"
        result = dict(_shared_report(
            {"prompt": args.prompt,
             "voice": f"{args.voice}, speed {args.speed:g} ({len(args.prompt)} chars, {len(phonemes)} phonemes)"},
            rep["engines"], rep["stages"]))
        result["output"] = output
        result["audio_s"] = round(audio_s, 3)
        result["min_section_snr_db"] = worst[1] if worst is not None else None
        result["section_snr_db"] = [[sec, val] for sec, val in snrs]
        result["programs_run"] = rep["programs_run"]
        result["instructions"] = rep["instructions"]
        result["from_bin"] = rep["from_bin"]
        result["flop_count_fallbacks"] = rep["flop_count_fallbacks"]
        result["debug_snr"] = bool(args.debug_snr)
        print("TEST_RESULT:" + json.dumps(result))
        return

    print(f"Running {args.device.upper()} reference inference ...")
    output = model(phonemes, ref_s[len(phonemes) - 1], speed=args.speed)

    import soundfile as sf
    sf.write(args.out, output.audio.numpy(), 24000, subtype='PCM_16')
    print(f"\nWrote {len(output.audio) / 24000:.2f}s of audio to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
