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
                         help="Accelerator engines to row-shard the generator's conv taps across "
                              "(1 = single engine). >1 compiles every run for now (no bin cache).")
    parser.add_argument("--no-snr", action="store_false", dest="debug_snr",
                         help="Skip the per-section SNR bisects against the CPU reference. "
                              "They run by default; pass this to opt out.")
    parser.set_defaults(debug_snr=True)
    args = parser.parse_args()

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
        return

    print(f"Running {args.device.upper()} reference inference ...")
    output = model(phonemes, ref_s[len(phonemes) - 1], speed=args.speed)

    import soundfile as sf
    sf.write(args.out, output.audio.numpy(), 24000, subtype='PCM_16')
    print(f"\nWrote {len(output.audio) / 24000:.2f}s of audio to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
