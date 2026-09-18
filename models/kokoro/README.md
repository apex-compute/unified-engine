# Kokoro-82M

This folder contains the Kokoro-82M TTS bring-up.

## Layout

- **kokoro_test.py** – entry point: weight/voice download, G2P frontend, CLI.
- **kokoro_cpu_reference.py** – the CPU/CUDA reference model (StyleTTS2 + ISTFTNet),
  a self-contained port of upstream `kokoro`. Also the source of the weight tensors
  the accelerator path uploads.
- **kokoro_fpga.py** – the accelerator forward pass on the UnifiedEngine. Single
  engine by default; `--engines N` row-shards the generator's conv taps across N.
- **kokoro_bin/** – frozen single-bin instruction image (`params.bin`,
  `programs.bin`, `programs.json`), keyed on the compiler fingerprint and
  padding caps rather than sequence length.
- **utility/** – bring-up/debug scripts for the single-bin instruction-invariance
  work (not part of the inference path).

## Prerequisites

```bash
pip install -q kokoro soundfile huggingface_hub
apt-get -qq -y install espeak-ng
```

## Usage

From the repo root directory:

```bash
python models/kokoro/kokoro_test.py
python models/kokoro/kokoro_test.py --text "some text" --voice af_heart --speed 1.0
```

Key flags (see `kokoro_test.py`'s argparse for the full list):

- `--prompt` / `--text` – input text to synthesize.
- `--voice` – voice preset (default `af_heart`).
- `--speed` – speech rate multiplier (default `1.0`).
- `--device` – reference device for the CPU/CUDA path.
- `--cuda` – run the CUDA reference model instead of the FPGA.
- `--fpga` – run the accelerator forward pass on the UnifiedEngine.
- `--dev` – FPGA device node (default `xdma0`).
- `--out` – output WAV path (default `kokoro_out.wav` next to this script).
- `--clean` – ignore/rebuild any cached `kokoro_bin/` image.
- `--no-bin-cache` – skip loading/writing the cached bin image entirely.
- `--dump-programs` – dump the compiled instruction programs to a JSON path
  (used by the `utility/` invariance-check tools).
- `--debug-snr` – report per-section SNR against the CPU reference.

## Status

The full forward pass runs on the FPGA via `kokoro_fpga.py`. Every section of
the model is ported: PL-BERT text encoder, prosody/duration prediction,
F0/N prediction, TextEncoder, the ISTFTNet decoder front (AdainResBlk1d
stacks), the Generator body (ConvTranspose1d ups + dilated AdaINResBlock1 +
Snake1D), SineGen/STFT harmonic source, and the exp/sin-cos + block-Toeplitz
iSTFT epilogue (see the section checklist at the top of `kokoro_fpga.py`).

Inference uses a single frozen instruction image cached in `kokoro_bin/`
(`params.bin`, `programs.bin`, `programs.json`). The image is keyed on the
compiler fingerprint and fixed padding caps rather than the exact sequence
length, so one compiled image serves any prompt without recompiling per
input.
