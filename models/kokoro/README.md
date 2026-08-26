# Kokoro-82M

This folder contains the Kokoro-82M TTS bring-up.

## Layout

- **kokoro_test.py** – Stage 1: plain CUDA reference inference via the upstream `kokoro` package.
- **kokoro_bin/** – Weights + config (downloaded from HF on first run).

## Prerequisites

```bash
pip install -q kokoro soundfile huggingface_hub
apt-get -qq -y install espeak-ng
```

## Usage

From the repo root directory:

```bash
python models/kokoro/kokoro_test.py
python models/kokoro/kokoro_test.py --text "some text" --voice af_heart --device cuda
```

## Status

Stage 1 only: downloads weights and runs reference CUDA inference to confirm the
model works end to end. Porting to our accelerator convention (weight dump to
bin/, custom kernels) comes next, following the pattern in models/parakeet.
