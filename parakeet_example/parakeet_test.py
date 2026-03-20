#!/usr/bin/env python3
"""
Parakeet-TDT-0.6B inference on accelerator.

  - Config from parakeet_config.json; weights from NeMo checkpoint.
  - Mel spectrogram computed on host CPU.
  - Encoder (subsampling + 24x conformer) compiled and run on accelerator.
  - TDT decode loop (LSTM predictor + joint network) run on accelerator.

Usage:
  python parakeet_test.py
  python parakeet_test.py --audio test.wav
  python parakeet_test.py --dev xdma0 [--cycle 5.63]

Fixed layout: parakeet_test.py, parakeet_config.json, and parakeet_bin/ live in the same folder.
  user_dma_core.py is one folder up; that parent is added to sys.path.
"""

import json
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import numpy as np
import torch
import torch.nn.functional as F

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, DRAM_INSTRUCTION_ADDR, TYPE, UE_VECTOR_SIZE,
    URAM_NEAR_FULL_ELEMENTS, UnifiedEngine, set_dma_device,
)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path=None):
    """Load parakeet_config.json."""
    if config_path is None:
        config_path = os.path.join(SCRIPT_DIR, "parakeet_config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def pad_to_multiple(n, multiple):
    return ((n + multiple - 1) // multiple) * multiple


# ---------------------------------------------------------------------------
# Host-side mel spectrogram (runs on CPU, not accelerator)
# ---------------------------------------------------------------------------

def compute_mel_spectrogram(waveform, cfg):
    """waveform: (B, samples) float32 → (B, T_mel, 128) bf16."""
    pre = cfg["preprocessing"]
    n_fft = pre["n_fft"]
    hop_length = pre["hop_length"]
    win_length = pre["win_length"]

    # TODO: load actual filterbank and window from weights
    fb = torch.zeros(1, pre["n_mels"], n_fft // 2 + 1)
    window = torch.hann_window(win_length)

    stft = torch.stft(waveform.float(), n_fft, hop_length, win_length,
                       window=window, center=True, pad_mode="reflect",
                       return_complex=True)
    mag = stft.abs()
    power = mag * mag
    mel = torch.matmul(fb, power)                      # (B, 128, T)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    mean = mel.mean(dim=-1, keepdim=True)
    var = mel.var(dim=-1, keepdim=True, unbiased=True)
    std = torch.clamp(torch.sqrt(var), min=1e-5)
    mel = (mel - mean) / std
    return mel.transpose(1, 2).to(torch.bfloat16)      # (B, T_mel, 128)


# ---------------------------------------------------------------------------
# Parakeet Unified Engine
# ---------------------------------------------------------------------------

class Parakeet_UnifiedEngine(UnifiedEngine):
    """UnifiedEngine subclass for Parakeet-TDT-0.6B."""

    def __init__(self, script_dir=None):
        super().__init__(BASE_ADDR=user_dma_core.UE_0_BASE_ADDR,
                         program_dram_base=DRAM_INSTRUCTION_ADDR)
        self.script_dir = script_dir or SCRIPT_DIR
        self._cfg = load_config()

        enc = self._cfg["encoder"]
        pred = self._cfg["predictor"]
        jnt = self._cfg["joint"]
        hw = self._cfg["hardware"]

        self.d_model = enc["d_model"]
        self.num_layers = enc["num_layers"]
        self.num_heads = enc["num_heads"]
        self.head_dim = enc["head_dim"]
        self.ff_dim = enc["ff_dim"]
        self.conv_kernel = enc["conv_kernel"]
        self.conv_pad = enc["conv_pad"]
        self.sub_channels = enc["sub_channels"]
        self.n_mels = enc["n_mels"]

        self.pred_hidden = pred["hidden_size"]
        self.vocab_size = pred["vocab_size"]

        self.joint_hidden = jnt["hidden_size"]
        self.joint_output_padded = jnt["output_size_padded"]
        self.blank_id = jnt["blank_id"]
        self.tdt_durations = jnt["tdt_durations"]
        self.max_symbols_per_step = jnt["max_symbols_per_step"]

        self.block_size = hw["block_size"]
        self.bytes_per_element = enc["bytes_per_element"]

    # -----------------------------------------------------------------------
    # Weight loading
    # -----------------------------------------------------------------------

    def weight_init(self):
        """Load and stage weights from checkpoint into DRAM."""
        # TODO: load NeMo checkpoint, stage conv weights (im2col pad),
        #       pad joint output weights, write all to DRAM.
        pass

    # -----------------------------------------------------------------------
    # Compile: Subsampling
    # -----------------------------------------------------------------------

    def compile_subsampling(self, T_mel):
        """Compile subsampling instruction stream.

        Input: (1, 1, T_mel, 128)
        Output: (1, L, 1024) where L = T_mel // 8

        Stages:
          1. Conv2d(1→256, k=3, s=2):  im2col (N, 64) @ (64, 256)
          2. DWConv2d(256, k=3, s=2) + PWConv2d(256→256):
                per-ch dot(64) + matmul (N, 256) @ (256, 256)
          3. DWConv2d(256, k=3, s=2) + PWConv2d(256→256): same
          4. Linear(4096→1024): matmul (L, 4096) @ (4096, 1024)
        """
        # TODO: emit instructions for each stage
        L = T_mel // 8
        return L

    # -----------------------------------------------------------------------
    # Compile: Conformer encoder (single layer)
    # -----------------------------------------------------------------------

    def compile_conformer_layer(self, L_pad):
        """Compile one conformer block instruction stream.

        All matmul K dims are 64-aligned by construction.
        L_pad = pad_to_multiple(L, 64).

        Sub-modules:
          FF1:  LN → matmul(L_pad,1024)@(1024,4096) → SiLU → matmul(L_pad,4096)@(4096,1024)
          Attn: LN → Q/K/V/Pos proj (1024→1024) → score matmuls → softmax → value matmul → out proj
          Conv: LN → PW1 matmul(1024→2048) → GLU → DW im2col(64,64)@(64,1) → BN → SiLU → PW2 matmul(1024→1024)
          FF2:  same as FF1
          LN_out
        """
        # TODO: emit instructions
        pass

    def compile_encoder(self, T_mel):
        """Compile full encoder: subsampling + 24 conformer layers."""
        L = self.compile_subsampling(T_mel)
        L_pad = pad_to_multiple(L, self.block_size)

        for layer_idx in range(self.num_layers):
            self.compile_conformer_layer(L_pad)

        return L, L_pad

    # -----------------------------------------------------------------------
    # Compile: LSTM predictor step
    # -----------------------------------------------------------------------

    def compile_predictor_step(self):
        """Compile one LSTM predictor step.

        Embedding lookup → 2× LSTM layers:
          Per layer:
            gates_ih: matmul (1, 640) @ (640, 2560)  K=640 ✓
            gates_hh: matmul (1, 640) @ (640, 2560)  K=640 ✓
            add → sigmoid/tanh gate activations → cell/hidden update
        """
        # TODO: emit instructions
        pass

    # -----------------------------------------------------------------------
    # Compile: Joint network
    # -----------------------------------------------------------------------

    def compile_joint(self):
        """Compile joint network step.

        enc_proj:  matmul (1, 1024) @ (1024, 640) + bias   K=1024 ✓
        pred_proj: matmul (1, 640)  @ (640, 640)  + bias   K=640  ✓
        add + ReLU
        output:    matmul (1, 640)  @ (640, 8256)  + bias   K=640 ✓  N=8256 ✓ (padded)
        """
        # TODO: emit instructions
        pass

    # -----------------------------------------------------------------------
    # Run: full pipeline
    # -----------------------------------------------------------------------

    def run_encoder(self, mel_bf16):
        """Run encoder on accelerator.

        mel_bf16: (1, T_mel, 128) bf16 tensor
        Returns: encoder output DRAM address, L, L_pad
        """
        T_mel = mel_bf16.shape[1]
        L, L_pad = self.compile_encoder(T_mel)

        # TODO: write mel to DRAM, execute instruction stream, read back enc_out
        return None, L, L_pad

    def run_decode(self, enc_out_addr, L):
        """Run TDT greedy decode loop.

        For each encoder timestep t:
          While symbols < max_symbols_per_step:
            1. Run predictor step → (1, 640)
            2. Run joint network → (1, 8256)
            3. argmax token_logits[:8193], argmax dur_logits[8193:8198]
            4. If blank: advance t by max(dur, 1), break
               If non-blank: emit token, advance t by dur if dur>0

        Returns: list of token IDs
        """
        # TODO: implement decode loop using compiled predictor + joint programs
        tokens = []
        t = 0
        last_token = self.blank_id
        # predictor_state = None

        while t < L:
            symbols = 0
            while symbols < self.max_symbols_per_step:
                # TODO: run predictor step (last_token, state)
                # TODO: run joint (enc_out[t], pred_out)
                # TODO: read back logits, argmax, decide emit/advance
                break
            t += 1  # placeholder

        return tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parakeet-TDT-0.6B accelerator inference")
    parser.add_argument("--audio", type=str, default=None, help="Path to 16kHz WAV file")
    parser.add_argument("--dev", type=str, default="xdma0", help="XDMA device")
    parser.add_argument("--cycle", type=float, default=5.63, help="Clock cycle in ns")
    args = parser.parse_args()

    cfg = load_config()
    set_dma_device(args.dev)

    print(f"Parakeet-TDT-0.6B on {args.dev}")
    print(f"  Encoder: {cfg['encoder']['num_layers']} conformer layers, d_model={cfg['encoder']['d_model']}")
    print(f"  Predictor: {cfg['predictor']['num_layers']}-layer LSTM, hidden={cfg['predictor']['hidden_size']}")
    print(f"  Joint output: {cfg['joint']['output_size']} → {cfg['joint']['output_size_padded']} (padded)")
    print(f"  Hardware block size: {cfg['hardware']['block_size']}")

    # --- Load audio ---
    if args.audio:
        import torchaudio
        waveform, sr = torchaudio.load(args.audio)
        if sr != cfg["preprocessing"]["sample_rate"]:
            waveform = torchaudio.functional.resample(waveform, sr, cfg["preprocessing"]["sample_rate"])
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
    else:
        # Dummy 5-second audio for testing
        duration_s = 5
        waveform = torch.randn(1, cfg["preprocessing"]["sample_rate"] * duration_s)
        print(f"  Using dummy {duration_s}s audio")

    print(f"  Audio: {waveform.shape[1]} samples ({waveform.shape[1]/cfg['preprocessing']['sample_rate']:.1f}s)")

    # --- Mel spectrogram (CPU) ---
    mel = compute_mel_spectrogram(waveform, cfg)
    T_mel = mel.shape[1]
    L = T_mel // 8
    L_pad = pad_to_multiple(L, cfg["hardware"]["block_size"])
    print(f"  Mel frames: {T_mel}, encoder steps: L={L}, L_pad={L_pad}")

    # --- Init engine ---
    engine = Parakeet_UnifiedEngine()
    engine.weight_init()

    # --- Run encoder ---
    enc_out_addr, L, L_pad = engine.run_encoder(mel)
    print(f"  Encoder done: L={L}, L_pad={L_pad}")

    # --- Run decoder ---
    tokens = engine.run_decode(enc_out_addr, L)
    print(f"  Decoded {len(tokens)} tokens")

    # TODO: detokenize using SentencePiece tokenizer from HF model
    print(f"  Tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")


if __name__ == "__main__":
    main()
