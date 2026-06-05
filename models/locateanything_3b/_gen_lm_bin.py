#!/usr/bin/env python3
"""Generate the LocateAnything-3B LM weight bin (if4) for the UnifiedEngine.

Mirrors qwen2.5_vl_3b_test.py::generate_lm_weights EXACTLY (same _weight_key
mapping, same if4 quant scope, same wire format, appended tied lm_head) but
reads the HF safetensors directly via a shim — so it needs neither the FPGA
hardware libs (user_dma_core) nor the stock LocateAnything model (which only
loads under transformers 4.57.1). Runs anywhere quant_lib imports.

The decoder is dimensionally identical to Qwen2.5-VL-3B's LM; only the vocab
(151936 -> 152681) and the plain-1D rope differ, neither of which touches this
bin. Output: locateanything_3b_bin/locateanything_3b_lm_if4.bin (+ .json).
"""
import glob
import json
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, REPO_ROOT)
import quant_lib  # noqa: E402

MODEL_DIR = os.path.join(SCRIPT_DIR, "locateanything_3b_bin", "LocateAnything-3B")
OUT_BIN = os.path.join(SCRIPT_DIR, "locateanything_3b_bin", "locateanything_3b_lm_if4.bin")

# Pad vocab 152681 -> 152704 (= 64*2386, divisible by 16/32/64) so the decoder's
# LM-head PBI matmul (N=vocab) can find a uniform N strip. Pad rows are zeros ->
# logit 0, harmless for greedy argmax. embedding_vocab in the FPGA config MUST match.
PAD_VOCAB = 152704


def _pad_vocab(t):
    if t.shape[0] >= PAD_VOCAB:
        return t
    pad = torch.zeros(PAD_VOCAB - t.shape[0], t.shape[1], dtype=t.dtype)
    return torch.cat([t, pad], dim=0)

# Manifest keys MUST match what qwen2.5_vl_3b_test.py::weight_init reads (the
# loader is the contract). It uses FULL 'language_model.*' names with the
# '.model' segment dropped, e.g.:
#   language_model.embed_tokens.weight                       (bf16)
#   language_model.layers.{i}.self_attn.q_proj.weight.if4    (quantized)
#   language_model.layers.{i}.self_attn.v_proj.weight        (bf16)
#   language_model.layers.{i}.self_attn.q_proj.bias          (bf16)
#   language_model.layers.{i}.input_layernorm.weight         (bf16)
#   language_model.norm.weight                               (bf16)
#   lm_head.weight.if4                                       (quantized, tied)
# q/k/gate/up weights are quantized; v/o AND down_proj weights, biases, norms,
# embed bf16. (LA fork: down_proj moved to bf16 vs the Qwen baseline — must stay
# in sync with la_decoder_engine.py::_LM_QUANT_LAYERS and the down_weight loader.)
_LM_QUANT_SUFFIXES = ('self_attn.q_proj.weight', 'self_attn.k_proj.weight',
                      'self_attn.o_proj.weight',
                      'mlp.gate_proj.weight', 'mlp.up_proj.weight',
                      'mlp.down_proj.weight')


def _qs_pack(precision, tensor):
    bf = tensor.detach().to(torch.bfloat16).cpu()
    if bf.dim() == 2 and bf.shape[1] % 64 == 0:
        n_blocks = bf.numel() // 64
        data_bytes, scale_bytes = quant_lib.quantize(precision, bf, block_size=64)
        return np.frombuffer(scale_bytes + data_bytes, dtype=np.uint8), n_blocks
    bf = bf.flatten()
    n_blocks = (bf.numel() + 63) // 64
    if bf.numel() != n_blocks * 64:
        bf = torch.nn.functional.pad(bf, (0, n_blocks * 64 - bf.numel()))
    bf = bf.view(1, -1)
    data_bytes, scale_bytes = quant_lib.quantize(precision, bf, block_size=64)
    return np.frombuffer(scale_bytes + data_bytes, dtype=np.uint8), n_blocks


def main(precision="if4"):
    from safetensors.torch import load_file
    sd_full = {}
    for f in sorted(glob.glob(os.path.join(MODEL_DIR, "*.safetensors"))):
        sd_full.update(load_file(f))

    # Map safetensors name -> loader key: drop the '.model' segment.
    #   language_model.model.layers.{i}.X -> language_model.layers.{i}.X
    #   language_model.model.embed_tokens.weight -> language_model.embed_tokens.weight
    #   language_model.model.norm.weight -> language_model.norm.weight
    items = {}
    for k, v in sd_full.items():
        if k.startswith("language_model.model."):
            items["language_model." + k[len("language_model.model."):]] = v
    assert items, "no language_model.model.* keys found"

    json_path = OUT_BIN.rsplit('.', 1)[0] + '.json'
    manifest = {}
    count = 0
    with open(OUT_BIN, 'wb') as f:
        for key, t in items.items():
            if key == "language_model.embed_tokens.weight":
                t = _pad_vocab(t)   # host-side embed table padded to PAD_VOCAB rows
            if any(key.endswith(s) for s in _LM_QUANT_SUFFIXES):
                data, _ = _qs_pack(precision, t)
                raw = data.tobytes()
                key = f'{key}.{precision}'
            else:
                raw = t.to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().tobytes()
            offset = f.tell()
            f.write(raw)
            manifest[key] = {'offset': offset, 'size': len(raw)}
            count += 1
    print(f"Weights: {count} tensors, {os.path.getsize(OUT_BIN)/1048576:.1f} MB -> {OUT_BIN}")

    # Tied LM head from input embedding (weight_init reads lm_head.weight.{prec})
    embed_w = _pad_vocab(sd_full['language_model.model.embed_tokens.weight'].detach().to(torch.bfloat16))
    combined, _ = _qs_pack(precision, embed_w)
    combined_bytes = combined.tobytes()
    with open(OUT_BIN, 'ab') as f:
        offset = f.tell()
        f.write(combined_bytes)
    manifest[f'lm_head.weight.{precision}'] = {'offset': offset, 'size': len(combined_bytes)}
    with open(json_path, 'w') as f:
        json.dump(manifest, f)
    print(f"LM head ({precision}) appended: {len(combined_bytes)/1048576:.1f} MB at 0x{offset:X}")
    print(f"manifest: {len(manifest)} keys -> {json_path}")
    print("sample keys:", sorted(manifest)[:3], "...", f"lm_head.weight.{precision}" in manifest)


if __name__ == "__main__":
    main()
