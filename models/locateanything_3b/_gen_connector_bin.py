#!/usr/bin/env python3
"""Generate the LocateAnything-3B connector (mlp1) weight bin for the UnifiedEngine.

The connector is nn.Sequential applied to the [N,4608] MoonViT features:
    0: LayerNorm(4608)        -> gamma + beta
    1: Linear(4608 -> 2048)   -> weight [2048,4608] + bias [2048]
    2: GELU()                 -> exact erf-GELU (engine uses sigmoid-approx; see note)
    3: Linear(2048 -> 2048)   -> weight [2048,2048] + bias [2048]

Structurally identical to qwen2.5_vl_3b's vision PatchMerger (norm -> matmul(gelu)
-> matmul), except the norm is a real LayerNorm (gamma+beta, mean-subtract) rather
than RMSNorm. We keep the two Linears in bf16 (not if4) so the only numerical
difference vs the bit-exact GPU connector is the engine's GELU approximation --
this isolates that variable for Stage-2 validation. ~27 MB.

Reads the HF safetensors directly (no FPGA libs, no stock model). Output:
    locateanything_3b_bin/locateanything_3b_connector.bin (+ .json)
"""
import glob
import json
import os

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "locateanything_3b_bin", "LocateAnything-3B")
OUT_BIN = os.path.join(SCRIPT_DIR, "locateanything_3b_bin", "locateanything_3b_connector.bin")

# nn.Sequential indices in our LocateAnything.mlp1 (matches HF safetensors keys).
_KEYS = [
    "mlp1.0.weight",  # LayerNorm gamma [4608]
    "mlp1.0.bias",    # LayerNorm beta  [4608]
    "mlp1.1.weight",  # Linear0 weight  [2048,4608]
    "mlp1.1.bias",    # Linear0 bias    [2048]
    "mlp1.3.weight",  # Linear1 weight  [2048,2048]
    "mlp1.3.bias",    # Linear1 bias    [2048]
]


def main():
    from safetensors.torch import load_file
    sd = {}
    for f in sorted(glob.glob(os.path.join(MODEL_DIR, "*.safetensors"))):
        sd.update(load_file(f))
    missing = [k for k in _KEYS if k not in sd]
    assert not missing, f"connector keys missing from safetensors: {missing}"

    manifest = {}
    with open(OUT_BIN, "wb") as f:
        for k in _KEYS:
            raw = sd[k].detach().to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().tobytes()
            offset = f.tell()
            f.write(raw)
            manifest[k] = {"offset": offset, "size": len(raw), "shape": list(sd[k].shape)}
    with open(OUT_BIN.rsplit(".", 1)[0] + ".json", "w") as f:
        json.dump(manifest, f)
    print(f"connector: {len(_KEYS)} tensors, {os.path.getsize(OUT_BIN)/1048576:.1f} MB -> {OUT_BIN}")
    for k in _KEYS:
        print(f"  {k:20s} {manifest[k]['shape']}")


if __name__ == "__main__":
    main()
