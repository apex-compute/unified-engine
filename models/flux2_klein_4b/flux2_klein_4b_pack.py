#!/usr/bin/env python3
"""Pack everything a FLUX.2 [klein] 4B run needs into ONE deployable file.

    flux2_klein_4b.bin
      magic "FLX2KLN1" | u64 header length | JSON header | 4 KiB-aligned sections

Sections: the Qwen3 token embedding, the IF4 text-encoder and DiT weight
images, the IF8 VAE weights, the tokenizer files, and one compiled program
set (every engine's images) per packed configuration -- the weights are
shared, each configuration (resolution x text length x steps x engines)
adds only its ~90 MB of programs. ``flux2_klein_4b_run_from_bin.py`` runs
from this file alone: no network, no HF checkpoint, no per-model JSON.

    python models/flux2_klein_4b/flux2_klein_4b_pack.py --config 256x256x64x4 --config 512x512x64x4 --engines 8 [--out flux2_klein_4b.bin]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import struct
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import flux2_klein_4b_ref as ref  # noqa: E402
import flux2_klein_4b_test as T  # noqa: E402
import flux2_klein_4b_weights as W  # noqa: E402

MAGIC = b"FLX2KLN1"
ALIGN = 4096
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "chat_template.jinja", "vocab.json", "merges.txt", "added_tokens.json")


def read_header(path: str) -> tuple[dict, int]:
    """Return (header dict, byte offset of the first section)."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != MAGIC:
            raise ValueError(f"{path}: not a FLUX.2 klein single bin (magic {magic!r})")
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen).decode())
    return header, 16 + hlen


def config_key(geo: ref.Flux2Geometry) -> str:
    return f"{geo.height}x{geo.width}x{geo.max_sequence_length}x{geo.num_inference_steps}"


def parse_config(spec: str) -> ref.Flux2Geometry:
    h, w, L, steps = (int(v) for v in spec.lower().split("x"))
    return ref.Flux2Geometry(h, w, L, steps)


def pack(cfg: dict, geos: list, engines: int, out_path: str) -> dict:
    paths = T._paths(cfg)
    for k in ("params_te_bin", "params_dit_bin", "vae_weights", "embed_tokens_bin"):
        if not os.path.exists(paths[k]):
            print("  exporting weights (one-time)...", flush=True)
            W.build_all(cfg)
            break
    # one program set per configuration (compiled offline into its own files)
    prog_dir = os.path.join(HERE, cfg["paths"]["bin_dir"], "programs")
    os.makedirs(prog_dir, exist_ok=True)
    programs = {}
    ue = None
    for geo in geos:
        key = config_key(geo)
        ue = T.Flux2KleinEngine(cfg, geo, num_engines=engines, offline=True)
        bin_p, meta_p = os.path.join(prog_dir, f"{key}_e{engines}.bin"), os.path.join(prog_dir, f"{key}_e{engines}.json")
        meta = None
        if os.path.exists(meta_p) and os.path.exists(bin_p):
            with open(meta_p) as f:
                m = json.load(f)
            if m.get("signature") == ue.program_signature():
                meta = m
        if meta is None:
            print(f"  compiling programs for {key}, {engines} engine(s)...", flush=True)
            meta = ue.compile_programs(bin_p, meta_p)
        programs[key] = (bin_p, meta)
    with open(paths["params_te_meta"]) as f:
        te_meta = json.load(f)
    with open(paths["params_dit_meta"]) as f:
        dit_meta = json.load(f)
    W.check_meta_matches_plan(te_meta, ue.te_layout, "te")
    W.check_meta_matches_plan(dit_meta, ue.dit_layout, "dit")

    snap = ref.snapshot_dir(cfg["paths"]["hf_model_repo"])
    sections: list[tuple[str, object]] = []        # (name, path or bytes)
    for fn in TOKENIZER_FILES:
        fp = os.path.join(snap, "tokenizer", fn)
        if os.path.exists(fp):
            sections.append((f"tokenizer/{fn}", fp))
    sections.append(("embed_tokens", paths["embed_tokens_bin"]))
    sections.append(("params_te", paths["params_te_bin"]))
    sections.append(("params_dit", paths["params_dit_bin"]))
    buf = io.BytesIO()
    torch.save(torch.load(paths["vae_weights"], map_location="cpu"), buf)
    sections.append(("vae_weights", buf.getvalue()))
    for key, (bin_p, _meta) in programs.items():
        sections.append((f"programs/{key}", bin_p))

    sizes = {n: (os.path.getsize(src) if isinstance(src, str) else len(src)) for n, src in sections}
    header = {
        "format": "flux2_klein_4b_single_bin", "version": 1,
        "model": cfg["paths"]["hf_model_repo"],
        "configs": {config_key(g): {"height": g.height, "width": g.width, "max_sequence_length": g.max_sequence_length,
                                    "num_inference_steps": g.num_inference_steps} for g in geos},
        "default_config": config_key(geos[0]),
        "engines": engines,
        "config": cfg,
        "programs_meta": {key: meta for key, (_bin, meta) in programs.items()},
        "te_meta": {k: v for k, v in te_meta.items() if k != "embed_tokens"},
        "dit_meta": dit_meta,
        "tokenizer_files": [n.split("/", 1)[1] for n, _ in sections if n.startswith("tokenizer/")],
        "sections": {},
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    # lay out sections after a header whose size we fix by padding the JSON
    hjson = json.dumps(header, sort_keys=True).encode()
    hlen = ((len(hjson) + 4096 + ALIGN - 1) // ALIGN) * ALIGN - 16     # room for the section table
    off = 16 + hlen
    for n, src in sections:
        off = (off + ALIGN - 1) // ALIGN * ALIGN
        header["sections"][n] = {"offset": off, "size": sizes[n]}
        off += sizes[n]
    hjson = json.dumps(header, sort_keys=True).encode()
    assert len(hjson) <= hlen, "header overflow"
    hjson = hjson + b" " * (hlen - len(hjson))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    t0 = time.time()
    with open(out_path, "wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<Q", hlen))
        out.write(hjson)
        for n, src in sections:
            sec = header["sections"][n]
            out.write(b"\0" * (sec["offset"] - out.tell()))
            if isinstance(src, str):
                with open(src, "rb") as f:
                    while True:
                        chunk = f.read(64 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
            else:
                out.write(src)
            assert out.tell() == sec["offset"] + sec["size"]
    total = os.path.getsize(out_path)
    print(f"  packed {len(sections)} sections, {total / 1e9:.2f} GB -> {out_path} in {time.time() - t0:.1f}s", flush=True)
    for n, _ in sections:
        print(f"    {n:32s} {sizes[n] / 1e6:10.1f} MB @ 0x{header['sections'][n]['offset']:X}")
    return header


def main(argv=None) -> int:
    cfg = T.load_config()
    m = cfg["model"]
    ap = argparse.ArgumentParser(description="Pack FLUX.2 klein 4B into one deployable bin")
    ap.add_argument("--config", action="append", default=None,
                    help="HxWxTOKENSxSTEPS, repeatable (default: 256x256x64x4 and 512x512x64x4)")
    ap.add_argument("--engines", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, cfg["paths"]["bin_dir"], "flux2_klein_4b.bin"))
    args = ap.parse_args(argv)
    specs = args.config or ["256x256x64x4", "512x512x64x4"]
    pack(cfg, [parse_config(c) for c in specs], args.engines, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
