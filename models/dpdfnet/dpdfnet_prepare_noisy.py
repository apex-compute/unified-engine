#!/usr/bin/env python3
"""Download the pinned 20-pair VoiceBank-DEMAND test subset without remixing.

Only selected WAV members are fetched from the official Edinburgh archives
using HTTP byte ranges. Existing files must match their pinned SHA256. The
resulting cases.json is accepted by the noisy-audio evaluation runner.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import struct
import urllib.request
import wave
import zipfile
import zlib


HERE = Path(__file__).resolve().parent
PINNED_CASES = HERE / "noisy_test_cases.json"
DEFAULT_OUTPUT = HERE / "dpdfnet_bin" / "noisy_eval_20260913" / "sources"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_range(archive, start, stop):
    if not 0 <= start <= stop < archive["bytes"]:
        raise ValueError("requested range is outside the pinned archive")
    request = urllib.request.Request(
        archive["url"], headers={"Range": f"bytes={start}-{stop}"})
    with urllib.request.urlopen(request, timeout=45) as response:
        expected = f"bytes {start}-{stop}/{archive['bytes']}"
        if response.status != 206 or response.headers.get("Content-Range") != expected:
            raise RuntimeError("source server did not honor the bounded archive range")
        data = response.read(stop - start + 2)
    if len(data) != stop - start + 1:
        raise RuntimeError("archive range length mismatch")
    return data


class RemoteArchive(io.RawIOBase):
    """Seekable HTTP view used only to read a ZIP's central directory."""

    def __init__(self, metadata):
        self.metadata, self.position = metadata, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        bases = {0: 0, 1: self.position, 2: self.metadata["bytes"]}
        if whence not in bases or bases[whence] + offset < 0:
            raise ValueError("invalid archive seek")
        self.position = bases[whence] + offset
        return self.position

    def read(self, size=-1):
        remaining = max(0, self.metadata["bytes"] - self.position)
        size = remaining if size < 0 else min(size, remaining)
        if not size:
            return b""
        if size > 2_000_000:
            raise RuntimeError("unexpectedly large ZIP-directory request")
        data = read_range(self.metadata, self.position, self.position + size - 1)
        self.position += size
        return data


def extract_member(archive, info, expected_size, expected_crc):
    if info.file_size != expected_size or info.CRC != int(expected_crc, 16):
        raise RuntimeError(f"ZIP metadata differs from pinned member {info.filename}")
    header = read_range(archive, info.header_offset, info.header_offset + 29)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError("invalid ZIP local header")
    flags, compression = struct.unpack_from("<HH", header, 6)
    name_size, extra_size = struct.unpack_from("<HH", header, 26)
    if flags & 1 or compression != info.compress_type:
        raise RuntimeError("encrypted or inconsistent ZIP member")
    start = info.header_offset + 30
    size = name_size + extra_size + info.compress_size
    raw = read_range(archive, start, start + size - 1)
    name = raw[:name_size].decode("utf-8" if flags & 0x800 else "cp437")
    if name != info.filename:
        raise RuntimeError("ZIP local filename differs from its directory entry")
    compressed = raw[name_size + extra_size:]
    if compression == zipfile.ZIP_DEFLATED:
        decoder = zlib.decompressobj(-15)
        data = decoder.decompress(compressed, expected_size + 1)
        if not decoder.eof or decoder.unused_data:
            raise RuntimeError("invalid or oversized ZIP compressed payload")
    elif compression == zipfile.ZIP_STORED:
        data = compressed
    else:
        raise RuntimeError(f"unsupported ZIP compression {compression}")
    if len(data) != expected_size or zlib.crc32(data) != info.CRC:
        raise RuntimeError(f"ZIP size or CRC check failed for {info.filename}")
    return data


def prepare(output_dir, limit=None):
    pinned = json.loads(PINNED_CASES.read_text())
    cases = pinned["cases"] if limit is None else pinned["cases"][:limit]
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for case in cases:
        for kind in ("clean", "noisy"):
            path = output_dir / case[kind]
            if not path.resolve().is_relative_to(output_dir):
                raise ValueError("pinned output path is outside the destination directory")
            if path.exists():
                if sha256(path) != case[kind + "_sha256"]:
                    raise RuntimeError(f"cached source checksum mismatch: {path}")
            else:
                pending.append((case, kind, path))
    catalogs = {}
    for kind in sorted({kind for _, kind, _ in pending}):
        with zipfile.ZipFile(RemoteArchive(pinned["archives"][kind])) as archive:
            catalogs[kind] = {info.filename: info for info in archive.infolist()}

    def download(task):
        case, kind, path = task
        info = catalogs[kind][case[kind + "_member"]]
        data = extract_member(
            pinned["archives"][kind], info,
            case[kind + "_bytes"], case[kind + "_crc32"])
        if hashlib.sha256(data).hexdigest() != case[kind + "_sha256"]:
            raise RuntimeError(f"downloaded source checksum mismatch: {case['id']} {kind}")
        with wave.open(io.BytesIO(data), "rb") as source:
            if (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getnframes()) \
                    != (1, 2, case["sample_rate"], case["samples"]):
                raise RuntimeError(f"unexpected source WAV format: {case['id']} {kind}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".download")
        try:
            temporary.write_bytes(data)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return f"Downloaded {kind}/{path.name}"

    with ThreadPoolExecutor(max_workers=4) as executor:
        for message in executor.map(download, pending):
            print(message, flush=True)
    result = {**pinned, "cases": cases,
              "pinned_manifest_sha256": sha256(PINNED_CASES),
              "download_method": "Official archive byte ranges; ZIP size/CRC and pinned WAV SHA256 verified."}
    manifest = output_dir / "cases.json"
    temporary = manifest.with_suffix(".json.download")
    try:
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(manifest)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Ready: {len(cases)} paired cases, {sum(case['duration_s'] for case in cases):.3f}s")
    print(f"Manifest: {manifest}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int,
                        help="download only the first N pinned cases, for a smaller smoke test")
    args = parser.parse_args()
    count = len(json.loads(PINNED_CASES.read_text())["cases"])
    if args.limit is not None and not 1 <= args.limit <= count:
        parser.error(f"--limit must be between 1 and {count}")
    prepare(args.output_dir, args.limit)


if __name__ == "__main__":
    main()
