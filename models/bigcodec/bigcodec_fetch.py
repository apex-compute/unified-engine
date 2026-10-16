#!/usr/bin/env python3
"""Fetch and verify the pinned official BigCodec checkpoint."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request


def fetch(output: Path) -> Path:
    manifest = json.loads((Path(__file__).parent / "upstream_manifest.json").read_text())
    checkpoint = manifest["checkpoint"]

    def valid(path):
        if not path.exists() or path.stat().st_size != checkpoint["bytes"]:
            return False
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest() == checkpoint["sha256"]

    if valid(output):
        return output
    if output.exists():
        raise ValueError(f"Existing checkpoint does not match the pinned SHA256: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="bigcodec-", suffix=".part", dir=output.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream, urllib.request.urlopen(checkpoint["url"], timeout=60) as response:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(chunk)
        if not valid(temporary):
            raise ValueError("Downloaded checkpoint failed size/SHA256 verification")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "bigcodec_bin" / "bigcodec.pt")
    args = parser.parse_args()
    print(fetch(args.output))
