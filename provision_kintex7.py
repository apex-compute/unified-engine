#!/usr/bin/env python3
"""Provision the unified-engine XC7K480T / MT28GU512 BPI-x16 board over JTAG."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile


PART = "xc7k480t"
REVERSE_BITS = bytes(int(f"{i:08b}"[::-1], 2) for i in range(256))


def validate_image(data):
    """Check the clear 7-series header in a write_cfgmem BPIx16 image.

    BPIx16 reverses bits within each byte and swaps bytes within each word.
    This checks format, EFUSE selection and ciphertext length, not the HMAC
    or whether a supplied AES key decrypts this image.
    """
    if not data or len(data) > 64 * 1024 * 1024 or len(data) % 4:
        raise ValueError("BIN must be nonempty, word aligned, and at most 64 MiB")
    header = bytearray(data[:4096].translate(REVERSE_BITS))
    header[0::2], header[1::2] = header[1::2], header[0::2]
    sync = header.find(bytes.fromhex("aa995566"))
    if sync < 0 or sync % 4:
        raise ValueError("Not a Vivado BPIx16 BIN (expected bit-swapped x16 sync word)")
    offset, mask, ctl, iv_seen = sync + 4, 0, 0, False
    while offset + 4 <= len(header):
        word = struct.unpack_from(">I", header, offset)[0]
        offset += 4
        if word == 0x20000000:  # Type-1 NOOP
            continue
        if word >> 29 != 1 or (word >> 27) & 3 != 2:
            break
        register, count = (word >> 13) & 0x3FFF, word & 0x7FF
        if not count or offset + count * 4 > len(header):
            break
        value = struct.unpack_from(">I", header, offset)[0]
        offset += count * 4
        if register == 6 and count == 1:
            mask = value
        elif register == 5 and count == 1:
            ctl = (ctl & ~mask) | (value & mask)
        elif register == 11 and count == 4:
            iv_seen = True
        elif register == 26 and count == 1:  # encrypted word count
            if ctl & 0x80000040 != 0x80000040 or not iv_seen:
                raise ValueError("BIN must enable AES encryption with the EFUSE key source")
            if value == 0 or value % 4 or offset + value * 4 > len(data):
                raise ValueError("Truncated or invalid encrypted payload in BIN")
            return
    raise ValueError("No supported encrypted 7-series EFUSE header found in BIN")


def read_key(path):
    """Return only validated AES material; never include NKY contents in errors."""
    if path.suffix.lower() != ".nky":
        raise ValueError("Use the matching plaintext .nky key file")
    text = path.read_text(encoding="ascii")
    devices = re.findall(r"\bDevice\s+([^;\s]+)\s*;", text, re.I)
    keys = re.findall(r"\bKey\s+0\s+([0-9a-fA-F_]+)\s*;", text, re.I)
    if len(devices) != 1 or devices[0].lower() != PART:
        raise ValueError(f"NKY must identify Device {PART}")
    if len(keys) != 1:
        raise ValueError("NKY must contain exactly one Key 0 AES key")
    key = keys[0].replace("_", "")
    if len(key) != 64 or int(key, 16) == 0:
        raise ValueError("NKY must contain a nonzero 256-bit AES key")
    return key


def parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=(
        "Default: read-only check. --program permanently burns the matching AES key, "
        "sets FUSE_CNTL 0x0c, then programs and verifies flash. No key is extracted "
        "from the BIN. Source Vivado settings64.sh first."))
    p.add_argument("bin", nargs="?", type=Path, help="encrypted write_cfgmem BPIx16 .bin")
    p.add_argument("--key", type=Path, help="matching .nky; default: BIN with .nky extension")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="read-only preflight (default)")
    mode.add_argument("--list", action="store_true", help="list JTAG targets/devices; no BIN needed")
    mode.add_argument("--program", action="store_true", help="authorize irreversible eFUSE + flash programming")
    mode.add_argument("--flash-only", action="store_true", help="program flash using an already-fused key; never burn fuses")
    p.add_argument("--target", default="", help="exact cable target path or unique cable serial")
    p.add_argument("--device", default="", help="exact JTAG device name, e.g. xc7k480t_0")
    p.add_argument("--dna", default="", help="require this device DNA (hex, as printed by --check)")
    p.add_argument("--server", default="localhost:3121", help="Vivado hw_server address")
    p.add_argument("--vivado", default="vivado", help="Vivado executable")
    p.add_argument("--output-dir", type=Path,
                   default=Path.home() / ".local/state/unified-engine/efuse",
                   help="private directory for per-run NKZ exports and result records")
    p.add_argument("--boot", action="store_true", help="boot from flash and check DONE after programming")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    mode = "list" if args.list else "program" if args.program else "flash" if args.flash_only else "check"
    if args.boot and mode not in {"program", "flash"}:
        p.error("--boot requires --program or --flash-only")
    if args.list and (args.bin or args.key):
        p.error("--list does not take BIN or --key")
    if args.flash_only and args.key:
        p.error("--flash-only uses the existing fused key and does not accept --key")
    if not args.list and args.bin is None:
        p.error("BIN is required (or use --list)")
    if args.dna and not re.fullmatch(r"(?:0[xX])?[0-9a-fA-F_]+", args.dna):
        p.error("--dna must be hexadecimal")
    try:
        image, key, key_path = b"", "", None
        if args.bin:
            if args.bin.suffix.lower() != ".bin":
                raise ValueError("Input must be an encrypted BPIx16 .bin file")
            if args.bin.stat().st_size > 64 * 1024 * 1024:
                raise ValueError("BIN exceeds the board's 64 MiB flash capacity")
            image = args.bin.read_bytes()
            validate_image(image)
            if mode != "flash" or args.key:
                key_path = (args.key or args.bin.with_suffix(".nky")).resolve()
                if not key_path.is_file():
                    raise ValueError(f"Matching NKY not found: {key_path}; supply --key. "
                                     "The AES key cannot be recovered from an encrypted BIN.")
                key = read_key(key_path)
            print(f"BIN: {args.bin.resolve()}\nSHA256: {hashlib.sha256(image).hexdigest()}", flush=True)
            print("Header: encrypted 7-series EFUSE / BPIx16. The key must come from this image's build.", flush=True)
        vivado = shutil.which(args.vivado)
        if not vivado:
            raise ValueError("Vivado not found; source its settings64.sh or use --vivado")
        backend = Path(__file__).with_suffix(".tcl").resolve()
        # All Vivado scratch files and secret NKZ exports are private, including
        # files created by subprocesses. Never run with the repository as cwd.
        old_umask = os.umask(0o077)
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            run_dir = Path(tempfile.mkdtemp(prefix="kintex7-", dir=args.output_dir.resolve()))
            try:
                if image:
                    (run_dir / "image.bin").write_bytes(image)
                if key:
                    # Only the device and AES key are needed to provision eFUSE.
                    (run_dir / "key.nky").write_text(f"Device {PART};\nKey 0 {key};\n", encoding="ascii")
                env = os.environ.copy()
                options = dict(MODE=mode, TARGET=args.target, DEVICE=args.device,
                               DNA=args.dna, SERVER=args.server, BOOT=str(int(args.boot)))
                env.update({f"K7_{name}": value for name, value in options.items()})
                print(f"Run records: {run_dir}", flush=True)
                command = [vivado, "-mode", "batch", "-notrace", "-nolog", "-nojournal",
                           "-source", str(backend)]
                with subprocess.Popen(command, cwd=run_dir, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, errors="replace") as proc:
                    for line in proc.stdout:
                        # Vivado versions can print AES values even without -verbose.
                        print(re.sub(r"(?<![0-9a-fA-F])[0-9a-fA-F_]{64,}(?![0-9a-fA-F])",
                                     "[redacted 256-bit value]", line), end="", flush=True)
                    status = proc.wait()
                # Some Vivado launch failures have historically returned zero.
                if status == 0 and not (run_dir / "SUCCESS").is_file():
                    status = 1
                record = dict(mode=mode, image=str(args.bin.resolve()) if args.bin else None,
                              image_sha256=hashlib.sha256(image).hexdigest() if image else None,
                              key_file=str(key_path) if key_path else None, returncode=status)
                (run_dir / "result.json").write_text(json.dumps(record, indent=2) + "\n")
                return status if status > 0 else (1 if status else 0)
            finally:
                (run_dir / "key.nky").unlink(missing_ok=True)
                (run_dir / "image.bin").unlink(missing_ok=True)
        finally:
            os.umask(old_umask)
    except (OSError, ValueError, UnicodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Inspect device state before retrying provisioning.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
