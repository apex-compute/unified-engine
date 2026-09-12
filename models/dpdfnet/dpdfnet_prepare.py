#!/usr/bin/env python3
"""Download, validate and audit the official streaming DPDFNet2 ONNX graph."""

import argparse
import json
from pathlib import Path

from dpdfnet_common import DEFAULT_MODEL_PATH, download_model, inspect_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--audit-json", type=Path)
    args = parser.parse_args()

    model = args.model.expanduser().resolve()
    if args.download or args.force_download:
        model = download_model(model, force=args.force_download)
    elif not model.is_file():
        parser.error(f"model not found: {model}; pass --download")

    report = inspect_model(model)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.audit_json is not None:
        output = args.audit_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")
        print(f"Audit: {output}")
    print(rendered)


if __name__ == "__main__":
    main()

