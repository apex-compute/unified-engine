"""Recheck clean-reference quality of existing bus outputs without hardware."""

from pathlib import Path
import hashlib
import json
import sys

import soundfile as sf


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
from models.dpdfnet.dpdfnet_audio_metrics import evaluate_audio


def main():
    recorded = json.loads((HERE / "bus_clean_reference_quality.json").read_text())
    audio = {}
    for name, entry in recorded["files"].items():
        path = ROOT / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Input SHA256 mismatch: {path}")
        samples, rate = sf.read(path, dtype="float64")
        if rate != recorded["sample_rate"] or samples.shape != (recorded["samples"],):
            raise ValueError(f"Unexpected audio rate or shape: {path}")
        audio[name] = samples

    result = {
        name: evaluate_audio(audio["clean"], samples, recorded["sample_rate"])
        for name, samples in audio.items() if name != "clean"
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
