"""Exercise the unified server's resident chat worker on five real, reserved FPGA cards."""
import argparse
import os
import re
import signal
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models/unified_server"))
import tp_models
from unified_server import Board, ChatJob


def reserved_cards(value, sysfs=Path("/sys/class/xdma")):
    devices = [v.strip() for v in value.split(",")]
    if (len(devices) != 5 or len(set(devices)) != 5
            or any(not re.fullmatch(r"xdma[0-9]+", d) for d in devices)):
        raise ValueError("Set APEX_TP_DEVICES to five distinct, reserved XDMA cards")
    pci = [(sysfs / f"{d}_user" / "device").resolve(strict=True) for d in devices]
    if len(set(pci)) != 5:
        raise ValueError(f"The devices must refer to five different PCI functions: {pci}")
    return devices


def prepare(model, source, work):
    """Share cached inputs and build the port, decoder programs and logs in a fresh workspace."""
    target = work / "models" / model
    target.mkdir(parents=True)
    slices = "tp" + tp_models.MODELS[model][2]
    shard_dir = target / slices
    shard_dir.mkdir()
    (shard_dir / "plan.json").symlink_to((source / slices / "plan.json").resolve(strict=True))
    for k in range(10):
        card = f"card_{k:02d}"
        (shard_dir / card).mkdir()
        for file in (source / slices / card).iterdir():
            if file.suffix == ".pt" or file.name == "config.json":
                (shard_dir / card / file.name).symlink_to(file.resolve(strict=True))
    (target / "weights").symlink_to((source / "weights").resolve(strict=True))
    if model != "gptoss20b":
        (target / "vision").symlink_to((source / "vision").resolve(strict=True))
    if model in ("q35_9b", "q38_27b"):
        (target / "qwen3.5_2b_bin").symlink_to((source / "qwen3.5_2b_bin").resolve(strict=True))

    # Both MoE runners import the generated Qwen 9B engine class.
    os.environ["APEX_TP_WORK"] = str(work)
    os.environ["GEMMA4_WORK"] = str(work / "models/gemma4_26b")
    os.environ["GPTOSS_WORK"] = str(work / "models/gptoss20b")
    port = "q38_27b" if model == "q38_27b" else "q35_9b"
    config = ["--cfg9b"] if port == "q35_9b" else [
        "--hfcfg", str(target / "qwen3.5_2b_bin/Qwen3.8-27B/config.json")]
    patches = "silu kpad attn1 noeye heads shard recgrp tp abfuse vtshare membase cardsum dynlen ctx16k if4mix"
    subprocess.run([sys.executable, str(tp_models.ROOT / "port_build.py"), "--name", port,
                    *config, *patches.split()], check=True)


def check_reply(job, devices, expected):
    metrics = job.metrics or {}
    resident = metrics.get("resident") or {}
    if resident.get("devices") != devices or resident.get("engines") != 10:
        raise AssertionError(f"Unexpected hardware for {job.model}: {resident}")
    if (metrics.get("generated_tokens", 0) < 1 or metrics.get("stopped")
            or not re.search(expected, job.text, re.IGNORECASE)):
        raise AssertionError(f"Unexpected {job.model} answer: {job.text!r}")
    if job.image and metrics.get("image_tokens", 0) < 1:
        raise AssertionError(f"{job.model} answered without encoding the image")


def run_turn(board, model, devices, case, prompt, expected, image=None):
    job = ChatJob(model, 10, [{"role": "user", "content": prompt}], image, None, 192, {},
                  conversation=case)
    try:
        board._run_chat(job)
        check_reply(job, devices, expected)
    except Exception:
        print("".join(job.log), file=sys.stderr)
        for log in sorted((tp_models.work_dir(model) / "logs").glob("card_*.log")):
            with log.open(errors="replace") as stream:
                print(f"{log.name}:\n" + "".join(deque(stream, maxlen=15)), file=sys.stderr)
        raise
    speed = job.metrics.get("decode_wall_tok_s")
    rate = f", {speed:.2f} decode tokens/s" if speed else ""
    result = f"{model} {case} passed on {', '.join(devices)} (10 engines{rate})"
    print(f"{result}: {job.text.strip()}", flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
            print(f"- {result}", file=summary)


def interrupt(signum, frame):
    raise TimeoutError(f"Five-card test interrupted by {signal.Signals(signum).name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tp_models.MODELS)
    parser.add_argument("--devices", default=os.environ.get("APEX_TP_DEVICES", ""))
    args = parser.parse_args()
    devices = reserved_cards(args.devices)
    source = tp_models.work_dir(args.model).resolve(strict=True)
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGALRM, interrupt)
    signal.alarm(30 * 60)
    with tempfile.TemporaryDirectory(prefix="five-card-") as tmp:
        work = Path(tmp)
        prepare(args.model, source, work)
        board = Board(sys.executable, devices[0], 2, ",".join(devices))
        try:
            run_turn(board, args.model, devices, "geography", "In one short sentence, name the capital of France.", r"\bParis\b")
            run_turn(board, args.model, devices, "arithmetic", "What is 17 plus 25? Answer with the number.", r"\b42\b")
            if args.model != "gptoss20b":
                path = work / "square.png"
                image = Image.new("RGB", (256, 256), "white")
                ImageDraw.Draw(image).rectangle((16, 16, 239, 239), fill="red")
                image.save(path)
                run_turn(board, args.model, devices, "image", "What color is the large square? Answer with its color only.",
                         r"^\s*red[.!]?\s*$", str(path))
        finally:
            board.stop_all()
            signal.alarm(0)


if __name__ == "__main__":
    main()
