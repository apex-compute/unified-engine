"""Download the Hugging Face checkpoints into the places the slicing scripts read them from.

    python tools/download_models.py gpt-oss-20b          # or: gemma-4-26b, qwen3.5-9b, qwen3.8-27b, all

  qwen3.5-9b   Qwen/Qwen3.5-9B     -> $APEX_TP_WORK/models/q35_9b/qwen3.5_2b_bin/Qwen3.5-9B      (~19 GB)
  qwen3.8-27b  Qwen/Qwen3.8-27B    -> $APEX_TP_WORK/models/q38_27b/qwen3.5_2b_bin/Qwen3.8-27B    (~52 GB)
  gemma-4-26b  google/gemma-4-26b-a4b-it -> $GEMMA4_MODEL (default ~/apex_exp/models/gemma4_26b/hf/gemma-4-26b-a4b-it, ~52 GB)
  gpt-oss-20b  openai/gpt-oss-20b  -> $GPTOSS_MODEL (default ~/apex_exp/models/gpt_oss_20b/hf/gpt-oss-20b, ~13 GB;
                                      the original/ and metal/ copies of the weights are skipped)

Host-side preparation only; nothing here touches the cards."""
import os
import sys
import time

from huggingface_hub import snapshot_download

WORK = os.path.expanduser(os.environ.get("APEX_TP_WORK", "~/apex_tp"))
MODELS = {
    "qwen3.5-9b": ("Qwen/Qwen3.5-9B", os.path.join(WORK, "models", "q35_9b", "qwen3.5_2b_bin", "Qwen3.5-9B"), None),
    "qwen3.8-27b": ("Qwen/Qwen3.8-27B", os.path.join(WORK, "models", "q38_27b", "qwen3.5_2b_bin", "Qwen3.8-27B"), None),
    "gemma-4-26b": ("google/gemma-4-26b-a4b-it",
                    os.path.expanduser(os.environ.get("GEMMA4_MODEL", "~/apex_exp/models/gemma4_26b/hf/gemma-4-26b-a4b-it")),
                    None),
    "gpt-oss-20b": ("openai/gpt-oss-20b",
                    os.path.expanduser(os.environ.get("GPTOSS_MODEL", "~/apex_exp/models/gpt_oss_20b/hf/gpt-oss-20b")),
                    ["original/*", "metal/*"]),
}

names = sys.argv[1:] or ["all"]
if names == ["all"]:
    names = list(MODELS)
for name in names:
    if name not in MODELS:
        raise SystemExit("unknown model %r (choose from: %s, all)" % (name, ", ".join(MODELS)))
    repo, dst, skip = MODELS[name]
    os.makedirs(dst, exist_ok=True)
    t = time.time()
    print("START %s -> %s" % (repo, dst), flush=True)
    snapshot_download(repo, local_dir=dst, ignore_patterns=skip, max_workers=8)
    print("DONE %s in %.0f s" % (repo, time.time() - t), flush=True)
