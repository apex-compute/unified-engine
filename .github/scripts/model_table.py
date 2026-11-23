#!/usr/bin/env python3
"""One row per model for the top of the HW CI job summary.

Reads what the CI job leaves in the workspace and prints a Markdown table:

  model_auto_test_results*.txt            kintex7 (p620) model round
  run_ci.log                              kintex7 run_ci.sh output (conv model verdicts)
  alveo_u50/...                           the same for the Alveo U50 (p2 xdma0)
                                          (its log is alveo_u50/run.log)

Status is one verdict per model over every board, registry entry and pass that
tested it: FAIL if any of them failed, PASS if at least one ran and none failed,
"-" if nothing tested it in this run (a skipped entry does not count either way).

Weight bin and DRAM usage are read from the run. A cell shows one value; when the
two boards differ it shows "kintex7 / Alveo U50". Type and multi-core support are
a fixed table (MODELS below).

usage: model_table.py [workspace dir, default .]
"""
import os
import re
import sys

# name, type, multi-core support, model_auto_test.py registry entries of this model.
# conv=True: the model is run by user_hw_test.py conv_model_tests(), single engine.
SINGLE, MULTI, ONLY8, NONE = "single only", "2 / 8", "8 only", "no callsites"
MODELS = [
    ("gemma3",            "LM",            MULTI,  ["gemma3"], False),
    ("llama3.2_1b",       "LM",            MULTI,  ["llama3.2_1b"], False),
    ("llama3.2_3b",       "LM",            SINGLE, ["llama3.2_3b"], False),
    ("qwen3_0.6b",        "LM",            MULTI,  ["qwen3_0.6b"], False),
    ("qwen3_1.7b",        "LM",            SINGLE, ["qwen3_1.7b"], False),
    ("qwen3_4b",          "LM",            SINGLE, ["qwen3_4b"], False),
    ("qwen3.5_2b",        "VLM",           MULTI,  ["qwen3.5_2b", "qwen3.5_2b_vlm"], False),
    ("gemma4_e2b",        "VLM",           MULTI,  ["gemma4_e2b", "gemma4_e2b_run_from_bin"], False),
    ("gemma4_e4b",        "VLM",           MULTI,  ["gemma4_e4b", "gemma4_e4b_run_from_bin"], False),
    ("qwen2.5_vl_3b",     "VLM",           MULTI,  ["qwen2.5_vl_3b"], False),
    ("smolvlm2",          "VLM",           MULTI,  ["smolvlm2"], False),
    ("locateanything_3b", "VLM-Grounding", SINGLE, ["locateanything_3b"], False),
    ("qwen2.5_omni_7b",   "Omni",          ONLY8,
     ["qwen2.5_omni_7b", "qwen2.5_omni_7b_vlm", "qwen2.5_omni_7b_audio", "qwen2.5_omni_7b_joint"], False),
    ("alpamayo2_edge",    "VLA",           ONLY8,  ["alpamayo2_edge"], False),
    ("pi05",              "VLA",           MULTI,  ["pi05"], False),
    ("verapulse",         "VLA",           MULTI,  ["verapulse"], False),
    ("act",               "Action-Policy", MULTI,  ["act"], False),
    ("flux2_klein_4b",    "T2I",           MULTI,  ["flux2_klein_4b", "flux2_klein_4b_vae"], False),
    ("moonshine",         "ASR",           MULTI,  ["moonshine"], False),
    ("parakeet",          "ASR",           SINGLE, ["parakeet"], False),
    ("kokoro",            "TTS",           SINGLE, ["kokoro"], False),
    ("bigcodec",          "Audio-Codec",   SINGLE, ["bigcodec"], True),
    ("dpdfnet",           "Audio-Enhance", SINGLE, ["dpdfnet"], True),
    ("dpdfnet8khz",       "Audio-Enhance", SINGLE, ["dpdfnet8khz"], True),
    ("mobilenetv2_224",   "Vision-Cls",    SINGLE, ["mobilenetv2_224"], False),
    ("swin",              "Vision-Cls",    SINGLE, ["swin"], False),
    ("mobilenetv2_ssd",   "Vision-Det",    SINGLE, ["mobilenetv2_ssd"], False),
    ("yolov5s",           "Vision-Det",    SINGLE, ["yolov5s"], True),
    ("yolov5n",           "Vision-Det",    SINGLE, ["yolov5n"], True),
    ("yolo26s",           "Vision-Det",    NONE,   ["yolo26s"], False),
    ("yolo26n",           "Vision-Det",    NONE,   ["yolo26n"], False),
    ("mobilesam",         "Vision-Seg",    SINGLE, ["mobilesam"], False),
    ("unet",              "Vision-Seg",    SINGLE, ["unet"], True),
]

# The bin each conv model runs by default (user_hw_test.py conv_model_tests()).
CONV_BINS = {
    "yolov5s":     "models/yolov5s/yolov5s_bin/yolov5s-andromeda_winograd.bin",
    "yolov5n":     "models/yolov5n/yolov5n_bin/yolov5n-andromeda_winograd.bin",
    "unet":        "models/unet/unet_bin/unet-andromeda.bin",
    "dpdfnet":     "models/dpdfnet/dpdfnet_bin/dpdfnet2-andromeda.bin",
    "dpdfnet8khz": "models/dpdfnet8khz/dpdfnet8khz_bin/dpdfnet2_8khz-andromeda.bin",
    "bigcodec":    "models/bigcodec/bigcodec_bin/bigcodec-andromeda.bin",
}

BOARDS = [("kintex7", ""), ("alveo_u50", "alveo_u50/")]   # display order: kintex7 / Alveo U50
REPORTS = ["model_auto_test_results.txt", "model_auto_test_results_pass1.txt",
           "model_auto_test_results_shuffle.txt"]
LOGS = {"kintex7": "run_ci.log", "alveo_u50": "alveo_u50/run.log"}


def parse_report(path):
    """{entry: {"result": str, "weights": {file: MB}, "from_ckpt": bool, "dram": MB | None}}"""
    out, cur = {}, None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"^Test: (\S+)", line)
            if m:
                cur = out.setdefault(m.group(1), {"result": "", "weights": {}, "from_ckpt": False,
                                                  "dram": None})
                continue
            if cur is None:
                continue
            m = re.match(r"^\s+Result\s*:\s*(\S+)", line)
            if m:
                cur["result"] = m.group(1).upper()
                continue
            m = re.match(r"^\s+weight\s+(\S+)\s+([\d.]+) MB", line)
            if m:
                # a hash-tagged variant of one file (params_controllers_<hash>.bin) counts once
                name = re.sub(r"_[0-9a-f]{12}(?=\.bin$)", "", m.group(1))
                cur["weights"][name] = float(m.group(2))
                continue
            if re.match(r"^\s+weight\s+none on disk", line):
                cur["from_ckpt"] = True
                continue
            m = re.match(r"^\s+dram\s+weights\s+([\d.]+) MB", line)
            if m:
                cur["dram"] = float(m.group(1))
    return out


def parse_conv(path):
    """{model: PASS | FAIL | SKIP} from the conv_model_tests() lines of a run_ci.sh log."""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"conv_model-([A-Za-z0-9_.]+): (PASS|FAIL|SKIP)\b", line)
            if m and out.get(m.group(1)) != "FAIL":     # a FAIL is never overwritten
                out[m.group(1)] = m.group(2)
    return out


def per_board(values):
    """'12.3 MB', or 'a / b MB' (kintex7 / Alveo U50) when the boards differ; None if empty."""
    vals = [values[b] for b, _ in BOARDS if values.get(b) is not None]
    if not vals:
        return None
    if len(set(f"{v:.1f}" for v in vals)) == 1:
        return f"{vals[0]:.1f} MB"
    return " / ".join(f"{values[b]:.1f}" if values.get(b) is not None else "-" for b, _ in BOARDS) + " MB"


def main():
    ws = sys.argv[1] if len(sys.argv) > 1 else "."
    results = {}     # entry -> [(board, result)]
    weights = {}     # entry -> {board: {file: MB}}
    from_ckpt = set()
    dram = {}        # entry or conv model -> {board: MB}
    conv = {}        # model -> [(board, result)]
    boards_seen = []
    for board, sub in BOARDS:
        seen = False
        for rep in REPORTS:
            p = os.path.join(ws, sub, rep)
            if not os.path.isfile(p):
                continue
            seen = True
            for entry, d in parse_report(p).items():
                results.setdefault(entry, []).append((board, d["result"]))
                if d["weights"]:
                    weights.setdefault(entry, {}).setdefault(board, {}).update(d["weights"])
                if d["from_ckpt"]:
                    from_ckpt.add(entry)
                if d["dram"] is not None:
                    cur = dram.setdefault(entry, {})
                    cur[board] = max(cur.get(board, 0.0), d["dram"])
        p = os.path.join(ws, LOGS[board])
        if os.path.isfile(p):
            got = parse_conv(p)
            seen = seen or bool(got)
            for model, r in got.items():
                conv.setdefault(model, []).append((board, r))
        if seen:
            boards_seen.append(board)

    known = {e for _, _, _, entries, _ in MODELS for e in entries}
    rows = list(MODELS)
    for entry in sorted(set(results) - known):      # a registry entry this table does not know yet
        rows.append((entry, "TBD", "TBD", [entry], False))

    def verdict(name, entries, is_conv):
        rs = [r for e in entries for _, r in results.get(e, [])]
        if is_conv:
            rs += [r for _, r in conv.get(name, [])]
        tested = [r for r in rs if r and r != "SKIP"]
        if any(r != "PASS" for r in tested):
            return "❌ FAIL"
        return "✅ PASS" if tested else "-"

    def weight_text(name, entries):
        # per board: the model's weight files, each counted once, summed
        totals = {}
        for board, _ in BOARDS:
            files = {}
            for e in entries:
                files.update(weights.get(e, {}).get(board, {}))
            if files:
                totals[board] = sum(files.values())
        text = per_board(totals)
        if text:
            return text
        if any(e in from_ckpt for e in entries):
            return "from checkpoint"
        p = os.path.join(ws, CONV_BINS.get(name, ""))
        if name in CONV_BINS and os.path.isfile(p):
            return f"{os.path.getsize(p) / 1e6:.1f} MB"
        return "not reported"

    def dram_text(name, entries):
        vals = {}
        for key in entries:
            for board, mb in dram.get(key, {}).items():
                vals[board] = max(vals.get(board, 0.0), mb)     # the largest of the model's entries
        return per_board(vals) or "not reported"

    n_fail = n_pass = 0
    lines = []
    for i, (name, typ, cores, entries, is_conv) in enumerate(rows, 1):
        v = verdict(name, entries, is_conv)
        n_fail += v.endswith("FAIL")
        n_pass += v.endswith("PASS")
        lines.append(f"| {i} | {name} | {typ} | {weight_text(name, entries)} | "
                     f"{dram_text(name, entries)} | {cores} | {v} |")

    print("## Model summary")
    print()
    print(f"**{n_pass} passed, {n_fail} failed, {len(rows) - n_pass - n_fail} not tested** "
          f"of {len(rows)} models; boards in this run: {', '.join(boards_seen) or 'none'}.")
    print()
    print("| # | Model | Type | Weight bin | DRAM usage | Multi-core | Status |")
    print("|---|---|---|---|---|---|---|")
    print("\n".join(lines))
    print()
    print("- **Status:** one verdict per model over every board, registry entry and pass that tested it. "
          "FAIL if any failed; `-` if nothing tested it in this run (skipped entries do not count).")
    print("- **Weight bin:** the weight files on disk. **DRAM usage:** the weights loaded in DRAM, "
          "reported by the model script (weights only; with several engines every copy counts). "
          "`a / b`: kintex7 / Alveo U50 when the two boards differ.")
    print("- **Multi-core:** `2 / 8` runs on both engines of the kintex7 and all 8 of the Alveo U50; "
          "`8 only` needs 8 engines; `single only` uses one engine; `no callsites`: no test runs it. "
          "Type and multi-core are a fixed table in `.github/scripts/model_table.py`.")
    print()


if __name__ == "__main__":
    main()
