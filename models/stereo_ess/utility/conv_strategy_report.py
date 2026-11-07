"""Graphs + markdown for conv_strategy_bench.py results (host only, no board).

    python models/stereo_ess/utility/conv_strategy_bench.py --report

Per conv type two PNGs in scratchpad/conv_strategy/img/: <type>_perf.png (speedup of each lowering vs the native
CONV2D walker, one bar group per measured shape) and <type>_snr.png (SNR of each lowering vs the fp32 reference,
same rows), and scratchpad/conv_strategy/CONV_STRATEGY.md linking them with the per-shape numbers.
"""
import json
import os
import re
import statistics

from conv_strategy_bench import PATHS, BSpec, type_rank, users_text, wino_desc

INK, INK2, MUTED, SURFACE, GRID_LINE = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"
COLOR = {"native16": "#eb6834", "direct": "#1baf7a", "winograd": "#2a78d6"}
NAME = {"native16": "native CONV2D (bf16 walker)", "direct": "im2col + matmul", "winograd": "Winograd"}
SHORTNAME = {"native16": "native", "direct": "im2col", "winograd": "Winograd"}
MARK = {"native16": "o", "direct": "s", "winograd": "D"}


def _ok(r):
    return r is not None and r.get("status") in ("ok", "fail") and r.get("ms")


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 12,
                         "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    return plt


def slug(label):
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


MIXED = ("ConvTranspose", "Conv2d patchify", "Conv2d large-kernel stem", "Conv1d strided")


def shape_text(sp):
    mixed = sp.label.startswith(MIXED)                    # families with several kernels: name the kernel per row
    if sp.H == 1 and sp.kh == 1:
        s = f"T={sp.W}  {sp.Cin}→{sp.Cout}"
        if sp.dw > 1:
            s += f"  dil {sp.dw}"
        if mixed:
            s += f"  (k{sp.kw}" + (f" s{sp.sw})" if sp.sw > 1 else ")")
        return s
    s = f"{sp.H}x{sp.W}  {sp.Cin}→{sp.Cout}"
    if sp.dh > 1 and "dilation" not in sp.label:
        s += f"  dil {sp.dh}"
    if mixed:
        s += f"  ({sp.kh}x{sp.kw}" + (f" s{sp.sh})" if sp.sh > 1 else ")")
    return s


def header(fig, title, sub, handles):
    """Title, subtitle, then the legend, top-left; returns the header height in inches."""
    fh = fig.get_figheight()
    fig.text(0.01, 1 - 0.18 / fh, title, fontsize=13, fontweight="bold", va="top", ha="left", color=INK)
    y = 0.50
    if sub:
        fig.text(0.01, 1 - y / fh, sub, fontsize=9.5, va="top", ha="left", color=INK2)
        y += 0.26
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.004, 1 - (y - 0.04) / fh), ncol=len(handles),
               frameon=False, fontsize=9)
    return y + 0.34


def short_reason(r, n=64):
    t = (r.get("reason") or r.get("status") or "").replace("\n", " ")
    t = re.sub(r"^(setup|compile): ", "", t)
    return t if len(t) <= n else t[:n - 1] + "…"


def baseline(res):
    return "native16" if _ok(res.get("native16")) else ("direct" if _ok(res.get("direct")) else None)


# ----------------------------------------------------------------------------------------------- graphs
def plot_perf(label, rows, png, meta):
    plt = _mpl()
    from matplotlib.patches import Patch
    paths = [p for p in PATHS if any(_ok(res.get(p)) for _, _, res in rows)]
    if not paths:
        return False
    ratios = [res[baseline(res)]["ms"] / res[p]["ms"] for _, _, res in rows if baseline(res)
              for p in paths if _ok(res.get(p))]
    xmax = max([1.0] + ratios)
    bar_h = 0.82 / len(paths)
    n = len(rows)
    fig, ax = plt.subplots(figsize=(11, 2.2 + n * (0.15 * len(paths) + 0.2)))
    labels = []
    for yi, (sp, ent, res) in enumerate(rows):
        bp = baseline(res)
        for pi, p in enumerate(paths):
            y = yi - 0.41 + bar_h * (pi + 0.5)
            r = res.get(p)
            if _ok(r) and bp:
                v = res[bp]["ms"] / r["ms"]
                ax.barh(y, v, height=bar_h * 0.8, color=COLOR[p], lw=0)
                if p != bp:
                    ax.text(v + xmax * 0.01, y, f"{v:.2f}x" + ("  (SNR fail)" if r["status"] == "fail" else ""),
                            va="center", fontsize=7.5, color=INK2)
            elif r is not None and r.get("status") in ("n/a", "skip"):
                if not (p == "winograd" and sp.wino is None and all(s_.wino is None for s_, _, _ in rows)):
                    ax.text(xmax * 0.01, y, f"{SHORTNAME[p]} {r['status']}: {short_reason(r)}", va="center",
                            fontsize=7, color=MUTED, style="italic")
        lab = f"{shape_text(sp)}\n{users_text(ent['users'], 2)}"
        if bp == "direct":
            lab += "  · vs im2col"
        labels.append(lab)
    ax.axvline(1.0, color=INK2, lw=1)
    ax.set_yticks(range(n), labels, fontsize=8)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(0, xmax * 1.16)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID_LINE, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("speedup vs native CONV2D (bf16 walker = 1.0; higher is faster)")
    wins = [s_ for s_, _, _ in rows if s_.wino]
    wnote = (f"Winograd route: {wino_desc(wins[0].wino)}" if wins and len({wino_desc(s_.wino) for s_ in wins}) == 1
             else ("Winograd route: " + ", ".join(sorted({wino_desc(s_.wino) for s_ in wins})) if wins
                   else f"Winograd n/a: {rows[0][0].wino_why}"))
    fh = fig.get_figheight()
    top = header(fig, f"{label}: speed per lowering", wnote, [Patch(color=COLOR[p], label=NAME[p]) for p in paths])
    fig.text(0.01, 0.004, f"{meta.get('engines', 8)} engines, bf16 activations + weights, median of 3 runs. Rows: "
             f"input map · Cin→Cout, then the models using the shape × executions per inference", fontsize=7.5,
             color=INK2)
    fig.tight_layout(rect=(0, 0.35 / fh, 1, 1 - top / fh))
    fig.savefig(png, dpi=130 if n < 60 else 100)
    plt.close(fig)
    return True


def plot_snr(label, rows, png, meta):
    plt = _mpl()
    from matplotlib.lines import Line2D
    paths = [p for p in PATHS if any(_ok(res.get(p)) for _, _, res in rows)]
    if not paths:
        return False
    n = len(rows)
    fig, ax = plt.subplots(figsize=(9, 2.0 + n * 0.3))
    vals = []
    for yi, (sp, ent, res) in enumerate(rows):
        xs = [res[p]["snr"] for p in paths if _ok(res.get(p))]
        if len(xs) > 1:
            ax.hlines(yi, min(xs), max(xs), color=GRID_LINE, lw=2.5, zorder=1)
        for p in paths:
            if _ok(res.get(p)):
                ax.scatter(res[p]["snr"], yi, marker=MARK[p], s=48, color=COLOR[p], edgecolor=SURFACE, lw=1.2,
                           zorder=3)
                vals.append(res[p]["snr"])
    lo, hi = (min(vals), max(vals)) if vals else (30, 60)
    ax.set_xlim(min(lo, 30) - 2, hi + 3)
    ax.axvline(30, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(30.3, -0.45, "30 dB pass line", fontsize=7.5, color=INK2, va="bottom")
    ax.set_yticks(range(n), [shape_text(sp) for sp, _, _ in rows], fontsize=8)
    ax.set_ylim(n - 0.5, -0.8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID_LINE, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("SNR vs fp32 reference on the same bf16 data (dB; higher is more accurate)")
    fh = fig.get_figheight()
    top = header(fig, f"{label}: accuracy per lowering", None,
                 [Line2D([], [], ls="", marker=MARK[p], markersize=7, color=COLOR[p], label=NAME[p]) for p in paths])
    fig.tight_layout(rect=(0, 0, 1, 1 - top / fh))
    fig.savefig(png, dpi=130 if n < 60 else 100)
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------------------------- markdown
def _fmt_ms(r):
    if _ok(r):
        return f"{r['ms']:.3f}" + (" (SNR fail)" if r["status"] == "fail" else "")
    if r is None:
        return "–"
    return r["status"]


def _count(ent):
    return sum(int(u.get("count") or 1) for u in ent["users"])


def report(shapes, skipped, res_path, out_dir):
    if not os.path.exists(res_path):
        raise SystemExit(f"no results at {res_path}: run the board benchmark first")
    with open(res_path) as f:
        saved = json.load(f)
    results, meta = saved["results"], saved.get("meta", {})
    img_dir = os.path.join(out_dir, "img")
    os.makedirs(img_dir, exist_ok=True)
    by_type = {}
    for sid, ent in shapes.items():
        if sid in results:
            sp = BSpec(sid, ent["shape"], ent["label"])
            by_type.setdefault(ent["label"], []).append((sp, ent, results[sid]))
    types = sorted(by_type, key=type_rank)
    summary, sections = [], []
    for label in types:
        rows = sorted(by_type[label], key=lambda t: t[0].nominal)
        sl = slug(label)
        perf_png, snr_png = os.path.join(img_dir, f"{sl}_perf.png"), os.path.join(img_dir, f"{sl}_snr.png")
        have = plot_perf(label, rows, perf_png, meta)
        plot_snr(label, rows, snr_png, meta)
        fastest = {p: 0 for p in PATHS}
        wino_vs, snrs = [], {p: [] for p in PATHS}
        for sp, ent, res in rows:
            ok = {p: res[p]["ms"] for p in PATHS if _ok(res.get(p)) and res[p]["status"] == "ok"}
            if ok:
                fastest[min(ok, key=ok.get)] += 1
            other = [ok[p] for p in ("native16", "direct") if p in ok]
            if "winograd" in ok and other:
                wino_vs.append(min(other) / ok["winograd"])
            for p in PATHS:
                if _ok(res.get(p)):
                    snrs[p].append(res[p]["snr"])
        med = lambda v: f"{statistics.median(v):.1f}" if v else "–"
        wtxt = (f"{statistics.median(wino_vs):.2f}x ({min(wino_vs):.2f}–{max(wino_vs):.2f})" if wino_vs else "n/a")
        execs = sum(_count(ent) for _, ent, _ in rows)
        summary.append(f"| [{label}](#{sl}) | {len(rows)} | {execs} | {fastest['native16']} / {fastest['direct']} / "
                       f"{fastest['winograd']} | {wtxt} | {med(snrs['native16'])} / {med(snrs['direct'])} / "
                       f"{med(snrs['winograd'])} |")
        wn = [sp for sp, _, _ in rows if sp.wino]
        route = (f"Winograd route: {', '.join(sorted({wino_desc(sp.wino) for sp in wn}))}." if wn
                 else f"Winograd does not apply ({rows[0][0].wino_why}).")
        verdict = (f"Fastest lowering per shape: native {fastest['native16']}, im2col {fastest['direct']}, "
                   f"Winograd {fastest['winograd']} (of {sum(fastest.values())} shapes that ran cleanly).")
        if wino_vs:
            verdict += (f" Winograd vs the faster of the other two: median {statistics.median(wino_vs):.2f}x, "
                        f"faster on {sum(v > 1.0 for v in wino_vs)} of {len(wino_vs)}.")
        lines = [f'<a id="{sl}"></a>', f"## {label}", "", f"{len(rows)} shapes, {execs} conv executions per inference "
                 f"across the models below. {route}", "", verdict, ""]
        if have:
            lines += [f"![{label}: speed per lowering](img/{sl}_perf.png)", "",
                      f"![{label}: accuracy per lowering](img/{sl}_snr.png)", ""]
        lines += ["<details><summary>Per-shape numbers</summary>", "",
                  "| Input · Cin→Cout | Used by (× per inference) | native ms | im2col ms | Winograd ms | fastest | "
                  "SNR dB native / im2col / Winograd |", "|---|---|---:|---:|---:|---|---|"]
        reasons = {}
        for sp, ent, res in rows:
            ok = {p: res[p]["ms"] for p in PATHS if _ok(res.get(p)) and res[p]["status"] == "ok"}
            best = SHORTNAME[min(ok, key=ok.get)] if ok else "–"
            snr = " / ".join(f"{res[p]['snr']:.1f}" if _ok(res.get(p)) else "–" for p in PATHS)
            lines.append(f"| {shape_text(sp)} | {users_text(ent['users'], 6)} | {_fmt_ms(res.get('native16'))} | "
                         f"{_fmt_ms(res.get('direct'))} | {_fmt_ms(res.get('winograd'))} | {best} | {snr} |")
            for p in PATHS:
                r = res.get(p)
                if r is not None and r.get("status") in ("n/a", "skip"):
                    reasons.setdefault(f"{SHORTNAME[p]} {r['status']}: {short_reason(r, 200)}", []).append(
                        shape_text(sp))
        lines += [""]
        if reasons:
            lines += ["Not run:", ""] + [f"- {k} ({len(v)} shape{'s' if len(v) > 1 else ''})"
                                         for k, v in reasons.items()] + [""]
        lines += ["</details>", ""]
        sections.append("\n".join(lines))

    missing = [sid for sid in shapes if sid not in results]
    why = {}
    for r, reason in skipped:
        why.setdefault(reason, []).append(r)
    nb = ["## Not benchmarked", ""]
    for reason, rs in sorted(why.items(), key=lambda t: -len(t[1])):
        nb.append(f"**{reason}** ({len(rs)} layers, {sum(int(r.get('count') or 1) for r in rs)} executions per "
                  f"inference): " + users_text(rs, 12))
        nb.append("")
    if missing:
        nb.append(f"**Not yet measured:** {len(missing)} of {len(shapes)} unique shapes have no board result "
                  f"(rerun the benchmark; it resumes).")
        nb.append("")

    md = [
        "# Conv lowering per layer: native CONV2D vs im2col vs Winograd",
        "",
        f"Board run {meta.get('date', '?')}, {meta.get('engines', 8)} engines, {meta.get('peak', 0):.0f} GFLOPS "
        f"bf16 matmul peak. Every conv the repo's models execute, collapsed to {len(shapes)} unique shapes, each "
        f"run three ways on the same bf16 input and weights. Generated by "
        f"`models/stereo_ess/utility/conv_strategy_bench.py --report` from `results_e{meta.get('engines', 8)}.json`.",
        "",
        "## How to read it",
        "",
        "- **native**: the CONV2D opcode with FLOAT16 weights held in URAM_B (`weight_reuse`), gather mode for "
        "small Cin. Each engine walks its own band of output rows (columns for 1D signals).",
        "- **im2col**: strided-DMA im2col into DRAM, then one bf16 matmul per row chunk, with K = kh·kw·Cin.",
        "- **Winograd**: F(2x2,3x3) (2D) or F(2,3) (1D), with the transforms as SRAM add/subs and 16 (or 4) matmuls "
        "with K = Cin. Dilated convs run through d×d parity sub-grids and kernels above 3 through 3-tap pieces "
        "summed in the M planes. It does not apply to stride > 1, 1x1, or kernels where the pieces save no "
        "multiplies.",
        "- **Speed graph**: per shape, each lowering's speedup over native (native = 1.0). Where native cannot run "
        "the shape, the row is normalised to im2col and says so.",
        "- **SNR graph**: the same rows, with each lowering's SNR against torch fp32 on the same bf16 data.",
        "- **Shapes**: IF4 and IF8 layers are benchmarked in bf16 too, since the question is the lowering, not the "
        "weight format. Channel counts that are not multiples of 64 are zero-padded to the 64-lane vector (the "
        "board does the same). im2col packs Cin ≤ 32 as 32-channel pixels. GFLOPS use the real shape.",
        "- **Transposed convs** run as their sub-pixel equivalent: a stride-1 conv with ceil(k/s) taps and "
        "Cout·s outputs, then depth-to-space. Video patch embeds fold their 2 temporal taps into Cin.",
        "- **Tiny convs** (dpdfnet's per-frame layers, 1-pixel maps) are bound by launch overhead, so their "
        "timings are an upper bound.",
        "",
        "## Summary per conv type",
        "",
        "| Conv type | shapes | executions / inference | fastest: native / im2col / Winograd | Winograd vs best "
        "other (median, range) | median SNR dB native / im2col / Winograd |",
        "|---|---:|---:|---|---|---|",
        *summary,
        "",
        *sections,
        *nb,
        "## Inventory caveats",
        "",
        "- The inventory comes from three code screeners: `scratchpad/conv_strategy/inventory/{vision,audio,vlm}"
        ".jsonl`, one record per layer with the default config of each model's test script.",
        "- **stereo_ess** is partial: the light_ess network lives on the `stereo_models` branch, which is not in "
        "this checkout. Only the four probe shapes (dec3, enc1, dec1, dec0) are included.",
        "- **flux2_klein_4b** covers the VAE decoder at 256 and 512. The encoder never runs in the text-to-image "
        "pipeline, so it is listed under not benchmarked.",
        "- **Optional-flag paths are included**: qwen2.5-omni `--audio` / `--image` / `--speak`, gemma4 "
        "`--image` / `--audio`, qwen2.5-vl `--image`, qwen3.5 `--vision-enable`.",
        "- **Inferred or downloaded inputs**: kokoro's frame counts are inferred, not measured. dpdfnet shapes "
        "come from its pinned ONNX graphs. Model inputs that are not in the repo (qwen-omni speaker mel, gemma4 "
        "audio length) come from config defaults.",
        "- **Padding**: asymmetric (causal / TF-same) padding is benchmarked as the symmetric split. The work is "
        "the same and the output is within one pixel.",
        "",
    ]
    md_path = os.path.join(out_dir, "CONV_STRATEGY.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"report: {md_path}  ({len(types)} conv types, {2 * len(types)} graphs in {img_dir})")


# ----------------------------------------------------------------------------------------------- by kernel
def kernel_category(sp, ent):
    """(category, order) for the by-kernel view, or None (dilated / >512 channels are left out)."""
    if sp.dh > 1 or sp.dw > 1 or max(sp.Cin, sp.Cout) > 512:
        return None
    op = ent["users"][0]["op"]
    one_d = sp.H == 1 and sp.kh == 1
    if op.startswith("conv_transpose"):
        return ("ConvTranspose (as its sub-pixel conv)", 9)
    if sp.label.startswith("Conv2d patchify"):
        return ("Conv2d patchify (kernel = stride)", 4)
    if one_d:
        return ("Conv1d, stride 1", 5) if sp.sw == 1 else ("Conv1d, strided", 6)
    return ("Conv2d, stride 1", 1) if sp.sh == sp.sw == 1 else ("Conv2d, strided", 2)


def kernel_name(sp):
    if sp.H == 1 and sp.kh == 1:
        return f"k={sp.kw}" + (f" s{sp.sw}" if sp.sw > 1 else "")
    return f"{sp.kh}x{sp.kw}" + (f" s{sp.sh}" if sp.sh > 1 else "")


def plot_kernel(title, groups, png, meta):
    """groups: [(x label, [res...])]. Vertical grouped bars: median speedup vs native, min-max whisker."""
    plt = _mpl()
    from matplotlib.patches import Patch
    paths = [p for p in PATHS if any(_ok(r.get(p)) and r[p]["status"] == "ok" for _, rs in groups for r in rs)]
    nx = len(groups)
    fig, ax = plt.subplots(figsize=(max(7.5, 1.0 + nx * (0.32 * len(paths) + 0.25)), 5.4))
    w = 0.8 / len(paths)
    ymax = 1.0
    ymax_hint = max([1.0] + [r[baseline(r)]["ms"] / r[p]["ms"] for _, rs in groups for r in rs if baseline(r)
                             for p in paths if _ok(r.get(p)) and r[p]["status"] == "ok"])
    na_native = []
    for xi, (lab, rs) in enumerate(groups):
        base = [r for r in rs if baseline(r)]
        for pi, p in enumerate(paths):
            v = sorted(r[baseline(r)]["ms"] / r[p]["ms"] for r in base
                       if _ok(r.get(p)) and r[p]["status"] == "ok")
            x = xi - 0.4 + w * (pi + 0.5)
            if not v:
                continue
            med = v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])
            ax.bar(x, med, width=w * 0.86, color=COLOR[p], lw=0, zorder=2)
            if len(v) > 1:
                ax.vlines(x, v[0], v[-1], color=INK2, lw=0.9, zorder=3)
            if p != "native16" or any(baseline(r) == "direct" for r in base):
                ax.text(x, max(med, v[-1]) + ymax_hint * 0.012, f"{med:.2f}", ha="center", va="bottom",
                        fontsize=6.3, color=INK2)
            ymax = max(ymax, v[-1], med)
        if any(baseline(r) == "direct" for r in rs):
            na_native.append(lab)
    ax.axhline(1.0, color=INK2, lw=1, zorder=1)
    ax.set_xticks(range(nx), [f"{lab}" + (f"\n({len(rs)} maps)" if len(rs) > 1 else "") for lab, rs in groups],
                  fontsize=7.5, rotation=0 if nx <= 10 else 45, ha="center" if nx <= 10 else "right")
    ax.set_xlim(-0.6, nx - 0.4)
    ax.set_ylim(0, ymax * 1.12)
    ax.set_ylabel("speedup vs native CONV2D (= 1.0; higher is faster)")
    ax.set_xlabel("input channels → output channels")
    ax.grid(axis="y", color=GRID_LINE, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    sub = "bar = median over the shape's map sizes, line = min–max; channels ≤ 512, dilation 1"
    if na_native:
        sub += f"; native n/a for {', '.join(na_native)} (normalised to im2col)"
    fh = fig.get_figheight()
    top = header(fig, title, sub, [Patch(color=COLOR[p], label=NAME[p]) for p in paths])
    fig.tight_layout(rect=(0, 0, 1, 1 - top / fh))
    fig.savefig(png, dpi=140)
    plt.close(fig)


def report_by_kernel(shapes, res_path, out_dir):
    with open(res_path) as f:
        saved = json.load(f)
    results, meta = saved["results"], saved.get("meta", {})
    img_dir = os.path.join(out_dir, "img_by_kernel")
    os.makedirs(img_dir, exist_ok=True)
    cats = {}
    for sid, ent in shapes.items():
        if sid not in results:
            continue
        sp = BSpec(sid, ent["shape"], ent["label"])
        c = kernel_category(sp, ent)
        if c is None:
            continue
        k = (sp.kh * sp.kw, sp.kh, sp.kw, sp.sh, sp.sw)
        cats.setdefault(c, {}).setdefault(k, []).append((sp, ent, results[sid]))
    md = ["# Conv lowering by kernel size: native CONV2D vs im2col vs Winograd", "",
          f"Board run {meta.get('date', '?')}, {meta.get('engines', 8)} engines, bf16. Every dilation-1 conv in the "
          "repo's models with at most 512 channels, grouped by conv category, then kernel size (small to large), then "
          "channel count. Bars are speedup over the native CONV2D walker (native = 1.0). When several map sizes share "
          "a channel pair, the bar is their median and the thin line their range. Full per-shape data: "
          "[CONV_STRATEGY.md](CONV_STRATEGY.md).", ""]
    for (cat, _), kernels in sorted(cats.items(), key=lambda t: t[0][1]):
        md += [f"## {cat}", ""]
        for k in sorted(kernels):
            rows = kernels[k]
            sp0 = rows[0][0]
            kn = kernel_name(sp0)
            pairs = {}
            for sp, ent, res in rows:
                pairs.setdefault((sp.Cin, sp.Cout), []).append((sp, ent, res))
            groups = [(f"{a}→{b}", [r for _, _, r in v]) for (a, b), v in sorted(pairs.items())]
            title = f"{cat}: {kn}"
            png = os.path.join(img_dir, f"{slug(cat)}__{slug(kn)}.png")
            plot_kernel(title, groups, png, meta)
            wino = [s.wino for s, _, _ in rows if s.wino]
            note = (f"Winograd: {wino_desc(wino[0])}." if wino else f"Winograd n/a ({sp0.wino_why}).")
            wins = {p: 0 for p in PATHS}
            for _, _, res in rows:
                ok = {p: res[p]["ms"] for p in PATHS if _ok(res.get(p)) and res[p]["status"] == "ok"}
                if ok:
                    wins[min(ok, key=ok.get)] += 1
            md += [f"### {kn}", "", f"{len(rows)} shapes, {len(pairs)} channel pairs. {note} Fastest: native "
                   f"{wins['native16']}, im2col {wins['direct']}, Winograd {wins['winograd']}.", "",
                   f"![{title}](img_by_kernel/{os.path.basename(png)})", ""]
    table = full_map(shapes, results)
    md += table
    with open(os.path.join(out_dir, "CONV_MAP.md"), "w") as f:
        f.write("\n".join(["# Conv lowering map: every conv shape, winner in bold", "",
                           f"Board run {meta.get('date', '?')}, {meta.get('engines', 8)} engines, bf16."] + table[1:]))
    path = os.path.join(out_dir, "CONV_BY_KERNEL.md")
    with open(path, "w") as f:
        f.write("\n".join(md))
    print(f"report: {path}")


def full_map(shapes, results):
    """One row per benchmarked conv shape (all of them, no channel / dilation filter): ms per lowering, winner bold."""
    rows = []
    for sid, ent in shapes.items():
        if sid not in results:
            continue
        sp = BSpec(sid, ent["shape"], ent["label"])
        op = ent["users"][0]["op"]
        one_d = sp.H == 1 and sp.kh == 1
        if op.startswith("conv_transpose"):
            cat, co = "ConvTranspose (sub-pixel)", 9
        elif sp.label.startswith("Conv2d patchify"):
            cat, co = "Conv2d patchify", 4
        elif one_d:
            cat, co = ("Conv1d", 5) if sp.sw == 1 else ("Conv1d strided", 6)
        else:
            cat, co = ("Conv2d", 1) if sp.sh == sp.sw == 1 else ("Conv2d strided", 2)
        rows.append((co, sp.kh * sp.kw, sp.kh, sp.kw, sp.sh, sp.dh if not one_d else sp.dw, sp.Cin, sp.Cout,
                     sp.H * sp.W, cat, sp, ent, results[sid]))
    rows.sort(key=lambda t: t[:9])
    out = ["## Full map: every conv shape", "",
           f"All {len(rows)} benchmarked shapes (every dilation and channel count), sorted by category, kernel, "
           "stride, dilation, then channels and map size. Times are ms per conv on 8 engines. The fastest lowering "
           "that passed the SNR check is **bold**. *n/a* = the lowering does not apply to the shape; "
           "*fail* = it ran but missed the 30 dB SNR check. Speedup = best of native / im2col over Winograd.", "",
           "| Category | Kernel | Stride | Dil | Input | Cin→Cout | Used by (× per inference) | native ms | "
           "im2col ms | Winograd ms | Winner | Wino speedup | SNR dB n / i / w |",
           "|---|---|---:|---:|---|---|---|---:|---:|---:|---|---:|---|"]
    for co, _, kh, kw, sh, d, ci, cout, _, cat, sp, ent, res in rows:
        ok = {p: res[p]["ms"] for p in PATHS if _ok(res.get(p)) and res[p]["status"] == "ok"}
        best = min(ok, key=ok.get) if ok else None
        cells = []
        for p in PATHS:
            r = res.get(p)
            if r is None:
                cells.append("–")
            elif r["status"] == "n/a":
                cells.append("*n/a*")
            elif r["status"] == "skip":
                cells.append("*skip*")
            elif r["status"] == "fail":
                cells.append(f"*fail* ({r['ms']:.3f})")
            else:
                cells.append(f"**{r['ms']:.3f}**" if p == best else f"{r['ms']:.3f}")
        other = [ok[p] for p in ("native16", "direct") if p in ok]
        ws = f"{min(other) / ok['winograd']:.2f}x" if "winograd" in ok and other else "–"
        one_d = sp.H == 1 and sp.kh == 1
        kern = f"1x{kw}" if one_d else f"{kh}x{kw}"
        inp = f"T={sp.W}" if one_d else f"{sp.H}x{sp.W}"
        stride = sp.sw if one_d else (sp.sh if sp.sh == sp.sw else f"{sp.sh}x{sp.sw}")
        snr = " / ".join(f"{res[p]['snr']:.1f}" if _ok(res.get(p)) else "–" for p in PATHS)
        out.append(f"| {cat} | {kern} | {stride} | {d} | {inp} | {ci}→{cout} | {users_text(ent['users'], 4)} | "
                   + " | ".join(cells) + f" | {SHORTNAME[best] if best else '–'} | {ws} | {snr} |")
    return out + [""]
