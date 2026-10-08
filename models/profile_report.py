"""Shared Markdown reporting for model ``--profile`` runs.

The model entry points own checkpoint emission and execution.  This module only
normalizes the resulting measurements so every model reports the same columns
and uses the same definitions:

* *issued FLOPs* are the operations emitted by the accelerator kernels;
* *effective FLOPs* are useful model operations at the requested sequence or
  context length (padding, duplicated GQA work, and similar implementation
  overhead are excluded);
* throughput always uses the FPGA hardware-counter time, while ``CPU ms`` is
  the host wall clock surrounding the same execution.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping


def _number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _gflop(value: Any) -> str:
    return _number(None if value is None else float(value) / 1e9, 3)


def _rate(flops: Any, hw_ms: Any) -> float | None:
    if flops is None or hw_ms is None or float(hw_ms) <= 0:
        return None
    return float(flops) / (float(hw_ms) * 1e6)


def _percent(value: Any, denominator: Any) -> float | None:
    if value is None or denominator is None or float(denominator) <= 0:
        return None
    return 100.0 * float(value) / float(denominator)


def _cell(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def measurement(*, label: str, hw_ms: float, cpu_ms: float | None = None,
                tokens: int | None = None, position: int | None = None,
                issued_flops: int | float | None = None,
                effective_flops: int | float | None = None,
                peak_gflops: float | None = None) -> dict[str, Any]:
    """Create one normalized overall or breakdown measurement row."""
    issued_rate = _rate(issued_flops, hw_ms)
    effective_rate = _rate(effective_flops, hw_ms)
    token_rate = (1000.0 * float(tokens) / float(hw_ms)
                  if tokens is not None and hw_ms > 0 else None)
    return {
        "label": label,
        "tokens": tokens,
        "position": position,
        "hw_ms": hw_ms,
        "cpu_ms": cpu_ms,
        "tok_s": token_rate,
        "issued_flops": issued_flops,
        "effective_flops": effective_flops,
        "issued_gflops": issued_rate,
        "effective_gflops": effective_rate,
        "issued_peak_pct": _percent(issued_rate, peak_gflops),
        "effective_peak_pct": _percent(effective_rate, peak_gflops),
    }


def aggregate_checkpoints(samples: Iterable[Mapping[str, Any]], *,
                          peak_gflops: float | None = None) -> list[dict[str, Any]]:
    """Aggregate repeated per-layer checkpoint samples by step name.

    Input samples use ``name``, ``hw_ms`` and optional issued/effective ``flops``.
    Ordering follows the first occurrence, which matches program order.
    """
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for sample in samples:
        name = str(sample["name"])
        row = grouped.setdefault(name, {
            "label": name, "hw_ms": 0.0, "samples": 0,
            "issued_flops": 0.0, "effective_flops": 0.0,
            "issued_known": True, "effective_known": True,
        })
        row["hw_ms"] += float(sample["hw_ms"])
        row["samples"] += 1
        for key, known in (("issued_flops", "issued_known"),
                           ("effective_flops", "effective_known")):
            value = sample.get(key)
            if value is None:
                row[known] = False
            else:
                row[key] += float(value)
    result = []
    for row in grouped.values():
        issued = row.pop("issued_flops") if row.pop("issued_known") else None
        effective = row.pop("effective_flops") if row.pop("effective_known") else None
        normalized = measurement(
            label=row["label"], hw_ms=row["hw_ms"], issued_flops=issued,
            effective_flops=effective, peak_gflops=peak_gflops)
        normalized["samples"] = row["samples"]
        result.append(normalized)
    return result


def write_profile_markdown(path: str | Path, *, title: str,
                           hardware: Mapping[str, Any],
                           overall: Iterable[Mapping[str, Any]],
                           breakdowns: Iterable[tuple[str, Iterable[Mapping[str, Any]]]],
                           notes: Iterable[str] = ()) -> str:
    """Write the common model-profile Markdown schema and return ``path``."""
    lines = [f"# {title}", "", "## Hardware and run configuration", "",
             "| Setting | Value |", "|---|---|"]
    for key, value in hardware.items():
        lines.append(f"| {_cell(key)} | {_cell(value)} |")

    lines += ["", "## Overall performance", "",
              "| Phase | Seq length / position | HW ms | CPU ms | tok/s | "
              "Issued GFLOP | Effective GFLOP | Issued GFLOPS | Effective GFLOPS | "
              "Issued % peak | Effective % peak |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in overall:
        tokens = row.get("tokens")
        position = row.get("position")
        if tokens is not None and position is not None:
            where = f"{tokens} token(s) @ position {position}"
        elif tokens is not None:
            where = tokens
        else:
            where = position
        lines.append(
            f"| {row['label']} | {where if where is not None else 'n/a'} | "
            f"{_number(row.get('hw_ms'))} | {_number(row.get('cpu_ms'))} | "
            f"{_number(row.get('tok_s'))} | {_gflop(row.get('issued_flops'))} | "
            f"{_gflop(row.get('effective_flops'))} | "
            f"{_number(row.get('issued_gflops'))} | {_number(row.get('effective_gflops'))} | "
            f"{_number(row.get('issued_peak_pct'))} | "
            f"{_number(row.get('effective_peak_pct'))} |")

    for heading, rows_iter in breakdowns:
        rows = list(rows_iter)
        total_ms = sum(float(row.get("hw_ms") or 0.0) for row in rows)
        lines += ["", f"## {heading}", "",
                  "| Step | Samples | HW ms | Share | Issued GFLOP | Effective GFLOP | "
                  "Issued GFLOPS | Effective GFLOPS | Issued % peak | Effective % peak |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            share = (100.0 * float(row.get("hw_ms") or 0.0) / total_ms
                     if total_ms else None)
            lines.append(
                f"| {row['label']} | {row.get('samples', 1)} | "
                f"{_number(row.get('hw_ms'))} | {_number(share)} | "
                f"{_gflop(row.get('issued_flops'))} | {_gflop(row.get('effective_flops'))} | "
                f"{_number(row.get('issued_gflops'))} | "
                f"{_number(row.get('effective_gflops'))} | "
                f"{_number(row.get('issued_peak_pct'))} | "
                f"{_number(row.get('effective_peak_pct'))} |")

    if notes:
        lines += ["", "## Notes", ""]
        lines.extend(f"- {note}" for note in notes)
    lines.append("")
    out = Path(path)
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)
