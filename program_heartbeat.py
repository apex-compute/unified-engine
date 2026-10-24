"""Per-program heartbeat: sample the engines of ONE program run until they halt.

A program run is silent between launch and HALT: wait_queue only polls the busy
bit and reports registers once, on timeout. ProgramHeartbeat samples every
engine of the run at a fixed interval, from right before launch until every
engine has halted, and appends the samples to a log as one block per run, so a
hang shows which engine stopped retiring instructions and at what count.

Per sample and engine it reads two AXI-Lite registers:

  UE_QUEUE_CTRL_ADDR      (0x54): bit 8 queue busy, bit 28 semaphore flag
  UE_INSTRUCTION_CTL_ADDR (0x70): instructions retired since the queue start
                                  (counted at issue, reset on each queue start)

Register reads only, on its own descriptor, so it does not perturb the run.
A board that has dropped off PCIe reads 0xffffffff and is logged as LOST.

Usage, around a program and its workers:

    with program_heartbeat(ue, "prefill chunk 1/2", [ue] + workers):
        start workers; ue.program_execute(...); wait workers

It is a no-op unless the caller enabled it with ``ue.heartbeat_log = path``
(optionally ``ue.heartbeat_interval = seconds``, default 1).

With ``ue.heartbeat_precheck = True`` every program is preceded by a board
check: andromeda's ``pcie_utils/hw_check.py --dev <dev>`` must show the board
alive and on the same FPGA version as the run's first check, and
``pcie_utils/temp_check.tcl`` (JTAG) records every FPGA's die temperature.
The temperature is recorded only; a failed board check raises before launch.
Set ANDROMEDA_DIR if the andromeda checkout is not ``~/andromeda``.
"""

import contextlib
import datetime
import os
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import Optional

from user_dma_core import (
    AXI_LITE_TRANSLATION_OFFSET,
    REG32_FORMAT,
    UE_INSTRUCTION_CTL_ADDR,
    UE_QUEUE_CTRL_ADDR,
)

LOST = 0xFFFFFFFF
ANDROMEDA_DIR = os.environ.get("ANDROMEDA_DIR", os.path.expanduser("~/andromeda"))
# First version each device reported in this process: later prechecks must match.
_run_version: dict[str, int] = {}


def precheck(user_device: str) -> tuple[list[str], Optional[str]]:
    """Board check before a program: (log lines, failure reason or None).

    Runs hw_check.py for the device (alive + same version as the run's first
    check), then temp_check.tcl over JTAG; temperatures are only recorded."""
    dev = os.path.basename(user_device).removesuffix("_user")
    utils = os.path.join(ANDROMEDA_DIR, "pcie_utils")
    lines, failure = [], None
    t0 = time.perf_counter()
    hw = subprocess.run([sys.executable, "hw_check.py", "--dev", dev], cwd=utils,
                        capture_output=True, text=True, timeout=60)
    out = (hw.stdout + hw.stderr).strip()
    lines.append(f"hw_check  {out.splitlines()[-1] if out else '(no output)'} "
                 f"[{time.perf_counter() - t0:.1f} s]")
    m = re.search(r"version=0x([0-9a-f]{8})", out)
    version = int(m.group(1), 16) if m else None
    if hw.returncode != 0 or version is None or version == LOST:
        failure = f"hw_check: {dev} is not responding (rc={hw.returncode})"
    else:
        first = _run_version.setdefault(user_device, version)
        if version != first:
            failure = (f"hw_check: {dev} reports 0x{version:08x}, but this run "
                       f"started on 0x{first:08x}")
    if failure:
        return lines, failure
    t1 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="temp_check_") as scratch:
        temp = subprocess.run(
            ["bash", "-c",
             f"source {os.path.join(ANDROMEDA_DIR, 'setting_tools.bash')} >/dev/null 2>&1; "
             f"vivado -mode batch -nolog -nojournal -source "
             f"{os.path.join(utils, 'temp_check.tcl')}"],
            cwd=scratch, capture_output=True, text=True, timeout=300)
    temps = [l for l in temp.stdout.splitlines() if l.startswith(("TEMP", "WARN"))]
    took = f"[{time.perf_counter() - t1:.1f} s]"
    lines += [f"temp      {l} {took}" for l in temps] or [
        f"temp      no reading (rc={temp.returncode}) {took}"]
    return lines, None


def human(n: int) -> str:
    """Instruction count with an adaptive k / M / B suffix, 3 significant digits."""
    if abs(n) < 1000:
        return str(n)
    for scale, unit in ((1_000, "k"), (1_000_000, "M"), (1_000_000_000, "B")):
        v = float(f"{n / scale:.3g}")            # round to 3 significant digits first
        # Promote when rounding reaches 1000 of this unit (999,999 -> 1.00M).
        if abs(v) < 1000 or unit == "B":
            return f"{v:.{max(0, 3 - len(str(int(abs(v)))))}f}{unit}"
    return str(n)


class ProgramHeartbeat:
    """Context manager that samples ``engine_bases`` while one program runs."""

    def __init__(self, user_device: str, engine_bases: list[int], log_path: str,
                 label: str, interval: float = 1.0, check_board: bool = False):
        self.check_board = check_board
        self.user_device = user_device
        self.engine_bases = list(engine_bases)
        self.log_path = log_path
        self.label = label
        self.interval = interval
        self._prev_pc: list[Optional[int]] = [None] * len(self.engine_bases)

    def _read(self, base: int, offset: int) -> int:
        data = os.pread(self._fd, 4, base + offset - AXI_LITE_TRANSLATION_OFFSET)
        return struct.unpack(REG32_FORMAT, data)[0]

    def _sample(self) -> str:
        cells = []
        for e, base in enumerate(self.engine_bases):
            try:
                queue = self._read(base, UE_QUEUE_CTRL_ADDR)
                pc = self._read(base, UE_INSTRUCTION_CTL_ADDR)
            except (OSError, struct.error) as error:
                cells.append(f"e{e} READ_ERROR({type(error).__name__})")
                continue
            if queue == LOST and pc == LOST:
                cells.append(f"e{e} LOST")
                self._prev_pc[e] = None
                continue
            busy, flag = (queue >> 8) & 1, (queue >> 28) & 1
            prev = self._prev_pc[e]
            self._prev_pc[e] = pc
            note = ""
            if prev is None:
                delta = ""
            elif pc < prev:
                # One program per block, so pc only climbs; a drop means the
                # engine was restarted while this program was supposed to run.
                delta, note = f"-{human(prev - pc)}", " RESTART?"
            else:
                delta = f"+{human(pc - prev)}"
                if busy and pc == prev:
                    note = " STALL"
            cells.append(f"e{e} {'B' if busy else '-'}{'F' if flag else '-'} "
                         f"pc={human(pc):>7} {delta:>8}{note}")
        return " | ".join(cells)

    def _write(self, line: str) -> None:
        self._log.write(line + "\n")
        self._log.flush()

    def _row(self, tag: str) -> None:
        self._write(f"  {tag:>9}  {self._sample()}")

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._row(f"t+{time.perf_counter() - self._t0:.1f}s")

    def __enter__(self):
        self._log = open(self.log_path, "a")
        bases = " ".join(f"e{e}@0x{b:08x}" for e, b in enumerate(self.engine_bases))
        self._write(f"=== {self.label} | {datetime.datetime.now():%Y-%m-%d %H:%M:%S} | "
                    f"{self.user_device} | {bases} | every {self.interval:g} s ===")
        if self.check_board:
            lines, failure = precheck(self.user_device)
            for line in lines:
                self._write(f"  precheck  {line}")
            if failure:
                self._write(f"=== not launched {self.label}: precheck failed: {failure} ===")
                self._write("")
                self._log.close()
                raise RuntimeError(f"precheck before {self.label}: {failure}")
        self._fd = os.open(self.user_device, os.O_RDONLY)
        self._write("  B=busy F=flag pc=instructions retired since this program started, "
                    "+N since the previous sample")
        # Before launch: still the previous program's count and any leftover flag.
        self._row("launch")
        self._t0 = time.perf_counter()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._stop.set()
        self._thread.join(timeout=self.interval + 1.0)
        elapsed = time.perf_counter() - self._t0
        self._row("halted" if exc_type is None else "at-error")
        status = "ok" if exc_type is None else f"{exc_type.__name__}: {exc}"
        # Exact final counts: they are deterministic per program and prompt length.
        totals = " ".join(f"e{e}={pc:,}" for e, pc in enumerate(self._prev_pc)
                          if pc is not None)
        self._write(f"=== end {self.label}: {status} | {elapsed:.1f} s | "
                    f"pc {totals} ===")
        self._write("")
        self._log.close()
        os.close(self._fd)
        return False


def program_heartbeat(ue, label: str, engines: Optional[list] = None):
    """A ProgramHeartbeat for ``engines`` (default ``[ue]``) if ``ue.heartbeat_log``
    is set, else a no-op context."""
    log_path = getattr(ue, "heartbeat_log", None)
    if not log_path:
        return contextlib.nullcontext()
    engines = engines if engines is not None else [ue]
    return ProgramHeartbeat(ue.user_device, [eng._base_addr for eng in engines], log_path,
                            label, getattr(ue, "heartbeat_interval", 1.0),
                            check_board=getattr(ue, "heartbeat_precheck", False))


def board_precheck(ue, label: str) -> None:
    """One precheck block in the heartbeat log, outside any sampled program (e.g.
    before a decode loop of many short programs). No-op unless ue.heartbeat_log
    and ue.heartbeat_precheck are set; raises if the board check fails."""
    log_path = getattr(ue, "heartbeat_log", None)
    if not log_path or not getattr(ue, "heartbeat_precheck", False):
        return
    lines, failure = precheck(ue.user_device)
    with open(log_path, "a") as log:
        log.write(f"=== {label} | {datetime.datetime.now():%Y-%m-%d %H:%M:%S} | "
                  f"{ue.user_device} | precheck only ===\n")
        for line in lines:
            log.write(f"  precheck  {line}\n")
        log.write(f"=== {'not started' if failure else 'ok'} {label}"
                  f"{': precheck failed: ' + failure if failure else ''} ===\n\n")
    if failure:
        raise RuntimeError(f"precheck before {label}: {failure}")
