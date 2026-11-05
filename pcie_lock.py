#!/usr/bin/env python3
"""The board locks every process on a host shares: CI, hand runs, other users.

ONE LOCK PER BOARD. ``/tmp/pcie_dev_<board>.lock`` (``pcie_dev_xdma1.lock``) is
held, as an exclusive flock, by whatever is running on that board, from the
first time it touches the board until it exits. A run on another board takes
that board's lock and never waits for this one.

A RESCAN TAKES THEM ALL. rescan_xilinx.sh removes every Xilinx device on the
host before it rescans the bus, and programming over JTAG, booting from flash
or an ICAP warm boot drops the board's PCIe link before that rescan. So each of
those runs under ``pcie_lock.py all -- CMD``: it waits until every XDMA board
(``xdma*``) on the host is free, holds all their locks while CMD runs and drops
them when it ends. No board run can start in between, and none is running when
the bus goes away.

Usage:
    python3 pcie_lock.py all [--wait SECONDS] -- CMD [ARG...]
    python3 pcie_lock.py status

A script that must always run under ``all`` locks itself (rescan_xilinx.sh):
    [[ -n "${PCIE_LOCK_ALL_HOLDER:-}" ]] && kill -0 "$PCIE_LOCK_ALL_HOLDER" 2>/dev/null ||
        exec python3 "$(dirname "$0")/pcie_lock.py" all -- "$0" "$@"

The locks are flocks, so a holder that dies releases them; nothing has to be
cleaned up. Every holder writes who it is into the lock file, so a blocked run
names what it waits on. A process that holds a lock exports
``UE_DEVICE_LOCK_HELD_<board>=<pid>`` to its children, and a descendant treats
that board as already held: a harness that holds the board can run tests on
it, and ``all`` can run rescan_xilinx.sh, which asks for ``all`` again, without
either deadlocking on its own ancestor. user_dma_core.claim_dma_device() takes
its board through acquire() below. Standard library only, so the shell scripts
can use it without the model environment.
"""

from __future__ import annotations

import fcntl
import getpass
import glob
import os
import subprocess
import sys
import time
from typing import Callable, Optional

LOCK_DIR = "/tmp"
# Set to share a board on purpose (debugging only): every lock is then skipped.
DISABLE_ENV = "UE_ALLOW_SHARED_DEVICE"
INHERIT_ENV_PREFIX = "UE_DEVICE_LOCK_HELD_"
# Set for the command ``all`` runs, to the PID holding every board: a script
# that locks itself (rescan_xilinx.sh) knows it already runs under the lock.
ALL_HOLDER_ENV = "PCIE_LOCK_ALL_HOLDER"
WAIT_ENV = "UE_DEVICE_LOCK_WAIT"
WAIT_SECONDS = 600.0      # 10 minutes
POLL_SECONDS = 2.0
REPORT_SECONDS = 60.0     # how often to say who we are waiting on


def lock_path(board: str) -> str:
    return os.path.join(LOCK_DIR, f"pcie_dev_{board}.lock")


def inherit_env(board: str) -> str:
    return INHERIT_ENV_PREFIX + board


def disabled() -> bool:
    return bool(os.environ.get(DISABLE_ENV))


def default_wait() -> float:
    try:
        return float(os.environ.get(WAIT_ENV, WAIT_SECONDS))
    except ValueError:
        return WAIT_SECONDS


def inherited_holder(board: str) -> Optional[int]:
    """PID of a LIVE ancestor holding ``board``, from the environment.

    flock is per open file description, so a child does not inherit its
    parent's lock: it would open the file itself and block on it, and the tree
    would wait for a lock its own parent never drops (nightly run 36846050434
    sat 600s on its own model_auto_test.py harness). The claim therefore rides
    in the environment, which subprocesses do inherit. The PID is checked for
    liveness so a stale variable cannot wave an unrelated process through.
    """
    raw = os.environ.get(inherit_env(board))
    if not raw:
        return None
    try:
        pid = int(raw)
    except ValueError:
        return None
    try:
        os.kill(pid, 0)          # liveness only; signal 0 delivers nothing
    except ProcessLookupError:
        return None
    except PermissionError:
        return pid               # alive, owned by another user
    return pid


def _open(path: str) -> int:
    # O_CREAT LAST, not first. /tmp is sticky and world-writable, and these lab
    # hosts run fs.protected_regular=2, under which O_CREAT on a file owned by
    # ANOTHER user is refused with EACCES even at mode 0666. The lock file
    # outlives whoever created it, so the second user through here is normally
    # not the owner -- opening plain O_RDWR (no O_CREAT) is allowed and keeps
    # the holder annotation writable across users. A read-only fd still flocks.
    for flags in (os.O_RDWR, os.O_CREAT | os.O_RDWR, os.O_RDONLY):
        try:
            fd = os.open(path, flags, 0o666)
            break
        except OSError:
            continue
    else:
        raise RuntimeError(f"cannot open the board lock at {path}")
    try:
        os.fchmod(fd, 0o666)   # no-op unless we own it; umask must not shut others out
    except OSError:
        pass
    return fd


def holder(fd: int) -> str:
    try:
        return os.pread(fd, 400, 0).decode("utf-8", "replace").strip()
    except OSError:
        return ""


def acquire(board: str, wait_seconds: Optional[float] = None,
            refusal: Optional[Callable[[str, float, str, str], str]] = None,
            path: Optional[str] = None) -> int:
    """Take ``board``'s lock, waiting up to ``wait_seconds``; return its fd.

    The caller holds the board until it closes the fd or exits. Raises
    RuntimeError naming the holder if the board is still busy at the deadline;
    ``refusal(board, waited, holder, path)`` may supply that message. ``path``
    defaults to lock_path(board).
    """
    path = path or lock_path(board)
    fd = _open(path)
    if wait_seconds is None:
        wait_seconds = default_wait()
    deadline = time.monotonic() + max(0.0, wait_seconds)
    next_report = 0.0        # report the first wait immediately
    waited = 0.0
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                who = holder(fd)
                os.close(fd)
                raise RuntimeError(
                    refusal(board, waited, who, path) if refusal else
                    f"{board} is still in use after waiting {waited:.0f}s"
                    + (f" -- {who}" if who else "")
                    + f"\n  Lock: {path}  ({WAIT_ENV}=<seconds> to change the "
                    f"{wait_seconds:.0f}s wait, 0 to fail on contact)") from None
            if waited >= next_report:
                who = holder(fd)
                print(f"[device-lock] {board} is busy" + (f" ({who})" if who else "")
                      + f"; waiting up to {remaining:.0f}s more...", flush=True)
                next_report = waited + REPORT_SECONDS
            time.sleep(min(POLL_SECONDS, max(0.05, remaining)))
            waited = wait_seconds - max(0.0, deadline - time.monotonic())
    if waited > 0:
        print(f"[device-lock] acquired {board} after {waited:.0f}s", flush=True)
    try:
        os.ftruncate(fd, 0)
        os.write(fd, (f"pid={os.getpid()} user={getpass.getuser()} "
                      f"host={os.uname().nodename} "
                      f"cmd={' '.join(sys.argv)}\n").encode())
    except OSError:
        pass   # lock-only fd; holding it matters, the annotation does not
    return fd


def release(board: str, fd: Optional[int]) -> None:
    if fd is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass
    # Only clear the marker we set ourselves: a descendant that releases must
    # not convince its siblings the board is free while the ancestor holds it.
    if os.environ.get(inherit_env(board)) == str(os.getpid()):
        os.environ.pop(inherit_env(board), None)


def boards() -> list[str]:
    """Every XDMA board on this host, named as its lock is (xdma0, xdma1).

    The live nodes, plus every board that has a lock file: a board that has
    dropped off the bus has no node but may still have a run holding its lock.
    """
    names = {os.path.basename(p)[:-len("_user")] for p in glob.glob("/dev/xdma*_user")}
    names |= {os.path.basename(p)[len("pcie_dev_"):-len(".lock")]
              for p in glob.glob(os.path.join(LOCK_DIR, "pcie_dev_xdma*.lock"))}
    return sorted(names)


def acquire_all(wait_seconds: Optional[float] = None) -> dict[str, int]:
    """Lock every board on the host, in name order, and mark them for children.

    Boards an ancestor already holds are skipped (it is waiting on us). Taking
    them in one order means two ``all`` holders cannot deadlock, and a board run
    only ever holds one lock, so it cannot form a cycle with either.
    """
    if wait_seconds is None:
        wait_seconds = default_wait()
    deadline = time.monotonic() + max(0.0, wait_seconds)
    held: dict[str, int] = {}
    try:
        for board in boards():
            if inherited_holder(board) is not None:
                continue
            held[board] = acquire(board, max(0.0, deadline - time.monotonic()))
            os.environ[inherit_env(board)] = str(os.getpid())
    except BaseException:
        for board, fd in held.items():
            release(board, fd)
        raise
    return held


def _status() -> int:
    for board in boards():
        path = lock_path(board)
        try:
            fd = _open(path)
        except RuntimeError as error:
            print(f"{board:12s} {error}")
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            print(f"{board:12s} free     (last: {holder(fd) or '-'})")
        except OSError:
            print(f"{board:12s} held by  {holder(fd) or '?'}")
        finally:
            os.close(fd)
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["status"]:
        return _status()
    if argv[:1] != ["all"]:
        print(__doc__.split("\n\n")[3], file=sys.stderr)
        return 2
    argv = argv[1:]
    wait_seconds = None
    if argv[:1] == ["--wait"] and len(argv) >= 2:
        wait_seconds = float(argv[1])
        argv = argv[2:]
    if argv[:1] != ["--"] or len(argv) < 2:
        print("usage: pcie_lock.py all [--wait SECONDS] -- CMD [ARG...]", file=sys.stderr)
        return 2
    command = argv[1:]
    if disabled():
        return subprocess.call(command)
    try:
        held = acquire_all(wait_seconds)
    except RuntimeError as error:
        print(f"[device-lock] not running {command[0]}: {error}", file=sys.stderr)
        return 75
    names = ", ".join(held) or "none (an ancestor holds every board)"
    print(f"[device-lock] holding every board for {os.path.basename(command[0])}: "
          f"{names}", flush=True)
    os.environ[ALL_HOLDER_ENV] = str(os.getpid())
    try:
        return subprocess.call(command)
    finally:
        for board, fd in held.items():
            release(board, fd)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
