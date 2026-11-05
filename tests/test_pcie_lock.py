"""pcie_lock.py: one lock per board, and ``all`` holds every board for a rescan.

Runs in a private lock directory, so no real board's lock is touched; the
processes that compete are real subprocesses, since flock is per open file.
"""

import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PCIE_UTILS = os.path.dirname(HERE)


class PcieLockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pcie_lock_test_")
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith("UE_DEVICE_LOCK_HELD_")}
        self.env["UE_DEVICE_LOCK_WAIT"] = "20"
        self.procs = []

    def tearDown(self):
        for proc in self.procs:
            proc.kill()
            proc.wait()

    def python(self, body, env=None):
        """Start a python that sees pcie_lock with LOCK_DIR set to the tmp dir."""
        code = (f"import sys; sys.path.insert(0, {PCIE_UTILS!r})\n"
                f"import pcie_lock; pcie_lock.LOCK_DIR = {self.tmp!r}\n"
                + textwrap.dedent(body))
        proc = subprocess.Popen([sys.executable, "-c", code], env=env or self.env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
        self.procs.append(proc)
        return proc

    def hold(self, board, seconds):
        """A board run: holds ``board`` for ``seconds``, says when it has it."""
        proc = self.python(f"""
            import time
            pcie_lock.acquire({board!r})
            print("held", flush=True)
            time.sleep({seconds})
        """)
        for line in proc.stdout:     # a waiting run says so first
            if line.strip() == "held":
                return proc
        self.fail(f"{board} was never held")

    def run_all(self, command, wait=20):
        """``all`` around a command, timed: returns (seconds, output)."""
        started = time.monotonic()
        proc = self.python(f"""
            sys.exit(pcie_lock.main(["all", "--wait", "{wait}", "--"] + {command!r}))
        """)
        out, _ = proc.communicate(timeout=60)
        return time.monotonic() - started, proc.returncode, out

    def test_other_boards_do_not_wait(self):
        self.hold("xdmatestA", 5)
        started = time.monotonic()
        self.hold("xdmatestB", 0)
        self.assertLess(time.monotonic() - started, 2)

    def test_all_waits_for_every_board_run(self):
        self.hold("xdmatestA", 3)
        self.hold("xdmatestB", 0.1)
        seconds, rc, out = self.run_all(["true"])
        self.assertEqual(rc, 0, out)
        self.assertGreaterEqual(seconds, 2.5, out)
        self.assertIn("xdmatestA", out)

    def test_board_run_waits_while_all_holds(self):
        self.hold("xdmatestA", 0)   # leave a lock file, so all knows the board
        allp = self.python("""
            sys.exit(pcie_lock.main(["all", "--", "sh", "-c", "echo rescanning; sleep 3"]))
        """)
        self.assertIn("rescanning", allp.stdout.readline() + allp.stdout.readline())
        started = time.monotonic()
        self.hold("xdmatestA", 0)
        self.assertGreaterEqual(time.monotonic() - started, 2)

    def test_all_gives_up_naming_the_holder(self):
        self.hold("xdmatestA", 10)
        seconds, rc, out = self.run_all(["true"], wait=1)
        self.assertEqual(rc, 75, out)
        self.assertIn("xdmatestA is still in use", out)

    def test_nested_all_does_not_wait_for_its_ancestor(self):
        self.hold("xdmatestA", 0)
        inner = (f"import sys; sys.path.insert(0, {PCIE_UTILS!r}); import pcie_lock; "
                 f"pcie_lock.LOCK_DIR = {self.tmp!r}; "
                 f"sys.exit(pcie_lock.main(['all', '--wait', '0', '--', 'true']))")
        seconds, rc, out = self.run_all([sys.executable, "-c", inner])
        self.assertEqual(rc, 0, out)
        self.assertIn("an ancestor holds every board", out)

    def test_all_inside_a_board_run_skips_that_board(self):
        """update_flash.py holds its board and then runs rescan_xilinx.sh."""
        proc = self.python(f"""
            import os, subprocess
            pcie_lock.acquire("xdmatestA")
            os.environ[pcie_lock.inherit_env("xdmatestA")] = str(os.getpid())
            inner = ("import sys; sys.path.insert(0, {PCIE_UTILS!r}); import pcie_lock; "
                     "pcie_lock.LOCK_DIR = {self.tmp!r}; "
                     "sys.exit(pcie_lock.main(['all', '--wait', '0', '--', 'true']))")
            sys.exit(subprocess.call([sys.executable, "-c", inner]))
        """)
        out, _ = proc.communicate(timeout=60)
        self.assertEqual(proc.returncode, 0, out)


if __name__ == "__main__":
    unittest.main()
