#!/usr/bin/env python3
"""One Qwen2.5-Omni 7B request for the unified web server.

Runs ``qwen2.5_omni_7b_test.py``'s own ``main()`` (its flags, its programs.bin
reuse via ``--run_from_bin``, its eight-engine stages) and adds the server's
channel: everything the model prints goes to stderr (the run log), and this
script's protocol goes to the original stdout as JSON lines:

    {"event": "token", "text": "..."}        each decoded piece
    {"event": "result", "metrics": {...}}    stage rows, clock / peak, TEST_RESULT
    {"event": "error", "error": "..."}

A ``stop`` line on stdin ends the generation after the current token.

Qwen's compiled programs depend on the prompt's shape (token count, media,
stages), so the model is set up per request rather than kept resident; with
``--run_from_bin`` a request of an already-built shape skips the compile.

    python models/unified_server/qwen_runner.py --multi-core 8 --run_from_bin --prompt "Hi"
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
QWEN = os.path.join(ROOT, "models", "qwen2.5_omni_7b", "qwen2.5_omni_7b_test.py")

_PROTO = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
os.dup2(2, 1)
sys.stdout = sys.stderr
_LOCK = threading.Lock()


def emit(event: str, **kw) -> None:
    with _LOCK:
        _PROTO.write(json.dumps({"event": event, **kw}, ensure_ascii=False, default=float) + "\n")
        _PROTO.flush()


class _OnToken:
    """Per-token hook for the decode loop (a plain callable: no method binding)."""

    def __init__(self, stop: threading.Event):
        self.stop = stop

    def __call__(self, token_id, text) -> bool:
        emit("token", text=text)
        return self.stop.is_set()


def main() -> int:
    os.chdir(ROOT)
    stop = threading.Event()

    def watch_stdin() -> None:
        for line in sys.stdin:
            if line.strip() == "stop":
                stop.set()
    threading.Thread(target=watch_stdin, daemon=True).start()

    spec = importlib.util.spec_from_file_location("qwen25_omni_test", QWEN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qwen25_omni_test"] = mod
    spec.loader.exec_module(mod)
    engine_cls = mod.Qwen25OmniUnifiedEngine
    engine_cls.on_token = _OnToken(stop)

    captured: dict = {}
    original_metrics = engine_cls.stage_metrics

    def stage_metrics(self, args):
        rows = original_metrics(self, args)
        captured["rows"] = rows
        try:
            captured["peak_gflops"] = float(self.vis_peak_gflops())
            captured["core_peak_gflops"] = float(self.per_core_peak_gflops())
            captured["engines"] = int(getattr(self, "multi_core", 8) or 8)
            captured["clock_mhz"] = 1000.0 / float(self._clock_period_ns)
            steps = list(getattr(self, "_decode_step_us", ()) or ())
            captured["decode_first_us"] = float(steps[0]) if steps else None
            captured["decode_steps"] = len(steps)
            captured["context_limit"] = int(getattr(self, "MAX_CONTEXT_SIZE", 0) or 0)
            captured["weights_dram_bytes"] = int(self.get_params_dram_usage())
        except Exception:  # noqa: BLE001 - reporting extras are best effort
            pass
        return rows
    engine_cls.stage_metrics = stage_metrics

    # TEST_RESULT is printed by main(); catch it on its way to the log.
    original_print = mod._ORIGINAL_PRINT

    def tee_print(*a, **k):
        if a and isinstance(a[0], str) and a[0].startswith("TEST_RESULT: "):
            try:
                captured["test_result"] = json.loads(a[0][len("TEST_RESULT: "):])
            except ValueError:
                pass
        return original_print(*a, **k)
    mod._ORIGINAL_PRINT = tee_print              # _quiet_print looks it up at call time

    sys.argv = [QWEN] + sys.argv[1:]
    try:
        mod.main()
    except SystemExit as e:
        if e.code not in (None, 0):
            emit("error", error=f"the Qwen run exited with {e.code}")
            return 1
    except BaseException as e:  # noqa: BLE001 - report it to the page
        traceback.print_exc()
        emit("error", error=f"{type(e).__name__}: {e}")
        return 1
    emit("result", metrics={**captured, "stopped": stop.is_set()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
