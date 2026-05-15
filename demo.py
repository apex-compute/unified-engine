#!/usr/bin/env python3
import curses
import io
import os
import shutil
import subprocess
import sys
import threading

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_samples")

MODELS = [
    {"name": "gemma3",        "test": "models/gemma3/gemma3_test.py",                "run": "models/gemma3/gemma3_run_from_bin.py",           "bin_dir": "models/gemma3/gemma3_bin",        "inputs": [{"type": "text",  "arg": "--prompt"}]},
    {"name": "gpt2",          "test": "models/gpt2/gpt2_test.py",                    "run": "models/gpt2/gpt2_run_from_bin.py",               "bin_dir": "models/gpt2/gpt2_bin",            "inputs": [{"type": "text",  "arg": "--prompt"}]},
    {"name": "llama3.2_1b",   "test": "models/llama3.2_1b/llama3.2_1b_test.py",     "run": "models/llama3.2_1b/llama3.2_1b_run_from_bin.py", "bin_dir": "models/llama3.2_1b/llama3.2_1b_bin", "inputs": [{"type": "text", "arg": "--prompt"}]},
    {"name": "mobilesam",     "test": "models/mobilesam/mobilesam_test.py",          "run": "models/mobilesam/mobilesam_run_from_bin.py",     "bin_dir": "models/mobilesam/mobilesam_bin",  "inputs": [{"type": "image", "arg": "--image"}], "output_image": "models/mobilesam/mask_point.png"},
    {"name": "parakeet",      "test": "models/parakeet/parakeet_test.py",            "run": "models/parakeet/parakeet_run_from_bin.py",       "bin_dir": "models/parakeet/parakeet_bin",    "inputs": [{"type": "audio", "arg": "--audio"}]},
    {"name": "qwen2.5_vl_3b", "test": "models/qwen2.5_vl_3b/qwen2.5_vl_3b_test.py","run": "models/qwen2.5_vl_3b/qwen2.5_vl_3b_run_from_bin.py", "bin_dir": "models/qwen2.5_vl_3b/qwen2.5_vl_3b_bin", "inputs": [{"type": "image", "arg": "--image"}, {"type": "text", "arg": "--prompt"}]},
    {"name": "qwen3_1.7b",    "test": "models/qwen3_1.7b/qwen3_1.7b_test.py",       "run": "models/qwen3_1.7b/qwen3_1.7b_run_from_bin.py",   "bin_dir": "models/qwen3_1.7b/qwen3_1.7b_bin", "inputs": [{"type": "text", "arg": "--prompt"}]},
    {"name": "smolvlm2",      "test": "models/smolvlm2/smolvlm2_test.py",           "run": "models/smolvlm2/smolvlm2_run_from_bin.py",       "bin_dir": "models/smolvlm2/smolvlm2_bin",    "inputs": [{"type": "image", "arg": "--image"}, {"type": "text", "arg": "--prompt"}]},
    {"name": "swin",          "test": "models/swin/swin_test.py",                    "run": "models/swin/swin_run_from_bin.py",               "bin_dir": "models/swin/swin_bin",            "inputs": [{"type": "image", "arg": "--image"}]},
]


def sample_files(input_type):
    exts = IMAGE_EXTS if input_type == "image" else AUDIO_EXTS
    try:
        return sorted(
            f for f in os.listdir(SAMPLES_DIR)
            if os.path.splitext(f)[1].lower() in exts
        )
    except OSError:
        return []

CP_TITLE    = 1
CP_HEADER   = 2
CP_SELECTED = 3
CP_OK       = 4
CP_ERR      = 5
CP_KEY      = 6
CP_OUTPUT   = 7
CP_RUNNING  = 8


def init_colors():
    curses.start_color()
    curses.use_default_colors()
    if curses.COLORS >= 256:
        BLUE = 74   # #5fafd7 — steel blue
    else:
        BLUE = curses.COLOR_CYAN
    curses.init_pair(CP_TITLE,    BLUE,               -1)
    curses.init_pair(CP_HEADER,   BLUE,               -1)
    curses.init_pair(CP_SELECTED, curses.COLOR_BLACK, BLUE)
    curses.init_pair(CP_OK,       BLUE,               -1)
    curses.init_pair(CP_ERR,      curses.COLOR_WHITE, -1)
    curses.init_pair(CP_KEY,      BLUE,               -1)
    curses.init_pair(CP_OUTPUT,   curses.COLOR_WHITE, -1)
    curses.init_pair(CP_RUNNING,  BLUE,               -1)


def bin_compiled(model):
    path = os.path.join(REPO_ROOT, model["bin_dir"])
    if not os.path.isdir(path):
        return False
    return any(os.scandir(path))


# ── output tracker ───────────────────────────────────────────────────────────
# Understands \r (in-place update) vs \n (new line) from the subprocess stream.
# completed  — lines that ended with \n (history)
# live       — the current \r-updated line (shown in-place, overwritten each draw)

class JobTracker:
    def __init__(self):
        self.lock      = threading.Lock()
        self.completed = []   # finalized lines (ended with \n)
        self.live      = ""   # current line: grows token by token, resets on \r
        self.done      = False
        self.rc        = None

    def feed_char(self, ch):
        with self.lock:
            if ch == '\r':
                self.live = ""          # in-place reset (timer update)
            elif ch == '\n':
                self.completed.append(self.live)
                self.live = ""
            else:
                self.live += ch         # token char or timer char accumulates

    def feed_line(self, text):
        """Inject a synthetic line (used before the subprocess starts)."""
        with self.lock:
            self.completed.append(text)

    def snapshot(self):
        with self.lock:
            return list(self.completed), self.live


# ── subprocess runner ────────────────────────────────────────────────────────

def run_script(script_rel, tracker: JobTracker, extra_args=None):
    script = os.path.join(REPO_ROOT, script_rel)
    cmd = [sys.executable, script] + (extra_args or [])
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # newline="" disables universal-newlines so \r stays \r and isn't
        # silently converted to \n (which would turn every timer tick into
        # a committed line instead of an in-place update).
        stdout = io.TextIOWrapper(proc.stdout, newline="",
                                  encoding="utf-8", errors="replace")
        while True:
            ch = stdout.read(1)
            if not ch:
                break
            tracker.feed_char(ch)
        proc.wait()
        tracker.rc = proc.returncode
    except Exception as e:
        tracker.feed_line(f"ERROR: {e}")
        tracker.rc = -1
    finally:
        tracker.done = True


def start_compile(model):
    bin_path = os.path.join(REPO_ROOT, model["bin_dir"])
    tracker = JobTracker()
    if os.path.isdir(bin_path):
        tracker.feed_line(f"rm -rf {model['bin_dir']}")
        shutil.rmtree(bin_path)
        tracker.feed_line("done.")
    else:
        tracker.feed_line(f"{model['bin_dir']} not found, skipping removal.")
    tracker.feed_line(f"Running {model['test']} ...")
    t = threading.Thread(target=run_script, args=(model["test"], tracker), daemon=True)
    t.start()
    return tracker


def start_run(model, extra_args=None):
    tracker = JobTracker()
    t = threading.Thread(target=run_script, args=(model["run"], tracker, extra_args), daemon=True)
    t.start()
    return tracker


# ── drawing ──────────────────────────────────────────────────────────────────

def draw(stdscr, selected, scroll_offset, tracker, status_msg):
    stdscr.erase()
    h, w = stdscr.getmaxyx()

    # Header
    try:
        stdscr.addstr(0, 2, "Apex Compute", curses.color_pair(CP_TITLE) | curses.A_BOLD)
        stdscr.addstr(1, 2, "Unified Engine", curses.color_pair(CP_OUTPUT))
        stdscr.addstr(2, 0, "─" * w, curses.color_pair(CP_HEADER))
    except curses.error:
        pass

    # Layout
    list_w    = 36
    div_x     = list_w
    out_x     = div_x + 1
    out_w     = max(0, w - out_x)
    list_rows = h - 7

    # Column header
    stdscr.addstr(3, 0, f"  {'MODEL':<20}  {'STATUS':<8}",
                  curses.color_pair(CP_HEADER) | curses.A_UNDERLINE)

    # Model rows
    for i, model in enumerate(MODELS):
        row = i - scroll_offset
        if row < 0 or row >= list_rows:
            continue
        y = 4 + row
        compiled = bin_compiled(model)
        status   = "OK" if compiled else "--"
        sp       = curses.color_pair(CP_OK) if compiled else curses.color_pair(CP_ERR)
        line     = f"  {model['name']:<20}  "
        if i == selected:
            stdscr.addstr(y, 0, (line + status)[:list_w],
                          curses.color_pair(CP_SELECTED) | curses.A_BOLD)
        else:
            stdscr.addstr(y, 0, line[:list_w])
            if len(line) < list_w:
                stdscr.addstr(y, len(line), status[:list_w - len(line)], sp)

    # Divider
    for row in range(3, h - 3):
        try:
            stdscr.addch(row, div_x, curses.ACS_VLINE)
        except curses.error:
            pass

    # Output panel header
    if out_w > 0:
        stdscr.addstr(3, out_x, " Output "[:out_w],
                      curses.color_pair(CP_HEADER) | curses.A_UNDERLINE)

    # Output panel content
    if tracker and out_w > 0:
        completed, live = tracker.snapshot()
        panel_rows = h - 7   # rows between panel header (row 3) and status bar (h-3)

        # Wrap live text into rows so we know how many rows it needs
        live_rows = []
        if live:
            live_rows = [live[i:i + out_w] for i in range(0, max(1, len(live)), out_w)]

        summary_rows = 1 if tracker.done else 0
        available = max(0, panel_rows - len(live_rows) - summary_rows)

        # Build visible history back-to-front, respecting wrapped row counts
        visible = []
        rows_used = 0
        for line in reversed(completed):
            need = max(1, len(line) // out_w + (1 if len(line) % out_w else 0)) if line else 1
            if rows_used + need > available:
                break
            visible.insert(0, line)
            rows_used += need

        # Clear the entire output area first so stale chars don't linger
        blank = " " * out_w
        for r in range(panel_rows):
            try:
                stdscr.addstr(4 + r, out_x, blank)
            except curses.error:
                pass

        # History lines — wrap long lines to out_w
        row_i = 0
        for line in visible:
            if not line:
                # blank separator
                row_i += 1
                continue
            wrapped = [line[i:i + out_w] for i in range(0, len(line), out_w)]
            for chunk in wrapped:
                try:
                    stdscr.addstr(4 + row_i, out_x, chunk, curses.color_pair(CP_OUTPUT))
                except curses.error:
                    pass
                row_i += 1

        next_y = 4 + row_i

        # Live line — in-place timer or streaming decoder text
        for chunk in live_rows:
            try:
                stdscr.addstr(next_y, out_x, chunk,
                              curses.color_pair(CP_RUNNING) | curses.A_BOLD)
            except curses.error:
                pass
            next_y += 1

        # Exit summary
        if tracker.done:
            color = curses.color_pair(CP_OK) if tracker.rc == 0 else curses.color_pair(CP_ERR)
            try:
                stdscr.addstr(next_y, out_x, f"[exit {tracker.rc}]"[:out_w],
                              color | curses.A_BOLD)
            except curses.error:
                pass

    # Status bar
    if status_msg:
        try:
            stdscr.addstr(h - 3, 0, status_msg[:w], curses.color_pair(CP_RUNNING) | curses.A_BOLD)
        except curses.error:
            pass

    # Key hints
    hints = [("a", "compile all"), ("c", "compile selected"), ("r", "run selected"), ("q", "quit")]
    x = 1
    for key, label in hints:
        chunk = f"[{key}] {label}  "
        if x + len(chunk) >= w:
            break
        stdscr.addstr(h - 2, x, f"[{key}]", curses.color_pair(CP_KEY) | curses.A_BOLD)
        stdscr.addstr(h - 2, x + len(key) + 2, f" {label}  ")
        x += len(chunk)

    stdscr.refresh()


# ── file picker popup ────────────────────────────────────────────────────────

def file_picker_popup(stdscr, title, files):
    """Arrow-key file picker. Returns chosen filename or None on cancel."""
    if not files:
        return None

    h, w = stdscr.getmaxyx()
    box_h = min(len(files) + 4, h - 4)
    box_w = min(max(len(title) + 4, max(len(f) for f in files) + 6), w - 4)
    by = (h - box_h) // 2
    bx = (w - box_w) // 2

    win = curses.newwin(box_h, box_w, by, bx)
    win.keypad(True)

    cursor = 0
    visible_rows = box_h - 4   # border + title + hint

    while True:
        win.erase()
        win.border()
        win.addstr(1, (box_w - len(title)) // 2, title[:box_w - 2], curses.A_BOLD)

        scroll = max(0, cursor - visible_rows + 1)
        for i, fname in enumerate(files[scroll:scroll + visible_rows]):
            idx = i + scroll
            y = 2 + i
            label = f"  {fname}  "[:box_w - 2]
            if idx == cursor:
                win.addstr(y, 1, label, curses.color_pair(CP_SELECTED) | curses.A_BOLD)
            else:
                win.addstr(y, 1, label)

        hint = "↑↓ select  Enter confirm  Esc cancel"
        win.addstr(box_h - 1, max(1, (box_w - len(hint)) // 2),
                   hint[:box_w - 2], curses.color_pair(CP_KEY))
        win.refresh()

        key = win.getch()
        if key == curses.KEY_UP and cursor > 0:
            cursor -= 1
        elif key == curses.KEY_DOWN and cursor < len(files) - 1:
            cursor += 1
        elif key in (curses.KEY_ENTER, ord('\n'), ord('\r')):
            return files[cursor]
        elif key == 27:   # ESC
            return None


# ── text input popup ─────────────────────────────────────────────────────────

def text_input_popup(stdscr, title):
    """Single-line text entry. Returns the entered string, or None on cancel.
    Empty string means use the model default."""
    h, w = stdscr.getmaxyx()
    box_w = min(max(len(title) + 4, 60), w - 4)
    box_h = 6
    by = (h - box_h) // 2
    bx = (w - box_w) // 2

    win = curses.newwin(box_h, box_w, by, bx)
    win.keypad(True)
    curses.curs_set(1)

    text = ""
    inner_w = box_w - 4

    while True:
        win.erase()
        win.border()
        win.addstr(1, (box_w - len(title)) // 2, title[:box_w - 2], curses.A_BOLD)
        hint = "Enter confirm  Esc cancel  (empty = default)"
        win.addstr(box_h - 1, max(1, (box_w - len(hint)) // 2),
                   hint[:box_w - 2], curses.color_pair(CP_KEY))
        display = text[-inner_w:] if len(text) > inner_w else text
        win.addstr(3, 2, display + " " * (inner_w - len(display)),
                   curses.color_pair(CP_OUTPUT))
        win.move(3, 2 + min(len(text), inner_w))
        win.refresh()

        key = win.getch()
        if key in (curses.KEY_ENTER, ord('\n'), ord('\r')):
            curses.curs_set(0)
            return text
        elif key == 27:   # ESC
            curses.curs_set(0)
            return None
        elif key in (curses.KEY_BACKSPACE, 127, ord('\b')):
            text = text[:-1]
        elif 32 <= key <= 126:
            text += chr(key)


# ── confirm popup ────────────────────────────────────────────────────────────

def confirm_popup(stdscr, message):
    h, w = stdscr.getmaxyx()
    box_w = min(len(message) + 6, w - 4)
    box_h = 5
    win = curses.newwin(box_h, box_w, (h - box_h) // 2, (w - box_w) // 2)
    win.keypad(True)
    win.border()
    label = message[:box_w - 4]
    win.addstr(1, (box_w - len(label)) // 2, label, curses.A_BOLD)
    hint = "[y] yes    [n] no"
    win.addstr(3, (box_w - len(hint)) // 2, hint, curses.color_pair(CP_KEY))
    win.refresh()
    while True:
        key = win.getch()
        if key in (ord('y'), ord('Y')):
            return True
        if key in (ord('n'), ord('N'), 27):
            return False


# ── main loop ────────────────────────────────────────────────────────────────

def main(stdscr):
    curses.curs_set(0)
    stdscr.timeout(100)   # 100 ms refresh — smooth enough for live timer
    init_colors()

    selected        = 0
    scroll_offset   = 0
    tracker         = None
    status_msg      = ""
    compile_queue   = []
    running_model   = None   # model dict of the active run job

    while True:
        h, _ = stdscr.getmaxyx()
        list_rows = h - 7  # main loop copy

        # When a run job finishes, open any output image then clear state
        if running_model and tracker and tracker.done:
            img_rel = running_model.get("output_image")
            if img_rel and tracker.rc == 0:
                img_path = os.path.join(REPO_ROOT, img_rel)
                if os.path.isfile(img_path):
                    subprocess.Popen(["xdg-open", img_path],
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
            running_model = None

        # Clear status when a single job finishes
        if tracker and tracker.done and status_msg and not compile_queue:
            status_msg = ""

        # Advance compile-all queue when current job finishes
        if compile_queue and tracker and tracker.done:
            if tracker.rc != 0:
                status_msg = f"Compile FAILED (exit {tracker.rc})"
                compile_queue = []
            elif compile_queue:
                next_model = compile_queue.pop(0)
                status_msg = f"Compiling {next_model['name']} …"
                tracker = start_compile(next_model)
            else:
                status_msg = "All models compiled."
                compile_queue = []  # ensure cleared so single-job clear fires next tick

        draw(stdscr, selected, scroll_offset, tracker, status_msg)

        key = stdscr.getch()
        if key == -1:
            continue

        if key in (ord('q'), ord('Q')):
            break
        elif key == curses.KEY_UP:
            if selected > 0:
                selected -= 1
                if selected < scroll_offset:
                    scroll_offset = selected
        elif key == curses.KEY_DOWN:
            if selected < len(MODELS) - 1:
                selected += 1
                if selected >= scroll_offset + list_rows:
                    scroll_offset = selected - list_rows + 1

        elif key in (ord('a'), ord('A')):
            if tracker and not tracker.done:
                status_msg = "A job is already running."
            elif confirm_popup(stdscr, "Recompile ALL models?"):
                compile_queue = list(MODELS)
                first = compile_queue.pop(0)
                status_msg = f"Compiling {first['name']} …"
                tracker = start_compile(first)

        elif key in (ord('c'), ord('C')):
            if tracker and not tracker.done:
                status_msg = "A job is already running."
            else:
                model = MODELS[selected]
                status_msg = f"Compiling {model['name']} …"
                tracker = start_compile(model)
                compile_queue = []

        elif key in (ord('r'), ord('R')):
            if tracker and not tracker.done:
                status_msg = "A job is already running."
            else:
                model = MODELS[selected]
                if model["run"] is None:
                    status_msg = f"{model['name']} has no run script."
                elif not bin_compiled(model):
                    status_msg = f"{model['name']} not compiled yet — press c first."
                else:
                    extra_args = []
                    cancelled = False
                    for spec in model.get("inputs", []):
                        if spec["type"] == "text":
                            prompt = text_input_popup(
                                stdscr,
                                f"Prompt — {model['name']}",
                            )
                            if prompt is None:
                                cancelled = True
                                break
                            if prompt:
                                extra_args += [spec["arg"], prompt]
                            # empty string → no args → model uses its default
                        elif spec["type"] in ("image", "audio"):
                            files = sample_files(spec["type"])
                            chosen = file_picker_popup(
                                stdscr,
                                f"Select {spec['type']} — {model['name']}",
                                files,
                            )
                            if chosen is None:
                                cancelled = True
                                break
                            extra_args += [spec["arg"],
                                           os.path.join(SAMPLES_DIR, chosen)]
                    if cancelled:
                        continue
                    status_msg = f"Running {model['name']} …"
                    tracker = start_run(model, extra_args)
                    running_model = model
                    compile_queue = []


if __name__ == "__main__":
    curses.wrapper(main)
