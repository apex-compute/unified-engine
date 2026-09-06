#!/usr/bin/env python3
"""Qwen2.5-VL-3B on the accelerator -- gemma4-style refactor.

Structure follows models/gemma4_e2b: a thin entrypoint (this file) holding the
CLI, the DRAM map and the engine class, with each stage's methods in a sibling
mixin module. Vision lives in qwen2.5_vl_3b_vision.py; the LM mixin follows.

STATUS: engine construction, the DRAM map and vision weight loading. Program
compilation and execution are not wired up yet.

    python qwen2.5_vl_3b_test_new.py --dev xdma0 --image
    python qwen2.5_vl_3b_test_new.py --dev xdma0 --multi-core 8
"""
import argparse
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(SCRIPT_DIR)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

import torch

# --- BROAD PRINT SUPPRESSION FOR LIBRARIES ---
# The engine cores print per-instruction detail; emitting a 32-layer encoder
# would produce tens of thousands of lines. _set_silent() toggles this while a
# capture is running, and _loud() bypasses it for progress output.
import builtins

_original_print = builtins.print
_SILENT_MODE = False


def quiet_print(*args, **kwargs):
    if _SILENT_MODE:
        return
    _original_print(*args, **kwargs)


builtins.print = quiet_print
# ---------------------------------------------

import user_dma_core
from user_dma_core import UnifiedEngine, UE_VECTOR_SIZE, set_dma_device
from multi_engine_shard import MultiEngineScheduler, PrivateArena

# The sibling mixin file is named for the model (qwen2.5_vl_3b_vision.py), and
# "2.5" makes that an invalid module name, so it is loaded by path rather than
# by `import`. gemma4_e2b gets a plain import only because its name happens to
# be a valid identifier.
def _load_sibling(module_name: str, filename: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(SCRIPT_DIR, filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_vision_mod = _load_sibling("qwen2_5_vl_3b_vision", "qwen2.5_vl_3b_vision.py")
Qwen25VLVisionMixin = _vision_mod.Qwen25VLVisionMixin
_lm_mod = _load_sibling("qwen2_5_vl_3b_lm", "qwen2.5_vl_3b_lm.py")
Qwen25VLLMMixin = _lm_mod.Qwen25VLLMMixin

DEFAULT_IMAGE = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "..", "test_samples", "yosemite.jpg"))

# Static context capacity. Tensor sizing and decoder bounds follow this one
# constant; it must stay 64-aligned.
#
# WHY 2048 AND NOT 4096. Two terms scale with it and together overrun the
# 224 MiB tensor region: the KV cache (36 KiB/token -> 144 MiB at 4096) and
# unified_attention_core's scratch, whose score buffer is
# [aligned_seq, aligned_seq] -- 33 MiB at 4096 against 8.5 at 2048. The total
# is ~239 MiB at ctx 4096 / prefill 512 versus ~142 MiB at ctx 2048.
# Raising it to 4096 needs the map re-carved (params 1808 -> ~1804 MiB and ISA
# 16 -> 8 MiB buys ~12 MiB, still short), so it is a deliberate decision, not a
# constant to flip blind.
MAX_CONTEXT_SIZE = 2048
MIN_CONTEXT_SIZE = 512

# Engine ceiling for this model's private map: 8 cores x 256 MiB fills the low
# 2 GB exactly, which is the interleave granularity the DRAM controller wants.
MAX_ENGINES = 8


class Qwen25VL_UnifiedEngine(Qwen25VLLMMixin, Qwen25VLVisionMixin, UnifiedEngine):
    """Qwen2.5-VL-3B engine: DRAM map, weight loading, stage mixins."""

    def __init__(self, script_dir: str | None = None, multi_core: int = 1):
        if not 1 <= multi_core <= MAX_ENGINES:
            raise ValueError(
                f"multi_core must be between 1 and {MAX_ENGINES}, got {multi_core}")
        self.multi_core = multi_core
        self._multi_core_schedulers = {}

        # ------------------------------------------------------------------
        # DRAM MAP (4 GiB, 32-bit addressing -- everything below 0x1_0000_0000)
        #
        # MODEL MAP, upper 2 GB (identical at every engine count):
        #   PARAMS  weights  : 0x8000_0000 - 0xF100_0000  (1808 MiB)
        #   TENSOR  acts/KV  : 0xF100_0000 - 0xFB00_0000  ( 160 MiB)
        #   ISA     programs : 0xFB00_0000 - 0x1_0000_0000 (  80 MiB)
        #
        # ISA IS 80 MiB, NOT 16. The LM keeps a per-prompt prefill program AND
        # the decoder resident at once -- 8.7 + 7.5 MiB at a 37-token prompt,
        # and prefill grows with the prompt -- which overran a 16 MiB region.
        # The tensor side had the slack: it uses 125 MiB of its 224 at ctx 2048.
        #
        # THE PARAMS WINDOW IS TIME-SHARED, NOT SPLIT. Vision and LM weights do
        # not fit side by side -- LM is 1801.7 MiB and vision another 389.7 MiB,
        # against 2048 MiB of upper DRAM before a single activation. They do not
        # have to coexist: the encoder runs once, ahead of prefill, and its only
        # output is 144 x 2048 embeddings. So both load at 0x8000_0000:
        #
        #   phase A  vision weights (389.7 MiB) -> encoder runs -> 144 embeddings
        #            copied into the tensor region
        #   phase B  params cursor rewound, LM weights (1801.7 MiB) loaded over
        #            the same addresses -> prefill + decode
        #
        # Nothing from phase A may be read after phase B begins. The rewind is
        # explicit (reset_params_dram_addr) rather than implied by a cursor that
        # happens to be low, so the handover is visible at the call site.
        #
        # MULTI-CORE PRIVATE SPACE, the whole lower 2 GB -- empty at 1 core, and
        # handed to PrivateArena, which splits it into one window per engine laid
        # out [ weights | ISA | tensor ] with the two fixed slices at the top:
        #   8 engines -> 256 MiB/window: 224 MiB weights + 16 MiB ISA + 16 MiB tensor
        # The model map begins at 0x8000_0000 and allocates upward, so a window
        # carved from the low 2 GB cannot alias model memory however the model's
        # cursors move. Same arrangement as gemma4_e2b and gemma3.
        # ------------------------------------------------------------------
        self.DRAM_END = 0x100000000
        self.PARAMS_BASE = 0x80000000
        self.PARAMS_LIMIT = 0xF1000000        # 1808 MiB, > the 1801.7 MiB LM needs
        self.TENSOR_BASE = self.PARAMS_LIMIT
        self.ISA_BASE = 0xFB000000
        self.TENSOR_LIMIT = self.ISA_BASE
        # Vision loads at the base of the same window (see the note above).
        self.VISION_WEIGHT_BASE = self.PARAMS_BASE

        self.mc_arena = (PrivateArena(multi_core, arena_base=0x00000000,
                                      arena_bytes=self.PARAMS_BASE, verbose=True)
                         if multi_core > 1 else None)

        super().__init__(BASE_ADDR=user_dma_core.UE_0_BASE_ADDR,
                         params_dram_base=self.PARAMS_BASE,
                         program_dram_base=self.ISA_BASE,
                         tensor_dram_base=self.TENSOR_BASE)

        self.script_dir = script_dir or SCRIPT_DIR
        self._cfg = self.load_config(script_dir=self.script_dir)

        fi = self._cfg["file_info"]
        model = self._cfg["model"]
        self.vector_length = fi["hidden_size"]
        self.head_dim = fi["head_dim"]
        self.actual_head_dim = fi["actual_head_dim"]
        self.num_kv_heads = fi["num_kv_heads"]
        self.group_size = fi["group_size"]
        self.mlp_elements = fi["mlp_elements"]
        self.bytes_per_element = fi["bytes_per_element"]
        self.LAYER_SIZE = fi["num_layers"]
        self.EMBEDDING_ELEMENTS = fi["embedding_vocab"]

        self.MAX_CONTEXT_SIZE = MAX_CONTEXT_SIZE
        model["max_context_size"] = MAX_CONTEXT_SIZE
        if (self.MAX_CONTEXT_SIZE < MIN_CONTEXT_SIZE
                or self.MAX_CONTEXT_SIZE > MAX_CONTEXT_SIZE
                or self.MAX_CONTEXT_SIZE % 64):
            raise ValueError(
                f"max_context_size must be 64-aligned in "
                f"[{MIN_CONTEXT_SIZE}, {MAX_CONTEXT_SIZE}], got {self.MAX_CONTEXT_SIZE}")
        self.PREFILL_MAX_SEQ_LEN = int(
            model.get("prefill_max_seq_len", self.MAX_CONTEXT_SIZE))
        if self.PREFILL_MAX_SEQ_LEN > self.MAX_CONTEXT_SIZE:
            raise ValueError(
                f"prefill_max_seq_len ({self.PREFILL_MAX_SEQ_LEN}) cannot exceed "
                f"max_context_size ({self.MAX_CONTEXT_SIZE})")

        # Fixed ISA register bindings; dynamic allocation starts past them.
        fixed = self._cfg["fixed_isa_regs"]
        self.TMP_REG = fixed["TMP_REG"]
        self.gf_seq_len = fixed["GF_SEQ_LEN_REG"]
        self.gf_q_seq_len = fixed["GF_Q_SEQ_LEN_REG"]
        self.gf_aligned_seq_len = fixed["GF_ALIGNED_SEQ_LEN_REG"]
        self._isa_reg_base = max(fixed.values()) + 1
        self._isa_reg_counter = self._isa_reg_base
        # Decode runs every bulk op at M=1; a dedicated register holding 1 keeps
        # the emitters uniform with prefill's runtime row count.
        self.gf_one = self.alloc_isa_reg()
        self._isa_reg_base = self._isa_reg_counter

        self._end_of_turn_token_id = model["end_of_turn_token_id"]
        self.causal_mask_upper = False

    def _ensure_stage_scheduler(self, stage: str):
        """The shared MultiEngineScheduler for one stage, over the model arena.

        THE ARENA IS PASSED IN, NOT BUILT PER STAGE. Every stage's workers
        execute from their own window in ``self.mc_arena``; two allocators over
        one address range would hand out the same address twice and stage 2's
        worker programs would land on stage 1's. One object, one cursor per
        engine, for the whole run.
        """
        if self.multi_core == 1:
            return None
        sched = self._multi_core_schedulers.get(stage)
        if sched is None:
            sched = MultiEngineScheduler(
                self, num_engines=self.multi_core,
                engine_base_stride=0x00010000,
                arena=self.mc_arena,
                # One four-phase master/worker round per region instead of an
                # all-to-all barrier at entry and exit: O(N) checks rather than
                # O(N^2), and every flag edge is acknowledged so it does not
                # need NOP timing margin.
                region_rendezvous="master_worker",
                barrier_margin_nops=32,
                allow_unaligned_rows=True,
                allow_more_than_two_engines=self.multi_core > 2)
            self._multi_core_schedulers[stage] = sched
        return sched

    # ---- plumbing shared with the mixins ----------------------------------

    @staticmethod
    def load_config(config_path: str | None = None,
                    script_dir: str | None = None) -> dict:
        if config_path is None:
            config_path = os.path.join(script_dir or SCRIPT_DIR,
                                       "qwen2.5_vl_3b_config.json")
        with open(config_path) as f:
            return json.load(f)

    def _loud(self, *args, **kwargs) -> None:
        """Progress print that bypasses suppression (the saved original print)."""
        _original_print(*args, **kwargs)

    def _set_silent(self, on: bool) -> bool:
        """Toggle library-print suppression; returns the previous state so the
        caller can restore it. Encapsulates the module global so the mixins in
        the sibling files never need a `global _SILENT_MODE` of their own."""
        global _SILENT_MODE
        prev = _SILENT_MODE
        _SILENT_MODE = on
        return prev

    def _profile_tables(self, stages) -> list:
        """Per-phase markdown tables for one or more stages.

        Shared by every stage: the aggregation, the peak-utilisation guard and
        the column set are identical whether the samples came from the vision
        encoder, prefill, or a decode step.
        """
        peak = self.vis_peak_gflops()
        out = []
        for title, note, results in stages:
            if not results:
                continue
            rows = self._aggregate_vis_profile(results)
            total = sum(r["ms"] for r in rows) or 1.0
            out += [f"### {title}", ""]
            if note:
                out += [note, ""]
            out += ["| phase | calls | total ms | share | GFLOP | GFLOPS | % of peak |",
                    "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for name, n, ms, share, gf, gfs, util in self._vis_profile_table(rows, total):
                out.append(f"| {name} | {n} | {ms:.2f} | {share:.1f}% | "
                           f"{gf:.2f} | {gfs:.1f} | {util:.1f}% |")
            tot_gf = sum(r["flops"] for r in rows) / 1e9
            tot_gfs = tot_gf / (total / 1e3) if total else 0.0
            out.append(f"| **TOTAL** | {len(results)} | {total:.2f} | 100.0% | "
                       f"{tot_gf:.2f} | {tot_gfs:.1f} | "
                       f"{(100 * tot_gfs / peak if peak else 0):.1f}% |")
            out.append("")
        return out

    def write_run_summary(self, out_path: str, args, profiles=None) -> str:
        """Per-run Markdown summary: hardware, sizes, and per-stage metrics.

        Reads only attributes the stages already stashed plus cheap host-side
        bookkeeping, so calling it after a run launches no FPGA program.

        Two clocks are reported deliberately and they answer different
        questions. The HW counter times the program on the engine; the CPU timer
        wraps the whole step including the per-token host work (embedding
        gather, RoPE and bias DMA, preamble write, argmax readback, detokenize).
        Their ratio is the host overhead, which is what to attack if HW
        utilisation already looks good.
        """
        clock_ns = (getattr(self, "_clock_period_ns", None)
                    or user_dma_core.CLOCK_CYCLE_TIME_NS)
        freq_mhz = 1000.0 / clock_ns if clock_ns else 0.0
        peak = self.vis_peak_gflops()
        try:
            hw = f"0x{self.user_read_reg32(user_dma_core.UE_FPGA_VERSION_ADDR) & 0xFFFFFFFF:08x}"
        except Exception as exc:
            hw = f"(read failed: {exc})"

        def util(g):
            return f"{100.0 * g / peak:.1f}% of peak" if peak and g else "n/a"

        params_bin = os.path.join(self.script_dir, self._cfg["paths"]["params"])
        L = [
            "# qwen2.5_vl_3b run summary",
            "",
            "## Hardware",
            "",
            f"- **HW version:** {hw}",
            f"- **Device:** {args.dev}",
            f"- **Clock:** {clock_ns:.4f} ns ({freq_mhz:.1f} MHz)",
            f"- **AXI data width:** {user_dma_core.UE_AXI_DATA_WIDTH_BITS} bits",
            f"- **DRAM:** {user_dma_core.AVAILABLE_DRAM_SIZE_GB} GiB",
            f"- **Cores in use:** {self.multi_core} of "
            f"{user_dma_core.ANDROMEDA_CORE_COUNT} reported",
            f"- **Peak throughput:** {peak:.1f} GFLOPS "
            f"({freq_mhz:.1f} MHz x 128 x {self.multi_core} core(s))",
            "",
            "## Weights and programs",
            "",
            f"- **Weight bin:** `{os.path.basename(params_bin)}` — "
            f"{os.path.getsize(params_bin) / 2**20:.1f} MiB"
            if os.path.exists(params_bin) else "- **Weight bin:** n/a",
        ]
        vis_w = (getattr(self, "_vis_weight_end", 0)
                 - getattr(self, "_vis_weight_start", 0))
        if vis_w:
            L.append(f"- **Vision weight DRAM:** {vis_w / 2**20:.1f} MiB (IF4)")
        if getattr(self, "_lm_weight_init_done", False):
            L.append(f"- **LM weight DRAM:** "
                     f"{(self._lm_weight_end - self.PARAMS_BASE) / 2**20:.1f} MiB "
                     f"(IF4 + BF16 V/O; embedding host-side)")
        for name, prog in (("Vision encoder", getattr(self, "_vis_program_bytes", None)),
                           ("Prefill program", getattr(self, "_prefill_program", None)),
                           ("Decoder program", getattr(self, "_decoder_program", None))):
            blob = prog[1] if isinstance(prog, tuple) else prog
            if blob:
                L.append(f"- **{name}:** {len(blob) / 2**20:.2f} MiB")
        L.append("")

        if getattr(self, "_vis_latency_us", None):
            d = self._vision_dims()
            L += [
                "## Vision",
                "",
                f"- **Image:** `{os.path.basename(getattr(args, 'image', '') or '')}` "
                f"-> {d['VS']} patches -> {d['NUM_MERGED_TOKENS']} tokens",
                f"- **HW latency:** {self._vis_latency_us / 1e3:.1f} ms",
                f"- **Reported FLOPs:** {self._vis_total_flops / 1e9:.1f} GFLOP",
                f"- **Throughput:** {self._vis_gflops:.1f} GFLOPS "
                f"({util(self._vis_gflops)})",
                f"- **End-to-end (CPU timer):** {self._vis_wall_s:.2f} s",
                "",
            ]

        if getattr(self, "_latency_prefill_us", None):
            L += [
                "## Prefill",
                "",
                f"- **Sequence length:** {self._prefill_seq_len_run} tokens",
                f"- **HW latency:** {self._latency_prefill_us / 1e3:.1f} ms",
                f"- **Reported FLOPs:** {self._prefill_flops / 1e9:.1f} GFLOP",
                f"- **Throughput:** {self._prefill_gflops:.1f} GFLOPS "
                f"({util(self._prefill_gflops)})",
                f"- **End-to-end (CPU timer):** {self._prefill_wall_s:.2f} s",
                "",
            ]

        steps = getattr(self, "_decode_step_us", None)
        if steps:
            n = self._decode_n
            first_us = steps[0]
            hw_avg_us = self._decode_total_us / n
            L += [
                "## Decode",
                "",
                f"- **Tokens generated:** {n} (sequence total {self.seq_len})",
                f"- **First-token speed (peak, HW counter):** "
                f"{1e6 / first_us:.2f} tok/s ({first_us / 1e3:.1f} ms)",
                f"- **Average speed (HW counter):** {1e6 / hw_avg_us:.2f} tok/s "
                f"({hw_avg_us / 1e3:.1f} ms/token)",
                f"- **Average speed (CPU timer):** {n / self._decode_wall_s:.2f} tok/s "
                f"({1e3 * self._decode_wall_s / n:.1f} ms/token)",
                f"- **Host overhead:** "
                f"{100 * (1 - self._decode_total_us / 1e6 / self._decode_wall_s):.1f}% "
                f"of wall time outside the engine",
                f"- **FLOPs per token:** {self._decode_step_flops / n / 1e9:.2f} GFLOP",
                f"- **Average throughput:** {self._decode_gflops:.1f} GFLOPS "
                f"({util(self._decode_gflops)})",
                f"- **End-to-end (CPU timer):** {self._decode_wall_s:.2f} s",
                "",
            ]

        if profiles:
            L += ["## Per-phase profile", "",
                  "Phase latencies come from the HW counter between per-phase HALTs: "
                  "they exclude host time but include one stop/restart per phase. "
                  "`*` marks phases that run on core 0 only, whose % of peak is "
                  "measured against ONE core.", ""]
            L += self._profile_tables(profiles)

        prompt = getattr(self, "_prompt_text", None)
        if prompt is not None:
            L += ["## Prompt & output", "", "### Prompt", "", "```", prompt, "```", ""]
            L += ["### Decoded text", "", "```",
                  getattr(self, "_decoded_text", "") or "(none)", "```", ""]
        with open(out_path, "w") as f:
            f.write("\n".join(L))
        return out_path

    def describe_dram_map(self) -> str:
        lines = [
            "  model map (upper 2 GB):",
            f"    PARAMS  0x{self.PARAMS_BASE:08X} - 0x{self.PARAMS_LIMIT:08X}  "
            f"{(self.PARAMS_LIMIT - self.PARAMS_BASE) / 2**20:6.0f} MiB  "
            f"(time-shared: vision, then LM)",
            f"    TENSOR  0x{self.TENSOR_BASE:08X} - 0x{self.TENSOR_LIMIT:08X}  "
            f"{(self.TENSOR_LIMIT - self.TENSOR_BASE) / 2**20:6.0f} MiB",
            f"    ISA     0x{self.ISA_BASE:08X} - 0x{self.DRAM_END:09X}  "
            f"{(self.DRAM_END - self.ISA_BASE) / 2**20:6.0f} MiB",
        ]
        if self.mc_arena is not None:
            lines.append(self.mc_arena.describe())
        else:
            lines.append("  private map: single core, low 2 GB unused")
        return "\n".join(lines)

    def check_params_fit(self, needed_bytes: int, what: str) -> None:
        """Fail early if a phase's weights cannot fit the shared params window."""
        capacity = self.PARAMS_LIMIT - self.PARAMS_BASE
        if needed_bytes > capacity:
            raise MemoryError(
                f"{what} needs {needed_bytes / 2**20:.1f} MiB but the params "
                f"window is {capacity / 2**20:.1f} MiB "
                f"[0x{self.PARAMS_BASE:X}..0x{self.PARAMS_LIMIT:X})")


def add_engine_args(parser) -> None:
    """Knobs shared by every qwen2.5_vl_3b_new entrypoint."""
    parser.add_argument("--dev", type=str, default="xdma0",
                        help="DMA device name (e.g. xdma0, xdma1, efinix). Default: xdma0")
    parser.add_argument("--multi-core", nargs="?", const=2, default=1, type=int,
                        help=f"Enable multi-engine execution. Bare --multi-core selects 2 "
                             f"engines; the ceiling is the lower of {MAX_ENGINES} and the "
                             f"engine count HW_INFO reports for the loaded bitstream.")


def resolve_engine_config(parser, args) -> dict:
    """Validate the shared knobs against HW_INFO and sync the DMA globals.

    Mirrors gemma4_e2b_test.resolve_engine_config: HW_INFO is the sole source of
    clock, AXI width and engine count, and it must be read before any
    UnifiedEngine is constructed.
    """
    if not 1 <= args.multi_core <= MAX_ENGINES:
        parser.error(f"--multi-core must be between 1 and {MAX_ENGINES}")

    set_dma_device(args.dev)
    for _name, _mod in list(sys.modules.items()):
        if _name.startswith("qwen2_5_vl_3b") and _mod is not None:
            for _attr in ("DMA_DEVICE_H2C", "DMA_DEVICE_C2H", "DMA_DEVICE_USER"):
                if hasattr(_mod, _attr):
                    setattr(_mod, _attr, getattr(user_dma_core, _attr))

    user_dma_core.configure_clock_from_hardware()
    cores = user_dma_core.ANDROMEDA_CORE_COUNT
    if cores and args.multi_core > cores:
        parser.error(
            f"--multi-core {args.multi_core} exceeds the {cores} engine(s) this "
            f"board reports in HW_INFO")

    print(user_dma_core.hardware_info_summary())
    print(f"Using DMA device: {args.dev}")
    print(f"  H2C: {user_dma_core.DMA_DEVICE_H2C}")
    print(f"  C2H: {user_dma_core.DMA_DEVICE_C2H}")
    print(f"  USER: {user_dma_core.DMA_DEVICE_USER}")
    print(f"Setting CLOCK_CYCLE_TIME_NS = {user_dma_core.CLOCK_CYCLE_TIME_NS}")
    print(f"Engines: {args.multi_core}")
    return dict(multi_core=args.multi_core)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL-3B vision + LM on the accelerator (refactor).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python qwen2.5_vl_3b_test_new.py --dev xdma0 --image
  python qwen2.5_vl_3b_test_new.py --dev xdma0 --multi-core 8

numeric checks (FPGA vs host-simulate and HuggingFace) live in
qwen2.5_vl_3b_numeric.py.""")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt. Defaults to the built-in question in LM mode, "
                             "or 'Describe the picture in details.' with --image.")
    parser.add_argument("--image", type=str, nargs="?", const=DEFAULT_IMAGE, default=None,
                        help=f"VLM mode: run the vision encoder and merge image tokens into "
                             f"the prompt. Bare --image uses {os.path.basename(DEFAULT_IMAGE)}.")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Cap on generated tokens (default 256). A low cap is the "
                             "guard against a bad argmax decoding until the context "
                             "fills, which is indistinguishable from a hung board.")
    parser.add_argument("--profile-ctx", type=int, default=MAX_CONTEXT_SIZE,
                        help=f"Context length for the SECOND profiled decode step "
                             f"(default {MAX_CONTEXT_SIZE}, the full context). The first "
                             f"is taken right after prefill, so the pair brackets decode "
                             f"cost from the shortest to the longest KV history.")
    parser.add_argument("--profile", action="store_true",
                        help="Compile the encoder with per-phase HALT checkpoints and "
                             "print a HW-latency breakdown by phase. Read the share "
                             "column: it says which phase is worth sharding. Covers the "
                             "vision encoder, LM prefill, and two decode steps. Works "
                             "with --multi-core.")
    add_engine_args(parser)
    return parser


def clean_dram_4gb(ue=None, chunk_size_bytes: int = 64 * 1024 * 1024) -> None:
    """Fill the whole 32-bit DRAM space with 0xFF, both halves.

    Runs AFTER the software reset and BEFORE the model allocates or uploads
    anything, so every region this run then reads is either something it wrote
    itself or a loud NaN. Cleaning afterwards instead would only tidy up for the
    next process and would leave this run inheriting the last one's memory --
    which is the failure mode worth preventing.

    UnifiedEngine.clear_dram() only walks from DRAM_START_ADDR (0x8000_0000)
    upward, i.e. the 2 GB model map. This model also uses the low 2 GB for the
    per-engine private arena, so the low half is cleared here as well.

    0xFF (not zero) is deliberate, matching clear_dram: it decodes to NaN in
    bf16, so an uninitialised read shows up as NaN and propagates, instead of
    silently reading as a plausible 0.0.

    ``ue`` is optional -- pass the model's engine once it exists, or leave it
    None to open a bare one just for this (as gemma4_e2b's poison_dram does).
    """
    owned = ue is None
    if owned:
        ue = UnifiedEngine()          # bare engine: opens the device, self-tests
    fill = b"\xff" * chunk_size_bytes
    low_bytes = user_dma_core.DRAM_START_ADDR
    print(f"Clearing low DRAM [0x0..0x{low_bytes - 1:X}] "
          f"({low_bytes / 1024**3:.2f} GiB)")
    offset = 0
    while offset < low_bytes:
        n = min(chunk_size_bytes, low_bytes - offset)
        ue.dma_write(user_dma_core.DMA_DEVICE_H2C, offset, fill[:n], n)
        offset += n
        pct = offset / low_bytes
        bar = "\u2588" * int(40 * pct) + "\u2591" * (40 - int(40 * pct))
        print(f"\r  [{bar}] {pct * 100:5.1f}%  "
              f"{offset / 1024**2:.0f}/{low_bytes / 1024**2:.0f} MB", end="", flush=True)
    print()
    ue.clear_dram(chunk_size_bytes=chunk_size_bytes)
    if owned:
        del ue


def summary_filename(args, cores: int) -> str:
    """One .md per run, named for the CLI config that produced it.

    Script stem plus arg tags only -- `_<dev>`, `_image`, `_multi-core_N`,
    `_profile` -- so two runs collide only when they were genuinely the same
    configuration, and the filename says which one it was.
    """
    tags = [args.dev]
    if args.image:
        tags.append("image")
    if cores > 1:
        tags.append(f"multi-core_{cores}")
    if args.profile:
        tags.append("profile")
    return "qwen2.5_vl_3b_test_" + "_".join(tags) + ".md"


def process_image(image_path: str, size: int = 336) -> torch.Tensor:
    """Load an image as a normalized CHW tensor at the encoder's fixed size.

    The encoder program is compiled for exactly num_patches patches, so every
    image is resized to the canonical square first -- same fixed-shape approach
    gemma4_e2b takes.
    """
    from PIL import Image
    import numpy as np
    cfg_path = os.path.join(SCRIPT_DIR, "qwen2.5_vl_3b_config.json")
    with open(cfg_path) as f:
        img_cfg = json.load(f).get("image_processing", {})
    mean = img_cfg.get("normalize_mean", [0.48145466, 0.4578275, 0.40821073])
    std = img_cfg.get("normalize_std", [0.26862954, 0.26130258, 0.27577711])
    img = Image.open(image_path).convert("RGB").resize(
        (size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = (arr - np.array(mean)) / np.array(std)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    engine_kwargs = resolve_engine_config(parser, args)

    # Reset every engine this run will touch BEFORE anything else reaches the
    # hardware. software_reset_test is per-core: a run that died mid-rendezvous
    # leaves cores spin-waiting on a FLAG_CHECK with no timeout, and the next
    # process inherits engines that never accept a program.
    from user_hw_test import software_reset_test
    cores = args.multi_core or 1
    print(f"\n--- Software-resetting {cores} core(s) ---")
    software_reset_test(cores=cores)

    # Establish known DRAM state before the model allocates or uploads anything.
    print(f"\n--- Cleaning DRAM (4 GiB) ---")
    clean_dram_4gb()

    print(f"\n--- Building engine ---")
    ue = Qwen25VL_UnifiedEngine(**engine_kwargs)
    print(ue.describe_dram_map())

    def _run_lm(image_embeddings=None, prefill_tokens=None, positions=None):
        """Phase B: LM weights over the params window, then prefill + decode."""
        print(f"\n--- LM weight init ---")
        timer = time.perf_counter()
        ue.lm_weight_init()
        print(f"  loaded in {time.perf_counter() - timer:.2f}s")
        ue.lm_tensor_init()

        if prefill_tokens is None:
            if args.prompt:
                # tokenize=False then encode: apply_chat_template(tokenize=True)
                # returns a BatchEncoding on this transformers version, not a
                # list of ids, and everything downstream wants plain ints.
                text = ue.tokenizer.apply_chat_template(
                    [{"role": "user", "content": args.prompt}],
                    tokenize=False, add_generation_prompt=True)
                prefill_tokens = ue.tokenizer(text)["input_ids"]
            else:
                prefill_tokens = list(ue._cfg["default_prefill_tokens"])
                print(f"  using the built-in default prompt "
                      f"({len(prefill_tokens)} tokens)")
        ue._prompt_text = ue.tokenizer.decode(prefill_tokens)
        print(f"  prompt: {ue._prompt_text!r}")

        # PREFILL HAS NO LM HEAD. It fills the KV cache and stops; the logits
        # only exist in the decoder. So prefill consumes all but the LAST prompt
        # token, and the decode loop is seeded with that last token -- its step
        # produces the first generated token. Calling get_arg_max_index() after
        # prefill reads a stale register, which is what made the first run
        # decode garbage.
        context, seed = prefill_tokens[:-1], prefill_tokens[-1]

        print(f"\n--- LM compile ---")
        if args.profile:
            # Three programs: a checkpointed prefill, a checkpointed decoder for
            # the two profiled steps, and a PLAIN decoder to advance the context
            # between them -- a checkpointed program only runs segment by
            # segment, so building context with it would cost ~180 host round
            # trips per token.
            ue.compile_prefill(len(context), profile=True)
            ue.compile_decoder(profile=True)
            prof_dec = (ue._decoder_program, list(ue._decoder_checkpoints))
            ue.compile_decoder(profile=False)
        else:
            ue.compile_prefill(len(context))
            ue.compile_decoder()

        print(f"\n--- Prefill ({len(context)} tokens) ---")
        ue.run_prefill(context, image_embeddings=image_embeddings,
                       positions=None if positions is None
                       else positions[:len(context)],
                       profile=args.profile)

        if args.profile:
            print(f"\n--- Profiled decode: 1st token (ctx {ue.seq_len}) ---")
            first_res, tok, aligned_1 = ue.run_decode_step_profiled(seed, *prof_dec)
            ctx_1 = ue.seq_len
            # Jump straight to the target context. --profile measures TIME, not
            # numerics, and a step's cost is set by the KV *length* -- gf_seq_len,
            # gf_aligned_seq_len, the bias width -- not by what the cache holds.
            # Generating there would never arrive anyway: the model hits EOS long
            # before the context fills, and a checkpointed program costs ~180
            # round trips per token. So the position is simply forced.
            ue.seq_len = max(ue.seq_len, args.profile_ctx - 1)
            print(f"  forcing ctx {args.profile_ctx} (timing only; KV rows past "
                  f"the prompt are zeros, which does not change latency)")
            print(f"\n--- Profiled decode: at context (ctx {ue.seq_len}) ---")
            ctx_res, _, aligned_2 = ue.run_decode_step_profiled(tok, *prof_dec)
            profiles.extend([
                ("LM prefill", f"{len(context)} tokens.",
                 getattr(ue, "_prefill_profile", None)),
                ("LM decode — 1st token",
                 f"Context {ctx_1} tokens (aligned {aligned_1}).", first_res),
                ("LM decode — full context",
                 f"Context {ue.seq_len} tokens (aligned {aligned_2}).", ctx_res),
            ])
            return ""

        # "--- Decode run ---" ... "Decoder done in" is one of the stdout formats
        # model_auto_test._extract_decode_text knows how to slice; without a
        # recognised pair it sees no generated text and the check fails even on a
        # perfect run.
        print(f"\n--- Decode run ---")
        _, text = ue.run_decoder(seed, max_new_tokens=args.max_new_tokens)
        # Structured result for harness checks that prefer it to scraping stdout.
        print("TEST_RESULT: " + json.dumps({
            "model": "qwen2.5_vl_3b",
            "decoded_text": text,
            "tokens_generated": ue._decode_n,
            "decode_tok_s": ue._decode_n / ue._decode_wall_s,
            "first_token_tok_s": (1e6 / ue._decode_step_us[0]
                                  if ue._decode_step_us else None),
            "decode_gflops": ue._decode_gflops,
            "prefill_gflops": getattr(ue, "_prefill_gflops", None),
            "vision_gflops": getattr(ue, "_vis_gflops", None),
        }))
        return text

    profiles = []          # (title, note, samples) collected across stages

    def _write_summary():
        # Same tables the .md gets, echoed to the log: a profile run is usually
        # read once, at the terminal, and having to open a file to see the
        # result of the run you just watched is friction for no reason.
        for title, note, results in profiles:
            if results:
                ue.print_profile_table(title, results, note)
        out = os.path.join(SCRIPT_DIR, summary_filename(args, cores))
        try:
            ue.write_run_summary(out, args, profiles=profiles or None)
            print(f"\nWrote summary: {out}")
        except Exception as exc:
            print(f"[warn] failed to write summary: {exc}")

    if not args.image:
        _run_lm()
        _write_summary()
        print(f"\n--- Cleaning DRAM (4 GiB) ---")
        clean_dram_4gb(ue)
        return

    # Resolve a bare filename against the shipped test_samples directory.
    if not os.path.isfile(args.image):
        candidate = os.path.join(os.path.dirname(DEFAULT_IMAGE),
                                 os.path.basename(args.image))
        if os.path.isfile(candidate):
            args.image = candidate
    if not os.path.isfile(args.image):
        raise SystemExit(f"--image: file not found: {args.image!r} "
                         f"(also tried {os.path.dirname(DEFAULT_IMAGE)}/)")

    # Phase A: vision weights at the base of the shared params window.
    print(f"\n--- Vision weight init ---")
    timer = time.perf_counter()
    ue.vision_weight_init()
    print(f"  loaded in {time.perf_counter() - timer:.2f}s")
    print(ue.vision_weight_summary())

    # Host preprocessing must precede tensor init: the RoPE tables and the
    # block-diagonal window mask are built from this image's grid.
    print(f"\n--- Vision host preprocessing ({os.path.basename(args.image)}) ---")
    timer = time.perf_counter()
    ue.prepare_encoder_input(process_image(args.image))
    print(f"  done in {time.perf_counter() - timer:.2f}s")

    print(f"\n--- Vision tensor init ---")
    ue.vision_tensor_init()

    print(f"\n--- Vision encoder compile ---")
    ue.compile_vision_encoder(profile=args.profile)

    print(f"\n--- Vision encoder run ---")
    embeddings = ue.run_vision_encoder(profile=args.profile)

    if args.profile:
        d = ue._vision_dims()
        profiles.append(("Vision encoder",
                         f"{d['VL']} layers, {d['VS']} patches -> "
                         f"{d['NUM_MERGED_TOKENS']} tokens.",
                         getattr(ue, "_vis_profile", None)))
    print(f"\nEncoder output {tuple(embeddings.shape)}: "
          f"mean {embeddings.float().mean():+.4f}, std {embeddings.float().std():.4f}, "
          f"absmax {embeddings.float().abs().max():.4f}")

    # ---- hand off to the LM -------------------------------------------------
    # The encoder's output is already on the host, which is exactly what lets
    # the LM reclaim the params window those weights were computed in.
    ue._ensure_tokenizer()
    prompt = args.prompt or "Describe the picture in details."
    grid = ue._image_grid_thw
    n_img = int(grid.prod()) // (ue._vision_dims()["VMERGE"] ** 2)
    text = ue.tokenizer.apply_chat_template(
        [{"role": "user",
          "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
        tokenize=False, add_generation_prompt=True)
    # The template emits ONE <|image_pad|>; expand it to one per merged token so
    # each gets its own embedding row spliced in by run_prefill.
    text = text.replace("<|image_pad|>", "<|image_pad|>" * n_img)
    tokens = ue.tokenizer(text)["input_ids"]
    print(f"\n[Mode] VLM -- {n_img} image tokens, prompt: {prompt!r}")

    # mRoPE: image tokens carry (t, h, w) from the patch grid, so their position
    # is not their sequence index, and the text after them resumes past the
    # grid's extent rather than at seq_len. HF owns that arithmetic; the delta
    # it returns is what decode must add to every subsequent position.
    ids = torch.tensor([tokens])
    mm_type = torch.zeros_like(ids)
    mm_type[ids == 151655] = 1
    hf = ue._hf_model.model if hasattr(ue._hf_model, "model") else ue._hf_model
    pos, delta = hf.get_rope_index(ids, mm_type, image_grid_thw=grid)
    positions = pos[:, 0, :].transpose(0, 1).contiguous()          # [seq, 3]
    ue._rope_offset = int(delta.flatten()[0])
    print(f"  mRoPE: positions {tuple(positions.shape)}, decode offset {ue._rope_offset}")

    _run_lm(image_embeddings=embeddings, prefill_tokens=tokens, positions=positions)
    _write_summary()

    print(f"\n--- Cleaning DRAM (4 GiB) ---")
    clean_dram_4gb(ue)
    print("\nVLM run complete.")


if __name__ == "__main__":
    main()
