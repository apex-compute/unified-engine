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

DEFAULT_IMAGE = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "..", "test_samples", "yosemite.jpg"))

# Static context capacity. Tensor sizing and decoder bounds follow this one
# constant; it must stay 64-aligned.
MAX_CONTEXT_SIZE = 4096
MIN_CONTEXT_SIZE = 512

# Engine ceiling for this model's private map: 8 cores x 256 MiB fills the low
# 2 GB exactly, which is the interleave granularity the DRAM controller wants.
MAX_ENGINES = 8


class Qwen25VL_UnifiedEngine(Qwen25VLVisionMixin, UnifiedEngine):
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
        #   TENSOR  acts/KV  : 0xF100_0000 - 0xFF00_0000  ( 224 MiB)
        #   ISA     programs : 0xFF00_0000 - 0x1_0000_0000 (  16 MiB)
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
        self.ISA_BASE = 0xFF000000
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
                        help="Text prompt. Default is the built-in test question.")
    parser.add_argument("--image", type=str, nargs="?", const=DEFAULT_IMAGE, default=None,
                        help=f"VLM mode: run the vision encoder and merge image tokens into "
                             f"the prompt. Bare --image uses {os.path.basename(DEFAULT_IMAGE)}.")
    parser.add_argument("--profile", action="store_true",
                        help="Compile the encoder with per-phase HALT checkpoints and "
                             "print a HW-latency breakdown by phase. Read the share "
                             "column: it says which phase is worth sharding. Works with "
                             "--multi-core (checkpoints sit outside sharded regions).")
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

    if not args.image:
        # Weight phase only -- there is nothing for the encoder to run on.
        print(f"\n--- Vision weight init ---")
        timer = time.perf_counter()
        ue.vision_weight_init()
        print(f"  loaded in {time.perf_counter() - timer:.2f}s")
        print(ue.vision_weight_summary())
        print("\nNo --image given, so the encoder was not run. The LM path is "
              "not wired up yet.")
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
        name = (f"qwen2.5_vl_3b_vision_profile_{args.dev}"
                f"{'_multi-core_%d' % cores if cores > 1 else ''}.md")
        out = os.path.join(SCRIPT_DIR, name)
        try:
            ue.write_vision_profile_summary(out, args)
            print(f"\nWrote profile summary: {out}")
        except Exception as exc:
            print(f"[warn] failed to write profile summary: {exc}")
    print(f"\nEncoder output {tuple(embeddings.shape)}: "
          f"mean {embeddings.float().mean():+.4f}, std {embeddings.float().std():.4f}, "
          f"absmax {embeddings.float().abs().max():.4f}")
    print("\nVision path complete. The LM path is not wired up yet.")


if __name__ == "__main__":
    main()
