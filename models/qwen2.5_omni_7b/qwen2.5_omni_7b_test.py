#!/usr/bin/env python3
"""Qwen2.5-Omni-7B Thinker on the eight-engine, 8-GiB Alveo map.

This entry point implements text, image, and audio understanding with text
generation. For --speak, host Talker generation overlaps FPGA text decode;
Token2Wav then renders the completed codec stream. Vision, audio, and LM weights time-share one params window; encoder
outputs are staged through the host strictly for DMA/layout before the next
stage reclaims that window. No learned arithmetic executes there.
"""

from __future__ import annotations

import argparse
import builtins
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# This entry point is an FPGA runner, even when the activated PyTorch wheel was
# built with CUDA.  Hide every common discrete-accelerator backend before the
# first torch import.  If torch was imported and initialized by an embedding
# process before this module, the post-import check below fails closed instead
# of silently running any tensor operation on that accelerator.
for _visibility_var in (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
):
    os.environ[_visibility_var] = ""

import numpy as np
import torch


def _reject_visible_torch_accelerators() -> None:
    detected = []
    for name in ("cuda", "xpu", "mtia"):
        backend = getattr(torch, name, None)
        available = getattr(backend, "is_available", None)
        if available is not None and available():
            detected.append(f"{name} visible")
        checker = getattr(backend, "is_initialized", None)
        if checker is not None and checker():
            detected.append(f"{name} initialized")
    accelerator = getattr(torch, "accelerator", None)
    available = getattr(accelerator, "is_available", lambda: False)()
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(mps is not None and mps.is_available())
    if available:
        detected.append("torch.accelerator visible")
    if mps_available:
        detected.append("mps visible")
    if detected:
        details = ", ".join(dict.fromkeys(detected))
        raise RuntimeError(
            f"Qwen2.5-Omni requires CPU + FPGA execution, but PyTorch reports {details}. "
            "Launch this entry point in a fresh process; GPU "
            "visibility is disabled before torch import."
        )


_reject_visible_torch_accelerators()

import user_dma_core
from multi_engine_shard import (MultiEngineScheduler, require_multicore_dram,
                                tiled_window_bases)
from user_dma_core import UnifiedEngine, set_dma_device


# Capturing 28 decoder layers is intentionally quiet.  Progress messages use
# _loud(), which retains the original print function.
_ORIGINAL_PRINT = builtins.print
_SILENT_MODE = False


def _quiet_print(*args, **kwargs):
    if not _SILENT_MODE:
        _ORIGINAL_PRINT(*args, **kwargs)


builtins.print = _quiet_print


def _load_sibling(module_name: str, filename: str):
    path = os.path.join(SCRIPT_DIR, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_lm_mod = _load_sibling("qwen2_5_omni_7b_lm", "qwen2.5_omni_7b_lm.py")
_vision_mod = _load_sibling(
    "qwen2_5_omni_7b_vision", "qwen2.5_omni_7b_vision.py"
)
_audio_mod = _load_sibling("qwen2_5_omni_7b_audio", "qwen2.5_omni_7b_audio.py")
_model_flops = _load_sibling(
    "qwen2_5_omni_7b_model_flops", "qwen2.5_omni_7b_model_flops.py"
)
_position_mod = _load_sibling(
    "qwen2_5_omni_7b_positions", "qwen2.5_omni_7b_positions.py"
)
_weight_mod = _load_sibling(
    "qwen2_5_omni_7b_weights", "qwen2.5_omni_7b_weights.py"
)
_program_mod = _load_sibling(
    "qwen2_5_omni_7b_programs", "qwen2.5_omni_7b_programs.py"
)
_layout_mod = _load_sibling(
    "qwen2_5_omni_7b_layout", "qwen2.5_omni_7b_layout.py"
)
_state_mod = _load_sibling(
    "qwen2_5_omni_7b_state", "qwen2.5_omni_7b_state.py"
)
WindowedLayout = _layout_mod.WindowedLayout
FlatLayout = _layout_mod.FlatLayout
OMNI_WINDOW_BYTES = _layout_mod.OMNI_WINDOW_BYTES
FPGA_SPEECH_CONTEXT_SIZE = _layout_mod.FPGA_SPEECH_CONTEXT_SIZE

Qwen25OmniLMMixin = _lm_mod.Qwen25OmniLMMixin
Qwen25OmniVisionMixin = _vision_mod.Qwen25OmniVisionMixin
Qwen25OmniAudioMixin = _audio_mod.Qwen25OmniAudioMixin
build_multimodal_positions = _position_mod.build_multimodal_positions
ProgramBundle = _program_mod.ProgramBundle

# Speech output is optional: the Talker/Token2Wav weights live in the HF
# checkpoint, not in params.bin, so a tree without them must still run text.
try:
    _talker_mod = _load_sibling("qwen2_5_omni_7b_talker", "qwen2.5_omni_7b_talker.py")
except Exception:                                    # pragma: no cover
    _talker_mod = None
try:
    _talker_fpga_mod = _load_sibling(
        "qwen2_5_omni_7b_talker_fpga", "qwen2.5_omni_7b_talker_fpga.py")
except Exception:                                    # pragma: no cover
    _talker_fpga_mod = None


# FPGA Token2Wav runs on all eight engines, in bf16, after Thinker and Talker are done
# with the board. Beyond this many codec tokens its BigVGAN activations (rows grow 240x
# the mel frames) no longer fit the shared activation pool.
T2W_FPGA_MAX_CODES = 300

REQUIRED_ENGINES = _layout_mod.REQUIRED_ENGINES
REQUIRED_DRAM_GIB = _layout_mod.REQUIRED_DRAM_GIB
# CONTEXT AND PREFILL ARE ONE BUDGET. Prefill and decode read the same KV
# cache, so MAX_CONTEXT_SIZE bounds both: a prompt (vision soft tokens + text)
# may fill it, and generation continues inside it.
#
# Allocate for 8192 prefill rows and 8192 KV positions. Getting here (up from
# an earlier 4096 ceiling) needed three private-vs-shared DRAM layout changes,
# all in _base_lm_tensor_init (qwen2.5_omni_7b_lm.py) and the
# OMNI_PRIVATE_TENSOR_BYTES/OMNI_PRIVATE_RESERVE_BYTES budgets below:
#   1. The per-engine head-sharded attention scratch is
#      unified_attention_core's own fixed [aligned_seq_len, aligned_seq_len]
#      score buffer -- quadratic in context, ~132-188 MiB/engine at 8192 (was
#      34-62 MiB at 4096) -- a hardware-kernel floor, not something software
#      tiling can shrink. OMNI_PRIVATE_TENSOR_BYTES grew 64 -> 264 MiB/engine
#      to fit it.
#   2. LM_MLP_GATE/UP (296 MiB each at 8192) no longer fit the shared band
#      that growth left behind, so they became private per-engine
#      [P, LANE] lanes (LM_MLP_GATE_PER_ENGINE/LM_MLP_UP_PER_ENGINE) instead
#      of one shared [P, MLP] plane -- same reasoning as the down-projection
#      partials already were. LM_MLP_DOWN (the reduce_add destination) can no
#      longer alias UP's memory, so it gets its own small dedicated shared
#      allocation.
#   3. LM_K_CACHE/LM_V_CACHE (224 MiB each at 8192) hit the same shared-band
#      limit; split into one shared allocation PER LAYER (LM_K_CACHE_PER_LAYER/
#      LM_V_CACHE_PER_LAYER, ~8 MiB each) rather than one contiguous span --
#      split by layer, not by kv_head, so the per-layer grouped multi-head
#      prefill store (bf16_permute_dram_core) stays exactly one call, with
#      zero extra DMA-call cost.
# None of this changes per-layer kernel-call counts, DMA volume, or compute --
# it is purely an addressing/allocation change. Verified correct (coherent,
# fact-checked decode output) at 4096, 4352, 6144 and 8192 on real hardware.
MAX_CONTEXT_SIZE = 8192
# Prefill and decode execute whole 64-row attention tiles. Inputs remain bounded
# by the logical context, while tensors and the KV cache carry the padded tile.
PREFILL_INPUT_TOKEN_LIMIT = MAX_CONTEXT_SIZE
# Allocation bound only. Prefill runs ceil(seq_len/64)*64 rows for the ACTUAL
# prompt -- a 29-token prompt runs 64 rows -- so this sizes the [P, *] planes
# for the longest prompt the context allows, it does not force that shape.
PREFILL_MAX_SEQ_LEN = ((MAX_CONTEXT_SIZE + 63) // 64) * 64

# SELECTABLE VISION INPUT RESOLUTION. The encoder is carved for ONE patch
# count -- its tensors and attention bias are sized for it -- so the resolution
# is chosen before the engine is built, not per image. Each entry is
# self-consistent: (image_size / patch_size)^2 == num_patches, and
# num_patches / spatial_merge_size^2 == num_merged_tokens.
#
# The soft-token count is what reaches the LM and is charged against the
# context, so "medium" spends 1024 of the 2500-token budget on one image.
VISION_RESOLUTIONS = {
    "small":  {"image_size": 336, "num_patches":  576, "num_merged_tokens":  144},
    "medium": {"image_size": 896, "num_patches": 4096, "num_merged_tokens": 1024},
}
DEFAULT_VISION_RES = "small"
VISION_MAX_SOFT_TOKENS = max(
    r["num_merged_tokens"] for r in VISION_RESOLUTIONS.values()
)


# ==========================================================================
# PREFILL-LENGTH FITTING
# ==========================================================================
# --target-prefill-tokens GROWS THE TEXT PROMPT until the assembled prefill
# reaches an exact token count, because the media (if any) contribute a fixed
# token count and only the text is free. Fixed workload shapes (voice-command,
# single-camera, multi-camera, ...) are the CALLER's business -- see
# benchmark.py, which drives this flag with its own preset table -- this
# script only knows how to hit a number.
DEFAULT_PROMPT_BASE = "Respond to the request in detail."
# What the bare --dummy_prompt flag reads; any other file may be named
# explicitly, which is how the longer saved prompts are replayed.
DEFAULT_DUMMY_PROMPT = "dummy_prompt.md"

# Filler for the fitted prompt. It has to be REAL INSTRUCTION TEXT, not
# padding: the point of a long prefill is to measure the shape the model
# actually runs, and a prompt of repeated nonsense changes what attention does
# with it. These sentences are cycled and then trimmed to the exact token count
# the target needs.
_FIT_FILLER_SENTENCES = (
    "Address every part of the request before you stop, and do not end the "
    "answer while any part of it remains uncovered.",
    "Make the structure of the answer explicit, so the boundary between its "
    "parts is easy to locate at a glance.",
    "Keep each observation to a single sentence, so the shape of the response "
    "stays easy to follow from start to finish.",
    "Prefer concrete detail over general impression, and name what is actually "
    "present rather than what would usually be present.",
    "Stay grounded in the material you were given rather than in background "
    "knowledge about material of this kind.",
    "Use plain language throughout, and choose a concrete noun wherever an "
    "evaluative adjective would otherwise go.",
    "Do not repeat a point you have already made, and do not pad the answer "
    "once its subject has been covered.",
    "Say so briefly if something is genuinely unclear, instead of guessing at "
    "content that is not actually there.",
    "Work through the material methodically rather than compressing it into a "
    "single summarising line.",
    "Finish only once every part of the request has been addressed, and not "
    "before that point.",
)


def _filler_words(count: int) -> str:
    """``count`` words, cycling the filler sentences in order."""
    words: list[str] = []
    i = 0
    while len(words) < count:
        words.extend(_FIT_FILLER_SENTENCES[i % len(_FIT_FILLER_SENTENCES)].split())
        i += 1
    return " ".join(words[:count])


def _fit_prompt_to_prefill(tokenizer, render_len, target: int, base: str) -> str:
    """Build a prompt whose rendered prefill length is ``target`` tokens.

    ``render_len(prompt)`` returns the length of the FULL assembled sequence --
    chat scaffolding plus expanded media placeholders plus the prompt -- which
    is the number being targeted. The media contribution is fixed, so the
    prompt is the only free variable and the length is monotone in the word
    count: binary-search the words, then correct against a real render, because
    a token count measured on the prompt alone can differ by a token or two
    from the same text inside the template.
    """
    base_total = render_len(base)
    if base_total >= target:
        return base
    base_prompt_tokens = len(tokenizer(base).input_ids)
    fixed = base_total - base_prompt_tokens        # media + scaffolding
    want = target - fixed                          # prompt tokens needed

    def prompt_for(words: int) -> str:
        return f"{base} {_filler_words(words)}"

    lo, hi = 0, 16
    while len(tokenizer(prompt_for(hi)).input_ids) < want:
        hi *= 2
        if hi > 8192:
            raise RuntimeError(f"cannot reach {target} prefill tokens")
    while lo < hi:                                 # smallest words >= want
        mid = (lo + hi) // 2
        if len(tokenizer(prompt_for(mid)).input_ids) < want:
            lo = mid + 1
        else:
            hi = mid
    words = lo

    # Correct against the real render; walk down so the target is never exceeded.
    for _ in range(64):
        total = render_len(prompt_for(words))
        if total == target or (total < target and words == 0):
            break
        words += 1 if total < target else -1
        if words < 0:
            words = 0
            break
    return prompt_for(words)


def apply_vision_resolution(cfg: dict, name: str) -> dict:
    """Stamp one VISION_RESOLUTIONS entry into a loaded config, in place.

    params.bin does NOT depend on these -- the encoder is a transformer over a
    patch sequence, only patch_embed.proj is patch-shaped and it is per-patch,
    and position information is computed mRoPE rather than a learned table.
    qwen2.5_omni_7b_weights._VISION_RUNTIME_KEYS excludes them from the weight
    fingerprint for exactly that reason, so switching resolution never forces a
    re-quantization.
    """
    try:
        preset = VISION_RESOLUTIONS[name]
    except KeyError:
        raise ValueError(
            f"unknown vision resolution {name!r}; choose from "
            f"{sorted(VISION_RESOLUTIONS)}"
        ) from None
    vision = cfg["vision"]
    side = preset["image_size"] // int(vision["patch_size"])
    merge = int(vision["spatial_merge_size"]) ** 2
    if side * side != preset["num_patches"]:
        raise AssertionError(f"vision preset {name!r} patch count is inconsistent")
    if preset["num_patches"] // merge != preset["num_merged_tokens"]:
        raise AssertionError(f"vision preset {name!r} soft-token count is inconsistent")
    vision.update(preset)
    return cfg

# The DRAM layout (window geometry, budgets, arena construction, phase marks)
# lives in qwen2.5_omni_7b_layout.py; the engine only asks it for space.

# The build ID read from UE_FPGA_VERSION is recorded and used to pick the flag
# protocol, but it is not checked against an allowlist: any image that carries
# the arithmetic ISA Omni needs is allowed to run. The E7 image predates
# CHECK_CLEAR, so its runner uses host-separated one-shot flag rounds.
LEGACY_HOST_SEGMENTED_BUILD = 0xE7AC2CAF
ENGINE_BASE_STRIDE = 0x00010000
RESET_PROBE_ISA_ADDR = 0x1FA000000

DEFAULT_IMAGE = os.path.normpath(
    os.path.join(PROJECT_ROOT, "test_samples", "yosemite.jpg")
)
DEFAULT_AUDIO = os.path.normpath(
    os.path.join(PROJECT_ROOT, "test_samples", "apex.wav")
)
RUN_LOCK_PATH = "/tmp/apexcompute-qwen2.5-omni-7b.lock"
DEFAULT_SPEAKER = "Chelsie"
SPEECH_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)

# Every file that can change captured Omni instructions is bound into the
# programs artifact identity.  Paths are explicit so generated checkpoints,
# caches, logs, and program images can never enter the fingerprint by accident.
_PROGRAM_CODE_FILES = (
    os.path.join(SCRIPT_DIR, "qwen2.5_omni_7b_test.py"),
    os.path.join(SCRIPT_DIR, "qwen2.5_omni_7b_lm.py"),
    os.path.join(SCRIPT_DIR, "qwen2.5_omni_7b_vision.py"),
    os.path.join(SCRIPT_DIR, "qwen2.5_omni_7b_audio.py"),
    os.path.join(SCRIPT_DIR, "qwen2.5_omni_7b_positions.py"),
    os.path.join(SCRIPT_DIR, "qwen2.5_omni_7b_programs.py"),
    os.path.join(SCRIPT_DIR, "qwen2.5_omni_7b_layout.py"),
    os.path.join(SCRIPT_DIR, "qwen2.5_omni_7b_state.py"),
    os.path.join(SCRIPT_DIR, "qwen2.5_omni_7b_weights.py"),
    os.path.join(SCRIPT_DIR, "qwen2.5_omni_7b_talker_fpga.py"),
    os.path.join(SCRIPT_DIR, "qwen2.5_omni_7b_token2wav_fpga.py"),
    os.path.join(SCRIPT_DIR, "qwen2.5_omni_7b_t2w_cores.py"),
    os.path.join(SCRIPT_DIR, "qwen2.5_omni_7b_config.json"),
    os.path.join(PROJECT_ROOT, "andromeda_hw_info.py"),
    os.path.join(PROJECT_ROOT, "multi_engine_shard.py"),
    os.path.join(PROJECT_ROOT, "user_dma_core.py"),
)


def _acquire_run_lock() -> int:
    """Hold one process-wide lock across params, programs.bin, and FPGA use."""
    # The lock guards one shared FPGA, so every user must take the same file.
    # Whoever creates it first owns it, and their umask can leave it
    # unwritable for the next user; flock() only needs an open descriptor, so
    # fall back to read-only when write access is denied.
    try:
        fd = os.open(RUN_LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o666)
    except PermissionError:
        fd = os.open(RUN_LOCK_PATH, os.O_RDONLY)
    else:
        try:
            os.fchmod(fd, 0o666)
        except OSError:
            pass
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise SystemExit(
            "another Qwen2.5-Omni process is already converting artifacts or "
            "using the FPGA; wait for it to finish before retrying"
        ) from None
    return fd


@contextmanager
def _exclusive_run_lock():
    fd = _acquire_run_lock()
    try:
        yield
    finally:
        os.close(fd)


class Qwen25OmniUnifiedEngine(
    Qwen25OmniLMMixin,
    Qwen25OmniVisionMixin,
    Qwen25OmniAudioMixin,
    UnifiedEngine,
):
    """Concrete Thinker engine and its fixed eight-engine, 8-GiB memory map."""

    def __init__(self, script_dir: str | None = None, multi_core: int = 8,
                 fpga_build: int | None = None,
                 vision_res: str = DEFAULT_VISION_RES,
                 fpga_talker: bool = False, layout: str = "windowed",
                 flat_plan=None):
        if multi_core != REQUIRED_ENGINES:
            raise ValueError(
                f"Qwen2.5-Omni-7B requires exactly {REQUIRED_ENGINES} engines, "
                f"got {multi_core}"
            )
        reported_cores = user_dma_core.ANDROMEDA_CORE_COUNT
        if reported_cores is not None and reported_cores < REQUIRED_ENGINES:
            raise ValueError(
                f"the board image must report at least {REQUIRED_ENGINES} engines; "
                f"HW_INFO reports {reported_cores}"
            )
        self.multi_core = multi_core
        self.fpga_talker = bool(fpga_talker)
        self.speak_as: str | None = None      # set before weight/tensor init
        self.fpga_build = None if fpga_build is None else int(fpga_build)
        self._multi_core_schedulers: dict[str, MultiEngineScheduler] = {}
        self._worker_isa_used: dict[int, dict[str, int]] = {}
        # Resident-program state (layouts with resident_programs): one worker
        # pool shared by every stage's scheduler, the program images already in
        # DRAM, and what each stage's tensors looked like when it was compiled.
        self._worker_pool: list | None = None
        self._resident_stages: set[str] = set()
        self._resident_images: dict[int, dict[int, set]] = {}
        self._resident_hashes: dict = {}
        self._stage_tensor_state: dict[str, dict] = {}
        self._raw_stages: dict[str, dict] = {}
        # --run_from_bin: what the compilers left in Python objects, per stage,
        # and the stage images as read from programs.bin.
        self._compile_state: dict[str, dict] = {}
        self._bin_stage_state: dict[str, tuple] = {}
        self._params_regions: dict[str, dict[str, Any]] | None = None
        self._program_bundle: ProgramBundle | None = None
        self._packaged_program_stages: set[str] = set()
        self._loaded_program_stages: set[str] = set()
        self._executed_program_stages: set[str] = set()
        self._program_stage_profiles: dict[str, bool] = {}
        self._runtime_processor_dir: str | None = None
        # Audit trail of FPGA-selected token IDs.  run_decoder resets it per
        # request, but _decode_token also runs from the profiled single-step
        # API, which --profile drives WITHOUT ever entering run_decoder -- so
        # the list has to exist from construction.
        self._fpga_decode_token_ids: list[int] = []

        # The DRAM map (see qwen2.5_omni_7b_layout.py): the board gate, window
        # geometry, private reserve and phase marks all live there.
        if layout == "flat":
            self.layout = FlatLayout(multi_core, flat_plan)
        else:
            self.layout = WindowedLayout(multi_core, speech=self.fpga_talker)
        self._private_tensor_bytes = self.layout.private_tensor_bytes
        self._private_reserve_bytes = self.layout.private_reserve_bytes
        self.DRAM_END = self.layout.device_end
        self.WINDOW_BYTES = self.layout.window_bytes
        # ISA lives inside the windows. Engine 0's slice is the master area;
        # every worker's is layout.isa_base(i).
        self.ISA_BASE = self.layout.isa_base(0)
        self.MASTER_ISA_RESERVE = self.layout.isa_bytes
        self.MASTER_ISA_LIMIT = self.layout.isa_limit(0)
        self.WORKER_ISA_STRIDE = OMNI_WINDOW_BYTES

        # TENSORS ARE CARVED PER BUFFER, NOT FROM ONE EXTENT. Only an individual
        # buffer has to be contiguous -- the KV cache, an [M, N] activation
        # plane. The largest at the current allocation is the 140 MiB TP
        # down-output scratch, which is also overlaid by shorter-lived Q/K/V,
        # attention-result and norm tensors. Carving physical buffers separately
        # spreads them over the shared pool instead of requiring one impossible
        # contiguous tensor extent.
        self.TENSOR_BASE = 0
        self.TENSOR_LIMIT = self.layout.tensor_capacity
        self._tensor_staged = 0
        # No dedicated extents left to reserve -- lm_head no longer stages an
        # unsharded device blob (see OMNI_PRIVATE_RESERVE_BYTES's comment).
        # _reserved_extent_for/reset_params_dram_addr keep working unchanged
        # against an empty dict; a future genuinely-unscatterable object would
        # add its own entry here.
        self._reserved_extents: dict[str, list[int]] = {}

        # PARAMS IS NO LONGER A WINDOW, IT IS AN ACCOUNTING ORIGIN. Weight
        # sections are placed individually by alloc_shared, so there is no
        # params cursor to walk; PARAMS_BASE/_LIMIT keep the bookkeeping that
        # every caller already does ("bytes staged so far", "capacity left")
        # working against the pool instead of against a contiguous range.
        self.PARAMS_BASE = 0
        self.PARAMS_LIMIT = self.layout.shared_capacity
        self.VISION_WEIGHT_BASE = self.PARAMS_BASE
        self._params_staged = 0

        super().__init__(
            BASE_ADDR=user_dma_core.UE_0_BASE_ADDR,
            params_dram_base=self.PARAMS_BASE,
            program_dram_base=self.ISA_BASE,
            tensor_dram_base=self.TENSOR_BASE,
        )

        self.script_dir = script_dir or SCRIPT_DIR
        self._cfg = apply_vision_resolution(
            self.load_config(script_dir=self.script_dir), vision_res)
        self.vision_res = vision_res
        self._validate_config_and_map()

        fi = self._cfg["file_info"]
        model = self._cfg["model"]
        self.vector_length = int(fi["hidden_size"])
        self.head_dim = int(fi["head_dim"])
        self.actual_head_dim = int(fi["actual_head_dim"])
        self.num_kv_heads = int(fi["num_kv_heads"])
        self.group_size = int(fi["group_size"])
        self.mlp_elements = int(fi["mlp_elements"])
        self.bytes_per_element = int(fi["bytes_per_element"])
        self.LAYER_SIZE = int(fi["num_layers"])
        self.EMBEDDING_ELEMENTS = int(fi["embedding_vocab"])
        self.MAX_CONTEXT_SIZE = getattr(
            self.layout, "max_context",
            FPGA_SPEECH_CONTEXT_SIZE if fpga_talker else MAX_CONTEXT_SIZE)
        self.PREFILL_MAX_SEQ_LEN = self.MAX_CONTEXT_SIZE
        model["max_context_size"] = self.MAX_CONTEXT_SIZE
        model["prefill_max_seq_len"] = self.PREFILL_MAX_SEQ_LEN

        fixed = self._cfg["fixed_isa_regs"]
        self.TMP_REG = int(fixed["TMP_REG"])
        self.gf_seq_len = int(fixed["GF_SEQ_LEN_REG"])
        self.gf_q_seq_len = int(fixed["GF_Q_SEQ_LEN_REG"])
        self.gf_aligned_seq_len = int(fixed["GF_ALIGNED_SEQ_LEN_REG"])
        self._isa_reg_base = max(int(v) for v in fixed.values()) + 1
        self._isa_reg_counter = self._isa_reg_base
        self.gf_one = self.alloc_isa_reg()
        self._isa_reg_base = self._isa_reg_counter

        self._end_of_turn_token_id = int(model["end_of_turn_token_id"])
        self.causal_mask_upper = False

    def _export_prefill_hidden(self) -> bool:
        """Speech needs the Thinker's final hidden for every prompt row."""
        return self.speak_as is not None

    # -- params allocation against the shared pool ---------------------------
    #
    # The base class walks one cursor through a contiguous params window. There
    # is no such window in this map, so these four methods redirect the same API
    # at PrivateArena's shared pool: each section is placed individually in
    # whichever window has room, and the "cursor" becomes a byte counter so that
    # every caller's `end - PARAMS_BASE` accounting still reports what it always
    # reported -- bytes staged.

    def _alloc_per_engine_attn_scratch(self, engine_idx: int, size_bytes: int) -> int:
        """From the shared pool, pinned to the engine's own window.

        At 4096 patches this is 35.65 MiB per engine -- far past the per-core
        tensor slice, and growing the slice to fit would shrink every window's
        gap below the 260 MiB the untied head needs contiguously. The pool has
        the room, and pinning engine_idx keeps each engine reading its own
        window exactly as the private slice did. It is released with the rest
        of the vision tensors.
        """
        return self.layout.alloc_shared_up(
            size_bytes, f"vis.attn_scratch.e{engine_idx}", engine_idx=engine_idx)

    def _reserved_extent_for(self, label: str | None):
        """The dedicated extent a label is served from, or None for the pool."""
        if not label:
            return None
        for name, extent in self._reserved_extents.items():
            if label.startswith(name):
                return extent
        return None

    def allocate_params_dram(self, size_bytes: int, label: str | None = None,
                             align_bytes: int = 64) -> int:
        align = max(align_bytes, 128)
        extent = self._reserved_extent_for(label)
        if extent is not None:
            base, limit, cursor = extent
            addr = (cursor + align - 1) & ~(align - 1)
            if addr + size_bytes > limit:
                raise MemoryError(
                    f"{label}: needs 0x{addr + size_bytes:X}, past its reserved "
                    f"extent ending at 0x{limit:X} "
                    f"({(limit - base) / 2**20:.0f} MiB)")
            extent[2] = addr + size_bytes
            self._params_staged += size_bytes
            self._dram_addresses[label] = addr
            return addr
        # 128 B, not the caller's 64: a shared section can land anywhere in a
        # window, and the SRAM row is the alignment every DMA base owes.
        addr = self.layout.alloc_shared(
            size_bytes, label or "params", align=align)
        self._params_staged += size_bytes
        if label is not None:
            self._dram_addresses[label] = addr
        return addr

    def get_params_dram_addr(self) -> int:
        """Bytes staged into the pool, as an address in the PARAMS_BASE origin."""
        return self.PARAMS_BASE + self._params_staged

    def get_params_dram_usage(self) -> int:
        return self._params_staged

    def reset_params_dram_addr(self) -> None:
        """Hand the previous phase's weights back to the windows.

        Vision, audio and the shared LM weights time-share the pool. The base
        class reclaims by rewinding one cursor; here it is a release back to the
        mark taken when the phase began. The caller must already have
        invalidated its cached addresses -- the next DMA overwrites these bytes.
        """
        reclaimed = self.layout.release_weight_phase()
        for extent in self._reserved_extents.values():
            extent[2] = extent[0]
        self._params_staged = 0
        if reclaimed:
            self._loud(f"  [map] reclaimed {reclaimed / 2**20:.1f} MiB of shared "
                       f"pool from the previous phase")

    # -- tensor allocation against the shared pool ---------------------------

    def allocate_tensor_dram(self, size_bytes: int, label: str | None = None,
                             align_bytes: int = 64) -> int:
        addr = self.layout.alloc_shared_up(
            size_bytes, label or "tensor", align=max(align_bytes, 128))
        self._tensor_staged += size_bytes
        if label is not None:
            self._dram_addresses[label] = addr
        return addr

    def get_tensor_dram_addr(self) -> int:
        """Bytes staged, in the TENSOR_BASE origin -- see allocate_params_dram."""
        return self.TENSOR_BASE + self._tensor_staged

    def get_tensor_dram_usage(self) -> int:
        return self._tensor_staged

    def reset_tensor_dram_addr(self) -> None:
        """Release every tensor carved since the phase mark back to the pool.

        Vision, audio and the LM each re-carve the whole tensor set: vision runs
        to completion and its output is on the host, so nothing it allocated
        outlives it. With one extent that was a cursor rewind; here it is a
        release, and the bytes genuinely return to the pool for the next phase.
        """
        self.layout.release_tensor_phase()
        self._tensor_staged = 0

    def tensor_phase_mark(self):
        """A rollback point for a partially-built tensor set.

        Covers the private per-engine cursors as well as the shared pool: a
        partially built set can hold both, so a rollback that reclaimed only the
        shared half would leak private slice space on every retry.
        """
        return (self.layout.tensor_phase_mark(), self._tensor_staged)

    def tensor_phase_restore(self, mark) -> None:
        layout_mark, staged = mark
        self.layout.tensor_phase_restore(layout_mark)
        self._tensor_staged = staged

    def dma_to_accelerator_memory(
        self, dma_address: int, data: torch.Tensor
    ) -> None:
        """Upload BF16 data and reject partial H2C transfers.

        The base helper historically ignored ``dma_write``'s byte count.  For
        this multi-gigabyte, phase-shared model, launching after a short input,
        mask, RoPE, or cache-clear write would consume stale DRAM and could
        silently produce plausible but incorrect output.
        """
        if data.dtype != torch.bfloat16:
            raise AssertionError("Data must be in bf16 format")
        expected = data.numel() * 2
        written = self.dma_write(
            user_dma_core.DMA_DEVICE_H2C, dma_address, data, expected
        )
        if written != expected:
            raise IOError(
                f"BF16 H2C DMA at 0x{dma_address:X} wrote "
                f"{written} of {expected} bytes"
            )

    @staticmethod
    def load_config(
        config_path: str | None = None, script_dir: str | None = None
    ) -> dict:
        path = config_path or os.path.join(
            script_dir or SCRIPT_DIR, "qwen2.5_omni_7b_config.json"
        )
        with open(path) as file_obj:
            return json.load(file_obj)

    def _validate_config_and_map(self) -> None:
        self.layout.validate_config(self._cfg["hardware"])
        if self._cfg["file_info"]["hidden_size"] != 3584:
            raise ValueError("this runtime is compiled only for the 3584-wide 7B Thinker")
        if set(self._cfg["precision"]["lm_quantized_projections"]) != {
            "q", "k", "v", "o", "gate", "up", "down"
        }:
            raise ValueError(
                "prefill requires IF4 Q/K/V/O/GATE/UP/DOWN"
            )
        if set(self._cfg["precision"].get("decode_bf16_projections", ())) != {"o"}:
            raise ValueError(
                "the retained params.bin requires its legacy BF16 O region"
            )
        if self._cfg["precision"].get("embedding") != "bf16":
            raise ValueError(
                "Qwen2.5-Omni requires host BF16 token embeddings"
            )

    def _read_params_region(self, name: str) -> dict[str, Any]:
        """Return one validated disk region from the unified params artifact."""
        if self._params_regions is None:
            bin_path = _weight_mod.ensure_params_bin(self.script_dir)
            json_path = bin_path.rsplit(".", 1)[0] + ".json"
            with open(json_path) as file_obj:
                manifest = json.load(file_obj)
            regions = manifest.get("regions") or {}
            file_size = os.path.getsize(bin_path)
            validated: dict[str, dict[str, Any]] = {}
            for region_name, region in regions.items():
                offset, size = int(region["offset"]), int(region["size"])
                if offset < 0 or size < 0 or offset + size > file_size:
                    raise ValueError(
                        f"params region {region_name!r} lies outside {bin_path}"
                    )
                sections = region.get("manifest") or {}
                for section_name, section in sections.items():
                    section_offset = int(section["offset"])
                    section_size = int(section["size"])
                    if (
                        section_offset < 0
                        or section_size < 0
                        or section_offset + section_size > size
                    ):
                        raise ValueError(
                            f"section {region_name}/{section_name} lies outside its region"
                        )
                validated[region_name] = {
                    "bin_path": bin_path,
                    "base_offset": offset,
                    "size": size,
                    "sections": sections,
                }
            self._params_regions = validated
        try:
            return self._params_regions[name]
        except KeyError:
            raise KeyError(
                f"params artifact has no {name!r} region; available: "
                f"{sorted(self._params_regions)}"
            ) from None

    @staticmethod
    def _sha256_path(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as file_obj:
            while chunk := file_obj.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def configure_runtime_artifacts(
        self, params_path: str, processor_dir: str
    ) -> None:
        """Bind program packaging to the exact params, code, and U55 image.

        This is deliberately called only after HW_INFO and every engine's FPGA
        build stamp have been read.  A program captured for a different build,
        AXI width, topology, params generation, source tree, or DRAM map must
        fail validation rather than reach a queue.
        """
        params_path = os.path.realpath(params_path)
        configured_params = os.path.realpath(
            os.path.join(self.script_dir, self._cfg["paths"]["params"])
        )
        if params_path != configured_params:
            raise ValueError(
                f"runtime params path {params_path!r} differs from configured "
                f"artifact {configured_params!r}"
            )
        processor_dir = os.path.realpath(processor_dir)
        if not os.path.isdir(processor_dir):
            raise FileNotFoundError(
                f"validated runtime processor bundle is absent: {processor_dir}"
            )
        if self.fpga_build is None:
            raise RuntimeError(
                "FPGA build must be read from all eight engines before creating "
                "the programs.bin identity"
            )
        if user_dma_core.HW_INFO_RAW is None:
            raise RuntimeError(
                "HW_INFO must be read before creating the programs.bin identity"
            )

        params_json = params_path.rsplit(".", 1)[0] + ".json"
        with open(params_json, "rb") as file_obj:
            params_manifest_bytes = file_obj.read()
        params_manifest = json.loads(params_manifest_bytes)
        code_files: dict[str, str] = {}
        for path in _PROGRAM_CODE_FILES:
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"program identity source file is absent: {path}"
                )
            relative = os.path.relpath(path, PROJECT_ROOT)
            code_files[relative] = self._sha256_path(path)
        code_canonical = json.dumps(
            code_files, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        handshake = (
            "host_segmented"
            if self.fpga_build == LEGACY_HOST_SEGMENTED_BUILD
            else "four_phase"
        )
        identity = {
            "model": "Qwen2.5-Omni-7B-Thinker",
            "program_abi": 1,
            "params": {
                "manifest_sha256": hashlib.sha256(
                    params_manifest_bytes
                ).hexdigest(),
                "schema_version": params_manifest["schema_version"],
                "config_sha256": params_manifest["config_sha256"],
                "generation_id": params_manifest["generation_id"],
                "params_size": params_manifest["params_size"],
                "model_revision": params_manifest["model_revision"],
                "generation_trailer_bytes": params_manifest[
                    "generation_trailer_bytes"
                ],
                "regions_sha256": hashlib.sha256(
                    json.dumps(
                        params_manifest["regions"],
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ).encode("ascii")
                ).hexdigest(),
            },
            "code": {
                "aggregate_sha256": hashlib.sha256(code_canonical).hexdigest(),
                "files": code_files,
            },
            "hardware": {
                "fpga_build": f"0x{self.fpga_build:08X}",
                "hw_info_raw": f"0x{int(user_dma_core.HW_INFO_RAW):08X}",
                "engines": REQUIRED_ENGINES,
                "dram_gib": REQUIRED_DRAM_GIB,
                "axi_data_width_bits": int(user_dma_core.UE_AXI_DATA_WIDTH_BITS),
                "reported_core_count": int(user_dma_core.ANDROMEDA_CORE_COUNT),
                "queue_mode": bool(user_dma_core.QUEUE_MODE_ENABLED),
                "engine_zero_base": int(user_dma_core.UE_0_BASE_ADDR),
                "engine_base_stride": ENGINE_BASE_STRIDE,
                "instruction_size_bytes": int(
                    user_dma_core.INSTRUCTION_SIZE_BYTES
                ),
                "vector_size": int(user_dma_core.UE_VECTOR_SIZE),
                "handshake": handshake,
                "region_rendezvous": "master_worker",
                "barrier_margin_nops": 32,
                "dram_map": self.layout.program_identity(
                    self.PARAMS_BASE, self.PARAMS_LIMIT,
                    self.TENSOR_BASE, self.TENSOR_LIMIT),
            },
        }
        self._program_bundle = ProgramBundle(
            os.path.dirname(params_path), identity,
            stem="programs" if self.layout.kind == "windowed" else "programs_flat"
        )
        self._packaged_program_stages.clear()
        self._loaded_program_stages.clear()
        self._executed_program_stages.clear()
        self._program_stage_profiles.clear()
        self._runtime_processor_dir = processor_dir
        self._loud(
            f"Program artifact: {self._program_bundle.bin_path} "
            "(compile -> atomic store -> validated reload, or loaded as stored with --run_from_bin)"
        )

    def _invalidate_program_stage(self, *stages: str) -> None:
        for stage in stages:
            self._packaged_program_stages.discard(stage)
            self._loaded_program_stages.discard(stage)
            self._executed_program_stages.discard(stage)

    # Capture wrappers invalidate the disk-execution proof before changing any
    # in-memory ISA.  A caller cannot compile a new shape and accidentally run
    # an older section that happened to remain in programs.bin.
    def compile_vision_encoder(self, profile: bool = False) -> int:
        self._invalidate_program_stage("vision")
        result = super().compile_vision_encoder(profile=profile)
        self._program_stage_profiles["vision"] = bool(profile)
        return result

    def compile_audio_encoder(self) -> int:
        self._invalidate_program_stage("audio")
        result = super().compile_audio_encoder()
        self._program_stage_profiles["audio"] = False
        return result

    def compile_prefill(
        self, seq_len: int, layer_size: int | None = None,
        profile: bool = False,
    ) -> int:
        # Decoder addresses are allocated after this exact prefill image.
        self._invalidate_program_stage("prefill", "decode")
        result = super().compile_prefill(
            seq_len, layer_size=layer_size, profile=profile
        )
        self._program_stage_profiles["prefill"] = bool(profile)
        return result

    def compile_decoder(
        self, layer_size: int | None = None, profile: bool = False
    ) -> int:
        self._invalidate_program_stage("decode")
        result = super().compile_decoder(
            layer_size=layer_size, profile=profile
        )
        self._decoder_layers_compiled = (
            int(self._lm_dims()["NL"])
            if layer_size is None
            else int(layer_size)
        )
        self._program_stage_profiles["decode"] = bool(profile)
        return result

    def register_raw_stage(self, stage: str, master: tuple, workers: list,
                           metadata: dict) -> None:
        """Record a stage whose programs are already in DRAM (Talker, Token2Wav).

        Those stages write their programs while compiling, so there is nothing to
        upload; the images are read back from DRAM so the same programs.bin
        generation holds, hashes and verifies them like the others.
        """
        self._raw_stages[stage] = {"master": master, "workers": workers,
                                   "metadata": metadata}

    def read_back(self, engine, addr: int, size: int) -> bytes:
        buf = bytearray(size)
        if engine.dma_read(engine.c2h_device, addr, buf, size) != size:
            raise IOError(f"short program read-back at 0x{addr:X}")
        return bytes(buf)

    def _program_stage_state(self, stage: str):
        raw = self._raw_stages.get(stage)
        if raw is not None:
            return int(raw["master"][0]), bytes(raw["master"][1]), raw["workers"]
        loaded = self._bin_stage_state.get(stage)
        if loaded is not None:
            return loaded
        if stage == "vision":
            return (
                int(self._vis_program_addr),
                bytes(self._vis_program_bytes),
                self._vis_worker_programs,
            )
        if stage == "audio":
            return (
                int(self._audio_program_addr),
                bytes(self._audio_program_bytes),
                self._audio_worker_programs,
            )
        if stage == "prefill":
            base, blob = self._prefill_program
            return int(base), bytes(blob), self._prefill_workers
        if stage == "decode":
            base, blob = self._decoder_program
            return int(base), bytes(blob), self._decoder_workers
        raise ValueError(f"unknown program stage {stage!r}")

    def _program_stage_metadata(self, stage: str) -> dict[str, Any]:
        raw = self._raw_stages.get(stage)
        if raw is not None:
            return {"kind": "resident_at_compile", **raw["metadata"]}
        common: dict[str, Any] = {
            "engines": REQUIRED_ENGINES,
            "profile": bool(self._program_stage_profiles.get(stage, False)),
        }
        scheduler = self._multi_core_schedulers.get(stage)
        if scheduler is not None and scheduler.host_segmented:
            common["host_segment_starts"] = scheduler.host_segment_starts()
        if stage == "vision":
            grid = torch.as_tensor(self._image_grid_thw, dtype=torch.long)
            return {
                **common,
                "grid_thw": grid.tolist(),
                "patch_k": int(self._vis_patch_k),
                "encoder_size": len(self._vis_encoder_program_bytes),
                "patch_size": len(self._vis_patch_program_bytes),
                "patch_program_addr": f"0x{self._vis_patch_program_addr:X}",
                "aligned_sequence_rows": int(self._vis_aligned_S),
                "cu_window_seqlens": list(self._cu_window_seqlens),
                "checkpoints": list(getattr(self, "_vis_checkpoints", ())),
                "total_flops": int(self._vis_total_flops),
            }
        if stage == "audio":
            return {
                **common,
                "feature_lengths": list(self._audio_feature_lens),
                "chunk_feature_lengths": list(self._audio_chunk_feature_lens),
                "aftercnn_lengths": list(self._audio_aftercnn_lens),
                "chunk_aftercnn_lengths": list(
                    self._audio_chunk_aftercnn_lens
                ),
                "output_lengths": list(self._audio_output_lengths),
                "cu_seqlens": torch.as_tensor(
                    self._audio_cu_seqlens, dtype=torch.long
                ).tolist(),
                "conv1_rows": int(self._audio_conv1_input.shape[0]),
                "sequence_rows": int(self._audio_seq_len),
                "aligned_sequence_rows": int(self._audio_aligned_seq_len),
                "output_tokens": int(self._audio_num_tokens),
                "audio_dimensions": self._audio_dims(),
                "input_generation": int(self._audio_input_generation),
                "tensor_generation": int(self._audio_tensor_generation),
                "total_flops": int(self._audio_total_flops),
            }
        if stage == "prefill":
            return {
                **common,
                "logical_sequence_length": int(self._prefill_seq_len),
                "execution_rows": int(
                    self._prefill_execution_rows_compiled
                ),
                "layers": int(self._prefill_layers),
                "output_address": f"0x{int(self.LM_PREFILL_OUT):X}",
                "aligned_execution_rows": (
                    (int(self._prefill_execution_rows_compiled) + 63) // 64
                ) * 64,
                "checkpoints": list(getattr(self, "_prefill_checkpoints", ())),
                "total_flops": int(self._prefill_flops),
            }
        if stage == "decode":
            worker_regs = [
                {str(key): int(value) for key, value in registers.items()}
                for registers in getattr(self, "_decode_attn_worker_regs", ())
            ]
            dims = self._lm_dims()
            if self.DECODE_O_IF4:
                o_layers = [self._decode_shards.get(("o", layer))
                            for layer in range(int(self._decoder_layers_compiled))]
                if (not o_layers or any(weight is None or len(weight.shards) != REQUIRED_ENGINES
                                        or weight.data_type is not user_dma_core.TYPE.IF4
                                        for weight in o_layers)):
                    raise RuntimeError("decode metadata requires eight IF4 O shards per layer")
                columns = int(dims["H"]) // REQUIRED_ENGINES
                expected = [(engine * columns, columns)
                            for engine in range(REQUIRED_ENGINES)]
                for weight in o_layers:
                    if [(int(shard.col_offset), int(shard.cols))
                            for shard in weight.shards] != expected:
                        raise RuntimeError("decode IF4 O shards do not cover output columns")
                o_layout = {
                    "mode": "private_if4_column_shards",
                    "precision": "if4",
                    "engines": REQUIRED_ENGINES,
                    "columns_per_engine": columns,
                    "weight_bytes_per_layer_per_engine": columns * int(dims["H"]) // 2,
                    "scale_bytes_per_layer_per_engine": columns * int(dims["H"]) // 64 * 2,
                    "slot_stride_bytes": int(self.layout.stride),
                }
            else:
                stripes = getattr(self, "_decode_o_stripes", None)
                if stripes is None or len(stripes) != REQUIRED_ENGINES:
                    raise RuntimeError(
                        "decode programs.bin metadata requires eight BF16 O stripes"
                    )
                stripe_bytes = {
                    int(stripe["end"]) - int(stripe["base"])
                    for stripe in stripes
                }
                stripe_cols = {int(stripe["cols"]) for stripe in stripes}
                layer_bytes = {int(stripe["layer_bytes"]) for stripe in stripes}
                if (len(stripe_bytes) != 1 or len(stripe_cols) != 1
                        or len(layer_bytes) != 1):
                    raise RuntimeError("BF16 O stripe geometry is not uniform")
                o_layout = {
                    "mode": "upper_params_column_stripes",
                    "precision": "bf16",
                    "engines": len(stripes),
                    "columns_per_engine": next(iter(stripe_cols)),
                    "layer_bytes_per_engine": next(iter(layer_bytes)),
                    "stripe_bytes_per_engine": next(iter(stripe_bytes)),
                    "slot_stride_bytes": int(self.layout.stride),
                    "extent_end": f"0x{max(int(s['end']) for s in stripes):X}",
                }
            kv_groups = int(dims["KVH"])
            prefill_base, prefill_blob = self._prefill_program
            decode_base, decode_blob = self._decoder_program
            expected_decode_base = (
                (int(prefill_base) + len(prefill_blob) + 63) // 64
            ) * 64
            expected_preamble = (
                (int(decode_base) + len(decode_blob) + 63) // 64
            ) * 64
            if int(decode_base) != expected_decode_base:
                raise RuntimeError(
                    f"decoder base 0x{int(decode_base):X} does not immediately "
                    f"follow prefill at 0x{expected_decode_base:X}"
                )
            if int(self._decoder_preamble) != expected_preamble:
                raise RuntimeError(
                    f"decoder preamble 0x{int(self._decoder_preamble):X} does not "
                    f"immediately follow decoder at 0x{expected_preamble:X}"
                )
            return {
                **common,
                "layers": int(self._decoder_layers_compiled),
                "max_context_size": int(self.MAX_CONTEXT_SIZE),
                "runtime_preamble_addr": f"0x{int(self._decoder_preamble):X}",
                "runtime_preamble_reserve": 512,
                "decode_output_address": f"0x{int(self.LM_DECODE_OUT):X}",
                "total_flops": int(self._decoder_flops),
                "fixed_flops": int(self._decoder_flops_fixed),
                "attention_flops_per_aligned_row": float(
                    self._decoder_attn_per_aligned
                ),
                "attention_worker_registers": worker_regs,
                "checkpoints": list(getattr(self, "_decoder_checkpoints", ())),
                "lm_head_argmax_mode": "sharded_host_reduce",
                "embedding_precision": self._cfg["precision"]["embedding"],
                "decode_bf16_projections": sorted(self._decode_bf16_projections()),
                "decode_attention": {
                    "mode": "one_complete_gqa_group_per_engine",
                    "kv_groups": kv_groups,
                    "heads_per_group": int(dims["G"]),
                    "head_dim": int(dims["AHD"]),
                    "active_engines": list(range(kv_groups)),
                    "idle_handshake_engines": list(
                        range(kv_groups, REQUIRED_ENGINES)
                    ),
                },
                "decode_o_layout": o_layout,
                "prefill_program_base": f"0x{int(prefill_base):X}",
                "prefill_program_size": len(prefill_blob),
                "prefill_program_sha256": hashlib.sha256(
                    prefill_blob
                ).hexdigest(),
                "prefill_sequence_length": int(self._prefill_seq_len),
            }
        raise ValueError(f"unknown program stage {stage!r}")

    def _program_stage_sections(self, stage: str) -> list[dict[str, Any]]:
        """Return the compiled master + seven worker section descriptors."""
        master_addr, master_blob, workers = self._program_stage_state(stage)
        sections: list[dict[str, Any]] = [
            {
                "engine_index": 0,
                "dram_base": master_addr,
                "bytes": master_blob,
            }
        ]
        for engine_index, _worker, address, blob in workers:
            sections.append(
                {
                    "engine_index": int(engine_index),
                    "dram_base": int(address),
                    "bytes": bytes(blob),
                }
            )
        engines = [section["engine_index"] for section in sections]
        if engines != list(range(REQUIRED_ENGINES)):
            raise RuntimeError(
                f"{stage} must package ordered master + workers for engines "
                f"0-7; got {engines}"
            )
        return sections

    def store_program_stages(self, *stages: str) -> None:
        """Publish one or more complete stages in one artifact generation."""
        if self._program_bundle is None:
            raise RuntimeError(
                "configure_runtime_artifacts() must run before packaging ISA"
            )
        if not stages or len(set(stages)) != len(stages):
            raise ValueError("program stage transaction must be non-empty and unique")
        requests = {
            stage: {
                "sections": self._program_stage_sections(stage),
                "metadata": self._program_stage_metadata(stage),
            }
            for stage in stages
        }
        disk_stages = self._program_bundle.store_stages(requests)
        for stage in stages:
            disk_sections = disk_stages[stage]
            # Reopen the complete sidecar as well as the binary. ProgramBundle's
            # store return proves byte publication; this second boundary also
            # proves persisted bases and compile metadata before authorization.
            if disk_sections != self._read_program_stage_from_disk(stage):
                raise RuntimeError(
                    f"programs.bin stage {stage!r} changed after atomic publication"
                )
            self._install_program_stage(stage, disk_sections)
        self._packaged_program_stages.update(stages)
        detail = ", ".join(
            f"{stage}={sum(len(blob) for blob in disk_stages[stage].values()) / 2**20:.2f} MiB"
            for stage in stages
        )
        self._loud(
            f"  [Program bin] atomically stored {len(stages)} stage(s), "
            f"8 engine images each: {detail}"
        )

    def store_program_stage(self, stage: str) -> None:
        """Backward-friendly one-stage program artifact transaction."""
        self.store_program_stages(stage)

    def _install_program_stage(
        self, stage: str, disk_sections: dict[int, bytes]
    ) -> None:
        if set(disk_sections) != set(range(REQUIRED_ENGINES)):
            raise RuntimeError(
                f"programs.bin stage {stage!r} is incomplete; engines are "
                f"{sorted(disk_sections)}"
            )
        master_addr, compiled_master, workers = self._program_stage_state(stage)
        master = bytes(disk_sections[0])
        if stage in self._raw_stages:
            for engine_index, _worker, _address, compiled_blob in workers:
                if bytes(disk_sections[int(engine_index)]) != bytes(compiled_blob):
                    raise RuntimeError(f"programs.bin {stage} worker {engine_index} differs "
                                       "from the program in DRAM")
            if master != compiled_master:
                raise RuntimeError(f"programs.bin {stage} master differs from the program in DRAM")
            self._loaded_program_stages.add(stage)
            return
        if master != compiled_master:
            raise RuntimeError(
                f"programs.bin {stage} master differs from the ISA compiled "
                "for this request"
            )
        reloaded_workers = []
        for engine_index, worker, address, compiled_blob in workers:
            disk_blob = bytes(disk_sections[int(engine_index)])
            if disk_blob != bytes(compiled_blob):
                raise RuntimeError(
                    f"programs.bin {stage} worker {engine_index} differs from "
                    "the ISA compiled for this request"
                )
            reloaded_workers.append(
                (int(engine_index), worker, int(address), disk_blob)
            )

        if stage == "vision":
            encoder_size = len(self._vis_encoder_program_bytes)
            patch_size = len(self._vis_patch_program_bytes)
            if encoder_size + patch_size != len(master):
                raise RuntimeError(
                    "programs.bin vision composite no longer matches its "
                    "encoder/patch boundaries"
                )
            if self._vis_patch_program_addr != master_addr + encoder_size:
                raise RuntimeError("vision patch entry is not contiguous with encoder")
            self._vis_program_bytes = master
            self._vis_encoder_program_bytes = master[:encoder_size]
            self._vis_patch_program_bytes = master[encoder_size:]
            self._vis_worker_programs = reloaded_workers
        elif stage == "audio":
            self._audio_program_bytes = master
            self._audio_worker_programs = reloaded_workers
        elif stage == "prefill":
            self._prefill_program = (master_addr, master)
            self._prefill_workers = reloaded_workers
        elif stage == "decode":
            self._decoder_program = (master_addr, master)
            self._decoder_workers = reloaded_workers
        else:
            raise ValueError(f"unknown program stage {stage!r}")
        self._loaded_program_stages.add(stage)

    def _read_program_stage_from_disk(self, stage: str) -> dict[int, bytes]:
        """Validate bytes, DRAM bases, and metadata for one compiled stage."""
        if self._program_bundle is None:
            raise RuntimeError("program bundle has not been configured")
        master_addr, _master_blob, workers = self._program_stage_state(stage)
        worker_indices = [int(item[0]) for item in workers]
        if worker_indices != list(range(1, REQUIRED_ENGINES)):
            raise RuntimeError(
                f"compiled {stage} worker order must be engines 1-7, got "
                f"{worker_indices}"
            )
        expected_bases = {0: int(master_addr)}
        expected_bases.update(
            {int(index): int(address) for index, _worker, address, _blob in workers}
        )
        expected_metadata = json.loads(
            json.dumps(
                self._program_stage_metadata(stage),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
        manifest, payload = self._program_bundle.load()
        records = {}
        for section in manifest["sections"]:
            if section["name"] != stage:
                continue
            engine_index = int(section["engine_index"])
            persisted_base = int(section["dram_base"], 16)
            if persisted_base != expected_bases.get(engine_index):
                raise RuntimeError(
                    f"programs.json {stage} engine {engine_index} base "
                    f"0x{persisted_base:X} differs from compiled base "
                    f"{expected_bases.get(engine_index)!r}"
                )
            if section["metadata"] != expected_metadata:
                raise RuntimeError(
                    f"programs.json {stage} engine {engine_index} compile "
                    "metadata differs from the live compiler state"
                )
            start = int(section["file_offset"])
            records[engine_index] = payload[start:start + int(section["size"])]
        if set(records) != set(range(REQUIRED_ENGINES)):
            raise RuntimeError(
                f"programs.bin stage {stage!r} has engines {sorted(records)}, "
                "expected 0-7"
            )
        return records

    def _reload_program_stage(self, stage: str) -> None:
        if stage in self._resident_stages:
            return
        if self._program_bundle is None or stage not in self._packaged_program_stages:
            raise RuntimeError(
                f"{stage} execution is forbidden until its master + seven "
                "worker images have been stored in programs.bin"
            )
        disk_sections = self._read_program_stage_from_disk(stage)
        self._install_program_stage(stage, disk_sections)
        total = sum(len(blob) for blob in disk_sections.values())
        self._loud(
            f"  [Program bin] reloaded + validated {stage}: "
            f"{total / 2**20:.2f} MiB"
        )

    # The disk reload is part of execution, not optional diagnostics.  Dynamic
    # token/media payloads and tiny flag/decode dispatch preambles are still
    # generated at run time; every stable learned stage image comes from disk.
    def run_vision_encoder(self, *args, **kwargs):
        self._reload_program_stage("vision")
        result = super().run_vision_encoder(*args, **kwargs)
        self._executed_program_stages.add("vision")
        return result

    def run_audio_encoder(self, *args, **kwargs):
        self._reload_program_stage("audio")
        result = super().run_audio_encoder(*args, **kwargs)
        self._executed_program_stages.add("audio")
        return result

    def run_prefill(self, *args, **kwargs):
        self._reload_program_stage("prefill")
        result = super().run_prefill(*args, **kwargs)
        self._executed_program_stages.add("prefill")
        return result

    def run_decode_step_profiled(
        self, token: int, program, checkpoints, workers=None,
        timeout_s: float = 60.0,
    ):
        self._reload_program_stage("decode")
        if program != self._decoder_program:
            raise RuntimeError(
                "profiled decode refused an in-memory program that is not the "
                "validated programs.bin decoder section"
            )
        if workers is not None:
            supplied = [(idx, addr, bytes(blob)) for idx, _w, addr, blob in workers]
            loaded = [
                (idx, addr, bytes(blob))
                for idx, _w, addr, blob in self._decoder_workers
            ]
            if supplied != loaded:
                raise RuntimeError(
                    "profiled decode worker images do not match programs.bin"
                )
        result = super().run_decode_step_profiled(
            token,
            self._decoder_program,
            checkpoints,
            workers=self._decoder_workers,
            timeout_s=timeout_s,
        )
        self._executed_program_stages.add("decode")
        return result

    def run_decoder(self, *args, **kwargs):
        self._reload_program_stage("decode")
        result = super().run_decoder(*args, **kwargs)
        self._executed_program_stages.add("decode")
        return result

    # ---- Run summary ------------------------------------------------------
    # Same contract as gemma4_e2b_test.write_run_summary and the qwen2.5_vl_3b
    # writer this model inherits its stages from: everything below reads an
    # attribute a stage already stashed plus cheap host bookkeeping (file
    # sizes, a program manifest, one register read), so writing a summary
    # after a run launches no FPGA program and cannot perturb what it reports.

    def per_core_peak_gflops(self) -> float:
        """One engine's peak -- the denominator for anything core-0-only."""
        cores = getattr(self, "multi_core", 1) or 1
        return self.vis_peak_gflops() / cores

    def stage_metrics(self, args) -> list[dict]:
        """Work, FPGA time and wall time for every stage this run executed.

        One row per stage in execution order. ``flops`` is the stage's own
        reported FLOP count, ``us`` its HW-counter latency, and ``wall`` the
        CPU timer around it; the difference between the last two is host
        overhead (DMA, layout, detokenize), which is what to attack when
        utilisation already looks good.
        """
        rows: list[dict] = []
        if getattr(self, "_vis_latency_us", None):
            dims = self._vision_dims()
            # frames > 1 (--frames): _vis_total_flops is already the real
            # N-frame total (_run_vision aggregates it), so the true/model
            # FLOP count -- computed here from ONE frame's dims -- needs the
            # same x frames to stay comparable in the "Useful" column.
            frames = int(getattr(args, "frames", 1) or 1)
            detail = f"{dims['VS']} patches -> {dims['NUM_MERGED_TOKENS']} soft tokens"
            if frames > 1:
                detail = f"{frames} x ({detail})"
            rows.append({
                "stage": "Vision encoder",
                "detail": detail,
                "flops": float(self._vis_total_flops),
                "model_flops": float(self._model_flops_vision(dims) or 0.0) * frames,
                "us": float(self._vis_latency_us),
                "wall": float(getattr(self, "_vis_wall_s", 0.0)),
            })
        if getattr(self, "_audio_latency_us", None):
            rows.append({
                "stage": "Audio encoder",
                "detail": f"{getattr(self, '_audio_num_tokens', 0)} soft tokens",
                "flops": float(getattr(self, "_audio_total_flops", 0.0)),
                "model_flops": self._model_flops_audio(),
                "us": float(self._audio_latency_us),
                "wall": float(getattr(self, "_audio_wall_s", 0.0)),
            })
        if getattr(self, "_latency_prefill_us", None):
            rows.append({
                "stage": "Prefill",
                "detail": f"{getattr(self, '_prefill_seq_len_run', 0)} tokens",
                "flops": float(getattr(self, "_prefill_flops", 0.0)),
                "model_flops": self._model_flops_prefill(),
                "us": float(self._latency_prefill_us),
                "wall": float(getattr(self, "_prefill_wall_s", 0.0)),
            })
        steps = getattr(self, "_decode_step_us", None)
        if steps:
            rows.append({
                "stage": "Decode",
                "detail": f"{len(steps)} steps, "
                          f"{getattr(self, '_decode_n', len(steps))} tokens kept",
                "flops": float(getattr(self, "_decode_step_flops", 0.0)),
                "model_flops": self._model_flops_decode(len(steps)),
                "us": float(getattr(self, "_decode_total_us", 0.0)),
                "wall": float(getattr(self, "_decode_wall_s", 0.0)),
            })
        peak = self.vis_peak_gflops()
        core_peak = self.per_core_peak_gflops()
        for row in rows:
            row["gflops"] = row["flops"] / (row["us"] * 1e3) if row["us"] else 0.0
            row["util_pct"] = 100.0 * row["gflops"] / peak if peak else 0.0
            row["speedup"] = row["gflops"] / core_peak if core_peak else 0.0
            model = row.get("model_flops")
            row["model_gflops"] = (
                model / (row["us"] * 1e3) if model and row["us"] else None
            )
            row["model_util_pct"] = (
                100.0 * row["model_gflops"] / peak
                if row["model_gflops"] and peak else None
            )
            row["useful_pct"] = (
                100.0 * model / row["flops"] if model and row["flops"] else None
            )
        return rows

    # Model-FLOP counters.  Each one converts what the stage actually ran into
    # the logical shapes qwen2.5_omni_7b_model_flops prices, and returns None
    # rather than raising if a stage did not record what it needs -- a report
    # must never be the reason a completed run fails.

    def _model_flops_vision(self, dims: dict) -> float | None:
        try:
            return float(_model_flops.vision_flops(
                self._cfg,
                patches=int(dims["VS"]),
                merged_tokens=int(dims["NUM_MERGED_TOKENS"]),
            ))
        except Exception:
            return None

    def _model_flops_audio(self) -> float | None:
        try:
            chunks = [int(n) for n in self._audio_chunk_aftercnn_lens]
            return float(_model_flops.audio_flops(
                self._cfg,
                conv1_rows=sum(int(n) for n in self._audio_chunk_feature_lens),
                encoder_states=int(self._audio_seq_len),
                chunk_states=chunks,
                pooled_tokens=int(self._audio_num_tokens),
            ))
        except Exception:
            return None

    def _model_flops_prefill(self) -> float | None:
        try:
            return float(_model_flops.prefill_flops(
                self._cfg, int(self._prefill_seq_len_run)))
        except Exception:
            return None

    def _model_flops_decode(self, steps: int) -> float | None:
        # Step i attended the KV history it actually had: the run ends at
        # self.seq_len and every step advanced it by one.
        try:
            end = int(self.seq_len)
            contexts = range(end - int(steps) + 1, end + 1)
            return float(_model_flops.decode_flops(self._cfg, contexts))
        except Exception:
            return None

    def _model_flops_vision_by_phase(self, dims: dict) -> dict[str, int] | None:
        try:
            return _model_flops.vision_flops_by_phase(
                self._cfg,
                patches=int(dims["VS"]),
                merged_tokens=int(dims["NUM_MERGED_TOKENS"]),
            )
        except Exception:
            return None

    def _model_flops_prefill_by_phase(self) -> dict[str, int] | None:
        try:
            return _model_flops.prefill_flops_by_phase(
                self._cfg, int(self._prefill_seq_len_run))
        except Exception:
            return None

    def _stage_table(self, rows: list[dict]) -> list[str]:
        """Headline table: work, time, throughput, % of peak, core scaling,
        and the model's effective throughput against the same measured time.

        `Work`/`Throughput`/`% of peak` are what this engine ISSUED, at the
        padded and tile-aligned shapes the hardware ran: a 64-row execution
        multiple for a short prompt, windowed attention widened to a full
        mask, an aligned head. `Model GFLOP`/`Effective GFLOPS` are what the
        architecture owes at its own dimensions -- true prompt length, true
        attention windows, matrix products only -- divided by the SAME
        measured FPGA time, so `Effective GFLOPS` is comparable to any other
        implementation of this model on any hardware. `Useful` is how much
        of the issued work the model needed; `n/a` where a stage's model
        FLOPs could not be priced, not a misleading 0.
        """
        cores = getattr(self, "multi_core", 1) or 1
        out = [
            "| Stage | Shape | Work (GFLOP) | Model GFLOP | Useful | FPGA "
            "time (ms) | Throughput (GFLOPS) | Effective GFLOPS | % of peak "
            "| x 1-engine peak | CPU wall (s) |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
            "| ---: | ---: |",
        ]
        suspect = False
        for row in rows:
            model = row.get("model_flops")
            if model:
                # Above 100% the engine billed FEWER FLOPs than the
                # architecture requires, which cannot happen: padding and
                # alignment only ever ADD work. It means that stage's own
                # accounting is undercounting what it issued, so flag it
                # rather than printing it deadpan.
                mark = ""
                if row["useful_pct"] > 100.5:
                    mark = " !"
                    suspect = True
                model_s = f"{model / 1e9:.2f}"
                useful_s = f"{row['useful_pct']:.1f}%"
                eff_s = f"{row['model_gflops']:.1f}"
            else:
                mark, model_s, useful_s, eff_s = "", "n/a", "n/a", "n/a"
            out.append(
                f"| {row['stage']}{mark} | {row['detail']} | "
                f"{row['flops'] / 1e9:.2f} | {model_s} | {useful_s} | "
                f"{row['us'] / 1e3:.1f} | {row['gflops']:.1f} | {eff_s} | "
                f"{row['util_pct']:.1f}% | {row['speedup']:.2f}x | "
                f"{row['wall']:.2f} |"
            )
        total_flops = sum(row["flops"] for row in rows)
        total_us = sum(row["us"] for row in rows)
        total_wall = sum(row["wall"] for row in rows)
        peak = self.vis_peak_gflops()
        core_peak = self.per_core_peak_gflops()
        total_gflops = total_flops / (total_us * 1e3) if total_us else 0.0
        priced = [row for row in rows if row.get("model_flops")]
        if priced and len(priced) == len(rows):
            model_total = sum(row["model_flops"] for row in priced)
            total_useful_s = (
                f"{(100.0 * model_total / total_flops if total_flops else 0.0):.1f}%"
            )
            total_model_s = f"{model_total / 1e9:.2f}"
            total_eff_s = f"{(model_total / (total_us * 1e3) if total_us else 0.0):.1f}"
        else:
            total_model_s = total_useful_s = total_eff_s = "n/a"
        out.append(
            f"| **TOTAL** | {cores} engines | **{total_flops / 1e9:.2f}** | "
            f"**{total_model_s}** | **{total_useful_s}** | "
            f"**{total_us / 1e3:.1f}** | **{total_gflops:.1f}** | "
            f"**{total_eff_s}** | "
            f"**{(100.0 * total_gflops / peak if peak else 0.0):.1f}%** | "
            f"**{(total_gflops / core_peak if core_peak else 0.0):.2f}x** | "
            f"**{total_wall:.2f}** |"
        )
        if suspect:
            out += [
                "",
                "`!` marks a stage whose issued FLOPs came out BELOW the "
                "model's requirement. Padding and alignment can only add "
                "work, so that stage's own FLOP accounting is undercounting "
                "what it issued -- treat its `% of peak` as understated, and "
                "`Effective GFLOPS` as the reliable rate.",
            ]
        return out

    def _profile_tables(self, stages) -> list[str]:
        """Per-phase markdown tables, one per profiled stage.

        Aggregation, the serial-phase peak guard and the column set are shared
        with the terminal breakdown (print_profile_table), so the .md and the
        console can never disagree about a phase.

        ``stages`` entries are ``(title, note, results, model_phases)``.
        ``model_phases`` (a dict, or None) keys must match
        ``_aggregate_vis_profile``'s phase names exactly -- the checkpoint
        name after its "L<idx>:" prefix is stripped, e.g. "attention",
        "qkv_proj". Vision's separate FPGA patch program contributes a
        "patch_embed" result before those checkpoints. The ``*_by_phase``
        functions use these same boundaries; bundled checkpoints such as
        "o_proj+mlp" remain bundled in the model counts. A missing mapping
        is shown as "n/a", never silently counted as zero.
        """
        peak = self.vis_peak_gflops()
        out: list[str] = []
        for title, note, results, model_phases in stages:
            if not results:
                continue
            rows = self._aggregate_vis_profile(results)
            total = sum(row["ms"] for row in rows) or 1.0
            out += [f"### {title}", ""]
            if note:
                out += [note, ""]
            has_model = bool(model_phases)
            header = ["Phase", "Calls", "Total ms", "Share", "GFLOP"]
            if has_model:
                header += ["Model GFLOP", "Useful"]
            header += ["GFLOPS", "Effective GFLOPS" if has_model else None,
                      "% of peak"]
            header = [h for h in header if h is not None]
            out.append("| " + " | ".join(header) + " |")
            out.append("|" + "|".join(
                ":--" if h == "Phase" else "--:" for h in header) + "|")
            model_total = 0
            for row_, (name, n, ms, share, gflop, gflops, util) in zip(
                rows, self._vis_profile_table(rows, total)
            ):
                cells = [name, str(n), f"{ms:.2f}", f"{share:.1f}%",
                        f"{gflop:.2f}"]
                if has_model:
                    phase_model = model_phases.get(row_["phase"])
                    if phase_model is None:
                        cells += ["n/a", "n/a"]
                        eff_s = "n/a"
                    else:
                        model_total += phase_model
                        model_gflop = phase_model / 1e9
                        flops = row_["flops"]
                        useful = 100.0 * phase_model / flops if flops else 0.0
                        eff = model_gflop / (ms / 1e3) if ms else 0.0
                        cells += [f"{model_gflop:.2f}", f"{useful:.1f}%"]
                        eff_s = f"{eff:.1f}"
                    cells += [f"{gflops:.1f}", eff_s]
                else:
                    cells += [f"{gflops:.1f}"]
                cells += [f"{util:.1f}%"]
                out.append("| " + " | ".join(cells) + " |")
            total_gflop = sum(row["flops"] for row in rows) / 1e9
            total_rate = total_gflop / (total / 1e3) if total else 0.0
            total_cells = ["**TOTAL**", str(len(results)), f"**{total:.2f}**",
                          "100.0%", f"**{total_gflop:.2f}**"]
            if has_model:
                total_flops = sum(row["flops"] for row in rows)
                total_useful = (100.0 * model_total / total_flops
                                if total_flops else 0.0)
                total_eff = model_total / 1e9 / (total / 1e3) if total else 0.0
                total_cells += [f"**{model_total / 1e9:.2f}**",
                                f"**{total_useful:.1f}%**"]
            total_cells += [f"**{total_rate:.1f}**"]
            if has_model:
                total_cells += [f"**{total_eff:.1f}**"]
            total_cells += [
                f"**{(100.0 * total_rate / peak if peak else 0.0):.1f}%**"]
            out.append("| " + " | ".join(total_cells) + " |")
            out.append("")
        return out

    def _program_section_lines(self) -> list[str]:
        """Per-stage programs.bin section sizes, read from the manifest."""
        bundle = getattr(self, "_program_bundle", None)
        if bundle is None:
            return []
        try:
            manifest, _payload = bundle.load()
        except Exception as exc:
            return [f"- **programs.bin:** (manifest unreadable: {exc})"]
        bin_path = bundle.bin_path
        lines = [
            f"- **Program bin:** `{os.path.basename(str(bin_path))}` — "
            f"{os.path.getsize(bin_path) / 2**20:.2f} MiB "
            f"({manifest['section_count']} sections)"
        ]
        per_stage: dict[str, list[int]] = {}
        for section in manifest["sections"]:
            per_stage.setdefault(section["name"], []).append(int(section["size"]))
        for stage in sorted(per_stage):
            sizes = per_stage[stage]
            lines.append(
                f"  - **{stage}:** {sum(sizes) / 2**20:.2f} MiB across "
                f"{len(sizes)} engine sections "
                f"(master {sizes[0] / 1024:.1f} KiB)"
            )
        return lines


    def write_run_summary(self, out_path: str, args, profiles=None) -> str:
        """Write the per-run Markdown summary and return the path.

        Two clocks are reported on purpose. The HW counter times the program on
        the engines; the CPU timer wraps the whole stage including host work
        (media preprocessing, embedding gather, RoPE and bias DMA, preamble
        write, argmax readback, detokenize). Their ratio is the host overhead.
        """
        clock_ns = (getattr(self, "_clock_period_ns", None)
                    or user_dma_core.CLOCK_CYCLE_TIME_NS)
        freq_mhz = 1000.0 / clock_ns if clock_ns else 0.0
        cores = getattr(self, "multi_core", 1) or 1
        peak = self.vis_peak_gflops()
        core_peak = self.per_core_peak_gflops()
        try:
            hw = (f"0x{self.user_read_reg32(user_dma_core.UE_FPGA_VERSION_ADDR) & 0xFFFFFFFF:08x}")
        except Exception as exc:
            hw = f"(read failed: {exc})"

        params_bin = os.path.join(self.script_dir, self._cfg["paths"]["params"])
        lines = [
            "# qwen2.5_omni_7b run summary",
            "",
            f"- **Mode:** {_result_mode(args)}",
            "",
            "## Hardware",
            "",
            f"- **HW version:** {hw}",
            f"- **Device:** {args.dev}",
            f"- **Clock:** {clock_ns:.4f} ns ({freq_mhz:.1f} MHz)",
            f"- **AXI data width:** {user_dma_core.UE_AXI_DATA_WIDTH_BITS} bits",
            f"- **DRAM:** {user_dma_core.AVAILABLE_DRAM_SIZE_GB} GiB",
            f"- **Engines in use:** {cores} of "
            f"{user_dma_core.ANDROMEDA_CORE_COUNT} reported",
            f"- **Peak throughput:** {peak:.1f} GFLOPS "
            f"({freq_mhz:.1f} MHz x 128 FLOP/cycle x {cores} engines)",
            f"- **Per-engine peak:** {core_peak:.1f} GFLOPS",
            "",
            "## Weights and programs",
            "",
        ]
        if os.path.exists(params_bin):
            lines.append(
                f"- **Weight bin:** `{os.path.basename(params_bin)}` — "
                f"{os.path.getsize(params_bin) / 2**20:.1f} MiB (validated "
                f"against params.json)"
            )
        if getattr(self, "_lm_weight_init_done", False):
            lines.append(
                f"- **LM weight DRAM:** "
                f"{(self._lm_weight_end - self.PARAMS_BASE) / 2**20:.1f} MiB "
                f"(IF4 projections incl. V; BF16 embedding on host)"
            )
        lines += self._program_section_lines()
        lines.append("")

        rows = self.stage_metrics(args)
        if rows:
            lines += [
                "## Stage summary",
                "",
                "FPGA time is the HW counter; CPU wall is the host-side timer "
                "around the same stage. `x 1-engine peak` is the achieved rate "
                "divided by ONE engine's peak: the effective speedup the "
                f"{cores}-engine split delivered, against a ceiling of "
                f"{cores}.00x. `Model GFLOP` is what the architecture owes at "
                "its own dimensions -- true prompt length, true attention "
                "windows, matrix products only; `Effective GFLOPS` divides it "
                "by the same measured FPGA time, so it is comparable to any "
                "other implementation of this model on any hardware; `Useful` "
                "is how much of the issued work the model needed.",
                "",
            ]

            lines += self._stage_table(rows)
            lines.append("")

        if getattr(self, "_vis_latency_us", None):
            dims = self._vision_dims()
            frames = int(getattr(args, "frames", 1) or 1)
            frame_tag = f"{frames} x " if frames > 1 else ""
            lines += [
                "## Vision",
                "",
                f"- **Image:** `{os.path.basename(getattr(args, 'image', '') or '')}` "
                f"-> {frame_tag}({dims['VS']} patches -> {dims['NUM_MERGED_TOKENS']} "
                f"soft tokens) = {frames * dims['NUM_MERGED_TOKENS']} soft tokens total",
                f"- **Work:** {self._vis_total_flops / 1e9:.1f} GFLOP",
                f"- **HW latency:** {self._vis_latency_us / 1e3:.1f} ms",
                f"- **Throughput:** {self._vis_gflops:.1f} GFLOPS "
                f"({100.0 * self._vis_gflops / peak if peak else 0.0:.1f}% of peak)",
                f"- **End-to-end (CPU timer):** "
                f"{getattr(self, '_vis_wall_s', 0.0):.2f} s",
                "",
            ]

        if getattr(self, "_audio_latency_us", None):
            audio_gflops = getattr(self, "_audio_gflops", 0.0)
            lines += [
                "## Audio",
                "",
                f"- **Audio:** `{os.path.basename(getattr(args, 'audio', '') or '')}` "
                f"-> {getattr(self, '_audio_num_tokens', 0)} soft tokens",
                f"- **Work:** {getattr(self, '_audio_total_flops', 0) / 1e9:.1f} GFLOP",
                f"- **HW latency:** {self._audio_latency_us / 1e3:.1f} ms",
                f"- **Throughput:** {audio_gflops:.1f} GFLOPS "
                f"({100.0 * audio_gflops / peak if peak else 0.0:.1f}% of peak)",
                f"- **End-to-end (CPU timer):** "
                f"{getattr(self, '_audio_wall_s', 0.0):.2f} s",
                "",
            ]

        if getattr(self, "_latency_prefill_us", None):
            prefill_gflops = getattr(self, "_prefill_gflops", 0.0)
            lines += [
                "## Prefill",
                "",
                f"- **Sequence length:** "
                f"{getattr(self, '_prefill_seq_len_run', 0)} tokens",
                f"- **Work:** {getattr(self, '_prefill_flops', 0) / 1e9:.1f} GFLOP",
                f"- **HW latency:** {self._latency_prefill_us / 1e3:.1f} ms",
                f"- **Throughput:** {prefill_gflops:.1f} GFLOPS "
                f"({100.0 * prefill_gflops / peak if peak else 0.0:.1f}% of peak)",
                f"- **End-to-end (CPU timer):** "
                f"{getattr(self, '_prefill_wall_s', 0.0):.2f} s",
                "- **Embedding:** host BF16 gather and row DMA are included "
                "in CPU time, not FPGA time",
                "",
            ]

        # TTFT ends at the decode-ready state, so it carries whichever encoders
        # this request ran plus prefill.
        pre_hw_us = float(getattr(self, "_latency_prefill_us", 0.0) or 0.0)
        pre_wall = float(getattr(self, "_prefill_wall_s", 0.0) or 0.0)
        if pre_hw_us or pre_wall:
            enc_hw_us = float(getattr(self, "_vis_latency_us", 0.0) or 0.0)
            enc_hw_us += float(getattr(self, "_audio_latency_us", 0.0) or 0.0)
            enc_wall = float(getattr(self, "_vis_wall_s", 0.0) or 0.0)
            enc_wall += float(getattr(self, "_audio_wall_s", 0.0) or 0.0)
            covered = [name for name, seen in (
                ("vision", getattr(self, "_vis_latency_us", None)),
                ("audio", getattr(self, "_audio_latency_us", None)),
            ) if seen] + ["prefill"]
            lines += [
                "## Time to first token",
                "",
                f"- **TTFT (HW counter; {' + '.join(covered)}):** "
                f"{(enc_hw_us + pre_hw_us) / 1e3:.1f} ms",
                f"- **TTFT (CPU timer; {' + '.join(covered)}):** "
                f"{enc_wall + pre_wall:.2f} s",
                "  (includes prefill host embedding lookup and DMA)",
                "",
            ]

        steps = getattr(self, "_decode_step_us", None)
        if steps:
            n = len(steps)
            first_us = steps[0]
            total_us = float(getattr(self, "_decode_total_us", 0.0))
            wall = float(getattr(self, "_decode_wall_s", 0.0))
            hw_avg_us = total_us / n if n else 0.0
            decode_gflops = getattr(self, "_decode_gflops", 0.0)
            lines += [
                "## Decode",
                "",
                f"- **Steps:** {n} (kept {getattr(self, '_decode_n', n)} tokens, "
                f"sequence total {self.seq_len})",
                f"- **First-token speed (HW counter):** {1e6 / first_us:.2f} tok/s "
                f"({first_us / 1e3:.1f} ms)",
                f"- **Average speed (HW counter):** "
                f"{(1e6 / hw_avg_us if hw_avg_us else 0.0):.2f} tok/s "
                f"({hw_avg_us / 1e3:.1f} ms/token)",
                f"- **Average speed (CPU timer):** "
                f"{(n / wall if wall else 0.0):.2f} tok/s "
                f"({(1e3 * wall / n if n else 0.0):.1f} ms/token)",
                f"- **Host overhead:** "
                f"{(100.0 * (1 - total_us / 1e6 / wall) if wall else 0.0):.1f}% "
                f"of wall time outside the engines",
                f"- **Work per token:** "
                f"{getattr(self, '_decode_step_flops', 0) / n / 1e9:.2f} GFLOP",
                f"- **Throughput:** {decode_gflops:.1f} GFLOPS "
                f"({100.0 * decode_gflops / peak if peak else 0.0:.1f}% of peak)",
                f"- **End-to-end (CPU timer):** {wall:.2f} s",
                "",
            ]

        speech = getattr(self, "_speech_result", None)
        if speech:
            codec_tokens = int(speech["codec_tokens"])
            talker_s = float(speech["talker_wall_s"])
            token2wav_s = float(speech["token2wav_wall_s"])
            audio_s = float(speech["seconds"])
            talker_hw = (f"{float(speech['talker_hw_us']) / 1e3:.1f} ms"
                         if speech.get("talker_hw_us") is not None else "—")
            token2wav_hw = (f"{float(speech['token2wav_hw_us']) / 1e3:.1f} ms"
                            if speech.get("token2wav_hw_us") is not None else "—")
            lines += [
                "## Speech output",
                "",
                "Talker and Token2Wav are measured separately by CPU timers. "
                "The device column identifies where each stage ran; `—` means "
                "no FPGA HW counter applies to that stage.",
                "",
                "| Stage | Device | Output | HW counter | CPU wall | Rate |",
                "| :--- | :--- | ---: | ---: | ---: | ---: |",
                f"| Talker | {speech['talker_device']} | {codec_tokens} codec tokens "
                f"| {talker_hw} | {talker_s:.1f} s | "
                f"{codec_tokens / talker_s if talker_s else 0.0:.1f} codec tok/s |",
                f"| Token2Wav | {speech['token2wav_device']} | {audio_s:.2f} s audio "
                f"| {token2wav_hw} | {token2wav_s:.1f} s | "
                f"{audio_s / token2wav_s if token2wav_s else 0.0:.2f}× real time |",
                "",
                *_speech_performance_lines(speech),
                *([f"- **Token2Wav breakdown (CPU timer):** setup "
                   f"{float(speech['token2wav_setup_s']):.1f} s, DiT (36 evaluations) "
                   f"{float(speech['token2wav_dit_s']):.1f} s, BigVGAN "
                   f"{float(speech['token2wav_bigvgan_s']):.1f} s"]
                  if speech.get("token2wav_dit_s") is not None else []),
                *([f"- **Length cap:** the Talker reached the {T2W_FPGA_MAX_CODES}-codec-token "
                   "FPGA Token2Wav limit before emitting its end token, so speech is cut "
                   "there (any remaining text is not spoken)."]
                  if speech.get("codec_truncated") else []),
                f"- **Decode/Talker overlap (CPU timer):** "
                f"{float(speech.get('decode_talker_overlap_s', 0.0)):.2f} s",
                f"- **Speaker:** {speech['speaker']}",
                "",
                "### Workflow",
                "",
                "1. **Thinker (Qwen2 7B decoder, IF4 weights, 8 FPGA engines):** prefill the "
                "prompt, then decode the reply tokens; each step's final hidden state is "
                "kept for the Talker.",
                "2. **Talker (24-layer Qwen2 decoder, IF4, 8 FPGA engines):** takes each "
                "reply token's Thinker hidden state plus its embedding (host adds the two "
                "3584-wide rows), projects to 896, runs 24 layers over a device-resident KV "
                "cache and the codec head, and the host samples one codec token per step "
                "(top-k 40, top-p 0.8, temperature 0.9, repetition penalty 1.05).",
                f"3. **Token2Wav on {speech['token2wav_device']}:** a flow-matching DiT "
                "(22 blocks, 36 Runge-Kutta evaluations with a 2-way guidance batch) turns the "
                "codec tokens into an 80-bin mel spectrogram; a BigVGAN vocoder (6 "
                "polyphase upsample stages, 18 anti-aliased SnakeBeta residual blocks) turns "
                "the mel into a 24 kHz waveform. The host only adds the speaker/codec "
                "constants and the Runge-Kutta bookkeeping.",
                "4. The waveform is written next to this summary as a `.wav`.",
                f"- **WAV:** `{os.path.basename(speech['wav'])}` "
                f"({audio_s:.2f} s at {_talker_mod.SAMPLE_RATE} Hz)",
                "",
            ]
            if speech.get("talker_prefix_wall_s") is not None:
                lines += [
                    "FPGA Talker runs after Thinker decode because both own the same "
                    "eight queues; the 0 s overlap is intentional. Talker time "
                    "includes conditioning over the prompt before codec generation. "
                    "One-time weight staging and program compilation are reported "
                    "separately from active Talker execution.",
                    "",
                    f"- **FPGA speech context budget:** "
                    f"{self.MAX_CONTEXT_SIZE} Thinker tokens; "
                    f"{self.layout.tensor_bytes / 2**20:.0f} MiB private tensors "
                    f"and {self._private_reserve_bytes / 2**20:.0f} MiB private "
                    "weight reserve per engine",
                    f"- **Talker setup (readback, weights, compilation):** "
                    f"{float(speech['talker_setup_wall_s']):.2f} s CPU",
                    f"- **Talker prefix:** {float(speech['talker_prefix_wall_s']):.2f} s "
                    f"CPU, {float(speech['talker_prefix_hw_us']) / 1e6:.2f} s "
                    "core-0 HW counter",
                    f"- **Talker codec decode:** "
                    f"{float(speech['talker_decode_wall_s']):.2f} s CPU, "
                    f"{float(speech['talker_decode_hw_us']) / 1e6:.2f} s "
                    "core-0 HW counter",
                    f"- **Active codec decode speed:** "
                    f"{codec_tokens / float(speech['talker_decode_wall_s']):.1f} "
                    "codec tok/s (CPU timer)",
                    f"- **Talker total including setup:** "
                    f"{float(speech['talker_total_wall_s']):.2f} s CPU",
                    "",
                ]

        if profiles:
            lines += [
                "## Per-phase profile",
                "",
                "Phase latencies come from FPGA hardware counters. Vision's "
                "patch_embed is a separate program; the remaining phases are "
                "timed between per-phase HALTs and include a stop/restart per "
                "phase. They exclude host time, so the SHARE column says "
                "which phase is worth sharding next. `*` marks phases that run "
                "on engine 0 only, whose % of peak is measured against ONE "
                "engine.",
                "",
            ]
            lines += self._profile_tables(profiles)

        prompt = getattr(self, "_prompt_text", None)
        if prompt is not None:
            lines += ["## Prompt & output", "", "### Prompt", "", "```",
                      prompt, "```", ""]
            lines += ["### Decoded text", "", "```",
                      getattr(self, "_decoded_text", "") or "(none)", "```", ""]

        with open(out_path, "w") as handle:
            handle.write("\n".join(lines))
        return out_path

    def _loud(self, *args, **kwargs) -> None:
        _ORIGINAL_PRINT(*args, **kwargs)

    def _set_silent(self, on: bool) -> bool:
        global _SILENT_MODE
        previous = _SILENT_MODE
        _SILENT_MODE = bool(on)
        return previous

    def _note_worker_isa(self, idx: int, stage: str, nbytes: int) -> None:
        # Every stage appends a two-instruction flag-clear program at runtime.
        # Decode then appends one add-set per live worker register plus a jump.
        runtime_bytes = 2 * user_dma_core.INSTRUCTION_SIZE_BYTES
        if stage == "decode":
            regs = getattr(self, "_decode_attn_worker_regs", ())
            reg_count = len(regs[idx - 1]) if idx - 1 < len(regs) else 5
            raw = (reg_count + 1) * user_dma_core.INSTRUCTION_SIZE_BYTES
            runtime_bytes += ((raw + 63) // 64) * 64
        self._worker_isa_used.setdefault(idx, {})[stage] = (
            int(nbytes) + runtime_bytes
        )

    def _ensure_stage_scheduler(self, stage: str):
        scheduler = self._multi_core_schedulers.get(stage)
        if scheduler is None:
            pool_kw = ({"workers": self._worker_pool}
                       if self._worker_pool is not None else {})
            scheduler = MultiEngineScheduler(
                self,
                num_engines=REQUIRED_ENGINES,
                engine_base_stride=0x00010000,
                arena=self.layout,
                **pool_kw,
                handshake=(
                    "host_segmented"
                    if self.fpga_build == LEGACY_HOST_SEGMENTED_BUILD
                    else "four_phase"
                ),
                region_rendezvous="master_worker",
                barrier_margin_nops=32,
                allow_unaligned_rows=True,
                allow_more_than_two_engines=True,
            )
            self._multi_core_schedulers[stage] = scheduler
            if self.layout.resident_programs and self._worker_pool is None:
                # ONE allocator per worker engine for the whole run, so each
                # stage's programs land after the previous stage's.
                self._worker_pool = list(scheduler.workers)
                for worker in self._worker_pool:
                    self._guard_resident_dma(worker)
        return scheduler

    # -- resident programs ------------------------------------------------
    def reset_program_dram_addr(self) -> None:
        """Stage compiles start from here; resident stages keep their own range."""
        if not self.layout.resident_programs:
            super().reset_program_dram_addr()

    def _guard_resident_dma(self, engine) -> None:
        """Skip re-writing a program image that is already in DRAM.

        The stage runners upload their program before every run. Once an image
        has been installed (install_resident_programs) that write is redundant
        and, for vision's 120 MiB, not free. Only the registered (address, size)
        of a stage image is skipped; every other write -- tensors, the small
        flag-clear programs, decode preambles -- goes through.
        """
        if getattr(engine, "_resident_guarded", False):
            return
        original = engine.dma_write
        images = self._resident_images.setdefault(id(engine), {})

        def dma_write(device, address, buffer, size, _orig=original, _images=images):
            if int(size) in _images.get(int(address), ()):
                return int(size)
            return _orig(device, address, buffer, size)

        engine.dma_write = dma_write
        engine._resident_guarded = True

    def snapshot_stage_tensors(self, stage: str, before: dict) -> None:
        """Remember where a stage's tensors were carved when its program was built."""
        labels = {k: v for k, v in self._dram_addresses.items()
                  if before.get(k) != v}
        self._stage_tensor_state[stage] = {
            "labels": labels,
            "marks": (self.layout.tensor_phase_mark(), self._tensor_staged),
        }

    def enter_stage_tensors(self, stage: str, init) -> None:
        """Give a stage its tensors back at the addresses its program uses.

        Tensors alias across stages, so a stage's contents (constants, zeroed
        caches) have to be rewritten when it starts. Re-running its tensor
        init does exactly that, and because the carve is deterministic it
        yields the same addresses; any difference would mean the compiled
        program points at the wrong buffers, so it is checked.
        """
        state = self._stage_tensor_state[stage]
        entered = time.perf_counter()
        self.reset_tensor_dram_addr()
        init()
        moved = {k: (v, self._dram_addresses.get(k)) for k, v in state["labels"].items()
                 if self._dram_addresses.get(k) != v}
        if moved:
            k, (old, new) = next(iter(moved.items()))
            raise RuntimeError(
                f"{stage} tensors moved since compile ({len(moved)} buffer(s), "
                f"e.g. {k}: 0x{old:X} -> 0x{new if new is not None else 0:X})")
        # Whatever the compile allocated after tensor init stays allocated.
        layout_mark, staged = state["marks"]
        self.layout.tensor_phase_set(layout_mark)
        self._tensor_staged = staged
        self._loud(f"  [{stage}] tensors re-entered in "
                   f"{time.perf_counter() - entered:.2f}s (no weights or programs loaded)")

    def verify_resident_images(self, stage: str | None = None) -> list[str]:
        """Read every resident image back and compare it with what was installed."""
        bad = []
        for (name, who, addr), (engine, size, digest) in self._resident_hashes.items():
            if stage is not None and name != stage:
                continue
            buf = bytearray(size)
            got = engine.dma_read(engine.c2h_device, addr, buf, size)
            if got != size or hashlib.sha1(bytes(buf)).hexdigest() != digest:
                bad.append(f"{name} {who} @0x{addr:X} ({size} B)")
        return bad

    @staticmethod
    def _capture_bytes(engine, emit) -> bytes:
        """Instruction bytes ``emit`` produces on ``engine``; nothing is written."""
        engine.clear_inst_id()
        engine.clear_capture_buffer()
        engine.start_capture()
        emit()
        engine.stop_capture()
        blob = b"".join(i.get_bytes() for i in engine.capture_buffer)
        engine.clear_capture_buffer()
        return blob

    def _install_table(self, engine, blobs: list[bytes], what: str, limit=None) -> tuple:
        """Write fixed-stride launch entries at the engine's program cursor."""
        stride = (max(len(b) for b in blobs) + 63) // 64 * 64
        table = b"".join(b.ljust(stride, b"\0") for b in blobs)
        base = engine.get_program_dram_addr()
        if limit is not None and base + len(table) > limit:
            raise MemoryError(f"{what} ({len(table) / 2**10:.0f} KiB) exceeds the ISA slice")
        if engine.dma_write(user_dma_core.DMA_DEVICE_H2C, base, table, len(table)) != len(table):
            raise IOError(f"{what}: short DMA")
        engine.allocate_program_dram(len(table))
        return base, stride, table

    def install_decode_launch_tables(self) -> None:
        """Precompile every decode launch entry, so a step is started by address.

        The master entry for position p primes the KV-row and aligned-length
        registers and jumps into the decoder body; each worker has one entry
        per 64-row aligned length. They replace the per-step preamble the run
        loop used to generate and write.
        """
        if self._device_embedding_enabled():
            return                      # that path emits a token-dependent lookup
        if self._ensure_stage_scheduler("decode").host_segmented:
            return                      # legacy rendezvous needs host-built segments
        addr, _ = self._decoder_program
        entries = []
        for pos in range(self.MAX_CONTEXT_SIZE):
            aligned = ((pos + 1 + 63) // 64) * 64
            entries.append(self._capture_bytes(self, lambda p=pos, a=aligned: (
                self.generate_instruction_add_set(self.gf_seq_len, p),
                self.generate_instruction_add_set(self.gf_aligned_seq_len, a),
                self.generate_instruction_jump_abs(
                    user_dma_core.ue_35bit_addr_shifter(addr)))))
        base, stride, table = self._install_table(
            self, entries, "decode launch table", self.MASTER_ISA_LIMIT)
        dec_sched = self._ensure_stage_scheduler("decode")
        buckets = (self.MAX_CONTEXT_SIZE + 63) // 64
        workers, worker_images = [], []
        for idx, worker, waddr, _blob in self._decoder_workers:
            blobs = []
            for b in range(1, buckets + 1):
                sets = self._decode_attn_worker_gpr_sets(dec_sched, 64 * b)[idx - 1]
                blobs.append(self._capture_bytes(worker, lambda s=sets, a=waddr: (
                    [worker.generate_instruction_add_set(r, v) for r, v in s],
                    worker.generate_instruction_jump_abs(
                        user_dma_core.ue_35bit_addr_shifter(a)))))
            wbase, wstride, wtable = self._install_table(
                worker, blobs, f"decode worker {idx} launch table",
                self.layout.isa_limit(idx))
            workers.append((idx, worker, wbase, wstride))
            worker_images.append((idx, worker, wbase, wtable))
        self._decode_launch = {"master": (base, stride), "workers": workers}
        self.register_raw_stage(
            "decode_launch", (base, table), worker_images,
            {"positions": self.MAX_CONTEXT_SIZE, "aligned_buckets": buckets,
             "master_stride": stride})

    def install_flag_clear_programs(self) -> None:
        """One precompiled flag-clear program per engine, launched by address.

        The runtime used to generate this two-instruction program and write it at
        each engine's program cursor before every launch, which put it on top of
        whatever followed the stage that had just run.
        """
        engines = [(0, self)] + list(enumerate(self._worker_pool or (), start=1))
        images = []
        for idx, engine in engines:
            blob = self._capture_bytes(engine, lambda e=engine: (
                e.generate_instruction_flag_clear(), e.generate_instruction_halt()))
            limit = self.MASTER_ISA_LIMIT if idx == 0 else self.layout.isa_limit(idx)
            base, _stride, table = self._install_table(engine, [blob], "flag-clear", limit)
            engine._flag_clear_addr = base
            images.append((idx, engine, base, table))
        self.register_raw_stage(
            "flag_clear", (images[0][2], images[0][3]), images[1:], {"kind": "flag_clear"})

    # -- --run_from_bin ---------------------------------------------------
    # Program images come from programs.bin; these attributes are the images
    # themselves, so they are restored from the bin and never pickled.
    _PROGRAM_ATTRS = frozenset({
        "_vis_program_bytes", "_vis_encoder_program_bytes", "_vis_patch_program_bytes",
        "_vis_worker_programs", "_audio_program_bytes", "_audio_worker_programs",
        "_prefill_program", "_prefill_workers", "_decoder_program", "_decoder_workers",
    })

    def engine_names(self, cores=None) -> dict:
        """Name every engine object a stored value may refer to."""
        names = {"master": self}
        for i, worker in enumerate(self._worker_pool or (), start=1):
            names[f"worker{i}"] = worker
        if cores is not None:
            for i, engine in enumerate(cores.engines):
                names[f"core{i}"] = engine
        return names

    def keep_compile_state(self, key: str, recorder, skip=()) -> None:
        """Remember what a compile call left behind, for --run_from_bin."""
        self._compile_state[key] = recorder.delta(skip=self._PROGRAM_ATTRS | set(skip))

    def apply_compile_state(self, key: str, target=None) -> None:
        for name, value in self._compile_state[key].items():
            setattr(target if target is not None else self, name, value)

    @staticmethod
    def stage_images_from_bin(manifest: dict, payload: bytes) -> dict:
        """{stage: {engine: (dram address, bytes)}} exactly as programs.json records."""
        stages: dict[str, dict] = {}
        for s in manifest["sections"]:
            start = int(s["file_offset"])
            stages.setdefault(s["name"], {})[int(s["engine_index"])] = (
                int(s["dram_base"], 16), payload[start:start + int(s["size"])])
        return stages

    def use_bin_stage(self, stage: str, images: dict, workers: dict, raw: bool = False,
                      metadata: dict | None = None) -> None:
        """Make a stage's programs the ones stored in programs.bin (no compile)."""
        master_addr, master = images[0]
        stage_workers = [(idx, workers[idx], addr, blob)
                         for idx, (addr, blob) in sorted(images.items()) if idx]
        if raw:
            self.register_raw_stage(stage, (master_addr, master), stage_workers,
                                    metadata or {})
            return
        # The same attributes a compile would have set; the stage's metadata and
        # its runner read them.
        if stage == "vision":
            # run_vision_encoder keeps the encoder and the patch program as one
            # contiguous image; the boundary was recorded when it was compiled.
            enc = int(self._compile_state["vision_extra"]["encoder_len"])
            self._vis_program_bytes = master
            self._vis_encoder_program_bytes = master[:enc]
            self._vis_patch_program_bytes = master[enc:]
            self._vis_worker_programs = stage_workers
        elif stage == "audio":
            self._audio_program_bytes = master
            self._audio_worker_programs = stage_workers
        elif stage == "prefill":
            self._prefill_program = (master_addr, master)
            self._prefill_workers = stage_workers
        elif stage == "decode":
            self._decoder_program = (master_addr, master)
            self._decoder_workers = stage_workers
        self._bin_stage_state[stage] = (master_addr, master, stage_workers)

    def compile_or_restore(self, key: str, compile_fn, from_bin: bool,
                           scheduler_stage: str | None = None):
        """Run a compile and remember what it leaves behind, or restore that."""
        if from_bin:
            if scheduler_stage is not None:
                self._ensure_stage_scheduler(scheduler_stage)
            self.apply_compile_state(key)
            return None
        recorder = _state_mod.Recorder(self)
        result = compile_fn()
        self.keep_compile_state(key, recorder)
        return result

    def reserve_isa_gap(self, *stages: str) -> None:
        """Leave room after a stage's program for the writes made at run time.

        The gap starts at the END of the stage's images, not at wherever the
        cursor happens to be: some compilers (audio) do not advance the master
        cursor until the program is uploaded, so a gap added to the cursor would
        sit inside the image and the next stage would be built on top of it.
        """
        gap = _layout_mod.FLAT_ISA_GAP_MIB * 2**20
        align = lambda n: (n + 63) // 64 * 64
        for stage in stages:
            master_addr, master, workers = self._program_stage_state(stage)
            self._next_program_dram_addr = max(
                self._next_program_dram_addr, align(int(master_addr) + len(master)))
            for _idx, worker, addr, blob in workers:
                worker._next_program_dram_addr = max(
                    worker._next_program_dram_addr, align(int(addr) + len(blob)))
        self.allocate_program_dram(gap)
        for worker in self._worker_pool or ():
            worker.allocate_program_dram(gap)

    def install_resident_programs(self, stages, load_raw: bool = False) -> None:
        """Load every stage's master and worker images into DRAM, once."""
        started = time.perf_counter()
        total = 0
        # Every stage keeps its own range on every engine; two images that
        # overlap would silently corrupt the earlier one, so refuse before any DMA.
        spans: dict[int, list[tuple[int, int, str]]] = {}
        for stage in stages:
            master_addr, master, workers = self._program_stage_state(stage)
            spans.setdefault(0, []).append(
                (int(master_addr), int(master_addr) + len(master), stage))
            for idx, _worker, addr, blob in workers:
                spans.setdefault(int(idx), []).append(
                    (int(addr), int(addr) + len(blob), stage))
        for engine_spans in spans.values():
            engine_spans.sort()
            for (lo0, hi0, s0), (lo1, hi1, s1) in zip(engine_spans, engine_spans[1:]):
                if lo1 < hi0:
                    raise RuntimeError(
                        f"{s0} image [0x{lo0:X}, 0x{hi0:X}) overlaps {s1} image "
                        f"starting at 0x{lo1:X}")
        self._guard_resident_dma(self)
        for stage in stages:
            if stage in self._raw_stages:
                # Written while it compiled, or (load_raw) uploaded from the bin now.
                master_addr, master, workers = self._program_stage_state(stage)
                for engine, addr, blob in [(self, master_addr, master)] + [
                        (w, int(a), bytes(b)) for _i, w, a, b in workers]:
                    if load_raw:
                        if engine is not self:
                            self._guard_resident_dma(engine)
                        if engine.dma_write(user_dma_core.DMA_DEVICE_H2C, addr, blob,
                                            len(blob)) != len(blob):
                            raise IOError(f"{stage}: short program DMA at 0x{addr:X}")
                        total += len(blob)
                    key = (stage, "master" if engine is self else f"worker@{id(engine) % 9973}", addr)
                    self._resident_hashes[key] = (engine, len(blob),
                                                  hashlib.sha1(blob).hexdigest())
                self._resident_stages.add(stage)
                continue
            self._reload_program_stage(stage)         # disk bytes, validated
            master_addr, master, workers = self._program_stage_state(stage)
            engine_name = lambda e: "master" if e is self else f"worker@{id(e) % 9973}"
            images = [(self, master_addr, bytes(master))]
            images += [(w, int(a), bytes(b)) for _i, w, a, b in workers]
            for engine, addr, blob in images:
                if engine is not self:
                    self._guard_resident_dma(engine)
                if engine.dma_write(user_dma_core.DMA_DEVICE_H2C, addr, blob, len(blob)) != len(blob):
                    raise IOError(f"{stage}: short program DMA at 0x{addr:X}")
                self._resident_images[id(engine)].setdefault(addr, set()).add(len(blob))
                if stage == "vision" and engine is self:
                    # run_vision_encoder also uploads the encoder alone.
                    self._resident_images[id(engine)][addr].add(
                        len(self._vis_encoder_program_bytes))
                self._resident_hashes[(stage, engine_name(engine), addr)] = (
                    engine, len(blob), hashlib.sha1(blob).hexdigest())
                total += len(blob)
            self._resident_stages.add(stage)
        self._loud(f"  [Program bin] installed {total / 2**20:.1f} MiB of "
                   f"{len(self._resident_stages)} stage image(s) in "
                   f"{time.perf_counter() - started:.1f}s; they stay resident")

    def _master_isa_program_ends(self) -> list[tuple[str, int]]:
        runtime_bytes = 2 * user_dma_core.INSTRUCTION_SIZE_BYTES
        programs = [("allocator", self.get_program_dram_addr())]
        for stage, address_attr, bytes_attr in (
            ("vision", "_vis_program_addr", "_vis_program_bytes"),
            ("audio", "_audio_program_addr", "_audio_program_bytes"),
        ):
            if hasattr(self, address_attr) and hasattr(self, bytes_attr):
                programs.append(
                    (
                        stage,
                        int(getattr(self, address_attr))
                        + len(getattr(self, bytes_attr))
                        + runtime_bytes,
                    )
                )
        for stage, program_attr in (
            ("prefill", "_prefill_program"),
            ("decoder", "_decoder_program"),
        ):
            if hasattr(self, program_attr):
                address, blob = getattr(self, program_attr)
                programs.append((stage, int(address) + len(blob) + runtime_bytes))
        return programs

    def check_master_isa(self) -> None:
        programs = self._master_isa_program_ends()
        stage, end = max(programs, key=lambda item: item[1])
        if end > self.MASTER_ISA_LIMIT:
            raise MemoryError(
                f"{stage} master program ends at 0x{end:X}, beyond the worker ISA base "
                f"0x{self.MASTER_ISA_LIMIT:X}"
            )

    @staticmethod
    def _group_allocs(allocs: list[dict]) -> list[tuple[str, int, int]]:
        """Collapse per-layer carves into one row per kind.

        A 28-layer model makes hundreds of allocations whose names differ only
        by a layer index; the map is only legible once they are summed.
        """
        import re
        groups: dict[str, list[int]] = {}
        for a in allocs:
            key = str(a["what"])
            key = re.sub(r"_L\d+", "", key)            # layer index
            key = re.sub(r"\.n\d+", " N-shard", key)   # per-engine N shard
            key = re.sub(r"\.k\d+", " K-shard", key)   # per-engine K shard
            key = re.sub(r"[._]?core\d+", "", key)      # per-engine stripe
            key = re.sub(r"\.(data|scale)$", "", key)   # blob halves are one thing
            key = re.sub(r"[._]\d+$", "", key)
            key = key.replace("_", " ").strip(". ") or "?"
            groups.setdefault(key, []).append(int(a["size"]))
        rows = [(k, sum(v), len(v)) for k, v in groups.items()]
        rows.sort(key=lambda r: -r[1])
        return rows

    def dram_layout_lines(self) -> list[str]:
        """The whole 8 GiB: windows, private shards, shared pool, tensors.

        Report both payload and the intentionally untouched ISA guard strip.
        """
        MiB = float(2**20)
        arena = self.layout
        ne = arena.num_engines
        win = arena.stride
        out = [
            f"Device: {arena.arena_bytes / 2**30:.0f} GiB, {ne} x "
            f"{win / MiB:.0f} MiB private windows tiling [0x0, 0x{arena.arena_bytes:X}).",
            "",
            "Inside every window, low to high:",
            f"  private weight reserve {arena._private_reserve[0] / MiB:7.0f} MiB",
            f"  shared weights/tensors {(arena.weight_bytes() - arena._private_reserve[0]) / MiB:7.0f} MiB",
            f"  private tensor slice   {arena.tensor_bytes / MiB:7.0f} MiB",
            f"  untouched ISA guard    {arena.isa_guard_bytes / MiB:7.0f} MiB",
            f"  ISA slice              {arena.isa_bytes / MiB:7.0f} MiB",
            "",
            "| core | window base | private | shared wts | tensors | free |",
            "| ---: | :--- | ---: | ---: | ---: | ---: |",
        ]
        priv, sdown, sup, free = (arena.usage(), arena.shared_usage(),
                                  arena.shared_up_usage(), arena.shared_free())
        for i in range(ne):
            out.append(
                f"| {i} | 0x{arena.region(i).base:09X} | {priv[i] / MiB:.1f} MiB "
                f"| {sdown[i] / MiB:.1f} MiB | {sup[i] / MiB:.1f} MiB "
                f"| {free[i] / MiB:.1f} MiB |")
        out.append(
            f"| **all** | | **{sum(priv) / MiB:.0f}** | **{sum(sdown) / MiB:.0f}** "
            f"| **{sum(sup) / MiB:.0f}** | **{sum(free) / MiB:.0f}** |")

        rows = self._group_allocs(arena.weight_allocations())
        if rows:
            out += ["", "PRIVATE per core -- one shard of each weight, never duplicated:", ""]
            for name, total, n in rows[:16]:
                out.append(f"  {name:34s} {total / ne / MiB:8.2f} MiB/core"
                           f"  {total / MiB:9.1f} MiB total  ({n} carves)")
            out.append(f"  {'-' * 34} {'':>8}          {'':>9}")
            out.append(f"  {'in use':34s} {sum(priv) / ne / MiB:8.2f} MiB/core"
                       f"  {sum(priv) / MiB:9.1f} MiB total")
            out.append(f"  {'reserved':34s} "
                       f"{arena._private_reserve[0] / MiB:8.2f} MiB/core")

        rows = self._group_allocs(arena.shared_allocations())
        if rows:
            out += ["", "SHARED WEIGHTS -- read by every core, placed section by "
                    "section across the window tails:", ""]
            for name, total, n in rows[:12]:
                out.append(f"  {name:34s} {total / MiB:9.2f} MiB"
                           f"   ({n} section{'s' if n != 1 else ''})")

        rows = self._group_allocs(arena._shared_up_allocs)
        if rows:
            out += ["", "TENSORS -- KV cache, activations and scratch, carved "
                    "per buffer from the same pool:", ""]
            for name, total, n in rows[:14]:
                out.append(f"  {name:34s} {total / MiB:9.2f} MiB"
                           f"   ({n} buffer{'s' if n != 1 else ''})")
        out += ["",
                f"Context {self.MAX_CONTEXT_SIZE} tokens (prefill and decode share "
                f"one KV cache); prefill allocation {self.PREFILL_MAX_SEQ_LEN} rows, "
                f"run at ceil(seq_len/64)*64."]
        return out

    def isa_usage_lines(self) -> list[str]:
        _stage, end = max(
            self._master_isa_program_ends(), key=lambda item: item[1]
        )
        lines = [
            f"  core 0 ISA: {(end - self.ISA_BASE) / 2**20:.2f} / "
            f"{self.MASTER_ISA_RESERVE / 2**20:.0f} MiB"
        ]
        for idx in range(1, REQUIRED_ENGINES):
            per_stage = self._worker_isa_used.get(idx, {})
            # Stage programs are uploaded immediately before their phase and
            # deliberately overwrite the same worker slice.
            peak = max(per_stage.values(), default=0)
            details = ", ".join(
                f"{name} {size / 2**20:.2f}"
                for name, size in sorted(per_stage.items())
            )
            lines.append(
                f"  core {idx} ISA peak: {peak / 2**20:.2f} / "
                f"{self.layout.isa_bytes / 2**20:.0f} MiB"
                + (f" ({details} MiB)" if details else "")
            )
        return lines

    def describe_dram_map(self) -> str:
        return "\n".join(
            [
                "U55 8-GiB map:",
                self.layout.describe(),
                f"  PARAMS  0x{self.PARAMS_BASE:09X}..0x{self.PARAMS_LIMIT:09X} "
                f"{(self.PARAMS_LIMIT - self.PARAMS_BASE) / 2**20:.0f} MiB "
                "(vision/audio/LM time-shared)",
                f"  TENSOR  0x{self.TENSOR_BASE:09X}..0x{self.TENSOR_LIMIT:09X} "
                f"{(self.TENSOR_LIMIT - self.TENSOR_BASE) / 2**20:.0f} MiB",
                f"  ISA     0x{self.ISA_BASE:09X}..0x{self.MASTER_ISA_LIMIT:09X} "
                f"{(self.MASTER_ISA_LIMIT - self.ISA_BASE) / 2**20:.0f} MiB",
                f"  ISA guard (per core): {self.layout.isa_guard_bytes / 2**20:.0f} MiB "
                "untouched immediately below ISA",
            ]
        )


# Backward-friendly spelling for scripts that follow the older model classes.
Qwen25Omni_UnifiedEngine = Qwen25OmniUnifiedEngine


def _sync_selected_dma_device() -> None:
    """Refresh module-level DMA aliases imported before --dev was resolved."""
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        if "qwen2_5_omni" not in name and not name.endswith("_for_omni"):
            continue
        for attr in ("DMA_DEVICE_H2C", "DMA_DEVICE_C2H", "DMA_DEVICE_USER"):
            if hasattr(module, attr):
                setattr(module, attr, getattr(user_dma_core, attr))


def reset_selected_engines(cores: int = REQUIRED_ENGINES) -> int:
    """Reset and HALT-probe the selected engines without the legacy DRAM test.

    ``user_hw_test.software_reset_test`` intentionally initializes through the
    release-only ``0xfe984d16`` gate and writes a destructive DRAM self-test at
    a legacy fixed address.  Omni has its own explicit map and also supports
    the newer queue-CONFIG images, whose backward-compatible
    matmul/dequantize/argmax path is used here. Validate every build stamp
    before touching hardware, then reset and execute a bare HALT on cores 0-7.
    """
    if cores != REQUIRED_ENGINES:
        raise ValueError(
            f"Qwen2.5-Omni reset requires exactly {REQUIRED_ENGINES} engines"
        )

    engines = [
        UnifiedEngine(
            BASE_ADDR=user_dma_core.UE_0_BASE_ADDR + core * ENGINE_BASE_STRIDE,
            program_dram_base=RESET_PROBE_ISA_ADDR,
            init_unified_engine=False,
        )
        for core in range(cores)
    ]
    versions = [
        int(engine.user_read_reg32(user_dma_core.UE_FPGA_VERSION_ADDR))
        & 0xFFFFFFFF
        for engine in engines
    ]
    if len(set(versions)) != 1:
        got = ", ".join(f"core {core}=0x{version:08x}"
                        for core, version in enumerate(versions))
        raise RuntimeError(f"selected engines report mixed FPGA builds: {got}")

    reset_command = 0x80008000
    for engine in engines:
        engine.write_reg32(user_dma_core.UE_QUEUE_CTRL_ADDR, reset_command)
    deadline = time.monotonic() + 3.0
    for core, engine in enumerate(engines):
        while engine.is_queue_busy() and time.monotonic() < deadline:
            time.sleep(0.001)
        if engine.is_queue_busy():
            raise TimeoutError(f"engine {core} remained busy after software reset")

    # One shared immutable HALT image is sufficient; each core has an
    # independent queue but reads the same global instruction DRAM.
    probe = engines[0]
    probe.clear_inst_id()
    probe.clear_capture_buffer()
    probe.start_capture()
    probe.generate_instruction_halt()
    probe.stop_capture()
    expected = probe.get_capture_instruction_size_bytes()
    written = probe.write_captured_instructions_to_dram(RESET_PROBE_ISA_ADDR)
    if written != expected:
        raise IOError(f"HALT probe DMA wrote {written} of {expected} bytes")
    probe.clear_capture_buffer()

    for core, engine in enumerate(engines):
        engine.start_execute_from_dram(RESET_PROBE_ISA_ADDR)
        engine.wait_queue(3.0)
        if engine.is_queue_busy():
            raise TimeoutError(f"engine {core} failed its post-reset HALT probe")
    return versions[0]


def resolve_engine_config(parser: argparse.ArgumentParser, args) -> dict[str, int]:
    if args.multi_core != REQUIRED_ENGINES:
        parser.error(f"--multi-core must be exactly {REQUIRED_ENGINES}")
    set_dma_device(args.dev)
    _sync_selected_dma_device()
    user_dma_core.configure_clock_from_hardware()
    cores = user_dma_core.ANDROMEDA_CORE_COUNT
    dram = user_dma_core.AVAILABLE_DRAM_SIZE_GB
    if cores is None or cores < REQUIRED_ENGINES:
        parser.error(
            f"Qwen2.5-Omni needs at least 8 available engines; "
            f"HW_INFO reports {cores!r}"
        )
    try:
        require_multicore_dram(REQUIRED_ENGINES, "Qwen2.5-Omni-7B")
        # Same board gate the constructor takes, asked here so an unsupported
        # image fails at argparse like every other hardware requirement rather
        # than after the run lock and the weight cache have been opened.
        tiled_window_bases(REQUIRED_ENGINES, OMNI_WINDOW_BYTES, "Qwen2.5-Omni-7B")
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(user_dma_core.hardware_info_summary())
    print(
        f"Using engines 0-{REQUIRED_ENGINES - 1} of {cores} available on "
        f"{args.dev}; {dram} GiB DRAM"
    )
    return {"multi_core": REQUIRED_ENGINES}


def _resolve_sample(path: str | None, default_path: str, flag: str) -> str | None:
    if path is None:
        return None
    candidates = [path]
    if not os.path.isabs(path):
        candidates += [
            os.path.join(PROJECT_ROOT, path),
            os.path.join(os.path.dirname(default_path), os.path.basename(path)),
        ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise SystemExit(f"{flag}: file not found: {path!r}")


def _load_audio(
    path: str,
    sample_rate: int,
    fraction: float = 1.0,
    target_seconds: float | None = None,
) -> np.ndarray:
    import soundfile as sf
    import torchaudio.functional as audio_functional

    samples, source_rate = sf.read(
        path, dtype="float32", always_2d=True
    )
    if samples.shape[0] == 0:
        raise ValueError(f"audio file is empty: {path}")
    if source_rate <= 0:
        raise ValueError(f"audio file has invalid sample rate {source_rate}")
    mono = torch.from_numpy(samples).mean(dim=1)
    if source_rate != sample_rate:
        mono = audio_functional.resample(mono, source_rate, sample_rate)
    if not torch.isfinite(mono).all().item():
        raise ValueError(f"audio contains NaN or Inf: {path}")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"audio fraction must be in (0, 1], got {fraction}")
    if fraction < 1.0:
        # Truncate at the SOURCE, after resampling: the encoder derives its mel
        # frame count, and therefore its soft-token count, from the sample
        # count, so this is what makes a preset's audio contribution fixed.
        kept = max(int(mono.shape[0] * fraction), sample_rate // 10)
        mono = mono[:kept]
    if target_seconds is not None:
        target_samples = int(round(float(target_seconds) * sample_rate))
        if target_samples <= 0:
            raise ValueError(
                f"audio target duration must be positive, got {target_seconds}"
            )
        # The bundled clip is deterministic benchmark material. Repeat it when
        # it is shorter than the requested camera-query duration, then trim to
        # exactly the requested number of samples.
        if mono.shape[0] < target_samples:
            repeats = (target_samples + mono.shape[0] - 1) // mono.shape[0]
            mono = mono.repeat(repeats)
        mono = mono[:target_samples]
    return mono.contiguous().numpy().astype(np.float32, copy=False)


def _resolve_dummy_prompt(path: str) -> str:
    """Find a --dummy_prompt file: as given, else beside this script.

    Relative paths resolve against the working directory first so a path the
    shell tab-completed behaves as typed; the script-dir fallback is what makes
    the bare flag and ``--dummy_prompt dummy_prompt_4072.md`` work from any cwd.
    """
    candidates = [path] if os.path.isabs(path) else [
        path, os.path.join(SCRIPT_DIR, path)]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "--dummy_prompt file not found; tried " + ", ".join(
            os.path.abspath(candidate) for candidate in candidates))


def _default_prompt(args) -> str:
    if args.dummy_prompt:
        resolved = _resolve_dummy_prompt(args.dummy_prompt)
        with open(resolved, encoding="utf-8") as source:
            text = source.read()
        if not text.strip():
            raise ValueError(f"--dummy_prompt file {resolved} is empty")
        return text
    if args.prompt:
        return args.prompt
    if getattr(args, "target_prefill_tokens", None):
        # Fitted later, once the processor can measure the assembled length.
        return args.prompt_base
    if args.image and args.audio:
        return "First transcribe the audio. Then briefly describe the image."
    if args.image:
        return "Describe the picture in detail."
    if args.audio:
        return "Transcribe the speech exactly."
    return "Explain why the sky is blue in one sentence."


def _prepare_processor_inputs(args, cfg: dict, processor_dir: str):
    """Use the stripped local processor for placeholders and feature masks."""
    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        processor_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    prompt = _default_prompt(args)
    content: list[dict[str, Any]] = []
    images = None
    audio = None

    if args.image:
        size = int(cfg["vision"]["image_size"])
        with Image.open(args.image) as source:
            image = source.convert("RGB").resize(
                (size, size), Image.Resampling.BILINEAR
            )
            frames = max(1, int(getattr(args, "frames", 1) or 1))
            images = [image.copy() for _ in range(frames)]
    if args.audio:
        samples = _load_audio(
            args.audio, int(cfg["audio"]["sample_rate"]),
            fraction=float(getattr(args, "audio_fraction", 1.0) or 1.0),
            target_seconds=getattr(args, "audio_seconds", None))
        audio = [samples]

    # In the joint case, match media order to the requested answer order. This
    # avoids asking the greedy Thinker to jump back over a completed image
    # response before it reaches the short, deterministic transcript check.
    if args.audio:
        content.append({"type": "audio", "audio": args.audio})
    if args.image:
        # One block per --frames camera frame: each gets its own image
        # placeholder run in the tokenized prompt, which _run_vision fills
        # with that frame's own real vision-encoder output, in this same
        # order (see run_prefill's slot-splicing, which is already
        # count-agnostic -- it just fills however many image_token_id
        # positions exist).
        content.extend({"type": "image", "image": args.image} for _ in images)
    content.append({"type": "text", "text": prompt})
    def _assemble(prompt_text: str):
        body = [item for item in content if item["type"] != "text"]
        body.append({"type": "text", "text": prompt_text})
        messages = []
        if args.speak is not None:
            # Qwen's Talker is conditioned on the Thinker's system prompt.
            # The generic chat-template default yields incoherent speech even
            # when the Thinker's decoded text is correct.
            messages.append({"role": "system", "content": SPEECH_SYSTEM_PROMPT})
        messages.append({"role": "user", "content": body})
        text = processor.apply_chat_template(
            messages,
            tokenize=False, add_generation_prompt=True)
        call_kwargs: dict[str, Any] = {
            "text": text, "padding": True, "return_tensors": "pt",
        }
        if images is not None:
            call_kwargs["images"] = images
        if audio is not None:
            call_kwargs["audio"] = audio
        return text, processor(**call_kwargs)

    # --target-prefill-tokens GROWS THE PROMPT to a target prefill length. It
    # has to happen here: only the processor knows how many tokens this
    # particular audio clip and image expand to.
    target = getattr(args, "target_prefill_tokens", None)
    if target and not args.prompt:
        prompt = _fit_prompt_to_prefill(
            processor.tokenizer,
            lambda text: int(_assemble(text)[1]["input_ids"].shape[1]),
            int(target), prompt if args.dummy_prompt else args.prompt_base)
    rendered, processed = _assemble(prompt)

    input_ids = processed["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"processor returned input_ids shape {tuple(input_ids.shape)}")
    attention_mask = processed.get("attention_mask")
    if attention_mask is not None and not torch.all(attention_mask == 1).item():
        raise ValueError("single-request processor output unexpectedly contains text padding")
    tokens = [int(value) for value in input_ids[0].tolist()]
    return processor, processed, tokens, prompt, rendered


def _request_signature(ue, args, context, tokens, processed) -> dict:
    """Everything the compiled programs depend on, so a stored bin is only reused
    for a request of the same shape (token values and pixel values do not matter)."""
    sig = {
        "layout": ue.layout.kind,
        "stages": sorted(
            ["lm"] + (["vision"] if args.image else []) + (["audio"] if args.audio else [])
            + (["talker"] if args.speak is not None and not args.speak_host else [])
            + (["t2w"] if args.speak is not None and not args.speak_host
               and not args.token2wav_host else [])),
        "vision_res": args.vision_res, "frames": int(args.frames),
        "max_context": int(ue.MAX_CONTEXT_SIZE), "profile": bool(args.profile),
        "prefill_tokens": len(context), "prompt_tokens": len(tokens),
        "t2w_max_codes": int(T2W_FPGA_MAX_CODES),
    }
    if args.image:
        sig["image_grid_thw"] = processed["image_grid_thw"].tolist()
        sig["pixel_rows"] = int(processed["pixel_values"].shape[0])
    if args.audio:
        sig["audio_features"] = list(processed["input_features"].shape)
        sig["audio_mask_ones"] = int(processed["feature_attention_mask"].sum())
    return sig


def _prepare_unified(ue: Qwen25OmniUnifiedEngine, args, processed, context,
                     cfg: dict, tokens: list[int], signature: dict | None = None,
                     from_bin: bool = False) -> list[str]:
    """Everything every enabled stage needs, before any stage runs.

    Weights go to their resident pools, each stage's tensors are carved (they
    alias, so this only fixes the addresses the programs are built against),
    every program is compiled into its own ISA range and the whole set is
    published as one programs.bin generation and loaded into DRAM once.

    ``from_bin`` (--run_from_bin) compiles nothing: the stage images are taken
    from the existing programs.bin, the compilers' bookkeeping from its state
    file, and the Talker's and Token2Wav's weights from the speech-weights file.
    The weights of the Thinker, vision and audio still load from params.bin.
    """
    started = time.perf_counter()
    stages: list[str] = []
    saved = images = manifest = payload = None
    speak_fpga = args.speak is not None and not args.speak_host
    bin_path = str(ue._program_bundle.bin_path)
    if from_bin:
        # Reuse the stored programs only if they were built for this configuration
        # (layout, context, which stages, shapes -- the prompt's text does not matter,
        # its token count does). Anything else: build from scratch, as without the flag.
        problem = None
        generation = None
        try:
            with open(ue._program_bundle.json_path) as stream:
                generation = json.load(stream).get("generation_id")
        except (OSError, ValueError):
            problem = f"{bin_path} or its manifest does not exist"
        if problem is None:
            # The request first: it names what differs (context, stages, prompt length).
            problem = _state_mod.reuse_problem(bin_path, generation, signature)
        if problem is None:
            try:
                manifest, payload = ue._program_bundle.load()   # code/layout identity, hashes
            except Exception as exc:  # noqa: BLE001 - e.g. ProgramBundleError: stale identity
                problem = f"programs.bin does not match this code or layout ({exc})"
        if problem is not None:
            print(f"  [run_from_bin] not reusing {os.path.basename(bin_path)}: {problem}\n"
                  "  [run_from_bin] building the programs from scratch", flush=True)
            from_bin = False
            manifest = payload = None
    print("\n--- Unified prepare: weights, tensors, programs (all stages) ---"
          + ("   [programs from programs.bin, no compile]" if from_bin else ""))
    if not from_bin:
        # A build starts from nothing, so no stage of an earlier configuration lingers.
        for stale in (bin_path, str(ue._program_bundle.json_path),
                      _state_mod.state_path(bin_path)):
            if os.path.exists(stale):
                os.unlink(stale)
    if from_bin:
        first = "vision" if args.image else "audio" if args.audio else "prefill"
        ue._ensure_stage_scheduler(first)           # creates the shared worker pool
        saved = _state_mod.load(bin_path, manifest["generation_id"], signature,
                                ue.engine_names())
        images = ue.stage_images_from_bin(manifest, payload)
        ue._compile_state.update(saved["compile_state"])
        ue._stage_tensor_state.update(saved["stage_tensor_state"])
        ue._worker_isa_used.update(saved["worker_isa_used"])
        print(f"  [run_from_bin] {bin_path}: generation {manifest['generation_id'][:12]}, "
              f"{len(images)} stage(s) recorded", flush=True)
    phase = "prepare_bin" if from_bin else "prepare"
    worker_map = lambda: {i: w for i, w in enumerate(ue._worker_pool, start=1)}

    def section_metadata(stage: str) -> dict:
        for s in manifest["sections"]:
            if s["name"] == stage:
                return s["metadata"]
        raise KeyError(stage)

    if args.image:
        _run_vision(ue, processed, args.profile, phase=phase)
        stages.append("vision")
    if args.audio:
        _run_audio(ue, processed, phase=phase)
        stages.append("audio")
    before = dict(ue._dram_addresses)
    ue.lm_weight_init()
    ue.lm_tensor_init()
    layers = int(ue._lm_dims()["NL"])
    ue.compile_or_restore(
        "prefill", lambda: ue.compile_prefill(len(context), profile=args.profile),
        from_bin, "prefill")
    # The decode shards are weights (they are placed in the private windows); build
    # them explicitly so a run that does not compile still has them.
    ue._ensure_decode_shards(ue._ensure_stage_scheduler("decode"), layers)
    ue.compile_or_restore("decode", lambda: ue.compile_decoder(profile=args.profile),
                          from_bin, "decode")
    if not from_bin:
        ue.check_master_isa()
        ue.snapshot_stage_tensors("lm", before)
    stages += ["prefill", "decode"]
    if from_bin:
        ue._packaged_program_stages.update(stages)
        for stage in stages:            # vision, audio, prefill, decode
            ue.use_bin_stage(stage, images[stage], worker_map())
        ue._decode_launch = saved["decode_launch"]
        ue.use_bin_stage("decode_launch", images["decode_launch"], worker_map(), raw=True,
                         metadata=section_metadata("decode_launch"))
        stages.append("decode_launch")
    else:
        ue.install_decode_launch_tables()
        if "decode_launch" in ue._raw_stages:
            stages.append("decode_launch")
        ue.reserve_isa_gap("prefill", "decode")
    persist: dict = {}
    if speak_fpga:
        # The Talker's weights and program are request-independent, so they are
        # staged now too, and so is Token2Wav (built for its longest output).
        from transformers import Qwen2_5OmniConfig
        model_dir = os.path.join(SCRIPT_DIR, cfg["paths"]["hf_model_dir"])
        speech_cfg = Qwen2_5OmniConfig.from_pretrained(model_dir).talker_config
        ue._talker_ctx = _stage_talker(
            ue, args, model_dir, speech_cfg, len(tokens), saved, images,
            section_metadata("talker") if from_bin else None)
        persist.update(ue._talker_ctx["persist"] or {})
        stages.append("talker")
        print("  [Talker FPGA] weights and program resident", flush=True)
        if not args.token2wav_host:
            ue._t2w = _stage_token2wav(
                ue, model_dir, saved, images,
                section_metadata("token2wav") if from_bin else None)
            persist.update(ue._t2w["persist"] or {})
            stages.append("token2wav")
            print(f"  [Token2Wav FPGA] weights and programs resident "
                  f"({ue._t2w['prepare_s']:.1f}s)", flush=True)
    if from_bin:
        ue.use_bin_stage("flag_clear", images["flag_clear"], worker_map(), raw=True,
                         metadata=section_metadata("flag_clear"))
    else:
        ue.install_flag_clear_programs()
    stages.append("flag_clear")
    if not from_bin:
        # One programs.bin generation for every stage.
        ue.store_program_stages(*stages)
    # Every stage image, the Talker's and Token2Wav's included, is loaded into
    # DRAM from the bin now; nothing is uploaded when a stage starts.
    ue.install_resident_programs(stages, load_raw=True)
    if from_bin:
        for name, addr in saved["flag_clear"].items():
            ue.engine_names()[name]._flag_clear_addr = addr
    else:
        _save_bin_state(ue, bin_path, signature, persist)
    for line in ue.isa_usage_lines():
        print(line)
    print(f"  unified prepare done in {time.perf_counter() - started:.1f}s", flush=True)
    return stages


def _save_bin_state(ue, bin_path: str, signature: dict, persist: dict) -> None:
    """Store, beside programs.bin, what --run_from_bin needs to skip the compile."""
    manifest, _payload = ue._program_bundle.load()
    names = ue.engine_names()
    payload = {
        "compile_state": ue._compile_state,
        "stage_tensor_state": ue._stage_tensor_state,
        "worker_isa_used": ue._worker_isa_used,
        "decode_launch": getattr(ue, "_decode_launch", None),
        "flag_clear": {name: engine._flag_clear_addr for name, engine in names.items()
                       if getattr(engine, "_flag_clear_addr", None) is not None},
        "talker": persist.get("talker"),
        "t2w_state": persist.get("t2w_state"),
    }
    _state_mod.check_picklable(
        {k: v for k, v in payload.items() if isinstance(v, dict)}, names)
    _state_mod.save(bin_path, manifest["generation_id"], signature, payload, names)
    size = os.path.getsize(_state_mod.state_path(bin_path)) / 2**10
    print(f"  [Program bin] state for --run_from_bin saved ({size:.0f} KiB)", flush=True)


def _run_vision(
    ue: Qwen25OmniUnifiedEngine, processed, profile: bool = False,
    phase: str = "all",
) -> torch.Tensor | None:
    """Encode 1 or more camera frames and return their embeddings concatenated
    in frame order (matching the same-order image placeholder blocks
    _prepare_processor_inputs emitted, which run_prefill's slot-splicing then
    fills positionally).

    Multiple frames (--frames) currently all carry the SAME --image content --
    the processor was given N copies of one image, so every frame's grid and
    pixel patches are identical by construction. The encoder is therefore
    compiled ONCE, from frame 0's input, and just re-launched (a fresh HALT-
    resume, no recompile, no re-DMA of patch/rotary/window buffers) for each
    later frame -- exactly the real cost of a genuine second/third/fourth
    camera frame of the same shape, which is what --frames is for.
    """
    grid = processed["image_grid_thw"]
    if grid.ndim != 2 or grid.shape[1] != 3:
        raise ValueError(f"malformed image_grid_thw, got shape {tuple(grid.shape)}")
    frames = int(grid.shape[0])
    # The encoder is compiled for ONE patch count -- the tensors and the
    # attention bias are carved for it -- so the grid is checked, but the
    # expected value comes from the configured image geometry rather than a
    # literal: image_size / patch_size per side, squared, is num_patches.
    vis_cfg = ue._cfg["vision"]
    side = int(vis_cfg["image_size"]) // int(vis_cfg["patch_size"])
    if side * side != int(vis_cfg["num_patches"]):
        raise ValueError(
            f"vision config is inconsistent: image_size "
            f"{vis_cfg['image_size']} / patch_size {vis_cfg['patch_size']} "
            f"gives {side}x{side} = {side * side} patches, but num_patches is "
            f"{vis_cfg['num_patches']}"
        )
    vs = int(vis_cfg["num_patches"])
    expected_row = torch.tensor([1, side, side], dtype=grid.dtype)
    for f in range(frames):
        if not torch.equal(grid[f].cpu(), expected_row):
            raise ValueError(
                f"frame {f}: the encoder is carved for {vis_cfg['image_size']}x"
                f"{vis_cfg['image_size']} (image_grid_thw [1,{side},{side}], "
                f"{vis_cfg['num_patches']} patches -> "
                f"{vis_cfg['num_merged_tokens']} soft tokens), got {grid[f].tolist()}"
            )
    pixel_values = processed["pixel_values"]
    if pixel_values.shape[0] != frames * vs:
        raise ValueError(
            f"pixel_values has {pixel_values.shape[0]} patch rows, expected "
            f"{frames} frame(s) x {vs} patches = {frames * vs}"
        )
    # All frames replay the same --image, so every frame's patch chunk must be
    # byte-identical to frame 0's -- confirms the processor didn't apply any
    # per-instance jitter/augmentation that would make the replay-without-
    # re-DMA shortcut below silently wrong.
    frame0 = pixel_values[:vs]
    for f in range(1, frames):
        if not torch.equal(pixel_values[f * vs:(f + 1) * vs], frame0):
            raise ValueError(
                f"frame {f}'s pixel patches differ from frame 0's -- --frames "
                "only supports replaying identical frames of the same --image"
            )
    started = time.perf_counter()
    if phase == "run":
        print(f"\n--- Vision stage ({frames} frame(s)) ---")
        ue.enter_stage_tensors("vision", ue.vision_tensor_init)
    else:
        from_bin = phase == "prepare_bin"
        if phase.startswith("prepare"):
            print("  [Vision] weights and tensors; program "
                  + ("from programs.bin" if from_bin else "compiled"))
        else:
            print(f"\n--- Vision stage ({frames} frame(s)) ---")
        ue.vision_weight_init()
        ue.prepare_encoder_input(frame0, grid[:1])
        before = dict(ue._dram_addresses)
        ue.reset_tensor_dram_addr()
        ue.vision_tensor_init()
        ue.compile_or_restore("vision", lambda: ue.compile_vision_encoder(profile=profile),
                              from_bin, "vision")
        if not from_bin:
            ue._compile_state["vision_extra"] = {
                "encoder_len": len(ue._vis_encoder_program_bytes)}
            ue.check_master_isa()
        if phase.startswith("prepare"):
            if not from_bin:
                ue.snapshot_stage_tensors("vision", before)
                ue.reserve_isa_gap("vision")
            return None
        ue.store_program_stage("vision")
    # run_vision_encoder's own bookkeeping (_vis_latency_us/_vis_wall_s/
    # _vis_gflops/_vis_num_tokens) is PER CALL -- each call OVERWRITES it, with
    # no notion of "N frames in one run". Aggregate across the loop ourselves
    # so write_run_summary (and the TEST_RESULT JSON, which read these same
    # attrs after this function returns) report the real N-frame totals
    # instead of silently keeping just the last frame's own single-frame
    # numbers -- the per-frame FLOP count (fixed at compile time, identical
    # every call since every frame is the same shape) times the frame count.
    per_frame_flops = float(ue._vis_total_flops)
    frame_embeddings = []
    total_latency_us = 0.0
    total_wall_s = 0.0
    for f in range(frames):
        frame_embeddings.append(ue.run_vision_encoder(profile=profile))
        total_latency_us += float(ue._vis_latency_us)
        total_wall_s += float(getattr(ue, "_vis_wall_s", 0.0))
    embeddings = torch.cat(frame_embeddings, dim=0)
    if frames > 1:
        ue._vis_latency_us = total_latency_us
        ue._vis_total_flops = per_frame_flops * frames
        ue._vis_wall_s = total_wall_s
        ue._vis_gflops = (ue._vis_total_flops / (total_latency_us * 1e-6)
                          if total_latency_us > 0 else 0.0)
        ue._vis_num_tokens = int(embeddings.shape[0])
    print(
        f"  vision -> {tuple(embeddings.shape)} in "
        f"{time.perf_counter() - started:.2f}s wall"
    )
    return embeddings


def _run_audio(ue: Qwen25OmniUnifiedEngine, processed, phase: str = "all"):
    started = time.perf_counter()
    if phase == "run":
        print("\n--- Audio stage ---")
        metadata = ue._audio_metadata
        ue.enter_stage_tensors("audio", ue.audio_tensor_init)
    else:
        from_bin = phase == "prepare_bin"
        print(("  [Audio] weights and tensors; program "
               + ("from programs.bin" if from_bin else "compiled"))
              if phase.startswith("prepare") else "\n--- Audio stage ---")
        ue.audio_weight_init()
        metadata = ue.prepare_audio_input(
            processed["input_features"], processed["feature_attention_mask"]
        )
        ue._audio_metadata = metadata
        before = dict(ue._dram_addresses)
        ue.reset_tensor_dram_addr()
        ue.audio_tensor_init()
        ue.compile_or_restore("audio", ue.compile_audio_encoder, from_bin, "audio")
        if not from_bin:
            ue.check_master_isa()
        if phase.startswith("prepare"):
            if not from_bin:
                ue.snapshot_stage_tensors("audio", before)
                ue.reserve_isa_gap("audio")
            return None, metadata
        ue.store_program_stage("audio")
    if os.environ.get("OMNI_VERIFY_IMAGES"):
        bad = ue.verify_resident_images()
        print(f"  [verify] resident images before audio run: "
              f"{'ALL MATCH' if not bad else 'MISMATCH ' + '; '.join(bad)}", flush=True)
    embeddings = ue.run_audio_encoder()
    print(
        f"  audio -> {tuple(embeddings.shape)} in "
        f"{time.perf_counter() - started:.2f}s wall"
    )
    return embeddings, metadata


def _synthesize_speech(ue, args, cfg: dict, prompt_tokens: list[int]) -> dict:
    """Thinker state off the accelerator -> Talker -> Token2Wav -> .wav.

    Prefill processes all prompt tokens except the seed. Decode's first step
    consumes that seed, so its hidden completes the prompt hidden stream;
    subsequent captured steps are the generated reply's hidden stream.
    """
    import torch as _torch

    steps = ue._speech_steps or []
    if not steps:
        raise RuntimeError("--speak: no decode steps were captured")
    if steps[0][0] != prompt_tokens[-1]:
        raise RuntimeError("--speak: first captured token is not the prompt seed")
    reply_steps = steps[1:]
    if not reply_steps:
        raise RuntimeError("--speak: decoder produced no reply tokens to speak")
    H = int(ue.vector_length)
    T = len(prompt_tokens)
    model_dir = os.path.join(SCRIPT_DIR, cfg["paths"]["hf_model_dir"])

    t0 = time.perf_counter()
    prefill_rows = ue.dma_from_accelerator_memory(
        ue.LM_PREFILL_NORM, (T - 1, H)).float()
    prefill_hidden = _torch.cat((prefill_rows, steps[0][1].float()), dim=0).unsqueeze(0)
    step_hidden = _torch.cat([h.float() for _, h in reply_steps], dim=0).unsqueeze(0)
    readback_s = time.perf_counter() - t0

    emb = _talker_mod.ThinkerEmbeddings(model_dir)
    media = {int(cfg["tokens"]["image_token_id"]), int(cfg["tokens"]["audio_token_id"]),
             int(cfg["tokens"]["video_token_id"])}
    prefill_embeds = emb.rows(
        prompt_tokens, zero_at=[i for i, t in enumerate(prompt_tokens) if t in media])
    step_embeds = emb.rows([t for t, _ in reply_steps])

    hs = _talker_mod.HostSpeech(model_dir, speaker=args.speak)
    t1 = time.perf_counter()
    wav = hs.speak(
        input_ids=_torch.tensor([prompt_tokens], dtype=_torch.long),
        prefill_hidden=prefill_hidden, prefill_embeds=prefill_embeds,
        step_hidden=step_hidden, step_embeds=step_embeds,
        first_reply_token=reply_steps[0][0],
        embed_lookup=lambda ids: emb.rows(ids.flatten().tolist()),
    )
    speak_s = time.perf_counter() - t1
    w = wav[0] if isinstance(wav, (tuple, list)) else wav
    out = os.path.join(SCRIPT_DIR, run_summary_filename(args).replace(".md", ".wav"))
    _talker_mod.write_wav(out, w)
    n = int(w.reshape(-1).shape[0])
    print(f"\n[Speak] {args.speak}: {n} samples = "
          f"{n / _talker_mod.SAMPLE_RATE:.2f}s @ {_talker_mod.SAMPLE_RATE} Hz "
          f"-> {os.path.basename(out)}  (FPGA readback {readback_s:.2f}s, "
          f"host talker+vocoder {speak_s:.1f}s)")
    return {"speaker": args.speak, "wav": out, "samples": n,
            "seconds": n / _talker_mod.SAMPLE_RATE,
            "readback_s": readback_s, "host_s": speak_s,
            **(hs.last_metrics or {})}


def _token2wav_on_fpga(vocoder, codes: list[int], spk: dict, layout=None):
    """DiT + BigVGAN on all eight engines, after the Thinker and Talker have finished.

    Neither stage's weights are read again, so this lays its own address map over
    the whole board (see qwen2.5_omni_7b_t2w_cores.py), overwriting whatever the
    earlier stages left there. Every engine is software-reset first, and each region
    of the Token2Wav program is a host-synchronised barrier across the engines.
    """
    import torch as _torch
    t2w_mod = _load_sibling("qwen2_5_omni_7b_token2wav_fpga",
                            "qwen2.5_omni_7b_token2wav_fpga.py")
    cores_mod = _load_sibling("qwen2_5_omni_7b_t2w_cores", "qwen2.5_omni_7b_t2w_cores.py")
    print(f"  [Token2Wav FPGA] {len(codes)} codec tokens -> {len(codes) * 2} mel frames "
          f"on {REQUIRED_ENGINES} engines", flush=True)
    started = time.perf_counter()
    memory_map = layout.t2w_memory_map() if getattr(layout, "kind", "") == "flat" else None
    cores = cores_mod.Cores(REQUIRED_ENGINES, memory_map=memory_map)
    pipe = t2w_mod.Token2WavFpga(
        cores, vocoder, codes=len(codes),
        verbose=bool(os.environ.get("OMNI_T2W_VERBOSE")))
    setup_s = time.perf_counter() - started
    wav, info = pipe.synthesize(
        _torch.tensor([codes], dtype=_torch.long), spk["cond"].float(),
        spk["ref_mel"].float())
    for line in cores_mod.footprint_lines(cores):
        print(line, flush=True)
    print(f"  [Token2Wav FPGA] setup {setup_s:.1f}s, DiT {info['dit_s']:.1f}s, "
          f"BigVGAN {info['bigvgan_s']:.1f}s", flush=True)
    return wav.detach(), {"token2wav_setup_s": setup_s,
                          "token2wav_dit_s": info["dit_s"],
                          "token2wav_bigvgan_s": info["bigvgan_s"],
                          **{f"t2w_{k}": v for k, v in info.items() if k != "mel"}}


def _load_vocoder(model_dir: str):
    from transformers import Qwen2_5OmniConfig, Qwen2_5OmniToken2WavModel
    config = Qwen2_5OmniConfig.from_pretrained(model_dir)
    vocoder = Qwen2_5OmniToken2WavModel(config.token2wav_config)
    vocoder.load_state_dict(
        _talker_mod._load_submodule_state(model_dir, "token2wav"), strict=True)
    return vocoder.float().eval()


_T2W_HOST_BUILDS: dict = {}


def _host_build_token2wav(model_dir: str, plan) -> dict:
    """Build Token2Wav's weights and programs on the CPU (no board involved).

    The engines write to a HostImage, so the result is exactly the bytes that
    belong at each address of the flat map: the weights go to params.bin, the
    programs to programs.bin. Cached so a run that needs both builds it once.
    """
    cores_mod = _load_sibling("qwen2_5_omni_7b_t2w_cores", "qwen2.5_omni_7b_t2w_cores.py")
    t2w_mod = _load_sibling("qwen2_5_omni_7b_token2wav_fpga",
                            "qwen2.5_omni_7b_token2wav_fpga.py")
    layout = FlatLayout(REQUIRED_ENGINES, plan)       # only for its Token2Wav map
    memory_map = layout.t2w_memory_map()
    key = json.dumps(memory_map, sort_keys=True, default=list)
    if key in _T2W_HOST_BUILDS:
        return _T2W_HOST_BUILDS[key]
    started = time.perf_counter()
    vocoder = _load_vocoder(model_dir)
    cores = cores_mod.ImageCores(REQUIRED_ENGINES, memory_map=memory_map)
    pipe = t2w_mod.Token2WavFpga(
        cores, vocoder, codes=T2W_FPGA_MAX_CODES,
        verbose=bool(os.environ.get("OMNI_T2W_VERBOSE")))
    build = {"cores": cores, "pipe": pipe, "memory_map": memory_map, "vocoder": vocoder,
             "built_s": time.perf_counter() - started}
    _T2W_HOST_BUILDS[key] = build
    print(f"  [Token2Wav] weights and programs built on the host in "
          f"{build['built_s']:.1f}s", flush=True)
    return build


def _ensure_speech_params(script_dir: str, cfg: dict, args, plan) -> None:
    """Make sure params.bin carries the Talker's and Token2Wav's weights.

    They are appended once (the Thinker regions are not rewritten) and loaded at
    init like every other weight. This runs before the engine exists because the
    programs.bin identity covers the params manifest.
    """
    speak_fpga = args.speak is not None and not args.speak_host
    if not speak_fpga:
        return
    need_t2w = not args.token2wav_host
    model_dir = os.path.join(script_dir, cfg["paths"]["hf_model_dir"])
    regions = _weight_mod.params_regions(script_dir)
    if "talker" not in regions and "token2wav" in regions:
        _weight_mod.drop_params_regions(script_dir, ["token2wav"])   # keep the order
        regions = _weight_mod.params_regions(script_dir)
    if "talker" not in regions:
        print("  [params] adding the Talker's weights to params.bin (quantized once) ...",
              flush=True)
        weights = _talker_fpga_mod.TalkerWeights(None, model_dir, verbose=False)
        _weight_mod.append_params_region(
            script_dir, "talker", weights.params_items(), meta={"layers": 24})
    if need_t2w and "token2wav" not in regions:
        print("  [params] adding Token2Wav's weights to params.bin ...", flush=True)
        build = _host_build_token2wav(model_dir, plan)
        cores_mod = _load_sibling("qwen2_5_omni_7b_t2w_cores", "qwen2.5_omni_7b_t2w_cores.py")
        cores = build["cores"]
        items = (
            (f"core{e}", "raw", (size,),
             cores.image.read(cores.engines[e]._params_dram_base + off, size))
            for e, off, size in cores_mod.weight_images(cores))
        _weight_mod.append_params_region(
            script_dir, "token2wav", items, meta={"max_codes": T2W_FPGA_MAX_CODES})


def _stage_token2wav(ue, model_dir: str, saved: dict | None = None,
                     images: dict | None = None, metadata: dict | None = None) -> dict:
    """Make Token2Wav resident for its longest supported output, before any stage runs.

    The programs are built for T2W_FPGA_MAX_CODES codec tokens; a shorter
    output runs the same programs with the extra frames masked out (DiT) or
    zero (BigVGAN) and trims the waveform, so nothing needs compiling when the
    Talker finishes.

    Weights come from params.bin's token2wav region and programs from
    programs.bin, like every other stage. A run that builds the programs gets
    them from a CPU build (no board involved); with ``saved`` (--run_from_bin)
    they come from the stored bin and the compilers' state from the state file.
    """
    t2w_mod = _load_sibling("qwen2_5_omni_7b_token2wav_fpga",
                            "qwen2.5_omni_7b_token2wav_fpga.py")
    cores_mod = _load_sibling("qwen2_5_omni_7b_t2w_cores", "qwen2.5_omni_7b_t2w_cores.py")
    from_bin = saved is not None
    started = time.perf_counter()
    persist = None
    if from_bin:
        vocoder = _load_vocoder(model_dir)
        memory_map = ue.layout.t2w_memory_map()
    else:
        build = _host_build_token2wav(model_dir, ue.layout.plan)
        vocoder, memory_map = build["vocoder"], build["memory_map"]
    cores = cores_mod.Cores(REQUIRED_ENGINES, memory_map=memory_map)
    cores.upload_weights = False        # the weights are loaded from params.bin below
    names = ue.engine_names(cores)
    if from_bin:
        state = _state_mod.loads(saved["t2w_state"], names)
    else:
        state = {"pipe": build["pipe"].compile_state(),
                 "cores": cores_mod.compile_state(build["cores"])}
    pipe = t2w_mod.Token2WavFpga(
        cores, vocoder, codes=T2W_FPGA_MAX_CODES, compile=False, state=state["pipe"],
        verbose=bool(os.environ.get("OMNI_T2W_VERBOSE")))
    cores_mod.restore_state(cores, state["cores"])
    # Weights: params.bin -> DRAM (engine 0: the pool; the others: their constants).
    region = ue._read_params_region("token2wav")
    bases = [memory_map["weights_base"]] + [memory_map["consts"][e]
                                             for e in range(1, REQUIRED_ENGINES)]
    with open(region["bin_path"], "rb") as handle:
        for key, section in region["sections"].items():
            e = int(key.removeprefix("core"))
            handle.seek(region["base_offset"] + int(section["offset"]))
            data = handle.read(int(section["size"]))
            if len(data) != int(section["size"]):
                raise IOError(f"params.bin truncated in token2wav section {key}")
            for done in range(0, len(data), 64 * 2**20):
                chunk = data[done:done + 64 * 2**20]
                if cores.engines[e].dma_write(user_dma_core.DMA_DEVICE_H2C,
                                              bases[e] + done, chunk, len(chunk)) != len(chunk):
                    raise IOError(f"token2wav weights: short DMA to engine {e}")
    # Programs: the stored images (from the bin) or the ones just built.
    if from_bin:
        ue.use_bin_stage("token2wav", images["token2wav"],
                         {e: cores.engines[e] for e in range(REQUIRED_ENGINES)},
                         raw=True, metadata=metadata)
    else:
        built = build["cores"]
        program_images = [
            (e, cores.engines[e], base, built.image.read(base, size))
            for e, base, size in cores_mod.program_ranges(built, memory_map)]
        ue.register_raw_stage(
            "token2wav", (program_images[0][2], program_images[0][3]), program_images[1:],
            {"max_codes": T2W_FPGA_MAX_CODES, "regions": len(built.regions),
             "region_sizes_engine0": [size for _addr, size in
                                      (r[0] for r in built.regions)]})
        persist = {"t2w_state": _state_mod.dumps(state, names)}
    for line in cores_mod.footprint_lines(cores):
        print(line, flush=True)
    return {"pipe": pipe, "cores": cores, "cores_mod": cores_mod, "persist": persist,
            "prepare_s": time.perf_counter() - started}


def _run_token2wav(ue, codes: list[int], spk: dict):
    import torch as _torch
    t2w = ue._t2w
    pipe = t2w["pipe"]
    print(f"  [Token2Wav FPGA] {len(codes)} codec tokens -> {len(codes) * 2} mel frames "
          f"on {REQUIRED_ENGINES} engines (programs built for {pipe.max_codes}; shorter output is masked and trimmed)",
          flush=True)
    pipe.set_codes(len(codes))
    pipe.reset_state()
    wav, info = pipe.synthesize(
        _torch.tensor([codes], dtype=_torch.long), spk["cond"].float(),
        spk["ref_mel"].float())
    print(f"  [Token2Wav FPGA] setup 0.0s (prepared in {t2w['prepare_s']:.1f}s before "
          f"any stage), DiT {info['dit_s']:.1f}s, BigVGAN {info['bigvgan_s']:.1f}s",
          flush=True)
    return wav.detach(), {"token2wav_setup_s": 0.0,
                          "token2wav_prepared_s": t2w["prepare_s"],
                          "token2wav_dit_s": info["dit_s"],
                          "token2wav_bigvgan_s": info["bigvgan_s"],
                          **{f"t2w_{k}": v for k, v in info.items() if k != "mel"}}


def _stage_talker(ue, args, model_dir: str, speech_cfg, prompt_len: int,
                  saved: dict | None = None, images: dict | None = None,
                  metadata: dict | None = None) -> dict:
    """Stage the Talker's weights, build its tensors and compile its step.

    Everything here depends on the request only through the prompt length, so
    the unified prepare calls it before any stage runs; the legacy flow calls
    it when the Talker starts. With ``saved`` (--run_from_bin) the weights are
    restored from the speech-weights file and the step from programs.bin.
    """
    import torch as _torch
    from_bin = saved is not None
    # The Thinker's scratch and (windowed map only) shared weights are released;
    # its private projection shards stay, and the speech map reserved room for
    # the Talker's column shards beside them.
    ue.reset_tensor_dram_addr()
    if ue.layout.evicts_weights:
        ue.reset_params_dram_addr()
    scheduler = ue._ensure_stage_scheduler("talker")
    if not from_bin:
        scheduler.preclear_flags()
    weights = _talker_fpga_mod.TalkerWeights(ue, model_dir, scheduler=scheduler)
    # The weights come from params.bin's talker region in every run: its IF4
    # matrices are already quantized, staging only slices them into the shards.
    weights.stage(region=ue._read_params_region("talker"))
    before = dict(ue._dram_addresses)
    codec_embed = weights._tensor("talker.model.embed_tokens.weight").to(
        _torch.bfloat16)
    prefix_len = prompt_len + 2        # prompt rows + BOS + the first reply row
    max_codec_tokens = min(4096, ue.MAX_CONTEXT_SIZE - prefix_len)
    if max_codec_tokens < 1:
        raise ValueError("prompt leaves no context for FPGA Talker codec tokens")
    if not args.token2wav_host:
        # FPGA Token2Wav is verified up to this many codec tokens; speech is cut here
        # rather than handed to the CPU vocoder.
        max_codec_tokens = min(max_codec_tokens, T2W_FPGA_MAX_CODES)
    max_ctx = ((prefix_len + max_codec_tokens + 63) // 64) * 64
    runner = _talker_fpga_mod.TalkerRunner(
        ue, weights, max_ctx=max_ctx, scheduler=scheduler)

    def reinit() -> None:
        runner._alloc_tensors()
        runner.alloc_attention_scratch(aligned=max_ctx)
        runner.zero_state()
        runner.build_rope_table()

    runner.alloc_attention_scratch(aligned=max_ctx)
    runner.zero_state()
    runner.build_rope_table()
    persist = None
    if from_bin:
        for key, value in saved["talker"]["runner_state"].items():
            setattr(runner, key, value)
        ue.use_bin_stage("talker", images["talker"],
                         {i: w for i, w in enumerate(ue._worker_pool, start=1)},
                         raw=True, metadata=metadata)
    else:
        recorder = _state_mod.Recorder(runner)
        runner.compile_reusable_step()
        persist = {"talker": {"runner_state": recorder.delta()}}
        if ue.layout.resident_programs:
            ue.snapshot_stage_tensors("talker", before)
            # The step is already in DRAM (it is written as it is compiled); read it
            # back so the one programs.bin generation holds it too.
            body = runner._launch_end - runner._program_addr    # body, spare slot, table
            workers = [
                (idx, w, int(addr),
                 ue.read_back(w, int(addr), w.get_program_dram_addr() - int(addr)))
                for idx, (w, addr) in enumerate(
                    zip(scheduler.workers, runner._worker_addrs), start=1)]
            ue.register_raw_stage(
                "talker",
                (runner._program_addr, ue.read_back(ue, runner._program_addr, body)),
                workers,
                {"layers": 24, "max_ctx": max_ctx, "prefix_len": prefix_len,
                 "preamble_addr": f"0x{runner._preamble_addr:X}"})
            ue.reserve_isa_gap()
    return {"scheduler": scheduler, "weights": weights, "runner": runner,
            "codec_embed": codec_embed, "prefix_len": prefix_len,
            "max_codec_tokens": max_codec_tokens, "max_ctx": max_ctx,
            "reinit": reinit, "persist": persist}


def _synthesize_speech_fpga(ue, args, cfg: dict, prompt_tokens: list[int]) -> dict:
    """Run all Talker matmuls on eight engines, then vocode codec IDs on CPU.

    The same eight queues serve Thinker and Talker, so the Talker starts after
    Thinker decode has finished. This keeps ISA/flag ownership unambiguous and
    lets the two stages reuse their shared tensor/weight pool safely.
    """
    import torch as _torch
    from transformers import Qwen2_5OmniConfig, Qwen2_5OmniToken2WavModel

    steps = ue._speech_steps or []
    if not steps or steps[0][0] != prompt_tokens[-1]:
        raise RuntimeError("FPGA Talker needs the Thinker seed hidden row")
    replies = steps[1:]
    if not replies:
        raise RuntimeError("FPGA Talker needs at least one reply token")
    model_dir = os.path.join(SCRIPT_DIR, cfg["paths"]["hf_model_dir"])
    speech_cfg = Qwen2_5OmniConfig.from_pretrained(model_dir).talker_config
    spk = _torch.load(os.path.join(model_dir, "spk_dict.pt"),
                      map_location="cpu", weights_only=False)[args.speak]
    bos = int(spk["bos_token"])

    started = time.perf_counter()
    H, T = int(ue.vector_length), len(prompt_tokens)
    prefill_rows = ue.dma_from_accelerator_memory(
        ue.LM_PREFILL_NORM, (T - 1, H)).float()
    hidden = _torch.cat((prefill_rows, steps[0][1].float().reshape(1, H)), dim=0)
    embed = _talker_mod.ThinkerEmbeddings(model_dir)
    media = {int(cfg["tokens"]["image_token_id"]),
             int(cfg["tokens"]["audio_token_id"]),
             int(cfg["tokens"]["video_token_id"])}
    prefill_embeds = embed.rows(
        prompt_tokens,
        zero_at=[i for i, token in enumerate(prompt_tokens) if token in media],
    ).reshape(T, H)
    prefix = [row.to(_torch.bfloat16) for row in hidden + prefill_embeds.float()]
    prefix.append(embed.rows([bos]).reshape(H).to(_torch.bfloat16))
    first_token, first_hidden = replies[0]
    prefix.append((first_hidden.float().reshape(H)
                   + embed.rows([first_token]).float().reshape(H)).to(_torch.bfloat16))
    readback_s = time.perf_counter() - started

    # After the Thinker finishes, its scratch and shared weights can be
    # released. Its private projection shards remain resident; the speech map
    # reserved enough additional private space for Talker's column shards.
    ctx = getattr(ue, "_talker_ctx", None)
    if ctx is None:
        ctx = _stage_talker(ue, args, model_dir, speech_cfg, T)
    else:
        # Weights and program are already resident; only the tensors (aliased
        # with the other stages') need their contents back.
        ue.enter_stage_tensors("talker", ctx["reinit"])
        ctx["scheduler"].preclear_flags()
    scheduler, weights, runner = ctx["scheduler"], ctx["weights"], ctx["runner"]
    codec_embed = ctx["codec_embed"]
    prefix[-2] += codec_embed[int(speech_cfg.tts_codec_pad_token_id)]
    prefix[-1] += codec_embed[int(speech_cfg.tts_codec_start_token_id)]
    prefix_len = len(prefix)
    if prefix_len != ctx["prefix_len"]:
        raise RuntimeError(
            f"Talker was built for {ctx['prefix_len']} prefix rows, got {prefix_len}")
    max_codec_tokens, max_ctx = ctx["max_codec_tokens"], ctx["max_ctx"]

    R = _talker_fpga_mod.TalkerRunner
    layer_mm = (2 * R.H * R.Q_SIZE + 2 * 2 * R.H * R.KV_SIZE + 2 * R.Q_SIZE * R.H
                + 3 * 2 * R.H * R.MLP)
    step_mm = 2 * R.THINKER_H * R.H + 24 * layer_mm + 2 * R.H * R.VOCAB
    flops = {"model": 0, "issued": 0, "steps": 0}
    if abs(runner.step_flops_fixed - step_mm) > 0.01 * step_mm:
        print(f"  [Talker FPGA] warning: emitted per-step FLOPs {runner.step_flops_fixed:,} "
              f"differ from the analytic projection count {step_mm:,}", flush=True)

    def count_step(pos: int) -> None:
        # Attention: QK^T and PV for the 12 query heads over the live KV rows (model) or
        # over the 64-aligned rows the kernel actually reads (issued).
        aligned = ((pos + 64) // 64) * 64
        flops["model"] += step_mm + 24 * 4 * R.QH * R.AHD * (pos + 1)
        # Issued = what the emitted program reports for every engine's shards plus
        # attention at the 64-aligned KV length actually read.
        flops["issued"] += (runner.step_flops_fixed
                            + runner.attn_flops_per_aligned * aligned)
        flops["steps"] += 1

    total_hw_us = 0.0
    prefix_hw_us = 0.0
    prefix_started = time.perf_counter()
    logits = None
    for pos, row in enumerate(prefix):
        logits, hw_us = runner.run_step(row, pos)
        count_step(pos)
        total_hw_us += hw_us
        prefix_hw_us += hw_us
        if (pos + 1) % 64 == 0 or pos + 1 == prefix_len:
            print(f"  [Talker FPGA] conditioned {pos + 1}/{prefix_len} prefix rows",
                  flush=True)
    prefix_wall_s = time.perf_counter() - prefix_started
    setup_wall_s = prefix_started - started

    def sample(logit_row, previous: list[int]) -> int:
        score = logit_row.float()
        score[int(speech_cfg.tts_codec_start_token_id)] = -float("inf")
        for token in set(previous):
            if score[token] > 0:
                score[token] /= 1.05
            else:
                score[token] *= 1.05
        values, indices = _torch.topk(score / 0.9, k=40)
        probs = _torch.softmax(values, dim=0)
        probs[(probs.cumsum(0) - probs) >= 0.8] = 0
        probs /= probs.sum()
        return int(indices[_torch.multinomial(probs, 1)])

    codes: list[int] = []
    decode_started = time.perf_counter()
    eos = {8292, 8294}
    truncated = False
    for idx in range(max_codec_tokens):
        code = sample(logits, codes)
        if code in eos:
            break
        codes.append(code)
        if idx + 1 >= max_codec_tokens:
            truncated = not args.token2wav_host and max_codec_tokens == T2W_FPGA_MAX_CODES
            break
        if idx + 1 < len(replies):
            token, next_hidden = replies[idx + 1]
            condition = next_hidden.float().reshape(H) + embed.rows([token]).float().reshape(H)
        else:
            special = (int(speech_cfg.tts_text_end_token_id)
                       if idx + 1 == len(replies)
                       else int(speech_cfg.tts_text_pad_token_id))
            condition = embed.rows([special]).float().reshape(H)
        row = (codec_embed[code].float() + condition).to(_torch.bfloat16)
        logits, hw_us = runner.run_step(row, prefix_len + idx)
        count_step(prefix_len + idx)
        total_hw_us += hw_us
        if len(codes) % 64 == 0:
            print(f"  [Talker FPGA] generated {len(codes)} codec tokens",
                  flush=True)
    decode_wall_s = time.perf_counter() - decode_started
    talker_wall_s = prefix_wall_s + decode_wall_s
    talker_total_wall_s = time.perf_counter() - started
    if not codes:
        raise ValueError("FPGA Talker emitted EOS before any codec token")
    print(f"  [Talker FPGA] {len(codes)} codec tokens in {talker_wall_s:.1f}s "
          f"active (setup {setup_wall_s:.1f}s, prefix {prefix_wall_s:.1f}s, "
          f"codec decode {decode_wall_s:.1f}s); "
          f"starting {'CPU' if args.token2wav_host else 'FPGA'} Token2Wav", flush=True)
    if truncated:
        print(f"  [Speak] the Talker reached the {T2W_FPGA_MAX_CODES}-token FPGA Token2Wav "
              f"limit (about {T2W_FPGA_MAX_CODES / 50:.0f} s of audio) before emitting its "
              "end token; speech is cut here", flush=True)

    vocoder_started = time.perf_counter()
    prepared_t2w = getattr(ue, "_t2w", None)
    if prepared_t2w is None:
        vocoder = _load_vocoder(model_dir)
    token2wav_device = "Host CPU"
    t2w_detail: dict = {}
    if not args.token2wav_host:
        for line in getattr(ue.layout, "peak_lines", lambda: [])():
            print(line, flush=True)
        if prepared_t2w is not None:
            wav, t2w_detail = _run_token2wav(ue, codes, spk)
        else:
            wav, t2w_detail = _token2wav_on_fpga(vocoder, codes, spk, ue.layout)
        token2wav_device = f"FPGA ({REQUIRED_ENGINES} cores, bf16)"
    else:
        with _torch.no_grad():
            wave = vocoder(
                _torch.tensor([codes], dtype=_torch.long),
                conditioning=spk["cond"].float(),
                reference_mel=spk["ref_mel"].float(),
            )
        wav = wave[0] if isinstance(wave, (tuple, list)) else wave
    token2wav_s = time.perf_counter() - vocoder_started
    out = os.path.join(SCRIPT_DIR, run_summary_filename(args).replace(".md", ".wav"))
    _talker_mod.write_wav(out, wav)
    samples = int(wav.reshape(-1).numel())
    print(f"  [Speak] wrote {os.path.basename(out)}: "
          f"{samples / _talker_mod.SAMPLE_RATE:.2f}s audio", flush=True)
    return {
        "speaker": args.speak, "wav": out, "samples": samples,
        "seconds": samples / _talker_mod.SAMPLE_RATE,
        "readback_s": readback_s,
        "talker_device": "FPGA (8 cores)", "talker_wall_s": talker_wall_s,
        "talker_hw_us": total_hw_us, "talker_prefix_wall_s": prefix_wall_s,
        "talker_setup_wall_s": setup_wall_s,
        "talker_total_wall_s": talker_total_wall_s,
        "talker_prefix_hw_us": prefix_hw_us,
        "talker_decode_wall_s": decode_wall_s,
        "talker_decode_hw_us": total_hw_us - prefix_hw_us,
        "codec_tokens": len(codes),
        "token2wav_device": token2wav_device, "token2wav_wall_s": token2wav_s,
        "codec_truncated": truncated,
        "talker_flops_model": flops["model"], "talker_flops_issued": flops["issued"],
        "talker_steps": flops["steps"], "peak_gflops": float(ue.vis_peak_gflops()),
        **t2w_detail,
        "decode_talker_overlap_s": 0.0,
    }


def _prepare_streamed_speech(ue, args, cfg: dict, prompt_tokens: list[int]):
    """Read the prompt state and load Talker before concurrent FPGA decode."""
    import torch as _torch

    started = time.perf_counter()
    H = int(ue.vector_length)
    T = len(prompt_tokens)
    prefill_rows = ue.dma_from_accelerator_memory(
        ue.LM_PREFILL_NORM, (T - 1, H)).float()
    readback_s = time.perf_counter() - started
    model_dir = os.path.join(SCRIPT_DIR, cfg["paths"]["hf_model_dir"])
    emb = _talker_mod.ThinkerEmbeddings(model_dir)
    media = {int(cfg["tokens"]["image_token_id"]),
             int(cfg["tokens"]["audio_token_id"]),
             int(cfg["tokens"]["video_token_id"])}
    prefill_embeds = emb.rows(
        prompt_tokens,
        zero_at=[i for i, t in enumerate(prompt_tokens) if t in media],
    )
    lookup = lambda ids: emb.rows(ids.flatten().tolist())
    hs = _talker_mod.HostSpeech(model_dir, speaker=args.speak)
    stream = _talker_mod.ReplyStream(
        lookup, hs.talker.text_eos_token, hs.talker.text_pad_token)
    ue._speech_step_callback = stream.push
    return (stream, hs, _torch.tensor([prompt_tokens], dtype=_torch.long),
            prefill_rows, prefill_embeds, lookup, readback_s, started)


def _finish_streamed_speech(args, session, wav, decode_started: float,
                            decode_finished: float) -> dict:
    stream, hs, _, _, _, _, readback_s, started = session
    w = wav
    w = w[0] if isinstance(w, (tuple, list)) else w
    out = os.path.join(SCRIPT_DIR, run_summary_filename(args).replace(".md", ".wav"))
    _talker_mod.write_wav(out, w)
    n = int(w.reshape(-1).shape[0])
    talker_started = getattr(stream, "talker_started_at", decode_finished)
    talker_finished = getattr(stream, "talker_finished_at", decode_finished)
    overlap = max(0.0, min(talker_finished, decode_finished)
                  - max(talker_started, decode_started))
    finished = time.perf_counter()
    host_s = finished - talker_started
    setup_s = talker_started - started
    print(f"\n[Speak] {args.speak}: {n} samples = "
          f"{n / _talker_mod.SAMPLE_RATE:.2f}s @ {_talker_mod.SAMPLE_RATE} Hz "
          f"-> {os.path.basename(out)}  (FPGA readback {readback_s:.2f}s, "
          f"host setup {setup_s:.1f}s, Talker + vocoder {host_s:.1f}s, "
          f"decode/Talker overlap {overlap:.2f}s)")
    return {"speaker": args.speak, "wav": out, "samples": n,
            "seconds": n / _talker_mod.SAMPLE_RATE,
            "readback_s": readback_s,
            "host_s": host_s,
            "host_setup_s": setup_s,
            "decode_talker_overlap_s": overlap,
            **(hs.last_metrics or {})}


def _result_mode(args) -> str:
    if args.image and args.audio:
        return "image+audio"
    if args.image:
        return "image"
    if args.audio:
        return "audio"
    return "text"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Qwen2.5-Omni-7B Thinker: text/image/audio input to text on "
            "the eight-engine U55"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""examples:
  python {os.path.basename(__file__)} --multi-core 8 --prompt "If x + 3 = 5, what is x?"
  python {os.path.basename(__file__)} --multi-core 8 --dummy_prompt
  python {os.path.basename(__file__)} --multi-core 8 --dummy_prompt dummy_prompt_4072.md
  python {os.path.basename(__file__)} --multi-core 8 --image
  python {os.path.basename(__file__)} --multi-core 8 --audio
""",
    )
    parser.add_argument("--dev", default="xdma0", help="DMA device (default xdma0)")
    parser.add_argument(
        "--vision-res",
        choices=sorted(VISION_RESOLUTIONS),
        default=DEFAULT_VISION_RES,
        help=(
            "vision input resolution; the encoder is carved for one patch "
            "count. small=336x336 -> 144 soft tokens (default), "
            "medium=896x896 -> 1024 soft tokens. The soft tokens are charged "
            "against the context, so medium spends half of it on one image."
        ),
    )
    parser.add_argument(
        "--multi-core",
        nargs="?",
        const=REQUIRED_ENGINES,
        default=REQUIRED_ENGINES,
        type=int,
        help="engine count; this model requires exactly 8",
    )
    parser.add_argument(
        "--layout", choices=("windowed", "flat"), default="windowed",
        help="DRAM map: eight private 1 GiB windows (default), or one "
             "consecutive 8 GiB range where every stage's weights stay resident",
    )
    parser.add_argument(
        "--run_from_bin", "--run-from-bin", action="store_true",
        help="with --layout flat: do not compile. Load the stage programs from the "
             "existing programs bin (and the compilers' state and the speech weights "
             "stored beside it), checked against this request; a bin built for a "
             "different request is refused. Build it by running once without the flag.",
    )
    parser.add_argument(
        "--max-context", type=int, default=None,
        help="with --layout flat: build for at most this many tokens; by default "
             "the largest context the DRAM left after every stage's weights allows",
    )
    prompt_source = parser.add_mutually_exclusive_group()
    prompt_source.add_argument("--prompt", default=None, help="user text prompt")
    prompt_source.add_argument(
        "--dummy_prompt", "--dummy-prompt",
        nargs="?", const=DEFAULT_DUMMY_PROMPT, default=None, metavar="FILE",
        help="read the user text verbatim from a file instead of --prompt. "
             f"Bare flag uses {DEFAULT_DUMMY_PROMPT} beside this script; pass "
             "a path to use another file (resolved against the working "
             "directory first, then beside this script), e.g. "
             "--dummy_prompt dummy_prompt_4072.md. The file's own token count "
             "is what gets prefilled -- no filler is added unless "
             "--target-prefill-tokens is also given.",
    )
    parser.add_argument(
        "--speak", nargs="?", const=DEFAULT_SPEAKER, default=None,
        metavar="SPEAKER",
        help=("synthesise speech for the reply (8-core FPGA Talker + 8-core FPGA Token2Wav). Bare "
              f"--speak uses {DEFAULT_SPEAKER}; the other voice is Ethan. "
              "Writes a .wav next to the run summary."),
    )
    parser.add_argument(
        "--speak-host", action="store_true",
        help="use the previous streaming CPU Talker instead of the FPGA Talker",
    )
    parser.add_argument(
        "--token2wav-host", action="store_true",
        help=("run Token2Wav (DiT + BigVGAN) on the CPU in FP32 instead of the eight FPGA "
              f"engines. By default --speak is FPGA end to end, and speech is capped at "
              f"{T2W_FPGA_MAX_CODES} codec tokens (about {T2W_FPGA_MAX_CODES / 50:.0f} s) "
              "because that is the length FPGA Token2Wav is verified for."),
    )
    parser.add_argument(
        "--target-prefill-tokens",
        type=int,
        default=None,
        help="grow the prompt (with --prompt-base plus generic filler text) "
             "until the assembled prefill reaches exactly this many tokens; "
             "ignored if --prompt is given. Fixed-shape workload presets "
             "(voice-command / single-camera / multi-camera, ...) live in "
             "benchmark.py, which drives this flag. With --dummy_prompt, "
             "grow the contents of dummy_prompt.md instead.",
    )
    parser.add_argument(
        "--prompt-base",
        default=DEFAULT_PROMPT_BASE,
        help="instruction text --target-prefill-tokens pads with filler to "
             "reach the target length",
    )
    parser.add_argument(
        "--audio-fraction",
        type=float,
        default=1.0,
        help="trim loaded audio to this fraction of its samples before "
             "encoding, e.g. 0.5 for half length (default 1.0, full clip)",
    )
    parser.add_argument(
        "--audio-seconds",
        type=float,
        default=None,
        help="repeat/trim loaded audio to exactly this many seconds before "
             "encoding (default: use the clip as-is, subject to "
             "--audio-fraction)",
    )
    parser.add_argument(
        "--image",
        nargs="?",
        const=DEFAULT_IMAGE,
        default=None,
        help=f"optional image; bare --image uses {os.path.basename(DEFAULT_IMAGE)}",
    )
    parser.add_argument(
        "--frames", type=int, default=1, metavar="N",
        help="with --image, encode this many camera frames in one real "
             "continuous run (a multi-camera workload) instead of one. Each "
             "frame is a genuine separate FPGA vision-encoder invocation "
             "(real DMA, real compute) and gets its own image placeholder "
             "block in the prompt; all frames currently replay the SAME "
             "--image (identical pixel content), not N distinct images -- "
             "the vision/prefill cost this measures is real either way, "
             "since each invocation actually runs. Requires --image.",
    )
    parser.add_argument(
        "--audio",
        nargs="?",
        const=DEFAULT_AUDIO,
        default=None,
        help=f"optional audio; bare --audio uses {os.path.basename(DEFAULT_AUDIO)}",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="greedy generation cap (default 128)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="compile the vision encoder, prefill and decoder with per-phase "
             "HALT checkpoints and report a per-phase FPGA-latency breakdown "
             "instead of generating. Read the share column: it says which "
             "phase is worth sharding next. The audio encoder has no "
             "checkpoints and is reported at stage level only.",
    )
    parser.add_argument(
        "--summary",
        default=None,
        metavar="PATH",
        help="write the run-summary Markdown here instead of the default "
             "config-named file next to this script",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="skip writing the run-summary Markdown",
    )
    return parser


def _speech_performance_lines(speech: dict) -> list[str]:
    """Work, time, throughput and % of peak for the speech stages (empty if not recorded)."""
    if speech.get("talker_flops_model") is None:
        return []
    peak = float(speech["peak_gflops"])
    rows = []

    def row(name, device, model_f, issued_f, secs, throughput):
        eff = model_f / secs / 1e9 if secs else 0.0
        iss = issued_f / secs / 1e9 if secs else 0.0
        rows.append(
            f"| {name} | {device} | {model_f / 1e9:,.1f} | {issued_f / 1e9:,.1f} | "
            f"{secs:.2f} s | {eff:,.1f} | {iss:,.1f} | "
            f"{100.0 * eff / peak if peak else 0.0:.1f}% | "
            f"{100.0 * iss / peak if peak else 0.0:.1f}% | {throughput} |")
        return model_f, issued_f, secs

    tk_secs = float(speech["talker_hw_us"]) / 1e6
    codec = int(speech["codec_tokens"])
    dec_secs = float(speech["talker_decode_hw_us"]) / 1e6
    parts = [row("Talker (prefix + decode)", speech["talker_device"],
                 float(speech["talker_flops_model"]), float(speech["talker_flops_issued"]),
                 tk_secs, f"{codec / dec_secs:.1f} codec tok/s (decode, HW)" if dec_secs else "—")]
    if speech.get("t2w_dit_flops_model") is not None:
        evals = int(speech["t2w_evaluations"])
        frames = int(speech["t2w_mel_frames"])
        dit_s = float(speech["t2w_dit_device_s"])
        big_s = float(speech["t2w_bigvgan_device_s"])
        parts.append(row("Token2Wav DiT", speech["token2wav_device"],
                         float(speech["t2w_dit_flops_model"]),
                         float(speech["t2w_dit_flops_issued"]), dit_s,
                         f"{evals / dit_s:.1f} evaluations/s ({frames} frames)" if dit_s else "—"))
        parts.append(row("Token2Wav BigVGAN", speech["token2wav_device"],
                         float(speech["t2w_bigvgan_flops_model"]),
                         float(speech["t2w_bigvgan_flops_issued"]), big_s,
                         f"{float(speech['seconds']) / big_s:.2f}x real time" if big_s else "—"))
    tm = sum(p[0] for p in parts)
    ti = sum(p[1] for p in parts)
    ts = sum(p[2] for p in parts)
    row("**Speech total**", "", tm, ti, ts,
        f"{float(speech['seconds']) / ts:.2f}x real time" if ts else "—")
    return [
        "### Performance (matmul-class FLOPs)",
        "",
        f"Peak is {peak:.1f} GFLOPS (all eight engines). **Model** FLOPs are the work the "
        "network defines for this input; **issued** FLOPs include what padding makes the "
        "engines do (64-row attention windows, 64-aligned channels, rows rounded up to the "
        "chunk size). Talker time is the on-device latency counter summed over every "
        "step; Token2Wav time is the wall time the engines spent running regions "
        "(host-side constants and Runge-Kutta bookkeeping are excluded). Elementwise ops "
        "(norms, activations, FIR filters) are not counted.",
        "",
        "| Stage | Device | Model GFLOP | Issued GFLOP | Device time | Effective GFLOPS "
        "| Issued GFLOPS | % of peak (effective) | % of peak (issued) | Throughput |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
        *rows,
        "",
    ]


def run_summary_filename(args) -> str:
    """Per-run summary filename encoding the CLI config, e.g.

    ``--dev xdma0 --image --multi-core 8`` ->
    ``qwen2.5_omni_7b_test_xdma0_image_multi-core_8.md``.
    ``--speak Chelsie`` adds ``speak_Chelsie`` before the core-count tag;
    the matching WAV uses the same stem.

    Device and mode are always present, in that order; a profile run is tagged
    so its phase breakdown never overwrites a generation run's summary. Callers
    driving distinct fixed shapes at the same mode (e.g. benchmark.py's
    presets) pass ``--summary`` explicitly rather than relying on this name.
    """
    parts = ["qwen2.5_omni_7b_test", args.dev, _result_mode(args)]
    if args.dummy_prompt:
        # Name the file, so runs from different dummy prompts do not overwrite
        # one another's summaries.
        stem = os.path.splitext(os.path.basename(args.dummy_prompt))[0]
        parts.append("dummy-prompt" if stem == "dummy_prompt" else stem)
    if getattr(args, "speak", None) is not None:
        parts.append(f"speak_{args.speak}")
        if getattr(args, "speak_host", False):
            parts.append("host")
        if getattr(args, "token2wav_host", False):
            parts.append("t2whost")
    if getattr(args, "profile", False):
        parts.append("profile")
    parts.append(f"multi-core_{args.multi_core}")
    return "_".join(parts) + ".md"



def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if os.environ.get("OMNI_WATCHDOG_S"):
        # Debug aid for a stalled board: dump every thread's stack to stderr
        # every N seconds, so a hang shows where the host is blocked.
        import faulthandler
        faulthandler.dump_traceback_later(
            int(os.environ["OMNI_WATCHDOG_S"]), repeat=True, file=sys.stderr)
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.speak_host and args.speak is None:
        parser.error("--speak-host requires --speak")
    if args.token2wav_host and args.speak is None:
        parser.error("--token2wav-host requires --speak")
    if args.speak is not None and _talker_mod is not None and args.speak not in _talker_mod.SPEAKERS:
        parser.error(f"--speak speaker must be one of {_talker_mod.SPEAKERS}")
    if args.no_summary and args.summary:
        parser.error("--summary and --no-summary are mutually exclusive")
    if args.frames < 1:
        parser.error("--frames must be positive")
    if args.frames > 1 and not args.image:
        parser.error("--frames requires --image")
    if args.run_from_bin and args.layout != "flat":
        print("[run_from_bin] only applies to --layout flat; the windowed layout "
              "always builds its programs")
        args.run_from_bin = False
    with _exclusive_run_lock():
        _main_locked(parser, args)


def _main_locked(parser: argparse.ArgumentParser, args) -> None:
    """Run artifact preparation and FPGA execution under the global lock."""
    args.image = _resolve_sample(args.image, DEFAULT_IMAGE, "--image")
    args.audio = _resolve_sample(args.audio, DEFAULT_AUDIO, "--audio")
    engine_kwargs = resolve_engine_config(parser, args)

    cfg = apply_vision_resolution(
        Qwen25OmniUnifiedEngine.load_config(script_dir=SCRIPT_DIR),
        args.vision_res,
    )
    # Fetch/convert before constructing a device-owning engine.  The conversion
    # streams Thinker shards and skips Talker/token2wav-only checkpoint shards.
    params_path = _weight_mod.ensure_params_bin(SCRIPT_DIR)
    processor_dir = _weight_mod.ensure_processor_bundle(
        SCRIPT_DIR, verbose=False
    )
    print(f"Thinker params: {params_path}")
    processor, processed, tokens, prompt, rendered = _prepare_processor_inputs(
        args, cfg, processor_dir
    )
    if len(tokens) < 2:
        raise ValueError("chat template produced fewer than two tokens")
    context, seed = tokens[:-1], tokens[-1]
    context_limit = (FPGA_SPEECH_CONTEXT_SIZE
                     if args.speak is not None and not args.speak_host
                     else PREFILL_INPUT_TOKEN_LIMIT)
    flat_plan = None
    if args.layout == "flat":
        # Weights first: the plan sizes them, then the context that fits what
        # is left. It needs no hardware, so it also bounds the prompt check.
        stages = {"lm"}
        if args.image:
            stages.add("vision")
        if args.audio:
            stages.add("audio")
        if args.speak is not None and not args.speak_host:
            stages.add("talker")
            if not args.token2wav_host:
                stages.add("t2w")
        flat_plan = _layout_mod.plan_flat(
            stages, args.vision_res, args.max_context,
            device_bytes=user_dma_core.AVAILABLE_DRAM_SIZE_GB * 2**30)
        for line in flat_plan.lines():
            print(line)
        context_limit = flat_plan.max_context
        # The Talker's and Token2Wav's weights live in params.bin like the rest. They
        # are appended here, before the engine exists, because the programs.bin
        # identity covers the params manifest.
        _ensure_speech_params(SCRIPT_DIR, cfg, args, flat_plan)
    if len(context) > context_limit:
        raise ValueError(
            f"templated prompt needs {len(context)} prefill tokens; limit is "
            f"{context_limit}. Shorten the prompt or media input."
        )
    if args.speak is not None and not args.speak_host and len(tokens) + 2 >= context_limit:
        raise ValueError(
            f"FPGA speech needs prompt + PAD/BOS + at least one codec position "
            f"inside {context_limit} positions; prompt has {len(tokens)} tokens"
        )

    print(f"\n--- Software-resetting {REQUIRED_ENGINES} engines ---")
    fpga_build = reset_selected_engines()
    print(
        f"Software reset + HALT probe passed on engines 0-"
        f"{REQUIRED_ENGINES - 1} (FPGA build 0x{fpga_build:08x})"
    )
    print("\n--- Building U55 engine ---")
    ue = Qwen25OmniUnifiedEngine(
        script_dir=SCRIPT_DIR, fpga_build=fpga_build,
        vision_res=args.vision_res,
        fpga_talker=args.speak is not None and not args.speak_host,
        layout=args.layout,
        flat_plan=flat_plan,
        **engine_kwargs
    )
    ue.speak_as = args.speak          # before lm_tensor_init sizes the buffers
    if args.speak is not None and _talker_mod is None:
        raise SystemExit("--speak needs qwen2.5_omni_7b_talker.py, which failed to import")
    if args.speak is not None and not args.speak_host and _talker_fpga_mod is None:
        raise SystemExit("--speak needs qwen2.5_omni_7b_talker_fpga.py")
    ue.configure_runtime_artifacts(params_path, processor_dir)
    ue.tokenizer = processor.tokenizer
    ue.processor = processor
    ue._prompt_text = prompt
    print(ue.describe_dram_map())

    image_embeddings = None
    audio_embeddings = None
    audio_metadata = None
    unified = ue.layout.resident_programs
    run_phase = "run" if unified else "all"
    if unified:
        signature = _request_signature(ue, args, context, tokens, processed)
        _prepare_unified(ue, args, processed, context, cfg, tokens, signature,
                         from_bin=args.run_from_bin)
        # From here on the host should only launch programs by address. Count any
        # instruction the host still writes, and report it at the end of the run.
        runtime_writes = ue._runtime_instruction_writes = [0]
        original_write = UnifiedEngine.write_captured_instructions_to_dram

        def counted_write(engine, *a, **k):
            runtime_writes[0] += 1
            return original_write(engine, *a, **k)

        UnifiedEngine.write_captured_instructions_to_dram = counted_write
    if args.image:
        image_embeddings = _run_vision(ue, processed, profile=args.profile,
                                       phase=run_phase)
    if args.audio:
        audio_embeddings, audio_metadata = _run_audio(ue, processed, phase=run_phase)

    image_grid = processed.get("image_grid_thw") if args.image else None
    audio_lengths = (
        audio_metadata["output_lengths"] if audio_metadata is not None else None
    )
    positions, rope_delta = build_multimodal_positions(
        tokens,
        image_grid_thw=image_grid,
        audio_token_lengths=audio_lengths,
        image_token_id=int(cfg["tokens"]["image_token_id"]),
        audio_token_id=int(cfg["tokens"]["audio_token_id"]),
        video_token_id=int(cfg["tokens"]["video_token_id"]),
        spatial_merge_size=int(cfg["vision"]["spatial_merge_size"]),
        position_ids_per_second=int(cfg["vision"]["tokens_per_second"]),
    )
    ue._rope_offset = int(rope_delta)
    print(
        f"\n[Mode] {_result_mode(args)}: {len(context)} prefill tokens, "
        f"mRoPE delta {rope_delta}, prompt {prompt!r}"
    )

    def _write_summary(profiles=None) -> None:
        """Render the run summary; a reporting failure never fails the run."""
        if args.no_summary:
            return
        out = args.summary or os.path.join(SCRIPT_DIR, run_summary_filename(args))
        try:
            ue.write_run_summary(out, args, profiles=profiles or None)
            print(f"\nWrote run summary: {out}")
        except Exception as exc:  # noqa: BLE001 - reporting is best effort
            print(f"[warn] failed to write run summary: {exc}")

    print("\n--- Thinker LM stage ---")
    started = time.perf_counter()
    if unified:
        ue.enter_stage_tensors("lm", ue.lm_tensor_init)
    else:
        ue.lm_weight_init()
        ue.lm_tensor_init()
        ue.compile_prefill(len(context), profile=args.profile)
        # Decoder setup installs IF4 projection shards in private windows; the
        # BF16 embedding table stays on the host and only selected rows are DMA'd.
        ue.compile_decoder(profile=args.profile)
        ue.check_master_isa()
        # These bodies are address-coupled: decoder starts immediately after this
        # exact prefill image. Publish both in one programs.bin generation.
        ue.store_program_stages("prefill", "decode")
        for line in ue.isa_usage_lines():
            print(line)
    ue.run_prefill(
        context,
        image_embeddings=image_embeddings,
        audio_embeddings=audio_embeddings,
        positions=positions[: len(context)],
        profile=args.profile,
    )
    ue.activate_decode_shared_weights()

    if args.profile:
        # The checkpointed decoder is the one published in programs.bin, so it
        # is also the one that must run: run_decode_step_profiled refuses any
        # other image. Only the 1st-token step is profiled -- it is taken
        # right after prefill, at the prompt's own context.
        prof_program = ue._decoder_program
        prof_checkpoints = list(ue._decoder_checkpoints)
        prof_workers = list(getattr(ue, "_decoder_workers", []))
        print(f"\n--- Profiled decode: 1st token (ctx {ue.seq_len}) ---")
        first_results, _next_token, aligned_first = ue.run_decode_step_profiled(
            seed, prof_program, prof_checkpoints, workers=prof_workers
        )
        ctx_first = ue.seq_len
        # A 4th element, model_phases, is the stage's TRUE (unpadded) work
        # broken out by the same checkpoint names the profile results use --
        # computed here, where the right context/dims are in scope, rather
        # than guessed from the title string inside _profile_tables.
        profiles = []
        if args.image:
            dims = ue._vision_dims()
            profiles.append((
                "Vision encoder",
                f"{dims['VS']} patches -> {dims['NUM_MERGED_TOKENS']} soft tokens. "
                "patch_embed is a separate FPGA program before the encoder checkpoints.",
                getattr(ue, "_vis_profile", None),
                ue._model_flops_vision_by_phase(dims),
            ))
        profiles += [
            ("Prefill", f"{len(context)} tokens.",
             getattr(ue, "_prefill_profile", None),
             ue._model_flops_prefill_by_phase()),
            ("Decode - 1st token",
             f"Context {ctx_first} tokens (aligned {aligned_first}).",
             first_results,
             _model_flops.decode_step_flops_by_phase(ue._cfg, ctx_first)),
        ]
        for title, note, results, _model_phases in profiles:
            if results:
                ue.print_profile_table(title, results, note=note)
        print(f"\nThinker profile done in {time.perf_counter() - started:.2f}s wall")
        _write_summary(profiles)
        return

    print("\n--- Decode run ---")
    speech = None
    if args.speak is None:
        _, decoded_text = ue.run_decoder(seed, max_new_tokens=args.max_new_tokens)
        thinker_finished = time.perf_counter()
    elif not args.speak_host:
        ue._speech_steps = []
        _, decoded_text = ue.run_decoder(seed, max_new_tokens=args.max_new_tokens)
        thinker_finished = time.perf_counter()
        speech = _synthesize_speech_fpga(ue, args, cfg, tokens)
        ue._speech_result = speech
    else:
        ue._speech_steps = []
        speech_session = _prepare_streamed_speech(ue, args, cfg, tokens)
        stream, hs, input_ids, prefill_rows, prefill_embeds, lookup, _, _ = speech_session
        decoder_result: dict[str, Any] = {}

        def _decode_worker():
            decoder_result["started"] = time.perf_counter()
            try:
                decoder_result["output"] = ue.run_decoder(
                    seed, max_new_tokens=args.max_new_tokens)
            except BaseException as exc:
                decoder_result["error"] = exc
            finally:
                decoder_result["finished"] = time.perf_counter()
                ue._speech_step_callback = None
                if "error" in decoder_result:
                    stream.abort(decoder_result["error"])
                else:
                    stream.close()

        decoder_thread = threading.Thread(
            target=_decode_worker, name="omni-fpga-decode", daemon=True)
        decoder_thread.start()
        print("  [Speak] FPGA decode and host Talker running in parallel",
              flush=True)
        try:
            wav = hs.speak_stream(
                input_ids=input_ids,
                prefill_hidden_rows=prefill_rows,
                prefill_embeds=prefill_embeds,
                reply_stream=stream,
                embed_lookup=lookup,
            )
        except Exception as exc:
            decoder_thread.join(timeout=60)
            if decoder_thread.is_alive():
                raise TimeoutError("FPGA decode did not finish after Talker failure") from exc
            if "error" in decoder_result:
                raise decoder_result["error"] from exc
            print(f"[Speak] streaming path failed ({exc}); retrying the "
                  "completed Thinker response sequentially", flush=True)
            wav = None
        decoder_thread.join(timeout=60)
        if decoder_thread.is_alive():
            raise TimeoutError("FPGA decoder did not finish within 60 seconds")
        if "error" in decoder_result:
            raise decoder_result["error"]
        _, decoded_text = decoder_result["output"]
        thinker_finished = decoder_result["finished"]
        if wav is None:
            speech = _synthesize_speech(ue, args, cfg, tokens)
        else:
            speech = _finish_streamed_speech(
                args, speech_session, wav,
                decoder_result["started"], decoder_result["finished"])
        ue._speech_result = speech
    lm_wall = thinker_finished - started
    print(f"\nThinker stage done in {lm_wall:.2f}s wall")
    if os.environ.get("OMNI_VERIFY_IMAGES"):
        bad = ue.verify_resident_images()
        print(f"  [verify] resident images after the run: "
              f"{'ALL MATCH' if not bad else 'MISMATCH ' + '; '.join(bad)}", flush=True)
    for line in getattr(ue.layout, "peak_lines", lambda: [])():
        print(line, flush=True)
    if unified:
        print(f"  [run] instruction writes by the host after the prepare phase: "
              f"{ue._runtime_instruction_writes[0]}", flush=True)

    visible_generated = int(getattr(ue, "_decode_n", 0))
    generated = len(getattr(ue, "_decode_step_us", ()))
    decode_wall = float(getattr(ue, "_decode_wall_s", 0.0))
    prefill_wall = float(getattr(ue, "_prefill_wall_s", 0.0))
    prefill_us = float(getattr(ue, "_latency_prefill_us", 0.0))
    decode_tok_s = generated / decode_wall if decode_wall > 0 else None
    expected_program_stages = {"prefill", "decode"}
    if args.image:
        expected_program_stages.add("vision")
    if args.audio:
        expected_program_stages.add("audio")
    if ue._executed_program_stages != expected_program_stages:
        raise RuntimeError(
            "not every model stage completed from programs.bin: expected "
            f"{sorted(expected_program_stages)}, executed "
            f"{sorted(ue._executed_program_stages)}"
        )
    program_manifest, program_payload = ue._program_bundle.load()
    # Talker and Token2Wav programs are written to DRAM as they compile and sit in
    # the same generation as raw stages; the Thinker stages must all have run.
    if program_manifest["section_count"] != REQUIRED_ENGINES * (
        len(expected_program_stages) + len(ue._raw_stages)
    ):
        raise RuntimeError(
            "combined programs.bin does not contain one master + seven worker "
            "sections for every executed stage"
        )
    try:
        _summary_rows = ue.stage_metrics(args)
    except Exception:  # noqa: BLE001 - reporting must not fail a good run
        _summary_rows = []
    reported_prefill_tokens = len(context)
    result = {
        "model": "qwen2.5_omni_7b",
        "mode": _result_mode(args),
        "params_source": "params.bin",
        "program_source": "programs.bin",
        "program_stages": sorted(ue._executed_program_stages),
        "programs_bin": str(ue._program_bundle.bin_path),
        "programs_size_bytes": int(program_manifest["programs_size"]),
        "program_payload_bytes": len(program_payload),
        "program_section_count": int(program_manifest["section_count"]),
        "decoded_text": decoded_text,
        # Canonical model_auto_test fields.
        "prefill_tokens": reported_prefill_tokens,
        "decoded_tokens": generated,
        "prefill_speed_tok_s": (
            reported_prefill_tokens / prefill_wall if prefill_wall > 0 else None
        ),
        "decode_speed_tok_s": decode_tok_s,
        "prefill_size_kb": len(ue._prefill_program[1]) / 1024,
        "decoder_size_kb": len(ue._decoder_program[1]) / 1024,
        # Model-specific aliases retained for direct consumers.
        "tokens_generated": visible_generated,
        "decode_steps": generated,
        "stop_token_id": (
            ue._fpga_decode_token_ids[-1]
            if getattr(ue, "_fpga_decode_token_ids", None)
            and ue._fpga_decode_token_ids[-1] in ue._decode_stop_token_ids()
            else None
        ),
        "decode_tok_s": decode_tok_s,
        "prefill_hw_tok_s": (
            reported_prefill_tokens / (prefill_us * 1e-6) if prefill_us > 0 else None
        ),
        "first_token_tok_s": (
            1e6 / ue._decode_step_us[0]
            if getattr(ue, "_decode_step_us", None)
            else None
        ),
        "model_gflops_effective": {
            row["stage"]: round(row["model_gflops"], 3)
            for row in _summary_rows
            if row.get("model_gflops")
        },
        "model_gflop_work": {
            row["stage"]: round(row["model_flops"] / 1e9, 3)
            for row in _summary_rows
            if row.get("model_flops")
        },
        "prefill_gflops": getattr(ue, "_prefill_gflops", None),
        "decode_gflops": getattr(ue, "_decode_gflops", None),
        "vision_gflops": getattr(ue, "_vis_gflops", None),
        "audio_gflops": getattr(ue, "_audio_gflops", None),
        "prompt_tokens": reported_prefill_tokens + 1,
        "rope_delta": rope_delta,
    }
    print("TEST_RESULT: " + json.dumps(result, ensure_ascii=False))
    _write_summary()


if __name__ == "__main__":
    main()
