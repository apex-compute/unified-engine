#!/usr/bin/env python3
"""Qwen2.5-Omni-7B Thinker on the eight-engine, 8-GiB Alveo map.

This entry point implements text, image, and audio understanding with text
generation.  The speech Talker/token2wav path is deliberately not part of this
binary. Vision, audio, and LM weights time-share one params window; encoder
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
import subprocess
import sys
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
            f"Qwen2.5-Omni is FPGA-only, but PyTorch reports {details}. "
            "Launch this entry point in a fresh process; GPU "
            "visibility is disabled before torch import."
        )


_reject_visible_torch_accelerators()

import user_dma_core
from multi_engine_shard import (MultiEngineScheduler, PrivateArena,
                                require_multicore_dram, tiled_window_bases)
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

Qwen25OmniLMMixin = _lm_mod.Qwen25OmniLMMixin
Qwen25OmniVisionMixin = _vision_mod.Qwen25OmniVisionMixin
Qwen25OmniAudioMixin = _audio_mod.Qwen25OmniAudioMixin
build_multimodal_positions = _position_mod.build_multimodal_positions
ProgramBundle = _program_mod.ProgramBundle


REQUIRED_ENGINES = 8
REQUIRED_DRAM_GIB = 8
# CONTEXT AND PREFILL ARE ONE BUDGET. Prefill and decode read the same KV
# cache, so MAX_CONTEXT_SIZE bounds both: a prompt (vision soft tokens + text)
# may fill it, and generation continues inside it.
MAX_CONTEXT_SIZE = 2500
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
# FIXED-SHAPE RUN PRESETS
# ==========================================================================
# --low, --medium and --high pin a whole run shape rather than a set of files, so the
# two are reproducible prefill lengths instead of "whatever this wav happened
# to be". Each preset fixes the media and then GROWS THE TEXT PROMPT until the
# assembled prefill reaches its target, because the media contribute a fixed
# token count and only the text is free:
#
#   low     audio at half length (~64 soft tokens) + text  -> ~850 prefill
#   medium  896x896 image (1024 soft tokens) + full audio
#           (128 soft tokens) + text                       -> ~2048 prefill
#   high    4 x 896x896 camera frames + 6 s audio + text   -> 6144 aggregate
#           performance-only input, executed as 3 x 2048 resident chunks
#
# Both fit the 2560-row prefill allocation. Medium targets a power-of-two
# 2048-row run and leaves about 452 tokens of total-context decode headroom.
RUN_PRESETS = {
    "low": {
        "vision_res": None,
        "image": False,
        "audio_fraction": 0.5,
        "prefill_tokens": 850,
    },
    "medium": {
        "vision_res": "medium",
        "image": True,
        "audio_fraction": 1.0,
        "prefill_tokens": 2048,
    },
    "high": {
        "vision_res": "medium",
        "image": True,
        "audio_fraction": 1.0,
        # 600 valid mel frames is the configured audio-encoder maximum.
        "audio_seconds": 6.0,
        "vision_runs": 4,
        "prefill_tokens": 6144,
        # Processor length includes the final seed token; 2049 renders 2048
        # rows into the prefill program.
        "prefill_chunk_tokens": 2049,
        "prefill_chunks": 3,
    },
}

# Filler for the fitted prompts. It has to be REAL INSTRUCTION TEXT, not
# padding: the point of a long prefill is to measure the shape the model
# actually runs, and a prompt of repeated nonsense changes what attention does
# with it. These sentences are cycled and then trimmed to the exact token count
# the target needs.
# THE INSTRUCTION IS PER PRESET; THE FILLER IS NOT. A preset's prompt is mostly
# padding by token count, so whatever the padding says gets repeated two or
# three times and drowns out the instruction. The first --medium prompt cycled
# sentences that kept re-asserting "transcribe the audio", and the model did
# exactly that and stopped -- a correct 1899-token prefill, but only half the
# answer. The instruction now states the parts once, up front, and the filler
# is neutral guidance that pushes toward COVERING EVERYTHING rather than toward
# either modality.
_PRESET_PROMPT_BASE = {
    "low": (
        "Transcribe the speech exactly as it is spoken, preserving the wording "
        "and the order of the terms."
    ),
    "medium": (
        "Do two things, in this order. First, transcribe the speech exactly as "
        "it is spoken, preserving the wording and the order of the terms. "
        "Second, describe the image in detail, covering the lighting, the "
        "terrain, the vegetation and the sky. Both parts are required."
    ),
    "high": (
        "Benchmark a multi-camera request with a long spoken query and text "
        "history. Numeric correctness and response coherence are not part of "
        "this performance-only run."
    ),
}

_PRESET_PROMPT_FILLER = (
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


def _preset_filler_words(count: int) -> str:
    """``count`` words, cycling the filler sentences in order."""
    words: list[str] = []
    i = 0
    while len(words) < count:
        words.extend(_PRESET_PROMPT_FILLER[i % len(_PRESET_PROMPT_FILLER)].split())
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
        return f"{base} {_preset_filler_words(words)}"

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

# ==========================================================================
# THE WINDOW IS ONE HBM SWITCH REGION (Alveo U50 and U55C)
# ==========================================================================
# A window is 1 GiB because that is the granularity the HBM fabric arbitrates
# on, and eight of them tile the 8 GiB device so every core reads its own.
# Which board is underneath decides only WHY 1 GiB is the right number:
#
#   U55C (HW_INFO cores == 12)  one memory controller owns one contiguous
#       1 GiB, and the reordered SAXI wiring puts core i on controller i for
#       i < 8. The window IS the controller. Measured on the PRE-reorder
#       bitstream, where all twelve engines crowded onto MC0-MC3, decode ran at
#       77.3 GFLOPS against 140.1 for the old 512 MiB map; with the ports
#       reordered the same map gives 190.2.
#   U50 (HW_INFO cores == 8)    the contended unit is the 1 GiB four-pseudo-
#       channel switch region, not the 512 MiB controller: two engines in one
#       region still read at the full per-engine rate, three halve it. A 1 GiB
#       window IS one switch region, so this map puts ONE engine in each.
#
# The U50 is where the map looks portable but is not obviously so, because an
# engine owns two 512 MiB segments outright -- one per 4 GiB stack, on its own
# SAXI port -- and its 1 GiB window here is neither of them. That is deliberate.
# Port ownership decides whether a read takes a lateral hop, and at these
# transfer sizes the hop is free: the port-ordered bases and a flat 512 MiB
# stride both measure 85.2 GB/s, the per-engine AXI ceiling. Crowding is what
# costs bandwidth, and a contiguous 1 GiB window per core cannot crowd. Keeping
# the window contiguous is also what keeps the map feasible at all -- the untied
# head needs 276 MiB in one piece (OMNI_LM_HEAD_BYTES), which no 512 MiB segment
# could still offer beside ~708 MiB of private shards.
#
# multi_engine_shard.tiled_window_bases() owns both boards' answers and
# validates the result (in range, disjoint, no crowded region); a board it has
# not characterised is refused rather than guessed at.

# The private window geometry this map is built for. It is deliberately NOT
# multi_engine_shard.MULTICORE_WINDOW_BYTES: that constant is 512 MiB and three
# other multi-core models are validated against it. Nor does it track either
# board's constant: it is THIS MAP's geometry, and every number below --
# OMNI_PRIVATE_RESERVE_BYTES, OMNI_LM_HEAD_BYTES, the ISA and tensor slices --
# is tuned against it. tiled_window_bases() refuses a board whose own window
# differs (the U55C's becomes 2 GiB when its HBM is upgraded), so the map gets
# re-tuned deliberately instead of silently running at the wrong size.
OMNI_WINDOW_BYTES = 0x4000_0000            # 1 GiB per core, 8 GiB total
# Measured worst case is core 0: 4.28 MiB of tensor-parallel prefill + 1.02 MiB
# of decoder, plus the vision/audio encoder programs. 16 MiB is ~3x that, and
# every MiB here is a MiB the window whose gap hosts the shared tensor extent
# does not have -- that core carries the extent AND a full private shard set.
OMNI_ISA_BYTES = 16 * 2**20                # per-core ISA slice, inside the window
# Per-core scratch, inside the window. The head-sharded prefill attention keeps
# ONE private scratch per engine, (AHD + aligned_P) * aligned_P + aligned_P *
# AHD elements -- 13.75 MiB at the 2560-row prefill allocation. Vision's much
# larger 4096-patch worker scratch is pinned in the shared pool instead.
OMNI_PRIVATE_TENSOR_BYTES = 16 * 2**20

# What the private shards need per core, declared BEFORE any shared byte is
# lent. Measured on the 8-engine map:
#   gate/up N-shards (both phases)  240.8   down N-shard (decode)  120.4
#   down K-shard (prefill TP)       120.4   attn decode shard       38.3
#   lm_head shard                    34.5   embedding shard         67.3
#   decode BF16 O shard              85.8                        = 707.5 MiB
# Asserted against actual usage after loading, so drift fails loudly instead of
# silently eating the pool.
OMNI_PRIVATE_RESERVE_BYTES = 712 * 2**20

# THE TWO OBJECTS THAT CANNOT BE SCATTERED. Weight sections are placed one at a
# time into whichever window has room, which works for the 196 attention
# sections (max 6.5 MiB) and the encoders (max 13.3 MiB). It does not work for
# the untied head: one 276 MiB IF4 blob that must be contiguous, and that no
# window can still host once attention has been spread over them. It gets a
# dedicated extent carved at init, next to the tensor extent, before anything
# competes for the space.
OMNI_LM_HEAD_BYTES = 280 * 2**20

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
    os.path.join(SCRIPT_DIR, "qwen2.5_omni_7b_weights.py"),
    os.path.join(SCRIPT_DIR, "qwen2.5_omni_7b_config.json"),
    os.path.join(PROJECT_ROOT, "models", "qwen2.5_vl_3b", "qwen2.5_vl_3b_lm.py"),
    os.path.join(PROJECT_ROOT, "models", "qwen2.5_vl_3b", "qwen2.5_vl_3b_vision.py"),
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
                 vision_res: str = DEFAULT_VISION_RES):
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
        # At LEAST 8 GiB, not exactly: the map needs 8 GiB and a larger device
        # simply leaves the top unused. The check lives in the library so every
        # multi-core model states the same requirement the same way.
        require_multicore_dram(multi_core, "Qwen2.5-Omni-7B")
        # THE BOARD GATE, taken before a single byte is laid out: the library
        # answers with this board's window bases or refuses the board outright.
        expected_bases = tiled_window_bases(
            REQUIRED_ENGINES, OMNI_WINDOW_BYTES, "Qwen2.5-Omni-7B")

        self.multi_core = multi_core
        self.fpga_build = None if fpga_build is None else int(fpga_build)
        self._multi_core_schedulers: dict[str, MultiEngineScheduler] = {}
        self._worker_isa_used: dict[int, dict[str, int]] = {}
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

        # U55 8-GiB DRAM map -- EIGHT 1-GiB PRIVATE WINDOWS THAT TILE THE DEVICE
        #
        #   core i -> [i GiB, (i+1) GiB), and inside each window, low to high:
        #     weights  984 MiB   private shards, bump-allocated UP from the base
        #     (gap)               the shared pool, bump-allocated DOWN from 984
        #     ISA       32 MiB
        #     tensor     8 MiB   per-engine scratch
        #
        # WHY THE WHOLE DEVICE IS PRIVATE WINDOWS. At 1 GiB per core the windows
        # span all 8 GiB, so there is no region left ABOVE the arena to hold the
        # weights and tensors every core reads. The empty tail of each window is
        # the only space there is, so shared data is carved from there and the
        # arena arbitrates between the two cursors (PrivateArena.alloc_shared).
        #
        # WHY THE PREFILL MLP IS TENSOR-PARALLEL. A shared copy of the decoder
        # is 3655 MiB and does not fit beside the private shards: it packs into
        # the gaps with only 8 x ~42 MiB left, and the 112-MiB KV cache then has
        # nowhere contiguous to go. Sharding the MLP -- 2889 MiB, 79% of the
        # decoder -- over the engines is what makes this map fit, and it is also
        # FASTER at the real prefill tile: +14.2% measured at M=64, the size a
        # <=64-token prompt runs. Only attention, the untied head and the norms
        # stay shared, 765 MiB placed section by section across the gaps.
        #
        # Vision, audio and the shared LM weights still TIME-SHARE the pool, now
        # via shared_mark()/shared_release() instead of one contiguous window.
        # The top of the highest window. On the 8 GiB image that is the whole
        # device; on the 16 GiB one the windows are eight (stack, MC) regions
        # spread over it, so this is a bound rather than "the device".
        self.DRAM_END = max(expected_bases) + OMNI_WINDOW_BYTES
        self.WINDOW_BYTES = OMNI_WINDOW_BYTES
        # BUILT FROM THE BOARD'S BASES, not from base-plus-stride: on the 16 GiB
        # image the assignment is a permutation (core 1 at 9 GiB, core 9 at
        # 1 GiB) that no stride expresses.
        self.mc_arena = PrivateArena(
            REQUIRED_ENGINES,
            windows=[(base, OMNI_WINDOW_BYTES) for base in expected_bases],
            isa_bytes=OMNI_ISA_BYTES,
            tensor_bytes=OMNI_PRIVATE_TENSOR_BYTES,
            verbose=True,
        )
        if self.mc_arena.stride != OMNI_WINDOW_BYTES:
            raise AssertionError(
                f"private windows are 0x{self.mc_arena.stride:X}, not the "
                f"0x{OMNI_WINDOW_BYTES:X} this map is built for")
        # The arena now takes the library's bases verbatim, so this asserts the
        # construction rather than a coincidence -- it still fails HERE, at
        # init, if the two ever drift apart.
        actual_bases = [self.mc_arena.window_base(i) for i in range(REQUIRED_ENGINES)]
        if actual_bases != expected_bases:
            raise AssertionError(
                "private windows do not match this board's map: "
                f"{[hex(a) for a in actual_bases]} against "
                f"{[hex(b) for b in expected_bases]}")
        _device_bytes = user_dma_core.AVAILABLE_DRAM_SIZE_GB * 2**30
        if self.DRAM_END > _device_bytes:
            raise AssertionError(
                f"{REQUIRED_ENGINES} x {OMNI_WINDOW_BYTES // 2**20} MiB windows "
                f"reach 0x{self.DRAM_END:X}, past the "
                f"{user_dma_core.AVAILABLE_DRAM_SIZE_GB} GiB HW_INFO reports")

        # ISA lives INSIDE the windows now. Engine 0's slice is the master area;
        # every worker's is arena.isa_base(i), one window stride apart. The old
        # WORKER_ISA_BASE was only ever used as the master's upper bound, so that
        # bound is now named for what it is.
        self.ISA_BASE = self.mc_arena.isa_base(0)
        self.MASTER_ISA_RESERVE = OMNI_ISA_BYTES
        self.MASTER_ISA_LIMIT = self.mc_arena.isa_limit(0)
        self.WORKER_ISA_STRIDE = OMNI_WINDOW_BYTES

        # PRIVATE SPACE IS CLAIMED BEFORE ANY SHARED BYTE IS LENT. The shard
        # sizes are known from the manifest; the pool is whatever is left.
        self.mc_arena.reserve_private(OMNI_PRIVATE_RESERVE_BYTES)

        # TENSORS ARE CARVED PER BUFFER, NOT FROM ONE EXTENT. Only an individual
        # buffer has to be contiguous -- the KV cache, an [M, N] activation
        # plane. The largest at the current allocation is the 140 MiB TP
        # down-output scratch, which is also overlaid by shorter-lived Q/K/V,
        # attention-result and norm tensors. Carving physical buffers separately
        # spreads them over the shared pool instead of requiring one impossible
        # contiguous tensor extent.
        self.TENSOR_BASE = 0
        self.TENSOR_LIMIT = sum(self.mc_arena.shared_free())
        self._tensor_staged = 0
        self._tensor_phase_mark = self.mc_arena.shared_up_mark()
        head_base = self.mc_arena.alloc_shared(OMNI_LM_HEAD_BYTES, "LM_HEAD.window")
        self._reserved_extents = {
            "lm_head": [head_base, head_base + OMNI_LM_HEAD_BYTES, head_base],
        }

        # PARAMS IS NO LONGER A WINDOW, IT IS AN ACCOUNTING ORIGIN. Weight
        # sections are placed individually by alloc_shared, so there is no
        # params cursor to walk; PARAMS_BASE/_LIMIT keep the bookkeeping that
        # every caller already does ("bytes staged so far", "capacity left")
        # working against the pool instead of against a contiguous range.
        self.PARAMS_BASE = 0
        self.PARAMS_LIMIT = sum(self.mc_arena.shared_free())
        self.VISION_WEIGHT_BASE = self.PARAMS_BASE
        self._params_staged = 0
        self._params_phase_mark = self.mc_arena.shared_mark()

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
        self.MAX_CONTEXT_SIZE = MAX_CONTEXT_SIZE
        self.PREFILL_MAX_SEQ_LEN = PREFILL_MAX_SEQ_LEN
        model["max_context_size"] = MAX_CONTEXT_SIZE
        model["prefill_max_seq_len"] = PREFILL_MAX_SEQ_LEN

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
        return self.mc_arena.alloc_shared_up(
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
        addr = self.mc_arena.alloc_shared(
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
        reclaimed = self.mc_arena.shared_release(self._params_phase_mark)
        for extent in self._reserved_extents.values():
            extent[2] = extent[0]
        self._params_staged = 0
        if reclaimed:
            self._loud(f"  [map] reclaimed {reclaimed / 2**20:.1f} MiB of shared "
                       f"pool from the previous phase")

    # -- tensor allocation against the shared pool ---------------------------

    def allocate_tensor_dram(self, size_bytes: int, label: str | None = None,
                             align_bytes: int = 64) -> int:
        addr = self.mc_arena.alloc_shared_up(
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
        self.mc_arena.shared_up_release(self._tensor_phase_mark)
        self._tensor_staged = 0

    def tensor_phase_mark(self):
        """A rollback point for a partially-built tensor set."""
        return (self.mc_arena.shared_up_mark(), self._tensor_staged)

    def tensor_phase_restore(self, mark) -> None:
        arena_mark, staged = mark
        self.mc_arena.shared_up_release(arena_mark)
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
        hw = self._cfg["hardware"]
        expected = {
            "required_engines": REQUIRED_ENGINES,
            "required_dram_gib": REQUIRED_DRAM_GIB,
            "private_arena_base": 0,
            "private_arena_bytes": REQUIRED_ENGINES * OMNI_WINDOW_BYTES,
            "private_window_bytes": OMNI_WINDOW_BYTES,
            "private_isa_bytes": OMNI_ISA_BYTES,
            "private_tensor_bytes": OMNI_PRIVATE_TENSOR_BYTES,
            # The DRAM this map CLAIMS, which is eight 1 GiB windows wherever
            # the board puts them -- not self.DRAM_END. On the 8 GiB image the
            # windows tile the device and the two are the same number; on the
            # 16 GiB image they are eight of sixteen (stack, MC) regions and
            # DRAM_END is the top of the highest, 16 GiB. Checking DRAM_END
            # here would demand a config edit per board for a contract that
            # does not change: the map needs 8 GiB of private windows.
            "dram_limit": REQUIRED_ENGINES * OMNI_WINDOW_BYTES,
        }
        for name, wanted in expected.items():
            raw = hw.get(name)
            actual = int(raw, 0) if isinstance(raw, str) else int(raw)
            if actual != wanted:
                raise ValueError(
                    f"config hardware.{name}={raw!r}, expected 0x{wanted:X}"
                )
        if self.mc_arena.stride != int(hw["private_window_bytes"], 0):
            raise AssertionError("PrivateArena did not produce 1-GiB windows")
        expected_weight_bytes = (OMNI_WINDOW_BYTES - OMNI_ISA_BYTES
                                 - OMNI_PRIVATE_TENSOR_BYTES)
        if self.mc_arena.weight_bytes() != expected_weight_bytes:
            raise AssertionError(
                f"each engine must have {expected_weight_bytes // 2**20} MiB for its "
                f"private shards and the shared pool, got "
                f"{self.mc_arena.weight_bytes() // 2**20} MiB")
        if self._cfg["file_info"]["hidden_size"] != 3584:
            raise ValueError("this runtime is compiled only for the 3584-wide 7B Thinker")
        if set(self._cfg["precision"]["lm_quantized_projections"]) != {
            "q", "k", "o", "gate", "up", "down"
        }:
            raise ValueError(
                "prefill requires BF16 V and IF4 Q/K/O/GATE/UP/DOWN"
            )
        if set(self._cfg["precision"].get("decode_bf16_projections", ())) != {"o"}:
            raise ValueError(
                "decode requires a time-shared BF16 O projection region"
            )
        if self._cfg["precision"].get("embedding") != "if8":
            raise ValueError(
                "the FPGA-only runtime requires precision.embedding='if8'"
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
                "dram_map": {
                    "private_base": 0,
                    "params_base": self.PARAMS_BASE,
                    "params_limit": self.PARAMS_LIMIT,
                    "tensor_base": self.TENSOR_BASE,
                    "tensor_limit": self.TENSOR_LIMIT,
                    "master_isa_base": self.ISA_BASE,
                    "worker_isa_base": self.MASTER_ISA_LIMIT,
                    # Per engine, not base + i * stride: on a board map the
                    # windows are a permutation and no stride reaches them.
                    "worker_isa_bases": [self.mc_arena.isa_base(i)
                                         for i in range(REQUIRED_ENGINES)],
                    "dram_end": self.DRAM_END,
                },
            },
        }
        self._program_bundle = ProgramBundle(
            os.path.dirname(params_path), identity, stem="programs"
        )
        self._packaged_program_stages.clear()
        self._loaded_program_stages.clear()
        self._executed_program_stages.clear()
        self._program_stage_profiles.clear()
        self._runtime_processor_dir = processor_dir
        self._loud(
            f"Program artifact: {self._program_bundle.bin_path} "
            "(fresh compile -> atomic store -> validated reload)"
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

    def _program_stage_state(self, stage: str):
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
            if (
                len(stripe_bytes) != 1
                or len(stripe_cols) != 1
                or len(layer_bytes) != 1
            ):
                raise RuntimeError("BF16 O stripe geometry is not uniform")
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
                "fpga_global_argmax": bool(self._fpga_global_argmax_emitted),
                "embedding_precision": self._cfg["precision"]["embedding"],
                "decode_bf16_projections": list(
                    self._cfg["precision"]["decode_bf16_projections"]
                ),
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
                "decode_o_layout": {
                    "mode": "upper_params_column_stripes",
                    "precision": "bf16",
                    "engines": len(stripes),
                    "columns_per_engine": next(iter(stripe_cols)),
                    "layer_bytes_per_engine": next(iter(layer_bytes)),
                    "stripe_bytes_per_engine": next(iter(stripe_bytes)),
                    "slot_stride_bytes": int(self.mc_arena.stride),
                    "extent_end": f"0x{max(int(s['end']) for s in stripes):X}",
                },
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
            vision_runs = int(getattr(self, "_high_vision_runs", 1))
            rows.append({
                "stage": "Vision encoder",
                "detail": (
                    f"{vision_runs} frames x {dims['VS']} patches -> "
                    f"{vision_runs * dims['NUM_MERGED_TOKENS']} soft tokens"
                    if vision_runs > 1 else
                    f"{dims['VS']} patches -> {dims['NUM_MERGED_TOKENS']} soft tokens"
                ),
                "flops": float(self._vis_total_flops),
                "model_flops": (
                    float(self._model_flops_vision(dims) or 0.0) * vision_runs
                ),
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
                "model_flops": getattr(
                    self, "_high_segmented_prefill_model_flops",
                    self._model_flops_prefill(),
                ),
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

    def _stage_table(self, rows: list[dict]) -> list[str]:
        """Headline table: work, time, throughput, % of peak, core scaling."""
        cores = getattr(self, "multi_core", 1) or 1
        out = [
            "| Stage | Shape | Work (GFLOP) | FPGA time (ms) | Throughput "
            "(GFLOPS) | % of peak | x 1-engine peak | CPU wall (s) |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            out.append(
                f"| {row['stage']} | {row['detail']} | {row['flops'] / 1e9:.2f} | "
                f"{row['us'] / 1e3:.1f} | {row['gflops']:.1f} | "
                f"{row['util_pct']:.1f}% | {row['speedup']:.2f}x | "
                f"{row['wall']:.2f} |"
            )
        total_flops = sum(row["flops"] for row in rows)
        total_us = sum(row["us"] for row in rows)
        total_wall = sum(row["wall"] for row in rows)
        peak = self.vis_peak_gflops()
        core_peak = self.per_core_peak_gflops()
        total_gflops = total_flops / (total_us * 1e3) if total_us else 0.0
        out.append(
            f"| **TOTAL** | {cores} engines | **{total_flops / 1e9:.2f}** | "
            f"**{total_us / 1e3:.1f}** | **{total_gflops:.1f}** | "
            f"**{(100.0 * total_gflops / peak if peak else 0.0):.1f}%** | "
            f"**{(total_gflops / core_peak if core_peak else 0.0):.2f}x** | "
            f"**{total_wall:.2f}** |"
        )
        return out

    def _effective_table(self, rows: list[dict]) -> list[str]:
        """Throughput measured against the MODEL's work, not the engine's.

        Every stage bills the FLOPs it issued, at the padded and tile-aligned
        shapes the hardware ran: a 64-row execution multiple for a 31-token
        prompt, windowed attention widened to a full mask, an aligned head.
        Dividing the architecture's own FLOP count by the same measured time
        gives the effective rate -- useful work per second -- which is what
        compares across implementations and accelerators.  ``Useful`` is the
        ratio: how much of what the engine issued the model actually needed.
        """
        priced = [row for row in rows if row.get("model_flops")]
        if not priced:
            return []
        peak = self.vis_peak_gflops()
        out = [
            "## Effective throughput (model FLOPs)",
            "",
            "`Model GFLOP` is what the architecture owes at its own "
            "dimensions -- true prompt length, true attention windows, matrix "
            "products only. `Issued GFLOP` is what this engine billed at the "
            "shapes it actually ran. `Effective GFLOPS` divides the first by "
            "the measured FPGA time, so it is comparable to any other "
            "implementation of this model on any hardware; `Useful` is how "
            "much of the issued work the model needed.",
            "",
            "| Stage | Model GFLOP | Issued GFLOP | Useful | FPGA time (ms) | "
            "Effective GFLOPS | % of peak |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        suspect = False
        for row in priced:
            # Above 100% the engine billed FEWER FLOPs than the architecture
            # requires, which cannot happen: padding and alignment only ever
            # ADD work.  It means that stage's own accounting is undercounting
            # what it issued, so flag it rather than printing it deadpan.
            mark = ""
            if row["useful_pct"] > 100.5:
                mark = " !"
                suspect = True
            out.append(
                f"| {row['stage']}{mark} | {row['model_flops'] / 1e9:.2f} | "
                f"{row['flops'] / 1e9:.2f} | {row['useful_pct']:.1f}% | "
                f"{row['us'] / 1e3:.1f} | {row['model_gflops']:.1f} | "
                f"{row['model_util_pct']:.1f}% |"
            )
        model_total = sum(row["model_flops"] for row in priced)
        issued_total = sum(row["flops"] for row in priced)
        us_total = sum(row["us"] for row in priced)
        rate = model_total / (us_total * 1e3) if us_total else 0.0
        out.append(
            f"| **TOTAL** | **{model_total / 1e9:.2f}** | "
            f"**{issued_total / 1e9:.2f}** | "
            f"**{(100.0 * model_total / issued_total if issued_total else 0.0):.1f}%** | "
            f"**{us_total / 1e3:.1f}** | **{rate:.1f}** | "
            f"**{(100.0 * rate / peak if peak else 0.0):.1f}%** |"
        )
        out.append("")
        if suspect:
            out += [
                "`!` marks a stage whose issued FLOPs came out BELOW the "
                "model's requirement. Padding and alignment can only add work, "
                "so that stage's own FLOP accounting is undercounting what it "
                "issued -- treat its `% of peak` in the stage summary above as "
                "understated, and the effective rate here as the reliable one.",
                "",
            ]
        return out

    def _profile_tables(self, stages) -> list[str]:
        """Per-phase markdown tables, one per profiled stage.

        Aggregation, the serial-phase peak guard and the column set are shared
        with the terminal breakdown (print_profile_table), so the .md and the
        console can never disagree about a phase.
        """
        peak = self.vis_peak_gflops()
        out: list[str] = []
        for title, note, results in stages:
            if not results:
                continue
            rows = self._aggregate_vis_profile(results)
            total = sum(row["ms"] for row in rows) or 1.0
            out += [f"### {title}", ""]
            if note:
                out += [note, ""]
            out += [
                "| Phase | Calls | Total ms | Share | GFLOP | GFLOPS | % of peak |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            for name, n, ms, share, gflop, gflops, util in self._vis_profile_table(
                rows, total
            ):
                out.append(
                    f"| {name} | {n} | {ms:.2f} | {share:.1f}% | {gflop:.2f} | "
                    f"{gflops:.1f} | {util:.1f}% |"
                )
            total_gflop = sum(row["flops"] for row in rows) / 1e9
            total_rate = total_gflop / (total / 1e3) if total else 0.0
            out.append(
                f"| **TOTAL** | {len(results)} | **{total:.2f}** | 100.0% | "
                f"**{total_gflop:.2f}** | **{total_rate:.1f}** | "
                f"**{(100.0 * total_rate / peak if peak else 0.0):.1f}%** |"
            )
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

    def _multi_core_lines(self, rows: list[dict], profiles=None) -> list[str]:
        """How well each stage actually used the engine split.

        ``x 1-engine peak`` is the same measurement as ``% of peak``, just
        expressed as a speedup, so restating it as an efficiency column would
        add nothing.  What the table adds is the IMPLIED SERIAL FRACTION: solve
        Amdahl for s given the achieved speedup S over N engines,
        ``s = (N/S - 1) / (N - 1)``.  It is an upper bound on the truly serial
        work, because everything else that costs time -- rendezvous waits, DMA
        stalls, tiles that do not fill the MAC array -- lands in it too.  A
        --profile run is what separates those: phases marked ``*`` there are
        the genuinely engine-0-only ones.
        """
        cores = getattr(self, "multi_core", 1) or 1
        core_peak = self.per_core_peak_gflops()
        out = [
            "## Multi-core scaling",
            "",
            f"This model requires exactly {REQUIRED_ENGINES} engines, so no "
            f"1-engine baseline can be measured for comparison.  Speedup is "
            f"therefore taken against one engine's PEAK "
            f"({core_peak:.1f} GFLOPS), which makes it a lower bound on the "
            f"sharding's true benefit: a stage that is inefficient for reasons "
            f"unrelated to sharding is charged for that here as well.",
            "",
            f"| Stage | Throughput (GFLOPS) | x 1-engine peak (max "
            f"{cores}.00x) | Implied serial fraction |",
            "| :--- | ---: | ---: | ---: |",
        ]
        for row in rows:
            speedup = row["speedup"]
            if speedup > 0 and cores > 1:
                serial = (cores / speedup - 1.0) / (cores - 1)
                serial_s = f"{100.0 * max(0.0, min(1.0, serial)):.1f}%"
            else:
                serial_s = "n/a"
            out.append(
                f"| {row['stage']} | {row['gflops']:.1f} | {speedup:.2f}x | "
                f"{serial_s} |"
            )
        out.append("")

        # With profile data the engine-0-only phases can be named outright,
        # which is the actionable half: those are what sharding has to remove.
        serial_rows = []
        for title, _note, results in (profiles or []):
            if not results:
                continue
            aggregated = self._aggregate_vis_profile(results)
            total_ms = sum(row["ms"] for row in aggregated) or 1.0
            for row in aggregated:
                if row.get("serial"):
                    serial_rows.append(
                        (title, row["phase"], row["ms"],
                         100.0 * row["ms"] / total_ms)
                    )
        if serial_rows:
            out += [
                "Phases that ran on engine 0 alone, and their share of their "
                "stage's FPGA time:",
                "",
                "| Stage | Phase | ms | Share of stage |",
                "| :--- | :--- | ---: | ---: |",
            ]
            for title, phase, ms, share in serial_rows:
                out.append(f"| {title} | {phase} | {ms:.2f} | {share:.1f}% |")
            out.append("")
        elif not profiles:
            out += [
                "Run with `--profile` to attribute that serial fraction to "
                "named phases.",
                "",
            ]
        return out

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
        high = getattr(self, "_high_benchmark", None)
        if high:
            lines[4:4] = [
                "## High-load benchmark contract",
                "",
                f"- **Camera workload:** {high['vision_runs']} independent "
                f"896x896 frames ({high['vision_soft_tokens']} vision tokens total)",
                f"- **Spoken query:** {high['audio_seconds']:.1f} s",
                f"- **Aggregate input:** {high['aggregate_tokens']} tokens",
                f"- **Measured prefill:** {high['prefill_chunks']} independent x "
                f"{high['chunk_tokens']} tokens; each pass runs on the FPGA and "
                "reuses the resident KV allocation",
                "- **Numerics:** intentionally unchecked; media embeddings and KV "
                "history do not carry between benchmark chunks",
                "",
                "This is a throughput/latency benchmark, not a claim that a "
                "monolithic 6144-token request fits. The current physical KV "
                f"capacity remains {self.KV_CONTEXT_CAPACITY} rows.",
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
                f"(IF4 + BF16 V/O, IF8 embedding)"
            )
        lines += self._program_section_lines()
        layout = self.dram_layout_lines()
        if layout:
            # The per-core table is real Markdown; everything else is an aligned
            # listing that only survives inside a fence. Emit each contiguous run
            # in the form it needs.
            lines += ["", "## DRAM layout"]
            run: list[str] = []
            run_is_table = False

            def _flush(target=lines):
                if not run:
                    return
                if run_is_table:
                    target.append("")
                    target.extend(run)
                else:
                    target += ["", "```"] + run + ["```"]
                run.clear()

            for line in layout:
                stripped = line.rstrip()
                is_table = stripped.startswith("|")
                if is_table != run_is_table:
                    _flush()
                    run_is_table = is_table
                if stripped or not is_table:
                    run.append(stripped)
            _flush()
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
                f"{cores}.00x.",
                "",
            ]
            lines += self._stage_table(rows)
            lines.append("")
            lines += self._effective_table(rows)

        if getattr(self, "_vis_latency_us", None):
            dims = self._vision_dims()
            vision_runs = int(getattr(self, "_high_vision_runs", 1))
            lines += [
                "## Vision",
                "",
                f"- **Image:** `{os.path.basename(getattr(args, 'image', '') or '')}` "
                f"x {vision_runs} frame(s) -> {vision_runs * dims['VS']} patches "
                f"-> {vision_runs * dims['NUM_MERGED_TOKENS']} soft tokens",
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
            ttft_kind = "measured segmented; " if high else ""
            lines += [
                "## Time to first token",
                "",
                f"- **TTFT ({ttft_kind}HW counter; {' + '.join(covered)}):** "
                f"{(enc_hw_us + pre_hw_us) / 1e3:.1f} ms",
                f"- **TTFT ({ttft_kind}CPU timer; {' + '.join(covered)}):** "
                f"{enc_wall + pre_wall:.2f} s",
            ]
            if high:
                estimate_us = float(high["monolithic_prefill_estimate_us"])
                lines += [
                    f"- **Estimated monolithic prefill:** {estimate_us / 1e3:.1f} ms "
                    f"for {high['aggregate_tokens']} tokens, using the measured "
                    "segmented effective GFLOPS and static full-context FLOPs",
                    f"- **Estimated monolithic TTFT:** "
                    f"{(enc_hw_us + estimate_us) / 1e3:.1f} ms",
                ]
            lines.append("")

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

        if rows:
            lines += self._multi_core_lines(rows, profiles)

        if profiles:
            lines += [
                "## Per-phase profile",
                "",
                "Phase latencies come from the HW counter between per-phase "
                "HALTs: they exclude host time but include one stop/restart per "
                "phase, so the SHARE column is the number to act on -- it says "
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
            scheduler = MultiEngineScheduler(
                self,
                num_engines=REQUIRED_ENGINES,
                engine_base_stride=0x00010000,
                arena=self.mc_arena,
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
        return scheduler

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

        Replaces the old per-core ISA table. The ISA slice is 16 MiB of a 1 GiB
        window and was never the interesting number; where the weights, the KV
        cache and the scratch actually sit is.
        """
        MiB = float(2**20)
        arena = self.mc_arena
        ne = arena.num_engines
        win = arena.stride
        out = [
            f"Device: {arena.arena_bytes / 2**30:.0f} GiB, {ne} x "
            f"{win / MiB:.0f} MiB private windows tiling [0x0, 0x{arena.arena_bytes:X}).",
            "",
            "Inside every window, low to high:",
            f"  private weight arena   {arena.weight_bytes() / MiB:7.0f} MiB   "
            f"shards grow UP; shared tensors grow UP above the reserve",
            f"  (shared pool)                      shared weights grow DOWN "
            f"from the top of the arena",
            f"  ISA slice              {arena.isa_bytes / MiB:7.0f} MiB",
            f"  tensor scratch         {arena.tensor_bytes / MiB:7.0f} MiB   "
            f"per-engine private attention scratch",
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
                f"{self.mc_arena.isa_bytes / 2**20:.0f} MiB"
                + (f" ({details} MiB)" if details else "")
            )
        return lines

    def describe_dram_map(self) -> str:
        return "\n".join(
            [
                "U55 8-GiB map:",
                self.mc_arena.describe(),
                f"  PARAMS  0x{self.PARAMS_BASE:09X}..0x{self.PARAMS_LIMIT:09X} "
                f"{(self.PARAMS_LIMIT - self.PARAMS_BASE) / 2**20:.0f} MiB "
                "(vision/audio/LM time-shared)",
                f"  TENSOR  0x{self.TENSOR_BASE:09X}..0x{self.TENSOR_LIMIT:09X} "
                f"{(self.TENSOR_LIMIT - self.TENSOR_BASE) / 2**20:.0f} MiB",
                f"  ISA     0x{self.ISA_BASE:09X}..0x{self.MASTER_ISA_LIMIT:09X} "
                f"{(self.MASTER_ISA_LIMIT - self.ISA_BASE) / 2**20:.0f} MiB",
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


def _default_prompt(args) -> str:
    if args.prompt:
        return args.prompt
    if getattr(args, "preset", None):
        # Fitted later, once the processor can measure the assembled length.
        return _PRESET_PROMPT_BASE[args.preset]
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
            images = [image.copy()]
    if args.audio:
        preset = RUN_PRESETS.get(getattr(args, "preset", None) or "", {})
        samples = _load_audio(
            args.audio, int(cfg["audio"]["sample_rate"]),
            fraction=float(preset.get("audio_fraction", 1.0)),
            target_seconds=preset.get("audio_seconds"))
        audio = [samples]

    # In the joint case, match media order to the requested answer order. This
    # avoids asking the greedy Thinker to jump back over a completed image
    # response before it reaches the short, deterministic transcript check.
    if args.audio:
        content.append({"type": "audio", "audio": args.audio})
    if args.image:
        content.append({"type": "image", "image": args.image})
    content.append({"type": "text", "text": prompt})
    def _assemble(prompt_text: str):
        body = [item for item in content if item["type"] != "text"]
        body.append({"type": "text", "text": prompt_text})
        text = processor.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=False, add_generation_prompt=True)
        call_kwargs: dict[str, Any] = {
            "text": text, "padding": True, "return_tensors": "pt",
        }
        if images is not None:
            call_kwargs["images"] = images
        if audio is not None:
            call_kwargs["audio"] = audio
        return text, processor(**call_kwargs)

    # A preset fixes the media and then GROWS THE PROMPT to a target prefill
    # length. It has to happen here: only the processor knows how many tokens
    # this particular audio clip and image expand to.
    preset_name = getattr(args, "preset", None)
    if preset_name and not args.prompt:
        target = int(RUN_PRESETS[preset_name].get(
            "prefill_chunk_tokens", RUN_PRESETS[preset_name]["prefill_tokens"]
        ))
        prompt = _fit_prompt_to_prefill(
            processor.tokenizer,
            lambda text: int(_assemble(text)[1]["input_ids"].shape[1]),
            target, _PRESET_PROMPT_BASE[preset_name])
    rendered, processed = _assemble(prompt)

    input_ids = processed["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"processor returned input_ids shape {tuple(input_ids.shape)}")
    attention_mask = processed.get("attention_mask")
    if attention_mask is not None and not torch.all(attention_mask == 1).item():
        raise ValueError("single-request processor output unexpectedly contains text padding")
    tokens = [int(value) for value in input_ids[0].tolist()]
    return processor, processed, tokens, prompt, rendered


def _run_vision(
    ue: Qwen25OmniUnifiedEngine, processed, profile: bool = False
) -> torch.Tensor:
    grid = processed["image_grid_thw"]
    if tuple(grid.shape) != (1, 3):
        raise ValueError(f"only one image is supported, got grid {tuple(grid.shape)}")
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
    expected_grid = torch.tensor([[1, side, side]], dtype=grid.dtype)
    if not torch.equal(grid.cpu(), expected_grid):
        raise ValueError(
            f"the encoder is carved for {vis_cfg['image_size']}x"
            f"{vis_cfg['image_size']} (image_grid_thw [1,{side},{side}], "
            f"{vis_cfg['num_patches']} patches -> "
            f"{vis_cfg['num_merged_tokens']} soft tokens), got {grid.tolist()}"
        )
    print("\n--- Vision stage ---")
    started = time.perf_counter()
    ue.vision_weight_init()
    ue.prepare_encoder_input(processed["pixel_values"], grid)
    ue.reset_tensor_dram_addr()
    ue.vision_tensor_init()
    ue.compile_vision_encoder(profile=profile)
    ue.check_master_isa()
    ue.store_program_stage("vision")
    embeddings = ue.run_vision_encoder(profile=profile)
    print(
        f"  vision -> {tuple(embeddings.shape)} in "
        f"{time.perf_counter() - started:.2f}s wall"
    )
    return embeddings


def _run_audio(ue: Qwen25OmniUnifiedEngine, processed):
    print("\n--- Audio stage ---")
    started = time.perf_counter()
    ue.audio_weight_init()
    metadata = ue.prepare_audio_input(
        processed["input_features"], processed["feature_attention_mask"]
    )
    ue.reset_tensor_dram_addr()
    ue.audio_tensor_init()
    ue.compile_audio_encoder()
    ue.check_master_isa()
    ue.store_program_stage("audio")
    embeddings = ue.run_audio_encoder()
    print(
        f"  audio -> {tuple(embeddings.shape)} in "
        f"{time.perf_counter() - started:.2f}s wall"
    )
    return embeddings, metadata


def apply_run_preset(args) -> None:
    """Turn --low / --medium / --high into the media flags they stand for.

    The preset decides WHICH media are present and at what size; the prompt
    that brings the prefill up to the target is fitted later, once the
    processor can measure how many tokens this audio and image expand to. An
    explicitly passed --image/--audio/--vision-res still wins, so a preset can
    be used as a starting point.
    """
    if not args.preset:
        return
    spec = RUN_PRESETS[args.preset]
    if args.audio is None:
        args.audio = DEFAULT_AUDIO
    if spec["image"] and args.image is None:
        args.image = DEFAULT_IMAGE
    if spec["vision_res"] and args.vision_res == DEFAULT_VISION_RES:
        args.vision_res = spec["vision_res"]


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
    parser.add_argument("--prompt", default=None, help="user text prompt")
    presets = parser.add_mutually_exclusive_group()
    presets.add_argument(
        "--low", dest="preset", action="store_const", const="low",
        help=("fixed shape: half-length audio plus a fitted prompt, "
              f"~{RUN_PRESETS['low']['prefill_tokens']} prefill tokens"),
    )
    presets.add_argument(
        "--medium", dest="preset", action="store_const", const="medium",
        help=("fixed shape: 896x896 image (1024 soft tokens) plus full audio "
              f"and a fitted prompt, ~{RUN_PRESETS['medium']['prefill_tokens']} "
              "prefill tokens"),
    )
    presets.add_argument(
        "--high", dest="preset", action="store_const", const="high",
        help=("performance-only fixed shape: 4 x 896x896 camera frames, "
              "6 seconds of audio, and 6144 aggregate input tokens projected "
              "as 3 x one isolated 2048-token FPGA prefill measurement"),
    )
    parser.set_defaults(preset=None)
    parser.add_argument(
        "--image",
        nargs="?",
        const=DEFAULT_IMAGE,
        default=None,
        help=f"optional image; bare --image uses {os.path.basename(DEFAULT_IMAGE)}",
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
        "--profile-ctx",
        type=int,
        default=MAX_CONTEXT_SIZE,
        help=f"context length for the SECOND profiled decode step (default "
             f"{MAX_CONTEXT_SIZE}, the full context). The first is taken right "
             f"after prefill, so the pair brackets decode cost from the "
             f"shortest to the longest KV history. --profile only.",
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
    parser.add_argument(
        "--_high-phase",
        dest="high_phase",
        choices=("vision", "audio", "prefill", "decode"),
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def run_summary_filename(args) -> str:
    """Per-run summary filename encoding the CLI config, e.g.

    ``--dev xdma0 --image --multi-core 8`` ->
    ``qwen2.5_omni_7b_test_xdma0_image_multi-core_8.md``.

    Device and mode are always present, in that order; a profile run is tagged
    so its phase breakdown never overwrites a generation run's summary.

    A PRESET IS PART OF THE NAME because it is not visible in the mode. --low
    is mode "audio" and --medium/--high are "image+audio", exactly like the plain flags
    they build on, so without this a preset run and an ad-hoc run of the same
    mode overwrite each other despite being different shapes -- 849 prefill
    tokens against 154, which is the whole point of having the preset.
    """
    parts = ["qwen2.5_omni_7b_test", args.dev, _result_mode(args)]
    if getattr(args, "preset", None):
        parts.append(args.preset)
    if getattr(args, "profile", False):
        parts.append("profile")
    parts.append(f"multi-core_{args.multi_core}")
    return "_".join(parts) + ".md"


def _write_high_summary(args, phases: dict[str, dict], process_wall: dict[str, float]) -> str:
    """Combine independently executed high-load phase measurements."""
    vision = phases["vision"]
    audio = phases["audio"]
    sample = phases["prefill"]
    decode = phases["decode"]
    chunks = int(RUN_PRESETS["high"]["prefill_chunks"])
    aggregate_tokens = int(sample["tokens"]) * chunks
    prefill = dict(sample)
    for key in ("flops", "model_flops", "hw_us", "wall_s"):
        prefill[key] = float(sample[key]) * chunks
    prefill["detail"] = (
        f"{chunks} x {sample['tokens']} independent chunks = "
        f"{aggregate_tokens} aggregate tokens"
    )
    prefill["gflops"] = (
        prefill["flops"] / (prefill["hw_us"] * 1e3)
        if prefill["hw_us"] else 0.0
    )
    prefill["effective_gflops"] = (
        prefill["model_flops"] / (prefill["hw_us"] * 1e3)
        if prefill["hw_us"] else 0.0
    )
    mono_flops = float(sample["monolithic_model_flops"])
    mono_us = (
        mono_flops / (prefill["effective_gflops"] * 1e3)
        if prefill["effective_gflops"] else 0.0
    )
    rows = [vision, audio, prefill, decode]
    # Every phase result already carries model_flops (the unpadded work this
    # architecture actually owes); turn it into effective GFLOPS for ALL FOUR
    # stages, not prefill alone -- vision and audio were being captured on the
    # wire and then dropped when this table got built.
    for row in rows:
        row["effective_gflops"] = (
            float(row["model_flops"]) / (float(row["hw_us"]) * 1e3)
            if row.get("hw_us") else 0.0
        )
    peak = float(vision["peak_gflops"])
    enc_us = float(vision["hw_us"]) + float(audio["hw_us"])
    segmented_ttft_us = enc_us + float(prefill["hw_us"])
    monolithic_ttft_us = enc_us + mono_us

    lines = [
        "# qwen2.5_omni_7b high-load benchmark",
        "",
        "## Contract",
        "",
        "- **Execution:** vision, audio, prefill, and decode each ran in a "
        "fresh Python process with its own eight-core reset, compilation, and "
        "FPGA execution.",
        "- **Camera:** 4 independent 896x896 frames, 4096 vision tokens total.",
        "- **Audio:** 6.0 seconds, the configured 600-mel-frame maximum.",
        f"- **Prefill:** {aggregate_tokens} aggregate tokens measured as "
        f"{chunks} x {sample['tokens']} independent resident chunks, real FPGA "
        "runs replayed -- not a shorter stand-in. A genuine single continuous "
        f"{aggregate_tokens}-token prefill does not fit: its own buffers "
        "(down-projection TP scratch) scale with prompt length and collide "
        "with the ~707 MiB/core of real resident weights in the same 1 GiB "
        "window, and there is no way to free that room without changing the "
        "TP degree or the weight residency being measured.",
        f"- **Decode:** measured with the KV cache pre-filled to the same "
        f"{aggregate_tokens}-token depth; rows beyond the real prefill chunk "
        "are zero-filled placeholders, so decode's attention shape and DMA "
        "cost are real but its logits are not. Decode's own buffers (KV "
        "cache, attention scratch) are not weight-adjacent, so this one "
        "fits without the prefill tradeoff above.",
        "- **Numerics/coherency:** intentionally unchecked throughout.",
        "",
        "## Hardware-counter performance",
        "",
        "| Stage | Shape | Work (GFLOP) | FPGA time (ms) | Issued GFLOPS | "
        "% peak (issued) | Effective GFLOPS | % peak (effective) | "
        "CPU execution wall (s) |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        gflops = float(row["gflops"])
        eff = float(row["effective_gflops"])
        lines.append(
            f"| {row['stage']} | {row['detail']} | {float(row['flops']) / 1e9:.1f} | "
            f"{float(row['hw_us']) / 1e3:.1f} | {gflops:.1f} | "
            f"{100.0 * gflops / peak:.1f}% | {eff:.1f} | "
            f"{100.0 * eff / peak:.1f}% | {float(row['wall_s']):.2f} |"
        )
    lines += [
        "",
        "## TTFT",
        "",
        f"- **Projected segmented TTFT (HW counters):** {segmented_ttft_us / 1e3:.1f} ms",
        f"- **Projected segmented TTFT (CPU execution timers):** "
        f"{float(vision['wall_s']) + float(audio['wall_s']) + float(prefill['wall_s']):.2f} s",
        f"- **Estimated monolithic 6144-token prefill:** {mono_us / 1e3:.1f} ms "
        f"({mono_flops / 1e9:.1f} model GFLOP at the measured "
        f"{prefill['effective_gflops']:.1f} effective GFLOPS)",
        f"- **Estimated monolithic TTFT:** {monolithic_ttft_us / 1e3:.1f} ms",
        "",
        "The monolithic figure is an extrapolation calibrated by FPGA-measured "
        "throughput; it is not a 6144-row hardware execution.",
        "",
        "## Decode",
        "",
        f"- **Resident context:** {decode['context']} tokens",
        f"- **Steps:** {decode['steps']}",
        f"- **First token:** {1e6 / float(decode['first_us']):.2f} tok/s",
        f"- **Average (HW counter):** {1e6 * int(decode['steps']) / float(decode['hw_us']):.2f} tok/s",
        f"- **Average (CPU timer):** {int(decode['steps']) / float(decode['wall_s']):.2f} tok/s",
        f"- **Throughput:** {float(decode['gflops']):.1f} GFLOPS "
        f"({100.0 * float(decode['gflops']) / peak:.1f}% peak)",
        "",
        "## Independent phase process wall time",
        "",
        "This includes reset, weight loading, compilation, and execution and is "
        "reported separately from inference TTFT.",
        "",
        "| Phase process | Wall (s) |",
        "| :--- | ---: |",
    ]
    for name in ("vision", "audio", "prefill", "decode"):
        lines.append(f"| {name} | {process_wall[name]:.2f} |")
    lines.append("")
    out = args.summary or os.path.join(SCRIPT_DIR, run_summary_filename(args))
    with open(out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return out


def _run_high_orchestrator(args) -> None:
    """Run every high-load phase in a fresh process, then combine metrics."""
    phases: dict[str, dict] = {}
    process_wall: dict[str, float] = {}
    for phase in ("vision", "audio", "prefill", "decode"):
        command = [
            sys.executable, os.path.abspath(__file__),
            "--dev", args.dev,
            "--multi-core", str(args.multi_core),
            "--high", "--_high-phase", phase,
            "--max-new-tokens", "32",
            "--no-summary",
        ]
        print(f"\n{'=' * 72}\nHIGH PHASE: {phase}\n{'=' * 72}", flush=True)
        started = time.perf_counter()
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        result = None
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            if line.startswith("HIGH_PHASE_RESULT: "):
                result = json.loads(line[len("HIGH_PHASE_RESULT: "):])
        returncode = proc.wait()
        process_wall[phase] = time.perf_counter() - started
        if returncode:
            raise RuntimeError(f"high phase {phase!r} exited with status {returncode}")
        if result is None:
            raise RuntimeError(f"high phase {phase!r} produced no metric record")
        phases[phase] = result
    out = _write_high_summary(args, phases, process_wall)
    print(f"\nWrote combined high-load summary: {out}")
    print("HIGH_RESULT: " + json.dumps({
        "summary": out, "phases": phases, "process_wall_s": process_wall,
    }, ensure_ascii=False))


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    apply_run_preset(args)
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.profile_ctx < 2 or args.profile_ctx > MAX_CONTEXT_SIZE:
        parser.error(f"--profile-ctx must be between 2 and {MAX_CONTEXT_SIZE}")
    if args.preset == "high" and args.profile:
        parser.error("--high already performs a multi-pass benchmark; do not combine it with --profile")
    if args.no_summary and args.summary:
        parser.error("--summary and --no-summary are mutually exclusive")
    if args.preset == "high" and args.high_phase is None:
        _run_high_orchestrator(args)
        return
    with _exclusive_run_lock():
        _main_locked(parser, args)


def _main_locked(parser: argparse.ArgumentParser, args) -> None:
    """Run artifact preparation and FPGA execution under the global lock."""
    global MAX_CONTEXT_SIZE
    if getattr(args, "high_phase", None) == "decode":
        # The decode phase's whole point is a resident KV depth this map
        # cannot normally hold (that is why --high chunks prefill at all).
        # Numerics are already unchecked for --high; widen ONLY the KV/decode
        # context ceiling (MAX_CONTEXT_SIZE) in THIS ISOLATED SUBPROCESS so
        # decode's shape and DMA cost are real at the aggregate length instead
        # of at one 2048-token chunk. PREFILL_MAX_SEQ_LEN stays untouched: it
        # sizes the PREFILL program's own buffers (I/O, TP-down scratch) to
        # what this subprocess actually prefills -- one real 2048-token chunk
        # -- and bloating it alongside MAX_CONTEXT_SIZE ballooned buffers that
        # have nothing to do with KV depth for no reason.
        #
        # A REAL, CONTINUOUS 6144-token prefill was tried and reverted: its
        # own buffers scale with the compiled length too (TP-down scratch is
        # tp_ne * P * H, ~339.5 MiB at P=6144), and that has to coexist with
        # the ~707 MiB/core of REAL resident weights in the same 1 GiB window.
        # There is no bookkeeping fix for that -- shrinking the declared
        # private reserve does not help, since shared_free() takes the max
        # with the ACTUAL weight cursor, which real weight staging advances
        # regardless of what is declared. The only ways through are reducing
        # the down-projection's TP degree (understates real prefill
        # throughput) or shrinking real weight residency (skips real weight
        # DMA traffic) -- both change what is measured. So prefill keeps its
        # x3 chunk replay below, which needs neither: the three 2048-token
        # chunks are independent and identical in shape, so replaying one
        # real, fully-TP-sharded, fully-resident measurement three times is
        # the true cost of three of them, not an approximation.
        aggregate = (int(RUN_PRESETS["high"]["prefill_chunk_tokens"]) - 1) * \
            int(RUN_PRESETS["high"]["prefill_chunks"])
        target = ((aggregate + args.max_new_tokens + 63) // 64) * 64
        MAX_CONTEXT_SIZE = max(MAX_CONTEXT_SIZE, target)
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
    if len(context) > PREFILL_INPUT_TOKEN_LIMIT:
        raise ValueError(
            f"templated prompt needs {len(context)} prefill tokens; limit is "
            f"{PREFILL_INPUT_TOKEN_LIMIT}. Shorten the prompt or media input."
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
        vision_res=args.vision_res, **engine_kwargs
    )
    ue.configure_runtime_artifacts(params_path, processor_dir)
    ue.tokenizer = processor.tokenizer
    ue.processor = processor
    ue._prompt_text = prompt
    print(ue.describe_dram_map())

    high_phase = getattr(args, "high_phase", None)
    if high_phase == "vision":
        runs = int(RUN_PRESETS["high"]["vision_runs"])
        measurements = []
        embeddings = None
        for index in range(runs):
            print(f"\n=== Camera frame {index + 1}/{runs} ===")
            if index == 0:
                embeddings = _run_vision(ue, processed, profile=False)
            else:
                started = time.perf_counter()
                embeddings = ue.run_vision_encoder(profile=False)
                print(
                    f"  vision -> {tuple(embeddings.shape)} in "
                    f"{time.perf_counter() - started:.2f}s wall"
                )
            measurements.append((
                int(ue._vis_total_flops), float(ue._vis_latency_us),
                float(ue._vis_wall_s),
            ))
        ue._high_vision_runs = runs
        ue._vis_total_flops = sum(item[0] for item in measurements)
        ue._vis_latency_us = sum(item[1] for item in measurements)
        ue._vis_wall_s = sum(item[2] for item in measurements)
        row = ue.stage_metrics(args)[0]
        print("HIGH_PHASE_RESULT: " + json.dumps({
            "stage": row["stage"], "detail": row["detail"],
            "flops": row["flops"], "model_flops": row["model_flops"],
            "hw_us": row["us"], "wall_s": row["wall"],
            "gflops": row["gflops"], "peak_gflops": ue.vis_peak_gflops(),
        }))
        return

    if high_phase == "audio":
        _embeddings, _metadata = _run_audio(ue, processed)
        row = ue.stage_metrics(args)[0]
        print("HIGH_PHASE_RESULT: " + json.dumps({
            "stage": row["stage"], "detail": row["detail"],
            "flops": row["flops"], "model_flops": row["model_flops"],
            "hw_us": row["us"], "wall_s": row["wall"],
            "gflops": row["gflops"], "peak_gflops": ue.vis_peak_gflops(),
        }))
        return

    if high_phase in ("prefill", "decode"):
        print(f"\n--- Isolated {high_phase} LM setup ---")
        ue.lm_weight_init()
        ue.lm_tensor_init()
        ue.compile_prefill(len(context), profile=False)
        # Decoder compilation must precede the first prefill because it copies
        # the down-projection image before prefill repacks that image in place.
        ue.compile_decoder(profile=False)
        ue.check_master_isa()
        ue.store_program_stages("prefill", "decode")
        ue.run_prefill(context, positions=torch.arange(len(context)))
        if high_phase == "prefill":
            row = next(r for r in ue.stage_metrics(args) if r["stage"] == "Prefill")
            aggregate = len(context) * int(RUN_PRESETS["high"]["prefill_chunks"])
            print("HIGH_PHASE_RESULT: " + json.dumps({
                "stage": row["stage"], "detail": row["detail"],
                "tokens": len(context),
                "flops": row["flops"], "model_flops": row["model_flops"],
                "monolithic_model_flops": float(
                    _model_flops.prefill_flops(cfg, aggregate)
                ),
                "hw_us": row["us"], "wall_s": row["wall"],
                "gflops": row["gflops"], "peak_gflops": ue.vis_peak_gflops(),
            }))
            return
        ue.activate_decode_shared_weights()
        real_prefill_rows = int(ue.seq_len)
        aggregate = (int(RUN_PRESETS["high"]["prefill_chunk_tokens"]) - 1) * \
            int(RUN_PRESETS["high"]["prefill_chunks"])
        if aggregate > real_prefill_rows:
            # SPEED ONLY, NOT NUMERICS: rows [real_prefill_rows:aggregate) were
            # never actually prefilled. Fill them with finite zeros (matches
            # lm_reset_attention_state's own convention) so the padding cannot
            # produce NaN, then claim the position so decode's per-step bias
            # and KV stride treat all of it as live. This makes decode run its
            # real attention/matmul shape against the real 6144-token
            # aggregate length instead of against one 2048-token chunk; the
            # logits it produces from the fake rows are meaningless, which is
            # already true of --high generally.
            AHD, KVH, NL = ue.actual_head_dim, ue.num_kv_heads, ue.LAYER_SIZE
            pad_rows = aggregate - real_prefill_rows
            stride = ue.KV_STRIDE_HEAD  # bytes per KV head's full C-row plane
            for cache in (ue.LM_K_CACHE, ue.LM_V_CACHE):
                for li in range(NL):
                    for h in range(KVH):
                        plane = cache + (li * KVH + h) * stride
                        off = plane + real_prefill_rows * AHD * ue.bytes_per_element
                        ue.dma_to_accelerator_memory(
                            off, torch.zeros(pad_rows * AHD, dtype=torch.bfloat16))
            ue.seq_len = aggregate
        resident_context = int(ue.seq_len)
        # High is explicitly timing-only: force all requested steps even if a
        # numerically meaningless token happens to equal EOS.
        ue._decode_stop_token_ids = lambda: set()
        ue.run_decoder(seed, max_new_tokens=args.max_new_tokens)
        row = next(r for r in ue.stage_metrics(args) if r["stage"] == "Decode")
        steps = len(ue._decode_step_us)
        print("HIGH_PHASE_RESULT: " + json.dumps({
            "stage": row["stage"], "detail": row["detail"],
            "context": resident_context, "steps": steps,
            "first_us": float(ue._decode_step_us[0]),
            "flops": row["flops"], "model_flops": row["model_flops"],
            "hw_us": row["us"], "wall_s": row["wall"],
            "gflops": row["gflops"], "peak_gflops": ue.vis_peak_gflops(),
        }))
        return

    image_embeddings = None
    audio_embeddings = None
    audio_metadata = None
    preset_spec = RUN_PRESETS.get(getattr(args, "preset", None) or "", {})
    if args.image:
        vision_runs = int(preset_spec.get("vision_runs", 1))
        vision_measurements = []
        for frame_index in range(vision_runs):
            if vision_runs > 1:
                print(f"\n=== High benchmark camera frame {frame_index + 1}/{vision_runs} ===")
            if frame_index == 0:
                # Compile and package one shape-specific program. Camera frames
                # share geometry, so later frames execute the same validated
                # FPGA image rather than consuming another worker ISA slice.
                image_embeddings = _run_vision(ue, processed, profile=args.profile)
            else:
                frame_started = time.perf_counter()
                image_embeddings = ue.run_vision_encoder(profile=False)
                print(
                    f"  vision -> {tuple(image_embeddings.shape)} in "
                    f"{time.perf_counter() - frame_started:.2f}s wall"
                )
            vision_measurements.append((
                int(ue._vis_total_flops),
                float(ue._vis_latency_us),
                float(getattr(ue, "_vis_wall_s", 0.0)),
            ))
        if vision_runs > 1:
            ue._high_vision_runs = vision_runs
            ue._vis_total_flops = sum(row[0] for row in vision_measurements)
            ue._vis_latency_us = sum(row[1] for row in vision_measurements)
            ue._vis_wall_s = sum(row[2] for row in vision_measurements)
            ue._vis_gflops = (
                ue._vis_total_flops / (ue._vis_latency_us * 1e3)
                if ue._vis_latency_us else 0.0
            )
    if args.audio:
        audio_embeddings, audio_metadata = _run_audio(ue, processed)

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

    if args.preset == "high":
        # Repeated encoder rendezvous leave no useful queue state for the LM.
        # A software reset clears all eight queues without touching DRAM, so
        # the final frame/audio embeddings remain available for prefill.
        print("\n--- High benchmark encoder/LM queue reset ---")
        reset_build = reset_selected_engines()
        if reset_build != fpga_build:
            raise RuntimeError(
                f"FPGA build changed across high benchmark reset: "
                f"0x{fpga_build:08x} -> 0x{reset_build:08x}"
            )
        print("Software reset + HALT probe passed on engines 0-7; DRAM preserved")

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
    ue.lm_weight_init()
    ue.lm_tensor_init()
    ue.compile_prefill(len(context), profile=args.profile)
    # Decoder setup installs the device-side embedding and copies every
    # reusable projection into private windows. BF16 O addresses are compiled
    # now, but their shared overlay is deliberately deferred until prefill.
    ue.compile_decoder(profile=args.profile)
    ue.check_master_isa()
    # These bodies are address-coupled: decoder starts immediately after this
    # exact prefill image. Publish both in one programs.bin generation.
    ue.store_program_stages("prefill", "decode")
    for line in ue.isa_usage_lines():
        print(line)
    prefill_chunks = int(preset_spec.get("prefill_chunks", 1))
    prefill_measurements = []
    for chunk_index in range(prefill_chunks):
        if prefill_chunks > 1:
            print(
                f"\n=== High benchmark prefill chunk "
                f"{chunk_index + 1}/{prefill_chunks} ({len(context)} tokens) ==="
            )
        ue.run_prefill(
            context,
            image_embeddings=image_embeddings,
            audio_embeddings=audio_embeddings,
            positions=positions[: len(context)],
            profile=args.profile,
        )
        prefill_measurements.append((
            int(ue._prefill_flops),
            float(ue._latency_prefill_us),
            float(getattr(ue, "_prefill_wall_s", 0.0)),
            float(ue._model_flops_prefill() or 0.0),
        ))
    if prefill_chunks > 1:
        aggregate_tokens = len(context) * prefill_chunks
        ue._prefill_flops = sum(row[0] for row in prefill_measurements)
        ue._latency_prefill_us = sum(row[1] for row in prefill_measurements)
        ue._prefill_wall_s = sum(row[2] for row in prefill_measurements)
        ue._prefill_seq_len_run = aggregate_tokens
        ue._high_segmented_prefill_model_flops = sum(
            row[3] for row in prefill_measurements
        )
        ue._prefill_gflops = (
            ue._prefill_flops / (ue._latency_prefill_us * 1e3)
            if ue._latency_prefill_us else 0.0
        )
        monolithic_flops = float(_model_flops.prefill_flops(cfg, aggregate_tokens))
        segmented_effective_gflops = (
            ue._high_segmented_prefill_model_flops
            / (ue._latency_prefill_us * 1e3)
            if ue._latency_prefill_us else 0.0
        )
        monolithic_estimate_us = (
            monolithic_flops / (segmented_effective_gflops * 1e3)
            if segmented_effective_gflops else 0.0
        )
        dims = ue._vision_dims()
        ue._high_benchmark = {
            "vision_runs": int(getattr(ue, "_high_vision_runs", 1)),
            "vision_soft_tokens": (
                int(getattr(ue, "_high_vision_runs", 1))
                * int(dims["NUM_MERGED_TOKENS"])
            ),
            "audio_seconds": float(preset_spec["audio_seconds"]),
            "prefill_chunks": prefill_chunks,
            "chunk_tokens": len(context),
            "aggregate_tokens": aggregate_tokens,
            "segmented_prefill_model_flops": ue._high_segmented_prefill_model_flops,
            "monolithic_prefill_model_flops": monolithic_flops,
            "monolithic_prefill_estimate_us": monolithic_estimate_us,
        }
        print(
            f"\n[High benchmark] measured {prefill_chunks} x {len(context)} = "
            f"{aggregate_tokens} aggregate prefill tokens; monolithic "
            f"{aggregate_tokens}-token prefill estimate "
            f"{monolithic_estimate_us / 1e6:.2f}s HW"
        )
    ue.activate_decode_shared_weights()

    if args.profile:
        # The checkpointed decoder is the one published in programs.bin, so it
        # is also the one that must run: run_decode_step_profiled refuses any
        # other image.  Two steps bracket decode cost -- one at the prompt's
        # own context, one at --profile-ctx -- because a step's price is set by
        # the KV length, while the projections stay the same size.
        prof_program = ue._decoder_program
        prof_checkpoints = list(ue._decoder_checkpoints)
        prof_workers = list(getattr(ue, "_decoder_workers", []))
        print(f"\n--- Profiled decode: 1st token (ctx {ue.seq_len}) ---")
        first_results, next_token, aligned_first = ue.run_decode_step_profiled(
            seed, prof_program, prof_checkpoints, workers=prof_workers
        )
        ctx_first = ue.seq_len
        if args.profile_ctx - 1 > ue.seq_len:
            # --profile measures TIME, not numerics.  Forcing the position is
            # how the long-context step is reached at all: the model hits EOS
            # long before the context fills, and a checkpointed program costs
            # one host round trip per phase per token.  KV rows past the prompt
            # are zeros, which does not change latency.
            ue.seq_len = args.profile_ctx - 1
            print(f"  forcing ctx {args.profile_ctx} (timing only)")
        print(f"\n--- Profiled decode: at context (ctx {ue.seq_len}) ---")
        ctx_results, _token, aligned_ctx = ue.run_decode_step_profiled(
            next_token, prof_program, prof_checkpoints, workers=prof_workers
        )
        profiles = []
        if args.image:
            dims = ue._vision_dims()
            profiles.append((
                "Vision encoder",
                f"{dims['VS']} patches -> {dims['NUM_MERGED_TOKENS']} soft tokens.",
                getattr(ue, "_vis_profile", None),
            ))
        profiles += [
            ("Prefill", f"{len(context)} tokens.",
             getattr(ue, "_prefill_profile", None)),
            ("Decode - 1st token",
             f"Context {ctx_first} tokens (aligned {aligned_first}).",
             first_results),
            ("Decode - at context",
             f"Context {ue.seq_len} tokens (aligned {aligned_ctx}).",
             ctx_results),
        ]
        for title, note, results in profiles:
            if results:
                ue.print_profile_table(title, results, note=note)
        print(f"\nThinker profile done in {time.perf_counter() - started:.2f}s wall")
        _write_summary(profiles)
        return

    print("\n--- Decode run ---")
    _, decoded_text = ue.run_decoder(seed, max_new_tokens=args.max_new_tokens)
    lm_wall = time.perf_counter() - started
    print(f"\nThinker stage done in {lm_wall:.2f}s wall")

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
    if program_manifest["section_count"] != REQUIRED_ENGINES * len(
        expected_program_stages
    ):
        raise RuntimeError(
            "combined programs.bin does not contain one master + seven worker "
            "sections for every executed stage"
        )
    try:
        _summary_rows = ue.stage_metrics(args)
    except Exception:  # noqa: BLE001 - reporting must not fail a good run
        _summary_rows = []
    reported_prefill_tokens = int(
        getattr(ue, "_high_benchmark", {}).get("aggregate_tokens", len(context))
    )
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
        "prompt_tokens": reported_prefill_tokens + (prefill_chunks if prefill_chunks > 1 else 1),
        "rope_delta": rope_delta,
    }
    if getattr(ue, "_high_benchmark", None):
        result["high_benchmark"] = ue._high_benchmark
    print("TEST_RESULT: " + json.dumps(result, ensure_ascii=False))
    _write_summary()


if __name__ == "__main__":
    main()
