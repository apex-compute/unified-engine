#!/usr/bin/env python3
"""Qwen2.5-Omni-7B Thinker on the eight-engine, 8-GiB U55 map.

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
from multi_engine_shard import (MULTICORE_WINDOW_BYTES, MultiEngineScheduler,
                                PrivateArena, multicore_arena_bytes,
                                require_multicore_dram)
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
MAX_CONTEXT_SIZE = 2048
PREFILL_INPUT_TOKEN_LIMIT = 384
# Eight row-sharded engines require one full 64-row hardware block apiece.
# Logical prompts stay capped at 384; the remaining rows are finite, masked
# execution padding and do not become visible context tokens.
PREFILL_MAX_SEQ_LEN = REQUIRED_ENGINES * 64

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
    """Concrete Thinker engine and its fixed eight-engine U55 memory map."""

    def __init__(self, script_dir: str | None = None, multi_core: int = 8,
                 fpga_build: int | None = None):
        if multi_core != REQUIRED_ENGINES:
            raise ValueError(
                f"Qwen2.5-Omni-7B requires exactly {REQUIRED_ENGINES} engines, "
                f"got {multi_core}"
            )
        reported_cores = user_dma_core.ANDROMEDA_CORE_COUNT
        if reported_cores is not None and reported_cores < REQUIRED_ENGINES:
            raise ValueError(
                f"the U55 image must report at least {REQUIRED_ENGINES} engines; "
                f"HW_INFO reports {reported_cores}"
            )
        # At LEAST 8 GiB, not exactly: the map needs 8 GiB and a larger device
        # simply leaves the top unused. The check lives in the library so every
        # multi-core model states the same requirement the same way.
        require_multicore_dram(multi_core, "Qwen2.5-Omni-7B")

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

        # U55 8-GiB DRAM map
        #
        #   [0, 4 GiB)       8 x 512-MiB private engine windows
        #                     (504 MiB weight shards + 8 MiB scratch each)
        #   [4 GiB, 7.625)   transient params (vision, audio, then LM)
        #   [7.625, 7.90625) activations/KV cache
        #   [7.90625, 8 GiB) ISA: 40 MiB master + 7 x 8 MiB workers
        #
        # Prefill Q/K/O/GATE/UP/DOWN and the untied head are IF4; V stays BF16
        # for attention accuracy, putting shared device LM weights at ~3655 MiB.
        # Decode omits O from its private shards and phase-loads the 686-MiB BF16
        # O overlay as eight 85.75-MiB upper-PARAMS stripes only after full
        # prefill. The distinct IF8 embedding is loaded after decode sharding.
        # The measured private total is 501.27 MiB/core
        # (434.0 MiB decoder + 67.29 MiB embedding), leaving 2.73 MiB guarded
        # headroom inside the 504-MiB weight arena; lookup/dequantization remains
        # on the FPGA.
        self.DRAM_END = 0x200000000
        self.PARAMS_BASE = 0x100000000
        self.PARAMS_LIMIT = 0x1E8000000
        self.TENSOR_BASE = self.PARAMS_LIMIT
        self.TENSOR_LIMIT = 0x1FA000000
        self.ISA_BASE = self.TENSOR_LIMIT
        self.VISION_WEIGHT_BASE = self.PARAMS_BASE

        self.MASTER_ISA_RESERVE = 40 * 2**20
        self.WORKER_ISA_BASE = self.ISA_BASE + self.MASTER_ISA_RESERVE
        self.WORKER_ISA_STRIDE = 8 * 2**20
        if self.WORKER_ISA_BASE + 7 * self.WORKER_ISA_STRIDE != self.DRAM_END:
            raise AssertionError("worker ISA slices do not terminate at 8 GiB")

        # Engine 0 uses the master ISA area.  PrivateArena still keeps window 0
        # reserved for its decode weight shard, hence the external-ISA base is
        # one stride before the first actually used worker slice.
        assert multicore_arena_bytes(REQUIRED_ENGINES) == self.PARAMS_BASE, (
            f"{REQUIRED_ENGINES} x {MULTICORE_WINDOW_BYTES // 2**20} MiB private "
            f"windows end at 0x{multicore_arena_bytes(REQUIRED_ENGINES):X}, not "
            f"at the params base 0x{self.PARAMS_BASE:X}")
        self.mc_arena = PrivateArena(
            REQUIRED_ENGINES,
            arena_base=0,
            arena_bytes=multicore_arena_bytes(REQUIRED_ENGINES),
            tensor_bytes=8 * 2**20,
            external_isa=(
                self.WORKER_ISA_BASE - self.WORKER_ISA_STRIDE,
                self.WORKER_ISA_STRIDE,
            ),
            verbose=True,
        )

        super().__init__(
            BASE_ADDR=user_dma_core.UE_0_BASE_ADDR,
            params_dram_base=self.PARAMS_BASE,
            program_dram_base=self.ISA_BASE,
            tensor_dram_base=self.TENSOR_BASE,
        )

        self.script_dir = script_dir or SCRIPT_DIR
        self._cfg = self.load_config(script_dir=self.script_dir)
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
            "private_arena_bytes": self.PARAMS_BASE,
            "private_window_bytes": 0x20000000,
            "model_base": self.PARAMS_BASE,
            "params_limit": self.PARAMS_LIMIT,
            "tensor_limit": self.TENSOR_LIMIT,
            "dram_limit": self.DRAM_END,
        }
        for name, wanted in expected.items():
            raw = hw.get(name)
            actual = int(raw, 0) if isinstance(raw, str) else int(raw)
            if actual != wanted:
                raise ValueError(
                    f"config hardware.{name}={raw!r}, expected 0x{wanted:X}"
                )
        if self.mc_arena.stride != int(hw["private_window_bytes"], 0):
            raise AssertionError("PrivateArena did not produce 512-MiB windows")
        if self.mc_arena.weight_bytes() != 504 * 2**20:
            raise AssertionError("each engine must have 504 MiB for decode shards")
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
                    "worker_isa_base": self.WORKER_ISA_BASE,
                    "worker_isa_stride": self.WORKER_ISA_STRIDE,
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
            rows.append({
                "stage": "Vision encoder",
                "detail": f"{dims['VS']} patches -> "
                          f"{dims['NUM_MERGED_TOKENS']} soft tokens",
                "flops": float(self._vis_total_flops),
                "model_flops": self._model_flops_vision(dims),
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
        isa = self.isa_usage_lines()
        if isa:
            lines += ["", "### ISA usage", "", "```"]
            lines += [line.rstrip() for line in isa]
            lines += ["```"]
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
            lines += [
                "## Vision",
                "",
                f"- **Image:** `{os.path.basename(getattr(args, 'image', '') or '')}` "
                f"-> {dims['VS']} patches -> {dims['NUM_MERGED_TOKENS']} soft tokens",
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
            lines += [
                "## Time to first token",
                "",
                f"- **TTFT (HW counter; {' + '.join(covered)}):** "
                f"{(enc_hw_us + pre_hw_us) / 1e3:.1f} ms",
                f"- **TTFT (CPU timer; {' + '.join(covered)}):** "
                f"{enc_wall + pre_wall:.2f} s",
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
        if end > self.WORKER_ISA_BASE:
            raise MemoryError(
                f"{stage} master program ends at 0x{end:X}, beyond the worker ISA base "
                f"0x{self.WORKER_ISA_BASE:X}"
            )

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
            # deliberately overwrite the same 8-MiB worker slice.
            peak = max(per_stage.values(), default=0)
            details = ", ".join(
                f"{name} {size / 2**20:.2f}"
                for name, size in sorted(per_stage.items())
            )
            lines.append(
                f"  core {idx} ISA peak: {peak / 2**20:.2f} / 8 MiB"
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
                f"  ISA     0x{self.ISA_BASE:09X}..0x{self.DRAM_END:09X} "
                f"{(self.DRAM_END - self.ISA_BASE) / 2**20:.0f} MiB",
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


def _load_audio(path: str, sample_rate: int) -> np.ndarray:
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
    return mono.contiguous().numpy().astype(np.float32, copy=False)


def _default_prompt(args) -> str:
    if args.prompt:
        return args.prompt
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
        samples = _load_audio(args.audio, int(cfg["audio"]["sample_rate"]))
        audio = [samples]

    # In the joint case, match media order to the requested answer order. This
    # avoids asking the greedy Thinker to jump back over a completed image
    # response before it reaches the short, deterministic transcript check.
    if args.audio:
        content.append({"type": "audio", "audio": args.audio})
    if args.image:
        content.append({"type": "image", "image": args.image})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    call_kwargs: dict[str, Any] = {
        "text": rendered,
        "padding": True,
        "return_tensors": "pt",
    }
    if images is not None:
        call_kwargs["images"] = images
    if audio is not None:
        call_kwargs["audio"] = audio
    processed = processor(**call_kwargs)

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
    expected_grid = torch.tensor([[1, 24, 24]], dtype=grid.dtype)
    if not torch.equal(grid.cpu(), expected_grid):
        raise ValueError(
            f"the fixed 336x336 encoder requires image_grid_thw [1,24,24], "
            f"got {grid.tolist()}"
        )
    print("\n--- Vision stage ---")
    started = time.perf_counter()
    ue.vision_weight_init()
    ue.prepare_encoder_input(processed["pixel_values"], grid)
    ue._tensor_dram_addr = ue._tensor_dram_base
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
    ue._tensor_dram_addr = ue._tensor_dram_base
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
        "--multi-core",
        nargs="?",
        const=REQUIRED_ENGINES,
        default=REQUIRED_ENGINES,
        type=int,
        help="engine count; this model requires exactly 8",
    )
    parser.add_argument("--prompt", default=None, help="user text prompt")
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
    return parser


def run_summary_filename(args) -> str:
    """Per-run summary filename encoding the CLI config, e.g.

    ``--dev xdma0 --image --multi-core 8`` ->
    ``qwen2.5_omni_7b_test_xdma0_image_multi-core_8.md``.

    Device and mode are always present, in that order; a profile run is tagged
    so its phase breakdown never overwrites a generation run's summary.
    """
    parts = ["qwen2.5_omni_7b_test", args.dev, _result_mode(args)]
    if getattr(args, "profile", False):
        parts.append("profile")
    parts.append(f"multi-core_{args.multi_core}")
    return "_".join(parts) + ".md"


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.profile_ctx < 2 or args.profile_ctx > MAX_CONTEXT_SIZE:
        parser.error(f"--profile-ctx must be between 2 and {MAX_CONTEXT_SIZE}")
    if args.no_summary and args.summary:
        parser.error("--summary and --no-summary are mutually exclusive")
    with _exclusive_run_lock():
        _main_locked(parser, args)


def _main_locked(parser: argparse.ArgumentParser, args) -> None:
    """Run artifact preparation and FPGA execution under the global lock."""
    args.image = _resolve_sample(args.image, DEFAULT_IMAGE, "--image")
    args.audio = _resolve_sample(args.audio, DEFAULT_AUDIO, "--audio")
    engine_kwargs = resolve_engine_config(parser, args)

    cfg = Qwen25OmniUnifiedEngine.load_config(script_dir=SCRIPT_DIR)
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
        script_dir=SCRIPT_DIR, fpga_build=fpga_build, **engine_kwargs
    )
    ue.configure_runtime_artifacts(params_path, processor_dir)
    ue.tokenizer = processor.tokenizer
    ue.processor = processor
    ue._prompt_text = prompt
    print(ue.describe_dram_map())

    image_embeddings = None
    audio_embeddings = None
    audio_metadata = None
    if args.image:
        image_embeddings = _run_vision(ue, processed, profile=args.profile)
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
        "prefill_tokens": len(context),
        "decoded_tokens": generated,
        "prefill_speed_tok_s": (
            len(context) / prefill_wall if prefill_wall > 0 else None
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
            len(context) / (prefill_us * 1e-6) if prefill_us > 0 else None
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
        "prompt_tokens": len(tokens),
        "rope_delta": rope_delta,
    }
    print("TEST_RESULT: " + json.dumps(result, ensure_ascii=False))
    _write_summary()


if __name__ == "__main__":
    main()
