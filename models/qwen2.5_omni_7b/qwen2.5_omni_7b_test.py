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
import importlib.util
import json
import math
import os
import sys
import time
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
from multi_engine_shard import MultiEngineScheduler, PrivateArena
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
_position_mod = _load_sibling(
    "qwen2_5_omni_7b_positions", "qwen2.5_omni_7b_positions.py"
)
_weight_mod = _load_sibling(
    "qwen2_5_omni_7b_weights", "qwen2.5_omni_7b_weights.py"
)

Qwen25OmniLMMixin = _lm_mod.Qwen25OmniLMMixin
Qwen25OmniVisionMixin = _vision_mod.Qwen25OmniVisionMixin
Qwen25OmniAudioMixin = _audio_mod.Qwen25OmniAudioMixin
build_multimodal_positions = _position_mod.build_multimodal_positions


REQUIRED_ENGINES = 8
REQUIRED_DRAM_GIB = 8
MAX_CONTEXT_SIZE = 2048
PREFILL_INPUT_TOKEN_LIMIT = 384
# Eight row-sharded engines require one full 64-row hardware block apiece.
# Logical prompts stay capped at 384; the remaining rows are finite, masked
# execution padding and do not become visible context tokens.
PREFILL_MAX_SEQ_LEN = REQUIRED_ENGINES * 64

# All three U55 images share the non-CONV arithmetic ISA used by Omni. The E7
# image predates CHECK_CLEAR, so its runner uses host-separated one-shot flag
# rounds; it is intentionally supported without changing the FPGA image.
LEGACY_HOST_SEGMENTED_BUILD = 0xE7AC2CAF
SUPPORTED_FPGA_BUILDS = frozenset(
    {0xFE984D16, 0x0305D87D, LEGACY_HOST_SEGMENTED_BUILD}
)
ENGINE_BASE_STRIDE = 0x00010000
RESET_PROBE_ISA_ADDR = 0x1FA000000

DEFAULT_IMAGE = os.path.normpath(
    os.path.join(PROJECT_ROOT, "test_samples", "yosemite.jpg")
)
DEFAULT_AUDIO = os.path.normpath(
    os.path.join(PROJECT_ROOT, "test_samples", "apex.wav")
)


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
        reported_dram = user_dma_core.AVAILABLE_DRAM_SIZE_GB
        if reported_cores is not None and reported_cores < REQUIRED_ENGINES:
            raise ValueError(
                f"the U55 image must report at least {REQUIRED_ENGINES} engines; "
                f"HW_INFO reports {reported_cores}"
            )
        if reported_dram is not None and reported_dram != REQUIRED_DRAM_GIB:
            raise ValueError(
                f"the U55 image must report exactly {REQUIRED_DRAM_GIB} GiB DRAM; "
                f"HW_INFO reports {reported_dram} GiB"
            )

        self.multi_core = multi_core
        self.fpga_build = None if fpga_build is None else int(fpga_build)
        if self.fpga_build is not None and self.fpga_build not in SUPPORTED_FPGA_BUILDS:
            raise ValueError(
                f"unsupported Qwen2.5-Omni FPGA build 0x{self.fpga_build:08x}")
        self._multi_core_schedulers: dict[str, MultiEngineScheduler] = {}
        self._worker_isa_used: dict[int, dict[str, int]] = {}
        self._params_regions: dict[str, dict[str, Any]] | None = None

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
        # O overlay only after full prefill. The distinct IF8 embedding is loaded
        # after decode sharding. The measured private total is 501.27 MiB/core
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
        self.mc_arena = PrivateArena(
            REQUIRED_ENGINES,
            arena_base=0,
            arena_bytes=self.PARAMS_BASE,
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
    the newer ``0x0305d87d`` queue-CONFIG image, whose backward-compatible
    matmul/dequantize/argmax path is used here.  Validate every build stamp
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
    unsupported = sorted(set(versions).difference(SUPPORTED_FPGA_BUILDS))
    if unsupported:
        supported = ", ".join(
            f"0x{version:08x}" for version in sorted(SUPPORTED_FPGA_BUILDS)
        )
        got = ", ".join(f"core {core}=0x{version:08x}"
                        for core, version in enumerate(versions))
        raise RuntimeError(
            f"unsupported FPGA build(s): {got}; supported builds: {supported}"
        )
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
    if dram != REQUIRED_DRAM_GIB:
        parser.error(
            f"Qwen2.5-Omni requires the 8-GiB U55 map; HW_INFO reports {dram!r} GiB"
        )
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


def _prepare_processor_inputs(args, cfg: dict):
    """Use the official processor for placeholder expansion and feature masks."""
    from PIL import Image
    from transformers import AutoProcessor

    model_dir = os.path.join(SCRIPT_DIR, cfg["paths"]["hf_model_dir"])
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
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


def _run_vision(ue: Qwen25OmniUnifiedEngine, processed) -> torch.Tensor:
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
    ue.compile_vision_encoder()
    ue.check_master_isa()
    embeddings = ue.run_vision_encoder()
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
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    args.image = _resolve_sample(args.image, DEFAULT_IMAGE, "--image")
    args.audio = _resolve_sample(args.audio, DEFAULT_AUDIO, "--audio")
    engine_kwargs = resolve_engine_config(parser, args)

    cfg = Qwen25OmniUnifiedEngine.load_config(script_dir=SCRIPT_DIR)
    # Fetch/convert before constructing a device-owning engine.  The conversion
    # streams Thinker shards and skips Talker/token2wav-only checkpoint shards.
    params_path = _weight_mod.ensure_params_bin(SCRIPT_DIR)
    print(f"Thinker params: {params_path}")
    processor, processed, tokens, prompt, rendered = _prepare_processor_inputs(
        args, cfg
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
    ue.tokenizer = processor.tokenizer
    ue.processor = processor
    ue._prompt_text = prompt
    print(ue.describe_dram_map())

    image_embeddings = None
    audio_embeddings = None
    audio_metadata = None
    if args.image:
        image_embeddings = _run_vision(ue, processed)
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

    print("\n--- Thinker LM stage ---")
    started = time.perf_counter()
    ue.lm_weight_init()
    ue.lm_tensor_init()
    ue.compile_prefill(len(context))
    # Decoder setup installs the device-side embedding and copies every
    # reusable projection into private windows. BF16 O addresses are compiled
    # now, but their shared overlay is deliberately deferred until prefill.
    ue.compile_decoder()
    ue.check_master_isa()
    for line in ue.isa_usage_lines():
        print(line)
    ue.run_prefill(
        context,
        image_embeddings=image_embeddings,
        audio_embeddings=audio_embeddings,
        positions=positions[: len(context)],
    )
    ue.activate_decode_shared_weights()
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
    result = {
        "model": "qwen2.5_omni_7b",
        "mode": _result_mode(args),
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
        "prefill_gflops": getattr(ue, "_prefill_gflops", None),
        "decode_gflops": getattr(ue, "_decode_gflops", None),
        "vision_gflops": getattr(ue, "_vis_gflops", None),
        "audio_gflops": getattr(ue, "_audio_gflops", None),
        "prompt_tokens": len(tokens),
        "rope_delta": rope_delta,
    }
    print("TEST_RESULT: " + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
