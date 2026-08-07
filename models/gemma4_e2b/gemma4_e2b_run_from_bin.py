#!/usr/bin/env python3
"""
Gemma4 E2B — execute-only inference from the pre-compiled instruction bin.

Self-contained: does NOT import from gemma4_e2b_test.py and does NOT need
the original HF model directory (gemma-4-E2B-it/). Designed for demo
deploys where only this one Python file plus the bin artifacts ship.

Required files in `gemma4_e2b_bin/` (produced once by gemma4_e2b_test.py
on a beefier machine):

  params.bin         FPGA weights (2.4 GB)
  host_weights.bin + .json          host-side side-cache (4.4 GB)
  programs.bin + .json    unified instruction bin (1.1 GB)
  tokenizer/                        minimal tokenizer subset (~32 MB):
                                      tokenizer.json, tokenizer_config.json,
                                      chat_template.jinja, processor_config.json

Usage (run from repo root):
  python models/gemma4_e2b/gemma4_e2b_run_from_bin.py                       # LM, default prompt
  python models/gemma4_e2b/gemma4_e2b_run_from_bin.py --prompt "..."        # LM, custom prompt
  python models/gemma4_e2b/gemma4_e2b_run_from_bin.py --vision-enable       # VLM, default image
  python models/gemma4_e2b/gemma4_e2b_run_from_bin.py --image PATH ...      # VLM, custom image
  python models/gemma4_e2b/gemma4_e2b_run_from_bin.py --dev xdma0 --cycle 5.62
"""

import json
import math
import os
import sys

# Force HuggingFace into offline mode BEFORE any `transformers` /
# `huggingface_hub` import below. The user does NOT need to set these
# in their shell — the script always sets them itself, regardless of
# inherited environment. This makes run_from_bin safe on a Pi / demo
# host with no internet and no `gemma-4-E2B-it/` folder. Every
# from_pretrained call also passes local_files_only=True for
# belt-and-suspenders insurance.
os.environ["HF_HUB_OFFLINE"]      = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"

# This file's folder: gemma4_e2b_bin/, *.json, decoder_program.json live here. user_dma_core is two levels up (repo root).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

# -----------------------------------------------------------------------------
# Vision fixed-shape contract.
#
# Every input image is pre-resized to VISION_CANONICAL_SIZE before the HF
# Gemma4Processor runs. This guarantees the processor emits a fixed
# num_patches = VISION_FIXED_NUM_PATCHES, which makes the vision encoder bin
# safe to merge into the single instruction bin (see
# notes_gemma4_e2b.md → "Vision encoder fixed-shape contract").
#
# Empirically the processor already maps every tested input size to 2520
# patches, so the resize is defensive — it locks the contract in *our* code
# so we are not at the mercy of HF processor version changes. A runtime
# assert in set_prefill_seq_vlm fails loudly if the count ever drifts.
# -----------------------------------------------------------------------------
VISION_CANONICAL_SIZE = (896, 896)   # (width, height) for PIL.Image.resize
VISION_FIXED_NUM_PATCHES = 2520

# -----------------------------------------------------------------------------
# Audio fixed-shape contract (mirrors gemma4_e2b_test.py).
# HF Gemma3n's audio processor pads/truncates every audio file to a fixed
# input_features shape so the chat template can hard-code 128 <|audio|>
# placeholder tokens. T_raw is whatever HF produces; the runtime asserts the
# observed shape matches the value baked into the unified bin's manifest.
# -----------------------------------------------------------------------------
AUDIO_FIXED_SOFT_TOKENS = 128

# Default sample assets used to derive canonical-shape inputs.
_TEST_SAMPLES = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "test_samples"))
DEFAULT_IMAGE = os.path.join(_TEST_SAMPLES, "yosemite.jpg")
DEFAULT_AUDIO = os.path.join(_TEST_SAMPLES, "apex.wav")

# Bumped any time the on-disk programs.bin layout / ISA semantics
# change in an incompatible way. On version mismatch the bin is rebuilt from
# scratch rather than incrementally extended.
# v2-dynvis: vision attention uses andromeda's flash_attention_core_dynamic
#            (dedicated VIS_ATTN_P scores buffer). Tensor layout unchanged, so
#            replay only needs the matching version tag.
# v3-uac:    vision attention uses the library unified_attention_core; scores +
#            scaled_q fold into VIS_FLASH_SCRATCH (resized to aligned_S² +
#            2·HD·aligned_S). Tensor layout CHANGED — this allocation must mirror
#            test.py exactly or the replayed bin's baked addresses drift.
INSTRUCTION_BIN_COMPILE_VERSION = "v3-uac"

# We run on FPGA + CPU only; disable CUDA before importing torch so PyTorch
# doesn't probe the GPU driver (avoids a noisy "Error 804: forward compatibility"
# warning on hosts whose CUDA driver/runtime doesn't match the installed GPU).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")

# Embedding table is read zero-copy from the read-only mmap'd weight bin via
# torch.frombuffer (avoids a 770 MB host allocation). We never write to it,
# so the "buffer is not writable" UserWarning is harmless — silence it.
import warnings as _warnings
_warnings.filterwarnings(
    "ignore",
    message="The given buffer is not writable",
    category=UserWarning,
)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
# Deliberately NOT importing AutoModelForImageTextToText / huggingface_hub:
# run_from_bin loads zero model weights from the HF hub, and dragging those
# in adds a cold-start hub probe that fails on offline / no-HF-folder hosts.
import time
# pcie_utils imports (run from andromeda/pcie_utils or with PYTHONPATH)
import user_dma_core
from user_dma_core import DMA_DEVICE_H2C, DRAM_INSTRUCTION_ADDR, TYPE, UE_FMAX_CONTEXT_SIZE, UE_VECTOR_SIZE, UE_ARGMAX_INDEX, URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS, URAM_NEAR_FULL_SIZE, URAM_START_ADDR, URAM_SECTION, set_dma_device
from user_dma_core import UnifiedEngine
from user_dma_core import ue_35bit_addr_shifter

# Audio encoder primitives — imported lazily because (a) audio mode is not
# yet folded into the unified instruction bin, so it is refused at the
# top of main() with a clear error, and (b) a deploy host that never runs
# audio shouldn't need to ship audio_primitives.py.
try:
    from audio_primitives import (
        silu_core_dram as _aud_silu,
        glu_core_dram as _aud_glu,
        half_step_residual_core_dram as _aud_half_step,
        build_toeplitz_for_depthwise as _aud_build_toeplitz,
        depthwise_conv1d_core_dram as _aud_depthwise_conv1d,
        copy_dram_to_dram_chunked as _aud_copy_chunked,
        eltwise_add_core_dram as _aud_eltwise_add,
    )
except ImportError:
    _aud_silu = _aud_glu = _aud_half_step = _aud_build_toeplitz = None
    _aud_depthwise_conv1d = _aud_copy_chunked = _aud_eltwise_add = None

# --- BROAD PRINT SUPPRESSION FOR LIBRARIES ---
import builtins

_original_print = builtins.print
_SILENT_MODE = False

def quiet_print(*args, **kwargs):
    """Suppress prints when _SILENT_MODE is True; otherwise print normally."""
    if _SILENT_MODE:
        return
    _original_print(*args, **kwargs)

builtins.print = quiet_print
# ---------------------------------------------

def _parse_offset(val) -> int:
    """Parse offset/size from JSON: int or hex string like '0x24000000'."""
    if isinstance(val, str):
        return int(val, 0)
    return int(val)

# Canonical codec (single source of truth for INT4 / FP4 / IF4 packing) lives
# in template/quant_lib.py. Used only for VLM / audio at run time
# (vision_weight_init quantizes vision weights from the HF model). LM-only
# deploys never hit this code path, so the import is lazy — that lets the
# deploy host omit quant_lib.py entirely for text-only inference.
try:
    from quant_lib import quantize as _qs_quantize
except ImportError:
    _qs_quantize = None

# Parallel quantization. The IF4 codec is pure-CPU and stateless, so we run
# many tensor quants concurrently via a thread pool — torch ops release the
# GIL and ThreadPool avoids the multi-GB pickle cost of a process pool.
from concurrent.futures import ThreadPoolExecutor as _QuantPool
_QUANT_WORKERS = max(1, (os.cpu_count() or 4) - 1)


import json as _json


class _ProgramCache:
    """Per-segment FPGA program cache. Maps a stable string key (e.g.
    ``"vis_L3_q_proj"``) to the captured ISA instruction bytes for that
    segment.  On cache hit, callers skip the Python ISA-emission compile and
    DMA the cached bytes directly to program DRAM.

    Persisted as one bin file (concatenated bytes) plus one json sidecar
    holding ``{key: (offset, size)}``. The cache is keyed at the file level
    by something the producer chooses (e.g. num_patches for vision); changing
    that file path forces a fresh build.

    Cache invariants the caller MUST guarantee:
      - Tensor DRAM addresses referenced inside the captured bytes (weight
        addresses, scratch buffers) are deterministic across runs. If you
        change weight upload order or buffer layout, delete the cache file.
      - The compile path under the same key produces the same bytes given
        the same inputs (it does today — the per-layer compile is pure
        function of layer_idx + num_patches + weight DRAM addresses).
    """

    def __init__(self, bin_path: str, meta_path: str):
        self.bin_path = bin_path
        self.meta_path = meta_path
        self._key_to_bytes: dict[str, bytes] = {}
        self._dirty: bool = False

    def load(self) -> int:
        if not (os.path.exists(self.bin_path) and os.path.exists(self.meta_path)):
            return 0
        with open(self.meta_path, "r") as f:
            meta = _json.load(f)
        with open(self.bin_path, "rb") as f:
            blob = f.read()
        self._key_to_bytes = {
            e["key"]: blob[e["offset"]: e["offset"] + e["size"]]
            for e in meta["entries"]
        }
        return len(self._key_to_bytes)

    def save(self) -> None:
        if not self._dirty:
            return
        os.makedirs(os.path.dirname(self.bin_path), exist_ok=True)
        blob = bytearray()
        entries = []
        for key, bts in self._key_to_bytes.items():
            entries.append({"key": key, "offset": len(blob), "size": len(bts)})
            blob.extend(bts)
        with open(self.bin_path, "wb") as f:
            f.write(blob)
        with open(self.meta_path, "w") as f:
            _json.dump({"entries": entries}, f, indent=2)
        self._dirty = False

    def get(self, key: str) -> bytes | None:
        return self._key_to_bytes.get(key)

    def put(self, key: str, bts: bytes) -> None:
        self._key_to_bytes[key] = bts
        self._dirty = True

    def __len__(self) -> int:
        return len(self._key_to_bytes)


def _parallel_quantize(precision: str, tensors: list[torch.Tensor],
                        block_size: int = 64) -> list[tuple[bytes, bytes]]:
    """Quantize a list of [N, K] bf16 tensors in parallel via the canonical
    codec wrapper. Returns a parallel list of (data_bytes, scale_bytes).
    Use this whenever you have many same-codec tensors to pack at once
    (LM weight bin, vision/audio encoder upload). Order is preserved."""
    if len(tensors) == 0:
        return []
    if len(tensors) == 1:
        return [_qs_quantize(precision, tensors[0], block_size=block_size)]
    with _QuantPool(max_workers=_QUANT_WORKERS) as ex:
        return list(ex.map(
            lambda t: _qs_quantize(precision, t, block_size=block_size),
            tensors,
        ))

LM_QUANT_PRECISION = "if4"      # LM matmuls (Q/K/V/O proj, MLP up/gate/down, lm_head)
VISION_QUANT_PRECISION = "if4"  # vision encoder matmuls (Q/K/V/O proj, MLP gate/up/down)
AUDIO_QUANT_PRECISION = "if4"   # audio encoder matmuls (Conformer attn + ffw + lconv1d projections).
                                # No SW eval data for audio yet — we picked IF4 because the wire
                                # format is byte-identical to FP4, so falling back to "fp4" is a
                                # one-line change with no FPGA work needed.

def _host_rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS norm on host (for per-layer projection norm)."""
    x_f = x.float()
    rms = (x_f ** 2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return ((x_f / rms) * gamma.float()).to(x.dtype)


def _ensure_hf_model(script_dir: str, cfg: dict):
    """run_from_bin never loads the HF model. All weights live in
    ``params.bin`` (LM + vision_section + audio_section
    + host_section). This stub is kept only because legacy reachable
    paths may still reference it; calling it is a programming error.
    """
    raise RuntimeError(
        "run_from_bin: HF model load is forbidden. All weights — LM, "
        "vision, audio — live in params.bin sections. If "
        "this error fires, the bin was produced before the audio_section "
        "was added; regenerate it on the build host with "
        "`python gemma4_e2b_test.py` and copy the new weight bin here.")


# -----------------------------------------------------------------------------
# Gemma4 E2B unified engine
# -----------------------------------------------------------------------------
class Gemma4_UnifiedEngine(UnifiedEngine):
    """UnifiedEngine with Gemma4 E2B dims: loads config + weight bin, compile_prefill/compile_decoder, run_prefill/run_decoder. Numeric checks in gemma4_e2b_numeric.py."""

    def __init__(self, script_dir: str | None = None, local_weights: bool = False, dual_engine: bool = False, engine_slave: bool = False):
        engine_base = user_dma_core.UE_0_BASE_ADDR + 0x00010000 if engine_slave else user_dma_core.UE_0_BASE_ADDR
        # Gemma4 FIXED DRAM layout (FULL 4 GB; see notes/notes_gemma4_e2b_vision.md
        # "Master layout table"). All addresses below 0x100000000 (DMA-mapped DRAM
        # is 0x00000000 – 0xFFFFFFFF — same as qwen2.5; old DRAM_START_ADDR=0x80000000
        # only used the upper 2 GB and wasted the rest).
        #   Weight LM     : 0x00000000 – 0x64000000  (1600 MB)
        #   Weight Vision : 0x64000000 – 0x6c000000  (128 MB)
        #   Weight Audio  : 0x6c000000 – 0x78000000  (192 MB)
        #   Act. Scratch  : 0x78000000 – 0x88000000  (256 MB) ← tensor_base default
        #   Act. KV cache : 0x88000000 – 0x98000000  (256 MB; tail of activation region)
        #   ISA Audio     : 0x98000000 – 0xa0000000  (128 MB)
        #   ISA Unified   : 0xa0000000 – 0x100000000 (1.5 GB) ← program_base default
        #     (formerly split into vision/prefill/decoder regions; collapsed
        #     into one contiguous region for the unified programs.bin
        #     which holds LM prefill+decode buckets + vision encoder ISA in
        #     one contiguous blob. Total fits comfortably under 4 GB; the prior
        #     base 0xC0000000 caused the bin to overflow 4 GB once vision
        #     (~385 MB) was appended to LM (~661 MB).)
        _params_base  = 0x00000000   # Weight region start
        _tensor_base  = 0x78000000   # Activation region start (stage scratch)
        _program_base = 0xa0000000   # unified bin base, gives 1.5 GB headroom
        if engine_slave:
            _program_base += 0x10000000
        super().__init__(BASE_ADDR=engine_base,
                          params_dram_base=_params_base,
                          program_dram_base=_program_base,
                          tensor_dram_base=_tensor_base)
        self.dual_engine = dual_engine
        self.script_dir = script_dir or SCRIPT_DIR
        self._cfg = self.load_config(script_dir=self.script_dir)
        self.weight_defs = self._cfg["_weight_defs"]

        fi = self._cfg["file_info"]
        model = self._cfg["model"]
        paths = self._cfg["paths"]

        self.vector_length = fi["hidden_size"]
        self.head_dim = fi["head_dim"]  # 512 (max, for uniform sizing)
        self.head_dim_sliding = fi["head_dim_sliding"]  # 256
        self.per_layer_input_dim = fi["per_layer_input_dim"]  # 256
        self.bytes_per_element = fi["bytes_per_element"]
        self.group_size = fi["group_size"]
        self.mlp_elements = fi["mlp_elements"]
        self.mlp_elements_wide = fi.get("mlp_elements_wide", fi["mlp_elements"])
        self.q_size = self.head_dim * self.group_size * self.bytes_per_element
        self.k_size = self.head_dim * self.bytes_per_element
        self.MAX_CONTEXT_SIZE = model["max_context_size"]
        self.LAYER_SIZE = fi["num_layers"]
        self.EMBEDDING_ELEMENTS = fi["embedding_vocab"]
        self._isa_reg_counter = 1
        fixed = self._cfg.get("fixed_isa_regs", {})
        # Dynamic-PBI register binding (matches gemma4_e2b_test.py).
        self.TMP_REG       = fixed["TMP_REG"]
        self.gpr_seq_len    = fixed["GPR_SEQ_LEN_REG"]
        self.gpr_q_seq_len  = fixed["GPR_Q_SEQ_LEN_REG"]
        self.gpr_bucket_idx = fixed["GPR_BUCKET_IDX_REG"]
        # Legacy register-name aliases (same indices). Legacy emission paths
        # still use these names until they're ported; see notes_gemma4_e2b.md.
        self._isa_reg_counter = 5
        self._isa_reg_base = 5  # one-shot mode resets the allocator to this base per sub-op
        self.causal_mask_upper = False
        self._rope_global_layers = set(model["rope_global_layers"])
        self._full_attention_layers = set(model["full_attention_layers"])
        self._double_wide_mlp_first = model.get("double_wide_mlp_first_layer", fi["num_layers"])
        self._end_of_turn_token_id = model["end_of_turn_token_id"]
        # Sliding-attention window. Sliding layers (i.e. layers NOT in
        # full_attention_layers) are limited to attending the last
        # `sliding_window` tokens. Default to MAX_CONTEXT_SIZE so older configs
        # without this key keep their old behaviour (no real windowing).
        self.sliding_window = model.get("sliding_window", model["max_context_size"])
        # max_prefill_seq_len caps the flash attention scratch/bias buffer sizes
        # independently of max_context_size. Flash buffers are only used at full
        # size during prefill (for the whole prompt); decode uses 1 row. Sizing
        # them for max_context_size wastes ~200+ MB of DRAM that the decoder
        # program bin later tries to occupy, causing overlap.
        self.max_prefill_seq_len = model.get("max_prefill_seq_len", model["max_context_size"])
        # KV sharing map: built from HF model during weight_init
        self._kv_shared_map = {}  # layer_idx -> reference_layer_idx (populated in weight_init)
        self._gamma_bin_offset = self._cfg["special"]["rms_norm"]["gamma_offset"]
        self._per_layer_model_proj_scale = model["per_layer_model_proj_scale"]
        self._per_layer_input_scale = model["per_layer_input_scale"]
        self.prefill_seq = None
        self.engine_slave = engine_slave
        # Identity matrix cache (fix for flash_attention_core DRAM leak)
        self._identity_dram_addr = None
        self._identity_dram_written = False

        self._weights_bin_rel = "gemma4_e2b_bin/params.bin" if local_weights else paths["weights_bin"]
        self.weight_init()
        self.tensor_init()
        self._preallocate_identity_matrix()

    # ─── Identity matrix cache (fix for flash_attention_core DRAM leak) ─────
    _IDENTITY_MAT_BYTES = UE_VECTOR_SIZE * UE_VECTOR_SIZE * 2  # 8192

    def _preallocate_identity_matrix(self) -> None:
        if self._identity_dram_addr is not None:
            return
        self._identity_dram_addr = self.allocate_params_dram(self._IDENTITY_MAT_BYTES)
        eye = torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16)
        super().dma_write(DMA_DEVICE_H2C, self._identity_dram_addr, eye, self._IDENTITY_MAT_BYTES)
        self._identity_dram_written = True

    def _flash_attention_core_cached(self, **kwargs) -> int:
        saved = self._next_params_dram_addr
        self._next_params_dram_addr = self._identity_dram_addr
        result = self.flash_attention_core(**kwargs)
        self._next_params_dram_addr = saved
        return result

    def dma_write(self, device, addr, data, size):
        if (getattr(self, '_identity_dram_written', False)
                and self._identity_dram_addr is not None
                and addr == self._identity_dram_addr
                and size == self._IDENTITY_MAT_BYTES):
            return size  # redundant identity DMA already on-card; report success
        return super().dma_write(device, addr, data, size)

    def _emit_sram_eltwise_chunked(self, kind: str,
                                    addr_A: int, addr_B: int, addr_out: int,
                                    num_elements: int,
                                    chunk: int = 131072) -> None:
        """Emit chunked eltwise_add/eltwise_mul instructions into the current
        capture buffer.

        URAM A (0x10000..0x90000) and URAM B (0x90000..0x110000) each hold
        0x80000 bytes = 262144 bf16 elements. A single unchunked eltwise_*
        therefore crashes into the other buffer once num_elements exceeds
        262144 (for Gemma4 VLM prefill this happens at ~43 image tokens with
        cur_mlp=6144, ~22 tokens with cur_mlp=12288, and ~170 tokens with
        vector_length=1536). Chunking to 131072 elements per call is a safe,
        uniform limit that leaves room for both operands in the two buffers.

        Intended for use INSIDE compile_prefill / compile_decoder, which
        capture all instructions in one program (unlike the compare script's
        _run_eltwise_*_chunked which compiles a fresh program per chunk).
        """
        bpe = self.bytes_per_element
        for off in range(0, num_elements, chunk):
            n = min(chunk, num_elements - off)
            self.accelerator_memory_to_sram(
                accelerator_dram_address=addr_A + off * bpe,
                sram_address=0x10000, element_size=n)
            self.accelerator_memory_to_sram(
                accelerator_dram_address=addr_B + off * bpe,
                sram_address=0x90000, element_size=n)
            if kind == "add":
                self.eltwise_add_core(
                    vector_A_sram_start_addr=0x10000,
                    vector_B_sram_start_addr=0x90000,
                    vector_C_sram_wb_addr=0x10000,
                    element_size=n)
            elif kind == "mul":
                self.eltwise_mul_core(
                    vector_A_sram_start_addr=0x10000,
                    vector_B_sram_start_addr=0x90000,
                    vector_C_sram_wb_addr=0x10000,
                    element_size=n)
            else:
                raise ValueError(f"unknown eltwise kind: {kind!r}")
            self.sram_to_accelerator_memory(
                sram_address=0x10000,
                accelerator_dram_address=addr_out + off * bpe,
                element_size=n)

    def _emit_sram_copy_chunked(self, src_addr: int, dst_addr: int,
                                 num_elements: int, chunk: int = 131072) -> None:
        """Emit a chunked DRAM→SRAM→DRAM copy into the capture buffer. Used to
        shuttle large activations when a direct DRAM-to-DRAM path isn't handy."""
        bpe = self.bytes_per_element
        for off in range(0, num_elements, chunk):
            n = min(chunk, num_elements - off)
            self.accelerator_memory_to_sram(
                accelerator_dram_address=src_addr + off * bpe,
                sram_address=0x10000, element_size=n)
            self.sram_to_accelerator_memory(
                sram_address=0x10000,
                accelerator_dram_address=dst_addr + off * bpe,
                element_size=n)

    def _emit_sram_broadcast_mul_chunked(self, src_addr: int, dst_addr: int,
                                          num_elements: int, scalar: float,
                                          chunk: int = 131072) -> None:
        """Emit a chunked in-place broadcast_mul (scalar multiply) through
        SRAM into the capture buffer."""
        bpe = self.bytes_per_element
        for off in range(0, num_elements, chunk):
            n = min(chunk, num_elements - off)
            self.accelerator_memory_to_sram(
                accelerator_dram_address=src_addr + off * bpe,
                sram_address=0x10000, element_size=n)
            self.broadcast_mul(scalar=scalar,
                               sram_start_addr=0x10000,
                               sram_wb_addr=0x10000,
                               element_size=n)
            self.sram_to_accelerator_memory(
                sram_address=0x10000,
                accelerator_dram_address=dst_addr + off * bpe,
                element_size=n)

    @staticmethod
    def load_config(config_path: str | None = None, script_dir: str | None = None) -> dict:
        """Load gemma4_e2b_config.json and build weight_defs (offset/size dict) from regions."""
        if config_path is None:
            script_dir = script_dir or SCRIPT_DIR
            config_path = os.path.join(script_dir, "gemma4_e2b_config.json")
        with open(config_path, "r") as f:
            cfg = json.load(f)
        weight_defs = {"LAYER_WEIGHT_SIZE": cfg["file_info"]["layer_size"]}
        for key, r in cfg.get("regions", {}).items():
            weight_defs[key] = _parse_offset(r["offset"])
            weight_defs[f"{key}_SIZE"] = r["size"]
        for key, r in cfg.get("non_layer_regions", {}).items():
            weight_defs[key] = _parse_offset(r["offset"])
            weight_defs[f"{key}_SIZE"] = r["size"]
        cfg["_weight_defs"] = weight_defs
        return cfg

    def set_prefill_seq(self, prompt: str | None = None) -> None:
        """Set self.prefill_seq from a text prompt (tokenize with chat template) or from config default."""
        if prompt is not None:
            conversation = [{"role": "user", "content": prompt}]
            prompt_with_template = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            self.prefill_seq = tuple(self.tokenizer.encode(prompt_with_template, add_special_tokens=True))
            print(f"Prefill from prompt ({len(self.prefill_seq)} tokens): {prompt!r}")
        else:
            self.prefill_seq = tuple(self._cfg["default_prefill_tokens"])
            decoded = self.tokenizer.decode(list(self.prefill_seq), skip_special_tokens=True)
            print(f"Prefill from default ({len(self.prefill_seq)} tokens): {decoded!r}")

    def _run_vision_encoder_fpga(self, image_path: str, prompt: str) -> tuple[torch.Tensor, list[int], list[int]]:
        """Run the full vision encoder on FPGA: patch embedder → 16 encoder
        layers → pooler → embed_vision projection. Returns
        (image_features, token_ids, mm_types).

        DRAM-layout note: vision reuses the LM tensor DRAM region via
        vision_tensor_init (same as the compare script). This clobbers
        IDENTITY_DRAM_ADDR, the KV cache zero-pad, and the LM flash scratch
        buffers that the LM decoder relies on — so we restore IDENTITY_DRAM
        and re-run the LM tensor zero-pads at the end. Prefill re-uploads
        its own input/bias tensors in run_prefill so they don't need to be
        preserved across this call.
        """
        from PIL import Image
        from transformers import AutoProcessor

        # run_from_bin VLM path: no HF model load. The vision encoder
        # weights are read from the pre-quantized side-cache
        # (vision_weights.bin) by vision_weight_init; the image / chat
        # processor only needs tokenizer + processor_config.json which
        # live in the bundled tokenizer subset.
        tok_subset = os.path.join(self.script_dir, "gemma4_e2b_bin", "tokenizer")
        if os.path.exists(os.path.join(tok_subset, "processor_config.json")):
            proc_dir = tok_subset
        else:
            proc_dir = os.path.join(self.script_dir, self._cfg["paths"]["hf_model_dir"])
            if not os.path.exists(os.path.join(proc_dir, "processor_config.json")):
                raise RuntimeError(
                    f"VLM mode needs processor_config.json in either:\n"
                    f"  {tok_subset}/  (preferred — bundled tokenizer subset)\n"
                    f"  {proc_dir}/    (legacy — full HF model directory)\n"
                    "Re-generate by running gemma4_e2b_test.py on a beefier host.")
        processor = AutoProcessor.from_pretrained(proc_dir, local_files_only=True)
        hf_model = None   # vision_weight_init / _load_or_build_vision_rope_pads ignore this in cache mode

        # Tokenize + patch extraction on host via the processor.
        #
        # FIXED-SHAPE PREPROCESSING: We resize every input image to
        # VISION_CANONICAL_SIZE before feeding the processor. The Gemma4 image
        # processor already maps any input dimension to num_patches=2520
        # internally (verified empirically: 224², 448², 672², 768², 896²,
        # 1024², 640×480, 1920×1080 all → 2520 patches). The explicit resize
        # here is intentional anyway:
        #   1. Locks the contract in the user's code, not just the processor's
        #      behavior — protects against future HF processor changes.
        #   2. Makes num_patches a true compile-time constant so the vision
        #      encoder bin is shape-fixed and safe to merge into the single
        #      instruction bin (see design note: notes_gemma4_e2b.md
        #      "Vision encoder fixed-shape contract").
        #   3. Reduces processor work on huge inputs (e.g. 4K photos).
        # The runtime assert below fails loudly if a future processor version
        # ever produces a different patch count.
        image = Image.open(image_path).convert("RGB")
        image = image.resize(VISION_CANONICAL_SIZE, Image.BICUBIC)
        conversation = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text_prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text_prompt], images=[[image]], return_tensors="pt")

        pixel_values = inputs['pixel_values']                  # [1, S_patches, 768]
        pixel_position_ids = inputs['image_position_ids']       # [1, S_patches, 2]
        padding_positions = (pixel_position_ids == -1).all(dim=-1)  # [1, S_patches]
        num_patches = pixel_values.shape[1]
        assert num_patches == VISION_FIXED_NUM_PATCHES, (
            f"Vision shape contract broken: got num_patches={num_patches}, "
            f"expected {VISION_FIXED_NUM_PATCHES}. The cached vision bin "
            f"(and any combined instruction bin) was compiled for the fixed "
            f"patch count; a mismatch will produce wrong output. Check that "
            f"the HF processor still maps {VISION_CANONICAL_SIZE} → "
            f"{VISION_FIXED_NUM_PATCHES} patches.")

        # Save LM allocator state so we can restore it after vision finishes.
        _tensor_dram_save = self._tensor_dram_addr
        _prog_dram_addr_save = self._next_program_dram_addr
        _prog_dram_base_save = self._program_dram_base

        # ---- Unified-bin vision path ----
        # Vision encoder ISA lives inside programs.bin at address
        # `lm_base + lm_size`. If the unified bin doesn't exist yet (first run
        # ever, or LM-only-first scenario), build the LM core now; then
        # extend with vision. This keeps the single-bin contract intact across
        # any LM/VLM run order.
        bin_dir    = os.path.join(self.script_dir, "gemma4_e2b_bin")
        instr_bin  = os.path.join(bin_dir, "programs.bin")
        instr_meta = os.path.join(bin_dir, "programs.json")
        if not (os.path.exists(instr_bin) and os.path.exists(instr_meta)):
            raise RuntimeError(
                f"VLM mode requires a pre-compiled unified instruction bin. Missing:\n"
                f"  {instr_bin}\n"
                f"  {instr_meta}\n"
                f"Generate them on a beefier host with `python gemma4_e2b_test.py --vision-enable` "
                f"and copy gemma4_e2b_bin/ to this host.")
        with open(instr_meta, "r") as f:
            _peek = json.load(f)
        if _peek.get("compile_version") != INSTRUCTION_BIN_COMPILE_VERSION:
            raise RuntimeError(
                f"Instruction bin compile_version mismatch (disk: {_peek.get('compile_version')!r}, "
                f"expected {INSTRUCTION_BIN_COMPILE_VERSION!r}). "
                f"Delete {instr_bin} and rerun.")
        if not _peek.get("contains_vision"):
            raise RuntimeError(
                "VLM mode requires the instruction bin to already contain the vision encoder.\n"
                f"  contains_vision = False in {instr_meta}\n"
                "Re-generate with `python gemma4_e2b_test.py --vision-enable` on a host that has "
                "the HF model, then copy gemma4_e2b_bin/ here.")
        _lm_base = int(_peek["instruction_base_addr"], 16)
        _lm_size = _peek["instruction_total_size"]
        # If the bin already contains vision, vision_target_addr is the
        # already-baked vision start; otherwise the next free byte after LM.
        vision_target_addr = (int(_peek["vision_program_start_addr"], 16)
                              if _peek.get("contains_vision")
                              else _lm_base + _lm_size)

        # Vision setup. Pass program_base so PBI baking in subsequent compile
        # (if we need to extend) targets the right unified-bin address.
        # Silence library tile-planner prints across the setup block
        # (belt-and-suspenders: _SILENT_MODE + builtins.print swap).
        global _SILENT_MODE
        _SILENT_MODE = True
        _orig_builtin_print = builtins.print
        builtins.print = lambda *a, **kw: None
        try:
            self.vision_weight_init(hf_model)
            self.vision_tensor_init(num_patches, program_base=vision_target_addr)
            self.set_vision_attention_bias(padding_positions)
        finally:
            _SILENT_MODE = False
            builtins.print = _orig_builtin_print

        # 2D-RoPE tables (cos / neg_sin / sin_hi tiled+padded) for FPGA RoPE.
        # If the unified bin already contains rope pads (subsequent VLM run),
        # read the bytes directly from the bin file at the manifest offsets.
        # Else compute fresh on host and pass them into the extend call so they
        # land in the same unified bin alongside the vision ISA.
        cos_pad_tiled = neg_sin_pad_tiled = sin_hi_pad_tiled = None
        if (_peek.get("contains_vision")
                and all(k in _peek for k in (
                    "vision_rope_cos_offset", "vision_rope_neg_sin_offset",
                    "vision_rope_sin_hi_offset"))):
            HD = self.VIS_HEAD_DIM
            rope_w = HD // 2       # RoPE opt B: 32-wide rows (no upper-32 zero pad)
            n_rows = num_patches   # RoPE opt A: one row per patch (not per patch,head)
            bytes_per_buffer = n_rows * rope_w * self.bytes_per_element
            print(f"  [Vision] [host] reading rope pads from unified bin "
                  f"(offsets cos={_peek['vision_rope_cos_offset']}, "
                  f"neg_sin={_peek['vision_rope_neg_sin_offset']}, "
                  f"sin_hi={_peek['vision_rope_sin_hi_offset']})", flush=True)
            with open(instr_bin, "rb") as f:
                def _slice_pad(off, sz):
                    f.seek(off)
                    buf = f.read(sz)
                    if len(buf) != sz:
                        raise RuntimeError(f"truncated rope pad read at offset {off}")
                    return torch.frombuffer(bytearray(buf), dtype=torch.bfloat16).reshape(n_rows, rope_w).clone()
                cos_pad_tiled     = _slice_pad(_peek["vision_rope_cos_offset"],     _peek["vision_rope_cos_size"])
                neg_sin_pad_tiled = _slice_pad(_peek["vision_rope_neg_sin_offset"], _peek["vision_rope_neg_sin_size"])
                sin_hi_pad_tiled  = _slice_pad(_peek["vision_rope_sin_hi_offset"],  _peek["vision_rope_sin_hi_size"])
        else:
            cos_pad_tiled, neg_sin_pad_tiled, sin_hi_pad_tiled = self._load_or_build_vision_rope_pads(
                hf_model, pixel_position_ids, num_patches)
        self.dma_to_accelerator_memory(self.VIS_ROPE_COS_PAD_TILED, cos_pad_tiled)
        self.dma_to_accelerator_memory(self.VIS_ROPE_NEG_SIN_PAD_TILED, neg_sin_pad_tiled)
        self.dma_to_accelerator_memory(self.VIS_ROPE_SIN_HI_PAD_TILED, sin_hi_pad_tiled)

        # No extend path in run_from_bin: bin must already contain vision
        # (asserted at top of this method). The asserts above already
        # guaranteed it; the original test.py path would call
        # extend_instruction_bin_with_vision here.

        # Load (or reload) the unified bin to FPGA program DRAM.
        # Both LM and vision sections now sit at their baked absolute
        # addresses; PBI JUMP_ABS targets resolve correctly.
        manifest = self.load_instruction_bin()
        vision_program_addr = manifest["_vision_addr_int"]

        # Execute vision pipeline: patch_embed (FPGA, compile-and-run) → encoder
        # one-shot (start at vision_program_addr from manifest).
        self._vis_fpga_t0 = time.perf_counter()
        self.vision_patch_embed(pixel_values.cpu(), pixel_position_ids.cpu(),
                                 padding_positions.cpu())
        # FPGA dispatch quirk: start_execute_from_dram is reliable only when
        # preceded by a DMA write that touches the program region. After
        # patch_embed runs many small compile-and-run programs, the FPGA's
        # instruction prefetch / queue state appears stale; without a fresh
        # DMA write the next start_execute halts almost immediately (~0.01s
        # wait_queue). The legacy execute_vision_encoder_bin path mirrors
        # this implicitly because it DMAs the vision bytes right before
        # executing them. We re-DMA only the vision section (already loaded
        # by load_instruction_bin, but the kick is what matters).
        vision_size = manifest["vision_program_size"]
        vision_offset_in_bin = vision_program_addr - manifest["_base_addr_int"]
        bin_path_disk = os.path.join(self.script_dir, manifest["instruction_bin"])
        with open(bin_path_disk, "rb") as f:
            f.seek(vision_offset_in_bin)
            vision_bytes_disk = f.read(vision_size)
        _t_dma = time.perf_counter()
        self.dma_write(DMA_DEVICE_H2C, vision_program_addr, vision_bytes_disk, vision_size)
        print(f"  [Vision] re-DMA vision section ({vision_size/1024/1024:.1f} MB in {time.perf_counter()-_t_dma:.2f}s) — kicks FPGA before execute", flush=True)
        print(f"  [Vision] launching encoder one-shot at 0x{vision_program_addr:X} ...", flush=True)
        _exec_t0 = time.perf_counter()
        # Heartbeat thread mirrors execute_vision_encoder_bin's UX.
        import threading
        _hb_stop = threading.Event()
        def _hb_run(_anchor=self._vis_fpga_t0):
            while not _hb_stop.wait(10):
                _original_print(f"  [Vision] ... running on FPGA ({time.perf_counter()-_anchor:.0f}s)", flush=True)
        _hb_th = threading.Thread(target=_hb_run, daemon=True)
        _hb_th.start()
        try:
            self.start_execute_from_dram(vision_program_addr)
            self.wait_queue(180.0)
        finally:
            _hb_stop.set()
            _hb_th.join(timeout=1.0)
        print(f"  [Vision] encoder one-shot done in {time.perf_counter()-_exec_t0:.2f}s", flush=True)
        # Layer 0 reads VIS_IO_A and writes VIS_IO_B; layer 1 reads B and
        # writes A; ... so for VIS_LAYERS=16 (even), final output is in IO_A.
        final_buf = self.VIS_IO_A if self.VIS_LAYERS % 2 == 0 else self.VIS_IO_B
        encoder_out = self.dma_from_accelerator_memory(
            final_buf, (num_patches, self.VIS_H)).cpu()

        # Pooler + embed_vision tail on FPGA → [N_soft, 1536]
        image_features = self.vision_embed_project(
            encoder_out, pixel_position_ids.cpu(), padding_positions.cpu())
        image_features = image_features.to(torch.bfloat16)

        # ---- Restore LM state so decoder can run ----
        # Restore allocator pointers. Vision tensors clobbered the LM tensor
        # DRAM region (KV cache + flash buffers + IDENTITY), so we re-DMA the
        # pieces the LM decoder reads without re-writing first: IDENTITY (used
        # by decoder_attention_core) and the KV cache (positions beyond the
        # current seq_len are masked by the bias, but must contain safe
        # values — not NaN/Inf from vision intermediates — or bf16 Q·K will
        # overflow to NaN and destroy the softmax).
        self._tensor_dram_addr = _tensor_dram_save
        self._next_program_dram_addr = _prog_dram_addr_save
        self._program_dram_base = _prog_dram_base_save
        # Re-zero the KV cache slots.
        # NOTE: self.k_size is in BYTES (head_dim * bytes_per_element), but
        # torch.zeros(N, dtype=bfloat16) takes N as element count. Using
        # k_size as the element multiplier produces a tensor that's 2× the
        # actual KV slot size and overflows the next buffer. tensor_init has
        # the same bug but it's masked by subsequent initializations of the
        # buffers it overflows into. We must size correctly here because
        # we re-upload IDENTITY_DRAM right after and don't want to clobber it.
        kv_slot_elems = self._num_kv_slots * self.MAX_CONTEXT_SIZE * self.head_dim
        kv_zero_pad = torch.zeros(kv_slot_elems, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, kv_zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_K_ROPE_DRAM, kv_zero_pad)
        # Re-DMA identity matrix AFTER the KV zero-pads so it can't be
        # overwritten by overflow. decoder_attention_core reads this at
        # Stage 1 (I @ V^T); zeroing it produces all-zero attention output.
        self.dma_to_accelerator_memory(
            self.IDENTITY_DRAM_ADDR,
            torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))

        token_ids = inputs['input_ids'][0].tolist()
        mm_types = inputs['mm_token_type_ids'][0].tolist()

        del hf_model

        # Per-layer _vis_program_cache (the legacy mechanism that pickled
        # individual layer ISA bytes to disk) is intentionally not used here.
        # The unified instruction bin already caches the entire vision
        # encoder ISA on disk, so subsequent runs hit `contains_vision=true`
        # in the manifest and skip ISA emission entirely.

        print(f"  [Vision] FPGA path complete: {num_patches} patches → "
              f"{image_features.shape[0]} soft tokens, "
              f"image_features {tuple(image_features.shape)}")
        return image_features, token_ids, mm_types

    def _run_audio_encoder_fpga(self, audio_path: str, prompt: str) -> tuple[torch.Tensor, list[int], list[int]]:
        """Run the full audio encoder via the unified instruction bin.

        Synced from gemma4_e2b_test.py — load+execute path only (run_from_bin
        never compiles). The bin must already contain the audio section
        (built by gemma4_e2b_test.py's compile_instruction_bin at first run
        and shipped alongside this script). Steps:

          1. Load HF processor + audio file → input_features (host).
          2. Verify the unified bin manifest has the audio section and that
             T_raw matches the value baked at compile time.
          3. Replicate the compile-time allocator order
             (audio_weight_init → audio_tensor_init → audio_subsample_fpga
              lazy alloc) so DRAM addresses match the baked audio ISA.
          4. Host DMAs upload the actual mel features to AUD_SUB_INPUT and
             zero AUD_SUB_R_COMBINED / AUD_SUB_PATCHES1 / AUD_IO_A.
          5. start_execute_from_dram(audio_program_addr); wait.
          6. Read AUD_FEATURES_FINAL → audio_features.
          7. Restore LM KV cache + IDENTITY for the decoder.
        """
        from transformers import AutoProcessor
        # run_from_bin audio path: no HF model load. Audio weights come from
        # the combined weights bin's audio_section (folded by test.py's
        # weight_bin_generate); the processor only needs the bundled
        # tokenizer/processor configs.
        hf_model = None
        tok_subset = os.path.join(self.script_dir, "gemma4_e2b_bin", "tokenizer")
        if os.path.exists(os.path.join(tok_subset, "processor_config.json")):
            proc_dir = tok_subset
        else:
            proc_dir = os.path.join(self.script_dir, self._cfg["paths"]["hf_model_dir"])
            if not os.path.exists(os.path.join(proc_dir, "processor_config.json")):
                raise RuntimeError(
                    f"Audio mode needs processor_config.json in either:\n"
                    f"  {tok_subset}/\n"
                    f"  {proc_dir}/\n"
                    "Re-generate by running gemma4_e2b_test.py.")
        processor = AutoProcessor.from_pretrained(proc_dir, local_files_only=True)

        try:
            import soundfile as sf
        except ImportError as e:
            raise RuntimeError(
                "soundfile is required for audio input. `pip install soundfile`.") from e
        audio_array, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=-1)  # mono
        target_sr = getattr(processor.feature_extractor, "sampling_rate", 16000)
        if sr != target_sr:
            t = torch.from_numpy(audio_array).float().unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t,
                              size=int(audio_array.shape[0] * target_sr / sr),
                              mode="linear", align_corners=False)
            audio_array = t.squeeze().numpy()
        print(f"  [Audio] loaded {audio_path}: {audio_array.shape[0]/target_sr:.2f}s @ {target_sr} Hz")

        conversation = [{"role": "user", "content": [
            {"type": "audio"},
            {"type": "text", "text": prompt},
        ]}]
        text_prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(
            text=[text_prompt],
            audio=[audio_array],
            sampling_rate=target_sr,
            return_tensors="pt",
        )
        input_features = inputs["input_features"]
        input_features_mask = inputs.get("input_features_mask")
        token_ids = inputs["input_ids"][0].tolist()
        mm_types = inputs["mm_token_type_ids"][0].tolist()
        print(f"  [Audio] {len(token_ids)} prompt tokens "
              f"({sum(1 for m in mm_types if m == 3)} audio, "
              f"{sum(1 for m in mm_types if m == 0)} text)")

        _tensor_dram_save = self._tensor_dram_addr
        _prog_dram_addr_save = self._next_program_dram_addr
        _prog_dram_base_save = self._program_dram_base

        bin_dir    = os.path.join(self.script_dir, "gemma4_e2b_bin")
        instr_bin  = os.path.join(bin_dir, "programs.bin")
        instr_meta = os.path.join(bin_dir, "programs.json")
        if not (os.path.exists(instr_bin) and os.path.exists(instr_meta)):
            raise SystemExit(
                "[Audio] programs.bin missing — generate it on the "
                "build host with `python gemma4_e2b_test.py --audio-enable` and "
                "copy the bin into gemma4_e2b_bin/ here.")
        with open(instr_meta, "r") as f:
            _peek = json.load(f)
        if not _peek.get("contains_audio"):
            raise SystemExit(
                "[Audio] unified bin lacks the audio section. Rebuild with "
                "gemma4_e2b_test.py so compile_instruction_bin folds audio in.")
        T_raw = int(input_features.shape[1])
        bin_T_raw = _peek["audio_T_raw"]
        if T_raw != bin_T_raw:
            raise SystemExit(
                f"[Audio] input_features shape mismatch: got T_raw={T_raw}, "
                f"bin was compiled for T_raw={bin_T_raw}.")
        H0 = (T_raw + 2 - 3) // 2 + 1
        H1 = (H0 + 2 - 3) // 2 + 1
        T_sub = H1

        # Replicate the compile-time allocator order so addresses baked into
        # the audio ISA resolve to the same DRAM bytes at runtime.
        # CRITICAL: compile_instruction_bin calls vision_weight_init BEFORE
        # audio_weight_init, and vision_weight_init advances the tensor DRAM
        # cursor by ~80 MB (vision weights land in tensor DRAM, not params).
        # audio_weight_init then saves the cursor at V+80MB; audio_tensor_init
        # allocates audio buffers from V+80MB. The audio ISA bakes those
        # V+80MB addresses. Skipping vision_weight_init at runtime puts audio
        # buffers 80 MB lower → bin reads uninitialized memory → NaN audio
        # features → all-<pad> decode. So we run vision_weight_init at audio
        # runtime too (cheap: loads from combined weight bin's vision_section).
        # Silence the library's tile-planner prints across the encoder
        # init + subsample setup. _SILENT_MODE catches anything routed
        # through our `quiet_print` wrapper; the builtins.print swap catches
        # library prints that pre-captured a reference to the original
        # builtin (belt-and-suspenders).
        global _SILENT_MODE
        _SILENT_MODE = True
        _orig_builtin_print = builtins.print
        builtins.print = lambda *a, **kw: None
        try:
            self.vision_weight_init(hf_model)
            self.audio_weight_init(hf_model, reset_allocator=False)
            self.audio_tensor_init(T_sub)
            L_pad = self._aud_L_pad
            H = self.AUD_H

            # Host DMAs: upload mel + zero scratch via audio_subsample_fpga in
            # capture-suppressed mode (capture_buffer=None) so FPGA emits are
            # no-ops while host DMAs still move bytes to FPGA DRAM.
            _saved_capture_buffer = self.capture_buffer
            _saved_capture_count  = self.capture_count
            _saved_is_capture_on  = self.is_capture_on
            self.capture_buffer = None
            self.is_capture_on  = False
            self._oneshot_mode  = True
            try:
                self.audio_subsample_fpga(input_features, input_features_mask)
            finally:
                self._oneshot_mode  = False
                self.capture_buffer = _saved_capture_buffer
                self.capture_count  = _saved_capture_count
                self.is_capture_on  = _saved_is_capture_on
        finally:
            _SILENT_MODE = False
            builtins.print = _orig_builtin_print
        self.dma_to_accelerator_memory(self.AUD_IO_A,
            torch.zeros(L_pad * H, dtype=torch.bfloat16))

        audio_program_addr = int(_peek["audio_program_start_addr"], 16)
        audio_program_size = _peek["audio_program_size"]
        # FPGA dispatch quirk (also documented in _run_vision_encoder_fpga):
        # start_execute_from_dram is unreliable in a fresh process unless a
        # DMA write touches the program region first. After load_instruction_bin
        # the audio section IS in DRAM, but in a fresh process the FPGA's
        # instruction prefetch queue may be stale (loaded from a previous
        # session). Re-DMA just the audio section to kick the prefetch.
        audio_offset_in_bin = audio_program_addr - int(_peek["instruction_base_addr"], 16)
        bin_path_disk = os.path.join(self.script_dir, _peek["instruction_bin"])
        with open(bin_path_disk, "rb") as f:
            f.seek(audio_offset_in_bin)
            audio_bytes_disk = f.read(audio_program_size)
        _t_dma = time.perf_counter()
        self.dma_write(DMA_DEVICE_H2C, audio_program_addr, audio_bytes_disk, audio_program_size)
        print(f"  [Audio] re-DMA audio section ({audio_program_size/1024/1024:.1f} MB in "
              f"{time.perf_counter()-_t_dma:.2f}s) — kicks FPGA before execute", flush=True)
        print(f"  [Audio] launching audio one-shot at 0x{audio_program_addr:X} "
              f"({audio_program_size/1024/1024:.1f} MB ISA)", flush=True)
        aud_t0 = time.perf_counter()
        self.start_execute_from_dram(audio_program_addr)
        self.wait_queue(180.0)
        print(f"    [Audio] one-shot audio pipeline done in {time.perf_counter()-aud_t0:.1f}s")

        OUT_DIM = self.AUD_OUT_DIM
        audio_features = self.dma_from_accelerator_memory(
            self.AUD_FEATURES_FINAL, (L_pad, OUT_DIM)).cpu()[:T_sub].to(torch.bfloat16)
        n_audio_slots = sum(1 for m in mm_types if m == 3)
        if audio_features.shape[0] != n_audio_slots:
            print(f"  [Audio] WARNING: encoder produced {audio_features.shape[0]} "
                  f"soft tokens but prompt has {n_audio_slots} audio slots; "
                  f"truncating/padding to fit.")
            if audio_features.shape[0] > n_audio_slots:
                audio_features = audio_features[:n_audio_slots]
            else:
                pad = torch.zeros(n_audio_slots - audio_features.shape[0],
                                  audio_features.shape[1], dtype=torch.bfloat16)
                audio_features = torch.cat([audio_features, pad], dim=0)

        # ---- Restore LM state so decoder can run ----
        self._tensor_dram_addr = _tensor_dram_save
        self._next_program_dram_addr = _prog_dram_addr_save
        self._program_dram_base = _prog_dram_base_save
        kv_slot_elems = self._num_kv_slots * self.MAX_CONTEXT_SIZE * self.head_dim
        kv_zero_pad = torch.zeros(kv_slot_elems, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, kv_zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_K_ROPE_DRAM, kv_zero_pad)
        self.dma_to_accelerator_memory(
            self.IDENTITY_DRAM_ADDR,
            torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))

        del hf_model

        print(f"  [Audio] FPGA path complete: {T_sub} frames → "
              f"{audio_features.shape[0]} soft tokens, "
              f"audio_features {tuple(audio_features.shape)}")
        return audio_features, token_ids, mm_types

    def set_prefill_seq_audio(self, audio_path: str, prompt: str = None) -> None:
        """Set prefill_seq for audio input: run audio encoder on FPGA,
        merge audio features into the LM embedding stream during run_prefill
        at mm_types == 3 positions."""
        if prompt is None:
            prompt = "Describe what you hear."
        audio_features, token_ids, mm_types = self._run_audio_encoder_fpga(audio_path, prompt)
        self.prefill_seq = tuple(token_ids)
        self._audio_features = audio_features  # [N_soft_tokens, 1536]
        self._mm_types = mm_types
        n_audio = (torch.tensor(mm_types) == 3).sum().item()
        n_text = (torch.tensor(mm_types) == 0).sum().item()
        print(f"Audio prefill: {len(token_ids)} tokens ({n_audio} audio, {n_text} text)")
        print(f"Audio features: {tuple(audio_features.shape)}")

    def set_prefill_seq_vlm(self, image_path: str, prompt: str = None) -> None:
        """Set prefill_seq for VLM: run vision encoder on FPGA, merge image
        features into embeddings during run_prefill."""
        if prompt is None:
            prompt = "Describe this image in detail."

        image_features, token_ids, mm_types = self._run_vision_encoder_fpga(image_path, prompt)

        # Store for later use
        self.prefill_seq = tuple(token_ids)
        self._image_features = image_features  # [N_soft_tokens, 1536]
        self._mm_types = mm_types

        print(f"VLM prefill: {len(token_ids)} tokens ({(torch.tensor(mm_types) == 1).sum().item()} image, "
              f"{(torch.tensor(mm_types) == 0).sum().item()} text)")

    def reset_isa_reg_counter(self) -> None:
        """Reset the ISA register allocation counter to 1 (register 0 is hard-wired zero)."""
        self._isa_reg_counter = 1

    def alloc_isa_reg(self, reset: bool = False) -> int:
        """
        Allocate the next available general-purpose ISA register.

        General-purpose ISA registers are 32 bits wide: regs 0..15.
        Register 0 is a hard-wired zero register, so allocation starts from 1.

        Args:
            reset: If True, reset the counter to 1 before allocation (default: False)

        Returns:
            The allocated register index (1-15)

        Raises:
            ValueError: If all available registers (1-15) have been allocated
        """
        if reset:
            self._isa_reg_counter = 1

        if self._isa_reg_counter > 15:
            raise ValueError("Exceeded available ISA registers (max 15)")

        reg_idx = self._isa_reg_counter
        self._isa_reg_counter += 1
        return reg_idx


    def isa_add_set_core(self, dst_reg_idx: int, immediate_value: int, timeout_s: float = 10.0) -> None:
        """
        Run a minimal program that sets one ISA register to an immediate value (ADD SET then HALT):
        start_capture -> generate_instruction_add_set -> stop_capture -> halt -> write to DRAM -> execute -> wait.
        Use e.g. isa_add_set_core(self.gpr_seq_len, self.seq_len).
        """
        self.isa_add_set_multi([(dst_reg_idx, immediate_value)], timeout_s=timeout_s)

    def isa_add_set_multi(self, reg_values: list[tuple[int, int]], timeout_s: float = 10.0) -> None:
        """Set multiple ISA registers in one program submission.

        Each capture → DMA → execute → wait cycle costs ~5–10 ms of round-trip
        overhead, so combining 3 register sets per decode token into a single
        program saves two round trips. Order of writes inside the program
        matches the order of `reg_values`.
        """
        self.clear_inst_id()
        self.start_capture()
        for dst_reg_idx, immediate_value in reg_values:
            self.generate_instruction_add_set(dst_reg_idx, immediate_value)
        self.stop_capture()
        self.generate_instruction_halt()
        program_addr = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(program_addr)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        self.clear_capture_buffer()
        self.start_execute_from_dram(program_addr)
        self.wait_queue(timeout_s)

    def write_captured_instructions_to_file(self, start_addr: int, filename: str = "captured_instructions.bin") -> None:
        """
        Write all captured instructions to a binary file.

        Args:
            start_addr: DRAM address where instructions are intended to be stored (used for logging/naming if needed)
            filename: Name of the file to write to
        """
        if not hasattr(self, 'capture_buffer') or not self.capture_buffer:
            return

        all_instructions_bytes = bytearray()
        for inst in self.capture_buffer:
            all_instructions_bytes.extend(inst.get_bytes())

        with open(filename, "wb") as f:
            f.write(all_instructions_bytes)

        print(f"Successfully wrote {len(self.capture_buffer)} captured instructions ({len(all_instructions_bytes)} bytes) to {filename}")

    def load_instructions(self, bin_path: str) -> tuple[int, int]:
        """Load decoder instruction bin from file into program DRAM. Returns (start_addr, total_size)."""
        with open(bin_path, "rb") as f:
            data = f.read()
        total_size = len(data)
        start_addr = self.allocate_program_dram(total_size)
        self.dma_write(DMA_DEVICE_H2C, start_addr, data, total_size)
        print(f"    Loaded {total_size} bytes from instruction.bin to DRAM at 0x{start_addr:x}")
        return start_addr, total_size

    # Overwrite UnifiedEngine allocate_params_dram
    def allocate_params_dram(self, size_bytes: int) -> int:
        """
        Allocate memory from the params DRAM region incrementally.

        Args:
            size_bytes: Number of bytes to allocate

        Returns:
            The DRAM address of the allocated block (address before increment).
        """
        params_dram_addr = self._next_params_dram_addr
        self._next_params_dram_addr += size_bytes
        return params_dram_addr

    def clear_inst_id(self) -> None:
        """Reset instruction ID counter for the next capture."""
        self._inst_id = 0

    def get_arg_max_index(self) -> int:
        """Get the arg max index from the Unified Engine"""
        return self.read_reg32(UE_ARGMAX_INDEX)

    # ---- On-FPGA repetition penalty (llama3.2_1b mechanism, default OFF) -----
    # The LM-head matmul (baked in the bin) adds PENALTY_BIAS_DRAM as its C term;
    # when GEMMA4_PENALTY=1 the decode loop refreshes this bias each step so the
    # HW argmax of (logits + bias) returns the penalized token — no readback.
    def _structural_token_ids(self) -> set:
        """Token ids never repetition-penalized (punctuation/whitespace/special);
        exempting these 'glue' tokens stops penalized text collapsing. Cached."""
        cached = getattr(self, "_struct_ids_cache", None)
        if cached is not None:
            return cached
        import string
        allowed = set(string.punctuation) | set(string.whitespace) | set("—–’‘“”…·•‹›«»¡¿")
        ids = set(int(i) for i in (getattr(self.tokenizer, "all_special_ids", []) or []))
        for i in range(self.EMBEDDING_ELEMENTS):
            s = self.tokenizer.decode([i]).strip()
            if s == "" or all(ch in allowed for ch in s):
                ids.add(i)
        self._struct_ids_cache = ids
        return ids

    def _structural_ids_tensor(self) -> torch.Tensor:
        t = getattr(self, "_struct_ids_tensor_cache", None)
        if t is None:
            t = torch.tensor(sorted(self._structural_token_ids()), dtype=torch.long)
            self._struct_ids_tensor_cache = t
        return t

    def _write_penalty_bias(self, prev_tokens) -> None:
        """Build the per-vocab additive bias from the windowed token frequency and
        DMA it to PENALTY_BIAS_DRAM. bias[t] = clamp(-alpha*count[t], min=-cap);
        structural tokens stay 0. Identical formula to gemma4_e2b_test.py."""
        vocab = self.EMBEDDING_ELEMENTS
        alpha = float(getattr(self, "pen_alpha", 1.0))
        cap = float(getattr(self, "pen_cap", 20.0))
        W = int(getattr(self, "rep_window", 256))
        window = prev_tokens[-W:]
        count = torch.zeros(vocab, dtype=torch.float32)
        if window:
            win = torch.tensor(window, dtype=torch.long)
            count.index_add_(0, win, torch.ones(win.numel(), dtype=torch.float32))
            count[self._structural_ids_tensor()] = 0.0
        bias = (-alpha * count).clamp(min=-cap).to(torch.bfloat16).view(1, vocab)
        # --- anti-loop hard ban (overrides the structural exemption above) ---
        # The structural exemption keeps glue tokens ("|", newline, space) out of
        # the soft frequency penalty so penalized text doesn't turn to word-salad
        # -- but that is exactly why the penalty alone can't break a degenerate
        # collapse whose tokens ARE structural (e.g. a "|"<->"\n" 2-cycle, which
        # naive consecutive-run detection would miss). Instead: over the last
        # `recent_w` generated tokens, hard-ban any token that fills >= `loop_thr`
        # of them (single-token run = all; 2-cycle = ~half each). No coherent text
        # fills a third of a short window with one token, so it never fires on
        # real output. (E2B VLM is empirically immune to the E4B image cycle, but
        # the penalty path is shared; harmless here, present for symmetry.)
        # Tunable: GEMMA4_PEN_LOOP_RECENT (window, 0=off), GEMMA4_PEN_LOOP_THR.
        # Identical formula to gemma4_e2b_test.py.
        recent_w = int(getattr(self, "pen_loop_recent", 24))
        loop_thr = int(getattr(self, "pen_loop_thr", 8))
        if recent_w > 0 and len(prev_tokens) >= recent_w:
            from collections import Counter
            _cnt = Counter(int(t) for t in prev_tokens[-recent_w:])
            _ban = [tok for tok, c in _cnt.items() if c >= loop_thr]
            if _ban:
                bias[0, torch.tensor(_ban, dtype=torch.long)] = -1e9  # finite, bf16-safe
        self.dma_to_accelerator_memory(self.PENALTY_BIAS_DRAM, bias)

    # rope_hf_core override removed — use base-class implementation in
    # user_dma_core.py which supports gr_weight_dram for dynamic-PBI decode.

    def decoder_attention_core(self, head_dim: int, seq_len: int, Q_DRAM_ADDR: int, K_DRAM_ADDR: int, V_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, SCRATCH_DRAM_ADDR: int, IDENTITY_DRAM_ADDR: int =None, BIAS_DRAM_ADDR: int = None,
                            debug_mode: bool = False, SM_OUTPUT_DRAM_ADDR: int = None, q_rows: int = 1) -> None:
        """Decoder-side attention kernel for a single KV head (GQA).

        q_rows: number of Q rows to compute against this shared K/V in a
            single call. In a GQA layer with group_size query heads sharing
            one KV head, pass q_rows=group_size to process all group_size
            query heads in one call — K, V, V^T are loaded from DRAM once
            and reused across all q_rows. Q_DRAM_ADDR must point to q_rows
            contiguous head_dim-sized vectors, and OUTPUT_DRAM_ADDR must
            have room for q_rows contiguous head_dim-sized outputs.
            Default (q_rows=1) preserves the original per-head behaviour.
        """

        bytes_per_element = 2
        bias_enable = True if BIAS_DRAM_ADDR is not None else False

        if debug_mode: # DEBUG only, needs to be allocated in DRAM
            assert SM_OUTPUT_DRAM_ADDR is not None, "SM_OUTPUT_DRAM_ADDR is not set for debug mode"

        # SCRATCH_DRAM_ADDR is used for V^T
        SCRATCH_DRAM_PARTIAL_SM = SCRATCH_DRAM_ADDR + head_dim * seq_len * bytes_per_element # used for partial softmax output

        # ----------------------------------------------------------------------------------------------------------------
        # I @ V^T: (head_dim, head_dim) @ (seq_len, head_dim)^T -> (head_dim, seq_len)
        # Convention: first matrix I is (M, K), second V^T is (K, N), output  (M, N)
        M = head_dim   # identity length (rows of I)
        K = head_dim  # identity dimension (inner product dim)
        N = seq_len   # V length (columns of V^T)

        identity_tensor = torch.eye(head_dim, dtype=torch.bfloat16)

        # transfer identity matrix to URAM_A start
        self.accelerator_memory_to_sram(accelerator_dram_address=IDENTITY_DRAM_ADDR,
                                        sram_address=0,
                                        element_size=UE_VECTOR_SIZE * UE_VECTOR_SIZE)

        usable_uram_a_start_addr = UE_VECTOR_SIZE * UE_VECTOR_SIZE * bytes_per_element

        # URAM_B is used for V matrix, we need to chunk the V matrix into smaller chunks that can fit in URAM_B
        usable_uram_b_elements = URAM_NEAR_FULL_ELEMENTS
        N_chunk = min(N, (usable_uram_b_elements // K) // UE_VECTOR_SIZE * UE_VECTOR_SIZE)
        N_chunk_aligned = None
        if N_chunk < UE_VECTOR_SIZE:
            if (K * 32) <= usable_uram_b_elements:
                N_chunk = 32
            elif (K * 16) <= usable_uram_b_elements:
                N_chunk = 16
            else:
                assert False, f"K={K} is too large to fit in usable URAM elements={usable_uram_b_elements}"
            N_chunk_aligned = UE_VECTOR_SIZE

        usable_uram_a_elements = URAM_FULL_ELEMENTS - UE_VECTOR_SIZE * UE_VECTOR_SIZE
        output_N_size = N_chunk_aligned if N_chunk_aligned is not None else N_chunk
        M_chunk = min(M, usable_uram_a_elements // output_N_size)
        assert M_chunk >= 1 and M_chunk <= M, f"M_chunk={M_chunk} must be greater than 0 and less than M={M}"


        output_sram_wb_addr = usable_uram_a_start_addr
        uram_b_start_addr = 0x80000
        for i, m_take in self.chunk_ranges(M, M_chunk):
            for j, n_take in self.chunk_ranges(N, N_chunk):

                self.accelerator_memory_to_sram(accelerator_dram_address=V_DRAM_ADDR + j * K * bytes_per_element,
                                            sram_address=uram_b_start_addr,
                                            element_size=n_take * K)

                for output_row in range(m_take):
                    if N_chunk_aligned is None:
                        out_sram_offset = output_row * n_take * bytes_per_element
                    else:
                        out_sram_offset = output_row * N_chunk_aligned * bytes_per_element

                    ones_idx = identity_tensor[output_row+i, :].reshape(-1, UE_VECTOR_SIZE).sum(axis=1).argmax(axis=0)
                    vector_idx = identity_tensor[output_row+i, :].reshape(-1, UE_VECTOR_SIZE)[ones_idx, :].argmax(axis=0)

                    self.start_queue_for_bf16_matvec_operation(max_clear_en=0,
                                                            fmax_context_addr=0,
                                                            vector_sram_start_addr=0x00000 + vector_idx * UE_VECTOR_SIZE * bytes_per_element,
                                                            matrix_sram_start_addr=uram_b_start_addr + ones_idx * UE_VECTOR_SIZE * bytes_per_element,
                                                            output_sram_wb_addr=output_sram_wb_addr + out_sram_offset,
                                                            K=UE_VECTOR_SIZE,
                                                            N=n_take,
                                                            stride_z=m_take)

                start_dram_address_of_partial_matrix = SCRATCH_DRAM_ADDR + i * N * bytes_per_element + j * bytes_per_element # the space needed is head_dim x seq_len

                if N_chunk_aligned is None:
                    self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr,
                                                    accelerator_dram_address=start_dram_address_of_partial_matrix,
                                                    element_size=m_take * n_take,
                                                    stride_bytes_per_chunk=n_take * bytes_per_element,
                                                    stride_jump_bytes=N * bytes_per_element)
                else:
                    for o_row_idx in range(m_take):
                        self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr + o_row_idx * N_chunk_aligned * bytes_per_element,
                                                        accelerator_dram_address=start_dram_address_of_partial_matrix + o_row_idx * N * bytes_per_element,
                                                        element_size=n_take)

        # ----------------------------------------------------------------------------------------------------------------
        # Q @ K^T: (q_rows, head_dim) @ (head_dim, seq_len) -> (q_rows, seq_len)
        # Convention: first matrix Q is (M, K), second K^T is (K, N), output scores (M, N)
        # With q_rows > 1 (GQA head fusion), a single load of each K chunk
        # is reused across all q_rows Q rows — amortizes the K DMA.
        M = q_rows    # query length (rows of Q)
        K = head_dim  # head dimension (inner product dim)
        N = seq_len   # key length (columns of K^T)
        # Calculate N_chunk
        usable_uram_b_elements = URAM_NEAR_FULL_ELEMENTS
        N_chunk = min(N, (usable_uram_b_elements // K) // UE_VECTOR_SIZE * UE_VECTOR_SIZE)
        N_chunk_aligned = None
        if N_chunk < UE_VECTOR_SIZE:
            if (K * 32) <= usable_uram_b_elements:
                N_chunk = 32
            elif (K * 16) <= usable_uram_b_elements:
                N_chunk = 16
            else:
                assert False, f"K={K} is too large to fit in usable URAM elements={usable_uram_b_elements}"
            N_chunk_aligned = UE_VECTOR_SIZE

        usable_uram_a_elements = URAM_FULL_ELEMENTS
        output_N_size = N_chunk_aligned if N_chunk_aligned is not None else N_chunk
        M_chunk = min(UE_FMAX_CONTEXT_SIZE, M, usable_uram_a_elements // (K + output_N_size))
        assert M_chunk >= 1 and M_chunk <= M, f"M_chunk={M_chunk} must be greater than 0 and less than M={M}"


        uram_a_start_addr = 0x00000
        uram_b_start_addr = 0x80000
        for i, m_take in self.chunk_ranges(M, M_chunk):
            self.accelerator_memory_to_sram(accelerator_dram_address=Q_DRAM_ADDR + i * K * bytes_per_element,
                                            sram_address=uram_a_start_addr,
                                            element_size=m_take * K)

            self.broadcast_mul(scalar=1 / math.sqrt(head_dim),
                                    sram_start_addr=uram_a_start_addr,
                                    sram_wb_addr=uram_a_start_addr,
                                    element_size=m_take * K)

            output_sram_wb_addr = uram_a_start_addr + m_take * K * bytes_per_element

            assert output_sram_wb_addr < 0x80000, f"output_sram_wb_addr={output_sram_wb_addr} is greater than 0x80000, which is the size of URAM_B"

            clear_en = 1
            for j, n_take in self.chunk_ranges(N, N_chunk):
                self.accelerator_memory_to_sram(accelerator_dram_address=K_DRAM_ADDR + j * K * bytes_per_element,
                                            sram_address=uram_b_start_addr,
                                            element_size=n_take * K)

                if bias_enable:
                    self.accelerator_memory_to_bias_sram(accelerator_dram_address=BIAS_DRAM_ADDR + j * bytes_per_element,
                                                       element_size=n_take)

                assert m_take * K + n_take * m_take<= URAM_FULL_ELEMENTS

                for output_row in range(m_take):
                    if N_chunk_aligned is None:
                        out_sram_offset = output_row * n_take * bytes_per_element
                    else:
                        out_sram_offset = output_row * N_chunk_aligned * bytes_per_element

                    self.start_queue_for_bf16_matvec_operation(max_clear_en=clear_en,
                                                            fmax_context_addr=output_row,
                                                            vector_sram_start_addr=uram_a_start_addr + output_row * K * bytes_per_element,
                                                            matrix_sram_start_addr=uram_b_start_addr,
                                                            output_sram_wb_addr=output_sram_wb_addr + out_sram_offset,
                                                            K=K,
                                                            N=n_take,
                                                            bias_enable=bias_enable)
                    clear_en = 0

                start_dram_address_of_partial_matrix = SCRATCH_DRAM_PARTIAL_SM + j * bytes_per_element

                if N_chunk_aligned is None:
                    self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr,
                                                    accelerator_dram_address=start_dram_address_of_partial_matrix,
                                                    element_size=m_take * n_take,
                                                    stride_bytes_per_chunk=n_take * bytes_per_element,
                                                    stride_jump_bytes=N * bytes_per_element)
                else:
                    for o_row_idx in range(m_take):
                        self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr + o_row_idx * N_chunk_aligned * bytes_per_element,
                                                        accelerator_dram_address=start_dram_address_of_partial_matrix + o_row_idx * N * bytes_per_element,
                                                        element_size=n_take)


            # SOFTMAX CALCULATION
            max_m_take = min((URAM_FULL_ELEMENTS - UE_VECTOR_SIZE) // N, UE_FMAX_CONTEXT_SIZE) # worst case scenario, leave one row for output

            for m_take_chunk_idx, m_take_chunk_size in self.chunk_ranges(m_take, max_m_take):
                self.accelerator_memory_to_sram(accelerator_dram_address=SCRATCH_DRAM_PARTIAL_SM + m_take_chunk_idx * N * bytes_per_element,
                                            sram_address=uram_a_start_addr,
                                            element_size=m_take_chunk_size * N)

                # Reuse input sram_wb_addr for softmax output
                for row_idx in range(m_take_chunk_size):
                    self.start_queue_for_bf16_softmax_operation(fmax_context_addr=row_idx + m_take_chunk_idx,
                                                                vector_sram_start_addr=uram_a_start_addr + row_idx * N * bytes_per_element,
                                                                output_sram_wb_addr=uram_a_start_addr + row_idx * N * bytes_per_element,
                                                                N=N)


                # softmax output tap point - DEBUG only
                if debug_mode:
                    self.sram_to_accelerator_memory(sram_address=uram_a_start_addr,
                                    accelerator_dram_address=SM_OUTPUT_DRAM_ADDR + (i + m_take_chunk_idx) * N * bytes_per_element,
                                    element_size=m_take_chunk_size * N)

                v_tr_row_chunk_size = min((URAM_NEAR_FULL_ELEMENTS // seq_len // UE_VECTOR_SIZE) * UE_VECTOR_SIZE,
                                        ((URAM_FULL_ELEMENTS - m_take_chunk_size * seq_len) // m_take_chunk_size // UE_VECTOR_SIZE) * UE_VECTOR_SIZE,
                                        head_dim)

                v_tr_row_chunk_size_aligned = None
                if v_tr_row_chunk_size < UE_VECTOR_SIZE:
                    v_tr_row_chunk_size_aligned = UE_VECTOR_SIZE
                    if seq_len * 32 <= URAM_NEAR_FULL_ELEMENTS:
                        v_tr_row_chunk_size = 32
                    elif seq_len * 16 <= URAM_NEAR_FULL_ELEMENTS:
                        v_tr_row_chunk_size = 16
                    else:
                        assert False, f"v_tr_row_chunk_size={v_tr_row_chunk_size} is too large to fit in URAM_NEAR_FULL_ELEMENTS={URAM_NEAR_FULL_ELEMENTS}"

                v_t_sram_start_addr = 0x80000 # URAM_B start
                output_sram_wb_addr = uram_a_start_addr + m_take_chunk_size * seq_len * bytes_per_element

                for v_tr_column_idx, v_tr_column_take in self.chunk_ranges(head_dim, v_tr_row_chunk_size):
                    self.accelerator_memory_to_sram(accelerator_dram_address=SCRATCH_DRAM_ADDR + v_tr_column_idx * seq_len * bytes_per_element,
                                                sram_address=v_t_sram_start_addr,
                                                element_size=v_tr_column_take * seq_len)

                    for p_row_idx in range(m_take_chunk_size):
                        if v_tr_row_chunk_size_aligned is None:
                            output_sram_wb_offset = p_row_idx * v_tr_column_take * bytes_per_element
                        else:
                            output_sram_wb_offset = 0

                        self.start_queue_for_bf16_matvec_operation(max_clear_en=0,
                                                                fmax_context_addr=0,
                                                                vector_sram_start_addr=uram_a_start_addr + p_row_idx * seq_len * bytes_per_element,
                                                                matrix_sram_start_addr=v_t_sram_start_addr,
                                                                output_sram_wb_addr=output_sram_wb_addr + output_sram_wb_offset,
                                                                K=seq_len,
                                                                N=v_tr_column_take)

                        if v_tr_row_chunk_size_aligned is not None:
                            self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr + output_sram_wb_offset,
                                                            accelerator_dram_address=OUTPUT_DRAM_ADDR + (i + m_take_chunk_idx) * head_dim * bytes_per_element
                                                                                                        + v_tr_column_idx * bytes_per_element
                                                                                                        + p_row_idx * head_dim * bytes_per_element,
                                                            element_size=v_tr_column_take)


                    if v_tr_row_chunk_size_aligned is None:
                        self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr,
                                                        accelerator_dram_address=OUTPUT_DRAM_ADDR + (i + m_take_chunk_idx) * head_dim * bytes_per_element + v_tr_column_idx * bytes_per_element,
                                                        element_size=m_take_chunk_size * v_tr_column_take,
                                                        stride_bytes_per_chunk=v_tr_column_take * bytes_per_element,
                                                        stride_jump_bytes=head_dim * bytes_per_element)

        # Total Theoretical FLOPS
        total_flops = 1 * head_dim # q_scale
        total_flops += 2 * 1 * head_dim * seq_len # Q @ K^T
        total_flops += 1 * seq_len * 5 # softmax
        total_flops += 2 * 1 * seq_len * head_dim # sm @ v
        return total_flops

    # ================================================================
    #  Vision encoder on FPGA
    # ================================================================

    # Vision encoder constants
    VIS_H = 768          # hidden size
    VIS_HEADS = 12       # number of attention heads
    VIS_HEAD_DIM = 64    # head_dim = 768 / 12
    VIS_MLP = 3072       # intermediate_size
    VIS_LAYERS = 16      # num_hidden_layers
    VIS_ROPE_DIM = 32    # half of head_dim for 2D RoPE (64 / 2)

    def _vision_weight_init_from_combined_bin(self, weights_bin_path: str, vision_section: dict) -> None:
        """Load pre-quantized vision weights from the vision section of the
        combined weights bin and DMA to FPGA. No HF model, no quantization.
        Mirrors the address allocation order of the legacy HF path so
        DRAM addresses match what's baked into the instruction bin.
        """
        base_offset = int(vision_section["offset"])
        meta = vision_section["manifest"]
        if meta.get("vision_quant_precision") != VISION_QUANT_PRECISION:
            raise RuntimeError(
                f"vision section quant precision mismatch (disk: {meta.get('vision_quant_precision')!r}, "
                f"expected {VISION_QUANT_PRECISION!r}). Regenerate the weights bin.")
        self.VIS_ROPE_PERM = torch.tensor(meta["VIS_ROPE_PERM"], dtype=torch.long)
        self.VIS_POOL_K    = int(meta["VIS_POOL_K"])
        self.VIS_TEXT_H    = int(meta["VIS_TEXT_H"])
        L_count = int(meta["num_layers"])
        sections = meta["sections"]

        def _str_to_float(v):
            if v == "inf":  return float("inf")
            if v == "-inf": return -float("inf")
            return float(v)
        self._vis_clip_ranges = []
        for cr in meta["clip_ranges"]:
            row = {}
            for k, v in cr.items():
                row[k] = {
                    "input":  (_str_to_float(v["input"][0]),  _str_to_float(v["input"][1])),
                    "output": (_str_to_float(v["output"][0]), _str_to_float(v["output"][1])),
                }
            self._vis_clip_ranges.append(row)

        print(f"\n[Vision] Loading pre-quantized vision weights from combined weights bin "
              f"({VISION_QUANT_PRECISION.upper()} block=64 + BF16 norms) ...")
        f = open(weights_bin_path, "rb")
        try:
            def _dma_section(key: str) -> int:
                s = sections[key]
                f.seek(base_offset + s["offset"])
                bts = f.read(s["size"])
                if len(bts) != s["size"]:
                    raise RuntimeError(f"truncated section read {key} at offset {base_offset + s['offset']}")
                addr = self.allocate_tensor_dram(s["size"])
                self.dma_write(DMA_DEVICE_H2C, addr, bts, s["size"])
                return addr

            layer_weight_addrs = []
            for li in range(L_count):
                pre = f"layer{li}"
                addrs = {}
                for n in ["input_layernorm", "post_attention_layernorm",
                          "pre_feedforward_layernorm", "post_feedforward_layernorm",
                          "q_norm", "k_norm"]:
                    addrs[n] = _dma_section(f"{pre}.{n}")
                for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]:
                    scale_addr = _dma_section(f"{pre}.{proj_name}.scale")
                    data_addr  = _dma_section(f"{pre}.{proj_name}.data")
                    addrs[proj_name] = {
                        "data": data_addr, "scale": scale_addr,
                        "shape": tuple(sections[f"{pre}.{proj_name}.data"]["shape"]),
                    }
                layer_weight_addrs.append(addrs)
            self._vis_weight_addrs = layer_weight_addrs

            # Match the legacy vision_weight_init allocation order exactly so
            # that DRAM addresses produced here are identical to the addresses
            # baked into the instruction bin. Diverging order = wrong
            # tensors at every vision op → garbage output.
            s = sections["pos_embedding_table"]
            f.seek(base_offset + s["offset"])
            bts = f.read(s["size"])
            self._vis_pos_embed_table = torch.frombuffer(
                bytearray(bts), dtype=torch.bfloat16
            ).reshape(*s["shape"]).clone()

            self.VIS_EMBED_NORM_HAS_SCALE = False
            self.VIS_EMBED_NORM_GAMMA = _dma_section("embed_norm_gamma")

            for nm in ("patch_proj", "embed_proj"):
                scale_addr = _dma_section(f"{nm}.scale")
                data_addr  = _dma_section(f"{nm}.data")
                info = {"data": data_addr, "scale": scale_addr,
                        "shape": tuple(sections[f"{nm}.data"]["shape"])}
                if nm == "patch_proj":
                    self.VIS_PATCH_PROJ_INFO = info
                else:
                    self.VIS_EMBED_PROJ_INFO = info

            # V-norm ones-gamma allocated LAST in the legacy path.
            self.VIS_V_NORM_ONES_GAMMA = _dma_section("v_norm_ones_gamma")
        finally:
            f.close()
        self._vis_weight_end = self.get_tensor_dram_addr()
        print(f"  Vision weights loaded from combined bin. Tensor DRAM usage: "
              f"{self.get_tensor_dram_usage()/(1024*1024):.1f} MB")

    def vision_weight_init(self, hf_model) -> None:
        """Upload vision encoder weights to FPGA DRAM.

        run_from_bin reads pre-quantized vision weights from the vision
        section inside the combined weights bin — no HF model, no IF4
        quantization. The legacy HF + quantize path is intentionally
        absent from this script.

        Idempotent: a second call (e.g., when audio runtime invokes it to
        match the compile-time tensor-DRAM cursor) is a no-op.
        """
        if getattr(self, "_vision_weight_init_done", False):
            return
        master = getattr(self, "_weights_master", None)
        if master is None or "vision_section" not in master or "manifest" not in master["vision_section"]:
            raise RuntimeError(
                "VLM mode in run_from_bin needs the vision section inside the combined "
                "weights bin. Either weight_init didn't run, or the bin was produced under "
                "the old multi-file layout. Regenerate by running gemma4_e2b_test.py.")
        weights_bin_path = os.path.join(self.script_dir, self._weights_bin_rel)
        self._vision_weight_init_from_combined_bin(weights_bin_path, master["vision_section"])
        self._vision_weight_init_done = True
        return

        # LEGACY HF + quantize path commented out (2026-05-21) — run_from_bin
        # must never touch the HF model. Kept in source as """ ... """ block
        # so we can revive it if the combined-bin path ever fails.
        """
        vt = hf_model.model.vision_tower
        bpe = self.bytes_per_element
        H = self.VIS_H
        MLP = self.VIS_MLP
        HD = self.VIS_HEAD_DIM
        NH = self.VIS_HEADS

        # 2D-RoPE → 1D-RoPE permutation (per-head, applied to head_dim axis).
        # Permuted layout: [bx[0:16], by[0:16], bx[16:32], by[16:32]] per head.
        # With this perm applied to q/k/v output rows AND cos/sin tables,
        # vision's 2D RoPE becomes a single 1D RoPE call. Standalone test
        # at test_rope_split64.py validates the math up to M=30240.
        # The perm is its own inverse (involution).
        self.VIS_ROPE_PERM = torch.cat([
            torch.arange(0, 16),
            torch.arange(32, 48),
            torch.arange(16, 32),
            torch.arange(48, 64),
        ])

        def _perm_qkv_output_rows(weight: torch.Tensor) -> torch.Tensor:
            in_features = weight.shape[1]
            return weight.reshape(NH, HD, in_features)[:, self.VIS_ROPE_PERM, :].reshape(NH * HD, in_features).contiguous()

        def _perm_o_input_cols(weight: torch.Tensor) -> torch.Tensor:
            out_features = weight.shape[0]
            return weight.reshape(out_features, NH, HD)[:, :, self.VIS_ROPE_PERM].reshape(out_features, NH * HD).contiguous()

        print(f"\n[Vision] Uploading vision encoder weights to DRAM "
              f"({VISION_QUANT_PRECISION.upper()} block=64 + BF16 norms) ...")

        def _upload_q4_64_batch(tensors: list[torch.Tensor]) -> list[dict]:
            '''Quantize a list of [N, K] bf16 weights in parallel via the
            canonical codec wrapper, then DMA-upload each (scale + data)
            serially. Returns a parallel list of
            {'data': addr, 'scale': addr, 'shape': (N, K)} dicts.

            Codebook is selected by VISION_QUANT_PRECISION ("if4" by default,
            "fp4" as a byte-compatible fallback).'''
            for t in tensors:
                assert t.dim() == 2, f"expected 2D weight, got {t.shape}"
                assert t.shape[1] % 64 == 0, f"K={t.shape[1]} not divisible by 64"
            tensors_c = [t.contiguous() for t in tensors]
            quant_results = _parallel_quantize(VISION_QUANT_PRECISION, tensors_c)
            infos: list[dict] = []
            for w, (data_bytes, scale_bytes) in zip(tensors_c, quant_results):
                N, K = w.shape
                scale_addr = self.allocate_tensor_dram(len(scale_bytes))
                self.dma_write(DMA_DEVICE_H2C, scale_addr, scale_bytes, len(scale_bytes))
                data_addr = self.allocate_tensor_dram(len(data_bytes))
                self.dma_write(DMA_DEVICE_H2C, data_addr, data_bytes, len(data_bytes))
                infos.append({"data": data_addr, "scale": scale_addr, "shape": (N, K)})
            return infos

        # Per-layer weights: pack all layers contiguously
        layer_weight_addrs = []
        for li in range(self.VIS_LAYERS):
            L = vt.encoder.layers[li]
            addrs = {}
            # Norms (BF16, small)
            for norm_name in ["input_layernorm", "post_attention_layernorm",
                              "pre_feedforward_layernorm", "post_feedforward_layernorm"]:
                w = getattr(L, norm_name).weight.detach().cpu().to(torch.bfloat16)
                addr = self.allocate_tensor_dram(w.numel() * bpe)
                self.dma_to_accelerator_memory(addr, w)
                addrs[norm_name] = addr

            # Q/K norms (BF16). Gamma is shape (HD,) and applied per head_dim
            # element — permute to line up with the permuted Q/K dims.
            for qk_norm in ["q_norm", "k_norm"]:
                w = getattr(L.self_attn, qk_norm).weight.detach().cpu().to(torch.bfloat16)
                w = w[self.VIS_ROPE_PERM].contiguous()
                addr = self.allocate_tensor_dram(w.numel() * bpe)
                self.dma_to_accelerator_memory(addr, w)
                addrs[qk_norm] = addr

            # All 7 IF4 projections (Q/K/V/O attn + gate/up/down MLP) batched:
            # parallel-quantize all 7 tensors, then serial DMA upload.
            # q/k/v output rows and o_proj input cols are pre-permuted within
            # each head — see VIS_ROPE_PERM.
            q4_names = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]
            q4_tensors = []
            for proj_name in ["q_proj", "k_proj", "v_proj"]:
                w = getattr(L.self_attn, proj_name).linear.weight.detach().cpu().to(torch.bfloat16)
                q4_tensors.append(_perm_qkv_output_rows(w))
            w_o = L.self_attn.o_proj.linear.weight.detach().cpu().to(torch.bfloat16)
            q4_tensors.append(_perm_o_input_cols(w_o))
            for mlp_name in ["gate_proj", "up_proj", "down_proj"]:
                q4_tensors.append(getattr(L.mlp, mlp_name).linear.weight
                                   .detach().cpu().to(torch.bfloat16))
            for name, info in zip(q4_names, _upload_q4_64_batch(q4_tensors)):
                addrs[name] = info

            layer_weight_addrs.append(addrs)

        self._vis_weight_addrs = layer_weight_addrs

        # Patch embedder + embed_vision projector are both IF4 (block=64).
        # Patch embedder input_proj: Linear(768, 768).
        # embed_vision.embedding_projection: Linear(768, 1536).
        # Setup non-quant pieces first (position table on host, BF16 ones-gamma
        # for embed_pre_projection_norm), then parallel-quantize both
        # projections together.
        pe = vt.patch_embedder
        ev = hf_model.model.embed_vision

        # Position embedding table [2, 10240, 768] kept on host — the lookup
        # is a gather and only runs once per image, not a hot path.
        self._vis_pos_embed_table = pe.position_embedding_table.detach().cpu().to(torch.bfloat16)

        # embedding_pre_projection_norm is a Gemma4RMSNorm(with_scale=False),
        # i.e. pure x * rsqrt(mean(x^2)+eps). Upload an all-ones gamma so the
        # standard rms_norm_core_dram (which multiplies by gamma) is equivalent
        # to "no scale".
        self.VIS_EMBED_NORM_HAS_SCALE = False
        ones_gamma = torch.ones(H, dtype=torch.bfloat16)
        self.VIS_EMBED_NORM_GAMMA = self.allocate_tensor_dram(ones_gamma.numel() * bpe)
        self.dma_to_accelerator_memory(self.VIS_EMBED_NORM_GAMMA, ones_gamma)

        w_patch_proj = pe.input_proj.weight.detach().cpu().to(torch.bfloat16)
        w_embed_proj = ev.embedding_projection.weight.detach().cpu().to(torch.bfloat16)
        # Parallel-quantize both projections; serial DMA upload in patch / embed order.
        patch_info, embed_info = _upload_q4_64_batch([w_patch_proj, w_embed_proj])
        self.VIS_PATCH_PROJ_INFO = patch_info
        self.VIS_EMBED_PROJ_INFO = embed_info
        self.VIS_TEXT_H = w_embed_proj.shape[0]  # 1536 for Gemma4 E2B

        # Config values needed at inference time (stored to avoid passing
        # hf_model through every vision call).
        self.VIS_POOL_K = vt.config.pooling_kernel_size  # typically 3

        # Extract clip ranges from Gemma4ClippableLinear wrappers
        clip_ranges = []
        for li in range(self.VIS_LAYERS):
            L = vt.encoder.layers[li]
            cr = {}
            for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                proj = getattr(L.self_attn, proj_name)
                if proj.use_clipped_linears:
                    cr[proj_name] = {
                        "input": (proj.input_min.item(), proj.input_max.item()),
                        "output": (proj.output_min.item(), proj.output_max.item()),
                    }
                else:
                    cr[proj_name] = {
                        "input": (float("-inf"), float("inf")),
                        "output": (float("-inf"), float("inf")),
                    }
            for mlp_name in ["gate_proj", "up_proj", "down_proj"]:
                proj = getattr(L.mlp, mlp_name)
                if proj.use_clipped_linears:
                    cr[mlp_name] = {
                        "input": (proj.input_min.item(), proj.input_max.item()),
                        "output": (proj.output_min.item(), proj.output_max.item()),
                    }
                else:
                    cr[mlp_name] = {
                        "input": (float("-inf"), float("inf")),
                        "output": (float("-inf"), float("inf")),
                    }
            clip_ranges.append(cr)
        self._vis_clip_ranges = clip_ranges
        print(f"  Clip ranges loaded for {len(clip_ranges)} layers "
              f"(q_proj input: [{clip_ranges[0]['q_proj']['input'][0]:.2f}, "
              f"{clip_ranges[0]['q_proj']['input'][1]:.2f}])")

        # Constant ones-gamma for V-norm (Gemma4 V-norm has no learnable scale).
        # rms_norm_core_dram requires a gamma DRAM addr; uploading ones(HD)
        # gives the equivalent of "no scale" since (x/rms(x)) * 1 = x/rms(x).
        v_ones = torch.ones(HD, dtype=torch.bfloat16)
        self.VIS_V_NORM_ONES_GAMMA = self.allocate_tensor_dram(HD * bpe)
        self.dma_to_accelerator_memory(self.VIS_V_NORM_ONES_GAMMA, v_ones)

        # Save vision weight end address — program DRAM must start AFTER this
        self._vis_weight_end = self.get_tensor_dram_addr()
        print(f"  Vision weights uploaded. Tensor DRAM usage: {self.get_tensor_dram_usage()/(1024*1024):.1f} MB")
        """
        # End of LEGACY HF-fallback path (commented out).

    def vision_tensor_init(self, num_patches: int, *, program_base: int | None = None) -> None:
        """Allocate DRAM for vision encoder intermediate tensors.

        Reuses the LM tensor DRAM region since vision and LM don't run simultaneously.

        program_base: if None (compare-script default), vision programs go to
            DRAM_INSTRUCTION_ADDR (0xD0000000), which is inside the LM weights
            region. That's safe when no LM code runs after vision. The test
            script passes an explicit safe address (above LM tensors and clear
            of LM weights) so LM decode can run correctly post-vision.
        """
        bpe = self.bytes_per_element
        H = self.VIS_H
        HD = self.VIS_HEAD_DIM
        NH = self.VIS_HEADS
        MLP = self.VIS_MLP
        S = num_patches  # sequence length for vision

        # Reset tensor allocator — vision tensors reuse LM tensor DRAM region
        self.reset_tensor_dram_addr()
        # Vision ISA sub-region per the fixed Gemma4 DRAM layout (full 4 GB) —
        # see notes/notes_gemma4_e2b_vision.md "Master layout table".
        #   Audio  ISA: 0x98000000 – 0xa0000000 (128 MB)
        #   Vision ISA: 0xa0000000 – 0xc0000000 (512 MB)   ← we use this
        #   Prefill ISA: 0xc0000000 – 0xd0000000 (256 MB)
        #   Decoder ISA: 0xd0000000 – 0xd8000000 (128 MB)
        # 512 MB easily fits G=16 one-shot (~400 MB ISA).
        VISION_ISA_BASE = 0xa0000000
        if program_base is None:
            self._next_program_dram_addr = VISION_ISA_BASE
            self._program_dram_base = VISION_ISA_BASE
        else:
            self._next_program_dram_addr = program_base
            self._program_dram_base = program_base
        print(f"\n[Vision] Allocating vision tensor DRAM for {S} patches ...")

        # Layer I/O (double-buffered)
        self.VIS_IO_A = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_IO_B = self.allocate_tensor_dram(S * H * bpe)

        # Intermediates
        self.VIS_NORM_OUT = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_Q_DRAM = self.allocate_tensor_dram(S * NH * HD * bpe)
        self.VIS_K_DRAM = self.allocate_tensor_dram(S * NH * HD * bpe)
        self.VIS_V_DRAM = self.allocate_tensor_dram(S * NH * HD * bpe)
        self.VIS_Q_NORM = self.allocate_tensor_dram(S * NH * HD * bpe)
        self.VIS_K_NORM = self.allocate_tensor_dram(S * NH * HD * bpe)
        self.VIS_ATTN_OUT = self.allocate_tensor_dram(S * H * bpe)  # after o_proj
        self.VIS_POST_ATTN_NORM = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_POST_ATTN_RES = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_PRE_FFN_NORM = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_MLP_GATE = self.allocate_tensor_dram(S * MLP * bpe)
        self.VIS_MLP_UP = self.allocate_tensor_dram(S * MLP * bpe)
        self.VIS_MLP_MULT = self.allocate_tensor_dram(S * MLP * bpe)
        self.VIS_MLP_DOWN = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_POST_FFN_NORM = self.allocate_tensor_dram(S * H * bpe)

        # Flash attention buffers (per-head, process heads sequentially)
        aligned_S = ((S + 63) // 64) * 64
        self.VIS_FLASH_Q = self.allocate_tensor_dram(aligned_S * HD * bpe)
        self.VIS_FLASH_K = self.allocate_tensor_dram(aligned_S * HD * bpe)
        self.VIS_FLASH_V = self.allocate_tensor_dram(aligned_S * HD * bpe)
        self.VIS_FLASH_OUT = self.allocate_tensor_dram(aligned_S * HD * bpe)
        # Allocate BIAS BEFORE SCRATCH and add a guard padding around BIAS so any
        # adjacent-buffer corruption is impossible. PBI flash needs scratch of
        # (HD + aligned_S) * aligned_S * BF16 bytes (V^T HD*S*BF16 + partial-softmax
        # S*S*BF16). The older formula `max(HD, 512) * S * 2 + HD * S * 2` was
        # undersized by ~10MB at S=2560 and silently corrupted whatever lived
        # next to SCRATCH. We now (a) size SCRATCH correctly, (b) put BIAS
        # BEFORE SCRATCH, (c) pad both sides of BIAS to be safe.
        # See notes/notes_gemma4_e2b_vision.md "Known issue" + test_pbi_flash_bug_minimal.py.
        BIAS_GUARD_BYTES = 4 * 1024 * 1024  # 4 MB guard pad
        self._vis_bias_guard_pre  = self.allocate_tensor_dram(BIAS_GUARD_BYTES)
        self.VIS_FLASH_BIAS = self.allocate_tensor_dram(aligned_S * aligned_S * bpe)
        self._vis_bias_guard_post = self.allocate_tensor_dram(BIAS_GUARD_BYTES)
        # v3-uac: unified_attention_core fold layout — Vᵀ + scores + scaled_q in
        # one SCRATCH = aligned_S² + 2·HD·aligned_S BF16 elements. MUST match
        # test.py's vision_tensor_init exactly (same size + alloc order) so the
        # baked bin's addresses land correctly. run_from_bin loads; it never re-emits.
        self.VIS_FLASH_SCRATCH = self.allocate_tensor_dram(
            (aligned_S * aligned_S + 2 * HD * aligned_S) * 2)
        # VIS_ATTN_P is unused by unified_attention_core (scores fold into SCRATCH),
        # but keep the allocation so the alloc ORDER — and thus every subsequent baked
        # address — stays identical to test.py's vision_tensor_init.
        self.VIS_ATTN_P = self.allocate_tensor_dram(aligned_S * aligned_S * 2)

        # 64×64 BF16 identity matrix for FPGA clamp passes (used by
        # _emit_clamp_dram_to_dram). Pre-allocated once so input clipping
        # can run on FPGA without dragging data through host. 8 KB.
        self.VIS_IDENTITY_64 = self.allocate_tensor_dram(64 * 64 * bpe)
        self.dma_to_accelerator_memory(self.VIS_IDENTITY_64, torch.eye(64, dtype=torch.bfloat16))
        # Scratch buffers for FPGA input clipping (replaces host _host_clip_dram).
        # SCRATCH_H is sized (S, H) bf16 — used for q/k/v/o input clips and
        # gate/up input clips. SCRATCH_MLP is sized (S, MLP) bf16 — used for
        # gate output (post-GELU) clip and down_proj input clip.
        self.VIS_INPUT_CLIP_H_SCRATCH   = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_INPUT_CLIP_MLP_SCRATCH = self.allocate_tensor_dram(S * MLP * bpe)

        # #4: head-major Q/K/V/OUT buffers — (NH, aligned_S, HD) bf16, contiguous
        # per head. Populated by FPGA transposes after RoPE; the per-head
        # attention loop reads from offsets within these buffers (no host
        # gather, no per-head DMA upload). Pre-zero so padding rows
        # [S, aligned_S) stay zero across runs.
        NH_alloc = self.VIS_HEADS
        hm_bytes = NH_alloc * aligned_S * HD * bpe
        self.VIS_FLASH_Q_HM   = self.allocate_tensor_dram(hm_bytes)
        self.VIS_FLASH_K_HM   = self.allocate_tensor_dram(hm_bytes)
        self.VIS_FLASH_V_HM   = self.allocate_tensor_dram(hm_bytes)
        self.VIS_FLASH_OUT_HM = self.allocate_tensor_dram(hm_bytes)
        zeros_hm = torch.zeros(NH_alloc * aligned_S * HD, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.VIS_FLASH_Q_HM, zeros_hm)
        self.dma_to_accelerator_memory(self.VIS_FLASH_K_HM, zeros_hm)
        self.dma_to_accelerator_memory(self.VIS_FLASH_V_HM, zeros_hm)
        self.dma_to_accelerator_memory(self.VIS_FLASH_OUT_HM, zeros_hm)

        # Bidirectional bias: all zeros, alignment padding columns masked.
        # NOTE: real image padding patches (position_ids == -1) still need to
        # be masked — call set_vision_attention_bias(padding_positions) before
        # running attention.
        bias = torch.zeros(aligned_S, aligned_S, dtype=torch.bfloat16)
        bias[:, S:] = float("-inf")  # alignment padding columns
        self.dma_to_accelerator_memory(self.VIS_FLASH_BIAS, bias)

        # RoPE tables (legacy, kept for back-compat — unused after #3)
        self.VIS_ROPE_COS = self.allocate_tensor_dram(S * HD * bpe)
        self.VIS_ROPE_SIN = self.allocate_tensor_dram(S * HD * bpe)
        # Three padded RoPE tables for FPGA 2D RoPE (qwen3.5 split-64 trick).
        # RoPE opt A: ONE row per PATCH (S rows), NOT per (patch,head) — 2D-RoPE
        # is head-independent, so _emit_vision_rope_2d reads each patch row NH×.
        # RoPE opt B: rows are 32-wide (HD//2), NOT 64-wide (only half_rot=32 is
        # read). Must match gemma4_e2b_test.py exactly (same alloc size/order) so
        # baked vision addresses line up.
        self.VIS_ROPE_COS_PAD_TILED     = self.allocate_tensor_dram(S * (HD // 2) * bpe)
        self.VIS_ROPE_NEG_SIN_PAD_TILED = self.allocate_tensor_dram(S * (HD // 2) * bpe)
        self.VIS_ROPE_SIN_HI_PAD_TILED  = self.allocate_tensor_dram(S * (HD // 2) * bpe)

        # Identity matrix for flash attention
        self.VIS_IDENTITY = self.allocate_tensor_dram(HD * HD * bpe)
        self.dma_to_accelerator_memory(self.VIS_IDENTITY,
                                        torch.eye(HD, dtype=torch.bfloat16))

        # Embed-vision (pooler tail) scratch. N_soft (post-pooler, post-mask)
        # is image-dependent but bounded above by S / pooling_kernel_size^2; we
        # size at S to be safe and allocate in BF16.
        text_h = getattr(self, "VIS_TEXT_H", 1536)
        self.VIS_EMBED_POOL = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_EMBED_NORMED = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_EMBED_OUT = self.allocate_tensor_dram(S * text_h * bpe)

        self._vis_num_patches = S
        self._vis_aligned_S = aligned_S
        self._vis_padding_mask = None  # [S] bool, set by set_vision_attention_bias
        total = self.get_tensor_dram_usage()
        print(f"  Vision tensor DRAM allocated. Total usage: {total} bytes")

    def set_vision_attention_bias(self, padding_positions: torch.Tensor) -> None:
        """Rebuild VIS_FLASH_BIAS so attention masks BOTH real padding patches
        (position_ids == -1) and the alignment padding at the end.

        padding_positions: [B, S] or [S] bool — True where patch is padding.
        Must be called after vision_tensor_init and before running attention.
        """
        S = self._vis_num_patches
        aligned_S = self._vis_aligned_S
        if padding_positions.dim() == 2:
            pad = padding_positions[0].cpu().bool()
        else:
            pad = padding_positions.cpu().bool()
        assert pad.shape[0] == S, f"padding mask has {pad.shape[0]} entries, expected {S}"
        self._vis_padding_mask = pad
        n_pad = int(pad.sum().item())
        n_valid = S - n_pad

        # Mask every padding column: real image padding + alignment padding.
        col_mask = torch.zeros(aligned_S, dtype=torch.bool)
        col_mask[:S] = pad
        col_mask[S:] = True
        bias = torch.zeros(aligned_S, aligned_S, dtype=torch.bfloat16)
        bias[:, col_mask] = float("-inf")
        self.dma_to_accelerator_memory(self.VIS_FLASH_BIAS, bias)

    def _run_eltwise_add_chunked(self, a_addr: int, b_addr: int, out_addr: int, num_elements: int) -> None:
        """Element-wise add two DRAM tensors, one SRAM-sized chunk per FPGA program."""
        CHUNK = 65536  # 128KB per buffer, safe for SRAM gap 0x10000-0x90000
        bpe = self.bytes_per_element
        for off in range(0, num_elements, CHUNK):
            n = min(CHUNK, num_elements - off)
            def _fn(a=a_addr + off * bpe, b=b_addr + off * bpe, o=out_addr + off * bpe, sz=n):
                self.accelerator_memory_to_sram(a, 0x10000, sz)
                self.accelerator_memory_to_sram(b, 0x90000, sz)
                self.eltwise_add_core(0x10000, 0x90000, 0x10000, sz)
                self.sram_to_accelerator_memory(0x10000, o, sz)
            self._compile_and_run_single("eltwise_add_chunk", _fn)

    def _run_eltwise_mul_chunked(self, a_addr: int, b_addr: int, out_addr: int, num_elements: int) -> None:
        """Element-wise multiply two DRAM tensors, one SRAM-sized chunk per FPGA program."""
        CHUNK = 65536  # 128KB per buffer, safe for SRAM
        bpe = self.bytes_per_element
        for off in range(0, num_elements, CHUNK):
            n = min(CHUNK, num_elements - off)
            def _fn(a=a_addr + off * bpe, b=b_addr + off * bpe, o=out_addr + off * bpe, sz=n):
                self.accelerator_memory_to_sram(a, 0x10000, sz)
                self.accelerator_memory_to_sram(b, 0x90000, sz)
                self.eltwise_mul_core(0x10000, 0x90000, 0x10000, sz)
                self.sram_to_accelerator_memory(0x10000, o, sz)
            self._compile_and_run_single("eltwise_mul_chunk", _fn)

    def _compile_and_run_single(self, label: str, compile_fn,
                                 *, cache: _ProgramCache | None = None,
                                 cache_key: str | None = None) -> None:
        """Compile a single operation, run it. Reuses same program DRAM slot.

        If ``cache`` and ``cache_key`` are provided, look up the cached ISA
        bytes for that key. On hit, DMA-write the cached bytes to program
        DRAM and execute (skipping the Python ISA emission entirely). On
        miss, do the normal capture, store the resulting bytes in the cache,
        then execute as before.

        One-shot mode (``self._oneshot_mode = True``): just emit ISA into
        the caller's already-open ``start_capture`` buffer. No execute, no
        per-program cache, no program-DRAM reset. Used by
        ``compile_and_run_vision_encoder_oneshot`` to capture all 16 layers
        as a single program.
        """
        if getattr(self, "_oneshot_mode", False):
            # Reset the PBI-pointer and ISA-register allocators to their fixed base
            # before each sub-op: per-op PBI cores (matmat_mul_core_pbi) alloc 13
            # inst-pointers + 4 isa-regs and never release them, so without this the
            # counters climb across the many sub-ops in one program and exhaust the
            # 15-pointer pool, corrupting later matmuls (cos→0.9). Must stay
            # byte-identical to gemma4_e2b_test.py's copy.
            self.reset_inst_ptr_counter()
            self._isa_reg_counter = self._isa_reg_base
            compile_fn()
            return

        import builtins
        _orig_print = builtins.print
        builtins.print = lambda *a, **kw: None

        # Reset program DRAM to reuse space (each op is independent)
        self.reset_program_dram_addr()

        cached_bytes = cache.get(cache_key) if (cache is not None and cache_key) else None
        if cached_bytes is not None:
            # Cache hit: DMA the cached instruction stream and execute.
            prog_addr = self.get_program_dram_addr()
            self.dma_write(DMA_DEVICE_H2C, prog_addr, cached_bytes, len(cached_bytes))
            self.allocate_program_dram(len(cached_bytes))
            builtins.print = _orig_print
            self.start_execute_from_dram(prog_addr)
            self.wait_queue(120.0)
            return

        self.start_capture()
        compile_fn()
        self.stop_capture()
        self.generate_instruction_halt()
        if cache is not None and cache_key:
            cache.put(cache_key,
                      bytes(b"".join(inst.get_bytes() for inst in self.capture_buffer)))
        prog_addr = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(prog_addr)
        sz = self.get_capture_instruction_size_bytes()
        self.allocate_program_dram(sz)
        self.clear_capture_buffer()
        builtins.print = _orig_print

        self.start_execute_from_dram(prog_addr)
        self.wait_queue(120.0)

    def _host_clip_dram(self, addr: int, shape: tuple, clip_min: float, clip_max: float) -> None:
        """Read tensor from FPGA DRAM, clip on host, write back."""
        t = self.dma_from_accelerator_memory(addr, shape).cpu()
        t = t.clamp(min=clip_min, max=clip_max)
        self.dma_to_accelerator_memory(addr, t)

    def _load_or_build_vision_rope_pads(self, hf_model, pixel_position_ids, num_patches):
        """Bin-only stub: in run_from_bin we NEVER touch the HF model. Vision
        rope_pads are embedded in programs.bin (offsets in the
        manifest) by compile_instruction_bin on the build host. The caller
        in _run_vision_encoder_fpga reads them directly from the bin file
        — this method is only invoked if those offsets are missing from the
        manifest, which signals a stale bin produced before the embedding
        was added. Surface a clear "rebuild the bin" error instead of
        falling back to HF.
        """
        raise RuntimeError(
            "run_from_bin: vision rope_pads are missing from the unified "
            "instruction bin's manifest. The bin you are running was built "
            "before vision rope_pads were embedded — rebuild it on the build "
            "host with `python gemma4_e2b_test.py --vision-enable` and ship "
            "the new programs.{bin,json}. (Never touching the HF "
            "model is a hard contract for run_from_bin.)")

    def _emit_qkv_transpose_to_hm(self, src_dram: int, dst_dram: int,
                                    S: int, dst_aligned_S: int) -> None:
        """Emit FPGA ISA: transpose vision Q/K/V (S*NH, HD) → (NH, aligned_S, HD).

        Source rows are interleaved (row i = (s, h) where s=i//NH, h=i%NH).
        Destination is head-major; only the first S rows of each head are
        written (rows [S, aligned_S) stay at their pre-zeroed value).

        Validated standalone in test_vision_qkv_transpose.py (SNR=inf at S=2520).
        """
        NH = self.VIS_HEADS
        HD = self.VIS_HEAD_DIM
        BF16 = 2
        row_bytes = HD * BF16
        src_jump_bytes = NH * row_bytes
        dst_head_bytes = dst_aligned_S * row_bytes
        chunk_S = max(1, URAM_NEAR_FULL_ELEMENTS // HD)
        SA_BUF = 0x00000

        for h in range(NH):
            src_base = src_dram + h * row_bytes
            dst_base = dst_dram + h * dst_head_bytes
            s = 0
            while s < S:
                this_chunk = min(chunk_S, S - s)
                self.accelerator_memory_to_sram(
                    accelerator_dram_address=src_base + s * src_jump_bytes,
                    sram_address=SA_BUF,
                    element_size=this_chunk * HD,
                    stride_bytes_per_chunk=row_bytes,
                    stride_jump_bytes=src_jump_bytes,
                )
                self.sram_to_accelerator_memory(
                    sram_address=SA_BUF,
                    accelerator_dram_address=dst_base + s * row_bytes,
                    element_size=this_chunk * HD,
                )
                s += this_chunk

    def _emit_attn_out_transpose_to_interleaved(self, src_dram: int, dst_dram: int,
                                                  S: int, src_aligned_S: int) -> None:
        """Emit FPGA ISA: inverse transpose attn output (NH, aligned_S, HD) → (S, NH*HD).

        Reads only the first S rows of each head (the rest is padding, ignored).
        Writes interleaved to dst (s, h, :) for each (s, h) pair.

        Validated standalone in test_vision_qkv_transpose.py (SNR=inf at S=2520).
        """
        NH = self.VIS_HEADS
        HD = self.VIS_HEAD_DIM
        BF16 = 2
        row_bytes = HD * BF16
        dst_jump_bytes = NH * row_bytes
        src_head_bytes = src_aligned_S * row_bytes
        chunk_S = max(1, URAM_NEAR_FULL_ELEMENTS // HD)
        SA_BUF = 0x00000

        for h in range(NH):
            src_base = src_dram + h * src_head_bytes
            dst_base = dst_dram + h * row_bytes
            s = 0
            while s < S:
                this_chunk = min(chunk_S, S - s)
                self.accelerator_memory_to_sram(
                    accelerator_dram_address=src_base + s * row_bytes,
                    sram_address=SA_BUF,
                    element_size=this_chunk * HD,
                )
                self.sram_to_accelerator_memory(
                    sram_address=SA_BUF,
                    accelerator_dram_address=dst_base + s * dst_jump_bytes,
                    element_size=this_chunk * HD,
                    stride_bytes_per_chunk=row_bytes,
                    stride_jump_bytes=dst_jump_bytes,
                )
                s += this_chunk

    def _prime_M(self, M: int) -> int:
        """Emit ADD_SET gpr_seq_len <- M and return the register index, for use as
        ``gpr_M_reg=self._prime_M(M)`` on matmat_mul_core. Folds compile-time M
        tiling into one runtime ISA loop body (bin shrink), bit-exact to legacy.
        Must stay byte-identical to gemma4_e2b_test.py's copy."""
        self.generate_instruction_add_set(self.gpr_seq_len, M)
        return self.gpr_seq_len

    def _rms_norm_dram_pbi(self, M: int, N: int, A_DRAM_ADDR: int,
                            OUTPUT_DRAM_ADDR: int, GAMMA_DRAM_ADDR: int) -> None:
        """rms_norm_core_dram via the runtime ISA loop (gpr_M_reg) to shrink the
        captured bin. Bit-exact to legacy (same RMS/gamma math, only the
        M-direction iteration folded into one ISA loop body). gpr_seq_len is
        LM-only, free in the vision/audio program. Must stay byte-identical to
        gemma4_e2b_test.py's copy."""
        self.generate_instruction_add_set(self.gpr_seq_len, M)
        self.rms_norm_core_dram(
            M=M, N=N, A_DRAM_ADDR=A_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
            GAMMA_DRAM_ADDR=GAMMA_DRAM_ADDR,
            gpr_M_reg=self.gpr_seq_len)

    def _emit_vision_rope_2d(self, src_dram: int, out_dram: int,
                              cos_pad_dram: int,
                              neg_sin_pad_dram: int,
                              sin_hi_pad_dram: int,
                              M: int) -> None:
        """Apply vision 2D RoPE to M consecutive rows of HD=64 elements.

        Adapts qwen3.5's split-64 RoPE workaround for the SRAM 128-byte
        alignment problem. cos_pad/neg_sin_pad/sin_hi_pad rows must be
        pre-computed with permuted half-values in cols [0:32] and zeros
        in cols [32:64]; q/k/v weights pre-permuted so this 1D RoPE
        produces correct 2D RoPE result (see VIS_ROPE_PERM).

        The standalone test in test_rope_split64.py validates this primitive
        in isolation up to M=30240 with SNR=51 dB.
        """
        rot_dim = 64
        half_rot = rot_dim // 2
        BF16 = 2
        row_bytes = rot_dim * BF16
        half_bytes = half_rot * BF16

        SA_X_LO    = 0x40000
        SA_X_HI    = 0x40080
        SA_OUT_LO  = 0x40100
        SA_OUT_HI  = 0x40180
        SA_TMP_A   = 0x40200
        SB_COS     = 0x80000
        SB_NEG_SIN = 0x80080
        SB_SIN_HI  = 0x80100
        SB_TMP_B   = 0x80180

        # RoPE opt A: cos/neg_sin/sin_hi hold ONE row per PATCH (S rows), not per
        # (patch,head) — 2D-RoPE is head-independent. Loop patch-outer / head-inner:
        # load a patch's coeffs ONCE (reused by its NH consecutive Q/K rows), so the
        # tables are NH× smaller and the rope DMA reads drop NH×. Q/K rows are still
        # the flat M=S*NH sequence (t_reg); at rope time Q/K are patch-major/head-minor.
        rope_reads = [(cos_pad_dram,     SB_COS,     half_rot),
                      (neg_sin_pad_dram, SB_NEG_SIN, half_rot),
                      (sin_hi_pad_dram,  SB_SIN_HI,  half_rot)]
        src_reads  = [(src_dram,             SA_X_LO, half_rot),
                      (src_dram + half_bytes, SA_X_HI, half_rot)]
        writes     = [(SA_OUT_LO, out_dram,             half_rot),
                      (SA_OUT_HI, out_dram + half_bytes, half_rot)]

        # Per-row DRAM addresses computed at runtime into TMP_REG, consumed by the
        # memcpy via general_reg_src (REG_REWRITE). The result MUST land in TMP_REG
        # (== general_reg_src): add_imm(off_reg, base, TMP_REG) computes
        # regfile[TMP_REG] = regfile[off_reg] + base. patch_reg drives the per-PATCH
        # rope address; t_reg the flat per-ROW Q/K address. gpr_bucket_idx is free
        # here (the flash sets it per-head AFTER rope). Stays identical to
        # gemma4_e2b_test.py.
        t_reg     = self.gpr_seq_len
        off_reg   = self.gpr_q_seq_len
        patch_reg = self.gpr_bucket_idx
        S = M // self.VIS_HEADS
        self.generate_instruction_add_set(t_reg, 0)
        self.generate_instruction_add_set(patch_reg, 0)
        self.loop_start(loop_cnt=S)                       # OUTER: patches
        # Load this patch's rope coeffs ONCE (shared by its VIS_HEADS heads).
        # Opt B: rope rows are 32-wide → stride is half_bytes (64), not row_bytes.
        self.generate_instruction_reg_mul_imm(off_reg, patch_reg, ue_35bit_addr_shifter(half_bytes))
        for base, sram, elems in rope_reads:
            self.generate_instruction_add_imm(off_reg, ue_35bit_addr_shifter(base), self.TMP_REG)
            self.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=sram,
                                            element_size=elems, general_reg_src=self.TMP_REG)
        self.loop_start(loop_cnt=self.VIS_HEADS)          # INNER: heads of this patch
        self.generate_instruction_reg_mul_imm(off_reg, t_reg, ue_35bit_addr_shifter(row_bytes))
        for base, sram, elems in src_reads:
            self.generate_instruction_add_imm(off_reg, ue_35bit_addr_shifter(base), self.TMP_REG)
            self.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=sram,
                                            element_size=elems, general_reg_src=self.TMP_REG)
        self.eltwise_mul_core(SA_X_LO, SB_COS,     SB_TMP_B,  half_rot)
        self.eltwise_mul_core(SA_X_HI, SB_NEG_SIN, SA_TMP_A,  half_rot)
        self.eltwise_add_core(SA_TMP_A, SB_TMP_B,  SA_OUT_LO, half_rot)
        self.eltwise_mul_core(SA_X_HI, SB_COS,    SB_TMP_B,  half_rot)
        self.eltwise_mul_core(SA_X_LO, SB_SIN_HI, SA_TMP_A,  half_rot)
        self.eltwise_add_core(SA_TMP_A, SB_TMP_B, SA_OUT_HI, half_rot)
        for sram, base, elems in writes:
            self.generate_instruction_add_imm(off_reg, ue_35bit_addr_shifter(base), self.TMP_REG)
            self.sram_to_accelerator_memory(sram_address=sram, accelerator_dram_address=0,
                                            element_size=elems, general_reg_src=self.TMP_REG)
        self.generate_instruction_add_inc(t_reg)
        self.loop_end()                                   # end INNER (heads)
        self.generate_instruction_add_inc(patch_reg)
        self.loop_end()                                   # end OUTER (patches)

    def _emit_clamp_dram_to_dram(self, src_dram: int, dst_dram: int,
                                   num_elements: int,
                                   clamp_min: float, clamp_max: float,
                                   *, identity_addr: int | None = None,
                                   use_pbi: bool = True,) -> None:
        """FPGA DRAM→DRAM clamp via matmul-with-identity-weight + fused clamp.

        HW only routes through LALU (which has CLAMP) during DOT_PRODUCT,
        not eltwise/broadcast. Trick: matmul A=src (M=N/64, K=64) by a
        pre-stored bf16 identity (K=64, N=64) gives output = src (passthrough),
        then fused clamp applies. Validated standalone in test_clamp_dram.py
        (SNR=inf at all sizes up to 1.9M elements).

        Replaces _host_clip_dram (read DRAM → torch.clamp → write DRAM) with
        a pure FPGA op so vision/audio encoders can be captured one-shot.

        ``identity_addr``: 64x64 bf16 identity. Defaults to VIS_IDENTITY_64;
            audio passes AUD_IDENTITY_64 so vision init isn't required.
        """
        assert num_elements % 64 == 0, f"num_elements ({num_elements}) must be multiple of 64"
        if identity_addr is None:
            assert hasattr(self, "VIS_IDENTITY_64"), \
                "VIS_IDENTITY_64 not allocated; pass identity_addr=AUD_IDENTITY_64"
            identity_addr = self.VIS_IDENTITY_64
        saved_a = user_dma_core.LALU_CLAMP_RELU_A
        saved_b = user_dma_core.LALU_CLAMP_RELU_B
        try:
            user_dma_core.LALU_CLAMP_RELU_A = self.float_to_bf16(clamp_min)
            user_dma_core.LALU_CLAMP_RELU_B = self.float_to_bf16(clamp_max)
            _mm_kw = {"gpr_M_reg": self._prime_M(num_elements // 64)} if use_pbi else {}
            self.matmat_mul_core(
                M=num_elements // 64, K=64, N=64,
                A_DRAM_ADDR=src_dram,
                B_DRAM_ADDR=identity_addr,
                OUTPUT_DRAM_ADDR=dst_dram,
                is_B_quantized=False,
                clamp_enable=True,
                **_mm_kw,
            )
        finally:
            user_dma_core.LALU_CLAMP_RELU_A = saved_a
            user_dma_core.LALU_CLAMP_RELU_B = saved_b

    def _aud_clip_dram(self, addr: int, shape: tuple,
                        clamp_min: float, clamp_max: float) -> None:
        """Audio-encoder clip dispatcher (Phase 1).

        GEMMA4_AUDIO_FPGA_CLIP=1 (default): FPGA clamp via AUD_IDENTITY_64.
        GEMMA4_AUDIO_FPGA_CLIP=0          : legacy host clamp (read DRAM →
                                            torch.clamp → write DRAM).
        Both paths produce bit-identical results (validated in phase1 of
        compare_gemma4_e2b_audio.py); FPGA path removes a PCIe roundtrip.
        """
        rows, cols = shape
        num_elements = rows * cols
        if os.environ.get("GEMMA4_AUDIO_FPGA_CLIP", "1") == "1":
            assert hasattr(self, "AUD_IDENTITY_64"), \
                "AUD_IDENTITY_64 not allocated (audio_tensor_init)"
            label = f"aud_clip_{addr:08x}_{num_elements}"
            # use_pbi=False: PBI clamp hangs the FPGA in the audio conv sequence
            # (address/context-sensitive HW edge case; see test.py + the repro
            # AUDIO_pbi_clamp_hang_repro.py). Legacy clamp is bit-exact, never hangs.
            self._compile_and_run_single(label, lambda: self._emit_clamp_dram_to_dram(
                src_dram=addr, dst_dram=addr,
                num_elements=num_elements,
                clamp_min=clamp_min, clamp_max=clamp_max,
                identity_addr=self.AUD_IDENTITY_64, use_pbi=False))
        else:
            self._host_clip_dram(addr, shape, clamp_min, clamp_max)

    def _vis_mm_gpr(self, M: int):
        """gpr_M_reg for a vision/encoder matmul. PBI M-loop by default (folds the
        compile-time M tiling into one runtime ISA loop → large one-shot bin
        shrink); ``VIS_MATMUL_LEGACY=1`` forces the legacy static unroll. PBI
        dispatch is bit-exact to legacy in one-execute on HW v0x5f9f99db
        (validated: compare --mode all-saves, max|d|=0 / 16 layers). Emits
        ADD_SET(gpr_seq_len, M); call inside the active capture, once per matmul.
        Stays identical to gemma4_e2b_test.py."""
        if os.environ.get("VIS_MATMUL_LEGACY") == "1":
            return None
        if self.capture_buffer is None:
            return None
        return self._prime_M(M)

    def _aud_mm_gpr(self, M: int):
        """gpr_M_reg for an audio (conformer) matmul / norm — PBI M-loop by default,
        ``AUD_MATMUL_LEGACY=1`` forces legacy. Applied to the per-layer FFN /
        projection / conv matmuls + rms-norms and the subsample/embed matmuls. The
        subsample layer_norms (ln0/ln1) and the embed bias-matmul stay legacy (those
        PBI paths diverge from legacy in the one-shot bin; validated on apex.wav).
        Returns None when emit is suppressed (capture_buffer is None) so the runtime
        host-DMA-only subsample pass doesn't hit matmat_mul_core_pbi's loop_start.
        Stays identical to gemma4_e2b_test.py."""
        if os.environ.get("AUD_MATMUL_LEGACY") == "1":
            return None
        if self.capture_buffer is None:
            return None
        return self._prime_M(M)

    def _matmul_with_output_clamp(self, *, clamp_min: float, clamp_max: float, **mm_kwargs) -> None:
        """matmat_mul_core with arbitrary output-clamp bounds.

        Why: HW supports ``clamp(x, a, b) = max(a, min(x, b))`` via
        LALU_MODE.CLAMP, but matmat_mul_core reads bounds from module-level
        constants (LALU_CLAMP_RELU_A/B = 0.0, +inf — i.e. ReLU). To use
        custom bounds without modifying user_dma_core.py, monkey-patch those
        constants for the single call. Sequential ISA emission only —
        not thread-safe.
        """
        saved_a, saved_b = user_dma_core.LALU_CLAMP_RELU_A, user_dma_core.LALU_CLAMP_RELU_B
        try:
            user_dma_core.LALU_CLAMP_RELU_A = self.float_to_bf16(clamp_min)
            user_dma_core.LALU_CLAMP_RELU_B = self.float_to_bf16(clamp_max)
            return self.matmat_mul_core(clamp_enable=True, **mm_kwargs)
        finally:
            user_dma_core.LALU_CLAMP_RELU_A = saved_a
            user_dma_core.LALU_CLAMP_RELU_B = saved_b

    def compile_vision_layer(self, layer_idx: int) -> int:
        """Run vision layer part A: pre_norm + Q/K/V projections + Q/K norms.

        Each operation compiled and run separately to avoid FPGA instruction limits.
        Gemma4 vision uses Gemma4ClippableLinear which clips input/output — we apply
        clipping on host between FPGA matmul calls.

        Per-segment compile bytes are cached in self._vis_program_cache (when set
        by _run_vision_encoder_fpga) so subsequent runs skip the Python ISA
        emission and just DMA the cached bytes to program DRAM.
        """
        S = self._vis_num_patches
        H = self.VIS_H
        HD = self.VIS_HEAD_DIM
        NH = self.VIS_HEADS
        w = self._vis_weight_addrs[layer_idx]
        clips = self._vis_clip_ranges[layer_idx]
        cache = getattr(self, "_vis_program_cache", None)

        def _ck(label: str) -> str:
            return f"vis_L{layer_idx}_{label}"

        INPUT_DRAM = self.VIS_IO_A if layer_idx % 2 == 0 else self.VIS_IO_B

        # 1. Input RMSNorm
        self._compile_and_run_single("pre_norm", lambda: self._rms_norm_dram_pbi(
            M=S, N=H, A_DRAM_ADDR=INPUT_DRAM,
            OUTPUT_DRAM_ADDR=self.VIS_NORM_OUT,
            GAMMA_DRAM_ADDR=w["input_layernorm"]),
            cache=cache, cache_key=_ck("pre_norm"))

        # 2. Q projection (IF4): FPGA clamp pre_norm into scratch, then matmul
        #    reads from scratch with fused output clamp. (No host bouncing.)
        self._compile_and_run_single("clip_in_q", lambda: self._emit_clamp_dram_to_dram(
            src_dram=self.VIS_NORM_OUT, dst_dram=self.VIS_INPUT_CLIP_H_SCRATCH,
            num_elements=S * H,
            clamp_min=clips["q_proj"]["input"][0], clamp_max=clips["q_proj"]["input"][1]),
            cache=cache, cache_key=_ck("clip_in_q"))
        self._compile_and_run_single("q_proj", lambda: self._matmul_with_output_clamp(
            M=S, K=H, N=NH * HD,
            A_DRAM_ADDR=self.VIS_INPUT_CLIP_H_SCRATCH,
            B_DRAM_ADDR=w["q_proj"]["data"],
            OUTPUT_DRAM_ADDR=self.VIS_Q_DRAM,
            is_B_quantized=True,
            data_type=TYPE.IF4,
            SCALE_DRAM_ADDR=w["q_proj"]["scale"],
            clamp_min=clips["q_proj"]["output"][0],
            clamp_max=clips["q_proj"]["output"][1],
            gpr_M_reg=self._vis_mm_gpr(M=S)),
            cache=cache, cache_key=_ck("q_proj"))

        # 3. K projection (IF4)
        self._compile_and_run_single("clip_in_k", lambda: self._emit_clamp_dram_to_dram(
            src_dram=self.VIS_NORM_OUT, dst_dram=self.VIS_INPUT_CLIP_H_SCRATCH,
            num_elements=S * H,
            clamp_min=clips["k_proj"]["input"][0], clamp_max=clips["k_proj"]["input"][1]),
            cache=cache, cache_key=_ck("clip_in_k"))
        self._compile_and_run_single("k_proj", lambda: self._matmul_with_output_clamp(
            M=S, K=H, N=NH * HD,
            A_DRAM_ADDR=self.VIS_INPUT_CLIP_H_SCRATCH,
            B_DRAM_ADDR=w["k_proj"]["data"],
            OUTPUT_DRAM_ADDR=self.VIS_K_DRAM,
            is_B_quantized=True,
            data_type=TYPE.IF4,
            SCALE_DRAM_ADDR=w["k_proj"]["scale"],
            clamp_min=clips["k_proj"]["output"][0],
            clamp_max=clips["k_proj"]["output"][1],
            gpr_M_reg=self._vis_mm_gpr(M=S)),
            cache=cache, cache_key=_ck("k_proj"))

        # 4. V projection (IF4)
        self._compile_and_run_single("clip_in_v", lambda: self._emit_clamp_dram_to_dram(
            src_dram=self.VIS_NORM_OUT, dst_dram=self.VIS_INPUT_CLIP_H_SCRATCH,
            num_elements=S * H,
            clamp_min=clips["v_proj"]["input"][0], clamp_max=clips["v_proj"]["input"][1]),
            cache=cache, cache_key=_ck("clip_in_v"))
        self._compile_and_run_single("v_proj", lambda: self._matmul_with_output_clamp(
            M=S, K=H, N=NH * HD,
            A_DRAM_ADDR=self.VIS_INPUT_CLIP_H_SCRATCH,
            B_DRAM_ADDR=w["v_proj"]["data"],
            OUTPUT_DRAM_ADDR=self.VIS_V_DRAM,
            is_B_quantized=True,
            data_type=TYPE.IF4,
            SCALE_DRAM_ADDR=w["v_proj"]["scale"],
            clamp_min=clips["v_proj"]["output"][0],
            clamp_max=clips["v_proj"]["output"][1],
            gpr_M_reg=self._vis_mm_gpr(M=S)),
            cache=cache, cache_key=_ck("v_proj"))

        # 5. Q norm
        self._compile_and_run_single("q_norm", lambda: self._rms_norm_dram_pbi(
            M=S * NH, N=HD,
            A_DRAM_ADDR=self.VIS_Q_DRAM,
            OUTPUT_DRAM_ADDR=self.VIS_Q_NORM,
            GAMMA_DRAM_ADDR=w["q_norm"]),
            cache=cache, cache_key=_ck("q_norm"))

        # 6. K norm
        self._compile_and_run_single("k_norm", lambda: self._rms_norm_dram_pbi(
            M=S * NH, N=HD,
            A_DRAM_ADDR=self.VIS_K_DRAM,
            OUTPUT_DRAM_ADDR=self.VIS_K_NORM,
            GAMMA_DRAM_ADDR=w["k_norm"]),
            cache=cache, cache_key=_ck("k_norm"))

        return 0

    def host_vision_v_norm_rope_gather(self, layer_idx: int,
                                        cos_2d: torch.Tensor = None,
                                        sin_2d: torch.Tensor = None) -> None:
        """V-norm, 2D RoPE, per-head transpose, and Q pre-scale — all on FPGA.

        - V norm (FPGA): rms_norm_core_dram with constant ones(HD) gamma.
        - 2D RoPE (FPGA): qwen3.5 split-64 trick. See VIS_ROPE_PERM.
        - Per-head transpose (FPGA, #4): (S*NH, HD) interleaved →
          (NH, aligned_S, HD) head-major into VIS_FLASH_*_HM. Eliminates
          the host gather + per-head DMA upload of the old per-head loop.
        - Q pre-scale (FPGA, #4): broadcast_mul Q_HM by sqrt(HD) to cancel
          flash_attention_core's internal 1/sqrt(d).

        cos_2d/sin_2d args kept for back-compat with compare scripts; unused.
        Sets self._vis_q_roped / k_roped / v_normed only when called by the
        compare's per-head host-attention path (gated by _vis_keep_host_qkv).
        """
        S = self._vis_num_patches
        HD = self.VIS_HEAD_DIM
        NH = self.VIS_HEADS
        aligned_S = self._vis_aligned_S
        cache = getattr(self, "_vis_program_cache", None)

        # FPGA V-norm: in-place over VIS_V_DRAM with constant ones gamma.
        self._compile_and_run_single("v_norm", lambda: self._rms_norm_dram_pbi(
            M=S * NH, N=HD,
            A_DRAM_ADDR=self.VIS_V_DRAM,
            OUTPUT_DRAM_ADDR=self.VIS_V_DRAM,
            GAMMA_DRAM_ADDR=self.VIS_V_NORM_ONES_GAMMA),
            cache=cache, cache_key=f"vis_L{layer_idx}_v_norm")

        # FPGA 2D RoPE on Q (in-place at VIS_Q_NORM) and K (at VIS_K_NORM).
        self._compile_and_run_single("rope_q", lambda: self._emit_vision_rope_2d(
            src_dram=self.VIS_Q_NORM,
            out_dram=self.VIS_Q_NORM,
            cos_pad_dram=self.VIS_ROPE_COS_PAD_TILED,
            neg_sin_pad_dram=self.VIS_ROPE_NEG_SIN_PAD_TILED,
            sin_hi_pad_dram=self.VIS_ROPE_SIN_HI_PAD_TILED,
            M=S * NH),
            cache=cache, cache_key=f"vis_L{layer_idx}_rope_q")
        self._compile_and_run_single("rope_k", lambda: self._emit_vision_rope_2d(
            src_dram=self.VIS_K_NORM,
            out_dram=self.VIS_K_NORM,
            cos_pad_dram=self.VIS_ROPE_COS_PAD_TILED,
            neg_sin_pad_dram=self.VIS_ROPE_NEG_SIN_PAD_TILED,
            sin_hi_pad_dram=self.VIS_ROPE_SIN_HI_PAD_TILED,
            M=S * NH),
            cache=cache, cache_key=f"vis_L{layer_idx}_rope_k")

        # FPGA per-head transpose for Q/K/V into head-major buffers.
        self._compile_and_run_single("transpose_q", lambda: self._emit_qkv_transpose_to_hm(
            src_dram=self.VIS_Q_NORM, dst_dram=self.VIS_FLASH_Q_HM,
            S=S, dst_aligned_S=aligned_S),
            cache=cache, cache_key=f"vis_L{layer_idx}_transpose_q")
        self._compile_and_run_single("transpose_k", lambda: self._emit_qkv_transpose_to_hm(
            src_dram=self.VIS_K_NORM, dst_dram=self.VIS_FLASH_K_HM,
            S=S, dst_aligned_S=aligned_S),
            cache=cache, cache_key=f"vis_L{layer_idx}_transpose_k")
        self._compile_and_run_single("transpose_v", lambda: self._emit_qkv_transpose_to_hm(
            src_dram=self.VIS_V_DRAM, dst_dram=self.VIS_FLASH_V_HM,
            S=S, dst_aligned_S=aligned_S),
            cache=cache, cache_key=f"vis_L{layer_idx}_transpose_v")

        # Q pre-scale on FPGA (cancel flash_attention_core's internal 1/sqrt(d)).
        # Apply to all NH * aligned_S * HD elements in Q_HM in place.
        # Padding rows are zero so multiplying them by sqrt(HD) keeps them zero.
        self._compile_and_run_single("q_prescale", lambda: self._emit_sram_broadcast_mul_chunked(
            src_addr=self.VIS_FLASH_Q_HM, dst_addr=self.VIS_FLASH_Q_HM,
            num_elements=NH * aligned_S * HD,
            scalar=math.sqrt(HD)),
            cache=cache, cache_key=f"vis_L{layer_idx}_q_prescale")

        # Compare-script back-compat: if the per-head host-attention path is
        # still expected, also read Q/K/V back to host. Production path skips
        # this (run_vision_attention_all_heads now reads from FPGA HM buffers).
        if getattr(self, "_vis_keep_host_qkv", False):
            q_roped = self.dma_from_accelerator_memory(
                self.VIS_Q_NORM, (S * NH, HD)).cpu()
            k_roped = self.dma_from_accelerator_memory(
                self.VIS_K_NORM, (S * NH, HD)).cpu()
            v_normed = self.dma_from_accelerator_memory(
                self.VIS_V_DRAM, (S * NH, HD)).cpu().to(torch.bfloat16)
            self._vis_q_roped = q_roped.reshape(S, NH, HD).float()
            self._vis_k_roped = k_roped.reshape(S, NH, HD).float()
            self._vis_v_normed = v_normed.reshape(S, NH, HD)

    def run_vision_attention_all_heads(self, layer_idx: int) -> None:
        """Run attention for all heads: host gather -> FPGA attn -> host scatter.

        Q pre-scale is done on host to avoid SRAM overflow (aligned_S * HD can
        exceed available SRAM for large patch counts).

        Per-head FPGA program is cached in self._vis_program_cache.
        """
        bpe = self.bytes_per_element
        S = self._vis_num_patches
        HD = self.VIS_HEAD_DIM
        NH = self.VIS_HEADS
        aligned_S = self._vis_aligned_S
        cache = getattr(self, "_vis_program_cache", None)

        # #4: per-head FPGA flash_attention reads from head-major Q_HM/K_HM/V_HM
        # at offsets (no host gather, no per-head DMA upload). Each head's
        # cached program hardcodes its base addresses, so we get NH cached
        # programs per layer (vs the prior NH cache entries that all read
        # from the SAME single-head buffer). Outputs written to OUT_HM,
        # then inverse-transposed to interleaved VIS_Q_DRAM for o_proj.
        head_stride_bytes = aligned_S * HD * bpe
        for h in range(NH):
            q_addr = self.VIS_FLASH_Q_HM   + h * head_stride_bytes
            k_addr = self.VIS_FLASH_K_HM   + h * head_stride_bytes
            v_addr = self.VIS_FLASH_V_HM   + h * head_stride_bytes
            o_addr = self.VIS_FLASH_OUT_HM + h * head_stride_bytes
            self._compile_and_run_single(f"attn_h{h}", lambda q=q_addr, k=k_addr, v=v_addr, o=o_addr:
                self._flash_attention_core_cached(
                    head_dim=HD,
                    seq_len=aligned_S,
                    Q_DRAM_ADDR=q,
                    K_DRAM_ADDR=k,
                    V_DRAM_ADDR=v,
                    OUTPUT_DRAM_ADDR=o,
                    SCRATCH_DRAM_ADDR=self.VIS_FLASH_SCRATCH,
                    BIAS_DRAM_ADDR=self.VIS_FLASH_BIAS),  # safe now that VIS_FLASH_SCRATCH is correctly sized; ~6× per-layer ISA shrink enables true 1-trigger one-shot
                cache=cache, cache_key=f"vis_L{layer_idx}_attn_h{h}")

        # FPGA inverse transpose: OUT_HM (NH, aligned_S, HD) → VIS_Q_DRAM (S, NH*HD)
        self._compile_and_run_single("transpose_attn_out", lambda:
            self._emit_attn_out_transpose_to_interleaved(
                src_dram=self.VIS_FLASH_OUT_HM, dst_dram=self.VIS_Q_DRAM,
                S=S, src_aligned_S=aligned_S),
            cache=cache, cache_key=f"vis_L{layer_idx}_transpose_attn_out")

    def compile_vision_layer_post_attn(self, layer_idx: int) -> int:
        """Run post-attention: O proj + post_attn_norm + residual + MLP + output.

        Each operation compiled and run separately. Per-segment compile bytes
        are cached in self._vis_program_cache for cross-run reuse (matmul/norm
        sites only — eltwise chunked passes stay uncached for simplicity).
        """
        S = self._vis_num_patches
        H = self.VIS_H
        HD = self.VIS_HEAD_DIM
        NH = self.VIS_HEADS
        MLP = self.VIS_MLP
        w = self._vis_weight_addrs[layer_idx]
        cache = getattr(self, "_vis_program_cache", None)

        def _ck(label: str) -> str:
            return f"vis_L{layer_idx}_{label}"

        INPUT_DRAM = self.VIS_IO_A if layer_idx % 2 == 0 else self.VIS_IO_B
        OUTPUT_DRAM = self.VIS_IO_B if layer_idx % 2 == 0 else self.VIS_IO_A

        clips = self._vis_clip_ranges[layer_idx]

        # O projection (IF4): FPGA input clamp into scratch; fused output clamp
        self._compile_and_run_single("clip_in_o", lambda: self._emit_clamp_dram_to_dram(
            src_dram=self.VIS_Q_DRAM, dst_dram=self.VIS_INPUT_CLIP_H_SCRATCH,
            num_elements=S * NH * HD,
            clamp_min=clips["o_proj"]["input"][0], clamp_max=clips["o_proj"]["input"][1]),
            cache=cache, cache_key=_ck("clip_in_o"))
        self._compile_and_run_single("o_proj", lambda: self._matmul_with_output_clamp(
            M=S, K=NH * HD, N=H,
            A_DRAM_ADDR=self.VIS_INPUT_CLIP_H_SCRATCH,
            B_DRAM_ADDR=w["o_proj"]["data"],
            OUTPUT_DRAM_ADDR=self.VIS_ATTN_OUT,
            is_B_quantized=True,
            data_type=TYPE.IF4,
            SCALE_DRAM_ADDR=w["o_proj"]["scale"],
            clamp_min=clips["o_proj"]["output"][0],
            clamp_max=clips["o_proj"]["output"][1],
            gpr_M_reg=self._vis_mm_gpr(M=S)),
            cache=cache, cache_key=_ck("o_proj"))

        # Post-attention norm
        self._compile_and_run_single("post_attn_norm", lambda: self._rms_norm_dram_pbi(
            M=S, N=H,
            A_DRAM_ADDR=self.VIS_ATTN_OUT,
            OUTPUT_DRAM_ADDR=self.VIS_POST_ATTN_NORM,
            GAMMA_DRAM_ADDR=w["post_attention_layernorm"]),
            cache=cache, cache_key=_ck("post_attn_norm"))

        # Residual: input + post_attn_norm (chunked, each chunk a separate program)
        sz_h = S * H
        self._run_eltwise_add_chunked(INPUT_DRAM, self.VIS_POST_ATTN_NORM,
                                       self.VIS_POST_ATTN_RES, sz_h)

        # Pre-FFN norm
        self._compile_and_run_single("pre_ffn_norm", lambda: self._rms_norm_dram_pbi(
            M=S, N=H,
            A_DRAM_ADDR=self.VIS_POST_ATTN_RES,
            OUTPUT_DRAM_ADDR=self.VIS_PRE_FFN_NORM,
            GAMMA_DRAM_ADDR=w["pre_feedforward_layernorm"]),
            cache=cache, cache_key=_ck("pre_ffn_norm"))

        # MLP gate (IF4, GELU): FPGA input clamp; gate output clamp also FPGA
        # (post-GELU; can't fuse since GELU+CLAMP both occupy LALU).
        self._compile_and_run_single("clip_in_gate", lambda: self._emit_clamp_dram_to_dram(
            src_dram=self.VIS_PRE_FFN_NORM, dst_dram=self.VIS_INPUT_CLIP_H_SCRATCH,
            num_elements=S * H,
            clamp_min=clips["gate_proj"]["input"][0], clamp_max=clips["gate_proj"]["input"][1]),
            cache=cache, cache_key=_ck("clip_in_gate"))
        self._compile_and_run_single("gate_proj", lambda: self.matmat_mul_core(
            M=S, K=H, N=MLP,
            A_DRAM_ADDR=self.VIS_INPUT_CLIP_H_SCRATCH,
            B_DRAM_ADDR=w["gate_proj"]["data"],
            OUTPUT_DRAM_ADDR=self.VIS_MLP_GATE,
            is_B_quantized=True,
            data_type=TYPE.IF4,
            SCALE_DRAM_ADDR=w["gate_proj"]["scale"],
            gelu_enable=True,
            gpr_M_reg=self._vis_mm_gpr(M=S)),
            cache=cache, cache_key=_ck("gate_proj"))
        self._compile_and_run_single("clip_out_gate", lambda: self._emit_clamp_dram_to_dram(
            src_dram=self.VIS_MLP_GATE, dst_dram=self.VIS_MLP_GATE,
            num_elements=S * MLP,
            clamp_min=clips["gate_proj"]["output"][0], clamp_max=clips["gate_proj"]["output"][1]),
            cache=cache, cache_key=_ck("clip_out_gate"))

        # MLP up (IF4): FPGA input clamp; fused output clamp
        self._compile_and_run_single("clip_in_up", lambda: self._emit_clamp_dram_to_dram(
            src_dram=self.VIS_PRE_FFN_NORM, dst_dram=self.VIS_INPUT_CLIP_H_SCRATCH,
            num_elements=S * H,
            clamp_min=clips["up_proj"]["input"][0], clamp_max=clips["up_proj"]["input"][1]),
            cache=cache, cache_key=_ck("clip_in_up"))
        self._compile_and_run_single("up_proj", lambda: self._matmul_with_output_clamp(
            M=S, K=H, N=MLP,
            A_DRAM_ADDR=self.VIS_INPUT_CLIP_H_SCRATCH,
            B_DRAM_ADDR=w["up_proj"]["data"],
            OUTPUT_DRAM_ADDR=self.VIS_MLP_UP,
            is_B_quantized=True,
            data_type=TYPE.IF4,
            SCALE_DRAM_ADDR=w["up_proj"]["scale"],
            clamp_min=clips["up_proj"]["output"][0],
            clamp_max=clips["up_proj"]["output"][1],
            gpr_M_reg=self._vis_mm_gpr(M=S)),
            cache=cache, cache_key=_ck("up_proj"))

        # gate * up (chunked, each chunk a separate program)
        self._run_eltwise_mul_chunked(self.VIS_MLP_GATE, self.VIS_MLP_UP,
                                       self.VIS_MLP_MULT, S * MLP)

        # MLP down (IF4): FPGA input clamp; fused output clamp
        self._compile_and_run_single("clip_in_down", lambda: self._emit_clamp_dram_to_dram(
            src_dram=self.VIS_MLP_MULT, dst_dram=self.VIS_INPUT_CLIP_MLP_SCRATCH,
            num_elements=S * MLP,
            clamp_min=clips["down_proj"]["input"][0], clamp_max=clips["down_proj"]["input"][1]),
            cache=cache, cache_key=_ck("clip_in_down"))
        self._compile_and_run_single("down_proj", lambda: self._matmul_with_output_clamp(
            M=S, K=MLP, N=H,
            A_DRAM_ADDR=self.VIS_INPUT_CLIP_MLP_SCRATCH,
            B_DRAM_ADDR=w["down_proj"]["data"],
            OUTPUT_DRAM_ADDR=self.VIS_MLP_DOWN,
            is_B_quantized=True,
            data_type=TYPE.IF4,
            SCALE_DRAM_ADDR=w["down_proj"]["scale"],
            clamp_min=clips["down_proj"]["output"][0],
            clamp_max=clips["down_proj"]["output"][1],
            gpr_M_reg=self._vis_mm_gpr(M=S)),
            cache=cache, cache_key=_ck("down_proj"))

        # Post-FFN norm
        self._compile_and_run_single("post_ffn_norm", lambda: self._rms_norm_dram_pbi(
            M=S, N=H,
            A_DRAM_ADDR=self.VIS_MLP_DOWN,
            OUTPUT_DRAM_ADDR=self.VIS_POST_FFN_NORM,
            GAMMA_DRAM_ADDR=w["post_feedforward_layernorm"]),
            cache=cache, cache_key=_ck("post_ffn_norm"))

        # Post-FFN residual -> OUTPUT (chunked, each chunk a separate program)
        self._run_eltwise_add_chunked(self.VIS_POST_ATTN_RES, self.VIS_POST_FFN_NORM,
                                       OUTPUT_DRAM, sz_h)

        return 0

    def run_vision_layer(self, layer_idx: int,
                          cos_2d: torch.Tensor = None,
                          sin_2d: torch.Tensor = None) -> int:
        """Run one full vision encoder layer on HW.

        Orchestrates: pre_norm + Q/K/V/O projections + Q/K norms (FPGA) →
        V norm + 2D RoPE + per-head transpose + Q pre-scale (FPGA) →
        per-head flash attention reading from head-major buffers (FPGA) →
        attn-out inverse transpose + O proj + post_attn_norm + residual +
        MLP + post_ffn_norm + residual (FPGA).

        Input must already be in VIS_IO_A (if layer_idx even) or VIS_IO_B
        (if layer_idx odd). Output lands in the other of the two buffers.
        Returns the output DRAM address so callers can read back.

        cos_2d/sin_2d args are unused since #3 (RoPE on FPGA reads from
        VIS_ROPE_*_PAD_TILED DRAM); kept for back-compat with compare scripts.
        """
        t0 = time.perf_counter()
        self.compile_vision_layer(layer_idx)
        tA = time.perf_counter() - t0; t0 = time.perf_counter()
        self.host_vision_v_norm_rope_gather(layer_idx, cos_2d, sin_2d)
        tB = time.perf_counter() - t0; t0 = time.perf_counter()
        self.run_vision_attention_all_heads(layer_idx)
        tC = time.perf_counter() - t0; t0 = time.perf_counter()
        self.compile_vision_layer_post_attn(layer_idx)
        tD = time.perf_counter() - t0
        print(f"  [Vision L{layer_idx+1}/{self.VIS_LAYERS}] done (A={tA:.2f}s B={tB:.2f}s C={tC:.2f}s D={tD:.2f}s, total={tA+tB+tC+tD:.2f}s)", flush=True)
        return self.VIS_IO_B if layer_idx % 2 == 0 else self.VIS_IO_A

    def _vision_encoder_bin_paths(self, num_patches: int) -> tuple[int, list[tuple[int, int]], str, str]:
        """Resolve G, group plan, and bin/meta paths for the vision encoder
        one-shot. Pure resolution — no I/O, no state mutation."""
        L = self.VIS_LAYERS
        try:
            G = int(os.environ.get("GEMMA4_VISION_LAYERS_PER_GROUP", "16"))
        except ValueError:
            G = 16
        G = max(1, min(G, L))
        groups = []
        i = 0
        while i < L:
            n = min(G, L - i)
            groups.append((i, n))
            i += n
        cache_dir = os.path.join(self.script_dir, "gemma4_e2b_bin")
        os.makedirs(cache_dir, exist_ok=True)
        bin_path  = os.path.join(cache_dir, f"vision_encoder_oneshot_v6_{num_patches}p_g{G}.bin")
        meta_path = os.path.join(cache_dir, f"vision_encoder_oneshot_v6_{num_patches}p_g{G}.json")
        return G, groups, bin_path, meta_path

    def compile_vision_encoder_bin(self, num_patches: int) -> None:
        """Compile the 16-layer vision encoder ISA into a bin file on disk.
        Atomic write + post-save assertions. Skips if the bin is already cached.
        No FPGA activity — pure host-side ISA emission."""
        L = self.VIS_LAYERS
        G, groups, bin_path, meta_path = self._vision_encoder_bin_paths(num_patches)

        if os.path.exists(bin_path) and os.path.exists(meta_path):
            print(f"  [Vision] bin cached, skipping compile: {os.path.basename(bin_path)}", flush=True)
            return

        print(f"  [Vision] compiling {L} layers into bin (G={G}) ...", flush=True)
        t_capture = time.perf_counter()
        group_offsets = []
        group_sizes = []
        full_bytes = bytearray()
        import builtins
        _orig_print = builtins.print
        global _SILENT_MODE
        for gi, (start, n) in enumerate(groups):
            # CRITICAL: reset program DRAM addr BEFORE start_capture so PBI
            # loops in this group bake correct absolute jump targets.
            self.reset_program_dram_addr()
            self._oneshot_mode = True
            _SILENT_MODE = True
            self.clear_capture_buffer()
            self.start_capture()
            builtins.print = lambda *a, **kw: None
            try:
                for li in range(start, start + n):
                    t_layer = time.perf_counter()
                    self.compile_vision_layer(li)
                    self.host_vision_v_norm_rope_gather(li)
                    self.run_vision_attention_all_heads(li)
                    self.compile_vision_layer_post_attn(li)
                    # Per-layer heartbeat. Must use the module-level
                    # `_original_print` (the REAL print captured at import
                    # time, line 61) — NOT `_orig_print` here which is the
                    # `quiet_print` wrapper that respects _SILENT_MODE and
                    # would silently drop the heartbeat.
                    _original_print(
                        f"  [Vision] layer {li-start+1}/{n} compiled in {time.perf_counter()-t_layer:.2f}s",
                        flush=True,
                    )
            finally:
                builtins.print = _orig_print
                _SILENT_MODE = False
                self._oneshot_mode = False
            self.stop_capture()
            self.generate_instruction_halt()
            grp_bytes = bytearray()
            for inst in self.capture_buffer:
                grp_bytes.extend(inst.get_bytes())
            self.clear_capture_buffer()
            group_offsets.append(len(full_bytes))
            full_bytes.extend(grp_bytes)
            group_sizes.append(len(grp_bytes))
            _original_print(f"  [Vision] {n} layers captured: {len(grp_bytes)/1024/1024:.1f} MB", flush=True)
        # Atomic write: write to .tmp then rename so a crash mid-write
        # cannot leave a half-written bin behind.
        bin_tmp = bin_path + ".tmp"
        meta_tmp = meta_path + ".tmp"
        with open(bin_tmp, "wb") as f:
            f.write(full_bytes)
            f.flush()
            os.fsync(f.fileno())
        with open(meta_tmp, "w") as f:
            json.dump({
                "num_patches": num_patches,
                "vis_layers": L,
                "layers_per_group": G,
                "groups": groups,
                "group_offsets": group_offsets,
                "group_sizes": group_sizes,
                "total_bytes": len(full_bytes),
            }, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.rename(bin_tmp, bin_path)
        os.rename(meta_tmp, meta_path)
        # Verify on disk before continuing.
        assert os.path.exists(bin_path), f"vision bin save failed: {bin_path} not on disk"
        assert os.path.exists(meta_path), f"vision meta save failed: {meta_path} not on disk"
        bin_disk_size = os.path.getsize(bin_path)
        assert bin_disk_size == len(full_bytes), \
            f"vision bin size mismatch: disk={bin_disk_size}, expected={len(full_bytes)}"
        print(f"  [Vision] bin saved ({bin_disk_size/1024/1024:.1f} MB, {time.perf_counter()-t_capture:.1f}s)", flush=True)

    def execute_vision_encoder_bin(self, num_patches: int) -> float:
        """Load the encoder bin from disk, DMA it to FPGA program DRAM, and
        execute. Returns total elapsed seconds (DMA + execute across groups)."""
        G, groups, bin_path, meta_path = self._vision_encoder_bin_paths(num_patches)
        # Read bin from disk, DMA to FPGA, execute each group with halt+wait between.
        t_read = time.perf_counter()
        with open(meta_path, "r") as f:
            meta = json.load(f)
        group_offsets = meta["group_offsets"]
        group_sizes = meta["group_sizes"]
        with open(bin_path, "rb") as f:
            full_bytes = f.read()
        print(f"  [Vision] bin loaded: {len(full_bytes)/1024/1024:.1f} MB in {time.perf_counter()-t_read:.2f}s", flush=True)

        import threading
        # Use the unified vision FPGA timer if set by the caller (so heartbeat
        # elapsed counts from patch-embed start). Fall back to local t0 if not.
        t0 = time.perf_counter()
        _hb_anchor = getattr(self, "_vis_fpga_t0", t0)
        for gi in range(len(group_offsets)):
            grp_bytes = full_bytes[group_offsets[gi]:group_offsets[gi] + group_sizes[gi]]
            self.reset_program_dram_addr()
            program_addr = self.get_program_dram_addr()
            t_dma = time.perf_counter()
            self.dma_write(DMA_DEVICE_H2C, program_addr, grp_bytes, len(grp_bytes))
            self.allocate_program_dram(len(grp_bytes))
            dt_dma = time.perf_counter() - t_dma
            print(f"  [Vision] running on FPGA: DMA {len(grp_bytes)/1024/1024:.1f} MB in {dt_dma:.2f}s ({len(grp_bytes)/1024/1024/dt_dma:.0f} MB/s)", flush=True)
            # Trigger FPGA execute and wait for halt — long blocking step
            # (~60-80 s for vision G=16). 10s heartbeat avoids "stuck" feel.
            t_exec = time.perf_counter()
            _hb_stop = threading.Event()
            def _hb_run(_anchor=_hb_anchor):
                while not _hb_stop.wait(10):
                    _original_print(f"  [Vision] ... running on FPGA ({time.perf_counter()-_anchor:.0f}s)", flush=True)
            _hb_th = threading.Thread(target=_hb_run, daemon=True)
            _hb_th.start()
            try:
                self.start_execute_from_dram(program_addr)
                self.wait_queue(180.0)
            finally:
                _hb_stop.set()
                _hb_th.join(timeout=1.0)
            print(f"  [Vision] running on FPGA: encoder execute done in {time.perf_counter()-t_exec:.2f}s", flush=True)
        elapsed = time.perf_counter() - t0
        return elapsed

    def vision_patch_embed(self,
                            pixel_values: torch.Tensor,
                            pixel_position_ids: torch.Tensor,
                            padding_positions: torch.Tensor) -> torch.Tensor:
        """Run the Gemma4 vision patch embedder on FPGA.

        Mirrors Gemma4VisionPatchEmbedder.forward:
          scaled = 2 * (pixel_values - 0.5)
          hidden = input_proj(scaled)               # FPGA, IF4 (block=64)
          pos    = pos_table[0, x] + pos_table[1, y]  # host gather
          pos[padding] = 0
          return hidden + pos                        # FPGA, chunked eltwise add

        Uses VIS_IO_B as a pixel scratchpad and VIS_NORM_OUT as the position-
        embedding staging buffer (both are otherwise unused before layer 0
        runs). Final patch embeddings land in VIS_IO_A.

        Returns the patch embeddings [S, H] bf16 (read back for comparison).
        """
        S = self._vis_num_patches
        H = self.VIS_H

        pv = pixel_values
        if pv.dim() == 3 and pv.shape[0] == 1:
            pv = pv.squeeze(0)
        assert pv.shape == (S, H), f"pixels shape {pv.shape}, expected ({S}, {H})"
        pids = pixel_position_ids
        if pids.dim() == 3 and pids.shape[0] == 1:
            pids = pids.squeeze(0)
        pad = padding_positions
        if pad.dim() == 2 and pad.shape[0] == 1:
            pad = pad.squeeze(0)

        # Host: scale pixels, upload to VIS_IO_B as temp.
        scaled = (2.0 * (pv.float() - 0.5)).to(torch.bfloat16).contiguous()
        self.dma_to_accelerator_memory(self.VIS_IO_B, scaled)

        # FPGA: input_proj matmul (IF4 (block=64)) → VIS_IO_A
        w = self.VIS_PATCH_PROJ_INFO
        self._compile_and_run_single("patch_input_proj", lambda: self.matmat_mul_core(
            M=S, K=H, N=H,
            A_DRAM_ADDR=self.VIS_IO_B,
            B_DRAM_ADDR=w["data"],
            OUTPUT_DRAM_ADDR=self.VIS_IO_A,
            is_B_quantized=True,
            data_type=TYPE.IF4,
            SCALE_DRAM_ADDR=w["scale"],
            gpr_M_reg=self._vis_mm_gpr(M=S)))

        # Host: gather position embeddings [S, H], zero padding rows.
        table = self._vis_pos_embed_table.float()          # [2, P, H]
        clamped = pids.clamp(min=0).long().cpu()           # [S, 2]
        pe_sum = (table[0, clamped[:, 0]] + table[1, clamped[:, 1]])  # [S, H]
        pe_sum[pad.cpu()] = 0.0
        pe_bf16 = pe_sum.to(torch.bfloat16).contiguous()

        # Upload to VIS_NORM_OUT (scratch) and FPGA eltwise-add into VIS_IO_A
        self.dma_to_accelerator_memory(self.VIS_NORM_OUT, pe_bf16)
        self._run_eltwise_add_chunked(
            self.VIS_IO_A, self.VIS_NORM_OUT, self.VIS_IO_A, S * H)

        return self.dma_from_accelerator_memory(self.VIS_IO_A, (S, H)).cpu()

    def vision_embed_project(self,
                              hidden_states: torch.Tensor,
                              pixel_position_ids: torch.Tensor,
                              padding_positions: torch.Tensor) -> torch.Tensor:
        """Run the Gemma4 vision pooler + embed_vision tail.

        Mirrors Gemma4VisionModel.forward's tail:
          hidden = masked_fill(hidden, padding)
          hidden = _avg_pool_by_positions(hidden, ids, output_length)   # host
          hidden = hidden * sqrt(hidden_size)                            # host
          hidden = hidden[pooler_mask]                                   # host
          hidden = rms_norm(hidden, with_scale=False)                    # FPGA
          out    = embedding_projection @ hidden                          # FPGA (FP4)

        Pooling and gather/mask stay on host because they are indexed by
        position (hard to express with matmat cores) and only touch <~800
        rows. The per-element RMSNorm and the Linear(768, 1536) are on FPGA.

        hidden_states: [S, 768] or [1, S, 768] — output of the encoder.
        Returns image_features [N_final, 1536] bf16.
        """
        H = self.VIS_H
        text_h = self.VIS_TEXT_H
        pool_k = self.VIS_POOL_K

        # Bring everything to CPU — the HF model may be on GPU, but the
        # pooler math is cheap and the FPGA DMA path expects CPU tensors.
        hidden_states = hidden_states.detach().cpu()
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        S = hidden_states.shape[1]
        output_length = S // (pool_k * pool_k)

        pids = pixel_position_ids.detach().cpu()
        if pids.dim() == 2:
            pids = pids.unsqueeze(0)
        pad = padding_positions.detach().cpu()
        if pad.dim() == 1:
            pad = pad.unsqueeze(0)

        # Host: pooler.forward equivalent (zero padding, spatial avg pool,
        # scale by sqrt(hidden_size), strip masked rows).
        # This inlines Gemma4VisionPooler._avg_pool_by_positions so the method
        # has no dependency on the HF module tree at runtime.
        with torch.no_grad():
            h = hidden_states.float().clone()
            h.masked_fill_(pad.unsqueeze(-1), 0.0)
            input_seq_len = h.shape[1]
            k = int((input_seq_len // output_length) ** 0.5)
            k_squared = k * k
            if k_squared * output_length != input_seq_len:
                raise ValueError(
                    f"Cannot pool {h.shape} to {output_length}: k={k}^2 × length="
                    f"{output_length} must equal {input_seq_len}.")
            clamped = pids.clamp(min=0)                         # [1, S, 2]
            max_x = clamped[..., 0].max(dim=-1, keepdim=True)[0] + 1
            kernel_idxs = torch.div(clamped, k, rounding_mode="floor")
            kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
            weights = F.one_hot(kernel_idxs.long(), output_length).float() / k_squared  # [1, S, L]
            pooled = weights.transpose(1, 2) @ h                # [1, L, 768]
            pooler_mask = torch.logical_not((weights == 0).all(dim=1))  # [1, L]
            pooled = pooled * (H ** 0.5)
            pooled = pooled[pooler_mask]                        # [N_final, 768]
        N_final = int(pooled.shape[0])
        assert N_final <= S, f"pooler produced {N_final} rows, scratch sized for {S}"

        pooled_bf16 = pooled.to(torch.bfloat16).contiguous()
        self.dma_to_accelerator_memory(self.VIS_EMBED_POOL, pooled_bf16)

        # FPGA: RMSNorm (with_scale=False → gamma=ones)
        self._compile_and_run_single("embed_pre_norm", lambda: self.rms_norm_core_dram(
            M=N_final, N=H,
            A_DRAM_ADDR=self.VIS_EMBED_POOL,
            OUTPUT_DRAM_ADDR=self.VIS_EMBED_NORMED,
            GAMMA_DRAM_ADDR=self.VIS_EMBED_NORM_GAMMA))

        # FPGA: embedding_projection (IF4 (block=64))
        w = self.VIS_EMBED_PROJ_INFO
        self._compile_and_run_single("embed_projection", lambda: self.matmat_mul_core(
            M=N_final, K=H, N=text_h,
            A_DRAM_ADDR=self.VIS_EMBED_NORMED,
            B_DRAM_ADDR=w["data"],
            OUTPUT_DRAM_ADDR=self.VIS_EMBED_OUT,
            is_B_quantized=True,
            data_type=TYPE.IF4,
            SCALE_DRAM_ADDR=w["scale"],
            gpr_M_reg=self._vis_mm_gpr(M=N_final)))

        return self.dma_from_accelerator_memory(self.VIS_EMBED_OUT, (N_final, text_h)).cpu()

    # ================================================================
    #  Audio encoder on FPGA  (Conformer, ported from Parakeet)
    # ================================================================
    #
    # Architecture (mirrors Gemma4AudioModel in HF):
    #   log-mel features → SubSampleConvProjection (host) → 12 × Gemma4AudioLayer
    #   → output_proj (1024 → 1536) → Gemma4MultimodalEmbedder (RMSNorm + Linear)
    #
    # Per layer (Gemma4AudioLayer.forward):
    #   FFN1 macaron half (RMSNorm in/out, Linear+SiLU+Linear, *0.5 residual)
    #   → norm_pre_attn → chunked self-attn (rel-pos, soft-cap) → norm_post_attn → +residual
    #   → lconv1d (RMSNorm, Linear→GLU, depthwise k=5, conv_norm, SiLU, Linear, +residual)
    #   → FFN2 macaron half
    #   → norm_out
    #
    # Stages currently running on host (small, low frequency):
    #   - SubSampleConvProjection (Conv2d stem, ~64 KB weights)
    #   - chunked self-attn rel-pos / soft-cap / softmax
    #   - depthwise Conv1d (light conv module, kernel size 5)
    #   - audio_embed_project (output_proj + multimodal embedder)

    def audio_config_init(self) -> None:
        """Parse Gemma4AudioConfig values from the local HF model directory's
        config.json. Cached on self for use by audio_weight_init/tensor_init."""
        if hasattr(self, '_audio_cfg'):
            return  # already initialized
        import json as _json
        model_dir = os.path.join(self.script_dir, self._cfg["paths"]["hf_model_dir"])
        cfg_path = os.path.join(model_dir, "config.json")
        with open(cfg_path) as f:
            full = _json.load(f)
        ac = full.get("audio_config")
        if ac is None:
            raise RuntimeError(
                f"audio_config missing from {cfg_path}. "
                f"This model does not include the audio encoder.")
        self._audio_cfg = ac
        self.AUD_H = ac["hidden_size"]                      # 1024
        self.AUD_HEADS = ac["num_attention_heads"]          # 8
        self.AUD_HEAD_DIM = ac["hidden_size"] // ac["num_attention_heads"]  # 128
        self.AUD_FFN = ac["hidden_size"] * 4                # 4096
        self.AUD_LAYERS = ac["num_hidden_layers"]           # 12
        self.AUD_CONV_K = ac["conv_kernel_size"]            # 5
        self.AUD_CHUNK = ac["attention_chunk_size"]         # 12
        self.AUD_CTX_LEFT = ac["attention_context_left"]    # 13
        self.AUD_CTX_RIGHT = ac["attention_context_right"]  # 0
        # context_size = chunk + (left-1) + right
        self.AUD_CTX = self.AUD_CHUNK + self.AUD_CTX_LEFT - 1 + self.AUD_CTX_RIGHT
        self.AUD_SOFT_CAP = ac["attention_logit_cap"]       # 50
        self.AUD_RESIDUAL_W = ac["residual_weight"]         # 0.5
        self.AUD_RMS_EPS = ac["rms_norm_eps"]
        self.AUD_OUT_DIM = ac["output_proj_dims"]           # 1536 (LM hidden size)
        self.AUD_SUB_CHANS = ac["subsampling_conv_channels"]  # [128, 32]
        self.AUD_USE_CLIP = ac["use_clipped_linears"]
        self.AUD_INVALID_LOGIT = ac["attention_invalid_logits_value"]
        # q_scale = (head_dim^-0.5) / log(2),  k_scale = log(1 + e) / log(2)
        self.AUD_Q_SCALE = (self.AUD_HEAD_DIM ** -0.5) / math.log(2)
        self.AUD_K_SCALE = math.log(1.0 + math.e) / math.log(2)
        print(f"[Audio] config loaded: {self.AUD_LAYERS} layers, "
              f"H={self.AUD_H}, heads={self.AUD_HEADS}, FFN={self.AUD_FFN}, "
              f"conv_k={self.AUD_CONV_K}, chunk={self.AUD_CHUNK}, ctx={self.AUD_CTX}")

    def _audio_weight_init_from_combined_bin(self, weights_bin_path: str, audio_section: dict) -> None:
        """Load pre-quantized audio weights from the combined weights bin.
        Mirrors _vision_weight_init_from_combined_bin: no HF model, no
        quantization. Allocation order matches audio_weight_init's HF path
        so DRAM addresses baked into the captured audio ISA resolve.

        G_s0 and ID_64 are deterministic and recomputed on host here
        (not stored in the bin).
        """
        self.audio_config_init()
        bpe = self.bytes_per_element
        base_offset = int(audio_section["offset"])
        meta = audio_section["manifest"]
        if meta.get("audio_quant_precision") != AUDIO_QUANT_PRECISION:
            raise RuntimeError(
                f"audio section quant precision mismatch (disk: "
                f"{meta.get('audio_quant_precision')!r}, expected "
                f"{AUDIO_QUANT_PRECISION!r}). Regenerate the weights bin.")
        L_count = int(meta["num_layers"])
        sections = meta["sections"]

        _aud_weight_tensor_cursor_save = self._tensor_dram_addr
        self._tensor_dram_addr = 0x6c000000

        print(f"\n[Audio] Loading pre-quantized audio weights from combined weights bin "
              f"({AUDIO_QUANT_PRECISION.upper()} block=64 + BF16 norms) ...")
        f = open(weights_bin_path, "rb")
        try:
            def _dma_section(key: str) -> int:
                s = sections[key]
                f.seek(base_offset + s["offset"])
                bts = f.read(s["size"])
                if len(bts) != s["size"]:
                    raise RuntimeError(f"truncated section read {key} at offset {base_offset + s['offset']}")
                addr = self.allocate_tensor_dram(s["size"])
                self.dma_write(DMA_DEVICE_H2C, addr, bts, s["size"])
                return addr

            def _alloc_if4(name: str) -> dict:
                scale_addr = _dma_section(f"{name}.scale")
                data_addr  = _dma_section(f"{name}.data")
                return {"data": data_addr, "scale": scale_addr,
                        "shape": tuple(sections[f"{name}.data"]["shape"])}

            self._aud_sub_conv0_addrs = _alloc_if4("subsample.conv0")
            self._aud_sub_conv1_addrs = _alloc_if4("subsample.conv1")
            self._aud_sub_proj_addrs  = _alloc_if4("subsample.proj")

            self._aud_sub_ln0_gamma_addr = _dma_section("subsample.ln0_gamma")
            self._aud_sub_ln1_gamma_addr = _dma_section("subsample.ln1_gamma_64")
            id_64_addr = self.allocate_tensor_dram(64 * 64 * bpe)
            self.dma_to_accelerator_memory(id_64_addr,
                torch.eye(64, dtype=torch.bfloat16).contiguous())
            self._aud_sub_id_64_addr = id_64_addr
            # Trick 9: shared LayerNorm zeros base — seeded here on the load path
            # (replays the kernel's compile-time-only zeros write) at the same addr
            # the baked LN read uses; passed to both LNs via ZEROS_DRAM_ADDR.
            ln_zeros_addr = self.allocate_tensor_dram(128 * bpe)
            self.dma_to_accelerator_memory(ln_zeros_addr,
                torch.zeros(128, dtype=torch.bfloat16).contiguous())
            self._aud_ln_zeros_addr = ln_zeros_addr
            VS = UE_VECTOR_SIZE
            _W_in_s0 = 128
            _W_out_s0 = (_W_in_s0 + 2 - 3) // 2 + 1
            K_g_s0 = ((3 * _W_in_s0 + VS - 1) // VS) * VS
            N_g_s0 = _W_out_s0 * 64
            G_s0 = torch.zeros(N_g_s0, K_g_s0, dtype=torch.bfloat16)
            for kh in range(3):
                for kw in range(3):
                    for ow in range(_W_out_s0):
                        col = ow * 2 - 1 + kw
                        if 0 <= col < _W_in_s0:
                            G_s0[ow * 64 + kh * 3 + kw, kh * _W_in_s0 + col] = 1.0
            g_s0_addr = self.allocate_tensor_dram(N_g_s0 * K_g_s0 * bpe)
            self.dma_to_accelerator_memory(g_s0_addr, G_s0.contiguous())
            self._aud_sub_G_s0_addr = g_s0_addr
            self._aud_sub_K_g_s0 = K_g_s0
            self._aud_sub_N_g_s0 = N_g_s0

            layer_addrs: list[dict] = []
            clip_ranges: list[dict] = []
            def _str_to_float(v):
                if v == "inf":  return float("inf")
                if v == "-inf": return -float("inf")
                return float(v)
            def _to_old_clip(cr: dict) -> dict:
                return {
                    "in_min":  _str_to_float(cr["in_min"]),
                    "in_max":  _str_to_float(cr["in_max"]),
                    "out_min": _str_to_float(cr["out_min"]),
                    "out_max": _str_to_float(cr["out_max"]),
                }
            hf_cache: list[dict] = []
            for li in range(L_count):
                pre = f"layer{li}"
                addrs: dict = {}
                addrs["FF1_PRE_NORM"]    = _dma_section(f"{pre}.ff1_pre_norm")
                addrs["FF1_POST_NORM"]   = _dma_section(f"{pre}.ff1_post_norm")
                addrs["ATTN_PRE_NORM"]   = _dma_section(f"{pre}.attn_pre_norm")
                addrs["PER_DIM_SCALE"]   = _dma_section(f"{pre}.per_dim_scale")
                addrs["ATTN_POST_NORM"]  = _dma_section(f"{pre}.attn_post_norm")
                addrs["CONV_PRE_NORM"]   = _dma_section(f"{pre}.conv_pre_norm")
                # CONV_DW_W is both DMA'd (for FPGA) AND kept on host because
                # audio_tensor_init reads dw_w[:, t] to build per-tap tile buffers.
                dw_section = sections[f"{pre}.conv_dw_w"]
                f.seek(base_offset + dw_section["offset"])
                dw_bytes = f.read(dw_section["size"])
                addrs["CONV_DW_W"] = self.allocate_tensor_dram(dw_section["size"])
                self.dma_write(DMA_DEVICE_H2C, addrs["CONV_DW_W"], dw_bytes, dw_section["size"])
                dw_w_host = torch.frombuffer(
                    bytearray(dw_bytes), dtype=torch.bfloat16
                ).reshape(*dw_section["shape"]).clone()
                addrs["CONV_NORM"]       = _dma_section(f"{pre}.conv_norm")
                addrs["FF2_PRE_NORM"]    = _dma_section(f"{pre}.ff2_pre_norm")
                addrs["FF2_POST_NORM"]   = _dma_section(f"{pre}.ff2_post_norm")
                addrs["NORM_OUT"]        = _dma_section(f"{pre}.norm_out")
                for proj in ["FF1_W1", "FF1_W2", "Q_PROJ", "K_PROJ", "V_PROJ",
                              "O_PROJ", "REL_K_PROJ", "CONV_LIN_START", "CONV_LIN_END",
                              "FF2_W1", "FF2_W2"]:
                    addrs[proj] = _alloc_if4(f"{pre}.{proj.lower()}")
                layer_addrs.append(addrs)
                hf_cache.append({"dw_w": dw_w_host})
                cr_src = meta["clip_ranges"][li]
                layer_cr: dict = {}
                for k in ("ff1_w1", "ff1_w2", "q_proj", "k_proj", "v_proj", "o_proj",
                          "conv_lin_start", "conv_lin_end", "ff2_w1", "ff2_w2"):
                    layer_cr[k.upper()] = _to_old_clip(cr_src[k])
                clip_ranges.append(layer_cr)
            self._aud_weight_addrs = layer_addrs
            self._aud_clip_ranges = clip_ranges
            self._aud_hf_layers = hf_cache

            self._aud_output_proj_addrs = _alloc_if4("output_proj")
            self._aud_embedder_proj_addrs = _alloc_if4("embedder_proj")
            self._aud_output_proj_b_addr = _dma_section("output_proj_bias")
        finally:
            f.close()

        audio_weight_end = self._tensor_dram_addr
        AUDIO_WEIGHT_REGION_END = 0x78000000
        assert audio_weight_end <= AUDIO_WEIGHT_REGION_END, (
            f"Audio weights overflowed dedicated region: cursor="
            f"0x{audio_weight_end:X} > 0x{AUDIO_WEIGHT_REGION_END:X}")
        self._tensor_dram_addr = _aud_weight_tensor_cursor_save

        print(f"[Audio] uploaded {self.AUD_LAYERS} layers + subsample + projector "
              f"(weights in params region 0x6c000000-0x{audio_weight_end:X}, "
              f"{(audio_weight_end - 0x6c000000)/(1024*1024):.1f} MB from bin; "
              f"tensor cursor restored to 0x{self._tensor_dram_addr:X})")

    def audio_weight_init(self, hf_model, *, reset_allocator: bool = True) -> None:
        """Upload Gemma4 audio encoder weights to FPGA DRAM.

        Prefers the pre-quantized audio section inside the combined
        weights bin (no HF model, no IF4 quantization). Falls back to
        the legacy HF-model + quantize path only when the bin lacks the
        audio section — but in run_from_bin we refuse the fallback (HF
        model is forbidden).
        """
        # Prefer the combined-bin fast path (HF-free).
        master = getattr(self, "_weights_master", None)
        if master is not None and "audio_section" in master and "manifest" in master["audio_section"]:
            weights_bin_path = os.path.join(self.script_dir, self._weights_bin_rel)
            self._audio_weight_init_from_combined_bin(weights_bin_path, master["audio_section"])
            return
        # No bin section AND no HF: refuse. run_from_bin is HF-free.
        if hf_model is None:
            raise RuntimeError(
                "audio_weight_init: combined-bin audio section not present. "
                "run_from_bin needs the audio_section folded into "
                "params.bin. Regenerate on the build host via "
                "`python gemma4_e2b_test.py` and copy the new weight bin here.")
        self.audio_config_init()
        am = hf_model.model.audio_tower
        bpe = self.bytes_per_element

        if reset_allocator:
            self._aud_tensor_dram_save = self._tensor_dram_addr
            self.reset_tensor_dram_addr()
            print(f"\n[Audio] Tensor allocator reset for audio weights")

        # Route audio weight uploads to the dedicated Weight Audio region
        # (params DRAM 0x6c000000–0x78000000, 192 MB). Stacking 156 MB of
        # audio weights in front of LM activations / encoder intermediates
        # would push tensor DRAM past 0x98000000 and silently corrupt the
        # downstream LM state (see gemma4 audio design note §6).
        _aud_weight_tensor_cursor_save = self._tensor_dram_addr
        self._tensor_dram_addr = 0x6c000000  # Weight Audio region base

        print(f"\n[Audio] Uploading audio encoder weights to DRAM "
              f"({AUDIO_QUANT_PRECISION.upper()} block=64 + BF16 norms) ...")

        def _upload_bf16(name: str, w: torch.Tensor) -> int:
            t = w.detach().cpu().to(torch.bfloat16).contiguous()
            sz = t.numel() * bpe
            addr = self.allocate_tensor_dram(sz)
            self.dma_to_accelerator_memory(addr, t)
            return addr

        def _upload_fp4_batch(named_tensors: list[tuple[str, torch.Tensor]]) -> dict[str, dict]:
            """Quantize many [N, K] BF16 weights in parallel via the canonical
            codec wrapper (selected by AUDIO_QUANT_PRECISION; "if4" by default,
            "fp4" as a byte-compatible fallback), then DMA-upload each
            (scale + data) serially. Returns {key: {'data','scale','shape'}}
            in the same order as named_tensors."""
            for name, w in named_tensors:
                assert w.dim() == 2, f"{name}: expected 2D weight, got {tuple(w.shape)}"
                assert w.shape[1] % 64 == 0, f"{name}: K={w.shape[1]} not divisible by 64"
            tensors_c = [w.contiguous() for _, w in named_tensors]
            quant_results = _parallel_quantize(AUDIO_QUANT_PRECISION, tensors_c)
            out: dict[str, dict] = {}
            for (name, _), w, (data_bytes, scale_bytes) in zip(named_tensors, tensors_c, quant_results):
                N, K = w.shape
                scale_addr = self.allocate_tensor_dram(len(scale_bytes))
                self.dma_write(DMA_DEVICE_H2C, scale_addr, scale_bytes, len(scale_bytes))
                data_addr = self.allocate_tensor_dram(len(data_bytes))
                self.dma_write(DMA_DEVICE_H2C, data_addr, data_bytes, len(data_bytes))
                out[name] = {"data": data_addr, "scale": scale_addr, "shape": (N, K)}
            return out

        # ---- Subsample conv weights ----
        # Host copies (used by audio_subsample_host fallback and as the
        # source for FPGA-side quantization in Phase A).
        sub = am.subsample_conv_projection
        self._aud_sub_w0_conv = sub.layer0.conv.weight.detach().cpu().to(torch.bfloat16)
        self._aud_sub_w0_norm = sub.layer0.norm.weight.detach().cpu().to(torch.bfloat16)
        self._aud_sub_w1_conv = sub.layer1.conv.weight.detach().cpu().to(torch.bfloat16)
        self._aud_sub_w1_norm = sub.layer1.norm.weight.detach().cpu().to(torch.bfloat16)
        self._aud_sub_proj_w  = sub.input_proj_linear.weight.detach().cpu().to(torch.bfloat16)

        # Phase A: FPGA-side subsample weights. Layer-0 conv (out=128, in=1, 3, 3)
        # → flatten to (128, 9), pad K to 64. Layer-1 conv (out=32, in=128, 3, 3)
        # → permute (out, kh, kw, in), pad N from 32 to 64 by DUPLICATING rows
        # 0-31 (Phase A2.3 trick — makes conv1 output cols 32-63 mirror 0-31 so
        # standard layer_norm_core_dram with N=64 is mathematically equivalent
        # to the unbuildable N=32 LN).
        K0_pad = 64
        w0 = self._aud_sub_w0_conv.reshape(128, 9)
        w0_padded = torch.zeros(128, K0_pad, dtype=torch.bfloat16)
        w0_padded[:, :9] = w0
        w1 = self._aud_sub_w1_conv.permute(0, 2, 3, 1).reshape(32, 9 * 128)
        w1_padded = torch.zeros(64, 9 * 128, dtype=torch.bfloat16)
        w1_padded[:32] = w1
        w1_padded[32:] = w1  # duplicate rows 0-31 -> 32-63
        # Phase A2.3: pad proj weight K=1024 → 2048 (zeros in duplicate cols).
        W1_sub = 32
        proj_orig = self._aud_sub_proj_w
        proj_padded = torch.zeros(1024, 2 * 1024, dtype=torch.bfloat16)
        for w1_idx in range(W1_sub):
            proj_padded[:, w1_idx * 64:w1_idx * 64 + 32] = proj_orig[:, w1_idx * 32:(w1_idx + 1) * 32]
        sub_if4_named = [
            ("AUD_SUB_CONV0_W", w0_padded),
            ("AUD_SUB_CONV1_W", w1_padded),
            ("AUD_SUB_PROJ_W",  proj_padded),
        ]
        sub_addrs = _upload_fp4_batch(sub_if4_named)
        self._aud_sub_conv0_addrs = sub_addrs["AUD_SUB_CONV0_W"]
        self._aud_sub_conv1_addrs = sub_addrs["AUD_SUB_CONV1_W"]
        self._aud_sub_proj_addrs  = sub_addrs["AUD_SUB_PROJ_W"]

        self._aud_sub_ln0_gamma_addr = _upload_bf16("aud_sub_ln0_gamma", self._aud_sub_w0_norm)
        # Phase A2.3: gamma_64 = concat(gamma_32, gamma_32) for the duplicated
        # 64-channel LN1 input.
        ln1_gamma_64 = torch.cat([self._aud_sub_w1_norm, self._aud_sub_w1_norm], dim=0)
        self._aud_sub_ln1_gamma_addr = _upload_bf16("aud_sub_ln1_gamma_64", ln1_gamma_64)
        # Identity matrix for FPGA ReLU on (N1_pad, 64) via matmul + clamp.
        self._aud_sub_id_64_addr = _upload_bf16("aud_sub_id_64",
            torch.eye(64, dtype=torch.bfloat16))
        # Trick 9: shared LayerNorm zeros base (mirror id_64; same addr as combined-bin path).
        self._aud_ln_zeros_addr = _upload_bf16("aud_ln_zeros",
            torch.zeros(128, dtype=torch.bfloat16))

        # Phase A2.1: FPGA im2col stage-0 gather matrix G_s0 (parakeet pattern).
        VS = UE_VECTOR_SIZE
        _W_in_s0 = 128
        _W_out_s0 = (_W_in_s0 + 2 - 3) // 2 + 1
        K_g_s0 = ((3 * _W_in_s0 + VS - 1) // VS) * VS
        N_g_s0 = _W_out_s0 * 64
        G_s0 = torch.zeros(N_g_s0, K_g_s0, dtype=torch.bfloat16)
        for kh in range(3):
            for kw in range(3):
                for ow in range(_W_out_s0):
                    col = ow * 2 - 1 + kw
                    if 0 <= col < _W_in_s0:
                        G_s0[ow * 64 + kh * 3 + kw, kh * _W_in_s0 + col] = 1.0
        self._aud_sub_G_s0_addr = _upload_bf16("aud_sub_G_s0", G_s0)
        self._aud_sub_K_g_s0 = K_g_s0
        self._aud_sub_N_g_s0 = N_g_s0

        # ---- Per-layer encoder weights ----
        # Cache BF16 host copies of weights used by host-fallback ops
        # (chunked attention, depthwise conv). These avoid having to
        # dequantize FP4 or re-load HF model later.
        hf_cache: list[dict] = []
        layer_addrs: list[dict] = []
        clip_ranges: list[dict] = []
        for li in range(self.AUD_LAYERS):
            L = am.layers[li]
            addrs: dict = {}
            crs: dict = {}
            hfc: dict = {}
            # Host-only weight cache for the chunked-attn and depthwise host fallbacks.
            hfc["rel_k_w"] = L.self_attn.relative_k_proj.weight.detach().cpu().to(torch.bfloat16)
            hfc["o_w"] = L.self_attn.post.linear.weight.detach().cpu().to(torch.bfloat16)
            hfc["dw_w"] = L.lconv1d.depthwise_conv1d.weight.detach().cpu().to(torch.bfloat16).squeeze(1)
            hf_cache.append(hfc)

            # All BF16 norms / per-block scales for this layer (cheap, inline).
            ff1 = L.feed_forward1
            sa = L.self_attn
            cv = L.lconv1d
            ff2 = L.feed_forward2
            addrs["FF1_PRE_NORM"]    = _upload_bf16("ff1_pre_norm", ff1.pre_layer_norm.weight)
            addrs["FF1_POST_NORM"]   = _upload_bf16("ff1_post_norm", ff1.post_layer_norm.weight)
            addrs["ATTN_PRE_NORM"]   = _upload_bf16("attn_pre_norm", L.norm_pre_attn.weight)
            addrs["PER_DIM_SCALE"]   = _upload_bf16("per_dim_scale", sa.per_dim_scale)
            addrs["ATTN_POST_NORM"]  = _upload_bf16("attn_post_norm", L.norm_post_attn.weight)
            addrs["CONV_PRE_NORM"]   = _upload_bf16("conv_pre_norm", cv.pre_layer_norm.weight)
            # depthwise: weight shape [hidden, 1, k] → store as (hidden, k) BF16
            dw_w = cv.depthwise_conv1d.weight.detach().cpu().to(torch.bfloat16).squeeze(1)
            addrs["CONV_DW_W"]       = _upload_bf16("conv_dw_w", dw_w)
            addrs["CONV_NORM"]       = _upload_bf16("conv_norm", cv.conv_norm.weight)
            addrs["FF2_PRE_NORM"]    = _upload_bf16("ff2_pre_norm", ff2.pre_layer_norm.weight)
            addrs["FF2_POST_NORM"]   = _upload_bf16("ff2_post_norm", ff2.post_layer_norm.weight)
            addrs["NORM_OUT"]        = _upload_bf16("norm_out", L.norm_out.weight)

            # All 11 IF4 weights for this layer batched: parallel-quantize, then
            # serial DMA upload. Same canonical codec wrapper as LM/vision.
            if4_named = [
                ("FF1_W1",         ff1.ffw_layer_1.linear.weight.detach().cpu().to(torch.bfloat16)),
                ("FF1_W2",         ff1.ffw_layer_2.linear.weight.detach().cpu().to(torch.bfloat16)),
                ("Q_PROJ",         sa.q_proj.linear.weight.detach().cpu().to(torch.bfloat16)),
                ("K_PROJ",         sa.k_proj.linear.weight.detach().cpu().to(torch.bfloat16)),
                ("V_PROJ",         sa.v_proj.linear.weight.detach().cpu().to(torch.bfloat16)),
                ("O_PROJ",         sa.post.linear.weight.detach().cpu().to(torch.bfloat16)),
                ("REL_K_PROJ",     sa.relative_k_proj.weight.detach().cpu().to(torch.bfloat16)),
                ("CONV_LIN_START", cv.linear_start.linear.weight.detach().cpu().to(torch.bfloat16)),
                ("CONV_LIN_END",   cv.linear_end.linear.weight.detach().cpu().to(torch.bfloat16)),
                ("FF2_W1",         ff2.ffw_layer_1.linear.weight.detach().cpu().to(torch.bfloat16)),
                ("FF2_W2",         ff2.ffw_layer_2.linear.weight.detach().cpu().to(torch.bfloat16)),
            ]
            addrs.update(_upload_fp4_batch(if4_named))

            # Clip ranges (CPU-only metadata, no DMA).
            crs["FF1_W1"]         = self._extract_clips(ff1.ffw_layer_1)
            crs["FF1_W2"]         = self._extract_clips(ff1.ffw_layer_2)
            crs["Q_PROJ"]         = self._extract_clips(sa.q_proj)
            crs["K_PROJ"]         = self._extract_clips(sa.k_proj)
            crs["V_PROJ"]         = self._extract_clips(sa.v_proj)
            crs["O_PROJ"]         = self._extract_clips(sa.post)
            crs["CONV_LIN_START"] = self._extract_clips(cv.linear_start)
            crs["CONV_LIN_END"]   = self._extract_clips(cv.linear_end)
            crs["FF2_W1"]         = self._extract_clips(ff2.ffw_layer_1)
            crs["FF2_W2"]         = self._extract_clips(ff2.ffw_layer_2)

            layer_addrs.append(addrs)
            clip_ranges.append(crs)

        self._aud_weight_addrs = layer_addrs
        self._aud_clip_ranges = clip_ranges
        self._aud_hf_layers = hf_cache

        # ---- Output projection (1024 → 1536) and multimodal embedder
        # output_proj has bias=True, the embedder embedding_projection has bias=False.
        # Multimodal embedder is on hf_model.model.embed_audio (mirrors embed_vision).
        self._aud_output_proj_w = am.output_proj.weight.detach().cpu().to(torch.bfloat16)
        self._aud_output_proj_b = am.output_proj.bias.detach().cpu().to(torch.bfloat16)
        ea = hf_model.model.embed_audio
        self._aud_embedder_proj_w = ea.embedding_projection.weight.detach().cpu().to(torch.bfloat16)

        # Phase B: FPGA-side output_proj + multimodal embedder.
        embed_if4_named = [
            ("AUD_OUTPUT_PROJ", self._aud_output_proj_w),
            ("AUD_EMBEDDER_PROJ", self._aud_embedder_proj_w),
        ]
        embed_addrs = _upload_fp4_batch(embed_if4_named)
        self._aud_output_proj_addrs = embed_addrs["AUD_OUTPUT_PROJ"]
        self._aud_embedder_proj_addrs = embed_addrs["AUD_EMBEDDER_PROJ"]
        self._aud_output_proj_b_addr = _upload_bf16(
            "aud_output_proj_bias", self._aud_output_proj_b)

        audio_weight_end = self._tensor_dram_addr
        AUDIO_WEIGHT_REGION_END = 0x78000000
        assert audio_weight_end <= AUDIO_WEIGHT_REGION_END, (
            f"Audio weights overflowed dedicated region: cursor="
            f"0x{audio_weight_end:X} > 0x{AUDIO_WEIGHT_REGION_END:X}")
        self._tensor_dram_addr = _aud_weight_tensor_cursor_save

        print(f"[Audio] uploaded {self.AUD_LAYERS} layers + subsample + projector "
              f"(weights in params region 0x6c000000-0x{audio_weight_end:X}, "
              f"{(audio_weight_end - 0x6c000000)/(1024*1024):.1f} MB; "
              f"tensor cursor restored to 0x{self._tensor_dram_addr:X})")

    @staticmethod
    def _extract_clips(clippable_linear) -> dict:
        """Pull (in_min, in_max, out_min, out_max) from a Gemma4ClippableLinear,
        defaulting to ±inf if `use_clipped_linears` is False or bounds aren't loaded.
        """
        cl = clippable_linear
        if not getattr(cl, "use_clipped_linears", False):
            return {"in_min": float("-inf"), "in_max": float("inf"),
                    "out_min": float("-inf"), "out_max": float("inf")}
        return {
            "in_min": float(cl.input_min.item()),
            "in_max": float(cl.input_max.item()),
            "out_min": float(cl.output_min.item()),
            "out_max": float(cl.output_max.item()),
        }

    def audio_tensor_init(self, num_frames: int) -> None:
        """Allocate intermediate DRAM buffers for the Conformer encoder. All
        sized for the *padded* L_pad = ceil(num_frames / 64) * 64 frames.
        """
        self.audio_config_init()
        bpe = self.bytes_per_element
        H = self.AUD_H
        FF = self.AUD_FFN
        VS = UE_VECTOR_SIZE  # 64

        L_pad = ((num_frames + VS - 1) // VS) * VS
        self._aud_num_frames = num_frames
        self._aud_L_pad = L_pad
        print(f"\n[Audio] Allocating audio tensor DRAM for {num_frames} frames (L_pad={L_pad})")

        # Layer I/O double-buffered (so layer i reads from one and writes to the other)
        self.AUD_IO_A = self.allocate_tensor_dram(L_pad * H * bpe)
        self.AUD_IO_B = self.allocate_tensor_dram(L_pad * H * bpe)

        # Norm output (used as input to all post-norm matmuls in a layer)
        self.AUD_NORM_OUT = self.allocate_tensor_dram(L_pad * H * bpe)

        # Saved residual for half-step macaron
        self.AUD_RESIDUAL = self.allocate_tensor_dram(L_pad * H * bpe)

        # FFN intermediate (S × 4*H)
        self.AUD_FFN_MID = self.allocate_tensor_dram(L_pad * FF * bpe)
        # FFN second-stage output (back to S × H)
        self.AUD_FFN_OUT = self.allocate_tensor_dram(L_pad * H * bpe)

        # SiLU scratch — needed because silu_core_dram reads x WHILE writing
        # sigmoid(x), so input and output buffers must be distinct. Same size
        # as AUD_FFN_MID (L_pad × FF).
        self.AUD_SILU_OUT = self.allocate_tensor_dram(L_pad * FF * bpe)

        # Identity matrices for SiLU/GLU sigmoid-via-matmul. We need TWO
        # because matmat reads B with row stride = N (the matmul output dim).
        # If we passed an FFxFF identity to a 1024x1024 matmul, the row stride
        # mismatch would corrupt the result. Allocate one per N value used:
        #   AUD_IDENTITY_FF: 4096x4096 — used by FFN1/FFN2 SiLU
        #   AUD_IDENTITY_H : 1024x1024 — used by conv module SiLU and GLU
        self.AUD_IDENTITY_FF = self.allocate_tensor_dram(FF * FF * bpe)
        self.dma_to_accelerator_memory(self.AUD_IDENTITY_FF,
            torch.eye(FF, dtype=torch.bfloat16).contiguous())
        self.AUD_IDENTITY_H = self.allocate_tensor_dram(H * H * bpe)
        self.dma_to_accelerator_memory(self.AUD_IDENTITY_H,
            torch.eye(H, dtype=torch.bfloat16).contiguous())
        # 64×64 identity for FPGA standalone clamp via matmul-w/-identity
        # trick (see _emit_clamp_dram_to_dram). Required to replace
        # _host_clip_dram with a pure-FPGA op (Phase 1).
        self.AUD_IDENTITY_64 = self.allocate_tensor_dram(VS * VS * bpe)
        self.dma_to_accelerator_memory(self.AUD_IDENTITY_64,
            torch.eye(VS, dtype=torch.bfloat16).contiguous())
        # Alias used by the compare script's audio FFN1 verification path.
        self.AUD_IDENTITY = self.AUD_IDENTITY_FF

        # Attention scratch buffers (small)
        HD = self.AUD_HEAD_DIM
        NH = self.AUD_HEADS
        self.AUD_Q   = self.allocate_tensor_dram(L_pad * NH * HD * bpe)
        self.AUD_K   = self.allocate_tensor_dram(L_pad * NH * HD * bpe)
        self.AUD_V   = self.allocate_tensor_dram(L_pad * NH * HD * bpe)
        self.AUD_REL_K_PROJ_OUT = self.allocate_tensor_dram(self.AUD_CTX_LEFT * NH * HD * bpe)
        self.AUD_ATTN_OUT = self.allocate_tensor_dram(L_pad * H * bpe)

        # Phase 2A: depthwise conv1d FPGA buffers (4-tap shifted-eltwise).
        # AUD_DW_ZERO_KM1 holds (K-1) rows of zeros for the top-of-shifted-buf
        # reset before each layer; the SHIFT/SCRATCH scratch is reused from
        # AUD_FFN_MID (L_pad × FF = 4× L_pad × H) at compile time.
        K = self.AUD_CONV_K
        if K > 1:
            self.AUD_DW_ZERO_KM1 = self.allocate_tensor_dram((K - 1) * H * bpe)
            self.dma_to_accelerator_memory(
                self.AUD_DW_ZERO_KM1,
                torch.zeros((K - 1) * H, dtype=torch.bfloat16).contiguous())
            # Per-(layer, tap) tiled weight buffers: each (L_pad, H) bf16,
            # built by broadcasting w[c, t] across all L_pad rows. Stored into
            # _aud_weight_addrs[li]["CONV_DW_TAP_TILES"][t]. Audio weight init
            # must have already cached the host kernel.
            assert hasattr(self, "_aud_hf_layers"), \
                "audio_weight_init must run BEFORE audio_tensor_init"
            for li in range(self.AUD_LAYERS):
                dw_w = self._aud_hf_layers[li]["dw_w"]  # (H, K) bf16
                tile_addrs = []
                for t in range(K):
                    addr = self.allocate_tensor_dram(L_pad * H * bpe)
                    # tile[r, c] = w[c, t]
                    tile = dw_w[:, t].to(torch.bfloat16).contiguous()  # (H,)
                    tile = tile.unsqueeze(0).expand(L_pad, H).contiguous()
                    self.dma_to_accelerator_memory(addr, tile)
                    tile_addrs.append(addr)
                self._aud_weight_addrs[li]["CONV_DW_TAP_TILES"] = tile_addrs

        # Phase 2B.b: per-layer Q-scale tile = q_scale * softplus(per_dim_scale),
        # tiled to (L_pad, H) with the (HD,) vector broadcast across heads.
        # eltwise_mul AUD_Q × Q_SCALE_TILE reproduces:
        #   Q[r, h, d] *= q_scale * softplus(per_dim_scale[d])
        HD = self.AUD_HEAD_DIM
        NH = self.AUD_HEADS
        for li in range(self.AUD_LAYERS):
            pds_addr = self._aud_weight_addrs[li]["PER_DIM_SCALE"]
            pds = self.dma_from_accelerator_memory(pds_addr, (HD,)).cpu().float()
            scale_vec = (F.softplus(pds) * self.AUD_Q_SCALE).to(torch.bfloat16)  # (HD,)
            row_vec = scale_vec.repeat(NH).contiguous()                          # (H,)
            tile = row_vec.unsqueeze(0).expand(L_pad, H).contiguous()
            addr = self.allocate_tensor_dram(L_pad * H * bpe)
            self.dma_to_accelerator_memory(addr, tile)
            self._aud_weight_addrs[li]["Q_SCALE_TILE"] = addr

        # Phase 2B.c: FPGA chunked attention scratch + pre-baked constants.
        # ──────────────────────────────────────────────────────────────────
        chunk = self.AUD_CHUNK                    # 12
        ctx_size = self.AUD_CTX                   # 24
        # ctx_pad must satisfy ctx_pad >= (chunk - 1) + num_pos_pad so the per-row
        # rel-shift can write num_pos_pad=VS=64 elements at offset r in [0, chunk)
        # without overflowing. Round to next VS multiple. For chunk=12, num_pos_pad=64,
        # that's 75 -> 128. Per-row reads are VS-aligned, destination offset is
        # row index r (not VS-aligned, which the FPGA permits).
        ctx_pad = ((chunk - 1 + VS + VS - 1) // VS) * VS
        max_past = self.AUD_CTX_LEFT - 1          # 12
        max_future = self.AUD_CTX_RIGHT           # 0
        num_pos = self.AUD_CTX_LEFT               # 13
        num_pos_pad = VS                          # 64
        T = num_frames
        num_blocks = (T + chunk - 1) // chunk
        chunk_pad = VS                            # 64 (chunk padded for matmul A row count)
        T_pad_padded = ((num_blocks * chunk + VS - 1) // VS) * VS
        # matrix_bd matmul writes N=bd_unshifted_N columns per row, structured as:
        #   cols [0, chunk_pad): zero  (so per-row rel-shift reads from offset
        #                               chunk_pad-r get pre-pad zeros for c < r)
        #   cols [chunk_pad, chunk_pad+num_pos_pad): real Q @ rel_k row
        #   cols [chunk_pad+num_pos_pad, bd_unshifted_N): zero (post-pad zeros
        #                                                       for c >= r+num_pos_pad)
        # The zero bands are baked into the per-head REL_K_T row layout, so the
        # matmul output naturally carries the structure with no separate fill step.
        bd_unshifted_N = ((chunk_pad * 2 + num_pos_pad + VS - 1) // VS) * VS
        self._aud_num_blocks = num_blocks
        self._aud_ctx_pad = ctx_pad
        self._aud_num_pos_pad = num_pos_pad
        self._aud_chunk_pad = chunk_pad
        self._aud_T_pad_padded = T_pad_padded
        self._aud_bd_unshifted_N = bd_unshifted_N

        # K_PADDED / V_PADDED: max_past || L_pad || (max_future + chunk - 1) padded
        L_padded_full_unaligned = max_past + L_pad + max_future + chunk - 1
        L_padded_full = ((L_padded_full_unaligned + VS - 1) // VS) * VS
        self._aud_L_padded_full = L_padded_full
        self.AUD_K_PADDED = self.allocate_tensor_dram(L_padded_full * H * bpe)
        self.AUD_V_PADDED = self.allocate_tensor_dram(L_padded_full * H * bpe)
        # Zero the whole padded buffer once (pad regions remain zero across layers,
        # the middle gets overwritten by AUD_K / AUD_V each layer).
        zeros_buf = torch.zeros(L_padded_full * H, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.AUD_K_PADDED, zeros_buf)
        self.dma_to_accelerator_memory(self.AUD_V_PADDED, zeros_buf)

        # Per-block K and V context blocks: (num_blocks, ctx_pad=64, H).
        self.AUD_K_CTX_BLOCKS = self.allocate_tensor_dram(num_blocks * ctx_pad * H * bpe)
        self.AUD_V_CTX_BLOCKS = self.allocate_tensor_dram(num_blocks * ctx_pad * H * bpe)
        # Pre-zero so trailing rows (ctx_size..ctx_pad) are always zero, regardless of layer.
        block_zeros = torch.zeros(num_blocks * ctx_pad * H, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.AUD_K_CTX_BLOCKS, block_zeros)
        self.dma_to_accelerator_memory(self.AUD_V_CTX_BLOCKS, block_zeros)

        # K_CTX_T_BLOCKS: per-(block, head) head slice of K_CTX_BLOCKS stored as
        # (num_blocks, NH, ctx_pad, HD). NOTE: not transposed; this IS the FPGA-
        # native B layout (N=ctx_pad, K=HD) so matmul A @ B^T = Q @ K^T.
        self.AUD_K_CTX_T_BLOCKS = self.allocate_tensor_dram(
            num_blocks * NH * ctx_pad * HD * bpe)

        # V_CTX_T_BLOCKS: per-(block, head) V_ctx ACTUALLY TRANSPOSED to (HD, ctx_pad).
        # Built via matmul-with-AUD_IDENTITY_HD trick (FPGA strided DMA can't do the
        # column-stride read needed for an explicit transpose). This is the FPGA-
        # native B layout (N=HD, K=ctx_pad) for the attn @ V matmul.
        self.AUD_V_CTX_T_BLOCKS = self.allocate_tensor_dram(
            num_blocks * NH * HD * ctx_pad * bpe)

        # Q_HEAD_BLOCK scratch: (chunk_pad=64, HD). Reused per (block, head).
        self.AUD_Q_HEAD_BLOCK = self.allocate_tensor_dram(chunk_pad * HD * bpe)

        # K_BLOCK_HEAD scratch: (ctx_pad, HD). Reused per (block, head).
        self.AUD_K_BLOCK_HEAD = self.allocate_tensor_dram(ctx_pad * HD * bpe)

        # V_BLOCK_HEAD scratch: (ctx_pad, HD). Reused per (block, head) — staging
        # for the V_ctx_T transpose-via-matmul.
        self.AUD_V_BLOCK_HEAD = self.allocate_tensor_dram(ctx_pad * HD * bpe)

        # Identity for HD-sized matmul-with-identity (e.g. for transpose
        # via permute, sigmoid via matmul, etc.). HD=128 here.
        self.AUD_IDENTITY_HD = self.allocate_tensor_dram(HD * HD * bpe)
        self.dma_to_accelerator_memory(self.AUD_IDENTITY_HD,
            torch.eye(HD, dtype=torch.bfloat16).contiguous())
        # ctx_pad-sized identity for tanh-via-identity-matmul on (chunk_pad, ctx_pad)
        self.AUD_IDENTITY_CTX = self.allocate_tensor_dram(ctx_pad * ctx_pad * bpe)
        self.dma_to_accelerator_memory(self.AUD_IDENTITY_CTX,
            torch.eye(ctx_pad, dtype=torch.bfloat16).contiguous())

        # MATRIX_AC: (num_blocks, NH, chunk_pad, ctx_pad). First `chunk` rows valid.
        self.AUD_MATRIX_AC = self.allocate_tensor_dram(
            num_blocks * NH * chunk_pad * ctx_pad * bpe)

        # MATRIX_BD_UNSHIFTED: (NH, T_pad_padded, bd_unshifted_N=192). Per-head Q@rel_k_T
        # output with zero pre/post padding bands so per-row rel-shift can do
        # non-aligned source reads while keeping VS-aligned destination writes.
        self.AUD_MATRIX_BD_UNSHIFTED = self.allocate_tensor_dram(
            NH * T_pad_padded * bd_unshifted_N * bpe)

        # MATRIX_BD_SHIFTED: (num_blocks, NH, chunk_pad, ctx_pad) — rel-shifted.
        self.AUD_MATRIX_BD_SHIFTED = self.allocate_tensor_dram(
            num_blocks * NH * chunk_pad * ctx_pad * bpe)
        # Pre-zero so unfilled cells stay zero. Rel-shift writes only valid range.
        self.dma_to_accelerator_memory(self.AUD_MATRIX_BD_SHIFTED,
            torch.zeros(num_blocks * NH * chunk_pad * ctx_pad, dtype=torch.bfloat16))

        # REL_K_T per layer: per-head slice of rel_k = pos_emb @ relative_k_proj,
        # stored as (NH, bd_unshifted_N, HD) with zero pre-pad + real + zero post-pad
        # row bands (see bd_unshifted_N comment). This is the native B layout
        # (N=bd_unshifted_N, K=HD) for matrix_bd matmul.
        self.AUD_REL_K_T = self.allocate_tensor_dram(NH * bd_unshifted_N * HD * bpe)
        # Pre-zero the entire REL_K_T buffer so the pre/post-pad bands stay zero
        # across layer iterations; only rows [chunk_pad, chunk_pad+num_pos_pad)
        # are rewritten by the per-layer build step.
        self.dma_to_accelerator_memory(self.AUD_REL_K_T,
            torch.zeros(NH * bd_unshifted_N * HD, dtype=torch.bfloat16).contiguous())

        # REL_K_OUT scratch (per layer): pos_emb_padded @ rel_k_proj -> (num_pos_pad, H).
        self.AUD_REL_K_OUT = self.allocate_tensor_dram(num_pos_pad * H * bpe)

        # Per-head Q_HEAD_FULL scratch for matrix_bd matmul: (T_pad_padded, HD).
        # Rows [0, L_pad) = AUD_Q[:, h*HD:(h+1)*HD]; rows [L_pad, T_pad_padded) = 0.
        self.AUD_Q_HEAD_FULL = self.allocate_tensor_dram(T_pad_padded * HD * bpe)
        # Pre-zero so the tail rows stay zero across iterations.
        self.dma_to_accelerator_memory(self.AUD_Q_HEAD_FULL,
            torch.zeros(T_pad_padded * HD, dtype=torch.bfloat16).contiguous())

        # POS_EMB_PADDED: (num_pos_pad=64, H). First num_pos rows = audio_rel_pos_host(),
        # rest zero. Uploaded once at init time (shared across layers).
        pos_emb = self.audio_rel_pos_host()  # (num_pos, H) bf16
        pos_emb_pad = torch.zeros(num_pos_pad, H, dtype=torch.bfloat16)
        pos_emb_pad[:num_pos] = pos_emb
        self.AUD_POS_EMB_PADDED = self.allocate_tensor_dram(num_pos_pad * H * bpe)
        self.dma_to_accelerator_memory(self.AUD_POS_EMB_PADDED, pos_emb_pad.contiguous())

        # MASK_ADDEND: (num_blocks, NH, chunk_pad, ctx_pad) bf16 — 0 where valid,
        # -1e9 elsewhere. Tiled across heads (identical per-head) so it can be
        # eltwise-added to logits via a single (num_blocks*NH*chunk_pad, ctx_pad)
        # tensor op. Memory is 8× a per-(b,c,c) tile but simplifies the addition.
        mask_5d = self._aud_make_blocked_mask(T, num_blocks, chunk, max_past, max_future)
        mask_b = mask_5d.view(num_blocks, chunk, ctx_size)
        invalid_val = float(self.AUD_INVALID_LOGIT)
        addend_bcc = torch.full((num_blocks, chunk_pad, ctx_pad), invalid_val, dtype=torch.bfloat16)
        addend_bcc[:, :chunk, :ctx_size] = torch.where(
            mask_b, torch.tensor(0.0, dtype=torch.bfloat16),
                     torch.tensor(invalid_val, dtype=torch.bfloat16))
        # Tile to (num_blocks, NH, chunk_pad, ctx_pad)
        addend = addend_bcc.unsqueeze(1).expand(num_blocks, NH, chunk_pad, ctx_pad).contiguous()
        self.AUD_MASK_ADDEND = self.allocate_tensor_dram(
            num_blocks * NH * chunk_pad * ctx_pad * bpe)
        self.dma_to_accelerator_memory(self.AUD_MASK_ADDEND, addend)

        # LOGITS scratch (reused per (block, head)): (chunk_pad, ctx_pad).
        self.AUD_LOGITS_BH = self.allocate_tensor_dram(chunk_pad * ctx_pad * bpe)
        # OUT scratch per (block, head): (chunk_pad, HD).
        self.AUD_ATTN_OUT_BH = self.allocate_tensor_dram(chunk_pad * HD * bpe)

        # AUD_Q_HEAD_BLOCK / AUD_K_BLOCK_HEAD: bottom rows (above the real data
        # rows) must stay zero across iterations. Initialize once.
        self.dma_to_accelerator_memory(self.AUD_Q_HEAD_BLOCK,
            torch.zeros(chunk_pad * HD, dtype=torch.bfloat16).contiguous())
        self.dma_to_accelerator_memory(self.AUD_K_BLOCK_HEAD,
            torch.zeros(ctx_pad * HD, dtype=torch.bfloat16).contiguous())

        # Phase B: FPGA output_proj + multimodal embedder.
        OUT_DIM = self.AUD_OUT_DIM  # 1536
        # Output of output_proj: (L_pad, OUT_DIM). First T_sub rows valid.
        self.AUD_FEATURES_MID = self.allocate_tensor_dram(L_pad * OUT_DIM * bpe)
        # Output of multimodal embedder: same shape; the final audio_features.
        self.AUD_FEATURES_FINAL = self.allocate_tensor_dram(L_pad * OUT_DIM * bpe)
        # All-ones gamma for the embedder's RMSNorm (with_scale=False in HF).
        self.AUD_EMB_ONES_GAMMA = self.allocate_tensor_dram(OUT_DIM * bpe)
        self.dma_to_accelerator_memory(self.AUD_EMB_ONES_GAMMA,
            torch.ones(OUT_DIM, dtype=torch.bfloat16).contiguous())

        # Phase 2B.c step 3: per-r shift matrices M_r for rel-shift via matmul.
        # M_r in (N=ctx_pad, K=num_pos_pad) layout has M_r[c, p]=1 if c == p+r
        # AND p < num_pos AND c < r+num_pos (HF rel-shift output is zero outside
        # the [r, r+num_pos) column range).
        # Sidesteps the 32-byte AXI src alignment by using only matmul (which is
        # well-defined for arbitrary M/K/N as long as K and N are 64-aligned).
        # Stored at AUD_REL_SHIFT_M[r] = base + r * ctx_pad * num_pos_pad * bpe.
        self.AUD_REL_SHIFT_M = self.allocate_tensor_dram(chunk * ctx_pad * num_pos_pad * bpe)
        shift_tensor = torch.zeros(chunk, ctx_pad, num_pos_pad, dtype=torch.bfloat16)
        for r in range(chunk):
            for p in range(num_pos):
                c = p + r
                if c < ctx_pad:
                    shift_tensor[r, c, p] = 1.0
        self.dma_to_accelerator_memory(self.AUD_REL_SHIFT_M,
            shift_tensor.contiguous().view(-1))

        # Zero source for trailing-row fill in AUD_Q_HEAD_BLOCK on partial last block.
        self.AUD_ZEROS_CHUNK_HD = self.allocate_tensor_dram(chunk_pad * HD * bpe)
        self.dma_to_accelerator_memory(self.AUD_ZEROS_CHUNK_HD,
            torch.zeros(chunk_pad * HD, dtype=torch.bfloat16).contiguous())

        print(f"[Audio] tensor DRAM usage: {self.get_tensor_dram_usage()/(1024*1024):.1f} MB")

    def audio_subsample_fpga(self,
                              input_features: torch.Tensor,
                              input_features_mask: torch.Tensor | None = None
                              ) -> tuple[torch.Tensor, torch.Tensor]:
        """FPGA port of audio_subsample_host (synced from gemma4_e2b_test).

        At runtime in run_from_bin we call this in capture-suppressed mode
        (capture_buffer=None) so the FPGA ISA emits become no-ops while the
        host DMAs (mel upload, scratch zero) and the lazy buffer allocations
        still run — that primes the FPGA DRAM in exactly the state the
        captured audio ISA in the unified bin expects.
        """
        if not hasattr(self, "_aud_sub_conv0_addrs"):
            raise RuntimeError("audio_weight_init must run before audio_subsample_fpga")
        H_OUT = self.AUD_H
        bpe = self.bytes_per_element
        x = input_features.detach().cpu()
        mask = input_features_mask
        if x.dim() != 3 or x.shape[0] != 1:
            raise RuntimeError(f"audio_subsample_fpga only supports B=1; got {tuple(x.shape)}")
        T_raw = int(x.shape[1])
        N_MELS = int(x.shape[2])
        assert N_MELS == 128, f"expected n_mels=128, got {N_MELS}"
        H0 = (T_raw + 2 - 3) // 2 + 1
        W0 = (N_MELS + 2 - 3) // 2 + 1
        H1 = (H0 + 2 - 3) // 2 + 1
        W1 = (W0 + 2 - 3) // 2 + 1
        N0 = H0 * W0
        N1 = H1 * W1
        N1_pad = ((N1 + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
        N0_pad = ((N0 + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE

        if mask is not None:
            x = x * mask[:, :, None].to(x.dtype)
        x = x.to(torch.bfloat16).squeeze(0).contiguous()

        K_g_s0 = self._aud_sub_K_g_s0
        N_g_s0 = self._aud_sub_N_g_s0
        H0_pad = ((H0 + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
        patches0_buf_bytes = H0_pad * N_g_s0 * bpe
        T_raw_pad = T_raw + (T_raw & 1)

        cache_key = (N0_pad, N1_pad, H0_pad, T_raw_pad)
        if getattr(self, "_aud_sub_buf_key", None) != cache_key:
            self._aud_sub_buf_key = cache_key
            self.AUD_SUB_INPUT = self.allocate_tensor_dram(T_raw_pad * 128 * bpe)
            self.AUD_SUB_R_COMBINED = self.allocate_tensor_dram(H0_pad * K_g_s0 * bpe)
            self.AUD_SUB_PATCHES0 = self.allocate_tensor_dram(patches0_buf_bytes)
            self.AUD_SUB_PATCHES1 = self.allocate_tensor_dram(N1_pad * 1152 * bpe)
            self.AUD_SUB_OUT0 = self.allocate_tensor_dram(N0_pad * 128 * bpe)
            self.AUD_SUB_OUT1 = self.allocate_tensor_dram(N1_pad * 64 * bpe)
            self.AUD_SUB_ID_128 = self.allocate_tensor_dram(128 * 128 * bpe)
            self.dma_to_accelerator_memory(self.AUD_SUB_ID_128,
                torch.eye(128, dtype=torch.bfloat16).contiguous())
            self.AUD_SUB_OUT1_COMPACT = self.allocate_tensor_dram(N1_pad * 32 * bpe)
            self.AUD_SUB_FLAT = self.allocate_tensor_dram(N1_pad * 1024 * bpe)

        x_input = torch.zeros(T_raw_pad, 128, dtype=torch.bfloat16)
        x_input[:T_raw] = x
        self.dma_to_accelerator_memory(self.AUD_SUB_INPUT, x_input.contiguous())
        self.dma_to_accelerator_memory(self.AUD_SUB_R_COMBINED,
            torch.zeros(H0_pad * K_g_s0, dtype=torch.bfloat16))

        self._compile_and_run_single("aud_sub_im2col_s0",
            lambda: self._emit_aud_sub_im2col_s0(T_raw, H0, H0_pad))

        c0 = self._aud_sub_conv0_addrs
        self._compile_and_run_single("aud_sub_conv0", lambda: self.matmat_mul_core(
            M=N0_pad, K=64, N=128,
            A_DRAM_ADDR=self.AUD_SUB_PATCHES0,
            B_DRAM_ADDR=c0["data"],
            OUTPUT_DRAM_ADDR=self.AUD_SUB_OUT0,
            is_B_quantized=True, data_type=TYPE.IF4,
            SCALE_DRAM_ADDR=c0["scale"],
            gpr_M_reg=self._aud_mm_gpr(M=N0_pad)))
        self._compile_and_run_single("aud_sub_ln0", lambda: self.layer_norm_core_dram(
            M=N0_pad, N=128,
            A_DRAM_ADDR=self.AUD_SUB_OUT0,
            OUTPUT_DRAM_ADDR=self.AUD_SUB_OUT0,
            GAMMA_DRAM_ADDR=self._aud_sub_ln0_gamma_addr,
            ZEROS_DRAM_ADDR=self._aud_ln_zeros_addr))
        self._compile_and_run_single("aud_sub_relu0", lambda: self.matmat_mul_core(
            M=N0_pad, K=128, N=128,
            A_DRAM_ADDR=self.AUD_SUB_OUT0,
            B_DRAM_ADDR=self.AUD_SUB_ID_128,
            OUTPUT_DRAM_ADDR=self.AUD_SUB_OUT0,
            clamp_enable=True,
            gpr_M_reg=self._aud_mm_gpr(M=N0_pad)))

        self.dma_to_accelerator_memory(self.AUD_SUB_PATCHES1,
            torch.zeros(N1_pad * 1152, dtype=torch.bfloat16))
        self._compile_and_run_single("aud_sub_im2col_s1",
            lambda: self._emit_aud_sub_im2col_s1(H0, W0, H1, W1))

        c1 = self._aud_sub_conv1_addrs
        self._compile_and_run_single("aud_sub_conv1", lambda: self.matmat_mul_core(
            M=N1_pad, K=1152, N=64,
            A_DRAM_ADDR=self.AUD_SUB_PATCHES1,
            B_DRAM_ADDR=c1["data"],
            OUTPUT_DRAM_ADDR=self.AUD_SUB_OUT1,
            is_B_quantized=True, data_type=TYPE.IF4,
            SCALE_DRAM_ADDR=c1["scale"],
            gpr_M_reg=self._aud_mm_gpr(M=N1_pad)))
        self._compile_and_run_single("aud_sub_ln1", lambda: self.layer_norm_core_dram(
            M=N1_pad, N=64,
            A_DRAM_ADDR=self.AUD_SUB_OUT1,
            OUTPUT_DRAM_ADDR=self.AUD_SUB_OUT1,
            GAMMA_DRAM_ADDR=self._aud_sub_ln1_gamma_addr,
            ZEROS_DRAM_ADDR=self._aud_ln_zeros_addr))
        self._compile_and_run_single("aud_sub_relu1", lambda: self.matmat_mul_core(
            M=N1_pad, K=64, N=64,
            A_DRAM_ADDR=self.AUD_SUB_OUT1,
            B_DRAM_ADDR=self._aud_sub_id_64_addr,
            OUTPUT_DRAM_ADDR=self.AUD_SUB_OUT1,
            clamp_enable=True,
            gpr_M_reg=self._aud_mm_gpr(M=N1_pad)))

        mask_s2 = mask[:, ::4] if mask is not None else None

        proj = self._aud_sub_proj_addrs
        H1_pad = ((H1 + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
        in_oneshot = getattr(self, "_oneshot_mode", False)
        proj_out_addr = self.AUD_IO_A if in_oneshot else self.AUD_SUB_OUT0
        self._compile_and_run_single("aud_sub_proj", lambda: self.matmat_mul_core(
            M=H1_pad, K=2 * 1024, N=1024,
            A_DRAM_ADDR=self.AUD_SUB_OUT1,
            B_DRAM_ADDR=proj["data"],
            OUTPUT_DRAM_ADDR=proj_out_addr,
            is_B_quantized=True, data_type=TYPE.IF4,
            SCALE_DRAM_ADDR=proj["scale"],
            gpr_M_reg=self._aud_mm_gpr(M=H1_pad)))

        if in_oneshot:
            return None, mask_s2

        hidden = self.dma_from_accelerator_memory(
            self.AUD_SUB_OUT0, (H1_pad, 1024)).cpu()[:H1].to(torch.bfloat16).unsqueeze(0)
        return hidden, mask_s2

    def _emit_aud_sub_im2col_s0(self, T_raw: int, H0: int, H0_pad: int) -> None:
        """FPGA emitter for stage-0 im2col (parakeet pattern).
        Mirror of gemma4_e2b_test.py's _emit_aud_sub_im2col_s0."""
        bpe_local = self.bytes_per_element
        W_in_local = 128
        stride = 2
        padding = 1
        row_bytes = W_in_local * bpe_local
        K_g_s0 = self._aud_sub_K_g_s0
        N_g_s0 = self._aud_sub_N_g_s0
        for kh in range(3):
            oh_start = max(0, (padding - kh + stride - 1) // stride)
            oh_end = min(H0, (T_raw + padding - kh + stride - 1) // stride)
            n_rows = oh_end - oh_start
            if n_rows <= 0:
                continue
            first_input_row = oh_start * stride - padding + kh
            read_bytes = n_rows * row_bytes
            max_read = (URAM_NEAR_FULL_SIZE // row_bytes) * row_bytes
            offset = 0
            while offset < read_bytes:
                chunk = min(read_bytes - offset, max_read)
                src_row = first_input_row + (offset // row_bytes) * stride
                oh_base = oh_start + offset // row_bytes
                src = self.AUD_SUB_INPUT + src_row * row_bytes
                self.ue_memcpy_from_dram(
                    src, chunk, 0, URAM_START_ADDR,
                    URAM_SECTION.URAM_A.value,
                    stride_bytes_per_chunk=row_bytes,
                    stride_jump_bytes=stride * row_bytes)
                dst = (self.AUD_SUB_R_COMBINED
                       + oh_base * K_g_s0 * bpe_local
                       + kh * W_in_local * bpe_local)
                self.ue_memcpy_to_dram(
                    0, URAM_SECTION.URAM_A.value, URAM_START_ADDR,
                    dst, chunk,
                    stride_bytes_per_chunk=row_bytes,
                    stride_jump_bytes=K_g_s0 * bpe_local)
                offset += chunk
        self.matmat_mul_core(
            M=H0_pad, K=K_g_s0, N=N_g_s0,
            A_DRAM_ADDR=self.AUD_SUB_R_COMBINED,
            B_DRAM_ADDR=self._aud_sub_G_s0_addr,
            OUTPUT_DRAM_ADDR=self.AUD_SUB_PATCHES0)

    def _emit_aud_sub_im2col_s1(self, H0: int, W0: int, H1: int, W1: int) -> None:
        """FPGA emitter for stage-1 im2col (multi-channel).
        Mirror of gemma4_e2b_test.py's _emit_aud_sub_im2col_s1."""
        bpe_local = self.bytes_per_element
        chunk_bytes = 128 * bpe_local
        src_stride = 2 * chunk_bytes
        dst_stride = 1152 * bpe_local
        for kh in range(3):
            oh1_start = max(0, (1 - kh + 1) // 2)
            oh1_end = min(H1, (H0 + 2 - kh) // 2)
            for kw in range(3):
                ow1_start = max(0, (1 - kw + 1) // 2)
                ow1_end = min(W1, (W0 + 2 - kw) // 2)
                n_ow1 = ow1_end - ow1_start
                if n_ow1 <= 0:
                    continue
                slot = kh * 3 + kw
                chunk_total_bytes = n_ow1 * chunk_bytes
                for oh1 in range(oh1_start, oh1_end):
                    in_row = oh1 * 2 - 1 + kh
                    in_col_start = ow1_start * 2 - 1 + kw
                    src = (self.AUD_SUB_OUT0
                           + (in_row * W0 + in_col_start) * chunk_bytes)
                    self.ue_memcpy_from_dram(
                        src, chunk_total_bytes, 0, URAM_START_ADDR,
                        URAM_SECTION.URAM_A.value,
                        stride_bytes_per_chunk=chunk_bytes,
                        stride_jump_bytes=src_stride)
                    dst = (self.AUD_SUB_PATCHES1
                           + ((oh1 * W1 + ow1_start) * 1152 + slot * 128) * bpe_local)
                    self.ue_memcpy_to_dram(
                        0, URAM_SECTION.URAM_A.value, URAM_START_ADDR,
                        dst, chunk_total_bytes,
                        stride_bytes_per_chunk=chunk_bytes,
                        stride_jump_bytes=dst_stride)

    def _emit_aud_embed_project_chain(self) -> None:
        """Phase B embed+projector ISA emitter. Reads from AUD_IO_A (encoder
        output), writes to AUD_FEATURES_FINAL via AUD_FEATURES_MID."""
        H = self.AUD_H
        OUT_DIM = self.AUD_OUT_DIM
        L_pad = self._aud_L_pad

        op = self._aud_output_proj_addrs
        self._compile_and_run_single("aud_embed_output_proj",
            lambda: self.matmat_mul_core(
                M=L_pad, K=H, N=OUT_DIM,
                A_DRAM_ADDR=self.AUD_IO_A,
                B_DRAM_ADDR=op["data"],
                OUTPUT_DRAM_ADDR=self.AUD_FEATURES_MID,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=op["scale"],
                C_DRAM_ADDR=self._aud_output_proj_b_addr,
                bias_mode="broadcast_N"))

        # LEGACY (no gpr_M_reg) — PBI on these once-run embed ops silently
        # diverges in the one-shot bin (passes per-sub-op compare but corrupts
        # features end-to-end). Mirrors gemma4_e2b_test.py + e4b.
        self._compile_and_run_single("aud_embed_rmsnorm",
            lambda: self.rms_norm_core_dram(
                M=L_pad, N=OUT_DIM,
                A_DRAM_ADDR=self.AUD_FEATURES_MID,
                OUTPUT_DRAM_ADDR=self.AUD_FEATURES_MID,
                GAMMA_DRAM_ADDR=self.AUD_EMB_ONES_GAMMA))

        em = self._aud_embedder_proj_addrs
        self._compile_and_run_single("aud_embed_emb_proj",
            lambda: self.matmat_mul_core(
                M=L_pad, K=OUT_DIM, N=OUT_DIM,
                A_DRAM_ADDR=self.AUD_FEATURES_MID,
                B_DRAM_ADDR=em["data"],
                OUTPUT_DRAM_ADDR=self.AUD_FEATURES_FINAL,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=em["scale"]))

    def audio_subsample_host(self,
                              input_features: torch.Tensor,
                              input_features_mask: torch.Tensor | None = None
                              ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the SubSampleConvProjection stem on host. Mirrors
        Gemma4AudioSubSampleConvProjection.forward.

        Inputs:
            input_features: [B, T, n_mels] from the feature extractor (B=1).
            input_features_mask: [B, T] boolean valid-frame mask.

        Returns:
            hidden_states: [B, T_sub, hidden_size] post-subsample BF16
            mask:          [B, T_sub] downsampled mask
        """
        self.audio_config_init()
        if not hasattr(self, "_aud_sub_w0_conv"):
            raise RuntimeError("audio_weight_init must be called before audio_subsample_host")

        with torch.no_grad():
            x = input_features.to(torch.bfloat16).unsqueeze(1)  # [B, 1, T, n_mels]
            mask = input_features_mask
            # ---- layer 0 ----
            if mask is not None:
                x = x * mask[:, None, :, None].to(x.dtype)
            x = F.conv2d(x, self._aud_sub_w0_conv, bias=None, stride=2, padding=1)
            # LayerNorm over channel dim (out_channels). HF runs it as
            # norm(x.permute(0,2,3,1)).permute(0,3,1,2). bias=False.
            x_perm = x.permute(0, 2, 3, 1).float()
            x_perm = F.layer_norm(x_perm, (x_perm.shape[-1],),
                                  weight=self._aud_sub_w0_norm.float(), bias=None,
                                  eps=self._audio_cfg["rms_norm_eps"])
            x = x_perm.permute(0, 3, 1, 2).contiguous().to(torch.bfloat16)
            x = F.relu(x)
            if mask is not None:
                mask = mask[:, ::2]
            # ---- layer 1 ----
            if mask is not None:
                x = x * mask[:, None, :, None].to(x.dtype)
            x = F.conv2d(x, self._aud_sub_w1_conv, bias=None, stride=2, padding=1)
            x_perm = x.permute(0, 2, 3, 1).float()
            x_perm = F.layer_norm(x_perm, (x_perm.shape[-1],),
                                  weight=self._aud_sub_w1_norm.float(), bias=None,
                                  eps=self._audio_cfg["rms_norm_eps"])
            x = x_perm.permute(0, 3, 1, 2).contiguous().to(torch.bfloat16)
            x = F.relu(x)
            if mask is not None:
                mask = mask[:, ::2]
            # ---- final input_proj_linear ----
            B, _, T_sub, _ = x.shape
            x = x.permute(0, 2, 3, 1).contiguous().reshape(B, T_sub, -1)
            x = F.linear(x.float(), self._aud_sub_proj_w.float()).to(torch.bfloat16)
        return x, mask

    def audio_rel_pos_host(self) -> torch.Tensor:
        """Compute the Gemma4 audio relative-position encoding on host.
        Returns [context_size, hidden_size] BF16.

        Mirrors Gemma4AudioRelPositionalEncoding.forward but does not need a
        hidden_states tensor — the table only depends on context_size and
        hidden_size, both of which come from config.
        """
        self.audio_config_init()
        H = self.AUD_H
        # NOTE: Gemma4 hardcodes ``position_ids = arange(12, -1, -1)`` (13 positions)
        # in HF Gemma4AudioRelPositionalEncoding.forward; this matches the
        # default chunk=12, ctx_left=13, ctx_right=0 case where context_size = 24.
        # The "13" comes from chunk_size+1 (or context_left). For other configs
        # we follow the same pattern: arange(context_left, -1, -1).
        num_pos = self.AUD_CTX_LEFT
        with torch.no_grad():
            num_timescales = H // 2
            log_inc = math.log(10000.0 / 1.0) / max(num_timescales - 1, 1)
            inv_ts = torch.exp(torch.arange(num_timescales, dtype=torch.float32) * -log_inc)
            pos = torch.arange(num_pos - 1, -1, -1, dtype=torch.float32).unsqueeze(-1)  # [num_pos, 1]
            scaled = pos * inv_ts.unsqueeze(0)  # [num_pos, num_timescales]
            pe = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return pe.to(torch.bfloat16)

    def audio_embed_project_host(self, encoder_out: torch.Tensor) -> torch.Tensor:
        """Run output_proj (1024→1536, with bias) and the multimodal embedder
        (RMSNorm with_scale=False + Linear 1536→1536 bias=False) on host.

        Mirrors Gemma4AudioModel.output_proj followed by Gemma4MultimodalEmbedder.

        encoder_out: [N, 1024] from the encoder cumulative pass.
        Returns audio_features: [N, 1536] BF16, ready to merge into the LM
            embedding stream like image features.
        """
        if not hasattr(self, "_aud_output_proj_w"):
            raise RuntimeError("audio_weight_init must be called before audio_embed_project_host")
        with torch.no_grad():
            x = encoder_out.detach().cpu()
            if x.dim() == 3:
                x = x.squeeze(0)
            x = F.linear(x.float(), self._aud_output_proj_w.float(),
                         bias=self._aud_output_proj_b.float())  # [N, 1536]
            # Multimodal embedder: RMSNorm (with_scale=False) + Linear (no bias)
            ms = x.pow(2).mean(-1, keepdim=True) + self._audio_cfg["rms_norm_eps"]
            x = x * torch.pow(ms, -0.5)
            x = F.linear(x, self._aud_embedder_proj_w.float())
        return x.to(torch.bfloat16)

    def compile_audio_layer_ffn1(self, layer_idx: int) -> None:
        """Compile + run FFN1 macaron half for one Conformer layer.

        Reads from AUD_IO_A (or B depending on layer parity), writes back to
        the same buffer. The half-step residual is applied in place.

        Sequence (mirrors Gemma4AudioFeedForward.forward):
            residual = x
            x = clamp(x)                                  ← skipped, BF16 won't overflow
            x = pre_layer_norm(x)                         ← RMSNorm
            x = ffw_layer_1(x)                            ← Linear 1024→4096 (Clippable)
            x = SiLU(x)                                   ← x * sigmoid(x)
            x = ffw_layer_2(x)                            ← Linear 4096→1024 (Clippable)
            x = clamp(x)                                  ← skipped
            x = post_layer_norm(x)                        ← RMSNorm
            x = x * residual_weight  +  residual          ← *0.5 + residual
        """
        self.audio_config_init()
        H = self.AUD_H
        FF = self.AUD_FFN
        L_pad = self._aud_L_pad
        S = self._aud_num_frames  # only the first S rows are valid; rest are padding
        w = self._aud_weight_addrs[layer_idx]
        cr = self._aud_clip_ranges[layer_idx]
        cache = getattr(self, "_aud_program_cache", None)

        def _ck(label: str) -> str:
            return f"aud_L{layer_idx}_{label}"

        IN_BUF = self.AUD_IO_A  # all audio layers operate in place on IO_A

        # Save residual (for half-step add at the end)
        self._aud_copy_buf("ffn1_save_residual", IN_BUF, self.AUD_RESIDUAL, L_pad * H, row_n=H)

        # 1. pre_layer_norm
        self._compile_and_run_single("aud_ff1_pre_norm", lambda: self.rms_norm_core_dram(
            M=L_pad, N=H,
            A_DRAM_ADDR=IN_BUF,
            OUTPUT_DRAM_ADDR=self.AUD_NORM_OUT,
            GAMMA_DRAM_ADDR=w["FF1_PRE_NORM"],
            gpr_M_reg=self._aud_mm_gpr(M=L_pad)),
            cache=cache, cache_key=_ck("ff1_pre_norm"))

        # 2. ffw_layer_1 (clippable, IF4 (block=64)). Apply input clip on host,
        #    run matmul, apply output clip on host.
        self._aud_clip_dram(self.AUD_NORM_OUT, (L_pad, H),
                              cr["FF1_W1"]["in_min"], cr["FF1_W1"]["in_max"])
        ff1w1 = w["FF1_W1"]
        self._compile_and_run_single("aud_ff1_w1", lambda: self.matmat_mul_core(
            M=L_pad, K=H, N=FF,
            A_DRAM_ADDR=self.AUD_NORM_OUT,
            B_DRAM_ADDR=ff1w1["data"],
            OUTPUT_DRAM_ADDR=self.AUD_FFN_MID,
            is_B_quantized=True,
            data_type=TYPE.IF4,
            SCALE_DRAM_ADDR=ff1w1["scale"],
            gpr_M_reg=self._aud_mm_gpr(M=L_pad)),
            cache=cache, cache_key=_ck("ff1_w1"))
        self._aud_clip_dram(self.AUD_FFN_MID, (L_pad, FF),
                              cr["FF1_W1"]["out_min"], cr["FF1_W1"]["out_max"])

        # 3. SiLU on (L_pad, FF) — out goes to a SEPARATE buffer because the
        # silu helper does an in-place sigmoid on its OUTPUT and would
        # destroy x if A_DRAM == OUTPUT_DRAM.
        self._compile_and_run_single("aud_ff1_silu", lambda: _aud_silu(
            self, L_pad, FF, self.AUD_FFN_MID, self.AUD_SILU_OUT, self.AUD_IDENTITY_FF),
            cache=cache, cache_key=_ck("ff1_silu"))

        # 4. ffw_layer_2 (clippable, IF4 (block=64)). Reads from SiLU output.
        self._aud_clip_dram(self.AUD_SILU_OUT, (L_pad, FF),
                              cr["FF1_W2"]["in_min"], cr["FF1_W2"]["in_max"])
        ff1w2 = w["FF1_W2"]
        self._compile_and_run_single("aud_ff1_w2", lambda: self.matmat_mul_core(
            M=L_pad, K=FF, N=H,
            A_DRAM_ADDR=self.AUD_SILU_OUT,
            B_DRAM_ADDR=ff1w2["data"],
            OUTPUT_DRAM_ADDR=self.AUD_FFN_OUT,
            is_B_quantized=True,
            data_type=TYPE.IF4,
            SCALE_DRAM_ADDR=ff1w2["scale"],
            gpr_M_reg=self._aud_mm_gpr(M=L_pad)),
            cache=cache, cache_key=_ck("ff1_w2"))
        self._aud_clip_dram(self.AUD_FFN_OUT, (L_pad, H),
                              cr["FF1_W2"]["out_min"], cr["FF1_W2"]["out_max"])

        # 5. post_layer_norm
        self._compile_and_run_single("aud_ff1_post_norm", lambda: self.rms_norm_core_dram(
            M=L_pad, N=H,
            A_DRAM_ADDR=self.AUD_FFN_OUT,
            OUTPUT_DRAM_ADDR=self.AUD_FFN_OUT,
            GAMMA_DRAM_ADDR=w["FF1_POST_NORM"],
            gpr_M_reg=self._aud_mm_gpr(M=L_pad)),
            cache=cache, cache_key=_ck("ff1_post_norm"))

        # 6. Half-step residual: out = residual + 0.5 * ffn_out → IN_BUF
        self._compile_and_run_single("aud_ff1_half_residual", lambda: _aud_half_step(
            self, L_pad, H,
            self.AUD_RESIDUAL, self.AUD_FFN_OUT, IN_BUF),
            cache=cache, cache_key=_ck("ff1_half_residual"))

    def compile_audio_layer_attn(self, layer_idx: int) -> None:
        """Run the self-attention block of one Conformer layer.

        Sequence (mirrors Gemma4AudioLayer.forward + Gemma4AudioAttention.forward):

            residual = x          # (the layer's running hidden, IN_BUF)
            x = norm_pre_attn(x)  # RMSNorm  ← FPGA
            q = q_proj(x); k = k_proj(x); v = v_proj(x)  ← FPGA (FP4)
            q = q * q_scale * softplus(per_dim_scale)
            k = k * k_scale
            ▼ chunked local attention (rel-pos, soft-cap, mask, softmax) ← HOST for now
            attn_out = post(attn_out)   ← FPGA (FP4)
            x = norm_post_attn(attn_out)  ← FPGA
            x = x + residual               ← FPGA eltwise

        IN_BUF parity: layer 0 reads from AUD_IO_A, layer 1 from AUD_IO_B,
        etc. We write the post-attn-residual result back to the SAME buffer
        the layer started in (so the conv module reads from there).
        """
        self.audio_config_init()
        H = self.AUD_H
        HD = self.AUD_HEAD_DIM
        NH = self.AUD_HEADS
        L_pad = self._aud_L_pad
        T = self._aud_num_frames
        w = self._aud_weight_addrs[layer_idx]
        cr = self._aud_clip_ranges[layer_idx]
        cache = getattr(self, "_aud_program_cache", None)

        def _ck(label: str) -> str:
            return f"aud_L{layer_idx}_{label}"

        IN_BUF = self.AUD_IO_A  # all audio layers operate in place on IO_A

        # Save residual: copy current IN_BUF state into AUD_RESIDUAL.
        # Note: AUD_RESIDUAL is reused by every sub-block (FFN1, attn, conv,
        # FFN2) within this layer. Each sub-block must save its own residual
        # before any FPGA writes happen.
        self._aud_copy_buf("attn_save_residual", IN_BUF, self.AUD_RESIDUAL,
                            L_pad * H, row_n=H)

        # 1. norm_pre_attn (RMSNorm with learned scale)
        self._compile_and_run_single("aud_attn_pre_norm", lambda: self.rms_norm_core_dram(
            M=L_pad, N=H,
            A_DRAM_ADDR=IN_BUF,
            OUTPUT_DRAM_ADDR=self.AUD_NORM_OUT,
            GAMMA_DRAM_ADDR=w["ATTN_PRE_NORM"],
            gpr_M_reg=self._aud_mm_gpr(M=L_pad)),
            cache=cache, cache_key=_ck("attn_pre_norm"))

        # 2. Q / K / V projections (IF4 (block=64), ClippableLinear).
        # Writes go directly into AUD_Q / AUD_K / AUD_V which are sized
        # (L_pad, NH*HD) = (L_pad, H) bf16.
        for proj_name, addr_key, dst in [
            ("Q_PROJ", "Q_PROJ", self.AUD_Q),
            ("K_PROJ", "K_PROJ", self.AUD_K),
            ("V_PROJ", "V_PROJ", self.AUD_V),
        ]:
            self._aud_clip_dram(self.AUD_NORM_OUT, (L_pad, H),
                                  cr[addr_key]["in_min"], cr[addr_key]["in_max"])
            wq = w[addr_key]
            label = f"aud_attn_{proj_name.lower()}"
            self._compile_and_run_single(label, lambda d=dst, ww=wq: self.matmat_mul_core(
                M=L_pad, K=H, N=H,
                A_DRAM_ADDR=self.AUD_NORM_OUT,
                B_DRAM_ADDR=ww["data"],
                OUTPUT_DRAM_ADDR=d,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=ww["scale"],
                gpr_M_reg=self._aud_mm_gpr(M=L_pad)),
                cache=cache, cache_key=_ck(f"attn_{proj_name.lower()}"))
            self._aud_clip_dram(dst, (L_pad, H),
                                  cr[addr_key]["out_min"], cr[addr_key]["out_max"])

        # Phase 2B.b: FPGA Q / K scaling.
        # Q *= q_scale * softplus(per_dim_scale[d])  — eltwise_mul by Q_SCALE_TILE
        # K *= k_scale                                — broadcast_mul by AUD_K_SCALE
        fpga_qk_scale = os.environ.get("GEMMA4_AUDIO_FPGA_QK_SCALE", "1") == "1"
        if fpga_qk_scale:
            self._compile_and_run_single("aud_attn_q_scale",
                lambda: self._emit_aud_q_scale_fpga(layer_idx),
                cache=cache, cache_key=_ck("attn_q_scale"))
            self._compile_and_run_single("aud_attn_k_scale",
                lambda: self._emit_aud_k_scale_fpga(),
                cache=cache, cache_key=_ck("attn_k_scale"))

        # Phase 2B.c debug: compute matrix_ac on FPGA and compare against host.
        # No effect on attn output (host still runs the full math); only prints
        # cosine similarity per layer so we can validate the FPGA matmul + tile
        # construction in isolation before wiring it into the main data flow.
        if os.environ.get("GEMMA4_AUDIO_FPGA_MAC_DEBUG", "0") == "1":
            self._compile_and_run_single("aud_attn_build_kctx_t",
                lambda: self._emit_aud_attn_build_kctx_t(layer_idx),
                cache=cache, cache_key=_ck("attn_build_kctx_t"))
            if os.environ.get("GEMMA4_AUDIO_FPGA_MAC_ONLY_FIRST", "0") == "1":
                self._compile_and_run_single("aud_attn_matrix_ac_first",
                    lambda: self._emit_aud_attn_matrix_ac_probe_first(layer_idx),
                    cache=None, cache_key=None)
            else:
                self._compile_and_run_single("aud_attn_matrix_ac",
                    lambda: self._emit_aud_attn_matrix_ac(layer_idx),
                    cache=cache, cache_key=_ck("attn_matrix_ac"))
            self._aud_validate_matrix_ac(layer_idx, T)
            self._compile_and_run_single("aud_attn_matrix_bd_unshifted",
                lambda: self._emit_aud_attn_matrix_bd_unshifted(layer_idx),
                cache=cache, cache_key=_ck("attn_matrix_bd_unshifted"))
            self._aud_validate_matrix_bd_unshifted(layer_idx, T)
            self._compile_and_run_single("aud_attn_rel_shift",
                lambda: self._emit_aud_attn_rel_shift(layer_idx),
                cache=cache, cache_key=_ck("attn_rel_shift"))
            self._aud_validate_matrix_bd_shifted(layer_idx, T)

        # 3..N: chunked local attention. Default GEMMA4_AUDIO_FPGA_OPROJ=1
        # runs the attention math (Q@K^T → rel-shift → softcap → softmax → @V)
        # on host but performs the output projection (clamp → matmul → clamp)
        # on FPGA. GEMMA4_AUDIO_FPGA_OPROJ=0 falls back to fully host attention
        # + o_proj (legacy path).
        if os.environ.get("GEMMA4_AUDIO_FPGA_OPROJ", "1") == "1":
            preproj = self._aud_chunked_attn_host_preproj(
                layer_idx, T, qk_prescaled=fpga_qk_scale)  # [T, H]
            staged = torch.zeros(L_pad, H, dtype=torch.bfloat16)
            staged[:T] = preproj
            self.dma_to_accelerator_memory(self.AUD_ATTN_OUT, staged.contiguous())
            # FPGA o_proj: clamp(in) → matmul → clamp(out), all in place on
            # AUD_ATTN_OUT.
            self._aud_clip_dram(self.AUD_ATTN_OUT, (L_pad, H),
                                  cr["O_PROJ"]["in_min"], cr["O_PROJ"]["in_max"])
            op = w["O_PROJ"]
            self._compile_and_run_single("aud_attn_o_proj",
                lambda: self.matmat_mul_core(
                    M=L_pad, K=H, N=H,
                    A_DRAM_ADDR=self.AUD_ATTN_OUT,
                    B_DRAM_ADDR=op["data"],
                    OUTPUT_DRAM_ADDR=self.AUD_ATTN_OUT,
                    is_B_quantized=True, data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=op["scale"],
                gpr_M_reg=self._aud_mm_gpr(M=L_pad)),
                cache=cache, cache_key=_ck("attn_o_proj"))
            self._aud_clip_dram(self.AUD_ATTN_OUT, (L_pad, H),
                                  cr["O_PROJ"]["out_min"], cr["O_PROJ"]["out_max"])
        else:
            attn_out_host = self._aud_chunked_attn_host(layer_idx, T)
            staged = torch.zeros(L_pad, H, dtype=torch.bfloat16)
            staged[:T] = attn_out_host
            self.dma_to_accelerator_memory(self.AUD_ATTN_OUT, staged.contiguous())

        # 4. norm_post_attn (RMSNorm)
        self._compile_and_run_single("aud_attn_post_norm", lambda: self.rms_norm_core_dram(
            M=L_pad, N=H,
            A_DRAM_ADDR=self.AUD_ATTN_OUT,
            OUTPUT_DRAM_ADDR=self.AUD_ATTN_OUT,
            GAMMA_DRAM_ADDR=w["ATTN_POST_NORM"],
            gpr_M_reg=self._aud_mm_gpr(M=L_pad)),
            cache=cache, cache_key=_ck("attn_post_norm"))

        # 5. Residual add: out = norm_post_attn(attn) + saved residual → IN_BUF
        self._compile_and_run_single("aud_attn_residual", lambda: _aud_eltwise_add(
            self, L_pad, H, self.AUD_RESIDUAL, self.AUD_ATTN_OUT, IN_BUF),
            cache=cache, cache_key=_ck("attn_residual"))

    def _aud_validate_matrix_ac(self, layer_idx: int, T: int) -> None:
        """Compare FPGA-computed matrix_ac (in AUD_MATRIX_AC) against host math.
        Prints cosine similarity, mean|d|, max|d| per layer. No side effects."""
        H = self.AUD_H
        HD = self.AUD_HEAD_DIM
        NH = self.AUD_HEADS
        chunk = self.AUD_CHUNK
        ctx_size = self.AUD_CTX
        ctx_pad = self._aud_ctx_pad
        chunk_pad = self._aud_chunk_pad
        num_blocks = self._aud_num_blocks
        max_past = self.AUD_CTX_LEFT - 1
        max_future = self.AUD_CTX_RIGHT
        L_pad = self._aud_L_pad

        # ---- Intermediate sanity probes ----
        # 1. K_PADDED: expected = [max_past zeros, K[0:L_pad], trailing zeros]
        K_full = self.dma_from_accelerator_memory(self.AUD_K, (L_pad, H)).cpu()
        L_pf = self._aud_L_padded_full
        K_pad_fpga = self.dma_from_accelerator_memory(self.AUD_K_PADDED, (L_pf, H)).cpu()
        K_pad_expect = torch.zeros(L_pf, H, dtype=torch.bfloat16)
        K_pad_expect[max_past:max_past + L_pad] = K_full
        cos_kpad = torch.nn.functional.cosine_similarity(
            K_pad_fpga.float().flatten().unsqueeze(0),
            K_pad_expect.float().flatten().unsqueeze(0)).item()
        kpad_d = (K_pad_fpga.float() - K_pad_expect.float()).abs()
        print(f"  [probe K_PADDED] L{layer_idx+1} cos={cos_kpad:.6f}  max|d|={kpad_d.max():.4f}")

        # 2. K_CTX_BLOCKS: per-block windowed view of K_PADDED
        K_ctx_fpga = self.dma_from_accelerator_memory(
            self.AUD_K_CTX_BLOCKS, (num_blocks, ctx_pad, H)).cpu().float()
        K_pad_e = K_pad_expect.float()
        K_ctx_exp = torch.zeros(num_blocks, ctx_pad, H)
        for b in range(num_blocks):
            K_ctx_exp[b, :ctx_size] = K_pad_e[b * chunk:b * chunk + ctx_size]
        cos_kctx = torch.nn.functional.cosine_similarity(
            K_ctx_fpga.flatten().unsqueeze(0),
            K_ctx_exp.flatten().unsqueeze(0)).item()
        print(f"  [probe K_CTX  ] L{layer_idx+1} cos={cos_kctx:.6f}  "
              f"max|d|={(K_ctx_fpga - K_ctx_exp).abs().max():.4f}")

        # 3. K_CTX_T_BLOCKS (block 0, head 0): now in (ctx_pad, HD) layout,
        # directly the (N=ctx_pad, K=HD) B-matrix the matmul expects.
        K_T_fpga = self.dma_from_accelerator_memory(
            self.AUD_K_CTX_T_BLOCKS, (num_blocks, NH, ctx_pad, HD)).cpu().float()
        K_T_exp_b0h0 = torch.zeros(ctx_pad, HD)
        K_T_exp_b0h0[:ctx_size] = K_ctx_exp[0, :ctx_size, 0:HD]
        cos_kt = torch.nn.functional.cosine_similarity(
            K_T_fpga[0, 0].flatten().unsqueeze(0),
            K_T_exp_b0h0.flatten().unsqueeze(0)).item()
        print(f"  [probe K_T b0h0] L{layer_idx+1} cos={cos_kt:.6f}  "
              f"max|d|={(K_T_fpga[0, 0] - K_T_exp_b0h0).abs().max():.4f}")

        # 4. Q_HEAD_BLOCK after the last iteration. If MAC_ONLY_FIRST=1, only
        # (0, 0) ran, so the buffer holds that. Otherwise it holds (num_blocks-1, NH-1).
        Q_full = self.dma_from_accelerator_memory(self.AUD_Q, (L_pad, H)).cpu().float()
        if os.environ.get("GEMMA4_AUDIO_FPGA_MAC_ONLY_FIRST", "0") == "1":
            b_last, h_last = 0, 0
        else:
            b_last = num_blocks - 1; h_last = NH - 1
        Q_block_fpga = self.dma_from_accelerator_memory(
            self.AUD_Q_HEAD_BLOCK, (chunk_pad, HD)).cpu().float()
        Q_exp = torch.zeros(chunk_pad, HD)
        Q_3d = Q_full.view(L_pad, NH, HD)
        rows_to_take = min(chunk, L_pad - b_last * chunk)
        if rows_to_take > 0:
            Q_exp[:rows_to_take] = Q_3d[b_last * chunk:b_last * chunk + rows_to_take, h_last]
        cos_q = torch.nn.functional.cosine_similarity(
            Q_block_fpga.flatten().unsqueeze(0),
            Q_exp.flatten().unsqueeze(0)).item()
        print(f"  [probe Q_blk b{b_last}h{h_last}] cos={cos_q:.6f}  "
              f"max|d|={(Q_block_fpga - Q_exp).abs().max():.4f}  "
              f"rows_valid={rows_to_take}")

        # 5. Direct matmul check (block 0, head 0): manually compute
        #    Q_HEAD_BLOCK @ K_CTX_T_BLOCKS[0, 0]^T on HOST and compare to FPGA matrix_ac[0, 0].
        Q_blk = self.dma_from_accelerator_memory(
            self.AUD_Q_HEAD_BLOCK, (chunk_pad, HD)).cpu().float()
        K_B_b0h0 = self.dma_from_accelerator_memory(
            self.AUD_K_CTX_T_BLOCKS, (ctx_pad, HD)).cpu().float()
        host_matmul = Q_blk @ K_B_b0h0.T  # (chunk_pad, ctx_pad)
        fpga_matmul = self.dma_from_accelerator_memory(
            self.AUD_MATRIX_AC, (chunk_pad, ctx_pad)).cpu().float()
        cos_mm = torch.nn.functional.cosine_similarity(
            host_matmul.flatten().unsqueeze(0),
            fpga_matmul.flatten().unsqueeze(0)).item()
        dmm = (host_matmul - fpga_matmul).abs()
        print(f"  [probe matmul b0h0] cos={cos_mm:.6f}  "
              f"max|d|={dmm.max():.4f}  host_amean={host_matmul.abs().mean():.4f}  "
              f"fpga_amean={fpga_matmul.abs().mean():.4f}")
        print(f"     host row 0 first 8: {host_matmul[0, :8].tolist()}")
        print(f"     fpga row 0 first 8: {fpga_matmul[0, :8].tolist()}")

        # Host reference: read AUD_Q, AUD_K (already scaled), run HF block math.
        Q_full = self.dma_from_accelerator_memory(self.AUD_Q, (L_pad, H)).cpu()[:T].float()
        K_full = self.dma_from_accelerator_memory(self.AUD_K, (L_pad, H)).cpu()[:T].float()
        Q_full = Q_full.view(T, NH, HD)
        K_full = K_full.view(T, NH, HD)
        # Mirror HF _convert_to_block + _extract_block_context
        pad = num_blocks * chunk - T
        Q_pad = F.pad(Q_full.unsqueeze(0), (0, 0, 0, 0, 0, pad)).view(1, num_blocks, chunk, NH, HD)
        K_pad = F.pad(K_full.unsqueeze(0), (0, 0, 0, 0, max_past, max_future + chunk - 1))
        K_ctx = K_pad.unfold(1, ctx_size, chunk)  # (1, num_blocks, NH, HD, ctx_size)
        K_ctx = torch.movedim(K_ctx, -1, 2).contiguous()  # (1, num_blocks, ctx_size, NH, HD)
        queries = Q_pad.permute(0, 3, 1, 2, 4)            # (1, NH, num_blocks, chunk, HD)
        K_ctx_perm = K_ctx.permute(0, 3, 1, 4, 2)         # (1, NH, num_blocks, HD, ctx_size)
        host_ac = (queries @ K_ctx_perm).squeeze(0)       # (NH, num_blocks, chunk, ctx_size)
        host_ac = host_ac.permute(1, 0, 2, 3).contiguous()  # (num_blocks, NH, chunk, ctx_size)

        # FPGA result: (num_blocks, NH, chunk_pad, ctx_pad) → slice to (num_blocks, NH, chunk, ctx_size)
        fpga_full = self.dma_from_accelerator_memory(self.AUD_MATRIX_AC,
            (num_blocks, NH, chunk_pad, ctx_pad)).cpu().float()
        fpga_ac = fpga_full[:, :, :chunk, :ctx_size]

        a = host_ac.flatten(); b = fpga_ac.flatten()
        cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        d = (host_ac - fpga_ac).abs()
        print(f"  [FPGA matrix_ac] L{layer_idx+1} cos={cos:.6f}  mean|d|={d.mean():.4f}  "
              f"max|d|={d.max():.4f}  host_amean={host_ac.abs().mean():.4f}  "
              f"fpga_amean={fpga_ac.abs().mean():.4f}")
        # Per-(block, head) breakdown
        for bb in range(num_blocks):
            for hh in [0, NH - 1]:
                hv = host_ac[bb, hh].flatten(); fv = fpga_ac[bb, hh].flatten()
                cv = torch.nn.functional.cosine_similarity(
                    hv.unsqueeze(0), fv.unsqueeze(0)).item()
                if cv < 0.99:
                    print(f"     b={bb} h={hh} cos={cv:.4f} "
                          f"host_amean={hv.abs().mean():.3f} "
                          f"fpga_amean={fv.abs().mean():.3f}")

    def _aud_validate_matrix_bd_unshifted(self, layer_idx: int, T: int) -> None:
        """Compare FPGA-computed matrix_bd_unshifted against host. FPGA layout
        is (NH, T_pad_padded, bd_N) with cols [chunk_pad, chunk_pad+num_pos)
        carrying real Q@rel_k values; other cols are zero pad."""
        H = self.AUD_H
        HD = self.AUD_HEAD_DIM
        NH = self.AUD_HEADS
        L_pad = self._aud_L_pad
        T_pad_padded = self._aud_T_pad_padded
        num_pos = self.AUD_CTX_LEFT
        num_pos_pad = self._aud_num_pos_pad
        chunk_pad = self._aud_chunk_pad
        bd_N = self._aud_bd_unshifted_N

        Q_full = self.dma_from_accelerator_memory(
            self.AUD_Q, (L_pad, H)).cpu().float().view(L_pad, NH, HD)
        pos_emb = self.audio_rel_pos_host().float()
        rel_k_w = self._get_audio_rel_k_proj_weight(layer_idx).float()
        rel_k = (pos_emb @ rel_k_w.T).view(-1, NH, HD)
        # Build host expected with the padded layout.
        host_bd = torch.zeros(NH, T_pad_padded, bd_N)
        for h in range(NH):
            host_bd[h, :L_pad, chunk_pad:chunk_pad + num_pos] = \
                Q_full[:, h, :] @ rel_k[:, h, :].T

        fpga_bd = self.dma_from_accelerator_memory(
            self.AUD_MATRIX_BD_UNSHIFTED,
            (NH, T_pad_padded, bd_N)).cpu().float()
        cos = torch.nn.functional.cosine_similarity(
            host_bd.flatten().unsqueeze(0),
            fpga_bd.flatten().unsqueeze(0)).item()
        d = (host_bd - fpga_bd).abs()
        print(f"  [FPGA matrix_bd_unshifted] L{layer_idx+1} cos={cos:.6f}  "
              f"mean|d|={d.mean():.4f}  max|d|={d.max():.4f}  "
              f"host_amean={host_bd.abs().mean():.4f}  fpga_amean={fpga_bd.abs().mean():.4f}")
        # Spot-check row 1 cols around the real-data band [chunk_pad, chunk_pad+num_pos).
        print(f"    FPGA bd_unsh[h=0, t=1, 60:80] = "
              f"{[f'{v:.3f}' for v in fpga_bd[0, 1, 60:80].tolist()]}")
        print(f"    HOST bd_unsh[h=0, t=1, 60:80] = "
              f"{[f'{v:.3f}' for v in host_bd[0, 1, 60:80].tolist()]}")

    def _aud_validate_matrix_bd_shifted(self, layer_idx: int, T: int) -> None:
        """Compare FPGA-computed matrix_bd_shifted against host rel-shift of HF math."""
        H = self.AUD_H
        HD = self.AUD_HEAD_DIM
        NH = self.AUD_HEADS
        L_pad = self._aud_L_pad
        chunk = self.AUD_CHUNK
        ctx_size = self.AUD_CTX
        ctx_pad = self._aud_ctx_pad
        chunk_pad = self._aud_chunk_pad
        num_blocks = self._aud_num_blocks
        num_pos = self.AUD_CTX_LEFT

        # Host: same path as HF.
        Q_full = self.dma_from_accelerator_memory(
            self.AUD_Q, (L_pad, H)).cpu().float().view(1, L_pad, NH, HD)
        pad = num_blocks * chunk - T
        Q_pad = F.pad(Q_full, (0, 0, 0, 0, 0, pad))  # (1, num_blocks*chunk, NH, HD)
        queries = Q_pad.view(1, num_blocks, chunk, NH, HD).permute(0, 3, 1, 2, 4)
        queries_flat = queries.reshape(1, NH, -1, HD)
        pos_emb = self.audio_rel_pos_host().float()
        rel_k_w = self._get_audio_rel_k_proj_weight(layer_idx).float()
        rel_k = (pos_emb @ rel_k_w.T).view(-1, NH, HD)
        rel_k_perm = rel_k.permute(1, 2, 0)
        bd_unsh = queries_flat @ rel_k_perm  # (1, NH, num_blocks*chunk, num_pos)
        bd_unsh = bd_unsh.reshape(1, NH, num_blocks, chunk, num_pos)
        bd_sh_hf = self._aud_rel_shift_host(bd_unsh, ctx_size)  # (1, NH, num_blocks, chunk, ctx_size)
        # Match FPGA layout (num_blocks, NH, chunk_pad, ctx_pad)
        host_bd_sh = torch.zeros(num_blocks, NH, chunk_pad, ctx_pad)
        host_bd_sh[:, :, :chunk, :ctx_size] = bd_sh_hf.squeeze(0).permute(1, 0, 2, 3)

        fpga_bd_sh = self.dma_from_accelerator_memory(
            self.AUD_MATRIX_BD_SHIFTED,
            (num_blocks, NH, chunk_pad, ctx_pad)).cpu().float()
        # FPGA fills ctx columns [r:r+num_pos_pad] but only [r:r+num_pos) are
        # semantically valid; positions [r+num_pos:r+num_pos_pad) are 0 from the
        # source's zero-padded tail. We compare the valid range.
        host_v = host_bd_sh[:, :, :chunk, :ctx_size]
        fpga_v = fpga_bd_sh[:, :, :chunk, :ctx_size]
        cos = torch.nn.functional.cosine_similarity(
            host_v.flatten().unsqueeze(0),
            fpga_v.flatten().unsqueeze(0)).item()
        d = (host_v - fpga_v).abs()
        print(f"  [FPGA matrix_bd_shifted] L{layer_idx+1} cos={cos:.6f}  "
              f"mean|d|={d.mean():.4f}  max|d|={d.max():.4f}  "
              f"host_amean={host_v.abs().mean():.4f}  fpga_amean={fpga_v.abs().mean():.4f}")
        # Spot check row 0 of (b=0, h=0)
        print(f"     host bd_sh[0,0,0,:14] = {host_v[0, 0, 0, :14].tolist()}")
        print(f"     fpga bd_sh[0,0,0,:14] = {fpga_v[0, 0, 0, :14].tolist()}")
        print(f"     host bd_sh[0,0,1,:14] = {host_v[0, 0, 1, :14].tolist()}")
        print(f"     fpga bd_sh[0,0,1,:14] = {fpga_v[0, 0, 1, :14].tolist()}")
        # Inspect full row 1 of FPGA to find where the data actually went.
        fpga_full = self.dma_from_accelerator_memory(
            self.AUD_MATRIX_BD_SHIFTED,
            (num_blocks, NH, chunk_pad, ctx_pad)).cpu().float()
        row1 = fpga_full[0, 0, 1]
        max_val = row1.abs().max().item()
        nz_idxs = (row1.abs() > 1.0).nonzero().squeeze(-1).tolist()
        print(f"     fpga row 1 max={max_val:.3f}  nonzero idxs={nz_idxs[:20]}")
        # Dump the actual nonzero values
        if nz_idxs:
            print(f"     fpga row 1 vals at nz idxs = "
                  f"{[f'{row1[i].item():.2f}' for i in nz_idxs[:14]]}")
        # And row 0 nonzero indices to confirm r=0 case
        row0 = fpga_full[0, 0, 0]
        nz0 = (row0.abs() > 1.0).nonzero().squeeze(-1).tolist()
        print(f"     fpga row 0 nonzero idxs={nz0[:14]}  vals={[f'{row0[i].item():.2f}' for i in nz0[:8]]}")
        # And rel_shift src/dst for (b=0, h=0, r=1) explicit
        h_, b_, r_ = 0, 0, 1
        bd_N_v = self._aud_bd_unshifted_N
        T_pp = self._aud_T_pad_padded
        src_offset_bytes = ((h_ * T_pp + b_ * chunk + r_) * bd_N_v + (chunk_pad - r_)) * 2
        dst_offset_bytes = ((b_ * NH + h_) * chunk_pad * ctx_pad + r_ * ctx_pad) * 2
        print(f"     r=1 src_offset={src_offset_bytes}b  dst_offset={dst_offset_bytes}b  "
              f"src_aligned={src_offset_bytes % 128==0}  dst_aligned={dst_offset_bytes % 128==0}")

    def _aud_chunked_attn_host_preproj(self, layer_idx: int, T: int,
                                          *, qk_prescaled: bool = False) -> torch.Tensor:
        """Run chunked self-attention math on host, returning the attn@V
        result [T, H] BEFORE the o_proj output projection. The caller is
        expected to run o_proj (clamp → matmul → clamp) on FPGA. Mirrors
        steps 1–9 of ``_aud_chunked_attn_host`` exactly; only step 10 (the
        o_proj) is omitted.

        ``qk_prescaled``: when True, AUD_Q / AUD_K in DRAM already include the
        per-dim Q-scale (q_scale * softplus(per_dim_scale)) and the K-scale
        (k_scale), so the host scaling lines are skipped."""
        H = self.AUD_H
        HD = self.AUD_HEAD_DIM
        NH = self.AUD_HEADS
        L_pad = self._aud_L_pad
        chunk_size = self.AUD_CHUNK
        max_past = self.AUD_CTX_LEFT - 1
        max_future = self.AUD_CTX_RIGHT
        context_size = chunk_size + max_past + max_future
        soft_cap = self.AUD_SOFT_CAP
        invalid_logit = self.AUD_INVALID_LOGIT
        w_layer = self._aud_weight_addrs[layer_idx]

        Q = self.dma_from_accelerator_memory(self.AUD_Q, (L_pad, H)).cpu()[:T].float()
        K = self.dma_from_accelerator_memory(self.AUD_K, (L_pad, H)).cpu()[:T].float()
        V = self.dma_from_accelerator_memory(self.AUD_V, (L_pad, H)).cpu()[:T].float()

        Q = Q.view(1, T, NH, HD); K = K.view(1, T, NH, HD); V = V.view(1, T, NH, HD)

        if not qk_prescaled:
            per_dim_scale = self.dma_from_accelerator_memory(
                w_layer["PER_DIM_SCALE"], (HD,)).cpu().float()
            Q = Q * self.AUD_Q_SCALE * F.softplus(per_dim_scale)
            K = K * self.AUD_K_SCALE

        num_blocks = (T + chunk_size - 1) // chunk_size
        pad = num_blocks * chunk_size - T
        Q_pad = F.pad(Q, (0, 0, 0, 0, 0, pad))
        Q_blocks = Q_pad.view(1, num_blocks, chunk_size, NH, HD).contiguous()

        K_pad = F.pad(K, (0, 0, 0, 0, max_past, max_future + chunk_size - 1))
        V_pad = F.pad(V, (0, 0, 0, 0, max_past, max_future + chunk_size - 1))
        K_ctx = K_pad.unfold(1, context_size, chunk_size)
        V_ctx = V_pad.unfold(1, context_size, chunk_size)
        K_ctx = torch.movedim(K_ctx, -1, 2).contiguous()
        V_ctx = torch.movedim(V_ctx, -1, 2).contiguous()

        pos_emb = self.audio_rel_pos_host().float()
        rel_k_w = self._get_audio_rel_k_proj_weight(layer_idx).float()
        rel_k = (pos_emb @ rel_k_w.T).view(-1, NH, HD)

        queries = Q_blocks.permute(0, 3, 1, 2, 4)
        K_ctx_perm = K_ctx.permute(0, 3, 1, 4, 2)
        matrix_ac = queries @ K_ctx_perm

        queries_flat = queries.reshape(1, NH, -1, HD)
        rel_k_perm = rel_k.permute(1, 2, 0)
        matrix_bd = queries_flat @ rel_k_perm
        matrix_bd = matrix_bd.reshape(1, NH, num_blocks, chunk_size, -1)
        matrix_bd = self._aud_rel_shift_host(matrix_bd, context_size)

        attn_w = matrix_ac + matrix_bd
        attn_w = soft_cap * torch.tanh(attn_w / soft_cap)
        mask_5d = self._aud_make_blocked_mask(T, num_blocks, chunk_size,
                                                 max_past, max_future)
        attn_w = attn_w.masked_fill(~mask_5d, invalid_logit)
        attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(V_ctx.dtype)

        V_ctx_perm = V_ctx.permute(0, 3, 1, 2, 4)
        attn_out = attn_w @ V_ctx_perm
        attn_out = attn_out.permute(0, 2, 3, 1, 4).reshape(1, num_blocks * chunk_size, -1)
        attn_out = attn_out[:, :T].contiguous().squeeze(0)  # [T, H]
        return attn_out.to(torch.bfloat16)

    def _aud_chunked_attn_host(self, layer_idx: int, T: int) -> torch.Tensor:
        """HOST implementation of Gemma4 chunked local self-attention.

        Reads Q, K, V from FPGA DRAM, runs the exact HF attention math
        (rel-pos bias, soft-cap tanh, mask, softmax, attn@V), then runs the
        post (output) projection — also on host since the input length is
        small. Returns [T, H] bf16 (the OUT proj output).
        """
        H = self.AUD_H
        HD = self.AUD_HEAD_DIM
        NH = self.AUD_HEADS
        L_pad = self._aud_L_pad
        chunk_size = self.AUD_CHUNK
        max_past = self.AUD_CTX_LEFT - 1
        max_future = self.AUD_CTX_RIGHT
        context_size = chunk_size + max_past + max_future
        soft_cap = self.AUD_SOFT_CAP
        invalid_logit = self.AUD_INVALID_LOGIT
        w_layer = self._aud_weight_addrs[layer_idx]
        cr = self._aud_clip_ranges[layer_idx]

        # 1. Read Q, K, V from DRAM (only the first T rows are real)
        Q = self.dma_from_accelerator_memory(self.AUD_Q, (L_pad, H)).cpu()[:T].float()
        K = self.dma_from_accelerator_memory(self.AUD_K, (L_pad, H)).cpu()[:T].float()
        V = self.dma_from_accelerator_memory(self.AUD_V, (L_pad, H)).cpu()[:T].float()

        # Reshape to [B=1, T, NH, HD]
        Q = Q.view(1, T, NH, HD)
        K = K.view(1, T, NH, HD)
        V = V.view(1, T, NH, HD)

        # 2. Per-head scaling — match HF lines 220-221, 278-279.
        per_dim_scale = self.dma_from_accelerator_memory(
            w_layer["PER_DIM_SCALE"], (HD,)).cpu().float()
        q_scale = self.AUD_Q_SCALE
        k_scale = self.AUD_K_SCALE
        Q = Q * q_scale * F.softplus(per_dim_scale)
        K = K * k_scale

        # 3. _convert_to_block on Q
        num_blocks = (T + chunk_size - 1) // chunk_size
        pad = num_blocks * chunk_size - T
        Q_pad = F.pad(Q, (0, 0, 0, 0, 0, pad))
        Q_blocks = Q_pad.view(1, num_blocks, chunk_size, NH, HD).contiguous()

        # 4. _extract_block_context on K and V
        K_pad = F.pad(K, (0, 0, 0, 0, max_past, max_future + chunk_size - 1))
        V_pad = F.pad(V, (0, 0, 0, 0, max_past, max_future + chunk_size - 1))
        K_ctx = K_pad.unfold(1, context_size, chunk_size)  # [1, num_blocks, NH, HD, context_size]
        V_ctx = V_pad.unfold(1, context_size, chunk_size)
        K_ctx = torch.movedim(K_ctx, -1, 2).contiguous()   # [1, num_blocks, context_size, NH, HD]
        V_ctx = torch.movedim(V_ctx, -1, 2).contiguous()

        # 5. relative_k_proj on the relative position embedding
        pos_emb = self.audio_rel_pos_host().float()  # [num_pos, H]
        # relative_k_proj weight is IF4-quantized on FPGA; recompute the
        # rel-pos contribution on host using the BF16 reference cached during
        # weight upload (avoids dequantizing the IF4 bytes).
        rel_k_w = self._get_audio_rel_k_proj_weight(layer_idx).float()  # [H, H]
        rel_k = pos_emb @ rel_k_w.T  # [num_pos, H]
        rel_k = rel_k.view(-1, NH, HD)

        # 6. Q @ K^T
        queries = Q_blocks.permute(0, 3, 1, 2, 4)  # [B, NH, num_blocks, chunk, HD]
        K_ctx_perm = K_ctx.permute(0, 3, 1, 4, 2)  # [B, NH, num_blocks, HD, context]
        matrix_ac = queries @ K_ctx_perm  # [B, NH, num_blocks, chunk, context]

        # 7. Q @ rel_k
        queries_flat = queries.reshape(1, NH, -1, HD)  # [B, NH, num_blocks*chunk, HD]
        rel_k_perm = rel_k.permute(1, 2, 0)  # [NH, HD, num_pos]
        matrix_bd = queries_flat @ rel_k_perm  # [B, NH, num_blocks*chunk, num_pos]
        matrix_bd = matrix_bd.reshape(1, NH, num_blocks, chunk_size, -1)
        matrix_bd = self._aud_rel_shift_host(matrix_bd, context_size)

        # 8. add + soft-cap + mask + softmax
        attn_w = matrix_ac + matrix_bd
        attn_w = soft_cap * torch.tanh(attn_w / soft_cap)

        # Mask: build the same blocked 5D mask HF builds via
        # _convert_4d_mask_to_blocked_5d.
        mask_5d = self._aud_make_blocked_mask(T, num_blocks, chunk_size,
                                                max_past, max_future)
        attn_w = attn_w.masked_fill(~mask_5d, invalid_logit)
        attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(V_ctx.dtype)

        # 9. attn @ V
        V_ctx_perm = V_ctx.permute(0, 3, 1, 2, 4)  # [B, NH, num_blocks, context, HD]
        attn_out = attn_w @ V_ctx_perm  # [B, NH, num_blocks, chunk, HD]
        attn_out = attn_out.permute(0, 2, 3, 1, 4).reshape(1, num_blocks * chunk_size, -1)
        attn_out = attn_out[:, :T].contiguous()  # [1, T, H]

        # 10. post (output) projection — apply Clippable input/output clipping
        attn_out_clip = attn_out.squeeze(0).clamp(
            min=cr["O_PROJ"]["in_min"], max=cr["O_PROJ"]["in_max"])
        out_w = self._get_audio_o_proj_weight(layer_idx).float()  # [H, H]
        result = attn_out_clip.float() @ out_w.T
        result = result.clamp(min=cr["O_PROJ"]["out_min"], max=cr["O_PROJ"]["out_max"])
        return result.to(torch.bfloat16)

    def _aud_rel_shift_host(self, x: torch.Tensor, context_size: int) -> torch.Tensor:
        """Host port of Gemma4AudioAttention._rel_shift."""
        batch_size, num_heads, num_blocks, block_size, position_length = x.shape
        x = F.pad(x, (0, context_size + 1 - position_length))
        x = x.view(batch_size, num_heads, num_blocks, block_size * (context_size + 1))
        x = x[..., : block_size * context_size]
        return x.view(batch_size, num_heads, num_blocks, block_size, context_size)

    def _aud_make_blocked_mask(self, T: int, num_blocks: int, chunk_size: int,
                                max_past: int, max_future: int) -> torch.Tensor:
        """Build the 5D blocked attention mask matching HF's
        _convert_4d_mask_to_blocked_5d for a single sequence of length T,
        with no padding (B=1). Result: [1, 1, num_blocks, chunk_size, context_size].
        """
        padded_seq_len = num_blocks * chunk_size
        # 4D causal-by-distance mask: True if (q_idx - kv_idx) within window
        valid = torch.zeros(padded_seq_len, padded_seq_len, dtype=torch.bool)
        valid[:T, :T] = True
        # Sliding-window constraint
        i_idx = torch.arange(padded_seq_len).unsqueeze(1)
        j_idx = torch.arange(padded_seq_len).unsqueeze(0)
        dist = i_idx - j_idx
        within = ((dist >= 0) & (dist < max_past + 1)) | ((dist < 0) & (-dist <= max_future))
        valid = valid & within
        # Pad to [num_blocks * chunk_size + max_past + max_future] on the kv axis
        mask_5d = valid.view(1, 1, num_blocks, chunk_size, padded_seq_len)
        mask_5d = F.pad(mask_5d, (max_past, max_future), value=False)
        block_starts = torch.arange(num_blocks) * chunk_size
        offsets = torch.arange(chunk_size + max_past + max_future)
        kv_indices = block_starts[:, None] + offsets[None, :]
        kv_indices = kv_indices[None, None, :, None, :].expand(1, 1, -1, chunk_size, -1)
        return mask_5d.gather(-1, kv_indices)

    def _get_audio_rel_k_proj_weight(self, layer_idx: int) -> torch.Tensor:
        """Lazily fetch the BF16 relative_k_proj weight from the host-cached
        HF model (we kept references during audio_weight_init via _aud_hf)."""
        return self._aud_hf_layers[layer_idx]["rel_k_w"]

    def _get_audio_o_proj_weight(self, layer_idx: int) -> torch.Tensor:
        return self._aud_hf_layers[layer_idx]["o_w"]

    def compile_audio_layer_conv(self, layer_idx: int) -> None:
        """Run the lconv1d (light Conv1d) module of one Conformer layer.

        Sequence (mirrors Gemma4AudioLightConv1d.forward):
            residual = x
            x = pre_layer_norm(x)             ← FPGA RMSNorm
            x = linear_start(x)               ← FPGA matmul (1024 → 2048)
            x = GLU(x)                        ← FPGA helper (split halves + sigmoid + mul)
            x = depthwise_conv1d(x)           ← HOST (kernel size 5)
            x = conv_norm(x)                  ← FPGA RMSNorm
            x = SiLU(x)                       ← FPGA helper
            x = linear_end(x)                 ← FPGA matmul (1024 → 1024)
            x = x + residual                  ← FPGA eltwise
        """
        self.audio_config_init()
        H = self.AUD_H
        L_pad = self._aud_L_pad
        T = self._aud_num_frames
        w = self._aud_weight_addrs[layer_idx]
        cr = self._aud_clip_ranges[layer_idx]
        cache = getattr(self, "_aud_program_cache", None)

        def _ck(label: str) -> str:
            return f"aud_L{layer_idx}_{label}"

        IN_BUF = self.AUD_IO_A  # all audio layers operate in place on IO_A

        self._aud_copy_buf("conv_save_residual", IN_BUF, self.AUD_RESIDUAL, L_pad * H, row_n=H)

        # 1. pre_layer_norm
        self._compile_and_run_single("aud_conv_pre_norm", lambda: self.rms_norm_core_dram(
            M=L_pad, N=H,
            A_DRAM_ADDR=IN_BUF,
            OUTPUT_DRAM_ADDR=self.AUD_NORM_OUT,
            GAMMA_DRAM_ADDR=w["CONV_PRE_NORM"],
            gpr_M_reg=self._aud_mm_gpr(M=L_pad)),
            cache=cache, cache_key=_ck("conv_pre_norm"))

        # 2. linear_start (1024 → 2048). Output goes to AUD_FFN_MID temporarily
        # (which is L_pad × FF=4096 = enough for L_pad × 2H=2048).
        self._aud_clip_dram(self.AUD_NORM_OUT, (L_pad, H),
                              cr["CONV_LIN_START"]["in_min"], cr["CONV_LIN_START"]["in_max"])
        cls = w["CONV_LIN_START"]
        self._compile_and_run_single("aud_conv_lin_start", lambda: self.matmat_mul_core(
            M=L_pad, K=H, N=2 * H,
            A_DRAM_ADDR=self.AUD_NORM_OUT,
            B_DRAM_ADDR=cls["data"],
            OUTPUT_DRAM_ADDR=self.AUD_FFN_MID,
            is_B_quantized=True, data_type=TYPE.IF4,
            SCALE_DRAM_ADDR=cls["scale"],
            gpr_M_reg=self._aud_mm_gpr(M=L_pad)),
            cache=cache, cache_key=_ck("conv_lin_start"))
        self._aud_clip_dram(self.AUD_FFN_MID, (L_pad, 2 * H),
                              cr["CONV_LIN_START"]["out_min"], cr["CONV_LIN_START"]["out_max"])

        # 3. GLU: split (L_pad, 2H) into gate=(L_pad, H) and value=(L_pad, H),
        # then output = gate * sigmoid(value).
        # The linear output is laid out as [L_pad, 2H] row-major. Row r:
        #   gate[r]  = AUD_FFN_MID[r, 0:H]
        #   value[r] = AUD_FFN_MID[r, H:2H]
        # We can pass these as DRAM bases to glu_core_dram, but the helper
        # assumes both halves are in CONTIGUOUS L_pad×H buffers, not a single
        # interleaved buffer. Easiest fix: copy each half to its own buffer.
        # AUD_CONV_GATE/VALUE were dropped in tensor_init's Phase-3 cleanup;
        # reuse AUD_NORM_OUT (L_pad×H, free at this point) as the gate buffer
        # and AUD_RESIDUAL would clobber the residual we need later — so use
        # AUD_FFN_OUT (L_pad×H) as the value buffer instead.
        # Then the helper writes the GLU output back over AUD_NORM_OUT.
        # row_bytes_2H = 2*H*bpe; gate at offset 0, value at offset H*bpe per row.
        self._aud_split_2h_to_halves("conv_glu_split",
            self.AUD_FFN_MID, self.AUD_NORM_OUT, self.AUD_FFN_OUT, L_pad, H)
        self._compile_and_run_single("aud_conv_glu", lambda: _aud_glu(
            self, L_pad, H,
            self.AUD_NORM_OUT,  # GATE (a)
            self.AUD_FFN_OUT,   # VALUE (b, will be sigmoided in place)
            self.AUD_NORM_OUT,  # OUTPUT
            self.AUD_IDENTITY_H),  # H×H identity for K=N=H matmul
            cache=cache, cache_key=_ck("conv_glu"))

        # 4. depthwise_conv1d (FPGA 4-tap shifted-eltwise; default ON).
        # GEMMA4_AUDIO_FPGA_DWCONV=0 falls back to host F.conv1d for parity.
        self._aud_dw_conv1d_dispatch(
            layer_idx, in_addr=self.AUD_NORM_OUT, out_addr=self.AUD_NORM_OUT,
            cache=cache, cache_key=_ck("conv_dw"))

        # 5. conv_norm (RMSNorm with learned scale)
        self._compile_and_run_single("aud_conv_norm", lambda: self.rms_norm_core_dram(
            M=L_pad, N=H,
            A_DRAM_ADDR=self.AUD_NORM_OUT,
            OUTPUT_DRAM_ADDR=self.AUD_NORM_OUT,
            GAMMA_DRAM_ADDR=w["CONV_NORM"],
            gpr_M_reg=self._aud_mm_gpr(M=L_pad)),
            cache=cache, cache_key=_ck("conv_norm"))

        # 6. SiLU
        self._compile_and_run_single("aud_conv_silu", lambda: _aud_silu(
            self, L_pad, H, self.AUD_NORM_OUT, self.AUD_FFN_OUT, self.AUD_IDENTITY_H),
            cache=cache, cache_key=_ck("conv_silu"))

        # 7. linear_end (1024 → 1024)
        self._aud_clip_dram(self.AUD_FFN_OUT, (L_pad, H),
                              cr["CONV_LIN_END"]["in_min"], cr["CONV_LIN_END"]["in_max"])
        cle = w["CONV_LIN_END"]
        self._compile_and_run_single("aud_conv_lin_end", lambda: self.matmat_mul_core(
            M=L_pad, K=H, N=H,
            A_DRAM_ADDR=self.AUD_FFN_OUT,
            B_DRAM_ADDR=cle["data"],
            OUTPUT_DRAM_ADDR=self.AUD_FFN_OUT,
            is_B_quantized=True, data_type=TYPE.IF4,
            SCALE_DRAM_ADDR=cle["scale"],
            gpr_M_reg=self._aud_mm_gpr(M=L_pad)),
            cache=cache, cache_key=_ck("conv_lin_end"))
        self._aud_clip_dram(self.AUD_FFN_OUT, (L_pad, H),
                              cr["CONV_LIN_END"]["out_min"], cr["CONV_LIN_END"]["out_max"])

        # 8. Residual: out = AUD_FFN_OUT + AUD_RESIDUAL → IN_BUF
        self._compile_and_run_single("aud_conv_residual", lambda: _aud_eltwise_add(
            self, L_pad, H, self.AUD_RESIDUAL, self.AUD_FFN_OUT, IN_BUF),
            cache=cache, cache_key=_ck("conv_residual"))

    def _aud_split_2h_to_halves(self, label: str,
                                 src_2h: int, dst_a: int, dst_b: int,
                                 L_pad: int, H: int) -> None:
        """Split (L_pad, 2H) row-major buffer into (L_pad, H) gate and value
        buffers via strided SRAM copies. Compiled as one program."""
        bpe = self.bytes_per_element
        def _fn():
            # Row-by-row strided copy from interleaved to two contiguous halves.
            # Use accelerator_memory_to_sram with stride for the gather.
            # Strided gather: read H elements from each row at offset 0, jump 2H per row
            self.accelerator_memory_to_sram(
                accelerator_dram_address=src_2h,
                sram_address=0x00000,
                element_size=L_pad * H,
                stride_bytes_per_chunk=H * bpe,
                stride_jump_bytes=2 * H * bpe,
            )
            self.sram_to_accelerator_memory(0x00000, dst_a, L_pad * H)
            self.accelerator_memory_to_sram(
                accelerator_dram_address=src_2h + H * bpe,
                sram_address=0x00000,
                element_size=L_pad * H,
                stride_bytes_per_chunk=H * bpe,
                stride_jump_bytes=2 * H * bpe,
            )
            self.sram_to_accelerator_memory(0x00000, dst_b, L_pad * H)
        self._compile_and_run_single(label, _fn)

    def _aud_depthwise_conv1d_host(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Host fallback for Gemma4AudioCausalConv1d.
        x: [T, H], w: [H, k]. Returns [T, H] bf16."""
        T, H = x.shape
        k = w.shape[1]
        # Causal padding: left_pad = k-1, no right pad
        x_t = x.transpose(0, 1).unsqueeze(0).float()  # [1, H, T]
        x_p = F.pad(x_t, (k - 1, 0))
        w_t = w.unsqueeze(1).float()  # [H, 1, k] for groups=H depthwise conv
        out = F.conv1d(x_p, w_t, bias=None, stride=1, padding=0, groups=H)
        return out.squeeze(0).transpose(0, 1).to(torch.bfloat16)  # [T, H]

    def _aud_dw_conv1d_dispatch(self, layer_idx: int, *,
                                  in_addr: int, out_addr: int,
                                  cache, cache_key: str) -> None:
        """Phase 2A dispatcher. GEMMA4_AUDIO_FPGA_DWCONV=1 (default) → FPGA
        4-tap shifted-eltwise. =0 → host F.conv1d (legacy)."""
        H = self.AUD_H
        L_pad = self._aud_L_pad
        T = self._aud_num_frames
        if os.environ.get("GEMMA4_AUDIO_FPGA_DWCONV", "1") == "1":
            self._compile_and_run_single(
                f"aud_L{layer_idx}_conv_dw",
                lambda: self._emit_aud_dw_conv1d_fpga(layer_idx,
                                                       in_addr=in_addr,
                                                       out_addr=out_addr),
                cache=cache, cache_key=cache_key)
        else:
            x = self.dma_from_accelerator_memory(in_addr, (L_pad, H)).cpu()
            dw_w = self._aud_hf_layers[layer_idx]["dw_w"]
            y = self._aud_depthwise_conv1d_host(x[:T], dw_w)
            padded = torch.zeros(L_pad, H, dtype=torch.bfloat16)
            padded[:T] = y
            self.dma_to_accelerator_memory(out_addr, padded.contiguous())

    def _emit_aud_q_scale_fpga(self, layer_idx: int) -> None:
        """Phase 2B.b Q-scaling on FPGA: AUD_Q *= Q_SCALE_TILE (eltwise_mul).
        The tile is pre-computed in audio_tensor_init as
        q_scale * softplus(per_dim_scale) broadcast over (L_pad, H)."""
        from audio_primitives import (URAM_A_BASE, URAM_B_BASE, _row_chunk)
        H = self.AUD_H
        L_pad = self._aud_L_pad
        bpe = self.bytes_per_element
        scale_addr = self._aud_weight_addrs[layer_idx]["Q_SCALE_TILE"]
        M_chunk = _row_chunk(L_pad, H, divisor=2)
        rb = H * bpe
        for m_start in range(0, L_pad, M_chunk):
            m_take = min(M_chunk, L_pad - m_start)
            n = m_take * H
            self.accelerator_memory_to_sram(self.AUD_Q + m_start * rb,
                                             URAM_A_BASE, n)
            self.accelerator_memory_to_sram(scale_addr + m_start * rb,
                                             URAM_B_BASE, n)
            self.eltwise_mul_core(vector_A_sram_start_addr=URAM_A_BASE,
                                   vector_B_sram_start_addr=URAM_B_BASE,
                                   vector_C_sram_wb_addr=URAM_A_BASE,
                                   element_size=n)
            self.sram_to_accelerator_memory(URAM_A_BASE,
                                             self.AUD_Q + m_start * rb, n)

    def _emit_aud_k_scale_fpga(self) -> None:
        """Phase 2B.b K-scaling on FPGA: AUD_K *= AUD_K_SCALE (broadcast_mul)."""
        from audio_primitives import URAM_A_BASE, _row_chunk
        H = self.AUD_H
        L_pad = self._aud_L_pad
        bpe = self.bytes_per_element
        scale = float(self.AUD_K_SCALE)
        M_chunk = _row_chunk(L_pad, H, divisor=1)
        rb = H * bpe
        for m_start in range(0, L_pad, M_chunk):
            m_take = min(M_chunk, L_pad - m_start)
            n = m_take * H
            self.accelerator_memory_to_sram(self.AUD_K + m_start * rb,
                                             URAM_A_BASE, n)
            self.broadcast_mul(scalar=scale,
                                sram_start_addr=URAM_A_BASE,
                                sram_wb_addr=URAM_A_BASE,
                                element_size=n)
            self.sram_to_accelerator_memory(URAM_A_BASE,
                                             self.AUD_K + m_start * rb, n)

    def _emit_aud_attn_build_kctx_t(self, layer_idx: int) -> None:
        """Phase 2B.c step 1: build K_PADDED, K_CTX_BLOCKS, K_CTX_HEAD_BLOCKS on FPGA.

        K_PADDED          := zero-padded K (max_past leading zeros + AUD_K + trailing).
        K_CTX_BLOCKS      := per-block window of K_PADDED, shape (num_blocks, ctx_pad, H).
                             Rows [0:ctx_size) are real; [ctx_size:ctx_pad) are zero.
        K_CTX_T_BLOCKS    := per-(block, head) head slice of K_CTX_BLOCKS, shape
                             (num_blocks, NH, ctx_pad, HD).  NOTE: this is the
                             NATIVE B layout for FPGA matmul, which expects B as
                             (N, K) row-major and computes A @ B^T. So passing
                             K_ctx (ctx_pad, HD) as B yields A @ K_ctx^T = Q @ K^T.
                             No explicit transpose required.
        """
        from audio_primitives import (copy_dram_to_dram_chunked,
                                       URAM_A_BASE)
        H = self.AUD_H
        HD = self.AUD_HEAD_DIM
        NH = self.AUD_HEADS
        L_pad = self._aud_L_pad
        chunk = self.AUD_CHUNK
        ctx_size = self.AUD_CTX
        ctx_pad = self._aud_ctx_pad
        num_blocks = self._aud_num_blocks
        max_past = self.AUD_CTX_LEFT - 1
        bpe = self.bytes_per_element

        copy_dram_to_dram_chunked(self, self.AUD_K,
                                   self.AUD_K_PADDED + max_past * H * bpe,
                                   L_pad * H, row_n=H)
        for b in range(num_blocks):
            copy_dram_to_dram_chunked(self,
                self.AUD_K_PADDED + b * chunk * H * bpe,
                self.AUD_K_CTX_BLOCKS + b * ctx_pad * H * bpe,
                ctx_size * H, row_n=H)

        # Per-(block, head) extract head slice (ctx_pad, HD) into K_CTX_T_BLOCKS.
        # This is the per-(b, h) B matrix in (N=ctx_pad, K=HD) layout that
        # matmat_mul_core will read as A @ B^T = Q @ K^T.
        for b in range(num_blocks):
            for h in range(NH):
                src = self.AUD_K_CTX_BLOCKS + (b * ctx_pad * H + h * HD) * bpe
                self.accelerator_memory_to_sram(
                    accelerator_dram_address=src,
                    sram_address=URAM_A_BASE,
                    element_size=ctx_pad * HD,
                    stride_bytes_per_chunk=HD * bpe,
                    stride_jump_bytes=H * bpe)
                dst = self.AUD_K_CTX_T_BLOCKS + (b * NH + h) * ctx_pad * HD * bpe
                self.sram_to_accelerator_memory(URAM_A_BASE,
                    dst, ctx_pad * HD)

    def _emit_aud_attn_matrix_bd_unshifted(self, layer_idx: int) -> None:
        """Phase 2B.c step 2: matrix_bd_unshifted = Q @ rel_k^T (per head).

        REL_K_T per-head layout is (bd_unshifted_N, HD) with rows structured as:
          [0, chunk_pad):                    zero (pre-pad)  — survives across layers
          [chunk_pad, chunk_pad+num_pos_pad): real rel_k rows
          [chunk_pad+num_pos_pad, end):      zero (post-pad) — survives across layers
        Pre-pad/post-pad bands are zeroed once at audio_tensor_init. Per layer
        we only re-fill the real-row band; pre/post bands persist as zeros.

        matrix_bd_unshifted matmul N = bd_unshifted_N so output rows have the
        same pre-pad/real/post-pad structure. The rel-shift then reads
        bd_unshifted_N - chunk_pad = chunk_pad+num_pos_pad = ctx_pad columns
        starting at non-aligned offset (chunk_pad - r) in the source.
        """
        from audio_primitives import URAM_A_BASE
        H = self.AUD_H
        HD = self.AUD_HEAD_DIM
        NH = self.AUD_HEADS
        L_pad = self._aud_L_pad
        T_pad_padded = self._aud_T_pad_padded
        num_pos_pad = self._aud_num_pos_pad
        chunk_pad = self._aud_chunk_pad
        bd_N = self._aud_bd_unshifted_N
        bpe = self.bytes_per_element
        w = self._aud_weight_addrs[layer_idx]

        # (a) REL_K_OUT = POS_EMB_PADDED @ REL_K_PROJ  (num_pos_pad, H, H)
        rk = w["REL_K_PROJ"]
        self.matmat_mul_core(M=num_pos_pad, K=H, N=H,
            A_DRAM_ADDR=self.AUD_POS_EMB_PADDED,
            B_DRAM_ADDR=rk["data"],
            OUTPUT_DRAM_ADDR=self.AUD_REL_K_OUT,
            is_B_quantized=True, data_type=TYPE.IF4,
            SCALE_DRAM_ADDR=rk["scale"])

        # (b) Per-head strided extract of REL_K_OUT[:, h*HD:(h+1)*HD] (num_pos_pad, HD)
        # into REL_K_T[h, chunk_pad:chunk_pad+num_pos_pad, :]. The destination is
        # at row offset chunk_pad in a (bd_N, HD) per-head buffer; chunk_pad*HD*bpe
        # is VS-aligned (chunk_pad=VS) so the write is VS-aligned.
        for h in range(NH):
            src = self.AUD_REL_K_OUT + h * HD * bpe
            self.accelerator_memory_to_sram(
                accelerator_dram_address=src,
                sram_address=URAM_A_BASE,
                element_size=num_pos_pad * HD,
                stride_bytes_per_chunk=HD * bpe,
                stride_jump_bytes=H * bpe)
            dst = self.AUD_REL_K_T + (h * bd_N + chunk_pad) * HD * bpe
            self.sram_to_accelerator_memory(URAM_A_BASE, dst, num_pos_pad * HD)

        # (c) + (d) Per-head Q-extract + matmul (N = bd_N).
        for h in range(NH):
            src = self.AUD_Q + h * HD * bpe
            self.accelerator_memory_to_sram(
                accelerator_dram_address=src,
                sram_address=URAM_A_BASE,
                element_size=L_pad * HD,
                stride_bytes_per_chunk=HD * bpe,
                stride_jump_bytes=H * bpe)
            self.sram_to_accelerator_memory(URAM_A_BASE,
                self.AUD_Q_HEAD_FULL, L_pad * HD)

            B_addr = self.AUD_REL_K_T + h * bd_N * HD * bpe
            bd_addr = self.AUD_MATRIX_BD_UNSHIFTED + h * T_pad_padded * bd_N * bpe
            self.matmat_mul_core(M=T_pad_padded, K=HD, N=bd_N,
                A_DRAM_ADDR=self.AUD_Q_HEAD_FULL,
                B_DRAM_ADDR=B_addr,
                OUTPUT_DRAM_ADDR=bd_addr)

    def _emit_aud_attn_rel_shift(self, layer_idx: int) -> None:
        """Phase 2B.c step 3: build AUD_MATRIX_BD_SHIFTED via per-(b, h, r) matmul.

        Reading non-aligned source rows fails because the FPGA rounds the source
        DRAM address DOWN to a 32-byte (16-element) boundary, scrambling the
        shift for r in [1, 11]. Instead we use a tiny matmul per row:

            bd_shifted[b, h, r, :ctx_pad] = bd_unshifted[h, b*chunk+r, :num_pos_pad]
                                            @ M_r[:num_pos_pad, :ctx_pad].T

        where M_r[p, c] = 1 if c == p+r AND p < num_pos else 0. The shift
        matrices M_r are pre-built at audio_tensor_init in (N=ctx_pad, K=num_pos_pad)
        layout so FPGA's native A @ B^T computes the right thing.
        """
        NH = self.AUD_HEADS
        chunk = self.AUD_CHUNK
        ctx_pad = self._aud_ctx_pad
        chunk_pad = self._aud_chunk_pad
        num_blocks = self._aud_num_blocks
        T_pad_padded = self._aud_T_pad_padded
        num_pos_pad = self._aud_num_pos_pad
        bd_N = self._aud_bd_unshifted_N
        bpe = self.bytes_per_element

        for b in range(num_blocks):
            for h in range(NH):
                for r in range(chunk):
                    # A = bd_unshifted[h, b*chunk+r, chunk_pad:chunk_pad+num_pos_pad]
                    # The real values live at cols [chunk_pad, chunk_pad+num_pos_pad)
                    # within bd_unshifted's row (zero-bands flank them).
                    A = (self.AUD_MATRIX_BD_UNSHIFTED
                         + ((h * T_pad_padded + b * chunk + r) * bd_N + chunk_pad) * bpe)
                    B = self.AUD_REL_SHIFT_M + r * ctx_pad * num_pos_pad * bpe
                    C = (self.AUD_MATRIX_BD_SHIFTED
                         + ((b * NH + h) * chunk_pad + r) * ctx_pad * bpe)
                    self.matmat_mul_core(M=1, K=num_pos_pad, N=ctx_pad,
                        A_DRAM_ADDR=A, B_DRAM_ADDR=B, OUTPUT_DRAM_ADDR=C)

    def _emit_aud_attn_matrix_ac_probe_first(self, layer_idx: int) -> None:
        """Emits only block 0 head 0 Q-extract + matmul for debugging."""
        from audio_primitives import URAM_A_BASE
        H = self.AUD_H
        HD = self.AUD_HEAD_DIM
        chunk = self.AUD_CHUNK
        ctx_pad = self._aud_ctx_pad
        chunk_pad = self._aud_chunk_pad
        bpe = self.bytes_per_element
        b, h = 0, 0
        src = self.AUD_Q + (b * chunk * H + h * HD) * bpe
        self.accelerator_memory_to_sram(
            accelerator_dram_address=src,
            sram_address=URAM_A_BASE,
            element_size=chunk * HD,
            stride_bytes_per_chunk=HD * bpe,
            stride_jump_bytes=H * bpe)
        self.sram_to_accelerator_memory(URAM_A_BASE,
            self.AUD_Q_HEAD_BLOCK, chunk * HD)
        K_B = self.AUD_K_CTX_T_BLOCKS  # b=0, h=0
        ac = self.AUD_MATRIX_AC
        self.matmat_mul_core(M=chunk_pad, K=HD, N=ctx_pad,
            A_DRAM_ADDR=self.AUD_Q_HEAD_BLOCK,
            B_DRAM_ADDR=K_B,
            OUTPUT_DRAM_ADDR=ac)

    def _emit_aud_attn_matrix_ac(self, layer_idx: int) -> None:
        """Phase 2B.c step 2: per-(block, head) Q[b,h] @ K_CTX^T[b,h] -> matrix_ac[b,h].

        Q extraction: strided DMA AUD_Q[b*chunk:(b+1)*chunk, h*HD:(h+1)*HD] into
        AUD_Q_HEAD_BLOCK (top valid_rows rows; rows [valid_rows:chunk_pad] forced to
        zero per-iteration so the last partial block sees the zeros HF F.pad inserts).

        Matmul: (chunk_pad=64, HD) @ (N=ctx_pad=64, K=HD) -> (chunk_pad, ctx_pad).
        FPGA computes A @ B^T natively (B in (N, K) layout). Only output rows
        [0:chunk) × cols [0:ctx_size) are semantically valid.
        """
        from audio_primitives import URAM_A_BASE
        H = self.AUD_H
        HD = self.AUD_HEAD_DIM
        NH = self.AUD_HEADS
        L_pad = self._aud_L_pad
        chunk = self.AUD_CHUNK
        ctx_pad = self._aud_ctx_pad
        chunk_pad = self._aud_chunk_pad
        num_blocks = self._aud_num_blocks
        bpe = self.bytes_per_element

        for b in range(num_blocks):
            valid_rows = min(chunk, L_pad - b * chunk)
            for h in range(NH):
                # Zero AUD_Q_HEAD_BLOCK[valid_rows:chunk_pad] first (matters only
                # for the partial last block, but cheap so do it always).
                if valid_rows < chunk_pad:
                    fill_elems = (chunk_pad - valid_rows) * HD
                    self.accelerator_memory_to_sram(
                        accelerator_dram_address=self.AUD_ZEROS_CHUNK_HD,
                        sram_address=URAM_A_BASE,
                        element_size=fill_elems)
                    self.sram_to_accelerator_memory(URAM_A_BASE,
                        self.AUD_Q_HEAD_BLOCK + valid_rows * HD * bpe,
                        fill_elems)
                if valid_rows > 0:
                    src = self.AUD_Q + (b * chunk * H + h * HD) * bpe
                    self.accelerator_memory_to_sram(
                        accelerator_dram_address=src,
                        sram_address=URAM_A_BASE,
                        element_size=valid_rows * HD,
                        stride_bytes_per_chunk=HD * bpe,
                        stride_jump_bytes=H * bpe)
                    self.sram_to_accelerator_memory(URAM_A_BASE,
                        self.AUD_Q_HEAD_BLOCK, valid_rows * HD)

                # B = K_ctx[b, h] in (N=ctx_pad, K=HD); FPGA computes A @ B^T.
                K_B = self.AUD_K_CTX_T_BLOCKS + (b * NH + h) * ctx_pad * HD * bpe
                ac = self.AUD_MATRIX_AC + (b * NH + h) * chunk_pad * ctx_pad * bpe
                self.matmat_mul_core(M=chunk_pad, K=HD, N=ctx_pad,
                    A_DRAM_ADDR=self.AUD_Q_HEAD_BLOCK,
                    B_DRAM_ADDR=K_B,
                    OUTPUT_DRAM_ADDR=ac)

    def _emit_aud_dw_conv1d_fpga(self, layer_idx: int, *,
                                   in_addr: int, out_addr: int) -> None:
        """4-tap shifted-eltwise depthwise causal conv1d on FPGA.

        For causal conv with kernel size K and per-channel weight w[c, t]:
            y[r, c] = sum_{t=0..K-1} w[c, t] * x[r - (K-1-t), c]   for r ≥ K-1-t
                                                                   else 0

        Implementation:
          1. Pre-stage SHIFT[0:K-1, :] = 0 (copy from AUD_DW_ZERO_KM1).
          2. For each tap t = 0 .. K-1, with shift_t = K-1-t:
               Build SHIFT[shift_t:L_pad, :] = x[0:L_pad-shift_t, :] via a
               DRAM-to-DRAM copy. Rows [0:shift_t] are already zero (from
               step 1 for shift_t = K-1, and preserved across taps because we
               iterate from largest shift to smallest, so each tap's zero-pad
               region is a subset of the previous tap's untouched region).
               Then SCRATCH = SHIFT * tap_tile[t]   (eltwise_mul).
               For t == 0: ACCUM = SCRATCH (direct write).
               For t >  0: ACCUM = ACCUM + SCRATCH (eltwise_add into out).
          3. Final ACCUM lives at ``out_addr``.

        We use AUD_FFN_MID (L_pad × 4H) as scratch:
          SHIFT   = AUD_FFN_MID + 0
          SCRATCH = AUD_FFN_MID + L_pad*H*bpe
        AUD_FFN_OUT is a third (L_pad × H) buffer used to hold ``in_addr``'s
        contents when in_addr == out_addr (so we don't read after partial
        write). Because in_addr == AUD_NORM_OUT == out_addr in the conv
        pipeline, we copy AUD_NORM_OUT into AUD_FFN_OUT once at the top and
        read x from AUD_FFN_OUT throughout.
        """
        H = self.AUD_H
        L_pad = self._aud_L_pad
        K = self.AUD_CONV_K
        bpe = self.bytes_per_element
        row_bytes = H * bpe
        w = self._aud_weight_addrs[layer_idx]
        tap_tiles = w["CONV_DW_TAP_TILES"]
        assert len(tap_tiles) == K, f"tap_tiles len {len(tap_tiles)} != K {K}"

        from audio_primitives import (
            eltwise_add_core_dram, copy_dram_to_dram_chunked,
            URAM_A_BASE, URAM_B_BASE, _row_chunk,
        )

        # Stage x into AUD_FFN_OUT (safe vs. in-place out_addr).
        x_buf = self.AUD_FFN_OUT
        copy_dram_to_dram_chunked(self, in_addr, x_buf, L_pad * H, row_n=H)

        shift_buf = self.AUD_FFN_MID
        scratch_buf = self.AUD_FFN_MID + L_pad * H * bpe
        accum_buf = out_addr

        # Pre-zero SHIFT[0:K-1, :] using the cached zero tile. After this,
        # every subsequent copy into SHIFT[shift_t:L_pad, :] preserves the
        # top zero region because shift_t decreases monotonically.
        copy_dram_to_dram_chunked(
            self, self.AUD_DW_ZERO_KM1, shift_buf, (K - 1) * H, row_n=H)

        def _eltwise_mul_dram(src_a: int, src_b: int, dst: int, M: int, N: int):
            """eltwise_mul over (M, N) bf16 DRAM tensors, chunked."""
            M_chunk = _row_chunk(M, N, divisor=2)
            rb = N * bpe
            for m_start in range(0, M, M_chunk):
                m_take = min(M_chunk, M - m_start)
                n = m_take * N
                self.accelerator_memory_to_sram(src_a + m_start * rb,
                                                URAM_A_BASE, n)
                self.accelerator_memory_to_sram(src_b + m_start * rb,
                                                URAM_B_BASE, n)
                self.eltwise_mul_core(vector_A_sram_start_addr=URAM_A_BASE,
                                       vector_B_sram_start_addr=URAM_B_BASE,
                                       vector_C_sram_wb_addr=URAM_A_BASE,
                                       element_size=n)
                self.sram_to_accelerator_memory(URAM_A_BASE,
                                                dst + m_start * rb, n)

        for t in range(K):
            shift_t = (K - 1) - t  # K-1, K-2, ..., 0
            # Place x[0:L_pad-shift_t] into SHIFT[shift_t:L_pad].
            n_rows = L_pad - shift_t
            if n_rows > 0:
                copy_dram_to_dram_chunked(
                    self, x_buf,
                    shift_buf + shift_t * row_bytes,
                    n_rows * H, row_n=H)
            # SCRATCH = SHIFT * tap_tile[t]
            _eltwise_mul_dram(shift_buf, tap_tiles[t], scratch_buf, L_pad, H)
            if t == 0:
                # ACCUM := SCRATCH
                copy_dram_to_dram_chunked(self, scratch_buf, accum_buf,
                                           L_pad * H, row_n=H)
            else:
                # ACCUM += SCRATCH
                eltwise_add_core_dram(self, M=L_pad, N=H,
                                       A_DRAM_ADDR=accum_buf,
                                       B_DRAM_ADDR=scratch_buf,
                                       OUTPUT_DRAM_ADDR=accum_buf)

    def compile_audio_layer_ffn2(self, layer_idx: int) -> None:
        """Run the FFN2 macaron half. Identical to FFN1 except for the
        weight keys (FF2_*) so we just call _compile_audio_ffn_macaron with
        the FF2 weight prefix."""
        self._compile_audio_ffn_macaron(layer_idx, prefix="FF2")

    def _compile_audio_ffn_macaron(self, layer_idx: int, *, prefix: str) -> None:
        """Generic Gemma4AudioFeedForward macaron half: works for FFN1 and
        FFN2. ``prefix`` selects the weight keys: 'FF1' or 'FF2'."""
        H = self.AUD_H
        FF = self.AUD_FFN
        L_pad = self._aud_L_pad
        w = self._aud_weight_addrs[layer_idx]
        cr = self._aud_clip_ranges[layer_idx]
        cache = getattr(self, "_aud_program_cache", None)
        plow = prefix.lower()

        def _ck(label: str) -> str:
            return f"aud_L{layer_idx}_{label}"

        IN_BUF = self.AUD_IO_A  # all audio layers operate in place on IO_A

        self._aud_copy_buf(f"{plow}_save_residual",
                            IN_BUF, self.AUD_RESIDUAL, L_pad * H, row_n=H)

        self._compile_and_run_single(f"aud_{plow}_pre_norm", lambda: self.rms_norm_core_dram(
            M=L_pad, N=H,
            A_DRAM_ADDR=IN_BUF,
            OUTPUT_DRAM_ADDR=self.AUD_NORM_OUT,
            GAMMA_DRAM_ADDR=w[f"{prefix}_PRE_NORM"],
            gpr_M_reg=self._aud_mm_gpr(M=L_pad)),
            cache=cache, cache_key=_ck(f"{plow}_pre_norm"))

        self._aud_clip_dram(self.AUD_NORM_OUT, (L_pad, H),
                              cr[f"{prefix}_W1"]["in_min"], cr[f"{prefix}_W1"]["in_max"])
        w1 = w[f"{prefix}_W1"]
        self._compile_and_run_single(f"aud_{plow}_w1", lambda: self.matmat_mul_core(
            M=L_pad, K=H, N=FF,
            A_DRAM_ADDR=self.AUD_NORM_OUT,
            B_DRAM_ADDR=w1["data"],
            OUTPUT_DRAM_ADDR=self.AUD_FFN_MID,
            is_B_quantized=True, data_type=TYPE.IF4,
            SCALE_DRAM_ADDR=w1["scale"],
            gpr_M_reg=self._aud_mm_gpr(M=L_pad)),
            cache=cache, cache_key=_ck(f"{plow}_w1"))
        self._aud_clip_dram(self.AUD_FFN_MID, (L_pad, FF),
                              cr[f"{prefix}_W1"]["out_min"], cr[f"{prefix}_W1"]["out_max"])

        self._compile_and_run_single(f"aud_{plow}_silu", lambda: _aud_silu(
            self, L_pad, FF, self.AUD_FFN_MID, self.AUD_SILU_OUT, self.AUD_IDENTITY_FF),
            cache=cache, cache_key=_ck(f"{plow}_silu"))

        self._aud_clip_dram(self.AUD_SILU_OUT, (L_pad, FF),
                              cr[f"{prefix}_W2"]["in_min"], cr[f"{prefix}_W2"]["in_max"])
        w2 = w[f"{prefix}_W2"]
        self._compile_and_run_single(f"aud_{plow}_w2", lambda: self.matmat_mul_core(
            M=L_pad, K=FF, N=H,
            A_DRAM_ADDR=self.AUD_SILU_OUT,
            B_DRAM_ADDR=w2["data"],
            OUTPUT_DRAM_ADDR=self.AUD_FFN_OUT,
            is_B_quantized=True, data_type=TYPE.IF4,
            SCALE_DRAM_ADDR=w2["scale"],
            gpr_M_reg=self._aud_mm_gpr(M=L_pad)),
            cache=cache, cache_key=_ck(f"{plow}_w2"))
        self._aud_clip_dram(self.AUD_FFN_OUT, (L_pad, H),
                              cr[f"{prefix}_W2"]["out_min"], cr[f"{prefix}_W2"]["out_max"])

        self._compile_and_run_single(f"aud_{plow}_post_norm", lambda: self.rms_norm_core_dram(
            M=L_pad, N=H,
            A_DRAM_ADDR=self.AUD_FFN_OUT,
            OUTPUT_DRAM_ADDR=self.AUD_FFN_OUT,
            GAMMA_DRAM_ADDR=w[f"{prefix}_POST_NORM"],
            gpr_M_reg=self._aud_mm_gpr(M=L_pad)),
            cache=cache, cache_key=_ck(f"{plow}_post_norm"))

        self._compile_and_run_single(f"aud_{plow}_half_residual",
            lambda: _aud_half_step(self, L_pad, H,
                self.AUD_RESIDUAL, self.AUD_FFN_OUT, IN_BUF),
            cache=cache, cache_key=_ck(f"{plow}_half_residual"))

    def compile_audio_layer_norm_out(self, layer_idx: int) -> None:
        """Final per-layer RMSNorm. Writes back into IN_BUF in place."""
        H = self.AUD_H
        L_pad = self._aud_L_pad
        w = self._aud_weight_addrs[layer_idx]
        cache = getattr(self, "_aud_program_cache", None)
        IN_BUF = self.AUD_IO_A  # all audio layers operate in place on IO_A
        self._compile_and_run_single("aud_norm_out", lambda: self.rms_norm_core_dram(
            M=L_pad, N=H,
            A_DRAM_ADDR=IN_BUF,
            OUTPUT_DRAM_ADDR=IN_BUF,
            GAMMA_DRAM_ADDR=w["NORM_OUT"],
            gpr_M_reg=self._aud_mm_gpr(M=L_pad)),
            cache=cache, cache_key=f"aud_L{layer_idx}_norm_out")

    def run_audio_layer(self, layer_idx: int) -> int:
        """Run a complete Conformer layer in place: FFN1 → attn → conv →
        FFN2 → norm_out. Returns the DRAM address holding the result.

        All sub-blocks read and write AUD_IO_A; we never ping-pong, so the
        return value is always AUD_IO_A. Caller can DMA from there.
        """
        self.compile_audio_layer_ffn1(layer_idx)
        self.compile_audio_layer_attn(layer_idx)
        self.compile_audio_layer_conv(layer_idx)
        self.compile_audio_layer_ffn2(layer_idx)
        self.compile_audio_layer_norm_out(layer_idx)
        return self.AUD_IO_A

    def _aud_copy_buf(self, label: str, src: int, dst: int, n_elems: int,
                       row_n: int | None = None) -> None:
        """Copy n_elems bf16 elements from src DRAM to dst DRAM via SRAM,
        chunked so URAM_A doesn't overflow on long buffers.

        row_n: optional row width for clean row-aligned chunking.
        """
        def _fn():
            _aud_copy_chunked(self, src, dst, n_elems, row_n=row_n)
        self._compile_and_run_single(label, _fn)

    def get_embedding_for_tokens(self, token_ids: list[int] | tuple) -> torch.Tensor:
        """Return (len(token_ids), vector_length) bfloat16 tensor from self.embedding_weight (HF, scale applied)."""
        tid_t = torch.tensor(token_ids, dtype=torch.long)
        out = torch.zeros(len(token_ids), self.vector_length, dtype=torch.bfloat16)
        valid = tid_t < self.embedding_weight.shape[0]
        out[valid] = self.embedding_weight[tid_t[valid]]
        return out

    def _load_rope_host(self, rope_theta: float | None = None, rope_local_base: float | None = None) -> None:
        """Generate RoPE (cos, cos, -sin, sin) on host and write to DRAM. Uses config for sizes and num_positions.
        LOCAL: head_dim=256, full rotation. GLOBAL: head_dim=512, partial rotation (first 128 dims)."""
        rope_cfg = self._cfg["special"]["rope"]
        theta = rope_theta if rope_theta is not None else rope_cfg["theta"]
        local_base = rope_local_base if rope_local_base is not None else rope_cfg["local_base"]
        num_rope_positions = rope_cfg["num_positions"]
        partial_rotary_factor = rope_cfg["partial_rotary_factor_global"]

        # LOCAL RoPE: head_dim_sliding=256, full rotation, D=128
        D_local = self.head_dim_sliding // 2  # 128
        inv_freq_local = 1.0 / (local_base ** (torch.arange(D_local, dtype=torch.float32) / D_local))
        pos = torch.arange(num_rope_positions, dtype=torch.float32)
        freqs_local = torch.outer(pos, inv_freq_local)
        cos_local = freqs_local.cos().to(torch.bfloat16)
        sin_local = freqs_local.sin().to(torch.bfloat16)
        rope_local = torch.cat([cos_local, cos_local, -sin_local, sin_local], dim=1)
        sz = self.weight_defs["ROPE_LOCAL_SIZE"]
        raw = rope_local.contiguous().view(torch.uint8).numpy().tobytes()
        raw = (raw + b"\x00" * sz)[:sz]
        addr = self.allocate_params_dram(sz)
        self.dma_write(DMA_DEVICE_H2C, addr, raw, sz)
        self.DRAM_ADDR_ROPE_LOCAL = addr

        # GLOBAL RoPE: head_dim=512, partial_rotary_factor=0.25, rotary_dims=128, D=64
        rotary_dims = int(self.head_dim * partial_rotary_factor)  # 128
        D_global = rotary_dims // 2  # 64
        inv_freq_global = 1.0 / (theta ** (torch.arange(D_global, dtype=torch.float32) / D_global))
        freqs_global = torch.outer(pos, inv_freq_global)
        cos_global = freqs_global.cos().to(torch.bfloat16)
        sin_global = freqs_global.sin().to(torch.bfloat16)
        rope_global = torch.cat([cos_global, cos_global, -sin_global, sin_global], dim=1)
        sz = self.weight_defs["ROPE_GLOBAL_SIZE"]
        raw = rope_global.contiguous().view(torch.uint8).numpy().tobytes()
        raw = (raw + b"\x00" * sz)[:sz]
        addr = self.allocate_params_dram(sz)
        self.dma_write(DMA_DEVICE_H2C, addr, raw, sz)
        self.DRAM_ADDR_ROPE_GLOBAL = addr

    def _load_host_weights_from_combined_bin(self, host_section: dict, base_offset: int) -> None:
        """mmap the combined weights bin and create read-only torch tensor
        views over the host section. Zero-copy: nothing materializes in
        RSS until a row is indexed.

        `host_section["manifest"]` gives tensor offsets RELATIVE to the
        host section start; we add `base_offset` (the host section's
        absolute file offset) when creating the view.
        """
        sub = host_section["manifest"]

        def _view(section_name: str) -> torch.Tensor:
            s = sub[section_name]
            shape = tuple(s["shape"])
            n_elems = 1
            for d in shape:
                n_elems *= d
            return torch.frombuffer(
                self.weight_bin,
                dtype=torch.bfloat16,
                count=n_elems,
                offset=base_offset + s["offset"],
            ).reshape(shape)

        self.embed_tokens_per_layer_weight = _view("embed_tokens_per_layer")
        self.per_layer_model_proj_weight   = _view("per_layer_model_proj")
        self.per_layer_proj_norm_weight    = _view("per_layer_proj_norm")
        self._layer_scalars = list(sub["layer_scalars"])
        self._kv_shared_map = {int(k): int(v) for k, v in sub.get("kv_shared_map", {}).items()}
        print(f"[weight_init] host section mmap'd at file offset 0x{base_offset:X}: "
              f"embed_tokens_per_layer={tuple(self.embed_tokens_per_layer_weight.shape)} bf16 "
              f"({sub['embed_tokens_per_layer']['size']/1024**3:.2f} GiB, page-cached on demand)")

    def weight_init(self) -> None:
        """Ensure weight bin exists (generate from HF if missing), then mmap it
        and initialize FPGA DRAM: embedding, layers from bin, RoPE, OUTPUT_NORM/LM_HEAD.

        Host-side tensors needed for per-layer-input computation
        (per_layer_embed_tokens, per_layer_model_proj, per_layer_proj_norm,
        layer_scalars, kv_shared_map) come from `host_weights.bin` if it
        exists, mmap'd so RSS stays minimal. Otherwise we fall back to
        loading the HF model — which costs 6-12 GB host RAM and is OOM on
        a 16 GB Raspberry Pi. The first run on a beefier machine should
        generate the side-cache so subsequent runs anywhere can skip the
        HF model entirely.
        """
        import mmap as _mmap

        full_path = os.path.join(self.script_dir, self._weights_bin_rel)
        if os.path.exists(full_path):
            print(f"Weight bin exists, skip generation: {full_path}")
        else:
            print(f"Weight bin not found, generating: {full_path}")
            weight_bin_generate(output_path=full_path)

        # mmap the weight bin — read-only, OS pages in only what's touched.
        # Replaces a 2.4 GB f.read() that pinned the whole bin in RSS.
        self._weight_bin_fp = open(full_path, "rb")
        self.weight_bin = _mmap.mmap(self._weight_bin_fp.fileno(), 0,
                                     prot=_mmap.PROT_READ)

        # Master manifest describes the three sections inside the combined
        # weights bin (lm, vision, host). It MUST be present alongside the
        # bin; we don't fall back to HF here (this is run-from-bin).
        master_meta_path = full_path.rsplit(".", 1)[0] + ".json"
        if not os.path.exists(master_meta_path):
            raise RuntimeError(
                f"run_from_bin: combined weights manifest is required and not found.\n"
                f"  Expected: {master_meta_path}\n"
                "Generate on a beefier host by running gemma4_e2b_test.py once, then copy "
                "gemma4_e2b_bin/ to this host.")
        with open(master_meta_path, "r") as f:
            self._weights_master = json.load(f)

        # Embedding: a zero-copy mmap view directly into the weight bin.
        # No 770 MB host allocation; only the touched rows (one per decode
        # token, ~3 KB) cost RSS. Read-only is fine because we only do
        # `embedding_weight[token_ids]` lookups.
        emb_cfg = self._cfg["special"]["embedding"]
        token_embd_offset = _parse_offset(emb_cfg["token_embd_offset"])
        vocab_size  = self.EMBEDDING_ELEMENTS         # 262144
        emb_dim     = self.vector_length              # 1536
        self.embedding_weight = torch.frombuffer(
            self.weight_bin,
            dtype=torch.bfloat16,
            count=vocab_size * emb_dim,
            offset=token_embd_offset,
        ).reshape(vocab_size, emb_dim)

        host_section = self._weights_master["host_section"]
        self._load_host_weights_from_combined_bin(host_section, host_section["offset"])

        # Tokenizer: load from the bundled subset in gemma4_e2b_bin/tokenizer/.
        # That folder is produced by gemma4_e2b_test.py's tokenizer_subset_extract
        # and holds just the files HF needs (tokenizer.json, tokenizer_config.json,
        # chat_template.jinja, processor_config.json). The full HF model
        # directory (gemma-4-E2B-it/) with its ~10 GB safetensors is NOT
        # required at deploy time. Fall back to that directory only if the
        # subset is missing (legacy bin layouts).
        tok_subset = os.path.join(self.script_dir, "gemma4_e2b_bin", "tokenizer")
        if os.path.exists(os.path.join(tok_subset, "tokenizer.json")):
            tok_dir = tok_subset
        else:
            tok_dir = os.path.join(self.script_dir, self._cfg["paths"]["hf_model_dir"])
            if not os.path.exists(os.path.join(tok_dir, "tokenizer.json")):
                raise RuntimeError(
                    f"Tokenizer files not found. Expected either:\n"
                    f"  {tok_subset}/tokenizer.json (preferred)\n"
                    f"  {tok_dir}/tokenizer.json (legacy)\n"
                    f"Generate the tokenizer subset by running gemma4_e2b_test.py "
                    f"once and copy gemma4_e2b_bin/tokenizer/ to this host.")
        self.tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True, local_files_only=True)

        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        base_layer0 = self.weight_defs["BLK0_ATTN_NORM_WEIGHT"]
        blk0_regions = [
            (s["key"], f"{s['key']}_SIZE", s["attr"])
            for s in self._cfg["layers"]["structure"]
        ]
        non_layer = [
            (s["key"], f"{s['key']}_SIZE", s["attr"])
            for s in self._cfg["layers"]["non_layer"]
            if s["key"] not in ("ROPE_LOCAL", "ROPE_GLOBAL")  # RoPE loaded via _load_rope_host()
        ]

        last_structure_key = self._cfg["layers"]["structure"][-1]["key"]
        layer0_end = (self.weight_defs[last_structure_key] - base_layer0
                      + self.weight_defs[f"{last_structure_key}_SIZE"])
        assert layer0_end <= LAYER_WEIGHT_SIZE, (
            f"Layer 0 size overflow: computed {layer0_end} > LAYER_WEIGHT_SIZE {LAYER_WEIGHT_SIZE}"
        )

        print(f"\n--- Loading weights to DRAM ---")
        layers_total = self.LAYER_SIZE * LAYER_WEIGHT_SIZE
        layers_base_dram = self.allocate_params_dram(layers_total)
        load_t0 = time.perf_counter()
        for layer_idx in range(self.LAYER_SIZE):
            if layer_idx > 0 and layer_idx % 10 == 0:
                print(f"    layer {layer_idx}/{self.LAYER_SIZE} loaded ({time.perf_counter()-load_t0:.1f}s)")
            for off_key, sz_key, attr in blk0_regions:
                off = self.weight_defs[off_key]
                sz = self.weight_defs[sz_key]
                bin_off = off + layer_idx * LAYER_WEIGHT_SIZE
                raw = self.weight_bin[bin_off : bin_off + sz]
                offset_in_layer = off - base_layer0
                dram_addr = layers_base_dram + layer_idx * LAYER_WEIGHT_SIZE + offset_in_layer
                self.dma_write(DMA_DEVICE_H2C, dram_addr, raw, sz)
            if layer_idx == 0:
                for off_key, sz_key, attr in blk0_regions:
                    off = self.weight_defs[off_key]
                    offset_in_layer = off - base_layer0
                    setattr(self, attr, layers_base_dram + offset_in_layer)
        print(f"  Loaded {self.LAYER_SIZE} layers ({layers_total/(1024*1024):.1f} MB)")

        for off_key, sz_key, attr in non_layer:
            off = self.weight_defs[off_key]
            sz = self.weight_defs[sz_key]
            raw = self.weight_bin[off : off + sz]
            addr = self.allocate_params_dram(sz)
            self.dma_write(DMA_DEVICE_H2C, addr, raw, sz)
            setattr(self, attr, addr)

        self._load_rope_host()
        print(f"  Total weight DRAM: {self.get_params_dram_usage()/(1024*1024):.1f} MB")
        print("Tokenizer loaded.")

    def tensor_init(self) -> None:
        """Initialize hardware DRAM for gemma4 E2B model (layer-wise overlap except for kv cache).
        KV cache uses max head_dim (512) for uniform sizing."""
        seq_len = self.MAX_CONTEXT_SIZE
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        # Flash attention scratch/bias buffers are only used at full capacity
        # during prefill. Sizing them for MAX_CONTEXT_SIZE wastes ~200+ MB that
        # the decoder program bin would otherwise overlap with. Use
        # max_prefill_seq_len (capped at MAX_CONTEXT_SIZE) instead.
        prefill_seq_len = min(self.max_prefill_seq_len, self.MAX_CONTEXT_SIZE)
        prefill_q_seq_len = prefill_seq_len * self.group_size
        prefill_aligned_seq_len = ((prefill_q_seq_len + 63) // 64) * 64

        # Build compact KV slot map: only layers that own KV state get a slot.
        # KV-shared layers point at their reference layer's slot, so L15-34 do not
        # consume cache space (saves ~40 MB at MAX_CONTEXT_SIZE=1024).
        non_shared_layers = [l for l in range(self.LAYER_SIZE) if l not in self._kv_shared_map]
        self._kv_slot_for_layer = {}
        for slot, l in enumerate(non_shared_layers):
            self._kv_slot_for_layer[l] = slot
        for shared_l, ref_l in self._kv_shared_map.items():
            self._kv_slot_for_layer[shared_l] = self._kv_slot_for_layer[ref_l]
        self._num_kv_slots = len(non_shared_layers)
        _kv_saved = (self.LAYER_SIZE - self._num_kv_slots) * self.MAX_CONTEXT_SIZE * self.k_size * 2  # K+V
        print(f"KV cache: {self._num_kv_slots} unique slots (of {self.LAYER_SIZE} layers), saved {_kv_saved / (1024*1024):.1f} MB via KV sharing")
        # Allocate shared memory for k v cache (k rope and v projection) and zero pad for decoder use:
        # Uses max head_dim (512) = self.k_size for uniform sizing
        self.LAYER0_V_DRAM = self.allocate_tensor_dram(self._num_kv_slots * self.MAX_CONTEXT_SIZE * self.k_size)
        self.LAYER0_K_ROPE_DRAM = self.allocate_tensor_dram(self._num_kv_slots * self.MAX_CONTEXT_SIZE * self.k_size)
        zero_pad = torch.zeros(self._num_kv_slots * self.MAX_CONTEXT_SIZE * self.k_size, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_K_ROPE_DRAM, zero_pad)
        # Allocate memory for constant zero tensor, identity matrix, and bias:
        zero_add = torch.zeros(seq_len * self.head_dim * self.bytes_per_element, dtype=torch.bfloat16)
        self.ZERO_DRAM_ADDR = self.allocate_tensor_dram(seq_len * self.head_dim * self.bytes_per_element)
        self.dma_to_accelerator_memory(self.ZERO_DRAM_ADDR, zero_add)
        self.IDENTITY_DRAM_ADDR = self.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * self.bytes_per_element)
        self.dma_to_accelerator_memory(self.IDENTITY_DRAM_ADDR, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))
        # Allocate memory for flash attention and zero pad (sized for prefill only):
        self.LAYER0_FLASH_Q_DRAM = self.allocate_tensor_dram(prefill_aligned_seq_len * self.head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_K_DRAM = self.allocate_tensor_dram(prefill_aligned_seq_len * self.head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_V_DRAM = self.allocate_tensor_dram(prefill_aligned_seq_len * self.head_dim * self.bytes_per_element)
        zero_pad = torch.zeros(prefill_aligned_seq_len * self.head_dim * self.bytes_per_element, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_Q_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_K_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_V_DRAM, zero_pad)
        # Allocate memory for layer intermediate tensors:
        self.LAYER0_INPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_Q_DRAM = self.allocate_tensor_dram(seq_len * self.q_size)
        self.LAYER0_K_DRAM = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_K_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_Q_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.q_size)
        self.LAYER0_FLASH_OUTPUT_DRAM = self.allocate_tensor_dram(prefill_aligned_seq_len * self.head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_SCRATCH_DRAM = self.allocate_tensor_dram(max(self.head_dim, UE_FMAX_CONTEXT_SIZE) * prefill_aligned_seq_len * 2 + self.head_dim * prefill_aligned_seq_len * 2)
        # Two flash-attention bias buffers: full-attention layers attend to
        # the entire causal window, sliding-attention layers are limited to
        # `sliding_window` tokens. compile_prefill / compile_decoder pick the
        # right address per layer; run_prefill / run_decoder upload both.
        # Sized for prefill (not MAX_CONTEXT_SIZE) — see comment above.
        self.LAYER0_FLASH_BIAS_FULL_DRAM = self.allocate_tensor_dram(prefill_aligned_seq_len * prefill_aligned_seq_len * self.bytes_per_element)
        self.LAYER0_FLASH_BIAS_SLIDING_DRAM = self.allocate_tensor_dram(prefill_aligned_seq_len * prefill_aligned_seq_len * self.bytes_per_element)
        # Backwards-compat alias (older callers use the singular name).
        self.LAYER0_FLASH_BIAS_DRAM = self.LAYER0_FLASH_BIAS_FULL_DRAM
        # Dynamic-PBI flash-attention scratch buffer for the bucket dispatcher.
        # Size MUST match gemma4_e2b_test.py (which compiles the bin) so
        # downstream tensor addresses agree — the bin's ISA bakes test.py's
        # tensor addresses absolutely.
        self.LAYER0_FLASH_ATTN_P_DRAM = self.allocate_tensor_dram(prefill_aligned_seq_len * prefill_aligned_seq_len * self.bytes_per_element)
        self.LAYER0_ATTN_PROJ_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_ATTN_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_ATTN_RESIDUAL_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_MLP_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        mlp_max = max(self.mlp_elements, self.mlp_elements_wide)
        self.LAYER0_MLP_GATE_DRAM = self.allocate_tensor_dram(seq_len * mlp_max * 2)
        self.LAYER0_MLP_UP_DRAM = self.allocate_tensor_dram(seq_len * mlp_max * 2)
        self.LAYER0_MLP_MULT_DRAM = self.allocate_tensor_dram(seq_len * mlp_max * 2)
        self.LAYER0_MLP_DOWN_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_MLP_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.OUTPUT_NORM_DRAM = self.allocate_tensor_dram(1 * self.vector_length * self.bytes_per_element)
        self.LOGITS_DRAM = self.allocate_tensor_dram(1 * self.EMBEDDING_ELEMENTS * self.bytes_per_element)
        # On-FPGA repetition-penalty bias (LM-head matmul C term). MUST be
        # allocated here, immediately after LOGITS_DRAM and with identical
        # order/size to gemma4_e2b_test.py, so its baked address matches the bin.
        # All-zero == no penalty (pure greedy); _write_penalty_bias() fills it
        # when GEMMA4_PENALTY=1. See notes_repetition_penalty_fpga_bias.md.
        self.PENALTY_BIAS_DRAM = self.allocate_tensor_dram(1 * self.EMBEDDING_ELEMENTS * self.bytes_per_element)

        # Per-layer input injection buffers
        # PER_LAYER_INPUTS_DRAM: holds per_layer_inputs for all layers: MAX_CONTEXT_SIZE x 35 x 256 x 2 bytes
        self.PER_LAYER_INPUTS_DRAM = self.allocate_tensor_dram(self.MAX_CONTEXT_SIZE * self.LAYER_SIZE * self.per_layer_input_dim * self.bytes_per_element)
        # Intermediate DRAMs for per-layer injection
        self.LAYER0_PER_LAYER_GATE_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.per_layer_input_dim * self.bytes_per_element)
        self.LAYER0_PER_LAYER_PROJ_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * self.bytes_per_element)

        print(f"    Tensor DRAM usage: {self.get_tensor_dram_usage()/(1024*1024):.1f} MB")

    def program_execute(self, program_start_addr: int = user_dma_core.DRAM_INSTRUCTION_ADDR, timeout: float = 300.0, flops: float = None) -> None:
        """Execute compiled program from DRAM instruction memory."""
        self.start_execute_from_dram(program_start_addr)
        latency, flop_rate_program = 0, 0
        if timeout == 0:
            print("Program started")
        else:
            self.wait_queue(timeout)
            latency = self.report_latency_in_us()
            print(f"    Total program execution latency = {latency} us")
            if flops is not None:
                try:
                    flop_rate_program, _ = self.report_flop_rate_gflops(flops)
                except ZeroDivisionError:
                    flop_rate_program = 0.0   # transient 0-latency HW counter read → skip GFLOPS, don't abort the run
        return latency, flop_rate_program

    def _get_layer_attention_dims(self, layer_idx: int) -> tuple[int, int, int]:
        """Return (cur_head_dim, cur_q_size, cur_k_size) for a given layer index."""
        if layer_idx in self._full_attention_layers:
            cur_head_dim = self.head_dim  # 512
            cur_q_size = cur_head_dim * self.group_size  # 4096
            cur_k_size = cur_head_dim  # 512
        else:
            cur_head_dim = self.head_dim_sliding  # 256
            cur_q_size = cur_head_dim * self.group_size  # 2048
            cur_k_size = cur_head_dim  # 256
        return cur_head_dim, cur_q_size, cur_k_size

    def _get_rope_dims(self, layer_idx: int) -> int:
        """Return the number of dims to apply RoPE on for a given layer.
        Sliding layers: full rotation on head_dim_sliding=256, so N=256.
        Full attention layers: partial rotation, only first 128 dims of head_dim=512, so N=128."""
        if layer_idx in self._full_attention_layers:
            partial_rotary_factor = self._cfg["special"]["rope"]["partial_rotary_factor_global"]
            return int(self.head_dim * partial_rotary_factor)  # 128
        else:
            return self.head_dim_sliding  # 256

    def _get_mlp_elements(self, layer_idx: int) -> int:
        """Return MLP intermediate size for a given layer (wide for KV-shared layers)."""
        if layer_idx >= self._double_wide_mlp_first:
            return self.mlp_elements_wide
        return self.mlp_elements

    def _compile_per_layer_injection(self, layer_idx: int, layer_off: int, seq_len: int) -> int:
        """Compile per-layer input injection block. Returns flops added."""
        total_flops = 0
        # gate = gelu(per_layer_input_gate @ hidden_state): Linear(1536->256) + GELU
        total_flops += self.matmat_mul_core(M=seq_len, K=self.vector_length, N=self.per_layer_input_dim,
            A_DRAM_ADDR=self.LAYER0_OUTPUT_DRAM,
            B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PER_LAYER_GATE + layer_off,
            OUTPUT_DRAM_ADDR=self.LAYER0_PER_LAYER_GATE_OUTPUT_DRAM,
            gelu_enable=True)
        # gated = gate * per_layer_input[layer_idx]  (small: seq_len * 256)
        per_layer_input_addr = self.PER_LAYER_INPUTS_DRAM + layer_idx * seq_len * self.per_layer_input_dim * self.bytes_per_element
        self._emit_sram_eltwise_chunked(
            "mul",
            self.LAYER0_PER_LAYER_GATE_OUTPUT_DRAM,
            per_layer_input_addr,
            self.LAYER0_PER_LAYER_GATE_OUTPUT_DRAM,
            seq_len * self.per_layer_input_dim)
        # projected = per_layer_projection @ gated: Linear(256->1536)
        total_flops += self.matmat_mul_core(M=seq_len, K=self.per_layer_input_dim, N=self.vector_length,
            A_DRAM_ADDR=self.LAYER0_PER_LAYER_GATE_OUTPUT_DRAM,
            B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PER_LAYER_PROJ + layer_off,
            OUTPUT_DRAM_ADDR=self.LAYER0_PER_LAYER_PROJ_OUTPUT_DRAM)

        # FUSED: rms_norm(projected) + (hidden + normed) + layer_scalar
        # Previously: rms_norm_core_dram wrote the normed output to DRAM,
        # then a chunked SRAM pass read it back for the residual add. This
        # DRAM write+read pair is eliminated by running the per-row RMS norm
        # on projection data that's ALREADY in URAM_A, then immediately
        # loading the residual hidden-state chunk into URAM_B and doing the
        # add + scalar-mul before the single final writeback.
        #
        # SRAM layout (chunked by row):
        #   URAM_A [0x10000..0x80000]: projection chunk in-place (M_chunk rows × N)
        #   URAM_B [0x80000..0x80000+N*bpe]: gamma (loaded once outside the loop)
        #   URAM_B [0x80000+N*bpe..]: hidden-state chunk (M_chunk rows × N)
        bpe = self.bytes_per_element
        N = self.vector_length  # 1536
        gamma_addr = self.DRAM_ADDR_LAYER0_POST_PER_LAYER_NORM_GAMMA + layer_off
        gamma_sram = 0x80000
        hidden_sram_base = 0x80000 + N * bpe  # leave room for gamma
        proj_sram_base = 0x10000
        # Max rows per chunk is bounded by whichever URAM region is smaller
        # after reserving Q (0x00000-0x10000), gamma, and accounting for
        # two-row-aligned layout.
        uram_a_free_elements = (0x80000 - proj_sram_base) // bpe   # 229376
        uram_b_free_elements = (0x100000 - hidden_sram_base) // bpe  # 259072
        max_rows_per_chunk = min(uram_a_free_elements, uram_b_free_elements) // N
        if max_rows_per_chunk < 1:
            max_rows_per_chunk = 1
        # Upload gamma once for all chunks
        self.accelerator_memory_to_sram(
            accelerator_dram_address=gamma_addr,
            sram_address=gamma_sram, element_size=N)
        for m_off in range(0, seq_len, max_rows_per_chunk):
            m_take = min(max_rows_per_chunk, seq_len - m_off)
            chunk_bytes_offset = m_off * N * bpe
            # Load projection chunk into URAM_A
            self.accelerator_memory_to_sram(
                accelerator_dram_address=self.LAYER0_PER_LAYER_PROJ_OUTPUT_DRAM + chunk_bytes_offset,
                sram_address=proj_sram_base, element_size=m_take * N)
            # Per-row RMS norm in-place with the URAM_B-resident gamma
            for row in range(m_take):
                row_sram = proj_sram_base + row * N * bpe
                self.rms_norm_core(row_sram, row_sram, N, gamma_sram)
            # Load hidden-state chunk into URAM_B (past gamma)
            self.accelerator_memory_to_sram(
                accelerator_dram_address=self.LAYER0_OUTPUT_DRAM + chunk_bytes_offset,
                sram_address=hidden_sram_base, element_size=m_take * N)
            # Residual add: normed (URAM_A) + hidden (URAM_B) → URAM_A
            self.eltwise_add_core(
                vector_A_sram_start_addr=proj_sram_base,
                vector_B_sram_start_addr=hidden_sram_base,
                vector_C_sram_wb_addr=proj_sram_base,
                element_size=m_take * N)
            # Multiply by layer_scalar in-place
            self.broadcast_mul(
                scalar=self._layer_scalars[layer_idx],
                sram_start_addr=proj_sram_base,
                sram_wb_addr=proj_sram_base,
                element_size=m_take * N)
            # Single writeback per chunk
            self.sram_to_accelerator_memory(
                sram_address=proj_sram_base,
                accelerator_dram_address=self.LAYER0_OUTPUT_DRAM + chunk_bytes_offset,
                element_size=m_take * N)
        return total_flops

    def _emit_gqa_duplicate_pbi(self, src_dram_base: int, dst_dram_base: int,
                                cur_head_dim: int, template_seq_len: int,
                                gpr_seq_len: int, sram_addr: int = 0x10000,
                                src_row_bytes: int = None) -> None:
        """§4.4 PBI hardware loop replacing the static per-token GQA replication.
        Per token (runtime trip count = gpr_seq_len) read its one cur_head_dim row
        from the KV cache (src stride src_row_bytes, default one head; passes
        k_size here) into a fixed SRAM slot, then scatter group_size contiguous
        copies into the flat FLASH buffer via a pbi write-pointer. SRAM-safe.
        Output layout: FLASH row (i*group_size + g). Mirrors gemma4_e2b_test.py."""
        bpe = self.bytes_per_element
        row_bytes = cur_head_dim * bpe
        src_stride = src_row_bytes if src_row_bytes is not None else row_bytes
        _, sram_words = self.sram_address_to_uram_address(sram_addr)
        ptr = self.alloc_inst_ptr()
        self.generate_instruction_pbi_init(
            dram_shared_addr=dst_dram_base, dma_length=row_bytes,
            output_size=0, uram_length=0,
            uram_a_start_addr=sram_words, uram_b_start_addr=sram_words,
            uram_wb_addr=0, uram_dst_addr=0, fmax_context_addr=0,
            inst_pointer_idx=ptr)
        t_reg = self.alloc_isa_reg()
        self.generate_instruction_add_set(t_reg, 0)
        self.loop_start(loop_cnt=template_seq_len, gpr_loop_cnt=gpr_seq_len)
        self.generate_instruction_reg_mul_imm(
            self.TMP_REG, t_reg, ue_35bit_addr_shifter(src_stride))
        self.generate_instruction_add_imm(
            self.TMP_REG, ue_35bit_addr_shifter(src_dram_base), self.TMP_REG)
        self.accelerator_memory_to_sram(
            accelerator_dram_address=0, sram_address=sram_addr,
            element_size=cur_head_dim, general_reg_src=self.TMP_REG)
        self.loop_start(self.group_size)
        self.sram_to_accelerator_memory(
            sram_address=0, accelerator_dram_address=row_bytes,
            element_size=cur_head_dim, inst_pointer_idx=ptr,
            memcpy_length_bytes=0)
        self.loop_end()
        self.generate_instruction_add_inc(t_reg)
        self.loop_end()
        self.release_isa_reg()       # t_reg
        self.release_inst_ptr(ptr)

    def _emit_strided_copy_pbi(self, src_base: int, dst_base: int, copy_elems: int,
                               src_row_bytes: int, dst_row_bytes: int,
                               n_template: int, gpr_loop: int,
                               sram_addr: int = 0x10000) -> None:
        """§4.4 PBI hardware loop: for n_template rows (runtime trip count =
        gpr_loop) copy copy_elems bf16 from src to dst, advancing src by
        src_row_bytes and dst by dst_row_bytes each iteration (register-computed).
        Mirrors gemma4_e2b_test.py."""
        i_reg = self.alloc_isa_reg()
        self.generate_instruction_add_set(i_reg, 0)
        self.loop_start(loop_cnt=n_template, gpr_loop_cnt=gpr_loop)
        self.generate_instruction_reg_mul_imm(
            self.TMP_REG, i_reg, ue_35bit_addr_shifter(src_row_bytes))
        self.generate_instruction_add_imm(
            self.TMP_REG, ue_35bit_addr_shifter(src_base), self.TMP_REG)
        self.accelerator_memory_to_sram(
            accelerator_dram_address=0, sram_address=sram_addr,
            element_size=copy_elems, general_reg_src=self.TMP_REG)
        self.generate_instruction_reg_mul_imm(
            self.TMP_REG, i_reg, ue_35bit_addr_shifter(dst_row_bytes))
        self.generate_instruction_add_imm(
            self.TMP_REG, ue_35bit_addr_shifter(dst_base), self.TMP_REG)
        self.sram_to_accelerator_memory(
            sram_address=sram_addr, accelerator_dram_address=0,
            element_size=copy_elems, general_reg_src=self.TMP_REG)
        self.generate_instruction_add_inc(i_reg)
        self.loop_end()
        self.release_isa_reg()       # i_reg

    def compile_prefill(self, seq_len: int, layer_size: int = 35) -> tuple[None, int]:
        """Emit dynamic-PBI prefill into the currently open capture buffer.

        DYNAMIC PBI: seq_len-agnostic program. matmul / rms_norm / eltwise
        use gpr_M_reg=self.gpr_seq_len; flash_attention_core dispatches via
        gpr_bucket_idx + ATTN_P_DRAM. Template iteration counts come from
        prefill_max_seq_len so the captured program is portable across all
        actual seq_lens ≤ prefill_max.

        ``seq_len`` arg is FLOPs/template only — host preamble primes
        gpr_seq_len / gpr_q_seq_len / gpr_bucket_idx with runtime values.
        """
        template_seq_len = int(self._cfg["model"].get(
            "prefill_max_seq_len",
            self._cfg["model"].get("max_prefill_seq_len",
                                    self._cfg["model"]["max_context_size"])))
        seq_len = template_seq_len
        self.seq_len = seq_len
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        seq_len_engine0 = seq_len  # dual-engine path retired under dynamic PBI

        flash_num_buckets = (template_seq_len * self.group_size + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
        global _SILENT_MODE
        _SILENT_MODE = True
        total_flops = 0
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        _original_print(f"  Emitting ISA for prefill: {layer_size} layers, seq_len={seq_len}")
        prefill_t0 = time.perf_counter()

        # Shared-subroutine attention (shared_design_notes Trick 5) — mirror of
        # gemma4_e2b_test.py. Two bodies after HALT (full head_dim=512 + FULL
        # bias, sliding head_dim=256 + SLIDING bias); every call site jumps in
        # via flash_ret_id and the body returns via JUMP_REG_ABS. Must stay
        # byte-identical to test.py so the bin layout matches.
        prefill_program_dram_base = self.get_program_dram_addr()
        flash_ret_id = self.alloc_isa_reg()
        full_call_sites: list[int] = []
        sliding_call_sites: list[int] = []
        for layer_idx in range(layer_size):
            if layer_idx > 0 and layer_idx % 10 == 0:
                _original_print(f"    prefill layer {layer_idx}/{layer_size} ({time.perf_counter()-prefill_t0:.1f}s)")
            layer_off = layer_idx * LAYER_WEIGHT_SIZE
            cur_head_dim, cur_q_size, cur_k_size = self._get_layer_attention_dims(layer_idx)
            cur_mlp = self._get_mlp_elements(layer_idx)
            rope_n = self._get_rope_dims(layer_idx)

            if layer_idx != 0 and not self.engine_slave:
                self._emit_sram_copy_chunked(
                    self.LAYER0_OUTPUT_DRAM, self.LAYER0_INPUT_DRAM,
                    seq_len * self.vector_length)
            if not self.engine_slave:
                total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                                    OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off,
                                                    gpr_M_reg=self.gpr_seq_len)
            if not self.engine_slave:
                # Master: set -> clear -> workload -> check | set -> clear -> workload -> check |
                self.generate_instruction_flag_set()
                self.generate_instruction_flag_clear()
                # Q projection: N = cur_q_size (actual per-layer Q output dim)
                total_flops += self.matmat_mul_core(M=seq_len_engine0, K=self.vector_length, N=cur_q_size,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off,
                    gpr_M_reg=self.gpr_seq_len,
                    )
                if layer_idx not in self._kv_shared_map:
                    # Non-shared layer: compute K/V projections normally.
                    # Shared layers skip entirely — their attention reads K/V directly
                    # from the reference layer's slot via _kv_slot_for_layer.
                    # K projection: N = cur_k_size (actual per-layer K output dim)
                    total_flops += self.matmat_mul_core(M=seq_len_engine0, K=self.vector_length, N=cur_k_size,
                        A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                        B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off,
                        OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                        is_B_quantized=True,
                        data_type=TYPE.IF4,
                        SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off,
                        gpr_M_reg=self.gpr_seq_len,
                        )
                    # V projection: write to temp buffer first, then scatter to KV cache at k_size stride
                    total_flops += self.matmat_mul_core(M=seq_len_engine0, K=self.vector_length, N=cur_k_size,
                        A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                        B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off,
                        OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,  # temp buffer
                        is_B_quantized=True,
                        data_type=TYPE.IF4,
                        SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off,
                        gpr_M_reg=self.gpr_seq_len,
                        )
                    # V norm + scatter to KV cache at k_size stride — §4.4 PBI loop
                    # (mirrors gemma4_e2b_test.py).
                    v_cache_base = self.LAYER0_V_DRAM + self._kv_slot_for_layer[layer_idx] * self.MAX_CONTEXT_SIZE * self.k_size
                    _vi = self.alloc_isa_reg()
                    self.generate_instruction_add_set(_vi, 0)
                    self.loop_start(loop_cnt=seq_len_engine0, gpr_loop_cnt=self.gpr_seq_len)
                    self.generate_instruction_reg_mul_imm(self.TMP_REG, _vi, ue_35bit_addr_shifter(cur_k_size * self.bytes_per_element))
                    self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(self.LAYER0_FLASH_V_DRAM), self.TMP_REG)
                    self.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=0x10000, element_size=cur_k_size, general_reg_src=self.TMP_REG)
                    self.rms_norm_core(0x10000, 0x10000, cur_k_size)  # no gamma
                    self.generate_instruction_reg_mul_imm(self.TMP_REG, _vi, ue_35bit_addr_shifter(self.k_size))
                    self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(v_cache_base), self.TMP_REG)
                    self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=0, element_size=cur_k_size, general_reg_src=self.TMP_REG)
                    self.generate_instruction_add_inc(_vi)
                    self.loop_end()
                    self.release_isa_reg()  # _vi
                if self.dual_engine:
                    self.generate_instruction_flag_check(target_engine_idx=1)
            else:
                # Slave(run first): check -> clear -> workload -> set | check -> clear -> workload -> set |
                self.generate_instruction_flag_check(target_engine_idx=0)
                self.generate_instruction_flag_clear()
                total_flops += self.matmat_mul_core(M=seq_len_engine1, K=self.vector_length, N=cur_q_size,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM + seq_len_engine0 * self.vector_length * self.bytes_per_element,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM + seq_len_engine0 * cur_q_size * self.bytes_per_element,
                    is_B_quantized=True,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off,
                    gpr_M_reg=self.gpr_seq_len,
                    )
                total_flops += self.matmat_mul_core(M=seq_len_engine1, K=self.vector_length, N=cur_k_size,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM + seq_len_engine0 * self.vector_length * self.bytes_per_element,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM + seq_len_engine0 * cur_k_size * self.bytes_per_element,
                    is_B_quantized=True,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off,
                    gpr_M_reg=self.gpr_seq_len,
                    )
                total_flops += self.matmat_mul_core(M=seq_len_engine1, K=self.vector_length, N=cur_k_size,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM + seq_len_engine0 * self.vector_length * self.bytes_per_element,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_V_DRAM + self._kv_slot_for_layer[layer_idx] * self.MAX_CONTEXT_SIZE * self.k_size + seq_len_engine0 * cur_k_size * self.bytes_per_element,
                    is_B_quantized=True,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off,
                    gpr_M_reg=self.gpr_seq_len,
                    )
                self.generate_instruction_flag_set()
            if not self.engine_slave:
                # Q norm always needed (Q is always computed fresh)
                total_flops += self.rms_norm_core_dram(M=seq_len * self.group_size, N=cur_head_dim, A_DRAM_ADDR=self.LAYER0_Q_DRAM,
                                OUTPUT_DRAM_ADDR=self.LAYER0_Q_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_NORM_GAMMA + layer_off,
                                                gpr_M_reg=self.gpr_q_seq_len)

                ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE_GLOBAL if layer_idx in self._rope_global_layers else self.DRAM_ADDR_ROPE_LOCAL

                # §4.4 PBI-loop RoPE + GQA replication (mirrors gemma4_e2b_test.py).
                # K rope built CONTIGUOUS in MLP scratch then spread to the k_size-
                # strided KV cache; Q → cur_head_dim-contiguous FLASH_Q. Sliding
                # layers full-rotary (one rope call); global layers partial-rotary
                # (gather→rope→scatter→copy).
                bpe = self.bytes_per_element
                head_bytes = cur_head_dim * bpe
                rope_bytes = rope_n * bpe
                sin_addr = ROPE_WEIGHT_ADDR + rope_n * bpe
                q_rows = seq_len * self.group_size
                kv_slot_off = self._kv_slot_for_layer[layer_idx] * self.MAX_CONTEXT_SIZE * self.k_size
                k_cache_base = self.LAYER0_K_ROPE_DRAM + kv_slot_off
                v_cache_base = self.LAYER0_V_DRAM + kv_slot_off
                K_TMP = self.LAYER0_MLP_MULT_DRAM
                tmp_in = self.LAYER0_MLP_GATE_DRAM
                tmp_out = self.LAYER0_MLP_UP_DRAM
                non_shared = layer_idx not in self._kv_shared_map
                if non_shared:
                    total_flops += self.rms_norm_core_dram(M=seq_len, N=cur_head_dim, A_DRAM_ADDR=self.LAYER0_K_DRAM,
                                    OUTPUT_DRAM_ADDR=self.LAYER0_K_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_NORM_GAMMA + layer_off,
                                    gpr_M_reg=self.gpr_seq_len)
                if rope_n == cur_head_dim:
                    if non_shared:
                        total_flops += self.rope_hf_core_dram(M=seq_len, N=rope_n,
                            input_dram_addr=self.LAYER0_K_NORM_DRAM, output_dram_addr=K_TMP,
                            cos_dram_addr=ROPE_WEIGHT_ADDR, sin_dram_addr=sin_addr, gpr_M_reg=self.gpr_seq_len)
                    total_flops += self.rope_hf_core_dram_gqa(M=seq_len, group_size=self.group_size, N=rope_n,
                        input_dram_addr=self.LAYER0_Q_NORM_DRAM, output_dram_addr=self.LAYER0_FLASH_Q_DRAM,
                        cos_dram_addr=ROPE_WEIGHT_ADDR, sin_dram_addr=sin_addr, gpr_M_reg=self.gpr_seq_len)
                else:
                    non_rot = cur_head_dim - rope_n
                    if non_shared:
                        self._emit_strided_copy_pbi(self.LAYER0_K_NORM_DRAM, tmp_in, rope_n, head_bytes, rope_bytes, seq_len, self.gpr_seq_len)
                        total_flops += self.rope_hf_core_dram(M=seq_len, N=rope_n, input_dram_addr=tmp_in, output_dram_addr=tmp_out, cos_dram_addr=ROPE_WEIGHT_ADDR, sin_dram_addr=sin_addr, gpr_M_reg=self.gpr_seq_len)
                        self._emit_strided_copy_pbi(tmp_out, K_TMP, rope_n, rope_bytes, head_bytes, seq_len, self.gpr_seq_len)
                        self._emit_strided_copy_pbi(self.LAYER0_K_NORM_DRAM + rope_bytes, K_TMP + rope_bytes, non_rot, head_bytes, head_bytes, seq_len, self.gpr_seq_len)
                    self._emit_strided_copy_pbi(self.LAYER0_Q_NORM_DRAM, tmp_in, rope_n, head_bytes, rope_bytes, q_rows, self.gpr_q_seq_len)
                    total_flops += self.rope_hf_core_dram_gqa(M=seq_len, group_size=self.group_size, N=rope_n, input_dram_addr=tmp_in, output_dram_addr=tmp_out, cos_dram_addr=ROPE_WEIGHT_ADDR, sin_dram_addr=sin_addr, gpr_M_reg=self.gpr_seq_len)
                    self._emit_strided_copy_pbi(tmp_out, self.LAYER0_FLASH_Q_DRAM, rope_n, rope_bytes, head_bytes, q_rows, self.gpr_q_seq_len)
                    self._emit_strided_copy_pbi(self.LAYER0_Q_NORM_DRAM + rope_bytes, self.LAYER0_FLASH_Q_DRAM + rope_bytes, non_rot, head_bytes, head_bytes, q_rows, self.gpr_q_seq_len)
                if non_shared:
                    self._emit_strided_copy_pbi(K_TMP, k_cache_base, cur_head_dim, head_bytes, self.k_size, seq_len, self.gpr_seq_len)
                self._emit_gqa_duplicate_pbi(k_cache_base, self.LAYER0_FLASH_K_DRAM, cur_head_dim, seq_len, self.gpr_seq_len, src_row_bytes=self.k_size)
                self._emit_gqa_duplicate_pbi(v_cache_base, self.LAYER0_FLASH_V_DRAM, cur_head_dim, seq_len, self.gpr_seq_len, src_row_bytes=self.k_size)

                # Gemma4 uses scaling=1.0 (no 1/sqrt(d) in attention scores).
                # flash_attention_core internally applies 1/sqrt(head_dim), so pre-scale Q by sqrt(head_dim) to cancel it.
                self._emit_sram_broadcast_mul_chunked(
                    self.LAYER0_FLASH_Q_DRAM, self.LAYER0_FLASH_Q_DRAM,
                    aligned_seq_len * cur_head_dim, math.sqrt(cur_head_dim))

                # Pick the per-layer bias: full attention layers see the
                # entire causal window; sliding-attention layers are limited
                # to `sliding_window` tokens (run_prefill builds both biases).
                bias_addr_layer = (self.LAYER0_FLASH_BIAS_FULL_DRAM
                                   if layer_idx in self._full_attention_layers
                                   else self.LAYER0_FLASH_BIAS_SLIDING_DRAM)
                # NOTE: flash_attention stays in LEGACY () regardless
                # of _PBI. The PBI flash_attention back-to-back bug
                # (see memory: fpga_pbi_flash_back_to_back_bug) degrades every
                # call after the first within the same program execution to
                # cos≈0.94, and prefill emits 35 of these in one program → all
                # K/V cache entries past layer 0 are wrong → first decode token
                # becomes garbage / stop token. matmul + rms_norm PBI are safe
                # back-to-back, so we keep those on.
                # Shared-subroutine attention: set return address + jump to the
                # matching subroutine (full vs sliding) compiled once after HALT.
                # Prefill is marshal-free (Q/K/V/OUTPUT already in fixed buffers).
                self.pad_capture_to_64b_boundary()
                _ret_word_addr = ue_35bit_addr_shifter(
                    prefill_program_dram_base + (self.capture_count + 2) * user_dma_core.INSTRUCTION_SIZE_BYTES)
                self.generate_instruction_add_set(flash_ret_id, _ret_word_addr)
                (full_call_sites if layer_idx in self._full_attention_layers
                 else sliding_call_sites).append(self.capture_count)
                self.generate_instruction_jump_abs(target_instruction_word_addr=0)
            if not self.engine_slave:
                # Master: set -> clear -> workload -> check | set -> clear -> workload -> check |
                self.generate_instruction_flag_set()
                self.generate_instruction_flag_clear()
                # O projection: INT4, K=cur_q_size
                total_flops += self.matmat_mul_core(M=seq_len_engine0, K=cur_q_size, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off,
                    gpr_M_reg=self.gpr_seq_len,
                    )
                if self.dual_engine:
                    self.generate_instruction_flag_check(target_engine_idx=1)
            else:
                # Slave(run first): check -> clear -> workload -> set | check -> clear -> workload -> set |
                self.generate_instruction_flag_check(target_engine_idx=0)
                self.generate_instruction_flag_clear()
                total_flops += self.matmat_mul_core(M=seq_len_engine1, K=cur_q_size, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM + seq_len_engine0 * cur_q_size * self.bytes_per_element,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM + seq_len_engine0 * self.vector_length * self.bytes_per_element,
                    is_B_quantized=True,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off,
                    gpr_M_reg=self.gpr_seq_len,
                    )
                self.generate_instruction_flag_set()
            if not self.engine_slave:
                total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                                OUTPUT_DRAM_ADDR=self.LAYER0_POST_ATTN_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_NORM_GAMMA + layer_off,
                                                gpr_M_reg=self.gpr_seq_len)
                self._emit_sram_eltwise_chunked(
                    "add", self.LAYER0_INPUT_DRAM, self.LAYER0_POST_ATTN_NORM_DRAM,
                    self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                    seq_len * self.vector_length)
                total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                                OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off,
                                                gpr_M_reg=self.gpr_seq_len)
            if not self.engine_slave:
                # Master: set -> clear -> workload -> check | set -> clear -> workload -> check |
                self.generate_instruction_flag_set()
                self.generate_instruction_flag_clear()
                total_flops += self.matmat_mul_core(M=seq_len_engine0, K=self.vector_length, N=cur_mlp,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off,
                    gelu_enable=True,
                    gpr_M_reg=self.gpr_seq_len,
                    )
                total_flops += self.matmat_mul_core(M=seq_len_engine0, K=self.vector_length, N=cur_mlp,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off,
                    gpr_M_reg=self.gpr_seq_len,
                    )
                if self.dual_engine:
                    self.generate_instruction_flag_check(target_engine_idx=1)
            else:
                # Slave: check -> clear -> workload -> set | check -> clear -> workload -> set |
                self.generate_instruction_flag_check(target_engine_idx=0)
                self.generate_instruction_flag_clear()
                total_flops += self.matmat_mul_core(M=seq_len_engine1, K=self.vector_length, N=cur_mlp,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM + seq_len_engine0 * self.vector_length * self.bytes_per_element,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM + seq_len_engine0 * cur_mlp * self.bytes_per_element,
                    is_B_quantized=True,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off,
                    gelu_enable=True,
                    gpr_M_reg=self.gpr_seq_len,
                    )
                total_flops += self.matmat_mul_core(M=seq_len_engine1, K=self.vector_length, N=cur_mlp,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM + seq_len_engine0 * self.vector_length * self.bytes_per_element,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM + seq_len_engine0 * cur_mlp * self.bytes_per_element,
                    is_B_quantized=True,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off,
                    gpr_M_reg=self.gpr_seq_len,
                    )
                self.generate_instruction_flag_set()
            if not self.engine_slave:
                self._emit_sram_eltwise_chunked(
                    "mul", self.LAYER0_MLP_GATE_DRAM, self.LAYER0_MLP_UP_DRAM,
                    self.LAYER0_MLP_MULT_DRAM,
                    seq_len * cur_mlp)
            if not self.engine_slave:
                # Master: set -> clear -> workload -> check | set -> clear -> workload -> check |
                self.generate_instruction_flag_set()
                self.generate_instruction_flag_clear()
                total_flops += self.matmat_mul_core(M=seq_len_engine0, K=cur_mlp, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off,
                    gpr_M_reg=self.gpr_seq_len,
                    )
                if self.dual_engine:
                    self.generate_instruction_flag_check(target_engine_idx=1)
            else:
                # Slave: check -> clear -> workload -> set | check -> clear -> workload -> set |
                self.generate_instruction_flag_check(target_engine_idx=0)
                self.generate_instruction_flag_clear()
                total_flops += self.matmat_mul_core(M=seq_len_engine1, K=cur_mlp, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM + seq_len_engine0 * cur_mlp * self.bytes_per_element,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM + seq_len_engine0 * self.vector_length * self.bytes_per_element,
                    is_B_quantized=True,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off,
                    gpr_M_reg=self.gpr_seq_len,
                    )
                self.generate_instruction_flag_set()
            if not self.engine_slave:
                total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                                OUTPUT_DRAM_ADDR=self.LAYER0_POST_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_FFW_NORM_GAMMA + layer_off,
                                                gpr_M_reg=self.gpr_seq_len)
                self._emit_sram_eltwise_chunked(
                    "add", self.LAYER0_POST_ATTN_RESIDUAL_DRAM, self.LAYER0_POST_MLP_NORM_DRAM,
                    self.LAYER0_OUTPUT_DRAM,
                    seq_len * self.vector_length)

                # Per-layer input injection (NEW for Gemma4 E2B)
                total_flops += self._compile_per_layer_injection(layer_idx, layer_off, seq_len)

        # Terminate the normal path with HALT; the flash subroutines follow and
        # are reachable only via the JUMP_ABS call sites above (mirror of test.py).
        self.generate_instruction_halt()

        def _emit_flash_subroutine(_hd, _bias_addr):
            self.pad_capture_to_64b_boundary()
            return self.flash_attention_core(
                head_dim=_hd,
                seq_len=aligned_seq_len,
                Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
                K_DRAM_ADDR=self.LAYER0_FLASH_K_DRAM,
                V_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                BIAS_DRAM_ADDR=_bias_addr,
                ATTN_P_DRAM_ADDR=self.LAYER0_FLASH_ATTN_P_DRAM,
                gpr_bucket_idx=self.gpr_bucket_idx,
                num_buckets=flash_num_buckets,
                IDENTITY_DRAM_ADDR=self.IDENTITY_DRAM_ADDR,
                gpr_ret_id=flash_ret_id,
            )

        if full_call_sites:
            _full_start, _full_flops = _emit_flash_subroutine(
                self.head_dim, self.LAYER0_FLASH_BIAS_FULL_DRAM)
            for _idx in full_call_sites:
                self._patch_jump_immediate(_idx, ue_35bit_addr_shifter(_full_start))
            total_flops += int(_full_flops[-1] if isinstance(_full_flops, (list, tuple)) else _full_flops) * len(full_call_sites)
        if sliding_call_sites:
            _sl_start, _sl_flops = _emit_flash_subroutine(
                self.head_dim_sliding, self.LAYER0_FLASH_BIAS_SLIDING_DRAM)
            for _idx in sliding_call_sites:
                self._patch_jump_immediate(_idx, ue_35bit_addr_shifter(_sl_start))
            total_flops += int(_sl_flops[-1] if isinstance(_sl_flops, (list, tuple)) else _sl_flops) * len(sliding_call_sites)

        self.release_isa_reg()  # flash_ret_id
        _SILENT_MODE = False
        return None, total_flops

    def _compute_per_layer_inputs(self, token_ids, embedding_tensor: torch.Tensor) -> torch.Tensor:
        """Compute per-layer inputs on host side.
        Args:
            token_ids: token id sequence (list or tuple)
            embedding_tensor: (seq_len, hidden_size) bf16 tensor (already scaled)
        Returns:
            per_layer_inputs: (seq_len, LAYER_SIZE, per_layer_input_dim) bf16 tensor
        """
        seq_len = len(token_ids)
        tid_t = torch.tensor(token_ids, dtype=torch.long)

        # VLM mode: replace image token IDs with pad_token_id for per_layer_embed lookup
        if hasattr(self, '_mm_types') and self._mm_types is not None:
            mm_mask = torch.tensor(self._mm_types[:len(token_ids)])
            tid_t_for_pli = tid_t.clone()
            tid_t_for_pli[mm_mask == 1] = 0  # pad_token_id
        else:
            tid_t_for_pli = tid_t

        # per_layer_embed: lookup from embed_tokens_per_layer [262144, 8960] -> [seq_len, 8960] -> [seq_len, 35, 256]
        per_layer_embed = self.embed_tokens_per_layer_weight[tid_t_for_pli]  # [seq_len, 8960]
        per_layer_embed = per_layer_embed.reshape(seq_len, self.LAYER_SIZE, self.per_layer_input_dim)  # [seq_len, 35, 256]

        # per_layer_proj: (per_layer_model_projection @ embedding.T).T
        # per_layer_model_proj_weight is [8960, 1536], embedding_tensor is [seq_len, 1536]
        # We want [seq_len, 8960] = embedding_tensor @ per_layer_model_proj_weight.T
        # But need to undo the embedding scale first: use the unscaled embedding
        # Actually the spec says to use the embedding_tensor (already scaled), and apply proj_scale
        per_layer_proj = (embedding_tensor.float() @ self.per_layer_model_proj_weight.float().T)  # [seq_len, 8960]
        per_layer_proj = (per_layer_proj * self._per_layer_model_proj_scale).to(torch.bfloat16)
        per_layer_proj = per_layer_proj.reshape(seq_len, self.LAYER_SIZE, self.per_layer_input_dim)  # [seq_len, 35, 256]

        # rms_norm per_layer_proj along last dim with per_layer_proj_norm_weight
        per_layer_proj = _host_rms_norm(per_layer_proj, self.per_layer_proj_norm_weight)

        # per_layer_inputs = (per_layer_proj + per_layer_embed) * per_layer_input_scale
        per_layer_inputs = ((per_layer_proj.float() + per_layer_embed.float()) * self._per_layer_input_scale).to(torch.bfloat16)

        return per_layer_inputs  # [seq_len, 35, 256]

    def run_prefill(self, prefill_program_addr: int, prefill_seq=None, flops: int = None) -> dict:
        """
        Run prefill. Uses prefill_seq if provided, otherwise self.prefill_seq (set by set_prefill_seq()).

        Args:
            prefill_program_addr: The address of the prefill program in DRAM.
            prefill_seq: Optional sequence; if None, uses self.prefill_seq.
            flops: The number of FLOPS to use for the prefill.

        Returns:
            A tuple containing the latency and flop rate.
        """
        if prefill_seq is None:
            prefill_seq = self.prefill_seq
        if prefill_seq is None:
            prefill_seq = tuple(self._cfg["default_prefill_tokens"])
        # Prefill processes all but the last token
        if len(prefill_seq) > 1:
            prefill_seq = prefill_seq[:-1]
            assert len(prefill_seq) == self.seq_len, f"Expected seq_len {self.seq_len}, but got {len(prefill_seq)}"
        else:
            raise ValueError("Prefill sequence must have at least 2 tokens.")

        seq_len = len(prefill_seq)
        assert seq_len <= self.max_prefill_seq_len, (
            f"Prefill length {seq_len} exceeds max_prefill_seq_len {self.max_prefill_seq_len}. "
            f"Bump 'max_prefill_seq_len' in gemma4_e2b_config.json (and raise _tensor_estimate "
            f"in __init__ accordingly) to support longer prompts."
        )
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64

        print(f"[Prefill] [host] looking up token embeddings for {seq_len} tokens...", flush=True)
        embedding_tensor = self.get_embedding_for_tokens(prefill_seq)

        # Multimodal merge: replace image/audio placeholder embeddings with
        # encoder-produced soft-token features. Uses mm_token_type_ids where
        # 1=image, 3=audio (HF convention, see transformers/processing_utils.py
        # create_mm_token_type_ids).
        if hasattr(self, '_mm_types') and self._mm_types is not None:
            mm_types = torch.tensor(self._mm_types[:len(prefill_seq)])
            if hasattr(self, '_image_features') and self._image_features is not None:
                image_mask = (mm_types == 1)
                embedding_tensor[image_mask] = self._image_features[:image_mask.sum()].to(embedding_tensor.dtype)
                print(f"[Prefill] merged {image_mask.sum().item()} image features into embeddings")
            if hasattr(self, '_audio_features') and self._audio_features is not None:
                audio_mask = (mm_types == 3)
                embedding_tensor[audio_mask] = self._audio_features[:audio_mask.sum()].to(embedding_tensor.dtype)
                print(f"[Prefill] merged {audio_mask.sum().item()} audio features into embeddings")

        print(f"[Prefill] uploading embeddings to FPGA DRAM...", flush=True)
        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)

        # Compute per-layer inputs on host and DMA to FPGA
        print(f"[Prefill] [host] computing per-layer inputs ({seq_len} tokens x {self.LAYER_SIZE} layers)...", flush=True)
        per_layer_inputs = self._compute_per_layer_inputs(prefill_seq, embedding_tensor)  # [seq_len, 35, 256]
        # Permute to [35, seq_len, 256] so each layer's data is contiguous in DRAM
        per_layer_inputs_flat = per_layer_inputs.permute(1, 0, 2).contiguous()  # [35, seq_len, 256]
        print(f"[Prefill] uploading per-layer inputs to FPGA DRAM...", flush=True)
        self.dma_to_accelerator_memory(self.PER_LAYER_INPUTS_DRAM, per_layer_inputs_flat)

        # Clear multimodal state now that prefill's per-layer-inputs have been computed.
        # Decode reuses _compute_per_layer_inputs with seq_len=1, and if _mm_types
        # is still set it would incorrectly treat every decode token as a
        # multimodal position (replacing its ID with pad_token_id=0 for
        # per_layer_embed lookup),
        # producing garbage per-layer injection and all-pad output. Mirror the
        # compare script's pattern: clear right after use.
        self._mm_types = None
        self._image_features = None
        self._audio_features = None

        # Build BOTH prefill bias matrices: full (causal) for full-attention
        # layers, and sliding (causal AND within `sliding_window` tokens) for
        # sliding-attention layers. compile_prefill picks per-layer.
        # Both biases are in q_seq_len space (each token has group_size query
        # heads, K is GQA-duplicated to match), so the window is converted
        # from token space to q-position space by multiplying by group_size.
        full_bias = torch.full((aligned_seq_len, aligned_seq_len), float("-inf"), dtype=torch.bfloat16)
        valid_mask = torch.tril(torch.ones(aligned_seq_len, aligned_seq_len, dtype=torch.bool), diagonal=0) if not self.causal_mask_upper else torch.triu(torch.ones(aligned_seq_len, aligned_seq_len, dtype=torch.bool), diagonal=0)
        full_bias.masked_fill_(valid_mask, 0.0)
        full_bias[:, q_seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_FULL_DRAM, full_bias)

        # Sliding bias is identical to full when seq_len ≤ sliding_window;
        # otherwise it additionally masks anything older than the window.
        if seq_len <= self.sliding_window:
            sliding_bias = full_bias
        else:
            window_q = self.sliding_window * self.group_size
            sliding_bias = torch.full((aligned_seq_len, aligned_seq_len), float("-inf"), dtype=torch.bfloat16)
            i_idx = torch.arange(aligned_seq_len).unsqueeze(1)
            j_idx = torch.arange(aligned_seq_len).unsqueeze(0)
            i_token = i_idx // self.group_size
            j_token = j_idx // self.group_size
            in_window = (i_token - j_token) < self.sliding_window
            sliding_mask = valid_mask & in_window
            sliding_bias.masked_fill_(sliding_mask, 0.0)
            sliding_bias[:, q_seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_SLIDING_DRAM, sliding_bias)

        # Dynamic-PBI preamble: prime gpr_seq_len / gpr_q_seq_len / gpr_bucket_idx
        # so the captured PBI body reads the right row counts.
        bucket_idx = aligned_seq_len // UE_VECTOR_SIZE       # 1-based
        self.isa_add_set_core(self.gpr_seq_len,    seq_len)
        self.isa_add_set_core(self.gpr_q_seq_len,  q_seq_len)
        self.isa_add_set_core(self.gpr_bucket_idx, bucket_idx)

        print(f"[Prefill] [exec] launching prefill program on FPGA ({seq_len} tokens, {self.LAYER_SIZE} layers)...", flush=True)
        # Heartbeat thread: program_execute blocks until the FPGA halts, with no
        # intermediate visibility. Print elapsed seconds every 10s so the user
        # sees liveness during the ~30-60s prefill execution.
        import threading
        _pf_t0 = time.perf_counter()
        _pf_stop = threading.Event()
        def _pf_hb():
            while not _pf_stop.wait(10):
                print(f"[Prefill] [exec]   ... still running on FPGA ({time.perf_counter()-_pf_t0:.0f}s elapsed)", flush=True)
        _pf_th = threading.Thread(target=_pf_hb, daemon=True)
        _pf_th.start()
        try:
            latency, flop_rate_program = self.program_execute(prefill_program_addr, flops=flops)
        finally:
            _pf_stop.set()
            _pf_th.join(timeout=1.0)
        return latency, flop_rate_program

    def compile_decoder(self, layer_size: int = 35) -> tuple[None, list[int], list[int]]:
        """Compile decoder programs for seq_len buckets, or load from existing bin/meta.
        Returns (decoder_bin_path, program_sizes[8], total_flops_list[8])."""
        # Dynamic PBI: single decoder program with bucket dispatcher for
        # decoder_group_attention_core. Per-token KV/RoPE addresses computed
        # via reg_mul_imm(gpr_seq_len, stride) + add_imm(base) → TMP_REG.
        # Trailing add_inc(gpr_seq_len) self-advances the position counter.
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        dec_num_buckets = (self.MAX_CONTEXT_SIZE + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE

        global _SILENT_MODE
        _SILENT_MODE = True
        _original_print(f"  Emitting dynamic-PBI decoder: 1 segment x {layer_size} layers, dec_buckets={dec_num_buckets}")
        seg_t0 = time.perf_counter()
        count_at_start = self.capture_count
        total_flops = 0
        gpr_one = self.alloc_isa_reg()
        self.generate_instruction_add_set(gpr_one, 1)

        # Shared-subroutine decoder attention (shared_design_notes Trick 5), sliding-only
        # safe subset — mirror of gemma4_e2b_test.py. Full layers stay inline.
        decoder_program_dram_base = self.get_program_dram_addr()
        flash_ret_id = self.alloc_isa_reg()
        dec_sliding_call_sites: list[int] = []
        for _bi_unused in [0]:
            seq_len = self.MAX_CONTEXT_SIZE  # template / FLOPs only
            for layer_idx in range(layer_size):
                layer_off = layer_idx * LAYER_WEIGHT_SIZE
                cur_head_dim, cur_q_size, cur_k_size = self._get_layer_attention_dims(layer_idx)
                cur_mlp = self._get_mlp_elements(layer_idx)
                rope_n = self._get_rope_dims(layer_idx)

                # Layer-input source:
                #   layer 0: LAYER0_INPUT_DRAM (uploaded by run_decoder each step)
                #   layer i>0: LAYER0_OUTPUT_DRAM (written by the previous layer's
                #     per_layer_injection). No copy needed — LAYER0_OUTPUT_DRAM is
                #     only overwritten at the end of the current layer (in the
                #     MLP residual add at line ~2955), which happens AFTER we
                #     consume it as the attention-residual source. So reading it
                #     here and for the attention residual below is safe.
                layer_input_addr = self.LAYER0_INPUT_DRAM if layer_idx == 0 else self.LAYER0_OUTPUT_DRAM
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=layer_input_addr,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off,
                              gpr_M_reg=gpr_one)
                # Q/K/V projections: use per-layer dims
                total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=cur_q_size,
                                                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                                                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off,
                                                    OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                                                    data_type=TYPE.IF4,
                                                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off,
                                                    )
                if layer_idx in self._kv_shared_map:
                    ref_layer = self._kv_shared_map[layer_idx]
                    kv_layer_for_attn = ref_layer  # read from reference layer's KV cache
                else:
                    kv_layer_for_attn = layer_idx  # read from own KV cache
                    # K projection
                    total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=cur_k_size,
                        A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                        B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off,
                        OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                        data_type=TYPE.IF4,
                        SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off,
                        )
                    # V projection
                    total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=cur_k_size,
                        A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                        B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off,
                        OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                        data_type=TYPE.IF4,
                        SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off,
                        )
                    self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_FLASH_V_DRAM, sram_address=0x10000, element_size=cur_k_size)
                    # V norm (Gemma4: normalize V without learnable scale)
                    self.rms_norm_core(0x10000, 0x10000, cur_k_size)  # no gamma
                    # V scatter to V cache at decode_pos: addr = LAYER0_V_DRAM + slot * MAX_CTX * k_size + gpr_seq_len * k_size.
                    _v_slot_base = self.LAYER0_V_DRAM + self._kv_slot_for_layer[layer_idx] * self.MAX_CONTEXT_SIZE * self.k_size
                    self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gpr_seq_len, ue_35bit_addr_shifter(self.k_size))
                    self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(_v_slot_base), self.TMP_REG)
                    self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=0, element_size=cur_k_size, general_reg_src=self.TMP_REG)
                    # RMS norm on K
                    total_flops += self.rms_norm_core_dram(M=1, N=cur_head_dim, A_DRAM_ADDR=self.LAYER0_K_DRAM,
                                  OUTPUT_DRAM_ADDR=self.LAYER0_K_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_NORM_GAMMA + layer_off,
                                  gpr_M_reg=gpr_one)

                # Q norm: M=group_size (compile-time constant) — legacy static-M.
                total_flops += self.rms_norm_core_dram(M=self.group_size, N=cur_head_dim, A_DRAM_ADDR=self.LAYER0_Q_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_Q_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_NORM_GAMMA + layer_off)

                ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE_GLOBAL if layer_idx in self._rope_global_layers else self.DRAM_ADDR_ROPE_LOCAL
                rope_row = 2 * rope_n * self.bytes_per_element  # cos+sin pair stride per token

                kv_slot_off_local = self._kv_slot_for_layer[layer_idx] * self.MAX_CONTEXT_SIZE * self.k_size
                k_rope_base = self.LAYER0_K_ROPE_DRAM + kv_slot_off_local

                if layer_idx not in self._kv_shared_map:
                    # K-RoPE at decode_pos. Output to LAYER0_K_DRAM scratch (NOT
                    # to k_rope_base, which is cache position 0 — writing there
                    # would corrupt the first prefill token's K every step).
                    self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gpr_seq_len, ue_35bit_addr_shifter(rope_row))
                    self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(ROPE_WEIGHT_ADDR), self.TMP_REG)
                    total_flops += self.rope_hf_core_decode(
                        N=rope_n,
                        input_dram_addr=self.LAYER0_K_NORM_DRAM,
                        output_dram_addr=self.LAYER0_K_DRAM,
                        gr_weight_dram=self.TMP_REG)
                    self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gpr_seq_len, ue_35bit_addr_shifter(self.k_size))
                    self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(k_rope_base), self.TMP_REG)
                    self.accelerator_memcpy(self.LAYER0_K_DRAM, 0, rope_n * self.bytes_per_element, gr_dst_addr=self.TMP_REG)

                # Q-RoPE: same cos/sin for all group_size heads (same decode_pos).
                self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gpr_seq_len, ue_35bit_addr_shifter(rope_row))
                self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(ROPE_WEIGHT_ADDR), self.TMP_REG)
                for g in range(self.group_size):
                    total_flops += self.rope_hf_core_decode(
                        N=rope_n,
                        input_dram_addr=self.LAYER0_Q_NORM_DRAM + g * cur_head_dim * self.bytes_per_element,
                        output_dram_addr=self.LAYER0_FLASH_Q_DRAM + g * cur_head_dim * self.bytes_per_element,
                        gr_weight_dram=self.TMP_REG)

                # Partial-rotary non-rotated dims (full-attention layers only).
                if layer_idx in self._full_attention_layers and rope_n < cur_head_dim:
                    remaining = cur_head_dim - rope_n
                    for g in range(self.group_size):
                        src = self.LAYER0_Q_NORM_DRAM + g * cur_head_dim * self.bytes_per_element + rope_n * self.bytes_per_element
                        dst = self.LAYER0_FLASH_Q_DRAM + g * cur_head_dim * self.bytes_per_element + rope_n * self.bytes_per_element
                        self.accelerator_memory_to_sram(src, 0x10000, remaining)
                        self.sram_to_accelerator_memory(0x10000, dst, remaining)
                    if layer_idx not in self._kv_shared_map:
                        # K non-rotated dims at decode_pos: addr = k_rope_base + gpr_seq_len * k_size + rope_n_bytes
                        src = self.LAYER0_K_NORM_DRAM + rope_n * self.bytes_per_element
                        k_cache_nrot_base = k_rope_base + rope_n * self.bytes_per_element
                        self.accelerator_memory_to_sram(src, 0x10000, remaining)
                        self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gpr_seq_len, ue_35bit_addr_shifter(self.k_size))
                        self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(k_cache_nrot_base), self.TMP_REG)
                        self.sram_to_accelerator_memory(0x10000, 0, remaining, general_reg_src=self.TMP_REG)

                # Q pre-scaling: cancel decoder_group_attention_core's internal 1/sqrt(head_dim).
                self.accelerator_memory_to_sram(self.LAYER0_FLASH_Q_DRAM, 0x30000, self.group_size * cur_head_dim)
                self.broadcast_mul(scalar=math.sqrt(cur_head_dim), sram_start_addr=0x30000, sram_wb_addr=0x30000, element_size=self.group_size * cur_head_dim)
                self.sram_to_accelerator_memory(0x30000, self.LAYER0_FLASH_Q_DRAM, self.group_size * cur_head_dim)

                # K/V cache reads (KV-shared layers point at source layer's cache).
                kv_slot_off_read = self._kv_slot_for_layer[kv_layer_for_attn] * self.MAX_CONTEXT_SIZE * self.k_size
                kv_k_base = self.LAYER0_K_ROPE_DRAM + kv_slot_off_read
                kv_v_base = self.LAYER0_V_DRAM + kv_slot_off_read

                # Stride-mismatch gather for sliding layers (cur_head_dim*bpe != k_size).
                # decoder_group_attention_core reads K/V at head_dim*bpe stride, but
                # KV cache uses k_size stride per token; gather to contiguous buffer.
                if cur_head_dim * self.bytes_per_element != self.k_size:
                    for t in range(self.MAX_CONTEXT_SIZE):
                        self.accelerator_memory_to_sram(kv_k_base + t * self.k_size, 0x10000, cur_head_dim)
                        self.sram_to_accelerator_memory(0x10000, self.LAYER0_FLASH_K_DRAM + t * cur_head_dim * self.bytes_per_element, cur_head_dim)
                        self.accelerator_memory_to_sram(kv_v_base + t * self.k_size, 0x20000, cur_head_dim)
                        self.sram_to_accelerator_memory(0x20000, self.LAYER0_FLASH_V_DRAM + t * cur_head_dim * self.bytes_per_element, cur_head_dim)
                    dec_k_addr = self.LAYER0_FLASH_K_DRAM
                    dec_v_addr = self.LAYER0_FLASH_V_DRAM
                else:
                    dec_k_addr = kv_k_base
                    dec_v_addr = kv_v_base

                bias_addr_layer = (self.LAYER0_FLASH_BIAS_FULL_DRAM
                                   if layer_idx in self._full_attention_layers
                                   else self.LAYER0_FLASH_BIAS_SLIDING_DRAM)
                if layer_idx in self._full_attention_layers:
                    _dec_flops_list = self.decoder_group_attention_core(
                        group_size=self.group_size,
                        head_dim=cur_head_dim,
                        seq_len=seq_len,
                        Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
                        K_DRAM_ADDR=dec_k_addr,
                        V_DRAM_ADDR=dec_v_addr,
                        OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                        IDENTITY_DRAM_ADDR=self.IDENTITY_DRAM_ADDR,
                        SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                        BIAS_DRAM_ADDR=bias_addr_layer,
                        gpr_bucket_idx=self.gpr_bucket_idx,
                        num_buckets=dec_num_buckets)
                    total_flops += int(sum(_dec_flops_list)) if isinstance(_dec_flops_list, (list, tuple)) else int(_dec_flops_list)
                else:
                    self.pad_capture_to_64b_boundary()
                    _ret_word_addr = ue_35bit_addr_shifter(
                        decoder_program_dram_base + (self.capture_count + 2) * user_dma_core.INSTRUCTION_SIZE_BYTES)
                    self.generate_instruction_add_set(flash_ret_id, _ret_word_addr)
                    dec_sliding_call_sites.append(self.capture_count)
                    self.generate_instruction_jump_abs(target_instruction_word_addr=0)
                # O projection: INT4, K=cur_q_size (actual per-layer attention output dim)
                total_flops += self.quantized_matmat_core(M=1, K=cur_q_size, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off,
                    )
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_POST_ATTN_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_NORM_GAMMA + layer_off,
                              gpr_M_reg=gpr_one)

                # Attention residual: use layer_input_addr (LAYER0_OUTPUT_DRAM
                # for layers > 0, LAYER0_INPUT_DRAM for layer 0) — same source
                # as the pre-norm above. This avoids the LAYER0_OUTPUT → LAYER0_INPUT
                # copy that used to run at the top of every layer.
                self.accelerator_memory_to_sram(accelerator_dram_address=layer_input_addr, sram_address=0x10000, element_size=self.vector_length)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_NORM_DRAM, sram_address=0x90000, element_size=self.vector_length)
                self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, element_size=self.vector_length)

                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off,
                              gpr_M_reg=gpr_one)

                total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=cur_mlp,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off,
                    gelu_enable=True,
                    )
                total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=cur_mlp,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off,
                    )

                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_GATE_DRAM, sram_address=0x10000, element_size=cur_mlp)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_UP_DRAM, sram_address=0x90000, element_size=cur_mlp)
                self.eltwise_mul_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=cur_mlp)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_MLP_MULT_DRAM, element_size=cur_mlp)

                total_flops += self.matmat_mul_core(M=1, K=cur_mlp, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off,
                        gpr_M_reg=gpr_one,
                    )
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_POST_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_FFW_NORM_GAMMA + layer_off,
                              gpr_M_reg=gpr_one)

                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, sram_address=0x10000, element_size=self.vector_length)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_MLP_NORM_DRAM, sram_address=0x90000, element_size=self.vector_length)
                self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, element_size=self.vector_length)

                # Per-layer input injection (NEW for Gemma4 E2B) - decoder uses seq_len=1
                total_flops += self._compile_per_layer_injection(layer_idx, layer_off, 1)

            if layer_size == self.LAYER_SIZE:
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_OUTPUT_DRAM,
                    OUTPUT_DRAM_ADDR=self.OUTPUT_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_OUTPUT_NORM_GAMMA,
                    gpr_M_reg=gpr_one)
                total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=self.EMBEDDING_ELEMENTS,
                    A_DRAM_ADDR=self.OUTPUT_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD_QUANT,
                    OUTPUT_DRAM_ADDR=self.LOGITS_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD_SCALE,
                    gpr_M_reg=gpr_one,
                    )

            # Advance decode_pos for next token.
            self.generate_instruction_add_inc(self.gpr_seq_len)
            self.generate_instruction_halt()

            # Shared sliding-attention decoder subroutine (mirror of test.py).
            if dec_sliding_call_sites:
                self.pad_capture_to_64b_boundary()
                _dec_sub_start, _dec_flops = self.decoder_group_attention_core(
                    group_size=self.group_size,
                    head_dim=self.head_dim_sliding,
                    seq_len=seq_len,
                    Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
                    K_DRAM_ADDR=self.LAYER0_FLASH_K_DRAM,
                    V_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                    OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                    IDENTITY_DRAM_ADDR=self.IDENTITY_DRAM_ADDR,
                    SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                    BIAS_DRAM_ADDR=self.LAYER0_FLASH_BIAS_SLIDING_DRAM,
                    gpr_bucket_idx=self.gpr_bucket_idx,
                    num_buckets=dec_num_buckets,
                    gpr_ret_id=flash_ret_id)
                for _idx in dec_sliding_call_sites:
                    self._patch_jump_immediate(_idx, ue_35bit_addr_shifter(_dec_sub_start))
                total_flops += int(_dec_flops[-1] if isinstance(_dec_flops, (list, tuple)) else _dec_flops) * len(dec_sliding_call_sites)
            self.release_isa_reg()  # flash_ret_id
            instr_count = self.capture_count - count_at_start
            _original_print(f"    decoder segment ({instr_count} instr) done in {time.perf_counter()-seg_t0:.1f}s")
        program_sizes = [instr_count * 32]
        total_flops_list = [total_flops]
        _SILENT_MODE = False
        return None, program_sizes, total_flops_list

    # ------------------------------------------------------------------
    # Single instruction bin: prefill (all buckets) + decode (all buckets)
    # ------------------------------------------------------------------
    def compile_instruction_bin(self, layer_size: int = 35) -> tuple[str, dict]:
        """Compile prefill buckets + decode buckets into ONE capture session,
        dump to programs.bin + programs.json. Mirrors
        gemma3's single-bin pattern. Absolute addrs in the manifest are
        baked against the program-DRAM address captured at start_capture,
        so the loader MUST DMA the bin to that exact same base address.
        """
        bin_dir = os.path.join(self.script_dir, "gemma4_e2b_bin")
        os.makedirs(bin_dir, exist_ok=True)
        bin_path  = os.path.join(bin_dir, "programs.bin")
        meta_path = os.path.join(bin_dir, "programs.json")

        prefill_max_seq_len = self._cfg["model"].get("prefill_max_seq_len", 512)

        global _SILENT_MODE
        _SILENT_MODE = True
        self.clear_inst_id()
        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()

        instruction_base_addr = self.get_program_dram_addr()

        # Dynamic PBI: single prefill program.
        print(f"[instr-bin] Compiling 1 prefill program (template seq_len={prefill_max_seq_len})...")
        prefill_count_at_start = self.capture_count
        t0 = time.perf_counter()
        _, prefill_total_flops_scalar = self.compile_prefill(seq_len=prefill_max_seq_len,
                                                              layer_size=layer_size)
        prefill_size_bytes = (self.capture_count - prefill_count_at_start) * 32
        prefill_program_addr = instruction_base_addr + prefill_count_at_start * 32
        _original_print(f"  [prefill] {prefill_size_bytes/1024:.1f} KB "
                        f"({time.perf_counter()-t0:.1f}s)")

        # Dynamic PBI: single decoder program.
        decoder_count_at_start = self.capture_count
        print(f"[instr-bin] Compiling 1 decoder program (dynamic PBI)...")
        _, decoder_program_sizes, decoder_total_flops = self.compile_decoder(
            layer_size=layer_size)
        decoder_size_bytes = decoder_program_sizes[0]
        decoder_program_addr = instruction_base_addr + decoder_count_at_start * 32

        self.stop_capture()
        _SILENT_MODE = False

        all_bytes = bytearray()
        for inst in self.capture_buffer:
            all_bytes.extend(inst.get_bytes())
        with open(bin_path, "wb") as f:
            f.write(all_bytes)

        manifest = {
            "compile_version": INSTRUCTION_BIN_COMPILE_VERSION,
            "instruction_bin": os.path.relpath(bin_path, self.script_dir),
            "instruction_base_addr": f"0x{instruction_base_addr:X}",
            "instruction_total_size": len(all_bytes),
            # Dynamic PBI: single prefill / decoder program (no buckets).
            "prefill_max_seq_len": prefill_max_seq_len,
            "prefill_program_start_addr": f"0x{prefill_program_addr:X}",
            "prefill_program_size": prefill_size_bytes,
            "prefill_total_flops": prefill_total_flops_scalar,
            "decoder_max_seq_len": self.MAX_CONTEXT_SIZE,
            "decoder_program_start_addr": f"0x{decoder_program_addr:X}",
            "decoder_program_size": decoder_size_bytes,
            "decoder_total_flops": decoder_total_flops[0],
            "layer_size": layer_size,
            "contains_vision": False,
            "contains_audio":  False,
        }
        with open(meta_path, "w") as f:
            json.dump(manifest, f, indent=2)

        self.clear_capture_buffer()
        print(f"[instr-bin] Wrote {len(all_bytes)/1024/1024:.2f} MB → {bin_path}")
        print(f"[instr-bin] Manifest: {meta_path}")
        return bin_path, manifest

    def load_instruction_bin(self) -> dict:
        """Load programs.bin into program DRAM at the manifest's
        baked base address. Returns the manifest dict (with int addrs).
        Sets self._instruction_loaded = True; subsequent prefill/decode
        dispatches read from manifest addrs.
        """
        bin_dir = os.path.join(self.script_dir, "gemma4_e2b_bin")
        bin_path  = os.path.join(bin_dir, "programs.bin")
        meta_path = os.path.join(bin_dir, "programs.json")
        if not (os.path.exists(bin_path) and os.path.exists(meta_path)):
            raise FileNotFoundError(
                f"Instruction bin missing: {bin_path}. Run compile_instruction_bin() first.")
        with open(meta_path, "r") as f:
            manifest = json.load(f)
        # Read only the program region (instruction_total_size bytes) via
        # chunked DMA from disk, avoiding a ~1 GB f.read() that would spike
        # host RSS (problematic on 16 GB Raspberry Pi). Read in 64 MB chunks
        # so peak transient is bounded — the dma_write transfers to FPGA
        # then we drop the chunk before reading the next one.
        prog_size = manifest["instruction_total_size"]
        base_addr = int(manifest["instruction_base_addr"], 16)
        cur_addr  = self.get_program_dram_addr()
        if cur_addr != base_addr:
            # Rewind allocator so the bin lands at the baked base. The PBI
            # JUMP_ABS targets in the bin reference that exact address.
            self._next_program_dram_addr = base_addr
        CHUNK = 64 * 1024 * 1024
        bytes_written = 0
        with open(bin_path, "rb") as _f:
            while bytes_written < prog_size:
                sz = min(CHUNK, prog_size - bytes_written)
                chunk = _f.read(sz)
                if len(chunk) != sz:
                    raise RuntimeError(
                        f"Truncated instruction bin: read {bytes_written + len(chunk)} of {prog_size} bytes.")
                self.dma_write(DMA_DEVICE_H2C, base_addr + bytes_written, chunk, sz)
                bytes_written += sz
                del chunk
        self.allocate_program_dram(prog_size)

        # Resolve addr strings → ints for caller convenience.
        # Dynamic PBI: single prefill / decoder program.
        manifest["_prefill_addr_int"] = int(manifest["prefill_program_start_addr"], 16)
        manifest["_decoder_addr_int"] = int(manifest["decoder_program_start_addr"], 16)
        manifest["_base_addr_int"]     = base_addr
        if manifest.get("contains_vision") and "vision_program_start_addr" in manifest:
            manifest["_vision_addr_int"] = int(manifest["vision_program_start_addr"], 16)
        if manifest.get("contains_audio") and "audio_program_start_addr" in manifest:
            manifest["_audio_addr_int"] = int(manifest["audio_program_start_addr"], 16)
        print(f"[instr-bin] Loaded {prog_size/1024/1024:.2f} MB at 0x{base_addr:X}"
              + (" (+vision)" if manifest.get("contains_vision") else "")
              + (" (+audio)"  if manifest.get("contains_audio") else ""))
        return manifest

    # ------------------------------------------------------------------
    # LEGACY (commented out 2026-05-21):
    # Incremental vision extension. Folded into single-pass
    # compile_instruction_bin (LM+vision+audio in one capture session) on
    # the build host. run_from_bin is execute-only — it never needs to
    # extend. Kept in source for fast revival if we ever need the
    # add-vision-later workflow back.
    # ------------------------------------------------------------------
    """
    def extend_instruction_bin_with_vision(self, num_patches: int,
                                            rope_pads_tuple: tuple | None = None,
                                            position_ids_sha256: str | None = None) -> dict:
        '''Append the vision encoder program to programs.bin so
        the unified bin holds LM + vision in one file. The vision ISA is
        compiled with PBI baked against `lm_base + lm_size`, exactly where
        the new bytes land at runtime — so JUMP_ABS targets in the vision
        section resolve correctly when the combined bin is DMA'd to the
        original base address.

        Preconditions (caller must have done these — typically
        `_run_vision_encoder_fpga` sets them up):
          - vision_weight_init(hf_model) has been called
          - vision_tensor_init(num_patches, program_base=vision_target_addr)
            has been called with program_base = lm_base + lm_size
          - rope pad DMAs (cos/neg_sin/sin_hi) are done
          - set_vision_attention_bias has been called
          - the LM instruction bin already exists on disk

        On exit:
          - programs.bin is rewritten atomically (old LM bytes +
            new vision bytes)
          - programs.json gains: contains_vision=true,
            vision_program_start_addr, vision_program_size,
            vision_num_patches, vision_layers_per_group, vision_groups,
            vision_group_offsets (relative to vision section),
            vision_group_sizes
          - instruction_total_size is updated to include the vision bytes

        Returns the updated manifest dict (with int addrs resolved).
        '''
        bin_dir = os.path.join(self.script_dir, "gemma4_e2b_bin")
        bin_path  = os.path.join(bin_dir, "programs.bin")
        meta_path = os.path.join(bin_dir, "programs.json")

        # 1. Read existing bin + manifest.
        with open(meta_path, "r") as f:
            manifest = json.load(f)
        if manifest.get("compile_version") != INSTRUCTION_BIN_COMPILE_VERSION:
            raise RuntimeError(
                f"Instruction bin compile_version mismatch "
                f"(disk: {manifest.get('compile_version')!r}, "
                f"expected: {INSTRUCTION_BIN_COMPILE_VERSION!r}). "
                f"Delete {bin_path} and rerun to rebuild from scratch.")
        if manifest.get("contains_vision"):
            print(f"[instr-bin] Vision already in unified bin; nothing to extend.")
            return manifest
        with open(bin_path, "rb") as f:
            old_bytes = f.read()
        if len(old_bytes) != manifest["instruction_total_size"]:
            raise RuntimeError(
                f"Instruction bin size mismatch (disk: {len(old_bytes)}, "
                f"manifest: {manifest['instruction_total_size']}). "
                f"Delete {bin_path} and rerun to rebuild.")

        base_addr = int(manifest["instruction_base_addr"], 16)
        vision_target_addr = base_addr + len(old_bytes)
        # Caller MUST have done vision_tensor_init(program_base=vision_target_addr).
        # Verify the PBI bake base matches what we expect, otherwise the
        # appended JUMPs will land in garbage.
        if self._program_dram_base != vision_target_addr:
            raise RuntimeError(
                f"PBI bake base mismatch: _program_dram_base=0x{self._program_dram_base:X}, "
                f"expected 0x{vision_target_addr:X} (= base + lm_size). "
                f"Caller must call vision_tensor_init(num_patches, "
                f"program_base=vision_target_addr) before extending.")
        # Reset allocator to base so PBI bakes target = base + capture_count*32.
        self._next_program_dram_addr = vision_target_addr

        # 2. Emit vision ISA into capture buffer.
        # Single-group only (G=VIS_LAYERS): the unified-bin dispatch is one
        # start_execute per vision pass. Multi-group (G<L) would require
        # multi-step execute that the unified path doesn't support yet.
        L = self.VIS_LAYERS
        G = L
        print(f"[instr-bin] Extending with vision: {L} layers, baked at 0x{vision_target_addr:X}", flush=True)

        import builtins
        _orig_print = builtins.print
        global _SILENT_MODE
        self._oneshot_mode = True
        _SILENT_MODE = True
        self.clear_capture_buffer()
        self.start_capture()
        builtins.print = lambda *a, **kw: None
        try:
            t_compile = time.perf_counter()
            for li in range(L):
                t_layer = time.perf_counter()
                self.compile_vision_layer(li)
                self.host_vision_v_norm_rope_gather(li)
                self.run_vision_attention_all_heads(li)
                self.compile_vision_layer_post_attn(li)
                _original_print(
                    f"  [Vision] layer {li+1}/{L} compiled in {time.perf_counter()-t_layer:.2f}s",
                    flush=True)
        finally:
            builtins.print = _orig_print
            _SILENT_MODE = False
            self._oneshot_mode = False
        self.stop_capture()
        self.generate_instruction_halt()
        vision_bytes = bytearray()
        for inst in self.capture_buffer:
            vision_bytes.extend(inst.get_bytes())
        self.clear_capture_buffer()
        vision_size = len(vision_bytes)
        _original_print(f"[instr-bin] Vision ISA: {vision_size/1024/1024:.1f} MB "
                        f"({time.perf_counter()-t_compile:.1f}s)", flush=True)

        # 3. Serialize rope pads (data, not program ISA). They get appended
        # to the bin file AFTER the program region so the single bin file
        # holds everything vision needs. The program region (instruction_total_size)
        # covers LM + vision ISA only — rope pad bytes sit in the file but
        # are NOT DMA'd to program DRAM; the runtime reads them from disk
        # and DMAs to vision tensor DRAM (VIS_ROPE_*_PAD_TILED) on each
        # VLM run, since those tensor addresses are allocator-dependent.
        if rope_pads_tuple is not None:
            cos_pad, neg_sin_pad, sin_hi_pad = rope_pads_tuple
            cos_b      = cos_pad.contiguous().view(torch.uint8).numpy().tobytes()
            neg_sin_b  = neg_sin_pad.contiguous().view(torch.uint8).numpy().tobytes()
            sin_hi_b   = sin_hi_pad.contiguous().view(torch.uint8).numpy().tobytes()
            rope_blob  = cos_b + neg_sin_b + sin_hi_b
        else:
            cos_b = neg_sin_b = sin_hi_b = rope_blob = b""

        # 4. Concatenate and atomically rewrite the bin.
        # Layout: [LM core | vision ISA | rope pads]
        #         └────── program region ─────┘ └data┘
        program_region = old_bytes + bytes(vision_bytes)
        new_bytes      = program_region + rope_blob
        program_size   = len(program_region)
        bin_tmp  = bin_path  + ".tmp"
        meta_tmp = meta_path + ".tmp"
        with open(bin_tmp, "wb") as f:
            f.write(new_bytes)
            f.flush()
            os.fsync(f.fileno())
        # 5. Update manifest. Keep all old fields; add vision section info.
        # instruction_total_size = program-region size (what gets DMA'd to
        # program DRAM). total_file_size = full file size including
        # trailing data sections (rope pads).
        manifest["instruction_total_size"] = program_size
        manifest["total_file_size"]        = len(new_bytes)
        manifest["contains_vision"]        = True
        manifest["vision_program_start_addr"] = f"0x{vision_target_addr:X}"
        manifest["vision_program_size"]    = vision_size
        manifest["vision_num_patches"]     = num_patches
        manifest["vision_layers_per_group"] = G
        manifest["vision_groups"]          = [[0, L]]      # one group covering all layers
        manifest["vision_group_offsets"]   = [0]           # rel to vision section start
        manifest["vision_group_sizes"]     = [vision_size]
        if rope_blob:
            # File offsets (absolute byte positions in the bin) for the
            # three rope pad tensors. Each is (num_patches * NH, HD) bf16.
            cos_off     = program_size
            neg_sin_off = cos_off + len(cos_b)
            sin_hi_off  = neg_sin_off + len(neg_sin_b)
            manifest["vision_rope_cos_offset"]     = cos_off
            manifest["vision_rope_cos_size"]       = len(cos_b)
            manifest["vision_rope_neg_sin_offset"] = neg_sin_off
            manifest["vision_rope_neg_sin_size"]   = len(neg_sin_b)
            manifest["vision_rope_sin_hi_offset"]  = sin_hi_off
            manifest["vision_rope_sin_hi_size"]    = len(sin_hi_b)
            if position_ids_sha256:
                manifest["vision_rope_position_ids_sha256"] = position_ids_sha256
        with open(meta_tmp, "w") as f:
            json.dump(manifest, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.rename(bin_tmp, bin_path)
        os.rename(meta_tmp, meta_path)
        assert os.path.getsize(bin_path) == len(new_bytes), "extended bin disk size mismatch"
        # Clean up the now-redundant standalone rope pad cache file (and any
        # other legacy per-section caches) so the user sees ONE bin on disk.
        for _stale in (f"vision_rope_pads_{num_patches}p.bin",
                       f"vision_rope_pads_{num_patches}p.json",
                       f"vision_encoder_oneshot_v6_{num_patches}p_g{L}.bin",
                       f"vision_encoder_oneshot_v6_{num_patches}p_g{L}.json",
                       f"vision_program_cache_v9_{num_patches}p.bin",
                       f"vision_program_cache_v9_{num_patches}p.json"):
            _stale_path = os.path.join(bin_dir, _stale)
            if os.path.exists(_stale_path):
                os.remove(_stale_path)
                _original_print(f"[instr-bin] Removed legacy cache: {_stale}")
        print(f"[instr-bin] Rewrote {len(new_bytes)/1024/1024:.2f} MB → {bin_path}"
              f" (program region {program_size/1024/1024:.2f} MB, data region {len(rope_blob)/1024/1024:.2f} MB)")
        print(f"[instr-bin] Manifest updated: contains_vision=true, vision at 0x{vision_target_addr:X}")
        return manifest
    """
    # End of LEGACY extend_instruction_bin_with_vision (commented out).

    # ------------------------------------------------------------------
    # Bucketed prefill dispatch (used with single instruction bin)
    # ------------------------------------------------------------------
    def run_prefill_bucketed(self, manifest: dict, prefill_seq=None) -> tuple[int, float]:
        """Pick the smallest prefill bucket >= actual prompt len, pad the
        prompt with its last token, dispatch the corresponding bucket
        program, then zero out KV-cache positions written by padding so
        decode does not attend to garbage.
        Returns (latency, flop_rate). On exit self.seq_len equals the
        ACTUAL prompt length (not bucket length) so decode advances from
        the right offset.
        """
        if prefill_seq is None:
            prefill_seq = self.prefill_seq
        if prefill_seq is None or len(prefill_seq) < 2:
            raise ValueError("Prefill sequence must have at least 2 tokens.")

        prefill_tokens = tuple(prefill_seq[:-1])
        actual_seq_len = len(prefill_tokens)
        prefill_max = manifest["prefill_max_seq_len"]
        if actual_seq_len > prefill_max:
            raise ValueError(
                f"prefill length {actual_seq_len} exceeds compiled "
                f"prefill_max_seq_len={prefill_max}. Recompile with a larger value.")

        prefill_program_addr = manifest["_prefill_addr_int"]
        flops = manifest["prefill_total_flops"]

        # Pad target. The compiled prefill is dynamic-PBI (bulk matmuls read
        # M from gpr_seq_len at execute time; see compile_prefill), so padding
        # to prefill_max spends template-sized matmul work on every prompt.
        # GEMMA4_DYN_PREFILL=1 pads only to the next 64-aligned bucket:
        # compiled per-token loops beyond the bucket still iterate at
        # PREFILL_MAX, but their outputs land in KV rows that (a) causal
        # prefill attention never lets real rows see, (b) the decode bias
        # masks until (c) decode overwrites them position by position. The
        # matmul M drops to the bucket length. Off by default until the
        # A/B token-stream parity run passes on hardware.
        if os.environ.get("GEMMA4_DYN_PREFILL", "0") == "1":
            pad_to = min(prefill_max, ((actual_seq_len + 63) // 64) * 64)
            # Report FLOPs scaled to the primed M (linear approximation;
            # the attention term is quadratic, so the printed GFLOPS is
            # indicative only in this mode).
            flops = int(flops * pad_to / prefill_max)
        else:
            pad_to = prefill_max

        # Pad so per-token loops up to the primed length see valid data.
        pad_token = prefill_tokens[-1]
        pad_count = pad_to - actual_seq_len
        padded_tokens = list(prefill_tokens) + [pad_token] * pad_count
        padded_seq = tuple(padded_tokens) + (prefill_seq[-1],)
        if hasattr(self, "_mm_types") and self._mm_types is not None and pad_count > 0:
            self._mm_types = list(self._mm_types[:actual_seq_len]) + [0] * pad_count

        print(f"[dyn-prefill] actual={actual_seq_len}, template_buffer={prefill_max}, "
              f"gpr_seq_len={pad_to}, addr=0x{prefill_program_addr:X}")

        # ------------------------------------------------------------------
        # LM-state restore (post vision/audio compile). vision_tensor_init
        # deliberately reuses LM tensor DRAM (resets the allocator cursor)
        # so any compile_instruction_bin call that included image_path /
        # audio_path leaves vision encoder data in LM tensor regions.
        # Mirrors set_prefill_seq_vlm's restoration so the pure-LM path
        # produces the same DRAM state the LM ISA expects. Idempotent for
        # LM-only-bin runs (just a few small DMAs).
        # ------------------------------------------------------------------
        self.software_reset()  # clear stuck queue state from any prior failing run
        from user_dma_core import UE_VECTOR_SIZE as _UE_VS
        # Zero ENTIRE K/V cache (0..MAX_CONTEXT_SIZE). Prefill overwrites
        # positions 0..prefill_max-1 with K/V data; positions prefill_max..
        # MAX stay zero so decoder's gather + bias-masked softmax sees clean
        # zeros (not NaN-prone vision floats). NOTE: tensor size is in
        # element count (bfloat16), not bytes — self.head_dim, not self.k_size.
        num_slots = getattr(self, "_num_kv_slots", self.LAYER_SIZE)
        kv_slot_elems = num_slots * self.MAX_CONTEXT_SIZE * self.head_dim
        kv_zero_pad = torch.zeros(kv_slot_elems, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, kv_zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_K_ROPE_DRAM, kv_zero_pad)
        # Re-DMA the identity matrix (decoder_attention_core reads it for
        # the I @ V^T step). Done AFTER the K/V zeros so accidental adjacent
        # buffer overflow from those big DMAs cannot clobber IDENTITY.
        # Re-zero the flash-attention Q/K/V gather buffers (tensor_init does this).
        # Decoder writes them per gather but LM prefill's per-token loops at
        # template_seq_len assume zero-padding for rows past actual_seq_len.
        try:
            from user_dma_core import UE_FMAX_CONTEXT_SIZE  # noqa: F401
            _pre_align = ((self.max_prefill_seq_len * self.group_size + 63) // 64) * 64
        except Exception:
            _pre_align = self.max_prefill_seq_len * self.group_size
        flash_qkv_elems = _pre_align * self.head_dim
        _flash_zero = torch.zeros(flash_qkv_elems, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_Q_DRAM, _flash_zero)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_K_DRAM, _flash_zero)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_V_DRAM, _flash_zero)
        # Re-DMA the identity matrix (decoder_attention_core reads it for
        # the I @ V^T step). Done AFTER the K/V + FLASH zeros so accidental
        # adjacent-buffer overflow from those big DMAs cannot clobber IDENTITY.
        self.dma_to_accelerator_memory(self.IDENTITY_DRAM_ADDR,
                                       torch.eye(_UE_VS, dtype=torch.bfloat16))
        print(f"[dyn-prefill] LM-state restored ({num_slots} KV slots zeroed, IDENTITY re-uploaded)")

        self.seq_len = pad_to
        latency, flop_rate = self.run_prefill(prefill_program_addr,
                                              prefill_seq=padded_seq,
                                              flops=flops)
        self.seq_len = actual_seq_len
        return latency, flop_rate

    def run_decoder(self, decoder_program_sizes: list[int], decoder_base_addr: int, token_id: int, flops_per_token: list[int] | None = None) -> dict:
        """Run decode loop with dynamic PBI.

        Single decoder program — gpr_seq_len primed ONCE; program self-advances
        via trailing add_inc. gpr_bucket_idx re-set each step.
        """
        if token_id is None:
            print("No last token available for decode.")
            return {}

        global _SILENT_MODE
        max_seq_len = self.MAX_CONTEXT_SIZE
        _maxdec = os.environ.get("GEMMA4_MAX_DECODE")   # benchmark cap (e.g. 128); default off
        if _maxdec:
            max_seq_len = min(max_seq_len, self.seq_len + int(_maxdec))
        total_latency, total_flop_rate = 0, 0
        prog_addr = decoder_base_addr
        flops_per_token_scalar = flops_per_token[0] if flops_per_token else None

        # On-FPGA repetition penalty (llama3.2_1b mechanism), DEFAULT OFF. The
        # LM-head matmul (baked in the bin) adds PENALTY_BIAS_DRAM as its C term;
        # keeping it zero == pure greedy (HW argmax, no readback). GEMMA4_PENALTY=1
        # refreshes the bias each step so the HW argmax returns the penalized token.
        _pen_off        = os.environ.get("GEMMA4_PENALTY", "0") != "1"
        self.pen_alpha  = float(os.environ.get("GEMMA4_PEN_ALPHA", "1.0"))
        self.pen_cap    = float(os.environ.get("GEMMA4_PEN_CAP", "20.0"))
        self.rep_window = int(os.environ.get("GEMMA4_REP_WINDOW", "256"))
        _greedy_until   = int(os.environ.get("GEMMA4_GREEDY_UNTIL", "0"))
        self.pen_loop_recent = int(os.environ.get("GEMMA4_PEN_LOOP_RECENT", "24"))  # anti-loop window (0=off)
        self.pen_loop_thr    = int(os.environ.get("GEMMA4_PEN_LOOP_THR", "8"))       # ban tok at >= thr of last RECENT
        _gen_tokens: list[int] = []
        _n_generated = 0
        self.dma_to_accelerator_memory(
            self.PENALTY_BIAS_DRAM,
            torch.zeros(1, self.EMBEDDING_ELEMENTS, dtype=torch.bfloat16))
        if not _pen_off:
            print(f"[decode] on-FPGA repetition penalty ON "
                  f"(alpha={self.pen_alpha} cap={self.pen_cap} window={self.rep_window} "
                  f"greedy_until={_greedy_until} loop={self.pen_loop_recent}/{self.pen_loop_thr}); "
                  f"unset GEMMA4_PENALTY for pure greedy")

        # Prime gpr_seq_len once (= prompt length); program advances it per step.
        self.isa_add_set_core(self.gpr_seq_len, self.seq_len)
        print("\n------------------------------ DECODE START ------------------------------\n", flush=True)

        # Live decode status bar (mirrors llama3.2_1b): pin the bottom terminal
        # row via an ANSI scroll region; generated tokens stream above it while a
        # tokens/s counter refreshes in place. TTY-only (skipped when piped).
        import shutil
        _dec_start_seq = self.seq_len
        _dec_timer = time.perf_counter()
        _first_tok_dt = None   # wall-clock of the 1st decoded token → peak tok/s
        _decoded_n = 0         # number of decode steps (for average tok/s)
        _use_status = sys.stdout.isatty()
        def _status_setup():
            rows = shutil.get_terminal_size().lines
            sys.stdout.write(f"\033[1;{rows - 1}r")   # scroll region = rows 1..rows-1
            sys.stdout.write(f"\033[{rows - 1};1H")   # park cursor at bottom of region
            sys.stdout.flush()
        def _status_update():
            rows = shutil.get_terminal_size().lines
            n = self.seq_len - _dec_start_seq
            elapsed = time.perf_counter() - _dec_timer
            rate = n / elapsed if elapsed > 0 else 0.0
            sys.stdout.write("\0337")                  # save cursor
            sys.stdout.write(f"\033[{rows};1H\033[2K") # bottom row, clear it
            sys.stdout.write(f" decoding… {n} tokens  (pos {self.seq_len}/{self.MAX_CONTEXT_SIZE})  "
                             f"{elapsed:.1f}s  {rate:.1f} tok/s")
            sys.stdout.write("\0338")                  # restore cursor
            sys.stdout.flush()
        def _status_teardown():
            rows = shutil.get_terminal_size().lines
            sys.stdout.write("\033[r")                 # reset scroll region
            sys.stdout.write(f"\033[{rows};1H\033[2K") # clear the status row
            sys.stdout.flush()
        if _use_status:
            _status_setup()

        while self.seq_len < max_seq_len:
            _SILENT_MODE = True
            _tok_t0 = time.perf_counter()               # per-token wall-clock start
            self.seq_len += 1
            aligned_seq_len = ((self.seq_len + 63) // 64) * 64
            dec_bucket_idx = aligned_seq_len // UE_VECTOR_SIZE  # 1-based
            self.isa_add_set_core(self.gpr_bucket_idx, dec_bucket_idx)

            embedding_tensor = self.get_embedding_for_tokens([token_id])
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)

            # Compute per-layer inputs for single token and DMA
            per_layer_inputs = self._compute_per_layer_inputs([token_id], embedding_tensor)  # [1, 35, 256]
            # Permute to [35, 1, 256] so each layer's data is contiguous in DRAM
            self.dma_to_accelerator_memory(self.PER_LAYER_INPUTS_DRAM, per_layer_inputs.permute(1, 0, 2).contiguous())

            # Build BOTH decode bias rows. Full attention layers see all
            # positions [0, seq_len); sliding-attention layers see only the
            # last `sliding_window` tokens [max(0, seq_len-window), seq_len).
            full_bias_row = torch.full((1, aligned_seq_len), -1e36, dtype=torch.bfloat16)
            full_bias_row[0, :self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_FULL_DRAM, full_bias_row)
            if self.seq_len <= self.sliding_window:
                sliding_bias_row = full_bias_row
            else:
                sliding_bias_row = torch.full((1, aligned_seq_len), -1e36, dtype=torch.bfloat16)
                window_start = self.seq_len - self.sliding_window
                sliding_bias_row[0, window_start:self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_SLIDING_DRAM, sliding_bias_row)

            # On-FPGA penalty: refresh THIS step's per-vocab bias (the LM-head
            # matmul's C term) once past the greedy gate, so the HW argmax of
            # (logits + bias) returns the penalized token. No logit readback.
            if not _pen_off and _n_generated >= _greedy_until:
                self._write_penalty_bias(_gen_tokens)

            latency, flop_rate_program = self.program_execute(prog_addr, flops=flops_per_token_scalar)
            total_latency += latency
            total_flop_rate += flop_rate_program
            token_id = self.get_arg_max_index()
            _gen_tokens.append(int(token_id))
            _n_generated += 1
            token_char = self.tokenizer.decode([token_id])
            _SILENT_MODE = False

            _tok_dt = time.perf_counter() - _tok_t0
            if _first_tok_dt is None:
                _first_tok_dt = _tok_dt                  # 1st token (shortest context) → peak
            _decoded_n += 1

            if token_id in [1, self._end_of_turn_token_id]:
                if _use_status:
                    _status_teardown()
                print(f"\nStop token {token_id} reached.")
                break
            print(token_char, end="", flush=True)
            if _use_status:
                _status_update()
        else:
            if _use_status:
                _status_teardown()
        # Decode-speed report (loop wall-clock basis — matches gemma4_e2b_test.py).
        _elapsed = time.perf_counter() - _dec_timer
        _peak = (1.0 / _first_tok_dt) if _first_tok_dt else 0.0
        _avg = (_decoded_n / _elapsed) if _elapsed > 0 else 0.0
        print(f"\nDecode speed: peak (1st token) {_peak:.2f} tok/s, "
              f"average {_avg:.2f} tok/s  ({_decoded_n} tokens in {_elapsed:.2f}s)")
        return self.seq_len, total_latency, total_flop_rate

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Gemma4 E2B layer-0 prefill: run on accelerator, verify with torch ref.")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt. Default is a built-in prompt per mode "
                             "(LM: a test question; VLM: 'Describe this image in detail.'; "
                             "Audio: 'Describe what you hear.').")
    # Modality selection. Two ways to enable VLM or audio:
    #   (a) Pass a path via --image / --audio — that path is used AND the
    #       matching encoder is enabled automatically.
    #   (b) Pass --vision-enable / --audio-enable (no path) — the template
    #       uses a default example file shipped at ../../test_samples/
    #       (relative to this script) so the CLI works out-of-the-box.
    # Passing both modalities at once is rejected; LM-only is the default
    # when neither is selected.
    parser.add_argument("--image", type=str, default=None,
                        help="Path to image file for VLM inference. Implies --vision-enable.")
    parser.add_argument("--vision-enable", action="store_true",
                        help="Enable VLM mode with the default example image "
                             "(../../test_samples/yosemite.jpg, relative to this script). "
                             "Ignored if --image is also given.")
    parser.add_argument("--audio", type=str, default=None,
                        help="Path to audio file (.wav, .flac, etc.). Implies --audio-enable.")
    parser.add_argument("--audio-enable", action="store_true",
                        help="Enable audio mode with the default example audio "
                             "(../../test_samples/apex.wav, relative to this script). "
                             "Ignored if --audio is also given.")
    parser.add_argument("--local-weights", action="store_true", help="Use gemma4_e2b_bin/params.bin instead of generated params.bin")
    parser.add_argument('--dev', type=str, default='xdma0',
                        help='DMA device name (e.g., xdma0, xdma1). Default: xdma0')
    parser.add_argument('--cycle', type=float, default=5.62,
                        help='Clock cycle time in nanoseconds (default: 3.0, use 2.5 for alveo)')
    args = parser.parse_args()

    set_dma_device(args.dev)
    global DMA_DEVICE_H2C, DMA_DEVICE_C2H, DMA_DEVICE_USER
    DMA_DEVICE_H2C = user_dma_core.DMA_DEVICE_H2C
    DMA_DEVICE_C2H = user_dma_core.DMA_DEVICE_C2H
    DMA_DEVICE_USER = user_dma_core.DMA_DEVICE_USER
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle
    print(f"Using DMA device: {args.dev}")
    print(f"  H2C: {DMA_DEVICE_H2C}")
    print(f"  C2H: {DMA_DEVICE_C2H}")
    print(f"  USER: {DMA_DEVICE_USER}")
    print(f"Setting CLOCK_CYCLE_TIME_NS = {user_dma_core.CLOCK_CYCLE_TIME_NS}")

    # --- Resolve modality and input path -----------------------------------
    # The script has three modes: LM (text only), VLM (image + text),
    # and audio (audio + text). Exactly one encoder modality can be active
    # per run. Enabling rules:
    #   * --image PATH             → VLM with that image
    #   * --vision-enable (no path) → VLM with the shipped default image
    #   * --audio PATH             → audio with that file
    #   * --audio-enable (no path) → audio with the shipped default file
    #   * none of the above        → pure LM
    # Default example inputs live in a shared test_samples directory at the
    # template level (two levels up from this script). Multiple model folders
    # can share the same example files.
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _TEST_SAMPLES = os.path.normpath(os.path.join(_HERE, "..", "..", "test_samples"))
    DEFAULT_IMAGE = os.path.join(_TEST_SAMPLES, "yosemite.jpg")
    DEFAULT_AUDIO = os.path.join(_TEST_SAMPLES, "apex.wav")

    vision_on = bool(args.image) or args.vision_enable
    audio_on = bool(args.audio) or args.audio_enable
    if vision_on and audio_on:
        raise SystemExit(
            "Only one encoder modality per run. Choose either --image / --vision-enable "
            "OR --audio / --audio-enable, not both.")

    image_path = args.image or (DEFAULT_IMAGE if args.vision_enable else None)
    audio_path = args.audio or (DEFAULT_AUDIO if args.audio_enable else None)
    if vision_on and image_path and not os.path.exists(image_path):
        raise SystemExit(f"Image file not found: {image_path}")
    if audio_on and audio_path and not os.path.exists(audio_path):
        raise SystemExit(f"Audio file not found: {audio_path}")

    ue = Gemma4_UnifiedEngine(local_weights=args.local_weights)

    # Execute-only: the unified instruction bin MUST already exist on disk
    # (built once on the beefier build host by gemma4_e2b_test.py). Refuse
    # to compile here. Load BEFORE set_prefill_seq_{vlm,audio} so the
    # encoder run paths find the bin already in DRAM with its manifest.
    bin_dir = os.path.join(SCRIPT_DIR, "gemma4_e2b_bin")
    instr_bin = os.path.join(bin_dir, "programs.bin")
    instr_meta = os.path.join(bin_dir, "programs.json")
    if not (os.path.exists(instr_bin) and os.path.exists(instr_meta)):
        raise SystemExit(
            f"ERROR: programs.bin not found.\n"
            f"  Missing: {instr_bin}\n"
            f"           {instr_meta}\n"
            f"Generate it on the build host with:\n"
            f"  python models/gemma4_e2b/gemma4_e2b_test.py --audio-enable\n"
            f"  python models/gemma4_e2b/gemma4_e2b_test.py --vision-enable\n"
            f"(any single first run builds LM+vision+audio together), then\n"
            f"copy gemma4_e2b_bin/programs.{{bin,json}} here.")
    print(f"[run_from_bin] instruction bin present, loading...")
    manifest = ue.load_instruction_bin()

    try:
        if vision_on:
            print(f"[Mode] VLM — image: {image_path}")
            ue.set_prefill_seq_vlm(image_path, prompt=args.prompt)
        elif audio_on:
            print(f"[Mode] Audio — audio: {audio_path}")
            ue.set_prefill_seq_audio(audio_path, prompt=args.prompt)
        elif args.prompt:
            print(f"[Mode] LM — prompt: {args.prompt!r}")
            ue.set_prefill_seq(args.prompt)
        else:
            print(f"[Mode] LM — default prompt")
            ue.set_prefill_seq()
    except BaseException:
        raise
    if os.environ.get("GEMMA4_ENCODER_ONLY"):
        print("[Mode] GEMMA4_ENCODER_ONLY set — exiting after encoder, "
              "skipping prefill+decode.")
        return

    # Reload manifest if vision/audio executed (they call load_instruction_bin
    # internally after their tensor allocations).
    if vision_on or audio_on:
        manifest = ue.load_instruction_bin()
    # Dynamic PBI: single decoder program. Wrap in 1-element list for signature.
    decoder_program_sizes = [manifest["decoder_program_size"]]
    flops_per_token = [manifest["decoder_total_flops"]]
    decoder_base_addr = manifest["_decoder_addr_int"]
    # NOTE: the decoder bin is loaded into DRAM AFTER run_prefill completes,
    # reusing the prefill bin's DRAM region. For long prompts (e.g. VLM with
    # ~280 tokens) the prefill bin is ~180 MB and decoder bin is ~66 MB; holding
    # both at once would overflow the 4 GB DRAM boundary. Since the prefill bin
    # is dead code once run_prefill finishes, overwriting it with the decoder
    # bin keeps peak program-DRAM usage at max(prefill, decoder) instead of
    # prefill + decoder.

    print(f"\n--- Starting prefill ---")
    # Compress runs of repeated tokens (e.g. ~270 image-padding tokens in VLM)
    # into a "<id>*N" form so the prompt list stays readable. Runs of length
    # >= 4 are collapsed; shorter runs print verbatim.
    def _fmt_tokens(seq, min_run=4):
        parts = []
        i = 0
        while i < len(seq):
            j = i
            while j < len(seq) and seq[j] == seq[i]:
                j += 1
            run = j - i
            if run >= min_run:
                parts.append(f"{seq[i]}*{run}")
            else:
                parts.extend(str(t) for t in seq[i:j])
            i = j
        return ", ".join(parts)
    # Collapse runs of identical <...>-style special tokens (e.g. 256 ×
    # <|image|>) into "<|image|>*256" form so the printed prompt stays
    # readable in VLM mode.
    import re as _re_collapse
    _collapse_re = _re_collapse.compile(r'(<[^<>]+>)\1{3,}')
    def _collapse_text_runs(s):
        return _collapse_re.sub(
            lambda m: f"{m.group(1)}*{(m.end() - m.start()) // len(m.group(1))}", s)
    print(f"--- Prompt begin ({len(ue.prefill_seq)} tokens) ---")
    print(f"  [{_fmt_tokens(ue.prefill_seq)}]")
    try:
        _prompt_text = ue.tokenizer.decode(list(ue.prefill_seq), skip_special_tokens=False)
        print(f"  text: {_collapse_text_runs(_prompt_text)!r}")
    except Exception as _e:
        print(f"  text: (decode failed: {_e})")
    print(f"--- Prompt end ---")
    timer=time.perf_counter()
    latency_hw_prefill, flop_rate_hw_prefill = ue.run_prefill_bucketed(manifest)
    latency_prefill = time.perf_counter() - timer
    print(f"Prefill execute done in {latency_prefill:.2f} seconds, start decoding...\n", flush=True)

    timer=time.perf_counter()
    token_cnt_decoded, latency_hw_decoder, flop_rate_hw_decoder = ue.run_decoder(decoder_program_sizes, decoder_base_addr, token_id=ue.prefill_seq[-1], flops_per_token=flops_per_token)
    latency_decoder = time.perf_counter() - timer
    print(f"\nDecoder done in {latency_prefill + latency_decoder:.2f} seconds, speed: {(token_cnt_decoded - len(ue.prefill_seq) + 1) / latency_decoder:.2f} tokens/s, total {token_cnt_decoded} tokens.")
    print(f"HW counter: Latency: {(latency_hw_prefill + latency_hw_decoder) / 1e6:.2f} seconds, decoder average Gflops: {flop_rate_hw_decoder / (token_cnt_decoded - len(ue.prefill_seq) + 1):.2f} Gflops")
    print("Gemma4 E2B test ends.")

if __name__ == "__main__":
    raise SystemExit(
        "DEPRECATED / OUTDATED: gemma4_e2b_run_from_bin.py is no longer runnable. It "
        "still calls flash_attention_core / decoder_group_attention_core, which were "
        "removed from user_dma_core.py during the unified_attention_core migration, so "
        "its compile paths reference methods that no longer exist. Use the migrated "
        "gemma4_e2b_test.py (unified_attention_core) instead."
    )
