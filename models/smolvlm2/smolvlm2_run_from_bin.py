#!/usr/bin/env python3
"""SmolVLM2-500M inference from pre-compiled bin files. Run smolvlm2_test.py first to generate bins."""
import builtins
import json
import math
import os
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

# Suppress prints during decode (same as smolvlm2_test.py)
_original_print = builtins.print
_SILENT_MODE = False
def _quiet_print(*args, **kwargs):
    if not _SILENT_MODE:
        _original_print(*args, **kwargs)
builtins.print = _quiet_print

import torch
from transformers import AutoTokenizer

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, TYPE, UE_VECTOR_SIZE, UE_ARGMAX_INDEX, UE_ARGMAX1_INDEX,
    URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS,
    DRAM_INSTRUCTION_ADDR, INSTRUCTION_REG_REWRITE, MEMCPY_TYPE,
    UnifiedEngine, set_dma_device, ue_35bit_addr_shifter,
)

# =============================================================================
# Config loading
# =============================================================================
def _load_smolvlm2_config(path: str = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smolvlm2_config.json")
    with open(path) as f:
        return json.load(f)

_SMOLVLM2_CFG = _load_smolvlm2_config()

# =============================================================================
# Helper functions
# =============================================================================
def init_hang_prevention(ue) -> None:
    """Stop stale execution and write HALT to instruction DRAM base."""
    print("[Init] Hang prevention: disabling instruction execution...")
    ue.dram_inst_running(False)
    ue.start_capture()
    ue.generate_instruction_halt()
    ue.stop_capture()
    halt_bytes = bytearray()
    for inst in ue.capture_buffer:
        halt_bytes.extend(inst.get_bytes())
    ue.dma_write(DMA_DEVICE_H2C, DRAM_INSTRUCTION_ADDR, halt_bytes, len(halt_bytes))
    ue.clear_capture_buffer()
    print("[Init] HALT written to instruction DRAM base")

def isa_set_register(ue, dst_reg_idx: int, immediate_value: int, timeout_s: float = 10.0) -> None:
    """Set one ISA register to an immediate value via minimal program."""
    ue._inst_id = 0
    ue.start_capture()
    ue.generate_instruction_add_set(dst_reg_idx, immediate_value)
    ue.generate_instruction_halt()
    ue.stop_capture()
    program_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())
    ue.clear_capture_buffer()
    ue.start_execute_from_dram(program_addr)
    ue.wait_queue(timeout_s)

def _make_add_set_bytes(dst_reg: int, immediate_value: int) -> bytes:
    """Build raw 32-byte ADD_SET instruction: dst_reg = immediate_value.
    Encoding must match user_dma_core.generate_instruction / andromeda.c layout."""
    import struct
    INSTRUCTION_ADD = 2
    INST_ADD_SET = 4
    w = [0] * 8
    w[0] = (INSTRUCTION_ADD & 0xF) << 8
    w[1] = ((INST_ADD_SET & 0xF) << 0) | \
           ((dst_reg & 0xF) << 4) | \
           ((dst_reg & 0xF) << 8) | \
           ((0 & 0xF) << 12) | \
           ((immediate_value & 0xFFFF) << 16)
    w[2] = (immediate_value >> 16) & 0xFFFF
    result = bytearray(32)
    for i in range(8):
        result[i*4:(i+1)*4] = struct.pack('<I', w[i] & 0xFFFFFFFF)
    return bytes(result)

def accelerator_memory_to_sram_reg(ue, accelerator_dram_address: int, sram_address: int,
                                    element_size: int, general_reg_src: int,
                                    stride_bytes_per_chunk: int = 0, stride_jump_bytes: int = 0) -> None:
    """accelerator_memory_to_sram with register-based DRAM address."""
    element_size_bytes = element_size * 2
    uram_type, uram_start_addr = ue.sram_address_to_uram_address(sram_address)
    ue.ue_memcpy_from_dram(accelerator_dram_address, element_size_bytes, MEMCPY_TYPE.URAM.value,
                           uram_start_addr, uram_type.value,
                           stride_bytes_per_chunk=stride_bytes_per_chunk,
                           stride_jump_bytes=stride_jump_bytes,
                           general_reg_src=general_reg_src)

def capture_to_raw(ue):
    """Stop capture, extract raw instruction bytes (no halt), clear buffer."""
    ue.stop_capture()
    raw = bytearray()
    for inst in ue.capture_buffer:
        raw.extend(inst.get_bytes())
    ue.clear_capture_buffer()
    return bytes(raw)

def generate_halt_raw(ue):
    """Return raw bytes for a single HALT instruction."""
    ue.start_capture()
    ue.generate_instruction_halt()
    ue.stop_capture()
    raw = bytearray()
    for inst in ue.capture_buffer:
        raw.extend(inst.get_bytes())
    ue.clear_capture_buffer()
    return bytes(raw)

# =============================================================================
# SmolVLM2_UnifiedEngine class
# =============================================================================
class SmolVLM2_UnifiedEngine(UnifiedEngine):
    """SmolVLM2-500M accelerator engine: weight loading and inference."""
    # --- Model dimensions (SmolVLM2-500M) — loaded from smolvlm2_config.json ---
    _cfg = _SMOLVLM2_CFG
    HIDDEN_SIZE       = _cfg["lm"]["hidden_size"]
    NUM_LAYERS        = _cfg["lm"]["num_layers"]
    NUM_HEADS         = _cfg["lm"]["num_heads"]
    NUM_KV_HEADS      = _cfg["lm"]["num_kv_heads"]
    GROUP_SIZE        = _cfg["lm"]["group_size"]
    HEAD_DIM          = _cfg["lm"]["head_dim"]
    INTERMEDIATE_SIZE = _cfg["lm"]["intermediate_size"]
    VOCAB_SIZE        = _cfg["lm"]["vocab_size"]
    ROPE_THETA        = _cfg["lm"]["rope_theta"]
    RMS_NORM_EPS      = _cfg["lm"]["rms_norm_eps"]
    MAX_POSITION_EMBEDDINGS = _cfg["lm"]["max_position_embeddings"]
    # ISA register assignments (fixed, used by compile_decoder and run_decoder)
    V_CACHE_SIZE_REG = _cfg["fixed_isa_regs"]["V_CACHE_SIZE_REG"]
    ROPE_SIZE_REG    = _cfg["fixed_isa_regs"]["ROPE_SIZE_REG"]
    TMP_REG          = _cfg["fixed_isa_regs"]["TMP_REG"]

    def __init__(self, script_dir: str = None, lm_weights: str = None, vision_weights: str = None,
                 vision_bf16: bool = True):
        self.script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
        self._cfg = _SMOLVLM2_CFG
        # DRAM layout: bf16 layout needed when vision uses BF16 weights
        if vision_bf16:
            dl = self._cfg["dram_layout"]["bf16"]
            super().__init__(
                params_dram_base=int(dl["params_dram_base"], 16),
                tensor_dram_base=int(dl["tensor_dram_base"], 16),
                program_dram_base=int(dl["program_dram_base"], 16),
            )
        else:
            super().__init__()
        self._isa_reg_counter = 1
        self.vision_bf16 = vision_bf16
        self.lm_bf16 = False

        # Weight bin paths (generated by --gen-weights)
        self.lm_weights_path = lm_weights or os.path.join(self.script_dir, self._cfg["paths"]["lm_weights"])
        self.vision_weights_path = vision_weights or os.path.join(self.script_dir, self._cfg["paths"]["vision_weights"])

    # --- ISA register helpers ---
    def reset_isa_reg_counter(self) -> None:
        self._isa_reg_counter = 1

    def alloc_isa_reg(self, reset: bool = False) -> int:
        if reset:
            self._isa_reg_counter = 1
        if self._isa_reg_counter > 15:
            raise ValueError("Exceeded available ISA registers (max 15)")
        reg = self._isa_reg_counter
        self._isa_reg_counter += 1
        return reg

    def clear_inst_id(self) -> None:
        self._inst_id = 0

    def get_arg_max_index(self) -> int:
        return self.read_reg32(UE_ARGMAX_INDEX)

    def isa_add_set_core(self, dst_reg_idx: int, immediate_value: int, timeout_s: float = 10.0) -> None:
        """Set ISA register to immediate value."""
        isa_set_register(self, dst_reg_idx, immediate_value, timeout_s)

    def load_snapshot(self) -> bool:
        """Load params DRAM from snapshot bin + restore all address metadata. Returns True if loaded."""
        bin_dir = os.path.join(self.script_dir, "smolvlm2_bin")
        bin_path = os.path.join(bin_dir, "params.bin")
        meta_path = os.path.join(bin_dir, "params.json")
        if not os.path.exists(bin_path) or not os.path.exists(meta_path):
            return False
        with open(meta_path) as f:
            meta = json.load(f)
        total = meta["params_size"]
        CHUNK = 1 * 1024 * 1024
        with open(bin_path, "rb") as f:
            offset = 0
            while offset < total:
                data = f.read(min(CHUNK, total - offset))
                self.dma_write(DMA_DEVICE_H2C, self._params_dram_base + offset, data, len(data))
                offset += len(data)
        self.allocate_params_dram(total)
        self.allocate_tensor_dram(meta["tensor_size"])
        for key, val in meta.items():
            if key not in ("params_size", "tensor_size"):
                setattr(self, key, val)
        model_dir = os.path.join(self.script_dir, "smolvlm2_bin", "SmolVLM2-500M-Video-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        kv_zeros = torch.zeros(self.NUM_LAYERS * self.NUM_KV_HEADS * self.max_seq_len * self.HEAD_DIM, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_K_DRAM, kv_zeros)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, kv_zeros)
        _original_print(f"  Snapshot loaded: {total / 1024**2:.1f} MB")
        return True

    def _load_bin(self, bin_path: str) -> int:
        """Load a program bin file into program DRAM. Returns DRAM address."""
        with open(bin_path, "rb") as f:
            data = f.read()
        addr = self.get_program_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, addr, data, len(data))
        self.allocate_program_dram(len(data))
        print(f"    Loaded {len(data)} bytes from {os.path.basename(bin_path)}")
        return addr

    def load_encoder(self) -> int:
        """Load pre-compiled encoder from bin. Reproduces compile-time DRAM allocations."""
        N = 768  # vision hidden size
        N_HEADS = 12
        bpe = 2
        zeros = torch.zeros(N, dtype=torch.bfloat16)
        identity = torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16)
        identity_size = UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe

        num_vis_layers = self._num_vis_layers
        for _ in range(num_vis_layers):
            # LN1 zeros
            addr = self.get_params_dram_addr()
            self.allocate_params_dram(N * bpe)
            self.dma_write(DMA_DEVICE_H2C, addr, zeros, N * bpe)
            # 12× flash_attention_core identity matrices
            for _ in range(N_HEADS):
                addr = self.get_params_dram_addr()
                self.allocate_params_dram(identity_size)
                self.dma_write(DMA_DEVICE_H2C, addr, identity, identity_size)
            # LN2 post-add zeros
            addr = self.get_params_dram_addr()
            self.allocate_params_dram(N * bpe)
            self.dma_write(DMA_DEVICE_H2C, addr, zeros, N * bpe)
        # Final post-layernorm zeros
        addr = self.get_params_dram_addr()
        self.allocate_params_dram(N * bpe)
        self.dma_write(DMA_DEVICE_H2C, addr, zeros, N * bpe)

        return self._load_bin(os.path.join(self.script_dir, "smolvlm2_bin", "encoder_program.bin"))

    def load_prefill(self, seq_len: int) -> None:
        """Load pre-compiled prefill raw bytes from bin. Sets up causal mask and scratch."""
        S = ((seq_len + 63) // 64) * 64
        self.prefill_seq_len = seq_len
        self._prefill_padded = S
        bpe = 2
        # Causal mask for the padded size
        causal = torch.full((S, S), -1e38, dtype=torch.bfloat16)
        causal = torch.triu(causal, diagonal=1)
        causal[:, seq_len:] = -1e38
        causal[seq_len:, :] = -1e38
        causal[seq_len:, 0] = 0.0
        self.dma_write(DMA_DEVICE_H2C, self.CAUSAL_MASK_DRAM, causal.flatten(), S * S * bpe)
        # Load prefill raw bytes (no halt)
        bin_path = os.path.join(self.script_dir, "smolvlm2_bin", f"prefill_program_S{S}.bin")
        with open(bin_path, "rb") as f:
            self._prefill_raw = f.read()
        self._halt_raw = generate_halt_raw(self)
        # Allocate scratch for fused program
        max_embed_insts = seq_len * 2 * 32
        max_merge_insts = IMAGE_SEQ_LEN * 2 * 32
        worst_case = max_embed_insts + max_merge_insts + len(self._prefill_raw) + len(self._halt_raw)
        self._prefill_scratch_addr = self.get_program_dram_addr()
        self.allocate_program_dram(worst_case)
        print(f"    Loaded prefill raw ({len(self._prefill_raw)} bytes) + scratch ({worst_case} bytes)")

    def load_decoder(self) -> None:
        """Load pre-compiled decoder from bin (embed programs + bucket programs)."""
        bin_dir = os.path.join(self.script_dir, "smolvlm2_bin")
        meta_path = os.path.join(bin_dir, "decoder_program.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self._kv_len_buckets = meta["kv_len_buckets"]
        # Load bucket programs
        bin_path = os.path.join(bin_dir, "decoder_program.bin")
        with open(bin_path, "rb") as f:
            bucket_all = f.read()
        self._decode_bucket_raw = []
        for bm in meta["bucket_meta"]:
            self._decode_bucket_raw.append(bucket_all[bm["offset"]:bm["offset"] + bm["size"]])
        # Load embed programs (stride-indexed)
        embed_bin_path = os.path.join(bin_dir, "decoder_embed.bin")
        with open(embed_bin_path, "rb") as f:
            embed_all = f.read()
        stride = meta["embed_stride"]
        count = meta["embed_count"]
        self._decode_embed_raw = [embed_all[i * stride:(i + 1) * stride] for i in range(count)]
        # Allocate scratch for fused decode program: reg_set (64B) + embed + largest bucket
        max_embed = max(len(r) for r in self._decode_embed_raw)
        max_bucket = max(len(r) for r in self._decode_bucket_raw)
        worst_case = 64 + max_embed + max_bucket  # 64 bytes for 2 x 32-byte ADD_SET
        self._decode_scratch_addr = self.get_program_dram_addr()
        self.allocate_program_dram(worst_case)
        print(f"    Loaded decoder: {len(self._decode_bucket_raw)} buckets, "
              f"{len(self._decode_embed_raw)} embed programs, scratch {worst_case} bytes")

    # --- Run ---
    def run_encoder(self, encoder_addr: int, pixel_values) -> None:
        """Run vision encoder. Output lands in VIS_CONNECTOR_DRAM."""
        pixels_bf16 = pixel_values.to(torch.bfloat16).contiguous().flatten()
        self.dma_to_accelerator_memory(self.VIS_PIXEL_IN_DRAM, pixels_bf16)
        # Encoder FLOPs: patch proj + pos add + N layers + post LN + connector
        VS, VH, VD, VN, VI = 1024, 768, 64, 12, 3072
        n_vis = 12  # SigLIP-base vision encoder always has 12 layers
        attn_per_head = VS * VD + 2 * VS * VD * VS + VS * VS * 5 + 2 * VS * VS * VD
        per_layer = (7 * VS * VH + 3 * (2 * VS * VH * VH + VS * VH)
            + VN * attn_per_head + 2 * VS * VH * VH + VS * VH
            + 8 * VS * VH + 2 * VS * VH * VI + VS * VI
            + 2 * VS * VI * VH + VS * VH + VS * VH)
        enc_flops = (2 * VS * VH * VH + VS * VH + VS * VH
            + n_vis * per_layer + 7 * VS * VH
            + 2 * 64 * 12288 * self.HIDDEN_SIZE)
        self.program_execute(encoder_addr, timeout=30.0, total_flops=enc_flops)

    def run_prefill(self, token_ids, has_image: bool = False, total_flops: int = None) -> int:
        """On-device prefill: embed gather + vision merge + decoder layers. Returns argmax."""
        seq_len = len(token_ids)
        self.seq_len = seq_len
        S = self._prefill_padded
        H = self.HIDDEN_SIZE
        bpe = 2
        embed_row_bytes = H * bpe

        # Epsilon-fill padding rows to prevent RMS norm NaN on zero rows
        if S > seq_len:
            pad_rows = S - seq_len
            epsilon_fill = torch.full((pad_rows * H,), 1e-6, dtype=torch.bfloat16)
            pad_offset = seq_len * embed_row_bytes
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM + pad_offset, epsilon_fill)

        # Generate on-device embedding gather instructions
        self.start_capture()
        for t in range(seq_len):
            token_id = token_ids[t]
            src_addr = self.embed_addr + token_id * embed_row_bytes
            dst_addr = self.LAYER0_INPUT_DRAM + t * embed_row_bytes
            self.accelerator_memory_to_sram(
                accelerator_dram_address=src_addr,
                sram_address=0x00000,
                element_size=H)
            self.sram_to_accelerator_memory(
                sram_address=0x00000,
                accelerator_dram_address=dst_addr,
                element_size=H)

        # Generate on-device vision merge instructions (overwrite image token positions)
        if has_image:
            img_positions = [i for i, t in enumerate(token_ids) if t == IMAGE_TOKEN_ID]
            if len(img_positions) > 0:
                assert len(img_positions) == IMAGE_SEQ_LEN, \
                    f"Expected {IMAGE_SEQ_LEN} image tokens, got {len(img_positions)}"
                for i, pos in enumerate(img_positions):
                    src_addr = self.VIS_CONNECTOR_DRAM + i * embed_row_bytes
                    dst_addr = self.LAYER0_INPUT_DRAM + pos * embed_row_bytes
                    self.accelerator_memory_to_sram(
                        accelerator_dram_address=src_addr,
                        sram_address=0x00000,
                        element_size=H)
                    self.sram_to_accelerator_memory(
                        sram_address=0x00000,
                        accelerator_dram_address=dst_addr,
                        element_size=H)

        dynamic_raw = capture_to_raw(self)

        # Fuse: dynamic embed/merge + prefill layers + halt → single dispatch
        fused = bytearray()
        fused.extend(dynamic_raw)
        fused.extend(self._prefill_raw)
        fused.extend(self._halt_raw)

        # Write fused program to scratch and dispatch
        self.dma_write(DMA_DEVICE_H2C, self._prefill_scratch_addr, bytes(fused), len(fused))
        self.start_execute_from_dram(self._prefill_scratch_addr)
        self.wait_queue(30.0)
        if total_flops is not None:
            self._last_hw_gflops, _ = self.report_flop_rate_gflops(total_flops)
            self._last_total_flops = total_flops
        else:
            self._last_hw_gflops = None
            self._last_total_flops = None
        return self.get_arg_max_index()

    def run_decoder(self, token_id: int, max_new_tokens: int = 512) -> list:
        """Auto-regressive decode loop. Returns generated token IDs."""
        bpe = 2
        global _SILENT_MODE
        generated = []
        H, D, I = self.HIDDEN_SIZE, self.HEAD_DIM, self.INTERMEDIATE_SIZE
        QH, KVH, G = self.NUM_HEADS, self.NUM_KV_HEADS, self.GROUP_SIZE
        self._decode_total_flops = 0
        self._decode_total_hw_ns = 0
        # All embed programs are the same size (2 DMA instructions)
        embed_size = len(self._decode_embed_raw[0])
        REG_SET_SIZE = 64  # 2 x 32-byte ADD_SET instructions
        bucket_offset = REG_SET_SIZE + embed_size  # bucket starts after reg_set + embed
        last_bucket_idx = -1  # track cached bucket to avoid re-DMA
        while len(generated) < max_new_tokens and self.seq_len < self.max_seq_len:
            _SILENT_MODE = True
            self.seq_len += 1
            pos = self.seq_len - 1
            # Decode attention bias: [GROUP_SIZE, max_seq_len] with proper stride
            # Each row must be max_seq_len wide to match bias_row_stride in attention
            aligned_kv = ((self.seq_len + 63) // 64) * 64
            bias = torch.full((self.GROUP_SIZE * self.max_seq_len,), -1e38, dtype=torch.bfloat16)
            for g in range(self.GROUP_SIZE):
                bias[g * self.max_seq_len:g * self.max_seq_len + self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.DECODE_BIAS_DRAM, bias)
            # Fuse: reg_set + embed + bucket → single dispatch (no separate isa_set_register)
            prog_idx = min((self.seq_len - 1) // 64, len(self._decode_bucket_raw) - 1)
            if prog_idx != last_bucket_idx:
                bucket_raw = self._decode_bucket_raw[prog_idx]
                self.dma_write(DMA_DEVICE_H2C,
                               self._decode_scratch_addr + bucket_offset,
                               bucket_raw, len(bucket_raw))
                last_bucket_idx = prog_idx
            # Build fused reg_set + embed bytes (64 + 64 = 128 bytes)
            reg_set_bytes = bytearray()
            reg_set_bytes.extend(_make_add_set_bytes(self.V_CACHE_SIZE_REG, ue_35bit_addr_shifter(pos * self.k_size)))
            reg_set_bytes.extend(_make_add_set_bytes(self.ROPE_SIZE_REG, ue_35bit_addr_shifter(pos * self.HEAD_DIM * bpe)))
            embed_raw = self._decode_embed_raw[token_id]
            fused_head = bytes(reg_set_bytes) + embed_raw
            # DMA fused reg_set + embed to start of scratch (bucket already cached after)
            self.dma_write(DMA_DEVICE_H2C, self._decode_scratch_addr, fused_head, len(fused_head))
            self.start_execute_from_dram(self._decode_scratch_addr)
            self.wait_queue(10.0)
            # Accumulate decode FLOPs from HW cycle counter
            hw_cycles = self.read_reg32(user_dma_core.UE_LATENCY_COUNT_ADDR)
            self._decode_total_hw_ns += hw_cycles * user_dma_core.CLOCK_CYCLE_TIME_NS
            per_layer = (4 * H + 2 * H * (H + 2 * KVH * D)
                + (QH + KVH) * D * 4
                + KVH * (D * aligned_kv + G * (D + 2 * D * aligned_kv
                    + aligned_kv + aligned_kv * 5 + 2 * aligned_kv * D))
                + 2 * H * H + 5 * H + 4 * H * I + I + 2 * I * H + H)
            self._decode_total_flops += self.NUM_LAYERS * per_layer + 4 * H + 2 * H * self.VOCAB_SIZE
            token_id = self.get_arg_max_index()
            generated.append(token_id)
            _SILENT_MODE = False
            if token_id in _SMOLVLM2_CFG["model"]["stop_token_ids"]:  # <|endoftext|>, BOS, PAD, <end_of_utterance>
                _original_print(f"\nStop token {token_id} reached.")
                break
            _original_print(self.tokenizer.decode([token_id]), end="", flush=True)
        _SILENT_MODE = False
        self._decode_hw_gflops = (self._decode_total_flops / self._decode_total_hw_ns
                                  if self._decode_total_hw_ns > 0 else 0)
        return generated

    def program_execute(self, program_addr: int = user_dma_core.DRAM_INSTRUCTION_ADDR,
                        timeout: float = 10.0, total_flops: int = None) -> None:
        """Execute compiled program from DRAM."""
        self.start_execute_from_dram(program_addr)
        self.wait_queue(timeout)
        if total_flops is not None:
            self._last_hw_gflops = self.report_flop_rate_gflops(total_flops)
            self._last_total_flops = total_flops
        else:
            self._last_hw_gflops = None
            self._last_total_flops = None

# =============================================================================
# Image processing + input construction
# =============================================================================
IMAGE_TOKEN_ID = _SMOLVLM2_CFG["special_tokens"]["image_token_id"]
FAKE_TOKEN_ID = _SMOLVLM2_CFG["special_tokens"]["fake_token_id"]
GLOBAL_IMG_TOKEN_ID = _SMOLVLM2_CFG["special_tokens"]["global_img_token_id"]
IMAGE_SEQ_LEN = _SMOLVLM2_CFG["model"]["image_seq_len"]

def process_image(image_path: str, size: int = 512) -> torch.Tensor:
    """Load, resize, normalize image → [3, size, size] bf16."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # normalize to [-1, 1]
    return torch.from_numpy(arr).permute(2, 0, 1).to(torch.bfloat16)

def build_input_ids(tokenizer, prompt: str, has_image: bool = True) -> list:
    """Build token ID list with chat template and optional image token expansion."""
    if has_image:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    tokens = tokenizer.encode(prompt_text)
    if not has_image:
        return tokens
    # Expand single <image> token → <fake><global-img><image>×64<fake>
    expanded = []
    for t in tokens:
        if t == IMAGE_TOKEN_ID:
            expanded += [FAKE_TOKEN_ID, GLOBAL_IMG_TOKEN_ID] + [IMAGE_TOKEN_ID] * IMAGE_SEQ_LEN + [FAKE_TOKEN_ID]
        else:
            expanded.append(t)
    return expanded

# =============================================================================
# Main
# =============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="SmolVLM2-500M inference from pre-compiled bins")
    _d = _SMOLVLM2_CFG["defaults"]
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument("--prompt", type=str, default=_d["prompt"])
    parser.add_argument("--image", type=str, default=os.path.join(_root, _d["image"]))
    parser.add_argument("--dev", type=str, default=_d["dev"])
    parser.add_argument("--cycle", type=float, default=_d["cycle_ns"])
    parser.add_argument("--max-seq", type=int, default=_d["max_seq"])
    parser.add_argument("--vision-fp4", action="store_true")
    args = parser.parse_args()

    global _SILENT_MODE
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bin_dir = os.path.join(script_dir, "smolvlm2_bin")
    has_image = args.image is not None

    # Check all static bins before any hardware init
    def _missing_static_bins():
        missing = []
        for name in ("params.bin", "params.json", "decoder_program.bin"):
            if not os.path.exists(os.path.join(bin_dir, name)):
                missing.append(name)
        if has_image and not os.path.exists(os.path.join(bin_dir, "encoder_program.bin")):
            missing.append("encoder_program.bin")
        return missing

    early_missing = _missing_static_bins()
    if early_missing:
        _original_print("Missing bin files (run smolvlm2_test.py first to generate):")
        for f in early_missing:
            _original_print(f"  {os.path.join(bin_dir, f)}")
        return

    set_dma_device(args.dev)
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle
    _SILENT_MODE = True

    ue = SmolVLM2_UnifiedEngine(script_dir=script_dir, vision_bf16=not args.vision_fp4)
    init_hang_prevention(ue)
    ue.load_snapshot()

    token_ids = build_input_ids(ue.tokenizer, args.prompt, has_image=has_image)
    seq_len = len(token_ids)

    # Check prefill bin now that we know seq_len
    S = ((seq_len + 63) // 64) * 64
    prefill_bin = f"prefill_program_S{S}.bin"
    if not os.path.exists(os.path.join(bin_dir, prefill_bin)):
        _original_print(f"Missing bin file (run smolvlm2_test.py first to compile):")
        _original_print(f"  {os.path.join(bin_dir, prefill_bin)}")
        return

    if has_image:
        _original_print(f"Image: {os.path.abspath(args.image)}")
    _original_print(f"Prompt: {args.prompt!r} ({seq_len} tokens, image={'yes' if has_image else 'no'})")

    # Load programs from bin
    timer = time.perf_counter()
    if has_image:
        enc_addr = ue.load_encoder()
    ue.load_prefill(seq_len=seq_len)
    ue.load_decoder()
    _original_print(f"  Loaded from bin in {time.perf_counter() - timer:.2f}s")

    # Vision encoder
    if has_image:
        timer = time.perf_counter()
        pixel_values = process_image(args.image)
        stop_enc = threading.Event()
        def _enc_progress():
            while not stop_enc.wait(1.0):
                _original_print(f"\r  Vision encoder ({time.perf_counter() - timer:.0f}s)", end="", flush=True)
        threading.Thread(target=_enc_progress, daemon=True).start()
        ue.run_encoder(enc_addr, pixel_values)
        stop_enc.set()
        _original_print(f"\r  Vision encoder ({time.perf_counter() - timer:.0f}s) done")

    # Prefill
    timer = time.perf_counter()
    padded_seq = ((seq_len + 63) // 64) * 64
    H, D, I = ue.HIDDEN_SIZE, ue.HEAD_DIM, ue.INTERMEDIATE_SIZE
    KVH, QH, G = ue.NUM_KV_HEADS, ue.NUM_HEADS, ue.GROUP_SIZE
    S = padded_seq
    attn_per_group = (D * S + G * (S * D + 2 * S * D * S + S * S + S * S * 5 + 2 * S * S * D))
    per_layer = (4 * S * H + 2 * S * H * H + 2 * 2 * S * H * KVH * D
        + (QH + KVH) * S * D * 4 + KVH * attn_per_group
        + 2 * S * H * H + 5 * S * H + 2 * (2 * S * H * I) + S * I + 2 * S * I * H + S * H)
    prefill_flops = ue.NUM_LAYERS * per_layer + 4 * S * H + 2 * 1 * H * ue.VOCAB_SIZE
    stop_pf = threading.Event()
    def _pf_progress():
        while not stop_pf.wait(1.0):
            _original_print(f"\r  Prefill ({time.perf_counter() - timer:.0f}s)", end="", flush=True)
    threading.Thread(target=_pf_progress, daemon=True).start()
    hw_token = ue.run_prefill(token_ids, has_image=has_image, total_flops=prefill_flops)
    stop_pf.set()
    prefill_time = time.perf_counter() - timer
    _original_print(f"\r  Prefill ({prefill_time:.0f}s) done")

    # Zero stale KV cache positions from padding
    if padded_seq > seq_len:
        stale_size = (padded_seq - seq_len) * ue.HEAD_DIM
        stale_zeros = torch.zeros(stale_size, dtype=torch.bfloat16)
        for layer in range(ue.NUM_LAYERS):
            for h in range(ue.NUM_KV_HEADS):
                k_stale = ue.LAYER0_K_DRAM + layer * ue.KV_LAYER_STRIDE + h * ue.KV_HEAD_STRIDE + seq_len * ue.HEAD_DIM * 2
                v_stale = ue.LAYER0_V_DRAM + layer * ue.KV_LAYER_STRIDE + h * ue.KV_HEAD_STRIDE + seq_len * ue.HEAD_DIM * 2
                ue.dma_to_accelerator_memory(k_stale, stale_zeros)
                ue.dma_to_accelerator_memory(v_stale, stale_zeros)

    # Decode
    max_new = args.max_seq - seq_len
    _original_print(f"\n--- Starting decoder ---")
    _original_print(ue.tokenizer.decode([hw_token]), end="", flush=True)
    decode_timer = time.perf_counter()
    hw_tokens = ue.run_decoder(hw_token, max_new_tokens=max_new)
    decode_time = time.perf_counter() - decode_timer
    _original_print(f"\nDecoder done in {prefill_time + decode_time:.2f}s total, {len(hw_tokens)} tokens.")


if __name__ == "__main__":
    main()
