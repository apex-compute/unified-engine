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
    GPR_SEQ_LEN_REG    = _cfg["fixed_isa_regs"]["GPR_SEQ_LEN_REG"]
    GPR_Q_SEQ_LEN_REG  = _cfg["fixed_isa_regs"]["GPR_Q_SEQ_LEN_REG"]
    GPR_BUCKET_IDX_REG = _cfg["fixed_isa_regs"]["GPR_BUCKET_IDX_REG"]
    PREFILL_MAX_SEQ_LEN = _cfg["model"]["prefill_max_seq_len"]

    def __init__(self, script_dir: str = None):
        self.script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
        self._cfg = _SMOLVLM2_CFG
        # Fixed precision scheme: bf16 vision + q4 LM → the bf16 DRAM layout.
        dl = self._cfg["dram_layout"]["bf16"]
        super().__init__(
            params_dram_base=int(dl["params_dram_base"], 16),
            tensor_dram_base=int(dl["tensor_dram_base"], 16),
            program_dram_base=int(dl["program_dram_base"], 16),
        )
        self._isa_reg_counter = 1
        self.gpr_seq_len = self.GPR_SEQ_LEN_REG       # primed to S in run_prefill_v2 / seq pos in decode
        self.gpr_q_seq_len = self.GPR_Q_SEQ_LEN_REG   # primed to S*GROUP_SIZE in run_prefill_v2
        self.gpr_bucket_idx = self.GPR_BUCKET_IDX_REG  # primed to flash/decoder bucket selector
        self.vision_bf16 = True
        self.lm_bf16 = False

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
        bin_path = os.path.join(bin_dir, "weights.bin")
        meta_path = os.path.join(bin_dir, "weights.json")
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

    def load_all(self) -> dict:
        """Load the unified single instruction bin (encoder + decoder + prefill_v2) — mirror of
        smolvlm2_test.compile_all's cache-hit path. DMAs each segment to its recorded absolute address
        and sets the program/preamble addrs the run_* methods read. Must be called AFTER load_snapshot
        (which restores params, including the shared vis_zeros LN buffer — Trick 9, no replay needed)."""
        bin_dir = os.path.join(self.script_dir, "smolvlm2_bin")
        with open(os.path.join(bin_dir, "instructions.json")) as f:
            meta = json.load(f)
        with open(os.path.join(bin_dir, "instructions.bin"), "rb") as f:
            raw = f.read()
        for seg in meta["segments"]:
            b = raw[seg["off"]:seg["off"] + seg["size"]]
            self.dma_write(DMA_DEVICE_H2C, seg["addr"], b, len(b))
        self._vis_program_addr = meta["encoder_addr"]
        self._decoder_program_addr = meta["decoder_addr"]
        self._decoder_preamble_addr = meta["decoder_preamble"]
        self._decoder_num_buckets = meta["decoder_num_buckets"]
        self._prefill_v2_addr = meta["prefill_addr"]
        self._prefill_v2_preamble_addr = meta["prefill_preamble"]
        self._prefill_v2_postamble_addr = meta["prefill_postamble"]
        self._prefill_v2_final_buf = meta["prefill_final_buf"]
        self._next_program_dram_addr = meta["end_addr"]
        print(f"    Loaded unified instruction bin ({len(raw)/1024/1024:.1f} MB, "
              f"{len(meta['segments'])} segments)")
        return meta

    def load_encoder(self) -> int:
        """Load pre-compiled encoder from bin. Reproduces compile-time DRAM allocations."""
        N = 768  # vision hidden size
        N_HEADS = 12
        bpe = 2
        zeros = torch.zeros(N, dtype=torch.bfloat16)
        identity = torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16)
        identity_size = UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe

        num_vis_layers = self._num_vis_layers
        # §7: the shared flash body uses the pre-seeded identity_addr and allocates NO params, so the
        # old per-layer "12× flash identity matrices" replay is gone. compile_encoder's only lazy
        # params are the layer_norm zero buffers: LN1 + LN2 per layer, then one final post-layernorm.
        for _ in range(num_vis_layers):
            for _ in range(2):  # LN1 + LN2 zero buffers
                addr = self.get_params_dram_addr()
                self.allocate_params_dram(N * bpe)
                self.dma_write(DMA_DEVICE_H2C, addr, zeros, N * bpe)
        # Final post-layernorm zeros
        addr = self.get_params_dram_addr()
        self.allocate_params_dram(N * bpe)
        self.dma_write(DMA_DEVICE_H2C, addr, zeros, N * bpe)

        return self._load_bin(os.path.join(self.script_dir, "smolvlm2_bin", "encoder_program.bin"))

    def load_prefill_v2(self) -> None:
        """Load the cached seq-len-agnostic prefill program (dynamic-PBI / §7). Replays
        compile_prefill_v2's exact program-DRAM allocations so the body lands at the address its
        flash absolute-jumps were baked against (must be loaded AFTER the decoder)."""
        PM = self.PREFILL_MAX_SEQ_LEN
        bin_path = os.path.join(self.script_dir, "smolvlm2_bin", "prefill_v2_program.bin")
        with open(bin_path, "rb") as f:
            raw = f.read()
        prefill_addr = self.get_program_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, prefill_addr, raw, len(raw))
        self.allocate_program_dram(len(raw))
        self._prefill_v2_addr = prefill_addr
        self._prefill_v2_preamble_addr = self.get_program_dram_addr()
        self.allocate_program_dram(PM * 2 * 32 + IMAGE_SEQ_LEN * 2 * 32 + 256)
        self._prefill_v2_postamble_addr = self.get_program_dram_addr()
        self.allocate_program_dram(4096)
        self._prefill_v2_final_buf = self.LAYER0_INPUT_DRAM if self.NUM_LAYERS % 2 == 0 else self.LAYER0_OUTPUT_DRAM
        print(f"    Loaded prefill v2 @0x{prefill_addr:X}: {len(raw)} bytes")

    def load_decoder(self) -> None:
        """Load pre-compiled single decoder program (dynamic-PBI / §7, runtime bucket dispatch)."""
        bin_dir = os.path.join(self.script_dir, "smolvlm2_bin")
        meta_path = os.path.join(bin_dir, "decoder_program.json")
        with open(meta_path) as f:
            meta = json.load(f)
        if "num_buckets" not in meta:
            raise RuntimeError(
                "decoder_program.json is from the old bucket format; "
                "delete decoder_program.bin and decoder_program.json and recompile via smolvlm2_test.py"
            )
        bin_path = os.path.join(bin_dir, "decoder_program.bin")
        with open(bin_path, "rb") as f:
            raw = f.read()
        self._decoder_program_addr = self.get_program_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, self._decoder_program_addr, raw, len(raw))
        self.allocate_program_dram(len(raw))
        self._decoder_preamble_addr = self.get_program_dram_addr()
        self.allocate_program_dram(256)
        self._decoder_num_buckets = meta["num_buckets"]
        print(f"    Loaded decoder @0x{self._decoder_program_addr:X}: {len(raw)} bytes, {self._decoder_num_buckets} buckets")

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

    def run_prefill_v2(self, token_ids, has_image: bool = False, total_flops: int = None) -> int:
        """Run the seq-len-agnostic prefill (dynamic-PBI / §7): host prep (epsilon pad + block bias),
        a preamble that does on-device embed/merge + primes gpr_bucket_idx + jumps into the cached
        prefill, then a postamble that does final-norm + LM head on the runtime last token. Mirrors
        smolvlm2_test.run_prefill_v2 exactly (runtime-only; no compilation). Returns argmax."""
        from user_dma_core import TYPE
        seq_len = len(token_ids)
        self.seq_len = seq_len
        self._prefill_token_ids = list(token_ids)  # seeds the repetition-penalty window in run_decoder
        H, D, G, bpe = self.HIDDEN_SIZE, self.HEAD_DIM, self.GROUP_SIZE, 2
        PM = self.PREFILL_MAX_SEQ_LEN
        assert seq_len <= PM, f"prompt {seq_len} exceeds PREFILL_MAX_SEQ_LEN={PM}"
        qmax = self.PREFILL_QMAX
        qS = seq_len * G
        aligned_q = ((qS + 63) // 64) * 64
        bucket_idx = aligned_q // UE_VECTOR_SIZE
        embed_row_bytes = H * bpe

        # 1. epsilon-fill INPUT padding rows [seq_len:PM] (non-attention ops run static PM rows)
        if PM > seq_len:
            epsilon = torch.full(((PM - seq_len) * H,), 1e-6, dtype=torch.bfloat16)
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM + seq_len * embed_row_bytes, epsilon)
        # 2. fill flash Q/K/V tails finite so masked-out padded rows can't produce NaN in softmax
        eps_flash = torch.full((qmax * D,), 1e-6, dtype=torch.bfloat16)
        for buf in (self.FLASH_Q_DRAM, self.FLASH_K_DRAM, self.FLASH_V_DRAM):
            self.dma_to_accelerator_memory(buf, eps_flash)
        # 3. block-diagonal causal bias [aligned_q, aligned_q]: allow (t,g)->(t',g') iff g==g' & t'<=t
        rq = torch.arange(aligned_q).unsqueeze(1)
        rk = torch.arange(aligned_q).unsqueeze(0)
        allow = (rq < qS) & (rk < qS) & (rq % G == rk % G) & (rk // G <= rq // G)
        bias = torch.where(allow, torch.tensor(0.0), torch.tensor(-1e38)).to(torch.bfloat16)
        bias[torch.arange(aligned_q) >= qS, 0] = 0.0  # padded query rows attend row 0 (discarded)
        self.dma_to_accelerator_memory(self.FLASH_BIAS_DRAM, bias)

        # 4. Preamble: on-device embed gather + vision merge + prime gpr_bucket_idx + jump into prefill
        self.clear_inst_id()
        self.start_capture()
        for t in range(seq_len):
            src = self.embed_addr + token_ids[t] * embed_row_bytes
            self.accelerator_memory_to_sram(src, 0x00000, H)
            self.sram_to_accelerator_memory(0x00000, self.LAYER0_INPUT_DRAM + t * embed_row_bytes, H)
        if has_image:
            img_positions = [i for i, t in enumerate(token_ids) if t == IMAGE_TOKEN_ID]
            if len(img_positions) > 0:
                assert len(img_positions) == IMAGE_SEQ_LEN, \
                    f"Expected {IMAGE_SEQ_LEN} image tokens, got {len(img_positions)}"
                for i, pos in enumerate(img_positions):
                    self.accelerator_memory_to_sram(self.VIS_CONNECTOR_DRAM + i * embed_row_bytes, 0x00000, H)
                    self.sram_to_accelerator_memory(0x00000, self.LAYER0_INPUT_DRAM + pos * embed_row_bytes, H)
        self.generate_instruction_add_set(self.gpr_bucket_idx, bucket_idx)
        self.generate_instruction_jump_abs(ue_35bit_addr_shifter(self._prefill_v2_addr))
        self.stop_capture()
        self.write_captured_instructions_to_dram(self._prefill_v2_preamble_addr)
        self.clear_capture_buffer()
        self.start_execute_from_dram(self._prefill_v2_preamble_addr)
        self.wait_queue(180.0)

        # 5. Postamble: final norm over ALL tokens + LM head on the last real token.
        last_off = (seq_len - 1) * H * bpe
        self.clear_inst_id()
        self.start_capture()
        self.rms_norm_core_dram(M=seq_len, N=H, A_DRAM_ADDR=self._prefill_v2_final_buf,
            OUTPUT_DRAM_ADDR=self.FINAL_NORM_DRAM, GAMMA_DRAM_ADDR=self.final_norm_addr)
        last_norm = self.FINAL_NORM_DRAM + last_off
        if self.lm_bf16:
            self.matmat_mul_core(M=1, K=H, N=self.VOCAB_SIZE, A_DRAM_ADDR=last_norm,
                B_DRAM_ADDR=self.lm_head_weight, OUTPUT_DRAM_ADDR=self.LOGITS_DRAM)
        else:
            self.quantized_matmat_core(M=1, K=H, N=self.VOCAB_SIZE, A_DRAM_ADDR=last_norm,
                B_DRAM_ADDR=self.lm_head_data, OUTPUT_DRAM_ADDR=self.LOGITS_DRAM,
                SCALE_DRAM_ADDR=self.lm_head_scale, data_type=TYPE.IF4)
        self.generate_instruction_halt()
        self.stop_capture()
        self.write_captured_instructions_to_dram(self._prefill_v2_postamble_addr)
        self.clear_capture_buffer()
        self.start_execute_from_dram(self._prefill_v2_postamble_addr)
        self.wait_queue(30.0)
        if total_flops is not None:
            self._last_hw_gflops, _ = self.report_flop_rate_gflops(total_flops)
            self._last_total_flops = total_flops
        else:
            self._last_hw_gflops = None
            self._last_total_flops = None
        return self.get_arg_max_index()

    def _structural_token_ids(self):
        """Token ids never repetition-penalized: punctuation, whitespace, and special tokens. Cached."""
        cached = getattr(self, "_struct_ids_cache", None)
        if cached is not None:
            return cached
        import string
        allowed = set(string.punctuation) | set(string.whitespace) | set("—–’‘“”…·•‹›«»¡¿")
        ids = set(int(i) for i in (getattr(self.tokenizer, "all_special_ids", []) or []))
        ids |= set(int(v) for v in _SMOLVLM2_CFG["special_tokens"].values())
        for i in range(self.VOCAB_SIZE):
            s = self.tokenizer.decode([i]).strip()
            if s == "" or all(ch in allowed for ch in s):
                ids.add(i)
        self._struct_ids_cache = ids
        return ids

    def _structural_ids_tensor(self):
        t = getattr(self, "_struct_ids_tensor_cache", None)
        if t is None:
            t = torch.tensor(sorted(self._structural_token_ids()), dtype=torch.long)
            self._struct_ids_tensor_cache = t
        return t

    def _write_penalty_bias(self, prev_tokens) -> None:
        """Per-vocab additive repetition bias → PENALTY_BIAS_DRAM (decode LM-head C term). bias[t] =
        clamp(−alpha·count[t], min=−cap); structural tokens stay 0. HW argmax of (logits+bias) penalizes."""
        vocab = self.VOCAB_SIZE
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
        self.dma_to_accelerator_memory(self.PENALTY_BIAS_DRAM, bias)

    def run_decoder(self, token_id: int, max_new_tokens: int = 512) -> list:
        """Auto-regressive decode loop (dynamic-PBI / §7). Each step writes a tiny preamble that
        primes gpr_seq_len + gpr_bucket_idx and jumps into the single cached decoder program."""
        global _SILENT_MODE
        generated = []
        H = self.HIDDEN_SIZE
        bpe = 2
        embed_row_bytes = H * bpe
        num_buckets = self._decoder_num_buckets
        decoder_program_addr = self._decoder_program_addr
        preamble_addr = self._decoder_preamble_addr

        # On-FPGA repetition penalty (same as test.py): the decode LM-head always adds PENALTY_BIAS_DRAM;
        # zero it up front, then refresh each step past the greedy_until gate from the windowed frequency.
        _fpga_penalty = bool(getattr(self, "fpga_penalty", True))
        _greedy_until = int(getattr(self, "greedy_until", 0))
        self._generated_tokens = list(getattr(self, "_prefill_token_ids", [])) + [token_id]
        self.dma_to_accelerator_memory(self.PENALTY_BIAS_DRAM,
                                       torch.zeros(1, self.VOCAB_SIZE, dtype=torch.bfloat16))
        if _fpga_penalty:
            self._structural_ids_tensor()  # warm the structural-token cache off the first decode token

        _original_print(self.tokenizer.decode([token_id]), end="", flush=True)
        while len(generated) < max_new_tokens and self.seq_len < self.max_seq_len:
            _SILENT_MODE = True
            self.seq_len += 1
            aligned_kv = ((self.seq_len + 63) // 64) * 64
            bucket_idx = min(aligned_kv // UE_VECTOR_SIZE, num_buckets)

            # Decode bias: [GROUP_SIZE, max_seq_len] with zeros at valid KV positions
            bias = torch.full((self.GROUP_SIZE, self.max_seq_len), -1e38, dtype=torch.bfloat16)
            bias[:, :self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.DECODE_BIAS_DRAM, bias)

            # Host-side embed DMA — avoids on-device C2H before decoder
            src_addr = self.embed_addr + token_id * embed_row_bytes
            embed_vec = self.dma_from_accelerator_memory(src_addr, torch.Size([H]))
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embed_vec)

            # Preamble: prime gpr_seq_len + gpr_bucket_idx + jump into decoder
            self.clear_inst_id()
            self.start_capture()
            self.generate_instruction_add_set(self.gpr_seq_len, self.seq_len - 1)
            self.generate_instruction_add_set(self.gpr_bucket_idx, bucket_idx)
            self.generate_instruction_jump_abs(ue_35bit_addr_shifter(decoder_program_addr))
            self.stop_capture()
            self.write_captured_instructions_to_dram(preamble_addr)
            self.clear_capture_buffer()

            if _fpga_penalty and len(generated) >= _greedy_until:
                self._write_penalty_bias(self._generated_tokens)

            self.start_execute_from_dram(preamble_addr)
            self.wait_queue(30.0)

            # HW argmax: read the 4-byte (penalized) argmax-index register the LM-head left.
            token_id = self.get_arg_max_index()

            generated.append(token_id)
            self._generated_tokens.append(token_id)
            _SILENT_MODE = False
            if token_id in _SMOLVLM2_CFG["model"]["stop_token_ids"]:
                _original_print(f"\nStop token {token_id} reached.")
                break
            _original_print(self.tokenizer.decode([token_id]), end="", flush=True)
        _SILENT_MODE = False
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
def _clock_ns_default_for_device(device: str) -> float:
    """Return default clock period (ns) for FPGA type — mirrors user_hw_test.py."""
    if device == "kintex7":                       return 5.1594
    if device in ("rk", "puzhi"):                 return 3.0
    if device in ("bittware", "bittware_256"):     return 3.3333
    if device == "alveo":                          return 4.0
    return 10.0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SmolVLM2-500M inference from pre-compiled bins")
    _d = _SMOLVLM2_CFG["defaults"]
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _default_image = os.path.join(_root, _d["image"])
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt. Default: a text question (LM-only) or 'Describe this image.' (VLM).")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to an image. Providing it enables VLM. Default: none (LM-only text).")
    parser.add_argument("--vision-enable", action="store_true",
                        help="Enable the vision encoder (VLM) using --image, or the default sample image.")
    parser.add_argument("--dev", type=str, default=_d["dev"])
    parser.add_argument("--cycle", type=float, default=None, help='Clock cycle time in ns. Overrides --device default.')
    parser.add_argument("--device", type=str, default='kintex7', help='FPGA board profile (kintex7, rk, puzhi, bittware, bittware_256, alveo).')
    parser.add_argument("--max-seq", type=int, default=_d["max_seq"])
    parser.add_argument("--pure-greedy", action="store_true", help="Disable the on-FPGA repetition penalty.")
    parser.add_argument("--pen-alpha", type=float, default=1.0, help="Penalty strength α. Default 1.0.")
    parser.add_argument("--pen-cap", type=float, default=20.0, help="Max penalty magnitude. Default 20.0.")
    parser.add_argument("--rep-window", type=int, default=256, help="Penalty token window. Default 256.")
    parser.add_argument("--greedy-until", type=int, default=None,
                        help="Greedy for the first N tokens, then penalty. Default: 0 (LM) / 64 (VLM).")
    args = parser.parse_args()

    global _SILENT_MODE
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bin_dir = os.path.join(script_dir, "smolvlm2_bin")

    # Check all static bins before any hardware init
    def _missing_static_bins():
        missing = []
        for name in ("weights.bin", "weights.json", "instructions.bin", "instructions.json"):
            if not os.path.exists(os.path.join(bin_dir, name)):
                missing.append(name)
        return missing

    early_missing = _missing_static_bins()
    if early_missing:
        _original_print("Missing bin files (run smolvlm2_test.py first to generate):")
        for f in early_missing:
            _original_print(f"  {os.path.join(bin_dir, f)}")
        return

    # VLM is opt-in at runtime: --vision-enable or --image. The bin is always the full VLM bin, so the
    # encoder is available either way; this only decides whether vision RUNS. Default = LM-only text.
    vision_on = bool(args.vision_enable) or (args.image is not None
                                             and str(args.image).strip().lower() not in ("none", ""))
    if vision_on and (args.image is None or str(args.image).strip().lower() in ("none", "")):
        args.image = _default_image
    if args.prompt is None:           # mode-appropriate default prompt (a text Q for LM-only)
        args.prompt = _d["vlm_prompt"] if vision_on else _d["lm_prompt"]
    has_image = vision_on

    set_dma_device(args.dev)
    axi_width_bits = 512 if args.device in ("bittware", "rk") else 256
    os.environ["UE_AXI_DATA_WIDTH_BITS"] = str(axi_width_bits)
    user_dma_core.UE_AXI_DATA_WIDTH_BITS = axi_width_bits
    clock = args.cycle if args.cycle is not None else _clock_ns_default_for_device(args.device)
    user_dma_core.CLOCK_CYCLE_TIME_NS = clock
    user_dma_core.UE_PEAK_GFLOPS = 0.128 / clock
    _original_print(f"FPGA profile: device={args.device}, clock={clock:.4f} ns, UE_AXI_DATA_WIDTH_BITS={axi_width_bits}")
    _SILENT_MODE = True

    ue = SmolVLM2_UnifiedEngine(script_dir=script_dir)
    init_hang_prevention(ue)
    ue.load_snapshot()

    token_ids = build_input_ids(ue.tokenizer, args.prompt, has_image=has_image)
    seq_len = len(token_ids)

    if has_image:
        _original_print(f"Image: {os.path.abspath(args.image)}")
    _original_print(f"Prompt: {args.prompt!r} ({seq_len} tokens, image={'yes' if has_image else 'no'})")

    # Load the ONE unified instruction bin (encoder + decoder + prefill_v2).
    timer = time.perf_counter()
    uni_meta = ue.load_all()
    enc_addr = uni_meta["encoder_addr"]
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
    hw_token = ue.run_prefill_v2(token_ids, has_image=has_image, total_flops=prefill_flops)
    stop_pf.set()
    prefill_time = time.perf_counter() - timer
    _original_print(f"\r  Prefill ({prefill_time:.0f}s) done")

    # Zero out stale KV cache positions left by the static PREFILL_MAX prefill v2 (writes cache rows
    # [seq_len:PREFILL_MAX] from epsilon-padded input — decode must not read those as real).
    zero_to = ue.PREFILL_MAX_SEQ_LEN
    if zero_to > seq_len:
        stale_size = (zero_to - seq_len) * ue.HEAD_DIM
        stale_zeros = torch.zeros(stale_size, dtype=torch.bfloat16)
        for layer in range(ue.NUM_LAYERS):
            for h in range(ue.NUM_KV_HEADS):
                k_stale = ue.LAYER0_K_DRAM + layer * ue.KV_LAYER_STRIDE + h * ue.KV_HEAD_STRIDE + seq_len * ue.HEAD_DIM * 2
                v_stale = ue.LAYER0_V_DRAM + layer * ue.KV_LAYER_STRIDE + h * ue.KV_HEAD_STRIDE + seq_len * ue.HEAD_DIM * 2
                ue.dma_to_accelerator_memory(k_stale, stale_zeros)
                ue.dma_to_accelerator_memory(v_stale, stale_zeros)

    # Decode — greedy_until is mode-dependent unless overridden (VLM postpones; LM penalizes from start).
    _greedy_until = args.greedy_until if args.greedy_until is not None \
        else (_d["vlm_greedy_until"] if vision_on else _d["lm_greedy_until"])
    ue.fpga_penalty = not args.pure_greedy
    ue.pen_alpha, ue.pen_cap, ue.rep_window, ue.greedy_until = args.pen_alpha, args.pen_cap, args.rep_window, _greedy_until
    max_new = args.max_seq - seq_len
    _original_print(f"\nPrompt:   {args.prompt}")
    _original_print(f"Response: ", end="", flush=True)   # the generated answer streams right after this
    # run_decoder prints the seed token itself (mirrors smolvlm2_test.run_decoder), so don't pre-print.
    decode_timer = time.perf_counter()
    hw_tokens = ue.run_decoder(hw_token, max_new_tokens=max_new)
    decode_time = time.perf_counter() - decode_timer
    _original_print(f"\nDecoder done in {prefill_time + decode_time:.2f}s total, {len(hw_tokens)} tokens.")


if __name__ == "__main__":
    main()
