#!/usr/bin/env python3
"""Qwen3-1.7B inference from pre-compiled bin files. Run qwen3_1.7b_test.py first to generate bins."""
import builtins
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

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
    DMA_DEVICE_H2C, TYPE, UE_FMAX_CONTEXT_SIZE, UE_VECTOR_SIZE,
    UE_ARGMAX_INDEX, URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS,
    UnifiedEngine, set_dma_device,
)
from model_lib_core import load_config_with_weight_defs

def _set_silent(val: bool) -> None:
    global _SILENT_MODE
    _SILENT_MODE = val

BIN_DIR = os.path.join(SCRIPT_DIR, "qwen3_1.7b_bin")


def check_bins_early(weights_bin_full):
    missing = []
    if not os.path.exists(weights_bin_full):
        missing.append(os.path.relpath(weights_bin_full, SCRIPT_DIR))
    for name in ("decoder_program.bin", "decoder_program.json"):
        if not os.path.exists(os.path.join(BIN_DIR, name)):
            missing.append(name)
    return missing


class Qwen3_1_7b_UnifiedEngine(UnifiedEngine):

    def __init__(self, script_dir=None, weights_bin=None):
        self._identity_dram_written = False
        self._identity_dram_addr = None
        self._IDENTITY_MAT_BYTES = UE_VECTOR_SIZE * UE_VECTOR_SIZE * 2
        super().__init__(
            params_dram_base=0x00000000,
            tensor_dram_base=0x58000000,
            program_dram_base=0x98000000,
        )
        self.script_dir = script_dir or SCRIPT_DIR
        self._cfg = load_config_with_weight_defs(os.path.join(self.script_dir, "qwen3_1.7b_config.json"))
        self.weight_defs = self._cfg["_weight_defs"]

        fi = self._cfg["file_info"]
        model = self._cfg["model"]
        paths = self._cfg["paths"]

        self.vector_length = fi["hidden_size"]
        self.head_dim = fi["head_dim"]
        self.actual_head_dim = fi["actual_head_dim"]
        self.num_kv_heads = fi["num_kv_heads"]
        self.bytes_per_element = fi["bytes_per_element"]
        self.group_size = fi["group_size"]
        self.mlp_elements = fi["mlp_elements"]
        self.hf_model_dir = os.path.join(self.script_dir, paths["hf_model_dir"])
        self.q_size = self.head_dim * self.group_size * self.bytes_per_element
        self.k_size = self.head_dim * self.bytes_per_element
        self.MAX_CONTEXT_SIZE = model["max_context_size"]
        self.LAYER_SIZE = fi["num_layers"]
        self.EMBEDDING_ELEMENTS = fi["embedding_vocab"]
        self._isa_reg_counter = 1
        fixed = self._cfg.get("fixed_isa_regs", {})
        self.V_CACHE_SIZE_REG = fixed["V_CACHE_SIZE_REG"]
        self.TMP_REG = fixed["TMP_REG"]
        self.ROPE_SIZE_REG = fixed["ROPE_SIZE_REG"]
        self.causal_mask_upper = False
        self._rope_global_layers = set(model["rope_global_layers"])
        self._end_of_turn_token_id = model["end_of_turn_token_id"]
        self._gamma_bin_offset = self._cfg["special"]["rms_norm"]["gamma_offset"]
        self._has_post_attn_norm = False
        self._has_post_mlp_norm = False

        bin_path = weights_bin or paths["weights_bin"]
        full_path = os.path.join(self.script_dir, bin_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Bin file not found: {full_path}")
        with open(full_path, "rb") as f:
            self.weight_bin = f.read()
        self.weight_init()
        self.tensor_init()

    # ------------------------------------------------------------------
    # Identity matrix caching (avoids 224 DMA writes during compile)
    # ------------------------------------------------------------------
    def _preallocate_identity_matrix(self):
        if self._identity_dram_addr is not None:
            return
        self._identity_dram_addr = super().allocate_params_dram(self._IDENTITY_MAT_BYTES)
        eye = torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16)
        super().dma_write(DMA_DEVICE_H2C, self._identity_dram_addr, eye, self._IDENTITY_MAT_BYTES)
        self._identity_dram_written = True

    def _flash_attention_core_cached(self, **kwargs):
        saved = self._next_params_dram_addr
        self._next_params_dram_addr = self._identity_dram_addr
        result = self.flash_attention_core(**kwargs)
        self._next_params_dram_addr = saved
        return result

    def dma_write(self, device, addr, data, size):
        if (self._identity_dram_written
                and self._identity_dram_addr is not None
                and addr == self._identity_dram_addr
                and size == self._IDENTITY_MAT_BYTES):
            return
        super().dma_write(device, addr, data, size)

    # ------------------------------------------------------------------
    # DRAM allocator override
    # ------------------------------------------------------------------
    def allocate_params_dram(self, size_bytes):
        params_dram_addr = self._next_params_dram_addr
        self._next_params_dram_addr += size_bytes
        return params_dram_addr

    # ------------------------------------------------------------------
    # ISA register helpers
    # ------------------------------------------------------------------
    def reset_isa_reg_counter(self):
        self._isa_reg_counter = 1

    def alloc_isa_reg(self, reset=False):
        if reset:
            self._isa_reg_counter = 1
        if self._isa_reg_counter > 15:
            raise ValueError("Exceeded available ISA registers (max 15)")
        reg_idx = self._isa_reg_counter
        self._isa_reg_counter += 1
        return reg_idx

    def clear_inst_id(self):
        self._inst_id = 0

    def isa_add_set_core(self, dst_reg_idx, immediate_value, timeout_s=10.0):
        self.clear_inst_id()
        self.start_capture()
        self.generate_instruction_add_set(dst_reg_idx, immediate_value)
        self.stop_capture()
        self.generate_instruction_halt()
        program_addr = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(program_addr)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        self.clear_capture_buffer()
        self.start_execute_from_dram(program_addr)
        self.wait_queue(timeout_s)

    # ------------------------------------------------------------------
    # Instruction loader
    # ------------------------------------------------------------------
    def load_instructions(self, bin_path):
        with open(bin_path, "rb") as f:
            data = f.read()
        total_size = len(data)
        start_addr = self.allocate_program_dram(total_size)
        self.dma_write(DMA_DEVICE_H2C, start_addr, data, total_size)
        return start_addr, total_size

    # ------------------------------------------------------------------
    # Embedding lookup
    # ------------------------------------------------------------------
    def get_embedding_for_tokens(self, token_ids):
        tid_t = torch.tensor(token_ids, dtype=torch.long)
        out = torch.zeros(len(token_ids), self.vector_length, dtype=torch.bfloat16)
        valid = tid_t < self.embedding_weight.shape[0]
        out[valid] = self.embedding_weight[tid_t[valid]]
        return out

    # ------------------------------------------------------------------
    # RoPE table (host-computed, DMA'd to params DRAM)
    # ------------------------------------------------------------------
    def _load_rope_host(self, rope_theta=None):
        rope_cfg = self._cfg["special"]["rope"]
        theta = rope_theta if rope_theta is not None else rope_cfg["theta"]
        num_rope_positions = rope_cfg["num_positions"]
        D_per_head = self.actual_head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(D_per_head, dtype=torch.float32) / D_per_head))
        pos = torch.arange(num_rope_positions, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        cos_ = freqs.cos().to(torch.bfloat16)
        sin_ = freqs.sin().to(torch.bfloat16)
        rope_tensor = torch.cat([cos_, cos_, -sin_, sin_], dim=1)
        rope_raw = rope_tensor.contiguous().view(torch.uint8).numpy().tobytes()
        local_sz  = self.weight_defs["ROPE_LOCAL_SIZE"]
        global_sz = self.weight_defs["ROPE_GLOBAL_SIZE"]
        local_raw  = (rope_raw + b"\x00" * local_sz)[:local_sz]
        global_raw = (rope_raw + b"\x00" * global_sz)[:global_sz]
        rope_buf = local_raw + global_raw
        rope_base = self.allocate_params_dram(len(rope_buf))
        self.dma_write(DMA_DEVICE_H2C, rope_base, rope_buf, len(rope_buf))
        self.DRAM_ADDR_ROPE_LOCAL  = rope_base
        self.DRAM_ADDR_ROPE_GLOBAL = rope_base + local_sz

    # ------------------------------------------------------------------
    # Weight init — reads embedding directly from weights bin (no HF model load)
    # ------------------------------------------------------------------
    def weight_init(self):
        emb_cfg = self._cfg["special"]["embedding"]
        token_embd_offset = int(emb_cfg["token_embd_offset"], 16)
        token_embd_size   = int(emb_cfg["token_embd_size"], 16)
        vocab_size   = emb_cfg["vocab_size"]
        emb_dim      = emb_cfg["embedding_dim"]
        emb_bytes = vocab_size * emb_dim * self.bytes_per_element
        raw_emb = bytearray(self.weight_bin[token_embd_offset : token_embd_offset + emb_bytes])
        self.embedding_weight = torch.frombuffer(raw_emb, dtype=torch.bfloat16).reshape(vocab_size, emb_dim).clone()
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_dir, trust_remote_code=True)

        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        base_layer0 = self.weight_defs["BLK0_ATTN_NORM_WEIGHT"]
        blk0_regions = [
            (s["key"], f"{s['key']}_SIZE", s["attr"])
            for s in self._cfg["layers"]["structure"]
        ]
        non_layer = [
            (s["key"], f"{s['key']}_SIZE", s["attr"])
            for s in self._cfg["layers"]["non_layer"]
            if s["key"] not in ("ROPE_LOCAL", "ROPE_GLOBAL")
        ]

        layers_total = self.LAYER_SIZE * LAYER_WEIGHT_SIZE
        layers_base_dram = self.allocate_params_dram(layers_total)
        bin_layers_start = base_layer0
        self.dma_write(DMA_DEVICE_H2C, layers_base_dram,
                       self.weight_bin[bin_layers_start : bin_layers_start + layers_total],
                       layers_total)
        for off_key, sz_key, attr in blk0_regions:
            offset_in_layer = self.weight_defs[off_key] - base_layer0
            setattr(self, attr, layers_base_dram + offset_in_layer)

        nl_slices = [self.weight_bin[self.weight_defs[k] : self.weight_defs[k] + self.weight_defs[s]]
                     for k, s, _ in non_layer]
        nl_buf = b"".join(nl_slices)
        nl_base_dram = self.allocate_params_dram(len(nl_buf))
        self.dma_write(DMA_DEVICE_H2C, nl_base_dram, nl_buf, len(nl_buf))
        nl_offset = 0
        for off_key, sz_key, attr in non_layer:
            setattr(self, attr, nl_base_dram + nl_offset)
            nl_offset += self.weight_defs[sz_key]

        self._load_rope_host()

    # ------------------------------------------------------------------
    # Tensor init
    # ------------------------------------------------------------------
    def tensor_init(self):
        seq_len = self.MAX_CONTEXT_SIZE
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        ahd  = self.actual_head_dim
        nkvh = self.num_kv_heads
        bpe  = self.bytes_per_element

        zero_add = torch.zeros(seq_len * self.head_dim * bpe, dtype=torch.bfloat16)
        self.ZERO_DRAM_ADDR = self.allocate_tensor_dram(seq_len * self.head_dim * bpe)
        self.dma_to_accelerator_memory(self.ZERO_DRAM_ADDR, zero_add)
        self.IDENTITY_DRAM_ADDR = self.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe)
        self.dma_to_accelerator_memory(self.IDENTITY_DRAM_ADDR, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))

        self.LAYER0_FLASH_Q_DRAM = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        self.LAYER0_FLASH_K_DRAM = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        self.LAYER0_FLASH_V_DRAM = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        zero_pad = torch.zeros(aligned_seq_len * ahd * bpe, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_Q_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_K_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_V_DRAM, zero_pad)

        self.LAYER0_FLASH_OUT_HEAD_DRAM    = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        self.LAYER0_FLASH_OUTPUT_DRAM      = self.allocate_tensor_dram(seq_len * self.head_dim * self.group_size * bpe)
        self.LAYER0_FLASH_SCRATCH_DRAM     = self.allocate_tensor_dram(max(ahd, UE_FMAX_CONTEXT_SIZE) * aligned_seq_len * 2 + ahd * aligned_seq_len * 2)
        self.LAYER0_FLASH_BIAS_DRAM        = self.allocate_tensor_dram(aligned_seq_len * aligned_seq_len * bpe)
        self.LAYER0_INPUT_DRAM             = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_NORM_DRAM          = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_Q_DRAM                 = self.allocate_tensor_dram(seq_len * self.q_size)
        self.LAYER0_K_DRAM                 = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_K_NORM_DRAM            = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_Q_NORM_DRAM            = self.allocate_tensor_dram(seq_len * self.q_size)
        self.LAYER0_V_PROJ_TEMP            = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_ATTN_PROJ_OUTPUT_DRAM  = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_ATTN_NORM_DRAM    = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_ATTN_RESIDUAL_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_MLP_NORM_DRAM      = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_MLP_GATE_DRAM          = self.allocate_tensor_dram(seq_len * self.mlp_elements * 2)
        self.LAYER0_MLP_UP_DRAM            = self.allocate_tensor_dram(seq_len * self.mlp_elements * 2)
        self.LAYER0_MLP_MULT_DRAM          = self.allocate_tensor_dram(seq_len * self.mlp_elements * 2)
        self.LAYER0_MLP_DOWN_DRAM          = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_MLP_NORM_DRAM     = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_OUTPUT_DRAM            = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.OUTPUT_NORM_DRAM              = self.allocate_tensor_dram(1 * self.vector_length * bpe)
        self.LOGITS_DRAM                   = self.allocate_tensor_dram(1 * self.EMBEDDING_ELEMENTS * bpe)

        kv_cache_total = self.LAYER_SIZE * nkvh * self.MAX_CONTEXT_SIZE * ahd * bpe
        self.LAYER0_V_DRAM       = self.allocate_tensor_dram(kv_cache_total)
        self.LAYER0_K_ROPE_DRAM  = self.allocate_tensor_dram(kv_cache_total)
        zero_kv = torch.zeros(kv_cache_total, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, zero_kv)
        self.dma_to_accelerator_memory(self.LAYER0_K_ROPE_DRAM, zero_kv)

        self._preallocate_identity_matrix()

    # ------------------------------------------------------------------
    # Program execute helper
    # ------------------------------------------------------------------
    def program_execute(self, program_start_addr, timeout=10.0, gflops=None):
        self.start_execute_from_dram(program_start_addr)
        self.wait_queue(timeout)
        if gflops is not None:
            self.report_flop_rate_gflops(gflops)

    # ------------------------------------------------------------------
    # Compile prefill (runs each invocation — no prefill bin support)
    # ------------------------------------------------------------------
    def compile_prefill(self, seq_len, layer_size=None):
        if layer_size is None:
            layer_size = self.LAYER_SIZE
        seq_len -= 1
        self.seq_len = seq_len
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64

        ahd  = self.actual_head_dim
        nkvh = self.num_kv_heads
        qpkv = self.group_size
        bpe  = self.bytes_per_element
        hd   = self.head_dim
        rope_row_bytes = ahd * 2 * bpe

        _set_silent(True)
        self.start_capture()
        total_flops = 0
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        _original_print(f"  Compiling prefill seq_len={seq_len}, {layer_size} layers...")
        for layer_idx in range(layer_size):
            _original_print(f"    prefill layer {layer_idx + 1}/{layer_size}", end="\r", flush=True)
            layer_off = layer_idx * LAYER_WEIGHT_SIZE
            if layer_idx != 0:
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, sram_address=0x10000, element_size=seq_len * self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_INPUT_DRAM, element_size=seq_len * self.vector_length)

            total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off)
            total_flops += self.quantized_matmat_core(M=seq_len, K=self.vector_length, N=hd * qpkv,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off, data_type=TYPE.INT4)
            total_flops += self.quantized_matmat_core(M=seq_len, K=self.vector_length, N=hd,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off, data_type=TYPE.INT4)
            total_flops += self.quantized_matmat_core(M=seq_len, K=self.vector_length, N=hd,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_V_PROJ_TEMP,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off, data_type=TYPE.INT4)

            total_flops += self.rms_norm_core_dram(M=seq_len * nkvh, N=ahd, A_DRAM_ADDR=self.LAYER0_K_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_K_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_NORM_GAMMA + layer_off)
            total_flops += self.rms_norm_core_dram(M=seq_len * nkvh * qpkv, N=ahd, A_DRAM_ADDR=self.LAYER0_Q_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_Q_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_NORM_GAMMA + layer_off)

            ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE_LOCAL
            for t in range(seq_len):
                cos_addr = ROPE_WEIGHT_ADDR + t * rope_row_bytes
                sin_addr = cos_addr + ahd * bpe
                for kv_h in range(nkvh):
                    total_flops += self.rope_core_dram_step(
                        N=ahd,
                        input_dram_addr=self.LAYER0_K_NORM_DRAM + (t * nkvh + kv_h) * ahd * bpe,
                        output_dram_addr=self.LAYER0_K_NORM_DRAM + (t * nkvh + kv_h) * ahd * bpe,
                        cos_dram_addr=cos_addr, sin_dram_addr=sin_addr)
                for q_h in range(nkvh * qpkv):
                    total_flops += self.rope_core_dram_step(
                        N=ahd,
                        input_dram_addr=self.LAYER0_Q_NORM_DRAM + (t * nkvh * qpkv + q_h) * ahd * bpe,
                        output_dram_addr=self.LAYER0_Q_NORM_DRAM + (t * nkvh * qpkv + q_h) * ahd * bpe,
                        cos_dram_addr=cos_addr, sin_dram_addr=sin_addr)

            for kv_h in range(nkvh):
                k_cache_base = (self.LAYER0_K_ROPE_DRAM
                                + layer_idx * nkvh * self.MAX_CONTEXT_SIZE * ahd * bpe
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)
                v_cache_base = (self.LAYER0_V_DRAM
                                + layer_idx * nkvh * self.MAX_CONTEXT_SIZE * ahd * bpe
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)

                for t in range(seq_len):
                    k_src = self.LAYER0_K_NORM_DRAM + (t * nkvh + kv_h) * ahd * bpe
                    self.accelerator_memory_to_sram(k_src, 0x10000, ahd)
                    self.sram_to_accelerator_memory(0x10000, k_cache_base + t * ahd * bpe, ahd)
                    for g in range(qpkv):
                        self.sram_to_accelerator_memory(0x10000, self.LAYER0_FLASH_K_DRAM + (t * qpkv + g) * ahd * bpe, ahd)

                for t in range(seq_len):
                    v_src = self.LAYER0_V_PROJ_TEMP + (t * nkvh + kv_h) * ahd * bpe
                    self.accelerator_memory_to_sram(v_src, 0x20000, ahd)
                    self.sram_to_accelerator_memory(0x20000, v_cache_base + t * ahd * bpe, ahd)
                    for g in range(qpkv):
                        self.sram_to_accelerator_memory(0x20000, self.LAYER0_FLASH_V_DRAM + (t * qpkv + g) * ahd * bpe, ahd)

                for t in range(seq_len):
                    for q in range(qpkv):
                        q_src = self.LAYER0_Q_NORM_DRAM + (t * nkvh * qpkv + kv_h * qpkv + q) * ahd * bpe
                        self.accelerator_memory_to_sram(q_src, 0x30000, ahd)
                        self.sram_to_accelerator_memory(0x30000, self.LAYER0_FLASH_Q_DRAM + (t * qpkv + q) * ahd * bpe, ahd)

                total_flops += self._flash_attention_core_cached(
                    head_dim=ahd, seq_len=aligned_seq_len,
                    Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
                    K_DRAM_ADDR=self.LAYER0_FLASH_K_DRAM,
                    V_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                    OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUT_HEAD_DRAM,
                    SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                    BIAS_DRAM_ADDR=self.LAYER0_FLASH_BIAS_DRAM,
                )

                out_h_base = kv_h * qpkv * ahd * bpe
                for t in range(seq_len):
                    for g in range(qpkv):
                        src = self.LAYER0_FLASH_OUT_HEAD_DRAM + (t * qpkv + g) * ahd * bpe
                        dst = self.LAYER0_FLASH_OUTPUT_DRAM + t * hd * qpkv * bpe + out_h_base + g * ahd * bpe
                        self.accelerator_memory_to_sram(src, 0x40000, ahd)
                        self.sram_to_accelerator_memory(0x40000, dst, ahd)

            total_flops += self.quantized_matmat_core(M=seq_len, K=hd * qpkv, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off, data_type=TYPE.INT4)

            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_INPUT_DRAM, sram_address=0x10000, element_size=seq_len * self.vector_length)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM, sram_address=0x90000, element_size=seq_len * self.vector_length)
            self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=seq_len * self.vector_length)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, element_size=seq_len * self.vector_length)

            total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off)

            total_flops += self.quantized_matmat_core(M=seq_len, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off, data_type=TYPE.INT4, silu_enable=True)
            total_flops += self.quantized_matmat_core(M=seq_len, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off, data_type=TYPE.INT4)

            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_GATE_DRAM, sram_address=0x10000, element_size=seq_len * self.mlp_elements)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_UP_DRAM, sram_address=0x90000, element_size=seq_len * self.mlp_elements)
            self.eltwise_mul_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=seq_len * self.mlp_elements)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_MLP_MULT_DRAM, element_size=seq_len * self.mlp_elements)

            total_flops += self.quantized_matmat_core(M=seq_len, K=self.mlp_elements, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off, data_type=TYPE.INT4)

            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, sram_address=0x10000, element_size=seq_len * self.vector_length)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_DOWN_DRAM, sram_address=0x90000, element_size=seq_len * self.vector_length)
            self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=seq_len * self.vector_length)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, element_size=seq_len * self.vector_length)

        self.stop_capture()
        self.generate_instruction_halt()
        prefill_program_addr = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(prefill_program_addr)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        # Save bin for future runs
        prefill_bytes = bytearray()
        for inst in self.capture_buffer:
            prefill_bytes.extend(inst.get_bytes())
        bin_path  = os.path.join(BIN_DIR, f"prefill_program_S{seq_len + 1}.bin")
        meta_path = os.path.join(BIN_DIR, f"prefill_program_S{seq_len + 1}.json")
        with open(bin_path, "wb") as f:
            f.write(prefill_bytes)
        with open(meta_path, "w") as f:
            json.dump({"seq_len": seq_len + 1, "total_flops": total_flops}, f)
        self.clear_capture_buffer()
        _set_silent(False)
        _original_print()
        return prefill_program_addr, total_flops

    def load_prefill(self, seq_len: int):
        """Load pre-compiled prefill from bin. Returns (program_addr, total_flops)."""
        bin_path  = os.path.join(BIN_DIR, f"prefill_program_S{seq_len}.bin")
        meta_path = os.path.join(BIN_DIR, f"prefill_program_S{seq_len}.json")
        with open(bin_path, "rb") as f:
            data = f.read()
        with open(meta_path) as f:
            meta = json.load(f)
        self.seq_len = seq_len - 1
        addr = self.get_program_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, addr, data, len(data))
        self.allocate_program_dram(len(data))
        _original_print(f"  Prefill loaded from bin ({len(data)} bytes)")
        return addr, meta["total_flops"]

    # ------------------------------------------------------------------
    # Run prefill
    # ------------------------------------------------------------------
    def run_prefill(self, prefill_program_addr, prefill_seq, gflops=None):
        if len(prefill_seq) > 1:
            prefill_seq = prefill_seq[:-1]
        else:
            raise ValueError("Prefill sequence must have at least 2 tokens.")

        seq_len = len(prefill_seq)
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64

        embedding_tensor = self.get_embedding_for_tokens(prefill_seq)
        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
        bias_one_group = torch.full((aligned_seq_len, aligned_seq_len), float("-inf"), dtype=torch.bfloat16)
        valid_mask = torch.tril(torch.ones(aligned_seq_len, aligned_seq_len, dtype=torch.bool), diagonal=0)
        bias_one_group.masked_fill_(valid_mask, 0.0)
        bias_one_group[:, q_seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_one_group)
        self.program_execute(prefill_program_addr, gflops=gflops)

    # ------------------------------------------------------------------
    # Run decoder
    # ------------------------------------------------------------------
    def run_decoder(self, decoder_program_sizes, decoder_base_addr, token_id, gflops_per_token=None):
        if token_id is None:
            return 0

        _qwen3_stop_tokens = {151643, 151645, self._end_of_turn_token_id}
        ahd = self.actual_head_dim
        bpe = self.bytes_per_element
        _kv_stride   = ahd * bpe
        _rope_stride = ahd * 2 * bpe
        max_seq_len = self.MAX_CONTEXT_SIZE

        while self.seq_len < max_seq_len:
            _set_silent(True)
            self.seq_len += 1
            aligned_seq_len = ((self.seq_len + 63) // 64) * 64
            prog_idx = min((self.seq_len - 1) // 256, 7)
            prog_addr = decoder_base_addr + sum(decoder_program_sizes[:prog_idx])
            gflops = gflops_per_token[prog_idx] if gflops_per_token else None

            self.isa_add_set_core(self.V_CACHE_SIZE_REG, (self.seq_len - 1) * _kv_stride)
            self.isa_add_set_core(self.ROPE_SIZE_REG,    (self.seq_len - 1) * _rope_stride)

            embedding_tensor = self.get_embedding_for_tokens([token_id])
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)

            bias_host = torch.full((1, aligned_seq_len), -1e36, dtype=torch.bfloat16)
            bias_host[0, :self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_host)

            self.start_execute_from_dram(prog_addr)
            self.wait_queue(10.0)
            token_id = self.read_reg32(UE_ARGMAX_INDEX)
            token_char = self.tokenizer.decode([token_id])
            _set_silent(False)
            if token_id in _qwen3_stop_tokens:
                print(f"\nStop token {token_id} reached.")
                break
            print(token_char, end="", flush=True)
        return self.seq_len


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3-1.7B inference from pre-compiled bins")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--local-weights", action="store_true")
    parser.add_argument("--dev", type=str, default="xdma0")
    parser.add_argument("--cycle", type=float, default=1/0.17)
    args = parser.parse_args()

    if args.local_weights:
        weights_bin_rel = "qwen3_1.7b_bin/full_model_weights.bin"
    else:
        weights_bin_rel = "qwen3_1.7b_bin/weights_qwen3_1.7b_hf.bin"
    weights_bin_full = os.path.join(SCRIPT_DIR, weights_bin_rel)

    missing = check_bins_early(weights_bin_full)
    if missing:
        _original_print("Missing bin files (run qwen3_1.7b_test.py first to compile):")
        for f in missing:
            _original_print(f"  {f}")
        return

    set_dma_device(args.dev)
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle

    global _SILENT_MODE
    _SILENT_MODE = True
    ue = Qwen3_1_7b_UnifiedEngine(script_dir=SCRIPT_DIR, weights_bin=weights_bin_rel)
    _SILENT_MODE = False

    cfg = load_config_with_weight_defs(os.path.join(SCRIPT_DIR, "qwen3_1.7b_config.json"))

    if args.prompt is not None:
        conversation = [{"role": "user", "content": args.prompt}]
        prompt_with_template = ue.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        prefill_seq = tuple(ue.tokenizer.encode(prompt_with_template, add_special_tokens=False))
        _original_print(f"Prompt ({len(prefill_seq)} tokens): {args.prompt!r}")
    else:
        prefill_seq = tuple(cfg["default_prefill_tokens"])

    prefill_bin = os.path.join(BIN_DIR, f"prefill_program_S{len(prefill_seq)}.bin")
    timer = time.perf_counter()
    if os.path.exists(prefill_bin):
        _original_print(f"\n--- Loading prefill bin ({len(prefill_seq)} tokens) ---")
        prefill_program_addr, gflops_prefill = ue.load_prefill(seq_len=len(prefill_seq))
    else:
        _original_print(f"\n--- Compiling prefill ({len(prefill_seq)} tokens) ---")
        prefill_program_addr, gflops_prefill = ue.compile_prefill(seq_len=len(prefill_seq))
    _original_print(f"  Done in {time.perf_counter() - timer:.2f}s")

    decoder_bin_path  = os.path.join(BIN_DIR, "decoder_program.bin")
    decoder_meta_path = os.path.join(BIN_DIR, "decoder_program.json")
    with open(decoder_meta_path) as f:
        meta = json.load(f)
    if "instruction_counts" in meta:
        decoder_program_sizes = [c * 32 for c in meta["instruction_counts"]]
    else:
        decoder_program_sizes = meta["program_sizes"]
    gflops_per_token = meta["total_flops"]
    decoder_base_addr, _ = ue.load_instructions(decoder_bin_path)
    _original_print("  Decoder loaded from bin")

    _original_print(f"\n--- Starting prefill ({len(prefill_seq)} tokens) ---")
    timer = time.perf_counter()
    ue.run_prefill(prefill_program_addr, prefill_seq=prefill_seq, gflops=gflops_prefill)
    latency_prefill = time.perf_counter() - timer
    _original_print(f"  Done in {latency_prefill:.2f}s")

    _original_print(f"\n--- Starting decoder ---")
    timer = time.perf_counter()
    token_cnt = ue.run_decoder(decoder_program_sizes, decoder_base_addr,
                               token_id=prefill_seq[-1], gflops_per_token=gflops_per_token)
    latency_decoder = time.perf_counter() - timer
    _original_print(f"\nDecoder done in {latency_prefill + latency_decoder:.2f}s, total {token_cnt} tokens.")


if __name__ == "__main__":
    main()
