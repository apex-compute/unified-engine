#!/usr/bin/env python3
"""Llama-3.2-1B inference from pre-compiled bin files.

Compile first (auto-skipped if already compiled):
    python llama3.2_1b_test.py

Then run:
    python llama3.2_1b_run_from_bin.py
    python llama3.2_1b_run_from_bin.py --prompt "your question"
    python llama3.2_1b_run_from_bin.py --dev xdma0 [--cycle 5.88]
"""

import json
import math
import os
import sys
import threading
import time

import torch
from transformers import AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, TYPE, UE_FMAX_CONTEXT_SIZE, UE_VECTOR_SIZE, UE_ARGMAX_INDEX,
    URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS, set_dma_device, ue_35bit_addr_shifter,
)
from user_dma_core import UnifiedEngine

import builtins
_original_print = builtins.print
_SILENT_MODE = False

def quiet_print(*args, **kwargs):
    if _SILENT_MODE:
        return
    _original_print(*args, **kwargs)

builtins.print = quiet_print


def _parse_offset(val) -> int:
    if isinstance(val, str):
        return int(val, 0)
    return int(val)


def _load_config(script_dir: str) -> dict:
    config_path = os.path.join(script_dir, "llama3.2_1b_config.json")
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


class Llama32_1b_UnifiedEngine(UnifiedEngine):

    def __init__(self, script_dir: str | None = None, weights_bin: str | None = None):
        super().__init__(program_dram_base=0xC0000000)
        self.script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
        self._cfg = _load_config(self.script_dir)
        self.weight_defs = self._cfg["_weight_defs"]

        fi = self._cfg["file_info"]
        model = self._cfg["model"]
        paths = self._cfg["paths"]

        self.vector_length = fi["hidden_size"]
        self.head_dim = fi["head_dim"]
        self.bytes_per_element = fi["bytes_per_element"]
        self.group_size = fi["group_size"]
        self.mlp_elements = fi["mlp_elements"]
        self.hf_model_dir = os.path.join(self.script_dir, paths["hf_model_dir"])
        self.q_size = self.head_dim * self.group_size * self.bytes_per_element
        self.k_size = self.head_dim * self.bytes_per_element
        self.actual_head_dim = 64
        self.num_kv_heads = self.head_dim // self.actual_head_dim  # 8
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

        self._has_q_k_norm = False
        self._has_post_attn_norm = False
        self._has_post_mlp_norm = False

        bin_path = weights_bin or paths["weights_bin"]
        full_path = os.path.join(self.script_dir, bin_path)
        if not os.path.exists(full_path):
            alt = os.path.join(self.script_dir, paths.get("local_weights_bin", ""))
            if os.path.exists(alt):
                full_path = alt
            else:
                raise FileNotFoundError(f"Weight bin not found: {full_path}")
        # software_reset BEFORE any DMA-to-DRAM. Running it after weight_init corrupts
        # the most recently written DRAM pages (the start of the params region).
        # On slow buses (Pi PCIe Gen2 x1), the reset-ack returns before the FPGA-side
        # reset is fully settled, so the first weight DMA still gets clobbered. The
        # extra sleep gives the controller time to quiesce.
        self.software_reset()
        time.sleep(0.5)
        with open(full_path, "rb") as f:
            self.weight_bin = f.read()
        self.weight_init()
        self.tensor_init()

    def reset_isa_reg_counter(self) -> None:
        self._isa_reg_counter = 1

    def alloc_isa_reg(self, reset: bool = False) -> int:
        if reset:
            self._isa_reg_counter = 1
        if self._isa_reg_counter > 15:
            raise ValueError("Exceeded available ISA registers (max 15)")
        reg_idx = self._isa_reg_counter
        self._isa_reg_counter += 1
        return reg_idx

    def isa_add_set_core(self, dst_reg_idx: int, immediate_value: int, timeout_s: float = 10.0) -> None:
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

    def load_instructions(self, bin_path: str) -> tuple[int, int]:
        with open(bin_path, "rb") as f:
            data = f.read()
        total_size = len(data)
        start_addr = self.allocate_program_dram(total_size)
        self.dma_write(DMA_DEVICE_H2C, start_addr, data, total_size)
        _original_print(f"    Loaded {total_size} bytes from {os.path.basename(bin_path)} to DRAM at 0x{start_addr:x}")
        return start_addr, total_size

    def allocate_params_dram(self, size_bytes: int) -> int:
        params_dram_addr = self._next_params_dram_addr
        self._next_params_dram_addr += size_bytes
        return params_dram_addr

    def clear_inst_id(self) -> None:
        self._inst_id = 0

    def get_arg_max_index(self) -> int:
        return self.read_reg32(UE_ARGMAX_INDEX)

    def rope_hf_core(self, N: int, input_dram_addr: int, output_dram_addr: int,
                     cos_dram_addr: int, sin_dram_addr: int,
                     rope_size_reg: int = None, output_addr_inc_reg: int = None,
                     tmp_reg: int = None) -> int:
        assert N % UE_VECTOR_SIZE == 0 and N >= 64
        assert N % 2 == 0
        assert N >= 128
        half = N // 2
        bytes_per_elem = 2
        sram_x   = 0x00000
        sram_a   = 0x20000
        sram_d   = 0x40000
        sram_cos = 0x80000
        sram_sin = 0x80000 + N * bytes_per_elem
        sram_bc  = 0x80000 + N * bytes_per_elem * 2
        self.accelerator_memory_to_sram(accelerator_dram_address=input_dram_addr, sram_address=sram_x, element_size=N)
        if rope_size_reg is not None:
            self.generate_instruction_add_imm(rope_size_reg, ue_35bit_addr_shifter(cos_dram_addr), tmp_reg)
            self.accelerator_memory_to_sram(accelerator_dram_address=cos_dram_addr, sram_address=sram_cos, element_size=N)
            self.overwrite_instruction_with_general_register(tmp_reg)
            self.generate_instruction_add_imm(rope_size_reg, ue_35bit_addr_shifter(sin_dram_addr), tmp_reg)
            self.accelerator_memory_to_sram(accelerator_dram_address=sin_dram_addr, sram_address=sram_sin, element_size=N)
            self.overwrite_instruction_with_general_register(tmp_reg)
        else:
            self.accelerator_memory_to_sram(accelerator_dram_address=cos_dram_addr, sram_address=sram_cos, element_size=N)
            self.accelerator_memory_to_sram(accelerator_dram_address=sin_dram_addr, sram_address=sram_sin, element_size=N)
        self.eltwise_mul_core(vector_A_sram_start_addr=sram_x, vector_B_sram_start_addr=sram_cos, vector_C_sram_wb_addr=sram_a, element_size=N)
        self.eltwise_mul_core(vector_A_sram_start_addr=sram_x + half * bytes_per_elem, vector_B_sram_start_addr=sram_sin, vector_C_sram_wb_addr=sram_bc, element_size=half)
        self.eltwise_mul_core(vector_A_sram_start_addr=sram_x, vector_B_sram_start_addr=sram_sin + half * bytes_per_elem, vector_C_sram_wb_addr=sram_bc + half * bytes_per_elem, element_size=half)
        self.eltwise_add_core(vector_A_sram_start_addr=sram_a, vector_B_sram_start_addr=sram_bc, vector_C_sram_wb_addr=sram_d, element_size=N)
        if output_addr_inc_reg is not None:
            self.generate_instruction_add_imm(output_addr_inc_reg, ue_35bit_addr_shifter(output_dram_addr), tmp_reg)
            self.sram_to_accelerator_memory(sram_address=sram_d, accelerator_dram_address=output_dram_addr, element_size=N)
            self.overwrite_instruction_with_general_register(tmp_reg)
        else:
            self.sram_to_accelerator_memory(sram_address=sram_d, accelerator_dram_address=output_dram_addr, element_size=N)
        return 4 * N

    def decoder_attention_core(self, head_dim: int, seq_len: int,
                               Q_DRAM_ADDR: int, K_DRAM_ADDR: int, V_DRAM_ADDR: int,
                               OUTPUT_DRAM_ADDR: int, SCRATCH_DRAM_ADDR: int,
                               IDENTITY_DRAM_ADDR: int = None, BIAS_DRAM_ADDR: int = None,
                               debug_mode: bool = False, SM_OUTPUT_DRAM_ADDR: int = None) -> None:
        bytes_per_element = 2
        bias_enable = True if BIAS_DRAM_ADDR is not None else False
        if debug_mode:
            assert SM_OUTPUT_DRAM_ADDR is not None
        SCRATCH_DRAM_PARTIAL_SM = SCRATCH_DRAM_ADDR + head_dim * seq_len * bytes_per_element
        M = head_dim
        K = head_dim
        N = seq_len
        identity_tensor = torch.eye(head_dim, dtype=torch.bfloat16)
        self.accelerator_memory_to_sram(accelerator_dram_address=IDENTITY_DRAM_ADDR, sram_address=0, element_size=UE_VECTOR_SIZE * UE_VECTOR_SIZE)
        usable_uram_a_start_addr = UE_VECTOR_SIZE * UE_VECTOR_SIZE * bytes_per_element
        usable_uram_b_elements = URAM_NEAR_FULL_ELEMENTS
        N_chunk = min(N, (usable_uram_b_elements // K) // UE_VECTOR_SIZE * UE_VECTOR_SIZE)
        N_chunk_aligned = None
        if N_chunk < UE_VECTOR_SIZE:
            if (K * 32) <= usable_uram_b_elements:
                N_chunk = 32
            elif (K * 16) <= usable_uram_b_elements:
                N_chunk = 16
            else:
                assert False, f"K={K} too large"
            N_chunk_aligned = UE_VECTOR_SIZE
        usable_uram_a_elements = URAM_FULL_ELEMENTS - UE_VECTOR_SIZE * UE_VECTOR_SIZE
        output_N_size = N_chunk_aligned if N_chunk_aligned is not None else N_chunk
        M_chunk = min(M, usable_uram_a_elements // output_N_size)
        assert M_chunk >= 1 and M_chunk <= M
        output_sram_wb_addr = usable_uram_a_start_addr
        uram_b_start_addr = 0x80000
        for i, m_take in self.chunk_ranges(M, M_chunk):
            for j, n_take in self.chunk_ranges(N, N_chunk):
                self.accelerator_memory_to_sram(accelerator_dram_address=V_DRAM_ADDR + j * K * bytes_per_element, sram_address=uram_b_start_addr, element_size=n_take * K)
                for output_row in range(m_take):
                    if N_chunk_aligned is None:
                        out_sram_offset = output_row * n_take * bytes_per_element
                    else:
                        out_sram_offset = output_row * N_chunk_aligned * bytes_per_element
                    ones_idx = identity_tensor[output_row+i, :].reshape(-1, UE_VECTOR_SIZE).sum(axis=1).argmax(axis=0)
                    vector_idx = identity_tensor[output_row+i, :].reshape(-1, UE_VECTOR_SIZE)[ones_idx, :].argmax(axis=0)
                    self.start_queue_for_bf16_matvec_operation(max_clear_en=0, fmax_context_addr=0,
                        vector_sram_start_addr=0x00000 + vector_idx * UE_VECTOR_SIZE * bytes_per_element,
                        matrix_sram_start_addr=uram_b_start_addr + ones_idx * UE_VECTOR_SIZE * bytes_per_element,
                        output_sram_wb_addr=output_sram_wb_addr + out_sram_offset, K=UE_VECTOR_SIZE, N=n_take, stride_z=m_take)
                start_dram_address_of_partial_matrix = SCRATCH_DRAM_ADDR + i * N * bytes_per_element + j * bytes_per_element
                if N_chunk_aligned is None:
                    self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr, accelerator_dram_address=start_dram_address_of_partial_matrix,
                        element_size=m_take * n_take, stride_bytes_per_chunk=n_take * bytes_per_element, stride_jump_bytes=N * bytes_per_element)
                else:
                    for o_row_idx in range(m_take):
                        self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr + o_row_idx * N_chunk_aligned * bytes_per_element,
                            accelerator_dram_address=start_dram_address_of_partial_matrix + o_row_idx * N * bytes_per_element, element_size=n_take)
        M = 1
        K = head_dim
        N = seq_len
        usable_uram_b_elements = URAM_NEAR_FULL_ELEMENTS
        N_chunk = min(N, (usable_uram_b_elements // K) // UE_VECTOR_SIZE * UE_VECTOR_SIZE)
        N_chunk_aligned = None
        if N_chunk < UE_VECTOR_SIZE:
            if (K * 32) <= usable_uram_b_elements:
                N_chunk = 32
            elif (K * 16) <= usable_uram_b_elements:
                N_chunk = 16
            else:
                assert False
            N_chunk_aligned = UE_VECTOR_SIZE
        usable_uram_a_elements = URAM_FULL_ELEMENTS
        output_N_size = N_chunk_aligned if N_chunk_aligned is not None else N_chunk
        M_chunk = min(UE_FMAX_CONTEXT_SIZE, M, usable_uram_a_elements // (K + output_N_size))
        assert M_chunk >= 1 and M_chunk <= M
        uram_a_start_addr = 0x00000
        uram_b_start_addr = 0x80000
        for i, m_take in self.chunk_ranges(M, M_chunk):
            self.accelerator_memory_to_sram(accelerator_dram_address=Q_DRAM_ADDR + i * K * bytes_per_element, sram_address=uram_a_start_addr, element_size=m_take * K)
            self.broadcast_mul(scalar=1 / math.sqrt(head_dim), sram_start_addr=uram_a_start_addr, sram_wb_addr=uram_a_start_addr, element_size=m_take * K)
            output_sram_wb_addr = uram_a_start_addr + m_take * K * bytes_per_element
            assert output_sram_wb_addr < 0x80000
            clear_en = 1
            for j, n_take in self.chunk_ranges(N, N_chunk):
                self.accelerator_memory_to_sram(accelerator_dram_address=K_DRAM_ADDR + j * K * bytes_per_element, sram_address=uram_b_start_addr, element_size=n_take * K)
                if bias_enable:
                    self.accelerator_memory_to_bias_sram(accelerator_dram_address=BIAS_DRAM_ADDR + j * bytes_per_element, element_size=n_take)
                assert m_take * K + n_take * m_take <= URAM_FULL_ELEMENTS
                for output_row in range(m_take):
                    if N_chunk_aligned is None:
                        out_sram_offset = output_row * n_take * bytes_per_element
                    else:
                        out_sram_offset = output_row * N_chunk_aligned * bytes_per_element
                    self.start_queue_for_bf16_matvec_operation(max_clear_en=clear_en, fmax_context_addr=output_row,
                        vector_sram_start_addr=uram_a_start_addr + output_row * K * bytes_per_element,
                        matrix_sram_start_addr=uram_b_start_addr, output_sram_wb_addr=output_sram_wb_addr + out_sram_offset,
                        K=K, N=n_take, bias_enable=bias_enable)
                    clear_en = 0
                start_dram_address_of_partial_matrix = SCRATCH_DRAM_PARTIAL_SM + j * bytes_per_element
                if N_chunk_aligned is None:
                    self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr, accelerator_dram_address=start_dram_address_of_partial_matrix,
                        element_size=m_take * n_take, stride_bytes_per_chunk=n_take * bytes_per_element, stride_jump_bytes=N * bytes_per_element)
                else:
                    for o_row_idx in range(m_take):
                        self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr + o_row_idx * N_chunk_aligned * bytes_per_element,
                            accelerator_dram_address=start_dram_address_of_partial_matrix + o_row_idx * N * bytes_per_element, element_size=n_take)
            max_m_take = min((URAM_FULL_ELEMENTS - UE_VECTOR_SIZE) // N, UE_FMAX_CONTEXT_SIZE)
            for m_take_chunk_idx, m_take_chunk_size in self.chunk_ranges(m_take, max_m_take):
                self.accelerator_memory_to_sram(accelerator_dram_address=SCRATCH_DRAM_PARTIAL_SM + m_take_chunk_idx * N * bytes_per_element,
                    sram_address=uram_a_start_addr, element_size=m_take_chunk_size * N)
                for row_idx in range(m_take_chunk_size):
                    self.start_queue_for_bf16_softmax_operation(fmax_context_addr=row_idx + m_take_chunk_idx,
                        vector_sram_start_addr=uram_a_start_addr + row_idx * N * bytes_per_element,
                        output_sram_wb_addr=uram_a_start_addr + row_idx * N * bytes_per_element, N=N)
                if debug_mode:
                    self.sram_to_accelerator_memory(sram_address=uram_a_start_addr,
                        accelerator_dram_address=SM_OUTPUT_DRAM_ADDR + (i + m_take_chunk_idx) * N * bytes_per_element,
                        element_size=m_take_chunk_size * N)
                v_tr_row_chunk_size = min((URAM_NEAR_FULL_ELEMENTS // seq_len // UE_VECTOR_SIZE) * UE_VECTOR_SIZE,
                    ((URAM_FULL_ELEMENTS - m_take_chunk_size * seq_len) // m_take_chunk_size // UE_VECTOR_SIZE) * UE_VECTOR_SIZE, head_dim)
                v_tr_row_chunk_size_aligned = None
                if v_tr_row_chunk_size < UE_VECTOR_SIZE:
                    v_tr_row_chunk_size_aligned = UE_VECTOR_SIZE
                    if seq_len * 32 <= URAM_NEAR_FULL_ELEMENTS:
                        v_tr_row_chunk_size = 32
                    elif seq_len * 16 <= URAM_NEAR_FULL_ELEMENTS:
                        v_tr_row_chunk_size = 16
                    else:
                        assert False
                v_t_sram_start_addr = 0x80000
                output_sram_wb_addr = uram_a_start_addr + m_take_chunk_size * seq_len * bytes_per_element
                for v_tr_column_idx, v_tr_column_take in self.chunk_ranges(head_dim, v_tr_row_chunk_size):
                    self.accelerator_memory_to_sram(accelerator_dram_address=SCRATCH_DRAM_ADDR + v_tr_column_idx * seq_len * bytes_per_element,
                        sram_address=v_t_sram_start_addr, element_size=v_tr_column_take * seq_len)
                    for p_row_idx in range(m_take_chunk_size):
                        if v_tr_row_chunk_size_aligned is None:
                            output_sram_wb_offset = p_row_idx * v_tr_column_take * bytes_per_element
                        else:
                            output_sram_wb_offset = 0
                        self.start_queue_for_bf16_matvec_operation(max_clear_en=0, fmax_context_addr=0,
                            vector_sram_start_addr=uram_a_start_addr + p_row_idx * seq_len * bytes_per_element,
                            matrix_sram_start_addr=v_t_sram_start_addr, output_sram_wb_addr=output_sram_wb_addr + output_sram_wb_offset,
                            K=seq_len, N=v_tr_column_take)
                        if v_tr_row_chunk_size_aligned is not None:
                            self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr + output_sram_wb_offset,
                                accelerator_dram_address=OUTPUT_DRAM_ADDR + (i + m_take_chunk_idx) * head_dim * bytes_per_element
                                    + v_tr_column_idx * bytes_per_element + p_row_idx * head_dim * bytes_per_element,
                                element_size=v_tr_column_take)
                    if v_tr_row_chunk_size_aligned is None:
                        self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr,
                            accelerator_dram_address=OUTPUT_DRAM_ADDR + (i + m_take_chunk_idx) * head_dim * bytes_per_element + v_tr_column_idx * bytes_per_element,
                            element_size=m_take_chunk_size * v_tr_column_take, stride_bytes_per_chunk=v_tr_column_take * bytes_per_element,
                            stride_jump_bytes=head_dim * bytes_per_element)
        total_flops = 1 * head_dim
        total_flops += 2 * 1 * head_dim * seq_len
        total_flops += 1 * seq_len * 5
        total_flops += 2 * 1 * seq_len * head_dim
        return total_flops

    def get_embedding_for_tokens(self, token_ids: list[int] | tuple) -> torch.Tensor:
        tid_t = torch.tensor(token_ids, dtype=torch.long)
        out = torch.zeros(len(token_ids), self.vector_length, dtype=torch.bfloat16)
        valid = tid_t < self.embedding_weight.shape[0]
        out[valid] = self.embedding_weight[tid_t[valid]]
        return out

    def _load_rope_host(self, rope_theta: float | None = None, rope_local_base: float | None = None) -> None:
        rope_cfg = self._cfg["special"]["rope"]
        theta = rope_theta if rope_theta is not None else rope_cfg["theta"]
        local_base = rope_local_base if rope_local_base is not None else rope_cfg["local_base"]
        num_rope_positions = rope_cfg["num_positions"]
        D_per_head = self.actual_head_dim // 2
        for name, theta_val, sz_key, attr in [
            ("ROPE_LOCAL", local_base, "ROPE_LOCAL_SIZE", "DRAM_ADDR_ROPE_LOCAL"),
            ("ROPE_GLOBAL", theta, "ROPE_GLOBAL_SIZE", "DRAM_ADDR_ROPE_GLOBAL"),
        ]:
            inv_freq = 1.0 / (theta_val ** (torch.arange(D_per_head, dtype=torch.float32) / D_per_head))
            pos = torch.arange(num_rope_positions, dtype=torch.float32)
            freqs = torch.outer(pos, inv_freq)
            cos_head = freqs.cos().to(torch.bfloat16)
            sin_head = freqs.sin().to(torch.bfloat16)
            cos_full = cos_head.repeat(1, self.num_kv_heads)
            sin_full = sin_head.repeat(1, self.num_kv_heads)
            rope_tensor = torch.cat([cos_full, cos_full, -sin_full, sin_full], dim=1)
            sz = self.weight_defs[sz_key]
            raw = rope_tensor.contiguous().view(torch.uint8).numpy().tobytes()
            raw = (raw + b"\x00" * sz)[:sz]
            addr = self.allocate_params_dram(sz)
            self.dma_write(DMA_DEVICE_H2C, addr, raw, sz)
            setattr(self, attr, addr)

    def weight_init(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_dir, trust_remote_code=True)

        emb_cfg = self._cfg["special"]["embedding"]
        token_embd_offset = _parse_offset(emb_cfg["token_embd_offset"])
        token_embd_size = _parse_offset(emb_cfg["token_embd_size"])
        vocab_size = emb_cfg["vocab_size"]
        embedding_dim = emb_cfg["embedding_dim"]
        raw_emb = self.weight_bin[token_embd_offset:token_embd_offset + token_embd_size]
        emb_tensor = torch.frombuffer(bytearray(raw_emb), dtype=torch.bfloat16).reshape(vocab_size, embedding_dim)
        self.embedding_weight = emb_tensor.clone()

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

        _original_print(f"\n--- Weights DRAM allocation, start at DRAM address: {self.get_params_dram_addr()} ---")
        layers_total = self.LAYER_SIZE * LAYER_WEIGHT_SIZE
        layers_base_dram = self.allocate_params_dram(layers_total)
        for layer_idx in range(self.LAYER_SIZE):
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
        _original_print(f"Layers 0..{self.LAYER_SIZE - 1} loaded: 0x{layers_base_dram:X} size {layers_total}")

        for off_key, sz_key, attr in non_layer:
            off = self.weight_defs[off_key]
            sz = self.weight_defs[sz_key]
            raw = self.weight_bin[off : off + sz]
            addr = self.allocate_params_dram(sz)
            self.dma_write(DMA_DEVICE_H2C, addr, raw, sz)
            setattr(self, attr, addr)

        self._load_rope_host()
        _original_print(f"    Weights end at DRAM address: 0x{self.get_params_dram_addr():X}")
        _original_print("Tokenizer loaded successfully.")

    def tensor_init(self) -> None:
        seq_len = self.MAX_CONTEXT_SIZE
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64

        _original_print(f"Allocate tensor dram start at DRAM address: 0x{self.get_tensor_dram_addr():X}")
        self.LAYER0_V_DRAM = self.allocate_tensor_dram(self.LAYER_SIZE * self.MAX_CONTEXT_SIZE * self.k_size)
        self.LAYER0_K_ROPE_DRAM = self.allocate_tensor_dram(self.LAYER_SIZE * self.MAX_CONTEXT_SIZE * self.k_size)
        zero_pad = torch.zeros(self.LAYER_SIZE * self.MAX_CONTEXT_SIZE * self.k_size, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_K_ROPE_DRAM, zero_pad)
        zero_add = torch.zeros(seq_len * self.head_dim * self.bytes_per_element, dtype=torch.bfloat16)
        self.ZERO_DRAM_ADDR = self.allocate_tensor_dram(seq_len * self.head_dim * self.bytes_per_element)
        self.dma_to_accelerator_memory(self.ZERO_DRAM_ADDR, zero_add)
        self.IDENTITY_DRAM_ADDR = self.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * self.bytes_per_element)
        self.dma_to_accelerator_memory(self.IDENTITY_DRAM_ADDR, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))
        self.LAYER0_FLASH_Q_DRAM = self.allocate_tensor_dram(aligned_seq_len * self.head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_K_DRAM = self.allocate_tensor_dram(aligned_seq_len * self.head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_V_DRAM = self.allocate_tensor_dram(aligned_seq_len * self.head_dim * self.bytes_per_element)
        zero_pad = torch.zeros(aligned_seq_len * self.head_dim * self.bytes_per_element, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_Q_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_K_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_V_DRAM, zero_pad)
        self.LAYER0_INPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_Q_DRAM = self.allocate_tensor_dram(seq_len * self.q_size)
        self.LAYER0_K_DRAM = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_K_NORM_DRAM = self.LAYER0_K_DRAM
        self.LAYER0_Q_NORM_DRAM = self.LAYER0_Q_DRAM
        self.LAYER0_V_PROJ_TEMP = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_FLASH_OUT_HEAD_DRAM = self.allocate_tensor_dram(aligned_seq_len * self.actual_head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.head_dim * self.group_size * self.bytes_per_element)
        self.LAYER0_FLASH_SCRATCH_DRAM = self.allocate_tensor_dram(max(self.head_dim, UE_FMAX_CONTEXT_SIZE) * aligned_seq_len * 2 + self.head_dim * aligned_seq_len * 2)
        self.LAYER0_FLASH_BIAS_DRAM = self.allocate_tensor_dram(aligned_seq_len * aligned_seq_len * self.bytes_per_element)
        self.LAYER0_ATTN_PROJ_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_ATTN_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_ATTN_RESIDUAL_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_MLP_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_MLP_GATE_DRAM = self.allocate_tensor_dram(seq_len * self.mlp_elements * 2)
        self.LAYER0_MLP_UP_DRAM = self.allocate_tensor_dram(seq_len * self.mlp_elements * 2)
        self.LAYER0_MLP_MULT_DRAM = self.allocate_tensor_dram(seq_len * self.mlp_elements * 2)
        self.LAYER0_MLP_DOWN_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_MLP_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.OUTPUT_NORM_DRAM = self.allocate_tensor_dram(1 * self.vector_length * self.bytes_per_element)
        self.LOGITS_DRAM = self.allocate_tensor_dram(1 * self.EMBEDDING_ELEMENTS * self.bytes_per_element)
        _original_print(f"    Tensor dram end: 0x{self.get_tensor_dram_addr():X}, usage: {self.get_tensor_dram_usage()} bytes")

    def program_execute(self, program_start_addr: int, timeout: float = 10.0, gflops: float = None) -> None:
        self.start_execute_from_dram(program_start_addr)
        self.wait_queue(timeout)
        latency = self.report_latency_in_us()
        _original_print(f"    Total program execution latency = {latency} us")
        if gflops is not None:
            gflops_program, _ = self.report_flop_rate_gflops(gflops)
            _original_print(f"    GFLOPS: {gflops_program:.2f}")

    def run_decoder(self, decoder_program_sizes: list[int], decoder_base_addr: int, token_id: int,
                    gflops_per_token: list[int] | None = None) -> int:
        if token_id is None:
            _original_print("No last token available for decode.")
            return 0
        _llama_stop_tokens = {128001, 128008, self._end_of_turn_token_id}
        global _SILENT_MODE
        max_seq_len = self.MAX_CONTEXT_SIZE
        while self.seq_len < max_seq_len:
            _SILENT_MODE = True
            self.seq_len += 1
            aligned_seq_len = ((self.seq_len + 63) // 64) * 64
            prog_idx = min((self.seq_len - 1) // 64, 7)
            prog_addr = decoder_base_addr + sum(decoder_program_sizes[:prog_idx])
            _kv_stride = self.actual_head_dim * self.bytes_per_element
            _rope_row  = self.head_dim * 2 * self.bytes_per_element
            self.isa_add_set_core(self.V_CACHE_SIZE_REG, ue_35bit_addr_shifter((self.seq_len - 1) * _kv_stride))
            self.isa_add_set_core(self.ROPE_SIZE_REG,    ue_35bit_addr_shifter((self.seq_len - 1) * _rope_row))
            embedding_tensor = self.get_embedding_for_tokens([token_id])
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
            bias_host = torch.full((1, aligned_seq_len), -1e36, dtype=torch.bfloat16)
            bias_host[0, :self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_host)
            self.start_execute_from_dram(prog_addr)
            self.wait_queue(10.0)
            token_id = self.get_arg_max_index()
            token_char = self.tokenizer.decode([token_id])
            _SILENT_MODE = False
            if token_id in _llama_stop_tokens:
                _original_print(f"\nStop token {token_id} reached.")
                break
            _original_print(token_char, end="", flush=True)

        return self.seq_len


def _progress_timer(label: str, start_time: float, stop_event: threading.Event) -> None:
    while not stop_event.wait(1.0):
        _original_print(f"\r  {label} ({time.perf_counter() - start_time:.0f}s)", end="", flush=True)


def check_bins_early(script_dir: str) -> list[str]:
    bin_dir = os.path.join(script_dir, "llama3.2_1b_bin")
    missing = []
    for name in ["prefill_program.bin", "prefill_program.json", "decoder_program.bin", "decoder_program.json"]:
        p = os.path.join(bin_dir, name)
        if not os.path.exists(p):
            missing.append(os.path.join("llama3.2_1b_bin", name))
    return missing


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Llama-3.2-1B inference from pre-compiled bins.")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--local-weights", action="store_true")
    parser.add_argument("--dev", type=str, default="xdma0")
    parser.add_argument("--cycle", type=float, default=1/0.17)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    missing = check_bins_early(script_dir)
    if missing:
        _original_print("Missing bin files (run python llama3.2_1b_test.py first):")
        for f in missing:
            _original_print(f"  {f}")
        return

    set_dma_device(args.dev)
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle
    _original_print(f"Llama-3.2-1B on {args.dev} (from pre-compiled bins)")

    weights_bin = ("llama3.2_1b_bin/full_model_weights.bin" if args.local_weights
                   else "llama3.2_1b_bin/weights_llama3.2_1b_hf.bin")

    t0 = time.perf_counter()
    _original_print("Loading weights...")
    ue = Llama32_1b_UnifiedEngine(script_dir=script_dir, weights_bin=weights_bin)
    _original_print(f"  Weights + tensors: {time.perf_counter() - t0:.2f}s")

    cfg = _load_config(script_dir)
    if args.prompt is not None:
        conversation = [{"role": "user", "content": args.prompt}]
        prompt_with_template = ue.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        prefill_seq = tuple(ue.tokenizer.encode(prompt_with_template, add_special_tokens=False))
        _original_print(f"Prompt ({len(prefill_seq)} tokens): {args.prompt!r}")
    else:
        prefill_seq = tuple(cfg["default_prefill_tokens"])
        _original_print(f"Default prompt ({len(prefill_seq)} tokens)")

    # Load prefill bins and pick bucket
    prefill_bin_path  = os.path.join(script_dir, "llama3.2_1b_bin", "prefill_program.bin")
    prefill_meta_path = os.path.join(script_dir, "llama3.2_1b_bin", "prefill_program.json")
    with open(prefill_meta_path) as f:
        prefill_meta = json.load(f)
    prefill_buckets = prefill_meta["buckets"]
    prefill_program_sizes = prefill_meta.get("program_sizes", [c * 32 for c in prefill_meta["instruction_counts"]])
    gflops_prefill_list = prefill_meta["total_flops"]

    actual_seq_len = len(prefill_seq) - 1  # last token is first decoder input
    bucket_idx = next(i for i, b in enumerate(prefill_buckets) if b - 1 >= actual_seq_len)
    bucket_seq_len = prefill_buckets[bucket_idx] - 1
    gflops_prefill = gflops_prefill_list[bucket_idx]

    _original_print(f"\n--- Loading programs ---")
    t_load = time.perf_counter()
    prefill_base_addr, _ = ue.load_instructions(prefill_bin_path)
    prefill_program_addr = prefill_base_addr + sum(prefill_program_sizes[:bucket_idx])
    # seq_len = actual (not bucket) so decoder starts at the right position
    ue.seq_len = actual_seq_len

    decoder_bin_path  = os.path.join(script_dir, "llama3.2_1b_bin", "decoder_program.bin")
    decoder_meta_path = os.path.join(script_dir, "llama3.2_1b_bin", "decoder_program.json")
    with open(decoder_meta_path) as f:
        decoder_meta = json.load(f)
    decoder_program_sizes = decoder_meta.get("program_sizes", [c * 32 for c in decoder_meta["instruction_counts"]])
    gflops_per_token = decoder_meta["total_flops"]
    decoder_base_addr, _ = ue.load_instructions(decoder_bin_path)
    _original_print(f"  Programs loaded: {time.perf_counter() - t_load:.3f}s")

    _original_print(f"\n--- Prefill ({actual_seq_len} tokens, bucket={bucket_seq_len}) ---")
    t_prefill = time.perf_counter()
    stop = threading.Event()
    timer = threading.Thread(target=_progress_timer, args=("Prefill running...", t_prefill, stop), daemon=True)
    timer.start()

    embedding_tensor = ue.get_embedding_for_tokens(list(prefill_seq[:-1]))
    if actual_seq_len < bucket_seq_len:
        pad = embedding_tensor[-1:].repeat(bucket_seq_len - actual_seq_len, 1)
        embedding_tensor = torch.cat([embedding_tensor, pad], dim=0)
    q_seq_len = bucket_seq_len * ue.group_size
    aligned_seq_len = ((q_seq_len + 63) // 64) * 64
    ue.dma_to_accelerator_memory(ue.LAYER0_INPUT_DRAM, embedding_tensor)
    bias_one_group = torch.full((aligned_seq_len, aligned_seq_len), float("-inf"), dtype=torch.bfloat16)
    rows = torch.arange(aligned_seq_len).unsqueeze(1)
    cols = torch.arange(aligned_seq_len).unsqueeze(0)
    valid_mask = (cols // ue.group_size) <= (rows // ue.group_size)
    bias_one_group.masked_fill_(valid_mask, 0.0)
    bias_one_group[:, q_seq_len:] = float("-inf")
    ue.dma_to_accelerator_memory(ue.LAYER0_FLASH_BIAS_DRAM, bias_one_group)
    ue.program_execute(prefill_program_addr, gflops=gflops_prefill)

    # Zero KV cache padding positions so they don't corrupt decoder attention
    if actual_seq_len < bucket_seq_len:
        zero_vec = torch.zeros(ue.actual_head_dim, dtype=torch.bfloat16)
        bpe = ue.bytes_per_element
        ahd = ue.actual_head_dim
        for layer_idx in range(ue.LAYER_SIZE):
            for kv_h in range(ue.num_kv_heads):
                k_base = (ue.LAYER0_K_ROPE_DRAM + layer_idx * ue.MAX_CONTEXT_SIZE * ue.k_size + kv_h * ue.MAX_CONTEXT_SIZE * ahd * bpe)
                v_base = (ue.LAYER0_V_DRAM + layer_idx * ue.MAX_CONTEXT_SIZE * ue.k_size + kv_h * ue.MAX_CONTEXT_SIZE * ahd * bpe)
                for t in range(actual_seq_len, bucket_seq_len):
                    ue.dma_to_accelerator_memory(k_base + t * ahd * bpe, zero_vec)
                    ue.dma_to_accelerator_memory(v_base + t * ahd * bpe, zero_vec)

    stop.set(); timer.join()
    _original_print(f"\r  Prefill: {time.perf_counter() - t_prefill:.3f}s")

    _original_print("\n--- Decoding ---")
    t_decode = time.perf_counter()
    token_cnt = ue.run_decoder(decoder_program_sizes, decoder_base_addr, token_id=prefill_seq[-1], gflops_per_token=gflops_per_token)
    latency_decode = time.perf_counter() - t_decode
    _original_print(f"\n  Decode: {latency_decode:.2f}s, {token_cnt} tokens total.")

    # Trailing software_reset so the next process starts with a quiesced queue.
    global _SILENT_MODE
    _SILENT_MODE = True
    ue.software_reset()


if __name__ == "__main__":
    main()
