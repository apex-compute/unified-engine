#!/usr/bin/env python3
"""Parakeet-TDT-0.6B inference from pre-compiled bin files. Run parakeet_test.py first to generate bins."""
import json
import math
import os
import builtins
import struct
import sys
import time
import threading

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

import numpy as np
import torch

# Silent mode: suppress internal prints during inference
_original_print = builtins.print
_SILENT_MODE = False

def quiet_print(*args, **kwargs):
    if _SILENT_MODE:
        return
    _original_print(*args, **kwargs)

builtins.print = quiet_print

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, DMA_DEVICE_C2H, DRAM_INSTRUCTION_ADDR, TYPE, UE_VECTOR_SIZE,
    URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS, UnifiedEngine, set_dma_device,
    UE_MODE, BROADCAST_MODE, LALU_MODE, MEMCPY_TYPE, URAM_SECTION, UE_ARGMAX_INDEX,
    ue_35bit_addr_shifter
)

URAM_A_BASE = 0x00000
URAM_B_BASE = 0x80000

WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "parakeet_bin", "model_weights.ckpt")
TOKENIZER_PATH = os.path.join(SCRIPT_DIR, "parakeet_bin", "tokenizer.model")
BIN_DIR = os.path.join(SCRIPT_DIR, "parakeet_bin")

# ---------------------------------------------------------------------------
# Config / utility
# ---------------------------------------------------------------------------
def load_config(config_path=None):
    """Load parakeet_config.json."""
    if config_path is None:
        config_path = os.path.join(SCRIPT_DIR, "parakeet_config.json")
    with open(config_path, "r") as f:
        return json.load(f)

def pad_to_multiple(n, multiple):
    return ((n + multiple - 1) // multiple) * multiple

def conv2d_outsize(H, k=3, s=2, p=1):
    """Output spatial dim for conv2d/pool with given kernel, stride, padding."""
    return (H + 2 * p - k) // s + 1

def read_dram(engine, addr, numel):
    """Read bf16 tensor from accelerator DRAM."""
    buf = torch.zeros(numel, dtype=torch.bfloat16)
    engine.dma_read(DMA_DEVICE_C2H, addr, buf, numel * engine.bytes_per_element)
    return buf

def frame_waveform(waveform, cfg):
    """Frame waveform for HW mel: center-pad + overlapping windows → (T_mel, n_fft) bf16.

    Replicates torch.stft framing (center=True, pad_mode='reflect') without the FFT.
    Window is already absorbed into HW DFT coefficients (DFT_COS / DFT_SIN).
    """
    pre = cfg["preprocessing"]
    n_fft = pre["n_fft"]           # 512
    hop_length = pre["hop_length"] # 160
    pad = n_fft // 2
    sig = waveform.float()[0]                                        # (samples,)
    padded = torch.nn.functional.pad(sig.unsqueeze(0), (pad, pad), mode="reflect").squeeze(0)
    frames = padded.unfold(0, n_fft, hop_length)                     # (T_mel, n_fft)
    return frames.to(torch.bfloat16).contiguous()

def allocate_identity(ue, N):
    """Allocate and write an (N, N) bf16 identity matrix to DRAM. Returns DRAM address."""
    assert N % UE_VECTOR_SIZE == 0, f"N={N} must be a multiple of {UE_VECTOR_SIZE}"
    identity = torch.eye(N, dtype=torch.bfloat16).contiguous()
    addr = ue.get_params_dram_addr()
    ue.allocate_params_dram(N * N * 2)
    ue.dma_write(DMA_DEVICE_H2C, addr, identity, N * N * 2)
    return addr

# ---------------------------------------------------------------------------
# Parakeet DRAM partition over 4GB address space:
#   2.5 GB params / 750 MB tensors / 750 MB programs
PARAKEET_PARAMS_BASE  = 0x00000000   # 3 GB for weights + identities + Toeplitz DW conv matrices
PARAKEET_TENSOR_BASE  = 0xC0000000   # 768 MB for intermediate activations + im2col temp buffers
PARAKEET_PROGRAM_BASE = 0xF0000000   # 256 MB for compiled instruction programs

class Parakeet_UnifiedEngine(UnifiedEngine):
    """UnifiedEngine subclass for Parakeet-TDT-0.6B."""

    def __init__(self, script_dir=None, clock_period_ns=None, engine_slave=False):
        program_base = PARAKEET_PROGRAM_BASE + 0x08000000 if engine_slave else PARAKEET_PROGRAM_BASE
        engine_base = user_dma_core.UE_0_BASE_ADDR + 0x00010000 if engine_slave else user_dma_core.UE_0_BASE_ADDR
        super().__init__(BASE_ADDR=engine_base, params_dram_base=PARAKEET_PARAMS_BASE, tensor_dram_base=PARAKEET_TENSOR_BASE, program_dram_base=program_base, clock_period_ns=clock_period_ns)
        self.engine_slave = engine_slave
        self.script_dir = script_dir or SCRIPT_DIR
        # Hang prevention: stop stale execution, write HALT to program base
        self.dram_inst_running(False)
        self.start_capture()
        self.generate_instruction_halt()
        self.stop_capture()
        halt_bytes = bytearray()
        for inst in self.capture_buffer:
            halt_bytes.extend(inst.get_bytes())
        self.dma_write(DMA_DEVICE_H2C, program_base, halt_bytes, len(halt_bytes))
        self.clear_capture_buffer()
        self._cfg = load_config()
        enc = self._cfg["encoder"]
        pred = self._cfg["predictor"]
        jnt = self._cfg["joint"]
        hw = self._cfg["hardware"]
        self.d_model = enc["d_model"]           # 1024
        self.num_layers = enc["num_layers"]     # 24
        self.num_heads = enc["num_heads"]       # 8
        self.head_dim = enc["head_dim"]         # 128
        self.ff_dim = enc["ff_dim"]             # 4096
        self.conv_kernel = enc["conv_kernel"]   # 9
        self.conv_pad = enc["conv_pad"]         # 4
        self.sub_channels = enc["sub_channels"] # 256
        self.n_mels = enc["n_mels"]             # 128
        self.pred_hidden = pred["hidden_size"]  # 640
        self.vocab_size = pred["vocab_size"]    # 8193
        self.joint_hidden = jnt["hidden_size"]  # 640
        self.joint_output_padded = jnt["output_size_padded"]
        self.blank_id = jnt["blank_id"]
        self.tdt_durations = jnt["tdt_durations"]
        self.max_symbols_per_step = jnt["max_symbols_per_step"]
        self.block_size = hw["block_size"]      # 64
        self.bytes_per_element = enc["bytes_per_element"]
        pre = self._cfg["preprocessing"]
        self.n_fft = pre["n_fft"]                                     # 512
        self.n_bins_pad = pad_to_multiple(pre["n_fft"] // 2 + 1, hw["block_size"])  # 320
        # ISA register assignments for decoder dynamic addressing
        regs = self._cfg.get("fixed_isa_regs", {})
        self.TOKEN_REG = regs.get("TOKEN_REG", 1)
        self.ENC_T_REG = regs.get("ENC_T_REG", 2)
        self.TMP_REG = regs.get("TMP_REG", 3)

    def copy_dram_layout(self, source):
        """Copy weight and tensor DRAM addresses from master engine for dual-engine mode."""
        if hasattr(source, 'w'):
            self.w = source.w
        if hasattr(source, 'layer_addrs'):
            self.layer_addrs = source.layer_addrs
        for attr in dir(source):
            if attr.endswith('_DRAM') and isinstance(getattr(source, attr), int):
                setattr(self, attr, getattr(source, attr))

    def isa_add_set_core(self, dst_reg_idx, immediate_value, timeout_s=10.0):
        """Set one ISA register to an immediate value via minimal program execution."""
        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_add_set(dst_reg_idx, immediate_value)
        self.generate_instruction_halt()
        self.stop_capture()
        prog = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(prog)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        self.clear_capture_buffer()
        self.start_execute_from_dram(prog)
        self.wait_queue(timeout_s)

    # XDMA driver has a transfer size limit (~16-64MB per os.write call).
    # Chunk large DMA uploads to avoid silent failures.
    DMA_CHUNK_BYTES = 1 * 1024 * 1024  # 1 MB per chunk

    def dma_to_accelerator_memory(self, dma_address, data):
        """Chunked DMA write that handles large transfers."""
        assert data.dtype == torch.bfloat16, "Data must be in bf16 format"
        flat = data.contiguous().flatten()
        total_bytes = flat.numel() * 2
        if total_bytes <= self.DMA_CHUNK_BYTES:
            self.dma_write(DMA_DEVICE_H2C, dma_address, flat, total_bytes)
            return
        elems_per_chunk = self.DMA_CHUNK_BYTES // 2
        offset = 0
        while offset < flat.numel():
            chunk_elems = min(elems_per_chunk, flat.numel() - offset)
            chunk = flat[offset:offset + chunk_elems].contiguous()
            addr = dma_address + offset * 2
            self.dma_write(DMA_DEVICE_H2C, addr, chunk, chunk_elems * 2)
            offset += chunk_elems

    def tensor_init(self, L_pad):
        """Allocate intermediate DRAM buffers."""
        # Zero entire tensor DRAM region up front to prevent stale NaN.
        tensor_budget = self._program_dram_base - self._tensor_dram_base
        print(f"  Zeroing tensor DRAM ({tensor_budget / 1024**2:.0f} MB budget)...")
        ZERO_CHUNK = 512 * 1024  # 512K elements = 1MB per chunk
        offset = 0
        total_elems = tensor_budget // 2
        while offset < total_elems:
            chunk_elems = min(ZERO_CHUNK, total_elems - offset)
            z = torch.zeros(chunk_elems, dtype=torch.bfloat16)
            self.dma_to_accelerator_memory(self._tensor_dram_base + offset * 2, z)
            offset += chunk_elems
        print(f"  Tensor DRAM zeroed.")

        bpe = self.bytes_per_element
        D, FF, H = self.d_model, self.ff_dim, self.pred_hidden
        dk = self.head_dim
        SC = self.sub_channels  # 256
        P_pad = pad_to_multiple(2 * L_pad - 1, self.block_size)
        N_tok_pad = pad_to_multiple(self.vocab_size, self.block_size)
        N_dur_pad = pad_to_multiple(len(self.tdt_durations), self.block_size)
        # Encoder intermediates
        self.INPUT_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.LN_OUT_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.FF_MID_DRAM = self.allocate_tensor_dram(L_pad * FF * bpe)
        self.FF_OUT_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.RESIDUAL_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.Q_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.K_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.V_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.POS_PROJ_DRAM = self.allocate_tensor_dram(P_pad * D * bpe)  # P_pad rows, not L_pad
        self.SCORE_DRAM = self.allocate_tensor_dram(L_pad * L_pad * bpe)
        self.ATTN_OUT_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.POS_EMB_DRAM = self.allocate_tensor_dram(P_pad * D * bpe)
        self.REL_SHIFT_DRAM = self.allocate_tensor_dram(L_pad * L_pad * bpe)
        self.CONV_A_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.CONV_B_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.CONV_T_DRAM = self.allocate_tensor_dram(D * L_pad * bpe)
        self.CONV_DW_DRAM = self.allocate_tensor_dram(D * L_pad * bpe)
        self.CONV_OUT_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.ATTN_VT_DRAM = self.allocate_tensor_dram(dk * L_pad * bpe)
        self.ATTN_MASK_DRAM = self.allocate_tensor_dram(L_pad * L_pad * bpe)
        self.PERMUTE_TEMP_DRAM = self.allocate_tensor_dram(D * L_pad * bpe)
        self.ENC_OUT_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        # L_pad-sized identity for softmax and SiLU in conv module
        self.IDENTITY_LPAD_DRAM = allocate_identity(self, L_pad)
        # Subsampling intermediates
        T_mel_max = L_pad * 8
        self.MEL_DRAM = self.allocate_tensor_dram(T_mel_max * self.n_mels * bpe)
        # HW mel pipeline buffers: must be allocated even in CPU-mel mode to keep
        # subsequent tensor addresses aligned with the compiled programs.
        K_padded = pad_to_multiple(257, self.block_size)  # 320
        self.POWER_DRAM = self.allocate_tensor_dram(T_mel_max * K_padded * bpe)
        self.MEL_TEMP_DRAM = self.allocate_tensor_dram(self.n_mels * T_mel_max * bpe)
        T_mel_max_padded = pad_to_multiple(T_mel_max, self.block_size)
        self.ZEROS_DRAM = self.allocate_tensor_dram(T_mel_max_padded * bpe)
        self.MASK_DRAM = self.allocate_tensor_dram(T_mel_max_padded * bpe)
        # Temp buffer for R_combined (stage 0 im2col row selection)
        H0_max = (T_mel_max + 2 - 3) // 2 + 1
        self.IM2COL_R_DRAM = self.allocate_tensor_dram(H0_max * 3 * self.n_mels * bpe)
        H0, W0 = T_mel_max // 2, self.n_mels // 2  # after stage 0
        H1, W1 = H0 // 2, W0 // 2                  # after stage 1
        N0 = H0 * W0
        N1 = H1 * W1
        N1_pad = pad_to_multiple(N1, self.block_size)
        H2, W2_tmp = H1 // 2, W1 // 2
        N2 = H2 * W2_tmp
        N2_pad = pad_to_multiple(N2, self.block_size)
        N_dw_max_pad = max(N1_pad, N2_pad)
        max_patch_size = max(N0 * 64, SC * N_dw_max_pad * 64)
        self.SUB_PATCH_DRAM = self.allocate_tensor_dram(max_patch_size * bpe)
        # Temp buffers for depthwise im2col (stages 1, 2)
        max_N_dw_in = max(N0, N1_pad)  # max spatial positions for DW input
        max_M_aligned = pad_to_multiple(max_N_dw_in, self.block_size)
        self.IM2COL_TRANSPOSE_DRAM = self.allocate_tensor_dram(max_M_aligned * SC * bpe)
        self.IM2COL_PERMUTE_TEMP_DRAM = self.allocate_tensor_dram(
            SC * pad_to_multiple(max_N_dw_in, self.block_size) * bpe)
        # R_all: SC * H_out_pad * K_g per DW stage (K_g = pad(3*W_in, 64))
        H1_pad_max = N1_pad // W1 if W1 > 0 else 0
        H2_pad_max = N2_pad // W2_tmp if W2_tmp > 0 else 0
        K_g_s1 = pad_to_multiple(3 * W0, self.block_size)
        K_g_s2 = pad_to_multiple(3 * W1, self.block_size)
        r_all_max = max(SC * H1_pad_max * K_g_s1, SC * H2_pad_max * K_g_s2) if W1 > 0 else 0
        self.IM2COL_R_DW_DRAM = self.allocate_tensor_dram(r_all_max * bpe)
        self.SUB_OUT0_DRAM = self.allocate_tensor_dram(N0 * SC * bpe)
        self.SUB_DW_OUT_DRAM = self.allocate_tensor_dram(SC * N_dw_max_pad * bpe)
        self.SUB_PW_IN_DRAM = self.allocate_tensor_dram(N_dw_max_pad * SC * bpe)
        self.SUB_FLAT_DRAM = self.allocate_tensor_dram(L_pad * 4096 * bpe)
        # Decoder intermediates
        self.PRED_EMB_DRAM = self.allocate_tensor_dram(H * bpe)
        self.PRED_GATES_DRAM = self.allocate_tensor_dram(4 * H * bpe)
        self.PRED_GATES2_DRAM = self.allocate_tensor_dram(4 * H * bpe)
        self.PRED_H0_DRAM = self.allocate_tensor_dram(H * bpe)
        self.PRED_C0_DRAM = self.allocate_tensor_dram(H * bpe)
        self.PRED_H1_DRAM = self.allocate_tensor_dram(H * bpe)
        self.PRED_C1_DRAM = self.allocate_tensor_dram(H * bpe)
        self.PRED_OUT_DRAM = self.allocate_tensor_dram(H * bpe)
        # On-device LSTM state save buffers (for blank rollback without host DMA)
        self.SAVED_H0_DRAM = self.allocate_tensor_dram(H * bpe)
        self.SAVED_C0_DRAM = self.allocate_tensor_dram(H * bpe)
        self.SAVED_H1_DRAM = self.allocate_tensor_dram(H * bpe)
        self.SAVED_C1_DRAM = self.allocate_tensor_dram(H * bpe)
        self.JOINT_ENC_DRAM = self.allocate_tensor_dram(D * bpe)
        self.JOINT_PRED_DRAM = self.allocate_tensor_dram(self.joint_hidden * bpe)
        self.JOINT_SUM_DRAM = self.allocate_tensor_dram(self.joint_hidden * bpe)
        self.JOINT_TOK_DRAM = self.allocate_tensor_dram(N_tok_pad * bpe)
        self.JOINT_DUR_DRAM = self.allocate_tensor_dram(N_dur_pad * bpe)
        tensor_used = self.get_tensor_dram_usage()
        tensor_limit = self._program_dram_base - self._tensor_dram_base
        print(f"  Tensor alloc: {tensor_used / 1024**2:.1f} MB "
              f"(budget: {tensor_limit / 1024**2:.0f} MB)")
        assert tensor_used <= tensor_limit, (
            f"TENSOR DRAM OVERFLOW: {tensor_used/1024**2:.1f} MB used > "
            f"{tensor_limit/1024**2:.0f} MB budget. "
            f"Tensors bleed into program region!"
        )

    def load_params(self, L_pad):
        """Load params DRAM from bin. Returns True if loaded, False if not found/mismatch."""
        bin_dir = os.path.join(self.script_dir, "parakeet_bin")
        bin_path = os.path.join(bin_dir, "params.bin")
        meta_path = os.path.join(bin_dir, "params.json")
        if not os.path.exists(bin_path) or not os.path.exists(meta_path):
            return False
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("L_pad") != L_pad:
            _original_print(f"  Params bin L_pad={meta.get('L_pad')} != {L_pad}, reloading weights")
            return False
        total = meta["size"]
        CHUNK = 1 * 1024 * 1024
        with open(bin_path, "rb") as f:
            offset = 0
            while offset < total:
                data = f.read(min(CHUNK, total - offset))
                self.dma_write(DMA_DEVICE_H2C, self._params_dram_base + offset, data, len(data))
                offset += len(data)
        self.allocate_params_dram(total)
        _original_print(f"  Loading Toeplitz Staged BF16 Weights... {total / 1024**2:.1f} MB from bin")
        return True

    def load_programs(self, L_pad):
        """Load compiled programs from bin. Returns dict of {name: dram_addr} or None if not found."""
        bin_dir = os.path.join(self.script_dir, "parakeet_bin")
        suffix = "_slave" if self.engine_slave else ""
        bin_path = os.path.join(bin_dir, f"programs{suffix}.bin")
        meta_path = os.path.join(bin_dir, f"programs{suffix}.json")
        if not os.path.exists(bin_path) or not os.path.exists(meta_path):
            return None
        with open(meta_path) as f:
            manifest = json.load(f)
        if manifest.get("L_pad") != L_pad:
            _original_print(f"  Bin L_pad={manifest.get('L_pad')} != current {L_pad}, recompiling")
            return None
        with open(bin_path, "rb") as f:
            all_bytes = f.read()
        addrs = {}
        for name, meta in manifest["programs"].items():
            data = all_bytes[meta["offset"]:meta["offset"] + meta["size"]]
            addr = self.get_program_dram_addr()
            alignment_gap = (-addr) % 64
            self.dma_write(DMA_DEVICE_H2C, addr, data, len(data))
            self.allocate_program_dram(len(data) - alignment_gap)
            addrs[name] = addr
        _original_print(f"  Programs loaded: {len(all_bytes)} bytes from bin")
        return addrs

    def make_rel_pos_emb(self, seq_len):
        """Generate relative positional encoding: (2*seq_len-1, D) bf16."""
        D = self.d_model
        max_len = 2 * seq_len
        pe = torch.zeros(max_len, D)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, D, 2, dtype=torch.float32) * -(math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe_pos = pe[:seq_len]
        pe_neg = torch.flip(pe[1:seq_len], [0])
        return torch.cat([pe_neg, pe_pos], dim=0).to(torch.bfloat16)

    def program_execute(self, program_addr, timeout=50.0, flops=None):
        """Execute compiled program. Returns (latency_us, gflops)."""
        self.start_execute_from_dram(program_addr)
        latency_us, gflops = 0, 0
        if timeout > 0:
            self.wait_queue(timeout)
            latency_us = self.report_latency_in_us()
            if flops:
                gflops = self.report_flop_rate_gflops(flops)
        return latency_us, gflops

    def get_arg_max_index(self):
        """Read hardware argmax register."""
        return self.read_reg32(UE_ARGMAX_INDEX)

    def run_decode(self, enc_out_addr, L):
        """TDT greedy decode with on-device DMA. Host only sets registers and dispatches programs."""
        bpe = self.bytes_per_element
        H = self.pred_hidden
        D = self.d_model
        pred_prog = self.progs["pred"][0]
        tok_prog = self.progs["joint_tok"][0]
        dur_prog = self.progs["joint_dur"][0]
        restore_prog = self.progs["state_restore"][0]
        # Pre-zero all decoder intermediate buffers
        zeros_h = torch.zeros(H, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.PRED_H0_DRAM, zeros_h)
        self.dma_to_accelerator_memory(self.PRED_C0_DRAM, zeros_h)
        self.dma_to_accelerator_memory(self.PRED_H1_DRAM, zeros_h)
        self.dma_to_accelerator_memory(self.PRED_C1_DRAM, zeros_h)
        self.dma_to_accelerator_memory(self.PRED_EMB_DRAM, zeros_h)
        self.dma_to_accelerator_memory(self.PRED_OUT_DRAM, zeros_h)
        self.dma_to_accelerator_memory(self.PRED_GATES_DRAM, torch.zeros(4 * H, dtype=torch.bfloat16))
        self.dma_to_accelerator_memory(self.PRED_GATES2_DRAM, torch.zeros(4 * H, dtype=torch.bfloat16))
        self.dma_to_accelerator_memory(self.JOINT_ENC_DRAM, torch.zeros(D, dtype=torch.bfloat16))
        self.dma_to_accelerator_memory(self.JOINT_PRED_DRAM, zeros_h)
        self.dma_to_accelerator_memory(self.JOINT_SUM_DRAM, zeros_h)
        N_tok_pad = pad_to_multiple(self.vocab_size, self.block_size)
        N_dur_pad = pad_to_multiple(len(self.tdt_durations), self.block_size)
        self.dma_to_accelerator_memory(self.JOINT_TOK_DRAM, torch.zeros(N_tok_pad, dtype=torch.bfloat16))
        self.dma_to_accelerator_memory(self.JOINT_DUR_DRAM, torch.zeros(N_dur_pad, dtype=torch.bfloat16))
        tokens = []
        t = 0
        last_token = self.blank_id
        total_steps = 0
        while t < L:
            symbols = 0
            while symbols < self.max_symbols_per_step:
                # Set registers: TOKEN_REG = embedding offset, ENC_T_REG = encoder position offset
                self.isa_add_set_core(self.TOKEN_REG, ue_35bit_addr_shifter(last_token * H * bpe))
                self.isa_add_set_core(self.ENC_T_REG, ue_35bit_addr_shifter(t * D * bpe))
                # Predictor: state save + embedding lookup (register-addressed) + LSTM
                self.program_execute(pred_prog)
                # Joint token: enc_out[t] copy (register-addressed) + pred_out copy + projections + argmax
                self.program_execute(tok_prog)
                token_id = self.get_arg_max_index()
                # Joint duration -> hardware argmax
                self.program_execute(dur_prog)
                dur_idx = self.get_arg_max_index()
                dur = self.tdt_durations[dur_idx] if dur_idx < len(self.tdt_durations) else 0
                total_steps += 1

                if token_id == self.blank_id:
                    # Restore LSTM state from on-device save buffers
                    self.program_execute(restore_prog)
                    t += max(dur, 1)
                    break
                else:
                    tokens.append(token_id)
                    last_token = token_id
                    symbols += 1
                    if dur > 0:
                        t += dur
                        break
            else:
                t += 1
        _original_print(f"  Decode: {total_steps} joint steps, {len(tokens)} tokens emitted")
        return tokens


def check_bins_early():
    missing = []
    for name in ("params.bin", "programs.bin", "mel_fb.npy", "mel_window.npy"):
        if not os.path.exists(os.path.join(BIN_DIR, name)):
            missing.append(name)
    return missing


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parakeet-TDT-0.6B inference from pre-compiled bins")
    parser.add_argument("--audio", type=str, default=None)
    parser.add_argument("--dev", type=str, default="xdma0")
    parser.add_argument("--cycle", type=float, default=5.63)
    parser.add_argument("--max-seconds", type=float, default=None)
    args = parser.parse_args()

    missing = check_bins_early()
    if missing:
        _original_print("Missing bin files (run parakeet_test.py first to compile):")
        for f in missing:
            _original_print(f"  {os.path.join(BIN_DIR, f)}")
        return

    cfg = load_config()
    set_dma_device(args.dev)
    # Refresh local bindings shadowed at import time so DMA goes to the right device
    import sys as _sys, user_dma_core as _udc
    _mod = _sys.modules[__name__]
    _mod.DMA_DEVICE_H2C = _udc.DMA_DEVICE_H2C
    _mod.DMA_DEVICE_C2H = _udc.DMA_DEVICE_C2H
    _mod.DMA_DEVICE_USER = _udc.DMA_DEVICE_USER
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle

    audio_path = args.audio or os.path.join(SCRIPT_DIR, cfg["defaults"]["default_audio"])
    assert os.path.exists(audio_path), f"Audio file not found: {audio_path}"

    import soundfile as sf
    import torchaudio
    data, sr = sf.read(audio_path, dtype="float32")
    waveform = torch.from_numpy(data).T if data.ndim > 1 else torch.from_numpy(data).unsqueeze(0)
    if sr != cfg["preprocessing"]["sample_rate"]:
        waveform = torchaudio.functional.resample(waveform, sr, cfg["preprocessing"]["sample_rate"])
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if args.max_seconds is not None:
        waveform = waveform[:, :int(args.max_seconds * cfg["preprocessing"]["sample_rate"])]

    audio_dur = waveform.shape[1] / cfg["preprocessing"]["sample_rate"]
    _original_print(f"Parakeet-TDT-0.6B on {args.dev} ({audio_dur:.1f}s audio)")

    global _SILENT_MODE
    _SILENT_MODE = True

    engine = Parakeet_UnifiedEngine(clock_period_ns=args.cycle)

    import numpy as np
    if waveform.shape[0] > 1:
        waveform = waveform[:1, :]  # mono
    n_mels = cfg["encoder"]["n_mels"]
    pre = cfg["preprocessing"]
    fb = torch.from_numpy(np.load(os.path.join(BIN_DIR, "mel_fb.npy")))
    window = torch.from_numpy(np.load(os.path.join(BIN_DIR, "mel_window.npy")))
    stft = torch.stft(waveform.float(), pre["n_fft"], pre["hop_length"], pre["win_length"],
                      window=window, center=True, pad_mode="reflect", return_complex=True)
    power = stft.abs() ** 2
    mel = torch.matmul(fb, power)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    mean = mel.mean(dim=-1, keepdim=True)
    std = torch.clamp(torch.sqrt(mel.var(dim=-1, keepdim=True, unbiased=True)), min=1e-5)
    mel_cpu = ((mel - mean) / std).squeeze(0).transpose(0, 1).to(torch.bfloat16).contiguous()
    T_mel = mel_cpu.shape[0]

    H0, W0 = conv2d_outsize(T_mel), conv2d_outsize(n_mels)
    H1, W1 = conv2d_outsize(H0), conv2d_outsize(W0)
    H2, W2 = conv2d_outsize(H1), conv2d_outsize(W1)
    N0 = H0 * W0
    L = H2
    L_pad = pad_to_multiple(L, cfg["hardware"]["block_size"])

    params_loaded = engine.load_params(L_pad)
    if not params_loaded:
        _original_print(f"Missing bin file (run parakeet_test.py first to compile):")
        _original_print(f"  {os.path.join(BIN_DIR, 'params.bin')} (L_pad={L_pad} mismatch or missing)")
        return

    engine.tensor_init(L_pad)

    D = engine.d_model
    bpe = engine.bytes_per_element
    P_pad = pad_to_multiple(2 * L_pad - 1, engine.block_size)

    rel_pe = engine.make_rel_pos_emb(L_pad)
    if rel_pe.shape[0] < P_pad:
        pe_padded = torch.zeros(P_pad, D, dtype=torch.bfloat16)
        pe_padded[:rel_pe.shape[0], :] = rel_pe
        rel_pe = pe_padded
    engine.dma_to_accelerator_memory(engine.POS_EMB_DRAM, rel_pe.contiguous())
    mask = torch.full((L_pad, L_pad), -1e38, dtype=torch.bfloat16)
    mask[:L, :L] = 0.0
    mask[L:, 0] = 0.0
    engine.dma_to_accelerator_memory(engine.ATTN_MASK_DRAM, mask.contiguous())

    loaded = engine.load_programs(L_pad)
    if not loaded:
        _original_print(f"Missing bin file (run parakeet_test.py first to compile):")
        _original_print(f"  {os.path.join(BIN_DIR, 'programs.bin')} (L_pad={L_pad} mismatch or missing)")
        return

    im2col_s0      = loaded["im2col_s0"]
    prog_s0        = loaded["prog_s0"]
    im2col_s1      = loaded["im2col_s1"]
    prog_s1        = loaded["prog_s1"]
    im2col_s2      = loaded["im2col_s2"]
    prog_s2        = loaded["prog_s2"]
    prog_flatten_lin = loaded["prog_flatten_lin"]
    enc_prog_addr  = loaded["encoder"]
    pred_prog      = loaded["pred"]
    tok_prog       = loaded["tok"]
    dur_prog       = loaded["dur"]
    restore_prog   = loaded["restore"]

    engine.progs = {"pred": (pred_prog, 0), "joint_tok": (tok_prog, 0), "joint_dur": (dur_prog, 0), "state_restore": (restore_prog, 0)}

    import sentencepiece as spm

    t_start = time.perf_counter()
    stop = threading.Event()
    def _progress(label, start):
        while not stop.wait(1.0):
            _original_print(f"\r  {label} ({time.perf_counter() - start:.0f}s)", end="", flush=True)
    threading.Thread(target=_progress, args=("Executing", t_start), daemon=True).start()

    # CPU mel (same pipeline as parakeet_test.py)
    engine.dma_to_accelerator_memory(engine.MEL_DRAM, mel_cpu)

    # Subsampling (single-engine)
    engine.dma_to_accelerator_memory(engine.SUB_OUT0_DRAM, torch.zeros(N0 * engine.sub_channels, dtype=torch.bfloat16))
    engine.program_execute(im2col_s0)
    engine.program_execute(prog_s0)
    engine.program_execute(im2col_s1)
    engine.program_execute(prog_s1)
    engine.program_execute(im2col_s2)
    engine.program_execute(prog_s2)
    engine.dma_to_accelerator_memory(engine.INPUT_DRAM, torch.zeros(L_pad * D, dtype=torch.bfloat16))
    engine.program_execute(prog_flatten_lin)
    if L_pad > H2:
        ef = torch.zeros((L_pad - H2) * D, dtype=torch.bfloat16)
        ef[0::2] = 0.1; ef[1::2] = -0.1
        engine.dma_to_accelerator_memory(engine.INPUT_DRAM + H2 * D * bpe, ef)
    t_sub_done = time.perf_counter()

    # Encoder (single-engine)
    engine.start_execute_from_dram(enc_prog_addr)
    engine.wait_queue(120.0)
    t_enc_done = time.perf_counter()

    # Decoder
    hw_enc_out = read_dram(engine, engine.INPUT_DRAM, L_pad * D)
    engine.dma_to_accelerator_memory(engine.ENC_OUT_DRAM, hw_enc_out.contiguous())
    hw_tokens = engine.run_decode(engine.ENC_OUT_DRAM, L)
    t_dec_done = time.perf_counter()

    stop.set()
    _original_print(f"\r  Executing ({t_dec_done - t_start:.0f}s) done")

    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER_PATH)
    vocab_sz = sp.GetPieceSize()
    hw_text = sp.DecodeIds([t for t in hw_tokens if 0 <= t < vocab_sz])

    _original_print(f"\n  >>> {hw_text}")
    _original_print(f"\n  Timing:")
    _original_print(f"    Subsampling:         {t_sub_done - t_start:.3f}s")
    _original_print(f"    Encoder (24 layers): {t_enc_done - t_sub_done:.3f}s")
    _original_print(f"    Decoder:             {t_dec_done - t_enc_done:.3f}s")
    _original_print(f"    Total:               {t_dec_done - t_start:.3f}s")


if __name__ == "__main__":
    main()
