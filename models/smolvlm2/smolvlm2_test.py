#!/usr/bin/env python3
"""SmolVLM2-500M on accelerator: bf16 (vision) & q4_64 (language). Use --vision-fp4 for FP4 vision."""
import builtins
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

# Suppress prints during decode/compile (same as Gemma3)
_original_print = builtins.print
_SILENT_MODE = False
def _quiet_print(*args, **kwargs):
    if not _SILENT_MODE:
        _original_print(*args, **kwargs)
builtins.print = _quiet_print
import torch
from huggingface_hub import snapshot_download

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, DMA_DEVICE_C2H, TYPE, UE_VECTOR_SIZE, UE_ARGMAX_INDEX, UE_ARGMAX1_INDEX,
    URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS,
    DRAM_INSTRUCTION_ADDR, INSTRUCTION_REG_REWRITE, MEMCPY_TYPE,
    UnifiedEngine, ue_35bit_addr_shifter, INSTRUCTION_SIZE_BYTES, UE_MODE,
)
from nn_lib import (
    smart_bf16_permute_core,
    store_weight, store_quantized_weight, store_identity_matrix,
    eltwise_add_core_dram, eltwise_mul_core_dram,
    rms_norm_core_dram_post_add, layer_norm_core_dram_post_add,
)
from quant_lib import quantize_q4_64 as _mlc_quantize_q4_64
def _load_smolvlm2_config(path: str | None = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smolvlm2_config.json")
    with open(path) as f:
        return json.load(f)

_SMOLVLM2_CFG = _load_smolvlm2_config()
HF_MODEL_REPO = _SMOLVLM2_CFG["paths"]["hf_model_repo"]
# =============================================================================
# Helper Methods for SmolVLM2
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
# GGUF generation — quantization helpers
# =============================================================================
def quantize_q4_64(tensor):
    """Quantize an LM weight tensor to q4_64 (used inline by weight_init — no intermediate bin)."""
    return _mlc_quantize_q4_64(tensor)

class SmolVLM2_UnifiedEngine(UnifiedEngine):
    """SmolVLM2-500M accelerator engine: weight loading, compile, inference."""
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
    # Prefill PBI (seq-len-agnostic, gemma3-style): three runtime GPRs primed by run_prefill_v2.
    #   gpr_seq_len    — token count S (matmul/norm/rope/eltwise/gather row loops)
    #   gpr_q_seq_len  — S * GROUP_SIZE (token-major stacked Q rope row loop)
    #   gpr_bucket_idx — aligned(S*GROUP_SIZE)/64, 1-based flash bucket selector
    GPR_SEQ_LEN_REG    = _cfg["fixed_isa_regs"]["GPR_SEQ_LEN_REG"]
    GPR_Q_SEQ_LEN_REG  = _cfg["fixed_isa_regs"]["GPR_Q_SEQ_LEN_REG"]
    GPR_BUCKET_IDX_REG = _cfg["fixed_isa_regs"]["GPR_BUCKET_IDX_REG"]
    # Max prompt length the compile-once prefill program supports (sets flash bucket count).
    PREFILL_MAX_SEQ_LEN = _cfg["model"]["prefill_max_seq_len"]

    def __init__(self, script_dir: str = None):
        self.script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
        self._cfg = _SMOLVLM2_CFG
        # Fixed precision scheme: bf16 vision + q4 LM → the bf16 DRAM layout (params 1 GB / tensors /
        # instructions). No other precision options.
        dl = self._cfg["dram_layout"]["bf16"]
        super().__init__(
            params_dram_base=int(dl["params_dram_base"], 16),
            tensor_dram_base=int(dl["tensor_dram_base"], 16),
            program_dram_base=int(dl["program_dram_base"], 16),
        )
        self._isa_reg_counter = 1
        self.gpr_seq_len = self.GPR_SEQ_LEN_REG      # primed to S in run_prefill_v2
        self.gpr_q_seq_len = self.GPR_Q_SEQ_LEN_REG  # primed to S*GROUP_SIZE in run_prefill_v2
        self.gpr_bucket_idx = self.GPR_BUCKET_IDX_REG  # primed to flash bucket in run_prefill_v2
        # Unified single-bin assembly (compile_all): when active, the three compile_* methods stash
        # their (program_addr, bytes) into these instead of writing separate per-program bins.
        self._unified_active = False
        self._seg_encoder = self._seg_prefill = self._seg_decoder = None
        # Fixed scheme flags (kept so the compile/run weight-path branches resolve consistently).
        self.vision_bf16 = True
        self.lm_bf16 = False
    # --- ISA register helpers (same as Gemma3) ---
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
    def zero_dram(self, chunk_size_bytes: int = 64 * 1024 * 1024) -> None:
        """Zero-fill the DRAM working range [DRAM_START_ADDR..0xFFFFFFFF].

        NOTE: the base ``clear_dram()`` fills with 0xFF, which is NaN in bf16. That
        poisons any region read-before-write on the *next* run (back-to-back runs
        emit token 0 / endoftext immediately). Calling this at startup guarantees a
        clean (zeroed) DRAM regardless of what the previous run left behind.
        Temporary mitigation until the read-before-write gap is found and fixed.
        """
        start = user_dma_core.DRAM_START_ADDR
        total = 0xFFFFFFFF - start + 1
        zeros = b"\x00" * chunk_size_bytes
        offset = 0
        bar_width = 40
        _original_print(f"Zeroing DRAM [{hex(start)}..0xffffffff] ({total / 1024**3:.2f} GB)")
        while offset < total:
            n = min(chunk_size_bytes, total - offset)
            self.dma_write(DMA_DEVICE_H2C, start + offset, zeros[:n], n)
            offset += n
            pct = offset / total
            filled = int(bar_width * pct)
            bar = '█' * filled + '░' * (bar_width - filled)
            _original_print(f"\r  [{bar}] {pct*100:5.1f}%  {offset/1024**2:.0f}/{total/1024**2:.0f} MB",
                            end='', flush=True)
        _original_print()
    def get_arg_max_index(self) -> int:
        return self.read_reg32(UE_ARGMAX_INDEX)
    def dbg_dram(self, addr: int, n_elems: int, name: str, ref: "torch.Tensor" = None,
                 raise_on_nan: bool = True):
        """Read n_elems bf16 from DRAM and report NaN/Inf + stats (and max-abs-err / cosine vs an
        optional CPU reference). Used by the debug harness to localize where activations go bad."""
        buf = bytearray(n_elems * 2)
        self.dma_read(DMA_DEVICE_C2H, addr, buf, n_elems * 2)
        t = torch.frombuffer(bytes(buf), dtype=torch.bfloat16).float()
        n_nan = int(torch.isnan(t).sum()); n_inf = int(torch.isinf(t).sum())
        finite = t[torch.isfinite(t)]
        lo = float(finite.min()) if finite.numel() else float("nan")
        hi = float(finite.max()) if finite.numel() else float("nan")
        mn = float(finite.mean()) if finite.numel() else float("nan")
        info = {"t": t, "nan": n_nan, "inf": n_inf, "cos": None, "cos_excl8": None,
                "cos_tok": None, "relL2": None}
        msg = (f"[dbg] {name:30s} n={n_elems:<8d} nan={n_nan:<6d} inf={n_inf:<5d} "
               f"min={lo:+.3f} max={hi:+.3f} mean={mn:+.4f}")
        if ref is not None:
            rflat = ref.flatten().float()
            m = min(rflat.numel(), t.numel())
            d, r = t[:m], rflat[:m]
            err = float((d - r).abs().max())
            info["cos"] = float(torch.nn.functional.cosine_similarity(d, r, dim=0))
            info["relL2"] = float((d - r).norm() / (r.norm() + 1e-9))
            msg += f"  maxerr={err:.3f} cos={info['cos']:.4f} relL2={info['relL2']:.3f}"
            # Outlier-robust: SmolLM2 "massive activations" (a few huge channels) dominate plain
            # cosine. If ref is 2D [T,H], also report cosine with the top-8 |channel| dims masked
            # and the mean per-token cosine — these expose errors the big dims hide.
            if ref.dim() == 2 and d.numel() == ref.numel():
                T, Hd = ref.shape
                dd, rr = d.view(T, Hd), r.view(T, Hd)
                chan = rr.abs().mean(0)
                topk = torch.topk(chan, min(8, Hd)).indices
                mask = torch.ones(Hd); mask[topk] = 0.0
                info["cos_excl8"] = float(torch.nn.functional.cosine_similarity(
                    (dd * mask).flatten(), (rr * mask).flatten(), dim=0))
                info["cos_tok"] = float(torch.nn.functional.cosine_similarity(dd, rr, dim=1).mean())
                msg += f" cos_excl8={info['cos_excl8']:.4f} cos_tok={info['cos_tok']:.4f}"
        print(msg, flush=True)
        if raise_on_nan and (n_nan or n_inf):
            raise FloatingPointError(f"NaN/Inf detected in {name} (nan={n_nan}, inf={n_inf})")
        return info
    def isa_add_set_core(self, dst_reg_idx: int, immediate_value: int, timeout_s: float = 10.0) -> None:
        """Set ISA register to immediate value."""
        isa_set_register(self, dst_reg_idx, immediate_value, timeout_s)

    # --- Weight loading ---
    def weight_init(self) -> None:
        """Build ALL weights into params DRAM straight from the HF model — q4_64 LM (quantized on the
        fly) + bf16 vision. No intermediate weight bins: dump_snapshot then captures the assembled
        params DRAM into the single weights.bin. (Fixed scheme: bf16 vision + q4 LM; no other options.)"""
        from transformers import AutoTokenizer, AutoModelForImageTextToText
        model_dir = os.path.join(self.script_dir, "smolvlm2_bin", "SmolVLM2-500M-Video-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        hf_model = AutoModelForImageTextToText.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map=None).eval()

        # --- Language model weights: q4_64 projections; bf16 embeddings/norms ---
        text_model = hf_model.model.text_model
        embed_bf16 = text_model.embed_tokens.weight.data.to(torch.bfloat16)
        self.embedding_weight = embed_bf16.clone()  # CPU copy for host-side token embedding lookup
        self.embed_addr = store_weight(self, embed_bf16)
        self.lm_layer_addrs = []
        for layer in text_model.layers:
            la = {}
            for proj, attr in [('q', 'q_proj'), ('k', 'k_proj'), ('v', 'v_proj'), ('o', 'o_proj'),
                               ('gate', 'gate_proj'), ('up', 'up_proj'), ('down', 'down_proj')]:
                module = layer.self_attn if proj in ('q', 'k', 'v', 'o') else layer.mlp
                data, _ = quantize_q4_64(getattr(module, attr).weight.data)
                la[f'{proj}_scale'], la[f'{proj}_data'] = store_quantized_weight(self, data.tobytes())
            la['ln1_gamma'] = store_weight(self, layer.input_layernorm.weight.data.to(torch.bfloat16))
            la['ln2_gamma'] = store_weight(self, layer.post_attention_layernorm.weight.data.to(torch.bfloat16))
            self.lm_layer_addrs.append(la)
        self.final_norm_addr = store_weight(self, text_model.norm.weight.data.to(torch.bfloat16))
        data, _ = quantize_q4_64(hf_model.lm_head.weight.data)
        self.lm_head_scale, self.lm_head_data = store_quantized_weight(self, data.tobytes())
        print(f"LM weights loaded (Q4): {self.NUM_LAYERS} layers, params DRAM usage: {self.get_params_dram_usage()} bytes")
        # Identity matrix for decode attention
        self.identity_addr = store_identity_matrix(self)
        params_used = self.get_params_dram_usage()
        params_limit = self._tensor_dram_base - self._params_dram_base
        _original_print(f"  Params DRAM: {params_used/1024/1024:.1f} MB used / {params_limit/1024/1024:.0f} MB available"
                        + (" OVERFLOW!" if params_used > params_limit else ""))

        # --- Vision encoder weights (bf16) — always built so the instruction bin is the full VLM bin
        # (one bin, robust + easy to maintain); whether vision RUNS is a runtime decision (--image). ---
        NPS = 32  # num_patches_per_side = 512 / 16
        boundaries = torch.arange(1.0 / NPS, 1.0, 1.0 / NPS, dtype=torch.float32)
        frac = torch.arange(NPS, dtype=torch.float32) / NPS * (1 - 1e-6)
        buckets = torch.bucketize(frac, boundaries, right=True)
        vis_position_ids = (buckets[:, None] * NPS + buckets[None, :]).flatten()
        vis_enc = hf_model.model.vision_model
        self.vis_layer_addrs = []
        for layer in vis_enc.encoder.layers:
            la = {}
            for proj, attr in [('q', 'q_proj'), ('k', 'k_proj'), ('v', 'v_proj'), ('o', 'out_proj')]:
                linear = getattr(layer.self_attn, attr)
                la[f'{proj}_weight'] = store_weight(self, linear.weight.data)
                la[f'{proj}_bias'] = store_weight(self, linear.bias.data)
            for proj, attr in [('fc1', 'fc1'), ('fc2', 'fc2')]:
                linear = getattr(layer.mlp, attr)
                la[f'{proj}_weight'] = store_weight(self, linear.weight.data)
                la[f'{proj}_bias'] = store_weight(self, linear.bias.data)
            for ln, attr in [('ln1', 'layer_norm1'), ('ln2', 'layer_norm2')]:
                norm = getattr(layer, attr)
                la[f'{ln}_weight'] = store_weight(self, norm.weight.data)
                la[f'{ln}_bias'] = store_weight(self, norm.bias.data)
            self.vis_layer_addrs.append(la)
        vis_layers = len(vis_enc.encoder.layers)
        self.patch_weight_addr = store_weight(self, vis_enc.embeddings.patch_embedding.weight.data.reshape(768, 768))
        self.patch_bias_addr = store_weight(self, vis_enc.embeddings.patch_embedding.bias.data)
        pos_table = vis_enc.embeddings.position_embedding.weight.data  # [1024, 768]
        self.pos_embed_addr = store_weight(self, pos_table[vis_position_ids])
        self.vis_post_ln_weight = store_weight(self, vis_enc.post_layernorm.weight.data)
        self.vis_post_ln_bias = store_weight(self, vis_enc.post_layernorm.bias.data)
        self.connector_weight_addr = store_weight(self, hf_model.model.connector.modality_projection.proj.weight.data)
        del hf_model
        params_used = self.get_params_dram_usage()
        _original_print(f"  Vision BF16 loaded: {vis_layers} layers, total params: {params_used/1024/1024:.1f} MB")
    def tensor_init(self, max_seq_len: int = 512) -> None:
        """Allocate DRAM for activations, KV cache, masks, RoPE."""
        self.max_seq_len = max_seq_len
        seq_len = max_seq_len
        bpe = 2
        # KV cache offset constants (Gemma3-style flat layout: [layer, head, seq, dim])
        self.bytes_per_element = bpe
        self.k_size = self.HEAD_DIM * bpe                             # 128 bytes per position per head
        self.KV_HEAD_STRIDE = seq_len * self.k_size                   # one head, all positions
        self.KV_LAYER_STRIDE = self.NUM_KV_HEADS * self.KV_HEAD_STRIDE  # all heads, one layer

        print(f"Allocate tensor dram start at DRAM address: 0x{self.get_tensor_dram_addr():X}")
        # KV cache: [NUM_LAYERS, NUM_KV_HEADS, max_seq_len, HEAD_DIM], zero-padded
        kv_cache_total = self.NUM_LAYERS * self.KV_LAYER_STRIDE
        self.LAYER0_K_DRAM = self.allocate_tensor_dram(kv_cache_total)
        self.LAYER0_V_DRAM = self.allocate_tensor_dram(kv_cache_total)
        kv_zeros = torch.zeros(self.NUM_LAYERS * self.NUM_KV_HEADS * seq_len * self.HEAD_DIM, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_K_DRAM, kv_zeros)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, kv_zeros)
        # Decoder layer intermediates (shared across 32 layers):
        self.LAYER0_INPUT_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LAYER0_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LAYER0_PRE_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LAYER0_Q_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LAYER0_K_PROJ_DRAM = self.allocate_tensor_dram(seq_len * self.NUM_KV_HEADS * self.HEAD_DIM * bpe)
        self.LAYER0_V_PROJ_DRAM = self.allocate_tensor_dram(seq_len * self.NUM_KV_HEADS * self.HEAD_DIM * bpe)
        self.LAYER0_Q_PERM_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LAYER0_ATTN_OUT_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LAYER0_ATTN_SCRATCH_DRAM = self.allocate_tensor_dram((self.HEAD_DIM * seq_len + max(self.HEAD_DIM, 64) * seq_len) * bpe)
        self.LAYER0_ATTN_RESULT_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LAYER0_O_PROJ_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LAYER0_RESIDUAL_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LAYER0_MLP_GATE_DRAM = self.allocate_tensor_dram(seq_len * self.INTERMEDIATE_SIZE * bpe)
        self.LAYER0_MLP_UP_DRAM = self.allocate_tensor_dram(seq_len * self.INTERMEDIATE_SIZE * bpe)
        self.LAYER0_MLP_MULT_DRAM = self.allocate_tensor_dram(seq_len * self.INTERMEDIATE_SIZE * bpe)
        self.LAYER0_MLP_DOWN_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.FINAL_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LOGITS_DRAM = self.allocate_tensor_dram(1 * self.VOCAB_SIZE * bpe)
        # Causal mask for prefill (written by compile_prefill/load_prefill before use)
        self.CAUSAL_MASK_DRAM = self.allocate_tensor_dram(seq_len * seq_len * bpe)
        # Decode bias: [GROUP_SIZE, max_seq_len] — written each decode step
        self.DECODE_BIAS_DRAM = self.allocate_tensor_dram(self.GROUP_SIZE * seq_len * bpe)
        # --- gemma3-style stacked-GQA flash buffers (seq-len-agnostic prefill v2) ---
        # One bucket-dispatcher flash per kv-group operates on a square of aligned(S*GROUP_SIZE)
        # rows. Buffers are sized for the largest bucket = aligned(PREFILL_MAX_SEQ_LEN*GROUP_SIZE).
        G = self.GROUP_SIZE
        qmax = ((self.PREFILL_MAX_SEQ_LEN * G + 63) // 64) * 64
        self.PREFILL_QMAX = qmax
        self.FLASH_NUM_BUCKETS = qmax // UE_VECTOR_SIZE
        D = self.HEAD_DIM
        self.FLASH_Q_DRAM = self.allocate_tensor_dram(qmax * D * bpe)       # token-major stacked Q
        self.FLASH_K_DRAM = self.allocate_tensor_dram(qmax * D * bpe)       # K duplicated token-major
        self.FLASH_V_DRAM = self.allocate_tensor_dram(qmax * D * bpe)       # V duplicated token-major
        self.FLASH_OUT_DRAM = self.allocate_tensor_dram(qmax * D * bpe)     # attention output (stacked)
        self.FLASH_SCRATCH_DRAM = self.allocate_tensor_dram(
            UE_VECTOR_SIZE * qmax * bpe + D * qmax * bpe)                   # Vᵀ + softmax scratch
        self.FLASH_ATTN_P_DRAM = self.allocate_tensor_dram(qmax * qmax * bpe)  # fused QKᵀ+softmax probs
        self.FLASH_BIAS_DRAM = self.allocate_tensor_dram(qmax * qmax * bpe)    # block-diagonal causal bias
        # RoPE cos/sin tables
        self._load_rope_tables()
        # Vision encoder intermediates (fixed seq=1024, hidden=768, intermediate=3072):
        VS, VH, VI = 1024, 768, 3072
        self.VIS_PIXEL_IN_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_PATCH_PERM_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_PATCH_PROJ_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_IO_A_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_IO_B_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_LN_OUT_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_Q_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_K_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_V_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_Q_PERM_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_K_PERM_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_V_PERM_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_ATTN_OUT_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_ATTN_SCRATCH_DRAM = self.allocate_tensor_dram((64 * VS + 64 * VS) * bpe)
        self.VIS_ATTN_RESULT_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_O_PROJ_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_RESIDUAL_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_MLP_INTER_DRAM = self.allocate_tensor_dram(VS * VI * bpe)
        self.VIS_MLP_OUT_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_POST_LN_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        # Connector: pixel shuffle [1024,768]→[64,12288] (scale_factor=4), linear→[64,960]
        self.VIS_SHUFFLED_DRAM = self.allocate_tensor_dram(64 * 12288 * bpe)
        self.VIS_CONNECTOR_DRAM = self.allocate_tensor_dram(64 * self.HIDDEN_SIZE * bpe)
        # Permute params (identity matrix + temp for bf16_smart_permute_core)
        permute_size = UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe
        self.PERMUTE_PARAMS_DRAM = self.get_params_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, self.PERMUTE_PARAMS_DRAM,
                       torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16), permute_size)
        self.allocate_params_dram(permute_size)
        self.PERMUTE_TEMP_DRAM = self.get_tensor_dram_addr()
        self.allocate_tensor_dram(VS * VH * bpe * 2)  # temp space for permute decomposition
        # §7 vision shared-subroutine flash: fixed single-head operand buffers ([VS, VD]) + the
        # bucket-dispatcher's softmax-probs scratch ([VS, VS]). Each per-(layer,head) call site
        # marshals its head's Q/K/V here, jumps into the one shared flash body, and copies VIS_FLASH_OUT
        # back. Appended at the END of tensor_init so no earlier address shifts (run_from_bin mirrors).
        VD = self.HEAD_DIM
        self.VIS_FLASH_Q_DRAM = self.allocate_tensor_dram(VS * VD * bpe)
        self.VIS_FLASH_K_DRAM = self.allocate_tensor_dram(VS * VD * bpe)
        self.VIS_FLASH_V_DRAM = self.allocate_tensor_dram(VS * VD * bpe)
        self.VIS_FLASH_OUT_DRAM = self.allocate_tensor_dram(VS * VD * bpe)
        self.VIS_FLASH_ATTN_P_DRAM = self.allocate_tensor_dram(VS * VS * bpe)
        # Fixed scratch for the SHARED vision layer body: the small per-layer params (biases ≤ 3072,
        # layer_norm gamma/beta = 768) are marshalled here from their register-driven source address
        # so the matmul/layernorm kernels read a static address. (Big weights go through gpr_B_reg.)
        VI = 3072  # vision intermediate (largest bias = fc1)
        self.VIS_BIAS_SCRATCH = self.allocate_tensor_dram(VI * bpe)
        self.VIS_GAMMA_SCRATCH = self.allocate_tensor_dram(VH * bpe)
        self.VIS_BETA_SCRATCH = self.allocate_tensor_dram(VH * bpe)
        print(f"    Allocate tensor dram end at DRAM address: 0x{self.get_tensor_dram_addr():X}, usage: {self.get_tensor_dram_usage()} bytes")
    def _load_rope_tables(self) -> None:
        """Precompute cos/sin tables, pre-negate sin, DMA to device."""
        D, S, bpe = self.HEAD_DIM, self.max_seq_len, 2
        inv_freq = 1.0 / (self.ROPE_THETA ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
        freqs = torch.outer(torch.arange(S, dtype=torch.float32), inv_freq)  # [S, D/2]
        cos_full = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=1).to(torch.bfloat16).contiguous()
        sin_full = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=1).to(torch.bfloat16).contiguous()
        sin_full[:, :D // 2] = -sin_full[:, :D // 2]  # pre-negate for HW RoPE kernel
        table_size = S * D * bpe
        self.ROPE_COS_DRAM = self.get_params_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, self.ROPE_COS_DRAM, cos_full.flatten(), table_size)
        self.allocate_params_dram(table_size)
        self.ROPE_SIN_DRAM = self.get_params_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, self.ROPE_SIN_DRAM, sin_full.flatten(), table_size)
        self.allocate_params_dram(table_size)
        self._hw_cos, self._hw_sin = cos_full, sin_full  # CPU copies for decode position updates
        # Contiguous packed table for the prefill d64 PBI RoPE (rope_hf_core_dram_d64_pbi):
        # per token [cos(D) || sin(D)] = 2D elems, sin first half pre-negated. The d64 path
        # asserts sin_dram_addr == cos_dram_addr + D*bpe, so cos/sin must be interleaved per
        # token (NOT two separate [S,D] tables like ROPE_COS_DRAM/ROPE_SIN_DRAM above).
        packed = torch.cat([cos_full, sin_full], dim=1).contiguous()  # [S, 2D]
        self.ROPE_PACKED_DRAM = self.get_params_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, self.ROPE_PACKED_DRAM, packed.flatten(), S * 2 * D * bpe)
        self.allocate_params_dram(S * 2 * D * bpe)
        # Token-major-DUPLICATED packed table for the gemma3-style stacked-Q rope. The stacked Q
        # for one kv-group is [S, GROUP_SIZE, D] flattened to rows r = t*G + g; roping it as one
        # M=S*G d64 loop needs row r to use token (r//G)'s cos/sin. There is no d64 GQA rope, so
        # we pre-duplicate each token's [cos||sin] row GROUP_SIZE times → row r holds packed[r//G].
        G = self.GROUP_SIZE
        packed_gqa = packed.repeat_interleave(G, dim=0).contiguous()  # [G*S, 2D]
        self.ROPE_PACKED_GQA_DRAM = self.get_params_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, self.ROPE_PACKED_GQA_DRAM, packed_gqa.flatten(), G * S * 2 * D * bpe)
        self.allocate_params_dram(G * S * 2 * D * bpe)
        # Padded-split cos/sin for prefill RoPE (D<128 path):
        # Each 32-element half is zero-padded to 64 elements = 128 bytes = 1 URAM row
        half = D // 2
        if D < 128:
            cos_lo = cos_full[:, :half]  # [S, 32]
            cos_hi = cos_full[:, half:]  # [S, 32]
            sin_lo = sin_full[:, :half]  # [S, 32] (already pre-negated)
            sin_hi = sin_full[:, half:]  # [S, 32]
            pad_D = UE_VECTOR_SIZE  # 64 — pad each half to full URAM row
            for name, data in [('COS_LO', cos_lo), ('COS_HI', cos_hi),
                               ('SIN_LO', sin_lo), ('SIN_HI', sin_hi)]:
                padded = torch.zeros(S, pad_D, dtype=torch.bfloat16)
                padded[:, :half] = data
                padded = padded.contiguous()
                addr = self.get_params_dram_addr()
                self.dma_write(DMA_DEVICE_H2C, addr, padded.flatten(), S * pad_D * bpe)
                self.allocate_params_dram(S * pad_D * bpe)
                setattr(self, f'ROPE_{name}_PAD_DRAM', addr)
            print(f"    RoPE padded-split [{S}, {pad_D}] × 4 arrays DMA'd")
        print(f"    RoPE cos/sin [{S}, {D}] DMA'd, theta={self.ROPE_THETA}")

        # On-FPGA repetition-penalty per-vocab additive bias (the decode LM-head's C term, broadcast_N).
        # Allocated LAST so its presence never shifts any earlier baked tensor address; zeros = no penalty.
        self.PENALTY_BIAS_DRAM = self.allocate_tensor_dram(1 * self.VOCAB_SIZE * bpe)

    # --- Compile ---
    def compile_encoder(self) -> int:
        """Compile vision encoder program (PORTABLE — released API only). Returns DRAM address.

        12 structurally-identical SigLIP layers, UNROLLED, with every per-row op a PBI hardware loop so
        the captured bin is structure-bound, not M-bound. No ``gpr_B_reg`` / layer-body sharing (that
        needed a non-released ``matmat_mul_core`` extension — reverted). The two PBI row counts have
        DIFFERENT meaning so they need two primed registers: ``matmat``/``eltwise`` ``gpr_M_reg`` is the
        row count (= S); ``layer_norm`` ``gpr_M_reg`` is the *chunk* count (= S // chunk_size). §7 shared
        flash + the strided-DMA fused head transpose are kept as-is.
        """
        from user_dma_core import TYPE
        S, H, D, N_HEADS, I = 1024, 768, 64, 12, 3072
        bpe = 2

        # layer_norm_core_dram_pbi chunking (replicated from the engine so we prime n_chunks correctly):
        # ops_per_row = 4 + gamma + beta = 6; max_chunk = (256-4)//6; ideal = min(URAM//N, S, max_chunk);
        # then the largest divisor of S that is <= ideal. All encoder LNs share M=S, N=H, gamma+beta.
        _ln_ops_per_row = 6
        _ln_chunk = min(URAM_NEAR_FULL_ELEMENTS // H, S, (256 - 4) // _ln_ops_per_row)
        while S % _ln_chunk != 0:
            _ln_chunk -= 1
        ln_n_chunks = S // _ln_chunk

        self.start_capture()

        # PBI row counts, primed once. vis_S_reg = S rows (matmat/eltwise); vis_ln_chunks = S//chunk (LN).
        vis_S_reg = self.alloc_isa_reg()
        vis_ln_chunks = self.alloc_isa_reg()
        self.generate_instruction_add_set(vis_S_reg, S)
        self.generate_instruction_add_set(vis_ln_chunks, ln_n_chunks)

        # === Patch embedding: pixels → [1024, 768] ===
        P = 16
        H_patches = 32  # 512 / 16
        smart_bf16_permute_core(self, dims=[3, H_patches, P, H_patches, P], permute_indices=[1, 3, 0, 2, 4],
            input_dram_addr=self.VIS_PIXEL_IN_DRAM, output_dram_addr=self.VIS_PATCH_PERM_DRAM,
            params_dram_addr=self.PERMUTE_PARAMS_DRAM, temp_dram_start=self.PERMUTE_TEMP_DRAM)
        self.matmat_mul_core(M=S, K=H, N=H, A_DRAM_ADDR=self.VIS_PATCH_PERM_DRAM, B_DRAM_ADDR=self.patch_weight_addr,
            OUTPUT_DRAM_ADDR=self.VIS_PATCH_PROJ_DRAM, C_DRAM_ADDR=self.patch_bias_addr, bias_mode="broadcast_N",
            gpr_M_reg=vis_S_reg)
        eltwise_add_core_dram(self, size=S * H, A_DRAM_ADDR=self.VIS_PATCH_PROJ_DRAM,
            B_DRAM_ADDR=self.pos_embed_addr, OUTPUT_DRAM_ADDR=self.VIS_IO_A_DRAM)

        # §7 shared-subroutine attention (vision): the per-head flash body is compiled ONCE after the
        # encoder HALT; every per-(layer,head) call site marshals its head's [S, D] Q/K/V into the fixed
        # VIS_FLASH buffers via a strided DMA (= the [S,768]→[12,S,64] head transpose), bakes ADD_SET
        # gpr_bucket(=S/64=16)+gpr_ret, and jumps in. One-shot program; vis_prog_base == the bin load addr.
        vis_prog_base = self.get_program_dram_addr()
        vis_gpr_bucket = self.alloc_isa_reg()
        vis_gpr_ret = self.alloc_isa_reg()
        vis_call_sites: list[int] = []
        vis_num_buckets = S // UE_VECTOR_SIZE   # 16 (single full-attention segment)

        def vis_matmul(M, K, N, A, la, proj, OUT, bias=None, **kw):
            self.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f'{proj}_weight'],
                OUTPUT_DRAM_ADDR=OUT, C_DRAM_ADDR=bias, bias_mode="broadcast_N", gpr_M_reg=vis_S_reg, **kw)

        for layer_idx, la in enumerate(self.vis_layer_addrs):
            h_in  = self.VIS_IO_A_DRAM if layer_idx % 2 == 0 else self.VIS_IO_B_DRAM
            h_out = self.VIS_IO_B_DRAM if layer_idx % 2 == 0 else self.VIS_IO_A_DRAM
            # LN1 (PBI)
            self.layer_norm_core_dram(M=S, N=H, A_DRAM_ADDR=h_in, OUTPUT_DRAM_ADDR=self.VIS_LN_OUT_DRAM,
                GAMMA_DRAM_ADDR=la['ln1_weight'], BETA_DRAM_ADDR=la['ln1_bias'], gpr_M_reg=vis_ln_chunks)
            # Q/K/V projections (PBI)
            for proj, dst in [('q', self.VIS_Q_DRAM), ('k', self.VIS_K_DRAM), ('v', self.VIS_V_DRAM)]:
                vis_matmul(S, H, H, self.VIS_LN_OUT_DRAM, la, proj, dst, bias=la[f'{proj}_bias'])
            # §7 attention with fused strided-DMA head transpose (per-head gather → flash → scatter).
            elems = S * D
            col_stride = D * bpe   # one head's column block width
            row_jump = H * bpe     # full [S, 768] row stride
            for h in range(N_HEADS):
                col = h * col_stride
                for src, dst in ((self.VIS_Q_DRAM + col, self.VIS_FLASH_Q_DRAM),
                                 (self.VIS_K_DRAM + col, self.VIS_FLASH_K_DRAM),
                                 (self.VIS_V_DRAM + col, self.VIS_FLASH_V_DRAM)):
                    self.accelerator_memory_to_sram(src, 0x00000, elems,
                        stride_bytes_per_chunk=col_stride, stride_jump_bytes=row_jump)
                    self.sram_to_accelerator_memory(0x00000, dst, elems)
                self.generate_instruction_add_set(vis_gpr_bucket, S // UE_VECTOR_SIZE)
                self.pad_capture_to_64b_boundary()
                return_word_addr = ue_35bit_addr_shifter(
                    vis_prog_base + (self.capture_count + 2) * INSTRUCTION_SIZE_BYTES)
                self.generate_instruction_add_set(vis_gpr_ret, return_word_addr)
                vis_call_sites.append(self.capture_count)
                self.generate_instruction_jump_abs(target_instruction_word_addr=0)
                self.accelerator_memory_to_sram(self.VIS_FLASH_OUT_DRAM, 0x00000, elems)
                self.sram_to_accelerator_memory(0x00000, self.VIS_ATTN_RESULT_DRAM + col, elems,
                    stride_bytes_per_chunk=col_stride, stride_jump_bytes=row_jump)
            # O projection + residual (eltwise) + LN2 (PBI) — split from the static post-add
            vis_matmul(S, H, H, self.VIS_ATTN_RESULT_DRAM, la, 'o', self.VIS_O_PROJ_DRAM, bias=la['o_bias'])
            eltwise_add_core_dram(self, size=S * H, A_DRAM_ADDR=h_in, B_DRAM_ADDR=self.VIS_O_PROJ_DRAM,
                OUTPUT_DRAM_ADDR=self.VIS_RESIDUAL_DRAM)
            self.layer_norm_core_dram(M=S, N=H, A_DRAM_ADDR=self.VIS_RESIDUAL_DRAM, OUTPUT_DRAM_ADDR=self.VIS_LN_OUT_DRAM,
                GAMMA_DRAM_ADDR=la['ln2_weight'], BETA_DRAM_ADDR=la['ln2_bias'], gpr_M_reg=vis_ln_chunks)
            # MLP: fc1 + GELU, fc2, residual
            vis_matmul(S, H, I, self.VIS_LN_OUT_DRAM, la, 'fc1', self.VIS_MLP_INTER_DRAM, bias=la['fc1_bias'], gelu_enable=True)
            vis_matmul(S, I, H, self.VIS_MLP_INTER_DRAM, la, 'fc2', self.VIS_MLP_OUT_DRAM, bias=la['fc2_bias'])
            eltwise_add_core_dram(self, size=S * H,
                A_DRAM_ADDR=self.VIS_RESIDUAL_DRAM, B_DRAM_ADDR=self.VIS_MLP_OUT_DRAM, OUTPUT_DRAM_ADDR=h_out)

        # Post-layernorm (PBI), pixel shuffle, connector (M=64, static).
        final_vis = self.VIS_IO_A_DRAM if len(self.vis_layer_addrs) % 2 == 0 else self.VIS_IO_B_DRAM
        self.layer_norm_core_dram(M=S, N=H, A_DRAM_ADDR=final_vis, OUTPUT_DRAM_ADDR=self.VIS_POST_LN_DRAM,
            GAMMA_DRAM_ADDR=self.vis_post_ln_weight, BETA_DRAM_ADDR=self.vis_post_ln_bias, gpr_M_reg=vis_ln_chunks)
        smart_bf16_permute_core(self, dims=[8, 4, 8, 4, H], permute_indices=[0, 2, 1, 3, 4],
            input_dram_addr=self.VIS_POST_LN_DRAM, output_dram_addr=self.VIS_SHUFFLED_DRAM,
            params_dram_addr=self.PERMUTE_PARAMS_DRAM, temp_dram_start=self.PERMUTE_TEMP_DRAM)
        self.matmat_mul_core(M=64, K=12288, N=self.HIDDEN_SIZE,
            A_DRAM_ADDR=self.VIS_SHUFFLED_DRAM, B_DRAM_ADDR=self.connector_weight_addr,
            OUTPUT_DRAM_ADDR=self.VIS_CONNECTOR_DRAM)

        self.generate_instruction_halt()   # encoder ends; shared flash body follows.

        # ---- shared flash body (the §7 attention), after HALT; patch the attention jumps ----
        vis_sub_start, _vis_flops = self.flash_attention_core(head_dim=D, seq_len=S,
            Q_DRAM_ADDR=self.VIS_FLASH_Q_DRAM, K_DRAM_ADDR=self.VIS_FLASH_K_DRAM,
            V_DRAM_ADDR=self.VIS_FLASH_V_DRAM, OUTPUT_DRAM_ADDR=self.VIS_FLASH_OUT_DRAM,
            SCRATCH_DRAM_ADDR=self.VIS_ATTN_SCRATCH_DRAM, ATTN_P_DRAM_ADDR=self.VIS_FLASH_ATTN_P_DRAM,
            IDENTITY_DRAM_ADDR=self.identity_addr,
            gpr_bucket_idx=vis_gpr_bucket, num_buckets=vis_num_buckets, gpr_ret_id=vis_gpr_ret)
        for _idx in vis_call_sites:
            self._patch_jump_immediate(_idx, ue_35bit_addr_shifter(vis_sub_start))
        self.release_isa_reg()  # vis_gpr_ret
        self.release_isa_reg()  # vis_gpr_bucket
        self.release_isa_reg()  # vis_ln_chunks
        self.release_isa_reg()  # vis_S_reg
        self.stop_capture()
        all_bytes = bytearray()
        for inst in self.capture_buffer:
            all_bytes.extend(inst.get_bytes())
        program_addr = self.get_program_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, program_addr, all_bytes, len(all_bytes))
        self.allocate_program_dram(len(all_bytes))
        self.clear_capture_buffer()
        if self._unified_active:
            self._seg_encoder = (program_addr, bytes(all_bytes))
        else:
            bin_path = os.path.join(self.script_dir, "smolvlm2_bin", "encoder_program.bin")
            with open(bin_path, "wb") as f:
                f.write(all_bytes)
        self._vis_program_addr = program_addr
        print(f"    Vision encoder compiled: {len(all_bytes)} bytes at 0x{program_addr:X}")
        return program_addr

    def compile_prefill_v2(self, halt_after_layer: int = None, debug_dump: bool = False) -> None:
        """Seq-len-agnostic gemma3-style prefill. Compiled ONCE for PREFILL_MAX; the runtime length
        is read by the bucket-dispatcher flash via gpr_bucket_idx. Permutes/matmuls/norms run static
        at PREFILL_MAX (shorter prompts padded with epsilon, masked by the block-diagonal bias);
        only the d64 RoPE (gpr) and the O(seq^2) flash (bucket) are runtime-sized.

        Per kv-group GQA into one flash: gather the 3 q-heads token-major [PM,G,D]; rope it (M=PM*G,
        token-major-duplicated table); rope the kv-head into the decoder KV cache then duplicate its
        rows xG token-major; one flash_attention_core(gpr_bucket_idx); un-stack the output back into
        [PM,H]. Ends after the 32 layers with HALT (no final-norm/LM head — those depend on the
        runtime last-token offset and are emitted by run_prefill_v2). Captured in place at a fixed
        program-DRAM address so the flash dispatcher's absolute jumps stay valid; run_prefill_v2's
        preamble primes gpr_bucket_idx and jump_abs-es into this program.
        """
        from user_dma_core import TYPE
        H, D, I = self.HIDDEN_SIZE, self.HEAD_DIM, self.INTERMEDIATE_SIZE
        KV = self.NUM_KV_HEADS * D
        G = self.GROUP_SIZE
        bpe = 2
        PM = self.PREFILL_MAX_SEQ_LEN          # static row count for non-attention ops
        qmax = self.PREFILL_QMAX                # aligned(PM*G) — stacked flash rows
        num_buckets = self.FLASH_NUM_BUCKETS

        def lm_matmul(M, K, N, A, proj, la, OUT, **kw):
            # PREFILL is M=PM (>1) → true GEMM via matmat_mul_core with gpr_M_reg (core_changes §3h):
            # the M-tile loop becomes a runtime WHILE-loop (primed to PM via gpr_seq_len), so the body
            # is emitted once instead of unrolled per output row. The M=1 GEMV quantized_matmat_core is
            # decode-only (compile_decoder keeps it). NOTE gpr_M_reg dispatches to the PBI path.
            self.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f'{proj}_data'],
                OUTPUT_DRAM_ADDR=OUT, is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=la[f'{proj}_scale'], gpr_M_reg=self.gpr_seq_len, **kw)

        def strided_copy(src, src_jump, dst, dst_jump, rows, width):
            # static [rows, width] copy: strided gather -> contiguous SRAM -> strided scatter
            self.accelerator_memory_to_sram(src, 0x00000, rows * width,
                stride_bytes_per_chunk=width * bpe, stride_jump_bytes=src_jump)
            self.sram_to_accelerator_memory(0x00000, dst, rows * width,
                stride_bytes_per_chunk=width * bpe, stride_jump_bytes=dst_jump)

        def duplicate_gqa_rows(src_sram_addr, dst_dram_addr):
            # token-major duplication: dst row t*G+g = src row t (kv head broadcast across the group)
            row_bytes = D * bpe
            row_uram_words = row_bytes // (UE_VECTOR_SIZE * bpe)
            _, src_uram_addr = self.sram_address_to_uram_address(src_sram_addr)
            ptr = self.alloc_inst_ptr()
            self.generate_instruction_pbi_init(dram_shared_addr=dst_dram_addr, dma_length=row_bytes,
                output_size=0, uram_length=0, uram_a_start_addr=src_uram_addr,
                uram_b_start_addr=src_uram_addr, uram_wb_addr=0, uram_dst_addr=0,
                fmax_context_addr=0, inst_pointer_idx=ptr)
            self.loop_start(loop_cnt=PM)
            self.loop_start(G)
            self.sram_to_accelerator_memory(sram_address=0, accelerator_dram_address=row_bytes,
                element_size=D, inst_pointer_idx=ptr, memcpy_length_bytes=0)
            self.loop_end()
            self.generate_instruction_pbi_inc(dram_shared_addr=0, dma_length=0, output_size=0,
                uram_length=0, uram_a_start_addr=row_uram_words, uram_b_start_addr=row_uram_words,
                uram_wb_addr=0, uram_dst_addr=0, fmax_context_addr=0, inst_pointer_idx=ptr)
            self.loop_end()
            self.release_inst_ptr(ptr)

        # Debug: per-layer hidden-state dump [NUM_LAYERS, PM, H] so one run captures every layer's
        # output for NaN/cosine comparison against a CPU reference (localizes the first bad layer).
        if debug_dump:
            self.DBG_LAYER_DUMP = self.allocate_tensor_dram(self.NUM_LAYERS * PM * H * bpe)
            print(f"    [dbg] layer dump @0x{self.DBG_LAYER_DUMP:X} ({self.NUM_LAYERS}x{PM}x{H})")
        # Scratch ISA regs start past the fixed gpr regs (1..6) so rope/flash scratch don't clobber them.
        self._isa_reg_counter = max(self._cfg["fixed_isa_regs"].values()) + 1
        self.start_capture()
        prefill_addr = self.get_program_dram_addr()
        # Constant runtime regs for the d64 RoPE loops (PM is compile-time fixed).
        self.generate_instruction_add_set(self.gpr_seq_len, PM)
        self.generate_instruction_add_set(self.gpr_q_seq_len, PM * G)
        # §7 shared-subroutine attention: compile the bucket flash body ONCE after the HALT and have
        # every per-(layer,kv-group) call site jump in (priming gpr_ret_id as the return address).
        # gpr_bucket_idx is primed once at runtime by run_prefill_v2's preamble (same seq_len for all
        # groups/layers), so the call sites only set the return addr + jump.
        gpr_ret_id = self.alloc_isa_reg()
        flash_call_sites: list[int] = []
        for layer_idx in range(self.NUM_LAYERS):
            la = self.lm_layer_addrs[layer_idx]
            h_in  = self.LAYER0_INPUT_DRAM if layer_idx % 2 == 0 else self.LAYER0_OUTPUT_DRAM
            h_out = self.LAYER0_OUTPUT_DRAM if layer_idx % 2 == 0 else self.LAYER0_INPUT_DRAM
            # Input layernorm (fused with previous layer's MLP residual for layers 1+)
            # PBI per-row norms (gpr_M_reg=gpr_seq_len) — hardware row loop, not a static-M unroll
            # (that unrolled rms_norm_core_dram_post_add was 1.49 MiB of the prefill). The fused
            # residual+norm is split into eltwise (residual sum) + rms_norm, both PBI, like qwen2.5.
            if layer_idx == 0:
                self.rms_norm_core_dram(M=PM, N=H, A_DRAM_ADDR=h_in,
                    OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=la['ln1_gamma'],
                    gpr_M_reg=self.gpr_seq_len)
            else:
                self.eltwise_core_dram(M=PM, N=H, dram_a=self.LAYER0_RESIDUAL_DRAM,
                    dram_b=self.LAYER0_MLP_DOWN_DRAM, dram_out=h_in, mode=UE_MODE.ELTWISE_ADD,
                    gpr_M_reg=self.gpr_seq_len)
                self.rms_norm_core_dram(M=PM, N=H, A_DRAM_ADDR=h_in,
                    OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=la['ln1_gamma'],
                    gpr_M_reg=self.gpr_seq_len)
            # Q/K/V projections
            lm_matmul(PM, H, H,  self.LAYER0_PRE_NORM_DRAM, 'q', la, self.LAYER0_Q_DRAM)
            lm_matmul(PM, H, KV, self.LAYER0_PRE_NORM_DRAM, 'k', la, self.LAYER0_K_PROJ_DRAM)
            lm_matmul(PM, H, KV, self.LAYER0_PRE_NORM_DRAM, 'v', la, self.LAYER0_V_PROJ_DRAM)
            # K per kv-head -> decoder KV cache (contiguous [PM,D]) + RoPE
            for h in range(self.NUM_KV_HEADS):
                k_cache = self.LAYER0_K_DRAM + layer_idx * self.KV_LAYER_STRIDE + h * self.KV_HEAD_STRIDE
                strided_copy(self.LAYER0_K_PROJ_DRAM + h * D * bpe, KV * bpe, k_cache, D * bpe, PM, D)
                self.rope_hf_core_dram(M=PM, N=D, input_dram_addr=k_cache, output_dram_addr=k_cache,
                    cos_dram_addr=self.ROPE_PACKED_DRAM, sin_dram_addr=self.ROPE_PACKED_DRAM + D * bpe,
                    gpr_M_reg=self.gpr_seq_len)
            # V per kv-head -> decoder KV cache (no RoPE)
            for h in range(self.NUM_KV_HEADS):
                v_cache = self.LAYER0_V_DRAM + layer_idx * self.KV_LAYER_STRIDE + h * self.KV_HEAD_STRIDE
                strided_copy(self.LAYER0_V_PROJ_DRAM + h * D * bpe, KV * bpe, v_cache, D * bpe, PM, D)
            # Per kv-group GQA: stack Q token-major, rope, duplicate K/V, one bucket flash, un-stack
            for kv_b in range(self.NUM_KV_HEADS):
                k_cache = self.LAYER0_K_DRAM + layer_idx * self.KV_LAYER_STRIDE + kv_b * self.KV_HEAD_STRIDE
                v_cache = self.LAYER0_V_DRAM + layer_idx * self.KV_LAYER_STRIDE + kv_b * self.KV_HEAD_STRIDE
                # gather the group's 3 q-heads into token-major FLASH_Q (row t*G+g = q-head kv_b*G+g)
                for g in range(G):
                    strided_copy(self.LAYER0_Q_DRAM + (kv_b * G + g) * D * bpe, H * bpe,
                                 self.FLASH_Q_DRAM + g * D * bpe, G * D * bpe, PM, D)
                # rope stacked Q over M=PM*G using the token-major-duplicated table
                self.rope_hf_core_dram(M=PM * G, N=D, input_dram_addr=self.FLASH_Q_DRAM,
                    output_dram_addr=self.FLASH_Q_DRAM, cos_dram_addr=self.ROPE_PACKED_GQA_DRAM,
                    sin_dram_addr=self.ROPE_PACKED_GQA_DRAM + D * bpe, gpr_M_reg=self.gpr_q_seq_len)
                # duplicate K/V rows token-major into the flash buffers
                self.accelerator_memory_to_sram(k_cache, 0x10000, PM * D)
                duplicate_gqa_rows(0x10000, self.FLASH_K_DRAM)
                self.accelerator_memory_to_sram(v_cache, 0x20000, PM * D)
                duplicate_gqa_rows(0x20000, self.FLASH_V_DRAM)
                # §7 call site: jump into the shared flash subroutine (compiled after HALT). The
                # group's Q/K/V are already marshalled into the fixed FLASH_Q/K/V buffers above, so
                # this is a pure mechanical wrap (set return addr + jump; bucket primed by preamble).
                self.pad_capture_to_64b_boundary()
                return_word_addr = ue_35bit_addr_shifter(
                    prefill_addr + (self.capture_count + 2) * INSTRUCTION_SIZE_BYTES)
                self.generate_instruction_add_set(gpr_ret_id, return_word_addr)
                flash_call_sites.append(self.capture_count)
                self.generate_instruction_jump_abs(target_instruction_word_addr=0)
                # un-stack FLASH_OUT [PM,G,D] -> ATTN_RESULT [PM,H] at this group's head columns
                for g in range(G):
                    strided_copy(self.FLASH_OUT_DRAM + g * D * bpe, G * D * bpe,
                                 self.LAYER0_ATTN_RESULT_DRAM + (kv_b * G + g) * D * bpe, H * bpe, PM, D)
            # O projection + residual + RMS norm
            lm_matmul(PM, H, H, self.LAYER0_ATTN_RESULT_DRAM, 'o', la, self.LAYER0_O_PROJ_DRAM)
            self.eltwise_core_dram(M=PM, N=H, dram_a=h_in, dram_b=self.LAYER0_O_PROJ_DRAM,
                dram_out=self.LAYER0_RESIDUAL_DRAM, mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=self.gpr_seq_len)
            self.rms_norm_core_dram(M=PM, N=H, A_DRAM_ADDR=self.LAYER0_RESIDUAL_DRAM,
                OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=la['ln2_gamma'],
                gpr_M_reg=self.gpr_seq_len)
            # MLP: gate+SiLU, up, gate*up, down
            lm_matmul(PM, H, I, self.LAYER0_PRE_NORM_DRAM, 'gate', la, self.LAYER0_MLP_GATE_DRAM, silu_enable=True)
            lm_matmul(PM, H, I, self.LAYER0_PRE_NORM_DRAM, 'up', la, self.LAYER0_MLP_UP_DRAM)
            eltwise_mul_core_dram(self, size=PM * I,
                A_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM, B_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                OUTPUT_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM)
            lm_matmul(PM, I, H, self.LAYER0_MLP_MULT_DRAM, 'down', la, self.LAYER0_MLP_DOWN_DRAM)
            # Debug: dump this layer's output (residual + MLP) to its own slot; does not disturb the
            # fused path (next layer's input norm recomputes from RESIDUAL + MLP_DOWN).
            if debug_dump:
                eltwise_add_core_dram(self, size=PM * H,
                    A_DRAM_ADDR=self.LAYER0_RESIDUAL_DRAM, B_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                    OUTPUT_DRAM_ADDR=self.DBG_LAYER_DUMP + layer_idx * PM * H * bpe)
            # MLP residual: materialize the layer output (normally fused into the next layer's input
            # norm; done explicitly for the last layer, or the bisection halt layer, so it can be read).
            if layer_idx == self.NUM_LAYERS - 1 or layer_idx == halt_after_layer:
                eltwise_add_core_dram(self, size=PM * H,
                    A_DRAM_ADDR=self.LAYER0_RESIDUAL_DRAM, B_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                    OUTPUT_DRAM_ADDR=h_out)
            if halt_after_layer is not None and layer_idx == halt_after_layer:
                # Bisection: stop after this layer; h_out holds its output hidden state.
                self._prefill_v2_layer_out_buf = h_out
                break
        self.generate_instruction_halt()
        # §7: emit the shared flash subroutine AFTER the HALT (reachable only via the call-site jumps),
        # then back-patch every recorded call site to its start.
        flash_sub_start, flash_flops = self.flash_attention_core(head_dim=D, seq_len=qmax,
            Q_DRAM_ADDR=self.FLASH_Q_DRAM, K_DRAM_ADDR=self.FLASH_K_DRAM,
            V_DRAM_ADDR=self.FLASH_V_DRAM, OUTPUT_DRAM_ADDR=self.FLASH_OUT_DRAM,
            SCRATCH_DRAM_ADDR=self.FLASH_SCRATCH_DRAM, ATTN_P_DRAM_ADDR=self.FLASH_ATTN_P_DRAM,
            IDENTITY_DRAM_ADDR=self.identity_addr, BIAS_DRAM_ADDR=self.FLASH_BIAS_DRAM,
            gpr_bucket_idx=self.gpr_bucket_idx, num_buckets=num_buckets, gpr_ret_id=gpr_ret_id)
        for _idx in flash_call_sites:
            self._patch_jump_immediate(_idx, ue_35bit_addr_shifter(flash_sub_start))
        self.release_isa_reg()  # gpr_ret_id
        self.stop_capture()
        self.write_captured_instructions_to_dram(prefill_addr)
        size_bytes = self.get_capture_instruction_size_bytes()
        # Cache the full program (gemma3 convention) so later runs load in ~1s instead of recompiling.
        # Only the canonical program (no debug dump / no early halt). Under compile_all, the bytes are
        # stashed for the unified instructions.bin instead of a separate prefill_v2_program.bin.
        if halt_after_layer is None and not debug_dump:
            raw = bytearray()
            for inst in self.capture_buffer:
                raw.extend(inst.get_bytes())
            if self._unified_active:
                self._seg_prefill = (prefill_addr, bytes(raw))
            else:
                bin_path = os.path.join(self.script_dir, "smolvlm2_bin", "prefill_v2_program.bin")
                with open(bin_path, "wb") as f:
                    f.write(bytes(raw))
        self.allocate_program_dram(size_bytes)
        self.clear_capture_buffer()
        self._prefill_v2_addr = prefill_addr
        # Reserve preamble (embed/merge + reg prime + jump) and postamble (final-norm + LM head)
        # scratch regions; both are captured fresh per run.
        self._prefill_v2_preamble_addr = self.get_program_dram_addr()
        self.allocate_program_dram(PM * 2 * 32 + IMAGE_SEQ_LEN * 2 * 32 + 256)
        self._prefill_v2_postamble_addr = self.get_program_dram_addr()
        self.allocate_program_dram(4096)
        # Final hidden lands in the buffer matching the layer ping-pong parity.
        self._prefill_v2_final_buf = self.LAYER0_INPUT_DRAM if self.NUM_LAYERS % 2 == 0 else self.LAYER0_OUTPUT_DRAM
        print(f"    Prefill v2 compiled @0x{prefill_addr:X}: {size_bytes} bytes, "
              f"{num_buckets} buckets, PM={PM}, qmax={qmax}")

    def _replay_encoder_ln_params(self) -> None:
        """Replay compile_encoder's lazy layer_norm zero-buffer allocations (LN1 + LN2 per layer + one
        final post-layernorm) so the params-DRAM pointer + baked addresses match a fresh compile. Used by
        the cached-load path (compile_all cache hit / run_from_bin). The unrolled PBI encoder emits
        layer_norm_core_dram (PBI) for every LN, each allocating one N-bf16 zeros buffer; the §7 flash
        body + matmat/eltwise allocate NO params, so these LN zeros are the encoder's ONLY lazy params
        (traced: N_vis*2+1)."""
        N, bpe = 768, 2
        zeros = torch.zeros(N, dtype=torch.bfloat16)
        for _ in range(len(self.vis_layer_addrs)):
            for _ in range(2):  # LN1 + LN2
                addr = self.get_params_dram_addr()
                self.allocate_params_dram(N * bpe)
                self.dma_write(DMA_DEVICE_H2C, addr, zeros, N * bpe)
        addr = self.get_params_dram_addr()  # final post-layernorm
        self.allocate_params_dram(N * bpe)
        self.dma_write(DMA_DEVICE_H2C, addr, zeros, N * bpe)

    def _restore_unified_addrs(self, meta: dict) -> None:
        """Set the program/preamble addresses the run_* methods read, from the unified-bin meta."""
        self._vis_program_addr = meta["encoder_addr"]
        self._decoder_program_addr = meta["decoder_addr"]
        self._decoder_preamble_addr = meta["decoder_preamble"]
        self._decoder_num_buckets = meta["decoder_num_buckets"]
        self._prefill_v2_addr = meta["prefill_addr"]
        self._prefill_v2_preamble_addr = meta["prefill_preamble"]
        self._prefill_v2_postamble_addr = meta["prefill_postamble"]
        self._prefill_v2_final_buf = meta["prefill_final_buf"]
        self._next_program_dram_addr = meta["end_addr"]

    def compile_all(self) -> dict:
        """Build (or load) the COMPLETE single instruction bin — encoder + decoder + prefill_v2 in ONE
        instructions.bin. Mirrors qwen2.5_vl compile_all. The compile order (encoder, decoder,
        prefill_v2) matches the validated 3-bin order, so every §7 JUMP_ABS bakes against the exact
        program-DRAM address it did before; the bin just bundles the three segments, each recorded with
        its absolute load address (see fpga_pbi_jump_target_bake). Cache key = layout signature."""
        bin_dir = os.path.join(self.script_dir, "smolvlm2_bin")
        os.makedirs(bin_dir, exist_ok=True)
        uni_bin = os.path.join(bin_dir, "instructions.bin")
        uni_meta = os.path.join(bin_dir, "instructions.json")
        sig = {"prefill_max_seq_len": self.PREFILL_MAX_SEQ_LEN, "num_layers": self.NUM_LAYERS,
               "num_vis_layers": len(self.vis_layer_addrs), "lm_head": "penalty_bias_argmax"}

        # ---- cache hit: replay the encoder LN params, DMA each segment to its stored abs addr ----
        if os.path.exists(uni_bin) and os.path.exists(uni_meta):
            with open(uni_meta) as f:
                meta = json.load(f)
            if all(meta.get(k) == v for k, v in sig.items()):
                self._replay_encoder_ln_params()
                with open(uni_bin, "rb") as f:
                    raw = f.read()
                for seg in meta["segments"]:
                    b = raw[seg["off"]:seg["off"] + seg["size"]]
                    self.dma_write(DMA_DEVICE_H2C, seg["addr"], b, len(b))
                self._restore_unified_addrs(meta)
                print(f"  Loaded unified instruction bin ({len(raw)/1024/1024:.1f} MB, "
                      f"{len(meta['segments'])} segments)")
                return meta
            print(f"  instructions.bin layout signature {dict((k, meta.get(k)) for k in sig)} "
                  f"≠ current {sig} — rebuilding.")

        # ---- cache miss: compile all three in ONE session (encoder, decoder, prefill_v2) ----
        self._unified_active = True
        self._seg_encoder = self._seg_prefill = self._seg_decoder = None
        try:
            enc_addr = self.compile_encoder()
            self.compile_decoder()
            self.compile_prefill_v2()
        finally:
            self._unified_active = False
        end_addr = self.get_program_dram_addr()
        segments, raw = [], bytearray()
        for name, seg in (("encoder", self._seg_encoder), ("decoder", self._seg_decoder),
                          ("prefill", self._seg_prefill)):
            if seg is None:
                continue
            addr, b = seg
            segments.append({"name": name, "addr": addr, "off": len(raw), "size": len(b)})
            raw.extend(b)
        meta = {**sig, "segments": segments, "end_addr": end_addr,
                "encoder_addr": enc_addr,
                "decoder_addr": self._decoder_program_addr,
                "decoder_preamble": self._decoder_preamble_addr,
                "decoder_num_buckets": self._decoder_num_buckets,
                "prefill_addr": self._prefill_v2_addr,
                "prefill_preamble": self._prefill_v2_preamble_addr,
                "prefill_postamble": self._prefill_v2_postamble_addr,
                "prefill_final_buf": self._prefill_v2_final_buf}
        with open(uni_bin, "wb") as f:
            f.write(raw)
        with open(uni_meta, "w") as f:
            json.dump(meta, f)
        print(f"  Complete instruction bin: {len(raw)/1024/1024:.1f} MB, "
              f"{len(segments)} segments ({'+'.join(s['name'] for s in segments)}) → {uni_bin}")
        return meta

    def load_prefill_v2(self) -> None:
        """Load the cached seq-len-agnostic prefill program. Replays compile_prefill_v2's exact
        program-DRAM allocations so the body lands at the same address its flash absolute-jumps
        were baked against (must be called at the same point in the alloc sequence: after the
        decoder is loaded)."""
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

    def run_prefill_v2(self, token_ids, has_image: bool = False, total_flops: int = None,
                       skip_postamble: bool = False) -> int:
        """Run the seq-len-agnostic prefill: host prep (epsilon pad + block bias), a preamble that
        does on-device embed/merge + primes gpr_bucket_idx + jumps into the cached prefill, then a
        postamble that does final-norm + LM head on the runtime last token. Returns argmax.

        skip_postamble=True runs only the preamble->prefill (for bisection: read intermediate
        buffers afterward) and returns None."""
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
        if skip_postamble:
            return None  # bisection: caller reads intermediate buffers

        # 5. Postamble: final norm over ALL tokens (identical to the proven old prefill) + LM head on
        # the last real token. (RMSNorm is per-row, so M=S then index == M=1 on the last row; using
        # M=S removes the M=1 path as a variable.)
        last_off = (seq_len - 1) * H * bpe
        self.clear_inst_id()
        self.start_capture()
        self.rms_norm_core_dram(M=seq_len, N=H, A_DRAM_ADDR=self._prefill_v2_final_buf,
            OUTPUT_DRAM_ADDR=self.FINAL_NORM_DRAM, GAMMA_DRAM_ADDR=self.final_norm_addr)
        last_norm = self.FINAL_NORM_DRAM + last_off
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

    def compile_decoder(self) -> None:
        """Compile single decoder program with runtime kv_len dispatch via gpr_bucket_idx."""
        from user_dma_core import TYPE
        H, D, I = self.HIDDEN_SIZE, self.HEAD_DIM, self.INTERMEDIATE_SIZE
        bpe = 2
        G = self.GROUP_SIZE
        num_buckets = (self.max_seq_len + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE

        def lm_matmul(M, K, N, A, proj, la, OUT, **kw):
            self.quantized_matmat_core(M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f'{proj}_data'],
                OUTPUT_DRAM_ADDR=OUT, SCALE_DRAM_ADDR=la[f'{proj}_scale'], data_type=TYPE.IF4, **kw)

        # Guarantee counter starts at 1 regardless of prior compilations.
        self.reset_isa_reg_counter()
        # Reserve fixed ISA registers (1-6) so alloc_isa_reg() inside
        # decoder_group_attention_core_pbi / matmat_mul_core never clobbers
        # V_CACHE_SIZE_REG(1), ROPE_SIZE_REG(2), TMP_REG(3),
        # gpr_seq_len(4), gpr_q_seq_len(5), gpr_bucket_idx(6) at runtime.
        _NUM_FIXED_REGS = 6
        for _ in range(_NUM_FIXED_REGS):
            self.alloc_isa_reg()

        global _SILENT_MODE
        _SILENT_MODE = True
        self.clear_inst_id()
        self.start_capture()
        # §7 shared-subroutine decode attention: the bucketized decoder-group-attention body is
        # compiled ONCE after the HALT; every per-(layer,kv-group) call site marshals its K/V history
        # + Q into the fixed FLASH buffers and jumps in (priming gpr_ret_id as the return address).
        # gpr_bucket_idx is primed once per step by run_decoder's preamble. dec_program_base equals
        # _decoder_program_addr (no program-DRAM advances during capture).
        dec_program_base = self.get_program_dram_addr()
        gpr_ret_id = self.alloc_isa_reg()
        dec_call_sites: list[int] = []

        for layer_idx in range(self.NUM_LAYERS):
            la = self.lm_layer_addrs[layer_idx]
            h_in  = self.LAYER0_INPUT_DRAM if layer_idx % 2 == 0 else self.LAYER0_OUTPUT_DRAM
            h_out = self.LAYER0_OUTPUT_DRAM if layer_idx % 2 == 0 else self.LAYER0_INPUT_DRAM

            # RMS norm (input_layernorm)
            self.rms_norm_core_dram(M=1, N=H, A_DRAM_ADDR=h_in,
                OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=la['ln1_gamma'])
            # Q/K/V projections
            lm_matmul(1, H, H,                     self.LAYER0_PRE_NORM_DRAM, 'q', la, self.LAYER0_Q_DRAM)
            lm_matmul(1, H, self.NUM_KV_HEADS * D, self.LAYER0_PRE_NORM_DRAM, 'k', la, self.LAYER0_K_PROJ_DRAM)
            lm_matmul(1, H, self.NUM_KV_HEADS * D, self.LAYER0_PRE_NORM_DRAM, 'v', la, self.LAYER0_V_PROJ_DRAM)

            # Position-based byte offsets are gpr_seq_len * stride. Following gemma3's decode
            # convention, recompute the offset register(s) immediately before each use rather
            # than once per layer, so no intervening op can leave a stale value in the offset
            # register between the compute site and the consuming DMA/RoPE.
            #   V_CACHE_SIZE_REG = gpr_seq_len * k_size   (byte offset into KV cache)
            #   ROPE_SIZE_REG    = gpr_seq_len * D * bpe   (byte offset into cos/sin tables)

            # Write V to KV cache at current position
            for h in range(self.NUM_KV_HEADS):
                v_base = self.LAYER0_V_DRAM + layer_idx * self.KV_LAYER_STRIDE + h * self.KV_HEAD_STRIDE
                self.accelerator_memory_to_sram(self.LAYER0_V_PROJ_DRAM + h * D * bpe, 0x10000, D)
                # Compute full destination address into TMP_REG immediately before the store.
                self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gpr_seq_len, ue_35bit_addr_shifter(self.k_size))
                self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(v_base), self.TMP_REG)
                self.sram_to_accelerator_memory(0x10000, 0, D, general_reg_src=self.TMP_REG)

            # RoPE K: d64 packed-table rope, written straight into the KV cache at the current
            # position. Same convention as prefill's rope_hf_core_dram_d64_pbi (packed [cos||sin],
            # sin first-half pre-negated, every URAM slice row-aligned). Table + output offsets are
            # derived from gpr_seq_len inside the helper via register-addressed DMAs.
            for h in range(self.NUM_KV_HEADS):
                k_base = self.LAYER0_K_DRAM + layer_idx * self.KV_LAYER_STRIDE + h * self.KV_HEAD_STRIDE
                self.rope_hf_core_decode(N=D,
                    input_dram_addr=self.LAYER0_K_PROJ_DRAM + h * D * bpe,
                    output_dram_addr=k_base, output_pos_strided=True,
                    packed_table_addr=self.ROPE_PACKED_DRAM,
                    pos_reg=self.gpr_seq_len, tmp_reg=self.TMP_REG)

            # RoPE Q: same d64 rope, output to the fixed Q_PERM slot (attention reads from there).
            for h in range(self.NUM_HEADS):
                self.rope_hf_core_decode(N=D,
                    input_dram_addr=self.LAYER0_Q_DRAM + h * D * bpe,
                    output_dram_addr=self.LAYER0_Q_PERM_DRAM + h * D * bpe, output_pos_strided=False,
                    packed_table_addr=self.ROPE_PACKED_DRAM,
                    pos_reg=self.gpr_seq_len, tmp_reg=self.TMP_REG)

            # GQA decode attention (§7): per kv-group, marshal K/V history + Q into the fixed FLASH
            # buffers, jump into the shared decoder-attention subroutine, copy the head output back.
            for kv_b in range(self.NUM_KV_HEADS):
                q_start = kv_b * G * D * bpe
                k_cache_base = self.LAYER0_K_DRAM + layer_idx * self.KV_LAYER_STRIDE + kv_b * self.KV_HEAD_STRIDE
                v_cache_base = self.LAYER0_V_DRAM + layer_idx * self.KV_LAYER_STRIDE + kv_b * self.KV_HEAD_STRIDE
                # Gather the valid K/V history (gpr_bucket_idx buckets of UE_VECTOR_SIZE tokens) into the
                # fixed FLASH_K/FLASH_V buffers — both are contiguous [seq, D] in the per-head cache, so
                # one runtime PBI loop (trip = gpr_bucket_idx) copies them, leaving the bin seq-agnostic.
                self._emit_pbi_scatter_per_token(
                    read_base=k_cache_base, read_stride_bytes=UE_VECTOR_SIZE * D * bpe,
                    write_specs=[(self.FLASH_K_DRAM, UE_VECTOR_SIZE * D * bpe)],
                    sram_byte_addr=0x50000, element_count=UE_VECTOR_SIZE * D,
                    gpr_seq_len=self.gpr_bucket_idx, template_seq_len=num_buckets)
                self._emit_pbi_scatter_per_token(
                    read_base=v_cache_base, read_stride_bytes=UE_VECTOR_SIZE * D * bpe,
                    write_specs=[(self.FLASH_V_DRAM, UE_VECTOR_SIZE * D * bpe)],
                    sram_byte_addr=0x60000, element_count=UE_VECTOR_SIZE * D,
                    gpr_seq_len=self.gpr_bucket_idx, template_seq_len=num_buckets)
                # Scatter the group's Q [G, D] into the fixed FLASH_Q base (drop the per-group offset).
                self.accelerator_memory_to_sram(self.LAYER0_Q_PERM_DRAM + q_start, 0x30000, G * D)
                self.sram_to_accelerator_memory(0x30000, self.FLASH_Q_DRAM, G * D)
                # §7 call site: prime return addr + jump into the shared decoder-attention subroutine.
                self.pad_capture_to_64b_boundary()
                return_word_addr = ue_35bit_addr_shifter(
                    dec_program_base + (self.capture_count + 2) * INSTRUCTION_SIZE_BYTES)
                self.generate_instruction_add_set(gpr_ret_id, return_word_addr)
                dec_call_sites.append(self.capture_count)
                self.generate_instruction_jump_abs(target_instruction_word_addr=0)
                # Copy the group output [G, D] back to this group's columns in ATTN_OUT.
                self.accelerator_memory_to_sram(self.FLASH_OUT_DRAM, 0x40000, G * D)
                self.sram_to_accelerator_memory(0x40000, self.LAYER0_ATTN_OUT_DRAM + q_start, G * D)

            # O projection
            lm_matmul(1, H, H, self.LAYER0_ATTN_OUT_DRAM, 'o', la, self.LAYER0_O_PROJ_DRAM)
            # Post-attn residual + RMS norm
            rms_norm_core_dram_post_add(self, M=1, N=H,
                A_DRAM_ADDR=h_in, B_DRAM_ADDR=self.LAYER0_O_PROJ_DRAM,
                ADDOUTPUT_DRAM_ADDR=self.LAYER0_RESIDUAL_DRAM,
                NORMOUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                GAMMA_DRAM_ADDR=la['ln2_gamma'])
            # MLP
            lm_matmul(1, H, I, self.LAYER0_PRE_NORM_DRAM, 'gate', la, self.LAYER0_MLP_GATE_DRAM, silu_enable=True)
            lm_matmul(1, H, I, self.LAYER0_PRE_NORM_DRAM, 'up',   la, self.LAYER0_MLP_UP_DRAM)
            self.accelerator_memory_to_sram(self.LAYER0_MLP_GATE_DRAM, 0x10000, I)
            self.accelerator_memory_to_sram(self.LAYER0_MLP_UP_DRAM,   0x90000, I)
            self.eltwise_mul_core(0x10000, 0x90000, 0x10000, I)
            self.sram_to_accelerator_memory(0x10000, self.LAYER0_MLP_MULT_DRAM, I)
            lm_matmul(1, I, H, self.LAYER0_MLP_MULT_DRAM, 'down', la, self.LAYER0_MLP_DOWN_DRAM)
            # MLP residual
            self.accelerator_memory_to_sram(self.LAYER0_RESIDUAL_DRAM,  0x10000, H)
            self.accelerator_memory_to_sram(self.LAYER0_MLP_DOWN_DRAM,  0x90000, H)
            self.eltwise_add_core(0x10000, 0x90000, 0x10000, H)
            self.sram_to_accelerator_memory(0x10000, h_out, H)

        # Final norm + LM head
        final_buf = self.LAYER0_INPUT_DRAM if self.NUM_LAYERS % 2 == 0 else self.LAYER0_OUTPUT_DRAM
        self.rms_norm_core_dram(M=1, N=H, A_DRAM_ADDR=final_buf,
            OUTPUT_DRAM_ADDR=self.FINAL_NORM_DRAM, GAMMA_DRAM_ADDR=self.final_norm_addr)
        # On-FPGA repetition penalty (llama3.2-style): the LM-head adds PENALTY_BIAS_DRAM as its per-vocab
        # C bias (broadcast_N), so the HW argmax already returns the penalized token — no logit readback.
        # The bias is always wired (zeros = pure greedy), so there is ONE decoder bin. write_back_disable
        # skips the LOGITS_DRAM write (decode reads only the argmax register).
        self.quantized_matmat_core(M=1, K=H, N=self.VOCAB_SIZE, A_DRAM_ADDR=self.FINAL_NORM_DRAM,
            B_DRAM_ADDR=self.lm_head_data, OUTPUT_DRAM_ADDR=self.LOGITS_DRAM,
            SCALE_DRAM_ADDR=self.lm_head_scale, data_type=TYPE.IF4,
            C_DRAM_ADDR=self.PENALTY_BIAS_DRAM, bias_mode="broadcast_N", write_back_disable=True)

        # Advance token position on-device so next decode step writes to the right KV slot
        self.generate_instruction_add_inc(self.gpr_seq_len)
        self.generate_instruction_halt()
        # §7: emit the shared decoder-attention subroutine AFTER the HALT (reachable only via the
        # call-site jumps), then back-patch every recorded call site to its start. use_pbi=False keeps
        # the legacy flash body (avoids PBI-flash back-to-back corruption); identical numerics to the
        # prior inline call, just one copy instead of NUM_LAYERS*NUM_KV_HEADS.
        dec_sub_start, _dec_flops = self.decoder_group_attention_core_pbi(
            group_size=G, head_dim=D,
            Q_DRAM_ADDR=self.FLASH_Q_DRAM, K_DRAM_ADDR=self.FLASH_K_DRAM, V_DRAM_ADDR=self.FLASH_V_DRAM,
            OUTPUT_DRAM_ADDR=self.FLASH_OUT_DRAM, SCRATCH_DRAM_ADDR=self.LAYER0_ATTN_SCRATCH_DRAM,
            BIAS_DRAM_ADDR=self.DECODE_BIAS_DRAM, IDENTITY_DRAM_ADDR=self.identity_addr,
            gpr_bucket_idx=self.gpr_bucket_idx, num_buckets=num_buckets, use_pbi=False,
            gpr_ret_id=gpr_ret_id)
        for _idx in dec_call_sites:
            self._patch_jump_immediate(_idx, ue_35bit_addr_shifter(dec_sub_start))
        self.release_isa_reg()  # gpr_ret_id
        self.stop_capture()
        for _ in range(_NUM_FIXED_REGS):
            self.release_isa_reg()
        _SILENT_MODE = False

        raw = bytearray()
        for inst in self.capture_buffer:
            raw.extend(inst.get_bytes())
        self.clear_capture_buffer()

        bin_dir = os.path.join(self.script_dir, "smolvlm2_bin")
        # Load into program DRAM for immediate use (same session)
        self._decoder_program_addr = self.get_program_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, self._decoder_program_addr, bytes(raw), len(raw))
        self.allocate_program_dram(len(raw))
        self._decoder_preamble_addr = self.get_program_dram_addr()
        self.allocate_program_dram(256)
        self._decoder_num_buckets = num_buckets
        if self._unified_active:
            self._seg_decoder = (self._decoder_program_addr, bytes(raw))
        else:
            with open(os.path.join(bin_dir, "decoder_program.bin"), "wb") as f:
                f.write(bytes(raw))
            with open(os.path.join(bin_dir, "decoder_program.json"), "w") as f:
                json.dump({"decoder_program_size": len(raw), "num_buckets": num_buckets, "version": 2}, f)
        print(f"    Decoder compiled: single program {len(raw)} bytes at 0x{self._decoder_program_addr:X}, {num_buckets} attention buckets")

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

        num_vis_layers = len(self.vis_layer_addrs)
        # §7: the shared flash body uses the pre-seeded identity_addr and allocates NO params, so the
        # old per-layer "12× flash identity matrices" replay is gone. compile_encoder's only lazy
        # params are the layer_norm zero buffers: each layer allocates LN1 (layer_norm_core_dram) +
        # LN2 (layer_norm_core_dram_post_add), then one final post-layernorm — traced as 25 × (N*bpe).
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

    def dump_snapshot(self) -> None:
        """Dump params DRAM + all runtime address metadata to smolvlm2_bin/weights.bin + weights.json."""
        bin_dir = os.path.join(self.script_dir, "smolvlm2_bin")
        bin_path = os.path.join(bin_dir, "weights.bin")
        meta_path = os.path.join(bin_dir, "weights.json")
        total = self.get_params_dram_usage()
        CHUNK = 1 * 1024 * 1024
        with open(bin_path, "wb") as f:
            offset = 0
            while offset < total:
                sz = min(CHUNK, total - offset)
                buf = bytearray(sz)
                self.dma_read(DMA_DEVICE_C2H, self._params_dram_base + offset, buf, sz)
                f.write(buf)
                offset += sz
        addr_attrs = [
            "embed_addr", "ROPE_COS_DRAM", "ROPE_SIN_DRAM", "PERMUTE_PARAMS_DRAM",
            "LAYER0_K_DRAM", "LAYER0_V_DRAM", "LAYER0_INPUT_DRAM", "LAYER0_OUTPUT_DRAM",
            "LAYER0_PRE_NORM_DRAM", "LAYER0_Q_DRAM", "LAYER0_K_PROJ_DRAM", "LAYER0_V_PROJ_DRAM",
            "LAYER0_Q_PERM_DRAM", "LAYER0_ATTN_OUT_DRAM", "LAYER0_ATTN_SCRATCH_DRAM",
            "LAYER0_ATTN_RESULT_DRAM", "LAYER0_O_PROJ_DRAM", "LAYER0_RESIDUAL_DRAM",
            "LAYER0_MLP_GATE_DRAM", "LAYER0_MLP_UP_DRAM", "LAYER0_MLP_MULT_DRAM",
            "LAYER0_MLP_DOWN_DRAM", "FINAL_NORM_DRAM", "LOGITS_DRAM", "PENALTY_BIAS_DRAM",
            "CAUSAL_MASK_DRAM", "DECODE_BIAS_DRAM",
            "VIS_PIXEL_IN_DRAM", "VIS_PATCH_PERM_DRAM", "VIS_PATCH_PROJ_DRAM",
            "VIS_IO_A_DRAM", "VIS_IO_B_DRAM", "VIS_LN_OUT_DRAM",
            "VIS_Q_DRAM", "VIS_K_DRAM", "VIS_V_DRAM",
            "VIS_Q_PERM_DRAM", "VIS_K_PERM_DRAM", "VIS_V_PERM_DRAM",
            "VIS_ATTN_OUT_DRAM", "VIS_ATTN_SCRATCH_DRAM", "VIS_ATTN_RESULT_DRAM",
            "VIS_O_PROJ_DRAM", "VIS_RESIDUAL_DRAM", "VIS_MLP_INTER_DRAM", "VIS_MLP_OUT_DRAM",
            "VIS_POST_LN_DRAM", "VIS_SHUFFLED_DRAM", "VIS_CONNECTOR_DRAM", "PERMUTE_TEMP_DRAM",
            # Prefill v2 (gemma3-style) RoPE tables + stacked-GQA flash buffers
            "ROPE_PACKED_DRAM", "ROPE_PACKED_GQA_DRAM",
            "FLASH_Q_DRAM", "FLASH_K_DRAM", "FLASH_V_DRAM", "FLASH_OUT_DRAM",
            "FLASH_SCRATCH_DRAM", "FLASH_ATTN_P_DRAM", "FLASH_BIAS_DRAM",
        ]
        meta = {
            "params_size": total,
            "tensor_size": self.get_tensor_dram_usage(),
            "max_seq_len": self.max_seq_len,
            "k_size": self.k_size,
            "bytes_per_element": self.bytes_per_element,
            "KV_HEAD_STRIDE": self.KV_HEAD_STRIDE,
            "KV_LAYER_STRIDE": self.KV_LAYER_STRIDE,
            "vision_bf16": self.vision_bf16,
            "_num_vis_layers": len(self.vis_layer_addrs),
            "PREFILL_QMAX": self.PREFILL_QMAX,
            "FLASH_NUM_BUCKETS": self.FLASH_NUM_BUCKETS,
            # LM weight addresses: run_prefill_v2 re-captures the decoder + LM postamble fresh
            # each run, so it needs the real per-layer/final/lm-head addresses (NOT just a count).
            "lm_layer_addrs": self.lm_layer_addrs,
            "final_norm_addr": self.final_norm_addr,
        }
        meta["lm_head_data"] = self.lm_head_data
        meta["lm_head_scale"] = self.lm_head_scale
        for attr in addr_attrs:
            meta[attr] = getattr(self, attr)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        _original_print(f"  Snapshot dumped: {total / 1024**2:.1f} MB → {bin_path}")

    def load_snapshot(self) -> bool:
        """Load params DRAM from snapshot bin + restore all address metadata. Returns True if loaded."""
        bin_dir = os.path.join(self.script_dir, "smolvlm2_bin")
        bin_path = os.path.join(bin_dir, "weights.bin")
        meta_path = os.path.join(bin_dir, "weights.json")
        if not os.path.exists(bin_path) or not os.path.exists(meta_path):
            return False
        with open(meta_path) as f:
            meta = json.load(f)
        # Self-heal stale snapshots: ones dumped before LM weight addresses were saved lack
        # these keys and would crash run_prefill_v2. Force a recompile + fresh dump instead.
        if "lm_layer_addrs" not in meta or "final_norm_addr" not in meta:
            _original_print("  Snapshot stale (missing LM addresses) — recompiling.")
            return False
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
        self.bytes_per_element = 2  # always bf16; not in snapshot, needed by attention PBI body
        # load_encoder/run_encoder only need len(vis_layer_addrs) (the per-layer weight addresses
        # are already baked into the precompiled encoder_program.bin). Restore a length-correct
        # placeholder list from the saved layer count so those len() calls work post-snapshot.
        self.vis_layer_addrs = [None] * self._num_vis_layers
        from transformers import AutoTokenizer
        model_dir = os.path.join(self.script_dir, "smolvlm2_bin", "SmolVLM2-500M-Video-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        kv_zeros = torch.zeros(self.NUM_LAYERS * self.NUM_KV_HEADS * self.max_seq_len * self.HEAD_DIM, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_K_DRAM, kv_zeros)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, kv_zeros)
        _original_print(f"  Snapshot loaded: {total / 1024**2:.1f} MB")
        return True

    def load_decoder(self) -> None:
        """Load pre-compiled decoder program from bin."""
        bin_dir = os.path.join(self.script_dir, "smolvlm2_bin")
        meta_path = os.path.join(bin_dir, "decoder_program.json")
        with open(meta_path) as f:
            meta = json.load(f)
        if "num_buckets" not in meta:
            raise RuntimeError(
                "decoder_program.json is from the old bucket format; "
                "delete decoder_program.bin and decoder_program.json to recompile"
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
        n_vis = len(self.vis_layer_addrs)
        attn_per_head = VS * VD + 2 * VS * VD * VS + VS * VS * 5 + 2 * VS * VS * VD
        per_layer = (7 * VS * VH + 3 * (2 * VS * VH * VH + VS * VH)
            + VN * attn_per_head + 2 * VS * VH * VH + VS * VH
            + 8 * VS * VH + 2 * VS * VH * VI + VS * VI
            + 2 * VS * VI * VH + VS * VH + VS * VH)
        enc_flops = (2 * VS * VH * VH + VS * VH + VS * VH
            + n_vis * per_layer + 7 * VS * VH
            + 2 * 64 * 12288 * self.HIDDEN_SIZE)
        self.program_execute(encoder_addr, timeout=30.0, total_flops=enc_flops)
    def _clear_scratch_dram(self) -> None:
        """Zero all decoder scratch regions (not KV cache) between decode steps."""
        H, I, D = self.HIDDEN_SIZE, self.INTERMEDIATE_SIZE, self.HEAD_DIM
        bpe = 2
        S = 1  # decode uses seq_len=1 scratch
        KV = self.NUM_KV_HEADS * D
        z_H   = torch.zeros(H,   dtype=torch.bfloat16)
        z_KV  = torch.zeros(KV,  dtype=torch.bfloat16)
        z_I   = torch.zeros(I,   dtype=torch.bfloat16)
        z_scr = torch.zeros(self.HEAD_DIM * 1 + max(self.HEAD_DIM, 64) * 1, dtype=torch.bfloat16)
        z_V   = torch.zeros(self.VOCAB_SIZE, dtype=torch.bfloat16)
        for addr in (self.LAYER0_INPUT_DRAM, self.LAYER0_OUTPUT_DRAM,
                     self.LAYER0_PRE_NORM_DRAM, self.LAYER0_Q_DRAM,
                     self.LAYER0_Q_PERM_DRAM, self.LAYER0_ATTN_OUT_DRAM,
                     self.LAYER0_ATTN_RESULT_DRAM, self.LAYER0_O_PROJ_DRAM,
                     self.LAYER0_RESIDUAL_DRAM, self.LAYER0_MLP_DOWN_DRAM,
                     self.FINAL_NORM_DRAM):
            self.dma_to_accelerator_memory(addr, z_H)
        for addr in (self.LAYER0_K_PROJ_DRAM, self.LAYER0_V_PROJ_DRAM):
            self.dma_to_accelerator_memory(addr, z_KV)
        for addr in (self.LAYER0_MLP_GATE_DRAM, self.LAYER0_MLP_UP_DRAM, self.LAYER0_MLP_MULT_DRAM):
            self.dma_to_accelerator_memory(addr, z_I)
        self.dma_to_accelerator_memory(self.LAYER0_ATTN_SCRATCH_DRAM, z_scr)
        self.dma_to_accelerator_memory(self.LOGITS_DRAM, z_V)

    def _structural_token_ids(self):
        """Token ids that must NEVER be repetition-penalized: punctuation, whitespace, and special
        tokens (image/fake/global/end-of-utterance). These 'glue' tokens recur in any text; penalizing
        them over a long generation is what starves a small model of grammar and yields word-salad.
        Computed once from the vocab + config special tokens, cached."""
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
        """Build the per-vocab additive repetition bias from windowed token frequency and DMA it to
        PENALTY_BIAS_DRAM (the decode LM-head's C term, broadcast_N). bias[t] = clamp(−alpha·count[t],
        min=−cap); structural tokens stay 0. The HW argmax of (logits + bias) returns the penalized
        token — no logit readback. One full-buffer DMA per step (incremental writes pay more in device
        open/close than the tiny transfer saves)."""
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

    def run_decoder(self, token_id: int, max_new_tokens: int = 512, clear_scratch: bool = False) -> list:
        """Auto-regressive decode loop. Returns generated token IDs.
        clear_scratch=True zeros all scratch DRAM between steps (diagnostic)."""
        global _SILENT_MODE
        generated = []
        H = self.HIDDEN_SIZE
        bpe = 2
        embed_row_bytes = H * bpe
        num_buckets = self._decoder_num_buckets
        decoder_program_addr = self._decoder_program_addr
        preamble_addr = self._decoder_preamble_addr

        # On-FPGA repetition penalty: track every token seen (prompt + the prefill seed token + decoded)
        # for the windowed bias. The decode LM-head always adds PENALTY_BIAS_DRAM as its C term, so zero
        # it up front (→ pure greedy until the greedy_until gate), then refresh per step past the gate.
        _fpga_penalty = bool(getattr(self, "fpga_penalty", True))
        _greedy_until = int(getattr(self, "greedy_until", 0))
        self._generated_tokens = list(getattr(self, "_prefill_token_ids", [])) + [token_id]
        self.dma_to_accelerator_memory(self.PENALTY_BIAS_DRAM,
                                       torch.zeros(1, self.VOCAB_SIZE, dtype=torch.bfloat16))
        if _fpga_penalty:
            self._structural_ids_tensor()  # warm the structural-token cache off the first decode token

        # Live t/s counter (same as gemma3): pin the bottom terminal row as a status line
        # via an ANSI scroll region; tokens stream above it and the rate refreshes in place.
        # Only when stdout is a real TTY (skip when piped/redirected).
        import shutil
        timer = time.perf_counter()
        _use_status = sys.stdout.isatty()
        def _status_setup():
            rows = shutil.get_terminal_size().lines
            sys.stdout.write(f"\033[1;{rows - 1}r")    # scroll region = rows 1..rows-1
            sys.stdout.write(f"\033[{rows - 1};1H")    # park cursor at bottom of region
            sys.stdout.flush()
        def _status_update():
            rows = shutil.get_terminal_size().lines
            elapsed = time.perf_counter() - timer
            rate = len(generated) / elapsed if elapsed > 0 else 0.0
            sys.stdout.write("\0337")                   # save cursor
            sys.stdout.write(f"\033[{rows};1H\033[2K")   # bottom row, clear it
            sys.stdout.write(f" decoding… {len(generated)} tokens  (pos {self.seq_len}/{self.max_seq_len})  "
                             f"{elapsed:.1f}s  {rate:.1f} tok/s")
            sys.stdout.write("\0338")                   # restore cursor
            sys.stdout.flush()
        def _status_teardown():
            rows = shutil.get_terminal_size().lines
            sys.stdout.write("\033[r")                  # reset scroll region
            sys.stdout.write(f"\033[{rows};1H\033[2K")   # clear the status row
            sys.stdout.flush()
        if _use_status:
            _status_setup()
        # Emit the prefill seed token now (after scroll-region setup, so it isn't clobbered).
        _original_print(self.tokenizer.decode([token_id]), end="", flush=True)

        while len(generated) < max_new_tokens and self.seq_len < self.max_seq_len:
            _SILENT_MODE = True

            if clear_scratch:
                _original_print(f"[dbg] step={len(generated)+1} clearing scratch DRAM before exec")
                self._clear_scratch_dram()

            self.seq_len += 1
            aligned_kv = ((self.seq_len + 63) // 64) * 64
            bucket_idx = min(aligned_kv // UE_VECTOR_SIZE, num_buckets)

            # Decode bias: [GROUP_SIZE, max_seq_len] with zeros at valid KV positions
            bias = torch.full((self.GROUP_SIZE, self.max_seq_len), -1e38, dtype=torch.bfloat16)
            bias[:, :self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.DECODE_BIAS_DRAM, bias)

            # Host-side embed DMA (same as gemma3) — avoids on-device C2H before decoder
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

            # Refresh the per-vocab penalty bias (this step's LM-head C term) from the windowed token
            # frequency once past the greedy_until gate, so the HW argmax of (logits + bias) returns the
            # penalized token directly. Before the gate the bias stays zero → pure greedy.
            if _fpga_penalty and len(generated) >= _greedy_until:
                self._write_penalty_bias(self._generated_tokens)

            self.start_execute_from_dram(preamble_addr)
            self.wait_queue(30.0)

            # HW argmax: the LM-head matmul leaves the (penalized) argmax index in a register, so read 4
            # bytes instead of DMA-ing back the full VOCAB_SIZE logits (~98 KB) + host argmax every token.
            token_id = self.get_arg_max_index()

            generated.append(token_id)
            self._generated_tokens.append(token_id)
            _SILENT_MODE = False
            if token_id in _SMOLVLM2_CFG["model"]["stop_token_ids"]:
                if _use_status:
                    _status_teardown()
                _original_print(f"\nStop token {token_id} reached.")
                break
            _original_print(self.tokenizer.decode([token_id]), end="", flush=True)
            if _use_status:
                _status_update()
        else:
            if _use_status:
                _status_teardown()
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
def _hf_reference(model_dir, tokenizer, prompt, image_path, n_greedy=8):
    """HF CPU fp32 reference for --debug. Returns (token_ids, hidden[L+1] x [S,H], logits[S,V],
    image_features[64,H] or None, greedy_tokens[n_greedy]). Image mode runs the full multimodal
    forward so per-layer hidden states match the device's image-mode prefill."""
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.float32,
        device_map=None, attn_implementation="eager").cpu().eval()
    image_features = None
    if image_path:
        from PIL import Image
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        img = Image.open(image_path).convert("RGB")
        # Use the DEVICE token layout (build_input_ids) for both, so sequences align 1:1.
        token_ids = build_input_ids(tokenizer, prompt, has_image=True)
        pv = proc.image_processor(images=[img], return_tensors="pt")
        kw = {"input_ids": torch.tensor([token_ids]),
              "pixel_values": pv["pixel_values"].to(torch.float32)}
        if "pixel_attention_mask" in pv:
            kw["pixel_attention_mask"] = pv["pixel_attention_mask"]
        with torch.no_grad():
            try:
                feat = model.get_image_features(pixel_values=kw["pixel_values"],
                                                pixel_attention_mask=kw.get("pixel_attention_mask"))
                image_features = torch.as_tensor(feat).reshape(-1, model.config.text_config.hidden_size).float()
            except Exception as e:
                print(f"[dbg] get_image_features unavailable ({e}); vision checked by NaN/stats only")
            out = model(output_hidden_states=True, use_cache=False, **kw)
            gen = model.generate(max_new_tokens=n_greedy, do_sample=False, **kw)
    else:
        token_ids = build_input_ids(tokenizer, prompt, has_image=False)
        ids = torch.tensor([token_ids])
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            gen = model.generate(input_ids=ids, max_new_tokens=n_greedy, do_sample=False)
    hidden = [h[0].float() for h in out.hidden_states]
    logits = out.logits[0].float()
    greedy = gen[0][len(token_ids):].tolist()
    del model
    return token_ids, hidden, logits, image_features, greedy


def run_debug(ue, args, script_dir, has_image):
    """End-to-end NaN/cosine localizer (one device session). Verifies, with a HF CPU reference:
    (1) vision encoder, (2) prefill_v2 every layer, (3) first-token logits / postamble, (4) the
    decode loop. Compile chatter is suppressed; a single clear DIAGNOSIS is printed at the end.
    Use --image=None for the clean text-only LM check (no vision confound)."""
    import io, contextlib
    quiet = lambda: contextlib.redirect_stdout(io.StringIO())  # hide compile/run chatter

    model_dir = os.path.join(script_dir, _SMOLVLM2_CFG["paths"]["hf_model_dir"])
    H, bpe = ue.HIDDEN_SIZE, 2
    PM = ue.PREFILL_MAX_SEQ_LEN
    F = {}  # findings

    print("\nLoading HF reference (CPU fp32)…", flush=True)
    with quiet():
        token_ids, hidden, logits, image_features, hf_greedy = _hf_reference(
            model_dir, ue.tokenizer, args.prompt, args.image if has_image else None)
    seq_len = len(token_ids)
    gold = int(logits[-1].argmax())
    print(f"=== DEBUG  tokens={seq_len}  image={has_image}  prompt={args.prompt!r} ===")
    print(f"HF golden first token id={gold} {ue.tokenizer.decode([gold])!r}", flush=True)

    # --- Stage 1: vision encoder ---
    if has_image:
        print("\n--- Stage 1: vision encoder (VIS_CONNECTOR vs HF image features) ---", flush=True)
        with quiet():
            enc_addr = ue.compile_encoder()
            ue.run_encoder(enc_addr, process_image(args.image))
        vis = ue.dbg_dram(ue.VIS_CONNECTOR_DRAM, 64 * H, "VIS_CONNECTOR",
                          ref=image_features if image_features is not None else None, raise_on_nan=False)
        F["vision_cos_excl8"] = vis["cos_excl8"]
        F["vision_nan"] = vis["nan"] + vis["inf"]

    # --- Stage 2: prefill_v2 per-layer hidden vs HF (one run) ---
    print("\n--- Stage 2: prefill_v2 per-layer hidden vs HF (one run) ---", flush=True)
    print("  compiling prefill_v2 (~496K instrs, silent, ~30-90s)…", flush=True)
    with quiet():
        ue.compile_prefill_v2(debug_dump=True)
    print("  running prefill_v2 on device…", flush=True)
    with quiet():
        ue.run_prefill_v2(token_ids, has_image=has_image, skip_postamble=True)
    layer_excl8, layer_nan = [], 0
    for n in range(ue.NUM_LAYERS):
        r = ue.dbg_dram(ue.DBG_LAYER_DUMP + n * PM * H * bpe, seq_len * H, f"L{n:02d}_out",
                        ref=hidden[n + 1][:seq_len], raise_on_nan=False)
        layer_excl8.append(r["cos_excl8"]); layer_nan += r["nan"] + r["inf"]
    # L31 vs HF is raw-vs-final-normed (artifact) -> judge layers on L0..L30 only.
    early = [c for c in layer_excl8[:3] if c is not None]
    midlate = [c for c in layer_excl8[3:ue.NUM_LAYERS - 1] if c is not None]
    F["prefill_early_min"] = min(early) if early else None
    F["prefill_midlate_min"] = min(midlate) if midlate else None
    F["prefill_nan"] = layer_nan

    # --- Stage 3: full prefill -> first-token logits (postamble: final-norm + LM head) ---
    print("\n--- Stage 3: full prefill first token (postamble) ---", flush=True)
    with quiet():
        tok = ue.run_prefill_v2(token_ids, has_image=has_image)
    lg = ue.dbg_dram(ue.LOGITS_DRAM, ue.VOCAB_SIZE, "LOGITS", ref=logits[-1], raise_on_nan=False)
    F["first_tok"] = tok; F["first_tok_match"] = (tok == gold); F["logits_cos"] = lg["cos"]
    # Split postamble: compare the device's NORMED final hidden (FINAL_NORM, M=1 last token) to HF's
    # final-normed last hidden (hidden[-1] is post-final-norm). High cos -> norm/L31 fine, LM head is
    # the bug; low cos -> final-norm or layer-31 output is the bug.
    hf_fn = hidden[-1][seq_len - 1]
    print(f"   (HF final-normed last hidden range min={float(hf_fn.min()):+.2f} max={float(hf_fn.max()):+.2f})")
    fn = ue.dbg_dram(ue.FINAL_NORM_DRAM + (seq_len - 1) * ue.HIDDEN_SIZE * 2, ue.HIDDEN_SIZE,
                     "FINAL_NORM(last)", ref=hf_fn, raise_on_nan=False)
    F["final_norm_cos"] = fn["cos"]

    # --- Stage 4: decode loop vs HF greedy continuation ---
    print("\n--- Stage 4: decode loop vs HF greedy ---", flush=True)
    n_dec = len(hf_greedy)
    try:
        bin_dir = os.path.join(script_dir, "smolvlm2_bin")
        have_dec = os.path.exists(os.path.join(bin_dir, "decoder_program.bin"))
        print(f"  {'loading' if have_dec else 'compiling'} decoder"
              f"{' (~49K embed programs, slow, minutes)' if not have_dec else ''}…", flush=True)
        with quiet():
            (ue.load_decoder if have_dec else ue.compile_decoder)()
            dev_cont = ue.run_decoder(tok, max_new_tokens=n_dec - 1)
        dev_seq = [tok] + list(dev_cont)
        m = 0
        for a, b in zip(dev_seq, hf_greedy):
            if a == b: m += 1
            else: break
        F["decode_match"] = m; F["decode_total"] = n_dec
        F["dev_text"] = ue.tokenizer.decode(dev_seq)
        F["hf_text"] = ue.tokenizer.decode(hf_greedy)
    except Exception as e:
        F["decode_err"] = repr(e)

    # ---------------- DIAGNOSIS ----------------
    def ok(c, th=0.90): return c is not None and c >= th
    print("\n" + "=" * 70)
    print("                        D I A G N O S I S")
    print("=" * 70)
    if has_image:
        print(f"  vision encoder : cos_excl8={F.get('vision_cos_excl8')}  nan/inf={F.get('vision_nan')}")
    print(f"  prefill layers : L0-2 min cos_excl8={F['prefill_early_min']:.3f}  "
          f"L3-30 min={F['prefill_midlate_min']:.3f}  nan/inf={F['prefill_nan']}")
    print(f"  first token    : device={F['first_tok']} {ue.tokenizer.decode([F['first_tok']])!r}  "
          f"golden={gold} {ue.tokenizer.decode([gold])!r}  "
          f"{'MATCH' if F['first_tok_match'] else 'MISMATCH'}  (logits cos={F['logits_cos']:.4f})")
    fnc = F.get("final_norm_cos")
    print(f"  final-norm     : normed last-hidden cos vs HF = {fnc if fnc is None else round(fnc,4)}")
    if "decode_err" in F:
        print(f"  decode loop    : ERROR {F['decode_err']}")
    else:
        print(f"  decode loop    : matched {F['decode_match']}/{F['decode_total']} greedy tokens")
        print(f"     device: {F['dev_text']!r}")
        print(f"     golden: {F['hf_text']!r}")
    print("-" * 70)

    # Per-stage PASS/FAIL — evaluate EVERY stage independently (don't stop at the first failure).
    def line(name, passed, note=""):
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:16s} {note}")
    stages = []
    if has_image:
        v_ok = (F.get("vision_nan", 0) == 0) and ok(F.get("vision_cos_excl8"), 0.85)
        stages.append(("vision encoder", v_ok,
                       f"cos_excl8={F.get('vision_cos_excl8')} nan={F.get('vision_nan')} "
                       "(preprocessing differs; low cos may be benign)"))
    p_ok = (F["prefill_nan"] == 0) and ok(F["prefill_early_min"])
    stages.append(("prefill layers", p_ok,
                   f"L0-2 min cos_excl8={F['prefill_early_min']:.3f} nan={F['prefill_nan']}"))
    fn_ok = ok(fnc, 0.95)
    stages.append(("final-norm/L31", fn_ok, f"normed-hidden cos={fnc if fnc is None else round(fnc,4)}"))
    head_ok = F["first_tok_match"]
    stages.append(("LM head / token", head_ok,
                   f"device={F['first_tok']!r} golden={gold!r} logits_cos={F['logits_cos']:.3f}"))
    dec_ok = ("decode_err" not in F) and (F.get("decode_match", 0) == F.get("decode_total", -1))
    stages.append(("decoder loop", dec_ok,
                   F.get("decode_err", f"matched {F.get('decode_match')}/{F.get('decode_total')}")))
    for name, passed, note in stages:
        line(name, passed, note)
    print("-" * 70)
    failed = [n for n, p, _ in stages if not p]
    if not failed:
        print("  >>> ALL STAGES PASS — output matches HF.")
    else:
        print(f"  >>> FIRST FAILING STAGE: {failed[0].upper()}   (all failures: {', '.join(failed)})")
    print("=" * 70, flush=True)


# =============================================================================
# CPU decode verification (--verify_prefill)
# =============================================================================
def _dump_prefill_dram_state(ue, token_ids, hw_first_token):
    """Read-only inspection of the DRAM sections that get handed to the decoder.

    No device program runs (pure dma_from_accelerator_memory), so this cannot hang.
    Shows (1) the prefill final-position logits that produced the seed token, and
    (2) how the KV cache is populated across positions for a few sample layers/heads —
    real data must occupy [0:seq_len) and be zero at [seq_len:]."""
    import torch
    seq_len = len(token_ids)
    bpe = 2
    HD = ue.HEAD_DIM
    _original_print("\n=== HW DRAM state at end of prefill ===")
    _original_print(f"  seq_len={seq_len}  PREFILL_MAX={ue.PREFILL_MAX_SEQ_LEN}  "
                    f"NUM_LAYERS={ue.NUM_LAYERS}  NUM_KV_HEADS={ue.NUM_KV_HEADS}  HEAD_DIM={HD}")

    # (1) Prefill final logits -> seed token fed to decode
    logits = ue.dma_from_accelerator_memory(ue.LOGITS_DRAM, torch.Size([ue.VOCAB_SIZE])).float()
    top5 = logits.topk(5)
    _original_print(f"  prefill LOGITS_DRAM argmax={int(logits.argmax())} "
                    f"({ue.tokenizer.decode([int(logits.argmax())])})  "
                    f"top5_ids={top5.indices.tolist()} top5_vals={[round(v,3) for v in top5.values.tolist()]}")
    _original_print(f"  seed token handed to decode (hw_first_token)={hw_first_token} "
                    f"({ue.tokenizer.decode([hw_first_token])})  "
                    f"-> {'MATCH' if int(logits.argmax())==hw_first_token else 'MISMATCH'} with prefill argmax")

    # (2) KV cache population per position (head 0) for sample layers
    sample_layers = sorted(set([0, ue.NUM_LAYERS // 2, ue.NUM_LAYERS - 1]))
    sample_pos = sorted(set([0, 1, seq_len - 2, seq_len - 1, seq_len, seq_len + 1]))
    sample_pos = [p for p in sample_pos if 0 <= p < ue.PREFILL_MAX_SEQ_LEN]
    _original_print("  KV-cache L2 norms (head 0)  [pos<seq_len should be nonzero, pos>=seq_len should be ~0]:")
    for layer in sample_layers:
        kbase = ue.LAYER0_K_DRAM + layer * ue.KV_LAYER_STRIDE  # head 0
        vbase = ue.LAYER0_V_DRAM + layer * ue.KV_LAYER_STRIDE
        kparts, vparts = [], []
        for p in sample_pos:
            k = ue.dma_from_accelerator_memory(kbase + p * HD * bpe, torch.Size([HD])).float()
            v = ue.dma_from_accelerator_memory(vbase + p * HD * bpe, torch.Size([HD])).float()
            kparts.append(f"p{p}={k.norm():.2f}")
            vparts.append(f"p{p}={v.norm():.2f}")
        _original_print(f"    L{layer:2d} K: " + "  ".join(kparts))
        _original_print(f"    L{layer:2d} V: " + "  ".join(vparts))
    _original_print("=== end HW DRAM state ===\n")


def _verify_prefill_cpu(ue, token_ids, hw_first_token, model_dir, image_path, prompt, max_new=30):
    """Use AutoProcessor + model.generate() — the standard HF path — to get ground-truth output."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image

    _original_print("  Loading HF model + processor on CPU...")
    hf = AutoModelForImageTextToText.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map="cpu", attn_implementation="eager").eval()
    proc = AutoProcessor.from_pretrained(model_dir, local_files_only=True)

    has_image = image_path is not None
    if has_image:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        img = Image.open(image_path).convert("RGB")
        prompt_text = proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs = proc(text=prompt_text, images=[img], return_tensors="pt")
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        prompt_text = proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs = proc(text=prompt_text, return_tensors="pt")
    inputs = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in inputs.items()}

    _original_print(f"  hw first token: {hw_first_token} ({ue.tokenizer.decode([hw_first_token])})")
    _original_print("  CPU generate output: ", end="", flush=True)
    with torch.no_grad():
        out_ids = hf.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    new_ids = out_ids[0, inputs["input_ids"].shape[1]:]
    _original_print(proc.decode(new_ids, skip_special_tokens=True))
    del hf


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
    from user_dma_core import set_dma_device

    parser = argparse.ArgumentParser(description="SmolVLM2-500M on accelerator (bf16 vision + q4 LM)")
    _d = _SMOLVLM2_CFG["defaults"]
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _default_image = os.path.join(_root, _d["image"])
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt. Default: a text question (LM-only) or 'Describe this image.' (VLM).")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to an image. Providing it enables VLM. Default: none (LM-only text).")
    parser.add_argument("--vision-enable", action="store_true",
                        help="Enable the vision encoder (VLM) using --image, or the default sample image if none is given.")
    parser.add_argument("--dev", type=str, default=_d["dev"], help="DMA device name")
    parser.add_argument("--cycle", type=float, default=None, help='Clock cycle time in ns. Overrides --device default.')
    parser.add_argument("--device", type=str, default='kintex7', help='FPGA board profile (kintex7, rk, puzhi, bittware, bittware_256, alveo).')
    parser.add_argument("--max-seq", type=int, default=_d["max_seq"], help="Max sequence length")
    parser.add_argument("--debug", action="store_true", help="Staged NaN/cosine localizer: verify vision + per-layer prefill_v2 vs HF CPU reference, then exit (no decode)")
    parser.add_argument("--verify_prefill", action="store_true", help="After hardware prefill, run CPU decode using HF model to confirm model output (skips hardware decoder)")
    parser.add_argument("--clear_scratch", action="store_true", help="Zero all scratch DRAM between decode steps (diagnostic for step-2 hang)")
    pen = parser.add_argument_group("on-FPGA repetition penalty (ENABLED by default; breaks decode loops)")
    pen.add_argument("--pure-greedy", action="store_true",
                     help="Disable the on-FPGA repetition penalty — plain greedy decode.")
    pen.add_argument("--pen-alpha", type=float, default=1.0, help="Penalty strength α (bias = −α·count). Default 1.0.")
    pen.add_argument("--pen-cap", type=float, default=20.0, help="Max penalty magnitude (clamp). Default 20.0.")
    pen.add_argument("--rep-window", type=int, default=256, help="Token window counted for the penalty. Default 256.")
    pen.add_argument("--greedy-until", type=int, default=None,
                     help="Pure greedy for the first N decoded tokens, then penalty turns on. "
                          "Default: 0 (LM-only, loops early) or 64 (VLM, let the caption form first).")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, _SMOLVLM2_CFG["paths"]["hf_model_dir"])
    # Download HF model if needed (the only weight source — weight_init quantizes/loads from it,
    # producing the single weights.bin snapshot; there are no intermediate weight bins).
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        print(f"Downloading {HF_MODEL_REPO} ...")
        snapshot_download(repo_id=HF_MODEL_REPO, local_dir=model_dir, local_dir_use_symlinks=False, ignore_patterns=["onnx/*"])

    # --- Hardware inference ---
    set_dma_device(args.dev)
    axi_width_bits = 512 if args.device in ("bittware", "rk") else 256
    os.environ["UE_AXI_DATA_WIDTH_BITS"] = str(axi_width_bits)
    user_dma_core.UE_AXI_DATA_WIDTH_BITS = axi_width_bits
    clock = args.cycle if args.cycle is not None else _clock_ns_default_for_device(args.device)
    user_dma_core.CLOCK_CYCLE_TIME_NS = clock
    user_dma_core.UE_PEAK_GFLOPS = 0.128 / clock
    print(f"FPGA profile: device={args.device}, clock={clock:.4f} ns, UE_AXI_DATA_WIDTH_BITS={axi_width_bits}")
    global _SILENT_MODE
    _SILENT_MODE = True
    ue = SmolVLM2_UnifiedEngine(script_dir=script_dir)
    # VLM (vision) is opt-in at RUNTIME: enabled by --vision-enable or by passing --image. The instruction
    # bin is ALWAYS the full VLM bin (encoder+decoder+prefill) — built once, robust + easy to maintain;
    # this flag only decides whether the vision encoder actually RUNS this invocation. Default = LM-only.
    vision_on = bool(args.vision_enable) or (args.image is not None
                                             and str(args.image).strip().lower() not in ("none", ""))
    if vision_on and (args.image is None or str(args.image).strip().lower() in ("none", "")):
        args.image = _default_image   # --vision-enable with no --image → use the default sample image
    if args.prompt is None:           # mode-appropriate default prompt (a text Q for LM-only)
        args.prompt = _d["vlm_prompt"] if vision_on else _d["lm_prompt"]
    # No startup DRAM zeroing: each run already zero_dram()s at the END (below), so the next run starts
    # clean; weights/instructions are DMA'd over their regions. (Avoids the slow 2 GB zero every launch.)
    init_hang_prevention(ue)
    # Load the snapshot (weights.bin) ONLY when the instruction bin also exists — i.e. a pure load run.
    # If instructions.bin is missing we must compile_all, and compile_encoder needs the full vision
    # weight addresses that weight_init sets but the snapshot does not restore. So: load-both, or
    # build-both — never load-snapshot-then-compile.
    _bin_dir = os.path.join(script_dir, "smolvlm2_bin")
    _have_instr_bin = (os.path.exists(os.path.join(_bin_dir, "instructions.bin"))
                       and os.path.exists(os.path.join(_bin_dir, "instructions.json")))
    if not (_have_instr_bin and ue.load_snapshot()):
        ue.weight_init()
        ue.tensor_init(max_seq_len=args.max_seq)
        ue.dump_snapshot()
    has_image = vision_on
    token_ids = build_input_ids(ue.tokenizer, args.prompt, has_image=has_image)
    seq_len = len(token_ids)
    _SILENT_MODE = False
    if has_image:
        image_path = os.path.abspath(args.image)
        _original_print(f"Image: {image_path}")
    _original_print(f"Prompt: {args.prompt!r} ({seq_len} tokens, image={'yes' if has_image else 'no'})")
    if args.debug:
        run_debug(ue, args, script_dir, has_image)
        return
    # --- Compile (or load from bin) ---
    _SILENT_MODE = True
    timer = time.perf_counter()
    bin_dir = os.path.join(script_dir, "smolvlm2_bin")
    S = ((seq_len + 63) // 64) * 64
    # ONE unified instruction bin (encoder + decoder + prefill_v2). compile_all() builds it on the
    # first run and loads it (cache key = layout signature) on subsequent runs.
    use_bin = (os.path.exists(os.path.join(bin_dir, "instructions.bin"))
               and os.path.exists(os.path.join(bin_dir, "instructions.json")))
    import threading, io, contextlib
    _real_out = sys.stdout  # spinner writes here even while stdout is redirected
    stop_compile = threading.Event()
    def _compile_progress():
        while not stop_compile.wait(1.0):
            elapsed = time.perf_counter() - timer
            _original_print(f"\r  Compiling ({elapsed:.0f}s)", end="", flush=True, file=_real_out)
    if not use_bin:
        t = threading.Thread(target=_compile_progress, daemon=True)
        t.start()
    # Hard-silence the core's compile/capture chatter (M_chunk/URAM/Capture stopped/…),
    # which leaks past _SILENT_MODE. The live spinner above writes to _real_out, so it survives.
    with contextlib.redirect_stdout(io.StringIO()):
        uni_meta = ue.compile_all()
        enc_addr = uni_meta["encoder_addr"]
    stop_compile.set()
    _SILENT_MODE = False
    elapsed = time.perf_counter() - timer
    if use_bin:
        _original_print(f"  Loaded from bin in {elapsed:.2f}s")
    else:
        _original_print(f"\r  Compiling ({elapsed:.0f}s) done")
    # --- Vision encoder ---
    if has_image:
        timer = time.perf_counter()
        pixel_values = process_image(args.image)
        stop_enc = threading.Event()
        def _enc_progress():
            while not stop_enc.wait(1.0):
                _original_print(f"\r  Vision encoder ({time.perf_counter() - timer:.0f}s)", end="", flush=True)
        t_enc = threading.Thread(target=_enc_progress, daemon=True)
        t_enc.start()
        _SILENT_MODE = True
        ue.run_encoder(enc_addr, pixel_values)
        _SILENT_MODE = False
        stop_enc.set()
        t_enc.join()
        enc_time = time.perf_counter() - timer
        _original_print(f"\r  Vision encoder ({enc_time:.0f}s) done")
    # --- Prefill ---
    timer = time.perf_counter()
    padded_seq = ((seq_len + 63) // 64) * 64
    S = padded_seq
    H, D, I = ue.HIDDEN_SIZE, ue.HEAD_DIM, ue.INTERMEDIATE_SIZE
    KVH, QH, G = ue.NUM_KV_HEADS, ue.NUM_HEADS, ue.GROUP_SIZE
    KV = KVH * D
    attn_per_group = (D * S + G * (S * D + 2 * S * D * S + S * S + S * S * 5 + 2 * S * S * D))
    per_layer = (4 * S * H + 2 * S * H * H + 2 * 2 * S * H * KV
        + (QH + KVH) * S * D * 4 + KVH * attn_per_group
        + 2 * S * H * H + 5 * S * H + 2 * (2 * S * H * I) + S * I + 2 * S * I * H + S * H)
    prefill_flops = ue.NUM_LAYERS * per_layer + 4 * S * H + 2 * 1 * H * ue.VOCAB_SIZE
    stop_pf = threading.Event()
    def _pf_progress():
        while not stop_pf.wait(1.0):
            _original_print(f"\r  Prefill ({time.perf_counter() - timer:.0f}s)", end="", flush=True)
    t_pf = threading.Thread(target=_pf_progress, daemon=True)
    t_pf.start()
    _SILENT_MODE = True
    hw_token = ue.run_prefill_v2(token_ids, has_image=has_image, total_flops=prefill_flops)
    _SILENT_MODE = False
    stop_pf.set()
    t_pf.join()
    prefill_time = time.perf_counter() - timer
    _original_print(f"\r  Prefill ({prefill_time:.0f}s) done")
    # Zero out stale KV cache positions left by the static PREFILL_MAX prefill v2 (it writes cache
    # rows [seq_len:PREFILL_MAX] from epsilon-padded input — decode must not read those as real).
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
    if args.verify_prefill:
        _dump_prefill_dram_state(ue, token_ids, hw_token)
        _original_print(f"\n--- verify_prefill: CPU ground-truth decode ---")
        _verify_prefill_cpu(ue, token_ids, hw_token, model_dir,
                            image_path=args.image if has_image else None,
                            prompt=args.prompt, max_new=30)
        _original_print("verify_prefill done — exiting (no hardware decode).")
        return

    # --- Decode (on-device embed fused with decoder, single dispatch per token) ---
    # On-FPGA repetition penalty config (runtime-only; the bias is always wired into the decode LM-head).
    # greedy_until is mode-dependent unless the user overrode it: VLM captions are bounded + good early,
    # so postpone the penalty; LM-only loops early, so penalize from the first token.
    _greedy_until = args.greedy_until if args.greedy_until is not None \
        else (_d["vlm_greedy_until"] if vision_on else _d["lm_greedy_until"])
    ue.fpga_penalty = not args.pure_greedy
    ue.pen_alpha, ue.pen_cap, ue.rep_window, ue.greedy_until = args.pen_alpha, args.pen_cap, args.rep_window, _greedy_until
    max_new = args.max_seq - seq_len
    _original_print(f"\nPrompt:   {args.prompt}")
    _original_print(f"Response: ", end="", flush=True)   # the generated answer streams right after this
    decode_timer = time.perf_counter()
    hw_tokens = ue.run_decoder(hw_token, max_new_tokens=max_new, clear_scratch=args.clear_scratch)
    decode_time = time.perf_counter() - decode_timer
    total_time = prefill_time + decode_time
    n_generated = len(hw_tokens)
    _original_print(f"\nDecoder done in {total_time:.2f} seconds, total {n_generated} tokens.")
    _original_print(
        f"SmolVLM2 test ends. prefill {round(seq_len / prefill_time, 2) if prefill_time > 0 else 0.0} tok/s, "
        f"decode {round(n_generated / decode_time, 2) if decode_time > 0 else 0.0} tok/s.")

    # Reset device DRAM + soft-reset at end of execution so leftover program/KV/scratch
    # state doesn't contaminate the next model run. Use zero_dram() (not clear_dram(),
    # which fills 0xFF = NaN in bf16 and would poison a read-before-write region).
    ue.zero_dram()
    _SILENT_MODE = True
    ue.software_reset()

if __name__ == "__main__":
    main()
