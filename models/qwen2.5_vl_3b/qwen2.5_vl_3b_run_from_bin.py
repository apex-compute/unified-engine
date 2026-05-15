#!/usr/bin/env python3
"""Qwen2.5-VL-3B inference from pre-compiled bins. Run qwen2.5_vl_3b_test.py first to compile."""
import builtins
import json
import mmap
import os
import sys
import time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, TYPE, UE_FMAX_CONTEXT_SIZE, UE_VECTOR_SIZE, UE_ARGMAX1_INDEX,
    URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS, set_dma_device,
    UnifiedEngine, DRAM_INSTRUCTION_ADDR,
    ue_35bit_addr_shifter,
)

_original_print = builtins.print
_SILENT_MODE = False

def _set_silent(val: bool) -> None:
    global _SILENT_MODE
    _SILENT_MODE = val

def quiet_print(*args, **kwargs):
    if not _SILENT_MODE:
        _original_print(*args, **kwargs)

builtins.print = quiet_print

def _parse_offset(val) -> int:
    if isinstance(val, str):
        return int(val, 0)
    return int(val)

_VALID_PRECISIONS = ('int4', 'fp4', 'if4')

# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------

def store_weight(ue, tensor, padded_shape=None):
    """Pad, convert to bf16, DMA to device DRAM. Returns DRAM address."""
    bf16 = tensor.to(torch.bfloat16)
    if padded_shape is not None:
        padded = torch.zeros(padded_shape, dtype=torch.bfloat16)
        slices = tuple(slice(0, s) for s in bf16.shape)
        padded[slices] = bf16
        bf16 = padded
    bf16 = bf16.contiguous()
    nbytes = bf16.numel() * 2
    addr = ue.get_params_dram_addr()
    ue.dma_write(DMA_DEVICE_H2C, addr, bf16.flatten(), nbytes)
    ue.allocate_params_dram(nbytes)
    return addr

def store_quantized_weight(ue, raw_data):
    """Store Q4_64 raw data as scale + packed in DRAM. Returns (scale_addr, data_addr)."""
    raw_bytes = raw_data.tobytes() if hasattr(raw_data, 'tobytes') else bytes(raw_data)
    n_blocks = len(raw_bytes) // 34
    scales_size = n_blocks * 2
    data_size = n_blocks * 32
    scales_np = np.frombuffer(raw_bytes[:scales_size], dtype=np.uint16).copy()
    scale_tensor = torch.from_numpy(scales_np).view(torch.bfloat16)
    scale_addr = ue.get_params_dram_addr()
    ue.dma_write(DMA_DEVICE_H2C, scale_addr, scale_tensor, scales_size)
    ue.allocate_params_dram(scales_size)
    data_np = np.frombuffer(raw_bytes[scales_size:scales_size + data_size], dtype=np.uint8).copy()
    data_tensor = torch.from_numpy(data_np)
    data_addr = ue.get_params_dram_addr()
    ue.dma_write(DMA_DEVICE_H2C, data_addr, data_tensor, data_size)
    ue.allocate_params_dram(data_size)
    return scale_addr, data_addr

def load_weight_cache(bin_path):
    """Load bin+json weight file via mmap. Returns {tensor_name: raw_numpy_data}."""
    json_path = bin_path.rsplit('.', 1)[0] + '.json'
    with open(json_path) as f:
        manifest = json.load(f)
    cache = {}
    with open(bin_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        for name, meta in manifest.items():
            cache[name] = np.frombuffer(mm[meta['offset']:meta['offset'] + meta['size']], dtype=np.uint8).copy()
        mm.close()
    return cache

def store_identity_matrix(ue):
    bpe = 2
    size = UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe
    addr = ue.get_params_dram_addr()
    ue.dma_write(DMA_DEVICE_H2C, addr, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16), size)
    ue.allocate_params_dram(size)
    return addr

def _lm_precision(cfg: dict) -> str:
    p = cfg.get("precision", {}).get("lm", "if4")
    if p not in _VALID_PRECISIONS:
        raise ValueError(f"config precision.lm={p!r} not in {_VALID_PRECISIONS}")
    return p

def _vision_precision(cfg: dict) -> str:
    p = cfg.get("precision", {}).get("vision", "int4")
    if p not in _VALID_PRECISIONS:
        raise ValueError(f"config precision.vision={p!r} not in {_VALID_PRECISIONS}")
    return p

def _load_config(script_dir: str) -> dict:
    config_path = os.path.join(script_dir, "qwen2.5_vl_3b_config.json")
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

def process_image(image_path: str, size: int = 448) -> torch.Tensor:
    from PIL import Image
    img_cfg = _load_config(SCRIPT_DIR).get("image_processing", {})
    resize = img_cfg.get("resize", size)
    mean = img_cfg.get("normalize_mean", [0.48145466, 0.4578275, 0.40821073])
    std = img_cfg.get("normalize_std", [0.26862954, 0.26130258, 0.27577711])
    img = Image.open(image_path).convert("RGB").resize((resize, resize), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    for c in range(3):
        arr[:, :, c] = (arr[:, :, c] - mean[c]) / std[c]
    return torch.from_numpy(arr).permute(2, 0, 1).to(torch.bfloat16)

def check_bins(cfg, script_dir, has_image: bool, prefill_seq_len: int) -> list[str]:
    """Return list of missing bin files that are required."""
    paths = cfg["paths"]
    bin_dir = os.path.join(script_dir, "qwen2.5_vl_3b_bin")
    missing = []
    for key in ("lm_weights", "vision_weights"):
        p = os.path.join(script_dir, paths[key])
        if not os.path.exists(p):
            missing.append(paths[key])
        jp = p.rsplit('.', 1)[0] + '.json'
        if not os.path.exists(jp):
            missing.append(jp)
    if has_image:
        enc_bin = os.path.join(bin_dir, "encoder_program.bin")
        if not os.path.exists(enc_bin):
            missing.append("qwen2.5_vl_3b_bin/encoder_program.bin")
    pf_bin = os.path.join(bin_dir, f"prefill_program_s{prefill_seq_len}.bin")
    pf_meta = os.path.join(bin_dir, f"prefill_program_s{prefill_seq_len}.json")
    if not os.path.exists(pf_bin):
        missing.append(f"qwen2.5_vl_3b_bin/prefill_program_s{prefill_seq_len}.bin")
    if not os.path.exists(pf_meta):
        missing.append(f"qwen2.5_vl_3b_bin/prefill_program_s{prefill_seq_len}.json")
    dec_bin = os.path.join(script_dir, paths.get("decoder_program_bin", "qwen2.5_vl_3b_bin/decoder_program.bin"))
    dec_meta = os.path.join(script_dir, paths.get("decoder_program_meta", "qwen2.5_vl_3b_bin/decoder_program.json"))
    if not os.path.exists(dec_bin):
        missing.append(paths.get("decoder_program_bin", "qwen2.5_vl_3b_bin/decoder_program.bin"))
    if not os.path.exists(dec_meta):
        missing.append(paths.get("decoder_program_meta", "qwen2.5_vl_3b_bin/decoder_program.json"))
    return missing

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Qwen25VL3B_UnifiedEngine(UnifiedEngine):

    def __init__(self, script_dir=None):
        self._identity_dram_written = False
        self._identity_dram_addr = None
        self._IDENTITY_MAT_BYTES = UE_VECTOR_SIZE * UE_VECTOR_SIZE * 2
        super().__init__(
            params_dram_base=0x00000000,
            tensor_dram_base=0x90000000,
            program_dram_base=0xA0000000,
        )
        self.script_dir = script_dir or SCRIPT_DIR
        self._cfg = _load_config(self.script_dir)
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
        self._has_q_k_norm = False
        self._has_post_attn_norm = False
        self._has_post_mlp_norm = False
        self._has_qkv_bias = True
        self._vis_cfg = self._cfg.get("vision", {})

        self.weight_init()
        self.tensor_init()

    _IDENTITY_MAT_BYTES = UE_VECTOR_SIZE * UE_VECTOR_SIZE * 2

    def _preallocate_identity_matrix(self) -> None:
        if self._identity_dram_addr is not None:
            return
        self._identity_dram_addr = super().allocate_params_dram(self._IDENTITY_MAT_BYTES)
        eye = torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16)
        super().dma_write(DMA_DEVICE_H2C, self._identity_dram_addr, eye, self._IDENTITY_MAT_BYTES)
        self._identity_dram_written = True

    def dma_write(self, device, addr, data, size):
        if (self._identity_dram_written
                and self._identity_dram_addr is not None
                and addr == self._identity_dram_addr
                and size == self._IDENTITY_MAT_BYTES):
            return
        super().dma_write(device, addr, data, size)

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
        print(f"    Loaded {total_size} bytes from {os.path.basename(bin_path)} to DRAM at 0x{start_addr:x}")
        return start_addr, total_size

    def allocate_params_dram(self, size_bytes: int) -> int:
        params_dram_addr = self._next_params_dram_addr
        self._next_params_dram_addr += size_bytes
        return params_dram_addr

    def clear_inst_id(self) -> None:
        self._inst_id = 0

    def get_arg_max_index(self) -> int:
        return self.read_reg32(UE_ARGMAX1_INDEX)

    def get_embedding_for_tokens(self, token_ids) -> torch.Tensor:
        tid_t = torch.tensor(token_ids, dtype=torch.long)
        out = torch.zeros(len(token_ids), self.vector_length, dtype=torch.bfloat16)
        valid = tid_t < self.embedding_weight.shape[0]
        out[valid] = self.embedding_weight[tid_t[valid]]
        return out

    def _load_rope_host(self, rope_theta=None) -> None:
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
        rope_sz = self.weight_defs["ROPE_SIZE"]
        rope_raw_padded = (rope_raw + b"\x00" * rope_sz)[:rope_sz]
        rope_base = self.allocate_params_dram(rope_sz)
        self.dma_write(DMA_DEVICE_H2C, rope_base, rope_raw_padded, rope_sz)
        self.DRAM_ADDR_ROPE = rope_base
        self._rope_inv_freq = inv_freq
        self._rope_theta = theta
        self._rope_table_bytes = rope_raw_padded

    def load_mrope_for_prefill(self, prefill_seq, image_grid_thw=None) -> None:
        if image_grid_thw is None:
            return
        mrope_sections = self._cfg["special"]["rope"].get("mrope_section", [16, 24, 24])
        spatial_merge_size = self._vis_cfg["spatial_merge_size"]
        image_pad_id = 151655
        vision_start_id = 151652
        seq_len = len(prefill_seq)
        tokens = list(prefill_seq)
        pos_t = torch.zeros(seq_len, dtype=torch.long)
        pos_h = torch.zeros(seq_len, dtype=torch.long)
        pos_w = torch.zeros(seq_len, dtype=torch.long)
        img_token_positions = [i for i, t in enumerate(tokens) if t == image_pad_id]
        if len(img_token_positions) > 0 and image_grid_thw is not None:
            img_first = img_token_positions[0]
            st_idx = 0
            for i in range(img_first):
                pos_t[i] = st_idx + i; pos_h[i] = st_idx + i; pos_w[i] = st_idx + i
            t_grid, h_grid, w_grid = image_grid_thw[0].tolist()
            llm_h = h_grid // spatial_merge_size
            llm_w = w_grid // spatial_merge_size
            text_start = img_first + st_idx
            for idx, tok_pos in enumerate(img_token_positions):
                t_i = idx // (llm_h * llm_w)
                hw_i = idx % (llm_h * llm_w)
                h_i = hw_i // llm_w; w_i = hw_i % llm_w
                pos_t[tok_pos] = text_start
                pos_h[tok_pos] = text_start + h_i
                pos_w[tok_pos] = text_start + w_i
            max_img_pos = max(pos_t[img_token_positions].max(), pos_h[img_token_positions].max(), pos_w[img_token_positions].max()) + 1
            img_last = img_token_positions[-1]
            for i in range(img_last + 1, seq_len):
                offset = i - (img_last + 1)
                pos_t[i] = max_img_pos + offset; pos_h[i] = max_img_pos + offset; pos_w[i] = max_img_pos + offset
        else:
            for i in range(seq_len):
                pos_t[i] = i; pos_h[i] = i; pos_w[i] = i
        D = self.actual_head_dim // 2
        inv_freq = self._rope_inv_freq
        s0, s1, s2 = mrope_sections
        inv_sec0 = inv_freq[:s0]; inv_sec1 = inv_freq[s0:s0+s1]; inv_sec2 = inv_freq[s0+s1:]
        freqs = torch.zeros(seq_len, D, dtype=torch.float32)
        freqs[:, :s0]       = pos_t.float().unsqueeze(1) * inv_sec0.unsqueeze(0)
        freqs[:, s0:s0+s1]  = pos_h.float().unsqueeze(1) * inv_sec1.unsqueeze(0)
        freqs[:, s0+s1:]    = pos_w.float().unsqueeze(1) * inv_sec2.unsqueeze(0)
        cos_ = freqs.cos().to(torch.bfloat16); sin_ = freqs.sin().to(torch.bfloat16)
        rope_tensor = torch.cat([cos_, cos_, -sin_, sin_], dim=1)
        rope_bytes = rope_tensor.contiguous().view(torch.uint8).numpy().tobytes()
        self.dma_write(DMA_DEVICE_H2C, self.DRAM_ADDR_ROPE, rope_bytes, len(rope_bytes))
        max_pos = max(pos_t.max().item(), pos_h.max().item(), pos_w.max().item())
        self._mrope_delta = int(max_pos + 1 - seq_len)
        _original_print(f"    mRoPE table [{seq_len}, 256] written (rope_delta={self._mrope_delta})")

    def restore_rope_for_decoder(self) -> None:
        if hasattr(self, '_rope_table_bytes'):
            self.dma_write(DMA_DEVICE_H2C, self.DRAM_ADDR_ROPE,
                           self._rope_table_bytes, len(self._rope_table_bytes))

    def weight_init(self) -> None:
        """Load LM + vision weights from bin files to DRAM. Uses mmap — no full-bin RAM load."""
        from transformers import AutoTokenizer
        from huggingface_hub import snapshot_download
        model_dir = os.path.join(self.script_dir, self._cfg["paths"]["hf_model_dir"])
        hf_repo = self._cfg["paths"]["hf_model_repo"]
        if not os.path.exists(os.path.join(model_dir, "config.json")):
            _original_print(f"Downloading HF tokenizer files from {hf_repo} ...")
            snapshot_download(repo_id=hf_repo, local_dir=model_dir, local_dir_use_symlinks=False)

        lm_bin_path = os.path.join(self.script_dir, self._cfg["paths"]["lm_weights"])
        vis_bin_path = os.path.join(self.script_dir, self._cfg["paths"]["vision_weights"])
        if not os.path.exists(lm_bin_path) or not os.path.exists(vis_bin_path):
            raise FileNotFoundError(
                f"Weight bins missing. Run qwen2.5_vl_3b_test.py first to generate them.\n"
                f"  LM:     {lm_bin_path}\n  Vision: {vis_bin_path}"
            )

        print(f"  Reading LM weight bin...", flush=True)
        lm_cache = load_weight_cache(lm_bin_path)

        embed_raw = lm_cache['language_model.embed_tokens.weight']
        embed_bf16 = torch.from_numpy(embed_raw.copy()).view(torch.bfloat16).reshape(self.EMBEDDING_ELEMENTS, self.vector_length)
        self.embedding_weight = embed_bf16.clone()

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        lm_prec = _lm_precision(self._cfg)
        print(f"  Loading {self.LAYER_SIZE} LM layers to FPGA DRAM ({lm_prec})...", flush=True)
        self.lm_layer_addrs = []
        for i in range(self.LAYER_SIZE):
            if i and (i % 8 == 0 or i == self.LAYER_SIZE - 1):
                print(f"    layer {i + 1}/{self.LAYER_SIZE} loaded", flush=True)
            la = {}
            prefix = f'language_model.layers.{i}'
            for proj, hf_sub in [('q', 'self_attn.q_proj'), ('k', 'self_attn.k_proj'),
                                  ('gate', 'mlp.gate_proj'), ('up', 'mlp.up_proj'),
                                  ('down', 'mlp.down_proj')]:
                la[f'{proj}_scale'], la[f'{proj}_data'] = store_quantized_weight(self, lm_cache[f'{prefix}.{hf_sub}.weight.{lm_prec}'])
            la['v_weight'] = store_weight(self, torch.from_numpy(lm_cache[f'{prefix}.self_attn.v_proj.weight'].copy()).view(torch.bfloat16))
            la['o_weight'] = store_weight(self, torch.from_numpy(lm_cache[f'{prefix}.self_attn.o_proj.weight'].copy()).view(torch.bfloat16))
            for proj, hf_sub in [('q', 'self_attn.q_proj'), ('k', 'self_attn.k_proj'), ('v', 'self_attn.v_proj')]:
                bias_key = f'{prefix}.{hf_sub}.bias'
                if bias_key in lm_cache:
                    la[f'{proj}_bias'] = store_weight(self, torch.from_numpy(lm_cache[bias_key].copy()).view(torch.bfloat16))
                else:
                    bias_size = self.vector_length if proj == 'q' else self.head_dim
                    la[f'{proj}_bias'] = store_weight(self, torch.zeros(bias_size, dtype=torch.bfloat16))
            la['ln1_gamma'] = store_weight(self, torch.from_numpy(lm_cache[f'{prefix}.input_layernorm.weight'].copy()).view(torch.bfloat16))
            la['ln2_gamma'] = store_weight(self, torch.from_numpy(lm_cache[f'{prefix}.post_attention_layernorm.weight'].copy()).view(torch.bfloat16))
            self.lm_layer_addrs.append(la)

        self.final_norm_addr = store_weight(self, torch.from_numpy(lm_cache['language_model.norm.weight'].copy()).view(torch.bfloat16))
        lm_head_key = f'lm_head.weight.{lm_prec}'
        if lm_head_key not in lm_cache:
            raise RuntimeError(f"LM head entry {lm_head_key!r} missing from bin. Delete the bin and regenerate.")
        self.lm_head_scale, self.lm_head_data = store_quantized_weight(self, lm_cache[lm_head_key])
        self.identity_addr = store_identity_matrix(self)
        del lm_cache

        self._load_rope_host()
        self._load_vision_weights(vis_bin_path)
        print(f"    Weights end at DRAM 0x{self.get_params_dram_addr():X}, usage: {self.get_params_dram_usage()} bytes")
        print("Tokenizer loaded successfully.")

    def _load_vision_weights(self, vis_bin_path: str) -> None:
        vis_cache = load_weight_cache(vis_bin_path)
        vis_cfg = self._vis_cfg
        vis_depth = vis_cfg["depth"]
        VI = vis_cfg["intermediate_size"]
        VIS_MLP_PAD = ((VI + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
        vis_prec = _vision_precision(self._cfg)
        print(f"  Loading {vis_depth} vision layers to FPGA DRAM ({vis_prec})...", flush=True)
        self.vis_layer_addrs = []
        for i in range(vis_depth):
            if i and (i % 8 == 0 or i == vis_depth - 1):
                print(f"    layer {i + 1}/{vis_depth} loaded", flush=True)
            la = {}
            prefix = f'visual.blocks.{i}'
            la['qk_scale'], la['qk_data'] = store_quantized_weight(self, vis_cache[f'{prefix}.attn.qk_padded.weight.{vis_prec}'])
            la['qk_bias'] = store_weight(self, torch.from_numpy(vis_cache[f'{prefix}.attn.qk_padded.bias'].copy()).view(torch.bfloat16))
            la['v_scale'], la['v_data'] = store_quantized_weight(self, vis_cache[f'{prefix}.attn.v_padded.weight.{vis_prec}'])
            la['v_bias'] = store_weight(self, torch.from_numpy(vis_cache[f'{prefix}.attn.v_padded.bias'].copy()).view(torch.bfloat16))
            la['o_scale'], la['o_data'] = store_quantized_weight(self, vis_cache[f'{prefix}.attn.proj.weight.{vis_prec}'])
            la['o_bias'] = store_weight(self, torch.from_numpy(vis_cache[f'{prefix}.attn.proj.bias'].copy()).view(torch.bfloat16))
            for proj, hf_sub in [('gate', 'mlp.gate_proj'), ('up', 'mlp.up_proj'), ('down', 'mlp.down_proj')]:
                la[f'{proj}_scale'], la[f'{proj}_data'] = store_quantized_weight(self, vis_cache[f'{prefix}.{hf_sub}.weight.{vis_prec}'])
                la[f'{proj}_bias'] = store_weight(self, torch.from_numpy(vis_cache[f'{prefix}.{hf_sub}.bias'].copy()).view(torch.bfloat16))
            la['norm1_weight'] = store_weight(self, torch.from_numpy(vis_cache[f'{prefix}.norm1.weight'].copy()).view(torch.bfloat16))
            la['norm2_weight'] = store_weight(self, torch.from_numpy(vis_cache[f'{prefix}.norm2.weight'].copy()).view(torch.bfloat16))
            self.vis_layer_addrs.append(la)
        self.patch_weight_scale, self.patch_weight_data = store_quantized_weight(self, vis_cache[f'visual.patch_embed.proj.weight.{vis_prec}'])
        self.merger_ln_q_weight = store_weight(self, torch.from_numpy(vis_cache['visual.merger.ln_q.weight'].copy()).view(torch.bfloat16))
        self.merger_mlp0_scale, self.merger_mlp0_data = store_quantized_weight(self, vis_cache[f'visual.merger.mlp.0.weight.{vis_prec}'])
        self.merger_mlp0_bias = store_weight(self, torch.from_numpy(vis_cache['visual.merger.mlp.0.bias'].copy()).view(torch.bfloat16))
        self.merger_mlp2_scale, self.merger_mlp2_data = store_quantized_weight(self, vis_cache[f'visual.merger.mlp.2.weight.{vis_prec}'])
        self.merger_mlp2_bias = store_weight(self, torch.from_numpy(vis_cache['visual.merger.mlp.2.bias'].copy()).view(torch.bfloat16))
        print(f"Vision weights loaded ({vis_prec}): {vis_depth} layers + merger")

    def tensor_init(self) -> None:
        seq_len = self.MAX_CONTEXT_SIZE
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        ahd = self.actual_head_dim
        nkvh = self.num_kv_heads
        bpe = self.bytes_per_element

        print(f"Allocate tensor dram start at DRAM address: 0x{self.get_tensor_dram_addr():X}")
        zero_add = torch.zeros(seq_len * self.head_dim, dtype=torch.bfloat16)
        self.ZERO_DRAM_ADDR = self.allocate_tensor_dram(seq_len * self.head_dim * bpe)
        self.dma_to_accelerator_memory(self.ZERO_DRAM_ADDR, zero_add)
        self.IDENTITY_DRAM_ADDR = self.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe)
        self.dma_to_accelerator_memory(self.IDENTITY_DRAM_ADDR, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))

        self.LAYER0_FLASH_Q_DRAM = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        self.LAYER0_FLASH_K_DRAM = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        self.LAYER0_FLASH_V_DRAM = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        zero_pad = torch.zeros(aligned_seq_len * ahd, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_Q_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_K_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_V_DRAM, zero_pad)

        self.LAYER0_FLASH_OUT_HEAD_DRAM = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        self.LAYER0_FLASH_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.head_dim * self.group_size * bpe)
        self.LAYER0_FLASH_SCRATCH_DRAM = self.allocate_tensor_dram(max(ahd, UE_FMAX_CONTEXT_SIZE) * aligned_seq_len * 2 + ahd * aligned_seq_len * 2)
        aligned_tok = ((seq_len + 63) // 64) * 64
        self.LAYER0_FLASH_BIAS_DRAM = self.allocate_tensor_dram(aligned_tok * aligned_tok * bpe)

        self.LAYER0_INPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_Q_DRAM = self.allocate_tensor_dram(seq_len * self.q_size)
        self.LAYER0_K_DRAM = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_K_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_Q_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.q_size)
        self.LAYER0_V_PROJ_TEMP = self.allocate_tensor_dram(seq_len * self.k_size)
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
        self.OUTPUT_NORM_DRAM = self.allocate_tensor_dram(1 * self.vector_length * bpe)
        self.LOGITS_DRAM = self.allocate_tensor_dram(1 * self.EMBEDDING_ELEMENTS * bpe)

        vis_cfg = self._vis_cfg
        self.NUM_MERGED_TOKENS = vis_cfg.get("num_merged_tokens", 256)
        VS = vis_cfg["num_patches"]
        VH = vis_cfg["hidden_size"]
        VN = vis_cfg["num_heads"]
        VD = vis_cfg["head_dim"]
        VI = vis_cfg["intermediate_size"]
        VIS_MLP_PAD = ((VI + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
        VD_PAD = 128
        VH_OUT = vis_cfg["out_hidden_size"]
        VMERGE = vis_cfg["spatial_merge_size"]

        self.VIS_PATCH_INPUT_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_IO_A_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_IO_B_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_NORM_OUT_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_QK_DRAM = self.allocate_tensor_dram(VS * VN * VD_PAD * 2 * bpe)
        self.VIS_V_DRAM = self.allocate_tensor_dram(VS * VN * VD_PAD * bpe)
        self.VIS_Q_PAD_DRAM = self.allocate_tensor_dram(VN * VS * VD_PAD * bpe)
        self.VIS_K_PAD_DRAM = self.allocate_tensor_dram(VN * VS * VD_PAD * bpe)
        self.VIS_V_PAD_DRAM = self.allocate_tensor_dram(VN * VS * VD_PAD * bpe)
        vis_pad_zeros = torch.zeros(VN * VS * VD_PAD, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.VIS_Q_PAD_DRAM, vis_pad_zeros)
        self.dma_to_accelerator_memory(self.VIS_K_PAD_DRAM, vis_pad_zeros)
        self.dma_to_accelerator_memory(self.VIS_V_PAD_DRAM, vis_pad_zeros)
        self.VIS_ATTN_OUT_DRAM = self.allocate_tensor_dram(VN * VS * VD_PAD * bpe)
        self.VIS_ATTN_SCRATCH_DRAM = self.allocate_tensor_dram(
            max(VD_PAD, UE_FMAX_CONTEXT_SIZE) * VS * 2 + VD_PAD * VS * 2)
        self.VIS_ATTN_RESULT_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_ATTN_TRIM_DRAM = self.allocate_tensor_dram(VN * VS * VD * bpe)
        self.VIS_O_PROJ_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_RESIDUAL_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_MLP_GATE_DRAM = self.allocate_tensor_dram(VS * VIS_MLP_PAD * bpe)
        self.VIS_MLP_UP_DRAM = self.allocate_tensor_dram(VS * VIS_MLP_PAD * bpe)
        self.VIS_MLP_MULT_DRAM = self.allocate_tensor_dram(VS * VIS_MLP_PAD * bpe)
        self.VIS_MLP_DOWN_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_POST_NORM_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        merge_dim = VMERGE * VMERGE * VH
        self.VIS_MERGED_DRAM = self.allocate_tensor_dram(self.NUM_MERGED_TOKENS * merge_dim * bpe)
        self.VIS_MERGER_INTER_DRAM = self.allocate_tensor_dram(self.NUM_MERGED_TOKENS * merge_dim * bpe)
        self.VIS_ENCODER_OUT_DRAM = self.allocate_tensor_dram(self.NUM_MERGED_TOKENS * VH_OUT * bpe)
        self.VIS_ROPE_COS_DRAM = self.allocate_tensor_dram(VS * VD_PAD * bpe)
        self.VIS_ROPE_SIN_DRAM = self.allocate_tensor_dram(VS * VD_PAD * bpe)
        permute_size = UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe
        self.VIS_PERMUTE_PARAMS_DRAM = self.get_params_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, self.VIS_PERMUTE_PARAMS_DRAM,
                       torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16), permute_size)
        self.allocate_params_dram(permute_size)
        self.VIS_PERMUTE_TEMP_DRAM = self.allocate_tensor_dram(VN * VS * VD_PAD * bpe * 2)

        kv_cache_elems = self.LAYER_SIZE * nkvh * self.MAX_CONTEXT_SIZE * ahd
        kv_cache_bytes = kv_cache_elems * bpe
        self.LAYER0_V_DRAM = self.allocate_tensor_dram(kv_cache_bytes)
        self.LAYER0_K_ROPE_DRAM = self.allocate_tensor_dram(kv_cache_bytes)
        zero_kv = torch.zeros(kv_cache_elems, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, zero_kv)
        self.dma_to_accelerator_memory(self.LAYER0_K_ROPE_DRAM, zero_kv)

        print(f"    Tensor dram end at DRAM 0x{self.get_tensor_dram_addr():X}, usage: {self.get_tensor_dram_usage()} bytes")
        self._preallocate_identity_matrix()

    def _corrected_hw_latency_us(self, wall_time_s: float) -> float:
        raw_us = self.report_latency_in_us()
        overflow_us = (2**32) * self._clock_period_ns / 1e3
        corrected = raw_us
        while corrected + overflow_us / 2 < wall_time_s * 1e6:
            corrected += overflow_us
        return corrected

    @staticmethod
    def _wait_with_heartbeat(wait_fn, label: str, interval_s: float = 5.0):
        import threading
        stop = threading.Event()
        def _beat():
            n = 0
            while not stop.wait(interval_s):
                n += interval_s
                print(f"      [{label}] still running ({n:.0f}s)...", flush=True)
        t = threading.Thread(target=_beat, daemon=True)
        t0 = time.perf_counter()
        t.start()
        try:
            wait_fn()
        finally:
            stop.set()
            t.join(timeout=interval_s)
        return time.perf_counter() - t0

    def program_execute(self, program_start_addr: int, timeout: float = 120.0, gflops: float = None) -> None:
        print(f"  Running program on FPGA (DRAM addr 0x{program_start_addr:X})...", flush=True)
        self.start_execute_from_dram(program_start_addr)
        wall_s = self._wait_with_heartbeat(lambda: self.wait_queue(timeout), label="FPGA")
        latency_us = self._corrected_hw_latency_us(wall_s)
        print(f"    Total program execution latency = {latency_us:.0f} us")
        if gflops is not None:
            gflops_program = gflops / (latency_us * 1e-6) / 1e9
            self._last_hw_gflops = gflops_program
            print(f"    Throughput: {gflops_program:.2f} GFLOPS")

    # ------------------------------------------------------------------
    # Bin loaders (replace compile_* methods)
    # ------------------------------------------------------------------

    def load_encoder_bin(self) -> int:
        """Load pre-compiled encoder program. Returns program DRAM address."""
        bin_dir = os.path.join(self.script_dir, "qwen2.5_vl_3b_bin")
        enc_bin = os.path.join(bin_dir, "encoder_program.bin")
        addr, _ = self.load_instructions(enc_bin)
        return addr

    def load_prefill_bin(self, seq_len: int) -> tuple[int, int]:
        """Load pre-compiled prefill program for seq_len. Returns (addr, flops)."""
        bin_dir = os.path.join(self.script_dir, "qwen2.5_vl_3b_bin")
        cache_path = os.path.join(bin_dir, f"prefill_program_s{seq_len}.bin")
        meta_path = os.path.join(bin_dir, f"prefill_program_s{seq_len}.json")
        with open(meta_path) as f:
            meta = json.load(f)
        with open(cache_path, "rb") as f:
            prog_bytes = f.read()
        addr = self.get_program_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, addr, prog_bytes, len(prog_bytes))
        self.allocate_program_dram(len(prog_bytes))
        self.seq_len = seq_len - 1
        print(f"    Prefill program (seq_len={seq_len - 1}) loaded to 0x{addr:X}")
        return addr, meta["total_flops"]

    def load_decoder_bin(self) -> tuple[list[int], list[int], int]:
        """Load pre-compiled decoder program. Returns (program_sizes, flops_list, base_addr)."""
        paths_cfg = self._cfg.get("paths", {})
        dec_bin = os.path.join(self.script_dir, paths_cfg.get("decoder_program_bin", "qwen2.5_vl_3b_bin/decoder_program.bin"))
        dec_meta = os.path.join(self.script_dir, paths_cfg.get("decoder_program_meta", "qwen2.5_vl_3b_bin/decoder_program.json"))
        with open(dec_meta) as f:
            meta = json.load(f)
        if "instruction_counts" in meta:
            program_sizes = [c * 32 for c in meta["instruction_counts"]]
        else:
            program_sizes = meta["program_sizes"]
        flops_list = meta["total_flops"]
        base_addr, _ = self.load_instructions(dec_bin)
        return program_sizes, flops_list, base_addr

    # ------------------------------------------------------------------
    # Vision encoder
    # ------------------------------------------------------------------

    def prepare_encoder_input(self, pixel_values: torch.Tensor, image_grid_thw=None) -> dict:
        vis_cfg = self._vis_cfg
        VS = vis_cfg["num_patches"]
        VH = vis_cfg["hidden_size"]
        VD = vis_cfg["head_dim"]
        VD_PAD = 128
        model_dir = os.path.join(self.script_dir, self._cfg["paths"]["hf_model_dir"])
        if not hasattr(self, '_hf_model'):
            print("    Building visual encoder shell from config (no safetensors)...")
            # Build the model from config only (random weights). rot_pos_emb and
            # get_window_index are deterministic — they don't use learned weights —
            # so random weights are fine for them. patch_embed is the only layer
            # we actually invoke, and we overwrite its weights below from the bin
            # we dumped during compile (patch_embed_proj_weight.bin).
            from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration
            config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
            self._hf_model = Qwen2_5_VLForConditionalGeneration._from_config(config)
            self._hf_model = self._hf_model.to(torch.bfloat16)
            if not hasattr(self._hf_model, 'visual') and hasattr(self._hf_model, 'model') and hasattr(self._hf_model.model, 'visual'):
                self._hf_model.visual = self._hf_model.model.visual
            vis_bin_path = os.path.join(self.script_dir, self._cfg["paths"]["vision_weights"])
            pe_raw_path = os.path.join(os.path.dirname(vis_bin_path), "patch_embed_proj_weight.bin")
            pe_meta_path = pe_raw_path + ".json"
            if not (os.path.exists(pe_raw_path) and os.path.exists(pe_meta_path)):
                raise FileNotFoundError(
                    f"Missing {pe_raw_path} (+ .json). Recompile qwen2.5_vl_3b "
                    "with the updated test.py to regenerate.")
            with open(pe_meta_path) as _jf:
                _pe_meta = json.load(_jf)
            _pe_w = torch.from_numpy(np.fromfile(pe_raw_path, dtype=np.uint16)).view(torch.bfloat16).reshape(_pe_meta["shape"])
            self._hf_model.visual.patch_embed.proj.weight.data = _pe_w
            self._hf_model.eval()
            print("    Visual shell ready (patch_embed weights loaded from bin).")
        model = self._hf_model
        with torch.no_grad():
            if image_grid_thw is not None:
                pixel_values_hf = pixel_values.to(torch.bfloat16)
            else:
                from transformers import AutoTokenizer, Qwen2VLImageProcessor
                from PIL import Image
                img_np = pixel_values.float().permute(1, 2, 0).numpy()
                img_cfg = self._cfg.get("image_processing", {})
                mean = img_cfg.get("normalize_mean", [0.48145466, 0.4578275, 0.40821073])
                std = img_cfg.get("normalize_std", [0.26862954, 0.26130258, 0.27577711])
                for c in range(3):
                    img_np[:, :, c] = img_np[:, :, c] * std[c] + mean[c]
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                img_size = vis_cfg.get("image_size", 336)
                pil_img = pil_img.resize((img_size, img_size), Image.Resampling.BILINEAR)
                image_processor = Qwen2VLImageProcessor.from_pretrained(model_dir, trust_remote_code=True)
                img_inputs = image_processor(images=[pil_img], return_tensors="pt")
                pixel_values_hf = img_inputs["pixel_values"].to(torch.bfloat16)
                image_grid_thw = img_inputs["image_grid_thw"]
            visual = model.visual
            patch_embeds = visual.patch_embed(pixel_values_hf)
            assert patch_embeds.shape == (VS, VH), f"Patch embed shape {patch_embeds.shape} != ({VS}, {VH})"
            rotary_pos_emb = visual.rot_pos_emb(image_grid_thw)
            window_index, cu_window_seqlens = visual.get_window_index(image_grid_thw)
            spatial_merge_unit = visual.spatial_merge_size ** 2
            patch_embeds = patch_embeds.reshape(VS // spatial_merge_unit, spatial_merge_unit, -1)
            patch_embeds = patch_embeds[window_index, :, :]
            patch_embeds = patch_embeds.reshape(VS, -1)
            rotary_pos_emb = rotary_pos_emb.reshape(VS // spatial_merge_unit, spatial_merge_unit, -1)
            rotary_pos_emb = rotary_pos_emb[window_index, :, :]
            rotary_pos_emb = rotary_pos_emb.reshape(VS, -1)
            self._vis_reverse_index = torch.argsort(window_index)
            if isinstance(cu_window_seqlens, torch.Tensor):
                self._cu_window_seqlens = cu_window_seqlens.unique_consecutive().tolist()
            else:
                self._cu_window_seqlens = [cu_window_seqlens[0]] + [
                    cu_window_seqlens[i] for i in range(1, len(cu_window_seqlens))
                    if cu_window_seqlens[i] != cu_window_seqlens[i-1]]
        return {'patch_embeds': patch_embeds, 'rotary_pos_emb': rotary_pos_emb, 'image_grid_thw': image_grid_thw}

    def run_encoder(self, program_addr: int, pixel_values: torch.Tensor, image_grid_thw=None) -> None:
        vis_cfg = self._vis_cfg
        VS = vis_cfg["num_patches"]
        VH = vis_cfg["hidden_size"]
        VN = vis_cfg["num_heads"]
        VD = vis_cfg["head_dim"]
        VD_PAD = 128
        bpe = 2
        prep = self.prepare_encoder_input(pixel_values, image_grid_thw)
        patch_embeds = prep['patch_embeds']
        rotary_pos_emb = prep['rotary_pos_emb']
        image_grid_thw = prep['image_grid_thw']
        patch_bf16 = patch_embeds.to(torch.bfloat16).contiguous().flatten()
        self.dma_to_accelerator_memory(self.VIS_PATCH_INPUT_DRAM, patch_bf16)
        vis_pad_zeros = torch.zeros(VN * VS * VD_PAD, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.VIS_Q_PAD_DRAM, vis_pad_zeros)
        self.dma_to_accelerator_memory(self.VIS_K_PAD_DRAM, vis_pad_zeros)
        self.dma_to_accelerator_memory(self.VIS_V_PAD_DRAM, vis_pad_zeros)
        half_d = VD // 2
        with torch.no_grad():
            cos_raw = rotary_pos_emb.cos().to(torch.bfloat16)
            sin_raw = rotary_pos_emb.sin().to(torch.bfloat16)
            cos_table = torch.ones(VS, VD_PAD, dtype=torch.bfloat16)
            cos_table[:, :half_d] = cos_raw
            cos_table[:, 64:64 + half_d] = cos_raw
            sin_table = torch.zeros(VS, VD_PAD, dtype=torch.bfloat16)
            sin_table[:, :half_d] = -sin_raw
            sin_table[:, 64:64 + half_d] = sin_raw
            self.dma_to_accelerator_memory(self.VIS_ROPE_COS_DRAM, cos_table.flatten())
            self.dma_to_accelerator_memory(self.VIS_ROPE_SIN_DRAM, sin_table.flatten())
        print(f"    Running vision encoder on FPGA (~40-60s)...", flush=True)
        self.start_execute_from_dram(program_addr)
        wall_s = self._wait_with_heartbeat(lambda: self.wait_queue(600.0), label="vision")
        latency_us = self._corrected_hw_latency_us(wall_s)
        print(f"    Vision encoder: {latency_us/1e6:.2f}s")
        num_merged = self.NUM_MERGED_TOKENS
        out_hidden = self._vis_cfg["out_hidden_size"]
        if hasattr(self, '_vis_reverse_index'):
            merger_out = self.dma_from_accelerator_memory(self.VIS_ENCODER_OUT_DRAM, (num_merged, out_hidden)).cpu()
            merger_out = merger_out[self._vis_reverse_index].contiguous()
            self.dma_to_accelerator_memory(self.VIS_ENCODER_OUT_DRAM, merger_out.flatten())
        self._vis_num_tokens = num_merged
        self._image_grid_thw = image_grid_thw

    def run_prefill(self, prefill_program_addr: int, prefill_seq, gflops: int = None, has_image: bool = False) -> None:
        if prefill_seq is None:
            prefill_seq = tuple(self._cfg["default_prefill_tokens"])
        if len(prefill_seq) > 1:
            prefill_seq = prefill_seq[:-1]
            assert len(prefill_seq) == self.seq_len, f"Expected seq_len {self.seq_len}, got {len(prefill_seq)}"
        else:
            raise ValueError("Prefill sequence must have at least 2 tokens.")
        seq_len = len(prefill_seq)
        aligned_tok = ((seq_len + 63) // 64) * 64
        embedding_tensor = self.get_embedding_for_tokens(prefill_seq)
        if has_image:
            vision_token_id = 151655
            img_positions = [i for i, t in enumerate(prefill_seq) if t == vision_token_id]
            num_vis = getattr(self, '_vis_num_tokens', 0)
            if len(img_positions) > 0 and num_vis > 0:
                vis_embeddings = self.dma_from_accelerator_memory(
                    self.VIS_ENCODER_OUT_DRAM, (num_vis, self.vector_length)).cpu()
                embed_reshaped = embedding_tensor.reshape(seq_len, self.vector_length)
                n_replace = min(len(img_positions), num_vis)
                for i in range(n_replace):
                    embed_reshaped[img_positions[i]] = vis_embeddings[i]
                embedding_tensor = embed_reshaped.flatten()
                print(f"    Merged {n_replace} vision tokens into prefill")
        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
        bias = torch.full((aligned_tok, aligned_tok), float("-inf"), dtype=torch.bfloat16)
        valid_mask = torch.tril(torch.ones(aligned_tok, aligned_tok, dtype=torch.bool), diagonal=0)
        bias.masked_fill_(valid_mask, 0.0)
        bias[:, seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias)
        self.program_execute(prefill_program_addr, gflops=gflops)

    def run_decoder(self, decoder_program_sizes: list[int], decoder_base_addr: int, token_id: int,
                    gflops_per_token=None, repetition_penalty: float = 1.2) -> int:
        if token_id is None:
            print("No last token for decode.")
            return self.seq_len
        _stop_tokens = {151643, 151645, self._end_of_turn_token_id}
        ahd = self.actual_head_dim
        bpe = self.bytes_per_element
        _kv_stride = ahd * bpe
        _rope_stride = ahd * 2 * bpe
        generated_tokens = set()
        max_seq_len = self.MAX_CONTEXT_SIZE
        while self.seq_len < max_seq_len:
            _set_silent(True)
            self.seq_len += 1
            aligned_seq_len = ((self.seq_len + 63) // 64) * 64
            prog_idx = min((self.seq_len - 1) // 64, len(decoder_program_sizes) - 1)
            prog_addr = decoder_base_addr + sum(decoder_program_sizes[:prog_idx])
            self.isa_add_set_core(self.V_CACHE_SIZE_REG, ue_35bit_addr_shifter((self.seq_len - 1) * _kv_stride))
            _rope_pos = self.seq_len - 1 + getattr(self, '_rope_offset', 0)
            self.isa_add_set_core(self.ROPE_SIZE_REG, ue_35bit_addr_shifter(_rope_pos * _rope_stride))
            embedding_tensor = self.get_embedding_for_tokens([token_id])
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
            bias_host = torch.full((1, aligned_seq_len), -1e36, dtype=torch.bfloat16)
            bias_host[0, :self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_host)
            self.start_execute_from_dram(prog_addr)
            self.wait_queue(10.0)
            if repetition_penalty != 1.0 and len(generated_tokens) > 0:
                logits = self.dma_from_accelerator_memory(self.LOGITS_DRAM, (self.EMBEDDING_ELEMENTS,)).cpu().float()
                for prev_tok in generated_tokens:
                    if logits[prev_tok] > 0:
                        logits[prev_tok] /= repetition_penalty
                    else:
                        logits[prev_tok] *= repetition_penalty
                token_id = int(logits.argmax().item())
            else:
                token_id = self.get_arg_max_index()
            generated_tokens.add(token_id)
            token_char = self.tokenizer.decode([token_id])
            _set_silent(False)
            if token_id in _stop_tokens:
                print(f"\nStop token {token_id} reached.")
                break
            print(token_char, end="", flush=True)

        return self.seq_len


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Qwen2.5-VL-3B inference from pre-compiled bins")
    parser.add_argument("--prompt", type=str, default="please describe the image in details.")
    parser.add_argument("--image", type=str,
                        default=os.path.join(SCRIPT_DIR, "../../test_samples/yosemite.jpg"),
                        help="Image path, or 'none' for text-only")
    parser.add_argument("--dev", type=str, default="xdma0")
    parser.add_argument("--cycle", type=float, default=5.63)
    parser.add_argument("--rep-penalty", type=float, default=1.05)
    args = parser.parse_args()

    if args.image and args.image.lower() == "none":
        args.image = None
    has_image = args.image is not None

    set_dma_device(args.dev)
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle

    script_dir = SCRIPT_DIR
    cfg = _load_config(script_dir)
    tokenizer_dir = os.path.join(script_dir, cfg["paths"]["hf_model_dir"])

    # Build prompt and tokenize to get prefill_seq length for bin check
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    prompt_text = args.prompt or "Describe this image."
    if has_image:
        num_vis_tokens = cfg["vision"]["num_merged_tokens"]
        image_pad_id = 151655
        vision_start_id = 151652
        vision_end_id = 151653
        conversation_text = [{"role": "user", "content": prompt_text}]
        prompt_with_template = tokenizer.apply_chat_template(conversation_text, tokenize=False, add_generation_prompt=True)
        text_tokens = list(tokenizer.encode(prompt_with_template, add_special_tokens=False))
        vis_placeholder = [vision_start_id] + [image_pad_id] * num_vis_tokens + [vision_end_id]
        insert_pos = 0
        for idx in range(len(text_tokens) - 2):
            if text_tokens[idx:idx+3] == [151644, 872, 198]:
                insert_pos = idx + 3
                break
        prefill_seq = tuple(text_tokens[:insert_pos] + vis_placeholder + text_tokens[insert_pos:])
    else:
        conversation = [{"role": "user", "content": prompt_text}]
        prompt_with_template = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        prefill_seq = tuple(tokenizer.encode(prompt_with_template, add_special_tokens=False))

    missing = check_bins(cfg, script_dir, has_image, prefill_seq_len=len(prefill_seq))
    if missing:
        _original_print("Missing bin files (run qwen2.5_vl_3b_test.py first to compile):")
        for m in missing:
            _original_print(f"  {m}")
        return

    _original_print(f"\n--- Configuration ---")
    _original_print(f"  Image:  {args.image if has_image else '(none)'}")
    _original_print(f"  Prompt: {prompt_text!r}")
    _original_print(f"  Prefill: {len(prefill_seq)} tokens")

    _set_silent(True)
    ue = Qwen25VL3B_UnifiedEngine(script_dir=script_dir)
    ue.software_reset()
    _set_silent(False)

    # Drain any stale FPGA execution
    ue.dram_inst_running(False)
    ue.start_capture()
    ue.generate_instruction_halt()
    ue.stop_capture()
    halt_bytes = bytearray()
    for inst in ue.capture_buffer:
        halt_bytes.extend(inst.get_bytes())
    ue.dma_write(DMA_DEVICE_H2C, DRAM_INSTRUCTION_ADDR, halt_bytes, len(halt_bytes))
    ue.clear_capture_buffer()

    if has_image:
        _original_print(f"\n--- Vision Encoder ---")
        timer_vis = time.perf_counter()
        pixel_values = process_image(args.image)
        _original_print(f"Image loaded: {args.image} -> {pixel_values.shape}")
        ue.prepare_encoder_input(pixel_values)
        encoder_addr = ue.load_encoder_bin()
        ue.run_encoder(encoder_addr, pixel_values)
        t_vis = time.perf_counter() - timer_vis
        _original_print(f"Vision encoder: {t_vis:.2f}s")

    if has_image:
        image_grid_thw = getattr(ue, '_image_grid_thw', None)
        ue.load_mrope_for_prefill(prefill_seq, image_grid_thw=image_grid_thw)

    _original_print(f"\n--- Prefill ({len(prefill_seq)} tokens) ---")
    timer = time.perf_counter()
    prefill_program_addr, gflops_prefill = ue.load_prefill_bin(seq_len=len(prefill_seq))
    ue.run_prefill(prefill_program_addr, prefill_seq=prefill_seq, gflops=gflops_prefill, has_image=has_image)
    t_prefill = time.perf_counter() - timer
    _original_print(f"  {t_prefill:.2f}s")

    _original_print(f"\n--- Decoder ---")
    timer = time.perf_counter()
    if has_image:
        ue.restore_rope_for_decoder()
        ue._rope_offset = getattr(ue, '_mrope_delta', 0)
    else:
        ue._rope_offset = 0
    decoder_program_sizes, gflops_per_token, decoder_base_addr = ue.load_decoder_bin()
    token_cnt = ue.run_decoder(decoder_program_sizes, decoder_base_addr,
                               token_id=prefill_seq[-1],
                               gflops_per_token=gflops_per_token,
                               repetition_penalty=args.rep_penalty)
    t_decode = time.perf_counter() - timer
    n_new = token_cnt - len(prefill_seq)
    _original_print(f"\n  {t_decode:.2f}s ({n_new} tokens, {t_decode/max(n_new,1):.2f}s/tok)")


if __name__ == "__main__":
    main()
