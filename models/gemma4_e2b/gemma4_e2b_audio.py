#!/usr/bin/env python3
"""Production Gemma4 E2B audio encoder for the refactored FPGA pipeline.

``Gemma4AudioMixin`` mirrors the lifecycle of ``Gemma4VisionMixin``:
weights are loaded once, tensors are allocated for the fixed processor shape,
the full encoder is captured as one straight-line FPGA program, and runtime
only uploads input data, launches that program, and returns LM soft tokens.
"""
import math
import os
import sys
import time

_SD = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SD)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SD)))

import torch
import torch.nn.functional as F
from user_dma_core import (
    DMA_DEVICE_H2C, TYPE, UE_VECTOR_SIZE, UE_MODE,
    URAM_NEAR_FULL_SIZE, URAM_SECTION, URAM_START_ADDR,
)
from audio_primitives import (
    silu_core_dram as _aud_silu,
    glu_core_dram as _aud_glu,
    half_step_residual_core_dram as _aud_half_step,
    depthwise_conv1d_core_dram as _aud_depthwise_conv1d,
    copy_dram_to_dram_chunked as _aud_copy_chunked,
    eltwise_add_core_dram as _aud_eltwise_add,
)

AUDIO_QUANT_PRECISION = "if4"
AUDIO_FIXED_SOFT_TOKENS = 128
AUDIO_ISA_BASE = 0x98000000
AUDIO_PROGRAM_VERSION = 7


def build_audio_weight_section(hf_model, parallel_quantize) -> tuple[bytes, dict]:
    """Pre-quantize the audio encoder weights once and return
    (section_bytes, section_manifest). Mirrors _build_vision_section_bytes.

    Bin layout includes: subsample stem (conv0/conv1/proj with the
    Phase A2.3 row-duplication + K=2048 padding baked in), 12 Conformer
    layers, output_proj + multimodal embedder. Generated tensors
    (G_s0 / gamma_64 / ID_64) are recomputed at runtime by
    audio_weight_init from canonical-shape constants and don't need to
    be stored.

    Caller writes section_bytes into the combined weights bin and the
    manifest goes into the master JSON under "audio_section".
    """
    am = hf_model.model.audio_tower
    ea = hf_model.model.embed_audio

    sections: list[tuple[str, bytes, list[int], str]] = []

    def _add_bf16(key: str, w: torch.Tensor) -> None:
        wc = w.contiguous()
        sections.append((key, wc.view(torch.uint8).numpy().tobytes(), list(wc.shape), "bf16"))

    def _add_if4_batch(tensors: list[tuple[str, torch.Tensor]]) -> None:
        for name, w in tensors:
            assert w.dim() == 2, f"{name}: expected 2D weight, got {tuple(w.shape)}"
            assert w.shape[1] % 64 == 0, f"{name}: K={w.shape[1]} not divisible by 64"
        tensors_c = [w.contiguous() for _, w in tensors]
        results = parallel_quantize(AUDIO_QUANT_PRECISION, tensors_c)
        for (name, _), w, (data_b, scale_b) in zip(tensors, tensors_c, results):
            sections.append((f"{name}.scale", scale_b, list(w.shape), f"{AUDIO_QUANT_PRECISION}_scale"))
            sections.append((f"{name}.data",  data_b,  list(w.shape), f"{AUDIO_QUANT_PRECISION}_data"))

    # ---- Subsample weights with Phase A2.3 transformations baked in ----
    sub = am.subsample_conv_projection
    w0 = sub.layer0.conv.weight.detach().cpu().to(torch.bfloat16).reshape(128, 9)
    w0_padded = torch.zeros(128, 64, dtype=torch.bfloat16)
    w0_padded[:, :9] = w0
    w1 = sub.layer1.conv.weight.detach().cpu().to(torch.bfloat16).permute(0, 2, 3, 1).reshape(32, 9 * 128)
    w1_padded = torch.zeros(64, 9 * 128, dtype=torch.bfloat16)
    w1_padded[:32] = w1
    w1_padded[32:] = w1  # Phase A2.3: duplicate rows for N=32 LN trick
    W1_sub = 32
    proj_orig = sub.input_proj_linear.weight.detach().cpu().to(torch.bfloat16)
    proj_padded = torch.zeros(1024, 2 * 1024, dtype=torch.bfloat16)
    for w1_idx in range(W1_sub):
        proj_padded[:, w1_idx * 64:w1_idx * 64 + 32] = proj_orig[:, w1_idx * 32:(w1_idx + 1) * 32]
    _add_if4_batch([
        ("subsample.conv0", w0_padded),
        ("subsample.conv1", w1_padded),
        ("subsample.proj",  proj_padded),
    ])
    # LN gammas (BF16). gamma_64 = concat(gamma_32, gamma_32) baked here.
    ln0_gamma = sub.layer0.norm.weight.detach().cpu().to(torch.bfloat16)
    ln1_gamma_32 = sub.layer1.norm.weight.detach().cpu().to(torch.bfloat16)
    _add_bf16("subsample.ln0_gamma", ln0_gamma)
    _add_bf16("subsample.ln1_gamma_64", torch.cat([ln1_gamma_32, ln1_gamma_32], dim=0))

    # ---- Per-layer Conformer weights ----
    L_count = len(am.layers)
    clip_ranges: list[dict] = []
    def _finite(x):
        if x == float("inf"):  return "inf"
        if x == -float("inf"): return "-inf"
        return float(x)
    def _proj_clip(proj):
        if not getattr(proj, "use_clipped_linears", False):
            return {"in_min": "-inf", "in_max": "inf", "out_min": "-inf", "out_max": "inf"}
        return {
            "in_min":  _finite(proj.input_min.item()),
            "in_max":  _finite(proj.input_max.item()),
            "out_min": _finite(proj.output_min.item()),
            "out_max": _finite(proj.output_max.item()),
        }

    for li in range(L_count):
        L = am.layers[li]
        pre = f"layer{li}"
        ff1 = L.feed_forward1
        sa = L.self_attn
        cv = L.lconv1d
        ff2 = L.feed_forward2
        # BF16 norms / scales (8 per layer)
        _add_bf16(f"{pre}.ff1_pre_norm",    ff1.pre_layer_norm.weight.detach().cpu().to(torch.bfloat16))
        _add_bf16(f"{pre}.ff1_post_norm",   ff1.post_layer_norm.weight.detach().cpu().to(torch.bfloat16))
        _add_bf16(f"{pre}.attn_pre_norm",   L.norm_pre_attn.weight.detach().cpu().to(torch.bfloat16))
        _add_bf16(f"{pre}.per_dim_scale",   sa.per_dim_scale.detach().cpu().to(torch.bfloat16))
        _add_bf16(f"{pre}.attn_post_norm",  L.norm_post_attn.weight.detach().cpu().to(torch.bfloat16))
        _add_bf16(f"{pre}.conv_pre_norm",   cv.pre_layer_norm.weight.detach().cpu().to(torch.bfloat16))
        dw_w = cv.depthwise_conv1d.weight.detach().cpu().to(torch.bfloat16).squeeze(1)
        _add_bf16(f"{pre}.conv_dw_w",       dw_w)
        _add_bf16(f"{pre}.conv_norm",       cv.conv_norm.weight.detach().cpu().to(torch.bfloat16))
        _add_bf16(f"{pre}.ff2_pre_norm",    ff2.pre_layer_norm.weight.detach().cpu().to(torch.bfloat16))
        _add_bf16(f"{pre}.ff2_post_norm",   ff2.post_layer_norm.weight.detach().cpu().to(torch.bfloat16))
        _add_bf16(f"{pre}.norm_out",        L.norm_out.weight.detach().cpu().to(torch.bfloat16))
        # IF4 projections (11 per layer)
        _add_if4_batch([
            (f"{pre}.ff1_w1",         ff1.ffw_layer_1.linear.weight.detach().cpu().to(torch.bfloat16)),
            (f"{pre}.ff1_w2",         ff1.ffw_layer_2.linear.weight.detach().cpu().to(torch.bfloat16)),
            (f"{pre}.q_proj",         sa.q_proj.linear.weight.detach().cpu().to(torch.bfloat16)),
            (f"{pre}.k_proj",         sa.k_proj.linear.weight.detach().cpu().to(torch.bfloat16)),
            (f"{pre}.v_proj",         sa.v_proj.linear.weight.detach().cpu().to(torch.bfloat16)),
            (f"{pre}.o_proj",         sa.post.linear.weight.detach().cpu().to(torch.bfloat16)),
            (f"{pre}.rel_k_proj",     sa.relative_k_proj.weight.detach().cpu().to(torch.bfloat16)),
            (f"{pre}.conv_lin_start", cv.linear_start.linear.weight.detach().cpu().to(torch.bfloat16)),
            (f"{pre}.conv_lin_end",   cv.linear_end.linear.weight.detach().cpu().to(torch.bfloat16)),
            (f"{pre}.ff2_w1",         ff2.ffw_layer_1.linear.weight.detach().cpu().to(torch.bfloat16)),
            (f"{pre}.ff2_w2",         ff2.ffw_layer_2.linear.weight.detach().cpu().to(torch.bfloat16)),
        ])
        # Per-layer clip ranges for the ClippableLinear wrappers (small JSON).
        clip_ranges.append({
            "ff1_w1":         _proj_clip(ff1.ffw_layer_1),
            "ff1_w2":         _proj_clip(ff1.ffw_layer_2),
            "q_proj":         _proj_clip(sa.q_proj),
            "k_proj":         _proj_clip(sa.k_proj),
            "v_proj":         _proj_clip(sa.v_proj),
            "o_proj":         _proj_clip(sa.post),
            "conv_lin_start": _proj_clip(cv.linear_start),
            "conv_lin_end":   _proj_clip(cv.linear_end),
            "ff2_w1":         _proj_clip(ff2.ffw_layer_1),
            "ff2_w2":         _proj_clip(ff2.ffw_layer_2),
        })

    # ---- Output projection + multimodal embedder ----
    out_proj_w = am.output_proj.weight.detach().cpu().to(torch.bfloat16)
    out_proj_b = am.output_proj.bias.detach().cpu().to(torch.bfloat16)
    embedder_w = ea.embedding_projection.weight.detach().cpu().to(torch.bfloat16)
    _add_if4_batch([
        ("output_proj",   out_proj_w),
        ("embedder_proj", embedder_w),
    ])
    _add_bf16("output_proj_bias", out_proj_b)

    # Concatenate and build manifest.
    out = bytearray()
    section_meta: dict = {}
    cur = 0
    for key, b, shape, dtype in sections:
        section_meta[key] = {"offset": cur, "size": len(b), "shape": shape, "dtype": dtype}
        out.extend(b)
        cur += len(b)
    manifest = {
        "audio_quant_precision": AUDIO_QUANT_PRECISION,
        "num_layers": L_count,
        "AUD_H":         int(am.config.hidden_size),
        "AUD_HEADS":     int(am.config.num_attention_heads),
        "AUD_HEAD_DIM":  int(am.config.hidden_size) // int(am.config.num_attention_heads),
        "AUD_FFN":       int(am.config.hidden_size) * 4,
        "AUD_OUT_DIM":   int(ea.embedding_projection.weight.shape[0]),
        "sections":      section_meta,
        "clip_ranges":   clip_ranges,
    }
    print(f"  Audio section: {cur/1024**2:.1f} MiB, {len(section_meta)} tensors")
    return bytes(out), manifest

class Gemma4AudioMixin:
    """Production-only audio methods for ``Gemma4_UnifiedEngine``."""

    def _emit_audio_core(self, emit) -> None:
        """Reset per-core PBI allocators, then emit into the open capture."""
        self.reset_inst_ptr_counter()
        self._isa_reg_counter = self._isa_reg_base
        emit()

    def _aud_mm_gpr(self, M: int):
        if self.capture_buffer is None:
            return None
        self.generate_instruction_add_set(self.gpr_seq_len, M)
        return self.gpr_seq_len

    def _aud_clip_dram(self, addr: int, shape: tuple,
                       clamp_min: float, clamp_max: float) -> None:
        rows, cols = shape
        self._emit_audio_core(lambda: self.matmat_mul_core(
            M=rows * cols // 64, K=64, N=64,
            A_DRAM_ADDR=addr, B_DRAM_ADDR=self.AUD_IDENTITY_64,
            OUTPUT_DRAM_ADDR=addr, is_B_quantized=False,
            clamp_enable=True, clamp_min=clamp_min, clamp_max=clamp_max,
            gpr_M_reg=self._aud_mm_gpr(rows * cols // 64)))

    def _compile_and_run_single(self, _label: str, emit, **_unused) -> None:
        """Compatibility shim for the reference call chains: production audio
        is always emitted into one open capture and never runs per operation."""
        self._emit_audio_core(emit)

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
        # Phase 4: PBI variants of per-layer rms_norms + matmuls. Real ISA-loop
        # compression for norms; per-tile pointer-backed descriptors for matmuls.
        # 14 percent encoder-bin shrink end-to-end. Always on.
        print(f"[Audio] config loaded: {self.AUD_LAYERS} layers, "
              f"H={self.AUD_H}, heads={self.AUD_HEADS}, FFN={self.AUD_FFN}, "
              f"conv_k={self.AUD_CONV_K}, chunk={self.AUD_CHUNK}, ctx={self.AUD_CTX}")

    def _audio_weight_init_from_combined_bin(self, weights_bin_path: str, audio_section: dict) -> None:
        """Load pre-quantized audio weights from the combined weights bin.
        Mirrors _vision_weight_init_from_combined_bin: no HF model, no
        quantization. The allocation ORDER below MUST match audio_weight_init
        (HF path) so that DRAM addresses baked into the captured audio ISA
        resolve to the same buffers at runtime.

        Generated tensors (G_s0 / ID_64) are not in the bin — they are
        deterministic functions of canonical-shape constants and we
        recompute them here on host before DMA.
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

        # Route audio weight uploads to the dedicated Weight Audio params
        # region (0x6c000000–0x78000000); restore the tensor cursor at the
        # end so encoder intermediates start from the LM tensor end. Same
        # contract as audio_weight_init (HF path).
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
                # Order MUST match _upload_fp4_batch: scale first, then data.
                scale_addr = _dma_section(f"{name}.scale")
                data_addr  = _dma_section(f"{name}.data")
                return {"data": data_addr, "scale": scale_addr,
                        "shape": tuple(sections[f"{name}.data"]["shape"])}

            # ---- Subsample IF4 weights (conv0, conv1, proj) — batched order ----
            self._aud_sub_conv0_addrs = _alloc_if4("subsample.conv0")
            self._aud_sub_conv1_addrs = _alloc_if4("subsample.conv1")
            self._aud_sub_proj_addrs  = _alloc_if4("subsample.proj")

            # ---- Subsample BF16 gammas and runtime-generated helpers ----
            self._aud_sub_ln0_gamma_addr = _dma_section("subsample.ln0_gamma")
            self._aud_sub_ln1_gamma_addr = _dma_section("subsample.ln1_gamma_64")
            # ID_64 is just eye(64); recompute on host.
            id_64_addr = self.allocate_tensor_dram(64 * 64 * bpe)
            self.dma_to_accelerator_memory(id_64_addr,
                torch.eye(64, dtype=torch.bfloat16).contiguous())
            self._aud_sub_id_64_addr = id_64_addr
            # Trick 9: shared LayerNorm zeros base (N=128 covers ln0 N=128 + ln1 N=64).
            # Seeded here like id_64 (deterministic addr, replayed on run_from_bin load);
            # passed to both subsample LNs via ZEROS_DRAM_ADDR.
            ln_zeros_addr = self.allocate_tensor_dram(128 * bpe)
            self.dma_to_accelerator_memory(ln_zeros_addr,
                torch.zeros(128, dtype=torch.bfloat16).contiguous())
            self._aud_ln_zeros_addr = ln_zeros_addr
            # G_s0 is deterministic (parakeet pattern, depends only on n_mels=128);
            # recompute on host. Must follow exact size used by audio_weight_init.
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

            # ---- Per-layer Conformer weights ----
            # Allocation order MUST match audio_weight_init exactly:
            # BF16 norms first (in order), then 11 IF4 projections batched.
            layer_addrs: list[dict] = []
            clip_ranges: list[dict] = []
            def _str_to_float(v):
                if v == "inf":  return float("inf")
                if v == "-inf": return -float("inf")
                return float(v)
            def _to_old_clip(cr: dict) -> dict:
                # _audio_weight_init returns dict with keys "in_min" etc; same here.
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
                # audio_tensor_init reads dw_w[:, t] to build per-tap tile
                # buffers. Read the bytes once, parse to a host tensor, then
                # DMA from those bytes — avoids a separate C2H read.
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
                # IF4 batch (11 tensors in the audio_weight_init order)
                for proj in ["FF1_W1", "FF1_W2", "Q_PROJ", "K_PROJ", "V_PROJ",
                              "O_PROJ", "REL_K_PROJ", "CONV_LIN_START", "CONV_LIN_END",
                              "FF2_W1", "FF2_W2"]:
                    addrs[proj] = _alloc_if4(f"{pre}.{proj.lower()}")
                layer_addrs.append(addrs)
                hf_cache.append({"dw_w": dw_w_host})  # other fields not needed in FPGA pipeline
                # Clip ranges per layer
                cr_src = meta["clip_ranges"][li]
                layer_cr: dict = {}
                for k in ("ff1_w1", "ff1_w2", "q_proj", "k_proj", "v_proj", "o_proj",
                          "conv_lin_start", "conv_lin_end", "ff2_w1", "ff2_w2"):
                    layer_cr[k.upper()] = _to_old_clip(cr_src[k])
                clip_ranges.append(layer_cr)
            self._aud_weight_addrs = layer_addrs
            self._aud_clip_ranges = clip_ranges
            self._aud_hf_layers = hf_cache

            # ---- Output projection + multimodal embedder (IF4 batch + BF16 bias) ----
            self._aud_output_proj_addrs = _alloc_if4("output_proj")
            self._aud_embedder_proj_addrs = _alloc_if4("embedder_proj")
            self._aud_output_proj_b_addr = _dma_section("output_proj_bias")
        finally:
            f.close()

        # Sanity: assert audio weights fit in the 192 MB region.
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

    def audio_weight_init(self) -> None:
        """Load the pre-quantized audio section exactly once."""
        if getattr(self, "_audio_weight_init_done", False):
            return
        section = getattr(self, "_weights_master", {}).get("audio_section")
        if not section or "manifest" not in section:
            raise RuntimeError(
                "audio section missing from params.bin; regenerate the combined "
                "weight image with audio enabled")
        self._audio_weight_init_from_combined_bin(
            os.path.join(self.script_dir, self._weights_bin_rel), section)
        self._audio_weight_init_done = True

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
        # rms_norm_core_dram expects a gamma tile; passing ones gives plain RMSNorm.
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
        """FPGA port of audio_subsample_host (Phase A).

        Real compute (matmul, LayerNorm, ReLU, Linear) on FPGA. The im2col
        data rearrangement is computed on host (deterministic shuffle, no
        learned params -- same category as the log-mel feature extraction
        in stage 0). When fully validated, the im2col can move to FPGA via
        the parakeet im2col-via-G pattern.

        Mirrors Gemma4AudioSubSampleConvProjection:
            mask -> Conv2d -> LayerNorm -> ReLU -> mask -> Conv2d -> LayerNorm
            -> ReLU -> flatten -> input_proj_linear

        Returns (hidden_states, mask_downsampled_2x) for parity with
        audio_subsample_host.
        """
        if not hasattr(self, "_aud_sub_conv0_addrs"):
            raise RuntimeError("audio_weight_init must run before audio_subsample_fpga")
        H_OUT = self.AUD_H            # 1024
        bpe = self.bytes_per_element
        x = input_features.detach().cpu()    # (B, T_raw, n_mels)
        mask = input_features_mask
        if x.dim() != 3 or x.shape[0] != 1:
            raise RuntimeError(f"audio_subsample_fpga only supports B=1; got {tuple(x.shape)}")
        T_raw = int(x.shape[1])
        N_MELS = int(x.shape[2])
        assert N_MELS == 128, f"expected n_mels=128, got {N_MELS}"
        # Padding=1 stride=2 on both dims: T_raw must be even for the integer
        # divisions below to line up.
        H0 = (T_raw + 2 - 3) // 2 + 1     # post-conv0 height (time)
        W0 = (N_MELS + 2 - 3) // 2 + 1    # post-conv0 width (freq) = 64
        H1 = (H0 + 2 - 3) // 2 + 1        # post-conv1 height
        W1 = (W0 + 2 - 3) // 2 + 1        # post-conv1 width = 32
        N0 = H0 * W0                       # patches count after conv0 = output rows
        N1 = H1 * W1                       # patches count after conv1
        N1_pad = ((N1 + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
        N0_pad = ((N0 + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE

        # ---- Apply mask on host (zero invalid input rows) ----
        if mask is not None:
            x = x * mask[:, :, None].to(x.dtype)
        x = x.to(torch.bfloat16).squeeze(0).contiguous()  # (T_raw, n_mels=128)

        # ---- Stage 0 im2col on FPGA (Phase A2.1) ------------------------
        # Parakeet pattern: gather 3 strided rows of input into R_combined
        # (H0_pad, K_g=384), then matmul R_combined @ G_s0^T to produce the
        # (H0_pad, W_out*64=4096) im2col patches. The output's byte layout
        # is bit-identical to the (N0_pad=H0*W0, 64) patches0 view that
        # conv0 consumes, because W_out_s0 = 64 (matmul N-stride matches
        # patch row stride).
        K_g_s0 = self._aud_sub_K_g_s0   # 384
        N_g_s0 = self._aud_sub_N_g_s0   # 4096
        H0_pad = ((H0 + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
        # AUD_SUB_PATCHES0 must hold matmul output H0_pad * N_g_s0 * bpe.
        patches0_buf_bytes = H0_pad * N_g_s0 * bpe
        # Source input buffer: T_raw rows of 128 mel bins, aligned to a
        # 64-byte boundary on the DMA side. Pad T_raw to even for stride=2.
        T_raw_pad = T_raw + (T_raw & 1)

        # Allocate buffers lazily on first call. Size by the current audio's
        # N0/N1, padded to VS. Re-uses the same DRAM addresses on subsequent
        # calls if the dimensions match (cached by an explicit size tag).
        cache_key = (N0_pad, N1_pad, H0_pad, T_raw_pad)
        if getattr(self, "_aud_sub_buf_key", None) != cache_key:
            self._aud_sub_buf_key = cache_key
            # Raw mel input (T_raw_pad, 128) — FPGA-side source for the
            # strided gather DMAs that build R_combined.
            self.AUD_SUB_INPUT = self.allocate_tensor_dram(T_raw_pad * 128 * bpe)
            # R_combined (H0_pad, K_g_s0): 3 strided rows of input
            # concatenated per output time index, padded for matmul K alignment.
            self.AUD_SUB_R_COMBINED = self.allocate_tensor_dram(H0_pad * K_g_s0 * bpe)
            # patches0: holds the im2col matmul output (H0_pad, N_g_s0). Sized
            # for the matmul write; conv0 reads the first (N0_pad, 64) view.
            self.AUD_SUB_PATCHES0 = self.allocate_tensor_dram(patches0_buf_bytes)
            self.AUD_SUB_PATCHES1 = self.allocate_tensor_dram(N1_pad * 1152 * bpe)
            # Conv outputs (M, N_pad). N0 → 128, N1 → 64 (real 32).
            self.AUD_SUB_OUT0 = self.allocate_tensor_dram(N0_pad * 128 * bpe)
            self.AUD_SUB_ACT0 = self.allocate_tensor_dram(N0_pad * 128 * bpe)
            self.AUD_SUB_OUT1 = self.allocate_tensor_dram(N1_pad * 64 * bpe)
            self.AUD_SUB_ACT1 = self.allocate_tensor_dram(N1_pad * 64 * bpe)
            # Identity matrices for activation-via-matmul (shape (N, N) where
            # N = post-LN feature dim, multiple of 64 required).
            self.AUD_SUB_ID_128 = self.allocate_tensor_dram(128 * 128 * bpe)
            self.dma_to_accelerator_memory(self.AUD_SUB_ID_128,
                torch.eye(128, dtype=torch.bfloat16).contiguous())
            # Compact (N1, 32) buffer for stage-1 LN input. LN needs N=32 to
            # average over the right feature count.
            self.AUD_SUB_OUT1_COMPACT = self.allocate_tensor_dram(N1_pad * 32 * bpe)
            # Flatten target for the final Linear input.
            # Final shape: (N1, 32*32=1024). N1 < L_pad; we pad with zeros so
            # the encoder seed only uses the first T_sub=N1 rows.
            self.AUD_SUB_FLAT = self.allocate_tensor_dram(N1_pad * 1024 * bpe)

        # DMA raw input (T_raw rows, padded with zeros to T_raw_pad).
        x_input = torch.zeros(T_raw_pad, 128, dtype=torch.bfloat16)
        x_input[:T_raw] = x
        self.dma_to_accelerator_memory(self.AUD_SUB_INPUT, x_input.contiguous())

        # Pre-zero R_combined. Padding rows (oh outside [0, H0)) stay zero so
        # the matmul produces zero rows there, which translate to zero patches
        # for the N0_pad-N0 padding positions consumed by conv0.
        self.dma_to_accelerator_memory(self.AUD_SUB_R_COMBINED,
            torch.zeros(H0_pad * K_g_s0, dtype=torch.bfloat16))

        # Emit the im2col segment (3 strided DMA gathers + 1 matmul).
        self._compile_and_run_single("aud_sub_im2col_s0",
            lambda: self._emit_aud_sub_im2col_s0(T_raw, H0, H0_pad))

        # ---- Stage 0 on FPGA: matmul -> LN -> ReLU ----------------------
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
            # layer_norm PBI left legacy: novel one-shot path not covered by the
            # vision all-saves validation; runs once so no bin-shrink benefit.
        # The bitstream clamp path turns negative inputs into +inf. GELU uses
        # the stable activation unit and is the smooth ReLU approximation used
        # by this FPGA implementation.
        self._compile_and_run_single("aud_sub_act0", lambda: self.matmat_mul_core(
            M=N0_pad, K=128, N=128,
            A_DRAM_ADDR=self.AUD_SUB_OUT0,
            B_DRAM_ADDR=self.AUD_SUB_ID_128,
            OUTPUT_DRAM_ADDR=self.AUD_SUB_ACT0,
            gelu_enable=True,
            gpr_M_reg=None))

        # ---- Stage 1 im2col on FPGA (Phase A2.2) ------------------------
        # 9*H1 strided DMA pairs (one per (kh, kw, oh1)) gather 128-channel
        # chunks from AUD_SUB_OUT0 and scatter them into the (kh*3+kw)-th
        # 128-column block of AUD_SUB_PATCHES1. Padding positions (oh1*2-1+kh
        # out of [0, H0) or ow1*2-1+kw out of [0, W0)) are left as zero from
        # the pre-zero step below — handles all border cases without branches
        # in the emitter.
        mask_s1 = mask[:, ::2] if mask is not None else None
        if mask_s1 is not None and not getattr(self, "_oneshot_mode", False):
            # Mask must be applied to AUD_SUB_OUT0 BEFORE im2col so masked
            # time rows don't bleed into patches1. Zero affected rows in DRAM.
            valid_h0 = mask_s1[0, :H0].to(torch.bfloat16)
            if (valid_h0 == 0).any():
                out0_local = self.dma_from_accelerator_memory(
                    self.AUD_SUB_ACT0, (N0_pad, 128)).cpu()
                for oh in range(H0):
                    if valid_h0[oh] == 0:
                        out0_local[oh * W0:(oh + 1) * W0].zero_()
                self.dma_to_accelerator_memory(self.AUD_SUB_ACT0, out0_local.contiguous())

        # Pre-zero patches1 so out-of-bounds (kh, kw) positions stay zero.
        self.dma_to_accelerator_memory(self.AUD_SUB_PATCHES1,
            torch.zeros(N1_pad * 1152, dtype=torch.bfloat16))
        self._compile_and_run_single("aud_sub_im2col_s1",
            lambda: self._emit_aud_sub_im2col_s1(H0, W0, H1, W1))

        # ---- Stage 1 on FPGA: matmul -> LN1 -> ReLU1 -> proj ------------
        # Phase A2.3: with the duplicate-channel trick (conv1 weight rows
        # 32-63 = rows 0-31), conv1 output is (N1_pad, 64) where cols 32-63
        # mirror cols 0-31. LN1 then runs as a standard N=64 LayerNorm (mean
        # and variance over the 64 duplicated values equal mean/variance over
        # the real 32 values), avoiding the N=32 alignment headache. ReLU1
        # runs as identity-matmul + clamp on (N1_pad, 64). Finally, because
        # the (N1_pad, 64) buffer's byte layout *is already* the (H1, W1*64)
        # flat-for-proj layout, the flatten step is a zero-op: we pass
        # AUD_SUB_OUT1 directly to the proj matmul with K=2048, and the proj
        # weight has zeros in the duplicate columns so the math reduces to
        # the original proj_orig @ flat_orig.
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
            # layer_norm PBI left legacy (see ln0 note above).
        self._compile_and_run_single("aud_sub_act1", lambda: self.matmat_mul_core(
            M=N1_pad, K=64, N=64,
            A_DRAM_ADDR=self.AUD_SUB_OUT1,
            B_DRAM_ADDR=self._aud_sub_id_64_addr,
            OUTPUT_DRAM_ADDR=self.AUD_SUB_ACT1,
            gelu_enable=True,
            gpr_M_reg=None))

        # Optional: apply layer-1 mask on FPGA-side (zero invalid time rows
        # after LN+ReLU). For the apex.wav test case mask is typically None;
        # if non-None, we re-zero affected rows in DRAM. Cheap (N1*64*bpe).
        mask_s2 = mask[:, ::4] if mask is not None else None
        if mask_s2 is not None and not getattr(self, "_oneshot_mode", False):
            valid_h = mask_s2[0, :H1].to(torch.bfloat16)
            if (valid_h == 0).any():
                out1_local = self.dma_from_accelerator_memory(
                    self.AUD_SUB_ACT1, (N1_pad, 64)).cpu()
                for oh in range(H1):
                    if valid_h[oh] == 0:
                        out1_local[oh * W1:(oh + 1) * W1].zero_()
                self.dma_to_accelerator_memory(self.AUD_SUB_ACT1, out1_local.contiguous())

        # ---- Final input_proj_linear on FPGA -----------------------------
        # AUD_SUB_OUT1 (N1_pad, 64) bytes IS already (H1_pad, W1*64=2048) bytes
        # row-major — no flatten DMA needed. proj weight is padded to K=2048
        # with zeros in duplicate cols so proj_new @ out1 = proj_orig @ first-32.
        # Output buffer: in oneshot fold (_oneshot_mode=True), proj writes
        # straight to AUD_IO_A so the encoder one-shot bin consumes it without
        # a host roundtrip. Otherwise (per-op compile path), output stays in
        # AUD_SUB_OUT0 scratch and the caller seeds AUD_IO_A separately.
        proj = self._aud_sub_proj_addrs
        H1_pad = ((H1 + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
        in_oneshot = getattr(self, "_oneshot_mode", False)
        proj_out_addr = self.AUD_IO_A if in_oneshot else self.AUD_SUB_OUT0
        self._compile_and_run_single("aud_sub_proj", lambda: self.matmat_mul_core(
            M=H1_pad, K=2 * 1024, N=1024,
            A_DRAM_ADDR=self.AUD_SUB_ACT1,  # reused as flat — see comment above
            B_DRAM_ADDR=proj["data"],
            OUTPUT_DRAM_ADDR=proj_out_addr,
            is_B_quantized=True, data_type=TYPE.IF4,
            SCALE_DRAM_ADDR=proj["scale"],
            gpr_M_reg=self._aud_mm_gpr(M=H1_pad)))

        if in_oneshot:
            # In oneshot mode, the FPGA hasn't executed yet — there's nothing
            # to read back. Subsample output lives in AUD_IO_A and the encoder
            # one-shot bin will read it as the input sequence.
            return None, mask_s2

        # Read back as the host-side sub_hidden contract: (B=1, T_sub, H=1024).
        hidden = self.dma_from_accelerator_memory(
            self.AUD_SUB_OUT0, (H1_pad, 1024)).cpu()[:H1].to(torch.bfloat16).unsqueeze(0)
        return hidden, mask_s2

    def _emit_aud_sub_im2col_s0(self, T_raw: int, H0: int, H0_pad: int) -> None:
        """FPGA emitter for stage-0 im2col (parakeet pattern).

        Builds R_combined(H0_pad, K_g=384) in AUD_SUB_R_COMBINED via 3 strided
        DMAs (one per kh in {0,1,2}), each gathering H0 rows of the raw mel
        input with stride=2 (matching the stride-2 conv0) and scattering them
        into the kh-th 128-column block of R_combined. Then matmuls R @ G_s0
        to produce the (N0_pad, 64) im2col patches at AUD_SUB_PATCHES0.

        Pre-conditions (the caller's responsibility):
          - AUD_SUB_INPUT holds (T_raw_pad, 128) of mel input (host DMA).
          - AUD_SUB_R_COMBINED was pre-zeroed (padding rows stay zero).
        """
        bpe_local = self.bytes_per_element
        W_in_local = 128
        stride = 2
        padding = 1
        row_bytes = W_in_local * bpe_local  # 256 — already 32-byte aligned for AXI
        K_g_s0 = self._aud_sub_K_g_s0       # 384
        N_g_s0 = self._aud_sub_N_g_s0       # 4096

        for kh in range(3):
            # Valid oh range where input row = oh*stride + kh - padding is in
            # [0, T_raw). Solve for oh: oh*2 - 1 + kh ∈ [0, T_raw).
            #   lower: oh >= (1 - kh) / 2  →  oh_start = ceil((padding-kh)/stride)
            #   upper: oh <  (T_raw + 1 - kh) / 2  →  oh_end = ceil((T_raw+padding-kh)/stride)
            oh_start = max(0, (padding - kh + stride - 1) // stride)
            oh_end = min(H0, (T_raw + padding - kh + stride - 1) // stride)
            n_rows = oh_end - oh_start
            if n_rows <= 0:
                continue
            first_input_row = oh_start * stride - padding + kh
            read_bytes = n_rows * row_bytes

            # Chunk to fit URAM. row_bytes (256) divides URAM_NEAR_FULL_SIZE so
            # chunks are integer multiples of row_bytes.
            max_read = (URAM_NEAR_FULL_SIZE // row_bytes) * row_bytes
            offset = 0
            while offset < read_bytes:
                chunk = min(read_bytes - offset, max_read)
                src_row = first_input_row + (offset // row_bytes) * stride
                oh_base = oh_start + offset // row_bytes

                # Strided read: gather every-other (stride=2) mel row into
                # URAM_A, packed contiguous.
                src = self.AUD_SUB_INPUT + src_row * row_bytes
                self.ue_memcpy_from_dram(
                    src, chunk, 0, URAM_START_ADDR,
                    URAM_SECTION.URAM_A.value,
                    stride_bytes_per_chunk=row_bytes,
                    stride_jump_bytes=stride * row_bytes)

                # Strided write: scatter contiguous URAM rows into the kh-th
                # 128-column block of R_combined. Each row writes 128 bf16
                # elements at offset (oh*K_g + kh*W_in)*bpe in DRAM.
                dst = (self.AUD_SUB_R_COMBINED
                       + oh_base * K_g_s0 * bpe_local
                       + kh * W_in_local * bpe_local)
                self.ue_memcpy_to_dram(
                    0, URAM_SECTION.URAM_A.value, URAM_START_ADDR,
                    dst, chunk,
                    stride_bytes_per_chunk=row_bytes,
                    stride_jump_bytes=K_g_s0 * bpe_local)
                offset += chunk

        # Matmul: R_combined(H0_pad, K_g) @ G_s0(N_g, K_g)^T → (H0_pad, N_g).
        # Byte layout is identical to (N0_pad, 64) patches0 view since
        # W_out_s0 = 64 (matmul stride per row = N_g = W_out_s0 * 64 = patch
        # row stride * W_out_s0).
        self.matmat_mul_core(
            M=H0_pad, K=K_g_s0, N=N_g_s0,
            A_DRAM_ADDR=self.AUD_SUB_R_COMBINED,
            B_DRAM_ADDR=self._aud_sub_G_s0_addr,
            OUTPUT_DRAM_ADDR=self.AUD_SUB_PATCHES0)

    def _emit_aud_sub_im2col_s1(self, H0: int, W0: int, H1: int, W1: int) -> None:
        """FPGA emitter for stage-1 im2col (multi-channel).

        Unlike stage-0 (single-channel, parakeet G-matrix), stage-1 has 128
        channels per spatial position, so we use direct strided-DMA gather +
        scatter rather than a permutation matmul. For each valid
        (kh, kw, oh1) triple, ONE strided-read pulls W1 chunks of 128 bf16
        elements (256 bytes each) from AUD_SUB_OUT0 (source stride = 2 spatial
        positions = 512 bytes); ONE strided-write scatters them into the
        (kh*3+kw)-th 128-column block of AUD_SUB_PATCHES1 (dest stride = one
        patches1 row = 1152*bpe = 2304 bytes).

        Out-of-bounds positions (oh1*2-1+kh outside [0, H0) or
        ow1*2-1+kw outside [0, W0)) are skipped — the caller pre-zeros
        AUD_SUB_PATCHES1 so those slots remain zero.

        Pre-conditions (the caller's responsibility):
          - AUD_SUB_ACT0 holds the (H0, W0, 128) activated stage-0 output
            in row-major (oh, ow, c) layout.
          - AUD_SUB_PATCHES1 was pre-zeroed.
        """
        bpe_local = self.bytes_per_element
        chunk_bytes = 128 * bpe_local       # 256 — one 128-channel pixel
        src_stride = 2 * chunk_bytes        # 512 — ow1-step in source
        dst_stride = 1152 * bpe_local       # 2304 — row-step in patches1
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
                    in_row = oh1 * 2 - 1 + kh           # 0 <= in_row < H0
                    in_col_start = ow1_start * 2 - 1 + kw
                    src = (self.AUD_SUB_ACT0
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

    def _emit_aud_embed_project_chain(self) -> None:
        """Phase B embed+projector ISA emitter (no DMA-back, no host work).

        Designed to be folded into the encoder one-shot capture: when called
        with ``self._oneshot_mode=True``, the three sub-ops emit inline into
        the caller's open capture instead of issuing separate per-op triggers.
        Reads from AUD_IO_A (encoder output), writes to AUD_FEATURES_FINAL.
        """
        H = self.AUD_H
        OUT_DIM = self.AUD_OUT_DIM
        L_pad = self._aud_L_pad

        # Step 1: encoder_out @ W_op.T + b_op  ->  AUD_FEATURES_MID
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
                # bias-matmul PBI left legacy: C_DRAM bias path not covered by the
                # vision all-saves validation; runs once, produces audio_features.

        # Step 2: RMSNorm with all-ones gamma (HF Gemma4MultimodalEmbedder uses
        # embedding_pre_projection_norm with with_scale=False, so ones-gamma is
        # mathematically equivalent). LEGACY (no gpr_M_reg) — PBI on this once-run
        # op silently diverges in the one-shot bin (passes per-sub-op compare but
        # corrupts audio features end-to-end → e2b "I don't hear any audio"
        # degenerate decode, since the smaller LM is more sensitive to feature
        # scale errors than e4b). Mirrors gemma4_e4b which never had PBI here.
        self._compile_and_run_single("aud_embed_rmsnorm",
            lambda: self.rms_norm_core_dram(
                M=L_pad, N=OUT_DIM,
                A_DRAM_ADDR=self.AUD_FEATURES_MID,
                OUTPUT_DRAM_ADDR=self.AUD_FEATURES_MID,
                GAMMA_DRAM_ADDR=self.AUD_EMB_ONES_GAMMA))

        # Step 3: x @ W_em.T  ->  AUD_FEATURES_FINAL. LEGACY — see Step 2 rationale.
        em = self._aud_embedder_proj_addrs
        self._compile_and_run_single("aud_embed_emb_proj",
            lambda: self.matmat_mul_core(
                M=L_pad, K=OUT_DIM, N=OUT_DIM,
                A_DRAM_ADDR=self.AUD_FEATURES_MID,
                B_DRAM_ADDR=em["data"],
                OUTPUT_DRAM_ADDR=self.AUD_FEATURES_FINAL,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=em["scale"]))

    def _aud_ffn_mm(self, kind: str, w_entry: dict, *, cache=None, cache_key: str = None) -> None:
        """One conformer-FFN matmul. kind='w1' (H→FF) or 'w2' (FF→H). Per-layer
        weights → emitted inline per call (sharing these would need a register
        weight-address in matmat_mul_core, a kernel change; the §7 win comes from
        the layer-invariant SiLU, see _aud_ffn_silu). Both macaron halves of all
        layers reuse the same two shapes (identical A/OUT scratch)."""
        H, FF, L_pad = self.AUD_H, self.AUD_FFN, self._aud_L_pad
        if kind == "w1":
            K, N, A_IN, OUT = H, FF, self.AUD_NORM_OUT, self.AUD_FFN_MID
        else:
            K, N, A_IN, OUT = FF, H, self.AUD_SILU_OUT, self.AUD_FFN_OUT
        self._compile_and_run_single(cache_key or f"aud_ffn_{kind}", lambda: self.matmat_mul_core(
            M=L_pad, K=K, N=N, A_DRAM_ADDR=A_IN,
            B_DRAM_ADDR=w_entry["data"], OUTPUT_DRAM_ADDR=OUT,
            is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=w_entry["scale"],
            gpr_M_reg=self._aud_mm_gpr(M=L_pad)),
            cache=cache, cache_key=cache_key)

    def _aud_ffn_silu(self, **_unused) -> None:
        self._emit_audio_core(lambda: _aud_silu(
            self, self._aud_L_pad, self.AUD_FFN,
            self.AUD_FFN_MID, self.AUD_SILU_OUT, self.AUD_IDENTITY_FF))

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
        self._aud_ffn_mm("w1", w["FF1_W1"], cache=cache, cache_key=_ck("ff1_w1"))
        self._aud_clip_dram(self.AUD_FFN_MID, (L_pad, FF),
                              cr["FF1_W1"]["out_min"], cr["FF1_W1"]["out_max"])

        # 3. SiLU on (L_pad, FF) — §7-FFN shares ONE silu body across all layers.
        self._aud_ffn_silu(cache=cache, cache_key=_ck("ff1_silu"))

        # 4. ffw_layer_2 (clippable, IF4 (block=64)). Reads from SiLU output.
        self._aud_clip_dram(self.AUD_SILU_OUT, (L_pad, FF),
                              cr["FF1_W2"]["in_min"], cr["FF1_W2"]["in_max"])
        self._aud_ffn_mm("w2", w["FF1_W2"], cache=cache, cache_key=_ck("ff1_w2"))
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

        # Phase 2B.b: Q / K scaling on FPGA.
        # Q *= q_scale * softplus(per_dim_scale[d])  — eltwise_mul by Q_SCALE_TILE
        # K *= k_scale                                — broadcast_mul by AUD_K_SCALE
        self._compile_and_run_single("aud_attn_q_scale",
            lambda: self._emit_aud_q_scale_fpga(layer_idx),
            cache=cache, cache_key=_ck("attn_q_scale"))
        self._compile_and_run_single("aud_attn_k_scale",
            lambda: self._emit_aud_k_scale_fpga(),
            cache=cache, cache_key=_ck("attn_k_scale"))

        # Phase 2B.c: chunked attention chain (Q@K^T, rel-shift, softcap-tanh,
        # mask, softmax, attn@V) writes pre-o_proj output to AUD_ATTN_OUT.
        self._compile_and_run_single(
            "aud_attn_fpga_chain",
            lambda: self._emit_aud_attn_fpga_chain(layer_idx))
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

        # 4. depthwise_conv1d (FPGA 4-tap shifted-eltwise).
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

    def _aud_dw_conv1d_dispatch(self, layer_idx: int, *,
                                  in_addr: int, out_addr: int,
                                  cache, cache_key: str) -> None:
        """Phase 2A depthwise conv1d on FPGA (4-tap shifted-eltwise)."""
        self._compile_and_run_single(
            f"aud_L{layer_idx}_conv_dw",
            lambda: self._emit_aud_dw_conv1d_fpga(layer_idx,
                                                   in_addr=in_addr,
                                                   out_addr=out_addr),
            cache=cache, cache_key=cache_key)

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

    def _emit_aud_attn_build_vctx_t(self, layer_idx: int) -> None:
        """Phase 2B.c step 5 prep: build V_PADDED, V_CTX_BLOCKS, V_CTX_T_BLOCKS.

        V_PADDED         := zero-padded V (mirrors K_PADDED).
        V_CTX_BLOCKS     := per-block window of V_PADDED, (num_blocks, ctx_pad, H).
        V_CTX_T_BLOCKS   := per-(block, head) ACTUAL transpose of V_BLOCK_HEAD,
                            shape (num_blocks, NH, HD, ctx_pad). Transposition uses
                            matmul-with-AUD_IDENTITY_HD trick since the FPGA strided
                            DMA can't do per-column reads (stride 2 bytes < 32 byte
                            AXI alignment). The matmul A=I (HD, HD) @ B=V_BLOCK_HEAD
                            (in (N=ctx_pad, K=HD) layout) yields output[m, n]
                            = sum_k delta(m, k) * V_BLOCK_HEAD[n, k] = V_BLOCK_HEAD[n, m],
                            which is V_BLOCK_HEAD transposed.
        """
        from audio_primitives import (copy_dram_to_dram_chunked, URAM_A_BASE)
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

        copy_dram_to_dram_chunked(self, self.AUD_V,
                                   self.AUD_V_PADDED + max_past * H * bpe,
                                   L_pad * H, row_n=H)
        for b in range(num_blocks):
            copy_dram_to_dram_chunked(self,
                self.AUD_V_PADDED + b * chunk * H * bpe,
                self.AUD_V_CTX_BLOCKS + b * ctx_pad * H * bpe,
                ctx_size * H, row_n=H)

        # Per-(block, head): strided extract head slice into V_BLOCK_HEAD, then
        # matmul-transpose into V_CTX_T_BLOCKS.
        for b in range(num_blocks):
            for h in range(NH):
                src = self.AUD_V_CTX_BLOCKS + (b * ctx_pad * H + h * HD) * bpe
                self.accelerator_memory_to_sram(
                    accelerator_dram_address=src,
                    sram_address=URAM_A_BASE,
                    element_size=ctx_pad * HD,
                    stride_bytes_per_chunk=HD * bpe,
                    stride_jump_bytes=H * bpe)
                self.sram_to_accelerator_memory(URAM_A_BASE,
                    self.AUD_V_BLOCK_HEAD, ctx_pad * HD)
                # matmul I (HD, HD) @ V_BLOCK_HEAD^T (HD, ctx_pad) = V transposed.
                dst = self.AUD_V_CTX_T_BLOCKS + (b * NH + h) * HD * ctx_pad * bpe
                self.matmat_mul_core(M=HD, K=HD, N=ctx_pad,
                    A_DRAM_ADDR=self.AUD_IDENTITY_HD,
                    B_DRAM_ADDR=self.AUD_V_BLOCK_HEAD,
                    OUTPUT_DRAM_ADDR=dst)

    def _emit_aud_attn_rel_k_proj(self, layer_idx: int) -> None:
        """Part (a) of matrix_bd: REL_K_OUT = POS_EMB_PADDED @ REL_K_PROJ[layer]
        → fixed AUD_REL_K_OUT. This is the ONLY per-layer step of the whole
        attention chain (uses the layer's REL_K_PROJ weight); §7 hoists it to each
        call site so the rest of the chain (which reads only fixed buffers) can be
        shared across layers as one subroutine."""
        H = self.AUD_H
        num_pos_pad = self._aud_num_pos_pad
        rk = self._aud_weight_addrs[layer_idx]["REL_K_PROJ"]
        self.matmat_mul_core(M=num_pos_pad, K=H, N=H,
            A_DRAM_ADDR=self.AUD_POS_EMB_PADDED,
            B_DRAM_ADDR=rk["data"],
            OUTPUT_DRAM_ADDR=self.AUD_REL_K_OUT,
            is_B_quantized=True, data_type=TYPE.IF4,
            SCALE_DRAM_ADDR=rk["scale"])

    def _emit_aud_attn_matrix_bd_unshifted(self, layer_idx: int, *, skip_rel_k_proj: bool = False) -> None:
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

        # (a) REL_K_OUT = POS_EMB_PADDED @ REL_K_PROJ  (num_pos_pad, H, H).
        # §7: hoisted to the call site when skip_rel_k_proj (shared-chain build).
        if not skip_rel_k_proj:
            self._emit_aud_attn_rel_k_proj(layer_idx)

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

    def _emit_aud_softcap_tanh_dram(self, addr: int, M: int, N: int,
                                     soft_cap: float, identity_addr: int) -> None:
        """In-place soft_cap * tanh(x / soft_cap) on a (M, N) bf16 DRAM tensor.

        Decomposes via tanh(y) = 2*sigmoid(2y) - 1:
            out = 2*soft_cap * sigmoid(2x / soft_cap) - soft_cap
        Steps: broadcast_mul(2/sc) -> sigmoid via identity matmul -> broadcast_mul(2*sc)
               -> broadcast_add(-sc).
        ``identity_addr`` must point to an (N, N) bf16 identity matrix.
        """
        from audio_primitives import URAM_A_BASE, _row_chunk
        bpe = self.bytes_per_element
        a = 2.0 / soft_cap
        b = 2.0 * soft_cap
        row_bytes = N * bpe
        M_chunk = _row_chunk(M, N, divisor=2)

        for m_start in range(0, M, M_chunk):
            m_take = min(M_chunk, M - m_start)
            n = m_take * N
            chunk = addr + m_start * row_bytes
            self.accelerator_memory_to_sram(chunk, URAM_A_BASE, n)
            self.broadcast_mul(scalar=a,
                                sram_start_addr=URAM_A_BASE,
                                sram_wb_addr=URAM_A_BASE,
                                element_size=n)
            self.sram_to_accelerator_memory(URAM_A_BASE, chunk, n)

        # In-place sigmoid via identity matmul over all M rows.
        self.matmat_mul_core(M=M, K=N, N=N,
            A_DRAM_ADDR=addr, B_DRAM_ADDR=identity_addr,
            OUTPUT_DRAM_ADDR=addr, sigmoid_enable=True)

        for m_start in range(0, M, M_chunk):
            m_take = min(M_chunk, M - m_start)
            n = m_take * N
            chunk = addr + m_start * row_bytes
            self.accelerator_memory_to_sram(chunk, URAM_A_BASE, n)
            self.broadcast_mul(scalar=b,
                                sram_start_addr=URAM_A_BASE,
                                sram_wb_addr=URAM_A_BASE,
                                element_size=n)
            self.broadcast_add(scalar=-soft_cap,
                                sram_start_addr=URAM_A_BASE,
                                sram_wb_addr=URAM_A_BASE,
                                element_size=n)
            self.sram_to_accelerator_memory(URAM_A_BASE, chunk, n)

    def _emit_aud_attn_logits_softcap_mask_softmax(self, layer_idx: int) -> None:
        """Phase 2B.c steps 4-6: produce attn_w in AUD_MATRIX_AC by:
            (i)   AUD_MATRIX_AC += AUD_MATRIX_BD_SHIFTED      (eltwise_add)
            (ii)  AUD_MATRIX_AC = soft_cap * tanh(AUD_MATRIX_AC / soft_cap)
            (iii) AUD_MATRIX_AC += AUD_MASK_ADDEND            (-1e9 outside valid)
            (iv)  AUD_MATRIX_AC = softmax(AUD_MATRIX_AC, dim=-1)
        Operates on the flat (num_blocks*NH*chunk_pad, ctx_pad) view of the tile.
        """
        from audio_primitives import eltwise_add_core_dram
        NH = self.AUD_HEADS
        chunk_pad = self._aud_chunk_pad
        ctx_pad = self._aud_ctx_pad
        num_blocks = self._aud_num_blocks
        M_total = num_blocks * NH * chunk_pad

        # (i) AC += BD_SHIFTED
        eltwise_add_core_dram(self, M=M_total, N=ctx_pad,
            A_DRAM_ADDR=self.AUD_MATRIX_AC,
            B_DRAM_ADDR=self.AUD_MATRIX_BD_SHIFTED,
            OUTPUT_DRAM_ADDR=self.AUD_MATRIX_AC)
        # (ii) softcap in place
        self._emit_aud_softcap_tanh_dram(
            self.AUD_MATRIX_AC, M=M_total, N=ctx_pad,
            soft_cap=float(self.AUD_SOFT_CAP),
            identity_addr=self.AUD_IDENTITY_CTX)
        # (iii) AC += MASK_ADDEND (additive bias: 0 for valid, -1e9 for masked)
        eltwise_add_core_dram(self, M=M_total, N=ctx_pad,
            A_DRAM_ADDR=self.AUD_MATRIX_AC,
            B_DRAM_ADDR=self.AUD_MASK_ADDEND,
            OUTPUT_DRAM_ADDR=self.AUD_MATRIX_AC)
        # (iv) softmax along last dim via matmul-with-identity
        self.matmat_mul_core(M=M_total, K=ctx_pad, N=ctx_pad,
            A_DRAM_ADDR=self.AUD_MATRIX_AC,
            B_DRAM_ADDR=self.AUD_IDENTITY_CTX,
            OUTPUT_DRAM_ADDR=self.AUD_MATRIX_AC,
            softmax_enable=True)

    def _emit_aud_attn_fpga_chain(self, layer_idx: int, *, skip_rel_k_proj: bool = False) -> None:
        """Phase 2B.c full FPGA chunked-attention chain. Called inline (no
        per-sub-op capture), so it works inside either an outer
        ``_compile_and_run_single`` capture OR the encoder one-shot capture
        with ``_oneshot_mode=True``.

        Assumes Q/K/V are already populated in AUD_Q/K/V (post-QK scaling)
        and writes the pre-o_proj attention output to AUD_ATTN_OUT[:T, :H].
        Subsequent o_proj clamps+matmul transforms it in place.
        """
        self._emit_aud_attn_build_kctx_t(layer_idx)
        self._emit_aud_attn_matrix_ac(layer_idx)
        self._emit_aud_attn_matrix_bd_unshifted(layer_idx, skip_rel_k_proj=skip_rel_k_proj)
        self._emit_aud_attn_rel_shift(layer_idx)
        self._emit_aud_attn_build_vctx_t(layer_idx)
        self._emit_aud_attn_logits_softcap_mask_softmax(layer_idx)
        self._emit_aud_attn_value_and_scatter(layer_idx)

    def _emit_aud_attn_value_and_scatter(self, layer_idx: int) -> None:
        """Phase 2B.c step 7: per-(block, head) attn @ V_ctx, then scatter to
        AUD_ATTN_OUT.

        Per (b, h):
            attn_tile = AUD_MATRIX_AC[b, h]    (chunk_pad, ctx_pad)
            V_T       = AUD_V_CTX_T_BLOCKS[b, h]  (HD, ctx_pad)  -- B in (N=HD, K=ctx_pad)
            tmp = attn_tile @ V_T^T            (chunk_pad, HD)  -- attn @ V_ctx
            scatter tmp[:valid_rows] to AUD_ATTN_OUT[b*chunk:b*chunk+valid_rows,
                                                     h*HD:(h+1)*HD]
        where valid_rows = min(chunk, L_pad - b*chunk).
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
                attn = self.AUD_MATRIX_AC + (b * NH + h) * chunk_pad * ctx_pad * bpe
                V_T  = self.AUD_V_CTX_T_BLOCKS + (b * NH + h) * HD * ctx_pad * bpe
                self.matmat_mul_core(M=chunk_pad, K=ctx_pad, N=HD,
                    A_DRAM_ADDR=attn,
                    B_DRAM_ADDR=V_T,
                    OUTPUT_DRAM_ADDR=self.AUD_ATTN_OUT_BH)
                if valid_rows <= 0:
                    continue
                # Strided scatter top valid_rows rows of (chunk_pad, HD) into
                # AUD_ATTN_OUT[b*chunk:b*chunk+valid_rows, h*HD:(h+1)*HD]:
                # row stride at dst = H*bpe (full row of AUD_ATTN_OUT),
                # row stride at src = HD*bpe (tile is contiguous).
                self.accelerator_memory_to_sram(
                    accelerator_dram_address=self.AUD_ATTN_OUT_BH,
                    sram_address=URAM_A_BASE,
                    element_size=valid_rows * HD)
                dst = self.AUD_ATTN_OUT + (b * chunk * H + h * HD) * bpe
                self.sram_to_accelerator_memory(
                    sram_address=URAM_A_BASE,
                    accelerator_dram_address=dst,
                    element_size=valid_rows * HD,
                    stride_bytes_per_chunk=HD * bpe,
                    stride_jump_bytes=H * bpe)

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
        self._aud_ffn_mm("w1", w[f"{prefix}_W1"], cache=cache, cache_key=_ck(f"{plow}_w1"))
        self._aud_clip_dram(self.AUD_FFN_MID, (L_pad, FF),
                              cr[f"{prefix}_W1"]["out_min"], cr[f"{prefix}_W1"]["out_max"])

        self._aud_ffn_silu(cache=cache, cache_key=_ck(f"{plow}_silu"))

        self._aud_clip_dram(self.AUD_SILU_OUT, (L_pad, FF),
                              cr[f"{prefix}_W2"]["in_min"], cr[f"{prefix}_W2"]["in_max"])
        self._aud_ffn_mm("w2", w[f"{prefix}_W2"], cache=cache, cache_key=_ck(f"{plow}_w2"))
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

    def compile_audio_fpga(self, input_features: torch.Tensor,
                           input_features_mask: torch.Tensor | None = None) -> None:
        """Capture subsampling, 12 Conformer layers, and projection as one
        straight-line program in the dedicated audio ISA region."""
        meta, _ = self._get_program_section("audio")
        tensor_base = self.AUD_IO_A
        if (meta is not None
                and int(meta.get("tensor_base", "0"), 16) == tensor_base
                and meta.get("program_version") == AUDIO_PROGRAM_VERSION):
            print("  [Audio] reusing existing audio instruction section")
            return

        saved_base = self._program_dram_base
        saved_next = self._next_program_dram_addr
        self._program_dram_base = AUDIO_ISA_BASE
        self._next_program_dram_addr = AUDIO_ISA_BASE
        self.reset_program_dram_addr()
        base_addr = self.get_program_dram_addr()
        self.clear_capture_buffer()
        self.start_capture()
        self._oneshot_mode = True
        previous_silent = self._set_silent(True)
        started = time.perf_counter()
        try:
            self.audio_subsample_fpga(input_features, input_features_mask)
            for layer_idx in range(self.AUD_LAYERS):
                self.compile_audio_layer_ffn1(layer_idx)
                self.compile_audio_layer_attn(layer_idx)
                self.compile_audio_layer_conv(layer_idx)
                self.compile_audio_layer_ffn2(layer_idx)
                self.compile_audio_layer_norm_out(layer_idx)
            self._emit_aud_embed_project_chain()
            self.generate_instruction_halt()
            self.stop_capture()
            program = b"".join(inst.get_bytes() for inst in self.capture_buffer)
            self.clear_capture_buffer()
        finally:
            self._oneshot_mode = False
            self._set_silent(previous_silent)
            self._program_dram_base = saved_base
            self._next_program_dram_addr = saved_next

        self._store_program_section("audio", base_addr, program, {
            "num_layers": self.AUD_LAYERS,
            "num_frames": self._aud_num_frames,
            "padded_frames": self._aud_L_pad,
            "tensor_base": f"0x{tensor_base:X}",
            "program_version": AUDIO_PROGRAM_VERSION,
            "includes_subsample": True,
        })
        print(f"  [Audio] stored {len(program)/1024/1024:.1f} MB FPGA section "
              f"at 0x{base_addr:X} in {time.perf_counter()-started:.1f}s")

    def run_audio_fpga(self, input_features: torch.Tensor,
                       input_features_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Upload the runtime mel tensor, execute the compiled audio section,
        and return the fixed number of LM soft-token embeddings."""
        # audio_subsample_fpga performs the deterministic input/scratch uploads.
        saved_buffer, saved_on = self.capture_buffer, self.is_capture_on
        self.capture_buffer = None
        self.is_capture_on = False
        self._oneshot_mode = True
        try:
            self.audio_subsample_fpga(input_features, input_features_mask)
        finally:
            self._oneshot_mode = False
            self.capture_buffer = saved_buffer
            self.is_capture_on = saved_on
        self.dma_to_accelerator_memory(
            self.AUD_IO_A,
            torch.zeros(self._aud_L_pad * self.AUD_H, dtype=torch.bfloat16))

        meta, program = self._get_program_section("audio")
        if meta is None:
            raise FileNotFoundError("audio instruction section is not compiled")
        program_addr = int(meta["dram_base"], 16)
        self._next_program_dram_addr = program_addr
        self.dma_write(DMA_DEVICE_H2C, program_addr, program, len(program))
        self.allocate_program_dram(len(program))
        self.start_execute_from_dram(program_addr)
        self.wait_queue(600.0)
        return self.dma_from_accelerator_memory(
            self.AUD_FEATURES_FINAL,
            (self._aud_L_pad, self.AUD_OUT_DIM)).cpu()[:self._aud_num_frames]

    def _run_audio_encoder_fpga(self, audio_path: str, prompt: str):
        """LM-facing audio interface: preprocess, compile/run FPGA, and return
        ``(soft_tokens, token_ids, mm_token_types)``."""
        from transformers import AutoProcessor
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError("soundfile is required for audio input") from exc

        self._ensure_model_loaded()
        processor = AutoProcessor.from_pretrained(self.hf_model_dir)
        samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if samples.ndim > 1:
            samples = samples.mean(axis=-1)
        target_rate = processor.feature_extractor.sampling_rate
        if sample_rate != target_rate:
            wave = torch.from_numpy(samples).float()[None, None]
            samples = F.interpolate(
                wave, size=int(samples.shape[0] * target_rate / sample_rate),
                mode="linear", align_corners=False).squeeze().numpy()

        conversation = [{"role": "user", "content": [
            {"type": "audio"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text], audio=[samples], sampling_rate=target_rate,
                           return_tensors="pt")
        input_features = inputs["input_features"]
        input_mask = inputs.get("input_features_mask")
        token_ids = inputs["input_ids"][0].tolist()
        mm_types = inputs["mm_token_type_ids"][0].tolist()

        tensor_cursor = self._tensor_dram_addr
        program_cursor = self._next_program_dram_addr
        program_base = self._program_dram_base
        self.audio_weight_init()
        raw_frames = int(input_features.shape[1])
        h0 = (raw_frames + 1) // 2
        soft_frames = (h0 + 1) // 2
        self.audio_tensor_init(soft_frames)
        self.compile_audio_fpga(input_features, input_mask)
        features = self.run_audio_fpga(input_features, input_mask)
        if not torch.isfinite(features).all():
            raise RuntimeError("FPGA audio encoder produced non-finite features")

        slots = sum(kind == 3 for kind in mm_types)
        if features.shape[0] < slots:
            features = torch.cat((features, torch.zeros(
                slots - features.shape[0], features.shape[1],
                dtype=features.dtype)), dim=0)
        features = features[:slots].to(torch.bfloat16)
        self._tensor_dram_addr = tensor_cursor
        self._next_program_dram_addr = program_cursor
        self._program_dram_base = program_base
        return features, token_ids, mm_types

    def _aud_copy_buf(self, label: str, src: int, dst: int, n_elems: int,
                       row_n: int | None = None) -> None:
        """Copy n_elems bf16 elements from src DRAM to dst DRAM via SRAM,
        chunked so URAM_A doesn't overflow on long buffers.

        row_n: optional row width for clean row-aligned chunking.
        """
        def _fn():
            _aud_copy_chunked(self, src, dst, n_elems, row_n=row_n)
        self._compile_and_run_single(label, _fn)
