#!/usr/bin/env python3
"""LM and optional vision numeric checks for the REFACTOR engine.

Builds full-precision "hf" and IF4-quantized "hostref" vision references and
runs the FPGA side through Gemma4_UnifiedEngine. It SNR-compares the
self._vis_ckpt readbacks at two stages:

  encoder_out   -> reference B  (after all 16 encoder layers, pre-pool)
  image_features-> reference C  (after pooler + embedding projection)

Then checks LM prefill final hidden states and the first decoder token's final
hidden state, output norm, logits and argmax. The LM HF reference consumes the
exact FPGA-produced image soft tokens, so vision error does not leak into LM.

Interpretation (same as numeric.py):
  * FPGA vs HOSTREF (both IF4) should be HIGH SNR — a low value is a real kernel
    bug, not quantization.
  * FPGA vs HF is lower by the IF4 quantization loss (ground-truth gap).

Requires params.bin regenerated with the [LM | vision | host] layout.

Usage:
  python models/gemma4_e2b/gemma4_e2b_numeric.py            # LM-only, default prompt
  python models/gemma4_e2b/gemma4_e2b_numeric.py --image    # yosemite + VLM
  python models/gemma4_e2b/gemma4_e2b_numeric.py --image people.jpg
  python models/gemma4_e2b/gemma4_e2b_numeric.py --dev xdma0 --cycle 5.042
"""
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)                                   # gemma4_e2b_test / _numeric
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))  # user_dma_core (repo root)

import user_dma_core
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor
from user_dma_core import calculate_snr, set_dma_device
import gemma4_e2b_test as g4r
import gemma4_e2b_lm as g4_lm_module
import gemma4_e2b_vision as g4_vision_module
import quant_lib


def _align(ref, res):
    """Flatten batch dimensions and align host and FPGA rows."""
    ref = ref.detach().float()
    res = res.detach().float()
    if ref.dim() == 3 and ref.shape[0] == 1:
        ref = ref.squeeze(0)
    if res.dim() == 3 and res.shape[0] == 1:
        res = res.squeeze(0)
    if ref.shape != res.shape:
        ref = ref.reshape(-1, ref.shape[-1])
        res = res.reshape(-1, res.shape[-1])
        rows = min(ref.shape[0], res.shape[0])
        ref, res = ref[:rows], res[:rows]
    return ref, res


def report(name, ref, res, row_mask=None):
    """Print SNR, relative L2 error, and maximum absolute error."""
    ref, res = _align(ref, res)
    if row_mask is not None and row_mask.numel() == ref.shape[0]:
        ref, res = ref[row_mask], res[row_mask]
    finite = torch.isfinite(ref).all(dim=-1) & torch.isfinite(res).all(dim=-1)
    ref, res = ref[finite], res[finite]
    if ref.numel() == 0:
        print(f"  [numeric] {name:20s} no finite overlap")
        return float("-inf")
    snr = float(calculate_snr(ref, res))
    error = float(torch.linalg.vector_norm(res - ref))
    signal = float(torch.linalg.vector_norm(ref))
    relative = error / signal if signal > 0 else float("inf")
    max_delta = float((res - ref).abs().max())
    print(f"  [numeric] {name:20s} SNR={snr:7.2f} dB  rel_L2={relative:.4g}  "
          f"max|Δ|={max_delta:.4g}  rows={ref.shape[0]} shape={tuple(res.shape)}")
    return snr


def quantize_vision_tower_(vision_tower, precision, block=64):
    """Round-trip every vision Linear weight through the FPGA precision."""
    count = 0
    with torch.no_grad():
        for module in vision_tower.modules():
            weight = getattr(module, "weight", None)
            if (isinstance(module, nn.Linear) and weight is not None
                    and weight.dim() == 2 and weight.shape[1] % block == 0):
                quant_input = weight.detach().cpu().to(torch.bfloat16)
                n, k = quant_input.shape
                data, scale = quant_lib.quantize(precision, quant_input, block)
                dequantized = quant_lib.dequant(
                    precision, data, scale, n, k, block)
                weight.data.copy_(dequantized.to(weight.dtype).to(weight.device))
                count += 1
    return count


def _output_tensor(output):
    """Extract a tensor from HF tensor/tuple/ModelOutput hook results."""
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def build_vision_references(ue, image_path, prompt):
    """Build only B/C references; no patch-embed or layer-0 checkpoints."""
    hf_model, model_dir = g4r._ensure_hf_model(ue.script_dir, ue._cfg)
    hf_model.eval()
    processor = AutoProcessor.from_pretrained(model_dir)
    image = Image.open(image_path).convert("RGB").resize(
        g4r.VISION_CANONICAL_SIZE, Image.BICUBIC)
    conversation = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[[image]], return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    position_ids = inputs["image_position_ids"]
    padding = (position_ids == -1).all(dim=-1).squeeze(0)
    vision_tower = hf_model.model.vision_tower

    def _capture():
        checkpoint = {}
        hook = vision_tower.encoder.register_forward_hook(
            lambda _m, _i, output: checkpoint.__setitem__(
                "B", _output_tensor(output).detach().float().cpu()))
        try:
            with torch.no_grad():
                output = hf_model.model.get_image_features(
                    pixel_values=pixel_values.to(torch.bfloat16),
                    image_position_ids=position_ids)
        finally:
            hook.remove()
        checkpoint["C"] = getattr(
            output, "pooler_output", output).detach().float().cpu()
        return checkpoint

    print("  [numeric] vision reference 1/2: full-precision HF")
    hf_ref = _capture()
    count = quantize_vision_tower_(
        vision_tower, g4r.VISION_QUANT_PRECISION)
    print(f"  [numeric] vision reference 2/2: {count} IF4 Linear weights")
    host_ref = _capture()
    return {"hf": hf_ref, "hostref": host_ref}, {"padding": padding}


def build_hf_lm_reference(ue, inputs_embeds, prepared_per_layer_inputs):
    """Run unmodified HF LM on the exact external tensors consumed by FPGA."""
    hf_model, _ = g4r._ensure_hf_model(ue.script_dir, ue._cfg)
    hf_model.eval()
    language_model = hf_model.model.language_model
    inputs_embeds = inputs_embeds.unsqueeze(0).to(next(hf_model.parameters()).dtype)
    prepared_per_layer_inputs = prepared_per_layer_inputs.unsqueeze(0).to(
        inputs_embeds.dtype)

    layer_out = {}
    def _capture_last_layer(_module, _args, output):
        layer_out["hidden"] = _output_tensor(output).detach().float().cpu()

    hook = hf_model.model.language_model.layers[-1].register_forward_hook(
        _capture_last_layer)
    original_project = language_model.project_per_layer_inputs
    language_model.project_per_layer_inputs = (
        lambda _inputs_embeds, per_layer_inputs: per_layer_inputs)
    try:
        with torch.no_grad():
            output = language_model(
                inputs_embeds=inputs_embeds,
                per_layer_inputs=prepared_per_layer_inputs,
                attention_mask=torch.ones(inputs_embeds.shape[:2], dtype=torch.long),
                use_cache=False,
                return_dict=True)
    finally:
        language_model.project_per_layer_inputs = original_project
        hook.remove()

    hidden = layer_out["hidden"]
    norm = output.last_hidden_state.detach().float().cpu()
    logits = hf_model.lm_head(output.last_hidden_state)
    softcap = hf_model.config.text_config.final_logit_softcapping
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    return {
        "hidden": hidden,
        "norm": norm,
        "logits": logits.detach().float().cpu(),
    }


def _bf16_region(ue, key, shape, layer_idx=None):
    """Read one exact BF16 tensor from params.bin (including stored gamma offset)."""
    off = ue.weight_defs[key]
    if layer_idx is not None:
        off += layer_idx * ue.weight_defs["LAYER_WEIGHT_SIZE"]
    count = 1
    for dim in shape:
        count *= dim
    return torch.frombuffer(
        ue.weight_bin, dtype=torch.bfloat16, count=count, offset=off).reshape(shape)


def _if4_region(ue, scale_key, data_key, n, k, layer_idx=None):
    """Dequantize the exact adaptive-IF4 bytes consumed by an FPGA projection."""
    layer_off = (layer_idx or 0) * ue.weight_defs["LAYER_WEIGHT_SIZE"]
    scale_off = ue.weight_defs[scale_key] + layer_off
    data_off = ue.weight_defs[data_key] + layer_off
    scale_size = n * (k // 64) * 2
    data_size = n * k // 2
    scales = bytes(ue.weight_bin[scale_off:scale_off + scale_size])
    data = bytes(ue.weight_bin[data_off:data_off + data_size])
    return quant_lib.dequant("if4", data, scales, n, k, block_size=64)


def _hw_rms(x, gamma=None):
    """FPGA RMS convention, including its SRAM BF16 stage boundaries."""
    x = x.to(torch.bfloat16)
    # rms_norm_core broadcasts the LALU's reciprocal-RMS result into a BF16
    # SRAM vector, then performs gamma as a second BF16 elementwise operation.
    inv_rms = torch.rsqrt(
        torch.mean(x.float().square(), dim=-1, keepdim=True)).to(torch.bfloat16)
    out = (x * inv_rms).to(torch.bfloat16)
    if gamma is not None:
        out = (out * gamma.to(torch.bfloat16)).to(torch.bfloat16)
    return out


def _hw_linear(x, weight, gelu=False):
    """Matmul with one BF16 writeback; GELU stays fused before writeback."""
    out = x.to(torch.bfloat16).float() @ weight.to(torch.bfloat16).float().T
    if gelu:
        # LALU_ACT_GELU_B=0xbfd9 encodes -1.6953125; the fused datapath
        # computes x*sigmoid(1.6953125*x) directly from the matmul accumulator.
        out = out * torch.sigmoid(1.6953125 * out)
    return out.to(torch.bfloat16)


def _hw_add(a, b):
    """One FPGA BF16 elementwise-add writeback."""
    return (a.to(torch.bfloat16) + b.to(torch.bfloat16)).to(torch.bfloat16)


def _hw_mul(a, b):
    """One FPGA BF16 elementwise-multiply writeback."""
    return (a.to(torch.bfloat16) * b.to(torch.bfloat16)).to(torch.bfloat16)


def _hw_rope(x, rope_row, rotary_dim):
    """Apply the split-half RoPE used by rope_hf_core; retain unrotated tail."""
    half = rotary_dim // 2
    cos = rope_row[..., :rotary_dim].float()
    stored_sin = rope_row[..., rotary_dim:2 * rotary_dim].float()
    x0 = x[..., :half].float()
    x1 = x[..., half:rotary_dim].float()
    rotated = torch.cat((
        x0 * cos[..., :half] + x1 * stored_sin[..., :half],
        x1 * cos[..., half:] + x0 * stored_sin[..., half:],
    ), dim=-1)
    if rotary_dim < x.shape[-1]:
        rotated = torch.cat((rotated, x[..., rotary_dim:].float()), dim=-1)
    return rotated.to(torch.bfloat16)


def _hw_attention(q, k, v, sliding_window=None):
    """Token-space equivalent of GQA duplication + unified_attention_core."""
    seq_len, groups, _ = q.shape
    # QK matmul writes BF16 scores before the bias/softmax phase.
    scores = torch.einsum(
        "tgd,sd->tgs",
        q.to(torch.bfloat16).float(),
        k.to(torch.bfloat16).float()).to(torch.bfloat16)
    qi = torch.arange(seq_len).view(seq_len, 1)
    ki = torch.arange(seq_len).view(1, seq_len)
    valid = ki <= qi
    if sliding_window is not None:
        valid &= ki > (qi - sliding_window)
    scores.masked_fill_(~valid[:, None, :], float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(torch.bfloat16)
    return torch.einsum(
        "tgs,sd->tgd", probs.float(),
        v.to(torch.bfloat16).float()).to(torch.bfloat16)


def build_fpga_mimic_lm_reference(ue, inputs_embeds, prepared_per_layer_inputs):
    """Reproduce compile_prefill/decode operation-by-operation from params.bin.

    This deliberately does not call the HF language-model forward. Inputs are
    exact FPGA DRAM readbacks and every projection uses the serialized IF4/BF16
    weight tensor that the compiled program consumes.
    """
    x = inputs_embeds.to(torch.bfloat16).clone()
    pli = prepared_per_layer_inputs.to(torch.bfloat16)
    seq_len = x.shape[0]
    kv_cache = {}
    num_pos = ue._cfg["special"]["rope"]["num_positions"]
    rope_local = _bf16_region(
        ue, "ROPE_LOCAL", (num_pos, ue.head_dim_sliding * 2))
    global_rotary_dim = int(
        ue.head_dim * ue._cfg["special"]["rope"]["partial_rotary_factor_global"])
    rope_global_raw = _bf16_region(
        ue, "ROPE_GLOBAL", (num_pos, global_rotary_dim * 2))

    for layer_idx in range(ue.LAYER_SIZE):
        head_dim, q_size, k_size = ue._get_layer_attention_dims(layer_idx)
        mlp_dim = ue._get_mlp_elements(layer_idx)
        rotary_dim = ue._get_rope_dims(layer_idx)
        residual = x

        gamma = _bf16_region(
            ue, "BLK0_ATTN_NORM_WEIGHT", (ue.vector_length,), layer_idx)
        pre_norm = _hw_rms(x, gamma)
        q_w = _if4_region(ue, "BLK0_ATTN_Q_WEIGHT_SCALE",
                          "BLK0_ATTN_Q_WEIGHT_DATA", q_size,
                          ue.vector_length, layer_idx)
        q = _hw_linear(pre_norm, q_w).reshape(seq_len, ue.group_size, head_dim)

        if layer_idx not in ue._kv_shared_map:
            k_w = _if4_region(ue, "BLK0_ATTN_K_WEIGHT_SCALE",
                              "BLK0_ATTN_K_WEIGHT_DATA", k_size,
                              ue.vector_length, layer_idx)
            v_w = _if4_region(ue, "BLK0_ATTN_V_WEIGHT_SCALE",
                              "BLK0_ATTN_V_WEIGHT_DATA", k_size,
                              ue.vector_length, layer_idx)
            k = _hw_linear(pre_norm, k_w)
            v = _hw_rms(_hw_linear(pre_norm, v_w))
            gamma_k = _bf16_region(
                ue, "BLK0_ATTN_K_NORM_WEIGHT", (ue.head_dim,), layer_idx)[:head_dim]
            k = _hw_rms(k, gamma_k)
        gamma_q = _bf16_region(
            ue, "BLK0_ATTN_Q_NORM_WEIGHT", (ue.head_dim,), layer_idx)[:head_dim]
        q = _hw_rms(q, gamma_q)

        if layer_idx in ue._rope_global_layers:
            rope = rope_global_raw[:, :2 * rotary_dim]
        else:
            rope = rope_local[:, :2 * rotary_dim]
        q = _hw_rope(q, rope[:seq_len, None, :], rotary_dim)
        if layer_idx not in ue._kv_shared_map:
            k = _hw_rope(k, rope[:seq_len], rotary_dim)
            kv_cache[layer_idx] = (k, v)
        else:
            k, v = kv_cache[ue._kv_shared_map[layer_idx]]

        sliding = None if layer_idx in ue._full_attention_layers else ue.sliding_window
        attn = _hw_attention(q, k, v, sliding).reshape(seq_len, q_size)
        o_w = _if4_region(ue, "BLK0_ATTN_OUTPUT_WEIGHT_SCALE",
                          "BLK0_ATTN_OUTPUT_WEIGHT_DATA", ue.vector_length,
                          q_size, layer_idx)
        attn_out = _hw_linear(attn, o_w)
        gamma = _bf16_region(
            ue, "BLK0_POST_ATTENTION_NORM_WEIGHT", (ue.vector_length,), layer_idx)
        x = _hw_add(residual, _hw_rms(attn_out, gamma))

        gamma = _bf16_region(
            ue, "BLK0_FFN_NORM_WEIGHT", (ue.vector_length,), layer_idx)
        pre_mlp = _hw_rms(x, gamma)
        gate_w = _if4_region(ue, "BLK0_FFN_GATE_WEIGHT_SCALE",
                             "BLK0_FFN_GATE_WEIGHT_DATA", mlp_dim,
                             ue.vector_length, layer_idx)
        up_w = _if4_region(ue, "BLK0_FFN_UP_WEIGHT_SCALE",
                           "BLK0_FFN_UP_WEIGHT_DATA", mlp_dim,
                           ue.vector_length, layer_idx)
        gate = _hw_linear(pre_mlp, gate_w, gelu=True)
        up = _hw_linear(pre_mlp, up_w)
        mlp_product = _hw_mul(gate, up)
        del gate_w, up_w, gate, up
        down_w = _if4_region(ue, "BLK0_FFN_DOWN_WEIGHT_SCALE",
                             "BLK0_FFN_DOWN_WEIGHT_DATA", ue.vector_length,
                             mlp_dim, layer_idx)
        down = _hw_linear(mlp_product, down_w)
        gamma = _bf16_region(
            ue, "BLK0_POST_FFW_NORM_WEIGHT", (ue.vector_length,), layer_idx)
        x = _hw_add(x, _hw_rms(down, gamma))

        gate_w = _bf16_region(
            ue, "BLK0_PER_LAYER_INPUT_GATE_WEIGHT",
            (ue.per_layer_input_dim, ue.vector_length), layer_idx)
        inj_gate = _hw_linear(x, gate_w, gelu=True)
        inj_gate = _hw_mul(inj_gate, pli[:, layer_idx])
        proj_w = _bf16_region(
            ue, "BLK0_PER_LAYER_PROJECTION_WEIGHT",
            (ue.vector_length, ue.per_layer_input_dim), layer_idx)
        injection = _hw_linear(inj_gate, proj_w)
        gamma = _bf16_region(
            ue, "BLK0_POST_PER_LAYER_INPUT_NORM_WEIGHT",
            (ue.vector_length,), layer_idx)
        injection = _hw_rms(injection, gamma)
        x = _hw_add(x, injection)
        layer_scale = torch.as_tensor(
            ue._layer_scalars[layer_idx]).to(torch.bfloat16)
        x = _hw_mul(x, layer_scale)

    hidden = x.float().cpu()
    out_gamma = _bf16_region(ue, "OUTPUT_NORM_WEIGHT", (ue.vector_length,))
    norm = _hw_rms(x, out_gamma)
    # Prefill never executes LM-head. Compute only the final row, matching the
    # first decoder token and avoiding a seq_len x 262144 host allocation.
    lm_head = _if4_region(ue, "LM_HEAD_WEIGHT_SCALE", "LM_HEAD_WEIGHT_DATA",
                          ue.EMBEDDING_ELEMENTS, ue.vector_length)
    logits = _hw_linear(norm[-1:], lm_head)
    return {"hidden": hidden, "norm": norm.float().cpu(),
            "logits": logits.float().cpu()}


def main():
    p = argparse.ArgumentParser(
        description="Gemma4 E2B numeric check: LM prefill/decode, optionally vision.")
    p.add_argument("--image", type=str, nargs="?", const=g4r.DEFAULT_IMAGE, default=None,
                   help="Enable VLM checks. Bare --image uses the default image; an optional "
                        "path or filename selects another image.")
    p.add_argument("--prompt", type=str, default=None,
                   help="Text prompt. Without --image, default is the built-in LM test question.")
    p.add_argument("--dev", type=str, default="xdma0")
    p.add_argument("--cycle", type=float, default=1000 / 198.3256)
    args = p.parse_args()

    set_dma_device(args.dev)
    # The split mixin modules import DMA_DEVICE_* by value, so keep all three
    # module namespaces synchronized with set_dma_device().
    for module in (g4r, g4_lm_module, g4_vision_module):
        module.DMA_DEVICE_H2C = user_dma_core.DMA_DEVICE_H2C
        module.DMA_DEVICE_C2H = user_dma_core.DMA_DEVICE_C2H
        module.DMA_DEVICE_USER = user_dma_core.DMA_DEVICE_USER
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle
    print(f"Using DMA device: {args.dev}  (cycle {args.cycle:.4f} ns)")

    ue = g4r.Gemma4_UnifiedEngine()

    if args.image:
        # Resolve a bare filename against test_samples/, like test.py.
        image_path = args.image
        if not os.path.isfile(image_path):
            candidate = os.path.join(
                os.path.dirname(g4r.DEFAULT_IMAGE), os.path.basename(image_path))
            if os.path.isfile(candidate):
                image_path = candidate
        if not os.path.isfile(image_path):
            raise SystemExit(f"Image not found: {image_path}")

        # FPGA vision encoder plus the existing vision numeric comparisons.
        print(f"\n[numeric] running REFACTOR FPGA vision encoder on {image_path} ...")
        ue.set_prefill_seq_vlm(image_path, args.prompt)
        ckpt = getattr(ue, "_vis_ckpt", None)
        if not ckpt:
            raise SystemExit("FPGA checkpoints missing (self._vis_ckpt not set).")

        print("\n[numeric] building vision references ...")
        vision_prompt = args.prompt or "Describe this image in detail."
        refs, meta = build_vision_references(ue, image_path, vision_prompt)
        real = ~meta["padding"]
        stages = [("B encoder_out", "B", "encoder_out", real),
                  ("C image_features", "C", "image_features", None)]

        print("\n[numeric] ===== FPGA (refactor) vs references — SNR dB (real patches) =====")
        print("  FPGA vs HOSTREF (IF4 — mimics hardware; high = kernel correct):")
        for name, rkey, ckey, mask in stages:
            report(name, refs["hostref"][rkey], ckpt[ckey], row_mask=mask)
        print("  FPGA vs HF (full-precision ground truth; gap = IF4 quant loss):")
        for name, rkey, ckey, mask in stages:
            report(name, refs["hf"][rkey], ckpt[ckey], row_mask=mask)
    elif args.prompt:
        print(f"[numeric] LM-only mode, prompt: {args.prompt!r}")
        ue.set_prefill_seq(args.prompt)
    else:
        print("[numeric] LM-only mode, built-in default prompt")
        ue.set_prefill_seq()

    ue.compile_gemma4()
    lm_meta = ue._load_program_section("lm")
    ue._preamble_addr = ue.get_program_dram_addr()
    prefill_addr = int(lm_meta["prefill_program_start_addr"], 16)
    decoder_addr = int(lm_meta["decoder_program_start_addr"], 16)

    print("\n[numeric] running FPGA LM prefill ...")
    ue.run_prefill(prefill_addr, flops=lm_meta["prefill_total_flops"])
    prefill_len = len(ue.prefill_seq) - 1
    prefill_inputs = ue.dma_from_accelerator_memory(
        ue.LAYER0_INPUT_DRAM,
        (prefill_len, ue.vector_length)).cpu()
    prefill_per_layer = ue.dma_from_accelerator_memory(
        ue.PER_LAYER_INPUTS_DRAM,
        (prefill_len * ue.LAYER_SIZE, ue.per_layer_input_dim)).reshape(
            prefill_len, ue.LAYER_SIZE, ue.per_layer_input_dim).cpu()
    hw_prefill_hidden = ue.dma_from_accelerator_memory(
        ue.LAYER0_OUTPUT_DRAM,
        (prefill_len, ue.vector_length)).float().cpu()

    print("\n[numeric] running exactly one FPGA decoder step ...")
    saved_max_context = ue.MAX_CONTEXT_SIZE
    ue.MAX_CONTEXT_SIZE = len(ue.prefill_seq)
    try:
        ue.run_decoder(
            [lm_meta["decoder_program_size"]], decoder_addr,
            token_id=ue.prefill_seq[-1],
            flops_per_token=[lm_meta["decoder_total_flops"]])
    finally:
        ue.MAX_CONTEXT_SIZE = saved_max_context

    decoder_input = ue.dma_from_accelerator_memory(
        ue.LAYER0_INPUT_DRAM, (1, ue.vector_length)).cpu()
    decoder_per_layer = ue.dma_from_accelerator_memory(
        ue.PER_LAYER_INPUTS_DRAM,
        (ue.LAYER_SIZE, ue.per_layer_input_dim)).reshape(
            1, ue.LAYER_SIZE, ue.per_layer_input_dim).cpu()
    hw_decoder_hidden = ue.dma_from_accelerator_memory(
        ue.LAYER0_OUTPUT_DRAM, (1, ue.vector_length)).float().cpu()
    hw_decoder_norm = ue.dma_from_accelerator_memory(
        ue.OUTPUT_NORM_DRAM, (1, ue.vector_length)).float().cpu()
    hw_decoder_logits = ue.dma_from_accelerator_memory(
        ue.LOGITS_DRAM, (1, ue.EMBEDDING_ELEMENTS)).float().cpu()

    all_inputs = torch.cat((prefill_inputs, decoder_input), dim=0)
    all_per_layer = torch.cat((prefill_per_layer, decoder_per_layer), dim=0)

    print("\n[numeric] building full-precision LM references from exact FPGA inputs ...")
    hf_prefill = build_hf_lm_reference(
        ue, prefill_inputs, prefill_per_layer)
    hf_decode = build_hf_lm_reference(ue, all_inputs, all_per_layer)

    print("[numeric] building operation-level FPGA-mimicking LM reference "
          "from exact params.bin weights ...")
    mimic_decode = build_fpga_mimic_lm_reference(
        ue, all_inputs, all_per_layer)
    mimic_prefill_hidden = mimic_decode["hidden"][:prefill_len]

    print("\n[numeric] ===== LM PREFILL (exact FPGA inputs) =====")
    print("  FPGA vs HOSTREF (operation-level params.bin oracle):")
    report("prefill hidden", mimic_prefill_hidden, hw_prefill_hidden)
    print("  FPGA vs HF (full-precision projections):")
    report("prefill hidden", hf_prefill["hidden"], hw_prefill_hidden)

    ref_pos = all_inputs.shape[0] - 1
    mimic_hidden = mimic_decode["hidden"][ref_pos:ref_pos + 1]
    mimic_norm = mimic_decode["norm"][ref_pos:ref_pos + 1]
    mimic_logits = mimic_decode["logits"]
    hf_hidden = hf_decode["hidden"][:, ref_pos]
    hf_norm = hf_decode["norm"][:, ref_pos]
    hf_logits = hf_decode["logits"][:, ref_pos]

    print("\n[numeric] ===== LM DECODER (first generated token) =====")
    print("  FPGA vs HOSTREF (operation-level params.bin oracle):")
    report("decoder hidden", mimic_hidden, hw_decoder_hidden)
    report("decoder norm", mimic_norm, hw_decoder_norm)
    report("decoder logits", mimic_logits, hw_decoder_logits)
    print("  FPGA vs HF (full-precision projections):")
    report("decoder hidden", hf_hidden, hw_decoder_hidden)
    report("decoder norm", hf_norm, hw_decoder_norm)
    report("decoder logits", hf_logits, hw_decoder_logits)
    print(f"  argmax: HOSTREF={int(mimic_logits.argmax())} "
          f"HF={int(hf_logits.argmax())} FPGA={int(hw_decoder_logits.argmax())}")


if __name__ == "__main__":
    main()
