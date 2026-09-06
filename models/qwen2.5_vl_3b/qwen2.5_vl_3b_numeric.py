#!/usr/bin/env python3
"""Vision numeric checks for the Qwen2.5-VL-3B refactor engine.

Compares the FPGA encoder against TWO references, which answer different
questions:

  HOSTSIM  a host simulation of what the hardware is *supposed* to compute --
           the same IF4 weight bytes out of params.bin, the same 80->128 head
           padding, the same block-diagonal window mask, the same op order and
           BF16 writeback boundaries. FPGA vs HOSTSIM should be HIGH SNR. A low
           value here is a real kernel or wiring bug; quantization cannot
           explain it, because both sides use identical quantized weights.

  HF       the unquantized bf16 HuggingFace vision tower. FPGA vs HF is lower
           than FPGA vs HOSTSIM by exactly the IF4 loss, so HOSTSIM vs HF is
           the quantization floor -- the best any correct implementation of
           this weight format could do.

Read them together: if FPGA-vs-HOSTSIM is high and FPGA-vs-HF is low, the port
is right and 4-bit weights are the whole story. If FPGA-vs-HOSTSIM is also low,
the port is broken and HF is irrelevant until it is fixed.

Two stages are compared at each reference, so a failure localizes:
  encoder_out   after all 32 layers, before the merger  [576, 1280]
  merged        after ln_q + merger MLP                 [144, 2048]

TOKEN ORDER -- get this wrong and every number below is meaningless.

Window attention needs each 112-px window's tokens contiguous, so
``visual.get_window_index()`` permutes the 144 merge-units (a merge-unit is a
2x2 group of 4 patches) out of raster order into WINDOW order. The encoder --
ours and HF's alike -- runs entirely in window order.
``_vis_reverse_index = argsort(window_index)`` undoes it.

  encoder_out  window order on BOTH sides. HF's ``last_hidden_state`` is never
               un-permuted, and the device buffers never leave window order, so
               compare them AS-IS. Do not apply the reverse index.
  merged       raster order on both sides. HF's ``pooler_output`` is already
               un-permuted, and ``run_vision_encoder()`` reverse-indexes its
               return, so those match too. The raw VIS_ENCODER_OUT readback is
               still in window order -- that one is compared against the
               hostsim, which is also window order.

A wrong reverse-index reads as ~0 dB SNR with row-cosine still ~0.96, not as an
obvious failure: these hidden states have std ~241 with outliers ~33000 in a
few channels every token shares, so any two tokens point nearly the same way
and only SNR notices the mispairing.

Usage:
  python models/qwen2.5_vl_3b_new/qwen2.5_vl_3b_numeric.py --image
  python models/qwen2.5_vl_3b_new/qwen2.5_vl_3b_numeric.py --image people.jpg
  python models/qwen2.5_vl_3b_new/qwen2.5_vl_3b_numeric.py --image --layers 4
"""
import argparse
import math
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

import torch
import torch.nn as nn

import user_dma_core
from user_dma_core import calculate_snr
import quant_lib

_qt = None          # qwen2.5_vl_3b_test_new, loaded by path (see _load)
_qv = None          # qwen2.5_vl_3b_vision


def _load():
    """Import the dotted-name sibling modules by path ("2.5" is not an identifier)."""
    global _qt, _qv
    import importlib.util
    for name, filename in (("qwen2_5_vl_3b_test_new", "qwen2.5_vl_3b_test_new.py"),):
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(SCRIPT_DIR, filename))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        _qt = mod
    _qv = sys.modules["qwen2_5_vl_3b_vision"]     # test_new loads it on import


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def report(name, ref, res):
    """Print SNR, relative L2, worst row cosine and max abs error."""
    ref = ref.detach().float().reshape(-1, ref.shape[-1])
    res = res.detach().float().reshape(-1, res.shape[-1])
    rows = min(ref.shape[0], res.shape[0])
    ref, res = ref[:rows], res[:rows]
    snr = float(calculate_snr(ref, res))
    signal = float(torch.linalg.vector_norm(ref))
    rel = float(torch.linalg.vector_norm(res - ref)) / signal if signal else float("inf")
    row_cos = torch.nn.functional.cosine_similarity(res, ref, dim=-1)
    print(f"  [numeric] {name:28s} SNR={snr:7.2f} dB  rel_L2={rel:.4g}  "
          f"row_cos min={float(row_cos.min()):.5f} mean={float(row_cos.mean()):.5f}  "
          f"max|d|={float((res - ref).abs().max()):.4g}")
    return snr


# ---------------------------------------------------------------------------
# HOSTSIM: operation-by-operation mimic of the emitted encoder program
# ---------------------------------------------------------------------------

def _hw_rms(x, gamma):
    """rms_norm_core convention: BF16 reciprocal-RMS broadcast, then BF16 gamma."""
    x = x.to(torch.bfloat16)
    inv = torch.rsqrt(torch.mean(x.float().square(), dim=-1, keepdim=True)).to(torch.bfloat16)
    return ((x * inv).to(torch.bfloat16) * gamma.to(torch.bfloat16)).to(torch.bfloat16)


def _hw_linear(x, weight, bias=None, silu=False, gelu=False):
    """matmat_mul_core: A @ B.T with broadcast bias, fused act, one BF16 writeback."""
    out = x.to(torch.bfloat16).float() @ weight.to(torch.bfloat16).float().T
    if bias is not None:
        out = out + bias.to(torch.bfloat16).float()
    if silu:
        out = out * torch.sigmoid(out)
    if gelu:
        # LALU_ACT_GELU_B = 0xbfd9 -> -1.6953125; the fused datapath computes
        # x*sigmoid(1.6953125*x) straight out of the matmul accumulator.
        out = out * torch.sigmoid(1.6953125 * out)
    return out.to(torch.bfloat16)


def _hw_rope(x, cos, sin):
    """rope_hf_core_dram: out = x*cos + rotate_half(x)*sin, sin pre-signed.

    Operates at the PADDED width, where rotate-half pairs lane i with i+N/2 --
    which is exactly why the weight generator stores Q/K rearranged into
    [0:40] and [64:104].
    """
    half = x.shape[-1] // 2
    rot = torch.cat((x[..., half:], x[..., :half]), dim=-1)
    return (x.float() * cos.float() + rot.float() * sin.float()).to(torch.bfloat16)


def _hw_attention(q, k, v, bias, q_scale):
    """unified_attention_core for one head: BF16 scores, additive bias, softmax."""
    scores = (q.to(torch.bfloat16).float() * q_scale) @ k.to(torch.bfloat16).float().T
    scores = scores.to(torch.bfloat16).float() + bias.float()
    probs = torch.softmax(scores, dim=-1).to(torch.bfloat16)
    return (probs.float() @ v.to(torch.bfloat16).float()).to(torch.bfloat16)


class VisionWeights:
    """The exact bytes the FPGA reads, dequantized on the host."""

    def __init__(self, ue):
        region = ue._read_vision_region()
        self.sections = region["sections"]
        self.base = region["base_offset"]
        with open(region["bin_path"], "rb") as f:
            f.seek(self.base)
            self.blob = f.read(region["size"])
        d = ue._vision_dims()
        self.d = d

    def _raw(self, key):
        s = self.sections[key]
        return self.blob[s["offset"]:s["offset"] + s["size"]]

    def bf16(self, key):
        return torch.frombuffer(bytearray(self._raw(key)), dtype=torch.bfloat16)

    def if4(self, key, n, k):
        """Split the [scales | data] section and dequantize to [n, k] BF16."""
        raw = self._raw(key)
        n_blocks = len(raw) // 34
        if n_blocks * 64 != n * k:
            raise ValueError(f"{key}: {n_blocks} blocks != {n}x{k}/64")
        scales, data = raw[:n_blocks * 2], raw[n_blocks * 2:]
        return quant_lib.dequant("if4", data, scales, n, k, block_size=64)


class HFVisionWeightsBF16:
    """The same padded tensors, built straight from the HF model at BF16.

    Exposes the identical interface as VisionWeights, so the hostsim body runs
    unchanged and the ONLY difference between the two runs is 4-bit weights.
    The padding must match the weight generator exactly (see the old build's
    generate_vision_weights): Q/K are REARRANGED into lanes [0:40] and [64:104]
    so that rotate-half at width 128 pairs the same dims it would at 80, while
    V is SEQUENTIAL [0:80] -- which is also why the attention output trims
    lanes 0..79.
    """

    def __init__(self, ue):
        d = ue._vision_dims()
        VN, VD, VD_PAD, VH = d["VN"], d["VD"], d["VD_PAD"], d["VH"]
        VI, VI_PAD = d["VI"], d["VI_PAD"]
        half = VD // 2
        visual = ue._hf_model.visual
        w = {}
        with torch.no_grad():
            for i, blk in enumerate(visual.blocks):
                pre = f"visual.blocks.{i}"
                qkv_w = blk.attn.qkv.weight.detach().to(torch.bfloat16).view(3, VN, VD, VH)
                qkv_b = blk.attn.qkv.bias.detach().to(torch.bfloat16).view(3, VN, VD)
                qk_w = torch.zeros(2 * VN * VD_PAD, VH, dtype=torch.bfloat16)
                qk_b = torch.zeros(2 * VN * VD_PAD, dtype=torch.bfloat16)
                for proj in range(2):
                    for h in range(VN):
                        hs = (proj * VN + h) * VD_PAD
                        qk_w[hs:hs + half] = qkv_w[proj, h, :half]
                        qk_w[hs + 64:hs + 64 + half] = qkv_w[proj, h, half:]
                        qk_b[hs:hs + half] = qkv_b[proj, h, :half]
                        qk_b[hs + 64:hs + 64 + half] = qkv_b[proj, h, half:]
                w[f"{pre}.attn.qk_padded.weight.if4"] = qk_w
                w[f"{pre}.attn.qk_padded.bias"] = qk_b
                v_w = torch.zeros(VN * VD_PAD, VH, dtype=torch.bfloat16)
                v_b = torch.zeros(VN * VD_PAD, dtype=torch.bfloat16)
                for h in range(VN):
                    hs = h * VD_PAD
                    v_w[hs:hs + VD] = qkv_w[2, h]
                    v_b[hs:hs + VD] = qkv_b[2, h]
                w[f"{pre}.attn.v_padded.weight.if4"] = v_w
                w[f"{pre}.attn.v_padded.bias"] = v_b
                w[f"{pre}.attn.proj.weight.if4"] = blk.attn.proj.weight.detach().to(torch.bfloat16)
                w[f"{pre}.attn.proj.bias"] = blk.attn.proj.bias.detach().to(torch.bfloat16)
                for name in ("gate_proj", "up_proj"):
                    lin = getattr(blk.mlp, name)
                    pw = torch.zeros(VI_PAD, VH, dtype=torch.bfloat16)
                    pw[:VI] = lin.weight.detach().to(torch.bfloat16)
                    pb = torch.zeros(VI_PAD, dtype=torch.bfloat16)
                    pb[:VI] = lin.bias.detach().to(torch.bfloat16)
                    w[f"{pre}.mlp.{name}.weight.if4"] = pw
                    w[f"{pre}.mlp.{name}.bias"] = pb
                dw = torch.zeros(VH, VI_PAD, dtype=torch.bfloat16)
                dw[:, :VI] = blk.mlp.down_proj.weight.detach().to(torch.bfloat16)
                w[f"{pre}.mlp.down_proj.weight.if4"] = dw
                w[f"{pre}.mlp.down_proj.bias"] = blk.mlp.down_proj.bias.detach().to(torch.bfloat16)
                w[f"{pre}.norm1.weight"] = blk.norm1.weight.detach().to(torch.bfloat16)
                w[f"{pre}.norm2.weight"] = blk.norm2.weight.detach().to(torch.bfloat16)
            w["visual.merger.ln_q.weight"] = visual.merger.ln_q.weight.detach().to(torch.bfloat16)
            for idx in (0, 2):
                w[f"visual.merger.mlp.{idx}.weight.if4"] = visual.merger.mlp[idx].weight.detach().to(torch.bfloat16)
                w[f"visual.merger.mlp.{idx}.bias"] = visual.merger.mlp[idx].bias.detach().to(torch.bfloat16)
        self._w = w

    def bf16(self, key):
        return self._w[key]

    def if4(self, key, n, k):
        t = self._w[key]
        if tuple(t.shape) != (n, k):
            raise ValueError(f"{key}: {tuple(t.shape)} != ({n}, {k})")
        return t


def build_hostsim_reference(ue, patch_embeds, rope_table, bias_full, bias_window,
                            num_layers=None, weights=None, tag="hostsim"):
    """Recompute the encoder on the host exactly as the FPGA program does.

    Stays in WINDOW order throughout -- the same order the FPGA buffers hold --
    so comparing against the raw device readback needs no reordering, which is
    what makes this reference unambiguous.
    """
    d = ue._vision_dims()
    VS, VH, VN, VD, VD_PAD = d["VS"], d["VH"], d["VN"], d["VD"], d["VD_PAD"]
    VI_PAD, VMERGE, VH_OUT = d["VI_PAD"], d["VMERGE"], d["VH_OUT"]
    T, FULL = d["NUM_MERGED_TOKENS"], d["FULL_ATTN_LAYERS"]
    VL = d["VL"] if num_layers is None else num_layers
    q_scale = 1.0 / math.sqrt(VD)
    W = VisionWeights(ue) if weights is None else weights

    # rope_table is [VN*VS, 2, VD_PAD], tiled per head; one head's slice suffices.
    cos = rope_table[:VS, 0, :]
    sin = rope_table[:VS, 1, :]

    x = patch_embeds.to(torch.bfloat16)
    t0 = time.perf_counter()
    for li in range(VL):
        pre = f"visual.blocks.{li}"
        h = _hw_rms(x, W.bf16(f"{pre}.norm1.weight"))

        qk = _hw_linear(h, W.if4(f"{pre}.attn.qk_padded.weight.if4",
                                 2 * VN * VD_PAD, VH),
                        W.bf16(f"{pre}.attn.qk_padded.bias"))
        v = _hw_linear(h, W.if4(f"{pre}.attn.v_padded.weight.if4", VN * VD_PAD, VH),
                       W.bf16(f"{pre}.attn.v_padded.bias"))

        # [VS, 2*VN, VD_PAD] -> head-major; groups 0..VN-1 are Q, VN.. are K.
        qk = qk.reshape(VS, 2 * VN, VD_PAD).permute(1, 0, 2)
        q_h, k_h = qk[:VN], qk[VN:]
        v_h = v.reshape(VS, VN, VD_PAD).permute(1, 0, 2)
        q_h = _hw_rope(q_h, cos, sin)
        k_h = _hw_rope(k_h, cos, sin)

        bias = bias_full if li in FULL else bias_window
        out_h = torch.stack([_hw_attention(q_h[i], k_h[i], v_h[i], bias, q_scale)
                             for i in range(VN)])
        # head-major -> token-major, then drop each head's pad lanes.
        attn = out_h.permute(1, 0, 2)[:, :, :VD].reshape(VS, VN * VD)

        o = _hw_linear(attn, W.if4(f"{pre}.attn.proj.weight.if4", VH, VH),
                       W.bf16(f"{pre}.attn.proj.bias"))
        res = (x.to(torch.bfloat16) + o).to(torch.bfloat16)

        h2 = _hw_rms(res, W.bf16(f"{pre}.norm2.weight"))
        gate = _hw_linear(h2, W.if4(f"{pre}.mlp.gate_proj.weight.if4", VI_PAD, VH),
                          W.bf16(f"{pre}.mlp.gate_proj.bias"), silu=True)
        up = _hw_linear(h2, W.if4(f"{pre}.mlp.up_proj.weight.if4", VI_PAD, VH),
                        W.bf16(f"{pre}.mlp.up_proj.bias"))
        mult = (gate * up).to(torch.bfloat16)
        down = _hw_linear(mult, W.if4(f"{pre}.mlp.down_proj.weight.if4", VH, VI_PAD),
                          W.bf16(f"{pre}.mlp.down_proj.bias"))
        x = (res + down).to(torch.bfloat16)
        print(f"\r    {tag} layer {li + 1}/{VL}", end="", flush=True)
    print(f"   ({time.perf_counter() - t0:.1f}s)")

    encoder_out = x
    merge_dim = VMERGE * VMERGE * VH
    pn = _hw_rms(x, W.bf16("visual.merger.ln_q.weight"))
    # 2x2 merge is a reshape: window reordering already put each unit's 4
    # patches in consecutive rows.
    m = pn.reshape(T, merge_dim)
    inter = _hw_linear(m, W.if4("visual.merger.mlp.0.weight.if4", merge_dim, merge_dim),
                       W.bf16("visual.merger.mlp.0.bias"), gelu=True)
    merged = _hw_linear(inter, W.if4("visual.merger.mlp.2.weight.if4", VH_OUT, merge_dim),
                        W.bf16("visual.merger.mlp.2.bias"))
    return encoder_out, merged


# ---------------------------------------------------------------------------
# HF: unquantized bf16 tower
# ---------------------------------------------------------------------------

def build_hf_reference(ue):
    """Run the stock HF vision tower on the same preprocessed pixels."""
    with torch.no_grad():
        out = ue._hf_model.visual(ue._vis_hf_pixels.to(torch.bfloat16),
                                  grid_thw=ue._image_grid_thw)
    if hasattr(out, "pooler_output"):
        return out.last_hidden_state.float(), out.pooler_output.float()
    if isinstance(out, (tuple, list)):
        return None, out[0].float()
    return None, out.float()


def _run_lm_check(ue, args):
    """Bisect the LM prefill: FPGA vs IF4 hostsim, layer by layer.

    Compares the hidden state after N layers, read straight out of the
    ping-pong buffer the last layer wrote. FPGA-vs-HOSTSIM is the diagnostic --
    both sides use identical quantized weights, so a low SNR there is emission,
    not quantization.
    """
    ue.lm_weight_init()
    ue.lm_tensor_init()
    tokens = list(ue._cfg["default_prefill_tokens"])[:-1]
    layers = args.layers or ue._lm_dims()["NL"]
    print(f"\n--- LM prefill check: {len(tokens)} tokens, {layers} layer(s) ---")

    ue.compile_prefill(len(tokens), layer_size=layers)
    ue.run_prefill(tokens)
    got = ue.dma_from_accelerator_memory(
        ue.LM_PREFILL_OUT, (len(tokens), ue._lm_dims()["H"])).cpu()

    print("  building IF4 hostsim ...")
    sim = build_lm_hostsim(ue, tokens, num_layers=layers)
    print(f"\n=== FPGA vs HOSTSIM (both IF4 -- low SNR here is emission, not quant) ===")
    report(f"hidden after {layers} layer(s)", sim, got)
    if not args.skip_hf:
        ref = build_lm_hf_reference(ue, tokens, num_layers=layers)
        print(f"\n=== vs HF (adds IF4 loss) ===")
        report("HOSTSIM vs HF", ref, sim)
        report("FPGA vs HF", ref, got)
    print(f"\n--- Cleaning DRAM (4 GiB) ---")
    _qt.clean_dram_4gb(ue)


class _HostOnlyEngine:
    """Factory for an engine whose DMAs are no-ops, so the host references can
    be built without a board. Allocation and program emission still run for
    real; only the device traffic is dropped."""

    @staticmethod
    def make(cls, **kwargs):
        stub = type("HostOnly" + cls.__name__, (cls,), dict(
            dma_write=lambda self, dev, addr, data, size: None,
            dma_to_accelerator_memory=lambda self, addr, data: None,
            dma_from_accelerator_memory=lambda self, addr, shape: torch.zeros(
                shape, dtype=torch.bfloat16),
            start_execute_from_dram=lambda self, addr: None,
            wait_queue=lambda self, timeout=0.0: 0.0,
            report_latency_in_us=lambda self: 1.0,
        ))
        return stub(**kwargs)


# ---------------------------------------------------------------------------

def main():
    _load()
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL-3B vision numeric check: FPGA vs host-simulate vs HF.")
    parser.add_argument("--image", type=str, nargs="?",
                        const=_qt.DEFAULT_IMAGE, default=_qt.DEFAULT_IMAGE,
                        help="Image to encode (default: the shipped sample).")
    parser.add_argument("--layers", type=int, default=None,
                        help="Compile and simulate only the first N encoder layers. "
                             "Bisects a mismatch: run 1, 2, 4 ... until SNR collapses.")
    parser.add_argument("--skip-hf", action="store_true",
                        help="Skip the HuggingFace reference (host-simulate only).")
    parser.add_argument("--weights", choices=("if4", "bf16", "both"), default="if4",
                        help="Weights for the host simulation. 'if4' is what the FPGA "
                             "runs today. 'bf16' answers what an unquantized vision bin "
                             "would score WITHOUT generating one or touching the board. "
                             "'both' runs the pair and prints the quantization delta.")
    parser.add_argument("--lm", action="store_true",
                        help="Check the LM prefill path instead of the vision encoder: "
                             "FPGA vs an IF4 host simulation, and both vs HuggingFace.")
    parser.add_argument("--no-fpga", action="store_true",
                        help="Host only: skip the device entirely (no reset, no run, no "
                             "DRAM clean). Use with --weights both to price BF16 vision.")
    _qt.add_engine_args(parser)
    args = parser.parse_args()
    if not args.no_fpga:
        _qt.resolve_engine_config(parser, args)
        from user_hw_test import software_reset_test
        cores = args.multi_core or 1
        print(f"\n--- Software-resetting {cores} core(s) ---")
        software_reset_test(cores=cores)
        print(f"\n--- Cleaning DRAM (4 GiB) ---")
        _qt.clean_dram_4gb()
    else:
        # The allocators and the emitter still need the AXI width and clock that
        # HW_INFO would supply; nothing here reaches the device.
        user_dma_core.CLOCK_CYCLE_TIME_NS = 2.7272630423897644
        user_dma_core.UE_AXI_DATA_WIDTH_BITS = 256
        user_dma_core.AVAILABLE_DRAM_SIZE_GB = 4
        cores = 1
        print("\n--- Host-only mode: the device is not touched ---")

    if not os.path.isfile(args.image):
        cand = os.path.join(os.path.dirname(_qt.DEFAULT_IMAGE),
                            os.path.basename(args.image))
        args.image = cand if os.path.isfile(cand) else args.image
    if not os.path.isfile(args.image):
        raise SystemExit(f"--image: file not found: {args.image!r}")

    if args.no_fpga:
        ue = _HostOnlyEngine.make(_qt.Qwen25VL_UnifiedEngine, multi_core=cores)
    else:
        ue = _qt.Qwen25VL_UnifiedEngine(multi_core=cores)

    if args.lm:
        return _run_lm_check(ue, args)

    ue.vision_weight_init()
    print(f"\n--- Host preprocessing ({os.path.basename(args.image)}) ---")
    ue.prepare_encoder_input(_qt.process_image(args.image))
    ue.vision_tensor_init()

    d = ue._vision_dims()
    VS, VH, T, VH_OUT = d["VS"], d["VH"], d["NUM_MERGED_TOKENS"], d["VH_OUT"]
    unit = d["VMERGE"] ** 2
    rev = ue._vis_reverse_index

    fpga_encoder_out = fpga_merged_win = None
    if not args.no_fpga:
        ue.compile_vision_encoder()
        ue.run_vision_encoder()
        # Raw device readbacks, both still in WINDOW order.
        final_buf = ue.VIS_IO_A if d["VL"] % 2 == 0 else ue.VIS_IO_B
        fpga_encoder_out = ue.dma_from_accelerator_memory(final_buf, (VS, VH)).cpu()
        fpga_merged_win = ue.dma_from_accelerator_memory(
            ue.VIS_ENCODER_OUT, (T, VH_OUT)).cpu()

    rope = ue._build_rope_tables().reshape(-1, 2, d["VD_PAD"])
    bias_full, bias_window = ue._build_attention_bias()
    want = ("if4", "bf16") if args.weights == "both" else (args.weights,)
    sims = {}
    for kind in want:
        print(f"\n--- HOSTSIM reference ({kind.upper()} weights) ---")
        provider = None if kind == "if4" else HFVisionWeightsBF16(ue)
        sims[kind] = build_hostsim_reference(
            ue, ue._vis_patch_embeds, rope, bias_full.float(), bias_window.float(),
            num_layers=args.layers, weights=provider, tag=f"hostsim-{kind}")

    if fpga_encoder_out is not None and "if4" in sims:
        sim_enc, sim_mrg = sims["if4"]
        print(f"\n=== FPGA vs HOSTSIM-IF4 (both IF4 -- a low SNR here is a real bug) ===")
        report("encoder_out [576,1280]", sim_enc, fpga_encoder_out)
        if args.layers is None:
            report("merged [144,2048]", sim_mrg, fpga_merged_win)

    if not args.skip_hf:
        print(f"\n--- HF reference (unquantized bf16) ---")
        hf_hidden, hf_merged = build_hf_reference(ue)
        print(f"\n=== vs HF ===")
        for kind in want:
            sim_enc, sim_mrg = sims[kind]
            if hf_hidden is not None and args.layers is None:
                # Both sides are in window order -- compare as-is (see TOKEN ORDER).
                report(f"encoder_out: HOSTSIM-{kind} vs HF", hf_hidden, sim_enc)
            if args.layers is None:
                report(f"merged: HOSTSIM-{kind} vs HF", hf_merged, sim_mrg[rev])
        if fpga_encoder_out is not None:
            if hf_hidden is not None and args.layers is None:
                report("encoder_out: FPGA vs HF", hf_hidden, fpga_encoder_out)
            if args.layers is None:
                report("merged: FPGA vs HF", hf_merged, ue._vis_embeddings)

    if args.weights == "both" and args.layers is None:
        print(f"\n=== IF4 quantization cost (BF16 hostsim as the ceiling) ===")
        report("encoder_out: IF4 vs BF16", sims["bf16"][0], sims["if4"][0])
        report("merged: IF4 vs BF16", sims["bf16"][1], sims["if4"][1])

    if args.no_fpga:
        return





# ---------------------------------------------------------------------------
# LM references
# ---------------------------------------------------------------------------

class LMWeights:
    """The exact LM bytes the FPGA reads, dequantized on the host."""

    def __init__(self, ue):
        region = ue._read_lm_region()
        self.sections = region["sections"]
        with open(region["bin_path"], "rb") as f:
            f.seek(region["base_offset"])
            self.blob = f.read(region["size"])

    def _raw(self, key):
        s = self.sections[key]
        return self.blob[s["offset"]:s["offset"] + s["size"]]

    def bf16(self, key, shape=None):
        t = torch.frombuffer(bytearray(self._raw(key)), dtype=torch.bfloat16)
        return t.reshape(shape) if shape else t

    def if4(self, key, n, k):
        raw = self._raw(key)
        nb = len(raw) // 34
        if nb * 64 != n * k:
            raise ValueError(f"{key}: {nb} blocks != {n}x{k}/64")
        return quant_lib.dequant("if4", raw[nb * 2:], raw[:nb * 2], n, k, block_size=64)


def build_lm_hostsim(ue, tokens, num_layers=None, weights=None):
    """Recompute LM prefill on the host exactly as the FPGA program does.

    Returns the hidden state after ``num_layers`` layers, so a mismatch can be
    bisected to a layer instead of only observed at the logits.
    """
    d = ue._lm_dims()
    H, AHD, KVH, QH, G, MLP = d["H"], d["AHD"], d["KVH"], d["QH"], d["G"], d["MLP"]
    NL = d["NL"] if num_layers is None else num_layers
    W = LMWeights(ue) if weights is None else weights
    M = len(tokens)

    x = ue.get_embedding_for_tokens(tokens).to(torch.bfloat16)
    pos = torch.arange(M)
    half = AHD // 2
    theta = ue._cfg["special"]["rope"]["theta"]
    inv = 1.0 / (theta ** (torch.arange(half, dtype=torch.float32) / half))
    fr = torch.outer(pos.float(), inv)
    cos = torch.cat([fr.cos(), fr.cos()], -1).to(torch.bfloat16)
    sin = torch.cat([fr.sin(), fr.sin()], -1).to(torch.bfloat16)

    def rope(t):                      # t: [M, heads, AHD]
        c = cos[:, None, :].float()
        s = sin[:, None, :].float()
        rot = torch.cat((-t[..., half:].float(), t[..., :half].float()), -1)
        return (t.float() * c + rot * s).to(torch.bfloat16)

    causal = torch.full((M, M), float("-inf"))
    causal.masked_fill_(torch.tril(torch.ones(M, M, dtype=torch.bool)), 0.0)
    scale = 1.0 / math.sqrt(AHD)

    for li in range(NL):
        pre = f"language_model.layers.{li}"
        h = _hw_rms(x, W.bf16(f"{pre}.input_layernorm.weight"))
        q = _hw_linear(h, W.if4(f"{pre}.self_attn.q_proj.weight.if4", QH * AHD, H),
                       W.bf16(f"{pre}.self_attn.q_proj.bias"))
        k = _hw_linear(h, W.if4(f"{pre}.self_attn.k_proj.weight.if4", KVH * AHD, H),
                       W.bf16(f"{pre}.self_attn.k_proj.bias"))
        v = _hw_linear(h, W.bf16(f"{pre}.self_attn.v_proj.weight", (KVH * AHD, H)),
                       W.bf16(f"{pre}.self_attn.v_proj.bias"))
        q = rope(q.reshape(M, QH, AHD))
        k = rope(k.reshape(M, KVH, AHD))
        v = v.reshape(M, KVH, AHD)

        heads = []
        for qh in range(QH):
            kv = qh // G
            sc = (q[:, qh].float() * scale) @ k[:, kv].float().T
            sc = sc.to(torch.bfloat16).float() + causal
            heads.append((torch.softmax(sc, -1).to(torch.bfloat16).float()
                          @ v[:, kv].float()).to(torch.bfloat16))
        attn = torch.stack(heads, 1).reshape(M, QH * AHD)

        o = _hw_linear(attn, W.bf16(f"{pre}.self_attn.o_proj.weight", (H, QH * AHD)))
        x = (x.to(torch.bfloat16) + o).to(torch.bfloat16)
        h2 = _hw_rms(x, W.bf16(f"{pre}.post_attention_layernorm.weight"))
        gate = _hw_linear(h2, W.if4(f"{pre}.mlp.gate_proj.weight.if4", MLP, H), silu=True)
        up = _hw_linear(h2, W.if4(f"{pre}.mlp.up_proj.weight.if4", MLP, H))
        down = _hw_linear((gate * up).to(torch.bfloat16),
                          W.if4(f"{pre}.mlp.down_proj.weight.if4", H, MLP))
        x = (x + down).to(torch.bfloat16)
        print(f"\r    lm hostsim layer {li + 1}/{NL}", end="", flush=True)
    print()
    return x


def build_lm_hf_reference(ue, tokens, num_layers=None):
    """HF hidden states for the same tokens, layer by layer (unquantized bf16)."""
    from transformers import Qwen2_5_VLForConditionalGeneration
    if not hasattr(ue, "_hf_model"):
        ue._hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            os.path.join(ue.script_dir, ue._cfg["paths"]["hf_model_dir"]),
            torch_dtype=torch.bfloat16)
        ue._hf_model.eval()
    lm = getattr(ue._hf_model, "model", ue._hf_model)
    lm = getattr(lm, "language_model", lm)
    with torch.no_grad():
        out = lm(input_ids=torch.tensor([list(tokens)]), output_hidden_states=True)
    hs = out.hidden_states                      # tuple: embeddings, then each layer
    NL = ue._lm_dims()["NL"] if num_layers is None else num_layers
    return hs[NL].squeeze(0).float()


if __name__ == "__main__":
    main()
