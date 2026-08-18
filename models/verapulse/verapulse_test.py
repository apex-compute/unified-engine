"""VeraPulse PulseVLA-LIBERO-0.5B on the unified engine -- SKELETON.

Every method is a stub carrying the contract, shapes, and hardware rules it must
satisfy. Fill them in one stage at a time, gating each on a >=40 dB SNR probe
against the pure-torch reference before moving on.

REUSE, don't re-derive:
  * models/smolvlm2/ IS the VLM half of this model, already brought up and validated
    on hardware -- same ViT (12x768), same pixel-shuffle x4 connector, same SmolLM2
    decoder (960, 15q/5kv, hd 64, 32 layers). The LM section below mirrors
    smolvlm2_test.py::compile_prefill structure verbatim (lm_matmul / strided_copy /
    duplicate_gqa_rows, per-kv-group stacked-Q flash, ping-pong h_in/h_out).
  * models/pi05_libero/ supplies the STAGE structure: compile-once + bins, prefix-KV
    dump for oracle isolation, action expert + flow-matching denoise loop, SNR gating.
    Its ATTENTION emitters do NOT transfer -- pi05 is MQA (1 kv head), this is GQA.

Architecture (checkpoint header + SmolVLM2-500M reference):

    vision   SigLIP-style ViT   12 layers  hidden 768   12q/12kv  hd 64  MHA, LN+bias, GELU
             512x512 image / patch 16 -> 32x32 = 1024 tokens per camera, 2 cameras
    connector pixel-shuffle x4 (1024x768 -> 64x12288) -> linear -> 64 tokens x 960
    lm       SmolLM2 decoder    32 layers  hidden 960   15q/5kv  hd 64  GQA g3, RMS, SiLU, RoPE
    expert   action expert      32 layers  hidden 480   15q/5kv  hd 64  GQA g3, RMS, SiLU
             self-attn on layers i%2==0, cross-attn onto the frozen VLM prefix K/V otherwise
    head     flow matching, 10 Euler steps, chunk 50 x action_dim 7 (padded to 32)

  No stack is MQA. GQA group 3 -> the 5 kv heads are row-replicated to 15 token-major
  before flash (duplicate_gqa_rows). expert kv_out 320 == lm kv_out 320, which is what
  lets cross-attention consume the prefix KV cache without any reshaping -- it is still
  RE-PROJECTED through the cross layer's own k_proj/v_proj, which are (320,320) in the
  checkpoint (self layers' are (320,480)).

  RoPE theta = 10000.0, the value in the verapulse checkpoint's own config.json. Note
  this DIFFERS from upstream SmolVLM2 (100000.0) and therefore from
  models/smolvlm2/smolvlm2_config.json -- when lifting _load_rope_tables from there,
  the tables must be rebuilt at 10000, not copied.

Bring-up order (each stage gated at >=40 dB, pad rows excluded):
    1. vision layer 0        5. prefix (lm) full + KV cache
    2. vision full           6. expert step 0 (with a CPU-provided prefix KV)
    3. connector             7. full 10-step denoise -> actions
    4. prefix (lm) layer 0   8. LIBERO closed loop

Hardware rules that shape this file -- see verapulse_config.json["hw_notes"]:
  * every head_dim is 64: no 72->128 pad, no unpad selection matrix. Flash is happy.
  * expert hidden 480 is NOT %64: pad to 512 in the STORED weights, and fold
    sqrt(480/512) = 0.968246 into the expert RMSNorm gammas (scale DOWN -- normalizing
    over 512 makes the denominator smaller, so the kernel output is larger).
  * HW GELU is x*sigmoid(1.702*x) -- the oracle must match.
  * flash stays static/legacy (PBI flash address injection dies on re-execution).
  * build any extra engine BEFORE weight_init -- the ctor DMA-writes 16KB of noise to a
    HARDCODED 0x80000000, so an engine built AFTER weight_init shreds the head of params.
    This model is maximally exposed (params_dram_base 0x0 -> absolute 0x80000000 -> the
    first-stored weight is in the blast radius). If it must be built later, snapshot
    0x80000000..+64KB into a BFLOAT16 buffer and restore -- dma_read only round-trips
    bits losslessly for bf16/int32; any other dtype corrupts more than it fixes.
  * strided SRAM->DRAM copies need the per-index offset in the DEST base address.
    Three unavoidable instances here: the attention head unstack (+h*64*2), the
    patchify->patch_embed channel-major flatten, and the pixel-shuffle gather.
  * ue_selector's runtime_addr() and append_row() share ONE scratch GPR (_addr_tmp), so
    emit every op consuming a computed address -- all the RoPE calls -- BEFORE any
    append_row(). This is the bug that gave 94% relative error on Q while K stayed golden.
  * never skip compute for masked/zero slots; mask via the attention bias instead.

Usage (once implemented):
    python verapulse_test.py --stage vision --snr
    python verapulse_test.py --stage all --dump-bins
"""

import argparse
import builtins
import contextlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# ======================================================================================
# quiet mode -- installed BEFORE user_dma_core/nn_lib are imported so their prints route
# through it. Those libraries emit per-op FLOP/URAM dumps during compilation: thousands
# of lines that bury the timing and NaN signal we actually care about. Same mechanism
# pi05_libero/qwen3_4b/locateanything_3b use.
# ======================================================================================
_SILENT_MODE = False
_original_print = builtins.print


def _quiet_print(*args, **kwargs):
    if not _SILENT_MODE:
        _original_print(*args, **kwargs)


builtins.print = _quiet_print


@contextlib.contextmanager
def silenced():
    """Suppress library chatter for the duration of a block. Our own reporting uses
    _original_print so it survives."""
    global _SILENT_MODE
    prev, _SILENT_MODE = _SILENT_MODE, True
    try:
        yield
    finally:
        _SILENT_MODE = prev


# ======================================================================================
# phase timing -- COMPILE and EXECUTE are different costs with different fixes.
# Compile time is paid once and is fixed by shrinking shape diversity / rolling loops;
# execute time is paid every inference and is fixed by sharding or fewer DRAM round
# trips. Reporting one number for both hides which one is the problem.
# ======================================================================================
@dataclass
class Phases:
    rows: list = field(default_factory=list)      # (label, kind, seconds)

    @contextlib.contextmanager
    def track(self, label, kind):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.rows.append((label, kind, time.perf_counter() - t0))

    def summary(self, title="timing"):
        if not self.rows:
            return
        w = max(len(r[0]) for r in self.rows)
        _original_print(f"\n=== {title.upper()} ===")
        tot = {}
        for label, kind, s in self.rows:
            tot[kind] = tot.get(kind, 0.0) + s
            _original_print(f"  {kind:<8} {label:<{w}}  {s:8.2f}s")
        _original_print("  " + "-" * (w + 21))
        for kind, s in sorted(tot.items(), key=lambda kv: -kv[1]):
            _original_print(f"  {kind:<8} {'TOTAL':<{w}}  {s:8.2f}s")
        grand = sum(tot.values())
        _original_print(f"  {'':<8} {'WALL':<{w}}  {grand:8.2f}s")


PHASES = Phases()

from user_dma_core import (  # noqa: E402  (off-limits to edit)
    DMA_DEVICE_C2H, DMA_DEVICE_H2C, UE_MODE, UE_VECTOR_SIZE,
    URAM_NEAR_FULL_ELEMENTS, UnifiedEngine,
    ue_35bit_addr_shifter,
)
from nn_lib import (  # noqa: E402
    smart_bf16_permute_core, store_weight, eltwise_add_core_dram,
    eltwise_mul_core_dram, silu_core_dram, store_identity_matrix,
)


# ======================================================================================
# config
# ======================================================================================

def _load_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "verapulse_config.json")
    with open(path) as f:
        return json.load(f)


_CFG = _load_config()


# ======================================================================================
# checkpoint: minimal fetch + canonical naming
# ======================================================================================
#
# No .npy export stage, no separate module: weight_init reads the safetensors state
# dict directly and quantizes on the fly, exactly like smolvlm2_test.py reads the HF
# model directly. params.bin is the only weight artifact.

HF_REPO = _CFG["paths"]["hf_model_repo"]          # verapulse/pulsevla-libero-0.5b
HF_FILES = _CFG["paths"]["hf_files"]              # config/model/norm_stats/tokenizer


CKPT_DIR_NAME = HF_REPO.replace("/", "__")   # verapulse__pulsevla-libero-0.5b
CKPT_NUM_TENSORS = 787                       # exact count; a mismatch means the mapping
                                             # silently dropped or aliased a weight


def _ckpt_dir(script_dir=None):
    script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, _CFG["paths"]["bin_dir"], CKPT_DIR_NAME)


def ensure_checkpoint(script_dir=None):
    """huggingface_hub.snapshot_download(HF_REPO, allow_patterns=HF_FILES) into
    <script_dir>/verapulse_bin/<repo>/ so the checkpoint travels with the model dir
    instead of landing in the machine-global HF cache. Idempotent. Returns the dir."""
    target = _ckpt_dir(script_dir)
    # Idempotent by CONTENT, not by directory existence: a half-finished download
    # leaves the dir behind, and a missing model.safetensors must re-fetch rather
    # than fail later inside safetensors with an unrelated error.
    if all(os.path.exists(os.path.join(target, f)) for f in HF_FILES):
        return target
    from huggingface_hub import snapshot_download
    os.makedirs(target, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO, allow_patterns=list(HF_FILES),
        local_dir=target,
        # keep the real files in the model dir, not symlinks into ~/.cache: the bins
        # and the checkpoint are meant to travel together.
        local_dir_use_symlinks=False)
    missing = [f for f in HF_FILES if not os.path.exists(os.path.join(target, f))]
    if missing:
        raise RuntimeError(f"snapshot_download({HF_REPO}) left files missing: {missing}")
    return target


# Canonical renaming table. Explicit and total: every upstream name must match exactly
# one rule or load_state_dict RAISES. A catch-all "strip the prefix and hope" rename is
# the known corruption hazard here -- it maps an unseen tensor onto a plausible-looking
# canonical name, and the model then runs finite-but-wrong with no failure anywhere.
_VLM = "model.vlm_with_expert."


def _canonical_name(k):
    """Upstream safetensors key -> canonical name, or None if unmapped (-> caller raises)."""
    import re

    # --- action head / flow matching: top-level 'model.<proj>.<weight|bias>' ---
    m = re.fullmatch(r"model\.(state_proj|action_in_proj|action_time_mlp_in|"
                     r"action_time_mlp_out|action_out_proj)\.(weight|bias)", k)
    if m:
        return f"head.{m.group(1)}.{m.group(2)}"

    if not k.startswith(_VLM):
        return None
    r = k[len(_VLM):]

    # --- connector + shared token embedding ---
    if r == "connector.modality_projection.proj.weight":
        return "conn.proj.weight"
    if r == "embed_tokens.weight":
        return "lm.embed_tokens.weight"

    # --- vision tower ---
    if r.startswith("vision."):
        v = r[len("vision."):]
        m = re.fullmatch(r"embeddings\.patch_embedding\.(weight|bias)", v)
        if m:
            return f"vis.patch_embed.{m.group(1)}"
        if v == "embeddings.position_embedding.weight":
            return "vis.pos_embed.weight"
        m = re.fullmatch(r"post_layernorm\.(weight|bias)", v)
        if m:
            return f"vis.post_ln.{m.group(1)}"
        # encoder layers keep their LEAF name (q_proj/k_proj/v_proj/out_proj/fc1/fc2/
        # layer_norm1/layer_norm2) -- that is exactly what _weight_init_vision indexes.
        m = re.fullmatch(r"encoder\.layers\.(\d+)\.(?:self_attn|mlp)\."
                         r"(q_proj|k_proj|v_proj|out_proj|fc1|fc2)\.(weight|bias)", v)
        if m:
            return f"vis.{m.group(1)}.{m.group(2)}.{m.group(3)}"
        m = re.fullmatch(r"encoder\.layers\.(\d+)\.(layer_norm1|layer_norm2)\.(weight|bias)", v)
        if m:
            return f"vis.{m.group(1)}.{m.group(2)}.{m.group(3)}"
        return None

    # --- lm (text) and action expert: identical shape, different canonical prefix ---
    for stack, pfx in (("text", "lm"), ("expert", "ae")):
        if not r.startswith(stack + "."):
            continue
        s = r[len(stack) + 1:]
        if s == "norm.weight":
            return f"{pfx}.final_norm.weight"
        m = re.fullmatch(r"layers\.(\d+)\.(?:self_attn|mlp)\."
                         r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight", s)
        if m:
            return f"{pfx}.{m.group(1)}.{m.group(2)}.weight"
        m = re.fullmatch(r"layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.weight", s)
        if m:
            return f"{pfx}.{m.group(1)}.{m.group(2)}.weight"
        return None
    return None


def load_state_dict(script_dir=None):
    """safetensors.torch.load_file(model.safetensors) -> {canonical_name: tensor}.

    Upstream names are [out, in] and prefixed 'model.vlm_with_expert.'; canonicalize to
        vision.encoder.layers.3.self_attn.q_proj.weight -> vis.3.q_proj.weight
        text.layers.7.mlp.gate_proj.weight             -> lm.7.gate_proj.weight
        expert.layers.7.self_attn.o_proj.weight        -> ae.7.o_proj.weight
        connector.modality_projection.proj.weight      -> conn.proj.weight
        model.action_out_proj.weight                   -> head.action_out_proj.weight
    787 tensors, all F32 on disk -> cast at store time.

    Raises on ANY name it cannot map and on any duplicate canonical name -- a silent
    catch-all rename would produce a model that runs, is finite, and is wrong."""
    from safetensors.torch import load_file
    path = os.path.join(ensure_checkpoint(script_dir), "model.safetensors")
    raw = load_file(path)

    sd, unmapped = {}, []
    for k, t in raw.items():
        name = _canonical_name(k)
        if name is None:
            unmapped.append(k)
            continue
        if name in sd:
            raise KeyError(f"canonical name collision: {k!r} -> {name!r} already taken")
        # fp32 on disk; store_weight casts to bf16 on the way to params DRAM, so keep
        # full precision here and let each _weight_init_* decide its own quantization.
        sd[name] = t.to(torch.float32)

    if unmapped:
        raise KeyError(
            f"{len(unmapped)} checkpoint tensors have no canonical mapping, e.g. "
            f"{unmapped[:5]} -- extend _canonical_name rather than adding a catch-all")
    if len(sd) != CKPT_NUM_TENSORS:
        raise RuntimeError(
            f"mapped {len(sd)} tensors, expected {CKPT_NUM_TENSORS} -- the checkpoint "
            f"or the renaming table drifted")
    return sd


def assert_bf16_only(cfg):
    """BF16 IS A HARDWARE CONSTRAINT HERE, NOT A BRING-UP PREFERENCE: the multipliers
    are 16-bit, so every weight goes to params DRAM as bf16 and every matmat_mul_core
    runs the bf16 B-operand path (no is_B_quantized / data_type=TYPE.IF4 /
    SCALE_DRAM_ADDR). There is deliberately no q4_64 quantizer in this file -- adding
    one would silently route weights onto the IF4 path.

    Called from weight_init so a config edit cannot quietly re-enable quantization."""
    bad = {k: v for k, v in cfg["ops"].items()
           if k.endswith("_quant") and v != "bf16"}
    if bad:
        raise ValueError(
            f"bf16-only model: ops{bad} must all be 'bf16'. The multipliers are 16-bit; "
            f"quantized weights have no supported path in this bring-up.")


# ======================================================================================
# open questions -- flip these to settle them against upstream
# ======================================================================================

@dataclass
class OpenChoices:
    prefix_order: str = "images_text_state"
    """SETTLED against upstream's smolvla/modeling.py::embed_prefix, which appends
    images, then language, then state. Not an open question any more."""

    self_attn_on_even: bool = True
    """expert layer i self-attends when (i % self_attn_every_n_layers == 0)."""

    suffix_rope_continues: bool = True
    """True: suffix positions continue from the prefix. False: restart at arange(50)."""

    rope_positions_cumsum: bool = True
    """True: positions = cumsum(mask)-1, so padded text does not advance the counter."""

    rope_q_on_cross: bool = True
    """whether Q is rotated on cross-attn layers (K comes pre-rotated from the cache)."""


# ======================================================================================
# activations -- true vs what the hardware actually computes
# ======================================================================================

def gelu_tanh(x):
    """SigLIP's gelu_pytorch_tanh: what the MODEL specifies."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def gelu_hw(x):
    """x*sigmoid(1.702x): what the FPGA LALU computes (== CLIP's quick_gelu)."""
    return x * torch.sigmoid(1.702 * x)


# ======================================================================================
# metrics -- the gate every HW stage is scored against
# ======================================================================================

def _align(hw, ref, valid_rows):
    """Common prep: to float64, apply the row mask. FP64 IS LOAD-BEARING, not caution --
    accumulating sums of squares / dot products over ~1e6 elements in fp32 costs ~5e-5 of
    relative accuracy, which is the SAME ORDER as the cos-sim differences we read
    (0.9999 vs 1.0000). Measured: cos(a,a) returned 1.0000486 in fp32 on a 1.5M-element
    tensor -- above 1.0, which is impossible, and proof the low digits were noise."""
    hw = torch.as_tensor(hw).double()
    ref = torch.as_tensor(ref).double()
    if valid_rows is not None:
        hw, ref = hw[valid_rows], ref[valid_rows]
    return hw, ref


def snr_db(hw, ref, valid_rows=None):
    """dB SNR. valid_rows selects the REAL rows; masked/pad rows MUST be excluded or the
    metric reads uniformly broken (measured: identical valid rows read -36.8 dB with pad
    rows included vs +inf without)."""
    hw, ref = _align(hw, ref, valid_rows)
    noise = (hw - ref).pow(2).sum()
    if noise == 0:
        return float("inf")
    return float(10.0 * torch.log10(ref.pow(2).sum() / noise))


def rms(t, valid_rows=None):
    """Root-mean-square. THE ONLY ONE OF THE THREE THAT SEES GAIN: cos-sim is blind to a
    uniform scale factor (measured: cos(10x, x) == 1.000000 exactly), and SNR conflates
    gain error with noise. A systematic gain bug -- e.g. inverting the expert's
    sqrt(480/512) RMSNorm gamma fold -- shows up here and NOWHERE else."""
    t = torch.as_tensor(t).double()
    if valid_rows is not None:
        t = t[valid_rows]
    return float(t.pow(2).mean().sqrt())


def cos_sim(hw, ref, valid_rows=None):
    """Judge long quantized chains (the 10-step denoise) by this, not max-abs rel err.
    Blind to gain by construction -- pair it with rms()."""
    hw, ref = _align(hw, ref, valid_rows)
    return float(F.cosine_similarity(hw.flatten(), ref.flatten(), dim=0))


# ======================================================================================
# PER-STAGE SNR FLOORS -- measured, not guessed.
#
# A flat >=40 dB gate is WRONG for a deep bf16 stack and was rejecting a correct encoder.
# bf16 has an 8-bit mantissa (~0.4% per rounding, ~48 dB from a SINGLE op); 12 or 32
# layers of accumulation cannot hold 40 dB no matter how perfect the emitter is.
#
# Measured on the real checkpoint (fp32 oracle vs bf16, same GELU, identical inputs):
#     vision post_ln (12L)   29.0 dB     <- hardware measured 32.7 dB, i.e. ABOVE this
#     connector              34.8 dB        (real matmul accumulates wider than bf16,
#     prefix hidden (32L)    28.8 dB         so the pure-bf16 sim is pessimistic)
#     prefix KV layer0       43.0 dB
#     prefix KV last         37.2 dB
#     actions (10 steps)     40.1 dB
#
# Floors below are set a few dB UNDER the measured ceiling: a stage at or near its
# ceiling is correct, a stage far below it has a real bug. Cos-sim is the primary
# signal for these deep stacks (see the gemma3 lesson); SNR is the secondary.
# Re-measure with `--stage ref --real` after any change that alters numerics.
# ======================================================================================
SNR_FLOOR = {
    "vision":    26.0,     # ceiling 29.0
    "connector": 31.0,     # ceiling 34.8
    "prefix":    25.0,     # ceiling 28.8
    "prefix_kv": 33.0,     # ceiling 37.2 (last layer; layer0 is 43.0)
    "actions":   36.0,     # ceiling 40.1
    "single_op": 40.0,     # a lone op with no accumulation SHOULD hold 40
}
COS_FLOOR = 0.999          # structural check: catches scrambling that SNR alone may not


def report(name, hw, ref, valid_rows=None, threshold=40.0, expect_pass=True):
    """The ONE comparison format used by every section: SNR, cos-sim, and both RMS values.

    Three metrics because each is blind to something the others catch:
      SNR  -- overall error power; the headline number, but conflates gain with noise
      cos  -- shape/direction; survives scaling, so it isolates structural error
      RMS  -- magnitude; the ONLY one that sees a uniform gain bug (cos(10x,x)==1.0)
    A stage with high cos but mismatched RMS is scaled wrong. High SNR with low cos is
    not possible. Low cos with matched RMS is scrambled, not attenuated.

    `expect_pass=False` marks a NEGATIVE CONTROL -- deliberately-corrupted input the gate
    MUST reject; the tag then reflects whether the gate behaved, not whether the tensors
    matched."""
    s, c = snr_db(hw, ref, valid_rows), cos_sim(hw, ref, valid_rows)
    rh, rr = rms(hw, valid_rows), rms(ref, valid_rows)
    passed = s >= threshold
    tag = ("PASS" if passed else "FAIL") if expect_pass else \
          ("gate OK" if not passed else "GATE BROKEN")
    gain = (rh / rr) if rr else float("nan")
    flag = "  <- GAIN" if (rr and abs(gain - 1.0) > 0.02) else ""
    print(f"  [{tag:^11s}] {name:28s} snr={s:8.2f}dB cos={c:.6f} "
          f"rms={rh:.4f}/{rr:.4f}{flag}")
    return passed if expect_pass else (not passed)


# ======================================================================================
# torch reference -- the oracle every HW stage is scored against, and the --fake path
# ======================================================================================

# ======================================================================================
# STRICT 16-BIT MODE
#
# torch's bf16 matmul ACCUMULATES IN FP32 internally, and this reference additionally
# upcast in rms_norm and softmax. So "bf16 mode" was really bf16-storage /
# fp32-accumulate -- an OPTIMISTIC bound, not a floor. It reported a 32.16 dB ceiling for
# the 32-layer prefix while the hardware measured 20.66 dB, and that 11 dB gap was an
# artifact of the simulation, not necessarily a hardware defect.
#
# With _STRICT16 on, every reduction rounds back to bf16 as it goes:
#   - matmul accumulates in chunks of ACC_CHUNK along K, rounding to bf16 after each
#     chunk (ACC_CHUNK=64 mirrors the 64-wide vector the hardware reduces over)
#   - softmax runs in bf16
#   - rms_norm / layer_norm reductions stay in bf16
# That is a genuine 16-bit floor. It is SLOW (a python loop over K/64 chunks), so it is
# opt-in via --strict16.
# ======================================================================================
_STRICT16 = False
ACC_CHUNK = 64


@contextlib.contextmanager
def strict16(on=True):
    global _STRICT16
    prev, _STRICT16 = _STRICT16, on
    try:
        yield
    finally:
        _STRICT16 = prev


def mm(a, b):
    """Matmul that respects _STRICT16. Outside strict mode this is a plain `a @ b`."""
    if not _STRICT16 or a.dtype != torch.bfloat16:
        return a @ b
    K = a.shape[-1]
    out = None
    for i in range(0, K, ACC_CHUNK):
        part = (a[..., i:i + ACC_CHUNK] @ b[..., i:i + ACC_CHUNK, :]).to(torch.bfloat16)
        out = part if out is None else (out + part).to(torch.bfloat16)
    return out


def _red(x):
    """Reduction dtype: fp32 normally, bf16 under strict mode."""
    return x if (_STRICT16 and x.dtype == torch.bfloat16) else x.float()


def rms_norm(x, gamma, eps):
    var = _red(x).pow(2).mean(-1, keepdim=True)
    return (_red(x) * torch.rsqrt(var + eps)).to(x.dtype) * gamma


def layer_norm(x, gamma, beta, eps):
    return F.layer_norm(x, (x.shape[-1],), gamma, beta, eps)


def rope_tables(positions, head_dim, theta):
    """Indexed by position VALUE, so data-dependent positions change only the indexing."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    f = torch.outer(positions.float(), inv)
    return torch.cat([f.cos(), f.cos()], -1), torch.cat([f.sin(), f.sin()], -1)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], -1)


def apply_rope(x, cos, sin):
    """Tables are built in fp32 for accuracy, then cast to x's dtype so a bf16 forward
    stays bf16 end to end. Without the cast, q/k silently upcast to fp32 while v stays
    bf16 -- which both breaks the matmul and would have made any bf16 measurement a lie."""
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return x * cos.unsqueeze(0) + rotate_half(x) * sin.unsqueeze(0)


def attend(q, k, v, bias=None):
    """q [hq,s,d], k/v [hkv,t,d] -> [s, hq*d]. GQA via repeat_interleave == the hardware's
    duplicate_gqa_rows."""
    hq, s, d = q.shape
    hkv = k.shape[0]
    if hkv != hq:
        rep = hq // hkv
        k, v = k.repeat_interleave(rep, 0), v.repeat_interleave(rep, 0)
    scores = mm(q, k.transpose(-1, -2)) / math.sqrt(d)
    if bias is not None:
        scores = scores + bias.to(scores.dtype)
    # softmax in fp32 then back: the hardware flash kernel also accumulates the
    # reduction wider than bf16, so doing it in bf16 here would understate the ceiling.
    p = (scores.softmax(-1) if (_STRICT16 and scores.dtype == torch.bfloat16)
         else scores.float().softmax(-1).to(v.dtype))
    return mm(p, v).transpose(0, 1).reshape(s, hq * d)


def pixel_shuffle(x, scale):
    """[1024,768] scale 4 -> [64,12288]. Transposes token order -- the half no weight
    permutation can absorb, hence smart_bf16_permute_core on device."""
    seq, embed = x.shape
    h = w = int(seq ** 0.5)
    x = x.view(h, w, embed).view(h, w // scale, embed * scale).permute(1, 0, 2)
    x = x.reshape(w // scale, h // scale, embed * scale * scale).permute(1, 0, 2)
    return x.reshape(seq // (scale * scale), embed * scale * scale)


def sincos_time_embed(t, dim, min_period, max_period):
    """lerobot/openpi create_sinusoidal_pos_embedding: log-spaced periods, sin THEN cos."""
    frac = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float64)
    period = min_period * (max_period / min_period) ** frac
    ang = float(t) * (1.0 / period) * 2.0 * math.pi
    return torch.cat([torch.sin(ang), torch.cos(ang)]).float()


def build_time_table(cfg, pad_to=None):
    """Host-build the [N, pad] sincos timestep table. THE ONLY place this math lives --
    the engine DMAs the result and the reference reports it, so the two cannot drift.

    Schedule matches the Euler loop exactly: t = 1.0, dt = -1/N (so 1.0, 0.9 ... 0.1 for
    N=10). Periods log-spaced min_period -> max_period; sin FIRST then cos (concat, not
    interleave) -- lerobot/openpi create_sinusoidal_pos_embedding order.

    Computed in fp64 and cast once: input-independent, hence exact constants. There is no
    sine or cosine primitive in the ISA to lower it onto anyway.

    Returns (table [N, pad] bf16, ts [N] list). pad_to widens rows past `dim` with zeros
    so a row DMAs into the padded expert slot without a sub-row copy; those lanes must
    STAY zero or the expert's RMSNorm gamma fold breaks."""
    HEAD = cfg["action_head"]
    N, dim = HEAD["num_denoise_steps"], HEAD["time_embed"]["dim"]
    min_p, max_p = HEAD["time_embed"]["min_period"], HEAD["time_embed"]["max_period"]
    pad = pad_to or dim
    frac = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float64)
    period = min_p * (max_p / min_p) ** frac
    dt, rows, ts, t = -1.0 / N, [], [], 1.0
    for _ in range(N):
        sinusoid = t * (1.0 / period) * 2 * math.pi
        row = torch.zeros(pad, dtype=torch.float64)
        row[:dim] = torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)])
        rows.append(row)
        ts.append(t)
        t += dt
    return torch.stack(rows).to(torch.bfloat16).contiguous(), ts


def fake_state_dict(cfg, tiny=False, seed=0):
    """Correctly-shaped random tensors under the canonical names, scaled 1/sqrt(fan_in)
    so activations stay O(1) through 32 layers -- a run that NaNs teaches nothing."""
    g = torch.Generator().manual_seed(seed)
    sd = {}

    def w(name, *shape):
        sd[name] = torch.randn(*shape, generator=g) / math.sqrt(shape[-1])

    # BIASES AND NORM SCALES ARE RANDOM, NOT ones/zeros. They used to be exactly
    # ones() and zeros(), which made the dummy run BLIND to the entire bias path:
    # bias_mode="broadcast_N" added zero and BETA_DRAM_ADDR added zero on every call,
    # so any bug in applying them was invisible. The vision tower is the only stack
    # WITH biases (q/k/v/out_proj, fc1/fc2, patch_embed, LN beta -- the LM and expert
    # are bias-free RMSNorm), and it passed on dummy weights while failing structurally
    # on real ones. A dummy fixture that cannot fail the way production fails is worse
    # than no fixture: it manufactures confidence.
    #
    # Kept small (0.1 sigma) and gamma centred on 1.0 so activations stay O(1) through
    # 32 layers -- the point is to EXERCISE the path, not to stress it.
    def ones(name, n):
        sd[name] = 1.0 + 0.1 * torch.randn(n, generator=g)

    def zeros(name, n):
        sd[name] = 0.1 * torch.randn(n, generator=g)

    V, L, E = cfg["vision"], cfg["lm"], cfg["expert"]
    C, HEAD = cfg["connector"], cfg["action_head"]
    vL, lL, eL = ((2, 2, 2) if tiny else
                  (V["num_layers"], L["num_layers"], E["num_layers"]))

    w("vis.patch_embed.weight", V["hidden_size"], V["num_channels"] * V["patch_size"] ** 2)
    zeros("vis.patch_embed.bias", V["hidden_size"])
    w("vis.pos_embed.weight", V["num_patches"], V["hidden_size"])
    for i in range(vL):
        for p in ("q_proj", "k_proj", "v_proj", "out_proj"):
            w(f"vis.{i}.{p}.weight", V["hidden_size"], V["hidden_size"])
            zeros(f"vis.{i}.{p}.bias", V["hidden_size"])
        w(f"vis.{i}.fc1.weight", V["intermediate_size"], V["hidden_size"])
        zeros(f"vis.{i}.fc1.bias", V["intermediate_size"])
        w(f"vis.{i}.fc2.weight", V["hidden_size"], V["intermediate_size"])
        zeros(f"vis.{i}.fc2.bias", V["hidden_size"])
        for n in ("layer_norm1", "layer_norm2"):
            ones(f"vis.{i}.{n}.weight", V["hidden_size"])
            zeros(f"vis.{i}.{n}.bias", V["hidden_size"])
    ones("vis.post_ln.weight", V["hidden_size"])
    zeros("vis.post_ln.bias", V["hidden_size"])

    w("conn.proj.weight", C["output_size"], C["input_size"])

    w("lm.embed_tokens.weight", L["vocab_size"], L["hidden_size"])
    kv = L["num_kv_heads"] * L["head_dim"]
    for i in range(lL):
        w(f"lm.{i}.q_proj.weight", L["num_heads"] * L["head_dim"], L["hidden_size"])
        w(f"lm.{i}.k_proj.weight", kv, L["hidden_size"])
        w(f"lm.{i}.v_proj.weight", kv, L["hidden_size"])
        w(f"lm.{i}.o_proj.weight", L["hidden_size"], L["num_heads"] * L["head_dim"])
        w(f"lm.{i}.gate_proj.weight", L["intermediate_size"], L["hidden_size"])
        w(f"lm.{i}.up_proj.weight", L["intermediate_size"], L["hidden_size"])
        w(f"lm.{i}.down_proj.weight", L["hidden_size"], L["intermediate_size"])
        ones(f"lm.{i}.input_layernorm.weight", L["hidden_size"])
        ones(f"lm.{i}.post_attention_layernorm.weight", L["hidden_size"])
    ones("lm.final_norm.weight", L["hidden_size"])

    ekv = E["num_kv_heads"] * E["head_dim"]
    for i in range(eL):
        w(f"ae.{i}.q_proj.weight", E["q_out"], E["hidden_size"])
        w(f"ae.{i}.k_proj.weight", ekv, E["hidden_size"])
        w(f"ae.{i}.v_proj.weight", ekv, E["hidden_size"])
        w(f"ae.{i}.o_proj.weight", E["hidden_size"], E["q_out"])
        w(f"ae.{i}.gate_proj.weight", E["intermediate_size"], E["hidden_size"])
        w(f"ae.{i}.up_proj.weight", E["intermediate_size"], E["hidden_size"])
        w(f"ae.{i}.down_proj.weight", E["hidden_size"], E["intermediate_size"])
        ones(f"ae.{i}.input_layernorm.weight", E["hidden_size"])
        ones(f"ae.{i}.post_attention_layernorm.weight", E["hidden_size"])
    ones("ae.final_norm.weight", E["hidden_size"])

    w("head.state_proj.weight", L["hidden_size"], HEAD["state_dim"])
    zeros("head.state_proj.bias", L["hidden_size"])
    w("head.action_in_proj.weight", E["hidden_size"], HEAD["max_action_dim"])
    zeros("head.action_in_proj.bias", E["hidden_size"])
    w("head.action_time_mlp_in.weight", E["hidden_size"], 2 * E["hidden_size"])
    zeros("head.action_time_mlp_in.bias", E["hidden_size"])
    w("head.action_time_mlp_out.weight", E["hidden_size"], E["hidden_size"])
    zeros("head.action_time_mlp_out.bias", E["hidden_size"])
    w("head.action_out_proj.weight", HEAD["max_action_dim"], E["hidden_size"])
    zeros("head.action_out_proj.bias", HEAD["max_action_dim"])
    return sd


class VeraPulseRef:
    """Pure-torch oracle. Runs with real weights, or with --fake dummies for plumbing."""

    def __init__(self, sd, cfg=None, choices=None, hw_gelu=False, tiny=False):
        self.sd = sd
        self.cfg = cfg or _CFG
        self.ch = choices or OpenChoices()
        self.gelu = gelu_hw if hw_gelu else gelu_tanh
        # Both default TRUE = upstream's actual expert. Flip to False only to attribute
        # error between the two faults; the device does neither today.
        self.expert_kv_concat = True        # fault 4: prefix K/V into self-attn layers
        self.expert_cross_reproject = True  # fault 5: (320,320) reprojection on cross
        self.expert_causal_suffix = True    # fault 6: causal mask over the action chunk
        V, L, E = self.cfg["vision"], self.cfg["lm"], self.cfg["expert"]
        self.n_vis = 2 if tiny else V["num_layers"]
        self.n_lm = 2 if tiny else L["num_layers"]
        self.n_ae = 2 if tiny else E["num_layers"]

    def mirror_device_expert(self):
        """Turn all three expert fixes OFF, so this reference computes what the EMITTER
        computes today. Use with hw_gelu=True to ask the one question the upstream gate
        cannot: is the accelerator faithfully executing the (wrong) model we gave it?

        If device-vs-mirror is high, the only thing wrong is the model, and landing
        faults 4/5/6 in the emitter will fix it. If it is low, there is a SECOND,
        device-side fault and the emitter work alone will not be enough. Those two
        outcomes call for completely different next moves, which is why the distinction
        is worth a dedicated comparison rather than an assumption.

        THIS MUST TRACK THE EMITTER. As each fault lands on device, flip it back on here,
        or the mirror silently stops mirroring and its number becomes meaningless in a way
        that looks like a hardware regression.
        Current emitter state: ALL FIVE (4,5,6,7,8) landed. So the mirror no longer models
        a deliberately-wrong model -- it is now the CORRECT model, and device-vs-mirror is
        a pure EXECUTION gate: it asks whether the accelerator computes what we asked,
        with the model faults out of the picture and GELU common-mode. It is therefore the
        CEILING on the sharp upstream gate: no model fix can push actions-vs-upstream
        above whatever this reads."""
        self.expert_kv_concat = True         # fault 4: landed
        self.expert_cross_reproject = True   # fault 5: landed
        self.expert_causal_suffix = True     # fault 6: landed
        return self

    def astype(self, dtype):
        """Same weights, different precision. Running an identical forward in fp32 and in
        bf16 and scoring one against the other measures the PRECISION CEILING of each
        stage -- the best SNR the hardware could possibly hit even with a perfect
        emitter. A HW stage scoring near this number is correct; a stage scoring far
        below it has a real bug. Without this, "45 dB" is a number with nothing to
        compare against."""
        out = VeraPulseRef.__new__(VeraPulseRef)
        out.__dict__.update(self.__dict__)
        out.sd = {k: v.to(dtype) for k, v in self.sd.items()}
        out.dtype = dtype
        return out

    @classmethod
    def from_fake(cls, cfg=None, tiny=False, seed=0, **kw):
        cfg = cfg or _CFG
        return cls(fake_state_dict(cfg, tiny=tiny, seed=seed), cfg, tiny=tiny, **kw)

    @classmethod
    def from_checkpoint(cls, path=None, cfg=None, **kw):
        """Real weights. One fixup vs fake_state_dict: the checkpoint stores
        patch_embed.weight as the raw Conv2d kernel [768,3,16,16], while forward_vision
        (like the device) consumes it as a [768, C*P*P] matmul operand. Flatten it here,
        at the single point weights enter, rather than in every consumer."""
        cfg = cfg or _CFG
        sd = load_state_dict(path)
        V = cfg["vision"]
        pw = sd["vis.patch_embed.weight"]
        if pw.dim() == 4:
            sd = dict(sd)
            sd["vis.patch_embed.weight"] = pw.reshape(
                V["hidden_size"], V["num_channels"] * V["patch_size"] ** 2)
        return cls(sd, cfg, **kw)

    def w(self, n):
        return self.sd[n]

    def forward_vision(self, patches, trace=None):
        """trace: optional list; per layer we append (idx, rms, absmax) of the residual
        stream. A healthy ViT drifts smoothly -- a step change or a monotone blow-up
        localizes the bad layer without needing hardware."""
        V = self.cfg["vision"]
        eps, nh, hd = V["layer_norm_eps"], V["num_heads"], V["head_dim"]
        x = mm(patches, self.w("vis.patch_embed.weight").T) + self.w("vis.patch_embed.bias")
        x = x + self.w("vis.pos_embed.weight")[: x.shape[0]]
        for i in range(self.n_vis):
            h = layer_norm(x, self.w(f"vis.{i}.layer_norm1.weight"),
                           self.w(f"vis.{i}.layer_norm1.bias"), eps)
            q, k, v = (mm(h, self.w(f"vis.{i}.{p}.weight").T) + self.w(f"vis.{i}.{p}.bias")
                       for p in ("q_proj", "k_proj", "v_proj"))
            s = h.shape[0]
            q, k, v = (t.view(s, nh, hd).transpose(0, 1) for t in (q, k, v))
            x = x + mm(attend(q, k, v), self.w(f"vis.{i}.out_proj.weight").T) \
                + self.w(f"vis.{i}.out_proj.bias")
            h = layer_norm(x, self.w(f"vis.{i}.layer_norm2.weight"),
                           self.w(f"vis.{i}.layer_norm2.bias"), eps)
            h = self.gelu(mm(h, self.w(f"vis.{i}.fc1.weight").T) + self.w(f"vis.{i}.fc1.bias"))
            x = x + mm(h, self.w(f"vis.{i}.fc2.weight").T) + self.w(f"vis.{i}.fc2.bias")
            if trace is not None:
                trace.append((i, float(x.pow(2).mean().sqrt()), float(x.abs().max())))
        return layer_norm(x, self.w("vis.post_ln.weight"), self.w("vis.post_ln.bias"), eps)

    def forward_connector(self, vis_out):
        return mm(pixel_shuffle(vis_out, self.cfg["connector"]["pixel_shuffle_scale_factor"]),
                  self.w("conn.proj.weight").T)

    def build_prefix(self, vision_tokens, token_ids, state):
        L = self.cfg["lm"]
        st = (mm(state, self.w("head.state_proj.weight").T)
              + self.w("head.state_proj.bias")).unsqueeze(0)
        txt = self.w("lm.embed_tokens.weight")[token_ids]
        # sqrt(hidden) on images and language, NOT on state -- upstream embed_prefix.
        emb_scale = math.sqrt(vision_tokens.shape[-1])
        vision_tokens = vision_tokens * emb_scale
        txt = txt * emb_scale
        order = self.ch.prefix_order
        if order == "state_images_text":
            x = torch.cat([st, vision_tokens, txt], 0)
        elif order == "images_text_state":
            x = torch.cat([vision_tokens, txt, st], 0)
        elif order == "scatter_into_text":
            x = txt.clone()
            slots = (token_ids == L["image_token_id"]).nonzero().flatten()
            n = min(len(slots), vision_tokens.shape[0])
            if n:
                x[slots[:n]] = vision_tokens[:n]
            x = torch.cat([x, st], 0)
        else:
            raise ValueError(order)
        valid = torch.ones(x.shape[0], dtype=torch.bool)
        pos = (valid.long().cumsum(0) - 1 if self.ch.rope_positions_cumsum
               else torch.arange(x.shape[0]))
        return x, valid, pos

    def prefix_bias(self, n):
        """Block-causal additive mask over n UNPADDED prefix rows.

        Upstream's att_masks are [0]*images + [0]*language + [1]*state, and
        make_att_2d_masks turns that into `cumsum[j] <= cumsum[i]`: images and language
        attend among themselves but NOT to the state row; the state row attends to all.
        With order images->text->state the state row is the last one, so this is a single
        masked column. Returns None for orders where state is not last."""
        if self.ch.prefix_order != "images_text_state" or n < 2:
            return None
        b = torch.zeros(n, n, dtype=torch.float32)
        b[:n - 1, n - 1] = float("-inf")
        return b

    def forward_prefix(self, x, positions, bias="auto", trace=None):
        """bias defaults to "auto" = the block-causal mask this model actually uses.

        IT DEFAULTS ON DELIBERATELY. When the mask was opt-in, forward_full passed it and
        _CpuBackend -- the path LIBERO actually runs -- did not, so the closed-loop
        policy silently ran a fully-bidirectional prefix while the gate that was supposed
        to catch that ran the correct one. A correctness detail that every call site has
        to remember is a correctness detail that will be forgotten. Pass bias=None to
        opt OUT explicitly, or an explicit tensor to override."""
        if isinstance(bias, str):
            assert bias == "auto", f"bias must be a tensor, None, or 'auto'; got {bias!r}"
            bias = self.prefix_bias(x.shape[0])
        """Returns (hidden, kv_cache). kv_cache[i] = (k,v) at [5,S,64] -- the expert's
        cross-attention source, and on HW the thing that must survive all 10 steps.

        trace: optional list; per layer we append (idx, rms, absmax, k_rms, v_rms). The
        KV stats matter more than the hidden state here -- the cache is the expert's
        ONLY input, so a layer whose K/V collapses or explodes breaks the action head
        even while the hidden stream looks fine."""
        L = self.cfg["lm"]
        nh, nkv, hd, eps = L["num_heads"], L["num_kv_heads"], L["head_dim"], L["rms_norm_eps"]
        cos, sin = rope_tables(positions, hd, L["rope_theta"])
        s, cache = x.shape[0], []
        for i in range(self.n_lm):
            h = rms_norm(x, self.w(f"lm.{i}.input_layernorm.weight"), eps)
            q = (mm(h, self.w(f"lm.{i}.q_proj.weight").T)).view(s, nh, hd).transpose(0, 1)
            k = (mm(h, self.w(f"lm.{i}.k_proj.weight").T)).view(s, nkv, hd).transpose(0, 1)
            v = (mm(h, self.w(f"lm.{i}.v_proj.weight").T)).view(s, nkv, hd).transpose(0, 1)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            cache.append((k, v))
            x = x + mm(attend(q, k, v, bias), self.w(f"lm.{i}.o_proj.weight").T)
            h = rms_norm(x, self.w(f"lm.{i}.post_attention_layernorm.weight"), eps)
            g = F.silu(mm(h, self.w(f"lm.{i}.gate_proj.weight").T))
            u = mm(h, self.w(f"lm.{i}.up_proj.weight").T)
            x = x + mm((g * u), self.w(f"lm.{i}.down_proj.weight").T)
            if trace is not None:
                trace.append((i, float(x.pow(2).mean().sqrt()), float(x.abs().max()),
                              float(k.pow(2).mean().sqrt()), float(v.pow(2).mean().sqrt())))
        return rms_norm(x, self.w("lm.final_norm.weight"), eps), cache

    def _suffix_embed(self, x_t, t):
        """On HW the concat is split-column (480 at [0:480], time at [512:992]) because of
        the 480->512 pad; here it is a plain concat -- a LAYOUT difference, not a math
        one, and it must not change this reference."""
        HEAD = self.cfg["action_head"]
        a = mm(x_t, self.w("head.action_in_proj.weight").T) + self.w("head.action_in_proj.bias")
        te = sincos_time_embed(t, HEAD["time_embed"]["dim"],
                               HEAD["time_embed"]["min_period"],
                               HEAD["time_embed"]["max_period"])
        # same rule as apply_rope: the sincos table is built in fp64/fp32 for exactness
        # (it is a compile-time constant) but must enter the forward in the activation
        # dtype, or a bf16 run silently upcasts here and overstates the bf16 ceiling.
        te = te.to(a.dtype)
        h = torch.cat([a, te.unsqueeze(0).expand(a.shape[0], -1)], -1)
        h = mm(h, self.w("head.action_time_mlp_in.weight").T) + self.w("head.action_time_mlp_in.bias")
        h = F.silu(h)
        return mm(h, self.w("head.action_time_mlp_out.weight").T) \
            + self.w("head.action_time_mlp_out.bias")

    def forward_expert(self, x, prefix_kv, prefix_len, bias_self=None, bias_cross=None):
        E = self.cfg["expert"]
        nh, nkv, hd = E["num_heads"], E["num_kv_heads"], E["head_dim"]
        eps, every = E["rms_norm_eps"], E["self_attn_every_n_layers"]
        s = x.shape[0]
        pos = (torch.arange(prefix_len, prefix_len + s)
               if self.ch.suffix_rope_continues else torch.arange(s))
        cos, sin = rope_tables(pos, hd, self.cfg["lm"]["rope_theta"])
        for i in range(self.n_ae):
            is_self = (i % every == 0) == self.ch.self_attn_on_even
            h = rms_norm(x, self.w(f"ae.{i}.input_layernorm.weight"), eps)
            q = (mm(h, self.w(f"ae.{i}.q_proj.weight").T)).view(s, nh, hd).transpose(0, 1)
            if is_self:
                k = (mm(h, self.w(f"ae.{i}.k_proj.weight").T)).view(s, nkv, hd).transpose(0, 1)
                v = (mm(h, self.w(f"ae.{i}.v_proj.weight").T)).view(s, nkv, hd).transpose(0, 1)
                q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
                if self.expert_kv_concat:
                    # FAULT 4. Upstream's self-attn layers PREPEND the cached prefix K/V:
                    #     k = torch.cat([kv[idx]["key_states"], k], dim=1)
                    # so the expert attends over prefix_len+chunk keys, not chunk. The
                    # cached K is already roped with PREFIX positions and the expert's
                    # own K with suffix positions, which is consistent because the suffix
                    # positions continue from the prefix. Without this the expert's
                    # self-attention layers never see the observation at all.
                    pk, pv = prefix_kv[i]
                    k = torch.cat([pk, k], dim=1)
                    v = torch.cat([pv, v], dim=1)
                a = attend(q, k, v, bias_self)
            else:
                pk, pv = prefix_kv[i]
                if self.expert_cross_reproject:
                    # FAULT 5. Upstream RE-PROJECTS the cached VLM K/V through the
                    # expert's own k_proj/v_proj before attending. The checkpoint proves
                    # it: odd-layer k_proj is (320,320) -- it reads the VLM kv width --
                    # while even-layer k_proj is (320,480), the expert hidden width.
                    S = pk.shape[1]
                    kf = pk.transpose(0, 1).reshape(S, nkv * hd)     # [S, 320]
                    vf = pv.transpose(0, 1).reshape(S, nkv * hd)
                    k = mm(kf, self.w(f"ae.{i}.k_proj.weight").T).view(S, nkv, hd).transpose(0, 1)
                    v = mm(vf, self.w(f"ae.{i}.v_proj.weight").T).view(S, nkv, hd).transpose(0, 1)
                else:
                    k, v = pk, pv
                if self.ch.rope_q_on_cross:
                    # Upstream rebases the query positions to 0..chunk-1 on cross layers
                    # (exp_pos - exp_pos.min()) and does NOT rope the reprojected K.
                    cq, sq = rope_tables(torch.arange(s), hd, self.cfg["lm"]["rope_theta"])
                    q = apply_rope(q, cq, sq)
                a = attend(q, k, v, bias_cross)
            x = x + mm(a, self.w(f"ae.{i}.o_proj.weight").T)
            h = rms_norm(x, self.w(f"ae.{i}.post_attention_layernorm.weight"), eps)
            g = F.silu(mm(h, self.w(f"ae.{i}.gate_proj.weight").T))
            u = mm(h, self.w(f"ae.{i}.up_proj.weight").T)
            x = x + mm((g * u), self.w(f"ae.{i}.down_proj.weight").T)
        return rms_norm(x, self.w("ae.final_norm.weight"), eps)

    def denoise(self, prefix_kv, prefix_len, noise=None, trace=False,
                pad_to=None, prefix_pad_to=None):
        """Euler integration of the flow.

        MASKS. On hardware the suffix is CHUNK real rows padded to SUFFIX_LEN_PAD and the
        prefix is padded to PREFILL_MAX_SEQ_LEN, and _emit_expert_layer passes a
        COLUMN-ONLY mask to every flash call (pad rows are computed and simply unread).
        This reference is the oracle the hardware gate scores against, so it must carry
        the same masks -- running it unmasked lets pad columns into the softmax on ONE
        side of the comparison and the error compounds over 10 Euler steps.

        pad_to / prefix_pad_to are the PADDED widths. When they are None the caller is
        working with unpadded tensors (the pure-CPU reference stages), and no mask is
        needed because there is no padding to mask.
        """
        HEAD = self.cfg["action_head"]
        n, chunk, dim = HEAD["num_denoise_steps"], HEAD["chunk_size"], HEAD["max_action_dim"]
        x_t = noise if noise is not None else torch.randn(chunk, dim)
        dt, steps = -1.0 / n, []
        rows = x_t.shape[0]
        # NO SELF MASK. The device runs SUFFIX_LEN_PAD=64 rows and masks the 14 pad
        # columns; this reference runs exactly CHUNK=50 real rows and has no padding to
        # mask, so masked-64 and unpadded-50 compute the SAME thing. `pad_to` is accepted
        # for symmetry and to document the asymmetry, not used -- passing a [50,64] mask
        # onto [15,50,50] scores is a shape error, which is how this was caught.
        # FAULT 6: the suffix self-attention is CAUSAL, not bidirectional.
        # embed_suffix returns att_masks = ones(chunk), and make_att_2d_masks turns that
        # into cumsum = [1..50] and att_2d[i,j] = (cumsum[j] <= cumsum[i]) = (j <= i).
        # So action token i attends to prefix (all of it) plus actions 0..i only. The
        # prefix block stays fully visible: denoise_step builds
        #     full_2d = cat([prefix_pad expanded, make_att_2d_masks(suffix)], dim=2)
        # Getting this wrong does not NaN -- it just lets each action peek at later ones,
        # which inflates nothing and quietly changes the whole chunk.
        if self.expert_causal_suffix:
            causal = torch.zeros(rows, rows)
            causal[torch.triu(torch.ones(rows, rows, dtype=torch.bool),
                              diagonal=1)] = float("-inf")
        else:
            causal = torch.zeros(rows, rows)         # device-mirror: bidirectional
        if self.expert_kv_concat:
            b_self = torch.zeros(rows, prefix_len + rows)
            b_self[:, prefix_len:] = causal          # prefix cols all visible
        else:
            b_self = causal
        assert pad_to is None or rows <= pad_to, f"rows {rows} > pad_to {pad_to}"
        b_cross = None
        if prefix_pad_to is not None and prefix_len < prefix_pad_to:
            b_cross = torch.zeros(rows, prefix_pad_to)
            b_cross[:, prefix_len:] = float("-inf")
        for i in range(n):
            h = self._suffix_embed(x_t, 1.0 + i * dt)
            v = self.forward_expert(h, prefix_kv, prefix_len,
                                    bias_self=b_self, bias_cross=b_cross)
            v = mm(v, self.w("head.action_out_proj.weight").T) + self.w("head.action_out_proj.bias")
            x_t = x_t + dt * v
            if trace:
                steps.append(x_t.clone())
        return (x_t, steps) if trace else x_t

    def forward_full(self, patches_per_cam, state, token_ids, noise=None):
        """Returns EVERY stage tensor so the HW harness scores each independently."""
        out = {}
        vis = [self.forward_vision(p) for p in patches_per_cam]
        out["vision"] = torch.stack(vis)
        toks = torch.cat([self.forward_connector(v) for v in vis], 0)
        out["connector"] = toks
        x, valid, pos = self.build_prefix(toks, token_ids, state)
        out["prefix_in"], out["valid_mask"], out["positions"] = x, valid, pos
        hidden, kv = self.forward_prefix(x, pos)   # bias="auto" -> block-causal
        out["prefix_hidden"], out["prefix_kv"] = hidden, kv
        acts = self.denoise(kv, x.shape[0], noise=noise)
        out["actions_padded"] = acts
        out["actions"] = acts[:, : self.cfg["action_head"]["action_dim"]]
        return out


def fake_inputs(cfg, tiny=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    V, L, HEAD = cfg["vision"], cfg["lm"], cfg["action_head"]
    npatch = 64 if tiny else V["num_patches"]
    kdim = V["num_channels"] * V["patch_size"] ** 2
    return (torch.randn(V["num_image_slots"], npatch, kdim, generator=g),
            torch.randn(HEAD["state_dim"], generator=g),
            torch.randint(0, L["vocab_size"], (L["tokenizer_max_length"],), generator=g),
            torch.randn(HEAD["chunk_size"], HEAD["max_action_dim"], generator=g))


# ======================================================================================
# main engine
# ======================================================================================

class VeraPulse_UnifiedEngine(UnifiedEngine):
    _cfg = _CFG

    # ---- vision (MHA) ----
    V_HIDDEN     = _cfg["vision"]["hidden_size"]          # 768
    V_INTER      = _cfg["vision"]["intermediate_size"]    # 3072
    V_LAYERS     = _cfg["vision"]["num_layers"]           # 12
    V_HEADS      = _cfg["vision"]["num_heads"]            # 12   kv == q
    V_HEAD_DIM   = _cfg["vision"]["head_dim"]             # 64
    V_PATCHES    = _cfg["vision"]["num_patches"]          # 1024
    V_PATCH      = _cfg["vision"]["patch_size"]           # 16
    V_SLOTS      = _cfg["vision"]["num_image_slots"]      # 2 cameras (confirmed upstream)
    V_LN_EPS     = _cfg["vision"]["layer_norm_eps"]

    # ---- connector ----
    C_IN         = _cfg["connector"]["input_size"]        # 12288
    C_OUT        = _cfg["connector"]["output_size"]       # 960
    C_TOKENS     = _cfg["connector"]["tokens_out"]        # 64
    C_SHUFFLE    = _cfg["connector"]["pixel_shuffle_scale_factor"]  # 4

    # ---- lm / prefix (GQA g3) ----
    HIDDEN_SIZE       = _cfg["lm"]["hidden_size"]         # 960
    INTERMEDIATE_SIZE = _cfg["lm"]["intermediate_size"]   # 2560
    NUM_LAYERS        = _cfg["lm"]["num_layers"]          # 32
    NUM_HEADS         = _cfg["lm"]["num_heads"]           # 15
    NUM_KV_HEADS      = _cfg["lm"]["num_kv_heads"]        # 5
    GROUP_SIZE        = NUM_HEADS // NUM_KV_HEADS         # 3 -> duplicate_gqa_rows
    HEAD_DIM          = _cfg["lm"]["head_dim"]            # 64  -> rope half 32, 32-aligned
    VOCAB_SIZE        = _cfg["lm"]["vocab_size"]          # 49280
    ROPE_THETA        = _cfg["lm"]["rope_theta"]          # 10000.0 (verapulse's own value)
    RMS_NORM_EPS      = _cfg["lm"]["rms_norm_eps"]
    IMAGE_TOKEN_ID    = _cfg["lm"]["image_token_id"]      # 49190
    TEXT_LEN          = _cfg["lm"]["tokenizer_max_length"]  # 48

    # ---- expert (GQA g3) ----
    E_HIDDEN     = _cfg["expert"]["hidden_size"]          # 480  (NOT %64)
    E_HIDDEN_PAD = _cfg["expert"]["hidden_size_padded"]   # 512
    E_INTER      = _cfg["expert"]["intermediate_size"]    # 1280
    E_LAYERS     = _cfg["expert"]["num_layers"]           # 32
    E_Q_OUT      = _cfg["expert"]["q_out"]                # 960
    E_KV_OUT     = _cfg["expert"]["kv_out"]               # 320 == lm KV width
    E_SELF_EVERY = _cfg["expert"]["self_attn_every_n_layers"]  # 2

    # ---- head / flow ----
    STATE_DIM    = _cfg["action_head"]["state_dim"]       # 32
    ACTION_DIM   = _cfg["action_head"]["action_dim"]      # 7
    ACTION_PAD   = _cfg["action_head"]["max_action_dim"]  # 32
    CHUNK        = _cfg["action_head"]["chunk_size"]      # 50
    N_STEPS      = _cfg["action_head"]["num_denoise_steps"]  # 10 Euler steps
    # DIFFERENT QUANTITY that coincidentally also equals 10: how many of the predicted
    # chunk actually get executed before the policy re-plans. Do not conflate them --
    # N_STEPS is denoise iterations, N_ACTION_STEPS is the execution horizon.
    N_ACTION_STEPS = _cfg["action_head"]["n_action_steps"]   # 10
    TIME_DIM     = _cfg["action_head"]["time_embed"]["dim"]  # 480
    TIME_MIN     = _cfg["action_head"]["time_embed"]["min_period"]
    TIME_MAX     = _cfg["action_head"]["time_embed"]["max_period"]
    # max_action_dim 32 is below the 64-ALU floor / 128 B SRAM row, so both the action
    # matmul's K side and its N side run in a 64-lane slot with lanes [32:64] zeroed by
    # the stored weights. Derived, never hardcoded: a checkpoint with action_dim > 64
    # would need a real tiling, and this expression makes that visible.
    ACTION_DIM_PAD = ((_cfg["action_head"]["max_action_dim"] + UE_VECTOR_SIZE - 1)
                      // UE_VECTOR_SIZE) * UE_VECTOR_SIZE     # 64

    # ---- expert layer parity / cross-attention layer mapping -------------------------
    # BOTH ARE UNVERIFIED (see OpenChoices + config _OPEN_layer_parity/_OPEN_expert_rope).
    # They are read from the config, with the documented default, so settling either one
    # against the upstream reference is a one-line config edit and NOT an emitter change.
    #   self_attn_on_even   True  -> layer i self-attends when i % self_attn_every_n == 0
    #   cross_prefix_layer  "same_index" -> cross layer i reads LM layer i's cached K/V
    #                       "last"       -> every cross layer reads the final LM layer
    EXPERT_SELF_ON_EVEN = _cfg["expert"].get("self_attn_on_even", True)
    EXPERT_CROSS_LAYER_MAP = _cfg["expert"].get("cross_prefix_layer_map", "same_index")
    # True: suffix RoPE positions continue from the prefix (SmolVLA), False: restart at 0.
    EXPERT_ROPE_CONTINUES = _cfg["expert"].get("suffix_rope_continues", True)
    # True: Q is rotated on cross-attn layers too (K arrives pre-rotated from the cache).
    EXPERT_ROPE_Q_ON_CROSS = _cfg["expert"].get("rope_q_on_cross", True)

    # ---- sequence layout ----
    PREFIX_LEN         = _cfg["model"]["prefix_len"]        # 177
    PREFILL_MAX_SEQ_LEN = _cfg["model"]["prefix_len_padded"]  # 192 == "PM" below
    SUFFIX_LEN         = _cfg["model"]["suffix_len"]        # 50
    SUFFIX_LEN_PAD     = _cfg["model"]["suffix_len_padded"]  # 64
    VIS_SEQ            = _cfg["model"]["vision_seq_len"]    # 1024

    # Fused (piecewise LALU) vs composed (true x*sigmoid(x)) SiLU in the prefix MLP.
    # See compile_prefix; flip with --fused-silu/--no-fused-silu to measure the delta.
    # Compile only the first N ViT layers (None = all 12). A BISECTION HANDLE: the
    # vision gate only observes post_ln, i.e. the END of the stack, so a divergence
    # anywhere in 12 layers looks identical. Truncating lets us ask "does layer k
    # match?" directly. The oracle is truncated to the same depth, so the comparison
    # stays honest.
    VIS_LAYERS = None

    # Emit per-op probe copies inside the vision program so EVERY intermediate can be
    # read back and scored, not just post_ln. Costs 3 extra buffers + 3 DRAM copies;
    # only enabled by --bisect-vision.
    VIS_BISECT = False

    # Compile only the first N prefix layers (None = all 32), and probe layer N-1.
    # Truncating is what makes the probe cheap: the layer under test becomes the LAST
    # one, so its intermediates are still resident afterwards and only the buffer that
    # is reused WITHIN a layer (LM_PRE_NORM, written by both norms) needs a copy.
    PREFIX_LAYERS = None
    PREFIX_BISECT = False

    # Compile only the first N expert layers of ONE Euler step, and probe layer N-1.
    # Same truncation trick as the prefix bisect: the layer under test becomes the last,
    # so its intermediates survive and only buffers reused WITHIN a layer need a copy.
    EXPERT_LAYERS = None
    EXPERT_BISECT = False

    PREFIX_FUSED_SILU = True

    # Roll the 10 Euler steps into one hardware loop_start/loop_end body (small program)
    # vs statically unrolling them (10x program, but proven).
    #
    # DEFAULT IS FALSE -- STATIC UNROLL -- because pi05 tried the rolled form for this
    # exact stage and reverted: "A loop_start/loop_end runtime loop was tried (single
    # small compiled body, runtime trip count) but hung on any real repeat even after
    # fixing a genuine found bug (missing 64B alignment before the backward-branch jump
    # target); that combination isn't proven anywhere else in this codebase."
    #
    # Our own evidence agrees: _ae_duplicate_gqa_rows itself opens TWO nested hardware
    # loops, so the prefix runs it at depth 2 and completes in 3.5 s, while the rolled
    # denoise puts it at depth 3 and hangs. Depth 3 is the only structurally new thing
    # between a stage that works and one that does not.
    #
    # Cost of unrolling: ~3.1 MB -> ~31 MB of program, against a 2.85 GB arena. Cheap.
    DENOISE_ROLLED = False

    OPS = _cfg["ops"]
    VEC = 64                                              # 64-ALU floor / 128B row

    def __init__(self, script_dir=None, **kw):
        layout = self._cfg["dram_layout"]
        kw.setdefault("params_dram_base",  int(layout["params_dram_base"], 16))
        kw.setdefault("tensor_dram_base",  int(layout["tensor_dram_base"], 16))
        kw.setdefault("program_dram_base", int(layout["program_dram_base"], 16))
        super().__init__(**kw)
        self.script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
        self._programs = {}
        self._debug_counter = 0

    # ==================================================================================
    # 0. helpers
    # ==================================================================================

    @staticmethod
    def _pad_to(x, dim, axis):
        """Zero-pad along `axis` up to `dim`. Every 480->512 and 32->64 pad in this model
        goes through here at STORE time -- the reformat is a weight, never a runtime
        staging copy."""
        x = torch.as_tensor(x)
        cur = x.shape[axis]
        if cur == dim:
            return x
        if cur > dim:
            raise ValueError(f"_pad_to: axis {axis} is {cur}, cannot shrink to {dim}")
        # F.pad's pad list counts axes from the LAST one, two entries per axis
        # (before, after) -- build it explicitly so the intent survives a reader.
        pad = [0] * (2 * x.dim())
        pad[2 * (x.dim() - 1 - (axis % x.dim())) + 1] = dim - cur
        return F.pad(x, pad, value=0.0)

    def _snr(self, hw, cpu, valid_rows):
        """dB SNR. `valid_rows` is REQUIRED, not optional -- pad/masked rows MUST be
        excluded or the metric reads uniformly broken (pi05: -3.4 dB vs +50 dB on the
        same op). It must exclude more than the 15 prefix pad rows: also the padded text
        tokens inside rows 129..176, and the 14 suffix pad rows.

        Positional and mandatory on purpose: there is no default that is safe. Pass an
        explicit all-True mask (or slice(None)) at the one place where every row really
        is valid -- the 1024 vision patches -- so the choice is visible at the call site.
        """
        if valid_rows is None:
            raise ValueError(
                "_snr: valid_rows is required -- masked/pad rows must be excluded "
                "explicitly. Pass torch.ones(M, dtype=bool) if every row is real.")
        return snr_db(hw, cpu, valid_rows)

    # ==================================================================================
    # compile / execute split
    # ==================================================================================

    def compile_timed(self, compile_fn, label):
        """COMPILE-ONCE half. Silences the per-op library dumps and times the build.

        Kept separate from execution because the two have different fixes: compile time
        is paid once and shrinks by reducing shape diversity or rolling loops; execute
        time is paid every inference and shrinks by sharding or cutting DRAM traffic.
        One combined number tells you neither."""
        _original_print(f"  [{label}] compiling...", end="\r", flush=True)
        with PHASES.track(label, "compile"), silenced():
            addr = compile_fn()
        secs = PHASES.rows[-1][2]
        size = getattr(self, "_last_program_bytes", None)
        extra = f"  {size / 1e6:.1f} MB" if size else ""
        _original_print(f"  [{label}] compiled in {secs:.2f}s{extra}" + " " * 24)
        return addr

    def execute_timed(self, prog_addr, label, timeout=250.0):
        """EXECUTE-MANY half: re-run an already-compiled program. No capture, no
        allocation. Inputs must already be DMA'd to their baked addresses -- absolute
        addresses and GPR/PBI state are compiled into the program bytes."""
        with PHASES.track(label, "exec"), silenced():
            self.start_execute_from_dram(prog_addr)
            self._wait_with_heartbeat(label, timeout=timeout)
        _original_print(f"  [{label}] executed in {PHASES.rows[-1][2]:.2f}s")
        return prog_addr

    # ==================================================================================
    # 1. weights
    # ==================================================================================

    def weight_init(self, dummy=False, seed=0):
        """Download if needed, then build all weights into params DRAM in one pass.
        Everything is bf16 (assert_bf16_only enforces it): the multipliers are 16-bit, so
        there is no quantized path to fall onto. No intermediate weight bins -- dump_bins
        captures assembled params DRAM.

        dummy=True swaps the checkpoint for fake_state_dict(cfg, seed=seed): correctly
        shaped random tensors under the exact canonical names the _weight_init_* methods
        index, scaled 1/sqrt(fan_in) so 32 layers do not blow up. That is the PLUMBING
        path -- it exercises every store/emit/execute surface (identical shapes, identical
        program bytes, identical DRAM layout) without needing the 2.23 GB checkpoint on
        disk. It says NOTHING about model fidelity, only that the pipeline runs end to
        end and stays finite.
        The one shape difference between the two sources is patch_embed: the checkpoint
        ships the raw Conv2d kernel [768,3,16,16], fake_state_dict ships it already
        flattened to [768, C*P*P]. _weight_init_vision reshapes unconditionally, and
        reshape is a no-op on the already-flat tensor, so both sources land identically
        -- the same fixup VeraPulseRef.from_checkpoint applies on the oracle side.

        Because the dummy weights are NOT the checkpoint, the oracle every gate scores
        against has to be built from the SAME dummy tensors or every stage reads as
        broken. We pre-seed the _ref_oracle cache here (same seed, same cfg, hw_gelu=True)
        rather than teach each probe about the mode.

        ORDERING HAZARD -- READ BEFORE ADDING AN ENGINE: every extra engine/scheduler MUST
        already exist before this runs. Each UnifiedEngine ctor DMA-writes 16KB of noise
        to a HARDCODED 0x80000000, which is exactly where this model's first weight lands
        (params_dram_base is 0x0, i.e. absolute 0x80000000 -- maximally exposed). Building
        a worker after weight_init silently shreds the head of params: finite, plausible,
        wrong, and invisible to every gate downstream. If it is truly unavoidable,
        snapshot 0x80000000..+64KB into a BFLOAT16 buffer and restore it afterwards --
        dma_read only round-trips bits losslessly for bf16/int32, so any other dtype
        corrupts more than it repairs."""
        assert_bf16_only(self._cfg)     # 16-bit multipliers: no quantized path exists
        self._dummy_weights = bool(dummy)
        if dummy:
            sd = fake_state_dict(self._cfg, seed=seed)
            # Pre-seed the probe oracle so _ref_oracle() does NOT reach for the real
            # checkpoint (which would both download 2.23 GB and score the hardware
            # against weights it is not running).
            self._ref = VeraPulseRef(sd, self._cfg, hw_gelu=True)
            print(f"  weight_init: DUMMY weights (fake_state_dict seed={seed}) -- "
                  f"plumbing only, NOT a fidelity claim")
        else:
            sd = load_state_dict(self.script_dir)
        self._weight_init_vision(sd)
        self._weight_init_connector(sd)
        self._weight_init_lm(sd)
        # RoPE tables live in PARAMS DRAM (store_weight), so they are built here in the
        # weight phase -- not lazily at compile time. compile_prefix reads
        # ROPE_PACKED_DRAM / ROPE_PACKED_GQA_DRAM directly, and omitting this call is an
        # AttributeError at the first prefix compile, after vision has already run.
        self._load_rope_tables()
        self._weight_init_expert(sd)
        self._weight_init_head(sd)
        self._assert_params_fit()

    def _weight_init_vision(self, sd):
        """Store the SigLIP tower in bf16. sd holds fp32 tensors under the canonical
        names; store_weight casts fp32 -> bf16 on the way to params DRAM, so "load in
        bf16" costs nothing extra. NOTHING here is quantized -- see _weight_init_lm for
        why bf16 comes first.

        MHA: q/k/v are all [768,768], 12 heads each -- no GQA replication in the ViT.
        Every projection carries a bias and both LayerNorms carry beta, unlike the
        bias-free lm/expert RMSNorm stacks. All dims already %64; nothing to pad."""
        V = self._cfg["vision"]
        H, P, C = V["hidden_size"], V["patch_size"], V["num_channels"]
        NPS = V["num_patches_per_side"]

        # Conv2d [768,3,16,16] -> [768, 3*16*16=768]. The flatten is CHANNEL-MAJOR
        # (col = c*P*P + kh*P + kw); the on-device permute in compile_encoder feeds it in
        # that order, so no host-side patchify ordering contract is needed.
        self.patch_weight_addr = store_weight(
            self, sd["vis.patch_embed.weight"].reshape(H, C * P * P))
        self.patch_bias_addr = store_weight(self, sd["vis.patch_embed.bias"])

        # SmolVLM buckets fractional patch coords to integer ids. At a fully-populated
        # 32x32 grid this IS arange(1024), but the bucketize keeps a partial grid
        # correct. Gathered ONCE at store time -> the runtime sees a plain eltwise add.
        # PATCH CENTRES, not left edges. The previous form used
        #     frac = arange(NPS)/NPS * (1 - 1e-6)
        # which places coordinate k at k/NPS minus an epsilon -- just BELOW
        # boundaries[k-1] = k/NPS -- so bucketize returned k-1 for every k >= 1.
        # Measured: b = [0,0,1,2,...,30]; row 0 duplicated, row 31 never fetched, only
        # 961 of 1024 position ids distinct. Every patch then received its NEIGHBOUR's
        # position embedding. On hardware that surfaced as the `embed` op diverging
        # (cos 0.895, absmax 5.78 vs 11.97) while `patch` was clean at 52.6 dB --
        # pos_embed dominates the magnitude here (patch absmax 0.72 vs pos ~12), so a
        # shifted lookup wrecks the residual stream at the very first layer.
        #
        # Cell centres (k + 0.5)/NPS sit strictly between neighbouring boundaries and
        # bucketize to k for every k. Asserted rather than trusted: the failure mode is
        # silent -- plausible embeddings, just the wrong ones.
        boundaries = torch.arange(1.0 / NPS, 1.0, 1.0 / NPS, dtype=torch.float32)
        frac = (torch.arange(NPS, dtype=torch.float32) + 0.5) / NPS
        b = torch.bucketize(frac, boundaries, right=True)
        assert torch.equal(b, torch.arange(NPS)), (
            f"patch-coordinate bucketize is not the identity on a full {NPS}x{NPS} grid: "
            f"{b.tolist()[:8]}... -- every patch would get the wrong position embedding")
        pos_ids = (b[:, None] * NPS + b[None, :]).flatten()
        assert torch.equal(pos_ids, torch.arange(NPS * NPS)), (
            "pos_ids must be arange on a fully-populated grid")
        self.pos_embed_addr = store_weight(self, sd["vis.pos_embed.weight"][pos_ids])

        self.vis_layer_addrs = []
        for i in range(V["num_layers"]):
            la = {}
            for tag, name in (("q", "q_proj"), ("k", "k_proj"),
                              ("v", "v_proj"), ("o", "out_proj")):
                la[f"{tag}_weight"] = store_weight(self, sd[f"vis.{i}.{name}.weight"])
                la[f"{tag}_bias"] = store_weight(self, sd[f"vis.{i}.{name}.bias"])
            for tag in ("fc1", "fc2"):
                la[f"{tag}_weight"] = store_weight(self, sd[f"vis.{i}.{tag}.weight"])
                la[f"{tag}_bias"] = store_weight(self, sd[f"vis.{i}.{tag}.bias"])
            for tag, name in (("ln1", "layer_norm1"), ("ln2", "layer_norm2")):
                la[f"{tag}_weight"] = store_weight(self, sd[f"vis.{i}.{name}.weight"])
                la[f"{tag}_bias"] = store_weight(self, sd[f"vis.{i}.{name}.bias"])
            self.vis_layer_addrs.append(la)

        self.vis_post_ln_weight = store_weight(self, sd["vis.post_ln.weight"])
        self.vis_post_ln_bias = store_weight(self, sd["vis.post_ln.bias"])

        # smolvlm2 "Trick 9": layer_norm_core_dram self-allocates + dma_writes its zeros
        # and 1/N operands at COMPILE time, which the bin-load path never re-emits ->
        # stale DRAM / NaN. Seeding them here puts them in params.bin, so build and every
        # load path get them with no replay. All encoder LNs are N=768.
        self.vis_zeros_addr = store_weight(self, torch.zeros(H))
        self.vis_inv_n_addr = store_weight(self, torch.full((H,), 1.0 / H))

        # identity for smart_bf16_permute_core's transpose decomposition
        self.permute_params_addr = store_weight(
            self, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))

    def _weight_init_connector(self, sd):
        """conn.proj.weight[960,12288], stored exactly as shipped -- NOT pre-permuted.
        The pixel shuffle is done on device by smart_bf16_permute_core (natively
        supported, DMA-gather path), so there is nothing to fold into the weight."""
        self.conn_weight_addr = store_weight(self, sd["conn.proj.weight"])

    def _weight_init_lm(self, sd):
        """Prefix (VLM decoder) weights, ALL BF16 -- the multipliers are 16-bit, so unlike
        smolvlm2 (which q4_64s this stack) every projection is stored bf16 and every
        matmul below takes the bf16 B-operand path. Do NOT mirror smolvlm2's quantized
        LM path.

        Layer dict keys deliberately match the expert's (ae_layer_addrs) so _ae_matmul and
        the prefix matmul helper are interchangeable.

        No lm_head: this model has no vocab output. The last thing the prefix produces is
        a hidden state and, far more importantly, the per-layer K/V the action expert
        cross-attends into."""
        L = self._cfg["lm"]
        self.embedding_weight = sd["lm.embed_tokens.weight"].to(torch.bfloat16)
        self.embed_addr = store_weight(self, self.embedding_weight)
        self.lm_layer_addrs = []
        for i in range(L["num_layers"]):
            la = {}
            for tag, name in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"),
                              ("o", "o_proj"), ("gate", "gate_proj"), ("up", "up_proj"),
                              ("down", "down_proj")):
                la[f"{tag}_weight"] = store_weight(self, sd[f"lm.{i}.{name}.weight"])
            la["ln1_gamma"] = store_weight(self, sd[f"lm.{i}.input_layernorm.weight"])
            la["ln2_gamma"] = store_weight(self, sd[f"lm.{i}.post_attention_layernorm.weight"])
            self.lm_layer_addrs.append(la)
        self.final_norm_addr = store_weight(self, sd["lm.final_norm.weight"])
        if not hasattr(self, "identity_addr"):
            self.identity_addr = store_identity_matrix(self)

    def _weight_init_expert(self, sd):
        """Same dict-per-layer shape as the LM (self.ae_layer_addrs), so the expert
        emitter can reuse the lm_matmul helper unchanged.
        q[960,480] k[320,480] v[320,480] o[480,960] gate/up[1280,480] down[480,1280],
        2 norms[480], ae.final_norm[480].

        THE 480 PAD, applied here once in the stored weights:
          - K=480 operands (q,k,v,gate,up): pad K 480 -> 512 with zero rows.
          - N=480 outputs (o_proj, down_proj): pad N 480 -> 512 with zero cols.
          - norm gammas: pad to 512 with zeros AND scale by sqrt(480/512) = 0.968246, so
            RMS over 512 lanes (32 zero) equals RMS over 480. Scale DOWN: dividing by 512
            instead of 480 shrinks the denominator, so the kernel's output is LARGER by
            sqrt(512/480) and gamma must compensate downward. The reciprocal is a silent
            +6.7% gain per norm compounding over 65 expert norms (~1e9x, no NaN).
            Works because RMSNorm has no mean term; exact up to eps (~1e-6 relative,
            three orders below the bf16 quantum). A true LayerNorm could NOT be fixed
            this way -- zero padding shifts the mean, data-dependently.
        Bring the expert up in bf16 first; q4_64 only after it passes >=40 dB."""
        E = self._cfg["expert"]
        H, HP, I = E["hidden_size"], E["hidden_size_padded"], E["intermediate_size"]
        Q, KV = E["q_out"], E["kv_out"]

        # matmat_mul_core's B operand is [N, K] row-major (it applies the implicit
        # transpose and computes A @ B.T), which is exactly the HF [out, in] layout.
        # So "K-pad" == pad the LAST axis, "N-pad" == pad axis 0. Getting these two
        # backwards is silent: both produce a correctly-SHAPED operand.
        def kpad(t, k):                       # in-features 480 -> 512 (zero columns)
            return self._pad_to(t, k, 1)

        def npad(t, n):                       # out-features 480 -> 512 (zero rows)
            return self._pad_to(t, n, 0)

        # sqrt(480/512) = 0.968246. THE DIRECTION IS THE WHOLE POINT: the kernel
        # normalizes over HP=512 lanes of which 32 are zero, so its mean-square is
        # (480/512)x too small, its rsqrt is sqrt(512/480)x too large, and its output is
        # correspondingly too LARGE -- gamma must scale DOWN to cancel it. Using the
        # reciprocal instead is +6.7% per norm, compounding over 2*32+1 = 65 expert
        # norms to ~1e9x, finite and NaN-free the whole way.
        self.E_GAMMA_FOLD = math.sqrt(H / HP)

        def gamma(name):
            # scale BEFORE padding so the 32 pad lanes stay exactly zero: a nonzero
            # gamma there would multiply whatever junk the pad lanes hold and destroy
            # the invariant that the padded residual stream is zero past column 480.
            return store_weight(self, self._pad_to(sd[name] * self.E_GAMMA_FOLD, HP, 0))

        self.ae_layer_addrs = []
        for i in range(E["num_layers"]):
            la = {}
            # K=480 operands: the activation stream is 512 wide, so the weight needs 32
            # zero in-columns that multiply the (zero) pad lanes.
            la["q_weight"] = store_weight(self, kpad(sd[f"ae.{i}.q_proj.weight"], HP))
            # FAULT 5 -- K/V PROJECTIONS ARE NOT THE SAME SHAPE ON EVERY LAYER.
            # The checkpoint is the evidence, not any code reading:
            #     expert.layers.0.self_attn.k_proj.weight -> (320, 480)
            #     expert.layers.1.self_attn.k_proj.weight -> (320, 320)
            # SELF layers read the expert hidden width (480 -> padded 512, like q/gate/up).
            # CROSS layers read the VLM KV width (320): upstream
            # SmolVLMWithExpert.forward_cross_attn_layer flattens the cached prefix K/V to
            # [B, S, 320] and RE-PROJECTS it through this layer's own k_proj/v_proj before
            # attending. A (320,320) weight cannot consume a 480/512-wide activation, so
            # this shape difference alone proves the reprojection.
            # q_proj stays (960,480) on BOTH kinds -- the query always comes from the
            # expert's own residual stream.
            kv_is_self = self._ae_is_self_attn(i)
            kw_t, vw_t = sd[f"ae.{i}.k_proj.weight"], sd[f"ae.{i}.v_proj.weight"]
            exp_in = H if kv_is_self else KV
            assert tuple(kw_t.shape) == (KV, exp_in) and tuple(vw_t.shape) == (KV, exp_in), (
                f"ae.{i} k/v_proj are {tuple(kw_t.shape)}/{tuple(vw_t.shape)}, expected "
                f"{(KV, exp_in)} for a {'self' if kv_is_self else 'cross'} layer -- if this "
                f"fires, EXPERT_SELF_ON_EVEN (self_attn_on_even) is inverted")
            if kv_is_self:
                # 480 -> 512, zero in-columns against the padded residual stream.
                la["k_weight"] = store_weight(self, kpad(kw_t, HP))
                la["v_weight"] = store_weight(self, kpad(vw_t, HP))
            else:
                # NO K-PAD. The A operand on cross layers is the flattened prefix KV
                # cache, exactly 320 wide -- never the 512-padded expert stream. 320 is
                # 5*64 elements = 640 bytes = 5*128B rows, so it already satisfies both
                # the 64-element ALU floor and the 128-byte SRAM row rule; padding it to
                # 512 would make the operand widths disagree with the activation.
                assert (KV % UE_VECTOR_SIZE) == 0 and (KV * 2) % 128 == 0
                la["k_weight"] = store_weight(self, kw_t)
                la["v_weight"] = store_weight(self, vw_t)
            la["gate_weight"] = store_weight(self, kpad(sd[f"ae.{i}.gate_proj.weight"], HP))
            la["up_weight"] = store_weight(self, kpad(sd[f"ae.{i}.up_proj.weight"], HP))
            # N=480 outputs: 32 zero out-rows, which is what KEEPS lanes [480:512] of the
            # residual stream at exactly zero after every attention/MLP write-back. The
            # gamma fold above is only exact while that holds (_expert_pad_lane_check).
            la["o_weight"] = store_weight(self, npad(sd[f"ae.{i}.o_proj.weight"], HP))
            la["down_weight"] = store_weight(self, npad(sd[f"ae.{i}.down_proj.weight"], HP))
            la["ln1_gamma"] = gamma(f"ae.{i}.input_layernorm.weight")
            la["ln2_gamma"] = gamma(f"ae.{i}.post_attention_layernorm.weight")
            self.ae_layer_addrs.append(la)

        self.ae_final_norm_addr = gamma("ae.final_norm.weight")

        # Shape audit on layer 0 only -- every layer went through the same code path, so
        # one check catches a config/checkpoint drift without 32x the work.
        l0 = sd["ae.0.q_proj.weight"]
        assert tuple(l0.shape) == (Q, H), f"ae.0.q_proj is {tuple(l0.shape)}, expected {(Q, H)}"
        assert tuple(sd["ae.0.k_proj.weight"].shape) == (KV, H)
        assert tuple(sd["ae.0.o_proj.weight"].shape) == (H, Q)
        assert tuple(sd["ae.0.gate_proj.weight"].shape) == (I, H)
        assert tuple(sd["ae.0.down_proj.weight"].shape) == (H, I)
        assert HP % UE_VECTOR_SIZE == 0 and I % UE_VECTOR_SIZE == 0

    def _weight_init_head(self, sd):
        """head.state_proj[960,32]+bias, action_in_proj[480,32]+bias,
        action_time_mlp_in[480,960]+bias, action_time_mlp_out[480,480]+bias,
        action_out_proj[32,480]+bias.

        K=32 and N=32 violate the 64 floor -> pad K 32->64 (zero rows) and N 32->64
        (zero cols); slice [:, :7] on the host at the very end. 480 sides pad to 512.

        action_time_mlp_in IS THE SPECIAL CASE. Its K=960 is concat(action_in_proj_out
        [480], time_emb[480]). Once both halves are 512-padded, the time half starts at
        column 512, not 480, so a K=960 matmul reads the wrong columns for the entire
        time half. Keeping the concat contiguous at 960 is illegal -- the time half would
        begin at byte 960, not a multiple of 128. So K-pad it 960 -> 1024 with a SPLIT
        column permute: source [0:480] -> [0:480], source [480:960] -> [512:992], zeros
        in [480:512] and [992:1024]. Stored shape becomes [512, 1024].

        BIASES pad too, or they poison the pad lanes the RMSNorm fold depends on:
        action_in_proj/time_mlp_in/time_mlp_out .bias[480] -> 512, action_out_proj.bias
        [32] -> 64. state_proj.bias[960] is already clean. The expert stack is bias-free."""
        E, HEAD = self._cfg["expert"], self._cfg["action_head"]
        H, HP = E["hidden_size"], E["hidden_size_padded"]
        AD = HEAD["max_action_dim"]                       # 32 -- below the 64-ALU floor
        ADP = self.ACTION_DIM_PAD                          # 64
        V = UE_VECTOR_SIZE

        def kpad(t, k):
            return self._pad_to(t, k, 1)

        def npad(t, n):
            return self._pad_to(t, n, 0)

        # ---- state_proj [960,32] -> K-pad 32->64. Consumed by the PREFIX stage (row 0
        # of the observation), stored here because the checkpoint files it under head.*.
        # Its bias is 960 wide and already 64-aligned, so it needs nothing.
        self.state_proj_weight = store_weight(self, kpad(sd["head.state_proj.weight"], ADP))
        self.state_proj_bias = store_weight(self, sd["head.state_proj.bias"])

        # ---- action_in_proj [480,32] -> [512,64]: K-pad 32->64 AND N-pad 480->512.
        # The N-pad is what makes lanes [480:512] of the suffix stream zero from the very
        # first op, which is the precondition for the whole expert gamma fold.
        self.action_in_weight = store_weight(
            self, npad(kpad(sd["head.action_in_proj.weight"], ADP), HP))
        self.action_in_bias = store_weight(self, self._pad_to(sd["head.action_in_proj.bias"], HP, 0))

        # ---- action_time_mlp_in [480,960] -> [512,1024], SPLIT-COLUMN K permute.
        # K=960 is concat(action_in_proj_out[480], time_emb[480]). Both halves are
        # 512-padded at runtime, so the time half physically begins at column 512, not
        # 480. A plain K-pad 960->1024 (appending zeros) would leave the weight expecting
        # the time half at columns 480..959 while the buffer has it at 512..991 -- every
        # time column off by 32, finite and NaN-free and completely wrong. Keeping the
        # concat tight at 960 is not an option either: 960 bf16 = 1920 B, so the time half
        # would start mid-SRAM-row. Hence: scatter the two halves to 0 and 512.
        w_ti = sd["head.action_time_mlp_in.weight"]
        assert tuple(w_ti.shape) == (H, 2 * H), (
            f"action_time_mlp_in should be [{H},{2 * H}] = [out, concat(action,time)], "
            f"got {tuple(w_ti.shape)}")
        w_split = torch.zeros(H, 2 * HP, dtype=w_ti.dtype)
        w_split[:, :H] = w_ti[:, :H]                  # action half  -> columns [0:480]
        w_split[:, HP:HP + H] = w_ti[:, H:]           # time   half  -> columns [512:992]
        # columns [480:512] and [992:1024] stay zero: they multiply the runtime buffer's
        # pad lanes, which are themselves zero, so the product is zero twice over.
        self.time_mlp_in_weight = store_weight(self, npad(w_split, HP))
        self.time_mlp_in_bias = store_weight(
            self, self._pad_to(sd["head.action_time_mlp_in.bias"], HP, 0))

        # ---- action_time_mlp_out [480,480] -> [512,512] on both axes.
        self.time_mlp_out_weight = store_weight(
            self, npad(kpad(sd["head.action_time_mlp_out.weight"], HP), HP))
        self.time_mlp_out_bias = store_weight(
            self, self._pad_to(sd["head.action_time_mlp_out.bias"], HP, 0))

        # ---- action_out_proj [32,480] -> [64,512]: K-pad 480->512 and N-pad 32->64.
        w_out = kpad(sd["head.action_out_proj.weight"], HP)
        b_out = sd["head.action_out_proj.bias"]
        self.action_out_weight = store_weight(self, npad(w_out, ADP))
        self.action_out_bias = store_weight(self, self._pad_to(b_out, ADP, 0))

        # ---- the SAME projection pre-multiplied by the Euler step dt = -1/N.
        # The loop body computes x_t += dt * (W v + b); folding dt into W and b makes that
        # a plain eltwise add of the matmul result, with no on-device scalar multiply and
        # no extra pass over the tensor. dt is a compile-time constant of the schedule, so
        # this is exact up to one bf16 rounding of each weight (the same rounding the
        # unscaled copy pays anyway). The unscaled copy is kept for probes/oracles.
        dt = -1.0 / HEAD["num_denoise_steps"]
        self.AE_DT = dt
        self.action_out_weight_dt = store_weight(self, npad(w_out * dt, ADP))
        self.action_out_bias_dt = store_weight(self, self._pad_to(b_out * dt, ADP, 0))

        # ---- identity, needed by silu_core_dram (composed x*sigmoid(x)) and by flash.
        # _weight_init_lm is contracted to create it too; whichever runs first wins, and
        # storing it twice would silently waste params DRAM rather than fail.
        if not hasattr(self, "identity_addr"):
            self.identity_addr = store_identity_matrix(self)

        assert ADP % V == 0 and HP % V == 0
        assert AD <= ADP, f"max_action_dim {AD} exceeds its 64-lane slot"

    def _assert_params_fit(self):
        """Enforce params_end < tensor_dram_base < program_dram_base < 0x100000000 and
        print the region map. bf16 budget ~1.12 GB + 90 MB embed table; q4 LM shrinks it.

        Why this is a hard assert and not a warning: the weight blob overrunning the
        tensor base does not fault -- it silently overwrites activations mid-run, which
        is exactly the LocateAnything decode-NaN failure. And anything landing at or
        above absolute 0x100000000 dies with a deterministic [Errno 512] on dma_write
        (the 4 GB boundary bug in the XDMA path), which reads like a driver hiccup, not
        like a layout bug. Both are cheap to catch here and expensive to catch later."""
        DRAM_START = 0x80000000    # config bases are OFFSETS from here (params base 0x0)

        def absolute(base):
            # tolerate both conventions: offsets (this model) and already-absolute bases
            return base if base >= DRAM_START else DRAM_START + base

        p0 = absolute(self._params_dram_base)
        t0 = absolute(self._tensor_dram_base)
        g0 = absolute(self._program_dram_base)
        used = self.get_params_dram_usage()
        p_end = p0 + used
        mb = 1024 * 1024

        print("  params DRAM region map (absolute addresses):")
        print(f"    params  0x{p0:09X} .. 0x{p_end:09X}   {used / mb:8.1f} MB used "
              f"/ {(t0 - p0) / mb:8.1f} MB available")
        print(f"    tensor  0x{t0:09X} .. 0x{g0:09X}   {(g0 - t0) / mb:8.1f} MB region")
        print(f"    program 0x{g0:09X} .. 0x{0x100000000:09X}   "
              f"{(0x100000000 - g0) / mb:8.1f} MB region")

        assert p_end < t0, (
            f"params overflow: weights end at 0x{p_end:X} but tensor_dram_base is "
            f"0x{t0:X} -- the next activation write silently overwrites weights "
            f"({(p_end - t0) / mb:.1f} MB over)")
        assert t0 < g0, (f"tensor base 0x{t0:X} must be below program base 0x{g0:X}")
        assert g0 < 0x100000000, (
            f"program base 0x{g0:X} is at/above the 4 GB boundary -- dma_write there "
            f"fails deterministically with [Errno 512]")

    # ==================================================================================
    # 2. tensors / scratch  (naming mirrors smolvlm2_test.py::tensor_init)
    # ==================================================================================

    def tensor_init(self):
        """Allocate every activation buffer at the tensor base.

        PREFIX KV CACHE -- the one structural difference from smolvlm2. Same flat
        [layer, kv_head, seq, dim] layout and the same stride constants:
            self.k_size          = HEAD_DIM * 2                       # 128 B
            self.KV_HEAD_STRIDE  = PM * self.k_size
            self.KV_LAYER_STRIDE = NUM_KV_HEADS * self.KV_HEAD_STRIDE
            LAYER0_K_DRAM / LAYER0_V_DRAM = allocate(NUM_LAYERS * KV_LAYER_STRIDE)
        but here it is written ONCE per observation and read by the expert's
        cross-attention layers for all 10 denoise steps -- it must survive the whole
        denoise loop, so nothing else may reuse that region.

        Per-layer intermediates, shared across the 32 layers (ping-pong):
            LAYER0_INPUT/OUTPUT/PRE_NORM/RESIDUAL, LAYER0_Q, LAYER0_K_PROJ, LAYER0_V_PROJ,
            LAYER0_O_PROJ, LAYER0_ATTN_RESULT, LAYER0_MLP_GATE/UP/MULT/DOWN
        Flash staging: FLASH_Q/K/V/OUT/BIAS/SCRATCH sized for PM*GROUP_SIZE rows.
        Vision: [1024,768] x {x, resid, qkv, attn, mlp} per slot.
        Expert: [64,512] x_t, [64,960] q, [64,320] k/v, [64,1280] mlp, its own
            AE_FLASH_* staging at 64*GROUP_SIZE rows.
        Const: 10 precomputed timestep embeddings [10,480].

        IMPLEMENTED SO FAR: the full vision/connector path (tensor_init_vision) plus the
        prefix KV cache, which is allocated HERE rather than in the prefix stage on
        purpose -- it is the one region that must outlive everything (written once per
        observation, read by the expert's cross-attention on all 10 denoise steps), so
        it is claimed before any ping-pong buffer can be tempted to reuse it.
        The per-layer LM / expert / head buffers are a TODO for the prefix + expert
        stages; nothing in the vision path touches them."""
        print(f"  tensor DRAM starts at 0x{self.get_tensor_dram_addr():X}")
        self.assert_vision_dims()
        self.tensor_init_vision()
        vis_bytes = self.get_tensor_dram_usage()

        # ---- prefix KV cache: flat [layer, kv_head, seq, dim], smolvlm2's layout and
        # stride constants verbatim so the strided_copy emitters lift unchanged. ----
        bpe = 2
        PM = self.PREFILL_MAX_SEQ_LEN
        self.bytes_per_element = bpe
        self.k_size = self.HEAD_DIM * bpe                        # 128 B per position/head
        self.KV_HEAD_STRIDE = PM * self.k_size                   # one head, all positions
        self.KV_LAYER_STRIDE = self.NUM_KV_HEADS * self.KV_HEAD_STRIDE
        kv_bytes = self.NUM_LAYERS * self.KV_LAYER_STRIDE
        self.LAYER0_K_DRAM = self.allocate_tensor_dram(kv_bytes)
        self.LAYER0_V_DRAM = self.allocate_tensor_dram(kv_bytes)
        # Zero it: pad positions are computed but masked, and an UNWRITTEN DRAM address
        # returns EIO on dma_read (it never entered the ECC-initialized set), so a probe
        # of the cache would fail rather than report zeros.
        kv_zeros = torch.zeros(kv_bytes // bpe, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_K_DRAM, kv_zeros)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, kv_zeros)

        # TODO(prefix stage):  LAYER0_INPUT/OUTPUT/PRE_NORM/RESIDUAL, LAYER0_Q,
        #   LAYER0_K_PROJ/V_PROJ/O_PROJ/ATTN_RESULT, LAYER0_MLP_GATE/UP/MULT/DOWN,
        #   FLASH_Q/K/V/OUT/BIAS/SCRATCH at PM*GROUP_SIZE rows, ROPE_PACKED*_DRAM.
        # TODO(expert stage): AE x_t [64,512], q [64,960], k/v [64,320], mlp [64,1280],
        #   AE_FLASH_* at 64*GROUP_SIZE rows, and the [10,512] timestep table.
        # Neither group is referenced by compile_encoder/compile_connector, so the vision
        # bring-up is complete without them.

        total = self.get_tensor_dram_usage()
        region = (self._program_dram_base - self._tensor_dram_base)
        print(f"  tensor DRAM: vision {vis_bytes / 1024**2:.1f} MB + kv cache "
              f"{2 * kv_bytes / 1024**2:.1f} MB = {total / 1024**2:.1f} MB "
              f"/ {region / 1024**2:.0f} MB region")
        assert total <= region, (
            f"tensor region overflow: {total} B used, {region} B available -- "
            f"activations would spill into program DRAM")

    def _load_rope_tables(self):
        """cos/sin for the prefix, built at THIS model's theta.

        THETA IS 10000, from the verapulse checkpoint's own config -- NOT smolvlm2's
        100000. The table structure is lifted from smolvlm2::_load_rope_tables but the
        VALUES must be rebuilt; copying them silently rescales every position.

        Two tables, both in params DRAM so they land in params.bin:
          ROPE_PACKED_DRAM      [S, 2D] per-token [cos || sin], sin's first half
                                PRE-NEGATED for the HW kernel. Interleaved per token
                                because the d64 path asserts sin_addr == cos_addr + D*2.
          ROPE_PACKED_GQA_DRAM  the same rows repeat_interleave'd GROUP_SIZE times, so
                                row r of the stacked-Q buffer picks up token (r//G)'s
                                rotation. There is no d64 GQA rope core; this table IS
                                the mechanism.

        head_dim 64 -> half 32, which is 32-aligned, so no padded-split variant is needed.
        N=64 is legal but PBI-ONLY: always pass gpr_M_reg (a bare call falls through to
        rope_hf_core_dram_legacy, which asserts N >= 128)."""
        D, S, bpe = self.HEAD_DIM, self.PREFILL_MAX_SEQ_LEN, 2
        inv = 1.0 / (self.ROPE_THETA ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
        f = torch.outer(torch.arange(S, dtype=torch.float32), inv)
        cos = torch.cat([f.cos(), f.cos()], 1).to(torch.bfloat16).contiguous()
        sin = torch.cat([f.sin(), f.sin()], 1).to(torch.bfloat16).contiguous()
        sin[:, : D // 2] = -sin[:, : D // 2]          # pre-negate for the HW kernel
        packed = torch.cat([cos, sin], 1).contiguous()                     # [S, 2D]
        self.ROPE_PACKED_DRAM = store_weight(self, packed)
        gqa = packed.repeat_interleave(self.GROUP_SIZE, dim=0).contiguous()
        self.ROPE_PACKED_GQA_DRAM = store_weight(self, gqa)
        self._hw_cos, self._hw_sin = cos, sin
        print(f"    rope tables [{S},{2 * D}] theta={self.ROPE_THETA:g} "
              f"(+ x{self.GROUP_SIZE} gqa)")

    def build_attn_bias(self, valid_prefix_len=None):
        """Precomputed additive masks. Flash's causal flag alone cannot express any of
        these, so each is a real DRAM tensor.

        prefix: BIDIRECTIONAL over the valid rows -- this is an observation encode, not
        a causal decode. -inf on the pad rows/cols beyond valid_prefix_len.

        -inf, not a large negative, so softmax zeroes exactly. Pad rows are still
        COMPUTED (never skipped); the bias is what makes them harmless.

        valid_prefix_len is data-dependent: text is padded to tokenizer_max_length, so
        the real count is 1 + n_image + n_real_text and varies per prompt. Re-DMA this
        per observation -- it is data, changing it needs no recompile.

        SHAPE: the tensor is [QB,QB], NOT [PM,PM]. compile_prefix runs attention in the
        STACKED-Q layout: per kv-group it gathers that group's G q-heads token-major into
        LM_FLASH_Q_DRAM (row t*G+g = token t, head g) and replicates K/V x G to match, so
        the flash batch is QB = aligned(PM*G) rows and it passes aligned_seq_len=QB. A
        [PM,PM] mask is therefore UNDERSIZED -- the kernel would read QB*QB elements from
        a PM*PM buffer, i.e. run off the end of the allocation into whatever tensor was
        allocated next, and mask nothing correctly. smolvlm2_test.py sizes its
        FLASH_BIAS_DRAM the same way (qmax*qmax with qmax = aligned(PM*G)).

        Stacked row i is token i//G and stacked column j is token j//G, so the stacked
        mask is exactly the [PM,PM] logical mask repeat_interleave'd by G on BOTH axes.

        The alignment round-up is kept as a formula, never hardcoded: at PM=192, G=3 it
        is a no-op (576 = 9*64), but a config with a non-64-multiple PM*G would silently
        under-allocate if this were written as PM*G.

        NOTE vs smolvlm2: its prefill bias additionally requires g == g' (a block-diagonal
        over the G copies). That is an equivalent, not a different, mask here: K is an
        EXACT x G duplication, so admitting all G copies of a key triples both the softmax
        numerator and denominator and leaves the attention output unchanged. We use the
        plain repeat_interleave because it is the one construction that stays correct if
        the K duplication layout ever changes shape."""
        PM, G = self.PREFILL_MAX_SEQ_LEN, self.GROUP_SIZE
        # must equal what compile_prefix passes as batch/aligned_seq_len (self.LM_QB)
        QB = ((PM * G + self.VEC - 1) // self.VEC) * self.VEC
        valid = self.PREFIX_LEN if valid_prefix_len is None else int(valid_prefix_len)
        assert 0 < valid <= PM, f"valid_prefix_len {valid} outside [1,{PM}]"
        bias = torch.zeros(PM, PM, dtype=torch.float32)
        if valid < PM:
            bias[:, valid:] = float("-inf")       # nothing may attend TO a pad column
            bias[valid:, :] = float("-inf")       # pad rows attend nowhere ...
            bias[valid:, valid:] = 0.0            # ... but keep their own diagonal block
        # BLOCK-CAUSAL, not fully bidirectional. Upstream builds att_masks as
        # [0]*images + [0]*language + [1]*state and then
        #     cumsum = att_masks.cumsum(1);  att_2d = cumsum[:,None,:] <= cumsum[:,:,None]
        # With images+language in block 0 and state alone in block 1, that means images
        # and language attend bidirectionally among THEMSELVES but cannot see the state
        # row, while the state row sees everything. The state row is the last valid row
        # (layout is images -> text -> state), so exactly one column needs masking off
        # for every row above it.
        if valid >= 2:
            bias[:valid - 1, valid - 1] = float("-inf")
        # [PM,PM] -> [PM*G, PM*G]: stacked_bias[i,j] = bias[i//G, j//G]
        bias = bias.repeat_interleave(G, dim=0).repeat_interleave(G, dim=1)
        if bias.shape[0] < QB:
            # alignment tail (no-op at PM*G % 64 == 0). Same rule as the pad rows: the
            # tail rows may attend nowhere except their own block, or an all -inf row
            # gives 0/0 = NaN and flash smears it across the whole tile.
            pad = QB - bias.shape[0]
            bias = F.pad(bias, (0, pad, 0, pad), value=float("-inf"))
            bias[-pad:, -pad:] = 0.0
        if not hasattr(self, "PREFIX_BIAS_DRAM"):
            self.PREFIX_BIAS_DRAM = self.allocate_tensor_dram(QB * QB * 2)
        self.dma_to_accelerator_memory(self.PREFIX_BIAS_DRAM, bias.to(torch.bfloat16))
        self._prefix_valid_len = valid
        return self.PREFIX_BIAS_DRAM

    def compile_encoder(self):
        """One SigLIP pass over [1024,768] + the connector, compiled ONCE and executed
        once per camera slot (the program is the expensive artifact). Returns its DRAM
        address.

        Per layer: LN1(beta) -> q/k/v(+bias) -> per-head flash (MHA, 12 heads, hd 64,
        bidirectional) -> out_proj(+bias) -> residual -> LN2(beta) -> fc1(+bias)+GELU ->
        fc2(+bias) -> residual. Then post_ln -> pixel shuffle -> connector projection.

        All matmuls are bf16 B-operand (no is_B_quantized/data_type/SCALE_DRAM_ADDR) and
        PBI (gpr_M_reg=vis_S_reg), so the captured program is structure-bound, not
        M-bound. Flash stays static -- PBI flash address injection corrupts on the second
        execution, and this program runs twice, once per slot.

        Both slots are always encoded; no zero-slot skipping.

        NEXT STEP (not done here): pi05's _emit_encoder_body(ue, engine_idx, sched,
        row_offset, rows, kgrp, nk) shards this over engines by row and by a K-axis grid.
        This is the single-engine form; shard it once it passes >=40 dB."""
        V, C = self._cfg["vision"], self._cfg["connector"]
        S, H, I = V["num_patches"], V["hidden_size"], V["intermediate_size"]
        D, NH = V["head_dim"], V["num_heads"]
        P, CH, NPS = V["patch_size"], V["num_channels"], V["num_patches_per_side"]
        bpe = 2

        self.start_capture()
        vis_S_reg = self.alloc_isa_reg()
        self.generate_instruction_add_set(vis_S_reg, S)

        def vis_matmul(M, K, N, A, la, proj, OUT, bias=None, **kw):
            # bf16 B operand: no is_B_quantized / data_type / SCALE_DRAM_ADDR.
            self.matmat_mul_core(
                M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f"{proj}_weight"],
                OUTPUT_DRAM_ADDR=OUT, C_DRAM_ADDR=bias, bias_mode="broadcast_N",
                gpr_M_reg=vis_S_reg, **kw)

        # PATCHIFY IS DONE ON THE HOST -- see run_vision. It used to be a device
        # smart_bf16_permute_core(dims=[CH,NPS,P,NPS,P], perm=[1,3,0,2,4]), which is
        # CORRECT as index math (verified bit-exact vs conv2d unfold) but lands in
        # nn_lib's NON-64-ALIGNED branch, because the last dim is P=16 < UE_VECTOR_SIZE:
        #
        #     if last_dim < UE_VECTOR_SIZE or last_dim % UE_VECTOR_SIZE != 0:
        #         for j in range(total_elements // last_dim):
        #             ue.ue_memcpy_from_dram(...); ue.wait_queue()
        #             ue.ue_memcpy_to_dram(...);   ue.wait_queue()
        #
        # That emits 49152 x 2 descriptors (~98k of this program's 170k instructions),
        # every one staging through the SAME URAM address, serialized only by
        # ue.wait_queue() -- a HOST-side poll that does nothing inside a captured
        # instruction stream. At runtime they pipeline through one URAM slot and race.
        # Symptom on hardware: op-bisect showed the patch output with matching rms
        # (0.1329 vs 0.1331) and absmax (0.715 vs 0.721) but cos 0.196 -- every value
        # present, arrangement scrambled. The connector's shuffle is unaffected: its
        # last dim is 768, so it takes the aligned fast path (52 dB on hardware).
        #
        # The host does it instead, exactly as pi05 does. Same DMA volume either way
        # (1.5 MB per slot), and _host_patchify is verified bit-exact against
        # conv2d unfold.
        self.matmat_mul_core(
            M=S, K=CH * P * P, N=H, A_DRAM_ADDR=self.VIS_PIXEL_IN_DRAM,
            B_DRAM_ADDR=self.patch_weight_addr, OUTPUT_DRAM_ADDR=self.VIS_LN_OUT_DRAM,
            C_DRAM_ADDR=self.patch_bias_addr, bias_mode="broadcast_N",
            gpr_M_reg=vis_S_reg)
        if self.VIS_BISECT:
            self._probe_copy(self.VIS_LN_OUT_DRAM, self.VIS_P_PATCH_DRAM, S * H)
        eltwise_add_core_dram(
            self, size=S * H, A_DRAM_ADDR=self.VIS_LN_OUT_DRAM,
            B_DRAM_ADDR=self.pos_embed_addr, OUTPUT_DRAM_ADDR=self.VIS_IO_A_DRAM)

        n_vis = len(self.vis_layer_addrs) if self.VIS_LAYERS is None else int(self.VIS_LAYERS)
        assert 1 <= n_vis <= len(self.vis_layer_addrs), f"VIS_LAYERS={self.VIS_LAYERS}"
        if n_vis != len(self.vis_layer_addrs):
            print(f"    [bisect] compiling only {n_vis}/{len(self.vis_layer_addrs)} ViT layers")
        for i, la in enumerate(self.vis_layer_addrs[:n_vis]):
            h_in = self.VIS_IO_A_DRAM if i % 2 == 0 else self.VIS_IO_B_DRAM
            h_out = self.VIS_IO_B_DRAM if i % 2 == 0 else self.VIS_IO_A_DRAM

            self.layer_norm_core_dram(
                M=S, N=H, A_DRAM_ADDR=h_in, OUTPUT_DRAM_ADDR=self.VIS_LN_OUT_DRAM,
                GAMMA_DRAM_ADDR=la["ln1_weight"], BETA_DRAM_ADDR=la["ln1_bias"],
                gpr_M_reg=vis_S_reg, ZEROS_DRAM_ADDR=self.vis_zeros_addr,
                INV_N_DRAM_ADDR=self.vis_inv_n_addr)

            if self.VIS_BISECT and i == 0:
                self._probe_copy(self.VIS_LN_OUT_DRAM, self.VIS_P_LN1_DRAM, S * H)
            for proj, dst in (("q", self.VIS_Q_DRAM), ("k", self.VIS_K_DRAM),
                              ("v", self.VIS_V_DRAM)):
                vis_matmul(S, H, H, self.VIS_LN_OUT_DRAM, la, proj, dst,
                           bias=la[f"{proj}_bias"])

            # MHA, one head at a time: gather this head's [S,64] column block into the
            # fixed flash operands, run static flash, scatter back. The scatter DEST
            # carries + h*D*bpe -- omitting that per-head offset is the finite-but-
            # scrambled bug class (pi05 denoise #3), not a NaN.
            elems, col_stride, row_jump = S * D, D * bpe, H * bpe
            for h in range(NH):
                col = h * col_stride
                for src, dst in ((self.VIS_Q_DRAM + col, self.VIS_FLASH_Q_DRAM),
                                 (self.VIS_K_DRAM + col, self.VIS_FLASH_K_DRAM),
                                 (self.VIS_V_DRAM + col, self.VIS_FLASH_V_DRAM)):
                    self.accelerator_memory_to_sram(
                        src, 0x00000, elems, stride_bytes_per_chunk=col_stride,
                        stride_jump_bytes=row_jump)
                    self.sram_to_accelerator_memory(0x00000, dst, elems)
                self.unified_attention_core(
                    batch=S, aligned_seq_len=S, head_dim=D,
                    Q_DRAM_ADDR=self.VIS_FLASH_Q_DRAM,
                    K_DRAM_ADDR=self.VIS_FLASH_K_DRAM,
                    V_DRAM_ADDR=self.VIS_FLASH_V_DRAM,
                    BIAS_DRAM_ADDR=self.VIS_ATTN_BIAS_DRAM,
                    OUTPUT_DRAM_ADDR=self.VIS_FLASH_OUT_DRAM,
                    SCRATCH_DRAM_ADDR=self.VIS_ATTN_SCRATCH_DRAM,
                    IDENTITY_DRAM_ADDR=self.identity_addr,
                    gpr_batch_reg=vis_S_reg, gpr_aligned_seq_len_reg=vis_S_reg)
                self.accelerator_memory_to_sram(self.VIS_FLASH_OUT_DRAM, 0x00000, elems)
                self.sram_to_accelerator_memory(
                    0x00000, self.VIS_ATTN_RESULT_DRAM + col, elems,
                    stride_bytes_per_chunk=col_stride, stride_jump_bytes=row_jump)

            vis_matmul(S, H, H, self.VIS_ATTN_RESULT_DRAM, la, "o",
                       self.VIS_O_PROJ_DRAM, bias=la["o_bias"])
            eltwise_add_core_dram(
                self, size=S * H, A_DRAM_ADDR=h_in, B_DRAM_ADDR=self.VIS_O_PROJ_DRAM,
                OUTPUT_DRAM_ADDR=self.VIS_RESIDUAL_DRAM)
            self.layer_norm_core_dram(
                M=S, N=H, A_DRAM_ADDR=self.VIS_RESIDUAL_DRAM,
                OUTPUT_DRAM_ADDR=self.VIS_LN_OUT_DRAM,
                GAMMA_DRAM_ADDR=la["ln2_weight"], BETA_DRAM_ADDR=la["ln2_bias"],
                gpr_M_reg=vis_S_reg, ZEROS_DRAM_ADDR=self.vis_zeros_addr,
                INV_N_DRAM_ADDR=self.vis_inv_n_addr)

            if self.VIS_BISECT and i == 0:
                self._probe_copy(self.VIS_LN_OUT_DRAM, self.VIS_P_LN2_DRAM, S * H)
            # The fused GELU is x*sigmoid(1.702x); the model specifies gelu_pytorch_tanh.
            # Score the oracle with --hw-gelu or this stage falsely reads ~28 dB low.
            vis_matmul(S, H, I, self.VIS_LN_OUT_DRAM, la, "fc1",
                       self.VIS_MLP_INTER_DRAM, bias=la["fc1_bias"], gelu_enable=True)
            vis_matmul(S, I, H, self.VIS_MLP_INTER_DRAM, la, "fc2",
                       self.VIS_MLP_OUT_DRAM, bias=la["fc2_bias"])
            eltwise_add_core_dram(
                self, size=S * H, A_DRAM_ADDR=self.VIS_RESIDUAL_DRAM,
                B_DRAM_ADDR=self.VIS_MLP_OUT_DRAM, OUTPUT_DRAM_ADDR=h_out)

        final = (self.VIS_IO_A_DRAM if n_vis % 2 == 0
                 else self.VIS_IO_B_DRAM)
        self.layer_norm_core_dram(
            M=S, N=H, A_DRAM_ADDR=final, OUTPUT_DRAM_ADDR=self.VIS_POST_LN_DRAM,
            GAMMA_DRAM_ADDR=self.vis_post_ln_weight,
            BETA_DRAM_ADDR=self.vis_post_ln_bias, gpr_M_reg=vis_S_reg,
            ZEROS_DRAM_ADDR=self.vis_zeros_addr, INV_N_DRAM_ADDR=self.vis_inv_n_addr)

        self.compile_connector()

        self.generate_instruction_halt()
        self.release_isa_reg()
        self.stop_capture()

        raw = bytearray()
        for inst in self.capture_buffer:
            raw.extend(inst.get_bytes())
        addr = self.get_program_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, addr, raw, len(raw))
        self.allocate_program_dram(len(raw))
        self.clear_capture_buffer()
        self._vis_program_addr = addr
        print(f"    vision encoder + connector: {len(raw)} bytes @0x{addr:X}")
        return addr

    def compile_connector(self):
        """[1024,768] -> pixel shuffle x4 -> [64,12288] -> proj -> [64,960].

        ONE natively-supported permute. dims=[8,4,8,4,768] views the 1024 tokens as
        (by,dy,bx,dx); permute_indices=[0,2,1,3,4] reorders to (by,bx,dy,dx), whose
        row-major flattening IS [64, 16*768] -- reshape is free in a flat address space.

        Cheap because the last dim (768) stays put and is 64-aligned, so nn_lib takes the
        pure DMA-gather path, not the transpose path: per-output-row memcpy with the
        source offset from a precomputed index table and a sequential destination. The
        moving unit is a whole 768-element row (1536 B = 12 full SRAM rows), so no
        sub-row addressing and no per-index destination offset.

        Emitted inside compile_encoder's capture so a slot is ONE instruction stream --
        splitting it would forfeit compile-once and cost a host round-trip."""
        V, C = self._cfg["vision"], self._cfg["connector"]
        H, s = V["hidden_size"], C["pixel_shuffle_scale_factor"]
        g = V["num_patches_per_side"] // s

        smart_bf16_permute_core(
            self, dims=[g, s, g, s, H], permute_indices=[0, 2, 1, 3, 4],
            input_dram_addr=self.VIS_POST_LN_DRAM,
            output_dram_addr=self.VIS_SHUFFLED_DRAM,
            params_dram_addr=self.permute_params_addr,
            temp_dram_start=self.PERMUTE_TEMP_DRAM)

        # M=64 only, so PBI buys nothing -- keep it static.
        self.matmat_mul_core(
            M=C["tokens_out"], K=C["input_size"], N=C["output_size"],
            A_DRAM_ADDR=self.VIS_SHUFFLED_DRAM, B_DRAM_ADDR=self.conn_weight_addr,
            OUTPUT_DRAM_ADDR=self.VIS_CONNECTOR_DRAM)

    def tensor_init_vision(self):
        """Vision + connector activation buffers. Called by tensor_init."""
        V, C = self._cfg["vision"], self._cfg["connector"]
        S, H, I = V["num_patches"], V["hidden_size"], V["intermediate_size"]
        D, bpe = V["head_dim"], 2

        def a(n):
            return self.allocate_tensor_dram(n * bpe)

        self.VIS_PIXEL_IN_DRAM    = a(S * H)     # host-staged pixels, per slot
        self.VIS_PATCH_PERM_DRAM  = a(S * H)
        self.VIS_IO_A_DRAM        = a(S * H)     # layer ping-pong
        self.VIS_IO_B_DRAM        = a(S * H)
        self.VIS_LN_OUT_DRAM      = a(S * H)
        self.VIS_Q_DRAM           = a(S * H)
        self.VIS_K_DRAM           = a(S * H)
        self.VIS_V_DRAM           = a(S * H)
        self.VIS_ATTN_RESULT_DRAM = a(S * H)
        self.VIS_O_PROJ_DRAM      = a(S * H)
        self.VIS_RESIDUAL_DRAM    = a(S * H)
        self.VIS_MLP_INTER_DRAM   = a(S * I)     # the big one: 1024 x 3072
        self.VIS_MLP_OUT_DRAM     = a(S * H)
        self.VIS_POST_LN_DRAM     = a(S * H)

        self.VIS_FLASH_Q_DRAM   = a(S * D)
        self.VIS_FLASH_K_DRAM   = a(S * D)
        self.VIS_FLASH_V_DRAM   = a(S * D)
        self.VIS_FLASH_OUT_DRAM = a(S * D)
        self.VIS_ATTN_SCRATCH_DRAM = a((D + S) * S + S * D)
        # bidirectional, no mask: all-zero additive bias. Every one of the 1024 patches
        # is real, so nothing here needs -inf.
        self.VIS_ATTN_BIAS_DRAM = a(S * S)
        self.dma_to_accelerator_memory(
            self.VIS_ATTN_BIAS_DRAM, torch.zeros(S, S, dtype=torch.bfloat16))

        self.VIS_SHUFFLED_DRAM  = a(C["tokens_out"] * C["input_size"])
        self.VIS_CONNECTOR_DRAM = a(C["tokens_out"] * C["output_size"])
        self.PERMUTE_TEMP_DRAM = self.get_tensor_dram_addr()
        self.allocate_tensor_dram(S * H * bpe * 2)

        # Bisect probes. Three intermediates are otherwise DESTROYED before readback:
        # the patch projection and both LayerNorms all land in VIS_LN_OUT_DRAM, each
        # overwriting the last. Everything else in a single layer already has its own
        # buffer and survives, so only these three need a copy.
        self.VIS_P_PATCH_DRAM = a(S * H)
        self.VIS_P_LN1_DRAM   = a(S * H)
        self.VIS_P_LN2_DRAM   = a(S * H)

    def assert_vision_dims(self):
        """Fail loudly at build time if the config drifts from what this emitter and the
        checkpoint agree on."""
        V, C = self._cfg["vision"], self._cfg["connector"]
        s = C["pixel_shuffle_scale_factor"]
        assert V["hidden_size"] % UE_VECTOR_SIZE == 0
        assert V["intermediate_size"] % UE_VECTOR_SIZE == 0
        assert V["head_dim"] * V["num_heads"] == V["hidden_size"]
        assert V["head_dim"] % UE_VECTOR_SIZE == 0, "hd must be %64: no pad path here"
        assert V["num_patches"] == V["num_patches_per_side"] ** 2
        assert V["num_patches_per_side"] * V["patch_size"] == V["image_size"]
        assert V["num_patches_per_side"] % s == 0
        assert C["tokens_out"] == V["num_patches"] // (s * s)
        assert C["input_size"] == V["hidden_size"] * s * s
        assert V.get("num_kv_heads", V["num_heads"]) == V["num_heads"], "vision is MHA"

    def embed_and_concat_prefix(self, token_ids, vision_tokens, state, text_mask=None):
        """Build the [192,960] prefix on the host + DMA (smolvlm2's preamble does the
        analogous embed/merge):
             row 0          state_proj(state[32] padded to 64) -> 960
             rows 1..128    2 x 64 connector tokens
             rows 129..176  embed_tokens[text ids]  (48)
             rows 177..191  pad (masked, still computed)

        THIS LAYOUT IS A GUESS AND PROBABLY WRONG -- resolve it first, it invalidates
        everything downstream:
          - image_token_id 49190 exists, which implies image embeddings are SCATTERED
            into placeholder positions inside the tokenized text sequence (the
            SmolVLM/SmolVLA pattern), not concatenated in front of it. If so this is not
            state||images||text at all, and every RoPE position changes.
          - pi0/pi0.5 and SmolVLA put state LAST (images -> language -> state), not first.
          - the valid length is not the constant 177: text is PADDED to 48, so it is
            1 + 128 + n_real_text and varies per prompt. Return it, and feed it to
            build_attn_bias and to the cumsum that generates RoPE positions.

        IMPLEMENTED for prefix_order == "state_images_text", which is what
        VeraPulseRef.build_prefix does by default and therefore the only variant the
        122.83 dB reference actually blesses. THIS METHOD IS A MIRROR OF THAT REFERENCE:
        same projection, same embedding lookup, same row order. If build_prefix changes,
        this changes with it or the HW is scored against a model it is not running.

        Args:
            token_ids      [T] int64, T == tokenizer_max_length (48), RIGHT-padded
            vision_tokens  [n_vision, 960] the connector output (run_vision's return)
            state          [state_dim] robot proprioception, unnormalized by us
            (keyword) text_mask [T] bool -- which of the T text slots are REAL. See below.
        Returns:
            (x [PM,960] float32, valid_len int)  and stashes both on self.

        WHY valid_len IS RETURNED, not computed downstream: text is padded to 48, so the
        real row count is 1 + n_vision + n_real_text, data-dependent per prompt. Only the
        caller knows which text slots are real, so a text_mask may be passed; with no mask
        we count ALL T text slots as real (valid_len == PREFIX_LEN). That default is the
        conservative one -- it never masks a real token -- but it does leave padded text
        attending, so tokenize() should hand a mask through once it exists.

        RIGHT-PADDING IS LOAD-BEARING. compile_prefix rotates row r with rope table entry
        r (arange), while the reference uses positions = cumsum(mask)-1. Those two agree
        for every valid row IFF all pad slots sit AFTER all real ones -- i.e. the text is
        right-padded and the prefix pad rows are the tail. Left-padding, or an interior
        hole, silently shifts every subsequent RoPE position (the pi05 lesson) and there
        is no assert on device that can catch it.

        NO ROW MAY BE EXACTLY ZERO. rms_norm_core has no epsilon: an all-zero row gives
        rsqrt(0) = inf -> NaN, and the bidirectional attention then mixes that NaN into
        every real row, so a masked pad row is NOT harmless if it is zero. Fill 1e-6,
        exactly as run_denoise does for its x_t padding. pi05 hit this bug class twice."""
        PM, H = self.PREFILL_MAX_SEQ_LEN, self.HIDDEN_SIZE
        order = getattr(self, "prefix_order", None) or OpenChoices.prefix_order
        if order != "images_text_state":
            raise NotImplementedError(
                f"prefix_order={order!r}: only 'images_text_state' is implemented on HW. "
                f"That order is not a guess -- it is read off upstream's shipped "
                f"embed_prefix (smolvla/modeling.py), which appends images, then "
                f"language, then state. The other orders move the state row and/or "
                f"scatter the image embeddings into the token stream, which changes "
                f"every RoPE position and the bias's valid-row block.")

        vision_tokens = torch.as_tensor(vision_tokens, dtype=torch.float32)
        n_vision = vision_tokens.shape[0]
        assert vision_tokens.shape[1] == H, (
            f"vision tokens must be [n,{H}], got {tuple(vision_tokens.shape)}")
        assert n_vision == self.V_SLOTS * self.C_TOKENS, (
            f"expected {self.V_SLOTS}x{self.C_TOKENS} connector tokens, got {n_vision}")

        ids = torch.as_tensor(token_ids, dtype=torch.long).reshape(-1)
        assert ids.numel() == self.TEXT_LEN, (
            f"token_ids must be padded/truncated to tokenizer_max_length="
            f"{self.TEXT_LEN}, got {ids.numel()}")
        assert 1 + n_vision + ids.numel() <= PM, (
            f"prefix {1 + n_vision + ids.numel()} rows exceeds PREFILL_MAX_SEQ_LEN={PM}")

        # ---- row 0: state_proj(state) ------------------------------------------------
        # The stored weight is K-padded 32->64 (_weight_init_head), so the host must pad
        # the state vector the same way before the matmul or the columns do not line up.
        w, b = self._state_proj_cpu()
        s = torch.as_tensor(state, dtype=torch.float32).reshape(-1)
        assert s.numel() == self.STATE_DIM, (
            f"state must be [{self.STATE_DIM}], got {s.numel()}")
        s = F.pad(s, (0, self.ACTION_DIM_PAD - self.STATE_DIM), value=0.0)   # 32 -> 64
        st = (s @ w.T + b).reshape(1, H)

        # ---- rows 1..n_vision+..: images then text embeddings ------------------------
        # embedding_weight is the CPU bf16 copy _weight_init_lm keeps for exactly this
        # lookup; we do NOT gather on device (smolvlm2 does, because it decodes token by
        # token -- here the prefix text is known once per observation).
        txt = self.embedding_weight[ids].float()

        # ---- valid text length (needed BEFORE layout: state is packed after it) -------
        if text_mask is None:
            n_text = int(ids.numel())
        else:
            m = torch.as_tensor(text_mask).reshape(-1).bool()
            assert m.numel() == ids.numel()
            n_text = int(m.sum())
            assert bool(m[:n_text].all()), (
                "text_mask must be RIGHT-padded (all real slots first) -- an interior "
                "hole shifts every later RoPE position, and the device rotates by "
                "row index")

        # ---- the sqrt(hidden) embedding scale ----------------------------------------
        # Upstream scales image AND language embeddings by sqrt(hidden) (= sqrt(960),
        # ~30.98) and does NOT scale the state embedding:
        #     emb  = emb  * (emb.shape[-1] ** 0.5)      # images
        #     lang = lang * math.sqrt(lang.shape[-1])   # language
        # Omitting it does not blow up or NaN -- every downstream RMSNorm renormalizes,
        # so the prefix stays well-scaled and merely encodes the wrong thing. That is
        # precisely why no SNR gate here ever caught it.
        emb_scale = math.sqrt(H)
        vision_tokens = vision_tokens * emb_scale
        txt = txt * emb_scale

        # ---- layout: images -> real text -> state ------------------------------------
        # State is packed immediately after the REAL text tokens, not after the padded
        # 48-slot block. The device rotates row r by RoPE table entry r, while upstream
        # uses positions = cumsum(pad_mask) - 1; those two agree only when every valid
        # row is contiguous from 0. Leaving the text pad slots before the state row would
        # give state row index n_vision+TEXT_LEN but position n_vision+n_text.
        x = torch.full((PM, H), 1e-6, dtype=torch.float32)   # pad rows: 1e-6, never 0
        x[0:n_vision] = vision_tokens
        x[n_vision:n_vision + n_text] = txt[:n_text]
        x[n_vision + n_text] = st
        valid_len = n_vision + n_text + 1

        # Any exactly-zero row is a NaN factory in the epsilon-free RMSNorm. A real
        # embedding row can legitimately be zero-ish; check rather than assume.
        zero_rows = (x.abs().sum(1) == 0)
        if bool(zero_rows.any()):
            x[zero_rows] = 1e-6
        assert torch.isfinite(x).all(), "non-finite prefix input"

        self.dma_to_accelerator_memory(
            self.LM_INPUT_DRAM, x.reshape(-1).to(torch.bfloat16).contiguous())
        self._prefix_in, self._prefix_valid_len = x, valid_len
        # kept for --gate-upstream: the gate must feed upstream the SAME state vector the
        # device consumed (unpadded), not re-derive it and inherit a normalization diff.
        self._prefix_state_used = torch.as_tensor(state, dtype=torch.float32).reshape(-1)
        return x, valid_len

    def _state_proj_cpu(self):
        """CPU (weight, bias) for head.state_proj, as ACTUALLY STORED (K-padded 32->64).

        _weight_init_head keeps only the DRAM addresses, so read them back once and
        cache. bf16 round-trips through dma_read losslessly, so this is the stored value
        bit-for-bit -- and reading the DEVICE copy is strictly better than re-deriving
        from the checkpoint: it also proves the weight actually landed where the emitter
        thinks it did."""
        cached = getattr(self, "_state_proj_cpu_cache", None)
        if cached is None:
            w = self._read_bf16(self.state_proj_weight,
                                (self.HIDDEN_SIZE, self.ACTION_DIM_PAD),
                                label="state_proj.weight")
            b = self._read_bf16(self.state_proj_bias, (self.HIDDEN_SIZE,),
                                label="state_proj.bias")
            cached = (w, b)
            self._state_proj_cpu_cache = cached
        return cached

    def _lm_tensor_init(self):
        """Per-layer prefix buffers + flash staging. Separate from tensor_init (which
        owns the persistent KV cache) so the prefix is buildable on its own."""
        if getattr(self, "_lm_tensors_done", False):
            return
        PM, H, I = self.PREFILL_MAX_SEQ_LEN, self.HIDDEN_SIZE, self.INTERMEDIATE_SIZE
        D, G = self.HEAD_DIM, self.GROUP_SIZE
        KV = self.NUM_KV_HEADS * D
        bpe = 2
        a = lambda n: self.allocate_tensor_dram(n * bpe)
        self.LM_INPUT_DRAM     = a(PM * H)          # layer ping-pong
        self.LM_OUTPUT_DRAM    = a(PM * H)
        self.LM_PRE_NORM_DRAM  = a(PM * H)
        self.LM_RESIDUAL_DRAM  = a(PM * H)
        self.LM_Q_DRAM         = a(PM * H)
        self.LM_K_PROJ_DRAM    = a(PM * KV)
        self.LM_V_PROJ_DRAM    = a(PM * KV)
        self.LM_ATTN_RESULT_DRAM = a(PM * H)
        self.LM_O_PROJ_DRAM    = a(PM * H)
        self.LM_MLP_GATE_DRAM  = a(PM * I)
        self.LM_MLP_UP_DRAM    = a(PM * I)
        self.LM_MLP_MULT_DRAM  = a(PM * I)
        self.LM_MLP_DOWN_DRAM  = a(PM * I)
        # stacked-Q flash staging: one kv-group = PM tokens x G q-heads
        self.LM_QB = PM * G
        self.LM_FLASH_Q_DRAM   = a(self.LM_QB * D)
        self.LM_FLASH_K_DRAM   = a(self.LM_QB * D)
        self.LM_FLASH_V_DRAM   = a(self.LM_QB * D)
        self.LM_FLASH_OUT_DRAM = a(self.LM_QB * D)
        self.LM_FLASH_SCRATCH_DRAM = a((D + self.LM_QB) * self.LM_QB + self.LM_QB * D)
        # bisect probe: LM_PRE_NORM is written by BOTH norms in a layer, so ln1 is
        # gone by the time the layer ends. Allocated unconditionally (one [192,960]
        # buffer) rather than behind PREFIX_BISECT, so enabling the flag never shifts
        # any other tensor address -- a moved base silently invalidates cached bins.
        self.LM_P_NORM1_DRAM = a(PM * H)
        # ln2 needs its own probe too: compile_prefix's FINAL_NORM also targets
        # LM_PRE_NORM_DRAM, so by the time execution ends that buffer holds the final
        # norm, not ln2. Reading it there reported norm2 at rms 1.035 vs an expected
        # 0.098 (-19.9 dB, cos 0.684) while every op CONSUMING norm2 scored >0.9996 --
        # the giveaway that the probe, not the op, was wrong.
        self.LM_P_NORM2_DRAM = a(PM * H)
        # The attention bias lives on the STACKED grid, not the token grid: flash is
        # called with batch = aligned_seq_len = LM_QB, so it reads QB*QB elements. A
        # PM*PM allocation here would be 9x too small and the kernel would walk into the
        # next tensor. Same aligned formula as build_attn_bias -- the two MUST agree, so
        # the assert makes a future edit to either one fail loudly instead of silently
        # under-allocating (smolvlm2 sizes its FLASH_BIAS_DRAM as qmax*qmax the same way).
        QB_ALIGNED = ((self.LM_QB + self.VEC - 1) // self.VEC) * self.VEC
        assert QB_ALIGNED >= self.LM_QB
        if not hasattr(self, "PREFIX_BIAS_DRAM"):
            self.PREFIX_BIAS_DRAM = self.allocate_tensor_dram(QB_ALIGNED * QB_ALIGNED * bpe)
        self._lm_tensors_done = True

    def compile_prefix(self):
        """The prefix instruction stream: 32 SmolLM2 layers over [PM=192, 960], compiled
        ONCE as a single captured program. Returns its DRAM address.

        Ported from models/smolvlm2/smolvlm2_test.py::compile_prefill, which runs this
        exact stack and is HW-validated. FOUR DELIBERATE DELTAS -- everything else is the
        same shape:

          1. BIDIRECTIONAL, not causal. This is an observation encode, so the bias is the
             precomputed PREFIX_BIAS_DRAM mask and the runtime-seq-len/bucket dispatch
             machinery is dropped: PM is a compile-time constant here.
          2. NO LM head, no sampling, no preamble/postamble. final_norm is applied inside
             this program and it stops.
          3. THE PER-LAYER K/V IS PERSISTED. LAYER0_K/V_DRAM is not scratch -- it is the
             action expert's only prefix input and must survive all 10 denoise steps.
             Nothing else may reuse that region.
          4. New row layout (state + image + text), and rope at theta 10000.

        bf16 B-operands throughout (16-bit multipliers). Every per-row op is PBI via
        gpr_M_reg so the program is structure-bound, not M-bound. Flash stays
        address-static: only the two dimension GPRs are runtime, because PBI address
        injection into flash corrupts on re-execution and this program runs once per
        observation, repeatedly, across a LIBERO episode."""
        ue = self
        PM, H, I = self.PREFILL_MAX_SEQ_LEN, self.HIDDEN_SIZE, self.INTERMEDIATE_SIZE
        D, G = self.HEAD_DIM, self.GROUP_SIZE
        KV = self.NUM_KV_HEADS * D
        QB, bpe = self.LM_QB, 2

        self._lm_tensor_init()
        ue.start_capture()
        prog_addr = ue.get_program_dram_addr()
        m_reg = ue.alloc_isa_reg(); ue.generate_instruction_add_set(m_reg, PM)
        qb_reg = ue.alloc_isa_reg(); ue.generate_instruction_add_set(qb_reg, QB)

        def lm_matmul(M, K, N, A, la, proj, OUT, **kw):
            # bf16 B operand: no is_B_quantized / data_type / SCALE_DRAM_ADDR.
            ue.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=A,
                               B_DRAM_ADDR=la[f"{proj}_weight"], OUTPUT_DRAM_ADDR=OUT,
                               gpr_M_reg=m_reg, **kw)

        n_lm = len(self.lm_layer_addrs) if self.PREFIX_LAYERS is None else int(self.PREFIX_LAYERS)
        assert 1 <= n_lm <= len(self.lm_layer_addrs), f"PREFIX_LAYERS={self.PREFIX_LAYERS}"
        if n_lm != len(self.lm_layer_addrs):
            print(f"    [bisect] compiling only {n_lm}/{len(self.lm_layer_addrs)} prefix layers")
        for i, la in enumerate(self.lm_layer_addrs[:n_lm]):
            h_in = self.LM_INPUT_DRAM if i % 2 == 0 else self.LM_OUTPUT_DRAM
            h_out = self.LM_OUTPUT_DRAM if i % 2 == 0 else self.LM_INPUT_DRAM

            ue.rms_norm_core_dram(M=PM, N=H, A_DRAM_ADDR=h_in,
                                  OUTPUT_DRAM_ADDR=self.LM_PRE_NORM_DRAM,
                                  GAMMA_DRAM_ADDR=la["ln1_gamma"], gpr_M_reg=m_reg)
            if self.PREFIX_BISECT and i == n_lm - 1:
                # LM_PRE_NORM is written by BOTH norms in a layer; snapshot the first.
                self._probe_copy(self.LM_PRE_NORM_DRAM, self.LM_P_NORM1_DRAM, PM * H)
            lm_matmul(PM, H, H,  self.LM_PRE_NORM_DRAM, la, "q", self.LM_Q_DRAM)
            lm_matmul(PM, H, KV, self.LM_PRE_NORM_DRAM, la, "k", self.LM_K_PROJ_DRAM)
            lm_matmul(PM, H, KV, self.LM_PRE_NORM_DRAM, la, "v", self.LM_V_PROJ_DRAM)

            # K/V -> the PERSISTENT cache, one contiguous [PM,D] block per kv-head, then
            # rope K in place. The + h*KV_HEAD_STRIDE in the destination is the per-index
            # offset rule: without it every head lands on head 0's block and the expert
            # cross-attends to garbage -- finite and scrambled, never NaN.
            for h in range(self.NUM_KV_HEADS):
                k_dst = self.LAYER0_K_DRAM + i * self.KV_LAYER_STRIDE + h * self.KV_HEAD_STRIDE
                v_dst = self.LAYER0_V_DRAM + i * self.KV_LAYER_STRIDE + h * self.KV_HEAD_STRIDE
                self._ae_strided_copy(ue, self.LM_K_PROJ_DRAM + h * D * bpe, KV * bpe,
                                      k_dst, D * bpe, PM, D)
                self._ae_strided_copy(ue, self.LM_V_PROJ_DRAM + h * D * bpe, KV * bpe,
                                      v_dst, D * bpe, PM, D)
                # d64 rope is PBI-only; a bare call asserts N>=128 on the legacy path.
                ue.rope_hf_core_dram(M=PM, N=D, input_dram_addr=k_dst,
                                     output_dram_addr=k_dst,
                                     cos_dram_addr=self.ROPE_PACKED_DRAM,
                                     sin_dram_addr=self.ROPE_PACKED_DRAM + D * bpe,
                                     gpr_M_reg=m_reg)

            # per kv-group GQA: stack this group's G q-heads token-major, rope them with
            # the x G duplicated table, replicate K/V x G, one flash, un-stack.
            for kv_b in range(self.NUM_KV_HEADS):
                k_src = self.LAYER0_K_DRAM + i * self.KV_LAYER_STRIDE + kv_b * self.KV_HEAD_STRIDE
                v_src = self.LAYER0_V_DRAM + i * self.KV_LAYER_STRIDE + kv_b * self.KV_HEAD_STRIDE
                for g in range(G):
                    self._ae_strided_copy(ue, self.LM_Q_DRAM + (kv_b * G + g) * D * bpe,
                                          H * bpe, self.LM_FLASH_Q_DRAM + g * D * bpe,
                                          G * D * bpe, PM, D)
                ue.rope_hf_core_dram(M=QB, N=D, input_dram_addr=self.LM_FLASH_Q_DRAM,
                                     output_dram_addr=self.LM_FLASH_Q_DRAM,
                                     cos_dram_addr=self.ROPE_PACKED_GQA_DRAM,
                                     sin_dram_addr=self.ROPE_PACKED_GQA_DRAM + D * bpe,
                                     gpr_M_reg=qb_reg)
                ue.accelerator_memory_to_sram(k_src, 0x10000, PM * D)
                self._ae_duplicate_gqa_rows(ue, PM, 0x10000, self.LM_FLASH_K_DRAM)
                ue.accelerator_memory_to_sram(v_src, 0x20000, PM * D)
                self._ae_duplicate_gqa_rows(ue, PM, 0x20000, self.LM_FLASH_V_DRAM)
                ue.unified_attention_core(
                    batch=QB, aligned_seq_len=QB, head_dim=D,
                    Q_DRAM_ADDR=self.LM_FLASH_Q_DRAM, K_DRAM_ADDR=self.LM_FLASH_K_DRAM,
                    V_DRAM_ADDR=self.LM_FLASH_V_DRAM,
                    BIAS_DRAM_ADDR=self.PREFIX_BIAS_DRAM,
                    OUTPUT_DRAM_ADDR=self.LM_FLASH_OUT_DRAM,
                    SCRATCH_DRAM_ADDR=self.LM_FLASH_SCRATCH_DRAM,
                    IDENTITY_DRAM_ADDR=self.identity_addr,
                    gpr_batch_reg=qb_reg, gpr_aligned_seq_len_reg=qb_reg)
                for g in range(G):
                    self._ae_strided_copy(ue, self.LM_FLASH_OUT_DRAM + g * D * bpe,
                                          G * D * bpe,
                                          self.LM_ATTN_RESULT_DRAM + (kv_b * G + g) * D * bpe,
                                          H * bpe, PM, D)

            lm_matmul(PM, H, H, self.LM_ATTN_RESULT_DRAM, la, "o", self.LM_O_PROJ_DRAM)
            ue.eltwise_core_dram(M=PM, N=H, dram_a=h_in, dram_b=self.LM_O_PROJ_DRAM,
                                 dram_out=self.LM_RESIDUAL_DRAM,
                                 mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=m_reg)
            # split residual+norm: the fused post_add norm needs 4 advancing PBI pointers
            # against a limit of 3, so it would unroll M statically 32 layers deep.
            ue.rms_norm_core_dram(M=PM, N=H, A_DRAM_ADDR=self.LM_RESIDUAL_DRAM,
                                  OUTPUT_DRAM_ADDR=self.LM_PRE_NORM_DRAM,
                                  GAMMA_DRAM_ADDR=la["ln2_gamma"], gpr_M_reg=m_reg)
            # SiLU: fused vs composed is a MEASURED experiment, not a preference.
            # The fused LALU silu_enable is piecewise-approximate -- pi05 measured -6 dB
            # from ONE of them. There are 32 here, one per layer, and the hardware's
            # prefix error compounds per layer in exactly the shape an approximate
            # activation would produce (it BEATS a strict-bf16 sim at layer 0, 46.3 vs
            # 43.4 dB, then falls 7 dB BELOW it by layer 31 -- accumulation width alone
            # cannot cross over like that). PREFIX_FUSED_SILU flips it in one place so
            # the two can be compared on hardware.
            if self.PREFIX_BISECT and i == n_lm - 1:
                self._probe_copy(self.LM_PRE_NORM_DRAM, self.LM_P_NORM2_DRAM, PM * H)
            if self.PREFIX_FUSED_SILU:
                lm_matmul(PM, H, I, self.LM_PRE_NORM_DRAM, la, "gate",
                          self.LM_MLP_GATE_DRAM, silu_enable=True)
            else:
                lm_matmul(PM, H, I, self.LM_PRE_NORM_DRAM, la, "gate",
                          self.LM_MLP_GATE_DRAM)
                # silu_core_dram does an N x N IDENTITY matmul to get sigmoid, and the
                # stored identity is UE_VECTOR_SIZE x UE_VECTOR_SIZE (64x64). N is
                # therefore the IDENTITY's width, NOT the tensor's: SiLU is elementwise,
                # so view [PM, I] as [PM*I/64, 64] and the 64x64 identity applies. Passing
                # N=I=2560 asks for a 2560x2560 identity that does not exist -- the kernel
                # reads whatever follows in DRAM as the identity and the sigmoid is
                # garbage (measured: cos 0.53, and the program ballooned 3.4 -> 11.5 MB
                # from the K=N=2560 matmul). Same view the expert's time MLP uses.
                silu_core_dram(ue, M=(PM * I) // UE_VECTOR_SIZE, N=UE_VECTOR_SIZE,
                               A_DRAM_ADDR=self.LM_MLP_GATE_DRAM,
                               OUTPUT_DRAM_ADDR=self.LM_MLP_GATE_DRAM,
                               IDENTITY_DRAM_ADDR=self.identity_addr)
            lm_matmul(PM, H, I, self.LM_PRE_NORM_DRAM, la, "up", self.LM_MLP_UP_DRAM)
            eltwise_mul_core_dram(ue, size=PM * I, A_DRAM_ADDR=self.LM_MLP_GATE_DRAM,
                                  B_DRAM_ADDR=self.LM_MLP_UP_DRAM,
                                  OUTPUT_DRAM_ADDR=self.LM_MLP_MULT_DRAM)
            lm_matmul(PM, I, H, self.LM_MLP_MULT_DRAM, la, "down", self.LM_MLP_DOWN_DRAM)
            ue.eltwise_core_dram(M=PM, N=H, dram_a=self.LM_RESIDUAL_DRAM,
                                 dram_b=self.LM_MLP_DOWN_DRAM, dram_out=h_out,
                                 mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=m_reg)

        final = self.LM_INPUT_DRAM if n_lm % 2 == 0 else self.LM_OUTPUT_DRAM
        ue.rms_norm_core_dram(M=PM, N=H, A_DRAM_ADDR=final,
                              OUTPUT_DRAM_ADDR=self.LM_PRE_NORM_DRAM,
                              GAMMA_DRAM_ADDR=self.final_norm_addr, gpr_M_reg=m_reg)
        self.PREFIX_HIDDEN_DRAM = self.LM_PRE_NORM_DRAM

        ue.generate_instruction_halt()
        ue.release_isa_reg(); ue.release_isa_reg()
        ue.stop_capture()
        raw = bytearray()
        for inst in ue.capture_buffer:
            raw.extend(inst.get_bytes())
        ue.dma_write(DMA_DEVICE_H2C, prog_addr, raw, len(raw))
        ue.allocate_program_dram(len(raw))
        ue.clear_capture_buffer()
        self._prefix_program_addr = prog_addr
        print(f"    prefix ({self.NUM_LAYERS} layers, PM={PM}, GQA g{G}): "
              f"{len(raw) / 1e6:.2f} MB @0x{prog_addr:X}")
        return prog_addr

    def _precompute_time_embeddings(self):
        """Host-precompute all N sincos timestep embeddings as ONE contiguous [N, 512]
        bf16 table in params DRAM, and return its address.

        Ported from pi05_libero_test.py::_ae_build_sincos_table. The reason it is a
        TABLE and not a per-step host DMA is structural: the 10-step Euler loop is a real
        hardware loop_start/loop_end, and you cannot run Python between hardware loop
        iterations to push a fresh embedding. The loop body reads the current row through
        a GPR-held address that it advances itself (see _emit_time_embed_from_table).

        Schedule matches the Euler loop exactly: t = 1.0, dt = -1/N, so
        t = 1.0, 0.9, ... 0.1 for N=10. Periods are log-spaced min_period -> max_period;
        sin FIRST then cos (concat, not interleave) -- lerobot/openpi
        create_sinusoidal_pos_embedding order.

        Computed in fp64 on the host and cast once: this is input-independent, so it is
        exact constants, and there is no sine or cosine primitive anywhere in the ISA to
        lower it onto even if we wanted it on device.

        WHAT DOES **NOT** TRANSFER FROM pi05: _ae_precompute_cond_table folds the whole
        sincos -> time_mlp_in -> silu -> time_mlp_out -> silu chain into the table,
        because pi05's AdaRMSNorm conditioning vector is input-independent. OURS IS NOT:
        action_time_mlp_in consumes concat(action_in_proj(x_t), time_emb) and x_t changes
        every Euler step. Only the RAW SINUSOID is precomputable here; everything from
        action_time_mlp_in onward runs on device, once per step.

        Stored 512 wide, not 480: the innermost dim must be %64 so a row DMAs into the
        padded expert slot without a sub-row copy. Lanes [480:512] are zero and must stay
        zero -- the expert's RMSNorm gamma fold depends on it."""
        HEAD = self._cfg["action_head"]
        N, pad = HEAD["num_denoise_steps"], self.E_HIDDEN_PAD
        table, _ts = build_time_table(self._cfg, pad_to=pad)

        self.AE_TIME_TABLE_DRAM = self.get_params_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, self.AE_TIME_TABLE_DRAM,
                       table.flatten(), N * pad * 2)
        self.allocate_params_dram(N * pad * 2)
        # UNITS: the PBI DRAM_ADDR field this pointer feeds is a 35-bit WORD address
        # (ue_35bit_addr_shifter == byte >> 3), NOT a byte address. Advancing by the row's
        # BYTE size would stride 8 rows per step and read garbage timesteps 0,8,16,...
        # Stored in words, and named so, so the add_imm below cannot be misread.
        self.AE_TIME_ROW_WORDS = ue_35bit_addr_shifter(pad * 2)
        return self.AE_TIME_TABLE_DRAM

    def _emit_time_embed_from_table(self, addr_reg, dst_dram):
        """Read the CURRENT timestep row into dst_dram and advance addr_reg to the next.

        Ported from pi05_libero_test.py::_ae_time_embed_from_table. Two instructions of
        substance: a general_reg_src read (the address lives in a GPR, so one compiled
        body serves all N steps) and an add_imm that walks the pointer one row on.

        NOTE the divergence from pi05: there, this fed the time MLP directly, because the
        MLP was input-independent. Here the row is only HALF of action_time_mlp_in's
        input -- the caller must place it at the split-column offset alongside
        action_in_proj(x_t). See _emit_suffix_embed.

        _addr_tmp hazard: ue_selector's runtime_addr() and append_row() share one scratch
        GPR, so emit every op consuming a computed address BEFORE any append_row()."""
        pad = self.E_HIDDEN_PAD
        sram = 0x00000
        self.accelerator_memory_to_sram(0, sram, pad, general_reg_src=addr_reg)
        self.sram_to_accelerator_memory(sram, dst_dram, pad)
        self.generate_instruction_add_imm(addr_reg, self.AE_TIME_ROW_WORDS)

    # ----------------------------------------------------------------------------------
    # expert-stage support: buffers, tables, and the three emit primitives
    # ----------------------------------------------------------------------------------
    #
    # These mirror compile_prefix's contracted local helpers (lm_matmul / strided_copy /
    # duplicate_gqa_rows) but live as METHODS rather than closures, because the expert
    # emits them from a rolled loop body spanning 32 layers and because
    # _emit_expert_layer takes an explicit `ue` so the same body can later be re-emitted
    # onto a second engine (the multi-engine backlog) without restructuring.

    def _ae_is_self_attn(self, layer_idx):
        """Layer type. Config-driven on purpose -- the parity is UNVERIFIED, and if it is
        inverted every one of the 32 layers is wrong in a way that reads as generic drift
        rather than as a structural bug. Mirrors VeraPulseRef.forward_expert exactly:
            is_self = (i % every == 0) == self_attn_on_even"""
        every = self.E_SELF_EVERY
        return ((layer_idx % every) == 0) == bool(self.EXPERT_SELF_ON_EVEN)

    def _ae_cross_prefix_layer(self, layer_idx):
        """Which LM layer's cached K/V a cross-attention layer reads. Also UNVERIFIED.
        The reference uses prefix_kv[i % len(prefix_kv)], which for 32 == 32 layers is
        layer i -- that is the "same_index" default here."""
        m = self.EXPERT_CROSS_LAYER_MAP
        if m == "same_index":
            return layer_idx
        if m == "last":
            return self.NUM_LAYERS - 1
        raise ValueError(f"unknown cross_prefix_layer_map {m!r}")

    def _ae_rope_positions(self, valid_prefix_len=None):
        """Suffix RoPE positions [SUFFIX_LEN_PAD]. Continuing from the prefix means
        starting at the VALID prefix length, not the padded one: prefix positions are
        cumsum(attention_mask)-1, so the last real prefix token sits at valid_len-1 and
        the first action token continues at valid_len. Using the padded 192 here would
        shift every action token by the pad count -- small, smooth, and wrong.

        The 14 suffix pad rows (50..63) get positions like any other row: they are
        COMPUTED, never skipped, and masked out by the attention bias instead.

        FAULT 8. The base is the VALID prefix length, NOT the nominal PREFIX_LEN=177
        (= 1 state + 128 image + 48 PADDED text slots). Upstream (smolvla/modeling.py,
        denoise_step):
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)
            position_ids   = prefix_offsets[:, None] + torch.cumsum(suffix_pad, 1) - 1
        i.e. the first action token continues from the last REAL prefix token, so the
        base is 1 + n_image_tokens + n_REAL_text_tokens (e.g. 141 for a 12-token prompt),
        which is DATA-DEPENDENT. Using 177 shifted every action token's RoPE phase by
        (177 - valid_len), e.g. 36 positions.

        `valid_prefix_len=None` falls back to the nominal PREFIX_LEN so the table can be
        built at allocation time, before any prefix has run; run_denoise re-DMAs it with
        the real value (see _ae_refresh_runtime_constants)."""
        if not self.EXPERT_ROPE_CONTINUES:
            base = 0
        elif valid_prefix_len is None:
            base = self.PREFIX_LEN
        else:
            base = int(valid_prefix_len)
            assert 0 < base <= self.PREFILL_MAX_SEQ_LEN, (
                f"valid_prefix_len {base} outside [1,{self.PREFILL_MAX_SEQ_LEN}]")
        return torch.arange(base, base + self.SUFFIX_LEN_PAD, dtype=torch.float32)

    def _ae_rope_positions_cross(self):
        """FAULT 7. CROSS-attention query positions: REBASED to 0..chunk-1.

        forward_cross_attn_layer (smolvla/modeling.py):
            exp_pos = exp_pos - torch.min(exp_pos, dim=1, keepdim=True).values
            eq = apply_rope(eq, exp_pos)
        so on cross layers the expert's query is rotated at positions 0.., NOT at the
        prefix-continued positions. (The reprojected key is not roped at all -- the
        cached K already carries the prefix rotation.) VeraPulseRef.forward_expert does
        the same thing: `rope_tables(torch.arange(s), ...)` on the cross branch.

        Independent of the valid prefix length, so this table is a true compile-time
        constant and never needs re-DMA'ing."""
        return torch.arange(self.SUFFIX_LEN_PAD, dtype=torch.float32)

    def _ae_packed_rope_table(self, positions):
        """[S, 2*D] bf16 rows of [cos(D) | sin(D)] with sin's LOWER half pre-negated.

        That layout is the kernel's, not a choice: rope_hf_core_dram's d64 (padded-split)
        path asserts sin_dram_addr == cos_dram_addr + D*2, i.e. cos and sin must be
        contiguous within one row, and it computes
            out = x*cos + [x_hi*sin_lo | x_lo*sin_hi]
        with add-only hardware -- so the minus sign of rotate_half([-x_hi, x_lo]) has to
        be baked into the table's first half on the host. Built in fp32 and cast once."""
        D, theta = self.HEAD_DIM, self.ROPE_THETA
        inv = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
        f = torch.outer(positions.float(), inv)
        cos = torch.cat([f.cos(), f.cos()], -1)
        sin = torch.cat([f.sin(), f.sin()], -1)
        sin[:, : D // 2] *= -1.0                 # rotate_half's minus, folded in
        return torch.cat([cos, sin], -1).to(torch.bfloat16).contiguous()

    def _ae_tensor_init(self):
        """Allocate every action-expert / suffix-embed activation buffer, plus the two
        constant tensors the expert needs in DRAM (its RoPE tables and its two attention
        biases). Idempotent -- compile_denoise_loop calls it, and calling it twice would
        otherwise leak the whole set on a recompile.

        Deliberately SEPARATE from tensor_init's prefix TODO block: the expert must be
        buildable and gateable before compile_prefix exists (stage 6 of the bring-up
        order feeds it a CPU-provided prefix KV). It does depend on tensor_init having
        claimed the persistent KV cache, which is asserted below."""
        if getattr(self, "_ae_tensors_ready", False):
            return
        assert hasattr(self, "LAYER0_K_DRAM"), (
            "_ae_tensor_init requires tensor_init() to have allocated the prefix KV "
            "cache -- the expert's cross-attention layers read it directly")

        HP, I, Q, KV = self.E_HIDDEN_PAD, self.E_INTER, self.E_Q_OUT, self.E_KV_OUT
        D, G = self.HEAD_DIM, self.GROUP_SIZE
        M, ADP = self.SUFFIX_LEN_PAD, self.ACTION_DIM_PAD
        PM, bpe = self.PREFILL_MAX_SEQ_LEN, 2
        QB = M * G                       # stacked-Q flash rows: 64 tokens x 3 q-heads

        # THE COINCIDENCE THAT REMOVES A WHOLE COPY. unified_attention_core's contract is
        # Q[batch,D], K/V[aligned_seq_len,D], bias[batch,aligned_seq_len]. The stacked-Q
        # batch is 64*3 = 192, which is exactly the padded prefix length -- so a
        # cross-attn layer can point flash straight at a [192,64] reprojected-cache slice
        # with no replication at all, while a self-attn layer replicates its own 64 kv
        # rows x3 to 192 (that replication is mathematically free: softmax over G exact
        # copies of each key splits each weight G ways and the duplicated V sums it back).
        # Asserted rather than assumed: if either length moves, the cross path needs the
        # same duplicate_gqa_rows treatment as the self path.
        self.AE_FLASH_ROWS = QB
        assert QB == PM, (
            f"expert flash geometry assumes SUFFIX_LEN_PAD*GROUP_SIZE ({QB}) == "
            f"PREFILL_MAX_SEQ_LEN ({PM}); with those unequal the cross-attn K/V must be "
            f"replicated x{G} like the self-attn path and the bias re-shaped")

        def a(n):
            return self.allocate_tensor_dram(n * bpe)

        # ---- flow state + suffix embed ----
        self.AE_XT_DRAM      = a(M * ADP)        # x_t, [50,7] real inside [64,64]
        self.AE_XT_NEXT_DRAM = a(M * ADP)        # Euler target (never aliased with x_t)
        self.AE_VT_DRAM      = a(M * ADP)        # dt * action_out_proj(v)
        self.AE_AIN_DRAM     = a(M * HP)         # action_in_proj(x_t)
        self.AE_TIME_ROW_DRAM = a(HP)            # one timestep row, before broadcast
        # per-step x_t snapshots: [N_STEPS, 64, 64]
        self.AE_STEP_SNAP_DRAM = a(self.N_STEPS * self.SUFFIX_LEN_PAD * self.ACTION_DIM_PAD)
        self.AE_TMLP_IN_DRAM = a(M * 2 * HP)     # the [64,1024] SPLIT-COLUMN buffer
        self.AE_TMLP_HID_DRAM = a(M * HP)
        self.AE_TMLP_SILU_DRAM = a(M * HP)

        # ---- per-layer stream (ping-pong) ----
        self.AE_IO_A_DRAM    = a(M * HP)
        self.AE_IO_B_DRAM    = a(M * HP)
        self.AE_PRE_NORM_DRAM = a(M * HP)
        self.AE_RESIDUAL_DRAM = a(M * HP)
        self.AE_Q_DRAM       = a(M * Q)
        self.AE_K_PROJ_DRAM  = a(M * KV)
        self.AE_V_PROJ_DRAM  = a(M * KV)
        # per-kv-head contiguous [64,64] slices of this step's own K/V (self-attn only)
        self.AE_KV_HEAD_STRIDE = M * D * bpe
        self.AE_K_HEADS_DRAM = a(self.NUM_KV_HEADS * M * D)
        self.AE_V_HEADS_DRAM = a(self.NUM_KV_HEADS * M * D)
        # ---- self-attention COMBINED [prefix ; suffix] K/V staging (FAULT 4) --------
        # SmolVLMWithExpert.forward_attn_layer, use_cache=True / fill_kv_cache=False:
        #     k = torch.cat([kv[idx]["key_states"], k], dim=1)
        #     v = torch.cat([kv[idx]["value_states"], v], dim=1)
        # i.e. a SELF layer attends over prefix_len + chunk keys, NOT over its own 50
        # action tokens alone. Without this half the expert stack never sees the
        # observation at all (CPU: cos 0.413 -> 0.784 from this fault alone).
        #
        # Padded geometry: the cached prefix occupies PM=192 rows (valid rows first,
        # pad rows after, masked by the bias) and the suffix occupies SUFFIX_LEN_PAD=64,
        # so the combined key/value length is PM + M = 256 -- a multiple of
        # UE_VECTOR_SIZE, which unified_attention_core requires of aligned_seq_len.
        # batch stays QB=192 (the stacked-Q rows) and batch <= aligned_seq_len is the
        # kernel's only ordering constraint, so the old x G key replication is NOT
        # needed on the combined path: it only ever existed to lift the 64-row suffix
        # K/V up to the 192-row batch floor.
        #
        # ONE buffer pair, not five: the combined block is consumed by the flash call
        # of the kv-group that just built it, so the 5 groups reuse it in turn. Rebuilt
        # per layer per step (16 self layers x 10 Euler steps) out of the untouched
        # cache plus this layer's freshly roped suffix K/V.
        self.AE_COMBINED_LEN = PM + M                # 256 = prefix pad + suffix pad
        self.AE_CK_DRAM = a(self.AE_COMBINED_LEN * D)
        self.AE_CV_DRAM = a(self.AE_COMBINED_LEN * D)
        # ---- cross-attention K/V reprojection staging (FAULT 5) ---------------------
        # Cross layers do NOT feed the raw cached VLM K/V to flash. Upstream
        # (SmolVLMWithExpert.forward_cross_attn_layer) does:
        #     _k = k.reshape(*k.shape[:2], -1)                 # [B, S, 320] token-major
        #     ek = exp_layer.self_attn.k_proj(_k).view(..., nkv, head_dim)
        # and the checkpoint proves it independently: odd-layer k_proj is (320,320) --
        # the VLM kv width -- while even-layer k_proj is (320,480), the expert hidden
        # width. A (320,320) weight is only usable against a 320-wide activation.
        #
        # The cache is HEAD-MAJOR ([layer][kv_head][pos][D] blocks), but the matmul needs
        # a TOKEN-MAJOR [PM, 320] matrix, so the 5 head blocks are interleaved per token
        # by 5 strided copies (the exact inverse of the self-attn de-interleave above).
        # Then the reprojected [PM,320] is de-interleaved back into 5 contiguous [PM,D]
        # head blocks for flash.
        #
        # All four are SCRATCH, rebuilt from the (untouched) cache at every cross layer
        # of every Euler step, so one set is shared by all 16 cross layers x 10 steps.
        self.AE_XKV_HEAD_STRIDE = PM * D * bpe
        self.AE_XK_TOK_DRAM  = a(PM * KV)   # cached K interleaved token-major [192,320]
        self.AE_XV_TOK_DRAM  = a(PM * KV)
        self.AE_XK_PROJ_DRAM = a(PM * KV)   # k_proj(cached K), still token-major
        self.AE_XV_PROJ_DRAM = a(PM * KV)
        # de-interleaved [5,192,64] flash operands. Same byte size as the *_TOK buffers;
        # kept SEPARATE rather than aliased so the gather -> matmul -> scatter chain has
        # no read/write overlap to reason about.
        self.AE_XK_HEADS_DRAM = a(self.NUM_KV_HEADS * PM * D)
        self.AE_XV_HEADS_DRAM = a(self.NUM_KV_HEADS * PM * D)

        self.AE_ATTN_RESULT_DRAM = a(M * Q)
        self.AE_O_PROJ_DRAM  = a(M * HP)
        self.AE_MLP_GATE_DRAM = a(M * I)
        self.AE_MLP_UP_DRAM  = a(M * I)
        self.AE_MLP_MULT_DRAM = a(M * I)
        self.AE_MLP_DOWN_DRAM = a(M * HP)
        self.AE_FINAL_DRAM   = a(M * HP)
        # bisect probes: AE_PRE_NORM is written by BOTH norms in a layer, and the suffix
        # embed's output (layer-0 input) is overwritten by the ping-pong.
        self.AE_P_NORM1_DRAM = a(M * HP)
        self.AE_P_NORM2_DRAM = a(M * HP)
        self.AE_P_EMBED_DRAM = a(M * HP)

        # ---- flash staging (fixed operands; flash itself stays address-static) ----
        self.AE_FLASH_Q_DRAM   = a(QB * D)
        self.AE_FLASH_K_DRAM   = a(QB * D)
        self.AE_FLASH_V_DRAM   = a(QB * D)
        self.AE_FLASH_OUT_DRAM = a(QB * D)
        # scratch layout is the kernel's: V.T[D,S] + score[S,S] + scaled_q[batch,D],
        # with S = aligned_seq_len. Sized for the LARGEST S any flash call here uses --
        # the self path's combined PM+M=256 (FAULT 4), not the cross path's 192. The
        # dynamic path re-bases the three sub-buffers off the runtime aligned_seq_len,
        # so one over-sized allocation serves both widths; under-sizing it would let the
        # score block run off the end of the buffer into the next tensor.
        CW = self.AE_COMBINED_LEN
        self.AE_FLASH_SCRATCH_DRAM = a((D + CW) * CW + QB * D)

        # ---- constants: RoPE tables ----
        pos = self._ae_rope_positions()
        packed = self._ae_packed_rope_table(pos)                  # [64, 128]
        self.AE_ROPE_PACKED_DRAM = a(M * 2 * D)
        self.dma_to_accelerator_memory(self.AE_ROPE_PACKED_DRAM, packed.flatten())
        # row r of the stacked-Q buffer holds token r//G, so the table is duplicated
        # token-major the same way the Q rows are.
        gqa = packed.repeat_interleave(G, dim=0)                  # [192, 128]
        self.AE_ROPE_GQA_DRAM = a(QB * 2 * D)
        self.dma_to_accelerator_memory(self.AE_ROPE_GQA_DRAM, gqa.flatten())
        # FAULT 7: a SECOND table, identical in layout (packed [cos|sin] with sin's lower
        # half pre-negated, then x G token-major replication), but at REBASED positions
        # 0..SUFFIX_LEN_PAD-1. forward_cross_attn_layer ropes its query at
        #     exp_pos - exp_pos.min()      (i.e. 0..chunk-1)
        # and does NOT rope the reprojected key, so CROSS layers must use this table and
        # SELF layers must keep the prefix-continued one above. Position-only constant --
        # independent of the valid prefix length, so it is never re-DMA'd.
        gqa_x = self._ae_packed_rope_table(
            self._ae_rope_positions_cross()).repeat_interleave(G, dim=0)
        self.AE_ROPE_GQA_CROSS_DRAM = a(QB * 2 * D)
        self.dma_to_accelerator_memory(self.AE_ROPE_GQA_CROSS_DRAM, gqa_x.flatten())

        # ---- constants: attention biases -------------------------------------------
        # Built by _ae_attn_biases at the NOMINAL prefix length so the buffers exist
        # before any prefix has run; run_denoise re-DMAs both with the data-dependent
        # valid length (fault 8 / _ae_refresh_runtime_constants). Both are DATA, not
        # program, so the refresh needs no recompile.
        self_mask, cross_mask = self._ae_attn_biases(self.PREFIX_LEN)
        self.AE_BIAS_SELF_DRAM = a(QB * self.AE_COMBINED_LEN)
        self.dma_to_accelerator_memory(
            self.AE_BIAS_SELF_DRAM,
            self_mask.to(torch.bfloat16).contiguous().flatten())
        self.AE_BIAS_CROSS_DRAM = a(QB * PM)
        self.dma_to_accelerator_memory(
            self.AE_BIAS_CROSS_DRAM,
            cross_mask.to(torch.bfloat16).contiguous().flatten())

        # x_t is host-written per inference, but an address DRAM never saw returns EIO on
        # dma_read (it never entered the ECC-initialized set), so a probe before the first
        # write would fail rather than read zeros. Seed it.
        self.dma_to_accelerator_memory(
            self.AE_XT_DRAM, torch.zeros(M * ADP, dtype=torch.bfloat16))
        self._ae_tensors_ready = True

    def _ae_attn_biases(self, valid_prefix_len):
        """Build (self_bias [QB, PM+M], cross_bias [QB, PM]) for a given VALID prefix
        length. Pure host tensors -- no DRAM, no side effects -- so the same code builds
        the placeholders at allocation time and the real ones per inference.

        FAULT 8 / follow-up: the valid cutoff is 1 + n_images + n_REAL_text, NOT the
        nominal PREFIX_LEN. Both biases MUST use the same cutoff; if they disagreed,
        self and cross layers would mask different prefix widths."""
        PM, M, G = self.PREFILL_MAX_SEQ_LEN, self.SUFFIX_LEN_PAD, self.GROUP_SIZE
        QB = M * G
        valid = int(valid_prefix_len)
        assert 0 < valid <= PM, f"valid_prefix_len {valid} outside [1,{PM}]"
        # -inf, not a large negative: softmax must zero those columns EXACTLY, and a
        # -1e4 leaks a small but systematic contribution from every pad column.
        neg = float("-inf")
        # self: CAUSAL within the 50 real action tokens -- NOT bidirectional. It is
        # tempting to assume the observation-encoder's bidirectional rule here; that was
        # the bug. embed_suffix (smolvla/modeling.py) returns
        #     att = torch.ones(bsz, x.shape[1], dtype=torch.bool)
        # and make_att_2d_masks does
        #     cumsum = att_masks.cumsum(1); att_2d = cumsum[:,None,:] <= cumsum[:,:,None]
        # With all-ones att_masks cumsum = [1..50], so att_2d[i,j] = (j <= i): strictly
        # lower-triangular INCLUDING the diagonal. Action token i sees actions 0..i only.
        # (Mirrors VeraPulseRef.denoise's self.expert_causal_suffix block; on CPU adding
        # it moved the action chunk from cos 0.919 to ~119 dB vs upstream.)
        #
        # Stacked-Q layout: row r is token r//G, column c is token c//G, so the [QB,QB]
        # mask is the [tok,tok] causal mask expanded x G on both axes -- it is a real 2-D
        # tensor now, no longer one broadcast row.
        # Stacked-Q layout: flash row r is action token r//G. COLUMNS are plain suffix
        # token indices (0..M-1): FAULT 4 removed the x G key replication -- it only
        # existed to lift the 64-row suffix K/V to the 192-row batch floor, and the
        # combined [prefix ; suffix] sequence is 256 rows, comfortably above it. So the
        # suffix column block is [QB, M], not [QB, QB].
        row_tok = torch.arange(QB) // G                            # [QB] row -> token
        col_tok = torch.arange(M)                                  # [M]  col -> token
        pad_col = (col_tok >= self.SUFFIX_LEN)[None, :]            # [1,M] pad columns
        future  = col_tok[None, :] > row_tok[:, None]              # [QB,M] causal
        self_mask = torch.where(pad_col | future,
                                torch.tensor(neg), torch.tensor(0.0))
        # Pad rows (row_tok >= SUFFIX_LEN) keep every real column visible -- col_tok <=
        # row_tok holds there -- so no row is all -inf and softmax never NaNs. Those rows
        # are still computed; their output is simply never read.
        #
        # FAULT 4 widens this from [QB, QB] to [QB, PM + SUFFIX_LEN_PAD]. denoise_step:
        #     prefix_2d = prefix_pad[:, None, :].expand(bsz, suffix_len, prefix_len)
        #     suffix_2d = make_att_2d_masks(suffix_pad, suffix_att)      # causal
        #     full_2d   = torch.cat([prefix_2d, suffix_2d], dim=2)
        # so every action row sees ALL VALID prefix columns plus actions 0..i. The
        # prefix half is therefore a ROW-CONSTANT 0/-inf pattern -- exactly the cross
        # bias -- and the suffix half is the causal block built above. Column order
        # matches the combined K/V staging: prefix rows first, suffix rows after.
        #
        # VALID PREFIX LENGTH: `valid`, the data-dependent 1 + images + REAL text (e.g.
        # 141), NOT the nominal PREFIX_LEN=177. Both biases share it by construction.
        prefix_cols = torch.where(torch.arange(PM) < valid, 0.0, neg)
        self_mask = torch.cat([prefix_cols.expand(QB, PM), self_mask], dim=1)
        # ROW STRIDE IS PM + M = AE_COMBINED_LEN = 256, and the PREFIX block is only the
        # FIRST PM = 192 columns of each row; columns [192:256) are the suffix causal +
        # pad block built above and must never be touched by the valid-length cutoff.
        # No row is all -inf: valid >= 1 always leaves prefix column 0 visible on every
        # row, pad rows included, so softmax cannot produce NaN.
        assert self_mask.shape == (QB, self.AE_COMBINED_LEN), self_mask.shape
        assert bool((~torch.isinf(self_mask)).any(dim=1).all()), "all -inf row -> NaN"
        # cross: every action row attends every VALID prefix column, and nothing else.
        cross_mask = torch.where(torch.arange(PM) < valid, 0.0, neg).expand(QB, PM)
        return self_mask.contiguous(), cross_mask.contiguous()

    def _ae_refresh_runtime_constants(self, valid_prefix_len=None):
        """FAULT 8 (+ the two bias placeholders). Re-DMA every expert constant that
        depends on the DATA-DEPENDENT valid prefix length:
            AE_ROPE_PACKED_DRAM / AE_ROPE_GQA_DRAM   (self-branch Q/K rope base)
            AE_BIAS_SELF_DRAM   / AE_BIAS_CROSS_DRAM (valid-prefix cutoff)
        All four are DRAM DATA, not program, so this needs no recompile -- exactly the
        property AE_BIAS_CROSS_DRAM was already documented to have. AE_ROPE_GQA_CROSS_DRAM
        is NOT refreshed: its positions are rebased to 0 and carry no prefix dependence.

        Source of truth is embed_and_concat_prefix's valid_len, stashed as
        self._prefix_valid_len. If run_denoise is called before run_prefix there is no
        real prefix in the KV cache at all, so the fallback to the nominal PREFIX_LEN is
        cosmetic -- it only keeps the buffers self-consistent; it is warned about, once."""
        if valid_prefix_len is None:
            valid_prefix_len = getattr(self, "_prefix_valid_len", None)
        if valid_prefix_len is None:
            valid_prefix_len = self.PREFIX_LEN
            if not getattr(self, "_ae_valid_len_warned", False):
                print(f"  WARNING: no prefix has run (_prefix_valid_len unset); expert "
                      f"rope/bias fall back to the NOMINAL PREFIX_LEN="
                      f"{self.PREFIX_LEN}. Results are only meaningful once run_prefix "
                      f"has populated the KV cache.")
                self._ae_valid_len_warned = True
        valid = int(valid_prefix_len)
        if getattr(self, "_ae_constants_valid_len", None) == valid:
            return valid                       # already staged for this prefix
        G, M = self.GROUP_SIZE, self.SUFFIX_LEN_PAD

        packed = self._ae_packed_rope_table(self._ae_rope_positions(valid))
        self.dma_to_accelerator_memory(self.AE_ROPE_PACKED_DRAM, packed.flatten())
        self.dma_to_accelerator_memory(
            self.AE_ROPE_GQA_DRAM, packed.repeat_interleave(G, dim=0).flatten())

        self_mask, cross_mask = self._ae_attn_biases(valid)
        self.dma_to_accelerator_memory(
            self.AE_BIAS_SELF_DRAM, self_mask.to(torch.bfloat16).contiguous().flatten())
        self.dma_to_accelerator_memory(
            self.AE_BIAS_CROSS_DRAM, cross_mask.to(torch.bfloat16).contiguous().flatten())
        self._ae_constants_valid_len = valid
        print(f"  expert constants re-staged for valid_prefix_len={valid} "
              f"(rope base {valid}..{valid + M - 1}, bias cutoff col {valid})")
        return valid

        # x_t is host-written per inference, but an address DRAM never saw returns EIO on
        # dma_read (it never entered the ECC-initialized set), so a probe before the first
        # write would fail rather than read zeros. Seed it.
        self.dma_to_accelerator_memory(
            self.AE_XT_DRAM, torch.zeros(M * ADP, dtype=torch.bfloat16))
        self._ae_tensors_ready = True

    def _ae_matmul(self, ue, M, K, N, A, W, OUT, bias=None, m_reg=None, **kw):
        """One expert/head projection. BF16 B-OPERAND ONLY: no is_B_quantized, no
        data_type, no SCALE_DRAM_ADDR -- the multipliers are 16-bit and there is no
        quantized path in this file (assert_bf16_only enforces the config side).

        Always PBI (gpr_M_reg): M is 64 here, but the body is emitted 32 layers deep
        inside a rolled 10-step loop, so collapsing each matmul's M-unroll into a
        hardware row loop is the difference between a few MB of program and tens."""
        ue.matmat_mul_core(
            M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=W, OUTPUT_DRAM_ADDR=OUT,
            C_DRAM_ADDR=bias, bias_mode="broadcast_N",
            gpr_M_reg=self._ae_regs["m"] if m_reg is None else m_reg, **kw)

    def _ae_strided_copy(self, ue, src, src_jump, dst, dst_jump, rows, width):
        """Static [rows, width] strided gather -> contiguous SRAM -> strided scatter.
        Lifted verbatim from smolvlm2::compile_prefill.

        THE DEST BASE MUST ALREADY CARRY ITS PER-INDEX OFFSET (+h*D*2 on the attention
        un-stack). Dropping it does not fault and does not NaN: every head writes over
        columns [0:D], o_proj reads garbage for heads 1..14, and the result is a
        finite, plausible, wrong tensor (pi05 denoise bug #3)."""
        bpe = 2
        ue.accelerator_memory_to_sram(src, 0x00000, rows * width,
                                      stride_bytes_per_chunk=width * bpe,
                                      stride_jump_bytes=src_jump)
        ue.sram_to_accelerator_memory(0x00000, dst, rows * width,
                                      stride_bytes_per_chunk=width * bpe,
                                      stride_jump_bytes=dst_jump)

    def _ae_duplicate_gqa_rows(self, ue, rows, src_sram_addr, dst_dram_addr):
        """Token-major GQA broadcast: dst row t*G+g = src row t, for t in [0,rows).
        Lifted from smolvlm2::compile_prefill.

        This is memcpy-style PBI (a pointer whose DRAM address and URAM row advance),
        NOT PBI address injection into flash -- that distinction is the whole reason it
        may sit inside a loop that executes 10 times: Swin proved memcpy-staged PBI
        re-executes correctly where injected flash addresses corrupt on the 2nd pass."""
        D, G, bpe = self.HEAD_DIM, self.GROUP_SIZE, 2
        row_bytes = D * bpe
        row_uram_words = row_bytes // (UE_VECTOR_SIZE * bpe)
        _, src_uram_addr = ue.sram_address_to_uram_address(src_sram_addr)
        ptr = ue.alloc_inst_ptr()
        ue.generate_instruction_pbi_init(
            dram_shared_addr=dst_dram_addr, dma_length=row_bytes, output_size=0,
            uram_length=0, uram_a_start_addr=src_uram_addr,
            uram_b_start_addr=src_uram_addr, uram_wb_addr=0, uram_dst_addr=0,
            fmax_context_addr=0, inst_pointer_idx=ptr)
        ue.loop_start(loop_cnt=rows)
        ue.loop_start(G)
        # in pointer mode the "address" argument is a per-store DRAM INCREMENT, so each
        # of the G stores lands one row further on; the src URAM row only moves once per
        # token, via the pbi_inc below.
        ue.sram_to_accelerator_memory(sram_address=0, accelerator_dram_address=row_bytes,
                                      element_size=D, inst_pointer_idx=ptr,
                                      memcpy_length_bytes=0)
        ue.loop_end()
        ue.generate_instruction_pbi_inc(
            dram_shared_addr=0, dma_length=0, output_size=0, uram_length=0,
            uram_a_start_addr=row_uram_words, uram_b_start_addr=row_uram_words,
            uram_wb_addr=0, uram_dst_addr=0, fmax_context_addr=0, inst_pointer_idx=ptr)
        ue.loop_end()
        ue.release_inst_ptr(ptr)

    def _ae_dram_copy(self, ue, src, dst, elems):
        """Straight DRAM->DRAM copy through SRAM. Used only to retire the Euler update
        into x_t; an in-place eltwise (A == OUT) would alias the operand the kernel is
        still streaming, so the update lands in a separate buffer and is copied back."""
        ue.accelerator_memory_to_sram(src, 0x00000, elems)
        ue.sram_to_accelerator_memory(0x00000, dst, elems)

    def _emit_suffix_embed(self, addr_reg, step=None):
        """One Euler step's suffix embedding -- the block that sits BETWEEN the prefix and
        the expert's 32 layers, and runs once per denoise step.

            x_t [50,32] -> pad [64,64] -> action_in_proj -> [64,512]  (480 real + 0 pad)
            time row (from the table, via addr_reg)       -> [64,512] broadcast
            -> assembled [64,1024] split-column buffer
            -> action_time_mlp_in [512,1024] -> composed SiLU -> action_time_mlp_out
            -> [64,512]

        THE SPLIT-COLUMN LAYOUT IS NOT COSMETIC. action_time_mlp_in's K is
        concat(action_in_proj_out[480], time_emb[480]) = 960. Once both halves are
        512-padded the time half starts at column 512, not 480, so a K=960 matmul would
        read the wrong columns for the entire time half. Keeping the concat contiguous at
        960 is illegal -- the time half would begin at byte 960, not a multiple of 128,
        i.e. mid-SRAM-row. So the weight was K-padded 960 -> 1024 with a split permute at
        store time (see _weight_init_head) and the runtime buffer must match:
            [0:480]     action_in_proj output
            [480:512]   zero
            [512:992]   time embedding
            [992:1024]  zero

        COMPOSED SiLU, NOT FUSED. pi05 measured -6 dB on exactly this tensor using the
        LALU's fused piecewise-approximate silu_enable=True; the fix was silu_core_dram
        (true x*sigmoid(x) via identity-matmul sigmoid + elementwise multiply). We cannot
        dodge it by precomputing the way pi05 did, because this chain consumes x_t.
        """
        ue = self
        M, HP, ADP = self.SUFFIX_LEN_PAD, self.E_HIDDEN_PAD, self.ACTION_DIM_PAD
        bpe = 2
        row_in = 2 * HP * bpe                 # assembly buffer row stride: 1024 bf16
        half = HP * bpe                       # 512 bf16 = the split column offset

        # 1. THE TIMESTEP ROW FIRST. _emit_time_embed_from_table lowers to a DMA whose
        # DRAM address comes from a GPR (general_reg_src), i.e. a runtime-computed
        # address. ue_selector's runtime_addr() and append_row() share ONE scratch GPR,
        # so every op consuming a computed address must be emitted BEFORE any
        # append_row() in the same body -- this op is placed first for that reason, not
        # for dataflow reasons (nothing below depends on its ordering).
        if step is None:
            # ROLLED form: the row address must live in a GPR the body advances, since
            # there is no host in the middle of a hardware loop.
            self._emit_time_embed_from_table(addr_reg, self.AE_TIME_ROW_DRAM)
        else:
            # UNROLLED form (the default -- see DENOISE_ROLLED): address the row
            # STATICALLY, exactly as pi05 does
            #     _dram_copy(H*2, AE_COND_TABLE_DRAM + step*H*2, AE_COND_DRAM)
            # No GPR, no add_imm, no 35-bit word/byte conversion -- that pointer already
            # produced one bug here (advancing in BYTES where the PBI DRAM_ADDR field is
            # a word address, striding 8 rows per step). With the loop unrolled the
            # pointer buys nothing, so drop the whole class.
            self._probe_copy(self.AE_TIME_TABLE_DRAM + step * self.E_HIDDEN_PAD * 2,
                             self.AE_TIME_ROW_DRAM, self.E_HIDDEN_PAD)

        # 2. action_in_proj: [64,64] x [512,64]^T -> [64,512]. K=64 is the 32->64 pad
        # slot; lanes [480:512] of the output are zero because the weight's out-rows
        # 480..511 and the bias's lanes 480..511 are zero.
        self._ae_matmul(ue, M, ADP, HP, self.AE_XT_DRAM, self.action_in_weight,
                        self.AE_AIN_DRAM, bias=self.action_in_bias)

        # 3. Assemble the SPLIT-COLUMN [64,1024] operand. Two block copies, each 512
        # wide, together covering all 1024 columns:
        #     [0:480]    action_in_proj output      [480:512]  zero (its N-pad)
        #     [512:992]  the timestep embedding     [992:1024] zero (the table's pad)
        # so the zero gaps come for free from the two sources and nothing has to clear
        # them per step. The matching column scatter is baked into the STORED
        # action_time_mlp_in (see _weight_init_head) -- the two must move together.
        self._ae_strided_copy(ue, self.AE_AIN_DRAM, half,
                              self.AE_TMLP_IN_DRAM, row_in, M, HP)
        # The time row is identical for all 64 action tokens, so it is loaded to SRAM
        # once and stored 64 times. A strided scatter cannot do this: it walks the SOURCE
        # forward too, and we want the same source row every time.
        ue.accelerator_memory_to_sram(self.AE_TIME_ROW_DRAM, 0x00000, HP)
        for r in range(M):
            ue.sram_to_accelerator_memory(
                0x00000, self.AE_TMLP_IN_DRAM + r * row_in + half, HP)

        # 4. action_time_mlp_in [512,1024] -> [64,512].
        self._ae_matmul(ue, M, 2 * HP, HP, self.AE_TMLP_IN_DRAM,
                        self.time_mlp_in_weight, self.AE_TMLP_HID_DRAM,
                        bias=self.time_mlp_in_bias)

        # 5. COMPOSED SiLU, NOT the LALU's fused silu_enable. pi05 measured -6 dB on
        # exactly this tensor with the fused piecewise approximation; silu_core_dram is
        # true x*sigmoid(x) (identity-matmul sigmoid + elementwise multiply). We cannot
        # dodge it by precomputing the way pi05 did -- this chain consumes x_t, which
        # changes every Euler step. silu is elementwise, so the [64,512] tensor is fed
        # as [(64*512)/64, 64]: that keeps the sigmoid's identity operand at the stored
        # 64x64 one instead of demanding a 512x512 identity. Pad lanes survive it
        # unharmed: 0*sigmoid(0) == 0.
        silu_core_dram(ue, M=(M * HP) // UE_VECTOR_SIZE, N=UE_VECTOR_SIZE,
                       A_DRAM_ADDR=self.AE_TMLP_HID_DRAM,
                       OUTPUT_DRAM_ADDR=self.AE_TMLP_SILU_DRAM,
                       IDENTITY_DRAM_ADDR=self.identity_addr,
                       gpr_M_reg=self._ae_regs["silu"])

        # 6. action_time_mlp_out -> the expert's layer-0 input (ping-pong buffer A).
        self._ae_matmul(ue, M, HP, HP, self.AE_TMLP_SILU_DRAM,
                        self.time_mlp_out_weight, self.AE_IO_A_DRAM,
                        bias=self.time_mlp_out_bias)
        return self.AE_IO_A_DRAM

    def _emit_expert_layer(self, ue, layer_idx, step):
        """Same body shape as one compile_prefix layer at M=SUFFIX_LEN_PAD=64 on the
        512-padded stream, reusing lm_matmul/strided_copy/duplicate_gqa_rows against
        self.ae_layer_addrs.

        layer_idx % E_SELF_EVERY == 0 -> SELF attention:
            q/k/v from the expert stream, rope over the 50 action positions, CONCATENATE
            the cached prefix K/V in front of its own suffix K/V (FAULT 4; upstream
            forward_attn_layer's torch.cat), flash over the combined 192+64=256 keys
            with the [192, 256] self bias.
        otherwise                     -> CROSS attention:
            q from the expert stream only (q_proj [960,480]); K/V read from the FROZEN
            prefix cache at k_cache/v_cache(layer?) -- CONFIRM the layer mapping: the
            expert has 32 layers and so does the LM, but whether cross layer i reads LM
            layer i's KV or the final-layer KV must come from the reference, not a guess.
            the cached K/V is RE-PROJECTED through this layer's own k_proj/v_proj --
            which on cross layers are (320,320), reading the VLM kv width, not (320,480)
            -- and only then fed to flash with the [64,192] cross bias. No RoPE on that
            K (it already carries the prefix rotation); the query is roped.

        Then o_proj -> residual -> rms_norm -> gated SiLU (1280) -> residual.
        M=64 x 32 layers x 10 steps: route matmuls through gpr_M_reg PBI or the program
        bin explodes. Flash stays static. Assert lanes [480:512] stay zero.

        `step` is accepted for signature compatibility with an unrolled variant and for
        probe labelling; it MUST NOT change what is emitted. The 10 Euler steps share one
        compiled body (compile_denoise_loop rolls them into a hardware loop), so a body
        that varied with `step` could not be rolled at all."""
        assert step is None or self.N_STEPS > 0   # `step` is label-only; see docstring
        la = self.ae_layer_addrs[layer_idx]
        HP, I, Q, KV = self.E_HIDDEN_PAD, self.E_INTER, self.E_Q_OUT, self.E_KV_OUT
        D, G, M = self.HEAD_DIM, self.GROUP_SIZE, self.SUFFIX_LEN_PAD
        PM, bpe = self.PREFILL_MAX_SEQ_LEN, 2
        QB = self.AE_FLASH_ROWS                   # 64 tokens x 3 q-heads = 192
        regs = self._ae_regs
        is_self = self._ae_is_self_attn(layer_idx)

        # Ping-pong on layer parity. Layer 0 reads AE_IO_A, which is exactly where
        # _emit_suffix_embed left the embedded suffix.
        h_in = self.AE_IO_A_DRAM if layer_idx % 2 == 0 else self.AE_IO_B_DRAM
        h_out = self.AE_IO_B_DRAM if layer_idx % 2 == 0 else self.AE_IO_A_DRAM

        # ---- input RMSNorm ---------------------------------------------------------
        # rms_norm_core_dram's gpr_M_reg is M itself (unlike layer_norm_core_dram's,
        # which is the chunk COUNT M//chunk_size). The gamma carries the sqrt(480/512)
        # fold, so normalizing over 512 lanes -- 32 of them zero -- reproduces the 480
        # RMS exactly (RMSNorm has no mean term; a true LayerNorm could not be fixed
        # this way, since zero padding moves the mean data-dependently).
        ue.rms_norm_core_dram(M=M, N=HP, A_DRAM_ADDR=h_in,
                              OUTPUT_DRAM_ADDR=self.AE_PRE_NORM_DRAM,
                              GAMMA_DRAM_ADDR=la["ln1_gamma"], gpr_M_reg=regs["m"])

        # ---- Q, and (self layers only) K/V ------------------------------------------
        # Note the asymmetry that defines this model: the residual is HALF the LM's
        # width, but attention is not -- q_proj projects UP, 480 -> 960, and kv_out stays
        # 320. That is what makes the expert's attention geometry bit-identical to the
        # LM's, so the frozen prefix cache DROPS IN as a [192,64]-per-head operand --
        # but it is still re-projected first (cross k_proj is (320,320)); matching
        # geometry is not the same thing as skipping the projection.
        if self.EXPERT_BISECT and layer_idx == (self.EXPERT_LAYERS or self.E_LAYERS) - 1:
            self._probe_copy(self.AE_PRE_NORM_DRAM, self.AE_P_NORM1_DRAM, M * HP)
        self._ae_matmul(ue, M, HP, Q, self.AE_PRE_NORM_DRAM, la["q_weight"], self.AE_Q_DRAM)

        if is_self:
            self._ae_matmul(ue, M, HP, KV, self.AE_PRE_NORM_DRAM, la["k_weight"],
                            self.AE_K_PROJ_DRAM)
            self._ae_matmul(ue, M, HP, KV, self.AE_PRE_NORM_DRAM, la["v_weight"],
                            self.AE_V_PROJ_DRAM)
            # de-interleave [64,320] into 5 contiguous [64,64] head slices, then rope K
            # in place. V is never rotated.
            for h in range(self.NUM_KV_HEADS):
                k_head = self.AE_K_HEADS_DRAM + h * self.AE_KV_HEAD_STRIDE
                v_head = self.AE_V_HEADS_DRAM + h * self.AE_KV_HEAD_STRIDE
                self._ae_strided_copy(ue, self.AE_K_PROJ_DRAM + h * D * bpe, KV * bpe,
                                      k_head, D * bpe, M, D)
                self._ae_strided_copy(ue, self.AE_V_PROJ_DRAM + h * D * bpe, KV * bpe,
                                      v_head, D * bpe, M, D)
                # d64 RoPE is PBI-ONLY: bare rope_hf_core_dram falls through to the
                # legacy core, which asserts N >= 128. Always pass gpr_M_reg. Never the
                # _gqa variant either -- it asserts N >= 128 too; the group broadcast is
                # done by duplicate_gqa_rows instead.
                ue.rope_hf_core_dram(M=M, N=D, input_dram_addr=k_head,
                                     output_dram_addr=k_head,
                                     cos_dram_addr=self.AE_ROPE_PACKED_DRAM,
                                     sin_dram_addr=self.AE_ROPE_PACKED_DRAM + D * bpe,
                                     gpr_M_reg=regs["m"])
        else:
            # ---- CROSS layers: RE-PROJECT the frozen prefix K/V (FAULT 5) ------------
            # This layer's k_proj/v_proj are (320,320), NOT (320,480) like a self layer's
            # -- the checkpoint shape is the proof that the cached VLM K/V is not fed to
            # flash raw but pushed through the expert's own projections first:
            #     _k = k.reshape(*k.shape[:2], -1)                    # [S, 320]
            #     ek = exp_layer.self_attn.k_proj(_k).view(S, nkv, D)
            # (SmolVLMWithExpert.forward_cross_attn_layer; mirrored by
            # VeraPulseRef.forward_expert under self.expert_cross_reproject.)
            #
            # LAYOUT. The cache is HEAD-MAJOR: head h of prefix layer `pl` is a
            # contiguous [PM, D] block at LAYER0_K_DRAM + pl*KV_LAYER_STRIDE
            # + h*KV_HEAD_STRIDE. k_proj needs the TOKEN-MAJOR [PM, 320] view, i.e. row t
            # = concat over h of head h's row t. That is exactly one strided copy per
            # head -- contiguous gather (src jump D*bpe), strided scatter into column
            # block h (dst base + h*D*bpe, dst jump KV*bpe) -- the mirror image of the
            # self path's de-interleave a few lines up. Deliberately NOT
            # smart_bf16_permute_core: its last dim here would be D=64... and the
            # documented hazard is last_dim < 64 racing ~100k memcpys through one URAM
            # slot, so the strided-copy form is used for both directions to stay on the
            # proven path.
            pl = self._ae_cross_prefix_layer(layer_idx)
            k_base = self.LAYER0_K_DRAM + pl * self.KV_LAYER_STRIDE
            v_base = self.LAYER0_V_DRAM + pl * self.KV_LAYER_STRIDE
            for h in range(self.NUM_KV_HEADS):
                self._ae_strided_copy(ue, k_base + h * self.KV_HEAD_STRIDE, D * bpe,
                                      self.AE_XK_TOK_DRAM + h * D * bpe, KV * bpe, PM, D)
                self._ae_strided_copy(ue, v_base + h * self.KV_HEAD_STRIDE, D * bpe,
                                      self.AE_XV_TOK_DRAM + h * D * bpe, KV * bpe, PM, D)
            # [PM,320] @ (320,320).T -> [PM,320]. M is PM=192 rows here, not the suffix's
            # 64, so the PBI row-loop must run off regs["qb"] (192) -- AE_FLASH_ROWS ==
            # PM is asserted in _ae_tensor_init, which is what makes that reuse legal.
            self._ae_matmul(ue, PM, KV, KV, self.AE_XK_TOK_DRAM, la["k_weight"],
                            self.AE_XK_PROJ_DRAM, m_reg=regs["qb"])
            self._ae_matmul(ue, PM, KV, KV, self.AE_XV_TOK_DRAM, la["v_weight"],
                            self.AE_XV_PROJ_DRAM, m_reg=regs["qb"])
            # back to head-major [5, PM, D] so flash can take a contiguous K/V per group.
            for h in range(self.NUM_KV_HEADS):
                self._ae_strided_copy(ue, self.AE_XK_PROJ_DRAM + h * D * bpe, KV * bpe,
                                      self.AE_XK_HEADS_DRAM + h * self.AE_XKV_HEAD_STRIDE,
                                      D * bpe, PM, D)
                self._ae_strided_copy(ue, self.AE_XV_PROJ_DRAM + h * D * bpe, KV * bpe,
                                      self.AE_XV_HEADS_DRAM + h * self.AE_XKV_HEAD_STRIDE,
                                      D * bpe, PM, D)
            # NO RoPE on the reprojected K: the cached K was already rotated with the
            # PREFIX positions during prefill, and upstream does not rotate `ek`. Only
            # the expert's query is roped on cross layers (below), with positions
            # rebased to 0..chunk-1 (exp_pos - exp_pos.min()).

        # ---- per-kv-group stacked-Q flash -------------------------------------------
        # SELF layers now attend over the COMBINED [prefix ; suffix] key/value sequence
        # (FAULT 4), so their aligned_seq_len is PM + M = 256, not QB. batch stays QB.
        CW = self.AE_COMBINED_LEN
        seq_len = CW if is_self else PM
        seq_reg = regs["cw"] if is_self else regs["qb"]
        bias_addr = self.AE_BIAS_SELF_DRAM if is_self else self.AE_BIAS_CROSS_DRAM
        # Which prefix layer's cache a SELF layer concatenates: the reference uses
        # prefix_kv[i] on BOTH branches (VeraPulseRef.forward_expert), so reuse the same
        # mapping helper the cross path uses rather than hardcoding the index.
        self_pl = self._ae_cross_prefix_layer(layer_idx) if is_self else None
        for kv_b in range(self.NUM_KV_HEADS):
            # gather this group's 3 q-heads TOKEN-MAJOR: flash row t*G+g is q-head
            # kv_b*G+g of token t, so one flash call serves the whole group.
            for g in range(G):
                self._ae_strided_copy(ue, self.AE_Q_DRAM + (kv_b * G + g) * D * bpe,
                                      Q * bpe, self.AE_FLASH_Q_DRAM + g * D * bpe,
                                      G * D * bpe, M, D)
            if is_self or self.EXPERT_ROPE_Q_ON_CROSS:
                # rope the stacked buffer in one call at M=QB against the x G duplicated
                # table (row r uses token r//G).
                #
                # FAULT 7: WHICH table depends on the branch.
                #   SELF  -> AE_ROPE_GQA_DRAM, positions valid_prefix_len.. , matching
                #            the cached prefix K it now concatenates with (fault 4).
                #   CROSS -> AE_ROPE_GQA_CROSS_DRAM, positions REBASED to 0..chunk-1.
                #            forward_cross_attn_layer does
                #                exp_pos = exp_pos - exp_pos.min(dim=1, keepdim=True)
                #                eq = apply_rope(eq, exp_pos)
                #            and leaves the reprojected key unroped, so the query must be
                #            rotated from 0 here. Using the prefix-continued table (what
                #            this emitter did before) put every cross query at a phase
                #            offset of valid_prefix_len against a key that carries the
                #            prefix rotation -- a silent, smooth, entirely wrong result.
                rope_tbl = self.AE_ROPE_GQA_DRAM if is_self \
                    else self.AE_ROPE_GQA_CROSS_DRAM
                ue.rope_hf_core_dram(M=QB, N=D, input_dram_addr=self.AE_FLASH_Q_DRAM,
                                     output_dram_addr=self.AE_FLASH_Q_DRAM,
                                     cos_dram_addr=rope_tbl,
                                     sin_dram_addr=rope_tbl + D * bpe,
                                     gpr_M_reg=regs["qb"])

            if is_self:
                # ---- FAULT 4: build [prefix ; suffix] K/V for THIS kv-head -----------
                # forward_attn_layer (use_cache=True, fill_kv_cache=False):
                #     k = torch.cat([kv[idx]["key_states"], k], dim=1)
                #     v = torch.cat([kv[idx]["value_states"], v], dim=1)
                # Rows [0:PM) are the RAW cached prefix K/V for this head -- raw, not
                # reprojected: the reprojection is a CROSS-layer behaviour (odd-layer
                # k_proj is (320,320)); a self layer's own k_proj already produced its
                # suffix K from the expert stream. The cache is already head-major
                # [PM, D] per head, so this is a straight contiguous block copy, and the
                # cache itself is only READ (it must survive all 10 Euler steps).
                # Rows [PM:PM+M) are this layer's own roped suffix K/V, likewise a
                # contiguous [M, D] block. Deliberately _ae_strided_copy (contiguous
                # jumps) and NOT smart_bf16_permute_core -- last_dim D=64 sits on the
                # documented sub-64/URAM-slot hazard boundary, and the strided-copy form
                # is the proven path used everywhere else in this emitter.
                #
                # RoPE consistency: the cached K was rotated with PREFIX positions at
                # prefill and the suffix K with SUFFIX positions here, and the suffix
                # positions continue from the prefix -- which is exactly why upstream can
                # concatenate the two without re-rotating either.
                #
                # GQA: no x G replication any more. The old duplicate_gqa_rows pair only
                # existed to lift 64 kv rows to the 192-row batch floor
                # (batch <= aligned_seq_len); at 256 combined rows the constraint is met
                # outright. Correctness is unaffected -- all G q-heads of this group read
                # the SAME K/V head, which is what GQA means, and the stacked-Q batch
                # dimension is independent of the key length.
                k_pre = (self.LAYER0_K_DRAM + self_pl * self.KV_LAYER_STRIDE
                         + kv_b * self.KV_HEAD_STRIDE)
                v_pre = (self.LAYER0_V_DRAM + self_pl * self.KV_LAYER_STRIDE
                         + kv_b * self.KV_HEAD_STRIDE)
                k_suf = self.AE_K_HEADS_DRAM + kv_b * self.AE_KV_HEAD_STRIDE
                v_suf = self.AE_V_HEADS_DRAM + kv_b * self.AE_KV_HEAD_STRIDE
                self._ae_strided_copy(ue, k_pre, D * bpe, self.AE_CK_DRAM, D * bpe, PM, D)
                self._ae_strided_copy(ue, v_pre, D * bpe, self.AE_CV_DRAM, D * bpe, PM, D)
                self._ae_strided_copy(ue, k_suf, D * bpe,
                                      self.AE_CK_DRAM + PM * D * bpe, D * bpe, M, D)
                self._ae_strided_copy(ue, v_suf, D * bpe,
                                      self.AE_CV_DRAM + PM * D * bpe, D * bpe, M, D)
                k_addr, v_addr = self.AE_CK_DRAM, self.AE_CV_DRAM
            else:
                # THE REPROJECTED PREFIX K/V (FAULT 5), not the raw cache. Built once
                # above for all 5 groups as contiguous [PM=192, 64] blocks -- the same
                # geometry the raw cache had, so nothing downstream changes. Still no
                # x G replication: 192 == the stacked-Q batch. The raw cache itself is
                # only READ here and must survive all 10 steps untouched.
                off = kv_b * self.AE_XKV_HEAD_STRIDE
                k_addr = self.AE_XK_HEADS_DRAM + off
                v_addr = self.AE_XV_HEADS_DRAM + off

            # FLASH STAYS ADDRESS-STATIC. Only the two dimension GPRs are runtime (the
            # same thing compile_encoder does); PBI address injection into flash corrupts
            # on the 2nd execution and this body runs 10 times per inference.
            ue.unified_attention_core(
                batch=QB, aligned_seq_len=seq_len, head_dim=D,
                Q_DRAM_ADDR=self.AE_FLASH_Q_DRAM, K_DRAM_ADDR=k_addr, V_DRAM_ADDR=v_addr,
                BIAS_DRAM_ADDR=bias_addr, OUTPUT_DRAM_ADDR=self.AE_FLASH_OUT_DRAM,
                SCRATCH_DRAM_ADDR=self.AE_FLASH_SCRATCH_DRAM,
                IDENTITY_DRAM_ADDR=self.identity_addr,
                gpr_batch_reg=regs["qb"], gpr_aligned_seq_len_reg=seq_reg)

            # un-stack [64,G,D] -> [64,960] at THIS group's head columns. The + head*D*2
            # in the destination base is trap #8: without it every head writes columns
            # [0:64], o_proj reads garbage for 14 of the 15 heads, and the output is
            # finite and scrambled rather than NaN.
            for g in range(G):
                self._ae_strided_copy(ue, self.AE_FLASH_OUT_DRAM + g * D * bpe,
                                      G * D * bpe,
                                      self.AE_ATTN_RESULT_DRAM + (kv_b * G + g) * D * bpe,
                                      Q * bpe, M, D)

        # ---- o_proj + residual ------------------------------------------------------
        # o_weight is [512,960]: the N-pad puts zeros in out-rows 480..511, which is what
        # re-zeroes the residual stream's pad lanes after every attention write.
        self._ae_matmul(ue, M, Q, HP, self.AE_ATTN_RESULT_DRAM, la["o_weight"],
                        self.AE_O_PROJ_DRAM)
        ue.eltwise_core_dram(M=M, N=HP, dram_a=h_in, dram_b=self.AE_O_PROJ_DRAM,
                             dram_out=self.AE_RESIDUAL_DRAM, mode=UE_MODE.ELTWISE_ADD,
                             gpr_M_reg=regs["m"])
        # Split residual-then-norm rather than the fused post-add norm: the fused
        # layer_norm_core_dram_post_add has NO PBI (4 advancing pointers vs the <=3
        # limit), so it would unroll M statically 32 layers deep. Both halves here are
        # PBI.
        ue.rms_norm_core_dram(M=M, N=HP, A_DRAM_ADDR=self.AE_RESIDUAL_DRAM,
                              OUTPUT_DRAM_ADDR=self.AE_PRE_NORM_DRAM,
                              GAMMA_DRAM_ADDR=la["ln2_gamma"], gpr_M_reg=regs["m"])

        # ---- gated MLP --------------------------------------------------------------
        # The FUSED silu_enable is fine HERE and is what smolvlm2 ships; only the time
        # MLP needs the composed form (pi05 measured -6 dB there specifically).
        if self.EXPERT_BISECT and layer_idx == (self.EXPERT_LAYERS or self.E_LAYERS) - 1:
            self._probe_copy(self.AE_PRE_NORM_DRAM, self.AE_P_NORM2_DRAM, M * HP)
        self._ae_matmul(ue, M, HP, I, self.AE_PRE_NORM_DRAM, la["gate_weight"],
                        self.AE_MLP_GATE_DRAM, silu_enable=True)
        self._ae_matmul(ue, M, HP, I, self.AE_PRE_NORM_DRAM, la["up_weight"],
                        self.AE_MLP_UP_DRAM)
        eltwise_mul_core_dram(ue, size=M * I, A_DRAM_ADDR=self.AE_MLP_GATE_DRAM,
                              B_DRAM_ADDR=self.AE_MLP_UP_DRAM,
                              OUTPUT_DRAM_ADDR=self.AE_MLP_MULT_DRAM)
        self._ae_matmul(ue, M, I, HP, self.AE_MLP_MULT_DRAM, la["down_weight"],
                        self.AE_MLP_DOWN_DRAM)
        ue.eltwise_core_dram(M=M, N=HP, dram_a=self.AE_RESIDUAL_DRAM,
                             dram_b=self.AE_MLP_DOWN_DRAM, dram_out=h_out,
                             mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=regs["m"])
        return h_out

    def compile_denoise_loop(self):
        """10 Euler steps, dt = -1/10, all timesteps known at compile time:
             for step in range(10):
                 v = expert(x_t, t_step, prefix_kv)      # 32 layers, then ae.final_norm
                 x_t = x_t + dt * action_out_proj(v)

        Compile ONE body, not ten. The schedule being static does not make unrolling
        free -- a Python for-loop here is a 10x program multiplier on top of 32 layers.
        Roll it with gpr_M_reg plus an advancing pointer into the [10,512] timestep
        table, the same way the prefix collapses its M-unroll."""
        ue = self
        M, HP, ADP = self.SUFFIX_LEN_PAD, self.E_HIDDEN_PAD, self.ACTION_DIM_PAD

        self._ae_tensor_init()
        self._precompute_time_embeddings()

        # THE TIMESTEP POINTER IS A **WORD** ADDRESS. accelerator_memory_to_sram's
        # general_reg_src path lowers to PBI_MODE_REG on PBI_FIELD.DRAM_ADDR, and that
        # descriptor field is the 35-bit word address (ue_35bit_addr_shifter, byte >> 3),
        # not a byte address. So both the seed and the per-step advance must be in words.

        ue.start_capture()
        prog_addr = ue.get_program_dram_addr()
        # Start from a clean register file: the body is 32 layers deep and every core
        # allocates and releases its own scratch, so the four registers held across the
        # whole body must sit at the bottom (gpr_M_reg is restricted to 1..15).
        # BOTH counters, not just the reg one. alloc_isa_reg/alloc_inst_ptr are
        # per-PROCESS and start_capture() does NOT reset them, so vision (12 layers) and
        # prefix (32 layers) drift them upward before denoise compiles. pi05 documents
        # the consequence: denoise's regs and PBI pointers land on different physical
        # indices in a full run than in a standalone one -> reserved/stale regs -> 100%
        # NaN, not a crash. mobilesam resets both at the start of every stage.
        ue.reset_isa_reg_counter()
        ue.reset_inst_ptr_counter()
        regs = {"m": ue.alloc_isa_reg(),       # 64  -- rows of the suffix stream
                "qb": ue.alloc_isa_reg(),      # 192 -- stacked-Q flash rows / kv length
                "cw": ue.alloc_isa_reg(),      # 256 -- combined prefix+suffix kv length
                                               #        on SELF layers (FAULT 4)
                "silu": ue.alloc_isa_reg(),    # 512 -- the elementwise-reshaped SiLU
                "time": ue.alloc_isa_reg()}    # advancing timestep-table row pointer
        ue.generate_instruction_add_set(regs["m"], M)
        ue.generate_instruction_add_set(regs["qb"], self.AE_FLASH_ROWS)
        ue.generate_instruction_add_set(regs["cw"], self.AE_COMBINED_LEN)
        ue.generate_instruction_add_set(regs["silu"], (M * HP) // UE_VECTOR_SIZE)
        ue.generate_instruction_add_set(
            regs["time"], ue_35bit_addr_shifter(self.AE_TIME_TABLE_DRAM))
        self._ae_regs = regs

        # ONE BODY, TEN EXECUTIONS. A Python for-loop here would be a 10x multiplier on
        # top of 32 layers x 5 flash calls; the schedule being known at compile time does
        # not make unrolling free. The only per-step input is the timestep row, and it is
        # reached through the GPR pointer the body advances itself -- which is exactly
        # why _precompute_time_embeddings builds a TABLE instead of the host pushing an
        # embedding per step (there is no host in the middle of a hardware loop).
        # The seeds above are inside the captured program, so every re-execution of the
        # whole program restarts the schedule at t = 1.0.
        def _emit_one_euler_step(step=None):
            """One Euler step. Identical bytes every time it is emitted -- the only
            per-step input is the timestep row, reached through the GPR pointer
            _emit_suffix_embed advances itself. That is what makes the rolled and
            unrolled forms numerically identical."""
            self._emit_suffix_embed(regs["time"], step=step)   # -> AE_IO_A
            n_ae = self.E_LAYERS if self.EXPERT_LAYERS is None else int(self.EXPERT_LAYERS)
            if self.EXPERT_BISECT:
                self._probe_copy(self.AE_IO_A_DRAM, self.AE_P_EMBED_DRAM, M * HP)
            for layer_idx in range(n_ae):
                self._emit_expert_layer(ue, layer_idx, None)
            # 32 layers of ping-pong end in A for an even layer count; derive it rather
            # than assume it, so a tiny/partial expert still reads the right buffer.
            final_h = (self.AE_IO_A_DRAM if n_ae % 2 == 0
                       else self.AE_IO_B_DRAM)
            ue.rms_norm_core_dram(M=M, N=HP, A_DRAM_ADDR=final_h,
                                  OUTPUT_DRAM_ADDR=self.AE_FINAL_DRAM,
                                  GAMMA_DRAM_ADDR=self.ae_final_norm_addr,
                                  gpr_M_reg=regs["m"])
            # Euler update. dt is FOLDED INTO THE STORED action_out_proj weight and bias
            # (see _weight_init_head), so this matmul already produces dt*v and the
            # update is a plain add -- no on-device scalar multiply, no extra pass.
            self._ae_matmul(ue, M, HP, ADP, self.AE_FINAL_DRAM,
                            self.action_out_weight_dt, self.AE_VT_DRAM,
                            bias=self.action_out_bias_dt)
            eltwise_add_core_dram(ue, size=M * ADP, A_DRAM_ADDR=self.AE_XT_DRAM,
                                  B_DRAM_ADDR=self.AE_VT_DRAM,
                                  OUTPUT_DRAM_ADDR=self.AE_XT_NEXT_DRAM)
            # retire through a copy rather than an in-place eltwise: A == OUT would have
            # the kernel writing back over an operand it is still streaming.
            self._ae_dram_copy(ue, self.AE_XT_NEXT_DRAM, self.AE_XT_DRAM, M * ADP)
            # Snapshot x_t after this step (pi05's AE_STEP_SNAPSHOT_DRAM). Gives a
            # per-step divergence curve -- the Euler-step analogue of the per-layer KV
            # curve that separated "one broken layer" from "uniform drift" in the prefix.
            # Cheap: 10 x [64,64] = 80 KB total.
            if step is not None:
                self._ae_dram_copy(ue, self.AE_XT_DRAM,
                                   self.AE_STEP_SNAP_DRAM + step * M * ADP * 2, M * ADP)

        if self.DENOISE_ROLLED:
            # One body inside a hardware loop. SMALL but UNPROVEN: _ae_duplicate_gqa_rows
            # already opens two nested hardware loops, so this makes it depth 3. The
            # prefix runs the same helper at depth 2 and completes; the rolled denoise
            # hangs. pi05 reverted this exact arrangement for the same reason.
            ue.loop_start(loop_cnt=self.N_STEPS)
            _emit_one_euler_step()      # rolled: GPR-addressed row, no snapshots
            ue.loop_end()
        else:
            # STATIC UNROLL (default): 10 copies of the body, max loop depth 2 -- the
            # depth the prefix has already proven on this device. ~10x the program
            # (3.1 -> ~31 MB against a 2.85 GB arena), which is the price of a stage
            # that terminates. Same instruction sequence per copy, and the timestep
            # pointer advances inside each one, so the schedule still walks t = 1.0 ->
            # 0.1 exactly as the rolled form would.
            for _step in range(self.N_STEPS):
                _emit_one_euler_step(step=_step)

        ue.generate_instruction_halt()
        ue.stop_capture()

        raw = bytearray()
        for inst in ue.capture_buffer:
            raw.extend(inst.get_bytes())
        ue.dma_write(DMA_DEVICE_H2C, prog_addr, raw, len(raw))
        ue.allocate_program_dram(len(raw))
        ue.clear_capture_buffer()
        self._denoise_program_addr = prog_addr
        n_self = sum(1 for i in range(self.E_LAYERS) if self._ae_is_self_attn(i))
        print(f"    denoise loop ({self.N_STEPS} steps x {self.E_LAYERS} layers, "
              f"{n_self} self / {self.E_LAYERS - n_self} cross): "
              f"{len(raw)} bytes @0x{prog_addr:X}")
        return prog_addr

    # ==================================================================================
    # 6. run
    # ==================================================================================

    def _read_bf16(self, addr, shape, label=""):
        """Chunked + retried + CHECKED bf16 readback -> float32 tensor.

        raw dma_read() returns -1 on failure after merely PRINTING the errno, and the
        caller's freshly-allocated buffer is zero-filled -- so a failed read is
        indistinguishable from "the hardware computed zeros" unless the return code is
        checked. pi05 shipped an 18 MB .npz of pure zeros this way and spent hours
        blaming the model. Chunk to stay inside the XDMA C2H windows that actually
        work, and RAISE rather than hand back a plausible-looking zero buffer."""
        numel = 1
        for d in shape:
            numel *= d
        size = numel * 2
        buf = bytearray(size)
        mv = memoryview(buf)
        chunk_bytes, offset = 256 * 1024, 0
        tag = f" {label}" if label else ""
        while offset < size:
            n = min(chunk_bytes, size - offset)
            piece = bytearray(n)
            for _ in range(5):
                if self.dma_read(DMA_DEVICE_C2H, addr + offset, piece, n) == n:
                    break
            else:
                if chunk_bytes > 4096:
                    chunk_bytes //= 2      # bisect around the driver's bad windows
                    continue
                raise RuntimeError(
                    f"dma_read{tag} FAILED at 0x{addr + offset:X} "
                    f"(+{offset}/{size} B) even at the 4KB floor -- refusing to return "
                    f"a zero-filled buffer that would masquerade as real data")
            mv[offset:offset + n] = piece
            offset += n
        return torch.frombuffer(bytes(buf), dtype=torch.bfloat16).reshape(*shape).float()

    def run_vision(self, images_hwc):
        """[2,512,512,3] -> patchify [1024,768] per slot -> encoder -> connector ->
        [128,960] vision tokens.

        The encoder program is the expensive artifact, so it is compiled ONCE and
        executed once per camera slot. BOTH slots are always encoded -- a zero/blank
        camera is masked in the attention bias downstream, never skipped (skipping is a
        hard no in this project, and the patch-embed bias makes even an all-zero image
        produce non-zero rows, so there is nothing degenerate to shortcut anyway).

        STAGING LAYOUT: compile_encoder's first op is
        smart_bf16_permute_core(dims=[CH,NPS,P,NPS,P], perm=[1,3,0,2,4]), i.e. it reads
        a plain CHANNEL-PLANAR [3,512,512] image viewed as (c, patch_row, py, patch_col,
        px) and emits (patch_row, patch_col, c, py, px) -- rows = the 1024 patches,
        columns = the channel-major c*P*P + kh*P + kw flatten that patch_embed.weight
        was reshaped into. So the host stages [3,512,512], NOT [1024,768] patches: the
        patchify is the device's job here (unlike pi05, where the host patchifies).

        Intermediates for the SNR/cos gate are stashed on self._last_vision:
            post_ln[slot]   [1024,768]  ViT output after vis.post_ln
            connector[slot] [  64,960]  after pixel-shuffle + conn.proj
            tokens          [ 128,960]  the concatenation this returns
        """
        V, C = self._cfg["vision"], self._cfg["connector"]
        S, H = V["num_patches"], V["hidden_size"]
        CH, IMG = V["num_channels"], V["image_size"]
        slots, tok_out = V["num_image_slots"], C["tokens_out"]

        images = torch.as_tensor(images_hwc)
        assert images.shape == (slots, IMG, IMG, CH), (
            f"run_vision expects [{slots},{IMG},{IMG},{CH}] HWC images, "
            f"got {tuple(images.shape)}")

        # compile-once: the program is structure-bound (PBI on M), identical for every
        # slot, and recompiling per slot would also re-allocate program DRAM per slot.
        prog = getattr(self, "_vis_program_addr", None)
        if prog is None:
            prog = self.compile_encoder()

        # layer_norm_core_dram WRITES its zeros scratch, so the previous slot's execution
        # dirtied it. Restore before every run or slot 1 layer-norms against garbage
        # (pi05's per-slot refresh, same root cause).
        zeros_t = torch.zeros(H, dtype=torch.bfloat16)

        post_ln, conn = [], []
        for i in range(slots):
            # HOST PATCHIFY: [512,512,3] HWC -> [3,512,512] planar -> [1024, 768] with
            # columns in the Conv2d weight's channel-major (c, kh, kw) order. The device
            # used to do this permute, but its last dim (P=16) is under UE_VECTOR_SIZE
            # and nn_lib's unaligned branch races through a single URAM slot -- see
            # compile_encoder. _host_patchify is bit-exact vs conv2d unfold.
            planes = images[i].permute(2, 0, 1).contiguous().float()   # [3,512,512]
            patches = self._host_patchify(planes)                      # [1024, 768]
            self.dma_to_accelerator_memory(
                self.VIS_PIXEL_IN_DRAM,
                patches.reshape(-1).to(torch.bfloat16).contiguous())
            self.dma_to_accelerator_memory(self.vis_zeros_addr, zeros_t)

            self.program_execute(prog, timeout=300.0)

            post_ln.append(self._read_bf16(self.VIS_POST_LN_DRAM, (S, H),
                                           label=f"vis_post_ln[{i}]"))
            conn.append(self._read_bf16(self.VIS_CONNECTOR_DRAM, (tok_out, C["output_size"]),
                                        label=f"vis_connector[{i}]"))
            # The connector output must be COPIED OUT here: VIS_CONNECTOR_DRAM is a
            # single fixed buffer and the next slot's execution overwrites it. That copy
            # is what the readback above already is.
            assert torch.isfinite(conn[-1]).all(), f"vision slot {i} produced non-finite tokens"

        tokens = torch.cat(conn, 0)                                   # [slots*64, 960]
        self._last_vision = {"post_ln": post_ln, "connector": conn,
                             "tokens": tokens, "images": images}
        print(f"  vision: {slots} slots -> tokens {tuple(tokens.shape)} "
              f"absmax={tokens.abs().max():.4f}")
        return tokens

    def _execute_prefix(self, timeout=300.0):
        """EXECUTE the compiled prefix program. Mirrors _execute_denoise: the program is
        address-static, so re-execution just re-reads LM_INPUT_DRAM and PREFIX_BIAS_DRAM
        (both refreshed by run_prefix) and re-writes the KV cache in place."""
        with PHASES.track("exec prefix", "exec"), silenced():
            self.start_execute_from_dram(self._prefix_program_addr)
            self._wait_with_heartbeat("prefix", timeout=timeout)
        _original_print(f"  [prefix] executed in {PHASES.rows[-1][2]:.2f}s")

    def run_prefix(self, vision_tokens, token_ids, state, text_mask=None, timeout=300.0):
        """Run the 32-layer prefix over the assembled observation. Returns the final
        hidden [PM, 960]; the host needs nothing else back except probes.

        THE REAL PRODUCT IS THE KV CACHE, not the returned hidden state. Each layer
        writes its [PM, 320] K and V into LAYER0_K/V_DRAM, and that cache is the action
        expert's ONLY prefix input: every cross-attention layer reads it on every one of
        the N denoise steps. It must therefore survive the WHOLE denoise loop -- nothing
        may allocate over, stage through, or otherwise reuse LAYER0_K/V_DRAM between
        run_prefix and the last Euler step. tensor_init claims that region first for
        exactly this reason. Re-running run_prefix (a new observation) is the only thing
        allowed to overwrite it.

        Order of operations, and why:
          1. _lm_tensor_init  -- LM_INPUT_DRAM must exist before we DMA into it, and
             compile_prefix's allocator must not be the first thing to touch it.
          2. embed_and_concat_prefix -> the [PM,960] input + the data-dependent valid_len.
          3. build_attn_bias(valid_len) -- DATA ONLY. The mask changes per prompt (the
             real text length varies) but the program does not, so this must never
             trigger a recompile.
          4. _compile_once("prefix", ...) -- compile-once/execute-many. Calling
             compile_prefix directly would advance the program-DRAM allocator once per
             inference and kill a LIBERO episode after a few steps (pi05 died at 3).
        """
        self._lm_tensor_init()
        x, valid_len = self.embed_and_concat_prefix(
            token_ids, vision_tokens, state, text_mask=text_mask)
        self.build_attn_bias(valid_len)

        self._compile_once("prefix", self.compile_prefix, label="prefix")
        self._execute_prefix(timeout=timeout)

        hidden = self._read_bf16(self.PREFIX_HIDDEN_DRAM,
                                 (self.PREFILL_MAX_SEQ_LEN, self.HIDDEN_SIZE),
                                 label="prefix_hidden")
        # finite is not correctness, but non-finite IS a hard stop: a NaN here means the
        # KV cache is poisoned too, and every denoise step downstream is meaningless.
        assert torch.isfinite(hidden).all(), (
            "prefix produced non-finite hidden -- check the pad-row epsilon in "
            "embed_and_concat_prefix and that the bias's pad block keeps its diagonal "
            "(an all -inf row is 0/0 in softmax)")
        self._last_prefix = {"hidden": hidden, "valid_len": valid_len, "prefix_in": x}
        print(f"  prefix: {self.NUM_LAYERS} layers, valid {valid_len}/"
              f"{self.PREFILL_MAX_SEQ_LEN} rows -> hidden {tuple(hidden.shape)} "
              f"absmax={hidden[:valid_len].abs().max():.4f} (KV cache written)")
        return hidden

    def _wait_with_heartbeat(self, label, timeout=180.0, heartbeat_every=1.0):
        """wait_queue wrapper with a liveness heartbeat and a REAL timeout check.

        Ported from pi05_libero_test.py. Two reasons it exists, both learned the hard way
        there:

        1. user_dma_core.wait_queue() does NOT raise on timeout -- it prints
           "Error: wait_queue() timed out..." and returns normally, so the caller sails
           on as if the FPGA finished. We re-check is_queue_busy() ourselves afterwards
           and raise for real. Without this, a hung stage silently corrupts every
           downstream stage; pi05's prefix may never have actually completed before the
           denoise loop was compiled and run on top of it.

        2. The heartbeat makes a genuine hang (dead silent) distinguishable from a slow
           but progressing execute. Our vision encoder alone runs ~5.8 s per slot, and
           the rolled 10-step denoise loop is the one arrangement pi05 reverted after it
           hung -- so telling "slow" from "wedged" without guessing is worth a thread.

        Prints via _original_print so the heartbeat survives silenced()."""
        import threading
        t0 = time.perf_counter()
        stop = threading.Event()

        def _hb():
            while not stop.wait(heartbeat_every):
                _original_print(f"  [{label}] running... "
                                f"{time.perf_counter() - t0:.0f}s", end="\r", flush=True)

        th = threading.Thread(target=_hb, daemon=True)
        th.start()
        try:
            self.wait_queue(timeout)
        finally:
            stop.set()
            th.join(timeout=1.0)
        if self.is_queue_busy():
            _original_print()
            raise RuntimeError(
                f"[{label}] FPGA queue STILL BUSY after {timeout:.0f}s -- a real hang, "
                f"not a silent success. wait_queue() does not raise on its own; this "
                f"check exists precisely to catch that. Do not trust any downstream "
                f"result from this run.")
        _original_print(f"  [{label}] done in {time.perf_counter() - t0:.1f}s" + " " * 20)

    def _compile_once(self, key, compile_fn, label="compile_once"):
        """COMPILE-ONCE half of compile-once/execute-many (pi05's _compile_once).

        Captures, serializes, DMAs and advances the program-DRAM allocator EXACTLY the
        first time a key is seen; every later call returns the cached address with NO
        capture and NO allocation. That "no allocation" is the load-bearing part: without
        it the program pointer marches toward the 4 GB ceiling once per inference and the
        run dies after a few calls -- the bug that killed pi05 at 3 inferences before the
        compile-once fix took it to 252.

        Does NOT execute; the caller runs _execute_denoise."""
        cache = self.__dict__.setdefault("_prog_cache", {})
        meta = self.__dict__.setdefault("_prog_meta", {})
        if key in cache:
            return cache[key]
        _original_print(f"  [{label}] compiling ONCE...", end="\r", flush=True)
        before = self.get_program_dram_addr()
        with PHASES.track(f"compile {label}", "compile"), silenced():
            addr = compile_fn()
        secs = PHASES.rows[-1][2]
        size = self.get_program_dram_addr() - before
        cache[key], meta[key] = addr, (addr, size)
        _original_print(f"  [{label}] compiled in {secs:.2f}s  {size / 1e6:.2f} MB "
                        f"@0x{addr:X}" + " " * 16)
        return addr

    def _execute_denoise(self, timeout=300.0):
        """EXECUTE the compiled denoise program. Re-execution is deterministic: the
        prefix K/V it cross-attends into sits at fixed LAYER0_K/V_DRAM (refreshed by the
        prefix stage), the timestep pointer is re-seeded inside the captured program so
        every run restarts at t=1.0, and fresh noise is DMA'd to AE_XT_DRAM by the
        caller before this is called."""
        with PHASES.track("exec denoise", "exec"), silenced():
            self.start_execute_from_dram(self._denoise_program_addr)
            self._wait_with_heartbeat("denoise", timeout=timeout)
        _original_print(f"  [denoise] executed in {PHASES.rows[-1][2]:.2f}s")

    def run_denoise(self, noise=None, timeout=300.0):
        """Run the 10-step Euler loop on hardware. Returns actions [chunk, action_dim]
        (normalized -- the caller denormalizes with norm_stats).

        Structure mirrors pi05_libero_test.py's denoise stage: seed x_t, _compile_once,
        _execute_denoise, read back, slice.

        PAD ROWS MUST NOT BE EXACTLY ZERO. rms_norm_core has no epsilon, so an all-zero
        row gives rsqrt(0) = inf -> NaN, and self-attention over the 64-row suffix then
        mixes that NaN into every real row. pi05 hit this exact bug class twice (here and
        in embed_and_concat_prefix). So the padding is 1e-6, not 0. Note this is padding
        in the ROW/COLUMN sense (rows 50..63, columns 7..31) -- it is not skipping
        compute, the hardware still processes all 64x32.
        """
        # ORDER: the x_t buffer is allocated by _ae_tensor_init, which compile_denoise_loop
        # also calls -- but we DMA the noise BEFORE compiling, so claim the buffers first.
        # _ae_tensor_init is idempotent (_ae_tensors_ready), so the later call inside
        # compile_denoise_loop is a no-op rather than a double allocation.
        self._ae_tensor_init()

        # FAULT 8: the expert's RoPE base and both attention biases depend on the VALID
        # prefix length, which is prompt-dependent and only known after run_prefix. They
        # are DRAM data, not program, so re-staging them here costs four DMAs and NO
        # recompile -- the compiled denoise program reads whatever those addresses hold.
        # Must happen before _execute_denoise; doing it before _compile_once as well is
        # harmless and keeps any compile-time probe reading consistent constants.
        self._ae_refresh_runtime_constants()

        M, ADP = self.SUFFIX_LEN_PAD, self.ACTION_DIM_PAD
        chunk, adim = self.CHUNK, self.ACTION_DIM

        if noise is None:
            x0 = torch.full((M, ADP), 1e-6, dtype=torch.float32)
            g = torch.Generator().manual_seed(0)
            x0[:chunk, :adim] = torch.randn(chunk, adim, generator=g)
        else:
            noise = torch.as_tensor(noise, dtype=torch.float32)
            assert noise.shape[0] >= chunk and noise.shape[1] >= adim, (
                f"noise must cover at least [{chunk},{adim}], got {tuple(noise.shape)}")
            x0 = torch.full((M, ADP), 1e-6, dtype=torch.float32)
            x0[:chunk, :adim] = noise[:chunk, :adim]
        assert (x0 != 0).all(), "zero in x_t would NaN through the epsilon-free rms_norm"
        self.dma_to_accelerator_memory(
            self.AE_XT_DRAM, x0.reshape(-1).to(torch.bfloat16).contiguous())

        self._compile_once("denoise", self.compile_denoise_loop, label="denoise")
        self._execute_denoise(timeout=timeout)

        out = self._read_bf16(self.AE_XT_DRAM, (M, ADP), label="ae_x_t")
        assert torch.isfinite(out).all(), (
            "denoise produced non-finite x_t -- check the pad-row epsilon, the expert "
            "pad lanes [480:512], and whether the rolled loop actually repeated")
        self._last_denoise = {"x_t_padded": out, "noise": x0}
        actions = out[:chunk, :adim]
        print(f"  denoise: {self.N_STEPS} steps x {self.E_LAYERS} layers -> "
              f"actions {tuple(actions.shape)} absmax={actions.abs().max():.4f}")
        return actions

    def run_inference(self, images_hwc, token_ids, state, noise=None, snr=True,
                      stop_after=None, strict_gates=False):
        """vision -> prefix -> denoise on hardware. Returns the EXECUTED action slice
        [n_action_steps, action_dim] = [10, 7].

        With snr=True, gate each stage against the torch oracle and STOP AT THE FIRST
        stage below 40 dB. Failing early is the whole point: once a stage is wrong,
        everything downstream of it is scoring a different model, so a single end-to-end
        number cannot tell "the ViT drifted" from "the pixel shuffle is scrambled" from
        "the expert reads the wrong KV layer". The stages that HAVE gates run them; the
        ones whose gate is still a stub are named explicitly as UNGATED so nobody reads a
        silent pass as a verified one.

        Stage timing goes through PHASES. Note the "stage" rows are wall-clock ENVELOPES
        that CONTAIN that stage's own "compile"/"exec" rows -- read the per-kind TOTAL
        lines (compile vs exec is the split that matters: compile is paid once and shrinks
        by cutting shape diversity, exec is paid every inference and shrinks by sharding),
        not the WALL line, which counts the nested time twice.

        Stashes for callers/probes:
            self._last_inference["actions_padded"] [50, 32]  full chunk, padded DoF
            self._last_inference["actions_chunk"]  [50,  7]  full chunk, real DoF
            self._last_inference["actions"]        [10,  7]  what the robot executes
        """
        HEAD = self._cfg["action_head"]
        n_exec, adim, chunk = HEAD["n_action_steps"], HEAD["action_dim"], HEAD["chunk_size"]
        ungated, failed_gates = [], []

        # ---- 1. vision (ViT x2 slots + connector) --------------------------------
        with PHASES.track("stage vision", "stage"):
            vision_tokens = self.run_vision(images_hwc)
        if snr:
            with PHASES.track("gate vision+connector", "gate"):
                # _vision_snr_check scores the ViT post-LN FIRST and only then the
                # connector, so a shuffle bug is never reported as an encoder bug.
                ok = self._vision_snr_check(images_hwc, vision_tokens)
            if not ok:
                failed_gates.append("vision/connector")
                print(f"  [WARN] vision/connector gate below floor -- continuing; inspect cos and rms above")
                if strict_gates:
                    raise SystemExit("vision/connector gate FAILED and --strict-gates is set")

        # ---- 2. prefix (SmolLM2 decoder -> the persistent K/V cache) --------------
        # run_prefix returns nothing useful: its product is the on-device KV cache the
        # expert cross-attends into for all 10 denoise steps.
        if stop_after in ("vision", "connector"):
            print(f"\n  === STOPPED AFTER {stop_after.upper()} (--stop-after) ===")
            PHASES.summary("hardware timing")
            return None

        with PHASES.track("stage prefix", "stage"):
            self.run_prefix(vision_tokens, token_ids, state)
        if snr:
            # _prefix_snr_check is still a stub. Call it only if it has been implemented,
            # and SAY SO when it has not -- an unimplemented gate must never look like a
            # passed one. NotImplementedError is caught rather than pre-tested because
            # the stub raises from inside the body, not at lookup.
            stash = getattr(self, "_last_prefix", None)
            try:
                if stash is None:
                    raise NotImplementedError("run_prefix stashed no _last_prefix")
                with PHASES.track("gate prefix", "gate"):
                    ok = self._prefix_snr_check()
                if not ok:
                    failed_gates.append("prefix")
                    print(f"  [WARN] prefix gate below floor -- continuing; the expert reads this KV, so judge the actions too")
                    if strict_gates:
                        raise SystemExit("prefix gate FAILED and --strict-gates is set")
            except NotImplementedError:
                ungated.append("prefix (hidden + KV cache)")

        # ---- 3. denoise (10 Euler steps x 32 expert layers) -----------------------
        if stop_after == "prefix":
            print("\n  === STOPPED AFTER PREFIX (--stop-after) ===")
            print("    the KV cache is written and gated; the expert reads it next")
            PHASES.summary("hardware timing")
            return None

        with PHASES.track("stage denoise", "stage"):
            actions_chunk = self.run_denoise(noise=noise)      # [50, 7]
        if snr:
            try:
                with PHASES.track("gate denoise", "gate"):
                    ok = self._expert_step_snr_check()
                if not ok:
                    failed_gates.append("denoise")
                    print(f"  [WARN] denoise gate below floor -- continuing; this is the expert alone, prefix error is excluded")
                    if strict_gates:
                        raise SystemExit("denoise gate FAILED and --strict-gates is set")
            except NotImplementedError:
                ungated.append("expert / denoise")

        padded = self._last_denoise["x_t_padded"][:chunk, : self.ACTION_PAD]  # [50,32]
        executed = actions_chunk[:n_exec, :adim]                              # [10, 7]
        if failed_gates:
            print(f"\n  GATES BELOW FLOOR: {', '.join(failed_gates)}")
            print("    the run CONTINUED (gates are advisory unless --strict-gates).")
            print("    judge these on cos-sim and rms, not dB alone: a deep bf16 stack")
            print("    sits near its precision floor by construction, and cos ~0.995 on")
            print("    an intermediate can still give actions that match the oracle.")
        if ungated:
            print(f"  UNGATED STAGES: {', '.join(ungated)}")

        self._last_inference = {"actions_padded": padded,
                                "actions_chunk": actions_chunk,
                                "actions": executed,
                                "vision_tokens": vision_tokens}

        if snr:
            if ungated:
                print("  UNGATED STAGES (no SNR check ran -- 'finite' is NOT "
                      "correctness):")
                for u in ungated:
                    print(f"    - {u}")
            else:
                # NOT ">=40 dB" -- every stage is gated at its MEASURED bf16 floor
                # (SNR_FLOOR) plus cos >= COS_FLOOR. A flat 40 dB is unreachable for a
                # deep bf16 stack and previously rejected a correct encoder.
                print(f"  every stage gated (floors {SNR_FLOOR}, cos>={COS_FLOOR})")
        else:
            print("  (SNR gate disabled -- no stage was checked against the oracle)")

        assert torch.isfinite(executed).all(), "executed actions are non-finite"
        return executed

    # ==================================================================================
    # 7. validation probes
    # ==================================================================================

    # ---- oracle plumbing shared by every probe ------------------------------------
    #
    # hw_gelu=True IS NOT OPTIONAL. The model specifies gelu_pytorch_tanh, but the FPGA's
    # fused MLP activation is x*sigmoid(1.702x) (gelu_hw). Scoring hardware against a
    # gelu_tanh oracle costs ~28 dB on every vision MLP and reads exactly like a real
    # numerical bug, which is the most expensive false lead available here. The gate's
    # job is to score the hardware against what the hardware was ASKED to compute; the
    # gelu_tanh-vs-gelu_hw model-fidelity gap is a separate, deliberate approximation
    # measured once at the task level, not per stage.
    def _ref_oracle(self):
        """Pure-torch oracle built from the SAME real checkpoint the engine loaded.

        Cached: from_checkpoint re-reads a 2.23 GB safetensors file, and the vision gate
        alone would otherwise pay that twice (post-LN stage + connector stage)."""
        ref = getattr(self, "_ref", None)
        if ref is None:
            ref = VeraPulseRef.from_checkpoint(self.script_dir, self._cfg, hw_gelu=True)
            # The checkpoint ships patch_embed as a Conv2d kernel [768,3,16,16], but
            # forward_vision does a plain matmul against [768, C*P*P]. Flatten it here
            # with the SAME reshape _weight_init_vision uses on the way to params DRAM,
            # so the oracle and the device consume identically-ordered columns (and
            # _host_patchify's column order stays the one both agree on).
            pw = ref.sd["vis.patch_embed.weight"]
            if pw.dim() == 4:
                ref.sd["vis.patch_embed.weight"] = pw.reshape(pw.shape[0], -1)
            self._ref = ref
        return ref

    def _host_patchify(self, planes_chw):
        """[3,512,512] channel-planar -> [1024,768], byte-for-byte the ordering the
        DEVICE produces.

        compile_encoder's first op is
            smart_bf16_permute_core(dims=[CH,NPS,P,NPS,P], perm=[1,3,0,2,4])
        i.e. it views the staged image as (c, patch_row, py, patch_col, px) and emits
        (patch_row, patch_col, c, py, px). Row-major flattening of that is
        [1024, c*P*P + kh*P + kw] -- precisely the channel-major flatten
        _weight_init_vision reshaped patch_embed.weight [768,3,16,16] into, and
        numerically identical to conv2d(img, w, stride=16).

        If the host and the device disagreed about this ordering the whole gate would be
        meaningless: the oracle would be scoring a differently-shuffled image and any
        real defect would hide inside the resulting noise floor. So this must stay a
        mirror of compile_encoder's dims/perm, not an independently "obvious" patchify."""
        V = self._cfg["vision"]
        CH, NPS, P = V["num_channels"], V["num_patches_per_side"], V["patch_size"]
        x = torch.as_tensor(planes_chw, dtype=torch.float32).reshape(CH, NPS, P, NPS, P)
        return x.permute(1, 3, 0, 2, 4).reshape(V["num_patches"], CH * P * P).contiguous()

    @staticmethod
    def _diag(name, hw, ref):
        """Print what a human needs to localise a failure: scale on both sides and WHERE
        the worst deviation is. A pure dB number cannot distinguish "one scrambled row"
        from "uniformly noisy", and those have completely different causes."""
        d = (hw - ref).abs()
        flat = int(d.argmax())
        r, c = divmod(flat, hw.shape[-1])
        print(f"       {name}: hw absmax={hw.abs().max():.4f} ref absmax={ref.abs().max():.4f} "
              f"| max|dev|={d.max():.4e} at row {r} col {c} "
              f"(hw={hw[r, c]:+.4f} ref={ref[r, c]:+.4f})")

    def _vision_snr_check(self, images_hwc, hw_tokens):
        """Gate the vision path: ViT post-LN FIRST, then the connector.

        Reference GELU must be x*sigmoid(1.702*x), not erf (else ~28 dB false loss) --
        see _ref_oracle.

        STAGED ON PURPOSE. Post-LN is scored before the connector so a pixel-shuffle
        permutation bug cannot be reported as an encoder bug: the connector's failure
        mode (wrong gather indices) is finite, NaN-free, and produces exactly the same
        "low dB on the final tokens" symptom as a broken attention head. Scoring the
        encoder first splits those two apart in one run.

        PER SLOT, and the worst case is what decides. Both cameras run the same program
        against the same weights, so a slot-1-only failure means state left over from
        slot 0 (the layer_norm zeros scratch, a stale buffer) -- averaging the slots
        together would hide precisely the bug this model is most exposed to.

        valid_rows is all-True BY CONSTRUCTION: a 512x512 image at patch 16 fills the
        32x32 grid exactly, so all 1024 patches are real -- there is no padding to
        exclude here (unlike the prefix/suffix stages, which genuinely have pad rows).
        The mask is passed explicitly rather than defaulted so the claim is auditable at
        the call site, and it is all-True precisely so it cannot mask a real defect.

        THRESHOLD IS SNR_FLOOR["vision"] (26 dB), NOT 40. Measured ceiling for this
        stack is 29.0 dB (fp32 vs bf16, same GELU) and real hardware came in at 32.7 dB
        -- above the pure-bf16 sim, because the matmul accumulates wider internally. A
        flat 40 dB gate rejected a CORRECT encoder; do not restore it.

        Pass criteria are BOTH: snr >= floor (precision) AND cos >= COS_FLOOR
        (structure). Cos is what separates "bf16 noise" from "scrambled/misaligned" --
        a permute or offset bug collapses cos while leaving SNR plausible.
        """
        V = self._cfg["vision"]
        S, H = V["num_patches"], V["hidden_size"]
        stash = getattr(self, "_last_vision", None)
        assert stash is not None, "_vision_snr_check must run after run_vision"

        images = torch.as_tensor(stash["images"])
        ref = self._ref_oracle()
        if self.VIS_LAYERS is not None:
            # Truncate the ORACLE to the same depth or the comparison is meaningless.
            ref.n_vis = int(self.VIS_LAYERS)
            print(f"  [bisect] oracle truncated to {ref.n_vis} ViT layers")

        print("  vision gate (oracle: real checkpoint, hw_gelu=x*sigmoid(1.702x)):")
        # all-True by construction: 1024 real patches, zero padding in the vision stage.
        valid = torch.ones(S, dtype=torch.bool)

        worst, ok_all = float("inf"), True
        self._ref_post_ln = []
        for i in range(V["num_image_slots"]):
            # Derive the oracle's input from the SAME array the device consumed, through
            # the SAME reshape/permute the device's patchify performs.
            planes = images[i].permute(2, 0, 1).contiguous().float()   # HWC -> [3,512,512]
            with torch.no_grad():
                r = ref.forward_vision(self._host_patchify(planes))
            self._ref_post_ln.append(r)
            hw = stash["post_ln"][i]
            assert hw.shape == (S, H) == tuple(r.shape), (
                f"post_ln shape mismatch: hw {tuple(hw.shape)} ref {tuple(r.shape)}")
            s = self._snr(hw, r, valid)
            ok = report(f"vis.post_ln[slot {i}]", hw, r, valid,
                        threshold=SNR_FLOOR["vision"])
            # cos is the primary structural signal for a 12-layer bf16 stack: SNR near
            # the precision floor is expected, a collapsed cos is not.
            c = cos_sim(hw, r, valid)
            if c < COS_FLOOR:
                print(f"       cos {c:.6f} < {COS_FLOOR} -- STRUCTURAL error (scrambled "
                      f"or misaligned), not precision")
                ok = False
            self._diag(f"post_ln[{i}]", hw, r)
            worst, ok_all = min(worst, s), ok_all and ok

        print(f"    worst vision post_ln slot: {worst:.2f} dB")
        if not ok_all:
            # Fail at the FIRST stage below threshold: running the connector check now
            # would only restate the encoder's error with a permute layered on top.
            print("    [FAIL] ViT encoder below 40 dB -- stopping before the connector "
                  "so the encoder is not blamed on the shuffle (or vice versa)")
            return False

        return self._connector_snr_check(hw_tokens)

    def _connector_snr_check(self, hw_out):
        """Gate the pixel-shuffle + projection, INDEPENDENTLY of the encoder.

        This is the only place the on-device shuffle indices are validated. The
        connector's shuffle is smart_bf16_permute_core(dims=[8,4,8,4,768],
        perm=[0,2,1,3,4]); the oracle side is the torch pixel_shuffle() at module scope.
        A wrong permutation is a REAL and separate failure mode from the encoder: it is
        finite, never NaN, never out of range -- every value present, in the wrong row --
        so only a shuffle-sensitive metric catches it. Cosine similarity over the flat
        tensor is reported alongside SNR for exactly that reason.

        ISOLATION: the reference is fed the HARDWARE's post-LN tensor, not the oracle's.
        That makes this score the shuffle+matmul alone; encoder drift has already been
        gated one stage earlier, and folding it in again would only inflate the error
        budget and blur which stage moved.

        valid_rows is all-True by construction: 1024 patches / 16 = 64 connector tokens,
        all real, no padding.

        hw_out is the [128,960] concatenation run_vision returned; the per-slot halves
        are scored separately and the worst reported."""
        C, V = self._cfg["connector"], self._cfg["vision"]
        T, N = C["tokens_out"], C["output_size"]
        stash = getattr(self, "_last_vision", None)
        assert stash is not None, "_connector_snr_check must run after run_vision"
        ref = self._ref_oracle()

        hw_out = torch.as_tensor(hw_out, dtype=torch.float32)
        slots = V["num_image_slots"]
        assert hw_out.shape == (slots * T, N), (
            f"connector tokens should be [{slots * T},{N}], got {tuple(hw_out.shape)}")

        print("  connector gate (validates the smart_bf16_permute shuffle indices):")
        # all-True by construction: all 64 connector tokens per slot are real.
        valid = torch.ones(T, dtype=torch.bool)

        worst, ok_all = float("inf"), True
        for i in range(slots):
            with torch.no_grad():
                r = ref.forward_connector(stash["post_ln"][i])
            hw = stash["connector"][i]
            assert hw.shape == (T, N) == tuple(r.shape)
            s = self._snr(hw, r, valid)
            ok = report(f"connector[slot {i}]", hw, r, valid, threshold=SNR_FLOOR["connector"])
            self._diag(f"connector[{i}]", hw, r)
            worst, ok_all = min(worst, s), ok_all and ok

            # The returned [128,960] must be the per-slot buffers in order; if the
            # concatenation drifted, downstream prefix rows would be silently swapped.
            assert torch.equal(hw_out[i * T:(i + 1) * T], hw), (
                f"returned tokens rows {i * T}..{(i + 1) * T} do not match slot {i}'s "
                f"connector readback -- the concatenation order is wrong")

        print(f"    worst connector slot: {worst:.2f} dB")
        if not ok_all:
            print("    [FAIL] connector below 40 dB while the encoder passed -- suspect "
                  "the pixel-shuffle gather indices, not the ViT")
            return False

        # end-to-end (oracle patches -> oracle tokens) once both stages pass, as the
        # number that actually characterises what the prefix will consume. Only
        # available when the encoder stage ran first and cached its oracle post-LN.
        ref_post_ln = getattr(self, "_ref_post_ln", None)
        if ref_post_ln is not None:
            with torch.no_grad():
                e2e = torch.cat([ref.forward_connector(r) for r in ref_post_ln], 0)
            # CHAINED vision -> connector, so it inherits the ENCODER's error and can
            # never beat it: the per-slot connector checks above isolate the shuffle
            # (they scored ~52 dB against the HW post-LN), while this one measures the
            # whole path. Gate it at the VISION floor -- gating it at the connector's
            # would fail a correct pipeline every time.
            report("vision e2e tokens", hw_out, e2e,
                   torch.ones(hw_out.shape[0], dtype=torch.bool),
                   threshold=SNR_FLOOR["vision"])
        return True

    def _probe_copy(self, src, dst, elems):
        """CHUNKED DRAM->DRAM copy for bisect probes.

        _ae_dram_copy stages the whole transfer through SRAM in one shot, which is fine
        at its design size (the Euler retire, 4096 elements) and catastrophic here: a
        vision probe is 1024*768 = 786,432 elements against a URAM capacity of
        URAM_NEAR_FULL_ELEMENTS = 0xFFF*64 = 262,080. Overflowing it does not just lose
        the probe, it corrupts the pipeline the probe sits inside -- the patch probe is
        emitted between the patch matmul and the eltwise that builds `embed`, so an
        overflowing copy scrambles the real dataflow and the bisect then reports a
        divergence IT CAUSED. Instrumentation must not perturb what it measures."""
        step = (URAM_NEAR_FULL_ELEMENTS // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
        off = 0
        while off < elems:
            n = min(step, elems - off)
            self.accelerator_memory_to_sram(src + off * 2, 0x00000, n)
            self.sram_to_accelerator_memory(0x00000, dst + off * 2, n)
            off += n

    def bisect_vision(self, images_hwc):
        """Walk ONE ViT layer op by op and score EVERY intermediate against a
        step-by-step host reference. Answers "which op diverges", which the post_ln gate
        structurally cannot: it observes only the end of the stack.

        The reference below is written OUT LONGHAND rather than calling
        VeraPulseRef.forward_vision, on purpose. forward_vision returns one tensor; here
        we need the value after each individual op, in the same order and with the same
        operands the emitter used. Writing it separately also means a bug in the shared
        reference cannot hide by being present on both sides of the comparison.

        Run with VIS_LAYERS=1 so nothing downstream overwrites the probes.

        Reading it: the FIRST op whose cos drops is the culprit. Ops before it are fine
        by construction, ops after inherit its error and tell you nothing.
        """
        V = self._cfg["vision"]
        S, H, I = V["num_patches"], V["hidden_size"], V["intermediate_size"]
        D, NH, eps = V["head_dim"], V["num_heads"], V["layer_norm_eps"]
        sd = self._ref_oracle().sd
        w = lambda n: sd[n].float()

        images = torch.as_tensor(images_hwc)
        # THE LAST SLOT, NOT SLOT 0. run_vision encodes every camera in sequence through
        # ONE set of buffers, so when it returns the probes hold the FINAL slot's values.
        # Referencing images[0] compares slot 1's hardware against slot 0's oracle: two
        # different pictures through identical weights, which produces matching rms and
        # absmax with a collapsed cos -- indistinguishable from a permutation bug, and
        # it cost a full debugging cycle. Score the slot whose data is actually resident.
        slot = images.shape[0] - 1
        print(f"  (scoring slot {slot} -- the last one run_vision encoded, i.e. the one "
              f"still resident in the probe buffers)")
        planes = images[slot].permute(2, 0, 1).contiguous().float()
        x0 = self._host_patchify(planes).float()          # [S, C*P*P]

        # ---- host reference, op by op, in emission order -------------------------
        r = {}
        r["patch"] = x0 @ w("vis.patch_embed.weight").T + w("vis.patch_embed.bias")
        r["embed"] = r["patch"] + w("vis.pos_embed.weight")[:S]
        r["ln1"] = layer_norm(r["embed"], w("vis.0.layer_norm1.weight"),
                              w("vis.0.layer_norm1.bias"), eps)
        for p in ("q_proj", "k_proj", "v_proj"):
            r[p] = r["ln1"] @ w(f"vis.0.{p}.weight").T + w(f"vis.0.{p}.bias")
        q, k, v = (r[p].view(S, NH, D).transpose(0, 1)
                   for p in ("q_proj", "k_proj", "v_proj"))
        r["attn"] = attend(q, k, v)
        r["o_proj"] = r["attn"] @ w("vis.0.out_proj.weight").T + w("vis.0.out_proj.bias")
        r["resid1"] = r["embed"] + r["o_proj"]
        r["ln2"] = layer_norm(r["resid1"], w("vis.0.layer_norm2.weight"),
                              w("vis.0.layer_norm2.bias"), eps)
        # gelu_hw: the DEVICE computes x*sigmoid(1.702x), not gelu_tanh. Scoring the
        # fused epilogue against the model-spec activation would fake a ~28 dB loss.
        r["fc1"] = gelu_hw(r["ln2"] @ w("vis.0.fc1.weight").T + w("vis.0.fc1.bias"))
        r["fc2"] = r["fc1"] @ w("vis.0.fc2.weight").T + w("vis.0.fc2.bias")
        r["resid2"] = r["resid1"] + r["fc2"]

        # ---- hardware, same order ------------------------------------------------
        hw = {
            "patch":  self._read_bf16(self.VIS_P_PATCH_DRAM, (S, H), label="p_patch"),
            "embed":  self._read_bf16(self.VIS_IO_A_DRAM, (S, H), label="p_embed"),
            "ln1":    self._read_bf16(self.VIS_P_LN1_DRAM, (S, H), label="p_ln1"),
            "q_proj": self._read_bf16(self.VIS_Q_DRAM, (S, H), label="p_q"),
            "k_proj": self._read_bf16(self.VIS_K_DRAM, (S, H), label="p_k"),
            "v_proj": self._read_bf16(self.VIS_V_DRAM, (S, H), label="p_v"),
            "attn":   self._read_bf16(self.VIS_ATTN_RESULT_DRAM, (S, H), label="p_attn"),
            "o_proj": self._read_bf16(self.VIS_O_PROJ_DRAM, (S, H), label="p_o"),
            "resid1": self._read_bf16(self.VIS_RESIDUAL_DRAM, (S, H), label="p_r1"),
            "ln2":    self._read_bf16(self.VIS_P_LN2_DRAM, (S, H), label="p_ln2"),
            "fc1":    self._read_bf16(self.VIS_MLP_INTER_DRAM, (S, I), label="p_fc1"),
            "fc2":    self._read_bf16(self.VIS_MLP_OUT_DRAM, (S, H), label="p_fc2"),
            "resid2": self._read_bf16(self.VIS_IO_B_DRAM, (S, H), label="p_r2"),
        }

        # ---- OPERAND CHECK: verify the INPUTS before blaming the op ---------------
        # The op-by-op scores below compare OUTPUTS. If an output is wrong, that is only
        # evidence about the op when its operands are known good -- otherwise a bad
        # staged input or a mis-stored weight reads exactly like a broken kernel. Both
        # operands of the patch matmul are read back from DRAM and scored against the
        # host tensors that were supposed to be written there. bf16 round-trips
        # losslessly, so anything below ~infinite SNR here is a staging/store bug, not
        # precision.
        print(f"\n=== OPERAND CHECK (what actually landed in DRAM) ===")
        C, P = V["num_channels"], V["patch_size"]
        hw_A = self._read_bf16(self.VIS_PIXEL_IN_DRAM, (S, C * P * P), label="A_staged")
        host_A = x0.to(torch.bfloat16).float()
        report("A: staged patches", hw_A, host_A,
               torch.ones(S, dtype=torch.bool), threshold=60.0)

        hw_B = self._read_bf16(self.patch_weight_addr, (H, C * P * P), label="B_weight")
        host_B = w("vis.patch_embed.weight").reshape(H, C * P * P).to(torch.bfloat16).float()
        report("B: patch_embed weight", hw_B, host_B,
               torch.ones(H, dtype=torch.bool), threshold=60.0)

        hw_bias = self._read_bf16(self.patch_bias_addr, (1, H), label="bias")
        report("C: patch_embed bias", hw_bias,
               w("vis.patch_embed.bias").reshape(1, H).to(torch.bfloat16).float(),
               torch.ones(1, dtype=torch.bool), threshold=60.0)

        # If the operands are clean, recompute the op from the DEVICE'S OWN operands.
        # Matching here while the device output differs isolates the KERNEL; differing
        # here means our host model of the op (A @ B.T + c) is wrong, e.g. a transpose
        # convention -- which no amount of staring at the emitter would reveal.
        recomputed = hw_A @ hw_B.T + hw_bias.reshape(-1)
        print(f"    recomputed A@B.T+c from DEVICE operands: "
              f"cos vs oracle = {cos_sim(recomputed, r['patch'], torch.ones(S, dtype=torch.bool)):.6f}")
        alt = hw_A @ hw_B + hw_bias.reshape(-1)
        print(f"    same but WITHOUT the transpose (A@B+c):   "
              f"cos vs oracle = {cos_sim(alt, r['patch'], torch.ones(S, dtype=torch.bool)):.6f}"
              f"   <- if THIS one is ~1.0, the stored weight needs transposing")

        print(f"\n=== VISION LAYER-0 OP BISECT (slot {slot}) ===")
        print("  the FIRST row whose cos drops is the culprit; later rows inherit it.")
        order = ["patch", "embed", "ln1", "q_proj", "k_proj", "v_proj", "attn",
                 "o_proj", "resid1", "ln2", "fc1", "fc2", "resid2"]
        rows = None
        first_bad = None
        for name in order:
            a, b = hw[name], r[name]
            m = torch.ones(a.shape[0], dtype=torch.bool)
            s, c = snr_db(a, b, m), cos_sim(a, b, m)
            flag = ""
            if c < COS_FLOOR and first_bad is None:
                first_bad, flag = name, "   <== FIRST DIVERGENCE"
            elif c < COS_FLOOR:
                flag = "   (inherited)"
            print(f"  {name:8s} {str(tuple(a.shape)):12s} snr={s:8.2f}dB cos={c:.6f} "
                  f"rms={rms(a, m):8.4f}/{rms(b, m):8.4f} "
                  f"absmax={float(a.abs().max()):8.3f}/{float(b.abs().max()):8.3f}{flag}")
        if first_bad:
            print(f"\n  FIRST DIVERGENCE AT: {first_bad}")
            print(f"  everything before it matched, so the defect is in that op's emission")
            print(f"  (operands, addresses, or the kernel itself) -- not upstream.")
        else:
            print("\n  every op within tolerance -- layer 0 is clean; re-run with a")
            print("  deeper --vis-layers to find where it starts.")
        return first_bad

    def bisect_prefix(self):
        """Walk ONE prefix layer op by op, scoring every intermediate.

        Run with PREFIX_LAYERS = N so layer N-1 is the LAST compiled layer; its
        intermediates then survive execution and only LM_PRE_NORM (written by both norms)
        needs a probe copy.

        Reference is written longhand -- residual stream carried forward through layers
        0..N-2, then layer N-1 decomposed op by op. Deliberately NOT a call into
        VeraPulseRef.forward_prefix: that returns only the final hidden, and a shared
        helper's bug would cancel on both sides of the comparison.

        Reads: the FIRST op whose cos drops is the culprit; later rows inherit it. The
        per-layer KV curve says the damage starts at L2, so `--bisect-prefix 3` (compile
        3 layers, probe layer 2) is the run that matters.
        """
        L = self._cfg["lm"]
        PM, H, I = self.PREFILL_MAX_SEQ_LEN, self.HIDDEN_SIZE, self.INTERMEDIATE_SIZE
        NH, NKV, D = L["num_heads"], L["num_kv_heads"], L["head_dim"]
        eps, KV = L["rms_norm_eps"], NKV * D
        n_lm = int(self.PREFIX_LAYERS)
        tgt = n_lm - 1
        stash = getattr(self, "_last_prefix", None)
        assert stash is not None, "bisect_prefix must run after run_prefix"
        ref = self._ref_oracle()
        w = lambda n: ref.sd[n].float()

        x = stash["prefix_in"].float()
        pos = torch.arange(PM)
        cos_t, sin_t = rope_tables(pos, D, L["rope_theta"])

        # THE REFERENCE MUST CARRY THE SAME MASK THE DEVICE DOES. compile_prefix passes
        # PREFIX_BIAS_DRAM to every flash call, masking the pad columns beyond valid_len;
        # scoring against an UNMASKED reference makes attention look broken in a way that
        # grows with depth, because the pad rows start near 1e-6 and accumulate real
        # magnitude as layers run. Measured with the mask missing: attn 24.67 dB at
        # layer 0 (pads still tiny) collapsing to 0.74 dB at layer 1 (pads now large) --
        # a perfect impostor for a RoPE or GQA bug.
        vlen = int(stash.get("valid_len", self.PREFIX_LEN))
        bias = torch.zeros(PM, PM)
        if vlen < PM:
            bias[:, vlen:] = float("-inf")
            bias[vlen:, :] = float("-inf")
            bias[vlen:, vlen:] = 0.0

        def one_layer(h, li, capture=None):
            """Exactly the emitter's op order. `capture` collects layer-`li` internals."""
            n1 = rms_norm(h, w(f"lm.{li}.input_layernorm.weight"), eps)
            q = n1 @ w(f"lm.{li}.q_proj.weight").T
            k = n1 @ w(f"lm.{li}.k_proj.weight").T
            v = n1 @ w(f"lm.{li}.v_proj.weight").T
            qh = apply_rope(q.view(PM, NH, D).transpose(0, 1), cos_t, sin_t)
            kh = apply_rope(k.view(PM, NKV, D).transpose(0, 1), cos_t, sin_t)
            vh = v.view(PM, NKV, D).transpose(0, 1)
            a = attend(qh, kh, vh, bias)
            o = a @ w(f"lm.{li}.o_proj.weight").T
            r1 = h + o
            n2 = rms_norm(r1, w(f"lm.{li}.post_attention_layernorm.weight"), eps)
            g = F.silu(n2 @ w(f"lm.{li}.gate_proj.weight").T)
            u = n2 @ w(f"lm.{li}.up_proj.weight").T
            mult = g * u
            dn = mult @ w(f"lm.{li}.down_proj.weight").T
            out = r1 + dn
            if capture is not None:
                capture.update(norm1=n1, q_proj=q, k_proj=k, v_proj=v, attn=a,
                               o_proj=o, resid1=r1, norm2=n2, gate=g, up=u,
                               mult=mult, down=dn, hidden=out)
            return out

        with torch.no_grad():
            h = x
            for li in range(tgt):
                h = one_layer(h, li)
            r = {}
            one_layer(h, tgt, capture=r)

        h_out = self.LM_OUTPUT_DRAM if tgt % 2 == 0 else self.LM_INPUT_DRAM
        hw = {
            "norm1":  self._read_bf16(self.LM_P_NORM1_DRAM, (PM, H), label="p_n1"),
            "q_proj": self._read_bf16(self.LM_Q_DRAM, (PM, H), label="p_q"),
            "k_proj": self._read_bf16(self.LM_K_PROJ_DRAM, (PM, KV), label="p_k"),
            "v_proj": self._read_bf16(self.LM_V_PROJ_DRAM, (PM, KV), label="p_v"),
            "attn":   self._read_bf16(self.LM_ATTN_RESULT_DRAM, (PM, H), label="p_attn"),
            "o_proj": self._read_bf16(self.LM_O_PROJ_DRAM, (PM, H), label="p_o"),
            "resid1": self._read_bf16(self.LM_RESIDUAL_DRAM, (PM, H), label="p_r1"),
            "norm2":  self._read_bf16(self.LM_P_NORM2_DRAM, (PM, H), label="p_n2"),
            "gate":   self._read_bf16(self.LM_MLP_GATE_DRAM, (PM, I), label="p_gate"),
            "up":     self._read_bf16(self.LM_MLP_UP_DRAM, (PM, I), label="p_up"),
            "mult":   self._read_bf16(self.LM_MLP_MULT_DRAM, (PM, I), label="p_mult"),
            "down":   self._read_bf16(self.LM_MLP_DOWN_DRAM, (PM, H), label="p_down"),
            "hidden": self._read_bf16(h_out, (PM, H), label="p_hidden"),
        }
        # k_proj/v_proj are compared PRE-rope on the reference side; the device ropes
        # them in place into the KV cache, so score the un-roped projection here and let
        # the cache curve cover the roped values.
        r["k_proj"], r["v_proj"] = r["k_proj"], r["v_proj"]

        valid = torch.zeros(PM, dtype=torch.bool)
        valid[: int(stash.get("valid_len", self.PREFIX_LEN))] = True
        print(f"\n=== PREFIX LAYER-{tgt} OP BISECT ({n_lm} layers compiled) ===")
        print("  the FIRST row whose cos drops is the culprit; later rows inherit it.")
        first_bad = None
        for name in ("norm1", "q_proj", "k_proj", "v_proj", "attn", "o_proj",
                     "resid1", "norm2", "gate", "up", "mult", "down", "hidden"):
            a, b = hw[name], r[name]
            s, c = snr_db(a, b, valid), cos_sim(a, b, valid)
            flag = ""
            if c < COS_FLOOR and first_bad is None:
                first_bad, flag = name, "   <== FIRST DIVERGENCE"
            elif c < COS_FLOOR:
                flag = "   (inherited)"
            print(f"  {name:7s} {str(tuple(a.shape)):12s} snr={s:8.2f}dB cos={c:.6f} "
                  f"rms={rms(a, valid):8.4f}/{rms(b, valid):8.4f} "
                  f"absmax={float(a[valid].abs().max()):8.3f}/"
                  f"{float(b[valid].abs().max()):8.3f}{flag}")
        print(f"\n  {'FIRST DIVERGENCE AT: ' + first_bad if first_bad else 'layer clean'}")
        return first_bad

    def _prefix_snr_check(self):
        """Gate the prefix against the torch oracle. Runs after run_prefix.

        SCORED ON THE HARDWARE'S OWN VISION TOKENS, not the oracle's: feeding the
        reference the same [128,960] the device actually produced isolates the prefix
        from upstream vision drift, so a failure here is the prefix's fault. Same
        factoring the connector check uses.

        BOTH the hidden state and the KV CACHE are checked, and the KV matters more: the
        cache is the action expert's only prefix input, so a scrambled strided copy into
        LAYER0_K/V_DRAM breaks the actions while the hidden state still looks fine.
        Layer 0 and the last layer are both sampled -- layer 0 isolates the write path,
        the last layer shows accumulated drift.

        Thresholds are the MEASURED floors (SNR_FLOOR), not a flat 40 dB: a 32-layer bf16
        stack tops out near 28.8 dB. Cos is the structural signal; SNR is precision.

        Masked rows are excluded via the valid length run_prefix recorded -- the pad rows
        beyond valid_len hold different garbage on device than on host and would dominate
        the metric (the pi05 -3.4 dB vs +50 dB trap)."""
        stash = getattr(self, "_last_prefix", None)
        assert stash is not None, "_prefix_snr_check must run after run_prefix"
        PM = self.PREFILL_MAX_SEQ_LEN
        valid_len = int(stash.get("valid_len", self.PREFIX_LEN))
        valid = torch.zeros(PM, dtype=torch.bool)
        valid[:valid_len] = True          # pad rows beyond valid_len are NOT scored

        ref = self._ref_oracle()
        print(f"  prefix gate (oracle fed the HW's OWN assembled prefix, "
              f"valid={valid_len}/{PM}):")
        # Feed the oracle the EXACT [192,960] the device consumed, not a re-derivation
        # from token_ids/state. That isolates the 32-layer stack from the host-side
        # embedding assembly: if this fails, it is the emitted layers, not embed/state
        # projection. (embed_and_concat_prefix is separately checkable against
        # VeraPulseRef.build_prefix on the host, where no hardware is involved.)
        x = stash["prefix_in"].float()
        assert tuple(x.shape) == (PM, self.HIDDEN_SIZE), (
            f"prefix_in {tuple(x.shape)} != ({PM},{self.HIDDEN_SIZE})")
        # compile_prefix rotates row r with rope entry r (plain arange), which is only
        # equal to the reference's cumsum(mask)-1 because padding is right-aligned.
        pos = torch.arange(PM)
        # SAME MASK THE DEVICE USES. compile_prefix passes PREFIX_BIAS_DRAM to every
        # flash call; an unmasked oracle lets the pad rows (beyond valid_len) participate
        # in attention on the reference side only. That error compounds per layer -- the
        # pads start at 1e-6 and grow -- and produced the entire "prefix is broken"
        # picture: cos 0.552 over 32 layers, a KV curve decaying from 0.99999 to 0.47,
        # and an `attn` op that looked 25 dB below its own inputs. Every one of those was
        # the missing mask, not the emitter. Same construction as build_attn_bias.
        bias = torch.zeros(PM, PM)
        if valid_len < PM:
            bias[:, valid_len:] = float("-inf")
            bias[valid_len:, :] = float("-inf")
            bias[valid_len:, valid_len:] = 0.0
        with torch.no_grad():
            r_hidden, r_kv = ref.forward_prefix(x, pos, bias=bias)

        ok = report("prefix hidden", stash["hidden"], r_hidden, valid,
                    threshold=SNR_FLOOR["prefix"])
        c = cos_sim(stash["hidden"], r_hidden, valid)
        if c < COS_FLOOR:
            print(f"       cos {c:.6f} < {COS_FLOOR} -- STRUCTURAL error, not precision")
            ok = False

        # --- FULL PER-LAYER KV CURVE ------------------------------------------------
        # The cache holds every layer, so the whole degradation profile is one readback
        # away -- no recompile, no second run. This is the measurement that separates
        # the two candidate explanations, and eyeballing layers 0 and 31 alone cannot:
        #   * SMOOTH decay  -> per-layer numerical drift (precision floor)
        #   * CLIFF at layer k -> something structural happens AT k (a buffer that only
        #     collides past a certain offset, an allocator that wraps, a ping-pong parity
        #     bug that only bites on one side)
        D, NKV = self.HEAD_DIM, self.NUM_KV_HEADS
        print("    per-layer KV curve (k only, cos vs oracle):")
        curve = []
        for li in range(self.NUM_LAYERS):
            heads = [self._read_bf16(self.LAYER0_K_DRAM + li * self.KV_LAYER_STRIDE
                                     + h * self.KV_HEAD_STRIDE, (PM, D),
                                     label=f"curve_k_l{li}h{h}")
                     for h in range(NKV)]
            hw_t = torch.stack(heads)[:, :valid_len].reshape(-1, D)
            rf_t = torch.stack([r_kv[li][0][h] for h in range(NKV)])[:, :valid_len].reshape(-1, D)
            m = torch.ones(hw_t.shape[0], dtype=torch.bool)
            curve.append((li, snr_db(hw_t, rf_t, m), cos_sim(hw_t, rf_t, m),
                          rms(hw_t, m), rms(rf_t, m)))
        for li, s, c, rh, rr in curve:
            bar = "#" * max(0, min(40, int(c * 40)))
            print(f"      L{li:2d} snr={s:7.2f}dB cos={c:.6f} rms={rh:7.4f}/{rr:7.4f} {bar}")
        # RANK BY dB, NOT cos. cos is compressed near 1.0: a move from 0.99996 to
        # 0.99613 reads as a trivial -0.004 while being a TWENTY dB collapse, so ranking
        # by cos delta points at whatever noise happens to sit lowest on an already-
        # decayed tail. Measured here: cos-ranking flagged L19->L20 (-0.218) while the
        # actual event was L1->L2 (-19.9 dB); L20 then RECOVERED to 0.79/0.86/0.90 by
        # L30, and real cliffs do not recover.
        drops = [(curve[i][1] - curve[i - 1][1], i) for i in range(1, len(curve))]
        worst_drop, worst_i = min(drops)
        print(f"    largest single-layer SNR drop: {worst_drop:+.2f} dB at "
              f"L{worst_i - 1} -> L{worst_i}  (cos {curve[worst_i - 1][2]:.6f} -> "
              f"{curve[worst_i][2]:.6f})")
        # Is the decay smooth or is one layer anomalous? Compare the worst drop against
        # the median: drift degrades at a roughly constant per-layer rate.
        med = sorted(d for d, _ in drops)[len(drops) // 2]
        print(f"    median per-layer drop: {med:+.2f} dB")
        if worst_drop < 3 * med - 6:
            print(f"    ANOMALY at L{worst_i}: that layer loses far more than the others "
                  f"-- inspect it specifically, this is not uniform drift")
        else:
            print(f"    decay is roughly uniform -- consistent with per-layer numerical "
                  f"drift rather than one broken layer")
        # HW rms drifts ABOVE the oracle as depth grows (L31: 2.34 vs 1.87). A one-sided
        # magnitude growth is the signature of accumulation error, not of a permutation
        # or an addressing bug, which preserve magnitude.
        print(f"    rms ratio hw/ref: L0={curve[0][3] / curve[0][4]:.4f} "
              f"L{len(curve) - 1}={curve[-1][3] / curve[-1][4]:.4f} "
              f"(one-sided growth => accumulation, not scrambling)")

        kv_ok = True
        for tag, li in (("L0", 0), (f"L{self.NUM_LAYERS - 1}", self.NUM_LAYERS - 1)):
            for name, base, r_idx in (("k", self.LAYER0_K_DRAM, 0),
                                      ("v", self.LAYER0_V_DRAM, 1)):
                hw_heads, ref_heads = [], []
                for h in range(NKV):
                    addr = base + li * self.KV_LAYER_STRIDE + h * self.KV_HEAD_STRIDE
                    hw_heads.append(self._read_bf16(addr, (PM, D),
                                                    label=f"kv_{tag}_{name}_h{h}"))
                    ref_heads.append(r_kv[li][r_idx][h])
                hw_t = torch.stack(hw_heads)[:, :valid_len].reshape(-1, D)
                rf_t = torch.stack(ref_heads)[:, :valid_len].reshape(-1, D)
                m = torch.ones(hw_t.shape[0], dtype=torch.bool)
                kv_ok &= report(f"KV {tag} {name}", hw_t, rf_t, m,
                                threshold=SNR_FLOOR["prefix_kv"])
        if not kv_ok:
            print("    KV FAILED -- the expert cross-attends into this cache, so every")
            print("    action is wrong downstream. Suspect the per-head destination")
            print("    offset (+h*KV_HEAD_STRIDE) in compile_prefix's strided copy.")
        return ok and kv_ok

    def bisect_expert(self):
        """Walk ONE expert layer op by op at Euler step 0. Mirrors bisect_prefix.

        WHY step 0 and one layer: the per-step curve showed the hardware already 17 dB
        BELOW the strict-16 simulated floor at step 0 (34.92 vs 52.19 dB), i.e. a single
        pass through the expert is defective and the integrator merely amplifies it. So
        the target is one layer of one pass, not the loop.

        Also checks the 480->512 PAD LANES, which have never been verified on hardware
        and are the precondition for the sqrt(480/512) RMSNorm gamma fold being exact.
        """
        E = self._cfg["expert"]
        M, HP, I = self.SUFFIX_LEN_PAD, self.E_HIDDEN_PAD, self.E_INTER
        HR = E["hidden_size"]                      # 480 real lanes
        NH, NKV, D = E["num_heads"], E["num_kv_heads"], E["head_dim"]
        eps, Q, KV = E["rms_norm_eps"], self.E_Q_OUT, self.E_KV_OUT
        n_ae = int(self.EXPERT_LAYERS or self.E_LAYERS)
        tgt = n_ae - 1
        ref = self._ref_oracle()
        w = lambda n: ref.sd[n].float()

        # ---- PAD LANES: the invariant the gamma fold rests on ---------------------
        print("\n=== EXPERT PAD-LANE CHECK (lanes [480:512] must be exactly zero) ===")
        worst = 0.0
        for nm, addr, wide in (("suffix embed", self.AE_P_EMBED_DRAM, HP),
                               ("norm1", self.AE_P_NORM1_DRAM, HP),
                               ("norm2", self.AE_P_NORM2_DRAM, HP),
                               ("residual", self.AE_RESIDUAL_DRAM, HP),
                               ("o_proj", self.AE_O_PROJ_DRAM, HP),
                               ("mlp_down", self.AE_MLP_DOWN_DRAM, HP)):
            t = self._read_bf16(addr, (M, wide), label=f"pad_{nm}")
            pad = t[:, HR:]
            mx = float(pad.abs().max())
            worst = max(worst, mx)
            # bf16's smallest NORMAL is ~1.2e-38; anything at 1e-37 is a denormal that
            # contributes ~1e-74 to a mean-square whose real lanes are O(1). "Exactly
            # zero" was the wrong test -- it flagged harmless denormals as dirt. What
            # would actually break the sqrt(480/512) fold is a lane at O(1e-3) or above,
            # where 32 of them start moving the mean-square.
            tag = "OK" if mx < 1e-30 else "<== DIRTY (would perturb the gamma fold)"
            print(f"  {nm:14s} pad |max|={mx:.3e}  {tag}")
        if worst >= 1e-30:
            print(f"  PAD LANES DIRTY (max {worst:.3e}). The expert RMSNorm folds")
            print(f"  sqrt(480/512) into gamma on the assumption these are zero; nonzero")
            print(f"  lanes enter the mean-square and every expert norm is mis-scaled.")

        # ---- host reference, layer tgt, longhand ---------------------------------
        hw_embed = self._read_bf16(self.AE_P_EMBED_DRAM, (M, HP), label="p_embed")
        h = hw_embed.clone()          # start from the DEVICE's own suffix embedding so
                                      # this isolates the LAYER, not the embed path
        # FAULT 8: the device's self-branch rope base is the VALID prefix length, not the
        # nominal PREFIX_LEN (_ae_refresh_runtime_constants re-DMAs the table per
        # inference). Mirror it or this oracle scores a phase-shifted model.
        vlen_p = int(getattr(self, "_last_prefix", {}).get(
            "valid_len", getattr(self, "_prefix_valid_len", self.PREFIX_LEN)))
        pos = (torch.arange(vlen_p, vlen_p + M)
               if self.EXPERT_ROPE_CONTINUES else torch.arange(M))
        cos_t, sin_t = rope_tables(pos, D, self._cfg["lm"]["rope_theta"])
        # FAULT 7: cross layers rope Q at REBASED positions 0..M-1 (AE_ROPE_GQA_CROSS).
        cos_x, sin_x = rope_tables(torch.arange(M), D, self._cfg["lm"]["rope_theta"])

        # THE MASKS THE DEVICE USES. The suffix is CHUNK=50 real rows padded to
        # SUFFIX_LEN_PAD=64, and _emit_expert_layer passes AE_BIAS_SELF_DRAM (self) /
        # AE_BIAS_CROSS_DRAM (cross) to every flash call. Scoring against an unmasked
        # reference makes `attn` look broken while its inputs are clean -- the EXACT
        # artifact that made the prefix look like a 32-layer structural failure for
        # several runs. Same bug, written twice; hence the explicit construction here
        # rather than a None default.
        # COLUMN-ONLY, matching the device exactly. _ae_tensor_init builds
        #     col_tok = arange(QB) // G ; mask = where(col_tok < SUFFIX_LEN, 0, -inf)
        # i.e. it masks pad COLUMNS and leaves pad ROWS to compute normally ("rows are
        # all computed, pad rows included -- they just produce output nobody reads").
        # Masking rows here too made those 14 rows diverge wildly from the device and,
        # because they were also being SCORED, dragged attn from 18.9 dB down to 13.8 --
        # the mask "fix" was worse than no mask. Mirror the device, do not improve on it.
        chunk = self.CHUNK
        # FAULT 4: the device's self bias is now [QB, PM+M] over the COMBINED
        # [prefix ; suffix] key sequence. Per TOKEN (this oracle is token-major, the
        # device's stacked-Q rows are token r//G) that is:
        #   prefix cols : 0 for col < valid_len, -inf beyond -- FAULT 8 moved the device
        #                 off the nominal PREFIX_LEN onto the data-dependent valid
        #                 length, and both biases are re-DMA'd per inference.
        #   suffix cols : causal (fault 6) + pad-column masking.
        bias_self = torch.zeros(M, self.PREFILL_MAX_SEQ_LEN)
        bias_self[:, vlen_p:] = float("-inf")
        suf_b = torch.zeros(M, M)
        suf_b[:, chunk:] = float("-inf")
        suf_b[torch.triu(torch.ones(M, M, dtype=torch.bool), diagonal=1)] = float("-inf")
        bias_self = torch.cat([bias_self, suf_b], dim=1)           # [M, PM+M]
        PMp = self.PREFILL_MAX_SEQ_LEN
        bias_cross = torch.zeros(M, PMp)
        bias_cross[:, vlen_p:] = float("-inf")

        def one_layer(hh, li, capture=None):
            g_ = lambda nm: w(f"ae.{li}.{nm}")
            n1 = rms_norm(hh[:, :HR], g_("input_layernorm.weight"), eps)
            q = n1 @ g_("q_proj.weight").T
            is_self = self._ae_is_self_attn(li)
            # FAULT 7: SELF ropes Q at the prefix-continued positions, CROSS at the
            # rebased 0..M-1 ones (device: AE_ROPE_GQA_DRAM vs AE_ROPE_GQA_CROSS_DRAM).
            qh = apply_rope(q.view(M, NH, D).transpose(0, 1),
                            *( (cos_t, sin_t) if is_self else (cos_x, sin_x) ))
            if is_self:
                k = n1 @ g_("k_proj.weight").T
                v = n1 @ g_("v_proj.weight").T
                kh = apply_rope(k.view(M, NKV, D).transpose(0, 1), cos_t, sin_t)
                vh = v.view(M, NKV, D).transpose(0, 1)
                # FAULT 4: prepend the RAW cached prefix K/V, exactly as
                # forward_attn_layer's torch.cat does and as _emit_expert_layer now
                # stages into AE_CK_DRAM/AE_CV_DRAM. Raw, not reprojected -- the
                # reprojection is a cross-layer behaviour.
                ppl = self._ae_cross_prefix_layer(li)
                ppk, ppv = self._bisect_prefix_kv[ppl]
                kh = torch.cat([ppk, kh], dim=1)
                vh = torch.cat([ppv, vh], dim=1)
            else:
                pl = self._ae_cross_prefix_layer(li)
                pk, pv = self._bisect_prefix_kv[pl]
                # FAULT 5: the device re-projects the cached K/V through THIS layer's
                # k_proj/v_proj (which are (320,320) here, not (320,480)) before flash,
                # so this oracle must too or it scores the device against the wrong math.
                S = pk.shape[1]
                kf = pk.transpose(0, 1).reshape(S, NKV * D)          # [S, 320]
                vf = pv.transpose(0, 1).reshape(S, NKV * D)
                kh = (kf @ g_("k_proj.weight").T).view(S, NKV, D).transpose(0, 1)
                vh = (vf @ g_("v_proj.weight").T).view(S, NKV, D).transpose(0, 1)
                # no RoPE on kh: the cache already carries the prefix rotation.
            a = attend(qh, kh, vh, bias_self if is_self else bias_cross)
            o = a @ g_("o_proj.weight").T
            r1 = hh[:, :HR] + o
            n2 = rms_norm(r1, g_("post_attention_layernorm.weight"), eps)
            gg = F.silu(n2 @ g_("gate_proj.weight").T)
            uu = n2 @ g_("up_proj.weight").T
            dn = (gg * uu) @ g_("down_proj.weight").T
            out = r1 + dn
            if capture is not None:
                capture.update(norm1=n1, q_proj=q, attn=a, o_proj=o, resid1=r1,
                               norm2=n2, gate=gg, up=uu, down=dn, hidden=out,
                               is_self=is_self)
            return out

        with torch.no_grad():
            hh = h[:, :HR]
            for li in range(tgt):
                hh = one_layer(torch.cat([hh, torch.zeros(M, HP - HR)], 1), li)
            r = {}
            one_layer(torch.cat([hh, torch.zeros(M, HP - HR)], 1), tgt, capture=r)

        h_out = self.AE_IO_B_DRAM if tgt % 2 == 0 else self.AE_IO_A_DRAM
        cut = lambda t: t[:, :HR]
        hw = {
            "norm1":  cut(self._read_bf16(self.AE_P_NORM1_DRAM, (M, HP), label="n1")),
            "q_proj": self._read_bf16(self.AE_Q_DRAM, (M, Q), label="q"),
            "attn":   self._read_bf16(self.AE_ATTN_RESULT_DRAM, (M, Q), label="attn"),
            "o_proj": cut(self._read_bf16(self.AE_O_PROJ_DRAM, (M, HP), label="o")),
            "resid1": cut(self._read_bf16(self.AE_RESIDUAL_DRAM, (M, HP), label="r1")),
            "norm2":  cut(self._read_bf16(self.AE_P_NORM2_DRAM, (M, HP), label="n2")),
            "gate":   self._read_bf16(self.AE_MLP_GATE_DRAM, (M, I), label="gate"),
            "up":     self._read_bf16(self.AE_MLP_UP_DRAM, (M, I), label="up"),
            "down":   cut(self._read_bf16(self.AE_MLP_DOWN_DRAM, (M, HP), label="down")),
            "hidden": cut(self._read_bf16(h_out, (M, HP), label="hidden")),
        }
        # SCORE ONLY THE 50 REAL ACTION ROWS. The 14 pad rows are computed on both
        # sides but nobody reads them, and they hold unrelated garbage -- including them
        # is the pi05 trap (same op: -3.4 dB with pad rows, +50 dB without).
        rows = torch.zeros(M, dtype=torch.bool)
        rows[: self.CHUNK] = True
        print(f"\n=== EXPERT LAYER-{tgt} OP BISECT "
              f"({'SELF' if r['is_self'] else 'CROSS'}-attn, {n_ae} layers) ===")
        first_bad = None
        for name in ("norm1", "q_proj", "attn", "o_proj", "resid1", "norm2",
                     "gate", "up", "down", "hidden"):
            a, b = hw[name], r[name]
            s, c = snr_db(a, b, rows), cos_sim(a, b, rows)
            flag = ""
            if c < COS_FLOOR and first_bad is None:
                first_bad, flag = name, "   <== FIRST DIVERGENCE"
            elif c < COS_FLOOR:
                flag = "   (inherited)"
            print(f"  {name:7s} {str(tuple(a.shape)):11s} snr={s:8.2f}dB cos={c:.6f} "
                  f"rms={rms(a, rows):8.4f}/{rms(b, rows):8.4f}{flag}")
        print(f"\n  {'FIRST DIVERGENCE AT: ' + first_bad if first_bad else 'layer clean'}")
        return first_bad

    def _expert_step_snr_check(self):
        """Gate the action expert + 10-step denoise against the torch oracle.

        ISOLATION: the oracle is fed the HARDWARE'S OWN prefix KV cache and the
        HARDWARE'S OWN initial noise, read back from DRAM. So this measures the expert
        alone -- 32 layers x 10 Euler steps -- rather than inheriting the prefix's error.
        Without that, a prefix at 20 dB would make a perfect expert look broken.

        Scored on the ACTION CHUNK, which is the only output that matters. Per the
        gemma3 lesson, cos-sim is the primary signal for a chain this long: 10 sequential
        Euler steps accumulate drift that max-abs error exaggerates, and a policy whose
        actions are directionally right is a working policy.

        Threshold is SNR_FLOOR["actions"], the measured bf16 ceiling for the chain -- not
        a flat 40 dB, which 320 layer bodies cannot hold."""
        dn = getattr(self, "_last_denoise", None)
        assert dn is not None, "_expert_step_snr_check must run after run_denoise"
        PM, D, NKV = self.PREFILL_MAX_SEQ_LEN, self.HEAD_DIM, self.NUM_KV_HEADS
        chunk, adim = self.CHUNK, self.ACTION_PAD

        print("  expert gate (oracle fed the HW's OWN prefix KV + the HW's own noise):")
        # Pull the whole KV cache back so the oracle cross-attends into exactly what the
        # device did. 32 layers x 2 x 5 heads x [192,64] = 7.2 MB of reads; slow but this
        # is the only way to separate expert error from prefix error.
        kv = []
        for li in range(self.NUM_LAYERS):
            k_heads, v_heads = [], []
            for h in range(NKV):
                off = li * self.KV_LAYER_STRIDE + h * self.KV_HEAD_STRIDE
                k_heads.append(self._read_bf16(self.LAYER0_K_DRAM + off, (PM, D),
                                               label=f"kv_k_l{li}h{h}"))
                v_heads.append(self._read_bf16(self.LAYER0_V_DRAM + off, (PM, D),
                                               label=f"kv_v_l{li}h{h}"))
            kv.append((torch.stack(k_heads), torch.stack(v_heads)))

        # The HW seeded x_t as [64,64] with the real noise in [:50,:7] and 1e-6 padding;
        # the oracle's denoise wants [chunk, max_action_dim]. Slice the SAME values out
        # so both integrate identical trajectories -- a different seed here would be
        # comparing two valid but unrelated answers.
        x0 = dn["noise"][:chunk, :adim].float()
        ref = self._ref_oracle()
        valid_len = int(getattr(self, "_last_prefix", {}).get("valid_len",
                                                             self.PREFIX_LEN))
        with torch.no_grad():
            # PADDED WIDTHS: the KV read back from DRAM is PREFILL_MAX_SEQ_LEN wide and
            # the suffix runs SUFFIX_LEN_PAD rows, so the oracle needs the same
            # column masks the device applies. Omitting them let pad columns into the
            # softmax on the reference side only, compounding across 10 Euler steps.
            r_actions = ref.denoise(kv, valid_len, noise=x0,
                                    pad_to=self.SUFFIX_LEN_PAD,
                                    prefix_pad_to=self.PREFILL_MAX_SEQ_LEN)

        # ---- PER-STEP EULER CURVE (pi05's AE_STEP_SNAPSHOT_DRAM) ------------------
        # Same idea that cracked the prefix: one number at the end cannot distinguish
        # "step 3 broke" from "every step drifts a little". The oracle is re-integrated
        # step by step from the SAME KV and the SAME noise, so each row is honest.
        M, ADP = self.SUFFIX_LEN_PAD, self.ACTION_DIM_PAD
        if hasattr(self, "AE_STEP_SNAP_DRAM"):
            with torch.no_grad():
                _, ref_steps = ref.denoise(kv, valid_len, noise=x0, trace=True,
                                           pad_to=self.SUFFIX_LEN_PAD,
                                           prefix_pad_to=self.PREFILL_MAX_SEQ_LEN)
            print("    per-Euler-step x_t curve:")
            prev = None
            for si in range(self.N_STEPS):
                snap = self._read_bf16(self.AE_STEP_SNAP_DRAM + si * M * ADP * 2,
                                       (M, ADP), label=f"step{si}")[:chunk, :adim]
                rs = ref_steps[si][:, :adim]
                m = torch.ones(chunk, dtype=torch.bool)
                s, c = snr_db(snap, rs, m), cos_sim(snap, rs, m)
                d = "" if prev is None else f"  d={s - prev:+6.2f}dB"
                prev = s
                print(f"      step{si:2d} t={1.0 - si / self.N_STEPS:4.2f} "
                      f"snr={s:7.2f}dB cos={c:.6f} rms={rms(snap, m):.4f}/{rms(rs, m):.4f}{d}")
            print("      a CLIFF at one step => that step's emission; UNIFORM decay => "
                  "integration drift")

        hw_padded = dn["x_t_padded"][:chunk, :adim]
        rows = torch.ones(chunk, dtype=torch.bool)
        ok = report("denoise actions [50,32]", hw_padded, r_actions, rows,
                    threshold=SNR_FLOOR["actions"])
        c = cos_sim(hw_padded, r_actions, rows)
        if c < COS_FLOOR:
            print(f"       cos {c:.6f} < {COS_FLOOR} -- STRUCTURAL, not precision. "
                  f"Suspect the rolled loop (did it repeat?), the cross-attn layer "
                  f"mapping, or the 480->512 pad lanes.")
            ok = False

        # The real deliverable: the executed window, real DoF only.
        n_exec, ad = self.N_ACTION_STEPS, self.ACTION_DIM
        report(f"executed actions [{n_exec},{ad}]", hw_padded[:n_exec, :ad],
               r_actions[:n_exec, :ad], torch.ones(n_exec, dtype=torch.bool),
               threshold=SNR_FLOOR["actions"])
        self._diag("actions", hw_padded[:n_exec, :ad], r_actions[:n_exec, :ad])
        return ok

    def _expert_pad_lane_check(self):
        """Assert expert activation lanes [480:512] are exactly zero after every op --
        the 480->512 pad is only sound while they stay zero."""
        raise NotImplementedError

    def dump_prefix_kv(self, path):
        """Save the 32-layer prefix K/V so the torch reference can reproduce the expert
        exactly (pi05's debug_prefix_kv.npz pattern)."""
        raise NotImplementedError

    # ==================================================================================
    # 8. bins
    # ==================================================================================

    def precompile_all(self):
        raise NotImplementedError

    def dump_bins(self, bin_dir):
        """params.bin + programs.bin + programs.json. Carry a signature (num image slots,
        prefix_len, chunk, quant mode, rope theta) and refuse to load on mismatch.
        rm the bin dir after any compile-affecting edit -- stale bins reload silently."""
        raise NotImplementedError


# ======================================================================================
# run-from-bin variant
# ======================================================================================

class VeraPulse_Run(VeraPulse_UnifiedEngine):
    """Load params/programs from bins instead of compiling.

    __init__ MUST mirror the full 5-surface clear that survives software_reset():
      1. clear_on_chip_sram()            URAM A/B lower halves + scale/bias BRAM
      2. manual 2-instruction program    URAM A/B upper halves
      3. clear_argmax_and_pbi_regs()     PBI pointer table rows 1..15
      4. mini-program of add_set(reg,0)  ISA regs 1..15
      5. reset_program_dram_addr()       so bins land at DRAM_INSTRUCTION_ADDR
    Skipping any of these lets a previously-run model contaminate results."""

    def __init__(self, bin_dir=None, **kw):
        raise NotImplementedError

    def weight_init(self, dummy=False, seed=0):
        """No-op: params come from params.bin. Signature mirrors the base class so the
        same caller (main) drives either engine without knowing which it holds."""
        raise NotImplementedError

    def load_params(self):
        raise NotImplementedError

    def load_programs(self):
        raise NotImplementedError

    def _sig_check(self, name, live, want):
        raise NotImplementedError


# ======================================================================================
# data
# ======================================================================================

def load_sample_observation():
    """[2,512,512,3] uint8 images ('image' + 'image2' upstream) + state[32] + prompt.

    There is no sample observation shipped with this checkpoint (the HF repo carries
    config/model/norm_stats/tokenizer only), so this stays NotImplementedError on
    purpose: main() catches it and falls back to deterministic synthetic cameras, and
    a fabricated "sample" here would look like real robot data in every log it touched.
    The real observations come from the LIBERO simulator via libero_eval.py."""
    raise NotImplementedError


_TOKENIZER = None


def tokenize(prompt, max_len=None, return_mask=False):
    """LIBERO task string -> [max_len] int64 token ids, RIGHT-padded.

    Uses `tokenizers` (the checkpoint's own tokenizer.json) rather than transformers:
    the only thing needed is a byte-level BPE encode, and transformers would drag a
    version-sensitive AutoProcessor into a hardware bring-up that does not need one.

    Two details are load-bearing and both come from the SmolVLA input pipeline this
    checkpoint was trained with:

      * the task string is lowercased/stripped and gets a TRAILING NEWLINE. lerobot's
        SmolVLA policy does exactly `task if task.endswith("\\n") else task + "\\n"`
        before tokenizing, so omitting it shifts every token id the model ever saw.
      * padding is on the RIGHT with pad id 2 (<|im_end|>, this tokenizer's pad token).
        RIGHT-padding is not cosmetic: embed_and_concat_prefix rotates row r with rope
        table entry r, which only agrees with the reference's positions = cumsum(mask)-1
        when every pad slot sits after every real one. Left-padding silently shifts
        every RoPE position -- the pi05 lesson.

    Returns ids by default so main()'s single-argument call keeps working; pass
    return_mask=True (libero_eval.py does) to also get the [max_len] bool mask of REAL
    slots. That mask is what build_attn_bias needs: the valid prefix length is
    1 + n_vision + n_real_text and is DATA-DEPENDENT -- it is not the constant 177,
    which would let 48-n_real pad rows attend as if they were language.
    """
    global _TOKENIZER
    from tokenizers import Tokenizer
    max_len = max_len or _CFG["lm"]["tokenizer_max_length"]
    if _TOKENIZER is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            _CFG["paths"]["tokenizer_file"])
        if not os.path.exists(path):
            ensure_checkpoint()          # tokenizer.json ships with the checkpoint
        _TOKENIZER = Tokenizer.from_file(path)

    # NO strip(), NO lower(). Upstream's Tokenizer.encode does exactly
    #     s = (t + "\n") if add_newline else t
    # and nothing else. Case-folding is not a harmless normalization here: it changes
    # byte-level BPE token ids, and every shifted id shifts the language block the model
    # was trained on.
    text = str(prompt)
    if not text.endswith("\n"):
        text += "\n"
    ids = _TOKENIZER.encode(text).ids[:max_len]     # truncate, never wrap
    n_real = len(ids)
    PAD_ID = 2                                      # <|im_end|>, SmolVLM2's pad token
    out = torch.full((max_len,), PAD_ID, dtype=torch.long)
    out[:n_real] = torch.as_tensor(ids, dtype=torch.long)
    if not return_mask:
        return out
    mask = torch.zeros(max_len, dtype=torch.bool)
    mask[:n_real] = True
    return out, mask


_NORM_STATS = None


def load_norm_stats():
    """norm_stats.safetensors -> {'state': (mean[8], std[8]), 'action': (mean[7], std[7])}.

    MEAN/STD, not pi05's q01/q99 quantiles -- this checkpoint is a lerobot-family model
    and its norm_stats file carries observation.state.mean/std and action.mean/std.
    Getting the transform family wrong (or its direction) does not crash: it produces a
    policy that moves smoothly in the wrong units, which reads as "the model is bad"
    rather than "the harness is wrong". So both directions live here, next to the data:

        state fed to the model :  (state - mean) / std
        action out of the model:  a * std + mean          <- the env needs this one

    Cached: the file is small but this is called once per inference otherwise."""
    global _NORM_STATS
    if _NORM_STATS is not None:
        return _NORM_STATS
    from safetensors.torch import load_file
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        _CFG["paths"]["norm_stats"])
    if not os.path.exists(path):
        ensure_checkpoint()
    raw = load_file(path)
    need = ["observation.state.mean", "observation.state.std",
            "action.mean", "action.std"]
    missing = [k for k in need if k not in raw]
    if missing:
        raise KeyError(
            f"norm_stats.safetensors is missing {missing} (has {sorted(raw)}) -- refusing "
            f"to guess a normalization; a wrong one looks like a broken policy, not a bug")
    _NORM_STATS = {
        "state": (raw["observation.state.mean"].float(),
                  raw["observation.state.std"].float()),
        "action": (raw["action.mean"].float(), raw["action.std"].float()),
    }
    return _NORM_STATS


# ======================================================================================
# main
# ======================================================================================

def _patch_upstream_quick_gelu(up):
    """Swap upstream's vision activation to the one the SILICON computes.

    The accelerator's fused activation is x*sigmoid(1.702x) (quick_gelu, confirmed at
    three hardware test sites in user_hw_test.py); the model specifies
    gelu_pytorch_tanh. That substitution alone costs 8.4 dB at the vision output, which
    swamps every other error and makes a gate against exact upstream far too blunt to
    catch a real fault -- a cos floor loose enough to admit the activation tax is loose
    enough to admit genuine bugs.

    Patching upstream's OWN module keeps the oracle honest: this is still the
    checkpoint's forward pass, not our reconstruction, so the failure mode that hid the
    activation substitution for months (bending the reference toward the hardware)
    cannot recur. GELU simply becomes common-mode and cancels, and the gate goes back to
    measuring arithmetic.
    """
    import types
    qg = lambda t: t * torch.sigmoid(1.702 * t)

    def fwd(self, x):
        return self.fc2(qg(self.fc1(x)))

    for layer in up.model.vlm_with_expert.vision.encoder.layers:
        layer.mlp.forward = types.MethodType(fwd, layer.mlp)
    return up


def _upstream_load(quiet=False, hw_gelu=False):
    """The checkpoint's OWN shipped forward pass, as the gate oracle.

    hw_gelu=True patches the vision activation to the accelerator's quick_gelu, giving
    a SHARP gate (GELU cancels). hw_gelu=False is the honest fidelity number: distance
    from the model as published. Report both -- they answer different questions."""
    import os as _os
    bundle = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                           "verapulse_bin", "verapulse__pulsevla-libero-0.5b")
    if not _os.path.isdir(bundle):
        raise FileNotFoundError(
            f"upstream bundle not found at {bundle} -- it ships alongside the weights; "
            f"the downloader must not filter the .py files out")
    sys.path.insert(0, bundle)
    from safetensors.torch import load_file
    from smolvla import SmolVLA, load_smolvla_config
    cfg_u = load_smolvla_config(_os.path.join(bundle, "config.json"))
    up = SmolVLA(cfg_u).float().eval()
    up.load_state_dict(load_file(_os.path.join(bundle, "model.safetensors")))
    if hw_gelu:
        _patch_upstream_quick_gelu(up)
    if not quiet:
        print(f"  upstream oracle loaded "
              f"({sum(p.numel() for p in up.parameters())/1e6:.1f}M"
              f"{', vision GELU -> quick_gelu' if hw_gelu else ''})")
    return up


_UP_CACHE = {}


def _upstream_cached(hw_gelu):
    """Loading 2.23 GB twice per gate is pure latency; the gates want both variants."""
    if hw_gelu not in _UP_CACHE:
        _UP_CACHE[hw_gelu] = _upstream_load(hw_gelu=hw_gelu)
    return _UP_CACHE[hw_gelu]


def _upstream_prefix_gate(ue, images, token_ids, text_mask, hidden, kv):
    """Score the DEVICE's prefix hidden state and KV cache against upstream.

    Runs upstream's OWN embed_prefix + interleaved forward with fill_kv_cache=True, so
    the prefix ordering, the sqrt(hidden) embedding scale and the block-causal mask all
    come from the checkpoint's code rather than from our reading of it.

    The KV cache matters more than the hidden state: the hidden state is thrown away,
    while the cache is the action expert's ONLY view of the observation.
    """
    import math as _m
    up = _upstream_load()
    vwe = up.model.vlm_with_expert
    m = torch.as_tensor(text_mask).reshape(-1).bool()
    ids = torch.as_tensor(token_ids, dtype=torch.long).reshape(-1)
    with torch.no_grad():
        imgs = [images[s].permute(2, 0, 1).unsqueeze(0).contiguous().float()
                for s in range(images.shape[0])]
        # state: the device already has it K-padded; rebuild the same vector here
        st = torch.as_tensor(ue._prefix_state_used, dtype=torch.float32).reshape(1, -1)
        embs, pad, att = up.model.embed_prefix(
            imgs, [torch.ones(1, dtype=torch.bool)] * len(imgs),
            ids[None, :], m[None, :], st)
        att2d = make_att_2d_masks_local(pad, att)
        pos = torch.cumsum(pad, dim=1) - 1
        outs, kvu = vwe.forward(attention_mask=att2d, position_ids=pos,
                                past_key_values=None, inputs_embeds=[embs, None],
                                use_cache=True, fill_kv_cache=True)
    up_hidden = outs[0][0]                      # [S, 960]
    valid = int(pad[0].sum())
    print("\n  == DEVICE vs UPSTREAM: prefix ==")
    print(f"  upstream prefix rows={up_hidden.shape[0]} valid={valid}; "
          f"device valid_len={ue._prefix_valid_len}")
    if up_hidden.shape[0] != ue._prefix_valid_len:
        print(f"  !! ROW COUNT DISAGREES -- upstream keeps the {ids.numel()-int(m.sum())} "
              f"padded text slots as real rows, the device packs them out. Compare only "
              f"the valid rows.")
    n = min(valid, int(ue._prefix_valid_len))
    hw = torch.as_tensor(hidden).float()[:n]
    report("prefix hidden", hw, up_hidden[:n], threshold=20.0)
    # KV: upstream stores [B, S, n_kv, D]; device gives [n_kv, S, D] per layer
    worst = 1.0
    for li in (0, 1, 15, 31):
        ku = kvu[li]["key_states"][0][:n].permute(1, 0, 2)     # [5, n, 64]
        vu = kvu[li]["value_states"][0][:n].permute(1, 0, 2)
        kh, vh = torch.as_tensor(kv[li][0])[:, :n], torch.as_tensor(kv[li][1])[:, :n]
        ck, cv = cos_sim(kh, ku), cos_sim(vh, vu)
        worst = min(worst, ck, cv)
        print(f"    L{li:<2} K cos={ck:.6f}  V cos={cv:.6f}")
    print(f"  worst KV cos across sampled layers: {worst:.6f}")
    return worst


def _upstream_denoise_gate(ue, images, token_ids, text_mask, state, noise, hw_actions,
                           device_kv=None):
    """Score the DEVICE's denoised action chunk against upstream's sample_actions.

    Both sides integrate the SAME noise, so this is a deterministic comparison, not a
    sampling one -- any difference is the model, not the flow.

    EXPECT THIS TO FAIL until the two expert faults are fixed. Upstream's expert
    (smolvla/modeling.py):
      * EVEN (self-attn) layers concatenate the cached prefix K/V onto the suffix's own
        K/V, so the expert attends over prefix_len+50 keys. The device attends over 50.
      * ODD (cross-attn) layers REPROJECT the cached VLM K/V through the expert's own
        k_proj/v_proj -- which the checkpoint proves are (320,320) on odd layers versus
        (320,480) on even ones. The device feeds the raw cache straight in.
    The number here is the size of that gap, and it is the thing that has to move.
    """
    up = _upstream_cached(True)          # sharp variant: GELU cancels
    m = torch.as_tensor(text_mask).reshape(-1).bool()
    ids = torch.as_tensor(token_ids, dtype=torch.long).reshape(-1)
    st = torch.as_tensor(state, dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        imgs = [images[s].permute(2, 0, 1).unsqueeze(0).contiguous().float()
                for s in range(images.shape[0])]
        up_act = up.model.sample_actions(
            imgs, [torch.ones(1, dtype=torch.bool)] * len(imgs),
            ids[None, :], m[None, :], st,
            torch.as_tensor(noise, dtype=torch.float32)[None])[0]
    hw = torch.as_tensor(hw_actions).float()
    n = min(hw.shape[0], up_act.shape[0])
    d = min(hw.shape[1], up_act.shape[1])
    print("\n  == DEVICE vs UPSTREAM: denoise (same noise, deterministic) ==")
    # NOT a row mismatch: both sides are `chunk_size` rows. The column counts differ
    # because the model's action space is max_action_dim=32 of which only action_dim=7
    # are real DoF -- the other 25 are the multi-embodiment zero padding, which upstream
    # itself slices off before computing loss. The device already returns the 7.
    print(f"  device {tuple(hw.shape)}  upstream {tuple(up_act.shape)} "
          f"-> comparing {n} rows x {d} real DoF "
          f"(upstream cols {d}..{up_act.shape[1]-1} are action-dim padding)")
    report("actions vs upstream", hw[:n, :d], up_act[:n, :d], threshold=30.0)
    c = cos_sim(hw[:n, :d], up_act[:n, :d])
    print(f"  actions cos {c:.6f}")

    # THE DEVICE-MIRROR CHECK. "Expected to be bad" is not the same as "bad in the way we
    # predicted": a low upstream score is consistent BOTH with the three known model
    # faults AND with an additional device-side wiring bug sitting on top of them. This
    # scores the device against a reference deliberately configured to be just as wrong
    # (faults off, quick_gelu), which separates the two.
    if device_kv is not None:
        try:
            mirror = VeraPulseRef.from_checkpoint(hw_gelu=True).mirror_device_expert()
            # The cache is read back at the PADDED PM rows, but the reference works in
            # UNPADDED rows and sizes its own masks from prefix_len. Passing the padded
            # cache with the valid length made a [50,191] mask meet 242 keys. Slice here.
            vlen = int(ue._prefix_valid_len)
            device_kv = [(k[:, :vlen], v[:, :vlen]) for (k, v) in device_kv]
            with torch.no_grad():
                # Fed the DEVICE'S OWN KV cache, so vision and prefix differences cancel
                # and this isolates the EXPERT. Anything left is the expert alone.
                mr = mirror.denoise(device_kv, int(ue._prefix_valid_len),
                                    noise=torch.as_tensor(noise).float().clone())
            mrt = torch.as_tensor(mr).float()[:n, :d]
            report("actions vs device-mirror", hw[:n, :d], mrt, threshold=25.0)
            print("  ^ HIGH = the accelerator faithfully executes the wrong model, so "
                  "landing faults 4/5/6 in the emitter is sufficient.\n"
                  "    LOW  = a SECOND device-side fault on top; the emitter work alone "
                  "will not close it.")
        except Exception as _e:
            import traceback
            print(f"  (device-mirror check failed: {_e!r})")
            traceback.print_exc()
    print("  NOTE: expert faults 4,5,6,7,8 are ALL LANDED. This line is the SHARP gate "
          "(quick_gelu on both sides, so the activation substitution cancels) -- it "
          "grades arithmetic, not fidelity. The device-mirror line below is the ceiling: "
          "no model fix can push this above it.")
    return c


def make_att_2d_masks_local(pad_masks, att_masks):
    """Upstream's make_att_2d_masks, inlined so the gate does not depend on import order."""
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d & pad_2d


def _upstream_vision_gate(ue, images, tokens):
    """Score the DEVICE's vision+connector output against upstream's shipped forward.

    Returns True if the device is within the GELU-explained budget.

    The expected number here is NOT 40 dB and NOT 0.999. The accelerator's fused
    activation is x*sigmoid(1.702x) (quick_gelu) while the model specifies
    gelu_pytorch_tanh; measured on CPU with real weights, that substitution alone puts
    the 12-layer vision output at cos 0.9268 / 8.42 dB and the connector output at
    cos 0.9697 / 12.13 dB, with EXACT gelu scoring 113 dB. So a device reading near
    those numbers is behaving as designed. A device reading materially WORSE has a
    second fault that the reconstruction-based gate cannot see -- which is the entire
    reason this function exists.
    """
    hw = torch.as_tensor(tokens).float()
    print("\n  == DEVICE vs UPSTREAM: vision + connector ==")
    out = {}
    for hw_gelu in (False, True):
        try:
            up = _upstream_cached(hw_gelu)
        except FileNotFoundError as e:
            print(f"  --gate-upstream: {e}")
            return False
        vwe = up.model.vlm_with_expert
        # Feed upstream the IDENTICAL tensor the device consumed -- never re-derive it
        # from raw pixels, or a preprocessing difference gets blamed on the accelerator.
        with torch.no_grad():
            con = []
            for s in range(images.shape[0]):
                px = images[s].permute(2, 0, 1).unsqueeze(0).contiguous().float()
                con.append(vwe.connector(vwe.vision(px))[0])
            up_con = torch.cat(con, 0)
        assert hw.shape == up_con.shape, (
            f"device produced {tuple(hw.shape)}, upstream {tuple(up_con.shape)} -- a "
            f"shape disagreement is a layout bug, not a numerics one; do not score it")
        tag = "vs quick_gelu (SHARP)" if hw_gelu else "vs exact  (fidelity)"
        report(f"vision {tag}", hw, up_con, threshold=(30.0 if hw_gelu else 11.0))
        out[hw_gelu] = cos_sim(hw, up_con)
    print(f"  fidelity: cos {out[False]:.6f} -- distance from the model AS PUBLISHED; "
          f"floor set by quick_gelu, not by the accelerator")
    print(f"  sharp   : cos {out[True]:.6f} -- GELU cancels, so this one grades the "
          f"ARITHMETIC. Expect ~40 dB.")
    ok = out[True] >= 0.999 and out[False] >= 0.955
    print(f"  vision vs upstream: {'PASS' if ok else 'FAIL'}")
    if out[True] < 0.999:
        print("  the SHARP gate is what failed -- that is a real execution fault, not "
              "the activation tax. Bisect vision against upstream.")
    return ok


def main():
    ap = argparse.ArgumentParser()
    # This file is a HARDWARE bring-up entry point, so the device path is the default and
    # the host-only reference paths are the explicit opt-ins -- not the other way round.
    # The sentinel default (None) exists so --tiny, which is meaningless on hardware
    # (the device programs are compiled for the full 12/32/32-layer stacks), can select
    # the reference path when no --stage was typed, while an EXPLICIT --stage still wins.
    ap.add_argument("--stage", default=None,
                    choices=["ref", "prefix-only", "vision", "connector", "prefix",
                             "expert", "denoise", "all"],
                    help="DEVICE (runs on the FPGA): all (DEFAULT) = end-to-end "
                         "vision -> prefix -> denoise, prints the executed [10,7] action "
                         "slice; vision/connector = the vision-stage gate only. "
                         "HOST-ONLY (no FPGA): ref = torch reference (dummy weights unless "
                         "--real); prefix-only = the prefix in ISOLATION (synthetic vision "
                         "tokens, no ViT run) + a cross-check against transformers "
                         "LlamaModel; prefix = reference through the prefix, then a FORCED "
                         "STOP. expert/denoise are not implemented on device yet.")
    ap.add_argument("--bisect-expert", type=int, default=None, metavar="N",
                    help="compile N expert layers of ONE Euler step and walk layer N-1 "
                         "op by op, plus verify the 480->512 pad lanes. The per-step "
                         "curve puts the hardware 17 dB below the bf16 floor at step 0, "
                         "so one pass is defective -- start with --bisect-expert 1.")
    ap.add_argument("--bisect-prefix", type=int, default=None, metavar="N",
                    help="compile N prefix layers and walk layer N-1 op by op "
                         "(norm1/q/k/v/attn/o/resid1/norm2/gate/up/mult/down/hidden) "
                         "against a longhand host reference. The per-layer KV curve puts "
                         "the damage at L2, so --bisect-prefix 3 is the run that matters.")
    ap.add_argument("--bisect-vision", action="store_true",
                    help="walk ONE ViT layer op by op, scoring every intermediate "
                         "(patch/embed/ln1/q/k/v/attn/o/resid1/ln2/fc1/fc2/resid2) against "
                         "a step-by-step host reference. Forces --vis-layers 1 and "
                         "--stop-after vision. This is how you find WHICH op diverges.")
    ap.add_argument("--vis-layers", type=int, default=None,
                    help="compile only the first N ViT layers and truncate the oracle to "
                         "match. Bisection handle: the gate only sees post_ln, so this is "
                         "how you find WHICH layer diverges.")
    ap.add_argument("--strict-gates", action="store_true",
                    help="exit(1) on the first gate below its floor. Default is ADVISORY: "
                         "gates report and the run continues, because a deep bf16 stack "
                         "sits near its precision floor by construction and the actions "
                         "are what actually decide correctness.")
    ap.add_argument("--fused-silu", action=argparse.BooleanOptionalAction, default=True,
                    help="prefix MLP: fused LALU silu_enable (default) vs composed "
                         "silu_core_dram. --no-fused-silu tests whether the prefix's "
                         "per-layer error is the approximate activation.")
    ap.add_argument("--stop-after", default=None,
                    choices=["vision", "connector", "prefix", "denoise"],
                    help="hardware path: run up to and including this stage, gate it, "
                         "then stop. Use it to bring stages up one at a time instead of "
                         "debugging a failure through three stages of drift.")
    ap.add_argument("--weights", default="real", choices=["real", "dummy"],
                    help="DEVICE stages only: which weights weight_init stores. real "
                         "(default) = the downloaded checkpoint; dummy = fake_state_dict, "
                         "a shape-exact synthetic set for plumbing the full chain without "
                         "the 2.23 GB checkpoint. Dummy exercises every emitter and DMA "
                         "identically but is NOT a fidelity claim. Distinct from --real, "
                         "which selects real weights for the HOST reference stages; if "
                         "both are given, --weights governs the device and --real governs "
                         "the reference, and they never apply to the same stage.")
    ap.add_argument("--engines", type=int, default=1)
    ap.add_argument("--quant", default="bf16", choices=["bf16", "q4_64"])
    ap.add_argument("--gate-upstream", action=argparse.BooleanOptionalAction, default=True,
                    help="DEFAULT ON. Score the device against the checkpoint's OWN "
                         "shipped forward pass rather than VeraPulseRef. Gating against "
                         "our own reconstruction is what hid six model faults -- twice "
                         "the reference had been bent toward the hardware, so the gate "
                         "could only ever find hardware bugs, never model bugs. "
                         "--no-gate-upstream falls back to the reference gate.")
    ap.add_argument("--snr", action=argparse.BooleanOptionalAction, default=True,
                    help="per-stage >=40 dB gate; --no-snr to disable")
    ap.add_argument("--dump-bins", action="store_true")
    ap.add_argument("--from-bin", action="store_true")
    ap.add_argument("--download-only", action="store_true",
                    help="fetch the checkpoint and exit")
    ap.add_argument("--tiny", action="store_true",
                    help="reference stages: 2 layers per stack + 64 patches, fast")
    ap.add_argument("--real", action="store_true",
                    help="reference stages: load the REAL checkpoint instead of dummy "
                         "weights. Required for a trustworthy precision ceiling -- random "
                         "weights are worse-conditioned than trained ones.")
    ap.add_argument("--hw-gelu", action="store_true",
                    help="model the FPGA's x*sigmoid(1.702x) instead of gelu_tanh")
    ap.add_argument("--prefix-order", default=None,
                    choices=["state_images_text", "images_text_state", "scatter_into_text"])
    ap.add_argument("--images", default=None,
                    help="path to a .npy of [2,512,512,3] HWC camera images (uint8 or "
                         "already-normalized float); omitted -> deterministic synthetic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default=_CFG["defaults"]["prompt"])
    args = ap.parse_args()

    # Resolve the sentinel. Default = "all" (the full FPGA path). The one exception is
    # --tiny with no explicit --stage: --tiny means "2 layers per stack, 64 patches",
    # which only the host reference honours -- the device programs are compiled for the
    # real 12/32/32-layer stacks -- so silently sending it to hardware would run the FULL
    # model and quietly contradict the flag. An explicit --stage always wins.
    if args.stage is None:
        args.stage = "ref" if args.tiny else "all"
        if args.tiny:
            print("  (--tiny with no --stage: running the HOST reference, not the FPGA -- "
                  "layer truncation exists only in the torch oracle)")

    if args.download_only:
        print(ensure_checkpoint())
        return

    # --- prefix in isolation ------------------------------------------------------
    # Feeds the prefix SYNTHETIC vision tokens instead of running the 12-layer ViT twice,
    # so prefix numerics are isolated from any upstream drift and the run is seconds
    # rather than minutes. Also cross-checks the block math against transformers'
    # LlamaModel (SmolLM2 IS a Llama-architecture decoder), which is the only way to
    # answer "is our forward pass right" without hardware.
    if args.stage == "prefix-only":
        ch = OpenChoices()
        if args.prefix_order:
            ch.prefix_order = args.prefix_order
        L = _CFG["lm"]
        print(f"verapulse PREFIX ISOLATION | {'real' if args.real else 'dummy'} weights"
              f" | prefix_order={ch.prefix_order} | theta={L['rope_theta']:g}")
        ref = (VeraPulseRef.from_checkpoint(choices=ch) if args.real
               else VeraPulseRef.from_fake(tiny=args.tiny, seed=args.seed, choices=ch))
        if args.real and args.tiny:
            ref.n_lm = 2
        g = torch.Generator().manual_seed(args.seed)
        V, C = _CFG["vision"], _CFG["connector"]
        n_img = V["num_image_slots"] * C["tokens_out"]
        toks = torch.randn(n_img, L["hidden_size"], generator=g)
        state = torch.randn(_CFG["action_head"]["state_dim"], generator=g)
        ids = torch.randint(0, L["vocab_size"], (L["tokenizer_max_length"],), generator=g)

        def stat(name, t):
            print(f"  {name:18s} {str(tuple(t.shape)):20s} rms={t.pow(2).mean().sqrt():8.4f}"
                  f" absmax={t.abs().max():8.4f} finite={bool(torch.isfinite(t).all())}")

        x, valid, pos = ref.build_prefix(toks, ids, state)
        print(f"\n=== PREFIX  {ref.n_lm}L h={L['hidden_size']} "
              f"{L['num_heads']}q/{L['num_kv_heads']}kv x{L['head_dim']} GQA g"
              f"{L['num_heads'] // L['num_kv_heads']} ===")
        print(f"  layout           {x.shape[0]} rows = 1 state + {n_img} image + {len(ids)} text")
        stat("prefix_in", x)
        ptrace = []
        with PHASES.track("prefix fp32", "host"), torch.no_grad():
            hidden, kv = ref.forward_prefix(x, pos, trace=ptrace)
        for i, r, m, kr, vr in ptrace[:: max(1, len(ptrace) // 4)]:
            print(f"    layer {i:2d} resid rms={r:7.4f} absmax={m:7.4f} | "
                  f"k_rms={kr:6.4f} v_rms={vr:6.4f}")
        stat("prefix_hidden", hidden)
        mb = sum(k.numel() + v.numel() for k, v in kv) * 2 / 1e6
        print(f"  KV CACHE         {len(kv)} layers -> {mb:.1f} MB bf16 "
              f"(the expert's only prefix input)")

        # --- precision ceiling -----------------------------------------------------
        print("\n=== PRECISION (fp32 oracle vs bf16, the HW datatype) ===")
        allrows = lambda t: torch.ones(t.shape[0], dtype=torch.bool)
        refb = ref.astype(torch.bfloat16)
        with PHASES.track("prefix bf16", "host"), torch.no_grad():
            xb, _v, posb = refb.build_prefix(toks.to(torch.bfloat16), ids,
                                             state.to(torch.bfloat16))
            hb, kvb = refb.forward_prefix(xb, posb)
        report("prefix hidden", hb.float(), hidden, valid)
        report("KV layer0 k", kvb[0][0].float().flatten(0, 1), kv[0][0].flatten(0, 1),
               allrows(kvb[0][0].flatten(0, 1)))
        report("KV layer-1 k", kvb[-1][0].float().flatten(0, 1), kv[-1][0].flatten(0, 1),
               allrows(kvb[-1][0].flatten(0, 1)))

        # --- forward-pass correctness vs an INDEPENDENT implementation --------------
        print("\n=== FORWARD-PASS CHECK vs transformers LlamaModel ===")
        print("  SmolLM2 is a Llama-architecture decoder, so the same weights can be run")
        print("  through transformers' own block. CAUSAL on both sides -- the only thing")
        print("  under test is RMSNorm + RoPE + GQA + gated SiLU. Expect >100 dB.")
        try:
            from transformers.models.llama.modeling_llama import LlamaModel
            from transformers.models.llama.configuration_llama import LlamaConfig
            NL = min(4, ref.n_lm)
            hf_cfg = LlamaConfig(
                hidden_size=L["hidden_size"], intermediate_size=L["intermediate_size"],
                num_hidden_layers=NL, num_attention_heads=L["num_heads"],
                num_key_value_heads=L["num_kv_heads"], head_dim=L["head_dim"],
                vocab_size=L["vocab_size"], rms_norm_eps=L["rms_norm_eps"],
                rope_theta=L["rope_theta"], attention_bias=False, hidden_act="silu",
                max_position_embeddings=4096, tie_word_embeddings=False)
            with silenced():
                hf = LlamaModel(hf_cfg).eval()
            w = {"embed_tokens.weight": ref.w("lm.embed_tokens.weight"),
                 "norm.weight": ref.w("lm.final_norm.weight")}
            for i in range(NL):
                for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    w[f"layers.{i}.self_attn.{p}.weight"] = ref.w(f"lm.{i}.{p}.weight")
                for p in ("gate_proj", "up_proj", "down_proj"):
                    w[f"layers.{i}.mlp.{p}.weight"] = ref.w(f"lm.{i}.{p}.weight")
                w[f"layers.{i}.input_layernorm.weight"] = ref.w(f"lm.{i}.input_layernorm.weight")
                w[f"layers.{i}.post_attention_layernorm.weight"] = ref.w(
                    f"lm.{i}.post_attention_layernorm.weight")
            hf.load_state_dict(w, strict=False)
            S = x.shape[0]
            causal = torch.full((S, S), float("-inf")).triu(1)
            saved, ref.n_lm = ref.n_lm, NL
            with torch.no_grad():
                hf_out = hf(inputs_embeds=x.unsqueeze(0),
                            position_ids=pos.unsqueeze(0)).last_hidden_state[0]
                our_out, _ = ref.forward_prefix(x, pos, bias=causal)
            ref.n_lm = saved
            report(f"ours vs HF ({NL}L causal)", our_out, hf_out, valid, threshold=80.0)
        except Exception as e:
            print(f"  SKIPPED: {type(e).__name__}: {e}")

        PHASES.summary("prefix isolation timing (host torch)")
        return

    # --- reference stages (no hardware) ------------------------------------------
    # `ref` is the plumbing smoke test; `prefix` runs the oracle through the prefix and
    # then HARD STOPS. Neither touches the FPGA.
    if args.stage in ("ref", "prefix"):
        ch = OpenChoices()
        if args.prefix_order:
            ch.prefix_order = args.prefix_order
        V, L, C = _CFG["vision"], _CFG["lm"], _CFG["connector"]
        print(f"verapulse torch reference | stage={args.stage}"
              f"{' | tiny' if args.tiny else ''} | prefix_order={ch.prefix_order}")
        if args.real:
            print("  loading real checkpoint (2.23 GB fp32)...")
            ref = VeraPulseRef.from_checkpoint(choices=ch, hw_gelu=args.hw_gelu)
            ref.n_vis = 2 if args.tiny else _CFG["vision"]["num_layers"]
            ref.n_lm = 2 if args.tiny else _CFG["lm"]["num_layers"]
            ref.n_ae = 2 if args.tiny else _CFG["expert"]["num_layers"]
        else:
            ref = VeraPulseRef.from_fake(tiny=args.tiny, seed=args.seed, choices=ch,
                                         hw_gelu=args.hw_gelu)
        patches, state, ids, noise = fake_inputs(_CFG, tiny=args.tiny, seed=args.seed)

        def stat(name, t, note=""):
            f = bool(torch.isfinite(t).all())
            print(f"  {name:18s} {str(tuple(t.shape)):20s} "
                  f"rms={t.pow(2).mean().sqrt():8.4f} absmax={t.abs().max():8.4f} "
                  f"finite={f}{'  ' + note if note else ''}")
            if not f:
                raise SystemExit(f"NON-FINITE at {name} -- stop here, nothing downstream "
                                 f"of this is meaningful")

        # Every section below scores fp32 (the oracle) against bf16 (the HW datatype) on
        # the SAME inputs, and prints all three metrics. Three because each is blind to
        # something: SNR conflates gain with noise, cos-sim cannot see a uniform scale
        # factor at all (cos(10x,x) == 1.000000 exactly), and RMS cannot see structure.
        refb = ref.astype(torch.bfloat16)
        allrows = lambda t: torch.ones(t.shape[0], dtype=torch.bool)

        # ================= 1. VISION =================================================
        print(f"\n=== 1. VISION  SigLIP {ref.n_vis}L  h={V['hidden_size']} "
              f"{V['num_heads']}x{V['head_dim']} MHA  seq={patches.shape[1]}  "
              f"slots={patches.shape[0]} ===")
        vis, vtrace = [], []
        with PHASES.track("vision fp32", "host"), torch.no_grad():
            for s in range(patches.shape[0]):
                tr = []
                vis.append(ref.forward_vision(patches[s], trace=tr))
                vtrace.append(tr)
                stat(f"slot{s} post_ln", vis[-1])
        with PHASES.track("vision bf16", "host"), torch.no_grad():
            visb = [refb.forward_vision(p.to(torch.bfloat16)) for p in patches]
        for s in range(len(vis)):
            report(f"slot{s} post_ln bf16", visb[s].float(), vis[s], allrows(vis[s]))
        for i, r, m in vtrace[0][:: max(1, len(vtrace[0]) // 4)]:
            print(f"    layer {i:2d} residual  rms={r:8.4f}  absmax={m:8.4f}")
        print(f"    layer {vtrace[0][-1][0]:2d} residual  rms={vtrace[0][-1][1]:8.4f}"
              f"  absmax={vtrace[0][-1][2]:8.4f}   <- last")
        d = abs(vtrace[0][-1][1] - vtrace[0][0][1]) / max(vtrace[0][0][1], 1e-9)
        print(f"    residual drift first->last: {d * 100:5.1f}%  "
              f"({'smooth' if d < 10 else 'CHECK: large'})")

        # ================= 2. CONNECTOR ==============================================
        print(f"\n=== 2. CONNECTOR  pixel-shuffle x{C['pixel_shuffle_scale_factor']}  "
              f"{V['num_patches']}x{V['hidden_size']} -> "
              f"{C['tokens_out']}x{C['input_size']} -> {C['tokens_out']}x{C['output_size']} ===")
        with PHASES.track("connector", "host"), torch.no_grad():
            shuf = [pixel_shuffle(v, C["pixel_shuffle_scale_factor"]) for v in vis]
            per_slot = [ref.forward_connector(v) for v in vis]
        stat("shuffled", shuf[0], f"(slot0; {C['tokens_out']} tokens)")
        for s, t in enumerate(per_slot):
            stat(f"slot{s} tokens", t)
        toks = torch.cat(per_slot, 0)
        stat("all tokens", toks, f"({len(per_slot)} slots concatenated)")
        with torch.no_grad():
            toksb = torch.cat([refb.forward_connector(v) for v in visb], 0)
        report("connector bf16", toksb.float(), toks, allrows(toks))
        # the shuffle is a pure permutation: it must preserve the multiset of values
        a = torch.sort(vis[0].flatten()).values
        b = torch.sort(shuf[0].flatten()).values
        print(f"    permutation check: values preserved = {bool(torch.equal(a, b))}  "
              f"(a shuffle that drops/duplicates data fails here)")

        # ================= 3. PREFIX =================================================
        with torch.no_grad():
            x, valid, pos = ref.build_prefix(toks, ids, state)
        n_img = toks.shape[0]
        print(f"\n=== 3. PREFIX  SmolLM2 {ref.n_lm}L  h={L['hidden_size']} "
              f"{L['num_heads']}q/{L['num_kv_heads']}kv x{L['head_dim']} GQA g"
              f"{L['num_heads'] // L['num_kv_heads']}  theta={L['rope_theta']:g} ===")
        print(f"  layout ({ch.prefix_order}): {x.shape[0]} rows = "
              f"1 state + {n_img} image + {len(ids)} text")
        stat("prefix_in", x)
        print(f"  positions        first={pos[:6].tolist()} last={pos[-3:].tolist()}  "
              f"valid={int(valid.sum())}/{len(valid)}  "
              f"({'cumsum' if ch.rope_positions_cumsum else 'arange'})")
        ptrace = []
        with PHASES.track("prefix fp32", "host"), torch.no_grad():
            hidden, kv = ref.forward_prefix(x, pos, trace=ptrace)
        for i, r, m, kr, vr in ptrace[:: max(1, len(ptrace) // 4)]:
            print(f"    layer {i:2d} resid rms={r:7.4f} absmax={m:7.4f} | "
                  f"k_rms={kr:6.4f} v_rms={vr:6.4f}")
        li, lr, lm_, lkr, lvr = ptrace[-1]
        print(f"    layer {li:2d} resid rms={lr:7.4f} absmax={lm_:7.4f} | "
              f"k_rms={lkr:6.4f} v_rms={lvr:6.4f}   <- last")
        stat("prefix_hidden", hidden)
        k0, v0 = kv[0]
        mb = sum(k.numel() + v.numel() for k, v in kv) * 2 / 1e6
        print(f"  KV CACHE         {len(kv)} layers x k{tuple(k0.shape)} v{tuple(v0.shape)}"
              f"  = {mb:.1f} MB bf16")
        print(f"    this is the action expert's ONLY input from the prefix, and on HW it "
              f"must survive all {_CFG['action_head']['num_denoise_steps']} denoise steps")
        with PHASES.track("prefix bf16", "host"), torch.no_grad():
            xb, _vb, posb = refb.build_prefix(toksb, ids, state.to(torch.bfloat16))
            hiddenb, kvb = refb.forward_prefix(xb, posb)
        report("prefix hidden bf16", hiddenb.float(), hidden, valid)
        report("KV layer0 k bf16", kvb[0][0].float().flatten(0, 1),
               kv[0][0].flatten(0, 1), allrows(kvb[0][0].flatten(0, 1)))
        report("KV layer-1 k bf16", kvb[-1][0].float().flatten(0, 1),
               kv[-1][0].flatten(0, 1), allrows(kvb[-1][0].flatten(0, 1)))
        kr = [t[3] for t in ptrace]
        print(f"    k_rms across layers: min={min(kr):.4f} max={max(kr):.4f} "
              f"({'stable' if max(kr) / max(min(kr), 1e-9) < 50 else 'CHECK: wide spread'})")

        # ================= metric self-check =========================================
        print("\n=== METRIC SELF-CHECK (the gate itself, not the model) ===")
        print("  two positive controls then a NEGATIVE control: the last line feeds the")
        print("  gate deliberately-corrupted data and it MUST reject it. 'gate OK' there")
        print("  means the 40 dB threshold works; 'GATE BROKEN' would mean every later")
        print("  stage passes regardless of correctness.")
        report("identical", hidden, hidden)
        report("+0.3% noise", hidden + 0.003 * hidden.std() * torch.randn_like(hidden), hidden)
        report("+3% noise (must be rejected)",
               hidden + 0.03 * hidden.std() * torch.randn_like(hidden), hidden,
               expect_pass=False)

        # ================= 4. TIME EMBEDDING TABLE ===================================
        # The last thing before the action expert. Input-independent, so it is exact
        # constants -- but it is also the ONLY part of the suffix path that can be
        # verified without the expert existing, so verify it hard.
        HEADC = _CFG["action_head"]
        pad = _CFG["expert"]["hidden_size_padded"]
        dim = HEADC["time_embed"]["dim"]
        table, ts = build_time_table(_CFG, pad_to=pad)
        print(f"\n=== 4. TIME EMBEDDING TABLE  sincos dim={dim} -> stored {pad}  "
              f"N={HEADC['num_denoise_steps']} steps ===")
        stat("time_table", table.float(), f"({table.numel() * 2 / 1024:.1f} KB bf16)")
        print(f"  schedule         t = {ts[0]:.1f}, {ts[1]:.1f} ... {ts[-1]:.1f}  "
              f"dt = {-1.0 / HEADC['num_denoise_steps']:+.2f}")
        for i in (0, len(ts) // 2, len(ts) - 1):
            r = table[i].float()
            print(f"    step {i:2d} t={ts[i]:4.1f}  sin_half rms={r[:dim // 2].pow(2).mean().sqrt():.4f}"
                  f"  cos_half rms={r[dim // 2:dim].pow(2).mean().sqrt():.4f}"
                  f"  absmax={r.abs().max():.4f}")
        exact, _ = build_time_table(_CFG, pad_to=pad)
        report("stored bf16 vs exact", table.float(), exact.double(), allrows(table))
        padz = bool((table[:, dim:] == 0).all())
        inrange = bool(table[:, :dim].abs().max() <= 1.0)
        distinct = len({tuple(r.tolist()) for r in table}) == len(ts)
        print(f"    pad lanes [{dim}:{pad}] all zero : {padz}   "
              f"(the expert RMSNorm gamma fold depends on this)")
        print(f"    all |sin/cos| <= 1               : {inrange}")
        print(f"    all {len(ts)} step rows distinct        : {distinct}   "
              f"(identical rows would mean the schedule collapsed)")
        if not (padz and inrange and distinct):
            raise SystemExit("TIME TABLE INVARIANT VIOLATED -- fix before the expert")
        print(f"  NOTE only the RAW SINUSOID is precomputable. pi05 folded the whole")
        print(f"  sincos->MLP->silu chain into its table because its cond vector is")
        print(f"  input-independent; ours feeds action_time_mlp_in alongside")
        print(f"  action_in_proj(x_t), which changes every Euler step.")

        if args.stage == "prefix":
            # FORCED STOP. Everything past here -- the suffix embed, the action expert's
            # 32 layers, the 10-step denoise loop -- is unimplemented on HW AND rests on
            # the five unverified OpenChoices switches (layer parity, suffix rope base,
            # prefix row order). Delete this assert when the expert is gated >=40 dB.
            print("\n=== END OF PREFIX + TIME TABLE -- forced stop ===")
            assert False, ("FORCED STOP after the time embedding table: the suffix embed, "
                           "action expert and denoise loop beyond this point are neither "
                           "implemented nor verified. Remove this assert in main() when "
                           "the expert stage is gated at >=40 dB.")

        # ================= 5. ACTION EXPERT + 10-STEP DENOISE ========================
        E = _CFG["expert"]
        print(f"\n=== 5. ACTION EXPERT  SmolLM2-shaped {ref.n_ae}L  h={E['hidden_size']} "
              f"(->{E['hidden_size_padded']} padded)  {E['num_heads']}q/{E['num_kv_heads']}kv"
              f" x{E['head_dim']} GQA  self-attn every {E['self_attn_every_n_layers']} ===")
        n_self = sum(1 for i in range(ref.n_ae)
                     if (i % E["self_attn_every_n_layers"] == 0) == ch.self_attn_on_even)
        print(f"  layer split      {n_self} self-attn / {ref.n_ae - n_self} cross-attn"
              f"   (parity UNVERIFIED: self_attn_on_even={ch.self_attn_on_even})")
        print(f"  per inference    {HEADC['num_denoise_steps']} steps x {ref.n_ae} layers"
              f" = {HEADC['num_denoise_steps'] * ref.n_ae} layer bodies")
        with PHASES.track("denoise fp32 (10 steps)", "host"), torch.no_grad():
            acts, steps = ref.denoise(kv, x.shape[0], noise=noise, trace=True)
        for i in (0, len(steps) // 2, len(steps) - 1):
            s = steps[i]
            print(f"    step {i:2d} t={ts[i]:4.1f} -> x_t rms={s.pow(2).mean().sqrt():7.4f}"
                  f" absmax={s.abs().max():7.4f}")
        drift = [float((steps[i + 1] - steps[i]).pow(2).mean().sqrt())
                 for i in range(len(steps) - 1)]
        print(f"    per-step delta rms: first={drift[0]:.4f} last={drift[-1]:.4f}"
              f"  ({'converging' if drift[-1] <= drift[0] else 'CHECK: diverging'})")
        stat("actions_padded", acts)
        stat("actions", acts[:, : HEADC["action_dim"]],
             f"(first {HEADC['action_dim']} dims are the real DoF)")

        with PHASES.track("denoise bf16 (10 steps)", "host"), torch.no_grad():
            actsb = refb.denoise(kvb, xb.shape[0], noise=noise.to(torch.bfloat16))
        report("actions bf16 (10-step)", actsb.float(), acts, allrows(acts))
        print("  cos-sim is the honest metric for the denoise chain: 10 sequential bf16")
        print("  Euler steps accumulate drift that max-abs error exaggerates.")

        # ================= MODEL OUTPUT =============================================
        # What the robot actually receives. The expert predicts a chunk of `chunk_size`
        # (50) actions in a 32-wide padded space, but only the first `n_action_steps`
        # (10) are executed before the policy re-plans, and only `action_dim` (7) of the
        # 32 columns are real DoF -- the rest is zero padding to reach a %64-friendly
        # width. So [50,32] -> [10,7] is the real deliverable.
        n_exec, adim = HEADC["n_action_steps"], HEADC["action_dim"]
        executed = acts[:n_exec, :adim]
        executedb = actsb[:n_exec, :adim].float()
        print(f"\n=== MODEL OUTPUT  [{n_exec}, {adim}]  "
              f"(executed slice of the [{HEADC['chunk_size']}, {HEADC['max_action_dim']}] chunk) ===")
        stat("actions_executed", executed)
        report("executed bf16", executedb, executed, allrows(executed))
        print(f"  {'step':>4}  " + "".join(f"{'dof' + str(d):>9}" for d in range(adim)))
        for i in range(n_exec):
            print(f"  {i:>4}  " + "".join(f"{float(v):9.4f}" for v in executed[i]))
        drop = HEADC["chunk_size"] - n_exec
        print(f"  ({drop} predicted actions discarded before re-plan; "
              f"{HEADC['max_action_dim'] - adim} padding columns dropped)")
        PHASES.summary("reference timing (host torch, not hardware)")
        return

    if args.stage not in ("vision", "connector", "all"):
        raise NotImplementedError(
            f"--stage {args.stage}: HW path is a skeleton -- only vision/connector and "
            f"the end-to-end 'all' path are implemented. Bring a stage up and gate it "
            f"before wiring it here.")
    if args.from_bin or args.dump_bins:
        raise NotImplementedError("bins are not implemented yet for the vision stage")
    if args.quant != "bf16":
        raise NotImplementedError("the vision tower is stored bf16 only")

    # --- observation -------------------------------------------------------------
    V = _CFG["vision"]
    slots, IMG, CH = V["num_image_slots"], V["image_size"], V["num_channels"]
    if args.images:
        import numpy as np
        images = torch.as_tensor(np.load(args.images))
    else:
        try:
            images = load_sample_observation()[0]
        except NotImplementedError:
            # Deterministic synthetic cameras. The gate scores HW against the oracle on
            # whatever pixels were actually fed, so image REALISM is irrelevant to
            # correctness -- but determinism is not: the run must be reproducible, and
            # the two slots must DIFFER, or a slot-crosstalk bug would score perfectly.
            g = torch.Generator().manual_seed(args.seed)
            images = torch.randint(0, 256, (slots, IMG, IMG, CH), generator=g,
                                   dtype=torch.uint8)
            print("  (no sample observation available -- using deterministic synthetic "
                  f"images, seed={args.seed})")
    images = torch.as_tensor(images)
    if images.dtype == torch.uint8:
        images = images.float() / 255.0          # the ViT consumes normalized pixels
    images = images.float()
    assert images.shape == (slots, IMG, IMG, CH), (
        f"expected [{slots},{IMG},{IMG},{CH}] HWC images, got {tuple(images.shape)}")

    # --- hardware ----------------------------------------------------------------
    print(f"verapulse HW | stage={args.stage} | weights={args.weights} | snr={args.snr}")
    ue = VeraPulse_UnifiedEngine()
    # weight_init LAST among engine constructions: every UnifiedEngine ctor DMA-writes
    # 16KB of noise to a hardcoded 0x80000000, which is this model's first stored weight.
    # SiLU variant must be set BEFORE compile_prefix runs (it is read at emit time).
    if args.bisect_vision:
        # One layer only: the probes live in buffers that layer 1 would overwrite.
        args.vis_layers = args.vis_layers or 1
        args.stop_after = "vision"
        ue.VIS_BISECT = True
    if args.bisect_expert is not None:
        ue.EXPERT_LAYERS = args.bisect_expert
        ue.EXPERT_BISECT = True
    if args.bisect_prefix is not None:
        ue.PREFIX_LAYERS = args.bisect_prefix
        ue.PREFIX_BISECT = True
        args.stop_after = "prefix"
    ue.VIS_LAYERS = args.vis_layers
    ue.PREFIX_FUSED_SILU = args.fused_silu
    ue.weight_init(dummy=(args.weights == "dummy"), seed=args.seed)
    ue.tensor_init()

    # ================= END-TO-END: vision -> prefix -> denoise ======================
    if args.stage == "all":
        L, HEADC = _CFG["lm"], _CFG["action_head"]
        g = torch.Generator().manual_seed(args.seed)

        # Tokens. tokenize() is still a stub, so there is no way to turn --prompt into
        # real ids yet. Deterministic in-range random ids keep the shapes and the RoPE
        # positions honest and the run reproducible, but the SEMANTICS of the prompt are
        # NOT in this run -- say so rather than let a plausible action table imply the
        # robot was told anything.
        text_mask = None
        try:
            token_ids, text_mask = tokenize(args.prompt, L["tokenizer_max_length"],
                                            return_mask=True)
        except NotImplementedError:
            token_ids = torch.randint(0, L["vocab_size"], (L["tokenizer_max_length"],),
                                      generator=g)
            print(f"  (tokenize() is a stub -- using deterministic RANDOM token ids, "
                  f"seed={args.seed}. The prompt {args.prompt!r} is NOT encoded in this "
                  f"run; the action values are plumbing output, not a policy decision.)")
        # State: seeded so two runs are comparable. Same caveat -- it is not a real robot
        # proprioception vector, and norm_stats normalization is not applied (stub).
        state = torch.randn(HEADC["state_dim"], generator=g)
        noise = torch.randn(HEADC["chunk_size"], HEADC["max_action_dim"], generator=g)

        if args.bisect_expert is not None:
            ue.tensor_init()
            toks = ue.run_vision(images)
            ue.run_prefix(toks, token_ids, state)
            # Give the reference the DEVICE's own prefix KV so cross-attn layers are
            # scored against what the hardware actually read, not an oracle re-derivation.
            PM, D, NKV = ue.PREFILL_MAX_SEQ_LEN, ue.HEAD_DIM, ue.NUM_KV_HEADS
            ue._bisect_prefix_kv = [
                (torch.stack([ue._read_bf16(ue.LAYER0_K_DRAM + li * ue.KV_LAYER_STRIDE
                                            + h * ue.KV_HEAD_STRIDE, (PM, D),
                                            label=f"bk{li}{h}") for h in range(NKV)]),
                 torch.stack([ue._read_bf16(ue.LAYER0_V_DRAM + li * ue.KV_LAYER_STRIDE
                                            + h * ue.KV_HEAD_STRIDE, (PM, D),
                                            label=f"bv{li}{h}") for h in range(NKV)]))
                for li in range(ue.NUM_LAYERS)]
            ue.run_denoise()
            ue.bisect_expert()
            PHASES.summary("hardware timing")
            return

        if args.bisect_prefix is not None:
            ue.tensor_init()
            toks = ue.run_vision(images)
            ue.run_prefix(toks, token_ids, state)
            ue.bisect_prefix()
            PHASES.summary("hardware timing")
            return

        if args.bisect_vision:
            ue.tensor_init()
            toks = ue.run_vision(images)
            ue.bisect_vision(images)
            PHASES.summary("hardware timing")
            return

        if args.gate_upstream:
            # vision -> prefix, then score BOTH against the checkpoint's own forward.
            ue.tensor_init()
            toks = ue.run_vision(images)
            ok_v = _upstream_vision_gate(ue, images, toks)
            tm = text_mask if text_mask is not None else torch.ones(
                token_ids.numel(), dtype=torch.bool)
            hidden = ue.run_prefix(toks, token_ids, state, text_mask=tm)
            PM, D, NKV = ue.PREFILL_MAX_SEQ_LEN, ue.HEAD_DIM, ue.NUM_KV_HEADS
            kv = [(torch.stack([ue._read_bf16(ue.LAYER0_K_DRAM + li * ue.KV_LAYER_STRIDE
                                              + h * ue.KV_HEAD_STRIDE, (PM, D),
                                              label=f"gk{li}{h}") for h in range(NKV)]),
                   torch.stack([ue._read_bf16(ue.LAYER0_V_DRAM + li * ue.KV_LAYER_STRIDE
                                              + h * ue.KV_HEAD_STRIDE, (PM, D),
                                              label=f"gv{li}{h}") for h in range(NKV)]))
                  for li in range(ue.NUM_LAYERS)]
            worst = _upstream_prefix_gate(ue, images, token_ids, tm, hidden, kv)
            acts = ue.run_denoise(noise=noise)
            c_act = _upstream_denoise_gate(ue, images, token_ids, tm, state, noise, acts,
                                           device_kv=kv)

            # Same action table the non-gated path prints, so this is a strict superset
            # of the old default behaviour and nothing was lost by turning the gate on.
            a = torch.as_tensor(acts).float()
            n_exec, adim = HEADC["n_action_steps"], HEADC["action_dim"]
            print(f"\n  MODEL OUTPUT -- first {n_exec} of {HEADC['chunk_size']} actions, "
                  f"{adim} dof (normalized):")
            for i in range(min(n_exec, a.shape[0])):
                print(f"  {i:>4}  " + "".join(f"{float(v):9.4f}" for v in a[i, :adim]))
            PHASES.summary("hardware timing")

            # EXIT CODE POLICY. Non-zero means "the accelerator computed something
            # wrong", which is the SHARP vision gate. The prefix and denoise numbers are
            # currently limited by KNOWN, catalogued model faults that live in the
            # emitter (causal suffix mask, (320,320) cross reprojection, prefix-K/V
            # concat) -- reporting those loudly is useful, failing the run over them
            # every time until they land is not.
            print(f"\n  gate summary: vision {'ok' if ok_v else 'FAULT'} | "
                  f"prefix worst-KV cos {worst:.4f} | actions cos {c_act:.4f}")
            if c_act < 0.99:
                print("  actions are limited by the 3 known expert faults (4/5/6), which "
                      "are fixed in VeraPulseRef but NOT in the emitter.")
            sys.exit(0 if ok_v else 1)

        executed = ue.run_inference(images, token_ids, state, noise=noise,
                                    snr=args.snr, stop_after=args.stop_after,
                                    strict_gates=args.strict_gates)
        if executed is None:            # --stop-after halted before the action head
            return

        # Printed in the SAME format as the host reference's MODEL OUTPUT block so the
        # two can be diffed line for line -- that comparison is the point of matching it.
        n_exec, adim = HEADC["n_action_steps"], HEADC["action_dim"]
        print(f"\n=== MODEL OUTPUT  [{n_exec}, {adim}]  "
              f"(executed slice of the [{HEADC['chunk_size']}, "
              f"{HEADC['max_action_dim']}] chunk) ===")
        print(f"  {'step':>4}  " + "".join(f"{'dof' + str(d):>9}" for d in range(adim)))
        for i in range(n_exec):
            print(f"  {i:>4}  " + "".join(f"{float(v):9.4f}" for v in executed[i]))
        drop = HEADC["chunk_size"] - n_exec
        print(f"  ({drop} predicted actions discarded before re-plan; "
              f"{HEADC['max_action_dim'] - adim} padding columns dropped)")

        PHASES.summary("hardware timing (compile is paid once, exec every inference)")
        return

    tokens = ue.run_vision(images)

    if args.gate_upstream:
        # THE ONLY GATE THAT DOES NOT ROUTE THROUGH OUR OWN RECONSTRUCTION.
        # Every other vision number here scores the device against VeraPulseRef, which
        # is built with hw_gelu=True -- i.e. bent to match the hardware. That is how an
        # 8.4 dB activation substitution scored 0.999 for months. This scores the device
        # against the checkpoint's OWN shipped forward pass, so nothing can hide in the
        # agreement between two things we wrote.
        ok = _upstream_vision_gate(ue, images, tokens)
        sys.exit(0 if ok else 1)

    if not args.snr:
        print("  (SNR gate disabled -- 'finite' is NOT correctness; a clamped Inf is a "
              "finite wrong number that ships)")
        return

    if args.stage == "connector":
        ok = ue._connector_snr_check(tokens)
    else:
        ok = ue._vision_snr_check(images, tokens)
    print(f"  vision stage: {'PASS' if ok else 'FAIL'} (threshold 40 dB)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
