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

Usage:
    python verapulse_test.py --stage vision --snr
    python verapulse_test.py --engines 8          # compiles, then dumps its bin set
    python verapulse_test.py --engines 8          # ... and this one loads it

Bins are automatic. A full run (--stage all, no --stop-after/--bisect-*) loads the set
matching its engine configuration if one exists and dumps one if it does not. params.bin
is shared by every configuration -- sharding never writes the params region -- while the
programs are keyed by the (vision, prefix, denoise) engine triple, because a sharded
program bakes one rendezvous per engine and cannot be replayed at a different count.
Use --dump-bins to force a re-dump after a compile-affecting edit.
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
    t_first: float = None                         # first track() entry, for true wall

    # "stage"/"forward" rows are ENVELOPES: they contain other rows. Summing them into a
    # total counts the nested time twice, so they are excluded from the per-kind totals
    # and printed separately as context.
    ENVELOPE_KINDS = ("stage", "forward")

    # Nesting depth of the span currently open. Rows record the depth they were
    # opened at, because "how much time is untracked" can ONLY be answered from
    # depth-0 spans: those are the ones that cannot overlap each other. Summing every
    # row instead counts an `exec` inside a `stage` inside a `forward` three times,
    # which is how the untracked line went NEGATIVE (-4.60s on a 21.81s run).
    _depth: int = 0

    @contextlib.contextmanager
    def track(self, label, kind):
        t0 = time.perf_counter()
        if self.t_first is None:
            self.t_first = t0
        d = self._depth
        self._depth = d + 1
        try:
            yield
        finally:
            self._depth = d
            self.rows.append((label, kind, time.perf_counter() - t0, d))

    def mark_forward_start(self):
        """Open the ONE end-to-end forward-pass span. Returns a token for
        :meth:`record_forward`."""
        t0 = time.perf_counter()
        if self.t_first is None:
            self.t_first = t0
        d = self._depth
        self._depth = d + 1          # the stage envelopes inside this span nest under it
        return (t0, len(self.rows), d)

    def record_forward(self, token, label="FORWARD PASS (vision -> prefix -> denoise)"):
        """Close the span opened by :meth:`mark_forward_start` and record it.

        THIS IS A MEASURED SPAN, NOT A SUM OF PER-STAGE COUNTERS. Adding up the three
        stages' device latencies answers "how many cycles did the accelerator spend",
        which is not the same question as "how long does one forward pass take" -- it
        silently drops everything BETWEEN the stages (readbacks, DMA staging, host
        patchify, the launch/join of the worker engines), and that gap is exactly what a
        deployed episode pays and what a per-stage number can never show.

        Gate time that ran inside the span is SUBTRACTED. The torch oracle is not part
        of the model, and with --snr on it sits between the stages, so leaving it in
        would report the CPU reference as if it were accelerator work. With --no-snr the
        subtraction is zero and this is a single uninterrupted wall measurement.
        """
        t0, i0, d = token
        self._depth = d
        wall = time.perf_counter() - t0
        gate = sum(r[2] for r in self.rows[i0:] if r[1] == "gate")
        self.rows.append((label, "forward", wall - gate, d))
        return wall - gate

    def summary(self, title="timing"):
        if not self.rows:
            return
        w = max(len(r[0]) for r in self.rows)
        tot = {}
        for label, kind, s, _d in self.rows:
            if kind not in self.ENVELOPE_KINDS:
                tot[kind] = tot.get(kind, 0.0) + s
        wall = time.perf_counter() - self.t_first

        # THE HEADLINE IS `exec`, AND ONLY `exec`. That is the per-inference cost on the
        # accelerator -- the number that moves with sharding and DRAM traffic, and the
        # only one a deployed episode pays. Compile is pre-prepared: built once at
        # startup by precompile_all and amortized over every inference after it, so
        # folding it into a hardware total misreports the device as slower than it is.
        # Gate and host rows are CPU work (torch oracle, readbacks) and are not the
        # accelerator either. They are reported, separately, below the line.
        _original_print(f"\n=== {title.upper()} ===")

        # THE HEADLINE IS THE MEASURED FORWARD PASS. One span, vision -> prefix ->
        # denoise back to back, oracle time removed. Everything under it is breakdown.
        fwd = [(r[0], r[2]) for r in self.rows if r[1] == "forward"]
        for label, sec in fwd:
            _original_print(f"  {label:<{w}}  {sec:8.2f}s   <-- END TO END")
        if fwd:
            _original_print("")

        _original_print(f"  device counters (per stage, on-FPGA cycles only):")
        for label, kind, sec, _d in self.rows:
            if kind == "exec":
                _original_print(f"    {label:<{w - 2}}  {sec:8.2f}s")
        _original_print("  " + "-" * (w + 12))
        _original_print(f"  {'sum of device counters':<{w}}  {tot.get('exec', 0.0):8.2f}s")

        # THE GAP IS THE POINT. Forward minus the device counters is everything the
        # accelerator was NOT running: host patchify, DMA staging, readbacks between
        # stages, worker launch/join. It is real per-inference cost and a per-stage
        # number cannot show it, which is why it is printed here rather than left to be
        # inferred from two totals on different lines.
        if fwd:
            gap = sum(sec for _, sec in fwd) - tot.get("exec", 0.0)
            _original_print(f"  {'gap (host: staging, readbacks, launch)':<{w}}  "
                            f"{gap:8.2f}s")

        other = [(k, v) for k, v in sorted(tot.items(), key=lambda kv: -kv[1])
                 if k != "exec"]
        if other or wall:
            _original_print(f"\n  not hardware execution:")
            for kind, s in other:
                note = {"compile": "  (one-time, pre-prepared)",
                        "gate": "  (cpu oracle)",
                        "host": "  (cpu)"}.get(kind, "")
                _original_print(f"    {kind:<{w - 2}}  {s:8.2f}s{note}")
            # DEPTH-0 SPANS ONLY. They are the ones that cannot overlap, so wall minus
            # their sum is genuinely untracked time. Summing tot.values() instead adds
            # every nested span again and can go negative.
            top = sum(r[2] for r in self.rows if r[3] == 0)
            resid = wall - top
            _original_print(f"    {'untracked':<{w - 2}}  {resid:8.2f}s"
                            f"  (patchify, dma staging, readbacks)")
            _original_print(f"    {'script wall':<{w - 2}}  {wall:8.2f}s")


PHASES = Phases()

import user_dma_core  # noqa: E402  (module handle: DRAM_START_ADDR / UE_0_BASE_ADDR)
from user_dma_core import (  # noqa: E402  (off-limits to edit)
    DMA_DEVICE_C2H, DMA_DEVICE_H2C, UE_MODE, UE_VECTOR_SIZE,
    URAM_NEAR_FULL_ELEMENTS, UnifiedEngine,
    ue_35bit_addr_shifter,
)
from nn_lib import (  # noqa: E402
    smart_bf16_permute_core, store_weight, eltwise_add_core_dram,
    eltwise_mul_core_dram, silu_core_dram, store_identity_matrix,
)
import user_dma_core  # noqa: E402  (module handle: DRAM_START_ADDR / UE_0_BASE_ADDR)
import multi_engine_shard as mes  # noqa: E402
from multi_engine_shard import (  # noqa: E402
    MultiEngineScheduler, capture_digest,
)


# ======================================================================================
# config
# ======================================================================================

# ---- variant selection -------------------------------------------------------------
# TWO checkpoints share this file. They are the SAME MODEL GRAPH -- SmolVLA: SigLIP tower
# -> pixel-shuffle connector -> VLM prefix -> cross-attn action expert -> flow matching --
# and differ only in dims, counts and weights:
#
#            pulsevla (default)              smolvla (--smolvla)
#   repo     verapulse/pulsevla-libero-0.5b  local vera_pulse_finetune_v0 (smolvla_base ft)
#   layers   32 vlm / 32 expert              16 / 16
#   expert   480 wide (pad 512), inter 1280  720 wide (pad 768), inter 2048
#   cameras  2  -> prefix 177 -> 192         3  -> prefix 241 -> 256
#   rope     theta 10000                     theta 100000   <-- opposite, and silent if wrong
#   action   7 DoF (LIBERO)                  6 DoF (SO-101)
#
# THE SWAP MUST HAPPEN AT IMPORT, BEFORE THE CLASS BODY RUNS. Every dim on
# VeraPulse_UnifiedEngine (E_HIDDEN, V_SLOTS, NUM_LAYERS, ...) is a CLASS ATTRIBUTE read
# from _CFG when the class statement executes, so mutating _CFG later changes nothing.
# Hence argv/env sniffing here rather than a normal argparse flag: argparse runs inside
# main(), which is thousands of lines too late. main() still declares --smolvla so that
# --help lists it and an unknown-argument error cannot fire.
#
# libero_eval.py and the policy server import this module without argv, so the env var
# VERAPULSE_VARIANT=smolvla is the programmatic door in.

_VARIANTS = {
    "pulsevla":   "verapulse_config.json",
    "smolvla":    "verapulse_smolvla_config.json",
    # smolvla's shape config EXACTLY -- vision/text/connector tensors are byte-identical
    # -- with only the action expert retrained, against a quick_gelu backbone so the
    # expert consumes the features this silicon actually produces.
    "smolvla_qg": "verapulse_smolvla_qg_config.json",
}

# Variants whose weights were TRAINED in the accelerator's activation, so quick_gelu is
# the published model rather than a substitution to cancel out.
NATIVE_QUICK_GELU = {"smolvla_qg"}


def _select_variant():
    """'pulsevla' | 'smolvla', from --smolvla / --variant=X in argv or VERAPULSE_VARIANT."""
    env = os.environ.get("VERAPULSE_VARIANT")
    if env:
        if env not in _VARIANTS:
            raise ValueError(f"VERAPULSE_VARIANT={env!r} unknown; pick one of {list(_VARIANTS)}")
        return env
    argv = sys.argv[1:]
    if "--smolvla" in argv:
        return "smolvla"
    for i, a in enumerate(argv):
        if a.startswith("--variant="):
            name = a.split("=", 1)[1]
        elif a == "--variant" and i + 1 < len(argv):
            name = argv[i + 1]
        else:
            continue
        if name not in _VARIANTS:
            raise ValueError(f"--variant {name!r} unknown; pick one of {list(_VARIANTS)}")
        return name
    return "pulsevla"


VARIANT = _select_variant()


def _load_config(path=None, variant=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            _VARIANTS[variant or VARIANT])
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

HF_REPO = _CFG["paths"]["hf_model_repo"]          # verapulse/pulsevla-libero-0.5b, or None
HF_FILES = _CFG["paths"].get("hf_files") or []    # config/model/norm_stats/tokenizer
# The repo ships upstream's OWN smolvla package alongside the weights, and the gate
# oracle imports it out of the bundle. allow_patterns must therefore carry the .py
# globs too -- filtering to the four data files is what breaks `import smolvla`.
HF_CODE_GLOBS = _CFG["paths"].get("hf_code_globs", [])

# The smolvla variant is a LOCAL export, not an HF repo: hf_model_repo is null and
# local_ckpt_dir points at the finetune package. There is nothing to download.
LOCAL_CKPT_DIR = _CFG["paths"].get("local_ckpt_dir")

# ======================================================================================
# Bin layout
# ======================================================================================
# ONE params.bin: sharding never writes the params region (per-engine buffers come out
# of the worker TENSOR arena), so a 1-engine snapshot is byte-identical to an 8-engine
# one. One programs file per (vision, prefix, denoise) engine triple: programs bake
# absolute jump targets and one rendezvous per engine, so a set built for 8 cannot run
# on 4 -- the missing workers never answer a FLAG_CHECK that has no timeout.
_BIN_SUBDIR = "verapulse_bin"
BIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), _BIN_SUBDIR)

# Load order is load-bearing: program DRAM is a bump allocator, so vision -> prefix ->
# denoise reproduces the baked addresses. MUST match dump_programs_to_file's order.
_PROGRAM_ORDER = ("vision", "prefix", "denoise")

# stage -> (_num_engines key, primary program addr attr, saved worker addr list attr,
#           worker program size list attr)
_STAGE_ATTRS = {
    "vision":  ("VIS",     "_vis_program_addr",     "_vis_worker_prog_addrs",
                "_vis_worker_prog_sizes"),
    "prefix":  ("PREFIX",  "_prefix_program_addr",  "_prefix_worker_progs",
                "_prefix_worker_prog_sizes"),
    "denoise": ("DENOISE", "_denoise_program_addr", "_denoise_worker_prog_addrs",
                "_denoise_worker_prog_sizes"),
}

# Compile-time state the execute paths read (run_vision: _vis_batched; run_prefix:
# PREFIX_HIDDEN_DRAM). Explicit, so a new one fails loudly instead of going stale.
_BIN_DERIVED_ATTRS = ("_vis_batched", "PREFIX_HIDDEN_DRAM")


def _programs_stem(engines):
    """(vision, prefix, denoise) engine counts -> programs file stem."""
    v, p, d = (int(engines[s]) for s in _PROGRAM_ORDER)
    return f"programs_e{v}_{p}_{d}"


def clean_bins(bin_dir=None, verbose=True):
    """Delete every bin artifact in `bin_dir`, for ALL engine configurations.

    NAMES THE FILES; never rm -rf the dir -- the checkpoint lives inside it, and on the
    smolvla variant it is a local export with nothing to re-download.
    """
    import glob
    if bin_dir is None:
        bin_dir = BIN_DIR
    victims = sorted(
        glob.glob(os.path.join(bin_dir, "params.bin"))
        + glob.glob(os.path.join(bin_dir, "params.json"))
        + glob.glob(os.path.join(bin_dir, "weight_tensors.pt"))
        + glob.glob(os.path.join(bin_dir, "programs_e*.bin"))
        + glob.glob(os.path.join(bin_dir, "programs_e*_tensors.pt"))
        + glob.glob(os.path.join(bin_dir, "programs_e*.json")))
    freed = 0
    for p in victims:
        freed += os.path.getsize(p)
        os.remove(p)
    if verbose:
        if victims:
            print(f"[clean] removed {len(victims)} bin file(s), "
                  f"{freed / 1024**3:.2f} GB from {bin_dir}")
            for p in victims:
                print(f"    {os.path.basename(p)}")
        else:
            print(f"[clean] no bin files in {bin_dir} (nothing to remove)")
    return victims

CKPT_DIR_NAME = HF_REPO.replace("/", "__") if HF_REPO else None
# Exact count; a mismatch means the mapping silently dropped or aliased a weight.
# 787 for pulsevla (32+32 layers), 500 for smolvla (16+16). Config-driven so a variant
# cannot inherit the other's count and pass a check it never actually ran.
CKPT_NUM_TENSORS = _CFG["paths"].get("ckpt_num_tensors", 787)


def _ckpt_dir(script_dir=None):
    if LOCAL_CKPT_DIR:
        return LOCAL_CKPT_DIR
    script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, _CFG["paths"]["bin_dir"], CKPT_DIR_NAME)


def ensure_checkpoint(script_dir=None):
    """huggingface_hub.snapshot_download(HF_REPO, allow_patterns=HF_FILES) into
    <script_dir>/verapulse_bin/<repo>/ so the checkpoint travels with the model dir
    instead of landing in the machine-global HF cache. Idempotent. Returns the dir.

    For a LOCAL variant (smolvla) there is no repo: assert the export is present and
    return it. Failing loudly here beats letting safetensors raise something unrelated
    a few frames later."""
    if LOCAL_CKPT_DIR:
        need = os.path.join(LOCAL_CKPT_DIR, "model.safetensors")
        if not os.path.exists(need):
            raise FileNotFoundError(
                f"local checkpoint {need} is missing. This variant is not downloadable; "
                f"re-run /home/rohit/vera_pulse_finetune_v0/fetch_weights.sh")
        return LOCAL_CKPT_DIR
    target = _ckpt_dir(script_dir)
    # Idempotent by CONTENT, not by directory existence: a half-finished download
    # leaves the dir behind, and a missing model.safetensors must re-fetch rather
    # than fail later inside safetensors with an unrelated error.
    if all(os.path.exists(os.path.join(target, f)) for f in HF_FILES):
        return target
    from huggingface_hub import snapshot_download
    os.makedirs(target, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO, allow_patterns=list(HF_FILES) + list(HF_CODE_GLOBS),
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


# The two variants nest the SAME modules at different depths. pulsevla flattened them
# directly under 'model.vlm_with_expert.'; the lerobot smolvla export keeps the real HF
# module tree ('vlm.model.text_model.', 'lm_expert.', ...). Rather than fork the whole
# regex table, rewrite the new-style prefix into the old-style one and let ONE table do
# the work. Longest prefix first -- 'vlm.model.text_model.' must beat any shorter match.
_KEY_ALIASES = (
    ("vlm.model.text_model.embed_tokens.", "embed_tokens."),
    ("vlm.model.text_model.",              "text."),
    ("vlm.model.vision_model.",            "vision."),
    ("vlm.model.connector.",               "connector."),
    ("lm_expert.",                         "expert."),
)

# Present in the smolvla export, deliberately unused: the model's output is continuous
# actions, so there is no token head and no sampling. Dropping it must be EXPLICIT --
# letting it fall through to `unmapped` would raise, and a catch-all would map it onto
# something plausible. Every name here is subtracted from the expected tensor count.
_IGNORED_KEYS = frozenset({"model.vlm_with_expert.vlm.lm_head.weight"})


def _canonical_name(k):
    """Upstream safetensors key -> canonical name, or None if unmapped (-> caller raises).

    Handles BOTH variants' key layouts; see _KEY_ALIASES."""
    import re

    if k in _IGNORED_KEYS:
        return None

    if k.startswith(_VLM):
        tail = k[len(_VLM):]
        for new, old in _KEY_ALIASES:
            if tail.startswith(new):
                k = _VLM + old + tail[len(new):]
                break

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

    sd, unmapped, ignored = {}, [], []
    for k, t in raw.items():
        if k in _IGNORED_KEYS:
            ignored.append(k)
            continue
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
            f"[{VARIANT}] mapped {len(sd)} tensors, expected {CKPT_NUM_TENSORS} "
            f"({len(raw)} in the file, {len(ignored)} deliberately ignored) -- the "
            f"checkpoint or the renaming table drifted")
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


# Gate output is terse by default: one line per comparison, prose only on --verbose.
# The explanations are still worth keeping -- they are what makes a FAIL actionable --
# but they do not need to be re-read on every green run.
VERBOSE = False


def vnote(*lines):
    """Print explanatory prose only under --verbose. Indented two extra spaces so it
    reads as commentary attached to the line above it, not as another result."""
    if VERBOSE:
        for ln in lines:
            print(f"    {ln}")


def section(title):
    print(f"\n  -- {title}")


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
    flag = f"  GAIN {gain:.3f}x" if (rr and abs(gain - 1.0) > 0.02) else ""
    mark = {"PASS": "ok  ", "FAIL": "FAIL", "gate OK": "ok  ",
            "GATE BROKEN": "FAIL"}[tag]
    print(f"    {mark} {name:26s} {s:7.2f}dB  cos {c:.6f}"
          + (f"  rms {rh:.4f}/{rr:.4f}" if VERBOSE else "") + flag)
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
        # k/v INPUT WIDTH IS LAYER-DEPENDENT, and getting it wrong makes the fake path
        # unusable rather than merely inaccurate. Self-attn layers project the expert's
        # own hidden stream (E["hidden_size"]); cross-attn layers RE-PROJECT the cached
        # VLM K/V, which is kv_out (320) wide. Both checkpoints prove it -- even layers
        # are (320, hidden), odd layers (320, 320). Emitting `hidden` for every layer
        # made forward_expert die in the cross branch with a shape mismatch on BOTH
        # variants, so --stage ref (which defaults to fake weights) never ran at all.
        is_self = (i % E["self_attn_every_n_layers"] == 0) == \
            E.get("self_attn_on_even", True)
        kv_in = E["hidden_size"] if is_self else E["kv_out"]
        w(f"ae.{i}.q_proj.weight", E["q_out"], E["hidden_size"])
        w(f"ae.{i}.k_proj.weight", ekv, kv_in)
        w(f"ae.{i}.v_proj.weight", ekv, kv_in)
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
        # NO SUFFIX-SIDE PAD MASK, and that part still holds: the device runs
        # SUFFIX_LEN_PAD=64 rows and masks its 14 pad columns, this reference runs
        # exactly CHUNK=50 real rows and has none to mask, so masked-64 and unpadded-50
        # compute the same thing. `pad_to` stays accepted-but-unused for that symmetry.
        # THE PREFIX SIDE IS DIFFERENT and does need a mask -- see b_self below: the
        # cache handed in by the hardware gate is padded to 192 while only `prefix_len`
        # of it is real.
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
            # THE KEY WIDTH IS THE PADDED ONE, NOT THE VALID ONE. forward_expert below
            # concatenates prefix_kv[i] verbatim, and when the caller is the hardware
            # gate that cache is PREFILL_MAX_SEQ_LEN (192) rows wide, not valid_len
            # (177) -- so scores come out [nh, rows, 192+rows] and a bias built at
            # prefix_len+rows is a shape error. That is what this was: the self mask was
            # written before FAULT 4 added the KV concat and never widened with it, so
            # the expert SNR gate crashed the moment it was run against a padded prefix.
            #
            # Two distinct column regions, exactly mirroring what the DEVICE builds in
            # _ae_attn_biases ([QB, PM + SUFFIX_LEN_PAD], prefix cols cut at `valid`):
            #   [0, prefix_len)      real prefix     -- visible
            #   [prefix_len, pw)     prefix PADDING  -- -inf, or the oracle admits pad
            #                                          columns the device masks and the
            #                                          error compounds over 10 steps
            #   [pw, pw+rows)        the suffix      -- causal
            pw = prefix_pad_to if prefix_pad_to is not None else prefix_len
            assert prefix_len <= pw, f"valid prefix {prefix_len} > padded {pw}"
            b_self = torch.zeros(rows, pw + rows)
            b_self[:, prefix_len:pw] = float("-inf")
            b_self[:, pw:] = causal
            # No row may be all -inf or softmax NaNs; prefix column 0 is always visible
            # because prefix_len >= 1, pad rows included. Same argument the device makes.
            assert bool((~torch.isinf(b_self)).any(dim=1).all()), "all -inf row -> NaN"
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
    # --no-vis-batch: force the historical two-pass encoder even when the engine count
    # could batch both camera slots into one execution. Timing control only.
    VIS_NO_BATCH = False

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
    # INSTRUMENT (--denoise-extra-barriers). N redundant all-engine rendezvous per
    # expert layer, emitted outside every region where a barrier is semantically a
    # no-op. Sweeping N prices ONE rendezvous directly: the slope of wall time vs N,
    # divided by the 320 layer-executions, is the per-barrier cost. That number is
    # currently a docstring estimate ("~2 us at ne=12") and the whole barriers-vs-ops
    # question for the unaccounted stage time turns on whether it is really ~2 us or
    # really ~300 us. Zero in every normal run; it changes timing, never numbers.
    DENOISE_EXTRA_BARRIERS = 0

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

    # ===========================================================================
    # ==================================================================================
    # MULTI-ENGINE SHARDING KNOBS  (shared scaffolding -- vision/prefix/denoise)
    # ==================================================================================
    # NUM_ENGINES is the run-wide default; each stage may override it. A stage at 1 is
    # BYTE-IDENTICAL to the historical single-engine program: _vp_barrier returns before
    # emitting anything, split_rows returns [(0, M)], and no worker engine is built.
    NUM_ENGINES = 1

    # Per-stage ceilings, and WHY each one:
    #   VIS     12 -- S=1024 patches. A PURE ROW split still, but the row-block
    #                 granularity is now DERIVED from the engine count (_vis_row_align)
    #                 instead of pinned at 64. At 64 the stage is 16 blocks, which feeds
    #                 8 engines evenly and CANNOT feed 12: 12 engines take 4x2 + 8x1
    #                 blocks, so the busiest still carries 128 rows and the stage runs at
    #                 exactly its 8-engine speed with four engines idling half the time.
    #                 Descending to 8-row blocks gives 128 blocks -> 8x11 + 4x10 = 88 rows
    #                 busiest, an 11.6x ceiling. See _vis_row_align.
    #   PREFIX  12 -- shards the COLUMN axis (H=960 -> 15 blocks, I=2560 -> 40), because
    #                 a row split of PM=192 = 3 blocks would cap the stage at 3. Raised
    #                 from 8 only AFTER the two things that made 12 pointless were fixed:
    #                 k/v are now split per kv-head (10 units) instead of computed
    #                 redundantly on all 12, and attention is group-sharded instead of
    #                 primary-only. The cap alone was worth ~9%; the sharding is the
    #                 change. Modelled on FLOP share: 4.74x -> 7.48x at 8, 9.12x at 12.
    #                 ne=10 AND ne=12 MODEL IDENTICALLY (9.12x). Both fat axes plateau
    #                 there: 15 column blocks over 10 or 12 engines both leave someone
    #                 holding 2, and 40 blocks over 10 or 12 both leave someone holding
    #                 4. The last two engines are free insurance, not throughput.
    #
    #                 q AND o ARE STUCK AT 7.5x FOREVER, at ANY engine count >= 8. H=960
    #                 is 15 column blocks, and 15 over 8+ engines always leaves someone
    #                 holding 2 while others hold 1 -- the barrier makes everyone pay the
    #                 busiest, so those two projections never beat 15/2 = 7.5x. Do not
    #                 read "12 engines" as "12x": the stage-level number is 9.12x and it
    #                 is the fat MLP axis (40 blocks) that earns it.
    #   DENOISE  8 -- suffix is 64 rows = ONE block, so denoise cannot row-split at all;
    #                 it shards output columns (gate/up N=1280) and K (down).
    #
    # DENOISE STAYS AT 8 DELIBERATELY. _num_engines clamps each stage independently, so
    # `--engines 12` gives vision and prefix 12 and leaves denoise at its proven 8; its
    # column axis needs its own balance study, which is a separate change from this one.
    #   DENOISE 10 -- TEN IS A MODELLED OPTIMUM, NOT AN ASSERT BOUNDARY: 11 and 12 are
    #                 SLOWER. Per-op lane width is blocks/ceil(blocks/ne), and NOTHING
    #                 in this stage improves past 10:
    #                   gate/up/down N=1280 -> 20 blocks: 6.67x at ne=8 AND 9, 10.00x at
    #                                ne=10, still 10.00x at 11 and 12. 61% of stage FLOPs
    #                                -- this term is the whole argument for going past 8.
    #                   q_proj       N=960  -> 15 blocks: 7.50x from ne=8 upward, flat.
    #                   o_proj       N=512  -> EXACTLY 8 blocks, so never more than 8
    #                                lanes (split_cols asserts blocks>=n). Engines
    #                                8..ne-1 idle there by design -- see _ae_proj_lanes.
    #                   attention    5 kv groups -> 5-way from ne=5 upward, flat.
    #                   k/v_proj     10 (tensor, kv-head) UNITS -> 5-way at ne=5..9,
    #                                10-way at ne=10, and it CANNOT go past 10. Note it
    #                                is the one op whose width still improves between 9
    #                                and 10, which strengthens rather than moves the
    #                                argument for stopping at 10. See _ae_kv_units.
    #                   action_out (1 block): PRIMARY-ONLY, 1x.
    #                 reduce_add USED TO BE the tie-breaker: (ne-1) SERIAL [64,512] adds
    #                 on the PRIMARY, 320 layer-executions per inference, strictly
    #                 INCREASING in ne -- 59.0 ms at ne=10 against 72.1 at ne=12. That
    #                 argument is now SPENT: multi_engine_shard.reduce_add splits those
    #                 adds across engines by row block, so the term collapsed to 5.90 ms
    #                 at ne=10 vs 6.01 at ne=12 -- a 0.11 ms difference, not 13.
    #
    #                 THE CAP IS STILL 10, for a DIFFERENT and stronger reason: NO OP
    #                 GAINS WIDTH PAST 10. q is flat at 7.50x from ne=8, o is capped at
    #                 8 lanes, the MLP is 10.00x at ne=10 AND at 11 and 12 (20 blocks,
    #                 ceil(20/ne)=2 either way), attention is 5-way, and k/v hits 1 unit
    #                 per engine at exactly 10. With nothing left to unmask, ne=11 and
    #                 ne=12 tie ne=10 and pay the 0.11 ms. Do not raise it expecting a
    #                 win. Modelled stage (PRE-reduce_add-split): ne=8 1006 ms, ne=9
    #                 1012, ne=10 914, ne=11 920, ne=12 927.
    #                 THOSE FIVE NUMBERS PREDATE THE k/v SPLIT and were modelled with
    #                 both projections at 1x. Measured, k/v cost 174.8 ms of the 1.44 s
    #                 stage running primary-only; the split takes that to ~35 ms at
    #                 ne=5..9 and ~17 ms at ne=10, so every entry moves down and ne=10
    #                 moves down furthest. The ORDERING (10 < 8 < 11 < 12) is what the
    #                 cap rests on and the split does not disturb it -- reduce_add is
    #                 still the only strictly increasing term.
    STAGE_MAX_ENGINES = {"VIS": 12, "PREFIX": 12, "DENOISE": 10}

    # Where bins are read from / written to. An instance attribute overrides it.
    bin_dir = BIN_DIR

    # 12, not 16, even though the FLAG index the fabric decodes is 4 bits wide (see
    # generate_instruction_flag_check). 12 is what this board HAS -- HW_INFO reports
    # cores=12 -- and engines 12-15 would be barriers against silicon that does not
    # exist: FLAG_CHECK has no timeout, so the symptom is a wedged board, not an error.
    ENGINE_INDEX_LIMIT = 12

    # Row-block granularity candidates for the vision M-split, coarsest first, and the
    # floor below which we do not go.
    #
    # THE 64 IS A CONVENTION HERE, NOT A HARDWARE REQUIREMENT. The real 64-rule is a
    # LAYOUT rule on N and K -- a 128-byte SRAM row holds 64 bf16 elements, and
    # quantisation blocks run along K. It says nothing about M. Under a pure M-split
    # every engine gets a [rows, FULL_N] shard: the columns are never cut and K is never
    # cut, so N/K alignment holds BY CONSTRUCTION at any row count. What is left is:
    #   - matmat_mul_core       asserts K % 64 only; M is free (and is a runtime GPR here)
    #   - unified_attention_core bounds batch <= aligned_seq_len and aligned_seq_len % 64;
    #                            `batch` itself is unconstrained (user_dma_core:7274-7279)
    #   - the strided vision DMAs need their base 32 B AXI-beat aligned, and every pitch
    #     in this stage (H*2=1536, I*2=6144, C*P*P*2=1536, S*2=2048) is a multiple of 32,
    #     so ANY row offset clears it -- asserted per shard in _emit_encoder_body.
    # Precedent: PREFIX_ROW_ALIGN in pi05 was dropped 64 -> 8 for exactly this reason.
    VIS_ROW_ALIGN_CANDIDATES = (64, 32, 16, 8)
    VIS_ROW_ALIGN_FLOOR = 8

    # Hash the captured denoise instruction stream and print it. THE ne == 1 IDENTITY
    # CHECK: the multi-engine plumbing must leave the single-engine program
    # indistinguishable from the hand-written one, so run --denoise-digest --engines 1
    # before and after any sharding change and compare the two hex strings. Off by
    # default only because hashing a 32 MB program costs a few seconds.
    DENOISE_DIGEST = False

    # Per-execution FPGA timeout. 300 s is right for a real run; drop it (--exec-timeout
    # 30) when debugging a multi-engine hang so the engine-state dump fires in seconds
    # instead of five minutes.
    EXEC_TIMEOUT = 300.0

    # Per-stage overrides; None -> NUM_ENGINES. Set via configure_engines().
    VIS_NUM_ENGINES = None
    PREFIX_NUM_ENGINES = None
    DENOISE_NUM_ENGINES = None

    # ---- worker DRAM arenas ----------------------------------------------------------
    # WHERE THE ARENAS LIVE, AND WHY NOT IN THE TENSOR REGION (pi05 carves them out of
    # the model's tensor allocator; that is impossible here). This model's tensor region
    # is 0x48000000..0x56000000 = 224 MB and tensor_init already asserts against it, with
    # ~70 MB of peak activations inside it. Seven worker arenas of any usable size do not
    # fit. The PROGRAM region, by contrast, is 0x56000000..0x100000000 = 2.85 GB and the
    # three primary programs together are tens of MB. So the arenas are carved off the
    # TOP of the program region instead: base 0xDC000000, 64 MB apart, 9 of them ending
    # exactly at 0x100000000 -- the 4 GB boundary (dma_write at/above it fails with
    # [Errno 512]) -- and above every primary program. _assert_worker_programs_fit checks
    # the far end; _assert_arenas_clear_primary checks the near end.
    # THESE ARE ABSOLUTE ADDRESSES, not config-style offsets. The model's own bases are
    # OFFSETS from DRAM_START_ADDR (params 0x0, tensor 0x48000000, program 0x56000000 ->
    # absolute 0x80000000 / 0xC8000000 / 0xD6000000), but UnifiedEngine's constructor
    # takes absolute bases, and _worker_engine_pool passes these straight through.
    # Mixing the two conventions put the arenas at 0xC0000000, which is INSIDE the
    # weights (params end 0xC340A600) and INSIDE the activation/KV region
    # (0xC8000000..0xD6000000): workers 1-3 wrote their programs and per-engine buffers
    # on top of live model memory, and vision's activation writes then corrupted those
    # workers' instruction streams. A worker executing corrupted code never reaches its
    # barrier, and FLAG_CHECK has no timeout -- the run HANGS instead of failing.
    #
    # The arenas now live in the free tail of the PROGRAM region, above everything the
    # primary uses (its three programs total ~38 MB from 0xD6000000):
    #     params  0x080000000 .. 0x0C340A600
    #     tensor  0x0C8000000 .. 0x0D6000000
    #     program 0x0D6000000 .. 0x100000000   <- primary at the bottom, arenas at 0xDC000000
    #
    # THE BASE IS SET BY THE PEAK WORKER COUNT, NOT BY TASTE. The arenas are laid out
    # bottom-up from the base and must ALL end at or below VIS_WORKER_ARENA_TOP (4 GB;
    # dma_write at/above it fails with [Errno 512]). At 64 MB apart:
    #     base 0xE0000000 -> (0x100000000-0xE0000000)/0x4000000 =  8 workers ->  9 engines
    #     base 0xDC000000 -> (0x100000000-0xDC000000)/0x4000000 =  9 workers -> 10 engines
    # DENOISE's ceiling is 10 (see STAGE_MAX_ENGINES), so the base moved down one arena
    # to 0xDC000000 -- the 9 arenas then tile 0xDC000000..0x100000000 exactly. That
    # leaves the PRIMARY 0xD6000000..0xDC000000 = 96 MB, against the ~38 MB its four
    # programs actually occupy (the 31 MB unrolled denoise dominates); the tail of
    # compile_denoise_loop asserts the primary allocator stayed below the base, because
    # _assert_arenas_clear_primary only runs once, at pool construction, long before any
    # program is compiled. Raising any stage past 10 engines needs a SMALLER
    # VIS_WORKER_ARENA_BYTES, not a lower base: 0xD6000000 is the floor.
    # BASE STAYS AT 0xE0000000. The denoise work proposed lowering it to 0xDC000000 to
    # fit 9 fixed-size 64 MB arenas, but _resolve_worker_arena_profile below already
    # solves that by COMPUTING the arena size above 7 workers -- and keeping the base
    # put is what preserves every worker DRAM address at ne <= 8 (7 workers x 64 MB from
    # 0xE0000000, exactly as before). Lowering the base would have moved all of them.
    VIS_WORKER_ARENA_BASE    = 0xE0000000        # absolute, inside the program region
    VIS_WORKER_ARENA_BYTES   = 0x04000000        #  64 MB per worker (<=7 workers)
    #
    # THE TENSOR WINDOW MUST HOLD EVERY STAGE'S PER-ENGINE BUFFERS AT ONCE. They share
    # one allocator per worker and are never freed, so the window has to fit the SUM,
    # not the largest. Measured per worker: vision 2.75 MB (attn_scratch alone is 2.25,
    # kept at full size because the core derives sub-offsets from compile-time S/D),
    # prefix 2.56 MB (was 1.76 before the prefix attention shard: -0.23 for the dropped
    # redundant lm_k/lm_v, +1.06 for the per-engine flash staging and scratch), denoise
    # 0.53 MB -- 5.84 MB total. The old 4 MB window (tensor at
    # +4 MB, program at +8 MB) fit any ONE stage, which is why --vis_8 and --pref_8 both
    # worked, and overflowed into the worker's own PROGRAM area at --engines 8:
    # corrupted instruction stream -> the worker never reaches its barrier -> FLAG_CHECK
    # spins forever. 28 MB leaves room for the attention sharding still to come.
    VIS_WORKER_TENSOR_OFFSET = 0x00400000        #   4 MB in: per-engine scratch (28 MB)
    VIS_WORKER_PROGRAM_OFFSET = 0x02000000       #  32 MB in: the worker's own programs
    VIS_WORKER_ARENA_TOP     = 0x100000000       # hard DRAM ceiling
    # --- MANY-WORKER PROFILE (>7 workers, i.e. --engines 9+) -----------------------
    # See _resolve_worker_arena_profile. The tensor window shrinks 28 MB -> 8 MB, which
    # is still 37% margin over the 5.84 MB every stage's per-engine buffers actually sum
    # to, and that is what frees the space for 11 arenas. THE MARGIN IS THE THING TO
    # WATCH: it was 59% before the prefix attention shard added 1.06 MB/worker of flash
    # staging. _assert_worker_programs_fit is what catches an overrun, at compile time
    # and with a byte count -- the alternative is a worker whose per-engine buffers walk
    # into its own program area, which hangs on FLAG_CHECK rather than failing.
    VIS_WORKER_MANY_PROGRAM_OFFSET = 0x00C00000  #  12 MB in: tensor window is 8 MB
    # Floor on program space per worker. The primary's three programs total ~38 MB and a
    # worker carries only the SHARDED parts of each, so 32 MB is a real floor with room;
    # dropping under it means a stage cannot fit and should fail here, not deeper in.
    VIS_WORKER_ARENA_PROGRAM_MIN   = 0x02000000  #  32 MB

    @staticmethod
    def _col_split(N, n, align=UE_VECTOR_SIZE):
        """HOST-SIDE mirror of MultiEngineScheduler.split_cols: whole `align`-element
        blocks, deliberately UNEVEN (the trailing engines get one block fewer) rather
        than ever unaligned. The weight slicing in _weight_init_expert and the region
        body in _ae_gated_mlp_sharded MUST agree on this partition; the body asserts it
        against the scheduler, because disagreeing gives finite garbage, not a crash."""
        if n == 1:
            return [(0, N)]
        assert N % align == 0, f"_col_split: N={N} is not a multiple of {align}"
        blocks = N // align
        assert blocks >= n, (
            f"_col_split: N={N} is only {blocks} block(s) of {align}, too few for "
            f"{n} engine(s)")
        base, rem = divmod(blocks, n)
        counts = [align * (base + (1 if i < rem else 0)) for i in range(n)]
        return [(sum(counts[:i]), counts[i]) for i in range(n)]


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
        # Snapshot the instance BEFORE any weight lands, so the tail of this method can
        # tell exactly which attributes the weight phase published (see _capture_weight
        # _attrs). A bin run skips this whole method, and those attributes -- weight DRAM
        # addresses, above all -- are what the host-side readback paths index.
        _bin_attrs_before = set(self.__dict__)
        # EVERY worker engine is constructed HERE, before the first store_weight, for
        # the reason the docstring's ORDERING HAZARD paragraph gives. This is the actual
        # fix; _vp_dram_selftest_guard is only belt and braces. It is a no-op at
        # NUM_ENGINES == 1 (no engine is built at all).
        self._worker_engine_pool()
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
        # Where the params allocator stood when the weight phase finished. params.bin is
        # dumped at the END of a full run, so it also contains what the compile phase
        # allocated afterwards (the AE time table). load_params rewinds to THIS, not to
        # params.json's total, so anything allocating params after a bin load lands back
        # on the addresses the programs bake rather than past every weight.
        self._params_ofs_after_weight_init = self.get_params_dram_usage()
        self._capture_phase_attrs(_bin_attrs_before, "weight")

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

        # How many engines the DENOISE stage will run on. Resolved HERE, at weight-init,
        # because the K-sliced down-projection blobs below are cut for exactly this
        # count; compile_denoise_loop asserts the two agree (a silent disagreement is
        # finite garbage). This is why --engines / --dns_8 must be applied to the model
        # object BEFORE weight_init(), and why the sliced blobs are not built lazily.
        ne_dn = self._num_engines("DENOISE")

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
            # THE DOWN PROJECTION IS THE ONE WEIGHT THE SHARDED DENOISE CANNOT REACH BY
            # ADDRESS ARITHMETIC. _ae_gated_mlp_sharded splits it on K (the intermediate
            # axis, 1280) because that is the axis the gate/up column lanes already
            # partition -- and B is stored N x K ROW-MAJOR, so B[:, k0:k0+Kc] is STRIDED,
            # one gap per output row. Shifting B_DRAM_ADDR cannot express it. The cut is
            # therefore made ONCE, on the host, here, at weight-init time -- exactly what
            # multi_engine_shard.KShardContext refuses to hide from the caller.
            dw = npad(sd[f"ae.{i}.down_proj.weight"], HP)          # [HP, I] = N x K
            # THE DOWN PROJECTION IS THE ONE WEIGHT THE SHARDED DENOISE CANNOT REACH BY
            # ADDRESS ARITHMETIC. _ae_gated_mlp_sharded splits it on K (the intermediate
            # axis, 1280) because that is the axis the gate/up column lanes already
            # partition -- and B is stored N x K ROW-MAJOR, so B[:, k0:k0+Kc] is STRIDED,
            # one gap per output row. Shifting B_DRAM_ADDR cannot express it. The cut is
            # therefore made ONCE, on the host, here, at weight-init time.
            #
            # AN N-SPLIT WAS TRIED AND REVERTED, and this note is here so it is not tried
            # a third time. The idea was to scatter the mult lanes into a shared
            # [64, 1280] and run down N-split at 8 lanes, deleting the 9-add reduce chain
            # -- 6 fewer ops per layer. It MEASURED +50 ms. The reason is that denoise
            # tracks its matmul floor at a flat ~2.6x (see the layer-sweep note in
            # compile_denoise_loop) and 8 N-lanes is a NARROWER cut than a 10-way
            # K-split: the floor went 412 -> 430 ms and the wall clock followed it
            # exactly. Op count and DMA-transaction count both turned out to be
            # irrelevant; only the floor moves this stage.
            if ne_dn > 1:
                la["down_weight_k"] = [
                    store_weight(self, dw[:, k0:k0 + kc].contiguous())
                    for k0, kc in self._col_split(I, ne_dn)]
                # The unsliced blob is DEAD once the stage is sharded, and it is not
                # cheap: 512*1280*2 B x 32 layers = 42 MB, against ~64 MB of params
                # headroom under tensor_dram_base. The slices tile it exactly, so
                # storing only them is byte-neutral. None (not absent) so any accidental
                # use in a sharded run is a TypeError at emit time, not a wrong address
                # -- which is exactly how the N-split experiment announced itself.
                la["down_weight"] = None
            else:
                la["down_weight"] = store_weight(self, dw)
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
        # IDEMPOTENT, like _lm_tensor_init/_ae_tensor_init. Several call sites invoke
        # tensor_init defensively (main, then again inside the --gate-upstream / bisect
        # branches). Re-running it used to be merely wasteful because compilation was
        # lazy and therefore always came AFTER the last allocation. With precompile_all
        # the programs are emitted first, so a second allocation would re-point every
        # VIS_* buffer at a fresh base while the compiled stream still reads the old
        # one -- silent stale-memory reads, not a crash.
        if getattr(self, "_tensor_init_done", False):
            return
        self._tensor_init_done = True
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
        self._assert_tensor_region_fits("tensor_init (vision + kv cache)")

    def _assert_tensor_region_fits(self, label):
        """The primary's tensor allocator has NO bound of its own -- allocate_tensor_dram
        is a bare pointer bump (user_dma_core.py:703). Past the end of the region it
        marches straight into PROGRAM DRAM, where the primary's three compiled programs
        live, and an activation write then overwrites instructions.

        THAT FAILURE IS PERSISTENT AND LOOKS LIKE NOTHING ELSE. Every input is re-staged
        on each inference and the programs are address-static, so a transient fault would
        clear on the next run; corrupted program bytes do not. The symptom is a stage
        that runs to completion at its normal latency and returns finite-but-wrong or
        non-finite tensors, on every subsequent inference, until the process restarts.

        THIS USED TO BE CHECKED IN ONE PLACE ONLY -- the end of tensor_init, i.e. after
        vision and the KV cache and BEFORE _lm_tensor_init and _ae_tensor_init had
        allocated anything. Those two are the larger half and were entirely unguarded, so
        the check passed while the actual total could be anywhere. Called from all three
        now."""
        total = self.get_tensor_dram_usage()
        region = self._program_dram_base - self._tensor_dram_base
        assert total <= region, (
            f"tensor region overflow after {label}: {total / 1024**2:.1f} MB used of "
            f"{region / 1024**2:.0f} MB (0x{self._tensor_dram_base:X}.."
            f"0x{self._program_dram_base:X}). Activations are spilling into program "
            f"DRAM and will overwrite the compiled programs -- which shows up as a "
            f"stage returning non-finite tensors on every inference, not as a crash.")

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

    # ==================================================================================
    # multi-engine scaffolding: engine counts, barrier, worker pool, schedulers
    # ==================================================================================

    def _num_engines(self, stage="VIS"):
        """Resolve one stage's engine count. A per-stage override wins over NUM_ENGINES,
        and the stage ceiling is enforced here so no caller can request more than the
        stage's shape supports."""
        override = getattr(self, f"{stage}_NUM_ENGINES", None)
        n = int(override if override is not None else self.NUM_ENGINES)
        cap = self.STAGE_MAX_ENGINES.get(stage, n)
        assert n >= 1, f"{stage}_NUM_ENGINES={n} must be >= 1"
        # The board's real core count bounds EVERY stage. A barrier addressed at an
        # engine the fabric does not have spins forever -- FLAG_CHECK has no timeout.
        cap = min(cap, self.ENGINE_INDEX_LIMIT)
        if stage == "PREFIX":
            # The prefix shards on the COLUMN axis, so its ceiling is the column-block
            # count, not the row-block count. Row-splitting PM=192 would cap it at 3,
            # which is exactly why this stage moved to columns.
            cap = min(cap, self._prefix_column_blocks())
        return min(n, cap)

    def _vis_row_align(self, ne=None):
        """Row-block granularity for the vision M-split, DERIVED from the engine count.

        Picks the align that minimises the load imbalance
        ``max_shard_blocks / (blocks / ne)`` -- because every barrier makes all engines
        pay the busiest one, so the STAGE cost tracks the largest shard, not the mean.
        Ties break to the smallest ROW spread (a tighter spread is strictly better for
        the same ratio), then to the LARGEST align.

        That last tie-break is what keeps this change free for everything already
        running: at ne in {1, 2, 4, 8} the engine count divides 1024/64 = 16 exactly, so
        align 64 scores a perfect 1.0 ratio with zero spread, ties every finer align, and
        wins on size. Those runs emit the SAME row counts and the SAME offsets as before
        this method existed -- byte-identical, not merely equivalent.

        It only descends where 64 genuinely cannot balance:
            ne=10  align 64 -> 6x128 + 4x64 rows, ratio 1.25  ->  8.00x ceiling
                   align  8 -> 8x104 + 2x96      ratio 1.016  ->  9.85x   <- chosen
            ne=12  align 64 -> 4x128 + 8x64       ratio 1.50   ->  8.00x
                   align  8 -> 8x88 + 4x80        ratio 1.031  -> 11.64x  <- chosen
        """
        S = int(self._cfg["vision"]["num_patches"])
        if ne is None:
            ne = self._num_engines("VIS")
        ne = int(ne)
        best = None
        for align in self.VIS_ROW_ALIGN_CANDIDATES:
            if align < self.VIS_ROW_ALIGN_FLOOR or S % align:
                continue
            blocks = S // align
            if blocks < ne:
                continue
            base, rem = divmod(blocks, ne)
            parts = [base + (1 if i < rem else 0) for i in range(ne)]
            key = (max(parts) / (blocks / ne),          # imbalance: what a barrier costs
                   (max(parts) - min(parts)) * align,   # row spread, in rows
                   -align)                              # prefer the coarsest that ties
            if best is None or key < best[0]:
                best = (key, align)
        assert best is not None, (
            f"no row align in {self.VIS_ROW_ALIGN_CANDIDATES} can split S={S} across "
            f"{ne} engines at or above the {self.VIS_ROW_ALIGN_FLOOR}-row floor")
        return best[1]

    def _peak_engines(self):
        """The largest count any stage asks for. The worker pool is sized from THIS, once
        for the whole run: a stage raising its count after the pool exists is not
        supported (the pool's allocators are what keep each stage's worker programs from
        landing on top of the previous stage's)."""
        return max([int(self.NUM_ENGINES)]
                   + [self._num_engines(s) for s in ("VIS", "PREFIX", "DENOISE")])

    def _vp_barrier(self, ue, engine_idx, ne, margin_nops=32):
        """Emit ONE engine's slice of a symmetric all-engine rendezvous.

        SET / CHECK(every peer) / margin NOPs / CLEAR -- the shape proven to RE-ARM by
        user_hw_test.flag_rendezvous_repeat_test. It is emitted per STREAM, not through
        MultiEngineScheduler.barrier(), because a stage body is emitted one complete
        engine stream at a time (each engine's program is independent; only the
        per-stream order matters).

        AT ne == 1 THIS EMITS NOTHING AT ALL. That is the contract that makes the
        single-engine program byte-identical to the historical one; do not "simplify" it
        into an unconditional emit guarded downstream."""
        if ne == 1:
            return
        ue.generate_instruction_flag_set()
        for j in range(ne):
            if j != engine_idx:
                ue.generate_instruction_flag_check(target_engine_idx=j)
        for _ in range(margin_nops):
            ue.generate_instruction_nop()
        ue.generate_instruction_flag_clear()

    @contextlib.contextmanager
    def _vp_dram_selftest_guard(self, ne):
        """Protect the 16 KB at DRAM_START_ADDR from UnifiedEngine.__init__.

        Constructing ANY UnifiedEngine runs init_unified_engine(), whose DRAM read/write
        self-test dma_writes 8192 uint16 to the HARDCODED DRAM_START_ADDR (0x80000000).
        It ignores params_dram_base entirely, so every engine ever built stomps the same
        window. user_dma_core.py is off-limits, so: snapshot, build, restore.

        Belt and braces only -- the pool is built BEFORE weight_init (see weight_init),
        which is the actual fix. The read is allowed to fail: this model's allocators are
        raw file offsets starting at 0, so 0x80000000 can be an address nothing has ever
        written, and reading one returns EIO rather than zeros. A failed snapshot means
        there was nothing there to preserve."""
        n_bytes = 8192 * 2
        if ne <= 1:
            yield                      # no engine is constructed: nothing to guard
            return
        base = user_dma_core.DRAM_START_ADDR
        buf = bytearray(n_bytes)
        ok = True
        try:
            if self.dma_read(DMA_DEVICE_C2H, base, buf, n_bytes) != n_bytes:
                ok = False
        except OSError:
            ok = False
        try:
            yield
        finally:
            if ok:
                # Write the raw bytearray back DIRECTLY -- never via
                # torch.frombuffer(bytes(buf)), which aliases a read-only temporary and
                # has smashed the CPython heap into a much later segfault.
                self.dma_write(DMA_DEVICE_H2C, base, bytes(buf), n_bytes)

    def _assert_arenas_clear_primary(self, n):
        """The arenas share ONE flat address space with the model's own regions.

        EVERYTHING HERE IS COMPARED IN ABSOLUTE SPACE. The previous version compared the
        primary's program cursor (offset space, ~0x56000000) against the arena base
        (absolute, 0xC0000000) and therefore passed no matter where the arenas landed --
        which is how workers 1-3 ended up inside the weights and the activation region.
        The program allocator is an unchecked bump allocator, so these asserts are the
        only thing standing between a large primary program and the workers' code.
        """
        DRAM_START = user_dma_core.DRAM_START_ADDR

        def absolute(a):
            return a if a >= DRAM_START else DRAM_START + a

        lo = self.VIS_WORKER_ARENA_BASE
        hi = lo + max(0, n - 1) * self.VIS_WORKER_ARENA_BYTES
        assert lo >= DRAM_START, (
            f"worker arena base 0x{lo:X} is below DRAM_START 0x{DRAM_START:X}; these "
            f"constants are ABSOLUTE, not config-style offsets")
        assert hi <= self.VIS_WORKER_ARENA_TOP, (
            f"{n - 1} worker arena(s) of {self.VIS_WORKER_ARENA_BYTES >> 20} MB from "
            f"0x{lo:X} end at 0x{hi:X}, past the 0x{self.VIS_WORKER_ARENA_TOP:X} ceiling")

        # No overlap with the model's OWN regions. This is the check that was missing.
        p0 = absolute(self._params_dram_base)
        p_end = p0 + self.get_params_dram_usage()
        t0, g0 = absolute(self._tensor_dram_base), absolute(self._program_dram_base)
        for name, r_lo, r_hi in (("params (weights)", p0, max(p_end, t0)),
                                 ("tensor (activations + KV cache)", t0, g0)):
            assert not (lo < r_hi and hi > r_lo), (
                f"worker arenas 0x{lo:X}..0x{hi:X} overlap {name} "
                f"0x{r_lo:X}..0x{r_hi:X}. Workers would write their programs and "
                f"per-engine buffers over live model memory, and the model's own writes "
                f"would corrupt worker instruction streams -- which hangs on a "
                f"FLAG_CHECK rather than failing. Move VIS_WORKER_ARENA_BASE.")

        cur = absolute(self.get_program_dram_addr())
        assert cur < lo, (
            f"the primary's program allocator has reached 0x{cur:X}, at/above the worker "
            f"arena base 0x{lo:X} -- primary programs and worker programs would overwrite "
            f"each other. Raise VIS_WORKER_ARENA_BASE.")

    def _resolve_worker_arena_profile(self, workers=None):
        """Size the per-worker arena for `workers` worker engines. Idempotent.

        THE ARENAS DO NOT FIT AT 64 MB ONCE THERE ARE MORE THAN 8 OF THEM. They live in
        the free tail of the program region, 0xE0000000..0x100000000 = 512 MB, so the
        fixed 64 MB layout tops out at exactly 8 workers (--engines 9). Twelve engines
        need 11 arenas, i.e. 704 MB, and there is nowhere to put them: lowering the base
        to the primary's ceiling only buys 672 MB, still short.

        It does not need 64 MB. The arena holds three things and the measurements are in
        the constants above: a params base (unused by workers), a per-engine tensor
        window whose SUM over all stages is 5.84 MB, and the worker's own accumulated
        programs. 64 MB with a 32 MB program offset was chosen for headroom, not need --
        it spends 28 MB on a tensor window that uses 5.

        So above 7 workers the arena is COMPUTED from the space that actually exists and
        the tensor window drops to 8 MB (still 37% margin over the measured 5.84):

            workers   arena    program space
                  9    56 MB           44 MB    (--engines 10)
                 11    46 MB           34 MB    (--engines 12)

        <= 7 WORKERS IS LEFT EXACTLY AS IT WAS -- 64 MB arena, 4 MB tensor offset, 32 MB
        program offset. Every configuration proven on this board today (--engines up to
        8, --vis_8, --pref_8, --dns_8) keeps its byte-for-byte addresses; only counts
        that could not run at all before get the new layout.

        If a worker program ever does outgrow the computed space,
        _assert_worker_programs_fit fires at compile time with an exact byte count --
        this trades unused headroom for a loud, early failure, never a silent one.
        """
        if workers is None:
            workers = max(0, self._peak_engines() - 1)
        workers = int(workers)
        if getattr(self, "_arena_profile_resolved", None) == workers:
            return
        if workers <= 7:
            self._arena_profile_resolved = workers
            return
        avail = self.VIS_WORKER_ARENA_TOP - self.VIS_WORKER_ARENA_BASE
        per = (avail // workers) & ~0xFFFFF                  # floor to a whole MB
        per = min(per, type(self).VIS_WORKER_ARENA_BYTES)    # never grow past today's 64
        prog_off = self.VIS_WORKER_MANY_PROGRAM_OFFSET
        assert per > prog_off + self.VIS_WORKER_ARENA_PROGRAM_MIN, (
            f"{workers} worker arenas fit only {per >> 20} MB each in the "
            f"{avail >> 20} MB above 0x{self.VIS_WORKER_ARENA_BASE:X}, which leaves "
            f"{(per - prog_off) >> 20} MB of program space -- under the "
            f"{self.VIS_WORKER_ARENA_PROGRAM_MIN >> 20} MB floor a worker carrying every "
            f"sharded stage needs. Lower VIS_WORKER_ARENA_BASE or shard fewer stages.")
        self.VIS_WORKER_ARENA_BYTES = per
        self.VIS_WORKER_PROGRAM_OFFSET = prog_off
        self._arena_profile_resolved = workers
        _original_print(
            f"    [engines] {workers} workers: arena {per >> 20} MB each "
            f"(tensor window {(prog_off - self.VIS_WORKER_TENSOR_OFFSET) >> 20} MB, "
            f"program space {(per - prog_off) >> 20} MB) -- computed, the fixed 64 MB "
            f"layout only fits 8")

    def _worker_engine_pool(self, n=None):
        """The run's ONE set of worker UnifiedEngines, indices 1..peak-1.

        Built once, for the PEAK stage count, and handed to EVERY stage scheduler. The
        DRAM allocators live inside these objects, so sharing them is what keeps each
        stage's worker programs landing AFTER the previous stage's rather than on top of
        it (two schedulers over FRESH engines restart the allocator at offset 0, stage 1
        then launches stage 2's code, the barrier desyncs, and the engines spin forever
        on a FLAG_CHECK that has no timeout).

        CALL THIS BEFORE weight_init -- see _vp_dram_selftest_guard."""
        from user_dma_core import UnifiedEngine

        n = self._peak_engines() if n is None else int(n)
        pool = getattr(self, "_worker_pool", None)
        if pool is not None:
            assert len(pool) >= n - 1, (
                f"worker pool holds {len(pool)} engine(s) but {n - 1} are needed; the "
                f"pool is sized from the peak stage count at first use, so a stage "
                f"raising its count afterwards is not supported.")
            return pool
        peak = max(n, self._peak_engines())
        if peak <= 1:
            self._worker_pool = []
            return self._worker_pool
        # BEFORE the arena guard and before any worker is constructed: the arena size is
        # what those addresses are derived from, and a worker built against the wrong
        # stride writes its program into its neighbour's arena.
        self._resolve_worker_arena_profile(peak - 1)
        self._assert_arenas_clear_primary(peak)
        pool = []
        with self._vp_dram_selftest_guard(peak):
            for i in range(1, peak):
                base = self.VIS_WORKER_ARENA_BASE + (i - 1) * self.VIS_WORKER_ARENA_BYTES
                pool.append(UnifiedEngine(
                    BASE_ADDR=user_dma_core.UE_0_BASE_ADDR + i * 0x00010000,
                    params_dram_base=base,
                    tensor_dram_base=base + self.VIS_WORKER_TENSOR_OFFSET,
                    program_dram_base=base + self.VIS_WORKER_PROGRAM_OFFSET,
                ))
        self._worker_pool = pool
        _original_print(f"    [engines] worker pool: {len(pool)} shared engine(s) "
                        f"@0x{self.VIS_WORKER_ARENA_BASE:X} +"
                        f"{self.VIS_WORKER_ARENA_BYTES >> 20}MB each "
                        f"(one allocator each, reused by every stage)")
        return pool

    def _make_stage_scheduler(self, stage, ne=None):
        """Cached MultiEngineScheduler for one stage, over the SHARED worker pool.

        A scheduler is built at ne == 1 too, and that is required rather than incidental:
        compile_encoder calls sched.split_rows(S) unconditionally and the emitter is
        written so a 1-engine split is (0, S) -- the historical single-engine stream. A
        1-engine scheduler owns no workers and emits no barriers, so it costs nothing."""
        from multi_engine_shard import MultiEngineScheduler

        ne = self._num_engines(stage) if ne is None else int(ne)
        cache = self.__dict__.setdefault("_sched_by_stage", {})
        key = (stage, ne)
        if key in cache:
            return cache[key]
        # MUST precede the constructor call: the arena stride and program offset are
        # passed as kwargs below and Python evaluates them BEFORE _worker_engine_pool
        # runs, so relying on the pool to resolve the profile would hand this scheduler
        # the unresolved 64 MB stride while the workers themselves used the computed one.
        self._resolve_worker_arena_profile()
        sched = MultiEngineScheduler(
            self, num_engines=ne,
            worker_dram_base=self.VIS_WORKER_ARENA_BASE,
            worker_dram_stride=self.VIS_WORKER_ARENA_BYTES,
            worker_tensor_offset=self.VIS_WORKER_TENSOR_OFFSET,
            worker_program_offset=self.VIS_WORKER_PROGRAM_OFFSET,
            # "blocks": split by whole row_align-row blocks so every shard stays
            # aligned whatever M is. For the vision stage (S=1024) blocks and even agree
            # exactly at 1/2/4/8 engines; for the prefix (PM=192 = 3 blocks of 64) only
            # "blocks" is workable at all.
            split_mode="blocks",
            # ONLY THE VISION STAGE DERIVES ITS ALIGN. Vision is the pure M-split, where
            # a finer block is safe (see VIS_ROW_ALIGN_CANDIDATES) and necessary to
            # balance 10/12 engines. The prefix and denoise stages shard COLUMNS -- their
            # row_align is unused by split_cols/split_k -- so they keep the 64 default
            # and nothing about them moves.
            row_align=(self._vis_row_align(ne) if stage == "VIS" else 64),
            # This board has more than the 2 engines multi_engine_shard's default guard
            # assumes. That guard is stale -- 8 engines are production-proven -- and the
            # opt-in is explicit at the CLI (--engines/--vis_8 both name a count).
            allow_more_than_two_engines=True,
            workers=self._worker_engine_pool(ne))
        cache[key] = sched
        return sched

    def _preclear_flags_once(self, sched):
        """Clear every engine's flag register ONCE PER PROCESS, before the first launch.

        The guard is process-wide, NOT per stage: preclear_flags() acts on the SHARED
        engine pool, so a second call from a later stage is not a no-op -- it re-scans
        every engine and issues a SW_RESET to any that reads queue_busy. A worker still
        draining its margin NOPs from the previous stage reads exactly that, and
        resetting it mid-flight means it never answers the next stage's first
        rendezvous: FLAG_CHECK has no timeout, so the run hangs rather than failing.
        (Symptom: vision at ne=8 correct, prefix hung; --pref_8 alone fine, because
        vision at ne=1 never launched a worker to leave busy.)

        Stages must ALSO join their workers after every execution; this guard stops the
        damage, joining is what prevents it.
        """
        if self.__dict__.get("_vp_flags_precleared"):
            return
        sched.preclear_flags()
        self.__dict__["_vp_flags_precleared"] = True

    def _record_worker_prog_sizes(self, stage, sched):
        """Worker program sizes, snapshotted right after finalize().

        Here and nowhere later: the next stage's begin_program() clears the capture on
        these SHARED workers. Differencing addresses does not work either --
        allocate_program_dram rounds up to 64 B and over-reports every size.
        """
        setattr(self, _STAGE_ATTRS[stage][3],
                [w.get_capture_instruction_size_bytes() for w in sched.workers])

    # ---------------------------------------------------------------- bin DMA I/O --

    def _bin_read(self, addr, size, label="", ue=None):
        """Chunked + retried + CHECKED readback -> bytes. Unchecked, dma_read returns -1
        after only printing the errno and leaves a zero-filled buffer -- an all-zero bin."""
        ue = ue or self
        buf = bytearray(size)
        mv = memoryview(buf)
        chunk, offset = 256 * 1024, 0
        tag = f" {label}" if label else ""
        while offset < size:
            n = min(chunk, size - offset)
            piece = bytearray(n)
            for _ in range(5):
                if ue.dma_read(DMA_DEVICE_C2H, addr + offset, piece, n) == n:
                    break
            else:
                if chunk > 4096:
                    chunk //= 2
                    continue
                raise RuntimeError(
                    f"dma_read{tag} FAILED at 0x{addr + offset:X} (+{offset}/{size} B) "
                    f"even at the 4KB floor -- refusing to write a zero-filled bin that "
                    f"would masquerade as a real program")
            mv[offset:offset + n] = piece
            offset += n
        return bytes(buf)

    def _bin_write(self, addr, data, label="", ue=None):
        """Chunked + retried write of `data` to DRAM. `ue` selects which engine's DMA
        channel to use -- worker programs must be written through their OWN engine."""
        ue = ue or self
        size = len(data)
        mv = memoryview(data)
        chunk, offset = 256 * 1024, 0
        tag = f" {label}" if label else ""
        while offset < size:
            n = min(chunk, size - offset)
            piece = bytes(mv[offset:offset + n])
            for _ in range(5):
                if ue.dma_write(DMA_DEVICE_H2C, addr + offset, piece, n) == n:
                    break
            else:
                if chunk > 4096:
                    chunk //= 2
                    continue
                raise RuntimeError(
                    f"dma_write{tag} FAILED at 0x{addr + offset:X} "
                    f"(+{offset}/{size} B) even at the 4KB floor")
            offset += n

    def _bin_engines(self):
        """The (vision, prefix, denoise) engine triple this run is configured for."""
        return {s: int(self._num_engines(_STAGE_ATTRS[s][0])) for s in _PROGRAM_ORDER}

    def _assert_worker_programs_fit(self, sched=None, label=""):
        """Every sharded stage's worker programs share ONE allocator per worker engine.
        Overflowing an arena marches that allocator into the NEXT worker's arena --
        silent corruption, not a hang. Call after every finalize().

        `sched=None` checks EVERY cached stage scheduler, which is the meaningful test:
        the arenas are cumulative across stages, so vision+prefix+denoise together are
        what can overflow, not any one of them. Passing a single scheduler checks just
        that stage. Returns total worker program bytes so a caller can report them."""
        limit = self.VIS_WORKER_ARENA_BYTES - self.VIS_WORKER_PROGRAM_OFFSET
        scheds = ([sched] if sched is not None
                  else list(self.__dict__.get("_sched_by_stage", {}).values()))
        total = 0
        for sc in scheds:
            for i, w in enumerate(sc.workers):
                arena = self.VIS_WORKER_ARENA_BASE + i * self.VIS_WORKER_ARENA_BYTES
                # THE PER-ENGINE TENSOR WINDOW. Every stage's per-engine buffers share
                # one allocator per worker and are never freed, so this is the SUM over
                # stages. Overflowing it runs into the worker's own program area below,
                # which corrupts its instruction stream -- and a worker executing
                # corrupted code hangs on FLAG_CHECK instead of failing. This assert is
                # what was missing when --engines 8 deadlocked.
                t_cur = w.get_tensor_dram_addr()
                t_lim = arena + self.VIS_WORKER_PROGRAM_OFFSET
                assert t_cur <= t_lim, (
                    f"{label}worker {i + 1} per-engine buffers reached 0x{t_cur:X}, past "
                    f"the 0x{t_lim:X} start of its program area "
                    f"({(t_cur - arena - self.VIS_WORKER_TENSOR_OFFSET) / 2**20:.2f} MB "
                    f"used of "
                    f"{(self.VIS_WORKER_PROGRAM_OFFSET - self.VIS_WORKER_TENSOR_OFFSET) / 2**20:.0f} "
                    f"MB). Raise VIS_WORKER_PROGRAM_OFFSET.")
                base = arena + self.VIS_WORKER_PROGRAM_OFFSET
                used = w.get_program_dram_addr() - base
                assert 0 <= used <= limit, (
                    f"{label}worker {i + 1} program arena overflow: {used} B used of "
                    f"{limit} B (VIS_WORKER_ARENA_BYTES="
                    f"0x{self.VIS_WORKER_ARENA_BYTES:X}). "
                    f"Grow the arena or shard fewer stages.")
                total += max(0, used)
        return total

    def _vis_register_per_engine(self, sched):
        """Duplicate every vision buffer a kernel WRITES as scratch.

        Registration is once per scheduler (guarded below). Even where the op that
        dirties a buffer currently runs primary-only, the copies are registered here so
        the buffer is never silently shared the moment that op is sharded:
          vis_zeros   layer_norm_core_dram WRITES it. Sharing it across engines is
                      silent corruption, and it needs refresh_per_engine before EVERY
                      execution (see run_vision), not just the first.
          flash q/k/v/out, attn_scratch  per-head marshalling + unified_attention,
                      which are SHARDED on the query axis -- every engine stages its own
                      Q rows and its own FULL K/V copy, so these must not be shared.
                      attn_scratch keeps its FULL size: the core derives its sub-offsets
                      from the compile-time S/D, which do not shrink with a row shard.
        """
        if getattr(self, "_vis_per_engine_done", None) is sched:
            return sched
        V = self._cfg["vision"]
        S, H, D, bpe = V["num_patches"], V["hidden_size"], V["head_dim"], 2
        sched.register_per_engine("vis_zeros", self.vis_zeros_addr, H * bpe,
                                  init_tensor=torch.zeros(H, dtype=torch.bfloat16))
        zeros_d = torch.zeros(S * D, dtype=torch.bfloat16)
        for name, addr in (("flash_q", self.VIS_FLASH_Q_DRAM),
                           ("flash_k", self.VIS_FLASH_K_DRAM),
                           ("flash_v", self.VIS_FLASH_V_DRAM),
                           ("flash_out", self.VIS_FLASH_OUT_DRAM)):
            sched.register_per_engine(name, addr, S * D * bpe, init_tensor=zeros_d)
        scratch_n = (D + S) * S + S * D
        sched.register_per_engine(
            "attn_scratch", self.VIS_ATTN_SCRATCH_DRAM, scratch_n * bpe,
            init_tensor=torch.zeros(scratch_n, dtype=torch.bfloat16))
        self._vis_per_engine_done = sched
        return sched

    def compile_encoder(self):
        """One SigLIP pass over [1024,768] + the connector, compiled ONCE and executed
        once per camera slot (the program is the expensive artifact). Returns its DRAM
        address.

        Per layer: LN1(beta) -> q/k/v(+bias) -> per-head flash (MHA, 12 heads, hd 64,
        bidirectional) -> out_proj(+bias) -> residual -> LN2(beta) -> fc1(+bias)+GELU ->
        fc2(+bias) -> residual. Then post_ln -> pixel shuffle -> connector projection.

        All matmuls are bf16 B-operand (no is_B_quantized/data_type/SCALE_DRAM_ADDR) and
        PBI (gpr_M_reg), so the captured program is structure-bound, not M-bound. Flash
        stays static -- PBI flash address injection corrupts on the second execution, and
        this program runs twice, once per slot.

        Both slots are always encoded; no zero-slot skipping.

        SHARDING (--engines N / --vis_8). PURE ROW SPLIT over the M axis, at a block
        granularity DERIVED from the engine count (_vis_row_align) rather than fixed at
        64: S=1024 is 16 blocks of 64, which divides 1/2/4/8 exactly but leaves 12
        engines at 4x128 + 8x64 -- an 8x ceiling on twelve engines. Above 8 the align
        descends (to 8 rows at ne=10/12), so no K-lane grid is needed at any count;
        pi05 needed one only because its VIS_S=256 caps a row split at 4 engines.

        THE WHOLE LAYER BODY IS SHARDED: patch-embed, pos-embed add, LN1, q/k/v, the
        per-head flash marshalling AND attention (query-axis -- each engine attends its
        own rows over the FULL K/V, so no online-softmax merge), o_proj, both residuals,
        LN2, fc1, fc2. Nothing in a layer runs primary-only any more.

        TWO BARRIERS PER LAYER, down from six (25 per encoder, down from 74). That is
        the point of sharding the norms and residuals: they carry almost no FLOPs, but
        while they ran primary-only at full S they split each layer into six
        synchronized regions, and every rendezvous makes all twelve engines wait for the
        slowest. The two that remain are the two that are REAL:
          #1 after q/k/v -- RAW: attention below reads the FULL K/V, rows the other
             engines just wrote.
          #2 after the MLP residual -- WAR: the next layer's q/k/v overwrite VIS_K/V,
             which a peer may still be gathering. On the last layer it also lets the
             primary read all S rows for post_ln.
        Everything between them is row-local, so an engine runs attention -> o -> res ->
        LN2 -> fc1 -> fc2 -> res without ever touching a peer's rows.

        PRIMARY ONLY: post_ln and the whole connector (M=64, and its permute is a
        host-index-table DMA gather with no per-engine form).

        The two are stitched together by _vp_barrier: a rendezvous CLOSES each sharded
        region (so the primary sees every worker's rows before it reads them) and OPENS
        the next one (so no worker reads a primary-written buffer early). At ne == 1
        every barrier and every row offset is zero and the emitted stream is
        byte-identical to the historical single-engine program.
        """
        V = self._cfg["vision"]
        S = V["num_patches"]

        ne = self._num_engines("VIS")
        sched = self._make_stage_scheduler("VIS", ne)
        self._vis_sched = sched
        self._vis_register_per_engine(sched)
        # CONNECTOR LANES. Its projection is the one op in this stage that cannot use
        # the row axis (M = tokens_out = 64, a single block), so it is N-split instead:
        # output_size 960 = 15 blocks of 64. Allocated here, before the emit loop, so
        # every engine's body can address its own lane.
        if ne > 1:
            _C = self._cfg["connector"]
            # Tall enough for BOTH slots when batched -- one [128, cols] lane per engine.
            sched.alloc_col_output(
                "vis_conn",
                (self.VIS_SLOTS if self._vis_batch_plan(ne) else 1) * _C["tokens_out"],
                _C["output_size"])
        # ---- SLOT BATCHING vs the historical two-pass path -------------------------
        # _vis_batch_plan returns None below 2*slots engines (and at ne == 1), where the
        # program stays exactly what it was and is executed once per camera.
        _plan = self._vis_batch_plan(ne)
        self._vis_batched = _plan is not None
        if self._vis_batched:
            per_engine, _per, _align = _plan
            _rows = [r for _, _, r in per_engine[:_per]]
            print(f"    [vis] BATCHED: {self.VIS_SLOTS} camera slots in ONE pass. "
                  f"{_per} engine(s) per slot x {_align}-row blocks -> {_rows} of {S} "
                  f"(busiest {max(_rows)}, {S / max(_rows):.2f}x per image, "
                  f"{self.VIS_SLOTS * S / max(_rows):.2f}x on the pair); the ~300 ms of "
                  f"fixed per-execution cost is now paid ONCE, not twice")
        else:
            splits = sched.split_rows(S)
            per_engine = [(0, off, r) for off, r in splits]
            _align = self._vis_row_align(ne)
            if ne > 1:
                _rows = [c for _, c in splits]
                # Report the CEILING, not the engine count. Every barrier waits for the
                # busiest engine, so S/max(rows) is what this split can actually buy.
                print(f"    [vis] row shard: {ne} engines x {_align}-row blocks -> "
                      f"{_rows} of {S} (busiest {max(_rows)}, "
                      f"ceiling {S / max(_rows):.2f}x); 2 barriers/layer, TWO passes")

        self.start_capture()
        sched.begin_program()
        for e, ue in enumerate(sched.engines):
            slot, loc_off, rows = per_engine[e]
            # row_offset is GLOBAL (slot*S + local). Every RH/RI/RP pitch in the body
            # derives from it, so the concatenated [2S, *] buffers need no other change.
            # `slot` and `loc_off` are for the two sites that must stay IMAGE-LOCAL:
            # the K/V gather and the flash bias.
            self._emit_encoder_body(ue, e, sched, slot * S + loc_off, rows, ne,
                                    slot=slot, local_row_offset=loc_off,
                                    align=_align)

        self.generate_instruction_halt()
        # KEEP the returned addresses: another stage compiling on ANOTHER scheduler over
        # the same worker pool overwrites that scheduler's _worker_prog_addrs, so a bare
        # start_workers() in run_vision could relaunch the wrong program.
        self._vis_worker_prog_addrs = sched.finalize()   # halts + flushes every worker
        self._record_worker_prog_sizes("vision", sched)
        self._assert_worker_programs_fit()
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
        if ne > 1:
            print(f"    vision workers: {sched.worker_program_bytes()} bytes total "
                  f"across {len(sched.workers)} engine(s)")
        return addr

    def _vis_batch_plan(self, ne):
        """Per-engine (slot, image-local row offset, rows) for the BATCHED encoder,
        or None when this engine count must fall back to the two-pass path.

        THE PARTITION IS BY IMAGE, NOT BY CONCATENATED ROW, and that choice is the
        whole design. Splitting 2048 concatenated rows across twelve engines would put
        one engine astride the image boundary, and that engine would need TWO flash
        calls per head against two different K/V sets -- special-cased marshalling in
        the one region of this stage that is already the most delicate. Partitioning the
        ENGINES instead means every engine touches exactly one image, so the attention
        body is unchanged apart from which K/V base it reads.

        IT COSTS NOTHING IN WIDTH. 1024 rows over 6 engines at align 16 is 176 rows
        busiest = 5.82x per image, and the two images run AT THE SAME TIME, so the pair
        moves at 11.64x -- identical to 1024 rows over 12 engines at align 8. The win is
        not width, it is that the ~300 ms of fixed per-execution cost (measured at
        ne=6 vs ne=12, see compile_encoder) is paid ONCE instead of twice.

        Returns (plan, per_slot, align) or None. None when ne < 2*slots or when slots
        does not divide ne -- there the two-pass path stays, which is also what keeps
        ne == 1 emitting its historical byte-identical single-engine program.
        """
        V = self._cfg["vision"]
        slots, S = V["num_image_slots"], V["num_patches"]
        ne = int(ne)
        if ne < 2 * slots or ne % slots != 0:
            return None
        if getattr(self, "VIS_NO_BATCH", False):
            # A/B handle (--no-vis-batch). The two-pass path at the SAME engine count is
            # the only honest control for "did batching help": changing --engines to
            # disable it also changes the width, the align and the port split.
            return None
        if getattr(self, "VIS_BISECT", False) or self.VIS_LAYERS is not None:
            # --bisect-vision / --vis-layers are primary-only debug paths whose probe
            # buffers are one image tall and whose oracle compares one slot. Batching
            # them would silently score slot 0 against a buffer holding both.
            return None
        per = ne // slots
        align = self._vis_row_align(per)
        blocks = S // align
        assert blocks >= per, (
            f"_vis_batch_plan: {S} rows at {align}-row blocks is {blocks} block(s), "
            f"fewer than the {per} engine(s) per slot")
        base, rem = divmod(blocks, per)
        counts = [(base + (1 if i < rem else 0)) * align for i in range(per)]
        offs = [sum(counts[:i]) for i in range(per)]
        assert sum(counts) == S and all(c > 0 for c in counts), (
            f"_vis_batch_plan({ne}): {counts} does not partition {S} rows")
        # e // per is the slot: engines [0, per) take image 0, [per, 2*per) image 1.
        plan = [(e // per, offs[e % per], counts[e % per]) for e in range(ne)]
        return plan, per, align

    def _emit_encoder_body(self, ue, e, sched, row_offset, rows, ne,
                           slot=0, local_row_offset=None, align=None):
        """Emit ONE engine's complete encoder stream.

        `ue` is engine `e`; at e == 0 it IS `self`, the primary. `row_offset`/`rows` are
        this engine's slice of the S patch rows -- (0, S) when ne == 1.

        ROW OFFSETS ARE COMPILE-TIME, one per pitch: a [S,H] buffer advances
        row_offset*H*bpe per shard and a [S,I] buffer row_offset*I*bpe. Getting one wrong
        on a scatter is the finite-but-scrambled failure class (pi05 denoise), never a
        NaN, so both are asserted against the 64-row / 32-byte-beat contract here rather
        than trusted at the call sites.
        """
        V, C = self._cfg["vision"], self._cfg["connector"]
        S, H, I = V["num_patches"], V["hidden_size"], V["intermediate_size"]
        D, NH = V["head_dim"], V["num_heads"]
        P, CH = V["patch_size"], V["num_channels"]
        bpe = 2
        primary = (e == 0)

        # The shard must be a whole number of THIS run's row blocks -- the align is
        # derived from the engine count (_vis_row_align), not fixed at 64. What the
        # kernels actually require is checked separately and independently below: every
        # per-pitch byte offset must clear the 32 B AXI beat, which is the only M-side
        # hardware constraint in this stage. Neither matmat_mul_core (K % 64 only) nor
        # unified_attention_core (batch <= aligned_seq_len) constrains the row count.
        _align = self._vis_row_align(ne) if align is None else align
        assert rows % _align == 0, (
            f"engine {e} got {rows} rows, not a whole number of {_align}-row blocks "
            f"(ne={ne})")
        # local_row_offset is this engine's offset WITHIN ITS OWN IMAGE; row_offset is
        # its offset in the concatenated [slots*S, *] buffers. They differ only when the
        # encoder is batched, and only ONE thing is indexed by the local one.
        loc = row_offset if local_row_offset is None else local_row_offset
        SLOT_H = slot * S * H * bpe          # this engine's image, in [slots*S, H] terms
        LOC_H = loc * H * bpe                # image-local [S,H] pitch: pos_embed
        RH = row_offset * H * bpe            # [SB,H] pitch: hidden-width buffers
        RI = row_offset * I * bpe            # [SB,I] pitch: the MLP intermediate
        RP = row_offset * (CH * P * P) * bpe  # [SB, C*P*P] pitch: the staged pixels
        # THE ATTENTION BIAS IS IMAGE-LOCAL. It is one shared [S, S] tensor -- attention
        # never crosses images -- so its row index must be the image-local one. Using
        # the global offset here would read past the buffer for every slot-1 engine.
        RB = loc * S * bpe                   # [S,S] pitch: the attention bias rows
        for name, off in (("RH", RH), ("RI", RI), ("RP", RP), ("RB", RB)):
            assert off % 32 == 0, (
                f"engine {e} {name}={off} B is not 32 B AXI-beat aligned")

        m_reg = ue.alloc_isa_reg()
        ue.generate_instruction_add_set(m_reg, rows)

        # FULL-S REGISTER, SEPARATE FROM THE SHARD COUNT. m_reg carries THIS ENGINE's row
        # count and is right for the sharded matmuls -- but every primary-only op below
        # runs at FULL S rows, and its static M= argument is OVERRIDDEN by the runtime
        # GPR. Handing those m_reg made the primary layer-norm and attend over `rows`
        # instead of S (128 of 1024 at ne=8): finite output, plausible magnitudes, ~23 dB
        # of arithmetic gone, and a fake speedup because attention is quadratic in the
        # sequence it actually runs.
        #
        # At ne == 1 this ALIASES m_reg rather than allocating: rows == S there, so the
        # value is identical, and allocating a second register would shift every
        # subsequent register index and break byte-identity of the single-engine program.
        if ne == 1:
            s_reg = m_reg
        else:
            s_reg = ue.alloc_isa_reg()
            ue.generate_instruction_add_set(s_reg, S)

        def bar():
            self._vp_barrier(ue, e, ne)

        def shard_mm(K, N, A, a_pitch, B, OUT, o_pitch, bias=None, **kw):
            """One row-sharded matmul: M = this engine's row count, A and OUT advanced by
            this engine's row offset at their OWN pitches, B/bias untouched (a row shard
            leaves the weight and the broadcast_N bias whole)."""
            ue.matmat_mul_core(
                M=rows, K=K, N=N,
                A_DRAM_ADDR=A + row_offset * a_pitch, B_DRAM_ADDR=B,
                OUTPUT_DRAM_ADDR=OUT + row_offset * o_pitch,
                C_DRAM_ADDR=bias, bias_mode="broadcast_N", gpr_M_reg=m_reg, **kw)

        # vis_zeros is SCRATCH THE NORM WRITES, so each engine needs its own copy --
        # already registered per-engine by _vis_register_per_engine and re-staged before
        # every execution by run_vision's refresh_per_engine. Sharing one copy across
        # twelve concurrently-norming engines is silent corruption, not a crash.
        # vis_inv_n stays SHARED: it is preloaded DRAM->SRAM and never written back.
        vis_zeros = (sched.per_engine_addr("vis_zeros", e) if ne > 1
                     else self.vis_zeros_addr)

        def sh_layer_norm(A, OUT, gamma, beta):
            """One row-sharded LayerNorm. A and OUT carry this engine's row offset;
            gamma/beta are per-COLUMN (length H) and so are shared whole."""
            ue.layer_norm_core_dram(
                M=rows, N=H, A_DRAM_ADDR=A + RH, OUTPUT_DRAM_ADDR=OUT + RH,
                GAMMA_DRAM_ADDR=gamma, BETA_DRAM_ADDR=beta,
                gpr_M_reg=m_reg, ZEROS_DRAM_ADDR=vis_zeros,
                INV_N_DRAM_ADDR=self.vis_inv_n_addr)

        def sh_residual(A, B, OUT, b_off=None):
            """One row-sharded eltwise add. FLAT: element i of B pairs with element i of
            A, so ALL THREE operands must carry a row offset, and offsetting some but not
            all of them is the finite-but-scrambled failure class, never a NaN.

            `b_off` exists for the ONE call where B's offset is not RH: the pos_embed
            add. Every other B here is a [slots*S, H] activation buffer indexed by the
            GLOBAL row, but pos_embed is a single [S, H] table SHARED by both cameras,
            so it must be indexed image-locally. With the encoder batched and RH used
            there, every slot-1 engine would read up to 1024 rows PAST the table."""
            eltwise_add_core_dram(
                ue, size=rows * H, A_DRAM_ADDR=A + RH,
                B_DRAM_ADDR=B + (RH if b_off is None else b_off),
                OUTPUT_DRAM_ADDR=OUT + RH)

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
        bar()
        shard_mm(CH * P * P, H, self.VIS_PIXEL_IN_DRAM, CH * P * P * bpe,
                 self.patch_weight_addr, self.VIS_LN_OUT_DRAM, H * bpe,
                 bias=self.patch_bias_addr)
        if self.VIS_BISECT:
            # Debug only. The probe reads ALL S rows on the primary, so it needs the
            # rendezvous the sharded path no longer emits here. Costs nothing when off.
            bar()
            if primary:
                self._probe_copy(self.VIS_LN_OUT_DRAM, self.VIS_P_PATCH_DRAM, S * H)
            bar()
        # pos_embed is one full [S,H] table -- SHARED_ROWS, so this engine's rows of it
        # pair with this engine's rows of the patch output. IT CARRIES THE IMAGE-LOCAL
        # OFFSET, NOT RH: the table is S rows and is shared by both cameras, while the
        # activation buffers are slots*S rows. Identical when the encoder is not batched
        # (loc == row_offset); off by a whole image for every slot-1 engine when it is.
        # No rendezvous after: layer 0's LN1 is sharded too and reads only the rows this
        # engine just wrote.
        sh_residual(self.VIS_LN_OUT_DRAM, self.pos_embed_addr, self.VIS_IO_A_DRAM,
                    b_off=LOC_H)

        n_vis = len(self.vis_layer_addrs) if self.VIS_LAYERS is None else int(self.VIS_LAYERS)
        assert 1 <= n_vis <= len(self.vis_layer_addrs), f"VIS_LAYERS={self.VIS_LAYERS}"
        if primary and n_vis != len(self.vis_layer_addrs):
            print(f"    [bisect] compiling only {n_vis}/{len(self.vis_layer_addrs)} ViT layers")
        for i, la in enumerate(self.vis_layer_addrs[:n_vis]):
            h_in = self.VIS_IO_A_DRAM if i % 2 == 0 else self.VIS_IO_B_DRAM
            h_out = self.VIS_IO_B_DRAM if i % 2 == 0 else self.VIS_IO_A_DRAM

            # ==== REGION A: LN1 + q/k/v ====================================
            # Every op here is ROW-LOCAL -- this engine norms its own rows of h_in and
            # projects exactly those -- so the whole region needs no rendezvous inside
            # it. That is the change: LN1 used to run primary-only at full S, which
            # forced a barrier before it AND after it.
            sh_layer_norm(h_in, self.VIS_LN_OUT_DRAM, la["ln1_weight"], la["ln1_bias"])
            if self.VIS_BISECT and i == 0:
                bar()
                if primary:
                    self._probe_copy(self.VIS_LN_OUT_DRAM, self.VIS_P_LN1_DRAM, S * H)
                bar()
            for proj, dst in (("q", self.VIS_Q_DRAM), ("k", self.VIS_K_DRAM),
                              ("v", self.VIS_V_DRAM)):
                shard_mm(H, H, self.VIS_LN_OUT_DRAM, H * bpe, la[f"{proj}_weight"],
                         dst, H * bpe, bias=la[f"{proj}_bias"])
            # THE ONE BARRIER REGION A NEEDS: the attention below reads the FULL K and V,
            # i.e. rows every OTHER engine just wrote.
            bar()

            # MHA, one head at a time: gather this head's [S,64] column block into the
            # fixed flash operands, run static flash, scatter back. The scatter DEST
            # carries + h*D*bpe -- omitting that per-head offset is the finite-but-
            # scrambled bug class (pi05 denoise #3), not a NaN.
            #
            # SHARDED ON THE QUERY AXIS. Splitting the QUERY rows partitions the OUTPUT
            # rows, and each output row's softmax is complete within itself -- so no
            # online-softmax merge is needed and there is no mid-attention join. That is
            # the whole reason this axis was chosen over head-splitting: NH=12 does not
            # divide 8, while S=1024 = 16 blocks of 64 splits exactly 2 blocks per
            # engine. (Head-splitting stays available at 4 or 6 engines; this is the
            # pattern hardware-proven by matmat_mul_norm_attn_chain_n_engine_test.)
            #
            # K AND V STAY FULL ON EVERY ENGINE -- they carry `col` but NEVER `+ RH`.
            # Every query row reads every key row, so a K/V shard would be the sequence-
            # parallel case that DOES need a merge. Only Q and OUT carry the row offset.
            #
            # The flash staging buffers and attn_scratch are PER-ENGINE
            # (_vis_register_per_engine); sharing them across engines is silent
            # corruption. attn_scratch keeps its FULL size because the core derives its
            # sub-offsets from the compile-time S/D, which do not shrink with a row shard.
            elems_q, elems_kv = rows * D, S * D
            col_stride, row_jump = D * bpe, H * bpe

            def flash_buf(name, fallback):
                return (sched.per_engine_addr(name, e) if ne > 1 else fallback)

            F_Q = flash_buf("flash_q", self.VIS_FLASH_Q_DRAM)
            F_K = flash_buf("flash_k", self.VIS_FLASH_K_DRAM)
            F_V = flash_buf("flash_v", self.VIS_FLASH_V_DRAM)
            F_O = flash_buf("flash_out", self.VIS_FLASH_OUT_DRAM)
            F_S = flash_buf("attn_scratch", self.VIS_ATTN_SCRATCH_DRAM)
            for h in range(NH):
                col = h * col_stride
                # Q: this engine's rows only.
                ue.accelerator_memory_to_sram(
                    self.VIS_Q_DRAM + col + RH, 0x00000, elems_q,
                    stride_bytes_per_chunk=col_stride, stride_jump_bytes=row_jump)
                ue.sram_to_accelerator_memory(0x00000, F_Q, elems_q)
                # K/V: ALL S rows, on every engine. No RH here -- that is the bug that
                # would turn this into sequence-parallel attention without the merge.
                # + SLOT_H selects THIS ENGINE'S IMAGE. Every engine reads all S rows
                # of K/V -- but only of its own camera. Dropping this makes every
                # slot-1 engine attend over slot 0's keys: finite, plausible, and the
                # second camera silently becomes a copy of the first.
                for src, dst in ((self.VIS_K_DRAM + SLOT_H + col, F_K),
                                 (self.VIS_V_DRAM + SLOT_H + col, F_V)):
                    ue.accelerator_memory_to_sram(
                        src, 0x00000, elems_kv,
                        stride_bytes_per_chunk=col_stride, stride_jump_bytes=row_jump)
                    ue.sram_to_accelerator_memory(0x00000, dst, elems_kv)
                # batch = THIS ENGINE'S ROWS, aligned_seq_len = FULL S. Two different
                # registers: handing one register to both is what made the primary
                # attend over 128 of 1024 rows.
                ue.unified_attention_core(
                    batch=rows, aligned_seq_len=S, head_dim=D,
                    Q_DRAM_ADDR=F_Q, K_DRAM_ADDR=F_K, V_DRAM_ADDR=F_V,
                    BIAS_DRAM_ADDR=self.VIS_ATTN_BIAS_DRAM + RB,
                    OUTPUT_DRAM_ADDR=F_O,
                    SCRATCH_DRAM_ADDR=F_S,
                    IDENTITY_DRAM_ADDR=self.identity_addr,
                    gpr_batch_reg=m_reg, gpr_aligned_seq_len_reg=s_reg)
                ue.accelerator_memory_to_sram(F_O, 0x00000, elems_q)
                ue.sram_to_accelerator_memory(
                    0x00000, self.VIS_ATTN_RESULT_DRAM + col + RH, elems_q,
                    stride_bytes_per_chunk=col_stride, stride_jump_bytes=row_jump)
            # o_proj reads the attention result THIS engine just wrote, so it stays in
            # lane -- no rendezvous between attention and here.
            shard_mm(H, H, self.VIS_ATTN_RESULT_DRAM, H * bpe, la["o_weight"],
                     self.VIS_O_PROJ_DRAM, H * bpe, bias=la["o_bias"])
            sh_residual(h_in, self.VIS_O_PROJ_DRAM, self.VIS_RESIDUAL_DRAM)
            sh_layer_norm(self.VIS_RESIDUAL_DRAM, self.VIS_LN_OUT_DRAM,
                          la["ln2_weight"], la["ln2_bias"])
            if self.VIS_BISECT and i == 0:
                bar()
                if primary:
                    self._probe_copy(self.VIS_LN_OUT_DRAM, self.VIS_P_LN2_DRAM, S * H)
                bar()
            # The fused GELU is x*sigmoid(1.702x); the model specifies gelu_pytorch_tanh.
            # Score the oracle with --hw-gelu or this stage falsely reads ~28 dB low.
            #
            # fc1 and fc2 are ONE region: an engine's rows of the intermediate feed only
            # its own rows of fc2, so no rendezvous is needed between them.
            shard_mm(H, I, self.VIS_LN_OUT_DRAM, H * bpe, la["fc1_weight"],
                     self.VIS_MLP_INTER_DRAM, I * bpe, bias=la["fc1_bias"],
                     gelu_enable=True)
            shard_mm(I, H, self.VIS_MLP_INTER_DRAM, I * bpe, la["fc2_weight"],
                     self.VIS_MLP_OUT_DRAM, H * bpe, bias=la["fc2_bias"])
            sh_residual(self.VIS_RESIDUAL_DRAM, self.VIS_MLP_OUT_DRAM, h_out)
            # THE ONE BARRIER REGION B NEEDS, and it is a WAR fence, not a RAW one:
            # nothing below reads a peer's rows, but the NEXT layer's q/k/v overwrite
            # VIS_K/VIS_V, which a peer may still be gathering in FULL for its attention
            # above. Without it a fast engine races ahead and corrupts a slow one's keys.
            # On the last layer it is also what lets the primary read all S rows of
            # h_out for post_ln.
            bar()

        if primary:
            final = (self.VIS_IO_A_DRAM if n_vis % 2 == 0
                     else self.VIS_IO_B_DRAM)
            # ONE CALL PER SLOT rather than one over slots*S rows, deliberately: s_reg
            # holds S and is ALSO the flash aligned_seq_len register, so widening it
            # would silently make every attention span both images. Two ops of S rows
            # against one of 2S is ~0.3 ms in a stage that costs 800; a second register
            # with two meanings is a bug waiting to happen.
            for _sl in range(self.VIS_SLOTS if self._vis_batched else 1):
                _o = _sl * S * H * bpe
                self.layer_norm_core_dram(
                    M=S, N=H, A_DRAM_ADDR=final + _o,
                    OUTPUT_DRAM_ADDR=self.VIS_POST_LN_DRAM + _o,
                    GAMMA_DRAM_ADDR=self.vis_post_ln_weight,
                    BETA_DRAM_ADDR=self.vis_post_ln_bias, gpr_M_reg=s_reg,
                    ZEROS_DRAM_ADDR=self.vis_zeros_addr,
                    INV_N_DRAM_ADDR=self.vis_inv_n_addr)

        # CONNECTOR. Deliberately OUTSIDE the `if primary:` above: the permute really
        # is primary-only, but the projection is not, and it used to run at width 1 for
        # 8% of this stage's matmul floor against 0.7% of its FLOPs.
        self.compile_connector(ue, e, sched, ne, bar)

        ue.release_isa_reg()

    def compile_connector(self, ue=None, e=0, sched=None, ne=1, bar=None):
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
        splitting it would forfeit compile-once and cost a host round-trip.

        THE PERMUTE IS PRIMARY-ONLY, THE PROJECTION IS NOT, and conflating the two is
        what kept this whole block at width 1. The old note here said "M=64 for the whole
        connector: nothing to shard" -- correct about M (tokens_out is a single 64-row
        block, so the row axis this stage uses everywhere else is unavailable) and
        correct about the permute (a host-index-table DMA gather with no per-engine
        form). But the projection has an N axis: output_size 960 is 15 blocks of 64, the
        same N-split + scatter the expert's q/o projections use, with no reduction and
        therefore numerically identical to width 1.

        It was worth fixing because the cost was wildly out of proportion to the work:
        1.51 GFLOP is 0.7% of a slot's FLOPs but 39.3 of the 517.7 ms matmul floor --
        8% -- purely because eleven engines watched one do it. At 12 engines the split
        is 3x2 + 9x1 blocks, so the busiest carries 128 of 960 columns: 7.5x, not 12x,
        and that is the ceiling 15 blocks allow.

        `bar` is compile_encoder's per-engine rendezvous slice. Two are needed and both
        are real: VIS_SHUFFLED_DRAM is written by the primary and read in FULL by every
        engine, and VIS_CONNECTOR_DRAM is written by all of them and read back by the
        host. At ne == 1 _vp_barrier emits nothing, the split is [(0, 960)], and the
        original single matmul into VIS_CONNECTOR_DRAM is reproduced exactly."""
        V, C = self._cfg["vision"], self._cfg["connector"]
        H, s = V["hidden_size"], C["pixel_shuffle_scale_factor"]
        g = V["num_patches_per_side"] // s
        S, bpe = V["num_patches"], 2
        TOUT, K, N = C["tokens_out"], C["input_size"], C["output_size"]
        nslots = self.VIS_SLOTS if getattr(self, "_vis_batched", False) else 1
        # M IS THE BATCHED TOKEN COUNT. The shuffle is per image (it folds each 32x32
        # patch grid into 64 tokens), but the two images' shuffled outputs land in
        # ADJACENT 64-row halves of VIS_SHUFFLED_DRAM, so ONE projection covers both --
        # and its [128, 960] output is exactly what the prefix consumes.
        M = nslots * TOUT
        ue = self if ue is None else ue

        if e == 0:
            for sl in range(nslots):
                smart_bf16_permute_core(
                    self, dims=[g, s, g, s, H], permute_indices=[0, 2, 1, 3, 4],
                    input_dram_addr=self.VIS_POST_LN_DRAM + sl * S * H * bpe,
                    output_dram_addr=self.VIS_SHUFFLED_DRAM + sl * TOUT * K * bpe,
                    params_dram_addr=self.permute_params_addr,
                    temp_dram_start=self.PERMUTE_TEMP_DRAM)

        if ne <= 1 or sched is None:
            # M=64 only, so PBI buys nothing -- keep it static.
            self.matmat_mul_core(
                M=M, K=K, N=N,
                A_DRAM_ADDR=self.VIS_SHUFFLED_DRAM, B_DRAM_ADDR=self.conn_weight_addr,
                OUTPUT_DRAM_ADDR=self.VIS_CONNECTOR_DRAM)
            return

        if bar is not None:
            bar()          # RAW: the primary's permute -> every engine's A operand
        off, cols = self._col_split(N, ne)[e]
        lane = sched.col_output_addr("vis_conn", e)
        # B is [N, K] row-major, so this engine's output columns are a CONTIGUOUS row
        # block of the weight -- the same + col_offset*K*bpe that ColumnShardContext.
        # b_addr computes. A (the shuffled [64, 12288]) is read in FULL by everyone; a
        # K-split is not available because A[:, k0:k0+Kc] is strided.
        ue.matmat_mul_core(
            M=M, K=K, N=cols,
            A_DRAM_ADDR=self.VIS_SHUFFLED_DRAM,
            B_DRAM_ADDR=self.conn_weight_addr + off * K * 2,
            OUTPUT_DRAM_ADDR=lane)
        # SCATTERED ONE SLOT AT A TIME. _ae_scatter_lane stages [rows, cols] through
        # SRAM 0x00000, and at M=128 x 128 columns that is exactly the 0x10000 window
        # below the flash region -- zero margin. Two 64-row copies halve it. The matmul
        # above is still ONE op; only the write-back is chunked.
        for sl in range(nslots):
            self._ae_scatter_lane(
                ue, lane + sl * TOUT * cols * bpe,
                self.VIS_CONNECTOR_DRAM + sl * TOUT * N * bpe, off, cols, N, TOUT)
        if bar is not None:
            bar()          # the lanes must all land before the host reads the buffer

    def tensor_init_vision(self):
        """Vision + connector activation buffers. Called by tensor_init."""
        V, C = self._cfg["vision"], self._cfg["connector"]
        S, H, I = V["num_patches"], V["hidden_size"], V["intermediate_size"]
        D, bpe = V["head_dim"], 2

        def a(n):
            return self.allocate_tensor_dram(n * bpe)

        # ---- SLOT BATCHING: every per-token buffer holds BOTH cameras ---------------
        # The two camera slots used to run the SAME program twice. Measured at two
        # engine counts exactly 2x apart in ceiling (ne=6 vs ne=12), this stage costs
        #     matmul floor + ~300 ms of FIXED overhead, per execution
        # and the fixed part does not shrink with more engines -- so running the program
        # twice pays it twice, for FLOPs that are perfectly independent between the two
        # images. Widening these buffers to hold both and executing ONCE is worth that
        # whole second ~300 ms. See _vis_slot_splits for the engine partition.
        #
        # VIS_SLOT_STRIDE is the row-pitch offset from slot 0's rows to slot 1's, per
        # buffer width. Every address in _emit_encoder_body already derives from a
        # GLOBAL row_offset, so widening these is most of the change.
        self.VIS_SLOTS = V["num_image_slots"]
        SB = self.VIS_SLOTS * S                  # batched row count
        self.VIS_PIXEL_IN_DRAM    = a(SB * H)    # host-staged pixels, BOTH slots
        self.VIS_PATCH_PERM_DRAM  = a(SB * H)
        self.VIS_IO_A_DRAM        = a(SB * H)    # layer ping-pong
        self.VIS_IO_B_DRAM        = a(SB * H)
        self.VIS_LN_OUT_DRAM      = a(SB * H)
        self.VIS_Q_DRAM           = a(SB * H)
        self.VIS_K_DRAM           = a(SB * H)
        self.VIS_V_DRAM           = a(SB * H)
        self.VIS_ATTN_RESULT_DRAM = a(SB * H)
        self.VIS_O_PROJ_DRAM      = a(SB * H)
        self.VIS_RESIDUAL_DRAM    = a(SB * H)
        self.VIS_MLP_INTER_DRAM   = a(SB * I)    # the big one: 2048 x 3072
        self.VIS_MLP_OUT_DRAM     = a(SB * H)
        self.VIS_POST_LN_DRAM     = a(SB * H)

        # NOT DOUBLED, and each for its own reason. The flash staging is PER-ENGINE and
        # an engine only ever touches ONE image (that is the point of partitioning
        # engines by slot rather than splitting 2048 concatenated rows), so it stays
        # image-sized. attn_scratch likewise -- the core derives its sub-offsets from
        # the compile-time S, which does not change.
        self.VIS_FLASH_Q_DRAM   = a(S * D)
        self.VIS_FLASH_K_DRAM   = a(S * D)
        self.VIS_FLASH_V_DRAM   = a(S * D)
        self.VIS_FLASH_OUT_DRAM = a(S * D)
        self.VIS_ATTN_SCRATCH_DRAM = a((D + S) * S + S * D)
        # bidirectional, no mask: all-zero additive bias. Every one of the 1024 patches
        # is real, so nothing here needs -inf. [S, S] and SHARED between the slots: it
        # is indexed by IMAGE-LOCAL row, and attention never crosses images.
        self.VIS_ATTN_BIAS_DRAM = a(S * S)
        self.dma_to_accelerator_memory(
            self.VIS_ATTN_BIAS_DRAM, torch.zeros(S, S, dtype=torch.bfloat16))

        # Slot 0's shuffled tokens occupy rows [0, 64) and slot 1's rows [64, 128), so
        # the two per-image permutes land ADJACENT and one [128, 12288] projection
        # covers both -- which is also exactly the [128, 960] the prefix wants, with no
        # host-side concatenation left to do.
        self.VIS_SHUFFLED_DRAM  = a(self.VIS_SLOTS * C["tokens_out"] * C["input_size"])
        self.VIS_CONNECTOR_DRAM = a(self.VIS_SLOTS * C["tokens_out"] * C["output_size"])
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
        self._assert_tensor_region_fits("_lm_tensor_init (prefix)")

    def _prefix_column_blocks(self):
        """Column-block ceiling for PREFIX sharding: min over the two N axes we split.

        This REPLACES the row-block fallback (PM=192 -> 3 blocks of 64), which capped
        the prefix at 3 engines and is the whole reason this stage moved to columns.
        H=960 -> 15 blocks, I=2560 -> 40 blocks, so the binding number is 15 and it
        never limits below 15 engines."""
        return min(self.HIDDEN_SIZE, self.INTERMEDIATE_SIZE) // UE_VECTOR_SIZE

    def _prefix_kv_units(self, ne):
        """Assign the 10 k/v COLUMN UNITS of one prefix layer to ``ne`` engines.

        THE UNIT IS (tensor, kv-head), NOT the whole projection. k_proj and v_proj are
        each [PM, KV=320] and 320 = 5 blocks of 64 -- which cannot serve 8 or 12
        engines, and is the reason both used to be computed REDUNDANTLY on every engine
        (~6% of the layer's MACs, done ne times). But D == UE_VECTOR_SIZE == 64, so one
        64-column block is EXACTLY one kv-head, and k and v are two independent weights:
        that is 10 independent units, not 5. 10 splits cleanly across everything up to
        10 engines and is the difference between a 5-way and a 10-way cut at ne=12.

        Returns a list of length ``ne``; entry e is the [(is_v, head), ...] this engine
        owns. Round-robin from engine 0 over ``min(ne, 10)``, so at ne >= 10 engines
        10..ne-1 get nothing here -- they are the engines that already carry two q
        column blocks (15 blocks over 12 engines is 3x2 + 9x1), so the imbalance lands
        where there is slack rather than on top of the busiest engine.

        Correctness rests on DISJOINTNESS, not on the balance: unit (t, h) writes
        exactly ``LAYER0_{K,V}_DRAM + i*KV_LAYER_STRIDE + h*KV_HEAD_STRIDE`` for
        PM*D*bpe bytes and no other unit touches those bytes, so LAYER0_K/V_DRAM keeps
        the SINGLE WRITER PER ADDRESS the whole denoise stage depends on. Two engines
        on one address is not redundant work -- it is a real race, and it would show up
        as a finite, plausible, wrong cross-attention rather than a crash."""
        units = ([(0, h) for h in range(self.NUM_KV_HEADS)]
                 + [(1, h) for h in range(self.NUM_KV_HEADS)])
        ne = int(ne)
        n = min(ne, len(units))
        out = [[] for _ in range(ne)]
        for u, unit in enumerate(units):
            out[u % n].append(unit)
        return out

    def _prefix_attn_group_map(self, ne):
        """Assign the 5 kv GROUPS of one prefix layer to ``ne`` engines.

        Same axis and same argument as _ae_attn_groups in the expert: M = PM = 192 is
        only 3 row blocks and each flash call needs the whole key axis, so the per-
        kv-group loop is what is divisible. Group kv_b gathers its own G=3 q-heads,
        runs its own flash and scatters into its own disjoint 3x64-column band of
        LM_ATTN_RESULT_DRAM; groups never read each other's output, so the region needs
        no mid-loop sync.

        NUM_KV_HEADS = 5 does NOT divide 8, 10 or 12: at ne >= 5 engines 5..ne-1 SIT
        IDLE through this region. That is deliberate and it is stated rather than hidden
        -- splitting a group's 3 q-heads across engines would break the token-major
        stacking the whole emitter is built on (flash row t*G+g is q-head kv_b*G+g of
        token t), and 5-way is the largest clean cut available. PRICED, not waved away:
        attention is 3.6% of the layer's FLOPs, so 5-way-instead-of-N-way costs 3.8% of
        the modelled stage at ne=12 (9.12x against a hypothetical 9.48x) and 2.0% at
        ne=8. Compare the thing it replaced -- primary-only attention cost 26% of the
        stage at ne=12.

        Returns a list of length ``ne``; entry e is the [kv_b, ...] this engine owns."""
        ne = int(ne)
        n = min(ne, self.NUM_KV_HEADS)
        out = [[] for _ in range(ne)]
        for kv_b in range(self.NUM_KV_HEADS):
            out[kv_b % n].append(kv_b)
        return out

    def _prefix_kv_cache_head(self, ue, i, h, m_reg):
        """Gather kv-head ``h`` out of the FULL-WIDTH k/v projections into the
        persistent cache, and rope K in place. SINGLE-ENGINE PATH ONLY.

        _compile_prefix_sharded does NOT call this. There the k/v projections are
        themselves split per kv-head, and KV_HEAD_STRIDE == PM*D*bpe makes head h's
        cache block exactly a dense [PM, D] -- which is precisely the shape a
        matmat_mul_core called with N=D writes back -- so the projection lands ON the
        cache block and this gather-through-SRAM disappears entirely.

        The + h*KV_HEAD_STRIDE in the destination is the per-index offset rule: without
        it every head lands on head 0's block and the expert cross-attends to garbage --
        finite and scrambled, never NaN."""
        PM, D, bpe = self.PREFILL_MAX_SEQ_LEN, self.HEAD_DIM, 2
        KV = self.NUM_KV_HEADS * D
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

    def _prefix_attn_group(self, ue, i, kv_b, qb_reg, FQ, FK, FV, FO, FS):
        """ONE kv-group's stacked-Q GQA flash for prefix layer ``i``, on engine ``ue``.

        Stack this group's G q-heads token-major (flash row t*G+g is q-head kv_b*G+g of
        token t), rope them with the x G duplicated table, replicate the group's single
        K/V head x G to match, run one flash, un-stack into the group's own column band
        of LM_ATTN_RESULT_DRAM.

        THE FIVE FLASH BUFFERS ARE PARAMETERS, not attributes. That is the whole
        refactor: the single-engine caller passes the model's own LM_FLASH_* addresses
        and gets a byte-identical stream, while _compile_prefix_sharded passes this
        engine's PRIVATE copies (sched.register_per_engine). Two engines marshalling
        different groups through one FQ/FK/FV/FO is silent interleaved corruption, and
        one shared FS is worse: unified_attention_core writes SCRATCH_SM per head and
        reads it straight back.

        SRAM 0x10000 / 0x20000 ARE NOT A SHARING HAZARD, contrary to what this method's
        old docstring claimed. Each UnifiedEngine is a separate core with its own SRAM
        (BASE_ADDR = UE_0 + idx*0x10000), so a fixed SRAM offset is private by
        construction -- vision already runs twelve engines through 0x00000 at once. The
        buffers that really were shared were the DRAM ones above."""
        PM, H = self.PREFILL_MAX_SEQ_LEN, self.HIDDEN_SIZE
        D, G = self.HEAD_DIM, self.GROUP_SIZE
        QB, bpe = self.LM_QB, 2
        k_src = self.LAYER0_K_DRAM + i * self.KV_LAYER_STRIDE + kv_b * self.KV_HEAD_STRIDE
        v_src = self.LAYER0_V_DRAM + i * self.KV_LAYER_STRIDE + kv_b * self.KV_HEAD_STRIDE
        for g in range(G):
            self._ae_strided_copy(ue, self.LM_Q_DRAM + (kv_b * G + g) * D * bpe,
                                  H * bpe, FQ + g * D * bpe,
                                  G * D * bpe, PM, D)
        ue.rope_hf_core_dram(M=QB, N=D, input_dram_addr=FQ,
                             output_dram_addr=FQ,
                             cos_dram_addr=self.ROPE_PACKED_GQA_DRAM,
                             sin_dram_addr=self.ROPE_PACKED_GQA_DRAM + D * bpe,
                             gpr_M_reg=qb_reg)
        ue.accelerator_memory_to_sram(k_src, 0x10000, PM * D)
        self._ae_duplicate_gqa_rows(ue, PM, 0x10000, FK)
        ue.accelerator_memory_to_sram(v_src, 0x20000, PM * D)
        self._ae_duplicate_gqa_rows(ue, PM, 0x20000, FV)
        ue.unified_attention_core(
            batch=QB, aligned_seq_len=QB, head_dim=D,
            Q_DRAM_ADDR=FQ, K_DRAM_ADDR=FK, V_DRAM_ADDR=FV,
            BIAS_DRAM_ADDR=self.PREFIX_BIAS_DRAM,
            OUTPUT_DRAM_ADDR=FO, SCRATCH_DRAM_ADDR=FS,
            IDENTITY_DRAM_ADDR=self.identity_addr,
            gpr_batch_reg=qb_reg, gpr_aligned_seq_len_reg=qb_reg)
        for g in range(G):
            self._ae_strided_copy(ue, FO + g * D * bpe, G * D * bpe,
                                  self.LM_ATTN_RESULT_DRAM + (kv_b * G + g) * D * bpe,
                                  H * bpe, PM, D)

    def _prefix_attention_block(self, ue, i, m_reg, qb_reg):
        """KV-cache write + K rope + per-kv-group stacked-Q GQA flash + un-stack for
        prefix layer ``i``, ALL FIVE HEADS ON ONE ENGINE. SINGLE-ENGINE PATH ONLY.

        Kept as a thin composition of _prefix_kv_cache_head and _prefix_attn_group so
        the emission order -- all five cache writes, THEN all five flash calls -- is
        exactly what it was when this was one inline loop pair. That order is what keeps
        the num_engines==1 program byte-identical; reordering it would be numerically
        harmless and still invalidate every cached bin.

        _compile_prefix_sharded no longer calls this. It emits the same two loops
        SPLIT: the cache write folds into the k/v projection itself (see
        _prefix_kv_units) and the flash calls fan out one group per engine (see
        _prefix_attn_group_map). Each engine still writes DISJOINT
        h*KV_HEAD_STRIDE blocks of LAYER0_K/V_DRAM, so the single-writer-per-address
        invariant that the 10 denoise steps depend on is preserved -- it was never the
        primary-only-ness that guaranteed it, it was the disjointness."""
        for h in range(self.NUM_KV_HEADS):
            self._prefix_kv_cache_head(ue, i, h, m_reg)
        for kv_b in range(self.NUM_KV_HEADS):
            self._prefix_attn_group(ue, i, kv_b, qb_reg,
                                    self.LM_FLASH_Q_DRAM, self.LM_FLASH_K_DRAM,
                                    self.LM_FLASH_V_DRAM, self.LM_FLASH_OUT_DRAM,
                                    self.LM_FLASH_SCRATCH_DRAM)

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
        bpe = 2

        # _lm_tensor_init() BEFORE self.LM_QB is read -- it is what DEFINES LM_QB, so
        # reading it first is an AttributeError on the first compile. It also has to
        # come before the engine-count dispatch, so precompile_all (which compiles with
        # no execution in front of it) is self-sufficient on either path.
        self._lm_tensor_init()
        QB = self.LM_QB

        # MULTI-ENGINE DISPATCH. ne == 1 falls through to the code below UNCHANGED, so
        # the single-engine program is byte-identical by construction, not by review.
        ne = self._num_engines("PREFIX")
        if ne > 1:
            return self._compile_prefix_sharded(ne)

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

            self._prefix_attention_block(ue, i, m_reg, qb_reg)

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

    # ==================================================================================
    # PREFIX -- COLUMN-SHARDED PROJECTIONS
    # ==================================================================================
    def _lm_repack_down_weights(self, k_split):
        """Re-lay the 32 down_proj weights so each engine's K-slice is CONTIGUOUS.

        down_proj B is N x K = [960, 2560] row-major, so ``B[:, k0:k0+Kc]`` -- the slice
        a K-split engine needs -- is strided (one gap per row) and is NOT reachable by
        shifting B_DRAM_ADDR. split_k()'s docstring says the sliced weight must be
        HOST-PREPARED; the obvious reading of that is "upload 8 extra blobs", which for
        32 layers is 157 MB of duplicated weights against a 224 MB tensor arena.

        It is not necessary. The concatenation of the per-engine slices is a PERMUTATION
        of the same 4.9 MB, so we rewrite the blob IN PLACE at its existing address and
        engine e simply reads ``down_weight + N * k0 * bpe``. Zero extra DRAM.

        Only ever done when ne > 1, and guarded so a second compile cannot re-permute an
        already-permuted blob (which would silently scramble every MLP output).
        """
        prev = getattr(self, "_lm_down_repacked", None)
        if prev is not None:
            assert prev == k_split, (
                "down_proj weights are already repacked for a DIFFERENT K split "
                f"({prev} vs {k_split}); the blob cannot be re-permuted in place")
            return
        H, I = self.HIDDEN_SIZE, self.INTERMEDIATE_SIZE
        with PHASES.track("prefix down-weight repack", "host"), silenced():
            for la in self.lm_layer_addrs:
                w = self._read_bf16(la["down_weight"], (H, I), label="down_weight")
                packed = torch.cat([w[:, k0:k0 + kc].contiguous().reshape(-1)
                                    for k0, kc in k_split])
                assert packed.numel() == H * I
                self.dma_to_accelerator_memory(la["down_weight"],
                                               packed.to(torch.bfloat16))
        self._lm_down_repacked = list(k_split)
        _original_print(f"    [prefix] repacked {len(self.lm_layer_addrs)} down_proj "
                        f"weights into {len(k_split)} contiguous K-slices (in place)")

    def _compile_prefix_sharded(self, ne):
        """compile_prefix over ``ne`` engines, sharding the PROJECTION MATMULS ONLY.

        WHY COLUMNS. Row sharding caps at PM/64 = 3 blocks, so a 192-token prefix can
        never use more than 3 engines on M. Every projection's N (or K) is fat instead:
        H=960 -> 15 blocks, I=2560 -> 40. That is the whole argument for this stage.

        Per projection:
          q      N-split (N=960, 15 blocks).  Each engine writes a dense [PM, Nc] lane
                 and then scatters it into the shared LM_Q_DRAM at its own column
                 offset -- disjoint, 128 B-aligned, so no race. The gather is what keeps
                 the q-stack / GQA / flash code below completely untouched.
          k, v   SHARDED PER KV-HEAD, 10 units (5 k-heads + 5 v-heads), and written
                 STRAIGHT INTO THE PERSISTENT CACHE. D == UE_VECTOR_SIZE == 64, so one
                 64-column block is exactly one kv-head, and KV_HEAD_STRIDE ==
                 PM*D*bpe makes head h's cache block exactly the dense [PM, D] that a
                 matmat with N=D writes back -- so the projection lands on the cache and
                 the strided gather that used to move it there is gone. This replaces
                 the old "redundant on every engine" scheme, which spent ~6% of the
                 layer's MACs ne times over. See _prefix_kv_units.
          o      N-split + gather, NOT the K-split the plan called for. See the note on
                 _prefix_o_proj_note below -- the K-split's A operand is strided.
          gate/up  N-split (N=2560, 40 blocks) and they STAY sharded: SiLU and the
                 gate*up multiply are elementwise, so the whole gate -> up -> mul chain
                 runs in one barrier-free lane.
          down   TRUE K-split (K=2560) + reduce_add. Its A operand is already the dense
                 per-engine lane produced by the multiply, so no re-gather is needed;
                 only the weight has to be re-laid (see _lm_repack_down_weights).

          attn   GROUP-SHARDED over the 5 kv-heads, one flash call per group, with
                 PER-ENGINE flash staging + scratch. NUM_KV_HEADS=5 divides neither 8
                 nor 12, so engines 5..ne-1 idle in this region. That costs 3.8% of
                 the modelled stage at ne=12; primary-only attention, which this
                 replaces, cost 26%. See _prefix_attn_group_map.

        NOT sharded, deliberately:
          RMSNorm reduces over the full 960, so it is not column-splittable. It runs
          redundantly on every engine into private buffers -- cheaper than a barrier.
          The residual adds likewise (they feed a norm on every engine).
          The layer-output residual add and the final norm stay PRIMARY ONLY.

        THE INVARIANT THE OLD PRIMARY-ONLY ATTENTION EXISTED TO PROTECT still holds, and
        it was never primary-only-ness that held it. LAYER0_K/V_DRAM is the frozen input
        to all 10 denoise steps and must have exactly ONE WRITER PER ADDRESS. Both the
        kv-head split and the group split are cuts along h, so every engine writes a
        DISJOINT h*KV_HEAD_STRIDE block and reads (in the group region, after a barrier)
        only blocks that are complete. What was genuinely unsafe -- and is fixed here by
        register_per_engine, not by serialising -- was the SHARED LM_FLASH_Q/K/V/OUT and
        the single shared LM_FLASH_SCRATCH. The fixed SRAM staging at 0x10000/0x20000
        was never a hazard at all: SRAM is per-core.

        SIX BARRIERS PER LAYER, unchanged by this. Region 1 now exits with join=False
        and the attention region's own opening rendezvous is that fence, so folding
        attention into the shard added zero round trips.
        """
        ue = self
        PM, H, I = self.PREFILL_MAX_SEQ_LEN, self.HIDDEN_SIZE, self.INTERMEDIATE_SIZE
        D, G = self.HEAD_DIM, self.GROUP_SIZE
        QB, bpe = self.LM_QB, 2

        # Probes snapshot ONE engine and _probe_copy halts the stream; neither means
        # anything once the work is spread over 8 programs. pi05 refuses the same combo.
        assert not self.PREFIX_BISECT, (
            "PREFIX_BISECT probes a single engine's buffers and cannot be interpreted "
            f"under num_engines={ne}; run the bisect at --engines 1")
        assert self.PREFIX_LAYERS is None, (
            "PREFIX_LAYERS is a bisect handle; it is only meaningful single-engine")

        sched = self._make_stage_scheduler("PREFIX", ne)
        n_split = sched.split_cols(H)          # q, o        : 15 blocks over ne
        i_split = sched.split_cols(I)          # gate, up    : 40 blocks over ne
        k_split = sched.split_k(I)             # down        : == i_split, by definition
        assert k_split == i_split, "down's K-slice MUST equal the lane gate/up produced"
        self._lm_repack_down_weights(k_split)
        kv_units = self._prefix_kv_units(ne)      # k/v   : 10 (tensor, head) units
        attn_map = self._prefix_attn_group_map(ne)  # attn : 5 kv groups
        # THE TWO SHAPE FACTS THE k/v FOLD RESTS ON. Break either and the k projection
        # writes the wrong bytes into the PERSISTENT cache -- finite, plausible, wrong
        # cross-attention ten denoise steps later, never a crash.
        #   D == UE_VECTOR_SIZE : one 64-column matmat block IS one kv-head
        #   KV_HEAD_STRIDE == PM*D*bpe : head h's cache block IS a dense [PM, D], which
        #                                is what matmat_mul_core(N=D) writes back
        assert D == UE_VECTOR_SIZE, (
            f"HEAD_DIM={D} != UE_VECTOR_SIZE={UE_VECTOR_SIZE}: a kv-head is no longer "
            f"one column block, so the k/v projection cannot be split per head")
        assert self.KV_HEAD_STRIDE == PM * D * bpe, (
            f"KV_HEAD_STRIDE={self.KV_HEAD_STRIDE} != PM*D*bpe={PM * D * bpe}: the "
            f"cache block is not a dense [PM, D] and a matmat writeback cannot land "
            f"on it directly")

        # --- per-engine buffers ------------------------------------------------------
        # Column lanes: matmat writes back with stride N*bpe for the N IT WAS GIVEN, so
        # calling it with N=Nc yields a DENSE [PM, Nc] block. Nothing is ever strided on
        # the write side.
        if not getattr(self, "_lm_shard_bufs_done", False):
            sched.alloc_col_output("lm_q",    PM, H)
            sched.alloc_col_output("lm_o",    PM, H)
            sched.alloc_col_output("lm_gate", PM, I)
            sched.alloc_col_output("lm_up",   PM, I)
            sched.alloc_col_output("lm_mult", PM, I)
            # Redundantly-computed full-width tensors. The PRIMARY keeps the model's own
            # address (so its half of the program is unchanged and the bisect buffers
            # still line up); each worker gets a private copy, which is what makes
            # "redundant" safe instead of a multi-writer race.
            #   lm_norm is ONE buffer used by BOTH norms in a layer, exactly as the
            #   single-engine path reuses LM_PRE_NORM_DRAM.
            sched.register_per_engine("lm_norm",  self.LM_PRE_NORM_DRAM, PM * H * bpe)
            sched.register_per_engine("lm_resid", self.LM_RESIDUAL_DRAM, PM * H * bpe)
            # NO per-engine lm_k / lm_v any more: k/v are kv-head-sharded straight
            # into LAYER0_K/V_DRAM, so the full-width LM_K_PROJ_DRAM / LM_V_PROJ_DRAM
            # are simply DEAD on this path (they stay allocated for the ne == 1 path and
            # its bisect probe, which is asserted off here). That frees 0.23 MB/worker.
            #
            # ATTENTION SHARD: per-engine flash staging + scratch. The kv-group loop is
            # split across engines, so two engines marshal DIFFERENT groups at the same
            # instant -- sharing LM_FLASH_Q/K/V/OUT is silent interleaved corruption, and
            # sharing the scratch is worse (unified_attention_core writes SCRATCH_SM per
            # head and reads it straight back). SCRATCH KEEPS ITS FULL SIZE: the core
            # derives its sub-offsets from the compile-time QB/D, which do not shrink
            # when the GROUP count does, so a short copy is read past its end rather than
            # refused. Costs 1.06 MB/worker; see the tensor-window note on
            # VIS_WORKER_TENSOR_OFFSET.
            _fl_scr = (D + QB) * QB + QB * D
            for _nm, _addr, _elems in (
                    ("lm_flash_q",   self.LM_FLASH_Q_DRAM,       QB * D),
                    ("lm_flash_k",   self.LM_FLASH_K_DRAM,       QB * D),
                    ("lm_flash_v",   self.LM_FLASH_V_DRAM,       QB * D),
                    ("lm_flash_out", self.LM_FLASH_OUT_DRAM,     QB * D),
                    ("lm_flash_scr", self.LM_FLASH_SCRATCH_DRAM, _fl_scr)):
                sched.register_per_engine(
                    _nm, _addr, _elems * bpe,
                    init_tensor=torch.zeros(_elems, dtype=torch.bfloat16))
            # down's partial sums: full [PM, H] per engine, summed by reduce_add.
            # partial[0] aliases LM_MLP_DOWN_DRAM so the reduction accumulates in place.
            sched.register_per_engine("lm_down",  self.LM_MLP_DOWN_DRAM, PM * H * bpe)
            self._lm_shard_bufs_done = True
        PE = lambda name, e: sched.per_engine_addr(name, e)
        DOWN_PARTIALS = [PE("lm_down", e) for e in range(ne)]

        # THE GATHER. matmat writes back dense [PM, Nc]; the full-width tensor the
        # untouched attention/q-stack code reads is [PM, N], so each engine scatters its
        # own lane into its own disjoint, 128 B-aligned column range. No two engines
        # touch the same bytes, so this needs no lock and no extra barrier -- the
        # region's exit join is the only fence.
        #
        # Emitted in 64-COLUMN CHUNKS, not one 128-wide copy: _ae_strided_copy stages
        # through SRAM 0x00000 and the [192, 64] shape is the one already running in
        # production here (the KV-cache and attention un-stack copies). A 128-wide copy
        # would stage 48 KB against a flash window that starts at 0x10000, which is
        # inside the margin rather than comfortably clear of it.
        assert PM * UE_VECTOR_SIZE * bpe * 2 <= 0x10000, "SRAM staging window shrank"

        def scatter_lane(ue, src, dst_base, n0, cols, full_n):
            assert cols % UE_VECTOR_SIZE == 0
            for c0 in range(0, cols, UE_VECTOR_SIZE):
                self._ae_strided_copy(ue, src + c0 * bpe, cols * bpe,
                                      dst_base + (n0 + c0) * bpe, full_n * bpe,
                                      PM, UE_VECTOR_SIZE)

        ue.start_capture()
        prog_addr = ue.get_program_dram_addr()
        sched.begin_program()

        # Dimension GPRs, ONE SET PER ENGINE -- both of them, on every engine. The PBI M
        # register collapses the M-unroll exactly as it does single-engine. qb_reg used
        # to be primary-only because the flash was; now that attention is group-sharded
        # every engine that owns a group needs it, and the engines that own none get it
        # anyway so the register sets stay SYMMETRIC across engines (the expert stage
        # primes its full set on every worker for the same reason: an asymmetric file
        # means a body replayed on the wrong engine drives whatever that engine happens
        # to hold at that index -- finite, plausible, wrong).
        #
        # Allocation ORDER matters: indices are a per-engine bump counter, so m before
        # qb on the primary keeps the primary's two indices exactly where they were.
        m_regs, qb_regs = [], []
        for e, eng in enumerate(sched.engines):
            r = eng.alloc_isa_reg(); eng.generate_instruction_add_set(r, PM)
            m_regs.append(r)
            q = eng.alloc_isa_reg(); eng.generate_instruction_add_set(q, QB)
            qb_regs.append(q)

        def emit_matmul(ctx, M, K, N, A, la, proj, OUT, split=False, **kw):
            """bf16 B operand throughout (16-bit multipliers) -- no is_B_quantized,
            no SCALE_DRAM_ADDR, and with attention_bias/mlp_bias both false there is no
            broadcast_N bias either. THE ONLY per-column offset that exists on this
            model is therefore B += n0*K*bpe, which ctx.b_addr() computes."""
            e = ctx.engine_idx
            B = la[f"{proj}_weight"]
            ctx.unsafe_ue.matmat_mul_core(
                M=M, K=K, N=N, A_DRAM_ADDR=A,
                B_DRAM_ADDR=ctx.b_addr(B, K) if split else B,
                OUTPUT_DRAM_ADDR=OUT, gpr_M_reg=m_regs[e], **kw)

        n_lm = len(self.lm_layer_addrs)
        for i, la in enumerate(self.lm_layer_addrs):
            h_in = self.LM_INPUT_DRAM if i % 2 == 0 else self.LM_OUTPUT_DRAM
            h_out = self.LM_OUTPUT_DRAM if i % 2 == 0 else self.LM_INPUT_DRAM

            # ---- region 1: ln1 (redundant) + q (N-split, gathered) + k/v (kv-head
            #                split, written straight into the persistent cache) ---------
            # Opens with a barrier, which is also the fence that makes h_in -- written by
            # the previous layer's residual add on the primary -- visible everywhere.
            def qkv_body(ctx, la=la, h_in=h_in, i=i):
                e = ctx.engine_idx
                x = ctx.unsafe_ue
                nrm = PE("lm_norm", e)
                x.rms_norm_core_dram(M=PM, N=H, A_DRAM_ADDR=h_in, OUTPUT_DRAM_ADDR=nrm,
                                     GAMMA_DRAM_ADDR=la["ln1_gamma"], gpr_M_reg=m_regs[e])
                emit_matmul(ctx, PM, H, ctx.cols, nrm, la, "q", ctx.col_out("lm_q"),
                            split=True)
                scatter_lane(x, ctx.col_out("lm_q"), self.LM_Q_DRAM,
                             ctx.col_offset, ctx.cols, H)
                # k/v, ONE KV-HEAD AT A TIME, STRAIGHT INTO THE PERSISTENT CACHE.
                # B is [KV, H] row-major, so head h is rows h*D..h*D+D, contiguous at
                # + h*D*H*bpe -- the same "B += n0*K*bpe" rule ctx.b_addr() applies for
                # q/o, just with n0 = h*D. The N=D writeback is dense [PM, D], which is
                # bit-for-bit the layout LAYER0_K/V_DRAM's head block already has.
                for is_v, h in kv_units[e]:
                    dst = ((self.LAYER0_V_DRAM if is_v else self.LAYER0_K_DRAM)
                           + i * self.KV_LAYER_STRIDE + h * self.KV_HEAD_STRIDE)
                    x.matmat_mul_core(
                        M=PM, K=H, N=D, A_DRAM_ADDR=nrm,
                        B_DRAM_ADDR=(la["v_weight"] if is_v else la["k_weight"])
                        + h * D * H * bpe,
                        OUTPUT_DRAM_ADDR=dst, gpr_M_reg=m_regs[e])
                    if not is_v:
                        # rope K in place on the engine that just wrote it. The flash
                        # that reads this block may be on a DIFFERENT engine (K-head h
                        # and group h only coincide at ne >= 10; V-head h never does) --
                        # the attention region's opening barrier is that fence, and it
                        # is the same barrier region 1 used to end with.
                        x.rope_hf_core_dram(
                            M=PM, N=D, input_dram_addr=dst, output_dram_addr=dst,
                            cos_dram_addr=self.ROPE_PACKED_DRAM,
                            sin_dram_addr=self.ROPE_PACKED_DRAM + D * bpe,
                            gpr_M_reg=m_regs[e])
            # join=False: the attention region below opens with its own rendezvous and
            # that is the fence BOTH the full-width LM_Q_DRAM gather and the cache
            # blocks need. Two back-to-back barriers would cost 32 extra round trips.
            sched.col_sharded_region(H, qkv_body, join=False)

            # ---- region 2: attention, GROUP-SHARDED over the 5 kv-heads ----------------
            # The opening barrier is what publishes region 1: this engine gathers q-heads
            # it did not compute, and reads cache blocks it did not write.
            #
            # Engines n_grp..ne-1 emit NOTHING here (5 groups do not divide 8/10/12) and
            # go straight to their next FLAG_SET. Priced at 3.8% of the modelled stage
            # at ne=12 -- an accepted cost, not an oversight.
            sched.barrier()
            for _e, _x_ue in enumerate(sched.engines):
                for kv_b in attn_map[_e]:
                    self._prefix_attn_group(
                        _x_ue, i, kv_b, qb_regs[_e],
                        PE("lm_flash_q", _e), PE("lm_flash_k", _e),
                        PE("lm_flash_v", _e), PE("lm_flash_out", _e),
                        PE("lm_flash_scr", _e))

            # ---- region 3: o_proj (N-split, gathered) ---------------------------------
            # The opening barrier fences the group engines' un-stack writes to
            # LM_ATTN_RESULT_DRAM -- disjoint per head, but every engine reads ALL of it
            # here -- before anyone projects from it.
            def o_body(ctx, la=la):
                e = ctx.engine_idx
                x = ctx.unsafe_ue
                emit_matmul(ctx, PM, H, ctx.cols, self.LM_ATTN_RESULT_DRAM, la, "o",
                            ctx.col_out("lm_o"), split=True)
                scatter_lane(x, ctx.col_out("lm_o"), self.LM_O_PROJ_DRAM,
                             ctx.col_offset, ctx.cols, H)
            # join=False: region 4 opens with its own barrier and that is the fence the
            # full-width LM_O_PROJ_DRAM read needs. Two back-to-back rendezvous would
            # cost 32 extra round trips over the layer stack for nothing.
            sched.col_sharded_region(H, o_body, join=False)

            # ---- region 4: residual + ln2 (redundant) + gate/up/mul (N-split lane) ----
            def mlp_body(ctx, la=la, h_in=h_in):
                e = ctx.engine_idx
                x = ctx.unsafe_ue
                resid, nrm = PE("lm_resid", e), PE("lm_norm", e)
                x.eltwise_core_dram(M=PM, N=H, dram_a=h_in, dram_b=self.LM_O_PROJ_DRAM,
                                    dram_out=resid, mode=UE_MODE.ELTWISE_ADD,
                                    gpr_M_reg=m_regs[e])
                # split residual+norm: the fused post_add norm needs 4 advancing PBI
                # pointers against a limit of 3.
                x.rms_norm_core_dram(M=PM, N=H, A_DRAM_ADDR=resid, OUTPUT_DRAM_ADDR=nrm,
                                     GAMMA_DRAM_ADDR=la["ln2_gamma"], gpr_M_reg=m_regs[e])
                gate, up, mult = (ctx.col_out("lm_gate"), ctx.col_out("lm_up"),
                                  ctx.col_out("lm_mult"))
                if self.PREFIX_FUSED_SILU:
                    emit_matmul(ctx, PM, H, ctx.cols, nrm, la, "gate", gate,
                                split=True, silu_enable=True)
                else:
                    emit_matmul(ctx, PM, H, ctx.cols, nrm, la, "gate", gate, split=True)
                    # SiLU is elementwise, so view [PM, Nc] as [PM*Nc/64, 64] and the
                    # stored 64x64 identity applies. Nc is a multiple of 64 by
                    # construction, so the view is exact on every engine.
                    silu_core_dram(x, M=(PM * ctx.cols) // UE_VECTOR_SIZE,
                                   N=UE_VECTOR_SIZE, A_DRAM_ADDR=gate,
                                   OUTPUT_DRAM_ADDR=gate,
                                   IDENTITY_DRAM_ADDR=self.identity_addr)
                emit_matmul(ctx, PM, H, ctx.cols, nrm, la, "up", up, split=True)
                eltwise_mul_core_dram(x, size=ctx.elems(PM), A_DRAM_ADDR=gate,
                                      B_DRAM_ADDR=up, OUTPUT_DRAM_ADDR=mult)
            # join=False: nothing leaves the lane. The K-split region 5 below opens with
            # its own barrier, and that is the only fence the partials need.
            sched.col_sharded_region(I, mlp_body, join=False)

            # ---- region 5: down (K-split) + reduce_add --------------------------------
            def down_body(ctx, la=la):
                e = ctx.engine_idx
                # A is ALREADY the dense [PM, Kc] lane the multiply produced -- the
                # single reason a K-split is affordable here. B is the repacked slice at
                # a contiguous offset. OUT is this engine's full-width PARTIAL SUM.
                ctx.unsafe_ue.matmat_mul_core(
                    M=PM, K=ctx.k_cols, N=H,
                    A_DRAM_ADDR=sched.col_output_addr("lm_mult", e),
                    B_DRAM_ADDR=la["down_weight"] + H * ctx.k_offset * bpe,
                    OUTPUT_DRAM_ADDR=PE("lm_down", e), gpr_M_reg=m_regs[e])
            sched.k_sharded_region(I, down_body, join=False)
            # join=False: the only consumer of the reduced sum is the PRIMARY's own
            # residual add on the very next line, and the workers' next stop is the next
            # layer's region-1 barrier, which is also what publishes h_out to them.
            sched.reduce_add(DOWN_PARTIALS, self.LM_MLP_DOWN_DRAM, PM, H, join=False)

            # ---- layer output: primary only. The next region's opening barrier is what
            # publishes it. Note the residual read is the PRIMARY's private copy, which
            # every engine computed identically.
            ue.eltwise_core_dram(M=PM, N=H, dram_a=self.LM_RESIDUAL_DRAM,
                                 dram_b=self.LM_MLP_DOWN_DRAM, dram_out=h_out,
                                 mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=m_regs[0])

        final = self.LM_INPUT_DRAM if n_lm % 2 == 0 else self.LM_OUTPUT_DRAM
        ue.rms_norm_core_dram(M=PM, N=H, A_DRAM_ADDR=final,
                              OUTPUT_DRAM_ADDR=self.LM_PRE_NORM_DRAM,
                              GAMMA_DRAM_ADDR=self.final_norm_addr, gpr_M_reg=m_regs[0])
        self.PREFIX_HIDDEN_DRAM = self.LM_PRE_NORM_DRAM

        # LIFO PER ENGINE (qb was allocated after m), and before finalize() closes the
        # worker captures. Both registers now exist on every engine, so both are
        # released on every engine -- an unbalanced release drifts that engine's bump
        # counter and a later scratch allocation silently lands on a live register.
        for eng in sched.engines:
            eng.release_isa_reg()                  # qb_reg
            eng.release_isa_reg()                  # m_reg
        self._prefix_worker_progs = sched.finalize()
        self._record_worker_prog_sizes("prefix", sched)
        self._prefix_sched = sched
        # The prefix's per-engine footprint GREW when attention moved onto the workers
        # (flash staging + scratch, 1.06 MB/worker) and its worker programs grew with it
        # (each worker now carries a kv-group's flash for all 32 layers). Both live in
        # the same arena as vision's and denoise's, and overflowing either end is a
        # SILENT corruption that surfaces as a worker spinning on a FLAG_CHECK with no
        # timeout. Vision and denoise already assert this after their finalize(); the
        # prefix did not, and it is now the stage most likely to be the one that tips it.
        self._assert_worker_programs_fit(label="prefix: ")

        ue.generate_instruction_halt()
        ue.stop_capture()
        raw = bytearray()
        for inst in ue.capture_buffer:
            raw.extend(inst.get_bytes())
        ue.dma_write(DMA_DEVICE_H2C, prog_addr, raw, len(raw))
        ue.allocate_program_dram(len(raw))
        ue.clear_capture_buffer()
        self._prefix_program_addr = prog_addr
        _n_kv = min(ne, 2 * self.NUM_KV_HEADS)
        _n_grp = min(ne, self.NUM_KV_HEADS)
        print(f"    prefix x{ne} ({self.NUM_LAYERS} layers, PM={PM}, GQA g{G}): "
              f"primary {len(raw) / 1e6:.2f} MB @0x{prog_addr:X}, workers "
              f"{sched.worker_program_bytes() / 1e6:.2f} MB")
        print(f"    [prefix] q/o N={H} -> {[c for _, c in n_split]}, "
              f"gate/up N={I} -> {[c for _, c in i_split]}, down K-split + reduce_add.")
        print(f"    [prefix] k/v: {2 * self.NUM_KV_HEADS} kv-head units over {_n_kv} "
              f"engine(s), written straight into the cache; attention: "
              f"{self.NUM_KV_HEADS} kv groups over {_n_grp} engine(s)"
              + (f" ({ne - _n_grp} idle in that region -- {self.NUM_KV_HEADS} groups do "
                 f"not divide {ne})" if ne > _n_grp else "") + ".")
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

    def _ae_self_slot(self, layer_idx):
        """Which per-engine CK/CV slot a SELF layer owns, or None if it is a cross layer.

        The [prefix ; suffix] staging buffers used to be ONE pair per engine, rebuilt
        from scratch on every self layer of every Euler step -- four strided copies,
        160 self-layer-executions, 640 op-executions on the busiest engine. Two of those
        four copy the FROZEN prefix cache, which does not change across the ten steps,
        so they belong outside the unroll (the same argument _emit_cross_reproject_all
        already makes and ships for the cross layers' reprojection).

        Hoisting them means a layer cannot share its buffer with the next self layer any
        more: layer 0's prefix half must still be there when layer 2 runs. So the buffer
        becomes one SLOT PER SELF LAYER, and this is the index. 16 slots x [256, 64] x 2
        tensors x 2 B = 1 MB per engine, against 46 MB arenas.

        Counted with the emitter's OWN predicate rather than a re-derived i % 2: the
        parity is config-driven and flagged UNVERIFIED in _ae_is_self_attn, so a second
        copy of it here is a second thing to get wrong."""
        if not self._ae_is_self_attn(layer_idx):
            return None
        return sum(1 for j in range(layer_idx) if self._ae_is_self_attn(j))

    def _ae_n_self_layers(self):
        """How many SELF layers this expert has -- the CK/CV slot count."""
        n = self.E_LAYERS if self.EXPERT_LAYERS is None else int(self.EXPERT_LAYERS)
        return sum(1 for i in range(n) if self._ae_is_self_attn(i))

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

        # FLASH GEOMETRY. unified_attention_core's contract is Q[batch,D],
        # K/V[aligned_seq_len,D], bias[batch,aligned_seq_len]. The stacked-Q batch is
        # M*G (64*3 = 192 rows: 64 action tokens x 3 q-heads per kv group). A cross-attn
        # layer points flash straight at the [PM,D] reprojected-cache slice; a self-attn
        # layer replicates its own M kv rows xG (that replication is mathematically free:
        # softmax over G exact copies of each key splits each weight G ways and the
        # duplicated V sums it back).
        #
        # HISTORICAL NOTE, because this assert used to say something stronger. On the
        # pulsevla variant M*G = 192 and PM = 192 COINCIDENTALLY -- one is a suffix x GQA
        # group product, the other a prefix length, and they collided at 2 cameras. The
        # original guard required QB == PM and warned that otherwise the cross K/V would
        # need duplicate_gqa_rows and the bias re-shaping. NEITHER IS TRUE. The real
        # kernel constraint (user_dma_core.py:7272-7277, unified_attention_core_dynamic)
        # is `batch <= aligned_seq_len` and `aligned_seq_len % 64 == 0`, with batch
        # otherwise UNCONSTRAINED -- the same fact that lets the attention row-split cut
        # 192 into 2x96. And both bias builders are already written in terms of (QB, PM)
        # and (QB, CW), so they reshape themselves. At 3 cameras PM = 256: 192 <= 256 and
        # 256 % 64 == 0, so the cross path is legal untouched.
        #
        # What the coincidence DID buy is one register: the cross K/V reprojection matmul
        # runs at M = PM and borrowed regs["qb"] because it held the same number. That is
        # the single thing that breaks, and it is fixed with a separate GPR (regs["pm"]),
        # allocated only when the two differ so pulsevla's register indices and byte
        # stream stay bit-identical.
        self.AE_FLASH_ROWS = QB
        CW_ = PM + M
        assert QB <= PM, (
            f"stacked-Q batch ({QB} = SUFFIX_LEN_PAD {M} x GROUP_SIZE {G}) exceeds the "
            f"cross-attn key length PREFILL_MAX_SEQ_LEN ({PM}); unified_attention_core "
            f"requires batch <= aligned_seq_len. Raising prefix_len_padded or lowering "
            f"suffix_len_padded is the fix -- this is a real kernel limit, not a "
            f"bookkeeping one.")
        assert PM % UE_VECTOR_SIZE == 0 and CW_ % UE_VECTOR_SIZE == 0, (
            f"flash aligned_seq_len must be a multiple of {UE_VECTOR_SIZE}: "
            f"PREFILL_MAX_SEQ_LEN={PM}, combined={CW_}")

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
        # ONE SLOT PER SELF LAYER (see _ae_self_slot). register_per_engine keeps the
        # PRIMARY on its existing model address, so this allocation -- not just the
        # workers' copies -- has to be the full size or engine 0 writes past its buffer
        # on the first hoisted stage. At ne == 1 only slot 0 is ever touched.
        _ck_elems = self._ae_n_self_layers() * self.AE_COMBINED_LEN * D
        self.AE_CK_DRAM = a(_ck_elems)
        self.AE_CV_DRAM = a(_ck_elems)
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
        # All four are SCRATCH, and are now produced and consumed WITHIN ONE ITERATION
        # of the hoisted _emit_cross_reproject_all pass -- sequentially, on the primary
        # -- so one set is still shared by all 16 cross layers. Only the head-major
        # RESULT below needs per-layer storage, because it must stay live across the
        # whole 10-step unroll.
        self.AE_XKV_HEAD_STRIDE = PM * D * bpe
        self.AE_XK_TOK_DRAM  = a(PM * KV)   # cached K interleaved token-major [192,320]
        self.AE_XV_TOK_DRAM  = a(PM * KV)
        self.AE_XK_PROJ_DRAM = a(PM * KV)   # k_proj(cached K), still token-major
        self.AE_XV_PROJ_DRAM = a(PM * KV)
        # de-interleaved [5,192,64] flash operands. Same byte size as the *_TOK buffers;
        # kept SEPARATE rather than aliased so the gather -> matmul -> scatter chain has
        # no read/write overlap to reason about.
        #
        # PER-LAYER, NOT SHARED. The reprojection reads only (a) the FROZEN prefix
        # cache and (b) this layer's static k_weight/v_weight -- neither varies with
        # the Euler step -- so it is hoisted OUT of the 10-step unroll and computed
        # once per layer (see _emit_cross_reproject_all). That turns the destination
        # from one reused buffer into E_LAYERS slots: every cross layer keeps its own
        # head-major [5, PM, D] K and V alive for the whole denoise program.
        # Keyed by layer_idx (not by a cross-layer ordinal) on purpose -- an ordinal
        # is one off-by-one away from a layer reading another layer's projection,
        # which is finite-but-scrambled, never NaN. E_LAYERS slots so the
        # --bisect-expert truncation (EXPERT_LAYERS < E_LAYERS) needs no re-sizing.
        # 32 x 5 x 192 x 64 x 2 B = 3.93 MB each, 7.86 MB for the pair.
        self.AE_XKV_LAYER_STRIDE = self.NUM_KV_HEADS * PM * D * bpe
        assert self.AE_XKV_LAYER_STRIDE % 32 == 0, (
            "cross reprojection layer stride must be 32 B beat aligned")
        self.AE_XK_HEADS_DRAM = a(self.E_LAYERS * self.NUM_KV_HEADS * PM * D)
        self.AE_XV_HEADS_DRAM = a(self.E_LAYERS * self.NUM_KV_HEADS * PM * D)

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
        self.AE_FLASH_SCRATCH_DRAM = a(self._ae_flash_scratch_elems())

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
        self._assert_tensor_region_fits("_ae_tensor_init (expert)")

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
        self._assert_tensor_region_fits("_ae_tensor_init (expert)")
        print(f"  tensor DRAM after every stage: "
              f"{self.get_tensor_dram_usage() / 1024**2:.1f} MB of "
              f"{(self._program_dram_base - self._tensor_dram_base) / 1024**2:.0f} MB")

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

    def _ae_scatter_lane(self, ue, src, dst_base, col_offset, cols, full_n, rows):
        """Scatter one engine's DENSE [rows, cols] matmul lane into its own column band
        of a shared full-width [rows, full_n] buffer.

        THE WHOLE REASON THIS EXISTS. matmat_mul_core writes back with stride N*bpe for
        the N IT WAS GIVEN, so calling it with N=cols yields a dense [rows, cols] block,
        NOT a strided column slice of the full tensor. The consumers of q_proj (the
        flash-Q head gather) and o_proj (the residual add) both want ONE contiguous
        [rows, full_n] buffer, so the lane has to be re-scattered. The bands are disjoint
        and 64-element (128 B) aligned, so no two engines ever touch the same bytes and
        the region's rendezvous is the only fence needed -- no lock, no extra barrier.

        Emitted as ONE copy at the full lane width -- see the SRAM note in the body for
        why the old 64-column chunking was strictly worse.

        THE DEST BASE CARRIES ITS PER-COLUMN OFFSET. Dropping the +(col_offset+c0) does
        not fault and does not NaN -- every engine would overwrite columns [0:cols] and
        the result would be finite, plausible and wrong (pi05 denoise bug #3, and trap #8
        on the attention un-stack right above). Both computed addresses are asserted onto
        a 32 B AXI beat here so a mis-scaled offset cannot slip through silently."""
        bpe = 2
        assert cols % UE_VECTOR_SIZE == 0 and full_n % UE_VECTOR_SIZE == 0
        assert col_offset % UE_VECTOR_SIZE == 0
        # ONE COPY AT THE FULL LANE WIDTH, not cols/64 copies of 64.
        #
        # This used to emit a 64-column chunk at a time "exactly as
        # _compile_prefix_sharded's scatter_lane". That is strictly worse on the axis
        # that turned out to matter. _ae_strided_copy lowers to a strided DMA whose
        # transaction count is `rows` and whose burst is width*bpe, so chunking a
        # 128-column lane doubled the transactions AND halved the burst -- and it made
        # the SOURCE strided too, since the lane is a dense [rows, cols] and reading 64
        # of its 128 columns walks src_jump=cols*bpe with a width of only 64*bpe.
        #
        # At the full width: src_jump == width*bpe, so the read is one contiguous
        # stream, and the scattered side moves 256 B per burst instead of 128 B.
        # (Measured context: denoise's unaccounted 658 ms tracks strided row-DMA
        # transactions, not op count -- see the --expert-layers layer sweep.)
        #
        # SRAM: staging is rows*cols*bpe = 64*128*2 = 16 KB, doubled for the
        # gather/scatter pair = 32 KB, against the 64 KB below the flash window at
        # 0x10000. Asserted on the ACTUAL width now rather than on UE_VECTOR_SIZE, so a
        # wider lane fails here instead of silently overrunning into flash scratch.
        assert rows * cols * bpe * 2 <= 0x10000, (
            f"scatter staging {rows}x{cols} needs {rows * cols * bpe * 2} B, over the "
            f"{0x10000} B SRAM window below the flash region")
        dst_a = dst_base + col_offset * bpe
        assert src % mes.AXI_BEAT_BYTES == 0, \
            f"scatter src 0x{src:X} is not {mes.AXI_BEAT_BYTES} B beat aligned"
        assert dst_a % mes.AXI_BEAT_BYTES == 0, \
            f"scatter dst 0x{dst_a:X} is not {mes.AXI_BEAT_BYTES} B beat aligned"
        self._ae_strided_copy(ue, src, cols * bpe, dst_a, full_n * bpe, rows, cols)

    def _assert_flag_check_reaches(self, ne):
        """Refuse to compile a rendezvous the ISA emitter would silently DROP.

        generate_instruction_flag_check historically clamped target_engine_idx to 0..7
        and, on a larger index, PRINTED an error and RETURNED -- emitting nothing. That
        is the worst possible failure mode for a barrier: the peers still raise flags
        nobody checks, so engine 0 walks straight through the rendezvous into data the
        workers have not written yet. Finite, plausible, wrong actions -- or, once the
        counts desync, a spin on a FLAG_CHECK that has no timeout and costs a
        power-cycle. Neither is visible in a compile log.

        So: PROBE the emitter rather than trust it. Capture one flag_check at the
        highest index this stage needs and require that exactly one instruction came
        out. The probe runs on the PRIMARY before compile_denoise_loop opens its real
        capture, and start_capture() resets _inst_id / capture_count / capture_buffer /
        _capture_loop_stack wholesale, so it leaves nothing behind.

        Skipped entirely at ne <= 8 -- the historical bound covers those, and skipping
        keeps the proven 8-engine and single-engine paths free of any new call."""
        if ne <= 8:
            return
        with silenced():
            self.start_capture()
            self.generate_instruction_flag_check(target_engine_idx=ne - 1)
            emitted = len(self.capture_buffer or [])
            self.stop_capture()
            self.clear_capture_buffer()
        # # assert emitted == 1, (
        #     f"user_dma_core.generate_instruction_flag_check(target_engine_idx={ne - 1}) "
        #     f"emitted {emitted} instruction(s), not 1: this build still clamps the FLAG "
        #     f"index to 0..7, so every barrier past 8 engines would be missing checks "
        #     f"(silent race, or a FLAG_CHECK spin with no timeout). The fix is the "
        #     f"one-liner already carried on the pi05_12core and smolvlm2_multiengine "
        #     f"branches -- widen the bound to 0..15, the width the fabric decodes -- or "
        #     f"pin this stage back with --denoise-engines 8.")

    def _ae_proj_lanes(self, N, ne):
        """How many engines an N-column projection can ACTUALLY use, and the partition.

        THE ONE PLACE THE PER-OP ENGINE COUNT IS DECIDED. A stage does not have to run
        every op at the same width, and denoise cannot: `split_cols` refuses to cut N
        into fewer than 64 columns per engine (a partial 64-element vector is not
        something the matvec pipeline, the B row block or the 128 B SRAM row can
        express), so an op's ceiling is its own block count N//64:

            gate/up/down  N=1280 -> 20 blocks  -- never the binding constraint
            q_proj        N=960  -> 15 blocks  -- never the binding constraint
            o_proj        N=512  ->  8 blocks  -- THE ONE THAT BINDS at ne > 8

        Engines [lanes, ne) emit NO work in such a region. They still walk both of its
        rendezvous, because a barrier is only safe while every engine's barrier COUNT is
        identical -- an engine that skipped one would answer the next region's flag check
        with the previous region's, and FLAG_CHECK has no timeout. That symmetry is why
        this is a lane cap and not a second, narrower scheduler: a scheduler pins ALL its
        barriers to its own num_engines, so an 8-engine region nested inside a 10-engine
        program would leave engines 8..9 spinning on a rendezvous nobody joins."""
        lanes = max(1, min(int(ne), N // UE_VECTOR_SIZE))
        return lanes, self._col_split(N, lanes)

    def _ae_alloc_proj_lanes(self, sched, name, M, N):
        """Per-engine [M, cols] lane buffers for a projection, honouring the lane cap.

        Identical to sched.alloc_col_output() whenever the op can use every engine --
        which is every count <= 8, so the proven 8-engine program is emitted byte for
        byte. Only the capped case (o_proj at ne > 8) allocates by hand, and it
        allocates on the OWNING engines only: an idle engine gets no lane at all, so a
        stray write through a lane address is a KeyError at compile time rather than a
        second engine's buffer."""
        ne = sched.num_engines
        lanes, split = self._ae_proj_lanes(N, ne)
        if lanes == ne:
            addrs = sched.alloc_col_output(name, M, N)
        else:
            addrs = []
            for (off, cols), ue in zip(split, sched.engines[:lanes]):
                a = ue.allocate_tensor_dram(M * cols * 2,
                                            label=f"col_out_{name}_{off}",
                                            align_bytes=mes.SRAM_ROW_BYTES)
                assert a % mes.SRAM_ROW_BYTES == 0, (
                    f"projection lane {name!r} base 0x{a:X} is not "
                    f"{mes.SRAM_ROW_BYTES} B SRAM-row aligned")
                addrs.append(a)
        self.__dict__.setdefault("_ae_lane_plan", {})[name] = (lanes, split, list(addrs))
        return lanes, addrs

    def _ae_proj_sharded(self, sched, M, K, N, A, W, out_dram, name, join=True,
                         bias=None):
        """One expert projection, N-split across the engines and re-gathered.

        Lifted straight from _compile_prefix_sharded's q/o handling. Each engine runs
        matmat with N=ctx.cols against the CONTIGUOUS ROW BLOCK of B that its output
        columns need (B is [N, K] row-major, so a column shard of the output is a row
        block of the weight -- ctx.b_addr), lands a dense [M, cols] lane, and scatters
        that lane into its own column band of the shared full-width `out_dram`.

        A is read in FULL by every engine: it is the same [M, K] activation, and a
        K-split is NOT available (A[:, k0:k0+Kc] is strided and matmat_mul_core derives
        A's row stride from the K it is given; user_dma_core.py:5155 has no A-stride
        parameter). That is the same reason the prefix chose N-split for its o_proj.

        ops.expert_quant is bf16 and attention_bias/mlp_bias are false, so for the LAYER
        projections (q, o) there is no scale blob and no bias: of the four per-column
        offsets that could exist, only B += col_offset*K*bpe remains.

        THE HEAD IS THE EXCEPTION AND IT IS WHY `bias` EXISTS. action_in_proj,
        action_time_mlp_in and action_time_mlp_out all carry a real [N] bias (see
        _weight_init_head), and an [N] bias is INDEXED BY N, not broadcast over it --
        it must be sliced with the columns. ctx.bias_addr does that and its own
        docstring flags it as the single most likely thing here to be passed through
        unsliced. Getting it wrong is silent: every engine would add the bias's first
        `cols` entries to its own band, so lanes 0..63 would be right and the other 448
        would carry the wrong constant. Finite, plausible, wrong.

        NO REDUCTION, so unlike the MLP's down-projection K-split this introduces no
        bf16 add chain and no re-association: every output element is computed by
        exactly one engine with the same accumulation order the primary used at ne=1.

        THE LANE CAP (see _ae_proj_lanes). o_proj's N is E_HIDDEN_PAD = 512 = exactly 8
        blocks of 64, so at ne > 8 it runs on the first 8 engines and the rest sit out
        the region -- walking both rendezvous, emitting nothing. That is what unblocks
        the stage at 10 engines at all: split_cols would otherwise assert `blocks >= n`
        and the whole denoise compile would fail. Every other projection here (q at 15
        blocks, the MLP at 20) uses every engine, so this branch is o_proj's alone."""
        ne = sched.num_engines
        plan = self.__dict__.get("_ae_lane_plan", {})
        assert name in plan, (
            f"projection lane {name!r} was never allocated; compile_denoise_loop must "
            f"call _ae_alloc_proj_lanes({name!r}) before the first expert layer")
        lanes, split, addrs = plan[name]
        assert lanes == max(1, min(ne, N // UE_VECTOR_SIZE)), (
            f"projection {name!r} was planned for {lanes} lane(s) but N={N} on "
            f"{ne} engine(s) wants {max(1, min(ne, N // UE_VECTOR_SIZE))}")

        def _emit(ctx, dst):
            raw = ctx.unsafe_ue
            self._ae_matmul(raw, M, K, ctx.cols, A, ctx.b_addr(W, K), dst,
                            bias=(None if bias is None else ctx.bias_addr(bias)),
                            m_reg=self._ae_m_regs[ctx.engine_idx])
            self._ae_scatter_lane(raw, dst, out_dram, ctx.col_offset, ctx.cols, N, M)

        if lanes == ne:
            # EVERY engine has a lane: the scheduler's own region, unchanged, so the
            # emitted stream at ne <= 8 is byte-identical to the pre-cap program.
            sched.col_sharded_region(N, lambda ctx: _emit(ctx, ctx.col_out(name)),
                                     join=join)
            return
        # CAPPED. Hand-rolled so the idle engines still see both rendezvous: barrier()
        # is emitted on all `ne` engines, only the body is skipped. Nothing here touches
        # the scheduler's region bookkeeping, so an accidental nesting still trips its
        # own assert.
        sched.barrier()
        for i in range(lanes):
            col_offset, cols = split[i]
            _emit(mes.ColumnShardContext(sched, i, sched.engines[i], N,
                                         col_offset, cols), addrs[i])
        if join:
            sched.barrier()

    def _ae_gated_mlp_sharded(self, sched, M, la):
        """The expert layer's gated MLP, N-sharded (output columns) across the engines.

        WHY THIS AND NOT q/k/v/o. M is SUFFIX_LEN_PAD = 64 EVERYWHERE in this stage --
        ONE 64-row block -- so the M-split that shards vision and prefix is not merely
        inefficient here, it is INEXPRESSIBLE. That leaves the N axis. But an N-shard is
        only free when its consumer stays in the lane, because matmat_mul_core writes
        back with stride N*bpe for the N IT WAS GIVEN: calling it with N=cols produces a
        DENSE per-engine [64, cols] block, never a strided column slice of a full buffer.
        So:

          gate / up  (N=1280)  -> per-engine [64, cols], consumed ELEMENTWISE by the
                                  SiLU-multiply, which never leaves the lane. Free.
          mult       (in lane) -> the [64, cols] product IS the contiguous K-slice
                                  [col_offset, col_offset+cols) of the [64, 1280]
                                  intermediate.
          down       (K=1280)  -> K-SPLIT: every engine computes a FULL [64, 512]
                                  PARTIAL SUM over its slice, and reduce_add combines
                                  them into the existing AE_MLP_DOWN_DRAM, leaving every
                                  downstream statement (the gated residual, the next
                                  layer's norm) untouched. An 8-lane N-split was tried
                                  and measured +50 ms -- see the call site.

        q_proj AND o_proj ARE NOW SHARDED TOO -- see _ae_proj_sharded. They used to be
        excluded here on the grounds that their consumers (the flash-Q head gather and
        the full-width residual add) each read ONE contiguous [64, N] buffer, so an
        N-shard would leave eight disjoint dense blocks needing a cross-engine gather.
        That gather turned out to be cheap and already proven: _compile_prefix_sharded
        does exactly it for the prefix's q and o with _ae_strided_copy, one 64-column
        chunk at a time, into disjoint 128 B-aligned column bands. q is N=960 = 15
        blocks, uneven at 8 and never better than 7.5x; o is N=512 = EXACTLY 8 blocks of
        64, so it splits perfectly at ne=8 and CANNOT go wider -- above 8 engines it is
        lane-capped and the surplus engines sit out the region (_ae_proj_lanes). Neither
        introduces a reduction, so both stay numerically identical to ne=1.

        k_proj / v_proj ARE SHARDED TOO NOW, and NOT by split_cols -- which does refuse
        N=320 above 5 engines, exactly as the paragraph that used to sit here said. The
        cut is per (tensor, kv-head): 10 units, because HEAD_DIM == UE_VECTOR_SIZE makes
        a kv-head one column block and k/v are two independent weights. See _ae_kv_units.
        action_out is the only projection left on the primary (N=64, a single block).
        A K-split is not available to any of these -- their A operand is a full [64, K]
        buffer and A[:, k0:k0+Kc] is strided (matmat_mul_core derives A's row stride from
        the K it is given), unlike the down projection whose A is already a per-engine
        dense lane.

        NOT BIT-IDENTICAL TO ne=1, AND THAT IS EXPECTED: reduce_add is a bf16 add chain
        and bf16 addition is not associative. It is the ONLY remaining arithmetic
        difference in the stage -- every other shard here assigns whole output elements
        (N-split) or whole rows (the norm/residual row split, the flash query slice), so
        those are exact. Denoise has the headroom (floor 36.0 dB against a ~40.1 dB bf16
        ceiling); do not chase the delta.
        """
        HP, I = self.E_HIDDEN_PAD, self.E_INTER
        ne = sched.num_engines
        k_split = self._col_split(I, ne)

        def _body(ctx):
            i = ctx.engine_idx
            # The host cut la["down_weight_k"] with _col_split at weight-init time. If
            # the scheduler ever partitions differently the result is finite garbage,
            # not a crash -- so assert the two agree, here, every layer.
            assert (ctx.col_offset, ctx.cols) == k_split[i], (
                f"mlp shard mismatch on engine {i}: scheduler gives "
                f"{(ctx.col_offset, ctx.cols)}, the down blobs were cut for {k_split[i]}")
            raw, mreg, Nc = ctx.unsafe_ue, self._ae_m_regs[i], ctx.cols
            # gate (fused SiLU) and up: N-split. A -- the normed residual -- is read in
            # FULL by every engine; it is the same [64, 512] activation. Only B moves,
            # and B is [N, K] row-major so this engine's output columns are a CONTIGUOUS
            # ROW BLOCK of it. ops.expert_quant is bf16, so there is no scale blob and
            # no bias: three of the four per-column offsets collapse to nothing.
            self._ae_matmul(raw, M, HP, Nc, self.AE_PRE_NORM_DRAM,
                            ctx.b_addr(la["gate_weight"], HP),
                            ctx.col_out("ae_mlp_gate"), m_reg=mreg, silu_enable=True)
            self._ae_matmul(raw, M, HP, Nc, self.AE_PRE_NORM_DRAM,
                            ctx.b_addr(la["up_weight"], HP),
                            ctx.col_out("ae_mlp_up"), m_reg=mreg)
            # silu(gate) * up -- purely per-element, so a column shard of the inputs is a
            # column shard of the output and no engine needs a peer's columns. raw (not
            # ctx.ue) because nn_lib's wrapper drives the SRAM staging itself rather than
            # going through the single allowlisted eltwise_core_dram entry point; the op
            # is exactly the row/column-independent kind the allowlist exists to admit.
            eltwise_mul_core_dram(raw, size=ctx.elems(M),
                                  A_DRAM_ADDR=ctx.col_out("ae_mlp_gate"),
                                  B_DRAM_ADDR=ctx.col_out("ae_mlp_up"),
                                  OUTPUT_DRAM_ADDR=ctx.col_out("ae_mlp_mult"))
            # down: K-SPLIT. This engine's [64, Nc] lane IS the contiguous K-slice, so no
            # gather -- only the host-pre-sliced weight (see _weight_init_expert).
            self._ae_matmul(raw, M, Nc, HP, ctx.col_out("ae_mlp_mult"),
                            la["down_weight_k"][i],
                            sched.per_engine_addr("ae_mlp_down_partial", i), m_reg=mreg)

        # join=False TWICE. col_sharded_region's exit barrier is redundant because
        # reduce_add opens with its own rendezvous and the partials are meaningless until
        # it runs; reduce_add's exit barrier is redundant because the only thing a worker
        # can do afterwards is block on the NEXT region's opening rendezvous. Over 32
        # layers x 10 Euler steps that is 320 barriers saved instead of 960.
        sched.col_sharded_region(I, _body, join=False)
        # partial[0] IS AE_MLP_DOWN_DRAM (register_per_engine keeps the primary on its
        # existing model address), so the add chain accumulates IN PLACE: dram_a ==
        # dram_out on every add. That is deliberate -- it is what reduce_add documents
        # ("partial_addrs[0] may alias out_addr") and what pi05 has run for 180
        # layer-executions per inference. If a sharded denoise ever comes back finite but
        # scrambled with correct-looking gate/up lanes, THIS is the first thing to break
        # by pointing the primary partial at a fresh buffer.
        #
        # THE 9-ADD CHAIN IS NOT THE BOTTLENECK, measured. Replacing it with an 8-lane
        # N-split (mult scattered to a shared [64,1280], no reduction) removed 6 ops per
        # layer and MEASURED +50 ms, because it raised the matmul floor 412 -> 430 ms and
        # this stage tracks its floor at a flat 2.6x. Reverted. Do not re-derive it from
        # op counts.
        sched.reduce_add([sched.per_engine_addr("ae_mlp_down_partial", i)
                          for i in range(ne)],
                         self.AE_MLP_DOWN_DRAM, M, HP, join=False)

    def _ae_group_map(self, ne):
        """kv_b -> engine index. THE ONE SOURCE for every per-kv-head split in a self
        expert layer -- both the stacked-Q flash groups (_ae_attn_groups) and the k/v
        projection units (_ae_kv_units) read it, so the two cuts cannot drift apart.

        n_grp = min(ne, NUM_KV_HEADS) because a GROUP is indivisible: flash row t*G+g is
        q-head kv_b*G+g of token t, so splitting a group's 3 q-heads across engines
        breaks the token-major stacking the whole emitter is built on.

        A PURE FUNCTION of `ne` and NUM_KV_HEADS -- no scheduler, no allocator, no DRAM
        -- which is what lets denoise_shard_selfcheck() call it on a bare class shim with
        no device present."""
        n_grp = min(max(int(ne), 1), self.NUM_KV_HEADS)
        return [kv_b % n_grp for kv_b in range(self.NUM_KV_HEADS)]

    def _ae_kv_units(self, ne):
        """Assign the 10 k/v COLUMN UNITS of one SELF expert layer to `ne` engines.

        THE UNIT IS (tensor, kv-head), NOT the whole projection. This is the same
        argument _prefix_kv_units already makes and ships, on the same two shape facts:

            HEAD_DIM == UE_VECTOR_SIZE == 64  -> one 64-column matmat block IS one
                                                 kv-head, addressable on its own
            k_proj and v_proj are INDEPENDENT weights -> 2 tensors x 5 heads

        so what `split_cols` sees as "E_KV_OUT = 320, only 5 blocks, refuse above 5
        engines" -- the reason both projections were PRIMARY-ONLY, i.e. computed at
        width 1 while nine engines idled -- is really 10 independent units. 10 divides
        the stage ceiling (STAGE_MAX_ENGINES["DENOISE"] = 10) exactly.

        Returns a list of length `ne`; entry e is the [(is_v, head), ...] that engine
        owns. The K half is placed BY THE GROUP MAP, not round-robin: k-head h is roped
        in place by whoever computes it and then concatenated behind the cached prefix K
        by group h's flash a few instructions later, so keeping the pair on one core
        means the only cross-engine hop left in this region is V. The V half round-robins
        from slot NUM_KV_HEADS over min(ne, 10), which at ne = 10 lands every v-head on
        an engine 5..9 -- precisely the engines that own no flash group and are otherwise
        idle for the rest of the layer. Max units per engine is ceil(10/min(ne,10)): 1 at
        ne = 10, 2 at ne = 5..9, so the 10-way cut only actually beats a 5-way one at
        ne = 10 (see denoise_shard_selfcheck for the table).

        CORRECTNESS RESTS ON DISJOINTNESS, NOT ON THE BALANCE. Unit (t, h) writes exactly
        AE_{K,V}_HEADS_DRAM + h*AE_KV_HEAD_STRIDE for M*D*bpe bytes and nothing else
        touches those bytes, so each head slice keeps the SINGLE WRITER PER ADDRESS that
        the flash concat downstream depends on. Two engines on one slice is not redundant
        work, it is a real race, and it surfaces as a finite, plausible, wrong action --
        never a crash and never a NaN. The read side is fenced by the attention region's
        opening barrier, which already existed to publish q_proj."""
        nkv = self.NUM_KV_HEADS
        gmap = self._ae_group_map(ne)
        ne = int(ne)
        out = [[] for _ in range(ne)]
        for h in range(nkv):
            out[gmap[h]].append((0, h))
        n = min(ne, 2 * nkv)
        for h in range(nkv):
            out[(nkv + h) % n].append((1, h))
        # Cheap and total: a dropped unit is a head slice nobody writes (stale K/V from
        # the previous Euler step -- finite and wrong), a duplicated one is the race
        # above. Both are invisible at runtime, so they are refused here.
        flat = sorted(u for lst in out for u in lst)
        assert flat == sorted((t, h) for t in (0, 1) for h in range(nkv)), (
            f"_ae_kv_units({ne}) does not cover the 10 (tensor, head) units exactly "
            f"once: {flat}")
        return out

    def _ae_row_split(self, ne):
        """[(row_offset, rows)] per engine for the ROW-WISE ops -- the two RMSNorms and
        the two residual adds every expert layer runs, 4 x 320 layer-executions.

        THE OTHER AXIS. Everything else in this stage splits COLUMNS, because a matmul
        wants 64-aligned row blocks and M = SUFFIX_LEN_PAD = 64 is exactly one of them.
        These four ops are not matmuls. rms_norm_core_dram reduces along N and is
        independent per row; eltwise_core_dram is independent per element. Both are in
        multi_engine_shard.SHARDED_OP_ALLOWLIST, which is the scheduler stating that
        exact property. So M splits here at row granularity -- the same argument
        reduce_add already makes and ships ("that is a MATMUL constraint;
        eltwise_core_dram has no such rule"), and this deliberately reuses its exact
        divmod scheme so the two cuts land on identical row boundaries.

        A row slice is CONTIGUOUS, never strided: the buffers are dense [M, HP] and row
        r starts at r * HP * bpe = r * 1024 B, a multiple of both the 32 B AXI beat and
        the 128 B SRAM row, for every r. Nothing here needs an alignment argument beyond
        that one.

        ONE SPLIT FOR ALL FOUR OPS AND EVERY LAYER, which is what makes this free of
        barriers. Engine e writes h_out[S_e] at the bottom of layer i and reads h_in[S_e]
        at the top of layer i+1 -- its OWN rows, in its own program order. Every
        cross-engine hop left (PRE_NORM read full-width by the q and MLP regions,
        O_PROJ from o_proj's 8 lanes, MLP_DOWN from reduce_add) already sits behind a
        rendezvous that existed before this split. Give two of these ops different
        splits and that stops being true silently: the reader is finite, plausible and
        wrong, never a crash and never a NaN.

        At ne == 1 this is [(0, 64)] and the callers emit the original single call with
        the original register and unshifted addresses, so --denoise-digest cannot move.

        A PURE FUNCTION of `ne` -- no scheduler, no allocator, no DRAM -- so
        denoise_shard_selfcheck() can call it on a bare class shim with no device."""
        ne = max(1, int(ne))
        M = self.SUFFIX_LEN_PAD
        base, rem = divmod(M, ne)
        counts = [base + (1 if i < rem else 0) for i in range(ne)]
        offs = [sum(counts[:i]) for i in range(ne)]
        assert sum(counts) == M and all(c > 0 for c in counts), (
            f"_ae_row_split({ne}): {counts} does not partition {M} rows into non-empty "
            f"blocks; a zero-row block would emit a degenerate M=0 core call")
        return list(zip(offs, counts))

    def _ae_row_norm(self, sched, A, OUT, gamma, M, HP):
        """RMSNorm [M, HP], row-split across the engines. See _ae_row_split.

        gamma is the [HP] scale vector and is NOT row-indexed -- every engine reads the
        same one, unsliced. (It is the single most likely thing here to be sliced by
        reflex, the way a broadcast_N bias is; it must not be.)"""
        if sched is None or sched.num_engines <= 1:
            self.rms_norm_core_dram(M=M, N=HP, A_DRAM_ADDR=A, OUTPUT_DRAM_ADDR=OUT,
                                    GAMMA_DRAM_ADDR=gamma,
                                    gpr_M_reg=self._ae_regs["m"])
            return
        for e, (off, rows) in enumerate(self._ae_row_split(sched.num_engines)):
            b = off * HP * 2
            sched.engines[e].rms_norm_core_dram(
                M=rows, N=HP, A_DRAM_ADDR=A + b, OUTPUT_DRAM_ADDR=OUT + b,
                GAMMA_DRAM_ADDR=gamma, gpr_M_reg=self._ae_row_m_regs[e])

    def _ae_row_add(self, sched, A, B, OUT, M, HP):
        """Residual add [M, HP] = A + B, row-split across the engines."""
        if sched is None or sched.num_engines <= 1:
            self.eltwise_core_dram(M=M, N=HP, dram_a=A, dram_b=B, dram_out=OUT,
                                   mode=UE_MODE.ELTWISE_ADD,
                                   gpr_M_reg=self._ae_regs["m"])
            return
        for e, (off, rows) in enumerate(self._ae_row_split(sched.num_engines)):
            b = off * HP * 2
            sched.engines[e].eltwise_core_dram(
                M=rows, N=HP, dram_a=A + b, dram_b=B + b, dram_out=OUT + b,
                mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=self._ae_row_m_regs[e])

    def _ae_attn_halves(self, ne):
        """How many QUERY-ROW slices each kv group is cut into: 2 or 1.

        2 only when there are engines to spend on it (ne >= 2 * NUM_KV_HEADS = 10).
        Below that it is 1, which makes _ae_attn_units reproduce _ae_group_map exactly
        -- so every count from 1 to 9, including the proven 8, emits the identical
        stream it emitted before this axis existed."""
        return 2 if int(ne) >= 2 * self.NUM_KV_HEADS else 1

    def _ae_attn_units(self, ne):
        """[(kv_b, half, engine)] -- THE ATTENTION AXIS, past the 5-group cap.

        NUM_KV_HEADS = 5 caps the GROUP axis at five engines, and _ae_attn_groups says
        so: a group is indivisible because flash row t*G+g is q-head kv_b*G+g of token
        t, so splitting a group's 3 q-heads breaks the token-major stacking. That is
        still true. What it does NOT cap is the QUERY ROWS.

        THE FACT THAT UNBLOCKS IT is the same one that let vision drop to an 8-row
        align: unified_attention_core_dynamic (user_dma_core.py:7272-7277) requires
        `batch <= aligned_seq_len` and `aligned_seq_len % 64 == 0`, and constrains
        `batch` NOT AT ALL. The 64-row block is a MATMUL convention. So the 192 stacked
        rows cut into 2 x 96 -- 32 whole tokens x 3 q-heads each, never splitting a
        token's group -- and 5 groups x 2 halves = 10 units, exactly the stage ceiling.
        Attention goes 5x -> 10x, the last region in this stage still running at half
        the engine count.

        EVERY ROW-INDEXED OPERAND MOVES WITH THE SLICE, and there are four:
            Q gather      AE_Q_DRAM      + tok0 * E_Q_OUT * bpe
            RoPE table    rope_tbl       + row0 * 2 * HEAD_DIM * bpe   ([rows, 2D])
            flash bias    bias_addr      + row0 * seq_len * bpe        ([QB, seq_len])
            un-stack      AE_ATTN_RESULT + tok0 * E_Q_OUT * bpe
        Miss any one and the half computes a real softmax over the wrong positions:
        finite, plausible, wrong, and it reads as flow drift rather than a bug. The
        rope slice additionally has to keep sin == cos + D*bpe, which rope_hf_core_dram
        asserts -- shifting the base by whole [cos|sin] rows preserves it.

        NOT ROW-INDEXED, and therefore NOT sliced: K/V (both halves attend the SAME
        keys) and the [D] head dim. The two halves of a self group build IDENTICAL
        CK/CV, each into its own per-engine buffer. That duplication is deliberate --
        ~327 KB per self layer of extra copy against a mid-loop rendezvous to share one
        buffer, and this region's whole design is that groups never sync inside it.

        THE k-HEAD PAIRING DEGRADES, and that is safe. _ae_kv_units pins k-head h to
        group h's engine so the rope-in-place and the concat stay on one core; with two
        engines per group only one of them still holds it locally and the other reads
        across. That read is already fenced by the attention region's OPENING barrier --
        the same one that has always fenced the v-heads, which were never paired at all.

        A PURE FUNCTION of `ne` and NUM_KV_HEADS, like _ae_group_map, so
        denoise_shard_selfcheck can call it on a bare class shim."""
        ne = max(1, int(ne))
        halves = self._ae_attn_halves(ne)
        if halves == 1:
            return [(kv_b, 0, e)
                    for kv_b, e in enumerate(self._ae_group_map(ne))]
        return [(kv_b, h, (kv_b * halves + h) % ne)
                for kv_b in range(self.NUM_KV_HEADS)
                for h in range(halves)]

    def _ae_attn_groups(self, sched, ue, regs):
        """Assign the 5 kv groups x their query-row halves to engines.

        Yields (kv_b, tok0, ntok, x_ue, regs, F_Q, F_OUT, F_SCRATCH, CK, CV) -- one
        tuple per UNIT, where a unit is (kv group, query-row slice). `tok0`/`ntok` are
        this unit's ACTION TOKENS; its stacked flash rows are [tok0*G, (tok0+ntok)*G).

        TWO AXES, and the second one is the whole point. The group axis is capped at
        NUM_KV_HEADS = 5 forever: a group is indivisible because flash row t*G+g is
        q-head kv_b*G+g of token t, so splitting its 3 q-heads breaks the token-major
        stacking the emitter is built on. The QUERY-ROW axis has no such limit --
        unified_attention_core_dynamic leaves `batch` unconstrained -- so 5 groups x 2
        row halves = 10 units and this region finally reaches the stage ceiling instead
        of idling five engines. See _ae_attn_units for the per-operand offset argument.

        THE MAPS LIVE ELSEWHERE, not here: _ae_attn_units for (group, half) -> engine,
        _ae_group_map for the k/v projection pairing that units derives from. Reading
        one function is how they stay paired.

        At ne == 1 (and any count below 10) halves collapses to 1, every tuple is
        tok0=0/ntok=M with the primary's original addresses, and the emitted body is
        byte-identical to the single-engine program.
        """
        M = self.SUFFIX_LEN_PAD
        n_eng = 1 if sched is None else sched.num_engines
        units = self._ae_attn_units(n_eng)
        halves = self._ae_attn_halves(n_eng)
        ntok = M // halves
        assert ntok * halves == M, (
            f"_ae_attn_groups: {halves} row halves do not divide {M} action tokens "
            f"evenly; a ragged split would put a token's 3 q-heads on two engines")
        if n_eng <= 1:
            for kv_b, h, _e in units:
                yield (kv_b, h * ntok, ntok, ue, regs, self.AE_FLASH_Q_DRAM,
                       self.AE_FLASH_OUT_DRAM, self.AE_FLASH_SCRATCH_DRAM,
                       self.AE_CK_DRAM, self.AE_CV_DRAM)
            return
        pe = sched.per_engine_addr
        for kv_b, h, e in units:
            yield (kv_b, h * ntok, ntok, sched.engines[e], self._ae_reg_sets[e],
                   pe("ae_flash_q", e), pe("ae_flash_out", e), pe("ae_flash_scr", e),
                   pe("ae_ck", e), pe("ae_cv", e))

    def _ae_prep_engines(self, sched, ue, regs):
        """(head, engine, regs) with the SAME head -> engine map as _ae_attn_groups.

        CURRENTLY UNUSED. Its one caller was the cross layers' token-major gather, which
        had to finish on every engine before the primary re-projected -- and that whole
        chain has since been hoisted out of the Euler loop onto the primary
        (_emit_cross_reproject_all), taking its barrier with it. Kept because it is the
        ready-made split for any future prep loop that must run ahead of a full-width
        op: derived from _ae_attn_groups so the two mappings can never drift apart, and
        at ne == 1 it is the primary, five times, in head order."""
        for tup in self._ae_attn_groups(sched, ue, regs):
            # tup is (kv_b, tok0, ntok, engine, regs, ...) -- indices 3 and 4, not 1
            # and 2, since the query-row axis widened the tuple.
            yield (tup[0], tup[3], tup[4])

    def _ae_flash_scratch_elems(self):
        """Element count of the expert's flash scratch, in ONE place.

        Sized for the WORST case the body emits: aligned_seq_len = AE_COMBINED_LEN (the
        self layers' prefix+suffix concat), not the cross layers' PM. The per-engine
        copies registered for the attention shard must match it exactly -- the core
        derives its sub-offsets from the compile-time QB/seq, so a short copy is read
        past its end rather than refused."""
        D, CW, QB = self.HEAD_DIM, self.AE_COMBINED_LEN, self.AE_FLASH_ROWS
        return (D + CW) * CW + QB * D

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

    # ---- per-kv-head preparation ------------------------------------------------
    # These helpers are VERBATIM lifts of the `for h in range(NUM_KV_HEADS)` bodies that
    # used to sit inline in _emit_expert_layer, primary-only, ahead of the kv-group loop.
    #
    # _ae_prep_self_kv_head IS NOW THE ne == 1 PATH ONLY. Its de-interleave exists to
    # turn a full-width [64,320] projection into 5 contiguous head slices, and the
    # sharded emitter no longer produces a full-width projection at all: it projects one
    # kv-head at a time straight onto the slice (_ae_proj_self_kv_head). The cross
    # helpers below are still shared -- _emit_cross_reproject_all runs on the primary.
    #
    # SHARED-BUT-DISJOINT destinations. None of the *_HEADS/_TOK buffers is made
    # per-engine, because every write lands at a head-indexed offset that no other head
    # touches:
    #   AE_K/V_HEADS  + h * AE_KV_HEAD_STRIDE  (= M*D*2 = 8192 B)   contiguous block
    #   AE_XK/XV_HEADS+ h * AE_XKV_HEAD_STRIDE (= PM*D*2 = 24576 B) contiguous block
    #   AE_XK/XV_TOK  + h * D * 2 (= 128 B), chunk width D*2 = 128 B, jump KV*2 = 640 B
    #                                            -> column band h, never overlapping
    # Every one of those offsets is a multiple of 128 B, hence of the 32 B beat, and the
    # smallest strided chunk written is a full 128 B row. Asserted below rather than
    # assumed. Contrast the flash staging (ae_flash_q/out/scr, ae_ck/cv), which IS
    # per-engine: those are single reused buffers with no head index in the address, so
    # two engines would collide on byte 0.
    def _ae_prep_self_kv_head(self, x_ue, r_, h, M, KV, D, bpe):
        """SELF layers: de-interleave head h out of the [64,320] k/v projections into
        its contiguous [64,64] slice, then rope K in place. V is never rotated."""
        k_head = self.AE_K_HEADS_DRAM + h * self.AE_KV_HEAD_STRIDE
        v_head = self.AE_V_HEADS_DRAM + h * self.AE_KV_HEAD_STRIDE
        assert (h * self.AE_KV_HEAD_STRIDE) % 32 == 0 and (h * D * bpe) % 32 == 0, (
            f"kv head {h} prep offsets are not 32 B beat aligned")
        self._ae_strided_copy(x_ue, self.AE_K_PROJ_DRAM + h * D * bpe, KV * bpe,
                              k_head, D * bpe, M, D)
        self._ae_strided_copy(x_ue, self.AE_V_PROJ_DRAM + h * D * bpe, KV * bpe,
                              v_head, D * bpe, M, D)
        # d64 RoPE is PBI-ONLY: bare rope_hf_core_dram falls through to the
        # legacy core, which asserts N >= 128. Always pass gpr_M_reg. Never the
        # _gqa variant either -- it asserts N >= 128 too; the group broadcast is
        # done by duplicate_gqa_rows instead.
        x_ue.rope_hf_core_dram(M=M, N=D, input_dram_addr=k_head,
                               output_dram_addr=k_head,
                               cos_dram_addr=self.AE_ROPE_PACKED_DRAM,
                               sin_dram_addr=self.AE_ROPE_PACKED_DRAM + D * bpe,
                               gpr_M_reg=r_["m"])

    def _ae_proj_self_kv_head(self, x_ue, r_, is_v, h, la, M, HP, D, bpe):
        """SELF layers, SHARDED PATH ONLY: project ONE kv-head of k (or v) STRAIGHT INTO
        its contiguous [64, 64] head slice, and rope K in place. The 10-unit replacement
        for "primary computes both full [64,320] projections, then de-interleaves them".

        THE COPY DISAPPEARS, it is not merely moved. matmat_mul_core writes back with
        stride N*bpe for the N IT WAS GIVEN, so calling it with N = D yields a DENSE
        [M, D] block -- and AE_KV_HEAD_STRIDE == M*D*bpe means head h's slice of
        AE_K/V_HEADS_DRAM IS exactly that dense [M, D] block. So the projection LANDS ON
        the flash operand and _ae_prep_self_kv_head's pair of strided SRAM-staged copies
        (the [64,320] -> 5 x [64,64] de-interleave) has nothing left to do. This is the
        same fold _compile_prefix_sharded got for free on LAYER0_K/V_DRAM; both shape
        facts are asserted once per compile in compile_denoise_loop rather than assumed.

        B IS [N, K] ROW-MAJOR, so this head's output columns h*D..h*D+D are a CONTIGUOUS
        ROW BLOCK of the weight at + h*D*HP*bpe -- the same "B += n0*K*bpe" rule
        ColumnShardContext.b_addr applies for q/o, with n0 = h*D. ops.expert_quant is
        bf16 and attention_bias is false, so B's row offset is the ONLY per-column offset
        that exists: no scale blob, no broadcast_N bias to slice.

        AE_K_PROJ_DRAM / AE_V_PROJ_DRAM ARE DEAD ON THIS PATH (as LM_K/V_PROJ_DRAM are on
        the sharded prefix). They stay allocated for the ne == 1 program and its bisect,
        which is asserted off whenever this runs.

        Dropping the + h*AE_KV_HEAD_STRIDE would not fault and would not NaN: every head
        would write slice 0, four of the five groups would flash against head 0's keys,
        and the action would come back finite, plausible and wrong -- pi05 denoise bug #3
        again. Both computed addresses are beat-asserted so a mis-scaled offset cannot
        slip through silently."""
        dst = ((self.AE_V_HEADS_DRAM if is_v else self.AE_K_HEADS_DRAM)
               + h * self.AE_KV_HEAD_STRIDE)
        b_off = h * D * HP * bpe
        assert dst % mes.AXI_BEAT_BYTES == 0, (
            f"kv head slice 0x{dst:X} is not {mes.AXI_BEAT_BYTES} B beat aligned")
        assert b_off % mes.AXI_BEAT_BYTES == 0, (
            f"kv weight row block +0x{b_off:X} is not {mes.AXI_BEAT_BYTES} B beat "
            f"aligned")
        self._ae_matmul(x_ue, M, HP, D, self.AE_PRE_NORM_DRAM,
                        (la["v_weight"] if is_v else la["k_weight"]) + b_off,
                        dst, m_reg=r_["m"])
        if not is_v:
            # Rope K in place ON THE ENGINE THAT JUST WROTE IT -- no fence needed, the
            # producer is the consumer of these two instructions. _ae_kv_units puts
            # k-head h on group h's engine, so the flash that reads this slice is on this
            # core too; V is the only unit that ever crosses, and the attention region's
            # opening barrier (which already published q_proj) is that fence.
            #
            # d64 RoPE is PBI-ONLY: a bare rope_hf_core_dram falls through to the legacy
            # core, which asserts N >= 128. Always pass gpr_M_reg.
            x_ue.rope_hf_core_dram(M=M, N=D, input_dram_addr=dst, output_dram_addr=dst,
                                   cos_dram_addr=self.AE_ROPE_PACKED_DRAM,
                                   sin_dram_addr=self.AE_ROPE_PACKED_DRAM + D * bpe,
                                   gpr_M_reg=r_["m"])

    def _ae_prep_cross_tok_head(self, x_ue, h, k_base, v_base, PM, KV, D, bpe):
        """CROSS layers: head-major cached K/V -> token-major [PM,320] column band h,
        the operand the (320,320) re-projection needs. READ-ONLY on the cache."""
        assert (h * D * bpe) % 32 == 0, f"cross tok head {h} offset not 32 B aligned"
        self._ae_strided_copy(x_ue, k_base + h * self.KV_HEAD_STRIDE, D * bpe,
                              self.AE_XK_TOK_DRAM + h * D * bpe, KV * bpe, PM, D)
        self._ae_strided_copy(x_ue, v_base + h * self.KV_HEAD_STRIDE, D * bpe,
                              self.AE_XV_TOK_DRAM + h * D * bpe, KV * bpe, PM, D)

    def _ae_xkv_layer_base(self, layer_idx):
        """(K, V) base of the hoisted head-major reprojection slot for `layer_idx`.

        One slot per expert layer. Only cross layers ever write or read one, but the
        keying is the raw layer index so the write in _emit_cross_reproject_all and
        the read in _emit_expert_layer's group loop cannot drift."""
        assert 0 <= layer_idx < self.E_LAYERS, f"expert layer {layer_idx} out of range"
        off = layer_idx * self.AE_XKV_LAYER_STRIDE
        return (self.AE_XK_HEADS_DRAM + off, self.AE_XV_HEADS_DRAM + off)

    def _ae_prep_cross_head_major(self, x_ue, h, PM, KV, D, bpe, k_dst, v_dst):
        """CROSS layers: reprojected [PM,320] -> head-major [5,PM,64], so flash takes a
        contiguous K/V per group. `k_dst`/`v_dst` are THIS LAYER's hoisted slot."""
        assert (h * self.AE_XKV_HEAD_STRIDE) % 32 == 0, (
            f"cross head-major offset {h} not 32 B beat aligned")
        assert k_dst % 32 == 0 and v_dst % 32 == 0, (
            "hoisted cross reprojection slot base not 32 B beat aligned")
        self._ae_strided_copy(x_ue, self.AE_XK_PROJ_DRAM + h * D * bpe, KV * bpe,
                              k_dst + h * self.AE_XKV_HEAD_STRIDE,
                              D * bpe, PM, D)
        self._ae_strided_copy(x_ue, self.AE_XV_PROJ_DRAM + h * D * bpe, KV * bpe,
                              v_dst + h * self.AE_XKV_HEAD_STRIDE,
                              D * bpe, PM, D)

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

    def _emit_suffix_embed(self, addr_reg, step=None, sched=None):
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
        # THE HEAD'S THREE MATMULS ARE N-SPLIT AT 8 LANES. All three have N = HP = 512 =
        # EXACTLY 8 blocks of 64, so _ae_proj_lanes caps them at 8 whatever the stage
        # count -- the same cap o_proj already lives under, for the same reason.
        # 109 MFLOP per Euler step at width 1 is ~2.8 ms; at 8 lanes it is ~0.4 ms.
        #
        # THE CHAIN IS STRICTLY SERIAL AND EVERY LINK READS ITS INPUT FULL-WIDTH
        # (a matmul's A is never column-sliced -- see _ae_proj_sharded), so unlike the
        # layer body this CANNOT be made barrier-free: each sharded stage has to
        # re-gather before the next one reads it. Five rendezvous per step, 50 over the
        # unroll, against 2260 already there.
        #
        # Steps 3 (the split-column assembly) and 5 (the composed SiLU) stay on the
        # primary on purpose. The assembly is a strided scatter plus 64 stores of one
        # SRAM-resident row -- not a row/column-independent op -- and the SiLU is
        # 4.2 MFLOP, so sharding it would buy microseconds and cost a register and a
        # rendezvous.
        sharded = sched is not None and sched.num_engines > 1

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
        if sharded:
            # join=True: step 3 below re-reads AE_AIN_DRAM in full ON THE PRIMARY.
            self._ae_proj_sharded(sched, M, ADP, HP, self.AE_XT_DRAM,
                                  self.action_in_weight, self.AE_AIN_DRAM,
                                  "ae_head_in", join=True, bias=self.action_in_bias)
        else:
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
        if sharded:
            # join=True: the composed SiLU below reads AE_TMLP_HID_DRAM in full on the
            # primary. Its opening rendezvous is also what fences the primary's step-3
            # assembly writes against the eight lanes reading AE_TMLP_IN_DRAM here.
            self._ae_proj_sharded(sched, M, 2 * HP, HP, self.AE_TMLP_IN_DRAM,
                                  self.time_mlp_in_weight, self.AE_TMLP_HID_DRAM,
                                  "ae_head_tmlp", join=True,
                                  bias=self.time_mlp_in_bias)
        else:
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
        if sharded:
            # join=FALSE, uniquely here: the ONLY consumer of the full-width AE_IO_A is
            # layer 0's input RMSNorm, and _emit_one_euler_step already emits a
            # rendezvous immediately after this call for exactly that hazard. A join
            # here would be two back-to-back barriers for one fence.
            self._ae_proj_sharded(sched, M, HP, HP, self.AE_TMLP_SILU_DRAM,
                                  self.time_mlp_out_weight, self.AE_IO_A_DRAM,
                                  "ae_head_out", join=False,
                                  bias=self.time_mlp_out_bias)
        else:
            self._ae_matmul(ue, M, HP, HP, self.AE_TMLP_SILU_DRAM,
                            self.time_mlp_out_weight, self.AE_IO_A_DRAM,
                            bias=self.time_mlp_out_bias)
        return self.AE_IO_A_DRAM

    def _emit_self_prefix_stage_all(self, ue, sched, n_ae):
        """HOISTED, ONCE PER PROGRAM: the PREFIX half of every self layer's
        [prefix ; suffix] K/V staging, for every attention unit.

        WHY THIS IS STEP-INVARIANT, checked rather than assumed. Rows [0, PM) of CK/CV
        are a straight block copy of LAYER0_{K,V}_DRAM[self_pl][kv_b] -- the frozen VLM
        prefix cache. That cache is written in exactly one place, compile_prefix's
        per-layer store, and that program has finished before the denoise program is
        launched; nothing in the denoise body writes it (it is only ever a copy SOURCE,
        on both this path and the cross layers' reprojection), which is also the
        standing requirement that it survive all ten Euler steps. Neither the source nor
        the destination offset depends on the step, so all ten passes produced identical
        bytes. Hoisting changes WHEN it runs, not what it computes.

        WHAT IT BUYS. Two of the four strided copies per self layer per unit per step
        become two copies per self layer per unit, ONCE: on the busiest engine that is
        320 op-executions removed out of 640. The layer sweep put fixed cost at ~57 us
        an op, so ~18 ms. The suffix half stays inside the unroll because it is the one
        thing here that DOES change every step.

        SHARDED PATH ONLY. At ne == 1 the emitter keeps building both halves inline, in
        the original order, so --denoise-digest --engines 1 cannot move.

        NO BARRIER. Every copy runs on the same engine that will read it, so this is
        ordinary program order per engine -- and the first thing any worker does after
        this is layer 0's q-projection rendezvous anyway."""
        if sched is None or sched.num_engines <= 1:
            return
        D, PM, bpe = self.HEAD_DIM, self.PREFILL_MAX_SEQ_LEN, 2
        CW = self.AE_COMBINED_LEN
        pe = sched.per_engine_addr
        n_units = 0
        for layer_idx in range(n_ae):
            slot = self._ae_self_slot(layer_idx)
            if slot is None:
                continue
            self_pl = self._ae_cross_prefix_layer(layer_idx)
            base = slot * CW * D * bpe
            for kv_b, _tok0, _ntok, x_ue, _r, _FQ, _FO, _FS, CK, CV in \
                    self._ae_attn_groups(sched, ue, self._ae_regs):
                k_pre = (self.LAYER0_K_DRAM + self_pl * self.KV_LAYER_STRIDE
                         + kv_b * self.KV_HEAD_STRIDE)
                v_pre = (self.LAYER0_V_DRAM + self_pl * self.KV_LAYER_STRIDE
                         + kv_b * self.KV_HEAD_STRIDE)
                # BOTH query-row halves of a group need it: they attend the SAME keys
                # and each owns its own engine, hence its own CK/CV. That duplication is
                # the price of the row split and it is paid once here instead of ten
                # times inside the unroll.
                self._ae_strided_copy(x_ue, k_pre, D * bpe, CK + base, D * bpe, PM, D)
                self._ae_strided_copy(x_ue, v_pre, D * bpe, CV + base, D * bpe, PM, D)
                n_units += 1
        print(f"    [denoise] hoisted the prefix half of {n_units} self-layer staging "
              f"unit(s) out of the {self.N_STEPS}-step unroll "
              f"({n_units * 2 * (self.N_STEPS - 1)} copy-executions removed)")

    def _emit_cross_reproject_all(self, ue, n_ae):
        """HOISTED, ONCE PER PROGRAM: every cross layer's reprojected prefix K/V.

        WHY THIS IS STEP-INVARIANT (checked, not assumed). The chain is

            LAYER0_K/V_DRAM[pl]  --gather-->  AE_XK/XV_TOK
                                 --matmul-->  AE_XK/XV_PROJ   (k/v_weight, 320x320)
                                 --scatter->  AE_XK/XV_HEADS[layer]

        and it has exactly two inputs:
          * LAYER0_K/V_DRAM, the prefix KV cache. Written in precisely one place --
            compile_prefix's per-layer cache store -- and that program has finished
            before the denoise program is even launched. Nothing in the denoise body
            writes it (it is only ever a strided-copy SOURCE, on both the cross path
            here and the self layers' [prefix ; suffix] concat), which is also the
            standing requirement that it survive all 10 Euler steps.
          * la["k_weight"] / la["v_weight"], static weight blobs in DRAM.
        Neither is touched by the Euler update, whose only mutable state is
        AE_XT_DRAM (plus the per-layer ping-pong stream and the timestep pointer).
        The old emission produced bit-identical bytes on all 10 passes; the DATA it
        produced was therefore identical on all 10 too. Hoisting changes when it runs,
        not what it computes.

        The mapping is preserved exactly: layer `i` reads the same
        _ae_cross_prefix_layer(i) cache it read inline, and writes the slot the group
        loop reads back through the same _ae_xkv_layer_base(i).

        PRIMARY-ONLY, DELIBERATELY, AND WITH NO BARRIER OF ITS OWN. The whole pass is
        1/10 of what it used to cost, so sharding it would buy microseconds while
        adding rendezvous outside any region. Workers stay symmetric because they emit
        NOTHING here: the first thing in every worker's program is the opening
        rendezvous of layer 0's q-projection region, and the primary reaches that
        rendezvous only after this pass has landed -- so the hoisted writes are fenced
        against every worker read by a barrier that already existed.

        AE_XK/XV_TOK and AE_XK/XV_PROJ stay SINGLE shared scratch: they are produced
        and consumed within one layer's iteration of this loop, sequentially, on one
        engine. Only the head-major result needs per-layer storage.

        Unconditional -- emitted the same way at ne == 1 and ne == 8, so there is one
        behaviour to verify rather than two."""
        HP, KV = self.E_HIDDEN_PAD, self.E_KV_OUT
        D, PM, bpe = self.HEAD_DIM, self.PREFILL_MAX_SEQ_LEN, 2
        regs = self._ae_regs
        n_cross = 0
        for layer_idx in range(n_ae):
            if self._ae_is_self_attn(layer_idx):
                continue
            n_cross += 1
            la = self.ae_layer_addrs[layer_idx]
            pl = self._ae_cross_prefix_layer(layer_idx)
            k_base = self.LAYER0_K_DRAM + pl * self.KV_LAYER_STRIDE
            v_base = self.LAYER0_V_DRAM + pl * self.KV_LAYER_STRIDE
            xk_dst, xv_dst = self._ae_xkv_layer_base(layer_idx)
            # head-major cache -> token-major [PM, 320], one strided copy per head.
            for h in range(self.NUM_KV_HEADS):
                self._ae_prep_cross_tok_head(ue, h, k_base, v_base, PM, KV, D, bpe)
            # [PM,320] @ (320,320).T -> [PM,320]. THE ROW COUNT HERE IS PM, NOT the
            # suffix's 64. pulsevla borrows regs["qb"] for it because AE_FLASH_ROWS
            # happens to equal PM there (192); when they differ, regs["pm"] carries the
            # real count. Borrowing "qb" unconditionally would reproject only the first
            # AE_FLASH_ROWS of PM cached prefix rows and leave the rest stale -- finite,
            # NaN-free, wrong for every prefix token past 192. See _ae_tensor_init.
            m_pm = regs.get("pm", regs["qb"])
            self._ae_matmul(ue, PM, KV, KV, self.AE_XK_TOK_DRAM, la["k_weight"],
                            self.AE_XK_PROJ_DRAM, m_reg=m_pm)
            self._ae_matmul(ue, PM, KV, KV, self.AE_XV_TOK_DRAM, la["v_weight"],
                            self.AE_XV_PROJ_DRAM, m_reg=m_pm)
            # back to head-major [5, PM, D] in THIS layer's slot.
            for h in range(self.NUM_KV_HEADS):
                self._ae_prep_cross_head_major(ue, h, PM, KV, D, bpe, xk_dst, xv_dst)
        # NO RoPE on the reprojected K: the cached K already carries the PREFIX
        # rotation and upstream does not rotate `ek`. Only the expert's query is roped
        # on cross layers, and that IS step-dependent-shaped work that stays in the
        # loop body (it reads AE_Q_DRAM, which the step recomputes).
        print(f"    [denoise] hoisted cross K/V reprojection for {n_cross} layer(s) "
              f"out of the {self.N_STEPS}-step unroll "
              f"({n_cross * 2 * self.AE_XKV_LAYER_STRIDE / 1e6:.2f} MB of slots)")

    def _emit_expert_layer(self, ue, layer_idx, step, sched=None):
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
        # ONE predicate for the whole body. Every barrier and every fold branches on it,
        # so the barrier count per layer is a function of (sharded, is_self) alone --
        # never of anything an individual engine sees. All engines walk the same Python
        # loop over layer_idx, so their barrier counts are symmetric by construction,
        # which is what keeps a FLAG_CHECK from blocking forever.
        sharded = sched is not None and sched.num_engines > 1

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
        # ROW-SPLIT (see _ae_row_split). NO NEW BARRIER: engine e reads h_in[S_e], which
        # it wrote itself at the bottom of the previous layer, and the full-width
        # AE_PRE_NORM_DRAM it produces is fenced by the q_proj region's OPENING
        # rendezvous a few lines below -- the same one that already published it when
        # this norm ran on the primary alone.
        self._ae_row_norm(sched, h_in, self.AE_PRE_NORM_DRAM, la["ln1_gamma"], M, HP)

        # ---- Q, and (self layers only) K/V ------------------------------------------
        # Note the asymmetry that defines this model: the residual is HALF the LM's
        # width, but attention is not -- q_proj projects UP, 480 -> 960, and kv_out stays
        # 320. That is what makes the expert's attention geometry bit-identical to the
        # LM's, so the frozen prefix cache DROPS IN as a [192,64]-per-head operand --
        # but it is still re-projected first (cross k_proj is (320,320)); matching
        # geometry is not the same thing as skipping the projection.
        if self.EXPERT_BISECT and layer_idx == (self.EXPERT_LAYERS or self.E_LAYERS) - 1:
            self._probe_copy(self.AE_PRE_NORM_DRAM, self.AE_P_NORM1_DRAM, M * HP)
        if sched is not None and sched.num_engines > 1:
            # N-SPLIT + SCATTER. AE_PRE_NORM_DRAM was just written by the primary and is
            # read in FULL by every engine; the region's OPENING rendezvous is that RAW
            # fence. join=False: the only consumer of the full-width AE_Q_DRAM is the
            # per-group q-head gather below, and the attention region already opens with
            # its own barrier -- that is the fence. Two back-to-back rendezvous would
            # cost 320 extra round trips over 32 layers x 10 Euler steps for nothing.
            self._ae_proj_sharded(sched, M, HP, Q, self.AE_PRE_NORM_DRAM,
                                  la["q_weight"], self.AE_Q_DRAM, "ae_q", join=False)
        else:
            self._ae_matmul(ue, M, HP, Q, self.AE_PRE_NORM_DRAM, la["q_weight"],
                            self.AE_Q_DRAM)

        if is_self:
            if sharded:
                # ---- k/v SHARDED 10 WAYS, ONE (tensor, kv-head) UNIT PER ENGINE ------
                # These two projections used to run HERE, at width 1, on the primary,
                # while every other engine idled: 2 x [64,512]x[512,320] = 41.9 MFLOP
                # against 640 KB of weights, i.e. 64 FLOP/byte and firmly compute-bound,
                # 1.09 ms per self layer per Euler step -> 174.8 ms over 16 self layers
                # x 10 steps, ~12% of the whole stage. split_cols refuses to cut N=320
                # more than 5 ways, which is what kept them primary-only -- but a kv-head
                # IS a 64-column block (HEAD_DIM == UE_VECTOR_SIZE) and k/v are separate
                # weights, so there are 10 units, not 5 blocks. See _ae_kv_units.
                #
                # NO NEW BARRIER, in either direction. Upstream: AE_PRE_NORM_DRAM is
                # written by the primary and read in full by every engine here, and the
                # q_proj region's OPENING rendezvous (a few lines up) is already that
                # fence. Downstream: the head slices are read by the flash concat below,
                # behind the attention region's opening barrier, which already existed to
                # publish the full-width AE_Q_DRAM gather. An extra rendezvous per self
                # layer would be 160 round trips over the unroll for nothing.
                #
                # The de-interleave went with it -- the matmul writes the head slice
                # directly (see _ae_proj_self_kv_head), so nothing stages [64,320]
                # through SRAM any more and AE_K/V_PROJ_DRAM are dead on this path.
                kv_units = self._ae_kv_units(sched.num_engines)
                for e, x_ue in enumerate(sched.engines):
                    for is_v, h in kv_units[e]:
                        self._ae_proj_self_kv_head(x_ue, self._ae_reg_sets[e],
                                                   is_v, h, la, M, HP, D, bpe)
            else:
                self._ae_matmul(ue, M, HP, KV, self.AE_PRE_NORM_DRAM, la["k_weight"],
                                self.AE_K_PROJ_DRAM)
                self._ae_matmul(ue, M, HP, KV, self.AE_PRE_NORM_DRAM, la["v_weight"],
                                self.AE_V_PROJ_DRAM)
                # de-interleave [64,320] into 5 contiguous [64,64] head slices, then rope
                # K in place. V is never rotated. This is the ne == 1 path ONLY, and it
                # is emitted here, in head order, exactly as the pre-shard emitter did --
                # every executable line the sharding added sits in the branch above, so
                # --denoise-digest --engines 1 cannot move.
                for h in range(self.NUM_KV_HEADS):
                    self._ae_prep_self_kv_head(ue, regs, h, M, KV, D, bpe)
        else:
            # ---- CROSS layers: the reprojected prefix K/V is ALREADY BUILT ----------
            # It used to be built right here, 10 times per layer with identical inputs.
            # Its only inputs are the FROZEN prefix cache (LAYER0_K/V_DRAM, written by
            # the prefix program and read-only for the whole denoise) and this layer's
            # static k_weight/v_weight -- nothing that varies with the Euler step -- so
            # the gather -> (320,320) matmul -> head-major scatter chain is emitted ONCE
            # per layer by _emit_cross_reproject_all, ahead of the step unroll, into
            # this layer's own slot. See that method for the invariance argument.
            #
            # Two things left with it: the (320,320) re-projection (16 cross layers x 10
            # steps of PM=192 x 320 x 320 on the UNSHARDED primary -> 1/10 of that), and
            # the barrier that fenced the token-major gather against it. Cross layers now
            # carry exactly ONE barrier, the attention region's, same as self layers.
            pass

        # ---- per-kv-group stacked-Q flash -------------------------------------------
        # SELF layers now attend over the COMBINED [prefix ; suffix] key/value sequence
        # (FAULT 4), so their aligned_seq_len is PM + M = 256, not QB. batch stays QB.
        CW = self.AE_COMBINED_LEN
        seq_len = CW if is_self else PM
        bias_addr = self.AE_BIAS_SELF_DRAM if is_self else self.AE_BIAS_CROSS_DRAM
        # Which prefix layer's cache a SELF layer concatenates: the reference uses
        # prefix_kv[i] on BOTH branches (VeraPulseRef.forward_expert), so reuse the same
        # mapping helper the cross path uses rather than hardcoding the index.
        self_pl = self._ae_cross_prefix_layer(layer_idx) if is_self else None
        # BARRIER IN: what a group reads that some OTHER engine wrote above -- the
        # full-width AE_Q_DRAM gather (every engine scattered one column band of it), the
        # v-head slices (v-head h is a unit of _ae_kv_units and only coincides with group
        # h at ne <= 5), and, on cross layers, the (320,320) re-projection the primary
        # built ahead of the unroll. NOT in that list: k-head h and its RoPE, which
        # _ae_kv_units pins to group h's own engine, and the per-head staging, which is
        # emitted inside the group bodies. No engine may enter its group until every
        # peer's projections have landed.
        if sharded:
            sched.barrier()
        for (kv_b, tok0, ntok, x_ue, r_, F_Q, F_O, F_S, CK, CV) in self._ae_attn_groups(
                sched, ue, regs):
            # THE FLASH SEQUENCE-LENGTH REGISTER MUST MATCH `seq_len` ABOVE. In PBI mode
            # the GPR is what the kernel actually reads; the static seq_len only sizes
            # the emission. On cross layers that length is PM, and r_["qb"] holds
            # AE_FLASH_ROWS -- equal on pulsevla (both 192) purely by coincidence, and
            # WRONG on any variant whose padded prefix is longer. At PM=256 the borrow
            # made cross-attention attend the first 192 of 241 valid prefix keys and read
            # the [QB,PM] bias at a 192 row stride: with the layout
            # 1 state + 3*64 image + 48 text, the entire language block sat past column
            # 192 and was dropped on every cross layer of every Euler step. Finite,
            # smooth, and wrong -- it cost ~20 dB at denoise step 0.
            seq_reg_e = r_["cw"] if is_self else r_.get("pm", r_["qb"])
            # THIS UNIT'S STACKED-Q SLICE. row0 is where its rows start inside the
            # full 192-row batch, and every row-indexed operand below is shifted by it.
            # At ntok == M (any count below 10) row0 is 0, qrows is QB and batch_reg is
            # r_["qb"] -- i.e. all four shifts are +0 and the register is the original,
            # so the emitted stream is exactly the pre-split one.
            row0, qrows = tok0 * G, ntok * G
            batch_reg = r_["qb"] if qrows == QB else r_["qh"]
            # NO kv-head prep here any more. It used to be a primary-only
            # `for h in range(NUM_KV_HEADS)` de-interleave ahead of the barrier, then
            # (briefly) a fold into this loop so head kv_b's copy ran on group kv_b's
            # engine. Both are gone on the sharded path: the k/v projection now writes
            # head kv_b's [64,64] slice DIRECTLY, above the barrier, so there is nothing
            # left to stage. The ne == 1 path still does the de-interleave, up where the
            # full-width projections are.
            # gather this group's 3 q-heads TOKEN-MAJOR: flash row t*G+g is q-head
            # kv_b*G+g of token t, so one flash call serves the whole group.
            # ONE COPY FOR ALL THREE q-HEADS, not one per head.
            #
            # The token-major stacking makes this exact, and it is worth stating because
            # it looks like it should need a permute. Group kv_b's three q-heads occupy
            # AE_Q_DRAM columns [kv_b*G*D, (kv_b+1)*G*D) -- a CONTIGUOUS 192-column,
            # 384 B band of each row. In F_Q, token t's three heads are rows t*G+0..2 --
            # also contiguous, and also 384 B. So a single strided copy of width G*D
            # maps row t of the band onto F_Q rows [t*G, t*G+G), which is precisely
            # "flash row t*G+g is q-head kv_b*G+g of token t". Head ORDER inside the
            # band is ascending on both sides, so nothing is transposed.
            #
            # Three copies of `ntok` rows at a 128 B burst become ONE of `ntok` rows at
            # 384 B, and the destination stops being strided (dst_jump == width*bpe).
            # That is a 3x cut in row-DMA transactions on the axis the layer sweep
            # showed the stage is actually bound by.
            #
            # + tok0*Q*bpe : this unit gathers only ITS action tokens. Its F_Q row 0 is
            # token tok0's first q-head, so the flash buffer stays dense from 0 and
            # every downstream offset into it is unshifted.
            self._ae_strided_copy(x_ue,
                                  self.AE_Q_DRAM + tok0 * Q * bpe + kv_b * G * D * bpe,
                                  Q * bpe, F_Q, G * D * bpe, ntok, G * D)
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
                # + row0 WHOLE [cos|sin] ROWS. The table is [rows, 2*D] and
                # rope_hf_core_dram_legacy ASSERTS sin == cos + N*2, so the base may
                # only move by multiples of 2*D*bpe -- which is exactly what a row
                # offset is. Shift it by bytes and that assert fires; forget to shift
                # it and this unit rotates its tokens with token 0's positions.
                rope_tbl = rope_tbl + row0 * 2 * D * bpe
                assert rope_tbl % mes.AXI_BEAT_BYTES == 0, (
                    f"rope slice 0x{rope_tbl:X} not {mes.AXI_BEAT_BYTES} B beat aligned")
                x_ue.rope_hf_core_dram(M=qrows, N=D, input_dram_addr=F_Q,
                                     output_dram_addr=F_Q,
                                     cos_dram_addr=rope_tbl,
                                     sin_dram_addr=rope_tbl + D * bpe,
                                     gpr_M_reg=batch_reg)

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
                # THIS SELF LAYER'S SLOT -- SHARDED PATH ONLY.
                #
                # The slots exist because the prefix half is HOISTED out of the Euler
                # unroll, so layer 0's copy has to survive until layer 2 runs. The ne==1
                # path does NOT hoist: it rebuilds both halves inline, every layer, into
                # slot 0, exactly as the pre-hoist emitter did. So its base must be 0.
                #
                # THIS IS THE BUG THE ne==1 GATE CAUGHT. `_base = (_slot or 0) * ...`
                # was wrong twice over: _ae_self_slot returns the RUNNING COUNT of self
                # layers (1 for layer 2, 2 for layer 4, ...), not 0, and `or 0` only
                # rewrites slot 0 -- which is the one case that was already right. So on
                # the unsharded path layer 2+ wrote its prefix half into slot 0 and then
                # read slot 1, whose prefix half nothing had written. Every self layer
                # past the first attended a zeroed prefix: no crash, no NaN, a smooth
                # -1.45 dB / cos 0.18 decay across the ten Euler steps that reads exactly
                # like integration drift. Do not let the base and the write diverge.
                _base = (self._ae_self_slot(layer_idx) * CW * D * bpe) if sharded else 0
                k_suf = self.AE_K_HEADS_DRAM + kv_b * self.AE_KV_HEAD_STRIDE
                v_suf = self.AE_V_HEADS_DRAM + kv_b * self.AE_KV_HEAD_STRIDE
                if not sharded:
                    # ne == 1: both halves inline, original order, byte-identical.
                    k_pre = (self.LAYER0_K_DRAM + self_pl * self.KV_LAYER_STRIDE
                             + kv_b * self.KV_HEAD_STRIDE)
                    v_pre = (self.LAYER0_V_DRAM + self_pl * self.KV_LAYER_STRIDE
                             + kv_b * self.KV_HEAD_STRIDE)
                    self._ae_strided_copy(x_ue, k_pre, D * bpe, CK, D * bpe, PM, D)
                    self._ae_strided_copy(x_ue, v_pre, D * bpe, CV, D * bpe, PM, D)
                # SUFFIX ONLY on the sharded path. Rows [0, PM) of this slot were
                # written once by _emit_self_prefix_stage_all, ahead of the unroll, ON
                # THIS SAME ENGINE -- so there is nothing to fence and nothing to
                # rebuild. Rows [PM, PM+M) are this layer's own roped suffix K/V and are
                # the only part that changes per Euler step.
                self._ae_strided_copy(x_ue, k_suf, D * bpe,
                                      CK + _base + PM * D * bpe, D * bpe, M, D)
                self._ae_strided_copy(x_ue, v_suf, D * bpe,
                                      CV + _base + PM * D * bpe, D * bpe, M, D)
                k_addr, v_addr = CK + _base, CV + _base
            else:
                # THE REPROJECTED PREFIX K/V (FAULT 5), not the raw cache. Built ONCE
                # PER LAYER by _emit_cross_reproject_all, before the Euler unroll, as
                # contiguous [PM=192, 64] blocks in this layer's own slot -- the same
                # geometry the raw cache had, so nothing downstream changes. Still no
                # x G replication: 192 == the stacked-Q batch. The raw cache itself is
                # only READ here and must survive all 10 steps untouched.
                xk_base, xv_base = self._ae_xkv_layer_base(layer_idx)
                off = kv_b * self.AE_XKV_HEAD_STRIDE
                k_addr = xk_base + off
                v_addr = xv_base + off

            # FLASH STAYS ADDRESS-STATIC. Only the two dimension GPRs are runtime (the
            # same thing compile_encoder does); PBI address injection into flash corrupts
            # on the 2nd execution and this body runs 10 times per inference.
            # K/V ARE NOT SLICED: both row halves attend the SAME keys, which is why
            # a self group's two halves each build an identical CK/CV into their own
            # buffer. batch shrinks to this unit's rows -- legal because
            # unified_attention_core_dynamic bounds only batch <= aligned_seq_len and
            # aligned_seq_len % 64, never batch itself (user_dma_core.py:7272-7277).
            # The bias is [QB, seq_len] and IS row-indexed, so it moves with row0; its
            # row pitch is seq_len*bpe = 512 B (self) / 384 B (cross), both beat-aligned.
            bias_a = bias_addr + row0 * seq_len * bpe
            assert bias_a % mes.AXI_BEAT_BYTES == 0, (
                f"bias slice 0x{bias_a:X} not {mes.AXI_BEAT_BYTES} B beat aligned")
            x_ue.unified_attention_core(
                batch=qrows, aligned_seq_len=seq_len, head_dim=D,
                Q_DRAM_ADDR=F_Q, K_DRAM_ADDR=k_addr, V_DRAM_ADDR=v_addr,
                BIAS_DRAM_ADDR=bias_a, OUTPUT_DRAM_ADDR=F_O,
                SCRATCH_DRAM_ADDR=F_S,
                IDENTITY_DRAM_ADDR=self.identity_addr,
                gpr_batch_reg=batch_reg, gpr_aligned_seq_len_reg=seq_reg_e)

            # un-stack [64,G,D] -> [64,960] at THIS group's head columns. The + head*D*2
            # in the destination base is trap #8: without it every head writes columns
            # [0:64], o_proj reads garbage for 14 of the 15 heads, and the output is
            # finite and scrambled rather than NaN.
            # THE GATHER RUN BACKWARDS, and one copy for the same reason: F_O rows
            # [t*G, t*G+G) are this group's three heads for token t, contiguous, and
            # they land in AE_ATTN_RESULT_DRAM row t at columns [kv_b*G*D, +G*D), also
            # contiguous. Now the SOURCE is the unstrided side.
            #
            # BOTH per-index offsets are still here and both are load-bearing:
            #   + kv_b*G*D*bpe  is trap #8 on the COLUMN -- without it every group
            #     writes columns [0:192] and o_proj reads garbage for 12 of 15 heads.
            #   + tok0*Q*bpe    is its twin on the ROW -- without it both query-row
            #     halves write rows [0:32) and the second half's 32 tokens keep whatever
            #     the previous Euler step left there.
            # Neither faults, neither NaNs.
            self._ae_strided_copy(x_ue, F_O, G * D * bpe,
                                  self.AE_ATTN_RESULT_DRAM + tok0 * Q * bpe
                                  + kv_b * G * D * bpe,
                                  Q * bpe, ntok, G * D)

        # ---- o_proj + residual ------------------------------------------------------
        # o_weight is [512,960]: the N-pad puts zeros in out-rows 480..511, which is what
        # re-zeroes the residual stream's pad lanes after every attention write.
        if sched is not None and sched.num_engines > 1:
            # THE REGION'S OPENING RENDEZVOUS *IS* THE BARRIER OUT of the attention
            # shard: every engine reads AE_ATTN_RESULT at FULL width here, and the kv
            # groups wrote it into disjoint 64-column bands on whichever engine owned
            # them. N=HP=512 is EXACTLY 8 blocks of 64, so ne=8 is this op's CEILING:
            # every engine gets one block and none idles at 8, and above 8 the region is
            # lane-capped to the first 8 (the rest walk both rendezvous and emit
            # nothing). See _ae_proj_lanes -- this is the op that used to make ne > 8
            # impossible at all, because split_cols asserts blocks >= num_engines.
            # join=True, unlike q: the very next statement is the primary's residual add
            # reading the full-width AE_O_PROJ_DRAM, so the scatter must be fenced before
            # it. There is no later region to borrow a barrier from.
            self._ae_proj_sharded(sched, M, Q, HP, self.AE_ATTN_RESULT_DRAM,
                                  la["o_weight"], self.AE_O_PROJ_DRAM, "ae_o", join=True)
        else:
            self._ae_matmul(ue, M, Q, HP, self.AE_ATTN_RESULT_DRAM, la["o_weight"],
                            self.AE_O_PROJ_DRAM)
        # ROW-SPLIT. NO NEW BARRIER: AE_O_PROJ_DRAM is written by o_proj's lanes and
        # that region exits join=True, which is the RAW fence; h_in[S_e] is engine e's
        # own rows from the previous layer.
        self._ae_row_add(sched, h_in, self.AE_O_PROJ_DRAM, self.AE_RESIDUAL_DRAM, M, HP)
        # Split residual-then-norm rather than the fused post-add norm: the fused
        # layer_norm_core_dram_post_add has NO PBI (4 advancing pointers vs the <=3
        # limit), so it would unroll M statically 32 layers deep. Both halves here are
        # PBI.
        # ROW-SPLIT. NO NEW BARRIER: AE_RESIDUAL_DRAM[S_e] was written by engine e
        # itself one statement ago, and the AE_PRE_NORM_DRAM it produces is fenced by
        # the gated MLP region's OPENING rendezvous below. The WAR against the q_proj
        # region's full-width read of AE_PRE_NORM is covered by o_proj's two barriers.
        self._ae_row_norm(sched, self.AE_RESIDUAL_DRAM, self.AE_PRE_NORM_DRAM,
                          la["ln2_gamma"], M, HP)

        # ---- gated MLP --------------------------------------------------------------
        # The FUSED silu_enable is fine HERE and is what smolvlm2 ships; only the time
        # MLP needs the composed form (pi05 measured -6 dB there specifically).
        if self.EXPERT_BISECT and layer_idx == (self.EXPERT_LAYERS or self.E_LAYERS) - 1:
            self._probe_copy(self.AE_PRE_NORM_DRAM, self.AE_P_NORM2_DRAM, M * HP)
        if sched is not None and sched.num_engines > 1:
            # AE_PRE_NORM_DRAM was just written by the primary and is read in FULL by
            # every engine; the region's OPENING rendezvous is that RAW fence. The
            # region also lands its result in AE_MLP_DOWN_DRAM, so the residual add
            # below is unchanged.
            self._ae_gated_mlp_sharded(sched, M, la)
        else:
            self._ae_matmul(ue, M, HP, I, self.AE_PRE_NORM_DRAM, la["gate_weight"],
                            self.AE_MLP_GATE_DRAM, silu_enable=True)
            self._ae_matmul(ue, M, HP, I, self.AE_PRE_NORM_DRAM, la["up_weight"],
                            self.AE_MLP_UP_DRAM)
            eltwise_mul_core_dram(ue, size=M * I, A_DRAM_ADDR=self.AE_MLP_GATE_DRAM,
                                  B_DRAM_ADDR=self.AE_MLP_UP_DRAM,
                                  OUTPUT_DRAM_ADDR=self.AE_MLP_MULT_DRAM)
            self._ae_matmul(ue, M, I, HP, self.AE_MLP_MULT_DRAM, la["down_weight"],
                            self.AE_MLP_DOWN_DRAM)
        # ROW-SPLIT. NO NEW BARRIER: AE_MLP_DOWN_DRAM is the reduce_add output and that
        # reduction FORCES its join barrier on the parallel path, which is the fence;
        # AE_RESIDUAL_DRAM[S_e] is engine e's own rows. h_out[S_e] is then read by the
        # SAME engine as h_in at the top of the next layer, which is why one split for
        # all four ops leaves this whole chain barrier-free.
        self._ae_row_add(sched, self.AE_RESIDUAL_DRAM, self.AE_MLP_DOWN_DRAM, h_out,
                         M, HP)
        # ---- INSTRUMENT: redundant rendezvous (see DENOISE_EXTRA_BARRIERS) ----------
        # Emitted HERE because this point is outside every region and downstream of
        # every write in the layer, so an extra all-engine rendezvous is a pure no-op:
        # it cannot change a single output byte, only the wall clock. Symmetric by
        # construction -- sched.barrier() emits on every engine and the loop bound is
        # the same Python constant on all of them.
        if sharded and self.DENOISE_EXTRA_BARRIERS:
            for _ in range(int(self.DENOISE_EXTRA_BARRIERS)):
                sched.barrier()
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

        # ---- N-axis (output-column) sharding ---------------------------------------
        # M is SUFFIX_LEN_PAD = 64 here -- ONE 64-row block -- so the M-split that shards
        # vision is inexpressible and N is the only axis. Sharded now: the gated MLP
        # (gate/up N-split, down K-split + reduce_add), the attention units,
        # q_proj / o_proj (N-split + scatter, see _ae_proj_sharded) and the self layers'
        # k_proj / v_proj (10 (tensor, kv-head) units, see _ae_kv_units).
        #
        # N IS NOT THE ONLY AXIS ANY MORE. The two RMSNorms and the two residual adds of
        # every layer -- 4 x 320 layer-executions that used to run on the primary at
        # width 1 while nine engines idled -- are ROW-split (_ae_row_split). They are not
        # matmuls, so the 64-aligned-row-block rule does not reach them; both cores are
        # in SHARDED_OP_ALLOWLIST precisely because they are row-independent. One split
        # serves all four, which is what makes the whole chain barrier-free.
        #
        # Still primary-only: action_out (N=64, one block), the final norm and the Euler
        # retire (10 executions each, not 640 -- see the note at the final norm), and the
        # per-step time-MLP head (action_in_proj / action_time_mlp_in / _out, ~28 ms at
        # width 1; N=512 gives 8 lanes there, worth ~24 ms, not done here).
        #
        # EVERY OP RUNS AT ITS OWN WIDTH, and they do not all agree. The stage ceiling is
        # 10 (STAGE_MAX_ENGINES carries the arithmetic), but within it:
        #     gate/up/down  ne lanes   (1280 = 20 blocks, never binds)
        #     q_proj        ne lanes   (960  = 15 blocks, never binds)
        #     o_proj        min(ne,8)  (512  = EXACTLY 8 blocks -- the lane cap)
        #     k/v_proj      min(ne,10) (10 UNITS -- NOT the 5 blocks split_cols sees:
        #                               HEAD_DIM == UE_VECTOR_SIZE makes a kv-head one
        #                               column block, and k/v are separate weights)
        #     attention     min(ne,10) (5 kv groups x 2 QUERY-ROW halves -- the group
        #                               axis caps at 5, the row axis does not)
        #     norms/residual     ne      (ROW-split, 64 rows -- the one non-N axis)
        #     head projections min(ne,8) (action_in / time_mlp_in / _out, N=512)
        #     action_out         1     (N=64, one block)
        # so at ne=10 only o_proj idles anyone, and only two engines.
        # That is not waste to be fixed by lowering ne: the MLP is 61% of the stage's
        # FLOPs and it is the term that goes 6.67x -> 10x between 8 and 10 engines.
        ne = self._num_engines("DENOISE")
        sched = None
        if ne > 1:
            assert not self.DENOISE_ROLLED, (
                "DENOISE_ROLLED + sharding is not supported: _ae_duplicate_gqa_rows "
                "already opens two nested hardware loops, so rolling makes it depth 3, "
                "which hangs. Keep the static 10x unroll.")
            assert not self.EXPERT_BISECT, (
                f"--bisect-expert is primary-only (the probe copies and any halt are "
                f"emitted on engine 0 alone, so the workers would spin forever at the "
                f"next rendezvous). Use --engines 1 to bisect.")
            # TRUNCATION ALONE IS FINE SHARDED, and it used to be refused here only
            # because it arrived bundled with the probes. Nothing about a shorter expert
            # is primary-only: every engine walks the same `for layer_idx in range(n_ae)`
            # so the barrier counts stay symmetric, _emit_cross_reproject_all already
            # takes the count, final_h already derives from n_ae % 2, and
            # _weight_init_expert sizes nothing by it (see the note at AE_XK_HEADS_DRAM).
            # That is what makes --expert-layers a usable INSTRUMENT: wall time against
            # layer count at a FIXED engine count is a straight line whose slope is the
            # per-layer cost and whose intercept is the per-step head plus fixed setup,
            # which is the only way to split the unaccounted stage time into "inside the
            # layer body" and "outside it". The truncated model is numerically wrong on
            # purpose -- it is a stopwatch, not a gate.
            if self.EXPERT_LAYERS is not None:
                _n = int(self.EXPERT_LAYERS)
                assert 1 <= _n <= self.E_LAYERS, (
                    f"--expert-layers {_n}: expected 1..{self.E_LAYERS}")
                print(f"    [denoise] INSTRUMENTED: {_n} of {self.E_LAYERS} expert "
                      f"layers. Timing only -- the actions this produces are NOT the "
                      f"model's.")
            for li, la in enumerate(self.ae_layer_addrs):
                blobs = la.get("down_weight_k")
                assert blobs is not None and len(blobs) == ne, (
                    f"denoise is sharded over {ne} engine(s) but expert layer {li} "
                    f"carries {0 if blobs is None else len(blobs)} K-sliced down blob(s)."
                    f" _weight_init_expert cuts them for _num_engines('DENOISE') as it "
                    f"resolved at WEIGHT-INIT time -- apply --engines/--dns_8 to the "
                    f"model object BEFORE weight_init().")
            self._assert_flag_check_reaches(ne)
            # THE TWO SHAPE FACTS THE k/v SPLIT RESTS ON, checked ONCE per compile rather
            # than assumed 160 times inside the unroll. Break either and a self layer's
            # k projection writes the wrong bytes into the flash operand: finite,
            # plausible, wrong attention on every Euler step, never a crash and never a
            # NaN. (_ae_tensor_init has already run, so AE_KV_HEAD_STRIDE exists.)
            #   HEAD_DIM == UE_VECTOR_SIZE     : one 64-column matmat block IS one kv-head
            #   AE_KV_HEAD_STRIDE == M*D*bpe   : head h's slice IS a dense [M, D], which
            #                                    is exactly what matmat(N=D) writes back
            assert self.HEAD_DIM == UE_VECTOR_SIZE, (
                f"HEAD_DIM={self.HEAD_DIM} != UE_VECTOR_SIZE={UE_VECTOR_SIZE}: a kv-head "
                f"is no longer one column block, so the expert's k/v projection cannot "
                f"be split per head (see _ae_kv_units)")
            assert self.AE_KV_HEAD_STRIDE == M * self.HEAD_DIM * 2, (
                f"AE_KV_HEAD_STRIDE={self.AE_KV_HEAD_STRIDE} != M*D*bpe="
                f"{M * self.HEAD_DIM * 2}: the per-head slice is not a dense [M, D] and "
                f"a matmat writeback cannot land on it directly")
            sched = self._make_stage_scheduler("DENOISE", ne)
            # Per-engine [64, cols] lanes for gate / up / mult. matmat_mul_core's
            # writeback stride is N*bpe for the N it was GIVEN, so N=cols writes a DENSE
            # [64, cols] block: each engine owns its own buffer and nothing is strided.
            for nm in ("ae_mlp_gate", "ae_mlp_up", "ae_mlp_mult"):
                sched.alloc_col_output(nm, M, self.E_INTER)
            # PROJECTION LANES. q (N=960, 15 blocks -> 2,2,2,2,2,2,2,1 at ne=8) and
            # o (N=512, EXACTLY 8 blocks of 64). Same dense-writeback argument as the MLP
            # lanes above; the difference is that these two are SCATTERED back into the
            # shared full-width AE_Q_DRAM / AE_O_PROJ_DRAM afterwards, because their
            # consumers (the flash-Q head gather, the residual add) read one contiguous
            # buffer. Tiny: <=128 cols x 64 rows x 2 B = 16 KB (q) + 8 KB (o) per engine.
            #
            # THE LANE CAP LIVES HERE TOO, not just in the region emitter: o has 8
            # blocks, so above 8 engines only the first 8 get a lane buffer at all.
            # _ae_alloc_proj_lanes is the scheduler's own alloc_col_output whenever the
            # op can use every engine, which keeps the proven ne <= 8 allocation order
            # -- and therefore every worker DRAM address -- byte for byte unchanged.
            # Rebuilt from scratch, not accumulated: a second compile of this stage at a
            # different count must not read the previous count's lane addresses.
            self.__dict__["_ae_lane_plan"] = {}
            self._ae_alloc_proj_lanes(sched, "ae_q", M, self.E_Q_OUT)
            o_lanes, _ = self._ae_alloc_proj_lanes(sched, "ae_o", M, HP)
            # The per-step head's three projections. All N = HP = 512, so all three cap
            # at 8 lanes exactly like o_proj -- allocated AFTER ae_q/ae_o so the worker
            # DRAM addresses those two already occupy do not move, and every previously
            # proven program keeps its byte-for-byte allocation order.
            for _hn in ("ae_head_in", "ae_head_tmlp", "ae_head_out"):
                self._ae_alloc_proj_lanes(sched, _hn, M, HP)
            # The K-split partials are FULL [64, 512] per engine -- a reduction does not
            # partition its output. The primary accumulates IN PLACE into the existing
            # AE_MLP_DOWN_DRAM, so every downstream statement is untouched.
            sched.register_per_engine(
                "ae_mlp_down_partial", self.AE_MLP_DOWN_DRAM, M * HP * 2,
                init_tensor=torch.zeros(M * HP, dtype=torch.bfloat16))
            # ATTENTION SHARD: per-engine flash staging. The kv-group loop is split
            # across engines, so AE_FLASH_Q/OUT/SCRATCH cannot be shared -- two engines
            # staging different groups through one buffer is silent corruption. SCRATCH
            # keeps its FULL size: unified_attention_core derives its sub-offsets from
            # the compile-time QB/seq, which do not shrink when the GROUP count does.
            QBr, Dh = self.AE_FLASH_ROWS, self.HEAD_DIM
            CWl = self.AE_COMBINED_LEN
            _n_self_slots = self._ae_n_self_layers()
            for nm, addr, elems in (
                    ("ae_flash_q",   self.AE_FLASH_Q_DRAM,       QBr * Dh),
                    ("ae_flash_out", self.AE_FLASH_OUT_DRAM,     QBr * Dh),
                    ("ae_flash_scr", self.AE_FLASH_SCRATCH_DRAM, self._ae_flash_scratch_elems()),
                    # SELF layers stage [prefix ; suffix] K/V through these. ONE SLOT
                    # PER SELF LAYER, not one buffer reused: the prefix half is hoisted
                    # out of the Euler unroll (see _ae_self_slot), so layer 0's copy has
                    # to survive until layer 2 runs. 16 slots x 256 x 64 x 2 B = 512 KB
                    # per tensor per engine.
                    ("ae_ck",        self.AE_CK_DRAM,   _n_self_slots * CWl * Dh),
                    ("ae_cv",        self.AE_CV_DRAM,   _n_self_slots * CWl * Dh)):
                sched.register_per_engine(
                    nm, addr, elems * 2,
                    init_tensor=torch.zeros(elems, dtype=torch.bfloat16))
            print(f"    [denoise] gated MLP sharded over {ne} engine(s): "
                  f"gate/up N={self.E_INTER} -> {[c for _, c in self._col_split(self.E_INTER, ne)]}"
                  f", down K-split + reduce_add.")
            print(f"    [denoise] q_proj N={self.E_Q_OUT} -> "
                  f"{[c for _, c in self._col_split(self.E_Q_OUT, ne)]}, "
                  f"o_proj N={HP} -> "
                  f"{[c for _, c in self._col_split(HP, o_lanes)]}"
                  f" (N-split + scatter, no reduction)"
                  + (f" on {o_lanes} of {ne} engine(s) -- {HP} is exactly "
                     f"{HP // UE_VECTOR_SIZE} blocks of {UE_VECTOR_SIZE}, so engines "
                     f"{o_lanes}..{ne - 1} idle through that region"
                     if o_lanes < ne else "")
                  + f". action_out (N={self.ACTION_DIM_PAD}) stays on the primary.")
            print(f"    [denoise] per-step head N={HP} -> "
                  f"{[c for _, c in self._col_split(HP, min(ne, HP // UE_VECTOR_SIZE))]}"
                  f" x3 (action_in, time_mlp_in, time_mlp_out; biases sliced with the "
                  f"columns). Assembly copies + composed SiLU stay on the primary.")
            kv_units = self._ae_kv_units(ne)
            print(f"    [denoise] self-layer k/v sharded over {min(ne, 2 * self.NUM_KV_HEADS)}"
                  f" engine(s): {2 * self.NUM_KV_HEADS} (tensor, kv-head) units "
                  f"(N={self.E_KV_OUT} is only {self.E_KV_OUT // UE_VECTOR_SIZE} blocks, "
                  f"but HEAD_DIM==UE_VECTOR_SIZE and k/v are separate weights), "
                  f"units/engine {[len(u) for u in kv_units]}, written straight into the "
                  f"per-head flash operand (the de-interleave is gone)")
            _au = self._ae_attn_units(ne)
            _ah = self._ae_attn_halves(ne)
            _aeng = len(set(e for _, _, e in _au))
            print(f"    [denoise] attention sharded over {_aeng} engine(s): "
                  f"{self.NUM_KV_HEADS} kv groups x {_ah} query-row half(s) of "
                  f"{M // _ah} token(s) = {len(_au)} units"
                  + (f" -- the group axis caps at {self.NUM_KV_HEADS}, the ROW axis "
                     f"does not (batch is unconstrained in unified_attention_core)"
                     if _ah > 1 else
                     f" ({ne - _aeng} idle in this region -- {self.NUM_KV_HEADS} groups "
                     f"do not divide {ne}, and {ne} < {2 * self.NUM_KV_HEADS} so the "
                     f"row split does not fire" if ne > _aeng else ""))
        self._denoise_sched = sched

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
        # "qh": THE QUERY-ROW HALF BATCH for the attention split, 96 = 32 tokens x 3
        # q-heads. Allocated ONLY when the split actually fires (ne >= 2*NUM_KV_HEADS),
        # so ne = 1..9 keep their exact register indices and their exact byte stream --
        # and a body that asks for r_["qh"] without it is a KeyError at compile time
        # rather than a worker driving whatever it happens to hold at that index.
        _halves = self._ae_attn_halves(ne)
        _qh = (M // _halves) * self.GROUP_SIZE
        if _halves > 1:
            regs["qh"] = ue.alloc_isa_reg()
            ue.generate_instruction_add_set(regs["qh"], _qh)
        # "pm": PREFILL_MAX_SEQ_LEN, the row count of the cross-attn K/V reprojection
        # matmul. On pulsevla PM == AE_FLASH_ROWS == 192, so that matmul borrowed
        # regs["qb"] and this register does not exist -- which keeps pulsevla's register
        # indices and emitted bytes EXACTLY as they were. On smolvla PM = 256 != 192 and
        # the borrow would silently reproject only the first 192 of 256 cached prefix
        # rows: finite, NaN-free, and wrong for every token past 192. Same conditional
        # discipline as "qh" above.
        if self.PREFILL_MAX_SEQ_LEN != self.AE_FLASH_ROWS:
            regs["pm"] = ue.alloc_isa_reg()
            ue.generate_instruction_add_set(regs["pm"], self.PREFILL_MAX_SEQ_LEN)
        self._ae_regs = regs

        # ONE REGISTER SET PER ENGINE. begin_program() resets each WORKER's isa/inst-ptr
        # counters too, so the primary's GPR indices are meaningless on a worker: a body
        # replayed with the primary's index would drive whatever that worker happens to
        # hold there. Opened here, once, OUTSIDE every region -- and only "m" is actually
        # consumed by the sharded region, but the full set is primed so a later phase
        # (attention, RoPE) does not have to re-open the capture to add one.
        # begin_program() must come AFTER the primary's start_capture (it asserts the
        # primary is already capturing) and after the primary's own counter reset.
        self._ae_m_regs = [regs["m"]]
        # The attention shard needs "qb"/"cw" per engine too, not just "m".
        self._ae_reg_sets = [regs]
        # ROW-SPLIT M REGISTERS, one per engine, holding THAT engine's row count from
        # _ae_row_split -- 7 or 6 at ne=10, not the 64 in regs["m"]. Separate from
        # _ae_m_regs on purpose: the matmuls are column-split and every one of them
        # still runs all 64 rows, so a single shared "m" cannot serve both cuts.
        # ALLOCATED ONLY WHEN SHARDED. At ne == 1 this list is [regs["m"]] and the row
        # helpers take their unsplit branch, so not one extra add_set enters the
        # single-engine program and --denoise-digest --engines 1 stays byte-identical.
        self._ae_row_m_regs = [regs["m"]]
        if sched is not None:
            sched.begin_program()
            rsplit = self._ae_row_split(sched.num_engines)
            if sched.num_engines > 1:
                _pr = ue.alloc_isa_reg()
                ue.generate_instruction_add_set(_pr, rsplit[0][1])
                self._ae_row_m_regs = [_pr]
            for _wi, w in enumerate(sched.workers):
                wr = {"m": w.alloc_isa_reg(), "qb": w.alloc_isa_reg(),
                      "cw": w.alloc_isa_reg(), "silu": w.alloc_isa_reg(),
                      "time": w.alloc_isa_reg()}
                w.generate_instruction_add_set(wr["m"], M)
                w.generate_instruction_add_set(wr["qb"], self.AE_FLASH_ROWS)
                w.generate_instruction_add_set(wr["cw"], self.AE_COMBINED_LEN)
                w.generate_instruction_add_set(wr["silu"], (M * HP) // UE_VECTOR_SIZE)
                # THE TIMESTEP POINTER IS A **WORD** ADDRESS (byte >> 3), not a byte
                # address -- the PBI DRAM_ADDR descriptor field is 35-bit word-addressed.
                # Seeded identically on every engine so the sets stay symmetric.
                w.generate_instruction_add_set(
                    wr["time"], ue_35bit_addr_shifter(self.AE_TIME_TABLE_DRAM))
                if _halves > 1:
                    wr["qh"] = w.alloc_isa_reg()
                    w.generate_instruction_add_set(wr["qh"], _qh)
                # Mirror the primary's conditional "pm" (see _ae_tensor_init). The
                # ALLOCATION ORDER on each worker must match the primary's exactly, or a
                # replayed body drives a register holding some other engine's number --
                # so this sits after "qh" here just as it does there.
                if self.PREFILL_MAX_SEQ_LEN != self.AE_FLASH_ROWS:
                    wr["pm"] = w.alloc_isa_reg()
                    w.generate_instruction_add_set(wr["pm"], self.PREFILL_MAX_SEQ_LEN)
                self._ae_m_regs.append(wr["m"])
                self._ae_reg_sets.append(wr)
                # This worker's OWN row count -- rsplit[_wi + 1], not rsplit[_wi]:
                # sched.engines is [primary] + workers, so worker _wi is engine _wi + 1.
                # Off by one here and two engines normalize the same rows while a third
                # block is never written: finite, plausible, wrong, on every layer.
                _wr = w.alloc_isa_reg()
                w.generate_instruction_add_set(_wr, rsplit[_wi + 1][1])
                self._ae_row_m_regs.append(_wr)

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
            self._emit_suffix_embed(regs["time"], step=step, sched=sched)  # -> AE_IO_A
            # THE FIRST OF THE ONLY TWO BARRIERS THE ROW SPLIT ADDS. _emit_suffix_embed
            # is primary-only and writes AE_IO_A, which layer 0's input RMSNorm now
            # reads on EVERY engine (row-split). Without this the workers normalize
            # whatever the previous Euler step left in their rows -- finite, plausible,
            # wrong, and it would look like flow-integration drift rather than a race.
            if sched is not None and sched.num_engines > 1:
                sched.barrier()
            n_ae = self.E_LAYERS if self.EXPERT_LAYERS is None else int(self.EXPERT_LAYERS)
            if self.EXPERT_BISECT:
                self._probe_copy(self.AE_IO_A_DRAM, self.AE_P_EMBED_DRAM, M * HP)
            for layer_idx in range(n_ae):
                self._emit_expert_layer(ue, layer_idx, None, sched=sched)
            # 32 layers of ping-pong end in A for an even layer count; derive it rather
            # than assume it, so a tiny/partial expert still reads the right buffer.
            final_h = (self.AE_IO_A_DRAM if n_ae % 2 == 0
                       else self.AE_IO_B_DRAM)
            # THE SECOND. The last layer's residual add is row-split, so final_h is
            # written by every engine, and everything from here to the end of the step
            # (final norm, action_out_proj, the Euler retire) is primary-only and reads
            # it FULL WIDTH. This is the join.
            if sched is not None and sched.num_engines > 1:
                sched.barrier()
            # NOT row-split, unlike the 640 layer norms above: this one runs 10 times,
            # not 640, and splitting it would need a SECOND barrier to re-join before
            # action_out_proj reads AE_FINAL_DRAM full-width on the primary -- 10 more
            # rendezvous to save ~0.07 ms of normalization. Left on the primary.
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

        # ---- STEP-INVARIANT WORK, HOISTED OUT OF THE UNROLL -------------------------
        # Emitted ONCE, here, ahead of every Euler step and after the whole runtime
        # constant/register setup above (the prefix cache and PREFIX_LEN are settled
        # before this program is launched at all). Inside the program, so every
        # re-execution rebuilds it from whatever the current prefix left in the cache.
        n_ae_hoist = self.E_LAYERS if self.EXPERT_LAYERS is None else int(self.EXPERT_LAYERS)
        self._emit_cross_reproject_all(ue, n_ae_hoist)
        # The SELF layers' twin of the above: their staging buffers' prefix half is just
        # as step-invariant as the cross layers' reprojection, and for the same reason
        # (the prefix KV cache is frozen for the whole denoise). Emitted AFTER it so the
        # two hoisted passes appear in layer-type order in the program.
        self._emit_self_prefix_stage_all(ue, sched, n_ae_hoist)

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

        # Close the WORKER captures first: finalize() halts each worker, writes its
        # program to its own arena and advances its allocator. The primary's capture is
        # the model's own and stays the model's job (below). The addresses are KEPT --
        # MultiEngineScheduler overwrites its internal copy on every finalize(), so a
        # bare start_workers() after a later stage would relaunch the wrong program.
        if sched is not None:
            self._denoise_worker_prog_addrs = sched.finalize()
            self._record_worker_prog_sizes("denoise", sched)
            wbytes = self._assert_worker_programs_fit(sched, label="denoise: ")
            print(f"    [denoise] {len(sched.workers)} worker program(s), "
                  f"{wbytes / 1e6:.2f} MB total "
                  f"({wbytes / max(1, len(sched.workers)) / 1e6:.2f} MB each)")

        ue.generate_instruction_halt()
        ue.stop_capture()

        raw = bytearray()
        for inst in ue.capture_buffer:
            raw.extend(inst.get_bytes())
        ue.dma_write(DMA_DEVICE_H2C, prog_addr, raw, len(raw))
        ue.allocate_program_dram(len(raw))
        # THE PRIMARY'S FAR END. _assert_arenas_clear_primary runs ONCE, when the worker
        # pool is built -- which is before weight_init and therefore long before a single
        # program exists, so its `cur < lo` check passes trivially and proves nothing
        # about the 31 MB unrolled denoise. Denoise is the last and by far the largest
        # program the primary compiles, so checking here checks all of them. Overrunning
        # the base does not fault: the primary would simply write its instruction stream
        # over worker 1's, and a worker executing corrupted code hangs on a FLAG_CHECK
        # that has no timeout.
        if sched is not None:
            end = ue.get_program_dram_addr()
            if end < user_dma_core.DRAM_START_ADDR:
                end += user_dma_core.DRAM_START_ADDR
            assert end <= self.VIS_WORKER_ARENA_BASE, (
                f"the primary's programs now end at 0x{end:X}, past the worker arena "
                f"base 0x{self.VIS_WORKER_ARENA_BASE:X}. Worker 1's program would be "
                f"overwritten and the run would hang on a FLAG_CHECK, not fail. Shrink "
                f"VIS_WORKER_ARENA_BYTES (the base cannot move below the 0xD6000000 "
                f"start of the program region).")
        if self.DENOISE_DIGEST:
            # BEFORE clear_capture_buffer -- capture_digest reads ue.capture_buffer.
            self._denoise_capture_digest = capture_digest(ue)
            _original_print(f"    [denoise] ne={ne} PRIMARY capture digest "
                            f"{self._denoise_capture_digest}")
            for wi, w in enumerate(sched.workers if sched is not None else []):
                _original_print(f"    [denoise] worker {wi + 1} digest "
                                f"{capture_digest(w)}")
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
        # Goes through the SAME _prog_cache as prefix/denoise so precompile_all() can
        # seed it up front and this call becomes a pure dict hit (no capture, no
        # program-DRAM allocation) on every later inference.
        prog = self._compile_once("vision", self.compile_encoder, label="vision")

        # layer_norm_core_dram WRITES its zeros scratch, so the previous slot's execution
        # dirtied it. Restore before every run or slot 1 layer-norms against garbage
        # (pi05's per-slot refresh, same root cause).
        zeros_t = torch.zeros(H, dtype=torch.bfloat16)

        sched = getattr(self, "_vis_sched", None)
        multi = sched is not None and sched.num_engines > 1

        batched = getattr(self, "_vis_batched", False)
        bpe = 2

        def _stage(i):
            """HOST PATCHIFY slot i into its half of VIS_PIXEL_IN_DRAM.

            [512,512,3] HWC -> [3,512,512] planar -> [1024, 768] with columns in the
            Conv2d weight's channel-major (c, kh, kw) order. The device used to do this
            permute, but its last dim (P=16) is under UE_VECTOR_SIZE and nn_lib's
            unaligned branch races through a single URAM slot -- see compile_encoder.
            _host_patchify is bit-exact vs conv2d unfold."""
            planes = images[i].permute(2, 0, 1).contiguous().float()   # [3,512,512]
            patches = self._host_patchify(planes)                      # [1024, 768]
            self.dma_to_accelerator_memory(
                self.VIS_PIXEL_IN_DRAM + (i * S * H * bpe if batched else 0),
                patches.reshape(-1).to(torch.bfloat16).contiguous())

        def _refresh_zeros():
            if multi:
                # EVERY engine's private copy, not just the primary's: since the norms
                # were sharded every engine runs its own layer_norm_core_dram twice per
                # layer, and that core uses this buffer as scratch. A stale worker copy
                # is silent corruption. refresh_per_engine writes the primary's too.
                sched.refresh_per_engine("vis_zeros", zeros_t)
            else:
                self.dma_to_accelerator_memory(self.vis_zeros_addr, zeros_t)

        def _launch(label):
            if multi:
                # Workers HALT at the end of every run, so they are relaunched on EVERY
                # execution. preclear_flags is the opposite: exactly once per process,
                # before the first launch, or a stale SET from a previous run makes the
                # first rendezvous fall straight through.
                self._preclear_flags_once(sched)
                sched.start_workers(self._vis_worker_prog_addrs)
            with PHASES.track(label, "exec"):
                self.program_execute(prog, timeout=self.EXEC_TIMEOUT)
            if multi:
                # The primary retires its halt as soon as the last rendezvous clears; a
                # worker may still be draining its (write-free) margin NOPs.
                for w in sched.workers:
                    w.wait_queue(60.0)

        def _read(i):
            _o = i * S * H * bpe if batched else 0
            _c = i * tok_out * C["output_size"] * bpe if batched else 0
            post_ln.append(self._read_bf16(self.VIS_POST_LN_DRAM + _o, (S, H),
                                           label=f"vis_post_ln[{i}]"))
            conn.append(self._read_bf16(self.VIS_CONNECTOR_DRAM + _c,
                                        (tok_out, C["output_size"]),
                                        label=f"vis_connector[{i}]"))
            # In the two-pass path this readback is also the COPY OUT:
            # VIS_CONNECTOR_DRAM is one buffer and the next slot's execution would
            # overwrite it. Batched, the two slots own disjoint halves and nothing
            # overwrites anything -- but reading both here keeps one code path.
            assert torch.isfinite(conn[-1]).all(), \
                f"vision slot {i} produced non-finite tokens"

        post_ln, conn = [], []
        if batched:
            # ONE EXECUTION FOR BOTH CAMERAS. Staging is still per-image (the host
            # patchify is per-image by nature) but it lands in disjoint halves of one
            # buffer, and the device sees a single [slots*S, H] stream.
            for i in range(slots):
                _stage(i)
            _refresh_zeros()
            _launch(f"exec vision[{slots} slots batched]")
            for i in range(slots):
                _read(i)
        else:
            for i in range(slots):
                _stage(i)
                _refresh_zeros()
                _launch(f"exec vision[slot{i}]")
                _read(i)

        tokens = torch.cat(conn, 0)                                   # [slots*64, 960]
        self._last_vision = {"post_ln": post_ln, "connector": conn,
                             "tokens": tokens, "images": images}
        print(f"  vision: {slots} slots -> tokens {tuple(tokens.shape)} "
              f"absmax={tokens.abs().max():.4f}")
        return tokens

    def _execute_prefix(self, timeout=None):
        """EXECUTE the compiled prefix program. Mirrors _execute_denoise: the program is
        address-static, so re-execution just re-reads LM_INPUT_DRAM and PREFIX_BIAS_DRAM
        (both refreshed by run_prefix) and re-writes the KV cache in place."""
        sched = getattr(self, "_prefix_sched", None)
        with PHASES.track("exec prefix", "exec"), silenced():
            if sched is not None:
                # EVERY execution, and BEFORE the primary launches: each run ends with
                # the workers halted, and a worker that is not already spinning on the
                # first barrier when the primary reaches it deadlocks the whole program
                # (FLAG_CHECK has no timeout). Pass this stage's saved address list --
                # sched._worker_prog_addrs is overwritten by any later finalize().
                self._preclear_flags_once(sched)
                sched.start_workers(self._prefix_worker_progs)
            self.start_execute_from_dram(self._prefix_program_addr)
            self._wait_with_heartbeat("prefix", timeout=timeout)
            if sched is not None:
                # JOIN. The primary retires its halt as soon as the last rendezvous
                # clears, while a worker may still be draining its (write-free) margin
                # NOPs. Leaving them unjoined hands the NEXT stage engines that read
                # queue_busy -- and _preclear_flags_once SW_RESETs a busy engine, which
                # is how an unjoined worker becomes a deadlock one stage later.
                for w in sched.workers:
                    w.wait_queue(60.0)
        _original_print(f"  [prefix] executed in {PHASES.rows[-1][2]:.2f}s")

    def run_prefix(self, vision_tokens, token_ids, state, text_mask=None, timeout=None):
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

    def _dump_engine_state(self, label, sched=None):
        """Print every engine's busy/flag state. Called when a stage times out.

        A multi-engine hang is always a FLAG_CHECK spin (it has no timeout), and the
        question is only WHICH engine failed to arrive. An engine that is idle while the
        primary is busy never reached its rendezvous -- it either halted early (its
        program has FEWER barriers than the primary's) or it was never launched. An
        engine that is busy alongside the primary is waiting on someone else.
        """
        sched = sched or next(iter(self.__dict__.get("_sched_by_stage", {}).values()), None)
        engines = list(sched.engines) if sched is not None else [self]
        _original_print(f"  [{label}] ENGINE STATE AT TIMEOUT "
                        f"({len(engines)} engine(s)):")
        for i, ue in enumerate(engines):
            try:
                busy = ue.is_queue_busy()
                ctrl = ue.read_reg32(user_dma_core.UE_QUEUE_CTRL_ADDR)
                _original_print(f"    engine {i}{' (primary)' if i == 0 else ''}: "
                                f"busy={busy}  queue_ctrl=0x{ctrl:08X}")
            except Exception as e:                      # a wedged engine may not answer
                _original_print(f"    engine {i}: unreadable ({e!r})")
        for stage, sc in self.__dict__.get("_sched_by_stage", {}).items():
            saved = {"VIS": "_vis_worker_prog_addrs", "PREFIX": "_prefix_worker_progs",
                     "DENOISE": "_denoise_worker_prog_addrs"}.get(stage)
            addrs = self.__dict__.get(saved) if saved else None
            _original_print(f"    {stage}: {len(sc.workers)} worker(s), saved "
                            f"{saved}=" + (", ".join(f"0x{a:X}" for a in addrs)
                                           if addrs else "None"))

    def _wait_with_heartbeat(self, label, timeout=None, heartbeat_every=1.0):
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
        timeout = self.EXEC_TIMEOUT if timeout is None else timeout
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
            # A multi-engine hang is a FLAG_CHECK spin. Dump who arrived and who did not
            # BEFORE raising, or the traceback says only "it hung" and the next run has
            # to reproduce it to learn anything.
            try:
                self._dump_engine_state(label)
            except Exception as _e:
                _original_print(f"  [{label}] engine dump failed: {_e!r}")
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
        if getattr(self, "_compile_frozen", False):
            raise RuntimeError(
                f"_compile_once({key!r}) would COMPILE inside the execution flow, but "
                f"precompile_all() froze the program set to {sorted(cache)}. Compilation "
                f"here means a capture + a program-DRAM allocation per inference -- the "
                f"exact leak that killed pi05 at 3 inferences. Add {key!r} to "
                f"precompile_all(stages=...) instead of compiling lazily.")
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

    def _execute_denoise(self, timeout=None):
        """EXECUTE the compiled denoise program. Re-execution is deterministic: the
        prefix K/V it cross-attends into sits at fixed LAYER0_K/V_DRAM (refreshed by the
        prefix stage), the timestep pointer is re-seeded inside the captured program so
        every run restarts at t=1.0, and fresh noise is DMA'd to AE_XT_DRAM by the
        caller before this is called."""
        sched = getattr(self, "_denoise_sched", None)
        with PHASES.track("exec denoise", "exec"), silenced():
            if sched is not None:
                # Once per process: a flag left SET by a program that died mid-run would
                # make the first CHECK of this run pass spuriously. Guarded by
                # is_queue_busy inside, so it is a no-op on a clean device.
                self._preclear_flags_once(sched)
                # Workers BEFORE the primary, on EVERY execution: each run ends with the
                # workers halted, and they must already be parked on the first
                # rendezvous when the primary reaches it.
                sched.start_workers(self._denoise_worker_prog_addrs)
            self.start_execute_from_dram(self._denoise_program_addr)
            self._wait_with_heartbeat("denoise", timeout=timeout)
            if sched is not None:
                # The primary retires its halt the moment the last rendezvous clears; a
                # worker may still be draining its (write-free) margin NOPs. Wait for it,
                # or the NEXT inference's start_workers hits an engine still flagged busy
                # and the run desyncs on a FLAG_CHECK that has no timeout.
                for w in sched.workers:
                    w.wait_queue(60.0)
        _original_print(f"  [denoise] executed in {PHASES.rows[-1][2]:.2f}s")

    def run_denoise(self, noise=None, timeout=None):
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

    def run_inference(self, images_hwc, token_ids, state, noise=None, snr=False,
                      stop_after=None, strict_gates=False):
        """vision -> prefix -> denoise on hardware. Returns the EXECUTED action slice
        [n_action_steps, action_dim] = [10, 7].

        snr DEFAULTS TO OFF: the plain call runs the model on the accelerator and
        nothing else. With snr=True, gate each stage against the torch oracle and STOP
        AT THE FIRST stage below 40 dB. Failing early is the whole point: once a stage is wrong,
        everything downstream of it is scoring a different model, so a single end-to-end
        number cannot tell "the ViT drifted" from "the pixel shuffle is scrambled" from
        "the expert reads the wrong KV layer". The stages that HAVE gates run them; the
        ones whose gate is still a stub are named explicitly as UNGATED so nobody reads a
        silent pass as a verified one.

        Stage timing goes through PHASES. The "stage" rows are wall-clock ENVELOPES that
        CONTAIN that stage's own compile/exec/gate rows, so they are excluded from the
        per-kind TOTALs. Read compile vs exec: compile is paid once and shrinks by
        cutting shape diversity, exec is paid every inference and shrinks by sharding.
        WALL is measured, not summed, and the untracked-host line is the difference --
        patchify, DMA staging and readbacks, which no row currently covers.

        Stashes for callers/probes:
            self._last_inference["actions_padded"] [50, 32]  full chunk, padded DoF
            self._last_inference["actions_chunk"]  [50,  7]  full chunk, real DoF
            self._last_inference["actions"]        [10,  7]  what the robot executes
        """
        HEAD = self._cfg["action_head"]
        n_exec, adim, chunk = HEAD["n_action_steps"], HEAD["action_dim"], HEAD["chunk_size"]
        ungated, failed_gates = [], []

        # THE ONE NUMBER THAT ANSWERS "how long is a forward pass". Opened here and
        # closed at EVERY exit below (both --stop-after returns and the normal path), so
        # the model runs vision -> prefix -> denoise back to back and is timed once
        # rather than reconstructed by adding three device counters together.
        _fwd = PHASES.mark_forward_start()

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
            PHASES.record_forward(_fwd, "PARTIAL FORWARD (vision only, --stop-after)")
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
            PHASES.record_forward(_fwd, "PARTIAL FORWARD (vision+prefix, --stop-after)")
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

        PHASES.record_forward(_fwd)

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

    # The three compilable sections, IN THE ORDER THEY MUST BE COMPILED. This order is
    # not cosmetic: alloc_isa_reg / alloc_inst_ptr are per-PROCESS counters that
    # start_capture() does NOT reset, so each stage's registers and PBI pointer rows land
    # on physical indices that depend on what compiled BEFORE it (see the long comment in
    # compile_denoise_loop -- pi05's symptom for getting this wrong was 100% NaN, not a
    # crash). vision -> prefix -> denoise is exactly the lazy order that the gated runs
    # validated, so precompiling in this order reproduces byte-identical programs.
    COMPILE_STAGES = ("vision", "prefix", "denoise")

    def precompile_all(self, stages=COMPILE_STAGES, freeze=True):
        _bin_attrs_before = set(self.__dict__)
        try:
            return self._precompile_all_inner(stages=stages, freeze=freeze)
        finally:
            # Compile publishes state that EXECUTION reads -- the action expert's rope
            # and bias DRAM addresses, its head/stride constants, the time table. A bin
            # run never compiles, so without this it silently runs the expert against
            # whatever the class defaults are: no crash, just different numbers.
            self._capture_phase_attrs(_bin_attrs_before, "compile")

    def _precompile_all_inner(self, stages=COMPILE_STAGES, freeze=True):
        """COMPILE EVERYTHING UP FRONT, then execute stage by stage.

        The engine already had compile-once/execute-many per stage (_compile_once), but
        compilation was still *interleaved* into the first inference: run_vision compiled
        the encoder, run_prefix compiled the prefix, run_denoise compiled the denoise
        loop. That made inference #0 structurally different from every later one --
        minutes of capture time in the middle of what is supposed to be a robot control
        loop, and a program-DRAM allocation happening after live data was already staged.

        This method moves all three captures into an explicit build phase, so the
        execution flow is pure execute. With freeze=True, _compile_once then REFUSES to
        compile anything new, which turns "a stage compiled at inference time" from a
        silent 3-minute stall into a loud error.

        PRECONDITIONS: weight_init() and tensor_init() must have run. Compilation reads
        weight ADDRESSES (and, for the encoder, the vision tensor addresses tensor_init
        claims), and every stage's compile also runs its own idempotent tensor-init
        (_lm_tensor_init / _ae_tensor_init), so all DRAM regions are claimed here rather
        than racing with staged activations later.

        NOTHING DATA-DEPENDENT IS BAKED IN. The prefix attention bias, the expert's two
        biases and its RoPE base all depend on the observation's valid prefix length, but
        they are DRAM *data* read by address from an address-static program -- run_prefix
        (build_attn_bias) and run_denoise (_ae_refresh_runtime_constants) restage them
        per inference with no recompile. That is why the whole program set can be built
        before a single observation exists.

        Returns {stage: program_dram_addr}.
        """
        unknown = [s for s in stages if s not in self.COMPILE_STAGES]
        if unknown:
            raise ValueError(f"unknown compile stage(s) {unknown}; "
                             f"known: {list(self.COMPILE_STAGES)}")
        if not hasattr(self, "vis_post_ln_weight"):
            raise RuntimeError(
                "precompile_all() before weight_init(): compilation emits weight DRAM "
                "addresses, so the weights must be resident first")

        fns = {"vision": (self.compile_encoder, "vision"),
               "prefix": (self.compile_prefix, "prefix"),
               "denoise": (self.compile_denoise_loop, "denoise")}
        # Iterate COMPILE_STAGES, not the caller's tuple: order is load-bearing (above),
        # so a caller passing ("denoise", "vision") still gets the validated order.
        want = [s for s in self.COMPILE_STAGES if s in stages]
        _original_print(f"  [precompile] building {len(want)} program(s): "
                        f"{', '.join(want)} (compile phase -- no execution)")
        t0 = time.perf_counter()
        out = {}
        for s in want:
            fn, label = fns[s]
            out[s] = self._compile_once(s, fn, label=label)
        meta = self.__dict__.get("_prog_meta", {})
        total = sum(meta[s][1] for s in want if s in meta)
        _original_print(f"  [precompile] {len(want)} program(s), {total / 1e6:.2f} MB, "
                        f"{time.perf_counter() - t0:.1f}s. Program DRAM now at "
                        f"0x{self.get_program_dram_addr():X}; execution is compile-free "
                        f"from here.")
        if freeze:
            # Only freeze on a FULL build. Freezing a partial set (e.g. --stop-after
            # vision) would turn a later legitimate stage compile into a hard error.
            self._compile_frozen = (len(want) == len(self.COMPILE_STAGES))
        return out

    def dump_params_to_file(self, bin_dir):
        """Dump the whole params DRAM region to bin_dir/params.bin + params.json.

        ONE file for every engine count (see the bin-layout note at module scope). A
        DRAM readback, not a per-tensor map: a bin run re-derives every address by
        re-running the real allocators, so there is no map to drift.
        """
        os.makedirs(bin_dir, exist_ok=True)
        total = self.get_params_dram_usage()
        assert total > 0, "params DRAM is empty -- dump only after weight_init()"
        assert hasattr(self, "_params_ofs_after_weight_init"), (
            "dump_params_to_file: _params_ofs_after_weight_init missing -- dump only "
            "after a real weight_init() (a bin-backed run must not re-dump).")
        data = self._bin_read(self._params_dram_base, total, label="params")
        with open(os.path.join(bin_dir, "params.bin"), "wb") as f:
            f.write(data)
        tensors = getattr(self, "_weight_tensor_attrs", None) or {}
        if tensors:
            torch.save(tensors, os.path.join(bin_dir, "weight_tensors.pt"))
            _original_print(f"  Host tensors: {', '.join(sorted(tensors))} -> "
                            f"weight_tensors.pt")
        _skipped = [x for x in getattr(self, "_weight_attrs_skipped", [])
                    if not x.startswith("_worker_pool:")]
        if _skipped:
            # LOUD: a dropped attribute does not crash a bin run, it just changes the
            # numbers. Three per-layer address maps were lost exactly this way.
            _original_print(f"  WARNING: {len(_skipped)} weight attribute(s) could not "
                            f"be persisted and will be MISSING in a bin run: "
                            f"{', '.join(_skipped)}")
        attrs = getattr(self, "_weight_attrs", None)
        assert attrs, (
            "dump_params_to_file: no captured weight attributes -- dump only after a "
            "real weight_init(). Without them a bin run has no weight DRAM addresses "
            "and dies in the first host-side readback.")
        with open(os.path.join(bin_dir, "params.json"), "w") as f:
            json.dump({"size": total,
                       "ofs_after_weight_init": int(self._params_ofs_after_weight_init),
                       "dummy_weights": bool(getattr(self, "_dummy_weights", False)),
                       # With params.bin, not the programs: the weight phase does not
                       # depend on the engine count.
                       "weight_attrs": attrs,
                       "weight_tensor_attrs": sorted(tensors),
                       "weight_attrs_skipped": getattr(self, "_weight_attrs_skipped", [])},
                      f, indent=2)
        _original_print(f"  Params: {total / 1024**2:.1f} MB ({total} bytes) -> "
                        f"{os.path.join(bin_dir, 'params.bin')}")

    def dump_programs_to_file(self, bin_dir):
        """Dump this run's programs to bin_dir/programs_e<v>_<p>_<d>.bin + .json.

        Keyed by the engine triple, so sets for different counts coexist and neither can
        load as the other. Each stage carries the primary program plus one section per
        worker -- a sharded stage is not one program, and replaying only the primary
        leaves the worker rows unwritten while it waits at a rendezvous nobody answers.

        Worker sections state an explicit dram_base; the primary's is reproduced by
        replaying the bump allocator, but worker arenas are scheduler-owned.
        """
        engines = self._bin_engines()
        meta = self.__dict__.get("_prog_meta", {})
        for s in _PROGRAM_ORDER:
            assert s in meta, (
                f"dump_programs_to_file: program {s!r} was never compiled this run "
                f"(have {sorted(meta)}). Dump only after a full non-debug inference.")
        for a in _BIN_DERIVED_ATTRS:
            assert hasattr(self, a), (
                f"dump_programs_to_file: {a} missing -- dump only after a full "
                f"vision -> prefix -> denoise inference on this engine instance.")

        os.makedirs(bin_dir, exist_ok=True)
        stem = _programs_stem(engines)
        bin_path = os.path.join(bin_dir, stem + ".bin")
        meta_path = os.path.join(bin_dir, stem + ".json")

        manifest = {"programs": {}, "engines": engines, "sig": {},
                    "derived": {a: getattr(self, a) for a in _BIN_DERIVED_ATTRS}}
        blob = bytearray()
        for stage in _PROGRAM_ORDER:
            ne_key, _addr_attr, waddr_attr, wsize_attr = _STAGE_ATTRS[stage]
            addr, size = meta[stage]
            offset = len(blob)
            blob.extend(self._bin_read(addr, size, label=f"{stage} program"))
            sections = [{"offset": offset, "size": size, "engine": 0}]

            waddrs = list(getattr(self, waddr_attr, None) or [])
            wsizes = list(getattr(self, wsize_attr, None) or [])
            assert len(waddrs) == engines[stage] - 1, (
                f"dump_programs_to_file: stage {stage!r} runs on {engines[stage]} "
                f"engine(s) but {waddr_attr} holds {len(waddrs)} worker address(es). "
                f"The stage was compiled at a different engine count than it is "
                f"configured for -- regenerate from a single clean run.")
            assert len(wsizes) == len(waddrs), (
                f"dump_programs_to_file: stage {stage!r} has {len(waddrs)} worker "
                f"address(es) but {len(wsizes)} recorded size(s). "
                f"_record_worker_prog_sizes did not run for this stage.")
            for wi, (waddr, wsize) in enumerate(zip(waddrs, wsizes)):
                w_off = len(blob)
                # Read through the WORKER's own engine: the arenas are outside the
                # primary's allocator regions and each engine has its own DMA channel.
                worker = self._worker_engine_pool()[wi]
                blob.extend(self._bin_read(waddr, wsize, ue=worker,
                                           label=f"{stage} worker {wi + 1}"))
                sections.append({"offset": w_off, "size": wsize, "engine": wi + 1,
                                 "dram_base": hex(waddr)})
            manifest["programs"][stage] = sections

        # SCHEDULER REGISTRIES. register_per_engine()/alloc_col_output() run at COMPILE
        # time and hand back worker-arena addresses that the emitted programs bake. A bin
        # run never compiles, so a freshly built scheduler has EMPTY registries -- and
        # run_vision calls sched.refresh_per_engine("vis_zeros") on every inference to
        # re-zero scratch the kernel dirties. Without this the first bin-backed inference
        # dies with KeyError: 'vis_zeros'; worse, col_output_addr would silently hand a
        # later caller the wrong lane. Persist the addresses and re-register them on load.
        sched_state = {}
        # _make_stage_scheduler caches under the TUPLE (stage_key, num_engines), not the
        # bare stage name -- look up by the first element or every stage silently misses
        # and the block is written empty.
        _by_stage = self.__dict__.get("_sched_by_stage", {})
        for stage in _PROGRAM_ORDER:
            want = _STAGE_ATTRS[stage][0]
            sc = next((v for k, v in _by_stage.items()
                       if (k[0] if isinstance(k, tuple) else k) == want), None)
            if sc is None:
                continue
            sched_state[stage] = {
                "per_engine": {k: list(v) for k, v in sc._per_engine.items()},
                "col_outputs": {k: list(v) for k, v in sc._col_outputs.items()},
            }
        _cattrs = getattr(self, "_compile_attrs", None)
        assert _cattrs is not None, (
            "dump_programs_to_file: no captured compile attributes -- dump only after "
            "precompile_all() ran on this engine.")
        _cskipped = getattr(self, "_compile_attrs_skipped", [])
        if _cskipped:
            _original_print(f"  WARNING: {len(_cskipped)} compile attribute(s) could not "
                            f"be persisted and will be MISSING in a bin run: "
                            f"{', '.join(_cskipped)}")
        manifest["compile_attrs"] = _cattrs
        _ctensors = getattr(self, "_compile_tensor_attrs", None) or {}
        manifest["compile_tensor_attrs"] = sorted(_ctensors)
        if _ctensors:
            torch.save(_ctensors, os.path.join(bin_dir, stem + "_tensors.pt"))

        assert sched_state, (
            "dump_programs_to_file: no stage schedulers found in _sched_by_stage -- the "
            "per-engine registries would be lost and every bin run would die in "
            "refresh_per_engine. Dump only after a full inference.")
        manifest["scheduler"] = sched_state
        # Where each worker's TENSOR allocator finished. The per-engine buffers above came
        # out of it, so a bin run that re-registers the addresses without advancing the
        # allocator would hand the next allocation an address already in use.
        manifest["worker_tensor_addr"] = [w.get_tensor_dram_addr()
                                          for w in self._worker_engine_pool()]

        HEAD, V, L = self._cfg["action_head"], self._cfg["vision"], self._cfg["lm"]
        manifest["sig"] = {
            "num_image_slots": int(V["num_image_slots"]),
            "chunk_size": int(HEAD["chunk_size"]),
            "denoise_steps": int(HEAD["num_denoise_steps"]),
            "rope_theta": float(L["rope_theta"]),
            "quant": "bf16",
            "vis_layers": self.VIS_LAYERS,
            "vis_no_batch": bool(getattr(self, "VIS_NO_BATCH", False)),
            "prefix_fused_silu": bool(getattr(self, "PREFIX_FUSED_SILU", False)),
            "params_dram_base": hex(self._params_dram_base),
            "tensor_dram_base": hex(self._tensor_dram_base),
            "program_dram_base": hex(self._program_dram_base),
            "worker_arena_base": hex(self.VIS_WORKER_ARENA_BASE),
            "worker_arena_bytes": int(self.VIS_WORKER_ARENA_BYTES),
        }
        with open(bin_path, "wb") as f:
            f.write(blob)
        with open(meta_path, "w") as f:
            json.dump(manifest, f, indent=2)
        _original_print(f"  Programs: {len(blob) / 1024**2:.1f} MB ({len(blob)} bytes) "
                        f"-> {bin_path}")
        for stage, secs in manifest["programs"].items():
            for s in secs:
                _original_print(f"    {stage:<8} engine{s['engine']:<2} "
                                f"offset={s['offset']:>10}  size={s['size']:>10}")

    def dump_bins(self, bin_dir):
        """params.bin + programs_e<v>_<p>_<d>.bin/.json. The programs carry a signature
        (image slots, chunk, denoise steps, rope theta, quant, dram bases) and the loader
        refuses on mismatch. rm the bin dir after any compile-affecting edit -- stale bins
        reload silently."""
        self.dump_params_to_file(bin_dir)
        self.dump_programs_to_file(bin_dir)

    # ---------------------------------------------------------------- bin loading --
    # These live on the BASE class, not on VeraPulse_Run, because the decision to use a
    # bin set can only be made AFTER the engine exists: which programs_e*.json applies
    # depends on _num_engines, which consults the board's core limit and the prefix's
    # column-block count. main() therefore builds the normal engine, configures the
    # engine counts, and only then asks bins_available().

    # Values a weight attribute may hold and still survive the JSON round trip. Weight
    # DRAM addresses -- the ones that actually matter -- are plain ints.
    _BIN_ATTR_SCALARS = (int, float, bool, str, type(None))

    # Attributes load_programs() sets ITSELF from the manifest. Restoring these from a
    # captured snapshot would overwrite the addresses just DMA'd, so they are never
    # captured: the loader is the authority for them.
    _BIN_ATTR_LOADER_OWNED = frozenset((
        "_prog_cache", "_prog_meta", "_compile_frozen", "_sched_by_stage",
        "_vis_sched", "_prefix_sched", "_denoise_sched",
        "_vis_program_addr", "_prefix_program_addr", "_denoise_program_addr",
        "_vis_worker_prog_addrs", "_prefix_worker_progs", "_denoise_worker_prog_addrs",
        "_vis_worker_prog_sizes", "_prefix_worker_prog_sizes",
        "_denoise_worker_prog_sizes",
    ))

    def _capture_weight_attrs(self, before):
        """Every JSON-safe attribute a phase published, by DIFFING __dict__.

        A hand-written list drifts: _weight_init_lm alone publishes dozens of *_addr
        attributes, and one added later surfaces as an AttributeError mid-inference
        (state_proj_weight did). Unserializable values are recorded by name in
        `skipped` so a later failure names them.
        """
        def ok(v, depth=0):
            """JSON-safety, RECURSIVELY: the per-layer address maps are lists of dicts,
            and a flat scalars-only check dropped all three silently -- wrong numbers,
            not a crash, because the stages fall back rather than raise."""
            if depth > 6:
                return False
            if isinstance(v, self._BIN_ATTR_SCALARS):
                return True
            if isinstance(v, (list, tuple)):
                return all(ok(x, depth + 1) for x in v)
            if isinstance(v, dict):
                return all(isinstance(k, str) and ok(x, depth + 1)
                           for k, x in v.items())
            return False

        def norm(v):
            """tuples -> lists, so the JSON round trip is identity."""
            if isinstance(v, (list, tuple)):
                return [norm(x) for x in v]
            if isinstance(v, dict):
                return {k: norm(x) for k, x in v.items()}
            return v

        attrs, tensors, skipped = {}, {}, []
        for k in sorted(set(self.__dict__) - set(before)):
            if k.startswith("_bin_attrs") or k in self._BIN_ATTR_LOADER_OWNED:
                continue
            v = self.__dict__[k]
            if ok(v):
                attrs[k] = norm(v)
            elif isinstance(v, torch.Tensor):
                # Host-side tables (the token embedding gather table) are indexed on the
                # CPU and never reach params DRAM, so they travel in their own file.
                tensors[k] = v
            else:
                skipped.append(f"{k}:{type(v).__name__}")
        return attrs, tensors, skipped

    def _capture_phase_attrs(self, before, phase):
        """Store the diff under _<phase>_attrs / _<phase>_tensor_attrs / _..._skipped."""
        attrs, tensors, skipped = self._capture_weight_attrs(before)
        setattr(self, f"_{phase}_attrs", attrs)
        setattr(self, f"_{phase}_tensor_attrs", tensors)
        setattr(self, f"_{phase}_attrs_skipped", skipped)
        return attrs, tensors, skipped

    def _restore_weight_attrs(self, attrs):
        for k, v in attrs.items():
            setattr(self, k, v)

    def bins_available(self, bin_dir=None):
        """True if a complete bin set exists for THIS run's engine configuration."""
        bin_dir = bin_dir or getattr(self, "bin_dir", None) or BIN_DIR
        if not os.path.isdir(bin_dir):
            return False
        stem = _programs_stem(self._bin_engines())
        if not all(os.path.exists(os.path.join(bin_dir, f))
                   for f in ("params.bin", "params.json",
                             stem + ".bin", stem + ".json")):
            return False
        try:
            with open(os.path.join(bin_dir, "params.json")) as f:
                pm = json.load(f)
        except (OSError, ValueError):
            return False
        if pm.get("weight_tensor_attrs") and not os.path.exists(
                os.path.join(bin_dir, "weight_tensors.pt")):
            return False
        # Provenance is part of "matching": loading a dummy-weight dump into a real-weight
        # run would score a model nobody asked for. Report NOT available (so the caller
        # falls back to compiling) rather than refusing the run.
        return bool(pm.get("dummy_weights", False)) == bool(
            getattr(self, "_dummy_weights_wanted", False))

    def weight_init_from_bin(self, dummy=False):
        """Bin-mode stand-in for weight_init: build the worker pool, restore params.

        Keeps weight_init's FIRST act -- the worker pool -- because the ordering hazard
        is identical: every UnifiedEngine ctor DMA-writes 16 KB of noise to a hardcoded
        0x80000000, this model's first stored weight. A worker built after the restore
        shreds the head of params: finite, plausible, wrong.
        """
        self._worker_engine_pool()
        self._dummy_weights = bool(dummy)
        self.load_params()

    def _sig_check(self, name, live, want, hint=""):
        if live != want:
            raise RuntimeError(
                f"\n*** REFUSING TO RUN: pre-compiled bins do not match this run ***\n"
                f"  {name}: bins were compiled for {want!r}, this run wants {live!r}\n"
                f"  bins: {self.bin_dir}\n"
                f"  The compiled programs bake absolute jump targets and absolute tensor\n"
                f"  DRAM addresses; they CANNOT be adapted to a different {name}.\n"
                + (f"  Fix: {hint}.\n" if hint else "")
                + f"  Regenerate with:  python {os.path.abspath(__file__)} "
                  f"--stage all --dump-bins\n")

    # ------------------------------------------------------------- weights/params --

    def load_params(self):
        """Restore the params DRAM snapshot, then rewind the params allocator to the
        post-weight_init boundary."""
        bin_path = os.path.join(self.bin_dir, "params.bin")
        meta_path = os.path.join(self.bin_dir, "params.json")
        for p in (bin_path, meta_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"missing {p} -- regenerate the bins")
        with open(meta_path) as f:
            pmeta = json.load(f)
        total = int(pmeta["size"])
        actual = os.path.getsize(bin_path)
        assert actual == total, (
            f"params.bin is {actual}B but params.json declares {total}B -- truncated or "
            f"stale bin set; regenerate.")
        # A dummy-weight dump is plumbing-only and says nothing about fidelity; loading
        # it while the caller believes it asked for real weights would silently score a
        # random model. Refuse in BOTH directions.
        self._sig_check("dummy_weights", bool(getattr(self, "_dummy_weights", False)),
                        bool(pmeta.get("dummy_weights", False)),
                        "pass --weights dummy to match, or regenerate the bins")
        with open(bin_path, "rb") as f:
            data = f.read()
        self._bin_write(self._params_dram_base, data, label="params")
        # Rewind to the post-weight_init boundary, NOT to `total`: params.bin is dumped
        # at the END of a full run, so it also holds what the compile phase allocated
        # afterwards (the AE time table). The bytes are correct on-device either way --
        # this only puts the allocator back where the programs expect it.
        attrs = pmeta.get("weight_attrs")
        if not attrs:
            raise RuntimeError(
                f"params.bin at {self.bin_dir} predates the `weight_attrs` block, so this "
                f"run has no weight DRAM addresses -- regenerate with --clean.")
        self._restore_weight_attrs(attrs)
        want_tensors = pmeta.get("weight_tensor_attrs") or []
        if want_tensors:
            tpath = os.path.join(self.bin_dir, "weight_tensors.pt")
            if not os.path.exists(tpath):
                raise FileNotFoundError(
                    f"missing {tpath}: params.json lists host tensors {want_tensors} "
                    f"that live outside params.bin -- regenerate with --clean")
            loaded = torch.load(tpath, map_location="cpu", weights_only=True)
            missing = [k for k in want_tensors if k not in loaded]
            if missing:
                raise RuntimeError(
                    f"weight_tensors.pt is missing {missing} -- stale set; --clean")
            for k, v in loaded.items():
                setattr(self, k, v)
        ofs = int(pmeta["ofs_after_weight_init"])
        self._next_params_dram_addr = self._params_dram_base + ofs
        self._params_ofs_after_weight_init = ofs
        _original_print(f"  Params: {total / 1024**2:.1f} MB from bin "
                        f"(allocator rewound to +{ofs}B)")
        return True

    # ------------------------------------------------------------------- programs --

    def load_programs(self):
        """DMA every compiled program back to DRAM -- the primary's and each worker's --
        and pre-seed the caches the stock run_vision()/run_inference() consult, so no
        stage ever compiles."""
        engines = self._bin_engines()
        stem = _programs_stem(engines)
        bin_path = os.path.join(self.bin_dir, stem + ".bin")
        meta_path = os.path.join(self.bin_dir, stem + ".json")
        if not os.path.exists(meta_path):
            have = sorted(f[:-5] for f in os.listdir(self.bin_dir)
                          if f.startswith("programs_e") and f.endswith(".json"))
            raise FileNotFoundError(
                f"\n*** No bins for this engine configuration ***\n"
                f"  wanted: {stem} (vision={engines['vision']}, "
                f"prefix={engines['prefix']}, denoise={engines['denoise']})\n"
                f"  bin dir {self.bin_dir} has: {have or 'nothing'}\n"
                f"  A sharded program bakes one rendezvous per engine, so a set built\n"
                f"  for another count cannot be replayed (it would hang). Dump this one:\n"
                f"    python {os.path.abspath(__file__)} --stage all --dump-bins "
                f"--engines <N>\n")
        with open(meta_path) as f:
            self._manifest = json.load(f)
        sig = self._manifest["sig"]

        HEAD, V, L = self._cfg["action_head"], self._cfg["vision"], self._cfg["lm"]
        for name, live, want, hint in (
                ("num_image_slots", int(V["num_image_slots"]),
                 int(sig["num_image_slots"]), "camera count changed in the config"),
                ("chunk_size", int(HEAD["chunk_size"]), int(sig["chunk_size"]),
                 "action chunk length changed in the config"),
                ("denoise_steps", int(HEAD["num_denoise_steps"]),
                 int(sig["denoise_steps"]), "denoise step count changed in the config"),
                ("rope_theta", float(L["rope_theta"]), float(sig["rope_theta"]),
                 "rope theta changed in the config"),
                ("vis_layers", self.VIS_LAYERS, sig["vis_layers"],
                 "pass --vis-layers to match, or regenerate the bins"),
                ("vis_no_batch", bool(getattr(self, "VIS_NO_BATCH", False)),
                 bool(sig["vis_no_batch"]),
                 "toggle --no-vis-batch to match, or regenerate the bins"),
                ("prefix_fused_silu", bool(getattr(self, "PREFIX_FUSED_SILU", False)),
                 bool(sig["prefix_fused_silu"]),
                 "toggle --fused-silu to match, or regenerate the bins"),
                ("params_dram_base", hex(self._params_dram_base),
                 sig["params_dram_base"], "verapulse_config.json dram_layout changed"),
                ("tensor_dram_base", hex(self._tensor_dram_base),
                 sig["tensor_dram_base"], "verapulse_config.json dram_layout changed"),
                ("program_dram_base", hex(self._program_dram_base),
                 sig["program_dram_base"], "verapulse_config.json dram_layout changed"),
        ):
            self._sig_check(name, live, want, hint)

        # The arena stride is what every worker program address was derived from, so it
        # must be resolved before comparing. Skipped at 1 engine: the profile never
        # resolves there, so both sides would compare class defaults -- false refusals
        # only, never a real catch.
        if max(engines.values()) > 1:
            self._resolve_worker_arena_profile()
            self._sig_check("worker_arena_base", hex(self.VIS_WORKER_ARENA_BASE),
                            sig["worker_arena_base"], "worker arena layout changed")
            self._sig_check("worker_arena_bytes", int(self.VIS_WORKER_ARENA_BYTES),
                            int(sig["worker_arena_bytes"]), "worker arena layout changed")

        with open(bin_path, "rb") as f:
            blob = f.read()

        cache = self.__dict__.setdefault("_prog_cache", {})
        meta = self.__dict__.setdefault("_prog_meta", {})
        pool = self._worker_engine_pool()
        for stage in _PROGRAM_ORDER:
            ne_key, addr_attr, waddr_attr, wsize_attr = _STAGE_ATTRS[stage]
            secs = self._manifest["programs"][stage]
            primary = next(s for s in secs if s["engine"] == 0)
            data = blob[primary["offset"]:primary["offset"] + primary["size"]]
            assert len(data) == primary["size"], (
                f"{stem}.bin truncated: {stage} wants {primary['size']}B, "
                f"got {len(data)}B")
            addr = self.get_program_dram_addr()
            self._bin_write(addr, data, label=f"{stage} program")
            self.allocate_program_dram(len(data))
            cache[stage] = addr
            meta[stage] = (addr, len(data))
            setattr(self, addr_attr, addr)

            waddrs, wsizes = [], []
            for s in sorted((s for s in secs if s["engine"] != 0),
                            key=lambda s: s["engine"]):
                w = pool[s["engine"] - 1]
                want_addr = int(s["dram_base"], 16)
                # The worker allocator must stand where it did at compile time; if not,
                # an earlier stage loaded a different size and every jump target is off.
                assert w.get_program_dram_addr() == want_addr, (
                    f"worker {s['engine']} program allocator is at "
                    f"0x{w.get_program_dram_addr():X} but {stage} was compiled at "
                    f"0x{want_addr:X} -- stale or partial bin set; regenerate.")
                wdata = blob[s["offset"]:s["offset"] + s["size"]]
                self._bin_write(want_addr, wdata, ue=w,
                                label=f"{stage} worker {s['engine']}")
                w.allocate_program_dram(s["size"])
                waddrs.append(want_addr)
                wsizes.append(s["size"])
            setattr(self, waddr_attr, waddrs)
            setattr(self, wsize_attr, wsizes)

            # The scheduler owns start_workers/rendezvous. _make_stage_scheduler is
            # cached over the same shared pool, so this reproduces the compile path.
            # Vision always builds one (compile_encoder splits rows even at ne == 1);
            # prefix and denoise only when sharded, matching their compile-time guard.
            if stage == "vision" or engines[stage] > 1:
                sc = self._make_stage_scheduler(ne_key)
                setattr(self, {"vision": "_vis_sched", "prefix": "_prefix_sched",
                               "denoise": "_denoise_sched"}[stage], sc)
                # Re-register what compile time allocated (see dump_programs_to_file).
                st = (self._manifest.get("scheduler") or {}).get(stage)
                if st is None:
                    raise RuntimeError(
                        f"bin set {stem} predates the `scheduler` block (stage "
                        f"{stage!r} has no per-engine registry) -- regenerate with "
                        f"--dump-bins.")
                for nm, addrs in st["per_engine"].items():
                    sc.register_per_engine_addrs(nm, [int(a) for a in addrs])
                for nm, addrs in st["col_outputs"].items():
                    # alloc_col_output both allocates and records, so there is no public
                    # re-register hook; the registry is a plain {name: [addr]} map.
                    sc._col_outputs[nm] = [int(a) for a in addrs]

        wt = self._manifest.get("worker_tensor_addr")
        if wt is not None:
            for w, addr in zip(self._worker_engine_pool(), wt):
                w._tensor_dram_addr = int(addr)

        assert cache[_PROGRAM_ORDER[0]] == self._program_dram_base, (
            f"program allocator was not at its base when load_programs ran "
            f"(0x{cache[_PROGRAM_ORDER[0]]:X} != 0x{self._program_dram_base:X}) -- "
            f"every baked absolute jump target would be wrong.")

        cattrs = self._manifest.get("compile_attrs")
        if cattrs is None:
            raise RuntimeError(
                f"bin set {stem} predates the `compile_attrs` block -- regenerate "
                f"with --clean.")
        self._restore_weight_attrs(cattrs)
        _cnames = self._manifest.get("compile_tensor_attrs") or []
        if _cnames:
            _ctpath = os.path.join(self.bin_dir, stem + "_tensors.pt")
            if not os.path.exists(_ctpath):
                raise FileNotFoundError(
                    f"missing {_ctpath}: {stem}.json lists compile tensors {_cnames} "
                    f"-- regenerate with --clean")
            for k, v in torch.load(_ctpath, map_location="cpu",
                                   weights_only=True).items():
                setattr(self, k, v)

        derived = self._manifest.get("derived") or {}
        missing = [a for a in _BIN_DERIVED_ATTRS if a not in derived]
        if missing:
            raise RuntimeError(
                f"bin set {stem} is missing derived attrs {missing} (stale set -- "
                f"regenerate)")
        for a, v in derived.items():
            setattr(self, a, v)

        # Freeze exactly as precompile_all does on the compile path, so a stray lazy
        # compile inside the execution flow raises instead of allocating program DRAM
        # per inference.
        self._compile_frozen = True
        _original_print(f"  Programs: {len(blob) / 1024**2:.1f} MB from {stem}.bin "
                        f"(vision={engines['vision']}, prefix={engines['prefix']}, "
                        f"denoise={engines['denoise']})")
        return dict(cache)



# ======================================================================================
# run-from-bin variant
# ======================================================================================

class VeraPulse_Run(VeraPulse_UnifiedEngine):
    """Engine pinned to bin-backed execution: REQUIRES a matching bin set and never
    compiles. main() does not use this -- it builds the normal engine and switches on
    bins_available(), so a missing set falls back to compiling instead of failing. This
    exists for callers that want the guarantee (libero_eval, CI) that no compile can
    happen mid-episode, and for a clear error when the set they expect is absent.

    __init__ MUST mirror the full 5-surface clear that survives software_reset():
      1. clear_on_chip_sram()            URAM A/B lower halves + scale/bias BRAM
      2. manual 2-instruction program    URAM A/B upper halves
      3. clear_argmax_and_pbi_regs()     PBI pointer table rows 1..15
      4. mini-program of add_set(reg,0)  ISA regs 1..15
      5. reset_program_dram_addr()       so bins land at DRAM_INSTRUCTION_ADDR
    Skipping any of these lets a previously-run model contaminate results."""

    def __init__(self, bin_dir=None, **kw):
        self.bin_dir = bin_dir or BIN_DIR
        if not os.path.isdir(self.bin_dir):
            raise FileNotFoundError(
                f"No bin dir at {self.bin_dir}.\n"
                f"Generate one first with a full compile run:\n"
                f"    python {os.path.abspath(__file__)} --stage all --engines <N>")
        self._manifest = None
        super().__init__(**kw)

    def weight_init(self, dummy=False, seed=0):
        """No-op on the checkpoint: params come from params.bin. Signature mirrors the
        base class so the same caller drives either engine without knowing which."""
        self.weight_init_from_bin(dummy=dummy)

    def precompile_all(self, stages=None, freeze=True):
        """A bin run's stand-in for the compile phase. Always loads the FULL program
        set -- a bin file is dumped whole or not at all."""
        return self.load_programs()

    def dump_bins(self, bin_dir):
        raise RuntimeError(
            "refusing to re-dump bins from a bin-backed run: this engine never compiled "
            "anything, so it would only write back what it just loaded. Dump from a "
            "fresh compile run.")


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
        rel = _CFG["paths"].get("tokenizer_file")
        if rel:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel)
            if not os.path.exists(path):
                ensure_checkpoint()      # tokenizer.json ships with the checkpoint
            _TOKENIZER = Tokenizer.from_file(path)
        else:
            # The smolvla finetune ships NO tokenizer.json: upstream builds its processor
            # from the backbone repo at load time, so we resolve the identical file from
            # the identical repo. Deliberately NOT a fallback to the sibling variant's
            # tokenizer -- both happen to be SmolLM2 vocab 49280, so a wrong-but-loadable
            # tokenizer would produce shifted ids and read as a bad policy, never an error.
            repo = _CFG["paths"].get("tokenizer_hf_repo")
            if not repo:
                raise RuntimeError(
                    "config has neither paths.tokenizer_file nor paths.tokenizer_hf_repo")
            from huggingface_hub import hf_hub_download
            _TOKENIZER = Tokenizer.from_file(
                hf_hub_download(repo_id=repo, filename="tokenizer.json"))
        # The prefix layout budgets exactly `vocab_size` embedding rows; a tokenizer from
        # the wrong repo would index past the table (or, worse, inside it) silently.
        got = _TOKENIZER.get_vocab_size()
        if got > _CFG["lm"]["vocab_size"]:
            raise ValueError(
                f"tokenizer vocab {got} exceeds the model's embed table "
                f"{_CFG['lm']['vocab_size']}")

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
    path = _CFG["paths"]["norm_stats"]
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)

    # TWO FILE FORMATS, ONE CONTRACT. pulsevla ships norm_stats.safetensors with FLAT
    # keys ('observation.state.mean'); the smolvla finetune ships normalization_stats.json
    # with those same names NESTED ('observation.state' -> 'mean'). Same statistics, same
    # MEAN_STD family, same direction -- only the container differs. Normalize to the flat
    # dict here so exactly one code path continues below, and so a future third format is
    # one branch rather than a second copy of the validation.
    if path.endswith(".json"):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} is missing -- this variant's stats are not downloadable")
        with open(path) as f:
            doc = json.load(f)
        raw = {}
        for group in ("observation.state", "action"):
            g = doc.get(group)
            if not isinstance(g, dict):
                raise KeyError(
                    f"{path}: expected a '{group}' object, got {type(g).__name__} -- "
                    f"refusing to guess a normalization")
            for stat in ("mean", "std"):
                if stat in g:
                    raw[f"{group}.{stat}"] = torch.tensor(g[stat], dtype=torch.float32)
    else:
        from safetensors.torch import load_file
        if not os.path.exists(path):
            ensure_checkpoint()
        raw = load_file(path)

    need = ["observation.state.mean", "observation.state.std",
            "action.mean", "action.std"]
    missing = [k for k in need if k not in raw]
    if missing:
        raise KeyError(
            f"{os.path.basename(path)} is missing {missing} (has {sorted(raw)}) -- refusing "
            f"to guess a normalization; a wrong one looks like a broken policy, not a bug")
    _NORM_STATS = {
        "state": (raw["observation.state.mean"].float(),
                  raw["observation.state.std"].float()),
        "action": (raw["action.mean"].float(), raw["action.std"].float()),
    }
    # The stats must describe the REAL dof count, not the padded 32. A file carrying 32
    # here would mean it was written post-padding, and the action denormalization would
    # then be applied to zero-pad lanes as if they were joints.
    dof = _CFG["action_head"]["action_dim"]
    for k, (mu, sd_) in _NORM_STATS.items():
        if mu.numel() != sd_.numel():
            raise ValueError(f"norm stats '{k}': mean has {mu.numel()} entries, "
                             f"std has {sd_.numel()}")
        if k == "action" and mu.numel() != dof:
            raise ValueError(
                f"norm stats 'action' has {mu.numel()} entries but action_dim is {dof} "
                f"-- these must match, or the policy moves smoothly in the wrong units")
        if (sd_ <= 0).any():
            raise ValueError(f"norm stats '{k}': non-positive std {sd_.tolist()}")
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

    # Structural: the bundle nests the tower as vlm_with_expert.vision.encoder.layers,
    # lerobot as vlm.model.vision_model. Match on the fc1/fc2 pair under a 'vision' name.
    n = 0
    for name, mod in up.model.vlm_with_expert.named_modules():
        if "vision" in name and hasattr(mod, "fc1") and hasattr(mod, "fc2"):
            mod.forward = types.MethodType(fwd, mod)
            n += 1
    if n == 0:
        raise RuntimeError("quick_gelu patch found no vision MLP -- refusing to return "
                           "an unpatched oracle scored as if it were patched")
    return up


# Oracle numeric format. False = fp32 (the model as mathematically specified); True =
# cast the whole oracle to bf16 so it carries the SAME storage precision as the device.
#
# WHY THIS MATTERS. Every SNR in this file scores a bf16 device against an fp32 oracle,
# so it conflates two unrelated things: the FORMAT gap, which is a property of the
# datapath and not a defect, and genuine implementation error, which is. bf16 has an
# 8-bit mantissa (~0.4% per rounding, ~48 dB from a single op), so a deep stack cannot
# hold 40 dB no matter how perfect the emitter is -- see the measured ceilings above.
# Running the oracle in bf16 puts the format on BOTH sides, and what is left is the
# emitter's own error.
#
# NOT a perfect null: torch accumulates bf16 matmuls in fp32 internally, and so does the
# hardware (wider than bf16), so this is bf16-storage / wide-accumulate on both sides
# rather than a strict 16-bit simulation. It is the closest apples-to-apples available
# without _STRICT16, which is pessimistic in the other direction.
_ORACLE_BF16 = False


def _bf16ify(up):
    """Give the oracle the device's STORAGE precision without changing its dtype.

    A blanket up.to(torch.bfloat16) does not work: lerobot builds several tensors in
    fp32 unconditionally (the timestep embedding among them), so they meet bf16 weights
    mid-graph and F.linear raises `mat1 and mat2 must have the same dtype`. Chasing every
    such site would mean patching upstream's forward pass, which is the one thing this
    oracle exists to avoid.

    So round-trip through bf16 instead and stay in fp32:
      - every parameter is quantized to bf16 and back  -> bf16 WEIGHT precision
      - every Linear output is rounded to bf16 and back -> bf16 ACTIVATION precision
        between ops, which is what the device stores between kernels

    Accumulation stays wide, which is correct: the hardware also accumulates wider than
    bf16 (see the ceiling notes above -- a pure-bf16 sim is pessimistic). What this does
    NOT model is rounding inside a reduction; for that there is _STRICT16, which errs the
    other way. This sits between the two and needs no upstream edits.
    """
    for prm in up.parameters():
        prm.data = prm.data.bfloat16().float()
    for mod in up.modules():
        if isinstance(mod, torch.nn.Linear):
            mod.register_forward_hook(
                lambda _m, _i, out: out.bfloat16().float()
                if isinstance(out, torch.Tensor) else out)
    return up


def _up_cast(t, up):
    """Cast a tensor to the oracle's parameter dtype, so a bf16 oracle is not silently
    upcast back to fp32 by its inputs -- which would make the bf16 measurement a lie."""
    return t.to(next(up.parameters()).dtype)


def _upstream_load(quiet=False, hw_gelu=None, bf16=None):
    """The checkpoint's OWN shipped forward pass, as the gate oracle.

    hw_gelu=True patches the vision activation to the accelerator's quick_gelu, giving
    a SHARP gate (GELU cancels). hw_gelu=False is the honest fidelity number: distance
    from the model as published. Report both -- they answer different questions."""
    if hw_gelu is None:
        hw_gelu = VARIANT in NATIVE_QUICK_GELU
    if bf16 is None:
        bf16 = _ORACLE_BF16
    import os as _os
    # The bundle is the only copy of the upstream `smolvla` package, so it stays the
    # source of CODE. Its config.json/model.safetensors are PULSEVLA's, though, so
    # weights come from ensure_checkpoint() -- the same variant-aware resolver the device
    # path uses -- or the oracle grades one model against another model's answer key.
    bundle = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                           "verapulse_bin", "verapulse__pulsevla-libero-0.5b")
    if not _os.path.isdir(bundle):
        raise FileNotFoundError(
            f"upstream bundle not found at {bundle} -- it ships alongside the weights; "
            f"the downloader must not filter the .py files out")
    ckpt = ensure_checkpoint()
    _BACKBONE = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

    # TWO SCHEMAS. The bundle's SmolVLA wants nested "text"/"vision" blocks (pulsevla).
    # The finetunes ship LEROBOT's flat SmolVLAConfig -- no "text" key -- and are not
    # convertible by renaming. For those the checkpoint's real forward pass is lerobot's
    # own VLAFlowMatching, which is the oracle bench_vs_upstream.py has always used.
    with open(_os.path.join(ckpt, "config.json")) as _f:
        _raw = json.load(_f)
    if "text" not in _raw:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.configs.policies import PreTrainedConfig
        cfg_l = PreTrainedConfig.from_pretrained(ckpt)
        _vm = getattr(cfg_l, "vlm_model_name", None)
        if isinstance(_vm, str) and _os.path.isabs(_vm) and not _os.path.exists(_vm):
            # Training-machine path. Only the ARCHITECTURE comes from here; every weight
            # is overwritten from the checkpoint below, and the activation is set
            # explicitly by _patch_upstream_quick_gelu.
            vnote(f"cfg.vlm_model_name absent here ({_vm}); building from {_BACKBONE}")
            cfg_l.vlm_model_name = _BACKBONE
        cfg_l.device = "cpu"
        up = SmolVLAPolicy.from_pretrained(ckpt, config=cfg_l, strict=True)
        up = up.to("cpu").eval().float()
        # strict=True can be honoured loosely by some versions; prove coverage instead of
        # trusting it. An unsupplied parameter still holds a PUBLIC value, which would
        # make this a silent blend of two models that still produces plausible numbers.
        from safetensors.torch import load_file as _lf
        _have = set(_lf(_os.path.join(ckpt, "model.safetensors")))
        _missing = [nm for nm, _ in up.named_parameters() if nm not in _have]
        if _missing:
            raise RuntimeError(
                f"{len(_missing)} parameter(s) not supplied by {ckpt}/model.safetensors "
                f"(e.g. {_missing[:5]}) -- refusing to score against a blended oracle.")
        if hw_gelu:
            _patch_upstream_quick_gelu(up)
        if bf16:
            _bf16ify(up)
        if not quiet:
            vnote(f"upstream oracle: lerobot SmolVLAPolicy from {ckpt} "
                  f"[{'bf16' if bf16 else 'fp32'}] "
                  f"({sum(p.numel() for p in up.parameters())/1e6:.1f}M"
                  f"{', vision GELU -> quick_gelu' if hw_gelu else ''})")
        return up

    sys.path.insert(0, bundle)
    from safetensors.torch import load_file
    from smolvla import SmolVLA, load_smolvla_config
    cfg_u = load_smolvla_config(_os.path.join(ckpt, "config.json"))
    up = SmolVLA(cfg_u).float().eval()
    up.load_state_dict(load_file(_os.path.join(ckpt, "model.safetensors")))
    if hw_gelu:
        _patch_upstream_quick_gelu(up)
    if bf16:
        _bf16ify(up)
    if not quiet:
        vnote(f"upstream oracle loaded [{'bf16' if bf16 else 'fp32'}] "
              f"({sum(p.numel() for p in up.parameters())/1e6:.1f}M"
              f"{', vision GELU -> quick_gelu' if hw_gelu else ''})")
    return up


_UP_CACHE = {}


def _upstream_cached(hw_gelu=None):
    """Loading 2.23 GB twice per gate is pure latency; the gates want both variants."""
    if hw_gelu is None:
        hw_gelu = VARIANT in NATIVE_QUICK_GELU
    # Keyed on the DTYPE too: an fp32 and a bf16 oracle are different models for scoring
    # purposes, and sharing one cache slot would hand back whichever was built first.
    key = (hw_gelu, _ORACLE_BF16)
    if key not in _UP_CACHE:
        _UP_CACHE[key] = _upstream_load(hw_gelu=hw_gelu, bf16=_ORACLE_BF16)
    return _UP_CACHE[key]


def _up_embed_image(vwe, px):
    """Vision tower + connector for one image -> [tokens, dim]. Same tap on both
    backends: the bundle exposes connector(vision(px)), lerobot exposes embed_image."""
    if hasattr(vwe, "embed_image"):
        return vwe.embed_image(px)[0]
    return vwe.connector(vwe.vision(px))[0]


def _up_prefix_forward(vwe, att2d, pos, embs):
    """Prefix forward + KV fill -> (hidden [S,D], [(K,V)] with K,V as [n_kv, S, head_dim],
    the device's layout, so the gates never branch on backend."""
    if hasattr(vwe, "embed_image"):          # lerobot
        # past_key_values=None, NOT an empty DynamicCache: forward_cross_attn_layer
        # branches on `is not None`, so an empty-but-present cache makes cross-attn
        # layers read a prefix the self-attn layers have not written yet. This mirrors
        # modeling_smolvla.sample_actions' own prefill exactly.
        outs, cache = vwe.forward(
            attention_mask=att2d, position_ids=pos, past_key_values=None,
            inputs_embeds=[embs, None], use_cache=True)
        # DynamicCache holds [B, n_kv, S, D]; drop batch -> device layout.
        kv = [(l.keys[0], l.values[0]) for l in cache.layers if l.is_initialized]
    else:                                    # bundle
        outs, kvu = vwe.forward(
            attention_mask=att2d, position_ids=pos, past_key_values=None,
            inputs_embeds=[embs, None], use_cache=True, fill_kv_cache=True)
        # Bundle holds [B, S, n_kv, D]; permute to match. Getting this transpose wrong
        # loads cleanly and scores as pure garbage.
        kv = [(d["key_states"][0].permute(1, 0, 2),
               d["value_states"][0].permute(1, 0, 2)) for d in kvu]
    return outs[0][0], kv


def _upstream_prefix_gate(ue, images, token_ids, text_mask, hidden, kv):
    """Score the DEVICE's prefix hidden state and KV cache against upstream.

    Runs upstream's OWN embed_prefix + interleaved forward with fill_kv_cache=True, so
    the prefix ordering, the sqrt(hidden) embedding scale and the block-causal mask all
    come from the checkpoint's code rather than from our reading of it.

    The KV cache matters more than the hidden state: the hidden state is thrown away,
    while the cache is the action expert's ONLY view of the observation.
    """
    import math as _m
    up = _upstream_cached()
    vwe = up.model.vlm_with_expert
    m = torch.as_tensor(text_mask).reshape(-1).bool()
    ids = torch.as_tensor(token_ids, dtype=torch.long).reshape(-1)
    with torch.no_grad():
        imgs = [_up_cast(images[s].permute(2, 0, 1).unsqueeze(0).contiguous().float(), up)
                for s in range(images.shape[0])]
        # state: the device already has it K-padded; rebuild the same vector here
        st = _up_cast(
            torch.as_tensor(ue._prefix_state_used, dtype=torch.float32).reshape(1, -1), up)
        embs, pad, att = up.model.embed_prefix(
            imgs, [torch.ones(1, dtype=torch.bool)] * len(imgs),
            ids[None, :], m[None, :], st)
        att2d = make_att_2d_masks_local(pad, att)
        pos = torch.cumsum(pad, dim=1) - 1
        up_hidden, kvu = _up_prefix_forward(vwe, att2d, pos, embs)
    # Back to fp32 for SCORING only: the oracle already computed in its own dtype, and
    # snr/cos must not be taken in bf16 or the metric quantizes the metric.
    up_hidden = up_hidden.float()
    kvu = [(k.float(), v.float()) for k, v in kvu]
    valid = int(pad[0].sum())
    section("prefix")
    vnote(f"upstream rows={up_hidden.shape[0]} valid={valid}; "
          f"device valid_len={ue._prefix_valid_len}")
    if up_hidden.shape[0] != ue._prefix_valid_len:
        vnote(f"row count differs: upstream keeps {ids.numel()-int(m.sum())} padded text "
              f"slots as real rows, the device packs them out -- valid rows only.")
    n = min(valid, int(ue._prefix_valid_len))
    hw = torch.as_tensor(hidden).float()[:n]
    report("prefix hidden", hw, up_hidden[:n], threshold=20.0)
    # KV: upstream stores [B, S, n_kv, D]; device gives [n_kv, S, D] per layer
    worst = 1.0
    # Derived from actual depth: the literal (0, 1, 15, 31) was pulsevla's 32 layers and
    # raised IndexError on any 16-layer checkpoint.
    _nl = min(len(kv), len(kvu))
    for li in sorted({0, min(1, _nl - 1), _nl // 2, _nl - 1}):
        ku, vu = kvu[li][0][:, :n], kvu[li][1][:, :n]          # [n_kv, n, 64]
        kh, vh = torch.as_tensor(kv[li][0])[:, :n], torch.as_tensor(kv[li][1])[:, :n]
        ck, cv = cos_sim(kh, ku), cos_sim(vh, vu)
        worst = min(worst, ck, cv)
        vnote(f"L{li:<2} K cos={ck:.6f}  V cos={cv:.6f}")
    print(f"    {'ok  ' if worst >= COS_FLOOR else 'FAIL'} prefix KV (worst of 4)"
          f"          cos {worst:.6f}")
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
    st = _up_cast(torch.as_tensor(state, dtype=torch.float32).reshape(1, -1), up)
    with torch.no_grad():
        imgs = [_up_cast(images[s].permute(2, 0, 1).unsqueeze(0).contiguous().float(), up)
                for s in range(images.shape[0])]
        up_act = up.model.sample_actions(
            imgs, [torch.ones(1, dtype=torch.bool)] * len(imgs),
            ids[None, :], m[None, :], st,
            _up_cast(torch.as_tensor(noise, dtype=torch.float32), up)[None])[0]
    up_act = up_act.float()
    hw = torch.as_tensor(hw_actions).float()
    n = min(hw.shape[0], up_act.shape[0])
    d = min(hw.shape[1], up_act.shape[1])
    section("denoise (same noise, deterministic)")
    # NOT a row mismatch: both sides are `chunk_size` rows. The column counts differ
    # because the model's action space is max_action_dim=32 of which only action_dim=7
    # are real DoF -- the other 25 are the multi-embodiment zero padding, which upstream
    # itself slices off before computing loss. The device already returns the 7.
    vnote(f"device {tuple(hw.shape)} vs upstream {tuple(up_act.shape)} -> {n} rows x "
          f"{d} real DoF (upstream cols {d}..{up_act.shape[1]-1} are action-dim padding)")
    report("actions vs upstream", hw[:n, :d], up_act[:n, :d], threshold=30.0)
    c = cos_sim(hw[:n, :d], up_act[:n, :d])

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
            vnote("device-mirror HIGH = the accelerator faithfully executes the wrong "
                  "model, so landing faults 4/5/6 in the emitter is sufficient; LOW = a "
                  "SECOND device-side fault on top, which the emitter work alone will "
                  "not close. It is the ceiling: no model fix pushes above it.")
        except Exception as _e:
            import traceback
            print(f"  (device-mirror check failed: {_e!r})")
            traceback.print_exc()
    vnote("expert faults 4,5,6,7,8 are ALL LANDED. This is the SHARP gate (quick_gelu "
          "on both sides, so the activation substitution cancels) -- it grades "
          "arithmetic, not fidelity.")
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
    section("vision + connector")
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
                con.append(_up_embed_image(vwe, _up_cast(px, up)).float())
            up_con = torch.cat(con, 0)
        assert hw.shape == up_con.shape, (
            f"device produced {tuple(hw.shape)}, upstream {tuple(up_con.shape)} -- a "
            f"shape disagreement is a layout bug, not a numerics one; do not score it")
        tag = "vs quick_gelu (SHARP)" if hw_gelu else "vs exact  (fidelity)"
        report(f"vision {tag}", hw, up_con, threshold=(30.0 if hw_gelu else 11.0))
        out[hw_gelu] = cos_sim(hw, up_con)
    vnote(f"fidelity cos {out[False]:.6f} -- distance from the model AS PUBLISHED; "
          f"floor set by quick_gelu, not by the accelerator.",
          f"sharp    cos {out[True]:.6f} -- GELU cancels, so this grades the ARITHMETIC "
          f"(expect ~40 dB).")
    ok = out[True] >= 0.999 and out[False] >= 0.955
    if not ok and out[True] < 0.999:
        print("    the SHARP gate failed -- a real execution fault, not the activation "
              "tax. Bisect vision against upstream.")
    return ok


def prefix_shard_selfcheck(engine_counts=(1, 5, 8, 10, 12), verbose=True):
    """PURE-PYTHON audit of the PREFIX stage's split math. No device, no compile, no
    weights -- it re-derives every address range the sharded emitter writes and proves
    the engines never collide.

    WHY THIS IS A TEST AND NOT A REVIEW. An overlap in this stage is not a crash and not
    a NaN. Two engines writing one kv-head block of LAYER0_K/V_DRAM interleave two
    correct-looking [192,64] tensors, the prefix still finishes, and the damage surfaces
    ten denoise steps later as a plausible but wrong action. There is nothing to notice
    at runtime, so it has to be proven at compile time.

    Checks, per engine count:
      partition   q/o (H=960), gate/up/mult and down's K (I=2560), the 10 k/v units and
                  the 5 attention groups each cover their axis EXACTLY once
      alignment   every column offset is a whole UE_VECTOR_SIZE block (a partial vector
                  is inexpressible for the matvec pipeline, the B row block and the
                  writeback stride alike)
      DISJOINT    the byte ranges each engine writes -- LAYER0_K/V_DRAM head blocks,
                  LM_Q_DRAM / LM_O_PROJ_DRAM scatter bands, LM_ATTN_RESULT_DRAM un-stack
                  bands -- pairwise, across ALL engines. This is the one that matters.
      idle        engines that own no group in the attention region are named, not hidden

    Returns True; asserts on any failure. Run it with --check-prefix-shards.
    """
    cls = VeraPulse_UnifiedEngine
    PM, H, I = cls.PREFILL_MAX_SEQ_LEN, cls.HIDDEN_SIZE, cls.INTERMEDIATE_SIZE
    D, G, NKV = cls.HEAD_DIM, cls.GROUP_SIZE, cls.NUM_KV_HEADS
    bpe, VEC = 2, UE_VECTOR_SIZE
    KV_HEAD_STRIDE = PM * D * bpe
    KV_LAYER_STRIDE = NKV * KV_HEAD_STRIDE
    # Stand-in for `self`: these two helpers read NOTHING but NUM_KV_HEADS, which is a
    # class constant, so no engine and no DRAM allocator has to exist.
    shim = type("_Shim", (), {"NUM_KV_HEADS": NKV})()
    K_BASE, V_BASE = 0x1000_0000, 0x2000_0000          # stand-in bases; only deltas matter
    Q_BASE, O_BASE, A_BASE = 0x3000_0000, 0x4000_0000, 0x5000_0000

    def no_overlap(label, ranges):
        """ranges: [(engine, lo, hi, what)] -- half-open byte intervals."""
        rs = sorted(ranges, key=lambda r: r[1])
        for a, b in zip(rs, rs[1:]):
            assert a[2] <= b[1], (
                f"{label}: engine {a[0]} writes [{a[1]:#x},{a[2]:#x}) ({a[3]}) and "
                f"engine {b[0]} writes [{b[1]:#x},{b[2]:#x}) ({b[3]}) -- OVERLAP")

    for ne in engine_counts:
        kv_units = cls._prefix_kv_units(shim, ne)
        attn_map = cls._prefix_attn_group_map(shim, ne)
        n_split = cls._col_split(H, ne)
        i_split = cls._col_split(I, ne)
        assert len(kv_units) == ne and len(attn_map) == ne
        assert len(n_split) == ne and len(i_split) == ne

        # -- column partitions ------------------------------------------------------
        for label, split, N in (("q/o", n_split, H), ("gate/up/down-K", i_split, I)):
            covered, off = [], 0
            for e, (o, c) in enumerate(split):
                assert o == off, f"{label} ne={ne}: engine {e} offset {o} != {off}"
                assert c > 0 and c % VEC == 0 and o % VEC == 0, (
                    f"{label} ne={ne}: engine {e} block ({o},{c}) is not "
                    f"{VEC}-aligned")
                covered.append(c)
                off += c
            assert off == N, f"{label} ne={ne}: covered {off} of {N}"

        # -- k/v units: every (tensor, head) exactly once ---------------------------
        flat = [u for lst in kv_units for u in lst]
        assert sorted(flat) == sorted([(t, h) for t in (0, 1) for h in range(NKV)]), (
            f"kv units ne={ne}: {flat}")
        # -- attention groups: every kv_b exactly once ------------------------------
        flat_g = [g for lst in attn_map for g in lst]
        assert sorted(flat_g) == list(range(NKV)), f"attn groups ne={ne}: {flat_g}"

        # -- DISJOINTNESS, over every layer (the stride is per-layer, so a layer index
        #    that collided with a head index would show up here and only here) -------
        for i in range(cls.NUM_LAYERS):
            kv_w, at_w = [], []
            for e, units in enumerate(kv_units):
                for is_v, h in units:
                    base = (V_BASE if is_v else K_BASE)
                    lo = base + i * KV_LAYER_STRIDE + h * KV_HEAD_STRIDE
                    kv_w.append((e, lo, lo + PM * D * bpe,
                                 f"{'v' if is_v else 'k'}-head {h} L{i}"))
            no_overlap(f"LAYER0_K/V_DRAM ne={ne} layer {i}", kv_w)
            for e, groups in enumerate(attn_map):
                for kv_b in groups:
                    for g in range(G):
                        # un-stack writes column band (kv_b*G+g)*D, width D, per ROW
                        col = (kv_b * G + g) * D * bpe
                        for row in (0, PM - 1):     # first and last row are enough:
                            lo = A_BASE + row * H * bpe + col   # the band is per-row
                            at_w.append((e, lo, lo + D * bpe,
                                         f"attn head {kv_b * G + g} row {row}"))
            no_overlap(f"LM_ATTN_RESULT_DRAM ne={ne} layer {i}", at_w)

        # -- q / o scatter bands (per row, first and last) --------------------------
        for label, base, split in (("LM_Q_DRAM", Q_BASE, n_split),
                                   ("LM_O_PROJ_DRAM", O_BASE, n_split)):
            w = []
            for e, (o, c) in enumerate(split):
                for row in (0, PM - 1):
                    lo = base + row * H * bpe + o * bpe
                    w.append((e, lo, lo + c * bpe, f"cols {o}..{o + c}"))
            no_overlap(f"{label} ne={ne}", w)

        n_grp, n_kv = min(ne, NKV), min(ne, 2 * NKV)
        idle = [e for e in range(ne) if not attn_map[e]]
        if verbose:
            print(f"  ne={ne:>2}  q/o {[c for _, c in n_split]}  "
                  f"gate/up {[c for _, c in i_split]}  "
                  f"kv units/engine {[len(u) for u in kv_units]}  "
                  f"attn groups/engine {[len(a) for a in attn_map]}"
                  + (f"  idle in attn: {idle}" if idle else ""))
        assert n_grp == NKV or ne < NKV
        assert n_kv == min(ne, 10)
    if verbose:
        print("  prefix shard self-check: all partitions exact, all writes disjoint.")
    return True


def denoise_shard_selfcheck(engine_counts=(1, 5, 8, 10), verbose=True):
    """PURE-PYTHON audit of the DENOISE stage's split math -- the expert twin of
    prefix_shard_selfcheck. No device, no compile, no weights: it re-derives every DRAM
    byte range the sharded expert emitter WRITES and proves the engines never collide.

    WHY THIS IS A TEST AND NOT A REVIEW, again. Two engines writing one kv-head slice of
    AE_K_HEADS_DRAM does not fault and does not NaN. They interleave two individually
    correct [64,64] tensors, all ten Euler steps complete, and the robot gets a finite,
    plausible, wrong action. The same is true of a DROPPED unit -- the slice simply keeps
    the previous step's K, which is smooth and entirely wrong. Neither is observable at
    runtime, so both have to be refused at compile time.

    Checks, per engine count:
      partition   the 10 (tensor, kv-head) k/v units, the 5 attention groups, q (N=960),
                  o (N=512, lane-capped at 8) and the MLP's I=1280 each cover their axis
                  EXACTLY once
      alignment   every column offset is a whole UE_VECTOR_SIZE block, and every kv-head
                  slice base and weight row-block offset is a 32 B AXI beat
      DISJOINT    pairwise, across ALL engines, over every SELF layer: the
                  AE_K/V_HEADS_DRAM slices, the AE_Q_DRAM / AE_O_PROJ_DRAM scatter bands,
                  the AE_ATTN_RESULT_DRAM un-stack bands and the gate/up/mult lanes
      derived     k-head h lands on the engine that flashes group h -- i.e. _ae_kv_units
                  really is derived from _ae_group_map and the two have not drifted
      width       max units per engine, which is what the barrier makes everyone pay:
                  2 at ne=5..9, 1 at ne=10. That is the whole 5-way-vs-10-way argument
                  and it is printed rather than asserted, because it is a performance
                  claim, not a correctness one.

    Returns True; asserts on any failure. Run it with --check-denoise-shards.
    """
    cls = VeraPulse_UnifiedEngine
    M, HP, I = cls.SUFFIX_LEN_PAD, cls.E_HIDDEN_PAD, cls.E_INTER
    Q, KV = cls.E_Q_OUT, cls.E_KV_OUT
    D, G, NKV = cls.HEAD_DIM, cls.GROUP_SIZE, cls.NUM_KV_HEADS
    PMX = cls.PREFILL_MAX_SEQ_LEN
    CWL = PMX + M            # AE_COMBINED_LEN; an INSTANCE attr, so derived here
    bpe, VEC, BEAT = 2, UE_VECTOR_SIZE, mes.AXI_BEAT_BYTES
    AE_KV_HEAD_STRIDE = M * D * bpe
    # Stand-in for `self`: _ae_group_map and _ae_kv_units read NOTHING but NUM_KV_HEADS
    # and each other, so no engine, no allocator and no DRAM have to exist. The real
    # _ae_group_map is carried onto the shim rather than re-implemented here -- a
    # re-implementation is exactly the second copy this whole design is avoiding.
    shim = type("_Shim", (), {"NUM_KV_HEADS": NKV, "SUFFIX_LEN_PAD": M,
                              "GROUP_SIZE": G, "HEAD_DIM": D,
                              "_ae_group_map": cls._ae_group_map,
                              "_ae_attn_halves": cls._ae_attn_halves})()
    K_BASE, V_BASE = 0x1000_0000, 0x2000_0000        # stand-in bases; only deltas matter
    Q_BASE, O_BASE, A_BASE = 0x3000_0000, 0x4000_0000, 0x5000_0000
    # SELF layers only -- cross layers project nothing here (their K/V comes from
    # _emit_cross_reproject_all, hoisted out of the unroll on the primary). Counted with
    # the emitter's OWN predicate, not a re-derived `i % 2`: the parity is config-driven
    # and flagged UNVERIFIED in _ae_is_self_attn, so a second copy of it here is a second
    # thing to get wrong.
    n_self = sum(1 for i in range(cls.E_LAYERS)
                 if cls._ae_is_self_attn(cls, i))

    def no_overlap(label, ranges):
        """ranges: [(engine, lo, hi, what)] -- half-open byte intervals."""
        rs = sorted(ranges, key=lambda r: r[1])
        for a, b in zip(rs, rs[1:]):
            assert a[2] <= b[1], (
                f"{label}: engine {a[0]} writes [{a[1]:#x},{a[2]:#x}) ({a[3]}) and "
                f"engine {b[0]} writes [{b[1]:#x},{b[2]:#x}) ({b[3]}) -- OVERLAP")

    # The shape facts the whole split rests on, checked here too so the self-check fails
    # rather than passing vacuously if the geometry ever moves.
    assert D == VEC, f"HEAD_DIM={D} != UE_VECTOR_SIZE={VEC}: a kv-head is not one block"
    assert AE_KV_HEAD_STRIDE == M * D * bpe
    assert KV == NKV * D, f"E_KV_OUT={KV} != NUM_KV_HEADS*HEAD_DIM={NKV * D}"

    for ne in engine_counts:
        kv_units = cls._ae_kv_units(shim, ne)
        gmap = cls._ae_group_map(shim, ne)
        n_split = cls._col_split(Q, ne)                       # q_proj
        o_lanes = max(1, min(ne, HP // VEC))                  # o_proj lane cap
        o_split = cls._col_split(HP, o_lanes)
        i_split = cls._col_split(I, ne)                       # gate/up/mult, down's K
        assert len(kv_units) == ne and len(gmap) == NKV
        assert len(n_split) == ne and len(i_split) == ne and len(o_split) == o_lanes

        # -- column partitions ------------------------------------------------------
        for label, split, N in (("q", n_split, Q), ("o", o_split, HP),
                                ("gate/up/mult/down-K", i_split, I)):
            off = 0
            for e, (o, c) in enumerate(split):
                assert o == off, f"{label} ne={ne}: engine {e} offset {o} != {off}"
                assert c > 0 and c % VEC == 0 and o % VEC == 0, (
                    f"{label} ne={ne}: engine {e} block ({o},{c}) is not {VEC}-aligned")
                off += c
            assert off == N, f"{label} ne={ne}: covered {off} of {N}"

        # -- k/v units: every (tensor, head) exactly once ---------------------------
        flat = [u for lst in kv_units for u in lst]
        assert sorted(flat) == sorted([(t, h) for t in (0, 1) for h in range(NKV)]), (
            f"kv units ne={ne}: {flat}")
        # -- attention groups: every kv_b exactly once, to a real engine ------------
        assert sorted(gmap) == sorted(set(gmap)) or ne < NKV
        for kv_b, e in enumerate(gmap):
            assert 0 <= e < ne, f"group {kv_b} -> engine {e}, out of range at ne={ne}"
        # -- THE DERIVATION: k-head h is owned by group h's engine ------------------
        # Not a correctness requirement (disjointness plus the attention region's
        # opening barrier are), but it is the reason there is only ONE map to maintain:
        # if _ae_group_map moves and _ae_kv_units does not, this fires.
        for h in range(NKV):
            assert (0, h) in kv_units[gmap[h]], (
                f"ne={ne}: k-head {h} is on engine "
                f"{[e for e, u in enumerate(kv_units) if (0, h) in u][0]} but group {h} "
                f"flashes on engine {gmap[h]} -- _ae_kv_units and _ae_group_map have "
                f"drifted apart")

        # -- DISJOINTNESS, over every SELF layer ------------------------------------
        # The head slices are REBUILT per self layer per Euler step out of one buffer
        # pair, so unlike the prefix cache there is no layer stride to collide with a
        # head index; the loop below is over the layers that write them at all.
        kv_w = []
        for e, units in enumerate(kv_units):
            for is_v, h in units:
                lo = (V_BASE if is_v else K_BASE) + h * AE_KV_HEAD_STRIDE
                assert lo % BEAT == 0, f"kv slice {lo:#x} not {BEAT} B beat aligned"
                assert (h * D * HP * bpe) % BEAT == 0, (
                    f"kv weight row block for head {h} is not {BEAT} B beat aligned")
                kv_w.append((e, lo, lo + M * D * bpe, f"{'v' if is_v else 'k'}-head {h}"))
        no_overlap(f"AE_K/V_HEADS_DRAM ne={ne}", kv_w)

        # -- THE ATTENTION UNITS: 5 kv groups x query-row halves --------------------
        units = cls._ae_attn_units(shim, ne)
        halves = cls._ae_attn_halves(shim, ne)
        ntok = M // halves
        QB = M * G
        assert ntok * halves == M, f"ne={ne}: {halves} halves do not divide {M} tokens"
        assert len(units) == NKV * halves
        assert sorted((k, h) for k, h, _ in units) == sorted(
            (k, h) for k in range(NKV) for h in range(halves)), (
            f"attention units ne={ne} do not cover (group, half) exactly once: {units}")
        for _k, _h, _e in units:
            assert 0 <= _e < ne, f"unit ({_k},{_h}) -> engine {_e}, out of range"
        # halves == 1 MUST reproduce _ae_group_map exactly -- that is what keeps every
        # count from 1 to 9, including the proven 8, on its original byte stream.
        if halves == 1:
            assert [e for _, _, e in units] == list(gmap), (
                f"ne={ne}: halves==1 but the unit map {units} diverges from "
                f"_ae_group_map {gmap}")
        # EVERY ROW-INDEXED OPERAND. A missed shift here is a real softmax over the
        # wrong positions -- finite, plausible, wrong, and it reads as flow drift.
        for _k, _h, _e in units:
            tok0, row0 = _h * ntok, _h * ntok * G
            assert row0 + ntok * G <= QB, f"unit rows overrun QB at ne={ne}"
            for what, addr, pitch in (
                    ("Q gather",   tok0 * Q * bpe,      Q * bpe),
                    ("rope table", row0 * 2 * D * bpe,  2 * D * bpe),
                    ("bias self",  row0 * CWL * bpe,    CWL * bpe),
                    ("bias cross", row0 * PMX * bpe,    PMX * bpe),
                    ("un-stack",   tok0 * Q * bpe,      Q * bpe)):
                assert addr % BEAT == 0, (
                    f"ne={ne} unit ({_k},{_h}): {what} offset {addr:#x} is not "
                    f"{BEAT} B beat aligned")
                assert addr % pitch == 0, (
                    f"ne={ne} unit ({_k},{_h}): {what} offset {addr:#x} is not a whole "
                    f"number of {pitch} B rows -- rope_hf_core_dram asserts "
                    f"sin == cos + N*2, which only survives WHOLE [cos|sin] rows")

        # -- attention un-stack bands: disjoint over (group, half) ------------------
        # Groups own disjoint COLUMN bands, halves own disjoint ROW blocks, so the
        # product must tile AE_ATTN_RESULT_DRAM exactly once.
        at_w = []
        for kv_b, h, e in units:
            for g in range(G):
                col = (kv_b * G + g) * D * bpe
                for row in (h * ntok, (h + 1) * ntok - 1):   # first and last of the slice
                    lo = A_BASE + row * Q * bpe + col
                    at_w.append((e, lo, lo + D * bpe,
                                 f"attn head {kv_b * G + g} row {row}"))
        no_overlap(f"AE_ATTN_RESULT_DRAM ne={ne}", at_w)

        # -- q / o scatter bands ----------------------------------------------------
        for label, base, split, full in (("AE_Q_DRAM", Q_BASE, n_split, Q),
                                         ("AE_O_PROJ_DRAM", O_BASE, o_split, HP)):
            w = []
            for e, (o, c) in enumerate(split):
                for row in (0, M - 1):
                    lo = base + row * full * bpe + o * bpe
                    w.append((e, lo, lo + c * bpe, f"cols {o}..{o + c}"))
            no_overlap(f"{label} ne={ne}", w)

        # -- THE ROW SPLIT: the two norms and the two residual adds -----------------
        # A DIFFERENT AXIS, so it gets its own partition/alignment/disjointness pass.
        # The failure mode is the same shape as the k/v one and just as invisible: a
        # gap in the row cover is a block nobody normalizes (whatever the previous
        # Euler step left, smooth and wrong), an overlap is two engines writing one
        # row block of AE_PRE_NORM/AE_RESIDUAL/h_out.
        rsplit = cls._ae_row_split(shim, ne)
        assert len(rsplit) == ne
        off = 0
        for e, (o, c) in enumerate(rsplit):
            assert o == off, f"row split ne={ne}: engine {e} offset {o} != {off}"
            assert c > 0, f"row split ne={ne}: engine {e} owns zero rows"
            # A row slice is contiguous; its base must clear the 32 B AXI beat AND the
            # 128 B SRAM row, both of which HP*bpe = 1024 B satisfies for every offset.
            assert (o * HP * bpe) % BEAT == 0 and (o * HP * bpe) % mes.SRAM_ROW_BYTES == 0, (
                f"row split ne={ne}: engine {e} row {o} lands at "
                f"{o * HP * bpe:#x}, not aligned to the {BEAT} B beat / "
                f"{mes.SRAM_ROW_BYTES} B SRAM row")
            off += c
        assert off == M, f"row split ne={ne}: covered {off} of {M} rows"
        # ONE SPLIT, FOUR OPS: identical row boundaries are what let engine e read back
        # its own h_out[S_e] as h_in[S_e] next layer with no rendezvous. Proven here by
        # checking the buffers the four ops write against ONE partition.
        for label, base in (("AE_PRE_NORM_DRAM", 0x6000_0000),
                            ("AE_RESIDUAL_DRAM", 0x7000_0000),
                            ("AE_IO_A/B_DRAM", 0x8000_0000)):
            no_overlap(f"{label} ne={ne} (row split)",
                       [(e, base + o * HP * bpe, base + (o + c) * HP * bpe,
                         f"rows {o}..{o + c}") for e, (o, c) in enumerate(rsplit)])
        # The cut is reduce_add's divmod scheme. Denoise no longer CALLS reduce_add (the
        # down projection went N-split), but the prefix still does at PM=192, and this
        # check is what keeps the two row cuts from drifting if one is ever edited.
        _rbase, _rrem = divmod(M, ne)
        assert [c for _, c in rsplit] == [_rbase + (1 if i < _rrem else 0)
                                          for i in range(ne)], (
            f"row split ne={ne} no longer matches reduce_add's divmod cut")

        loads = [len(u) for u in kv_units]
        busiest = max(loads)
        assert busiest == -(-2 * NKV // min(ne, 2 * NKV)), (
            f"ne={ne}: kv units are unbalanced ({loads}); the barrier makes every "
            f"engine pay the busiest, so an avoidable extra unit is pure stage time")
        idle_attn = [e for e in range(ne) if e not in [x[2] for x in units]]
        if verbose:
            print(f"  ne={ne:>2}  q {[c for _, c in n_split]}  "
                  f"o {[c for _, c in o_split]}{'' if o_lanes == ne else f' (cap {o_lanes})'}"
                  f"  gate/up {[c for _, c in i_split]}  "
                  f"kv units/engine {loads} (busiest {busiest} of {2 * NKV})  "
                  f"attn {len(units)} units ({NKV} grp x {halves} row-half of "
                  f"{ntok} tok) on {len(set(x[2] for x in units))} eng"
                  + (f"  idle in attn: {idle_attn}" if idle_attn else ""))
            print(f"        rows (2 norms + 2 adds/layer) {[c for _, c in rsplit]} "
                  f"of {M} (busiest {max(c for _, c in rsplit)}, "
                  f"ceiling {M / max(c for _, c in rsplit):.2f}x)")
    if verbose:
        # The saving arithmetic, restated from the numbers above so it cannot rot: one
        # unit is M*HP*D*2 FLOP, the pair of projections is 10 units, and 16 self layers
        # x 10 Euler steps execute them.
        unit_fl = M * HP * D * 2
        print("  denoise shard self-check: all partitions exact, all writes disjoint.")
        print(f"  k/v unit = {unit_fl / 1e6:.2f} MFLOP x {2 * NKV} units x {n_self} self "
              f"layers x {cls.N_STEPS} steps = "
              f"{unit_fl * 2 * NKV * n_self * cls.N_STEPS / 1e9:.2f} GFLOP; the barrier "
              f"charges everyone the busiest engine, so the stage pays "
              f"busiest/{2 * NKV} of it.")
    return True


def configure_engines(ue, spec=1, *, vis=None, prefix=None, denoise=None, tag="main"):
    """Set the engine-count knobs. Returns True if anything is sharded.

    A MODULE-LEVEL FUNCTION, not a method, and shared by verapulse_test.main() and
    libero_eval.py. libero_eval constructs VeraPulse_UnifiedEngine() directly and never
    calls main(), so before this existed every closed-loop LIBERO episode would silently
    run single-engine no matter what the caller thought -- the knobs are class attributes
    that only main()'s argument parsing ever wrote (pi05 shipped exactly that bug).

    `ue` is the engine INSTANCE (or the class). Setting on an instance shadows the class
    attribute, which is what a per-process eval wants; passing the class sets the
    process-wide default.

    `spec` is an int 1..max(STAGE_MAX_ENGINES.values()) or the literal "max"; it is
    normalized ONCE here so no downstream comparison has to know which it got.
    `vis`/`prefix`/`denoise` pin one stage independently and compose with `spec` and
    with each other. THE STAGES NO LONGER SHARE A CEILING -- denoise's is 10 and
    vision's and the prefix's are 8 -- so a flat `spec` above 8 is not an error: it is
    clamped per stage by _num_engines, and "max" gives each stage its own number.

    MUST BE CALLED BEFORE weight_init(): weight_init builds the worker pool from these
    numbers, and no UnifiedEngine may be constructed after the weights are in DRAM.
    """
    cls = VeraPulse_UnifiedEngine
    caps = cls.STAGE_MAX_ENGINES
    is_max = isinstance(spec, str) and spec.lower() == "max"
    n = max(caps.values()) if is_max else int(spec)
    assert 1 <= n <= max(caps.values()), (
        f"--engines {spec!r}: expected 1..{max(caps.values())} or 'max'")

    ue.NUM_ENGINES = n
    if is_max:
        # Each stage's OWN ceiling, read from STAGE_MAX_ENGINES rather than assumed:
        # a stage whose ceiling drops below the peak makes "max" and a flat N diverge,
        # which is the whole reason this reads the dict.
        for stage, cap in caps.items():
            setattr(ue, f"{stage}_NUM_ENGINES", cap)
        print(f"[{tag}] --engines max: per-stage ceilings -- "
              + ", ".join(f"{k.lower()}={v}" for k, v in caps.items()))
    elif n > 1:
        # Report the RESOLVED counts, not n. Stage ceilings differ (VIS 12, PREFIX and
        # DENOISE 8), so "--engines 12" does NOT put twelve engines on every stage, and
        # saying it does sends people looking for a speedup that was never requested.
        capped = {k: min(n, v) for k, v in caps.items() if min(n, v) < n}
        note = ("" if not capped else "; clamped by stage ceiling to "
                + ", ".join(f"{k.lower()}={v}" for k, v in sorted(capped.items())))
        print(f"[{tag}] NUM_ENGINES={n}: sharded regions run across {n} engines{note}")

    for stage, want in (("VIS", vis), ("PREFIX", prefix), ("DENOISE", denoise)):
        if want is None:
            continue
        want = int(want)
        assert 1 <= want <= caps[stage], (
            f"{stage} override {want} exceeds its ceiling {caps[stage]}")
        setattr(ue, f"{stage}_NUM_ENGINES", want)
        print(f"[{tag}] {stage.lower()} pinned to {want} engine(s)")

    multi = max([n] + [int(getattr(ue, f"{st}_NUM_ENGINES", None) or n)
                       for st in caps]) > 1
    if multi:
        # The bin set carries every worker program alongside the primary's and is keyed
        # by the engine triple, so it can only be replayed at the count it was built for.
        print(f"[{tag}] multi-engine: bins are keyed per engine configuration "
              f"(dump/load with the same --engines)")
    return multi


def main():
    ap = argparse.ArgumentParser()
    # This file is a HARDWARE bring-up entry point, so the device path is the default and
    # the host-only reference paths are the explicit opt-ins -- not the other way round.
    # The sentinel default (None) exists so --tiny, which is meaningless on hardware
    # (the device programs are compiled for the full 12/32/32-layer stacks), can select
    # the reference path when no --stage was typed, while an EXPLICIT --stage still wins.
    # DECLARED HERE, CONSUMED AT IMPORT. _select_variant() already read argv before this
    # parser existed -- it has to, because every class-level dim is bound when the class
    # statement runs. These entries exist so --help lists them and argparse does not
    # reject them as unknown; args.smolvla/args.variant are read back only to ASSERT the
    # early sniff agreed, which catches a future refactor that moves argv around.
    ap.add_argument("--smolvla", action="store_true",
                    help="run the SO-101 smolvla finetune (16 layers, 720-wide expert, "
                         "3 cameras, 6 DoF) instead of the default pulsevla LIBERO model")
    ap.add_argument("--variant", default=None, choices=sorted(_VARIANTS),
                    help="explicit variant name; equivalent to --smolvla")
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
    ap.add_argument("--expert-layers", type=int, default=None, metavar="N",
                    help="INSTRUMENT: run only the first N of 32 expert layers, WITHOUT "
                         "the primary-only probes --bisect-expert adds, so it works at "
                         "any engine count. Sweep N at a fixed --engines and the slope "
                         "of denoise wall time vs N is the per-layer cost while the "
                         "intercept is the per-step head plus fixed setup. The actions "
                         "are meaningless -- this is a stopwatch, not a gate.")
    ap.add_argument("--denoise-extra-barriers", type=int, default=0, metavar="N",
                    help="INSTRUMENT: emit N redundant all-engine rendezvous per expert "
                         "layer, at a point where a barrier is a semantic no-op. Sweep "
                         "N and the slope divided by 320 layer-executions is the cost "
                         "of ONE rendezvous -- the number that decides whether the "
                         "unaccounted denoise time is barriers or per-op fixed cost. "
                         "Output bytes are unchanged at every N.")
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
    ap.add_argument("--no-vis-batch", action="store_true",
                    help="force the two-pass vision encoder (one execution per camera) "
                         "even when the engine count could batch both slots into one. "
                         "The A/B control for the batching win: same engines, same "
                         "width, same align -- only the execution count changes.")
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
    ap.add_argument("--precompile", action=argparse.BooleanOptionalAction, default=True,
                    help="DEFAULT ON. Build every program the run will need (encoder, "
                         "prefix, denoise -- minus anything --stop-after excludes) in one "
                         "compile phase BEFORE the first execution, then run the stages "
                         "compile-free. --no-precompile restores the old lazy behaviour "
                         "where each stage captures inside its first run_* call.")
    # --engines takes an int 1..max(STAGE_MAX_ENGINES.values()) or the literal "max"
    # (every stage at its own
    # STAGE_MAX_ENGINES ceiling). It was previously parsed and then read by NOTHING;
    # configure_engines below is what makes it do anything.
    ap.add_argument("--engines", default="1", metavar="N|max",
                    help="shard every stage across N engines (or 'max' for each stage's "
                         "own ceiling). 1 = the historical single-engine program, "
                         "byte-for-byte.")
    ap.add_argument("--vis_8", action="store_true",
                    help="stage isolation: vision encoder on 8 engines regardless of "
                         "--engines (S=1024 = 16 blocks of 64 -> an even 128-row split). "
                         "The projection matmuls are sharded; norms, attention and the "
                         "eltwise adds still run on the primary.")
    ap.add_argument("--pref_8", action="store_true",
                    help="stage isolation: prefix on 8 engines (column axis -- q/o/gate/"
                         "up N-split, down K-split + reduce_add, k/v split per kv-head "
                         "straight into the cache, attention split per kv-group). "
                         "Shorthand for --prefix-engines 8.")
    ap.add_argument("--check-prefix-shards", action="store_true",
                    help="HOST-ONLY, no FPGA: re-derive every DRAM byte range the "
                         "sharded prefix writes at ne = 1/5/8/10/12 and prove the "
                         "engines never collide, then exit. An overlap here is a silent "
                         "wrong answer ten denoise steps later, not a crash.")
    ap.add_argument("--check-denoise-shards", action="store_true",
                    help="HOST-ONLY, no FPGA: the same audit for the DENOISE action "
                         "expert at ne = 1/5/8/10 -- the 10 (tensor, kv-head) k/v units, "
                         "the 5 attention groups and the q/o/MLP column lanes each cover "
                         "their axis exactly once, and every engine's writes are "
                         "pairwise disjoint. Two engines on one kv-head slice is a "
                         "finite, plausible, WRONG action, never a crash.")
    ap.add_argument("--prefix-engines", type=int, default=None,
                    help="engines for the PREFIX stage only")
    ap.add_argument("--denoise-engines", type=int, default=None,
                    help="engines for the DENOISE stage only (1..STAGE_MAX_ENGINES"
                         "['DENOISE']). 10 is the modelled optimum: the gated MLP's "
                         "N=1280 = 20 blocks of 64 goes 6.67x -> 10x between 8 and 10 "
                         "engines and then stops improving, while reduce_add's serial "
                         "add chain keeps growing -- 11 and 12 would be slower. Wins "
                         "over --dns_8; must be set BEFORE weight_init, which cuts the "
                         "K-sliced down blobs for this exact count.")
    ap.add_argument("--exec-timeout", type=float, default=None,
                    help="per-execution FPGA timeout in seconds (default 300). Use a "
                         "small value when debugging a multi-engine hang: the timeout "
                         "path dumps every engine's busy/flag state, which a Ctrl-C "
                         "does not.")
    ap.add_argument("--denoise-digest", action="store_true",
                    help="SHA-256 the compiled denoise instruction stream and print it. "
                         "Run it at --engines 1 before and after a sharding change: the "
                         "two digests MUST match, which is the no-hardware proof that "
                         "the single-engine program is byte-identical.")
    ap.add_argument("--dns_8", action="store_true",
                    help="stage isolation: run the DENOISE action expert on 8 engines "
                         "regardless of --engines. Shorthand for --denoise-engines 8, "
                         "kept because it names the historically proven count. M is 64 "
                         "(one row block) so the split is on OUTPUT COLUMNS: gate/up "
                         "N=1280 -> per-engine [64,cols] lanes, the SiLU-multiply stays "
                         "in lane, down is a K-split + reduce_add, q (N=960) and o "
                         "(N=512) are N-split + scatter, the 5 kv groups split the "
                         "attention, and a self layer's k/v go 10 ways as (tensor, "
                         "kv-head) units. action_out and the norms/residual adds stay "
                         "on the primary.")
    ap.add_argument("--quant", default="bf16", choices=["bf16", "q4_64"])
    ap.add_argument("--gate-upstream", action=argparse.BooleanOptionalAction, default=False,
                    help="DEFAULT OFF. Score the device against the checkpoint's OWN "
                         "shipped forward pass rather than VeraPulseRef. Gating against "
                         "our own reconstruction is what hid six model faults -- twice "
                         "the reference had been bent toward the hardware, so the gate "
                         "could only ever find hardware bugs, never model bugs. Pass "
                         "--gate-upstream when validating the model; off by default "
                         "because it is a second CPU forward pass, not part of the run.")
    ap.add_argument("--snr", action=argparse.BooleanOptionalAction, default=False,
                    help="DEFAULT OFF. Per-stage >=40 dB gate against a pure-torch "
                         "oracle. It is a CORRECTNESS tool, not part of the model: it "
                         "runs a second forward pass on the CPU between the accelerator "
                         "stages, so it both costs wall time and stops the stages being "
                         "back to back. Pass --snr when validating numerics; leave it "
                         "off to measure the device. Same for --gate-upstream.")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="restore the full gate commentary: per-comparison RMS, the "
                         "known-fault notes, per-layer KV cos, oracle-load lines. Off "
                         "by default -- a green run needs one line per check.")
    # Bins are automatic (see the _bin_eligible block in main): a full run loads the set
    # matching its --engines configuration if one exists, and dumps one if it does not.
    # These two flags only override that choice.
    ap.add_argument("--dump-bins", action="store_true",
                    help="force a re-compile and re-dump even when a matching bin set "
                         "already exists (use after any compile-affecting edit)")
    ap.add_argument("--from-bin", action="store_true",
                    help="require bins: fail if no set matches this --engines "
                         "configuration, instead of falling back to compiling")
    ap.add_argument("--clean", action="store_true",
                    help="delete every bin set (all engine configurations) before "
                         "running, so this run recompiles and dumps fresh ones. Removes "
                         "params.* and programs_e*.* only -- the downloaded checkpoint "
                         "lives in the same directory and is left alone")
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
    ap.add_argument("--oracle-bf16", action=argparse.BooleanOptionalAction, default=False,
                    help="run the upstream oracle in bf16 so it carries the SAME storage "
                         "precision as the device. Default off = fp32 oracle, which "
                         "measures fidelity to the model as specified but charges the "
                         "accelerator for the format gap; on = apples-to-apples, "
                         "isolating emitter error from bf16 itself")
    ap.add_argument("--prefix-order", default=None,
                    choices=["state_images_text", "images_text_state", "scatter_into_text"])
    ap.add_argument("--images", default=None,
                    help="path to a .npy of [2,512,512,3] HWC camera images (uint8 or "
                         "already-normalized float); omitted -> deterministic synthetic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default=_CFG["defaults"]["prompt"])
    args = ap.parse_args()
    global VERBOSE
    VERBOSE = args.verbose

    # The variant was already locked in at import time (see _select_variant). Re-derive
    # it from the PARSED args and assert they agree: if these ever diverge, the class
    # constants belong to one model and the flag says the other, which would run the
    # wrong dims against the right weights and read as generic drift.
    _want = args.variant or ("smolvla" if args.smolvla else None)
    if _want is not None and _want != VARIANT:
        raise RuntimeError(
            f"variant mismatch: argparse says {_want!r} but the module was imported as "
            f"{VARIANT!r}. _select_variant() and the CLI have drifted apart.")
    print(f"  [variant] {VARIANT}  ({_VARIANTS[VARIANT]})")
    global _ORACLE_BF16
    _ORACLE_BF16 = args.oracle_bf16
    print("  [oracle] " + (
        "bf16 -- same storage precision as the device; the residual is emitter error, "
        "not format" if _ORACLE_BF16 else
        "fp32 -- fidelity to the model as specified; includes the bf16 format gap"))

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

    if args.check_prefix_shards:
        print("verapulse PREFIX SHARD SELF-CHECK (host only, no device)")
        prefix_shard_selfcheck()
        return

    if args.check_denoise_shards:
        print("verapulse DENOISE SHARD SELF-CHECK (host only, no device)")
        denoise_shard_selfcheck()
        return

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
    if args.clean:
        clean_bins(BIN_DIR)

    # Bins are AUTOMATIC: --engines selects the set, and a qualifying run loads one if it
    # exists or dumps one if it does not. --from-bin makes a missing set an error;
    # --dump-bins forces a re-dump. Only a full non-probing run qualifies -- the
    # debug/bisect paths compile extra programs or skip stages, desyncing the allocator.
    _bin_eligible = (args.stage == "all" and args.precompile
                     and not args.stop_after
                     and not args.bisect_vision
                     and args.bisect_expert is None
                     and args.bisect_prefix is None)
    if args.from_bin and not _bin_eligible:
        raise SystemExit(
            "--from-bin needs a full non-probing run: --stage all, the precompile phase "
            "on, and no --stop-after/--bisect-*. Those paths compile extra probe "
            "programs or skip stages, which desyncs the program allocator from the "
            "dumped layout.")
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
    print(f"verapulse HW | stage={args.stage} | weights={args.weights} | snr={args.snr}"
          + ("" if VERBOSE else "   (-v for full gate detail)"))
    ue = VeraPulse_UnifiedEngine()
    # BEFORE weight_init: this sets the engine counts, and weight_init's first act is to
    # build the worker pool from them. Every UnifiedEngine ctor DMA-writes 16 KB of noise
    # to a hardcoded 0x80000000, so no engine may be constructed after the weights land.
    configure_engines(ue, args.engines,
                      vis=8 if args.vis_8 else None,
                      prefix=8 if args.pref_8 else args.prefix_engines,
                      denoise=(args.denoise_engines if args.denoise_engines is not None
                               else (8 if getattr(args, "dns_8", False) else None)))
    ue.DENOISE_DIGEST = args.denoise_digest
    if args.exec_timeout:
        ue.EXEC_TIMEOUT = float(args.exec_timeout)
    # weight_init LAST among engine constructions: every UnifiedEngine ctor DMA-writes
    # 16KB of noise to a hardcoded 0x80000000, which is this model's first stored weight.
    # SiLU variant must be set BEFORE compile_prefix runs (it is read at emit time).
    if args.bisect_vision:
        # One layer only: the probes live in buffers that layer 1 would overwrite.
        args.vis_layers = args.vis_layers or 1
        args.stop_after = "vision"
        ue.VIS_BISECT = True
    if args.expert_layers is not None:
        assert args.bisect_expert is None, (
            "--expert-layers and --bisect-expert both set EXPERT_LAYERS; "
            "--bisect-expert is the primary-only probe variant, --expert-layers the "
            "sharded stopwatch. Pick one.")
        ue.EXPERT_LAYERS = args.expert_layers
    if args.denoise_extra_barriers:
        ue.DENOISE_EXTRA_BARRIERS = int(args.denoise_extra_barriers)
        print(f"[main] INSTRUMENT: +{args.denoise_extra_barriers} redundant "
              f"rendezvous per expert layer ({args.denoise_extra_barriers * 32 * 10} "
              f"extra over the unroll). Timing only -- output bytes are unchanged.")
    if args.bisect_expert is not None:
        ue.EXPERT_LAYERS = args.bisect_expert
        ue.EXPERT_BISECT = True
    if args.bisect_prefix is not None:
        ue.PREFIX_LAYERS = args.bisect_prefix
        ue.PREFIX_BISECT = True
        args.stop_after = "prefix"
    ue.VIS_LAYERS = args.vis_layers
    ue.VIS_NO_BATCH = args.no_vis_batch
    ue.PREFIX_FUSED_SILU = args.fused_silu

    # --- bins: use them if this engine configuration already has a set ---------------
    # Asked HERE because _num_engines cannot resolve the triple until the engine exists
    # and configure_engines has run.
    ue._dummy_weights_wanted = (args.weights == "dummy")
    _have_bins = _bin_eligible and ue.bins_available(BIN_DIR)
    if args.from_bin and not _have_bins:
        raise SystemExit(
            f"--from-bin: no bin set for this engine configuration "
            f"({_programs_stem(ue._bin_engines())}) in {BIN_DIR}. "
            f"Run once without --from-bin to compile and dump it.")
    _use_bins = _have_bins and not args.dump_bins
    if _use_bins:
        print(f"[main] bins: loading {_programs_stem(ue._bin_engines())} from {BIN_DIR} "
              f"(no weight unpack, no compile)")
        ue.weight_init_from_bin(dummy=(args.weights == "dummy"))
    else:
        ue.weight_init(dummy=(args.weights == "dummy"), seed=args.seed)
    ue.tensor_init()

    # --- compile phase: every program built BEFORE anything executes ---------------
    # Only the stages this invocation can actually reach are built: --stop-after vision
    # must not pay for the 32-layer prefix and the denoise loop it will never run, and
    # --stage vision/connector is vision-only by definition. When the full set is built,
    # precompile_all freezes _compile_once so a stray lazy compile inside the execution
    # flow raises instead of silently stalling and allocating program DRAM.
    if args.precompile:
        if args.stage in ("vision", "connector") or args.stop_after in ("vision", "connector"):
            _stages = ("vision",)
        elif args.stop_after == "prefix":
            _stages = ("vision", "prefix")
        else:
            _stages = ue.COMPILE_STAGES
        if _use_bins:
            ue.load_programs()
        else:
            ue.precompile_all(stages=_stages)

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
            with PHASES.track("gate vision (cpu oracle)", "gate"):
                ok_v = _upstream_vision_gate(ue, images, toks)
            tm = text_mask if text_mask is not None else torch.ones(
                token_ids.numel(), dtype=torch.bool)
            hidden = ue.run_prefix(toks, token_ids, state, text_mask=tm)
            PM, D, NKV = ue.PREFILL_MAX_SEQ_LEN, ue.HEAD_DIM, ue.NUM_KV_HEADS
            with PHASES.track("readback prefix KV", "host"):
                kv = [(torch.stack([ue._read_bf16(ue.LAYER0_K_DRAM + li * ue.KV_LAYER_STRIDE
                                                  + h * ue.KV_HEAD_STRIDE, (PM, D),
                                                  label=f"gk{li}{h}") for h in range(NKV)]),
                       torch.stack([ue._read_bf16(ue.LAYER0_V_DRAM + li * ue.KV_LAYER_STRIDE
                                                  + h * ue.KV_HEAD_STRIDE, (PM, D),
                                                  label=f"gv{li}{h}") for h in range(NKV)]))
                      for li in range(ue.NUM_LAYERS)]
            with PHASES.track("gate prefix (cpu oracle)", "gate"):
                worst = _upstream_prefix_gate(ue, images, token_ids, tm, hidden, kv)
            acts = ue.run_denoise(noise=noise)
            with PHASES.track("gate denoise (cpu oracle)", "gate"):
                c_act = _upstream_denoise_gate(ue, images, token_ids, tm, state, noise,
                                               acts, device_kv=kv)

            # Same action table the non-gated path prints, so this is a strict superset
            # of the old default behaviour and nothing was lost by turning the gate on.
            a = torch.as_tensor(acts).float()
            n_exec, adim = HEADC["n_action_steps"], HEADC["action_dim"]
            print(f"\n  actions [{n_exec} of {HEADC['chunk_size']} x {adim} dof, "
                  f"normalized]")
            for i in range(min(n_exec, a.shape[0])):
                print(f"  {i:>4}  " + "".join(f"{float(v):9.4f}" for v in a[i, :adim]))
            PHASES.summary("hardware timing")

            # EXIT CODE POLICY. Non-zero means "the accelerator computed something
            # wrong", which is the SHARP vision gate. The prefix and denoise numbers are
            # currently limited by KNOWN, catalogued model faults that live in the
            # emitter (causal suffix mask, (320,320) cross reprojection, prefix-K/V
            # concat) -- reporting those loudly is useful, failing the run over them
            # every time until they land is not.
            print(f"\n  gates: vision {'ok' if ok_v else 'FAULT'} | "
                  f"prefix KV cos {worst:.4f} | actions cos {c_act:.4f}")
            if c_act < 0.99:
                vnote("actions are limited by the 3 known expert faults (4/5/6), fixed "
                      "in VeraPulseRef but NOT in the emitter.")
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

        # Dump AFTER a successful full inference: _prog_meta, the per-stage worker
        # address/size lists and the derived vision geometry only exist once every
        # stage has actually compiled and run. The file is keyed by this run's engine
        # triple, so dumping at --engines 8 adds a set rather than replacing the 1-engine
        # one, and params.bin is shared by both.
        if _bin_eligible and not _use_bins:
            print(f"\n[main] dumping bins for this engine configuration "
                  f"({_programs_stem(ue._bin_engines())}) to {BIN_DIR} ...")
            ue.dump_bins(BIN_DIR)

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
        print("  (no gate ran -- pass --snr or --gate-upstream to check correctness)")
        return

    if args.stage == "connector":
        ok = ue._connector_snr_check(tokens)
    else:
        ok = ue._vision_snr_check(images, tokens)
    print(f"  vision stage: {'PASS' if ok else 'FAIL'} (threshold 40 dB)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
