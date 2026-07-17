#!/usr/bin/env python3
"""pi05_libero inference from pre-compiled bins (no weight unpack, no compile).

Generate the bins first with a full run of the golden path:

    python pi05_libero_test.py            # compiles, runs, then dumps pi05_libero_bin/

then either re-run pi05_libero_test.py (it auto-detects the bins and routes here) or:

    python pi05_libero_run_from_bin.py

WHY THIS SUBCLASSES THE TEST ENGINE (and does not re-implement it, unlike qwen3's
fully-self-contained run_from_bin template):

  * pi05's engine is ~3300 lines across three stacks (SigLIP vision encoder, Gemma-2B
    prefix LM, Gemma-300M action expert). A weight-free re-implementation is not
    realistic to keep in sync with the validated path.
  * pi05 CANNOT regenerate params.bin from HF: the weights come from a 13GB offline
    weights_export/*.npy tree. So params.bin is a DRAM snapshot (mobilesam's approach),
    not a re-derivable artifact.

WHAT THE BINS BAKE (and why the sig guard below refuses rather than adapts):
All three programs bake ABSOLUTE jump targets and ABSOLUTE tensor DRAM addresses --
the denoise program is statically unrolled to 10 steps x 18 layers = 180 layer bodies
with 1440 flash-attention jump sites. Nothing about a compiled program can be retargeted
after the fact. A bin set is therefore valid for exactly ONE (prefix_seq_len,
denoise_steps, keep_masked_slots, dram-base) combination. On any mismatch we REFUSE with
an actionable error telling the user to regenerate -- we do NOT pad the sequence, do NOT
silently recompile, and do NOT ship multiple bin sets.

HOW EXECUTION IS REUSED (the trick that keeps this file small):
Pi05Libero_UnifiedEngine already routes every compile through _compile_once(key, fn),
which returns a cached prog_addr and skips compilation entirely on a cache hit. So
load_programs() simply PRE-SEEDS _prog_cache with the bin-loaded addresses. The stock,
validated run_vision()/run_inference() then execute exactly as they always do and never
compile anything. No inference logic is duplicated here.
"""
import argparse
import json
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

from user_dma_core import DMA_DEVICE_H2C

import pi05_libero_test as _t
from pi05_libero_test import BIN_DIR, Pi05Libero_UnifiedEngine

# Program load order is load-bearing: program DRAM is a bump allocator, so replaying
# encoder -> prefix -> denoise is what reproduces the absolute addresses the programs
# were compiled against. This MUST match dump_programs_to_file's write order.
_PROGRAM_ORDER = ("encoder", "prefix", "denoise")


class Pi05Libero_Run(Pi05Libero_UnifiedEngine):
    """Bin-backed pi05_libero engine. Inherits the entire validated inference path;
    only weight loading and program provisioning are replaced."""

    def __init__(self, bin_dir=None, **kw):
        self.bin_dir = bin_dir or BIN_DIR
        meta_path = os.path.join(self.bin_dir, "programs.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"No compiled bins at {self.bin_dir} (missing programs.json).\n"
                f"Generate them first with a full run of the golden path:\n"
                f"    python {os.path.join(SCRIPT_DIR, 'pi05_libero_test.py')}")
        with open(meta_path) as f:
            self._manifest = json.load(f)
        self.sig = self._manifest["sig"]
        super().__init__(**kw)

        # --- sig guard, part 1: everything knowable before the model sees an obs ---
        # (part 2 -- prefix_seq_len -- can only be checked once vision has told us how
        # many image rows this obs actually produced; see embed_and_concat_prefix.)
        self._sig_check("denoise_steps", int(self.AE_NUM_DENOISE_STEPS),
                        int(self.sig["denoise_steps"]),
                        "pass --steps to match, or regenerate the bins")
        self._sig_check("keep_masked_slots", bool(self.KEEP_MASKED_SLOTS),
                        bool(self.sig["keep_masked_slots"]),
                        "toggle --keep-masked-slots to match, or regenerate the bins")
        for _name, _live in (("program_dram_base", self._program_dram_base),
                             ("tensor_dram_base", self._tensor_dram_base),
                             ("params_dram_base", self._params_dram_base)):
            self._sig_check(_name, hex(_live), self.sig[_name],
                            "pi05_libero_config.json's dram_layout changed since the bins "
                            "were compiled -- regenerate the bins")

        # PREFIX_SEQ_LEN is normally set by compile_prefix, which never runs here.
        # tensor_init_action_expert reads it (P -> Tkv -> every AE buffer size), so it
        # must be restored from the sig BEFORE that runs, or the AE buffers get sized
        # for the getattr default of 1024 and land at addresses the denoise program
        # does not reference.
        self.PREFIX_SEQ_LEN = int(self.sig["prefix_seq_len"])

        # Vision geometry that _weight_init_vision normally derives and publishes
        # (VIS_H/VIS_NH/VIS_D/VIS_DP/VIS_I/VIS_I_PAD/VIS_S/VIS_PATCH_K/VIS_HEAD_OUT).
        # weight_init() below replaces that whole path with a params.bin restore, so these
        # would never exist -- and tensor_init -> _tensor_init_vision reads them within
        # milliseconds of construction. Restored verbatim from the dump (NOT recomputed
        # from config) so they always match what the loaded programs were compiled
        # against. Set here, before any allocator or obs code can touch them.
        derived = self._manifest.get("derived")
        if not derived:
            raise RuntimeError(
                f"\n*** Bin set at {self.bin_dir} predates the `derived` block ***\n"
                f"  programs.json has no derived vision geometry, so this engine cannot\n"
                f"  reconstruct VIS_* without the 13GB weights_export unpack.\n"
                f"  Regenerate:  rm -f {os.path.join(self.bin_dir, '*.bin')} "
                f"{os.path.join(self.bin_dir, '*.json')} && "
                f"python {os.path.join(SCRIPT_DIR, 'pi05_libero_test.py')}\n"
                f"  (Do NOT rm -rf the bin dir: weights_export/ lives inside it.)\n")
        missing = [a for a in _t._BIN_DERIVED_ATTRS if a not in derived]
        if missing:
            raise RuntimeError(
                f"bin set at {self.bin_dir} is missing derived attrs {missing} "
                f"(stale bin set -- regenerate)")
        for _a, _v in derived.items():
            setattr(self, _a, _v)

    # ------------------------------------------------------------------
    # sig guard
    # ------------------------------------------------------------------

    def _sig_check(self, name, live, want, hint):
        if live != want:
            raise RuntimeError(
                f"\n*** REFUSING TO RUN: pre-compiled bins do not match this run ***\n"
                f"  {name}: bins were compiled for {want!r}, this run wants {live!r}\n"
                f"  bins: {self.bin_dir}\n"
                f"  The compiled programs bake absolute jump targets and absolute tensor\n"
                f"  DRAM addresses; they CANNOT be adapted to a different {name}.\n"
                f"  Fix: {hint}.\n"
                f"  Regenerate with:  rm -f {os.path.join(self.bin_dir, '*.bin')} "
                f"{os.path.join(self.bin_dir, '*.json')} && "
                f"python {os.path.join(SCRIPT_DIR, 'pi05_libero_test.py')}\n"
                f"  (Do NOT rm -rf the bin dir: weights_export/ lives inside it.)\n")

    # ------------------------------------------------------------------
    # Weights / params
    # ------------------------------------------------------------------

    def weight_init(self):
        """Replaces the ~13GB weights_export unpack with a params.bin DRAM restore.

        The ONE thing still read from weights_export is the token embedding table: it is
        a HOST-side gather table (embed_and_concat_prefix indexes it per obs), never a
        params-DRAM resident, so it is not in params.bin and cannot be. Read-only.
        """
        self.embedding_table = self._npy(
            "PaliGemma.llm.embedder.input_embedding").to(torch.bfloat16)
        self.load_params()

    def load_params(self):
        """Restore the params DRAM snapshot and rewind the params allocator to the
        post-weight_init boundary."""
        bin_path = os.path.join(self.bin_dir, "params.bin")
        meta_path = os.path.join(self.bin_dir, "params.json")
        for p in (bin_path, meta_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"missing {p} -- regenerate the bins")
        with open(meta_path) as f:
            total = json.load(f)["size"]
        actual = os.path.getsize(bin_path)
        assert actual == total, (
            f"params.bin is {actual}B but params.json declares {total}B -- truncated or "
            f"stale bin set; regenerate.")
        CHUNK = self.DMA_CHUNK_BYTES
        with open(bin_path, "rb") as f:
            offset = 0
            while offset < total:
                data = f.read(min(CHUNK, total - offset))
                self._dma_write_retry(DMA_DEVICE_H2C, self._params_dram_base + offset,
                                      data, len(data))
                offset += len(data)

        # Rewind to the post-weight_init boundary, NOT to `total`. params.bin is dumped
        # at the END of a full run, so it also contains what tensor_init (identity
        # matrix), compile_prefix (attn bias, rope tables) and tensor_init_action_expert
        # allocated afterwards. Those later allocations are replayed below by the very
        # same methods the golden path uses; if the pointer started at `total` they would
        # be re-allocated PAST every weight instead of landing back on the addresses the
        # programs bake. The bytes are all already correct on-device either way -- this
        # only fixes where the allocator thinks it is.
        self._next_params_dram_addr = (self._params_dram_base
                                       + int(self.sig["params_ofs_after_weight_init"]))
        _t._original_print(
            f"  Params: {total / 1024**2:.1f} MB from bin "
            f"(allocator rewound to +{self.sig['params_ofs_after_weight_init']}B)")
        return True

    # ------------------------------------------------------------------
    # Programs
    # ------------------------------------------------------------------

    def load_programs(self):
        """DMA the three compiled programs back to DRAM and pre-seed the caches the
        stock run_vision()/run_inference() consult, so neither ever compiles."""
        bin_path = os.path.join(self.bin_dir, "programs.bin")
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"missing {bin_path} -- regenerate the bins")
        with open(bin_path, "rb") as f:
            blob = f.read()

        cache = self.__dict__.setdefault("_prog_cache", {})
        meta = self.__dict__.setdefault("_prog_meta", {})
        for name in _PROGRAM_ORDER:
            m = self._manifest["programs"][name]
            data = blob[m["offset"]:m["offset"] + m["size"]]
            assert len(data) == m["size"], (
                f"programs.bin truncated: {name} wants {m['size']}B, got {len(data)}B")
            addr = self.get_program_dram_addr()
            self._dma_write_retry(DMA_DEVICE_H2C, addr, data, len(data))
            self.allocate_program_dram(len(data))
            cache[name] = addr
            meta[name] = (addr, len(data))
            _t._original_print(f"    {name:<8} {len(data):>10}B -> 0x{addr:X}")

        # The first program must land exactly on the program base; if it does not, the
        # allocator was already advanced and every baked jump target is off.
        assert cache[_PROGRAM_ORDER[0]] == self._program_dram_base, (
            f"program allocator was not at its base when load_programs ran "
            f"(0x{cache[_PROGRAM_ORDER[0]]:X} != 0x{self._program_dram_base:X}) -- "
            f"every baked absolute jump target would be wrong.")

        # Pre-seed run_inference's compile-once latches so it takes its execute-only
        # branches. _prefix_bias_dram is the one address that cannot be re-derived: the
        # bias DATA depends on this obs's valid_len, so run_inference re-DMAs a fresh
        # bias to the addr baked into the prefix program.
        self._prefix_prog_addr = cache["prefix"]
        self._prefix_bias_dram = self._params_dram_base + int(self.sig["prefix_bias_ofs"])
        self._prefix_compiled = True
        self._denoise_prog_addr = cache["denoise"]
        self._denoise_compiled = True
        _t._original_print(f"  Programs: {len(blob) / 1024**2:.1f} MB from bin")
        return dict(cache)

    # ------------------------------------------------------------------
    # Allocator replay hooks
    # ------------------------------------------------------------------

    def tensor_init(self, max_seq_len=512):
        """Run the REAL allocator unchanged, then restore the two prefix K/V cache
        pointers that compile_prefix normally publishes.

        Load-bearing, and silent if missed: tensor_init_action_expert falls back to
        allocating a ZERO PLACEHOLDER prefix K/V (36 buffers out of the params arena) if
        prefix_k_cache_addr is absent. In a bin run compile_prefix never executes, so
        without this the action expert would (a) attend over zeros instead of the real
        prefix cache and (b) drag both allocators off-layout, moving every AE buffer away
        from the addresses the denoise program bakes. Same derivation compile_prefix:959
        and load_prefix_kv:2677 use -- pure arithmetic over tensor_init's own buffers.
        """
        super().tensor_init(max_seq_len)
        self.prefix_k_cache_addr = [self.LAYER0_K_DRAM + l * self.KV_LAYER_STRIDE
                                    for l in range(self.NUM_LAYERS)]
        self.prefix_v_cache_addr = [self.LAYER0_V_DRAM + l * self.KV_LAYER_STRIDE
                                    for l in range(self.NUM_LAYERS)]

    def tensor_init_action_expert(self):
        """Restore both bump allocators to their pre-AE positions, then run the REAL
        allocator unchanged.

        Without this the AE buffers move. In the golden path compile_prefix runs before
        this and allocates from BOTH the tensor arena (residual/rope buffers) and the
        params arena (attn bias + rope tables). A bin run never compiles, so both
        pointers are short by exactly that much, and every AE_* address -- including
        AE_XT_DRAM, where the noise goes in and the action chunk comes out -- would be
        computed at an address the denoise program never touches. The failure mode is a
        finite-but-garbage action chunk, not a crash.
        """
        if not getattr(self, "_ae_tensors_inited", False):
            self._tensor_dram_addr = (self._tensor_dram_base
                                      + int(self.sig["tensor_ofs_before_ae"]))
            self._next_params_dram_addr = (self._params_dram_base
                                           + int(self.sig["params_ofs_before_ae"]))
        return super().tensor_init_action_expert()

    def embed_and_concat_prefix(self, token_ids, vision_embeddings=None, seq_len=1024):
        """sig guard, part 2. This is the earliest point prefix_seq_len is known: it is
        DERIVED from however many vision rows run_vision actually returned (dead camera
        slots are dropped) plus PREFIX_TEXT_BUDGET. Checked here -- before the prefix or
        denoise programs execute -- so a mismatched obs refuses instead of running a
        program compiled for a different sequence length. (run_vision's encoder has
        already executed by now, but the encoder program is per-image-slot and entirely
        independent of prefix_seq_len, so that is harmless.)
        """
        self._sig_check(
            "prefix_seq_len", int(seq_len), int(self.sig["prefix_seq_len"]),
            f"this observation produced {seq_len - self.PREFIX_TEXT_BUDGET} vision rows "
            f"+ {self.PREFIX_TEXT_BUDGET} text budget = {seq_len} prefix tokens, but the "
            f"bins were compiled for {self.sig['prefix_seq_len']}. This usually means a "
            f"different number of live camera slots (a dead/all-zero slot is dropped). "
            f"Regenerate the bins from an observation with this camera configuration")
        valid_len = super().embed_and_concat_prefix(token_ids, vision_embeddings, seq_len)
        # Normally set by compile_prefix (which never runs here). dump_prefix_kv reads it,
        # so a bin run would AttributeError on the --debug K/V dump without this. The
        # prefix program is invariant to valid_len -- it only shapes the bias DATA that
        # run_inference re-DMAs -- so tracking it per obs is correct, not just defensive.
        self.prefix_valid_len = valid_len
        return valid_len

    def dump_bins(self, bin_dir):
        raise RuntimeError(
            "refusing to re-dump bins from a bin-backed run: this engine never compiled "
            "anything, so it would only write back what it just loaded (and its allocator "
            "checkpoints are restored, not measured). Dump from pi05_libero_test.py.")


def main():
    ap = argparse.ArgumentParser(
        description="pi05_libero inference from pre-compiled bins.")
    ap.add_argument("--bin-dir", default=BIN_DIR, help=f"bin directory (default: {BIN_DIR})")
    args, _ = ap.parse_known_args()

    if not os.path.exists(os.path.join(args.bin_dir, "programs.json")):
        raise SystemExit(
            f"No compiled bins at {args.bin_dir}.\n"
            f"Generate them first:  python {os.path.join(SCRIPT_DIR, 'pi05_libero_test.py')}")

    # Delegate to the golden path's main(), which auto-detects the bins and constructs
    # Pi05Libero_Run. Deliberately NOT a duplicate of main()'s obs pipeline: the
    # sample-frame loading, PaliGemma tokenization, state discretization and action
    # unnormalization are all validated there and must not fork.
    _t.main()


if __name__ == "__main__":
    main()
