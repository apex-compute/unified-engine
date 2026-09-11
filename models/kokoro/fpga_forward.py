"""
Kokoro FPGA forward pass, built section-by-section on the UnifiedEngine
accelerator. This module is meant to fully replace KokoroModel.forward's
CUDA/CPU path once every section below is ported -- not a permanent parallel
path. Sections not yet ported here fall back to calling the corresponding
CPU submodule from the already-loaded `model` (a KokoroModel instance from
kokoro_test.py), so the pipeline stays runnable end-to-end while individual
sections come online.

Section status:
  [x] Section 1: PL-BERT encoder (embedding + 12x shared Albert transformer
      layer + bert_encoder projection) -> d_en
  [x] Section 2: Prosody/duration path (DurationEncoder + predictor.lstm +
      duration_proj -> pred_dur). NOTE: still missing the pred_aln_trg
      (repeat_interleave + scatter) alignment-matrix construction that
      consumes pred_dur -- that lands with Section 3, since it's the bridge
      into F0Ntrain's `en = d.transpose(-1,-2) @ pred_aln_trg`.
  [x] Section 3: F0/N prediction (pred_aln_trg construction + predictor.shared
      LSTM + AdainResBlk1d stacks) -> F0_pred, N_pred. First section to
      introduce Conv1d/AdaIN1d/depthwise-ConvTranspose1d/nearest-upsample
      into the actual model (each individually validated earlier in the
      op-gap hardware proofs, composed here for the first time).
  [x] Section 4: TextEncoder (phoneme embed + 3x weight-norm Conv1d k=5 +
      LayerNorm + LeakyReLU, then BiLSTM) -> t_en. NOT a transformer -- the
      only transformer in Kokoro is Section 1's PL-BERT. Reuses Section 3's
      _conv1d/_leaky_relu/_lstm_bidir directly; the only new mechanics are
      k=5/pad=2 and the channels-last LayerNorm, which is a plain
      layer_norm_core_dram(M=T, N=C) in our [T, C] convention.
  [x] Section 5a: Decoder front (F0_conv/N_conv + encode + 4x decode
      AdainResBlk1d) -> the Generator's input. First section where the CHANNEL
      axis needs 64-alignment (514 -> 576, 1090 -> 1152); padding is applied to
      the conv weights' input-channel axis, not by scrubbing activations.
  [x] Section 5b: ISTFTNet Generator body (ConvTranspose1d ups, noise_convs,
      dilated AdaINResBlock1 stacks with Snake1D, conv_post) -> GeneratorFPGA.
      InstanceNorm at T=30601 uses a chunked-reduction path (no large-N
      transpose); every op is one program with scratch released afterwards.
  [x] Section 5c: SineGen + tanh(l_linear) + STFT + CORDIC mag/phase -> `har`
      (GeneratorFPGA._source_har). Host keeps only the f0-derived increment
      tables (double-bf16 splits need fp32) and the model's random noise draw.
  [x] Section 5d: exp (softmax epilogue) / sin-cos (magic mod) + block-Toeplitz
      iSTFT -> audio (GeneratorFPGA._istft).

Weights are pulled directly from the loaded CPU `model`'s state dict at call
time (no bin-dump step yet -- that's a later optimization once the full
pipeline is ported, following the pattern in models/parakeet).

ALBERT weight sharing: the checkpoint only has ONE transformer layer's
weights (`encoder.albert_layer_groups.0.albert_layers.0.*`, num_hidden_groups=1),
applied 12 times -- confirmed by inspecting the state dict keys. So Section 1
only DMAs a single layer's weights and loops the same hardware program 12x.

nn.Linear weight layout ([out_features, in_features]) already matches
matmat_mul_core's B operand layout ([N, K], see its docstring at
user_dma_core.py:5234-5254 -- "A @ B^T"), so every Linear weight below is
uploaded as-is, no transpose needed.
"""
import builtins
import hashlib
import json
import math
import os
import time
from contextlib import contextmanager

import torch

# set_dma_device() REBINDS user_dma_core.DMA_DEVICE_H2C/C2H at runtime (user_dma_core.py:216-222),
# so those two must be read through the module at call time, never imported by value.
import user_dma_core as _udc
from user_dma_core import (UnifiedEngine, UE_MODE, UE_VECTOR_SIZE, set_dma_device,
                           ue_35bit_addr_shifter, INSTRUCTION_SIZE_BYTES,
                           DRAM_ACTIVATION_ADDR, DRAM_INSTRUCTION_ADDR)

# ---------------------------------------------------------------------------
# Compile-print suppression + per-section timing
#
# The cores print their tiling decisions (M_chunk/N_chunk, URAM occupancy, FLOP
# counts, capture/DMA sizes) on every call, which buries this module's own
# output. Same treatment every other model in the repo uses: swap builtins.print
# for a gated one (see models/llama3.2_1b/llama3.2_1b_test.py:68-76) and keep an
# unsuppressed `report` for our own lines.
# ---------------------------------------------------------------------------
_original_print = print
_SILENT_MODE = False


def quiet_print(*args, **kwargs):
    """Suppress prints when _SILENT_MODE is True; otherwise print normally."""
    if _SILENT_MODE:
        return
    _original_print(*args, **kwargs)


builtins.print = quiet_print
report = _original_print          # this module's own output, never suppressed
_DEBUG_SNR = [False]              # per-section SNR bisects: opt in with --debug-snr


def report_snr(*args, **kwargs):
    """Accuracy diagnostics. Off by default -- they are bring-up instrumentation, and computing
    the CPU references for them costs real time on top of the printing."""
    if _DEBUG_SNR[0]:
        _original_print(*args, **kwargs)

_STATS: "dict[str, list]" = {}    # name -> [compile_s, exec_s, runs, instructions]
# Per-program fingerprints. Stage 1 of the single-bin work is making every captured program
# byte-identical regardless of prompt length; this is how we know which ones already are.
_PROGRAMS: "list[dict]" = []
# Where every captured program ended up. Stage 2 of the single-bin work: programs are bump-
# allocated and left resident, so this is the manifest a dump_programs()/load_programs() pair
# (models/parakeet/parakeet_test.py:1616-1660) needs to read them back out of DRAM by name.
_PROGRAM_MAP: "list[tuple]" = []   # (name, dram_addr, size_bytes), in emission order
_SEC_SEQ: "dict[str, int]" = {}    # section label -> programs emitted so far, for unique names
_DUMP_PROGRAMS = [None]
_DUMP_FULL = [True]   # also keep raw instruction bytes, for field-level diffing
_DRAM_HIGH = [0]          # peak activation bytes in use, for the budget report
_CUR = [None]
_T_CAPTURE = [None]


def _begin(name):
    """Bill every subsequent _timed_run to `name`, until the next _begin."""
    _STATS.setdefault(name, [0.0, 0.0, 0, 0])
    _CUR[0] = name


@contextmanager
def _section(name):
    prev = _CUR[0]
    _STATS.setdefault(name, [0.0, 0.0, 0, 0])
    _CUR[0] = name
    try:
        yield
    finally:
        _CUR[0] = prev


def _instrument(ue):
    """Timestamp every start_capture so _timed_run can bill the emission phase.
    Wrapping the instance rather than editing ~40 call sites also covers captures
    opened inside helpers."""
    if getattr(ue, "_kokoro_instrumented", False):
        return
    orig = ue.start_capture

    def wrapped(*a, **k):
        _T_CAPTURE[0] = time.perf_counter()
        return orig(*a, **k)

    ue._kokoro_raw_start_capture = orig
    ue.start_capture = wrapped

    # Capacity sizing trades DRAM for a fixed address map, so guard the activation region.
    # Overrunning it would silently walk into the instruction region at 0xD0000000 and corrupt
    # the program being executed -- a failure that would look like garbage output, not an error.
    orig_alloc = ue.allocate_tensor_dram

    def guarded_alloc(size_bytes, *a, **k):
        addr = orig_alloc(size_bytes, *a, **k)
        end = ue._tensor_dram_addr
        if end > KOKORO_TENSOR_END:
            raise MemoryError(
                f"activation DRAM exhausted: high-water 0x{end:X} passed the program region "
                f"at 0x{KOKORO_TENSOR_END:X} "
                f"({(end - KOKORO_TENSOR_BASE) / 2**20:.0f} MB used of "
                f"{(KOKORO_TENSOR_END - KOKORO_TENSOR_BASE) / 2**20:.0f} MB). Lower "
                f"MAX_FRAMES/MAX_T, or reuse buffers instead of allocating fresh ones per call.")
        _DRAM_HIGH[0] = max(_DRAM_HIGH[0], end - KOKORO_TENSOR_BASE)
        return addr

    ue.allocate_tensor_dram = guarded_alloc

    # Permanently reserve the fixed GPR block (see its definition). Never released, so PBI
    # loop counters allocated inside cores start past it and can never alias a dimension
    # register -- the same discipline as llama3.2_1b_config.json's `fixed_isa_regs` plus
    # `_isa_reg_counter = max(fixed.values()) + 1`.
    want = sorted(_RESERVED_GPRS)
    reserved = [ue.alloc_isa_reg() for _ in range(len(want))]
    # Reserve the normalisation-length pool immediately after, so the cores' private loop
    # counters start past it.
    _NREG_POOL.clear()
    _NREG_MAP.clear()
    _NREG_POOL.extend(ue.alloc_isa_reg() for _ in range(_NREG_POOL_SIZE))
    assert reserved == want, (
        f"ISA register reservation got {reserved}, expected {want} -- something "
        f"allocated a register before _instrument() ran")
    # Reserve the preamble slot before any body is placed, so jump_abs targets never collide
    # with it and the very first program lands above it.
    _reset_programs(ue)
    ue._kokoro_instrumented = True


# --- Per-run preamble ------------------------------------------------------------------------
#
# The frozen-bin contract, as llama3.2_1b implements it (llama3.2_1b_test.py:1633-1645): the
# captured BODY contains no prompt-dependent immediate, and a tiny per-run program primes the
# runtime dimension registers and then jumps into it. The doorbell always points at the preamble,
# never at the body.
#
# The preamble is rewritten to the SAME address every run and must NOT advance the program cursor
# (gemma4_e2b_lm.py:1846-1874) -- otherwise the cursor walks forward and eventually overwrites the
# body it is supposed to jump into.
# 64 instruction slots. Sized generously on purpose: the preamble carries one ADD_SET per primed
# register, and that count grows as more dimensions become runtime values (frame counts, pad-row
# counts, a reduction-length pair per distinct normalisation axis). The slot is reserved once at
# the bottom of the program region, so over-sizing it costs 2 KB and nothing else. Jump targets
# must be 64 B aligned.
_PREAMBLE_BYTES = 2048
_PREAMBLE_ADDR = [None]


def _raw_start_capture(ue):
    """The pre-instrumentation start_capture, so preamble emission stays out of the timings."""
    return getattr(ue, "_kokoro_raw_start_capture", ue.start_capture)

# Runtime values for the preamble-primed registers, set by run_fpga_forward once the real phoneme
# and frame counts are known. Body emission never reads these -- only the preamble does.
_RT = {}


def set_runtime_dims(T=None, n_frames=None, T_pad=None, nf_pad=None, pad_rows=None):
    """Record the real sequence lengths for this run. These reach the device ONLY through the
    preamble, so changing them does not change a single byte of any captured program."""
    if T is not None:
        _RT[GPR_T] = int(T)
    if n_frames is not None:
        _RT[GPR_F] = int(n_frames)
    if T_pad is not None:
        _RT[GPR_TPAD] = int(T_pad)
    if nf_pad is not None:
        _RT[GPR_NFPAD] = int(nf_pad)
    if pad_rows is not None:
        _RT[GPR_PADROWS] = int(pad_rows)


def set_runtime_norm_len(ue, cap, real_n, key=None):
    """Declare the real reduction length for a normalisation axis, returning its two registers.

    ``key`` picks the register pair (default: the cap). Two axes with the same cap but different
    real lengths in one run (kokoro's pre/post-upsample AdaINs) must pass distinct keys.
    ``sqrt`` is baked host-side as bf19 because the ISA has no square root (the same reason
    gemma3 pre-computes a table of them, gemma3_test.py:863-875)."""
    n_reg, sq_reg = _n_regs(int(cap) if key is None else key)
    _RT[n_reg] = int(real_n)
    _RT[sq_reg] = ue.float_to_bf19(float(real_n) ** 0.5)
    return n_reg, sq_reg


def _reset_programs(ue):
    """Rewind the program cursor, keeping the preamble slot reserved at the very bottom.

    Called ONCE, from _instrument(). It used to run after every program, which is why nothing
    could be dumped: all ~12 programs were written to the same address and only the last
    survived. Programs are now bump-allocated and stay resident for the whole run.
    """
    ue.reset_program_dram_addr()
    addr = ue.allocate_program_dram(_PREAMBLE_BYTES)
    if _PREAMBLE_ADDR[0] is None:
        _PREAMBLE_ADDR[0] = addr
    assert addr == _PREAMBLE_ADDR[0], "preamble slot moved between runs"
    _PROGRAM_MAP.clear()
    _SEC_SEQ.clear()


def _program_name():
    """Name a program by the section it is billed to plus its index within that section, so the
    manifest key survives across runs as long as the section emits the same programs in the same
    order -- which is the same invariant the fingerprint dump already checks."""
    sec = _CUR[0] or "unbilled"
    i = _SEC_SEQ.get(sec, 0)
    _SEC_SEQ[sec] = i + 1
    return f"{sec}#{i}"


_PROBE_ONCE = set()


def _nan_probe(ue, runner, label, addr, shape):
    """Under --debug-snr, report where a buffer first goes non-finite.

    The SNR bisect only shows -inf for a whole block, which says NaN happened SOMEWHERE inside it.
    This narrows it to the individual buffer. Fires once per label so a loop over layers does not
    flood the log.
    """
    if not _DEBUG_SNR[0] or label in _PROBE_ONCE:
        return
    _PROBE_ONCE.add(label)
    _debug_flush(ue, runner)
    t = ue.dma_from_accelerator_memory(addr, shape).float()
    n_nan = int(torch.isnan(t).sum())
    n_inf = int(torch.isinf(t).sum())
    flag = "  <-- NON-FINITE" if (n_nan or n_inf) else ""
    fin = torch.isfinite(t).reshape(-1)
    amax = t[torch.isfinite(t)].abs().max().item() if fin.any() else float("nan")
    extra = ""
    if not fin.all():
        bad = (~fin).nonzero().flatten()
        first = int(bad[0])
        # A contiguous non-finite TAIL means the op wrote only a prefix and the rest is untouched
        # DRAM -- a wrong write length. Scattered non-finites mean the op computed NaN. Completely
        # different bugs, so say which one this is.
        tail = bool(fin[:first].all() and not fin[first:].any())
        extra = (f" first_bad={first} (={first / t.shape[-1]:.2f} rows) "
                 f"{'CONTIGUOUS TAIL -> short write' if tail else 'scattered -> computed NaN'}")
    report(f"[fpga][probe] {label:34s} n={t.numel():<9d} nan={n_nan:<8d} inf={n_inf:<8d} "
           f"absmax={amax:.3e}{flag}{extra}")


def _debug_flush(ue, runner):
    """Make DRAM readable from inside an open capture, for the --debug-snr bisect only.

    Captures merged across whole sections, so a _check() in the middle of one now reads DRAM that
    the captured-but-not-yet-executed instructions have not written -- it would report the SNR of
    stale memory, which looks like a correctness bug that is not there. Flushing runs what has been
    captured so far and reopens a capture, so the read sees real results.

    This SPLITS the section into more programs than a normal run emits, so a --debug-snr run is not
    representative of the frozen image. That is the right trade: this path exists to localise a
    numerical fault, not to measure the freeze.
    """
    if not getattr(ue, "is_capture_on", False):
        return False
    runner(ue)
    ue.start_capture()
    return True


def _check_program_region(ue):
    """Nothing rewinds the program cursor any more, so it can now run off the end of the region.
    Past 0x100000000 the XDMA char device BLOCKS inside os.write() instead of returning an error
    (see the region map above), so an overrun would wedge the process with no diagnostic at all."""
    end = ue.get_program_dram_addr()
    if end > KOKORO_PROGRAM_END:
        raise MemoryError(
            f"program DRAM exhausted: cursor 0x{end:X} passed the end of DRAM at "
            f"0x{KOKORO_PROGRAM_END:X} "
            f"({(end - KOKORO_PROGRAM_BASE) / 2**20:.0f} MB used of "
            f"{(KOKORO_PROGRAM_END - KOKORO_PROGRAM_BASE) / 2**20:.0f} MB) after "
            f"{len(_PROGRAM_MAP)} programs.")


# --- Frozen-image cache ------------------------------------------------------------------------
#
# The point of everything above -- GPR-carried dimensions, cap-templated bodies, capped
# allocations -- is that the compiled instruction streams no longer depend on the prompt. So they
# can be compiled once, written to disk, and reloaded forever.
#
# The cache key is deliberately NOT the sequence length. parakeet keys its bins on L_pad and
# recompiles whenever the audio length changes (parakeet_test.py:1650); doing that here would
# reintroduce exactly the per-prompt recompile this work removed. The key is the compiler
# fingerprint plus the capacity caps, because those -- not the prompt -- are what the emitted
# bytes actually depend on.
# SCOPE, honestly: a cache hit skips every instruction DMA and every weight upload, and asserts
# that each program lands at the address it was compiled at -- so the bytes the accelerator runs
# are provably the frozen ones. It does NOT yet skip the Python emission pass, because that pass
# is also what computes the DRAM addresses the per-run host uploads target. Making the replay
# emission-free needs those addresses recorded in the manifest too; that is the next step, not a
# property of what is here.
_BIN_CACHE = [None]        # directory holding params.bin / programs.bin, or None to disable
_BIN_LOADED = [False]      # True when this run replayed a cached image instead of compiling
_LOADED_PROGRAMS: "dict" = {}   # name -> compiled address, from a loaded manifest


def _cache_key():
    """What the compiled image actually depends on. A change in any of these invalidates it."""
    src = open(os.path.abspath(__file__), "rb").read()
    return {
        "source_sha1": hashlib.sha1(src).hexdigest(),
        "max_T": MAX_T,
        "max_frames": MAX_FRAMES,
        "vector_size": UE_VECTOR_SIZE,
        "params_base": f"0x{KOKORO_PARAMS_BASE:X}",
        "tensor_base": f"0x{KOKORO_TENSOR_BASE:X}",
        "program_base": f"0x{KOKORO_PROGRAM_BASE:X}",
    }


# The three files that make up a frozen image. kokoro_test.py --clean removes exactly these.
FROZEN_IMAGE_FILES = ("params.bin", "programs.bin", "programs.json")


def dump_bin_cache(ue, out_dir):
    """Write the params region and every resident program to disk, with a manifest.

    Reads the regions back over C2H DMA rather than re-serialising host-side, so what lands on
    disk is exactly what the accelerator will replay.
    """
    os.makedirs(out_dir, exist_ok=True)
    params_end = ue.get_params_dram_addr()
    params_size = params_end - KOKORO_PARAMS_BASE
    blob = bytearray()
    CHUNK = 1 << 20
    for off in range(0, params_size, CHUNK):
        n = min(CHUNK, params_size - off)
        buf = bytearray(n)
        ue.dma_read(_udc.DMA_DEVICE_C2H, KOKORO_PARAMS_BASE + off, buf, n)
        blob.extend(buf)
    with open(os.path.join(out_dir, "params.bin"), "wb") as f:
        f.write(bytes(blob))

    prog_bytes = bytearray()
    manifest = {"key": _cache_key(), "params_size": params_size, "programs": {},
                # The programs' operands hold ABSOLUTE weight addresses, so a reload has to put
                # every constant back at the address it was compiled at. Persist the map rather
                # than re-uploading, which would bump-allocate fresh (wrong) addresses.
                "consts": [[k[0], list(k[1]), k[2], f"0x{a:X}"] for k, a in _UP_CACHE.items()]}
    for name, addr, size in _PROGRAM_MAP:
        manifest["programs"][name] = {"offset": len(prog_bytes), "size": size,
                                      "addr": f"0x{addr:X}"}
        buf = bytearray(size)
        ue.dma_read(_udc.DMA_DEVICE_C2H, addr, buf, size)
        prog_bytes.extend(buf)
    with open(os.path.join(out_dir, "programs.bin"), "wb") as f:
        f.write(bytes(prog_bytes))
    with open(os.path.join(out_dir, "programs.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    report(f"[fpga] wrote frozen image to {out_dir} "
           f"({params_size / 2**20:.0f} MB params, {len(prog_bytes) / 2**10:.0f} KB across "
           f"{len(_PROGRAM_MAP)} programs)")


def load_bin_cache(ue, in_dir):
    """Restore a previously dumped image. Returns the manifest, or None if unusable.

    Every program goes back to the ABSOLUTE address it was compiled at -- the jump targets and
    operand addresses baked into the stream are absolute, so relocating would corrupt them.
    """
    mpath = os.path.join(in_dir, "programs.json")
    if not (os.path.exists(mpath) and os.path.exists(os.path.join(in_dir, "params.bin"))
            and os.path.exists(os.path.join(in_dir, "programs.bin"))):
        return None
    with open(mpath) as f:
        manifest = json.load(f)
    if manifest.get("key") != _cache_key():
        report("[fpga] frozen image is stale (compiler or capacity changed), recompiling")
        return None

    with open(os.path.join(in_dir, "params.bin"), "rb") as f:
        params = f.read()
    if len(params) != manifest["params_size"]:
        return None
    ue.dma_write(_udc.DMA_DEVICE_H2C, KOKORO_PARAMS_BASE, params, len(params))

    _UP_CACHE.clear()
    for dtype, shape, digest, addr in manifest.get("consts", []):
        _UP_CACHE[(dtype, tuple(shape), digest)] = int(addr, 16)

    with open(os.path.join(in_dir, "programs.bin"), "rb") as f:
        progs = f.read()
    _LOADED_PROGRAMS.clear()
    for name, meta in manifest["programs"].items():
        addr = int(meta["addr"], 16)
        _LOADED_PROGRAMS[name] = addr
        chunk = progs[meta["offset"]:meta["offset"] + meta["size"]]
        ue.dma_write(_udc.DMA_DEVICE_H2C, addr, chunk, len(chunk))
    return manifest


def get_program_manifest():
    """{name: (dram_addr, size_bytes)} for every program still resident from the last run --
    the argument dump_programs() takes in parakeet_test.py."""
    return {name: (addr, size) for name, addr, size in _PROGRAM_MAP}


def _run_via_preamble(ue, body_addr):
    """Prime the runtime dimension registers, then jump into the frozen body."""
    if not _RT:
        # Nothing dynamic was declared: run the body directly, as before.
        ue.start_execute_from_dram(body_addr)
        return
    ue.clear_inst_id()
    # Deliberately the UNINSTRUMENTED start_capture: the preamble is emitted during the execute
    # phase, so letting it stamp the compile-phase timer would bill this run's preamble to the
    # NEXT section and report a negative compile time.
    _raw_start_capture(ue)()
    for reg, val in sorted(_RT.items()):
        ue.generate_instruction_add_set(reg, val)
    ue.generate_instruction_jump_abs(ue_35bit_addr_shifter(body_addr))
    ue.stop_capture()
    size = ue.get_capture_instruction_size_bytes()
    assert size <= _PREAMBLE_BYTES, (
        f"preamble is {size} B ({size // INSTRUCTION_SIZE_BYTES} instructions priming "
        f"{len(_RT)} registers), slot is {_PREAMBLE_BYTES} B -- raise _PREAMBLE_BYTES")
    ue.write_captured_instructions_to_dram(_PREAMBLE_ADDR[0])
    ue.clear_capture_buffer()
    ue.start_execute_from_dram(_PREAMBLE_ADDR[0])


def _timed_run(ue):
    """Shared by every section class. COMPILE = instruction emission (the Python
    capture pass) plus writing the program to DRAM. HW EXEC = the accelerator
    running it: start_execute_from_dram -> wait_queue."""
    t0 = time.perf_counter()
    ue.stop_capture()
    ue.generate_instruction_halt()
    size = ue.get_capture_instruction_size_bytes()
    prog = ue.get_program_dram_addr()
    name = _program_name()
    if _BIN_LOADED[0]:
        # The image already holds this program at this address, byte for byte -- that is the whole
        # point of the freeze. Skip the DMA and just take the slot. Emission still ran (it is what
        # computes the addresses and the host-side data uploads), but nothing is written.
        expect = _LOADED_PROGRAMS.get(name)
        if expect is not None and expect != prog:
            raise RuntimeError(
                f"frozen image mismatch: program {name!r} was compiled at 0x{expect:X} but this "
                f"run wants 0x{prog:X}. The image is not interchangeable with this code path.")
    else:
        ue.write_captured_instructions_to_dram(prog)
    # Bump-allocate and leave it there: every program of the run keeps its own address, so the
    # whole set can be DMA-read back afterwards and frozen into a .bin.
    addr = ue.allocate_program_dram(size, label=name)
    assert addr == prog, f"program moved between write and allocate: 0x{prog:X} -> 0x{addr:X}"
    _PROGRAM_MAP.append((name, prog, size))
    _check_program_region(ue)
    n_inst = size // INSTRUCTION_SIZE_BYTES
    if _DUMP_PROGRAMS[0] is not None:
        # Hash the exact instruction stream that was just written to DRAM. Any prompt-dependent
        # value baked into an operand -- a loop trip count, an ADD_SET immediate, a shifted DRAM
        # base -- changes this digest even when the instruction COUNT is unchanged.
        blob = bytearray()
        for inst in ue.capture_buffer:
            blob.extend(inst.get_bytes())
        _PROGRAMS.append({
            "section": _CUR[0], "idx": len(_PROGRAMS), "n_inst": n_inst,
            "sha1": hashlib.sha1(bytes(blob)).hexdigest(),
            # Per-instruction digests, so a diff can point at WHICH instruction carries a
            # prompt-dependent operand instead of only reporting that the program as a whole
            # changed. 8 hex chars per instruction keeps the dump small enough to keep around.
            "inst": [hashlib.sha1(bytes(blob[i:i + INSTRUCTION_SIZE_BYTES])).hexdigest()[:8]
                     for i in range(0, len(blob), INSTRUCTION_SIZE_BYTES)],
            "bytes": bytes(blob).hex() if _DUMP_FULL[0] else None,
        })
    t1 = time.perf_counter()
    _run_via_preamble(ue, prog)
    ue.wait_queue(30.0)
    t2 = time.perf_counter()
    ue.clear_capture_buffer()
    if _CUR[0] is not None:
        st = _STATS[_CUR[0]]
        st[0] += (t0 - (_T_CAPTURE[0] or t0)) + (t1 - t0)
        st[1] += (t2 - t1)
        st[2] += 1
        st[3] += n_inst
    _T_CAPTURE[0] = None


def _dump_program_fingerprints():
    """Write per-program (section, index, instruction count, sha1) so two prompt lengths can be
    diffed. A program whose digest matches across lengths is already cacheable as a bin."""
    path = _DUMP_PROGRAMS[0]
    if path is None:
        return
    with open(path, "w") as f:
        json.dump(_PROGRAMS, f, indent=1)
    report(f"[fpga] wrote {len(_PROGRAMS)} program fingerprints to {path}")


def _report_timings():
    if not _STATS:
        return
    report("")
    report(f"{'section':<30}{'compile':>10}{'hw exec':>10}{'runs':>6}{'instructions':>14}")
    report("-" * 70)
    tc = te = 0.0; ti = 0
    for name, (c, e, n, ni) in _STATS.items():
        report(f"{name:<30}{c:>9.3f}s{e:>9.3f}s{n:>6d}{ni:>14,d}")
        tc += c; te += e; ti += ni
    report("-" * 70)
    report(f"{'TOTAL':<30}{tc:>9.3f}s{te:>9.3f}s{'':>6}{ti:>14,d}")
    budget = (KOKORO_TENSOR_END - KOKORO_TENSOR_BASE) / 2**20
    report(f"activation DRAM peak: {_DRAM_HIGH[0] / 2**20:.0f} MB of {budget:.0f} MB "
           f"({100 * _DRAM_HIGH[0] / (budget * 2**20):.0f}%)")



def _round_up(n: int, mult: int) -> int:
    return ((n + mult - 1) // mult) * mult


# --- Constant-upload memoisation --------------------------------------------------------------
#
# _up() is called ~73 times, and the per-layer helpers call it on EVERY invocation: _conv1d
# re-uploads its kernel taps for each of its calls, _adain1d its fc weight and bias. Each call
# bump-allocated a fresh buffer and re-DMA'd bytes that were already in DRAM, so the activation
# region grew with the number of layer calls and a weight's address depended on how many calls
# happened to precede it -- neither of which a frozen instruction bin can tolerate.
#
# Keyed on CONTENT, not identity: the helpers build their operands with .contiguous()/.squeeze(),
# which hands back a fresh Python object (fresh id(), fresh data_ptr()) on every call even though
# the bytes are the model's constant weights. Sharing one buffer between two constants that
# happen to be byte-identical is safe -- every _up() address is used only as a read-only source
# operand (weights, biases, the identity matrix), never as a destination.
#
# Constants live in the PARAMS region, not the tensor region. Each section's forward() ends with
# reset_tensor_dram_addr(), which hands every tensor address back to the bump allocator -- so a
# weight parked there would be overwritten by the next section's activations, and could never be
# dumped as a persistent params.bin. allocate_params_dram is never reset, so a weight uploaded
# once keeps its address for the life of the process and across a cache reload.
#
# Deliberately NOT applied to _up_cap: its callers pass data that genuinely differs from run to
# run (inv_n, the padding mask, the tiled style vector, position/word/type embeddings, the
# alignment matrix), so those bytes have to reach DRAM on every call. _up_cap already gives them
# a prompt-independent ADDRESS by reserving a constant cap_elems, which is the property the
# frozen bin needs; re-DMAing into that fixed address is exactly the wanted behaviour.
_UP_CACHE: "dict" = {}


def _tensor_key(t: torch.Tensor):
    flat = t.detach().reshape(-1).contiguous()
    # bfloat16 has no numpy dtype, but int16 is the same width, so the raw bytes come out intact.
    raw = (flat.view(torch.int16) if flat.dtype == torch.bfloat16 else flat).numpy().tobytes()
    return (str(t.dtype), tuple(t.shape), hashlib.sha1(raw).hexdigest())


def _upload_const(ue, tensor: torch.Tensor) -> int:
    """Upload a constant tensor once for the whole process; later calls get the same address.

    Params DRAM, so this survives every section's reset_tensor_dram_addr() and the region can be
    dumped whole as params.bin.
    """
    key = _tensor_key(tensor)
    a = _UP_CACHE.get(key)
    if a is None and _BIN_LOADED[0]:
        raise RuntimeError(
            f"a tensor of shape {tuple(tensor.shape)} reached the constant memo during replay "
            f"but is not in the frozen image. Constants are all known at compile time, so this is "
            f"a per-run input (prompt/voice data) that must be uploaded with _up_cap instead.")
    if a is None:
        a = ue.allocate_params_dram(tensor.numel() * 2)
        end = ue.get_params_dram_addr()
        if end > KOKORO_TENSOR_BASE:
            raise MemoryError(
                f"params DRAM exhausted: high-water 0x{end:X} passed the activation region at "
                f"0x{KOKORO_TENSOR_BASE:X} "
                f"({(end - KOKORO_PARAMS_BASE) / 2**20:.0f} MB used of "
                f"{(KOKORO_TENSOR_BASE - KOKORO_PARAMS_BASE) / 2**20:.0f} MB)")
        ue.dma_to_accelerator_memory(a, tensor.reshape(-1))
        _UP_CACHE[key] = a
    return a


# Capacity sizing, so DRAM addresses stop moving with the prompt.
#
# allocate_tensor_dram is a bump pointer, so a buffer sized by T_pad shifts EVERY address after it
# -- which is why all 107 captured programs differed between two prompts even where the instruction
# counts were already identical. Sizing at a fixed capacity instead makes the address map constant,
# exactly what llama does with MAX_CONTEXT_SIZE (llama3.2_1b_test.py:459) while carrying the real
# length in GPRs. 64-byte alignment does NOT solve this: it rounds each allocation up, it does not
# make different sizes land on the same addresses.
MAX_T = 512          # phoneme context; PL-BERT's max_position_embeddings
# Ceiling set by the hardware, not by taste: an AdainResBlk1d that upsamples runs its
# InstanceNorm transpose at 2x the frame cap, and bf16_transpose_core_dynamic caps N at 4032
# (eff_z = M_chunk * N/64 must fit a 12-bit URAM_ROW_SIZE_Z field, user_dma_core.py:6687).
# So 2 * F_CAP <= 4032, i.e. MAX_FRAMES <= 2016, rounded down to a multiple of the vector size.
MAX_FRAMES = 1984    # duration-expanded frames (n_frames = pred_dur.sum()); ~49 s of audio
T_CAP = _round_up(MAX_T, UE_VECTOR_SIZE)
F_CAP = _round_up(MAX_FRAMES, UE_VECTOR_SIZE)
_TRANSPOSE_N_MAX = 4032
assert 2 * F_CAP <= _TRANSPOSE_N_MAX, (
    f"2*F_CAP={2 * F_CAP} exceeds the transpose cap {_TRANSPOSE_N_MAX}: an upsampling AdaIN block "
    f"would fail mid-run. Lower MAX_FRAMES.")

# MAX_FRAMES is bounded by the activation region. The default map only gives activations 512 MB
# (0xB0000000..0xD0000000), which caps frames near 512. The board has 4 GiB but the default map
# only ever uses the bottom 2 GiB, so kokoro moves its ACTIVATION region into the untouched upper
# half and leaves the params/program regions exactly where every other model expects them.
#
# This is reachable: the instruction address field is a 35-bit WORD address = 38-bit byte address
# (user_dma_core.py:292), and the host path seeks with os.lseek, whose off_t is 64-bit. The
# "32-bit address" note at user_dma_core.py:8498 constrains only that DRAM-clear helper.
#
# Section 5a is the consumer: ~100 live [rows, 1152] buffers plus the conv weights _conv1d
# re-uploads per call. At the 4096-row cap below that is roughly 1 GiB, inside the 2 GiB region.
# Kokoro FIXED DRAM layout. DRAM is the full 4 GiB at 0x00000000..0x100000000 -- the module-level
# DRAM_START_ADDR (0xB0000000-based) default map only uses the TOP HALF and leaves the bottom 2 GiB
# untouched. Big models already take the whole space this way; see gemma4_e4b_test.py:1015-1026,
# whose weight region starts at 0x00000000.
#
#   params  (weights)     : 0x00000000 - 0x10000000   (256 MB; kokoro is 82M params bf16 ~164 MB)
#   tensor  (activations) : 0x10000000 - 0xF0000000   (3.5 GB)
#   program (instructions): 0xF0000000 - 0x100000000  (256 MB; the image is ~107k x 32 B ~ 3.4 MB)
#
# Do NOT address past 0x100000000: that is the end of physical DRAM, and the XDMA character
# device BLOCKS inside os.write() rather than returning an error, so the process wedges on the
# first upload with the accelerator completely idle.
#
# allocate_tensor_dram has no overflow guard of its own -- gemma4 documents the same hazard at
# gemma4_e4b_test.py:1035-1038 ("silently scribbles into the audio ISA -> corruption / board
# hang"), which is why _instrument() wraps it below.
KOKORO_PARAMS_BASE = 0x00000000
KOKORO_TENSOR_BASE = 0x10000000
KOKORO_PROGRAM_BASE = 0xF0000000
KOKORO_PROGRAM_END = 0x100000000   # end of physical DRAM; see the warning above
KOKORO_TENSOR_END = KOKORO_PROGRAM_BASE

# The row capacity currently in force. Set per section, because the shared helpers (_conv1d,
# _adain1d, _adain_res_blk) are called with PHONEME counts from Sections 2/4 and with FRAME counts
# from Sections 3/5a, and every allocation inside them must be sized for whichever is larger in
# that section. It already includes the x2 from the upsample block, so every buffer in a section
# -- pre- or post-upsample -- fits within one number.
_ROW_CAP = [None]


def _cap() -> int:
    """Row capacity for allocations in the section currently executing."""
    assert _ROW_CAP[0] is not None, "_ROW_CAP not set -- section forgot to declare its row capacity"
    return _ROW_CAP[0]


def _set_row_cap(actual_rows: int, cap_rows: int, what: str):
    """Declare the row capacity for the section about to run, and fail loudly rather than
    silently corrupting: an over-long input would under-size every buffer and write past it."""
    if actual_rows > cap_rows:
        raise ValueError(
            f"{what}={actual_rows} exceeds the compiled capacity of {cap_rows} rows. The "
            f"instruction image is frozen against a fixed DRAM map, so a longer input cannot "
            f"reuse it -- shorten or chunk the input, or raise MAX_T / MAX_FRAMES, which needs "
            f"the 512 MB activation budget re-checked (see the note above).")
    _ROW_CAP[0] = cap_rows


def _bf16(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.bfloat16).contiguous()


class PLBertFPGA:
    """Section 1: PL-BERT (Albert) encoder + bert_encoder projection, run on the
    UnifiedEngine accelerator.
    """

    NEG_INF = -1.0e9  # masking value for padded key positions in the attention bias

    def __init__(self, model, ue: UnifiedEngine):
        self.model = model
        self.ue = ue
        albert = model.bert.albert
        cfg = albert.config
        sd = albert.state_dict()

        self.E = cfg.embedding_size          # 128
        self.H = cfg.hidden_size             # 768
        self.NH = cfg.num_attention_heads    # 12
        self.HD = self.H // self.NH          # 64
        self.NL = cfg.num_hidden_layers      # 12
        self.FFN = cfg.intermediate_size     # 2048

        g = lambda k: _bf16(sd[k])
        self.word_emb = g("embeddings.word_embeddings.weight")           # [178, 128]
        self.pos_emb = g("embeddings.position_embeddings.weight")        # [512, 128]
        self.type_emb0 = g("embeddings.token_type_embeddings.weight")[0]  # [128] (token_type_id always 0)
        self.emb_ln_w = g("embeddings.LayerNorm.weight")
        self.emb_ln_b = g("embeddings.LayerNorm.bias")
        self.map_in_w = g("encoder.embedding_hidden_mapping_in.weight")  # [768, 128]
        self.map_in_b = g("encoder.embedding_hidden_mapping_in.bias")

        p = "encoder.albert_layer_groups.0.albert_layers.0."
        self.q_w, self.q_b = g(p + "attention.query.weight"), g(p + "attention.query.bias")
        self.k_w, self.k_b = g(p + "attention.key.weight"), g(p + "attention.key.bias")
        self.v_w, self.v_b = g(p + "attention.value.weight"), g(p + "attention.value.bias")
        self.attn_out_w, self.attn_out_b = g(p + "attention.dense.weight"), g(p + "attention.dense.bias")
        self.attn_ln_w, self.attn_ln_b = g(p + "attention.LayerNorm.weight"), g(p + "attention.LayerNorm.bias")
        self.ffn_w, self.ffn_b = g(p + "ffn.weight"), g(p + "ffn.bias")
        self.ffn_out_w, self.ffn_out_b = g(p + "ffn_output.weight"), g(p + "ffn_output.bias")
        self.out_ln_w, self.out_ln_b = g(p + "full_layer_layer_norm.weight"), g(p + "full_layer_layer_norm.bias")

        self.bert_encoder_w = _bf16(model.bert_encoder.weight)  # [512, 768]
        self.bert_encoder_b = _bf16(model.bert_encoder.bias)

    def _up(self, tensor: torch.Tensor) -> int:
        """Constant weights/biases only -- memoised, see _upload_const. Anything whose CONTENT
        changes from run to run must go through _up_cap so it is re-DMA'd every call."""
        return _upload_const(self.ue, tensor)

    def _up_cap(self, tensor: torch.Tensor, cap_elems: int) -> int:
        """Upload `tensor` but RESERVE cap_elems, so the allocator advances by a constant amount.
        The tensor's own layout is untouched -- only the address becomes prompt-independent."""
        assert tensor.numel() <= cap_elems, (tensor.numel(), cap_elems)
        a = self.ue.allocate_tensor_dram(cap_elems * 2)
        self.ue.dma_to_accelerator_memory(a, tensor.reshape(-1))
        return a

    def _linear(self, x_dram, M, K, N, w_dram, b_dram, out_dram, gelu=False):
        return _dyn_matmul(
            self.ue,
            M=M, K=K, N=N, A_DRAM_ADDR=x_dram, B_DRAM_ADDR=w_dram, OUTPUT_DRAM_ADDR=out_dram,
            C_DRAM_ADDR=b_dram, bias_mode="broadcast_N", gelu_enable=gelu,
        )

    _GELU_NEW_C0 = 0.7978845608028654  # sqrt(2/pi)
    _GELU_NEW_C1 = 0.044715

    def _gelu_new(self, x_dram, numel, identity_dram, tmp1, tmp2, tmp3, out_dram):
        """Exact HF `gelu_new` (tanh-approximation GELU), matching Albert's actual configured
        hidden_act -- NOT matmat_mul_core's fused gelu_enable epilogue, which computes a different
        (sigmoid-based) approximation and was the source of a ~15dB SNR loss compounding across
        Kokoro's 12 shared transformer layers.

        gelu_new(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))

        tanh is derived EXACTLY from the native sigmoid activation via tanh(x) = 2*sigmoid(2x) - 1
        (the same identity validated at 40+dB in the LSTM hardware proof) -- built entirely from
        eltwise_core_dram (mul/add/broadcast) + activation_core(sigmoid), no new hardware op.
        numel must be a multiple of UE_VECTOR_SIZE; identity_dram must be a UE_VECTOR_SIZE-square
        bf16 identity matrix.
        """
        ue = self.ue
        M, N = numel // UE_VECTOR_SIZE, UE_VECTOR_SIZE
        mul = lambda a, b, o: _dyn_eltwise(ue, M, N, a, b, o, mode=UE_MODE.ELTWISE_MUL)
        add = lambda a, b, o: _dyn_eltwise(ue, M, N, a, b, o, mode=UE_MODE.ELTWISE_ADD)
        mulb = lambda a, s, o: _dyn_eltwise(ue, M, N, a, None, o, mode=UE_MODE.MUL_BROADCAST, scalar=s)
        addb = lambda a, s, o: _dyn_eltwise(ue, M, N, a, None, o, mode=UE_MODE.ADD_BROADCAST, scalar=s)

        mul(x_dram, x_dram, tmp1)                       # tmp1 = x^2
        mul(tmp1, x_dram, tmp1)                          # tmp1 = x^3
        mulb(tmp1, self._GELU_NEW_C1, tmp1)               # tmp1 = 0.044715*x^3
        add(x_dram, tmp1, tmp1)                           # tmp1 = x + 0.044715*x^3
        mulb(tmp1, self._GELU_NEW_C0, tmp1)               # tmp1 = u = sqrt(2/pi)*(...)
        mulb(tmp1, 2.0, tmp2)                             # tmp2 = 2u
        _dyn_activation(ue, M=M, N=N, A_DRAM_ADDR=tmp2, OUTPUT_DRAM_ADDR=tmp2,
                            IDENTITY_DRAM_ADDR=identity_dram, activation="sigmoid")
        mulb(tmp2, 2.0, tmp2)                             # tmp2 = 2*sigmoid(2u)
        addb(tmp2, -1.0, tmp2)                            # tmp2 = tanh(u) = 2*sigmoid(2u) - 1
        mul(x_dram, tmp2, tmp3)                           # tmp3 = x*tanh(u)
        add(x_dram, tmp3, tmp3)                           # tmp3 = x + x*tanh(u) = x*(1+tanh(u))
        mulb(tmp3, 0.5, out_dram)                         # out = 0.5*x*(1+tanh(u))

    def _run(self, ue):
        _timed_run(ue)

    def forward(self, input_ids: torch.LongTensor, debug_cpu_ref=None) -> torch.Tensor:
        """input_ids: LongTensor [T]. Returns d_en: FloatTensor [512, T], matching
        KokoroModel.forward_with_tokens's ``d_en = bert_encoder(bert(...)).transpose(-1,-2)``.
        """
        ue = self.ue
        T = input_ids.shape[0]
        T_pad = _round_up(T, UE_VECTOR_SIZE)
        _set_row_cap(T_pad, T_CAP, "T_pad")
        T_pad = Dim(T_pad, GPR_TPAD, cap=T_CAP)
        set_runtime_dims(T_pad=int(T_pad))
        E, H, NH, HD, FFN = self.E, self.H, self.NH, self.HD, self.FFN
        # Allocate at capacity, not at T_pad, so the DRAM address map is identical for every
        # prompt. T_pad still drives the op DIMENSIONS; only the sizes change here.
        TC = T_CAP

        # ---- upload weights (once per call; bin-dump caching is a later optimization) ----
        w_map_in, b_map_in = self._up(self.map_in_w), self._up(self.map_in_b)
        w_emb_ln, b_emb_ln = self._up(self.emb_ln_w), self._up(self.emb_ln_b)
        w_q, b_q = self._up(self.q_w), self._up(self.q_b)
        w_k, b_k = self._up(self.k_w), self._up(self.k_b)
        w_v, b_v = self._up(self.v_w), self._up(self.v_b)
        w_attn_out, b_attn_out = self._up(self.attn_out_w), self._up(self.attn_out_b)
        w_attn_ln, b_attn_ln = self._up(self.attn_ln_w), self._up(self.attn_ln_b)
        w_ffn, b_ffn = self._up(self.ffn_w), self._up(self.ffn_b)
        w_ffn_out, b_ffn_out = self._up(self.ffn_out_w), self._up(self.ffn_out_b)
        w_out_ln, b_out_ln = self._up(self.out_ln_w), self._up(self.out_ln_b)
        w_bert_enc, b_bert_enc = self._up(self.bert_encoder_w), self._up(self.bert_encoder_b)
        identity_dram = self._up(torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))

        key_mask_row = torch.zeros(T_pad, dtype=torch.float32)
        key_mask_row[T:] = self.NEG_INF
        attn_bias = key_mask_row.unsqueeze(0).expand(T_pad, T_pad).contiguous().to(torch.bfloat16)
        bias_dram = self._up_cap(attn_bias, TC * TC)

        # ---- embeddings: word (gathered HOST-side), position (contiguous rows 0..T_pad-1 of the
        #      table), token_type (row 0 broadcast to every position) ----
        #
        # The word gather used to be a captured row-copy per token, which was the last emission in
        # this section that scaled with T (2 instructions x T). The token ids are known at capture
        # time, so the identical result comes from indexing the table on the host and uploading the
        # [T_pad, E] slice directly: ZERO instructions, and the full [n_token, E] table no longer
        # has to be uploaded or occupy DRAM at all. Padding rows reuse row 0 exactly as before --
        # their content is irrelevant, the attention bias masks those positions out and only the
        # first T output rows are ever read back.
        ids_padded = torch.zeros(T_pad, dtype=torch.long)
        ids_padded[:T] = input_ids[:T].reshape(-1).long()
        word_dram = self._up_cap(self.word_emb[ids_padded].contiguous(), TC * E)

        pos_dram = self._up_cap(self.pos_emb[:T_pad], TC * E)
        type_dram = self._up_cap(self.type_emb0.unsqueeze(0).expand(T_pad, E).contiguous(), TC * E)

        emb_sum_dram = ue.allocate_tensor_dram(TC * E * 2)
        emb_ln_dram = ue.allocate_tensor_dram(TC * E * 2)
        hidden_dram = ue.allocate_tensor_dram(TC * H * 2)

        q_dram = ue.allocate_tensor_dram(TC * H * 2)
        k_dram = ue.allocate_tensor_dram(TC * H * 2)
        v_dram = ue.allocate_tensor_dram(TC * H * 2)
        q_heads_dram = ue.allocate_tensor_dram(TC * H * 2)
        k_heads_dram = ue.allocate_tensor_dram(TC * H * 2)
        v_heads_dram = ue.allocate_tensor_dram(TC * H * 2)
        attn_heads_out_dram = ue.allocate_tensor_dram(TC * H * 2)
        attn_merged_dram = ue.allocate_tensor_dram(TC * H * 2)
        attn_proj_dram = ue.allocate_tensor_dram(TC * H * 2)
        resid1_dram = ue.allocate_tensor_dram(TC * H * 2)
        attn_ln_dram = ue.allocate_tensor_dram(TC * H * 2)
        ffn_mid_dram = ue.allocate_tensor_dram(TC * FFN * 2)
        gelu_tmp1_dram = ue.allocate_tensor_dram(TC * FFN * 2)
        gelu_tmp2_dram = ue.allocate_tensor_dram(TC * FFN * 2)
        gelu_tmp3_dram = ue.allocate_tensor_dram(TC * FFN * 2)
        ffn_out_dram = ue.allocate_tensor_dram(TC * H * 2)
        resid2_dram = ue.allocate_tensor_dram(TC * H * 2)
        scratch_dram = ue.allocate_tensor_dram((HD + TC) * TC * 2 + TC * HD * 2)
        d_en_dram = ue.allocate_tensor_dram(TC * 512 * 2)

        from user_dma_core import calculate_snr

        # Buffer bisect results instead of printing inline -- each stage's capture/execute emits a
        # wall of M_chunk/URAM-usage compile logs that would otherwise bury the SNR lines between
        # every layer. Printed as one clean table at the end of forward().
        debug_log = []

        def _check(stage_name, dram_addr, shape, cpu_ref):
            if debug_cpu_ref is None:
                return
            _debug_flush(ue, self._run)
            got = ue.dma_from_accelerator_memory(dram_addr, shape)[:T].float()
            if isinstance(cpu_ref, (tuple, list)):
                cpu_ref = cpu_ref[0]          # HF AlbertLayer returns (hidden_states, ...)
            snr_db = calculate_snr(cpu_ref.detach().float().reshape(-1), got.reshape(-1))
            debug_log.append((stage_name, snr_db))

        # --- Embedding stage (own capture/execute so we can read back + compare before continuing) ---
        ue.start_capture()
        _dyn_eltwise(ue, T_pad, E, word_dram, pos_dram, emb_sum_dram, mode=UE_MODE.ELTWISE_ADD)
        _dyn_eltwise(ue, T_pad, E, emb_sum_dram, type_dram, emb_sum_dram, mode=UE_MODE.ELTWISE_ADD)
        _dyn_layernorm(ue, M=T_pad, N=E, A_DRAM_ADDR=emb_sum_dram, OUTPUT_DRAM_ADDR=emb_ln_dram,
                                 GAMMA_DRAM_ADDR=w_emb_ln, BETA_DRAM_ADDR=b_emb_ln)
        self._linear(emb_ln_dram, T_pad, E, H, w_map_in, b_map_in, hidden_dram)
        self._run(ue)

        if debug_cpu_ref is not None:
            emb_cpu = debug_cpu_ref["embeddings"](input_ids=input_ids.unsqueeze(0))
            hidden_cpu = debug_cpu_ref["map_in"](emb_cpu)  # [1, T, H] -- keep the batch dim; AlbertLayer requires 3D input
            _check("embedding+map_in", hidden_dram, (T_pad, H), hidden_cpu)

        for layer_idx in range(self.NL):
            ue.start_capture()
            self._linear(hidden_dram, T_pad, H, H, w_q, b_q, q_dram)
            self._linear(hidden_dram, T_pad, H, H, w_k, b_k, k_dram)
            self._linear(hidden_dram, T_pad, H, H, w_v, b_v, v_dram)

            # Interleaved [T_pad, NH, HD] -> grouped [NH, T_pad, HD] (per-head contiguous), via
            # native strided DMA -- no per-row Python loop needed.
            # The per-head LAYOUT is pinned to the cap, not to this prompt's T_pad. Each head's
            # block then begins at a fixed byte offset, so the 144 attention calls below carry
            # identical address operands for every prompt -- the buffers are already TC*H*2, so
            # a cap stride fits exactly. Attention still touches only T_pad rows per head,
            # because its trip count comes from the register, not from this stride.
            ue.bf16_permute_dram_core(num_groups=NH, group_rows=TC, row_width=HD,
                                       in_dram=q_dram, out_dram=q_heads_dram, write_grouped=True)
            ue.bf16_permute_dram_core(num_groups=NH, group_rows=TC, row_width=HD,
                                       in_dram=k_dram, out_dram=k_heads_dram, write_grouped=True)
            ue.bf16_permute_dram_core(num_groups=NH, group_rows=TC, row_width=HD,
                                       in_dram=v_dram, out_dram=v_heads_dram, write_grouped=True)

            head_stride = TC * HD * 2
            for h in range(NH):
                off = h * head_stride
                _dyn_attention(ue, 
                    batch=T_pad, aligned_seq_len=T_pad, head_dim=HD,
                    Q_DRAM_ADDR=q_heads_dram + off, K_DRAM_ADDR=k_heads_dram + off, V_DRAM_ADDR=v_heads_dram + off,
                    BIAS_DRAM_ADDR=bias_dram, OUTPUT_DRAM_ADDR=attn_heads_out_dram + off,
                    SCRATCH_DRAM_ADDR=scratch_dram, IDENTITY_DRAM_ADDR=identity_dram,
                )

            # Grouped [NH, T_pad, HD] (attention's per-head output) -> interleaved [T_pad, NH, HD]
            # == [T_pad, H] (write_grouped=False: in=grouped -> out=interleaved).
            ue.bf16_permute_dram_core(num_groups=NH, group_rows=TC, row_width=HD,
                                       in_dram=attn_heads_out_dram, out_dram=attn_merged_dram,
                                       write_grouped=False)

            self._linear(attn_merged_dram, T_pad, H, H, w_attn_out, b_attn_out, attn_proj_dram)
            _dyn_eltwise(ue, T_pad, H, hidden_dram, attn_proj_dram, resid1_dram, mode=UE_MODE.ELTWISE_ADD)
            _dyn_layernorm(ue, M=T_pad, N=H, A_DRAM_ADDR=resid1_dram, OUTPUT_DRAM_ADDR=attn_ln_dram,
                                     GAMMA_DRAM_ADDR=w_attn_ln, BETA_DRAM_ADDR=b_attn_ln)

            self._linear(attn_ln_dram, T_pad, H, FFN, w_ffn, b_ffn, ffn_mid_dram, gelu=False)
            self._gelu_new(ffn_mid_dram, T_pad * FFN, identity_dram,
                            gelu_tmp1_dram, gelu_tmp2_dram, gelu_tmp3_dram, ffn_mid_dram)
            self._linear(ffn_mid_dram, T_pad, FFN, H, w_ffn_out, b_ffn_out, ffn_out_dram)
            _dyn_eltwise(ue, T_pad, H, attn_ln_dram, ffn_out_dram, resid2_dram, mode=UE_MODE.ELTWISE_ADD)
            _dyn_layernorm(ue, M=T_pad, N=H, A_DRAM_ADDR=resid2_dram, OUTPUT_DRAM_ADDR=hidden_dram,
                                     GAMMA_DRAM_ADDR=w_out_ln, BETA_DRAM_ADDR=b_out_ln)
            self._run(ue)

            if debug_cpu_ref is not None:
                # Bisect within the layer: Q/K/V projections -> attention block (attn + dense +
                # residual + LN) -> full layer (+ FFN + residual + LN). Each stage is checked
                # against the SAME hidden_cpu input this hardware iteration used, so whichever
                # stage's SNR craters first is exactly where the bug lives.
                attn_mod = debug_cpu_ref["layer"].attention
                q_cpu = attn_mod.query(hidden_cpu)
                k_cpu = attn_mod.key(hidden_cpu)
                v_cpu = attn_mod.value(hidden_cpu)
                _check(f"layer {layer_idx} Q proj", q_dram, (T_pad, H), q_cpu)
                _check(f"layer {layer_idx} K proj", k_dram, (T_pad, H), k_cpu)
                _check(f"layer {layer_idx} V proj", v_dram, (T_pad, H), v_cpu)

                attn_block_cpu = attn_mod(hidden_cpu, attention_mask=debug_cpu_ref["ext_mask"])[0]
                _check(f"layer {layer_idx} attn block (post-LN)", attn_ln_dram, (T_pad, H), attn_block_cpu)

                # AlbertLayer.forward returns hidden_states directly (a plain Tensor, despite its
                # tuple[...] type hint) in this transformers version -- confirmed via direct
                # inspection, so no [0]/tuple-unpack here.
                hidden_cpu = debug_cpu_ref["layer"](hidden_cpu, attention_mask=debug_cpu_ref["ext_mask"])
                if isinstance(hidden_cpu, (tuple, list)):
                    hidden_cpu = hidden_cpu[0]      # newer transformers return (hidden_states, ...)
                _check(f"layer {layer_idx} full (post-FFN)", hidden_dram, (T_pad, H), hidden_cpu)

        ue.start_capture()
        self._linear(hidden_dram, T_pad, H, 512, w_bert_enc, b_bert_enc, d_en_dram)
        self._run(ue)
        ue.report_timing_and_instruction_count()

        d_en = ue.dma_from_accelerator_memory(d_en_dram, (T_pad, 512))[:T]  # [T, 512]
        ue.reset_tensor_dram_addr()

        if debug_log:
            report_snr("\n[fpga][debug] --- Section 1 bisect (SNR vs CPU, most-recent layer last) ---")
            for stage_name, snr_db in debug_log:
                flag = "  <-- LOW" if snr_db < 30.0 else ""
                report_snr(f"[fpga][debug] {stage_name:38s} {snr_db:8.2f} dB{flag}")
            report_snr("[fpga][debug] --- end bisect ---\n")

        return d_en.float().T  # [512, T]


class ProsodyDurationFPGA:
    """Section 2: the prosody/duration path -- DurationEncoder (3x BiLSTM + AdaLayerNorm) +
    predictor.lstm + duration_proj -> per-phoneme durations -> the frame alignment matrix.

    Confirmed from the checkpoint's actual weight shapes (not just source inspection):
    - Every one of the 4 BiLSTM instances (3 inside DurationEncoder + predictor.lstm) takes a
      640-wide input (weight_ih_l0 is [1024,640]): d_model(512) + style_dim(128) concatenated,
      because DurationEncoder re-concatenates style after EVERY AdaLayerNorm block, including the
      last -- so its returned `d` is itself 640-wide, matching predictor.lstm's expected input.
    - hidden=256 per direction (weight_hh_l0 is [1024,256], 4*256=1024 for the 4 gates), bf16
      throughout, no padding-to-64 needed anywhere here: LSTM steps are all M=1 matmuls (K/N
      already 64-aligned: 640, 256, 1024), so T itself never needs rounding the way the T_pad
      attention/matmul tiling in Section 1 did.

    nn.Linear/nn.LSTM weight layout already matches matmat_mul_core's B=[N,K] convention (see
    Section 1's header note), so every weight here is uploaded as-is, no transpose needed.
    """

    def __init__(self, model, ue: UnifiedEngine):
        self.model = model
        self.ue = ue
        # NOT reused from Section 1: PLBertFPGA.forward() calls reset_tensor_dram_addr() at the end
        # of its own run, which invalidates every DRAM address it allocated. Uploaded fresh in
        # forward() below (after this constructor's weight uploads, which happen before any capture).
        self.identity_dram = None
        self.style_dim = 128
        self.C = 512   # d_model
        self.H = 256   # per-direction LSTM hidden

        pred = model.predictor
        te = pred.text_encoder

        def lstm_w(m):
            return dict(
                Wx_f=_bf16(m.weight_ih_l0), Wh_f=_bf16(m.weight_hh_l0),
                b_f=_bf16(m.bias_ih_l0 + m.bias_hh_l0),
                Wx_b=_bf16(m.weight_ih_l0_reverse), Wh_b=_bf16(m.weight_hh_l0_reverse),
                b_b=_bf16(m.bias_ih_l0_reverse + m.bias_hh_l0_reverse),
            )

        def adaln_w(m):
            return dict(fc_w=_bf16(m.fc.weight), fc_b=_bf16(m.fc.bias))  # [1024,128], [1024]

        self.lstm_blocks = [lstm_w(te.lstms[0]), lstm_w(te.lstms[2]), lstm_w(te.lstms[4])]
        self.adaln_blocks = [adaln_w(te.lstms[1]), adaln_w(te.lstms[3]), adaln_w(te.lstms[5])]
        self.pred_lstm_w = lstm_w(pred.lstm)

        # duration_proj's output width is max_dur=50, NOT a multiple of UE_VECTOR_SIZE(64) -- unlike
        # every other matmul dim in this pipeline. Pad N to 64 with zero rows/entries (the padding
        # columns' garbage never gets read back; we always slice to :50 after readback) rather than
        # run matmat_mul_core with an unaligned N, which produced -inf dB (likely NaN/Inf from
        # unwritten padding in the hardware's internal N-chunk tiling).
        self.dur_n = 50
        self.dur_n_pad = _round_up(self.dur_n, UE_VECTOR_SIZE)
        dur_w_raw = _bf16(pred.duration_proj.linear_layer.weight)  # [50, 512]
        dur_b_raw = _bf16(pred.duration_proj.linear_layer.bias)    # [50]
        self.dur_w = torch.zeros(self.dur_n_pad, self.C, dtype=torch.bfloat16)
        self.dur_w[:self.dur_n] = dur_w_raw
        self.dur_b = torch.zeros(self.dur_n_pad, dtype=torch.bfloat16)
        self.dur_b[:self.dur_n] = dur_b_raw

    def _up(self, tensor: torch.Tensor) -> int:
        """Constant weights/biases only -- memoised, see _upload_const. Anything whose CONTENT
        changes from run to run must go through _up_cap so it is re-DMA'd every call."""
        return _upload_const(self.ue, tensor)

    def _up_cap(self, tensor: torch.Tensor, cap_elems: int) -> int:
        """Upload `tensor` but RESERVE cap_elems, so the allocator advances by a constant amount.
        The tensor's own layout is untouched -- only the address becomes prompt-independent."""
        assert tensor.numel() <= cap_elems, (tensor.numel(), cap_elems)
        a = self.ue.allocate_tensor_dram(cap_elems * 2)
        self.ue.dma_to_accelerator_memory(a, tensor.reshape(-1))
        return a

    def _run(self, ue):
        """Flush the current captured instruction stream to the device and execute it. dma_to/from_
        accelerator_memory are direct host<->device PCIe DMA calls (confirmed via user_dma_core.py:
        dma_write/dma_read on the raw xdma char devices) -- completely separate from this
        capture/execute queue mechanism, so _up() weight uploads are safe to interleave anywhere.
        Only the compute calls (matmat_mul_core, eltwise_core_dram, etc.) need this bracket.
        """
        _timed_run(ue)

    def _lstm_bidir(self, x_dram, T, Cin, w, out_dram):
        """Runs both directions of one BiLSTM over T (unrolled at capture time, same per-timestep
        decomposition validated at 40.5dB in the standalone LSTM hardware proof: batched gate
        matmul + native sigmoid + tanh(x)=2*sigmoid(2x)-1 + elementwise cell/hidden update).
        Writes directly into out_dram's [T, 2H] layout -- forward half at column 0, backward half
        at column H -- since every per-timestep write is M=1 (a single [1,H] vector), the
        destination address can be any byte offset, no separate merge/permute step needed.
        """
        ue, H = self.ue, self.H
        G = 4 * H

        def run_direction(reverse, Wx, Wh, b, col_off_bytes):
            wx_dram, wh_dram, b_dram = self._up(Wx), self._up(Wh), self._up(b)
            gates_x = ue.allocate_tensor_dram(G * 2)
            gates_h = ue.allocate_tensor_dram(G * 2)
            gates = ue.allocate_tensor_dram(G * 2)
            i_d, f_d, g_d, o_d = (ue.allocate_tensor_dram(H * 2) for _ in range(4))
            g_tmp, c_tmp, tanh_c, fc_, ig_ = (ue.allocate_tensor_dram(H * 2) for _ in range(5))
            h_bufs = [ue.allocate_tensor_dram(H * 2), ue.allocate_tensor_dram(H * 2)]
            c_bufs = [ue.allocate_tensor_dram(H * 2), ue.allocate_tensor_dram(H * 2)]
            ue.dma_to_accelerator_memory(h_bufs[0], torch.zeros(H, dtype=torch.bfloat16))
            ue.dma_to_accelerator_memory(c_bufs[0], torch.zeros(H, dtype=torch.bfloat16))

            M_H = H // UE_VECTOR_SIZE

            def sigmoid_ip(src, dst):
                _dyn_activation(ue, M=M_H, N=UE_VECTOR_SIZE, A_DRAM_ADDR=src, OUTPUT_DRAM_ADDR=dst,
                                    IDENTITY_DRAM_ADDR=self.identity_dram, activation="sigmoid")

            def tanh_via_sigmoid(src, tmp, dst):
                _dyn_eltwise(ue, 1, H, src, None, tmp, mode=UE_MODE.MUL_BROADCAST, scalar=2.0)
                sigmoid_ip(tmp, tmp)
                _dyn_eltwise(ue, 1, H, tmp, None, tmp, mode=UE_MODE.MUL_BROADCAST, scalar=2.0)
                _dyn_eltwise(ue, 1, H, tmp, None, dst, mode=UE_MODE.ADD_BROADCAST, scalar=-1.0)

            # ONE hardware loop instead of T Python-unrolled bodies. The recurrence is still
            # strictly sequential -- this changes only how the instruction stream is EMITTED, not
            # the order the accelerator executes in.
            #
            # Two things varied per timestep and both are now register-computed:
            #   * the input row x[t], gathered into a fixed scratch buffer;
            #   * the output row out[t], scattered from a fixed buffer.
            # Everything between keeps literal addresses, so the body is the same op sequence as
            # before; only the gather/scatter use general_reg_src. That avoids forcing every M=1
            # matmul onto the dynamic path just to source an address.
            #
            # h/c became SINGLE in-place buffers (they used to ping-pong per timestep, which a loop
            # body cannot express). Safe: gates_h reads h before h is rewritten at the end of the
            # body, and fc_ consumes c before c is rewritten -- every read precedes its write.
            x_scratch = ue.allocate_tensor_dram(Cin * 2)
            h_buf, c_buf = h_bufs[0], c_bufs[0]
            i_reg = ue.alloc_isa_reg()
            a_reg = ue.alloc_isa_reg()
            if isinstance(T, Dim) and T.base is not None:
                # i starts at T-1 when walking backwards, so it too must come from the register.
                ue.generate_instruction_add_imm(_emit_M(ue, T), -1, i_reg) if reverse \
                    else ue.generate_instruction_add_set(i_reg, 0)
                ue.loop_start(loop_cnt=_template(T), gpr_loop_cnt=_emit_M(ue, T))
            else:
                ue.generate_instruction_add_set(i_reg, (int(T) - 1) if reverse else 0)
                ue.loop_start(loop_cnt=int(T))

            ue.generate_instruction_reg_mul_imm(a_reg, i_reg, ue_35bit_addr_shifter(Cin * 2))
            ue.generate_instruction_add_imm(a_reg, ue_35bit_addr_shifter(x_dram), a_reg)
            ue.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=0x00000,
                                          element_size=Cin, general_reg_src=a_reg)
            ue.sram_to_accelerator_memory(sram_address=0x00000,
                                          accelerator_dram_address=x_scratch, element_size=Cin)

            _dyn_matmul(ue, M=1, K=Cin, N=G, A_DRAM_ADDR=x_scratch, B_DRAM_ADDR=wx_dram,
                        OUTPUT_DRAM_ADDR=gates_x, C_DRAM_ADDR=b_dram, bias_mode="broadcast_N")
            _dyn_matmul(ue, M=1, K=H, N=G, A_DRAM_ADDR=h_buf, B_DRAM_ADDR=wh_dram,
                        OUTPUT_DRAM_ADDR=gates_h)
            _dyn_eltwise(ue, 1, G, gates_x, gates_h, gates, mode=UE_MODE.ELTWISE_ADD)

            i_raw, f_raw, g_raw, o_raw = (gates + k * H * 2 for k in range(4))
            sigmoid_ip(i_raw, i_d); sigmoid_ip(f_raw, f_d); sigmoid_ip(o_raw, o_d)
            tanh_via_sigmoid(g_raw, g_tmp, g_d)

            _dyn_eltwise(ue, 1, H, f_d, c_buf, fc_, mode=UE_MODE.ELTWISE_MUL)
            _dyn_eltwise(ue, 1, H, i_d, g_d, ig_, mode=UE_MODE.ELTWISE_MUL)
            _dyn_eltwise(ue, 1, H, fc_, ig_, c_buf, mode=UE_MODE.ELTWISE_ADD)

            tanh_via_sigmoid(c_buf, c_tmp, tanh_c)
            _dyn_eltwise(ue, 1, H, o_d, tanh_c, h_buf, mode=UE_MODE.ELTWISE_MUL)

            ue.accelerator_memory_to_sram(accelerator_dram_address=h_buf, sram_address=0x00000,
                                          element_size=H)
            ue.generate_instruction_reg_mul_imm(a_reg, i_reg, ue_35bit_addr_shifter((2 * H) * 2))
            ue.generate_instruction_add_imm(a_reg, ue_35bit_addr_shifter(out_dram + col_off_bytes), a_reg)
            ue.sram_to_accelerator_memory(sram_address=0x00000, accelerator_dram_address=0,
                                          element_size=H, general_reg_src=a_reg)

            if reverse:
                ue.generate_instruction_add_dec(i_reg)
            else:
                ue.generate_instruction_add_inc(i_reg)
            ue.loop_end()
            ue.release_isa_reg()   # a_reg
            ue.release_isa_reg()   # i_reg

        run_direction(False, w["Wx_f"], w["Wh_f"], w["b_f"], 0)
        run_direction(True, w["Wx_b"], w["Wh_b"], w["b_b"], H * 2)

    def _concat_columns(self, a_dram, a_width, b_dram, b_width, T, out_dram, gpr_rows=None):
        """out[T, a_width+b_width] = concat([a[T,a_width], b[T,b_width]], dim=-1), via per-row
        copies (T is small here, <200 typically -- same technique as Section 1's embedding gather).
        """
        ue = self.ue
        out_w_bytes = (a_width + b_width) * 2
        a_row_bytes, b_row_bytes = a_width * 2, b_width * 2
        # Both halves staged in the same iteration, at distinct SRAM offsets so one loop body
        # covers the read-read-write-write sequence the unrolled version did per row.
        _pbi_row_loop(ue, T,
                      reads=[(a_dram, a_row_bytes, a_width, 0),
                             (b_dram, b_row_bytes, b_width, a_row_bytes)],
                      writes=[(out_dram, out_w_bytes, a_width, 0),
                              (out_dram + a_row_bytes, out_w_bytes, b_width, a_row_bytes)],
                      gpr_rows=gpr_rows)

    def _adaln(self, x_dram, T, C, style_dram, w, out_dram):
        """AdaLayerNorm: plain (no-affine) LayerNorm over the channel axis, then a style-conditioned
        affine (1+gamma)*x_hat + beta where [gamma;beta] = style @ fc_w^T + fc_b.

        gamma/beta are the SAME for every T position (style does not vary with time), so they are
        a [1, C] row that has to be applied down T rows. There is no per-channel-vector broadcast
        primitive, but a row loop with a ZERO source stride is one: it re-reads the same row every
        iteration and writes it to a walking destination. That keeps the whole thing on device --
        no readback, no host tiling, and no capture split.
        """
        ue = self.ue
        gb_dram = ue.allocate_tensor_dram(2 * C * 2)
        normed_dram = ue.allocate_tensor_dram(_cap() * C * 2)
        ng_dram = ue.allocate_tensor_dram(_cap() * C * 2)
        tmp_dram = ue.allocate_tensor_dram(_cap() * C * 2)
        gamma_tiled = ue.allocate_tensor_dram(_cap() * C * 2)
        beta_tiled = ue.allocate_tensor_dram(_cap() * C * 2)
        row_b = C * 2

        _dyn_matmul(ue, M=1, K=self.style_dim, N=2 * C, A_DRAM_ADDR=style_dram, B_DRAM_ADDR=self._up(w["fc_w"]),
                            OUTPUT_DRAM_ADDR=gb_dram, C_DRAM_ADDR=self._up(w["fc_b"]), bias_mode="broadcast_N")
        # [1, C] -> [T, C], on device. Source stride 0 = broadcast the single row.
        _pbi_row_loop(ue, T, reads=[(gb_dram, 0, C, 0)],
                      writes=[(gamma_tiled, row_b, C, 0)])
        _pbi_row_loop(ue, T, reads=[(gb_dram + row_b, 0, C, 0)],
                      writes=[(beta_tiled, row_b, C, 0)])

        _dyn_layernorm(ue, M=T, N=C, A_DRAM_ADDR=x_dram, OUTPUT_DRAM_ADDR=normed_dram)
        _dyn_eltwise(ue, T, C, normed_dram, gamma_tiled, ng_dram, mode=UE_MODE.ELTWISE_MUL)   # normed*gamma
        _dyn_eltwise(ue, T, C, normed_dram, ng_dram, tmp_dram, mode=UE_MODE.ELTWISE_ADD)      # normed*(1+gamma)
        _dyn_eltwise(ue, T, C, tmp_dram, beta_tiled, out_dram, mode=UE_MODE.ELTWISE_ADD)      # + beta

    def forward(self, d_en: torch.Tensor, style_vec: torch.Tensor, T: int, debug_cpu_ref=None):
        """d_en: [512, T] (Section 1's output). style_vec: [128] (ref_s[:,128:] squeezed).
        Returns (d [T,640], pred_dur [T] LongTensor) matching KokoroModel.forward_with_tokens's
        `d` and `pred_dur` at this point in the pipeline.
        """
        ue = self.ue
        C, style_dim = self.C, self.style_dim
        _set_row_cap(T, T_CAP, "T")
        # From here on T carries its own provenance: every kernel row count derived from it
        # becomes a register reference instead of a baked immediate, so this section's captured
        # programs stop depending on the phoneme count. Host-side arithmetic is unaffected --
        # Dim is an int.
        T = Dim(T, GPR_T, cap=T_CAP)
        set_runtime_dims(T=int(T))
        debug_log = []

        def _check(name, dram_addr, shape, cpu_ref, cols=None):
            """shape is the buffer's ACTUAL (possibly padded) layout in DRAM -- always read that,
            then slice to the first `cols` real columns if the buffer is wider than the logical
            (unpadded) data (e.g. duration_proj's N=50 padded to 64). Reading a shape narrower than
            the true row stride would misread strided/shifted data, not just extra padding.
            """
            if debug_cpu_ref is None:
                return
            _debug_flush(ue, self._run)
            got = ue.dma_from_accelerator_memory(dram_addr, shape).float()
            if cols is not None:
                got = got[:, :cols]
            from user_dma_core import calculate_snr
            snr_db = calculate_snr(cpu_ref.detach().float().reshape(-1), got.reshape(-1))
            debug_log.append((name, snr_db))

        self.identity_dram = self._up(torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))

        # Per-RUN input (the voice pack is indexed by phoneme count), so it must land at a fixed
        # activation address every run -- never in the content-hashed constant memo, whose
        # address would move with the prompt while the frozen program keeps reading the old one.
        style_dram = self._up_cap(_bf16(style_vec), _bf16(style_vec).numel())
        style_tiled_dram = self._up_cap(_bf16(style_vec).unsqueeze(0).expand(T, style_dim).contiguous(),
                                        _cap() * style_dim)

        x_dram = ue.allocate_tensor_dram(_cap() * C * 2)
        ue.dma_to_accelerator_memory(x_dram, _bf16(d_en.T))  # [T, 512]

        # --- Stage A: 3x (BiLSTM + AdaLayerNorm + style-concat) -> d ---
        # Nothing in the DurationEncoder needs the host any more (_adaln tiles gamma/beta on
        # device), so the whole stage is ONE capture: concat, then every LSTM/AdaLN/concat block.
        ue.start_capture()
        cur_dram = ue.allocate_tensor_dram(_cap() * (C + style_dim) * 2)
        self._concat_columns(x_dram, C, style_tiled_dram, style_dim, T, cur_dram)  # [T, 640]

        for lw, aw in zip(self.lstm_blocks, self.adaln_blocks):
            lstm_out_dram = ue.allocate_tensor_dram(_cap() * C * 2)
            self._lstm_bidir(cur_dram, T, C + style_dim, lw, lstm_out_dram)  # [T, 512]

            adaln_out_dram = ue.allocate_tensor_dram(_cap() * C * 2)
            self._adaln(lstm_out_dram, T, C, style_dram, aw, adaln_out_dram)  # [T, 512]

            cur_dram = ue.allocate_tensor_dram(_cap() * (C + style_dim) * 2)
            self._concat_columns(adaln_out_dram, C, style_tiled_dram, style_dim, T, cur_dram)  # [T, 640]
        self._run(ue)
        d_dram = cur_dram  # [T, 640] -- matches KokoroModel.forward_with_tokens's `d`
        _check("d (DurationEncoder output)", d_dram, (T, C + style_dim), debug_cpu_ref["d"] if debug_cpu_ref else None)

        # --- Stage B: predictor.lstm ---
        ue.start_capture()
        pred_lstm_out_dram = ue.allocate_tensor_dram(_cap() * C * 2)
        self._lstm_bidir(d_dram, T, C + style_dim, self.pred_lstm_w, pred_lstm_out_dram)  # [T, 512]
        self._run(ue)
        _check("predictor.lstm output", pred_lstm_out_dram, (T, C), debug_cpu_ref["lstm_out"] if debug_cpu_ref else None)

        # --- Stage C: duration_proj (N padded to 64, see __init__'s dur_w/dur_b comment) ---
        ue.start_capture()
        dur_dram = ue.allocate_tensor_dram(_cap() * self.dur_n_pad * 2)
        _dyn_matmul(ue, M=T, K=C, N=self.dur_n_pad, A_DRAM_ADDR=pred_lstm_out_dram, B_DRAM_ADDR=self._up(self.dur_w),
                            OUTPUT_DRAM_ADDR=dur_dram, C_DRAM_ADDR=self._up(self.dur_b), bias_mode="broadcast_N")
        self._run(ue)
        _check("duration_proj (pre-sigmoid)", dur_dram, (T, self.dur_n_pad),
                debug_cpu_ref["duration_raw"] if debug_cpu_ref else None, cols=self.dur_n)

        if debug_cpu_ref is not None:
            report_snr("\n[fpga][debug] --- Section 2 bisect (SNR vs CPU) ---")
            for name, snr_db in debug_log:
                flag = "  <-- LOW" if snr_db < 30.0 else ""
                report_snr(f"[fpga][debug] {name:38s} {snr_db:8.2f} dB{flag}")
            report_snr("[fpga][debug] --- end bisect ---\n")

        duration_raw = ue.dma_from_accelerator_memory(dur_dram, (T, self.dur_n_pad)).float()[:, :self.dur_n]
        duration = torch.sigmoid(duration_raw).sum(axis=-1)
        pred_dur = torch.round(duration).clamp(min=1).long()
        d_out = ue.dma_from_accelerator_memory(d_dram, (T, C + style_dim)).float()
        ue.reset_tensor_dram_addr()
        return d_out, pred_dur


def _wn_conv(module):
    """Materialize a torch.nn.utils.weight_norm-wrapped Conv1d/ConvTranspose1d's effective weight
    from its (weight_g, weight_v) parametrization: w = g * v / ||v||_{dim=0} (the ATen op PyTorch's
    weight_norm hook computes internally). Falls back to a plain `.weight` for un-wrapped modules
    (e.g. F0_proj/N_proj, confirmed via the checkpoint's state dict to have no weight_g/weight_v).
    Computed directly from (g, v) rather than relying on the module's forward-hook side effect,
    since some of these conv modules may not have been run yet at the point we read their weight.
    """
    if hasattr(module, "weight_v"):
        w = torch._weight_norm(module.weight_v, module.weight_g, 0)
    else:
        w = module.weight
    b = module.bias if module.bias is not None else None
    return _bf16(w), (_bf16(b) if b is not None else None)


class F0NPredictionFPGA:
    """Section 3: predictor.shared LSTM + pred_aln_trg (phoneme->frame alignment) construction +
    F0Ntrain's two AdainResBlk1d stacks (F0 and N branches) -> F0_pred, N_pred.

    New op types introduced here (all individually validated in the earlier op-gap hardware proofs,
    being composed into the model for the first time):
      - Conv1d (k=3, pad=1, regular + k=1 pointwise) -> shifted-matmul-accumulate / plain matmul
        (conv1d_shifted_matmul_test, both at 40-51dB on real hardware).
      - AdaIN1d (InstanceNorm1d + style-conditioned per-channel affine) -> layer_norm_core_dram on
        a transposed [C,T] view (InstanceNorm1d proof, 46.5dB) + the same style-affine round-trip
        pattern as Section 2's AdaLayerNorm.
      - Depthwise ConvTranspose1d (the `pool` in upsampling blocks) -> zero-insertion (native
        bf16_permute_dram_core strided scatter) + depthwise shifted-eltwise-MAC with the flipped
        kernel (conv_transpose1d_zero_insert_test + depthwise_conv1d_eltwise_test, 49-53dB).
      - Nearest 2x upsample (shortcut path) -> exact row-duplicate (interpolate_fixed_matmul_test's
        nearest mode was bit-exact / inf dB).
      - LeakyReLU(0.2) -> relu(x) - 0.2*relu(-x), derived exactly from the native `clamp` activation
        (clamp_min=0, clamp_max=inf IS relu), no new primitive.

    Internal layout convention: [T, C] row-major (T=frames as rows) throughout, matching Sections
    1-2 and the conv1d_shifted_matmul_test proof (shifted reads are row offsets). AdaIN1d needs the
    OPPOSITE layout ([C, T], channels as rows) to use layer_norm_core_dram's per-row reduction --
    bf16_transpose_core (a validated native primitive) converts between the two as needed.
    """

    def __init__(self, model, ue: UnifiedEngine):
        self.model = model
        self.ue = ue
        self.style_dim = 128
        self.H = 256  # predictor.shared per-direction hidden
        self.identity_dram = None  # uploaded fresh in forward(), same reasoning as Section 2

        pred = model.predictor

        def lstm_w(m):
            return dict(
                Wx_f=_bf16(m.weight_ih_l0), Wh_f=_bf16(m.weight_hh_l0),
                b_f=_bf16(m.bias_ih_l0 + m.bias_hh_l0),
                Wx_b=_bf16(m.weight_ih_l0_reverse), Wh_b=_bf16(m.weight_hh_l0_reverse),
                b_b=_bf16(m.bias_ih_l0_reverse + m.bias_hh_l0_reverse),
            )

        def block_w(m, has_upsample):
            w = dict(
                conv1_w=None, conv1_b=None, conv2_w=None, conv2_b=None,
                norm1_fc_w=_bf16(m.norm1.fc.weight), norm1_fc_b=_bf16(m.norm1.fc.bias),
                norm2_fc_w=_bf16(m.norm2.fc.weight), norm2_fc_b=_bf16(m.norm2.fc.bias),
            )
            w["conv1_w"], w["conv1_b"] = _wn_conv(m.conv1)
            w["conv2_w"], w["conv2_b"] = _wn_conv(m.conv2)
            if m.learned_sc:
                w["conv1x1_w"], _ = _wn_conv(m.conv1x1)
            if has_upsample:
                w["pool_w"], w["pool_b"] = _wn_conv(m.pool)
            return w

        self.shared_w = lstm_w(pred.shared)
        self.F0_blocks = [block_w(pred.F0[i], i == 1) for i in range(3)]
        self.N_blocks = [block_w(pred.N[i], i == 1) for i in range(3)]

        self.proj_n = 1
        self.proj_n_pad = _round_up(self.proj_n, UE_VECTOR_SIZE)
        f0_proj_w, f0_proj_b = pred.F0_proj.weight, pred.F0_proj.bias  # [1,256,1], [1] -- not weight_norm wrapped
        n_proj_w, n_proj_b = pred.N_proj.weight, pred.N_proj.bias
        self.F0_proj_w = torch.zeros(self.proj_n_pad, 256, dtype=torch.bfloat16)
        self.F0_proj_w[:1] = _bf16(f0_proj_w.squeeze(-1))
        self.F0_proj_b = torch.zeros(self.proj_n_pad, dtype=torch.bfloat16); self.F0_proj_b[:1] = _bf16(f0_proj_b)
        self.N_proj_w = torch.zeros(self.proj_n_pad, 256, dtype=torch.bfloat16)
        self.N_proj_w[:1] = _bf16(n_proj_w.squeeze(-1))
        self.N_proj_b = torch.zeros(self.proj_n_pad, dtype=torch.bfloat16); self.N_proj_b[:1] = _bf16(n_proj_b)

    def _up(self, tensor: torch.Tensor) -> int:
        """Constant weights/biases only -- memoised, see _upload_const. Anything whose CONTENT
        changes from run to run must go through _up_cap so it is re-DMA'd every call."""
        return _upload_const(self.ue, tensor)

    def _up_cap(self, tensor: torch.Tensor, cap_elems: int) -> int:
        """Upload `tensor` but RESERVE cap_elems, so the allocator advances by a constant amount.
        The tensor's own layout is untouched -- only the address becomes prompt-independent."""
        assert tensor.numel() <= cap_elems, (tensor.numel(), cap_elems)
        a = self.ue.allocate_tensor_dram(cap_elems * 2)
        self.ue.dma_to_accelerator_memory(a, tensor.reshape(-1))
        return a

    def _run(self, ue):
        _timed_run(ue)

    def _lstm_bidir(self, x_dram, T, Cin, w, out_dram):
        """Identical to ProsodyDurationFPGA's (see its docstring for the full derivation) --
        duplicated rather than shared across classes to keep each section's file self-contained."""
        ue, H = self.ue, self.H
        G = 4 * H

        def run_direction(reverse, Wx, Wh, b, col_off_bytes):
            wx_dram, wh_dram, b_dram = self._up(Wx), self._up(Wh), self._up(b)
            gates_x = ue.allocate_tensor_dram(G * 2)
            gates_h = ue.allocate_tensor_dram(G * 2)
            gates = ue.allocate_tensor_dram(G * 2)
            i_d, f_d, g_d, o_d = (ue.allocate_tensor_dram(H * 2) for _ in range(4))
            g_tmp, c_tmp, tanh_c, fc_, ig_ = (ue.allocate_tensor_dram(H * 2) for _ in range(5))
            h_bufs = [ue.allocate_tensor_dram(H * 2), ue.allocate_tensor_dram(H * 2)]
            c_bufs = [ue.allocate_tensor_dram(H * 2), ue.allocate_tensor_dram(H * 2)]
            ue.dma_to_accelerator_memory(h_bufs[0], torch.zeros(H, dtype=torch.bfloat16))
            ue.dma_to_accelerator_memory(c_bufs[0], torch.zeros(H, dtype=torch.bfloat16))
            M_H = H // UE_VECTOR_SIZE

            def sigmoid_ip(src, dst):
                _dyn_activation(ue, M=M_H, N=UE_VECTOR_SIZE, A_DRAM_ADDR=src, OUTPUT_DRAM_ADDR=dst,
                                    IDENTITY_DRAM_ADDR=self.identity_dram, activation="sigmoid")

            def tanh_via_sigmoid(src, tmp, dst):
                _dyn_eltwise(ue, 1, H, src, None, tmp, mode=UE_MODE.MUL_BROADCAST, scalar=2.0)
                sigmoid_ip(tmp, tmp)
                _dyn_eltwise(ue, 1, H, tmp, None, tmp, mode=UE_MODE.MUL_BROADCAST, scalar=2.0)
                _dyn_eltwise(ue, 1, H, tmp, None, dst, mode=UE_MODE.ADD_BROADCAST, scalar=-1.0)

            # ONE hardware loop instead of T Python-unrolled bodies. The recurrence is still
            # strictly sequential -- this changes only how the instruction stream is EMITTED, not
            # the order the accelerator executes in.
            #
            # Two things varied per timestep and both are now register-computed:
            #   * the input row x[t], gathered into a fixed scratch buffer;
            #   * the output row out[t], scattered from a fixed buffer.
            # Everything between keeps literal addresses, so the body is the same op sequence as
            # before; only the gather/scatter use general_reg_src. That avoids forcing every M=1
            # matmul onto the dynamic path just to source an address.
            #
            # h/c became SINGLE in-place buffers (they used to ping-pong per timestep, which a loop
            # body cannot express). Safe: gates_h reads h before h is rewritten at the end of the
            # body, and fc_ consumes c before c is rewritten -- every read precedes its write.
            x_scratch = ue.allocate_tensor_dram(Cin * 2)
            h_buf, c_buf = h_bufs[0], c_bufs[0]
            i_reg = ue.alloc_isa_reg()
            a_reg = ue.alloc_isa_reg()
            if isinstance(T, Dim) and T.base is not None:
                # i starts at T-1 when walking backwards, so it too must come from the register.
                ue.generate_instruction_add_imm(_emit_M(ue, T), -1, i_reg) if reverse \
                    else ue.generate_instruction_add_set(i_reg, 0)
                ue.loop_start(loop_cnt=_template(T), gpr_loop_cnt=_emit_M(ue, T))
            else:
                ue.generate_instruction_add_set(i_reg, (int(T) - 1) if reverse else 0)
                ue.loop_start(loop_cnt=int(T))

            ue.generate_instruction_reg_mul_imm(a_reg, i_reg, ue_35bit_addr_shifter(Cin * 2))
            ue.generate_instruction_add_imm(a_reg, ue_35bit_addr_shifter(x_dram), a_reg)
            ue.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=0x00000,
                                          element_size=Cin, general_reg_src=a_reg)
            ue.sram_to_accelerator_memory(sram_address=0x00000,
                                          accelerator_dram_address=x_scratch, element_size=Cin)

            _dyn_matmul(ue, M=1, K=Cin, N=G, A_DRAM_ADDR=x_scratch, B_DRAM_ADDR=wx_dram,
                        OUTPUT_DRAM_ADDR=gates_x, C_DRAM_ADDR=b_dram, bias_mode="broadcast_N")
            _dyn_matmul(ue, M=1, K=H, N=G, A_DRAM_ADDR=h_buf, B_DRAM_ADDR=wh_dram,
                        OUTPUT_DRAM_ADDR=gates_h)
            _dyn_eltwise(ue, 1, G, gates_x, gates_h, gates, mode=UE_MODE.ELTWISE_ADD)

            i_raw, f_raw, g_raw, o_raw = (gates + k * H * 2 for k in range(4))
            sigmoid_ip(i_raw, i_d); sigmoid_ip(f_raw, f_d); sigmoid_ip(o_raw, o_d)
            tanh_via_sigmoid(g_raw, g_tmp, g_d)

            _dyn_eltwise(ue, 1, H, f_d, c_buf, fc_, mode=UE_MODE.ELTWISE_MUL)
            _dyn_eltwise(ue, 1, H, i_d, g_d, ig_, mode=UE_MODE.ELTWISE_MUL)
            _dyn_eltwise(ue, 1, H, fc_, ig_, c_buf, mode=UE_MODE.ELTWISE_ADD)

            tanh_via_sigmoid(c_buf, c_tmp, tanh_c)
            _dyn_eltwise(ue, 1, H, o_d, tanh_c, h_buf, mode=UE_MODE.ELTWISE_MUL)

            ue.accelerator_memory_to_sram(accelerator_dram_address=h_buf, sram_address=0x00000,
                                          element_size=H)
            ue.generate_instruction_reg_mul_imm(a_reg, i_reg, ue_35bit_addr_shifter((2 * H) * 2))
            ue.generate_instruction_add_imm(a_reg, ue_35bit_addr_shifter(out_dram + col_off_bytes), a_reg)
            ue.sram_to_accelerator_memory(sram_address=0x00000, accelerator_dram_address=0,
                                          element_size=H, general_reg_src=a_reg)

            if reverse:
                ue.generate_instruction_add_dec(i_reg)
            else:
                ue.generate_instruction_add_inc(i_reg)
            ue.loop_end()
            ue.release_isa_reg()   # a_reg
            ue.release_isa_reg()   # i_reg

        run_direction(False, w["Wx_f"], w["Wh_f"], w["b_f"], 0)
        run_direction(True, w["Wx_b"], w["Wh_b"], w["b_b"], H * 2)

    def _device_row_copy(self, src_dram, dst_dram, num_rows, row_width, gpr_rows=None):
        """Pure on-device row-by-row copy: CAPTURED instructions (accelerator_memory_to_sram ->
        sram_to_accelerator_memory, same technique as Section 1's embedding gather), NOT a host
        round-trip. Safe to call from inside an already-open capture block -- unlike reading a
        same-capture-block predecessor's output back to the host, which would read stale/
        uninitialized DRAM (that instruction hasn't executed yet, only been recorded). This is the
        device-to-device copy this class uses everywhere it needs to restage data (padding,
        zero-insertion, etc).
        """
        ue = self.ue
        row_bytes = row_width * 2
        _pbi_row_loop(ue, num_rows,
                      reads=[(src_dram, row_bytes, row_width, 0)],
                      writes=[(dst_dram, row_bytes, row_width, 0)],
                      gpr_rows=gpr_rows)

    def _conv1d(self, x_dram, T, Cin, Cout, W, b, out_dram, kernel_size, pad, real_T=None):
        """Conv1d via K shifted-matmul-accumulate (conv1d_shifted_matmul_test's validated pattern).
        W: [Cout, Cin, kernel_size] bf16. kernel_size=1,pad=0 degenerates to a single plain matmul
        (used for the pointwise conv1x1/F0_proj/N_proj cases).

        ``real_T`` is the UNPADDED frame count when ``T`` has been rounded up to a multiple of 64
        for SRAM alignment. It matters: this conv's zero-padding must sit at the REAL sequence
        boundary, exactly where CPU Conv1d puts it. Rows real_T..T hold whatever the previous
        AdaIN left there (layer_norm of a zero row is not zero, and the preceding conv's bias
        lands there too), so without this the last real frame gets convolved against garbage
        instead of against zero -- one badly-wrong row in real_T, i.e. an SNR ceiling of
        10*log10(real_T) ~= 24 dB at 253 frames. Every other row is unaffected, which is why the
        symptom looks like a single low block that the next InstanceNorm partly renormalizes away.
        """
        ue = self.ue
        if kernel_size == 1:
            _dyn_matmul(ue, M=T, K=Cin, N=Cout, A_DRAM_ADDR=x_dram, B_DRAM_ADDR=self._up(W.squeeze(-1)),
                                OUTPUT_DRAM_ADDR=out_dram,
                                C_DRAM_ADDR=(self._up(b) if b is not None else None),
                                bias_mode="broadcast_N")
            return
        x_pad_dram = ue.allocate_tensor_dram((_cap() + 2 * pad) * Cin * 2)
        ue.dma_to_accelerator_memory(
            x_pad_dram, torch.zeros((T + 2 * pad) * Cin, dtype=torch.bfloat16))  # constant zero-fill, safe direct DMA
        self._device_row_copy(x_dram, x_pad_dram + pad * Cin * 2, T, Cin)  # on-device, NOT a host round-trip
        # Emitted UNCONDITIONALLY. Gating this on ``real_T < T`` made the program depend on whether
        # the prompt happened to pad: a 64-phoneme prompt emitted no pad ops, so an image compiled
        # on it was wrong for every other prompt (and vice versa: program addresses shifted by the
        # missing instructions). With zero pad rows the loop below still runs once, harmlessly.
        if real_T is not None:
            # Re-zero the alignment padding so the conv's boundary zero lands at the real end of
            # the sequence.
            #
            # This MUST be an on-device (captured) copy, not a dma_to_accelerator_memory. Host DMA
            # runs IMMEDIATELY at emit time, while _device_row_copy above is a CAPTURED instruction
            # that runs later during _run() -- so a host zero-fill here would be overwritten by that
            # copy when the program actually executes, and the change would be a silent no-op.
            # (The memset above survives only at rows 0 and T+1, which the copy never touches.)
            # Allocate and address at the CAP, not at this run's padding width. Sizing the
            # scratch by (T - real_T) would shift every later allocation in the section, and the
            # destination offset would be baked into the captured row copy -- both prompt-
            # dependent. The zero buffer is cap-sized and written from row 0; the copy walks a
            # runtime row count to a register-derived destination.
            # (T - real_T) is a DIFFERENCE of two runtime lengths, not an affine function of
            # either, so it gets its own preamble-primed register. Both T and real_T carry the
            # same upsample factor, so (T - real_T) == (nf_pad - n_frames) * T.mul.
            n_pad_rows = (Dim(T - real_T, GPR_PADROWS, mul=T.mul, cap=T.cap)
                          if _is_dyn(T) else (T - real_T))
            zeros_dram = ue.allocate_tensor_dram(_cap() * Cin * 2)
            ue.dma_to_accelerator_memory(zeros_dram, torch.zeros(_cap() * Cin, dtype=torch.bfloat16))
            # Destination row = pad + real_T + i: the start comes from real_T's register, so no
            # prompt-dependent byte offset is baked into the copy. Zero source stride (one zero
            # row, re-read every iteration) so the walk never runs off the end of the buffer.
            # Trip count is pad_rows + 1, never pad_rows: a prompt whose length is an exact
            # multiple of 64 has ZERO pad rows, and a hardware loop entered with count 0 wraps to
            # 2^32 and never returns (hung the board on a 64-phoneme prompt). The extra iteration
            # zeroes row pad + T -- the right-padding row, which is zero anyway and inside the
            # cap + 2*pad buffer.
            if _is_dyn(n_pad_rows):
                ue.generate_instruction_add_imm(_emit_M(ue, n_pad_rows), 1, GPR_TMP)
                gpr_cnt, cnt = GPR_TMP, _template(n_pad_rows) + 1
            else:
                gpr_cnt, cnt = None, int(n_pad_rows) + 1
            _pbi_row_loop(ue, cnt, reads=[(zeros_dram, 0, Cin, 0)],
                          writes=[(x_pad_dram, Cin * 2, Cin, 0)],
                          gpr_rows=gpr_cnt, row_start=real_T, row_start_off=pad)
        w_tap_dram = [self._up(W[:, :, k].contiguous()) for k in range(kernel_size)]
        acc_a = ue.allocate_tensor_dram(_cap() * Cout * 2)
        acc_b = ue.allocate_tensor_dram(_cap() * Cout * 2)
        for k in range(kernel_size):
            A_DRAM_ADDR = x_pad_dram + k * Cin * 2
            prev_acc = acc_a if k % 2 == 0 else acc_b
            cur_acc = acc_b if k % 2 == 0 else acc_a
            bias_addr = self._up(b) if (b is not None and k == 0) else None
            _dyn_matmul(ue, 
                M=T, K=Cin, N=Cout, A_DRAM_ADDR=A_DRAM_ADDR, B_DRAM_ADDR=w_tap_dram[k], OUTPUT_DRAM_ADDR=cur_acc,
                C_DRAM_ADDR=(bias_addr if k == 0 else (prev_acc if k > 0 else None)),
                bias_mode=("broadcast_N" if (k == 0 and bias_addr is not None) else "full_matrix"),
            )
        final_acc = acc_b if (kernel_size - 1) % 2 == 0 else acc_a
        if final_acc != out_dram:
            self._device_row_copy(final_acc, out_dram, T, Cout)  # on-device, NOT a host round-trip

    def _leaky_relu(self, x_dram, T, C, out_dram):
        ue = self.ue
        numel = T * C                       # a Dim: rows below stay register-driven
        M, N = numel // UE_VECTOR_SIZE, UE_VECTOR_SIZE
        # Allocate at the CAP: sizing by the prompt shifts every later buffer's address, which
        # bakes the prompt into every later program's operands.
        neg_dram = ue.allocate_tensor_dram(_cap() * C * 2)
        relu_dram = ue.allocate_tensor_dram(_cap() * C * 2)
        relu_neg_dram = ue.allocate_tensor_dram(_cap() * C * 2)
        _dyn_activation(ue, M=M, N=N, A_DRAM_ADDR=x_dram, OUTPUT_DRAM_ADDR=relu_dram,
                            IDENTITY_DRAM_ADDR=self.identity_dram, activation="clamp")
        _dyn_eltwise(ue, M, N, x_dram, None, neg_dram, mode=UE_MODE.MUL_BROADCAST, scalar=-1.0)
        _dyn_activation(ue, M=M, N=N, A_DRAM_ADDR=neg_dram, OUTPUT_DRAM_ADDR=relu_neg_dram,
                            IDENTITY_DRAM_ADDR=self.identity_dram, activation="clamp")
        _dyn_eltwise(ue, M, N, relu_neg_dram, None, relu_neg_dram, mode=UE_MODE.MUL_BROADCAST, scalar=0.2)
        _dyn_eltwise(ue, M, N, relu_dram, relu_neg_dram, out_dram, mode=UE_MODE.ELTWISE_SUB)

    def _nearest_upsample2x(self, x_dram, T, C, out_dram, gpr_rows=None):
        """Exact row-duplicate: out[2t]=out[2t+1]=x[t]. Bit-exact (interpolate_fixed_matmul_test's
        nearest mode measured inf dB), implemented as a per-row copy loop (same technique as
        Section 1's embedding gather)."""
        ue = self.ue
        row_bytes = C * 2
        # out[2t] = out[2t+1] = x[t]: one read, two writes whose destination stride is 2*row_bytes,
        # the second offset one row further in. Still bit-exact -- it is a pure row duplicate.
        _pbi_row_loop(ue, T,
                      reads=[(x_dram, row_bytes, C, 0)],
                      writes=[(out_dram, 2 * row_bytes, C, 0),
                              (out_dram + row_bytes, 2 * row_bytes, C, 0)],
                      gpr_rows=gpr_rows)

    def _depthwise_convtranspose_upsample2x(self, x_dram, T, C, W, b, out_dram):
        """Depthwise ConvTranspose1d(C,C,kernel=3,stride=2,padding=1,output_padding=1,groups=C):
        zero-insertion dilate (native bf16_permute_dram_core strided scatter, num_groups=2 so group1
        is an all-zeros buffer) + pad(left=1,right=2, derived in the conversation from the standard
        conv/conv_transpose duality: left=right_base=kernel-1-padding=1, right=right_base+output_
        padding=2) + depthwise 3-tap shifted-eltwise-MAC with the FLIPPED kernel taps. Output length
        (T-1)*stride-2*padding+kernel+output_padding = 2T (verified against the same formula).
        """
        ue = self.ue
        dilated_len = (T - 1) * 2 + 1
        # Grouped [2,T,C] buffer (group0=real x, group1=constant zeros), built via ONE on-device
        # copy (x_dram -> group0; group1 is a constant zero-fill, a safe direct DMA since it doesn't
        # depend on any same-capture-block predecessor) -- then scattered into the interleaved
        # [T,2,C] == dilated [2T-1,C] layout via bf16_permute_dram_core's native strided write.
        grouped_dram = ue.allocate_tensor_dram(2 * _cap() * C * 2)
        # Second half starts at the CAP offset so the captured row copy below carries a fixed
        # destination; the buffer is already cap-sized for both halves.
        ue.dma_to_accelerator_memory(grouped_dram + _cap() * C * 2,
                                     torch.zeros(_cap() * C, dtype=torch.bfloat16))
        self._device_row_copy(x_dram, grouped_dram, T, C)  # on-device, NOT a host round-trip
        dilated_dram = ue.allocate_tensor_dram(((2 * _cap()) + 1) * C * 2)
        # group_rows must match the cap offset the zero half was written at. Interleaving at the
        # cap still lands real row i at dilated row 2i, so rows past 2T are simply unused -- the
        # copy below takes only dilated_len of them.
        ue.bf16_permute_dram_core(num_groups=2, group_rows=_cap(), row_width=C,
                                   in_dram=grouped_dram, out_dram=dilated_dram, write_grouped=False)

        pad_l, pad_r = 1, 2
        L_pad = pad_l + dilated_len + pad_r
        x_dilated_pad = ue.allocate_tensor_dram((2 * _cap() + 4) * C * 2)
        ue.dma_to_accelerator_memory(x_dilated_pad, torch.zeros(L_pad * C, dtype=torch.bfloat16))  # constant, safe
        # Copy 2T rows (a Dim), not dilated_len = 2T-1 (a bare int that bakes the frame count).
        # Row 2T-1 of the dilated buffer is an interleaved zero row, landing on the right pad,
        # which is zero anyway.
        self._device_row_copy(dilated_dram, x_dilated_pad + pad_l * C * 2, 2 * T, C)  # on-device

        T_out = 2 * T
        assert L_pad - 3 + 1 == T_out, (L_pad, T_out)
        w_tap_tiled = [W[:, 0, 2 - k].unsqueeze(0).expand(T_out, C).contiguous() for k in range(3)]  # flipped taps
        w_tap_dram = [self._up_cap(w_tap_tiled[k], _cap() * C) for k in range(3)]
        tmp = [ue.allocate_tensor_dram(_cap() * C * 2) for _ in range(3)]
        acc_a = ue.allocate_tensor_dram(_cap() * C * 2)
        acc_b = ue.allocate_tensor_dram(_cap() * C * 2)
        running = None
        for k in range(3):
            shifted_x = x_dilated_pad + k * C * 2
            _dyn_eltwise(ue, T_out, C, shifted_x, w_tap_dram[k], tmp[k], mode=UE_MODE.ELTWISE_MUL)
            if k == 0:
                running = tmp[0]
                continue
            cur = acc_a if k % 2 == 1 else acc_b
            _dyn_eltwise(ue, T_out, C, tmp[k], running, cur, mode=UE_MODE.ELTWISE_ADD)
            running = cur
        b_tiled_dram = self._up_cap(b.unsqueeze(0).expand(T_out, C).contiguous(), _cap() * C)
        _dyn_eltwise(ue, T_out, C, running, b_tiled_dram, out_dram, mode=UE_MODE.ELTWISE_ADD)

    def _adain1d(self, x_dram, T, C, style_dram, fc_w, fc_b, out_dram, real_T=None):
        """AdaIN1d: InstanceNorm1d (no affine) + style-conditioned per-channel affine, entirely on
        device and sequence-length invariant. Ported step-for-step from user_hw_test.py's
        adain1d_frozen_device_test (47 dB, C=512/256, real_T 78..1920, one body per C):

          1. gb = style @ fc_w^T + fc_b                  [1, 2C]   fixed dims
          2. g1 = gamma + 1                              [1, C]    ADD_BROADCAST
          3. zero x's pad rows [T, cap)                  HW loop, trip count = cap - T (register)
          4. x_ct = transpose(x) with M = cap            [C, cap]  row stride == the baked LN N
          5. InstanceNorm: layer-norm short-N mode, N baked at cap, gpr_N_reg = real_T,
             gpr_sqrt_n_reg = sqrt(real_T), inv_n/mask tables (1/real_T, 1 in real lanes; 0 after)
          6. normed = transpose(normed_ct) with M = C    [cap, C]
          7. g1 / beta tiled down T rows                 zero-source-stride HW loops, trip = T
          8. out = normed * g1 + beta                    eltwise, rows = T

        Steps 4 and 6 execute at the cap: a [C, cap] view with a fixed row stride can only come
        from a transpose whose M IS the cap. Everything else runs at the real length. inv_n and
        mask are per-prompt DATA (like the LLMs' attention masks); no instruction depends on T.
        """
        ue = self.ue
        cap = _cap()
        assert int(T) < cap, (
            f"AdaIN pad-row loop needs at least one pad row: T={int(T)} must be < cap={cap}")
        real = int(real_T) if real_T is not None else int(T)
        row_b = C * 2
        gb_dram = ue.allocate_tensor_dram(2 * C * 2)
        g1_dram = ue.allocate_tensor_dram(C * 2)
        x_ct = ue.allocate_tensor_dram(C * cap * 2)
        normed_ct = ue.allocate_tensor_dram(C * cap * 2)
        normed = ue.allocate_tensor_dram(cap * C * 2)
        gamma_tiled = ue.allocate_tensor_dram(cap * C * 2)
        beta_tiled = ue.allocate_tensor_dram(cap * C * 2)
        ng_dram = ue.allocate_tensor_dram(cap * C * 2)
        # Per-prompt tables and constants. The norm-length registers are keyed by (cap, T's
        # derivation) so the pre- and post-upsample AdaINs at the same cap get separate pairs.
        n_reg, sq_reg = set_runtime_norm_len(ue, cap, real, key=(cap, _as_dim(T).mul))
        inv_n = torch.zeros(cap, dtype=torch.bfloat16); inv_n[:real] = 1.0 / real
        mask = torch.zeros(cap, dtype=torch.bfloat16); mask[:real] = 1.0
        invn_dram = self._up_cap(inv_n, cap)
        mask_dram = self._up_cap(mask, cap)
        ones_dram = self._up_cap(torch.ones(cap, dtype=torch.bfloat16), cap)
        zeros_dram = self._up_cap(torch.zeros(max(cap, C), dtype=torch.bfloat16), max(cap, C))

        # 1. style affine (M=1 -> legacy tiling, fixed dims)
        _dyn_matmul(ue, M=1, K=self.style_dim, N=2 * C, A_DRAM_ADDR=style_dram, B_DRAM_ADDR=self._up(fc_w),
                    OUTPUT_DRAM_ADDR=gb_dram, C_DRAM_ADDR=self._up(fc_b), bias_mode="broadcast_N")
        # 2. g1 = gamma + 1
        ue.eltwise_core_dram(1, C, gb_dram, None, g1_dram, UE_MODE.ADD_BROADCAST, scalar=1.0)
        # 3. zero x's pad rows [T, cap): stale DRAM there would poison the stats (0 * NaN = NaN)
        _zero_rows_from(ue, x_dram, row_b, C, start=T, count_cap=cap, zeros_src=zeros_dram)
        # 4. x_ct [C, cap] = transpose(x [cap, C]) -- M = cap so the row stride is cap
        ue.bf16_transpose_core(M=cap, N=C, INPUT_DRAM_ADDR=x_dram, OUTPUT_DRAM_ADDR=x_ct,
                               IDENTITY_DRAM_ADDR=self.identity_dram, gpr_M_reg=_emit_M(ue, cap))
        # 5. InstanceNorm over time, short-N mode
        _dyn_layernorm(ue, M=C, N=cap, A_DRAM_ADDR=x_ct, OUTPUT_DRAM_ADDR=normed_ct,
                       gpr_N_reg=n_reg, gpr_sqrt_n_reg=sq_reg, GAMMA_DRAM_ADDR=ones_dram,
                       INV_N_DRAM_ADDR=invn_dram, MASK_DRAM_ADDR=mask_dram,
                       ZEROS_DRAM_ADDR=zeros_dram)
        # 6. normed [cap, C] = transpose(normed_ct [C, cap])
        ue.bf16_transpose_core(M=C, N=cap, INPUT_DRAM_ADDR=normed_ct, OUTPUT_DRAM_ADDR=normed,
                               IDENTITY_DRAM_ADDR=self.identity_dram, gpr_M_reg=_emit_M(ue, C))
        # 7. tile g1 / beta down T rows (source stride 0 = broadcast the single row)
        _pbi_row_loop(ue, T, reads=[(g1_dram, 0, C, 0)], writes=[(gamma_tiled, row_b, C, 0)])
        _pbi_row_loop(ue, T, reads=[(gb_dram + row_b, 0, C, 0)], writes=[(beta_tiled, row_b, C, 0)])
        # 8. out = normed * g1 + beta
        _dyn_eltwise(ue, T, C, normed, gamma_tiled, ng_dram, mode=UE_MODE.ELTWISE_MUL)
        _dyn_eltwise(ue, T, C, ng_dram, beta_tiled, out_dram, mode=UE_MODE.ELTWISE_ADD)

    def _adain_res_blk(self, x_dram, T, Cin, Cout, upsample, w, style_dram, real_T=None):
        """One AdainResBlk1d. ``T`` is the 64-aligned row count the device operates on; ``real_T``
        is the unpadded frame count, forwarded to _conv1d so its zero-padding lands at the real
        sequence boundary. Returns (out_dram, T_out)."""
        ue = self.ue
        T_out = 2 * T if upsample else T
        real_T_out = None if real_T is None else (2 * real_T if upsample else real_T)
        learned_sc = Cin != Cout

        ue.start_capture()
        h1_dram = ue.allocate_tensor_dram(_cap() * Cin * 2)
        self._adain1d(x_dram, T, Cin, style_dram, w["norm1_fc_w"], w["norm1_fc_b"], h1_dram,
                      real_T=real_T)
        act1_dram = ue.allocate_tensor_dram(_cap() * Cin * 2)
        self._leaky_relu(h1_dram, T, Cin, act1_dram)
        if upsample:
            pooled_dram = ue.allocate_tensor_dram(_cap() * Cin * 2)
            self._depthwise_convtranspose_upsample2x(act1_dram, T, Cin, w["pool_w"], w["pool_b"], pooled_dram)
        else:
            pooled_dram = act1_dram
        conv1_out_dram = ue.allocate_tensor_dram(_cap() * Cout * 2)
        self._conv1d(pooled_dram, T_out, Cin, Cout, w["conv1_w"], w["conv1_b"], conv1_out_dram, kernel_size=3, pad=1,
                     real_T=real_T_out)
        h2_dram = ue.allocate_tensor_dram(_cap() * Cout * 2)
        self._adain1d(conv1_out_dram, T_out, Cout, style_dram, w["norm2_fc_w"], w["norm2_fc_b"], h2_dram,
                      real_T=real_T_out)
        act2_dram = ue.allocate_tensor_dram(_cap() * Cout * 2)
        self._leaky_relu(h2_dram, T_out, Cout, act2_dram)
        residual_dram = ue.allocate_tensor_dram(_cap() * Cout * 2)
        self._conv1d(act2_dram, T_out, Cout, Cout, w["conv2_w"], w["conv2_b"], residual_dram, kernel_size=3, pad=1,
                     real_T=real_T_out)

        if upsample:
            sc_dram = ue.allocate_tensor_dram(_cap() * Cin * 2)
            self._nearest_upsample2x(x_dram, T, Cin, sc_dram)
        else:
            sc_dram = x_dram
        if learned_sc:
            sc_proj_dram = ue.allocate_tensor_dram(_cap() * Cout * 2)
            self._conv1d(sc_dram, T_out, Cin, Cout, w["conv1x1_w"], None, sc_proj_dram, kernel_size=1, pad=0)
            sc_dram = sc_proj_dram

        out_dram = ue.allocate_tensor_dram(_cap() * Cout * 2)
        sum_dram = ue.allocate_tensor_dram(_cap() * Cout * 2)
        _dyn_eltwise(ue, T_out, Cout, residual_dram, sc_dram, sum_dram, mode=UE_MODE.ELTWISE_ADD)
        _dyn_eltwise(ue, T_out, Cout, sum_dram, None, out_dram, mode=UE_MODE.MUL_BROADCAST, scalar=0.7071067811865476)
        self._run(ue)
        return out_dram, T_out

    def forward(self, d: torch.Tensor, pred_dur: torch.LongTensor, style_vec: torch.Tensor, T: int,
                debug_cpu_ref=None):
        """d: [T,640] (Section 2's output). pred_dur: [T] LongTensor (Section 2's output, exact).
        style_vec: [128]. Returns (F0_pred [n_frames], N_pred [n_frames]) FloatTensors.
        """
        ue = self.ue
        _set_row_cap(2 * _round_up(int(pred_dur.sum().item()), UE_VECTOR_SIZE),
                     2 * F_CAP, "2 x padded n_frames")
        debug_log = []

        def _check(name, dram_addr, shape, cpu_ref, cols=None, rows=None):
            """`shape` is the buffer's ACTUAL (frame-padded) layout in DRAM -- always read that,
            then slice to the `rows` REAL frames before comparing, since the CPU reference is built
            at the unpadded frame count. Same rule as Section 2's `cols` for N-padded buffers."""
            if debug_cpu_ref is None:
                return
            _debug_flush(ue, self._run)
            got = ue.dma_from_accelerator_memory(dram_addr, shape).float()
            if rows is not None:
                got = got[:rows]
            if cols is not None:
                got = got[:, :cols]
            from user_dma_core import calculate_snr
            snr_db = calculate_snr(cpu_ref.detach().float().reshape(-1), got.reshape(-1))
            debug_log.append((name, snr_db))

        self.identity_dram = self._up(torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))
        # Per-RUN input (the voice pack is indexed by phoneme count), so it must land at a fixed
        # activation address every run -- never in the content-hashed constant memo, whose
        # address would move with the prompt while the frozen program keeps reading the old one.
        style_dram = self._up_cap(_bf16(style_vec), _bf16(style_vec).numel())

        # --- pred_aln_trg (host-built: pred_dur is already exact, this is pure data staging, same
        #     spirit as Section 1's contiguous position-embedding slice) + en = d^T @ pred_aln_trg ---
        n_frames = int(pred_dur.sum().item())
        frame_idx = torch.repeat_interleave(torch.arange(T), pred_dur)  # [n_frames], phoneme index per frame

        # T is the CONTRACTION dim of the `en` matmul below, and matmat_mul_core requires
        # K % UE_VECTOR_SIZE == 0 (user_dma_core.py:2814) -- the SRAM write-back offset is
        # m_take*K*bpe, which must land on a 128B boundary (user_dma_core.py:1685). A raw
        # T=95 gives 255*95*2 = 48450, i.e. 66 mod 128, and asserts. Pad K to T_pad on BOTH
        # operands; this is exact, not approximate: the padded rows of the one-hot
        # pred_aln_trg_T are all-zero and the padded columns of d^T are never selected, so
        # they contribute nothing to the contraction. Same treatment Section 1 gives T
        # (see PLBertFPGA.forward's T_pad).
        T_pad = _round_up(T, UE_VECTOR_SIZE)
        # The FRAME COUNT is itself a 64-constrained dimension downstream: _adain1d puts it on
        # bf16_transpose_core's M, and on layer_norm_core_dram's / eltwise_core_dram's N. Every
        # SRAM row is 64 bf16 elements (128B), so any offset computed as rows*dim*2 needs dim%64==0.
        # Pad it ONCE here: the only length transform in Section 3 is the x2 upsample, so every
        # downstream T_out stays a multiple of 64 for free.
        nf_pad = _round_up(n_frames, UE_VECTOR_SIZE)
        T_pad = Dim(T_pad, GPR_TPAD, cap=T_CAP)
        nf_pad = Dim(nf_pad, GPR_NFPAD, cap=F_CAP)
        # The REAL frame count is its own runtime dimension, distinct from the padded one. The
        # shared LSTM must step exactly n_frames times -- it is a sequential bidirectional pass, so
        # running it into the padding would have the backward direction start inside the pad and
        # corrupt every real timestep -- so its trip count cannot just be nf_pad. Carrying it as a
        # Dim puts that count in a register instead of baking it into the loop.
        n_frames = Dim(n_frames, GPR_F, cap=F_CAP)
        set_runtime_dims(n_frames=int(n_frames), T_pad=int(T_pad), nf_pad=int(nf_pad),
                         pad_rows=int(nf_pad) - int(n_frames))

        pred_aln_trg_T = torch.zeros(nf_pad, T_pad, dtype=torch.bfloat16)
        pred_aln_trg_T[torch.arange(n_frames), frame_idx] = 1.0   # rows >= n_frames stay all-zero
        d_T = torch.zeros(640, T_pad, dtype=torch.bfloat16)
        d_T[:, :T] = _bf16(d).T

        ue.start_capture()
        en_dram = ue.allocate_tensor_dram(_cap() * 640 * 2)
        _dyn_matmul(ue, M=nf_pad, K=T_pad, N=640, A_DRAM_ADDR=self._up_cap(pred_aln_trg_T, _cap() * T_CAP),
                            B_DRAM_ADDR=self._up_cap(d_T.contiguous(), 640 * T_CAP), OUTPUT_DRAM_ADDR=en_dram)
        self._run(ue)

        # --- predictor.shared LSTM ---
        shared_out_dram = ue.allocate_tensor_dram(_cap() * 512 * 2)
        # Zero the whole buffer up front: the LSTM below writes only the first n_frames rows (it is
        # a SEQUENTIAL bidirectional pass -- running it over the padding would have the backward
        # direction start inside the pad and corrupt every real timestep), so rows n_frames..nf_pad
        # would otherwise be stale DRAM rather than a defined value.
        ue.dma_to_accelerator_memory(shared_out_dram, torch.zeros(nf_pad * 512, dtype=torch.bfloat16))
        ue.start_capture()
        self._lstm_bidir(en_dram, n_frames, 640, self.shared_w, shared_out_dram)
        self._run(ue)
        _check("shared LSTM output", shared_out_dram, (nf_pad, 512),
               debug_cpu_ref["shared_out"] if debug_cpu_ref else None, rows=n_frames)

        def run_branch(blocks, proj_w, proj_b, proj_n_pad, branch_name, cpu_refs):
            # cur_T is the PADDED row count the device actually operates on; real_T tracks the
            # unpadded frame count, which is what the CPU references are sized at. Both double
            # on the upsample block, so cur_T stays a multiple of 64 throughout.
            cur_dram, cur_T, cur_C = shared_out_dram, nf_pad, 512
            real_T = n_frames
            for i, (blk_w, upsample) in enumerate(zip(blocks, (False, True, False))):
                out_C = 512 if i == 0 else 256
                cur_dram, cur_T = self._adain_res_blk(cur_dram, cur_T, cur_C, out_C, upsample, blk_w,
                                                     style_dram, real_T=real_T)
                real_T = 2 * real_T if upsample else real_T
                cur_C = out_C
                if cpu_refs is not None:
                    _check(f"{branch_name} block {i}", cur_dram, (cur_T, cur_C), cpu_refs[i], rows=real_T)

            ue.start_capture()
            proj_dram = ue.allocate_tensor_dram(_cap() * proj_n_pad * 2)
            _dyn_matmul(ue, M=cur_T, K=cur_C, N=proj_n_pad, A_DRAM_ADDR=cur_dram, B_DRAM_ADDR=self._up(proj_w),
                                OUTPUT_DRAM_ADDR=proj_dram, C_DRAM_ADDR=self._up(proj_b), bias_mode="broadcast_N")
            self._run(ue)
            if cpu_refs is not None:
                _check(f"{branch_name}_proj", proj_dram, (cur_T, proj_n_pad), cpu_refs[3], cols=1, rows=real_T)
            out = ue.dma_from_accelerator_memory(proj_dram, (cur_T, proj_n_pad)).float()[:real_T, 0]
            return out

        f0_refs = debug_cpu_ref["F0_blocks"] + [debug_cpu_ref["F0_pred_pre_squeeze"]] if debug_cpu_ref else None
        n_refs = debug_cpu_ref["N_blocks"] + [debug_cpu_ref["N_pred_pre_squeeze"]] if debug_cpu_ref else None
        F0_pred = run_branch(self.F0_blocks, self.F0_proj_w, self.F0_proj_b, self.proj_n_pad, "F0", f0_refs)
        N_pred = run_branch(self.N_blocks, self.N_proj_w, self.N_proj_b, self.proj_n_pad, "N", n_refs)

        if debug_cpu_ref is not None:
            report_snr("\n[fpga][debug] --- Section 3 bisect (SNR vs CPU) ---")
            for name, snr_db in debug_log:
                flag = "  <-- LOW" if snr_db < 30.0 else ""
                report_snr(f"[fpga][debug] {name:38s} {snr_db:8.2f} dB{flag}")
            report_snr("[fpga][debug] --- end bisect ---\n")

        ue.reset_tensor_dram_addr()
        return F0_pred, N_pred


class TextEncoderFPGA:
    """Section 4: TextEncoder (phoneme embedding + Conv1d stack + BiLSTM) -> t_en [C, T].

    This is NOT a transformer -- Kokoro's only transformer is Section 1's PL-BERT. This is
    StyleTTS2's text encoder (kokoro_test.py:76-110):

        embedding(n_token, C) -> [T, C]
        3 x ( weight_norm Conv1d(C, C, kernel_size=5, padding=2) -> LayerNorm(C) -> LeakyReLU(0.2) )
        BiLSTM(C -> C/2 per direction, concat -> C)

    Two layout notes that make this cheaper than it looks:

    * The CPU module works channels-FIRST ([B, C, T]) and its `LayerNorm` (kokoro_test.py:62-73)
      transposes to channels-LAST, normalizes over C, and transposes back. Our convention is
      already [T, C], so that is a plain `layer_norm_core_dram(M=T, N=C)` -- no transpose at all,
      unlike Section 3's AdaIN which genuinely needed the [C, T] view for InstanceNorm.
    * `nn.Dropout(0.2)` is a no-op in eval.

    The CPU forward also `masked_fill_`s padded positions to 0 after every conv. With a single
    unbatched utterance the CPU mask is all-False, so there is nothing to mask there -- but OUR
    T is padded to a multiple of 64 for SRAM alignment, and those padded rows must be zero at the
    conv boundary. That is exactly what `_conv1d`'s `real_T` does, so it is threaded through every
    conv here. See [[hw-sram-alignment-rules]] reasoning in _conv1d's docstring.
    """

    # Borrowed wholesale from Section 3 rather than duplicated a third time -- these are plain
    # functions of (self.ue, self.H, self._up, self.identity_dram), all of which this class
    # provides. Section 2/3 keep their own copies; nothing there changes.
    _lstm_bidir = F0NPredictionFPGA._lstm_bidir
    _conv1d = F0NPredictionFPGA._conv1d
    _leaky_relu = F0NPredictionFPGA._leaky_relu
    _device_row_copy = F0NPredictionFPGA._device_row_copy
    _up = F0NPredictionFPGA._up
    _up_cap = F0NPredictionFPGA._up_cap
    _run = F0NPredictionFPGA._run

    def __init__(self, model, ue: UnifiedEngine):
        self.model = model
        self.ue = ue
        te = model.text_encoder

        self.C = te.embedding.embedding_dim            # 512
        self.H = self.C // 2                           # 256 per LSTM direction
        self.kernel_size = te.cnn[0][0].kernel_size[0] # 5
        self.pad = (self.kernel_size - 1) // 2         # 2
        self.depth = len(te.cnn)                       # 3

        self.emb_w = _bf16(te.embedding.weight)        # [n_token, C]
        self.cnn_blocks = []
        for blk in te.cnn:
            conv, ln = blk[0], blk[1]
            w, b = _wn_conv(conv)                      # materialize weight_norm (g, v) -> w
            self.cnn_blocks.append(dict(
                conv_w=w, conv_b=b,
                ln_g=_bf16(ln.gamma), ln_b=_bf16(ln.beta),
            ))
        self.lstm_w = dict(
            Wx_f=_bf16(te.lstm.weight_ih_l0), Wh_f=_bf16(te.lstm.weight_hh_l0),
            b_f=_bf16(te.lstm.bias_ih_l0 + te.lstm.bias_hh_l0),
            Wx_b=_bf16(te.lstm.weight_ih_l0_reverse), Wh_b=_bf16(te.lstm.weight_hh_l0_reverse),
            b_b=_bf16(te.lstm.bias_ih_l0_reverse + te.lstm.bias_hh_l0_reverse),
        )

    def forward(self, input_ids: torch.Tensor, T: int, debug_cpu_ref=None):
        """input_ids: [T] LongTensor of phoneme ids. Returns t_en [C, T] (channels-first, matching
        KokoroModel.forward_with_tokens's `t_en`, which Section 5 consumes as `t_en @ pred_aln_trg`).
        """
        ue = self.ue
        C, T_pad = self.C, _round_up(T, UE_VECTOR_SIZE)
        _set_row_cap(T_pad, T_CAP, "T_pad")
        T_pad = Dim(T_pad, GPR_TPAD, cap=T_CAP)
        # The real phoneme count drives the LSTM trip count and _conv1d's pad-row zeroing, so it
        # must be a register too (it was a bare int: baked loop count, and GPR_PADROWS was left
        # holding Section 3's frame pad count).
        T = Dim(T, GPR_T, cap=T_CAP)
        set_runtime_dims(T=int(T), T_pad=int(T_pad), pad_rows=int(T_pad) - int(T))
        row_bytes = C * 2
        debug_log = []

        def _check(name, dram_addr, shape, cpu_ref, rows=None):
            from user_dma_core import calculate_snr
            if debug_cpu_ref is None or cpu_ref is None:
                return
            _debug_flush(ue, self._run)
            got = ue.dma_from_accelerator_memory(dram_addr, shape).float()
            if rows is not None:
                got = got[:rows]
            debug_log.append((name, calculate_snr(cpu_ref.detach().float().reshape(-1), got.reshape(-1))))

        self.identity_dram = self._up(torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))

        # ---- phoneme embedding gather (same row-copy technique as Section 1's) ----
        # Zero-fill first, at emit time: the captured gather below writes ONLY rows 0..T-1, and no
        # other captured op touches rows T..T_pad, so the host zeros survive to execution. (Where a
        # captured op DOES rewrite the region, a host DMA would be silently clobbered -- see
        # _conv1d's real_T handling.) Zero is also what the CPU mask puts in padded positions.
        # Gathered HOST-side: the ids are known at capture time, so indexing the table here gives
        # the identical result with ZERO instructions -- this was the last emission in this section
        # that scaled with T. Rows T..T_pad stay zero, which is what the CPU mask puts in padded
        # positions. The [n_token, C] table no longer needs uploading at all.
        emb_padded = torch.zeros(T_pad, C, dtype=torch.bfloat16)
        emb_padded[:T] = self.emb_w[input_ids[:T].reshape(-1).long()]
        x_dram = self._up_cap(emb_padded, _cap() * C)
        _check("embedding", x_dram, (T_pad, C),
               debug_cpu_ref["embedding"] if debug_cpu_ref else None, rows=T)

        # ---- 3 x (Conv1d k=5 -> LayerNorm -> LeakyReLU(0.2)) ----
        for i, w in enumerate(self.cnn_blocks):
            ue.start_capture()
            conv_dram = ue.allocate_tensor_dram(_cap() * C * 2)
            self._conv1d(x_dram, T_pad, C, C, w["conv_w"], w["conv_b"], conv_dram,
                         kernel_size=self.kernel_size, pad=self.pad, real_T=T)
            ln_dram = ue.allocate_tensor_dram(_cap() * C * 2)
            _dyn_layernorm(ue, M=T_pad, N=C, A_DRAM_ADDR=conv_dram, OUTPUT_DRAM_ADDR=ln_dram,
                                    GAMMA_DRAM_ADDR=self._up(w["ln_g"]), BETA_DRAM_ADDR=self._up(w["ln_b"]))
            act_dram = ue.allocate_tensor_dram(_cap() * C * 2)
            self._leaky_relu(ln_dram, T_pad, C, act_dram)
            self._run(ue)
            x_dram = act_dram
            _check(f"cnn block {i}", x_dram, (T_pad, C),
                   debug_cpu_ref["cnn"][i] if debug_cpu_ref else None, rows=T)

        # ---- BiLSTM (C -> 2 * C/2 = C) ----
        # Run over the REAL T only: this is a sequential bidirectional pass, so starting the
        # backward direction inside the alignment padding would corrupt every real timestep.
        out_dram = ue.allocate_tensor_dram(_cap() * C * 2)
        ue.dma_to_accelerator_memory(out_dram, torch.zeros(T_pad * C, dtype=torch.bfloat16))
        ue.start_capture()
        self._lstm_bidir(x_dram, T, C, self.lstm_w, out_dram)
        self._run(ue)
        _check("lstm output", out_dram, (T_pad, C),
               debug_cpu_ref["lstm"] if debug_cpu_ref else None, rows=T)

        if debug_cpu_ref is not None:
            report_snr("\n[fpga][debug] --- Section 4 bisect (SNR vs CPU) ---")
            for name, snr_db in debug_log:
                flag = "  <-- LOW" if snr_db < 30.0 else ""
                report_snr(f"[fpga][debug] {name:38s} {snr_db:8.2f} dB{flag}")
            report_snr("[fpga][debug] --- end bisect ---\n")

        # CPU returns [B, C, T] (channels-first); transpose our [T, C] to match.
        t_en = ue.dma_from_accelerator_memory(out_dram, (T_pad, C)).float()[:T].T.contiguous()
        ue.reset_tensor_dram_addr()
        return t_en




# Reserved ISA registers for the dynamic (PBI) kernel dimensions.
#
# gpr_M_reg is a PBI LOOP-COUNT field and is validated to 1..15 (user_dma_core.py:2255, 2482,
# 3007); only gpr_N_reg/gpr_K_reg and the address registers accept 1..63. So these must be low --
# and low is exactly where alloc_isa_reg() hands registers out from. _instrument() therefore burns
# the first three allocations permanently, so every later alloc starts at 4 and can never collide.
# Safe because nothing in the repo calls reset_isa_reg_counter() and start_capture() leaves the
# counter alone, so the reservation survives every capture in the run.
GPR_M, GPR_K, GPR_N = 1, 2, 3

# --- Runtime dimension registers (primed by the per-run PREAMBLE, never by the body) ----------
#
# The frozen-bin contract: a captured program may contain ADD_SET only for values that are the
# same for every prompt. K and N are channel widths, so their ADD_SETs are already prompt-
# invariant and stay in the body. M is a sequence length and must NOT be.
#
# Only two quantities are irreducible at runtime: the phoneme count T and the frame count
# n_frames. Every other varying M in this model is an affine function of one of them --
# `T`, `n_frames`, or `count * (C // 64)` for a flattened eltwise. Those are derived INSIDE the
# body with REG_ALU instructions whose immediates are channel constants, so the bytes are
# identical across prompts while the values are not.
#
# Register budget: gpr_M_reg is a PBI loop-count field validated to 1..15, so scratch row-count
# registers must stay low. GPR_T/GPR_F are used directly as gpr_M_reg when M is exactly the count,
# hence they too must be <= 15.
GPR_T = 4        # real phoneme count      (preamble-primed)
GPR_F = 5        # real frame count        (preamble-primed)
GPR_TMP = 6      # address / derivation scratch
GPR_MSCRATCH = (7, 8, 9)   # derived row counts; round-robin so nested derivations don't alias
# T rounded up to the 64-element vector size. Not derivable from GPR_T in one instruction
# (it needs add+shr+shl), and Section 1 uses it on every one of its 144 attention calls, so it
# earns a register of its own rather than being recomputed.
GPR_TPAD = 10
# Frame count rounded up to the vector size. Sections 3 and 5a both work in padded frames
# (Section 5a's `T` IS the frame count), so they share this register.
GPR_NFPAD = 11
# (padded frames - real frames), the conv zero-pad row count. NOT derivable from the other two in
# one instruction -- it is a difference of two registers, and the ISA's reg-ALU takes an immediate,
# not a second register, for subtraction. Every conv that needs it wants (nf_pad - n_frames)
# scaled by that level's upsample factor, so one primed register covers all of them.
GPR_PADROWS = 12

_RESERVED_GPRS = (GPR_M, GPR_K, GPR_N, GPR_T, GPR_F, GPR_TMP, GPR_TPAD, GPR_NFPAD,
                  GPR_PADROWS) + GPR_MSCRATCH
_NREG_POOL_SIZE = 12   # 6 distinct normalisation axes (N reg + sqrt(N) reg each)

# Registers carrying a runtime NORMALISATION length. InstanceNorm reduces over the TIME axis, so
# its length lands in the kernel's N, not its M -- no row register can express it. The dynamic
# layer-norm core takes the real length in gpr_N_reg and sqrt(real length) as a bf19 RSQRT scalar
# in gpr_sqrt_n_reg, with N itself baked at the cap.
#
# Keyed by the axis CAP rather than by the live length: the cap is the same for every prompt, so
# a given call site always resolves to the same pair of registers, while the VALUES written into
# them differ per run. Allocated above 15 because neither is a loop-count field.
# Reserved at instrument time alongside the dimension registers. These MUST come out of the
# permanently-reserved block: alloc_isa_reg() hands the cores their own private counters starting
# just past it, so a hardcoded index here would collide with the layer-norm core's `remaining` /
# `rows_take` loop counters. That collision is not a crash -- the core's row loop simply exits
# early and the output buffer is left part-written.
_NREG_POOL = []
_NREG_MAP = {}


def _n_regs(cap):
    """(gpr_N_reg, gpr_sqrt_n_reg) for a normalisation axis with this cap. Stable across prompts."""
    if cap not in _NREG_MAP:
        k = 2 * len(_NREG_MAP)
        assert k + 1 < len(_NREG_POOL), (
            f"ran out of reserved registers for runtime normalisation lengths "
            f"({len(_NREG_POOL)} reserved); raise _NREG_POOL_SIZE")
        _NREG_MAP[cap] = (_NREG_POOL[k], _NREG_POOL[k + 1])
    return _NREG_MAP[cap]


class Dim(int):
    """A sequence dimension: its value for THIS run, plus how to rebuild it on device.

    Subclasses ``int`` deliberately. A dimension is used two ways in this file -- as a number for
    host-side arithmetic (allocation sizes, byte offsets, ``range()`` over timesteps) and as the
    row count handed to a kernel. Only the second use needs to become a register. Inheriting from
    ``int`` lets a Dim flow through every helper unchanged and be picked out by ``isinstance``
    exactly where it matters.

    ``base`` is None for a compile-time constant (baked, as before), or GPR_T / GPR_F for a value
    the preamble primes. The runtime value is ``reg[base] * mul``.

    Multiplication by an integer preserves the base, because ``T * (C // 64)`` is still an affine
    function of T and is precisely the flattened-eltwise row count. Every other operation falls
    back to plain int, which is what a non-affine derivative should be.
    """

    def __new__(cls, value, base=None, mul=1, cap=None, off=0):
        self = super().__new__(cls, int(value))
        self.base = base
        self.mul = int(mul)
        # Affine offset: runtime value = reg[base] * mul + off. The generator's reflection pad
        # makes its row count 120*n_frames + 1, which no (base, mul) pair can express.
        self.off = int(off)
        # The value a kernel is TEMPLATED at. Cores shape their body from the compile-time M --
        # layer_norm_core_dram_dynamic picks its chunk height as min(M, 16), so M=7 emits a
        # 51-instruction body and M=66 a 105-instruction one even though both drive the trip
        # count from a register. Templating at the cap makes the body identical for every prompt;
        # the GPR still decides how much of it actually runs.
        self.cap = int(cap) if cap is not None else int(value)
        return self

    def __mul__(self, k):
        if isinstance(k, int) and not isinstance(k, Dim) and self.base is not None:
            return Dim(int(self) * k, self.base, self.mul * k, self.cap * k, self.off * k)
        return int(self) * int(k)

    __rmul__ = __mul__

    def __add__(self, k):
        """Adding a constant keeps provenance (it lands in ``off``); Dim + Dim degrades to int."""
        if isinstance(k, int) and not isinstance(k, Dim) and self.base is not None:
            return Dim(int(self) + k, self.base, self.mul, self.cap + k, self.off + k)
        return int(self) + int(k)

    __radd__ = __add__

    def __floordiv__(self, k):
        """Exact division keeps provenance (rows = T*C // 64 with C a multiple of 64); anything
        else degrades to a plain int, which then bakes -- the fingerprint diff will show it."""
        if (isinstance(k, int) and not isinstance(k, Dim) and self.base is not None
                and self.mul % k == 0 and self.off % k == 0 and int(self) % k == 0
                and self.cap % k == 0):
            return Dim(int(self) // k, self.base, self.mul // k, self.cap // k, self.off // k)
        return int(self) // int(k)

    def __repr__(self):
        return f"Dim({int(self)}, base={self.base}, mul={self.mul}, off={self.off})"


def _as_dim(m):
    return m if isinstance(m, Dim) else Dim(m)


_MSCRATCH_RR = [0]


def _emit_M(ue, dim):
    """Materialise ``dim`` into a register usable as gpr_M_reg, returning its index.

    Constant dims still go through ADD_SET (prompt-invariant, so frozen-safe). Derived dims are
    computed from their preamble-primed base with REG_ALU, whose immediate is a channel constant.
    """
    dim = _as_dim(dim)
    if dim.base is None:
        ue.generate_instruction_add_set(GPR_M, int(dim))
        return GPR_M
    if dim.mul == 1 and dim.off == 0:
        return dim.base
    reg = GPR_MSCRATCH[_MSCRATCH_RR[0] % len(GPR_MSCRATCH)]
    _MSCRATCH_RR[0] += 1
    if dim.mul == 1:
        ue.generate_instruction_add_imm(dim.base, dim.off, reg)
        return reg
    # reg_mul_imm zero-extends a 16-bit immediate (user_dma_core.py:9241).
    assert 0 < dim.mul <= 0xFFFF, f"affine coefficient {dim.mul} exceeds the 16-bit immediate field"
    ue.generate_instruction_reg_mul_imm(reg, dim.base, dim.mul)
    if dim.off:
        assert dim.off > 0, "negative affine offsets are not supported (immediate is unsigned)"
        ue.generate_instruction_add_imm(reg, dim.off, reg)
    return reg


def _emit_tiles(ue, dim, dst):
    """Emit ceil(dim / 64) into ``dst`` -- the 64-element tile count ("bucket index").

    The hardware has no divide, so this is (x + 63) >> 6, which is exact for the vector size.
    Llama primes this host-side as its own GPR; deriving it keeps the preamble to two writes.
    """
    dim = _as_dim(dim)
    if dim.base is None:
        ue.generate_instruction_add_set(dst, -(-int(dim) // UE_VECTOR_SIZE))
        return dst
    src = _emit_M(ue, dim)
    ue.generate_instruction_add_imm(src, UE_VECTOR_SIZE - 1, dst)
    ue.generate_instruction_shr(dst, dst, 6)
    return dst




# Below this many output rows the dynamic path is a net loss: it costs 3 ADD_SETs to prime the
# dimension registers, while the legacy path emits one matvec per row. The LSTM's per-timestep
# gate matmuls are all M=1, so they stay on the legacy path.
_DYN_M_MIN = 8


def _template(m):
    """The M a kernel should be CAPTURED at: the cap for a runtime dim, the value otherwise."""
    return m.cap if _is_dyn(m) else int(m)


def _is_dyn(m):
    """True when ``m`` is a runtime dimension, whatever its value happens to be this run.

    The small-M shortcut below must never be taken on a dynamic dimension: choosing the legacy
    path for T=7 and the dynamic path for T=66 emits two DIFFERENT programs for the same model,
    which is precisely what the frozen bin forbids. Provenance decides the path; the value only
    decides the template.
    """
    return isinstance(m, Dim) and m.base is not None


def _dyn_matmul(ue, M, K, N, **kw):
    """matmat_mul_core with the dimensions sourced from ISA registers.

    WHY: the legacy kernel emits ONE matvec instruction per output row
    (user_dma_core.py:5473, `for output_row in range(m_take)`), so a matmul over M=320 frames
    costs 320 instructions to emit. That is kokoro's dominant compile cost -- and it is also why
    the program is prompt-specific, since M is baked in. matmat_mul_core dispatches to
    matmat_mul_core_dynamic as soon as a dimension GPR is set (user_dma_core.py:5234), turning the
    software tile loops into ISA while-loops: O(1) instructions regardless of M.

    The dynamic kernel requires ALL THREE dims as registers (user_dma_core.py:5657), so K and N
    are primed too even though they are compile-time constants here.
    """
    Mv = int(M)
    if Mv < _DYN_M_MIN and not _is_dyn(M):
        return ue.matmat_mul_core(M=Mv, K=K, N=N, **kw)
    Mv = _template(M)
    m_reg = _emit_M(ue, M)
    # K and N are usually channel widths (constant immediates, frozen-safe as ADD_SETs). When one
    # is a sequence length -- Section 3's alignment matmul contracts over the phoneme axis -- it
    # arrives as a Dim and is copied out of its preamble register instead.
    _emit_into(ue, K, GPR_K)
    _emit_into(ue, N, GPR_N)
    return ue.matmat_mul_core(M=Mv, K=_template(K), N=_template(N),
                              gpr_M_reg=m_reg, gpr_K_reg=GPR_K, gpr_N_reg=GPR_N, **kw)


def _emit_into(ue, dim, dst):
    """Materialise ``dim`` into a SPECIFIC register (constants via ADD_SET, Dims from their base)."""
    dim = _as_dim(dim)
    if dim.base is None:
        ue.generate_instruction_add_set(dst, int(dim))
    elif dim.mul == 1:
        ue.generate_instruction_add_imm(dim.base, dim.off, dst)
    else:
        assert 0 < dim.mul <= 0xFFFF, dim.mul
        ue.generate_instruction_reg_mul_imm(dst, dim.base, dim.mul)
        if dim.off:
            ue.generate_instruction_add_imm(dst, dim.off, dst)



def _dyn_eltwise(ue, M, *args, **kw):
    """eltwise_core_dram with a runtime row register.

    eltwise_core_dram_legacy Python-unrolls its vertical tiles, so emission is O(M). That bites
    hardest in _leaky_relu, which flattens to M = numel/64 -- at the decoder's T=384, C=1152 that
    is M=6912, i.e. thousands of instructions for one activation. The dynamic core turns the tile
    loop into an ISA loop; passing gpr_M_reg alone is enough, the entrypoint allocates and seeds
    any other register it needs (user_dma_core.py:2049).
    """
    if (_is_dyn(M) or int(M) >= _DYN_M_MIN) and kw.get("gpr_M_reg") is None:
        kw["gpr_M_reg"] = _emit_M(ue, M)
        M = _template(M)
    return ue.eltwise_core_dram(int(M), *args, **kw)


_ZEROS_BUF = {}


def _zeros_buf(ue, n):
    """A shared, properly ALLOCATED zeros vector for the layer-norm core's reduction band.

    Without one the core takes get_params_dram_addr() as scratch and writes there without
    allocating, so the next constant upload silently lands on top of it. One buffer per length,
    in tensor DRAM, reused by every call.
    """
    a = _ZEROS_BUF.get(n)
    if a is None:
        a = ue.allocate_tensor_dram(n * 2)
        ue.dma_to_accelerator_memory(a, torch.zeros(n, dtype=torch.bfloat16))
        _ZEROS_BUF[n] = a
    return a


def _dyn_layernorm(ue, *args, **kw):
    """layer_norm_core_dram with a runtime row register -- its legacy path emits one instruction
    per row (user_dma_core.py:3973). Missing M/N/sqrt(N) registers are seeded by the entrypoint."""
    M = kw.get("M", args[0] if args else None)
    if M is not None and (_is_dyn(M) or int(M) >= _DYN_M_MIN) and kw.get("gpr_M_reg") is None:
        kw["gpr_M_reg"] = _emit_M(ue, M)
        # Write the CAP back to wherever M actually reaches the core, so the body is shaped by
        # the cap rather than by this prompt's length. A bare local assignment here is a no-op.
        if "M" in kw:
            kw["M"] = _template(M)
        else:
            args = (_template(M),) + tuple(args[1:])
        _N = kw.get("N", args[1] if len(args) > 1 else None)
        if _N is not None and kw.get("ZEROS_DRAM_ADDR") is None:
            kw["ZEROS_DRAM_ADDR"] = _zeros_buf(ue, int(_N))
    return ue.layer_norm_core_dram(*args, **kw)


def _dyn_transpose(ue, *args, **kw):
    """bf16_transpose_core with runtime dims -- legacy emits one matvec per output row
    (user_dma_core.py:6581)."""
    M = kw.get("M", args[0] if args else None)
    if M is not None and (_is_dyn(M) or int(M) >= _DYN_M_MIN) and kw.get("gpr_M_reg") is None:
        kw["gpr_M_reg"] = _emit_M(ue, M)
        # Write the CAP back to wherever M actually reaches the core, so the body is shaped by
        # the cap rather than by this prompt's length. A bare local assignment here is a no-op.
        if "M" in kw:
            kw["M"] = _template(M)
        else:
            args = (_template(M),) + tuple(args[1:])
    return ue.bf16_transpose_core(*args, **kw)


def _dyn_activation(ue, *args, **kw):
    """activation_core with a runtime row register (it applies the activation through the same
    identity-matmul path, so it inherits the per-row emission)."""
    M = kw.get("M", args[0] if args else None)
    if M is not None and (_is_dyn(M) or int(M) >= _DYN_M_MIN) and kw.get("gpr_M_reg") is None:
        kw["gpr_M_reg"] = _emit_M(ue, M)
        # Write the CAP back to wherever M actually reaches the core, so the body is shaped by
        # the cap rather than by this prompt's length. A bare local assignment here is a no-op.
        if "M" in kw:
            kw["M"] = _template(M)
        else:
            args = (_template(M),) + tuple(args[1:])
    return ue.activation_core(*args, **kw)



def _dyn_attention(ue, **kw):
    """unified_attention_core with its dimensions in registers.

    Section 1 calls this 144 times (12 heads x 12 layers) at batch = aligned_seq_len = T_pad, and
    the legacy path emits per-row work each time -- which is why Section 1 was the largest section
    (171k instructions at T=138) despite having no per-timestep loop of its own. The GPR parameters
    already existed (user_dma_core.py:7164); nothing was passing them.
    """
    batch = kw.get("batch")
    aligned = kw.get("aligned_seq_len")
    if batch is not None and (_is_dyn(batch) or int(batch) >= _DYN_M_MIN) and kw.get("gpr_batch_reg") is None:
        b_reg = _emit_M(ue, batch)
        kw["gpr_batch_reg"] = b_reg
        kw["batch"] = _template(batch)
        if aligned is not None:
            kw["aligned_seq_len"] = _template(aligned)
        if aligned is not None and int(aligned) == int(batch):
            # Already 64-aligned (Section 1 works in T_pad), so the two are the same register.
            kw["gpr_aligned_seq_len_reg"] = b_reg
        else:
            # aligned_seq_len is the 64-aligned batch, i.e. tiles*64 -- derived, not primed.
            _emit_tiles(ue, batch, GPR_K)
            ue.generate_instruction_shl(GPR_K, GPR_K, 6)
            kw["gpr_aligned_seq_len_reg"] = GPR_K
    return ue.unified_attention_core(**kw)


def _pbi_row_loop(ue, n_rows, reads, writes, gpr_rows=None, sram_addr=0x00000,
                  row_start=None, row_start_off=0):
    """One HARDWARE loop over `n_rows` rows instead of a Python-unrolled emission.

    `reads` / `writes` are lists of ``(dram_base, row_bytes, elems, sram_off)``. Each iteration
    stages every read into SRAM at its ``sram_off``, then writes every entry back out; both the
    source and destination addresses are recomputed per iteration into a scratch register, so the
    strides are independent and arbitrary.

    WHY: the Python loops this replaces emit 2-4 instructions PER ROW, which is what makes a
    kokoro program prompt-specific and slow to compile -- Section 4 emitted 18,430 instructions
    for 49 phonemes, almost all of it LSTM/row-copy unroll, and compile time runs ~6x the
    accelerator's own execution time. A hardware loop emits a fixed ~8 instructions regardless of
    row count. Same mechanism gemma3 uses (see _emit_strided_copy_pbi, gemma3_test.py:570).

    ``gpr_rows``: GPR index holding the trip count at RUNTIME. When given, the emitted program is
    independent of the row count and one bin serves any prompt. When ``None``, the count is baked
    (still a hardware loop, so the instruction-count win holds, but the bin stays prompt-specific).

    Addresses are computed in UE word units (byte >> 3), so every ``row_bytes`` must be a multiple
    of 8 -- true here since row widths are 64-aligned bf16, i.e. multiples of 128 bytes.

    ``relative=True`` (loop_start's default) closes with a backward RELA_JNZ, which is bounded by
    a 512-instruction i-cache window; these bodies are under a dozen instructions, well inside it.
    """
    for base, row_bytes, _elems, _off in list(reads) + list(writes):
        assert row_bytes % 8 == 0, f"row_bytes={row_bytes} must be a multiple of 8 (word-addressed)"
        assert base % 8 == 0, f"dram base 0x{base:X} must be 8-byte aligned"
    # A Dim carries how to rebuild the trip count on device, so the caller does not have to
    # thread a register index down by hand -- passing the Dim IS the dynamic request.
    if gpr_rows is None and isinstance(n_rows, Dim) and n_rows.base is not None:
        gpr_rows = _emit_M(ue, n_rows)
        n_rows = _template(n_rows)
    n_rows = int(n_rows)
    i_reg = ue.alloc_isa_reg()
    t_reg = ue.alloc_isa_reg()
    # ``row_start`` (a Dim, optional) + ``row_start_off`` (constant) is the first row index, so a
    # copy that begins at a runtime row (e.g. the pad after the real frames) needs no baked offset.
    if row_start is not None and _is_dyn(row_start):
        ue.generate_instruction_add_imm(_emit_M(ue, row_start), int(row_start_off), i_reg)
    else:
        ue.generate_instruction_add_set(i_reg, (int(row_start) if row_start is not None else 0)
                                        + int(row_start_off))
    ue.loop_start(loop_cnt=n_rows, gpr_loop_cnt=gpr_rows)
    for base, row_bytes, elems, off in reads:
        ue.generate_instruction_reg_mul_imm(t_reg, i_reg, ue_35bit_addr_shifter(row_bytes))
        ue.generate_instruction_add_imm(t_reg, ue_35bit_addr_shifter(base), t_reg)
        ue.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=sram_addr + off,
                                      element_size=elems, general_reg_src=t_reg)
    for base, row_bytes, elems, off in writes:
        ue.generate_instruction_reg_mul_imm(t_reg, i_reg, ue_35bit_addr_shifter(row_bytes))
        ue.generate_instruction_add_imm(t_reg, ue_35bit_addr_shifter(base), t_reg)
        ue.sram_to_accelerator_memory(sram_address=sram_addr + off, accelerator_dram_address=0,
                                      element_size=elems, general_reg_src=t_reg)
    ue.generate_instruction_add_inc(i_reg)
    ue.loop_end()
    ue.release_isa_reg()   # t_reg
    ue.release_isa_reg()   # i_reg



def _zero_rows_from(ue, dst, row_bytes, elems, start, count_cap, zeros_src, sram_addr=0x00000):
    """Zero rows [start, count_cap) of a row-major buffer with ONE hardware loop.

    ``start`` is a Dim: its register is the first row to clear and the trip count is
    ``count_cap - start``, derived on device, so the program is prompt-independent. The trip count
    must be >= 1 (a zero count wraps the loop counter); callers assert ``start < count_cap``.
    """
    assert row_bytes % 8 == 0 and dst % 8 == 0
    start_reg = _emit_M(ue, start)
    i_reg = ue.alloc_isa_reg()
    t_reg = ue.alloc_isa_reg()
    ue.accelerator_memory_to_sram(accelerator_dram_address=zeros_src, sram_address=sram_addr,
                                  element_size=elems)
    ue.generate_instruction_add_set(GPR_TMP, int(count_cap))
    ue.generate_instruction_reg_sub(GPR_TMP, GPR_TMP, start_reg)        # count = cap - start
    ue.generate_instruction_add_imm(start_reg, 0, i_reg)                 # i = start
    ue.loop_start(loop_cnt=int(count_cap), gpr_loop_cnt=GPR_TMP)
    ue.generate_instruction_reg_mul_imm(t_reg, i_reg, ue_35bit_addr_shifter(row_bytes))
    ue.generate_instruction_add_imm(t_reg, ue_35bit_addr_shifter(dst), t_reg)
    ue.sram_to_accelerator_memory(sram_address=sram_addr, accelerator_dram_address=0,
                                  element_size=elems, general_reg_src=t_reg)
    ue.generate_instruction_add_inc(i_reg)
    ue.loop_end()
    ue.release_isa_reg()   # t_reg
    ue.release_isa_reg()   # i_reg


def _pad_dim(t: torch.Tensor, dim: int, target: int) -> torch.Tensor:
    """Zero-pad `t` along `dim` up to `target`. Used to bring a CHANNEL count up to a multiple of
    UE_VECTOR_SIZE. Padding a conv weight's INPUT-channel axis is exact: those taps multiply the
    padded (don't-care) channels by zero, so whatever garbage sits there contributes nothing --
    which is why the activation buffers themselves never need re-zeroing on the channel axis.
    """
    if t.shape[dim] == target:
        return t
    assert t.shape[dim] < target, (t.shape, dim, target)
    shape = list(t.shape)
    shape[dim] = target - t.shape[dim]
    return torch.cat([t, torch.zeros(*shape, dtype=t.dtype)], dim=dim)


def _tile_pad_channels(x: torch.Tensor, c_real: int) -> None:
    """In-place: fill x[:, c_real:]'s alignment-padding channels by tiling the real ones.

    They must not be left at zero. AdaIN normalizes per channel over time, so an all-zero channel
    has zero variance, and the hardware RSQRT has no epsilon -- `0 * rsqrt(0)` = NaN, which then
    survives the zero conv taps that were supposed to discard the channel. Tiling gives the same
    finite statistics a real channel has; the zero taps still make the contribution exactly zero,
    so this changes nothing numerically about the real output.
    """
    c_pad = x.shape[1]
    if c_pad <= c_real:
        return
    n = c_pad - c_real
    reps = (n + c_real - 1) // c_real
    x[:, c_real:] = x[:, :c_real].repeat(1, reps)[:, :n]


class DecoderFPGA:
    """Section 5a: the Decoder FRONT -- everything up to (not including) the ISTFTNet Generator.

        F0 = F0_conv(F0_curve);  N = N_conv(N_curve)          (1->1 ch, k=3, stride=2)
        x  = encode( cat([asr, F0, N]) )                       AdainResBlk1d 514 -> 1024
        for 4 decode blocks:
            x = block( cat([x, asr_res, F0, N]) )              AdainResBlk1d 1090 -> 1024 (x3),
                                                               then 1090 -> 512 with upsample
        audio = generator(x, s, F0_curve)                      <-- still CPU, see below

    Every block here is the SAME AdainResBlk1d that Section 3 already runs, so this reuses
    `_adain_res_blk` and friends outright; the new mechanics are all layout:

    CHANNEL-AXIS ALIGNMENT (first time this bites -- Sections 1-4 only had to pad the TIME axis).
    The concatenations produce 512+1+1 = 514 and 1024+64+1+1 = 1090 channels, neither a multiple
    of 64. Channels land on `_conv1d`'s K and on `_adain1d`'s transpose M, so they must be padded
    (-> 576 and 1152). Padding is applied to the WEIGHTS' input-channel axis, so a zero tap
    discards each don't-care channel's contribution.

    That is necessary but NOT sufficient: the padded channels must also carry FINITE values.
    AdaIN normalizes per channel over time, so an all-zero padded channel has zero variance, and
    the hardware RSQRT path has no epsilon (user_dma_core.py:3864) -- `0 * rsqrt(0)` = `0 * inf`
    = NaN. NaN times a zero tap is still NaN, so weight padding alone cannot rescue it and the
    whole section reads -inf dB. The padded channels are therefore TILED from real channels:
    nonzero variance keeps the normalizer finite, and the zero taps still discard the result.

    STILL ON CPU, deliberately:
      * `F0_conv` / `N_conv` -- 1-in/1-out channel, k=3, stride=2 over ~506 samples. K=1 would have
        to be padded to 64 to reach the accelerator, i.e. 64x waste on a rounding-error amount of
        compute, and F0/N already arrive as host tensors from Section 3, so this costs no extra
        round trip.
      * `asr = t_en @ pred_aln_trg` and `asr_res` -- same reasoning, both operands already host-side.
      * The GENERATOR (Section 5b-5d). Not a blocker, just not written yet: the op gaps are already
        proven (`sincos_bounded_poly_test` for Snake1D/phase, `exp_via_sigmoid_test` for `spec`,
        `conv_transpose1d_zero_insert_test` for `ups`). The one genuinely open question is
        SineGen's UNBOUNDED accumulating phase -- the Taylor fit is proven on [-1,1] only.
    """

    _adain1d = F0NPredictionFPGA._adain1d
    _adain_res_blk = F0NPredictionFPGA._adain_res_blk
    _conv1d = F0NPredictionFPGA._conv1d
    _leaky_relu = F0NPredictionFPGA._leaky_relu
    _nearest_upsample2x = F0NPredictionFPGA._nearest_upsample2x
    _depthwise_convtranspose_upsample2x = F0NPredictionFPGA._depthwise_convtranspose_upsample2x
    _concat_columns = ProsodyDurationFPGA._concat_columns   # lives on Section 2, not Section 3
    _device_row_copy = F0NPredictionFPGA._device_row_copy
    _up = F0NPredictionFPGA._up
    _up_cap = F0NPredictionFPGA._up_cap
    _run = F0NPredictionFPGA._run

    def __init__(self, model, ue: UnifiedEngine):
        self.model = model
        self.ue = ue
        dec = model.decoder
        self.dec = dec
        self.style_dim = dec.encode.norm1.fc.in_features       # 128

        self.C_asr = dec.asr_res[0].in_channels                # 512
        self.C_res = dec.asr_res[0].out_channels               # 64
        self.C_enc_in = self.C_asr + 2                         # 514
        self.C_enc_out = dec.encode.conv1.out_channels         # 1024
        self.C_dec_in = self.C_enc_out + self.C_res + 2        # 1090
        self.C_enc_in_pad = _round_up(self.C_enc_in, UE_VECTOR_SIZE)   # 576
        self.C_dec_in_pad = _round_up(self.C_dec_in, UE_VECTOR_SIZE)   # 1152

        def adain_w(m, C_real, C_pad):
            """AdaIN1d fc -> [2*C_pad, style_dim], laid out gamma-block then beta-block so that
            _adain1d's `gb[:C]` / `gb[C:]` split still works at the PADDED width."""
            w, b = m.fc.weight, m.fc.bias
            g_w, be_w = w[:C_real], w[C_real:]
            g_b, be_b = b[:C_real], b[C_real:]
            pad_w = torch.zeros(C_pad - C_real, w.shape[1], dtype=w.dtype)
            pad_b = torch.zeros(C_pad - C_real, dtype=b.dtype)
            return (_bf16(torch.cat([g_w, pad_w, be_w, pad_w], dim=0)),
                    _bf16(torch.cat([g_b, pad_b, be_b, pad_b], dim=0)))

        def block_w(m, C_in_real, C_in_pad):
            w = {}
            c1w, c1b = _wn_conv(m.conv1)
            c2w, c2b = _wn_conv(m.conv2)
            w["conv1_w"] = _bf16(_pad_dim(c1w, 1, C_in_pad))   # [Cout, Cin_pad, 3]
            w["conv1_b"] = _bf16(c1b)
            w["conv2_w"], w["conv2_b"] = _bf16(c2w), _bf16(c2b)
            w["norm1_fc_w"], w["norm1_fc_b"] = adain_w(m.norm1, C_in_real, C_in_pad)
            w["norm2_fc_w"], w["norm2_fc_b"] = adain_w(m.norm2, m.conv2.out_channels,
                                                        m.conv2.out_channels)
            if m.learned_sc:
                sc_w, _ = _wn_conv(m.conv1x1)
                w["conv1x1_w"] = _bf16(_pad_dim(sc_w, 1, C_in_pad))
            if m.upsample_type != "none":
                pw, pb = _wn_conv(m.pool)                       # depthwise: [Cin, 1, 3]
                w["pool_w"] = _bf16(_pad_dim(pw, 0, C_in_pad))
                w["pool_b"] = _bf16(_pad_dim(pb, 0, C_in_pad))
            return w

        self.encode_w = block_w(dec.encode, self.C_enc_in, self.C_enc_in_pad)
        self.decode_w = [block_w(b, self.C_dec_in, self.C_dec_in_pad) for b in dec.decode]

    def forward(self, asr: torch.Tensor, F0_curve: torch.Tensor, N_curve: torch.Tensor,
                s: torch.Tensor, debug_cpu_ref=None):
        """asr: [C_asr, T] host. F0_curve/N_curve: [2T] host (Section 3's output, pre-F0_conv).
        s: [style_dim] host. Returns x [C_out, 2T] host -- the Generator's input.
        """
        ue = self.ue
        dec = self.dec
        debug_log = []

        def _check(name, dram_addr, shape, cpu_ref, rows=None):
            from user_dma_core import calculate_snr
            if debug_cpu_ref is None or cpu_ref is None:
                return
            _debug_flush(ue, self._run)
            got = ue.dma_from_accelerator_memory(dram_addr, shape).float()
            if rows is not None:
                got = got[:rows]
            debug_log.append((name, calculate_snr(cpu_ref.detach().float().reshape(-1), got.reshape(-1))))

        with torch.no_grad():
            F0 = dec.F0_conv(F0_curve.unsqueeze(0).unsqueeze(1)).squeeze(0)   # [1, T]
            Nc = dec.N_conv(N_curve.unsqueeze(0).unsqueeze(1)).squeeze(0)     # [1, T]
            asr_res = dec.asr_res(asr.unsqueeze(0)).squeeze(0)                # [C_res, T]

        T = asr.shape[-1]
        T_pad = _round_up(T, UE_VECTOR_SIZE)
        _set_row_cap(2 * T_pad, 2 * F_CAP, "2 x padded asr frames")
        # Section 5a's sequence axis is TIME, not phonemes: this T is n_frames.
        T_pad = Dim(T_pad, GPR_NFPAD, cap=F_CAP)
        # T here IS the frame count -- the same runtime quantity Section 3 calls n_frames, so it
        # shares GPR_F. It reaches the device as _conv1d's real_T, whose `T - real_T` zero-pad row
        # count would otherwise be baked into the captured row copy.
        T = Dim(T, GPR_F, cap=F_CAP)
        set_runtime_dims(n_frames=int(T), nf_pad=int(T_pad),
                         pad_rows=int(T_pad) - int(T))
        self.identity_dram = self._up(torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))
        # Per-RUN input (the voice pack is indexed by phoneme count), so it must land at a fixed
        # activation address every run -- never in the content-hashed constant memo, whose
        # address would move with the prompt while the frozen program keeps reading the old one.
        style_dram = self._up_cap(_bf16(s), _bf16(s).numel())

        # ---- encode: x0 = cat([asr, F0, N]) built host-side in one upload ----
        x0 = torch.zeros(T_pad, self.C_enc_in_pad, dtype=torch.bfloat16)
        x0[:T, :self.C_asr] = _bf16(asr.T)
        x0[:T, self.C_asr] = _bf16(F0[0])
        x0[:T, self.C_asr + 1] = _bf16(Nc[0])
        _tile_pad_channels(x0, self.C_enc_in)
        x_dram = self._up_cap(x0, _cap() * self.C_enc_in_pad)

        cur_dram, cur_C = self._adain_res_blk(
            x_dram, T_pad, self.C_enc_in_pad, self.C_enc_out, False, self.encode_w,
            style_dram, real_T=T)[0], self.C_enc_out
        _check("encode", cur_dram, (T_pad, self.C_enc_out),
               debug_cpu_ref["encode"] if debug_cpu_ref else None, rows=T)

        # The host-side half of every decode concat: [asr_res | F0 | N | zero-pad], one upload,
        # laid out so its width (C_dec_in_pad - C_enc_out) is itself 64-aligned and _concat_columns
        # can stitch it onto the device-resident x without any sub-64-wide copies.
        side_w = self.C_dec_in_pad - self.C_enc_out
        side = torch.zeros(T_pad, side_w, dtype=torch.bfloat16)
        side[:T, :self.C_res] = _bf16(asr_res.T)
        side[:T, self.C_res] = _bf16(F0[0])
        side[:T, self.C_res + 1] = _bf16(Nc[0])
        _tile_pad_channels(side, self.C_res + 2)
        side_dram = self._up_cap(side, _cap() * side_w)

        cur_T, real_T = T_pad, T
        res = True
        for i, (blk, w) in enumerate(zip(dec.decode, self.decode_w)):
            if res:
                ue.start_capture()
                cat_dram = ue.allocate_tensor_dram(_cap() * self.C_dec_in_pad * 2)
                self._concat_columns(cur_dram, self.C_enc_out, side_dram, side_w, cur_T, cat_dram)
                self._run(ue)
                cur_dram, cur_C = cat_dram, self.C_dec_in_pad
            up = blk.upsample_type != "none"
            cur_dram, cur_T = self._adain_res_blk(
                cur_dram, cur_T, cur_C, blk.conv2.out_channels, up, w, style_dram, real_T=real_T)
            cur_C = blk.conv2.out_channels
            real_T = 2 * real_T if up else real_T
            if up:
                res = False
            _check(f"decode {i}", cur_dram, (cur_T, cur_C),
                   debug_cpu_ref["decode"][i] if debug_cpu_ref else None, rows=real_T)

        if debug_cpu_ref is not None:
            report_snr("\n[fpga][debug] --- Section 5a bisect (SNR vs CPU) ---")
            for name, snr_db in debug_log:
                flag = "  <-- LOW" if snr_db < 30.0 else ""
                report_snr(f"[fpga][debug] {name:38s} {snr_db:8.2f} dB{flag}")
            report_snr("[fpga][debug] --- end bisect ---\n")

        x_out = ue.dma_from_accelerator_memory(cur_dram, (cur_T, cur_C)).float()[:real_T].T.contiguous()
        ue.reset_tensor_dram_addr()
        return x_out


# ---------------------------------------------------------------------------
# Section 5b: ISTFTNet Generator body
# ---------------------------------------------------------------------------
# The generator's own capacity. It runs at 20x and 120x the frame count, so a buffer at the
# model-wide MAX_FRAMES would be 61 MB and the section would need ~3 GB even with scratch reuse.
# 512 frames is ~12.8 s of audio.
MAX_GEN_FRAMES = 512
GEN_T0_CAP = 2 * MAX_GEN_FRAMES                                  # decoder-front output rows
GEN_T1_CAP = 20 * MAX_GEN_FRAMES                                 # after ups[0] (x10)
GEN_IN_CHUNK = 4096                                              # InstanceNorm reduction chunk
GEN_T2_CAP = _round_up(120 * MAX_GEN_FRAMES + 64, GEN_IN_CHUNK)  # after ups[1] (x6) + reflection pad
GEN_STEP_RAMP = 4096.0
# Snake range reduction: cos(2*a*x) = sin(2*pi*(a*x/pi + 0.25 mod 1)). The operand reaches ~15
# turns (measured on the CPU oracle), so the integer part is split off by MAGIC ROUNDING instead of
# a clamp staircase: n = (p + GEN_MAGIC) - GEN_MAGIC. With magic 192 and |p| < 64, p + magic lies in
# [128, 256) where the bf16 grid (8-bit significand) is exactly 1.0, so n is an integer and
# |p - n| <= 1 whether the unit rounds or truncates. HW-MEASURED (magic_round_test): the eltwise
# unit TRUNCATES -- with 256 the positive half sat on the grid-2 region and |p - n| reached 2.
# The residual is then shifted by +1 and wrapped once at 1.
GEN_MAGIC = 192.0


class GeneratorFPGA:
    """Section 5b: everything in ISTFTNet Generator.forward between `har` and `conv_post`, on
    hardware. `har` (SineGen source + STFT, Section 5c) and exp/sin + iSTFT (Section 5d) stay on
    the host for now; both are proven compositions in user_hw_test.py, not yet wired.

    Layout is [T, C] rows=time throughout, like Sections 3-5a. Every op is one captured program
    whose SCRATCH is released afterwards (``_block``): the tensor allocator is rewound to the mark
    taken after the op's output was allocated. Programs execute synchronously, so nothing still
    needs the scratch when it is reclaimed, and the rewind is deterministic, so the address map
    (and the frozen bin) is unchanged by it.

    Per-op mechanics, each HW-proven in user_hw_test.py before this was written:
      * ConvTranspose1d ups           zero-insert row loop + stride-1 conv with the flipped kernel
                                      (conv_transpose1d_zero_insert_test)
      * dilated Conv1d                shifted-matmul taps at offset k*dilation (conv1d_shifted_matmul_test)
      * AdaIN InstanceNorm at T=30601 chunked transpose + masked-ones matmul stats, Newton rsqrt,
                                      apply in [T, C] -- no large-N transpose (instance_norm_large_T_test)
      * Snake1D                       x + (1 - cos(2ax))/(2a), cos via staircase mod + quarter-wave
                                      fold + degree-9 Taylor (snake_vs_gelu_timing_test, sinegen test)
      * noise_convs                   im2col built on the host from `har` (already host data), one matmul
    """

    _up = F0NPredictionFPGA._up
    _up_cap = F0NPredictionFPGA._up_cap
    _run = F0NPredictionFPGA._run

    def __init__(self, model, ue: UnifiedEngine):
        self.model = model
        self.ue = ue
        gen = model.decoder.generator
        self.gen = gen
        self.style_dim = gen.resblocks[0].adain1[0].fc.in_features       # 128
        self.n_kernels = gen.num_kernels
        self.n_fft = gen.post_n_fft                                        # 20
        self.C_har = self.n_fft + 2                                        # 22
        self.ups = []
        for m in gen.ups:
            w, b = _wn_conv(m)                                             # [Cin, Cout, k]
            self.ups.append(dict(Cin=w.shape[0], Cout=w.shape[1], k=w.shape[2], stride=m.stride[0],
                                 pad=m.padding[0], w=w, b=b))
        self.noise_convs = []
        for m in gen.noise_convs:
            self.noise_convs.append(dict(Cout=m.out_channels, k=m.kernel_size[0], stride=m.stride[0],
                                         pad=m.padding[0], w=_bf16(m.weight), b=_bf16(m.bias)))
        self._const = {}
        self.resblocks = [self._resblk_w(m) for m in gen.resblocks]
        self.noise_res = [self._resblk_w(m) for m in gen.noise_res]
        cp_w, cp_b = _wn_conv(gen.conv_post)                              # [22, C, 7]
        self.conv_post = dict(Cout=UE_VECTOR_SIZE, k=cp_w.shape[2],
                              w=_bf16(_pad_dim(cp_w, 0, UE_VECTOR_SIZE)),
                              b=_bf16(_pad_dim(cp_b, 0, UE_VECTOR_SIZE)))

    @staticmethod
    def _resblk_w(m):
        w = dict(C=m.convs1[0].in_channels, k=m.convs1[0].kernel_size[0],
                 dil=[c.dilation[0] for c in m.convs1], taps=[])
        for c1, c2, n1, n2, a1, a2 in zip(m.convs1, m.convs2, m.adain1, m.adain2, m.alpha1, m.alpha2):
            c1w, c1b = _wn_conv(c1)
            c2w, c2b = _wn_conv(c2)
            w["taps"].append(dict(
                c1w=c1w, c1b=c1b, c2w=c2w, c2b=c2b,
                n1w=_bf16(n1.fc.weight), n1b=_bf16(n1.fc.bias),
                n2w=_bf16(n2.fc.weight), n2b=_bf16(n2.fc.bias),
                a1=_bf16(a1.reshape(-1)), a2=_bf16(a2.reshape(-1))))
        return w

    # ---- program / scratch plumbing -------------------------------------------------------------
    def _acquire(self, rows_cap, C):
        """A cap-sized activation buffer from the free list (or a fresh one at the live top).
        Call _release when its last reader has run. Deterministic, so the address map is fixed."""
        free = self._pool.setdefault((rows_cap, C), [])
        if free:
            return free.pop()
        return self.ue.allocate_tensor_dram(rows_cap * C * 2)

    def _release(self, addr):
        key = self._owner.pop(addr)
        self._pool[key].append(addr)

    def _program(self, emit):
        """Capture/run one program whose outputs the caller allocated already; scratch rewound."""
        ue = self.ue
        ue.start_capture()
        mark = ue._tensor_dram_addr
        emit()
        self._run(ue)
        ue._tensor_dram_addr = mark

    def _block(self, emit, out_rows_cap, C):
        """Capture one program: take its output buffer, mark, emit (scratch allocated inside),
        run, rewind the allocator to the mark. Returns the output address."""
        ue = self.ue
        ue.start_capture()
        out = self._acquire(out_rows_cap, C)
        self._owner[out] = (out_rows_cap, C)
        mark = ue._tensor_dram_addr
        emit(out)
        self._run(ue)
        ue._tensor_dram_addr = mark
        return out

    def _zeros_row(self, C):
        """A constant all-zero row of C elements (params region, memoised)."""
        return self._up(torch.zeros(C, dtype=torch.bfloat16))

    def _zero_rows(self, dst, C, n_rows, row_start=None, row_start_off=0):
        """Zero ``n_rows`` (int or Dim) rows of dst starting at row_start (+off): ONE HW loop."""
        _pbi_row_loop(self.ue, n_rows, reads=[(self._zeros_row(C), 0, C, 0)],
                      writes=[(dst, C * 2, C, 0)], row_start=row_start, row_start_off=row_start_off)

    def _copy_rows(self, src, dst, C, n_rows, dst_row_stride=1):
        _pbi_row_loop(self.ue, n_rows, reads=[(src, C * 2, C, 0)],
                      writes=[(dst, dst_row_stride * C * 2, C, 0)])

    def _tile_row(self, row_dram, dst, C, n_rows):
        """Broadcast one row down n_rows rows (zero source stride)."""
        _pbi_row_loop(self.ue, n_rows, reads=[(row_dram, 0, C, 0)], writes=[(dst, C * 2, C, 0)])

    def _step(self, M, N, src, dst):
        """dst = clamp(RAMP*src, 0, 1) -- the soft unit step, on a flat 64-wide view."""
        ue = self.ue
        rows = (M * N) // UE_VECTOR_SIZE
        _dyn_eltwise(ue, M, N, src, None, dst, mode=UE_MODE.MUL_BROADCAST, scalar=GEN_STEP_RAMP)
        _dyn_activation(ue, M=rows, N=UE_VECTOR_SIZE, A_DRAM_ADDR=dst, OUTPUT_DRAM_ADDR=dst,
                        IDENTITY_DRAM_ADDR=self.identity_dram, activation="clamp",
                        clamp_min=0.0, clamp_max=1.0)

    # ---- ops ----------------------------------------------------------------------------------
    def _conv1d(self, x, T, real_T, Cin, Cout, W, b, out, k, dilation=1):
        """Conv1d 'same' with dilation: y[t] = sum_k xpad[t + k*d] @ W[:, :, k]^T.

        ``T`` is the 64-aligned row count (Dim), ``real_T`` the real one (Dim); zero padding is
        placed at the REAL sequence end (rows [real_T, T) are re-zeroed, see Section 3's _conv1d
        for why that matters). W: [Cout, Cin, k] host bf16 (PyTorch layout; each tap is already
        the [N, K] operand matmat_mul_core wants).
        """
        ue = self.ue
        pad = dilation * (k - 1) // 2
        if k == 1:
            _dyn_matmul(ue, M=T, K=Cin, N=Cout, A_DRAM_ADDR=x, B_DRAM_ADDR=self._up(W[:, :, 0].contiguous()),
                        OUTPUT_DRAM_ADDR=out, C_DRAM_ADDR=self._up(b), bias_mode="broadcast_N")
            return
        xpad = ue.allocate_tensor_dram((_template(T) + 2 * pad + UE_VECTOR_SIZE) * Cin * 2)
        self._zero_rows(xpad, Cin, pad)                                   # top pad
        self._copy_rows(x, xpad + pad * Cin * 2, Cin, T)                  # real + alignment rows
        # rows [pad + real_T, pad + T + pad]: alignment padding (stale) + bottom pad. Count is
        # (T - real_T) + pad + 1 >= 1 always (a zero trip count wraps the loop counter).
        if _is_dyn(T) and _is_dyn(real_T):
            # (T - real_T) = mul*(nf_pad - nf) + (T.off - real_T.off): affine in GPR_PADROWS.
            assert T.mul == real_T.mul and T.off >= real_T.off, (T, real_T)
            n_tail = Dim(int(T) - int(real_T), GPR_PADROWS, mul=T.mul,
                         cap=T.mul * (UE_VECTOR_SIZE - 1) + T.off - real_T.off, off=T.off - real_T.off)
        else:
            n_tail = int(T) - int(real_T)
        self._zero_rows(xpad, Cin, n_tail + pad + 1, row_start=real_T, row_start_off=pad)
        acc = [ue.allocate_tensor_dram(_template(T) * Cout * 2) for _ in range(2)]
        for kk in range(k):
            last = kk == k - 1
            dst = out if last else acc[kk % 2]
            prev = None if kk == 0 else acc[(kk - 1) % 2]
            _dyn_matmul(ue, M=T, K=Cin, N=Cout, A_DRAM_ADDR=xpad + (kk * dilation) * Cin * 2,
                        B_DRAM_ADDR=self._up(W[:, :, kk].contiguous()), OUTPUT_DRAM_ADDR=dst,
                        C_DRAM_ADDR=(self._up(b) if kk == 0 else prev),
                        bias_mode=("broadcast_N" if kk == 0 else "full_matrix"))

    def _conv_transpose(self, x, T, Cin, Cout, W, b, out, k, stride, pad, T_out):
        """ConvTranspose1d(Cin, Cout, k, stride, padding=pad) with output length stride*T:
        zero-insert x (row t -> row (k-1) + stride*t of a zeroed buffer), then a stride-1 conv with
        the FLIPPED kernel read at offset (pad + k'):  out[t] = sum_k' xz[t + pad + k'] @ Wf[k']^T,
        Wf[k'] = W[:, :, k-1-k'].  W: [Cin, Cout, k] (PyTorch ConvTranspose layout).
        """
        ue = self.ue
        rows = _template(T_out) + 2 * k + UE_VECTOR_SIZE
        xz = ue.allocate_tensor_dram(rows * Cin * 2)
        self._zero_rows(xz, Cin, rows)
        self._copy_rows(x, xz + (k - 1) * Cin * 2, Cin, T, dst_row_stride=stride)
        acc = [ue.allocate_tensor_dram(_template(T_out) * Cout * 2) for _ in range(2)]
        for kk in range(k):
            wf = W[:, :, k - 1 - kk].T.contiguous()                        # [Cout, Cin] = [N, K]
            last = kk == k - 1
            dst = out if last else acc[kk % 2]
            prev = None if kk == 0 else acc[(kk - 1) % 2]
            _dyn_matmul(ue, M=T_out, K=Cin, N=Cout, A_DRAM_ADDR=xz + (pad + kk) * Cin * 2,
                        B_DRAM_ADDR=self._up(wf), OUTPUT_DRAM_ADDR=dst,
                        C_DRAM_ADDR=(self._up(b) if kk == 0 else prev),
                        bias_mode=("broadcast_N" if kk == 0 else "full_matrix"))

    def _leaky(self, x, T, C, out, slope):
        """leaky_relu = relu(x) - slope*relu(-x), both via clamp on a flat 64-wide view."""
        ue = self.ue
        rows = (T * C) // UE_VECTOR_SIZE
        pos = ue.allocate_tensor_dram(_template(T) * C * 2)
        neg = ue.allocate_tensor_dram(_template(T) * C * 2)
        _dyn_activation(ue, M=rows, N=UE_VECTOR_SIZE, A_DRAM_ADDR=x, OUTPUT_DRAM_ADDR=pos,
                        IDENTITY_DRAM_ADDR=self.identity_dram, activation="clamp")
        _dyn_eltwise(ue, rows, UE_VECTOR_SIZE, x, None, neg, mode=UE_MODE.MUL_BROADCAST, scalar=-1.0)
        _dyn_activation(ue, M=rows, N=UE_VECTOR_SIZE, A_DRAM_ADDR=neg, OUTPUT_DRAM_ADDR=neg,
                        IDENTITY_DRAM_ADDR=self.identity_dram, activation="clamp")
        _dyn_eltwise(ue, rows, UE_VECTOR_SIZE, neg, None, neg, mode=UE_MODE.MUL_BROADCAST, scalar=slope)
        _dyn_eltwise(ue, rows, UE_VECTOR_SIZE, pos, neg, out, mode=UE_MODE.ELTWISE_SUB)

    def _mask_vec(self, level, real_T):
        """Per-run reduction mask for one time level: 1/real_T on [0, real_T), 0 after, with a
        64*chunk zero tail so the [64, chunk] A-operand view at the last chunk stays in bounds
        (rows 1..63 of that view read the tail; only row 0 is used)."""
        key = ("mask", level)
        if key in self._live:
            return self._live[key]
        cap = self._caps[level]
        n = cap + 64 * GEN_IN_CHUNK
        m = torch.zeros(n, dtype=torch.bfloat16)
        m[:int(real_T)] = 1.0 / int(real_T)
        self._live[key] = self._up_cap(m, n)
        return self._live[key]

    def _reduce_T(self, src, level, C, xt, acc, dst_row):
        """dst_row [1, C] = sum_t mask[t] * src[t, :]  over ALL cap chunks (masked)."""
        ue = self.ue
        cap = self._caps[level]
        mask = self._live[("mask", level)]
        n_chunks = cap // GEN_IN_CHUNK
        for j in range(n_chunks):
            _dyn_transpose(ue, M=GEN_IN_CHUNK, N=C, INPUT_DRAM_ADDR=src + j * GEN_IN_CHUNK * C * 2,
                           OUTPUT_DRAM_ADDR=xt, IDENTITY_DRAM_ADDR=self.identity_dram)
            last = j == n_chunks - 1
            _dyn_matmul(ue, M=64, K=GEN_IN_CHUNK, N=C, A_DRAM_ADDR=mask + j * GEN_IN_CHUNK * 2,
                        B_DRAM_ADDR=xt, OUTPUT_DRAM_ADDR=(dst_row if last else acc[j % 2]),
                        C_DRAM_ADDR=(None if j == 0 else acc[(j - 1) % 2]),
                        bias_mode="full_matrix")

    def _adain(self, x, T, real_T, level, C, fc_w, fc_b, out, newton_iters=40, eps=1e-6, y0=1e-2):
        """AdaIN1d at generator scale: (x - mean) * (1 + gamma) * rstd + beta, per channel over
        time, entirely in [T, C]. Stats via _reduce_T; rstd by Newton from a constant seed (no
        rsqrt core; converges for var < 3/y0^2 = 3e4)."""
        ue = self.ue
        cap = self._caps[level]
        self._mask_vec(level, real_T)
        xt = ue.allocate_tensor_dram(C * GEN_IN_CHUNK * 2)
        acc = [ue.allocate_tensor_dram(64 * C * 2) for _ in range(2)]
        stat = ue.allocate_tensor_dram(64 * C * 2)        # row 0 = mean, later var
        meant = ue.allocate_tensor_dram(cap * C * 2)
        xc = ue.allocate_tensor_dram(cap * C * 2)
        sq = ue.allocate_tensor_dram(cap * C * 2)
        y = ue.allocate_tensor_dram(C * 2)
        tmp = ue.allocate_tensor_dram(C * 2)
        gb = ue.allocate_tensor_dram(2 * C * 2)
        scale = ue.allocate_tensor_dram(C * 2)
        scalet = ue.allocate_tensor_dram(cap * C * 2)
        betat = ue.allocate_tensor_dram(cap * C * 2)
        # style affine: gb = [gamma | beta]
        _dyn_matmul(ue, M=1, K=self.style_dim, N=2 * C, A_DRAM_ADDR=self.style_dram,
                    B_DRAM_ADDR=self._up(fc_w), OUTPUT_DRAM_ADDR=gb, C_DRAM_ADDR=self._up(fc_b),
                    bias_mode="broadcast_N")
        # Pad rows [real_T, cap) must be FINITE (they are multiplied by a zero mask, and 0*NaN=NaN).
        # cap - real_T is not affine in a primed register, so the count is derived on device.
        zrow = self._zeros_row(C)
        _zero_rows_from(ue, x, C * 2, C, start=real_T, count_cap=cap, zeros_src=zrow)
        self._reduce_T(x, level, C, xt, acc, stat)                        # row 0 = mean
        self._tile_row(stat, meant, C, T)
        _dyn_eltwise(ue, T, C, x, meant, xc, mode=UE_MODE.ELTWISE_SUB)
        _dyn_eltwise(ue, T, C, xc, xc, sq, mode=UE_MODE.ELTWISE_MUL)
        _zero_rows_from(ue, sq, C * 2, C, start=real_T, count_cap=cap, zeros_src=zrow)
        self._reduce_T(sq, level, C, xt, acc, stat)                       # row 0 = var
        ue.eltwise_core_dram(1, C, stat, None, stat, UE_MODE.ADD_BROADCAST, scalar=eps)
        ue.eltwise_core_dram(1, C, stat, None, y, UE_MODE.MUL_BROADCAST, scalar=0.0)
        ue.eltwise_core_dram(1, C, y, None, y, UE_MODE.ADD_BROADCAST, scalar=y0)
        for _ in range(newton_iters):
            ue.eltwise_core_dram(1, C, y, y, tmp, UE_MODE.ELTWISE_MUL)
            ue.eltwise_core_dram(1, C, tmp, stat, tmp, UE_MODE.ELTWISE_MUL)
            ue.eltwise_core_dram(1, C, tmp, None, tmp, UE_MODE.MUL_BROADCAST, scalar=-0.5)
            ue.eltwise_core_dram(1, C, tmp, None, tmp, UE_MODE.ADD_BROADCAST, scalar=1.5)
            ue.eltwise_core_dram(1, C, y, tmp, y, UE_MODE.ELTWISE_MUL)
        ue.eltwise_core_dram(1, C, gb, None, scale, UE_MODE.ADD_BROADCAST, scalar=1.0)   # 1 + gamma
        ue.eltwise_core_dram(1, C, scale, y, scale, UE_MODE.ELTWISE_MUL)                 # * rstd
        self._tile_row(scale, scalet, C, T)
        self._tile_row(gb + C * 2, betat, C, T)
        _dyn_eltwise(ue, T, C, xc, scalet, out, mode=UE_MODE.ELTWISE_MUL)
        _dyn_eltwise(ue, T, C, out, betat, out, mode=UE_MODE.ELTWISE_ADD)

    def _snake(self, x, T, C, alpha, out):
        """Snake1D: x + sin^2(a x)/a = x + (1 - cos(2 a x)) / (2 a), per-channel alpha.
        cos(2ax) = sin(2*pi*p), p = (a x / pi + 0.25) mod 1: staircase mod over
        [-GEN_SNAKE_TURNS, GEN_SNAKE_TURNS], quarter-wave fold, degree-9 Taylor."""
        ue = self.ue
        cap = _template(T)
        a_over_pi = ue.allocate_tensor_dram(cap * C * 2)
        inv2a = ue.allocate_tensor_dram(cap * C * 2)
        p = ue.allocate_tensor_dram(cap * C * 2)
        v = ue.allocate_tensor_dram(cap * C * 2)
        n = ue.allocate_tensor_dram(cap * C * 2)
        s = ue.allocate_tensor_dram(cap * C * 2)
        q = ue.allocate_tensor_dram(cap * C * 2)
        x2 = ue.allocate_tensor_dram(cap * C * 2)
        tm = ue.allocate_tensor_dram(cap * C * 2)
        self._tile_row(self._up((alpha.float() / math.pi).to(torch.bfloat16)), a_over_pi, C, T)
        self._tile_row(self._up((1.0 / (2.0 * alpha.float())).to(torch.bfloat16)), inv2a, C, T)
        # p = a x / pi + 0.25;  n = (p + MAGIC) - MAGIC  (integer part, see GEN_MAGIC);
        # r = p - n + 1 in (0, 2);  r -= step(r - 1)  -> [0, 1).  (The wrap band sits at 1, where the
        # bf16 grid is coarse enough that the soft step never lands inside it.)
        _dyn_eltwise(ue, T, C, x, a_over_pi, p, mode=UE_MODE.ELTWISE_MUL)
        _dyn_eltwise(ue, T, C, p, None, p, mode=UE_MODE.ADD_BROADCAST, scalar=0.25)
        _dyn_eltwise(ue, T, C, p, None, n, mode=UE_MODE.ADD_BROADCAST, scalar=GEN_MAGIC)
        _dyn_eltwise(ue, T, C, n, None, n, mode=UE_MODE.ADD_BROADCAST, scalar=-GEN_MAGIC)
        _dyn_eltwise(ue, T, C, p, n, p, mode=UE_MODE.ELTWISE_SUB)
        _dyn_eltwise(ue, T, C, p, None, p, mode=UE_MODE.ADD_BROADCAST, scalar=1.0)
        _dyn_eltwise(ue, T, C, p, None, s, mode=UE_MODE.ADD_BROADCAST, scalar=-1.0)
        self._step(T, C, s, s)
        _dyn_eltwise(ue, T, C, p, s, p, mode=UE_MODE.ELTWISE_SUB)                 # p in [0, 1)
        # quarter-wave fold: q = p + s1 (0.5 - 2p) + s2 (2p - 1.5),  s1 = step(p-.25), s2 = step(p-.75)
        _dyn_eltwise(ue, T, C, p, None, s, mode=UE_MODE.ADD_BROADCAST, scalar=-0.25)
        self._step(T, C, s, s)
        _dyn_eltwise(ue, T, C, p, None, v, mode=UE_MODE.MUL_BROADCAST, scalar=-2.0)
        _dyn_eltwise(ue, T, C, v, None, v, mode=UE_MODE.ADD_BROADCAST, scalar=0.5)     # 0.5 - 2p
        _dyn_eltwise(ue, T, C, v, s, v, mode=UE_MODE.ELTWISE_MUL)
        _dyn_eltwise(ue, T, C, p, v, q, mode=UE_MODE.ELTWISE_ADD)
        _dyn_eltwise(ue, T, C, p, None, s, mode=UE_MODE.ADD_BROADCAST, scalar=-0.75)
        self._step(T, C, s, s)
        _dyn_eltwise(ue, T, C, p, None, v, mode=UE_MODE.MUL_BROADCAST, scalar=2.0)
        _dyn_eltwise(ue, T, C, v, None, v, mode=UE_MODE.ADD_BROADCAST, scalar=-1.5)    # 2p - 1.5
        _dyn_eltwise(ue, T, C, v, s, v, mode=UE_MODE.ELTWISE_MUL)
        _dyn_eltwise(ue, T, C, q, v, q, mode=UE_MODE.ELTWISE_ADD)
        _dyn_eltwise(ue, T, C, q, None, q, mode=UE_MODE.MUL_BROADCAST, scalar=2.0 * math.pi)   # r
        # sin(r), degree 9, Horner in r^2
        _dyn_eltwise(ue, T, C, q, q, x2, mode=UE_MODE.ELTWISE_MUL)
        coef = [((-1) ** ((k - 1) // 2)) / math.factorial(k) for k in (9, 7, 5, 3, 1)]
        _dyn_eltwise(ue, T, C, x2, None, tm, mode=UE_MODE.MUL_BROADCAST, scalar=coef[0])
        for c in coef[1:]:
            _dyn_eltwise(ue, T, C, tm, None, tm, mode=UE_MODE.ADD_BROADCAST, scalar=c)
            if c != coef[-1]:
                _dyn_eltwise(ue, T, C, tm, x2, tm, mode=UE_MODE.ELTWISE_MUL)
        _dyn_eltwise(ue, T, C, tm, q, tm, mode=UE_MODE.ELTWISE_MUL)                     # cos(2ax)
        # out = x + (1 - cos) * inv2a
        _dyn_eltwise(ue, T, C, tm, None, tm, mode=UE_MODE.MUL_BROADCAST, scalar=-1.0)
        _dyn_eltwise(ue, T, C, tm, None, tm, mode=UE_MODE.ADD_BROADCAST, scalar=1.0)
        _dyn_eltwise(ue, T, C, tm, inv2a, tm, mode=UE_MODE.ELTWISE_MUL)
        _dyn_eltwise(ue, T, C, x, tm, out, mode=UE_MODE.ELTWISE_ADD)

    def _resblock(self, x, T, real_T, level, w):
        """AdaINResBlock1: for each dilation, x += conv2(snake(adain2(conv1(snake(adain1(x)))))).
        Returns the address of the block output (a fresh buffer; x is not modified)."""
        C, k, cap = w["C"], w["k"], self._caps[level]
        cur = x
        for i, (d, t) in enumerate(zip(w["dil"], w["taps"])):
            def chain(prev, emit):
                """next = emit(prev); the previous intermediate is dead once its program ran."""
                nxt = self._block(emit, cap, C)
                if prev is not None:
                    self._release(prev)
                return nxt
            h = chain(None, lambda o: self._adain(cur, T, real_T, level, C, t["n1w"], t["n1b"], o))
            h = chain(h, lambda o: self._snake(h, T, C, t["a1"], o))
            h = chain(h, lambda o: self._conv1d(h, T, real_T, C, C, t["c1w"], t["c1b"], o, k, d))
            h = chain(h, lambda o: self._adain(h, T, real_T, level, C, t["n2w"], t["n2b"], o))
            h = chain(h, lambda o: self._snake(h, T, C, t["a2"], o))
            h = chain(h, lambda o: self._conv1d(h, T, real_T, C, C, t["c2w"], t["c2b"], o, k, 1))
            nxt = chain(h, lambda o: _dyn_eltwise(self.ue, T, C, h, cur, o, mode=UE_MODE.ELTWISE_ADD))
            if cur is not x:
                self._release(cur)          # the caller still owns x
            cur = nxt
        return cur

    # ---- Section 5c: SineGen source + STFT -> har ----------------------------------------------
    GEN_UP = 300                       # prod(upsample_rates) * hop: fine samples per frame
    GEN_HARM = 9                       # harmonic_num + 1 lanes
    GEN_STFT_K = 384                   # samples one 64-frame STFT block reads (320 + window slack)
    GEN_HEAD_ROWS = 64                 # zero/reflect prefix rows before sample 0 in the time-major buffer

    def _source_prep(self, F0_curve: torch.Tensor, noise: torch.Tensor, T_f: int):
        """Host side of SineGen: the per-frame increment tables (double-bf16) and the per-run
        noise. F0_curve: [2*nf] (Section 3's output, the SineGen frame rate). noise: [S, 9] fp32
        already scaled by noise_amp (host randn -- it is model randomness, not model compute).
        Returns dict of [GEN_T0_CAP, 64] bf16 tables + the noise in j-major slabs."""
        gen = self.gen
        sg = gen.m_source.l_sin_gen
        V, H, UP = UE_VECTOR_SIZE, self.GEN_HARM, self.GEN_UP
        cap = GEN_T0_CAP
        f0 = F0_curve.detach().double().reshape(-1)[:T_f]
        harm = torch.arange(1, H + 1, dtype=torch.float64)
        rad = (f0[:, None] * harm[None, :] / sg.sampling_rate) % 1.0                # [T_f, H]
        rad_up = torch.cat([rad[1:], torch.zeros(1, H, dtype=torch.float64)], 0)     # slope, j >= 150
        rad_dn = rad.clone(); rad_dn[0] = 0                                           # slope, j < 150
        uv = (f0 > sg.voiced_threshold).double()

        def split(v):
            hi = v.to(torch.bfloat16); lo = (v - hi.double()).to(torch.bfloat16)
            Hh = torch.zeros(cap, V, dtype=torch.bfloat16); Hh[:T_f, :H] = hi
            Ll = torch.zeros(cap, V, dtype=torch.bfloat16); Ll[:T_f, :H] = lo
            return Hh, Ll
        t = {}
        t["D_h"], t["D_l"] = split((UP * rad) % 1.0)
        t["U_h"], t["U_l"] = split(rad_up)
        t["N_h"], t["N_l"] = split(-rad_dn)
        amp = torch.zeros(cap, V, dtype=torch.bfloat16)
        amp[:T_f, :H] = (sg.sine_amp * uv)[:, None].to(torch.bfloat16)               # sine_amp * uv
        t["AMP"] = amp
        nz = torch.zeros(UP, cap, V, dtype=torch.bfloat16)
        nz[:, :T_f, :H] = noise.reshape(T_f, UP, H).permute(1, 0, 2).to(torch.bfloat16)   # j-major
        t["NOISE"] = nz
        return t

    def _source_har(self, tables, T_f, S, F_rows, HAR):
        """Section 5c on device. T_f: frames (Dim, 2*nf). S: samples (Dim, 300*T_f). F_rows: STFT
        frames (Dim, S/5 + 1). Writes har [F_cap, 64] (mag in cols 0..10, phase 11..21, 0 after)
        into HAR (live, cap-sized). Everything is unrolled at the caps; the runtime lengths only
        steer the masked/strided copies, so the program is prompt-invariant.

        SineGen recipe = sinegen_wrapped_phase_test (49 dB on HW): two-sum accumulators with
        double-bf16 increments, wrapped phase, quarter-wave fold + degree-9 Taylor + lo correction.
        Sine lanes are written TIME-MAJOR (row = 300 r + j, plus reflect pads) by a strided row
        copy per fine step, so tanh(l_linear) and the STFT see a plain [samples, 64] buffer.
        STFT = one M=1 matmul per 64-frame block against a block-Toeplitz hann*DFT basis
        (stft_istft_matmul_test); magnitude/phase by CORDIC (cordic_atan2_magnitude_test).
        """
        ue = self.ue
        gen = self.gen
        V, H, UP = UE_VECTOR_SIZE, self.GEN_HARM, self.GEN_UP
        half = UP // 2
        cap_f = GEN_T0_CAP                       # frame cap
        row = V * 2
        blk = cap_f * row
        S_cap = UP * cap_f
        R = self.GEN_HEAD_ROWS + S_cap + V       # time-major rows incl. head/tail slack
        n_blocks = -(-GEN_T2_CAP // 64)          # STFT blocks at the cap
        up = lambda t: self._up_cap(t, t.numel())
        DH, DL, UH, UL, NH, NL, AMP = [up(tables[k]) for k in ("D_h", "D_l", "U_h", "U_l", "N_h", "N_l", "AMP")]
        NOISE = up(tables["NOISE"])
        ONES = self._up(torch.ones(cap_f, V, dtype=torch.bfloat16))
        ident = self.identity_dram
        OUTB = ue.allocate_tensor_dram(R * row)                      # time-major sine lanes
        PF, PFL = ue.allocate_tensor_dram(blk), ue.allocate_tensor_dram(blk)
        fr = {k: ue.allocate_tensor_dram(row) for k in ("p", "lo", "s", "bv", "t1", "t2")}
        fn = {k: ue.allocate_tensor_dram(blk) for k in
              ("p", "lo", "s", "bv", "t1", "t2", "q", "x2", "tm", "sn", "cs", "u", "w", "pc", "pm", "o")}
        LAST = ue.allocate_tensor_dram(row)

        def E(M, a, b, o, mode, sc=None):
            ue.eltwise_core_dram(M, V, a, b, o, mode, scalar=sc)

        def twosum(M, b, d):
            E(M, b["p"], d, b["s"], UE_MODE.ELTWISE_ADD); E(M, b["s"], b["p"], b["bv"], UE_MODE.ELTWISE_SUB)
            E(M, b["s"], b["bv"], b["t1"], UE_MODE.ELTWISE_SUB); E(M, b["p"], b["t1"], b["t1"], UE_MODE.ELTWISE_SUB)
            E(M, d, b["bv"], b["t2"], UE_MODE.ELTWISE_SUB); E(M, b["t1"], b["t2"], b["t1"], UE_MODE.ELTWISE_ADD)
            E(M, b["lo"], b["t1"], b["lo"], UE_MODE.ELTWISE_ADD); E(M, b["s"], b["lo"], b["p"], UE_MODE.ELTWISE_ADD)
            E(M, b["p"], b["s"], b["t1"], UE_MODE.ELTWISE_SUB); E(M, b["lo"], b["t1"], b["lo"], UE_MODE.ELTWISE_SUB)

        def step(M, src, dst):
            E(M, src, None, dst, UE_MODE.MUL_BROADCAST, GEN_STEP_RAMP)
            ue.activation_core(M=M, N=V, A_DRAM_ADDR=dst, OUTPUT_DRAM_ADDR=dst, IDENTITY_DRAM_ADDR=ident,
                               activation="clamp", clamp_min=0.0, clamp_max=1.0)

        def wrap_up(M, b):
            E(M, b["p"], None, b["t1"], UE_MODE.ADD_BROADCAST, -1.0); step(M, b["t1"], b["t2"])
            E(M, b["p"], b["t2"], b["p"], UE_MODE.ELTWISE_SUB)

        def wrap_dn(M, b):
            E(M, b["p"], None, b["t1"], UE_MODE.MUL_BROADCAST, -1.0); E(M, b["t1"], None, b["t1"], UE_MODE.ADD_BROADCAST, 1.0)
            step(M, b["t1"], b["q"]); twosum(M, b, b["q"])

        def sin2pi_fold(M, b, src, dst):
            E(M, src, None, b["t1"], UE_MODE.ADD_BROADCAST, -0.25); step(M, b["t1"], b["u"])
            E(M, src, None, b["t1"], UE_MODE.ADD_BROADCAST, -0.75); step(M, b["t1"], b["w"])
            E(M, src, None, b["t2"], UE_MODE.MUL_BROADCAST, 2.0)
            E(M, b["t2"], None, b["t1"], UE_MODE.MUL_BROADCAST, -1.0); E(M, b["t1"], None, b["t1"], UE_MODE.ADD_BROADCAST, 0.5)
            E(M, b["t1"], b["u"], b["t1"], UE_MODE.ELTWISE_MUL); E(M, src, b["t1"], b["q"], UE_MODE.ELTWISE_ADD)
            E(M, b["t2"], None, b["t2"], UE_MODE.ADD_BROADCAST, -1.5); E(M, b["t2"], b["w"], b["t2"], UE_MODE.ELTWISE_MUL)
            E(M, b["q"], b["t2"], b["q"], UE_MODE.ELTWISE_ADD); E(M, b["q"], None, b["q"], UE_MODE.MUL_BROADCAST, 2.0 * math.pi)
            E(M, b["q"], b["q"], b["x2"], UE_MODE.ELTWISE_MUL)
            coef = [((-1) ** ((k - 1) // 2)) / math.factorial(k) for k in (9, 7, 5, 3, 1)]
            E(M, b["x2"], None, b["tm"], UE_MODE.MUL_BROADCAST, coef[0])
            for c in coef[1:]:
                E(M, b["tm"], None, b["tm"], UE_MODE.ADD_BROADCAST, c)
                if c != coef[-1]:
                    E(M, b["tm"], b["x2"], b["tm"], UE_MODE.ELTWISE_MUL)
            E(M, b["tm"], b["q"], dst, UE_MODE.ELTWISE_MUL)

        def emit_sine(M, b, src, out):
            sin2pi_fold(M, b, src, b["sn"])
            E(M, src, None, b["pc"], UE_MODE.ADD_BROADCAST, 0.25)
            wrap_up(M, {"p": b["pc"], "t1": b["t1"], "t2": b["t2"]})
            sin2pi_fold(M, b, b["pc"], b["cs"])
            E(M, b["lo"], None, b["t1"], UE_MODE.MUL_BROADCAST, 2.0 * math.pi)
            E(M, b["t1"], b["cs"], b["t1"], UE_MODE.ELTWISE_MUL)
            E(M, b["sn"], b["t1"], out, UE_MODE.ELTWISE_ADD)

        def emit_step(j, src):
            """sine lanes for fine step j (all frames): amplitude*uv + noise_j, then scatter to
            time-major row 300 r + j (+ head rows), plus the reflect pads torch.stft(center=True)
            would add: head row 64-m <- sample m (m = j, r = 0), tail row S+k <- sample S-2-k."""
            o = fn["o"]
            emit_sine(cap_f, fn, src, o)
            E(cap_f, o, AMP, o, UE_MODE.ELTWISE_MUL)
            E(cap_f, o, NOISE + j * blk, o, UE_MODE.ELTWISE_ADD)
            base = OUTB + (self.GEN_HEAD_ROWS + j) * row
            _pbi_row_loop(ue, T_f, reads=[(o, row, V, 0)], writes=[(base, UP * row, V, 0)])
            if 1 <= j <= gen.stft.filter_length // 2:                       # head: x[-m] = x[m]
                _pbi_row_loop(ue, 1, reads=[(o, 0, V, 0)],
                              writes=[(OUTB + (self.GEN_HEAD_ROWS - j) * row, 0, V, 0)])
            k = UP - 2 - j
            if 0 <= k < gen.stft.filter_length // 2:                        # tail: x[S+k] = x[S-2-k]
                # LAST <- o[T_f - 1] (row index T_f - 1 = row_start T_f on a base one row back)
                _pbi_row_loop(ue, 1, reads=[(o - row, row, V, 0)], writes=[(LAST, 0, V, 0)], row_start=T_f)
                _pbi_row_loop(ue, 1, reads=[(LAST, 0, V, 0)], writes=[(OUTB + self.GEN_HEAD_ROWS * row, row, V, 0)],
                              row_start=S, row_start_off=k)

        # ---- 0. the time-major buffer must be finite everywhere the STFT blocks can read ----
        self._zero_rows(OUTB, V, R)
        # ---- 1. frame accumulator ----
        E(1, DH, None, fr["p"], UE_MODE.MUL_BROADCAST, 0.0); E(1, DH, None, fr["lo"], UE_MODE.MUL_BROADCAST, 0.0)
        for r in range(cap_f):
            twosum(1, fr, DH + r * row); E(1, fr["lo"], DL + r * row, fr["lo"], UE_MODE.ELTWISE_ADD); wrap_up(1, fr)
            E(1, fr["p"], None, PF + r * row, UE_MODE.MUL_BROADCAST, 1.0)
            E(1, fr["lo"], None, PFL + r * row, UE_MODE.MUL_BROADCAST, 1.0)
        # ---- 2a. fine steps counting up: j = 150 .. 299 ----
        E(cap_f, PF, None, fn["p"], UE_MODE.MUL_BROADCAST, 1.0); E(cap_f, PFL, None, fn["lo"], UE_MODE.MUL_BROADCAST, 1.0)
        E(cap_f, UH, None, fn["x2"], UE_MODE.MUL_BROADCAST, 0.5); E(cap_f, UL, None, fn["tm"], UE_MODE.MUL_BROADCAST, 0.5)
        twosum(cap_f, fn, fn["x2"]); E(cap_f, fn["lo"], fn["tm"], fn["lo"], UE_MODE.ELTWISE_ADD); wrap_up(cap_f, fn)
        emit_step(half, fn["p"])
        for jj in range(half + 1, UP):
            twosum(cap_f, fn, UH); E(cap_f, fn["lo"], UL, fn["lo"], UE_MODE.ELTWISE_ADD); wrap_up(cap_f, fn)
            emit_step(jj, fn["p"])
        # ---- 2b. counting down: j = 149 .. 0, phase held as q = p + 1 ----
        E(cap_f, PF, None, fn["p"], UE_MODE.MUL_BROADCAST, 1.0); E(cap_f, PFL, None, fn["lo"], UE_MODE.MUL_BROADCAST, 1.0)
        twosum(cap_f, fn, ONES)
        E(cap_f, NH, None, fn["x2"], UE_MODE.MUL_BROADCAST, 0.5); E(cap_f, NL, None, fn["tm"], UE_MODE.MUL_BROADCAST, 0.5)
        twosum(cap_f, fn, fn["x2"]); E(cap_f, fn["lo"], fn["tm"], fn["lo"], UE_MODE.ELTWISE_ADD); wrap_dn(cap_f, fn)
        E(cap_f, fn["p"], None, fn["pm"], UE_MODE.ADD_BROADCAST, -1.0)
        emit_step(half - 1, fn["pm"])
        for jj in range(half - 2, -1, -1):
            twosum(cap_f, fn, NH); E(cap_f, fn["lo"], NL, fn["lo"], UE_MODE.ELTWISE_ADD); wrap_dn(cap_f, fn)
            E(cap_f, fn["p"], None, fn["pm"], UE_MODE.ADD_BROADCAST, -1.0)
            emit_step(jj, fn["pm"])
        # ---- 3. sine_merge = tanh(l_linear(sines)) over the whole time-major buffer ----
        lin = gen.m_source.l_linear
        Wl = torch.zeros(V, V, dtype=torch.bfloat16); Wl[0, :H] = _bf16(lin.weight.reshape(-1))
        bl = torch.zeros(V, dtype=torch.bfloat16); bl[0] = _bf16(lin.bias.reshape(-1))
        MERGE = ue.allocate_tensor_dram(R * row)
        _dyn_matmul(ue, M=R, K=V, N=V, A_DRAM_ADDR=OUTB, B_DRAM_ADDR=self._up(Wl), OUTPUT_DRAM_ADDR=MERGE,
                    C_DRAM_ADDR=self._up(bl), bias_mode="broadcast_N")
        self._tanh(R, V, MERGE, MERGE)
        # ---- 4. column 0 -> contiguous signal: transpose [R, 64] -> [64, R]; row 0 is x_pad ----
        XP = ue.allocate_tensor_dram(V * R * 2)
        _dyn_transpose(ue, M=R, N=V, INPUT_DRAM_ADDR=MERGE, OUTPUT_DRAM_ADDR=XP, IDENTITY_DRAM_ADDR=ident)
        # ---- 5. STFT: spec block b = x_pad[48 + 320 b : +384] @ B^T, frames 64b..64b+63 ----
        n_fft, hop = gen.stft.filter_length, gen.stft.hop_length
        n_bins = n_fft // 2 + 1
        Bre, Bim = self._stft_basis(n_fft, hop)
        RE = ue.allocate_tensor_dram(n_blocks * 64 * row)
        IM = ue.allocate_tensor_dram(n_blocks * 64 * row)
        REraw = ue.allocate_tensor_dram(n_blocks * 64 * row)
        IMraw = ue.allocate_tensor_dram(n_blocks * 64 * row)
        self._dbg = dict(OUTB=OUTB, XP=XP, RE=REraw, IM=IMraw, R=R)
        off0 = self.GEN_HEAD_ROWS - n_fft // 2 - 6          # 48: first block's read start (8 B aligned)
        for b in range(n_blocks):
            a = XP + (off0 + 64 * hop * b) * 2
            ue.matmat_mul_core(M=1, K=self.GEN_STFT_K, N=64 * V, A_DRAM_ADDR=a, B_DRAM_ADDR=self._up(Bre),
                               OUTPUT_DRAM_ADDR=RE + b * 64 * row)
            ue.matmat_mul_core(M=1, K=self.GEN_STFT_K, N=64 * V, A_DRAM_ADDR=a, B_DRAM_ADDR=self._up(Bim),
                               OUTPUT_DRAM_ADDR=IM + b * 64 * row)
        # ---- 6. CORDIC vectoring: |z| and atan2(im, re), rows = all cap frames ----
        Mf = n_blocks * 64
        SX, SY, T1, T2, Z = [ue.allocate_tensor_dram(Mf * row) for _ in range(5)]
        E(Mf, RE, None, REraw, UE_MODE.MUL_BROADCAST, 1.0)                   # debug copies (pre-CORDIC)
        E(Mf, IM, None, IMraw, UE_MODE.MUL_BROADCAST, 1.0)
        # The spectrum bins are ~1e-2 and the clamp step's soft band is 1/GEN_STEP_RAMP = 2.4e-4:
        # unscaled, a few percent of the (near-empty) bins got a FRACTIONAL quadrant sign, i.e. a
        # garbage angle -- 10 dB off the model's own noise floor at conv_post. The angle is
        # scale-invariant, so scale up first (exact in bf16) and fold 1/scale into the gain.
        PRE = 1024.0
        E(Mf, RE, None, RE, UE_MODE.MUL_BROADCAST, PRE)
        E(Mf, IM, None, IM, UE_MODE.MUL_BROADCAST, PRE)
        iters = 8
        Kg = 1.0
        for k in range(iters):
            Kg *= math.sqrt(1.0 + 2.0 ** (-2 * k))

        def sign(src, dst):
            step(Mf, src, dst); E(Mf, dst, None, dst, UE_MODE.MUL_BROADCAST, 2.0); E(Mf, dst, None, dst, UE_MODE.ADD_BROADCAST, -1.0)
        step(Mf, RE, T1); E(Mf, T1, None, T1, UE_MODE.MUL_BROADCAST, -1.0); E(Mf, T1, None, T1, UE_MODE.ADD_BROADCAST, 1.0)
        # Quadrant sign with sign(0) = +1: the DC and Nyquist bins are purely real (im == 0 exactly),
        # where torch.angle gives atan2(0, x<0) = +pi. A plain clamp step gives sign(0) = -1, i.e.
        # -pi: a systematic 2*pi error on the STRONGEST channel (82% of DC frames in simulation).
        E(Mf, IM, None, T2, UE_MODE.ADD_BROADCAST, 4.0 / GEN_STEP_RAMP)   # > the step band, << PRE-scaled bins
        sign(T2, SY); E(Mf, T1, SY, Z, UE_MODE.ELTWISE_MUL); E(Mf, Z, None, Z, UE_MODE.MUL_BROADCAST, math.pi)
        sign(RE, SX); E(Mf, RE, SX, RE, UE_MODE.ELTWISE_MUL); E(Mf, IM, SX, IM, UE_MODE.ELTWISE_MUL)
        for k in range(iters):
            sign(IM, SY)
            E(Mf, IM, SY, T1, UE_MODE.ELTWISE_MUL); E(Mf, T1, None, T1, UE_MODE.MUL_BROADCAST, 2.0 ** (-k))
            E(Mf, RE, SY, T2, UE_MODE.ELTWISE_MUL); E(Mf, T2, None, T2, UE_MODE.MUL_BROADCAST, 2.0 ** (-k))
            E(Mf, RE, T1, RE, UE_MODE.ELTWISE_ADD); E(Mf, IM, T2, IM, UE_MODE.ELTWISE_SUB)
            E(Mf, SY, None, T1, UE_MODE.MUL_BROADCAST, math.atan(2.0 ** (-k))); E(Mf, Z, T1, Z, UE_MODE.ELTWISE_ADD)
        E(Mf, RE, None, RE, UE_MODE.MUL_BROADCAST, 1.0 / (Kg * PRE))         # magnitude
        # ---- 7. har = [mag(0..10) | phase(11..21) | 0]: two selection matmuls ----
        S1 = torch.zeros(V, V, dtype=torch.bfloat16); S2 = torch.zeros(V, V, dtype=torch.bfloat16)
        for k in range(n_bins):
            S1[k, k] = 1.0; S2[n_bins + k, k] = 1.0
        _dyn_matmul(ue, M=Mf, K=V, N=V, A_DRAM_ADDR=RE, B_DRAM_ADDR=self._up(S1), OUTPUT_DRAM_ADDR=T1)
        _dyn_matmul(ue, M=Mf, K=V, N=V, A_DRAM_ADDR=Z, B_DRAM_ADDR=self._up(S2), OUTPUT_DRAM_ADDR=HAR,
                    C_DRAM_ADDR=T1, bias_mode="full_matrix")
        # rows [F, F+64): the convs' right zero-padding (frame F would otherwise carry the tail)
        self._zero_rows(HAR, V, V, row_start=F_rows)

    def _tanh(self, M, N, x, out, x_split=0.6):
        """tanh without the 2*sigmoid(2x)-1 cancellation: sigmoid lands near 0.5 for small x, where
        bf16 has 2^-9 resolution, so that form caps the SOURCE signal at ~32 dB (measured in the
        5c bisect). Here |x| < x_split uses the odd Taylor series x - x^3/3 + 2x^5/15 - 17x^7/315
        (relative precision), and |x| >= x_split the sigmoid form (|tanh| >= 0.54 there, so the
        absolute 2^-9 error is < 1%). Blend weight s = step(|x| - x_split) from the clamp step."""
        ue = self.ue
        ident = self.identity_dram
        n = M * N
        xc, x2, poly, sig, sgn, w = [ue.allocate_tensor_dram(n * 2) for _ in range(6)]
        E = lambda a, b, o, mode, sc=None: ue.eltwise_core_dram(M, N, a, b, o, mode, scalar=sc)
        # xc = clamp(x, -split, split); poly(xc) by Horner in xc^2
        _dyn_activation(ue, M=M, N=N, A_DRAM_ADDR=x, OUTPUT_DRAM_ADDR=xc, IDENTITY_DRAM_ADDR=ident,
                        activation="clamp", clamp_min=-x_split, clamp_max=x_split)
        E(xc, xc, x2, UE_MODE.ELTWISE_MUL)
        coef = (-17.0 / 315.0, 2.0 / 15.0, -1.0 / 3.0, 1.0)
        E(x2, None, poly, UE_MODE.MUL_BROADCAST, coef[0])
        for c in coef[1:]:
            E(poly, None, poly, UE_MODE.ADD_BROADCAST, c)
            if c != coef[-1]:
                E(poly, x2, poly, UE_MODE.ELTWISE_MUL)
        E(poly, xc, poly, UE_MODE.ELTWISE_MUL)
        # sig = 2 sigmoid(2x) - 1
        E(x, None, sig, UE_MODE.MUL_BROADCAST, 2.0)
        _dyn_activation(ue, M=M, N=N, A_DRAM_ADDR=sig, OUTPUT_DRAM_ADDR=sig, IDENTITY_DRAM_ADDR=ident,
                        activation="sigmoid")
        E(sig, None, sig, UE_MODE.MUL_BROADCAST, 2.0); E(sig, None, sig, UE_MODE.ADD_BROADCAST, -1.0)
        # s = step(|x| - split), |x| = x * sign(x), sign = 2 step(x) - 1
        self._step(M, N, x, sgn)
        E(sgn, None, sgn, UE_MODE.MUL_BROADCAST, 2.0); E(sgn, None, sgn, UE_MODE.ADD_BROADCAST, -1.0)
        E(x, sgn, w, UE_MODE.ELTWISE_MUL); E(w, None, w, UE_MODE.ADD_BROADCAST, -x_split)
        self._step(M, N, w, w)
        # out = poly + s * (sig - poly)
        E(sig, poly, sig, UE_MODE.ELTWISE_SUB); E(sig, w, sig, UE_MODE.ELTWISE_MUL)
        E(poly, sig, out, UE_MODE.ELTWISE_ADD)

    def _stft_basis(self, n_fft, hop):
        """Block-Toeplitz hann*DFT bases for 64 frames per block: rows i*64 + k (frame i, bin k),
        columns = sample offset 6 + hop*i + n within the block's 384-sample read (which starts 16
        samples before the block's first frame's centre - n_fft/2, i.e. 6 before its first sample)."""
        V = UE_VECTOR_SIZE
        key = ("stft_basis", n_fft, hop)
        if key in self._const:
            return self._const[key]
        win = torch.hann_window(n_fft, periodic=True, dtype=torch.float64)
        n = torch.arange(n_fft, dtype=torch.float64)
        n_bins = n_fft // 2 + 1
        Bre = torch.zeros(64 * V, self.GEN_STFT_K, dtype=torch.float64)
        Bim = torch.zeros(64 * V, self.GEN_STFT_K, dtype=torch.float64)
        for i in range(64):
            c0 = 6 + hop * i
            for k in range(n_bins):
                Bre[i * V + k, c0:c0 + n_fft] = win * torch.cos(2 * math.pi * k * n / n_fft)
                Bim[i * V + k, c0:c0 + n_fft] = -win * torch.sin(2 * math.pi * k * n / n_fft)
        self._const[key] = (Bre.to(torch.bfloat16).contiguous(), Bim.to(torch.bfloat16).contiguous())
        return self._const[key]

    def _noise_conv_strided(self, HAR, T_out, Cout, W, b, k, stride, pad, out):
        """noise_convs[0]: Conv1d(22 -> Cout, k, stride, pad) over har rows: gather rows
        stride*t + kk - pad for each tap into a contiguous [T_out, 64] view (one HW loop per tap),
        then k shifted matmuls accumulate. W: [Cout, 22, k]."""
        ue = self.ue
        V = UE_VECTOR_SIZE
        cap = _template(T_out)
        g = ue.allocate_tensor_dram(cap * V * 2)
        acc = [ue.allocate_tensor_dram(cap * Cout * 2) for _ in range(2)]
        for kk in range(k):
            # source row = stride*t + (kk - pad): a negative start for kk < pad reads the zero rows
            # that precede HAR (HAR is allocated with a 64-row zero prefix).
            src = HAR + (kk - pad) * V * 2
            _pbi_row_loop(ue, T_out, reads=[(src, stride * V * 2, V, 0)], writes=[(g, V * 2, V, 0)])
            wk = _bf16(_pad_dim(W[:, :, kk].contiguous(), 1, V))            # [Cout, 64]
            last = kk == k - 1
            _dyn_matmul(ue, M=T_out, K=V, N=Cout, A_DRAM_ADDR=g, B_DRAM_ADDR=self._up(wk),
                        OUTPUT_DRAM_ADDR=(out if last else acc[kk % 2]),
                        C_DRAM_ADDR=(self._up(b) if kk == 0 else acc[(kk - 1) % 2]),
                        bias_mode=("broadcast_N" if kk == 0 else "full_matrix"))

    # ---- Section 5d: exp / sin -> complex spectrum -> iSTFT --------------------------------------
    GEN_EXP_C = 14.0            # softmax anchor: exp(x) = softmax([x.., C, -inf..]) * e^C, x <= ~4
    GEN_ISTFT_FRAMES = 68       # frames one 320-sample block depends on (64b-2 .. 64b+65)

    def _sincos_2pi_frac(self, T, C, p, sn, cs):
        """sn = sin(2*pi*p), cs = cos(2*pi*p) for any |p| < 64: magic-round mod 1 (see GEN_MAGIC),
        quarter-wave fold, degree-9 Taylor -- the Snake path, run twice (cos = sin at p + 0.25)."""
        ue = self.ue
        cap = _template(T)
        n, s, v, q, x2, tm, pf = [ue.allocate_tensor_dram(cap * C * 2) for _ in range(7)]
        E = lambda a, b, o, mode, sc=None: _dyn_eltwise(ue, T, C, a, b, o, mode=mode, scalar=sc)
        for shift, dst in ((0.0, sn), (0.25, cs)):
            E(p, None, pf, UE_MODE.ADD_BROADCAST, shift)
            E(pf, None, n, UE_MODE.ADD_BROADCAST, GEN_MAGIC); E(n, None, n, UE_MODE.ADD_BROADCAST, -GEN_MAGIC)
            E(pf, n, pf, UE_MODE.ELTWISE_SUB); E(pf, None, pf, UE_MODE.ADD_BROADCAST, 1.0)
            E(pf, None, s, UE_MODE.ADD_BROADCAST, -1.0); self._step(T, C, s, s); E(pf, s, pf, UE_MODE.ELTWISE_SUB)
            # fold
            E(pf, None, s, UE_MODE.ADD_BROADCAST, -0.25); self._step(T, C, s, s)
            E(pf, None, v, UE_MODE.MUL_BROADCAST, -2.0); E(v, None, v, UE_MODE.ADD_BROADCAST, 0.5)
            E(v, s, v, UE_MODE.ELTWISE_MUL); E(pf, v, q, UE_MODE.ELTWISE_ADD)
            E(pf, None, s, UE_MODE.ADD_BROADCAST, -0.75); self._step(T, C, s, s)
            E(pf, None, v, UE_MODE.MUL_BROADCAST, 2.0); E(v, None, v, UE_MODE.ADD_BROADCAST, -1.5)
            E(v, s, v, UE_MODE.ELTWISE_MUL); E(q, v, q, UE_MODE.ELTWISE_ADD)
            E(q, None, q, UE_MODE.MUL_BROADCAST, 2.0 * math.pi)
            E(q, q, x2, UE_MODE.ELTWISE_MUL)
            coef = [((-1) ** ((k - 1) // 2)) / math.factorial(k) for k in (9, 7, 5, 3, 1)]
            E(x2, None, tm, UE_MODE.MUL_BROADCAST, coef[0])
            for c in coef[1:]:
                E(tm, None, tm, UE_MODE.ADD_BROADCAST, c)
                if c != coef[-1]:
                    E(tm, x2, tm, UE_MODE.ELTWISE_MUL)
            E(tm, q, dst, UE_MODE.ELTWISE_MUL)

    def _istft_basis(self, n_fft, hop):
        """Bi [320, 68*64]: y[m] (block-relative) = sum over frames f_rel (0..67, frame 64b-2+f_rel)
        and bins of  w[n]/n_fft/1.5 * c_k * (re_k cos - im_k sin), n = m - (hop*(f_rel-2) - n_fft/2).
        Row layout per frame: re_k at col k, im_k at col 11+k. sum(w^2) over the 4 overlapping hann
        frames is exactly 1.5 everywhere torch.istft(center=True) keeps, so no edge correction."""
        key = ("istft_basis", n_fft, hop)
        if key in self._const:
            return self._const[key]
        V = UE_VECTOR_SIZE
        NF = self.GEN_ISTFT_FRAMES
        win = torch.hann_window(n_fft, periodic=True, dtype=torch.float64)
        n_bins = n_fft // 2 + 1
        ck = torch.full((n_bins,), 2.0, dtype=torch.float64); ck[0] = 1.0; ck[-1] = 1.0
        Bi = torch.zeros(64 * hop, NF * V, dtype=torch.float64)
        for f_rel in range(NF):
            for nn_ in range(n_fft):
                m = hop * (f_rel - 2) - n_fft // 2 + nn_
                if 0 <= m < 64 * hop:
                    for k in range(n_bins):
                        g = win[nn_] * ck[k] / n_fft / 1.5
                        Bi[m, f_rel * V + k] = g * math.cos(2 * math.pi * k * nn_ / n_fft)
                        Bi[m, f_rel * V + n_bins + k] = -g * math.sin(2 * math.pi * k * nn_ / n_fft)
        self._const[key] = Bi.to(torch.bfloat16).contiguous()
        return self._const[key]

    def _istft(self, y, T, real_T, S, OUT):
        """y [T, 64] = conv_post output (log-mag 0..10 | phase 11..21). OUT [S_cap] = audio samples.
        exp via the softmax epilogue (native LALU exp): row' = [x_0..x_10 | C | -100...], spec =
        softmax(row') * e^C  (the -100 lanes and the 1 + sum e^{x-C} < 1.0003 correction vanish).
        sin/cos via _sincos_2pi_frac; complex rows [re | im | 0]; iSTFT = one M=1 matmul per
        320-sample block over 68 frame rows (2 zero frames precede frame 0; frames >= F zeroed)."""
        ue = self.ue
        gen = self.gen
        V = UE_VECTOR_SIZE
        n_fft, hop = gen.stft.filter_length, gen.stft.hop_length
        nb = n_fft // 2 + 1
        cap = _template(T)
        row = V * 2
        C_ = self.GEN_EXP_C
        # --- exp ---
        Ssel = torch.zeros(V, V, dtype=torch.bfloat16)                 # keep cols 0..10, drop the rest
        for k in range(nb):
            Ssel[k, k] = 1.0
        bsel = torch.full((V,), -100.0, dtype=torch.bfloat16); bsel[:nb] = 0.0; bsel[nb] = C_
        rowp = ue.allocate_tensor_dram(cap * row)
        spec = ue.allocate_tensor_dram(cap * row)
        _dyn_matmul(ue, M=T, K=V, N=V, A_DRAM_ADDR=y, B_DRAM_ADDR=self._up(Ssel), OUTPUT_DRAM_ADDR=rowp,
                    C_DRAM_ADDR=self._up(bsel), bias_mode="broadcast_N")
        _dyn_activation(ue, M=T, N=V, A_DRAM_ADDR=rowp, OUTPUT_DRAM_ADDR=spec, IDENTITY_DRAM_ADDR=self.identity_dram,
                        activation="softmax")
        _dyn_eltwise(ue, T, V, spec, None, spec, mode=UE_MODE.MUL_BROADCAST, scalar=math.exp(C_))
        # --- sin / cos of the phase lanes (cols 11..21 of y), moved to cols 0..10 ---
        Sph = torch.zeros(V, V, dtype=torch.bfloat16)
        for k in range(nb):
            Sph[k, nb + k] = 1.0
        ph = ue.allocate_tensor_dram(cap * row)
        _dyn_matmul(ue, M=T, K=V, N=V, A_DRAM_ADDR=y, B_DRAM_ADDR=self._up(Sph), OUTPUT_DRAM_ADDR=ph)
        _dyn_eltwise(ue, T, V, ph, None, ph, mode=UE_MODE.MUL_BROADCAST, scalar=1.0 / (2.0 * math.pi))
        # The model's angle is phi = sin(x) (TorchSTFT.inverse gets `phase = torch.sin(x)` and
        # forms spec * exp(1j * phase)), so the complex parts are cos(phi), sin(phi) with |phi| <= 1:
        # inner sin by the wrapped path, outer pair by the bounded degree-7/6 Taylor series.
        phi = ue.allocate_tensor_dram(cap * row)
        cs0 = ue.allocate_tensor_dram(cap * row)
        self._sincos_2pi_frac(T, V, ph, phi, cs0)                          # phi = sin(x)
        sn = ue.allocate_tensor_dram(cap * row)
        cs = ue.allocate_tensor_dram(cap * row)
        x2 = ue.allocate_tensor_dram(cap * row)
        E = lambda a, b, o, mode, sc=None: _dyn_eltwise(ue, T, V, a, b, o, mode=mode, scalar=sc)
        E(phi, phi, x2, UE_MODE.ELTWISE_MUL)
        E(x2, None, cs, UE_MODE.MUL_BROADCAST, -1.0 / 720.0); E(cs, None, cs, UE_MODE.ADD_BROADCAST, 1.0 / 24.0)
        E(cs, x2, cs, UE_MODE.ELTWISE_MUL); E(cs, None, cs, UE_MODE.ADD_BROADCAST, -0.5)
        E(cs, x2, cs, UE_MODE.ELTWISE_MUL); E(cs, None, cs, UE_MODE.ADD_BROADCAST, 1.0)          # cos(phi)
        E(x2, None, sn, UE_MODE.MUL_BROADCAST, -1.0 / 5040.0); E(sn, None, sn, UE_MODE.ADD_BROADCAST, 1.0 / 120.0)
        E(sn, x2, sn, UE_MODE.ELTWISE_MUL); E(sn, None, sn, UE_MODE.ADD_BROADCAST, -1.0 / 6.0)
        E(sn, x2, sn, UE_MODE.ELTWISE_MUL); E(sn, None, sn, UE_MODE.ADD_BROADCAST, 1.0)
        E(sn, phi, sn, UE_MODE.ELTWISE_MUL)                                                     # sin(phi)
        # --- complex rows: [spec*cos (0..10) | spec*sin (11..21) | 0], with 64 zero rows in front ---
        n_blocks = _template(S) // (64 * hop)
        SPECB = ue.allocate_tensor_dram((V + cap + 2 * V) * row)
        SPEC = SPECB + V * row
        re = ue.allocate_tensor_dram(cap * row)
        im = ue.allocate_tensor_dram(cap * row)
        _dyn_eltwise(ue, T, V, spec, cs, re, mode=UE_MODE.ELTWISE_MUL)
        _dyn_eltwise(ue, T, V, spec, sn, im, mode=UE_MODE.ELTWISE_MUL)
        S1 = torch.zeros(V, V, dtype=torch.bfloat16); S2 = torch.zeros(V, V, dtype=torch.bfloat16)
        for k in range(nb):
            S1[k, k] = 1.0; S2[nb + k, k] = 1.0
        tmp = ue.allocate_tensor_dram(cap * row)
        _dyn_matmul(ue, M=T, K=V, N=V, A_DRAM_ADDR=re, B_DRAM_ADDR=self._up(S1), OUTPUT_DRAM_ADDR=tmp)
        _dyn_matmul(ue, M=T, K=V, N=V, A_DRAM_ADDR=im, B_DRAM_ADDR=self._up(S2), OUTPUT_DRAM_ADDR=SPEC,
                    C_DRAM_ADDR=tmp, bias_mode="full_matrix")
        self._dbg5d = dict(spec=spec, phi=phi, sn=sn, cs=cs, SPEC=SPEC)
        self._zero_rows(SPECB, V, V)                                       # the 2 (and more) frames before 0
        self._zero_rows(SPEC, V, 2 * V, row_start=real_T)                  # frames >= F contribute nothing
        # --- iSTFT ---
        Bi = self._istft_basis(n_fft, hop)
        K_i = self.GEN_ISTFT_FRAMES * V
        for b in range(n_blocks):
            ue.matmat_mul_core(M=1, K=K_i, N=64 * hop, A_DRAM_ADDR=SPEC + (64 * b - 2) * row,
                               B_DRAM_ADDR=self._up(Bi), OUTPUT_DRAM_ADDR=OUT + 64 * hop * b * 2)

    # ---- forward ------------------------------------------------------------------------------
    def forward(self, x_gen: torch.Tensor, s: torch.Tensor, F0_curve: torch.Tensor, noise: torch.Tensor,
                n_frames: int, debug_cpu_ref=None):
        """x_gen: [C0, 2*n_frames] host (Section 5a's output). s: [style_dim]. F0_curve: [2*n_frames]
        host (Section 3's output; SineGen's frame rate). noise: [S, 9] host, SineGen's additive
        noise already scaled by noise_amp (the model's randomness -- generate_source_noise()).
        Returns (audio [600*n_frames] host, conv_post output [22, F] host, F = 120*n_frames + 1).
        debug_cpu_ref["har"] (if given) is compared right after Section 5c.
        """
        ue = self.ue
        gen = self.gen
        V = UE_VECTOR_SIZE
        nf = n_frames
        assert nf <= MAX_GEN_FRAMES, (
            f"n_frames={nf} exceeds the generator capacity MAX_GEN_FRAMES={MAX_GEN_FRAMES}; the "
            f"program image is frozen against that cap (see GEN_T2_CAP).")
        nf_pad = _round_up(nf, V)
        set_runtime_dims(n_frames=nf, nf_pad=nf_pad, pad_rows=nf_pad - nf)
        _set_row_cap(GEN_T2_CAP, GEN_T2_CAP, "generator rows")
        # Time levels: 0 = decoder-front output (2 nf), 1 = after ups[0] (20 nf),
        # 2 = after ups[1] + reflection pad (120 nf + 1). Each (real, 64-aligned) pair is affine
        # in the two preamble-primed frame registers.
        T0, T0p = Dim(2 * nf, GPR_F, mul=2, cap=GEN_T0_CAP), Dim(2 * nf_pad, GPR_NFPAD, mul=2, cap=GEN_T0_CAP)
        T1, T1p = Dim(20 * nf, GPR_F, mul=20, cap=GEN_T1_CAP), Dim(20 * nf_pad, GPR_NFPAD, mul=20, cap=GEN_T1_CAP)
        T2r = Dim(120 * nf, GPR_F, mul=120, cap=GEN_T2_CAP - 64)              # before the pad
        T2, T2p = (Dim(120 * nf + 1, GPR_F, mul=120, cap=GEN_T2_CAP - 63, off=1),
                   Dim(120 * nf_pad + 64, GPR_NFPAD, mul=120, cap=GEN_T2_CAP, off=64))
        self._caps = {0: GEN_T0_CAP, 1: GEN_T1_CAP, 2: GEN_T2_CAP}
        self._live = {}
        self._pool, self._owner = {}, {}
        F_ = int(T2)
        S = Dim(self.GEN_UP * 2 * nf, GPR_F, mul=self.GEN_UP * 2, cap=self.GEN_UP * GEN_T0_CAP)   # samples
        assert noise.shape == (int(S), self.GEN_HARM), (noise.shape, int(S))
        self.identity_dram = self._up(torch.eye(V, dtype=torch.bfloat16))
        self.style_dram = self._up_cap(_bf16(s.reshape(-1)), self.style_dim)
        debug_log = []

        def _check(name, addr, rows, C, cpu_ref):
            if debug_cpu_ref is None or cpu_ref is None:
                return
            got = ue.dma_from_accelerator_memory(addr, (int(rows), C)).float()
            ref = cpu_ref.detach().float()
            if ref.shape[0] != got.shape[0]:
                ref = ref.T                                         # [C, T] -> [T, C]
            got = got[:ref.shape[0], :ref.shape[1]]
            from user_dma_core import calculate_snr
            debug_log.append((name, calculate_snr(ref.reshape(-1), got.reshape(-1))))

        # ---- inputs ----
        C0 = x_gen.shape[0]
        x0 = torch.zeros(GEN_T0_CAP, C0, dtype=torch.bfloat16)
        x0[:int(T0)] = _bf16(x_gen.T)
        x = self._up_cap(x0, GEN_T0_CAP * C0)
        # ---- Section 5c: har [F, 64] on device (mag 0..10 | phase 11..21 | 0). Allocated with a
        # 64-row zero prefix so noise_convs[0]'s strided gather can read its left zero-padding.
        n_blocks = -(-GEN_T2_CAP // 64)
        HARB = ue.allocate_tensor_dram((64 + n_blocks * 64) * V * 2)
        HAR = HARB + 64 * V * 2
        tables = self._source_prep(F0_curve, noise, int(T0))
        self._program(lambda: (self._zero_rows(HARB, V, 64),
                               self._source_har(tables, T0, S, T2, HAR)))
        if debug_cpu_ref is not None and debug_cpu_ref.get("har") is not None:
            from user_dma_core import calculate_snr
            # Intermediate probes (scratch is rewound but untouched until the next program runs).
            sg = self.gen.m_source.l_sin_gen
            with torch.no_grad():
                f0u = self.gen.f0_upsamp(F0_curve.reshape(1, 1, -1)).transpose(1, 2)
                fn_ = f0u * torch.arange(1, sg.harmonic_num + 2, dtype=f0u.dtype)[None, None, :]
                sine_ref = (sg._f02sine(fn_) * sg.sine_amp * sg._f02uv(f0u) + noise.unsqueeze(0)).squeeze(0)
                merge_ref = self.gen.m_source.l_tanh(self.gen.m_source.l_linear(sine_ref.unsqueeze(0))).reshape(-1)
                spec_ref = torch.stft(merge_ref, self.n_fft, self.gen.stft.hop_length, self.n_fft,
                                      window=torch.hann_window(self.n_fft, periodic=True), center=True,
                                      return_complex=True)                       # [11, F]
            d = self._dbg
            S_ = int(S)
            outb = ue.dma_from_accelerator_memory(d["OUTB"] + self.GEN_HEAD_ROWS * V * 2, (S_, V)).float()
            debug_log.append(("5c sine lanes (time-major)", calculate_snr(sine_ref.reshape(-1), outb[:, :self.GEN_HARM].reshape(-1))))
            xp = ue.dma_from_accelerator_memory(d["XP"], (d["R"],)).float()
            debug_log.append(("5c tanh(l_linear) signal", calculate_snr(merge_ref, xp[self.GEN_HEAD_ROWS:self.GEN_HEAD_ROWS + S_])))
            head_ref = merge_ref[1:self.n_fft // 2 + 1].flip(0); tail_ref = merge_ref[-self.n_fft // 2 - 1:-1].flip(0)
            debug_log.append(("5c reflect head", calculate_snr(head_ref, xp[self.GEN_HEAD_ROWS - 10:self.GEN_HEAD_ROWS])))
            debug_log.append(("5c reflect tail", calculate_snr(tail_ref, xp[self.GEN_HEAD_ROWS + S_:self.GEN_HEAD_ROWS + S_ + 10])))
            nb_ = self.n_fft // 2 + 1
            re_hw = ue.dma_from_accelerator_memory(d["RE"], (F_, V)).float()[:, :nb_]
            im_hw = ue.dma_from_accelerator_memory(d["IM"], (F_, V)).float()[:, :nb_]
            debug_log.append(("5c STFT re", calculate_snr(spec_ref.real.T.reshape(-1), re_hw.reshape(-1))))
            debug_log.append(("5c STFT im", calculate_snr(spec_ref.imag.T.reshape(-1), im_hw.reshape(-1))))
            got = ue.dma_from_accelerator_memory(HAR, (F_, V)).float()
            ref_h = debug_cpu_ref["har"].detach().float().T                   # [F, 22]
            nb = self.n_fft // 2 + 1
            mag_snr = calculate_snr(ref_h[:, :nb].reshape(-1), got[:, :nb].reshape(-1))
            ang_r, ang_g = ref_h[:, nb:2 * nb], got[:, nb:2 * nb]
            unit_snr = calculate_snr(torch.cat([torch.cos(ang_r), torch.sin(ang_r)]).reshape(-1),
                                     torch.cat([torch.cos(ang_g), torch.sin(ang_g)]).reshape(-1))
            debug_log.append(("har |X| (5c)", mag_snr))
            debug_log.append(("har phase as (cos,sin) (5c)", unit_snr))
            # per-bin breakdown: is the phase error in the strong bins, the empty bins, or branch-cut flips?
            d = (ang_g - ang_r)
            lines = []
            for k in range(nb):
                u = calculate_snr(torch.cat([torch.cos(ang_r[:, k]), torch.sin(ang_r[:, k])]),
                                  torch.cat([torch.cos(ang_g[:, k]), torch.sin(ang_g[:, k])]))
                raw = calculate_snr(ang_r[:, k], ang_g[:, k])
                flips = (d[:, k].abs() > 3.0).float().mean().item() * 100
                m = calculate_snr(ref_h[:, k], got[:, k])
                lines.append(f"bin{k}: |X| {ref_h[:, k].abs().mean().item():.4f} mag {m:5.1f} dB, "
                             f"phase unit {u:5.1f} dB raw {raw:5.1f} dB, |dphi|>3: {flips:4.1f}%")
            report_snr("[fpga][debug] har per-bin:\n  " + "\n  ".join(lines))
            har2 = debug_cpu_ref.get("har2")                                   # second noise seed
            if har2 is not None:
                a2 = har2.detach().float().T[:, nb:2 * nb]
                debug_log.append(("  floor: har phase seed-vs-seed",
                                  calculate_snr(torch.cat([torch.cos(ang_r), torch.sin(ang_r)]).reshape(-1),
                                                torch.cat([torch.cos(a2), torch.sin(a2)]).reshape(-1))))
                with torch.no_grad():
                    nc_a = self.gen.noise_convs[0](debug_cpu_ref["har"].unsqueeze(0))
                    nc_b = self.gen.noise_convs[0](har2.unsqueeze(0))
                debug_log.append(("  floor: noise_conv0 seed-vs-seed", calculate_snr(nc_a.reshape(-1), nc_b.reshape(-1))))
            # the strided conv alone, driven by the CPU's har: isolates _noise_conv_strided from 5c
            HARB2 = ue.allocate_tensor_dram((64 + n_blocks * 64) * V * 2)
            h2 = torch.zeros(64 + n_blocks * 64, V, dtype=torch.bfloat16)
            h2[64:64 + F_, :self.C_har] = _bf16(debug_cpu_ref["har"].T)
            ue.dma_to_accelerator_memory(HARB2, h2.reshape(-1))
            nc0_ = self.noise_convs[0]
            probe = self._block(lambda o: self._noise_conv_strided(HARB2 + 64 * V * 2, T1p, nc0_["Cout"], nc0_["w"],
                                                                   nc0_["b"], nc0_["k"], nc0_["stride"], nc0_["pad"], o),
                                GEN_T1_CAP, nc0_["Cout"])
            got_nc = ue.dma_from_accelerator_memory(probe, (int(T1p), nc0_["Cout"])).float()[:int(T1)]
            debug_log.append(("noise_conv0 on device from CPU har", calculate_snr(
                debug_cpu_ref["noise_conv"][0].detach().float().T.reshape(-1), got_nc.reshape(-1))))
            self._release(probe)
        nc0, nc1 = self.noise_convs[0], self.noise_convs[1]
        W1 = _bf16(_pad_dim(nc1["w"].reshape(nc1["Cout"], -1), 1, V))
        noise_conv = {
            0: lambda o: self._noise_conv_strided(HAR, T1p, nc0["Cout"], nc0["w"], nc0["b"], nc0["k"],
                                                  nc0["stride"], nc0["pad"], o),
            1: lambda o: _dyn_matmul(ue, M=T2p, K=V, N=nc1["Cout"], A_DRAM_ADDR=HAR, B_DRAM_ADDR=self._up(W1),
                                     OUTPUT_DRAM_ADDR=o, C_DRAM_ADDR=self._up(nc1["b"]), bias_mode="broadcast_N"),
        }
        # Reduction masks are LIVE for the whole section: allocate them before the first block,
        # since a block rewinds the allocator to its own mark when it finishes.
        self._mask_vec(1, T1)
        self._mask_vec(2, T2)

        self._owner[x] = (GEN_T0_CAP, C0)          # let the pool recycle the input buffer too
        E = lambda T_, C_, a, b, o, **kw: _dyn_eltwise(ue, T_, C_, a, b, o, **kw)
        ref = lambda *keys: (debug_cpu_ref and _dig(debug_cpu_ref, keys))

        def stage(i, x, T_in, T_in_p, Cin, Cout, T_out, T_out_p, cap_in, cap_out, level, refl=None):
            u = self.ups[i]
            h = self._block(lambda o: self._leaky(x, T_in_p, Cin, o, 0.1), cap_in, Cin)
            self._release(x)
            xu = self._block(lambda o: self._conv_transpose(h, T_in, Cin, Cout, u["w"], u["b"], o, u["k"],
                                                            u["stride"], u["pad"], T_out_p), cap_out, Cout)
            self._release(h)
            if refl is not None:
                # ReflectionPad1d((1, 0)): out[0] = in[1], out[1 + t] = in[t]
                def _refl(o):
                    self._copy_rows(xu, o + Cout * 2, Cout, refl)
                    self._copy_rows(xu + Cout * 2, o, Cout, 1)
                x = self._block(_refl, cap_out, Cout)
                self._release(xu)
            else:
                x = xu
            _check(f"ups{i}", x, T_out, Cout, ref("ups", i))
            xs = self._block(noise_conv[i], cap_out, Cout)
            _check(f"noise_conv{i}", xs, T_out, Cout, ref("noise_conv", i))
            xr = self._resblock(xs, T_out_p, T_out, level, self.noise_res[i])
            self._release(xs)
            _check(f"noise_res{i}", xr, T_out, Cout, ref("noise_res", i))
            xsum = self._block(lambda o: E(T_out_p, Cout, x, xr, o, mode=UE_MODE.ELTWISE_ADD), cap_out, Cout)
            self._release(x); self._release(xr)
            acc = None
            for j_ in range(self.n_kernels):
                r = self._resblock(xsum, T_out_p, T_out, level, self.resblocks[i * self.n_kernels + j_])
                _check(f"resblock{i}.{j_}", r, T_out, Cout, ref("resblocks", i, j_))
                if acc is None:
                    acc = r
                else:
                    nacc = self._block(lambda o, a=acc, r=r: E(T_out_p, Cout, a, r, o, mode=UE_MODE.ELTWISE_ADD),
                                       cap_out, Cout)
                    self._release(acc); self._release(r)
                    acc = nacc
            self._release(xsum)
            out = self._block(lambda o: E(T_out_p, Cout, acc, None, o, mode=UE_MODE.MUL_BROADCAST,
                                          scalar=1.0 / self.n_kernels), cap_out, Cout)
            self._release(acc)
            _check(f"stage{i}", out, T_out, Cout, ref("stage", i))
            return out

        C1, C2 = self.ups[0]["Cout"], self.ups[1]["Cout"]
        x = stage(0, x, T0, T0p, C0, C1, T1, T1p, GEN_T0_CAP, GEN_T1_CAP, 1)
        x = stage(1, x, T1, T1p, C1, C2, T2, T2p, GEN_T1_CAP, GEN_T2_CAP, 2, refl=T2r)

        # ================= conv_post =================
        h = self._block(lambda o: self._leaky(x, T2p, C2, o, 0.01), GEN_T2_CAP, C2)
        self._release(x)
        cp = self.conv_post
        y = self._block(lambda o: self._conv1d(h, T2p, T2, C2, cp["Cout"], cp["w"], cp["b"], o, cp["k"], 1),
                        GEN_T2_CAP, cp["Cout"])
        self._release(h)
        _check("conv_post", y, T2, cp["Cout"], ref("conv_post"))
        # ================= Section 5d: exp/sin + iSTFT -> audio =================
        AUDIO = ue.allocate_tensor_dram(_template(S) * 2)
        self._program(lambda: self._istft(y, T2p, T2, S, AUDIO))
        n_bufs = sum(len(v) for v in self._pool.values()) + len(self._owner)
        report(f"[fpga] Section 5b: {n_bufs} pooled activation buffers, tensor high-water "
               f"{_DRAM_HIGH[0] / 2**20:.0f} MB")

        if debug_cpu_ref is not None:
            report_snr("\n[fpga][debug] --- Section 5b bisect (SNR vs CPU, same har) ---")
            for name, snr_db in debug_log:
                flag = "  <-- LOW" if snr_db < 30.0 else ""
                report_snr(f"[fpga][debug] {name:38s} {snr_db:8.2f} dB{flag}")
            report_snr("[fpga][debug] --- end bisect ---\n")

        out = ue.dma_from_accelerator_memory(y, (int(T2p), cp["Cout"])).float()[:F_, :self.C_har].T.contiguous()
        audio = ue.dma_from_accelerator_memory(AUDIO, (int(S),)).float()
        if debug_cpu_ref is not None:
            from user_dma_core import calculate_snr
            # 5d in isolation: the device iSTFT of the device's OWN conv_post output vs torch's
            d5 = self._dbg5d
            nb = self.n_fft // 2 + 1
            yr = out.T                                                          # [F, 22]
            sp = ue.dma_from_accelerator_memory(d5["spec"], (int(T2p), V)).float()[:F_, :nb]
            report_snr(f"[fpga][debug] 5d exp(logmag) via softmax:        {calculate_snr(torch.exp(yr[:, :nb]).reshape(-1), sp.reshape(-1)):.2f} dB")
            phi = ue.dma_from_accelerator_memory(d5["phi"], (int(T2p), V)).float()[:F_, :nb]
            report_snr(f"[fpga][debug] 5d phi = sin(x):                    {calculate_snr(torch.sin(yr[:, nb:]).reshape(-1), phi.reshape(-1)):.2f} dB")
            sn_ = ue.dma_from_accelerator_memory(d5["sn"], (int(T2p), V)).float()[:F_, :nb]
            cs_ = ue.dma_from_accelerator_memory(d5["cs"], (int(T2p), V)).float()[:F_, :nb]
            pr = torch.sin(yr[:, nb:])
            report_snr(f"[fpga][debug] 5d sin(phi), cos(phi):              {calculate_snr(torch.sin(pr).reshape(-1), sn_.reshape(-1)):.2f} / {calculate_snr(torch.cos(pr).reshape(-1), cs_.reshape(-1)):.2f} dB")
            spr = ue.dma_from_accelerator_memory(d5["SPEC"], (int(T2p), V)).float()[:F_]
            zr = torch.exp(yr[:, :nb]) * torch.exp(1j * pr)
            report_snr(f"[fpga][debug] 5d complex rows re / im:            {calculate_snr(zr.real.reshape(-1), spr[:, :nb].reshape(-1)):.2f} / {calculate_snr(zr.imag.reshape(-1), spr[:, nb:2 * nb].reshape(-1)):.2f} dB")
            zdev = torch.complex(spr[:, :nb], spr[:, nb:2 * nb]).T                # the device's own rows
            report_snr(f"[fpga][debug] 5d complex rows (power-weighted): "
                       f"{calculate_snr(torch.view_as_real(zr.T).reshape(-1), torch.view_as_real(zdev).reshape(-1)):.2f} dB; "
                       f"per-bin re/im dB: " + " ".join(f"{calculate_snr(torch.view_as_real(zr.T[k]).reshape(-1), torch.view_as_real(zdev[k]).reshape(-1)):.0f}" for k in range(nb)))
            a_ref_rows = torch.istft(zr.T, self.n_fft, self.gen.stft.hop_length, self.n_fft,
                                     window=torch.hann_window(self.n_fft, periodic=True), center=True)
            a_from_dev_rows = torch.istft(zdev, self.n_fft, self.gen.stft.hop_length, self.n_fft,
                                          window=torch.hann_window(self.n_fft, periodic=True), center=True)
            report_snr(f"[fpga][debug] 5d iSTFT alone (torch istft of the device rows vs device audio): "
                       f"{calculate_snr(a_from_dev_rows, audio[:a_from_dev_rows.numel()]):.2f} dB; "
                       f"torch istft(ref rows) vs torch istft(device rows): {calculate_snr(a_ref_rows, a_from_dev_rows):.2f} dB")
            report_snr(f"[fpga][debug] 5d exp/sin+iSTFT vs host istft of the same conv_post: "
                       f"{calculate_snr(generator_host_istft(self.gen, out).reshape(-1), audio):.2f} dB")
        ue.reset_tensor_dram_addr()
        return audio, out


def _dig(d, keys):
    for k in keys:
        d = d[k]
    return d


def generator_cpu_reference(gen, x, s, har):
    """Generator.forward from `har` onward on the CPU, given the SAME har the FPGA path used,
    recording every stage so the device bisect has a like-for-like reference. x: [1, C, T],
    s: [1, style_dim], har: [1, 22, F]. Returns (conv_post output [1, 22, F], refs dict)."""
    import torch.nn.functional as F
    refs = {"ups": [], "noise_conv": [], "noise_res": [], "resblocks": [], "stage": []}
    with torch.no_grad():
        for i in range(gen.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1)
            x_source = gen.noise_convs[i](har)
            refs["noise_conv"].append(x_source.squeeze(0))
            x_source = gen.noise_res[i](x_source, s)
            refs["noise_res"].append(x_source.squeeze(0))
            x = gen.ups[i](x)
            if i == gen.num_upsamples - 1:
                x = gen.reflection_pad(x)
            refs["ups"].append(x.squeeze(0))
            x = x + x_source
            xs, blk = None, []
            for j in range(gen.num_kernels):
                r = gen.resblocks[i * gen.num_kernels + j](x, s)
                blk.append(r.squeeze(0))
                xs = r if xs is None else xs + r
            refs["resblocks"].append(blk)
            x = xs / gen.num_kernels
            refs["stage"].append(x.squeeze(0))
        x = F.leaky_relu(x)
        x = gen.conv_post(x)
        refs["conv_post"] = x.squeeze(0)
    return x, refs


def generate_source_noise(gen, F0_curve):
    """SineGen's additive noise for this utterance: randn scaled by noise_amp (per frame: noise_std
    where voiced, sine_amp/3 where not), time-major [S, 9]. The model's own randomness, drawn on
    the host and fed to BOTH the device path and the CPU reference so they are comparable."""
    sg = gen.m_source.l_sin_gen
    with torch.no_grad():
        f0 = gen.f0_upsamp(F0_curve.reshape(1, 1, -1)).transpose(1, 2)          # [1, S, 1]
        uv = sg._f02uv(f0)
        noise_amp = uv * sg.noise_std + (1 - uv) * sg.sine_amp / 3
        return (noise_amp * torch.randn(1, f0.shape[1], sg.dim)).squeeze(0)     # [S, 9]


def generator_host_har(gen, F0_curve, noise):
    """Section 5c on the host, deterministic given `noise`: f0 upsample -> SineGen -> l_linear/tanh
    -> STFT -> har [1, 22, F]. Mirrors SourceModuleHnNSF.forward with the noise injected."""
    sg = gen.m_source.l_sin_gen
    with torch.no_grad():
        f0 = gen.f0_upsamp(F0_curve.reshape(1, 1, -1)).transpose(1, 2)
        fn = f0 * torch.arange(1, sg.harmonic_num + 2, dtype=f0.dtype)[None, None, :]
        sine = sg._f02sine(fn) * sg.sine_amp
        uv = sg._f02uv(f0)
        sine = sine * uv + noise.unsqueeze(0)
        merge = gen.m_source.l_tanh(gen.m_source.l_linear(sine))                # [1, S, 1]
        har_spec, har_phase = gen.stft.transform(merge.transpose(1, 2).squeeze(1))
        return torch.cat([har_spec, har_phase], dim=1)


def generator_host_istft(gen, y):
    """Section 5d on the host: y [22, F] (conv_post output) -> audio [samples]."""
    with torch.no_grad():
        y = y.reshape(1, *y.shape).float()
        n = gen.post_n_fft // 2 + 1
        spec = torch.exp(y[:, :n, :])
        phase = torch.sin(y[:, n:, :])
        return gen.stft.inverse(spec, phase).squeeze()


def run_fpga_forward(model, phonemes: str, ref_s: torch.FloatTensor, speed: float = 1.0,
                     dev: str = "xdma0", debug: bool = False, dump_programs: str = None,
                     bin_cache: str = None):
    """Entry point called from kokoro_test.py --fpga. Only runs the section(s) currently ported to
    hardware (Section 1: PL-BERT) and reports their SNR against the CPU reference, with a bisect
    down to sub-stage granularity within each layer. Deliberately does NOT fall back to running the
    rest of the model on CPU -- --fpga means "run and validate what's actually on the FPGA today",
    not "run the full pipeline with FPGA sprinkled in". Returns None; callers should not expect
    audio until later sections are ported (see the section checklist at the top of this file).
    """
    set_dma_device(dev)
    _DEBUG_SNR[0] = debug
    _DUMP_PROGRAMS[0] = dump_programs
    _BIN_CACHE[0] = bin_cache
    _BIN_LOADED[0] = False
    ue = UnifiedEngine(params_dram_base=KOKORO_PARAMS_BASE,
                       program_dram_base=KOKORO_PROGRAM_BASE,
                       tensor_dram_base=KOKORO_TENSOR_BASE)
    _instrument(ue)
    global _SILENT_MODE
    _SILENT_MODE = True   # hide the cores' per-call tiling/FLOP prints

    if bin_cache:
        _manifest = load_bin_cache(ue, bin_cache)
        if _manifest is not None:
            _BIN_LOADED[0] = True
            # Fast-forward both bump allocators past what the image already occupies, so anything
            # allocated below lands where it did at compile time.
            ue.allocate_params_dram(_manifest["params_size"])
            report(f"[fpga] REPLAY: loaded frozen image from {bin_cache} "
                   f"({len(_manifest['programs'])} programs); no instruction compilation this run")
        else:
            report(f"[fpga] COMPILE: no usable frozen image in {bin_cache}; compiling and saving one")

    input_ids_list = [model.vocab[p] for p in phonemes if p in model.vocab]
    input_ids = torch.LongTensor([0, *input_ids_list, 0])
    T = input_ids.shape[0]
    assert T + 0 <= model.context_length, (T, model.context_length)

    # Bring-up instrumentation: exact HF module references for the embedding stage and the single
    # shared transformer layer, so PLBertFPGA.forward can print per-stage SNR and pinpoint exactly
    # where a divergence starts, instead of only seeing one aggregate end-to-end number.
    albert = model.bert.albert
    debug_cpu_ref = {
        "embeddings": albert.embeddings,
        "map_in": albert.encoder.embedding_hidden_mapping_in,
        "layer": albert.encoder.albert_layer_groups[0].albert_layers[0],
        # No padding in this debug reference (full, unpadded T), so omitting the mask entirely is
        # mathematically identical to an all-zero additive mask -- and sidesteps a transformers-
        # version-specific internal reshape in AlbertLayer's SDPA path when a pre-extended 4D mask
        # is passed directly to a single layer (bypassing AlbertModel.forward's own mask handling).
        "ext_mask": None,
    }

    _begin("1: PL-BERT encoder")
    report(f"[fpga] Section 1 (PL-BERT): T={T} phoneme tokens ...")
    plbert = PLBertFPGA(model, ue)
    with torch.no_grad():
        d_en = plbert.forward(input_ids, debug_cpu_ref=(debug_cpu_ref if debug else None))  # [512, T]

        input_ids_b = input_ids.unsqueeze(0)
        text_mask = torch.zeros(1, T, dtype=torch.bool)
        bert_dur_cpu = model.bert(input_ids_b, attention_mask=(~text_mask).int())
        d_en_cpu = model.bert_encoder(bert_dur_cpu).transpose(-1, -2).squeeze(0)  # [512, T]
        from user_dma_core import calculate_snr
        snr_db = calculate_snr(d_en_cpu.reshape(-1), d_en.reshape(-1))
        report_snr(f"[fpga] Section 1 (PL-BERT) end-to-end SNR vs CPU: {snr_db:.2f} dB")

        # ---- Section 2: prosody/duration path ----
        # WIRED: fed from Section 1's own FPGA output (`d_en`), not the clean CPU value. The
        # per-section SNRs printed below are therefore CUMULATIVE (they include every upstream
        # section's error), not isolated. To bisect a regression back to a single section, swap
        # the input back to the corresponding *_cpu value and compare.
        style_vec = ref_s.reshape(-1)[128:256]
        input_lengths = torch.full((1,), T, dtype=torch.long)
        ref_s_b = ref_s.reshape(1, -1)
        s_cpu = ref_s_b[:, 128:]
        d_cpu = model.predictor.text_encoder(d_en_cpu.unsqueeze(0), s_cpu, input_lengths, text_mask)  # [1,T,640]
        lstm_out_cpu, _ = model.predictor.lstm(d_cpu)  # [1,T,512]
        duration_raw_cpu = model.predictor.duration_proj(lstm_out_cpu)  # [1,T,50]

        _begin("2: Prosody / duration")
        report(f"\n[fpga] Section 2 (prosody/duration): T={T} ...")
        prosody = ProsodyDurationFPGA(model, ue)
        debug_cpu_ref2 = {
            "d": d_cpu.squeeze(0), "lstm_out": lstm_out_cpu.squeeze(0), "duration_raw": duration_raw_cpu.squeeze(0),
        }
        d_fpga, pred_dur_fpga = prosody.forward(d_en, style_vec, T,
                                                debug_cpu_ref=(debug_cpu_ref2 if debug else None))

        duration_cpu = torch.sigmoid(duration_raw_cpu.squeeze(0)).sum(axis=-1)
        pred_dur_cpu = torch.round(duration_cpu).clamp(min=1).long()
        pred_dur_snr = calculate_snr(pred_dur_cpu.float(), pred_dur_fpga.float())
        report_snr(f"[fpga] Section 2 (prosody/duration) end-to-end SNR (d) vs CPU: "
              f"{calculate_snr(d_cpu.squeeze(0).reshape(-1), d_fpga.reshape(-1)):.2f} dB")
        report_snr(f"[fpga] Section 2 pred_dur SNR vs CPU (post-round, should be near-exact): {pred_dur_snr:.2f} dB")
        n_mismatch = (pred_dur_cpu != pred_dur_fpga).sum().item()
        report_snr(f"[fpga] Section 2 pred_dur exact-match: {T - n_mismatch}/{T} positions "
              f"(mismatches usually a rounding tie at a .5 boundary, not a correctness bug)")

        # ---- Section 3: F0/N prediction ----
        # WIRED: fed from Section 2's own FPGA outputs (d_fpga / pred_dur_fpga).
        #
        # The CPU reference for this section is built from the FPGA's OWN upstream values, not
        # from d_cpu/pred_dur_cpu. Two reasons, and the second is not optional:
        #   1. It makes the SNRs below measure Section 3's own arithmetic, given the inputs it
        #      actually received -- rather than smearing upstream error into every stage number.
        #   2. pred_dur sets a data-dependent SHAPE (n_frames = pred_dur.sum()). Once chaining
        #      makes pred_dur_fpga differ from pred_dur_cpu by even one rounding tie, the two
        #      pipelines produce different-length tensors and an end-to-end comparison is not
        #      merely noisy, it is undefined -- calculate_snr raises on the shape mismatch.
        # The upstream divergence is still reported: see the pred_dur exact-match line above.
        indices_cpu = torch.repeat_interleave(torch.arange(T), pred_dur_fpga)
        pred_aln_trg_cpu = torch.zeros((T, indices_cpu.shape[0]))
        pred_aln_trg_cpu[indices_cpu, torch.arange(indices_cpu.shape[0])] = 1
        pred_aln_trg_cpu = pred_aln_trg_cpu.unsqueeze(0)
        en_cpu = d_fpga.unsqueeze(0).float().transpose(-1, -2) @ pred_aln_trg_cpu  # [1,640,n_frames]
        shared_lstm_out_cpu, _ = model.predictor.shared(en_cpu.transpose(-1, -2))  # [1,n_frames,512]

        def run_branch_cpu(blocks, proj):
            x = shared_lstm_out_cpu.transpose(-1, -2)  # [1,512,n_frames], channels-first (CPU conv layout)
            block_outs = []
            for blk in blocks:
                x = blk(x, s_cpu)
                block_outs.append(x.squeeze(0).transpose(-1, -2))  # store as [T,C] to match our device layout
            proj_out = proj(x)  # [1,1,n_frames_final]
            return block_outs, proj_out.squeeze(1).squeeze(0)  # [n_frames_final]

        F0_block_outs_cpu, F0_pred_cpu = run_branch_cpu(model.predictor.F0, model.predictor.F0_proj)
        N_block_outs_cpu, N_pred_cpu = run_branch_cpu(model.predictor.N, model.predictor.N_proj)

        n_frames = int(pred_dur_fpga.sum().item())
        n_frames_cpu = int(pred_dur_cpu.sum().item())
        if n_frames != n_frames_cpu:
            report_snr(f"[fpga][note] n_frames: fpga={n_frames} vs pure-CPU={n_frames_cpu} "
                  f"({n_frames - n_frames_cpu:+d} frames from pred_dur rounding ties upstream). "
                  f"Section 3's reference is built from the FPGA alignment, so the SNRs below "
                  f"remain valid measures of Section 3 itself.")
        _begin("3: F0 / N prediction")
        report(f"\n[fpga] Section 3 (F0/N prediction): n_frames={n_frames} ...")
        f0n = F0NPredictionFPGA(model, ue)
        debug_cpu_ref3 = {
            "shared_out": shared_lstm_out_cpu.squeeze(0),
            "F0_blocks": F0_block_outs_cpu, "F0_pred_pre_squeeze": F0_pred_cpu,
            "N_blocks": N_block_outs_cpu, "N_pred_pre_squeeze": N_pred_cpu,
        }
        F0_pred_fpga, N_pred_fpga = f0n.forward(d_fpga, pred_dur_fpga, style_vec, T,
                                                 debug_cpu_ref=(debug_cpu_ref3 if debug else None))

        f0_snr = calculate_snr(F0_pred_cpu.reshape(-1), F0_pred_fpga.reshape(-1))
        n_snr = calculate_snr(N_pred_cpu.reshape(-1), N_pred_fpga.reshape(-1))
        report_snr(f"[fpga] Section 3 F0_pred end-to-end SNR vs CPU: {f0_snr:.2f} dB")
        report_snr(f"[fpga] Section 3 N_pred end-to-end SNR vs CPU: {n_snr:.2f} dB")

        # ---- Section 4: TextEncoder (phoneme embed + Conv1d k=5 stack + BiLSTM) ----
        # Independent of Sections 2/3: it consumes the raw phoneme ids, not the prosody path, so
        # it is fed the real input rather than any upstream FPGA output. Its result joins the
        # pipeline in Section 5, as `asr = t_en @ pred_aln_trg`.
        _begin("4: TextEncoder")
        report(f"\n[fpga] Section 4 (TextEncoder): T={T} phoneme tokens ...")
        te = model.text_encoder
        emb_cpu = te.embedding(input_ids_b).squeeze(0)                 # [T, C]
        xc = emb_cpu.transpose(0, 1).unsqueeze(0)                      # [1, C, T] channels-first
        cnn_cpu = []
        for blk in te.cnn:
            xc = blk(xc)
            cnn_cpu.append(xc.squeeze(0).transpose(0, 1))              # store [T, C] to match device
        lstm_cpu, _ = te.lstm(xc.transpose(-1, -2))                    # [1, T, C]
        t_en_cpu = te(input_ids_b, input_lengths, text_mask).squeeze(0)  # [C, T]

        textenc = TextEncoderFPGA(model, ue)
        t_en_fpga = textenc.forward(input_ids, T, debug_cpu_ref=({
            "embedding": emb_cpu, "cnn": cnn_cpu, "lstm": lstm_cpu.squeeze(0),
        } if debug else None))
        report_snr(f"[fpga] Section 4 t_en end-to-end SNR vs CPU: "
              f"{calculate_snr(t_en_cpu.reshape(-1), t_en_fpga.reshape(-1)):.2f} dB")

        # ---- Section 5a: Decoder front (encode + 4 decode AdainResBlk1d) ----
        # Wired from Sections 3 and 4's own FPGA outputs: asr from t_en_fpga, F0/N from Section 3.
        s_dec = ref_s.reshape(1, -1)[:, :128]                      # decoder style (predictor uses [128:])
        asr_fpga = t_en_fpga @ pred_aln_trg_cpu.squeeze(0)          # [C, n_frames]
        dec = model.decoder
        with torch.no_grad():
            F0c = dec.F0_conv(F0_pred_fpga.unsqueeze(0).unsqueeze(1)).squeeze(0)
            Nc_ = dec.N_conv(N_pred_fpga.unsqueeze(0).unsqueeze(1)).squeeze(0)
            asr_res_cpu = dec.asr_res(asr_fpga.unsqueeze(0))
            enc_cpu = dec.encode(torch.cat([asr_fpga.unsqueeze(0), F0c.unsqueeze(0), Nc_.unsqueeze(0)], axis=1), s_dec)
            dec_cpu, xc, res_ = [], enc_cpu, True
            for blk in dec.decode:
                if res_:
                    xc = torch.cat([xc, asr_res_cpu, F0c.unsqueeze(0), Nc_.unsqueeze(0)], axis=1)
                xc = blk(xc, s_dec)
                if blk.upsample_type != "none":
                    res_ = False
                dec_cpu.append(xc.squeeze(0).transpose(0, 1))       # [T, C] to match device layout

        _begin("5a: Decoder front")
        report(f"\n[fpga] Section 5a (Decoder front): asr={tuple(asr_fpga.shape)} ...")
        decoder = DecoderFPGA(model, ue)
        x_gen = decoder.forward(asr_fpga, F0_pred_fpga, N_pred_fpga, s_dec.reshape(-1),
                                debug_cpu_ref=({"encode": enc_cpu.squeeze(0).transpose(0, 1),
                                                "decode": dec_cpu} if debug else None))
        report_snr(f"[fpga] Section 5a decoder-front SNR vs CPU: "
              f"{calculate_snr(dec_cpu[-1].reshape(-1), x_gen.T.reshape(-1)):.2f} dB")

        # ---- Section 5b: ISTFTNet Generator body ----
        # `har` (Section 5c: SineGen + STFT) and exp/sin + iSTFT (Section 5d) are host-side for
        # now; both are HW-proven compositions in user_hw_test.py (sinegen_wrapped_phase_test,
        # cordic_atan2_magnitude_test, stft_istft_matmul_test) awaiting wiring. har's noise terms
        # are random, so it is computed ONCE and fed to both the FPGA path and the CPU reference.
        gen = dec.generator
        n_frames_dec = x_gen.shape[-1] // 2
        noise = generate_source_noise(gen, F0_pred_fpga)                        # [S, 9]
        har = generator_host_har(gen, F0_pred_fpga, noise)                     # [1, 22, F] reference
        y_cpu, gen_refs = generator_cpu_reference(gen, x_gen.unsqueeze(0), s_dec, har)
        gen_refs["har"] = har.squeeze(0)
        _begin("5b-5d: Generator")
        report(f"\n[fpga] Section 5b-5d (SineGen/STFT + Generator body + iSTFT): n_frames={n_frames_dec}, "
               f"F={har.shape[-1]} STFT frames, {har.shape[-1] * gen.stft.hop_length} samples ...")
        generator = GeneratorFPGA(model, ue)
        audio, y_fpga = generator.forward(x_gen, s_dec.reshape(-1), F0_pred_fpga, noise, n_frames_dec,
                                          debug_cpu_ref=(gen_refs if debug else None))     # [S], [22, F]
        report_snr(f"[fpga] Section 5b conv_post output SNR vs CPU (same noise): "
                   f"{calculate_snr(y_cpu.squeeze(0).reshape(-1), y_fpga.reshape(-1)):.2f} dB")
        # The model's OWN noise floor: the same CPU generator with a second noise draw. The FPGA
        # cannot be expected to track the CPU closer than the CPU tracks itself across seeds.
        y_cpu2, _ = generator_cpu_reference(
            gen, x_gen.unsqueeze(0), s_dec, generator_host_har(gen, F0_pred_fpga, generate_source_noise(gen, F0_pred_fpga)))
        report_snr(f"[fpga] Section 5b conv_post CPU-vs-CPU across noise seeds (floor): "
                   f"{calculate_snr(y_cpu.squeeze(0).reshape(-1), y_cpu2.squeeze(0).reshape(-1)):.2f} dB")
        audio_cpu = generator_host_istft(gen, y_cpu.squeeze(0))
        report_snr(f"[fpga] Section 5d audio CPU-vs-CPU across noise seeds (floor): "
                   f"{calculate_snr(audio_cpu.reshape(-1), generator_host_istft(gen, y_cpu2.squeeze(0)).reshape(-1)):.2f} dB")
        report_snr(f"[fpga] Section 5d audio SNR vs CPU (same noise; waveform SNR, see METRIC note): "
                   f"{calculate_snr(audio_cpu.reshape(-1), audio.reshape(-1)):.2f} dB")

        _CUR[0] = None
        report(f"[fpga] Pipeline complete: sections 1-5d on hardware -- phonemes in, samples out.")
        _report_timings()
        _dump_program_fingerprints()
        # Freeze the image on the first successful run so later runs skip emission entirely.
        # Written after the pipeline completes, so a crashed run never leaves a half image behind.
        if _BIN_CACHE[0] and not _BIN_LOADED[0]:
            dump_bin_cache(ue, _BIN_CACHE[0])
        _SILENT_MODE = False     # hand normal printing back to the caller
        return audio

    return None
