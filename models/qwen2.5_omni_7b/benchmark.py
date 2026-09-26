"""Fixed-shape benchmark tiers for qwen2.5_omni_7b: a voice command (Low), a
single-camera query (Medium), and an op-by-op multi-camera estimate (High).

Low and Medium own the WORKLOAD SHAPES; qwen2.5_omni_7b_test.py stays generic
(it only knows "--target-prefill-tokens N", "--prompt-base TEXT",
"--audio-seconds S" -- not "low"/"medium"). Both are real, continuous FPGA
runs: this script just builds the right argv and calls the main script,
once, exactly as a user would from the command line.

High runs the real vision and audio encoders in a separate media-only process,
then measures LM operations separately. A real single continuous ~6144-token
prefill does not fit the resident model's DRAM map; no such model run is claimed.
The LM benchmark compiles and runs each real op (qkv/o/gate/up/down projections,
attention) across the REAL number of engines it uses in the real model, each
engine on its real per-engine shard, barrier-synchronized at the same two
points a real compiled layer is: together at the start (workers park on the
master's flag; the master raises it once every worker has reached this
point), and the master held at completion until every worker has finished
(see _emit_start_barrier/_emit_completion_barrier) -- the same
flag_set/flag_check primitives user_hw_test.py's own
quantized_matmat_mul_multi_cores_test/unified_attention_test already use and
already trust for multi-engine timing, just driven at qwen2.5_omni_7b's real
(M, K, N) dims and real engine counts instead of a synthetic shape. The full
prefill/decode cost is then DERIVED: each op's measured group latency (the
master's own, which already includes real handshake/rendezvous overhead)
times how many times it really occurs in one forward pass (NL layers,
sequential) -- NOT times the engine count, since the engines within one op
already ran concurrently inside the measurement itself.

VALIDATED, NOT ASSUMED: --low/--medium --standalone run this exact method at
850/2048 tokens, dims a real end-to-end run CAN also reach, specifically so
the derived numbers can be checked against real ones before trusting this
method for High's unreachable ~6144.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(SCRIPT_DIR, "qwen2.5_omni_7b_test.py")
HIGH_MEDIA_SCRIPT = os.path.join(SCRIPT_DIR, "benchmark_high_media.py")
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==========================================================================
# WORKLOAD SHAPES
# ==========================================================================
# Low and Medium's own tier definitions -- moved out of qwen2.5_omni_7b_test.py,
# which no longer knows what "low"/"medium" mean. Audio duration is specified
# directly (not a fraction of the bundled clip) so the shape is reproducible
# regardless of what audio file is passed.
PRESETS = {
    "low": {
        "image": False,
        "vision_res": None,
        "audio_seconds": 2.55,
        "target_prefill_tokens": 850,
        "prompt_base": (
            "Transcribe the speech exactly as it is spoken, preserving the "
            "wording and the order of the terms."
        ),
    },
    "medium": {
        "image": True,
        "vision_res": "medium",
        "audio_seconds": 5.10,
        "target_prefill_tokens": 2048,
        "prompt_base": (
            "Do two things, in this order. First, transcribe the speech "
            "exactly as it is spoken, preserving the wording and the order "
            "of the terms. Second, describe the image in detail, covering "
            "the lighting, the terrain, the vegetation and the sky. Both "
            "parts are required."
        ),
    },
}


# ==========================================================================
# RUNNING THE MAIN SCRIPT
# ==========================================================================

def _argv_for(spec: dict, *, dev: str, multi_core: int, max_new_tokens: int,
              summary: str) -> list[str]:
    argv = [
        sys.executable, MAIN_SCRIPT,
        "--dev", dev, "--multi-core", str(multi_core),
        "--target-prefill-tokens", str(spec["target_prefill_tokens"]),
        "--prompt-base", spec["prompt_base"],
        "--audio", "--audio-seconds", str(spec["audio_seconds"]),
        "--max-new-tokens", str(max_new_tokens),
        "--summary", summary,
    ]
    if spec.get("image"):
        argv += ["--image", "--vision-res", spec["vision_res"]]
    return argv


def _run(argv: list[str], label: str, result_prefix: str = "TEST_RESULT: ") -> dict | None:
    """Run the main script to completion, streaming its output, and return
    its TEST_RESULT payload."""
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}", flush=True)
    print(" ".join(argv), flush=True)
    proc = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    result = None
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        if line.startswith(result_prefix):
            result = json.loads(line[len(result_prefix):])
    returncode = proc.wait()
    if returncode:
        raise RuntimeError(f"{label} exited with status {returncode}")
    return result


def run_low(args) -> dict | None:
    summary = args.summary or os.path.join(
        SCRIPT_DIR, "qwen2.5_omni_7b_test_xdma0_audio_low_multi-core_8.md")
    argv = _argv_for(
        PRESETS["low"], dev=args.dev, multi_core=args.multi_core,
        max_new_tokens=args.max_new_tokens, summary=summary,
    )
    return _run(argv, "LOW -- voice command")


def run_medium(args) -> dict | None:
    summary = args.summary or os.path.join(
        SCRIPT_DIR, "qwen2.5_omni_7b_test_xdma0_image+audio_medium_multi-core_8.md")
    argv = _argv_for(
        PRESETS["medium"], dev=args.dev, multi_core=args.multi_core,
        max_new_tokens=args.max_new_tokens, summary=summary,
    )
    return _run(argv, "MEDIUM -- single camera")


# ==========================================================================
# HIGH: OP-BY-OP MICRO-BENCHMARK, DERIVED TOTALS
# ==========================================================================
# The real model's own dims, read straight from its config -- no import of
# any qwen2.5_omni_7b_*.py module, so this never pulls in torch/the DRAM-map
# machinery those carry. Mirrors _lm_dims() in qwen2.5_omni_7b_lm.py.
CONFIG_PATH = os.path.join(SCRIPT_DIR, "qwen2.5_omni_7b_config.json")
HIGH_PREFILL_TOKENS = 6144          # 3 x 2048, the aggregate the High tier claims
HIGH_DECODE_CONTEXT = 6144          # resident KV depth decode reads at
# One global latency correction for op-by-op decode only. The full-model
# first-token HW times at matching contexts were 78.2 ms (Low, 850 rows) and
# 99.2 ms (Medium, 2048 rows); the isolated-op totals were 71.75 and 89.19
# ms. Their pooled ratio is 1.102, rounded to 1.10. Never scale op rows or
# full-model measurements: this is an empirical estimate, not a HW counter.
EMPIRICAL_DECODE_LATENCY_SCALE = 1.10


def _lm_dims() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        cfg = json.load(handle)
    fi = cfg["file_info"]
    kvh = int(fi["num_kv_heads"])
    g = int(fi["group_size"])
    return dict(
        H=int(fi["hidden_size"]), AHD=int(fi["actual_head_dim"]),
        KVH=kvh, G=g, QH=kvh * g, MLP=int(fi["mlp_elements"]),
        NL=int(fi["num_layers"]), VOCAB=int(fi["embedding_vocab"]),
    )


def _make_multi_engine_ues(dev: str, num_engines: int, *, attention_only: bool = False):
    """``num_engines`` engines on their board-assigned private windows.

    Use the board's private windows. Attention-only calls have no weights:
    on U50 their tensors live in the second private 512 MiB segment while ISA
    stays in the first; on a contiguous window, tensors run from the normal
    tensor offset to a 16 MiB ISA reservation at the far end. Never join U50's
    non-contiguous segments. Matmul calls keep the usual layout.
    """
    import user_dma_core as udc
    import multi_engine_shard as mes

    udc.set_dma_device(dev)
    engine_base_stride = 0x00010000
    windows = mes.board_private_windows(num_engines)
    ues = []
    for i, window in enumerate(windows):
        if attention_only:
            if len(window.segments) > 1:
                tensor_base, tensor_bytes = window.segments[1]
                program_base = window.base + mes.ENGINE_PROGRAM_OFFSET
                program_limit = window.base + window.primary_bytes
            else:
                tensor_base = window.base + mes.ENGINE_TENSOR_OFFSET
                program_base = window.base + window.primary_bytes - 16 * 2**20
                tensor_bytes = program_base - tensor_base
                program_limit = window.base + window.primary_bytes
            if tensor_bytes <= 0 or program_base < window.base:
                raise MemoryError(f"engine {i} has no valid attention tensor/ISA layout")
        else:
            tensor_base = window.base + mes.ENGINE_TENSOR_OFFSET
            program_base = window.base + mes.ENGINE_PROGRAM_OFFSET
        ue = udc.UnifiedEngine(
            BASE_ADDR=udc.UE_0_BASE_ADDR + i * engine_base_stride,
            params_dram_base=window.base,
            tensor_dram_base=tensor_base,
            program_dram_base=program_base,
        )
        if attention_only:
            ue._benchmark_tensor_limit = tensor_base + tensor_bytes
            ue._benchmark_program_limit = program_limit
        ues.append(ue)
    return ues


def _reset_engines_at_start(dev: str) -> None:
    """Clear stale queue state once, before the first benchmark measurement."""
    import user_dma_core as udc

    for ue in _make_multi_engine_ues(dev, 8):
        ue.write_reg32(udc.UE_QUEUE_CTRL_ADDR, 0x80008000)
        ue.wait_queue(1.0)


def _emit_start_barrier(ue, is_master: bool, ne: int) -> None:
    """All ``ne`` engines begin together: workers park on engine 0's flag,
    the master raises it once every worker's program has reached this
    point. Same primitive (flag_set/flag_check) qwen2.5_omni_7b's own
    MultiEngineScheduler handshake is built from -- this is where real
    cross-engine wait time actually gets paid, and what a single-engine
    measurement cannot pay at all."""
    if ne <= 1:
        return
    if is_master:
        ue.generate_instruction_flag_set()
    else:
        ue.generate_instruction_flag_clear()
        ue.generate_instruction_flag_check(target_engine_idx=0)


def _emit_completion_barrier(ue, is_master: bool, ne: int) -> None:
    """The master does not retire until every worker has flagged done, so
    the master's OWN measured latency already bounds the whole round --
    exactly how qwen2.5_omni_7b's real per-layer rendezvous is timed."""
    if ne <= 1:
        return
    if is_master:
        for j in range(1, ne):
            ue.generate_instruction_flag_check(target_engine_idx=j)
        ue.generate_instruction_flag_clear()
    else:
        ue.generate_instruction_flag_set()


def _measure_matmul(dev: str, M: int, K: int, N: int, ne: int, quantized: bool,
                    k_sharded: bool = False,
                    reduce_k: bool = False) -> tuple[float, int]:
    """``ne`` engines, each computing its real column (or, if ``k_sharded``,
    row) shard of one (M, K, N) projection, barrier-synchronized at the same
    two points the real compiled program is: together at the start, and the
    master held at completion until every worker is done. For decode down,
    ``reduce_k`` additionally times the seven full-width BF16 additions on
    the master, exactly as the compiled decode path does. Returns the
    MASTER's latency (the group's real completion time, handshake overhead
    included) and the FULL op's total FLOPs.

    IF4 weights are written as zero-filled data+scale buffers directly, NOT
    through UnifiedEngine.quantize_weight: that method's per-64-element-
    block quantization is a plain Python loop (a second loop with a
    .item() per element for the nibble packing) -- fine at the shapes
    user_hw_test.py exercises it with, but gate/up/down here reach 1.06M
    blocks, turning one call into many CPU-bound minutes with no hardware
    involved at all. Numerics do not matter for a timing benchmark (content
    is never read back), and zero content already proved safe on real
    hardware (a scale of 0 times data of 0 is 0, no NaN/Inf) in this
    project's earlier fake-weight work.
    """
    import torch
    import user_dma_core as udc

    IF4_SCALE_BYTES, IF4_DATA_BYTES = 2, 32   # one bf16 scale + 32 packed bytes per 64-elem block
    ne = max(1, ne)
    if reduce_k and not k_sharded:
        raise ValueError("reduce_k requires K-sharded weights")
    if k_sharded:
        assert K % ne == 0, f"K={K} does not split evenly over {ne} engines"
        k_shard, n_shard = K // ne, N
    else:
        assert N % ne == 0, f"N={N} does not split evenly over {ne} engines"
        k_shard, n_shard = K, N // ne

    ues = _make_multi_engine_ues(dev, ne)
    # All K-shard partials must be readable by the master after the workers
    # finish. The real decode uses one shared scratch plane for these rows.
    partial_base = (ues[0].allocate_tensor_dram(ne * M * N * 2)
                    if reduce_k else None)
    reduced_addr = (ues[0].allocate_tensor_dram(M * N * 2)
                    if reduce_k else None)
    prog_addrs = []
    for i, ue in enumerate(ues):
        is_master = i == 0
        a_addr = ue.allocate_tensor_dram(M * k_shard * 2)
        ue.dma_to_accelerator_memory(a_addr, torch.randn(M, k_shard, dtype=torch.bfloat16))
        out_addr = (partial_base + i * M * N * 2 if reduce_k
                    else ue.allocate_tensor_dram(M * n_shard * 2))
        ue.start_capture()
        _emit_start_barrier(ue, is_master, ne)
        if quantized:
            n_blocks = (n_shard * k_shard) // udc.UE_VECTOR_SIZE
            data_bytes = n_blocks * IF4_DATA_BYTES
            scale_bytes = n_blocks * IF4_SCALE_BYTES
            b_addr = ue.allocate_tensor_dram(data_bytes)
            ue.dma_write(udc.DMA_DEVICE_H2C, b_addr, torch.zeros(data_bytes, dtype=torch.uint8), data_bytes)
            s_addr = ue.allocate_tensor_dram(scale_bytes)
            ue.dma_to_accelerator_memory(s_addr, torch.zeros(scale_bytes // 2, dtype=torch.bfloat16))
            ue.quantized_matmat_core(
                M=M, K=k_shard, N=n_shard, A_DRAM_ADDR=a_addr, B_DRAM_ADDR=b_addr,
                OUTPUT_DRAM_ADDR=out_addr, SCALE_DRAM_ADDR=s_addr, data_type=udc.TYPE.IF4)
        else:
            b_addr = ue.allocate_tensor_dram(n_shard * k_shard * 2)
            ue.dma_to_accelerator_memory(b_addr, torch.randn(n_shard, k_shard, dtype=torch.bfloat16))
            ue.matmat_mul_core(M=M, K=k_shard, N=n_shard, A_DRAM_ADDR=a_addr, B_DRAM_ADDR=b_addr,
                              OUTPUT_DRAM_ADDR=out_addr)
        _emit_completion_barrier(ue, is_master, ne)
        if reduce_k and is_master:
            acc_addr = partial_base
            for j in range(1, ne):
                ue.eltwise_core_dram(
                    M=M, N=N, dram_a=acc_addr,
                    dram_b=partial_base + j * M * N * 2,
                    dram_out=reduced_addr, mode=udc.UE_MODE.ELTWISE_ADD)
                acc_addr = reduced_addr
        ue.generate_instruction_halt()
        ue.stop_capture()
        prog_addr = ue.get_program_dram_addr()
        ue.write_captured_instructions_to_dram(prog_addr)
        ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())
        prog_addrs.append(prog_addr)

    for i in range(1, ne):
        ues[i].start_execute_from_dram(prog_addrs[i])
    ues[0].start_execute_from_dram(prog_addrs[0])
    ues[0].wait_queue(60.0)
    master_us = ues[0].report_latency_in_us()
    for ue in ues[1:]:
        ue.wait_queue(60.0)
    return master_us, 2 * M * K * N + ((ne - 1) * M * N if reduce_k else 0)


def _measure_lm_head_and_argmax(dev: str, hidden: int, vocab: int,
                                ne: int) -> tuple[float, int]:
    """Time the real decode head sequence: eight IF4 column shards write one
    shared logits row (matmul writeback ENABLED, as the real model's shard
    always is). Weights and inputs are synthetic; the measured FPGA
    instructions and shape are real.

    Global argmax no longer costs any FPGA time: each engine's MMIO argmax
    register captures its own shard's local winner as a free side effect of
    the matmul above, and the host reduces the eight candidates by reading
    just those two-byte values back from the shared logits row (mirrors
    MultiEngineScheduler.global_argmax / qwen2.5_omni_7b_lm._decode_token).
    That reduction happens after wait_queue, off the HW counter entirely --
    it used to be an on-device 64-wide identity-matrix scan of the full
    vocabulary (~2376 tile dispatches) that dominated this op's latency; the
    returned time is now just the matmul's.
    """
    import torch
    import user_dma_core as udc

    if ne != 8 or vocab % (ne * udc.UE_VECTOR_SIZE):
        raise ValueError("the Omni LM head requires eight 64-column-aligned shards")
    shard_n = vocab // ne
    ues = _make_multi_engine_ues(dev, ne)
    master = ues[0]
    input_addr = master.allocate_tensor_dram(hidden * 2)
    logits_addr = master.allocate_tensor_dram(vocab * 2)
    bias_addr = master.allocate_tensor_dram(vocab * 2)
    master.dma_to_accelerator_memory(
        input_addr, torch.zeros(hidden, dtype=torch.bfloat16))
    master.dma_to_accelerator_memory(
        bias_addr, torch.zeros(vocab, dtype=torch.bfloat16))

    prog_addrs = []
    for i, ue in enumerate(ues):
        n_blocks = hidden * shard_n // udc.UE_VECTOR_SIZE
        weight_bytes = n_blocks * 32
        scale_bytes = n_blocks * 2
        weight_addr = ue.allocate_tensor_dram(weight_bytes)
        scale_addr = ue.allocate_tensor_dram(scale_bytes)
        if ue.get_tensor_dram_addr() > ue.get_program_dram_addr():
            raise MemoryError(f"LM-head buffers for engine {i} overlap ISA")
        ue.dma_write(
            udc.DMA_DEVICE_H2C, weight_addr,
            torch.zeros(weight_bytes, dtype=torch.uint8), weight_bytes)
        ue.dma_to_accelerator_memory(
            scale_addr, torch.zeros(n_blocks, dtype=torch.bfloat16))

        ue.start_capture()
        _emit_start_barrier(ue, i == 0, ne)
        ue.quantized_matmat_core(
            M=1, K=hidden, N=shard_n,
            A_DRAM_ADDR=input_addr, B_DRAM_ADDR=weight_addr,
            OUTPUT_DRAM_ADDR=logits_addr + i * shard_n * 2,
            SCALE_DRAM_ADDR=scale_addr, data_type=udc.TYPE.IF4,
            C_DRAM_ADDR=bias_addr + i * shard_n * 2,
            bias_mode="broadcast_N")
        _emit_completion_barrier(ue, i == 0, ne)
        ue.generate_instruction_halt()
        ue.stop_capture()
        prog_addr = ue.get_program_dram_addr()
        ue.write_captured_instructions_to_dram(prog_addr)
        ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())
        prog_addrs.append(prog_addr)

    for i in range(1, ne):
        ues[i].start_execute_from_dram(prog_addrs[i])
    master.start_execute_from_dram(prog_addrs[0])
    master.wait_queue(60.0)
    master_us = master.report_latency_in_us()
    for ue in ues[1:]:
        ue.wait_queue(60.0)

    # Host-side reduction, off the HW counter: each engine's local argmax
    # index into its own shard, then one 2-byte DRAM read per engine to
    # compare values. Exercises the real path; result is unused (synthetic
    # zero data ties every candidate), matching every other op's zero-fill.
    best_val = None
    for i, ue in enumerate(ues):
        local_idx = ue.get_arg_max_index()
        if not 0 <= local_idx < shard_n:
            raise RuntimeError(
                f"engine {i} argmax index {local_idx} outside its "
                f"{shard_n}-column shard")
        buf = bytearray(64)
        addr = logits_addr + (i * shard_n + local_idx) * 2
        base = addr & ~0x3F
        off = addr - base
        got = master.dma_read(udc.DMA_DEVICE_C2H, base, buf, 64)
        if got != 64:
            raise IOError(f"bf16 argmax-candidate read at 0x{base:X} returned {got} of 64 bytes")
        bits = int.from_bytes(buf[off:off + 2], "little") << 16
        val = struct.unpack("<f", bits.to_bytes(4, "little"))[0]
        if best_val is None or val > best_val:
            best_val = val
    return master_us, 2 * hidden * vocab


def _measure_attention(dev: str, batch: int, aligned_seq_len: int, head_dim: int,
                       ne: int, calls_per_engine: int = 1) -> tuple[float, int]:
    """``ne`` engines, each running ``calls_per_engine`` real
    unified_attention_core calls back to back (its own heads or KV-group;
    prefill's QH heads do not divide evenly over 8 engines, so the busiest
    engine runs more than one sequentially -- ceil(QH / ne) -- while every
    engine still starts and finishes at the same barriers). Mirrors
    user_hw_test.py's unified_attention_test _run_case (static/legacy leg)
    per call."""
    import torch
    import user_dma_core as udc

    ne = max(1, ne)
    calls_per_engine = max(1, calls_per_engine)
    needed = _attention_bytes(batch, aligned_seq_len, head_dim)
    ues = _make_multi_engine_ues(dev, ne, attention_only=True)
    for i, ue in enumerate(ues):
        available = ue._benchmark_tensor_limit - ue._tensor_dram_base
        if needed > available:
            raise MemoryError(
                f"attention needs {needed / 2**20:.1f} MiB for engine {i}, "
                f"but its mapped tensor segment has {available / 2**20:.1f} MiB"
            )
    bpe = 2
    prog_addrs = []
    total_flops = 0
    for i, ue in enumerate(ues):
        is_master = i == 0
        # Buffers are allocated once and reused by each head call. The program
        # executes the real full-size op; no reduced-size latency is substituted.
        q_addr = ue.allocate_tensor_dram(batch * head_dim * bpe)
        k_addr = ue.allocate_tensor_dram(aligned_seq_len * head_dim * bpe)
        v_addr = ue.allocate_tensor_dram(aligned_seq_len * head_dim * bpe)
        bias_addr = ue.allocate_tensor_dram(batch * aligned_seq_len * bpe)
        out_addr = ue.allocate_tensor_dram(batch * head_dim * bpe)
        scratch_addr = ue.allocate_tensor_dram(
            (head_dim * aligned_seq_len + aligned_seq_len * aligned_seq_len
             + batch * head_dim) * bpe)
        identity_addr = ue.allocate_tensor_dram(udc.UE_VECTOR_SIZE * udc.UE_VECTOR_SIZE * bpe)
        if ue.get_tensor_dram_addr() > ue._benchmark_tensor_limit:
            raise MemoryError(
                f"attention tensors for engine {i} exceed mapped segment: "
                f"0x{ue.get_tensor_dram_addr():X} > "
                f"0x{ue._benchmark_tensor_limit:X}"
            )

        ue.start_capture()
        _emit_start_barrier(ue, is_master, ne)
        for _call in range(calls_per_engine):
            flops = ue.unified_attention_core(
                batch=batch, aligned_seq_len=aligned_seq_len, head_dim=head_dim,
                Q_DRAM_ADDR=q_addr, K_DRAM_ADDR=k_addr, V_DRAM_ADDR=v_addr,
                BIAS_DRAM_ADDR=bias_addr, OUTPUT_DRAM_ADDR=out_addr,
                SCRATCH_DRAM_ADDR=scratch_addr, IDENTITY_DRAM_ADDR=identity_addr,
            )
            total_flops += flops
        _emit_completion_barrier(ue, is_master, ne)
        ue.generate_instruction_halt()
        ue.stop_capture()
        prog_addr = ue.get_program_dram_addr()
        prog_bytes = ue.get_capture_instruction_size_bytes()
        if prog_addr + prog_bytes > ue._benchmark_program_limit:
            raise MemoryError(
                f"attention ISA for engine {i} exceeds mapped segment: "
                f"0x{prog_addr + prog_bytes:X} > "
                f"0x{ue._benchmark_program_limit:X}"
            )
        ue.write_captured_instructions_to_dram(prog_addr)
        ue.allocate_program_dram(prog_bytes)
        prog_addrs.append(prog_addr)

        ue.dma_to_accelerator_memory(q_addr, torch.randn(batch, head_dim, dtype=torch.bfloat16))
        ue.dma_to_accelerator_memory(k_addr, torch.randn(aligned_seq_len, head_dim, dtype=torch.bfloat16))
        ue.dma_to_accelerator_memory(v_addr, torch.randn(aligned_seq_len, head_dim, dtype=torch.bfloat16))
        ue.dma_to_accelerator_memory(bias_addr, torch.zeros(batch, aligned_seq_len, dtype=torch.bfloat16))
        ue.dma_to_accelerator_memory(identity_addr, torch.eye(udc.UE_VECTOR_SIZE, dtype=torch.bfloat16))

    for i in range(1, ne):
        ues[i].start_execute_from_dram(prog_addrs[i])
    ues[0].start_execute_from_dram(prog_addrs[0])
    ues[0].wait_queue(60.0)
    master_us = ues[0].report_latency_in_us()
    for ue in ues[1:]:
        ue.wait_queue(60.0)
    return master_us, int(total_flops)


def _attention_bytes(batch: int, aligned_seq_len: int, head_dim: int, bpe: int = 2) -> int:
    """Per-engine tensor DRAM one unified_attention_core call needs: Q + K +
    V + bias + out + scratch + identity. Mirrors _measure_attention's own
    allocations exactly, so this can be checked BEFORE allocating anything."""
    scratch = (head_dim * aligned_seq_len + aligned_seq_len * aligned_seq_len
              + batch * head_dim) * bpe
    q = batch * head_dim * bpe
    k = aligned_seq_len * head_dim * bpe
    v = aligned_seq_len * head_dim * bpe
    bias = batch * aligned_seq_len * bpe
    out = batch * head_dim * bpe
    identity = 64 * 64 * bpe
    return scratch + q + k + v + bias + out + identity


def run_op_by_op(args, *, prefill_tokens: int, decode_context: int,
                 label: str, default_out_name: str,
                 media_result: dict | None = None,
                 media_result_fn=None) -> dict:
    """The op-by-op measurement + derivation, generic over the target dims.

    Used by --high (dims no real run can reach) and by --low/--medium
    --standalone (dims a real run CAN reach, specifically so the derived
    numbers can be checked against that real run's own measured GFLOPS/
    tok-s -- see the module docstring's validation argument).

    `media_result_fn`, if given, is called to run the vision/audio media
    phase AFTER all LM ops are measured but BEFORE the report is written --
    so the LM ops are measured cold, never immediately preceded by the
    media phase's sustained compute (see run_high).
    """
    import user_dma_core as udc
    udc.set_dma_device(args.dev)
    udc.configure_clock_from_hardware()

    d = _lm_dims()
    H, AHD, KVH, G, QH, MLP, NL = d["H"], d["AHD"], d["KVH"], d["G"], d["QH"], d["MLP"], d["NL"]
    query_dim, kv_dim = QH * AHD, KVH * AHD
    NE = 8   # engines the real MLP tensor-parallel shard splits across
    peak_gflops = NE * 2 * udc.UE_VECTOR_SIZE / udc.CLOCK_CYCLE_TIME_NS
    op_rows = {"prefill": [], "decode": []}
    phase = "prefill"
    print(f"\n{'=' * 72}\n{label} -- op-by-op, derived (dims: {d})\n{'=' * 72}")

    def measure(op_label, M, K, N, ne, quantized, k_sharded=False,
                reduce_k=False):
        us, flops = _measure_matmul(
            args.dev, M, K, N, ne, quantized,
            k_sharded=k_sharded, reduce_k=reduce_k)
        gflops = flops / (us * 1e3) if us else 0.0
        op_rows[phase].append((op_label, f"{M} x {K} x {N}", ne,
                               "IF4" if quantized else "BF16", us / 1e3, flops))
        print(f"  {op_label:28s} M={M:<6d} K={K:<6d} N={N:<6d} ne={ne:<2d} "
              f"{'IF4' if quantized else 'BF16':4s}  {us / 1e3:8.3f} ms  {gflops:7.1f} GFLOPS")
        return us / 1e3

    # ---- prefill: one instance of each op at the real aggregate row count --
    # gate/up/down are compiled and run across the REAL number of engines,
    # each on its real per-engine shard, barrier-synchronized at the same
    # two points the real compiled program is (_measure_matmul's start/
    # completion barriers): qwen2.5_omni_7b always runs its MLP tensor-
    # parallel across all 8 engines (_prefill_mlp_tp_engines), gate/up
    # N-sharded and down K-sharded (_emit_prefill_mlp_tp/_stage_private_lm_
    # projection). q/k/v/o are measured the same way: a --medium
    # --standalone check against the real run (see the module docstring)
    # showed full-width, single-engine q/o alone implied a ~2.3x slower
    # derived prefill than the real one, while the real qkv_proj/o_proj
    # phases run at 300-330 GFLOPS -- ~7-8x higher, i.e. also engine-
    # parallel. query_dim and kv_dim both divide 8 evenly (3584/8=448,
    # 512/8=64), matching gate/up's split.
    print(f"\n--- Prefill op shapes (M={prefill_tokens}) ---")
    pm = prefill_tokens
    q_ms = measure("q_proj", pm, H, query_dim, NE, True)
    k_ms = measure("k_proj", pm, H, kv_dim, NE, True)
    v_ms = measure("v_proj", pm, H, kv_dim, NE, False)
    o_ms = measure("o_proj (prefill, IF4)", pm, query_dim, H, NE, True)
    gate_ms = measure("gate_proj", pm, H, MLP, NE, True)
    up_ms = measure("up_proj", pm, H, MLP, NE, True)
    down_ms = measure("down_proj (K-shard)", pm, MLP, H, NE, True, k_sharded=True)
    p_aligned = ((prefill_tokens + 63) // 64) * 64
    # Prefill attention is ALSO engine-parallel: sched.head_sharded_region
    # splits all QH heads across up to NE engines concurrently
    # (mode="qheads"); QH does not divide NE evenly (28 / 8), so the busiest
    # engine runs ceil(QH / NE) heads back to back -- _measure_attention's
    # calls_per_engine -- while every engine still starts/finishes at the
    # same barriers as the matmul shards above.
    heads_per_engine = math.ceil(QH / NE)
    attn_us, attn_flops = _measure_attention(
        args.dev, pm, p_aligned, AHD, NE, calls_per_engine=heads_per_engine)
    attn_ms = attn_us / 1e3
    op_rows["prefill"].append((
                                f"attention ({NE * heads_per_engine} measured head slots; {QH} model heads)",
                                f"batch={pm}, context={p_aligned}, head_dim={AHD}",
                                NE, "BF16", attn_ms, attn_flops))
    print(f"  {'attention':28s} batch={pm:<6d} aligned={p_aligned:<6d} "
          f"head_dim={AHD:<6d} ne={NE:<2d} x{heads_per_engine}/engine  {attn_ms:8.3f} ms  "
          f"{attn_flops / (attn_us * 1e3) if attn_us else 0.0:7.1f} GFLOPS")

    proj_ms_per_layer = q_ms + k_ms + v_ms + o_ms + gate_ms + up_ms + down_ms
    attn_ms_per_layer = attn_ms   # the group's own measured latency already covers all QH heads
    prefill_total_ms = NL * (proj_ms_per_layer + attn_ms_per_layer)
    # True model attention FLOPs computed directly (4 * query_dim * causal
    # KV-length sum), NOT from attn_flops: _measure_attention now sums FLOPs
    # across every engine/call in the group (ne * calls_per_engine slots,
    # which can exceed the real QH when QH does not divide ne evenly), so it
    # is no longer "one head's flops" to multiply back up by QH.
    prefill_flops = NL * (
        2 * pm * H * query_dim + 2 * pm * H * kv_dim * 2 + 2 * pm * query_dim * H
        + 2 * pm * H * MLP * 2 + 2 * pm * MLP * H
        + 4 * query_dim * (pm * (pm + 1) // 2)
    )
    prefill_gflops = prefill_flops / (prefill_total_ms * 1e6) if prefill_total_ms else 0.0
    prefill_tok_s = 1000.0 * prefill_tokens / prefill_total_ms if prefill_total_ms else 0.0
    prefill_pct_peak = 100.0 * prefill_gflops / peak_gflops
    prefill_issued_flops = NL * sum(row[-1] for row in op_rows["prefill"])
    print(f"\nPrefill (derived): {prefill_total_ms:.1f} ms for {prefill_tokens} tokens "
          f"({prefill_tok_s:.2f} tok/s, {prefill_gflops:.1f} GFLOPS, "
          f"{prefill_pct_peak:.1f}% of {peak_gflops:.1f} GFLOPS peak, "
          f"{prefill_flops / 1e9:.1f} GFLOP)")

    # ---- decode: same projections at M=1, attention at the resident context -
    # Decode column-shards q/k/v/o across all 8 engines: K/V each have
    # KVH * AHD = 512 output columns, or eight 64-column shards. Gate/up
    # retain their private N-shards; down reuses prefill's private K-shards
    # and sums eight full-width outputs after the completion barrier.
    phase = "decode"
    print("\n--- Decode op shapes (M=1) ---")
    dq_ms = measure("q_proj", 1, H, query_dim, NE, True)
    dk_ms = measure("k_proj", 1, H, kv_dim, NE, True)
    dv_ms = measure("v_proj", 1, H, kv_dim, NE, True)
    do_ms = measure("o_proj (decode, IF4)", 1, query_dim, H, NE, True)
    dgate_ms = measure("gate_proj", 1, H, MLP, NE, True)
    dup_ms = measure("up_proj", 1, H, MLP, NE, True)
    ddown_ms = measure("down_proj (K-shard + reduction)", 1, MLP, H, NE,
                       True, k_sharded=True, reduce_k=True)
    d_aligned = ((decode_context + 63) // 64) * 64
    # Decode's KVH groups run CONCURRENTLY, one per engine
    # (_decode_use_one_round_group_attention: "four complete GQA groups
    # concurrently on engines 0..3"), so ne=KVH and one call each -- the same
    # barrier-synchronized measurement as everything else above, not an
    # analytic x1 assumption.
    dattn_us, dattn_flops = _measure_attention(args.dev, G, d_aligned, AHD, KVH)
    dattn_ms = dattn_us / 1e3
    op_rows["decode"].append((f"attention ({KVH} KV groups)",
                               f"batch={G}, context={d_aligned}, head_dim={AHD}",
                               KVH, "BF16", dattn_ms, dattn_flops))
    print(f"  {'attention':28s} batch={G:<6d} aligned={d_aligned:<6d} "
          f"head_dim={AHD:<6d} ne={KVH:<2d} x1/engine  {dattn_ms:8.3f} ms  "
          f"{dattn_flops / (dattn_us * 1e3) if dattn_us else 0.0:7.1f} GFLOPS")

    head_us, head_flops = _measure_lm_head_and_argmax(
        args.dev, H, d["VOCAB"], NE)
    head_ms = head_us / 1e3
    op_rows["decode"].append(("LM head (argmax reduce is host-side, no HW cost)",
                               f"1 x {H} x {d['VOCAB']}",
                               NE, "IF4", head_ms, head_flops))
    print(f"  {'LM head':28s} K={H:<6d} vocab={d['VOCAB']:<6d} "
          f"ne={NE:<2d}  {head_ms:8.3f} ms  "
          f"{head_flops / (head_us * 1e3) if head_us else 0.0:7.1f} GFLOPS")

    dproj_ms_per_layer = dq_ms + dk_ms + dv_ms + do_ms + dgate_ms + dup_ms + ddown_ms
    dattn_ms_per_layer = dattn_ms   # the group's own measured latency already covers all KVH groups
    decode_step_ms = NL * (dproj_ms_per_layer + dattn_ms_per_layer) + head_ms
    tok_s = 1000.0 / decode_step_ms if decode_step_ms else 0.0
    corrected_decode_ms = decode_step_ms * EMPIRICAL_DECODE_LATENCY_SCALE
    corrected_tok_s = 1000.0 / corrected_decode_ms if corrected_decode_ms else 0.0
    print(f"\nDecode step (derived) @ {decode_context} resident context: "
          f"{decode_step_ms:.2f} ms ({tok_s:.2f} tok/s)")
    print(f"Decode step (empirical x{EMPIRICAL_DECODE_LATENCY_SCALE:.2f} latency): "
          f"{corrected_decode_ms:.2f} ms ({corrected_tok_s:.2f} tok/s)")

    # All LM ops are measured by this point; only now (after the numbers
    # above are locked in) do we run the media phase, if requested.
    if media_result_fn is not None:
        if media_result is not None:
            raise ValueError("pass either media_result or media_result_fn, not both")
        media_result = media_result_fn()
        if media_result is None:
            raise RuntimeError("media phase did not return measurements; LM op-by-op numbers above are still valid, but the TTFT section will be omitted")

    def report_rows(rows):
        result = [
            "| Full operation | Full shape | Engines | Type | HW ms | Issued GFLOP | Issued GFLOPS | % 8-engine peak |",
            "| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: |",
        ]
        for name, shape, engines, dtype, ms, flops in rows:
            gflops = flops / (ms * 1e6) if ms else 0.0
            result.append(
                f"| {name} | {shape} | {engines} | {dtype} | {ms:.3f} | "
                f"{flops / 1e9:.3f} | {gflops:.1f} | "
                f"{100.0 * gflops / peak_gflops:.1f}% |"
            )
        return result

    media_lines = []
    if media_result is not None:
        media_peak = float(media_result["peak_gflops"])
        if abs(media_peak - peak_gflops) > 0.01 * peak_gflops:
            raise RuntimeError(f"media/LM peak mismatch: {media_peak} vs {peak_gflops}")
        vision = media_result["vision"]
        audio = media_result["audio"]
        vision_ms = float(vision["hw_us"]) / 1e3
        audio_ms = float(audio["hw_us"]) / 1e3
        ttft_ms = vision_ms + audio_ms + prefill_total_ms

        def stage_row(name, shape, flops, model_flops, ms):
            issued = flops / (ms * 1e6) if ms else 0.0
            effective = model_flops / (ms * 1e6) if ms else 0.0
            return (f"| {name} | {shape} | {flops / 1e9:.1f} | "
                    f"{model_flops / 1e9:.1f} | {ms:.1f} | {issued:.1f} | "
                    f"{100 * issued / peak_gflops:.1f}% | {effective:.1f} | "
                    f"{100 * effective / peak_gflops:.1f}% |")

        media_lines = [
            "## Vision/image, audio, and projected TTFT",
            "",
            f"Vision and audio are **real FPGA media-stage runs** on HW "
            f"{media_result['hw_version']}; their times come from FPGA counters. "
            "Four 896 x 896 vision invocations replay the same bundled image "
            "(same workload shape, not four distinct camera frames). Audio "
            f"uses the bundled sample trimmed to {float(audio['seconds']):.1f} s. "
            "The LM prefill time is derived from isolated operation counters, "
            "not a full-model prefill run.",
            "",
            "| Stage | Workload | Issued GFLOP | Model GFLOP | HW/derived ms | Issued GFLOPS | % 8-core peak | Effective GFLOPS | % 8-core peak |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            stage_row("Vision/image", f"{vision['frames']} x {vision['frame_shape']}",
                      float(vision["flops"]), float(vision["model_flops"]), vision_ms),
            stage_row("Audio", f"{audio['seconds']} s", float(audio["flops"]),
                      float(audio["model_flops"]), audio_ms),
            stage_row("LM prefill (derived)", f"{prefill_tokens} tokens",
                      prefill_issued_flops, prefill_flops, prefill_total_ms),
            "",
            f"- Vision: {vision_ms / vision['frames']:.1f} ms/frame, "
            f"{1000 * vision['soft_tokens'] / vision_ms:.1f} soft tokens/s "
            f"({vision['patches_per_frame']} patches/frame, "
            f"{vision['soft_tokens']} soft tokens total).",
            f"- Audio: {audio_ms:.1f} ms, {audio['soft_tokens']} soft tokens.",
            f"- **Projected hardware TTFT: {ttft_ms:.1f} ms** = "
            f"{vision_ms:.1f} ms vision + {audio_ms:.1f} ms audio + "
            f"{prefill_total_ms:.1f} ms derived LM prefill. This is **not** "
            "a measured end-to-end or CPU-timer TTFT. It excludes host "
            "preparation, weight/program loading, compilation, inter-stage "
            "scheduling, and LM operations omitted below.",
            "",
        ]

    out_path = args.summary or os.path.join(SCRIPT_DIR, default_out_name)
    lines = [
        f"# qwen2.5_omni_7b {label} tier -- op-by-op derived benchmark",
        "",
        "Each row is a **full operation across the listed engines**, not a "
        "single-engine shard. Engines execute their shards concurrently; the "
        "master's hardware-counter time includes their start/completion "
        "barriers. The benchmark times isolated operations at full shapes, "
        "not a compiled multi-layer model run. The `--high` op-by-op path "
        "uses a fixed 8-engine configuration; its `--multi-core` CLI value "
        "does not change that configuration.",
        "",
        f"- **Prefill:** {prefill_tokens} tokens -> "
        f"**{prefill_total_ms:.1f} ms**; **{prefill_tok_s:.2f} tok/s**; "
        f"{prefill_gflops:.1f} GFLOPS (**{prefill_pct_peak:.1f}%** of "
        f"{peak_gflops:.1f} GFLOPS 8-engine peak)",
        f"- **Decode step, raw derived:** {decode_context} resident context -> "
        f"**{decode_step_ms:.2f} ms** ({tok_s:.2f} tok/s)",
        f"- **Decode step, empirical correction:** raw latency x "
        f"{EMPIRICAL_DECODE_LATENCY_SCALE:.2f} -> "
        f"**{corrected_decode_ms:.2f} ms** ({corrected_tok_s:.2f} tok/s)",
        "The single correction factor is the rounded pooled ratio of "
        "full-model first-token HW times to isolated-op totals at matched "
        "contexts: Low 78.2 / 71.75 ms (850 rows), Medium 99.2 / 89.19 ms "
        "(2048 rows). Sources: `qwen2.5_omni_7b_test_xdma0_audio_low_multi-core_8.md` "
        "and `qwen2.5_omni_7b_test_xdma0_image+audio_medium_multi-core_8.md`. "
        "This is an estimate, not a measured full-model time; the operation "
        "rows below remain unscaled. Applying it at HIGH's 6144-row context "
        "is unvalidated extrapolation.",
        "",
        *media_lines,
        "## Prefill: one layer's measured full operations",
        "",
        *report_rows(op_rows["prefill"]),
        "",
        "## Decode: one layer's operations, then one LM head per token",
        "",
        *report_rows(op_rows["decode"]),
        "",
        "`Issued GFLOPS = issued GFLOP / hardware-counter seconds`; `% 8-engine "
        "peak` uses the board's full peak even for decode attention, which uses "
        f"only {KVH} engines (whose ceiling is {100.0 * KVH / NE:.0f}%). "
        f"Prefill attention measures {heads_per_engine} calls on each of "
        f"{NE} engines ({NE * heads_per_engine} issued head slots), although "
        f"the model has {QH} heads. The prefill summary above uses "
        f"{QH}-head model FLOPs, not those issued slots.",
        "",
        "**Coverage of the derived totals:** the listed projection and "
        "unified-attention group latencies, including their in-group "
        f"barriers, are summed and multiplied by {NL} layers. Decode then "
        "adds one eight-engine LM head per token; its global argmax is a "
        "host-side reduction of the eight shards' local-argmax candidates, "
        "off the HW counter, so it adds no measured FPGA time. Other "
        "operations are **omitted**, not measured or silently assigned "
        "zero cost: normalization, RoPE, residual/elementwise work other than "
        "decode down's seven measured reduction additions, "
        "embeddings, KV-cache setup, host transfers, "
        "weight loading, compilation, and inter-op scheduling. The "
        "unified-attention kernel's internal work is included in its row. "
        "Thus these are partial, derived LM timings—not end-to-end "
        "prefill/decode or TTFT measurements.",
        "",
    ]
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"\nWrote op-by-op report: {out_path}")
    return {
        "prefill_ms": prefill_total_ms, "prefill_gflops": prefill_gflops,
        "prefill_tok_s": prefill_tok_s, "prefill_pct_peak": prefill_pct_peak,
        "decode_step_ms": decode_step_ms, "decode_tok_s": tok_s,
        "decode_corrected_ms": corrected_decode_ms,
        "decode_corrected_tok_s": corrected_tok_s,
    }


def run_high(args) -> None:
    if args.multi_core != 8:
        raise ValueError("--high requires --multi-core 8 for its fixed 8-engine LM measurements")

    # The LM op-by-op measurement runs FIRST and cold, before any vision/
    # audio load touches the board. Running ~90s of real vision compute
    # immediately before the LM ops was found to measurably slow them (e.g.
    # q_proj/gate_proj ~15-17% lower GFLOPS than measured standalone) -- a
    # board-thermal artifact of THIS benchmark's own stage ordering, not a
    # model regression. media_result_fn defers the media subprocess until
    # after every LM op is measured, so the numbers above never carry that
    # bias; the media phase's own HW-counter numbers are unaffected by
    # ordering and are still merged into the same report.
    def _run_media():
        return _run([
            sys.executable, HIGH_MEDIA_SCRIPT, "--dev", args.dev,
            "--multi-core", str(args.multi_core), "--frames", "4",
            "--audio-seconds", "6.0",
        ], "HIGH real vision/image and audio", result_prefix="HIGH_MEDIA_RESULT: ")

    run_op_by_op(
        args, prefill_tokens=HIGH_PREFILL_TOKENS, decode_context=HIGH_DECODE_CONTEXT,
        label="HIGH", default_out_name="qwen2.5_omni_7b_benchmark_high_op_by_op.md",
        media_result_fn=_run_media,
    )


def run_standalone(args, tier: str) -> None:
    """--low/--medium --standalone: the SAME op-by-op method at that tier's
    real dims, so its derived prefill/decode numbers can be compared directly
    against a real `benchmark.py --low`/`--medium` run's measured numbers --
    the check that --high's own (unreachable) dims cannot get."""
    spec = PRESETS[tier]
    tokens = int(spec["target_prefill_tokens"])
    run_op_by_op(
        args, prefill_tokens=tokens, decode_context=tokens,
        label=f"{tier.upper()} --standalone",
        default_out_name=f"qwen2.5_omni_7b_benchmark_{tier}_standalone_op_by_op.md",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="qwen2.5_omni_7b fixed-shape benchmark: --low / --medium / --high",
    )
    parser.add_argument("--dev", default="xdma0")
    parser.add_argument("--multi-core", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--summary", default=None, metavar="PATH")
    parser.add_argument(
        "--standalone", action="store_true",
        help="with --low/--medium: run the op-by-op method (see --high) at "
             "that tier's real dims INSTEAD of a real end-to-end run, so its "
             "derived numbers can be checked against a plain --low/--medium "
             "run's real measured GFLOPS/tok-s -- the validation --high's "
             "own unreachable dims cannot get directly. Ignored with --high.",
    )
    tier = parser.add_mutually_exclusive_group(required=True)
    tier.add_argument("--low", action="store_true", help="voice command: real FPGA run")
    tier.add_argument("--medium", action="store_true", help="single camera: real FPGA run")
    tier.add_argument("--high", action="store_true",
                      help="four real full-resolution vision invocations and "
                           "real audio, then isolated 8-core LM operations "
                           "at HIGH dimensions for a projected TTFT")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.standalone:
        _reset_engines_at_start(args.dev)
    if args.low:
        run_standalone(args, "low") if args.standalone else run_low(args)
    elif args.medium:
        run_standalone(args, "medium") if args.standalone else run_medium(args)
    elif args.high:
        run_high(args)


if __name__ == "__main__":
    main()
