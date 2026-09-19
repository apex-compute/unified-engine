#!/usr/bin/env python3
"""
Kokoro op-composition hardware proofs (1-D / sequence-model set), moved out of the shared
user_hw_test.py when model/kokoro was rebased onto main. Unchanged in the move. Companion:
kokoro_trig_and_funcs.py (transcendental / generator proofs).

    python models/kokoro/utility/kokoro_hw_tests.py [--dev xdma0] [--summary-path FILE]
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
import torch.nn.functional as F

from user_dma_core import (
    INSTRUCTION_SIZE_BYTES, UE_MODE, UE_VECTOR_SIZE, UnifiedEngine, calculate_snr,
    set_dma_device, ue_35bit_addr_shifter,
)
from user_hw_test import record_test, write_test_summary, _round_up_vec

_ADAIN_SENTINEL = -12288.0


def embedding_gather_test(vocab_size: int = 178, embed_dim: int = 512, seq_len: int = 32):
    """Proves embedding-table gather (row lookup by integer id) reduces to pure address
    arithmetic + DMA on this hardware, with no new hardware primitive required.

    Mechanism demonstrated: each row's absolute DRAM address is computed as
    ``table_dram + id * embed_dim * 2`` bytes. Because the ids are known at *capture* time
    in this proof, we bake each row's address directly as the ``accelerator_dram_address``
    argument to a plain accelerator_memory_to_sram -> sram_to_accelerator_memory copy (an
    "offset row copy" per seq position), unrolled across seq_len in the Python capture loop.

    This is the simpler of the two mechanisms the task called out (as opposed to loading the
    row offset into a GPR via generate_instruction_add_set and using accelerator_memory_to_sram
    with general_reg_src=reg). A fully dynamic version -- where the ids themselves are only
    known at *runtime* (e.g. read from a DRAM-resident id buffer into a register) -- would:
      1. DMA the id buffer into SRAM / read it into a GPR.
      2. Compute the byte offset in a register with generate_instruction_mul32_imm(dst, id_reg,
         embed_dim * 2) (row stride) plus generate_instruction_add_imm/add_reg against the table
         base register.
      3. Issue accelerator_memory_to_sram(..., general_reg_src=offset_reg) so the DMA source
         address is sourced from the register instead of a Python-baked immediate.
    That extension is out of scope here; this test only proves the "address arithmetic + DMA"
    mechanism itself using capture-time-known ids.
    """
    ue = UnifiedEngine()

    table = torch.randn(vocab_size, embed_dim, dtype=torch.bfloat16)
    ids = torch.randint(0, vocab_size, (seq_len,))
    reference = table[ids]  # (seq_len, embed_dim)

    row_bytes = embed_dim * 2
    table_dram = ue.allocate_tensor_dram(vocab_size * embed_dim * 2)
    out_dram = ue.allocate_tensor_dram(seq_len * embed_dim * 2)

    # capture instructions
    ue.start_capture()
    for i in range(seq_len):
        row_id = int(ids[i].item())
        src_addr = table_dram + row_id * row_bytes
        dst_addr = out_dram + i * row_bytes
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=src_addr,
            sram_address=0x00000,
            element_size=embed_dim,
        )
        ue.sram_to_accelerator_memory(
            sram_address=0x00000,
            accelerator_dram_address=dst_addr,
            element_size=embed_dim,
        )
    ue.stop_capture()
    ue.generate_instruction_halt()
    prog = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(prog)
    inst_bytes = ue.get_capture_instruction_size_bytes()
    ue.allocate_program_dram(inst_bytes)

    ue.dma_to_accelerator_memory(table_dram, table.reshape(-1).contiguous())

    ue.start_execute_from_dram(prog)
    ue.wait_queue(10.0)
    ue.report_timing_and_instruction_count()

    result = ue.dma_from_accelerator_memory(out_dram, (seq_len * embed_dim,))
    snr_db = calculate_snr(reference.reshape(-1), result)
    print(f"Reference SNR Analysis for Embedding Gather Test: {snr_db:.2f} dB")
    assert snr_db >= 80.0 or snr_db == float("inf"), \
        f"embedding_gather SNR {snr_db:.2f} dB must be at least 80 dB (exact memcpy expected)"
    record_test("embedding_gather", f"vocab={vocab_size},dim={embed_dim},seq={seq_len}", snr_db=snr_db)

    ue.clear_capture_buffer(); ue.reset_tensor_dram_addr(); ue.reset_program_dram_addr()


def cumsum_via_triangular_matmul_test(T: int = 128, D: int = 64):
    """Proves a running cumulative sum along the time axis (needed e.g. for Kokoro's SineGen
    phase accumulation) can be computed with the EXISTING matmat_mul_core, no new hardware op,
    by matmul-ing against a precomputed constant lower-triangular all-ones matrix:
    cumsum(x, dim=0) == L @ x, where L = tril(ones(T, T)).

    matmat_mul_core computes A @ B^T with A: MxK, B: NxK. To get L @ x (L: TxT, x: TxD) we pass
    A=L (M=T, K=T) and B=x^T-shaped-as-NxK i.e. we need B such that B^T == x, i.e. B = x^T
    (D x T). Since matmat_mul_core's B layout is N x K (row-major) and the result is A @ B^T,
    to compute L @ x directly we instead pass A=L (TxT) and B=x with N=D, K=T requires B stored
    as D x T (x transposed). To avoid a host-side transpose headache we instead exploit
    symmetry-free simplicity: matmat_mul_core(M=T, K=T, N=D, A=L, B=x_as_NxK) expects B as N x K
    = D x T, i.e. x transposed. We DMA x already transposed (x.T contiguous, shape D x T) as B,
    so the hardware computes L @ (x.T)^T = L @ x, which is exactly cumsum(x, dim=0).
    """
    ue = UnifiedEngine()

    x = torch.randn(T, D, dtype=torch.bfloat16)
    L = torch.tril(torch.ones(T, T, dtype=torch.bfloat16))

    reference_true_cumsum = torch.cumsum(x.float(), dim=0)
    reference_matmul = (L.float() @ x.float())
    # Sanity: the bf16 triangular-matmul formulation itself should closely track true cumsum.
    cumsum_match_snr = calculate_snr(reference_true_cumsum.reshape(-1), reference_matmul.reshape(-1))
    print(f"[host sanity] L@x vs torch.cumsum SNR: {cumsum_match_snr:.2f} dB")

    l_dram = ue.allocate_tensor_dram(T * T * 2)
    x_dram = ue.allocate_tensor_dram(T * D * 2)
    out_dram = ue.allocate_tensor_dram(T * D * 2)

    # capture instructions
    ue.start_capture()
    total_flops = ue.matmat_mul_core(M=T, K=T, N=D, A_DRAM_ADDR=l_dram, B_DRAM_ADDR=x_dram,
                                      OUTPUT_DRAM_ADDR=out_dram)
    ue.stop_capture()
    ue.generate_instruction_halt()
    prog = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(prog)
    inst_bytes = ue.get_capture_instruction_size_bytes()
    ue.allocate_program_dram(inst_bytes)

    ue.dma_to_accelerator_memory(l_dram, L.reshape(-1).contiguous())
    ue.dma_to_accelerator_memory(x_dram, x.T.contiguous().reshape(-1))

    ue.start_execute_from_dram(prog)
    ue.wait_queue(10.0)
    ue.report_timing_and_instruction_count()

    result = ue.dma_from_accelerator_memory(out_dram, (T * D,))
    snr_db = calculate_snr(reference_matmul.reshape(-1), result)
    gflops, _ = ue.report_flop_rate_gflops(total_flops)
    print(f"Reference SNR Analysis for Cumsum-via-Triangular-Matmul Test: {snr_db:.2f} dB, GFLOPS={gflops:.2f}")
    assert snr_db >= 40.0 or snr_db == float("inf"), \
        f"cumsum_triangular_matmul SNR {snr_db:.2f} dB must be at least 40 dB"
    record_test("cumsum_triangular_matmul", f"T={T},D={D}", snr_db=snr_db, gflops=gflops)

    ue.clear_capture_buffer(); ue.reset_tensor_dram_addr(); ue.reset_program_dram_addr()


def lstm_cell_hw_test(input_size: int = 64, hidden_size: int = 64, T: int = 8,
                       snr_threshold_db: float = 25.0):
    """Single LSTMCell over T timesteps, decomposed into UnifiedEngine primitives.

    Proves a bidirectional-LSTM building block (as used 5x in Kokoro TTS,
    hidden=512/256-per-direction) needs ZERO new hardware primitives: each
    timestep is just batched gate matmuls (with fused bias) + native sigmoid +
    tanh derived exactly as tanh(x) = 2*sigmoid(2x) - 1 (affine pre/post
    transform around the native sigmoid activation) + elementwise mul/add for
    the cell/hidden state update.

    This is a *single* LSTMCell (no bidirectionality, no multi-layer stacking)
    over a short sequence, to validate the per-timestep decomposition against
    PyTorch's own nn.LSTMCell (bit-for-bit gate semantics: gate order i,f,g,o).
    The per-step op sequence below is unrolled in Python at capture time (a
    plain `for t in range(T)` while start_capture() is active) rather than
    using ISA loop_start/loop_end -- once this decomposition is validated on
    real hardware, the loop can be swapped for a true ISA loop.

    snr_threshold_db defaults to a relaxed 25 dB (vs. the ~35-40 dB floors used
    by single-op tests elsewhere in this file) because a single LSTM timestep
    chains many bf16-precision ops (2 matmuls + add + 3x sigmoid-via-identity-
    matmul + 2x tanh-via-sigmoid-affine + 3 elementwise ops), and T of those
    chain sequentially -- each step's bf16 rounding compounds into the next.
    """
    import torch
    from user_dma_core import UnifiedEngine, UE_MODE, UE_VECTOR_SIZE, calculate_snr

    assert input_size % UE_VECTOR_SIZE == 0, "input_size must be a multiple of UE_VECTOR_SIZE"
    assert hidden_size % UE_VECTOR_SIZE == 0, "hidden_size must be a multiple of UE_VECTOR_SIZE"
    H = hidden_size
    G = 4 * H  # gate width (i, f, g, o concatenated, PyTorch nn.LSTMCell order)

    # ---- Host reference: PyTorch's own nn.LSTMCell is ground truth. ----
    torch.manual_seed(0)
    cell = torch.nn.LSTMCell(input_size, hidden_size)
    x_seq = torch.randn(T, input_size)
    h_ref = torch.zeros(hidden_size)
    c_ref = torch.zeros(hidden_size)
    ref_outputs = []
    with torch.no_grad():
        for t in range(T):
            h_ref, c_ref = cell(x_seq[t:t + 1], (h_ref.unsqueeze(0), c_ref.unsqueeze(0)))
            h_ref = h_ref.squeeze(0)
            c_ref = c_ref.squeeze(0)
            ref_outputs.append(h_ref.clone())
    ref_outputs = torch.stack(ref_outputs, dim=0)  # [T, H]

    with torch.no_grad():
        W_ih = cell.weight_ih.detach()          # [4H, input_size]
        W_hh = cell.weight_hh.detach()           # [4H, H]
        bias = (cell.bias_ih.detach() + cell.bias_hh.detach())  # [4H], fold into one bias for hw matmul

    # matmat_mul_core computes A @ B^T with B stored [N,K] row-major (confirmed at
    # user_dma_core.py:5234-5254). We want gates_x = x_t @ W_ih^T, i.e. B^T == W_ih^T,
    # so B == W_ih AS-IS ([4H, input_size] == [N,K]) -- no transpose. Same for W_hh.
    Wx = W_ih.contiguous().to(torch.bfloat16)   # [4H, input_size] == [G, input_size] == [N,K]
    Wh = W_hh.contiguous().to(torch.bfloat16)   # [4H, H] == [G, H] == [N,K]
    bias_bf16 = bias.to(torch.bfloat16).contiguous()
    x_seq_bf16 = x_seq.to(torch.bfloat16).contiguous()

    ue = UnifiedEngine()

    # ---- DRAM buffer allocation (all bf16, 2 bytes/elem). ----
    wx_dram = ue.allocate_tensor_dram(input_size * G * 2)
    wh_dram = ue.allocate_tensor_dram(H * G * 2)
    bias_dram = ue.allocate_tensor_dram(G * 2)
    ident_dram = ue.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * 2)

    # Two input-x slots (not strictly needed to double-buffer, but keeps each
    # timestep's x_t contiguous and simple to DMA up-front).
    x_dram = ue.allocate_tensor_dram(T * input_size * 2)

    # Gate scratch buffers (reused every timestep -- values are fully consumed
    # each iteration before being overwritten on the next).
    gates_x_dram = ue.allocate_tensor_dram(G * 2)
    gates_h_dram = ue.allocate_tensor_dram(G * 2)
    gates_dram = ue.allocate_tensor_dram(G * 2)  # combined i,f,g,o pre-activations
    # Individual post-activation gate buffers [1, H] each.
    i_dram = ue.allocate_tensor_dram(H * 2)
    f_dram = ue.allocate_tensor_dram(H * 2)
    g_dram = ue.allocate_tensor_dram(H * 2)
    o_dram = ue.allocate_tensor_dram(H * 2)
    g_tmp_dram = ue.allocate_tensor_dram(H * 2)   # scratch for tanh-via-sigmoid affine on g
    c_tmp_dram = ue.allocate_tensor_dram(H * 2)   # scratch for tanh-via-sigmoid affine on c_t
    tanh_c_dram = ue.allocate_tensor_dram(H * 2)
    fc_dram = ue.allocate_tensor_dram(H * 2)      # f_t * c_{t-1}
    ig_dram = ue.allocate_tensor_dram(H * 2)       # i_t * g_t

    # Persistent recurrent state -- two physical buffers, ping-ponged by
    # (Python-level) address swap each timestep so h_prev/c_prev always point
    # at "last iteration's freshly written" buffer.
    h_bufs = [ue.allocate_tensor_dram(H * 2), ue.allocate_tensor_dram(H * 2)]
    c_bufs = [ue.allocate_tensor_dram(H * 2), ue.allocate_tensor_dram(H * 2)]

    out_seq_dram = ue.allocate_tensor_dram(T * H * 2)  # full h_t sequence, written each step

    M_H = H // UE_VECTOR_SIZE  # activation tiling: N=UE_VECTOR_SIZE, M=H/UE_VECTOR_SIZE

    def sigmoid_inplace(src_dram, dst_dram):
        ue.activation_core(M=M_H, N=UE_VECTOR_SIZE, A_DRAM_ADDR=src_dram,
                            OUTPUT_DRAM_ADDR=dst_dram, IDENTITY_DRAM_ADDR=ident_dram,
                            activation="sigmoid")

    def tanh_via_sigmoid(src_dram, tmp_dram, dst_dram):
        # tanh(x) = 2*sigmoid(2x) - 1
        ue.eltwise_core_dram(M=1, N=H, dram_a=src_dram, dram_b=None, dram_out=tmp_dram,
                              mode=UE_MODE.MUL_BROADCAST, scalar=2.0)
        sigmoid_inplace(tmp_dram, tmp_dram)
        ue.eltwise_core_dram(M=1, N=H, dram_a=tmp_dram, dram_b=None, dram_out=tmp_dram,
                              mode=UE_MODE.MUL_BROADCAST, scalar=2.0)
        ue.eltwise_core_dram(M=1, N=H, dram_a=tmp_dram, dram_b=None, dram_out=dst_dram,
                              mode=UE_MODE.ADD_BROADCAST, scalar=-1.0)

    # ---- Capture: unrolled T-timestep program. ----
    ue.start_capture()
    h_prev, c_prev = h_bufs[0], c_bufs[0]
    for t in range(T):
        x_t_dram = x_dram + t * input_size * 2
        h_cur, c_cur = h_bufs[(t + 1) % 2], c_bufs[(t + 1) % 2]

        # gates_x = x_t @ Wx + bias  (bias fused via C_DRAM_ADDR, broadcast over M=1 row)
        ue.matmat_mul_core(M=1, K=input_size, N=G, A_DRAM_ADDR=x_t_dram, B_DRAM_ADDR=wx_dram,
                            OUTPUT_DRAM_ADDR=gates_x_dram, C_DRAM_ADDR=bias_dram,
                            bias_mode="broadcast_N")
        # gates_h = h_prev @ Wh  (no bias, already folded into gates_x)
        ue.matmat_mul_core(M=1, K=H, N=G, A_DRAM_ADDR=h_prev, B_DRAM_ADDR=wh_dram,
                            OUTPUT_DRAM_ADDR=gates_h_dram)
        # gates = gates_x + gates_h
        ue.eltwise_core_dram(M=1, N=G, dram_a=gates_x_dram, dram_b=gates_h_dram,
                              dram_out=gates_dram, mode=UE_MODE.ELTWISE_ADD)

        # Slice gates into i,f,g,o (contiguous H-wide, H*2 bytes apart) and activate.
        i_raw, f_raw, g_raw, o_raw = (gates_dram + k * H * 2 for k in range(4))
        sigmoid_inplace(i_raw, i_dram)
        sigmoid_inplace(f_raw, f_dram)
        sigmoid_inplace(o_raw, o_dram)
        tanh_via_sigmoid(g_raw, g_tmp_dram, g_dram)

        # c_t = f_t * c_{t-1} + i_t * g_t
        ue.eltwise_core_dram(M=1, N=H, dram_a=f_dram, dram_b=c_prev, dram_out=fc_dram,
                              mode=UE_MODE.ELTWISE_MUL)
        ue.eltwise_core_dram(M=1, N=H, dram_a=i_dram, dram_b=g_dram, dram_out=ig_dram,
                              mode=UE_MODE.ELTWISE_MUL)
        ue.eltwise_core_dram(M=1, N=H, dram_a=fc_dram, dram_b=ig_dram, dram_out=c_cur,
                              mode=UE_MODE.ELTWISE_ADD)

        # h_t = o_t * tanh(c_t) -- write directly into both the recurrent h
        # buffer (h_cur, carried forward as next step's h_prev) and the output
        # sequence buffer at its fixed offset (two writes of the same result,
        # avoids a needless extra copy op).
        tanh_via_sigmoid(c_cur, c_tmp_dram, tanh_c_dram)
        ue.eltwise_core_dram(M=1, N=H, dram_a=o_dram, dram_b=tanh_c_dram, dram_out=h_cur,
                              mode=UE_MODE.ELTWISE_MUL)
        ue.eltwise_core_dram(M=1, N=H, dram_a=o_dram, dram_b=tanh_c_dram,
                              dram_out=out_seq_dram + t * H * 2, mode=UE_MODE.ELTWISE_MUL)

        h_prev, c_prev = h_cur, c_cur

    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    # ---- Upload weights/inputs (identical weights the PyTorch reference used). ----
    ue.dma_to_accelerator_memory(wx_dram, Wx.reshape(-1).contiguous())
    ue.dma_to_accelerator_memory(wh_dram, Wh.reshape(-1).contiguous())
    ue.dma_to_accelerator_memory(bias_dram, bias_bf16)
    ue.dma_to_accelerator_memory(ident_dram, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16).reshape(-1).contiguous())
    ue.dma_to_accelerator_memory(x_dram, x_seq_bf16.reshape(-1).contiguous())
    zeros_h = torch.zeros(H, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(h_bufs[0], zeros_h)
    ue.dma_to_accelerator_memory(c_bufs[0], zeros_h)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0)
    ue.report_timing_and_instruction_count()

    output = ue.dma_from_accelerator_memory(out_seq_dram, (T, H))
    snr_db = calculate_snr(ref_outputs.to(torch.bfloat16), output)
    print(f"Reference SNR Analysis for LSTM Cell Step Test: {snr_db:.2f} dB")
    assert snr_db >= snr_threshold_db or snr_db == float("inf"), \
        f"SNR {snr_db:.2f} dB must be at least {snr_threshold_db:g} dB"
    record_test("lstm_cell_step", f"input={input_size},hidden={hidden_size},T={T}", snr_db=snr_db)

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()
    ue.reset_program_dram_addr()


def conv1d_shifted_matmul_test(C_in: int = 64, C_out: int = 64, T: int = 128, kernel_size: int = 3, dilation: int = 1):
    """Prove Conv1d(C_in, C_out, kernel_size, dilation, padding='same') == sum_k of
    shifted-input matmuls against per-tap weight slices, i.e. y[t] = sum_k X_pad[t + dilation*k] @ W[k].
    The "shift" is pure DRAM address offset (zero extra compute); accumulation across taps uses
    matmat_mul_core's bias_mode="full_matrix" elementwise-add of the running total ([T, C_out]).
    """
    assert kernel_size % 2 == 1, "this test assumes an odd kernel_size for symmetric 'same' padding"
    assert C_in % UE_VECTOR_SIZE == 0 and C_out % UE_VECTOR_SIZE == 0 and T % UE_VECTOR_SIZE == 0

    pad = dilation * (kernel_size // 2)

    # ---- host reference (PyTorch Conv1d weight layout: [C_out, C_in, kernel_size]) ----
    X = torch.randn(T, C_in, dtype=torch.bfloat16) / math.sqrt(C_in)
    W = torch.randn(C_out, C_in, kernel_size, dtype=torch.bfloat16) / math.sqrt(C_in * kernel_size)

    x_nchw = X.T.unsqueeze(0).float()  # [1, C_in, T]
    ref = torch.nn.functional.conv1d(x_nchw, W.float(), padding=pad, dilation=dilation)  # [1, C_out, T]
    ref = ref.squeeze(0).T.to(torch.bfloat16)  # [T, C_out]

    X_pad = torch.zeros(T + 2 * pad, C_in, dtype=torch.bfloat16)
    X_pad[pad:pad + T, :] = X
    # per-tap weight slice: W[:, :, k] is [C_out, C_in] -> transpose to [C_in, C_out] for A@B.T layout
    # (matmat_mul_core computes A @ B.T with B stored as [N, K] = [C_out, C_in], so B is just W[:, :, k] itself)
    W_taps = [W[:, :, k].contiguous() for k in range(kernel_size)]  # each [C_out, C_in]

    # ---- device ----
    ue = UnifiedEngine()

    x_pad_dram = ue.allocate_tensor_dram((T + 2 * pad) * C_in * 2)
    w_tap_dram = [ue.allocate_tensor_dram(C_out * C_in * 2) for _ in range(kernel_size)]
    acc_a = ue.allocate_tensor_dram(T * C_out * 2)
    acc_b = ue.allocate_tensor_dram(T * C_out * 2)

    ue.start_capture()
    for k in range(kernel_size):
        A_DRAM_ADDR = x_pad_dram + (k * dilation) * C_in * 2  # pure address offset == "shifted read"
        prev_acc = acc_a if k % 2 == 0 else acc_b
        cur_acc = acc_b if k % 2 == 0 else acc_a
        ue.matmat_mul_core(
            M=T, K=C_in, N=C_out,
            A_DRAM_ADDR=A_DRAM_ADDR, B_DRAM_ADDR=w_tap_dram[k], OUTPUT_DRAM_ADDR=cur_acc,
            C_DRAM_ADDR=(prev_acc if k > 0 else None),
            bias_mode="full_matrix",
        )
    ue.stop_capture()
    ue.generate_instruction_halt()

    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    ue.dma_to_accelerator_memory(x_pad_dram, X_pad)
    for k in range(kernel_size):
        ue.dma_to_accelerator_memory(w_tap_dram[k], W_taps[k])

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0)
    ue.report_timing_and_instruction_count()

    # Must match the loop's own cur_acc formula at k = kernel_size - 1 (previously had the
    # branches swapped, silently reading the prior tap's partial sum instead of the final one).
    final_acc = acc_b if (kernel_size - 1) % 2 == 0 else acc_a
    out = ue.dma_from_accelerator_memory(final_acc, (T, C_out))

    snr_db = calculate_snr(ref, out)
    print(f"conv1d_shifted_matmul SNR: {snr_db:.2f} dB (Cin={C_in},Cout={C_out},T={T},k={kernel_size},dil={dilation})")
    assert snr_db >= 40.0 or snr_db == float("inf"), f"SNR {snr_db:.2f} dB must be at least 40 dB"

    gflops = (2 * T * C_in * C_out * kernel_size) / 1e9
    record_test("conv1d_shifted_matmul", f"Cin={C_in},Cout={C_out},T={T},k={kernel_size},dil={dilation}",
                 snr_db=snr_db, gflops=gflops)


def depthwise_conv1d_eltwise_test(C: int = 64, T: int = 128, kernel_size: int = 3):
    """Prove depthwise/grouped Conv1d (groups=C) reduces to kernel_size elementwise
    multiply-accumulate steps, no matmul: y[t, c] = sum_k X_pad[t+k, c] * w[c, k], with w[c, k]
    broadcast across T. No per-channel-vector broadcast op exists on this hardware
    (MUL_BROADCAST only broadcasts a scalar), so the per-tap weight row is host-tiled to a
    full [T, C] buffer once and applied with plain ELTWISE_MUL + ELTWISE_ADD.
    """
    assert kernel_size % 2 == 1
    assert C % UE_VECTOR_SIZE == 0 and T % UE_VECTOR_SIZE == 0

    pad = kernel_size // 2

    # ---- host reference (PyTorch depthwise Conv1d weight [C, 1, kernel_size]) ----
    X = torch.randn(T, C, dtype=torch.bfloat16)
    w = torch.randn(C, kernel_size, dtype=torch.bfloat16) / math.sqrt(kernel_size)

    ref = torch.nn.functional.conv1d(
        X.T.unsqueeze(0).float(), w.unsqueeze(1).float(), groups=C, padding=pad
    )  # [1, C, T]
    ref = ref.squeeze(0).T.to(torch.bfloat16)  # [T, C]

    X_pad = torch.zeros(T + 2 * pad, C, dtype=torch.bfloat16)
    X_pad[pad:pad + T, :] = X
    # host-side "precompute": tile each tap's per-channel weight row across T (constant, built once)
    w_tap_tiled = [w[:, k].unsqueeze(0).expand(T, C).contiguous() for k in range(kernel_size)]

    # ---- device ----
    ue = UnifiedEngine()

    x_pad_dram = ue.allocate_tensor_dram((T + 2 * pad) * C * 2)
    w_tap_dram = [ue.allocate_tensor_dram(T * C * 2) for _ in range(kernel_size)]
    tmp_dram = ue.allocate_tensor_dram(T * C * 2)
    acc_a = ue.allocate_tensor_dram(T * C * 2)
    acc_b = ue.allocate_tensor_dram(T * C * 2)

    # Per-tap DRAM buffer holding X_pad[k:k+T] * w_tap[k] (the elementwise product); tap 0's
    # product IS the running total (no add needed yet), later taps accumulate via ELTWISE_ADD
    # into ping-pong buffers acc_a/acc_b.
    tmp_dram_per_tap = [tmp_dram] + [ue.allocate_tensor_dram(T * C * 2) for _ in range(kernel_size - 1)]

    ue.start_capture()
    for k in range(kernel_size):
        shifted_x = x_pad_dram + k * C * 2  # pure address offset == shifted read
        ue.eltwise_core_dram(M=T, N=C, dram_a=shifted_x, dram_b=w_tap_dram[k], dram_out=tmp_dram_per_tap[k], mode=UE_MODE.ELTWISE_MUL)
        if k == 0:
            running_total = tmp_dram_per_tap[0]
            continue
        prev_total = running_total
        cur_total = acc_a if k % 2 == 1 else acc_b
        ue.eltwise_core_dram(M=T, N=C, dram_a=tmp_dram_per_tap[k], dram_b=prev_total, dram_out=cur_total, mode=UE_MODE.ELTWISE_ADD)
        running_total = cur_total
    final_acc = running_total  # captured after the loop; valid since Python `for` leaks its scope
    ue.stop_capture()
    ue.generate_instruction_halt()

    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    ue.dma_to_accelerator_memory(x_pad_dram, X_pad)
    for k in range(kernel_size):
        ue.dma_to_accelerator_memory(w_tap_dram[k], w_tap_tiled[k])

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0)
    ue.report_timing_and_instruction_count()

    out = ue.dma_from_accelerator_memory(final_acc, (T, C))

    snr_db = calculate_snr(ref, out)
    print(f"depthwise_conv1d_eltwise SNR: {snr_db:.2f} dB (C={C},T={T},k={kernel_size})")
    assert snr_db >= 40.0 or snr_db == float("inf"), f"SNR {snr_db:.2f} dB must be at least 40 dB"

    record_test("depthwise_conv1d_eltwise", f"C={C},T={T},k={kernel_size}", snr_db=snr_db)


def interpolate_fixed_matmul_test(T_in: int = 64, C: int = 64, scale_factor: int = 4, mode: str = "linear"):
    """Fixed-ratio 1D upsample (F.interpolate, mode='linear'/'nearest') is a precomputed
    constant matmul: M_interp [T_out, T_in] @ X [T_in, C] == interpolate(X). Since the scale
    factor is a compile-time constant (Kokoro SineGen / UpSample1d), no new hardware op is
    needed -- this is executed on the existing matmat_mul_core.
    """
    T_out = T_in * scale_factor
    assert T_in % UE_VECTOR_SIZE == 0 and C % UE_VECTOR_SIZE == 0, \
        "T_in/C must be multiples of UE_VECTOR_SIZE for matmat_mul_core alignment"

    X = torch.randn(T_in, C, dtype=torch.bfloat16)

    # --- Host precompute: build the interpolation matrix M_interp [T_out, T_in] ---
    M_interp = torch.zeros(T_out, T_in, dtype=torch.float32)
    if mode == "nearest":
        for t_out in range(T_out):
            t_in = min(t_out // scale_factor, T_in - 1)
            M_interp[t_out, t_in] = 1.0
    elif mode == "linear":
        # align_corners=False convention (PyTorch default for interpolate).
        for t_out in range(T_out):
            src = (t_out + 0.5) / scale_factor - 0.5
            i0 = math.floor(src)
            frac = src - i0
            i0c = min(max(i0, 0), T_in - 1)
            i1c = min(max(i0 + 1, 0), T_in - 1)
            M_interp[t_out, i0c] += (1.0 - frac)
            M_interp[t_out, i1c] += frac
    else:
        raise ValueError(f"unsupported mode: {mode}")

    # --- Verify the constructed matrix matches torch's F.interpolate BEFORE touching hardware ---
    ref = F.interpolate(
        X.T.unsqueeze(0).float(), scale_factor=scale_factor, mode=mode,
        align_corners=False if mode == "linear" else None,
    ).squeeze(0).T  # [T_out, C]
    host_pred = (M_interp @ X.float())
    host_snr_db = calculate_snr(ref, host_pred)
    print(f"[interpolate_fixed_matmul_test] host-side matrix-vs-F.interpolate SNR: {host_snr_db:.2f} dB")
    if not (host_snr_db >= 35 or host_snr_db == float("inf")):
        print(f"WARNING: constructed interpolation matrix only reaches {host_snr_db:.2f} dB "
              f"against F.interpolate's align_corners=False convention; falling back is not "
              f"implemented here, proceeding anyway (see docstring).")

    M_interp_bf16 = M_interp.to(torch.bfloat16)

    ue = UnifiedEngine()
    # matmat_mul_core computes A @ B^T with A:[M,K] row-major, B:[N,K] row-major.
    # We want M_interp[T_out,T_in] @ X[T_in,C] == M_interp @ (X^T)^T, so B must be X^T, [C,T_in].
    A_DRAM_ADDR = ue.allocate_tensor_dram(T_out * T_in * 2)
    B_DRAM_ADDR = ue.allocate_tensor_dram(C * T_in * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(T_out * C * 2)

    ue.start_capture()
    ue.matmat_mul_core(
        M=T_out, K=T_in, N=C,
        A_DRAM_ADDR=A_DRAM_ADDR, B_DRAM_ADDR=B_DRAM_ADDR, OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
    )
    ue.stop_capture()
    ue.generate_instruction_halt()

    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    X_T = X.T.contiguous()  # [C, T_in]
    ue.dma_to_accelerator_memory(A_DRAM_ADDR, M_interp_bf16)
    ue.dma_to_accelerator_memory(B_DRAM_ADDR, X_T)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0)
    ue.report_timing_and_instruction_count()

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (T_out, C))

    snr_threshold = 35.0 if mode == "linear" else 40.0
    snr_db = calculate_snr(ref.to(torch.bfloat16), output)
    print(f"[interpolate_fixed_matmul_test] device SNR vs F.interpolate: {snr_db:.2f} dB")
    assert snr_db >= snr_threshold or snr_db == float("inf"), \
        f"SNR {snr_db:.2f} dB must be at least {snr_threshold} dB"

    gflops = 2 * T_out * T_in * C / 1e9
    record_test("interpolate_fixed_matmul", f"Tin={T_in},C={C},scale={scale_factor},mode={mode}",
                snr_db=snr_db, gflops=gflops)


def conv_transpose1d_zero_insert_test(C_in: int = 64, C_out: int = 64, T_in: int = 32,
                                       stride: int = 4, kernel_size: int = 8):
    """ConvTranspose1d(C_in, C_out, kernel_size, stride) reduces to zero-insertion upsample
    (pure strided memory scatter) + regular Conv1d with the flipped kernel (shifted-matmul-
    accumulate, same pattern as conv1d_shifted_matmul_test).
    """
    assert C_in % UE_VECTOR_SIZE == 0 and C_out % UE_VECTOR_SIZE == 0, \
        "C_in/C_out must be multiples of UE_VECTOR_SIZE for matmat_mul_core alignment"

    X = torch.randn(T_in, C_in, dtype=torch.bfloat16)
    W = torch.randn(C_in, C_out, kernel_size, dtype=torch.bfloat16) * 0.1  # nn.ConvTranspose1d layout

    T_out = (T_in - 1) * stride + kernel_size

    ref = F.conv_transpose1d(
        X.T.unsqueeze(0).float(), W.float(), stride=stride
    ).squeeze(0).T  # [T_out, C_out]

    # --- Step 1: zero-insertion dilation (host-built; see deviation note at top of file) ---
    dilated_len = (T_in - 1) * stride + 1
    X_dilated = torch.zeros(dilated_len, C_in, dtype=torch.bfloat16)
    X_dilated[::stride, :] = X

    # --- Step 2: zero-pad by kernel_size-1 on both sides ---
    pad = kernel_size - 1
    L_pad = dilated_len + 2 * pad
    X_dilated_pad = torch.zeros(L_pad, C_in, dtype=torch.bfloat16)
    X_dilated_pad[pad:pad + dilated_len, :] = X_dilated

    # Sanity check the length arithmetic: T_out taps of shifted-window matmul over X_dilated_pad
    # must produce exactly T_out output rows: L_pad - kernel_size + 1 == T_out.
    assert L_pad - kernel_size + 1 == T_out, \
        f"dilated+padded length mismatch: L_pad={L_pad}, kernel_size={kernel_size}, T_out={T_out}"

    # Flip the kernel taps (conv/conv_transpose duality): Wk[k] = W[:, :, kernel_size-1-k], [C_in,C_out]
    Wk = [W[:, :, kernel_size - 1 - k].contiguous() for k in range(kernel_size)]

    ue = UnifiedEngine()
    x_dilated_pad_dram = ue.allocate_tensor_dram(L_pad * C_in * 2)
    wk_dram = [ue.allocate_tensor_dram(C_out * C_in * 2) for _ in range(kernel_size)]
    out_dram = ue.allocate_tensor_dram(T_out * C_out * 2)

    ue.start_capture()
    for k in range(kernel_size):
        # Shifted read: row k of X_dilated_pad, T_out contiguous rows, stride C_in*2 bytes/row.
        a_addr = x_dilated_pad_dram + k * C_in * 2
        if k == 0:
            ue.matmat_mul_core(
                M=T_out, K=C_in, N=C_out,
                A_DRAM_ADDR=a_addr, B_DRAM_ADDR=wk_dram[k], OUTPUT_DRAM_ADDR=out_dram,
            )
        else:
            # Accumulate into the running sum: out = (A_k @ Wk[k]^T) + out, C_DRAM_ADDR=out_dram
            # (bias_mode="full_matrix") reads the previous partial sum as the bias-add operand
            # and the result is written back to the same address.
            ue.matmat_mul_core(
                M=T_out, K=C_in, N=C_out,
                A_DRAM_ADDR=a_addr, B_DRAM_ADDR=wk_dram[k], OUTPUT_DRAM_ADDR=out_dram,
                C_DRAM_ADDR=out_dram, bias_mode="full_matrix",
            )
    ue.stop_capture()
    ue.generate_instruction_halt()

    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    ue.dma_to_accelerator_memory(x_dilated_pad_dram, X_dilated_pad)
    # B operand for matmat_mul_core must be [N,K] = [C_out, C_in], i.e. Wk[k]^T.
    for k in range(kernel_size):
        ue.dma_to_accelerator_memory(wk_dram[k], Wk[k].T.contiguous())

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0)
    ue.report_timing_and_instruction_count()

    output = ue.dma_from_accelerator_memory(out_dram, (T_out, C_out))

    snr_db = calculate_snr(ref.to(torch.bfloat16), output)
    print(f"[conv_transpose1d_zero_insert_test] device SNR vs F.conv_transpose1d: {snr_db:.2f} dB")
    assert snr_db >= 35 or snr_db == float("inf"), f"SNR {snr_db:.2f} dB must be at least 35 dB"

    record_test("conv_transpose1d_zero_insert",
                f"Cin={C_in},Cout={C_out},Tin={T_in},stride={stride},k={kernel_size}",
                snr_db=snr_db)


def instance_norm1d_via_layernorm_test(C: int = 64, T: int = 128):
    """Proves InstanceNorm1d (Kokoro AdaIN1d's normalization step) is exactly
    ``layer_norm_core_dram`` applied to a ``[C, T]`` tensor with C as the row (M) axis and T as the
    reduced free (N) axis -- the natural Conv1d output layout, no permute needed -- followed by a
    per-channel affine applied host-side via a tiled eltwise multiply/add (since the native
    GAMMA/BETA_DRAM_ADDR vectors broadcast per-column, not per-row, so they cannot express a
    per-channel AdaIN affine directly; see header note).
    """
    torch.manual_seed(0)
    x = torch.randn(C, T, dtype=torch.bfloat16)
    gamma = (torch.randn(C, dtype=torch.bfloat16) * 0.2)
    beta = (torch.randn(C, dtype=torch.bfloat16) * 0.2)

    # Reference: InstanceNorm (no affine) then a separate per-channel AdaIN-style affine.
    normed_ref = torch.nn.functional.instance_norm(x.unsqueeze(0).float(), eps=1e-5).squeeze(0)
    ref = (1.0 + gamma.float()).unsqueeze(1) * normed_ref + beta.float().unsqueeze(1)
    ref = ref.to(torch.bfloat16)

    ue = UnifiedEngine()
    A = ue.allocate_tensor_dram(C * T * 2)
    NORMED = ue.allocate_tensor_dram(C * T * 2)
    GTILE = ue.allocate_tensor_dram(C * T * 2)
    BTILE = ue.allocate_tensor_dram(C * T * 2)
    OUT = ue.allocate_tensor_dram(C * T * 2)

    ue.start_capture()
    # Native gamma/beta omitted -> layer_norm_core_dram defaults them to a neutral ones(N) vector /
    # None (see user_dma_core.py lines 4028-4034), i.e. plain InstanceNorm with no affine baked in.
    total_flops = ue.layer_norm_core_dram(M=C, N=T, A_DRAM_ADDR=A, OUTPUT_DRAM_ADDR=NORMED)
    # Apply the AdaIN affine separately: y = (1+gamma)*normed + beta, gamma/beta tiled to [C, T]
    # host-side since only a single global scalar broadcast is exposed (see header note).
    total_flops += ue.eltwise_core_dram(C, T, NORMED, GTILE, OUT, UE_MODE.ELTWISE_MUL)
    total_flops += ue.eltwise_core_dram(C, T, OUT, BTILE, OUT, UE_MODE.ELTWISE_ADD)
    ue.stop_capture()
    ue.generate_instruction_halt()
    prog = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(prog)
    inst_bytes = ue.get_capture_instruction_size_bytes()
    ue.allocate_program_dram(inst_bytes)

    gtile = (1.0 + gamma.float()).unsqueeze(1).expand(C, T).contiguous().to(torch.bfloat16)
    btile = beta.float().unsqueeze(1).expand(C, T).contiguous().to(torch.bfloat16)
    ue.dma_to_accelerator_memory(A, x.reshape(-1).contiguous())
    ue.dma_to_accelerator_memory(GTILE, gtile.reshape(-1).contiguous())
    ue.dma_to_accelerator_memory(BTILE, btile.reshape(-1).contiguous())

    ue.start_execute_from_dram(prog)
    ue.wait_queue(10.0)
    ue.report_timing_and_instruction_count()
    gflops, _ = ue.report_flop_rate_gflops(total_flops)

    out = ue.dma_from_accelerator_memory(OUT, (C, T))
    snr_db = calculate_snr(ref, out)
    print(f"[instance_norm1d_via_layernorm] C={C} T={T} SNR={snr_db:.2f} dB GFLOPS={gflops:.2f}")
    assert snr_db >= 40.0 or snr_db == float("inf"), \
        f"instance_norm1d_via_layernorm C={C} T={T} SNR {snr_db:.2f} dB < 40 dB"
    record_test("instance_norm1d_via_layernorm", f"C={C},T={T}", snr_db=snr_db, gflops=gflops,
                inst_bytes=inst_bytes)
    ue.clear_capture_buffer(); ue.reset_tensor_dram_addr(); ue.reset_program_dram_addr()


def _adain_elems_written(out: torch.Tensor) -> int:
    """Number of leading elements the core actually wrote (sentinel-tail detector).

    Returns ``last_non_sentinel_index + 1`` over the flattened buffer, so a full write
    returns ``out.numel()`` and a short write returns the truncated element count.
    """
    flat = out.reshape(-1).float()
    non_sent = (flat != _ADAIN_SENTINEL).nonzero()
    return 0 if non_sent.numel() == 0 else int(non_sent[-1].item()) + 1


def adain1d_norm_formula_test(shapes=None, snr_threshold_db: float = 40.0,
                              assert_on_fail: bool = False):
    """Pin down the exact InstanceNorm formula Kokoro's ``_adain1d`` must use when the
    TIME axis is padded to a multiple of 64.

    ``_adain1d`` (models/kokoro/kokoro_fpga.py:1680) does InstanceNorm1d over TIME: it
    transposes ``[T, C] -> [C, T]`` and layer-norms with ``M=C`` rows and ``N=T`` columns.
    ``T`` is padded up to a multiple of 64 but only ``real_T`` frames carry signal, so the
    pad columns are *samples* in the per-channel mean/variance and their value matters.

    For each ``(C, real_T)`` this runs the same reference against three formulations and
    reports SNR plus a short-write diagnostic for each:

      * ``mean_pad``  -- (A), the CURRENT/WORKING path: fill the pad columns with the
        per-channel MEAN of the real columns, normalize over all ``T``, then rescale by
        ``pad_k = sqrt(real_T / T)``. Padding at the mean shifts the mean not at all and
        adds nothing to the sum of squares, so the variance is scaled by exactly
        ``real_T/T`` and ``pad_k`` undoes it. In the model ``pad_k`` is folded into the
        AdaIN affine as ``gamma' = pad_k*(1 + gamma) - 1``; here gamma=0, so folding it in
        is the same host-side scalar multiply, and the device math under test is identical.
      * ``short_n``   -- (B), the ATTEMPTED/BROKEN path: ``layer_norm_core_dram_dynamic``'s
        short-N mode. N is baked at the padded cap, ``INV_N_DRAM_ADDR`` carries ``1/real_T``
        in the first ``real_T`` lanes and 0 after, ``MASK_DRAM_ADDR`` carries 1/0 over the
        same split, and ``gpr_N_reg`` / ``gpr_sqrt_n_reg`` are primed with ``real_T`` and
        ``float_to_bf19(sqrt(real_T))``.
      * ``short_n_padN`` -- (B'), the same masked inv_n/mask, but ``gpr_N_reg`` primed with
        the PADDED N (so the DMA/store length still spans the whole row) while the RSQRT
        scalar keeps ``sqrt(real_T)``. This separates "the masking math is wrong" from
        "``gpr_N_reg`` < baked N truncates the writes".
      * ``plain``     -- (C), the baseline: conventional call, zero-padded input, uniform
        ``1/N`` over the padded ``T``, no mask, no correction. Its SNR shows how badly the
        padding biases an uncorrected InstanceNorm (the thing (A) and (B) both exist to fix).

    Short-write diagnostic: every output buffer is prefilled with ``_ADAIN_SENTINEL`` before
    the run, and the read-back is scanned for a contiguous unwritten tail. ``rows_written``
    (= elements written / N) vs ``rows_expected`` (= C) makes it obvious which ``(M, N)``
    combinations short-write -- the observed failure was M=512, N=1984 writing only 34 rows.

    ``shapes`` entries are ``(C, real_T)`` -- baked ``N = round_up(real_T, 64)``, i.e. tight
    padding -- or ``(C, real_T, baked_N)``, which decouples the two. The decoupled form is the
    one that matters: kokoro cap-templates the frame axis at ``F_CAP = 1984`` while a short
    prompt carries as few as 78 real frames, so the reduction runs over ~4% signal and ~96% pad.
    The default grid sweeps ``real_T/N`` from 4% to 100% with ``N`` pinned at 1984, and the
    summary prints the highest ratio at which each method still fails -- that threshold is what
    decides whether kokoro can bake N at the cap.

    Runs only on hardware. ``assert_on_fail=False`` by default so the whole grid is swept and
    reported even when a formulation is badly wrong -- that comparison is the point of the test.
    """
    if shapes is None:
        # Kokoro-typical (C, real_T): decoder channel counts x frame counts, then the
        # large-N cases that reproduce the short write (padded_T up to F_CAP = 1984).
        shapes = [(C, rt) for C in (512, 256, 128) for rt in (78, 128, 173, 348)]
        shapes += [(512, 1024), (256, 1984), (512, 1950), (512, 1984)]
        # Low real_T/N ratio at a LARGE baked N -- the regime kokoro actually runs in and the
        # one the tight-padding grid above never reaches (its worst ratio is 61%). Kokoro caps
        # the frame axis at F_CAP=1984 while a short prompt carries as few as 78 real frames
        # (4%), so the reduction is almost entirely pad lanes. The M=512, N=1984 short write
        # (34 rows) was observed in exactly this regime. Baked N is pinned at 1984 and real_T
        # swept, so the ratio is the only variable.
        shapes += [(512, rt, 1984) for rt in (78, 128, 173, 348, 992)]
        shapes += [(256, 78, 1984), (128, 78, 1984)]

    eps = 1e-5
    rows = []

    def _reference(x_real: torch.Tensor):
        """Exact float64 InstanceNorm over the real columns only (biased variance, as
        torch.nn.functional.instance_norm / nn.InstanceNorm1d use)."""
        xd = x_real.double()
        mean = xd.mean(dim=1, keepdim=True)
        var = xd.var(dim=1, unbiased=False, keepdim=True)
        return (xd - mean) / torch.sqrt(var + eps)

    def _run_legacy(x_pad, C, N):
        """Conventional baked-dims layer_norm_core_dram over [C, N]; no gamma/beta."""
        ue = UnifiedEngine()
        A = ue.allocate_tensor_dram(C * N * 2)
        O = ue.allocate_tensor_dram(C * N * 2)

        ue.start_capture()
        ue.layer_norm_core_dram(M=C, N=N, A_DRAM_ADDR=A, OUTPUT_DRAM_ADDR=O)
        ue.stop_capture()
        ue.generate_instruction_halt()
        prog = ue.get_program_dram_addr()
        ue.write_captured_instructions_to_dram(prog)
        inst_bytes = ue.get_capture_instruction_size_bytes()
        ue.allocate_program_dram(inst_bytes)
        ue.clear_capture_buffer()

        ue.dma_to_accelerator_memory(A, x_pad.reshape(-1).contiguous())
        ue.dma_to_accelerator_memory(O, torch.full((C * N,), _ADAIN_SENTINEL, dtype=torch.bfloat16))

        ue.start_execute_from_dram(prog)
        ue.wait_queue(30.0)
        out = ue.dma_from_accelerator_memory(O, (C, N))
        ue.clear_capture_buffer(); ue.reset_tensor_dram_addr(); ue.reset_program_dram_addr()
        return out, _adain_elems_written(out), inst_bytes

    def _run_short_n(x_pad, C, N, real_T, n_reg_value):
        """Dynamic core in short-N mode: baked N = padded N, masked inv_n + mask band,
        runtime sqrt(real_T). ``n_reg_value`` is what gpr_N_reg gets primed with."""
        ue = UnifiedEngine()
        A = ue.allocate_tensor_dram(C * N * 2)
        O = ue.allocate_tensor_dram(C * N * 2)
        G = ue.allocate_tensor_dram(N * 2)      # plain gamma (ones): the core wants it present
        INV = ue.allocate_tensor_dram(N * 2)    # 1/real_T in real lanes, 0 in pad lanes
        MASK = ue.allocate_tensor_dram(N * 2)   # 1 in real lanes, 0 in pad lanes

        regs = []
        def areg():
            r = ue.alloc_isa_reg(); regs.append(r); return r
        m_reg = areg(); n_reg = areg(); sqrt_reg = areg()
        a_reg = areg(); out_reg = areg(); g_reg = areg()
        invn_reg = areg(); mask_reg = areg()

        # 1. Compile once at template M=64, N=padded N; runtime M / N / sqrt(N) via GPRs.
        ue.start_capture()
        ue.layer_norm_core_dram_dynamic(
            M=64, N=N, A_DRAM_ADDR=A, OUTPUT_DRAM_ADDR=O,
            GAMMA_DRAM_ADDR=G, BETA_DRAM_ADDR=None, INV_N_DRAM_ADDR=INV, MASK_DRAM_ADDR=MASK,
            gpr_M_reg=m_reg, gpr_N_reg=n_reg, gpr_sqrt_n_reg=sqrt_reg,
            gpr_a_addr=a_reg, gpr_out_addr=out_reg, gpr_gamma_addr=g_reg, gpr_beta_addr=None,
            gpr_invn_addr=invn_reg, gpr_mask_addr=mask_reg,
        )
        ue.stop_capture()
        ue.generate_instruction_halt()
        main_prog = ue.get_program_dram_addr()
        ue.write_captured_instructions_to_dram(main_prog)
        main_inst_bytes = ue.get_capture_instruction_size_bytes()
        ue.allocate_program_dram(main_inst_bytes)

        # 2. Preamble: prime real M, the N register under test, sqrt(real_T) + DRAM bases, then jump.
        preamble = ue.get_program_dram_addr()
        ue.allocate_program_dram((len(regs) + 2) * INSTRUCTION_SIZE_BYTES)
        main_word_addr = ue_35bit_addr_shifter(main_prog)
        ue.clear_capture_buffer()
        ue.start_capture()
        ue.generate_instruction_add_set(m_reg, C)
        ue.generate_instruction_add_set(n_reg, n_reg_value)
        ue.generate_instruction_add_set(sqrt_reg, ue.float_to_bf19(float(real_T ** 0.5)))
        ue.generate_instruction_add_set(a_reg, A >> 3)
        ue.generate_instruction_add_set(out_reg, O >> 3)
        ue.generate_instruction_add_set(g_reg, G >> 3)
        ue.generate_instruction_add_set(invn_reg, INV >> 3)
        ue.generate_instruction_add_set(mask_reg, MASK >> 3)
        ue.generate_instruction_jump_abs(main_word_addr)
        ue.stop_capture()
        ue.write_captured_instructions_to_dram(preamble)

        inv_n = torch.zeros(N, dtype=torch.bfloat16)
        inv_n[:real_T] = 1.0 / real_T
        mask = torch.zeros(N, dtype=torch.bfloat16)
        mask[:real_T] = 1.0
        ue.dma_to_accelerator_memory(A, x_pad.reshape(-1).contiguous())
        ue.dma_to_accelerator_memory(G, torch.ones(N, dtype=torch.bfloat16))
        ue.dma_to_accelerator_memory(INV, inv_n)
        ue.dma_to_accelerator_memory(MASK, mask)
        ue.dma_to_accelerator_memory(O, torch.full((C * N,), _ADAIN_SENTINEL, dtype=torch.bfloat16))

        ue.start_execute_from_dram(preamble)
        ue.wait_queue(30.0)
        out = ue.dma_from_accelerator_memory(O, (C, N))
        for _ in regs:
            ue.release_isa_reg()
        ue.clear_capture_buffer(); ue.reset_tensor_dram_addr(); ue.reset_program_dram_addr()
        return out, _adain_elems_written(out), main_inst_bytes

    def _report(method, C, real_T, N, ref, got, elems, inst_bytes):
        snr_db = calculate_snr(ref.to(torch.bfloat16), got)
        rows_written = elems / N
        dims = f"C={C},realT={real_T},padT={N}"
        ratio = real_T / N
        rows.append((dims, method, snr_db, rows_written, C, elems, C * N, ratio))
        short = "SHORT-WRITE" if elems < C * N else "full"
        print(f"[adain1d_norm+{method}] {dims} SNR={snr_db:.2f} dB "
              f"rows_written={rows_written:.2f}/{C} ({short}, {elems}/{C * N} elems)")
        record_test(f"adain1d_norm+{method}", f"{dims},rows={rows_written:.2f}/{C}",
                    snr_db=snr_db, inst_bytes=inst_bytes)
        if assert_on_fail:
            assert snr_db >= snr_threshold_db or snr_db == float("inf"), \
                f"adain1d_norm+{method} {dims} SNR {snr_db:.2f} dB < {snr_threshold_db:g} dB"
            assert elems == C * N, \
                f"adain1d_norm+{method} {dims} short write: {elems}/{C * N} elements"
        return snr_db

    for shape in shapes:
        # A shape is (C, real_T) -- baked N = round_up(real_T, 64), tight padding -- or
        # (C, real_T, baked_N), which DECOUPLES the two so N can be a fixed cap far above the
        # real frame count. Kokoro cap-templates the frame axis, so baked_N is F_CAP while
        # real_T follows the prompt; the ratio real_T/N is the thing under test.
        if len(shape) == 3:
            C, real_T, N = shape
        else:
            C, real_T = shape
            N = _round_up_vec(real_T)
        assert N % UE_VECTOR_SIZE == 0, f"baked N={N} must be a multiple of {UE_VECTOR_SIZE}"
        assert N >= real_T, f"baked N={N} is smaller than real_T={real_T}"
        pad_k = (real_T / N) ** 0.5
        x_real = torch.randn(C, real_T, dtype=torch.bfloat16)
        ref = _reference(x_real)                       # float64, [C, real_T]

        # (A) pad the time columns at the per-channel mean, then undo the variance scaling.
        x_meanpad = torch.zeros(C, N, dtype=torch.bfloat16)
        x_meanpad[:, :real_T] = x_real
        chan_mean = x_real.float().mean(dim=1, keepdim=True).to(torch.bfloat16)
        x_meanpad[:, real_T:] = chan_mean.expand(C, N - real_T)

        # (B)/(C) zero-padded input.
        x_zeropad = torch.zeros(C, N, dtype=torch.bfloat16)
        x_zeropad[:, :real_T] = x_real

        out, elems, ib = _run_legacy(x_meanpad, C, N)
        got = (out[:, :real_T].float() * pad_k).to(torch.bfloat16)
        _report("mean_pad", C, real_T, N, ref, got, elems, ib)

        out, elems, ib = _run_short_n(x_zeropad, C, N, real_T, n_reg_value=real_T)
        _report("short_n", C, real_T, N, ref, out[:, :real_T], elems, ib)

        out, elems, ib = _run_short_n(x_zeropad, C, N, real_T, n_reg_value=N)
        _report("short_n_padN", C, real_T, N, ref, out[:, :real_T], elems, ib)

        out, elems, ib = _run_legacy(x_zeropad, C, N)
        _report("plain", C, real_T, N, ref, out[:, :real_T], elems, ib)

    # Per-shape summary table: the whole point is the side-by-side comparison.
    print("\n=== adain1d_norm_formula_test summary ===")
    print(f"{'shape':<28} {'realT/N':>8} {'method':<14} {'SNR dB':>9} {'rows_written':>13} "
          f"{'rows_exp':>9} {'write':>12}")
    for dims, method, snr_db, rows_written, rows_exp, elems, elems_exp, ratio in rows:
        write = "full" if elems == elems_exp else f"SHORT {elems}/{elems_exp}"
        print(f"{dims:<28} {ratio:>7.1%} {method:<14} {snr_db:>9.2f} {rows_written:>13.2f} "
              f"{rows_exp:>9d} {write:>12}")
    # The headline number: the highest real_T/N ratio at which each method still misbehaves.
    # If short_n is clean above some ratio and broken below it, that threshold decides whether
    # kokoro can bake N at F_CAP or must keep sizing N to the prompt.
    print("\n--- worst (highest) real_T/N ratio at which each method fails ---")
    for method in ("mean_pad", "short_n", "short_n_padN", "plain"):
        bad = [(ratio, dims, snr_db, elems, elems_exp)
               for dims, m, snr_db, _rw, _re, elems, elems_exp, ratio in rows
               if m == method and (elems != elems_exp or not (snr_db >= snr_threshold_db)
                                   or snr_db != snr_db)]
        if not bad:
            print(f"  {method:<14} clean at every ratio tested")
            continue
        ratio, dims, snr_db, elems, elems_exp = max(bad)
        why = f"SHORT {elems}/{elems_exp}" if elems != elems_exp else f"SNR {snr_db:.2f} dB"
        print(f"  {method:<14} fails up to ratio {ratio:.1%} ({dims}, {why}); "
              f"{sum(1 for b in bad)} of "
              f"{sum(1 for r in rows if r[1] == method)} shapes bad")
    print("=========================================\n")


def adain1d_frozen_device_test(C_list=(512, 256), real_T_list=(348, 78, 992, 173, 1920),
                               N_cap: int = 1984, style_dim: int = 128,
                               snr_threshold_db: float = 35.0, assert_on_fail: bool = False):
    """The WHOLE kokoro AdaIN1d op -- style affine, InstanceNorm over time, per-channel
    (1+gamma)*x_hat + beta -- compiled ONCE per channel count and replayed for several prompt
    lengths with NOTHING but the register preamble and the per-run data tables changing. No host
    readback, no host tiling, no capture split anywhere inside the op.

    Why this exists: ``_adain1d`` is the one kokoro op whose sequence length lands on the
    layer-norm N axis (it normalizes each channel across TIME), and the layer-norm core bakes N.
    Every other section is already byte-identical across prompts. This test is the proof that
    the op can be made sequence-length invariant with the cores as they are.

    Layout under test (x is [T, C], kokoro's convention; T_cap = N_cap frames):
      1. gb = style @ fc_w^T + fc_b                 -> [1, 2C]   (matmul, fixed dims)
      2. g1 = gb[:C] + 1                            -> [1, C]    (eltwise ADD_BROADCAST)
      3. zero x's pad rows [T64, N_cap)             (HW loop, trip count = pad-rows GPR)
      4. x_ct = transpose(x)  at M = N_cap          -> [C, N_cap] (row stride == baked LN N)
      5. InstanceNorm: layer_norm short-N mode, N baked at N_cap, gpr_N_reg = real_T,
         gpr_sqrt_n_reg = sqrt(real_T), inv_n/mask tables = 1/real_T,1 in real lanes, 0 after
      6. normed = transpose(normed_ct) at M = C     -> [N_cap, C]
      7. g1 / beta tiled down T64 rows              (HW zero-source-stride loops, trip = T64 GPR)
      8. out = normed * g1_tiled + beta_tiled       (eltwise, gpr_M_reg = T64)

    Steps 4 and 6 execute at the cap: a [C, N_cap] view with row stride N_cap can only be built by
    a transpose whose M IS N_cap, and the same holds going back. Everything else runs at the real
    length. The two transposes are 2*C*N_cap elements -- noise next to the section's matmuls.

    Per-run host inputs (the same class of thing as the GPR preamble and the LLMs' attention
    masks): x, style, inv_n, mask, the three GPRs. Per-run instruction changes: the preamble only.

    Checks per (C, real_T): SNR of out[:real_T] against a float64 AdaIN reference, and the
    sentinel-tail short-write detector on the [N_cap, C] output. Program bytes are read back from
    DRAM after every run and compared to the bytes written at compile time.
    """
    torch.manual_seed(0)
    eps = 1e-5
    T64 = lambda t: ((t + 63) // 64) * 64
    for rt in real_T_list:
        assert T64(rt) < N_cap, f"real_T={rt}: the pad-row loop needs >= 1 pad row below N_cap={N_cap}"
    results = []

    for C in C_list:
        ue = UnifiedEngine()
        # ---- DRAM plan (fixed for the life of the program) ----
        X = ue.allocate_tensor_dram(N_cap * C * 2)
        STYLE = ue.allocate_tensor_dram(64 * style_dim * 2)
        FCW = ue.allocate_tensor_dram(2 * C * style_dim * 2)
        FCB = ue.allocate_tensor_dram(2 * C * 2)
        GB = ue.allocate_tensor_dram(64 * 2 * C * 2)
        G1 = ue.allocate_tensor_dram(64 * C * 2)
        ZROW = ue.allocate_tensor_dram(N_cap * 2)          # zeros, >= max(C, N_cap) elements
        IDENT = ue.allocate_tensor_dram(64 * 64 * 2)
        XCT = ue.allocate_tensor_dram(C * N_cap * 2)
        NCT = ue.allocate_tensor_dram(C * N_cap * 2)
        NORMED = ue.allocate_tensor_dram(N_cap * C * 2)
        GT = ue.allocate_tensor_dram(N_cap * C * 2)
        BT = ue.allocate_tensor_dram(N_cap * C * 2)
        NG = ue.allocate_tensor_dram(N_cap * C * 2)
        OUT = ue.allocate_tensor_dram(N_cap * C * 2)
        ONES = ue.allocate_tensor_dram(N_cap * 2)
        INV = ue.allocate_tensor_dram(N_cap * 2)
        MASK = ue.allocate_tensor_dram(N_cap * 2)
        assert ZROW % 8 == 0 and X % 8 == 0 and GT % 8 == 0 and BT % 8 == 0

        # ---- registers: runtime ones (preamble-primed) first so they stay in the 1..15 M range ----
        t64_reg = ue.alloc_isa_reg()     # T rounded up to 64 (row count for the [T,C] ops)
        pad_reg = ue.alloc_isa_reg()     # N_cap - T64 (pad rows to zero)
        n_reg = ue.alloc_isa_reg()       # real_T (LN reduction length)
        sqrt_reg = ue.alloc_isa_reg()    # bf19 sqrt(real_T)
        m_cap_reg = ue.alloc_isa_reg()   # constant N_cap, seeded IN the body
        m_c_reg = ue.alloc_isa_reg()     # constant C, seeded IN the body
        i_reg = ue.alloc_isa_reg()
        t_reg = ue.alloc_isa_reg()

        def _zero_stride_tile(src_row, dst, row_bytes, elems, sram_off):
            """dst[i, :] = src_row for i in [0, T64): one HW loop, zero source stride."""
            ue.accelerator_memory_to_sram(accelerator_dram_address=src_row, sram_address=sram_off,
                                          element_size=elems)
            ue.generate_instruction_add_set(i_reg, 0)
            ue.loop_start(loop_cnt=N_cap, gpr_loop_cnt=t64_reg)
            ue.generate_instruction_reg_mul_imm(t_reg, i_reg, ue_35bit_addr_shifter(row_bytes))
            ue.generate_instruction_add_imm(t_reg, ue_35bit_addr_shifter(dst), t_reg)
            ue.sram_to_accelerator_memory(sram_address=sram_off, accelerator_dram_address=0,
                                          element_size=elems, general_reg_src=t_reg)
            ue.generate_instruction_add_inc(i_reg)
            ue.loop_end()

        # ---- compile the body ONCE ----
        ue.start_capture()
        ue.generate_instruction_add_set(m_cap_reg, N_cap)
        ue.generate_instruction_add_set(m_c_reg, C)
        # 1. style affine  gb = style @ fc_w^T + fc_b  (M=1: legacy tiling, fixed dims)
        ue.matmat_mul_core(M=1, K=style_dim, N=2 * C, A_DRAM_ADDR=STYLE, B_DRAM_ADDR=FCW,
                           OUTPUT_DRAM_ADDR=GB, C_DRAM_ADDR=FCB, bias_mode="broadcast_N")
        # 2. g1 = gamma + 1
        ue.eltwise_core_dram(1, C, GB, None, G1, UE_MODE.ADD_BROADCAST, scalar=1.0)
        # 3. zero x's pad rows: rows [T64, N_cap)
        ue.accelerator_memory_to_sram(accelerator_dram_address=ZROW, sram_address=0, element_size=C)
        ue.generate_instruction_add_imm(t64_reg, 0, i_reg)
        ue.loop_start(loop_cnt=N_cap, gpr_loop_cnt=pad_reg)
        ue.generate_instruction_reg_mul_imm(t_reg, i_reg, ue_35bit_addr_shifter(C * 2))
        ue.generate_instruction_add_imm(t_reg, ue_35bit_addr_shifter(X), t_reg)
        ue.sram_to_accelerator_memory(sram_address=0, accelerator_dram_address=0, element_size=C,
                                      general_reg_src=t_reg)
        ue.generate_instruction_add_inc(i_reg)
        ue.loop_end()
        # 4. x_ct [C, N_cap] = transpose(x [N_cap, C])  -- M = N_cap so the row stride is N_cap
        ue.bf16_transpose_core(M=N_cap, N=C, INPUT_DRAM_ADDR=X, OUTPUT_DRAM_ADDR=XCT,
                               IDENTITY_DRAM_ADDR=IDENT, gpr_M_reg=m_cap_reg)
        # 5. InstanceNorm over time, short-N mode
        ue.layer_norm_core_dram_dynamic(
            M=C, N=N_cap, A_DRAM_ADDR=XCT, OUTPUT_DRAM_ADDR=NCT,
            gpr_M_reg=m_c_reg, gpr_N_reg=n_reg, gpr_sqrt_n_reg=sqrt_reg,
            GAMMA_DRAM_ADDR=ONES, BETA_DRAM_ADDR=None, ZEROS_DRAM_ADDR=ZROW,
            INV_N_DRAM_ADDR=INV, MASK_DRAM_ADDR=MASK)
        # 6. normed [N_cap, C] = transpose(normed_ct [C, N_cap])
        ue.bf16_transpose_core(M=C, N=N_cap, INPUT_DRAM_ADDR=NCT, OUTPUT_DRAM_ADDR=NORMED,
                               IDENTITY_DRAM_ADDR=IDENT, gpr_M_reg=m_c_reg)
        # 7. tile g1 / beta down T64 rows
        _zero_stride_tile(G1, GT, C * 2, C, 0)
        _zero_stride_tile(GB + C * 2, BT, C * 2, C, 0)
        # 8. out = normed * g1 + beta   (row count from t64_reg)
        ue.eltwise_core_dram(N_cap, C, NORMED, GT, NG, UE_MODE.ELTWISE_MUL, gpr_M_reg=t64_reg)
        ue.eltwise_core_dram(N_cap, C, NG, BT, OUT, UE_MODE.ELTWISE_ADD, gpr_M_reg=t64_reg)
        ue.stop_capture()
        ue.generate_instruction_halt()
        body = ue.get_program_dram_addr()
        ue.write_captured_instructions_to_dram(body)
        body_bytes = ue.get_capture_instruction_size_bytes()
        ue.allocate_program_dram(body_bytes)
        n_inst = body_bytes // INSTRUCTION_SIZE_BYTES
        preamble = ue.get_program_dram_addr()
        ue.allocate_program_dram(8 * INSTRUCTION_SIZE_BYTES)
        body_word_addr = ue_35bit_addr_shifter(body)
        print(f"[adain1d_frozen] C={C}: body compiled once, {n_inst} instructions ({body_bytes} B)")

        # ---- constants (once) ----
        fc_w = (torch.randn(2 * C, style_dim) * 0.05).to(torch.bfloat16)
        fc_b = (torch.randn(2 * C) * 0.1).to(torch.bfloat16)
        style = torch.randn(style_dim).to(torch.bfloat16)
        style_pad = torch.zeros(64, style_dim, dtype=torch.bfloat16); style_pad[0] = style
        ue.dma_to_accelerator_memory(FCW, fc_w.reshape(-1).contiguous())
        ue.dma_to_accelerator_memory(FCB, fc_b.contiguous())
        ue.dma_to_accelerator_memory(STYLE, style_pad.reshape(-1).contiguous())
        ue.dma_to_accelerator_memory(ZROW, torch.zeros(N_cap, dtype=torch.bfloat16))
        ue.dma_to_accelerator_memory(IDENT, torch.eye(64, dtype=torch.bfloat16).reshape(-1).contiguous())
        ue.dma_to_accelerator_memory(ONES, torch.ones(N_cap, dtype=torch.bfloat16))
        gb_ref = (style.float() @ fc_w.float().T + fc_b.float())
        gamma_ref, beta_ref = gb_ref[:C].double(), gb_ref[C:].double()

        for real_T in real_T_list:
            t64 = T64(real_T)
            x_real = torch.randn(real_T, C, dtype=torch.bfloat16)
            xd = x_real.double()
            mean = xd.mean(dim=0, keepdim=True)
            var = xd.var(dim=0, unbiased=False, keepdim=True)
            ref = ((xd - mean) / torch.sqrt(var + eps)) * (1.0 + gamma_ref) + beta_ref   # [real_T, C]

            # per-run data: x (real rows; rows [real_T, t64) zero, rows >= t64 GARBAGE on purpose so
            # the device zero-fill is what makes them safe), inv_n / mask tables.
            x_dev = torch.full((N_cap, C), float("nan"), dtype=torch.bfloat16)
            x_dev[:real_T] = x_real
            x_dev[real_T:t64] = 0.0
            inv_n = torch.zeros(N_cap, dtype=torch.bfloat16); inv_n[:real_T] = 1.0 / real_T
            mask = torch.zeros(N_cap, dtype=torch.bfloat16); mask[:real_T] = 1.0
            ue.dma_to_accelerator_memory(X, x_dev.reshape(-1).contiguous())
            ue.dma_to_accelerator_memory(INV, inv_n)
            ue.dma_to_accelerator_memory(MASK, mask)
            for buf, n in ((OUT, N_cap * C), (NCT, C * N_cap), (NORMED, N_cap * C)):
                ue.dma_to_accelerator_memory(buf, torch.full((n,), _ADAIN_SENTINEL, dtype=torch.bfloat16))

            # the ONLY per-run instructions: prime 4 GPRs and jump into the frozen body
            ue.clear_capture_buffer()
            ue.start_capture()
            ue.generate_instruction_add_set(t64_reg, t64)
            ue.generate_instruction_add_set(pad_reg, N_cap - t64)
            ue.generate_instruction_add_set(n_reg, real_T)
            ue.generate_instruction_add_set(sqrt_reg, ue.float_to_bf19(float(real_T ** 0.5)))
            ue.generate_instruction_jump_abs(body_word_addr)
            ue.stop_capture()
            ue.write_captured_instructions_to_dram(preamble)

            ue.start_execute_from_dram(preamble)
            ue.wait_queue(60.0)
            out = ue.dma_from_accelerator_memory(OUT, (N_cap, C))
            nct = ue.dma_from_accelerator_memory(NCT, (C, N_cap))

            got = out[:real_T]
            snr_db = calculate_snr(ref.to(torch.bfloat16), got)
            elems_out = _adain_elems_written(out)
            elems_nct = _adain_elems_written(nct)
            rows_out = elems_out / C
            nan_out = int(torch.isnan(out[:t64].float()).sum().item())
            dims = f"C={C},realT={real_T},T64={t64},Ncap={N_cap}"
            print(f"[adain1d_frozen] {dims} SNR={snr_db:.2f} dB rows_written={rows_out:.1f}/{t64} "
                  f"nct_elems={elems_nct}/{C * N_cap} nan_in_out[:T64]={nan_out}")
            record_test("adain1d_frozen", dims, snr_db=snr_db, inst_bytes=body_bytes)
            results.append((dims, snr_db, rows_out, t64, elems_nct, C * N_cap, nan_out))
            if assert_on_fail:
                assert snr_db >= snr_threshold_db, f"{dims} SNR {snr_db:.2f} < {snr_threshold_db}"
                assert elems_out >= t64 * C, f"{dims} short write {elems_out}/{t64 * C}"

        for _ in range(8):
            ue.release_isa_reg()
        ue.clear_capture_buffer(); ue.reset_tensor_dram_addr(); ue.reset_program_dram_addr()

    print("\n=== adain1d_frozen_device_test summary (one body per C, replayed per real_T) ===")
    print(f"{'shape':<40} {'SNR dB':>8} {'rows_out':>12} {'nct_write':>16} {'nan':>5}")
    for dims, snr_db, rows_out, t64, en, en_exp, nan_out in results:
        w = "full" if en == en_exp else f"SHORT {en}/{en_exp}"
        print(f"{dims:<40} {snr_db:>8.2f} {rows_out:>7.1f}/{t64:<4d} {w:>16} {nan_out:>5d}")
    print("==============================================================================\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kokoro op-composition hardware proofs")
    parser.add_argument("--dev", type=str, default="xdma0")
    parser.add_argument("--summary-path", type=str, default=None)
    args = parser.parse_args()
    set_dma_device(args.dev)
    torch.manual_seed(0)
    embedding_gather_test(); cumsum_via_triangular_matmul_test(); lstm_cell_hw_test()
    conv1d_shifted_matmul_test(); depthwise_conv1d_eltwise_test(); interpolate_fixed_matmul_test()
    conv_transpose1d_zero_insert_test(); instance_norm1d_via_layernorm_test()
    adain1d_norm_formula_test(); adain1d_frozen_device_test()
    if args.summary_path:
        write_test_summary(args.summary_path)
