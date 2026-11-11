"""Unit test on one card for the 16K attention layout: persistent V^T [hd, T_AL] filled one 64-row
block at a time by a GPR-addressed transpose, Q.K^T over the filled length L (dynamic), P.V over the
whole persistent V^T (static K = T_AL, zero score tail). Torch reference only checks the numbers."""
import importlib.util
import os
import sys
import time

import torch

BASE = os.path.expanduser("~/apex_tp/models/q35_9b")
spec = importlib.util.spec_from_file_location("q35_16k_port", os.path.join(BASE, "qwen3.5_2b_test.py"))
T = importlib.util.module_from_spec(spec)
sys.modules["q35_16k_port"] = T
spec.loader.exec_module(T)
T.set_dma_device(sys.argv[1] if len(sys.argv) > 1 else "xdma0")
T.DMA_DEVICE_H2C = T.user_dma_core.DMA_DEVICE_H2C
T.DMA_DEVICE_C2H = T.user_dma_core.DMA_DEVICE_C2H
BF16 = 2
HD, T_AL, VEC = 256, 16384, 64
CH = 4096                          # V^T is kept in [HD, CH] chunks: matmul K must fit URAM
R_STRIDE, R_IN, R_OUT, R_LEN, R_BLKV, R_BLKVT = 57, 58, 59, 60, 61, 62
sh = T.ue_35bit_addr_shifter

with T._quiet():
    T.user_dma_core.configure_clock_from_hardware()
    ue = T.Qwen3_5_2b_UnifiedEngine(device="cpu", config_path=os.path.join(BASE, "qwen3.5_2b_config.json"))
    ue.software_reset()
    ue._preallocate_identity_matrix()
G = int(os.environ.get("G", "4"))  # query heads sharing the KV head (one matmul row each)
R_ROWSTR = 56
torch.manual_seed(1)
q = torch.randn(G, HD) * 0.5
K = torch.randn(T_AL, HD) * 0.5
V = torch.randn(T_AL, HD)
A = lambda n: ue.allocate_tensor_dram(n * BF16)
Q_D, K_D, V_D, VT_D, B_D = A(G * HD), A(T_AL * HD), A(T_AL * HD), A(HD * T_AL), A(T_AL)
SQ, SC_ROW, SC_CH, OUT = A(G * HD), A(G * T_AL), A(G * T_AL), A(G * HD)
for addr, x in ((Q_D, q), (K_D, K), (V_D, V)):
    ue.dma_write(T.DMA_DEVICE_H2C, addr, T._bf(x.flatten()), x.numel() * BF16)
for addr, n in ((VT_D, HD * T_AL), (SC_ROW, G * T_AL), (SC_CH, G * T_AL)):
    ue.dma_write(T.DMA_DEVICE_H2C, addr, T._bf(torch.zeros(n)), n * BF16)
ROWSTR_UNIT = int(os.environ.get("ROWSTR_UNIT", "1"))   # 1: elements, 2: bytes


def capture_run(emit):
    ue.clear_capture_buffer()
    ue.start_capture()
    emit()
    with T._quiet():
        T._exec_captured(ue)
    return ue.report_latency_in_us()


def block_transpose(blk):
    def emit():
        ue._isa_reg_counter = 6
        ue.generate_instruction_add_set(R_BLKV, sh(blk * HD * BF16))
        ue.generate_instruction_add_set(R_BLKVT, sh((blk // CH) * HD * CH * BF16 + (blk % CH) * BF16))
        ue.generate_instruction_add_set(R_STRIDE, CH * BF16)   # the transpose reads this stride in BYTES
        ue.generate_instruction_add_imm(R_BLKV, sh(V_D), R_IN)
        ue.generate_instruction_add_imm(R_BLKVT, sh(VT_D), R_OUT)
        ue.bf16_transpose_core(M=VEC, N=HD, INPUT_DRAM_ADDR=V_D, OUTPUT_DRAM_ADDR=VT_D,
                               IDENTITY_DRAM_ADDR=ue._identity_dram_addr,
                               gpr_input_addr=R_IN, gpr_out_addr=R_OUT, gpr_out_row_stride_reg=R_STRIDE)
    return capture_run(emit)


done_blocks = set()
qb, Kb, Vb = (x.to(torch.bfloat16).float() for x in (q, K, V))
for pos in (5, 70, 1000, 16000):
    L = min(T_AL, (pos + 64) // 64 * 64)
    t_tr = 0.0
    for blk in range(0, pos // 64 * 64 + 1, 64):
        if blk not in done_blocks:
            t_tr = block_transpose(blk)
            done_blocks.add(blk)
    vt = T._read_bf(ue, VT_D, (T_AL // CH, HD, CH)).float().permute(1, 0, 2).reshape(HD, T_AL)
    vt_err = (vt[:, :pos + 1] - Vb[:pos + 1].t()).abs().max().item()
    bias = torch.full((1, T_AL), float("-inf"), dtype=torch.bfloat16)
    bias[:, :pos + 1] = 0.0
    ue.dma_write(T.DMA_DEVICE_H2C, B_D, bias, T_AL * BF16)

    def attn():
        ue._isa_reg_counter = 6
        ue.generate_instruction_add_set(R_LEN, L)
        ue.eltwise_core_dram(M=G, N=HD, dram_a=Q_D, dram_b=None, dram_out=SQ,
                             mode=T.UE_MODE.MUL_BROADCAST, scalar=1.0 / HD ** 0.5)
        # Q.K^T + mask + softmax one head per matmul (an output row stride with softmax breaks rows >= 1)
        for g in range(G):
            ue.matmat_mul_core(M=1, K=HD, N=T_AL, A_DRAM_ADDR=SQ + g * HD * BF16, B_DRAM_ADDR=K_D,
                               OUTPUT_DRAM_ADDR=SC_ROW + g * T_AL * BF16,
                               softmax_enable=True, C_DRAM_ADDR=B_D, bias_mode="broadcast_N", gpr_N_reg=R_LEN)
        for c in range(T_AL // CH):        # scores [G, T_AL] -> chunk layout [n_ch][G][CH] (card-side copy)
            ue.accelerator_memory_to_sram(SC_ROW + c * CH * BF16, 0x00000, G * CH,
                                          stride_bytes_per_chunk=CH * BF16, stride_jump_bytes=T_AL * BF16)
            ue.sram_to_accelerator_memory(0x00000, SC_CH + c * G * CH * BF16, G * CH)
        for c in range(T_AL // CH):        # out = sum over chunks; the previous partial is the bias of the next
            ue.matmat_mul_core(M=G, K=CH, N=HD, A_DRAM_ADDR=SC_CH + c * G * CH * BF16,
                               B_DRAM_ADDR=VT_D + c * HD * CH * BF16, OUTPUT_DRAM_ADDR=OUT,
                               C_DRAM_ADDR=OUT if c else None, bias_mode="full_matrix")
    us = capture_run(attn)
    out = T._read_bf(ue, OUT, (G, HD)).float()
    qbg = q.to(torch.bfloat16).float()
    p = torch.softmax((qbg @ Kb[:pos + 1].t()) / HD ** 0.5, 1)
    ref = p @ Vb[:pos + 1]
    errs = [((out[g] - ref[g]).norm() / ref[g].norm()).item() for g in range(G)]
    err = float("nan") if any(e != e for e in errs) else max(errs)   # max() would silently skip NaN
    if os.environ.get("TAILDBG"):
        flat = T._read_bf(ue, SC_ROW, (G * T_AL,)).float()
        fin = (flat != 0) & torch.isfinite(flat)
        idx = fin.nonzero().flatten()
        runs = [int(idx[0])] + [int(idx[i]) for i in range(1, len(idx)) if idx[i] != idx[i - 1] + 1] if len(idx) else []
        print("    per-head errors %s | finite score runs start at %s (L=%d, T_AL=%d)"
              % (["%.4f" % e for e in errs], runs[:8], L, T_AL))
    sc = T._read_bf(ue, SC_ROW, (G, T_AL)).float()
    tail = sc[:, L:].abs().max().item() if L < T_AL else 0.0
    if tail and os.environ.get("TAILDBG"):
        for g in range(G):
            nz = (sc[g, L:] != 0).nonzero().flatten()
            if len(nz):
                print("    row %d: %d nonzero beyond L, first at L+%d..L+%d, sum of row %.4f, row[L:L+4] %s"
                      % (g, len(nz), int(nz[0]), int(nz[-1]), sc[g].sum().item(), sc[g, L:L + 4].tolist()))
        print("    row sums over [0,L): %s" % [round(sc[g, :L].sum().item(), 4) for g in range(G)])
    print("pos %5d L %5d | block transpose %.0f us, V^T max err %.3f | attention (%d heads) %.0f us, worst rel err %.4f, score tail max %.1g"
          % (pos, L, t_tr, vt_err, G, us, err, tail), flush=True)
