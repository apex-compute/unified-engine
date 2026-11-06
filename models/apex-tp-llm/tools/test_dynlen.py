"""Unit test on one card: decode attention of one head (1 query row against the KV cache) with the
static length (T_al) and with the dynlen register (L = pos+1 rounded up to 64). A torch reference is
computed on the host only to check the card's numbers (test, not inference)."""
import importlib.util
import os
import sys

import torch

BASE = os.path.expanduser("~/apex_tp/models/q35_9b")
spec = importlib.util.spec_from_file_location("q35_dl_port", os.path.join(BASE, "qwen3.5_2b_test.py"))
T = importlib.util.module_from_spec(spec)
sys.modules["q35_dl_port"] = T
spec.loader.exec_module(T)
T.set_dma_device(sys.argv[1] if len(sys.argv) > 1 else "xdma0")
T.DMA_DEVICE_H2C = T.user_dma_core.DMA_DEVICE_H2C
T.DMA_DEVICE_C2H = T.user_dma_core.DMA_DEVICE_C2H
BF16 = 2
HD, T_AL = 256, 4096

with T._quiet():
    T.user_dma_core.configure_clock_from_hardware()
    ue = T.Qwen3_5_2b_UnifiedEngine(device="cpu", config_path=os.path.join(BASE, "qwen3.5_2b_config.json"))
    ue.software_reset()
    ue._preallocate_identity_matrix()
torch.manual_seed(0)
q = torch.randn(HD) * 0.5
K = torch.randn(T_AL, HD) * 0.5
V = torch.randn(T_AL, HD)
A = lambda n: ue.allocate_tensor_dram(n * BF16)
Q_D, K_D, V_D, B_D = A(HD), A(T_AL * HD), A(T_AL * HD), A(T_AL)
VT_D, SCR, OUT = A(T_AL * HD), A(HD * T_AL + T_AL * T_AL + 64 * HD), A(HD)
for addr, x in ((Q_D, q), (K_D, K), (V_D, V)):
    ue.dma_write(T.DMA_DEVICE_H2C, addr, T._bf(x.flatten()), x.numel() * BF16)

for pos in (5, 63, 64, 100, 700):
    L = min(T_AL, (pos + 64) // 64 * 64)
    bias = torch.full((1, T_AL), float("-inf"), dtype=torch.bfloat16)
    bias[:, :pos + 1] = 0.0
    ue.dma_write(T.DMA_DEVICE_H2C, B_D, bias, T_AL * BF16)
    qb, Kb, Vb = (x.to(torch.bfloat16).float() for x in (q, K, V))
    p = torch.softmax((Kb[:pos + 1] @ qb) / HD ** 0.5, 0)
    ref = p @ Vb[:pos + 1]
    res = {}
    for mode in ("static", "dynlen"):
        ue.clear_capture_buffer()
        ue.start_capture()
        reg = None
        if mode == "dynlen":
            reg = ue._ISA_ATTN_LEN_REG
            ue.generate_instruction_add_set(reg, L)
        ue.bf16_transpose_core(M=T_AL, N=HD, INPUT_DRAM_ADDR=V_D, OUTPUT_DRAM_ADDR=VT_D,
                               IDENTITY_DRAM_ADDR=ue._identity_dram_addr, gpr_M_reg=reg)
        kw = {"gpr_len_reg": reg} if reg is not None else {}
        ue.flash_attention_core_cached(V_T_DRAM_ADDR=VT_D, head_dim=HD, seq_len=T_AL, batch=1,
                                       Q_DRAM_ADDR=Q_D, K_DRAM_ADDR=K_D, V_DRAM_ADDR=V_D,
                                       OUTPUT_DRAM_ADDR=OUT, SCRATCH_DRAM_ADDR=SCR, BIAS_DRAM_ADDR=B_D, **kw)
        with T._quiet():
            T._exec_captured(ue)
        us = ue.report_latency_in_us()
        out = T._read_bf(ue, OUT, (HD,)).float()
        # also inspect the intermediate V^T rows and scores
        vt = T._read_bf(ue, VT_D, (HD, L if mode == "dynlen" else T_AL)).float()
        err = (out - ref).norm() / ref.norm()
        vt_err = (vt[:, :pos + 1] - Vb[:pos + 1].t()).abs().max().item()
        res[mode] = "rel err %.4f  V^T max err %.3f  %.0f us" % (err, vt_err, us)
    print("pos %4d L %4d | static: %s | dynlen: %s" % (pos, L, res["static"], res["dynlen"]), flush=True)

    # The real program reads the register many times in a row: 1 transpose, several heads, next layer.
    OUTS = [A(HD) for _ in range(4)]
    ue.clear_capture_buffer()
    ue.start_capture()
    reg = ue._ISA_ATTN_LEN_REG
    ue.generate_instruction_add_set(reg, L)
    for rnd in range(2):
        ue.bf16_transpose_core(M=T_AL, N=HD, INPUT_DRAM_ADDR=V_D, OUTPUT_DRAM_ADDR=VT_D,
                               IDENTITY_DRAM_ADDR=ue._identity_dram_addr, gpr_M_reg=reg)
        for h in range(2):
            ue.flash_attention_core_cached(V_T_DRAM_ADDR=VT_D, head_dim=HD, seq_len=T_AL, batch=1,
                                           Q_DRAM_ADDR=Q_D, K_DRAM_ADDR=K_D, V_DRAM_ADDR=V_D,
                                           OUTPUT_DRAM_ADDR=OUTS[2 * rnd + h], SCRATCH_DRAM_ADDR=SCR,
                                           BIAS_DRAM_ADDR=B_D, gpr_len_reg=reg)
    with T._quiet():
        T._exec_captured(ue)
    errs = ["%.4f" % ((T._read_bf(ue, o, (HD,)).float() - ref).norm() / ref.norm()) for o in OUTS]
    print("          repeated use of the register: rel err per call %s" % " ".join(errs), flush=True)
