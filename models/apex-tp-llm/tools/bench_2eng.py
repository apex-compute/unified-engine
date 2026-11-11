"""Two engines on one card, two DDR3 channels (address bit 31). Times the same IF4 GEMV
(x[1,K] @ W[N,K]^T) run by engine 0 alone, by engine 1 alone, and by both at once, with the weights
in DDR0 (0x1000_0000) or DDR1 (0x9000_0000). Run with nothing else using the card."""
import importlib.util
import os
import sys
import time

import torch

BASE = os.path.expanduser("~/apex_tp/models/q35_9b")
spec = importlib.util.spec_from_file_location("q35_2e_port", os.path.join(BASE, "qwen3.5_2b_test.py"))
T = importlib.util.module_from_spec(spec)
sys.modules["q35_2e_port"] = T
spec.loader.exec_module(T)
U = T.user_dma_core
dev = sys.argv[1] if len(sys.argv) > 1 else "xdma0"
T.set_dma_device(dev)
U.configure_clock_from_hardware()
H2C = U.DMA_DEVICE_H2C
BF16 = 2
N, K = int(os.environ.get("N", "8192")), int(os.environ.get("K", "4096"))   # 16 MB of IF4 weights

W_DDR = {0: 0x1000_0000, 1: 0x9000_0000}            # weights (data, then scales) in each channel
# per engine: params / tensor / program cursors, apart from each other and from the weights
LAYOUT = {0: dict(params_dram_base=0x0400_0000, tensor_dram_base=0x0800_0000, program_dram_base=0x0C00_0000),
          1: dict(params_dram_base=0x8400_0000, tensor_dram_base=0x8800_0000, program_dram_base=0x8C00_0000)}

with T._quiet():
    eng = {i: U.UnifiedEngine(BASE_ADDR=U.UE_0_BASE_ADDR + i * 0x10000, init_unified_engine=False, **LAYOUT[i])
           for i in (0, 1)}
    for e in eng.values():
        e.software_reset()

torch.manual_seed(0)
scales, packed = T.quantize_fp4_64(torch.randn(N, K) * 0.02)
for ch, base in W_DDR.items():
    eng[0].dma_write(H2C, base, packed, packed.numel())
    eng[0].dma_write(H2C, base + packed.numel(), scales, scales.numel() * BF16)
x = T._bf(torch.randn(K))
prog = {}


def build(i, ch):
    """Program for engine i reading the weights in channel ch; returns its start address."""
    e = eng[i]
    X = LAYOUT[i]["tensor_dram_base"]
    OUT = X + 0x10_0000
    e.dma_write(H2C, X, x, K * BF16)
    e.clear_capture_buffer()
    e.start_capture()
    e.quantized_matmat_core(M=1, K=K, N=N, A_DRAM_ADDR=X, B_DRAM_ADDR=W_DDR[ch], OUTPUT_DRAM_ADDR=OUT,
                            SCALE_DRAM_ADDR=W_DDR[ch] + packed.numel(), data_type=U.TYPE.IF4)
    e.stop_capture()
    e.generate_instruction_halt()
    addr = LAYOUT[i]["program_dram_base"] + ch * 0x100_0000
    e.write_captured_instructions_to_dram(addr)
    e.clear_capture_buffer()
    return addr


for i in (0, 1):
    for ch in (0, 1):
        prog[i, ch] = build(i, ch)


def run(jobs, reps=20):
    best = None
    for _ in range(reps):
        t = time.perf_counter()
        for i, ch in jobs:
            eng[i].start_execute_from_dram(prog[i, ch])
        for i, _ in jobs:
            while eng[i].is_queue_busy():
                pass
        wall = (time.perf_counter() - t) * 1e6
        hw = [eng[i].report_latency_in_us() for i, _ in jobs]
        if best is None or wall < best[0]:
            best = (wall, hw)
    return best


MB = packed.numel() / 1e6
cases = [("engine 0 alone, weights DDR0", [(0, 0)]),
         ("engine 0 alone, weights DDR1", [(0, 1)]),
         ("engine 1 alone, weights DDR0", [(1, 0)]),
         ("engine 1 alone, weights DDR1", [(1, 1)]),
         ("both at once, e0 DDR0 + e1 DDR1", [(0, 0), (1, 1)]),
         ("both at once, e0 DDR1 + e1 DDR0", [(0, 1), (1, 0)]),
         ("both at once, both DDR0", [(0, 0), (1, 0)]),
         ("both at once, both DDR1", [(0, 1), (1, 1)])]
print("GEMV %d x %d IF4 = %.1f MB of weights per engine" % (N, K, MB))
for name, jobs in cases:
    wall, hw = run(jobs)
    total = MB * len(jobs)
    print("%-36s wall %7.0f us | engine time %s | %.2f GB/s total" % (
        name, wall, " ".join("%.0f us" % h for h in hw), total / wall * 1e3), flush=True)   # MB/us -> GB/s
