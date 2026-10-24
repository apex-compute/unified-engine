"""Per-card host<->card costs of one tensor-parallel sync point: small DMA writes/reads and the
launch + completion poll of an (almost) empty program. Run while no other process uses the cards."""
import glob
import importlib.util
import os
import statistics
import sys
import time

import torch

BASE = os.path.expanduser("~/apex_tp/models/q35_9b")
spec = importlib.util.spec_from_file_location("q35_pcie_port", os.path.join(BASE, "qwen3.5_2b_test.py"))
T = importlib.util.module_from_spec(spec)
sys.modules["q35_pcie_port"] = T
spec.loader.exec_module(T)
N = 300


def cards():
    out = []
    for u in glob.glob("/sys/class/xdma/xdma*_user"):
        dev = os.path.basename(u)[:-len("_user")]
        bdf = os.path.basename(os.path.realpath(os.path.join(u, "device")))
        rd = lambda f: open("/sys/bus/pci/devices/%s/%s" % (bdf, f)).read().strip()
        out.append((bdf, dev, "x" + rd("current_link_width")))
    return sorted(out)


def timed(fn):
    v = []
    for _ in range(N):
        t = time.perf_counter()
        fn()
        v.append((time.perf_counter() - t) * 1e6)
    v.sort()
    return statistics.median(v), v[int(0.95 * N)]


for bdf, dev, width in cards():
    T.set_dma_device(dev)
    T.DMA_DEVICE_H2C = T.user_dma_core.DMA_DEVICE_H2C
    T.DMA_DEVICE_C2H = T.user_dma_core.DMA_DEVICE_C2H
    with T._quiet():
        T.user_dma_core.configure_clock_from_hardware()
        ue = T.Qwen3_5_2b_UnifiedEngine(device="cpu", config_path=os.path.join(BASE, "qwen3.5_2b_config.json"))
        ue.software_reset()
    addr = 0x60000000
    res = {}
    for kb in (8, 40, 160):
        buf = torch.zeros(kb * 512, dtype=torch.bfloat16)
        res["write %3d KB" % kb] = timed(lambda: ue.dma_write(T.DMA_DEVICE_H2C, addr, buf, kb * 1024))
    rbuf = torch.empty(4096, dtype=torch.bfloat16)
    res["read    8 KB"] = timed(lambda: ue.dma_read(T.DMA_DEVICE_C2H, addr, rbuf, 8192))
    # an (almost) empty program: one halt
    prog = ue.get_program_dram_addr()
    ue.clear_capture_buffer()
    T.UnifiedEngine.start_capture(ue)
    ue.generate_instruction_halt()
    T.UnifiedEngine.stop_capture(ue)
    ue.write_captured_instructions_to_dram(prog)
    ue.clear_capture_buffer()

    def launch():
        ue.start_execute_from_dram(prog)
        while ue.is_queue_busy():
            pass
    res["launch+poll halt"] = timed(launch)
    res["latency reg read"] = timed(lambda: ue.report_latency_in_us())
    print("%s %s %s | " % (dev, bdf, width) + " | ".join("%s %.0f us (p95 %.0f)" % (k, m, p) for k, (m, p) in res.items()),
          flush=True)
