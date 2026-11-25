"""Small-DMA latency when all 5 cards are driven at once (one process per card, like the runner).
pwrite/pread on persistent fds only (never the vectored calls, which crash xdma). Cards must be idle.
    bench_dmaconc.py <cpu list for the 5 card processes> [spin]
spin=1: each process busy-polls its card's queue register between transfers, like the runner does."""
import glob
import multiprocessing as mp
import os
import statistics
import sys
import time

ADDR = 0x7F000000          # scratch in DDR0; every runner re-uploads weights and programs at start
N = 3000
QUEUE_REG = 0x02000000 + 0x54   # engine 0 queue control register, the one is_queue_busy() reads


def devs():
    out = []
    for u in glob.glob("/sys/class/xdma/xdma*_user"):
        dev = os.path.basename(u)[:-len("_user")]
        out.append((os.path.basename(os.path.realpath(os.path.join(u, "device"))), dev))
    return [d for _, d in sorted(out)]


def run(dev, cpu, spin, start, q):
    os.sched_setaffinity(0, {cpu})
    w = os.open("/dev/%s_h2c_0" % dev, os.O_RDWR)
    r = os.open("/dev/%s_c2h_0" % dev, os.O_RDWR)
    u = os.open("/dev/%s_user" % dev, os.O_RDWR)
    b = bytes(256)
    while start.value == 0:
        pass
    lw, lr = [], []
    for i in range(N):
        t = time.perf_counter_ns()
        os.pwrite(w, b, ADDR)
        t2 = time.perf_counter_ns()
        os.pread(r, 256, ADDR)
        t3 = time.perf_counter_ns()
        lw.append((t2 - t) / 1e3); lr.append((t3 - t2) / 1e3)
        if spin:                                   # ~200 us of register polling, like waiting on a segment
            e = time.perf_counter_ns() + 200_000
            while time.perf_counter_ns() < e:
                os.pread(u, 4, QUEUE_REG)
    lw.sort(); lr.sort()
    q.put("%s cpu %2d | write 256 B median %5.1f p90 %6.1f | read 256 B median %5.1f p90 %6.1f" % (
        dev, cpu, statistics.median(lw), lw[9 * N // 10], statistics.median(lr), lr[9 * N // 10]))


if __name__ == "__main__":
    cpus = [int(c) for c in sys.argv[1].split(",")]
    spin = len(sys.argv) > 2 and sys.argv[2] == "1"
    ctx = mp.get_context("fork")
    start, q = ctx.Value("i", 0), ctx.Queue()
    ps = [ctx.Process(target=run, args=(d, c, spin, start, q)) for d, c in zip(devs(), cpus)]
    for p in ps:
        p.start()
    time.sleep(1.0)
    start.value = 1
    res = [q.get() for _ in ps]
    for p in ps:
        p.join()
    print("cpus %s spin %d" % (sys.argv[1], spin))
    for line in sorted(res):
        print("  " + line)
