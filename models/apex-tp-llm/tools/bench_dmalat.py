"""Latency of single small DMA calls on a persistent fd (pwrite / pread only: never the vectored calls,
which crash xdma). Usage: bench_dmalat.py <dev> <cpu>. Writes go to a scratch area at the top of DDR0 that
no runner uses while this runs (run with the cards idle)."""
import os
import statistics
import sys
import time

dev, cpu = sys.argv[1], int(sys.argv[2])
os.sched_setaffinity(0, {cpu})
ADDR = 0x7F000000                      # scratch in DDR0; every runner re-uploads weights and programs at start
w = os.open("/dev/%s_h2c_0" % dev, os.O_RDWR)
r = os.open("/dev/%s_c2h_0" % dev, os.O_RDWR)


def cmpl_times():
    out = {}
    for t in os.listdir("/proc"):
        if not t.isdigit():
            continue
        try:
            name = open("/proc/%s/comm" % t).read().strip()
            if name.startswith("cmpl_status_th"):
                f = open("/proc/%s/stat" % t).read().split()
                out[name] = int(f[13]) + int(f[14])
        except OSError:
            pass
    return out


def lat(fn, n=2000):
    for _ in range(50):
        fn()
    v = []
    for _ in range(n):
        t = time.perf_counter_ns()
        fn()
        v.append((time.perf_counter_ns() - t) / 1e3)
    v.sort()
    return "%5.1f us (p10 %5.1f, p90 %5.1f)" % (statistics.median(v), v[n // 10], v[9 * n // 10])


b128, b28k = bytes(128), bytes(28800)
before = cmpl_times()
print("%s cpu %2d | write 128 B %s | write 28.8 KB %s | read 128 B %s | read 5.7 KB %s" % (
    dev, cpu,
    lat(lambda: os.pwrite(w, b128, ADDR)),
    lat(lambda: os.pwrite(w, b28k, ADDR)),
    lat(lambda: os.pread(r, 128, ADDR)),
    lat(lambda: os.pread(r, 5760, ADDR))), flush=True)
after = cmpl_times()
busy = {k: after[k] - before.get(k, 0) for k in after if after[k] - before.get(k, 0) > 0}
print("   completion threads that worked (ticks):", dict(sorted(busy.items())), flush=True)
