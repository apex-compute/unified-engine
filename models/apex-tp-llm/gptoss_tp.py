"""Run GPT-OSS-20B tensor-parallel on the cards: 5 cards x 2 engines = 10 shards (see gptoss_card.py).

Host loop per token: A (attention) / R (router) / E (experts) per layer, then the LM head.
After R the host reads the 32 router logits from one engine, picks the top 4 (indices only) and the
cards' processes copy that pick (indices + the 4 logits) into their engines before E.

    python gptoss_tp.py --engines 2 --prompt "..." --max-new-tokens 64
"""
import argparse
import ctypes
import glob
import json
import multiprocessing as mp
import os
import struct
import sys
import time
import traceback
from multiprocessing import shared_memory

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gptoss_card as G  # noqa: E402

WORK = os.path.expanduser(os.environ.get("GPTOSS_WORK", "~/apex_tp/models/gptoss20b"))
WDIR = os.path.join(WORK, "weights")
LOGDIR = os.path.join(WORK, "logs")
MODEL = os.path.expanduser(os.environ.get("GPTOSS_MODEL", "~/apex_exp/models/gpt_oss_20b/hf/gpt-oss-20b"))
BF16 = 2
SEG_QUIT, SEG_RESET, SEG_PARK = -1, -2, -3
FAILED = -1
CHANNEL_BYTES = 0x8000_0000
PROGRAM_BYTES = 64 << 20


def cards_in_pci_order():
    out = []
    for u in glob.glob("/sys/class/xdma/xdma*_user"):
        dev = os.path.basename(u)[:-len("_user")]
        out.append((os.path.basename(os.path.realpath(os.path.join(u, "device"))), dev))
    return [d for _, d in sorted(out)]


def cpu_plan(n):
    env = os.environ.get("APEX_TP_CPUS", "")
    if env:
        c = [int(v) for v in env.split(",")]
        return c[0], [c[1 + k % (len(c) - 1)] for k in range(n)]
    return 0, [1 + k % (os.cpu_count() - 1) for k in range(n)]


def engine_window(e, E, cdir):
    if E == 1:
        return {}
    w = e * CHANNEL_BYTES
    on_disk = sum(os.path.getsize(f) for f in glob.glob(os.path.join(cdir, "*.pt")))
    params = (int(on_disk * 1.05) + (128 << 20) + (16 << 20) - 1) // (16 << 20) * (16 << 20)
    return {"params_dram_base": w, "tensor_dram_base": w + params, "program_dram_base": w + CHANNEL_BYTES - PROGRAM_BYTES}


class Shared:
    def __init__(self, P, H, names=None):
        spec = (("cmd", np.int64, (4,)), ("x", np.int16, (H,)), ("done", np.int64, (P,)),
                # meta[k]: 0 card us, 1 process ms, then (token index, logit) for the shard's best 4: 2+2r / 3+2r
                ("meta", np.float64, (P, 12)), ("out", np.int16, (P, H)),
                ("selc", np.int64, (P, 4)))            # each card's expert pick (checked to agree)
        self.blocks, self.names = [], []
        for i, (name, dtype, shape) in enumerate(spec):
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            if names is None:
                shm = shared_memory.SharedMemory(create=True, size=size)
            else:
                try:
                    shm = shared_memory.SharedMemory(name=names[i], track=False)   # Python >= 3.13
                except TypeError:
                    shm = shared_memory.SharedMemory(name=names[i])
            self.blocks.append(shm); self.names.append(shm.name)
            arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            if names is None:
                arr[...] = 0
            setattr(self, name, arr)


def worker(c, E, P, dev, tdir, max_ctx, cpu, shm_names, conn):
    os.makedirs(LOGDIR, exist_ok=True)
    log = open(os.path.join(LOGDIR, "card_%02d.log" % c), "w", buffering=1)
    os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
    sys.stdout = sys.stderr = log
    import faulthandler
    faulthandler.enable(file=log)
    ks = [c * E + e for e in range(E)]
    sh = None
    try:
        os.sched_setaffinity(0, {cpu})
        ctypes.CDLL(None).prctl(29, 1, 0, 0, 0)
        torch.set_num_threads(1)
        t0 = time.time()
        plan = json.load(open(os.path.join(tdir, "plan.json")))
        shards = []
        for e, k in enumerate(ks):                     # build every engine before uploading anything
            T, qcfg = G.load_base_port("gptoss_port_%d" % k)
            cdir = os.path.join(tdir, "card_%02d" % k)
            shards.append(G.GptOssShard(T, qcfg, dev, e, cdir, WDIR, plan, max_ctx, window=engine_window(e, E, cdir)))
        with shards[0].T._quiet():
            for s in shards:
                s.prepare()
        H = shards[0].H
        n_parts = P // E                               # presum: one combined share per card
        vocab = [plan["cards"][k]["vocab"] for k in ks]
        prog0 = [s.ue.get_program_dram_addr() for s in shards]
        with shards[0].T._quiet():
            n_seg = shards[0].compile(n_parts, vocab[0][1] - vocab[0][0])
            for s, v in zip(shards[1:], vocab[1:]):    # one copy of the shares per card: engine 0's buffer
                n_seg = s.compile(n_parts, v[1] - v[0], parts_addr=shards[0].PARTS)
        # engine 0 sums the card's shares on the card: SHARE_x(e0) += SHARE_x(e1)
        add_prog = {}
        if E == 2:
            s0, s1 = shards
            U, ue0 = s0.U, s0.ue
            for kind, a0, a1 in (("A", s0.SHARE_A, s1.SHARE_A), ("E", s0.SHARE_E, s1.SHARE_E)):
                ue0.clear_capture_buffer(); U.UnifiedEngine.start_capture(ue0)
                ue0.ue_memcpy_from_dram(a0, H * BF16, U.MEMCPY_TYPE.URAM.value, U.URAM_START_ADDR, U.URAM_SECTION.URAM_A.value)
                ue0.ue_memcpy_from_dram(a1, H * BF16, U.MEMCPY_TYPE.URAM.value, U.URAM_START_ADDR, U.URAM_SECTION.URAM_B.value)
                ue0.eltwise_add_core(0x00000, 0x80000, 0x00000, H)
                ue0.ue_memcpy_to_dram(U.MEMCPY_TYPE.URAM.value, U.URAM_SECTION.URAM_A.value, U.URAM_START_ADDR, a0, H * BF16)
                ue0.generate_instruction_halt(); U.UnifiedEngine.stop_capture(ue0)
                addr = ue0.get_program_dram_addr()
                ue0.write_captured_instructions_to_dram(addr)
                ue0.allocate_program_dram(ue0.get_capture_instruction_size_bytes() + 64)
                ue0.clear_capture_buffer()
                add_prog[kind] = addr
        def tensor_mb(s):
            a, b = getattr(s.ue, "_next_tensor_dram_addr", None), getattr(s.ue, "_tensor_dram_base", None)
            return (a - b) / 1e6 if a is not None and b is not None else 0.0
        info = [{"dev": "%s/e%d" % (dev, e), "params_mb": (s.ue._next_params_dram_addr - s.ue._params_dram_base) / 1e6,
                 "prog_mb": (s.ue.get_program_dram_addr() - p0) / 1e6, "tensor_mb": tensor_mb(s),
                 "kv_mb": s.L * s.T_ext * (G.KQ + 2 * s.HD) * BF16 / 1e6,
                 "setup_s": time.time() - t0, "segs": n_seg,
                 "clock_mhz": 1000.0 / float(getattr(s.ue, "_clock_period_ns", 0) or 5.042),
                 "firmware": "%08x" % s.ue.get_hardware_version()} for e, (s, p0) in enumerate(zip(shards, prog0))]
        sh = Shared(P, H, shm_names)
        # the card's DMA channels opened ONCE: the vendor dma_write/dma_read open and close the device on
        # every call, which costs more than the small transfer itself (~170 transfers per token)
        h2c = os.open(shards[0].T.DMA_DEVICE_H2C, os.O_RDWR)
        c2h = os.open(shards[0].T.DMA_DEVICE_C2H, os.O_RDWR)
        for x in shards:
            x.fast_h2c = h2c

        def rd(addr, mv):
            # pread, NOT preadv/readv: the vectored calls enter xdma's aio path (cdev_aio_read), which
            # dereferences NULL on a synchronous request and takes the kernel down (seen 2026-10-21)
            data = os.pread(c2h, len(mv), addr)
            if len(data) != len(mv):
                raise OSError("short c2h read at 0x%x" % addr)
            mv[:] = data

        parity_in = [memoryview(sh.out[i * n_parts:(i + 1) * n_parts]).cast("B") for i in range(2)]
        parity_out = [memoryview(sh.out[i * n_parts + c]).cast("B") for i in range(2)]
        logits_i16 = np.zeros(64, dtype=np.int16)
        logits_mv = memoryview(logits_i16).cast("B")
        head_i16 = np.zeros(32, dtype=np.int16)
        head_mv = memoryview(head_i16).cast("B")
        prof = os.environ.get("GPTOSS_PROF") == "1"
        # engine control without the vendor's per-call wrappers: the same AXI-Lite registers and values as
        # start_execute_from_dram() / is_queue_busy(), one pwrite / pread on the engine's user fd
        U = shards[0].U
        pk32 = struct.Struct("=I").pack
        ctl = []
        for x in shards:
            b = x.ue._base_addr - U.AXI_LITE_TRANSLATION_OFFSET
            ctl.append((x.ue._get_user_fd(), b + U.UE_INSTRUCTION_ADDR, b + U.UE_QUEUE_CTRL_ADDR))
        go = [[pk32(U.ue_35bit_addr_shifter(a) & 0xFFFFFFFF) for a in x.seg_start] for x in shards]
        add_go = {kk: pk32(U.ue_35bit_addr_shifter(a) & 0xFFFFFFFF) for kk, a in add_prog.items()}
        pread, pwrite = os.pread, os.pwrite

        def wait_idle(fd, q):
            while pread(fd, 4, q)[1] & 1:              # queue_busy is bit 8 of UE_QUEUE_CTRL
                pass

        unpack32h = struct.Struct("<32h").unpack
        LOGITS = shards[0].LOGITS
        T = shards[0].T
        kinds = shards[0].seg_kind
        conn.send(("ready", info))
        cmd, done, meta = sh.cmd, sh.done, sh.meta
        fine, fine_n = {}, {}
        pick_t = [0, 0]

        def loop():
            last = 0
            n_share = 0                                # how many shares have been produced this token
            parked = False                             # between requests: poll slowly instead of spinning
            while True:
                while cmd[0] == last:
                    if parked:
                        time.sleep(0.0005)
                    else:
                        os.sched_yield()
                last = int(cmd[0]); s = int(cmd[1]); pos = int(cmd[2])
                parked = s == SEG_PARK
                if parked:
                    for k in ks:
                        done[k] = last
                    continue
                if s == SEG_QUIT:
                    for kk, v in fine.items():
                        n = max(fine_n[kk], 1)
                        print("FINE|%s|n %d|pre %.0f us|start %.0f us|wait %.0f us|latreg %.0f us|post %.0f us|hw %.0f us"
                              % ((kk, fine_n[kk]) + tuple(1e6 * x / n for x in v)), flush=True)
                    for e, x in enumerate(shards):
                        print("SMALL|e%d|pwrite %.1f us x %d|set_experts buffer %.1f us x %d" % (
                            e, x.prof_w[0] / 1e3 / max(x.prof_w[1], 1), x.prof_w[1],
                            x.prof_b[0] / 1e3 / max(x.prof_b[1], 1), x.prof_b[1]), flush=True)
                    if pick_t[1]:
                        print("PICK|logits read + pick + set_experts %.1f us x %d" % (pick_t[0] / 1e3 / pick_t[1], pick_t[1]),
                              flush=True)
                    break
                if s == SEG_RESET:
                    for x in shards:
                        x.reset_state()
                    for k in ks:
                        done[k] = last
                    continue
                n_share = run_seg(s, pos, n_share)
                if kinds[s] == "R":
                    # the expert pick happens HERE, in this card's process, from this card's own router logits:
                    # every card computes the router from the same summed residual with the same weights, so
                    # all cards pick the same experts and no global sync is needed between R and E
                    tp0 = time.perf_counter_ns()
                    vals = unpack32h(pread(c2h, 64, LOGITS))      # the 32 router logits, bf16 bit patterns
                    # bf16 bits -> an int key with the same order as the float values
                    keys = [v if v >= 0 else -32768 - v for v in vals]
                    top = sorted(range(32), key=keys.__getitem__, reverse=True)[:4]
                    sh.selc[c, :] = top
                    li = (s - 1) // 3
                    lg4 = [vals[i] for i in top]
                    for x in shards:
                        x.set_experts_bf(li, top, lg4)
                    pick_t[0] += time.perf_counter_ns() - tp0; pick_t[1] += 1
                    n_share = run_seg(s + 1, -1, n_share)
                for k in ks:
                    done[k] = last

        def run_seg(s, pos, n_share):
                t1 = time.perf_counter()
                kind = kinds[s]
                if s == 0:
                    xb = sh.x.tobytes()
                    for x in shards:
                        x.set_position(pos, xb)          # embedding row + masks + registers, one DMA
                    n_share = 0
                elif kind in ("R", "A", "H"):          # these start by adding the previous segment's shares
                    if os.pwrite(h2c, parity_in[(n_share - 1) % 2], shards[0].PARTS) != n_parts * H * BF16:
                        raise OSError("short parts write")
                tA = time.perf_counter()
                for (fd, ia, q), g in zip(ctl, go):
                    pwrite(fd, g[s], ia)
                tB = time.perf_counter()
                for fd, ia, q in ctl:
                    wait_idle(fd, q)
                tC = time.perf_counter()
                hw = [x.ue.report_latency_in_us() for x in shards] if prof else [0.0] * E
                tD = time.perf_counter()
                if kind in ("A", "E"):                 # this segment produced a share
                    share = shards[0].SHARE_A if kind == "A" else shards[0].SHARE_E
                    if E == 2:
                        fd, ia, q = ctl[0]
                        pwrite(fd, add_go[kind], ia)
                        wait_idle(fd, q)
                    rd(share, parity_out[n_share % 2])
                    n_share += 1
                elif kind == "H":                      # the shard's best `want` tokens (the card tracks 4)
                    want = max(1, min(int(cmd[3]), 4))
                    for x, k in zip(shards, ks):
                        for r in range(want):
                            idx = int(x.ue.get_arg_max_index(rank=r + 1))
                            rd(x.HEAD_OUT + (idx // 32) * 32 * BF16, head_mv)
                            meta[k, 2 + 2 * r] = idx
                            meta[k, 3 + 2 * r] = float(
                                (head_i16[idx % 32:idx % 32 + 1].astype(np.int32) << 16).view(np.float32)[0])
                if prof:
                    tE = time.perf_counter()
                    fk = "H" if kind == "H" else ("A0" if s == 0 else kind)
                    if fk not in fine:
                        fine[fk] = np.zeros(6); fine_n[fk] = 0
                    fine[fk] += (tA - t1, tB - tA, tC - tB, tD - tC, tE - tD, max(hw) / 1e6)
                    fine_n[fk] += 1
                    for x, k, us in zip(shards, ks, hw):
                        meta[k, 0] = us / 1e3
                        meta[k, 1] = (time.perf_counter() - t1) * 1e3
                return n_share

        with T._quiet():
            loop()
    except Exception:
        traceback.print_exc()
        try:
            if sh is not None:
                for k in ks:
                    sh.done[k] = FAILED
            conn.send(("error", traceback.format_exc()[-1500:]))
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", default="10")
    ap.add_argument("--engines", type=int, default=2)
    ap.add_argument("--prompt", default="What is the capital of France? Answer in one sentence.")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--reasoning", default="low")
    a = ap.parse_args()
    ctypes.CDLL(None).prctl(29, 1, 0, 0, 0)
    tdir = os.path.join(WORK, "tp%s" % a.tp)
    plan = json.load(open(os.path.join(tdir, "plan.json")))
    P, H = plan["P"], plan["hidden"]
    E = a.engines
    n_cards = P // E
    host_cpu, card_cpus = cpu_plan(n_cards)
    os.sched_setaffinity(0, {host_cpu})
    devs = cards_in_pci_order()[:n_cards]
    voff = [c["vocab"][0] for c in plan["cards"]]
    sh = Shared(P, H)
    ctx = mp.get_context("spawn")
    pipes, procs = [], []
    t0 = time.time()
    for c, dev in enumerate(devs):
        pa, pb = ctx.Pipe()
        p = ctx.Process(target=worker, args=(c, E, P, dev, tdir, a.max_context, card_cpus[c], sh.names, pb), daemon=True)
        p.start(); pipes.append(pa); procs.append(p)
    n_seg = None
    for c, pc in enumerate(pipes):
        m = pc.recv()
        if m[0] != "ready":
            print("ERROR|card %d|%s" % (c, m[1])); return 1
        for i in m[1]:
            n_seg = i["segs"]
            print("CARD|%s|weights %.0f MB|setup %.0f s" % (i["dev"], i["params_mb"], i["setup_s"]), flush=True)
    print("READY|%d cards x %d engines|%d segments/token|%.0f s" % (n_cards, E, n_seg, time.time() - t0), flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    text = tok.apply_chat_template([{"role": "user", "content": a.prompt}], tokenize=False,
                                   add_generation_prompt=True, reasoning_effort=a.reasoning)
    ids = tok.encode(text, add_special_tokens=False)
    # harmony: the analysis message ends with <|end|> and the final answer follows; stop at the turn's end
    stops = {tok.convert_tokens_to_ids(t) for t in ("<|return|>", "<|call|>")}
    shape = json.load(open(os.path.join(WDIR, "embed.json")))["shape"]
    embed = np.fromfile(os.path.join(WDIR, "embed.bin"), dtype=np.int16).reshape(tuple(shape))
    seq = [0]

    def command(s, pos):
        sh.cmd[1] = s; sh.cmd[2] = pos
        seq[0] += 1; sh.cmd[0] = seq[0]
        n = 0
        while not (sh.done == seq[0]).all():
            n += 1
            if n & 0x3FF == 0 and (sh.done == FAILED).any():
                k = int(np.argmax(sh.done == FAILED)) // E
                raise RuntimeError("card %d: %s" % (k, pipes[k].recv()[1] if pipes[k].poll(2.0) else "no detail"))

    seg_t, seg_hw, seg_wk = np.zeros(4), np.zeros(4), np.zeros(4)
    kinds = "ARE"

    disagree = [0]

    def token_step(t, pos, measure):
        sh.x[:] = embed[t]
        for s in range(n_seg):
            if s < n_seg - 1 and s % 3 == 2:           # E runs inside the R command (each card picks locally)
                continue
            t_s = time.perf_counter()
            command(s, pos if s == 0 else -1)
            if s < n_seg - 1 and s % 3 == 1 and (sh.selc[:n_cards] != sh.selc[0]).any():
                disagree[0] += 1
            if measure:
                ki = "AREH".index("H" if s == n_seg - 1 else kinds[s % 3])
                seg_t[ki] += time.perf_counter() - t_s
                seg_hw[ki] += sh.meta[:, 0].max() / 1e3        # slowest engine's own compute
                seg_wk[ki] += sh.meta[:, 1].max() / 1e3        # slowest card process, command to done
        k = int(np.argmax(sh.meta[:, 3]))
        return voff[k] + int(sh.meta[k, 2])

    command(SEG_RESET, -1)
    print("PROMPT|%d tokens" % len(ids), flush=True)
    t1 = time.time()
    nxt = None
    for pos, t in enumerate(ids):
        nxt = token_step(t, pos, False)
    tp = time.time() - t1
    print("PREFILL|%d tokens|%.1f s|%.2f tokens/s" % (len(ids), tp, len(ids) / tp), flush=True)
    out = [nxt]
    t2 = time.time()
    pos = len(ids)
    while len(out) < a.max_new_tokens and out[-1] not in stops:
        out.append(token_step(out[-1], pos, True)); pos += 1
    td = time.time() - t2
    nd = len(out) - 1
    print("DECODE|%d tokens|%.1f s|%.2f tokens/s|%.1f ms/token" % (nd, td, nd / max(td, 1e-9), 1e3 * td / max(nd, 1)),
          flush=True)
    if nd:
        print("HOST|per token: attention %.1f ms|router+experts %.1f ms|(experts alone) %.1f ms|head %.1f ms" %
              tuple(1e3 * v / nd for v in seg_t), flush=True)
    if nd and os.environ.get("GPTOSS_PROF") == "1":
        print("CARD|slowest engine compute: attention %.1f ms|router %.1f ms|experts %.1f ms|head %.1f ms" %
              tuple(1e3 * v / nd for v in seg_hw), flush=True)
        print("WORKER|slowest card process: attention %.1f ms|router %.1f ms|experts %.1f ms|head %.1f ms" %
              tuple(1e3 * v / nd for v in seg_wk), flush=True)
    print("PICK|layers where the cards' expert picks differed: %d" % disagree[0], flush=True)
    print("TEXT_BEGIN\n%s\nTEXT_END" % tok.decode(out), flush=True)
    sh.cmd[1] = SEG_QUIT; sh.cmd[0] = seq[0] + 1
    for p in procs:
        p.join(10)
    return 0


if __name__ == "__main__":
    sys.exit(main())
