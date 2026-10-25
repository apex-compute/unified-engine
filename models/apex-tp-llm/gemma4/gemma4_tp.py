"""Run Gemma 4 26B-A4B tensor-parallel on the cards: 5 cards x 2 engines = 10 shards (see gemma4_card.py).

Host loop per token: A (attention) / R (router) / E (FFN) per layer, then the LM head.
After R each card's process reads its own card's 128 router logits, picks the top 8 (indices only)
and copies that pick (indices + the 8 logits) into its engines before E. Every card computes the
router from the same summed residual, so all cards pick the same experts (checked every layer).

Shares: attention -> H per card, FFN -> 2H per card (dense | MoE, normed separately on the cards).

    python gemma4_tp.py --prompt "..." --max-new-tokens 64
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
import gemma4_card as G  # noqa: E402

WORK = os.path.expanduser(os.environ.get("GEMMA4_WORK", os.path.join(os.environ.get("APEX_TP_WORK", "~/apex_tp"), "models", "gemma4_26b")))
WDIR = os.path.join(WORK, "weights")
LOGDIR = os.path.join(WORK, "logs")
MODEL = os.path.expanduser(os.environ.get("GEMMA4_MODEL", "~/apex_exp/models/gemma4_26b/hf/gemma-4-26b-a4b-it"))
BF16 = 2
SEG_QUIT, SEG_RESET, SEG_PARK = -1, -2, -3
SEG_BLOCK_BEGIN, SEG_BLOCK_END = -4, -5          # an image block: embeddings in Shared.xb, (p0, n) in Shared.blk
BLK = 1 << 20                                    # command BLK + s: block program segment s for token cmd[2]
MAX_BLOCK = int(os.environ.get("GEMMA4_MAX_BLOCK", "0"))    # longest image block (0: no image support)
# the image encoder, tensor-parallel over the first VIS_P shards (0: none); levels = soft tokens per image
VIS_P = int(os.environ.get("GEMMA4_VISION_SHARDS", "0"))
VIS_LEVELS = [int(v) for v in os.environ.get("GEMMA4_VISION_LEVELS", "70,140,280").split(",")]
VIS_DEPTH = int(os.environ.get("GEMMA4_VISION_DEPTH", "0")) or None          # unit tests: stop after this many layers
VIS_PATH = os.path.join(WORK, "vision", "vision.pt")
VIS_HID, VIS_POS, VIS_OUT = 1152, 1024, 2816
SEG_VIS_BEGIN = -6                               # inputs of one image (level cmd[2]) from Shared.vin_* to the cards
VIS = 2 << 20                                    # command VIS + i: encoder segment i
FAILED = -1
CHANNEL_BYTES = 0x8000_0000
PROGRAM_BYTES = 64 << 20
NE, TOPK = 128, G.TOPK


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


class CardEmbed:
    """embed[token] -> the token's embedding row (int16 bf16 bits), read from the card that holds it.
    The table lives on the cards (each shard its vocabulary block, see slice_embed.py); the host only copies
    one row per token, over each card's second c2h channel (the card processes use the first)."""

    def __init__(self, plan, devs, E, emb_addr, H):
        self.voff = [c["vocab"][0] for c in plan["cards"]]
        self.addr, self.E, self.nb = emb_addr, E, H * BF16
        self.fds = [os.open("/dev/%s_c2h_1" % d, os.O_RDWR) for d in devs]

    def __getitem__(self, t):
        import bisect
        k = bisect.bisect_right(self.voff, int(t)) - 1
        data = os.pread(self.fds[k // self.E], self.nb, self.addr[k] + (int(t) - self.voff[k]) * self.nb)
        if len(data) != self.nb:
            raise OSError("short embedding read")
        return np.frombuffer(data, dtype=np.int16)

    def close(self):
        for fd in self.fds:
            os.close(fd)


class Shared:
    def __init__(self, P, H, n_cards, names=None):
        vnp = (9 * max(VIS_LEVELS) + 63) // 64 * 64 if VIS_P else 1
        spec = (("cmd", np.int64, (4,)), ("x", np.int16, (H,)), ("done", np.int64, (P,)),
                # meta[k]: 0 card us, 1 process ms, then (token index, logit) for the shard's best 4: 2+2r / 3+2r
                ("meta", np.float64, (P, 12)),
                ("out_a", np.int16, (n_cards, H)), ("out_e", np.int16, (n_cards, 2 * H)),
                # eager delivery: the command number for which card j's share in out_a / out_e is valid
                ("ver_a", np.int64, (n_cards,)), ("ver_e", np.int64, (n_cards,)),
                ("selc", np.int64, (n_cards, TOPK)),   # each card's expert pick (checked to agree)
                ("xb", np.int16, (max(MAX_BLOCK, 1), H)), ("blk", np.int64, (2,)),   # image block rows, (p0, n)
                # image encoder: inputs of one image, every card's share (flat, stride NP*HID), the soft tokens
                ("vin_pix", np.int16, (vnp, 768)), ("vin_oh", np.int16, (vnp, 2 * VIS_POS)),
                ("vin_bias", np.int16, (vnp, vnp)), ("vshare", np.int16, (max(n_cards, 1) * vnp * VIS_HID,)),
                ("vout", np.int16, (max(VIS_LEVELS) if VIS_P else 1, VIS_OUT)),
                ("vx", np.int16, (vnp if VIS_DEPTH else 1, VIS_HID)))
        self.blocks, self.names = [], []
        for i, (name, dtype, shape) in enumerate(spec):
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            if names is None:
                shm = shared_memory.SharedMemory(create=True, size=size)
            else:
                try:
                    shm = shared_memory.SharedMemory(name=names[i], track=False)
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
        n_cards = P // E
        rsplit = os.environ.get("GEMMA4_RSPLIT", "1") == "1"     # the card's engines split the router
        eager = os.environ.get("GEMMA4_EAGER", "0") == "1"       # deliver each share as soon as it exists (measured slower: off)
        shards = []
        for e, k in enumerate(ks):                     # build every engine before uploading anything
            T, qcfg = G.load_base_port("gemma4_port_%d" % k)
            cdir = os.path.join(tdir, "card_%02d" % k)
            shards.append(G.Gemma4Shard(T, qcfg, dev, e, cdir, WDIR, plan, max_ctx, window=engine_window(e, E, cdir),
                                        router_part=(e, E) if rsplit else (0, 1), max_block=MAX_BLOCK))
        with shards[0].T._quiet():
            for s in shards:
                s.prepare()
        H = shards[0].H
        vocab = [plan["cards"][k]["vocab"] for k in ks]
        prog0 = [s.ue.get_program_dram_addr() for s in shards]
        with shards[0].T._quiet():
            n_seg = shards[0].compile(n_cards, vocab[0][1] - vocab[0][0])
            for s, v in zip(shards[1:], vocab[1:]):    # one copy of the shares per card: engine 0's buffer
                n_seg = s.compile(n_cards, v[1] - v[0], parts_addr=shards[0].PARTS,
                                  logits_addr=shards[0].LOGITS + s.router_part[0] * s.NR * BF16 if rsplit else None)
            if MAX_BLOCK:
                for s in shards:
                    s.compile_block()

        def free_mb(s):
            """MB still free in this engine's params / tensor / program regions."""
            u = s.ue
            return ((getattr(u, "_params_limit", u._tensor_dram_base) - u._next_params_dram_addr) / 1e6,
                    (u._program_dram_base - u._tensor_dram_addr) / 1e6,
                    (u._model_end - u._next_program_dram_addr) / 1e6)
        free_lm = [free_mb(s) for s in shards]
        # the image encoder next to the LM on the first VIS_P shards (both engines of a card, or none)
        vshards, vprog, vadd = [], {}, {}
        vcards = VIS_P // E
        if VIS_P and ks[0] < VIS_P:
            import gemma4_vision_tp as GV
            tv = time.time()
            Wv = torch.load(VIS_PATH, weights_only=False, map_location="cpu")
            assert (Wv["dims"]["HID"], Wv["dims"]["POS"], Wv["dims"]["OUT"]) == (VIS_HID, VIS_POS, VIS_OUT)
            vnp = GV.level_np(max(VIS_LEVELS))
            with shards[0].T._quiet():
                for e, (s, k) in enumerate(zip(shards, ks)):
                    v = GV.VisionShard(s.T, s.ue, Wv, k, VIS_P)
                    v.load()
                    v0 = vshards[0] if vshards else None
                    v.buffers(vcards, inputs=v0.inputs if v0 else None, parts=v0.PARTS if v0 else None, np_max=vnp)
                    vshards.append(v)
                del Wv
                for S in VIS_LEVELS:
                    vprog[S] = [v.program(S, VIS_DEPTH) for v in vshards]
                    if E == 2:
                        vadd[S] = vshards[0].add_program(S, vshards[1].B["SHARE"])
            vinfo = {"load_s": time.time() - tv, "heads": [v.heads for v in vshards],
                     "blocks": [len(v.blocks) for v in vshards],
                     "programs": {S: [(p["instructions"], round(p["mb"], 1)) for p in ps] for S, ps in vprog.items()}}
        else:
            vinfo = None
        # engine 0 sums the card's shares on the card: SHARE_x(e0) += SHARE_x(e1); with eager delivery the sum
        # also goes straight into this card's own slot of the parts buffer (no host round trip for it)
        add_prog = {}
        if E == 2:
            s0, s1 = shards
            U, ue0 = s0.U, s0.ue
            for kind, a0, a1, n in (("A", s0.SHARE_A, s1.SHARE_A, H), ("E", s0.SHARE_E, s1.SHARE_E, 2 * H)):
                own = s0.PARTS + c * n * BF16
                ue0.clear_capture_buffer(); U.UnifiedEngine.start_capture(ue0)
                for off in range(0, n, H):
                    ue0.ue_memcpy_from_dram(a0 + off * BF16, H * BF16, U.MEMCPY_TYPE.URAM.value, U.URAM_START_ADDR,
                                            U.URAM_SECTION.URAM_A.value)
                    ue0.ue_memcpy_from_dram(a1 + off * BF16, H * BF16, U.MEMCPY_TYPE.URAM.value, U.URAM_START_ADDR,
                                            U.URAM_SECTION.URAM_B.value)
                    ue0.eltwise_add_core(0x00000, 0x80000, 0x00000, H)
                    ue0.ue_memcpy_to_dram(U.MEMCPY_TYPE.URAM.value, U.URAM_SECTION.URAM_A.value, U.URAM_START_ADDR,
                                          a0 + off * BF16, H * BF16)
                    if eager:
                        ue0.ue_memcpy_to_dram(U.MEMCPY_TYPE.URAM.value, U.URAM_SECTION.URAM_A.value,
                                              U.URAM_START_ADDR, own + off * BF16, H * BF16)
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
                 "kv_mb": sum(3 * d["T"] * d["hd"] * BF16 for d in s.layers if d["nq"]) / 1e6,
                 "setup_s": time.time() - t0, "segs": n_seg,
                 "clock_mhz": 1000.0 / float(getattr(s.ue, "_clock_period_ns", 0) or 5.042),
                 "firmware": "%08x" % s.ue.get_hardware_version(),
                 "free_lm_mb": free_lm[e], "free_mb": free_mb(s), "vision": vinfo,
                 "emb_addr": s.EMB} for e, (s, p0) in enumerate(zip(shards, prog0))]
        sh = Shared(P, H, n_cards, shm_names)
        h2c = os.open(shards[0].T.DMA_DEVICE_H2C, os.O_RDWR)
        c2h = os.open(shards[0].T.DMA_DEVICE_C2H, os.O_RDWR)
        for x in shards:
            x.fast_h2c = h2c

        def rd(addr, mv):
            # pread, NOT preadv/readv: the vectored calls enter xdma's aio path and crash the kernel
            data = os.pread(c2h, len(mv), addr)
            if len(data) != len(mv):
                raise OSError("short c2h read at 0x%x" % addr)
            mv[:] = data

        parts_a = memoryview(sh.out_a).cast("B")        # every card's attention share, card order
        parts_e = memoryview(sh.out_e).cast("B")        # every card's FFN share [dense | MoE], card order
        mine_a = memoryview(sh.out_a[c]).cast("B")
        mine_e = memoryview(sh.out_e[c]).cast("B")
        card_a = [memoryview(sh.out_a[j]).cast("B") for j in range(n_cards)]
        card_e = [memoryview(sh.out_e[j]).cast("B") for j in range(n_cards)]
        PARTS = shards[0].PARTS

        def deliver(ver, views, stride, seq):
            """Publish this card's share, then write every other card's share into this card's parts buffer
            as soon as it is published (the last one to arrive is the only write left on the critical path)."""
            ver[c] = seq
            pending = [j for j in range(n_cards) if j != c]
            t_lim = time.time() + 20.0
            while pending:
                for j in pending:
                    if ver[j] == seq:
                        if pwrite(h2c, views[j], PARTS + j * stride) != stride:
                            raise OSError("short share write")
                        pending.remove(j)
                        break
                else:
                    if (done == FAILED).any() or time.time() > t_lim:
                        raise RuntimeError("share of card(s) %s never arrived" % pending)
        head_i16 = np.zeros(32, dtype=np.int16)
        head_mv = memoryview(head_i16).cast("B")
        prof = os.environ.get("GEMMA4_PROF") == "1"
        U = shards[0].U
        pk32 = struct.Struct("=I").pack
        ctl = []
        for x in shards:
            b = x.ue._base_addr - U.AXI_LITE_TRANSLATION_OFFSET
            ctl.append((x.ue._get_user_fd(), b + U.UE_INSTRUCTION_ADDR, b + U.UE_QUEUE_CTRL_ADDR))
        go = [[pk32(U.ue_35bit_addr_shifter(a) & 0xFFFFFFFF) for a in x.seg_start] for x in shards]
        go_b = [[pk32(U.ue_35bit_addr_shifter(a) & 0xFFFFFFFF) for a in x.bseg_start] for x in shards] if MAX_BLOCK else None
        bkinds = shards[0].bseg_kind if MAX_BLOCK else None
        blayer = shards[0].bseg_layer if MAX_BLOCK else None
        add_go = {kk: pk32(U.ue_35bit_addr_shifter(a) & 0xFFFFFFFF) for kk, a in add_prog.items()}
        pread, pwrite = os.pread, os.pwrite

        def wait_idle(fd, q):
            while pread(fd, 4, q)[1] & 1:              # queue_busy is bit 8 of UE_QUEUE_CTRL
                pass

        unpack_lg = struct.Struct("<%dh" % NE).unpack
        LOGITS = shards[0].LOGITS
        T = shards[0].T
        kinds = shards[0].seg_kind
        conn.send(("ready", info))
        cmd, done, meta = sh.cmd, sh.done, sh.meta
        fine, fine_n = {}, {}
        pick_t = [0, 0, 0]                             # pick ns, count, ns still waiting for D after the pick

        def loop():
            last = 0
            parked = False
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
                    if pick_t[1]:
                        print("PICK|logits read + pick + set_experts %.1f us x %d|then waiting for D %.1f us" % (
                            pick_t[0] / 1e3 / pick_t[1], pick_t[1], pick_t[2] / 1e3 / pick_t[1]), flush=True)
                    break
                if s == SEG_RESET:
                    for x in shards:
                        x.reset_state()
                    for k in ks:
                        done[k] = last
                    continue
                if s == SEG_VIS_BEGIN or s >= VIS:
                    run_vis_cmd(s, pos)
                    for k in ks:
                        done[k] = last
                    continue
                if s in (SEG_BLOCK_BEGIN, SEG_BLOCK_END) or s >= BLK:
                    run_block_cmd(s, pos, last)
                    for k in ks:
                        done[k] = last
                    continue
                run_seg(s, pos, last)
                if kinds[s] == "R":
                    # start the dense FFN (D) right away: it needs no pick, so the pick below overlaps it
                    for (fd, ia, q), g in zip(ctl, go):
                        pwrite(fd, g[s + 1], ia)
                    # the expert pick happens HERE, in this card's process, from this card's own router logits
                    tp0 = time.perf_counter_ns()
                    vals = unpack_lg(pread(c2h, NE * BF16, LOGITS))     # 128 router logits, bf16 bit patterns
                    keys = [v if v >= 0 else -32768 - v for v in vals]   # same order as the float values
                    top = sorted(range(NE), key=keys.__getitem__, reverse=True)[:TOPK]
                    sh.selc[c, :] = top
                    li = (s - 1) // 4
                    lg = [vals[i] for i in top]
                    for x in shards:
                        x.set_experts_bf(li, top, lg)          # E's data slots: D does not touch them
                    pick_t[0] += time.perf_counter_ns() - tp0; pick_t[1] += 1
                    td0 = time.perf_counter_ns()
                    for fd, ia, q in ctl:
                        wait_idle(fd, q)
                    pick_t[2] += time.perf_counter_ns() - td0
                    run_seg(s + 2, -1, last)
                for k in ks:
                    done[k] = last

        vstate = {"S": None}
        vshare_mv = memoryview(sh.vshare).cast("B")

        def run_vis_cmd(s, arg):
            """Image encoder: begin (this image's inputs onto the card, level arg), or segment s - VIS."""
            if not vshards:
                return                                         # this card holds no encoder shard
            if s == SEG_VIS_BEGIN:
                S = int(arg)
                NP = vprog[S][0]["NP"]
                vstate.update(S=S, NP=NP)
                I = vshards[0].inputs
                for name, arr in (("PIXEL", sh.vin_pix[:NP]), ("ONEHOT", sh.vin_oh[:NP]),
                                  ("BIAS", sh.vin_bias[:NP, :NP])):
                    b = np.ascontiguousarray(arr).tobytes()
                    if pwrite(h2c, b, I[name]) != len(b):
                        raise OSError("short vision input write")
                return
            S, NP = vstate["S"], vstate["NP"]
            i = s - VIS
            kind = vprog[S][0]["kinds"][i][0]
            nb = NP * VIS_HID * BF16
            if i > 0:                                          # starts by summing every card's previous share
                if pwrite(h2c, vshare_mv[:vcards * nb], vshards[0].PARTS) != vcards * nb:
                    raise OSError("short vision parts write")
            for (fd, ia, q), v in zip(ctl, vprog[S]):
                pwrite(fd, pk32(U.ue_35bit_addr_shifter(v["start"][i]) & 0xFFFFFFFF), ia)
            for fd, ia, q in ctl:
                wait_idle(fd, q)
            if kind in ("A", "M"):
                if E == 2:
                    fd, ia, q = ctl[0]
                    pwrite(fd, pk32(U.ue_35bit_addr_shifter(vadd[S]) & 0xFFFFFFFF), ia)
                    wait_idle(fd, q)
                rd(vshards[0].B["SHARE"], vshare_mv[c * nb:(c + 1) * nb])
            elif vshards[0].k == 0:                            # P: the soft tokens (or X, unit tests)
                if VIS_DEPTH:
                    rd(vshards[0].B["X"], memoryview(sh.vx).cast("B")[:nb])
                else:
                    ob = S * VIS_OUT * BF16
                    rd(vshards[0].B["OUT"], memoryview(sh.vout).cast("B")[:ob])

        def run_block_cmd(s, t, seq):
            """Image block: begin (embeddings -> XS), one block segment for token t (R also runs D and E,
            with the expert pick in between, as in the normal loop), end (mask rows back to causal)."""
            assert MAX_BLOCK and not eager, "image blocks need GEMMA4_MAX_BLOCK and no eager delivery"
            if s == SEG_BLOCK_BEGIN:
                p0, n = int(sh.blk[0]), int(sh.blk[1])
                xb = sh.xb[:n].tobytes()
                for x in shards:
                    x.block_begin(p0, n, xb)
                return
            if s == SEG_BLOCK_END:
                for x in shards:
                    x.block_end()
                return
            bs = s - BLK
            run_bseg(bs, t)
            if bkinds[bs] == "R":
                for (fd, ia, q), g in zip(ctl, go_b):          # D right away; the pick overlaps it
                    pwrite(fd, g[bs + 1], ia)
                vals = unpack_lg(pread(c2h, NE * BF16, LOGITS))
                keys = [v if v >= 0 else -32768 - v for v in vals]
                top = sorted(range(NE), key=keys.__getitem__, reverse=True)[:TOPK]
                sh.selc[c, :] = top
                lg = [vals[i] for i in top]
                for x in shards:
                    x.set_experts_bf(blayer[bs], top, lg, block=True)
                for fd, ia, q in ctl:
                    wait_idle(fd, q)
                run_bseg(bs + 2, t)

        def run_bseg(bs, t):
            kind = bkinds[bs]
            if kind in ("K", "B"):
                for x in shards:
                    x.block_slots(bs, t)                       # token t's registers
                    if kind == "B":
                        x.block_mask(bs, t)                    # and its mask row
            elif kind == "R":
                if pwrite(h2c, parts_a, shards[0].PARTS) != len(parts_a):
                    raise OSError("short parts write")
            elif kind == "F":
                if pwrite(h2c, parts_e, shards[0].PARTS) != len(parts_e):
                    raise OSError("short parts write")
            for (fd, ia, q), g in zip(ctl, go_b):
                pwrite(fd, g[bs], ia)
            for fd, ia, q in ctl:
                wait_idle(fd, q)
            if kind in ("B", "E"):                             # this segment produced a share
                if E == 2:
                    fd, ia, q = ctl[0]
                    pwrite(fd, add_go["A" if kind == "B" else "E"], ia)
                    wait_idle(fd, q)
                if kind == "B":
                    rd(shards[0].SHARE_A, mine_a)
                else:
                    rd(shards[0].SHARE_E, mine_e)

        def run_seg(s, pos, seq):
            t1 = time.perf_counter()
            kind = kinds[s]
            if s == 0:
                xb = sh.x.tobytes()
                for x in shards:
                    x.set_position(pos, xb)            # embedding row + masks + registers, one DMA
            elif eager:
                pass                                   # the shares are already in the parts buffer
            elif kind == "R":                          # starts by summing the attention shares
                if pwrite(h2c, parts_a, shards[0].PARTS) != len(parts_a):
                    raise OSError("short parts write")
            elif kind in ("A", "H"):                   # starts by finishing the previous layer's FFN
                if pwrite(h2c, parts_e, shards[0].PARTS) != len(parts_e):
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
            if kind in ("A", "E"):                     # this segment produced a share
                if E == 2:
                    fd, ia, q = ctl[0]
                    pwrite(fd, add_go[kind], ia)
                    wait_idle(fd, q)
                if kind == "A":
                    rd(shards[0].SHARE_A, mine_a)
                    if eager:
                        deliver(sh.ver_a, card_a, H * BF16, seq)
                else:
                    rd(shards[0].SHARE_E, mine_e)
                    if eager:
                        deliver(sh.ver_e, card_e, 2 * H * BF16, seq)
            elif kind == "H":                          # the shard's best `want` tokens (the card tracks 4)
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
    ap.add_argument("--image-tokens", default=None,
                    help="soft tokens of one image [n, H] (bf16 .pt, from the card's image encoder); the image goes "
                         "before the prompt text")
    ap.add_argument("--image", default=None, help="an image file: encoded on the cards (GEMMA4_VISION_SHARDS)")
    ap.add_argument("--soft", type=int, default=70, help="soft tokens of the image (a GEMMA4_VISION_LEVELS value)")
    a = ap.parse_args()
    ctypes.CDLL(None).prctl(29, 1, 0, 0, 0)
    tdir = os.path.join(WORK, "tp%s" % a.tp)
    plan = json.load(open(os.path.join(tdir, "plan.json")))
    P, H = plan["P"], plan["hidden"]
    E = a.engines
    assert E == 2, "the shares are summed per card (2 engines)"
    n_cards = P // E
    host_cpu, card_cpus = cpu_plan(n_cards)
    os.sched_setaffinity(0, {host_cpu})
    devs = cards_in_pci_order()[:n_cards]
    voff = [c["vocab"][0] for c in plan["cards"]]
    sh = Shared(P, H, n_cards)
    ctx = mp.get_context("spawn")
    pipes, procs = [], []
    t0 = time.time()
    for c, dev in enumerate(devs):
        pa, pb = ctx.Pipe()
        p = ctx.Process(target=worker, args=(c, E, P, dev, tdir, a.max_context, card_cpus[c], sh.names, pb), daemon=True)
        p.start(); pipes.append(pa); procs.append(p)
    n_seg = None
    emb_addr = [None] * P
    for c, pc in enumerate(pipes):
        m = pc.recv()
        if m[0] != "ready":
            print("ERROR|card %d|%s" % (c, m[1])); return 1
        for e, i in enumerate(m[1]):
            n_seg = i["segs"]
            emb_addr[c * E + e] = i["emb_addr"]
            print("CARD|%s|weights %.0f MB|kv %.0f MB|program %.1f MB|setup %.0f s" % (
                i["dev"], i["params_mb"], i["kv_mb"], i["prog_mb"], i["setup_s"]), flush=True)
            print("FREE|%s|params / tensors / programs MB: after the LM %s|now %s" % (
                i["dev"], "/".join("%.0f" % v for v in i["free_lm_mb"]), "/".join("%.0f" % v for v in i["free_mb"])),
                flush=True)
            if i["vision"] and e == 0:
                print("VISION|card %d|heads %s|mlp blocks %s|load %.0f s|programs (instructions, MB) %s" % (
                    c, i["vision"]["heads"], i["vision"]["blocks"], i["vision"]["load_s"], i["vision"]["programs"]),
                    flush=True)
    print("READY|%d cards x %d engines|%d segments/token|%.0f s" % (n_cards, E, n_seg, time.time() - t0), flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    cfg_all = json.load(open(os.path.join(MODEL, "config.json")))
    stops = set(cfg_all.get("eos_token_id", [])) | {tok.convert_tokens_to_ids("<turn|>")}    # <eos>, <turn|>
    embed = CardEmbed(plan, devs, E, emb_addr, H)    # the table is on the cards; one row per token comes over
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

    def encode_image(path, S):
        """The image on the cards: the host resizes and cuts it into patches (the model's image processor) and
        writes the inputs; the encoder runs as 2 segments per layer over the cards. Returns the soft tokens."""
        import gemma4_vision_tp as GV
        from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor
        from PIL import Image
        assert VIS_P and S in VIS_LEVELS, "GEMMA4_VISION_SHARDS / GEMMA4_VISION_LEVELS"
        t0 = time.time()
        inp = Gemma4ImageProcessor(max_soft_tokens=S)(images=[Image.open(path).convert("RGB")], return_tensors="pt")
        pix, oh, bias, n_soft = GV.level_inputs(inp["pixel_values"], inp["image_position_ids"], S, VIS_POS)
        NP = pix.shape[0]
        sh.vin_pix[:NP] = pix.view(torch.int16).numpy()
        sh.vin_oh[:NP] = oh.view(torch.int16).numpy()
        sh.vin_bias[:NP, :NP] = bias.view(torch.int16).numpy()
        t1 = time.time()
        command(SEG_VIS_BEGIN, S)
        t2 = time.time()
        depth = VIS_DEPTH or 27
        for i in range(2 * depth + 1):
            command(VIS + i, -1)
        t3 = time.time()
        print("VISION|%d soft tokens (level %d, %d patch rows)|host prep %.2f s|inputs to cards %.2f s|encoder %.2f s"
              % (n_soft, S, NP, t1 - t0, t2 - t1, t3 - t2), flush=True)
        if VIS_DEPTH:
            torch.save({"x": torch.from_numpy(sh.vx[:NP].copy()).view(torch.bfloat16), "S": S, "depth": VIS_DEPTH,
                        "image": path}, "/tmp/gemma4_vision_tp_x.pt")
            return None
        out = sh.vout[:n_soft].copy()
        torch.save(torch.from_numpy(out).view(torch.bfloat16), "/tmp/gemma4_vision_tp_out.pt")
        return out

    img = None
    if a.image and os.environ.get("GEMMA4_VISION_BENCH"):  # every level twice, then quit
        for S in VIS_LEVELS:
            for _ in range(2):
                encode_image(a.image, S)
        sh.cmd[1] = SEG_QUIT; sh.cmd[0] = seq[0] + 1
        for p in procs:
            p.join(10)
        return 0
    if a.image:
        img = encode_image(a.image, a.soft)
        if VIS_DEPTH:
            sh.cmd[1] = SEG_QUIT; sh.cmd[0] = seq[0] + 1
            for p in procs:
                p.join(10)
            return 0
    elif a.image_tokens:
        img = torch.load(a.image_tokens).to(torch.bfloat16).view(torch.int16).numpy()
    if img is not None:
        assert img.shape[1] == H and img.shape[0] <= MAX_BLOCK, "set GEMMA4_MAX_BLOCK >= the image's soft tokens"
        content = [{"type": "image"}, {"type": "text", "text": a.prompt}]
    else:
        content = a.prompt
    text = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    IMG = tok.convert_tokens_to_ids("<|image|>")
    if img is not None:                                # the image: <|image> + one <|image|> per soft token + <image|>
        assert "<|image|>" in text, "the chat template left no image placeholder"
        text = text.replace("<|image|>", "<|image>" + "<|image|>" * img.shape[0] + "<image|>", 1)
    ids = tok.encode(text, add_special_tokens=False)

    seg_t, seg_hw, seg_wk = np.zeros(4), np.zeros(4), np.zeros(4)
    kinds = "ARDE"
    disagree = [0]

    def token_step(t, pos, measure):
        sh.x[:] = embed[t]
        for s in range(n_seg):
            if s < n_seg - 1 and s % 4 >= 2:           # D and E run inside the R command (each card picks locally)
                continue
            t_s = time.perf_counter()
            command(s, pos if s == 0 else -1)
            if s < n_seg - 1 and s % 4 == 1 and (sh.selc != sh.selc[0]).any():
                disagree[0] += 1
            if measure:
                ki = "ARDH".index("H" if s == n_seg - 1 else kinds[s % 4])
                seg_t[ki] += time.perf_counter() - t_s
                seg_hw[ki] += sh.meta[:, 0].max() / 1e3
                seg_wk[ki] += sh.meta[:, 1].max() / 1e3
        k = int(np.argmax(sh.meta[:, 3]))
        return voff[k] + int(sh.meta[k, 2])

    def block_step(rows, p0):
        """An image block at positions p0..: layer by layer through the block program (see compile_block)."""
        n = rows.shape[0]
        sh.xb[:n] = rows
        sh.blk[0], sh.blk[1] = p0, n
        command(SEG_BLOCK_BEGIN, -1)
        bk = [(k, i) for k, i in zip(*block_layout)]
        seg = {ki: s for s, ki in enumerate(bk)}
        for t in range(n):
            command(BLK + seg[("K", -1)], t)
        for i in range(n_layers):
            for t in range(n):
                command(BLK + seg[("B", i)], t)
                command(BLK + seg[("R", i)], t)                # D and E run inside it
                if (sh.selc != sh.selc[0]).any():
                    disagree[0] += 1
                command(BLK + seg[("F", i)], t)
        command(SEG_BLOCK_END, -1)

    n_layers = cfg_all["text_config"]["num_hidden_layers"] if "text_config" in cfg_all else cfg_all["num_hidden_layers"]
    block_layout = (["K"] + [k for _ in range(n_layers) for k in "BRDEF"],
                    [-1] + [i for i in range(n_layers) for _ in "BRDEF"])
    command(SEG_RESET, -1)
    print("PROMPT|%d tokens%s" % (len(ids), " (%d image tokens)" % img.shape[0] if img is not None else ""), flush=True)
    t1 = time.time()
    nxt = None
    pos = 0
    while pos < len(ids):
        if img is not None and ids[pos] == IMG:
            assert ids[pos:pos + img.shape[0]] == [IMG] * img.shape[0]
            tb = time.time()
            block_step(img, pos)
            print("IMAGE|%d tokens at %d|%.1f s" % (img.shape[0], pos, time.time() - tb), flush=True)
            pos += img.shape[0]
            continue
        nxt = token_step(ids[pos], pos, False)
        pos += 1
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
        print("HOST|per token: attention %.1f ms|router+FFN %.1f ms|(FFN alone) %.1f ms|head %.1f ms" %
              tuple(1e3 * v / nd for v in seg_t), flush=True)
    if nd and os.environ.get("GEMMA4_PROF") == "1":
        print("CARD|slowest engine compute: attention %.1f ms|router %.1f ms|FFN %.1f ms|head %.1f ms" %
              tuple(1e3 * v / nd for v in seg_hw), flush=True)
    print("PICK|layers where the cards' expert picks differed: %d" % disagree[0], flush=True)
    print("TOKENS|%s" % out[:24], flush=True)
    print("TEXT_BEGIN\n%s\nTEXT_END" % tok.decode(out), flush=True)
    sh.cmd[1] = SEG_QUIT; sh.cmd[0] = seq[0] + 1
    for p in procs:
        p.join(10)
    return 0


if __name__ == "__main__":
    sys.exit(main())
