"""Run a Qwen3.5-family model tensor-parallel: every card holds a slice of every layer and all cards work
on the same token at the same time (all model compute on the cards).

Per layer there are two meeting points. Each card computes its heads' share of the attention
output (segment A) and its neurons' share of the MLP output (segment B); the host adds the
shares to the residual stream and hands the new vector to all cards. The LM head is split by
vocabulary rows; every card reports its best token and the host keeps the best of those.

There are 2 x layers + 1 meeting points per token, so their fixed cost matters: commands and results travel through
shared memory (one counter bump wakes all cards, no per-card pipe message), each card process
sleeps through its segment's known run time with a tight timer, and one small program per token
sets both position registers.

    Q_MODEL=q35_9b python run_tp.py --tp 11v2 --prompt "..." --max-new-tokens 64
"""
import argparse
import ctypes
import glob
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from multiprocessing import shared_memory

import numpy as np
import torch

BASE = os.path.join(os.path.expanduser(os.environ.get("APEX_TP_WORK", "~/apex_tp")), "models", os.environ.get("Q_MODEL", "q35_9b"))
PORT = os.path.join(BASE, "qwen3.5_2b_test.py")
WDIR = os.path.join(BASE, "weights")
LOGDIR = os.path.join(BASE, "logs")
MODEL = os.path.dirname(sorted(glob.glob(os.path.join(BASE, "*_bin", "*", "config.json")))[0])
BF16 = 2
INST = 32
SEG_QUIT, SEG_RESET = -1, -2
FAILED = -1


def cards_in_pci_order():
    out = []
    for u in glob.glob("/sys/class/xdma/xdma*_user"):
        dev = os.path.basename(u)[:-len("_user")]
        out.append((os.path.basename(os.path.realpath(os.path.join(u, "device"))), dev))
    return [d for _, d in sorted(out)]


def load_port(name):
    spec = importlib.util.spec_from_file_location(name, PORT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Shared:
    """Command block written by the host, one result row per card."""

    def __init__(self, P, H, names=None):
        spec = (("cmd", np.int64, (4,)),            # sequence number, segment, position (-1: none), unused
                ("x", np.int16, (H,)),              # input vector for the segment (bf16 bits)
                ("done", np.int64, (P,)),           # sequence number each card finished last (FAILED on error)
                ("meta", np.float64, (P, 4)),       # card ms, round-trip ms, best logit, best index
                ("out", np.int16, (P, H)))          # each card's partial output (bf16 bits)
        self.blocks, self.names = [], []
        for i, (name, dtype, shape) in enumerate(spec):
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            if names is None:
                shm = shared_memory.SharedMemory(create=True, size=size)
            else:
                try:
                    shm = shared_memory.SharedMemory(name=names[i], track=False)   # the host process owns it
                except TypeError:
                    shm = shared_memory.SharedMemory(name=names[i])
            self.blocks.append(shm)
            self.names.append(shm.name)
            arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            if names is None:
                arr[...] = 0
            setattr(self, name, arr)

    def close(self, unlink=False):
        for name in ("cmd", "x", "done", "meta", "out"):
            delattr(self, name)
        for shm in self.blocks:
            shm.close()
            if unlink:
                shm.unlink()


class LazyLayers:
    """The card's layer slices, read from disk one at a time (a 27B card holds over 1 GB of them)."""

    def __init__(self, cdir, n):
        self.cdir, self.n = cdir, n

    def __len__(self):
        return self.n

    def __getitem__(self, g):
        if isinstance(g, slice):
            return [self[i] for i in range(*g.indices(self.n))]
        if g < 0:
            g += self.n
        if not 0 <= g < self.n:
            raise IndexError(g)
        return torch.load(os.path.join(self.cdir, "layer_%02d.pt" % g), weights_only=False, map_location="cpu")


ENGINE_STRIDE = 0x10000          # AXI-Lite register block of engine e: UE_0_BASE_ADDR + e * ENGINE_STRIDE
CHANNEL_BYTES = 0x8000_0000      # Kintex-7: address bit 31 picks the DDR3 channel (2 GiB each)
PROGRAM_BYTES = 64 << 20         # per engine, at the top of its window


def engine_window(e, E, cdir):
    """DRAM layout of engine e of E on one card. With two engines, engine e lives entirely in DDR
    channel e, so the two engines read their weights at the same time without sharing a channel
    (measured: 5.7 GB/s alone, 11.4 GB/s for both on different channels, 6.4 on the same one)."""
    if E == 1:
        return {}                                  # single engine: the port's membase layout
    w = e * CHANNEL_BYTES
    on_disk = sum(os.path.getsize(f) for f in glob.glob(os.path.join(cdir, "*.pt")))
    params = (int(on_disk * 1.03) + (96 << 20) + (16 << 20) - 1) // (16 << 20) * (16 << 20)
    prog = w + CHANNEL_BYTES - PROGRAM_BYTES
    assert params < CHANNEL_BYTES - PROGRAM_BYTES - (256 << 20), "the slice does not fit one DDR channel"
    return {"params_dram_base": w, "tensor_dram_base": w + params, "program_dram_base": prog}


def worker(c, E, P, dev, tdir, n_layers, max_ctx, cpu, nap, shm_names, conn, card_sum=True):
    """One process per card. It drives the card's E engines; engine e of card c is TP shard k = c*E + e."""
    os.makedirs(LOGDIR, exist_ok=True)
    log = open(os.path.join(LOGDIR, "card_%02d.log" % c), "w", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sys.stdout = sys.stderr = log
    sh = None
    ks = [c * E + e for e in range(E)]
    # presum: after each segment engine 0 adds engine 1's share into its own on the card, so the host
    # reads one share per card and writes one per card back (half the PCIe traffic of a share per engine)
    presum = card_sum and E == 2 and os.environ.get("APEX_TP_PRESUM", "1") != "0"
    try:
        if cpu is not None:
            os.sched_setaffinity(0, {cpu})
        ctypes.CDLL(None).prctl(29, 1, 0, 0, 0)     # PR_SET_TIMERSLACK = 1 ns: short sleeps end on time
        torch.set_num_threads(1)
        t0 = time.time()
        mods, ues, cdirs = [], [], []
        # build every engine before any weights go up: constructing one self-tests the first 16 KB of DRAM
        for e, k in enumerate(ks):
            T = load_port("q35_tp_port_%d" % k)    # one port module per engine: no shared module-level caches
            T.set_dma_device(dev)
            T.DMA_DEVICE_H2C = T.user_dma_core.DMA_DEVICE_H2C
            T.DMA_DEVICE_C2H = T.user_dma_core.DMA_DEVICE_C2H
            T.user_dma_core.configure_clock_from_hardware()
            cdir = os.path.join(tdir, "card_%02d" % k)
            ue = T.Qwen3_5_2b_UnifiedEngine(device="cpu", config_path=os.path.join(cdir, "config.json"),
                                            BASE_ADDR=T.user_dma_core.UE_0_BASE_ADDR + e * ENGINE_STRIDE,
                                            **engine_window(e, E, cdir))
            mods.append(T); ues.append(ue); cdirs.append(cdir)
        T = mods[0]
        cfg = json.load(open(os.path.join(cdirs[0], "config.json")))
        H = cfg["file_info"]["hidden_size"]
        info, eng = [], []
        for T_e, ue, cdir, k in zip(mods, ues, cdirs, ks):
            vocab = json.load(open(os.path.join(cdir, "config.json")))["file_info"]["embedding_vocab"]
            weights = {
                "layers": LazyLayers(cdir, n_layers),
                "final_norm_gamma": torch.load(os.path.join(WDIR, "final_norm.pt"), weights_only=False),
                "embed_weight": torch.zeros(vocab, 1, dtype=torch.bfloat16),   # only its row count is used here
            }
            weights["lm_head_scale"], weights["lm_head_data"] = torch.load(os.path.join(cdir, "lm_head.pt"),
                                                                           weights_only=False)
            ue._tp_mode = True
            ue._tp_segments = []
            ue._tp_card_sum = card_sum                # residual kept and summed on the card (cardsum patch)
            ue._tp_cards = P // E if presum else P    # presum: one combined share per card reaches the host
            ue.software_reset()
            ue._preallocate_identity_matrix()
            ue.prepare_inference(weights, max_context=max_ctx)
            del weights
        if card_sum and E > 1:
            # one copy of the shards' partial outputs per card: every engine's program reads engine 0's
            # buffer (one flat address space), so the host writes it once per card instead of per engine
            for ue in ues[1:]:
                ue._tp_parts_dram = ues[0]._tp_parts_dram
        for T_e, ue, cdir, k in zip(mods, ues, cdirs, ks):
            ue._decoder_bin_path = os.path.join(cdir, "decoder.bin")
            ue._decoder_meta_path = os.path.join(cdir, "decoder.json")
            for p in (ue._decoder_bin_path, ue._decoder_meta_path):
                if os.path.exists(p):
                    os.remove(p)
            bin_path, sizes, _ = ue.compile_decoder()
            ue.load_instructions(bin_path)
            segs = list(ue._tp_segments)
            assert len(segs) == 2 * n_layers, "expected two cuts per layer, got %d" % len(segs)
            g = {"T": T_e, "ue": ue, "k": k,
                 "starts": [ue._decoder_prog_addr] + [ue._decoder_prog_addr + idx * INST for idx, _ in segs],
                 "outs": [addr for _, addr in segs],  # segment s < 2L writes H values here; the last one is the head
                 "pos_prog": ue.get_program_dram_addr()}
            ue.allocate_program_dram(8 * INST)       # fixed slot for the per-token position program
            g["est"] = [0.0] * len(g["starts"])      # last measured run time of each segment, microseconds
            eng.append(g)
            info.append({"dev": "%s/e%d" % (dev, k - c * E),
                         "params_mb": (ue._next_params_dram_addr - ue._params_dram_base) / 1e6,
                         "prog_mb": sizes[0] / 1e6, "setup_s": time.time() - t0, "segs": len(g["starts"])})
        n_seg = len(eng[0]["starts"])
        T_al = ues[0].max_context_aligned
        row_byt = ues[0].full_head_dim * BF16
        rope_byt = ues[0].full_rotary_dim * BF16
        add_progs = []
        if presum:
            # per segment, a tiny engine-0 program: share(engine 0) += share(engine 1), in URAM
            U, ue0 = mods[0].user_dma_core, ues[0]
            nbytes = H * BF16
            for s in range(n_seg - 1):
                a0, a1 = eng[0]["outs"][s], eng[1]["outs"][s]
                ue0.clear_capture_buffer()
                U.UnifiedEngine.start_capture(ue0)
                ue0.ue_memcpy_from_dram(a0, nbytes, U.MEMCPY_TYPE.URAM.value, U.URAM_START_ADDR, U.URAM_SECTION.URAM_A.value)
                ue0.ue_memcpy_from_dram(a1, nbytes, U.MEMCPY_TYPE.URAM.value, U.URAM_START_ADDR, U.URAM_SECTION.URAM_B.value)
                ue0.eltwise_add_core(0x00000, 0x80000, 0x00000, H)
                ue0.ue_memcpy_to_dram(U.MEMCPY_TYPE.URAM.value, U.URAM_SECTION.URAM_A.value, U.URAM_START_ADDR, a0, nbytes)
                ue0.generate_instruction_halt()
                U.UnifiedEngine.stop_capture(ue0)
                addr = ue0.get_program_dram_addr()
                ue0.write_captured_instructions_to_dram(addr)
                ue0.allocate_program_dram(ue0.get_capture_instruction_size_bytes() + 64)
                ue0.clear_capture_buffer()
                add_progs.append(addr)

        sh = Shared(P, H, shm_names)
        x_t = torch.from_numpy(sh.x).view(torch.bfloat16)
        n_parts = P // E if presum else P
        parts_t = torch.from_numpy(sh.out[:n_parts]).view(torch.bfloat16)   # the shares the cards add, [n_parts, H]
        # presum double-buffers the combined shares by segment parity (P = 2 x cards rows): a fast card
        # writing segment s's share can never overwrite a row a slow card is still uploading for segment s
        parity_t = [torch.from_numpy(sh.out[i * n_parts:(i + 1) * n_parts]).view(torch.bfloat16)
                    for i in range(2 if presum else 1)]
        parity_out = [torch.from_numpy(sh.out[i * n_parts + c]).view(torch.bfloat16) for i in range(len(parity_t))]
        bias = torch.empty((1, T_al), dtype=torch.bfloat16)
        for g in eng:
            g["out_t"] = torch.from_numpy(sh.out[g["k"]]).view(torch.bfloat16)

        def run_all(addr_of, expect_us, limit_s):
            for g in eng:                             # start every engine, then wait for all of them
                g["ue"].start_execute_from_dram(addr_of(g))
            t_wait = time.time()
            if nap and expect_us > 2 * nap:
                time.sleep((expect_us - nap) * 1e-6)
            for g in eng:
                while g["ue"].is_queue_busy():
                    os.sched_yield()
                    if time.time() - t_wait >= limit_s:
                        raise RuntimeError("engine program at 0x%x did not finish within %.0f s" % (addr_of(g), limit_s))

        def write_position(g, pos):
            T_e, ue = g["T"], g["ue"]
            ue.dma_write(T_e.DMA_DEVICE_H2C, ue._bias_dram, bias, T_al * BF16)
            ue.reset_tensor_dram_addr()
            ue.clear_capture_buffer()
            T_e.UnifiedEngine.start_capture(ue)
            ue.generate_instruction_add_set(ue._ISA_POS_K_REG, T_e.ue_35bit_addr_shifter(pos * row_byt))
            ue.generate_instruction_add_set(ue._ISA_POS_ROPE_REG, T_e.ue_35bit_addr_shifter(pos * rope_byt))
            if hasattr(ue, "_ISA_ATTN_LEN_REG"):      # dynlen: attend over the filled cache only
                ue.generate_instruction_add_set(ue._ISA_ATTN_LEN_REG, min(T_al, (pos + 64) // 64 * 64))
            if hasattr(ue, "_ISA_BLK_V_REG"):         # ctx16k: the 64-row V block that holds pos, and its V^T columns
                blk, ch = pos // 64 * 64, ue._vt_chunk
                ue.generate_instruction_add_set(ue._ISA_BLK_V_REG, T_e.ue_35bit_addr_shifter(blk * row_byt))
                ue.generate_instruction_add_set(ue._ISA_BLK_VT_REG, T_e.ue_35bit_addr_shifter(
                    (blk // ch) * ue.full_head_dim * ch * BF16 + (blk % ch) * BF16))
            ue.generate_instruction_halt()
            T_e.UnifiedEngine.stop_capture(ue)
            ue.write_captured_instructions_to_dram(g["pos_prog"])
            ue.clear_capture_buffer()

        conn.send(("ready", info))
        last = 0
        cmd, done, meta = sh.cmd, sh.done, sh.meta
        with T._quiet():
            while True:
                while cmd[0] == last:
                    os.sched_yield()
                last = int(cmd[0]); s = int(cmd[1]); pos = int(cmd[2])
                if s == SEG_QUIT:
                    break
                if s == SEG_RESET:
                    for g in eng:
                        g["ue"].reset_state()
                    for k in ks:
                        done[k] = last
                    continue
                t1 = time.perf_counter()
                if pos >= 0:                          # first segment of a token
                    bias.fill_(float("-inf"))
                    bias[:, :pos + 1] = 0.0
                    for g in eng:
                        write_position(g, pos)
                    run_all(lambda g: g["pos_prog"], 0.0, 5.0)
                for i, g in enumerate(eng):
                    T_e, ue = g["T"], g["ue"]
                    if pos >= 0 or not card_sum:      # token start: embedding row (host-sum mode: the summed stream)
                        ue.dma_write(T_e.DMA_DEVICE_H2C, ue._decoder_X_dram, x_t, H * BF16)
                    elif i == 0:                      # cardsum: every share, once per card; the engines add them
                        src = parity_t[(s - 1) % 2] if presum else parts_t
                        ue.dma_write(T_e.DMA_DEVICE_H2C, ue._tp_parts_dram, src, n_parts * H * BF16)
                run_all(lambda g: g["starts"][s], max(g["est"][s] for g in eng), 60.0)
                hw_seg = [g["ue"].report_latency_in_us() for g in eng]
                if presum and s < n_seg - 1:
                    ue0 = eng[0]["ue"]
                    ue0.start_execute_from_dram(add_progs[s])
                    while ue0.is_queue_busy():
                        pass
                    # the card's combined share goes to its row of this segment's parity block
                    ue0.dma_read(eng[0]["T"].DMA_DEVICE_C2H, eng[0]["outs"][s], parity_out[s % 2], H * BF16)
                for g, hw_us in zip(eng, hw_seg):
                    T_e, ue, k = g["T"], g["ue"], g["k"]
                    g["est"][s] = hw_us
                    if s < n_seg - 1:
                        if not presum:
                            ue.dma_read(T_e.DMA_DEVICE_C2H, g["outs"][s], g["out_t"], H * BF16)
                    else:
                        idx = int(ue.get_arg_max_index(rank=1))
                        blk = ue.dma_from_accelerator_memory(ue.LOGITS_DRAM + (idx // 32) * 32 * BF16, torch.Size([1, 32]))
                        meta[k, 2] = float(blk.float()[0, idx % 32])
                        meta[k, 3] = idx
                    meta[k, 0] = hw_us / 1e3
                    meta[k, 1] = (time.perf_counter() - t1) * 1e3
                for k in ks:
                    done[k] = last
    except Exception:
        traceback.print_exc()
        try:
            if sh is not None:
                for k in ks:
                    sh.done[k] = FAILED
            conn.send(("error", traceback.format_exc()[-1500:]))
        except Exception:
            pass


def cpu_plan(P):
    """CPU of the host loop and of each card process. APEX_TP_CPUS="8,9,10,..." (host first) keeps them
    off the CPUs the xdma driver pins its completion threads to in poll mode (CPUs 0-7 on the Z420)."""
    env = os.environ.get("APEX_TP_CPUS", "")
    if env:
        c = [int(v) for v in env.split(",")]
        return c[0], [c[1 + k % (len(c) - 1)] for k in range(P)]
    n = os.cpu_count()
    return 0, [1 + k % (n - 1) for k in range(P)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", default="11v2", help="sliced weight set to use: <model dir>/tp<NAME>")
    ap.add_argument("--devs", default="", help="device names in card order (default: all cards in PCI order)")
    ap.add_argument("--prompt", default="Tell me about the Eiffel Tower. What year was it built?")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-context", type=int, default=512)
    ap.add_argument("--think", action="store_true", help="leave the model's thinking mode on (off by default)")
    ap.add_argument("--nap-us", type=float, default=60.0,
                    help="sleep until this many microseconds before a segment's expected end, then poll (0 = always poll)")
    ap.add_argument("--host-nap-us", type=float, default=0.0,
                    help="host: sleep until this long before the cards are expected back (0 = poll on its own CPU)")
    ap.add_argument("--engines", type=int, default=1,
                    help="engines used per card; each engine is one TP shard in its own DDR channel")
    ap.add_argument("--eval-file", default="", help="teacher-forced top-1 accuracy on this text instead of chatting")
    ap.add_argument("--eval-tokens", type=int, default=1000)
    ap.add_argument("--host-sum", action="store_true",
                    help="reference only: add the cards' shares to the residual on the host (default: on the cards)")
    a = ap.parse_args()
    host_nap = a.host_nap_us
    ctypes.CDLL(None).prctl(29, 1, 0, 0, 0)
    tdir = os.path.join(BASE, "tp%s" % a.tp)
    plan = json.load(open(os.path.join(tdir, "plan.json")))
    P, L, H = plan["P"], plan["layers"], plan["hidden"]       # P = TP shards = cards x engines per card
    E = a.engines
    assert P % E == 0, "%d shards do not split into %d engines per card" % (P, E)
    n_cards = P // E
    host_cpu, card_cpus = cpu_plan(n_cards)
    if not host_nap:
        os.sched_setaffinity(0, {host_cpu})
    devs = a.devs.split(",") if a.devs else cards_in_pci_order()[:n_cards]
    assert len(devs) == n_cards, "need %d cards, have %d" % (n_cards, len(devs))
    voff = [c["vocab"][0] for c in plan["cards"]]
    ncpu = os.cpu_count()

    sh = Shared(P, H)
    ctx = mp.get_context("spawn")
    pipes, procs = [], []
    t0 = time.time()
    for c, dev in enumerate(devs):
        pa, pb = ctx.Pipe()
        p = ctx.Process(target=worker, args=(c, E, P, dev, tdir, L, a.max_context,
                                             (c % ncpu) if host_nap else card_cpus[c], a.nap_us, sh.names, pb,
                                             not a.host_sum), daemon=True)
        p.start()
        pipes.append(pa); procs.append(p)
    n_seg = None
    shard_dev = [None] * P
    for c, pc in enumerate(pipes):
        m = pc.recv()
        if m[0] != "ready":
            print("ERROR|card %d|%s" % (c, m[1] if len(m) > 1 else m))
            return 1
        for e, i in enumerate(m[1]):
            k = c * E + e
            n_seg = i["segs"]
            shard_dev[k] = i["dev"]
            print("CARD|%d|%s|weights %.0f MB|program %.1f MB|setup %.0f s" % (
                k, i["dev"], i["params_mb"], i["prog_mb"], i["setup_s"]), flush=True)
    print("READY|%d cards x %d engines|%d segments/token|%.0f s" % (n_cards, E, n_seg, time.time() - t0), flush=True)
    pipes = [pipes[k // E] for k in range(P)]                   # error reports are looked up per shard below

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    text_in = tok.apply_chat_template([{"role": "user", "content": a.prompt}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=a.think)
    ids = [int(v) for v in tok.encode(text_in, add_special_tokens=True)]
    eot = tok.convert_tokens_to_ids("<|im_end|>")
    shape = json.load(open(os.path.join(WDIR, "embed.json")))["shape"]
    # the whole table in RAM: a memmap row costs a random disk read (~28 ms on this SSD) once the page cache drops it
    embed = np.fromfile(os.path.join(WDIR, "embed.bin"), dtype=np.int16).reshape(tuple(shape))
    assert len(ids) + a.max_new_tokens <= a.max_context

    seq = [0]
    back_us = [0.0] * n_seg       # how long the cards took to answer each segment last time

    def command(s, pos, limit_s=120.0):
        sh.cmd[1] = s
        sh.cmd[2] = pos
        seq[0] += 1
        sh.cmd[0] = seq[0]                    # the bump is what the cards watch for
        t = time.perf_counter()
        if host_nap and s >= 0 and back_us[s] > 2 * host_nap:
            time.sleep((back_us[s] - host_nap) * 1e-6)
        n = 0
        while not (sh.done == seq[0]).all():
            n += 1
            if n & 0x3FF == 0:
                if (sh.done == FAILED).any():
                    k = int(np.argmax(sh.done == FAILED))
                    m = pipes[k].recv() if pipes[k].poll(2.0) else ("error", "no detail")
                    raise RuntimeError("card %d: %s" % (k, m[1]))
                if time.perf_counter() - t > limit_s:
                    raise RuntimeError("cards %s did not respond" % np.nonzero(sh.done != seq[0])[0].tolist())
        if s >= 0:
            back_us[s] = (time.perf_counter() - t) * 1e6

    command(SEG_RESET, -1)
    hw = np.zeros((P, n_seg)); wall = np.zeros((P, n_seg)); seg_wall = np.zeros(n_seg); n_meas = [0]
    f32 = np.empty(H, dtype=np.float32)

    tparts = {"embed": 0.0, "commands": 0.0, "token": 0.0, "n": 0}

    def token_step(token_id, pos, measure):
        t_tok = time.perf_counter()
        sh.x[:] = embed[token_id]
        if measure:
            tparts["embed"] += time.perf_counter() - t_tok
        h = (sh.x.view(np.uint16).astype(np.uint32) << 16).view(np.float32).copy()
        for s in range(n_seg):
            t = time.perf_counter()
            command(s, pos if s == 0 else -1)
            if measure:
                seg_wall[s] += (time.perf_counter() - t) * 1e3
                hw[:, s] += sh.meta[:, 0]; wall[:, s] += sh.meta[:, 1]
            if s == n_seg - 1 or not a.host_sum:
                continue                      # cardsum: the cards add the shares themselves
            # --host-sum (reference only): residual stream += the cards' shares, in bf16
            parts = (sh.out.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            np.add(h, parts.sum(axis=0, dtype=np.float32), out=f32)
            xb = torch.from_numpy(f32).to(torch.bfloat16)
            sh.x[:] = xb.view(torch.int16).numpy()
            h = xb.float().numpy()
        if measure:
            n_meas[0] += 1
            tparts["token"] += time.perf_counter() - t_tok
        k = int(np.argmax(sh.meta[:, 2]))
        return voff[k] + int(sh.meta[k, 3])

    if a.eval_file:
        # Teacher-forced quality check, all on the cards: feed a fixed human-written text and count how
        # often the cards' top-1 prediction equals the actual next token. The host only compares ids.
        ev = [int(v) for v in tok.encode(open(a.eval_file).read(), add_special_tokens=False)][:a.eval_tokens]
        assert len(ev) <= a.max_context
        t1 = time.time()
        hits = 0
        for pos in range(len(ev) - 1):
            hits += token_step(ev[pos], pos, False) == ev[pos + 1]
        n = len(ev) - 1
        print("EVAL|%s|%d predictions|top-1 %d = %.2f%%|%.1f s" % (os.path.basename(a.eval_file), n, hits,
                                                                    100.0 * hits / n, time.time() - t1), flush=True)
        sh.cmd[1] = SEG_QUIT              # the cards exit without acknowledging
        sh.cmd[0] = seq[0] + 1
        return 0

    try:
        print("PROMPT|%d tokens|%r" % (len(ids), a.prompt), flush=True)
        t1 = time.time()
        nxt = None
        for pos, t in enumerate(ids):
            nxt = token_step(t, pos, False)
        t_prefill = time.time() - t1
        print("PREFILL|%d tokens|%.1f s|%.2f tokens/s" % (len(ids), t_prefill, len(ids) / t_prefill), flush=True)
        out = [nxt]
        t2 = time.time()
        pos = len(ids)
        while len(out) < a.max_new_tokens and out[-1] != eot:
            out.append(token_step(out[-1], pos, True))
            pos += 1
        t_dec = time.time() - t2
        n_dec = len(out) - 1
        text = tok.decode([t for t in out if t != eot], skip_special_tokens=False)
        print("DECODE|%d tokens|%.1f s|%.2f tokens/s|%.1f ms/token|%s" % (
            n_dec, t_dec, n_dec / max(t_dec, 1e-9), 1e3 * t_dec / max(n_dec, 1), "EOT" if out[-1] == eot else "length"),
            flush=True)
        n = max(n_meas[0], 1)
        print("TIMING|per decode token: token_step %.1f ms|embedding row %.2f ms|segment commands %.1f ms|"
              "outside token_step %.1f ms" % (1e3 * tparts["token"] / n, 1e3 * tparts["embed"] / n, seg_wall.sum() / n,
                                              1e3 * t_dec / max(n_dec, 1) - 1e3 * tparts["token"] / n), flush=True)
        hw /= n; wall /= n; seg_wall /= n
        lin = [s for s in range(n_seg - 1) if (s // 2) % 4 != 3]
        ful = [s for s in range(n_seg - 1) if (s // 2) % 4 == 3]
        for name, sel in (("linear attention", [s for s in lin if s % 2 == 0]), ("linear MLP", [s for s in lin if s % 2 == 1]),
                          ("full attention", [s for s in ful if s % 2 == 0]), ("full MLP", [s for s in ful if s % 2 == 1]),
                          ("LM head", [n_seg - 1])):
            print("SEGMENT|%-16s|count %2d|card compute avg %.2f ms, slowest card %.2f ms|slowest card round trip %.2f ms|"
                  "host %.2f ms|per token %.1f ms" % (
                      name, len(sel), hw[:, sel].mean(), hw[:, sel].max(axis=0).mean(), wall[:, sel].max(axis=0).mean(),
                      seg_wall[sel].mean(), seg_wall[sel].sum()), flush=True)
        print("SUMMARY|per token: host %.1f ms|slowest card compute total %.1f ms|average card compute %.1f ms" % (
            seg_wall.sum(), hw.max(axis=0).sum(), hw.mean(axis=0).sum()), flush=True)
        for k in range(P):
            print("CARD_TIME|%d|%s|compute %.1f ms/token|round trip %.1f ms/token" % (k, shard_dev[k], hw[k].sum(), wall[k].sum()))
        print("TOKENS|%s" % out[:24])
        print("TEXT_BEGIN")
        print(text)
        print("TEXT_END", flush=True)
    finally:
        sh.cmd[1] = SEG_QUIT
        sh.cmd[0] = seq[0] + 1
        for p in procs:
            p.join(timeout=20)
        sh.close(unlink=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
