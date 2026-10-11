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


def worker(k, P, dev, cdir, n_layers, max_ctx, cpu, nap, shm_names, conn):
    os.makedirs(LOGDIR, exist_ok=True)
    log = open(os.path.join(LOGDIR, "card_%02d.log" % k), "w", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sys.stdout = sys.stderr = log
    sh = None
    try:
        if cpu is not None:
            os.sched_setaffinity(0, {cpu})
        ctypes.CDLL(None).prctl(29, 1, 0, 0, 0)     # PR_SET_TIMERSLACK = 1 ns: short sleeps end on time
        torch.set_num_threads(1)
        t0 = time.time()
        T = load_port("q35_tp_port_%d" % k)
        T.set_dma_device(dev)
        T.DMA_DEVICE_H2C = T.user_dma_core.DMA_DEVICE_H2C
        T.DMA_DEVICE_C2H = T.user_dma_core.DMA_DEVICE_C2H
        T.user_dma_core.configure_clock_from_hardware()

        cfg_path = os.path.join(cdir, "config.json")
        cfg = json.load(open(cfg_path))
        H = cfg["file_info"]["hidden_size"]
        vocab = cfg["file_info"]["embedding_vocab"]
        weights = {
            "layers": LazyLayers(cdir, n_layers),
            "final_norm_gamma": torch.load(os.path.join(WDIR, "final_norm.pt"), weights_only=False),
            "embed_weight": torch.zeros(vocab, 1, dtype=torch.bfloat16),   # only its row count is used here
        }
        weights["lm_head_scale"], weights["lm_head_data"] = torch.load(os.path.join(cdir, "lm_head.pt"),
                                                                       weights_only=False)
        ue = T.Qwen3_5_2b_UnifiedEngine(device="cpu", config_path=cfg_path)
        ue._tp_mode = True
        ue._tp_segments = []
        ue.software_reset()
        ue._preallocate_identity_matrix()
        ue.prepare_inference(weights, max_context=max_ctx)
        del weights
        ue._decoder_bin_path = os.path.join(cdir, "decoder.bin")
        ue._decoder_meta_path = os.path.join(cdir, "decoder.json")
        for p in (ue._decoder_bin_path, ue._decoder_meta_path):
            if os.path.exists(p):
                os.remove(p)
        bin_path, sizes, _ = ue.compile_decoder()
        ue.load_instructions(bin_path)

        segs = list(ue._tp_segments)
        assert len(segs) == 2 * n_layers, "expected two cuts per layer, got %d" % len(segs)
        starts = [ue._decoder_prog_addr] + [ue._decoder_prog_addr + idx * INST for idx, _ in segs]
        outs = [addr for _, addr in segs]              # segment s < 2L writes H values here; the last one is the head
        n_seg = len(starts)
        est = [0.0] * n_seg                            # last measured run time of each segment, microseconds
        T_al = ue.max_context_aligned
        row_byt = ue.full_head_dim * BF16
        rope_byt = ue.full_rotary_dim * BF16
        x_dram = ue._decoder_X_dram
        logits = ue.LOGITS_DRAM
        pos_prog = ue.get_program_dram_addr()          # fixed slot for the per-token position program
        ue.allocate_program_dram(8 * INST)
        params_mb = (ue._next_params_dram_addr - ue._params_dram_base) / 1e6

        sh = Shared(P, H, shm_names)
        x_t = torch.from_numpy(sh.x).view(torch.bfloat16)
        out_t = torch.from_numpy(sh.out[k]).view(torch.bfloat16)
        bias = torch.empty((1, T_al), dtype=torch.bfloat16)

        def run(addr, expect_us, limit_s):
            ue.start_execute_from_dram(addr)
            t_wait = time.time()
            if nap and expect_us > 2 * nap:
                time.sleep((expect_us - nap) * 1e-6)   # more cards than CPUs: do not spin through the run
            while ue.is_queue_busy():
                os.sched_yield()
                if time.time() - t_wait >= limit_s:
                    raise RuntimeError("card program at 0x%x did not finish within %.0f s" % (addr, limit_s))

        def set_position(pos):
            bias.fill_(float("-inf"))
            bias[:, :pos + 1] = 0.0
            ue.dma_write(T.DMA_DEVICE_H2C, ue._bias_dram, bias, T_al * BF16)
            ue.reset_tensor_dram_addr()
            ue.clear_capture_buffer()
            T.UnifiedEngine.start_capture(ue)
            ue.generate_instruction_add_set(ue._ISA_POS_K_REG, T.ue_35bit_addr_shifter(pos * row_byt))
            ue.generate_instruction_add_set(ue._ISA_POS_ROPE_REG, T.ue_35bit_addr_shifter(pos * rope_byt))
            ue.generate_instruction_halt()
            T.UnifiedEngine.stop_capture(ue)
            ue.write_captured_instructions_to_dram(pos_prog)
            ue.clear_capture_buffer()
            run(pos_prog, 0.0, 5.0)

        conn.send(("ready", {"dev": dev, "params_mb": params_mb, "prog_mb": sizes[0] / 1e6,
                             "setup_s": time.time() - t0, "segs": n_seg}))
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
                    ue.reset_state()
                    done[k] = last
                    continue
                t1 = time.perf_counter()
                if pos >= 0:                          # first segment of a token
                    set_position(pos)
                ue.dma_write(T.DMA_DEVICE_H2C, x_dram, x_t, H * BF16)
                run(starts[s], est[s], 60.0)
                hw_us = ue.report_latency_in_us()
                est[s] = hw_us
                if s < n_seg - 1:
                    ue.dma_read(T.DMA_DEVICE_C2H, outs[s], out_t, H * BF16)
                else:
                    idx = int(ue.get_arg_max_index(rank=1))
                    blk = ue.dma_from_accelerator_memory(logits + (idx // 32) * 32 * BF16, torch.Size([1, 32]))
                    meta[k, 2] = float(blk.float()[0, idx % 32])
                    meta[k, 3] = idx
                meta[k, 0] = hw_us / 1e3
                meta[k, 1] = (time.perf_counter() - t1) * 1e3
                done[k] = last
    except Exception:
        traceback.print_exc()
        try:
            if sh is not None:
                sh.done[k] = FAILED
            conn.send(("error", traceback.format_exc()[-1500:]))
        except Exception:
            pass


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
    a = ap.parse_args()
    host_nap = a.host_nap_us
    if not host_nap:
        os.sched_setaffinity(0, {0})
    ctypes.CDLL(None).prctl(29, 1, 0, 0, 0)
    tdir = os.path.join(BASE, "tp%s" % a.tp)
    plan = json.load(open(os.path.join(tdir, "plan.json")))
    P, L, H = plan["P"], plan["layers"], plan["hidden"]
    devs = a.devs.split(",") if a.devs else cards_in_pci_order()[:P]
    assert len(devs) == P, "need %d cards, have %d" % (P, len(devs))
    voff = [c["vocab"][0] for c in plan["cards"]]
    ncpu = os.cpu_count()

    sh = Shared(P, H)
    ctx = mp.get_context("spawn")
    pipes, procs = [], []
    t0 = time.time()
    for k, dev in enumerate(devs):
        pa, pb = ctx.Pipe()
        p = ctx.Process(target=worker, args=(k, P, dev, os.path.join(tdir, "card_%02d" % k), L, a.max_context,
                                             (k % ncpu) if host_nap else 1 + k % (ncpu - 1), a.nap_us, sh.names, pb), daemon=True)
        p.start()
        pipes.append(pa); procs.append(p)
    n_seg = None
    for k, c in enumerate(pipes):
        m = c.recv()
        if m[0] != "ready":
            print("ERROR|card %d|%s" % (k, m[1] if len(m) > 1 else m))
            return 1
        i = m[1]
        n_seg = i["segs"]
        print("CARD|%d|%s|weights %.0f MB|program %.1f MB|setup %.0f s" % (
            k, i["dev"], i["params_mb"], i["prog_mb"], i["setup_s"]), flush=True)
    print("READY|%d cards|%d segments/token|%.0f s" % (P, n_seg, time.time() - t0), flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    text_in = tok.apply_chat_template([{"role": "user", "content": a.prompt}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=a.think)
    ids = [int(v) for v in tok.encode(text_in, add_special_tokens=True)]
    eot = tok.convert_tokens_to_ids("<|im_end|>")
    shape = json.load(open(os.path.join(WDIR, "embed.json")))["shape"]
    embed = np.memmap(os.path.join(WDIR, "embed.bin"), dtype=np.int16, mode="r", shape=tuple(shape))
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

    def token_step(token_id, pos, measure):
        sh.x[:] = embed[token_id]
        h = (sh.x.view(np.uint16).astype(np.uint32) << 16).view(np.float32).copy()
        for s in range(n_seg):
            t = time.perf_counter()
            command(s, pos if s == 0 else -1)
            if measure:
                seg_wall[s] += (time.perf_counter() - t) * 1e3
                hw[:, s] += sh.meta[:, 0]; wall[:, s] += sh.meta[:, 1]
            if s == n_seg - 1:
                break
            # residual stream += the cards' shares; carried at the cards' precision (bf16)
            parts = (sh.out.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            np.add(h, parts.sum(axis=0, dtype=np.float32), out=f32)
            xb = torch.from_numpy(f32).to(torch.bfloat16)
            sh.x[:] = xb.view(torch.int16).numpy()
            h = xb.float().numpy()
        if measure:
            n_meas[0] += 1
        k = int(np.argmax(sh.meta[:, 2]))
        return voff[k] + int(sh.meta[k, 3])

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
            print("CARD_TIME|%d|%s|compute %.1f ms/token|round trip %.1f ms/token" % (k, devs[k], hw[k].sum(), wall[k].sum()))
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
