"""OllApeX engine: keeps a Qwen3.5-family model loaded tensor-parallel on the FPGA cards and
generates tokens on request. All model compute runs on the cards.

Three kinds of processes:
  server (ollapex_server.py)  HTTP, tokenizer, chat template
  driver (this file)          one per loaded model; owns the token loop, pinned to CPU 0
  card workers (this file)    one per card; each drives its card's slice of every layer

The driver and the workers talk through shared memory (see ../run_tp.py for why). Between
requests the workers are parked: they poll slowly instead of spinning.
"""
import ctypes
import glob
import hashlib
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

EXP = os.path.join(os.path.expanduser(os.environ.get("APEX_TP_WORK", "~/apex_tp")), "models")
LOGDIR = os.environ.get("OLLAPEX_LOGDIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
BF16 = 2
INST = 32
SEG_QUIT, SEG_RESET, SEG_PARK, SEG_VISION, SEG_ROPE = -1, -2, -3, -4, -5
VIS_REQ, VIS_OUT, VIS_META = "/dev/shm/ollapex_vis_req.json", "/dev/shm/ollapex_vis_out.npy", "/dev/shm/ollapex_vis_meta.json"
ROPE_FILE = "/dev/shm/ollapex_rope.npz"
FAILED = -1
MAX_RANK = 4                      # the card tracks its four best logits
MAX_CTX = 1024                    # per-layer attention scratch of every card must fit its 512 MB tensor region

# name -> (directory under $APEX_TP_WORK/models, sliced weight set, display details)
MODELS = {
    "qwen3.8:27b": ("q38_27b", "11", {"family": "qwen3.8", "parameter_size": "27B"}),
    "qwen3.5:9b": ("q35_9b", "11v2", {"family": "qwen3.5", "parameter_size": "9B"}),
}


def model_paths(name):
    d, tp, _ = MODELS[name]
    base = os.path.join(EXP, d)
    hf = os.path.dirname(sorted(glob.glob(os.path.join(base, "*_bin", "*", "config.json")))[0])
    return {"base": base, "tdir": os.path.join(base, "tp" + tp), "wdir": os.path.join(base, "weights"),
            "port": os.path.join(base, "qwen3.5_2b_test.py"), "hf": hf,
            "config": os.path.join(base, "qwen3.5_2b_config.json"),
            "vision": os.path.join(base, "vision", "vision_weights.pt")}


def available_models():
    out = []
    for name in MODELS:
        try:
            p = model_paths(name)
            if os.path.exists(os.path.join(p["tdir"], "plan.json")):
                out.append(name)
        except IndexError:
            pass
    return out


def cards_in_pci_order():
    out = []
    for u in glob.glob("/sys/class/xdma/xdma*_user"):
        dev = os.path.basename(u)[:-len("_user")]
        bdf = os.path.basename(os.path.realpath(os.path.join(u, "device")))
        rd = lambda f: open("/sys/bus/pci/devices/%s/%s" % (bdf, f)).read().strip()
        out.append((bdf, dev, "%s x%s" % (rd("current_link_speed").replace(" PCIe", ""), rd("current_link_width"))))
    return sorted(out)


class Shared:
    """Command block written by the driver, one result row per card."""

    def __init__(self, P, H, names=None):
        spec = (("cmd", np.int64, (4,)),            # sequence number, segment, position (-1: none), candidates wanted
                ("x", np.int16, (H,)),              # input vector for the segment (bf16 bits)
                ("done", np.int64, (P,)),           # sequence number each card finished last (FAILED on error)
                ("meta", np.float64, (P, 2 + 2 * MAX_RANK)),   # card ms, round-trip ms, then (index, logit) per rank
                ("out", np.int16, (P, H)))          # each card's partial output (bf16 bits)
        self.blocks, self.names = [], []
        for i, (name, dtype, shape) in enumerate(spec):
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            if names is None:
                shm = shared_memory.SharedMemory(create=True, size=size)
            else:
                try:
                    shm = shared_memory.SharedMemory(name=names[i], track=False)   # the driver owns it
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
            try:
                shm.close()
                if unlink:
                    shm.unlink()
            except Exception:
                pass


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


def worker(k, P, dev, paths, n_layers, max_ctx, cpu, nap, shm_names, conn, with_vision=False):
    os.makedirs(LOGDIR, exist_ok=True)
    log = open(os.path.join(LOGDIR, "card_%02d.log" % k), "w", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sys.stdout = sys.stderr = log
    sh = None
    try:
        os.sched_setaffinity(0, {cpu})
        ctypes.CDLL(None).prctl(29, 1, 0, 0, 0)     # PR_SET_TIMERSLACK = 1 ns: short sleeps end on time
        torch.set_num_threads(1)
        t0 = time.time()
        spec = importlib.util.spec_from_file_location("ollapex_port_%d" % k, paths["port"])
        T = importlib.util.module_from_spec(spec)
        sys.modules["ollapex_port_%d" % k] = T
        spec.loader.exec_module(T)
        T.set_dma_device(dev)
        T.DMA_DEVICE_H2C = T.user_dma_core.DMA_DEVICE_H2C
        T.DMA_DEVICE_C2H = T.user_dma_core.DMA_DEVICE_C2H
        T.user_dma_core.configure_clock_from_hardware()

        cdir = os.path.join(paths["tdir"], "card_%02d" % k)
        cfg_path = os.path.join(cdir, "config.json")
        cfg = json.load(open(cfg_path))
        H = cfg["file_info"]["hidden_size"]
        vocab = cfg["file_info"]["embedding_vocab"]
        weights = {
            "layers": LazyLayers(cdir, n_layers),
            "final_norm_gamma": torch.load(os.path.join(paths["wdir"], "final_norm.pt"), weights_only=False),
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
        ue._decoder_bin_path = os.path.join(cdir, "decoder_ctx%d.bin" % max_ctx)
        ue._decoder_meta_path = os.path.join(cdir, "decoder_ctx%d.json" % max_ctx)
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
        # The LM never touches tensor memory above this address: that is where the vision encoder lives.
        lm_tensor_end = (ue._tensor_dram_addr + 0xFFFF) & ~0xFFFF
        tensor_limit = T.user_dma_core.DRAM_INSTRUCTION_ADDR
        vision = None
        if with_vision:
            import ollapex_vision
            vision = ollapex_vision.VisionCard(T, ue, paths["vision"], lm_tensor_end, tensor_limit)
        rot_dim = ue.full_rotary_dim
        info = {"dev": dev, "segs": n_seg, "setup_s": time.time() - t0, "prog_mb": sizes[0] / 1e6,
                "params_mb": (ue._next_params_dram_addr - ue._params_dram_base) / 1e6,
                "instructions": sizes[0] // INST, "tensor_free_mb": (tensor_limit - lm_tensor_end) / 1048576.0,
                "clock_mhz": 1000.0 / float(getattr(T.user_dma_core, "CLOCK_PERIOD_NS", 0) or 5.042)}

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

        conn.send(("ready", info))
        last = 0
        parked = True
        cmd, done, meta = sh.cmd, sh.done, sh.meta
        with T._quiet():
            while True:
                while cmd[0] == last:
                    if parked:
                        time.sleep(0.002)             # idle: do not burn a CPU
                    else:
                        os.sched_yield()
                last = int(cmd[0]); s = int(cmd[1]); pos = int(cmd[2]); want = int(cmd[3])
                if s == SEG_QUIT:
                    break
                if s == SEG_PARK:
                    parked = True
                    done[k] = last
                    continue
                parked = False
                if s == SEG_RESET:
                    ue.reset_state()
                    done[k] = last
                    continue
                if s == SEG_VISION:
                    if pos == k and vision is not None:
                        try:
                            rq = json.load(open(VIS_REQ))
                            toks, vi = vision.encode(rq["image"], rq["W"], rq["H"])
                            np.save(VIS_OUT, toks)
                            json.dump(vi, open(VIS_META, "w"))
                        except Exception:
                            traceback.print_exc()
                            json.dump({"error": traceback.format_exc()[-1500:]}, open(VIS_META, "w"))
                    done[k] = last
                    continue
                if s == SEG_ROPE:
                    # position rows of the attention layers' rotary tables, rewritten for image positions
                    z = np.load(ROPE_FILE)
                    off = int(z["i0"]) * rot_dim * BF16
                    for name, addr in (("cos", ue._fa_cos_pad_dram), ("nsin", ue._fa_neg_sin_pad_dram), ("sin", ue._fa_sin_hi_pad_dram)):
                        t = torch.from_numpy(np.ascontiguousarray(z[name])).view(torch.bfloat16)
                        ue.dma_write(T.DMA_DEVICE_H2C, addr + off, t, t.numel() * BF16)
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
                    for r in range(max(1, min(want, MAX_RANK))):
                        idx = int(ue.get_arg_max_index(rank=r + 1))
                        if not 0 <= idx < vocab:
                            idx = 0
                        blk = ue.dma_from_accelerator_memory(logits + (idx // 32) * 32 * BF16, torch.Size([1, 32]))
                        meta[k, 2 + 2 * r] = idx
                        meta[k, 3 + 2 * r] = float(blk.float()[0, idx % 32])
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


class Loaded:
    """A model resident on the cards (driver side)."""

    def __init__(self, name, num_ctx, emit, nap_us=30.0):
        self.name, self.num_ctx = name, num_ctx
        self.paths = model_paths(name)
        plan = json.load(open(os.path.join(self.paths["tdir"], "plan.json")))
        self.P, self.L, self.H = plan["P"], plan["layers"], plan["hidden"]
        self.voff = [c["vocab"][0] for c in plan["cards"]]
        self.plan = plan["cards"]
        cards = cards_in_pci_order()
        if len(cards) < self.P:
            raise RuntimeError("the model needs %d cards, %d cards are visible" % (self.P, len(cards)))
        self.cards = cards[:self.P]
        # the image encoder lives on the last card (it carries the fewest heads)
        self.vision_k = self.P - 1 if os.path.exists(self.paths["vision"]) else -1
        cfg = json.load(open(self.paths["config"]))
        self.rot_dim = cfg["file_info"]["full_attn_rotary_dim"]
        self.rope_theta = float(cfg["special"]["rope"]["theta_full"])
        hf = json.load(open(os.path.join(self.paths["hf"], "config.json")))
        self.mrope = hf.get("text_config", {}).get("rope_parameters", {}).get("mrope_section", [11, 11, 10])
        self.rope_from = None             # first position whose rotary row is not the default one
        ncpu = os.cpu_count()
        self.sh = Shared(self.P, self.H)
        ctx = mp.get_context("spawn")
        self.pipes, self.procs = [], []
        t0 = time.time()
        for k, (bdf, dev, link) in enumerate(self.cards):
            pa, pb = ctx.Pipe()
            p = ctx.Process(target=worker, args=(k, self.P, dev, self.paths, self.L, num_ctx, 1 + k % (ncpu - 1), nap_us,
                                                 self.sh.names, pb, k == self.vision_k), daemon=True)
            p.start()
            self.pipes.append(pa); self.procs.append(p)
        self.card_info = [None] * self.P
        pending = set(range(self.P))
        while pending:
            for k in list(pending):
                if self.pipes[k].poll(0.2):
                    m = self.pipes[k].recv()
                    if m[0] != "ready":
                        raise RuntimeError("card %d (%s): %s" % (k, self.cards[k][1], m[1] if len(m) > 1 else m))
                    i = m[1]
                    i.update({"index": k, "bdf": self.cards[k][0], "link": self.cards[k][2], "plan": self.plan[k]})
                    self.card_info[k] = i
                    pending.discard(k)
                    emit(("progress", {"ready": self.P - len(pending), "cards": self.P, "elapsed_s": time.time() - t0}))
                elif not self.procs[k].is_alive():
                    raise RuntimeError("the process of card %d ended unexpectedly (log: %s/card_%02d.log)" % (k, LOGDIR, k))
        self.n_seg = self.card_info[0]["segs"]
        self.load_s = time.time() - t0
        shape = json.load(open(os.path.join(self.paths["wdir"], "embed.json")))["shape"]
        self.embed = np.memmap(os.path.join(self.paths["wdir"], "embed.bin"), dtype=np.int16, mode="r", shape=tuple(shape))
        self.seq = 0
        self.fed = []                 # what the cards' state already contains: one key per position
        self.f32 = np.empty(self.H, dtype=np.float32)
        self.command(SEG_RESET, -1)
        self.command(SEG_PARK, -1)

    def vision(self, image, W, H):
        """Run the image encoder on its card. Returns (tokens int16 [n, H], info)."""
        if self.vision_k < 0:
            raise ValueError("this model has no image support on the cards")
        for f in (VIS_OUT, VIS_META):
            if os.path.exists(f):
                os.remove(f)
        json.dump({"image": image, "W": W, "H": H}, open(VIS_REQ, "w"))
        self.command(SEG_VISION, self.vision_k, limit_s=900.0)
        meta = json.load(open(VIS_META))
        self.command(SEG_PARK, -1)
        if "error" in meta:
            raise ValueError("image encoder error: " + meta["error"].strip().split("\n")[-1])
        meta["card"] = self.cards[self.vision_k][1]
        return np.load(VIS_OUT), meta

    def set_rope(self, i0, pos3):
        """Rewrite the rotary table rows for sequence indexes i0.. on every card. pos3[j] is the
        (t, h, w) position of index i0+j; text has t == h == w, image tokens differ per axis.
        Frequency slots are interleaved T,H,W,T,H,W,... as in the model (mrope sections)."""
        pos3 = np.asarray(pos3, dtype=np.float32).reshape(-1, 3)
        half = self.rot_dim // 2
        inv = 1.0 / (self.rope_theta ** (np.arange(0, self.rot_dim, 2, dtype=np.float32) / self.rot_dim))
        axis = np.zeros(half, dtype=np.int64)                 # which of (t, h, w) drives each frequency slot
        axis[1:self.mrope[1] * 3:3] = 1
        axis[2:self.mrope[2] * 3:3] = 2
        fr = torch.from_numpy(pos3[:, axis] * inv[None, :])  # float32, as the port computes its table
        z = {"i0": np.int64(i0)}
        for name, val in (("cos", fr.cos()), ("nsin", -fr.sin()), ("sin", fr.sin())):
            buf = torch.zeros(pos3.shape[0], self.rot_dim, dtype=torch.bfloat16)
            buf[:, :half] = val.to(torch.bfloat16)
            z[name] = buf.view(torch.int16).numpy()
        np.savez(ROPE_FILE, **z)
        self.command(SEG_ROPE, -1)

    def info(self):
        return {"model": self.name, "num_ctx": self.num_ctx, "cards": self.card_info, "segments_per_token": self.n_seg,
                "vision": self.vision_k >= 0, "vision_card": self.vision_k,
                "layers": self.L, "hidden": self.H, "load_s": self.load_s,
                "weights_mb": sum(c["params_mb"] for c in self.card_info), "details": MODELS[self.name][2]}

    def command(self, s, pos, want=1, limit_s=120.0):
        sh = self.sh
        sh.cmd[1] = s
        sh.cmd[2] = pos
        sh.cmd[3] = want
        self.seq += 1
        sh.cmd[0] = self.seq              # the bump is what the cards watch for
        t = time.perf_counter()
        n = 0
        while not (sh.done == self.seq).all():
            n += 1
            if n & 0x3FF == 0:
                if (sh.done == FAILED).any():
                    k = int(np.argmax(sh.done == FAILED))
                    m = self.pipes[k].recv() if self.pipes[k].poll(2.0) else ("error", "no detail")
                    raise RuntimeError("card %d: %s" % (k, m[1]))
                if time.perf_counter() - t > limit_s:
                    raise RuntimeError("cards %s did not respond" % np.nonzero(sh.done != self.seq)[0].tolist())

    def forward(self, vec_i16, pos, want, acc):
        """One position through all layers. want == 0: skip the LM head (prompt tokens whose
        prediction nobody reads). Returns the candidate list [(logit, token)] or None."""
        sh = self.sh
        sh.x[:] = vec_i16
        h = (sh.x.view(np.uint16).astype(np.uint32) << 16).view(np.float32).copy()
        last = self.n_seg - 1
        for s in range(self.n_seg if want else last):
            t = time.perf_counter()
            self.command(s, pos if s == 0 else -1, want)
            if acc is not None:
                acc["seg_wall"][s] += (time.perf_counter() - t) * 1e3
                acc["hw"][:, s] += sh.meta[:, 0]
                acc["wall"][:, s] += sh.meta[:, 1]
            if s == last:
                break
            # residual stream += the cards' shares; carried at the cards' precision (bf16)
            parts = (sh.out.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            np.add(h, parts.sum(axis=0, dtype=np.float32), out=self.f32)
            xb = torch.from_numpy(self.f32).to(torch.bfloat16)
            sh.x[:] = xb.view(torch.int16).numpy()
            h = xb.float().numpy()
        if not want:
            return None
        cands = []
        for k in range(self.P):
            for r in range(max(1, min(want, MAX_RANK))):
                cands.append((float(sh.meta[k, 3 + 2 * r]), self.voff[k] + int(sh.meta[k, 2 + 2 * r])))
        return cands

    def close(self):
        try:
            self.sh.cmd[1] = SEG_QUIT
            self.sh.cmd[0] = self.seq + 1
        except Exception:
            pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        self.sh.close(unlink=True)


def pick(cands, opts, rng):
    """Choose the next token from the cards' candidates. Greedy unless temperature > 0; then
    sample among the global top-k (k <= 4: each card reports its four best, so the global top
    four are always among them), optionally cut by top_p."""
    cands = sorted(cands, reverse=True)
    temp = float(opts.get("temperature", 0.0) or 0.0)
    if temp <= 0.0:
        return cands[0][1], cands[0][0]
    k = max(1, min(int(opts.get("top_k", MAX_RANK) or MAX_RANK), MAX_RANK))
    top = cands[:k]
    z = np.array([c[0] for c in top], dtype=np.float64) / temp
    p = np.exp(z - z.max())
    p /= p.sum()
    top_p = float(opts.get("top_p", 1.0) or 1.0)
    if top_p < 1.0:
        keep = int(np.searchsorted(np.cumsum(p), top_p) + 1)
        p = p[:keep] / p[:keep].sum()
        top = top[:keep]
    i = int(rng.choice(len(top), p=p))
    return top[i][1], top[i][0]


def seg_report(m, acc, n):
    """Per-token timing, grouped by what the segment does."""
    n = max(n, 1)
    hw, wall, sw = acc["hw"] / n, acc["wall"] / n, acc["seg_wall"] / n
    S = m.n_seg
    groups = (("linear_attention", [s for s in range(S - 1) if (s // 2) % 4 != 3 and s % 2 == 0]),
              ("linear_mlp", [s for s in range(S - 1) if (s // 2) % 4 != 3 and s % 2 == 1]),
              ("full_attention", [s for s in range(S - 1) if (s // 2) % 4 == 3 and s % 2 == 0]),
              ("full_mlp", [s for s in range(S - 1) if (s // 2) % 4 == 3 and s % 2 == 1]),
              ("lm_head", [S - 1]))
    segs = []
    for name, sel in groups:
        segs.append({"name": name, "count": len(sel), "card_ms_avg": float(hw[:, sel].mean()),
                     "card_ms_slowest": float(hw[:, sel].max(axis=0).mean()),
                     "roundtrip_ms_slowest": float(wall[:, sel].max(axis=0).mean()),
                     "host_ms": float(sw[sel].mean()), "per_token_ms": float(sw[sel].sum())})
    return {"segments": segs, "host_ms_per_token": float(sw.sum()),
            "slowest_card_compute_ms_per_token": float(hw.max(axis=0).sum()),
            "avg_card_compute_ms_per_token": float(hw.mean(axis=0).sum()),
            "cards": [{"index": k, "dev": m.cards[k][1], "compute_ms_per_token": float(hw[k].sum()),
                       "roundtrip_ms_per_token": float(wall[k].sum())} for k in range(m.P)]}


def generate(m, req, emit, abort):
    """req: ids (token ids), vecs ({position: int16 bytes} for positions fed a vector instead of a
    token embedding, e.g. image tokens), opts, stop_ids."""
    ids, opts = req["ids"], req.get("opts", {})
    vecs = req.get("vecs") or {}
    stop_ids = set(req.get("stop_ids") or [])
    keys = [(-1, hashlib.md5(vecs[i]).hexdigest()) if i in vecs else (t, "") for i, t in enumerate(ids)]
    if len(ids) >= m.num_ctx:
        raise ValueError("the prompt is %d tokens; the context limit (num_ctx) is %d" % (len(ids), m.num_ctx))
    num_predict = int(opts.get("num_predict", 256) or 256)
    if num_predict < 0:
        num_predict = m.num_ctx
    num_predict = max(1, min(num_predict, m.num_ctx - len(ids)))
    want = 1 if float(opts.get("temperature", 0.0) or 0.0) <= 0 else max(1, min(int(opts.get("top_k", MAX_RANK) or MAX_RANK), MAX_RANK))
    seed = opts.get("seed")
    rng = np.random.default_rng(None if seed in (None, -1) else int(seed))

    # Image tokens carry (t, h, w) positions; everything after an image is shifted accordingly.
    rope = req.get("rope")
    if rope:
        tail = [[i - rope["shift"]] * 3 for i in range(len(ids), m.num_ctx)]
        m.set_rope(rope["i0"], list(rope["pos"]) + tail)
        m.rope_from = rope["i0"] if m.rope_from is None else min(m.rope_from, rope["i0"])
    elif m.rope_from is not None:
        m.set_rope(m.rope_from, [[i] * 3 for i in range(m.rope_from, m.num_ctx)])
        m.rope_from = None

    # Reuse what the cards already hold when this prompt continues the previous exchange.
    reuse = 0
    if 0 < len(m.fed) < len(keys) and m.fed == keys[:len(m.fed)]:
        reuse = len(m.fed)
    else:
        m.command(SEG_RESET, -1)
        m.fed = []
    emit(("start", {"prompt_tokens": len(ids), "reused_tokens": reuse, "num_predict": num_predict, "num_ctx": m.num_ctx}))

    def vec_at(i, token):
        return np.frombuffer(vecs[i], dtype=np.int16) if i in vecs else m.embed[token]

    t0 = time.perf_counter()
    cands = None
    for i in range(reuse, len(ids)):
        if abort.is_set():
            break
        lastp = i == len(ids) - 1
        cands = m.forward(vec_at(i, ids[i]), i, want if lastp else 0, None)
        m.fed.append(keys[i])
        if (i - reuse) % 8 == 7 and not lastp:
            emit(("prefill", {"done": i + 1 - reuse, "total": len(ids) - reuse}))
    t_prefill = time.perf_counter() - t0
    n_prefill = len(m.fed) - reuse

    acc = {"hw": np.zeros((m.P, m.n_seg)), "wall": np.zeros((m.P, m.n_seg)), "seg_wall": np.zeros(m.n_seg)}
    out, reason = [], "stop"
    t1 = time.perf_counter()
    ttft = None
    pos = len(ids)
    if abort.is_set() or cands is None:
        reason = "abort"
    else:
        while True:
            tok, logit = pick(cands, opts, rng)
            out.append(tok)
            if ttft is None:
                ttft = time.perf_counter() - t0
            emit(("token", tok, logit))
            if tok in stop_ids:
                reason = "stop"
                break
            if len(out) >= num_predict:
                reason = "length"
                break
            if abort.is_set():
                reason = "abort"
                break
            cands = m.forward(m.embed[tok], pos, want, acc)
            m.fed.append((tok, ""))
            pos += 1
    t_dec = time.perf_counter() - t1
    n_steps = max(len(out) - 1, 0)        # decode forwards measured (the first token came from the prefill)
    stats = {"prompt_tokens": len(ids), "reused_tokens": reuse, "prefill_tokens": n_prefill, "prefill_s": t_prefill,
             "prefill_tok_s": (n_prefill / t_prefill) if t_prefill > 0 and n_prefill else 0.0,
             "eval_tokens": len(out), "eval_s": t_dec, "decode_steps": n_steps,
             "eval_tok_s": (n_steps / t_dec) if t_dec > 0 and n_steps else 0.0,
             "ms_per_token": (1e3 * t_dec / n_steps) if n_steps else 0.0, "ttft_s": ttft or 0.0,
             "total_s": time.perf_counter() - t0, "done_reason": reason, "context_used": len(m.fed) + (1 if out else 0),
             "num_ctx": m.num_ctx, "sampling": {"temperature": float(opts.get("temperature", 0.0) or 0.0),
                                                "top_k": want, "top_p": float(opts.get("top_p", 1.0) or 1.0),
                                                "seed": seed, "candidates_per_card": want},
             "segments_per_token": m.n_seg}
    if n_steps:
        stats.update(seg_report(m, acc, n_steps))
    m.command(SEG_PARK, -1)
    emit(("done", stats))


def driver_main(conn, abort):
    """Driver process: owns the loaded model. Commands in, events out, over one pipe."""
    os.sched_setaffinity(0, {0})
    ctypes.CDLL(None).prctl(29, 1, 0, 0, 0)
    torch.set_num_threads(1)
    m = None

    def emit(ev):
        conn.send(ev)

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        try:
            if msg[0] == "load":
                name, num_ctx = msg[1], int(msg[2])
                if m is not None:
                    m.close()
                    m = None
                m = Loaded(name, num_ctx, emit)
                emit(("ready", m.info()))
            elif msg[0] == "generate":
                if m is None:
                    raise RuntimeError("no model loaded")
                abort.clear()
                generate(m, msg[1], emit, abort)
            elif msg[0] == "vision":
                if m is None:
                    raise RuntimeError("no model loaded")
                toks, meta = m.vision(msg[1], msg[2], msg[3])
                emit(("vision_done", toks.tobytes(), list(toks.shape), meta))
            elif msg[0] == "unload":
                if m is not None:
                    m.close()
                    m = None
                emit(("unloaded",))
            elif msg[0] == "quit":
                break
        except Exception as e:
            if isinstance(e, ValueError):
                emit(("error", str(e), False))
            else:
                # the cards' state is unknown after a failure: drop the model so the next request reloads it
                detail = traceback.format_exc()[-1800:]
                try:
                    if m is not None:
                        m.close()
                except Exception:
                    pass
                m = None
                emit(("error", "%s\n%s" % (e, detail), True))
    if m is not None:
        m.close()
