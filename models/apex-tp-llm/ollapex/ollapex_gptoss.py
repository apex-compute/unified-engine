"""OllApeX backend for GPT-OSS-20B (MoE): the same interface as ollapex_engine.Loaded, driving the card
workers of gptoss_tp.py (5 cards x 2 engines, see gptoss_card.py). All model compute runs on the cards;
the host picks the next token from the cards' candidates and each card process picks its layer's top-4
experts from its own card's router logits.

Per token: A0, then per layer R(+E, run inside the R command) / A, then the LM head (H). Prompt tokens
whose prediction nobody reads skip H."""
import json
import multiprocessing as mp
import os
import time

import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # gptoss_tp / gptoss_card
import gptoss_tp as GT  # noqa: E402

SEG_QUIT, SEG_RESET, SEG_PARK = GT.SEG_QUIT, GT.SEG_RESET, GT.SEG_PARK
FAILED = GT.FAILED
MAX_RANK = 4


def pci_of(dev):
    """(PCI address, link width) of an xdma device, for the web UI's card list."""
    d = os.path.realpath("/sys/class/xdma/%s_user/device" % dev)
    try:
        width = open(os.path.join(d, "current_link_width")).read().strip()
    except OSError:
        width = "?"
    return os.path.basename(d), "x" + width


class LoadedGptOss:
    """GPT-OSS resident on the cards (driver side)."""

    def __init__(self, name, num_ctx, emit, paths, details, max_ctx):
        self.name, self.details = name, details
        self.num_ctx = min(num_ctx, max_ctx)
        self.paths = paths
        tdir = paths["tdir"]
        plan = json.load(open(os.path.join(tdir, "plan.json")))
        self.P, self.H = plan["P"], plan["hidden"]
        self.L = plan["config"]["num_hidden_layers"]
        self.voff = [c["vocab"][0] for c in plan["cards"]]
        self.E = int(os.environ.get("OLLAPEX_ENGINES", "2"))
        if self.P % self.E:
            raise RuntimeError("%d shards do not split into %d engines per card" % (self.P, self.E))
        n_cards = self.P // self.E
        devs = GT.cards_in_pci_order()
        if len(devs) < n_cards:
            raise RuntimeError("the model needs %d cards, %d cards are visible" % (n_cards, len(devs)))
        devs = devs[:n_cards]
        self.cards = [(None, devs[k // self.E], None) for k in range(self.P)]
        self.vision_k, self.rope_from = -1, None
        self.sh = GT.Shared(self.P, self.H)
        ctx = mp.get_context("spawn")
        cpus = GT.cpu_plan(n_cards)[1]
        t0 = time.time()
        card_pipes, self.procs = [], []
        for c, dev in enumerate(devs):
            pa, pb = ctx.Pipe()
            p = ctx.Process(target=GT.worker, args=(c, self.E, self.P, dev, tdir, self.num_ctx, cpus[c], self.sh.names, pb),
                            daemon=True)
            p.start()
            card_pipes.append(pa); self.procs.append(p)
        self.pipes = [card_pipes[k // self.E] for k in range(self.P)]
        self.card_info = [None] * self.P
        pending = set(range(n_cards))
        while pending:
            for c in list(pending):
                if card_pipes[c].poll(0.2):
                    m = card_pipes[c].recv()
                    if m[0] != "ready":
                        raise RuntimeError("card %d (%s): %s" % (c, devs[c], m[1] if len(m) > 1 else m))
                    bdf, link = pci_of(devs[c])
                    for e, i in enumerate(m[1]):
                        k = c * self.E + e
                        i.update({"index": k, "plan": plan["cards"][k], "bdf": bdf, "link": link,
                                  "instructions": int(i.get("prog_mb", 0.0) * 1e6 / 32)})
                        self.card_info[k] = i
                    pending.discard(c)
                    emit(("progress", {"ready": n_cards - len(pending), "cards": n_cards, "elapsed_s": time.time() - t0}))
                elif not self.procs[c].is_alive():
                    raise RuntimeError("the process of card %d ended unexpectedly (log: %s/card_%02d.log)" % (c, GT.LOGDIR, c))
        self.n_seg = self.card_info[0]["segs"]
        self.load_s = time.time() - t0
        shape = json.load(open(os.path.join(paths["wdir"], "embed.json")))["shape"]
        self.embed = np.fromfile(os.path.join(paths["wdir"], "embed.bin"), dtype=np.int16).reshape(tuple(shape))
        self.seq = 0
        self.fed = []
        self.command(SEG_RESET, -1)
        self.command(SEG_PARK, -1)

    def info(self):
        return {"model": self.name, "num_ctx": self.num_ctx, "cards": self.card_info, "segments_per_token": self.n_seg,
                "vision": False, "vision_card": -1, "layers": self.L, "hidden": self.H, "load_s": self.load_s,
                "weights_mb": sum(c["params_mb"] for c in self.card_info), "details": self.details}

    def set_rope(self, i0, pos3):
        raise ValueError("this model has no image support")

    def vision(self, image, W, H):
        raise ValueError("this model has no image support on the cards")

    def command(self, s, pos, want=1, limit_s=120.0):
        sh = self.sh
        sh.cmd[1] = s
        sh.cmd[2] = pos
        sh.cmd[3] = want
        self.seq += 1
        sh.cmd[0] = self.seq
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
        """One position through all layers; want == 0 skips the LM head. Returns [(logit, token)] or None."""
        sh = self.sh
        sh.x[:] = vec_i16
        last = self.n_seg - 1
        for s in range(self.n_seg if want else last):
            if s < last and s % 3 == 2:                # E runs inside the R command
                continue
            t = time.perf_counter()
            self.command(s, pos if s == 0 else -1, want)
            if acc is not None:
                acc["seg_wall"][s] += (time.perf_counter() - t) * 1e3
        if not want:
            return None
        cands = []
        for k in range(self.P):
            for r in range(max(1, min(want, MAX_RANK))):
                cands.append((float(sh.meta[k, 3 + 2 * r]), self.voff[k] + int(sh.meta[k, 2 + 2 * r])))
        return cands

    def seg_report(self, m, acc, n):
        n = max(n, 1)
        sw = acc["seg_wall"] / n
        last = self.n_seg - 1
        groups = (("attention", [s for s in range(last) if s % 3 == 0]),
                  ("router_and_experts", [s for s in range(last) if s % 3 == 1]),
                  ("lm_head", [last]))
        return {"segments": [{"name": g, "count": len(sel), "host_ms": float(sw[sel].mean()),
                              "per_token_ms": float(sw[sel].sum())} for g, sel in groups],
                "host_ms_per_token": float(sw.sum())}

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
        for b in self.sh.blocks:
            try:
                b.close()
                b.unlink()
            except Exception:
                pass
