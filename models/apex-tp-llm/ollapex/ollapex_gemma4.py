"""OllApeX backend for Gemma 4 26B-A4B (MoE): the same interface as ollapex_engine.Loaded, driving the card
workers of gemma4_tp.py (5 cards x 2 engines, see gemma4_card.py). All model compute runs on the cards;
the host picks the next token from the cards' candidates and each card process picks its layer's top-8
experts from its own card's router logits.

Per token: A0, then per layer R(+D, E, run inside the R command) / A, then the LM head (H). Prompt tokens
whose prediction nobody reads skip H.

Images: the encoder runs tensor-parallel on the first GEMMA4_VISION_SHARDS shards (default 8: the x4 cards),
at one of three detail levels (70 / 140 / 280 soft tokens). The soft tokens then go through the LM as one
block (exact bidirectional attention inside the image, see gemma4_card.compile_block)."""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

# image support (read by gemma4_tp at import; the card processes inherit it)
os.environ.setdefault("GEMMA4_VISION_SHARDS", "8")
os.environ.setdefault("GEMMA4_MAX_BLOCK", "288")
sys.path.insert(0, os.environ.get("GEMMA4_CODE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gemma4")))
import gemma4_tp as GT  # noqa: E402

DETAIL = {"low": 70, "medium": 140, "high": 280}      # soft tokens per image, at most

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


class LoadedGemma4:
    """Gemma 4 resident on the cards (driver side)."""

    def __init__(self, name, num_ctx, emit, paths, details, max_ctx):
        self.name, self.details = name, details
        self.num_ctx = min(num_ctx, max_ctx)
        self.paths = paths
        tdir = paths["tdir"]
        plan = json.load(open(os.path.join(tdir, "plan.json")))
        self.P, self.H = plan["P"], plan["hidden"]
        self.L = plan["config"]["num_hidden_layers"]
        self.voff = [c["vocab"][0] for c in plan["cards"]]
        self.E = 2
        n_cards = self.P // self.E
        devs = GT.cards_in_pci_order()
        if len(devs) < n_cards:
            raise RuntimeError("the model needs %d cards, %d cards are visible" % (n_cards, len(devs)))
        devs = devs[:n_cards]
        self.cards = [(None, devs[k // self.E], None) for k in range(self.P)]
        self.vision_k, self.rope_from = -1, None
        self.has_vision = bool(GT.VIS_P and GT.MAX_BLOCK and os.path.exists(GT.VIS_PATH))
        self.sh = GT.Shared(self.P, self.H, n_cards)
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
        self.vinfo = self.card_info[0].get("vision")
        self.has_vision = self.has_vision and bool(self.vinfo)
        self.load_s = time.time() - t0
        # the embedding table is on the cards: embed[token] copies that one row from the card that holds it
        self.embed = GT.CardEmbed(plan, devs, self.E, [i["emb_addr"] for i in self.card_info], self.H)
        self.seq = 0
        self.fed = []
        self.command(SEG_RESET, -1)
        self.command(SEG_PARK, -1)

    def info(self):
        return {"model": self.name, "num_ctx": self.num_ctx, "cards": self.card_info, "segments_per_token": self.n_seg,
                "vision": self.has_vision, "vision_card": -1,
                "vision_shards": GT.VIS_P if self.has_vision else 0, "vision_levels": DETAIL if self.has_vision else {},
                "vision_load_s": (self.vinfo or {}).get("load_s", 0.0), "max_image_tokens": GT.MAX_BLOCK, "layers": self.L, "hidden": self.H, "load_s": self.load_s,
                "weights_mb": sum(c["params_mb"] for c in self.card_info), "details": self.details}

    def set_rope(self, i0, pos3):
        raise ValueError("this model gives its image tokens plain 1D positions")

    def vision(self, image, W, H, level=None):
        """One image through the encoder on the cards. The host only resizes it and cuts it into patches (the
        model's image processor) and writes those inputs; the 27 layers run as 2 segments each over the cards.
        Returns the soft tokens (int16 view of bf16, [n, H]) and a report."""
        if not self.has_vision:
            raise ValueError("The loaded model cannot read images.")
        import torch
        import gemma4_vision_tp as GV
        from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor
        from PIL import Image
        S = int(level or DETAIL["low"])
        if S not in GT.VIS_LEVELS:
            raise ValueError("image detail level %d is not one of %s" % (S, GT.VIS_LEVELS))
        sh = self.sh
        t0 = time.time()
        inp = Gemma4ImageProcessor(max_soft_tokens=S)(images=[Image.open(image).convert("RGB")], return_tensors="pt")
        pix, oh, bias, n_soft = GV.level_inputs(inp["pixel_values"], inp["image_position_ids"], S, GT.VIS_POS)
        NP = pix.shape[0]
        sh.vin_pix[:NP] = pix.view(torch.int16).numpy()
        sh.vin_oh[:NP] = oh.view(torch.int16).numpy()
        sh.vin_bias[:NP, :NP] = bias.view(torch.int16).numpy()
        t1 = time.time()
        self.command(GT.SEG_VIS_BEGIN, S)
        n = 2 * 27 + 1                                    # A and M per layer, then P (pool + projection)
        for i in range(n):
            self.command(GT.VIS + i, -1, limit_s=300.0)
        t2 = time.time()
        toks = sh.vout[:n_soft].copy()
        pp = inp["image_position_ids"][0]
        pp = pp[(pp >= 0).all(-1)]
        g0, g1 = int(pp[:, 0].max()) + 1, int(pp[:, 1].max()) + 1     # patch grid
        progs = (self.vinfo or {}).get("programs", {})
        prog = progs.get(S) or progs.get(str(S)) or []
        meta = {"patches": int(len(pp)), "tokens": int(n_soft), "level": S, "grid": [g0, g1],
                "encoded": [16 * g0, 16 * g1], "card": "%d engines on %d cards" % (GT.VIS_P, GT.VIS_P // self.E),
                "card_s": t2 - t1, "prep_s": t1 - t0, "compile_s": 0.0, "weights_upload_s": 0.0,
                "instructions": int(sum(p[0] for p in prog) / len(prog)) if prog else 0, "segments": n}
        return toks, meta

    def forward_block(self, rows, p0):
        """An image block (its soft tokens [n, H], int16 view of bf16) at positions p0 .. p0+n-1, layer by layer
        through the block program: K for every token, then per layer B / R (D and E run inside it) / F per token."""
        sh = self.sh
        n = rows.shape[0]
        if n > GT.MAX_BLOCK:
            raise ValueError("an image of %d tokens is longer than the longest block (%d)" % (n, GT.MAX_BLOCK))
        sh.xb[:n] = rows
        sh.blk[0], sh.blk[1] = p0, n
        self.command(GT.SEG_BLOCK_BEGIN, -1)
        for t in range(n):                                # block segment 0: K; layer i: B R D E F = 1 + 5 i + 0..4
            self.command(GT.BLK, t)
        for i in range(self.L):
            for t in range(n):
                self.command(GT.BLK + 1 + 5 * i, t)
                self.command(GT.BLK + 2 + 5 * i, t)
                self.command(GT.BLK + 5 + 5 * i, t)
        self.command(GT.SEG_BLOCK_END, -1)

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
            if s < last and s % 4 >= 2:                # D and E run inside the R command
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
        groups = (("attention", [s for s in range(last) if s % 4 == 0]),
                  ("router_and_ffn", [s for s in range(last) if s % 4 == 1]),
                  ("lm_head", [last]))
        return {"segments": [{"name": g, "count": len(sel), "host_ms": float(sw[sel].mean()),
                              "per_token_ms": float(sw[sel].sum())} for g, sel in groups],
                "host_ms_per_token": float(sw.sum())}

    def close(self):
        try:
            self.embed.close()
        except Exception:
            pass
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
