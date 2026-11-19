#!/usr/bin/env python3
"""edge_device.DeviceRunner with a frame cache in front of the vision stage (A2E_VISION_CACHE=1).

A plan looks at 22 camera frames, and most of them were in an earlier plan too: the frame that
is "now" in one plan is "0.5 s ago", "1 s ago" and "2 s ago" in later ones.

    time (s)        0.0   0.5   1.0   1.5   2.0   2.4  2.5
    front camera     A     B     C     D     E     f    G
    plan at 2.5:    -2.0 = B   -1.0 = D   -0.5 = E   -0.1 = f   now = G
                     cache      cache      cache      NEW        NEW

The vision tower and the projector treat every frame on its own, so a frame's image tokens are
the same in every plan it appears in. This runner keeps them, keyed by the frame's exact pixels:

    frame seen before  ->  its saved image tokens are written into IMG by the host
    new frame          ->  vision + projector run on the board; the tokens are read back and saved

Prefill, decode and the action expert run in full on every plan (the frames change place in the
prompt, so nothing of the LM can be kept). Nothing is approximated: a frame that differs by one
pixel is a new frame.

The first plan after start-up or after a board reset has nothing to reuse and costs the full
time. With one plan per 0.5 s of driving, 10 of the 22 frames repeat from the third plan on
(measured on the closed-loop drives); at 10 Hz it would be 16 of 22.

The board program depends on which frames are new (their image slots, and the engine each runs
on), so one is compiled the first time a set of new frames appears and reused afterwards. All
22 new, or no room left for another program, runs the stock vision and projector programs.

  A2E_VISION_CACHE_PLANS   plans a frame's tokens are kept after their last use (default 24)
  A2E_VISION_CACHE_SETS    most programs to compile for sets of new frames (default 6)

  python models/alpamayo2_edge/sim/edge_device_cached.py replay <dump dir> [plan indices]
"""
import hashlib
import os
import time

import torch

from edge_device import DeviceRunner, cli

LM_RESERVE = 64 << 20       # program memory left alone for the LM programs of prompt lengths not seen yet


class CachedDeviceRunner(DeviceRunner):
    def __init__(self, *args, **kwargs):
        self.keep = int(os.environ.get("A2E_VISION_CACHE_PLANS", 24))
        self.max_sets = int(os.environ.get("A2E_VISION_CACHE_SETS", 6))
        self.enabled = True         # False: every frame goes through the board, nothing is kept
        self.tokens = {}            # frame key -> [image tokens (n, 2048) bf16, plan of last use]
        self.sets = {}              # new image slots (tuple) -> name of the board program for them
        self.plan = 0
        super().__init__(*args, **kwargs)

    # ---------------------------------------------------------------- program memory

    def _workers(self):
        return self.ue.sched.workers if self.ue.sched else []

    def _used(self):
        return [self.ue.get_program_dram_addr()] + [w.get_program_dram_usage() for w in self._workers()]

    def _free(self):
        """Program bytes left on the engine with the least."""
        return min([self.ue.program_limit - self.ue.get_program_dram_addr()]
                   + [self.T.PRIVATE_ISA_BYTES - w.get_program_dram_usage() for w in self._workers()])

    def _compile_vision(self):
        before = self._used()
        super()._compile_vision()       # all 22 frames: the first plan, and the fallback
        self.full_cost = max(a - b for a, b in zip(self._used(), before))       # no smaller set costs more

    # ---------------------------------------------------------------- the program for a set of new frames

    def _program(self, new):
        """Vision + projector for the image slots in `new`, or None when no more programs fit.
        The stock streams on fewer frames (ue.plan_images): an engine encodes its frames and, if
        it has time to spare, projects them; a second round projects what is left."""
        if new in self.sets:
            return self.sets[new]
        if len(self.sets) >= self.max_sets or self._free() < self.full_cost + LM_RESERVE:
            return None
        ue, t0 = self.ue, time.time()
        assign, early, late = ue.plan_images(new)

        def emit():
            ue.emit_vision(assign, early)
            if any(late):
                ue.emit_late(late)

        name = f"vision_new{len(self.sets)}"
        ue._compile(name, emit, parallel=True)
        self.sets[new] = name
        print(f"[device] vision program for {len(new)} new frames (slots {','.join(map(str, new))}) compiled in "
              f"{time.time() - t0:.0f} s; on the engines: " + " | ".join(",".join(map(str, a)) or "-" for a in assign), flush=True)
        return name

    # ---------------------------------------------------------------- one plan's image tokens

    def _vision(self, packed, ms):
        if not self.enabled:
            return super()._vision(packed, ms)
        ue, T = self.ue, self.T
        px = packed["tokenized_data.pixel_values"].float()
        self.plan += 1
        keys = [(im["key"], hashlib.blake2b(px[im["out_off"] : im["out_off"] + im["n"]].contiguous().numpy().tobytes(),
                                            digest_size=16).digest()) for im in ue.images]
        new, known = [], set(self.tokens)
        for i, k in enumerate(keys):        # a frame not seen before; a repeat inside this plan is computed once
            if k not in known:
                known.add(k)
                new.append(i)
        name = self._program(tuple(new)) if 0 < len(new) < len(keys) else None
        if new and name is None:            # every frame is new, or there is no room for another program
            new = list(range(len(keys)))
            super()._vision(packed, ms)
        else:
            if new:
                ue.vision_upload_pixels(px)
            ms[f"vision {len(new)}/{len(keys)}"] = ue._exec(name, timeout=300.0) / 1e3 if new else 0.0

        img, row = ue.buf["IMG"], T.PJ_OUT * T.BPE
        for i in new:                       # keep what the board just computed
            t0, n = ue.image_tokens(i)
            t = ue.dma_from_accelerator_memory(img + t0 * row, (n, T.PJ_OUT)).to(torch.bfloat16)
            assert not bool(torch.isnan(t.float()).any()), "NaN image tokens from the device"
            self.tokens[keys[i]] = [t.contiguous(), self.plan]
        for i in sorted(set(range(len(keys))) - set(new)):      # the rest come from the cache
            entry = self.tokens[keys[i]]
            entry[1] = self.plan
            ue.dma_to_accelerator_memory(img + ue.image_tokens(i)[0] * row, entry[0].flatten())
        self.tokens = {k: v for k, v in self.tokens.items() if v[1] > self.plan - self.keep}


if __name__ == "__main__":
    cli(CachedDeviceRunner)
