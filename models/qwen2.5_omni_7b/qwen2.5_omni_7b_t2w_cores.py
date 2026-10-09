#!/usr/bin/env python3
"""Bulk-synchronous eight-engine runner for the Token2Wav stage.

Each *region* is one captured program per engine. The host starts every engine's
program for a region and waits for all of them before launching the next region,
so a region boundary is a global barrier: everything an engine wrote is complete
and visible to every other engine afterwards. Engines read and write one shared
DRAM address space, so a region partitions its output (rows, or head/branch
planes) disjointly and the next region may read any engine's results.

This is deliberately not MultiEngineScheduler: it needs no device-side flags or
workers, only a register write to launch and a register read to poll, and it has
no private-arena accounting to satisfy -- Token2Wav runs after the Thinker and
Talker are finished and owns the whole 8 GiB board.

Address map (1 GiB window per engine, W = 7):
  engine e programs   e*GiB + 0x01000000   (engine 7: + 0x29000000)
  engine e scratch    e*GiB + 0x10000000   (engine 7: + 0x38000000); its constants +0x06000000
  weights             7*GiB + 0x00000000 ..
  shared pool         windows 0..6, e*GiB + 0x20000000 .. (e+1)*GiB   (activations; rewound per stage)
"""

from __future__ import annotations

import os
import time

import torch

import user_dma_core
from user_dma_core import DMA_DEVICE_H2C, UnifiedEngine

GIB = 1 << 30
ENGINE_STRIDE = 0x00010000
WEIGHT_BASE = 7 * GIB
POOL_SEGMENTS = [(e * GIB + 0x20000000, (e + 1) * GIB) for e in range(7)]
PROGRAM_OFFSET = 0x01000000         # no engine's program starts at DRAM address 0
PROGRAM_BYTES = 0x0F000000          # 240 MiB of program space per engine
POLL_S = float(os.environ.get("OMNI_T2W_POLL_US", "100")) * 1e-6


def _bf16_bytes(t: torch.Tensor) -> bytes:
    return t.detach().to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()


def legacy_memory_map(n: int) -> dict:
    """The standalone whole-board map: weights in window 7, the rest per window."""
    progs, scratch, consts = [], [], []
    for e in range(n):
        prog = e * GIB + (0x28000000 if e == 7 else 0) + PROGRAM_OFFSET
        progs.append((prog, prog + PROGRAM_BYTES))
        s = e * GIB + (0x38000000 if e == 7 else 0x10000000)
        scratch.append(s)
        consts.append(s + 0x06000000)
    return {"weights_base": WEIGHT_BASE, "pool_segments": list(POOL_SEGMENTS),
            "programs": progs, "scratch": scratch, "consts": consts}


class HostImage:
    """A sparse DRAM image in host memory.

    Everything an engine would DMA to the board lands here instead, so Token2Wav's
    weights and programs can be built (and stored) with no hardware. Writes of all
    zeros are dropped because unwritten bytes read as zero anyway.
    """

    PAGE = 1 << 20

    def __init__(self) -> None:
        self.pages: dict[int, bytearray] = {}

    def write(self, address: int, data: bytes) -> None:
        if not any(data):
            return
        end = address + len(data)
        pos = address
        while pos < end:
            page, off = divmod(pos, self.PAGE)
            n = min(self.PAGE - off, end - pos)
            buf = self.pages.get(page)
            if buf is None:
                buf = self.pages[page] = bytearray(self.PAGE)
            buf[off:off + n] = data[pos - address:pos - address + n]
            pos += n

    def read(self, address: int, size: int) -> bytes:
        out = bytearray(size)
        pos, end = address, address + size
        while pos < end:
            page, off = divmod(pos, self.PAGE)
            n = min(self.PAGE - off, end - pos)
            buf = self.pages.get(page)
            if buf is not None:
                out[pos - address:pos - address + n] = buf[off:off + n]
            pos += n
        return bytes(out)


class _HostImageEngine(UnifiedEngine):
    """An engine whose DMA goes to a HostImage (no board access at all)."""

    def __init__(self, *args, image: HostImage, **kwargs):
        super().__init__(*args, **kwargs)
        self._image = image

    def dma_write(self, device, address, buffer, size):
        if isinstance(buffer, torch.Tensor):
            if buffer.dtype == torch.bfloat16:
                data = buffer.view(torch.uint16).cpu().contiguous().numpy().tobytes()
            else:
                data = buffer.cpu().contiguous().numpy().tobytes()
        else:
            data = bytes(buffer)
        self._image.write(int(address), data[:size])
        return size

    def dma_read(self, device, address, buffer, size):
        buffer[:size] = self._image.read(int(address), size)
        return size


class Cores:
    """``memory_map`` places weights, activations, programs and per-engine scratch.

    Without one, Token2Wav owns the whole board (legacy_memory_map). With a
    layout's map it lives inside the regions that layout reserved for it, so
    nothing a previous stage left resident is overwritten.
    """

    hardware = True            # False: engines write to a HostImage (see ImageCores)
    upload_weights = True      # False: put_weight only assigns the address

    def _make_engine(self, e: int, **kwargs) -> UnifiedEngine:
        return UnifiedEngine(
            BASE_ADDR=user_dma_core.UE_0_BASE_ADDR + e * ENGINE_STRIDE,
            init_unified_engine=False, **kwargs)

    def __init__(self, num_engines: int = 8, verbose: bool = True,
                 memory_map: dict | None = None):
        self.n = num_engines
        self.verbose = verbose
        mm = memory_map or legacy_memory_map(num_engines)
        self.pool_segments = [tuple(s) for s in mm["pool_segments"]]
        self.weights_limit = mm.get("weights_limit")
        self.engines: list[UnifiedEngine] = []
        for e in range(num_engines):
            prog, prog_limit = mm["programs"][e]
            scratch = mm["scratch"][e]
            ue = self._make_engine(
                e, params_dram_base=mm["weights_base"] if e == 0 else mm["consts"][e],
                program_dram_base=prog, tensor_dram_base=scratch)
            ue._prog_limit = prog_limit
            self.engines.append(ue)
        self.regions: list[list[tuple[int, int]]] = []
        self.names: list[str] = []
        self._pool = [lo for lo, _hi in self.pool_segments]
        self.device_s = 0.0                    # wall time engines were running regions
        self.pool_peak = 0
        if self.hardware:
            self._reset_all()

    # ---------------------------------------------------------------- hardware
    def _reset_all(self) -> None:
        for ue in self.engines:
            ue.write_reg32(user_dma_core.UE_QUEUE_CTRL_ADDR, 0x80008000)
        deadline = time.monotonic() + 3.0
        for e, ue in enumerate(self.engines):
            while ue.is_queue_busy() and time.monotonic() < deadline:
                time.sleep(0.001)
            if ue.is_queue_busy():
                raise TimeoutError(f"engine {e} still busy after software reset")

    def run_region(self, rid: int, timeout: float = 120.0) -> None:
        progs = self.regions[rid]
        started = time.perf_counter()
        for ue, (addr, _size) in zip(self.engines, progs):
            ue.start_execute_from_dram(addr)
        deadline = time.monotonic() + timeout
        for e, ue in enumerate(self.engines):
            while ue.is_queue_busy():
                # A device that has dropped off PCIe reads all ones, which sets the busy
                # bit: that is a lost link, not a slow engine, so stop touching the board.
                if ue.read_reg32(user_dma_core.UE_QUEUE_CTRL_ADDR) & 0xFFFFFFFF == 0xFFFFFFFF:
                    raise RuntimeError(
                        f"region {rid} ({self.names[rid]}): the FPGA stopped answering "
                        f"(registers read 0xffffffff); the PCIe link is down and the board "
                        f"needs a flash reboot + PCIe rescan")
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"region {rid} ({self.names[rid]}): engine {e} still busy "
                        f"after {timeout:.0f}s")
                # Poll at ~10 kHz, not in a tight loop: a region takes milliseconds,
                # and two register reads per spin put a continuous read storm on the
                # link for the minute a Token2Wav run lasts (the link has dropped
                # mid-run; this is a mitigation, the cause is not known).
                time.sleep(POLL_S)
        self.device_s += time.perf_counter() - started

    def run(self, rids) -> None:
        for rid in rids:
            self.run_region(rid)

    # ------------------------------------------------------------------ memory
    @property
    def weights(self) -> UnifiedEngine:
        return self.engines[0]

    def put_weight(self, t: torch.Tensor) -> int:
        blob = _bf16_bytes(t)
        addr = self.engines[0].allocate_params_dram(len(blob))
        if not self.upload_weights:
            return addr          # the bytes are already in DRAM (loaded from params.bin)
        if self.weights_limit is not None and addr + len(blob) > self.weights_limit:
            raise MemoryError(
                f"t2w cores: weights need {(addr + len(blob) - self.engines[0]._params_dram_base) / 2**20:.1f} MiB, "
                f"past the {(self.weights_limit - self.engines[0]._params_dram_base) / 2**20:.0f} MiB pool")
        if self.engines[0].dma_write(DMA_DEVICE_H2C, addr, blob, len(blob)) != len(blob):
            raise IOError("t2w cores: short weight DMA")
        return addr

    def alloc(self, nbytes: int, align: int = 64) -> int:
        """First-fit bump allocation over the pool segments (each buffer is contiguous)."""
        for i, (_lo, hi) in enumerate(self.pool_segments):
            a = (self._pool[i] + align - 1) // align * align
            if a + nbytes <= hi:
                self._pool[i] = a + nbytes
                self.pool_peak = max(self.pool_peak, self.pool_used())
                return a
        raise MemoryError(f"t2w cores: shared pool exhausted ({nbytes / 2**20:.1f} MiB)")

    def pool_mark(self) -> tuple:
        return tuple(self._pool)

    def pool_rewind(self, mark: tuple) -> None:
        self._pool = list(mark)

    def pool_used(self) -> int:
        return sum(p - lo for p, (lo, _hi) in zip(self._pool, self.pool_segments))

    def write(self, addr: int, t: torch.Tensor) -> None:
        blob = _bf16_bytes(t)
        if self.engines[0].dma_write(DMA_DEVICE_H2C, addr, blob, len(blob)) != len(blob):
            raise IOError("t2w cores: short DMA write")

    def zero(self, addr: int, nbytes: int) -> None:
        step = 64 * 1024 * 1024
        for off in range(0, nbytes, step):
            n = min(step, nbytes - off)
            self.engines[0].dma_write(DMA_DEVICE_H2C, addr + off, bytes(n), n)

    def read(self, addr: int, n_elem: int) -> torch.Tensor:
        ue = self.engines[0]
        buf = bytearray(n_elem * 2)
        if ue.dma_read(ue.c2h_device, addr, buf, len(buf)) != len(buf):
            raise IOError("t2w cores: short DMA read")
        return torch.frombuffer(bytes(buf), dtype=torch.bfloat16).clone()

    # ----------------------------------------------------------------- capture
    def region(self, name: str, emit) -> int:
        """Capture ``emit(e, ue)`` on every engine as one region; returns its id."""
        progs = []
        for e, ue in enumerate(self.engines):
            ue.reset_isa_reg_counter()
            ue.reset_inst_ptr_counter()
            ue.clear_inst_id()
            ue.clear_capture_buffer()
            scratch_mark = ue._tensor_dram_addr
            ue.start_capture()
            emit(e, ue)
            ue.generate_instruction_halt()
            ue.stop_capture()
            size = ue.get_capture_instruction_size_bytes()
            addr = ue.get_program_dram_addr()
            if addr + size > ue._prog_limit:
                raise MemoryError(f"t2w cores: engine {e} program space exhausted in {name}")
            if ue.write_captured_instructions_to_dram(addr) != size:
                raise IOError(f"t2w cores: {name} engine {e} program DMA short write")
            ue.allocate_program_dram(size)
            ue.clear_capture_buffer()
            ue._tensor_dram_addr = scratch_mark       # region scratch is reusable
            progs.append((addr, size))
        self.regions.append(progs)
        self.names.append(name)
        return len(self.regions) - 1

    def program_bytes(self) -> int:
        return sum(s for progs in self.regions for _a, s in progs)


def footprint_lines(cores: "Cores") -> list[str]:
    """What Token2Wav needs resident, in the units a DRAM map is planned in."""
    mib = 2**20
    progs = [sum(progs[e][1] for progs in cores.regions) for e in range(cores.n)]
    w = cores.engines[0]
    consts = [ue._next_params_dram_addr - ue._params_dram_base
              for e, ue in enumerate(cores.engines)]
    return [
        f"  [T2W] footprint: weights {w.get_params_dram_usage() / mib:.1f} MiB, "
        f"activation pool peak {cores.pool_peak / mib:.1f} MiB, "
        f"programs {max(progs) / mib:.1f} MiB/engine max "
        f"({sum(progs) / mib:.1f} MiB total, {len(cores.regions)} regions), "
        f"per-engine constants {max(consts[1:]) / mib:.1f} MiB",
    ]


def compile_state(cores: "Cores") -> dict:
    """What compiling left in the runner: region table, pool and cursors."""
    return {
        "regions": cores.regions, "names": cores.names, "pool": list(cores._pool),
        "params_cursor": [ue._next_params_dram_addr for ue in cores.engines],
        "program_cursor": [ue._next_program_dram_addr for ue in cores.engines],
    }


def restore_state(cores: "Cores", state: dict) -> None:
    cores.regions = state["regions"]
    cores.names = state["names"]
    cores._pool = list(state["pool"])
    for ue, params, prog in zip(cores.engines, state["params_cursor"], state["program_cursor"]):
        ue._next_params_dram_addr = params
        ue._next_program_dram_addr = prog


class ImageCores(Cores):
    """Cores whose engines write to a HostImage: builds the whole Token2Wav
    weight set and program set on the CPU, for params.bin and programs.bin."""

    hardware = False

    def __init__(self, num_engines: int = 8, verbose: bool = True,
                 memory_map: dict | None = None):
        self.image = HostImage()
        super().__init__(num_engines, verbose, memory_map)

    def _make_engine(self, e: int, **kwargs) -> UnifiedEngine:
        return _HostImageEngine(
            BASE_ADDR=user_dma_core.UE_0_BASE_ADDR + e * ENGINE_STRIDE,
            init_unified_engine=False, image=self.image, **kwargs)


def weight_images(cores: "Cores") -> list[tuple[int, int, int]]:
    """(engine, offset from that engine's params base, size) of what was uploaded."""
    out = []
    for e, ue in enumerate(cores.engines):
        size = ue._next_params_dram_addr - ue._params_dram_base
        if size > 0:
            out.append((e, 0, size))
    return out


def program_ranges(cores: "Cores", memory_map: dict) -> list[tuple[int, int, int]]:
    """(engine, address, size) of every engine's contiguous program image."""
    out = []
    for e, ue in enumerate(cores.engines):
        base = int(memory_map["programs"][e][0])
        out.append((e, base, ue.get_program_dram_addr() - base))
    return out
