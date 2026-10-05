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


def _bf16_bytes(t: torch.Tensor) -> bytes:
    return t.detach().to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()


class Cores:
    def __init__(self, num_engines: int = 8, verbose: bool = True):
        self.n = num_engines
        self.verbose = verbose
        self.engines: list[UnifiedEngine] = []
        for e in range(num_engines):
            prog = e * GIB + (0x28000000 if e == 7 else 0) + PROGRAM_OFFSET
            scratch = e * GIB + (0x38000000 if e == 7 else 0x10000000)
            ue = UnifiedEngine(BASE_ADDR=user_dma_core.UE_0_BASE_ADDR + e * ENGINE_STRIDE,
                               params_dram_base=WEIGHT_BASE if e == 0 else scratch + 0x06000000,
                               program_dram_base=prog, tensor_dram_base=scratch,
                               init_unified_engine=False)
            ue._prog_limit = prog + PROGRAM_BYTES
            self.engines.append(ue)
        self.regions: list[list[tuple[int, int]]] = []
        self.names: list[str] = []
        self._pool = [lo for lo, _hi in POOL_SEGMENTS]
        self.device_s = 0.0                    # wall time engines were running regions
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
        if self.engines[0].dma_write(DMA_DEVICE_H2C, addr, blob, len(blob)) != len(blob):
            raise IOError("t2w cores: short weight DMA")
        return addr

    def alloc(self, nbytes: int, align: int = 64) -> int:
        """First-fit bump allocation over the pool segments (each buffer is contiguous)."""
        for i, (_lo, hi) in enumerate(POOL_SEGMENTS):
            a = (self._pool[i] + align - 1) // align * align
            if a + nbytes <= hi:
                self._pool[i] = a + nbytes
                return a
        raise MemoryError(f"t2w cores: shared pool exhausted ({nbytes / 2**20:.1f} MiB)")

    def pool_mark(self) -> tuple:
        return tuple(self._pool)

    def pool_rewind(self, mark: tuple) -> None:
        self._pool = list(mark)

    def pool_used(self) -> int:
        return sum(p - lo for p, (lo, _hi) in zip(self._pool, POOL_SEGMENTS))

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
