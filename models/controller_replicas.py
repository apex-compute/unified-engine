"""Immutable weight replicas for models retaining a fixed low-4-GiB layout.

These replicas serve existing row-sharded kernels. They do not relocate ISA,
activations, or caches, and cannot free a second controller on a full Kintex map.
"""
from __future__ import annotations

import user_dma_core as core
from multi_engine_shard import board_private_windows


def free_controller_windows(num_engines: int) -> list[tuple[int, int]]:
    """Free HBM controller segments outside the model's reserved [0, 4 GiB)."""
    if num_engines < 1:
        raise ValueError('engine count must be positive')
    if num_engines == 1:
        return []
    if core.ANDROMEDA_CORE_COUNT is None or core.AVAILABLE_DRAM_SIZE_GB is None:
        core.configure_clock_from_hardware()
    cores, gib = core.ANDROMEDA_CORE_COUNT, core.AVAILABLE_DRAM_SIZE_GB
    if num_engines > cores:
        raise ValueError(f'{num_engines} engines exceed HW_INFO count {cores}')
    if gib < 8:
        return []
    if (cores, gib) == (12, 16):
        board = board_private_windows(num_engines, reserve=(0, 4 << 30))
    else:
        board = board_private_windows(8 if cores == 12 else num_engines)
    return [(base, size) for window in board for base, size in window.segments
            if base >= 4 << 30][:num_engines]


class ControllerReplicas:
    """Lazily copy immutable source blobs; persist enough to restore cached ISA.

    U50 and 16-GiB U55C provide one free region per selected engine. An 8-GiB
    U55C has four free controllers after the shared model reserve, so engines
    share those four replicas cyclically. A 4-GiB device keeps source addresses.
    """
    def __init__(self, primary, num_engines: int):
        self.primary, self.num_engines = primary, int(num_engines)
        self.windows = free_controller_windows(self.num_engines)
        self._cursor = [base for base, _ in self.windows]
        self._copies = {}
        self.records = []
        self.layout = {'version': 1, 'engines': self.num_engines,
                       'windows': [list(w) for w in self.windows]}

    def copy(self, source: int, size: int, *, expected=None) -> list[int]:
        if size <= 0 or source < 0 or source + size > 4 << 30:
            raise ValueError('replica source must fit inside the low-4-GiB model')
        key = (int(source), int(size))
        if key in self._copies:
            addresses = self._copies[key]
            if expected is not None and addresses != expected:
                raise ValueError('cached replica address differs from live allocation')
            return addresses
        if not self.windows:
            addresses = [source] * self.num_engines
            if expected is not None and addresses != expected:
                raise ValueError('cached replica address differs from shared-source layout')
            return addresses
        starts = [(cursor + 63) & ~63 for cursor in self._cursor]
        for start, (base, capacity) in zip(starts, self.windows):
            if start + size > base + capacity:
                raise MemoryError('immutable weights exceed a free controller window')
        addresses = [starts[i % len(starts)] for i in range(self.num_engines)]
        if expected is not None and addresses != expected:
            raise ValueError('cached replica address differs from live allocation')
        # Preflight every destination before the first DMA. Read a source chunk
        # once, then distribute those exact bytes; reject short DMA immediately.
        primary = self.primary
        h2c = getattr(primary, 'h2c_device', core.DMA_DEVICE_H2C)
        c2h = getattr(primary, 'c2h_device', core.DMA_DEVICE_C2H)
        for offset in range(0, size, 1 << 20):
            count = min(1 << 20, size - offset)
            data = bytearray(count)
            if primary.dma_read(c2h, source + offset, data, count) != count:
                raise IOError('short DMA while reading immutable replica source')
            for start in starts:
                if primary.dma_write(h2c, start + offset, data, count) != count:
                    raise IOError('short DMA while writing immutable weight replica')
        self._cursor = [start + size for start in starts]
        self._copies[key] = addresses
        self.records.append([source, size, addresses])
        return addresses

    def address(self, engine: int, source: int, size: int) -> int:
        if not 0 <= engine < self.num_engines:
            raise ValueError('replica engine index out of range')
        return self.copy(source, size)[engine]

    def manifest(self):
        return {'layout': self.layout, 'copies': self.records}

    def restore(self, manifest):
        if manifest is None:
            if self.windows:
                raise ValueError('cached programs lack controller replicas; rebuild them')
            return
        if manifest.get('layout') != self.layout:
            raise ValueError('cached controller layout changed; rebuild programs')
        for source, size, addresses in manifest.get('copies', []):
            self.copy(int(source), int(size), expected=addresses)
