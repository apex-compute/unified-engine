"""Descriptor caching with established scalar DMA I/O for native streaming."""

from __future__ import annotations

import os
from pathlib import Path
import struct
import sys
import threading

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from user_dma_core import UnifiedEngine


class StreamingEngine(UnifiedEngine):
    """Cache DMA handles while retaining UnifiedEngine's buffer conversions.

    Transfers use only seek/read/write, serialized because seeking changes the
    cached descriptor's position. Readback keeps the shared engine's temporary
    byte buffer and tensor copy. Vector I/O is deliberately never used.
    Call ``close()`` when finished; it also releases the inherited user handle.
    """

    def __init__(self, *args, **kwargs):
        # Superclass initialization can call the overridden DMA methods.
        self._dma_lock = threading.RLock()
        self._dma_fds = {}
        self._dma_closed = False
        self._user_fd = None
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _write_bytes(buffer, size):
        """Match the shared engine's tensor, integer and byte conversions."""
        if isinstance(buffer, torch.Tensor):
            tensor_bytes = buffer.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            return memoryview(tensor_bytes.numpy()).cast("B")[:size]
        if isinstance(buffer, int):
            if size == 4:
                return struct.pack("I", buffer & 0xFFFFFFFF)
            if size == 2:
                return struct.pack("H", buffer & 0xFFFF)
            return buffer.to_bytes(size, byteorder="little", signed=False)
        if isinstance(buffer, (bytes, bytearray)):
            return memoryview(buffer)[:size]
        return bytes(buffer)[:size]

    @staticmethod
    def _copy_read_bytes(buffer, data_bytes, size):
        """Preserve byte-exact copies, short reads and noncontiguous tensors."""
        if isinstance(buffer, torch.Tensor):
            if buffer.device.type == "cpu" and buffer.is_contiguous():
                target = memoryview(buffer.detach().view(torch.uint8).numpy()).cast("B")
                take = min(len(data_bytes), len(target), size)
                target[:take] = data_bytes[:take]
            else:
                staged = buffer.detach().cpu().contiguous()
                target = memoryview(staged.view(torch.uint8).numpy()).cast("B")
                take = min(len(data_bytes), len(target), size)
                target[:take] = data_bytes[:take]
                with torch.no_grad():
                    buffer.copy_(staged.to(buffer.device))
        elif hasattr(buffer, "__setitem__"):
            buffer[:len(data_bytes)] = data_bytes

    def _get_dma_fd(self, key):
        fd = self._dma_fds.get(key)
        if fd is None:
            # Match UnifiedEngine's access flags for XDMA channel devices.
            fd = os.open(key[0], os.O_RDWR)
            self._dma_fds[key] = fd
        return fd

    def _discard_dma_fd(self, key):
        fd = self._dma_fds.pop(key, None)
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    def _dma_transfer(self, device, address, buffer, size, *, read):
        direction = "read" if read else "write"
        key = (os.fspath(device), direction)
        with self._dma_lock:
            try:
                if self._dma_closed:
                    raise RuntimeError("streaming engine is closed")
                data_bytes = None if read else self._write_bytes(buffer, size)
                fd = self._get_dma_fd(key)
                os.lseek(fd, address, os.SEEK_SET)
                # Keep the established scalar syscalls: the target host's
                # vector I/O path caused a kernel BUG during candidate probes.
                if not read:
                    return os.write(fd, data_bytes)
                data_bytes = os.read(fd, size)
                if len(data_bytes) == 0:
                    print("read failed: no data read")
                    return -1
                self._copy_read_bytes(buffer, data_bytes, size)
                return len(data_bytes)
            except Exception as error:
                self._discard_dma_fd(key)
                print(f"dma_{direction} error: {error}")
                return -1

    def dma_write(self, device: str, address: int, buffer, size: int) -> int:
        return self._dma_transfer(device, address, buffer, size, read=False)

    def dma_read(self, device: str, address: int, buffer, size: int) -> int:
        return self._dma_transfer(device, address, buffer, size, read=True)

    def _get_user_fd(self):
        with self._dma_lock:
            if self._dma_closed:
                raise RuntimeError("streaming engine is closed")
            return super()._get_user_fd()

    def close(self):
        """Release all owned descriptors exactly once, including on errors."""
        lock = getattr(self, "_dma_lock", None)
        if lock is None:
            return
        with lock:
            self._dma_closed = True
            for key in tuple(self._dma_fds):
                self._discard_dma_fd(key)
            fd = getattr(self, "_user_fd", None)
            self._user_fd = None
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def __enter__(self):
        if self._dma_closed:
            raise RuntimeError("streaming engine is closed")
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Python may already have torn down modules during finalization.
            pass
