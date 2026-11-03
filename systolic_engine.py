"""
Driver for the systolic IP (user.org:user:systolic:1.0) on kintex7.

CSR register map (offsets from s00_axi BAR base, default 0x02000000):
  0x00  START          write 1 to trigger; auto-clears after one clock cycle
  0x04  USE_MODE       0 = PCIe/DMA master (reads A/B from DDR), 1 = AXI slave/stream
  0x08  CFG_M          M rows   (must be multiple of 8)
  0x0C  CFG_N          N cols   (must be multiple of 8)
  0x10  CFG_K          inner K  (must be multiple of 8)
  0x14  A_ADDR_LSB     A matrix DDR base [31:0]
  0x18  A_ADDR_MSB     A matrix DDR base [63:32]
  0x1C  B_ADDR_LSB     B matrix DDR base [31:0]   (stored as [N,K])
  0x20  B_ADDR_MSB     B matrix DDR base [63:32]
  0x24  C_ADDR_LSB     output C DDR destination [31:0]
  0x28  C_ADDR_MSB     output C DDR destination [63:32]
  0x2C  DONE           sticky 1 when computation completes; clears on next START
  0x30  COMPUTE_CYCLES cycle count (valid after DONE asserts)
  0x34  DST_ADDR_LSB   secondary destination [31:0]
  0x38  DST_ADDR_MSB   secondary destination [63:32]

Memory layout (PCIe master mode, use_mode=0):
  A: [M, K] BF16 row-major at A_ADDR
  B: [N, K] BF16 row-major at B_ADDR   → hardware computes C = A @ B.T
  C: [M, N] BF16 row-major written to C_ADDR
"""

import math
import os
import struct
import time

import torch
import user_dma_core


class SystolicEngine:
    # CSR offsets from s00_axi base
    REG_START          = 0x00
    REG_USE_MODE       = 0x04
    REG_CFG_M          = 0x08
    REG_CFG_N          = 0x0C
    REG_CFG_K          = 0x10
    REG_A_ADDR_LSB     = 0x14
    REG_A_ADDR_MSB     = 0x18
    REG_B_ADDR_LSB     = 0x1C
    REG_B_ADDR_MSB     = 0x20
    REG_C_ADDR_LSB     = 0x24
    REG_C_ADDR_MSB     = 0x28
    REG_DONE           = 0x2C
    REG_COMPUTE_CYCLES = 0x30
    REG_DST_ADDR_LSB   = 0x34
    REG_DST_ADDR_MSB   = 0x38

    def __init__(self, csr_base: int = user_dma_core.UE_0_BASE_ADDR):
        self._base = csr_base  # BAR byte offset of s00_axi

    # ------------------------------------------------------------------
    # Raw CSR access via /dev/xdma*_user
    # ------------------------------------------------------------------

    def _read(self, offset: int) -> int:
        try:
            fd = os.open(user_dma_core.DMA_DEVICE_USER, os.O_RDONLY)
            try:
                raw = os.pread(fd, 4, self._base + offset)
                return struct.unpack(user_dma_core.REG32_FORMAT, raw)[0] if len(raw) == 4 else 0
            finally:
                os.close(fd)
        except OSError as e:
            print(f"SystolicEngine: CSR read @{offset:#x} failed: {e}")
            return 0

    def _write(self, offset: int, value: int):
        try:
            fd = os.open(user_dma_core.DMA_DEVICE_USER, os.O_RDWR)
            try:
                os.pwrite(fd, struct.pack(user_dma_core.REG32_FORMAT, value & 0xFFFFFFFF),
                          self._base + offset)
            finally:
                os.close(fd)
        except OSError as e:
            print(f"SystolicEngine: CSR write @{offset:#x}={value:#x} failed: {e}")

    # ------------------------------------------------------------------
    # DMA helpers
    # ------------------------------------------------------------------

    def h2c(self, ddr_addr: int, tensor: torch.Tensor):
        """Transfer BF16 tensor from host to DDR via XDMA H2C channel."""
        data = user_dma_core.tensor_le_bytes(tensor)
        try:
            fd = os.open(user_dma_core.DMA_DEVICE_H2C, os.O_RDWR)
            try:
                n = os.pwrite(fd, data, ddr_addr)
                if n != len(data):
                    print(f"SystolicEngine: H2C short write {n}/{len(data)} bytes")
            finally:
                os.close(fd)
        except OSError as e:
            print(f"SystolicEngine: H2C DMA to {ddr_addr:#x} failed: {e}")

    def c2h(self, ddr_addr: int, shape: tuple) -> torch.Tensor:
        """Read BF16 tensor from DDR via XDMA C2H channel."""
        n_bytes = math.prod(shape) * 2
        try:
            fd = os.open(user_dma_core.DMA_DEVICE_C2H, os.O_RDONLY)
            try:
                raw = os.pread(fd, n_bytes, ddr_addr)
            finally:
                os.close(fd)
        except OSError as e:
            print(f"SystolicEngine: C2H DMA from {ddr_addr:#x} failed: {e}")
            return torch.zeros(shape, dtype=torch.bfloat16)
        return user_dma_core.tensor_from_le_bytes(raw, torch.bfloat16).reshape(shape)

    # ------------------------------------------------------------------
    # Matrix multiply
    # ------------------------------------------------------------------

    def matmul(self, A_addr: int, B_addr: int, C_addr: int,
               M: int, K: int, N: int,
               timeout: float = 30.0) -> int:
        """
        Compute C[M,N] = A[M,K] @ B[N,K].T using the systolic array.

        A[M,K] BF16 row-major at A_addr in DDR.
        B[N,K] BF16 row-major at B_addr in DDR  (hardware transposes internally).
        Result C[M,N] BF16 written to C_addr in DDR.

        Returns compute_cycles from hardware (0 on timeout).
        """
        assert M % 8 == 0 and N % 8 == 0 and K % 8 == 0, \
            f"M/N/K must be multiples of 8, got M={M} N={N} K={K}"

        self._write(self.REG_USE_MODE,     1)            # AXI master mode: IP pulls A/B from DDR via m00_axi
        self._write(self.REG_CFG_M,        M)
        self._write(self.REG_CFG_N,        N)
        self._write(self.REG_CFG_K,        K)
        self._write(self.REG_A_ADDR_LSB,   A_addr & 0xFFFFFFFF)
        self._write(self.REG_A_ADDR_MSB,   (A_addr >> 32) & 0xFFFFFFFF)
        self._write(self.REG_B_ADDR_LSB,   B_addr & 0xFFFFFFFF)
        self._write(self.REG_B_ADDR_MSB,   (B_addr >> 32) & 0xFFFFFFFF)
        self._write(self.REG_C_ADDR_LSB,   C_addr & 0xFFFFFFFF)
        self._write(self.REG_C_ADDR_MSB,   (C_addr >> 32) & 0xFFFFFFFF)
        self._write(self.REG_DST_ADDR_LSB, C_addr & 0xFFFFFFFF)
        self._write(self.REG_DST_ADDR_MSB, (C_addr >> 32) & 0xFFFFFFFF)

        # Verify CSR writes reached hardware before triggering.
        rb_use  = self._read(self.REG_USE_MODE)
        rb_m    = self._read(self.REG_CFG_M)
        rb_n    = self._read(self.REG_CFG_N)
        rb_k    = self._read(self.REG_CFG_K)
        rb_alb  = self._read(self.REG_A_ADDR_LSB)
        rb_blb  = self._read(self.REG_B_ADDR_LSB)
        rb_clb  = self._read(self.REG_C_ADDR_LSB)
        print(f"  CSR readback: USE_MODE={rb_use} M={rb_m} N={rb_n} K={rb_k} "
              f"A={rb_alb:#010x} B={rb_blb:#010x} C={rb_clb:#010x}")
        if rb_use != 1:
            print(f"  WARNING: USE_MODE write failed! Got {rb_use}, expected 1")
        if rb_m != M or rb_n != N or rb_k != K:
            print(f"  WARNING: CSR write mismatch! "
                  f"Expected M={M} N={N} K={K}, got M={rb_m} N={rb_n} K={rb_k}")

        self._write(self.REG_START, 1)  # hardware auto-clears after one cycle

        deadline = time.time() + timeout
        last_log = time.time()
        while time.time() < deadline:
            done = self._read(self.REG_DONE)
            if done:
                return self._read(self.REG_COMPUTE_CYCLES)
            now = time.time()
            if now - last_log >= 5.0:
                elapsed = now - (deadline - timeout)
                print(f"  polling: DONE={done} COMPUTE_CYCLES={self._read(self.REG_COMPUTE_CYCLES):#x} "
                      f"elapsed={elapsed:.1f}s")
                last_log = now
            time.sleep(0.001)

        print(f"SystolicEngine.matmul: timeout after {timeout:.1f}s "
              f"(M={M} N={N} K={K})")
        print(f"  final CSR: USE_MODE={self._read(self.REG_USE_MODE)} "
              f"DONE={self._read(self.REG_DONE)} "
              f"CYCLES={self._read(self.REG_COMPUTE_CYCLES):#x} "
              f"CFG_M={self._read(self.REG_CFG_M)} CFG_N={self._read(self.REG_CFG_N)} "
              f"CFG_K={self._read(self.REG_CFG_K)}")
        self._post_timeout_dma_check()
        return 0

    def _post_timeout_dma_check(self):
        """After a matmul timeout, probe DDR0 and DDR1 via XDMA C2H to check
        whether the SmartConnect is deadlocked (both fail) or DDR0-specific."""
        import struct as _struct
        probe_addrs = [
            (0x00100000, "DDR0 (0x00100000 via M00_AXI)"),
            (0x80000000, "DDR1 (0x80000000 via M01_AXI)"),
        ]
        print("  post-timeout DMA probe:")
        for addr, label in probe_addrs:
            try:
                fd = os.open(user_dma_core.DMA_DEVICE_C2H, os.O_RDONLY)
                try:
                    raw = os.pread(fd, 32, addr)
                    ok = len(raw) == 32
                finally:
                    os.close(fd)
                print(f"    C2H {label}: {'OK' if ok else 'short read'} "
                      f"({len(raw)} bytes)")
            except OSError as e:
                print(f"    C2H {label}: FAILED: {e}")

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def dump_csrs(self):
        """Print all CSRs (useful for bring-up and debug)."""
        regs = [
            (self.REG_START,          "START"),
            (self.REG_USE_MODE,       "USE_MODE"),
            (self.REG_CFG_M,          "CFG_M"),
            (self.REG_CFG_N,          "CFG_N"),
            (self.REG_CFG_K,          "CFG_K"),
            (self.REG_A_ADDR_LSB,     "A_ADDR_LSB"),
            (self.REG_A_ADDR_MSB,     "A_ADDR_MSB"),
            (self.REG_B_ADDR_LSB,     "B_ADDR_LSB"),
            (self.REG_B_ADDR_MSB,     "B_ADDR_MSB"),
            (self.REG_C_ADDR_LSB,     "C_ADDR_LSB"),
            (self.REG_C_ADDR_MSB,     "C_ADDR_MSB"),
            (self.REG_DONE,           "DONE"),
            (self.REG_COMPUTE_CYCLES, "COMPUTE_CYCLES"),
            (self.REG_DST_ADDR_LSB,   "DST_ADDR_LSB"),
            (self.REG_DST_ADDR_MSB,   "DST_ADDR_MSB"),
        ]
        print(f"SystolicEngine CSR dump (base={self._base:#010x}):")
        for off, name in regs:
            print(f"  [{off:#04x}] {name:<20s} = {self._read(off):#010x}")
