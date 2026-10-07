"""Controller-private weight shards for models with unrolled M=1 decoders.

The scheduler owns allocation, weight copying, and four-phase rendezvous. This
adapter only matches a model's original projection addresses to those shards
and emits the existing IF4/IF8 kernel into matching primary/worker streams.
Prefill continues using the original full weight image.
"""

from __future__ import annotations

import time

import user_dma_core
from multi_engine_shard import MultiEngineScheduler, PrivateArena, can_split


def reset_engine_queues(num_engines: int, timeout: float = 1.0) -> None:
    """Reset reported engine queues without touching model DRAM or running BIST.

    Models already read HW_INFO before this call. The low-level full software
    reset also invokes a version-pinned DRAM self-test; queue recovery requires
    only the reset command and an observed idle queue on each selected engine.
    """
    if user_dma_core.ANDROMEDA_CORE_COUNT is None:
        user_dma_core.configure_clock_from_hardware()
    available = user_dma_core.ANDROMEDA_CORE_COUNT
    if not isinstance(available, int) or not 1 <= num_engines <= available:
        raise ValueError(f"requested {num_engines} engines; HW_INFO reports {available}")
    if timeout <= 0:
        raise ValueError("queue reset timeout must be positive")
    engines = [user_dma_core.UnifiedEngine(
        BASE_ADDR=user_dma_core.UE_0_BASE_ADDR + index * 0x10000,
        init_unified_engine=False) for index in range(num_engines)]
    for engine in engines:
        engine.write_reg32(user_dma_core.UE_QUEUE_CTRL_ADDR, 0x80008000)
    for index, engine in enumerate(engines):
        deadline = time.monotonic() + timeout
        while engine.is_queue_busy():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"engine {index} remains busy after queue reset")
            time.sleep(0.01)


class ControllerShardedDecoder:
    def __init__(self, primary, num_engines: int, windows):
        self.scheduler = MultiEngineScheduler(
            primary, num_engines=num_engines, handshake="four_phase",
            arena=PrivateArena(num_engines, windows=windows), verbose=True)
        self.scheduler.reset_workers()
        self.weights = {}
        self._sources = {}
        self._worker_programs = None

    def add_weight(self, name: str, main_weight_addr: int, main_scale_addr: int,
                   K: int, N: int, layers: int, main_layer_stride: int,
                   data_type=user_dma_core.TYPE.IF4):
        if not can_split(N, self.scheduler.num_engines):
            print(f"  {name}: N={N} has too few 64-column blocks for "
                  f"{self.scheduler.num_engines} engines; primary keeps this projection")
            return None
        if not 0 < K <= user_dma_core.SCALE_BRAM_ELEMENTS:
            raise ValueError(f"{name}: decode sharding requires K <= SCALE_BRAM_ELEMENTS")
        if layers < 1 or (layers > 1 and main_layer_stride <= 0):
            raise ValueError(f"{name}: invalid layer count or stride")
        sources = [main_weight_addr + layer * main_layer_stride for layer in range(layers)]
        if any(address in self._sources for address in sources):
            raise ValueError(f"{name}: original weight addresses already registered")
        sw = self.scheduler.shard_quantized_weight(
            name=name, main_weight_addr=main_weight_addr,
            main_scale_addr=main_scale_addr, K=K, N=N, layers=layers,
            main_layer_stride=main_layer_stride, data_type=data_type)
        self.weights[name] = sw
        for layer, address in enumerate(sources):
            self._sources[address] = (sw, layer, main_scale_addr + layer * main_layer_stride)
        return sw

    def begin(self) -> None:
        self.scheduler.begin_program()
        self.scheduler.primary.generate_instruction_flag_clear()
        for engine in self.scheduler.workers:
            engine.generate_instruction_flag_clear()

    def projection(self, *, M: int, K: int, N: int, A_DRAM_ADDR: int,
                   B_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, SCALE_DRAM_ADDR: int,
                   data_type=user_dma_core.TYPE.IF4, **extras) -> int | None:
        """Emit a registered projection, or return None for an unregistered weight.

        Each column block is a contiguous output slice at M=1. N-broadcast bias
        is sliced by the same column offset. All engines write their output,
        including LM-head candidates needed by :meth:`global_argmax`.
        """
        match = self._sources.get(B_DRAM_ADDR)
        if match is None:
            return None
        sw, layer, expected_scale = match
        if (M != 1 or K != sw.K or N != sw.N or data_type != sw.data_type
                or SCALE_DRAM_ADDR != expected_scale):
            raise ValueError(f"{sw.name}: projection shape, type, or scale differs from its shard")
        kwargs = dict(extras)
        # Model decoders know M=1 statically. A primary GPR has no meaning on
        # workers; literal-address kernels need no runtime row-count register.
        kwargs.pop("gpr_M_reg", None)
        if any(key.startswith("gpr_") and value is not None for key, value in kwargs.items()):
            raise ValueError("decode sharding requires literal input/output addresses")
        if kwargs.get("C_DRAM_ADDR") is not None and kwargs.get("bias_mode") != "broadcast_N":
            raise ValueError("decode sharding requires broadcast_N bias")
        kwargs["write_back_disable"] = False
        group = self.scheduler
        group.release()
        for index, engine in enumerate(group.engines):
            shard = sw.shard(index)
            if index:
                group.begin_worker_round(index)
            lane_kwargs = dict(kwargs)
            if lane_kwargs.get("C_DRAM_ADDR") is not None:
                lane_kwargs["C_DRAM_ADDR"] += shard.col_offset * 2
            engine.quantized_matmat_core(
                M=1, K=sw.K, N=shard.cols, A_DRAM_ADDR=A_DRAM_ADDR,
                B_DRAM_ADDR=shard.weight_addr + layer * shard.layer_stride,
                SCALE_DRAM_ADDR=shard.scale_addr + layer * shard.scale_layer_stride,
                OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR + shard.col_offset * 2,
                data_type=sw.data_type, **lane_kwargs)
            if index:
                group.end_worker_round(index)
        group.join()
        return 2 * K * N

    def finalize(self) -> list[int]:
        self._worker_programs = self.scheduler.finalize()
        self.scheduler.verify_private_space()
        return self._worker_programs

    def start(self) -> None:
        if self._worker_programs is None:
            raise RuntimeError("compile and finalize decoder workers before starting decode")
        self.scheduler.start_workers(self._worker_programs)

    def wait(self, timeout: float = 10.0) -> None:
        # wait_queue currently prints and returns on timeout, so explicitly
        # inspect BUSY before reading argmax or restarting a worker stream.
        for index, engine in enumerate(self.scheduler.engines):
            engine.wait_queue(timeout)
            if engine.is_queue_busy():
                raise TimeoutError(f"decode engine {index} did not halt within {timeout}s")

    def global_argmax(self, name: str, out_addr: int) -> int:
        return self.scheduler.global_argmax(self.weights[name], out_addr)
