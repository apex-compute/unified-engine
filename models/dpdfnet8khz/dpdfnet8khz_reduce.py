"""Reduce native complex pairs directly into a packed spectrum vector."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import user_dma_core as udc


class ReductionOptimizationMixin:
    """Replace a complex-axis projection and transpose with one matvec."""

    _REDUCE_VECTOR = 0
    _REDUCE_MATRIX = 0x80000
    _REDUCE_OUTPUT = 0x2000

    def emit_reduce_sum(self, index, node):
        source, output = self.layout(node.input[0]), self.layout(node.output[0])
        axes = (self._integer_initializer(node.input[1]).tolist()
                if len(node.input) > 1 else list(range(len(source.shape))))
        axes = [int(axis) % len(source.shape) for axis in axes]
        vector = self.reduce_resources.get(index)
        if (source.logical_last != 2 or source.padded_last != 64
                or source.rows not in (80, 81)
                or output.rows != 1 or output.logical_last != source.rows
                or axes != [len(source.shape) - 1] or vector is None
                or source.address % 128 or output.address % 128
                or self._REDUCE_VECTOR % 128 or self._REDUCE_MATRIX % 128
                or self._REDUCE_OUTPUT % 128
                or self._REDUCE_VECTOR < 0
                or self._REDUCE_VECTOR + 128 > self._REDUCE_OUTPUT
                or self._REDUCE_OUTPUT + output.size_bytes > udc.URAM_NEAR_FULL_SIZE
                or self._REDUCE_MATRIX < 0x80000
                or self._REDUCE_MATRIX - 0x80000 + source.size_bytes > udc.URAM_NEAR_FULL_SIZE):
            return super().emit_reduce_sum(index, node)

        emitter, engine = self.emitter, self.emitter.engine
        if source.name in emitter._zero_padding:
            engine.accelerator_memory_to_sram(
                source.address, self._REDUCE_MATRIX, source.physical_elements)
        else:
            # Zero weights cannot suppress NaN in unselected input lanes.
            # Every four-byte logical pair starts on an aligned physical row.
            engine.accelerator_memory_to_sram(
                emitter.zero_address, self._REDUCE_MATRIX, source.physical_elements)
            for row in range(source.rows):
                engine.accelerator_memory_to_sram(
                    source.address + row * 128, self._REDUCE_MATRIX + row * 128,
                    0, memcpy_length_bytes=4)
        # The shared reduction matrix's first row is [1, 1, 0, ...].
        engine.accelerator_memory_to_sram(vector, self._REDUCE_VECTOR, 64)
        engine.accelerator_memory_to_sram(
            emitter.zero_address, self._REDUCE_OUTPUT, output.physical_elements)
        engine.start_queue_for_bf16_matvec_operation(
            max_clear_en=0, fmax_context_addr=0,
            vector_sram_start_addr=self._REDUCE_VECTOR,
            matrix_sram_start_addr=self._REDUCE_MATRIX,
            output_sram_wb_addr=self._REDUCE_OUTPUT,
            K=64, N=source.rows, stride_z=64)
        engine.sram_to_accelerator_memory(
            self._REDUCE_OUTPUT, output.address, output.physical_elements)
        emitter.mark_padding_zero(output)
