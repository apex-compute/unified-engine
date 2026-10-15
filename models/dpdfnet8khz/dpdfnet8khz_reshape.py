"""Unpack the native deep-filter coefficient vector directly in SRAM."""

from __future__ import annotations

import torch
import user_dma_core as udc


class CoefficientReshapeOptimizationMixin:
    """Turn 800 packed coefficients into 80 zero-padded rows of ten."""

    _RESHAPE_INPUT = 0
    _RESHAPE_MATRIX = 0x80000
    _RESHAPE_OUTPUT = 0x2000
    _RESHAPE_INPUT_LANES = 896
    _RESHAPE_PHASES = 32

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Consecutive rows start ten lanes apart, giving 32 distinct offsets
        # inside a 64-lane SRAM line. A 128-lane vector also covers rows that
        # cross a line. The selector has no arithmetic beyond copying BF16.
        selectors = torch.zeros(self._RESHAPE_PHASES, 10, 128,
                                dtype=torch.bfloat16)
        for phase in range(self._RESHAPE_PHASES):
            offset = phase * 10 % 64
            for column in range(10):
                selectors[phase, column, offset + column] = 1
        self.coefficient_reshape_selectors = self.emitter.allocate_constant(selectors)

    def emit_view(self, index, node):
        source, output = self.layout(node.input[0]), self.layout(node.output[0])
        selector_bytes = self._RESHAPE_PHASES * 10 * 128 * 2
        if (source.rows != 1 or source.logical_last != 800
                or source.padded_last != 832
                or output.rows != 80 or output.logical_last != 10
                or output.padded_last != 64
                or source.address % 128 or output.address % 128
                or any(address % 128 for address in (
                    self._RESHAPE_INPUT, self._RESHAPE_MATRIX, self._RESHAPE_OUTPUT))
                or self._RESHAPE_INPUT < 0
                or self._RESHAPE_INPUT + self._RESHAPE_INPUT_LANES * 2 > self._RESHAPE_OUTPUT
                or self._RESHAPE_OUTPUT + output.size_bytes > udc.URAM_NEAR_FULL_SIZE
                or self._RESHAPE_MATRIX < 0x80000
                or self._RESHAPE_MATRIX - 0x80000 + selector_bytes > udc.URAM_NEAR_FULL_SIZE):
            return super().emit_view(index, node)

        emitter, engine = self.emitter, self.emitter.engine
        # Only the 800 logical input lanes are read. Copying its physical
        # padding would allow zero*NaN to contaminate the last output rows.
        # The final K=128 vector extends to lane 895, beyond the DRAM tensor.
        engine.accelerator_memory_to_sram(
            emitter.zero_address, self._RESHAPE_INPUT, self._RESHAPE_INPUT_LANES)
        engine.accelerator_memory_to_sram(
            source.address, self._RESHAPE_INPUT, source.logical_elements)
        engine.accelerator_memory_to_sram(
            self.coefficient_reshape_selectors, self._RESHAPE_MATRIX,
            selector_bytes // 2)
        engine.accelerator_memory_to_sram(
            emitter.zero_address, self._RESHAPE_OUTPUT, output.physical_elements)
        for row in range(80):
            input_line = row * 10 // 64
            engine.start_queue_for_bf16_matvec_operation(
                max_clear_en=0, fmax_context_addr=0,
                vector_sram_start_addr=self._RESHAPE_INPUT + input_line * 128,
                matrix_sram_start_addr=(self._RESHAPE_MATRIX
                                       + row % self._RESHAPE_PHASES * 10 * 128 * 2),
                output_sram_wb_addr=self._RESHAPE_OUTPUT + row * 128,
                K=128, N=10, stride_z=64)
        engine.sram_to_accelerator_memory(
            self._RESHAPE_OUTPUT, output.address, output.physical_elements)
        emitter.mark_padding_zero(output)
