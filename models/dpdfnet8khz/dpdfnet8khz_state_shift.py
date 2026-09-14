"""Fuse full-row packed-state shifts into one 128-input BF16 projection.

The shared state copier splits a shifted row across two 64-input projections
and adds their outputs. Each lane selects exactly one source value. A single
64-by-128 one-hot matrix expresses the same copy while retaining aligned DMA.
Boundary rows and source windows that include logical padding keep the shared
implementation. This mixin is intended to precede OptimizedGraphCompiler.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import user_dma_core as udc


class StateShiftOptimizationMixin:
    _STATE_SHIFT_INPUT_SRAM = 0
    _STATE_SHIFT_OUTPUT_SRAM = 0x20000
    _STATE_SHIFT_MATRIX_SRAM = 0x80000
    _STATE_SHIFT_BATCH_ROWS = 128

    @staticmethod
    def _full_shift_row(resource, row):
        """Return (first source row, phase) only for a complete safe copy."""
        if row["keep"] is not None or len(row["pieces"]) != 2:
            return None
        logical_destination = row["destination_row"] * 64
        relative = logical_destination - resource["destination_start"]
        if (relative < 0 or logical_destination + 64
                > resource["destination"].logical_elements):
            return None
        logical_source = resource["source_start"] + relative
        source_row, phase = divmod(logical_source, 64)
        if (phase == 0 or [piece[0] for piece in row["pieces"]] != [source_row, source_row + 1]
                or (source_row + 2) * 64 > resource["source"].logical_elements):
            # Do not bring possibly nonfinite source padding into a one-hot
            # product: zero times NaN is NaN. The proven boundary path remains.
            return None
        return source_row, phase

    def _prepare_state_copies(self):
        resources = super()._prepare_state_copies()
        self.state_shift128_selectors = {}
        for copies in resources.values():
            for resource in copies:
                for row in resource["rows"]:
                    selection = self._full_shift_row(resource, row)
                    if selection is None or selection[1] in self.state_shift128_selectors:
                        continue
                    phase = selection[1]
                    matrix = torch.zeros(64, 128, dtype=torch.bfloat16)
                    lanes = torch.arange(64)
                    matrix[lanes, phase + lanes] = 1
                    self.state_shift128_selectors[phase] = self.emitter.allocate_constant(matrix)
        return resources

    def emit_state_copy(self, resource):
        # The input windows overlap by 64 elements. Emit explicit SRAM vector
        # offsets; matmat_mul_core(M>1,K=128) would advance them by 128 instead.
        row_bytes = 128
        available = min(
            self._STATE_SHIFT_BATCH_ROWS,
            (self._STATE_SHIFT_OUTPUT_SRAM - self._STATE_SHIFT_INPUT_SRAM) // row_bytes - 1,
            (udc.URAM_NEAR_FULL_SIZE - self._STATE_SHIFT_OUTPUT_SRAM) // row_bytes)
        if (available < 1 or 64 * 128 * 2 > udc.URAM_NEAR_FULL_SIZE
                or self._STATE_SHIFT_INPUT_SRAM % row_bytes
                or self._STATE_SHIFT_OUTPUT_SRAM % row_bytes):
            return super().emit_state_copy(resource)
        if resource["prefix"]:
            super().emit_state_copy({**resource, "rows": []})
        rows, engine = resource["rows"], self.emitter.engine
        position = 0
        while position < len(rows):
            selection = self._full_shift_row(resource, rows[position])
            if selection is None:
                # Preserve the delegate's batching for aligned full rows and
                # any other contiguous range this optimization cannot fuse.
                end = position + 1
                while end < len(rows) and self._full_shift_row(resource, rows[end]) is None:
                    end += 1
                super().emit_state_copy({**resource, "prefix": 0, "rows": rows[position:end]})
                position = end
                continue
            first_source_row, phase = selection
            first_destination_row = rows[position]["destination_row"]
            end = position + 1
            while end < min(len(rows), position + available):
                following = self._full_shift_row(resource, rows[end])
                delta = end - position
                if (following != (first_source_row + delta, phase)
                        or rows[end]["destination_row"] != first_destination_row + delta):
                    break
                end += 1
            count = end - position
            engine.accelerator_memory_to_sram(
                self.state_shift128_selectors[phase], self._STATE_SHIFT_MATRIX_SRAM, 64 * 128)
            engine.accelerator_memory_to_sram(
                resource["source"].address + first_source_row * row_bytes,
                self._STATE_SHIFT_INPUT_SRAM, 0,
                memcpy_length_bytes=(count + 1) * row_bytes)
            for row in range(count):
                engine.start_queue_for_bf16_matvec_operation(
                    max_clear_en=0, fmax_context_addr=0,
                    vector_sram_start_addr=self._STATE_SHIFT_INPUT_SRAM + row * row_bytes,
                    matrix_sram_start_addr=self._STATE_SHIFT_MATRIX_SRAM,
                    output_sram_wb_addr=self._STATE_SHIFT_OUTPUT_SRAM + row * row_bytes,
                    # stride_z advances each 64-lane K block, not each
                    # complete 128-input matrix row. Match shared matmat.
                    K=128, N=64, stride_z=64)
            engine.sram_to_accelerator_memory(
                self._STATE_SHIFT_OUTPUT_SRAM,
                resource["destination"].address + first_destination_row * row_bytes,
                0, memcpy_length_bytes=count * row_bytes)
            position = end
