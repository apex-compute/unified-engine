"""Optional convolution layout acceleration for the native 8 kHz compiler.

Prepend ``ConvTransposeOptimizationMixin`` to the compiler's base classes to
enable this lowering. It retains the existing one-hot BF16 matvec transpose
arithmetic, but emits only consumed columns and writes directly from SRAM.
Unsupported shapes retain the compiler's original implementation.
"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import user_dma_core as udc


class ConvTransposeOptimizationMixin:
    """Prune padded transpose work without changing convolution arithmetic."""

    _CONV_IDENTITY_SRAM = 0
    _CONV_OUTPUT_SRAM = 64 * 64 * 2
    _CONV_INPUT_SRAM = 0x80000

    def _conv_buffers_fit(self, input_bytes, output_bytes):
        return (input_bytes <= udc.URAM_NEAR_FULL_SIZE
                and self._CONV_OUTPUT_SRAM + output_bytes
                <= udc.URAM_NEAR_FULL_SIZE)

    def _stage_conv_input(self, index, node):
        source = self.layout(node.input[0])
        aux = self.conv_aux[index]
        packed, attrs = aux["input"], aux["attrs"]
        batch, channels, height, width = source.shape
        pads = tuple(int(value) for value in attrs.get("pads", (0, 0, 0, 0)))
        padded_height, padded_width, packed_channels = packed.shape
        if packed_channels != channels:
            raise RuntimeError("Conv packed-input channel mismatch")
        row_bytes = 64 * 2
        output_bytes = height * width * row_bytes
        if (batch != 1 or not 1 <= channels <= 64
                or source.padded_last % 64 or packed.padded_last != 64
                or len(pads) != 4 or min(pads) < 0
                or padded_height != height + pads[0] + pads[2]
                or padded_width != width + pads[1] + pads[3]
                or not self._conv_buffers_fit(source.size_bytes, output_bytes)):
            return super()._stage_conv_input(index, node)

        emitter, engine = self.emitter, self.emitter.engine
        # Spatial padding and inactive channels must be zero. Clear the DRAM
        # destination before loading identity into the SRAM that emit_zero uses.
        emitter.emit_zero(packed)
        engine.accelerator_memory_to_sram(
            self.identity_address, self._CONV_IDENTITY_SRAM, 64 * 64)
        engine.accelerator_memory_to_sram(
            source.address, self._CONV_INPUT_SRAM, 0,
            memcpy_length_bytes=source.size_bytes)
        if channels < 64:
            engine.accelerator_memory_to_sram(
                emitter.zero_address, self._CONV_OUTPUT_SRAM, 0,
                memcpy_length_bytes=output_bytes)

        # Each channel is a row with H*padded(W) elements. Select one spatial
        # column from every channel exactly as bf16_transpose_core does. The
        # column's 64-wide block remains the same, including its BF16 arithmetic.
        for row in range(height):
            for column in range(width):
                block, lane = divmod(row * source.padded_last + column, 64)
                engine.start_queue_for_bf16_matvec_operation(
                    max_clear_en=0, fmax_context_addr=0,
                    vector_sram_start_addr=self._CONV_IDENTITY_SRAM + lane * row_bytes,
                    matrix_sram_start_addr=self._CONV_INPUT_SRAM + block * row_bytes,
                    output_sram_wb_addr=(self._CONV_OUTPUT_SRAM
                                        + (row * width + column) * row_bytes),
                    K=64, N=channels, stride_z=height * source.padded_last)
        for row in range(height):
            engine.sram_to_accelerator_memory(
                self._CONV_OUTPUT_SRAM + row * width * row_bytes,
                packed.address + ((row + pads[0]) * padded_width + pads[1]) * row_bytes,
                0, memcpy_length_bytes=width * row_bytes)

    def _unstage_conv_output(self, index, node):
        aux = self.conv_aux[index]
        packed = aux["output"]
        output = self.layout(node.output[0])
        batch, channels, height, width = output.shape
        # Every native 8 kHz convolution produces H=1. More general layouts
        # require separate alignment handling, so preserve their proven path.
        if (batch != 1 or height != 1 or not 1 <= channels <= 64
                or packed.shape != (height, width, channels)
                or packed.padded_last != 64 or output.padded_last % 64
                or not self._conv_buffers_fit(packed.size_bytes, output.size_bytes)):
            return super()._unstage_conv_output(index, node)

        emitter, engine = self.emitter, self.emitter.engine
        engine.accelerator_memory_to_sram(
            self.identity_address, self._CONV_IDENTITY_SRAM, 64 * 64)
        engine.accelerator_memory_to_sram(
            packed.address, self._CONV_INPUT_SRAM, 0,
            memcpy_length_bytes=packed.size_bytes)
        # A partial matvec writes the logical prefix only. Explicitly initialize
        # every padded lane before emitting full aligned output rows.
        if width != output.padded_last:
            engine.accelerator_memory_to_sram(
                emitter.zero_address, self._CONV_OUTPUT_SRAM, 0,
                memcpy_length_bytes=output.size_bytes)
        for channel in range(channels):
            engine.start_queue_for_bf16_matvec_operation(
                max_clear_en=0, fmax_context_addr=0,
                vector_sram_start_addr=self._CONV_IDENTITY_SRAM + channel * 128,
                matrix_sram_start_addr=self._CONV_INPUT_SRAM,
                output_sram_wb_addr=(self._CONV_OUTPUT_SRAM
                                    + channel * output.padded_last * 2),
                K=64, N=width, stride_z=64)
        engine.sram_to_accelerator_memory(
            self._CONV_OUTPUT_SRAM, output.address, 0,
            memcpy_length_bytes=output.size_bytes)
        emitter.mark_padding_zero(output)
