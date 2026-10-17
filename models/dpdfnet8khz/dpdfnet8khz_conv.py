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

    def _plan_auxiliary_layouts(self):
        super()._plan_auxiliary_layouts()
        self._plan_conv_relu_fusions()

    def _plan_conv_relu_fusions(self):
        consumers = {}
        for index, node in enumerate(self.model.graph.node):
            for name in node.input:
                consumers.setdefault(name, []).append(index)
        exposed = {value.name for value in self.model.graph.output} | {"state_in", "state_out"}
        self.conv_relu_fusions = {}
        self.conv_relu_skips = set()
        for index, node in enumerate(self.model.graph.node):
            if node.op_type != "Conv" or node.output[0] in exposed:
                continue
            uses = consumers.get(node.output[0], [])
            if len(uses) != 1:
                continue
            activation = self.model.graph.node[uses[0]]
            if (activation.op_type != "Relu" or len(activation.input) != 1
                    or len(activation.output) != 1):
                continue
            output = self.layouts[node.output[0]]
            activated = self.layouts[activation.output[0]]
            if (output.shape != activated.shape
                    or not self._conv_output_supported(self.conv_aux[index]["output"], output)):
                continue
            self.conv_relu_fusions[index] = activation.output[0]
            self.conv_relu_skips.add(uses[0])

    def emit_node(self, index, node):
        if index in getattr(self, "conv_relu_skips", ()):
            self.emitter.engine.generate_instruction_nop()
            return
        return super().emit_node(index, node)

    def _conv_buffers_fit(self, input_bytes, output_bytes):
        return (input_bytes <= udc.URAM_NEAR_FULL_SIZE
                and self._CONV_OUTPUT_SRAM + output_bytes
                <= udc.URAM_NEAR_FULL_SIZE)

    def _conv_output_supported(self, packed, output):
        if len(output.shape) != 4:
            return False
        batch, channels, height, width = output.shape
        return (batch == 1 and height == 1 and 1 <= channels <= 64
                and packed.shape == (height, width, channels)
                and packed.padded_last == 64 and output.padded_last % 64 == 0
                and self._conv_buffers_fit(packed.size_bytes, output.size_bytes))

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
        # RK build 5fbbfbf0 returns the previous scalar for consecutive N=1
        # transpose matvecs. One zero dummy channel keeps this operation on
        # the established N>=2 path without changing any logical result.
        compute_channels = max(2, channels)
        channel_bytes = height * source.padded_last * 2
        output_bytes = height * width * row_bytes
        if (batch != 1 or not 1 <= channels <= 64
                or source.padded_last % 64 or packed.padded_last != 64
                or len(pads) != 4 or min(pads) < 0
                or padded_height != height + pads[0] + pads[2]
                or padded_width != width + pads[1] + pads[3]
                or not self._conv_buffers_fit(compute_channels * channel_bytes, output_bytes)):
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
        if compute_channels != channels:
            engine.accelerator_memory_to_sram(
                emitter.zero_address, self._CONV_INPUT_SRAM + source.size_bytes, 0,
                memcpy_length_bytes=(compute_channels - channels) * channel_bytes)
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
                    K=64, N=compute_channels, stride_z=height * source.padded_last)
        for row in range(height):
            engine.sram_to_accelerator_memory(
                self._CONV_OUTPUT_SRAM + row * width * row_bytes,
                packed.address + ((row + pads[0]) * padded_width + pads[1]) * row_bytes,
                0, memcpy_length_bytes=width * row_bytes)

    def _unstage_conv_output(self, index, node):
        aux = self.conv_aux[index]
        packed = aux["output"]
        fused_output = getattr(self, "conv_relu_fusions", {}).get(index)
        output = self.layout(fused_output or node.output[0])
        batch, channels, height, width = output.shape
        # Every native 8 kHz convolution produces H=1. More general layouts
        # require separate alignment handling, so preserve their proven path.
        if not self._conv_output_supported(packed, output):
            if fused_output:
                raise RuntimeError("fused Conv/ReLU output no longer fits its planned transpose")
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
        activation = ({"lalu_mode": udc.LALU_MODE.CLAMP,
                       "lalu_a": udc.LALU_CLAMP_RELU_A,
                       "lalu_b": udc.LALU_CLAMP_RELU_B} if fused_output else {})
        for channel in range(channels):
            engine.start_queue_for_bf16_matvec_operation(
                max_clear_en=0, fmax_context_addr=0,
                vector_sram_start_addr=self._CONV_IDENTITY_SRAM + channel * 128,
                matrix_sram_start_addr=self._CONV_INPUT_SRAM,
                output_sram_wb_addr=(self._CONV_OUTPUT_SRAM
                                    + channel * output.padded_last * 2),
                K=64, N=width, stride_z=64, **activation)
        engine.sram_to_accelerator_memory(
            self._CONV_OUTPUT_SRAM, output.address, 0,
            memcpy_length_bytes=output.size_bytes)
        emitter.mark_padding_zero(output)
