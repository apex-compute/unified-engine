"""Native 80-lane row unpadding using the RK256 strided writeback contract.

Read DMA keeps beat-aligned 160-byte chunks packed. Writeback instead begins
each 160-byte chunk at a new 128-byte SRAM line after consuming its first two
lines. This asymmetry can discard the unused final 96 bytes of each padded
80-lane row. Verified on RK256 build df0749de with tagged logical words,
NaN padding, and destination guards, then compared against full model output.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import user_dma_core as udc


class Reshape80OptimizationMixin:
    def emit_view(self, index, node):
        source = self.layout(node.input[0])
        output = self.layout(node.output[0])
        if (udc.UE_AXI_DATA_WIDTH_BITS != 256
                or source.logical_last != 80 or source.padded_last != 128
                or source.rows < 4 or source.rows % 4
                or output.rows != 1
                or output.logical_elements != source.logical_elements
                or output.physical_elements != output.logical_elements
                or source.size_bytes > udc.URAM_NEAR_FULL_SIZE
                or source.address % 128 or output.address % 128
                or max(source.address, output.address)
                < min(source.address + source.size_bytes, output.address + output.size_bytes)):
            return super().emit_view(index, node)
        engine = self.emitter.engine
        engine.accelerator_memory_to_sram(source.address, 0, source.physical_elements)
        engine.sram_to_accelerator_memory(
            0, output.address, output.logical_elements,
            stride_bytes_per_chunk=160, stride_jump_bytes=160)
        self.emitter.mark_padding_zero(output)
