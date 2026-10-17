"""Pack complex pairs in SRAM without intermediate DRAM projections."""

from __future__ import annotations

import torch
import user_dma_core as udc


class PairPackOptimizationMixin:
    """Fuse the two interleaving projections of a padded complex tensor."""

    _PACK_INPUT = 0x80000
    _PACK_SELECTOR = 0xE0000
    _PACK_TEMP = 0x2000
    _PACK_OUTPUT = 0x4000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        selector = torch.zeros(64, 128, dtype=torch.bfloat16)
        for pair in range(32):
            selector[2 * pair, pair] = 1
            selector[2 * pair + 1, 64 + pair] = 1
        self.pair_pack_selector = self.emitter.allocate_constant(selector)

    def emit_view(self, index, node):
        if index not in self.pair_pack_aux:
            return super().emit_view(index, node)
        source, output = self.layout(node.input[0]), self.layout(node.output[0])
        # The one-hot transpose reads complete source rows. Unknown padding
        # could contain NaN, which zero selector weights do not suppress.
        if (source.logical_last != 2 or source.padded_last != 64
                or output.rows != 1 or output.logical_elements != source.rows * 2
                or source.name not in self.emitter._zero_padding
                or self._PACK_INPUT + source.size_bytes + 128 > self._PACK_SELECTOR
                or self._PACK_SELECTOR - 0x80000 + 64 * 128 * 2 > udc.URAM_NEAR_FULL_SIZE
                or self._PACK_OUTPUT + output.size_bytes > udc.URAM_NEAR_FULL_SIZE):
            return super().emit_view(index, node)
        engine = self.emitter.engine
        engine.accelerator_memory_to_sram(self.identity_address, 0, 64 * 64)
        engine.accelerator_memory_to_sram(
            self.pair_pack_selector, self._PACK_SELECTOR, 64 * 128)
        engine.accelerator_memory_to_sram(
            source.address, self._PACK_INPUT, source.physical_elements)
        for chunk, start in enumerate(range(0, source.rows, 32)):
            take = min(32, source.rows - start)
            # Both transpose vectors have 64 lanes. Clear their unused tails
            # before the fused projection, including the short final chunk.
            engine.accelerator_memory_to_sram(
                self.emitter.zero_address, self._PACK_TEMP, 128)
            if take == 1:
                # Keep the RK scalar writeback workaround: N=1 returns the
                # previous scalar. An aligned zero row makes N=2 equivalent.
                engine.accelerator_memory_to_sram(
                    self.emitter.zero_address,
                    self._PACK_INPUT + source.size_bytes, 64)
            for column in range(2):
                engine.start_queue_for_bf16_matvec_operation(
                    max_clear_en=0, fmax_context_addr=0,
                    vector_sram_start_addr=column * 128,
                    matrix_sram_start_addr=self._PACK_INPUT + start * 128,
                    output_sram_wb_addr=self._PACK_TEMP + column * 128,
                    K=64, N=max(2, take), stride_z=64)
            engine.start_queue_for_bf16_matvec_operation(
                max_clear_en=0, fmax_context_addr=0,
                vector_sram_start_addr=self._PACK_TEMP,
                matrix_sram_start_addr=self._PACK_SELECTOR,
                output_sram_wb_addr=self._PACK_OUTPUT + chunk * 128,
                # Stride advances each 64-lane K block. Contiguous K=128
                # weights use the same 64-lane stride as shared matmat.
                K=128, N=64, stride_z=64)
        engine.sram_to_accelerator_memory(
            self._PACK_OUTPUT, output.address, output.physical_elements)
        self.emitter.mark_padding_zero(output)
