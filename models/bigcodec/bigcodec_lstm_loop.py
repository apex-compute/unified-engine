"""Counted device loops for the existing SRAM LSTM arithmetic.

The first timestep uses the initialized hidden state. Remaining timesteps reuse
one absolute ISA loop, advancing only the hidden, input-projection and output
DRAM addresses. Parameters, cell storage and arithmetic operations are unchanged.
"""
from __future__ import annotations

import bigcodec_lstm as lstm
from bigcodec_device import udc


class _TimeAddresses:
    """Redirect timestep-dependent DRAM transfers through caller-owned GPRs."""

    def __init__(self, engine, *, width, hidden, projection, output,
                 hidden_reg, projection_reg, output_reg, temporary_reg):
        self.engine = engine
        self.width = width
        self.hidden = hidden
        self.projection = projection
        self.output = output
        self.hidden_reg = hidden_reg
        self.projection_reg = projection_reg
        self.output_reg = output_reg
        self.temporary_reg = temporary_reg

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def _projection_register(self, address):
        offset = address - self.projection
        if not 0 <= offset < 4 * self.width * 2 or offset % 128:
            raise ValueError('Invalid LSTM gate projection offset')
        if offset == 0:
            return self.projection_reg
        self.engine.generate_instruction_add_imm(
            src_reg_idx=self.projection_reg, immediate_value=offset >> 3,
            dst_reg_idx=self.temporary_reg)
        return self.temporary_reg

    def accelerator_memory_to_sram(self, address, sram, count):
        if address == self.hidden:
            return self.engine.accelerator_memory_to_sram(
                0, sram, count, general_reg_src=self.hidden_reg)
        if self.projection <= address < self.projection + 4 * self.width * 2:
            return self.engine.accelerator_memory_to_sram(
                0, sram, count, general_reg_src=self._projection_register(address))
        return self.engine.accelerator_memory_to_sram(address, sram, count)

    def accelerator_memory_to_bias_sram(self, address, count):
        if not self.projection <= address < self.projection + 4 * self.width * 2:
            return self.engine.accelerator_memory_to_bias_sram(address, count)
        register = self._projection_register(address)
        pointer = self.engine.alloc_inst_ptr()
        try:
            self.engine.generate_instruction_pbi_init(
                dma_length=count * 2, inst_pointer_idx=pointer)
            self.engine.generate_instruction_pbi_inc(
                inst_pointer_idx=pointer, pbi_field_select=udc.PBI_FIELD.DRAM_ADDR,
                general_reg_src=register)
            # PBI memcpy operands are deltas. The pointer already contains the
            # address and length, so both descriptor deltas must remain zero.
            self.engine.accelerator_memory_to_bias_sram(
                0, 0, inst_pointer_idx=pointer)
        finally:
            self.engine.release_inst_ptr(pointer)

    def sram_to_accelerator_memory(self, sram, address, count):
        if address == self.output:
            return self.engine.sram_to_accelerator_memory(
                sram, 0, count, general_reg_src=self.output_reg)
        return self.engine.sram_to_accelerator_memory(sram, address, count)


def emit_lstm(engine, plan, *, projection_fn=None, step_fn=None):
    """Emit both recurrent layers without adding a host execution boundary.

    Resolve the arithmetic callbacks at emission time so existing per-plan
    arithmetic selection applies equally to unrolled and counted execution.
    """
    projection_fn = projection_fn or lstm._recurrent_projection_sram
    step_fn = step_fn or lstm._lstm_step_sram
    width = plan.padded_width
    sram_fusion = (width <= lstm.TANH_CHUNK_ELEMENTS
                   and (plan.recurrent_precision == 'if8'
                        or width * 64 <= udc.URAM_NEAR_FULL_ELEMENTS))
    if not sram_fusion:
        raise ValueError('Counted LSTM requires the SRAM recurrent path')
    if plan.sequence < 1:
        raise ValueError('LSTM sequence must contain at least one timestep')
    addresses = {name: region[0] for name, region in plan.regions.items()}

    for layer_index, layer in enumerate(plan.layers):
        source = plan.input_address if layer_index == 0 else addresses['layer0_output']
        destination = addresses['layer0_output'] if layer_index == 0 else plan.output_address
        states = (('hidden', 'cell', 'cell_low')
                  if plan.compensated_cell and plan.preserve_cell_residual
                  else ('hidden', 'cell'))
        for name in states:
            for offset in range(0, width, 64):
                lstm._copy(engine, plan.zero_address, addresses[name] + offset * 2, 64)
        engine.matmat_mul_core(
            M=plan.sequence, K=width, N=4 * width, A_DRAM_ADDR=source,
            B_DRAM_ADDR=layer.input_weights, OUTPUT_DRAM_ADDR=addresses['input_gates'],
            C_DRAM_ADDR=layer.input_bias, bias_mode='broadcast_N')

        projection_fn(engine, plan, layer, addresses['hidden'], addresses['input_gates'])
        step_fn(engine, plan, addresses['input_gates'], addresses['cell'], destination)
        if plan.sequence == 1:
            continue

        registers = [engine.alloc_isa_reg() for _ in range(4)]
        hidden_reg, projection_reg, output_reg, temporary_reg = registers
        try:
            projection = addresses['input_gates'] + 4 * width * 2
            output = destination + width * 2
            engine.generate_instruction_add_set(hidden_reg, destination >> 3)
            engine.generate_instruction_add_set(projection_reg, projection >> 3)
            engine.generate_instruction_add_set(output_reg, output >> 3)
            wrapper = _TimeAddresses(
                engine, width=width, hidden=destination, projection=projection,
                output=output, hidden_reg=hidden_reg, projection_reg=projection_reg,
                output_reg=output_reg, temporary_reg=temporary_reg)
            # The arithmetic body can exceed the relative-jump cache window.
            engine.loop_start(loop_cnt=plan.sequence - 1, relative=False)
            projection_fn(wrapper, plan, layer, destination, projection)
            step_fn(wrapper, plan, projection, addresses['cell'], output)
            engine.generate_instruction_add_imm(hidden_reg, (width * 2) >> 3)
            engine.generate_instruction_add_imm(projection_reg, (4 * width * 2) >> 3)
            engine.generate_instruction_add_imm(output_reg, (width * 2) >> 3)
            engine.loop_end()
        finally:
            for _ in registers:
                engine.release_isa_reg()

    if plan.skip:
        engine.eltwise_core_dram(
            M=plan.sequence * width // 64, N=64, dram_a=plan.input_address,
            dram_b=plan.output_address, dram_out=plan.output_address,
            mode=udc.UE_MODE.ELTWISE_ADD)
