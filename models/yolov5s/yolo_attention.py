"""Resident spatial self-attention for YOLO heads with 64-lane Q/K/V maps."""
import math

import torch


def dimensions(source):
    if source.channels != 64 or source.physical_channels != 64 or source.channel_map != tuple(range(64)):
        raise ValueError('Attention requires contiguous, 64-channel packed maps')
    count = source.height * source.width
    return count, (count + 63) // 64 * 64


def scratch_bytes(source):
    _, aligned = dimensions(source)
    # Padded K/V, V transpose, score/probability matrix and scaled Q.
    return 2 * (4 * aligned * 64 + aligned * aligned)


def prepare(operation, sources, destination, image):
    if len(sources) != 3 or any(s.logical_shape != sources[0].logical_shape for s in sources):
        raise ValueError('Attention Q/K/V shapes must agree')
    for source in [*sources, destination]:
        dimensions(source)
    if destination.logical_shape != sources[0].logical_shape:
        raise ValueError('Attention output shape differs')
    scale = float(operation['scale'])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Attention scale must be finite and positive')
    count, aligned = dimensions(sources[0])
    mask = torch.zeros(count, aligned, dtype=torch.bfloat16)
    mask[:, count:] = -1e4
    return dict(sources=sources, destination=destination, count=count, aligned=aligned, scale=scale,
                mask=image.allocate(mask, alignment=128),
                identity=image.allocate(torch.eye(64, dtype=torch.bfloat16), alignment=128))


def emit(engine, plan, scratch, zero_address, zero_bytes):
    q, k, v = plan['sources']
    count, aligned = plan['count'], plan['aligned']
    size = aligned * 128
    for source, address in ((k, scratch), (v, scratch + size)):
        # Include the padding rows in every run; never reuse stale scratch.
        for offset in range(0, size, zero_bytes):
            take = min(zero_bytes, size - offset)
            engine.accelerator_memory_to_sram(zero_address, 0, 0, memcpy_length_bytes=take)
            engine.sram_to_accelerator_memory(0, address + offset, 0, memcpy_length_bytes=take)
        for offset in range(0, count * 128, zero_bytes):
            take = min(zero_bytes, count * 128 - offset)
            engine.accelerator_memory_to_sram(source.address + offset, 0, 0, memcpy_length_bytes=take)
            engine.sram_to_accelerator_memory(0, address + offset, 0, memcpy_length_bytes=take)
    batch_reg = engine.alloc_isa_reg()
    sequence_reg = engine.alloc_isa_reg()
    try:
        engine.generate_instruction_add_set(batch_reg, count)
        engine.generate_instruction_add_set(sequence_reg, aligned)
        engine.unified_attention_core(batch=count, aligned_seq_len=aligned, head_dim=64,
            Q_DRAM_ADDR=q.address, K_DRAM_ADDR=scratch, V_DRAM_ADDR=scratch + size,
            BIAS_DRAM_ADDR=plan['mask'], OUTPUT_DRAM_ADDR=plan['destination'].address,
            SCRATCH_DRAM_ADDR=scratch + 2 * size, IDENTITY_DRAM_ADDR=plan['identity'],
            gpr_batch_reg=batch_reg, gpr_aligned_seq_len_reg=sequence_reg, q_scale=plan['scale'])
    finally:
        engine.release_isa_reg()
        engine.release_isa_reg()
