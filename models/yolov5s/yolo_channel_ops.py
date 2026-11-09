"""Resident channel copies and block-sparse lowering of depthwise convolutions.

The 64-lane native convolution walker is reused per channel block. Original
IF8 codes/scales are sliced without requantization; off-block zeros need not
be read or multiplied. No intermediate data crosses PCIe.
"""
from dataclasses import replace

import torch


def copy_channels(engine, source, destination, *, start, count, zero_address, zero_bytes):
    """Copy contiguous physical channel ranges with complete 64-byte DMA beats."""
    pixel_bytes = count * 2
    if count <= 0 or start < 0 or pixel_bytes % 64 or start * 2 % 8:
        raise ValueError('Channel copy requires 32-channel chunks and ISA-aligned starts')
    if source.height != destination.height or source.width != destination.width:
        raise ValueError('Channel copy requires equal spatial dimensions')
    if start + count > source.physical_channels or count > destination.physical_channels:
        raise ValueError('Channel copy exceeds the physical map')
    if count < destination.physical_channels:
        for offset in range(0, destination.size_bytes, zero_bytes):
            size = min(zero_bytes, destination.size_bytes - offset)
            engine.accelerator_memory_to_sram(zero_address, 0, 0, memcpy_length_bytes=size)
            engine.sram_to_accelerator_memory(0, destination.address + offset, 0, memcpy_length_bytes=size)
    pixels = source.height * source.width
    batch = max(1, zero_bytes // pixel_bytes)
    for first in range(0, pixels, batch):
        size = min(batch, pixels - first) * pixel_bytes
        read_stride = {} if count == source.physical_channels else dict(
            stride_bytes_per_chunk=pixel_bytes, stride_jump_bytes=source.physical_channels * 2)
        write_stride = {} if count == destination.physical_channels else dict(
            stride_bytes_per_chunk=pixel_bytes, stride_jump_bytes=destination.physical_channels * 2)
        engine.accelerator_memory_to_sram(source.address + (first * source.physical_channels + start) * 2,
                                         0, 0, memcpy_length_bytes=size, **read_stride)
        engine.sram_to_accelerator_memory(0, destination.address + first * destination.physical_channels * 2,
                                         0, memcpy_length_bytes=size, **write_stride)


def emit_select(engine, operation, source, destination, zero_address, zero_bytes):
    start, count = operation['start'], operation['count']
    mapping = source.channel_map[start:start + count]
    if len(mapping) != count or mapping != tuple(range(mapping[0], mapping[0] + count)):
        raise ValueError('Channel selection must be contiguous in the packed source')
    if destination.channel_map != tuple(range(destination.channels)):
        raise ValueError('Channel selection output must be contiguous')
    copy_channels(engine, source, destination, start=mapping[0], count=count,
                  zero_address=zero_address, zero_bytes=zero_bytes)


def prepare_depthwise(operation, weight, source, destination, image, scratch, *, unpack_codes, prepare_conv):
    if weight['precision'] != 'if8' or weight['layout'] != 'channels':
        raise ValueError('Block-depthwise lowering requires channel-layout IF8 weights')
    codes = unpack_codes(weight['codes_packed'], weight['codes_shape'], 'if8')
    channels, ci, kh, kw = codes.shape
    if channels != ci or channels != source.channels or destination.channels != channels:
        raise ValueError('Depthwise convolution must have matching input/output channels')
    if source.channel_map != tuple(range(channels)):
        raise ValueError('Depthwise input must have contiguous channels')
    blocks = (channels + 63) // 64
    scales = weight['block_scales'].view(channels, kh * kw, blocks)
    plans = []
    for group in range(blocks):
        first, stop = group * 64, min(channels, (group + 1) * 64)
        # Reject accidental use on a dense kernel: each output channel can
        # depend only on the same input channel, including within its block.
        off_diagonal = codes[first:stop].clone()
        off_diagonal[torch.arange(stop - first), torch.arange(first, stop)] = 0
        if torch.any(off_diagonal):
            raise ValueError('Depthwise weights contain off-diagonal nonzero codes')
        group_codes = codes[first:stop, first:stop].contiguous()
        group_weight = dict(precision='if8', layout='channels',
            codes_packed=group_codes.flatten(), codes_shape=list(group_codes.shape),
            block_scales=scales[first:stop, :, group].contiguous(),
            bias=weight['bias'][first:stop].contiguous() if weight['bias'].numel() else weight['bias'])
        packed_source = replace(source, logical_shape=(stop - first, source.height, source.width),
                                physical_channels=64, channel_map=tuple(range(stop - first)), address=scratch)
        packed_destination = replace(destination, logical_shape=(stop - first, destination.height, destination.width),
                                     physical_channels=64, channel_map=tuple(range(stop - first)))
        plan = prepare_conv(operation, group_weight, packed_source, packed_destination, image)
        # Scatter native 64-lane outputs directly into the full destination.
        plan['destination'] = replace(packed_destination, physical_channels=destination.physical_channels,
                                      address=destination.address + first * 2)
        plans.append((first, packed_source, plan))
    return dict(source=source, groups=plans)


def emit_depthwise(engine, plan, zero_address, zero_bytes, *, emit_conv):
    for first, packed_source, convolution in plan['groups']:
        copy_channels(engine, plan['source'], packed_source, start=first, count=64,
                      zero_address=zero_address, zero_bytes=zero_bytes)
        emit_conv(engine, convolution, zero_address)
