"""BF16 BigCodec convolutions using bounded device im2col and matrix multiply.

All neural arithmetic stays in the resident instruction stream. Weights have
one BF16 copy in the image, independent of audio length. Tensors retain the
shared ``[time, pad64(channels)]`` ABI. Learned transpose convolutions use
exact polyphase kernels, including their output-padding samples.
"""

from __future__ import annotations

import torch

from bigcodec_conv import (
    ZERO_BYTES, _address, _disjoint, _pads, _positive, _shape, _stage_window,
    conv1d_output_shape, conv_transpose1d_output_shape, packed_bytes,
    shared, transpose_polyphase, udc,
)


def _geometry(input_shape, weight_shape, *, padding, stride, dilation,
              transpose, output_padding, chunk_rows):
    shape = _shape(input_shape)
    chunk_rows = _positive(chunk_rows, "chunk_rows")
    if chunk_rows > 64:
        raise ValueError("BF16 convolution chunks may contain at most 64 rows")
    if transpose:
        if dilation != 1:
            raise ValueError("dilated transpose convolution is unsupported")
        output = conv_transpose1d_output_shape(shape, weight_shape, stride=stride,
                                              padding=padding, output_padding=output_padding)
        windows = []
        for phase in range(min(stride, output[0])):
            taps = len(range((phase + padding) % stride, weight_shape[2], stride))
            rows = (output[0] - phase + stride - 1) // stride
            windows.append((rows + taps - 1, taps))
        stage_rows = max(rows for rows, _ in windows)
        kernel = max(taps for _, taps in windows)
    else:
        if output_padding:
            raise ValueError("output_padding only applies to transpose convolution")
        output = conv1d_output_shape(shape, weight_shape, stride=stride,
                                     padding=padding, dilation=dilation)
        stage_rows = shape[0] + sum(_pads(padding))
        kernel = weight_shape[2]
    row_bytes = packed_bytes((1, shape[1]))
    output_row_bytes = packed_bytes((1, output[1]))
    if kernel * row_bytes > udc.URAM_NEAR_FULL_SIZE:
        raise ValueError("one im2col patch exceeds SRAM capacity")
    chunk_rows = min(chunk_rows, output[0], max(1, udc.URAM_NEAR_FULL_SIZE // output_row_bytes))
    return {
        "output_shape": output, "stage_rows": stage_rows, "kernel": kernel,
        "chunk_rows": chunk_rows,
        "stage_bytes": stage_rows * row_bytes,
        "patch_bytes": chunk_rows * kernel * row_bytes,
        "result_bytes": chunk_rows * output_row_bytes,
    }


def conv_scratch_bytes(input_shape, weight_shape, *, padding=0, stride=1,
                       dilation=1, transpose=False, output_padding=0,
                       chunk_rows=64):
    geometry = _geometry(input_shape, weight_shape, padding=padding, stride=stride,
                         dilation=dilation, transpose=transpose,
                         output_padding=output_padding, chunk_rows=chunk_rows)
    return geometry["stage_bytes"] + geometry["patch_bytes"] + geometry["result_bytes"]


def _weights(image, weight, bias):
    weight = torch.as_tensor(weight).detach().cpu().bfloat16()
    if weight.ndim != 3 or min(weight.shape) <= 0 or not torch.isfinite(weight).all():
        raise ValueError("BF16 weights must be finite (out, in, kernel) tensors")
    outputs, inputs, kernel = weight.shape
    input_width, output_width = shared._align_up(inputs, 64), shared._align_up(outputs, 64)
    matrix = torch.zeros(output_width, kernel, input_width, dtype=torch.bfloat16)
    matrix[:outputs, :, :inputs] = weight.permute(0, 2, 1)
    resource = {"weight_address": image.allocate(matrix.reshape(output_width, -1), alignment=128),
                "kernel": kernel, "K": kernel * input_width, "N": output_width,
                "weight_bytes": matrix.numel() * 2, "bias_address": None}
    if bias is not None:
        bias = torch.as_tensor(bias).detach().cpu().bfloat16()
        if tuple(bias.shape) != (outputs,) or not torch.isfinite(bias).all():
            raise ValueError("BF16 bias must have one finite value per output channel")
        padded = torch.zeros(output_width, dtype=torch.bfloat16)
        padded[:outputs] = bias
        resource["bias_address"] = image.allocate(padded, alignment=128)
    return resource


def _prepare(name, weight, bias, *, input_shape, input_address, output_address,
              scratch_address, image, stride, padding, dilation, transpose,
              output_padding, chunk_rows):
    weight = torch.as_tensor(weight)
    input_shape = _shape(input_shape)
    geometry = _geometry(input_shape, weight.shape, padding=padding, stride=stride,
                         dilation=dilation, transpose=transpose,
                         output_padding=output_padding, chunk_rows=chunk_rows)
    source = _address(input_address, "input_address")
    destination = _address(output_address, "output_address")
    scratch = _address(scratch_address, "scratch_address")
    scratch_bytes = geometry["stage_bytes"] + geometry["patch_bytes"] + geometry["result_bytes"]
    _disjoint([("input", source, packed_bytes(input_shape)),
               ("output", destination, packed_bytes(geometry["output_shape"])),
               ("scratch", scratch, scratch_bytes)])
    phases = []
    if transpose:
        for phase in transpose_polyphase(weight, input_length=input_shape[0],
                                          stride=stride, padding=padding,
                                          output_padding=output_padding):
            phase.update(_weights(image, phase.pop("weight"), bias))
            phase["stride"] = 1
            phase["dilation"] = 1
            phases.append(phase)
    else:
        left, _ = _pads(padding)
        phases.append({"phase": 0, "input_start": -left,
                       "input_length": geometry["stage_rows"],
                       "output_length": geometry["output_shape"][0],
                       "stride": stride, "dilation": dilation,
                       **_weights(image, weight, bias)})
    return {
        "name": name, "precision": "bf16", "input_shape": input_shape,
        "output_shape": geometry["output_shape"], "input_address": source,
        "output_address": destination, "padded_input_address": scratch,
        "scratch_address": scratch, "scratch_bytes": scratch_bytes,
        "padding_bytes": geometry["stage_bytes"],
        "patch_address": scratch + geometry["stage_bytes"],
        "result_address": scratch + geometry["stage_bytes"] + geometry["patch_bytes"],
        "chunk_rows": geometry["chunk_rows"], "output_step": stride if transpose else 1,
        "phases": phases,
    }


def prepare_conv1d(name, weight, bias=None, *, input_shape, input_address,
                   output_address, scratch_address, image, stride=1, padding=0,
                   dilation=1, chunk_rows=64):
    return _prepare(name, weight, bias, input_shape=input_shape,
                     input_address=input_address, output_address=output_address,
                     scratch_address=scratch_address, image=image, stride=stride,
                     padding=padding, dilation=dilation, transpose=False,
                     output_padding=0, chunk_rows=chunk_rows)


def prepare_conv_transpose1d(name, weight, bias=None, *, input_shape,
                             input_address, output_address, scratch_address,
                             image, stride, padding=0, output_padding=0,
                             chunk_rows=64):
    return _prepare(name, weight, bias, input_shape=input_shape,
                     input_address=input_address, output_address=output_address,
                     scratch_address=scratch_address, image=image, stride=stride,
                     padding=padding, dilation=1, transpose=True,
                     output_padding=output_padding, chunk_rows=chunk_rows)


def _matmul(engine, *, M, K, N, **addresses):
    # Estimate the static kernel's dots and DMA descriptors. Deep kernels
    # force narrow output strips, so even a few input rows can otherwise
    # expand into hundreds of instructions. Keep cheap launches static.
    strip = min(N, (udc.URAM_NEAR_FULL_ELEMENTS // K) // 64 * 64)
    if strip < 64:
        strip = 32 if K * 32 <= udc.URAM_NEAR_FULL_ELEMENTS else 16
    rows = min(M, udc.URAM_FULL_ELEMENTS // (K + max(64, strip)))
    strips = (N + strip - 1) // strip
    descriptors = 0
    for first in range(0, M, rows):
        take = min(rows, M - first)
        descriptors += 1 + strips * (
            take + 1 + (addresses.get("C_DRAM_ADDR") is not None)
            + (take if strip < 64 else 1))
    register = None
    if descriptors >= 128:
        register = engine.alloc_isa_reg()
        engine.generate_instruction_add_set(register, M)
    try:
        engine.matmat_mul_core(M=M, K=K, N=N, gpr_M_reg=register, **addresses)
    finally:
        if register is not None:
            engine.release_isa_reg()


def _stage_patches(engine, plan, phase, *, first, take, row_bytes):
    """Fill [batch, tap, channel] patches using the smaller DMA traversal."""
    kernel = phase["kernel"]
    patch_row_bytes = kernel * row_bytes
    if kernel < take and take * row_bytes <= udc.URAM_NEAR_FULL_SIZE:
        # A strided gather reads one tap across all output rows. Strided
        # writeback interleaves those rows into the contiguous im2col matrix.
        for tap in range(kernel):
            address = plan["padded_input_address"] + (
                first * phase["stride"] + tap * phase["dilation"]) * row_bytes
            shared._copy_contiguous_or_strided_read(
                engine, source=address, sram=0, total=take * row_bytes,
                chunk=row_bytes, jump=phase["stride"] * row_bytes)
            shared._copy_contiguous_or_strided_write(
                engine, sram=0, destination=plan["patch_address"] + tap * row_bytes,
                total=take * row_bytes, chunk=row_bytes, jump=patch_row_bytes)
    else:
        for row in range(take):
            address = plan["padded_input_address"] + (first + row) * phase["stride"] * row_bytes
            shared._copy_contiguous_or_strided_read(
                engine, source=address, sram=0, total=patch_row_bytes,
                chunk=row_bytes, jump=phase["dilation"] * row_bytes)
            engine.sram_to_accelerator_memory(
                0, plan["patch_address"] + row * patch_row_bytes,
                0, memcpy_length_bytes=patch_row_bytes)


def emit_conv(engine, plan, *, zero_address):
    """Append all input staging, BF16 arithmetic and output copies to capture."""
    _address(zero_address, "zero_address")
    _disjoint([("input", plan["input_address"], packed_bytes(plan["input_shape"])),
               ("output", plan["output_address"], packed_bytes(plan["output_shape"])),
               ("scratch", plan["scratch_address"], plan["scratch_bytes"]),
               ("zero", zero_address, ZERO_BYTES)])
    row_bytes = packed_bytes((1, plan["input_shape"][1]))
    output_row_bytes = packed_bytes((1, plan["output_shape"][1]))
    for phase in plan["phases"]:
        _stage_window(engine, plan, phase, zero_address)
        for first in range(0, phase["output_length"], plan["chunk_rows"]):
            take = min(plan["chunk_rows"], phase["output_length"] - first)
            _stage_patches(engine, plan, phase, first=first, take=take, row_bytes=row_bytes)
            direct = plan["output_step"] == 1
            destination = (plan["output_address"] + first * output_row_bytes
                           if direct else plan["result_address"])
            kwargs = {} if phase["bias_address"] is None else {"C_DRAM_ADDR": phase["bias_address"]}
            _matmul(engine,
                M=take, K=phase["K"], N=phase["N"],
                A_DRAM_ADDR=plan["patch_address"], B_DRAM_ADDR=phase["weight_address"],
                OUTPUT_DRAM_ADDR=destination, **kwargs)
            if not direct:
                engine.accelerator_memory_to_sram(destination, 0, 0,
                                                  memcpy_length_bytes=take * output_row_bytes)
                shared._copy_contiguous_or_strided_write(
                    engine, sram=0,
                    destination=plan["output_address"] + (first * plan["output_step"] + phase["phase"]) * output_row_bytes,
                    total=take * output_row_bytes, chunk=output_row_bytes,
                    jump=plan["output_step"] * output_row_bytes)
