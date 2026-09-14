"""Dense BF16 convolution over DPDFNet's already padded HWC device tensors.

Spatial padding and zero channel padding are supplied by the existing input
stager. Weights remain BF16 in DRAM; im2col and matrix multiplication execute
on the device. No quantized weight stream or CPU neural operation is used.
"""

from __future__ import annotations

import torch

from dpdfnet_precompiled import shared, udc


def workspace_shape(packed_source, packed_output, weight_shape):
    """Return the bounded [output pixel, tap/channel] patch matrix shape."""
    if len(packed_source.shape) != 3 or len(packed_output.shape) != 3:
        raise ValueError("BF16 convolution requires packed HWC input and output")
    if len(weight_shape) != 4 or any(int(x) <= 0 for x in weight_shape):
        raise ValueError("BF16 convolution requires positive OIHW weight dimensions")
    oc, ic, kh, kw = map(int, weight_shape)
    if packed_source.logical_last != ic or packed_output.logical_last != oc:
        raise ValueError("BF16 convolution input/output channel mismatch")
    for layout in (packed_source, packed_output):
        if (layout.padded_last < layout.logical_last
                or layout.padded_last % udc.UE_VECTOR_SIZE):
            raise ValueError("BF16 convolution channels must be padded to 64 lanes")
    rows = int(packed_output.shape[0] * packed_output.shape[1])
    columns = kh * kw * packed_source.padded_last
    if (rows <= 0 or rows * columns * 2 > udc.URAM_NEAR_FULL_SIZE
            or rows * (columns + packed_output.padded_last) * 2 > udc.URAM_NEAR_FULL_SIZE
            or packed_output.padded_last * columns * 2 > udc.URAM_NEAR_FULL_SIZE):
        raise ValueError("BF16 convolution patch/matmul workspace exceeds SRAM capacity")
    return rows, columns


def _pair(attrs, name):
    value = tuple(int(x) for x in attrs.get(name, (1, 1)))
    if len(value) != 2 or min(value) <= 0:
        raise ValueError(f"BF16 convolution {name} must be two positive integers")
    return value


def _check_layouts(*layouts):
    for layout in layouts:
        if layout.address < 0 or layout.address % 64:
            raise ValueError("BF16 convolution DRAM layouts must be 64-byte aligned")
    for i, first in enumerate(layouts):
        for second in layouts[i + 1:]:
            if (first.address < second.address + second.size_bytes
                    and second.address < first.address + first.size_bytes):
                raise ValueError("BF16 convolution input/output/workspace allocations overlap")


def prepare_conv(emitter, packed_source, packed_output, attrs, weight, bias, workspace):
    """Allocate BF16 constants using a workspace planned before image capture.

    ``attrs['pads']`` has already been applied to ``packed_source`` by the caller.
    Both stride and dilation may differ between height and width.
    """
    weight = torch.as_tensor(weight).detach().cpu()
    rows, columns = workspace_shape(packed_source, packed_output, weight.shape)
    if (tuple(workspace.shape) != (rows, columns)
            or workspace.padded_last != columns):
        raise ValueError("BF16 convolution workspace shape mismatch")
    _check_layouts(packed_source, packed_output, workspace)
    if int(attrs.get("group", 1)) != 1:
        raise ValueError("BF16 dense convolution requires group=1")
    if attrs.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET", b"", ""):
        raise ValueError("BF16 convolution requires explicit pre-applied padding")
    stride, dilation = _pair(attrs, "strides"), _pair(attrs, "dilations")
    oc, ic, kh, kw = map(int, weight.shape)
    height, width, _ = packed_source.shape
    out_h, out_w, _ = packed_output.shape
    expected = ((height - dilation[0] * (kh - 1) - 1) // stride[0] + 1,
                (width - dilation[1] * (kw - 1) - 1) // stride[1] + 1)
    if expected != (out_h, out_w) or min(expected) <= 0:
        raise ValueError("BF16 convolution geometry disagrees with packed output shape")
    if "kernel_shape" in attrs and tuple(attrs["kernel_shape"]) != (kh, kw):
        raise ValueError("BF16 convolution kernel_shape disagrees with weights")
    row_bytes = packed_source.padded_last * 2
    if (out_w > 1 and stride[1] != 1
            and (row_bytes > udc.UE_STRIDE_CHUNK_MAX_BYTES
                 or stride[1] * row_bytes > udc.UE_STRIDE_JUMP_MAX_BYTES)):
        raise ValueError("BF16 convolution input stride exceeds DMA descriptor fields")
    rounded_weight = weight.to(torch.bfloat16)
    if not torch.isfinite(rounded_weight).all():
        raise ValueError("BF16 convolution weights must remain finite after conversion")
    padded_weight = torch.zeros(
        packed_output.padded_last, kh, kw, packed_source.padded_last,
        dtype=torch.bfloat16)
    padded_weight[:oc, :, :, :ic] = rounded_weight.permute(0, 2, 3, 1)
    padded_bias = None
    if bias is not None:
        bias = torch.as_tensor(bias).detach().cpu().to(torch.bfloat16)
        if bias.shape != (oc,) or not torch.isfinite(bias).all():
            raise ValueError("BF16 convolution bias must have finite output-channel values")
        padded_bias = torch.zeros(packed_output.padded_last, dtype=torch.bfloat16)
        padded_bias[:oc] = bias
    weight_address = emitter.allocate_constant(padded_weight.reshape(packed_output.padded_last, columns))
    bias_address = None if padded_bias is None else emitter.allocate_constant(padded_bias)
    return {"kind": "dense_bf16", "M": rows, "K": columns, "N": packed_output.padded_last,
            "input_address": packed_source.address, "output_address": packed_output.address,
            "workspace_address": workspace.address, "weight_address": weight_address,
            "bias_address": bias_address, "weight_size_bytes": padded_weight.numel() * 2,
            "bias_size_bytes": 0 if padded_bias is None else padded_bias.numel() * 2,
            "input_shape": tuple(packed_source.shape), "output_shape": tuple(packed_output.shape),
            "weight_shape": tuple(weight.shape), "input_channels_padded": packed_source.padded_last,
            "stride": stride, "dilation": dilation,
            "direct": (kh == kw == 1 and stride == (1, 1)
                       and (height, width) == (out_h, out_w))}


def emit_conv(engine, resource):
    """Emit tap-wise strided DMA im2col, then an unquantized BF16 matmul."""
    if resource["kind"] != "dense_bf16":
        raise ValueError("expected a dense_bf16 convolution resource")
    source = resource["input_address"]
    if not resource["direct"]:
        _, width, _ = resource["input_shape"]
        out_h, out_w, _ = resource["output_shape"]
        _, _, kh, kw = resource["weight_shape"]
        sy, sx = resource["stride"]
        dy, dx = resource["dilation"]
        row_bytes = resource["input_channels_padded"] * 2
        patch_bytes = resource["K"] * 2
        for oy in range(out_h):
            for ky in range(kh):
                for kx in range(kw):
                    origin = source + ((oy * sy + ky * dy) * width + kx * dx) * row_bytes
                    target = resource["workspace_address"] + (
                        oy * out_w * patch_bytes + (ky * kw + kx) * row_bytes)
                    shared._copy_contiguous_or_strided_read(
                        engine, source=origin, sram=0, total=out_w * row_bytes,
                        chunk=row_bytes, jump=sx * row_bytes)
                    shared._copy_contiguous_or_strided_write(
                        engine, sram=0, destination=target, total=out_w * row_bytes,
                        chunk=row_bytes, jump=patch_bytes)
        source = resource["workspace_address"]
    engine.matmat_mul_core(
        M=resource["M"], K=resource["K"], N=resource["N"],
        A_DRAM_ADDR=source, B_DRAM_ADDR=resource["weight_address"],
        OUTPUT_DRAM_ADDR=resource["output_address"], C_DRAM_ADDR=resource["bias_address"],
        bias_mode="broadcast_N", is_B_quantized=False)
