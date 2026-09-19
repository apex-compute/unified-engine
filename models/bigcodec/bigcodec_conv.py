"""Native BigCodec convolutions on padded channels-last BF16 tensors.

Preparation stores IF8-INT weights in the deployment image. Emission only
appends device instructions; it never opens a device or evaluates a neural
operation on the host. The caller owns the image, tensor arena and capture.

All shapes are ``(time, channels)``; each time row occupies ``pad64(channels)``
BF16 values. ``zero_address`` must point to at least ``ZERO_BYTES`` zero bytes.
The input padding and scatter workspaces are reusable between operations.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "models" / "yolov5s"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import user_dma_core as udc
import yolov5_precompiled as shared

ZERO_BYTES = shared._ACT_TEMPLATE_BYTES


def _positive(value, label):
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _shape(shape):
    if len(shape) != 2:
        raise ValueError("expected a (time, channels) shape")
    return tuple(_positive(value, label) for value, label in
                 zip(shape, ("time", "channels")))


def _pads(padding):
    values = (padding, padding) if isinstance(padding, int) else tuple(padding)
    if len(values) != 2 or any(isinstance(x, bool) or int(x) != x or x < 0
                               for x in values):
        raise ValueError("padding must be an integer or two nonnegative integers")
    return tuple(int(x) for x in values)


def packed_bytes(shape):
    time, channels = _shape(shape)
    return time * shared._align_up(channels, 64) * 2


def _address(address, label):
    if isinstance(address, bool) or int(address) != address or int(address) % 128:
        raise ValueError(f"{label} must be a 128-byte-aligned address")
    if not udc.DRAM_START_ADDR <= int(address) < 0x100000000:
        raise ValueError(f"{label} is outside the RK 2-GiB DRAM aperture")
    return int(address)


def _disjoint(regions):
    ordered = sorted((address, address + size, name)
                     for name, address, size in regions if size)
    if any(end > 0x100000000 for _, end, _ in ordered):
        raise ValueError("convolution tensor exceeds the RK DRAM aperture")
    for previous, current in zip(ordered, ordered[1:]):
        if current[0] < previous[1]:
            raise ValueError(f"{previous[2]} overlaps {current[2]}")


def conv1d_output_shape(input_shape, weight_shape, *, stride=1, padding=0,
                        dilation=1):
    time, channels = _shape(input_shape)
    if len(weight_shape) != 3:
        raise ValueError("Conv1d weight must have shape (out, in, kernel)")
    out, inputs, kernel = (_positive(x, "weight dimension") for x in weight_shape)
    stride, dilation = _positive(stride, "stride"), _positive(dilation, "dilation")
    left, right = _pads(padding)
    if inputs != channels:
        raise ValueError("Conv1d input channel mismatch")
    output = (time + left + right - dilation * (kernel - 1) - 1) // stride + 1
    if output <= 0:
        raise ValueError("Conv1d output length must be positive")
    return output, out


def conv1d_padding_bytes(input_shape, *, padding=0):
    time, channels = _shape(input_shape)
    left, right = _pads(padding)
    return packed_bytes((time + left + right, channels)) if left or right else 0


def conv_transpose1d_output_shape(input_shape, weight_shape, *, stride,
                                  padding=0, output_padding=0):
    time, channels = _shape(input_shape)
    if len(weight_shape) != 3:
        raise ValueError("ConvTranspose1d weight must have shape (in, out, kernel)")
    inputs, out, kernel = (_positive(x, "weight dimension") for x in weight_shape)
    stride = _positive(stride, "stride")
    if inputs != channels:
        raise ValueError("ConvTranspose1d input channel mismatch")
    if (not isinstance(padding, int) or isinstance(padding, bool) or padding < 0
            or not isinstance(output_padding, int) or isinstance(output_padding, bool)
            or not 0 <= output_padding < stride):
        raise ValueError("transpose padding must be nonnegative and output_padding < stride")
    if kernel < stride:
        raise ValueError("transpose kernels shorter than stride are unsupported")
    output = (time - 1) * stride - 2 * padding + kernel + output_padding
    if output <= 0:
        raise ValueError("ConvTranspose1d output length must be positive")
    return output, out


def transpose_polyphase(weight, *, input_length, stride, padding=0,
                         output_padding=0):
    """Return regular cross-correlation kernels and their exact input windows.

    For phase r, ``y[stride*j+r] = sum_k w[k]*x[(stride*j+r+pad-k)/stride]``.
    Reverse the contributing k values to obtain an ordinary Conv1d kernel.
    An input window beginning before zero or ending past input_length is
    zero-padded. No sample is duplicated, dropped or added for output_padding.
    """
    output, _ = conv_transpose1d_output_shape(
        (input_length, weight.shape[0]), weight.shape, stride=stride,
        padding=padding, output_padding=output_padding)
    phases = []
    for phase in range(min(stride, output)):
        take = (output - phase + stride - 1) // stride
        indices = list(range((phase + padding) % stride, weight.shape[2], stride))
        indices.reverse()
        first = (phase + padding - indices[0]) // stride
        phases.append({
            "phase": phase,
            "weight": weight[:, :, indices].permute(1, 0, 2).contiguous(),
            "input_start": first,
            "input_length": take + len(indices) - 1,
            "output_length": take,
        })
    return phases


def conv_transpose1d_padding_bytes(input_shape, weight_shape, *, stride,
                                   padding=0, output_padding=0):
    output, _ = conv_transpose1d_output_shape(
        input_shape, weight_shape, stride=stride, padding=padding,
        output_padding=output_padding)
    time, channels = _shape(input_shape)
    lengths = []
    for phase in range(min(stride, output)):
        count = len(range((phase + padding) % stride, weight_shape[2], stride))
        take = (output - phase + stride - 1) // stride
        lengths.append(take + count - 1)
    return packed_bytes((max(lengths), channels))


def quantize_conv1d_if8(weight, bias=None):
    """Encode finite weights using BF16 per-64-value INT8 block scales.

    Blocks follow the hardware walker: (tap, channel-block, lane), or packed
    (tap, channel) for profitable small-channel gather. The signed scale
    selects IF8's INT variant. Quantization agrees with quant_lib INT8 while
    avoiding its unused FP8 candidate search for large BigCodec kernels.
    """
    weight = torch.as_tensor(weight).detach().cpu().float().contiguous()
    if weight.ndim != 3 or min(weight.shape) <= 0:
        raise ValueError("weight must be a nonempty (out, in, kernel) tensor")
    if not torch.isfinite(weight).all() or not torch.isfinite(weight.bfloat16()).all():
        raise ValueError("convolution weights must be finite in BF16")
    out, channels, kernel = weight.shape
    cpad = shared._align_up(channels, 64)
    gather_chunks = math.ceil(kernel * channels / 64)
    gather = channels <= 255 and gather_chunks <= 4 and gather_chunks < kernel * (cpad // 64)
    if gather:
        padded = torch.zeros(out, gather_chunks * 64, dtype=torch.bfloat16)
        padded[:, :kernel * channels] = weight.permute(0, 2, 1).reshape(out, -1).bfloat16()
        blocks = padded.reshape(out, gather_chunks, 64)
    else:
        padded = torch.zeros(out, kernel, cpad, dtype=torch.bfloat16)
        padded[:, :, :channels] = weight.permute(0, 2, 1).bfloat16()
        blocks = padded.reshape(out, kernel * (cpad // 64), 64)
    scales = (blocks.float().abs().amax(-1).clamp(min=1e-8) / 127).bfloat16()
    codes = (blocks / scales.unsqueeze(-1)).bfloat16().round().clamp(-128, 127).to(torch.int8)
    if gather:
        logical = codes.reshape(out, -1)[:, :kernel * channels].reshape(out, kernel, channels)
    else:
        logical = codes.reshape(out, kernel, cpad)[:, :, :channels]
    encoded_bias = torch.empty(0, dtype=torch.bfloat16)
    if bias is not None:
        encoded_bias = torch.as_tensor(bias).detach().cpu().bfloat16().contiguous()
        if tuple(encoded_bias.shape) != (out,) or not torch.isfinite(encoded_bias).all():
            raise ValueError("bias must contain one finite BF16 value per output channel")
    logical = logical.permute(0, 2, 1).unsqueeze(2).contiguous().view(torch.uint8)
    return {
        "precision": "if8", "layout": "gather" if gather else "channels",
        "codes_packed": logical.flatten(), "codes_shape": list(logical.shape),
        "block_scales": -scales.contiguous(), "bias": encoded_bias,
    }


def _map(shape, address):
    time, channels = _shape(shape)
    return shared._PackedMap((channels, 1, time), shared._align_up(channels, 64),
                             tuple(range(channels)), address)


def _prepare_subplans(name, weight, bias, *, source, output_length,
                       output_address, image, stride, dilation,
                       weight_reuse_pixels):
    reuse = _positive(weight_reuse_pixels, "weight_reuse_pixels")
    plans = []
    # BigCodec C768/K7 otherwise selects OC96, which cannot be scattered from
    # half SRAM lines. An aligned <=64-output plan also bounds scale BRAM use.
    for start in range(0, weight.shape[0], 64):
        take = min(64, weight.shape[0] - start)
        encoded = quantize_conv1d_if8(weight[start:start + take],
                                     None if bias is None else bias[start:start + take])
        blocks = encoded["block_scales"].shape[1]
        target = _map((output_length, take), output_address)
        operation = {"name": f"{name}/oc{start}", "stride": stride,
                     "pad": 0, "dilation": dilation, "activate": False}
        plan = shared._prepare_conv_plan(
            operation, encoded, source, target, image,
            allow_half_vector_output=True,
            weight_stream_budget_bytes=64 * blocks * 64 * reuse)
        plan["channel_offset"] = start
        plans.append(plan)
    return plans


def _shorten_subplans(subplans, output_length):
    """Reuse immutable weights while shortening a final polyphase sequence."""
    result = []
    for original in subplans:
        plan = dict(original)
        tiles = []
        for oy, ox, th, tw, y0, x0, win_h, _ in original["tiles"]:
            if ox >= output_length:
                continue
            take = min(tw, output_length - ox)
            win_w = ((take - 1) * original["operation"]["stride"]
                     + original["kernel_w"])
            tiles.append((oy, ox, th, take, y0, x0, win_h, win_w))
        plan["tiles"] = tiles
        plan["groups"] = udc.conv2d_tile_geometry_groups(tiles)
        destination = original["destination"]
        plan["destination"] = _map((output_length, destination.channels), destination.address)
        result.append(plan)
    return result


def prepare_conv1d(name, weight, bias=None, *, input_shape, input_address,
                   output_address, padded_input_address, image, stride=1,
                   padding=0, dilation=1, weight_reuse_pixels=1):
    weight = torch.as_tensor(weight)
    input_shape = _shape(input_shape)
    output_shape = conv1d_output_shape(input_shape, weight.shape, stride=stride,
                                       padding=padding, dilation=dilation)
    left, right = _pads(padding)
    source_address = _address(input_address, "input_address")
    output_address = _address(output_address, "output_address")
    staging = _address(padded_input_address, "padded_input_address") if left or right else source_address
    stage_shape = (input_shape[0] + left + right, input_shape[1])
    regions = [("input", source_address, packed_bytes(input_shape)),
               ("output", output_address, packed_bytes(output_shape))]
    if staging != source_address:
        regions.append(("padded input", staging, packed_bytes(stage_shape)))
    elif left or right:
        raise ValueError("padded input must have a separate workspace")
    _disjoint(regions)
    result = {"name": name, "kind": "conv1d", "input_shape": input_shape,
              "output_shape": output_shape, "input_address": source_address,
              "output_address": output_address, "padded_input_address": staging,
              "input_start": -left, "input_length": stage_shape[0],
              "padding_bytes": packed_bytes(stage_shape) if left or right else 0}
    ct = math.ceil(input_shape[1] / 64)
    minimum_window = dilation * (weight.shape[2] - 1) + 1
    needs_polyphase = dilation > 1 and (
        dilation * minimum_window * ct > 0xFFF
        or minimum_window * ct > shared._ACT_URAM_LINES)
    if not needs_polyphase:
        plans = _prepare_subplans(
            name, weight, bias, source=_map(stage_shape, staging),
            output_length=output_shape[0], output_address=output_address,
            image=image, stride=stride, dilation=dilation,
            weight_reuse_pixels=weight_reuse_pixels)
    else:
        # The 2-D descriptor still encodes dilation*window_width*channel_tiles
        # as a 12-bit row stride when height==1. C768/K7/d9 needs5940, although
        # it has no second spatial row. Split time into dilation phases to use
        # the existing legal undilated descriptor without changing the driver.
        if staging == source_address:
            raise ValueError("large dilated Conv1d requires a padded input workspace")
        gcd = math.gcd(stride, dilation)
        output_step, inner_stride = dilation // gcd, stride // gcd
        longest = math.ceil(output_shape[0] / output_step)
        phase_input = (longest - 1) * inner_stride + weight.shape[2]
        if phase_input > stage_shape[0]:
            raise RuntimeError("dilation phases exceed the reserved padding workspace")
        plans = _prepare_subplans(
            name, weight, bias, source=_map((phase_input, input_shape[1]), staging),
            output_length=longest, output_address=output_address, image=image,
            stride=inner_stride, dilation=1,
            weight_reuse_pixels=weight_reuse_pixels)
        result["output_step"] = output_step
        result["dilation_phases"] = []
        for phase in range(min(output_step, output_shape[0])):
            take = (output_shape[0] - phase + output_step - 1) // output_step
            result["dilation_phases"].append({
                "phase": phase, "input_start": phase * stride - left,
                "input_step": dilation,
                "input_length": (take - 1) * inner_stride + weight.shape[2],
                "subplans": _shorten_subplans(plans, take),
            })
    result["subplans"] = plans
    result["scatter_scratch_bytes"] = scatter_scratch_bytes(plans)
    return result


def prepare_conv_transpose1d(name, weight, bias=None, *, input_shape,
                             input_address, output_address, padded_input_address,
                             image, stride, padding=0, output_padding=0,
                             weight_reuse_pixels=1):
    weight = torch.as_tensor(weight)
    input_shape = _shape(input_shape)
    output_shape = conv_transpose1d_output_shape(
        input_shape, weight.shape, stride=stride, padding=padding,
        output_padding=output_padding)
    source_address = _address(input_address, "input_address")
    output_address = _address(output_address, "output_address")
    staging = _address(padded_input_address, "padded_input_address")
    workspace = conv_transpose1d_padding_bytes(
        input_shape, weight.shape, stride=stride, padding=padding,
        output_padding=output_padding)
    _disjoint([("input", source_address, packed_bytes(input_shape)),
               ("output", output_address, packed_bytes(output_shape)),
               ("padded input", staging, workspace)])
    phases = []
    for phase in transpose_polyphase(weight, input_length=input_shape[0],
                                      stride=stride, padding=padding,
                                      output_padding=output_padding):
        phase["subplans"] = _prepare_subplans(
            f"{name}/phase{phase['phase']}", phase.pop("weight"), bias,
            source=_map((phase["input_length"], input_shape[1]), staging),
            output_length=phase["output_length"], output_address=output_address,
            image=image, stride=1, dilation=1,
            weight_reuse_pixels=weight_reuse_pixels)
        phases.append(phase)
    return {"name": name, "kind": "conv_transpose1d", "input_shape": input_shape,
            "output_shape": output_shape, "input_address": source_address,
            "output_address": output_address, "padded_input_address": staging,
            "padding_bytes": workspace, "stride": stride, "phases": phases,
            "scatter_scratch_bytes": max(scatter_scratch_bytes(p["subplans"]) for p in phases)}


def scatter_scratch_bytes(subplans):
    return max((max(t[2] * t[3] for t in p["tiles"]) * 64
                for p in subplans if p["oc_chunk"] == 32), default=0)


def _copy(engine, source, destination, size):
    for offset in range(0, size, ZERO_BYTES):
        take = min(ZERO_BYTES, size - offset)
        engine.accelerator_memory_to_sram(source + offset, 0, 0, memcpy_length_bytes=take)
        engine.sram_to_accelerator_memory(0, destination + offset, 0, memcpy_length_bytes=take)


def _stage_window(engine, plan, window, zero_address):
    source, destination = plan["input_address"], plan["padded_input_address"]
    if source == destination:
        return
    row_bytes = packed_bytes((1, plan["input_shape"][1]))
    length, start = window["input_length"], window["input_start"]
    size = length * row_bytes
    for offset in range(0, size, ZERO_BYTES):
        take = min(ZERO_BYTES, size - offset)
        engine.accelerator_memory_to_sram(zero_address, 0, 0, memcpy_length_bytes=take)
        engine.sram_to_accelerator_memory(0, destination + offset, 0, memcpy_length_bytes=take)
    step = window.get("input_step", 1)
    first = min(length, max(0, (-start + step - 1) // step))
    stop = max(first, min(length, (plan["input_shape"][0] - 1 - start) // step + 1))
    block_rows = max(1, ZERO_BYTES // row_bytes)
    for row in range(first, stop, block_rows):
        take = min(block_rows, stop - row)
        shared._copy_contiguous_or_strided_read(
            engine, source=source + (start + row * step) * row_bytes,
            sram=0, total=take * row_bytes, chunk=row_bytes, jump=step * row_bytes)
        engine.sram_to_accelerator_memory(0, destination + row * row_bytes, 0,
                                          memcpy_length_bytes=take * row_bytes)


def _emit_subplans(engine, plans, *, output_shape, output_address, phase,
                    output_step, zero_address, scratch_address):
    pixel_bytes = packed_bytes((1, output_shape[1]))
    for plan in plans:
        operation = plan["operation"]
        ct = math.ceil(plan["convolution_c"] / 64)
        for chunk in plan["chunks"]:
            oc = plan["channel_offset"] + chunk["oc0"]
            for start, stop, th, tw, _, win_w in plan["groups"]:
                blocks = plan["gather_chunks"] if plan["use_gather"] else plan["kernel_w"] * ct
                engine.accelerator_memory_to_scale_sram(chunk["scale_address"], plan["oc_chunk"] * blocks)
                if chunk["bias_address"] is not None:
                    engine.accelerator_memory_to_bias_sram(chunk["bias_address"], plan["oc_chunk"])
                for tile in plan["tiles"][start:stop]:
                    shared._stage_conv_window(engine, plan["source"], tile, 0, zero_address)
                    engine.start_queue_for_conv2d_operation(
                        act_sram_start_addr=0, output_sram_wb_addr=shared._WB_SRAM_ADDRESS,
                        weights_dram_addr=chunk["weight_address"], kernel_w=plan["kernel_w"],
                        kernel_h=1, ct=ct, oc_count=plan["oc_chunk"], out_w=tw, out_h=1,
                        w_pad=win_w, stride_s=operation["stride"], data_type=udc.TYPE.IF8,
                        bias_enable=plan["bias_enabled"], lalu_mode=udc.LALU_MODE.BYPASS,
                        lalu_a=0, lalu_b=0, dilation=operation["dilation"],
                        gather=plan["use_gather"], c_in=plan["convolution_c"])
                    destination = output_address + (output_step * tile[1] + phase) * pixel_bytes + oc * 2
                    if plan["oc_chunk"] == 32:
                        # SRAM addresses are128-byte aligned. Spill a packed
                        # OC32 tile, then reload each64-byte pixel at line0.
                        engine.sram_to_accelerator_memory(shared._WB_SRAM_ADDRESS, scratch_address, 0,
                                                          memcpy_length_bytes=tw * 64)
                        for column in range(tw):
                            engine.accelerator_memory_to_sram(scratch_address + column * 64, 0, 0,
                                                              memcpy_length_bytes=64)
                            engine.sram_to_accelerator_memory(0, destination + column * output_step * pixel_bytes,
                                                              0, memcpy_length_bytes=64)
                    else:
                        shared._copy_contiguous_or_strided_write(
                            engine, sram=shared._WB_SRAM_ADDRESS, destination=destination,
                            total=tw * plan["oc_chunk"] * 2, chunk=plan["oc_chunk"] * 2,
                            jump=output_step * pixel_bytes)


def _validate_emit_workspaces(plan, zero_address, scratch_address):
    _address(zero_address, "zero_address")
    _address(scratch_address, "scratch_address")
    regions = [("input", plan["input_address"], packed_bytes(plan["input_shape"])),
               ("output", plan["output_address"], packed_bytes(plan["output_shape"])),
               ("zero", zero_address, ZERO_BYTES),
               ("scatter scratch", scratch_address, plan["scatter_scratch_bytes"])]
    if plan["padding_bytes"]:
        regions.append(("padded input", plan["padded_input_address"], plan["padding_bytes"]))
    _disjoint(regions)


def emit_conv1d(engine, plan, *, zero_address, scratch_address):
    if plan["kind"] != "conv1d":
        raise ValueError("expected a Conv1d plan")
    _validate_emit_workspaces(plan, zero_address, scratch_address)
    if "dilation_phases" in plan:
        for phase in plan["dilation_phases"]:
            _stage_window(engine, plan, phase, zero_address)
            _emit_subplans(engine, phase["subplans"], output_shape=plan["output_shape"],
                           output_address=plan["output_address"], phase=phase["phase"],
                           output_step=plan["output_step"], zero_address=zero_address,
                           scratch_address=scratch_address)
        return
    _stage_window(engine, plan, plan, zero_address)
    _emit_subplans(engine, plan["subplans"], output_shape=plan["output_shape"],
                   output_address=plan["output_address"], phase=0, output_step=1,
                   zero_address=zero_address, scratch_address=scratch_address)


def emit_conv_transpose1d(engine, plan, *, zero_address, scratch_address):
    if plan["kind"] != "conv_transpose1d":
        raise ValueError("expected a ConvTranspose1d plan")
    _validate_emit_workspaces(plan, zero_address, scratch_address)
    for phase in plan["phases"]:
        _stage_window(engine, plan, phase, zero_address)
        _emit_subplans(engine, phase["subplans"], output_shape=plan["output_shape"],
                       output_address=plan["output_address"], phase=phase["phase"],
                       output_step=plan["stride"], zero_address=zero_address,
                       scratch_address=scratch_address)
