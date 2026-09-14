"""Offline lowering of BigCodec's two-layer residual LSTM to Andromeda ops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import user_dma_core as udc

TANH_ARGUMENT_LIMIT = 4.0
TANH_CHUNK_ELEMENTS = 4096
_TANH_NUMERATOR = (1 / 135135, 378 / 135135, 17325 / 135135, 1.0)
_TANH_DENOMINATOR = (28 / 135135, 3150 / 135135, 62370 / 135135, 1.0)
_TANH_NUMERATOR_BF16 = tuple(float(torch.tensor(value, dtype=torch.bfloat16)) for value in _TANH_NUMERATOR)
_TANH_DENOMINATOR_BF16 = tuple(float(torch.tensor(value, dtype=torch.bfloat16)) for value in _TANH_DENOMINATOR)


def tanh_scratch_bytes(count: int) -> int:
    if not isinstance(count, int) or count <= 0 or count % 64:
        raise ValueError("Tanh requires a positive multiple of 64 elements")
    return 3 * min(count, TANH_CHUNK_ELEMENTS) * 2


def pade_tanh(value, *, bf16=False):
    """Independent numeric contract for the bounded odd [7/6] Padé expression.

    x*(135135+17325*x²+378*x⁴+x⁶)/(135135+62370*x²+3150*x⁴+28*x⁶).
    Positive polynomial coefficients avoid cancellation near zero. This helper
    is for numerical validation only; emit_lstm never calls it.
    """
    q = (lambda x: torch.as_tensor(x).bfloat16().float()) if bf16 else (lambda x: x)
    value = q(value.clamp(-TANH_ARGUMENT_LIMIT, TANH_ARGUMENT_LIMIT))
    square = q(value * value)
    polys = []
    for coefficients in (_TANH_NUMERATOR, _TANH_DENOMINATOR):
        result = q(square * q(coefficients[0]))
        for index, coefficient in enumerate(coefficients[1:]):
            result = q(result + q(coefficient))
            if index < 2:
                result = q(result * square)
        polys.append(result)
    return q(value * q(polys[0] * q(1 / polys[1]))).clamp(-1, 1)


@dataclass(frozen=True)
class LSTMLayer:
    input_weights: int
    recurrent_weights: int
    input_bias: int
    recurrent_bias: int
    recurrent_scales: int | None = None


@dataclass(frozen=True)
class LSTMPlan:
    sequence: int
    width: int
    padded_width: int
    input_address: int
    output_address: int
    scratch_address: int
    scratch_bytes: int
    identity_address: int
    zero_address: int
    skip: bool
    layers: tuple[LSTMLayer, LSTMLayer]
    regions: dict[str, tuple[int, int]]
    recurrent_precision: str = "bf16"


def scratch_bytes(input_shape: tuple[int, int]) -> int:
    """Workspace bytes; does not include input, output, weights or 64x64 identity."""
    if len(input_shape) != 2 or any(int(x) != x or x <= 0 for x in input_shape):
        raise ValueError("LSTM input_shape must be positive (T, D)")
    sequence, width = map(int, input_shape)
    padded = (width + 63) // 64 * 64
    return (5 * sequence + 8) * padded * 2 + tanh_scratch_bytes(padded)


def quantize_recurrent_if8(weight):
    """Driver-compatible IF8 INT8 blocks, with negative BF16 scale tags.

    Each contiguous row is partitioned into 64-element blocks. Zero blocks
    use zero codes and scale -1, matching UnifiedEngine.quantize_weight.
    """
    weight = torch.as_tensor(weight).detach().cpu().bfloat16().contiguous()
    if (weight.ndim != 2 or any(size <= 0 or size % 64 for size in weight.shape)
            or not torch.isfinite(weight).all()):
        raise ValueError("IF8 recurrent weights must be finite with pad64 matrix dimensions")
    blocks = weight.reshape(-1, 64)
    maximum = blocks.abs().amax(-1)
    scale = (maximum / 127).bfloat16()
    scale = torch.where(maximum == 0, torch.ones_like(scale), scale)
    if not torch.isfinite(scale).all() or (scale == 0).any():
        raise ValueError("IF8 recurrent scales underflow or are nonfinite")
    codes = (blocks / scale[:, None]).round().clamp(-128, 127).to(torch.int8)
    return codes.reshape_as(weight).contiguous().view(torch.uint8), -scale.contiguous()


def prepare_lstm(lstm_module, image, *, input_shape: tuple[int, int],
                 input_address: int, output_address: int, scratch_address: int,
                 identity_address: int, zero_address: int, skip: bool = True,
                 recurrent_precision: str = "bf16") -> LSTMPlan:
    """Pack PyTorch i/f/g/o parameters with optional IF8 recurrent matrices.

    Inputs/outputs are time-major [T, pad64(D)], with zero padding lanes. The
    caller provides separate input, output and workspace regions, a 64x64 BF16
    identity, and at least 64 BF16 zeros. Parameters are only read at compilation.
    Input weights and biases always use BF16. IF8 recurrence adds raw byte codes
    and signed BF16 scale blocks; it does not change pointwise gate/state math.
    """
    workspace_bytes = scratch_bytes(input_shape)
    sequence, width = map(int, input_shape)
    if recurrent_precision not in ("bf16", "if8"):
        raise ValueError("LSTM recurrent precision must be bf16 or if8")
    if (not isinstance(lstm_module, torch.nn.LSTM)
            or lstm_module.input_size != width or lstm_module.hidden_size != width
            or lstm_module.num_layers != 2 or lstm_module.bidirectional
            or lstm_module.proj_size != 0 or not lstm_module.batch_first
            or not lstm_module.bias or lstm_module.training):
        raise ValueError("Expected an eval two-layer, unidirectional, batch-first LSTM with equal input/hidden width and biases")
    padded = (width + 63) // 64 * 64
    if recurrent_precision == "if8" and padded > TANH_CHUNK_ELEMENTS:
        raise ValueError("IF8 recurrent streaming requires padded hidden width <=4096")
    regions_to_check = {
        "input": (input_address, sequence * padded * 2),
        "output": (output_address, sequence * padded * 2),
        "scratch": (scratch_address, workspace_bytes),
        "identity": (identity_address, 64 * 64 * 2),
        "zero": (zero_address, 64 * 2),
    }
    ordered = sorted((int(address), int(address) + size, name)
                     for name, (address, size) in regions_to_check.items())
    for index, (start, end, name) in enumerate(ordered):
        if start < 0 or start % 128 or end > 1 << 35:
            raise ValueError(f"Invalid/alignment-unsafe LSTM {name} address")
        if index and start < ordered[index - 1][1]:
            raise ValueError(f"Overlapping LSTM regions: {ordered[index - 1][2]} and {name}")

    layers = []
    for index in range(2):
        addresses = []
        recurrent_scales = None
        for prefix in ("weight_ih", "weight_hh"):
            weight = getattr(lstm_module, f"{prefix}_l{index}").detach().cpu()
            if weight.shape != (4 * width, width) or not torch.isfinite(weight).all():
                raise ValueError(f"Invalid {prefix} tensor")
            packed = torch.zeros((4, padded, padded), dtype=torch.bfloat16)
            packed[:, :width, :width] = weight.reshape(4, width, width)
            matrix = packed.reshape(4 * padded, padded)
            if prefix == "weight_hh" and recurrent_precision == "if8":
                codes, scales = quantize_recurrent_if8(matrix)
                addresses.append(image.allocate(codes, alignment=128))
                recurrent_scales = image.allocate(scales, alignment=128)
            else:
                addresses.append(image.allocate(matrix, alignment=128))
        for prefix in ("bias_ih", "bias_hh"):
            bias = getattr(lstm_module, f"{prefix}_l{index}").detach().cpu()
            if bias.shape != (4 * width,) or not torch.isfinite(bias).all():
                raise ValueError(f"Invalid {prefix} tensor")
            packed = torch.zeros((4, padded), dtype=torch.bfloat16)
            packed[:, :width] = bias.reshape(4, width)
            addresses.append(image.allocate(packed.flatten(), alignment=128))
        layers.append(LSTMLayer(*addresses, recurrent_scales=recurrent_scales))

    cursor = scratch_address
    regions = {}
    for name, count in (("input_gates", sequence * 4 * padded),
                        ("layer0_output", sequence * padded),
                        ("hidden", padded), ("cell", padded),
                        ("gates", 4 * padded), ("temporary", padded),
                        ("cell_tanh", padded),
                        ("tanh_scratch", tanh_scratch_bytes(padded) // 2)):
        regions[name] = (cursor, count)
        cursor += count * 2
    if cursor - scratch_address != workspace_bytes:
        raise AssertionError("LSTM workspace accounting error")
    return LSTMPlan(sequence, width, padded, input_address, output_address,
                    scratch_address, workspace_bytes, identity_address,
                    zero_address, bool(skip), tuple(layers), regions, recurrent_precision)


def _copy(engine, source: int, output: int, count: int) -> None:
    for offset in range(0, count, udc.URAM_NEAR_FULL_ELEMENTS):
        take = min(count - offset, udc.URAM_NEAR_FULL_ELEMENTS)
        engine.accelerator_memory_to_sram(source + offset * 2, 0, take)
        engine.sram_to_accelerator_memory(0, output + offset * 2, take)


def sigmoid_identity(engine, source: int, output: int, count: int,
                     identity_address: int) -> None:
    if count <= 0 or count % 64:
        raise ValueError("Sigmoid requires a positive multiple of 64 elements")
    engine.matmat_mul_core(M=count // 64, K=64, N=64,
                           A_DRAM_ADDR=source, B_DRAM_ADDR=identity_address,
                           OUTPUT_DRAM_ADDR=output, sigmoid_enable=True)


def _identity_sram(engine, source, output, count, identity, mode, *, lower=0., upper=0.):
    """Apply one native LALU operation without spilling the vector to DRAM."""
    if mode == udc.LALU_MODE.CLAMP and count >= 1024:
        from bigcodec_device import sram_clamp
        sram_clamp(engine, source, output, count, scratch_address=0x20000,
                   lo=lower, hi=upper)
        return
    kwargs = {}
    if mode == udc.LALU_MODE.CLAMP:
        kwargs.update(lalu_a=engine.float_to_bf16(lower), lalu_b=engine.float_to_bf16(upper))
    elif mode == udc.LALU_MODE.ACT_NO_X:
        kwargs.update(lalu_a=udc.LALU_ACT_SIGMOID_A, lalu_b=udc.LALU_ACT_SIGMOID_B)
    elif mode == udc.LALU_MODE.MODE_RECIP:
        kwargs.update(lalu_scalar=engine.float_to_bf19(1.0))
    for offset in range(0, count * 2, 128):
        engine.start_queue_for_bf16_matvec_operation(
            max_clear_en=0, fmax_context_addr=0,
            vector_sram_start_addr=source + offset,
            matrix_sram_start_addr=identity, output_sram_wb_addr=output + offset,
            K=64, N=64, lalu_mode=mode, **kwargs)


def _tanh_sram(engine, source, output, count, *, identity=0x80000):
    """Same rounded Padé expression as pade_tanh, with every vector in SRAM.

    Source/output occupy URAM_A below 0x10000. Workspaces are A[0x10000:]
    and B[0x90000:]; the 64x64 identity remains at B[0x80000:0x82000].
    Each operation still writes BF16, preserving every former DRAM boundary.
    """
    if count <= 0 or count % 64 or count > TANH_CHUNK_ELEMENTS:
        raise ValueError("SRAM tanh requires 64..4096 elements in multiples of 64")
    if any(address < 0 or address % 128 or address + count * 2 > 0x10000
           for address in (source, output)):
        raise ValueError("SRAM tanh source/output must occupy the low URAM_A workspace")
    size = count * 2
    square, numerator, denominator = (0x10000 + index * size for index in range(3))
    value_b, square_b, reciprocal_b = (0x90000 + index * size for index in range(3))
    _identity_sram(engine, source, output, count, identity, udc.LALU_MODE.CLAMP,
                   lower=-TANH_ARGUMENT_LIMIT, upper=TANH_ARGUMENT_LIMIT)
    engine.broadcast_mul(1.0, output, value_b, count)
    engine.eltwise_mul_core(output, value_b, square, count)
    engine.broadcast_mul(1.0, square, square_b, count)
    for destination, coefficients in ((numerator, _TANH_NUMERATOR_BF16),
                                       (denominator, _TANH_DENOMINATOR_BF16)):
        engine.broadcast_mul(coefficients[0], square, destination, count)
        for index, coefficient in enumerate(coefficients[1:]):
            engine.broadcast_add(coefficient, destination, destination, count)
            if index < 2:
                engine.eltwise_mul_core(destination, square_b, destination, count)
    _identity_sram(engine, denominator, reciprocal_b, count, identity,
                   udc.LALU_MODE.MODE_RECIP)
    engine.eltwise_mul_core(numerator, reciprocal_b, numerator, count)
    engine.eltwise_mul_core(numerator, value_b, output, count)
    _identity_sram(engine, output, output, count, identity, udc.LALU_MODE.CLAMP,
                   lower=-1., upper=1.)


def tanh_identity(engine, source: int, output: int, count: int,
                  identity_address: int, *, scratch_address: int) -> None:
    """Bounded odd Padé tanh using SRAM vectors and native reciprocal.

    The existing DRAM scratch reservation is retained for artifact compatibility;
    computation no longer reads/writes it. Source/output may alias each other.
    All coefficients and arithmetic rounding boundaries remain BF16.
    """
    scratch_size = tanh_scratch_bytes(count)
    if any(address % 128 for address in (source, output, identity_address, scratch_address)):
        raise ValueError("Tanh addresses must be 128-byte aligned")
    for address in (source, output):
        if scratch_address < address + count * 2 and address < scratch_address + scratch_size:
            raise ValueError("Tanh workspace overlaps input/output")
    engine.accelerator_memory_to_sram(identity_address, 0x80000, 64 * 64)
    for offset in range(0, count, TANH_CHUNK_ELEMENTS):
        take = min(count - offset, TANH_CHUNK_ELEMENTS)
        source_chunk, output_chunk = source + offset * 2, output + offset * 2
        engine.accelerator_memory_to_sram(source_chunk, 0, take)
        _tanh_sram(engine, 0, 0, take)
        engine.sram_to_accelerator_memory(0, output_chunk, take)


def _recurrent_projection_sram(engine, plan, layer, hidden):
    """Read each recurrent weight once, retaining all gate outputs in URAM_A."""
    width = plan.padded_width
    if plan.recurrent_precision == "if8":
        if layer.recurrent_scales is None:
            raise ValueError("IF8 recurrent layer is missing its scales")
        tile = min(4 * width, udc.SCALE_BRAM_ELEMENTS // width * 64,
                   udc.BIAS_BRAM_ELEMENTS)
    else:
        tile = min(4 * width, (udc.URAM_NEAR_FULL_ELEMENTS // width) // 64 * 64)
    if width > TANH_CHUNK_ELEMENTS or tile < 64:
        raise ValueError("Recurrent SRAM projection exceeds its bounded workspace")
    engine.accelerator_memory_to_sram(hidden, 0, width)
    for first in range(0, 4 * width, tile):
        take = min(4 * width - first, tile)
        if plan.recurrent_precision == "if8":
            engine.accelerator_memory_to_scale_sram(layer.recurrent_scales + first * width // 64 * 2,
                                                    take * width // 64)
            engine.accelerator_memory_to_bias_sram(layer.recurrent_bias + first * 2, take)
            engine.start_queue_for_dot_product_operation(
                max_clear_en=0, fmax_context_addr=0,
                vector_sram_start_addr=0, output_sram_wb_addr=0x8000 + first * 2,
                K=width, N=take, dma_start_addr=layer.recurrent_weights + first * width,
                data_type=udc.TYPE.IF8, bias_enable=True)
            continue
        engine.accelerator_memory_to_sram(layer.recurrent_weights + first * width * 2,
                                          0x80000, take * width)
        engine.accelerator_memory_to_bias_sram(layer.recurrent_bias + first * 2, take)
        engine.start_queue_for_bf16_matvec_operation(
            max_clear_en=0, fmax_context_addr=0,
            vector_sram_start_addr=0, matrix_sram_start_addr=0x80000,
            output_sram_wb_addr=0x8000 + first * 2,
            K=width, N=take, bias_enable=True)


def _lstm_step_sram(engine, plan, projection, previous_cell, output):
    """Fuse pointwise gate/state math after one recurrent projection."""
    width = plan.padded_width
    engine.accelerator_memory_to_sram(projection, 0x80000, 4 * width)
    engine.eltwise_add_core(0x8000, 0x80000, 0, 4 * width)
    engine.accelerator_memory_to_sram(plan.identity_address, 0x80000, 64 * 64)
    input_gate, forget_gate, candidate, output_gate = (index * width * 2 for index in range(4))
    for gate in (input_gate, forget_gate, output_gate):
        _identity_sram(engine, gate, gate, width, 0x80000, udc.LALU_MODE.ACT_NO_X)
    _tanh_sram(engine, candidate, candidate, width)
    cell_a, cell_b, temporary_b = 0xC000, 0xA0000, 0xB0000
    engine.accelerator_memory_to_sram(previous_cell, cell_b, width)
    engine.eltwise_mul_core(forget_gate, cell_b, temporary_b, width)
    engine.broadcast_mul(1.0, candidate, cell_b, width)
    engine.eltwise_mul_core(input_gate, cell_b, cell_a, width)
    engine.eltwise_add_core(cell_a, temporary_b, cell_a, width)
    engine.sram_to_accelerator_memory(cell_a, previous_cell, width)
    _tanh_sram(engine, cell_a, cell_a, width)
    engine.broadcast_mul(1.0, output_gate, cell_b, width)
    engine.eltwise_mul_core(cell_a, cell_b, 0, width)
    engine.sram_to_accelerator_memory(0, output, width)


def emit_lstm(engine, plan: LSTMPlan) -> None:
    """Emit all recurrence into the device program; no host tensor computation."""
    width = plan.padded_width
    address = {name: region[0] for name, region in plan.regions.items()}

    def binary(left, right, output, mode, count=width):
        engine.eltwise_core_dram(M=count // 64, N=64, dram_a=left,
                                 dram_b=right, dram_out=output, mode=mode)

    for layer_index, layer in enumerate(plan.layers):
        source = plan.input_address if layer_index == 0 else address["layer0_output"]
        destination = address["layer0_output"] if layer_index == 0 else plan.output_address
        # Initial h/c belong to this utterance and layer, not the preceding file.
        for name in ("hidden", "cell"):
            for offset in range(0, width, 64):
                _copy(engine, plan.zero_address, address[name] + offset * 2, 64)
        engine.matmat_mul_core(M=plan.sequence, K=width, N=4 * width,
                               A_DRAM_ADDR=source, B_DRAM_ADDR=layer.input_weights,
                               OUTPUT_DRAM_ADDR=address["input_gates"],
                               C_DRAM_ADDR=layer.input_bias, bias_mode="broadcast_N")
        for timestep in range(plan.sequence):
            sram_fusion = (width <= TANH_CHUNK_ELEMENTS
                and (plan.recurrent_precision == "if8" or width * 64 <= udc.URAM_NEAR_FULL_ELEMENTS))
            previous_hidden = (destination + (timestep - 1) * width * 2
                               if sram_fusion and timestep else address["hidden"])
            if sram_fusion:
                _recurrent_projection_sram(engine, plan, layer, previous_hidden)
                _lstm_step_sram(engine, plan,
                    address["input_gates"] + timestep * 4 * width * 2,
                    address["cell"], destination + timestep * width * 2)
                continue
            engine.matmat_mul_core(M=1, K=width, N=4 * width,
                                   A_DRAM_ADDR=previous_hidden,
                                   B_DRAM_ADDR=layer.recurrent_weights,
                                   OUTPUT_DRAM_ADDR=address["gates"],
                                   C_DRAM_ADDR=layer.recurrent_bias, bias_mode="broadcast_N")
            binary(address["input_gates"] + timestep * 4 * width * 2,
                   address["gates"], address["gates"], udc.UE_MODE.ELTWISE_ADD, 4 * width)
            input_gate, forget_gate, candidate, output_gate = (
                address["gates"] + index * width * 2 for index in range(4))
            for gate in (input_gate, forget_gate, output_gate):
                sigmoid_identity(engine, gate, gate, width, plan.identity_address)
            tanh_identity(engine, candidate, candidate, width, plan.identity_address,
                           scratch_address=address["tanh_scratch"])
            binary(forget_gate, address["cell"], address["temporary"], udc.UE_MODE.ELTWISE_MUL)
            binary(input_gate, candidate, address["cell"], udc.UE_MODE.ELTWISE_MUL)
            binary(address["temporary"], address["cell"], address["cell"], udc.UE_MODE.ELTWISE_ADD)
            tanh_identity(engine, address["cell"], address["cell_tanh"], width, plan.identity_address,
                           scratch_address=address["tanh_scratch"])
            binary(output_gate, address["cell_tanh"], address["hidden"], udc.UE_MODE.ELTWISE_MUL)
            _copy(engine, address["hidden"], destination + timestep * width * 2, width)
    if plan.skip:
        binary(plan.input_address, plan.output_address, plan.output_address,
               udc.UE_MODE.ELTWISE_ADD, plan.sequence * width)
