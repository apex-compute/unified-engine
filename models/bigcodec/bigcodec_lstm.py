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


def scratch_bytes(input_shape: tuple[int, int]) -> int:
    """Workspace bytes; does not include input, output, weights or 64x64 identity."""
    if len(input_shape) != 2 or any(int(x) != x or x <= 0 for x in input_shape):
        raise ValueError("LSTM input_shape must be positive (T, D)")
    sequence, width = map(int, input_shape)
    padded = (width + 63) // 64 * 64
    return (5 * sequence + 8) * padded * 2 + tanh_scratch_bytes(padded)


def prepare_lstm(lstm_module, image, *, input_shape: tuple[int, int],
                 input_address: int, output_address: int, scratch_address: int,
                 identity_address: int, zero_address: int, skip: bool = True) -> LSTMPlan:
    """Pack PyTorch i/f/g/o parameters; image.allocate receives BF16 tensors.

    Inputs/outputs are time-major [T, pad64(D)], with zero padding lanes. The
    caller provides separate input, output and workspace regions, a 64x64 BF16
    identity, and at least 64 BF16 zeros. Parameters are only read at compilation.
    """
    workspace_bytes = scratch_bytes(input_shape)
    sequence, width = map(int, input_shape)
    if (not isinstance(lstm_module, torch.nn.LSTM)
            or lstm_module.input_size != width or lstm_module.hidden_size != width
            or lstm_module.num_layers != 2 or lstm_module.bidirectional
            or lstm_module.proj_size != 0 or not lstm_module.batch_first
            or not lstm_module.bias or lstm_module.training):
        raise ValueError("Expected an eval two-layer, unidirectional, batch-first LSTM with equal input/hidden width and biases")
    padded = (width + 63) // 64 * 64
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
        for prefix in ("weight_ih", "weight_hh"):
            weight = getattr(lstm_module, f"{prefix}_l{index}").detach().cpu()
            if weight.shape != (4 * width, width) or not torch.isfinite(weight).all():
                raise ValueError(f"Invalid {prefix} tensor")
            packed = torch.zeros((4, padded, padded), dtype=torch.bfloat16)
            packed[:, :width, :width] = weight.reshape(4, width, width)
            addresses.append(image.allocate(packed.reshape(4 * padded, padded), alignment=128))
        for prefix in ("bias_ih", "bias_hh"):
            bias = getattr(lstm_module, f"{prefix}_l{index}").detach().cpu()
            if bias.shape != (4 * width,) or not torch.isfinite(bias).all():
                raise ValueError(f"Invalid {prefix} tensor")
            packed = torch.zeros((4, padded), dtype=torch.bfloat16)
            packed[:, :width] = bias.reshape(4, width)
            addresses.append(image.allocate(packed.flatten(), alignment=128))
        layers.append(LSTMLayer(*addresses))

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
                    zero_address, bool(skip), tuple(layers), regions)


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


def tanh_identity(engine, source: int, output: int, count: int,
                  identity_address: int, *, scratch_address: int) -> None:
    """Bounded odd Padé tanh, with three reusable vectors and native reciprocal.

    Scratch size is tanh_scratch_bytes(count), capped at 24 KiB. Source/output
    may alias each other; scratch must be separate. All coefficients are BF16
    constants, exactly representable by the device's wider scalar encoding.
    """
    scratch_size = tanh_scratch_bytes(count)
    if any(address % 128 for address in (source, output, identity_address, scratch_address)):
        raise ValueError("Tanh addresses must be 128-byte aligned")
    for address in (source, output):
        if scratch_address < address + count * 2 and address < scratch_address + scratch_size:
            raise ValueError("Tanh workspace overlaps input/output")
    for offset in range(0, count, TANH_CHUNK_ELEMENTS):
        take = min(count - offset, TANH_CHUNK_ELEMENTS)
        source_chunk, output_chunk = source + offset * 2, output + offset * 2
        square, numerator, denominator = (scratch_address + index * take * 2 for index in range(3))

        def binary(left, right, destination, mode):
            engine.eltwise_core_dram(M=take // 64, N=64, dram_a=left,
                                     dram_b=right, dram_out=destination, mode=mode)

        def scalar(address, destination, value, mode):
            engine.eltwise_core_dram(M=take // 64, N=64, dram_a=address,
                                     dram_b=None, dram_out=destination, mode=mode, scalar=value)

        def clamp(address, destination, lower, upper):
            engine.matmat_mul_core(M=take // 64, K=64, N=64,
                A_DRAM_ADDR=address, B_DRAM_ADDR=identity_address,
                OUTPUT_DRAM_ADDR=destination, clamp_enable=True,
                clamp_min=lower, clamp_max=upper)

        clamp(source_chunk, output_chunk, -TANH_ARGUMENT_LIMIT, TANH_ARGUMENT_LIMIT)
        binary(output_chunk, output_chunk, square, udc.UE_MODE.ELTWISE_MUL)
        for destination, coefficients in ((numerator, _TANH_NUMERATOR_BF16), (denominator, _TANH_DENOMINATOR_BF16)):
            scalar(square, destination, coefficients[0], udc.UE_MODE.MUL_BROADCAST)
            for index, coefficient in enumerate(coefficients[1:]):
                scalar(destination, destination, coefficient, udc.UE_MODE.ADD_BROADCAST)
                if index < 2:
                    binary(destination, square, destination, udc.UE_MODE.ELTWISE_MUL)
        # Reciprocal is a native LALU operation on each identity matvec output.
        engine.accelerator_memory_to_sram(identity_address, 0x80000, 64 * 64)
        for row in range(take // 64):
            address = denominator + row * 128
            engine.accelerator_memory_to_sram(address, 0, 64)
            engine.start_queue_for_bf16_matvec_operation(
                max_clear_en=0, fmax_context_addr=0, vector_sram_start_addr=0,
                matrix_sram_start_addr=0x80000, output_sram_wb_addr=0,
                K=64, N=64, lalu_mode=udc.LALU_MODE.MODE_RECIP,
                lalu_scalar=engine.float_to_bf19(1.0))
            engine.sram_to_accelerator_memory(0, address, 64)
        binary(numerator, denominator, numerator, udc.UE_MODE.ELTWISE_MUL)
        binary(output_chunk, numerator, output_chunk, udc.UE_MODE.ELTWISE_MUL)
        clamp(output_chunk, output_chunk, -1.0, 1.0)


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
            engine.matmat_mul_core(M=1, K=width, N=4 * width,
                                   A_DRAM_ADDR=address["hidden"],
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
