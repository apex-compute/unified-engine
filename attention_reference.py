"""FP32 attention targets and storage-boundary references with exact ISA scales."""

import math
import operator

import torch

from storage_format import StorageFormat, as_storage_format


def snr_db(reference, actual):
    """Measure finite tensors in FP64, including exact and zero-signal cases."""
    reference, actual = reference.double(), actual.double()
    if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
        raise AssertionError("SNR requires finite reference and output")
    error = (actual - reference).square().sum().item()
    if error == 0:
        return math.inf
    signal = reference.square().sum().item()
    return 10 * math.log10(signal / error) if signal else -math.inf


def _decode_scale(scale_bits, scale_format):
    """Reinterpret the programmed 16-bit word; never re-round its coefficient."""
    scale_bits = operator.index(scale_bits)
    if not 0 <= scale_bits <= 0xFFFF:
        raise ValueError("scale_bits must be an unsigned 16-bit storage word")
    storage = as_storage_format(scale_format)
    return torch.tensor(scale_bits, dtype=torch.uint16).view(storage.dtype).item()


def _validate_inputs(q, k, v, bias, *, allow_fp32=False):
    def validate_dtype(tensor):
        if not (allow_fp32 and tensor.dtype == torch.float32):
            as_storage_format(tensor.dtype)

    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional stored tensor")
        validate_dtype(tensor)
        if tensor.device != q.device:
            raise ValueError("Attention operands must share one device")
    if q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0]:
        raise ValueError("Attention Q/K contraction and K/V sequence dimensions must match")
    if k.shape[0] == 0:
        raise ValueError("Attention requires a nonempty key/value sequence")
    if bias is not None:
        if not isinstance(bias, torch.Tensor):
            raise TypeError("bias must be a stored tensor or None")
        validate_dtype(bias)
        if bias.device != q.device or tuple(bias.shape) not in (
                (k.shape[0],), (q.shape[0], k.shape[0])):
            raise ValueError("bias must share the operand device and have shape [N] or [M, N]")


def _chunked_reference(q, k, v, bias, *, scale, scratch_format, output_dtype,
                       query_chunk_size, arithmetic_dtype=torch.float64):
    _validate_inputs(q, k, v, bias, allow_fp32=arithmetic_dtype == torch.float32)
    query_chunk_size = operator.index(query_chunk_size)
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    keys, values = k.to(arithmetic_dtype), v.to(arithmetic_dtype)
    output = torch.empty((q.shape[0], v.shape[1]), dtype=output_dtype, device=q.device)
    for start in range(0, q.shape[0], query_chunk_size):
        stop = min(start + query_chunk_size, q.shape[0])
        scaled_q = q[start:stop].to(arithmetic_dtype) * scale
        if scratch_format is not None:
            scaled_q_dtype = (torch.float16 if scratch_format == StorageFormat.FP16 else q.dtype)
            scaled_q = scaled_q.to(scaled_q_dtype).to(arithmetic_dtype)
        logits = scaled_q @ keys.T
        del scaled_q
        if bias is not None:
            logits += (bias if bias.ndim == 1 else bias[start:stop]).to(arithmetic_dtype)
        if scratch_format is not None:
            # The hardware fuses bias before the logits writeback boundary.
            logits = logits.to(scratch_format.dtype).to(arithmetic_dtype)
        probabilities = logits.softmax(dim=-1)
        del logits
        if scratch_format is not None:
            probabilities = probabilities.to(scratch_format.dtype).to(arithmetic_dtype)
        output[start:stop] = (probabilities @ values).to(output_dtype)
        del probabilities
    return output


def storage_attention_reference_fp64(q, k, v, bias=None, *, scale_bits,
                                     scale_format=StorageFormat.BF16,
                                     scratch_format=StorageFormat.FP16,
                                     query_chunk_size=128):
    """Model selected intermediate storage, returning an unrounded FP64 output.

    Scale Q into FP16 for FP16 scratch, otherwise into its input type.
    Fuse QK+bias before storing logits, and
    store softmax probabilities in ``scratch_format``. Arithmetic uses FP64;
    hardware EXP approximation and accumulator rounding are not modeled.
    """
    return _chunked_reference(q, k, v, bias,
        scale=_decode_scale(scale_bits, scale_format),
        scratch_format=as_storage_format(scratch_format),
        output_dtype=torch.float64, query_chunk_size=query_chunk_size)


def fp32_attention_reference(q, k, v, bias=None, *, scale_bits,
                             scale_format=StorageFormat.BF16,
                             query_chunk_size=128):
    """Compute a fixed FP32 target from CPU source tensors and the ISA scale.

    Q/K/V and bias may contain original FP32 values or stored BF16/FP16 values.
    Every arithmetic stage uses FP32: Q scaling, QK plus bias, softmax and P @ V.
    The returned target stays FP32, independently of hardware input, scratch or
    output formats. Pass the same original FP32 tensors for all hardware format
    profiles to include their input quantization error in the comparison.

    CPU arithmetic avoids TF32, and local autocast is disabled. Query chunks
    are capped at 128 rows so long sequences do not allocate an M-by-N score
    matrix. Neither input tensors nor global precision settings are changed.
    """
    _validate_inputs(q, k, v, bias, allow_fp32=True)
    if q.device.type != "cpu":
        raise ValueError("FP32 attention reference requires CPU operands")
    with torch.autocast(device_type="cpu", enabled=False):
        return _chunked_reference(q, k, v, bias,
            scale=_decode_scale(scale_bits, scale_format), scratch_format=None,
            output_dtype=torch.float32, arithmetic_dtype=torch.float32,
            query_chunk_size=min(operator.index(query_chunk_size), 128))
