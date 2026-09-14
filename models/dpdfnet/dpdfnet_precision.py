"""Read dense convolution weight precision from a validated deployment program."""

from __future__ import annotations

import numpy as np

import user_dma_core as udc


PRECISION_LABELS = {"bf16": "BF16", "if8": "IF8", "if4_if8": "IF4/IF8"}


def validate_weight_precision(payload, expected=None):
    """Identify the encoded weights and enforce a named runner's selection.

    Call after the model-specific artifact validator. Older 16-kHz bins have
    an incomplete precision label, so names and metadata are not authoritative.
    This is a runtime selection guard; dpdfnet_audit_bf16.py separately checks
    BF16 weight bytes against ONNX and binds them to individual convolutions.
    """
    if expected is not None and expected not in PRECISION_LABELS:
        raise ValueError(f"Unsupported requested weight precision: {expected}")
    hardware = payload["hardware"]
    image = hardware["model_image"].numpy().tobytes()
    offset, size = hardware["program_offset"], hardware["program_size"]
    if offset < 0 or size <= 0 or size % 32 or offset + size > len(image):
        raise ValueError("Invalid program bounds for weight precision inspection")
    words = np.frombuffer(image[offset:offset + size], dtype="<u4").reshape(-1, 8)
    kinds = (words[:, 0] >> 8) & 15
    modes = (words[:, 5] >> 12) & 15
    types = (words[:, 5] >> 21) & 3
    operations = np.isin(kinds, (udc.INSTRUCTION_UE_OP, udc.INSTRUCTION_UE_PBI))
    bf16 = operations & (modes == udc.UE_MODE.BF16_DOT_PRODUCT)
    if not np.any(bf16) or np.any(types[bf16] != 0):
        raise ValueError("Expected BF16 matrix instructions in the DPDFNet program")
    if np.any(operations & np.isin(modes, (
            udc.UE_MODE.DOT_PRODUCT, udc.UE_MODE.QUANTIZE, udc.UE_MODE.DEQUANTIZE))):
        raise ValueError("Unsupported quantized arithmetic outside dense CONV")
    convolution_types = frozenset(map(int, types[
        operations & (modes == udc.UE_MODE.CONV2D)]))
    actual = {
        frozenset(): "bf16",
        frozenset((int(udc.TYPE.IF8),)): "if8",
        frozenset((int(udc.TYPE.IF4), int(udc.TYPE.IF8))): "if4_if8",
    }.get(convolution_types)
    if actual is None:
        raise ValueError(f"Unsupported dense convolution data types: {sorted(convolution_types)}")
    if expected is not None and actual != expected:
        raise ValueError(
            f"This test requires {PRECISION_LABELS[expected]} dense convolution weights; "
            f"the selected bin contains {PRECISION_LABELS[actual]}. Select the matching --bin")
    return PRECISION_LABELS[actual]
