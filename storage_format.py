"""16-bit tensor storage formats; arithmetic remains BF19.

Each compute descriptor carries one independent format bit per storage
boundary, with BF16 selected by zero and FP16 selected by one.
"""

from dataclasses import dataclass, fields, replace
from enum import IntEnum
import math
import struct

import torch


UE_TENSOR_FORMATS_ADDR = 0xB4


class StorageFormat(IntEnum):
    BF16 = 0
    FP16 = 1

    @property
    def dtype(self):
        return torch.bfloat16 if self == StorageFormat.BF16 else torch.float16

    def encode_scalar(self, value: float) -> int:
        """Encode an ISA storage scalar, retaining historical BF16 truncation."""
        value = float(value)
        if self == StorageFormat.BF16:
            return struct.unpack('<I', struct.pack('<f', value))[0] >> 16
        # IEEE binary16 round-to-nearest-even, including signed zero and
        # gradual underflow. struct raises on overflow whereas IEEE returns Inf.
        try:
            return struct.unpack('<H', struct.pack('<e', value))[0]
        except OverflowError:
            return 0xFC00 if math.copysign(1.0, value) < 0 else 0x7C00


def as_storage_format(value) -> StorageFormat:
    """Accept the ABI enum/bit or the corresponding torch dtype, never reinterpret data."""
    if value == torch.bfloat16:
        return StorageFormat.BF16
    if value == torch.float16:
        return StorageFormat.FP16
    try:
        return StorageFormat(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"Expected BF16 or FP16 storage, got {value!r}") from error


@dataclass(frozen=True)
class TensorFormats:
    """Formats at an operation's eight independent 16-bit boundaries.

    A/B name physical URAM operands. Output also selects quantizer scale storage.
    Bias, input scales, TQ4 codebook, broadcast scalar, and LALU parameters are
    independent. Retained FMAX/reciprocal and raw 21-bit scalars remain BF19.
    """
    a: StorageFormat = StorageFormat.BF16
    b: StorageFormat = StorageFormat.BF16
    output: StorageFormat = StorageFormat.BF16
    bias: StorageFormat = StorageFormat.BF16
    scale: StorageFormat = StorageFormat.BF16
    codebook: StorageFormat = StorageFormat.BF16
    scalar: StorageFormat = StorageFormat.BF16
    params: StorageFormat = StorageFormat.BF16

    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(self, field.name, as_storage_format(getattr(self, field.name)))

    @classmethod
    def uniform(cls, storage=StorageFormat.BF16):
        return cls(**{field.name: storage for field in fields(cls)})

    def with_formats(self, **changes):
        return replace(self, **changes)

    @property
    def bits(self) -> int:
        return (int(self.a) | (int(self.b) << 1) | (int(self.output) << 2) |
                (int(self.bias) << 3) | (int(self.scale) << 4) |
                (int(self.codebook) << 5) | (int(self.scalar) << 6) |
                (int(self.params) << 7))
