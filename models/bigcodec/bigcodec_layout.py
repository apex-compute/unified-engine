"""Accepted BigCodec memory layouts within the RK board's 2 GiB DRAM."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryLayout:
    name: str
    input_base: int
    input_limit: int
    model_base: int
    model_limit: int
    tensor_base: int
    tensor_limit: int


# Retain the original addresses for existing bins and short-utterance captures.
LEGACY_LAYOUT = MemoryLayout(
    "legacy", 0x80000000, 0x90000000, 0x90000000, 0xB0000000,
    0xB0000000, 0xD0000000)
EXTENDED_LAYOUT = MemoryLayout(
    "extended", 0x80000000, 0x90000000, 0x90000000, 0xD0000000,
    0xD0000000, 0x100000000)
LAYOUTS = (LEGACY_LAYOUT, EXTENDED_LAYOUT)


def layout_for_hardware(hardware):
    """Accept an exact known arena layout; legacy bins need no extra tag."""
    keys = ("model_base", "model_limit", "tensor_base", "tensor_limit")
    if not isinstance(hardware, dict):
        raise ValueError("BigCodec hardware must specify an accepted memory layout")
    values = tuple(hardware.get(key) for key in keys)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise ValueError("BigCodec memory layout bounds must be integers")
    for layout in LAYOUTS:
        if values == tuple(getattr(layout, key) for key in keys):
            return layout
    raise ValueError("BigCodec memory layout does not match an accepted 2 GiB layout")
