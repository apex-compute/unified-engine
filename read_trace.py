#!/usr/bin/env python3
"""Read and export the contents of the BRAM trace buffer.

This script uses the `user_dma_core` utilities found in the same
folder.  The UnifiedEngine class already provides read_reg32()/write_reg32()
methods for accessing registers over the AXI-Lite user interface
(`/dev/xdma*_user`).

Usage:
    python3 read_trace.py [--output FILE] [--max N]

If the file already contains a write pointer (number of valid entries)
this script will dump all entries from index 0 up to pointer-1.  The
output is written as CSV "index,value" pairs.

Example:
    python3 read_trace.py --output trace.csv

"""
import csv
import os
from pathlib import Path
import sys
import torch

from user_dma_core import (
    INSTRUCTION_HALT,
    INSTRUCTION_JUMP,
    INSTRUCTION_SIZE_BYTES,
    UE_PIPELINE_COUNTER_CLK_DIV,
    UE_TRACE_BRAM_ADDR,
    UE_TRACE_BRAM_DATA,
    UE_TRACE_SIZE,
    UnifiedEngine,
)


def read_trace(ue: UnifiedEngine, instruction_count: int):
    """Retrieve trace values from the device.

    Returns:
        list of 32-bit integers read from the BRAM
    """
    trace = []
    for i in range(instruction_count):
        ue.write_reg32(UE_TRACE_BRAM_ADDR, i)  # set read address
        val = ue.read_reg32(UE_TRACE_BRAM_DATA)
        trace.append(val)
    return trace


def read_trace_addresses(ue: UnifiedEngine, addresses):
    """Read trace BRAM slots in the requested physical order."""
    values = []
    for address in addresses:
        ue.write_reg32(UE_TRACE_BRAM_ADDR, int(address))
        values.append(ue.read_reg32(UE_TRACE_BRAM_DATA))
    return values

def _pb_encode_varint(value):
    if value < 0:
        value += (1 << 64)
    parts = []
    while value > 0x7f:
        parts.append(0x80 | (value & 0x7f))
        value >>= 7
    parts.append(value & 0x7f)
    return bytes(parts)

def _pb_field_varint(field, val):
    return _pb_encode_varint((field << 3) | 0) + _pb_encode_varint(val)

def _pb_field_bytes(field, data):
    return _pb_encode_varint((field << 3) | 2) + _pb_encode_varint(len(data)) + data

def _pb_field_string(field, s):
    return _pb_field_bytes(field, s.encode('utf-8'))

def _pf_track_descriptor(uuid, name):
    return _pb_field_varint(1, uuid) + _pb_field_string(2, name)

def _pf_debug_annotation(name, value):
    msg = _pb_field_string(10, name)
    if isinstance(value, int):
        msg += _pb_field_varint(3, value) if value >= 0 else _pb_field_varint(4, value)
    else:
        msg += _pb_field_string(6, str(value))
    return msg

def _pf_track_event_begin(track_uuid, name, annotations=None):
    msg = _pb_field_varint(9, 1) + _pb_field_varint(11, track_uuid) + _pb_field_string(23, name)
    for a in (annotations or []):
        msg += _pb_field_bytes(4, a)
    return msg

def _pf_track_event_end(track_uuid):
    return _pb_field_varint(9, 2) + _pb_field_varint(11, track_uuid)

def _pf_packet(timestamp_ns=None, track_event=None, track_descriptor=None, seq_id=1, seq_flags=None):
    msg = b''
    if timestamp_ns is not None:
        msg += _pb_field_varint(8, timestamp_ns)
    msg += _pb_field_varint(10, seq_id)
    if track_event is not None:
        msg += _pb_field_bytes(11, track_event)
    if seq_flags is not None:
        msg += _pb_field_varint(13, seq_flags)
    if track_descriptor is not None:
        msg += _pb_field_bytes(60, track_descriptor)
    return msg


def build_perfetto(trace_values: list[int], ue: UnifiedEngine, out_path: str,
                   *, instructions=None, instruction_indices=None,
                   program_dram_addr=None):
    import io
    import contextlib
    events = []

    clock_period_ns = ue._clock_period_ns

    # Legacy callers decode the in-memory capture. Precompiled callers supply
    # the instructions recovered from their deployment artifact.
    insts = (list(instructions) if instructions is not None
             else ue.get_captured_instructions())
    if not insts:
        print("No instructions are available for trace decoding.")
        return False

    indices = (list(instruction_indices) if instruction_indices is not None
               else list(range(len(trace_values))))
    n = min(len(trace_values), len(indices))
    if any(index < 0 or index >= len(insts) for index in indices[:n]):
        raise ValueError("trace instruction index is outside the program")
    # Precompute timestamps in integer nanoseconds first. Perfetto internally
    # quantizes to ns; deriving ts/dur from the same ns grid avoids visual
    # 1ns gaps caused by float rounding.
    timestamps_ns = []
    for i in range(n):
        counter = trace_values[i]
        ts_ns = int(round(
            counter * UE_PIPELINE_COUNTER_CLK_DIV * clock_period_ns))
        timestamps_ns.append(ts_ns)

    # Print confirmation of conversion
    preview = [(trace_values[i], timestamps_ns[i] / 1000.0) for i in range(min(3, n))]
    print(
        f"Using clock period = {clock_period_ns:.6f} ns and trace divider "
        f"{UE_PIPELINE_COUNTER_CLK_DIV} -> timestamps in microseconds. "
        f"Example counter->us values: {preview}")

    # Keep absolute timestamps (do not rebase to 0). This preserves the
    # original hardware timebase even when earlier textual rows are skipped.
    if n == 0:
        print("No trace/instruction pairs to export.")
        return False

    TRACK_MAP = {
        "MEMCPY_FROM": 1,
        "MEMCPY_TO":   2,
        "COMPUTE":     3,
        "HALT":        4,
    }
    TRACK_LABELS = {
        0: "1-QUEUE",
        1: "2-DMA_FROM_DRAM",
        2: "4-DMA_TO_DRAM",
        3: "3-COMPUTE",
        4: "5-HALT",
    }

    def _pick_tracks(event_name: str, args: dict = None) -> list[int]:
        upper = event_name.upper()
        if "COMPUTE" in upper:
            # DOT_PRODUCT and DEQUANTIZE stream from DRAM during compute (dma_start=1).
            # BF16_DOT_PRODUCT is pure compute with no concurrent DMA.
            is_dma_op = (("DOT_PRODUCT" in upper and "BF16" not in upper)
                         or "DEQUANTIZE" in upper)
            dma_annotation = "enabled" in str(args.get("dma_start", "")).lower() if args else False
            if is_dma_op or dma_annotation:
                return [3, 1]  # COMPUTE + DMA_FROM_DRAM
            return [3]
        for key, tid in TRACK_MAP.items():
            if key in upper:
                return [tid]
        return [0]

    collected = []

    first_ts_ns = max(0, timestamps_ns[0])
    if first_ts_ns > 0:
        collected.append({
            "name": "UE_QUEUE_LOADING",
            "track": 0,
            "ts_ns": 0,
            "dur_ns": first_ts_ns,
            "args": {},
        })

    for i in range(n):
        counter = trace_values[i]
        ts_ns = timestamps_ns[i]

        is_last = (i == n - 1)
        if is_last:
            dur_ns = int(round(
                UE_PIPELINE_COUNTER_CLK_DIV * clock_period_ns))
        elif i < n - 1:
            dur_ns = max(0, timestamps_ns[i+1] - timestamps_ns[i])
        else:
            dur_ns = 0

        buf = io.StringIO()
        static_index = indices[i]
        instruction_address = (
            (ue.get_program_dram_addr()
             if program_dram_addr is None and hasattr(ue, "get_program_dram_addr")
             else (program_dram_addr or 0))
            + static_index * INSTRUCTION_SIZE_BYTES)
        with contextlib.redirect_stdout(buf):
            try:
                ue.parse_instruction(
                    insts[static_index], static_index, instruction_address)
            except Exception:
                try:
                    ue.parse_instruction(insts[static_index], static_index, 0)
                except Exception:
                    pass
        parsed = buf.getvalue().splitlines()

        hexdump = None
        detail_lines = []
        for line in parsed:
            s = line.rstrip()
            if not s:
                continue
            if hexdump is None and s.startswith('[') and '=' in s and '0x' in s:
                hexdump = s
                continue
            detail_lines.append(s.strip())

        name = None
        import re
        if detail_lines:
            mnemonic = None
            for dl in detail_lines:
                if re.match(r"^[A-Z][A-Z0-9_ ()-]+$", dl):
                    mnemonic = dl
                    break
            if mnemonic:
                token = mnemonic.strip()
                token = re.sub(r"[^A-Za-z0-9_ ()-]+", "", token)
                if not token:
                    token = f"inst_{i}"
                name = token[:32]
            else:
                first = detail_lines[0]
                token = first.split()[0] if first.split() else ''
                token = re.sub(r"[^A-Za-z0-9_()]+", "", token)
                token = token.strip('()')
                if not token:
                    fb = re.sub(r"[^A-Za-z0-9_]+", "_", first).strip('_')
                    token = fb if fb else f"inst_{i}"
                name = token[:32]
        else:
            name = f"inst_{i}"

        instruction_type = (int(insts[static_index].words[0]) >> 8) & 0xF
        if instruction_type == INSTRUCTION_HALT:
            name = "UE_HALT_INST"

        args = {
            "trace_row": i,
            "instruction_index": static_index,
            "instruction_address": instruction_address,
            "counter": counter,
            "hexdump": hexdump if hexdump is not None else "",
            "decoded_text": "\n".join(detail_lines),
        }
        for dl in detail_lines[1:]:
            if ':' in dl:
                key, val = dl.split(':', 1)
                key = key.strip().replace(' ', '_')
                val = val.strip()
                parsed_val = val
                if val.startswith('0x') or val.startswith('0X'):
                    try:
                        parsed_val = int(val, 16)
                    except Exception:
                        parsed_val = val
                else:
                    import re
                    m = re.match(r"^(-?\d+)", val)
                    if m:
                        try:
                            parsed_val = int(m.group(1))
                        except Exception:
                            parsed_val = val
                if key not in args:
                    args[key] = parsed_val
                else:
                    args[f"field_{key}"] = parsed_val

        if "MEMCPY" in name.upper() and "memcpy_type" in args:
            mt = str(args["memcpy_type"])
            import re as _re
            paren = _re.search(r"\(([^)]+)\)", mt)
            suffix = paren.group(1) if paren else mt
            name = f"{name} ({suffix})"

        for track_id in _pick_tracks(name, args):
            collected.append({
                "name": name,
                "track": track_id,
                "ts_ns": ts_ns,
                "dur_ns": dur_ns,
                "args": args,
            })

    # --- Generate native Perfetto protobuf trace (.pftrace) ---
    TRACK_UUIDS = {tid: 1000 + tid for tid in TRACK_LABELS}
    trace_bytes = b''

    for tid, label in TRACK_LABELS.items():
        td = _pf_track_descriptor(TRACK_UUIDS[tid], label)
        pkt = _pf_packet(track_descriptor=td, seq_flags=1)
        trace_bytes += _pb_field_bytes(1, pkt)

    for ev in collected:
        uuid = TRACK_UUIDS[ev['track']]
        annotations = [_pf_debug_annotation(k, v) for k, v in ev['args'].items()]

        te = _pf_track_event_begin(uuid, ev['name'], annotations)
        pkt = _pf_packet(timestamp_ns=ev['ts_ns'], track_event=te)
        trace_bytes += _pb_field_bytes(1, pkt)

        te = _pf_track_event_end(uuid)
        pkt = _pf_packet(timestamp_ns=ev['ts_ns'] + ev['dur_ns'], track_event=te)
        trace_bytes += _pb_field_bytes(1, pkt)

    try:
        with open(out_path, 'wb') as f:
            f.write(trace_bytes)
        print(f"Perfetto trace written to {out_path}")
        return True
    except OSError as e:
        print(f"Failed to write Perfetto trace: {e}")
        return False


def generate_circular_tail_trace(ue: UnifiedEngine, file_path: str | Path, *,
                                 instructions, program_dram_addr: int):
    """Export the chronological tail of one linear, HALT-terminated run.

    Trace BRAM is a circular buffer. Once 8192 decode events have occurred,
    its write pointer identifies the oldest retained slot rather than the
    number of valid rows.
    """
    insts = list(instructions)
    instruction_types = [
        (int(instruction.words[0]) >> 8) & 0xF for instruction in insts]
    halt_positions = [
        index for index, value in enumerate(instruction_types)
        if value == INSTRUCTION_HALT]
    if len(halt_positions) != 1:
        raise RuntimeError("trace export requires exactly one HALT")
    halt_index = halt_positions[0]
    if INSTRUCTION_JUMP in instruction_types[:halt_index]:
        raise RuntimeError(
            "tail trace decoding does not support control-flow instructions")

    executed_events = halt_index + 1
    retained_events = min(executed_events, UE_TRACE_SIZE)
    write_pointer = int(ue.read_reg32(UE_TRACE_BRAM_ADDR))
    expected_pointer = executed_events % UE_TRACE_SIZE
    if write_pointer != expected_pointer:
        raise RuntimeError(
            "trace write pointer does not match the whole-graph program: "
            f"read {write_pointer}, expected {expected_pointer} after "
            f"{executed_events} decode events")

    if executed_events < UE_TRACE_SIZE:
        physical_addresses = list(range(retained_events))
    else:
        physical_addresses = (
            list(range(write_pointer, UE_TRACE_SIZE))
            + list(range(write_pointer)))
    instruction_indices = list(range(
        executed_events - retained_events, executed_events))
    trace_values = read_trace_addresses(ue, physical_addresses)

    csv_path = Path(file_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("trace_row", "physical_address",
                         "instruction_index", "counter"))
        writer.writerows(zip(range(retained_events), physical_addresses,
                             instruction_indices, trace_values))

    perfetto_path = csv_path.with_name(csv_path.stem + "_perfetto.pftrace")
    if not build_perfetto(
            trace_values, ue, str(perfetto_path), instructions=insts,
            instruction_indices=instruction_indices,
            program_dram_addr=int(program_dram_addr)):
        raise RuntimeError("Perfetto tail trace generation failed")
    return {
        "csv": str(csv_path.resolve()),
        "perfetto": str(perfetto_path.resolve()),
        "retained_events": retained_events,
        "executed_events": executed_events,
        "write_pointer": write_pointer,
        "first_instruction_index": instruction_indices[0],
        "last_instruction_index": instruction_indices[-1],
    }


def generate_trace(ue: UnifiedEngine, file_path: str):

    if len(ue.get_captured_instructions()) > UE_TRACE_SIZE:
        print("Capture instruction size is too large to generate trace, skipping...")
        return

    traced_instruction_count = ue.read_reg32(UE_TRACE_BRAM_ADDR)
    print(f"traced instruction count = {traced_instruction_count}")

    trace_data = read_trace(ue, instruction_count=traced_instruction_count)
    print(f"read {len(trace_data)} entries")

    try:
        with open(file_path, "w") as f:
            for idx, val in enumerate(trace_data):
                f.write(f"{idx},{val}\n")
        print(f"saved to {file_path}")
    except OSError as e:
        print(f"error writing output file: {e}", file=sys.stderr)
        sys.exit(1)

    perf_out = os.path.splitext(file_path)[0] + "_perfetto.json"
    try:
        success = build_perfetto(trace_data, ue, perf_out)
        if not success:
            print("Perfetto generation failed")
    except Exception as e:
        print(f"Error generating perfetto JSON: {e}")
