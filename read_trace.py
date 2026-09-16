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
import sys
import os
import re

from user_dma_core import (
    UnifiedEngine,
    UE_TRACE_BRAM_ADDR,
    UE_TRACE_BRAM_DATA,
    UE_TRACE_SIZE,
    UE_INSTRUCTION_CTL_ADDR,
    UE_PIPELINE_COUNTER_CLK_DIV,
    _inst_desc_bits,
    INSTRUCTION_REG_ALU,
    INSTRUCTION_REG_ALU_NONPREFETCH,
    INSTRUCTION_PBI_SET,
    INSTRUCTION_JUMP,
    INSTRUCTION_HALT,
    INSTRUCTION_SIZE_BYTES,
    JUMP_MODE_ABSOLUTE,
    JUMP_MODE_REG_ABS,
    JUMP_MODE_JNZ,
    JUMP_MODE_JZ,
    JUMP_MODE_RELATIVE,
    JUMP_MODE_RELA_JNZ,
    JUMP_MODE_RELA_JZ,
    JUMP_MODE_REG_RELA,
    ALU_MODE_INC,
    ALU_MODE_DEC,
    ALU_MODE_ADD_REG,
    ALU_MODE_ADD_IMM,
    ALU_MODE_SET,
    ALU_MODE_MIN,
    ALU_MODE_SUB,
    ALU_MODE_SHR,
    ALU_MODE_SHL,
    ALU_MODE_MUL32_REG,
    ALU_MODE_MUL32_IMM,
    ALU_MODE_DIV_REG,
    ALU_MODE_MUL_SHL,
    ALU_MODE_MUL_SHR,
    INSTRUCTION_NOP,
    INSTRUCTION_UE_OP,
    INSTRUCTION_UE_PBI,
)

_ENGINE_INST_TYPES = {INSTRUCTION_UE_OP, INSTRUCTION_UE_PBI}
# The only two encodings that retire while the engine is busy; the aliases
# are the PREFETCH variants (see user_dma_core). Everything else waits on
# !engine_busy, so the first non-prefetch retire after an engine op marks
# the end of that op's busy window.
_PREFETCH_INST_TYPES = {INSTRUCTION_REG_ALU, INSTRUCTION_PBI_SET}

# RTL inst RAM line: 16384 bytes / 32-byte descriptor = 512 words
# (queue_state_module.sv INST_LINE_WORDS).
INST_LINE_WORDS = 512

# TRACE BRAM word: bit[31]=1 is an i-cache fill (not an ISA retire).
# Start and done are consecutive tagged words; bits[30:0] are pipeline_counter.
TRACE_QUEUE_BIT = 1 << 31
TRACE_TICK_MASK = (1 << 31) - 1


def split_trace_bram_words(raw: list[int]) -> tuple[list[int], list[dict]]:
    """Split TRACE BRAM words into retire ticks and i-cache fill events.

    Tagged rows (bit 31) are queue DMA, not instruction retires. Consecutive
    tags are start then done. Legacy dumps never set bit 31, so ``fills`` is
    empty and the jump-gap heuristic remains.

    Each fill records ``retires_before`` (ISA retire count preceding the start
    stamp) so Perfetto can label absolute-jump vs walking off the 512-word line.
    """
    retires: list[int] = []
    fills: list[dict] = []
    pending_start: int | None = None
    pending_start_row: int | None = None
    pending_retires_before: int = 0
    n_retires = 0
    for i, word in enumerate(raw):
        w = int(word) & 0xFFFFFFFF
        if w & TRACE_QUEUE_BIT:
            tick = w & TRACE_TICK_MASK
            if pending_start is None:
                pending_start = tick
                pending_start_row = i
                pending_retires_before = n_retires
            else:
                fills.append({
                    "start_tick": pending_start,
                    "done_tick": tick,
                    "bram_row": i,
                    "retires_before": pending_retires_before,
                })
                pending_start = None
                pending_start_row = None
        else:
            retires.append(w)
            n_retires += 1
    if pending_start is not None:
        fills.append({
            "start_tick": pending_start,
            "done_tick": pending_start,
            "bram_row": pending_start_row if pending_start_row is not None else len(raw),
            "retires_before": pending_retires_before,
        })
    return retires, fills


def _queue_loading_reason(ev: dict, pc_rows: list[dict] | None) -> str:
    """Name a tagged fill: absolute-jump DMA vs i-cache line empty.

    The previous ISA retire (if any) is the jump that entered RAM_DMA_START.
    No prior retire, or a non-jump retire, is FETCH ``inst_ram_empty`` (start
    of program or sequential 512-word miss).
    """
    rb = int(ev.get("retires_before", 0) or 0)
    if pc_rows and rb > 0:
        prev = pc_rows[rb - 1] if rb - 1 < len(pc_rows) else {}
        if _jump_triggers_icache_refill(prev):
            mode = str(prev.get("jump_mode", "") or "JUMP").upper()
            taken = str(prev.get("taken", "") or "").strip()
            if mode in ("ABSOLUTE", "REG_ABS"):
                return "absolute jump"
            reason = mode
            if taken:
                reason = f"{mode} {taken}"
            return reason
    return "icache empty"


def _jump_triggers_icache_refill(row: dict) -> bool:
    """True when RTL JUMP goes to STATE_RAM_DMA_START (full i-cache line DMA)."""
    mode = str(row.get("jump_mode", "")).upper()
    taken = str(row.get("taken", "")).lower() == "taken"
    if mode in ("ABSOLUTE", "REG_ABS"):
        return True
    if mode in ("JNZ", "JZ") and taken:
        return True
    return False


def _isa_jump_perfetto_name(jump_mode: str, taken: str, nest: str = "") -> str:
    """ISA_JUMP slice name. Refill is a following QUEUE_LOADING, not the jump itself."""
    mode = (jump_mode or "JUMP").upper()
    taken_s = (taken or "").strip().lower()
    parts = [mode]
    if nest:
        parts.append(nest)
    if taken_s in ("taken", "fall") and mode in ("JNZ", "JZ", "RELA_JNZ", "RELA_JZ"):
        parts.append(taken_s)
    core = " ".join(parts)
    if _jump_triggers_icache_refill({"jump_mode": mode, "taken": taken_s}):
        return f"ISA_JUMP ({core}, icache refill)"
    return f"ISA_JUMP ({core})"


def _perfetto_event_title(detail_lines: list[str], inst_obj, trace_row: int) -> str:
    """
    Turn parse_instruction() output into a short Perfetto slice name.

    The old regex required lines like ``UE_COMPUTE (BF16_DOT_PRODUCT)`` to match
    ``^[A-Z][A-Z0-9_ ()-]+$``, which rejects dots; ``PBI_SET (...)=`` rejects ``=``.
    We instead take the first decoder header line (UE_/ISA_/PBI_).
    """
    for dl in detail_lines:
        s = dl.strip()
        if s.startswith(("UE_", "ISA_", "PBI_")):
            return s[:56]
    for dl in detail_lines:
        s = dl.strip()
        if s.startswith("ISA_UNKNOWN") or "UNKNOWN(" in s:
            return s[:56]
    if inst_obj is not None:
        it = _inst_desc_bits(inst_obj.words, 8, 11)
        return f"ISA_INST (type={it}) r{trace_row}"
    return f"ISA_DECODE_{trace_row}"


def _isa_immediate_u32(w: list) -> int:
    """32-bit ISA immediate: inst_descriptor[85:54] (same as ue_isa_descriptor / RTL)."""
    return _inst_desc_bits(w, 54, 85) & 0xFFFFFFFF


def _isa_reg_fields(w: list) -> tuple[int, int, int, int, int]:
    """Return (isa_mode, src, dst, rst, imm32) using the 6-bit GPR fields."""
    return (
        _inst_desc_bits(w, 32, 35),
        _inst_desc_bits(w, 36, 41),
        _inst_desc_bits(w, 42, 47),
        _inst_desc_bits(w, 48, 53),
        _isa_immediate_u32(w),
    )


def _apply_trace_regfile_add(inst, regs: list) -> None:
    """Mirror REG_ALU (prefetch and non-prefetch) so JNZ/JZ see the same GPRs as HW."""
    w = inst.words
    itype = (w[0] >> 8) & 0xF
    if itype not in (INSTRUCTION_REG_ALU, INSTRUCTION_REG_ALU_NONPREFETCH):
        return
    isa_mode, src, dst, rst, imm32 = _isa_reg_fields(w)
    if dst == 0:
        return
    a = regs[src] & 0xFFFFFFFF
    b = regs[rst] & 0xFFFFFFFF
    sh = imm32 & 0x1F
    if isa_mode == ALU_MODE_INC:
        regs[dst] = (a + 1) & 0xFFFFFFFF
    elif isa_mode == ALU_MODE_DEC:
        regs[dst] = (a - 1) & 0xFFFFFFFF
    elif isa_mode == ALU_MODE_ADD_REG:
        regs[dst] = (a + b) & 0xFFFFFFFF
    elif isa_mode == ALU_MODE_ADD_IMM:
        regs[dst] = (a + imm32) & 0xFFFFFFFF
    elif isa_mode == ALU_MODE_SET:
        regs[dst] = imm32 & 0xFFFFFFFF
    elif isa_mode == ALU_MODE_MIN:
        regs[dst] = a if a < b else b
    elif isa_mode == ALU_MODE_SUB:
        regs[dst] = (a - b) & 0xFFFFFFFF
    elif isa_mode == ALU_MODE_SHR:
        regs[dst] = (a >> sh) & 0xFFFFFFFF
    elif isa_mode == ALU_MODE_SHL:
        regs[dst] = (a << sh) & 0xFFFFFFFF
    elif isa_mode == ALU_MODE_MUL32_REG:
        regs[dst] = (a * b) & 0xFFFFFFFF
    elif isa_mode == ALU_MODE_MUL32_IMM:
        regs[dst] = (a * (imm32 & 0xFFFF)) & 0xFFFFFFFF
    elif isa_mode == ALU_MODE_DIV_REG:
        regs[dst] = (a // b) & 0xFFFFFFFF if b else 0
    elif isa_mode == ALU_MODE_MUL_SHL:
        regs[dst] = ((a * b) << sh) & 0xFFFFFFFF
    elif isa_mode == ALU_MODE_MUL_SHR:
        regs[dst] = ((a * b) >> sh) & 0xFFFFFFFF


def _lookup_inst_at_byte(
    images: list[tuple[int, list]],
    target_byte: int,
) -> tuple[int, list, int] | None:
    """Return (image_base, insts, idx) for a DRAM instruction byte address."""
    for start, insts in images:
        off = int(target_byte) - int(start)
        if 0 <= off < len(insts) * INSTRUCTION_SIZE_BYTES and off % INSTRUCTION_SIZE_BYTES == 0:
            return int(start), insts, off // INSTRUCTION_SIZE_BYTES
    return None


def _absolute_jump_switch_image(
    imm_word_addr: int,
    images: list[tuple[int, list]],
    cur_base: int,
    cur_insts: list,
    rp_after_fetch: int,
    warnings: list,
) -> tuple[int, list, int]:
    """Map an absolute jump immediate (byte_addr >> 3) onto a written program image."""
    target_byte = (imm_word_addr & 0xFFFFFFFF) << 3
    found = _lookup_inst_at_byte(images, target_byte)
    if found is not None:
        return found
    n_insts = len(cur_insts)
    tgt_idx = (target_byte - int(cur_base)) // INSTRUCTION_SIZE_BYTES
    if 0 <= tgt_idx < n_insts:
        return cur_base, cur_insts, tgt_idx
    warnings.append(
        f"absolute jump target word {imm_word_addr:#x} "
        f"(byte {target_byte:#x}) is not in any written program image"
    )
    return cur_base, cur_insts, rp_after_fetch


def _advance_pc_after_decode(
    inst,
    regs: list,
    rp_after_fetch: int,
    images: list[tuple[int, list]],
    cur_base: int,
    cur_insts: list,
    warnings: list,
) -> tuple[int, list, int]:
    """
    Return (image_base, insts, inst_ram_read_ptr) before the next FETCH.

    ``rp_after_fetch`` is the RTL read pointer *after* fetching ``inst``
    (decoded_idx + 1), matching queue_state_module.sv for RELATIVE / RELA_*.
    Absolute jumps refetch the i-cache from ``immediate << 3``; we switch to
    the program image that contains that DRAM address (preamble → main).
    """
    w = inst.words
    itype = (w[0] >> 8) & 0xF
    if itype != INSTRUCTION_JUMP:
        return cur_base, cur_insts, rp_after_fetch

    isa_mode, src_reg, _dst, rst_reg, imm32 = _isa_reg_fields(w)
    rel_off = imm32 & 0x3FF  # RTL: inst_immediate_value[9:0]

    def _rel(off: int) -> tuple[int, list, int]:
        rp = (rp_after_fetch - off) if rp_after_fetch >= off else 0
        return cur_base, cur_insts, rp

    if isa_mode == JUMP_MODE_RELATIVE:
        return _rel(rel_off)
    if isa_mode == JUMP_MODE_RELA_JNZ:
        return _rel(rel_off) if regs[src_reg] != 0 else (cur_base, cur_insts, rp_after_fetch)
    if isa_mode == JUMP_MODE_RELA_JZ:
        return _rel(rel_off) if regs[src_reg] == 0 else (cur_base, cur_insts, rp_after_fetch)
    if isa_mode == JUMP_MODE_REG_RELA:
        return _rel(regs[rst_reg] & 0x3FF)
    if isa_mode == JUMP_MODE_ABSOLUTE:
        return _absolute_jump_switch_image(imm32, images, cur_base, cur_insts, rp_after_fetch, warnings)
    if isa_mode == JUMP_MODE_REG_ABS:
        return _absolute_jump_switch_image(
            regs[rst_reg], images, cur_base, cur_insts, rp_after_fetch, warnings
        )
    if isa_mode == JUMP_MODE_JNZ:
        if regs[src_reg] != 0:
            return _absolute_jump_switch_image(
                imm32, images, cur_base, cur_insts, rp_after_fetch, warnings
            )
        return cur_base, cur_insts, rp_after_fetch
    if isa_mode == JUMP_MODE_JZ:
        if regs[src_reg] == 0:
            return _absolute_jump_switch_image(
                imm32, images, cur_base, cur_insts, rp_after_fetch, warnings
            )
        return cur_base, cur_insts, rp_after_fetch
    warnings.append(f"unknown JUMP mode {isa_mode}; linear fall-through")
    return cur_base, cur_insts, rp_after_fetch


def trace_dynamic_decode_map(
    images: list[tuple[int, list]],
    n_trace: int,
    start_byte: int | None,
) -> tuple[list[tuple[int, int, object]], list[str]]:
    """
    Replay FETCH/DECODE for ``n_trace`` TRACE rows across written program images.

    Each row is ``(image_base, idx_in_image, inst)``. Starts at ``start_byte``
    (execute PC). Absolute jumps switch images so a preamble ADD_SET + abs jump
    into the PBI body still primes M/K/N and reaches writeback / HALT.
    """
    warnings: list[str] = []
    if n_trace <= 0:
        return [], []
    if not images:
        return [], ["no program images; cannot map trace rows"]

    loc = _lookup_inst_at_byte(images, int(start_byte)) if start_byte is not None else None
    if loc is None:
        cur_base, cur_insts = images[0][0], images[0][1]
        rp = 0
        if start_byte is not None:
            warnings.append(
                f"execute addr {int(start_byte):#x} not in any program image; "
                f"starting at {cur_base:#x}"
            )
    else:
        cur_base, cur_insts, rp = loc

    out: list[tuple[int, int, object]] = []
    regs = [0] * 64

    for k in range(n_trace):
        n_insts = len(cur_insts)
        if rp < 0:
            warnings.append(f"decode {k}: inst_ram_read_ptr negative, clamped to 0")
            rp = 0
        if rp >= n_insts:
            warnings.append(
                f"decode {k}: inst_ram_read_ptr={rp} >= image@{cur_base:#x} len={n_insts}; "
                "padding with last index"
            )
            tail = out[-1] if out else (cur_base, max(0, n_insts - 1), cur_insts[-1] if cur_insts else None)
            while len(out) < n_trace:
                out.append(tail)
            break

        idx = rp
        inst = cur_insts[idx]
        rp = rp + 1
        out.append((cur_base, idx, inst))

        _apply_trace_regfile_add(inst, regs)
        regs[0] = 0

        itype = _inst_desc_bits(inst.words, 8, 11)
        if itype == INSTRUCTION_HALT:
            last = (cur_base, idx, inst)
            while len(out) < n_trace:
                out.append(last)
            break

        cur_base, cur_insts, rp = _advance_pc_after_decode(
            inst, regs, rp, images, cur_base, cur_insts, warnings
        )

    return out, warnings


def trace_dynamic_inst_indices(
    insts: list,
    n_trace: int,
    program_dram_byte_addr: int | None,
) -> tuple[list[int], list[str]]:
    """
    Replay inst_ram_read_ptr / FETCH indexing for ``n_trace`` decode events.

    Models queue_state_module.sv: each trace row is one ``STATE_DECODE_TYPE``. Loops and
    backward relative jumps rewind ``inst_ram_read_ptr``; REG_ALU updates a GPR model so
    RELA_JNZ / RELA_JZ / JNZ / JZ match hardware. Absolute jump immediates are
    ``byte >> 3`` and map through ``program_dram_byte_addr`` (capture_buffer[0]).
    """
    if n_trace <= 0:
        return [], []
    if not insts:
        return [0] * n_trace, ["capture_buffer is empty; cannot map trace rows"]
    images = [(int(program_dram_byte_addr or 0), insts)]
    mapped, warnings = trace_dynamic_decode_map(images, n_trace, program_dram_byte_addr or 0)
    return [idx for _base, idx, _inst in mapped], warnings


_JUMP_MODE_NAMES = {
    JUMP_MODE_ABSOLUTE: "ABSOLUTE",
    JUMP_MODE_REG_ABS: "REG_ABS",
    JUMP_MODE_JNZ: "JNZ",
    JUMP_MODE_JZ: "JZ",
    JUMP_MODE_RELATIVE: "RELATIVE",
    JUMP_MODE_RELA_JNZ: "RELA_JNZ",
    JUMP_MODE_RELA_JZ: "RELA_JZ",
    JUMP_MODE_REG_RELA: "REG_RELA",
}

_REL_LOOP_MODES = {JUMP_MODE_RELATIVE, JUMP_MODE_RELA_JNZ, JUMP_MODE_RELA_JZ}


def _jump_encoded_rel_target(inst_idx: int, imm32: int) -> int:
    """i-cache target of a relative jump: (idx + 1) - imm[9:0], saturating at 0."""
    rel_off = imm32 & 0x3FF
    rp_after = inst_idx + 1
    return (rp_after - rel_off) if rp_after >= rel_off else 0


def _rel_loop_sites(
    mapped: list[tuple[int, int, object]],
) -> dict[tuple[int, int], dict]:
    """Unique backward RELA_* sites: (image, jump_pc) -> {head, jump, base}."""
    sites: dict[tuple[int, int], dict] = {}
    for base, idx, inst in mapped:
        if inst is None:
            continue
        if _inst_desc_bits(inst.words, 8, 11) != INSTRUCTION_JUMP:
            continue
        isa_mode, _src, _dst, _rst, imm32 = _isa_reg_fields(inst.words)
        if isa_mode not in _REL_LOOP_MODES:
            continue
        head = _jump_encoded_rel_target(idx, imm32)
        if head >= idx:
            continue
        sites[(base, idx)] = {"head": head, "jump": idx, "base": base}
    return sites


def _loop_nest_roles(
    sites: dict[tuple[int, int], dict],
) -> dict[tuple[int, int], str]:
    """
    Nest label from containment: loop B is inside A when
    A.head <= B.head and B.jump < A.jump (same image).

    depth 0 -> outer, deepest -> inner, in between -> mid / midN.
    A single loop is labeled loop.
    """
    if not sites:
        return {}
    depths: dict[tuple[int, int], int] = {}
    for key, s in sites.items():
        d = 0
        for k2, o in sites.items():
            if k2 == key or o["base"] != s["base"]:
                continue
            if o["head"] <= s["head"] and s["jump"] < o["jump"]:
                d += 1
        depths[key] = d
    max_d = max(depths.values())
    by_depth: dict[int, list[tuple[int, int]]] = {}
    for key, d in depths.items():
        by_depth.setdefault(d, []).append(key)
    roles: dict[tuple[int, int], str] = {}
    for d, keys in by_depth.items():
        keys.sort(key=lambda k: (sites[k]["base"], sites[k]["jump"]))
        for i, key in enumerate(keys, 1):
            if max_d == 0:
                roles[key] = "loop"
            elif d == 0:
                roles[key] = "outer"
            elif d == max_d:
                roles[key] = "inner"
            elif len(keys) == 1:
                roles[key] = "mid"
            else:
                roles[key] = f"mid{i}"
    return roles


def _ue_program_images(ue: UnifiedEngine) -> tuple[list[tuple[int, list]], int | None]:
    """DRAM program images + execute-start byte. Independent of HW pc_reg."""
    insts = ue.get_captured_instructions()
    images = list(getattr(ue, "_program_images", None) or [])
    if not images and insts:
        prog_addr = getattr(ue, "_last_program_write_addr", None)
        if prog_addr is None and hasattr(ue, "get_program_dram_addr"):
            try:
                prog_addr = ue.get_program_dram_addr() - len(insts) * INSTRUCTION_SIZE_BYTES
            except Exception:
                prog_addr = 0
        images = [(int(prog_addr or 0), insts)]
    start_byte = getattr(ue, "_last_execute_addr", None)
    if start_byte is None and images:
        start_byte = images[0][0]
    return images, start_byte


def replay_trace_pc_map(
    ue: UnifiedEngine,
    trace_values: list[int],
) -> tuple[list[tuple[int, int, object]], list[str]]:
    """Replay FETCH/DECODE for every TRACE row. ``pc`` is inst_ram_read_ptr."""
    images, start_byte = _ue_program_images(ue)
    if not images:
        return [], ["no program images; cannot map TRACE rows"]
    return trace_dynamic_decode_map(images, len(trace_values), start_byte)


def annotate_pc_map_rows(
    ue: UnifiedEngine,
    mapped: list[tuple[int, int, object]],
    trace_values: list[int],
) -> list[dict]:
    """
    One dict per TRACE row. Shared by the CSV map and Perfetto.

    ``pc`` / ``next_pc`` are fetch indices and rewind on taken RELA_JNZ.
    ``trace_row`` is the TRACE slot (always +1).
    """
    import io
    import contextlib

    nest_roles = _loop_nest_roles(_rel_loop_sites(mapped))
    rows: list[dict] = []
    n = len(mapped)
    for i, (base, idx, inst) in enumerate(mapped):
        itype = _inst_desc_bits(inst.words, 8, 11) if inst is not None else -1
        jump_mode = ""
        taken_s = ""
        enc_target = ""
        next_pc = mapped[i + 1][1] if i + 1 < n else ""
        hexdump = None
        detail_lines: list[str] = []
        name = f"ISA_DECODE_{i}"
        if inst is not None:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    ue.parse_instruction(inst, idx, base + idx * INSTRUCTION_SIZE_BYTES)
                except Exception:
                    try:
                        ue.parse_instruction(inst, idx, 0)
                    except Exception:
                        pass
            for line in buf.getvalue().splitlines():
                s = line.rstrip()
                if not s:
                    continue
                if hexdump is None and s.startswith("[") and "=" in s and "0x" in s:
                    hexdump = s
                    continue
                detail_lines.append(s.strip())
            name = _perfetto_event_title(detail_lines, inst, i)
            # generate_instruction_clear_fmax() has no opcode of its own: it
            # rides the fmax_clear strobe on a 1-row ELTWISE_ADD with writeback
            # disabled and throws the sum away. The engine really does run, so
            # keep it on the COMPUTE track and in the mode-derived naming --
            # only say which add it is, or it reads as a vector add that isn't.
            if ("max_clear: enabled" in detail_lines
                    and any("URAM_WB_DISABLE" in d for d in detail_lines)
                    and "UE_COMPUTE" in name):
                name = "UE_COMPUTE (FMAX_CLEAR)"
            if itype == INSTRUCTION_HALT:
                name = "ISA_HALT"
            elif itype == INSTRUCTION_NOP:
                name = "ISA_NOP"
            if itype == INSTRUCTION_JUMP:
                isa_mode, _src, _dst, _rst, imm32 = _isa_reg_fields(inst.words)
                jump_mode = _JUMP_MODE_NAMES.get(isa_mode, str(isa_mode))
                enc_target = _jump_encoded_rel_target(idx, imm32)
                if next_pc != "":
                    taken_s = "fall" if next_pc == idx + 1 else "taken"
                nest = nest_roles.get((base, idx), "")
                name = _isa_jump_perfetto_name(jump_mode, taken_s, nest)
        tick = int(trace_values[i]) if i < len(trace_values) else 0
        rows.append({
            "trace_row": i,
            "pc": idx,
            "next_pc": next_pc,
            "tick": tick,
            "image": base,
            "type": itype,
            "jump_mode": jump_mode,
            "taken": taken_s,
            "enc_target": enc_target,
            "nest": nest_roles.get((base, idx), "") if jump_mode else "",
            "name": name,
            "inst": inst,
            "detail_lines": detail_lines,
            "hexdump": hexdump,
        })
    return rows


def dump_trace_pc_map(
    ue: UnifiedEngine,
    trace_values: list[int],
    out_path: str,
    result: str | None = None,
    rows: list[dict] | None = None,
    mapped: list[tuple[int, int, object]] | None = None,
) -> tuple[list[tuple[int, int, object]], list[str]]:
    """Write one CSV line per TRACE row: fetch PC, tick, decoded name."""
    warnings: list[str] = []
    if rows is None:
        if mapped is None:
            mapped, warnings = replay_trace_pc_map(ue, trace_values)
        rows = annotate_pc_map_rows(ue, mapped, trace_values)
    elif mapped is None:
        mapped = [(r["image"], r["pc"], r["inst"]) for r in rows]

    with open(out_path, "w") as f:
        f.write(f"# RESULT: {result if result is not None else 'UNCHECKED'}\n")
        f.write(
            "trace_row,pc,tick,image,type,jump_mode,taken,enc_target,next_pc,nest,name\n"
        )
        for r in rows:
            f.write(
                f"{r['trace_row']},{r['pc']},{r['tick']},0x{r['image']:x},{r['type']},"
                f"{r['jump_mode']},{r['taken']},{r['enc_target']},{r['next_pc']},"
                f"{r.get('nest', '')},{r['name']}\n"
            )
    print(f"TRACE PC map written to {out_path}")
    return mapped, warnings


def verify_trace_loop_map(
    ue: UnifiedEngine,
    trace_values: list[int],
    mapped: list[tuple[int, int, object]] | None = None,
    warnings: list[str] | None = None,
) -> list[str]:
    """Offline check that replayed FETCH indices match encoded jumps. No Perfetto."""
    errors: list[str] = []
    if mapped is None:
        mapped, warnings = replay_trace_pc_map(ue, trace_values)
    errors.extend(warnings or [])
    if len(mapped) != len(trace_values):
        errors.append(f"map length {len(mapped)} != TRACE length {len(trace_values)}")
        return errors
    if not mapped:
        errors.append("empty TRACE map")
        return errors

    for i in range(1, len(trace_values)):
        if trace_values[i] < trace_values[i - 1]:
            errors.append(
                f"row {i}: TRACE tick went backwards {trace_values[i - 1]} -> {trace_values[i]}"
            )
            break

    last_base, last_idx, last_inst = mapped[-1]
    last_type = _inst_desc_bits(last_inst.words, 8, 11) if last_inst is not None else None
    if last_type != INSTRUCTION_HALT:
        errors.append(
            f"last TRACE row is type={last_type} image@0x{last_base:x}[{last_idx}], expected HALT"
        )

    type_at_site: dict[tuple[int, int], int] = {}
    for i, (base, idx, inst) in enumerate(mapped):
        if inst is None:
            errors.append(f"row {i}: no instruction")
            continue
        itype = _inst_desc_bits(inst.words, 8, 11)
        site = (base, idx)
        if site in type_at_site and type_at_site[site] != itype:
            errors.append(
                f"row {i}: image@0x{base:x}[{idx}] type {itype} != earlier {type_at_site[site]}"
            )
        type_at_site[site] = itype
        if itype != INSTRUCTION_JUMP or i + 1 >= len(mapped):
            continue
        isa_mode, _src, _dst, _rst, imm32 = _isa_reg_fields(inst.words)
        if isa_mode not in _REL_LOOP_MODES:
            continue
        nb, ni, _ = mapped[i + 1]
        fall = (nb == base and ni == idx + 1)
        enc = _jump_encoded_rel_target(idx, imm32)
        if not fall:
            if nb != base:
                errors.append(
                    f"row {i}: taken relative jump switched image 0x{base:x} -> 0x{nb:x}"
                )
            if ni != enc:
                errors.append(
                    f"row {i}: taken RELA next_idx={ni} != encoded target {enc} "
                    f"(idx={idx}, off={imm32 & 0x3FF})"
                )

    print("Offline TRACE loop map")
    print(f"  TRACE rows = {len(mapped)}  unique ticks = {len(set(trace_values))}")
    print(
        f"  last = image@0x{last_base:x}[{last_idx}] type={last_type} "
        f"(HALT={INSTRUCTION_HALT})"
    )
    print(f"  unique static sites decoded = {len(type_at_site)}")
    if errors:
        print(f"  FAIL {len(errors)} check(s)")
        for e in errors[:20]:
            print(f"    {e}")
        if len(errors) > 20:
            print(f"    ... {len(errors) - 20} more")
    else:
        print("  PASS: relative jumps land on encoded targets; last row is HALT")
    return errors


def read_trace(ue: UnifiedEngine, instruction_count: int = None):
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


def _layout_display_ns(
    itypes: list[int],
    ticks: list[int],
    timestamps_ns: list[int],
    cycle_ns: int,
) -> tuple[list[int], list[int | None]]:
    """Place retires on the timeline, honouring what TRACE can and cannot say.

    Non-prefetchable retires sit exactly on their stamped tick and are measured
    by it: the span to the next non-prefetchable retire is real elapsed time,
    because those instructions only retire once ``engine_busy`` clears. Their
    duration is left to ``build_perfetto`` for engine ops (which apply the
    i-cache-fill clamps) and filled in here for everything else.

    Prefetchable retires (REG_ALU_PREFETCH / PBI_SET_PREFETCH) have no duration
    to report at all: the RTL holds their TRACE write and stamps the counter at
    release, so the tick is a drain time, not a decode time. Give each of them
    one aclk cycle, chained so consecutive ones step rather than stack inside
    the /16 hole. Never invent a width for them -- an earlier version divided
    the hole between rows, and those synthetic widths read as measurements.
    """
    n = len(timestamps_ns)
    display = list(timestamps_ns)
    durs: list[int | None] = [None] * n

    next_np = [n] * n
    nxt = n
    for i in range(n - 1, -1, -1):
        next_np[i] = nxt
        if itypes[i] not in _PREFETCH_INST_TYPES:
            nxt = i

    cursor: int | None = None
    for i in range(n):
        if itypes[i] in _PREFETCH_INST_TYPES:
            ts = timestamps_ns[i]
            if cursor is not None and cursor > ts:
                ts = cursor
            display[i] = ts
            durs[i] = cycle_ns
            cursor = ts + cycle_ns
        else:
            display[i] = timestamps_ns[i]
            cursor = None
            if itypes[i] not in _ENGINE_INST_TYPES:
                j = next_np[i]
                # Exactly what TRACE says: the tick delta to the next
                # non-prefetch retire, no floor. A delta of 0 means the two
                # retires shared a tick and the counter cannot separate them;
                # it renders as one cycle only so the slice stays visible, not
                # as a claim that it took a cycle.
                durs[i] = (max(cycle_ns, timestamps_ns[j] - timestamps_ns[i])
                           if j < n else None)
    return display, durs


def build_perfetto(
    trace_values: list[int],
    ue: UnifiedEngine,
    out_path: str,
    pc_rows: list[dict] | None = None,
    queue_fills: list[dict] | None = None,
):
    """Emit Perfetto from an already-built TRACE→PC map. Does not read HW pc_reg."""
    clock_period_ns = ue._clock_period_ns

    parsed_retires, parsed_fills = split_trace_bram_words(trace_values)
    if parsed_fills:
        isa_ticks = parsed_retires
        hw_fills = parsed_fills
    else:
        isa_ticks = trace_values
        hw_fills = list(queue_fills or [])

    if pc_rows is None:
        mapped, replay_warnings = replay_trace_pc_map(ue, isa_ticks)
        for msg in replay_warnings:
            print(f"Trace PC replay: {msg}")
        pc_rows = annotate_pc_map_rows(ue, mapped, isa_ticks)
    if not pc_rows:
        print("No captured instructions found in `ue.capture_buffer`. Run capture before decoding.")
        return False
    if len(pc_rows) != len(isa_ticks):
        print(
            f"PC map length {len(pc_rows)} != retire TRACE {len(isa_ticks)}; "
            "refusing Perfetto (queue-tagged rows must be stripped first)."
        )
        return False

    n = len(isa_ticks)
    # Precompute timestamps in integer nanoseconds first. Perfetto internally
    # quantizes to ns; deriving ts/dur from the same ns grid avoids visual
    # 1ns gaps caused by float rounding.
    timestamps_ns = []
    for i in range(n):
        counter = isa_ticks[i]
        # Trace timestamps come from the prescaled pipeline counter: 1 count = 16 aclk cycles.
        ts_ns = int(round(counter * UE_PIPELINE_COUNTER_CLK_DIV * clock_period_ns))
        timestamps_ns.append(ts_ns)

    # Print confirmation of conversion
    preview = [(isa_ticks[i], timestamps_ns[i] / 1000.0) for i in range(min(3, n))]
    print(f"Using clock period = {clock_period_ns:.6f} ns -> timestamps in microseconds. Example: cycle->us for first entries: {preview}")

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
        2: "3-DMA_TO_DRAM",
        3: "4-COMPUTE",
        4: "5-HALT",
    }

    def _pick_tracks(event_name: str, args: dict = None) -> list[int]:
        upper = event_name.upper()
        if "RELA_JNZ" in upper:
            return [0]
        if "QUEUE_LOADING" in upper:
            return [0]
        if "COMPUTE" in upper:
            tracks = [3]
            is_dma_op = (("DOT_PRODUCT" in upper and "BF16" not in upper)
                         or "DEQUANTIZE" in upper)
            dma_annotation = "enabled" in str(args.get("dma_start", "")).lower() if args else False
            if is_dma_op or dma_annotation:
                tracks.append(1)
            return tracks
        if "MEMCPY_TO" in upper:
            return [2]
        if "MEMCPY_FROM" in upper:
            return [1]
        if "HALT" in upper:
            # Every other category returns a single track above; HALT is the
            # only TRACK_MAP key that also reaches the fallback, so without
            # this it was emitted twice -- once on 1-QUEUE, once on 5-HALT.
            return [TRACK_MAP["HALT"]]
        tracks = [0]
        for key, tid in TRACK_MAP.items():
            if key in upper and tid not in tracks:
                tracks.append(tid)
        return tracks

    collected = []

    def _emit_queue_loading(name: str, ts_ns: int, dur_ns: int, args: dict):
        collected.append({
            "name": name,
            "track": 0,
            "ts_ns": ts_ns,
            "dur_ns": dur_ns,
            "args": args,
        })

    # One TRACE tick is 16 aclk cycles. Same-tick slices and the trailing HALT
    # would otherwise have dur=0 / ~1 cycle and disappear in Perfetto.
    min_dur_ns = max(1, int(round(UE_PIPELINE_COUNTER_CLK_DIV * clock_period_ns)))

    def _tick_to_ns(tick: int) -> int:
        return int(round(int(tick) * UE_PIPELINE_COUNTER_CLK_DIV * clock_period_ns))

    if hw_fills:
        for ev in hw_fills:
            ts_ns = _tick_to_ns(ev["start_tick"])
            te_ns = _tick_to_ns(ev["done_tick"])
            dur_ns = te_ns - ts_ns
            if dur_ns <= 0:
                dur_ns = min_dur_ns
            reason = _queue_loading_reason(ev, pc_rows)
            _emit_queue_loading(
                f"UE_QUEUE_LOADING ({reason})",
                ts_ns,
                dur_ns,
                {
                    "reason": reason,
                    "start_tick": ev["start_tick"],
                    "done_tick": ev["done_tick"],
                    "bram_row": ev.get("bram_row", -1),
                    "retires_before": ev.get("retires_before", -1),
                },
            )
    else:
        # Legacy bitstream: no fill tags. Show start (empty i-cache) and
        # absolute / taken JNZ/JZ holes only — not RELA.
        first_ts_ns = max(0, timestamps_ns[0]) if timestamps_ns else 0
        if first_ts_ns > 0:
            _emit_queue_loading(
                "UE_QUEUE_LOADING (icache empty)",
                0,
                first_ts_ns,
                {"reason": "icache empty"},
            )

    # Prefetch REG/PBI overlap a busy engine. Stretching them to the next
    # TRACE row makes ISA_REG cover the whole COMPUTE. Bookkeeping is one
    # aclk when several share a /16 tick; UE_OP/UE_PBI extend to the next
    # engine op or HALT (that gap is engine_busy).
    def _row_itype(idx: int) -> int:
        if idx < 0 or idx >= len(pc_rows):
            return -1
        inst = pc_rows[idx].get("inst")
        if inst is not None:
            return _inst_desc_bits(inst.words, 8, 11)
        return int(pc_rows[idx].get("type", -1))

    cycle_ns = max(1, int(round(clock_period_ns)))
    display_ts_ns, prefetch_dur_ns = _layout_display_ns(
        [_row_itype(i) for i in range(n)],
        isa_ticks,
        timestamps_ns,
        cycle_ns,
    )

    # End an engine op's slice at the next NON-PREFETCHABLE retire, not the next
    # engine op. That row could only retire once engine_busy dropped, so it is
    # the true end of the busy window -- and stopping there means the slice no
    # longer spans a stalled JUMP that sits between two engine ops (pc 7's
    # memcpy_to used to swallow the RELA_JNZ whenever the next iteration's
    # memcpy_from issued a tick later than the jump retired).
    next_engine = [n] * n
    nxt = n
    for i in range(n - 1, -1, -1):
        next_engine[i] = nxt
        if _row_itype(i) not in _PREFETCH_INST_TYPES:
            nxt = i

    next_compute = [n] * n
    nxtc = n
    for i in range(n - 1, -1, -1):
        next_compute[i] = nxtc
        nm = (pc_rows[i].get("name") or "").upper() if i < len(pc_rows) else ""
        if "COMPUTE" in nm:
            nxtc = i

    for i in range(n):
        counter = isa_ticks[i]
        ts_ns = timestamps_ns[i]

        is_last = (i == n - 1)
        row = pc_rows[i] if i < len(pc_rows) else {}
        inst_base = row.get("image", 0)
        inst_idx = row.get("pc", -1)
        inst_obj = row.get("inst")
        itype = _row_itype(i)
        name = row.get("name") or f"ISA_DECODE_{i}"
        skip_compute = False
        if itype in _ENGINE_INST_TYPES:
            ts_ns = display_ts_ns[i]
            j = next_engine[i]
            if j < n:
                dur_ns = max(cycle_ns, timestamps_ns[j] - timestamps_ns[i])
                # ts comes from the spread timebase but dur from the raw one, so
                # a row whose display ts drifted inside a same-tick run would
                # overrun its successor by exactly that drift. COMPUTE has its
                # own clamp below; MEMCPY had none, which rendered two
                # non-prefetch UE_PBI memcpys overlapping on 2-DMA_FROM_DRAM.
                nxt_disp_ns = display_ts_ns[j]
                if nxt_disp_ns > ts_ns:
                    dur_ns = max(cycle_ns, min(dur_ns, nxt_disp_ns - ts_ns))
            else:
                dur_ns = min_dur_ns
            # Same-track COMPUTE: do not force 48ns min_dur over the next DOT.
            if "COMPUTE" in name.upper():
                k = next_compute[i]
                if k < n:
                    nxt_ts = display_ts_ns[k]
                    if nxt_ts <= ts_ns:
                        skip_compute = True
                    elif nxt_ts < ts_ns + dur_ns:
                        dur_ns = nxt_ts - ts_ns
            # Empty-line FETCH waits for !engine_busy, then RAM_DMA. Stretching
            # UE_OP to the next engine op would paint COMPUTE over QUEUE_LOADING.
            if hw_fills:
                for ev in hw_fills:
                    fs = _tick_to_ns(ev["start_tick"])
                    if fs > ts_ns:
                        dur_ns = min(dur_ns, fs - ts_ns)
                        break
        elif prefetch_dur_ns[i] is not None:
            ts_ns = display_ts_ns[i]
            dur_ns = prefetch_dur_ns[i]
        else:
            dur_ns = min_dur_ns
        hexdump = row.get("hexdump")
        detail_lines = list(row.get("detail_lines") or [])
        if is_last and inst_obj is not None:
            if _inst_desc_bits(inst_obj.words, 8, 11) == INSTRUCTION_HALT:
                name = "ISA_HALT"

        args = {
            "index": i,
            "fetch_inst_idx": inst_idx,
            "fetch_dram_addr": f"0x{inst_base + inst_idx * INSTRUCTION_SIZE_BYTES:x}" if inst_obj is not None else "",
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

        if inst_obj is not None:
            itype = _inst_desc_bits(inst_obj.words, 8, 11)
            if itype == INSTRUCTION_UE_PBI:
                args["is_pbi"] = 1
                # Bits [15:12] — same as Python ue_op_descriptor(inst_pointer_idx=...) /
                # alloc_inst_ptr() (1..15). Not the list subscript k in pointer_idx[k].
                ip = _inst_desc_bits(inst_obj.words, 12, 15)
                args["inst_pointer_idx"] = ip
                # When pointers are allocated sequentially from reset (matmul pattern:
                # pointer_idx = [alloc_inst_ptr() for _ in range(n)]), pointer_idx[k] == k+1,
                # so k == inst_pointer_idx - 1.
                if ip:
                    args["pointer_idx_slot"] = ip - 1

        if "MEMCPY" in name.upper() and "memcpy_type" in args:
            mt = str(args["memcpy_type"])
            import re as _re
            paren = _re.search(r"\(([^)]+)\)", mt)
            suffix = paren.group(1) if paren else mt
            name = f"{name} ({suffix})"

        if skip_compute:
            continue

        for track_id in _pick_tracks(name):
            collected.append({
                "name": name,
                "track": track_id,
                "ts_ns": ts_ns,
                "dur_ns": dur_ns,
                "args": args,
            })

        # Legacy bitstream: infer i-cache DMA from jump type + timestamp hole.
        # Tagged TRACE (bit 31) already emitted QUEUE_LOADING from hw_fills.
        if not hw_fills and _jump_triggers_icache_refill(row) and i + 1 < n:
            idle_ts = ts_ns + dur_ns
            idle_dur = timestamps_ns[i + 1] - idle_ts
            if idle_dur > 0:
                reason = _queue_loading_reason({"retires_before": i + 1}, pc_rows)
                _emit_queue_loading(
                    f"UE_QUEUE_LOADING ({reason})",
                    idle_ts,
                    idle_dur,
                    {
                        "index": i,
                        "fetch_inst_idx": inst_idx,
                        "reason": reason,
                        "jump_mode": str(row.get("jump_mode", "") or ""),
                        "taken": str(row.get("taken", "") or ""),
                        "counter": counter,
                    },
                )

    load_evs = [
        e for e in collected
        if e["name"].startswith("UE_QUEUE_LOADING") and e["track"] == 0
    ]
    print(f"Perfetto queue loads: {len(load_evs)}")
    for k, e in enumerate(load_evs, 1):
        print(
            f"  QUEUE_LOADING[{k}] {e['name']}  ts={e['ts_ns']/1000:.3f} us  "
            f"dur={e['dur_ns']/1000:.3f} us"
        )

    # --- Generate native Perfetto protobuf trace (.pftrace) ---
    TRACK_UUIDS = {tid: 1000 + tid for tid in TRACK_LABELS}
    trace_bytes = b''

    for tid, label in TRACK_LABELS.items():
        td = _pf_track_descriptor(TRACK_UUIDS[tid], label)
        pkt = _pf_packet(track_descriptor=td, seq_flags=1)
        trace_bytes += _pb_field_bytes(1, pkt)

    for ev in collected:
        uuid = TRACK_UUIDS[ev['track']]
        # Perfetto renders ts/dur in ns only. Carry the same spans in aclk
        # cycles and TRACE ticks as annotations so the hardware timebase is
        # readable in the details pane (and queryable) without converting by
        # hand. dur_ticks is fractional on purpose: a slice shorter than the
        # /16 prescaler is real, it just cannot be resolved by the counter.
        args = dict(ev['args'])
        dur_cycles = ev['dur_ns'] / clock_period_ns
        args['ts_cycles'] = int(round(ev['ts_ns'] / clock_period_ns))
        args['dur_cycles'] = int(round(dur_cycles))
        args['dur_ticks'] = f"{dur_cycles / UE_PIPELINE_COUNTER_CLK_DIV:.2f}"
        args['clock_period_ns'] = f"{clock_period_ns:.4f}"
        annotations = [_pf_debug_annotation(k, v) for k, v in args.items()]

        te = _pf_track_event_begin(uuid, ev['name'], annotations)
        pkt = _pf_packet(timestamp_ns=ev['ts_ns'], track_event=te)
        trace_bytes += _pb_field_bytes(1, pkt)

        te = _pf_track_event_end(uuid)
        pkt = _pf_packet(timestamp_ns=ev['ts_ns'] + ev['dur_ns'], track_event=te)
        trace_bytes += _pb_field_bytes(1, pkt)

    track_counts = {}
    for ev in collected:
        track_counts[ev["track"]] = track_counts.get(ev["track"], 0) + 1
    print(
        "Perfetto track occupancy: "
        + ", ".join(f"{TRACK_LABELS[tid]}={track_counts.get(tid, 0)}" for tid in TRACK_LABELS)
    )

    try:
        with open(out_path, 'wb') as f:
            f.write(trace_bytes)
        print(f"Perfetto trace written to {out_path}")
        return True
    except OSError as e:
        print(f"Failed to write Perfetto trace: {e}")
        return False


def generate_trace(ue: UnifiedEngine, file_path: str):
    """Dump TRACE, map each row to a fetch PC offline, then build Perfetto from that map."""

    n_static = len(ue.get_captured_instructions())
    for _addr, img in getattr(ue, "_program_images", None) or []:
        n_static = max(n_static, len(img))
    if n_static > UE_TRACE_SIZE:
        print("Capture instruction size is too large to generate trace, skipping...")
        return

    traced_instruction_count = ue.read_reg32(UE_TRACE_BRAM_ADDR)
    n_retire = ue.read_reg32(UE_INSTRUCTION_CTL_ADDR)
    print(f"traced instruction count = {traced_instruction_count}  PC retires = {n_retire}")

    if n_retire > UE_TRACE_SIZE:
        print(
            f"Retired instruction count {n_retire} exceeds TRACE buffer "
            f"({UE_TRACE_SIZE}), skipping..."
        )
        return

    wp = int(traced_instruction_count)
    trace_data = read_trace(ue, instruction_count=min(wp, UE_TRACE_SIZE))
    print(f"read {len(trace_data)} entries")
    retire_ticks, queue_fills = split_trace_bram_words(trace_data)
    print(
        f"TRACE retires = {len(retire_ticks)}  queue fill stamps = {len(queue_fills)}"
    )

    try:
        with open(file_path, "w") as f:
            for idx, val in enumerate(trace_data):
                f.write(f"{idx},{val}\n")
        print(f"saved to {file_path}")
    except OSError as e:
        print(f"error writing output file: {e}", file=sys.stderr)
        sys.exit(1)

    # 1) Offline TRACE → fetch-PC map (does not use HW pc_reg).
    mapped, warnings = replay_trace_pc_map(ue, retire_ticks)
    errors = verify_trace_loop_map(
        ue, retire_ticks, mapped=mapped, warnings=warnings
    )
    result = "PASS" if not errors else f"FAIL ({len(errors)} checks)"
    pc_rows = annotate_pc_map_rows(ue, mapped, retire_ticks)
    map_out = os.path.splitext(file_path)[0] + "_pc_map.csv"
    dump_trace_pc_map(
        ue, retire_ticks, map_out, result=result, rows=pc_rows, mapped=mapped
    )

    # 2) Perfetto is a view of that same map, not a second replay.
    perf_out = os.path.splitext(file_path)[0] + "_perfetto.json"
    print(f"Building Perfetto from offline PC map ({result})")
    try:
        success = build_perfetto(
            retire_ticks, ue, perf_out, pc_rows=pc_rows, queue_fills=queue_fills
        )
        if not success:
            print("Perfetto generation failed")
    except Exception as e:
        print(f"Error generating perfetto JSON: {e}")
