#!/usr/bin/env python3
"""
Standalone numerical probe for the Qwen2.5-VL vision-encoder RoPE PBI bug.

WHY: PBI rope produces coherent-but-wrong captions. The arithmetic (eltwise
mul/add on fixed SRAM addrs) is byte-identical to the proven-good legacy path,
so the only suspect is DMA *addressing* — whether the 4 PBI pointers
(q-load, cos-load, sin-load, out-store) actually auto-advance by one row
(row_bytes) per loop iteration, or whether some get stuck at row 0.

HOW (one device run, fully self-decoding): we run ONLY the PBI rope loop on
head 0 of VIS_Q_PAD_DRAM, feeding crafted markers:

    x[t,:]   = 2 ** ((t % 8) - 4)            # Q-load marker,  period 8  (fast)
    cos[t,:] = 2 ** (((t // 8) % 8) - 4)     # cos-load marker, period 64 (slow)
    sin      = 0                             # cross terms vanish

With sin=0 the rope reduces to out = x * cos, so:

    out[t]   = 2 ** ( (t%8 - 4) + (t//8 % 8 - 4) )

Decode the exponent of out[t] (powers of two are bf16-exact):

    exp(t) = e_q(t) + e_cos(t),  e_q = (t%8 - 4),  e_cos = (t//8 % 8 - 4)

    * both pointers advance   -> exp shows BOTH the fast(8) and slow(64) periods
    * cos-load STUCK at row 0 -> e_cos frozen at -4; only the fast period remains
    * q-load   STUCK at row 0 -> e_q  frozen at -4; only the slow period remains
    * out-store STUCK at row 0-> rows 1..VS-1 stay 0 (buffer pre-zeroed)

This isolates EXACTLY which pointer misbehaves — the core hypothesis is that
>2 simultaneous advancing load pointers don't all advance (the proven-good
rms_norm_core_dram_pbi uses only 2). Run #2 (if needed) swaps markers onto
sin to cover the sin-load + store pointers.

USAGE:  python rope_pbi_probe.py
"""
import os
import sys
import importlib.util
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from user_dma_core import (
    DMA_DEVICE_H2C, ue_35bit_addr_shifter, INSTRUCTION_SIZE_BYTES,
)


def _load_engine_class():
    """Import Qwen25VL3B_UnifiedEngine from its dotted-name module file."""
    path = os.path.join(SCRIPT_DIR, "qwen2.5_vl_3b_test.py")
    spec = importlib.util.spec_from_file_location("qwen25vl_probe_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Qwen25VL3B_UnifiedEngine


def build_markers(VS, VD_PAD):
    """Crafted cos/sin/x tables (see module docstring)."""
    bpe_dtype = torch.bfloat16
    t = torch.arange(VS)
    e_q = (t % 8) - 4                 # fast period 8
    e_cos = ((t // 8) % 8) - 4        # slow period 64
    x = (2.0 ** e_q.float()).unsqueeze(1).repeat(1, VD_PAD).to(bpe_dtype)
    cos = (2.0 ** e_cos.float()).unsqueeze(1).repeat(1, VD_PAD).to(bpe_dtype)
    sin = torch.zeros(VS, VD_PAD, dtype=bpe_dtype)
    return x, cos, sin, e_q, e_cos


def emit_pbi_rope_head0(ue, VS, VD_PAD, bpe):
    """Capture ONLY the PBI rope loop on head 0 of VIS_Q_PAD_DRAM, then halt.

    This is a verbatim transcription of the suspect loop from the
    pbi-optims-wip backup (@a4e53f7), scoped to a single head so the readback
    is unambiguous.
    """
    half = VD_PAD // 2
    row_bytes = VD_PAD * bpe
    sram_q = 0x00000
    sram_a = sram_q + row_bytes
    sram_cos = 0x80000
    sram_sin = sram_cos + row_bytes
    sram_bc = sram_sin + row_bytes
    _, q_uram = ue.sram_address_to_uram_address(sram_q)
    _, a_uram = ue.sram_address_to_uram_address(sram_a)
    _, cos_uram = ue.sram_address_to_uram_address(sram_cos)
    _, sin_uram = ue.sram_address_to_uram_address(sram_sin)

    head_addr = ue.VIS_Q_PAD_DRAM  # head 0

    ue.clear_inst_id()
    ue.start_capture()

    # Drive the loop trip count from a runtime GPR, exactly like the real code.
    vis_M_reg = ue.alloc_isa_reg()
    ue.generate_instruction_add_set(vis_M_reg, VS)

    p_q = ue.alloc_inst_ptr()
    p_cos = ue.alloc_inst_ptr()
    p_sin = ue.alloc_inst_ptr()
    p_out = ue.alloc_inst_ptr()
    ue.generate_instruction_pbi_init(
        dram_shared_addr=head_addr, dma_length=row_bytes,
        uram_dst_addr=q_uram, inst_pointer_idx=p_q)
    ue.generate_instruction_pbi_init(
        dram_shared_addr=ue.VIS_ROPE_COS_DRAM, dma_length=row_bytes,
        uram_dst_addr=cos_uram, inst_pointer_idx=p_cos)
    ue.generate_instruction_pbi_init(
        dram_shared_addr=ue.VIS_ROPE_SIN_DRAM, dma_length=row_bytes,
        uram_dst_addr=sin_uram, inst_pointer_idx=p_sin)
    ue.generate_instruction_pbi_init(
        dram_shared_addr=head_addr, dma_length=row_bytes,
        uram_a_start_addr=a_uram, uram_b_start_addr=a_uram,
        inst_pointer_idx=p_out)

    prog = ue.get_program_dram_addr()
    cnt = ue.capture_count
    ue.generate_instruction_jump_abs(
        ue_35bit_addr_shifter(prog + (cnt + 1) * INSTRUCTION_SIZE_BYTES))
    ue.loop_start(loop_cnt=VS, gpr_loop_cnt=vis_M_reg)
    ue.accelerator_memory_to_sram(
        accelerator_dram_address=row_bytes, sram_address=sram_q,
        element_size=0, inst_pointer_idx=p_q)
    ue.accelerator_memory_to_sram(
        accelerator_dram_address=row_bytes, sram_address=sram_cos,
        element_size=0, inst_pointer_idx=p_cos)
    ue.accelerator_memory_to_sram(
        accelerator_dram_address=row_bytes, sram_address=sram_sin,
        element_size=0, inst_pointer_idx=p_sin)
    ue.eltwise_mul_core(sram_q, sram_cos, sram_a, VD_PAD)
    ue.eltwise_mul_core(sram_q + half * bpe, sram_sin, sram_bc, half)
    ue.eltwise_mul_core(sram_q, sram_sin + half * bpe, sram_bc + half * bpe, half)
    ue.eltwise_add_core(sram_a, sram_bc, sram_a, VD_PAD)
    ue.sram_to_accelerator_memory(
        sram_address=sram_a, accelerator_dram_address=row_bytes,
        element_size=0, inst_pointer_idx=p_out)
    loop_size = ue.loop_end()
    print(f"  rope PBI loop body size: {loop_size} (<=256 required)")
    assert loop_size <= 256

    ue.release_inst_ptr(p_out)
    ue.release_inst_ptr(p_sin)
    ue.release_inst_ptr(p_cos)
    ue.release_inst_ptr(p_q)

    ue.stop_capture()
    ue.generate_instruction_halt()

    prog_bytes = bytearray()
    for inst in ue.capture_buffer:
        prog_bytes.extend(inst.get_bytes())
    program_addr = ue.get_program_dram_addr()
    ue.dma_write(DMA_DEVICE_H2C, program_addr, prog_bytes, len(prog_bytes))
    ue.allocate_program_dram(len(prog_bytes))
    ue.clear_capture_buffer()
    return program_addr


def decode(out, e_q, e_cos, VS, VD_PAD):
    """Classify pointer behavior from the readback."""
    col0 = out[:, 0].float()
    # exponent of each (bf16-exact powers of two; 0 -> -inf marker)
    exps = []
    for v in col0.tolist():
        if v == 0.0:
            exps.append(None)
        else:
            exps.append(round(torch.log2(torch.tensor(abs(v))).item()))
    exp_both = (e_q + e_cos).tolist()
    exp_qonly = (e_q + torch.full_like(e_cos, -4)).tolist()   # cos stuck at row0 (e_cos=-4)
    exp_cosonly = (torch.full_like(e_q, -4) + e_cos).tolist()  # q stuck at row0 (e_q=-4)

    n = VS
    zero_rows = sum(1 for i in range(1, n) if exps[i] is None)
    match_both = sum(1 for i in range(n) if exps[i] == exp_both[i])
    match_q = sum(1 for i in range(n) if exps[i] == exp_qonly[i])
    match_cos = sum(1 for i in range(n) if exps[i] == exp_cosonly[i])

    print("\n=== RoPE PBI pointer-advance verdict ===")
    print(f"  tokens (VS)            : {n}")
    print(f"  rows 1+ that are ZERO  : {zero_rows}  (high => out-store pointer stuck at row 0)")
    print(f"  match 'both advance'   : {match_both}/{n}")
    print(f"  match 'cos stuck row0' : {match_q}/{n}   (q advances, cos does NOT)")
    print(f"  match 'q stuck row0'   : {match_cos}/{n}   (cos advances, q does NOT)")
    print("\n  first 16 decoded exponents:")
    print(f"    measured : {exps[:16]}")
    print(f"    expected : {exp_both[:16]}")
    if match_both >= n - 2:
        print("\n  VERDICT: q-load AND cos-load BOTH advance correctly.")
        print("           Bug is NOT multi-load-pointer advance. Next: probe sin-load + out-store (run #2).")
    elif zero_rows >= n - 2:
        print("\n  VERDICT: out-STORE pointer stuck at row 0 (only row 0 written).")
    elif match_q >= n - 2:
        print("\n  VERDICT: cos-LOAD pointer stuck at row 0 (q advances, cos doesn't).")
        print("           => 3rd+ simultaneous PBI load pointer fails to advance. Fix: bulk/planar rope.")
    elif match_cos >= n - 2:
        print("\n  VERDICT: q-LOAD pointer stuck at row 0 (cos advances, q doesn't).")
    else:
        print("\n  VERDICT: UNCLASSIFIED — neither pattern matches; dumping more rows.")
        print(f"    measured[:32]: {exps[:32]}")


def main():
    EngineClass = _load_engine_class()
    print("Constructing engine (loads weights, allocates DRAM)...")
    ue = EngineClass(script_dir=SCRIPT_DIR)

    # Stop any stale FPGA execution.
    ue.dram_inst_running(False)

    vis_cfg = ue._vis_cfg
    VS = vis_cfg["num_patches"]
    VD_PAD = 128
    bpe = 2
    print(f"  VS={VS}, VD_PAD={VD_PAD}")

    # Zero the head-0 Q buffer so an unwritten row reads back as 0.
    ue.dma_to_accelerator_memory(
        ue.VIS_Q_PAD_DRAM, torch.zeros(VS * VD_PAD, dtype=torch.bfloat16))

    x, cos, sin, e_q, e_cos = build_markers(VS, VD_PAD)
    ue.dma_to_accelerator_memory(ue.VIS_Q_PAD_DRAM, x.flatten())
    ue.dma_to_accelerator_memory(ue.VIS_ROPE_COS_DRAM, cos.flatten())
    ue.dma_to_accelerator_memory(ue.VIS_ROPE_SIN_DRAM, sin.flatten())
    print("  markers DMA'd (x: period-8, cos: period-64, sin: 0)")

    program_addr = emit_pbi_rope_head0(ue, VS, VD_PAD, bpe)
    print(f"  program at 0x{program_addr:X}; executing...")
    ue.start_execute_from_dram(program_addr)
    ue.wait_queue(120.0)
    print("  done.")

    out = ue.dma_from_accelerator_memory(
        ue.VIS_Q_PAD_DRAM, (VS, VD_PAD)).cpu()
    decode(out, e_q, e_cos, VS, VD_PAD)


if __name__ == "__main__":
    main()
