"""Where the card time of one GPT-OSS layer goes: every step of the A / R / E segments compiled as its own
small program and timed with the engine's latency counter (single engine, one layer, cards otherwise idle).
    SHARD=4 POS=200 python prof_gptoss.py [xdma0]"""
import json
import os
import statistics
import sys

import torch

torch.set_num_threads(1)          # like the runner: the multi-threaded path dies with SIGILL on this Xeon
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import gptoss_card as G  # noqa: E402

WORK = os.path.expanduser("~/apex_tp/models/gptoss20b")
TP = os.path.join(WORK, "tp10")
plan = json.load(open(os.path.join(TP, "plan.json")))
SHARD = int(os.environ.get("SHARD", "4"))
POS = int(os.environ.get("POS", "200"))
REP = 5

T, qcfg = G.load_base_port("gptoss_prof_port")
s = G.GptOssShard(T, qcfg, sys.argv[1] if len(sys.argv) > 1 else "xdma0", 0, os.path.join(TP, "card_%02d" % SHARD),
                  os.path.join(WORK, "weights"), plan, max_ctx=int(os.environ.get("CTX", "2048")), n_layers=1)
with T._quiet():
    s.prepare()
    vocab = plan["cards"][SHARD]["vocab"]
    s.compile(parts_n=5, vocab_n=vocab[1] - vocab[0])
ue, U, H, HD = s.ue, s.U, s.H, s.HD
d = s.layers[0]
e = d["experts"][0]
nq, sh = d["nq"], T.ue_35bit_addr_shifter
BF16 = 2

# a real position: registers, masks and KV rows as they would be at POS
torch.manual_seed(0)
for p in range(POS + 1):
    x = (torch.randn(H) * 0.5).to(torch.bfloat16).view(torch.int16).numpy().tobytes()
    s.set_position(p, x)
    s.run(s.seg_start[0])
full = {k: s.run(s.seg_start[i]) for k, i in (("A (whole segment)", 0), ("R (whole segment)", 1))}
s.set_experts(0, [0, 1, 2, 3], [1.0, 0.5, 0.2, 0.1])
full["E (whole segment)"] = s.run(s.seg_start[2])


def timed(name, emit):
    def body(base):
        rc = ue._isa_reg_counter
        ue._isa_reg_counter = max(rc, 6)
        emit()
        ue._isa_reg_counter = rc
        ue.generate_instruction_halt()
    with T._quiet():
        addr = s._place(body)
    v = sorted(s.run(addr) for _ in range(REP))
    rows.append((name, statistics.median(v)))


def rope_q():
    for h in range(nq):
        T._emit_rope_split_64(ue, s.Q + h * HD * BF16, s.QR + h * HD * BF16, ue._fa_cos_pad_dram,
                              ue._fa_neg_sin_pad_dram, ue._fa_sin_hi_pad_dram, HD, 0, load_cos_sin=(h == 0))


def kv_write():
    for src, cache in ((s.KR, d["K"]), (s.Vt, d["V"])):
        ue.accelerator_memory_to_sram(src, 0x40000, HD)
        ue.generate_instruction_add_imm(ue._ISA_POS_K_REG, sh(cache), ue._ISA_TMP_REG)
        ue.sram_to_accelerator_memory(0x40000, cache, HD, general_reg_src=ue._ISA_TMP_REG)


def vt():
    ue.generate_instruction_add_set(G.R_VT_STRIDE, s.CH * BF16)
    ue.generate_instruction_add_imm(G.R_BLK_V, sh(d["V"]), G.R_VT_IN)
    ue.generate_instruction_add_imm(G.R_BLK_VT, sh(d["VT"]), G.R_VT_OUT)
    ue.bf16_transpose_core(M=G.VEC, N=HD, INPUT_DRAM_ADDR=d["V"], OUTPUT_DRAM_ADDR=d["VT"],
                           IDENTITY_DRAM_ADDR=ue._identity_dram_addr, gpr_input_addr=G.R_VT_IN,
                           gpr_out_addr=G.R_VT_OUT, gpr_out_row_stride_reg=G.R_VT_STRIDE)


def qk():
    for h in range(nq):
        ue.matmat_mul_core(M=1, K=HD, N=s.T_ext, A_DRAM_ADDR=s.SQ + h * HD * BF16, B_DRAM_ADDR=d["K"],
                           OUTPUT_DRAM_ADDR=d["SC_ROW"] + h * s.T_ext * BF16, softmax_enable=True,
                           C_DRAM_ADDR=d["BIAS"] + h * s.T_ext * BF16, bias_mode="broadcast_N", gpr_N_reg=G.R_LEN)


def chunks():
    CH = s.CH
    for c in range(s.T_ext // CH):
        ue.accelerator_memory_to_sram(d["SC_ROW"] + c * CH * BF16, 0x00000, nq * CH,
                                      stride_bytes_per_chunk=CH * BF16, stride_jump_bytes=s.T_ext * BF16)
        ue.sram_to_accelerator_memory(0x00000, d["SC_CH"] + c * nq * CH * BF16, nq * CH)


def pv():
    CH = s.CH
    for c in range(s.T_ext // CH):
        ue.matmat_mul_core(M=nq, K=CH, N=HD, A_DRAM_ADDR=d["SC_CH"] + c * nq * CH * BF16,
                           B_DRAM_ADDR=d["VT"] + c * HD * CH * BF16, OUTPUT_DRAM_ADDR=s.ATT,
                           C_DRAM_ADDR=s.ATT if c else None, bias_mode="full_matrix")


def mul_block():
    n = e["n"]
    ue.ue_memcpy_from_dram(s.HID, n * BF16, U.MEMCPY_TYPE.URAM.value, U.URAM_START_ADDR, U.URAM_SECTION.URAM_A.value)
    ue.ue_memcpy_from_dram(s.Uu, n * BF16, U.MEMCPY_TYPE.URAM.value, U.URAM_START_ADDR, U.URAM_SECTION.URAM_B.value)
    ue.eltwise_mul_core(0x00000, 0x80000, 0x00000, n)
    ue.ue_memcpy_to_dram(U.MEMCPY_TYPE.URAM.value, U.URAM_SECTION.URAM_A.value, U.URAM_START_ADDR, s.HID, n * BF16)


def combine(i):
    nb = H * BF16
    s._scalar_to_lalu(s.W4, s.onehot[i])
    ue.ue_memcpy_from_dram(s.Y, nb, U.MEMCPY_TYPE.URAM.value, U.URAM_START_ADDR, U.URAM_SECTION.URAM_A.value)
    ue.start_queue_broadcast(U.UE_MODE.MUL_BROADCAST, U.BROADCAST_MODE.LALU_RESULT, 0x00000, 0x00000, H)
    if i:
        ue.ue_memcpy_from_dram(s.ACC, nb, U.MEMCPY_TYPE.URAM.value, U.URAM_START_ADDR, U.URAM_SECTION.URAM_B.value)
        ue.eltwise_add_core(0x00000, 0x80000, 0x00000, H)
    ue.ue_memcpy_to_dram(U.MEMCPY_TYPE.URAM.value, U.URAM_SECTION.URAM_A.value, U.URAM_START_ADDR, s.ACC, nb)


n = e["n"]
rows = []
timed("A reduce (5 shares + residual)", s._reduce)
timed("A rms", lambda: s._rms(s.X, d["ln_in"], s.XN))
timed("A q gemv %dx%d" % (nq * HD, H), lambda: s._gemv(s.XN, d["q"], nq * HD, H, s.Q, bias=d["q_b"]))
timed("A k gemv", lambda: s._gemv(s.XN, d["k"], HD, H, s.Kt, bias=d["k_b"]))
timed("A v gemv", lambda: s._gemv(s.XN, d["v"], HD, H, s.Vt, bias=d["v_b"]))
timed("A rope q (%d heads)" % nq, rope_q)
timed("A rope k", lambda: T._emit_rope_split_64(ue, s.Kt, s.KR, ue._fa_cos_pad_dram, ue._fa_neg_sin_pad_dram,
                                                ue._fa_sin_hi_pad_dram, HD, 0, load_cos_sin=False))
timed("A kv write", kv_write)
timed("A V^T block", vt)
timed("A scale q", lambda: ue.eltwise_core_dram(M=nq, N=HD, dram_a=s.QR, dram_b=None, dram_out=s.SQ,
                                                 mode=U.UE_MODE.MUL_BROADCAST, scalar=0.125))
timed("A qk+softmax (%d heads)" % nq, qk)
timed("A score chunk copy", chunks)
timed("A p.v", pv)
timed("A o gemv %dx%d" % (H, nq * HD), lambda: s._gemv(s.ATT, d["o"], H, nq * HD, s.SHARE_A, bias=d["o_b"]))
timed("R reduce", s._reduce)
timed("R rms", lambda: s._rms(s.X, d["ln_post"], s.XN2))
timed("R router matmul", lambda: ue.matmat_mul_core(M=1, K=H, N=G.VEC, A_DRAM_ADDR=s.XN2, B_DRAM_ADDR=d["router_w"],
                                                     OUTPUT_DRAM_ADDR=s.LOGITS, C_DRAM_ADDR=d["router_b"],
                                                     bias_mode="broadcast_N"))
timed("E softmax of 4 logits", lambda: ue.matmat_mul_core(M=1, K=G.VEC, N=G.VEC, A_DRAM_ADDR=s.lg4_addr[0],
                                                           B_DRAM_ADDR=ue._identity_dram_addr, OUTPUT_DRAM_ADDR=s.W4,
                                                           softmax_enable=True))
timed("E1 gate gemv %dx%d" % (n, H), lambda: s._gemv(s.XN2, s._sub_pair(e["gu"], 0, H), n, H, s.G, bias=e["g_b"],
                                                     clamp=(float("-inf"), s.limit)))
timed("E1 gelu (identity matmul)", lambda: ue.matmat_mul_core(M=1, K=n, N=n, A_DRAM_ADDR=s.G, B_DRAM_ADDR=s._ident(n),
                                                               OUTPUT_DRAM_ADDR=s.HID, gelu_enable=True))
timed("E1 up gemv", lambda: s._gemv(s.XN2, s._sub_pair(e["gu"], n, H), n, H, s.Uu, bias=e["u_b1"],
                                    clamp=(-s.limit + 1.0, s.limit + 1.0)))
timed("E1 gate*up", mul_block)
timed("E1 down gemv %dx%d" % (H, n), lambda: s._gemv(s.HID, e["down"], H, n, s.Y, bias=e["down_b"]))
timed("E1 combine (first)", lambda: combine(0))
timed("E1 combine (later)", lambda: combine(1))

print("shard %d, %d q heads, %d neurons per expert, position %d" % (SHARD, nq, n, POS))
for k, v in full.items():
    print("  %-34s %7.1f us" % (k, v))
for k, v in rows:
    print("  %-34s %7.1f us" % (k, v))
a = sum(v for k, v in rows if k.startswith("A "))
r = sum(v for k, v in rows if k.startswith("R "))
e1 = sum(v for k, v in rows if k.startswith("E1 ") and "later" not in k)
print("  sum of A steps %.1f | R steps %.1f | one expert incl. combine %.1f" % (a, r, e1))
