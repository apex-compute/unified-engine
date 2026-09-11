"""Host-only (no FPGA) check that a kokoro residual block emits byte-identical instructions for
different frame counts -- INCLUDING a count that is an exact multiple of 64 (zero pad rows), which
is the case the two-prompt compare_programs run can miss.

    python models/kokoro/host_invariance_check.py

UnifiedEngine captures fine without hardware; only DMA is stubbed. Every differing instruction is
traced to the fpga_forward.py line that emitted it.
"""
import sys, os, traceback, functools
say = functools.partial(print, file=sys.stderr, flush=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
import models.kokoro.fpga_forward as F
from user_dma_core import UnifiedEngine


def emit(n_frames, upsample, Cin=512, Cout=512):
    ue = UnifiedEngine(params_dram_base=F.KOKORO_PARAMS_BASE, tensor_dram_base=F.KOKORO_TENSOR_BASE,
                       program_dram_base=F.KOKORO_PROGRAM_BASE)
    ue.dma_to_accelerator_memory = lambda *a, **k: None
    F._UP_CACHE.clear(); F._ZEROS_BUF.clear(); F._NREG_MAP.clear(); F._PREAMBLE_ADDR[0] = None
    F._MSCRATCH_RR[0] = 0
    F._instrument(ue)
    trace = {}
    _od = ue.ue_isa_descriptor
    def _td(*a, **k):
        trace[ue.capture_count] = traceback.format_stack(limit=8)[:-1]
        return _od(*a, **k)
    ue.ue_isa_descriptor = _td
    ue._kokoro_raw_start_capture = ue.start_capture
    nf_pad = F._round_up(n_frames, 64)
    F._set_row_cap(2 * nf_pad, 2 * F.F_CAP, "x")
    T = F.Dim(nf_pad, F.GPR_NFPAD, cap=F.F_CAP); real = F.Dim(n_frames, F.GPR_F, cap=F.F_CAP)
    F.set_runtime_dims(n_frames=n_frames, nf_pad=nf_pad, pad_rows=nf_pad - n_frames)

    class Shim: pass
    self = Shim(); self.ue = ue; self.style_dim = 128
    self._up = lambda t: F._upload_const(ue, t)
    self._up_cap = lambda t, c: F.F0NPredictionFPGA._up_cap(self, t, c)
    self.identity_dram = self._up(torch.eye(64, dtype=torch.bfloat16))
    for n in ("_adain1d", "_conv1d", "_leaky_relu", "_nearest_upsample2x", "_depthwise_convtranspose_upsample2x",
              "_device_row_copy", "_adain_res_blk"):
        setattr(self, n, getattr(F.F0NPredictionFPGA, n).__get__(self))
    self._run = lambda ue: ue.stop_capture()
    torch.manual_seed(0)
    bf = lambda *s: torch.randn(*s).to(torch.bfloat16)
    w = dict(norm1_fc_w=bf(2 * Cin, 128), norm1_fc_b=bf(2 * Cin), norm2_fc_w=bf(2 * Cout, 128), norm2_fc_b=bf(2 * Cout),
             conv1_w=bf(Cout, Cin, 3), conv1_b=bf(Cout), conv2_w=bf(Cout, Cout, 3), conv2_b=bf(Cout))
    if upsample:
        w["pool_w"] = bf(Cin, 1, 3); w["pool_b"] = bf(Cin)
    if Cin != Cout:
        w["conv1x1_w"] = bf(Cout, Cin, 1)
    style = self._up(bf(128))
    x = ue.allocate_tensor_dram(F._cap() * Cin * 2)
    self._adain_res_blk(x, T, Cin, Cout, upsample, w, style, real_T=real)
    n = ue.capture_count
    return b"".join(ins.get_bytes() for ins in ue.capture_buffer[:n]), trace


def main():
    bad_total = 0
    for up, cin, cout in ((False, 512, 512), (True, 512, 256)):
        ref, tref = emit(78, up, cin, cout)
        for nf in (64, 128, 173, 1920):
            got, _ = emit(nf, up, cin, cout)
            if len(ref) != len(got):
                say(f"upsample={up}: n_frames=78 vs {nf}: INSTRUCTION COUNT differs "
                      f"{len(ref)//32} vs {len(got)//32}")
                bad_total += 1
                continue
            bad = [i for i in range(len(ref) // 32) if ref[i*32:(i+1)*32] != got[i*32:(i+1)*32]]
            say(f"upsample={up} Cin={cin} Cout={cout}: 78 vs {nf} frames: "
                  f"{len(ref)//32} inst, {len(bad)} differ")
            bad_total += len(bad)
            for i in bad[:8]:
                say(f"  inst {i}:")
                for fr in tref.get(i, [])[-4:-1]:
                    say("      " + fr.strip().splitlines()[0])
    say("RESULT:", "IDENTICAL" if bad_total == 0 else f"{bad_total} DIFFERENCES")
    return 0 if bad_total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
