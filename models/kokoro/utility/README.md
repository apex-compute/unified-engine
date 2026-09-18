# Kokoro utility scripts

Model-specific tooling. Not part of the inference path -- kokoro_test.py never
imports any of these.

## Hardware proofs

- **kokoro_trig_and_funcs.py** -- the transcendental and generator-composition
  hardware proofs. The accelerator has no sign / floor / sqrt / divide / sin /
  cos / exp cores, so each was synthesised from clamp ramps, LALU eltwise,
  matmul and CORDIC rotations, then proven on hardware before kokoro_fpga.py
  was allowed to depend on it. Covers exp-via-sigmoid, bounded-domain sin/cos
  polynomials, magic-number range reduction (the missing floor), CORDIC
  atan2/magnitude, Snake1D-vs-gelu cost, plus the Section 5b-5d compositions
  (large-T InstanceNorm, SineGen wrapped phase, STFT/iSTFT as block-Toeplitz
  matmuls). Moved out of the repo-wide user_hw_test.py, where ~800 lines of
  kokoro-only bring-up sat in the shared regression suite.

      python models/kokoro/utility/kokoro_trig_and_funcs.py --dev xdma0

  Results append to the same record_test registry user_hw_test.py writes, so
  `--summary-path` output is directly comparable.

## Single-bin instruction invariance

Checks that a frozen kokoro_bin/ program image behaves identically across
prompts of different lengths -- the property the whole single-bin design rests
on, and one that any emitter edit can silently break.

- **host_invariance_check.py** -- host-only (no FPGA). Verifies a residual
  block emits byte-identical instructions across frame counts, including
  counts that are exact multiples of 64 (zero pad rows), which the two-prompt
  comparison can miss. Run this after touching kokoro_fpga.py.
- **compare_programs.py** -- on-hardware counterpart: diffs two
  `--dump-programs` JSONs to find which captured program is prompt-dependent.
