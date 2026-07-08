#!/usr/bin/env python3
"""Compile-only diagnostic: print self.capture_count at each per-head jump
site in layer 0, WITHOUT ever calling start_execute_from_dram. Pure Python/
compile-time instrumentation -- only touches hardware for weight_init's DMA
writes (already proven safe/repeatable), never triggers execution, so safe
to run even if the device's execute path is still suspect after a reset.

Goal: see whether capture_count crosses a suspicious boundary (256/512/1024,
matching the 256-instruction I-cache limit seen elsewhere in this codebase)
right around head3->head4, where debug_stepwise_prefix.py found the hang.
"""
from pi05_libero_test import Pi05Libero_UnifiedEngine, init_hang_prevention, _CFG
import numpy as np

ue = Pi05Libero_UnifiedEngine()
ue.DEBUG_STOP_AFTER = 12  # just past layer0's head loop (checkpoint 12 = attn_permute)
init_hang_prevention(ue)
ue.weight_init()
ue.tensor_init(_CFG["defaults"].get("max_seq", 512))

prompt_tokens = np.random.RandomState(1).randint(0, 257152, size=(16,))
seq_len = _CFG["model"]["prefill_max_seq_len"]
valid_len = ue.embed_and_concat_prefix(prompt_tokens, vision_embeddings=None, seq_len=seq_len)

ue.start_capture()
ue.compile_prefix(seq_len, valid_len)  # will raise _DebugStop internally, caught, returns normally
print("compile-only run finished (no execute attempted).")
