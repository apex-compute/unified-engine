# pi0.5 LIBERO blog post — running list of things to cover

Running cart of topics/details to include in the extensive write-up about this model
and the closed-loop LIBERO benchmarking work. Append as new items come up.

## Confirmed to include

- [x] Exact inputs/outputs of one inference run. (libero_eval.py, pi05_torch_ref.py,
      pi05_libero_test.py)
      - Raw obs: agentview_image + robot0_eye_in_hand_image (256x256x3 uint8, rotated
        180deg "to match training"), + 8-dim state (eef_pos 3, axis-angle(quat) 3,
        gripper_qpos 2).
      - Preprocessing: images resized/padded to 224x224x3, normalized to [-1,1]; a
        3rd all-zero 224x224x3 slot appended (LIBERO has 2 cams, model wants 3) and
        attention-masked out via masked_cols rather than fed as content. State is
        NOT a numeric tensor into the model -- it's quantile-normalized, digitized
        into 256 bins, and stringified into the text prompt
        "Task: {language}, State: {ints};\nAction: ", then SentencePiece-tokenized.
      - Tensors into one inference call: images (3,224,224,3) float [-1,1];
        prompt_tokens (1,T_text) int64; noise, real shape (10,7) fixed via
        RandomState(0), padded to hw tile sizes (64,64)/(1,10,32) with zeros beyond
        the 7 real action dims. FPGA additionally pads action horizon 10->64 and
        width 7->64 for tiling.
      - Output: (10,7) normalized action chunk. 10 = action-chunk horizon (NOT the
        replan cadence -- --replan-steps, default 10, controls how much of the
        chunk is actually executed before re-inferring). 7 =
        [dx,dy,dz,droll,dpitch,dyaw,gripper].
      - Postprocess: un-normalized [-1,1] -> robot units via q01/q99 action stats,
        one row popped per env.step().
- [x] How rollout frames get compiled into the .mp4s under document/videos/.
      Library: imageio.mimwrite (not cv2/mediapy/ffmpeg), fps=10 hardcoded. Frame
      source: agentview (base) camera only, the same 224x224 frame fed to the
      model, appended once per EXECUTED env step (once per popped plan action --
      not per inference call, not per raw physics substep). Trigger: one mp4 per
      episode at episode end unless --no-video. Filename:
      {backend}_{task_suite}_t{task_id}_e{trial}_{success|failure|error}.mp4.
      PIL overlay (task language + backend/quant tag) baked in only at write time,
      never fed back into the model.

## Backlog (from earlier research, not yet slotted into a section)

- [ ] compile-once fix: root cause was a dirtied `vis_zeros` DRAM buffer misread as
      scratch (not a compile bug); `_compile_once()` caches compiled programs so
      program-DRAM doesn't march toward the 4GB ceiling — went from dying at ~3
      inferences to surviving 252.
- [ ] Two backends (torch/GPU reference vs FPGA) share identical preprocessing so
      episodes are paired; torch treated as oracle, diffed against FPGA via
      --dump-actions / --diff-actions.
- [ ] LIBERO/robosuite/mujoco/bddl stack is not vendored — editable pip install from
      a separate openpi checkout, path-inserted via sys.path.
- [ ] Merged single-process env (pi_requirements.txt): numpy<2 pin is load-bearing —
      lets robosuite/gym 0.25 and user_dma_core coexist in one interpreter.
- [ ] Weight export/provenance: openpi checkpoint from GCS (not HF, not PyPI —
      PyPI package is an empty 0.0.0 stub), ~13GB weights_export/, 51 tensors +
      manifest.json.
- [ ] run_from_bin pattern (Pi05Libero_Run): precompiled programs.bin/params.bin with
      a signature guard refusing to run if denoise_steps/masked-slot-count/DRAM bases
      don't match compile-time assumptions.
- [ ] Success criterion is binary per-episode (robosuite BDDL goal predicate), timeout
      via max_steps; per-episode try/except + results JSON saved after every episode
      so a crash doesn't lose progress.
- [ ] Prefix masking optimization: 832→576 tokens by dropping the masked slot,
      bit-identical output (251dB), ~24% faster — survives because RoPE positions
      are cumsum(mask), not arange.
- [ ] RoPE bring-up history: was entirely missing at first (config said rope:none);
      prefix rope fixed to 47dB; action-expert suffix rope was a separate, later bug.
- [ ] Masked-row SNR pitfall: including masked/padding rows in SNR calcs cost hours
      of false leads (-3.4dB vs +50dB depending on masking) — general lesson for how
      correctness was measured throughout.
- [ ] FPGA hard-reboot risk during long eval runs (no kernel log) — why per-episode
      checkpointing in libero_eval.py matters operationally, not just for crash safety.

## Open questions / TBD
(add here as they come up)
