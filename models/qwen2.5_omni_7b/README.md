# Qwen2.5-Omni-7B

Qwen2.5-Omni-7B inference on the Apex Compute Unified Engine. This implementation
accelerates the **Thinker path only**:

- text input -> text output
- image + text input -> text output
- audio + text input -> text output
- image + audio + text input -> text output

The Talker and token2wav stages that synthesize speech are intentionally out of
scope. Their weights and working tensors do not fit concurrently with the
Thinker, vision, and audio stages in the current hardware layout.

## Hardware target

This model targets the 8 GiB Alveo U55C configuration and uses exactly eight
Unified Engine cores. Hardware topology, DRAM capacity, AXI width, and clock are
read from HW_INFO; startup fails rather than compiling an incompatible program
when the board does not report 8 GiB of DRAM and at least eight available
engines. On a 12-engine image, engines 0-7 are used and engines 8-11 remain idle.

Always pass `--multi-core 8`.

The runner detects the installed image and does not reprogram the FPGA. Builds
`0xfe984d16`, `0x0305d87d`, and `0xb3ed9175` use their native four-phase
in-program handshake. Build `0xe7ac2caf` is also supported through
host-separated one-shot rendezvous. Both paths use the same FPGA kernels and
numerical policy.

## Input limits

- Images use the fixed canonical 336 x 336-pixel geometry. Inputs are converted to RGB
  and resized before the vision encoder runs.
- Audio is processed in bounded 200-mel-frame chunks, with at most 600 mel
  frames and 150 audio soft tokens per request. Longer processed input is
  rejected.
- The complete templated prompt, including image and audio soft tokens, must fit
  the 2500-token prefill limit; generation is bounded by the 2500-token context.

Prefill retains the logical prompt length. Current images round execution to a
64-row multiple, giving the 31-token sanity prompt eight rows on each engine;
the legacy host-segmented image uses a conservative 512-row tile. The logical
input remains limited to 2500 tokens. Padding embeddings use a finite nonzero
sentinel, padding RoPE rows are finite, and padding key columns are masked, so
the extra rows cannot affect live-token attention.

Vision, audio, and Thinker weights are phase-shared in DRAM. Each input encoder
runs first, its soft-token output is retained, and the Thinker weights then reuse
the same model-weight window for prefill and decode. The transformer towers and
Thinker prefill use all eight engines. Decode's sharded projections and LM head,
including the BF16 O column stripes, also use engines 0-7. The optimized
one-round GQA path assigns its four complete KV groups to engines 0-3; engines
4-7 remain required handshake participants. Remaining serial layout/merge work
and the final global argmax run on engine 0, still on the FPGA. Engines 8-11
remain unused.

All learned inference arithmetic runs on the U55: vision patch projection,
audio conv1/conv2 and GELU, the vision/audio transformer towers, IF8 token
embedding lookup and dequantization, the Thinker, LM head, and final global
argmax. The Python host is limited to control/DMA, tokenization, file/media
decode, resize/resample/log-mel preparation, masking, and layout transforms.
The entrypoint hides CUDA/HIP/ROCm before importing PyTorch and fails closed if
PyTorch reports another accelerator backend active or visible.

The 152064 x 3584 embedding is distinct from the untied LM head and is preserved
as its own IF8 table. Its padded accelerator layout occupies 538.31 MiB, split
equally across the eight private windows (67.29 MiB/core). Decoder weights other
than O consume about 434.0 MiB/core; together with the embedding they use
501.27 MiB of each 504-MiB private weight arena, leaving 2.73 MiB/core. Each
512-MiB engine window reserves the remaining 8 MiB for tensor scratch, and each
worker has a separate 8-MiB ISA slice. Prompt rows are dequantized directly into
LM DRAM; a generated token's lookup is inlined into its decoder preamble. No
embedding row or logits vector is evaluated on the host.

The Thinker follows Qwen2.5-VL's mixed projection policy during prefill: V is
BF16 for attention accuracy, while Q/K/O and GATE/UP/DOWN are IF4. After a
successful full 28-layer prefill, the runtime reclaims the shared prefill image
and scatters the 686-MiB decode-only BF16 O region into eight 85.75-MiB
upper-PARAMS stripes: one 448-column shard per engine across all 28 layers.
Decode therefore uses BF16 V/O; O runs across engines 0-7 through the dense M=1
sharded tiler, while the remaining decoder projections and head retain their
private shards. This phase change moves no learned arithmetic to the host and
remains within the unchanged 8-GiB map. The LM head remains 64-column aligned at
152064 rows, while its 399 rows beyond the tokenizer's 151665 valid IDs receive
a device-side minimum-BF16 bias and therefore cannot win the FPGA global argmax.
Greedy generation stops only on Omni's declared `<|im_end|>` EOS (151645); its
distinct
`<|endoftext|>` padding ID (151643) is not treated as EOS.

## Measured decode performance

On U55C build `0x0305d87d` at 300 MHz, an eight-engine text run at short context
measures 10.04 tokens/s end to end and 10.26 tokens/s for the first FPGA decode
step. A 115-step arithmetic response measures 9.85 tokens/s as the KV history
grows. These rates include host control, DMA, FPGA argmax readback, and
detokenization; learned inference arithmetic remains on the FPGA.

## Performance reporting

Every run writes a Markdown summary next to this script, named for the CLI
config (`qwen2.5_omni_7b_test_xdma0_image_multi-core_8.md`). It is built only
from metrics the stages already recorded plus host bookkeeping, so writing it
launches no FPGA program and cannot perturb what it measures. Pass
`--summary PATH` to redirect it or `--no-summary` to skip it.

The summary reports, per stage (vision, audio, prefill, decode): work in
GFLOP, HW-counter latency, achieved GFLOPS, percent of the run's peak
(`freq_MHz x 128 FLOP/cycle x engines`), effective speedup against one
engine's peak, and the CPU wall time around the same stage. The gap between
the two clocks is host overhead. TTFT covers whichever encoders the request
ran plus prefill.

`--high` is a performance-only workload for a multi-camera request that cannot
fit as one resident context: four 896x896 camera frames (4096 vision tokens),
a 6-second spoken query (the 600-mel-frame encoder maximum), and 6144 aggregate
input tokens. Vision, audio, prefill, and decode each run in a fresh process
with an independent eight-core reset and compilation. Every camera frame runs
on the FPGA; segmented prefill is projected as three times one isolated
2048-token FPGA measurement using the 2560-row physical KV allocation. The
summary reports both the segmented TTFT and an estimated monolithic
6144-token TTFT derived from static full-context model FLOPs and the measured
segmented effective GFLOPS. It also reports decode speed at the final resident
2048-token chunk. Media embeddings and KV history are deliberately not carried
between chunks, so this benchmark measures performance and does not validate
numerics or claim that a 6144-token request fits in DRAM.

A second table reports **effective throughput against model FLOPs**: the work
the architecture owes at its own dimensions -- true prompt length, true
attention windows, matrix products only -- rather than the padded,
tile-aligned, mask-widened shapes the engine actually issues and bills. Those
counts come from `qwen2.5_omni_7b_model_flops.py`, derived entirely from the
model config so they follow it rather than drifting from it. Dividing model
FLOPs by the same measured FPGA time gives a rate comparable across
implementations and accelerators, and the `Useful` column is the ratio of the
two: how much of what the engine issued the model actually needed. A stage
marked `!` billed fewer FLOPs than the architecture requires, which is
impossible -- padding only adds work -- and means that stage's own accounting
is undercounting.

A multi-core scaling section converts each stage's speedup into an implied
serial fraction by solving Amdahl's law for `s`. Because this model requires
exactly eight engines, no single-engine baseline can be measured; the speedup
is taken against one engine's *peak*, which makes it a lower bound and folds
every other inefficiency into the serial estimate. `--profile` is what
separates those.

`--profile` compiles the vision encoder, prefill, and decoder with per-phase
HALT checkpoints, then runs a profiled prefill and two profiled decode steps --
one at the prompt's own context and one at `--profile-ctx` (default 2500) --
and reports a per-phase table for each: calls, total ms, share of stage, GFLOP,
GFLOPS, and percent of peak. Phases marked `*` run on engine 0 alone and are
scored against one engine's peak. The share column is the one to act on: it
says which phase is worth sharding next. A profile run replaces generation, and
its summary is written to a separate `..._profile_...md` so it never overwrites
a generation run's report. The audio encoder has no checkpoints, so it appears
at stage level only.

```bash
# Generation run + summary
python models/qwen2.5_omni_7b/qwen2.5_omni_7b_test.py --multi-core 8 --image

# Performance-only 4-camera / 6-second audio / 6144-token aggregate benchmark
python models/qwen2.5_omni_7b/qwen2.5_omni_7b_test.py --multi-core 8 --high

# Per-phase breakdown instead of generation
python models/qwen2.5_omni_7b/qwen2.5_omni_7b_test.py --multi-core 8 --image --profile
```

## Build and run

Run commands from the repository root after activating the PyTorch environment
created by the root README:

```bash
source ~/my_torch_env/bin/activate
```

The first test invocation downloads the checkpoint shards used by the Thinker,
vision encoder, and audio encoder, then performs a one-time conversion into the
cached `qwen2.5_omni_7b_bin/params.bin` bundle. `params.json` records the model
revision, configuration fingerprint, region bounds, and tensor manifest used to
validate that bundle before inference. The converter also creates
`qwen2.5_omni_7b_bin/processor/`, a minimal tokenizer/processor metadata bundle.
After conversion, a normal inference run reads all learned tensors from the
validated `params.bin` and loads preprocessing metadata locally from that
minimal bundle; it does not need the safetensor shards or full checkpoint
directory. To perform the one-time conversion separately, run:

```bash
python models/qwen2.5_omni_7b/qwen2.5_omni_7b_weights.py
```

The test entrypoint follows the default Gemma4 E2B program-image flow. On every
normal invocation it freshly compiles the stable ISA for the requested encoder
and Thinker stages against the live eight-engine topology, atomically stores the
master and seven worker sections in the combined
`qwen2.5_omni_7b_bin/programs.bin` plus `programs.json`, reopens and validates
those on-disk bytes, and executes the reloaded sections. It does not execute the
compiler's in-memory copy and does not currently expose a `--bin-reuse` mode.
The address-coupled prefill and decoder groups are published together in one
artifact generation. An inter-process file lock also rejects concurrent Omni
runs before they can race the artifact files or the FPGA queues.

Request-specific embedding, decode-dispatch, and rendezvous/flag ISA remains
runtime-generated because it contains token-, position-, address-, or
request-state values. This includes token-specific IF8 embedding lookup and
dequantization, which still executes entirely on the FPGA. This is the same
boundary used by the Gemma4 E2B flow: stable stage programs execute from the
validated program image, while per-request FPGA control/embedding ISA is
generated at runtime. All learned arithmetic remains FPGA-only.

The deploy-time artifacts are therefore:

- `params.bin` + `params.json` -- all learned Thinker, vision, audio, embedding,
  and LM-head tensors plus their validated manifest
- `programs.bin` + `programs.json` -- the freshly compiled, atomically packaged,
  and reloaded eight-engine stable stage ISA
- `processor/` -- the minimal local tokenizer and multimodal processor metadata

None of this changes or reloads the FPGA image. The runner requires the existing
U55C 8-GiB image, uses exactly engines 0-7, and leaves any additional engines
idle. CPU and GPU are not used for learned inference compute; the host duties
remain the preprocessing, layout, DMA, and control operations listed above.

```bash
# Text sanity check
python models/qwen2.5_omni_7b/qwen2.5_omni_7b_test.py --multi-core 8 \
  --prompt "If x + 3 = 5, what is x?"

# Image caption; bare --image uses test_samples/yosemite.jpg
python models/qwen2.5_omni_7b/qwen2.5_omni_7b_test.py --multi-core 8 --image

# Custom image
python models/qwen2.5_omni_7b/qwen2.5_omni_7b_test.py --multi-core 8 \
  --image /path/to/image.jpg --prompt "Describe this image in detail."

# Audio transcription
python models/qwen2.5_omni_7b/qwen2.5_omni_7b_test.py --multi-core 8 \
  --audio test_samples/apex.wav --prompt "Transcribe the speech exactly."

# Joint audio and image understanding (audio is placed first; the combined
# request fits the 2500-token prefill budget)
python models/qwen2.5_omni_7b/qwen2.5_omni_7b_test.py --multi-core 8 \
  --image --audio test_samples/apex.wav
```

The shipped audio fixture is stereo 48 kHz PCM. The preprocessing path mixes it
to mono and resamples it to the model's expected rate, so the default audio test
also covers those conversions.

## Automated hardware checks

The root model harness registers text, image, audio, and joint image+audio cases.
Each case runs on eight engines after random data has been written across the
full 8 GiB reported by HW_INFO.

```bash
# All four Qwen2.5-Omni cases
python model_auto_test.py --only \
  qwen2.5_omni_7b qwen2.5_omni_7b_vlm qwen2.5_omni_7b_audio \
  qwen2.5_omni_7b_joint --verbose

# Through Make (also clears compiled program bins first)
make model_test qwen2.5_omni_7b qwen2.5_omni_7b_vlm \
  qwen2.5_omni_7b_audio qwen2.5_omni_7b_joint verbose
```

The image check requires a coherent Yosemite description with scene keywords.
The audio check reads structured `TEST_RESULT` output and looks for multiple
concepts from the known `apex.wav` speech rather than requiring one exact
transcription.
