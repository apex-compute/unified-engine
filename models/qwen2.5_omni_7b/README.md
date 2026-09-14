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

This model targets the 8 GiB Alveo U55 configuration and uses exactly eight
Unified Engine cores. Hardware topology, DRAM capacity, AXI width, and clock are
read from HW_INFO; startup fails rather than compiling an incompatible program
when the board does not report 8 GiB of DRAM and at least eight available
engines. On a 12-engine image, engines 0-7 are used and engines 8-11 remain idle.

Always pass `--multi-core 8`.

The runner detects the installed image and does not reprogram the FPGA. Builds
`0xfe984d16` and `0x0305d87d` use their native four-phase in-program handshake.
The older `0xe7ac2caf` build is also supported through host-separated one-shot
rendezvous. Both paths use the same FPGA kernels and numerical policy.

## Input limits

- Images use the fixed canonical 336 x 336-pixel geometry. Inputs are converted to RGB
  and resized before the vision encoder runs.
- Audio is processed in bounded 200-mel-frame chunks, with at most 600 mel
  frames and 150 audio soft tokens per request. Longer processed input is
  rejected.
- The complete templated prompt, including image and audio soft tokens, must fit
  the 384-token prefill limit; generation is bounded by the 2048-token context.

Prefill retains the logical prompt length. Current images round execution to a
64-row multiple, giving the 31-token sanity prompt eight rows on each engine;
the legacy host-segmented image uses a conservative 512-row tile. The logical
input remains limited to 384 tokens. Padding embeddings use a finite nonzero
sentinel, padding RoPE rows are finite, and padding key columns are masked, so
the extra rows cannot affect live-token attention.

Vision, audio, and Thinker weights are phase-shared in DRAM. Each input encoder
runs first, its soft-token output is retained, and the Thinker weights then reuse
the same model-weight window for prefill and decode. The transformer towers,
attention, Thinker prefill, and sharded Thinker decode operations distribute
their heavy work over all eight engines. Layout/merge operations and the
phase-shared decode O projection execute on engine 0, still on the FPGA;
engines 8-11 remain unused.

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
and loads the 686-MiB decode-only BF16 O region. Decode therefore uses BF16 V/O;
O runs on engine 0 through the proven static M=1 FPGA tiler used by Gemma4 E2B,
while the remaining decoder projections and head retain their eight-engine
private shards. This phase change moves no learned arithmetic to the host and
remains within the unchanged 8-GiB map. The LM head remains 64-column aligned
at 152064 rows, while its 399 rows beyond the tokenizer's 151665 valid IDs
receive a device-side minimum-BF16 bias and therefore cannot win the FPGA
global argmax.
Greedy generation stops only on Omni's declared `<|im_end|>` EOS (151645); its
distinct
`<|endoftext|>` padding ID (151643) is not treated as EOS.

## Build and run

Run commands from the repository root after activating the PyTorch environment
created by the root README:

```bash
source ~/my_torch_env/bin/activate
```

The first test invocation downloads only
the checkpoint shards used by the Thinker and converts them into the cached
`qwen2.5_omni_7b_bin/params.bin` bundle. To perform that one-time conversion
separately, run:

```bash
python models/qwen2.5_omni_7b/qwen2.5_omni_7b_weights.py
```

The test entrypoint reuses a compatible parameter bundle, compiles the requested
encoder and Thinker programs for the live eight-engine topology, and runs the
sanity check. It does not expose a program-bin reuse mode.

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
# request fits the 384-token prefill budget)
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
