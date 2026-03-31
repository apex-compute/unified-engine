# Qwen2.5-VL-3B

Vision-Language Model (VLM) inference on FPGA accelerator. Supports image+text and text-only prompts.

## Layout

- **qwen2.5_vl_3b_test.py** – Full VLM pipeline: vision encoder + prefill + decode
- **qwen2.5_vl_3b_config.json** – Model dimensions, DRAM layout, paths
- **qwen2.5_vl_3b_bin/** – Weights, HF model, and cached programs (generated at runtime)
- **test_image.jpeg** – Default test image

## Usage

```bash
# VLM with default image (demo.jpeg + default prompt)
python qwen2.5_vl_3b/qwen2.5_vl_3b_test.py

# VLM with custom image + custom prompt
python qwen2.5_vl_3b/qwen2.5_vl_3b_test.py --image /path/to/photo.jpg --prompt "What do you see?"

# Text-only (no vision encoder)
python qwen2.5_vl_3b/qwen2.5_vl_3b_test.py --image none --prompt "What is 2+2?"
```

## Cached Files

The `qwen2.5_vl_3b_bin/` folder contains auto-generated files that speed up subsequent runs:

- **Weight binaries** (`*_lm_fp4.bin`, `*_vision_q4.bin`) – Quantized model weights. Generated once from the HF model on first run.
- **Encoder program** (`encoder_program.bin`) – Compiled vision encoder. Generated once since all images use the same 336×336 resolution.
- **Decoder program** (`decoder_program.bin`) – Compiled decoder. Generated once since it covers all sequence length buckets.
- **Prefill programs** (`prefill_program_s*.bin`) – Compiled prefill programs, cached **per token count**. A new file is created for each unique prompt length (e.g., `s26` for 26-token text, `s171` for 171-token VLM). This is expected — the compiled program depends on the sequence length. Previously-cached lengths load instantly (~0.1s vs ~45s compile time).

To force a clean rebuild, delete the cached files:
```bash
rm qwen2.5_vl_3b/qwen2.5_vl_3b_bin/*.bin
```
