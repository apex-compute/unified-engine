# Gemma4 E4B example

Gemma4 E4B accelerator inference: text only (LM), image + text (VLM), audio +
text, and image(s) + audio + text, from a single instruction bin.

E4B is the same Gemma 4 architecture as E2B with larger dimensions, so it runs
on the E2B implementation in [`../gemma4_e2b`](../gemma4_e2b/README.md).
`gemma4_e4b_test.py` is a launcher: it points that code at this directory's
config and artifacts and calls its `main()`. Every `gemma4_e2b_test.py`
option applies (`--multi-core`, `--image`, `--audio`, `--frames`,
`--audio-seconds`, `--target-prefill-tokens`, `--profile`, `--summary`, ...).

```bash
python models/gemma4_e4b/gemma4_e4b_test.py --dev xdma0 --multi-core 8 --prompt "x+3=5, what is x?"
python models/gemma4_e4b/gemma4_e4b_test.py --dev xdma0 --multi-core 8 --image
python models/gemma4_e4b/gemma4_e4b_test.py --dev xdma0 --multi-core 8 --audio \
  --prompt "Transcribe the speech exactly."
```

## What differs from E2B

| | E2B | E4B |
| :--- | ---: | ---: |
| Hidden size | 1536 | 2560 |
| Layers (full attention) | 35 (7) | 42 (7) |
| Query heads / KV heads | 8 / 1 | 8 / 2 (4 query heads per KV head) |
| MLP width | 6144, 12288 from layer 15 | 10240 |
| Layers owning a KV cache | 15 | 24 |
| KV cache at 8192 context | 144 MB | 448 MB |
| Weight DRAM (8-engine tiled map) | 1225 MB | 2482 MB |

The shared code handles the differences by configuration: `num_q_heads` /
`num_kv_heads` from `file_info`, the KV cache stored per KV head (query head
*h* reads KV head *h* / 4), one decode attention call per KV head, the
private-weight reserve per engine derived from the weight sizes (352 MiB for
E4B), and the KV cache placed in the windows' shared pool rather than the
480 MiB tensor arena. The vision and audio encoders are identical; their
outputs project into the 2560-wide LM.

## Files

- **gemma4_e4b_test.py** -- the launcher.
- **gemma4_e4b_config.json** -- model and layout config (context 8192,
  512-row prefill program, longer prompts prefilled in chunks).
- **gemma4_e4b_bin/** -- `params.bin` + `params.json` (combined weights
  `[LM | vision | audio | host]`, ~10 GB, generated from the HF model if
  missing), `programs.bin` + `programs.json` (instruction image),
  `tokenizer/`, and the HF model `gemma-4-E4B-it/` (downloaded on first use).

## Measured (Alveo U50, 366.7 MHz, 8 engines)

`--image` (yosemite, 272-token prompt), against E2B on the same code:

| | E2B | E4B |
| :--- | ---: | ---: |
| Vision encoder, HW | 4.72 s | 4.81 s |
| Prefill, HW | 3.88 s | 7.92 s (274.8 model GFLOPS, 73% of peak) |
| TTFT, HW | 8.67 s | 12.73 s |
| Decode, first token | 31.0 tok/s | 15.5 tok/s |

The text-only run decodes the same answer as the earlier standalone E4B
implementation, with prefill 1.16 s instead of 66.8 s (18-token prompt; the
old prefill ran padded on one engine) and decode 16.1 tok/s instead of 3.9.
