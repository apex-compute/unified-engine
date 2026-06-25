# GPT-2 Base (124M) on Unified Engine

GPT-2 Base running on the Unified Engine FPGA accelerator. Model downloaded from
HuggingFace (`openai-community/gpt2`).

## Key Differences from LLaMA/Gemma3/Qwen3

- **LayerNorm with bias** (not RMSNorm). Both weight and bias stored per norm.
- **Learned positional embeddings** (not RoPE). 1024 x 768 embedding table added host-side.
- **GELU activation** (not SwiGLU). Single c_fc projection, no gate.
- **Bias in all linear layers**: Q, K, V, output projection, c_fc, c_proj.
- **MHA** (not GQA): all 12 heads are both Q and KV heads. group_size=1.
- **Fused QKV**: HF model uses Conv1D c_attn; weights transposed and split during generation.
- **Weight tying**: lm_head weight = token embedding weight.

## Instruction Bin (dynamic PBI + shared-subroutine attention)

One prefill program + one decoder program are compiled into a single cached
instruction image `gpt2_bin/programs.bin` (+ `.json` meta, ~0.72 MiB
total). Runtime seq_len and attention bucket selection are driven by GPRs primed
by a tiny per-call preamble, so the same bin serves any prompt length up to
`prefill_context_size` (256) and any decode position up to `max_context_size`
(1024). The flash / decoder attention cores are compiled once as subroutines
after the program HALT; per-head call sites jump in and return.

Delete `gpt2_bin/programs.bin` to force a recompile (required after any
change to `tensor_init` buffer layout — DRAM addresses are baked into the bin).

## Usage

```bash
python3 -m venv ~/my_torch_env

source ~/my_torch_env/bin/activate

pip install -r requirements.txt

python models/gpt2/gpt2_test.py --prompt "The scientists at MIT announced today that they have discovered "
```

## Weight Binary

Generated automatically on first run from HuggingFace model. Stored at
`gpt2_bin/params.bin` (~311 MB).

BF16 weights (no quantization). The 124M parameter model is small enough to fit
in full precision, avoiding quality loss from weight quantization.
