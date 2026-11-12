"""Host only: what Strassen on 4-bit weights does to a text LLM's prefill.

The model runs whole text windows in one pass (that is a prefill), with its projections replaced:

    bf16        the published weights (the reference)
    if4         every projection quantized to the repo's IF4 blocks (what the FPGA runs today)
    if4 + S<L>  as if4, but the chosen projections run Strassen at level L: the 7^L leaf weight
                matrices are formed from the unquantized weights and each is quantized on its own

Strassen is modelled exactly as the device does it: the row split is real, so the quantization
error of one half of the tokens leaks into the other half. It is not a per-token weight change.

Reported per run: perplexity on the text, and how often the top-1 next token matches bf16's.

    python strassen/llm_quality.py                          # Qwen3-4B, levels 1-3 on up/down
    python strassen/llm_quality.py --ops q,o,up,down --levels 1
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from strassen_levels import quant4, weight_leaves  # noqa: E402  (host helpers; no device at import)

DEV = "cuda"
OPS = {"q": ("self_attn", "q_proj"), "k": ("self_attn", "k_proj"), "v": ("self_attn", "v_proj"), "o": ("self_attn", "o_proj"),
       "gate": ("mlp", "gate_proj"), "up": ("mlp", "up_proj"), "down": ("mlp", "down_proj")}


def deq(W):
    """IF4 quantize + dequantize of one [N, K] weight (bf16 on the GPU)."""
    return quant4(W.float(), "if4", DEV)[2].to(torch.bfloat16)


class StrassenLinear(torch.nn.Module):
    """y = x W^T by L levels of Strassen over IF4 leaves. Leaves live on the CPU (7^L of them)."""

    def __init__(self, W, L):
        super().__init__()
        self.L = L
        self.leaves = [deq(leaf).cpu() for leaf in weight_leaves(W.float().cpu(), L)]

    def run(self, x, L, leaf0):
        if L == 0:
            return x @ self.leaves[leaf0].to(DEV).float().T
        h, k = x.shape[0] // 2, x.shape[1] // 2
        x11, x12, x21, x22 = x[:h, :k], x[:h, k:], x[h:, :k], x[h:, k:]
        step = 7 ** (L - 1)
        p = [self.run(s, L - 1, leaf0 + i * step)
             for i, s in enumerate((x11 + x22, x21 + x22, x11, x22, x11 + x12, x21 - x11, x12 - x22))]
        top = torch.cat([p[0] + p[3] - p[4] + p[6], p[2] + p[4]], 1)
        return torch.cat([top, torch.cat([p[1] + p[3], p[0] - p[1] + p[2] + p[5]], 1)], 0)

    def forward(self, x):
        assert x.shape[0] == 1 and x.shape[1] % (1 << self.L) == 0, "one window, rows a multiple of 2^L"
        return self.run(x[0].float(), self.L, 0).to(x.dtype)[None]


def windows(tok, n, length):
    from datasets import load_dataset
    text = "\n\n".join(t for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
    ids = tok(text[: 40 * n * length], return_tensors="pt").input_ids[0]
    assert ids.numel() >= n * length
    return ids[: n * length].reshape(n, length)


@torch.no_grad()
def score(model, wins, ref_top=None):
    nll, tops = [], []
    for w in wins:
        logits = model(w[None].to(DEV)).logits[0, :-1].float()
        nll.append(torch.nn.functional.cross_entropy(logits, w[1:].to(DEV), reduction="none").cpu())
        tops.append(logits.argmax(-1).cpu())
    ppl, tops = float(torch.cat(nll).mean().exp()), torch.cat(tops)
    return ppl, tops, None if ref_top is None else float((tops == ref_top).float().mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=os.path.join(_HERE, "..", "models", "qwen3_4b", "qwen3_4b_bin", "Qwen3-4B"))
    ap.add_argument("--ops", default="up,down", help=f"projections that run Strassen: comma list of {', '.join(OPS)}")
    ap.add_argument("--levels", default="1,2,3")
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--length", type=int, default=512, help="tokens per window (the prefill length)")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16).to(DEV).eval()
    wins = windows(tok, args.windows, args.length)
    ops = args.ops.split(",")
    layers = model.model.layers
    print(f"{os.path.basename(os.path.normpath(args.model_dir))}: {len(layers)} layers, {args.windows} windows x {args.length} tokens (wikitext-2 test)")
    print(f"\n{'weights':<28} {'perplexity':>10} {'top-1 same as bf16':>19}")
    ppl, ref_top, _ = score(model, wins)
    print(f"{'bf16':<28} {ppl:10.3f} {'-':>19}")

    originals = {}
    for li, layer in enumerate(layers):         # every projection to IF4; keep the originals Strassen needs
        for op, (blk, name) in OPS.items():
            lin = getattr(getattr(layer, blk), name)
            if op in ops:
                originals[(li, op)] = lin.weight.data.cpu().clone()
            lin.weight.data = deq(lin.weight.data)
    ppl, _, same = score(model, wins, ref_top)
    print(f"{'if4 (all projections)':<28} {ppl:10.3f} {100 * same:18.1f}%")

    for L in (int(v) for v in args.levels.split(",")):
        for (li, op), W in originals.items():
            blk, name = OPS[op]
            setattr(getattr(layers[li], blk), name, StrassenLinear(W, L))
        torch.cuda.empty_cache()
        ppl, _, same = score(model, wins, ref_top)
        print(f"{f'if4 + Strassen L{L} on ' + args.ops:<28} {ppl:10.3f} {100 * same:18.1f}%", flush=True)


if __name__ == "__main__":
    main()
