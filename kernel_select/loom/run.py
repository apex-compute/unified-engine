#!/usr/bin/env python3
"""loom — the single runner. One command, everything auto-detected.

    python run.py <model-id> [--image x.jpg] [--ws N] [--max-new-tokens N]

Flow (no model-specific code anywhere):
    1. load HF model + config
    2. detect regions -> archetype per region (block_stack | decode | connector)
    3. print the plan + dimensions (the numbers)
    4. execute each workflow on the FPGA via its archetype path
    5. dump the ue-MLIR dialect + numbers to kernel_select/loom_dumps/
    6. report SNR vs torch

Vision block-stacks run end-to-end today (host patch-embed -> device encoder ->
host head). Decode is detected + dumped; its device path lands next.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import core
import engine as eng


def _load(model_id):
    """Pick the HF wrapper class from config.architectures so the loaded model
    exposes what each archetype needs (lm_head for decode, classifier for a pure
    vision classifier). Auto, no per-model hardcoding."""
    from transformers import (AutoConfig, AutoModelForImageTextToText,
                              AutoModelForCausalLM, AutoModelForImageClassification,
                              AutoModel)
    cfg = AutoConfig.from_pretrained(model_id)
    a = " ".join(getattr(cfg, "architectures", None) or []).lower()
    order = []
    if "imagetext" in a or "conditionalgeneration" in a or hasattr(cfg, "text_config"):
        order.append(AutoModelForImageTextToText)
    if "forcausallm" in a:
        order.append(AutoModelForCausalLM)
    if "imageclassification" in a or hasattr(cfg, "depths"):
        order.append(AutoModelForImageClassification)
    order += [AutoModelForCausalLM, AutoModel]
    for cls in order:
        try:
            return cls.from_pretrained(model_id, dtype=torch.float32).eval(), cfg
        except Exception:
            continue
    raise RuntimeError(f"could not load {model_id}")


def _tokenizer(model_id):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model_id)
    except Exception:
        return None


def _vision_inputs(model_id, image_path):
    from transformers import AutoImageProcessor
    from PIL import Image
    proc = AutoImageProcessor.from_pretrained(model_id)
    img = Image.open(image_path).convert("RGB")
    return proc(img, return_tensors="pt")["pixel_values"]


def _last_hidden(out):
    return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]


def _vlm_inputs(model_id, cfg, image_path, prompt):
    """Build VLM inputs via the HF processor: input_ids carrying <image> placeholder
    tokens + pixel_values. Returns (pixel_values, input_ids, image_token_id) or None
    if the model isn't a processor-backed VLM. Generic HF API, no model hardcoding."""
    try:
        from transformers import AutoProcessor
        from PIL import Image
        # do_image_splitting=False: one global image -> exactly image_seq_len tokens
        # (no tile/frame expansion). Matches the proven smolvlm2_test input prep.
        proc = AutoProcessor.from_pretrained(model_id, do_image_splitting=False)
        img = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(messages, add_generation_prompt=True)
        enc = proc(text=text, images=[img], return_tensors="pt")
        img_tok = getattr(cfg, "image_token_id", None)
        if img_tok is None:
            img_tok = proc.tokenizer.convert_tokens_to_ids("<image>")
        return enc["pixel_values"], enc["input_ids"], img_tok
    except Exception as e:
        print(f"[loom] VLM input prep failed ({e}); falling back to text-only.")
        return None


def _print_numbers(label, numbers):
    print(f"\n[loom] {label} numbers:")
    for k, v in numbers.items():
        print(f"   {k:>14} = {v}")


# --------------------------------------------------------------------------- #
# per-workload-kind drivers — each consumes/produces activations via `io`
# --------------------------------------------------------------------------- #
def w_encode(engine, wl, model, cfg, args, io):
    """Device vision encoder. Produces io['image_hidden'] for the connect workload;
    runs the classifier head + SNR only for pure classifiers."""
    from loom_ir import IRExecutor
    enc = wl.region.module
    ws = args.ws or int(getattr(cfg, "window_size", 7))
    px = io.get("pixel_values")
    if px is None:
        px = _vision_inputs(engine.model_id, args.image)
    if px.dim() == 5:                     # [B, num_images, C, H, W] -> [N, C, H, W]
        px = px.flatten(0, 1)

    # explicit host fallback (debugging only)
    if args.host_encoder:
        with torch.no_grad():
            io["image_hidden"] = _last_hidden(enc(px))
        return dict(host_encoder=True, hidden=tuple(io["image_hidden"].shape))

    # SINGLE-STAGE ViT/SigLIP encoder -> DEFAULT device path: lower the transformer
    # stack and feed it the host-computed embeddings. (Multi-stage Swin keeps the
    # proven full-model encoder_span path below.)
    multistage = hasattr(wl.region.config, "depths")
    if not multistage:
        import torch.nn as nn
        stack = getattr(enc, "encoder", enc)     # the repeating-layer container
        post = getattr(enc, "post_layernorm", None) or getattr(enc, "layernorm", None)

        class EncWithNorm(nn.Module):
            """The encoder pattern is: stack -> final LayerNorm. The post-norm is a
            REQUIRED op of the pattern, so it lowers onto device too (becomes the
            tail ue.layer_norm of the graph), not host."""
            def __init__(self):
                super().__init__()
                self.stack, self.post = stack, post
            def forward(self, x):
                try:
                    o = self.stack(x, attention_mask=None)
                except TypeError:
                    o = self.stack(x)
                h = o[0] if isinstance(o, (tuple, list)) else (
                    o.last_hidden_state if hasattr(o, "last_hidden_state") else o)
                return self.post(h) if self.post is not None else h

        with torch.no_grad():
            out = enc(px, output_hidden_states=True)
            emb = out.hidden_states[0]            # transformer-stack input
            enc_ref = _last_hidden(out)           # post-norm output (device target)
        hw, numbers = engine.run_encoder_stack(
            wl, stack=EncWithNorm(), embeds=emb, reference=enc_ref)
        io["image_hidden"] = hw.reshape(1, *hw.shape)   # post-norm, computed on device
        return numbers

    with torch.no_grad():
        out = enc(px, output_hidden_states=True)
        hs = getattr(out, "hidden_states", None)
        try:
            eo = enc.embeddings(px)
            emb = eo[0] if isinstance(eo, (tuple, list)) else eo
        except TypeError:
            emb = hs[0]
        enc_ref = hs[-1] if hs else _last_hidden(out)
    span = IRExecutor(enc, (px,), ws=ws, strict=False)
    enc_in, enc_out = span.encoder_span()
    hw, numbers = engine.run_block_stack(
        wl, model=enc, example_inputs=(px,), ws=ws,
        inject=(enc_in, emb), stop_after=enc_out, reference=enc_ref)
    io["image_hidden"] = hw                      # -> connect workload
    if hasattr(model, "classifier") and hasattr(model, "swin"):
        M, N = hw.shape
        with torch.no_grad():
            ref_top = int(model(px).logits[0].argmax())
            seq = model.swin.layernorm(hw.reshape(1, M, N))
            pooled = torch.nn.functional.adaptive_avg_pool1d(
                seq.transpose(1, 2), 1).flatten(1)
            hw_top = int(model.classifier(pooled)[0].argmax())
        numbers.update(torch_top=ref_top, device_top=hw_top, top_match=(hw_top == ref_top))
    return numbers


def w_connect(engine, wl, model, args, io):
    return engine.run_connect(wl, model=model, io=io)


def w_prefill(engine, wl, model, tok, args, io):
    return engine.run_prefill(wl, model=model, tokenizer=tok, prompt=args.prompt, io=io)


def w_decode(engine, wl, model, tok, args, io):
    eos = {tok.eos_token_id} - {None}
    out, text, numbers = engine.run_decode(
        wl, io=io, tokenizer=tok, max_new_tokens=args.max_new_tokens, eos_ids=eos)
    io["generated"] = (out, text)
    return numbers


def main():
    ap = argparse.ArgumentParser(prog="loom")
    ap.add_argument("model")
    ap.add_argument("--image", default=str(Path(__file__).resolve().parents[2]
                                           / "test_samples" / "vette.jpg"))
    ap.add_argument("--ws", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--prompt", default="x+3=5, what is x?")
    ap.add_argument("--host-encoder", action="store_true",
                    help="run the vision encoder on torch (host) instead of the "
                         "Swin-tuned device path; use for SigLIP/ViT VLM encoders")
    args = ap.parse_args()

    model, cfg = _load(args.model)
    engine = eng.Engine(args.model)

    # SCHEDULE: model -> ordered, numbered workloads (the forward pass).
    workloads = core.schedule(model, cfg, lower=None)
    print(f"\n[loom] {args.model}")
    print(f"[loom] detected {len(workloads)} workload(s):  "
          + "  →  ".join(wl.name for wl in workloads))
    for wl in workloads:
        print("  " + wl.report().replace("\n", "\n  "))

    # EXECUTE the schedule in order, threading activations through `io`.
    tok = None
    needs_tok = any(wl.kind in (core.PREFILL, core.DECODE_W) for wl in workloads)
    if needs_tok:
        tok = _tokenizer(args.model)
    io = {}

    # VLM: an encode + a decode workload -> prep image-grounded inputs so the
    # connect workload can merge image embeds into the prompt at <image> tokens.
    is_vlm = (any(wl.kind == core.ENCODE for wl in workloads)
              and any(wl.kind == core.DECODE_W for wl in workloads))
    if is_vlm:
        prepped = _vlm_inputs(args.model, cfg, args.image, args.prompt)
        if prepped is not None:
            io["pixel_values"], io["input_ids"], io["image_token_id"] = prepped
            print(f"[loom] VLM inputs: {io['input_ids'].shape[1]} tokens "
                  f"({int((io['input_ids'][0] == io['image_token_id']).sum())} image slots)")
    for wl in workloads:
        print(f"\n[loom] === {wl.name} [{wl.region.name}] ===")
        # Soft-reset the shared engine before each workload that builds a fresh
        # program — but NOT before decode, which continues prefill's on-device KV.
        if wl.kind != core.DECODE_W:
            engine.reset_device()
        if wl.kind == core.ENCODE:
            numbers = w_encode(engine, wl, model, cfg, args, io)
        elif wl.kind == core.CONNECT:
            numbers = w_connect(engine, wl, model, args, io)
        elif wl.kind == core.PREFILL:
            numbers = w_prefill(engine, wl, model, tok, args, io)
        elif wl.kind == core.DECODE_W:
            numbers = w_decode(engine, wl, model, tok, args, io)
        else:
            numbers = {"status": "kind not wired"}
        _print_numbers(wl.name, numbers)

    if "generated" in io:
        out, text = io["generated"]
        print(f"\n[loom] generated ({len(out)} tok): {text!r}")


if __name__ == "__main__":
    main()
