#!/usr/bin/env python3
"""Stock LocateAnything-3B reference on CUDA (transformers==4.57.1 venv).

Produces ground-truth boxes + captures intermediate tensors (vision embeds,
connector output, prefill logits) so our own-PyTorch reimplementation can be
parity-checked against it. Inputs are built manually (no decord/lmdb).

Run with:  ~/la_ref_env/bin/python3 _reference.py --image <path> --prompt-kind detect --query car
"""
import argparse, os, sys, time
import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "locateanything_3b_bin", "LocateAnything-3B")
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # .../unified-engine
IMG_CONTEXT = "<IMG_CONTEXT>"

PROMPTS = {
    "detect":       "Locate all the instances that matches the following description: {q}.",
    "ground_multi": "Locate all the instances that match the following description: {q}.",
    "ground_one":   "Locate a single instance that matches the following description: {q}.",
    "point":        "Point to: {q}.",
}


def build_text(question: str, num_image_tokens: int) -> str:
    # Mirrors processing_locateanything.py: py_apply_chat_template + replace_media_placeholder
    img_block = f"<image 1><img>{IMG_CONTEXT * num_image_tokens}</img>"
    return (
        "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n"
        "<|im_start|>user\n"
        f"{img_block}{question}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=os.path.join(REPO_ROOT, "test_samples", "vette.jpg"))
    ap.add_argument("--prompt-kind", default="detect", choices=list(PROMPTS))
    ap.add_argument("--query", default="car")
    ap.add_argument("--mode", default="slow", choices=["slow", "hybrid", "fast"])
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--capture", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "reference_capture.pt"))
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

    print(f"image: {args.image}")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    improc = AutoImageProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_DIR, dtype=torch.bfloat16,
                                      trust_remote_code=True).to("cuda").eval()

    assert tok.convert_tokens_to_ids(IMG_CONTEXT) == model.config.image_token_index, \
        "IMG_CONTEXT id mismatch"

    img = Image.open(args.image).convert("RGB")
    feat = improc(images=img, return_tensors="pt")
    pixel_values = feat["pixel_values"]                       # [num_patches, 3, 14, 14]
    _ghw = feat["image_grid_hws"]
    # generate() only moves image_grid_hws to the model device when it is a numpy
    # array; BatchFeature(return_tensors="pt") turns it into a CPU tensor, so coerce
    # back to numpy to avoid a CPU/CUDA device mismatch in the vision encoder.
    grid_hws = _ghw.cpu().numpy() if torch.is_tensor(_ghw) else np.asarray(_ghw)
    gh, gw = int(grid_hws[0][0]), int(grid_hws[0][1])
    mk = improc.merge_kernel_size
    num_image_tokens = (gh * gw) // (mk[0] * mk[1])
    print(f"grid_hws={grid_hws.tolist()} patches={pixel_values.shape[0]} "
          f"image_tokens={num_image_tokens} merge={mk}")

    question = PROMPTS[args.prompt_kind].format(q=args.query)
    text = build_text(question, num_image_tokens)
    enc = tok([text], return_tensors="pt", padding=False)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    n_img_in_ids = int((input_ids[0] == model.config.image_token_index).sum())
    print(f"prompt_len={input_ids.shape[1]} img_tokens_in_ids={n_img_in_ids} (expect {num_image_tokens})")
    assert n_img_in_ids == num_image_tokens, "image token count mismatch vs prompt"

    # ---- capture intermediate activations via hooks ----
    cap = {}
    h1 = model.vision_model.register_forward_hook(
        lambda m, i, o: cap.__setitem__("vit_embeds",
            (torch.cat(o, 0) if isinstance(o, (list, tuple)) else o).detach().float().cpu()))
    h2 = model.mlp1.register_forward_hook(
        lambda m, i, o: cap.__setitem__("connector_out", o.detach().float().cpu()))

    dev = "cuda"
    pv = pixel_values.to(dev, dtype=torch.bfloat16)
    ids = input_ids.to(dev)
    am = attention_mask.to(dev)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            pixel_values=pv, input_ids=ids, attention_mask=am,
            image_grid_hws=grid_hws, tokenizer=tok,
            max_new_tokens=args.max_new_tokens, use_cache=True,
            generation_mode=args.mode, do_sample=False, temperature=0.0,
            repetition_penalty=1.0, verbose=True,
        )
    dt = time.time() - t0
    h1.remove(); h2.remove()

    answer = out[0] if isinstance(out, tuple) else out
    print("\n================ RAW ANSWER ================")
    print(answer)
    print("============================================")
    import re
    boxes = re.findall(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer)
    print(f"\n{len(boxes)} boxes (normalized /1000): {boxes}")
    print(f"generate wall time: {dt:.1f}s  (mode={args.mode})")

    torch.save({
        "image": args.image, "query": question, "mode": args.mode,
        "pixel_values": pixel_values.cpu(), "image_grid_hws": grid_hws,
        "input_ids": input_ids.cpu(), "num_image_tokens": num_image_tokens,
        "answer": answer, "boxes": boxes,
        "vit_embeds": cap.get("vit_embeds"), "connector_out": cap.get("connector_out"),
    }, args.capture)
    print(f"captured reference -> {args.capture}")
    print(f"  vit_embeds {tuple(cap['vit_embeds'].shape)}  connector_out {tuple(cap['connector_out'].shape)}")


if __name__ == "__main__":
    main()
