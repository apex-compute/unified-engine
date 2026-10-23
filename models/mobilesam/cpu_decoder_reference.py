#!/usr/bin/env python
"""CPU oracle for the MobileSAM mask decoder: official mobile_sam decoder on the
HW's exact input (vette.jpg, /255, no mean/std) with one foreground point.
Prints all 4 predicted IoUs (mask tokens 0..3) so HW iou_pred can be judged.
  python models/mobilesam/cpu_decoder_reference.py [--point X Y] [--bf16]
"""
import argparse, os, sys
import numpy as np, torch
from PIL import Image
from mobile_sam import sam_model_registry

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(SCRIPT_DIR, "mobilesam_bin", "mobile_sam.pt")
IMG_PATH = os.path.join(SCRIPT_DIR, "..", "..", "test_samples", "vette.jpg")

ap = argparse.ArgumentParser()
ap.add_argument("--point", nargs=2, type=int, default=[300, 300])
ap.add_argument("--bf16", action="store_true", help="run the decoder in bf16")
ap.add_argument("--hw-emb", default=None, help="hw_decoder_inputs.npz: use the HW encoder embedding (neck) instead of the CPU one")
args = ap.parse_args()

img = Image.open(IMG_PATH).convert("RGB").resize((1024, 1024), Image.BILINEAR)
x = (torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0)[None]
sam = sam_model_registry["vit_t"](checkpoint=WEIGHTS).eval()
with torch.no_grad():
    if args.hw_emb:
        neck = torch.from_numpy(np.load(args.hw_emb)["neck"])      # (4096, 256) HWC from the HW
        emb = neck.reshape(64, 64, 256).permute(2, 0, 1)[None].contiguous()
        print("using HW encoder embedding from", args.hw_emb)
    else:
        emb = sam.image_encoder(x)                                    # (1,256,64,64)
    pts = torch.tensor([[[args.point[0] + 0.5, args.point[1] + 0.5]]], dtype=torch.float32)
    lbs = torch.tensor([[1]])
    sparse, dense = sam.prompt_encoder(points=(pts, lbs), boxes=None, masks=None)
    pe = sam.prompt_encoder.get_dense_pe()
    dec = sam.mask_decoder
    if args.bf16:
        dec = dec.to(torch.bfloat16); emb, sparse, dense, pe = [t.to(torch.bfloat16) for t in (emb, sparse, dense, pe)]
    masks, iou = dec.predict_masks(image_embeddings=emb, image_pe=pe,
                                   sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense)
    iou = iou[0].float()
    print(f"CPU iou_pred ({'bf16' if args.bf16 else 'fp32'}): {[round(v, 4) for v in iou.tolist()]}  best={iou.argmax().item()}")
    m = masks[0].float()
    print("mask areas (>0) at 256x256:", [(mm > 0).sum().item() for mm in m])
    np.savez(os.path.join(SCRIPT_DIR, "cpu_decoder_golden.npz"), iou=iou.numpy(), masks=m.numpy(),
             emb=emb[0].float().numpy())
    print("saved cpu_decoder_golden.npz")
