#!/usr/bin/env python3
"""
Export a LeRobot ACT policy (inference graph only) + a CPU golden reference.

Run this in the lerobot env (it needs lerobot/torchvision), NOT the HW env:

  ~/miniconda3/envs/lerobot/bin/python act_export.py                    # random-init weights
  ~/miniconda3/envs/lerobot/bin/python act_export.py --checkpoint <dir|hf-id>

Writes into act_bin/:
  act_weights.pt   plain fp32 state_dict of ACT.model minus the VAE encoder,
                   plus the two host-precomputed positional tables
                   (enc_pos [S,512], dec_pos [chunk,512]) and the model dims
  reference.npz    inputs (2 cams, state) and CPU fp32 golden intermediates:
                   backbone feature maps, encoder output, decoder output, actions

act_test.py (HW env) consumes only these two files; it never imports lerobot.

Fixed rig: 2 cameras 480x640, state/action dim 6 (5 arm joints + gripper), chunk 100.
"""
import argparse
import os

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.join(SCRIPT_DIR, "act_bin")

N_CAMS = 2
IMG_H, IMG_W = 480, 640
STATE_DIM = ACTION_DIM = 6   # 5 arm joints + gripper


def build_policy(checkpoint: str | None, seed: int):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    if checkpoint:
        policy = ACTPolicy.from_pretrained(checkpoint)
        cfg = policy.config
        cams = [k for k, f in cfg.input_features.items() if f.type == FeatureType.VISUAL]
        assert len(cams) == N_CAMS, f"checkpoint has {len(cams)} cameras, expected {N_CAMS}"
        assert cfg.action_feature.shape[0] == ACTION_DIM, cfg.action_feature.shape
        return policy, cfg, cams

    torch.manual_seed(seed)
    cams = [f"observation.images.cam{i}" for i in range(N_CAMS)]
    input_features = {"observation.state": PolicyFeature(FeatureType.STATE, (STATE_DIM,))}
    for c in cams:
        input_features[c] = PolicyFeature(FeatureType.VISUAL, (3, IMG_H, IMG_W))
    cfg = ACTConfig(input_features=input_features,
                    output_features={"action": PolicyFeature(FeatureType.ACTION, (ACTION_DIM,))},
                    pretrained_backbone_weights=None, device="cpu")
    policy = ACTPolicy(cfg)
    # Random init leaves FrozenBN at identity and biases at 0/small; perturb so the
    # BN fold and every bias path are actually exercised on HW.
    with torch.no_grad():
        for n, p in policy.model.named_parameters():
            if p.dim() == 1 and n.startswith("backbone"):
                p.add_(torch.randn_like(p) * 0.1)
        for n, b in policy.model.named_buffers():
            if n.endswith("running_var"):
                b.mul_(torch.rand_like(b) * 0.5 + 0.75)
            elif n.endswith("running_mean"):
                b.add_(torch.randn_like(b) * 0.1)
    return policy, cfg, cams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="ACT checkpoint dir or HF repo id")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(BIN_DIR, exist_ok=True)

    policy, cfg, cams = build_policy(args.checkpoint, args.seed)
    policy.eval()
    model = policy.model
    assert not cfg.pre_norm and cfg.feedforward_activation == "relu", "act_test.py assumes post-norm + ReLU"
    assert cfg.vision_backbone == "resnet18" and not cfg.replace_final_stride_with_dilation

    torch.manual_seed(args.seed + 1)
    images = [torch.rand(1, 3, IMG_H, IMG_W) for _ in cams]
    state = torch.randn(1, STATE_DIM)
    batch = {"observation.state": state, "observation.images": images}
    for c, im in zip(cams, images):
        batch[c] = im

    caps = {}
    def _cap(key):
        return lambda m, i, o: caps.setdefault(key, []).append(o.detach())

    hooks = [
        model.backbone.register_forward_hook(
            lambda m, i, o: caps.setdefault("feat", []).append(o["feature_map"].detach())),
        model.backbone.maxpool.register_forward_hook(_cap("pool")),
        model.backbone.layer1.register_forward_hook(_cap("layer1")),
        model.backbone.layer2.register_forward_hook(_cap("layer2")),
        model.backbone.layer3.register_forward_hook(_cap("layer3")),
        model.encoder.register_forward_hook(lambda m, i, o: caps.__setitem__("enc_out", o.detach())),
        model.decoder.register_forward_hook(lambda m, i, o: caps.__setitem__("dec_out", o.detach())),
    ]
    with torch.no_grad():
        actions, _ = model(batch)
    for h in hooks:
        h.remove()

    feats = caps["feat"]
    fh, fw = feats[0].shape[-2:]
    # Encoder positional table, built exactly as ACT.forward does: [latent, state, cam0 (h w), cam1 (h w)]
    with torch.no_grad():
        pos_1d = model.encoder_1d_feature_pos_embed.weight            # (2, 512)
        cam_pos = model.encoder_cam_feat_pos_embed(feats[0])           # (1, 512, fh, fw)
        cam_pos = cam_pos[0].permute(1, 2, 0).reshape(fh * fw, -1)    # (h*w, 512)
        enc_pos = torch.cat([pos_1d] + [cam_pos] * N_CAMS, dim=0)      # (S, 512)
        dec_pos = model.decoder_pos_embed.weight                        # (chunk, 512)

    sd = {k: v.detach().float().cpu() for k, v in model.state_dict().items()
          if not k.startswith("vae_encoder")}
    sd["_enc_pos"] = enc_pos.float().cpu()
    sd["_dec_pos"] = dec_pos.float().cpu()
    sd["_dims"] = {"dim_model": cfg.dim_model, "n_heads": cfg.n_heads, "dim_ff": cfg.dim_feedforward,
                   "n_enc": cfg.n_encoder_layers, "n_dec": cfg.n_decoder_layers,
                   "chunk": cfg.chunk_size, "latent": cfg.latent_dim, "state_dim": STATE_DIM,
                   "action_dim": ACTION_DIM, "n_cams": N_CAMS, "img_hw": (IMG_H, IMG_W),
                   "feat_hw": (fh, fw), "cams": cams,
                   "checkpoint": args.checkpoint or f"random(seed={args.seed})"}
    torch.save(sd, os.path.join(BIN_DIR, "act_weights.pt"))

    np.savez(os.path.join(BIN_DIR, "reference.npz"),
             images=torch.stack(images, 0)[:, 0].numpy(),          # (n_cams, 3, H, W)
             state=state[0].numpy(),
             feat=torch.stack(feats, 0)[:, 0].numpy(),             # (n_cams, 512, fh, fw)
             **{k: torch.stack(caps[k], 0)[:, 0].numpy() for k in ("pool", "layer1", "layer2", "layer3")},
             enc_out=caps["enc_out"][:, 0].numpy(),                 # (S, 512)
             dec_out=caps["dec_out"][:, 0].numpy(),                 # (chunk, 512)
             actions=actions[0].numpy())                            # (chunk, action_dim)

    n_params = sum(v.numel() for k, v in sd.items() if isinstance(v, torch.Tensor) and not k.startswith("_"))
    print(f"exported {n_params / 1e6:.2f}M inference params, S={enc_pos.shape[0]} tokens, "
          f"feat {fh}x{fw}, source={sd['_dims']['checkpoint']} -> {BIN_DIR}")


if __name__ == "__main__":
    main()
