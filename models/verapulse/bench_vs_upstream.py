#!/usr/bin/env python3
"""FPGA vs the checkpoint's OWN forward pass, on byte-identical inputs.

    PYTHONPATH=/home/rohit/unified-engine python models/verapulse/bench_vs_upstream.py --engines 8

WHY THIS EXISTS. verapulse_test's --snr gate scores the accelerator against
VeraPulseRef, our torch reconstruction. Both read the SAME config, so any assumption they
SHARE gates clean -- that is exactly how a 3-camera prefix scored 47 dB while the model
only ever saw 2. This script replaces the reference with lerobot 0.6.1's real
VLAFlowMatching (verified byte-identical to the checkpoint's shipped reference copies),
so it can see model errors the SNR gate structurally cannot.

IDENTICAL INPUTS, NOT EQUIVALENT ONES. Both paths are handed the same image tensor, the
same token ids, the same state and the same noise. Nothing goes through prepare_images or
prepare_state here, deliberately: those are lerobot's and would only be applied to one
side. Pixel realism is irrelevant -- the question is whether the accelerator computes the
same FUNCTION, and for that the inputs only have to match, not to be in-distribution.

The noise buffer is built exactly the way run_denoise builds it (1e-6 pad, randn in the
real DoF) so both integrate the same trajectory. A mismatched noise makes two correct
implementations disagree completely, which reads as a hardware fault.
"""

import argparse
import os
import sys

os.environ.setdefault("VERAPULSE_VARIANT", "smolvla")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import verapulse_test as vp  # noqa: E402


def snr_db(hw, ref):
    hw, ref = hw.float().flatten(), ref.float().flatten()
    noise = (hw - ref).pow(2).mean()
    if noise == 0:
        return float("inf")
    return float(10.0 * torch.log10(ref.pow(2).mean() / noise))


def cos_sim(hw, ref):
    hw, ref = hw.float().flatten(), ref.float().flatten()
    return float(torch.dot(hw, ref) / (hw.norm() * ref.norm()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hw-gelu", action="store_true",
                    help="patch UPSTREAM's vision MLP to quick_gelu so the hardware's "
                         "activation substitution (~8.4 dB at the vision output) is "
                         "common-mode and this measures arithmetic, not the known "
                         "approximation")
    ap.add_argument("--prompt", default="Pick up the tape.")
    ap.add_argument("--ckpt",
                    default="/home/rohit/vera_pulse_finetune_v0/checkpoints/run01_020000")
    args = ap.parse_args()

    cfg = vp._CFG
    V, HEAD = cfg["vision"], cfg["action_head"]
    slots, IMG = V["num_image_slots"], V["image_size"]
    chunk, adim, apad = HEAD["chunk_size"], HEAD["action_dim"], HEAD["max_action_dim"]

    # ---- deterministic shared inputs -------------------------------------------------
    g = torch.Generator().manual_seed(args.seed)
    images = torch.randint(0, 256, (slots, IMG, IMG, 3), generator=g,
                           dtype=torch.uint8).float() / 255.0
    state = torch.randn(HEAD["state_dim"], generator=g)
    token_ids, text_mask = vp.tokenize(args.prompt, cfg["lm"]["tokenizer_max_length"],
                                       return_mask=True)

    # Same construction as run_denoise: 1e-6 everywhere (an exact zero NaNs the
    # epsilon-free rms_norm), randn only in the real DoF.
    noise = torch.full((chunk, apad), 1e-6, dtype=torch.float32)
    noise[:, :adim] = torch.randn(chunk, adim, generator=torch.Generator().manual_seed(0))

    print(f"variant={vp.VARIANT} slots={slots} prompt={args.prompt!r} "
          f"real_tokens={int(text_mask.sum())}/{len(token_ids)}")

    # ---- 1. FPGA ---------------------------------------------------------------------
    ue = vp.VeraPulse_UnifiedEngine()
    vp.configure_engines(ue, args.engines, tag="bench")
    ue.weight_init()
    ue.tensor_init()
    ue.precompile_all(stages=("vision", "prefix", "denoise"))

    vis = ue.run_vision(images)
    ue.run_prefix(vis, token_ids, state, text_mask=text_mask)
    ue.run_denoise(noise)
    hw = ue._last_denoise["x_t_padded"][:chunk, :adim].float()      # [50, 6]

    # ---- 2. the real model -----------------------------------------------------------
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained(args.ckpt).to("cpu").eval()

    if args.hw_gelu:
        # THE ACTIVATION IS A KNOWN, DELIBERATE SUBSTITUTION, NOT A BUG. The model
        # specifies gelu_pytorch_tanh; the silicon's fused activation is quick_gelu
        # (x*sigmoid(1.702x)). That difference alone costs ~8.4 dB at the vision output
        # and would swamp everything else, making this comparison too blunt to catch a
        # real fault. Patching UPSTREAM'S OWN module (rather than bending our side toward
        # the hardware) keeps this the checkpoint's forward pass while making GELU
        # common-mode, so the number goes back to measuring arithmetic.
        import types
        qg = lambda t: t * torch.sigmoid(1.702 * t)

        def fwd(self, x):
            return self.fc2(qg(self.fc1(x)))

        n = 0
        for name, mod in policy.model.vlm_with_expert.named_modules():
            # Structural match, not a hardcoded path: the vision tower's MLP is the only
            # fc1/fc2 pair under a 'vision' qualified name. The LM and expert stacks are
            # gated SiLU (gate/up/down) and are untouched.
            if "vision" in name and hasattr(mod, "fc1") and hasattr(mod, "fc2"):
                mod.forward = types.MethodType(fwd, mod)
                n += 1
        if n == 0:
            raise RuntimeError("--hw-gelu found no vision MLP to patch")
        print(f"  [oracle] patched {n} vision MLP(s) to quick_gelu (GELU common-mode)")

    # sample_actions' own contract, bypassing prepare_images/prepare_state so both sides
    # see the SAME tensors. images: list of (B,C,H,W); state: (B,32); noise: (B,50,32).
    up_images = [images[i].permute(2, 0, 1).unsqueeze(0).float() for i in range(slots)]
    up_masks = [torch.ones(1, dtype=torch.bool) for _ in range(slots)]
    # BOOL, not long: make_att_2d_masks feeds this straight into torch.where as the
    # condition, and a Long mask raises rather than silently miscomputing.
    up_tok, up_lmask = token_ids.unsqueeze(0), text_mask.unsqueeze(0).to(torch.bool)

    m = policy.model
    with torch.no_grad():
        ref_full = m.sample_actions(up_images, up_masks, up_tok, up_lmask,
                                    state.unsqueeze(0), noise=noise.unsqueeze(0))
    ref = ref_full[0, :chunk, :adim].float()                        # [50, 6]

    # ---- 2b. upstream intermediates, for the stage ladder ----------------------------
    # Re-run the prefix half explicitly. sample_actions does exactly this internally; we
    # repeat it so the intermediates are in hand rather than hooking into it.
    from lerobot.policies.common.vla_utils import make_att_2d_masks
    with torch.no_grad():
        up_img_emb = torch.cat([m.vlm_with_expert.embed_image(im) for im in up_images],
                               dim=1)[0].float()                    # [128, 960] UNSCALED
        p_embs, p_pad, p_att = m.embed_prefix(up_images, up_masks, up_tok, up_lmask,
                                              state=state.unsqueeze(0))
        p_att2d = make_att_2d_masks(p_pad, p_att)
        p_pos = torch.cumsum(p_pad, dim=1) - 1
        _, up_pkv = m.vlm_with_expert.forward(
            attention_mask=p_att2d, position_ids=p_pos, past_key_values=None,
            inputs_embeds=[p_embs, None], use_cache=True)

    # THE TWO PREFIXES ARE LAID OUT DIFFERENTLY AND MUST BE ALIGNED BY MASK, NOT INDEX.
    # Upstream keeps all 48 language slots (6 real + 42 pad) and puts state at row 176;
    # our emitter COMPACTS to images -> real text -> state, so its valid rows are 0..134
    # contiguous. Selecting upstream's rows where pad_mask is True yields images, then the
    # real language tokens, then state -- the same content in the same order as our
    # 0..valid_len. Comparing raw row indices instead would diff image rows against
    # language rows and report garbage.
    keep = p_pad[0].bool()                                          # [177] -> 135 True
    valid = int(ue._last_prefix["valid_len"])
    assert int(keep.sum()) == valid, (
        f"upstream has {int(keep.sum())} valid prefix rows, FPGA reports {valid} -- the "
        f"two layouts disagree on what is real, so no row alignment is meaningful")

    # ---- 3. THE STAGE LADDER ---------------------------------------------------------
    # Read top to bottom and stop at the FIRST stage that falls off: everything below it
    # is scoring a different input, so only the first break localizes anything.
    def row(label, h, r):
        print(f"    {label:<34} {snr_db(h, r):8.2f} dB   cos {cos_sim(h, r):.6f}   "
              f"rms {float(h.float().pow(2).mean().sqrt()):7.4f}/"
              f"{float(r.float().pow(2).mean().sqrt()):7.4f}")

    print("\n=== STAGE LADDER (FPGA vs the checkpoint's own forward pass) ===")

    # A. vision: post-connector tokens, before embed_prefix's sqrt(960) scale.
    print("  A. vision + connector")
    for s in range(slots):
        row(f"connector[slot {s}]",
            ue._last_vision["connector"][s], up_img_emb[s * 64:(s + 1) * 64])
    row("vision tokens [128,960]", vis, up_img_emb)

    # B. the assembled prefix INPUT. Isolates embedding, the sqrt(960) scale, the state
    # projection and the row ordering from everything the transformer does to them.
    print("  B. prefix input (embeddings + scale + layout)")
    row("prefix_in [valid,960]",
        ue._last_prefix["prefix_in"][:valid], p_embs[0][keep])

    # C. per-layer KV cache. This is what the expert cross-attends into, so a break here
    # makes every action wrong regardless of the expert.
    print("  C. prefix KV cache (k, per layer)")
    D, NKV = ue.HEAD_DIM, ue.NUM_KV_HEADS
    PM = ue.PREFILL_MAX_SEQ_LEN
    for li in range(ue.NUM_LAYERS):
        hw_k = torch.stack([
            ue._read_bf16(ue.LAYER0_K_DRAM + li * ue.KV_LAYER_STRIDE
                          + h * ue.KV_HEAD_STRIDE, (PM, D), label=f"k_l{li}h{h}")
            for h in range(NKV)])[:, :valid].reshape(-1, D)
        # transformers 5.x returns a DynamicCache (a list of CacheLayer objects), not a
        # subscriptable tuple; older builds and the reference copies use a dict or a
        # tuple. Handle all three rather than pinning to one transformers version.
        if hasattr(up_pkv, "layers"):
            up_k = up_pkv.layers[li].keys
        elif hasattr(up_pkv, "key_cache"):
            up_k = up_pkv.key_cache[li]
        else:
            e = up_pkv[li]
            up_k = e["key_states"] if isinstance(e, dict) else e[0]
        up_k = up_k[0].float()[:, keep].reshape(-1, D)   # [nkv, seq, D] -> [nkv*valid, D]
        if li in (0, ue.NUM_LAYERS // 2, ue.NUM_LAYERS - 1):
            row(f"KV L{li} k", hw_k, up_k)
        elif li == 1:
            print("       ...")

    # D. actions.
    print("  D. denoise -> actions")
    print(f"\n=== FPGA vs UPSTREAM  [{chunk}, {adim}] ===")
    print(f"  SNR {snr_db(hw, ref):8.2f} dB   cos {cos_sim(hw, ref):.6f}   "
          f"gain {float(hw.norm() / ref.norm()):.4f}x")
    dev = (hw - ref).abs()
    i = int(dev.argmax())
    r, c = divmod(i, adim)
    print(f"  max|dev| {float(dev.max()):.4e} at row {r} dof{c} "
          f"(hw={float(hw[r, c]):+.4f} up={float(ref[r, c]):+.4f})")
    print("  per-dof SNR: " + "  ".join(
        f"dof{d}={snr_db(hw[:, d], ref[:, d]):6.2f}" for d in range(adim)))

    print(f"\n  {'row':>3}  " + "  ".join(f"{'hw dof'+str(d):>9}" for d in range(adim))
          + "   |  " + "  ".join(f"{'up dof'+str(d):>9}" for d in range(adim)))
    for r in list(range(4)) + list(range(chunk - 3, chunk)):
        print(f"  {r:3d}  " + "  ".join(f"{float(hw[r, d]):9.4f}" for d in range(adim))
              + "   |  " + "  ".join(f"{float(ref[r, d]):9.4f}" for d in range(adim)))

    ok = snr_db(hw, ref) >= 20.0 and cos_sim(hw, ref) >= 0.999
    print(f"\n  {'PASS' if ok else 'FAIL'}  (>=20 dB and cos >=0.999 vs the real model)")
    print("  a LOW number here with a passing --snr gate means the emitter and "
          "VeraPulseRef share a wrong assumption, not that the hardware is broken.")


if __name__ == "__main__":
    main()
