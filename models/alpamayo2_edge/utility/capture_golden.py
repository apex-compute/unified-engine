#!/usr/bin/env python3
"""Run NVIDIA's Alpamayo2Edge on one PhysicalAI-AV clip; dump its inputs and stage goldens.

Needs the reference env (../setup_ref_env.sh: transformers 5.17.0 + NVlabs/alpamayo2-edge)
and Hugging Face access to the gated nvidia/PhysicalAI-Autonomous-Vehicles dataset.

  ~/a2e_ref_env/bin/python models/alpamayo2_edge/utility/capture_golden.py                    # cuda / bf16
  ~/a2e_ref_env/bin/python models/alpamayo2_edge/utility/capture_golden.py --device cpu --dtype float32

Writes to --out (default ~/a2e_golden/<clip>_<t0>_<dtype>):
  packed.safetensors   model inputs, same keys as NVIDIA's CLI
  golden.safetensors   stage tensors (float32) read by alpamayo2_edge_cpu_reference.py --golden
  meta.json, prompt.txt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


def cpu(t):
    return t.detach().float().cpu().contiguous()


def hidden(out):
    return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]


def logged_future(avdi, clip_id, t0_us, steps=64):
    """Driven poses at t0 + 0.1 .. 6.4 s in the t0 ego frame, or None past the end of the clip."""
    ego = avdi.get_clip_feature(clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=True)
    stamps = t0_us + np.arange(0, steps + 1, dtype=np.int64) * 100_000
    if stamps[-1] > ego.time_range[1]:
        return None
    poses = ego(stamps).pose
    world_xyz, world_rot = np.asarray(poses.translation), poses.rotation
    t0_inverse = world_rot[0].inv()
    xyz = t0_inverse.apply(world_xyz - world_xyz[0])[1:]
    rot = (t0_inverse * world_rot).as_matrix()[1:]
    return torch.from_numpy(xyz).float(), torch.from_numpy(rot).float()


def synthetic_sample(t0_us, nav_text):
    """A sample in the dataset loader's format without the gated dataset.

    Parity only needs both implementations to see the same tensors, not a real driving scene.
    """
    from PIL import Image

    from alpamayo2_edge.sample import CAMERA_FRAME_COUNTS, CAMERA_PIXEL_BOUNDS

    samples = Path(__file__).resolve().parents[2] / "test_samples"
    photos = [
        torch.from_numpy(np.array(Image.open(samples / f).convert("RGB").resize((1920, 1080)))).permute(2, 0, 1)
        for f in ("vette.jpg", "people.jpg")
    ]
    frames, cameras, stamps = [], [], []
    for camera, count in CAMERA_FRAME_COUNTS.items():
        offsets = [-2_000_000, -1_000_000, -500_000, -100_000, 0] if count == 5 else [-1_000_000, -100_000, 0]
        for offset in offsets:
            frames.append(photos[len(frames) % 2])
            cameras.append(camera)
            stamps.append(t0_us + offset)
    xyz = torch.zeros(1, 20, 3)
    xyz[0, :, 0] = torch.arange(-19.0, 1.0)  # 1 m per 0.1 s
    return {
        "image_frames": torch.stack(frames).unsqueeze(1).contiguous(),
        "camera_indices": torch.tensor(cameras, dtype=torch.int64),
        "absolute_timestamps": torch.tensor(stamps, dtype=torch.int64),
        "ego_history_xyz": xyz,
        "ego_history_rot": torch.eye(3).expand(1, 20, 3, 3).contiguous(),
        "nav_text": [nav_text],
        "camera_pixel_bounds": dict(CAMERA_PIXEL_BOUNDS),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip-id", default="ea7bbd31-b7a5-4972-8dbd-7089e6b53de4")
    ap.add_argument("--t0-us", type=int, default=4_000_000)
    ap.add_argument("--nav-text", default="Turn right in 30m")
    ap.add_argument("--model-id", default="nvidia/Alpamayo2-Edge")
    ap.add_argument("--revision", default="f8951d095de3589b3face1dec3f4394c8ecedb38")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--out")
    ap.add_argument("--synthetic", action="store_true",
                    help="no dataset: sample photos as the 22 camera frames + a straight 10 m/s history")
    args = ap.parse_args()

    import transformers
    from alpamayo2_edge.checkpoint import load_model
    from alpamayo2_edge.helper import prepare_model_inputs, to_device
    from alpamayo2_edge.models.utils import fuse_traj_tokens

    name = "synthetic" if args.synthetic else f"{args.clip_id}_{args.t0_us}"
    out_dir = Path(args.out or f"~/a2e_golden/{name}_{args.dtype}").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        data, future = synthetic_sample(args.t0_us, args.nav_text), None
    else:
        import physical_ai_av
        from alpamayo2_edge.load_physical_aiavdataset import load_physical_aiavdataset

        avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
        data = load_physical_aiavdataset(args.clip_id, args.t0_us, nav_text=args.nav_text, avdi=avdi)
        future = logged_future(avdi, args.clip_id, args.t0_us)
    model, _ = load_model(args.model_id, device=args.device, dtype=args.dtype,
                          attn_implementation=args.attn, revision=args.revision)

    prepared = prepare_model_inputs(data, model.config, model.tokenizer)
    history = {key: prepared[key] for key in ("ego_history_xyz", "ego_history_rot")}
    packed = {f"tokenized_data.{k}": v.cpu().contiguous() for k, v in prepared["tokenized_data"].items()}
    packed["tokenized_data.input_ids"] = fuse_traj_tokens(
        model.history_traj_tokenizer, model.future_traj_tokenizer,
        prepared["tokenized_data"]["input_ids"], history, model.config.traj_ids,
    ).cpu().contiguous()
    packed.update({f"traj_data.{k}": v.cpu().contiguous() for k, v in history.items()})
    save_file(packed, str(out_dir / "packed.safetensors"))
    (out_dir / "prompt.txt").write_text(model.tokenizer.decode(packed["tokenized_data.input_ids"][0], skip_special_tokens=False))

    G, meta = {}, {}
    L = packed["tokenized_data.input_ids"].shape[1]  # the prefill covers the first L - 1 tokens
    vlm, expert = model.vlm.model, model.expert
    lm, space = vlm.language_model, model.expert.action_space
    last = len(lm.layers) - 1

    def on(module, fn):
        module.register_forward_hook(fn, with_kwargs=True)

    on(vlm.visual.embeddings, lambda m, a, k, o: G.__setitem__("vision_embed", cpu(o)))
    on(vlm.visual, lambda m, a, k, o: G.__setitem__("vision_out", cpu(hidden(o))))
    on(vlm.projector, lambda m, a, k, o: G.__setitem__("image_embeds", cpu(o)))

    def lm_hook(m, a, k, o):
        h = hidden(o)
        if "prefill_hidden" not in G and h.shape[1] == L - 1:
            G["prefill_hidden"] = cpu(h[0])
            G["prefill_pos"] = k["position_ids"][:, 0].detach().cpu().contiguous()

    on(lm, lm_hook)

    def layer_hook(index):
        def fn(m, a, k, o):
            h = o[0] if isinstance(o, tuple) else o
            if f"prefill_layer{index}" not in G and h.shape[1] == L - 1:
                G[f"prefill_layer{index}"] = cpu(h[0])
        return fn

    for index in (0, last):
        on(lm.layers[index], layer_hook(index))

    top_ids, top_vals = [], []

    def head_hook(m, a, k, o):  # one call per decode step, logits before any masking
        logits = o[0, -1].detach().float()
        if not top_ids:
            G["step_logits0"] = logits.cpu().contiguous()
        vals, idx = logits.topk(5)
        top_ids.append(idx.cpu())
        top_vals.append(vals.cpu())

    on(model.vlm.lm_head, head_hook)

    xs, ts, emb, hid, vs = [], [], [], [], []

    def in_proj_hook(m, a, k, o):  # hooks must return None, or they replace the module output
        xs.append(cpu(a[0][0]))
        ts.append(cpu(a[1].reshape(-1)[:1]))
        emb.append(cpu(o[0]))

    on(expert.action_in_proj, in_proj_hook)
    on(expert.expert, lambda m, a, k, o: hid.append(cpu(hidden(o)[0])))
    on(expert.action_out_proj, lambda m, a, k, o: vs.append(cpu(o[0])))

    sample_actions = model._sample_actions

    def traced_sample_actions(cache, prefix_length, position_ids, *rest):
        meta["prefix_length"] = int(prefix_length)
        G["expert_pos"] = position_ids[:, 0].detach().cpu().contiguous()
        for index in (0, last):
            layer = cache.layers[index]
            G[f"kv_k{index}"] = cpu(layer.keys[0, :, :prefix_length])
            G[f"kv_v{index}"] = cpu(layer.values[0, :, :prefix_length])
        return sample_actions(cache, prefix_length, position_ids, *rest)

    model._sample_actions = traced_sample_actions

    action_to_traj = space.action_to_traj

    def traced_action_to_traj(action, *a, **k):
        G["action"] = cpu(action[0])
        return action_to_traj(action, *a, **k)

    space.action_to_traj = traced_action_to_traj

    torch.manual_seed(args.seed)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    device_type = torch.device(args.device).type
    with torch.inference_mode(), torch.autocast(device_type=device_type, dtype=model.dtype, enabled=model.dtype != torch.float32):
        xyz, rot, _, extra = model.sample_trajectories_from_data(
            to_device(prepared, device=args.device),
            num_traj_sets=1, num_traj_samples=1, top_p=0.98, top_k=None, temperature=0.6,
            max_generation_length=args.max_new,
            diffusion_kwargs={"inference_step": args.steps, "generator": generator},
            return_extra=True,
        )

    generated = extra["generated_ids"][0]
    generated = generated[generated >= 0].cpu().contiguous()  # reasoning tokens, anchor included
    G.update(
        generated_ids=generated,
        step_top5_ids=torch.stack(top_ids), step_top5_vals=torch.stack(top_vals),
        x_steps=torch.stack(xs), t_steps=torch.cat(ts),
        expert_in=torch.stack(emb), expert_hidden=torch.stack(hid), v_steps=torch.stack(vs),
        pred_xyz=cpu(xyz[0, 0, 0]), pred_rot=cpu(rot[0, 0, 0]),
        # non-persistent buffers, built under the load dtype: record what the model really used
        freqs=cpu(expert.action_in_proj.timestep_fourier_encoder.freqs.reshape(-1)),
        act_stats=torch.stack([cpu(getattr(space, name)) for name in ("accel_mean", "accel_std", "curvature_mean", "curvature_std")]),
    )
    if future is not None:
        G["gt_xyz"], G["gt_rot"] = future
    save_file(G, str(out_dir / "golden.safetensors"))

    meta.update(
        clip_id="synthetic" if args.synthetic else args.clip_id, t0_us=args.t0_us, nav_text=args.nav_text,
        device=args.device, dtype=args.dtype, attn=args.attn, seed=args.seed, steps=args.steps,
        prompt_tokens=L, generated_tokens=int(generated.numel()),
        rope_delta=int(G["expert_pos"][0, 0]) - meta["prefix_length"],
        cot=str(np.asarray(extra["cot"]).reshape(-1)[0]),
        torch=torch.__version__, transformers=transformers.__version__,
        tensors={name: list(t.shape) for name, t in G.items()},
    )
    if future is not None:
        err = (G["pred_xyz"][:50, :2] - G["gt_xyz"][:50, :2]).norm(dim=-1).mean()
        meta["ade_5s_m"] = float(err)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "tensors"}, indent=2))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
