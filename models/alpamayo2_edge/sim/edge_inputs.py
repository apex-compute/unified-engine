#!/usr/bin/env python3
"""Raw sensor inputs -> the packed tensors Alpamayo2EdgeRef consumes. No NVIDIA package imports.

Raw:    22 uint8 RGB frames + camera ids + capture times (us), 20 ego poses in the t0 ego
        frame (10 Hz, last pose = t0 = identity), navigation text.
Packed: tokenized_data.{input_ids, attention_mask, mm_token_type_ids, image_grid_thw,
        pixel_values}, traj_data.{ego_history_xyz, ego_history_rot} -- the keys of
        packed.safetensors written by utility/capture_golden.py.

The frame schedule, prompt layout, resize rule, patch layout and history tokenization are
ported from NVIDIA's Apache-2.0 sources (github.com/NVlabs/alpamayo2-edge: sample.py,
chat_template.py, helper.py, load_physical_aiavdataset.py, models/delta_tokenizer.py,
models/utils.py) and from the Cosmos3-Edge image processor / Qwen2-VL smart_resize in
Hugging Face transformers (Apache-2.0). Copyright (c) NVIDIA CORPORATION & AFFILIATES and
The HuggingFace Team for the respective logic.

  # parity against the golden packed inputs (tokenizer + CPU only, no model weights loaded)
  python models/alpamayo2_edge/sim/edge_inputs.py --selfcheck
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

# ----------------------------------------------------------------------------- input contract
# sample.py / load_physical_aiavdataset.py / chat_template.py

CAMERA_NAMES = {  # model camera id -> PhysicalAI-AV sensor name (id 4, rear tele, is unused)
    0: "camera_cross_left_120fov",
    1: "camera_front_wide_120fov",
    2: "camera_cross_right_120fov",
    3: "camera_rear_left_70fov",
    5: "camera_rear_right_70fov",
    6: "camera_front_tele_30fov",
}
CAMERA_DISPLAY_NAMES = {
    0: "Front left camera",
    1: "Front camera",
    2: "Front right camera",
    3: "Rear left camera",
    5: "Rear right camera",
    6: "Front telephoto camera",
}
FRONT_OFFSETS_US = (-2_000_000, -1_000_000, -500_000, -100_000, 0)
OTHER_OFFSETS_US = (-1_000_000, -100_000, 0)
CAMERA_OFFSETS_US = {cam: FRONT_OFFSETS_US if cam in (1, 6) else OTHER_OFFSETS_US for cam in CAMERA_NAMES}
CAMERA_FRAME_COUNTS = {cam: len(offs) for cam, offs in CAMERA_OFFSETS_US.items()}  # 22 frames
CAMERA_PIXEL_BOUNDS = {cam: (163840, 196608) if cam in (1, 6) else (81920, 98304) for cam in CAMERA_NAMES}
NUM_FRAMES = sum(CAMERA_FRAME_COUNTS.values())

HISTORY_STEPS = 20
HISTORY_DT_US = 100_000
HISTORY_OFFSETS_US = tuple(range(-(HISTORY_STEPS - 1) * HISTORY_DT_US, 1, HISTORY_DT_US))  # -1.9 s .. 0
N_WAYPOINTS, WAYPOINT_DT_US = 64, 100_000

SYSTEM_PROMPT = "You are a driving assistant that generates safe and accurate actions."
DRIVING_TASK = "Your task: output the chain-of-thought reasoning of the driving process, then output the future trajectory."
HISTORY_XYZ_MIN, HISTORY_XYZ_MAX, HISTORY_BINS = (-4.0, -4.0, -10.0), (4.0, 4.0, 10.0), 1000


# ----------------------------------------------------------------------------- images

def smart_resize(height, width, factor, min_pixels, max_pixels):
    """Qwen2-VL rule: both sides multiples of `factor`, area inside [min_pixels, max_pixels]."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("aspect ratio above 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def resize_bicubic_uint8(chw, height, width):
    """Antialiased bicubic on a uint8 CHW tensor, as torchvision's tensor resize: float32
    interpolate -> clamp -> round -> uint8."""
    if chw.shape[-2:] == (height, width):
        return chw
    out = F.interpolate(chw[None].float(), size=(height, width), mode="bicubic", align_corners=False, antialias=True)
    return out.clamp_(0, 255).round_().to(torch.uint8)[0]


def patchify(chw, patch, merge):
    """(C, H, W) -> (H/p * W/p, p*p*C): 2x2 merge blocks row-major, the four patches of a block
    adjacent, each patch flattened (row, col, channel). Temporal patch size is 1."""
    c, h, w = chw.shape
    gh, gw = h // patch, w // patch
    x = chw.reshape(c, gh // merge, merge, patch, gw // merge, merge, patch)
    return x.permute(1, 4, 2, 5, 3, 6, 0).reshape(gh * gw, patch * patch * c), gh, gw


def as_chw_uint8(frame):
    """HWC or CHW uint8 RGB (numpy / torch, any device) -> CPU CHW uint8 tensor."""
    t = torch.as_tensor(frame).detach().cpu()
    if t.dtype != torch.uint8 or t.ndim != 3:
        raise ValueError(f"frames must be uint8 with 3 dims, got {t.dtype} {tuple(t.shape)}")
    if t.shape[-1] == 3 and t.shape[0] != 3:
        t = t.permute(2, 0, 1)
    if t.shape[0] != 3:
        raise ValueError(f"frames must be RGB, got shape {tuple(t.shape)}")
    return t.contiguous()


# ----------------------------------------------------------------------------- prompt + history

def canonical_order(camera_ids, timestamps_us):
    """Frame indices sorted by camera id, then capture time (stable)."""
    return sorted(range(len(camera_ids)), key=lambda i: (int(camera_ids[i]), int(timestamps_us[i])))


def time_labels(timestamps_us):
    """Capture time relative to the last frame of the camera-major order."""
    last = int(timestamps_us[-1])
    return [f"<t={(int(t) - last) / 1_000_000:z.1f}s>" for t in timestamps_us]  # z: no "-0.0"


def build_prompt(camera_ids, timestamps_us, image_tokens, nav_text, history_tokens=(HISTORY_STEPS - 1) * 3):
    """The checkpoint chat template applied to NVIDIA's driving conversation, as plain text.

    camera_ids / timestamps_us / image_tokens (merged tokens per image) are in canonical order.
    The history span is left as <|traj_history|> placeholders."""
    user, previous = [], None
    for camera, label, n in zip(camera_ids, time_labels(timestamps_us), image_tokens):
        if camera != previous:
            user.append(CAMERA_DISPLAY_NAMES[int(camera)] + ": ")
        user.append(label + "<|vision_start|>" + "<|image_pad|>" * int(n) + "<|vision_end|>")
        previous = camera
    user += [
        "\n", "<|traj_history_start|>" + "<|traj_history|>" * history_tokens + "<|traj_history_end|>",
        "\n", "Navigation input is: " + nav_text,
        "\n", DRIVING_TASK,
    ]
    return (
        # the checkpoint's chat template opens with a newline
        "\n<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n"
        "<|im_start|>user\n" + "".join(user) + "<|im_end|>\n"
        "<|im_start|>assistant\n<think></think>"
    )


def history_token_ids(xyz, id0):
    """Delta tokenizer (pad_origin_at_beginning=False, no yaw): 20 poses -> 19 deltas x (x, y, z),
    each quantized into 1000 bins over [-4, 4] / [-4, 4] / [-10, 10] m. xyz (20, 3) -> (57,)."""
    xyz = torch.as_tensor(xyz, dtype=torch.float32)
    lo, hi = torch.tensor(HISTORY_XYZ_MIN), torch.tensor(HISTORY_XYZ_MAX)
    d = (xyz[1:] - xyz[:-1] - lo) / (hi - lo)
    return (d * (HISTORY_BINS - 1)).round().long().clamp(0, HISTORY_BINS - 1).reshape(-1) + id0


# ----------------------------------------------------------------------------- builder

def default_model_dir():
    from alpamayo2_edge_cpu_reference import resolve_model_dir  # torch + safetensors only

    return resolve_model_dir(None)


class EdgeInputBuilder:
    """Holds the tokenizer and the checkpoint's processor constants; `pack` does one sample."""

    def __init__(self, model_dir=None):
        from transformers import AutoTokenizer

        self.dir = Path(model_dir).expanduser() if model_dir else default_model_dir()
        cfg = json.loads((self.dir / "config.json").read_text())
        ip = json.loads((self.dir / "processor_config.json").read_text())["image_processor"]
        assert cfg["frame_label"] == "relative_time" and cfg["token_layout"] == "camera_ts" and not cfg["encode_as_video"]
        assert cfg["include_camera_ids"] and cfg["tokens_per_history_traj"] == (HISTORY_STEPS - 1) * 3
        assert cfg["history_vocab_size"] == HISTORY_BINS and ip["temporal_patch_size"] == 1
        self.patch, self.merge = ip["patch_size"], ip["merge_size"]
        scale = 1.0 / ip["rescale_factor"]  # fused uint8 rescale + normalize
        self.mean = torch.tensor(ip["image_mean"], dtype=torch.float32) * scale
        self.std = torch.tensor(ip["image_std"], dtype=torch.float32) * scale
        self.image_token = cfg["vlm_config"]["image_token_id"]
        self.history_pad, self.history_id0 = cfg["traj_ids"]["history_pad"], cfg["traj_ids"]["history_id0"]
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.dir))

    def pixels(self, frame, camera):
        lo, hi = CAMERA_PIXEL_BOUNDS[camera]
        chw = as_chw_uint8(frame)
        h, w = smart_resize(chw.shape[1], chw.shape[2], self.patch * self.merge, lo, hi)
        patches, gh, gw = patchify(resize_bicubic_uint8(chw, h, w), self.patch, self.merge)
        x = patches.float().reshape(-1, self.patch ** 2, 3)
        x.sub_(self.mean).div_(self.std)
        return x.reshape(-1, self.patch ** 2 * 3), gh, gw

    def pack(self, frames, camera_ids, timestamps_us, ego_history_xyz, ego_history_rot, nav_text):
        """frames: 22 HWC (or CHW) uint8 RGB arrays in any order; camera_ids: model camera ids
        (0, 1, 2, 3, 5, 6); timestamps_us: capture times; ego_history_*: (20, 3) / (20, 3, 3) in the
        t0 ego frame, oldest first. Returns the packed dict (batch dims as in packed.safetensors)."""
        camera_ids = [int(c) for c in camera_ids]
        timestamps_us = [int(t) for t in timestamps_us]
        counts = {c: camera_ids.count(c) for c in set(camera_ids)}
        if len(frames) != NUM_FRAMES or counts != CAMERA_FRAME_COUNTS:
            raise ValueError(f"need frames per camera id {CAMERA_FRAME_COUNTS}, got {counts}")
        if not isinstance(nav_text, str) or not nav_text.strip():
            raise ValueError("nav_text must be a nonempty string")
        xyz = torch.as_tensor(np.asarray(ego_history_xyz), dtype=torch.float32).reshape(-1, 3)
        rot = torch.as_tensor(np.asarray(ego_history_rot), dtype=torch.float32).reshape(-1, 3, 3)
        if xyz.shape[0] != HISTORY_STEPS or rot.shape[0] != HISTORY_STEPS or not (torch.isfinite(xyz).all() and torch.isfinite(rot).all()):
            raise ValueError(f"ego history must be {HISTORY_STEPS} finite poses")

        order = canonical_order(camera_ids, timestamps_us)
        cams, stamps = [camera_ids[i] for i in order], [timestamps_us[i] for i in order]
        pixel_values, grids = [], []
        for i, cam in zip(order, cams):
            p, gh, gw = self.pixels(frames[i], cam)
            pixel_values.append(p)
            grids.append([1, gh, gw])
        text = build_prompt(cams, stamps, [gh * gw // self.merge ** 2 for _, gh, gw in grids], nav_text)
        ids = torch.tensor(self.tokenizer(text, add_special_tokens=False)["input_ids"], dtype=torch.int64)
        types = (ids == self.image_token).long()
        pad = ids == self.history_pad
        assert int(pad.sum()) == (HISTORY_STEPS - 1) * 3, "history placeholders were not tokenized one-to-one"
        ids = ids.masked_scatter(pad, history_token_ids(xyz, self.history_id0))
        return {
            "tokenized_data.input_ids": ids[None],
            "tokenized_data.attention_mask": torch.ones_like(ids)[None],
            "tokenized_data.mm_token_type_ids": types[None],
            "tokenized_data.image_grid_thw": torch.tensor(grids, dtype=torch.int64),
            "tokenized_data.pixel_values": torch.cat(pixel_values),
            "traj_data.ego_history_xyz": xyz[None, None].contiguous(),
            "traj_data.ego_history_rot": rot[None, None].contiguous(),
        }


# ----------------------------------------------------------------------------- selfcheck

def synthetic_raw(t0_us=4_000_000, nav_text="Turn right in 30m"):
    """The raw sample of utility/capture_golden.py::synthetic_sample: two photos resized to
    1920x1080 alternating over the 22 frames, a straight 10 m/s history."""
    from PIL import Image

    samples = HERE.parents[1] / "test_samples"
    photos = [np.array(Image.open(samples / f).convert("RGB").resize((1920, 1080))) for f in ("vette.jpg", "people.jpg")]
    frames, cameras, stamps = [], [], []
    for camera, offsets in CAMERA_OFFSETS_US.items():
        for offset in offsets:
            frames.append(photos[len(frames) % 2])
            cameras.append(camera)
            stamps.append(t0_us + offset)
    xyz = np.zeros((HISTORY_STEPS, 3), dtype=np.float32)
    xyz[:, 0] = np.arange(-19.0, 1.0)
    rot = np.tile(np.eye(3, dtype=np.float32), (HISTORY_STEPS, 1, 1))
    return dict(frames=frames, camera_ids=cameras, timestamps_us=stamps, ego_history_xyz=xyz, ego_history_rot=rot, nav_text=nav_text)


def selfcheck(golden_dir, model_dir=None):
    from safetensors.torch import load_file

    golden_dir = Path(golden_dir).expanduser()
    gold = load_file(str(golden_dir / "packed.safetensors"))
    meta = json.loads((golden_dir / "meta.json").read_text())
    builder = EdgeInputBuilder(model_dir)
    ours = builder.pack(**synthetic_raw(meta["t0_us"], meta["nav_text"]))
    ok = True
    for key in sorted(gold):
        g, o = gold[key], ours[key]
        if g.shape != o.shape:
            ok, line = False, f"SHAPE MISMATCH golden {tuple(g.shape)} ours {tuple(o.shape)}"
        elif g.is_floating_point():
            diff = float((g.float() - o.float()).abs().max())
            same = diff <= 1e-6
            ok &= same
            line = f"max|diff| {diff:.3g} over {tuple(g.shape)} ({'bit-exact' if torch.equal(g, o) else 'within 1e-6' if same else 'MISMATCH'})"
        else:
            same = torch.equal(g, o)
            ok &= same
            line = f"{'exact' if same else f'MISMATCH at {int((g != o).sum())} positions'} over {tuple(g.shape)}"
        print(f"{key:<36}{line}")
    prompt = builder.tokenizer.decode(ours["tokenized_data.input_ids"][0], skip_special_tokens=False)
    same = prompt == (golden_dir / "prompt.txt").read_text()
    ok &= same
    print(f"{'decoded prompt vs prompt.txt':<36}{'exact' if same else 'MISMATCH'}")
    print("selfcheck:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--golden", default="~/a2e_golden/synthetic_float32", help="capture_golden.py --synthetic output dir")
    ap.add_argument("--model-dir", help="checkpoint directory (default: HF cache snapshot)")
    args = ap.parse_args()
    if not args.selfcheck:
        ap.error("nothing to do: pass --selfcheck")
    sys.exit(0 if selfcheck(args.golden, args.model_dir) else 1)


if __name__ == "__main__":
    main()
