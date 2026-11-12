#!/usr/bin/env python3
"""AlpaSim driver plugin for Alpamayo 2 Edge (entry point group `alpasim.models`).

Written against github.com/NVlabs/alpasim main @ affc2eab (2026-08-18), read 2026-11-11:
  src/driver/src/alpasim_driver/models/base.py          BaseTrajectoryModel, PredictionInput, ModelPrediction
  src/driver/src/alpasim_driver/models/alpamayo2_model.py, alpamayo_base.py   (t0, history, route handling mirrored)
  src/driver/src/alpasim_driver/models/nav_text.py      route_to_nav_text (ported below)
  src/driver/src/alpasim_driver/{main,frame_cache,schema}.py                  how the servicer feeds a model
Ported logic is Apache-2.0, Copyright (c) 2025-2026 NVIDIA Corporation.

What the servicer hands predict() (verified in that source):
  camera_images     {logical camera id: [(timestamp_us = frame_end_us, HWC image)]}, oldest first,
                    exactly `context_length` frames per camera, each `subsample_factor` renders apart.
                    The per-camera cache holds context_length * subsample_factor frames and the driver
                    returns an empty trajectory until every cache is full.
  ego_pose_history  every pose of the session (never pruned), local -> rig, .timestamp_us /
                    .pose.vec.{x,y,z} / .pose.quat.{x,y,z,w}
  route             .waypoints[i].{x,y,z} in the rig frame at .timestamp_us (NaN rows = padding), or None

Edge needs frames back to t0 - 2.0 s, so with 10 Hz cameras the driver config must set
inference.context_length: 21 and inference.subsample_factor: 1 (any cadence works as long as
the buffer covers 2.0 s: frames are picked by timestamp, not by index). t0 is the newest ego
pose, as in AlpaSim's Alpamayo 2 driver.

The frame selection, history, route and nav-text logic is plain numpy and runs without
AlpaSim:   python models/alpamayo2_edge/sim/alpasim_edge_driver.py --selfcheck
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from edge_inputs import (  # noqa: E402
    CAMERA_NAMES, CAMERA_OFFSETS_US, HISTORY_OFFSETS_US, N_WAYPOINTS, NUM_FRAMES, WAYPOINT_DT_US,
)

try:
    from alpasim_driver.models.base import BaseTrajectoryModel, ModelInputValidationError, ModelPrediction

    HAVE_ALPASIM = True
except ImportError:  # offline stand-ins with the same fields, enough for --selfcheck
    HAVE_ALPASIM = False

    class ModelInputValidationError(ValueError):
        pass

    @dataclass
    class ModelPrediction:
        candidate_positions: np.ndarray
        candidate_rotations: np.ndarray
        selected_index: int = 0
        reasoning_text: str | None = None
        selected_plan: object = None
        model_t0_us: int | None = None
        pose_local_to_rig_t0: object = None
        waypoint_timestamps_us: np.ndarray | None = None

    class BaseTrajectoryModel:
        pass

CAMERA_IDS = {name: cam for cam, name in CAMERA_NAMES.items()}  # AlpaSim logical id -> model camera id
DEFAULT_CONTEXT_LENGTH = 21  # 10 Hz renders covering t0 - 2.0 s .. t0
OUTPUT_FREQUENCY_HZ = 10
DEFAULT_NAV_TEXT = "Continue straight"  # Edge has no unconditioned mode: nav text must be nonempty
FRAME_EARLY_SLACK_US = 5_000  # a frame this much after its slot still counts as "at" the slot
FRAME_MAX_LAG_US = 60_000  # a frame may be at most this much older than its slot


# ----------------------------------------------------------------------------- frames

def select_frames(camera_frames, t0_us, max_lag_us=FRAME_MAX_LAG_US):
    """Pick Edge's 22-frame schedule out of the buffered streams.

    camera_frames: {logical id: [(timestamp_us, image), ...]}. For every slot t0 + offset the
    newest frame at or before the slot is taken (NVIDIA's dataset loader does the same).
    Returns (frames, model camera ids, timestamps) in camera-major, oldest-first order."""
    missing = set(CAMERA_IDS) - set(camera_frames)
    if missing:
        raise ModelInputValidationError(f"Alpamayo 2 Edge needs cameras {sorted(CAMERA_IDS)}, missing {sorted(missing)}")
    frames, cameras, stamps = [], [], []
    for cam, name in CAMERA_NAMES.items():
        entries = sorted(((int(ts), image) for ts, image in camera_frames[name]), key=lambda e: e[0])
        times = np.array([ts for ts, _ in entries], dtype=np.int64)
        for offset in CAMERA_OFFSETS_US[cam]:
            slot = int(t0_us) + offset
            index = int(np.searchsorted(times, slot + FRAME_EARLY_SLACK_US, side="right")) - 1
            if index < 0 or slot - times[index] > max_lag_us:
                have = f"{(times[0] - t0_us) / 1e6:+.2f}..{(times[-1] - t0_us) / 1e6:+.2f} s" if len(times) else "none"
                raise ModelInputValidationError(
                    f"{name}: no frame for t0{offset / 1e6:+.1f} s (buffered: {have}, {len(times)} frames)")
            frames.append(entries[index][1])
            cameras.append(cam)
            stamps.append(int(times[index]))
    assert len(frames) == NUM_FRAMES
    return frames, cameras, stamps


# ----------------------------------------------------------------------------- poses

def quat_to_matrix(q_xyzw):
    q = np.asarray(q_xyzw, dtype=np.float64)
    x, y, z, w = (q / np.linalg.norm(q, axis=-1, keepdims=True)).T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def pose_arrays(poses):
    """PoseAtTime-like objects -> time-sorted (timestamps int64, xyz (N, 3), quat xyzw (N, 4))."""
    poses = sorted(poses, key=lambda p: int(p.timestamp_us))
    ts = np.array([int(p.timestamp_us) for p in poses], dtype=np.int64)
    xyz = np.array([[p.pose.vec.x, p.pose.vec.y, p.pose.vec.z] for p in poses], dtype=np.float64).reshape(-1, 3)
    quat = np.array([[p.pose.quat.x, p.pose.quat.y, p.pose.quat.z, p.pose.quat.w] for p in poses], dtype=np.float64).reshape(-1, 4)
    return ts, xyz, quat


def interpolate_poses(ts, xyz, quat, query_us):
    """Linear in position and in sign-aligned quaternion (then renormalized), as AlpaSim's
    build_ego_history. Raises if a query lies outside the recorded span."""
    query_us = np.atleast_1d(np.asarray(query_us, dtype=np.int64))
    if len(ts) < 2 or query_us.min() < ts[0] or query_us.max() > ts[-1]:
        span = f"{ts[0]}..{ts[-1]}" if len(ts) else "empty"
        raise ModelInputValidationError(f"ego pose history {span} us does not cover {query_us.min()}..{query_us.max()} us")
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    flip = np.cumprod(np.where((quat[1:] * quat[:-1]).sum(1) < 0, -1.0, 1.0))
    quat = np.concatenate([quat[:1], quat[1:] * flip[:, None]])
    rel, q = (ts - ts[0]).astype(np.float64), (query_us - ts[0]).astype(np.float64)
    p = np.stack([np.interp(q, rel, xyz[:, k]) for k in range(3)], -1)
    r = np.stack([np.interp(q, rel, quat[:, k]) for k in range(4)], -1)
    return p, r / np.linalg.norm(r, axis=1, keepdims=True)


def build_history(ts, xyz, quat, t0_us):
    """20 poses at t0 - 1.9 s .. t0 (10 Hz) in the rig frame at t0: xyz_i = R0^T (p_i - p0),
    rot_i = R0^T R_i -- the frame of NVIDIA's dataset loader. Also returns the local -> rig(t0)
    pose (xyz, quat xyzw) that maps the model's waypoints back."""
    p, q = interpolate_poses(ts, xyz, quat, int(t0_us) + np.array(HISTORY_OFFSETS_US, dtype=np.int64))
    rot = quat_to_matrix(q)
    r0_inv = rot[-1].T
    return ((p - p[-1]) @ r0_inv.T).astype(np.float32), (r0_inv @ rot).astype(np.float32), (p[-1], q[-1])


def route_in_rig_at_t0(waypoints, route_timestamp_us, ts, xyz, quat, pose_t0):
    """Route waypoints (N, 3), given in the rig frame at the route's own timestamp, re-expressed
    in the rig frame at t0. NaN rows stay NaN."""
    waypoints = np.asarray(waypoints, dtype=np.float64).reshape(-1, 3)
    p_r, q_r = interpolate_poses(ts, xyz, quat, int(np.clip(route_timestamp_us, ts[0], ts[-1])))
    local = waypoints @ quat_to_matrix(q_r[0]).T + p_r[0]
    return (local - pose_t0[0]) @ quat_to_matrix(pose_t0[1])


# ----------------------------------------------------------------------------- nav text

def route_to_nav_text(route_in_rig, *, min_lookahead_m=5.0, turn_angle_deg=30.0, distance_lookahead_m=40.0):
    """Port of alpasim_driver.models.nav_text.route_to_nav_text: "Turn left in {N}m",
    "Turn right in {N}m", "Continue straight", or None when the route has no usable waypoint.
    A turn is the first route segment ending 5..40 m away whose heading differs from the road
    heading near the ego by more than 30 degrees."""
    route_in_rig = np.asarray(route_in_rig, dtype=np.float64)
    if route_in_rig.size == 0:
        return None
    xy = route_in_rig[:, :2]
    points = xy[~np.isnan(xy).any(axis=1)]
    if len(points) == 0:
        return None
    if len(points) >= 2:  # resampled routes can repeat their last waypoint as padding
        keep = np.ones(len(points), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-3
        points = points[keep]
    if len(points) < 3:
        return "Continue straight"
    distances = np.linalg.norm(points, axis=1)
    near = np.flatnonzero(distances <= min_lookahead_m)
    start, end = (near[0], near[-1]) if len(near) >= 2 else (0, 1)
    base = points[end] - points[start]
    if np.linalg.norm(base) < 1e-3:
        base = points[1] - points[0]
    initial_heading = np.arctan2(base[1], base[0])
    segments = np.diff(points, axis=0)
    headings = np.arctan2(segments[:, 1], segments[:, 0])
    deviations = np.remainder(headings - initial_heading + np.pi, 2 * np.pi) - np.pi
    endpoint_distances = distances[1:]
    turning = (
        (endpoint_distances >= min_lookahead_m)
        & (endpoint_distances <= distance_lookahead_m)
        & (np.abs(deviations) > np.radians(turn_angle_deg))
    )
    if not turning.any():
        return "Continue straight"
    index = int(np.argmax(turning))
    direction = "left" if deviations[index] > 0 else "right"
    return f"Turn {direction} in {max(1, round(endpoint_distances[index]))}m"


def nav_text_from_route(route, ts, xyz, quat, pose_t0, default=DEFAULT_NAV_TEXT):
    if route is None or len(route.waypoints) == 0:
        return default
    waypoints = np.array([[w.x, w.y, w.z] for w in route.waypoints], dtype=np.float64)
    return route_to_nav_text(route_in_rig_at_t0(waypoints, int(route.timestamp_us), ts, xyz, quat, pose_t0)) or default


# ----------------------------------------------------------------------------- plugin

def _make_policy(model_dir, device):
    from edge_policy import DeviceBackend, EdgePolicy, HostReferenceBackend

    kind = os.environ.get("A2E_BACKEND", "host")  # ModelConfig is a closed schema, so the switch is an env var
    if kind == "host":
        return EdgePolicy(HostReferenceBackend(model_dir, device), model_dir)
    if kind == "device":
        return EdgePolicy(DeviceBackend(), model_dir)
    raise ValueError(f"A2E_BACKEND must be host or device, got {kind!r}")


class Alpamayo2EdgeModel(BaseTrajectoryModel):
    """Alpamayo 2 Edge: 6 cameras / 22 frames, 20-pose history, nav text -> one 6.4 s plan at 10 Hz."""

    MIN_FRAME_HW = None

    def __init__(self, policy, camera_ids, context_length=None):
        if set(camera_ids) != set(CAMERA_IDS):
            raise ValueError(f"Alpamayo 2 Edge expects cameras {sorted(CAMERA_IDS)}, got {sorted(camera_ids)}")
        self._policy = policy
        self._camera_ids = list(camera_ids)
        self._context_length = int(context_length or DEFAULT_CONTEXT_LENGTH)
        if self._context_length < 1:
            raise ValueError("context_length must be >= 1")

    @classmethod
    def from_config(cls, model_cfg, device, camera_ids, context_length, output_frequency_hz):
        if output_frequency_hz != OUTPUT_FREQUENCY_HZ:
            raise ValueError(f"Alpamayo 2 Edge plans at {OUTPUT_FREQUENCY_HZ} Hz, got output_frequency_hz={output_frequency_hz}")
        if getattr(model_cfg, "num_trajectory_samples", 1) != 1:
            raise ValueError("Alpamayo 2 Edge driver returns one trajectory: set num_trajectory_samples: 1")
        checkpoint = getattr(model_cfg, "checkpoint_path", None)
        model_dir = checkpoint if checkpoint and Path(str(checkpoint)).expanduser().is_dir() else None  # else: HF cache
        return cls(_make_policy(model_dir, device), camera_ids, context_length)

    def _encode_command(self, command):
        return None  # navigation reaches the model as text derived from the route

    @property
    def camera_ids(self):
        return self._camera_ids

    @property
    def context_length(self):
        return self._context_length

    @property
    def output_frequency_hz(self):
        return OUTPUT_FREQUENCY_HZ

    def predict(self, prediction_input):
        poses = prediction_input.ego_pose_history
        if not poses or len(poses) < 2:
            raise ModelInputValidationError("Alpamayo 2 Edge needs an ego pose history")
        ts, xyz, quat = pose_arrays(poses)
        t0_us = int(ts[-1])
        frames, cameras, stamps = select_frames(prediction_input.camera_images, t0_us)
        hist_xyz, hist_rot, pose_t0 = build_history(ts, xyz, quat, t0_us)
        nav_text = nav_text_from_route(prediction_input.route, ts, xyz, quat, pose_t0)
        way_xyz, way_rot, text = self._policy.predict(
            frames, cameras, stamps, hist_xyz, hist_rot, nav_text, seed=int(prediction_input.inference_seed))
        return ModelPrediction(
            candidate_positions=np.asarray(way_xyz, dtype=np.float64)[None],
            candidate_rotations=np.asarray(way_rot, dtype=np.float64)[None],
            selected_index=0,
            reasoning_text=f"[{nav_text}] {text}" if text else None,
            # all three must be set together: the servicer then anchors the plan at our t0
            model_t0_us=t0_us,
            pose_local_to_rig_t0=_alpasim_pose(pose_t0),
            waypoint_timestamps_us=t0_us + np.arange(1, N_WAYPOINTS + 1, dtype=np.uint64) * np.uint64(WAYPOINT_DT_US),
        )


def _alpasim_pose(pose_t0):
    xyz, quat_xyzw = pose_t0[0].astype(np.float32), pose_t0[1].astype(np.float32)
    if not HAVE_ALPASIM:
        return (xyz, quat_xyzw)
    from alpasim_utils.geometry import Pose

    return Pose(xyz, quat_xyzw)


# ----------------------------------------------------------------------------- selfcheck

def _pose(ts, xyz, yaw):
    return SimpleNamespace(timestamp_us=int(ts), pose=SimpleNamespace(
        vec=SimpleNamespace(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])),
        quat=SimpleNamespace(x=0.0, y=0.0, z=float(np.sin(yaw / 2)), w=float(np.cos(yaw / 2)))))


def _route(points, ts):
    return SimpleNamespace(timestamp_us=int(ts), waypoints=[SimpleNamespace(x=float(a), y=float(b), z=0.0) for a, b in points])


def selfcheck():
    t0, speed, yaw_rate, yaw_start = 7_300_000, 8.0, 0.2, 1.1  # a left arc in an arbitrary local frame
    # poses every 50 ms over the last 3 s, ending at t0 (sim-style: local -> rig)
    times = t0 + np.arange(-60, 1) * 50_000
    def arc(t_us):
        s = (np.asarray(t_us) - times[0]) * 1e-6
        yaw = yaw_start + yaw_rate * s
        r = speed / yaw_rate
        return np.stack([5 + r * (np.sin(yaw) - np.sin(yaw_start)), -3 - r * (np.cos(yaw) - np.cos(yaw_start)), np.full_like(s, 0.4)], -1), yaw
    pts, yaws = arc(times)
    poses = [_pose(t, p, y) for t, p, y in zip(times, pts, yaws)][::-1]  # unsorted on purpose
    ts, xyz, quat = pose_arrays(poses)

    # 1. history: last pose is the origin, earlier ones are the arc seen from the t0 rig
    hx, hr, pose_t0 = build_history(ts, xyz, quat, t0)
    assert hx.shape == (20, 3) and hr.shape == (20, 3, 3)
    assert np.abs(hx[-1]).max() == 0 and np.abs(hr[-1] - np.eye(3)).max() < 1e-6
    s = np.array(HISTORY_OFFSETS_US) * 1e-6
    r = speed / yaw_rate
    expect = np.stack([r * np.sin(yaw_rate * s), r * (1 - np.cos(yaw_rate * s)), 0 * s], -1)
    err_xyz = float(np.abs(hx - expect).max())
    err_yaw = float(np.abs(np.arctan2(hr[:, 1, 0], hr[:, 0, 0]) - yaw_rate * s).max())
    assert err_xyz < 2e-3 and err_yaw < 1e-4, (err_xyz, err_yaw)
    try:
        build_history(ts[-10:], xyz[-10:], quat[-10:], t0)  # only 0.45 s of poses
        raise AssertionError("short history was accepted")
    except ModelInputValidationError:
        pass
    print(f"history: 20 poses, max|xyz err| {err_xyz:.2e} m, max|yaw err| {err_yaw:.2e} rad vs analytic arc; short history rejected")

    # 2. frames: 10 Hz streams of 21 frames (context_length 21); pixel [0, 0, 0] encodes the frame index
    def stream(n=21, step=100_000, jitter=0):
        out = []
        for i in range(n):
            img = np.zeros((4, 6, 3), dtype=np.uint8)
            img[0, 0, 0] = i
            out.append((t0 - (n - 1 - i) * step + jitter, img))
        return out
    cams = {name: stream() for name in CAMERA_IDS}
    frames, cam_ids, stamps = select_frames(cams, t0)
    want_ids = [c for c, offs in CAMERA_OFFSETS_US.items() for _ in offs]
    want_ts = [t0 + o for c, offs in CAMERA_OFFSETS_US.items() for o in offs]
    assert cam_ids == want_ids and stamps == want_ts and len(frames) == 22
    assert [int(f[0, 0, 0]) for f in frames[3:8]] == [0, 10, 15, 19, 20]  # front wide: -2, -1, -0.5, -0.1, 0 s
    late = dict(cams, camera_front_tele_30fov=stream(jitter=-30_000))  # frames 30 ms before each slot
    assert select_frames(late, t0)[2][-1] == t0 - 30_000
    for bad, why in ((dict(cams, camera_front_wide_120fov=stream(n=20)), "1.9 s buffer"),
                     ({k: v for k, v in cams.items() if k != "camera_rear_left_70fov"}, "missing camera"),
                     (dict(cams, camera_cross_left_120fov=stream(n=11, step=200_000)), "5 Hz stream has no -0.1 s frame")):
        try:
            select_frames(bad, t0)
            raise AssertionError(f"accepted: {why}")
        except ModelInputValidationError:
            pass
    print("frames: 22 picked at the exact slots from 21-frame 10 Hz buffers; 30 ms-late stream ok; "
          "1.9 s buffer / missing camera / 5 Hz stream rejected")

    # 3. nav text
    straight = [(x, 0.0) for x in range(0, 60, 5)]
    left = [(x, 0.0) for x in range(0, 25, 5)] + [(20.0, y) for y in range(5, 40, 5)]
    right = [(x, 0.0) for x in range(0, 15, 5)] + [(10.0, -y) for y in range(5, 40, 5)]
    far = [(x, 0.0) for x in range(0, 65, 5)] + [(60.0, y) for y in range(5, 40, 5)]
    texts = [route_to_nav_text(np.array([(a, b, 0.0) for a, b in pts_])) for pts_ in (straight, left, right, far)]
    assert texts == ["Continue straight", "Turn left in 21m", "Turn right in 11m", "Continue straight"], texts
    assert route_to_nav_text(np.full((4, 3), np.nan)) is None and route_to_nav_text(np.zeros((0, 3))) is None
    # the same left-turn route submitted 1 s earlier, in that moment's rig frame, must read the same from t0
    t_r = t0 - 1_000_000
    p_r, q_r = interpolate_poses(ts, xyz, quat, t_r)
    in_local = np.array([(a, b, 0.4 - pose_t0[0][2]) for a, b in left]) @ quat_to_matrix(pose_t0[1]).T + pose_t0[0]
    in_old_rig = (in_local - p_r[0]) @ quat_to_matrix(q_r[0])
    old = SimpleNamespace(timestamp_us=t_r, waypoints=[SimpleNamespace(x=a, y=b, z=c) for a, b, c in in_old_rig])
    back = route_in_rig_at_t0(in_old_rig, t_r, ts, xyz, quat, pose_t0)
    err_route = float(np.abs(back[:, :2] - np.array(left)).max())
    assert err_route < 1e-6 and nav_text_from_route(old, ts, xyz, quat, pose_t0) == "Turn left in 21m"
    assert nav_text_from_route(None, ts, xyz, quat, pose_t0) == DEFAULT_NAV_TEXT
    print(f"nav text: {texts[:3]}, turn beyond 40 m -> {texts[3]!r}; stale-frame route round trip err {err_route:.1e} m; no route -> {DEFAULT_NAV_TEXT!r}")

    # 4. predict() with a fake policy: plumbing and the ModelPrediction contract
    seen = {}
    class FakePolicy:
        def predict(self, frames, camera_ids, timestamps, hx, hr, nav_text, seed=0):
            seen.update(n=len(frames), cams=list(camera_ids), nav=nav_text, seed=seed, hist=hx.shape)
            xyz = np.zeros((N_WAYPOINTS, 3), dtype=np.float32)
            xyz[:, 0] = speed * 0.1 * np.arange(1, N_WAYPOINTS + 1)
            return xyz, np.tile(np.eye(3, dtype=np.float32), (N_WAYPOINTS, 1, 1)), "Keep lane."
    model = Alpamayo2EdgeModel(FakePolicy(), list(CAMERA_IDS))
    inp = SimpleNamespace(camera_images=cams, ego_pose_history=poses, route=_route(left, t0), inference_seed=17,
                          command=None, speed=speed, acceleration=0.0, previous_plan=None)
    out = model.predict(inp)
    assert seen == dict(n=22, cams=want_ids, nav="Turn left in 21m", seed=17, hist=(20, 3)), seen
    assert out.candidate_positions.shape == (1, 64, 3) and out.candidate_rotations.shape == (1, 64, 3, 3)
    assert out.model_t0_us == t0 and out.waypoint_timestamps_us[0] == t0 + 100_000 and out.waypoint_timestamps_us[-1] == t0 + 6_400_000
    assert out.waypoint_timestamps_us.dtype == np.uint64 and out.selected_index == 0
    assert model.context_length == 21 and model.output_frequency_hz == 10 and model._encode_command(None) is None
    try:
        Alpamayo2EdgeModel(FakePolicy(), list(CAMERA_IDS)[:4])
        raise AssertionError("4-camera config was accepted")
    except ValueError:
        pass
    print(f"predict: ModelPrediction (1, 64, 3) / (1, 64, 3, 3), t0 {out.model_t0_us}, waypoints t0+0.1..6.4 s, "
          f"reasoning {out.reasoning_text!r}; 4-camera config rejected")
    print(f"selfcheck: PASS (AlpaSim {'importable: real base classes' if HAVE_ALPASIM else 'not installed: stand-in base classes'})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selfcheck", action="store_true")
    if not ap.parse_args().selfcheck:
        ap.error("nothing to do: pass --selfcheck")
    selfcheck()
