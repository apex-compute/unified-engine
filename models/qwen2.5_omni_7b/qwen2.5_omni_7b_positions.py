#!/usr/bin/env python3
"""Host-side temporal multimodal RoPE positions for Qwen2.5-Omni.

The accelerator consumes one ``(temporal, height, width)`` position triple per
prompt token.  Keeping this small routine independent of Transformers avoids
instantiating the 7B reference model merely to build integer position IDs.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _as_grids(value, name: str) -> list[tuple[int, int, int]]:
    if value is None:
        return []
    tensor = torch.as_tensor(value, dtype=torch.long)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        raise ValueError(f"{name} must have shape [items, 3], got {tuple(tensor.shape)}")
    grids = [tuple(int(x) for x in row.tolist()) for row in tensor]
    if any(t <= 0 or h <= 0 or w <= 0 for t, h, w in grids):
        raise ValueError(f"{name} entries must be positive, got {grids}")
    return grids


def _vision_positions(
    grid: tuple[int, int, int],
    start: int,
    merge: int,
    temporal_step: float,
) -> torch.Tensor:
    t, h, w = grid
    if h % merge or w % merge:
        raise ValueError(f"vision grid {grid} is not divisible by merge size {merge}")
    mh, mw = h // merge, w // merge
    temporal = torch.floor(torch.arange(t, dtype=torch.float32) * temporal_step).long()
    temporal = temporal.view(t, 1, 1).expand(t, mh, mw)
    height = torch.arange(mh).view(1, mh, 1).expand(t, mh, mw)
    width = torch.arange(mw).view(1, 1, mw).expand(t, mh, mw)
    return torch.stack((temporal, height, width), dim=-1).reshape(-1, 3) + start


def build_multimodal_positions(
    token_ids: Sequence[int] | torch.Tensor,
    *,
    image_grid_thw=None,
    video_grid_thw=None,
    audio_token_lengths: Sequence[int] | torch.Tensor | None = None,
    video_seconds_per_grid: Sequence[float] | torch.Tensor | None = None,
    image_token_id: int = 151655,
    video_token_id: int = 151656,
    audio_token_id: int = 151646,
    spatial_merge_size: int = 2,
    position_ids_per_second: int = 25,
) -> tuple[torch.Tensor, int]:
    """Return ``([seq, 3] positions, rope_delta)`` for one packed prompt.

    Consecutive placeholder runs represent one media item.  Image/video run
    lengths are checked against ``T*H*W/merge**2`` and audio runs against the
    already pooled audio-encoder output length.  Boundary tokens are ordinary
    sequential text positions, matching the upstream Thinker implementation.
    """
    ids = [int(x) for x in torch.as_tensor(token_ids).flatten().tolist()]
    images = _as_grids(image_grid_thw, "image_grid_thw")
    videos = _as_grids(video_grid_thw, "video_grid_thw")
    audio_lengths = ([] if audio_token_lengths is None else
                     [int(x) for x in torch.as_tensor(audio_token_lengths).flatten().tolist()])
    video_seconds = ([] if video_seconds_per_grid is None else
                     [float(x) for x in torch.as_tensor(video_seconds_per_grid).flatten().tolist()])
    if videos and not video_seconds:
        video_seconds = [1.0] * len(videos)
    if len(video_seconds) != len(videos):
        raise ValueError("video_seconds_per_grid must contain one value per video")

    output = torch.empty((len(ids), 3), dtype=torch.long)
    modality_ids = {image_token_id, video_token_id, audio_token_id}
    cursor = 0
    image_idx = video_idx = audio_idx = 0
    i = 0
    while i < len(ids):
        token = ids[i]
        if token not in modality_ids:
            output[i].fill_(cursor)
            cursor += 1
            i += 1
            continue

        end = i + 1
        while end < len(ids) and ids[end] == token:
            end += 1
        run = end - i
        if token == image_token_id:
            if image_idx >= len(images):
                raise ValueError("prompt contains more image runs than image_grid_thw entries")
            grid = images[image_idx]
            values = _vision_positions(
                grid, cursor, spatial_merge_size, float(position_ids_per_second))
            image_idx += 1
        elif token == video_token_id:
            if video_idx >= len(videos):
                raise ValueError("prompt contains more video runs than video_grid_thw entries")
            grid = videos[video_idx]
            values = _vision_positions(
                grid, cursor, spatial_merge_size,
                video_seconds[video_idx] * float(position_ids_per_second))
            video_idx += 1
        else:
            if audio_idx >= len(audio_lengths):
                raise ValueError("prompt contains more audio runs than audio_token_lengths entries")
            count = audio_lengths[audio_idx]
            values = torch.arange(count).view(-1, 1).expand(-1, 3) + cursor
            audio_idx += 1

        if len(values) != run:
            raise ValueError(
                f"placeholder run for token {token} has {run} row(s), but its encoder "
                f"metadata requires {len(values)}")
        output[i:end] = values
        cursor = int(values.max().item()) + 1 if len(values) else cursor
        i = end

    if image_idx != len(images) or video_idx != len(videos) or audio_idx != len(audio_lengths):
        raise ValueError(
            "unused modality metadata: "
            f"images {len(images) - image_idx}, videos {len(videos) - video_idx}, "
            f"audios {len(audio_lengths) - audio_idx}")
    delta = (int(output.max().item()) + 1 - len(ids)) if ids else 0
    return output, delta


__all__ = ["build_multimodal_positions"]
