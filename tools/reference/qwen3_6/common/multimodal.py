"""Qwen3.6 multimodal prompt metadata and visual-embedding alignment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .vision_ops import VISION_SPATIAL_MERGE


@dataclass(frozen=True, slots=True)
class MultimodalBatch:
    """One unpadded processor output with target-specific MRoPE metadata."""

    input_ids: torch.Tensor
    mm_token_type_ids: torch.Tensor
    position_ids: torch.Tensor
    rope_delta: int
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    pixel_values_videos: torch.Tensor | None
    video_grid_thw: torch.Tensor | None

    @property
    def prompt_length(self) -> int:
        return int(self.input_ids.numel())

    @property
    def image_tokens(self) -> int:
        return int((self.mm_token_type_ids == 1).sum().item())

    @property
    def video_tokens(self) -> int:
        return int((self.mm_token_type_ids == 2).sum().item())

    @property
    def has_vision(self) -> bool:
        return bool(self.image_tokens or self.video_tokens)

    def scatter_visual_embeddings_(
        self,
        token_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor | None,
        video_embeddings: torch.Tensor | None,
        *,
        offset: int = 0,
    ) -> torch.Tensor:
        """Replace media-placeholder rows in one prompt slice, preserving item order.

        ``offset`` addresses the slice in the complete processed prompt.  That makes
        the same operation usable for chunked prefill and one-token MTP lookahead.
        The supplied tensor is updated in place and returned.
        """

        if token_embeddings.ndim != 2:
            raise ValueError("token embeddings must have shape [tokens, hidden]")
        rows = token_embeddings.shape[0]
        if offset < 0 or offset + rows > self.prompt_length:
            raise ValueError("embedding slice is outside the processed prompt")

        image_rows = 0 if image_embeddings is None else image_embeddings.shape[0]
        video_rows = 0 if video_embeddings is None else video_embeddings.shape[0]
        if image_rows != self.image_tokens:
            raise ValueError("image placeholder/embedding count mismatch")
        if video_rows != self.video_tokens:
            raise ValueError("video placeholder/embedding count mismatch")

        all_types = self.mm_token_type_ids.flatten()
        slice_types = all_types[offset : offset + rows]
        prefix_types = all_types[:offset]
        image_begin = int((prefix_types == 1).sum().item())
        video_begin = int((prefix_types == 2).sum().item())
        image_count = int((slice_types == 1).sum().item())
        video_count = int((slice_types == 2).sum().item())
        device_types = slice_types.to(device=token_embeddings.device)

        if image_count:
            assert image_embeddings is not None
            token_embeddings[device_types == 1] = image_embeddings[
                image_begin : image_begin + image_count
            ].to(device=token_embeddings.device, dtype=token_embeddings.dtype)
        if video_count:
            assert video_embeddings is not None
            token_embeddings[device_types == 2] = video_embeddings[
                video_begin : video_begin + video_count
            ].to(device=token_embeddings.device, dtype=token_embeddings.dtype)
        return token_embeddings


def load_messages(path: str | Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, Mapping) and "messages" in value:
        value = value["messages"]
    if not isinstance(value, list) or not value or not all(
        isinstance(item, dict) for item in value
    ):
        raise ValueError(
            "messages JSON must be a non-empty message array or an object containing it"
        )
    return value


def _vision_position_ids(
    start: int, grid: Sequence[int], device: torch.device
) -> torch.Tensor:
    t, h, w = (int(value) for value in grid)
    merge = VISION_SPATIAL_MERGE
    if t <= 0 or h <= 0 or w <= 0 or h % merge or w % merge:
        raise ValueError(f"invalid vision grid {tuple(grid)}")
    gh, gw = h // merge, w // merge
    temporal = torch.arange(t, device=device, dtype=torch.long)
    temporal = temporal.repeat_interleave(gh * gw) + start
    height = torch.arange(gh, device=device, dtype=torch.long)
    height = height.repeat_interleave(gw).repeat(t) + start
    width = torch.arange(gw, device=device, dtype=torch.long)
    width = width.repeat(gh * t) + start
    return torch.stack((temporal, height, width))


def build_mrope_positions(
    mm_token_type_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
) -> tuple[torch.Tensor, int]:
    """Reproduce the checkpoint's three-axis RoPE index for one prompt."""

    types = mm_token_type_ids.to(dtype=torch.long).flatten()
    if types.numel() == 0:
        raise ValueError("multimodal prompt must not be empty")
    if torch.any((types < 0) | (types > 2)):
        raise ValueError("mm_token_type_ids must contain only text/image/video values 0/1/2")
    device = types.device
    images = [] if image_grid_thw is None else image_grid_thw.tolist()
    videos: list[list[int]] = []
    if video_grid_thw is not None:
        for t, h, w in video_grid_thw.tolist():
            videos.extend([[1, int(h), int(w)] for _ in range(int(t))])
    grids = {1: iter(images), 2: iter(videos)}
    consumed = {1: 0, 2: 0}
    available = {1: len(images), 2: len(videos)}
    parts: list[torch.Tensor] = []
    current = 0
    begin = 0
    values = types.tolist()
    while begin < len(values):
        modality = values[begin]
        end = begin + 1
        while end < len(values) and values[end] == modality:
            end += 1
        length = end - begin
        if modality == 0:
            pos = torch.arange(length, device=device, dtype=torch.long) + current
            parts.append(pos.unsqueeze(0).expand(3, -1))
            current += length
        else:
            try:
                grid = next(grids[modality])
            except StopIteration as exc:
                label = "image" if modality == 1 else "video"
                raise ValueError(f"more {label} placeholder runs than grid entries") from exc
            consumed[modality] += 1
            expected = int(grid[0]) * (int(grid[1]) // VISION_SPATIAL_MERGE) * (
                int(grid[2]) // VISION_SPATIAL_MERGE
            )
            if length != expected:
                label = "image" if modality == 1 else "video"
                raise ValueError(
                    f"{label} placeholder run has {length} tokens, expected {expected} "
                    f"for grid {grid}"
                )
            parts.append(_vision_position_ids(current, grid, device))
            current += max(int(grid[1]), int(grid[2])) // VISION_SPATIAL_MERGE
        begin = end

    for modality, label in ((1, "image"), (2, "video")):
        if consumed[modality] != available[modality]:
            raise ValueError(
                f"{label} grid count {available[modality]} does not match placeholder runs "
                f"{consumed[modality]}"
            )
    positions = torch.cat(parts, dim=1)
    delta = int(positions.max().item()) + 1 - len(values)
    return positions, delta


def _optional_cpu(output: Mapping[str, Any], name: str) -> torch.Tensor | None:
    value = output.get(name)
    return None if value is None else value.cpu()


def batch_from_processor_output(output: Mapping[str, Any]) -> MultimodalBatch:
    """Bind one library processor result to the target's execution metadata."""

    input_ids = output["input_ids"]
    mm_types = output.get("mm_token_type_ids")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("reference inference supports exactly one prompt")
    if mm_types is None or mm_types.shape != input_ids.shape:
        raise ValueError("processor did not return mm_token_type_ids matching input_ids")
    attention_mask = output.get("attention_mask")
    if attention_mask is not None and (
        attention_mask.shape != input_ids.shape
        or not bool(torch.all(attention_mask == 1))
    ):
        raise ValueError("reference inference requires one unpadded prompt")

    input_ids = input_ids[0].cpu()
    mm_types = mm_types[0].cpu()
    image_grid = _optional_cpu(output, "image_grid_thw")
    video_grid = _optional_cpu(output, "video_grid_thw")
    positions, delta = build_mrope_positions(mm_types, image_grid, video_grid)
    return MultimodalBatch(
        input_ids=input_ids,
        mm_token_type_ids=mm_types,
        position_ids=positions,
        rope_delta=delta,
        pixel_values=_optional_cpu(output, "pixel_values"),
        image_grid_thw=image_grid,
        pixel_values_videos=_optional_cpu(output, "pixel_values_videos"),
        video_grid_thw=video_grid,
    )


__all__ = [
    "MultimodalBatch",
    "batch_from_processor_output",
    "build_mrope_positions",
    "load_messages",
]
