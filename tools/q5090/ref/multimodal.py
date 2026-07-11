"""Hugging Face input processing and Qwen3.6 multimodal position metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Mapping, Sequence

import torch

from .config import VISION_CFG

IMAGE_PIXELS = 1024 * 1024
VIDEO_PIXELS = 4 * 1024 * 1024


def _fetch_videos_opencv(_processor, video_or_videos, sample_indices_fn=None):
    """Use the processor's sampler with a dependable local-file decoder."""
    from transformers.video_utils import load_video

    if isinstance(video_or_videos, list):
        fetched = [
            _fetch_videos_opencv(_processor, item, sample_indices_fn=sample_indices_fn)
            for item in video_or_videos
        ]
        return list(zip(*fetched))
    return load_video(video_or_videos, backend="opencv", sample_indices_fn=sample_indices_fn)


@dataclass
class MultimodalBatch:
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


def load_messages(path: str | Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, Mapping) and "messages" in value:
        value = value["messages"]
    if not isinstance(value, list) or not value or not all(isinstance(item, dict) for item in value):
        raise ValueError("messages JSON must be a non-empty message array or an object containing it")
    return value


def _vision_position_ids(start: int, grid: Sequence[int], device: torch.device) -> torch.Tensor:
    t, h, w = (int(value) for value in grid)
    merge = VISION_CFG.spatial_merge
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
    """Reproduce Qwen3.5 get_rope_index for one unpadded prompt."""
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
            expected = int(grid[0]) * (int(grid[1]) // VISION_CFG.spatial_merge) * (
                int(grid[2]) // VISION_CFG.spatial_merge
            )
            if length != expected:
                label = "image" if modality == 1 else "video"
                raise ValueError(
                    f"{label} placeholder run has {length} tokens, expected {expected} for grid {grid}"
                )
            parts.append(_vision_position_ids(current, grid, device))
            current += max(int(grid[1]), int(grid[2])) // VISION_CFG.spatial_merge
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


class Processor:
    """Thin adapter around the checkpoint's Hugging Face AutoProcessor."""

    def __init__(self, model_dir: str | Path, *, expected_token_ids: Mapping[str, int] | None = None):
        try:
            from transformers import AutoProcessor
            from transformers.utils import is_torchcodec_available
        except ImportError as exc:
            raise RuntimeError("multimodal reference inference requires transformers>=5.12") from exc
        self.model_dir = Path(model_dir)
        self.impl = AutoProcessor.from_pretrained(self.model_dir, local_files_only=True)
        if not is_torchcodec_available():
            self.impl.video_processor.fetch_videos = MethodType(
                _fetch_videos_opencv, self.impl.video_processor
            )
        names = set(self.impl.model_input_names)
        required = {
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "mm_token_type_ids",
        }
        missing = sorted(required - names)
        if missing:
            raise RuntimeError(
                "the active Hugging Face processor is too old for Qwen3.6 multimodal input; "
                f"missing {missing}; install transformers>=5.12"
            )
        if expected_token_ids:
            tokenizer = self.impl.tokenizer
            for token, expected in expected_token_ids.items():
                actual = tokenizer.convert_tokens_to_ids(token)
                if actual != expected:
                    raise ValueError(f"processor token {token}={actual}, q5090 expects {expected}")

    @staticmethod
    def _optional(output, name: str) -> torch.Tensor | None:
        value = output.get(name)
        return None if value is None else value.cpu()

    def process(self, messages: list[dict[str, Any]], *, thinking: bool) -> MultimodalBatch:
        output = self.impl.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=thinking,
            processor_kwargs={
                "images_kwargs": {
                    "size": {"shortest_edge": 32 * 32, "longest_edge": IMAGE_PIXELS}
                },
                "videos_kwargs": {
                    "size": {"shortest_edge": 128 * 32 * 32, "longest_edge": VIDEO_PIXELS}
                },
            },
        )
        input_ids = output["input_ids"]
        mm_types = output.get("mm_token_type_ids")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("q5090 reference multimodal mode supports exactly one prompt")
        if mm_types is None or mm_types.shape != input_ids.shape:
            raise ValueError("processor did not return mm_token_type_ids matching input_ids")
        attention_mask = output.get("attention_mask")
        if attention_mask is not None and (
            attention_mask.shape != input_ids.shape or not bool(torch.all(attention_mask == 1))
        ):
            raise ValueError("q5090 reference multimodal mode requires one unpadded prompt")
        input_ids = input_ids[0].cpu()
        mm_types = mm_types[0].cpu()
        image_grid = self._optional(output, "image_grid_thw")
        video_grid = self._optional(output, "video_grid_thw")
        positions, delta = build_mrope_positions(mm_types, image_grid, video_grid)
        return MultimodalBatch(
            input_ids=input_ids,
            mm_token_type_ids=mm_types,
            position_ids=positions,
            rope_delta=delta,
            pixel_values=self._optional(output, "pixel_values"),
            image_grid_thw=image_grid,
            pixel_values_videos=self._optional(output, "pixel_values_videos"),
            video_grid_thw=video_grid,
        )
