"""Fixed Qwen3.6-35B-A3B vision-tower schedule over a native NInfer artifact."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch

from tools.reference.qwen3_6.common.vision_ops import (
    add_bias,
    apply_vision_rope,
    gelu,
    interpolate_position_embedding,
    layer_norm,
    vision_attention,
    vision_cu_seqlens,
    vision_position_ids,
)

from .bindings import ArtifactBinding, PhysicalBlock
from .config import VISION_CFG
from .ops import linear, residual_add
from .weights import WeightStore


@dataclass(frozen=True)
class VisionStats:
    images: int
    videos: int
    raw_patches: int
    llm_tokens: int
    attention_pairs: int
    estimated_peak_bytes: int
    peak_allocated_bytes: int
    encode_seconds: float

    def summary(self) -> str:
        return (
            f"images={self.images} videos={self.videos} raw_patches={self.raw_patches} "
            f"llm_tokens={self.llm_tokens} attention_pairs={self.attention_pairs} "
            f"estimate={self.estimated_peak_bytes / (1 << 30):.2f}GiB "
            f"peak={self.peak_allocated_bytes / (1 << 30):.2f}GiB "
            f"seconds={self.encode_seconds:.3f}"
        )


@dataclass
class VisionOutput:
    image_embeddings: torch.Tensor | None
    video_embeddings: torch.Tensor | None
    stats: VisionStats


class VisionEncoder:
    """Encode media sequentially with item-bounded GPU activation lifetime."""

    def __init__(
        self,
        binding: ArtifactBinding,
        device: torch.device | str,
        *,
        compile_codec: bool,
        memory_bytes: int | None = None,
        headroom_bytes: int = 2 << 30,
        attention_limit: int | None = None,
    ):
        self.binding = binding
        self.device = torch.device(device)
        self.memory_bytes = memory_bytes
        self.headroom_bytes = headroom_bytes
        self.attention_limit = attention_limit
        self.weights = WeightStore(
            binding,
            self.device,
            capacity=1,
            text=False,
            vision=True,
            compile_codec=compile_codec,
            memory_bytes=memory_bytes,
            headroom_bytes=headroom_bytes,
        )

    def _weight(self, block: PhysicalBlock) -> torch.Tensor:
        return self.weights.tensor(block)

    @staticmethod
    def _grid_stats(grid: torch.Tensor | None) -> tuple[int, int, int]:
        if grid is None:
            return 0, 0, 0
        patches = sum(int(t) * int(h) * int(w) for t, h, w in grid.tolist())
        tokens = patches // VISION_CFG.merge_unit
        pairs = sum(int(t) * (int(h) * int(w)) ** 2 for t, h, w in grid.tolist())
        return patches, tokens, pairs

    @staticmethod
    def _estimate_peak(patches: int) -> int:
        # Live residual, qkv, attention output, MLP intermediate, one decoded
        # merger matrix, and conservative SDPA/allocator slack.
        activation_elems = patches * (
            3 * VISION_CFG.hidden + 3 * VISION_CFG.hidden + VISION_CFG.intermediate
        )
        return activation_elems * 2 + (768 << 20)

    def _validate_budget(self, patches: int, attention_pairs: int) -> int:
        estimate = self._estimate_peak(patches)
        if self.attention_limit is not None and attention_pairs > self.attention_limit:
            raise ValueError(
                f"vision attention work {attention_pairs} exceeds configured limit "
                f"{self.attention_limit}"
            )
        if self.device.type != "cuda":
            return estimate
        with torch.cuda.device(self.device):
            free, _ = torch.cuda.mem_get_info()
        available = min(free, self.memory_bytes) if self.memory_bytes is not None else free
        usable = max(0, available - self.headroom_bytes)
        if estimate > usable:
            raise MemoryError(
                f"estimated vision peak {estimate / (1 << 30):.2f}GiB exceeds usable "
                f"GPU memory {usable / (1 << 30):.2f}GiB"
            )
        return estimate

    @staticmethod
    def _item_slices(
        pixels: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        modality: str,
    ) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
        if pixels is None and grid_thw is None:
            return []
        if pixels is None or grid_thw is None:
            raise ValueError(f"{modality} pixels and grid must be provided together")
        if pixels.ndim != 2 or pixels.shape[1] != VISION_CFG.patch_dim:
            raise ValueError(
                f"{modality} pixels have shape {tuple(pixels.shape)}, expected "
                f"[patches,{VISION_CFG.patch_dim}]"
            )

        items: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        offset = 0
        for index, (t, h, w) in enumerate(grid_thw.tolist()):
            patches = int(t) * int(h) * int(w)
            end = offset + patches
            items.append(
                (
                    f"{modality}_{index:03d}",
                    pixels[offset:end],
                    grid_thw[index : index + 1],
                )
            )
            offset = end
        if offset != pixels.shape[0]:
            raise ValueError(
                f"{modality} grid describes {offset} patches, got {pixels.shape[0]}"
            )
        return items

    def _encode_item(
        self,
        pixels: torch.Tensor,
        grid_thw: torch.Tensor,
        name: str,
        tap: Callable[[str, torch.Tensor], None] | None,
    ) -> torch.Tensor:
        vision = self.binding.vision
        grid = grid_thw.to(device=self.device, dtype=torch.long)
        _, llm_tokens, _ = self._grid_stats(grid_thw)
        x = pixels.to(device=self.device, dtype=torch.bfloat16)
        weight = self._weight(vision.patch_embedding)
        x = add_bias(linear(x, weight), self._weight(vision.patch_embedding_bias))
        del weight
        position = interpolate_position_embedding(
            self._weight(vision.position_embedding), grid
        )
        x = residual_add(x, position)
        pos_ids = vision_position_ids(grid)
        cu_seqlens = vision_cu_seqlens(grid)
        if tap:
            tap(f"{name}/patch_embed", x)

        for layer in vision.layers:
            h = layer_norm(
                x,
                self._weight(layer.norm1_weight),
                self._weight(layer.norm1_bias),
            )
            weight = self._weight(layer.attention_qkv)
            qkv = add_bias(linear(h, weight), self._weight(layer.attention_qkv_bias))
            del weight, h
            qkv = qkv.reshape(-1, 3, VISION_CFG.heads, VISION_CFG.head_dim)
            q, k, v = qkv.unbind(1)
            q, k = apply_vision_rope(q, k, pos_ids)
            attended = vision_attention(q, k, v, cu_seqlens).reshape(
                -1, VISION_CFG.hidden
            )
            del qkv, q, k, v
            weight = self._weight(layer.attention_output)
            projected = add_bias(
                linear(attended, weight), self._weight(layer.attention_output_bias)
            )
            del weight, attended
            x = residual_add(x, projected)
            h = layer_norm(
                x,
                self._weight(layer.norm2_weight),
                self._weight(layer.norm2_bias),
            )
            weight = self._weight(layer.mlp_fc1)
            h = gelu(
                add_bias(linear(h, weight), self._weight(layer.mlp_fc1_bias)),
                approximate=True,
            )
            del weight
            weight = self._weight(layer.mlp_fc2)
            h = add_bias(linear(h, weight), self._weight(layer.mlp_fc2_bias))
            del weight
            x = residual_add(x, h)
            if tap and layer.index in {0, 13, 26}:
                tap(f"{name}/block_{layer.index:02d}", x)

        merger = vision.merger
        x = layer_norm(
            x,
            self._weight(merger.norm_weight),
            self._weight(merger.norm_bias),
        ).reshape(-1, VISION_CFG.merger_hidden)
        weight = self._weight(merger.fc1)
        x = gelu(
            add_bias(linear(x, weight), self._weight(merger.fc1_bias)),
            approximate=False,
        )
        del weight
        weight = self._weight(merger.fc2)
        x = add_bias(linear(x, weight), self._weight(merger.fc2_bias))
        del weight
        if tap:
            tap(f"{name}/merger", x)
        if x.shape != (llm_tokens, VISION_CFG.out_hidden):
            raise RuntimeError(
                f"{name} vision merger returned {tuple(x.shape)}, expected "
                f"({llm_tokens},{VISION_CFG.out_hidden})"
            )
        return x.to(device="cpu", dtype=torch.bfloat16)

    def encode(
        self,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
        *,
        tap: Callable[[str, torch.Tensor], None] | None = None,
    ) -> VisionOutput:
        image_items = self._item_slices(pixel_values, image_grid_thw, "image")
        video_items = self._item_slices(
            pixel_values_videos, video_grid_thw, "video"
        )
        items = image_items + video_items
        if not items:
            raise ValueError("vision encode requires image or video pixels and matching grids")

        item_stats = [self._grid_stats(grid) for _, _, grid in items]
        raw_patches = sum(item[0] for item in item_stats)
        llm_tokens = sum(item[1] for item in item_stats)
        attention_pairs = sum(item[2] for item in item_stats)
        max_item_patches = max(item[0] for item in item_stats)
        estimate = self._validate_budget(max_item_patches, attention_pairs)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        started = time.perf_counter()
        image_outputs = [
            self._encode_item(pixels, grid, name, tap)
            for name, pixels, grid in image_items
        ]
        video_outputs = [
            self._encode_item(pixels, grid, name, tap)
            for name, pixels, grid in video_items
        ]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        image_embeddings = torch.cat(image_outputs) if image_outputs else None
        video_embeddings = torch.cat(video_outputs) if video_outputs else None
        elapsed = time.perf_counter() - started
        peak = (
            torch.cuda.max_memory_allocated(self.device)
            if self.device.type == "cuda"
            else 0
        )
        stats = VisionStats(
            images=len(image_items),
            videos=len(video_items),
            raw_patches=raw_patches,
            llm_tokens=llm_tokens,
            attention_pairs=attention_pairs,
            estimated_peak_bytes=estimate,
            peak_allocated_bytes=peak,
            encode_seconds=elapsed,
        )
        return VisionOutput(image_embeddings, video_embeddings, stats)

    def close(self) -> None:
        self.weights.close()

    def __enter__(self) -> "VisionEncoder":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


__all__ = ["VisionEncoder", "VisionOutput", "VisionStats"]
