"""Static Qwen3.6 vision-tower schedule over q5090 weights."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch

from tools.q5090.reader import Reader
from tools.q5090_convert import format as fmt
from tools.q5090_convert import qtypes as qt

from .config import VISION_CFG
from .ops import linear, residual_add
from .vision_ops import (
    add_bias,
    apply_vision_rope,
    gelu,
    interpolate_position_embedding,
    layer_norm,
    vision_attention,
    vision_cu_seqlens,
    vision_position_ids,
)


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
    """Runs one request while retaining only small BF16 control tensors."""

    def __init__(
        self,
        reader: Reader,
        device: torch.device,
        *,
        compile_codec: bool,
        memory_bytes: int | None = None,
        headroom_bytes: int = 2 << 30,
        attention_limit: int | None = None,
    ):
        if not (reader.header["flags"] & fmt.FLAG_VISION_PRESENT):
            raise ValueError("multimodal input requires a q5090 VISION_ENCODER module")
        self.reader = reader
        self.device = device
        self.compile_codec = compile_codec
        self.memory_bytes = memory_bytes
        self.headroom_bytes = headroom_bytes
        self.attention_limit = attention_limit
        self._controls: dict[str, torch.Tensor] = {}

    def _weight(self, name: str) -> torch.Tensor:
        view = self.reader.views[name]
        if view.block.module_kind != qt.MODULE_VISION:
            raise ValueError(f"{name} is not a VISION_ENCODER tensor")
        if view.block.layout == qt.LAYOUT_CONTIGUOUS:
            cached = self._controls.get(name)
            if cached is not None:
                return cached
        value = self.reader.decode_view(
            view,
            self.device,
            compiled=self.compile_codec,
        )
        if view.block.layout == qt.LAYOUT_CONTIGUOUS:
            self._controls[name] = value
        return value

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

    def encode(
        self,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
        *,
        tap: Callable[[str, torch.Tensor], None] | None = None,
    ) -> VisionOutput:
        pixels = [value for value in (pixel_values, pixel_values_videos) if value is not None]
        grids = [value for value in (image_grid_thw, video_grid_thw) if value is not None]
        if not pixels or not grids:
            raise ValueError("vision encode requires image or video pixels and matching grids")
        if len(pixels) != len(grids):
            raise ValueError("vision pixels/grid modality mismatch")
        combined_pixels = torch.cat(pixels, dim=0)
        combined_grid = torch.cat(grids, dim=0).to(device=self.device, dtype=torch.long)
        if combined_pixels.shape != (int(combined_grid.prod(-1).sum()), VISION_CFG.patch_dim):
            raise ValueError(
                f"processor pixels have shape {tuple(combined_pixels.shape)}, expected "
                f"[{int(combined_grid.prod(-1).sum())},{VISION_CFG.patch_dim}]"
            )
        image_patches, image_tokens, image_pairs = self._grid_stats(image_grid_thw)
        video_patches, video_tokens, video_pairs = self._grid_stats(video_grid_thw)
        raw_patches = image_patches + video_patches
        llm_tokens = image_tokens + video_tokens
        attention_pairs = image_pairs + video_pairs
        estimate = self._validate_budget(raw_patches, attention_pairs)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        x = combined_pixels.to(device=self.device, dtype=torch.bfloat16)
        weight = self._weight("model.visual.patch_embed.proj.weight")
        x = add_bias(linear(x, weight), self._weight("model.visual.patch_embed.proj.bias"))
        del weight
        position = interpolate_position_embedding(
            self._weight("model.visual.pos_embed.weight"), combined_grid
        )
        x = residual_add(x, position)
        pos_ids = vision_position_ids(combined_grid)
        cu_seqlens = vision_cu_seqlens(combined_grid)
        if tap:
            tap("patch_embed", x)
        for layer in range(VISION_CFG.depth):
            prefix = f"model.visual.blocks.{layer}."
            h = layer_norm(
                x,
                self._weight(prefix + "norm1.weight"),
                self._weight(prefix + "norm1.bias"),
            )
            weight = self._weight(prefix + "attn.qkv.weight")
            qkv = add_bias(linear(h, weight), self._weight(prefix + "attn.qkv.bias"))
            del weight, h
            qkv = qkv.reshape(-1, 3, VISION_CFG.heads, VISION_CFG.head_dim)
            q, k, v = qkv.unbind(1)
            q, k = apply_vision_rope(q, k, pos_ids)
            attended = vision_attention(q, k, v, cu_seqlens).reshape(-1, VISION_CFG.hidden)
            del qkv, q, k, v
            weight = self._weight(prefix + "attn.proj.weight")
            projected = add_bias(
                linear(attended, weight), self._weight(prefix + "attn.proj.bias")
            )
            del weight, attended
            x = residual_add(x, projected)
            h = layer_norm(
                x,
                self._weight(prefix + "norm2.weight"),
                self._weight(prefix + "norm2.bias"),
            )
            weight = self._weight(prefix + "mlp.linear_fc1.weight")
            h = gelu(
                add_bias(linear(h, weight), self._weight(prefix + "mlp.linear_fc1.bias")),
                approximate=True,
            )
            del weight
            weight = self._weight(prefix + "mlp.linear_fc2.weight")
            h = add_bias(linear(h, weight), self._weight(prefix + "mlp.linear_fc2.bias"))
            del weight
            x = residual_add(x, h)
            if tap and layer in {0, 13, 26}:
                tap(f"block_{layer:02d}", x)
        x = layer_norm(
            x,
            self._weight("model.visual.merger.norm.weight"),
            self._weight("model.visual.merger.norm.bias"),
        ).reshape(-1, VISION_CFG.merger_hidden)
        weight = self._weight("model.visual.merger.linear_fc1.weight")
        x = gelu(
            add_bias(linear(x, weight), self._weight("model.visual.merger.linear_fc1.bias")),
            approximate=False,
        )
        del weight
        weight = self._weight("model.visual.merger.linear_fc2.weight")
        x = add_bias(linear(x, weight), self._weight("model.visual.merger.linear_fc2.bias"))
        del weight
        if tap:
            tap("merger", x)
        if x.shape != (llm_tokens, VISION_CFG.out_hidden):
            raise RuntimeError(
                f"vision merger returned {tuple(x.shape)}, expected "
                f"({llm_tokens},{VISION_CFG.out_hidden})"
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        image_embeddings = x[:image_tokens] if image_tokens else None
        video_embeddings = x[image_tokens:] if video_tokens else None
        elapsed = time.perf_counter() - started
        peak = (
            torch.cuda.max_memory_allocated(self.device)
            if self.device.type == "cuda"
            else 0
        )
        stats = VisionStats(
            images=0 if image_grid_thw is None else int(image_grid_thw.shape[0]),
            videos=0 if video_grid_thw is None else int(video_grid_thw.shape[0]),
            raw_patches=raw_patches,
            llm_tokens=llm_tokens,
            attention_pairs=attention_pairs,
            estimated_peak_bytes=estimate,
            peak_allocated_bytes=peak,
            encode_seconds=elapsed,
        )
        return VisionOutput(image_embeddings, video_embeddings, stats)

    def close(self) -> None:
        self._controls.clear()
