"""Library-backed tensor operators for the Qwen3.6 vision reference path."""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F

from .config import VISION_CFG
from .ops import bf16


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(
        x,
        (x.shape[-1],),
        weight.to(device=x.device, dtype=x.dtype),
        bias.to(device=x.device, dtype=x.dtype),
        VISION_CFG.norm_eps,
    ).to(torch.bfloat16)


def add_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return bf16(x.float() + bias.to(device=x.device, dtype=torch.float32))


def gelu(x: torch.Tensor, *, approximate: bool) -> torch.Tensor:
    return bf16(F.gelu(x.float(), approximate="tanh" if approximate else "none"))


def vision_position_ids(grid_thw: torch.Tensor) -> torch.Tensor:
    merge = VISION_CFG.spatial_merge
    device = grid_thw.device
    parts: list[torch.Tensor] = []
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        if h % merge or w % merge:
            raise ValueError(f"vision grid {(t, h, w)} is not divisible by merge size {merge}")
        hpos = torch.arange(h, device=device).unsqueeze(1).expand(-1, w)
        hpos = hpos.reshape(h // merge, merge, w // merge, merge).transpose(1, 2).flatten()
        wpos = torch.arange(w, device=device).unsqueeze(0).expand(h, -1)
        wpos = wpos.reshape(h // merge, merge, w // merge, merge).transpose(1, 2).flatten()
        parts.append(torch.stack((hpos, wpos), dim=-1).repeat(t, 1))
    return torch.cat(parts, dim=0)


def vision_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    lengths = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    return F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0), value=0)


def bilinear_indices_and_weights(grid_thw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    side = VISION_CFG.position_side
    merge = VISION_CFG.spatial_merge
    device = grid_thw.device
    index_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    weight_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        h_grid = torch.linspace(0, side - 1, h, device=device)
        w_grid = torch.linspace(0, side - 1, w, device=device)
        h_floor, w_floor = h_grid.int(), w_grid.int()
        h_ceil = (h_floor + 1).clamp(max=side - 1)
        w_ceil = (w_floor + 1).clamp(max=side - 1)
        h_frac, w_frac = h_grid - h_floor, w_grid - w_floor
        hfo, hco = h_floor * side, h_ceil * side
        indices = [
            (hfo[:, None] + w_floor[None, :]).flatten(),
            (hfo[:, None] + w_ceil[None, :]).flatten(),
            (hco[:, None] + w_floor[None, :]).flatten(),
            (hco[:, None] + w_ceil[None, :]).flatten(),
        ]
        weights = [
            ((1 - h_frac)[:, None] * (1 - w_frac)[None, :]).flatten(),
            ((1 - h_frac)[:, None] * w_frac[None, :]).flatten(),
            (h_frac[:, None] * (1 - w_frac)[None, :]).flatten(),
            (h_frac[:, None] * w_frac[None, :]).flatten(),
        ]
        hi = torch.arange(h, device=device).view(h // merge, merge)
        wi = torch.arange(w, device=device).view(w // merge, merge)
        reorder = (hi[:, :, None, None] * w + wi[None, None, :, :]).transpose(1, 2).flatten().repeat(t)
        for corner in range(4):
            index_parts[corner].append(indices[corner][reorder])
            weight_parts[corner].append(weights[corner][reorder])
    return (
        torch.stack([torch.cat(part) for part in index_parts]),
        torch.stack([torch.cat(part) for part in weight_parts]),
    )


def interpolate_position_embedding(table: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
    indices, weights = bilinear_indices_and_weights(grid_thw)
    gathered = table.index_select(0, indices.flatten()).reshape(4, -1, table.shape[-1])
    return bf16((gathered.float() * weights[:, :, None].float()).sum(0))


@lru_cache(maxsize=16)
def _vision_inv_freq(device_type: str, device_index: int | None) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    dim = VISION_CFG.head_dim // 2
    values = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    return 1.0 / (VISION_CFG.rope_theta ** (values / dim))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_vision_rope(
    q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    inv = _vision_inv_freq(q.device.type, q.device.index)
    rotary = (position_ids.to(q.device, torch.float32).unsqueeze(-1) * inv).flatten(1)
    emb = torch.cat((rotary, rotary), dim=-1)
    cos, sin = emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)
    qf, kf = q.float(), k.float()
    return (
        bf16(qf * cos + _rotate_half(qf) * sin),
        bf16(kf * cos + _rotate_half(kf) * sin),
    )


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if q.device.type != "cuda":
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        with sdpa_kernel(backends):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    except (ImportError, AttributeError):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)


def vision_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor
) -> torch.Tensor:
    """Packed non-causal MHA; equal consecutive segment lengths execute as one batch."""
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 3:
        raise ValueError("vision attention requires matching [P,H,D] q/k/v")
    boundaries = [int(value) for value in cu_seqlens.cpu().tolist()]
    if not boundaries or boundaries[0] != 0 or boundaries[-1] != q.shape[0]:
        raise ValueError("vision cu_seqlens do not cover q/k/v")
    out = torch.empty_like(q)
    segment = 0
    while segment + 1 < len(boundaries):
        length = boundaries[segment + 1] - boundaries[segment]
        run_end = segment + 1
        while (
            run_end + 1 < len(boundaries)
            and boundaries[run_end + 1] - boundaries[run_end] == length
        ):
            run_end += 1
        begin, end = boundaries[segment], boundaries[run_end]
        batch = run_end - segment

        def heads(x: torch.Tensor) -> torch.Tensor:
            return x[begin:end].reshape(
                batch, length, VISION_CFG.heads, VISION_CFG.head_dim
            ).transpose(1, 2)

        attended = _sdpa(heads(q), heads(k), heads(v))
        out[begin:end] = attended.transpose(1, 2).reshape(
            end - begin, VISION_CFG.heads, VISION_CFG.head_dim
        )
        segment = run_end
    return bf16(out)
