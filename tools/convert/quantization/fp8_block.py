"""Max-abs E4M3FN encoding with one BF16 multiplier per 128-by-128 block."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .fp8_row import quantize_fp32_rows


@dataclass(frozen=True, slots=True)
class BlockScaledFp8Words:
    codes: torch.Tensor
    scales: torch.Tensor


def quantize_blocks(weight: torch.Tensor) -> BlockScaledFp8Words:
    """Encode BF16/FP32 ``[N,K]`` values; both axes contain complete 128 tiles."""

    if weight.dtype not in (torch.bfloat16, torch.float32) or weight.dim() != 2:
        raise TypeError("block FP8 source must be a rank-two BF16 or FP32 tensor")
    n, k = weight.shape
    if n <= 0 or k <= 0 or n % 128 or k % 128:
        raise ValueError(
            "block FP8 source dimensions must be positive multiples of 128"
        )
    rows, columns = n // 128, k // 128
    groups = (
        weight.detach()
        .float()
        .reshape(rows, 128, columns, 128)
        .permute(0, 2, 1, 3)
        .reshape(rows * columns, 128 * 128)
    )
    encoded = quantize_fp32_rows(groups)
    codes = (
        encoded.codes.reshape(rows, columns, 128, 128).permute(0, 2, 1, 3).reshape(n, k)
    )
    return BlockScaledFp8Words(codes, encoded.scales.reshape(rows, columns))
