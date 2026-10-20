"""Import published E4M3FN matrices with BF16 128-by-128 block multipliers."""

from __future__ import annotations

import torch

from .logical import EncodedRows, LogicalSource
from .safetensors import SafetensorsSource


def fp8_block_source(
    store: SafetensorsSource, prefix: str, shape: tuple[int, int]
) -> LogicalSource:
    n, k = shape
    if n % 128 or k % 128:
        raise ValueError("FP8 block import requires N and K divisible by 128")
    for suffix, dims, dtype in (
        ("weight", shape, "F8_E4M3"),
        ("weight_scale_inv", (n // 128, k // 128), "BF16"),
    ):
        info = store.describe(prefix + "." + suffix)
        if info.shape != dims or info.dtype != dtype:
            raise ValueError(f"{prefix}.{suffix}: unexpected encoded geometry/type")

    def encoded(first: int, last: int) -> EncodedRows:
        if not 0 <= first < last <= n or first % 128 or last % 128:
            raise ValueError("FP8 encoded reads require complete 128-row blocks")
        codes = store.read_flat(prefix + ".weight", first * k, last * k)
        scales = store.read_flat(
            prefix + ".weight_scale_inv",
            first // 128 * (k // 128),
            last // 128 * (k // 128),
        )
        return EncodedRows(
            "fp8_e4m3fn_block128x128_bf16",
            codes.view(torch.uint8).reshape(last - first, k),
            scales.reshape((last - first) // 128, k // 128),
        )

    def values(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        first = begin // (128 * k) * 128
        last = (end + 128 * k - 1) // (128 * k) * 128
        words = encoded(first, last)
        scales = (
            words.scales.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
        )
        decoded = words.codes.view(torch.float8_e4m3fn).float() * scales
        return decoded.reshape(-1)[begin - first * k : end - first * k]

    return LogicalSource(
        shape,
        f"{store.path}:{prefix} (block FP8)",
        values,
        encoded,
        encoded_format="fp8_e4m3fn_block128x128_bf16",
    )
