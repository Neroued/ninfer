"""BF16-to-NVFP4 quantization with a stored FP32 weight divisor.

A dequantized weight is ``E2M1(code) * E4M3FN(scale) / weight_divisor``. The divisor
is chosen per parent so the largest block scale lands on the E4M3FN maximum, which
spends the whole scale range on the tensor instead of on the global magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct

import torch

_E2M1_MAX = 6.0
_E4M3FN_MAX = 448.0
_GROUP = 16
_MIN_SUBNORMAL_WORD = 0x01


@dataclass(frozen=True, slots=True)
class Nvfp4Words:
    codes: torch.Tensor
    scales: torch.Tensor


def weight_divisor_for(max_abs: float) -> bytes:
    """Return the FP32 divisor that maps ``max_abs`` onto the full scale range."""

    if max_abs != max_abs or max_abs in (float("inf"), float("-inf")) or max_abs < 0:
        raise ValueError("NVFP4 source contains NaN or infinity")
    if max_abs == 0.0:
        return struct.pack("<f", 1.0)
    divisor = struct.unpack(
        "<f", struct.pack("<f", _E2M1_MAX * _E4M3FN_MAX / max_abs)
    )[0]
    if divisor == float("inf") or divisor <= 0.0:
        raise ValueError("NVFP4 weight divisor is not representable as finite FP32")
    return struct.pack("<f", divisor)


def _e2m1_magnitude_index(magnitude: torch.Tensor) -> torch.Tensor:
    # Midpoints between E2M1 values resolve to the neighbour with the even code word.
    return (
        (magnitude > 0.25).to(torch.uint8)
        + (magnitude >= 0.75).to(torch.uint8)
        + (magnitude > 1.25).to(torch.uint8)
        + (magnitude >= 1.75).to(torch.uint8)
        + (magnitude > 2.5).to(torch.uint8)
        + (magnitude >= 3.5).to(torch.uint8)
        + (magnitude > 5.0).to(torch.uint8)
    )


def quantize_rows(values: torch.Tensor, weight_divisor: bytes) -> Nvfp4Words:
    """Quantize floating-point ``[N,K]`` rows (K divisible by 16) to NVFP4 words."""

    if values.dim() != 2 or not values.dtype.is_floating_point:
        raise TypeError("NVFP4 quantization source must be a floating-point matrix")
    rows, k = values.shape
    if rows <= 0 or k <= 0 or k % _GROUP:
        raise ValueError("NVFP4 quantization rows must have K divisible by 16")
    divisor = struct.unpack("<f", weight_divisor)[0]
    host = values.detach().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(host).all()):
        raise ValueError("NVFP4 quantization source contains NaN or infinity")
    blocks = host.reshape(rows, k // _GROUP, _GROUP)
    block_max = blocks.abs().amax(dim=2)
    raw_scale = (block_max * divisor / _E2M1_MAX).clamp_(max=_E4M3FN_MAX)
    scale_words = raw_scale.to(torch.float8_e4m3fn).view(torch.uint8)
    scale_words[(scale_words == 0) & (block_max > 0)] = _MIN_SUBNORMAL_WORD
    decoded = scale_words.view(torch.float8_e4m3fn).to(torch.float32)
    live = decoded > 0
    normalized = blocks * divisor / torch.where(live, decoded, torch.ones_like(decoded))[
        :, :, None
    ]
    normalized = torch.where(live[:, :, None], normalized, torch.zeros_like(normalized))
    magnitude = _e2m1_magnitude_index(normalized.abs())
    negative = torch.signbit(normalized).to(torch.uint8) << 3
    nibbles = (magnitude | negative).reshape(rows, k)
    packed = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    return Nvfp4Words(packed.contiguous(), scale_words.contiguous())
