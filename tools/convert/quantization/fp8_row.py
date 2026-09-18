"""Row-wise BF16-to-FP8 quantization with the existing exact rounding boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

_E4M3FN_MAX = np.float32(448.0)
_BF16_MIN_SUBNORMAL_WORD = np.uint16(0x0001)


def _positive_e4m3fn_values() -> np.ndarray:
    words = torch.arange(0x7F, dtype=torch.uint8)
    return words.view(torch.float8_e4m3fn).float().numpy()


_POSITIVE_E4M3FN_VALUES = _positive_e4m3fn_values()


@dataclass(frozen=True, slots=True)
class RowScaledFp8Words:
    codes: torch.Tensor
    scales: torch.Tensor


def _bf16_rne_words(values: np.ndarray) -> np.ndarray:
    """Round nonnegative binary32 values to exact BF16 words, ties to even."""

    if values.dtype != np.float32:
        raise TypeError("BF16 rounding input must be binary32")
    bits = values.view(np.uint32)
    upper = bits >> np.uint32(16)
    rounded = (
        bits.astype(np.uint64)
        + np.uint64(0x7FFF)
        + (upper & np.uint32(1)).astype(np.uint64)
    ) >> np.uint64(16)
    return rounded.astype(np.uint16)


def _bf16_words_to_float32(words: np.ndarray) -> np.ndarray:
    if words.dtype != np.uint16:
        raise TypeError("BF16 words must be uint16")
    return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _round_e4m3fn_rne(values: np.ndarray) -> np.ndarray:
    """Round bounded finite binary32 values to exact E4M3FN words."""

    if values.dtype != np.float32:
        raise TypeError("E4M3FN rounding input must be binary32")
    if not np.isfinite(values).all() or np.any(np.abs(values) > _E4M3FN_MAX):
        raise ValueError("E4M3FN rounding input must be finite and bounded")

    magnitude = np.abs(values)
    upper = np.searchsorted(_POSITIVE_E4M3FN_VALUES, magnitude, side="left").astype(
        np.int16
    )
    upper = np.minimum(upper, 0x7E)
    lower = np.maximum(upper - 1, 0)
    lower_distance = magnitude - _POSITIVE_E4M3FN_VALUES[lower]
    upper_distance = _POSITIVE_E4M3FN_VALUES[upper] - magnitude
    choose_upper = (upper_distance < lower_distance) | (
        (upper_distance == lower_distance) & ((upper & 1) == 0)
    )
    words = np.where(choose_upper, upper, lower).astype(np.uint8)
    words |= np.where(np.signbit(values), 0x80, 0).astype(np.uint8)
    return words


def _quantize_host_rows(host: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if host.dtype != np.float32 or host.ndim != 2:
        raise TypeError("FP8 quantization rows must be a rank-two binary32 array")
    if not np.isfinite(host).all():
        raise ValueError("FP8 quantization source contains NaN or infinity")

    max_abs = np.max(np.abs(host), axis=1)
    zero_rows = max_abs == np.float32(0.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw_scale = (max_abs.astype(np.float64) / float(_E4M3FN_MAX)).astype(np.float32)
    scale_words = _bf16_rne_words(raw_scale)
    underflow = (scale_words == 0) & ~zero_rows
    scale_words[underflow] = _BF16_MIN_SUBNORMAL_WORD

    scale32 = _bf16_words_to_float32(scale_words)
    invalid_scale = (~zero_rows) & (
        ~np.isfinite(scale32) | (scale32 <= np.float32(0.0))
    )
    if invalid_scale.any():
        raise ValueError("FP8 quantization row scale is not finite and positive")

    reciprocal = np.zeros(scale32.shape, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        reciprocal[~zero_rows] = (1.0 / scale32[~zero_rows].astype(np.float64)).astype(
            np.float32
        )
        normalized = (
            host.astype(np.float64) * reciprocal[:, None].astype(np.float64)
        ).astype(np.float32)
    if not np.isfinite(normalized[~zero_rows]).all():
        raise ValueError("FP8 quantization normalization produced a non-finite value")
    bounded = np.clip(normalized, -_E4M3FN_MAX, _E4M3FN_MAX)
    codes = _round_e4m3fn_rne(bounded)
    codes[zero_rows] = 0
    return codes, scale_words


def quantize_bf16_rows(weight: torch.Tensor) -> RowScaledFp8Words:
    """Quantize BF16 ``[N,K]`` values to E4M3FN codes and BF16 row scales."""

    if weight.dtype != torch.bfloat16 or weight.dim() != 2:
        raise TypeError("FP8 quantization source must be a rank-two BF16 tensor")
    if any(dim <= 0 for dim in weight.shape):
        raise ValueError("FP8 quantization source dimensions must be positive")
    host = weight.detach().to(device="cpu", dtype=torch.float32).numpy()
    code_words, scale_words = _quantize_host_rows(host)
    codes = torch.from_numpy(code_words.copy())
    signed_scales = torch.from_numpy(scale_words.view(np.int16).copy())
    return RowScaledFp8Words(codes, signed_scales.view(torch.bfloat16))
