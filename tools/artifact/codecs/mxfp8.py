"""Exact E4M3FN codes and UE8M0 K32 scales."""

from __future__ import annotations

from typing import Sequence

import torch

from ..layouts import row_group_scale_geometry
from ._tensor_bytes import (
    Payload,
    _exact_uint8_matrix,
    _payload_length,
    _payload_tensor,
)


def validate_mxfp8_words(codes: torch.Tensor, scales: torch.Tensor) -> None:
    if bool(((codes & 0x7F) == 0x7F).any()):
        raise ValueError("MXFP8 codes must be finite E4M3FN words")
    if bool((scales == 255).any()):
        raise ValueError("MXFP8 scales must be finite UE8M0 words")


def encode_mxfp8(
    code_words: torch.Tensor, scale_words: torch.Tensor, shape: Sequence[int]
) -> bytes:
    g = row_group_scale_geometry("mxfp8", shape)
    codes = _exact_uint8_matrix(code_words, tuple(shape), "MXFP8 codes")
    scales = _exact_uint8_matrix(
        scale_words, (*shape[:-1], g.groups_per_row), "MXFP8 scales"
    )
    validate_mxfp8_words(codes, scales)
    payload = bytearray(g.payload_bytes)
    payload[: g.code_plane_bytes] = codes.numpy().tobytes()
    payload[g.scale_plane_offset :] = scales.numpy().tobytes()
    return bytes(payload)


def decode_mxfp8_words(
    payload: Payload, shape: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    g = row_group_scale_geometry("mxfp8", shape)
    if _payload_length(payload) != g.payload_bytes:
        raise ValueError("MXFP8 payload size differs from its geometry")
    raw = _payload_tensor(payload, torch.device("cpu"))
    codes = raw[: g.code_plane_bytes].clone().reshape(tuple(shape))
    scales = raw[g.scale_plane_offset :].clone().reshape(*shape[:-1], g.groups_per_row)
    validate_mxfp8_words(codes, scales)
    return codes, scales


def dequantize_mxfp8(
    payload: Payload, shape: Sequence[int], dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    codes, scales = decode_mxfp8_words(payload, shape)
    multipliers = torch.exp2(scales.double() - 127).repeat_interleave(32, -1)
    return (codes.view(torch.float8_e4m3fn).double() * multipliers).to(dtype)
