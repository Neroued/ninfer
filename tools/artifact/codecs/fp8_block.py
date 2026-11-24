"""Exact E4M3FN weights and BF16 multipliers for 128-by-128 blocks."""

from __future__ import annotations

from typing import Sequence

import torch

from ..layouts import fp8_block_geometry
from ._tensor_bytes import (
    Payload,
    _exact_uint8_matrix,
    _payload_length,
    _payload_tensor,
)
from .direct import decode_direct, encode_direct


def validate_fp8_block_words(codes: torch.Tensor, scales: torch.Tensor) -> None:
    if bool(((codes & 0x7F) == 0x7F).any()):
        raise ValueError("block-scaled FP8 codes must be finite E4M3FN words")
    words = scales.view(torch.int16).to(torch.int32) & 0xFFFF
    if bool((((words & 0x8000) != 0) | ((words & 0x7F80) == 0x7F80)).any()):
        raise ValueError(
            "block-scaled FP8 scales must be nonnegative finite BF16 words"
        )
    if bool((words == 0).any()):
        blocks = codes.reshape(
            *codes.shape[:-2], scales.shape[-2], 128, scales.shape[-1], 128
        )
        nonzero = ((blocks & 0x7F) != 0).any(dim=-1).any(dim=-2)
        if bool(((words == 0) & nonzero).any()):
            raise ValueError("a zero block scale requires only signed-zero FP8 codes")


def encode_fp8_block_scaled(
    code_words: torch.Tensor, block_scales: torch.Tensor, shape: Sequence[int]
) -> bytes:
    geometry = fp8_block_geometry("fp8_e4m3fn_block128x128_bf16", shape)
    codes = _exact_uint8_matrix(code_words, tuple(shape), "block-scaled FP8 codes")
    scale_shape = (*shape[:-2], geometry.scale_rows, geometry.scale_columns)
    if block_scales.dtype != torch.bfloat16 or tuple(block_scales.shape) != scale_shape:
        raise TypeError(
            f"block-scaled FP8 scales must be BF16 with shape {scale_shape}"
        )
    scales = block_scales.detach().contiguous().cpu()
    validate_fp8_block_words(codes, scales)
    payload = bytearray(geometry.payload_bytes)
    payload[: geometry.code_plane_bytes] = codes.numpy().tobytes()
    payload[geometry.scale_plane_offset :] = encode_direct(scales, "bf16")
    return bytes(payload)


def decode_fp8_block_scaled_words(
    payload: Payload, shape: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    geometry = fp8_block_geometry("fp8_e4m3fn_block128x128_bf16", shape)
    if _payload_length(payload) != geometry.payload_bytes:
        raise ValueError("block-scaled FP8 payload size differs from its geometry")
    raw = _payload_tensor(payload, torch.device("cpu"))
    codes = raw[: geometry.code_plane_bytes].clone().reshape(tuple(shape))
    scales = decode_direct(
        raw[geometry.scale_plane_offset :],
        "bf16",
        (*shape[:-2], geometry.scale_rows, geometry.scale_columns),
    )
    validate_fp8_block_words(codes, scales)
    return codes, scales


def dequantize_fp8_block_scaled(
    payload: Payload, shape: Sequence[int], dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    codes, scales = decode_fp8_block_scaled_words(payload, shape)
    multipliers = scales.float().repeat_interleave(128, -2).repeat_interleave(128, -1)
    return (codes.view(torch.float8_e4m3fn).float() * multipliers).to(dtype)
