"""Exact NVFP4 codes, scale swizzling, and the stored FP32 weight divisor."""

from __future__ import annotations

import struct
from typing import Sequence

import torch

from ..formats import valid_positive_fp32_word
from ..layouts import block_scale_geometry
from ._tensor_bytes import (
    Payload,
    _exact_uint8_matrix,
    _payload_length,
    _payload_tensor,
)
from .direct import encode_direct


def swizzle_nvfp4_scales(
    natural_scales: torch.Tensor, shape: Sequence[int]
) -> torch.Tensor:
    """Swizzle each matrix's natural K16 scales in a matrix or bank."""

    geometry = block_scale_geometry("nvfp4", shape)
    source = _exact_uint8_matrix(
        natural_scales,
        (*shape[:-1], geometry.groups_per_row),
        "NVFP4 scales",
    )
    return (
        source.reshape(
            geometry.matrices * geometry.n // 128, 4, 32, geometry.k_tiles, 4
        )
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(-1)
    )


def unswizzle_nvfp4_scales(
    stored_scales: torch.Tensor, shape: Sequence[int]
) -> torch.Tensor:
    """Recover natural ``[N,K/16]`` E4M3FN words from registered layout bytes."""

    geometry = block_scale_geometry("nvfp4", shape)
    if (
        stored_scales.dtype != torch.uint8
        or stored_scales.dim() != 1
        or stored_scales.numel() != geometry.scale_plane_bytes
    ):
        raise TypeError(
            "stored NVFP4 scales must be one-dimensional uint8 with "
            f"{geometry.scale_plane_bytes} elements"
        )
    source = stored_scales.detach().contiguous().cpu()
    return (
        source.reshape(
            geometry.matrices * geometry.n // 128, geometry.k_tiles, 32, 4, 4
        )
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(*shape[:-1], geometry.groups_per_row)
    )


def _divisor_words(
    value: torch.Tensor | bytes | bytearray | memoryview, count: int
) -> bytes:
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.float32 or value.numel() != count:
            raise TypeError("NVFP4 requires one FP32 weight divisor per matrix")
        raw = encode_direct(value.reshape(-1), "fp32")
    else:
        raw = bytes(value)
        if len(raw) != 4 * count:
            raise TypeError("NVFP4 requires one FP32 weight divisor per matrix")
    if any(
        not valid_positive_fp32_word(word) for (word,) in struct.iter_unpack("<I", raw)
    ):
        raise ValueError("NVFP4 weight divisor must be finite and positive")
    return raw


def encode_nvfp4(
    packed_codes: torch.Tensor,
    natural_scales: torch.Tensor,
    weight_divisor: torch.Tensor | bytes | bytearray | memoryview,
    shape: Sequence[int],
) -> bytes:
    """Encode exact source NVFP4 words without numerical conversion."""

    geometry = block_scale_geometry("nvfp4", shape)
    codes = _exact_uint8_matrix(
        packed_codes,
        (*shape[:-1], geometry.k // 2),
        "NVFP4 packed codes",
    )
    scales = _exact_uint8_matrix(
        natural_scales,
        (*shape[:-1], geometry.groups_per_row),
        "NVFP4 scales",
    )
    invalid = ((scales & 0x80) != 0) | (scales == 0x7F)
    if bool(invalid.any()):
        raise ValueError("NVFP4 scales must be nonnegative finite E4M3FN words")
    divisor = _divisor_words(weight_divisor, geometry.matrices)
    swizzled = swizzle_nvfp4_scales(scales, shape)
    payload = bytearray(geometry.payload_bytes)
    payload[: geometry.code_plane_bytes] = codes.numpy().tobytes()
    payload[
        geometry.scale_plane_offset : geometry.scale_plane_offset
        + geometry.scale_plane_bytes
    ] = swizzled.numpy().tobytes()
    payload[geometry.weight_divisor_offset :] = divisor
    return bytes(payload)


def decode_nvfp4_words(
    payload: Payload,
    shape: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode exact packed code, natural scale, and divisor words."""

    geometry = block_scale_geometry("nvfp4", shape)
    if _payload_length(payload) != geometry.payload_bytes:
        raise ValueError(
            f"NVFP4 payload has {_payload_length(payload)} bytes, "
            f"expected {geometry.payload_bytes}"
        )
    raw = _payload_tensor(payload, torch.device("cpu"))
    codes = (
        raw[: geometry.code_plane_bytes].clone().reshape(*shape[:-1], geometry.k // 2)
    )
    stored_scales = raw[
        geometry.scale_plane_offset : geometry.scale_plane_offset
        + geometry.scale_plane_bytes
    ]
    scales = unswizzle_nvfp4_scales(stored_scales, shape)
    invalid = ((scales & 0x80) != 0) | (scales == 0x7F)
    if bool(invalid.any()):
        raise ValueError("NVFP4 scales must be nonnegative finite E4M3FN words")
    divisor_bytes = _divisor_words(
        bytes(raw[geometry.weight_divisor_offset :].numpy()), geometry.matrices
    )
    divisor = torch.frombuffer(bytearray(divisor_bytes), dtype=torch.float32).reshape(
        tuple(shape[:-2])
    )
    return codes, scales, divisor
