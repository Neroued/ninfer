"""Exact BF16, FP32, and INT32 words in contiguous little-endian layout."""

from __future__ import annotations

from math import prod
import sys
from typing import Sequence

import torch

from ..formats import DirectFormat
from ..layouts import _format, _shape
from ._tensor_bytes import Payload, _payload_length, _payload_tensor

_DIRECT_DTYPES = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "int32": torch.int32,
}


def encode_direct(tensor: torch.Tensor, format: str | DirectFormat) -> bytes:
    """Encode exact direct-format words; implicit dtype conversion is forbidden."""

    spec = _format(format)
    if not isinstance(spec, DirectFormat):
        raise ValueError("direct encoding requires BF16, FP32, or I32")
    expected_dtype = _DIRECT_DTYPES[spec.name]
    if tensor.dtype != expected_dtype:
        raise TypeError(
            f"{spec.name} encoding requires {expected_dtype}, got {tensor.dtype}"
        )
    if tensor.dim() > 16 or any(dim <= 0 for dim in tensor.shape):
        raise ValueError(
            "contiguous_le_v1 requires rank 0..16 with positive dimensions"
        )
    host = tensor.detach().contiguous().cpu().reshape(-1)
    raw = host.view(torch.uint8).reshape(host.numel(), spec.word_bytes)
    if sys.byteorder != "little":
        raw = raw.flip(1)
    return raw.reshape(-1).numpy().tobytes()


def decode_direct(
    payload: Payload,
    format: str | DirectFormat,
    shape: Sequence[int],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Decode exact direct-format words into their registered torch dtype."""

    spec = _format(format)
    if not isinstance(spec, DirectFormat):
        raise ValueError("direct decoding requires BF16, FP32, or I32")
    dims = _shape(shape)
    if len(dims) > 16:
        raise ValueError("contiguous_le_v1 supports rank 0 through 16")
    expected = prod(dims) * spec.word_bytes
    if _payload_length(payload) != expected:
        raise ValueError(
            f"direct payload has {_payload_length(payload)} bytes, expected {expected}"
        )
    target = torch.device(device)
    raw = _payload_tensor(payload, target)
    if sys.byteorder != "little" and target.type == "cpu":
        raw = raw.reshape(-1, spec.word_bytes).flip(1).contiguous().reshape(-1)
    return raw.view(_DIRECT_DTYPES[spec.name]).reshape(dims)
