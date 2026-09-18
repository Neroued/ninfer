"""Byte-buffer and exact tensor-word helpers shared by the concrete codecs."""

from __future__ import annotations

from typing import TypeAlias

import torch

Payload: TypeAlias = bytes | bytearray | memoryview | torch.Tensor


def _payload_length(payload: Payload) -> int:
    if isinstance(payload, torch.Tensor):
        if (
            payload.dtype != torch.uint8
            or payload.dim() != 1
            or not payload.is_contiguous()
        ):
            raise TypeError(
                "tensor payloads must be contiguous one-dimensional uint8 tensors"
            )
        return payload.numel()
    return memoryview(payload).nbytes


def _payload_tensor(payload: Payload, device: torch.device) -> torch.Tensor:
    if isinstance(payload, torch.Tensor):
        _payload_length(payload)
        return payload if payload.device == device else payload.to(device)
    raw = bytearray(payload)
    if not raw:
        return torch.empty(0, dtype=torch.uint8, device=device)
    host = torch.frombuffer(raw, dtype=torch.uint8)
    return host if device.type == "cpu" else host.to(device)


def _exact_uint8_matrix(
    tensor: torch.Tensor, shape: tuple[int, int], label: str
) -> torch.Tensor:
    if tensor.dtype != torch.uint8 or tuple(tensor.shape) != shape:
        raise TypeError(f"{label} must be uint8 with shape {shape}")
    return tensor.detach().contiguous().cpu()
