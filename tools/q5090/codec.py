"""q5090 decode path shared by the reader and reference model.

CPU and diagnostics reuse the canonical converter implementation. CUDA
ROW_SPLIT decoding uses fused torch.compile graphs so low-bit unpack does not
materialize the long chain of intermediate tensors used by the packer oracle.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from tools.q5090_convert import qtypes as qt
from tools.q5090_convert.layouts import decode_tensor as decode_tensor_oracle
from tools.q5090_convert.packing import row_split_plane_sizes


def _scales(scale_bytes: torch.Tensor) -> torch.Tensor:
    groups = scale_bytes.view(torch.float16).numel()
    return scale_bytes.view(torch.float16).reshape(groups, 1).float()


def _low(nibble: torch.Tensor, groups: int) -> torch.Tensor:
    packed = nibble.reshape(groups, 32).to(torch.int16)
    return torch.stack((packed & 0x0F, (packed >> 4) & 0x0F), dim=-1).reshape(groups, 64)


def _decode4(nibble, _high, scale_bytes, _byte_index, _shifts):
    scales = _scales(scale_bytes)
    unsigned = _low(nibble, scales.numel())
    codes = torch.where((unsigned & 8) != 0, unsigned - 16, unsigned).float()
    return (codes * scales).to(torch.bfloat16)


def _decode5(nibble, high, scale_bytes, byte_index, shifts):
    scales = _scales(scale_bytes)
    groups = scales.numel()
    upper = (high.reshape(groups, 8).index_select(1, byte_index).to(torch.int16) >> shifts) & 1
    unsigned = _low(nibble, groups) | (upper << 4)
    codes = torch.where((unsigned & 16) != 0, unsigned - 32, unsigned).float()
    return (codes * scales).to(torch.bfloat16)


def _decode6(nibble, high, scale_bytes, byte_index, shifts):
    scales = _scales(scale_bytes)
    groups = scales.numel()
    upper = (high.reshape(groups, 16).index_select(1, byte_index).to(torch.int16) >> shifts) & 3
    unsigned = _low(nibble, groups) | (upper << 4)
    codes = torch.where((unsigned & 32) != 0, unsigned - 64, unsigned).float()
    return (codes * scales).to(torch.bfloat16)


def _decode8(nibble, _high, scale_bytes, _byte_index, _shifts):
    scales = _scales(scale_bytes)
    codes = nibble.view(torch.int8).reshape(scales.numel(), -1).float()
    return (codes * scales).to(torch.bfloat16)


@lru_cache(maxsize=8)
def _compiled_decoder(bits: int):
    function = {4: _decode4, 5: _decode5, 6: _decode6, 8: _decode8}[bits]
    return torch.compile(function, dynamic=True, fullgraph=True)


@lru_cache(maxsize=32)
def _high_indices(
    device_type: str,
    device_index: int | None,
    bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device_type, device_index)
    if bits <= 4 or bits == 8:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return empty, empty
    bitpos = torch.arange(group_size, device=device, dtype=torch.int64) * (bits - 4)
    return torch.div(bitpos, 8, rounding_mode="floor"), bitpos % 8


def decode_tensor(
    payload: bytes | bytearray | memoryview | torch.Tensor,
    qtype: int,
    layout: int,
    logical: list[int],
    padded: list[int],
    device: torch.device | str = "cpu",
    *,
    output_dtype: torch.dtype = torch.float32,
    compiled: bool = True,
) -> torch.Tensor:
    device = torch.device(device)
    if layout == qt.LAYOUT_CONTIGUOUS and isinstance(payload, torch.Tensor):
        if payload.device != device:
            payload = payload.to(device)
        if qtype == qt.QT_BF16:
            out = payload.view(torch.bfloat16).reshape(logical)
            return out if output_dtype == torch.bfloat16 else out.to(output_dtype)
        if qtype == qt.QT_FP32:
            return payload.view(torch.float32).reshape(logical)
        if qtype == qt.QT_I32:
            return payload.view(torch.int32).reshape(logical)
        raise ValueError(f"unsupported CONTIGUOUS qtype {qtype}")
    if (
        layout != qt.LAYOUT_ROW_SPLIT
        or device.type != "cuda"
        or output_dtype != torch.bfloat16
    ):
        return decode_tensor_oracle(
            payload,
            qtype,
            layout,
            logical,
            padded,
            device=device,
            output_dtype=output_dtype,
        )
    if not isinstance(payload, torch.Tensor):
        host = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        payload = host.to(device)
    elif payload.device != device:
        payload = payload.to(device)
    spec = qt.QUANT_SPECS[qtype]
    n, k = logical
    _, kp = padded
    groups_per_row = kp // spec.group_size
    nibble_bytes, high_off, high_bytes, scale_off, scale_bytes, need = row_split_plane_sizes(
        n,
        groups_per_row,
        spec.nibble_bytes_per_group,
        spec.high_bytes_per_group,
    )
    if payload.numel() < need:
        raise ValueError(f"ROW_SPLIT payload too short: {payload.numel()} < {need}")
    byte_index, shifts = _high_indices(
        device.type, device.index, spec.bits, spec.group_size
    )
    function = (
        _compiled_decoder(spec.bits)
        if compiled
        else {4: _decode4, 5: _decode5, 6: _decode6, 8: _decode8}[spec.bits]
    )
    decoded = function(
        payload[:nibble_bytes],
        payload[high_off: high_off + high_bytes],
        payload[scale_off: scale_off + scale_bytes],
        byte_index,
        shifts,
    ).reshape(n, kp)[:, :k]
    return decoded if output_dtype == torch.bfloat16 else decoded.to(output_dtype)
