"""Persistent tensor layouts shared by NInfer converters and reference models.

This module maps already-selected numeric words to their registered byte layout.
It deliberately does not quantize floating-point source weights or assign model
roles to tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import prod
import operator
import sys
from types import MappingProxyType
from typing import Sequence, TypeAlias

import torch

from .numeric import (
    DirectFormat,
    NumericFormat,
    QuantFormat,
    get_format,
)


PLANE_ALIGNMENT = 256
K_ALIGNMENT = 128
_PACK_TEMP_BYTES = 4 * 1024 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class Layout:
    name: str
    alignment: int
    formats: frozenset[str]


@dataclass(frozen=True, slots=True)
class RowSplitGeometry:
    n: int
    k: int
    k_pad: int
    groups_per_row: int
    base_bytes_per_group: int
    high_bytes_per_group: int
    base_row_bytes: int
    high_row_bytes: int
    scale_row_bytes: int
    base_offset: int
    base_bytes: int
    high_offset: int
    high_bytes: int
    scale_offset: int
    scale_bytes: int
    payload_bytes: int


Plane: TypeAlias = bytes | bytearray | memoryview | torch.Tensor
Payload: TypeAlias = bytes | bytearray | memoryview | torch.Tensor


@dataclass(frozen=True, slots=True)
class RowPlanes:
    """Three plane spans for one or more rows, without inter-plane padding."""

    base: Plane
    high: Plane
    scale: Plane
    rows: int


CONTIGUOUS_LE_V1 = Layout(
    "contiguous-le-v1", 256, frozenset(("BF16", "FP32", "I32"))
)
ROW_SPLIT_K128_V1 = Layout(
    "row-split-k128-v1",
    256,
    frozenset(("Q4G64_F16S", "Q5G64_F16S", "Q6G64_F16S", "W8G32_F16S")),
)

LAYOUTS = MappingProxyType(
    {layout.name: layout for layout in (CONTIGUOUS_LE_V1, ROW_SPLIT_K128_V1)}
)


def align_up(value: int, alignment: int) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError("align_up requires a nonnegative value and positive alignment")
    return (value + alignment - 1) // alignment * alignment


def get_layout(name: str) -> Layout:
    try:
        return LAYOUTS[name]
    except KeyError:
        raise ValueError(f"unknown tensor layout: {name!r}") from None


def _format(value: str | NumericFormat) -> NumericFormat:
    if isinstance(value, str):
        return get_format(value)
    registered = get_format(value.name)
    if value != registered:
        raise ValueError(f"numeric format does not match registered {value.name!r}")
    return registered


def _layout(value: str | Layout) -> Layout:
    if isinstance(value, str):
        return get_layout(value)
    registered = get_layout(value.name)
    if value != registered:
        raise ValueError(f"layout does not match registered {value.name!r}")
    return registered


def _shape(value: Sequence[int], *, rank: int | None = None) -> tuple[int, ...]:
    dims = []
    for dim in value:
        if isinstance(dim, bool):
            raise ValueError("shape dimensions must be positive integers")
        try:
            item = operator.index(dim)
        except TypeError:
            raise ValueError("shape dimensions must be positive integers") from None
        if item <= 0:
            raise ValueError("shape dimensions must be positive integers")
        dims.append(item)
    result = tuple(dims)
    if rank is not None and len(result) != rank:
        raise ValueError(f"layout requires rank {rank}, got rank {len(result)}")
    return result


def row_split_geometry(
    format: str | QuantFormat, shape: Sequence[int]
) -> RowSplitGeometry:
    spec = _format(format)
    if not isinstance(spec, QuantFormat):
        raise ValueError("row-split-k128-v1 requires a grouped quantized format")
    n, k = _shape(shape, rank=2)
    k_pad = align_up(k, K_ALIGNMENT)
    groups_per_row = k_pad // spec.group_size
    base_bytes_per_group = spec.group_size if spec.bits == 8 else spec.group_size // 2
    high_bytes_per_group = (
        0 if spec.bits in (4, 8) else spec.group_size * (spec.bits - 4) // 8
    )
    base_row_bytes = groups_per_row * base_bytes_per_group
    high_row_bytes = groups_per_row * high_bytes_per_group
    scale_row_bytes = groups_per_row * 2
    base_bytes = n * base_row_bytes
    high_bytes = n * high_row_bytes
    scale_bytes = n * scale_row_bytes
    high_offset = align_up(base_bytes, PLANE_ALIGNMENT)
    scale_offset = high_offset + align_up(high_bytes, PLANE_ALIGNMENT)
    return RowSplitGeometry(
        n=n,
        k=k,
        k_pad=k_pad,
        groups_per_row=groups_per_row,
        base_bytes_per_group=base_bytes_per_group,
        high_bytes_per_group=high_bytes_per_group,
        base_row_bytes=base_row_bytes,
        high_row_bytes=high_row_bytes,
        scale_row_bytes=scale_row_bytes,
        base_offset=0,
        base_bytes=base_bytes,
        high_offset=high_offset,
        high_bytes=high_bytes,
        scale_offset=scale_offset,
        scale_bytes=scale_bytes,
        payload_bytes=scale_offset + scale_bytes,
    )


def encoded_size(
    layout: str | Layout,
    format: str | NumericFormat,
    shape: Sequence[int],
) -> int:
    layout_spec = _layout(layout)
    numeric_spec = _format(format)
    if numeric_spec.name not in layout_spec.formats:
        raise ValueError(
            f"layout {layout_spec.name!r} does not accept format {numeric_spec.name!r}"
        )
    if layout_spec is CONTIGUOUS_LE_V1:
        if not isinstance(numeric_spec, DirectFormat):
            raise ValueError("contiguous-le-v1 requires a direct format")
        dims = _shape(shape)
        if len(dims) > 16:
            raise ValueError("contiguous-le-v1 supports rank 0 through 16")
        return prod(dims) * numeric_spec.word_bytes
    if layout_spec is ROW_SPLIT_K128_V1:
        if not isinstance(numeric_spec, QuantFormat):
            raise ValueError("row-split-k128-v1 requires a grouped quantized format")
        return row_split_geometry(numeric_spec, shape).payload_bytes
    raise ValueError(f"unsupported tensor layout: {layout_spec.name!r}")


_DIRECT_DTYPES = {
    "BF16": torch.bfloat16,
    "FP32": torch.float32,
    "I32": torch.int32,
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
        raise ValueError("contiguous-le-v1 requires rank 0..16 with positive dimensions")
    host = tensor.detach().contiguous().cpu().reshape(-1)
    raw = host.view(torch.uint8).reshape(host.numel(), spec.word_bytes)
    if sys.byteorder != "little":
        raw = raw.flip(1)
    return raw.reshape(-1).numpy().tobytes()


def _payload_length(payload: Payload) -> int:
    if isinstance(payload, torch.Tensor):
        if payload.dtype != torch.uint8 or payload.dim() != 1 or not payload.is_contiguous():
            raise TypeError("tensor payloads must be contiguous one-dimensional uint8 tensors")
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
        raise ValueError("contiguous-le-v1 supports rank 0 through 16")
    expected = prod(dims) * spec.word_bytes
    if _payload_length(payload) != expected:
        raise ValueError(f"direct payload has {_payload_length(payload)} bytes, expected {expected}")
    target = torch.device(device)
    raw = _payload_tensor(payload, target)
    if sys.byteorder != "little" and target.type == "cpu":
        raw = raw.reshape(-1, spec.word_bytes).flip(1).contiguous().reshape(-1)
    return raw.view(_DIRECT_DTYPES[spec.name]).reshape(dims)


def _pack_low_nibbles(codes: torch.Tensor) -> torch.Tensor:
    groups, group_size = codes.shape
    out = torch.empty((groups, group_size // 2), dtype=torch.uint8, device=codes.device)
    chunk = max(1, _PACK_TEMP_BYTES // max(1, group_size * 2))
    for begin in range(0, groups, chunk):
        end = min(groups, begin + chunk)
        unsigned = codes[begin:end].to(torch.int16) & 0x0F
        out[begin:end] = (
            unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)
        ).to(torch.uint8)
    return out


def _pack_high_bits(codes: torch.Tensor, bits: int) -> torch.Tensor:
    groups, group_size = codes.shape
    high_bits = bits - 4
    bytes_per_group = group_size * high_bits // 8
    out = torch.empty((groups, bytes_per_group), dtype=torch.uint8, device=codes.device)
    shifts = torch.arange(high_bits, device=codes.device, dtype=torch.int32)
    bit_weights = 1 << torch.arange(8, device=codes.device, dtype=torch.int32)
    chunk = max(
        1,
        _PACK_TEMP_BYTES // max(1, group_size * high_bits * torch.int32.itemsize),
    )
    mask = (1 << bits) - 1
    for begin in range(0, groups, chunk):
        end = min(groups, begin + chunk)
        upper = ((codes[begin:end].to(torch.int32) & mask) >> 4) & (
            (1 << high_bits) - 1
        )
        unpacked = ((upper.unsqueeze(-1) >> shifts) & 1).reshape(
            end - begin, bytes_per_group, 8
        )
        out[begin:end] = (unpacked * bit_weights).sum(-1).to(torch.uint8)
    return out


def _pack_codes(codes: torch.Tensor, spec: QuantFormat) -> tuple[torch.Tensor, torch.Tensor]:
    if spec.bits == 8:
        return (
            codes.contiguous().view(torch.uint8),
            torch.empty((codes.shape[0], 0), dtype=torch.uint8, device=codes.device),
        )
    base = _pack_low_nibbles(codes)
    high = (
        torch.empty((codes.shape[0], 0), dtype=torch.uint8, device=codes.device)
        if spec.bits == 4
        else _pack_high_bits(codes, spec.bits)
    )
    return base, high


def encode_row_split(
    codes: torch.Tensor,
    scales: torch.Tensor,
    format: str | QuantFormat,
    shape: Sequence[int],
) -> bytes:
    """Pack physical grouped codes and scales into a row-split payload.

    ``codes`` has shape ``[N, groups_per_row, group_size]`` and dtype int8.
    ``scales`` has shape ``[N, groups_per_row]`` and dtype float16. The caller
    owns quantization and the required zero-filled physical padding groups.
    """

    spec = _format(format)
    if not isinstance(spec, QuantFormat):
        raise ValueError("row-split encoding requires a grouped quantized format")
    geometry = row_split_geometry(spec, shape)
    expected_codes = (geometry.n, geometry.groups_per_row, spec.group_size)
    expected_scales = (geometry.n, geometry.groups_per_row)
    if codes.dtype != torch.int8 or tuple(codes.shape) != expected_codes:
        raise TypeError(f"codes must be int8 with shape {expected_codes}")
    if scales.dtype != torch.float16 or tuple(scales.shape) != expected_scales:
        raise TypeError(f"scales must be float16 with shape {expected_scales}")
    if codes.device != scales.device:
        raise ValueError("codes and scales must be on the same device")
    grouped = codes.reshape(-1, spec.group_size)
    base, high = _pack_codes(grouped, spec)
    planes = RowPlanes(
        base.reshape(-1),
        high.reshape(-1),
        scales.contiguous().view(torch.uint8).reshape(-1),
        geometry.n,
    )
    payload = assemble_row_planes(planes, spec, geometry.k)
    assert isinstance(payload, torch.Tensor)
    return payload.cpu().numpy().tobytes()


def _byte_view(payload: bytes | bytearray | memoryview) -> memoryview:
    view = memoryview(payload)
    if not view.c_contiguous:
        raise TypeError("byte payloads must be contiguous")
    return view.cast("B")


def split_row_planes(
    payload: Payload,
    geometry: RowSplitGeometry,
    row_begin: int = 0,
    row_count: int | None = None,
) -> RowPlanes:
    """Return non-owning plane spans for a consecutive row range."""

    if _payload_length(payload) != geometry.payload_bytes:
        raise ValueError(
            f"row-split payload has {_payload_length(payload)} bytes, "
            f"expected {geometry.payload_bytes}"
        )
    count = geometry.n - row_begin if row_count is None else row_count
    if row_begin < 0 or count <= 0 or row_begin + count > geometry.n:
        raise IndexError("row range is outside the row-split tensor")
    source: memoryview | torch.Tensor
    source = payload if isinstance(payload, torch.Tensor) else _byte_view(payload)
    base_begin = geometry.base_offset + row_begin * geometry.base_row_bytes
    high_begin = geometry.high_offset + row_begin * geometry.high_row_bytes
    scale_begin = geometry.scale_offset + row_begin * geometry.scale_row_bytes
    return RowPlanes(
        source[base_begin : base_begin + count * geometry.base_row_bytes],
        source[high_begin : high_begin + count * geometry.high_row_bytes],
        source[scale_begin : scale_begin + count * geometry.scale_row_bytes],
        count,
    )


def _sequence_rows(rows: Sequence[int], n: int) -> list[int]:
    result = []
    for row in rows:
        if isinstance(row, bool):
            raise TypeError("row indices must be integers")
        try:
            index = operator.index(row)
        except TypeError:
            raise TypeError("row indices must be integers") from None
        if index < 0 or index >= n:
            raise IndexError("row index is outside the row-split tensor")
        result.append(index)
    if not result:
        raise ValueError("at least one row is required")
    return result


def gather_row_planes(
    payload: Payload,
    geometry: RowSplitGeometry,
    rows: Sequence[int] | torch.Tensor,
) -> RowPlanes:
    """Materialize arbitrary rows independently in each physical plane."""

    full = split_row_planes(payload, geometry)
    if isinstance(payload, torch.Tensor):
        if isinstance(rows, torch.Tensor):
            if rows.dim() != 1 or rows.numel() == 0:
                raise ValueError("rows must be a nonempty one-dimensional tensor")
            if rows.dtype not in (torch.int32, torch.int64):
                raise TypeError("row-index tensors must use int32 or int64")
            indices = rows.to(device=payload.device, dtype=torch.long)
        else:
            indices = torch.tensor(
                _sequence_rows(rows, geometry.n), dtype=torch.long, device=payload.device
            )
        count = indices.numel()

        def gather(plane: torch.Tensor, row_bytes: int) -> torch.Tensor:
            if row_bytes == 0:
                return plane[:0]
            return plane.reshape(geometry.n, row_bytes).index_select(0, indices).reshape(-1)

        assert isinstance(full.base, torch.Tensor)
        assert isinstance(full.high, torch.Tensor)
        assert isinstance(full.scale, torch.Tensor)
        return RowPlanes(
            gather(full.base, geometry.base_row_bytes),
            gather(full.high, geometry.high_row_bytes),
            gather(full.scale, geometry.scale_row_bytes),
            count,
        )

    if isinstance(rows, torch.Tensor):
        indices_list = _sequence_rows(rows.detach().cpu().tolist(), geometry.n)
    else:
        indices_list = _sequence_rows(rows, geometry.n)

    def gather_bytes(plane: Plane, row_bytes: int) -> bytes:
        if row_bytes == 0:
            return b""
        return b"".join(
            plane[index * row_bytes : (index + 1) * row_bytes]
            for index in indices_list
        )

    return RowPlanes(
        gather_bytes(full.base, geometry.base_row_bytes),
        gather_bytes(full.high, geometry.high_row_bytes),
        gather_bytes(full.scale, geometry.scale_row_bytes),
        len(indices_list),
    )


def _plane_length(plane: Plane) -> int:
    if isinstance(plane, torch.Tensor):
        if plane.dtype != torch.uint8 or plane.dim() != 1 or not plane.is_contiguous():
            raise TypeError("row planes must be contiguous one-dimensional uint8 tensors")
        return plane.numel()
    return memoryview(plane).nbytes


def _validate_planes(planes: RowPlanes, geometry: RowSplitGeometry) -> None:
    if planes.rows != geometry.n:
        raise ValueError(f"row planes contain {planes.rows} rows, expected {geometry.n}")
    actual = (
        _plane_length(planes.base),
        _plane_length(planes.high),
        _plane_length(planes.scale),
    )
    expected = (geometry.base_bytes, geometry.high_bytes, geometry.scale_bytes)
    if actual != expected:
        raise ValueError(f"row plane sizes are {actual}, expected {expected}")


def assemble_row_planes(
    planes: RowPlanes,
    format: str | QuantFormat,
    logical_k: int,
) -> bytes | torch.Tensor:
    """Assemble plane spans as a standalone row-split payload."""

    spec = _format(format)
    if not isinstance(spec, QuantFormat):
        raise ValueError("row plane assembly requires a grouped quantized format")
    geometry = row_split_geometry(spec, (planes.rows, logical_k))
    _validate_planes(planes, geometry)
    values = (planes.base, planes.high, planes.scale)
    tensor_mode = isinstance(planes.base, torch.Tensor)
    if any(isinstance(value, torch.Tensor) != tensor_mode for value in values):
        raise TypeError("row planes must all be tensors or all be byte buffers")
    if tensor_mode:
        assert isinstance(planes.base, torch.Tensor)
        assert isinstance(planes.high, torch.Tensor)
        assert isinstance(planes.scale, torch.Tensor)
        if planes.high.device != planes.base.device or planes.scale.device != planes.base.device:
            raise ValueError("tensor row planes must be on the same device")
        payload = torch.zeros(
            geometry.payload_bytes, dtype=torch.uint8, device=planes.base.device
        )
        payload[: geometry.base_bytes].copy_(planes.base)
        if geometry.high_bytes:
            payload[
                geometry.high_offset : geometry.high_offset + geometry.high_bytes
            ].copy_(planes.high)
        payload[
            geometry.scale_offset : geometry.scale_offset + geometry.scale_bytes
        ].copy_(planes.scale)
        return payload

    payload = bytearray(geometry.payload_bytes)
    payload[: geometry.base_bytes] = bytes(planes.base)
    if geometry.high_bytes:
        payload[
            geometry.high_offset : geometry.high_offset + geometry.high_bytes
        ] = bytes(planes.high)
    payload[
        geometry.scale_offset : geometry.scale_offset + geometry.scale_bytes
    ] = bytes(planes.scale)
    return bytes(payload)


def _plane_tensor(plane: Plane, device: torch.device) -> torch.Tensor:
    if isinstance(plane, torch.Tensor):
        _plane_length(plane)
        return plane if plane.device == device else plane.to(device)
    raw = bytearray(plane)
    if not raw:
        return torch.empty(0, dtype=torch.uint8, device=device)
    host = torch.frombuffer(raw, dtype=torch.uint8)
    return host if device.type == "cpu" else host.to(device)


def _planes_on_device(
    source: Payload | RowPlanes,
    geometry: RowSplitGeometry,
    device: torch.device,
) -> RowPlanes:
    if isinstance(source, RowPlanes):
        _validate_planes(source, geometry)
        return RowPlanes(
            _plane_tensor(source.base, device),
            _plane_tensor(source.high, device),
            _plane_tensor(source.scale, device),
            source.rows,
        )
    if _payload_length(source) != geometry.payload_bytes:
        raise ValueError(
            f"row-split payload has {_payload_length(source)} bytes, "
            f"expected {geometry.payload_bytes}"
        )
    resident = _payload_tensor(source, device)
    return split_row_planes(resident, geometry)


@lru_cache(maxsize=32)
def _high_indices(
    device_type: str,
    device_index: int | None,
    bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device_type) if device_index is None else torch.device(
        device_type, device_index
    )
    if bits in (4, 8):
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    bit_positions = torch.arange(group_size, device=device, dtype=torch.long) * (bits - 4)
    return bit_positions // 8, bit_positions % 8


def _unpack_codes(
    planes: RowPlanes,
    spec: QuantFormat,
    geometry: RowSplitGeometry,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert isinstance(planes.base, torch.Tensor)
    assert isinstance(planes.high, torch.Tensor)
    assert isinstance(planes.scale, torch.Tensor)
    groups = geometry.n * geometry.groups_per_row
    scales = planes.scale.view(torch.float16).reshape(
        geometry.n, geometry.groups_per_row
    )
    if spec.bits == 8:
        codes = planes.base.view(torch.int8).reshape(
            geometry.n, geometry.groups_per_row, spec.group_size
        )
        return scales, codes
    packed = planes.base.reshape(groups, geometry.base_bytes_per_group).to(torch.int16)
    low = torch.empty((groups, spec.group_size), dtype=torch.int16, device=packed.device)
    low[:, 0::2] = packed & 0x0F
    low[:, 1::2] = (packed >> 4) & 0x0F
    if spec.bits == 4:
        unsigned = low
    else:
        byte_indices, shifts = _high_indices(
            packed.device.type, packed.device.index, spec.bits, spec.group_size
        )
        high = (
            planes.high.reshape(groups, geometry.high_bytes_per_group)
            .index_select(1, byte_indices)
            .to(torch.int16)
            >> shifts
        ) & ((1 << (spec.bits - 4)) - 1)
        unsigned = low | (high << 4)
    sign = 1 << (spec.bits - 1)
    codes = torch.where((unsigned & sign) != 0, unsigned - (1 << spec.bits), unsigned)
    return scales, codes.to(torch.int8).reshape(
        geometry.n, geometry.groups_per_row, spec.group_size
    )


def decode_row_split_codes(
    source: Payload | RowPlanes,
    format: str | QuantFormat,
    shape: Sequence[int],
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode binary16 scales and physical signed-code groups.

    The returned shapes are ``[N, groups_per_row]`` and
    ``[N, groups_per_row, group_size]``. Logical K padding is retained here so
    kernels can consume group geometry directly; :func:`dequantize_row_split`
    removes it from the reconstructed tensor.
    """

    spec = _format(format)
    if not isinstance(spec, QuantFormat):
        raise ValueError("row-split decoding requires a grouped quantized format")
    geometry = row_split_geometry(spec, shape)
    planes = _planes_on_device(source, geometry, torch.device(device))
    return _unpack_codes(planes, spec, geometry)


def _scales(scale: torch.Tensor) -> torch.Tensor:
    return scale.view(torch.float16).reshape(-1, 1).float()


def _low_g64(base: torch.Tensor, groups: int) -> torch.Tensor:
    packed = base.reshape(groups, 32).to(torch.int16)
    return torch.stack((packed & 0x0F, (packed >> 4) & 0x0F), dim=-1).reshape(
        groups, 64
    )


def _dequant4(base, _high, scale, _byte_indices, _shifts):
    scales = _scales(scale)
    unsigned = _low_g64(base, scales.numel())
    codes = torch.where((unsigned & 8) != 0, unsigned - 16, unsigned).float()
    return (codes * scales).to(torch.bfloat16)


def _dequant5(base, high, scale, byte_indices, shifts):
    scales = _scales(scale)
    groups = scales.numel()
    upper = (
        high.reshape(groups, 8).index_select(1, byte_indices).to(torch.int16) >> shifts
    ) & 1
    unsigned = _low_g64(base, groups) | (upper << 4)
    codes = torch.where((unsigned & 16) != 0, unsigned - 32, unsigned).float()
    return (codes * scales).to(torch.bfloat16)


def _dequant6(base, high, scale, byte_indices, shifts):
    scales = _scales(scale)
    groups = scales.numel()
    upper = (
        high.reshape(groups, 16).index_select(1, byte_indices).to(torch.int16) >> shifts
    ) & 3
    unsigned = _low_g64(base, groups) | (upper << 4)
    codes = torch.where((unsigned & 32) != 0, unsigned - 64, unsigned).float()
    return (codes * scales).to(torch.bfloat16)


def _dequant8(base, _high, scale, _byte_indices, _shifts):
    scales = _scales(scale)
    codes = base.view(torch.int8).reshape(scales.numel(), -1).float()
    return (codes * scales).to(torch.bfloat16)


_EAGER_DEQUANTIZERS = {
    4: _dequant4,
    5: _dequant5,
    6: _dequant6,
    8: _dequant8,
}


@lru_cache(maxsize=4)
def _compiled_dequantizer(bits: int):
    return torch.compile(_EAGER_DEQUANTIZERS[bits], dynamic=True, fullgraph=True)


def dequantize_row_split(
    source: Payload | RowPlanes,
    format: str | QuantFormat,
    shape: Sequence[int],
    device: torch.device | str = "cpu",
    *,
    dtype: torch.dtype = torch.bfloat16,
    compiled: bool = False,
) -> torch.Tensor:
    """Reconstruct a logical ``[N,K]`` tensor from a row-split payload."""

    spec = _format(format)
    if not isinstance(spec, QuantFormat):
        raise ValueError("row-split dequantization requires a grouped quantized format")
    if not dtype.is_floating_point:
        raise TypeError("dequantized output dtype must be floating point")
    geometry = row_split_geometry(spec, shape)
    target = torch.device(device)
    planes = _planes_on_device(source, geometry, target)
    if dtype == torch.bfloat16:
        byte_indices, shifts = _high_indices(
            target.type, target.index, spec.bits, spec.group_size
        )
        function = (
            _compiled_dequantizer(spec.bits)
            if compiled and target.type == "cuda"
            else _EAGER_DEQUANTIZERS[spec.bits]
        )
        assert isinstance(planes.base, torch.Tensor)
        assert isinstance(planes.high, torch.Tensor)
        assert isinstance(planes.scale, torch.Tensor)
        physical = function(
            planes.base, planes.high, planes.scale, byte_indices, shifts
        ).reshape(geometry.n, geometry.k_pad)
        return physical[:, : geometry.k]
    scales, codes = _unpack_codes(planes, spec, geometry)
    physical = (
        codes.float() * scales.float().unsqueeze(-1)
    ).reshape(geometry.n, geometry.k_pad)
    return physical[:, : geometry.k].to(dtype)


__all__ = [
    "CONTIGUOUS_LE_V1",
    "K_ALIGNMENT",
    "LAYOUTS",
    "Layout",
    "PLANE_ALIGNMENT",
    "ROW_SPLIT_K128_V1",
    "RowPlanes",
    "RowSplitGeometry",
    "align_up",
    "assemble_row_planes",
    "decode_direct",
    "decode_row_split_codes",
    "dequantize_row_split",
    "encode_direct",
    "encode_row_split",
    "encoded_size",
    "gather_row_planes",
    "get_layout",
    "row_split_geometry",
    "split_row_planes",
]
