"""Persistent tensor layouts shared by NInfer converters and reference models.

This module maps already-selected numeric words to their registered byte layout.
It deliberately does not quantize floating-point source weights or assign model
roles to tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
import operator
from types import MappingProxyType
from typing import Mapping, Sequence


from .formats import (
    DirectFormat,
    Fp8BlockFormat,
    Fp8RowFormat,
    Nvfp4Format,
    Mxfp8Format,
    Nf4Format,
    NumericFormat,
    QuantFormat,
    get_format,
)

PLANE_ALIGNMENT = 256
K_ALIGNMENT = 128


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


@dataclass(frozen=True, slots=True)
class BlockScaleGeometry:
    matrices: int
    n: int
    k: int
    groups_per_row: int
    k_tiles: int
    code_plane_bytes: int
    scale_plane_offset: int
    scale_plane_bytes: int
    matrix_scale_offset: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class Fp8BlockGeometry:
    matrices: int
    n: int
    k: int
    scale_rows: int
    scale_columns: int
    code_plane_bytes: int
    scale_plane_offset: int
    scale_plane_bytes: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class RowScaleGeometry:
    n: int
    k: int
    code_plane_bytes: int
    scale_plane_offset: int
    scale_plane_bytes: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class RowGroupScaleGeometry:
    matrices: int
    n: int
    k: int
    groups_per_row: int
    code_plane_bytes: int
    scale_plane_offset: int
    scale_plane_bytes: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class Csf32Geometry:
    matrices: int
    n: int
    k: int
    scale_columns: int
    blocks_per_matrix: int
    pages_per_matrix: int
    offsets_per_matrix: int
    exception_count: int
    planes: tuple[tuple[str, int, int], ...]
    payload_bytes: int

    def plane(self, name: str) -> tuple[int, int]:
        for key, offset, size in self.planes:
            if key == name:
                return offset, size
        raise ValueError(f"unknown CSF plane: {name!r}")


@dataclass(frozen=True, slots=True)
class RowPagedGeometry:
    n: int
    k: int
    row_bytes: int
    rows_per_page: int
    page_count: int
    weight_divisor_offset: int
    payload_bytes: int

    def row_offset(self, row: int) -> int:
        if not 0 <= row < self.n:
            raise ValueError("row exceeds paged tensor")
        page, slot = divmod(row, self.rows_per_page)
        return page * 4096 + slot * self.row_bytes


CONTIGUOUS_LE_V1 = Layout("contiguous_le_v1", 256, frozenset(("bf16", "fp32", "int32")))
ROW_SPLIT_K128_V1 = Layout(
    "row_split_k128_v1",
    256,
    frozenset(("q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16", "q8_g32_fp16")),
)
BLOCK_SCALE_K16_M128X4_V1 = Layout(
    "block_scale_k16_m128x4_v1",
    256,
    frozenset(("nvfp4", "nvfp4_mul")),
)
BATCHED_BLOCK_SCALE_K16_M128X4_V1 = Layout(
    "batched_block_scale_k16_m128x4_v1", 256, frozenset(("nvfp4", "nvfp4_mul"))
)
BLOCK_SCALE_M128_K128_V1 = Layout(
    "block_scale_m128_k128_v1",
    256,
    frozenset(("fp8_e4m3fn_block128x128_bf16", "fp8_e4m3fn_block128x128_fp32")),
)
ROW_SCALE_V1 = Layout(
    "row_scale_v1",
    256,
    frozenset(("fp8_e4m3fn_row_bf16",)),
)
ROW_PAGED_G16_4K_V1 = Layout(
    "row_paged_g16_4k_v1", 4096, frozenset(("nf4_g16_e4m3fn",))
)
ROW_GROUP_SCALE_V1 = Layout("row_group_scale_v1", 256, frozenset(("mxfp8",)))
CSF_32X2_V1 = Layout("csf_32x2_v1", 256, frozenset(("nvfp4_mul",)))

LAYOUTS = MappingProxyType(
    {
        layout.name: layout
        for layout in (
            CONTIGUOUS_LE_V1,
            ROW_SPLIT_K128_V1,
            BLOCK_SCALE_K16_M128X4_V1,
            BATCHED_BLOCK_SCALE_K16_M128X4_V1,
            BLOCK_SCALE_M128_K128_V1,
            ROW_SCALE_V1,
            ROW_PAGED_G16_4K_V1,
            ROW_GROUP_SCALE_V1,
            CSF_32X2_V1,
        )
    }
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
        raise ValueError("row_split_k128_v1 requires a grouped quantized format")
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


def block_scale_geometry(
    format: str | Nvfp4Format, shape: Sequence[int]
) -> BlockScaleGeometry:
    spec = _format(format)
    if not isinstance(spec, Nvfp4Format):
        raise ValueError("block_scale_k16_m128x4_v1 requires NVFP4")
    dims = _shape(shape)
    if len(dims) not in (2, 3):
        raise ValueError("NVFP4 requires [N,K] or [E,N,K]")
    n, k = dims[-2:]
    matrices = dims[0] if len(dims) == 3 else 1
    if n % 128 != 0 or k % 64 != 0:
        raise ValueError(
            "block_scale_k16_m128x4_v1 requires N divisible by 128 "
            "and K divisible by 64"
        )
    code_plane_bytes = matrices * n * k // 2
    scale_plane_offset = align_up(code_plane_bytes, PLANE_ALIGNMENT)
    scale_plane_bytes = matrices * n * k // spec.group_size
    matrix_scale_offset = scale_plane_offset + scale_plane_bytes
    return BlockScaleGeometry(
        matrices=matrices,
        n=n,
        k=k,
        groups_per_row=k // spec.group_size,
        k_tiles=k // 64,
        code_plane_bytes=code_plane_bytes,
        scale_plane_offset=scale_plane_offset,
        scale_plane_bytes=scale_plane_bytes,
        matrix_scale_offset=matrix_scale_offset,
        payload_bytes=matrix_scale_offset + 4 * matrices,
    )


def fp8_block_geometry(
    format: str | Fp8BlockFormat, shape: Sequence[int]
) -> Fp8BlockGeometry:
    spec = _format(format)
    if not isinstance(spec, Fp8BlockFormat):
        raise ValueError("block_scale_m128_k128_v1 requires block-scaled FP8")
    dims = _shape(shape)
    if len(dims) not in (2, 3):
        raise ValueError("block-scaled FP8 requires [N,K] or [E,N,K]")
    n, k = dims[-2:]
    if n % spec.block_rows or k % spec.block_columns:
        raise ValueError("block-scaled FP8 requires N and K divisible by 128")
    matrices = dims[0] if len(dims) == 3 else 1
    code_bytes = matrices * n * k
    offset = align_up(code_bytes, PLANE_ALIGNMENT)
    scale_rows, scale_columns = n // spec.block_rows, k // spec.block_columns
    scale_bytes = matrices * scale_rows * scale_columns * spec.scale_bytes
    return Fp8BlockGeometry(
        matrices,
        n,
        k,
        scale_rows,
        scale_columns,
        code_bytes,
        offset,
        scale_bytes,
        offset + scale_bytes,
    )


def row_scale_geometry(
    format: str | Fp8RowFormat, shape: Sequence[int]
) -> RowScaleGeometry:
    spec = _format(format)
    if not isinstance(spec, Fp8RowFormat):
        raise ValueError("row_scale_v1 requires a row-scaled FP8 format")
    n, k = _shape(shape, rank=2)
    code_plane_bytes = n * k
    scale_plane_offset = align_up(code_plane_bytes, PLANE_ALIGNMENT)
    scale_plane_bytes = n * 2
    return RowScaleGeometry(
        n=n,
        k=k,
        code_plane_bytes=code_plane_bytes,
        scale_plane_offset=scale_plane_offset,
        scale_plane_bytes=scale_plane_bytes,
        payload_bytes=scale_plane_offset + scale_plane_bytes,
    )


def row_paged_geometry(
    format: str | Nf4Format, shape: Sequence[int]
) -> RowPagedGeometry:
    if not isinstance(_format(format), Nf4Format):
        raise ValueError("row_paged_g16_4k_v1 requires NF4 G16")
    n, k = _shape(shape, rank=2)
    if k % 16:
        raise ValueError("paged NF4 requires K divisible by 16")
    row_bytes = k // 2 + k // 16
    rows_per_page = 4096 // row_bytes
    if rows_per_page == 0:
        raise ValueError("encoded NF4 row exceeds 4096 bytes")
    pages = (n + rows_per_page - 1) // rows_per_page
    return RowPagedGeometry(
        n, k, row_bytes, rows_per_page, pages, pages * 4096, pages * 4096 + 4
    )


def row_group_scale_geometry(
    format: str | Mxfp8Format, shape: Sequence[int]
) -> RowGroupScaleGeometry:
    spec = _format(format)
    if not isinstance(spec, Mxfp8Format):
        raise ValueError("row_group_scale_v1 requires MXFP8")
    dims = _shape(shape)
    if len(dims) not in (2, 3):
        raise ValueError("MXFP8 requires [N,K] or [E,N,K]")
    n, k = dims[-2:]
    if k % spec.group_size:
        raise ValueError("MXFP8 requires K divisible by 32")
    matrices = dims[0] if len(dims) == 3 else 1
    codes = matrices * n * k
    scales = matrices * n * (k // spec.group_size)
    offset = align_up(codes, PLANE_ALIGNMENT)
    return RowGroupScaleGeometry(
        matrices, n, k, k // spec.group_size, codes, offset, scales, offset + scales
    )


def csf32_geometry(shape: Sequence[int], exception_count: int) -> Csf32Geometry:
    dims = _shape(shape)
    if len(dims) not in (2, 3):
        raise ValueError("CSF requires [N,K] or [E,N,K]")
    n, k = dims[-2:]
    matrices = dims[0] if len(dims) == 3 else 1
    if n % 32 or k % 16:
        raise ValueError("CSF requires N divisible by 32 and K divisible by 16")
    columns = k // 16
    if (
        type(exception_count) is not int
        or not 0 <= exception_count <= matrices * n * columns
        or 2 * exception_count >= 1 << 32
    ):
        raise ValueError("CSF exception count exceeds its logical or U32 byte domain")
    row_blocks, column_pairs = n // 32, (columns + 1) // 2
    blocks = row_blocks * column_pairs
    tiles = ((row_blocks + 3) // 4) * ((column_pairs + 1) // 2)
    pages = (tiles + 31) // 32
    sizes = (
        ("codes", matrices * n * k // 2),
        ("matrix_scale", matrices * 4),
        ("bases", matrices * n),
        ("nibbles", matrices * n * columns // 2),
        ("offsets", matrices * (blocks + pages) * 2),
        ("page_bases", matrices * pages * 4),
        ("exceptions", exception_count * 2),
    )
    offset, planes = 0, []
    for name, size in sizes:
        offset = align_up(offset, PLANE_ALIGNMENT)
        planes.append((name, offset, size))
        offset += size
    return Csf32Geometry(
        matrices,
        n,
        k,
        columns,
        blocks,
        pages,
        blocks + pages,
        exception_count,
        tuple(planes),
        offset,
    )


def encoded_size(
    layout: str | Layout,
    format: str | NumericFormat,
    shape: Sequence[int],
    layout_parameters: Mapping[str, int] | None = None,
) -> int:
    layout_spec = _layout(layout)
    numeric_spec = _format(format)
    parameters = {} if layout_parameters is None else layout_parameters
    if not isinstance(parameters, Mapping):
        raise TypeError("layout parameters must be a mapping")
    if layout_spec is CSF_32X2_V1:
        if set(parameters) != {"exception_count"}:
            raise ValueError(
                "CSF requires exactly the exception_count layout parameter"
            )
    elif parameters:
        raise ValueError(f"{layout_spec.name} does not accept layout parameters")
    if numeric_spec.name not in layout_spec.formats:
        raise ValueError(
            f"layout {layout_spec.name!r} does not accept format {numeric_spec.name!r}"
        )
    if layout_spec is CONTIGUOUS_LE_V1:
        if not isinstance(numeric_spec, DirectFormat):
            raise ValueError("contiguous_le_v1 requires a direct format")
        dims = _shape(shape)
        if len(dims) > 16:
            raise ValueError("contiguous_le_v1 supports rank 0 through 16")
        return prod(dims) * numeric_spec.word_bytes
    if layout_spec is ROW_SPLIT_K128_V1:
        if not isinstance(numeric_spec, QuantFormat):
            raise ValueError("row_split_k128_v1 requires a grouped quantized format")
        return row_split_geometry(numeric_spec, shape).payload_bytes
    if layout_spec in (BLOCK_SCALE_K16_M128X4_V1, BATCHED_BLOCK_SCALE_K16_M128X4_V1):
        if not isinstance(numeric_spec, Nvfp4Format):
            raise ValueError("block_scale_k16_m128x4_v1 requires NVFP4")
        _shape(shape, rank=2 if layout_spec is BLOCK_SCALE_K16_M128X4_V1 else 3)
        return block_scale_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is BLOCK_SCALE_M128_K128_V1:
        return fp8_block_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is ROW_SCALE_V1:
        if not isinstance(numeric_spec, Fp8RowFormat):
            raise ValueError("row_scale_v1 requires a row-scaled FP8 format")
        return row_scale_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is ROW_PAGED_G16_4K_V1:
        return row_paged_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is ROW_GROUP_SCALE_V1:
        return row_group_scale_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is CSF_32X2_V1:
        return csf32_geometry(shape, parameters["exception_count"]).payload_bytes
    raise ValueError(f"unsupported tensor layout: {layout_spec.name!r}")
