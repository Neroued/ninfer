"""Map typed values or exact codes/scales to complete parent objects and writer ranges.

This layer owns codec selection, plane offsets, and padding within an object.
ArtifactWriter owns file placement, sharding, write coverage, and publication.
"""

from __future__ import annotations

import numpy as np
import torch

from .codecs.direct import encode_direct
from .codecs.fp8_block import validate_fp8_block_words
from .codecs.fp8_row import encode_fp8_row_scaled
from .codecs.nvfp4 import matrix_scale_words, swizzle_nvfp4_scales
from .codecs.mxfp8 import validate_mxfp8_words
from .codecs._tensor_bytes import _exact_uint8_matrix
from .codecs.nf4 import divisor_word, nf4_rows
from .codecs.row_split import encode_row_split, split_row_planes
from .formats import (
    DirectFormat,
    Fp8BlockFormat,
    Fp8RowFormat,
    Nvfp4Format,
    Mxfp8Format,
    Nf4Format,
    QuantFormat,
    get_format,
)
from .layouts import (
    block_scale_geometry,
    fp8_block_geometry,
    row_scale_geometry,
    row_split_geometry,
    row_paged_geometry,
    row_group_scale_geometry,
    csf32_geometry,
)
from .schema import TensorObject
from .writer import ArtifactWriter


class TensorOutput:
    def __init__(self, writer: ArtifactWriter, object_id: str):
        obj = writer.by_id[object_id]
        if not isinstance(obj, TensorObject):
            raise TypeError(f"{object_id}: expected tensor output")
        self.writer = writer
        self.object = obj
        self.format = get_format(obj.format)
        self._padding_initialized = False
        self._matrix_scales: dict[int, bytes] = {}
        self._next_row = 0
        self._pending_rows = None
        self._pages_written = 0

    def write_bytes(self, offset: int, data: bytes | memoryview) -> None:
        self.writer.write_region(self.object.id, offset, data)

    def write_values(self, element_begin: int, values: torch.Tensor) -> None:
        if not isinstance(self.format, DirectFormat):
            raise TypeError(
                "choose the numerical encoder before writing quantized output"
            )
        self.write_bytes(
            element_begin * self.format.word_bytes, encode_direct(values, self.format)
        )

    def _padding(self) -> None:
        if self._padding_initialized:
            return
        obj = self.object
        if obj.layout == "csf_32x2_v1":
            g = csf32_geometry(obj.shape, obj.layout_parameters["exception_count"])
            previous = 0
            gaps = []
            for _, offset, size in g.planes:
                gaps.append((previous, offset))
                previous = offset + size
        elif isinstance(self.format, QuantFormat):
            g = row_split_geometry(self.format, obj.shape)
            gaps = (
                (g.base_bytes, g.high_offset),
                (g.high_offset + g.high_bytes, g.scale_offset),
            )
        elif isinstance(self.format, Fp8RowFormat):
            g = row_scale_geometry(self.format, obj.shape)
            gaps = ((g.code_plane_bytes, g.scale_plane_offset),)
        elif isinstance(self.format, Fp8BlockFormat):
            g = fp8_block_geometry(self.format, obj.shape)
            gaps = ((g.code_plane_bytes, g.scale_plane_offset),)
        elif isinstance(self.format, Mxfp8Format):
            g = row_group_scale_geometry(self.format, obj.shape)
            gaps = ((g.code_plane_bytes, g.scale_plane_offset),)
        elif isinstance(self.format, Nvfp4Format):
            g = block_scale_geometry(self.format, obj.shape)
            gaps = ((g.code_plane_bytes, g.scale_plane_offset),)
        else:
            gaps = ()
        for begin, end in gaps:
            if begin < end:
                self.writer.write_zeros(obj.id, begin, end - begin)
        self._padding_initialized = True

    def write_csf_plane(
        self, name: str, byte_offset: int, data: bytes | bytearray | memoryview
    ) -> None:
        if self.object.layout != "csf_32x2_v1":
            raise TypeError("CSF plane writes require a CSF tensor")
        g = csf32_geometry(
            self.object.shape, self.object.layout_parameters["exception_count"]
        )
        offset, size = g.plane(name)
        raw = memoryview(data).cast("B")
        if byte_offset < 0 or byte_offset > size or len(raw) > size - byte_offset:
            raise ValueError(f"CSF {name} write exceeds its plane")
        self._padding()
        if raw:
            self.write_bytes(offset + byte_offset, raw)

    def write_codes(
        self,
        row_begin: int,
        codes: torch.Tensor,
        scales: torch.Tensor,
        weight_scale: bytes | None = None,
        *,
        matrix: int = 0,
    ) -> None:
        obj = self.object
        if obj.layout == "csf_32x2_v1":
            raise TypeError("CSF output requires compressed plane writes")
        matrices = obj.shape[0] if len(obj.shape) == 3 else 1
        if len(obj.shape) not in (2, 3) or not 0 <= matrix < matrices:
            raise ValueError(f"{obj.id}: invalid encoded matrix index {matrix}")
        n, k = obj.shape[-2:]
        if not 0 <= row_begin < n:
            raise ValueError(f"{obj.id}: invalid encoded row origin {row_begin}")
        rows = codes.shape[0]
        if rows <= 0 or row_begin + rows > n:
            raise ValueError(f"{obj.id}: encoded rows exceed matrix")
        parent_row = matrix * n + row_begin
        self._padding()
        if isinstance(self.format, Nf4Format):
            self._write_nf4(row_begin, codes, scales, weight_scale)
        elif isinstance(self.format, QuantFormat):
            g = row_split_geometry(self.format, obj.shape)
            block = encode_row_split(codes, scales, self.format, (rows, k))
            planes = split_row_planes(block, row_split_geometry(self.format, (rows, k)))
            self.write_bytes(g.base_offset + row_begin * g.base_row_bytes, planes.base)
            self.write_bytes(g.high_offset + row_begin * g.high_row_bytes, planes.high)
            self.write_bytes(
                g.scale_offset + row_begin * g.scale_row_bytes, planes.scale
            )
        elif isinstance(self.format, Fp8RowFormat):
            g = row_scale_geometry(self.format, obj.shape)
            local = row_scale_geometry(self.format, (rows, k))
            block = memoryview(encode_fp8_row_scaled(codes, scales, (rows, k)))
            self.write_bytes(row_begin * k, block[: local.code_plane_bytes])
            self.write_bytes(
                g.scale_plane_offset + row_begin * 2, block[local.scale_plane_offset :]
            )
        elif isinstance(self.format, Fp8BlockFormat):
            if row_begin % 128 or rows % 128:
                raise ValueError(
                    f"{obj.id}: FP8 block output needs whole 128-row tiles"
                )
            g = fp8_block_geometry(self.format, obj.shape)
            q = _exact_uint8_matrix(codes, (rows, k), "block-scaled FP8 codes")
            dtype = (
                torch.bfloat16 if self.format.scale_format == "bf16" else torch.float32
            )
            if scales.dtype != dtype or tuple(scales.shape) != (
                rows // 128,
                g.scale_columns,
            ):
                raise TypeError(
                    "block-scaled FP8 scales differ from their format or geometry"
                )
            s = scales.detach().contiguous().cpu()
            validate_fp8_block_words(q, s)
            self.write_bytes(parent_row * k, memoryview(q.numpy()).cast("B"))
            self.write_bytes(
                g.scale_plane_offset
                + parent_row // 128 * g.scale_columns * self.format.scale_bytes,
                memoryview(s.view(torch.uint8).numpy()).cast("B"),
            )
        elif isinstance(self.format, Mxfp8Format):
            g = row_group_scale_geometry(self.format, obj.shape)
            q = _exact_uint8_matrix(codes, (rows, k), "MXFP8 codes")
            s = _exact_uint8_matrix(scales, (rows, g.groups_per_row), "MXFP8 scales")
            validate_mxfp8_words(q, s)
            self.write_bytes(parent_row * k, memoryview(q.numpy()).cast("B"))
            self.write_bytes(
                g.scale_plane_offset + parent_row * g.groups_per_row,
                memoryview(s.numpy()).cast("B"),
            )
        elif isinstance(self.format, Nvfp4Format):
            if row_begin % 128 or rows % 128 or weight_scale is None:
                raise ValueError(
                    f"{obj.id}: NVFP4 output needs whole 128-row tiles and weight scalar"
                )
            g = block_scale_geometry(self.format, obj.shape)
            q = _exact_uint8_matrix(codes, (rows, k // 2), "NVFP4 codes")
            s = _exact_uint8_matrix(scales, (rows, k // 16), "NVFP4 scales")
            if bool((s > 126).any()):
                raise ValueError("NVFP4 scales must be nonnegative finite E4M3FN words")
            scalar = matrix_scale_words(weight_scale, 1)
            swizzled = swizzle_nvfp4_scales(s, (rows, k))
            if matrix in self._matrix_scales and self._matrix_scales[matrix] != scalar:
                raise ValueError(
                    f"{obj.id}: matrix weight scalar changed between row blocks"
                )
            self.write_bytes(parent_row * (k // 2), memoryview(q.numpy()).cast("B"))
            self.write_bytes(
                g.scale_plane_offset + parent_row * (k // 16),
                memoryview(swizzled.numpy()).cast("B"),
            )
            if matrix not in self._matrix_scales:
                self.write_bytes(g.matrix_scale_offset + matrix * 4, scalar)
                self._matrix_scales[matrix] = scalar
        else:
            raise TypeError(f"{obj.id}: direct format does not accept quantized codes")

    def _write_nf4(self, row_begin, codes, scales, weight_divisor):
        g = row_paged_geometry(self.format, self.object.shape)
        if row_begin != self._next_row:
            raise ValueError("NF4 streaming output requires consecutive rows")
        divisor = divisor_word(weight_divisor)
        if 0 in self._matrix_scales and self._matrix_scales[0] != divisor:
            raise ValueError("NF4 divisor changed between row blocks")
        self._matrix_scales[0] = divisor
        rows = nf4_rows(codes, scales, g.k)
        self._next_row += rows.shape[0]
        if self._pending_rows is not None:
            rows = np.concatenate((self._pending_rows, rows), axis=0)
        full, tail = divmod(rows.shape[0], g.rows_per_page)
        final = self._next_row == g.n
        count = full + int(final and tail > 0)
        if count:
            pages = np.zeros((count, 4096), dtype=np.uint8)
            used = g.rows_per_page * g.row_bytes
            pages[:full, :used] = rows[: full * g.rows_per_page].reshape(full, used)
            if final and tail:
                pages[full, : tail * g.row_bytes] = rows[
                    full * g.rows_per_page :
                ].reshape(-1)
            self.write_bytes(self._pages_written * 4096, memoryview(pages).cast("B"))
            self._pages_written += count
        self._pending_rows = (
            rows[full * g.rows_per_page :].copy() if tail and not final else None
        )
        if final:
            self.write_bytes(g.weight_divisor_offset, divisor)
