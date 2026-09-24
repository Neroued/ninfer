"""Map typed values or exact codes/scales to complete parent objects and writer ranges.

This layer owns codec selection, plane offsets, and padding within an object.
ArtifactWriter owns file placement, sharding, write coverage, and publication.
"""

from __future__ import annotations

import torch

from .codecs.direct import encode_direct
from .codecs.fp8_row import encode_fp8_row_scaled
from .codecs.nvfp4 import encode_nvfp4
from .codecs.row_split import encode_row_split, split_row_planes
from .formats import (
    DirectFormat,
    Fp8RowFormat,
    Nvfp4Format,
    QuantFormat,
    get_format,
)
from .layouts import (
    block_scale_geometry,
    row_scale_geometry,
    row_split_geometry,
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
        self._divisors: list[bytes | None] = [None] * obj.divisors

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
        if isinstance(self.format, QuantFormat):
            g = row_split_geometry(self.format, obj.shape)
            gaps = (
                (g.base_bytes, g.high_offset),
                (g.high_offset + g.high_bytes, g.scale_offset),
            )
        elif isinstance(self.format, Fp8RowFormat):
            g = row_scale_geometry(self.format, obj.shape)
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

    def write_codes(
        self,
        row_begin: int,
        codes: torch.Tensor,
        scales: torch.Tensor,
        weight_divisor: bytes | None = None,
    ) -> None:
        obj = self.object
        if len(obj.shape) != 2 or not 0 <= row_begin < obj.shape[0]:
            raise ValueError(f"{obj.id}: invalid encoded row origin {row_begin}")
        rows, k = codes.shape[0], obj.shape[1]
        if rows <= 0 or row_begin + rows > obj.shape[0]:
            raise ValueError(f"{obj.id}: encoded rows exceed parent")
        self._padding()
        if isinstance(self.format, QuantFormat):
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
        elif isinstance(self.format, Nvfp4Format):
            if row_begin % 128 or rows % 128 or weight_divisor is None:
                raise ValueError(
                    f"{obj.id}: NVFP4 output needs whole 128-row tiles and weight divisor"
                )
            g = block_scale_geometry(self.format, obj.shape, len(self._divisors))
            local = block_scale_geometry(self.format, (rows, k))
            block = memoryview(encode_nvfp4(codes, scales, weight_divisor, (rows, k)))
            self.write_bytes(row_begin * (k // 2), block[: local.code_plane_bytes])
            self.write_bytes(
                g.scale_plane_offset + row_begin * (k // 16),
                block[local.scale_plane_offset : local.weight_divisor_offset],
            )
            # A row block belongs to exactly one source, so it names exactly one divisor.
            rows_per_divisor = g.n // len(self._divisors)
            index, offset_in_source = divmod(row_begin, rows_per_divisor)
            if offset_in_source + rows > rows_per_divisor:
                raise ValueError(
                    f"{obj.id}: a row block may not span two separately quantised sources"
                )
            if self._divisors[index] is None:
                self.write_bytes(g.weight_divisor_offset + index * 4, weight_divisor)
                self._divisors[index] = bytes(weight_divisor)
            elif self._divisors[index] != weight_divisor:
                raise ValueError(f"{obj.id}: weight divisor changed between row blocks")
        else:
            raise TypeError(f"{obj.id}: direct format does not accept quantized codes")
