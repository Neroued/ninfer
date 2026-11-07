"""Exact CSF32 scale reconstruction for artifact inspection and references."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import numpy as np
import torch

from ..layouts import csf32_geometry


def scale_blocks(rows: int, columns: int) -> Iterator[tuple[int, int, int, int]]:
    """Yield (page, row, column, width) in CSF32 storage order."""
    tile = 0
    for row_tile in range(0, rows, 128):
        for column_tile in range(0, columns, 4):
            for column in range(column_tile, min(column_tile + 4, columns), 2):
                for row in range(row_tile, min(row_tile + 128, rows), 32):
                    yield tile // 32, row, column, min(2, columns - column)
            tile += 1


def _plane(value: object, dtype: str, count: int, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().contiguous().cpu().numpy()
    if isinstance(value, (bytes, bytearray, memoryview)):
        if memoryview(value).nbytes != count * np.dtype(dtype).itemsize:
            raise ValueError(f"CSF {name} byte count differs from geometry")
        result = np.frombuffer(value, dtype=dtype)
    else:
        result = np.asarray(value)
        if result.dtype != np.dtype(dtype) or result.size != count:
            raise ValueError(f"CSF {name} dtype or element count differs from geometry")
    return result.reshape(-1)


def decode_csf32_scales(
    planes: Mapping[str, object], shape: Sequence[int], exception_count: int
) -> torch.Tensor:
    """Recover exact E4M3 bytes; no floating-point interpretation is performed."""
    g = csf32_geometry(shape, exception_count)
    e, n, c = g.matrices, g.n, g.scale_columns
    bases = _plane(planes["bases"], "u1", e * n, "bases").reshape(e, n)
    codes = _plane(planes["nibbles"], "u1", e * n * c // 2, "nibbles")
    offsets = _plane(
        planes["offsets"], "<u2", e * g.offsets_per_matrix, "offsets"
    ).reshape(e, g.offsets_per_matrix)
    page_bases = _plane(
        planes["page_bases"], "<u4", e * g.pages_per_matrix, "page_bases"
    ).reshape(e, g.pages_per_matrix)
    exceptions = _plane(planes["exceptions"], "<u2", exception_count, "exceptions")
    output = np.empty((e, n, c), dtype=np.uint8)
    expected_exception = 0
    for matrix in range(e):
        cursor = matrix * n * c // 2
        previous_page = -1
        for block, (page, row, column, width) in enumerate(scale_blocks(n, c)):
            start = int(offsets[matrix, block + page])
            end = int(offsets[matrix, block + page + 1])
            page_start = int(page_bases[matrix, page])
            if page_start % 2 or start > end or end - start > 32 * width:
                raise ValueError("CSF exception range is invalid")
            if page != previous_page and start:
                raise ValueError("CSF page must begin with a zero relative offset")
            begin, finish = page_start // 2 + start, page_start // 2 + end
            if begin != expected_exception or finish > exception_count:
                raise ValueError("CSF exception pages do not cover their arena")
            count = 32 * width
            packed = codes[cursor : cursor + count // 2]
            deltas = np.empty(count, dtype=np.uint8)
            deltas[::2], deltas[1::2] = packed & 15, packed >> 4
            restored = bases[matrix, row : row + 32, None].astype(
                np.uint16
            ) + deltas.reshape(32, width)
            records = exceptions[begin:finish]
            positions = (records & 255).astype(np.int64)
            local_rows, local_columns = positions // 2, positions % 2
            if (
                np.any(local_rows >= 32)
                or np.any(local_columns >= width)
                or np.unique(positions).size != positions.size
            ):
                raise ValueError("CSF exception positions are invalid")
            restored[local_rows, local_columns] = records >> 8
            if np.any(restored > 126):
                raise ValueError("CSF scales must be nonnegative finite E4M3 words")
            output[matrix, row : row + 32, column : column + width] = restored
            cursor += count // 2
            expected_exception = finish
            previous_page = page
    if expected_exception != exception_count:
        raise ValueError("CSF exceptions exceed represented scale blocks")
    return torch.from_numpy(output.reshape(*shape[:-1], c))
