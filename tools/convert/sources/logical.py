"""Lazy logical values, encoded rows, and transforms independent of source files."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Callable

import torch


@dataclass(frozen=True, slots=True)
class EncodedRows:
    format: str
    codes: torch.Tensor
    scales: torch.Tensor
    weight_divisor: bytes | None = None


@dataclass(frozen=True, slots=True)
class LogicalSource:
    """Values use flat C-order elements; an optional encoded reader uses full rows."""

    shape: tuple[int, ...]
    label: str
    read_values: Callable[[int, int], torch.Tensor]
    read_encoded: Callable[[int, int], EncodedRows] | None = None
    weight_divisor: Callable[[], bytes] | None = None
    input_divisor: Callable[[], bytes] | None = None

    def values(self, begin: int = 0, end: int | None = None) -> torch.Tensor:
        end = prod(self.shape) if end is None else end
        if not 0 <= begin <= end <= prod(self.shape):
            raise ValueError(
                f"{self.label}: logical range [{begin},{end}) exceeds {self.shape}"
            )
        values = self.read_values(begin, end)
        if values.numel() != end - begin:
            raise ValueError(
                f"{self.label}: source returned {values.numel()} values, expected {end-begin}"
            )
        return values.reshape(-1)

    def rows(self, begin: int, end: int) -> torch.Tensor:
        if len(self.shape) != 2:
            raise ValueError(f"{self.label}: row access requires a matrix")
        k = self.shape[1]
        return self.values(begin * k, end * k).reshape(end - begin, k)


def array_source(values: torch.Tensor, label: str) -> LogicalSource:
    data = values.detach().cpu().contiguous()
    return LogicalSource(
        tuple(data.shape), label, lambda begin, end: data.reshape(-1)[begin:end]
    )


def transpose_source(
    source: LogicalSource, axes: tuple[int, ...], shape: tuple[int, ...]
) -> LogicalSource:
    # The architecture uses this for small convolution and norm-related tensors.
    def read(begin: int, end: int) -> torch.Tensor:
        return (
            source.values()
            .reshape(source.shape)
            .permute(axes)
            .contiguous()
            .reshape(-1)[begin:end]
        )

    return LogicalSource(shape, f"transpose({source.label},{axes})", read)


def select_rows(
    source: LogicalSource, ranges: tuple[tuple[int, int], ...]
) -> LogicalSource:
    if len(source.shape) != 2 or not ranges:
        raise ValueError("row selection requires a matrix and nonempty ranges")
    n, k = source.shape
    for begin, end in ranges:
        if not 0 <= begin < end <= n:
            raise ValueError(f"{source.label}: invalid selected rows [{begin},{end})")
    rows = sum(end - begin for begin, end in ranges)

    def spans(begin: int, end: int):
        cursor = 0
        for first, last in ranges:
            length = (last - first) * k
            low, high = max(begin, cursor), min(end, cursor + length)
            if low < high:
                yield first * k + low - cursor, first * k + high - cursor
            cursor += length

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return source.values(0, 0)
        parts = [source.values(a, b) for a, b in spans(begin, end)]
        return torch.cat(parts) if len(parts) != 1 else parts[0]

    def encoded(begin: int, end: int) -> EncodedRows:
        if source.read_encoded is None:
            raise ValueError(f"{source.label}: encoded source rows are unavailable")
        pieces = [
            source.read_encoded(a // k, b // k) for a, b in spans(begin * k, end * k)
        ]
        first = pieces[0]
        if any(
            (part.format, part.weight_divisor) != (first.format, first.weight_divisor)
            for part in pieces
        ):
            raise ValueError(f"{source.label}: incompatible encoded row groups")
        return EncodedRows(
            first.format,
            torch.cat([p.codes for p in pieces]),
            torch.cat([p.scales for p in pieces]),
            first.weight_divisor,
        )

    return LogicalSource(
        (rows, k),
        f"rows({source.label}, {len(ranges)} spans, {rows} rows)",
        read,
        encoded if source.read_encoded is not None else None,
        source.weight_divisor,
        source.input_divisor,
    )


def gather_source(source: LogicalSource, indices: torch.Tensor) -> LogicalSource:
    if len(source.shape) != 2:
        raise ValueError("gather source must be a matrix")
    ids = indices.to(torch.int64).cpu().tolist()
    if not ids or min(ids) < 0 or max(ids) >= source.shape[0]:
        raise ValueError("gather indices are outside source rows")
    return select_rows(source, tuple((row, row + 1) for row in ids))
