"""Lazy logical values, encoded rows, and transforms independent of source files."""

from __future__ import annotations

from bisect import bisect_right
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
    encoded_format: str | Callable[[], str | None] | None = None

    def encoding(self) -> str | None:
        """Describe encoded input without reading its weight rows."""
        return (
            self.encoded_format()
            if callable(self.encoded_format)
            else self.encoded_format
        )

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
    ranges = tuple((int(begin), int(end)) for begin, end in ranges)

    # Precompute cumulative flattened-element endpoints once.
    #
    # The old implementation restarted from ranges[0] for every read.
    # Nested gathers (for example pruned full head -> proposal head) could
    # therefore become quadratic in the number of selected rows.
    cumulative_ends = []
    cursor = 0

    for begin, end in ranges:
        if not 0 <= begin < end <= n:
            raise ValueError(
                f"{source.label}: invalid selected rows [{begin},{end})"
            )

        cursor += (end - begin) * k
        cumulative_ends.append(cursor)

    total_elements = cursor
    rows = total_elements // k

    def spans(begin: int, end: int):
        if begin == end:
            return

        # First selected range whose cumulative end is strictly after begin.
        index = bisect_right(cumulative_ends, begin)

        range_begin = (
            0 if index == 0 else cumulative_ends[index - 1]
        )

        while index < len(ranges) and range_begin < end:
            first, _last = ranges[index]
            range_end = cumulative_ends[index]

            low = max(begin, range_begin)
            high = min(end, range_end)

            if low < high:
                yield (
                    first * k + low - range_begin,
                    first * k + high - range_begin,
                )

            range_begin = range_end
            index += 1

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return source.values(0, 0)

        parts = [
            source.values(a, b)
            for a, b in spans(begin, end)
        ]

        if not parts:
            raise RuntimeError(
                f"{source.label}: row selection produced no source spans"
            )

        return torch.cat(parts) if len(parts) != 1 else parts[0]

    def encoded(begin: int, end: int) -> EncodedRows:
        if source.read_encoded is None:
            raise ValueError(
                f"{source.label}: encoded source rows are unavailable"
            )

        pieces = [
            source.read_encoded(a // k, b // k)
            for a, b in spans(begin * k, end * k)
        ]

        if not pieces:
            raise RuntimeError(
                f"{source.label}: encoded row selection produced no spans"
            )

        first = pieces[0]

        if any(
            (part.format, part.weight_divisor)
            != (first.format, first.weight_divisor)
            for part in pieces
        ):
            raise ValueError(
                f"{source.label}: incompatible encoded row groups"
            )

        return EncodedRows(
            first.format,
            (
                torch.cat([p.codes for p in pieces])
                if len(pieces) != 1
                else first.codes
            ),
            (
                torch.cat([p.scales for p in pieces])
                if len(pieces) != 1
                else first.scales
            ),
            first.weight_divisor,
        )

    return LogicalSource(
        (rows, k),
        f"rows({source.label}, {len(ranges)} spans, {rows} rows)",
        read,
        encoded if source.read_encoded is not None else None,
        source.weight_divisor,
        source.input_divisor,
        encoded_format=source.encoded_format,
    )


def gather_source(source: LogicalSource, indices: torch.Tensor) -> LogicalSource:
    if len(source.shape) != 2:
        raise ValueError("gather source must be a matrix")

    ids = indices.to(torch.int64).cpu().tolist()

    if not ids or min(ids) < 0 or max(ids) >= source.shape[0]:
        raise ValueError("gather indices are outside source rows")

    # Preserve exact gather order, but collapse adjacent increasing IDs:
    #
    #   5, 6, 7, 20, 21, 4
    # becomes
    #   (5, 8), (20, 22), (4, 5)
    #
    # Duplicates and non-consecutive/reordered rows remain separate, so
    # gather semantics are unchanged.
    ranges = []
    first = ids[0]
    previous = ids[0]

    for row in ids[1:]:
        if row == previous + 1:
            previous = row
            continue

        ranges.append((first, previous + 1))
        first = row
        previous = row

    ranges.append((first, previous + 1))

    return select_rows(source, tuple(ranges))
