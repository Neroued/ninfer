"""Bounded v3 object writes with one fixed directory and logical payload space."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from copy import deepcopy
from pathlib import Path
import os
import tempfile
from typing import Iterable, Sequence
from uuid import uuid4

from .framing import HEADER, MAGIC, PART_MAGIC, PAYLOAD_ALIGNMENT
from .layouts import align_up
from .file_io import IO_CHUNK_BYTES, Writeback
from .schema import (
    ArtifactError,
    ArtifactObject,
    Directory,
    ObjectSpec,
    encode_directory,
    integer,
    parse_directory,
    plan_objects,
)

DEFAULT_MAX_FILE_BYTES = 32_000_000_000
ZERO_CHUNK_BYTES = 1024 * 1024


def layout_directory(
    entry_name: str,
    description: dict,
    payload_bytes: int,
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> tuple[Directory, bytes, int]:
    """Resolve JSON reserve and file segments before any payload is generated."""
    integer(payload_bytes, "payload bytes", positive=True)
    integer(max_file_bytes, "maximum file bytes", positive=True)
    value = deepcopy(description)
    value["files"] = [{"path": None, "payload_bytes": payload_bytes}]
    reserve = (
        align_up(HEADER.size + len(encode_directory(value)), PAYLOAD_ALIGNMENT)
        - HEADER.size
    )
    while True:
        entry_start = integer(HEADER.size + reserve, "entry payload offset")
        if entry_start >= max_file_bytes:
            raise ArtifactError(
                "maximum file bytes cannot contain the entry metadata and payload"
            )
        if entry_start + payload_bytes <= max_file_bytes:
            files = [{"path": None, "payload_bytes": payload_bytes}]
        else:
            entry_capacity = (
                (max_file_bytes - entry_start) // PAYLOAD_ALIGNMENT * PAYLOAD_ALIGNMENT
            )
            part_capacity = (
                (max_file_bytes - PAYLOAD_ALIGNMENT)
                // PAYLOAD_ALIGNMENT
                * PAYLOAD_ALIGNMENT
            )
            if min(entry_capacity, part_capacity) <= 0:
                raise ArtifactError(
                    "maximum file bytes cannot contain a positive aligned segment"
                )
            files = [{"path": None, "payload_bytes": entry_capacity}]
            remaining = payload_bytes - entry_capacity
            while remaining:
                count = min(remaining, part_capacity)
                files.append(
                    {
                        "path": f"{entry_name}.part-{len(files):04d}",
                        "payload_bytes": count,
                    }
                )
                remaining -= count
        value["files"] = files
        encoded = encode_directory(value)
        if len(encoded) <= reserve:
            directory = parse_directory(value, entry_name=entry_name)
            return directory, encoded + b" " * (reserve - len(encoded)), entry_start
        reserve = align_up(HEADER.size + len(encoded), PAYLOAD_ALIGNMENT) - HEADER.size


class _Coverage:
    def __init__(self) -> None:
        self.ranges: list[tuple[int, int]] = []

    def add(self, begin: int, end: int, label: str) -> None:
        index = bisect_left(self.ranges, (begin, end))
        if index and self.ranges[index - 1][1] > begin:
            raise ArtifactError(
                f"{label}: duplicate output coverage at [{begin},{end})"
            )
        if index < len(self.ranges) and self.ranges[index][0] < end:
            raise ArtifactError(
                f"{label}: duplicate output coverage at [{begin},{end})"
            )
        if index and self.ranges[index - 1][1] == begin:
            begin = self.ranges.pop(index - 1)[0]
            index -= 1
        if index < len(self.ranges) and self.ranges[index][0] == end:
            end = self.ranges.pop(index)[1]
        self.ranges.insert(index, (begin, end))


class ArtifactWriter:
    """Own temporary files and publish the entry after every declared byte is written."""

    def __init__(
        self,
        path: str | Path,
        specs: Sequence[ObjectSpec],
        *,
        components: dict,
        bindings: dict,
        uses: Sequence[dict] = (),
        metadata: dict | None = None,
        provenance: dict | None = None,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ) -> None:
        self.path = Path(path)
        self.objects = plan_objects(specs)
        self.by_id = {obj.id: obj for obj in self.objects}
        description = {
            "components": components,
            "objects": [obj.to_json() for obj in self.objects],
            "bindings": bindings,
            "uses": list(uses),
            "metadata": {} if metadata is None else metadata,
            "provenance": {} if provenance is None else provenance,
        }
        self.directory, encoded, self.payload_offset = layout_directory(
            self.path.name,
            description,
            self.objects[-1].offset + self.objects[-1].bytes,
            max_file_bytes=max_file_bytes,
        )
        self.artifact_id = uuid4().bytes
        self._coverage = {obj.id: _Coverage() for obj in self.objects}
        self._prefixes = [0]
        for file in self.directory.files:
            self._prefixes.append(self._prefixes[-1] + file.payload_bytes)
        self._destinations = [self.path] + [
            self.path.parent / file.path for file in self.directory.files[1:]
        ]
        for target in self._destinations:
            if target.exists():
                raise FileExistsError(target)
        self._fds: list[int] = []
        self._writeback = Writeback()
        self._temporary: list[Path] = []
        self._published: list[Path] = []
        self._finished = False
        self._failed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            for index, (target, file) in enumerate(
                zip(self._destinations, self.directory.files)
            ):
                fd, temporary = tempfile.mkstemp(
                    prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
                )
                self._fds.append(fd)
                self._temporary.append(Path(temporary))
                start = self.payload_offset if index == 0 else PAYLOAD_ALIGNMENT
                os.ftruncate(fd, integer(start + file.payload_bytes, "file bytes"))
                header = HEADER.pack(
                    MAGIC if index == 0 else PART_MAGIC,
                    len(encoded) if index == 0 else index,
                    self.artifact_id,
                )
                self._write_file(fd, 0, header)
                if index == 0:
                    self._write_file(fd, HEADER.size, encoded)
        except BaseException:
            self.abort()
            raise

    def _write_file(self, fd: int, offset: int, data: bytes | memoryview) -> None:
        view = memoryview(data).cast("B")
        while view:
            count = os.pwrite(fd, view[:IO_CHUNK_BYTES], offset)
            if count <= 0:
                raise OSError(f"short write at file offset {offset}")
            view = view[count:]
            offset += count
            self._writeback.written(fd, count)

    def _write_range(self, offset: int, data: memoryview) -> None:
        cursor = 0
        while cursor < len(data):
            index = bisect_right(self._prefixes, offset) - 1
            count = min(len(data) - cursor, self._prefixes[index + 1] - offset)
            start = self.payload_offset if index == 0 else PAYLOAD_ALIGNMENT
            self._write_file(
                self._fds[index],
                start + offset - self._prefixes[index],
                data[cursor : cursor + count],
            )
            cursor += count
            offset += count

    def write_region(
        self, object_id: str, offset: int, data: bytes | bytearray | memoryview
    ) -> None:
        if not self._fds or self._finished or self._failed:
            raise ArtifactError("writer is closed or has failed")
        try:
            obj = self.by_id[object_id]
        except KeyError as error:
            raise ArtifactError(f"unknown output object {object_id!r}") from error
        integer(offset, f"{object_id} relative offset")
        view = memoryview(data).cast("B")
        end = integer(offset + len(view), f"{object_id} output end")
        if end > obj.bytes:
            raise ArtifactError(
                f"{object_id}: output [{offset},{end}) exceeds {obj.bytes} bytes"
            )
        if not view:
            return
        try:
            self._coverage[object_id].add(offset, end, object_id)
            self._write_range(obj.offset + offset, view)
        except BaseException:
            self._failed = True
            raise

    def write_zeros(self, object_id: str, offset: int, length: int) -> None:
        integer(length, f"{object_id} zero length")
        self.write_region(object_id, offset, b"")
        zero = bytes(min(length, ZERO_CHUNK_BYTES))
        while length:
            count = min(length, len(zero))
            self.write_region(object_id, offset, memoryview(zero)[:count])
            offset += count
            length -= count

    def write_object(self, object_id: str, chunks: bytes | Iterable[bytes]) -> None:
        if isinstance(chunks, (bytes, bytearray, memoryview)):
            chunks = (chunks,)
        offset = 0
        for chunk in chunks:
            self.write_region(object_id, offset, chunk)
            offset += len(chunk)

    def finish(self) -> Directory:
        if self._finished:
            return self.directory
        if self._failed or not self._fds:
            raise ArtifactError("writer is closed or has failed")
        try:
            for obj in self.objects:
                if self._coverage[obj.id].ranges != [(0, obj.bytes)]:
                    raise ArtifactError(f"{obj.id}: output coverage is incomplete")
            for index, fd in enumerate(self._fds):
                start = self.payload_offset if index == 0 else PAYLOAD_ALIGNMENT
                expected = start + self.directory.files[index].payload_bytes
                if os.fstat(fd).st_size != expected:
                    raise ArtifactError(
                        f"file {index}: output length does not match directory"
                    )
            self._writeback.flush()
            while self._fds:
                os.close(self._fds.pop())
            for index in [*range(1, len(self._temporary)), 0]:
                target = self._destinations[index]
                os.link(self._temporary[index], target)
                self._published.append(target)
            for temporary in self._temporary:
                temporary.unlink()
            self._temporary.clear()
            self._finished = True
            return self.directory
        except BaseException:
            self.abort()
            raise

    def abort(self) -> None:
        while self._fds:
            os.close(self._fds.pop())
        if not self._finished:
            for path in (*self._temporary, *self._published):
                path.unlink(missing_ok=True)
        self._temporary.clear()
        self._published.clear()
        self._failed = True

    def __enter__(self) -> ArtifactWriter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.finish()
        else:
            self.abort()
