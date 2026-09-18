"""Demand-driven reads of a NInfer v3 entry and its recorded continuation files."""

from __future__ import annotations

from bisect import bisect_right
import os
from pathlib import Path
from typing import Iterator

from .framing import HEADER, MAGIC, PART_MAGIC, PAYLOAD_ALIGNMENT
from .layouts import align_up
from .file_io import discard_cached_pages, IO_CHUNK_BYTES
from .schema import (
    ArtifactError,
    ArtifactObject,
    Directory,
    ResourceObject,
    TensorObject,
    decode_directory,
    integer,
    validate_encoding,
)

READ_CHUNK_BYTES = IO_CHUNK_BYTES


class Artifact:
    """Own file handles; opening the entry does not open unused continuation files."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fds: dict[int, int] = {}
        self._validated: set[str] = set()
        try:
            fd = os.open(self.path, os.O_RDONLY)
            self._fds[0] = fd
            entry_bytes = os.fstat(fd).st_size
            raw = os.pread(fd, HEADER.size, 0)
            if len(raw) != HEADER.size:
                raise ArtifactError("truncated v3 entry header")
            magic, json_bytes, self.artifact_id = HEADER.unpack(raw)
            if magic != MAGIC:
                raise ArtifactError("expected NInfer v3 entry magic")
            integer(json_bytes, "json_bytes", positive=True)
            if json_bytes > entry_bytes - HEADER.size:
                raise ArtifactError("directory JSON exceeds entry file")
            raw_json = os.pread(fd, json_bytes, HEADER.size)
            if len(raw_json) != json_bytes:
                raise ArtifactError("truncated directory JSON")
            self.directory: Directory = decode_directory(
                raw_json, entry_name=self.path.name
            )
            self.payload_offset = integer(
                align_up(HEADER.size + json_bytes, PAYLOAD_ALIGNMENT),
                "entry payload offset",
            )
            expected = integer(
                self.payload_offset + self.directory.files[0].payload_bytes,
                "entry file bytes",
            )
            if entry_bytes != expected:
                raise ArtifactError(
                    f"entry length {entry_bytes} does not match {expected}"
                )
            self._prefixes = [0]
            for file in self.directory.files:
                self._prefixes.append(self._prefixes[-1] + file.payload_bytes)
            self.by_id = {obj.id: obj for obj in self.directory.objects}
        except BaseException:
            self.close()
            raise

    @classmethod
    def open(cls, path: str | Path) -> Artifact:
        return cls(path)

    @property
    def objects(self) -> tuple[ArtifactObject, ...]:
        return self.directory.objects

    @property
    def payload_bytes(self) -> int:
        return self.directory.payload_bytes

    @property
    def file_bytes(self) -> int:
        """Total declared file bytes, including framing in unopened files."""
        return (
            self.payload_bytes
            + self.payload_offset
            + (len(self.directory.files) - 1) * 4096
        )

    def _file(self, index: int) -> int:
        if 0 not in self._fds:
            raise ArtifactError("artifact is closed")
        if index in self._fds:
            return self._fds[index]
        file = self.directory.files[index]
        path = self.path.parent / file.path
        fd = os.open(path, os.O_RDONLY)
        try:
            raw = os.pread(fd, HEADER.size, 0)
            if len(raw) != HEADER.size:
                raise ArtifactError(f"{path}: truncated continuation header")
            magic, actual_index, artifact_id = HEADER.unpack(raw)
            if (
                magic != PART_MAGIC
                or actual_index != index
                or artifact_id != self.artifact_id
            ):
                raise ArtifactError(
                    f"{path}: continuation header does not belong to this entry"
                )
            expected = integer(
                PAYLOAD_ALIGNMENT + file.payload_bytes, "continuation file bytes"
            )
            if os.fstat(fd).st_size != expected:
                raise ArtifactError(
                    f"{path}: continuation length does not match {expected}"
                )
        except BaseException:
            os.close(fd)
            raise
        self._fds[index] = fd
        return fd

    def iter_range(
        self, offset: int, length: int, *, chunk_bytes: int = READ_CHUNK_BYTES
    ) -> Iterator[bytes]:
        """Yield exactly a logical payload range with bounded temporary memory."""
        integer(offset, "range offset")
        integer(length, "range length")
        integer(chunk_bytes, "read chunk bytes", positive=True)
        end = integer(offset + length, "range end")
        if end > self.payload_bytes:
            raise ArtifactError(
                f"range [{offset},{end}) exceeds payload {self.payload_bytes}"
            )
        if 0 not in self._fds:
            raise ArtifactError("artifact is closed")
        while offset < end:
            index = bisect_right(self._prefixes, offset) - 1
            count = min(end - offset, self._prefixes[index + 1] - offset, chunk_bytes)
            file_offset = self.payload_offset if index == 0 else PAYLOAD_ALIGNMENT
            file_offset += offset - self._prefixes[index]
            fd = self._file(index)
            data = os.pread(fd, count, file_offset)
            if len(data) != count:
                raise ArtifactError(
                    f"short read at logical offset {offset}: {len(data)} of {count}"
                )
            discard_cached_pages(fd, file_offset, count)
            yield data
            offset += count

    def read_range(self, offset: int, length: int) -> bytes:
        return b"".join(self.iter_range(offset, length))

    def object(self, object_id: str) -> ArtifactObject:
        try:
            obj = self.by_id[object_id]
        except KeyError as error:
            raise ArtifactError(f"missing object {object_id!r}") from error
        if object_id not in self._validated:
            validate_encoding(obj)
            self._validated.add(object_id)
        return obj

    def iter_object(
        self, object_id: str, *, chunk_bytes: int = READ_CHUNK_BYTES
    ) -> Iterator[bytes]:
        obj = self.object(object_id)
        yield from self.iter_range(obj.offset, obj.bytes, chunk_bytes=chunk_bytes)

    def read_object(self, object_id: str) -> bytes:
        return b"".join(self.iter_object(object_id))

    def close(self) -> None:
        for fd in self._fds.values():
            discard_cached_pages(fd)
            os.close(fd)
        self._fds.clear()

    def __enter__(self) -> Artifact:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
