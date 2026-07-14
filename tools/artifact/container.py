"""Minimal reader and streaming writer for the NInfer v1 object directory."""

from __future__ import annotations

import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TypeAlias

from .layouts import align_up, encoded_size, get_layout


MAGIC = b"NINFER\x00\x01"
PREFIX = struct.Struct("<8sQ")
PREFIX_BYTES = PREFIX.size
PAYLOAD_ALIGNMENT = 4096
RAW_BYTES_V1 = "raw-bytes-v1"

_ROOT_MEMBERS = frozenset({"model_id", "objects"})
_TENSOR_MEMBERS = frozenset(
    {"name", "kind", "shape", "format", "layout", "offset", "bytes"}
)
_RESOURCE_MEMBERS = frozenset({"name", "kind", "encoding", "offset", "bytes"})


class ArtifactError(ValueError):
    """The file does not satisfy the NInfer v1 directory contract."""


@dataclass(frozen=True, slots=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    format: str
    layout: str


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    name: str
    encoding: str
    bytes: int


ObjectSpec: TypeAlias = TensorSpec | ResourceSpec


@dataclass(frozen=True, slots=True)
class TensorObject:
    name: str
    shape: tuple[int, ...]
    format: str
    layout: str
    offset: int
    bytes: int

    @property
    def kind(self) -> str:
        return "tensor"

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "shape": list(self.shape),
            "format": self.format,
            "layout": self.layout,
            "offset": self.offset,
            "bytes": self.bytes,
        }


@dataclass(frozen=True, slots=True)
class ResourceObject:
    name: str
    encoding: str
    offset: int
    bytes: int

    @property
    def kind(self) -> str:
        return "resource"

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "encoding": self.encoding,
            "offset": self.offset,
            "bytes": self.bytes,
        }


ArtifactObject: TypeAlias = TensorObject | ResourceObject
PayloadChunk: TypeAlias = bytes | bytearray | memoryview
Payload: TypeAlias = PayloadChunk | Iterable[PayloadChunk]


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"{field} must be a nonempty string")
    return value


def _require_integer(value: object, field: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise ArtifactError(f"{field} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ArtifactError(f"{field} must be {qualifier}")
    return value


def _require_shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ArtifactError("tensor shape must be an array")
    return tuple(_require_integer(dim, "shape dimension", positive=True) for dim in value)


def _resource_alignment(encoding: str) -> int:
    if encoding != RAW_BYTES_V1:
        raise ArtifactError(f"unknown resource encoding: {encoding}")
    return 1


def object_alignment(obj: ArtifactObject) -> int:
    if isinstance(obj, TensorObject):
        return get_layout(obj.layout).alignment
    return _resource_alignment(obj.encoding)


def plan_objects(specs: Sequence[ObjectSpec]) -> tuple[ArtifactObject, ...]:
    """Assign aligned payload-relative offsets to an ordered complete inventory."""

    if not specs:
        raise ArtifactError("artifact inventory must not be empty")
    objects: list[ArtifactObject] = []
    names: set[str] = set()
    cursor = 0
    for spec in specs:
        name = _require_string(spec.name, "object name")
        if name in names:
            raise ArtifactError(f"duplicate object name: {name}")
        names.add(name)
        if isinstance(spec, TensorSpec):
            shape = tuple(_require_integer(dim, "shape dimension", positive=True) for dim in spec.shape)
            layout = get_layout(_require_string(spec.layout, "tensor layout"))
            payload_bytes = encoded_size(layout, spec.format, shape)
            offset = align_up(cursor, layout.alignment)
            obj: ArtifactObject = TensorObject(
                name=name,
                shape=shape,
                format=_require_string(spec.format, "tensor format"),
                layout=layout.name,
                offset=offset,
                bytes=payload_bytes,
            )
        elif isinstance(spec, ResourceSpec):
            encoding = _require_string(spec.encoding, "resource encoding")
            alignment = _resource_alignment(encoding)
            payload_bytes = _require_integer(spec.bytes, "resource bytes", positive=True)
            offset = align_up(cursor, alignment)
            obj = ResourceObject(name, encoding, offset, payload_bytes)
        else:
            raise TypeError(f"unsupported object spec type: {type(spec).__name__}")
        objects.append(obj)
        cursor = obj.offset + obj.bytes
    return tuple(objects)


def encode_directory(model_id: str, objects: Sequence[ArtifactObject]) -> bytes:
    _require_string(model_id, "model_id")
    if not objects:
        raise ArtifactError("objects must not be empty")
    value = {
        "model_id": model_id,
        "objects": [obj.to_json() for obj in objects],
    }
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _parse_object(value: object) -> ArtifactObject:
    if not isinstance(value, dict):
        raise ArtifactError("each object entry must be a JSON object")
    kind = value.get("kind")
    if kind == "tensor":
        if frozenset(value) != _TENSOR_MEMBERS:
            raise ArtifactError("tensor entry has missing or extra members")
        name = _require_string(value["name"], "tensor name")
        shape = _require_shape(value["shape"])
        format_name = _require_string(value["format"], "tensor format")
        layout_name = _require_string(value["layout"], "tensor layout")
        offset = _require_integer(value["offset"], "tensor offset")
        payload_bytes = _require_integer(value["bytes"], "tensor bytes", positive=True)
        try:
            expected = encoded_size(layout_name, format_name, shape)
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError(str(exc)) from exc
        if payload_bytes != expected:
            raise ArtifactError(
                f"tensor {name} stores {payload_bytes} bytes; layout requires {expected}"
            )
        return TensorObject(name, shape, format_name, layout_name, offset, payload_bytes)
    if kind == "resource":
        if frozenset(value) != _RESOURCE_MEMBERS:
            raise ArtifactError("resource entry has missing or extra members")
        name = _require_string(value["name"], "resource name")
        encoding = _require_string(value["encoding"], "resource encoding")
        _resource_alignment(encoding)
        offset = _require_integer(value["offset"], "resource offset")
        payload_bytes = _require_integer(value["bytes"], "resource bytes", positive=True)
        return ResourceObject(name, encoding, offset, payload_bytes)
    raise ArtifactError("object kind must be 'tensor' or 'resource'")


def parse_directory(data: bytes) -> tuple[str, tuple[ArtifactObject, ...]]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid JSON directory: {exc}") from exc
    if not isinstance(value, dict) or frozenset(value) != _ROOT_MEMBERS:
        raise ArtifactError("directory root must contain exactly model_id and objects")
    model_id = _require_string(value["model_id"], "model_id")
    raw_objects = value["objects"]
    if not isinstance(raw_objects, list) or not raw_objects:
        raise ArtifactError("objects must be a nonempty array")
    objects = tuple(_parse_object(item) for item in raw_objects)
    names: set[str] = set()
    for obj in objects:
        if obj.name in names:
            raise ArtifactError(f"duplicate object name: {obj.name}")
        names.add(obj.name)
    return model_id, objects


def _validate_ranges(
    objects: Sequence[ArtifactObject], payload_bytes: int
) -> dict[str, ArtifactObject]:
    cursor = 0
    index: dict[str, ArtifactObject] = {}
    for obj in objects:
        alignment = object_alignment(obj)
        if obj.offset < cursor:
            raise ArtifactError(f"object {obj.name} overlaps or is out of order")
        if obj.offset % alignment:
            raise ArtifactError(f"object {obj.name} is not {alignment}-byte aligned")
        end = obj.offset + obj.bytes
        if end > payload_bytes:
            raise ArtifactError(f"object {obj.name} extends beyond the file")
        cursor = end
        index[obj.name] = obj
    return index


class Artifact:
    """Mmap-backed, structurally validated `.ninfer` artifact."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = self.path.open("rb")
        self._mapping: mmap.mmap | None = None
        try:
            self._file.seek(0, 2)
            self.file_bytes = self._file.tell()
            self._file.seek(0)
            if self.file_bytes < PREFIX_BYTES:
                raise ArtifactError("artifact is shorter than the v1 prefix")
            prefix = self._file.read(PREFIX_BYTES)
            magic, json_bytes = PREFIX.unpack(prefix)
            if magic != MAGIC:
                raise ArtifactError("artifact magic is not NInfer v1")
            if json_bytes == 0:
                raise ArtifactError("json_bytes must be positive")
            metadata_end = PREFIX_BYTES + json_bytes
            self.payload_offset = align_up(metadata_end, PAYLOAD_ALIGNMENT)
            if metadata_end > self.file_bytes or self.payload_offset > self.file_bytes:
                raise ArtifactError("declared JSON or payload start extends beyond the file")
            directory = self._file.read(json_bytes)
            if len(directory) != json_bytes:
                raise ArtifactError("artifact JSON is truncated")
            self.model_id, self.objects = parse_directory(directory)
            payload_bytes = self.file_bytes - self.payload_offset
            self._index = _validate_ranges(self.objects, payload_bytes)
            self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except BaseException:
            if self._mapping is not None:
                self._mapping.close()
            self._file.close()
            raise

    @classmethod
    def open(cls, path: str | Path) -> "Artifact":
        return cls(path)

    def find(self, name: str) -> ArtifactObject:
        return self._index[name]

    def payload(self, obj: ArtifactObject | str) -> memoryview:
        if isinstance(obj, str):
            obj = self.find(obj)
        if self._mapping is None:
            raise RuntimeError("artifact is closed")
        begin = self.payload_offset + obj.offset
        return memoryview(self._mapping)[begin : begin + obj.bytes]

    def close(self) -> None:
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "Artifact":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _payload_chunks(payload: Payload) -> Iterator[memoryview]:
    if isinstance(payload, (bytes, bytearray, memoryview)):
        yield memoryview(payload).cast("B")
        return
    for chunk in payload:
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise TypeError("payload chunks must support the buffer protocol")
        yield memoryview(chunk).cast("B")


class ArtifactWriter:
    """Write one preplanned artifact payload at a time in directory order."""

    def __init__(self, path: str | Path, model_id: str, specs: Sequence[ObjectSpec]):
        self.path = Path(path)
        self.model_id = _require_string(model_id, "model_id")
        self.objects = plan_objects(specs)
        directory = encode_directory(self.model_id, self.objects)
        self.payload_offset = align_up(PREFIX_BYTES + len(directory), PAYLOAD_ALIGNMENT)
        self._file = self.path.open("wb")
        self._file.write(PREFIX.pack(MAGIC, len(directory)))
        self._file.write(directory)
        self._file.write(b"\x00" * (self.payload_offset - PREFIX_BYTES - len(directory)))
        self._next = 0
        self._cursor = 0
        self._finished = False

    def write(self, name: str, payload: Payload) -> None:
        if self._finished:
            raise RuntimeError("artifact writer is already finished")
        if self._next >= len(self.objects):
            raise ArtifactError("artifact already has every planned payload")
        obj = self.objects[self._next]
        if name != obj.name:
            raise ArtifactError(f"expected payload {obj.name}, got {name}")
        if obj.offset > self._cursor:
            self._file.write(b"\x00" * (obj.offset - self._cursor))
            self._cursor = obj.offset
        written = 0
        for chunk in _payload_chunks(payload):
            if written + len(chunk) > obj.bytes:
                raise ArtifactError(f"payload {name} exceeds its planned byte length")
            self._file.write(chunk)
            written += len(chunk)
        if written != obj.bytes:
            raise ArtifactError(f"payload {name} has {written} bytes; expected {obj.bytes}")
        self._cursor = obj.offset + obj.bytes
        self._next += 1

    def finish(self) -> None:
        if self._finished:
            return
        if self._next != len(self.objects):
            missing = self.objects[self._next].name
            raise ArtifactError(f"artifact is missing payload {missing}")
        self._file.truncate(self.payload_offset + self._cursor)
        self._file.flush()
        self._file.close()
        self._finished = True

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "ArtifactWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            try:
                self.finish()
            finally:
                if not self._finished:
                    self.close()
        else:
            self.close()


def write_artifact(
    path: str | Path,
    model_id: str,
    entries: Sequence[tuple[ObjectSpec, Payload]],
) -> tuple[ArtifactObject, ...]:
    specs = [spec for spec, _ in entries]
    with ArtifactWriter(path, model_id, specs) as writer:
        for spec, payload in entries:
            writer.write(spec.name, payload)
    return writer.objects
