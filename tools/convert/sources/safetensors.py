"""Local Safetensors headers, indexes, bounded reads, and direct tensor views."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from math import prod
import os
from pathlib import Path
import struct

import torch

from tools.artifact.file_io import discard_cached_pages
from .logical import LogicalSource

_DTYPES = {
    "BF16": (torch.bfloat16, 2),
    "F16": (torch.float16, 2),
    "F32": (torch.float32, 4),
    "F64": (torch.float64, 8),
    "I32": (torch.int32, 4),
    "I64": (torch.int64, 8),
    "I8": (torch.int8, 1),
    "U8": (torch.uint8, 1),
    "F8_E4M3": (torch.float8_e4m3fn, 1),
}


@dataclass(frozen=True, slots=True)
class TensorInfo:
    file: Path
    shape: tuple[int, ...]
    dtype: str
    offset: int
    bytes: int


class SafetensorsSource:
    """Read bounded flat regions; retain only a small set of file descriptors."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.root = self.path if self.path.is_dir() else self.path.parent
        config_path = self.root / "config.json"
        self.config = (
            json.loads(config_path.read_text()) if config_path.is_file() else {}
        )
        self._headers: dict[Path, dict[str, TensorInfo]] = {}
        self._fds: OrderedDict[Path, int] = OrderedDict()
        self.bytes_read = 0
        if self.path.is_file() and self.path.suffix == ".safetensors":
            file = self.path
            self.weight_map = {name: file for name in self._header(file)}
        else:
            index = (
                self.path
                if self.path.is_file()
                else self.root / "model.safetensors.index.json"
            )
            if index.is_file():
                data = json.loads(index.read_text())
                self.weight_map = {
                    name: index.parent / file
                    for name, file in data["weight_map"].items()
                }
            else:
                file = self.root / "model.safetensors"
                if file.is_file():
                    self.weight_map = {name: file for name in self._header(file)}
                elif self.path.is_dir():
                    # A custom recipe may supply every logical value from another source.
                    self.weight_map = {}
                else:
                    raise FileNotFoundError(self.path)

    def _header(self, file: Path) -> dict[str, TensorInfo]:
        if file in self._headers:
            return self._headers[file]
        with file.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"{file}: truncated safetensors header")
            length = struct.unpack("<Q", prefix)[0]
            file_bytes = os.fstat(stream.fileno()).st_size
            if length > file_bytes - 8:
                raise ValueError(f"{file}: safetensors header exceeds file")
            data = json.loads(stream.read(length))
        result = {}
        for name, item in data.items():
            if name == "__metadata__":
                continue
            begin, end = item["data_offsets"]
            dims = tuple(item["shape"])
            if not 0 <= begin <= end <= file_bytes - length - 8:
                raise ValueError(f"{file}/{name}: invalid source range")
            result[name] = TensorInfo(
                file, dims, item["dtype"], 8 + length + begin, end - begin
            )
        self._headers[file] = result
        return result

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def describe(self, name: str) -> TensorInfo:
        try:
            file = self.weight_map[name]
            return self._header(file)[name]
        except KeyError as error:
            raise ValueError(f"{self.path}: missing source tensor {name!r}") from error

    def _file(self, path: Path) -> int:
        if path in self._fds:
            self._fds.move_to_end(path)
            return self._fds[path]
        if len(self._fds) == 4:
            _, fd = self._fds.popitem(last=False)
            discard_cached_pages(fd)
            os.close(fd)
        fd = os.open(path, os.O_RDONLY)
        self._fds[path] = fd
        return fd

    def read_flat(
        self, name: str, begin: int = 0, end: int | None = None
    ) -> torch.Tensor:
        info = self.describe(name)
        elements = prod(info.shape)
        end = elements if end is None else end
        if not 0 <= begin <= end <= elements:
            raise ValueError(
                f"{name}: source element range [{begin},{end}) exceeds {info.shape}"
            )
        try:
            dtype, word_bytes = _DTYPES[info.dtype]
        except KeyError as error:
            raise ValueError(
                f"{name}: unsupported source dtype {info.dtype}"
            ) from error
        if info.bytes != elements * word_bytes:
            raise ValueError(f"{name}: source byte count disagrees with shape/dtype")
        if begin == end:
            return torch.empty(0, dtype=dtype)
        count = (end - begin) * word_bytes
        fd = self._file(info.file)
        offset = info.offset + begin * word_bytes
        raw = os.pread(fd, count, offset)
        if len(raw) != count:
            raise ValueError(f"{name}: short source read")
        self.bytes_read += count
        discard_cached_pages(fd, offset, count)
        return torch.frombuffer(bytearray(raw), dtype=dtype)

    def close(self) -> None:
        while self._fds:
            _, fd = self._fds.popitem()
            discard_cached_pages(fd)
            os.close(fd)

    def __enter__(self) -> SafetensorsSource:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def tensor_source(
    store: SafetensorsSource,
    name: str,
    shape: tuple[int, ...],
    *,
    offset: int = 0,
    source_shape: tuple[int, ...] | None = None,
) -> LogicalSource:
    expected = shape if source_shape is None else source_shape

    def read(begin: int, end: int) -> torch.Tensor:
        info = store.describe(name)
        if info.shape != expected:
            raise ValueError(
                f"{name}: expected source shape {expected}, got {info.shape}"
            )
        return store.read_flat(name, offset + begin, offset + end)

    return LogicalSource(
        shape, f"{store.path}:{name}[{offset}:{offset+prod(shape)}]", read
    )
