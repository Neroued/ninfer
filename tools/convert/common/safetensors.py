"""Lazy reads from a sharded Hugging Face safetensors checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open


@dataclass(frozen=True, slots=True)
class TensorMetadata:
    name: str
    shard: str
    shape: tuple[int, ...]
    dtype: str


class ShardReader:
    """Resolve an index lazily while keeping at most one shard handle open."""

    def __init__(
        self,
        model_dir: str | Path,
        index_filename: str = "model.safetensors.index.json",
    ) -> None:
        self.model_dir = Path(model_dir)
        index = json.loads((self.model_dir / index_filename).read_text())
        self.weight_map: dict[str, str] = dict(index["weight_map"])
        self._current_shard: str | None = None
        self._context = None
        self._handle = None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.weight_map)

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def _open_shard(self, shard: str):
        if shard == self._current_shard:
            return self._handle
        self.close()
        self._context = safe_open(
            str(self.model_dir / shard),
            framework="pt",
            device="cpu",
        )
        self._handle = self._context.__enter__()
        self._current_shard = shard
        return self._handle

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        handle = self._open_shard(shard)
        return handle.get_tensor(name)

    def metadata(self, names: Iterable[str]) -> dict[str, TensorMetadata]:
        self.close()
        by_shard: dict[str, list[str]] = {}
        for name in names:
            shard = self.weight_map[name]
            by_shard.setdefault(shard, []).append(name)

        result: dict[str, TensorMetadata] = {}
        for shard, shard_names in by_shard.items():
            with safe_open(
                str(self.model_dir / shard),
                framework="pt",
                device="cpu",
            ) as handle:
                for name in shard_names:
                    tensor_slice = handle.get_slice(name)
                    result[name] = TensorMetadata(
                        name=name,
                        shard=shard,
                        shape=tuple(tensor_slice.get_shape()),
                        dtype=str(tensor_slice.get_dtype()),
                    )
        return result

    def close(self) -> None:
        if self._context is not None:
            self._context.__exit__(None, None, None)
        self._current_shard = None
        self._context = None
        self._handle = None

    def __enter__(self) -> ShardReader:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
