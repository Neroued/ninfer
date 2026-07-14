"""Structured activation taps for cross-runtime diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


class NullTap:
    level = "none"

    def __call__(self, *_args, **_kwargs) -> None:
        return

    def close(self, **_metadata) -> None:
        return


@dataclass
class TensorRecord:
    name: str
    file: str
    shape: list[int]
    source_dtype: str
    phase: str
    step: int
    chunk: int
    position_begin: int
    position_end: int


class FileTap:
    def __init__(self, path: str | Path, level: str = "layer"):
        if level not in {"layer", "op"}:
            raise ValueError("tap level must be layer/op")
        self.root = Path(path)
        self.root.mkdir(parents=True, exist_ok=True)
        self.level = level
        self.records: list[TensorRecord] = []

    def __call__(
        self,
        name: str,
        value: torch.Tensor,
        *,
        phase: str,
        step: int,
        chunk: int,
        position_begin: int,
    ) -> None:
        if self.level == "layer" and "/op/" in name:
            return
        prefix = f"{phase}/{step:04d}" if phase != "prefill" else f"prefill/chunk_{chunk:04d}"
        relative = Path(prefix) / f"{name}.f32"
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        host = value.detach().float().cpu().contiguous()
        host.numpy().astype(np.float32, copy=False).tofile(target)
        tokens = value.shape[0] if value.ndim > 1 else 1
        shape = list(value.shape)
        if name == "logits" and value.ndim == 1:
            shape = [1, value.numel()]
        self.records.append(
            TensorRecord(
                name=name,
                file=str(relative),
                shape=shape,
                source_dtype=str(value.dtype).removeprefix("torch."),
                phase=phase,
                step=step,
                chunk=chunk,
                position_begin=position_begin,
                position_end=position_begin + tokens,
            )
        )

    def close(self, **metadata) -> None:
        manifest = {
            "format": "ninfer_activation_dump_v1",
            **metadata,
            "tensors": [record.__dict__ for record in self.records],
        }
        (self.root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
