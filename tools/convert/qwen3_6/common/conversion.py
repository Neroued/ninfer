"""Mechanical artifact-writing helpers shared by Qwen3.6 converters.

This module deliberately knows neither target identity nor checkpoint geometry.
Target packages own their config validation, complete object inventory, source
recipe, shortlist provenance, and conversion entry point.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import platform
import subprocess
from typing import Mapping, Sequence

import torch

from tools.artifact.container import (
    ArtifactObject,
    ObjectSpec,
    ResourceObject,
    ResourceSpec as ArtifactResourceSpec,
    TensorObject,
    TensorSpec as ArtifactTensorSpec,
    plan_objects,
)
from tools.artifact.layouts import align_up, encode_direct, encoded_size
from tools.convert.common.quantize import quantize_and_encode

from .inventory import DIRECT_FORMATS, ResourceSpec, StoredObjectSpec, TensorSpec


@dataclass(frozen=True, slots=True)
class ResourcePayload:
    name: str
    data: bytes


@dataclass(frozen=True, slots=True)
class ObjectPlan:
    specs: tuple[ObjectSpec, ...]
    objects: tuple[ArtifactObject, ...]

    @property
    def payload_span_bytes(self) -> int:
        last = self.objects[-1]
        return last.offset + last.bytes


def load_json(path: str | Path) -> dict[str, object]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def check_members(
    scope: str,
    actual: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    """Require the target-selected members while allowing unrelated metadata."""

    mismatches = [
        f"{scope}.{name}: expected {value!r}, got {actual.get(name)!r}"
        for name, value in expected.items()
        if actual.get(name) != value
    ]
    if mismatches:
        raise ValueError("checkpoint config mismatch:\n  " + "\n  ".join(mismatches))


def load_resources(
    model_dir: str | Path,
    resource_specs: Sequence[ResourceSpec],
) -> tuple[ResourcePayload, ...]:
    root = Path(model_dir)
    resources: list[ResourcePayload] = []
    for spec in resource_specs:
        filename = spec.name.removeprefix("frontend/")
        data = (root / filename).read_bytes()
        if not data:
            raise ValueError(f"frontend resource {filename} is empty")
        resources.append(ResourcePayload(spec.name, data))
    return tuple(resources)


def build_object_plan(
    object_specs: Sequence[StoredObjectSpec],
    resources: Mapping[str, bytes],
) -> ObjectPlan:
    expected_resources = tuple(
        spec.name for spec in object_specs if isinstance(spec, ResourceSpec)
    )
    if tuple(resources) != expected_resources:
        raise ValueError("resource mapping does not match canonical inventory order")

    specs: list[ObjectSpec] = []
    for spec in object_specs:
        if isinstance(spec, ResourceSpec):
            specs.append(
                ArtifactResourceSpec(spec.name, spec.encoding, len(resources[spec.name]))
            )
        elif isinstance(spec, TensorSpec):
            specs.append(
                ArtifactTensorSpec(spec.name, spec.shape, spec.format, spec.layout)
            )
        else:
            raise TypeError(f"unsupported inventory spec: {type(spec).__name__}")
    frozen_specs = tuple(specs)
    return ObjectPlan(frozen_specs, plan_objects(frozen_specs))


def encode_tensor_payload(
    tensor: torch.Tensor,
    spec: TensorSpec,
    device: str | torch.device,
) -> bytes:
    if spec.format in DIRECT_FORMATS:
        return encode_direct(tensor, spec.format)
    return quantize_and_encode(tensor, spec.format, device=device)


def object_statistics(objects: Sequence[ArtifactObject]) -> dict[str, object]:
    tensors = [obj for obj in objects if isinstance(obj, TensorObject)]
    resources = [obj for obj in objects if isinstance(obj, ResourceObject)]
    object_bytes = sum(obj.bytes for obj in objects)
    payload_span = objects[-1].offset + objects[-1].bytes
    return {
        "count": len(objects),
        "tensors": len(tensors),
        "resources": len(resources),
        "formats": dict(sorted(Counter(obj.format for obj in tensors).items())),
        "layouts": dict(sorted(Counter(obj.layout for obj in tensors).items())),
        "encodings": dict(
            sorted(Counter(obj.encoding for obj in resources).items())
        ),
        "tensor_bytes": sum(obj.bytes for obj in tensors),
        "resource_bytes": sum(obj.bytes for obj in resources),
        "object_bytes": object_bytes,
        "payload_span_bytes": payload_span,
        "alignment_bytes": payload_span - object_bytes,
    }


def tensor_payload_bytes(tensor_specs: Sequence[TensorSpec]) -> int:
    return sum(
        encoded_size(spec.layout, spec.format, spec.shape) for spec in tensor_specs
    )


def device_arena_bytes(tensor_specs: Sequence[TensorSpec], alignment: int = 256) -> int:
    cursor = 0
    for spec in tensor_specs:
        cursor = align_up(cursor, alignment)
        cursor += encoded_size(spec.layout, spec.format, spec.shape)
    return cursor


def converter_revision(repo_root: str | Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def environment(device: torch.device) -> dict[str, object]:
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "resolved_device": str(device),
        "gpu": gpu,
    }


def build_conversion_report(
    *,
    model_id: str,
    target_key: str,
    recipe_id: str,
    repo_root: str | Path,
    model_dir: str | Path,
    out_path: str | Path,
    arguments: Mapping[str, object],
    config_summary: Mapping[str, object],
    source_preflight: object,
    objects: Sequence[ArtifactObject],
    elapsed_seconds: float,
    final_bytes: int,
    device: torch.device,
    ranking_path: str | Path,
    revision: str | None = None,
    environment_summary: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the shared report envelope without defining target validity."""

    return {
        "model_id": model_id,
        "target_key": target_key,
        "recipe_id": recipe_id,
        "source": {
            "model_path": str(Path(model_dir).resolve()),
            "ranking_path": str(Path(ranking_path).resolve()),
        },
        "arguments": dict(arguments),
        "config_summary": dict(config_summary),
        "source_preflight": {
            "recipes": source_preflight.recipe_count,
            "tensors": source_preflight.source_tensor_count,
            "shards": source_preflight.source_shard_count,
            "dtypes": dict(source_preflight.source_dtype_counts),
        },
        "converter": {
            "revision": (
                converter_revision(repo_root) if revision is None else revision
            ),
            "environment": dict(
                environment(device)
                if environment_summary is None
                else environment_summary
            ),
        },
        "objects": object_statistics(objects),
        "elapsed_seconds": elapsed_seconds,
        "artifact": {
            "path": str(Path(out_path)),
            "bytes": final_bytes,
        },
    }


__all__ = [
    "ObjectPlan",
    "ResourcePayload",
    "build_conversion_report",
    "build_object_plan",
    "check_members",
    "device_arena_bytes",
    "encode_tensor_payload",
    "environment",
    "load_json",
    "load_resources",
    "object_statistics",
    "tensor_payload_bytes",
]
