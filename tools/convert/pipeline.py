"""Complete conversion from a configured logical model to a v3 file collection."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time
from typing import Callable

import torch

from tools.artifact.schema import ResourceSpec
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter, DEFAULT_MAX_FILE_BYTES

from .model import Model
from .recipe import Recipe, WeightJob


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"conversion report cannot serialize {type(value).__name__}")


def convert(
    model: Model,
    recipe: Recipe,
    output: str | Path,
    *,
    name: str | None = None,
    provenance: dict | None = None,
    device: str = "cuda",
    rows_per_chunk: int = 512,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    progress: Callable[[int, int, WeightJob], None] | None = None,
) -> dict:
    path = Path(output)
    report_path = Path(str(path) + ".conversion.json")
    if path.exists() or report_path.exists():
        raise FileExistsError(
            f"conversion output already exists: {path} or {report_path}"
        )
    chosen_device = torch.device(device)
    if chosen_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA conversion requested but CUDA is unavailable")
    start = time.perf_counter()
    prepared = recipe.prepare(device=str(chosen_device), rows_per_chunk=rows_per_chunk)
    preparation_seconds = time.perf_counter() - start
    metadata = {} if name is None else {"name": name}
    provenance = {} if provenance is None else provenance
    resources = [
        ResourceSpec(key, len(value)) for key, value in model.resources.items()
    ]
    specs = (
        resources
        + [job.spec for job in prepared.weights]
        + [spec for spec, _ in prepared.auxiliaries]
    )
    report = {
        "components": model.components,
        "name": name,
        "output": str(path),
        "device": str(chosen_device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "rows_per_chunk": rows_per_chunk,
        "provenance": provenance,
        "preparation_seconds": preparation_seconds,
        "parameters": len(model.parameters),
        "uses": len(prepared.uses),
        "objects": len(specs),
        "formats": dict(Counter(job.spec.format for job in prepared.weights)),
        "methods": [
            {
                "object": job.spec.id,
                "parameters": job.parameters,
                "sources": job.sources,
                "method": job.method_name,
                "method_parameters": job.method_parameters,
                "format": job.spec.format,
                "layout": job.spec.layout,
                "shape": job.spec.shape,
            }
            for job in prepared.weights
        ],
    }
    json.dumps(report, allow_nan=False, default=_json_default)
    with ArtifactWriter(
        path,
        specs,
        components=model.components,
        bindings=prepared.bindings,
        uses=prepared.uses,
        metadata=metadata,
        provenance=provenance,
        max_file_bytes=max_file_bytes,
    ) as writer:
        for object_id, data in model.resources.items():
            writer.write_object(object_id, data)
        for index, job in enumerate(prepared.weights):
            if progress is not None:
                progress(index, len(prepared.weights), job)
            try:
                job.prepared.produce(TensorOutput(writer, job.spec.id))
            except Exception as error:
                label = ", ".join(job.parameters[:4])
                raise ValueError(
                    f"{label} [{job.method_name}/{job.spec.format}]: {error}"
                ) from error
        for spec, data in prepared.auxiliaries:
            writer.write_object(spec.id, data)
        report["artifact_id"] = writer.artifact_id.hex()
        report["files"] = [
            {
                "path": str(path if i == 0 else path.parent / file.path),
                "payload_bytes": file.payload_bytes,
            }
            for i, file in enumerate(writer.directory.files)
        ]
        report["payload_bytes"] = writer.directory.payload_bytes
    report["seconds"] = time.perf_counter() - start
    temporary = Path(str(report_path) + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(
                report,
                stream,
                ensure_ascii=False,
                allow_nan=False,
                default=_json_default,
                indent=2,
            )
            stream.write("\n")
        temporary.replace(report_path)
    finally:
        temporary.unlink(missing_ok=True)
    return report
