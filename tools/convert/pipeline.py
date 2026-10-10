"""Complete conversion from a configured logical model to a v3 file collection."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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


class _BufferedObjectWriter:
    """Private single-object sink used by one CPU conversion worker."""

    def __init__(self, obj):
        self.by_id = {obj.id: obj}
        self._object = obj
        self._data = bytearray(obj.bytes)
        self._ranges = []

    def _record(self, object_id, offset, length):
        if object_id != self._object.id:
            raise ValueError(f"unexpected buffered object {object_id!r}")
        if offset < 0 or length < 0 or offset + length > self._object.bytes:
            raise ValueError(
                f"{object_id}: buffered output [{offset},{offset+length}) "
                f"exceeds {self._object.bytes} bytes"
            )
        if length:
            self._ranges.append((offset, offset + length))

    def write_region(self, object_id, offset, data):
        view = memoryview(data).cast("B")
        self._record(object_id, offset, len(view))
        self._data[offset : offset + len(view)] = view

    def write_zeros(self, object_id, offset, length):
        # bytearray() is already zero-filled. Only record coverage.
        self._record(object_id, offset, length)

    def finish(self):
        cursor = 0
        for begin, end in sorted(self._ranges):
            if begin < cursor:
                raise ValueError(
                    f"{self._object.id}: duplicate buffered coverage "
                    f"at [{begin},{end})"
                )
            if begin > cursor:
                raise ValueError(
                    f"{self._object.id}: missing buffered coverage "
                    f"at [{cursor},{begin})"
                )
            cursor = end

        if cursor != self._object.bytes:
            raise ValueError(
                f"{self._object.id}: missing buffered coverage "
                f"at [{cursor},{self._object.bytes})"
            )

        return self._data


def _produce_buffered(job, obj):
    sink = _BufferedObjectWriter(obj)
    job.prepared.produce(TensorOutput(sink, job.spec.id))
    return sink.finish()


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
    workers: int = 1,
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
    if type(workers) is not int or workers <= 0:
        raise ValueError("--workers must be a positive integer")
    if chosen_device.type != "cpu" and workers != 1:
        raise ValueError("--workers > 1 is currently supported only with --device cpu")
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
        "workers": workers,
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
        if workers == 1:
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
        else:
            total = len(prepared.weights)
            window = min(total, workers * 2)

            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="ninfer-convert",
            ) as executor:
                futures = {}

                def submit(index):
                    job = prepared.weights[index]
                    if progress is not None:
                        progress(index, total, job)
                    obj = writer.by_id[job.spec.id]
                    futures[index] = executor.submit(
                        _produce_buffered,
                        job,
                        obj,
                    )

                for index in range(window):
                    submit(index)

                next_submit = window

                for index, job in enumerate(prepared.weights):
                    future = futures.pop(index)

                    try:
                        data = future.result()
                    except Exception as error:
                        label = ", ".join(job.parameters[:4])
                        raise ValueError(
                            f"{label} [{job.method_name}/{job.spec.format}]: {error}"
                        ) from error

                    # Publication remains strictly ordered and single-threaded.
                    writer.write_object(job.spec.id, data)

                    if next_submit < total:
                        submit(next_submit)
                        next_submit += 1
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
