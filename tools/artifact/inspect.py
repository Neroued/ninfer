"""Inspect a `.ninfer` object directory without model-specific code."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .container import Artifact, ResourceObject, TensorObject


def artifact_summary(artifact: Artifact) -> dict[str, object]:
    tensors = [obj for obj in artifact.objects if isinstance(obj, TensorObject)]
    resources = [obj for obj in artifact.objects if isinstance(obj, ResourceObject)]
    return {
        "path": str(artifact.path),
        "model_id": artifact.model_id,
        "file_bytes": artifact.file_bytes,
        "payload_offset": artifact.payload_offset,
        "objects": len(artifact.objects),
        "tensors": len(tensors),
        "resources": len(resources),
        "formats": dict(sorted(Counter(obj.format for obj in tensors).items())),
        "layouts": dict(sorted(Counter(obj.layout for obj in tensors).items())),
        "encodings": dict(sorted(Counter(obj.encoding for obj in resources).items())),
    }


def _object_line(obj: TensorObject | ResourceObject) -> str:
    if isinstance(obj, TensorObject):
        shape = "[" + ",".join(str(dim) for dim in obj.shape) + "]"
        storage = f"{obj.format}/{obj.layout} {shape}"
    else:
        storage = obj.encoding
    return f"{obj.offset:>14} {obj.bytes:>14} {obj.kind:<8} {storage:<42} {obj.name}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--objects", action="store_true", help="list every stored object")
    parser.add_argument("--json", action="store_true", help="emit the summary as JSON")
    args = parser.parse_args()

    with Artifact.open(args.artifact) as artifact:
        summary = artifact_summary(artifact)
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print(f"model_id: {summary['model_id']}")
            print(
                f"objects: {summary['objects']} "
                f"({summary['tensors']} tensors, {summary['resources']} resources)"
            )
            print(f"file_bytes: {summary['file_bytes']}")
            print(f"payload_offset: {summary['payload_offset']}")
            print(f"formats: {summary['formats']}")
            print(f"layouts: {summary['layouts']}")
            print(f"encodings: {summary['encodings']}")
        if args.objects:
            for obj in artifact.objects:
                print(_object_line(obj))


if __name__ == "__main__":
    main()
