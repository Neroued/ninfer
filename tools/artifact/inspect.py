"""Inspect NInfer v3 configurations, objects, bindings and files without numerical libraries."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .reader import Artifact
from .schema import TensorObject


def artifact_summary(artifact: Artifact) -> dict:
    tensors = [obj for obj in artifact.objects if isinstance(obj, TensorObject)]
    return {
        "path": str(artifact.path),
        "version": 3,
        "artifact_id": artifact.artifact_id.hex(),
        "name": artifact.directory.metadata.get("name"),
        "components": artifact.directory.components,
        "file_bytes": artifact.file_bytes,
        "payload_bytes": artifact.payload_bytes,
        "files": len(artifact.directory.files),
        "objects": len(artifact.objects),
        "tensors": len(tensors),
        "resources": len(artifact.objects) - len(tensors),
        "bindings": len(artifact.directory.bindings),
        "uses": len(artifact.directory.uses),
        "formats": dict(sorted(Counter(obj.format for obj in tensors).items())),
        "layouts": dict(sorted(Counter(obj.layout for obj in tensors).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--objects", action="store_true", help="include physical object records"
    )
    parser.add_argument(
        "--bindings", action="store_true", help="include logical bindings and Uses"
    )
    parser.add_argument("--json", action="store_true", help="emit one JSON object")
    args = parser.parse_args()

    with Artifact(args.artifact) as artifact:
        summary = artifact_summary(artifact)
        if args.json:
            if args.objects:
                summary["object_records"] = [obj.to_json() for obj in artifact.objects]
            if args.bindings:
                summary["binding_records"] = artifact.directory.bindings
                summary["use_records"] = list(artifact.directory.uses)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return
        for key, value in summary.items():
            print(f"{key}: {value}")
        if args.objects:
            for obj in artifact.objects:
                storage = (
                    f"{obj.format}/{obj.layout} {list(obj.shape)}"
                    if isinstance(obj, TensorObject)
                    else obj.encoding
                )
                print(
                    f"{obj.offset:>14} {obj.bytes:>14} {obj.kind:<8} {storage:<42} {obj.id}"
                )
        if args.bindings:
            print(
                json.dumps(
                    {
                        "bindings": artifact.directory.bindings,
                        "uses": list(artifact.directory.uses),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
