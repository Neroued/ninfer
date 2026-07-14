from __future__ import annotations

import json
import struct

import pytest

from tools.artifact.container import (
    MAGIC,
    PAYLOAD_ALIGNMENT,
    PREFIX,
    Artifact,
    ArtifactError,
    ResourceSpec,
    TensorSpec,
    write_artifact,
)
from tools.artifact.inspect import artifact_summary
from tools.artifact.layouts import align_up, encoded_size


def _small_specs():
    return [
        ResourceSpec("frontend/tokenizer.json", "raw-bytes-v1", 2),
        TensorSpec("direct/bf16", (2,), "BF16", "contiguous-le-v1"),
        TensorSpec("direct/fp32", (1,), "FP32", "contiguous-le-v1"),
        TensorSpec("direct/i32", (1,), "I32", "contiguous-le-v1"),
        TensorSpec("quant/q4", (1, 64), "Q4G64_F16S", "row-split-k128-v1"),
        TensorSpec("quant/q5", (1, 64), "Q5G64_F16S", "row-split-k128-v1"),
        TensorSpec("quant/q6", (1, 64), "Q6G64_F16S", "row-split-k128-v1"),
        TensorSpec("quant/w8", (1, 32), "W8G32_F16S", "row-split-k128-v1"),
    ]


def _payload(spec):
    if isinstance(spec, ResourceSpec):
        return b"{}"
    size = encoded_size(spec.layout, spec.format, spec.shape)
    return bytes(size)


def test_v1_round_trip_covers_every_registered_storage(tmp_path):
    path = tmp_path / "small.ninfer"
    specs = _small_specs()
    entries = [(spec, _payload(spec)) for spec in specs]
    planned = write_artifact(path, "test-model", entries)

    prefix = path.read_bytes()[: PREFIX.size]
    magic, json_bytes = PREFIX.unpack(prefix)
    assert magic == MAGIC
    assert prefix[8:] == struct.pack("<Q", json_bytes)

    with Artifact.open(path) as artifact:
        assert artifact.model_id == "test-model"
        assert artifact.payload_offset == align_up(PREFIX.size + json_bytes, PAYLOAD_ALIGNMENT)
        assert artifact.objects == planned
        for spec, expected in entries:
            assert bytes(artifact.payload(spec.name)) == expected
        summary = artifact_summary(artifact)
        assert summary["objects"] == 8
        assert summary["formats"] == {
            "BF16": 1,
            "FP32": 1,
            "I32": 1,
            "Q4G64_F16S": 1,
            "Q5G64_F16S": 1,
            "Q6G64_F16S": 1,
            "W8G32_F16S": 1,
        }


def _write_raw(path, metadata: dict[str, object], payload: bytes = b"") -> None:
    encoded = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    payload_offset = align_up(PREFIX.size + len(encoded), PAYLOAD_ALIGNMENT)
    path.write_bytes(
        PREFIX.pack(MAGIC, len(encoded))
        + encoded
        + bytes(payload_offset - PREFIX.size - len(encoded))
        + payload
    )


def test_reader_rejects_invalid_framing_schema_and_geometry(tmp_path):
    path = tmp_path / "invalid.ninfer"

    path.write_bytes(PREFIX.pack(MAGIC, 100) + b"{}")
    with pytest.raises(ArtifactError, match="beyond the file"):
        Artifact.open(path)

    root = {
        "model_id": "test-model",
        "objects": [
            {
                "name": "a",
                "kind": "tensor",
                "shape": [1],
                "format": "I32",
                "layout": "contiguous-le-v1",
                "offset": 0,
                "bytes": 4,
            },
            {
                "name": "b",
                "kind": "resource",
                "encoding": "raw-bytes-v1",
                "offset": 2,
                "bytes": 2,
            },
        ],
    }
    _write_raw(path, root, b"\x00" * 4)
    with pytest.raises(ArtifactError, match="overlaps"):
        Artifact.open(path)

    root["source_recipe"] = "must not enter the container"
    _write_raw(path, root, b"\x00" * 4)
    with pytest.raises(ArtifactError, match="exactly"):
        Artifact.open(path)

    root = {
        "model_id": "test-model",
        "objects": [
            {
                "name": "bad-size",
                "kind": "tensor",
                "shape": [2],
                "format": "BF16",
                "layout": "contiguous-le-v1",
                "offset": 0,
                "bytes": 2,
            }
        ],
    }
    _write_raw(path, root, b"\x00" * 2)
    with pytest.raises(ArtifactError, match="layout requires"):
        Artifact.open(path)
