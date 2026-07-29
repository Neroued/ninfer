"""Activation parity metadata contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.parity.qwen3_6_27b.activations import compare


def _write_dump(root: Path, kv_dtype: str) -> None:
    root.mkdir()
    tensor = root / "embed.f32"
    np.array([1.0, -2.0], dtype=np.float32).tofile(tensor)
    manifest = {
        "format": "ninfer_activation_dump_v1",
        "kv_dtype": kv_dtype,
        "tensors": [
            {
                "phase": "prefill",
                "step": 0,
                "chunk": 0,
                "name": "embed",
                "shape": [2],
                "file": tensor.name,
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_matching_mixed_kv_dtype_is_compared(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_dump(reference, "int8:bf16")
    _write_dump(candidate, "int8:bf16")

    report = compare(reference, candidate, kv_dtype="int8:bf16")

    assert [item["status"] for item in report] == ["ok"]
    assert report[0]["kv_dtype"] == "int8:bf16"


def test_cross_format_comparison_is_rejected(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_dump(reference, "bf16")
    _write_dump(candidate, "bf16:int8")

    report = compare(reference, candidate, kv_dtype="bf16:int8")

    assert report == [
        {
            "key": ["metadata", "kv_dtype"],
            "status": "kv_dtype",
            "side": "reference",
            "expected": "bf16:int8",
            "actual": "bf16",
        }
    ]
