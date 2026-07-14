from __future__ import annotations

from collections import Counter

import torch

from tools.artifact.container import (
    Artifact,
    ArtifactWriter,
    ResourceSpec,
    TensorSpec,
    plan_objects,
)
from tools.artifact.layouts import decode_direct, dequantize_row_split, encoded_size
from tools.convert.qwen3_6_27b_rtx5090 import convert, inventory, recipe


def test_complete_inventory_has_one_preplanned_object_directory():
    resources = {spec.name: b"x" for spec in inventory.RESOURCE_SPECS}
    plan = convert.build_object_plan(resources)

    assert len(plan.specs) == 1172
    assert len(plan.objects) == 1172
    assert tuple(obj.name for obj in plan.objects) == tuple(
        spec.name for spec in inventory.OBJECT_SPECS
    )
    assert tuple(obj.name for obj in plan.objects[:6]) == tuple(resources)
    assert plan.objects[6].name == "text/token_embedding"
    assert plan.objects[-1].name == "vision/merger/norm/bias"
    assert Counter(
        obj.format for obj in plan.objects if hasattr(obj, "format")
    ) == inventory.FORMAT_COUNTS
    assert plan.payload_span_bytes == plan.objects[-1].offset + plan.objects[-1].bytes


def test_synthetic_encode_seam_and_descriptive_report(tmp_path):
    direct_target = inventory.TensorSpec(
        "mini/direct", (3,), inventory.BF16, inventory.CONTIGUOUS_LAYOUT
    )
    quant_target = inventory.TensorSpec(
        "mini/quant", (2, 65), inventory.Q4, inventory.ROW_SPLIT_LAYOUT
    )
    direct = torch.tensor([1.0, -0.0, 3.5], dtype=torch.bfloat16)
    quant = torch.linspace(-2, 2, 130, dtype=torch.float32).reshape(2, 65).to(
        torch.bfloat16
    )
    direct_payload = convert.encode_tensor_payload(direct, direct_target, "cpu")
    quant_payload = convert.encode_tensor_payload(quant, quant_target, "cpu")
    assert len(direct_payload) == encoded_size(
        direct_target.layout, direct_target.format, direct_target.shape
    )
    assert len(quant_payload) == encoded_size(
        quant_target.layout, quant_target.format, quant_target.shape
    )

    artifact_specs = (
        ResourceSpec("frontend/test.json", "raw-bytes-v1", 2),
        TensorSpec(
            direct_target.name,
            direct_target.shape,
            direct_target.format,
            direct_target.layout,
        ),
        TensorSpec(
            quant_target.name,
            quant_target.shape,
            quant_target.format,
            quant_target.layout,
        ),
    )
    path = tmp_path / "mini.ninfer"
    with ArtifactWriter(path, "mini-model", artifact_specs) as writer:
        writer.write("frontend/test.json", b"{}")
        writer.write(direct_target.name, direct_payload)
        writer.write(quant_target.name, quant_payload)

    with Artifact.open(path) as artifact:
        assert torch.equal(
            decode_direct(
                artifact.payload(direct_target.name),
                direct_target.format,
                direct_target.shape,
            ),
            direct,
        )
        decoded_quant = dequantize_row_split(
            artifact.payload(quant_target.name),
            quant_target.format,
            quant_target.shape,
            dtype=torch.float32,
        )
        assert decoded_quant.shape == quant.shape
        torch.testing.assert_close(
            decoded_quant, quant.float(), atol=0.2, rtol=0.0
        )

    source = recipe.SourcePreflight(
        recipe_count=2,
        source_tensor_count=1,
        source_shard_count=1,
        source_dtype_counts={"BF16": 1},
    )
    report = convert.build_conversion_report(
        model_dir=tmp_path / "model",
        out_path=path,
        arguments={"model": "model", "out": str(path), "device": "cpu"},
        config_summary={"text": {"hidden_size": 4}},
        source_preflight=source,
        objects=plan_objects(artifact_specs),
        elapsed_seconds=0.5,
        final_bytes=path.stat().st_size,
        device=torch.device("cpu"),
        ranking_path=tmp_path / "ranking.i64",
        revision="test-revision",
        environment={"python": "test", "torch": "test", "device": "cpu"},
    )
    assert report["model_id"] == inventory.MODEL_ID
    assert report["target_key"] == inventory.TARGET_KEY
    assert report["recipe_id"] == convert.RECIPE_ID
    assert report["source"]["model_path"].endswith("/model")
    assert report["arguments"]["device"] == "cpu"
    assert report["source_preflight"] == {
        "recipes": 2,
        "tensors": 1,
        "shards": 1,
        "dtypes": {"BF16": 1},
    }
    assert report["converter"]["revision"] == "test-revision"
    assert report["objects"]["count"] == 3
    assert report["artifact"]["bytes"] == path.stat().st_size
