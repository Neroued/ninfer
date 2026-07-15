from pathlib import Path

import torch

from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6_35b_a3b_rtx5090 import (
    convert,
    draft_head,
    inventory,
    recipe,
)


def test_preplanned_directory_matches_complete_inventory_and_byte_contract() -> None:
    convert.preflight_inventory()
    resources = {spec.name: b"x" for spec in inventory.RESOURCE_SPECS}
    plan = convert.build_object_plan(resources)

    names = tuple(spec.name for spec in inventory.OBJECT_SPECS)
    assert tuple(spec.name for spec in plan.specs) == names
    assert tuple(obj.name for obj in plan.objects) == names
    assert len(plan.objects) == 889
    assert (
        family_conversion.tensor_payload_bytes(inventory.TENSOR_SPECS)
        == 22_360_191_904
    )
    assert (
        family_conversion.device_arena_bytes(inventory.TENSOR_SPECS)
        == 22_360_207_360
    )
    assert convert.EXPECTED_COMPONENT_BYTES == {
        "main_text": 21_038_461_952,
        "draft_head": 143_130_624,
        "mtp": 897_934_336,
        "vision": 280_664_992,
    }


def test_report_retains_target_specific_provenance_and_component_bytes(
    tmp_path: Path,
) -> None:
    resources = {spec.name: b"x" for spec in inventory.RESOURCE_SPECS}
    plan = convert.build_object_plan(resources)
    source = recipe.SourcePreflight(883, 1045, 26, {"BF16": 1045})
    report = convert.build_conversion_report(
        model_dir=tmp_path / "model",
        out_path=tmp_path / "model.ninfer",
        arguments={},
        config_summary={"text": {"hidden_size": 2048}},
        source_preflight=source,
        objects=plan.objects,
        elapsed_seconds=1.0,
        final_bytes=123,
        device=torch.device("cpu"),
        ranking_path=draft_head.DEFAULT_RANKING,
        revision="test-revision",
        environment={"python": "test"},
    )

    assert report["source"]["gguf_evidence_path"] == str(
        convert.GGUF_EVIDENCE_PATH
    )
    assert report["draft_head"] == {
        "rows": 131072,
        "tokenizer_vocab_size": 248077,
        "ranking_source_target": "qwen3_6_27b_rtx5090",
        "shared_semantic_vocabulary": True,
    }
    assert report["quantization"] == {
        "encoder_profile": "MAXABS_F16_RECIP_RNE_V1",
        "component_tensor_bytes": {
            **convert.EXPECTED_COMPONENT_BYTES,
            "total": 22_360_191_904,
            "device_arena": 22_360_207_360,
        },
    }


def test_config_validation_requires_exact_moe_and_layer_schedule() -> None:
    text = dict(convert._TEXT_CONFIG)
    text["layer_types"] = [
        "full_attention"
        if layer in inventory.FULL_ATTENTION_LAYERS
        else "linear_attention"
        for layer in inventory.TEXT_LAYERS
    ]
    text["rope_parameters"] = dict(convert._ROPE_CONFIG)
    config = {
        **convert._ROOT_CONFIG,
        "text_config": text,
        "vision_config": dict(convert._VISION_CONFIG),
    }
    summary = convert.validate_config(config)
    assert summary["layer_types"]["full_attention_layers"] == list(
        inventory.FULL_ATTENTION_LAYERS
    )
    assert summary["text"]["num_experts"] == 256
    assert summary["text"]["num_experts_per_tok"] == 8

    text["num_experts_per_tok"] = 4
    try:
        convert.validate_config(config)
    except ValueError as error:
        assert "num_experts_per_tok" in str(error)
    else:
        raise AssertionError("invalid MoE top-k was accepted")
