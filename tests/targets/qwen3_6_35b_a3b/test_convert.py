from pathlib import Path

import torch

from tools.convert.qwen3_6_35b_a3b import (
    convert,
    draft_head,
    inventory,
    recipe,
)


def test_report_retains_target_specific_provenance_and_component_bytes(
    tmp_path: Path,
) -> None:
    resources = {spec.name: b"x" for spec in inventory.RESOURCE_SPECS}
    plan = convert.build_object_plan(resources)
    base_source = recipe.SourcePreflight(883, 1045, 26, {"BF16": 1045})
    dflash_source = recipe.SourcePreflight(51, 69, 1, {"BF16": 69})
    report = convert.build_conversion_report(
        model_dir=tmp_path / "model",
        dflash_model_dir=tmp_path / "dflash",
        out_path=tmp_path / "model.ninfer",
        arguments={},
        base_config_summary={"text": {"hidden_size": 2048}},
        dflash_config_summary={"hidden_size": 2048},
        base_source_preflight=base_source,
        dflash_source_preflight=dflash_source,
        objects=plan.objects,
        elapsed_seconds=1.0,
        final_bytes=123,
        device=torch.device("cpu"),
        ranking_path=draft_head.DEFAULT_RANKING,
        revision="test-revision",
        environment={"python": "test"},
    )

    assert report["model_id"] == inventory.MODEL_ID
    assert report["target_key"] == inventory.TARGET_KEY
    assert report["recipe_id"] == convert.RECIPE_ID
    assert report["source"]["base_model_path"] == str(
        (tmp_path / "model").resolve()
    )
    assert report["source"]["dflash_model_path"] == str(
        (tmp_path / "dflash").resolve()
    )
    assert report["source_preflight"]["base"]["tensors"] == 1045
    assert report["source_preflight"]["dflash"]["tensors"] == 69
    assert report["source_preflight"]["combined"]["tensors"] == 1114
    assert report["source"]["gguf_evidence_path"] == str(
        convert.GGUF_EVIDENCE_PATH
    )
    assert report["draft_head"] == {
        "rows": 131072,
        "tokenizer_vocab_size": 248077,
        "ranking_source_target": "qwen3_6_27b",
        "shared_semantic_vocabulary": True,
    }
    assert report["quantization"] == {
        "encoder_profile": "MAXABS_F16_RECIP_RNE_V1",
        "component_tensor_bytes": {
            **convert.EXPECTED_COMPONENT_BYTES,
            "total": 22_770_245_536,
            "all_tensor_device_arena": 22_770_260_992,
            "default_resident": 22_360_191_904,
            "default_resident_device_arena": 22_360_207_360,
        },
    }
