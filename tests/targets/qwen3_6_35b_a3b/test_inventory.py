from collections import Counter

import pytest

from tools.convert.qwen3_6_35b_a3b_rtx5090 import inventory


def _tensors() -> dict[str, inventory.TensorSpec]:
    return {spec.name: spec for spec in inventory.TENSOR_SPECS}


def test_complete_inventory_order_formats_and_sizes() -> None:
    assert inventory.MODEL_ID == "qwen3.6-35b-a3b"
    assert inventory.TARGET_KEY == "qwen3_6_35b_a3b_rtx5090"
    assert (
        len(inventory.TEXT_CORE_TENSOR_SPECS),
        len(inventory.DRAFT_HEAD_TENSOR_SPECS),
        len(inventory.MTP_TENSOR_SPECS),
        len(inventory.VISION_TENSOR_SPECS),
        len(inventory.TENSOR_SPECS),
        len(inventory.RESOURCE_SPECS),
        len(inventory.OBJECT_SPECS),
    ) == (533, 2, 15, 333, 883, 6, 889)

    names = tuple(spec.name for spec in inventory.OBJECT_SPECS)
    assert len(names) == len(set(names))
    assert names[:6] == (
        "frontend/tokenizer.json",
        "frontend/tokenizer_config.json",
        "frontend/chat_template.jinja",
        "frontend/generation_config.json",
        "frontend/preprocessor_config.json",
        "frontend/video_preprocessor_config.json",
    )
    assert names[6] == "text/token_embedding"
    positions = {name: index for index, name in enumerate(names)}
    assert positions["text/layers/39/moe/shared_down"] < positions["text/final_norm"]
    assert positions["text/draft_head_token_ids"] < positions["mtp/input_projection"]
    assert positions["mtp/final_norm"] < positions["vision/patch_embedding"]

    assert inventory.FORMAT_COUNTS == {
        inventory.BF16: 461,
        inventory.FP32: 60,
        inventory.I32: 1,
        inventory.Q4: 95,
        inventory.Q5: 91,
        inventory.Q6: 5,
        inventory.W8: 170,
    }
    assert inventory.LAYOUT_COUNTS == {
        inventory.CONTIGUOUS_LAYOUT: 522,
        inventory.ROW_SPLIT_LAYOUT: 361,
    }
    assert Counter(spec.format for spec in inventory.TENSOR_SPECS) == Counter(
        inventory.FORMAT_COUNTS
    )


def test_key_dense_moe_mtp_and_vision_signatures() -> None:
    tensors = _tensors()
    assert tensors["text/token_embedding"] == inventory.TensorSpec(
        "text/token_embedding", (248320, 2048), inventory.W8, inventory.ROW_SPLIT_LAYOUT
    )
    assert tensors["text/layers/3/attention/query_key_gate_value"].shape == (
        9216,
        2048,
    )
    assert tensors["text/layers/3/attention/query_key_gate_value"].format == inventory.W8
    assert tensors["text/layers/0/gdn/a_b_projection"] == inventory.TensorSpec(
        "text/layers/0/gdn/a_b_projection",
        (64, 2048),
        inventory.BF16,
        inventory.CONTIGUOUS_LAYOUT,
    )
    assert tensors["text/layers/0/gdn/query_key_value_z"].shape == (12288, 2048)
    assert tensors["text/layers/0/moe/routed_gate_up"].shape == (262144, 2048)
    assert tensors["text/layers/0/moe/routed_gate_up"].format == inventory.Q4
    assert tensors["text/layers/0/moe/routed_down"].shape == (524288, 512)
    assert tensors["text/layers/34/moe/routed_down"].format == inventory.Q6
    assert tensors["text/layers/38/moe/routed_down"].format == inventory.Q6
    assert tensors["text/layers/39/moe/routed_down"].format == inventory.Q6
    assert tensors["text/layers/33/moe/routed_down"].format == inventory.Q5
    assert tensors["mtp/layer/moe/routed_gate_up"].format == inventory.W8
    assert tensors["mtp/layer/moe/routed_down"].shape == (524288, 512)
    assert tensors["vision/merger/fc2"].shape == (2048, 4608)

    assert inventory.FULL_ATTENTION_LAYERS == tuple(range(3, 40, 4))
    assert len(inventory.GDN_LAYERS) == 30
    assert inventory.Q6_ROUTED_DOWN_LAYERS == (34, 38, 39)


def test_expert_row_addressing_is_half_split_and_expert_major() -> None:
    assert inventory.routed_gate_up_row(0, 0, 0) == 0
    assert inventory.routed_gate_up_row(0, 1, 0) == 512
    assert inventory.routed_gate_up_row(1, 0, 0) == 1024
    assert inventory.routed_gate_up_row(255, 1, 511) == 262143
    assert inventory.routed_down_row(0, 0) == 0
    assert inventory.routed_down_row(1, 0) == 2048
    assert inventory.routed_down_row(255, 2047) == 524287

    with pytest.raises(IndexError):
        inventory.routed_gate_up_row(0, 2, 0)
    with pytest.raises(IndexError):
        inventory.routed_down_row(256, 0)


def test_logical_views_cover_fused_dense_and_shared_rows() -> None:
    assert len(inventory.LOGICAL_ROW_VIEW_SPECS) == 22
    views = {view.name_pattern: view for view in inventory.LOGICAL_ROW_VIEW_SPECS}
    assert (
        views["text/layers/{l}/attention/key"].row_begin,
        views["text/layers/{l}/attention/key"].row_end,
    ) == (4096, 4608)
    assert (
        views["text/layers/{l}/gdn/a_projection"].row_begin,
        views["text/layers/{l}/gdn/b_projection"].row_begin,
    ) == (0, 32)
    assert (
        views["text/layers/{l}/moe/shared_expert/gate"].row_begin,
        views["text/layers/{l}/moe/shared_expert/up"].row_begin,
    ) == (0, 512)
    assert len(inventory.ALIAS_SPECS) == 4
