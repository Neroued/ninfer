from collections import Counter

from tools.convert.qwen3_6_27b_rtx5090 import inventory


def _tensor_by_name() -> dict[str, inventory.TensorSpec]:
    return {spec.name: spec for spec in inventory.TENSOR_SPECS}


def test_complete_full_only_inventory_and_canonical_order() -> None:
    assert inventory.MODEL_ID == "qwen3.6-27b"
    assert inventory.TARGET_KEY == "qwen3_6_27b_rtx5090"

    assert len(inventory.TEXT_CORE_TENSOR_SPECS) == 819
    assert len(inventory.DRAFT_HEAD_TENSOR_SPECS) == 2
    assert len(inventory.MTP_TENSOR_SPECS) == 12
    assert len(inventory.VISION_TENSOR_SPECS) == 333
    assert len(inventory.TENSOR_SPECS) == 1166
    assert len(inventory.RESOURCE_SPECS) == 6
    assert len(inventory.OBJECT_SPECS) == 1172

    names = [spec.name for spec in inventory.OBJECT_SPECS]
    assert len(names) == len(set(names))
    assert tuple(names[:6]) == (
        "frontend/tokenizer.json",
        "frontend/tokenizer_config.json",
        "frontend/chat_template.jinja",
        "frontend/generation_config.json",
        "frontend/preprocessor_config.json",
        "frontend/video_preprocessor_config.json",
    )
    assert names[6] == "text/token_embedding"
    assert names[-1] == "vision/merger/norm/bias"

    positions = {name: index for index, name in enumerate(names)}
    assert positions["text/layers/0/input_norm"] < positions["text/layers/63/input_norm"]
    assert positions["text/layers/63/mlp/down"] < positions["text/final_norm"]
    assert positions["text/final_norm"] < positions["text/output_head"]
    assert positions["text/output_head"] < positions["text/draft_head"]
    assert positions["text/draft_head_token_ids"] < positions["mtp/input_projection"]
    assert positions["mtp/final_norm"] < positions["vision/patch_embedding"]
    assert positions["vision/layers/26/norm2/bias"] < positions["vision/merger/fc1"]


def test_format_layout_counts_and_key_signatures() -> None:
    assert inventory.FORMAT_COUNTS == {
        "BF16": 582,
        "FP32": 96,
        "I32": 1,
        "Q4G64_F16S": 183,
        "Q5G64_F16S": 294,
        "Q6G64_F16S": 3,
        "W8G32_F16S": 7,
    }
    assert inventory.LAYOUT_COUNTS == {
        "contiguous-le-v1": 679,
        "row-split-k128-v1": 487,
    }
    assert Counter(spec.format for spec in inventory.TENSOR_SPECS) == inventory.FORMAT_COUNTS
    assert Counter(spec.layout for spec in inventory.TENSOR_SPECS) == inventory.LAYOUT_COUNTS

    tensors = _tensor_by_name()
    assert tensors["text/token_embedding"] == inventory.TensorSpec(
        "text/token_embedding", (248320, 5120), "Q6G64_F16S", "row-split-k128-v1"
    )
    assert tensors["text/layers/0/gdn/convolution"] == inventory.TensorSpec(
        "text/layers/0/gdn/convolution", (4, 10240), "BF16", "contiguous-le-v1"
    )
    assert tensors["text/layers/3/attention/query_key"] == inventory.TensorSpec(
        "text/layers/3/attention/query_key",
        (7168, 5120),
        "Q4G64_F16S",
        "row-split-k128-v1",
    )
    assert tensors["text/draft_head_token_ids"].shape == (131072,)
    assert tensors["text/draft_head_token_ids"].format == "I32"
    assert tensors["mtp/layer/attention/query_key_gate_value"].shape == (14336, 5120)
    assert tensors["mtp/layer/attention/query_key_gate_value"].format == "W8G32_F16S"
    assert tensors["vision/patch_embedding"].shape == (1152, 1536)
    assert tensors["vision/merger/fc2"].shape == (5120, 4608)

    assert inventory.FULL_ATTENTION_LAYERS == tuple(range(3, 64, 4))
    assert len(inventory.GDN_LAYERS) == 48


def test_fixed_logical_row_views_and_aliases() -> None:
    assert len(inventory.LOGICAL_ROW_VIEW_SPECS) == 14
    assert len({view.name_pattern for view in inventory.LOGICAL_ROW_VIEW_SPECS}) == 14
    for view in inventory.LOGICAL_ROW_VIEW_SPECS:
        assert view.row_begin >= 0
        assert view.row_end > view.row_begin
        assert view.row_count == view.shape[0]
        assert view.shape[1] == 5120

    views = {view.name_pattern: view for view in inventory.LOGICAL_ROW_VIEW_SPECS}
    assert (
        views["text/layers/{l}/attention/key"].parent_pattern,
        views["text/layers/{l}/attention/key"].row_begin,
        views["text/layers/{l}/attention/key"].row_end,
        views["text/layers/{l}/attention/key"].layers,
    ) == (
        "text/layers/{l}/attention/query_key",
        6144,
        7168,
        inventory.FULL_ATTENTION_LAYERS,
    )
    assert (
        views["text/layers/{l}/gdn/key"].row_begin,
        views["text/layers/{l}/gdn/key"].row_end,
        views["text/layers/{l}/gdn/key"].layers,
    ) == (2048, 4096, inventory.GDN_LAYERS)
    assert (
        views["mtp/layer/attention/output_gate"].row_begin,
        views["mtp/layer/attention/output_gate"].row_end,
    ) == (7168, 13312)
    assert (
        views["mtp/layer/attention/value"].row_begin,
        views["mtp/layer/attention/value"].row_end,
    ) == (13312, 14336)

    assert len(inventory.ALIAS_SPECS) == 4
    aliases = {alias.role_pattern: alias for alias in inventory.ALIAS_SPECS}
    assert aliases["mtp/token_embedding"].object_patterns == ("text/token_embedding",)
    assert aliases["mtp/full_output_head"].object_patterns == ("text/output_head",)
    assert aliases["mtp/optimized_proposal_head"].object_patterns == (
        "text/draft_head",
        "text/draft_head_token_ids",
    )
    convolution = aliases["text/layers/{l}/gdn/channel_major_convolution"]
    assert convolution.object_patterns == ("text/layers/{l}/gdn/convolution",)
    assert convolution.layers == inventory.GDN_LAYERS
    assert convolution.axis_order == (1, 0)
