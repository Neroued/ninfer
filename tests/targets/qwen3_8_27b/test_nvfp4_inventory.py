from tools.convert.qwen3_8_27b import convert_nvfp4
from tools.convert.qwen3_8_27b import inventory_nvfp4 as inventory
from tools.convert.qwen3_8_27b import recipe_nvfp4 as recipe


def _tensors() -> dict[str, inventory.TensorSpec]:
    return {spec.name: spec for spec in inventory.TENSOR_SPECS}


def test_complete_nvfp4_inventory_and_format_allocation() -> None:
    assert (
        inventory.MODEL_ID,
        inventory.WEIGHTS_ID,
        inventory.TARGET_KEY,
    ) == ("qwen3.8-27b", "nvfp4", "qwen3_8_27b")
    assert len(inventory.OBJECT_SPECS) == 1124
    assert len(inventory.TENSOR_SPECS) == 1118
    assert len(inventory.RESOURCE_SPECS) == 6
    assert inventory.FORMAT_COUNTS == {
        "BF16": 534,
        "FP32": 208,
        "I32": 1,
        "Q4G64_F16S": 55,
        "Q5G64_F16S": 54,
        "Q6G64_F16S": 1,
        "W8G32_F16S": 7,
        "NVFP4": 112,
        "FP8_E4M3FN_ROW_BF16S": 146,
    }
    assert inventory.LAYOUT_COUNTS == {
        "contiguous-le-v1": 743,
        "row-split-k128-v1": 117,
        "blockscale-k16-m128x4-v1": 112,
        "row-scale-v1": 146,
    }
    assert len(inventory.LOGICAL_ROW_VIEW_SPECS) == 18
    assert len(inventory.ALIAS_SPECS) == 4
    assert convert_nvfp4.RECIPE_ID == "qwen3_8_27b_nvfp4-v1"
    assert convert_nvfp4.OUTPUT_BASENAME == "qwen3_8_27b_nvfp4.ninfer"


def test_fused_parent_signatures_and_no_avoidable_split_objects() -> None:
    tensors = _tensors()
    assert tensors["text/token_embedding"] == inventory.TensorSpec(
        "text/token_embedding",
        (248320, 5120),
        "FP8_E4M3FN_ROW_BF16S",
        "row-scale-v1",
    )
    assert tensors[
        "text/layers/3/attention/query_key_gate_value"
    ] == inventory.TensorSpec(
        "text/layers/3/attention/query_key_gate_value",
        (14336, 5120),
        "FP8_E4M3FN_ROW_BF16S",
        "row-scale-v1",
    )
    assert tensors["text/layers/0/gdn/a_b_projection"].shape == (96, 5120)
    assert tensors["text/layers/0/gdn/query_key_value_z"].shape == (
        16384,
        5120,
    )
    assert tensors["text/layers/0/mlp/gate_up"] == inventory.TensorSpec(
        "text/layers/0/mlp/gate_up",
        (34816, 5120),
        "NVFP4",
        "blockscale-k16-m128x4-v1",
    )
    assert tensors["text/layers/56/mlp/gate_up"].format == inventory.FP8
    for forbidden in (
        "text/layers/3/attention/query",
        "text/layers/3/attention/key",
        "text/layers/0/gdn/query",
        "text/layers/0/gdn/z",
        "text/layers/0/mlp/gate",
        "text/layers/0/mlp/up",
    ):
        assert forbidden not in tensors


def test_closed_dual_source_recipe_and_tensor_ownership() -> None:
    assert (
        len(recipe.FP8_WEIGHT_RECIPES),
        len(recipe.NVFP4_WEIGHT_RECIPES),
        len(recipe.INPUT_DIVISOR_RECIPES),
        len(recipe.WEIGHT_DIVISOR_GROUPS),
        len(recipe.FP8_SOURCES),
        len(recipe.NVFP4_SOURCES),
        len(recipe.QUANTIZED_DIRECT_RECIPES),
        len(recipe.OFFICIAL_RECIPES),
        len(recipe.SOURCE_REQUIREMENTS),
    ) == (145, 112, 112, 56, 233, 168, 401, 348, 1587)
    dtype_counts: dict[str, int] = {}
    for _, dtype in recipe.SOURCE_REQUIREMENTS.values():
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    assert dtype_counts == {
        "BF16": 682,
        "F8_E4M3": 401,
        "U8": 168,
        "F32": 336,
    }
    assert recipe.OFFICIAL_EMBEDDING_SOURCE.name == (
        "model.language_model.embed_tokens.weight"
    )

    owned = (
        {"text/token_embedding"}
        | set(recipe.FP8_WEIGHTS_BY_NAME)
        | set(recipe.NVFP4_WEIGHTS_BY_NAME)
        | set(recipe.INPUT_DIVISORS_BY_NAME)
        | set(recipe.QUANTIZED_DIRECT_BY_NAME)
        | set(recipe.OFFICIAL_RECIPES_BY_NAME)
    )
    assert owned == {spec.name for spec in inventory.TENSOR_SPECS}
