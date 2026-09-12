from tools.convert.qwen3_8_27b import inventory_nvfp4 as inventory


def _tensors() -> dict[str, inventory.TensorSpec]:
    return {spec.name: spec for spec in inventory.TENSOR_SPECS}


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
