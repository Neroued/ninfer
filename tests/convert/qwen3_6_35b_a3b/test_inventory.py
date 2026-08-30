from tools.convert.qwen3_6_35b_a3b import inventory


def test_identity() -> None:
    assert inventory.MODEL_ID == "qwen3.6-35b-a3b"
    assert inventory.WEIGHTS_ID == "groupwise-int"
    assert inventory.TARGET_KEY == "qwen3_6_35b_a3b"


def _tensors() -> dict[str, inventory.TensorSpec]:
    return {spec.name: spec for spec in inventory.TENSOR_SPECS}


def test_key_dense_moe_mtp_and_vision_signatures() -> None:
    tensors = _tensors()
    assert tensors["text/token_embedding"] == inventory.TensorSpec(
        "text/token_embedding",
        (248320, 2048),
        inventory.W8,
        inventory.ROW_SPLIT_LAYOUT,
    )
    assert tensors["text/draft_head"] == inventory.TensorSpec(
        "text/draft_head",
        (131072, 2048),
        inventory.Q4,
        inventory.ROW_SPLIT_LAYOUT,
    )
    assert tensors["text/draft_head_token_ids"].shape == (131072,)
    assert tensors["text/draft_head_token_ids"].format == inventory.I32
    assert tensors["text/layers/3/attention/query_key_gate_value"].shape == (
        9216,
        2048,
    )
    assert (
        tensors["text/layers/3/attention/query_key_gate_value"].format
        == inventory.W8
    )
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
    assert tensors["dflash/feature_projection"] == inventory.TensorSpec(
        "dflash/feature_projection",
        (2048, 16384),
        inventory.W8,
        inventory.ROW_SPLIT_LAYOUT,
    )
    assert tensors["dflash/layers/0/attention/query_key_value"].shape == (
        6144,
        2048,
    )
    assert tensors["dflash/layers/5/mlp/gate_up"].shape == (12288, 2048)
    assert tensors["dflash/layers/5/mlp/down"].shape == (2048, 6144)
    assert tensors["dflash/final_norm"].format == inventory.BF16

    assert inventory.FULL_ATTENTION_LAYERS == tuple(range(3, 40, 4))
    assert len(inventory.GDN_LAYERS) == 30
    assert inventory.Q6_ROUTED_DOWN_LAYERS == (34, 38, 39)
    assert inventory.DFLASH_LAYERS == tuple(range(6))
    assert len(inventory.DFLASH_TENSOR_SPECS) == 51
    assert len(inventory.TENSOR_SPECS) == 934
    assert len(inventory.OBJECT_SPECS) == 940


def test_runtime_view_parents_are_present_in_the_converted_inventory() -> None:
    """Protect physical parents consumed through runtime aliases and row views."""

    tensors = _tensors()
    assert tensors["text/token_embedding"].shape == (248320, 2048)
    assert tensors["text/output_head"].shape == (248320, 2048)
    assert tensors["text/draft_head"].shape == (131072, 2048)

    row_views = {
        "text/layers/3/attention/query_key_gate_value": (
            (0, 4096),
            (4096, 4608),
            (4608, 8704),
            (8704, 9216),
        ),
        "text/layers/0/gdn/a_b_projection": ((0, 32), (32, 64)),
        "text/layers/0/gdn/query_key_value_z": (
            (0, 2048),
            (2048, 4096),
            (4096, 8192),
            (8192, 12288),
        ),
        "text/layers/0/moe/router_shared_gate": ((0, 256), (256, 257)),
        "text/layers/0/moe/shared_gate_up": ((0, 512), (512, 1024)),
        "mtp/layer/attention/query_key_gate_value": (
            (0, 4096),
            (4096, 4608),
            (4608, 8704),
            (8704, 9216),
        ),
        "dflash/layers/0/attention/query_key_value": (
            (0, 4096),
            (4096, 5120),
            (5120, 6144),
        ),
        "dflash/layers/0/mlp/gate_up": ((0, 6144), (6144, 12288)),
    }
    for name, ranges in row_views.items():
        rows = tensors[name].shape[0]
        assert ranges[0][0] == 0
        assert ranges[-1][1] == rows
        assert all(begin < end for begin, end in ranges)
        assert all(
            left[1] == right[0]
            for left, right in zip(ranges, ranges[1:])
        )

    # DFlash reuses the Text embedding/output objects; its mask embedding is row 248077.
    assert 248077 < tensors["text/token_embedding"].shape[0]
