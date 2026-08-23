"""Persistent-object contract for the Qwen3.8-27B NVFP4 artifact."""

from __future__ import annotations

from tools.convert.qwen3_6.common.inventory import (
    BF16,
    CONTIGUOUS_LAYOUT,
    FP32,
    I32,
    LogicalAliasSpec,
    LogicalRowViewSpec,
    Q4,
    Q5,
    Q6,
    RESOURCE_SPECS,
    ROW_SPLIT_LAYOUT,
    ResourceSpec,
    StoredObjectSpec,
    TensorSpec,
    W8,
    build_vision_specs,
)


MODEL_ID = "qwen3.8-27b"
WEIGHTS_ID = "nvfp4"
TARGET_KEY = "qwen3_8_27b"

NVFP4 = "NVFP4"
FP8 = "FP8_E4M3FN_ROW_BF16S"
BLOCK_SCALE_LAYOUT = "blockscale-k16-m128x4-v1"
ROW_SCALE_LAYOUT = "row-scale-v1"

FORMAT_NAMES = (BF16, FP32, I32, Q4, Q5, Q6, W8, NVFP4, FP8)
LAYOUT_NAMES = (
    CONTIGUOUS_LAYOUT,
    ROW_SPLIT_LAYOUT,
    BLOCK_SCALE_LAYOUT,
    ROW_SCALE_LAYOUT,
)

FULL_ATTENTION_LAYERS = tuple(range(3, 64, 4))
GDN_LAYERS = tuple(
    layer for layer in range(64) if layer not in FULL_ATTENTION_LAYERS
)
NVFP4_MLP_LAYERS = tuple(range(56))
FP8_MLP_LAYERS = tuple(range(56, 64))


def tensor_spec(
    name: str,
    shape: tuple[int, ...],
    numeric_format: str,
) -> TensorSpec:
    if numeric_format in (BF16, FP32, I32):
        layout = CONTIGUOUS_LAYOUT
    elif numeric_format in (Q4, Q5, Q6, W8):
        layout = ROW_SPLIT_LAYOUT
    elif numeric_format == NVFP4:
        layout = BLOCK_SCALE_LAYOUT
    elif numeric_format == FP8:
        layout = ROW_SCALE_LAYOUT
    else:
        raise ValueError(f"unsupported Qwen3.8 NVFP4 format: {numeric_format}")
    return TensorSpec(name, shape, numeric_format, layout)


def _build_text_core_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        tensor_spec("text/token_embedding", (248320, 5120), FP8),
    ]
    for layer in range(64):
        prefix = f"text/layers/{layer}/"
        specs.append(tensor_spec(prefix + "input_norm", (5120,), BF16))
        if layer in FULL_ATTENTION_LAYERS:
            specs.extend(
                (
                    tensor_spec(
                        prefix + "attention/query_key_gate_value",
                        (14336, 5120),
                        FP8,
                    ),
                    tensor_spec(prefix + "attention/query_norm", (256,), BF16),
                    tensor_spec(prefix + "attention/key_norm", (256,), BF16),
                    tensor_spec(
                        prefix + "attention/output", (5120, 6144), FP8
                    ),
                )
            )
        else:
            specs.extend(
                (
                    tensor_spec(prefix + "gdn/a_log", (48,), FP32),
                    tensor_spec(prefix + "gdn/dt_bias", (48,), FP32),
                    tensor_spec(
                        prefix + "gdn/convolution", (4, 10240), BF16
                    ),
                    tensor_spec(
                        prefix + "gdn/a_b_projection", (96, 5120), BF16
                    ),
                    tensor_spec(
                        prefix + "gdn/query_key_value_z",
                        (16384, 5120),
                        FP8,
                    ),
                    tensor_spec(prefix + "gdn/norm", (128,), BF16),
                    tensor_spec(prefix + "gdn/output", (5120, 6144), FP8),
                )
            )

        specs.append(
            tensor_spec(prefix + "post_attention_norm", (5120,), BF16)
        )
        if layer in NVFP4_MLP_LAYERS:
            specs.extend(
                (
                    tensor_spec(
                        prefix + "mlp/gate_up", (34816, 5120), NVFP4
                    ),
                    tensor_spec(
                        prefix
                        + "mlp/gate_up_projection/input_scale_divisor",
                        (),
                        FP32,
                    ),
                    tensor_spec(
                        prefix + "mlp/down", (5120, 17408), NVFP4
                    ),
                    tensor_spec(
                        prefix + "mlp/down_projection/input_scale_divisor",
                        (),
                        FP32,
                    ),
                )
            )
        else:
            specs.extend(
                (
                    tensor_spec(prefix + "mlp/gate_up", (34816, 5120), FP8),
                    tensor_spec(prefix + "mlp/down", (5120, 17408), FP8),
                )
            )

    specs.extend(
        (
            tensor_spec("text/final_norm", (5120,), BF16),
            tensor_spec("text/output_head", (248320, 5120), FP8),
        )
    )
    return tuple(specs)


def _build_draft_head_specs() -> tuple[TensorSpec, ...]:
    return (
        tensor_spec("text/draft_head", (131072, 5120), Q4),
        tensor_spec("text/draft_head_token_ids", (131072,), I32),
    )


def _build_mtp_specs() -> tuple[TensorSpec, ...]:
    return (
        tensor_spec("mtp/input_projection", (5120, 10240), W8),
        tensor_spec("mtp/embedding_norm", (5120,), BF16),
        tensor_spec("mtp/hidden_norm", (5120,), BF16),
        tensor_spec("mtp/layer/input_norm", (5120,), BF16),
        tensor_spec(
            "mtp/layer/attention/query_key_gate_value", (14336, 5120), W8
        ),
        tensor_spec("mtp/layer/attention/query_norm", (256,), BF16),
        tensor_spec("mtp/layer/attention/key_norm", (256,), BF16),
        tensor_spec("mtp/layer/attention/output", (5120, 6144), W8),
        tensor_spec("mtp/layer/post_attention_norm", (5120,), BF16),
        tensor_spec("mtp/layer/mlp/gate_up", (34816, 5120), W8),
        tensor_spec("mtp/layer/mlp/down", (5120, 17408), W8),
        tensor_spec("mtp/final_norm", (5120,), BF16),
    )


TEXT_CORE_TENSOR_SPECS = _build_text_core_specs()
DRAFT_HEAD_TENSOR_SPECS = _build_draft_head_specs()
MTP_TENSOR_SPECS = _build_mtp_specs()
VISION_TENSOR_SPECS = build_vision_specs(5120)

TENSOR_SPECS = (
    TEXT_CORE_TENSOR_SPECS
    + DRAFT_HEAD_TENSOR_SPECS
    + MTP_TENSOR_SPECS
    + VISION_TENSOR_SPECS
)
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS

FORMAT_COUNTS = {
    numeric_format: sum(spec.format == numeric_format for spec in TENSOR_SPECS)
    for numeric_format in FORMAT_NAMES
}
LAYOUT_COUNTS = {
    layout: sum(spec.layout == layout for spec in TENSOR_SPECS)
    for layout in LAYOUT_NAMES
}

LOGICAL_ROW_VIEW_SPECS = (
    LogicalRowViewSpec(
        "text/layers/{l}/attention/query",
        "text/layers/{l}/attention/query_key_gate_value",
        0,
        6144,
        (6144, 5120),
        FULL_ATTENTION_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/attention/key",
        "text/layers/{l}/attention/query_key_gate_value",
        6144,
        7168,
        (1024, 5120),
        FULL_ATTENTION_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/attention/output_gate",
        "text/layers/{l}/attention/query_key_gate_value",
        7168,
        13312,
        (6144, 5120),
        FULL_ATTENTION_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/attention/value",
        "text/layers/{l}/attention/query_key_gate_value",
        13312,
        14336,
        (1024, 5120),
        FULL_ATTENTION_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/query",
        "text/layers/{l}/gdn/query_key_value_z",
        0,
        2048,
        (2048, 5120),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/key",
        "text/layers/{l}/gdn/query_key_value_z",
        2048,
        4096,
        (2048, 5120),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/value",
        "text/layers/{l}/gdn/query_key_value_z",
        4096,
        10240,
        (6144, 5120),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/z",
        "text/layers/{l}/gdn/query_key_value_z",
        10240,
        16384,
        (6144, 5120),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/a_projection",
        "text/layers/{l}/gdn/a_b_projection",
        0,
        48,
        (48, 5120),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/b_projection",
        "text/layers/{l}/gdn/a_b_projection",
        48,
        96,
        (48, 5120),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/mlp/gate",
        "text/layers/{l}/mlp/gate_up",
        0,
        17408,
        (17408, 5120),
        tuple(range(64)),
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/mlp/up",
        "text/layers/{l}/mlp/gate_up",
        17408,
        34816,
        (17408, 5120),
        tuple(range(64)),
    ),
    LogicalRowViewSpec(
        "mtp/layer/attention/query",
        "mtp/layer/attention/query_key_gate_value",
        0,
        6144,
        (6144, 5120),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/attention/key",
        "mtp/layer/attention/query_key_gate_value",
        6144,
        7168,
        (1024, 5120),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/attention/output_gate",
        "mtp/layer/attention/query_key_gate_value",
        7168,
        13312,
        (6144, 5120),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/attention/value",
        "mtp/layer/attention/query_key_gate_value",
        13312,
        14336,
        (1024, 5120),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/mlp/gate",
        "mtp/layer/mlp/gate_up",
        0,
        17408,
        (17408, 5120),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/mlp/up",
        "mtp/layer/mlp/gate_up",
        17408,
        34816,
        (17408, 5120),
        None,
    ),
)

ALIAS_SPECS = (
    LogicalAliasSpec("mtp/token_embedding", ("text/token_embedding",)),
    LogicalAliasSpec("mtp/full_output_head", ("text/output_head",)),
    LogicalAliasSpec(
        "mtp/optimized_proposal_head",
        ("text/draft_head", "text/draft_head_token_ids"),
    ),
    LogicalAliasSpec(
        "text/layers/{l}/gdn/channel_major_convolution",
        ("text/layers/{l}/gdn/convolution",),
        layers=GDN_LAYERS,
        axis_order=(1, 0),
    ),
)

NVFP4_TENSOR_SPECS = tuple(
    spec for spec in TENSOR_SPECS if spec.format == NVFP4
)
FP8_TENSOR_SPECS = tuple(
    spec for spec in TENSOR_SPECS if spec.format == FP8
)
INPUT_SCALE_DIVISOR_SPECS = tuple(
    spec
    for spec in TENSOR_SPECS
    if spec.format == FP32 and spec.name.endswith("/input_scale_divisor")
)


def validate_inventory() -> None:
    names = tuple(spec.name for spec in OBJECT_SPECS)
    if len(names) != len(set(names)):
        raise ValueError("Qwen3.8 NVFP4 inventory contains duplicate names")
    if (
        len(TEXT_CORE_TENSOR_SPECS),
        len(DRAFT_HEAD_TENSOR_SPECS),
        len(MTP_TENSOR_SPECS),
        len(VISION_TENSOR_SPECS),
        len(TENSOR_SPECS),
        len(OBJECT_SPECS),
        len(NVFP4_TENSOR_SPECS),
        len(FP8_TENSOR_SPECS),
        len(INPUT_SCALE_DIVISOR_SPECS),
    ) != (771, 2, 12, 333, 1118, 1124, 112, 146, 112):
        raise ValueError("registered Qwen3.8 NVFP4 inventory is incomplete")
    if FORMAT_COUNTS != {
        BF16: 534,
        FP32: 208,
        I32: 1,
        Q4: 55,
        Q5: 54,
        Q6: 1,
        W8: 7,
        NVFP4: 112,
        FP8: 146,
    }:
        raise ValueError(f"unexpected numeric allocation: {FORMAT_COUNTS}")
    if LAYOUT_COUNTS != {
        CONTIGUOUS_LAYOUT: 743,
        ROW_SPLIT_LAYOUT: 117,
        BLOCK_SCALE_LAYOUT: 112,
        ROW_SCALE_LAYOUT: 146,
    }:
        raise ValueError(f"unexpected layout allocation: {LAYOUT_COUNTS}")


validate_inventory()


__all__ = [
    "ALIAS_SPECS",
    "BF16",
    "BLOCK_SCALE_LAYOUT",
    "CONTIGUOUS_LAYOUT",
    "DRAFT_HEAD_TENSOR_SPECS",
    "FORMAT_COUNTS",
    "FORMAT_NAMES",
    "FP32",
    "FP8",
    "FP8_MLP_LAYERS",
    "FP8_TENSOR_SPECS",
    "FULL_ATTENTION_LAYERS",
    "GDN_LAYERS",
    "I32",
    "INPUT_SCALE_DIVISOR_SPECS",
    "LAYOUT_COUNTS",
    "LAYOUT_NAMES",
    "LOGICAL_ROW_VIEW_SPECS",
    "MODEL_ID",
    "MTP_TENSOR_SPECS",
    "NVFP4",
    "NVFP4_MLP_LAYERS",
    "NVFP4_TENSOR_SPECS",
    "OBJECT_SPECS",
    "Q4",
    "Q5",
    "Q6",
    "RESOURCE_SPECS",
    "ROW_SCALE_LAYOUT",
    "ROW_SPLIT_LAYOUT",
    "ResourceSpec",
    "StoredObjectSpec",
    "TARGET_KEY",
    "TENSOR_SPECS",
    "TEXT_CORE_TENSOR_SPECS",
    "TensorSpec",
    "VISION_TENSOR_SPECS",
    "W8",
    "tensor_spec",
    "validate_inventory",
]
