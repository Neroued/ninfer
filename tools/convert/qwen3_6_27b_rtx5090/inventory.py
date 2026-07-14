"""Persistent-object contract for the complete Qwen3.6-27B artifact.

This module contains only target storage roles. Source-checkpoint mapping and
materialization live in the sibling conversion recipe.
"""

from __future__ import annotations

from dataclasses import dataclass


MODEL_ID = "qwen3.6-27b"
TARGET_KEY = "qwen3_6_27b_rtx5090"

CONTIGUOUS_LAYOUT = "contiguous-le-v1"
ROW_SPLIT_LAYOUT = "row-split-k128-v1"
RESOURCE_ENCODING = "raw-bytes-v1"

BF16 = "BF16"
FP32 = "FP32"
I32 = "I32"
Q4 = "Q4G64_F16S"
Q5 = "Q5G64_F16S"
Q6 = "Q6G64_F16S"
W8 = "W8G32_F16S"

DIRECT_FORMATS = frozenset((BF16, FP32, I32))
FORMAT_NAMES = (BF16, FP32, I32, Q4, Q5, Q6, W8)
LAYOUT_NAMES = (CONTIGUOUS_LAYOUT, ROW_SPLIT_LAYOUT)

FULL_ATTENTION_LAYERS = tuple(range(3, 64, 4))
GDN_LAYERS = tuple(layer for layer in range(64) if layer not in FULL_ATTENTION_LAYERS)
VISION_LAYERS = tuple(range(27))


@dataclass(frozen=True, slots=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    format: str
    layout: str

    @property
    def kind(self) -> str:
        return "tensor"


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    name: str
    encoding: str = RESOURCE_ENCODING

    @property
    def kind(self) -> str:
        return "resource"


@dataclass(frozen=True, slots=True)
class LogicalRowViewSpec:
    name_pattern: str
    parent_pattern: str
    row_begin: int
    row_end: int
    shape: tuple[int, int]
    layers: tuple[int, ...] | None

    @property
    def row_count(self) -> int:
        return self.row_end - self.row_begin


@dataclass(frozen=True, slots=True)
class LogicalAliasSpec:
    role_pattern: str
    object_patterns: tuple[str, ...]
    layers: tuple[int, ...] | None = None
    axis_order: tuple[int, ...] | None = None


StoredObjectSpec = TensorSpec | ResourceSpec


def _tensor(name: str, shape: tuple[int, ...], numeric_format: str) -> TensorSpec:
    layout = CONTIGUOUS_LAYOUT if numeric_format in DIRECT_FORMATS else ROW_SPLIT_LAYOUT
    return TensorSpec(name=name, shape=shape, format=numeric_format, layout=layout)


def _build_text_core_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        _tensor("text/token_embedding", (248320, 5120), Q6),
    ]

    for layer in range(64):
        prefix = f"text/layers/{layer}/"
        specs.append(_tensor(prefix + "input_norm", (5120,), BF16))

        if layer in FULL_ATTENTION_LAYERS:
            specs.extend(
                (
                    _tensor(prefix + "attention/query_key", (7168, 5120), Q4),
                    _tensor(prefix + "attention/gate_value", (7168, 5120), Q5),
                    _tensor(prefix + "attention/query_norm", (256,), BF16),
                    _tensor(prefix + "attention/key_norm", (256,), BF16),
                    _tensor(prefix + "attention/output", (5120, 6144), Q5),
                )
            )
        else:
            specs.extend(
                (
                    _tensor(prefix + "gdn/a_log", (48,), FP32),
                    _tensor(prefix + "gdn/dt_bias", (48,), FP32),
                    _tensor(prefix + "gdn/convolution", (4, 10240), BF16),
                    _tensor(prefix + "gdn/a_projection", (48, 5120), BF16),
                    _tensor(prefix + "gdn/b_projection", (48, 5120), BF16),
                    _tensor(prefix + "gdn/query_key", (4096, 5120), Q4),
                    _tensor(prefix + "gdn/value", (6144, 5120), Q5),
                    _tensor(prefix + "gdn/norm", (128,), BF16),
                    _tensor(prefix + "gdn/z", (6144, 5120), Q5),
                    _tensor(prefix + "gdn/output", (5120, 6144), Q5),
                )
            )

        specs.extend(
            (
                _tensor(prefix + "post_attention_norm", (5120,), BF16),
                _tensor(prefix + "mlp/gate_up", (34816, 5120), Q4),
                _tensor(prefix + "mlp/down", (5120, 17408), Q5),
            )
        )

    specs.extend(
        (
            _tensor("text/final_norm", (5120,), BF16),
            _tensor("text/output_head", (248320, 5120), Q6),
        )
    )
    return tuple(specs)


def _build_draft_head_specs() -> tuple[TensorSpec, ...]:
    return (
        _tensor("text/draft_head", (131072, 5120), Q4),
        _tensor("text/draft_head_token_ids", (131072,), I32),
    )


def _build_mtp_specs() -> tuple[TensorSpec, ...]:
    return (
        _tensor("mtp/input_projection", (5120, 10240), W8),
        _tensor("mtp/embedding_norm", (5120,), BF16),
        _tensor("mtp/hidden_norm", (5120,), BF16),
        _tensor("mtp/layer/input_norm", (5120,), BF16),
        _tensor("mtp/layer/attention/query_key_gate_value", (14336, 5120), W8),
        _tensor("mtp/layer/attention/query_norm", (256,), BF16),
        _tensor("mtp/layer/attention/key_norm", (256,), BF16),
        _tensor("mtp/layer/attention/output", (5120, 6144), W8),
        _tensor("mtp/layer/post_attention_norm", (5120,), BF16),
        _tensor("mtp/layer/mlp/gate_up", (34816, 5120), W8),
        _tensor("mtp/layer/mlp/down", (5120, 17408), W8),
        _tensor("mtp/final_norm", (5120,), BF16),
    )


def _build_vision_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        _tensor("vision/patch_embedding", (1152, 1536), Q6),
        _tensor("vision/patch_embedding_bias", (1152,), BF16),
        _tensor("vision/position_embedding", (2304, 1152), BF16),
    ]

    for layer in VISION_LAYERS:
        prefix = f"vision/layers/{layer}/"
        specs.extend(
            (
                _tensor(prefix + "attention/qkv", (3456, 1152), Q4),
                _tensor(prefix + "attention/qkv_bias", (3456,), BF16),
                _tensor(prefix + "attention/output", (1152, 1152), Q5),
                _tensor(prefix + "attention/output_bias", (1152,), BF16),
                _tensor(prefix + "mlp/fc1", (4304, 1152), Q4),
                _tensor(prefix + "mlp/fc1_bias", (4304,), BF16),
                _tensor(prefix + "mlp/fc2", (1152, 4304), Q5),
                _tensor(prefix + "mlp/fc2_bias", (1152,), BF16),
                _tensor(prefix + "norm1/weight", (1152,), BF16),
                _tensor(prefix + "norm1/bias", (1152,), BF16),
                _tensor(prefix + "norm2/weight", (1152,), BF16),
                _tensor(prefix + "norm2/bias", (1152,), BF16),
            )
        )

    specs.extend(
        (
            _tensor("vision/merger/fc1", (4608, 4608), W8),
            _tensor("vision/merger/fc1_bias", (4608,), BF16),
            _tensor("vision/merger/fc2", (5120, 4608), W8),
            _tensor("vision/merger/fc2_bias", (5120,), BF16),
            _tensor("vision/merger/norm/weight", (1152,), BF16),
            _tensor("vision/merger/norm/bias", (1152,), BF16),
        )
    )
    return tuple(specs)


RESOURCE_SPECS = tuple(
    ResourceSpec(name)
    for name in (
        "frontend/tokenizer.json",
        "frontend/tokenizer_config.json",
        "frontend/chat_template.jinja",
        "frontend/generation_config.json",
        "frontend/preprocessor_config.json",
        "frontend/video_preprocessor_config.json",
    )
)

TEXT_CORE_TENSOR_SPECS = _build_text_core_specs()
DRAFT_HEAD_TENSOR_SPECS = _build_draft_head_specs()
MTP_TENSOR_SPECS = _build_mtp_specs()
VISION_TENSOR_SPECS = _build_vision_specs()

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
        "text/layers/{l}/attention/query_key",
        0,
        6144,
        (6144, 5120),
        FULL_ATTENTION_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/attention/key",
        "text/layers/{l}/attention/query_key",
        6144,
        7168,
        (1024, 5120),
        FULL_ATTENTION_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/attention/output_gate",
        "text/layers/{l}/attention/gate_value",
        0,
        6144,
        (6144, 5120),
        FULL_ATTENTION_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/attention/value",
        "text/layers/{l}/attention/gate_value",
        6144,
        7168,
        (1024, 5120),
        FULL_ATTENTION_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/query",
        "text/layers/{l}/gdn/query_key",
        0,
        2048,
        (2048, 5120),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/key",
        "text/layers/{l}/gdn/query_key",
        2048,
        4096,
        (2048, 5120),
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
