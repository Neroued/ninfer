"""Leaf storage contracts shared by explicit Qwen3.6 artifact targets.

This module owns only format/layout vocabulary, passive inventory records, the
common frontend resources, and the checkpoint-invariant Vision inventory.
Each target still owns its complete ordered object inventory.
"""

from __future__ import annotations

from dataclasses import dataclass


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


def tensor_spec(
    name: str,
    shape: tuple[int, ...],
    numeric_format: str,
) -> TensorSpec:
    """Build a tensor spec with the canonical layout for its numeric format."""

    layout = CONTIGUOUS_LAYOUT if numeric_format in DIRECT_FORMATS else ROW_SPLIT_LAYOUT
    return TensorSpec(name=name, shape=shape, format=numeric_format, layout=layout)


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


def build_vision_specs(text_width: int) -> tuple[TensorSpec, ...]:
    """Build the shared Qwen3.6 Vision inventory for a target text width."""

    specs: list[TensorSpec] = [
        tensor_spec("vision/patch_embedding", (1152, 1536), Q6),
        tensor_spec("vision/patch_embedding_bias", (1152,), BF16),
        tensor_spec("vision/position_embedding", (2304, 1152), BF16),
    ]

    for layer in VISION_LAYERS:
        prefix = f"vision/layers/{layer}/"
        specs.extend(
            (
                tensor_spec(prefix + "attention/qkv", (3456, 1152), Q4),
                tensor_spec(prefix + "attention/qkv_bias", (3456,), BF16),
                tensor_spec(prefix + "attention/output", (1152, 1152), Q5),
                tensor_spec(prefix + "attention/output_bias", (1152,), BF16),
                tensor_spec(prefix + "mlp/fc1", (4304, 1152), Q4),
                tensor_spec(prefix + "mlp/fc1_bias", (4304,), BF16),
                tensor_spec(prefix + "mlp/fc2", (1152, 4304), Q5),
                tensor_spec(prefix + "mlp/fc2_bias", (1152,), BF16),
                tensor_spec(prefix + "norm1/weight", (1152,), BF16),
                tensor_spec(prefix + "norm1/bias", (1152,), BF16),
                tensor_spec(prefix + "norm2/weight", (1152,), BF16),
                tensor_spec(prefix + "norm2/bias", (1152,), BF16),
            )
        )

    specs.extend(
        (
            tensor_spec("vision/merger/fc1", (4608, 4608), W8),
            tensor_spec("vision/merger/fc1_bias", (4608,), BF16),
            tensor_spec("vision/merger/fc2", (text_width, 4608), W8),
            tensor_spec("vision/merger/fc2_bias", (text_width,), BF16),
            tensor_spec("vision/merger/norm/weight", (1152,), BF16),
            tensor_spec("vision/merger/norm/bias", (1152,), BF16),
        )
    )
    return tuple(specs)


__all__ = [
    "BF16",
    "CONTIGUOUS_LAYOUT",
    "DIRECT_FORMATS",
    "FORMAT_NAMES",
    "FP32",
    "I32",
    "LAYOUT_NAMES",
    "LogicalAliasSpec",
    "LogicalRowViewSpec",
    "Q4",
    "Q5",
    "Q6",
    "RESOURCE_ENCODING",
    "RESOURCE_SPECS",
    "ROW_SPLIT_LAYOUT",
    "ResourceSpec",
    "StoredObjectSpec",
    "TensorSpec",
    "VISION_LAYERS",
    "W8",
    "build_vision_specs",
    "tensor_spec",
]
