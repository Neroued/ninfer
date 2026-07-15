"""Typed binding of a native artifact to the Qwen3.6-35B-A3B target."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import torch

from tools.artifact import Artifact, ResourceObject, TensorObject, decode_direct


MODEL_ID = "qwen3.6-35b-a3b"
TOKENIZER_VOCAB_SIZE = 248077

CONTIGUOUS = "contiguous-le-v1"
ROW_SPLIT = "row-split-k128-v1"
RESOURCE_ENCODING = "raw-bytes-v1"

BF16 = "BF16"
FP32 = "FP32"
I32 = "I32"
Q4 = "Q4G64_F16S"
Q5 = "Q5G64_F16S"
Q6 = "Q6G64_F16S"
W8 = "W8G32_F16S"

FULL_ATTENTION_LAYERS = tuple(range(3, 40, 4))
GDN_LAYERS = tuple(
    layer for layer in range(40) if layer not in FULL_ATTENTION_LAYERS
)
Q6_ROUTED_DOWN_LAYERS = frozenset((34, 38, 39))
VISION_LAYERS = tuple(range(27))

Component = Literal["text", "draft", "mtp", "vision"]


class BindingError(ValueError):
    """The generic artifact does not implement this exact target profile."""


@dataclass(frozen=True, slots=True)
class PhysicalBlock:
    tensor_id: int
    descriptor: TensorObject
    component: Component

    @property
    def shape(self) -> tuple[int, ...]:
        return self.descriptor.shape

    @property
    def format(self) -> str:
        return self.descriptor.format

    @property
    def layout(self) -> str:
        return self.descriptor.layout

    @property
    def payload_bytes(self) -> int:
        return self.descriptor.bytes


@dataclass(frozen=True, slots=True)
class LogicalRowView:
    """A consecutive logical row interval within a rank-two parent."""

    block: PhysicalBlock
    row_begin: int
    row_count: int
    shape: tuple[int, int]

    @property
    def row_end(self) -> int:
        return self.row_begin + self.row_count


@dataclass(frozen=True, slots=True)
class AxisView:
    block: PhysicalBlock
    axes: tuple[int, ...]
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ExpertBank:
    """One equal-stride expert bank addressed only for selected expert ids."""

    block: PhysicalBlock
    experts: int
    rows_per_expert: int
    split_rows: int | None


WeightObject: TypeAlias = PhysicalBlock | LogicalRowView | AxisView
RowAddressable: TypeAlias = PhysicalBlock | LogicalRowView


@dataclass(frozen=True, slots=True)
class BoundResource:
    descriptor: ResourceObject


@dataclass(frozen=True, slots=True)
class FrontendResources:
    tokenizer_json: BoundResource
    tokenizer_config_json: BoundResource
    chat_template_jinja: BoundResource
    generation_config_json: BoundResource
    preprocessor_config_json: BoundResource
    video_preprocessor_config_json: BoundResource


@dataclass(frozen=True, slots=True)
class MoeBinding:
    router_shared_gate: PhysicalBlock
    router: LogicalRowView
    shared_gate: LogicalRowView
    routed_gate_up: ExpertBank
    routed_down: ExpertBank
    shared_gate_up: PhysicalBlock
    shared_expert_gate: LogicalRowView
    shared_up: LogicalRowView
    shared_down: PhysicalBlock


@dataclass(frozen=True, slots=True)
class FullAttentionBinding:
    query_key_gate_value: PhysicalBlock
    query: LogicalRowView
    key: LogicalRowView
    output_gate: LogicalRowView
    value: LogicalRowView
    query_norm: PhysicalBlock
    key_norm: PhysicalBlock
    output: PhysicalBlock


@dataclass(frozen=True, slots=True)
class GdnBinding:
    a_log: PhysicalBlock
    dt_bias: PhysicalBlock
    convolution_storage: PhysicalBlock
    convolution: AxisView
    a_b_projection: PhysicalBlock
    a_projection: LogicalRowView
    b_projection: LogicalRowView
    query_key_value_z: PhysicalBlock
    query: LogicalRowView
    key: LogicalRowView
    value: LogicalRowView
    z: LogicalRowView
    norm: PhysicalBlock
    output: PhysicalBlock


@dataclass(frozen=True, slots=True)
class TextLayerBinding:
    index: int
    input_norm: PhysicalBlock
    attention: FullAttentionBinding | None
    gdn: GdnBinding | None
    post_attention_norm: PhysicalBlock
    moe: MoeBinding


@dataclass(frozen=True, slots=True)
class DraftHeadBinding:
    weight: PhysicalBlock
    token_ids: PhysicalBlock


@dataclass(frozen=True, slots=True)
class TextBinding:
    token_embedding: PhysicalBlock
    layers: tuple[TextLayerBinding, ...]
    final_norm: PhysicalBlock
    output_head: PhysicalBlock
    draft_head: DraftHeadBinding


@dataclass(frozen=True, slots=True)
class MtpLayerBinding:
    input_norm: PhysicalBlock
    attention: FullAttentionBinding
    post_attention_norm: PhysicalBlock
    moe: MoeBinding


@dataclass(frozen=True, slots=True)
class MtpBinding:
    token_embedding: PhysicalBlock
    full_output_head: PhysicalBlock
    optimized_proposal_head: DraftHeadBinding
    input_projection: PhysicalBlock
    embedding_norm: PhysicalBlock
    hidden_norm: PhysicalBlock
    layer: MtpLayerBinding
    final_norm: PhysicalBlock


@dataclass(frozen=True, slots=True)
class VisionLayerBinding:
    index: int
    attention_qkv: PhysicalBlock
    attention_qkv_bias: PhysicalBlock
    attention_output: PhysicalBlock
    attention_output_bias: PhysicalBlock
    mlp_fc1: PhysicalBlock
    mlp_fc1_bias: PhysicalBlock
    mlp_fc2: PhysicalBlock
    mlp_fc2_bias: PhysicalBlock
    norm1_weight: PhysicalBlock
    norm1_bias: PhysicalBlock
    norm2_weight: PhysicalBlock
    norm2_bias: PhysicalBlock


@dataclass(frozen=True, slots=True)
class VisionMergerBinding:
    fc1: PhysicalBlock
    fc1_bias: PhysicalBlock
    fc2: PhysicalBlock
    fc2_bias: PhysicalBlock
    norm_weight: PhysicalBlock
    norm_bias: PhysicalBlock


@dataclass(frozen=True, slots=True)
class VisionBinding:
    patch_embedding: PhysicalBlock
    patch_embedding_bias: PhysicalBlock
    position_embedding: PhysicalBlock
    layers: tuple[VisionLayerBinding, ...]
    merger: VisionMergerBinding


@dataclass(frozen=True, slots=True)
class _ExpectedTensor:
    name: str
    shape: tuple[int, ...]
    format: str
    layout: str


@dataclass(frozen=True, slots=True)
class _ExpectedResource:
    name: str
    encoding: str = RESOURCE_ENCODING


_ExpectedObject: TypeAlias = _ExpectedTensor | _ExpectedResource


def _tensor(name: str, shape: tuple[int, ...], format_name: str) -> _ExpectedTensor:
    layout = CONTIGUOUS if format_name in (BF16, FP32, I32) else ROW_SPLIT
    return _ExpectedTensor(name, shape, format_name, layout)


def _moe_contract(
    prefix: str,
    gate_up_format: str,
    down_format: str,
) -> tuple[_ExpectedTensor, ...]:
    return (
        _tensor(prefix + "router_shared_gate", (257, 2048), BF16),
        _tensor(prefix + "routed_gate_up", (262144, 2048), gate_up_format),
        _tensor(prefix + "routed_down", (524288, 512), down_format),
        _tensor(prefix + "shared_gate_up", (1024, 2048), W8),
        _tensor(prefix + "shared_down", (2048, 512), W8),
    )


def _text_contract() -> tuple[_ExpectedTensor, ...]:
    tensors: list[_ExpectedTensor] = [
        _tensor("text/token_embedding", (248320, 2048), W8)
    ]
    for layer in range(40):
        prefix = f"text/layers/{layer}/"
        tensors.append(_tensor(prefix + "input_norm", (2048,), BF16))
        if layer in FULL_ATTENTION_LAYERS:
            tensors.extend(
                (
                    _tensor(
                        prefix + "attention/query_key_gate_value",
                        (9216, 2048),
                        W8,
                    ),
                    _tensor(prefix + "attention/query_norm", (256,), BF16),
                    _tensor(prefix + "attention/key_norm", (256,), BF16),
                    _tensor(prefix + "attention/output", (2048, 4096), W8),
                )
            )
        else:
            tensors.extend(
                (
                    _tensor(prefix + "gdn/a_log", (32,), FP32),
                    _tensor(prefix + "gdn/dt_bias", (32,), FP32),
                    _tensor(prefix + "gdn/convolution", (4, 8192), BF16),
                    _tensor(prefix + "gdn/a_b_projection", (64, 2048), BF16),
                    _tensor(
                        prefix + "gdn/query_key_value_z", (12288, 2048), W8
                    ),
                    _tensor(prefix + "gdn/norm", (128,), BF16),
                    _tensor(prefix + "gdn/output", (2048, 4096), W8),
                )
            )
        tensors.append(_tensor(prefix + "post_attention_norm", (2048,), BF16))
        down_format = Q6 if layer in Q6_ROUTED_DOWN_LAYERS else Q5
        tensors.extend(_moe_contract(prefix + "moe/", Q4, down_format))
    tensors.extend(
        (
            _tensor("text/final_norm", (2048,), BF16),
            _tensor("text/output_head", (248320, 2048), Q6),
        )
    )
    return tuple(tensors)


def _draft_contract() -> tuple[_ExpectedTensor, ...]:
    return (
        _tensor("text/draft_head", (131072, 2048), Q4),
        _tensor("text/draft_head_token_ids", (131072,), I32),
    )


def _mtp_contract() -> tuple[_ExpectedTensor, ...]:
    tensors = [
        _tensor("mtp/input_projection", (2048, 4096), W8),
        _tensor("mtp/embedding_norm", (2048,), BF16),
        _tensor("mtp/hidden_norm", (2048,), BF16),
        _tensor("mtp/layer/input_norm", (2048,), BF16),
        _tensor(
            "mtp/layer/attention/query_key_gate_value", (9216, 2048), W8
        ),
        _tensor("mtp/layer/attention/query_norm", (256,), BF16),
        _tensor("mtp/layer/attention/key_norm", (256,), BF16),
        _tensor("mtp/layer/attention/output", (2048, 4096), W8),
        _tensor("mtp/layer/post_attention_norm", (2048,), BF16),
    ]
    tensors.extend(_moe_contract("mtp/layer/moe/", W8, W8))
    tensors.append(_tensor("mtp/final_norm", (2048,), BF16))
    return tuple(tensors)


def _vision_contract() -> tuple[_ExpectedTensor, ...]:
    tensors: list[_ExpectedTensor] = [
        _tensor("vision/patch_embedding", (1152, 1536), Q6),
        _tensor("vision/patch_embedding_bias", (1152,), BF16),
        _tensor("vision/position_embedding", (2304, 1152), BF16),
    ]
    for layer in VISION_LAYERS:
        prefix = f"vision/layers/{layer}/"
        tensors.extend(
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
    tensors.extend(
        (
            _tensor("vision/merger/fc1", (4608, 4608), W8),
            _tensor("vision/merger/fc1_bias", (4608,), BF16),
            _tensor("vision/merger/fc2", (2048, 4608), W8),
            _tensor("vision/merger/fc2_bias", (2048,), BF16),
            _tensor("vision/merger/norm/weight", (1152,), BF16),
            _tensor("vision/merger/norm/bias", (1152,), BF16),
        )
    )
    return tuple(tensors)


_RESOURCE_CONTRACT = tuple(
    _ExpectedResource(name)
    for name in (
        "frontend/tokenizer.json",
        "frontend/tokenizer_config.json",
        "frontend/chat_template.jinja",
        "frontend/generation_config.json",
        "frontend/preprocessor_config.json",
        "frontend/video_preprocessor_config.json",
    )
)
_TEXT_CONTRACT = _text_contract()
_DRAFT_CONTRACT = _draft_contract()
_MTP_CONTRACT = _mtp_contract()
_VISION_CONTRACT = _vision_contract()
_OBJECT_CONTRACT: tuple[_ExpectedObject, ...] = (
    _RESOURCE_CONTRACT
    + _TEXT_CONTRACT
    + _DRAFT_CONTRACT
    + _MTP_CONTRACT
    + _VISION_CONTRACT
)


def _component(name: str) -> Component:
    if name.startswith("vision/"):
        return "vision"
    if name.startswith("mtp/"):
        return "mtp"
    if name in ("text/draft_head", "text/draft_head_token_ids"):
        return "draft"
    return "text"


def _validate_inventory(artifact: Artifact) -> None:
    if artifact.model_id != MODEL_ID:
        raise BindingError(
            f"artifact model_id is {artifact.model_id!r}; expected {MODEL_ID!r}"
        )
    if len(artifact.objects) != len(_OBJECT_CONTRACT):
        raise BindingError(
            f"artifact has {len(artifact.objects)} objects; "
            f"expected {len(_OBJECT_CONTRACT)}"
        )
    expected_by_name = {obj.name: obj for obj in _OBJECT_CONTRACT}
    actual_names = {obj.name for obj in artifact.objects}
    expected_names = frozenset(expected_by_name)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise BindingError(
            f"artifact object names differ; missing={missing!r}, extra={extra!r}"
        )
    for actual in artifact.objects:
        expected = expected_by_name[actual.name]
        if isinstance(expected, _ExpectedTensor):
            signature = (
                expected.name,
                expected.shape,
                expected.format,
                expected.layout,
            )
            if not isinstance(actual, TensorObject) or (
                actual.name,
                actual.shape,
                actual.format,
                actual.layout,
            ) != signature:
                raise BindingError(
                    f"object {actual.name!r} does not match tensor signature "
                    f"{signature!r}"
                )
        else:
            signature = (expected.name, expected.encoding)
            if not isinstance(actual, ResourceObject) or (
                actual.name,
                actual.encoding,
            ) != signature:
                raise BindingError(
                    f"object {actual.name!r} does not match resource signature "
                    f"{signature!r}"
                )


def _row_view(
    block: PhysicalBlock,
    row_begin: int,
    row_end: int,
    views: list[LogicalRowView],
) -> LogicalRowView:
    if len(block.shape) != 2:
        raise BindingError("logical row view parent must be rank two")
    shape = (row_end - row_begin, block.shape[1])
    if row_begin < 0 or row_begin >= row_end or row_end > block.shape[0]:
        raise BindingError("logical row view is outside its physical block")
    view = LogicalRowView(block, row_begin, row_end - row_begin, shape)
    views.append(view)
    return view


def _expert_bank(
    block: PhysicalBlock,
    rows_per_expert: int,
    split_rows: int | None,
    banks: list[ExpertBank],
) -> ExpertBank:
    if block.layout != ROW_SPLIT or block.shape[0] != 256 * rows_per_expert:
        raise BindingError("expert bank does not match its equal-stride geometry")
    bank = ExpertBank(block, 256, rows_per_expert, split_rows)
    banks.append(bank)
    return bank


def _attention_binding(
    prefix: str,
    blocks: dict[str, PhysicalBlock],
    row_views: list[LogicalRowView],
) -> FullAttentionBinding:
    parent = blocks[prefix + "query_key_gate_value"]
    return FullAttentionBinding(
        query_key_gate_value=parent,
        query=_row_view(parent, 0, 4096, row_views),
        key=_row_view(parent, 4096, 4608, row_views),
        output_gate=_row_view(parent, 4608, 8704, row_views),
        value=_row_view(parent, 8704, 9216, row_views),
        query_norm=blocks[prefix + "query_norm"],
        key_norm=blocks[prefix + "key_norm"],
        output=blocks[prefix + "output"],
    )


def _moe_binding(
    prefix: str,
    blocks: dict[str, PhysicalBlock],
    row_views: list[LogicalRowView],
    expert_banks: list[ExpertBank],
) -> MoeBinding:
    router_shared_gate = blocks[prefix + "router_shared_gate"]
    shared_gate_up = blocks[prefix + "shared_gate_up"]
    return MoeBinding(
        router_shared_gate=router_shared_gate,
        router=_row_view(router_shared_gate, 0, 256, row_views),
        shared_gate=_row_view(router_shared_gate, 256, 257, row_views),
        routed_gate_up=_expert_bank(
            blocks[prefix + "routed_gate_up"], 1024, 512, expert_banks
        ),
        routed_down=_expert_bank(
            blocks[prefix + "routed_down"], 2048, None, expert_banks
        ),
        shared_gate_up=shared_gate_up,
        shared_expert_gate=_row_view(shared_gate_up, 0, 512, row_views),
        shared_up=_row_view(shared_gate_up, 512, 1024, row_views),
        shared_down=blocks[prefix + "shared_down"],
    )


class ArtifactBinding:
    """Complete typed target binding over one open generic artifact."""

    def __init__(self, artifact: Artifact, *, owns_artifact: bool = False):
        _validate_inventory(artifact)
        self._artifact = artifact
        self._owns_artifact = owns_artifact

        resources: dict[str, BoundResource] = {}
        blocks: dict[str, PhysicalBlock] = {}
        tensors: list[PhysicalBlock] = []
        for obj in artifact.objects:
            if isinstance(obj, ResourceObject):
                resources[obj.name] = BoundResource(obj)
            else:
                block = PhysicalBlock(len(tensors), obj, _component(obj.name))
                tensors.append(block)
                blocks[obj.name] = block

        self.tensors = tuple(tensors)
        self.frontend = FrontendResources(
            resources["frontend/tokenizer.json"],
            resources["frontend/tokenizer_config.json"],
            resources["frontend/chat_template.jinja"],
            resources["frontend/generation_config.json"],
            resources["frontend/preprocessor_config.json"],
            resources["frontend/video_preprocessor_config.json"],
        )

        row_views: list[LogicalRowView] = []
        axis_views: list[AxisView] = []
        expert_banks: list[ExpertBank] = []
        layers: list[TextLayerBinding] = []
        for layer in range(40):
            prefix = f"text/layers/{layer}/"
            if layer in FULL_ATTENTION_LAYERS:
                attention = _attention_binding(
                    prefix + "attention/", blocks, row_views
                )
                gdn = None
            else:
                convolution_storage = blocks[prefix + "gdn/convolution"]
                convolution = AxisView(
                    convolution_storage, (1, 0), (8192, 4)
                )
                axis_views.append(convolution)
                a_b = blocks[prefix + "gdn/a_b_projection"]
                qkvz = blocks[prefix + "gdn/query_key_value_z"]
                gdn = GdnBinding(
                    a_log=blocks[prefix + "gdn/a_log"],
                    dt_bias=blocks[prefix + "gdn/dt_bias"],
                    convolution_storage=convolution_storage,
                    convolution=convolution,
                    a_b_projection=a_b,
                    a_projection=_row_view(a_b, 0, 32, row_views),
                    b_projection=_row_view(a_b, 32, 64, row_views),
                    query_key_value_z=qkvz,
                    query=_row_view(qkvz, 0, 2048, row_views),
                    key=_row_view(qkvz, 2048, 4096, row_views),
                    value=_row_view(qkvz, 4096, 8192, row_views),
                    z=_row_view(qkvz, 8192, 12288, row_views),
                    norm=blocks[prefix + "gdn/norm"],
                    output=blocks[prefix + "gdn/output"],
                )
                attention = None

            layers.append(
                TextLayerBinding(
                    index=layer,
                    input_norm=blocks[prefix + "input_norm"],
                    attention=attention,
                    gdn=gdn,
                    post_attention_norm=blocks[prefix + "post_attention_norm"],
                    moe=_moe_binding(
                        prefix + "moe/", blocks, row_views, expert_banks
                    ),
                )
            )

        draft_head = DraftHeadBinding(
            blocks["text/draft_head"], blocks["text/draft_head_token_ids"]
        )
        self.text = TextBinding(
            token_embedding=blocks["text/token_embedding"],
            layers=tuple(layers),
            final_norm=blocks["text/final_norm"],
            output_head=blocks["text/output_head"],
            draft_head=draft_head,
        )

        self.mtp = MtpBinding(
            token_embedding=self.text.token_embedding,
            full_output_head=self.text.output_head,
            optimized_proposal_head=draft_head,
            input_projection=blocks["mtp/input_projection"],
            embedding_norm=blocks["mtp/embedding_norm"],
            hidden_norm=blocks["mtp/hidden_norm"],
            layer=MtpLayerBinding(
                input_norm=blocks["mtp/layer/input_norm"],
                attention=_attention_binding(
                    "mtp/layer/attention/", blocks, row_views
                ),
                post_attention_norm=blocks[
                    "mtp/layer/post_attention_norm"
                ],
                moe=_moe_binding(
                    "mtp/layer/moe/", blocks, row_views, expert_banks
                ),
            ),
            final_norm=blocks["mtp/final_norm"],
        )

        vision_layers: list[VisionLayerBinding] = []
        for layer in VISION_LAYERS:
            prefix = f"vision/layers/{layer}/"
            vision_layers.append(
                VisionLayerBinding(
                    index=layer,
                    attention_qkv=blocks[prefix + "attention/qkv"],
                    attention_qkv_bias=blocks[prefix + "attention/qkv_bias"],
                    attention_output=blocks[prefix + "attention/output"],
                    attention_output_bias=blocks[
                        prefix + "attention/output_bias"
                    ],
                    mlp_fc1=blocks[prefix + "mlp/fc1"],
                    mlp_fc1_bias=blocks[prefix + "mlp/fc1_bias"],
                    mlp_fc2=blocks[prefix + "mlp/fc2"],
                    mlp_fc2_bias=blocks[prefix + "mlp/fc2_bias"],
                    norm1_weight=blocks[prefix + "norm1/weight"],
                    norm1_bias=blocks[prefix + "norm1/bias"],
                    norm2_weight=blocks[prefix + "norm2/weight"],
                    norm2_bias=blocks[prefix + "norm2/bias"],
                )
            )
        self.vision = VisionBinding(
            patch_embedding=blocks["vision/patch_embedding"],
            patch_embedding_bias=blocks["vision/patch_embedding_bias"],
            position_embedding=blocks["vision/position_embedding"],
            layers=tuple(vision_layers),
            merger=VisionMergerBinding(
                fc1=blocks["vision/merger/fc1"],
                fc1_bias=blocks["vision/merger/fc1_bias"],
                fc2=blocks["vision/merger/fc2"],
                fc2_bias=blocks["vision/merger/fc2_bias"],
                norm_weight=blocks["vision/merger/norm/weight"],
                norm_bias=blocks["vision/merger/norm/bias"],
            ),
        )
        self.row_views = tuple(row_views)
        self.axis_views = tuple(axis_views)
        self.expert_banks = tuple(expert_banks)
        self._validate_draft_ids()

    @classmethod
    def open(cls, path: str | Path) -> ArtifactBinding:
        artifact = Artifact.open(path)
        try:
            return cls(artifact, owns_artifact=True)
        except BaseException:
            artifact.close()
            raise

    @classmethod
    def bind(cls, artifact: Artifact) -> ArtifactBinding:
        return cls(artifact, owns_artifact=False)

    @property
    def model_id(self) -> str:
        return self._artifact.model_id

    def payload(self, block: PhysicalBlock) -> memoryview:
        return self._artifact.payload(block.descriptor)

    def resource_bytes(self, resource: BoundResource) -> bytes:
        return bytes(self._artifact.payload(resource.descriptor))

    def blocks_for(self, *components: Component) -> tuple[PhysicalBlock, ...]:
        wanted = frozenset(components)
        return tuple(block for block in self.tensors if block.component in wanted)

    def _validate_draft_ids(self) -> None:
        block = self.text.draft_head.token_ids
        token_ids = decode_direct(
            self.payload(block), block.format, block.shape, device="cpu"
        )
        if int(token_ids.min()) < 0 or int(token_ids.max()) >= TOKENIZER_VOCAB_SIZE:
            raise BindingError("draft token IDs are outside 0..248076")
        if torch.unique(token_ids).numel() != token_ids.numel():
            raise BindingError("draft token IDs are not unique")

    def close(self) -> None:
        if self._owns_artifact:
            self._artifact.close()

    def __enter__(self) -> ArtifactBinding:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = [
    "ArtifactBinding",
    "AxisView",
    "BindingError",
    "BoundResource",
    "DraftHeadBinding",
    "ExpertBank",
    "FrontendResources",
    "FullAttentionBinding",
    "GdnBinding",
    "LogicalRowView",
    "MoeBinding",
    "MtpBinding",
    "MtpLayerBinding",
    "PhysicalBlock",
    "RowAddressable",
    "TextBinding",
    "TextLayerBinding",
    "VisionBinding",
    "VisionLayerBinding",
    "VisionMergerBinding",
    "WeightObject",
]
