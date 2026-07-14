"""Structural and representative-source verification for a 27B `.ninfer` artifact."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tempfile
from typing import Sequence

import numpy as np
from safetensors import safe_open
import torch

from tools.artifact.container import Artifact, ResourceObject, TensorObject, object_alignment
from tools.artifact.layouts import (
    align_up,
    decode_direct,
    decode_row_split_codes,
    encoded_size,
    gather_row_planes,
    row_split_geometry,
)
from tools.artifact.numeric import QuantFormat, get_format
from tools.convert.common.safetensors import ShardReader

from . import draft_head, inventory, recipe


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DIRECT_PROBE_OBJECTS = (
    "text/layers/0/input_norm",
    "text/layers/0/gdn/a_log",
    "text/layers/0/gdn/convolution",
)
QUANT_PROBE_OBJECTS = (
    "text/layers/3/attention/query_key",
    "text/layers/0/gdn/value",
    "vision/patch_embedding",
    "mtp/layer/attention/query_key_gate_value",
    "text/draft_head",
)

_FP16_MIN_SUBNORMAL = 2.0**-24


class VerificationError(ValueError):
    """The artifact does not match the registered 27B target contract."""


@dataclass(frozen=True, slots=True)
class StructureSummary:
    objects: int
    tensors: int
    resources: int
    payload_bytes: int
    row_view_templates: int
    row_view_bindings: int
    alias_templates: int
    alias_bindings: int


@dataclass(frozen=True, slots=True)
class PayloadSummary:
    direct_probes: int
    quant_probes: int
    quant_rows: int
    quant_groups: int
    draft_rows: int
    resources: int
    processor_class: str
    generation_config_class: str


@dataclass(frozen=True, slots=True)
class VerificationSummary:
    structure: StructureSummary
    payload: PayloadSummary


def _contract_error(message: str) -> None:
    raise VerificationError(message)


def _object_index(objects: Sequence[ResourceObject | TensorObject]):
    return {obj.name: obj for obj in objects}


def validate_logical_bindings(
    objects: Sequence[ResourceObject | TensorObject],
) -> tuple[int, int]:
    """Validate every fixed row view and alias against bound physical objects."""

    index = _object_index(objects)
    row_bindings = 0
    for view in inventory.LOGICAL_ROW_VIEW_SPECS:
        layers: tuple[int | None, ...]
        layers = (None,) if view.layers is None else view.layers
        for layer in layers:
            parent_name = (
                view.parent_pattern
                if layer is None
                else view.parent_pattern.format(l=layer)
            )
            parent = index.get(parent_name)
            if not isinstance(parent, TensorObject):
                _contract_error(f"logical view parent is missing: {parent_name}")
            if len(parent.shape) != 2:
                _contract_error(f"logical view parent is not a matrix: {parent_name}")
            if view.row_end > parent.shape[0]:
                _contract_error(f"logical view exceeds parent rows: {view.name_pattern}")
            if view.shape != (view.row_end - view.row_begin, parent.shape[1]):
                _contract_error(f"logical view shape is inconsistent: {view.name_pattern}")
            row_bindings += 1

    alias_bindings = 0
    for alias in inventory.ALIAS_SPECS:
        layers = (None,) if alias.layers is None else alias.layers
        for layer in layers:
            names = tuple(
                pattern if layer is None else pattern.format(l=layer)
                for pattern in alias.object_patterns
            )
            bound = [index.get(name) for name in names]
            if any(obj is None for obj in bound):
                _contract_error(f"logical alias has a missing object: {alias.role_pattern}")
            if alias.axis_order is not None:
                if len(bound) != 1 or not isinstance(bound[0], TensorObject):
                    _contract_error(f"axis alias does not bind one tensor: {alias.role_pattern}")
                source_shape = bound[0].shape
                if tuple(sorted(alias.axis_order)) != tuple(range(len(source_shape))):
                    _contract_error(f"axis alias is invalid: {alias.role_pattern}")
                target_shape = tuple(source_shape[axis] for axis in alias.axis_order)
                if target_shape != (10240, 4):
                    _contract_error(f"GDN convolution alias has shape {target_shape}")
            alias_bindings += 1

    return row_bindings, alias_bindings


def validate_structure(artifact: Artifact) -> StructureSummary:
    """Validate the complete directory without reading tensor payload values."""

    if artifact.model_id != inventory.MODEL_ID:
        _contract_error(
            f"model_id is {artifact.model_id!r}, expected {inventory.MODEL_ID!r}"
        )
    if len(artifact.objects) != len(inventory.OBJECT_SPECS):
        _contract_error(
            f"artifact has {len(artifact.objects)} objects, expected "
            f"{len(inventory.OBJECT_SPECS)}"
        )

    cursor = 0
    tensor_count = 0
    resource_count = 0
    formats: Counter[str] = Counter()
    layouts: Counter[str] = Counter()
    for position, (actual, expected) in enumerate(
        zip(artifact.objects, inventory.OBJECT_SPECS)
    ):
        if actual.name != expected.name:
            _contract_error(
                f"object {position} is {actual.name!r}, expected {expected.name!r}"
            )
        expected_offset = align_up(cursor, object_alignment(actual))
        if actual.offset != expected_offset:
            _contract_error(
                f"{actual.name}: offset {actual.offset}, expected {expected_offset}"
            )

        if isinstance(expected, inventory.TensorSpec):
            if not isinstance(actual, TensorObject):
                _contract_error(f"{actual.name}: expected a tensor descriptor")
            signature = (actual.shape, actual.format, actual.layout)
            registered = (expected.shape, expected.format, expected.layout)
            if signature != registered:
                _contract_error(
                    f"{actual.name}: signature {signature} does not match {registered}"
                )
            required_bytes = encoded_size(actual.layout, actual.format, actual.shape)
            if actual.bytes != required_bytes:
                _contract_error(
                    f"{actual.name}: stores {actual.bytes} bytes, expected {required_bytes}"
                )
            tensor_count += 1
            formats[actual.format] += 1
            layouts[actual.layout] += 1
        else:
            if not isinstance(actual, ResourceObject):
                _contract_error(f"{actual.name}: expected a resource descriptor")
            if actual.encoding != expected.encoding:
                _contract_error(
                    f"{actual.name}: encoding {actual.encoding!r}, expected {expected.encoding!r}"
                )
            resource_count += 1

        cursor = actual.offset + actual.bytes

    if dict(formats) != inventory.FORMAT_COUNTS:
        _contract_error(f"numeric-format counts are {dict(formats)}")
    if dict(layouts) != inventory.LAYOUT_COUNTS:
        _contract_error(f"layout counts are {dict(layouts)}")

    payload_bytes = artifact.file_bytes - artifact.payload_offset
    if cursor != payload_bytes:
        _contract_error(f"payload ends at {cursor}, file contains {payload_bytes} bytes")

    row_bindings, alias_bindings = validate_logical_bindings(artifact.objects)
    return StructureSummary(
        objects=len(artifact.objects),
        tensors=tensor_count,
        resources=resource_count,
        payload_bytes=payload_bytes,
        row_view_templates=len(inventory.LOGICAL_ROW_VIEW_SPECS),
        row_view_bindings=row_bindings,
        alias_templates=len(inventory.ALIAS_SPECS),
        alias_bindings=alias_bindings,
    )


def _three_indices(count: int) -> tuple[int, ...]:
    return tuple(dict.fromkeys((0, count // 2, count - 1)))


def _logical_words(tensor: torch.Tensor, format_name: str) -> torch.Tensor:
    contiguous = tensor.detach().contiguous().cpu()
    if format_name == inventory.BF16:
        return contiguous.view(torch.int16)
    if format_name in (inventory.FP32, inventory.I32):
        return contiguous.view(torch.int32)
    raise TypeError(f"{format_name} is not a direct format")


def _verify_direct_probe(
    artifact: Artifact,
    source_reader: ShardReader,
    object_name: str,
) -> None:
    obj = artifact.find(object_name)
    if not isinstance(obj, TensorObject):
        _contract_error(f"{object_name} is not a tensor")
    expected = recipe.materialize_recipe(
        recipe.RECIPES_BY_NAME[object_name],
        source_reader,
    )
    stored = decode_direct(artifact.payload(obj), obj.format, obj.shape)
    expected_words = _logical_words(expected, obj.format).reshape(-1)
    stored_words = _logical_words(stored, obj.format).reshape(-1)
    indices = torch.tensor(_three_indices(stored_words.numel()), dtype=torch.long)
    if not torch.equal(
        stored_words.index_select(0, indices),
        expected_words.index_select(0, indices),
    ):
        _contract_error(f"{object_name}: representative direct words differ")


class _SourceSlices:
    """Read only selected first-axis rows from recipe source tensors."""

    def __init__(self, reader: ShardReader) -> None:
        self.model_dir = reader.model_dir
        self.weight_map = reader.weight_map

    def rows(self, source: recipe.SourceTensor, rows: Sequence[int]) -> torch.Tensor:
        shard = self.weight_map[source.name]
        with safe_open(
            str(self.model_dir / shard),
            framework="pt",
            device="cpu",
        ) as handle:
            tensor_slice = handle.get_slice(source.name)
            pieces = [tensor_slice[row : row + 1] for row in rows]
        return torch.cat(pieces, dim=0)


def _qproj_rows(
    expression: recipe.Reshape,
    rows: Sequence[int],
    sources: _SourceSlices,
) -> torch.Tensor | None:
    selected = expression.source
    if not isinstance(selected, recipe.Slice) or selected.axis != 1:
        return None
    per_head = selected.source
    if not isinstance(per_head, recipe.Reshape):
        return None
    source = per_head.source
    if not isinstance(source, recipe.SourceTensor) or len(source.shape) != 2:
        return None
    if len(per_head.shape) != 3 or per_head.shape[-1] != source.shape[-1]:
        return None
    part_rows = selected.end - selected.begin
    if expression.shape != (per_head.shape[0] * part_rows, source.shape[-1]):
        return None
    source_rows = [
        (row // part_rows) * per_head.shape[1]
        + selected.begin
        + (row % part_rows)
        for row in rows
    ]
    return sources.rows(source, source_rows)


def _materialize_rows(
    expression: recipe.Expression,
    rows: Sequence[int],
    sources: _SourceSlices,
    draft_ids: torch.Tensor,
) -> torch.Tensor:
    """Materialize selected rows from a rank-two recipe result."""

    shape = recipe.expression_shape(expression)
    if len(shape) != 2:
        raise TypeError(f"row probes require a matrix expression, got {shape}")

    if isinstance(expression, recipe.SourceTensor):
        return sources.rows(expression, rows)

    if isinstance(expression, recipe.Slice):
        if expression.axis != 0:
            raise TypeError("only leading-axis slices can be probed directly")
        return _materialize_rows(
            expression.source,
            [expression.begin + row for row in rows],
            sources,
            draft_ids,
        )

    if isinstance(expression, recipe.Reshape):
        qproj = _qproj_rows(expression, rows, sources)
        if qproj is not None:
            return qproj
        if isinstance(expression.source, recipe.SourceTensor):
            source = expression.source
            if source.shape[0] == expression.shape[0]:
                selected = sources.rows(source, rows)
                return selected.reshape(len(rows), expression.shape[1])
        raise TypeError(f"unsupported row-wise reshape probe: {expression}")

    if isinstance(expression, recipe.Concat):
        if expression.axis != 0:
            raise TypeError("row probes require leading-axis concatenation")
        boundaries: list[tuple[int, int, recipe.Expression]] = []
        begin = 0
        for part in expression.sources:
            part_rows = recipe.expression_shape(part)[0]
            boundaries.append((begin, begin + part_rows, part))
            begin += part_rows
        selected_rows = []
        for row in rows:
            for part_begin, part_end, part in boundaries:
                if part_begin <= row < part_end:
                    selected_rows.append(
                        _materialize_rows(
                            part,
                            [row - part_begin],
                            sources,
                            draft_ids,
                        )
                    )
                    break
            else:
                raise IndexError(f"row {row} is outside concatenated shape {shape}")
        return torch.cat(selected_rows, dim=0)

    if isinstance(expression, recipe.Cast):
        return _materialize_rows(
            expression.source, rows, sources, draft_ids
        ).to(torch.float32)

    if isinstance(expression, recipe.GatherRows):
        source_rows = [int(draft_ids[row]) for row in rows]
        return sources.rows(expression.source, source_rows)

    raise TypeError(f"unsupported matrix row probe: {type(expression).__name__}")


def _profile_quantize_rows(
    source_rows: torch.Tensor,
    format_name: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an independent host oracle to selected logical rows."""

    format_spec = get_format(format_name)
    if not isinstance(format_spec, QuantFormat):
        raise TypeError(f"{format_name} is not quantized")
    geometry = row_split_geometry(format_spec, source_rows.shape)
    values = source_rows.detach().cpu().float().numpy()
    if not np.isfinite(values).all():
        _contract_error("quantized source probe contains NaN or infinity")
    if geometry.k_pad != geometry.k:
        padded = np.zeros(
            (geometry.n, geometry.k_pad),
            dtype=np.float32,
        )
        padded[:, : geometry.k] = values
        values = padded
    grouped = values.reshape(
        geometry.n,
        geometry.groups_per_row,
        format_spec.group_size,
    )
    max_abs = np.max(np.abs(grouped), axis=-1)
    with np.errstate(over="ignore", invalid="ignore"):
        raw_scale = (
            max_abs.astype(np.float64) / float(format_spec.qmax)
        ).astype(np.float32)
        scales = raw_scale.astype(np.float16)
    underflow = (scales == 0) & (max_abs > 0)
    if underflow.any():
        scales = scales.copy()
        scales[underflow] = np.array(_FP16_MIN_SUBNORMAL, dtype=np.float16)
    if np.any((max_abs > 0) & (~np.isfinite(scales) | (scales <= 0))):
        _contract_error("quantized source probe has an invalid binary16 scale")
    reciprocal = np.zeros(scales.shape, dtype=np.float32)
    positive = scales > 0
    reciprocal[positive] = (
        1.0 / scales[positive].astype(np.float64)
    ).astype(np.float32)
    # Products of two binary32 values are exact in binary64. Casting explicitly
    # supplies the specified binary32 rounding before integral ties-to-even.
    normalized = (
        grouped.astype(np.float64) * reciprocal.astype(np.float64)[..., None]
    ).astype(np.float32)
    codes = np.clip(
        np.rint(normalized),
        format_spec.qmin,
        format_spec.qmax,
    ).astype(np.int8)
    return torch.from_numpy(scales).to(device), torch.from_numpy(codes).to(device)


def verify_quantized_rows(
    payload: bytes | bytearray | memoryview,
    format_name: str,
    shape: tuple[int, int],
    row_indices: Sequence[int],
    source_rows: torch.Tensor,
    device: str | torch.device = "cpu",
) -> int:
    """Compare representative stored groups with profile results for source rows."""

    geometry = row_split_geometry(format_name, shape)
    gathered = gather_row_planes(payload, geometry, row_indices)
    target = torch.device(device)
    stored_scales, stored_codes = decode_row_split_codes(
        gathered,
        format_name,
        (len(row_indices), shape[1]),
        device=target,
    )
    expected_scales, expected_codes = _profile_quantize_rows(
        source_rows,
        format_name,
        target,
    )
    group_indices = torch.tensor(
        _three_indices(geometry.groups_per_row),
        dtype=torch.long,
        device=target,
    )
    if not torch.equal(
        stored_scales.index_select(1, group_indices).view(torch.int16),
        expected_scales.index_select(1, group_indices).view(torch.int16),
    ):
        _contract_error("representative stored quantization scales differ")
    if not torch.equal(
        stored_codes.index_select(1, group_indices),
        expected_codes.index_select(1, group_indices),
    ):
        _contract_error("representative stored quantization codes differ")
    return len(row_indices) * len(group_indices)


def validate_draft_token_ids(token_ids: torch.Tensor) -> None:
    if token_ids.dtype != torch.int32 or tuple(token_ids.shape) != (draft_head.DRAFT_HEAD_N,):
        _contract_error("draft token IDs must be I32[131072]")
    if int(token_ids.min()) < 0 or int(token_ids.max()) >= draft_head.TOKENIZER_VOCAB_SIZE:
        _contract_error("draft token IDs are outside the tokenizer domain")
    if torch.unique(token_ids).numel() != draft_head.DRAFT_HEAD_N:
        _contract_error("draft token IDs are not unique")


def _load_and_validate_draft_ids(
    artifact: Artifact,
    model_dir: Path,
) -> torch.Tensor:
    obj = artifact.find(draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT)
    if not isinstance(obj, TensorObject):
        _contract_error("draft token ID object is not a tensor")
    token_ids = decode_direct(artifact.payload(obj), obj.format, obj.shape)
    validate_draft_token_ids(token_ids)

    draft_expression = recipe.RECIPES_BY_NAME[
        draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT
    ].expression
    if not isinstance(draft_expression, recipe.DraftHeadTokenIds):
        _contract_error("draft ID recipe is not the registered derivation")
    context = draft_head.compute_shortlist(
        PROJECT_ROOT / draft_expression.ranking_path,
        model_dir,
        n=draft_expression.rows,
        vocab=draft_expression.vocab_rows,
    )
    expected = draft_head.materialize_draft_head_token_ids(context)
    if not torch.equal(token_ids, expected):
        _contract_error("stored draft token IDs differ from the registered shortlist")
    return token_ids


def _verify_resources_and_frontend(
    artifact: Artifact,
    model_dir: Path,
) -> tuple[str, str]:
    payloads: dict[str, bytes] = {}
    for spec in inventory.RESOURCE_SPECS:
        obj = artifact.find(spec.name)
        if not isinstance(obj, ResourceObject):
            _contract_error(f"{spec.name} is not a resource")
        payload = bytes(artifact.payload(obj))
        filename = spec.name.removeprefix("frontend/")
        source = (model_dir / filename).read_bytes()
        if payload != source:
            _contract_error(f"frontend resource differs from source: {spec.name}")
        payloads[filename] = payload

    from transformers import AutoProcessor, GenerationConfig

    with tempfile.TemporaryDirectory(prefix="ninfer-frontend-") as temporary:
        directory = Path(temporary)
        for filename, payload in payloads.items():
            (directory / filename).write_bytes(payload)
        processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
        generation_config = GenerationConfig.from_pretrained(
            directory,
            local_files_only=True,
        )
        if getattr(processor, "tokenizer", None) is None:
            _contract_error("AutoProcessor did not construct its tokenizer")
        return type(processor).__name__, type(generation_config).__name__


def verify_payloads(
    artifact: Artifact,
    model_dir: str | Path,
    device: str | torch.device = "cpu",
) -> PayloadSummary:
    """Verify representative values without requantizing complete matrices."""

    source_dir = Path(model_dir)
    recipe.preflight_sources(source_dir)
    draft_ids = _load_and_validate_draft_ids(artifact, source_dir)
    target = torch.device(device)

    quant_groups = 0
    quant_rows = 0
    with ShardReader(source_dir) as source_reader:
        for object_name in DIRECT_PROBE_OBJECTS:
            _verify_direct_probe(artifact, source_reader, object_name)

        source_slices = _SourceSlices(source_reader)
        for object_name in QUANT_PROBE_OBJECTS:
            obj = artifact.find(object_name)
            if not isinstance(obj, TensorObject) or len(obj.shape) != 2:
                _contract_error(f"quantized probe is not a matrix: {object_name}")
            rows = _three_indices(obj.shape[0])
            source_rows = _materialize_rows(
                recipe.RECIPES_BY_NAME[object_name].expression,
                rows,
                source_slices,
                draft_ids,
            )
            quant_groups += verify_quantized_rows(
                artifact.payload(obj),
                obj.format,
                obj.shape,
                rows,
                source_rows,
                target,
            )
            quant_rows += len(rows)

    processor_class, generation_config_class = _verify_resources_and_frontend(
        artifact,
        source_dir,
    )
    return PayloadSummary(
        direct_probes=len(DIRECT_PROBE_OBJECTS),
        quant_probes=len(QUANT_PROBE_OBJECTS),
        quant_rows=quant_rows,
        quant_groups=quant_groups,
        draft_rows=draft_ids.numel(),
        resources=len(inventory.RESOURCE_SPECS),
        processor_class=processor_class,
        generation_config_class=generation_config_class,
    )


def verify_artifact(
    artifact: Artifact,
    model_dir: str | Path,
    device: str | torch.device = "cpu",
) -> VerificationSummary:
    structure = validate_structure(artifact)
    payload = verify_payloads(artifact, model_dir, device)
    return VerificationSummary(structure=structure, payload=payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a Qwen3.6-27B NInfer artifact against its source checkpoint"
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    with Artifact.open(arguments.artifact) as artifact:
        summary = verify_artifact(artifact, arguments.model, arguments.device)
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
